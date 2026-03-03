"""MSS decoding loop with adaptive gating and panic fallback."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

from .backend import Backend
from .scoring import (
    entropy_from_probs,
    proxy_spill,
    raw_spill,
    robust_zscores,
    softmax,
    surprisal_from_logits,
    surprisal_from_logprobs,
)


@dataclass(frozen=True)
class MSSConfig:
    k: int = 3
    beta: float = 2.0
    tau: float = 1.0
    lambda_proxy: float = 0.5
    max_tokens: int = 128
    uncertainty_top1_prob: float = 0.92
    uncertainty_entropy: float = 1.5
    adaptive_k: bool = True
    panic_margin: float = 2.0
    panic_retry_k: int = 5


@dataclass(frozen=True)
class CandidateScore:
    token_id: int
    surprisal: float
    spill_raw: float
    spill_norm: float
    score: float


@dataclass(frozen=True)
class StepDecision:
    token_id: int
    strategy: str
    top1_prob: float
    entropy: float
    candidates: tuple[CandidateScore, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DecodeTrace:
    emitted: tuple[int, ...]
    decisions: tuple[StepDecision, ...]
    step_durations_s: tuple[float, ...]
    total_duration_s: float
    ttft_s: float


class MinSpillDecoder:
    def __init__(self, backend: Backend, config: MSSConfig | None = None) -> None:
        self.backend = backend
        self.config = config or MSSConfig()

    def generate(self, prompt_ids: Sequence[int]) -> tuple[list[int], list[StepDecision]]:
        trace = self.generate_trace(prompt_ids)
        return list(trace.emitted), list(trace.decisions)

    def generate_trace(self, prompt_ids: Sequence[int]) -> DecodeTrace:
        context = list(prompt_ids)
        emitted: list[int] = []
        decisions: list[StepDecision] = []
        durations: list[float] = []

        start = time.perf_counter()

        for _ in range(self.config.max_tokens):
            step_start = time.perf_counter()
            decision = self.select_next_token(context)
            step_end = time.perf_counter()
            token_id = decision.token_id
            context.append(token_id)
            emitted.append(token_id)
            decisions.append(decision)
            durations.append(step_end - step_start)
            if token_id == self.backend.eos_token_id:
                break

        total = time.perf_counter() - start
        ttft = durations[0] if durations else 0.0
        return DecodeTrace(
            emitted=tuple(emitted),
            decisions=tuple(decisions),
            step_durations_s=tuple(durations),
            total_duration_s=total,
            ttft_s=ttft,
        )

    def select_next_token(self, context_ids: Sequence[int]) -> StepDecision:
        current = self.backend.next_step(context_ids, top_n=max(self.config.k, 2))
        ranked = self._rank_current_candidates(current)
        top_token, top1_prob, entropy = self._confidence(current, ranked)

        if top1_prob >= self.config.uncertainty_top1_prob and entropy <= self.config.uncertainty_entropy:
            return StepDecision(
                token_id=top_token,
                strategy="greedy_fast_path",
                top1_prob=top1_prob,
                entropy=entropy,
            )

        if self.config.adaptive_k:
            one_eval = self._evaluate_candidates(context_ids, [top_token], current)
            if one_eval and one_eval[0].spill_raw <= self.config.tau:
                return StepDecision(
                    token_id=top_token,
                    strategy="adaptive_k_top1",
                    top1_prob=top1_prob,
                    entropy=entropy,
                    candidates=tuple(one_eval),
                )

        base_candidates = [token for token, _ in ranked[: self.config.k]]
        scored = self._evaluate_candidates(context_ids, base_candidates, current)

        if scored and all(c.spill_norm > self.config.tau + self.config.panic_margin for c in scored):
            expanded = [token for token, _ in ranked[: max(self.config.k, self.config.panic_retry_k)]]
            rescored = self._evaluate_candidates(context_ids, expanded, current)
            if rescored and all(c.spill_norm > self.config.tau + self.config.panic_margin for c in rescored):
                return StepDecision(
                    token_id=top_token,
                    strategy="panic_greedy_fallback",
                    top1_prob=top1_prob,
                    entropy=entropy,
                    candidates=tuple(rescored),
                )
            if rescored:
                winner = min(rescored, key=lambda c: c.score)
                return StepDecision(
                    token_id=winner.token_id,
                    strategy="panic_retry_k",
                    top1_prob=top1_prob,
                    entropy=entropy,
                    candidates=tuple(rescored),
                )

        winner = min(scored, key=lambda c: c.score)
        return StepDecision(
            token_id=winner.token_id,
            strategy="mss_topk",
            top1_prob=top1_prob,
            entropy=entropy,
            candidates=tuple(scored),
        )

    def _rank_current_candidates(self, current) -> list[tuple[int, float]]:
        if current.logits is not None:
            return sorted(enumerate(current.logits), key=lambda kv: kv[1], reverse=True)
        if current.logprobs:
            return sorted(current.logprobs.items(), key=lambda kv: kv[1], reverse=True)
        raise ValueError("backend returned neither logits nor logprobs")

    def _confidence(self, current, ranked: list[tuple[int, float]]) -> tuple[int, float, float]:
        top_token = ranked[0][0]

        if current.logits is not None:
            probs = softmax(current.logits)
            entropy = entropy_from_probs(probs)
            return top_token, probs[top_token], entropy

        if not current.logprobs:
            return top_token, 0.0, 0.0

        probs = {i: math.exp(lp) for i, lp in current.logprobs.items()}
        total = sum(probs.values()) or 1.0
        normalized = {i: p / total for i, p in probs.items()}
        entropy = entropy_from_probs(normalized.values())
        return top_token, normalized.get(top_token, 0.0), entropy

    def _evaluate_candidates(self, context_ids: Sequence[int], token_ids: Sequence[int], current) -> list[CandidateScore]:
        candidates: list[CandidateScore] = []
        raw_spills: list[float] = []

        next_contexts = [[*context_ids, token_id] for token_id in token_ids]
        next_steps = self.backend.next_steps(next_contexts, top_n=max(2, len(token_ids)))
        if len(next_steps) != len(token_ids):
            raise ValueError("backend.next_steps returned unexpected number of results")

        for token_id, next_step in zip(token_ids, next_steps):

            if current.logits is not None and next_step.logits is not None:
                spill = raw_spill(next_step.logits, current.logits[token_id])
                surprisal = surprisal_from_logits(current.logits, token_id)
            else:
                if not next_step.logprobs:
                    raise ValueError("proxy mode requires next-step logprobs")
                spill = proxy_spill(list(next_step.logprobs.values()), self.config.lambda_proxy)
                if current.logprobs:
                    surprisal = surprisal_from_logprobs(current.logprobs, token_id)
                elif current.logits is not None:
                    surprisal = surprisal_from_logits(current.logits, token_id)
                else:
                    raise ValueError("cannot compute surprisal without logits/logprobs")

            raw_spills.append(spill)
            candidates.append(
                CandidateScore(
                    token_id=token_id,
                    surprisal=surprisal,
                    spill_raw=spill,
                    spill_norm=0.0,
                    score=0.0,
                )
            )

        norm = robust_zscores(raw_spills)
        rescored: list[CandidateScore] = []
        for item, spill_norm in zip(candidates, norm):
            penalty = max(0.0, spill_norm - self.config.tau)
            score = item.surprisal + self.config.beta * penalty
            rescored.append(
                CandidateScore(
                    token_id=item.token_id,
                    surprisal=item.surprisal,
                    spill_raw=item.spill_raw,
                    spill_norm=spill_norm,
                    score=score,
                )
            )

        return rescored
