"""Min-Spill Search generation loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .backend import Backend, LogitResult
from .config import MSSConfig
from .scorer import (
    compute_spill_proxy,
    compute_spill_raw,
    robust_zscore,
    score_raw,
    score_thresholded,
)


@dataclass
class CandidateScore:
    token_id: int
    token_text: str
    log_prob: float
    spill_raw: float
    spill_normalized: float
    score: float


@dataclass
class TokenEvent:
    step: int
    candidates: list[CandidateScore]
    selected: int  # index into candidates
    wall_time_ms: float
    fast_path: bool = False
    panic: bool = False


@dataclass
class GenerateResult:
    text: str
    token_ids: list[int]
    events: list[TokenEvent] = field(default_factory=list)
    total_time_ms: float = 0.0


def _surprisal(prob: float) -> float:
    return -np.log(max(prob, 1e-12))


def _compute_candidate_spills(
    backend: Backend,
    context: list[int],
    current_result: LogitResult,
    candidate_ids: list[int],
) -> list[tuple[float, float]]:
    """Return (raw_spill, proxy_spill_or_raw) for each candidate.

    In raw mode both values are the same raw spill.
    In proxy mode both values are the proxy spill.
    """
    lookahead_seqs = [context + [cid] for cid in candidate_ids]
    lookahead_results = backend.get_logits_batch(lookahead_seqs)

    spills: list[tuple[float, float]] = []
    is_raw = backend.mode() == "raw"

    for i, (cid, lr) in enumerate(zip(candidate_ids, lookahead_results)):
        if is_raw:
            assert current_result.logits is not None
            cand_logit = float(current_result.logits[cid])
            raw = compute_spill_raw(cand_logit, lr.log_z)
            spills.append((raw, raw))
        else:
            assert lr.entropy is not None and lr.top1_margin is not None
            proxy = compute_spill_proxy(lr.entropy, lr.top1_margin)
            spills.append((proxy, proxy))

    return spills


def generate(
    prompt: str,
    backend: Backend,
    config: MSSConfig | None = None,
    variant: Literal["thresholded", "raw"] = "thresholded",
    on_token: Callable[[TokenEvent], None] | None = None,
) -> GenerateResult:
    """Run Min-Spill Search decoding.

    Parameters
    ----------
    prompt:
        Input text.
    backend:
        Inference backend (must satisfy Backend protocol).
    config:
        Hyperparameters.  Defaults to ``MSSConfig()``.
    variant:
        ``"thresholded"`` (primary, with τ) or ``"raw"`` (variant A ablation).
    on_token:
        Optional callback fired after each token is selected.
    """
    cfg = config or MSSConfig()
    context = backend.tokenize(prompt)
    eos = backend.eos_token_id()
    events: list[TokenEvent] = []
    generated_ids: list[int] = []
    t_start = time.monotonic()

    for step in range(cfg.max_tokens):
        t_step = time.monotonic()

        # 1. Forward pass on current context.
        result = backend.get_logits(context)

        # Compute full probability distribution if we have logits.
        if result.logits is not None:
            logits = result.logits
            if cfg.temperature > 0:
                logits = logits / cfg.temperature
            full_probs = np.exp(logits - np.logaddexp.reduce(logits))
        else:
            full_probs = None

        # 2. Extract top-k candidates.
        k = min(cfg.k, len(result.top_k_ids))
        candidate_ids = [int(result.top_k_ids[i]) for i in range(k)]
        candidate_probs = [
            float(full_probs[cid]) if full_probs is not None
            else float(result.top_k_probs[i])
            for i, cid in enumerate(candidate_ids)
        ]

        # 3. Adaptive gate: skip lookahead if model is confident.
        top1_prob = candidate_probs[0] if candidate_probs else 0.0
        if cfg.adaptive and top1_prob >= cfg.confidence_threshold:
            winner_idx = 0
            winner_id = candidate_ids[0]
            cands = [CandidateScore(
                token_id=candidate_ids[0],
                token_text=backend.detokenize([candidate_ids[0]]),
                log_prob=np.log(max(top1_prob, 1e-12)),
                spill_raw=0.0,
                spill_normalized=0.0,
                score=_surprisal(top1_prob),
            )]
            event = TokenEvent(
                step=step,
                candidates=cands,
                selected=0,
                wall_time_ms=(time.monotonic() - t_step) * 1000,
                fast_path=True,
            )
            events.append(event)
            if on_token:
                on_token(event)

            generated_ids.append(winner_id)
            context.append(winner_id)
            if winner_id == eos:
                break
            continue

        # 4. Lookahead for each candidate.
        spills = _compute_candidate_spills(
            backend, context, result, candidate_ids,
        )
        raw_spill_values = np.array([s[0] for s in spills])

        # 5. Normalise.
        if len(raw_spill_values) > 1:
            norm_spills = robust_zscore(raw_spill_values)
        else:
            norm_spills = np.zeros_like(raw_spill_values)

        # 6. Score each candidate.
        scored: list[CandidateScore] = []
        for i, cid in enumerate(candidate_ids):
            surp = _surprisal(candidate_probs[i])
            if variant == "thresholded":
                sc = score_thresholded(
                    surp, float(norm_spills[i]),
                    beta=cfg.beta, tau=cfg.tau,
                )
            else:
                sc = score_raw(
                    surp, float(raw_spill_values[i]),
                    alpha=cfg.alpha, beta=cfg.beta,
                )
            scored.append(CandidateScore(
                token_id=cid,
                token_text=backend.detokenize([cid]),
                log_prob=np.log(max(candidate_probs[i], 1e-12)),
                spill_raw=float(raw_spill_values[i]),
                spill_normalized=float(norm_spills[i]),
                score=sc,
            ))

        # 7. Panic check: all candidates above threshold.
        all_panic = all(
            ns > cfg.tau + cfg.panic_margin for ns in norm_spills
        )
        panic = False
        if all_panic and cfg.k < 5:
            # Could retry with k=5 here; for now just flag and pick best.
            panic = True

        # 8. Select winner (lowest score).
        winner_idx = min(range(len(scored)), key=lambda i: scored[i].score)
        winner_id = scored[winner_idx].token_id

        event = TokenEvent(
            step=step,
            candidates=scored,
            selected=winner_idx,
            wall_time_ms=(time.monotonic() - t_step) * 1000,
            panic=panic,
        )
        events.append(event)
        if on_token:
            on_token(event)

        generated_ids.append(winner_id)
        context.append(winner_id)
        if winner_id == eos:
            break

    total = (time.monotonic() - t_start) * 1000
    text = backend.detokenize(generated_ids)
    return GenerateResult(
        text=text,
        token_ids=generated_ids,
        events=events,
        total_time_ms=total,
    )


def generate_greedy(
    prompt: str,
    backend: Backend,
    max_tokens: int = 256,
) -> GenerateResult:
    """Standard greedy (argmax) decoding — baseline for comparison."""
    context = backend.tokenize(prompt)
    eos = backend.eos_token_id()
    generated_ids: list[int] = []
    events: list[TokenEvent] = []
    t_start = time.monotonic()

    for step in range(max_tokens):
        t_step = time.monotonic()
        result = backend.get_logits(context)
        winner_id = int(result.top_k_ids[0])

        cand = CandidateScore(
            token_id=winner_id,
            token_text=backend.detokenize([winner_id]),
            log_prob=float(np.log(max(float(result.top_k_probs[0]), 1e-12))),
            spill_raw=0.0,
            spill_normalized=0.0,
            score=0.0,
        )
        event = TokenEvent(
            step=step,
            candidates=[cand],
            selected=0,
            wall_time_ms=(time.monotonic() - t_step) * 1000,
            fast_path=True,
        )
        events.append(event)

        generated_ids.append(winner_id)
        context.append(winner_id)
        if winner_id == eos:
            break

    total = (time.monotonic() - t_start) * 1000
    text = backend.detokenize(generated_ids)
    return GenerateResult(
        text=text,
        token_ids=generated_ids,
        events=events,
        total_time_ms=total,
    )
