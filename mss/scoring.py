"""Core math utilities for MSS scoring."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    peak = max(values)
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def log_softmax(values: Sequence[float]) -> list[float]:
    lse = logsumexp(values)
    return [v - lse for v in values]


def softmax(values: Sequence[float]) -> list[float]:
    return [math.exp(v) for v in log_softmax(values)]


def entropy_from_probs(probs: Iterable[float]) -> float:
    entropy = 0.0
    for p in probs:
        if p > 0.0:
            entropy -= p * math.log(p)
    return entropy


def entropy_from_logprobs(logprobs: Sequence[float]) -> float:
    probs = [math.exp(v) for v in logprobs]
    total = sum(probs)
    if total <= 0.0:
        return 0.0
    return entropy_from_probs([p / total for p in probs])


def robust_zscores(values: Sequence[float], epsilon: float = 1e-6) -> list[float]:
    if not values:
        raise ValueError("values cannot be empty")
    if len(values) == 1:
        return [0.0]

    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])

    abs_devs = sorted(abs(v - median) for v in values)
    mad_mid = len(abs_devs) // 2
    if len(abs_devs) % 2:
        mad = abs_devs[mad_mid]
    else:
        mad = 0.5 * (abs_devs[mad_mid - 1] + abs_devs[mad_mid])

    scale = mad + epsilon
    return [(v - median) / scale for v in values]


def raw_spill(next_logits: Sequence[float], candidate_logit: float) -> float:
    return logsumexp(next_logits) - candidate_logit


def top1_margin_from_logprobs(next_logprobs: Sequence[float]) -> float:
    if not next_logprobs:
        return 0.0
    probs = sorted((math.exp(lp) for lp in next_logprobs), reverse=True)
    if len(probs) == 1:
        return probs[0]
    return probs[0] - probs[1]


def proxy_spill(next_logprobs: Sequence[float], lambda_proxy: float = 0.5) -> float:
    entropy = entropy_from_logprobs(next_logprobs)
    margin = top1_margin_from_logprobs(next_logprobs)
    return entropy + lambda_proxy * (1.0 - margin)


def surprisal_from_logits(logits: Sequence[float], token_id: int) -> float:
    lps = log_softmax(logits)
    return -lps[token_id]


def surprisal_from_logprobs(logprobs: dict[int, float], token_id: int) -> float:
    if token_id not in logprobs:
        raise KeyError(f"token {token_id} missing from logprobs")
    values = list(logprobs.values())
    probs = [math.exp(v) for v in values]
    norm = sum(probs)
    if norm <= 0.0:
        return 0.0
    normalized_lp = logprobs[token_id] - math.log(norm)
    return -normalized_lp
