"""Evaluation harness primitives for MSS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .backend import Backend
from .decoder import DecodeTrace, MSSConfig, MinSpillDecoder


@dataclass(frozen=True)
class EvalCase:
    name: str
    prompt_ids: tuple[int, ...]
    expected_tokens: tuple[int, ...]


@dataclass(frozen=True)
class EvalResult:
    name: str
    emitted: tuple[int, ...]
    passed: bool
    total_duration_s: float
    ttft_s: float
    mean_step_s: float
    mss_invocation_rate: float


def run_cases(decoder: MinSpillDecoder, cases: Sequence[EvalCase]) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in cases:
        trace = decoder.generate_trace(case.prompt_ids)
        prefix = tuple(trace.emitted[: len(case.expected_tokens)])
        passed = prefix == case.expected_tokens
        results.append(
            EvalResult(
                name=case.name,
                emitted=tuple(trace.emitted),
                passed=passed,
                total_duration_s=trace.total_duration_s,
                ttft_s=trace.ttft_s,
                mean_step_s=_mean_step(trace),
                mss_invocation_rate=_mss_invocation_rate(trace),
            )
        )
    return results


@dataclass(frozen=True)
class TextEvalCase:
    name: str
    prompt: str
    expected_substring: str


@dataclass(frozen=True)
class TextEvalResult:
    name: str
    output_text: str
    passed: bool
    total_duration_s: float
    ttft_s: float
    mean_step_s: float
    mss_invocation_rate: float


@dataclass(frozen=True)
class ComparisonResult:
    name: str
    greedy_passed: bool
    mss_passed: bool
    greedy_total_duration_s: float
    mss_total_duration_s: float
    greedy_ttft_s: float
    mss_ttft_s: float
    greedy_output: str
    mss_output: str


@dataclass(frozen=True)
class EvalSummary:
    total_cases: int
    passed_cases: int
    pass_rate: float
    mean_total_duration_s: float
    p95_total_duration_s: float
    mean_ttft_s: float
    p95_ttft_s: float
    mean_mss_invocation_rate: float


@dataclass(frozen=True)
class ComparisonSummary:
    total_cases: int
    greedy_passed_cases: int
    mss_passed_cases: int
    greedy_pass_rate: float
    mss_pass_rate: float
    mean_greedy_total_duration_s: float
    mean_mss_total_duration_s: float
    mean_total_duration_overhead: float
    mean_greedy_ttft_s: float
    mean_mss_ttft_s: float
    mean_ttft_overhead: float


def run_text_cases(
    decoder: MinSpillDecoder,
    backend: Backend,
    cases: Sequence[TextEvalCase],
) -> list[TextEvalResult]:
    results: list[TextEvalResult] = []
    for case in cases:
        prompt_ids = backend.encode(case.prompt)
        trace = decoder.generate_trace(prompt_ids)
        output = backend.decode(trace.emitted)
        passed = case.expected_substring in output
        results.append(
            TextEvalResult(
                name=case.name,
                output_text=output,
                passed=passed,
                total_duration_s=trace.total_duration_s,
                ttft_s=trace.ttft_s,
                mean_step_s=_mean_step(trace),
                mss_invocation_rate=_mss_invocation_rate(trace),
            )
        )
    return results


def run_text_comparison(
    backend: Backend,
    cases: Sequence[TextEvalCase],
    mss_config: MSSConfig,
    greedy_config: MSSConfig | None = None,
) -> list[ComparisonResult]:
    greedy_cfg = greedy_config or MSSConfig(
        k=1,
        beta=0.0,
        tau=1.0,
        uncertainty_top1_prob=0.0,
        uncertainty_entropy=1e9,
        adaptive_k=False,
        max_tokens=mss_config.max_tokens,
    )

    greedy_decoder = MinSpillDecoder(backend, greedy_cfg)
    mss_decoder = MinSpillDecoder(backend, mss_config)

    greedy_rows = {row.name: row for row in run_text_cases(greedy_decoder, backend, cases)}
    mss_rows = {row.name: row for row in run_text_cases(mss_decoder, backend, cases)}

    out: list[ComparisonResult] = []
    for case in cases:
        g = greedy_rows[case.name]
        m = mss_rows[case.name]
        out.append(
            ComparisonResult(
                name=case.name,
                greedy_passed=g.passed,
                mss_passed=m.passed,
                greedy_total_duration_s=g.total_duration_s,
                mss_total_duration_s=m.total_duration_s,
                greedy_ttft_s=g.ttft_s,
                mss_ttft_s=m.ttft_s,
                greedy_output=g.output_text,
                mss_output=m.output_text,
            )
        )

    return out


def summarize_results(rows: Sequence[TextEvalResult]) -> EvalSummary:
    if not rows:
        return EvalSummary(
            total_cases=0,
            passed_cases=0,
            pass_rate=0.0,
            mean_total_duration_s=0.0,
            p95_total_duration_s=0.0,
            mean_ttft_s=0.0,
            p95_ttft_s=0.0,
            mean_mss_invocation_rate=0.0,
        )

    total = len(rows)
    passed = sum(1 for row in rows if row.passed)
    totals = [row.total_duration_s for row in rows]
    ttfts = [row.ttft_s for row in rows]
    invocations = [row.mss_invocation_rate for row in rows]
    return EvalSummary(
        total_cases=total,
        passed_cases=passed,
        pass_rate=passed / total,
        mean_total_duration_s=_mean(totals),
        p95_total_duration_s=_percentile(totals, 95.0),
        mean_ttft_s=_mean(ttfts),
        p95_ttft_s=_percentile(ttfts, 95.0),
        mean_mss_invocation_rate=_mean(invocations),
    )


def summarize_comparison(rows: Sequence[ComparisonResult]) -> ComparisonSummary:
    if not rows:
        return ComparisonSummary(
            total_cases=0,
            greedy_passed_cases=0,
            mss_passed_cases=0,
            greedy_pass_rate=0.0,
            mss_pass_rate=0.0,
            mean_greedy_total_duration_s=0.0,
            mean_mss_total_duration_s=0.0,
            mean_total_duration_overhead=0.0,
            mean_greedy_ttft_s=0.0,
            mean_mss_ttft_s=0.0,
            mean_ttft_overhead=0.0,
        )

    total = len(rows)
    greedy_passed = sum(1 for row in rows if row.greedy_passed)
    mss_passed = sum(1 for row in rows if row.mss_passed)
    greedy_totals = [row.greedy_total_duration_s for row in rows]
    mss_totals = [row.mss_total_duration_s for row in rows]
    greedy_ttfts = [row.greedy_ttft_s for row in rows]
    mss_ttfts = [row.mss_ttft_s for row in rows]
    return ComparisonSummary(
        total_cases=total,
        greedy_passed_cases=greedy_passed,
        mss_passed_cases=mss_passed,
        greedy_pass_rate=greedy_passed / total,
        mss_pass_rate=mss_passed / total,
        mean_greedy_total_duration_s=_mean(greedy_totals),
        mean_mss_total_duration_s=_mean(mss_totals),
        mean_total_duration_overhead=_safe_ratio(_mean(mss_totals), _mean(greedy_totals)),
        mean_greedy_ttft_s=_mean(greedy_ttfts),
        mean_mss_ttft_s=_mean(mss_ttfts),
        mean_ttft_overhead=_safe_ratio(_mean(mss_ttfts), _mean(greedy_ttfts)),
    )


def _mean_step(trace: DecodeTrace) -> float:
    if not trace.step_durations_s:
        return 0.0
    return sum(trace.step_durations_s) / len(trace.step_durations_s)


def _mss_invocation_rate(trace: DecodeTrace) -> float:
    if not trace.decisions:
        return 0.0
    mss_steps = sum(
        1
        for d in trace.decisions
        if d.strategy in {"mss_topk", "panic_retry_k", "adaptive_k_top1"}
    )
    return mss_steps / len(trace.decisions)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0.0:
        return 0.0
    return num / den
