"""Evaluation runner: greedy vs MSS variants across stress tests.

Usage:
    python -m evals.runner --model /path/to/model.gguf
    python -m evals.runner --mock  # for structure validation
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

import click

from spillage.backend import Backend
from spillage.backend_mock import MockBackend
from spillage.config import MSSConfig
from spillage.sampler import GenerateResult, generate, generate_delta_e_gated, generate_greedy, generate_seq_gated

from .cases import TestCase, _extract_answer, get_cases, load_simple_qa, load_capitals
from .report import print_report, GoNoGo

VALID_MODES = ["greedy", "mss-thresholded", "mss-raw", "mss-gated", "mss-gated-abs", "mss-seq-gated"]


@contextmanager
def _suppress_stderr():
    """Suppress C-level stderr (ggml CUDA graph warmup spam)."""
    stderr_fd = sys.stderr.fileno()
    saved = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)


@dataclass
class RunResult:
    case_name: str
    mode: str  # "greedy", "mss-thresholded", "mss-raw"
    output_text: str
    passed: bool
    token_count: int
    total_time_ms: float
    ms_per_token: float
    # Diagnostics from the first non-fast-path event (if any).
    divergence_step: int | None = None
    divergence_candidates: list[dict] | None = None
    # Per-token ∆E (spilled energy) for detection analysis.
    delta_e_per_token: list[float] | None = None
    # Per-token log probability for confidence baseline comparison.
    log_prob_per_token: list[float] | None = None
    # Whether sequence was retried (mss-seq-gated mode).
    retried: bool = False


def _load_baseline(path: str) -> dict[str, dict[str, bool]]:
    """Load a previous JSON results file → {case_name: {mode: passed}}."""
    with open(path) as f:
        data = json.load(f)
    baseline: dict[str, dict[str, bool]] = {}
    for r in data.get("results", []):
        baseline.setdefault(r["case"], {})[r["mode"]] = r["passed"]
    return baseline


def _run_single(
    case: TestCase,
    mode: str,
    backend: Backend,
    cfg: MSSConfig,
    max_tokens: int,
) -> RunResult:
    t0 = time.monotonic()

    with _suppress_stderr():
        if mode == "greedy":
            result = generate_greedy(case.prompt, backend, max_tokens=max_tokens)
        elif mode == "mss-thresholded":
            result = generate(case.prompt, backend, config=cfg, variant="thresholded")
        elif mode == "mss-raw":
            result = generate(case.prompt, backend, config=cfg, variant="raw")
        elif mode == "mss-gated":
            result = generate_delta_e_gated(case.prompt, backend, config=cfg, scoring="lowest")
        elif mode == "mss-gated-abs":
            result = generate_delta_e_gated(case.prompt, backend, config=cfg, scoring="abs")
        elif mode == "mss-seq-gated":
            result = generate_seq_gated(case.prompt, backend, config=cfg)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    elapsed = (time.monotonic() - t0) * 1000
    n_tokens = len(result.token_ids)
    passed = case.check(result.text)

    # Find first non-fast-path event for diagnostics.
    div_step = None
    div_cands = None
    for ev in result.events:
        if not ev.fast_path:
            div_step = ev.step
            div_cands = [
                {
                    "text": c.token_text,
                    "logp": round(c.log_prob, 3),
                    "spill": round(c.spill_raw, 3),
                    "norm": round(c.spill_normalized, 3),
                    "score": round(c.score, 3),
                    "selected": i == ev.selected,
                }
                for i, c in enumerate(ev.candidates)
            ]
            break

    # Collect per-token ∆E and log_prob for post-hoc detection analysis.
    delta_es = [round(ev.candidates[ev.selected].spill_raw, 4) for ev in result.events]
    log_probs = [round(ev.candidates[ev.selected].log_prob, 4) for ev in result.events]

    return RunResult(
        case_name=case.name,
        mode=mode,
        output_text=result.text,
        passed=passed,
        token_count=n_tokens,
        total_time_ms=elapsed,
        ms_per_token=elapsed / max(n_tokens, 1),
        divergence_step=div_step,
        divergence_candidates=div_cands,
        delta_e_per_token=delta_es,
        log_prob_per_token=log_probs,
        retried=result.retried,
    )


def run_eval(
    backend: Backend,
    modes: list[str] | None = None,
    category: str | None = None,
    cases: list[TestCase] | None = None,
    max_tokens: int = 64,
    cfg: MSSConfig | None = None,
    baseline: dict[str, dict[str, bool]] | None = None,
) -> list[RunResult]:
    """Run test cases across all modes."""
    if modes is None:
        modes = ["greedy", "mss-thresholded", "mss-raw"]
    if cfg is None:
        cfg = MSSConfig(max_tokens=max_tokens, adaptive=False)
    else:
        cfg = MSSConfig(
            k=cfg.k, beta=cfg.beta, tau=cfg.tau, alpha=cfg.alpha,
            lambda_proxy=cfg.lambda_proxy,
            confidence_threshold=cfg.confidence_threshold,
            panic_margin=cfg.panic_margin,
            max_tokens=max_tokens,
            temperature=cfg.temperature,
            adaptive=False,
            delta_e_threshold=cfg.delta_e_threshold,
        )

    if cases is None:
        cases = get_cases(category)
    results: list[RunResult] = []

    for i, case in enumerate(cases, 1):
        mode_results = []
        for mode in modes:
            result = _run_single(case, mode, backend, cfg, max_tokens)
            results.append(result)
            mode_results.append(result)

        parts = []
        for r in mode_results:
            mark = "✓" if r.passed else "✗"
            retry_tag = "®" if r.retried else ""
            extracted = _extract_answer(r.output_text) or r.output_text
            extracted = extracted.replace("\n", " ")
            # Show baseline delta if available.
            if baseline and case.name in baseline and r.mode in baseline[case.name]:
                was = baseline[case.name][r.mode]
                if was != r.passed:
                    delta = "↑" if r.passed else "↓"
                else:
                    delta = "="
                parts.append(f"{r.mode}:{mark}{delta}{retry_tag} {extracted!r}")
            else:
                parts.append(f"{r.mode}:{mark}{retry_tag} {extracted!r}")
        print(f"[{i:3d}/{len(cases)}] {case.name}\n  {'  '.join(parts)}", flush=True)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--model", type=click.Path(exists=True), envvar="SPILLAGE_MODEL", default=None)
@click.option("--mock", is_flag=True, help="Use MockBackend (structure validation only).")
@click.option("--suite", type=click.Choice(["builtin", "simpleqa", "capitals"]), default="builtin", show_default=True)
@click.option("--simpleqa-n", default=100, show_default=True)
@click.option("--simpleqa-seed", default=42, show_default=True)
@click.option("--category", type=click.Choice(["factual", "math", "pattern"]), default=None)
@click.option("--modes", default="greedy,mss-thresholded,mss-raw", show_default=True,
              help="Comma-separated list of modes to run.")
@click.option("--no-think", "no_think", is_flag=True,
              help="Prepend /no_think to prompts (disables thinking on Qwen3/3.5).")
@click.option("--max-tokens", default=None, type=int,
              help="Max tokens (default: 64 builtin, 128 simpleqa).")
@click.option("--k", default=3, show_default=True)
@click.option("--beta", default=2.0, show_default=True)
@click.option("--tau", default=1.0, show_default=True)
@click.option("--delta-e-threshold", "delta_e_threshold", default=-4.5, show_default=True,
              help="Sequence-level ∆E threshold for mss-seq-gated mode.")
@click.option("--ngl", default=99, show_default=True, help="GPU layers to offload.")
@click.option("--json-out", type=click.Path(), default=None, help="Save results as JSON.")
@click.option("--baseline", "baseline_path", type=click.Path(exists=True), default=None,
              help="Previous JSON results to compare against (shows ↑↓= deltas).")
def main(
    model: str | None,
    mock: bool,
    suite: str,
    simpleqa_n: int,
    simpleqa_seed: int,
    category: str | None,
    modes: str,
    no_think: bool,
    max_tokens: int | None,
    k: int,
    beta: float,
    tau: float,
    delta_e_threshold: float,
    ngl: int,
    json_out: str | None,
    baseline_path: str | None,
) -> None:
    """Run MSS evaluation harness."""
    if mock:
        backend: Backend = MockBackend(vocab_size=16)
    elif model:
        from spillage.backend_native import NativeBackend
        with _suppress_stderr():
            backend = NativeBackend(model_path=model, n_gpu_layers=ngl, verbose=False)
    else:
        click.echo("Error: provide --model or --mock.", err=True)
        raise SystemExit(1)

    mode_list = [m.strip() for m in modes.split(",")]
    invalid = [m for m in mode_list if m not in VALID_MODES]
    if invalid:
        click.echo(f"Unknown modes: {invalid}. Valid: {VALID_MODES}", err=True)
        raise SystemExit(1)

    if max_tokens is None:
        max_tokens = 128 if suite == "simpleqa" else 32

    if suite == "simpleqa":
        cases = load_simple_qa(n=simpleqa_n, seed=simpleqa_seed)
        click.echo(f"Loaded {len(cases)} SimpleQA cases (seed={simpleqa_seed}).", err=True)
    elif suite == "capitals":
        cases = load_capitals()
        click.echo(f"Loaded {len(cases)} capitals cases.", err=True)
    else:
        cases = get_cases(category)

    if no_think:
        from dataclasses import replace
        cases = [replace(c, prompt=f"/no_think {c.prompt}") for c in cases]
        click.echo("Thinking disabled (/no_think prefix added).", err=True)

    baseline = _load_baseline(baseline_path) if baseline_path else None

    cfg = MSSConfig(k=k, beta=beta, tau=tau, max_tokens=max_tokens, delta_e_threshold=delta_e_threshold)
    results = run_eval(backend, modes=mode_list, cases=cases,
                       max_tokens=max_tokens, cfg=cfg, baseline=baseline)

    verdict = print_report(results)

    if json_out:
        import os
        data = {
            "model": os.path.basename(model) if model else "mock",
            "suite": suite,
            "modes": mode_list,
            "no_think": no_think,
            "max_tokens": max_tokens,
            "results": [
                {
                    "case": r.case_name,
                    "mode": r.mode,
                    "output": r.output_text,
                    "passed": r.passed,
                    "tokens": r.token_count,
                    "ms_total": round(r.total_time_ms, 1),
                    "ms_per_token": round(r.ms_per_token, 2),
                    "divergence_step": r.divergence_step,
                    "divergence_candidates": r.divergence_candidates,
                    "delta_e": r.delta_e_per_token,
                    "log_prob": r.log_prob_per_token,
                    "retried": r.retried,
                }
                for r in results
            ],
            "verdict": verdict.name,
        }
        with open(json_out, "w") as f:
            json.dump(data, f, indent=2)
        click.echo(f"\nJSON results written to {json_out}", err=True)


if __name__ == "__main__":
    main()
