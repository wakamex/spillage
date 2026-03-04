"""Report formatting and go/no-go verdict for eval results."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from .runner import RunResult


class GoNoGo(Enum):
    GO = "GO"
    NO_GO = "NO_GO"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# Thresholds from CLAUDE_PROPOSAL Section 2.
ERROR_REDUCTION_TARGET = 0.20  # ≥20% relative reduction
LATENCY_OVERHEAD_TARGET = 2.0  # ≤2x greedy P95


def print_report(results: list[RunResult]) -> GoNoGo:
    """Print a formatted results table and return the go/no-go verdict."""
    if not results:
        click.echo("No results to report.")
        return GoNoGo.INSUFFICIENT_DATA

    # Group by case.
    cases: dict[str, dict[str, RunResult]] = {}
    for r in results:
        cases.setdefault(r.case_name, {})[r.mode] = r

    # --- Results table ---
    click.echo("\n" + "=" * 80)
    click.echo("MSS EVALUATION REPORT")
    click.echo("=" * 80)

    header = f"{'Case':<25s} {'Mode':<20s} {'Pass':>5s} {'Tokens':>7s} {'ms/tok':>8s} {'Output (truncated)'}"
    click.echo(header)
    click.echo("-" * 80)

    for case_name, modes in sorted(cases.items()):
        for mode_name, r in sorted(modes.items()):
            status = "OK" if r.passed else "FAIL"
            text = r.output_text[:30].replace("\n", " ")
            click.echo(
                f"{case_name:<25s} {mode_name:<20s} {status:>5s} "
                f"{r.token_count:>7d} {r.ms_per_token:>8.2f} {text}"
            )
        click.echo()

    # --- Divergence diagnostics ---
    click.echo("-" * 80)
    click.echo("DIVERGENCE DIAGNOSTICS (first non-fast-path step)")
    click.echo("-" * 80)

    for case_name, modes in sorted(cases.items()):
        for mode_name, r in sorted(modes.items()):
            if r.divergence_candidates:
                click.echo(f"\n  {case_name} / {mode_name} (step {r.divergence_step}):")
                for c in r.divergence_candidates:
                    marker = ">>>" if c["selected"] else "   "
                    click.echo(
                        f"    {marker} {c['text']!r:12s} "
                        f"logp={c['logp']:+.3f}  "
                        f"spill={c['spill']:.3f}  "
                        f"norm={c['norm']:+.3f}  "
                        f"score={c['score']:.3f}"
                    )

    # --- Go/No-Go ---
    click.echo("\n" + "=" * 80)
    click.echo("GO / NO-GO VERDICT")
    click.echo("=" * 80)

    # Compute metrics.
    greedy_pass = sum(1 for r in results if r.mode == "greedy" and r.passed)
    greedy_total = sum(1 for r in results if r.mode == "greedy")
    greedy_rate = greedy_pass / max(greedy_total, 1)
    greedy_times = [r.ms_per_token for r in results if r.mode == "greedy" and r.token_count > 0]
    greedy_p95 = sorted(greedy_times)[int(len(greedy_times) * 0.95)] if greedy_times else 0.0

    click.echo(f"  Greedy pass rate:       {greedy_pass}/{greedy_total} ({greedy_rate:.0%})")

    # Report each non-greedy mode separately.
    mss_modes = sorted({r.mode for r in results if r.mode != "greedy"})
    best_verdict = GoNoGo.INSUFFICIENT_DATA

    for mss_mode in mss_modes:
        mss_pass = sum(1 for r in results if r.mode == mss_mode and r.passed)
        mss_total = sum(1 for r in results if r.mode == mss_mode)
        mss_rate = mss_pass / max(mss_total, 1)

        if greedy_rate > 0:
            error_reduction = (mss_rate - greedy_rate) / (1.0 - greedy_rate + 1e-8)
        else:
            error_reduction = mss_rate

        mss_times = [r.ms_per_token for r in results if r.mode == mss_mode and r.token_count > 0]
        if mss_times and greedy_p95 > 0:
            mss_p95 = sorted(mss_times)[int(len(mss_times) * 0.95)]
            latency_ratio = mss_p95 / greedy_p95
        else:
            latency_ratio = float("nan")

        click.echo(f"\n  {mss_mode}:")
        click.echo(f"    Pass rate:            {mss_pass}/{mss_total} ({mss_rate:.0%})")
        click.echo(f"    Error reduction:      {error_reduction:+.1%} (target: ≥{ERROR_REDUCTION_TARGET:.0%})")
        click.echo(f"    Latency ratio (P95):  {latency_ratio:.2f}x (target: ≤{LATENCY_OVERHEAD_TARGET:.1f}x)")

        err_ok = error_reduction >= ERROR_REDUCTION_TARGET
        lat_ok = latency_ratio <= LATENCY_OVERHEAD_TARGET or latency_ratio != latency_ratio
        click.echo(f"    [{'PASS' if err_ok else 'FAIL'}] Error reduction gate")
        click.echo(f"    [{'PASS' if lat_ok else 'FAIL'}] Latency gate")

        if mss_total >= 3:
            if err_ok and lat_ok:
                if best_verdict != GoNoGo.GO:
                    best_verdict = GoNoGo.GO
            else:
                if best_verdict == GoNoGo.INSUFFICIENT_DATA:
                    best_verdict = GoNoGo.NO_GO

    verdict = best_verdict

    if verdict == GoNoGo.INSUFFICIENT_DATA:
        click.echo(f"\n  VERDICT: {verdict.value} (need more test cases)")
    elif verdict == GoNoGo.GO:
        click.echo(f"\n  VERDICT: {verdict.value} — MSS shows improvement.")
    else:
        click.echo(f"\n  VERDICT: {verdict.value} — MSS does not meet targets. Review diagnostics.")

    click.echo("=" * 80)
    return verdict
