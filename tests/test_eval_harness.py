from mss.backend import MockBackend
from mss.decoder import MSSConfig, MinSpillDecoder
from mss.eval_harness import (
    EvalCase,
    TextEvalCase,
    run_cases,
    run_text_cases,
    run_text_comparison,
    summarize_comparison,
    summarize_results,
)


def test_run_cases_returns_metrics() -> None:
    backend = MockBackend(
        {
            tuple(): [5.0, 1.0, -1.0],
            (0,): [4.0, 1.0, -1.0],
            (0, 0): [0.0, 0.0, 6.0],
        },
        eos_token_id=2,
    )
    decoder = MinSpillDecoder(backend, MSSConfig(max_tokens=3))
    cases = [EvalCase(name="basic", prompt_ids=tuple(), expected_tokens=(0, 0, 2))]

    results = run_cases(decoder, cases)
    assert len(results) == 1
    row = results[0]
    assert row.name == "basic"
    assert row.passed is True
    assert row.total_duration_s >= 0.0
    assert row.ttft_s >= 0.0
    assert row.mean_step_s >= 0.0
    assert 0.0 <= row.mss_invocation_rate <= 1.0


def test_run_text_comparison_emits_two_modes() -> None:
    backend = MockBackend(
        {
            tuple(): [3.0, 2.9, -4.0],
            (0,): [4.0, 4.0, 4.0],
            (1,): [0.0, -1.0, -2.0],
            (2,): [0.0, 0.0, 0.0],
        },
        eos_token_id=2,
    )
    cases = [TextEvalCase(name="cmp", prompt="", expected_substring="1")]
    cfg = MSSConfig(k=2, beta=2.0, tau=0.0, adaptive_k=False, max_tokens=1)

    rows = run_text_comparison(backend, cases, cfg)
    assert len(rows) == 1
    row = rows[0]
    assert row.name == "cmp"
    assert row.greedy_output in {"0", "1"}
    assert row.mss_output == "1"


def test_summaries_compute_rates_and_overheads() -> None:
    backend = MockBackend(
        {
            tuple(): [3.0, 2.0, -1.0],
            (0,): [2.0, 1.0, -1.0],
            (0, 0): [0.0, 0.0, 4.0],
            (1,): [0.0, -1.0, -2.0],
            (2,): [0.0, 0.0, 0.0],
        },
        eos_token_id=2,
    )
    cases = [TextEvalCase(name="s1", prompt="", expected_substring="0")]
    mss_cfg = MSSConfig(k=2, beta=1.5, tau=0.0, adaptive_k=False, max_tokens=1)
    decoder = MinSpillDecoder(backend, mss_cfg)

    single_rows = run_text_cases(decoder, backend, cases)
    single_summary = summarize_results(single_rows)
    assert single_summary.total_cases == 1
    assert 0.0 <= single_summary.pass_rate <= 1.0
    assert single_summary.p95_total_duration_s >= 0.0

    cmp_rows = run_text_comparison(backend, cases, mss_cfg)
    cmp_summary = summarize_comparison(cmp_rows)
    assert cmp_summary.total_cases == 1
    assert 0.0 <= cmp_summary.greedy_pass_rate <= 1.0
    assert 0.0 <= cmp_summary.mss_pass_rate <= 1.0
    assert cmp_summary.mean_total_duration_overhead >= 0.0
