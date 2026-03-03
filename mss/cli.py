"""CLI for MSS decoder prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .config import load_mss_config
from .decoder import MinSpillDecoder
from .eval_harness import (
    TextEvalCase,
    run_text_cases,
    run_text_comparison,
    summarize_comparison,
    summarize_results,
)
from .llama_server_backend import LlamaServerBackend, LlamaServerConfig
from .presets import PRESETS, get_preset
from .stress_suites import stress_suite_core


def _parse_prompt_ids(text: str) -> list[int]:
    cleaned = text.replace(",", " ").strip()
    if not cleaned:
        return []
    return [int(part) for part in cleaned.split()]


def _build_backend(args: argparse.Namespace) -> LlamaServerBackend:
    default_top_n = max(getattr(args, "k", 3), 2)
    cfg = LlamaServerConfig(
        base_url=args.server,
        eos_token_id=args.eos_token_id,
        timeout_s=args.timeout,
        prefer_raw_logits=not args.no_raw_logits,
        default_top_n=default_top_n,
        api_mode=args.api_mode,
        model=args.model,
    )
    return LlamaServerBackend(cfg)


def _load_text_cases(path: str) -> list[TextEvalCase]:
    out: list[TextEvalCase] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_no}: expected object")
        name = str(row.get("name", f"case_{line_no}"))
        prompt = row.get("prompt")
        expected_substring = row.get("expected_substring")
        if not isinstance(prompt, str) or not isinstance(expected_substring, str):
            raise ValueError(f"line {line_no}: `prompt` and `expected_substring` must be strings")
        out.append(TextEvalCase(name=name, prompt=prompt, expected_substring=expected_substring))
    return out


def _add_common_decode_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to TOML MSS config")
    parser.add_argument(
        "--mode",
        choices=("mss", "greedy"),
        default="mss",
        help="Decoding mode",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(sorted(PRESETS.keys())),
        default="default",
        help="Model preset for default k/beta/tau",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--uncertainty-top1-prob", type=float, default=0.92)
    parser.add_argument("--uncertainty-entropy", type=float, default=1.5)
    parser.add_argument("--lambda-proxy", type=float, default=0.5)
    parser.add_argument("--no-adaptive-k", action="store_true")
    parser.add_argument("--panic-margin", type=float, default=2.0)
    parser.add_argument("--panic-retry-k", type=int, default=5)


def _add_backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--eos-token-id", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--api-mode", choices=("native", "oai"), default="native")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-raw-logits", action="store_true")


def _decode_cmd(args: argparse.Namespace) -> int:
    backend = _build_backend(args)
    preset_overrides = get_preset(args.preset)
    explicit_overrides = {
        "k": args.k,
        "beta": args.beta,
        "tau": args.tau,
        "max_tokens": args.max_tokens,
        "uncertainty_top1_prob": args.uncertainty_top1_prob,
        "uncertainty_entropy": args.uncertainty_entropy,
        "lambda_proxy": args.lambda_proxy,
        "adaptive_k": not args.no_adaptive_k,
        "panic_margin": args.panic_margin,
        "panic_retry_k": args.panic_retry_k,
    }
    overrides = {**preset_overrides, **explicit_overrides}
    if args.mode == "greedy":
        overrides.update(
            {
                "k": 1,
                "beta": 0.0,
                "adaptive_k": False,
                "uncertainty_top1_prob": 0.0,
                "uncertainty_entropy": 1e9,
            }
        )
    decoder = MinSpillDecoder(backend, load_mss_config(args.config, overrides))

    if args.prompt_ids:
        prompt_ids = _parse_prompt_ids(args.prompt_ids)
    else:
        prompt_ids = backend.encode(args.prompt)

    trace = decoder.generate_trace(prompt_ids)

    print(f"tokens: {' '.join(str(t) for t in trace.emitted)}")
    if not args.no_decode_text:
        try:
            print("text:")
            print(backend.decode(trace.emitted))
        except Exception as exc:  # noqa: BLE001
            print(f"text decode failed: {exc}")

    print(f"steps: {len(trace.decisions)}")
    print(f"ttft_s: {trace.ttft_s:.6f}")
    print(f"total_s: {trace.total_duration_s:.6f}")

    if args.verbose:
        for idx, decision in enumerate(trace.decisions):
            print(
                f"step={idx} token={decision.token_id} strategy={decision.strategy} "
                f"top1_prob={decision.top1_prob:.4f} entropy={decision.entropy:.4f}"
            )
            for cand in decision.candidates:
                print(
                    f"  cand={cand.token_id} surprisal={cand.surprisal:.4f} "
                    f"spill_raw={cand.spill_raw:.4f} spill_norm={cand.spill_norm:.4f} "
                    f"score={cand.score:.4f}"
                )

    return 0


def _eval_cmd(args: argparse.Namespace) -> int:
    backend = _build_backend(args)
    preset_overrides = get_preset(args.preset)
    explicit_overrides = {
        "k": args.k,
        "beta": args.beta,
        "tau": args.tau,
        "max_tokens": args.max_tokens,
        "uncertainty_top1_prob": args.uncertainty_top1_prob,
        "uncertainty_entropy": args.uncertainty_entropy,
        "lambda_proxy": args.lambda_proxy,
        "adaptive_k": not args.no_adaptive_k,
        "panic_margin": args.panic_margin,
        "panic_retry_k": args.panic_retry_k,
    }
    overrides = {**preset_overrides, **explicit_overrides}
    if args.mode == "greedy":
        overrides.update(
            {
                "k": 1,
                "beta": 0.0,
                "adaptive_k": False,
                "uncertainty_top1_prob": 0.0,
                "uncertainty_entropy": 1e9,
            }
        )
    decoder = MinSpillDecoder(backend, load_mss_config(args.config, overrides))

    if args.cases:
        cases = _load_text_cases(args.cases)
    elif args.suite == "core":
        cases = stress_suite_core(snapshot_date=args.snapshot_date)
    else:
        raise ValueError(f"unknown suite: {args.suite}")

    if args.compare_greedy:
        rows = run_text_comparison(backend, cases, decoder.config)
        dumped: list[dict[str, object]] = []
        for row in rows:
            payload = {
                "name": row.name,
                "greedy_passed": row.greedy_passed,
                "mss_passed": row.mss_passed,
                "greedy_total_duration_s": row.greedy_total_duration_s,
                "mss_total_duration_s": row.mss_total_duration_s,
                "greedy_ttft_s": row.greedy_ttft_s,
                "mss_ttft_s": row.mss_ttft_s,
                "greedy_output": row.greedy_output,
                "mss_output": row.mss_output,
            }
            dumped.append(payload)
            print(json.dumps(payload, ensure_ascii=True))

        summary = summarize_comparison(rows)
        summary_payload = {
            "summary_type": "comparison",
            "total_cases": summary.total_cases,
            "greedy_passed_cases": summary.greedy_passed_cases,
            "mss_passed_cases": summary.mss_passed_cases,
            "greedy_pass_rate": summary.greedy_pass_rate,
            "mss_pass_rate": summary.mss_pass_rate,
            "mean_greedy_total_duration_s": summary.mean_greedy_total_duration_s,
            "mean_mss_total_duration_s": summary.mean_mss_total_duration_s,
            "mean_total_duration_overhead": summary.mean_total_duration_overhead,
            "mean_greedy_ttft_s": summary.mean_greedy_ttft_s,
            "mean_mss_ttft_s": summary.mean_mss_ttft_s,
            "mean_ttft_overhead": summary.mean_ttft_overhead,
        }
        print(json.dumps(summary_payload, ensure_ascii=True))
        if args.results_jsonl:
            _write_results_jsonl(args.results_jsonl, dumped)
        if args.summary_json:
            Path(args.summary_json).write_text(
                json.dumps(summary_payload, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
        return 0 if summary.mss_passed_cases == summary.total_cases else 1

    results = run_text_cases(decoder, backend, cases)
    dumped: list[dict[str, object]] = []
    for row in results:
        payload = {
            "name": row.name,
            "passed": row.passed,
            "total_duration_s": row.total_duration_s,
            "ttft_s": row.ttft_s,
            "mean_step_s": row.mean_step_s,
            "mss_invocation_rate": row.mss_invocation_rate,
            "output_text": row.output_text,
        }
        dumped.append(payload)
        print(json.dumps(payload, ensure_ascii=True))

    summary = summarize_results(results)
    summary_payload = {
        "summary_type": "single_mode",
        "total_cases": summary.total_cases,
        "passed_cases": summary.passed_cases,
        "pass_rate": summary.pass_rate,
        "mean_total_duration_s": summary.mean_total_duration_s,
        "p95_total_duration_s": summary.p95_total_duration_s,
        "mean_ttft_s": summary.mean_ttft_s,
        "p95_ttft_s": summary.p95_ttft_s,
        "mean_mss_invocation_rate": summary.mean_mss_invocation_rate,
    }
    print(json.dumps(summary_payload, ensure_ascii=True))
    if args.results_jsonl:
        _write_results_jsonl(args.results_jsonl, dumped)
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary_payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if summary.passed_cases == summary.total_cases else 1


def _inspect_cmd(args: argparse.Namespace) -> int:
    backend = _build_backend(args)
    caps = backend.probe_capabilities(probe_text=args.probe_text, top_n=args.top_n)
    payload = {
        "tokenize_ok": caps.tokenize_ok,
        "detokenize_ok": caps.detokenize_ok,
        "raw_logits_ok": caps.raw_logits_ok,
        "sparse_logprobs_ok": caps.sparse_logprobs_ok,
        "notes": list(caps.notes),
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if (caps.raw_logits_ok or caps.sparse_logprobs_ok) else 1


def _write_results_jsonl(path: str, rows: list[dict[str, object]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
    Path(path).write_text(payload, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mss")
    sub = parser.add_subparsers(dest="cmd", required=True)

    decode_cmd = sub.add_parser("decode", help="Generate with MSS")
    _add_backend_options(decode_cmd)
    _add_common_decode_options(decode_cmd)
    decode_cmd.add_argument("--prompt", default="")
    decode_cmd.add_argument("--prompt-ids", default="")
    decode_cmd.add_argument("--no-decode-text", action="store_true")
    decode_cmd.add_argument("--verbose", action="store_true")
    decode_cmd.set_defaults(func=_decode_cmd)

    eval_cmd = sub.add_parser("eval", help="Run JSONL text eval cases")
    _add_backend_options(eval_cmd)
    _add_common_decode_options(eval_cmd)
    eval_cmd.add_argument("--cases", help="Path to JSONL cases")
    eval_cmd.add_argument(
        "--suite",
        choices=("core",),
        default="core",
        help="Built-in suite to run when --cases is omitted",
    )
    eval_cmd.add_argument(
        "--snapshot-date",
        default="2026-03-03",
        help="Date stamp injected into built-in time-sensitive prompts",
    )
    eval_cmd.add_argument(
        "--compare-greedy",
        action="store_true",
        help="Also run greedy baseline and emit side-by-side results",
    )
    eval_cmd.add_argument(
        "--summary-json",
        default="",
        help="Optional path to write aggregate summary JSON",
    )
    eval_cmd.add_argument(
        "--results-jsonl",
        default="",
        help="Optional path to write per-case JSONL results",
    )
    eval_cmd.set_defaults(func=_eval_cmd)

    inspect_cmd = sub.add_parser("inspect", help="Probe backend capabilities")
    _add_backend_options(inspect_cmd)
    inspect_cmd.add_argument("--probe-text", default="hello")
    inspect_cmd.add_argument("--top-n", type=int, default=5)
    inspect_cmd.set_defaults(func=_inspect_cmd)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "decode" and not args.prompt and not args.prompt_ids:
        parser.error("decode requires --prompt or --prompt-ids")

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
