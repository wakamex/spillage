from pathlib import Path

from mss.cli import _build_backend, _write_results_jsonl, build_parser


def test_parser_has_inspect_and_eval_commands() -> None:
    parser = build_parser()
    assert parser.prog == "mss"
    ns = parser.parse_args(["decode", "--prompt-ids", "1 2", "--mode", "greedy"])
    assert ns.mode == "greedy"


def test_build_backend_handles_inspect_namespace() -> None:
    parser = build_parser()
    ns = parser.parse_args(["inspect"])
    backend = _build_backend(ns)
    assert backend.config.default_top_n == 3


def test_write_results_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "rows.jsonl"
    _write_results_jsonl(str(out), [{"name": "a", "passed": True}, {"name": "b", "passed": False}])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"name": "a"' in lines[0]
