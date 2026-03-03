# MSS Prototype

Initial implementation scaffold for Min-Spill Search (MSS):
- backend protocol + mock backend
- llama-server HTTP backend adapter (`/completion`, `/tokenize`, `/detokenize`)
- scoring utilities (raw and proxy spill)
- decoder loop with uncertainty gating and panic fallback
- unit tests for scoring, backend parsing, decoder behavior

## Quickstart

Run tests:

```bash
python -m pytest
```

Probe backend capabilities:

```bash
python -m mss.cli inspect \
  --server http://127.0.0.1:8080 \
  --probe-text "hello"
```

Decode against a local server:

```bash
python -m mss.cli decode \
  --server http://127.0.0.1:8080 \
  --preset qwen35_35b \
  --mode mss \
  --prompt "The founder of Blue Origin is" \
  --max-tokens 16 \
  --verbose
```

Run JSONL eval cases:

```bash
python -m mss.cli eval \
  --server http://127.0.0.1:8080 \
  --cases cases/core.jsonl
```

Run built-in core stress suite:

```bash
python -m mss.cli eval \
  --server http://127.0.0.1:8080 \
  --suite core \
  --snapshot-date 2026-03-03 \
  --compare-greedy \
  --summary-json reports/core_summary.json \
  --results-jsonl reports/core_rows.jsonl
```

`cases.jsonl` schema (one object per line):

```json
{"name":"entity_swap","prompt":"The founder of Blue Origin is","expected_substring":"Jeff Bezos"}
```

## Llama-Server Findings (2026-03-03)

Environment:

- model: `Qwen3.5-9B-UD-Q4_K_XL.gguf`
- vocab size: `248320`
- server endpoint: `http://127.0.0.1:8083`

Stock server behavior (before fork patch):

- Native `/completion` did not return top-level dense logits even with `"logits": true`.
- `n_predict: 0` still produced one token (`tokens_predicted: 1`) on this build.
- `n_probs` can approximate full distribution when set to vocab size (`248320` here).

Fork patch behavior (current):

- `/completion` supports `return_logits: true` (alias `logits: true`) and returns top-level dense `logits`.
- `python -m mss.cli inspect --server http://127.0.0.1:8083` now reports:
  - `raw_logits_ok=true`
  - `sparse_logprobs_ok=true`
- MSS backend requests `return_logits: true` for raw-logit path and caches capability failures to avoid repeated failed probes.

Core comparison run (patched server):

- command:
  - `python -m mss.cli eval --server http://127.0.0.1:8083 --suite core --snapshot-date 2026-03-03 --compare-greedy --preset qwen35_9b --max-tokens 24 --summary-json /tmp/mss_core_summary_patched.json --results-jsonl /tmp/mss_core_rows_patched.jsonl`
- results:
  - `greedy_pass_rate=0.6667` (2/3)
  - `mss_pass_rate=0.6667` (2/3)
  - `mean_total_duration_overhead=2.0553x`
  - `mean_ttft_overhead=2.1453x`
