# spillage — Min-Spill Search

Lookahead re-ranking decoder that reduces hallucination by penalising tokens whose successor distributions are unexpectedly broad ("energy spill"). Implemented as a thin Python layer over a local `llama-server` or a native in-process GGUF backend.

## How it works

At each decoding step, MSS evaluates the top-k candidate tokens and measures the KL divergence of the successor distribution (the *spill*). Tokens that would put the model into a high-entropy state are penalised:

```
score(t) = surprisal(t) + beta * max(0, norm_spill(t) - tau)
```

The token with the lowest score is selected. When top-1 probability exceeds a confidence threshold, the lookahead is skipped (fast path).

## Quickstart

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m pytest          # 56+ tests
```

## CLI

### Generate with MSS decoding (native GGUF backend)

```bash
spillage gen \
  --model /path/to/model.gguf \
  --prompt "The founder of Blue Origin is" \
  --k 3 --beta 2.0 --tau 1.0 \
  --verbose
```

### Greedy baseline

```bash
spillage greedy \
  --model /path/to/model.gguf \
  --prompt "The chemical symbol for gold is"
```

### Probe a llama-server for supported capabilities

```bash
spillage inspect --url http://127.0.0.1:8080
```

Output (JSON):

```json
{
  "tokenize_ok": true,
  "detokenize_ok": true,
  "raw_logits_ok": true,
  "sparse_logprobs_ok": true,
  "notes": []
}
```

Returns exit code 1 if neither raw logits nor sparse logprobs are available.

### Run the eval harness

```bash
python -m evals --model /path/to/model.gguf --ngl 99
```

Or against a llama-server:

```bash
python -m evals --url http://127.0.0.1:8080
```

## Configuration

Default hyperparameters are in `configs/mss_default.toml`:

```toml
k = 3
beta = 2.0
tau = 1.0
confidence_threshold = 0.92
panic_margin = 2.0
```

## JSONL eval cases

`cases/core.jsonl` contains minimal smoke-test cases. Schema:

```json
{"name": "entity_swap", "prompt": "...", "expected_substring": "Jeff Bezos"}
```

The full eval suite (14 cases) is in `evals/cases.py`.

## Native backend

The native backend (`spillage/backend_native.py`) uses a thin C wrapper (`libspillage_llama.so`) built against `libllama.so` to get full dense logits in-process — no HTTP round-trips. See `CLAUDE_FINDINGS.md` for build instructions and eval results.

Required env var: `SPILLAGE_LLAMA_LIB=/path/to/libspillage_llama.so`

## Findings

- `CLAUDE_FINDINGS.md` — native backend, full eval results (14 cases, Qwen3.5-35B-A3B)
- `GEMINI_FINDINGS.md` — HTTP backend, calibration experiments (Qwen3.5-9B)
