# Min-Spill Search (MSS) — Implementation Proposal (v3)

## 0. Comparative Ratings

| Document | Score | Key Strength | Key Weakness |
|----------|-------|-------------|--------------|
| motivation.md Variant A (lines 1–93) | **48** | Clear intuition, usable pseudocode | No threshold, formula inconsistency, no architecture |
| motivation.md Variant B (lines 95–158) | **61** | Threshold τ, model-specific presets | logprobs≠logits confusion, no code |
| GEMINI_PROPOSAL.md | **72** | Best KV-cache branching design, adaptive gating | Thin on details, no acceptance criteria, doesn't question fundamentals |
| CODEX_PROPOSAL.md | **79** | Proxy spill fallback, robust z-score normalization, quantified go/no-go | Doesn't catch formula inconsistency, doesn't question theoretical basis |
| CLAUDE_PROPOSAL.md v2 | **78** | Catches formula bug, questions core hypothesis, multi-token analysis | Missing normalization, no adaptive gating, no proxy mode |

This v3 integrates the best ideas from all proposals.

---

## 1. Critical Review of motivation.md

The document proposes a one-step lookahead re-ranking strategy: instead of
greedy argmax, evaluate the top-k candidates by checking which one produces the
most "stable" successor state. Two scoring variants are given across two
overlapping (and sometimes contradictory) sections:

| Variant | Formula | Notes |
|---------|---------|-------|
| **A** (Section 1) | `S = α·surprisal + β·spill` | Raw spill, no threshold. Minimize. |
| **B** (Section 4, "refined") | `S = logprob − β·max(0, ΔE − τ)` | Thresholded. Maximize. |

Variant B is strictly better for production — the threshold τ avoids penalising
natural energy fluctuations. Variant A is useful as an ablation baseline. We
implement both and default to B.

### Issues found in the motivation doc

#### 1. ΔE formula inconsistency

The prose (line 19) defines:

> ΔE_t = log Z(x_≤t) − f(x_t)

This uses the **current** state's partition function. But the pseudocode
(lines 69–72) computes:

```python
log_z_next = torch.logsumexp(logits_next, dim=-1)  # NEXT state
spill = log_z_next - f_xt
```

These are different quantities. The pseudocode version (next-state log Z) is
the one that makes sense for lookahead — it measures how "disrupted" the
model's belief is *after* committing to a candidate. We adopt the **code's**
definition and note the prose is wrong.

#### 2. The logprobs-vs-logits confusion

Section 2 of the refined variant says:

> "you can leverage the logits and log_probs returned by the server"

The OpenAI-compatible endpoint returns `logprobs` for the **top N chosen
tokens**, not raw logits for the full vocabulary. You cannot compute
`logsumexp(logits)` from partial log-probabilities. To compute log Z you need
one of:
- The native llama.cpp `/completion` endpoint with `"logits": true`.
- `llama-cpp-python` bindings in-process.
- A proxy spill mode (see Section 3 below).

#### 3. Multi-token entities are invisible to per-token spill

The doc's example says "Jeff Bezos" should show low ΔE. But tokenisation
splits this into (at least) two tokens: `"Jeff"` and `" Bezos"`. The spill
signal likely only fires on the *second* token — the first token `"Jeff"` is
compatible with many continuations and won't show a spike. Meanwhile, `"Elon"`
also won't spike until `" Musk"` follows.

This means **single-step lookahead may not catch the divergence at the
critical token.** For entity-swap scenarios we may need 2-step lookahead,
or a post-hoc "entity consistency" check that operates on decoded spans
rather than individual tokens.

#### 4. Why would successor-state log Z spike for a wrong fact?

This is the core theoretical assumption and it's never justified. The
argument is roughly: "a factually wrong token puts the model in an
incoherent state, which shows up as high log Z (flat distribution = model
doesn't know what comes next)."

This is plausible but not guaranteed:
- **False positives:** Creative/novel text produces high successor entropy
  too. A poem or a joke might trigger spill penalties despite being correct.
- **False negatives:** A confidently wrong model (trained on stale data)
  may produce *low* log Z for wrong continuations, because its internal
  world-model is consistently wrong.

We should treat MSS as a **hypothesis to validate**, not a proven technique.
The eval harness is therefore the most important deliverable.

#### 5. Raw ΔE values are scale-dependent across contexts

(From CODEX/GEMINI) Raw ΔE magnitudes shift depending on context length,
model state, and prompt domain. A fixed τ threshold (4.2 or 4.8) will
over-fire in some contexts and under-fire in others. Per-step normalization
is needed.

#### 6. Other missing concerns

- **EOS handling**: if EOS is in the top-k, it needs scoring too.
- **Tokenization bridge**: pseudocode operates on token IDs but user
  interacts with text. Need tokenize/detokenize plumbing.
- **Temperature interaction**: doc assumes T=0. With T>0 the logits are
  scaled before softmax — this changes both surprisal and ΔE magnitudes.
- **τ calibration**: values 4.2 and 4.8 are asserted without derivation.

---

## 2. Research Hypothesis & Success Criteria

(Adapted from CODEX — this was missing from v2.)

**Hypothesis:** MSS can reduce factual/logical errors versus greedy decoding
at comparable fluency, with bounded latency overhead when using batched
one-step lookahead.

**Go/no-go gates** (must pass ALL to ship MSS as default mode):

| Metric | Target |
|--------|--------|
| Error rate reduction on stress tasks | ≥ 20% relative to greedy |
| Fluency degradation (perplexity proxy) | ≤ 15% increase |
| P95 latency per token | ≤ 2x greedy (sequential) or ≤ 1.3x (adaptive) |
| False positive rate (spill on correct tokens) | < 5% of steps |

If these gates fail, MSS is **not shipped** — the eval harness produces a
"no-go" finding with diagnostic data explaining where the signal broke down.

---

## 3. Unified Scoring Function

We unify the two motivation.md variants and add robust normalization
(from CODEX/GEMINI):

### Primary mode (raw logits available)

```python
# Per-candidate raw spill
spill_raw = log_z_next - candidate_logit          # next-state definition

# Normalize across the k candidates at this step
norm_spill = robust_zscore(spill_raw, all_k_spills)

# Final score (minimize)
score = surprisal + beta * max(0, norm_spill - tau)
```

Where:
```python
def robust_zscore(value, values):
    """Z-score using median and MAD — robust to outliers."""
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + 1e-8
    return (value - med) / mad
```

After z-normalization, τ operates on a standardized scale. Default: τ = 1.0.

### Proxy mode (raw logits unavailable)

(Adapted from CODEX/GEMINI.) When the backend only returns top-N log-probs:

```python
spill_proxy = entropy(p_next) + lambda_ * (1 - top1_margin(p_next))
norm_spill = robust_zscore(spill_proxy, all_k_proxy_spills)
score = surprisal + beta * max(0, norm_spill - tau)
```

Where `top1_margin = p_top1 - p_top2`. This approximates the "logical rift"
using distribution flatness rather than raw partition function values.

### Variant A (ablation baseline)

For ablation, also implement the raw unthresholded version:
```python
score_a = alpha * surprisal + beta * spill_raw   # no threshold, no normalization
```

### Defaults

| Parameter | Raw mode | Proxy mode |
|-----------|----------|------------|
| k | 3 | 3 |
| beta | 2.0 | 2.0 |
| tau (post-normalization) | 1.0 | 1.0 |
| lambda | — | 0.5 |

Model-specific overrides (to be validated in Phase 5):

| Model | beta | tau | Notes |
|-------|------|----|-------|
| Qwen 3.5 35B | 1.8 | 1.2 | Higher inertia → needs less aggressive penalty |
| Qwen 3.5 9B | 2.2 | 0.8 | Shallower wells → tighter leash |

---

## 4. Adaptive Compute Gating

(From GEMINI — this was entirely missing from v2.)

Most tokens are "easy" — the model is confident and ΔE is low. Running
k lookaheads on every token wastes compute. Instead:

```
if top1_prob >= 0.92 and entropy < entropy_threshold:
    # Fast path: greedy pick, no lookahead
    select argmax token
else:
    # Slow path: full MSS with k candidates
    run lookahead and scoring
```

This makes MSS overhead proportional to the model's uncertainty, not the
sequence length. On typical text, ~70-80% of tokens take the fast path.

**Escalation:** Start with k=1 (single lookahead on top candidate). If its
ΔE > τ, escalate to k=3. This is "adaptive k" and further reduces cost.

**Panic fallback** (from CODEX): If ALL k candidates have norm_spill > τ +
panic_margin (e.g., 2.0), the model is deeply confused. In this case:
1. Retry with k=5.
2. If still all high-spill, fall back to greedy for this step and flag it
   in diagnostics.

---

## 5. Server Strategy: llama-server Fork Analysis

### The problem

Stock llama-server does **not** expose full vocabulary logits via any HTTP
endpoint. The native `/completion` accepts `n_probs` but returns only top-N
probabilities, not raw logits. `"logits": true` is not a supported parameter
on the HTTP API (it exists in the C API only). The OAI-compat `/v1/completions`
doesn't support `logprobs` at all.

This means the entire MSS scoring function (`log Z = logsumexp(full_logits)`)
cannot be computed through the stock HTTP API. Additionally, KV-cache
fork/extend/prune is not exposed (only save/restore to disk), and there is no
"evaluate k continuations from a shared prefix" endpoint.

### Three strategies evaluated

#### Strategy A: HTTP client with top-N approximation (no server changes)

- Use `n_probs` with large N (100–1000) to approximate log Z.
- Multiple HTTP round trips per token (1 + k lookaheads).
- **Pros:** zero server changes, fastest to implement.
- **Cons:** log Z approximation is lossy; HTTP overhead dominates latency
  (~3-5x greedy); serializing even top-1000 probs per request is wasteful.
- **Verdict:** sufficient for proxy mode, but inadequate for raw-logit mode.

#### Strategy B: `llama-cpp-python` in-process (no server at all)

- Use the Python bindings to call the C API directly.
- Full access to logits, `llama_kv_cache_seq_cp` for cache branching,
  batched evaluation — no HTTP overhead.
- **Pros:** full logit access, exact log Z, zero serialization cost, cache
  branching via `llama_kv_cache_seq_cp`, single-process simplicity.
- **Cons:** ties deployment to a specific llama.cpp version, Python GIL
  constrains parallelism, more complex setup than hitting an endpoint.
- **Verdict:** best for validation (Phases 1–5). Full signal, minimal
  infrastructure.

#### Strategy C: Fork llama-server (custom endpoint)

Add a single `/mss_step` endpoint (~200 lines of C++ in `server.cpp`):

```
POST /mss_step
{
  "prompt": "The founder of Blue Origin is...",
  "candidates": [12045, 8834, 2917],   // top-k token IDs
  "beta": 2.0, "tau": 1.0
}
→ {
    "selected": 12045,
    "selected_text": "Jeff",
    "diagnostics": [
      {"token_id": 12045, "text": "Jeff", "spill": 1.2, "score": 0.8},
      {"token_id": 8834,  "text": "El",   "spill": 7.1, "score": 5.9},
      {"token_id": 2917,  "text": "Mark", "spill": 6.3, "score": 4.8}
    ]
  }
```

Internally: fork KV-cache via `llama_kv_cache_seq_cp`, evaluate k branches
in one batch, compute log Z + scoring in C++, return only the result.
Raw logits never cross the HTTP boundary.

- **Pros:** production-grade latency (~1.1x greedy per GEMINI's estimate),
  clean HTTP API, logits stay server-side.
- **Cons:** maintenance burden (llama.cpp moves fast — ~50 commits/day on
  main), delays Phase 1, scope creep risk. **And if the signal doesn't
  exist, the fork is wasted.**

### Recommendation: staged approach

| Stage | When | Strategy | Why |
|-------|------|----------|-----|
| **Validation** (Phases 1–5) | Now | **B** (`llama-cpp-python`) | Full signal access, no fork overhead, answers the go/no-go question first |
| **Productionization** (Phase 6+) | After go | **C** (minimal fork) | Only justified once the signal is validated. Target upstream inclusion. |
| **Fallback** | If B is too painful | **A** (HTTP + proxy) | Degraded but functional. Tests proxy mode fidelity. |

This means:
- The primary backend for Phases 1–5 is `llama-cpp-python`, not `httpx`.
- The `LlamaCppBackend` wraps the Python bindings, not the HTTP API.
- An `HttpBackend` exists as a secondary implementation for proxy-mode
  testing and for users who already have a running server.
- The fork (Strategy C) only happens if Phase 5 says "go" AND profiling
  shows the Python bindings are a deployment bottleneck.

---

## 6. Architecture

```
spillage/
├── pyproject.toml
├── spillage/
│   ├── __init__.py
│   ├── backend.py              # Backend protocol
│   ├── backend_native.py       # Primary: llama-cpp-python (full logits, cache branching)
│   ├── backend_http.py         # Secondary: HTTP client (proxy mode, n_probs)
│   ├── backend_mock.py         # Testing: synthetic logit distributions
│   ├── scorer.py               # ΔE, robust z-score, variants A/B, proxy mode
│   ├── sampler.py              # MSS generation loop + adaptive gating
│   ├── cli.py                  # CLI entry point (click)
│   └── config.py               # hyperparams, model presets
├── tests/
│   ├── conftest.py             # MockBackend fixture
│   ├── test_scorer.py
│   ├── test_sampler.py
│   └── test_backend.py
├── evals/
│   ├── cases.py                # stress test definitions
│   ├── runner.py               # runs greedy vs MSS-A vs MSS-B vs MSS-proxy
│   └── report.py               # formats results table + go/no-go verdict
├── motivation.md
└── CLAUDE_PROPOSAL.md
```

---

## 6. Implementation Plan

### Phase 1 — Scaffolding, Backend & Tokenization

**Goal:** Stand up the project and talk to a local llama.cpp server.

#### 1.1 Project skeleton

- `pyproject.toml` with deps: `httpx`, `numpy`, `click`, `pytest`,
  `pytest-asyncio`.
- Minimal `spillage/__init__.py`.

#### 1.2 `backend.py` — Inference Backend Protocol

```python
class Backend(Protocol):
    def get_logits(self, token_ids: list[int]) -> LogitResult: ...
    def get_logits_batch(self, token_id_seqs: list[list[int]]) -> list[LogitResult]: ...
    def tokenize(self, text: str) -> list[int]: ...
    def detokenize(self, token_ids: list[int]) -> str: ...
    def mode(self) -> Literal["raw", "proxy"]: ...
    def eos_token_id(self) -> int: ...

@dataclass
class LogitResult:
    # Raw mode: full vocabulary
    logits: np.ndarray | None    # (V,) or None in proxy mode
    log_z: float                 # logsumexp(logits) if raw; NaN if proxy
    # Always available
    top_k_ids: np.ndarray        # (k,) indices of highest logits/probs
    top_k_logits: np.ndarray     # (k,) corresponding values
    top_k_probs: np.ndarray      # (k,) probabilities
    # Proxy mode fields
    entropy: float | None        # entropy of successor distribution
    top1_margin: float | None    # p_top1 - p_top2
```

Concrete implementations:

- **`NativeBackend`** (`backend_native.py`) — wraps `llama-cpp-python`
  in-process. Loads the GGUF model directly. Full access to raw logits via
  `llama_get_logits()`, KV-cache branching via `llama_kv_cache_seq_cp()`,
  and batched evaluation. Tokenize/detokenize via the model's built-in
  tokenizer. **This is the primary backend for validation.**

- **`HttpBackend`** (`backend_http.py`) — hits llama-server's `/completion`
  with `n_probs: N`. Operates in proxy mode only (no full logits). Useful
  for users who already have a running server and for testing proxy mode
  fidelity. Connection via `httpx.AsyncClient`.

- **`MockBackend`** (`backend_mock.py`) — returns synthetic logit
  distributions. Supports both raw and proxy modes. Allows injecting
  specific ΔE scenarios (low-spill, high-spill, equal, panic).

#### 1.3 Backend smoke tests

Verify against the actual model file:
- `NativeBackend`: loads GGUF, `get_logits()` returns full vocab (V,)
  array, `log_z` matches manual `logsumexp`, tokenize/detokenize
  round-trips, `llama_kv_cache_seq_cp` works for branching.
- `HttpBackend`: `n_probs=100` returns top-100 probs, proxy fields
  (entropy, top1_margin) populated correctly.
- Both: EOS token ID is correct for the loaded model.

**Acceptance:** all three backends (`Native`, `Http`, `Mock`) pass the
`Backend` protocol. `NativeBackend` returns full logits. `HttpBackend`
operates in proxy mode. Both raw and proxy paths exercised.

---

### Phase 2 — Scorer

**Goal:** Implement all scoring math, fully testable without a server.

#### 2.1 `scorer.py`

```python
def compute_spill_raw(candidate_logit: float, log_z_next: float) -> float:
    """ΔE = log_z_next - candidate_logit (next-state definition)."""
    return log_z_next - candidate_logit

def compute_spill_proxy(entropy: float, top1_margin: float,
                        lambda_: float = 0.5) -> float:
    """Proxy spill when raw logits are unavailable."""
    return entropy + lambda_ * (1.0 - top1_margin)

def robust_zscore(values: np.ndarray) -> np.ndarray:
    """Per-step normalization using median and MAD."""
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + 1e-8
    return (values - med) / mad

def score_thresholded(surprisal: float, norm_spill: float,
                      beta: float = 2.0, tau: float = 1.0) -> float:
    """Primary scoring function. Lower is better."""
    return surprisal + beta * max(0.0, norm_spill - tau)

def score_raw(surprisal: float, spill: float,
              alpha: float = 1.0, beta: float = 1.5) -> float:
    """Variant A ablation. Lower is better."""
    return alpha * surprisal + beta * spill
```

**Acceptance:** unit tests covering:
- Zero-spill case (norm_spill < τ → no penalty).
- High-spill case (norm_spill >> τ → correct candidate re-ranked above "vibe").
- Edge: all candidates have equal spill → robust_zscore returns zeros.
- Proxy spill tracks raw spill directionally on synthetic distributions.
- Variant A vs thresholded produce different rankings on the same input.

---

### Phase 3 — The Generation Loop (Sampler)

**Goal:** Decode a full response using MSS with adaptive gating.

#### 3.1 `sampler.py`

```python
async def generate(
    prompt: str,
    backend: Backend,
    *,
    k: int = 3,
    beta: float = 2.0,
    tau: float = 1.0,
    max_tokens: int = 256,
    variant: Literal["thresholded", "raw"] = "thresholded",
    temperature: float = 0.0,
    adaptive: bool = True,
    confidence_threshold: float = 0.92,
    panic_margin: float = 2.0,
    on_token: Callable[[TokenEvent], None] | None = None,
) -> GenerateResult:
```

Loop:

1. **Get logits** for current prompt → extract top-k token IDs + logits.
2. If `temperature > 0`, apply temperature scaling before top-k extraction.
3. **Adaptive gate** — if `adaptive=True` and top-1 probability ≥
   `confidence_threshold`, skip lookahead (greedy pick). Log as "fast path"
   in diagnostics.
4. **Lookahead** — for each candidate:
   - Detokenize candidate ID to text.
   - Append to prompt.
   - Get successor logits via `backend.get_logits(extended_prompt)`.
   - Use `get_logits_batch` if available.
5. **Compute spill** (raw or proxy depending on `backend.mode()`).
6. **Normalize** spill values via `robust_zscore` across the k candidates.
7. **Score** each candidate.
8. **Panic check** — if all norm_spill > τ + panic_margin:
   - If k < 5, retry with k=5.
   - Otherwise, fall back to greedy and flag in diagnostics.
9. **Select** candidate with minimum score.
10. **EOS check** — if selected is EOS or all top-k are EOS, stop.
11. Append winning token text to prompt. Fire `on_token` callback.
12. Repeat until max_tokens or EOS.

#### 3.2 `TokenEvent` for diagnostics

```python
@dataclass
class TokenEvent:
    step: int
    candidates: list[CandidateScore]
    selected: int                      # index into candidates
    wall_time_ms: float
    fast_path: bool                    # True if adaptive gate skipped lookahead
    panic: bool                        # True if panic fallback triggered

@dataclass
class CandidateScore:
    token_id: int
    token_text: str
    log_prob: float
    spill_raw: float                   # raw ΔE (or proxy spill)
    spill_normalized: float            # after robust z-score
    score: float
```

#### 3.3 KV-cache branching

(From GEMINI — more explicit than v2.)

With `llama.cpp --parallel k+1`:
- **Slot 0 (root):** holds the current committed context.
- **Slots 1..k (branches):** forked from slot 0, each appending one
  candidate token.
- After scoring, the winning branch is promoted to slot 0; losing slots
  are cleared.
- This avoids re-processing the full prefix on each step.

The naive fallback (sequential requests, one slot, relying on prompt cache)
also works — just slower. We start with the naive path and optimise later.

**Acceptance:** end-to-end generation via MockBackend with predictable
outputs (deterministic synthetic logits → deterministic token selection).
Integration test against a live server producing coherent text that
differs from greedy on at least one stress-test prompt.

---

### Phase 4 — CLI

**Goal:** Usable from the terminal.

```
spillage generate --prompt "..." [--k 3] [--beta 2.0] [--tau 1.0]
                  [--variant thresholded|raw] [--max-tokens 256]
                  [--temperature 0.0] [--adaptive / --no-adaptive]
                  [--server http://localhost:8080]
                  [--model-preset qwen35-35b|qwen35-9b|auto]
                  [--verbose] [--json-trace FILE]

spillage greedy   --prompt "..." [--max-tokens 256]
                  [--server http://localhost:8080]

spillage audit    --server http://localhost:8080
                  # reports endpoint capabilities (raw logits? batching? EOS ID?)
```

- `--model-preset` sets beta/tau from the model presets table.
- `--json-trace` dumps all TokenEvents to a file for post-hoc analysis.
- `spillage audit` runs the Phase 1.3 endpoint capability check.
- Stdin support: `echo "prompt" | spillage generate`.

**Acceptance:** run from terminal, see streaming output, verbose mode shows
full decision trace, json-trace is valid parseable JSON.

---

### Phase 5 — Evaluation Harness

**Goal:** Quantify whether MSS actually helps. **This is the most important
phase** — the theoretical basis is unproven and the hyperparameters are
uncalibrated.

#### 5.1 Stress tests

| Test | Prompt | Pass condition |
|------|--------|----------------|
| Entity Swap | "The founder of SpaceX is Elon Musk, but the founder of Blue Origin is..." | "Jeff Bezos" in output |
| Multi-digit Math | "What is 347 + 268?" | "615" in output |
| Dead Person Bio | "Is [living person] alive? They..." | No death date generated |
| Single-token Factual | "The chemical symbol for gold is..." | "Au" (cleanest ΔE signal) |
| Repetition Trap | "A B C A B C A B C A B..." | Continues with "C" |
| Negation | "The capital of Australia is not Sydney, it is..." | "Canberra" |

#### 5.2 Comparison modes

Run each test under:
1. **Greedy** — standard argmax.
2. **MSS thresholded (raw)** — primary mode, raw logits.
3. **MSS thresholded (proxy)** — proxy mode, for comparison.
4. **MSS raw (variant A)** — ablation.

Report per test: token trace at divergence point, all candidate ΔE values,
correctness verdict.

#### 5.3 τ calibration

1. Run 50+ "known-correct" prompts through greedy decoding.
2. Record ΔE (raw and normalized) at every token.
3. Plot the distribution. Set τ at the 95th percentile of normalized ΔE.
4. Compare against the motivation.md recommendations and the z-normalized
   defaults.

#### 5.4 Ablation matrix

(From CODEX.)

| Ablation | What it tests |
|----------|---------------|
| Raw mode vs proxy mode | Is the full-logit signal significantly better? |
| With vs without z-normalization | Does normalization help or hurt? |
| k=2 vs k=3 vs k=5 | Cost-quality tradeoff |
| Adaptive gating on vs off | Latency savings vs missed catches |
| beta sweep (1.0, 1.5, 2.0, 3.0) | Penalty sensitivity |

#### 5.5 Latency measurement

Wall-clock time per token: greedy vs MSS-sequential vs MSS-adaptive.
Report median, P95, total generation time.

#### 5.6 Go/no-go verdict

The eval runner outputs a structured verdict against the gates in Section 2.
If any gate fails, the report explains which and recommends next steps
(parameter tuning, deeper lookahead, or abandoning the approach).

**Acceptance:** `python -m evals.runner --server URL` produces a results
table, calibration histogram, ablation matrix, and a go/no-go line.

---

### Phase 6 — Optimisations (stretch, deferred, requires Phase 5 go)

Prioritised by expected impact:

1. **llama-server fork** — if the native Python backend is validated but
   deployment needs an HTTP API, fork llama-server to add `/mss_step`
   endpoint (see Section 5, Strategy C). Target upstream inclusion to
   avoid long-term maintenance. Only justified if Phase 5 passes go/no-go.
2. **2-step lookahead for entities** — when ΔE is borderline (within 20%
   of τ), do a second lookahead step to catch multi-token entity spills.
3. **Speculative decoding hybrid** — use a smaller draft model for
   lookahead forward passes. Cheaper but noisier signal.

---

## 7. Risks and Mitigations

(From CODEX — this was missing from v2.)

| Risk | Mitigation |
|------|-----------|
| **Core signal doesn't exist** — ΔE doesn't reliably distinguish correct from incorrect tokens | Phase 5 go/no-go gate. If correct/incorrect ΔE distributions overlap > 50%, stop. |
| **`llama-cpp-python` version churn** — bindings lag behind llama.cpp main | Pin to a known-good release. The validation phase only needs stable inference, not bleeding-edge features. |
| **Proxy mode is useless** — entropy+margin doesn't track raw ΔE | Phase 5 ablation tests this directly. If proxy fails, raw logits are a hard requirement and HTTP-only deployment needs the fork. |
| **Premature fork** — investing in server C++ before validating the signal | Staged approach: fork only after Phase 5 go. |
| **Fork maintenance burden** — llama.cpp has ~50 commits/day | Keep the patchset minimal (one endpoint, ~200 LOC). Target upstream inclusion. If rejected upstream, maintain as a patch series, not a full fork. |
| **Spill normalization is noisy** — MAD is unstable with k=3 | Test with k=5; consider IQR if MAD fails. Fall back to unnormalized if signal is cleaner. |
| **Over-penalises creative text** — poems/jokes have naturally high ΔE | Add creative-writing prompts to eval. Consider per-domain τ presets. |
| **Multi-token entities invisible** to 1-step lookahead | Phase 6 item 2 (2-step lookahead). Measure miss rate in Phase 5. |
| **τ values don't transfer across temperature** | Calibrate τ separately at T=0 and T=0.7. Document which T the presets assume. |

---

## 8. Open Questions

1. **Does the signal actually exist?** Phase 5 answers this.

2. **Which llama.cpp endpoint to use?** Phase 1.3 audits this against the
   actual running server.

3. **Is one-step lookahead enough?** The multi-token entity problem
   suggests it may not be. Start with 1-step, measure miss rate, add
   2-step in Phase 6 if needed.

4. **Is z-normalization actually better?** It makes τ scale-invariant, but
   with only k=3 samples the median/MAD estimate is noisy. The ablation
   matrix (Phase 5.4) tests this directly.

5. **Proxy mode fidelity** — does entropy + margin track raw ΔE well enough
   to preserve the signal? If not, proxy mode is useless and we must
   require raw logit access. The ablation answers this.

---

## 10. Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `llama-cpp-python` | Primary backend — in-process inference with full logit access | >=0.3 |
| `numpy` | Logit math (logsumexp, softmax, MAD) | >=1.26 |
| `click` | CLI framework | >=8.0 |
| `httpx` | Secondary backend — HTTP client for proxy mode | >=0.27 |
| `pytest` | Testing | >=8.0 |

Runtime: a GGUF model file for the native backend, OR a running
llama-server for the HTTP backend (proxy mode only).
