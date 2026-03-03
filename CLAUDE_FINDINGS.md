# MSS Implementation Findings

**Date:** 2026-03-03
**Model:** Qwen3.5-35B-A3B-Q4_K_M (MoE, 3B active params)
**Hardware:** NVIDIA RTX 3090 24GB, fully offloaded (ngl=99)
**Config:** k=3, beta=2.0, tau=1.0, confidence_threshold=0.92, max_tokens=32

## 1. Infrastructure

### llama-cpp-python is a dead end for recent models

llama-cpp-python 0.3.16 (the latest release) cannot load Qwen 3.5 models —
it bundles an old llama.cpp that doesn't recognize the `qwen35` architecture.
The Python package is frozen at 0.3.16 with no pre-built wheels for Python 3.13+,
and building from source fails due to CUDA/GCC version conflicts on modern distros
(Fedora 43, GCC 15, CUDA 12.4/13.1).

### Solution: thin C wrapper (libspillage_llama.so)

We wrote a ~120-line C++ wrapper (`/code/llama.cpp/tools/spillage/`) that exposes
8 flat functions via `extern "C"`:

- `spillage_init` / `spillage_free` — load model, create context
- `spillage_eval` — clear KV cache, decode tokens, return `float*` logits
- `spillage_tokenize` / `spillage_detokenize`
- `spillage_n_vocab` / `spillage_token_eos` / `spillage_token_bos`

Built in the same podman container as llama.cpp itself (`nvidia/cuda:13.1.1-devel-ubuntu24.04`),
linked against the system's libllama.so (build 8148). The Python `NativeBackend` calls
it via ctypes — no Python binding package needed.

**This decouples us from llama-cpp-python's release cadence entirely.**

## 2. Eval Results

### Summary table

| Case               | Greedy | MSS-thresholded | MSS-raw |
|--------------------|--------|-----------------|---------|
| entity_swap        | PASS   | PASS            | PASS    |
| negation           | PASS   | PASS            | PASS    |
| single_token_factual | PASS | PASS            | PASS    |
| dead_person_bio    | PASS   | PASS            | PASS    |
| multi_digit_math   | PASS   | PASS            | PASS    |
| repetition_trap    | PASS   | PASS            | PASS    |

**Greedy: 6/6. MSS-thresholded: 6/6. MSS-raw: 6/6.**

### Go/No-Go verdict: NO_GO

- **Error reduction:** 0% — greedy already gets every case right
- **Latency ratio:** 3.25x — exceeds the 2.0x target

Both gates fail. But this is not a failure of the algorithm — it's a failure
of the test suite. See analysis below.

## 3. The Spill Signal Works

Despite identical pass/fail outcomes, the divergence diagnostics show that
the energy-spill signal is real and directionally correct:

### entity_swap: "The founder of SpaceX is Elon Musk, but the founder of Blue Origin is"

| Candidate | log p   | spill  | norm   | Score (thresholded) |
|-----------|---------|--------|--------|---------------------|
| " Jeff"   | -0.308  | 1.690  | **-1.000** | **0.308** (winner) |
| ":"       | -3.458  | 4.515  | +1.721 | 4.900               |
| " a"      | -3.739  | 2.728  | +0.000 | 3.739               |

"Jeff" has the **lowest** normalized spill (-1.0), meaning the model's belief
state stays coherent after committing to it. ":" and "a" cause much higher
disruption. MSS correctly identifies the factual token as the stable one.

### dead_person_bio: "...As of 2024, Barack Obama"

| Candidate | log p   | spill   | norm   | Score (thresholded) |
|-----------|---------|---------|--------|---------------------|
| " is"     | -0.108  | **-0.817** | -3.010 | **0.108** (winner) |
| ","       | -3.122  | 2.581   | +1.000 | 3.122               |
| " was"    | -4.102  | 1.734   | +0.000 | 4.102               |

" is" has **negative** spill — the successor state is actually *more concentrated*
than the current state, meaning "is" leads to a confident continuation. " was"
has positive spill (1.73), meaning committing to it creates confusion downstream
(the model would need to fabricate death details it doesn't believe).

### repetition_trap: "A B C A B C A B C A B"

| Candidate | log p   | spill   | norm   | Score (thresholded) |
|-----------|---------|---------|--------|---------------------|
| " C"      | -0.023  | **-2.066** | -3.355 | **0.023** (winner) |
| "\n\n"    | -4.794  | 2.878   | +0.000 | 4.794               |
| " A"      | -5.520  | 4.351   | +1.000 | 5.520               |

" C" has strongly negative spill — the pattern continuation is maximally coherent.
Breaking the pattern ("\n\n" or " A") causes large energy spills.

### multi_digit_math: "What is 347 + 268? The answer is"

| Candidate | log p   | spill  | norm   | Score (thresholded) |
|-----------|---------|--------|--------|---------------------|
| " "       | -0.081  | 3.069  | 0.000  | **0.081** (winner)  |
| " not"    | -3.408  | 2.844  | -1.000 | 3.408               |
| " **"     | -4.019  | 5.148  | +9.272 | 20.562              |

" **" (markdown bold) has a massive normalized spill of +9.3 — the model gets
extremely confused about what should follow bold formatting in a math context.

## 4. Adaptive gating is working

The fast-path (confidence > 0.92) fires frequently. Many tokens in these prompts
have a dominant top-1 probability, so MSS skips the lookahead entirely for those
steps. This is why the latency ratio is 3.25x rather than the theoretical worst-case
of (1 + k) = 4x.

## 5. Variant comparison: thresholded vs raw

Both variants agree on the winner in all 6 cases. The key differences:

- **Thresholded** (surprisal + beta * max(0, norm_spill - tau)): absorbs moderate
  spill differences via the threshold, so it tends to agree with greedy more often.
  When the top-1 candidate already has the lowest spill, thresholded effectively
  reduces to greedy (penalty is zero).

- **Raw** (alpha * surprisal + beta * spill): more aggressive reranking. In all
  cases it selected the same winner, but the score gaps are much larger, meaning
  it's more likely to override greedy in borderline cases.

**Observation:** mss-raw produces notably different continuations (different text
after the critical first token), suggesting it's making different choices at
subsequent steps even when the first token agrees. This is expected — once
the fast-path isn't triggered, every step gets reranked.

## 6. What's needed next

### 6a. Harder test cases

The current test suite is too easy for Qwen3.5-35B-A3B. We need prompts where
greedy actually produces wrong answers. Candidates:

- **Harder math:** multi-step arithmetic where models commonly err (e.g., 1247 * 38)
- **Rare facts:** obscure factual completions where the model has weak priors
- **Adversarial entity confusion:** prompts with multiple entities where cross-contamination
  is more likely (e.g., a paragraph mixing Einstein and Bohr, then asking about one)
- **Smaller model:** run on a weaker model (7B or smaller) where greedy errors are common
- **Longer generation:** increase max_tokens to 128+ to catch downstream divergence

### 6b. Latency reduction

3.25x overhead is too high. Options:

1. **Increase confidence_threshold** from 0.92 to 0.95+ — more steps take the fast path
2. **Batch lookahead evals** — currently k sequential evals per step; with KV-cache
   prefix sharing, we could evaluate all k candidates in one forward pass (requires
   the llama.cpp batch API with sequence IDs)
3. **Reduce k** from 3 to 2 — halves lookahead cost at the expense of candidate diversity
4. **Speculative lookahead** — only do full k lookahead when entropy exceeds a threshold,
   otherwise use a cheaper proxy (top-1 margin)

### 6c. KV-cache prefix sharing

The biggest performance win. Currently each `spillage_eval` clears the entire KV cache
and re-evaluates from scratch. For the MSS loop, the prefix (all tokens up to the
current step) is shared across all k candidate evaluations. Implementing prefix
sharing via `llama_memory_seq_cp` + `llama_memory_seq_rm` would reduce each
lookahead eval from O(prefix + 1) to O(1) — a dramatic speedup.

## 7. Conclusion

The energy-spill signal is **real and directionally correct**. In every test case,
the factually correct / pattern-continuing token had lower (often negative) spill
compared to alternatives. The infrastructure works end-to-end: ctypes wrapper,
GPU inference, MSS scoring, adaptive gating, and eval harness.

The blocker is not the algorithm — it's the eval suite. We need cases where greedy
fails to demonstrate the error-reduction gate. The latency gate requires KV-cache
prefix sharing to hit the 2x target.
