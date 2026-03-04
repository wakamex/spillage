# MSS v2 — Adapting Spilled Energy for Decoding

## Context

Our `spill_raw` (scorer.py:23) IS the paper's ∆E — same formula:
```
∆E = logsumexp(logits at step i+1) - logit at step i
```

But using it for candidate SELECTION doesn't work because the logit term dominates
(70-80% of ∆E is just surprisal). Capitals benchmark: greedy 55%, MSS 48% (worse).

The paper uses ∆E for post-hoc DETECTION (AUROC 73-77%), not for decoding.

## Plan

### Phase 1: Validate ∆E as detector on our benchmark ✅ DONE
- Generated greedily on capitals → `results/capitals_qwen35_0b8_greedy_v3.json`
- Per-token ∆E recorded at zero extra cost (consecutive forward passes)
- **Results** (Qwen3.5-0.8B, 73 capitals, min-pooling):
  - AUROC = **0.802**
  - Optimal threshold = **-4.529** (Youden's J)
  - TPR = 54.8%, FPR = 0.0%, F1 = 0.708
  - Correct min ∆E: median=-6.693, mean=-6.923
  - Incorrect min ∆E: median=-4.354, mean=-5.116
- **Gate passed**: ∆E clearly separates correct from incorrect.

### Phase 2: Per-token ∆E-gated decoding ✅ DONE — FAILED
- Implemented `generate_delta_e_gated()` with per-token gating
- **Result**: correct and incorrect tokens have nearly identical per-token ∆E distributions
- Gate triggers randomly → lookahead picks worse tokens → score degrades
- Per-token gating is not viable

### Phase 3: Sequence-level ∆E reject/retry ✅ DONE — NEGATIVE RESULT
- Implemented `generate_seq_gated()` and `mss-seq-gated` mode
- **Capitals** (Qwen3.5-0.8B, threshold -3.0): 68/73 greedy → 68/73 seq-gated (no improvement)
  - Only 1/73 flagged for retry (Philippines, min ∆E=-1.267), MSS produced garbage on retry
  - 4/5 wrong answers have "normal" min ∆E (confident hallucinations)
- **Key finding**: min(∆E) over the sequence is driven by the repetition tail, not answer tokens
  - Most negative ∆E always at steps 20+ (repetitive continuation), not the answer
  - Detected degeneration (Philippines), not hallucination

### Phase 4: ∆E vs log_prob comparison ✅ DONE — ∆E ≈ CONFIDENCE
- Added per-token `log_prob` collection alongside `delta_e`
- **Head-to-head AUROC** (Qwen3.5-0.8B, 73 capitals):

  | Metric        | AUROC | Direction       |
  |---------------|-------|-----------------|
  | min_dE        | 0.747 | lower=correct   |
  | min_logprob   | 0.815 | higher=correct  |
  | mean_dE       | 0.721 | lower=correct   |
  | mean_logprob  | 0.606 | higher=correct  |
  | median_dE     | 0.725 | lower=correct   |
  | median_logprob| 0.538 | higher=correct  |

- **On answer tokens only** (first 3-5 tokens): ∆E at chance (0.36-0.51), log_prob at 0.85
- **SimpleQA** (Qwen3.5-35B-A3B, 100 cases, 21% pass rate): both near chance (~0.5-0.65)
- **Conclusion**: ∆E = log_z(i+1) - logit(i), and the logit term dominates.
  min(∆E) ≈ -max(logit) — it's a confidence proxy, not a distinct energy signal.
  The paper's AUROC results are likely reproducible with plain min(log_prob).

## Key Takeaways

1. **∆E is not useful for decoding** — per-token gating triggers randomly, sequence-level
   gating catches degeneration but not confident hallucinations
2. **∆E ≈ confidence** — the logit term dominates, making ∆E a proxy for token probability.
   The energy conservation framing is theoretically elegant but practically equivalent to
   simple confidence-based detection
3. **The paper's contribution is valid but narrow** — ∆E is a good training-free detector
   threshold, as claimed. It's just not a *different* signal from confidence
4. **MSS lookahead doesn't help** — even when flagged sequences are retried with full
   MSS, the alternative decoding doesn't fix confident hallucinations
