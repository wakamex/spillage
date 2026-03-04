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

### Phase 3: Sequence-level ∆E reject/retry ← CURRENT
- **Insight**: per-token ∆E gating failed, but sequence-level min ∆E still separates
  correct vs incorrect (AUROC=0.802)
- **Approach**: run greedy first (cheap), check min(∆E) over whole sequence,
  only retry with full MSS if flagged
- New mode: `mss-seq-gated`
  1. `generate_greedy()` → result + per-token ∆E (already tracked)
  2. Compute min(∆E) over the sequence
  3. If min ∆E < threshold → accept (energy-consistent, probably correct)
  4. If min ∆E ≥ threshold → reject, re-run with `generate()` (full MSS)
- Reuses `delta_e_threshold` from MSSConfig (default -4.5)
- Adds `retried` field to `GenerateResult` for diagnostics

### Phase 4: Multi-token ∆E lookahead (if Phase 3 helps)
- At flagged steps: for each candidate, generate 3-5 greedy continuation tokens
- Compute ∆E at each continuation step
- Use min-pooling (paper's best strategy) to score each candidate's trajectory
- Select candidate with best (lowest) min-pooled ∆E
- Only feasible at Phase-3-gated steps to control cost

## Key Insight

Sequence-level min ∆E is a good detector of incorrect sequences. Use it to gate
expensive MSS retries: run greedy first, and only retry with full lookahead when
the energy profile flags the sequence as suspicious.
