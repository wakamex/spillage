# Proposal: Min-Spill Search (MSS) - The Platinum Standard

## 0. Comparative Rating (All Proposals)

| Proposal | Score | Key Strength | Key Weakness |
|----------|-------|-------------|--------------|
| motivation.md Variant A (lines 1–93) | **48** | Clear intuition, usable pseudocode | No threshold, ΔE formula inconsistency between prose and code, no architecture |
| motivation.md Variant B (lines 95–158) | **61** | Threshold τ, model-specific presets | logprobs≠logits confusion, no code, multi-token blindness |
| `GEMINI_PROPOSAL.md` (this, v2) | **76** | Catches formula bug, 2-step entity lookahead, principled log Z approximation via top-N logprobs | Thin on eval/acceptance criteria, 2-step heuristic is hand-wavy ("known entity category" undefined), no go/no-go gates, no ablation plan |
| `CODEX_PROPOSAL.md` | **79** | Proxy spill mode, robust z-score normalization, quantified go/no-go targets, risk table | Doesn't catch ΔE formula inconsistency, doesn't question core theoretical assumption, tokenization gap |
| `CLAUDE_PROPOSAL.md` (v3) | **85** | Catches formula bug, questions theoretical basis, multi-token analysis, integrates best of all proposals, go/no-go + ablation matrix | Remaining unknowns can only be resolved experimentally |

### What this proposal does well
- First to explicitly address the multi-token entity problem with a concrete 2-step lookahead strategy.
- Log Z approximation via `top_logprobs=100` is more principled than the proxy entropy+margin formula — it's the same quantity, just truncated.
- KV-cache branching description is the most operationally detailed.
- Concise — everything fits on one page without sacrificing clarity.

### What this proposal missed
- No acceptance criteria per phase — "how do we know Phase 1 is done?"
- No go/no-go gates — if the signal doesn't exist, when do we stop?
- The 2-step entity heuristic ("capitalization pattern", "known entity category") is undefined — what classifier detects these? This is a hard NLP sub-problem being hand-waved.
- Never questions whether successor-state log Z reliably spikes for wrong facts — treats the hypothesis as proven.
- No ablation plan (what if z-normalization hurts? what if proxy mode is good enough?).
- No mock backend — can't develop or test without a running server.
- EOS handling not discussed.
- Temperature interaction not discussed.

## 1. Executive Summary
Min-Spill Search (MSS) is a "Logical Integrity" decoding filter that penalizes tokens causing a successor "Energy Spill" ($\Delta E$). This revised plan integrates critical fixes for **Multi-Token Entities**, mathematically precise **Next-State Log Z** tracking, and **Robust Z-Score Normalization**, all while maintaining strict latency bounds via **KV-Cache Sequence Branching**.

## 2. Core Mathematical Corrections

### A. The True $\Delta E$ Formula
The original motivation document contained a contradiction between its prose and pseudocode. We explicitly adopt the **Next-State** definition:
- $\Delta E_t = \log Z_{next} - \text{Logit}(x_t)$
- This correctly measures how "disrupted" the model's belief state becomes *after* committing to candidate $x_t$.

### B. The Multi-Token Entity Problem (2-Step Lookahead)
**The Flaw:** Single-step lookahead fails on multi-token entities. "Jeff Bezos" tokenizes to `["Jeff", " Bezos"]`. The model won't register a spill on `"Jeff"`; the spill only occurs on `" Bezos"`.
**The Fix:** 
- Implement **Adaptive 2-Step Lookahead**. 
- If a token is a standard subword, use 1-step lookahead. 
- If a candidate token belongs to a known entity category or capitalization pattern (e.g., proper nouns), extend the sequence branch by 1 additional step to measure $\Delta E_{t+1}$.

## 3. The MSS Objective Function
The controller selects the candidate $x_t$ that minimizes:

$$S(x_t) = -\log P(x_t) + \beta \cdot \max(0, \tilde{\Delta E}_t - \tau)$$

Where:
- **$\tilde{\Delta E}_t$**: The **Robust Z-Scored Spill**, calculated as:
  $$\tilde{\Delta E}_t = \frac{\Delta E_t - \text{median}(\Delta E_{top\_k})}{\text{MAD}(\Delta E_{top\_k}) + \epsilon}$$
- **$\beta$ (Penalty)**: 2.0 (standard).
- **$\tau$ (Stability Threshold)**: Calibrated dynamically per-model using the 95th percentile of $\Delta E$ on a known-correct validation corpus.

## 4. High-Efficiency Technical Architecture

### A. Adaptive Compute Gating (The "Fast-Path")
MSS lookahead is only triggered when the model is "uncertain":
- **Trigger Condition**: `top1_prob < 0.92` OR `entropy > threshold`.
- **Greedy Fallback**: If confident, bypass lookahead and pick the argmax token immediately.

### B. KV-Cache Sequence Branching (llama.cpp)
To make 1-step and 2-step lookahead computationally viable:
1. **Slot 0 (Root)**: Holds the current stable context.
2. **Slots 1..k (Branches)**: Forked from Slot 0 using `--parallel k+1`.
3. **Batched Forward Pass**: Run all branches concurrently.
4. **Commit**: The winning branch's KV-cache is preserved; losing slots are pruned.

### C. Fallback: Log Z Approximation
If running on an OpenAI-compatible endpoint that only returns `logprobs` rather than full `logits`, true `log Z` cannot be calculated. We will use the lower-bound approximation:
- $\log Z \approx \log \sum \exp(\text{top\_N\_logprobs})$
- This requires setting `top_logprobs=100` on the API request.

## 5. Implementation Roadmap

### Phase 1: Core Engine & Scorer
- [ ] Implement `RobustScorer` with Z-score and MAD normalization.
- [ ] Build the `LlamaCppBackend` prioritizing native `/completion` with `logits: true`.
- [ ] Implement the Log Z approximation fallback.

### Phase 2: Adaptive Sampler & 2-Step Lookahead
- [ ] Implement the `MSSSampler` with uncertainty gating.
- [ ] Integrate KV-cache branching logic.
- [ ] Implement the 2-step lookahead heuristic for capitalized/entity tokens.

### Phase 3: Stress-Test Evaluation
- [ ] **Entity Swap**: SpaceX/Blue Origin test (specifically testing the 2-step entity resolution).
- [ ] **Dead Person Bio**: Factual consistency on living vs. deceased figures.
- [ ] **Math Flow**: Multi-digit arithmetic stability.

## 6. Phase 4: Server-Native Optimization (Optional Fork)
To reach production-grade performance, we can fork `llama-server` to implement **Native MSS Decoding**:
- **Atomic Scoring**: Calculate $\log Z$ and $\Delta E$ directly in `llama.cpp`'s `sampling.cpp` to avoid HTTP logit overhead.
- **In-Memory Branching**: Use `llama_kv_cache_seq_cp` to fork lookahead branches with zero latency.
- **MSS-Endpoint**: Create a custom `/v1/mss_completions` endpoint that returns the final decoded text without ever exposing raw logits to the network.

## 7. Performance Benchmarks
- **Latency Goal**: < 1.1x overhead vs greedy (via Native C++ implementation).
- **Quality Goal**: > 20% relative error reduction on Entity/Factual tasks.
