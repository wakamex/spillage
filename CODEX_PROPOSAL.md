# Min-Spill Search (MSS) Implementation Proposal

## 0) Comparative Rating (All Proposals)

Scoring criteria: theoretical rigor, implementability on local endpoints,
evaluation design, and honest treatment of unknowns.

| Proposal | Score | Key Strength | Key Weakness |
|----------|-------|-------------|--------------|
| motivation.md Variant A (lines 1–93) | **48** | Clear intuition, usable pseudocode | No threshold, ΔE formula inconsistency between prose and code, no architecture |
| motivation.md Variant B (lines 95–158) | **61** | Threshold τ, model-specific presets | logprobs≠logits confusion, no code, multi-token blindness |
| `GEMINI_PROPOSAL.md` (v2) | **76** | Catches formula bug, 2-step entity lookahead idea, principled log Z approximation | Thin on eval/acceptance criteria, 2-step heuristic is hand-wavy, no go/no-go gates |
| `CODEX_PROPOSAL.md` (this) | **79** | Proxy spill mode, robust z-score normalization, quantified go/no-go targets, risk table | Doesn't catch the ΔE formula inconsistency, doesn't question whether core signal exists, backend takes token_ids (tokenization gap) |
| `CLAUDE_PROPOSAL.md` (v3) | **85** | Catches formula bug, questions theoretical basis, multi-token analysis, integrates best of all proposals, go/no-go + ablation matrix | Remaining unknowns can only be resolved by running experiments |

### What this proposal does well
- Proxy spill mode is a practical fallback other proposals lacked initially.
- Go/no-go gates with quantified thresholds are disciplined.
- Risk/mitigation table forces honest accounting of failure modes.

### What this proposal missed
- The ΔE formula inconsistency (prose says current-state log Z, code says next-state) — should have caught this.
- Never questions whether successor-state log Z actually spikes for wrong facts — this is the entire theoretical bet and it's unvalidated.
- Backend interface uses `context_ids` (token IDs) but the user sends text — the tokenization bridge is hand-waved.
- MAD-based z-normalization with only k=3 samples is statistically fragile (median of 3 values, MAD of 3 values).

## 1) Evaluation of `motivation.md`

### What is strong
- The core direction is practical: top-`k` one-step lookahead is much cheaper than beam search.
- The proposal targets concrete failure modes (entity confusions, arithmetic slips, stale factual priors).
- Batching lookahead candidates is the right performance strategy for local inference backends.

### What needs clarification before implementation
- There are two different scoring formulations in the draft. We should standardize on one objective.
- `Delta E = log Z_next - f(x_t)` requires **raw logits**. If the endpoint only exposes normalized `logprobs`, `logsumexp(logprobs)=0`, so this signal collapses.
- `log Z_next` across different contexts can be scale-shifty; we need per-step normalization (or a shift-invariant proxy) before thresholding.
- Time-sensitive fact tests (for example CEO changes) require date-pinned ground truth, or we risk noisy evaluation.

## 2) Final Research Hypothesis

MSS can reduce factual/logical errors versus greedy decoding at comparable fluency, with bounded latency overhead when using batched one-step lookahead.

Primary success target:
- At least 20% relative reduction in error rate on stress tasks.
- At most 15% degradation in fluency/perplexity proxy.
- P95 latency increase <= 1.8x over greedy.

## 3) Proposed Decoder Specification

Use one canonical per-token score:

`score_i = surprisal_i + beta * max(0, norm_spill_i - tau)`

Where:
- `surprisal_i = -log p_t(i)`
- `spill_i_raw = logsumexp(logits_next_i) - logit_t(i)` (raw-logit mode)
- `norm_spill_i = robust_zscore(spill_i_raw over top-k candidates at step t)`
- `robust_zscore(v) = (v - median(v)) / (MAD(v) + 1e-6)`

If raw logits are unavailable, use proxy mode:
- `spill_i_proxy = entropy(p_next_i) + lambda * (1 - top1_margin_i)`
- `norm_spill_i = zscore(spill_i_proxy)`

Decision:
- Choose candidate with minimum `score_i`.

Defaults:
- `k=3`, `beta=2.0`, `tau=1.0` (after z-normalization), `lambda=0.5` (proxy mode).

Model-specific starting defaults:
- Qwen 3.5 35B: `k=3`, `beta=1.8`, `tau=1.2`
- Qwen 3.5 9B: `k=3`, `beta=2.2`, `tau=0.8`

Adaptive compute policy:
- Run MSS lookahead only when uncertainty is high (`top1_prob < 0.92` or entropy above threshold).
- Otherwise use greedy token pick to cap latency overhead.
- Add `adaptive_k`: start with `k=1`, escalate to `k=3` only if uncertainty/spill gates trigger.

Termination and failure handling:
- Include EOS in candidate set and score it with the same function.
- If all candidates exceed a high-spill guardrail (`norm_spill > tau + panic_margin`), trigger a "logical panic" fallback:
  1. retry with larger `k` once (`k=5`), then
  2. fall back to standard greedy for the step with a diagnostic flag.

## 4) System Architecture

### Components
- `mss/decoder.py`: token-step logic and scoring.
- `mss/backend.py`: unified interface for local endpoints.
- `mss/eval_harness.py`: benchmark runner and metric aggregation.
- `configs/mss_*.yaml`: model-specific hyperparameters.
- `reports/`: experiment outputs and plots.

### Backend interface
- `get_step_logits(context_ids, top_k, return_raw_logits=True)`
- `batch_get_next_logits(list_of_context_ids, return_raw_logits=True)`

### Endpoint requirements
- Must support top-`k` candidate extraction at current step.
- Should support batched forward pass for candidate continuations.
- Prefer raw logits; if not available, run proxy spill mode.

Supported backend modes:
- Native local runtime path with raw-logit access.
- OpenAI-compatible path (`logprobs`) with proxy spill mode when raw logits are absent.

## 5) Implementation Plan (Phased)

### Phase 0: Spec Freeze (0.5 day)
- Lock the scoring formula and fallback behavior.
- Define exact metrics and acceptance thresholds.
- Deliverable: `docs/mss_spec.md`.

### Phase 1: Endpoint Capability Audit (0.5 day)
- Verify whether current local endpoints expose raw logits and batching.
- Implement adapter for active backend (likely OpenAI-compatible wrapper over GGUF runtime).
- Validate EOS token id mapping and tokenizer round-trip behavior.
- Deliverable: passing backend smoke tests.

### Phase 2: MSS Decoder MVP (1 day)
- Implement top-`k` extraction + batched one-step lookahead.
- Implement raw-logit spill and proxy spill paths.
- Implement adaptive compute gating (`always_on` and `uncertainty_only` modes).
- Add deterministic mode (`temperature=0`, fixed seed where supported).
- Deliverable: `mss/decoder.py` with unit tests.

### Phase 3: Evaluation Harness (1 day)
- Build three stress suites:
1. `dead_person_bio` (alive/deceased disambiguation, date-pinned labels).
2. `multi_digit_math` (generated arithmetic with exact-match scoring).
3. `entity_swap_recent` (time-sensitive entity-role facts with snapshot date).
- Implement baselines: greedy, top-p, and MSS.
- Deliverable: reproducible benchmark script + JSONL outputs.

### Phase 4: Hyperparameter Tuning (1 day)
- Grid search over `k`, `beta`, `tau` per model (Qwen 3.5 35B and 9B).
- Optimize for error reduction under latency constraints.
- Calibrate `tau` from empirical spill distributions (e.g., 95th percentile of "correct-token" spill on calibration set).
- Deliverable: tuned config files and Pareto table.

### Phase 5: Ablation + Report (0.5-1 day)
- Compare:
1. MSS raw-logit vs MSS proxy.
2. With vs without spill normalization.
3. `k=2/3/5` cost-quality tradeoff.
- Deliverable: `reports/mss_findings.md` with go/no-go recommendation.

## 6) Metrics and Acceptance

### Quality
- Exact-match accuracy per suite.
- Hallucination/error rate (task-specific rubric).
- Consistency under paraphrased prompts.

### Efficiency
- Time-to-first-token (TTFT) delta versus greedy.
- Mean latency/token and P95 latency.
- Tokens/sec throughput.
- Additional forward-pass cost factor.
- MSS invocation rate under adaptive compute policy.

### Safety/robustness
- Refusal/regression rate on harmless prompts.
- Output length drift versus baseline.

Go/no-go:
- Ship MSS as optional decoding mode only if all primary success targets are met.

## 7) Risks and Mitigations

- **Risk:** Raw logits unavailable in production endpoint.
  - **Mitigation:** Implement proxy mode; prioritize backend patch for raw logits.
- **Risk:** MSS needs server features not exposed by stock `llama-server` (raw logits, branch/commit cache ops, per-step telemetry).
  - **Mitigation:** Keep a minimal MSS-specific fork/patchset only if profiling shows API limits are the bottleneck; upstream patches where possible to reduce long-term maintenance.
- **Risk:** Spill metric is noisy across contexts.
  - **Mitigation:** Robust per-step normalization and threshold tuning.
- **Risk:** MSS over-penalizes fluent tokens and hurts style quality.
  - **Mitigation:** Calibrate `beta/tau`; add fluency guardrail metric.
- **Risk:** Time-sensitive dataset labels become stale.
  - **Mitigation:** Pin dataset snapshot date in metadata and rerun label checks periodically.

## 8) Immediate Next Actions

1. Confirm endpoint capabilities (raw logits + batching) for each target model runtime.
2. Implement decoder MVP with dual scoring paths (raw/proxy).
3. Stand up the three-suite benchmark and run first baseline vs MSS comparison.
4. Decide `llama-server` strategy after profiling: upstream-only, minimal local patchset, or maintained fork.
