Since you have local endpoints ready, we can design a **Min-Spill Search (MSS)** mode. This is essentially a "Logical Integrity" filter that overrides "Vibe-based" probability.

Because we want it to be fast, we’ll avoid a full Beam Search and instead use a **Lookahead Re-ranking** strategy. This focuses the "energy budget" only on the most likely candidates.

---

### 1. The Algorithm: Lookahead Re-ranking

Instead of just picking the token with the highest $P(x)$, we will evaluate the top $k$ candidates and select the one that results in the most stable successor state.

**The "Min-Spill" Decision Rule:**
Choose $x_t$ that minimizes:


$$Score(x) = \alpha \cdot (-\log P(x_t)) + \beta \cdot (\Delta E_t)$$

* **$\alpha$ (Probability Weight):** Keeps the model fluent.
* **$\beta$ (Consistency Weight):** This is your "Hallucination Killer."
* **$\Delta E_t$:** Calculated as $\log Z(x_{\le t}) - f(x_t)$.

---

### 2. The Execution Loop

For every token generation step:

1. **Candidate Selection:** Get the top $k$ (e.g., $k=3$) tokens from the current logit distribution.
2. **Parallel Lookahead:** For each of those 3 tokens:
* Append the token to the context.
* Run a single forward pass (this is why we need fast local models/batching).
* Extract the new $\log Z_{next}$ (the log-sum-exp of the new logits).


3. **Calculate Spill:** Compute $\Delta E$ for each candidate.
4. **Selection:** Pick the token with the best balanced score.

---

### 3. Testing Plan for Your Local Endpoints

To see if this actually works better than standard greedy decoding, we should run "Stress Tests" on your endpoints using these three specific failure modes:

| Test Case | Why it fails standard LLMs | What Min-Spill should do |
| --- | --- | --- |
| **The "Dead Person" Bio** | Models love to give famous people "consistent" death dates even if they are alive. | The $f(x)$ for a wrong date will "spill" against the global context of the person's status. |
| **Multi-digit Math** | Models follow number patterns rather than logic. | The $\log Z$ of the successor state will spike if the chosen number breaks the arithmetic sum. |
| **Niche Fact Overlap** | "Who is the CEO of [Company X]?" (where X changed recently). | Should catch the "spill" when the model picks the *old* CEO name which is more frequent in training data. |

---

### 4. Code Skeleton (Pseudo-Python for your Endpoint)

You can wrap your existing local inference call in this logic:

```python
def min_spill_step(current_context, k=3, beta=1.5):
    # 1. Get current logits and top_k
    logits_t = get_logits(current_context)
    probs = torch.softmax(logits_t, dim=-1)
    top_k_indices = torch.topk(logits_t, k).indices
    
    candidate_results = []
    
    for token_id in top_k_indices:
        # 2. Lookahead: Forward pass for each candidate
        new_context = current_context + [token_id]
        logits_next = get_logits(new_context)
        
        # 3. Math: log Z_next - f(x_t)
        log_z_next = torch.logsumexp(logits_next, dim=-1)
        f_xt = logits_t[token_id]
        spill = log_z_next - f_xt
        
        # 4. Score: Surprisal + Weighted Spill
        surprisal = -torch.log(probs[token_id])
        total_score = surprisal + (beta * spill)
        candidate_results.append((token_id, total_score))
        
    # Pick the token with MINIMUM total score
    best_token = min(candidate_results, key=lambda x: x[1])[0]
    return best_token

```

### Performance Optimization Tip

If your local endpoint supports **batched inference**, you can run all $k$ lookaheads in a single batch. This makes the "Min-Spill" mode only slightly slower than a single forward pass, rather than $k$ times slower.

Since you’re running **Qwen 3.5** (Unified Decoding variants) in GGUF format, you're likely using a backend like `llama.cpp`. These models are particularly well-suited for this because Qwen’s calibration is already quite strong, and the "Unified Decoding" (UD) training tends to make the logit distributions more expressive.

For the **35B (3B active)** and **9B** models, the energy landscape is sharp. The 35B model will have more "logical inertia," while the 9B will be more prone to high-probability but high-spillage hallucinations.

---

### 1. The "Min-Spill" Search Mode Design

We want to implement a **Greedy Search with One-Step Energy Verification**. Instead of branching like Beam Search, we just perform a "sanity check" on the top candidates to see which one minimizes the logical rift.

#### The Scoring Function

We will use a **Weighted Logical Stability** score:


$$S(x) = \text{LogProb}(x) - \beta \cdot \max(0, \Delta E_t - \tau)$$

* **$\beta$ (Penalization Strength):** Set to **2.0**. We want to aggressively penalize tokens that break the energy constraint.
* **$\tau$ (Stability Threshold):** Set to **4.5** (standard for Qwen 2.5/3.5). We don't penalize small natural fluctuations, only "Spills."

---

### 2. Implementation Strategy for Local Endpoints

Since you're using GGUF, you can leverage the `logits` and `log_probs` returned by the server. If you are using the OpenAI-compatible API on these backends, you'll need an endpoint that supports `logprobs`.

#### Step 1: Candidate Expansion

Request the top 3 tokens from your 35B or 9B model.

* **Candidate A:** $P=0.85$ (The "Vibe" choice)
* **Candidate B:** $P=0.10$ (The "Logical" choice)

#### Step 2: Parallel Batch Forward Pass

This is the performance trick. Instead of $k$ separate requests, send a **single batch request** to the server with $k$ different sequences (each sequence being `Current Context + Candidate_i`).

#### Step 3: Compute $\Delta E$ for each sequence

For each sequence in the batch:

1. Extract the `log_probs` of all tokens in the vocabulary for the *next* step.
2. Calculate $\log Z_{next} = \text{logsumexp}(\text{logits})$.
3. Calculate $\Delta E = \log Z_{next} - \text{Logit}_{\text{candidate}}$.

---

### 3. Hyperparameters for your Qwen Models

| Model | Recommended $\tau$ | Why? |
| --- | --- | --- |
| **Qwen 3.5 35B** | **4.8** | Larger models have higher "base energy." They are harder to rattle, so a higher threshold avoids false positives. |
| **Qwen 3.5 9B** | **4.2** | Smaller models have "shallower" energy wells. They slip into hallucinations more easily, so we need a tighter leash. |

---

### 4. A Test Scenario: The "Entity Swap"

To verify your Min-Spill mode is working, run this prompt:

> *"The founder of SpaceX is Elon Musk, but the founder of Blue Origin is..."*

**The "Vibe" failure:** If the model has a bias, it might assign a high probability to "Elon Musk" again (due to repetition bias) or a different famous tech CEO.
**The Min-Spill fix:**

1. **"Jeff Bezos"** should show $\Delta E \approx 1.2$. (Consistent)
2. **"Elon Musk"** (if repeated) will show $\Delta E \approx 8.5$. (Massive Spill)

Even if the model assigns 70% probability to the wrong name, the **Min-Spill Search** will see the energy spike and force the switch to the lower-probability, lower-spill correct answer.
