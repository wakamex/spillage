import numpy as np
import asyncio
from typing import AsyncGenerator, List, Optional
from .backend import Backend, LogitResult
from .scorer import RobustScorer, MSSScoringConfig

def softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax implementation."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def calculate_entropy(probs: np.ndarray) -> float:
    """Calculate the Shannon entropy of a probability distribution."""
    return -np.sum(probs * np.log(probs + 1e-10))

class MSSSampler:
    """The Platinum Standard MSS Sampler with Adaptive Gating."""
    
    def __init__(self, 
                 backend: Backend, 
                 k: int = 3, 
                 uncertainty_threshold: float = 0.92,
                 config: MSSScoringConfig = MSSScoringConfig()):
        self.backend = backend
        self.k = k
        self.uncertainty_threshold = uncertainty_threshold
        self.scorer = RobustScorer(config)

    async def calibrate_tau(self, calibration_prompts: List[str], percentile: float = 95.0) -> float:
        """
        Dynamically calibrates tau by measuring the model's 'resting' energy spill.
        Sets tau to the Nth percentile of observed spills on neutral text.
        """
        all_spills = []
        print(f"Calibrating Tau over {len(calibration_prompts)} prompts...")
        
        for prompt in calibration_prompts:
            res = await self.backend.get_logits(prompt, top_k=self.k)
            top_k_indices = np.argsort(res.logits)[-self.k:][::-1]
            top_k_logits = res.logits[top_k_indices]
            
            # Simple 1-step lookahead for all k candidates to gather spill distribution
            candidate_texts = []
            for cid in [int(res.token_ids[i]) for i in range(len(top_k_indices))]:
                candidate_texts.append(await self.backend.detokenize([cid]))
            
            lookahead_prompts = [prompt + text for text in candidate_texts]
            lookahead_results = await self.backend.get_logits_batch(lookahead_prompts, top_k=1)
            
            step_spills = [
                self.scorer.compute_spill(top_k_logits[i], lookahead_results[i].log_z)
                for i in range(self.k)
            ]
            all_spills.extend(step_spills)
            
        # Set tau to the requested percentile of normalized spills
        from .scorer import robust_zscore
        norm_spills = robust_zscore(np.array(all_spills))
        new_tau = float(np.percentile(norm_spills, percentile))
        
        print(f"Calibration Complete. New Tau: {new_tau:.4f} (at {percentile}th percentile)")
        self.scorer.config.tau = new_tau
        return new_tau

    async def generate(self, 
                       prompt: str, 
                       max_tokens: int = 128) -> AsyncGenerator[str, None]:
        """Generates tokens using Min-Spill Search with Adaptive Gating."""
        current_prompt = prompt
        
        for _ in range(max_tokens):
            # 1. Get current logits
            res = await self.backend.get_logits(current_prompt, top_k=self.k)
            probs = softmax(res.logits)
            top_k_indices = np.argsort(res.logits)[-self.k:][::-1]
            top_k_logits = res.logits[top_k_indices]
            top_k_probs = probs[top_k_indices]
            top1_prob = top_k_probs[0]
            
            # DEBUG
            # print(f"Top1 Prob: {top1_prob:.4f}")

            # 2. Adaptive Gating: Only run MSS if the model is uncertain
            if top1_prob >= self.uncertainty_threshold:
                # Fast-Path: Greedy pick
                winner_idx = top_k_indices[0]
                selected_token_id = int(res.token_ids[winner_idx])
                selected_text = await self.backend.detokenize([selected_token_id])
                # print(f"FAST: '{selected_text}'")
            else:
                # MSS Slow-Path: Lookahead Re-ranking
                candidate_ids = [int(res.token_ids[i]) for i in range(len(top_k_indices))]
                candidate_texts = []
                for cid in candidate_ids:
                    text = await self.backend.detokenize([cid])
                    candidate_texts.append(text)
                
                # print(f"SLOW Candidates: {candidate_texts}")
                
                # Parallel Lookahead
                lookahead_prompts = [current_prompt + text for text in candidate_texts]
                lookahead_results = await self.backend.get_logits_batch(lookahead_prompts, top_k=1)
                
                # Check for 2-Step Lookahead (Heuristic: Capitalized token)
                # If a token starts with a capital, it might be an entity start.
                final_lookahead_results = []
                for i, text in enumerate(candidate_texts):
                    if text.strip() and text.strip()[0].isupper():
                        # Extend the lookahead by one step for potential entities
                        next_res = lookahead_results[i]
                        best_next_token_id = int(next_res.token_ids[np.argmax(next_res.logits)])
                        best_next_text = await self.backend.detokenize([best_next_token_id])
                        
                        deep_prompt = lookahead_prompts[i] + best_next_text
                        deep_res = await self.backend.get_logits(deep_prompt, top_k=1)
                        final_lookahead_results.append(deep_res)
                    else:
                        final_lookahead_results.append(lookahead_results[i])
                
                # Calculate Spills and Scores
                spills = np.array([
                    self.scorer.compute_spill(top_k_logits[i], final_lookahead_results[i].log_z)
                    for i in range(self.k)
                ])
                
                # Re-rank based on MSS scores (Surprisal + Penalized Spill)
                # log_probs for candidates
                candidate_log_probs = np.log(top_k_probs + 1e-10)
                scores = self.scorer.score_candidates(candidate_log_probs, spills)
                
                # Select token with MINIMUM score
                winner_idx = np.argmin(scores)
                selected_token_id = candidate_ids[winner_idx]
                selected_text = candidate_texts[winner_idx]

            # End of generation check
            if selected_text == "": # Simple EOS check for prototype
                break
                
            current_prompt += selected_text
            yield selected_text
