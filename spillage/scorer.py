import numpy as np
from dataclasses import dataclass

def robust_zscore(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Calculates the Robust Z-Score using Median Absolute Deviation (MAD)."""
    if len(values) == 0:
        return np.array([])
    median = np.median(values)
    abs_diff = np.abs(values - median)
    mad = np.median(abs_diff)
    return (values - median) / (mad + epsilon)

@dataclass
class MSSScoringConfig:
    beta: float = 2.0
    tau: float = 4.2  # Default for Qwen 3.5 9B
    alpha: float = 1.0

class RobustScorer:
    """Implements the Min-Spill Search (MSS) Platinum scoring logic."""
    
    def __init__(self, config: MSSScoringConfig = MSSScoringConfig()):
        self.config = config

    def compute_spill(self, candidate_logit: float, log_z_next: float) -> float:
        """
        ΔE = log Z_next - Logit(x_t)
        Higher values indicate more 'spillage' (logical disruption).
        """
        return log_z_next - candidate_logit

    def score_candidates(self, log_probs: np.ndarray, spills: np.ndarray) -> np.ndarray:
        """
        Calculates the final MSS scores for a set of candidates.
        
        Formula: S(x_t) = -log P(x_t) + beta * max(0, norm_spill_i - tau)
        Note: Lower scores are better.
        """
        if len(spills) == 0:
            return np.array([])
            
        norm_spills = robust_zscore(spills)
        surprisal = -log_probs
        
        # We only penalize if normalized spill exceeds the threshold tau
        penalties = self.config.beta * np.maximum(0, norm_spills - self.config.tau)
        total_scores = (self.config.alpha * surprisal) + penalties
        
        return total_scores
