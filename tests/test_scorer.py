import numpy as np
from spillage.scorer import RobustScorer, MSSScoringConfig

def test_robust_scorer_math():
    config = MSSScoringConfig(beta=2.0, tau=0.5)
    scorer = RobustScorer(config)
    
    # 1. Test case: No spill (spills are all equal, so normalized spill is 0)
    # log_probs: [0.9, 0.1] -> [-0.105, -2.302]
    # spills: [10, 10]
    log_probs = np.array([-0.105, -2.302])
    spills = np.array([10.0, 10.0])
    
    scores = scorer.score_candidates(log_probs, spills)
    
    # Normalized spills should be 0, so penalties should be 0.
    # Scores should be equal to surprisal (-log_probs)
    assert np.allclose(scores, -log_probs)

def test_robust_scorer_penalty():
    # Set tau to 0.0 to trigger penalties easily
    config = MSSScoringConfig(beta=2.0, tau=0.0)
    scorer = RobustScorer(config)
    
    # Candidate 0: High prob, High spill
    # Candidate 1: Low prob, Low spill
    log_probs = np.array([-0.1, -1.0])
    spills = np.array([20.0, 10.0])
    
    scores = scorer.score_candidates(log_probs, spills)
    
    # Candidate 0 should have a much higher score (worse) because of the penalty
    assert scores[0] > scores[1]

if __name__ == "__main__":
    test_robust_scorer_math()
    test_robust_scorer_penalty()
    print("Scorer tests passed!")
