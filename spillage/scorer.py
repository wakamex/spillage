"""Energy-spill scoring for Min-Spill Search.

All functions are pure numpy — no I/O, no backend dependency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Raw spill (requires full logits)
# ---------------------------------------------------------------------------

def compute_spill_raw(candidate_logit: float, log_z_next: float) -> float:
    """ΔE = log_z_next − candidate_logit  (next-state definition).

    A high value means committing to *candidate* produces a successor state
    whose partition function is much larger than the candidate's own logit —
    i.e. the model becomes "confused" after this token.
    """
    return log_z_next - candidate_logit


# ---------------------------------------------------------------------------
# Proxy spill (when only top-N probs are available)
# ---------------------------------------------------------------------------

def compute_spill_proxy(
    entropy: float,
    top1_margin: float,
    lambda_: float = 0.5,
) -> float:
    """Approximate spill from successor distribution statistics.

    Uses entropy (flatness) plus a penalty for low top-1 margin, weighted by
    *lambda_*.
    """
    return entropy + lambda_ * (1.0 - top1_margin)


# ---------------------------------------------------------------------------
# Per-step normalisation
# ---------------------------------------------------------------------------

def robust_zscore(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Z-score using median and MAD — robust to outliers.

    With only k=3 samples this is noisy, but still useful to make τ
    scale-invariant.  Falls back gracefully: if all values are equal the
    result is all zeros.
    """
    values = np.asarray(values, dtype=np.float64)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) + 1e-8
    return (values - med) / mad


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_thresholded(
    surprisal: float,
    norm_spill: float,
    beta: float = 2.0,
    tau: float = 1.0,
) -> float:
    """Primary MSS score.  **Lower is better.**

    Penalises the candidate only if its normalised spill exceeds τ.
    """
    return surprisal + beta * max(0.0, norm_spill - tau)


def score_raw(
    surprisal: float,
    spill: float,
    alpha: float = 1.0,
    beta: float = 1.5,
) -> float:
    """Variant-A ablation score (no threshold).  **Lower is better.**"""
    return alpha * surprisal + beta * spill
