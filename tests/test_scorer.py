"""Tests for spillage.scorer — pure math, no server needed."""

from __future__ import annotations

import numpy as np
import pytest

from spillage.scorer import (
    compute_spill_proxy,
    compute_spill_raw,
    robust_zscore,
    score_raw,
    score_thresholded,
)


class TestComputeSpillRaw:
    def test_basic(self):
        # log_z_next=5.0, candidate_logit=3.0 → spill=2.0
        assert compute_spill_raw(3.0, 5.0) == pytest.approx(2.0)

    def test_negative_spill(self):
        # Candidate logit higher than log_z_next → negative spill (stable).
        assert compute_spill_raw(6.0, 4.0) == pytest.approx(-2.0)

    def test_zero_spill(self):
        assert compute_spill_raw(4.0, 4.0) == pytest.approx(0.0)


class TestComputeSpillProxy:
    def test_high_entropy_low_margin(self):
        # High entropy (3.0) + low margin (0.1) → high proxy spill.
        result = compute_spill_proxy(3.0, 0.1, lambda_=0.5)
        assert result == pytest.approx(3.0 + 0.5 * 0.9)

    def test_low_entropy_high_margin(self):
        # Low entropy (0.5) + high margin (0.9) → low proxy spill.
        result = compute_spill_proxy(0.5, 0.9, lambda_=0.5)
        assert result == pytest.approx(0.5 + 0.5 * 0.1)

    def test_lambda_zero_ignores_margin(self):
        result = compute_spill_proxy(2.0, 0.1, lambda_=0.0)
        assert result == pytest.approx(2.0)


class TestRobustZscore:
    def test_basic_three_values(self):
        vals = np.array([1.0, 2.0, 5.0])
        z = robust_zscore(vals)
        # median=2.0, deviations=[1,0,3], MAD=1.0
        assert z[0] == pytest.approx(-1.0, abs=1e-6)
        assert z[1] == pytest.approx(0.0, abs=1e-6)
        assert z[2] == pytest.approx(3.0, abs=1e-6)

    def test_equal_values_returns_zeros(self):
        vals = np.array([3.0, 3.0, 3.0])
        z = robust_zscore(vals)
        # All equal → all deviations zero → MAD ≈ 1e-8 → z ≈ 0.
        np.testing.assert_allclose(z, 0.0, atol=1e-4)

    def test_two_values(self):
        vals = np.array([1.0, 5.0])
        z = robust_zscore(vals)
        # median=3.0, deviations=[2,2], MAD=2.0
        assert z[0] == pytest.approx(-1.0, abs=1e-6)
        assert z[1] == pytest.approx(1.0, abs=1e-6)

    def test_single_value(self):
        z = robust_zscore(np.array([7.0]))
        assert z[0] == pytest.approx(0.0, abs=1e-4)


class TestScoreThresholded:
    def test_below_threshold_no_penalty(self):
        # norm_spill=0.5, tau=1.0 → no penalty, score = surprisal only.
        assert score_thresholded(2.0, 0.5, beta=2.0, tau=1.0) == pytest.approx(2.0)

    def test_above_threshold_penalised(self):
        # norm_spill=3.0, tau=1.0 → penalty = 2.0 * (3.0 - 1.0) = 4.0
        assert score_thresholded(2.0, 3.0, beta=2.0, tau=1.0) == pytest.approx(6.0)

    def test_at_threshold_boundary(self):
        assert score_thresholded(1.0, 1.0, beta=2.0, tau=1.0) == pytest.approx(1.0)


class TestScoreRaw:
    def test_basic(self):
        assert score_raw(2.0, 3.0, alpha=1.0, beta=1.5) == pytest.approx(6.5)

    def test_alpha_scales_surprisal(self):
        assert score_raw(2.0, 0.0, alpha=3.0, beta=1.0) == pytest.approx(6.0)


class TestVariantAvsB:
    """Variants A and B can produce different rankings on the same input.

    The thresholded variant absorbs moderate spill differences (that's the
    point of τ), so it may pick a different winner than variant A which
    penalises all spill.
    """

    def test_raw_penalises_high_spill(self):
        # Candidate A: low surprisal, high spill.
        # Candidate B: high surprisal, low spill.
        score_a = score_raw(0.5, 5.0)   # 0.5 + 1.5*5.0 = 8.0
        score_b = score_raw(3.0, 0.5)   # 3.0 + 1.5*0.5 = 3.75
        # Variant A picks B (lower score) because it penalises all spill.
        assert score_b < score_a

    def test_thresholded_absorbs_moderate_spill(self):
        # With z-normalization of [5.0, 0.5], norm values are [1.0, -1.0].
        # At tau=1.0, norm_spill=1.0 is exactly at boundary → zero penalty.
        # So the thresholded variant picks on surprisal alone.
        norms = robust_zscore(np.array([5.0, 0.5]))
        score_a = score_thresholded(0.5, float(norms[0]))  # low surprisal
        score_b = score_thresholded(3.0, float(norms[1]))   # high surprisal
        # Thresholded picks A (lower surprisal wins when spill is absorbed).
        assert score_a < score_b

    def test_variants_can_disagree(self):
        # This is the key insight: variant A and thresholded can pick
        # different winners depending on the surprisal/spill tradeoff.
        raw_prefers_b = score_raw(0.5, 5.0) > score_raw(3.0, 0.5)
        norms = robust_zscore(np.array([5.0, 0.5]))
        thresh_prefers_a = (
            score_thresholded(0.5, float(norms[0]))
            < score_thresholded(3.0, float(norms[1]))
        )
        assert raw_prefers_b and thresh_prefers_a  # they disagree


class TestScoringEquivalence:
    """Tests adapted from Gemini's implementation verifying core scoring invariants.

    These validate that equal spills produce zero-penalty scores (surprisal-only)
    and that high-spill candidates are penalized relative to low-spill ones.
    """

    def test_equal_spills_no_penalty(self):
        # When all candidates have equal raw spill, robust_zscore → 0 for all.
        # With norm_spill=0 and tau=0.5, penalty = beta * max(0, 0 - 0.5) = 0.
        # Scores should equal surprisal (-log_prob) for each candidate.
        log_probs = np.array([-0.105, -2.302])
        spills = np.array([10.0, 10.0])

        norm_spills = robust_zscore(spills)
        scores = np.array([
            score_thresholded(-log_probs[i], float(norm_spills[i]), beta=2.0, tau=0.5)
            for i in range(len(log_probs))
        ])

        # Penalties should be zero; scores equal to surprisal.
        np.testing.assert_allclose(scores, -log_probs, atol=1e-6)

    def test_high_spill_candidate_penalised(self):
        # Candidate 0: high log-prob, high spill.
        # Candidate 1: low log-prob, low spill.
        # With tau=0.0, the high-spill candidate should score worse.
        log_probs = np.array([-0.1, -1.0])
        spills = np.array([20.0, 10.0])

        norm_spills = robust_zscore(spills)
        scores = np.array([
            score_thresholded(-log_probs[i], float(norm_spills[i]), beta=2.0, tau=0.0)
            for i in range(len(log_probs))
        ])

        # High-spill candidate (0) should have a worse (higher) score.
        assert scores[0] > scores[1]
