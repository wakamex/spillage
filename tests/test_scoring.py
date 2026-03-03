import math

from mss.scoring import proxy_spill, raw_spill, robust_zscores


def test_raw_spill_uses_logsumexp_minus_candidate_logit() -> None:
    next_logits = [2.0, 2.0, 2.0]
    got = raw_spill(next_logits, candidate_logit=1.0)
    expected = math.log(3.0) + 1.0
    assert abs(got - expected) < 1e-6


def test_robust_zscores_constant_values_are_zero() -> None:
    assert robust_zscores([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_proxy_spill_prefers_sharper_distributions() -> None:
    flat = [math.log(1 / 3), math.log(1 / 3), math.log(1 / 3)]
    sharp = [math.log(0.98), math.log(0.01), math.log(0.01)]
    assert proxy_spill(flat) > proxy_spill(sharp)
