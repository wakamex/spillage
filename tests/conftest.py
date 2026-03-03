"""Shared fixtures for spillage tests."""

from __future__ import annotations

import numpy as np
import pytest

from spillage.backend_mock import MockBackend


@pytest.fixture
def mock_backend() -> MockBackend:
    """A MockBackend with a small vocab and no custom logit table."""
    return MockBackend(vocab_size=16)


@pytest.fixture
def mock_backend_proxy() -> MockBackend:
    """A MockBackend in proxy mode."""
    return MockBackend(vocab_size=16, proxy_mode=True)


@pytest.fixture
def high_spill_backend() -> MockBackend:
    """Backend where context [1] → token 3 has low spill but token 5 has high spill.

    Simulates: after choosing token 3 (e.g. "Jeff") the model is confident
    (low log_z_next), but after choosing token 5 (e.g. "Elon") the model
    is confused (high log_z_next).
    """
    vocab_size = 16

    # Current context [1]: token 3 and 5 are the top candidates.
    current_logits = np.zeros(vocab_size, dtype=np.float64)
    current_logits[5] = 5.0   # "Elon" — high probability (the "vibe" choice)
    current_logits[3] = 3.0   # "Jeff" — lower probability (the "correct" choice)

    # After choosing token 3 ("Jeff"): model is confident, low-magnitude.
    # log_z ≈ 3.87, spill = 3.87 - 3.0 = 0.87
    after_3 = np.zeros(vocab_size, dtype=np.float64)
    after_3[4] = 3.5  # " Bezos" mildly dominant

    # After choosing token 5 ("Elon"): model is confused, high spread.
    # log_z = 6 + ln(16) ≈ 8.77, spill = 8.77 - 5.0 = 3.77
    after_5 = np.full(vocab_size, 6.0, dtype=np.float64)

    table = {
        (1,): current_logits,
        (1, 3): after_3,
        (1, 5): after_5,
    }
    return MockBackend(vocab_size=vocab_size, logit_table=table)
