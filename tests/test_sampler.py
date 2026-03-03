"""Tests for the MSS generation loop."""

from __future__ import annotations

import numpy as np
import pytest

from spillage.backend_mock import MockBackend
from spillage.config import MSSConfig
from spillage.sampler import (
    TokenEvent,
    generate,
    generate_greedy,
)


def _make_eos_backend() -> MockBackend:
    """Backend where context [1] → top token is EOS (0)."""
    vocab_size = 8
    logits = np.zeros(vocab_size, dtype=np.float64)
    logits[0] = 10.0  # EOS is dominant
    return MockBackend(
        vocab_size=vocab_size,
        logit_table={(1,): logits},
        eos_id=0,
    )


class TestGenerateGreedy:
    def test_produces_output(self, mock_backend: MockBackend):
        result = generate_greedy("Hello", mock_backend, max_tokens=5)
        assert len(result.token_ids) > 0
        assert len(result.text) > 0
        assert len(result.events) == len(result.token_ids)

    def test_stops_at_eos(self):
        backend = _make_eos_backend()
        result = generate_greedy("Hello", backend, max_tokens=100)
        # Should stop after one token (the EOS).
        assert result.token_ids[-1] == 0
        assert len(result.token_ids) <= 2  # at most 1-2 tokens

    def test_respects_max_tokens(self, mock_backend: MockBackend):
        result = generate_greedy("Hello", mock_backend, max_tokens=3)
        assert len(result.token_ids) <= 3

    def test_events_have_fast_path_true(self, mock_backend: MockBackend):
        result = generate_greedy("Hello", mock_backend, max_tokens=3)
        for event in result.events:
            assert event.fast_path is True


class TestGenerateMSS:
    def test_produces_output(self, mock_backend: MockBackend):
        cfg = MSSConfig(max_tokens=5, adaptive=False)
        result = generate("Hello", mock_backend, config=cfg)
        assert len(result.token_ids) > 0
        assert len(result.text) > 0

    def test_stops_at_eos(self):
        backend = _make_eos_backend()
        cfg = MSSConfig(max_tokens=100, adaptive=False)
        result = generate("Hello", backend, config=cfg)
        assert result.token_ids[-1] == 0

    def test_respects_max_tokens(self, mock_backend: MockBackend):
        cfg = MSSConfig(max_tokens=3, adaptive=False)
        result = generate("Hello", mock_backend, config=cfg)
        assert len(result.token_ids) <= 3

    def test_events_populated(self, mock_backend: MockBackend):
        cfg = MSSConfig(max_tokens=3, adaptive=False)
        result = generate("Hello", mock_backend, config=cfg)
        assert len(result.events) == len(result.token_ids)
        for event in result.events:
            assert len(event.candidates) > 0
            assert 0 <= event.selected < len(event.candidates)

    def test_candidate_scores_populated(self, mock_backend: MockBackend):
        cfg = MSSConfig(max_tokens=2, adaptive=False)
        result = generate("Hello", mock_backend, config=cfg)
        for event in result.events:
            for cand in event.candidates:
                assert cand.token_text != ""
                assert isinstance(cand.score, float)


class TestAdaptiveGating:
    def test_fast_path_when_confident(self):
        """When top-1 prob > threshold, MSS should skip lookahead."""
        vocab_size = 8
        # One dominant token → high confidence.
        logits = np.zeros(vocab_size, dtype=np.float64)
        logits[3] = 20.0  # overwhelming favorite
        backend = MockBackend(
            vocab_size=vocab_size,
            logit_table={(1,): logits},
        )
        cfg = MSSConfig(
            max_tokens=1,
            adaptive=True,
            confidence_threshold=0.9,
        )
        result = generate("Hello", backend, config=cfg)
        assert result.events[0].fast_path is True

    def test_slow_path_when_uncertain(self, mock_backend: MockBackend):
        """Default mock has near-uniform logits → should use lookahead."""
        cfg = MSSConfig(
            max_tokens=1,
            adaptive=True,
            confidence_threshold=0.99,
        )
        result = generate("Hello", mock_backend, config=cfg)
        assert result.events[0].fast_path is False
        assert len(result.events[0].candidates) >= 2


class TestMSSReranking:
    def test_mss_can_override_greedy(self, high_spill_backend: MockBackend):
        """MSS should pick the low-spill token even if it's not top-1."""
        # High beta + low tau: aggressively penalise spill so the penalty
        # can overcome the ~2-point surprisal gap between tokens 5 and 3.
        cfg = MSSConfig(
            max_tokens=1,
            adaptive=False,
            k=3,
            beta=5.0,
            tau=0.0,
        )
        # Greedy would pick token 5 (highest logit).
        greedy = generate_greedy("Hello", high_spill_backend, max_tokens=1)

        # MSS should prefer token 3 (lower spill).
        mss = generate("Hello", high_spill_backend, config=cfg)

        # Verify greedy picks the high-prob token.
        assert greedy.token_ids[0] == 5  # "Elon"

        # MSS should pick a different token (token 3, "Jeff") due to spill.
        assert mss.token_ids[0] != greedy.token_ids[0]
        assert mss.token_ids[0] == 3  # "Jeff"

    def test_variant_raw_also_works(self, high_spill_backend: MockBackend):
        """Variant A (raw) should also penalise high-spill candidates."""
        cfg = MSSConfig(max_tokens=1, adaptive=False, k=3, beta=3.0)
        result = generate(
            "Hello", high_spill_backend,
            config=cfg, variant="raw",
        )
        # Should still avoid token 5 (high spill).
        assert result.token_ids[0] == 3


class TestCallbackAndTiming:
    def test_on_token_called(self, mock_backend: MockBackend):
        events_received: list[TokenEvent] = []
        cfg = MSSConfig(max_tokens=3, adaptive=False)
        generate(
            "Hello", mock_backend, config=cfg,
            on_token=events_received.append,
        )
        assert len(events_received) == 3

    def test_total_time_positive(self, mock_backend: MockBackend):
        cfg = MSSConfig(max_tokens=2, adaptive=False)
        result = generate("Hello", mock_backend, config=cfg)
        assert result.total_time_ms > 0

    def test_per_step_time_positive(self, mock_backend: MockBackend):
        cfg = MSSConfig(max_tokens=2, adaptive=False)
        result = generate("Hello", mock_backend, config=cfg)
        for event in result.events:
            assert event.wall_time_ms >= 0


class TestProxyMode:
    def test_generates_with_proxy_backend(self, mock_backend_proxy: MockBackend):
        cfg = MSSConfig(max_tokens=3, adaptive=False)
        result = generate("Hello", mock_backend_proxy, config=cfg)
        assert len(result.token_ids) == 3
        for event in result.events:
            assert len(event.candidates) > 0
