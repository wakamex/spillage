"""Tests for MockBackend — verifies the Backend protocol contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spillage.backend import Backend
from spillage.backend_mock import MockBackend


class TestMockBackendRaw:
    def test_implements_protocol(self, mock_backend: MockBackend):
        assert isinstance(mock_backend, Backend)

    def test_mode_is_raw(self, mock_backend: MockBackend):
        assert mock_backend.mode() == "raw"

    def test_logit_result_shapes(self, mock_backend: MockBackend):
        result = mock_backend.get_logits([1, 2, 3])
        assert result.logits is not None
        assert result.logits.shape == (16,)
        assert len(result.top_k_ids) == 5
        assert len(result.top_k_logits) == 5
        assert len(result.top_k_probs) == 5

    def test_log_z_matches_logsumexp(self, mock_backend: MockBackend):
        result = mock_backend.get_logits([1])
        expected = float(np.logaddexp.reduce(result.logits))
        assert result.log_z == pytest.approx(expected)

    def test_probs_sum_to_one(self, mock_backend: MockBackend):
        result = mock_backend.get_logits([1, 2])
        # Full probs from logits should sum to 1.
        full_probs = np.exp(result.logits - np.logaddexp.reduce(result.logits))
        assert float(np.sum(full_probs)) == pytest.approx(1.0)

    def test_top_k_sorted_descending(self, mock_backend: MockBackend):
        result = mock_backend.get_logits([1])
        # top_k_logits should be in descending order.
        for i in range(len(result.top_k_logits) - 1):
            assert result.top_k_logits[i] >= result.top_k_logits[i + 1]

    def test_proxy_fields_none_in_raw_mode(self, mock_backend: MockBackend):
        result = mock_backend.get_logits([1])
        assert result.entropy is None
        assert result.top1_margin is None

    def test_eos_token_id(self, mock_backend: MockBackend):
        assert mock_backend.eos_token_id() == 0


class TestMockBackendProxy:
    def test_mode_is_proxy(self, mock_backend_proxy: MockBackend):
        assert mock_backend_proxy.mode() == "proxy"

    def test_logits_none_in_proxy_mode(self, mock_backend_proxy: MockBackend):
        result = mock_backend_proxy.get_logits([1])
        assert result.logits is None
        assert math.isnan(result.log_z)

    def test_proxy_fields_populated(self, mock_backend_proxy: MockBackend):
        result = mock_backend_proxy.get_logits([1])
        assert result.entropy is not None
        assert result.entropy >= 0.0
        assert result.top1_margin is not None

    def test_top_k_populated(self, mock_backend_proxy: MockBackend):
        result = mock_backend_proxy.get_logits([1])
        assert len(result.top_k_ids) == 5
        assert len(result.top_k_probs) == 5


class TestMockBackendBatch:
    def test_batch_returns_list(self, mock_backend: MockBackend):
        results = mock_backend.get_logits_batch([[1], [1, 2], [1, 3]])
        assert len(results) == 3

    def test_batch_matches_individual(self, mock_backend: MockBackend):
        seqs = [[1], [1, 2]]
        batch = mock_backend.get_logits_batch(seqs)
        for seq, batch_result in zip(seqs, batch):
            single = mock_backend.get_logits(seq)
            assert batch_result.log_z == pytest.approx(single.log_z)


class TestMockBackendTokenizer:
    def test_detokenize_known(self, mock_backend: MockBackend):
        assert mock_backend.detokenize([1, 2]) == "Hello world"

    def test_tokenize_known(self, mock_backend: MockBackend):
        ids = mock_backend.tokenize("Hello world")
        assert ids == [1, 2]

    def test_round_trip(self, mock_backend: MockBackend):
        text = "Jeff Bezos"
        ids = mock_backend.tokenize(text)
        reconstructed = mock_backend.detokenize(ids)
        assert reconstructed == text


class TestHighSpillBackend:
    def test_custom_logit_table_applied(self, high_spill_backend: MockBackend):
        # Context [1] should return custom logits where token 5 > token 3.
        result = high_spill_backend.get_logits([1])
        assert result.logits is not None
        assert result.logits[5] > result.logits[3]

    def test_spill_difference(self, high_spill_backend: MockBackend):
        """Token 5 should produce higher successor log_z than token 3."""
        from spillage.scorer import compute_spill_raw

        current = high_spill_backend.get_logits([1])
        after_3 = high_spill_backend.get_logits([1, 3])
        after_5 = high_spill_backend.get_logits([1, 5])

        spill_3 = compute_spill_raw(float(current.logits[3]), after_3.log_z)
        spill_5 = compute_spill_raw(float(current.logits[5]), after_5.log_z)

        # Token 5 ("Elon") should have much higher spill than token 3 ("Jeff").
        assert spill_5 > spill_3
        assert spill_5 - spill_3 > 2.0  # meaningful gap


class TestHttpBackendProbeCapabilities:
    """Tests for HttpBackend.probe_capabilities() using a mock httpx client."""

    def _make_backend(self, responses: dict) -> "HttpBackend":
        """Build an HttpBackend whose _client is patched with fixed responses."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from spillage.backend_http import HttpBackend

        backend = HttpBackend.__new__(HttpBackend)
        backend._url = "http://localhost:8080"
        backend._n_probs = 5
        backend._eos = None

        mock_client = MagicMock()

        def _post(url: str, **kwargs):
            path = url.split("localhost:8080", 1)[-1]
            payload = kwargs.get("json", {})
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = responses.get(path, {})
            return resp

        mock_client.post.side_effect = _post
        backend._client = mock_client
        return backend

    def test_reports_all_ok(self):
        from spillage.backend_http import HttpBackend

        responses = {
            "/tokenize": {"tokens": [5, 6]},
            "/detokenize": {"content": "hello"},
            "/completion": {
                "logits": [1.0, 0.5, -0.5, 0.0, 0.2],
                "completion_probabilities": [
                    {"probs": [{"id": 0, "logprob": -0.1}, {"id": 1, "logprob": -1.2}]}
                ],
            },
        }
        backend = self._make_backend(responses)
        caps = backend.probe_capabilities("hello", top_n=2)
        assert caps.tokenize_ok is True
        assert caps.detokenize_ok is True
        assert caps.sparse_logprobs_ok is True

    def test_tokenize_failure_recorded_in_notes(self):
        responses = {"/tokenize": {}}  # missing "tokens" key → exception
        backend = self._make_backend(responses)
        # Patch tokenize to raise so the probe records the failure.
        from unittest.mock import patch
        with patch.object(backend, "tokenize", side_effect=RuntimeError("conn refused")):
            caps = backend.probe_capabilities()
        assert caps.tokenize_ok is False
        assert any("tokenize failed" in n for n in caps.notes)
