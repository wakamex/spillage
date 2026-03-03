from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from mss.llama_server_backend import LlamaServerBackend, LlamaServerConfig


def test_backend_encode_decode_roundtrip() -> None:
    def request_fn(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        path = urlparse(url).path
        if path == "/tokenize":
            assert payload["content"] == "hi"
            return {"tokens": [10, 11]}
        if path == "/detokenize":
            assert payload["tokens"] == [10, 11]
            return {"content": "hi"}
        raise AssertionError(f"unexpected path: {path}")

    backend = LlamaServerBackend(LlamaServerConfig(base_url="http://localhost:8080"), request_fn=request_fn)

    ids = backend.encode("hi")
    assert ids == [10, 11]
    assert backend.decode(ids) == "hi"


def test_backend_prefers_raw_logits_when_available() -> None:
    def request_fn(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        path = urlparse(url).path
        if path != "/completion":
            raise AssertionError(f"unexpected path: {path}")
        assert payload["n_predict"] == 0
        assert payload["return_logits"] is True
        return {"logits": [3.0, 2.0, 1.0]}

    backend = LlamaServerBackend(LlamaServerConfig(base_url="http://localhost:8080"), request_fn=request_fn)
    step = backend.next_step([1, 2, 3], top_n=2)

    assert step.logits == [3.0, 2.0, 1.0]
    assert list(step.logprobs) == [0, 1]


def test_backend_sparse_native_fallback() -> None:
    calls: list[dict[str, Any]] = []

    def request_fn(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        path = urlparse(url).path
        if path != "/completion":
            raise AssertionError(f"unexpected path: {path}")
        calls.append(payload)
        if payload["n_predict"] == 0:
            # Force fallback path by omitting logits.
            return {"unexpected": True}
        return {
            "completion_probabilities": [
                {
                    "probs": [
                        {"id": 4, "logprob": -0.2},
                        {"id": 7, "logprob": -1.2},
                        {"id": 9, "prob": 0.1},
                    ]
                }
            ]
        }

    cfg = LlamaServerConfig(base_url="http://localhost:8080", prefer_raw_logits=True)
    backend = LlamaServerBackend(cfg, request_fn=request_fn)
    step = backend.next_step([1, 2], top_n=2)

    assert step.logits is None
    assert set(step.logprobs or {}) == {4, 7}
    assert len(calls) == 2


def test_backend_caches_missing_raw_logits_capability() -> None:
    calls: list[dict[str, Any]] = []

    def request_fn(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        path = urlparse(url).path
        if path != "/completion":
            raise AssertionError(f"unexpected path: {path}")
        calls.append(payload)
        if payload["n_predict"] == 0:
            # Raw logits unavailable in this backend build.
            return {"unexpected": True}
        return {
            "completion_probabilities": [
                {
                    "top_logprobs": [
                        {"id": 4, "logprob": -0.2},
                        {"id": 7, "logprob": -1.2},
                    ]
                }
            ]
        }

    cfg = LlamaServerConfig(base_url="http://localhost:8080", prefer_raw_logits=True)
    backend = LlamaServerBackend(cfg, request_fn=request_fn)

    step1 = backend.next_step([1, 2], top_n=2)
    step2 = backend.next_step([1, 2, 4], top_n=2)

    assert step1.logits is None
    assert step2.logits is None
    # First next_step does raw-logits probe + sparse fallback.
    # Second next_step skips raw-logits probe and goes straight to sparse.
    assert len(calls) == 3
    assert calls[0]["n_predict"] == 0
    assert calls[1]["n_predict"] == 1
    assert calls[2]["n_predict"] == 1


def test_backend_sparse_native_new_completion_probabilities_shape() -> None:
    def request_fn(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        path = urlparse(url).path
        if path != "/completion":
            raise AssertionError(f"unexpected path: {path}")
        if payload["n_predict"] == 0:
            # Force fallback path by omitting logits.
            return {"unexpected": True}
        return {
            "completion_probabilities": [
                {
                    "id": 11,
                    "logprob": -1.0,
                    "top_logprobs": [
                        {"id": 11, "logprob": -1.0},
                        {"id": 0, "logprob": -2.0},
                    ],
                }
            ]
        }

    cfg = LlamaServerConfig(base_url="http://localhost:8080", prefer_raw_logits=True)
    backend = LlamaServerBackend(cfg, request_fn=request_fn)
    step = backend.next_step([1, 2], top_n=2)

    assert step.logits is None
    assert set(step.logprobs or {}) == {11, 0}


def test_backend_next_steps_calls_next_step_per_context() -> None:
    def request_fn(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        path = urlparse(url).path
        if path != "/completion":
            raise AssertionError(f"unexpected path: {path}")
        return {"logits": [1.0, 0.0, -1.0]}

    backend = LlamaServerBackend(LlamaServerConfig(base_url="http://localhost:8080"), request_fn=request_fn)
    rows = backend.next_steps([[1], [1, 2]], top_n=2)
    assert len(rows) == 2
    assert rows[0].logits == [1.0, 0.0, -1.0]


def test_probe_capabilities_reports_success() -> None:
    def request_fn(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        path = urlparse(url).path
        if path == "/tokenize":
            return {"tokens": [5, 6]}
        if path == "/detokenize":
            return {"content": "hi"}
        if path == "/completion":
            if payload.get("n_predict") == 0:
                return {"logits": [1.0, 0.0, -1.0]}
            return {
                "completion_probabilities": [
                    {"probs": [{"id": 0, "logprob": -0.1}, {"id": 1, "logprob": -1.2}]}
                ]
            }
        raise AssertionError(f"unexpected path: {path}")

    backend = LlamaServerBackend(LlamaServerConfig(base_url="http://localhost:8080"), request_fn=request_fn)
    caps = backend.probe_capabilities("hi", top_n=2)
    assert caps.tokenize_ok is True
    assert caps.detokenize_ok is True
    assert caps.raw_logits_ok is True
    assert caps.sparse_logprobs_ok is True
