"""HTTP backend for llama-server (proxy mode only).

Uses the native /completion endpoint with n_probs to get top-N
probabilities. Cannot compute exact log Z — operates in proxy mode.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

try:
    import httpx
except ImportError as exc:
    raise ImportError("httpx is required for HttpBackend.") from exc

from .backend import Backend, LogitResult


class HttpBackend:
    """Proxy-mode backend hitting a llama-server's /completion endpoint.

    Parameters
    ----------
    base_url:
        Server URL, e.g. ``http://localhost:8080``.
    n_probs:
        Number of top probabilities to request per step.
    timeout:
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        n_probs: int = 100,
        timeout: float = 30.0,
    ) -> None:
        self._url = base_url.rstrip("/")
        self._n_probs = n_probs
        self._client = httpx.Client(timeout=timeout)
        self._eos: int | None = None

    def _completion(self, prompt: str) -> dict:
        """Call /completion with n_predict=0 to get logprobs without generating."""
        resp = self._client.post(
            f"{self._url}/completion",
            json={
                "prompt": prompt,
                "n_predict": 1,
                "n_probs": self._n_probs,
                "temperature": 0.0,
                "cache_prompt": True,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def _parse_completion(self, data: dict) -> LogitResult:
        """Parse /completion response into a LogitResult (proxy mode)."""
        # The response contains completion_probabilities or similar.
        # Structure varies by llama.cpp version. Handle common formats.
        probs_list = data.get("completion_probabilities", [])
        if probs_list:
            # completion_probabilities is a list of per-token prob arrays.
            # We want the first (and only, since n_predict=1) entry.
            entry = probs_list[0]
            top_probs = entry.get("probs", [])
        else:
            top_probs = []

        if not top_probs:
            # Fallback: check for content + probs in different format.
            top_probs = data.get("probs", [])

        ids = []
        logprobs = []
        probs = []
        for item in top_probs:
            tok_id = item.get("tok_id", item.get("id", 0))
            prob = item.get("prob", 0.0)
            ids.append(tok_id)
            probs.append(prob)
            logprobs.append(float(np.log(max(prob, 1e-12))))

        if not ids:
            # Empty response — return a degenerate result.
            return LogitResult(
                logits=None,
                log_z=float("nan"),
                top_k_ids=np.array([], dtype=np.int64),
                top_k_logits=np.array([], dtype=np.float64),
                top_k_probs=np.array([], dtype=np.float64),
                entropy=0.0,
                top1_margin=1.0,
            )

        arr_probs = np.array(probs, dtype=np.float64)
        arr_ids = np.array(ids, dtype=np.int64)
        arr_logprobs = np.array(logprobs, dtype=np.float64)

        # Compute proxy fields.
        safe_probs = arr_probs + 1e-12
        entropy = float(-np.sum(arr_probs * np.log(safe_probs)))
        sorted_p = np.sort(arr_probs)[::-1]
        margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0

        return LogitResult(
            logits=None,
            log_z=float("nan"),
            top_k_ids=arr_ids,
            top_k_logits=arr_logprobs,
            top_k_probs=arr_probs,
            entropy=entropy,
            top1_margin=margin,
        )

    def get_logits(self, token_ids: list[int]) -> LogitResult:
        # HttpBackend receives token_ids but the server expects text.
        # We detokenize first, then send as prompt.
        text = self.detokenize(token_ids)
        data = self._completion(text)
        return self._parse_completion(data)

    def get_logits_batch(
        self, token_id_seqs: list[list[int]]
    ) -> list[LogitResult]:
        return [self.get_logits(seq) for seq in token_id_seqs]

    def tokenize(self, text: str) -> list[int]:
        resp = self._client.post(
            f"{self._url}/tokenize",
            json={"content": text},
        )
        resp.raise_for_status()
        return resp.json().get("tokens", [])

    def detokenize(self, token_ids: list[int]) -> str:
        resp = self._client.post(
            f"{self._url}/detokenize",
            json={"tokens": token_ids},
        )
        resp.raise_for_status()
        return resp.json().get("content", "")

    def mode(self) -> Literal["proxy"]:
        return "proxy"

    def eos_token_id(self) -> int:
        if self._eos is None:
            # Try to get from model props.
            try:
                resp = self._client.get(f"{self._url}/props")
                resp.raise_for_status()
                props = resp.json()
                self._eos = props.get("default_generation_settings", {}).get(
                    "eos_token_id", 2
                )
            except Exception:
                self._eos = 2  # common default
        return self._eos
