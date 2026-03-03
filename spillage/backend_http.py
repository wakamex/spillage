"""HTTP backend for llama-server (proxy mode only).

Uses the native /completion endpoint with n_probs to get top-N
probabilities. Cannot compute exact log Z — operates in proxy mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    import httpx
except ImportError as exc:
    raise ImportError("httpx is required for HttpBackend.") from exc

from .backend import Backend, LogitResult


@dataclass(frozen=True)
class BackendCapabilities:
    """Results of probing a backend for supported features."""
    tokenize_ok: bool
    detokenize_ok: bool
    raw_logits_ok: bool
    sparse_logprobs_ok: bool
    notes: tuple[str, ...]


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

    def probe_capabilities(self, probe_text: str = "hello", top_n: int = 5) -> BackendCapabilities:
        """Probe what the server supports: tokenize, raw logits, sparse logprobs."""
        notes: list[str] = []
        tokenize_ok = False
        detokenize_ok = False
        raw_logits_ok = False
        sparse_ok = False
        probe_ids: list[int] = []

        try:
            probe_ids = self.tokenize(probe_text)
            tokenize_ok = True
        except Exception as exc:  # noqa: BLE001
            notes.append(f"tokenize failed: {exc}")

        if tokenize_ok:
            try:
                self.detokenize(probe_ids)
                detokenize_ok = True
            except Exception as exc:  # noqa: BLE001
                notes.append(f"detokenize failed: {exc}")

        if not probe_ids:
            probe_ids = [0]
            notes.append("using fallback probe token id [0]")

        # Try raw logits (requires patched server with return_logits support).
        try:
            resp = self._client.post(
                f"{self._url}/completion",
                json={
                    "prompt": probe_ids,
                    "n_predict": 0,
                    "temperature": 0.0,
                    "return_logits": True,
                    "logits": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            for key in ("logits", "next_token_logits"):
                val = data.get(key)
                if isinstance(val, list) and val:
                    raw_logits_ok = True
                    break
            if not raw_logits_ok:
                notes.append("server responded but no dense logits in response")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"raw logits probe failed: {exc}")

        # Try sparse logprobs (always available via n_probs).
        try:
            result = self.get_logits(probe_ids)
            sparse_ok = len(result.top_k_ids) > 0
        except Exception as exc:  # noqa: BLE001
            notes.append(f"sparse logprob probe failed: {exc}")

        return BackendCapabilities(
            tokenize_ok=tokenize_ok,
            detokenize_ok=detokenize_ok,
            raw_logits_ok=raw_logits_ok,
            sparse_logprobs_ok=sparse_ok,
            notes=tuple(notes),
        )

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
