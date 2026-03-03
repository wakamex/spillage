"""HTTP backend for llama-server style endpoints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .backend import (
    BackendProtocolError,
    BackendRequestError,
    StepResult,
)
from .scoring import log_softmax

RequestFn = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _default_request(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urlopen(req, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BackendRequestError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except URLError as exc:
        raise BackendRequestError(f"request failed for {url}: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BackendProtocolError(f"non-JSON response from {url}: {raw[:200]!r}") from exc

    if not isinstance(parsed, dict):
        raise BackendProtocolError(f"unexpected response type from {url}: {type(parsed)!r}")
    return parsed


@dataclass(frozen=True)
class LlamaServerConfig:
    base_url: str
    eos_token_id: int = 2
    timeout_s: float = 30.0
    prefer_raw_logits: bool = True
    default_top_n: int = 20
    api_mode: str = "native"  # native|oai
    model: str | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    tokenize_ok: bool
    detokenize_ok: bool
    raw_logits_ok: bool
    sparse_logprobs_ok: bool
    notes: tuple[str, ...]


class LlamaServerBackend:
    """Backend implementation for local llama-server.

    Native mode (`/completion`) is token-id based and preferred for MSS.
    OAI mode (`/v1/completions`) is sparse-logprobs only and approximates token IDs.
    """

    def __init__(self, config: LlamaServerConfig, request_fn: RequestFn | None = None) -> None:
        self.config = config
        self.eos_token_id = config.eos_token_id
        self._request = request_fn or _default_request
        # Tri-state cache:
        # - None: unknown (not probed yet)
        # - True: confirmed available
        # - False: confirmed unavailable
        self._raw_logits_available: bool | None = None

    def encode(self, text: str) -> list[int]:
        response = self._post(
            "/tokenize",
            {
                "content": text,
            },
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(v, int) for v in tokens):
            raise BackendProtocolError("/tokenize response missing integer `tokens`")
        return tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        response = self._post(
            "/detokenize",
            {
                "tokens": list(token_ids),
            },
        )
        content = response.get("content")
        if not isinstance(content, str):
            raise BackendProtocolError("/detokenize response missing string `content`")
        return content

    def next_step(
        self,
        context_ids: Sequence[int],
        *,
        top_n: int | None = None,
        require_logits: bool = False,
    ) -> StepResult:
        requested_top_n = max(2, top_n or self.config.default_top_n)

        should_try_raw = require_logits or (
            self.config.prefer_raw_logits and self._raw_logits_available is not False
        )
        if should_try_raw:
            try:
                logits = self._fetch_raw_logits(context_ids)
                self._raw_logits_available = True
                lps = log_softmax(logits)
                top_ids = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:requested_top_n]
                sparse = {token_id: lps[token_id] for token_id in top_ids}
                return StepResult(logits=logits, logprobs=sparse)
            except BackendRequestError:
                self._raw_logits_available = False
                if require_logits:
                    raise
            except BackendProtocolError:
                self._raw_logits_available = False
                if require_logits:
                    raise

        if self.config.api_mode == "oai":
            return self._fetch_sparse_oai(context_ids, requested_top_n)
        return self._fetch_sparse_native(context_ids, requested_top_n)

    def next_steps(
        self,
        contexts: Sequence[Sequence[int]],
        *,
        top_n: int | None = None,
        require_logits: bool = False,
    ) -> list[StepResult]:
        # Conservative default: one request per branch. This keeps behavior
        # stable across different server builds and response schemas.
        return [
            self.next_step(ctx, top_n=top_n, require_logits=require_logits)
            for ctx in contexts
        ]

    def probe_capabilities(self, probe_text: str = "hello", top_n: int = 5) -> BackendCapabilities:
        notes: list[str] = []
        tokenize_ok = False
        detokenize_ok = False
        raw_logits_ok = False
        sparse_ok = False

        probe_ids: list[int] = []

        try:
            probe_ids = self.encode(probe_text)
            tokenize_ok = True
        except Exception as exc:  # noqa: BLE001
            notes.append(f"tokenize failed: {exc}")

        if tokenize_ok:
            try:
                _ = self.decode(probe_ids)
                detokenize_ok = True
            except Exception as exc:  # noqa: BLE001
                notes.append(f"detokenize failed: {exc}")

        if not probe_ids:
            probe_ids = [0]
            notes.append("using fallback probe token id [0]")

        try:
            step = self.next_step(probe_ids, top_n=top_n, require_logits=True)
            raw_logits_ok = step.logits is not None and len(step.logits) > 0
        except Exception as exc:  # noqa: BLE001
            notes.append(f"raw logits probe failed: {exc}")

        try:
            step = self.next_step(probe_ids, top_n=top_n, require_logits=False)
            sparse_ok = step.logprobs is not None and len(step.logprobs) > 0
        except Exception as exc:  # noqa: BLE001
            notes.append(f"sparse logprob probe failed: {exc}")

        return BackendCapabilities(
            tokenize_ok=tokenize_ok,
            detokenize_ok=detokenize_ok,
            raw_logits_ok=raw_logits_ok,
            sparse_logprobs_ok=sparse_ok,
            notes=tuple(notes),
        )

    def _fetch_raw_logits(self, context_ids: Sequence[int]) -> list[float]:
        data = self._post(
            "/completion",
            {
                "prompt": list(context_ids),
                "n_predict": 0,
                "temperature": 0.0,
                "cache_prompt": True,
                "return_logits": True,
                "logits": True,
            },
        )

        for key in ("logits", "next_token_logits"):
            logits = data.get(key)
            if isinstance(logits, list) and all(isinstance(v, (int, float)) for v in logits):
                return [float(v) for v in logits]

        raise BackendProtocolError("native /completion response missing full-vocab logits")

    def _fetch_sparse_native(self, context_ids: Sequence[int], top_n: int) -> StepResult:
        data = self._post(
            "/completion",
            {
                "prompt": list(context_ids),
                "n_predict": 1,
                "temperature": 0.0,
                "cache_prompt": True,
                "n_probs": top_n,
            },
        )
        sparse = self._extract_sparse_logprobs(data, top_n)
        return StepResult(logits=None, logprobs=sparse)

    def _fetch_sparse_oai(self, context_ids: Sequence[int], top_n: int) -> StepResult:
        prompt = self.decode(context_ids)
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": top_n,
            "echo": False,
        }
        if self.config.model:
            payload["model"] = self.config.model

        data = self._post("/v1/completions", payload)

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise BackendProtocolError("oai response missing `choices`")

        first = choices[0]
        if not isinstance(first, dict):
            raise BackendProtocolError("oai response choice has invalid shape")

        logprobs = first.get("logprobs")
        if not isinstance(logprobs, dict):
            raise BackendProtocolError("oai response missing `choices[0].logprobs`")

        top_logprobs = logprobs.get("top_logprobs")
        if not isinstance(top_logprobs, list) or not top_logprobs:
            raise BackendProtocolError("oai response missing `top_logprobs`")

        first_map = top_logprobs[0]
        if not isinstance(first_map, dict):
            raise BackendProtocolError("oai `top_logprobs[0]` is not a map")

        out: dict[int, float] = {}
        for token_text, logp in first_map.items():
            if not isinstance(token_text, str) or not isinstance(logp, (int, float)):
                continue
            token_ids = self.encode(token_text)
            if not token_ids:
                continue
            # Approximation: in OAI mode we map token string to its final token id.
            out[token_ids[-1]] = float(logp)

        if not out:
            raise BackendProtocolError("unable to map oai top_logprobs to token IDs")

        sorted_items = sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return StepResult(logits=None, logprobs=dict(sorted_items))

    def _extract_sparse_logprobs(self, data: dict[str, Any], top_n: int) -> dict[int, float]:
        result: dict[int, float] = {}

        # Common llama.cpp shape (older): completion_probabilities[0].probs[]
        # Newer shape: completion_probabilities[0].top_logprobs[]
        probs_container = data.get("completion_probabilities")
        if isinstance(probs_container, list) and probs_container:
            first = probs_container[0]
            if isinstance(first, dict):
                probs = first.get("probs")
                if isinstance(probs, list):
                    for item in probs:
                        if not isinstance(item, dict):
                            continue
                        token_id = self._coerce_token_id(
                            item.get("id", item.get("token_id", item.get("tok", item.get("token"))))
                        )
                        if token_id is None:
                            continue
                        logprob = self._coerce_logprob(item)
                        if logprob is None:
                            continue
                        result[token_id] = logprob

                # Newer shape: list[{"id": int, "logprob": float, ...}]
                top_logprobs = first.get("top_logprobs")
                if isinstance(top_logprobs, list):
                    for item in top_logprobs:
                        if not isinstance(item, dict):
                            continue
                        token_id = self._coerce_token_id(item.get("id", item.get("token_id")))
                        if token_id is None:
                            continue
                        logprob = self._coerce_logprob(item)
                        if logprob is None:
                            continue
                        result[token_id] = logprob

        # Alternate shape: top_logprobs map token_id -> logprob.
        if not result:
            top_logprobs = data.get("top_logprobs")
            if isinstance(top_logprobs, dict):
                for token_key, logp in top_logprobs.items():
                    token_id = self._coerce_token_id(token_key)
                    if token_id is None or not isinstance(logp, (int, float)):
                        continue
                    result[token_id] = float(logp)

        if not result:
            raise BackendProtocolError("unable to parse sparse token logprobs from /completion response")

        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True)[:top_n])

    def _coerce_token_id(self, raw: Any) -> int | None:
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str):
            try:
                return int(raw)
            except ValueError:
                return None
        return None

    def _coerce_logprob(self, item: dict[str, Any]) -> float | None:
        logp = item.get("logprob")
        if isinstance(logp, (int, float)):
            return float(logp)

        prob = item.get("prob")
        if isinstance(prob, (int, float)) and prob > 0.0:
            return math.log(float(prob))
        return None

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = urljoin(self.config.base_url.rstrip("/") + "/", path.lstrip("/"))
        return self._request(url, payload, self.config.timeout_s)
