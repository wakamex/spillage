"""Backend protocol and mock backend for MSS development."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .scoring import log_softmax


@dataclass(frozen=True)
class StepResult:
    """One-step model output.

    `logits` should contain full-vocabulary raw logits when available.
    `logprobs` is an optional sparse token->logprob map.
    """

    logits: list[float] | None = None
    logprobs: dict[int, float] | None = None


class BackendError(RuntimeError):
    """Base backend exception."""


class BackendProtocolError(BackendError):
    """Raised when backend responses are malformed."""


class BackendRequestError(BackendError):
    """Raised when backend requests fail."""


class Backend(Protocol):
    eos_token_id: int

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs into text."""

    def next_step(
        self,
        context_ids: Sequence[int],
        *,
        top_n: int | None = None,
        require_logits: bool = False,
    ) -> StepResult:
        """Return next-token distribution for the provided context."""

    def next_steps(
        self,
        contexts: Sequence[Sequence[int]],
        *,
        top_n: int | None = None,
        require_logits: bool = False,
    ) -> list[StepResult]:
        """Return next-token distributions for multiple contexts."""


class MockBackend:
    """Test backend using deterministic context->logits mapping."""

    def __init__(
        self,
        logits_by_context: Mapping[tuple[int, ...], Sequence[float]],
        *,
        eos_token_id: int,
    ) -> None:
        self.eos_token_id = eos_token_id
        self._table = {
            tuple(context): [float(v) for v in logits]
            for context, logits in logits_by_context.items()
        }

    def next_step(
        self,
        context_ids: Sequence[int],
        *,
        top_n: int | None = None,
        require_logits: bool = False,
    ) -> StepResult:
        key = tuple(context_ids)
        if key not in self._table:
            raise KeyError(f"missing mock logits for context {key}")

        logits = self._table[key]
        if require_logits:
            return StepResult(logits=list(logits), logprobs=None)

        if top_n is None:
            top_n = len(logits)

        lps = log_softmax(logits)
        top_ids = sorted(range(len(logits)), key=lambda i: lps[i], reverse=True)[:top_n]
        sparse = {idx: lps[idx] for idx in top_ids}
        return StepResult(logits=list(logits), logprobs=sparse)

    def next_steps(
        self,
        contexts: Sequence[Sequence[int]],
        *,
        top_n: int | None = None,
        require_logits: bool = False,
    ) -> list[StepResult]:
        return [
            self.next_step(ctx, top_n=top_n, require_logits=require_logits)
            for ctx in contexts
        ]

    def encode(self, text: str) -> list[int]:
        text = text.strip()
        if not text:
            return []
        return [int(part) for part in text.split()]

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(str(tok) for tok in token_ids)
