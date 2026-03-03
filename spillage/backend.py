"""Backend protocol and shared types for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np


@dataclass
class LogitResult:
    """Result of a single forward pass at the last token position."""

    # Raw mode: full vocabulary logits.  None in proxy mode.
    logits: np.ndarray | None

    # logsumexp(logits) if raw mode; NaN if proxy mode.
    log_z: float

    # Top-k candidates (always populated).
    top_k_ids: np.ndarray       # (k,) int token indices
    top_k_logits: np.ndarray    # (k,) float raw logits or log-probs
    top_k_probs: np.ndarray     # (k,) float probabilities

    # Proxy mode fields (None in raw mode).
    entropy: float | None = None
    top1_margin: float | None = None


@runtime_checkable
class Backend(Protocol):
    """Minimal interface an MSS backend must satisfy."""

    def get_logits(self, token_ids: list[int]) -> LogitResult:
        """Run a forward pass and return logits at the last position."""
        ...

    def get_logits_batch(
        self, token_id_seqs: list[list[int]]
    ) -> list[LogitResult]:
        """Batch version of get_logits.  Default loops sequentially."""
        ...

    def tokenize(self, text: str) -> list[int]:
        ...

    def detokenize(self, token_ids: list[int]) -> str:
        ...

    def mode(self) -> Literal["raw", "proxy"]:
        ...

    def eos_token_id(self) -> int:
        ...
