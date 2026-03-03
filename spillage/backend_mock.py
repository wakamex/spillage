"""Mock backend for testing without a running model server."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .backend import Backend, LogitResult

# A small fake vocabulary for deterministic testing.
_DEFAULT_VOCAB = {
    0: "<eos>",
    1: "Hello",
    2: " world",
    3: "Jeff",
    4: " Bezos",
    5: "Elon",
    6: " Musk",
    7: "Au",
    8: "Ag",
    9: "615",
    10: "614",
}


def _softmax(logits: NDArray[np.floating]) -> NDArray[np.floating]:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class MockBackend:
    """Deterministic backend driven by a logit lookup table.

    Parameters
    ----------
    vocab_size:
        Size of the fake vocabulary.
    logit_table:
        Mapping from ``tuple(token_ids)`` to a 1-D logits array of length
        *vocab_size*.  If a context is not in the table a default
        near-uniform distribution is returned.
    proxy_mode:
        If True, behaves as a proxy backend (no full logits exposed).
    eos_id:
        Token ID for end-of-sequence.
    """

    def __init__(
        self,
        vocab_size: int = 16,
        logit_table: dict[tuple[int, ...], NDArray[np.floating]] | None = None,
        proxy_mode: bool = False,
        eos_id: int = 0,
        vocab: dict[int, str] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self._table = logit_table or {}
        self._proxy = proxy_mode
        self._eos = eos_id
        self._vocab = vocab or _DEFAULT_VOCAB
        # Reverse map for tokenize.
        self._token_to_id: dict[str, int] = {v: k for k, v in self._vocab.items()}

    # -- Backend protocol --------------------------------------------------

    def get_logits(self, token_ids: list[int]) -> LogitResult:
        key = tuple(token_ids)
        if key in self._table:
            logits = self._table[key].astype(np.float64)
        else:
            # Default: slightly prefer higher token ids to break ties
            # deterministically, but keep distribution near-uniform.
            logits = np.linspace(0.0, 0.5, self.vocab_size, dtype=np.float64)

        log_z = float(np.logaddexp.reduce(logits))
        probs = _softmax(logits)

        top_k = min(len(logits), 5)
        top_idx = np.argsort(logits)[::-1][:top_k]

        if self._proxy:
            ent = float(-np.sum(probs * np.log(probs + 1e-12)))
            sorted_probs = np.sort(probs)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
            return LogitResult(
                logits=None,
                log_z=float("nan"),
                top_k_ids=top_idx,
                top_k_logits=np.log(probs[top_idx] + 1e-12),
                top_k_probs=probs[top_idx],
                entropy=ent,
                top1_margin=margin,
            )

        return LogitResult(
            logits=logits,
            log_z=log_z,
            top_k_ids=top_idx,
            top_k_logits=logits[top_idx],
            top_k_probs=probs[top_idx],
        )

    def get_logits_batch(
        self, token_id_seqs: list[list[int]]
    ) -> list[LogitResult]:
        return [self.get_logits(seq) for seq in token_id_seqs]

    def tokenize(self, text: str) -> list[int]:
        # Naive greedy match against vocab strings.
        ids: list[int] = []
        remaining = text
        while remaining:
            best_id: int | None = None
            best_len = 0
            for tok_str, tok_id in self._token_to_id.items():
                if remaining.startswith(tok_str) and len(tok_str) > best_len:
                    best_id = tok_id
                    best_len = len(tok_str)
            if best_id is None:
                remaining = remaining[1:]  # skip unknown char
            else:
                ids.append(best_id)
                remaining = remaining[best_len:]
        return ids

    def detokenize(self, token_ids: list[int]) -> str:
        return "".join(self._vocab.get(tid, "?") for tid in token_ids)

    def mode(self) -> Literal["raw", "proxy"]:
        return "proxy" if self._proxy else "raw"

    def eos_token_id(self) -> int:
        return self._eos
