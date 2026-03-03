"""Native backend using libspillage_llama (ctypes wrapper around llama.cpp).

This is the primary backend for MSS validation — full logit access,
no HTTP overhead, direct KV-cache control. Uses the spillage_llama
shared library built from /code/llama.cpp/tools/spillage/.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .backend import Backend, LogitResult

# Default search paths for the shared library.
_DEFAULT_LIB_PATHS = [
    os.environ.get("SPILLAGE_LLAMA_LIB", ""),
    "/code/llama.cpp/build/bin/libspillage_llama.so",
]


def _find_lib() -> str:
    for p in _DEFAULT_LIB_PATHS:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "libspillage_llama.so not found. Set SPILLAGE_LLAMA_LIB env var "
        "or build from /code/llama.cpp (branch spillage-bindings)."
    )


def _load_lib(path: str | None = None) -> ctypes.CDLL:
    lib_path = path or _find_lib()
    # Also add the directory to LD path so libllama.so etc. are found.
    lib_dir = str(Path(lib_path).parent)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in existing:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}"

    lib = ctypes.CDLL(lib_path)

    # spillage_init
    lib.spillage_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
    lib.spillage_init.restype = ctypes.c_void_p

    # spillage_free
    lib.spillage_free.argtypes = [ctypes.c_void_p]
    lib.spillage_free.restype = None

    # spillage_n_vocab
    lib.spillage_n_vocab.argtypes = [ctypes.c_void_p]
    lib.spillage_n_vocab.restype = ctypes.c_int

    # spillage_token_eos
    lib.spillage_token_eos.argtypes = [ctypes.c_void_p]
    lib.spillage_token_eos.restype = ctypes.c_int

    # spillage_token_bos
    lib.spillage_token_bos.argtypes = [ctypes.c_void_p]
    lib.spillage_token_bos.restype = ctypes.c_int

    # spillage_tokenize
    lib.spillage_tokenize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_bool,
    ]
    lib.spillage_tokenize.restype = ctypes.c_int

    # spillage_detokenize
    lib.spillage_detokenize.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ]
    lib.spillage_detokenize.restype = ctypes.c_int

    # spillage_eval
    lib.spillage_eval.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
    ]
    lib.spillage_eval.restype = ctypes.POINTER(ctypes.c_float)

    return lib


# Singleton library handle.
_lib: ctypes.CDLL | None = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load_lib()
    return _lib


def _top_k(logits: NDArray[np.floating], k: int) -> tuple[NDArray, NDArray]:
    """Return (indices, values) of the top-k logits."""
    k = min(k, len(logits))
    idx = np.argpartition(logits, -k)[-k:]
    order = np.argsort(logits[idx])[::-1]
    idx = idx[order]
    return idx, logits[idx]


class NativeBackend:
    """Wraps libspillage_llama for in-process GGUF inference with full logit access.

    Parameters
    ----------
    model_path:
        Path to a GGUF model file.
    n_ctx:
        Context window size.
    n_gpu_layers:
        Layers to offload to GPU (-1 = all, 99 = all).
    top_k:
        Number of top candidates to include in LogitResult.
    verbose:
        Whether to print llama.cpp logs.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 99,
        top_k: int = 10,
        verbose: bool = False,
    ) -> None:
        self._lib = _get_lib()
        self._ctx = self._lib.spillage_init(
            model_path.encode("utf-8"), n_ctx, n_gpu_layers, verbose,
        )
        if not self._ctx:
            raise RuntimeError(f"Failed to load model: {model_path}")
        self._top_k = top_k
        self._n_vocab = self._lib.spillage_n_vocab(self._ctx)

    def __del__(self) -> None:
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.spillage_free(self._ctx)
            self._ctx = None

    def _eval_and_get_logits(self, token_ids: list[int]) -> NDArray[np.floating]:
        """Clear cache, evaluate tokens, return logits at last position."""
        arr = (ctypes.c_int32 * len(token_ids))(*token_ids)
        ptr = self._lib.spillage_eval(self._ctx, arr, len(token_ids))
        if not ptr:
            raise RuntimeError("spillage_eval failed")
        # Copy into numpy array (pointer is valid until next eval).
        logits = np.ctypeslib.as_array(ptr, shape=(self._n_vocab,)).copy()
        return logits.astype(np.float64)

    def get_logits(self, token_ids: list[int]) -> LogitResult:
        logits = self._eval_and_get_logits(token_ids)
        log_z = float(np.logaddexp.reduce(logits))

        top_ids, top_vals = _top_k(logits, self._top_k)
        probs = np.exp(logits - log_z)
        top_probs = probs[top_ids]

        return LogitResult(
            logits=logits,
            log_z=log_z,
            top_k_ids=top_ids,
            top_k_logits=top_vals,
            top_k_probs=top_probs,
        )

    def get_logits_batch(
        self, token_id_seqs: list[list[int]]
    ) -> list[LogitResult]:
        return [self.get_logits(seq) for seq in token_id_seqs]

    def tokenize(self, text: str) -> list[int]:
        max_tokens = len(text) * 4 + 16  # generous upper bound
        out = (ctypes.c_int32 * max_tokens)()
        n = self._lib.spillage_tokenize(
            self._ctx, text.encode("utf-8"), out, max_tokens, True,
        )
        if n < 0:
            raise RuntimeError(f"tokenize failed: {n}")
        return list(out[:n])

    def detokenize(self, token_ids: list[int]) -> str:
        buf_size = len(token_ids) * 32 + 64
        arr = (ctypes.c_int32 * len(token_ids))(*token_ids)
        buf = ctypes.create_string_buffer(buf_size)
        n = self._lib.spillage_detokenize(self._ctx, arr, len(token_ids), buf, buf_size)
        if n < 0:
            raise RuntimeError(f"detokenize failed: {n}")
        return buf.raw[:n].decode("utf-8", errors="replace")

    def mode(self) -> Literal["raw"]:
        return "raw"

    def eos_token_id(self) -> int:
        return self._lib.spillage_token_eos(self._ctx)
