"""Hyperparameters and model presets for Min-Spill Search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MSSConfig:
    k: int = 3
    beta: float = 2.0
    tau: float = 1.0
    alpha: float = 1.0
    lambda_proxy: float = 0.5
    confidence_threshold: float = 0.92
    panic_margin: float = 2.0
    max_tokens: int = 256
    temperature: float = 0.0
    adaptive: bool = True
    delta_e_threshold: float = -4.5


# Model-specific presets (tau/beta on z-normalized scale).
MODEL_PRESETS: dict[str, MSSConfig] = {
    "qwen35-35b": MSSConfig(beta=1.8, tau=1.2),
    "qwen35-9b": MSSConfig(beta=2.2, tau=0.8),
    "default": MSSConfig(),
}


def get_preset(name: str) -> MSSConfig:
    return MODEL_PRESETS.get(name, MODEL_PRESETS["default"])
