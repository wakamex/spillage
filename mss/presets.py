"""Model-specific MSS presets."""

from __future__ import annotations

from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "qwen35_35b": {
        "k": 3,
        "beta": 1.8,
        "tau": 1.2,
    },
    "qwen35_9b": {
        "k": 3,
        "beta": 2.2,
        "tau": 0.8,
    },
}


def get_preset(name: str) -> dict[str, Any]:
    if name not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown preset {name!r}; valid presets: {valid}")
    return dict(PRESETS[name])
