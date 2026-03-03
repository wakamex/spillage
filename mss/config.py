"""Config loading helpers for MSS."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import tomllib

from .decoder import MSSConfig


def load_config_file(path: str | Path) -> dict[str, object]:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config root must be a TOML table")
    return raw


def apply_overrides(base: MSSConfig, overrides: dict[str, object]) -> MSSConfig:
    allowed = {f.name for f in fields(MSSConfig)}
    filtered = {k: v for k, v in overrides.items() if k in allowed}
    return replace(base, **filtered)


def load_mss_config(path: str | Path | None = None, overrides: dict[str, object] | None = None) -> MSSConfig:
    cfg = MSSConfig()
    if path:
        cfg = apply_overrides(cfg, load_config_file(path))
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg
