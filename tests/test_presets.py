from mss.presets import PRESETS, get_preset


def test_get_preset_known_values() -> None:
    q35 = get_preset("qwen35_35b")
    assert q35["beta"] == 1.8
    assert q35["tau"] == 1.2


def test_get_preset_returns_copy() -> None:
    row = get_preset("default")
    row["k"] = 999
    assert "k" not in PRESETS["default"]


def test_get_preset_unknown_raises() -> None:
    try:
        get_preset("nope")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown preset" in str(exc)
