from mss.stress_suites import stress_suite_core


def test_stress_suite_core_contains_three_cases() -> None:
    suite = stress_suite_core(snapshot_date="2026-03-03")
    assert len(suite) == 3
    assert suite[0].name == "entity_swap"
    assert "2026-03-03" in suite[2].prompt
