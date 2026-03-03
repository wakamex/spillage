"""Built-in stress suites for MSS validation."""

from __future__ import annotations

from .eval_harness import TextEvalCase

DEFAULT_SNAPSHOT_DATE = "2026-03-03"


def stress_suite_core(snapshot_date: str = DEFAULT_SNAPSHOT_DATE) -> list[TextEvalCase]:
    """Return a compact suite aligned to the MSS proposal.

    The snapshot date is included directly in prompts to reduce time-based ambiguity.
    """

    return [
        TextEvalCase(
            name="entity_swap",
            prompt="The founder of SpaceX is Elon Musk, but the founder of Blue Origin is",
            expected_substring="Jeff Bezos",
        ),
        TextEvalCase(
            name="multi_digit_math",
            prompt="What is 347 + 268? Answer with just the number.",
            expected_substring="615",
        ),
        TextEvalCase(
            name="dead_person_bio_alive_guard",
            prompt=(
                "As of "
                f"{snapshot_date}, Keanu Reeves is alive or dead? "
                "Answer with one word."
            ),
            expected_substring="alive",
        ),
    ]
