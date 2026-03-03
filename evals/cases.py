"""Stress test case definitions for MSS evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TestCase:
    name: str
    prompt: str
    check: Callable[[str], bool]
    description: str
    category: str  # "factual", "math", "pattern"


def _contains(text: str, *targets: str) -> bool:
    lower = text.lower()
    return any(t.lower() in lower for t in targets)


def _not_contains(text: str, *targets: str) -> bool:
    lower = text.lower()
    return not any(t.lower() in lower for t in targets)


# ---------------------------------------------------------------------------
# Test cases from motivation.md + additions from CLAUDE_PROPOSAL
# ---------------------------------------------------------------------------

CASES: list[TestCase] = [
    # --- Factual ---
    TestCase(
        name="entity_swap",
        prompt="The founder of SpaceX is Elon Musk, but the founder of Blue Origin is",
        check=lambda t: _contains(t, "Jeff Bezos", "Bezos"),
        description="Should produce 'Jeff Bezos', not repeat 'Elon Musk'.",
        category="factual",
    ),
    TestCase(
        name="negation",
        prompt="The capital of Australia is not Sydney, it is",
        check=lambda t: _contains(t, "Canberra"),
        description="Should produce 'Canberra', not 'Sydney'.",
        category="factual",
    ),
    TestCase(
        name="single_token_factual",
        prompt="The chemical symbol for gold is",
        check=lambda t: _contains(t, "Au"),
        description="'Au' — single-token, cleanest ΔE signal.",
        category="factual",
    ),
    TestCase(
        name="dead_person_bio",
        prompt="Nelson Mandela passed away in 2013. Is Barack Obama alive? As of 2024, Barack Obama",
        check=lambda t: (
            _not_contains(t, "passed away", "died", "death", "deceased")
            and _contains(t, "alive", "living", "is", "continues")
        ),
        description="Should not generate death information for a living person.",
        category="factual",
    ),

    # --- Math ---
    TestCase(
        name="multi_digit_math",
        prompt="What is 347 + 268? The answer is",
        check=lambda t: _contains(t, "615"),
        description="Should produce '615'.",
        category="math",
    ),

    # --- Pattern ---
    TestCase(
        name="repetition_trap",
        prompt="A B C A B C A B C A B",
        check=lambda t: t.strip().startswith(" C") or t.strip().startswith("C"),
        description="Should continue pattern with 'C'.",
        category="pattern",
    ),
]


def get_cases(category: str | None = None) -> list[TestCase]:
    """Return test cases, optionally filtered by category."""
    if category is None:
        return list(CASES)
    return [c for c in CASES if c.category == category]
