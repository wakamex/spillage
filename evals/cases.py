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

    # --- Math (harder) ---
    TestCase(
        name="hard_arithmetic",
        prompt="Calculate: 1247 × 38 = ",
        check=lambda t: _contains(t, "47386"),
        description="Multi-digit multiplication — models often get this wrong.",
        category="math",
    ),
    TestCase(
        name="subtraction_borrow",
        prompt="What is 5003 - 2847? The answer is",
        check=lambda t: _contains(t, "2156"),
        description="Subtraction with multiple borrows — common error source.",
        category="math",
    ),

    # --- Factual (harder) ---
    TestCase(
        name="entity_dense_paragraph",
        prompt=(
            "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911. "
            "Her daughter Irène Joliot-Curie also won the Nobel Prize in Chemistry in 1935. "
            "The year Marie Curie won her second Nobel Prize was"
        ),
        check=lambda t: _contains(t, "1911"),
        description="Dense entity paragraph — model must track 'second' across multiple dates.",
        category="factual",
    ),
    TestCase(
        name="false_presupposition",
        prompt="Everyone knows that the Great Wall of China is visible from space. In reality, this claim is",
        check=lambda t: _contains(t, "false", "myth", "incorrect", "not true", "misleading", "wrong", "untrue", "debunked"),
        description="Must reject a common false presupposition.",
        category="factual",
    ),
    TestCase(
        name="reverse_order_recall",
        prompt=(
            "The first three US presidents were Washington, Adams, and Jefferson. "
            "In reverse order, the third, second, and first presidents were"
        ),
        check=lambda t: (
            t.lower().find("jefferson") < t.lower().find("adams") < t.lower().find("washington")
            if all(x in t.lower() for x in ("jefferson", "adams", "washington"))
            else False
        ),
        description="Must reverse the order: Jefferson, Adams, Washington.",
        category="factual",
    ),
    TestCase(
        name="similar_sounding_capital",
        prompt="The capital of Slovakia is not Bratislava's neighbor Vienna, it is",
        check=lambda t: _contains(t, "Bratislava"),
        description="Confusing phrasing — must still say Bratislava despite the misdirection.",
        category="factual",
    ),

    # --- Pattern ---
    TestCase(
        name="repetition_trap",
        prompt="A B C A B C A B C A B",
        check=lambda t: t.strip().startswith(" C") or t.strip().startswith("C"),
        description="Should continue pattern with 'C'.",
        category="pattern",
    ),
    TestCase(
        name="fibonacci_pattern",
        prompt="1, 1, 2, 3, 5, 8, 13, 21, 34,",
        check=lambda t: _contains(t, "55"),
        description="Must continue Fibonacci: 34 + 21 = 55.",
        category="pattern",
    ),
    TestCase(
        name="letter_shift",
        prompt="a->b, c->d, e->f, g->",
        check=lambda t: t.strip().startswith("h"),
        description="Simple letter-shift pattern: g->h.",
        category="pattern",
    ),
]


def get_cases(category: str | None = None) -> list[TestCase]:
    """Return test cases, optionally filtered by category."""
    if category is None:
        return list(CASES)
    return [c for c in CASES if c.category == category]
