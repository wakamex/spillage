"""Stress test case definitions for MSS evaluation."""

from __future__ import annotations

import csv
import functools
import pathlib
import re
import subprocess
from dataclasses import dataclass
from typing import Callable

_CASES_DIR = pathlib.Path(__file__).parent.parent / "cases"


@dataclass(frozen=True)
class TestCase:
    name: str
    prompt: str
    check: Callable[[str], bool]
    description: str
    category: str  # "factual", "math", "pattern"


_ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models."""
    if "<think>" in text and "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


def _extract_answer(text: str) -> str:
    """Extract the answer after 'Answer:' prefix, stripping think tags and markdown."""
    text = _strip_think(text)
    text = text.replace("**", "").replace("*", "")
    m = _ANSWER_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: first non-empty line.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


@functools.lru_cache(maxsize=2048)
def _claude_judge(question: str, expected: str, extracted: str) -> bool:
    """Use `claude -p` to judge semantic equivalence. Cached by (expected, extracted)."""
    # Reject clearly garbage extractions (truncated thinking blocks).
    stripped = extracted.strip()
    if not stripped or stripped in ("<think>", "</think>") or stripped.startswith("<think>") and "</think>" not in stripped:
        return False
    # Fast path: exact substring match avoids a claude call.
    if expected.lower() in extracted.lower():
        return True
    prompt = (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Model answer: {extracted}\n\n"
        "Is the model answer correct? Allow for semantic equivalence, abbreviations, "
        "and minor formatting differences. Reply with only 'yes' or 'no'."
    )
    try:
        import os
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=30, env=env,
        )
        return result.stdout.strip().lower().startswith("yes")
    except Exception:
        # Fallback to substring if claude unavailable.
        return expected.lower() in extracted.lower()


def _contains(text: str, *targets: str) -> bool:
    lower = _strip_think(text).lower()
    return any(t.lower() in lower for t in targets)


def _not_contains(text: str, *targets: str) -> bool:
    lower = _strip_think(text).lower()
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
    """Return built-in stress test cases, optionally filtered by category."""
    if category is None:
        return list(CASES)
    return [c for c in CASES if c.category == category]


def load_simple_qa(n: int | None = None, seed: int = 42) -> list[TestCase]:
    """Load cases from cases/simple_qa.csv (SimpleQA benchmark).

    Parameters
    ----------
    n:
        Number of cases to return. If None, return all 4332.
    seed:
        Random seed for reproducible shuffling when n is set.
    """
    csv_path = _CASES_DIR / "simple_qa.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"SimpleQA data not found at {csv_path}. "
            "It should be committed at cases/simple_qa.csv."
        )

    import random as _random

    rows: list[TestCase] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            question = row["problem"].strip()
            answer = row["answer"].strip()
            name = f"simpleqa_{i:04d}"
            # Prompt: direct-answer suffix. Think-tag output is stripped by the checker.
            prompt = f"{question} Answer:"
            rows.append(TestCase(
                name=name,
                prompt=prompt,
                check=lambda t, q=question, a=answer: _claude_judge(q, a, _extract_answer(t)),
                description=f"SimpleQA: expected '{answer}'",
                category="simpleqa",
            ))

    if n is not None:
        rng = _random.Random(seed)
        rows = rng.sample(rows, min(n, len(rows)))
        rows.sort(key=lambda c: c.name)

    return rows
