"""
agents/review_agent.py — Alignment Review Agent
=================================================

Evaluates a draft lesson plan against a strict four-point rubric and
returns structured JSON feedback.  Any failed criterion includes a
specific, actionable reason that the Rewrite Agent can act on.

Rubric Criteria
---------------
1. **Standards Alignment** — Does the objective use the exact verb and
   cognitive complexity stated in the confirmed standard?
2. **Objective-to-Assessment Match** — Does the assessment accurately
   measure the stated objective?
3. **Activity-to-Objective Match** — Do guided and independent practice
   activities build the exact skill outlined in the objective?
4. **Pacing Realism** — Do activity time estimates sum to roughly the
   stated lesson duration, accounting for transitions?

Usage
-----
>>> from agents.review_agent import review_lesson_plan
>>> result = review_lesson_plan(
...     draft="# Lesson Plan: Main Idea ...",
...     standard_code="ELA.5.R.1.2",
...     standard_text="Explain how relevant details support...",
...     duration="45 minutes",
... )
>>> result["is_approved"]
True
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ==============================================================================
# Configuration
# ==============================================================================
load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
# Low temperature for strict, consistent evaluation
DEFAULT_TEMPERATURE: float = 0.2


# ==============================================================================
# System Prompt
# ==============================================================================

REVIEW_SYSTEM_PROMPT: str = """\
You are a rigorous K-12 curriculum alignment reviewer.  Your ONLY job
is to evaluate a draft lesson plan against a confirmed state standard
using a strict four-point rubric.

## RUBRIC (evaluate each criterion independently)

### 1. Standards Alignment
Does the lesson's stated **Objective** use the EXACT cognitive verb
(e.g., "analyze", "explain", "determine") and the EXACT skill/concept
described in the confirmed standard?
- PASS: The objective mirrors the standard's verb and cognitive
  complexity precisely.
- FAIL: The objective uses a different verb, a weaker/stronger
  synonym, or omits a key component of the standard.

### 2. Objective-to-Assessment Match
Does the **Assessment** (e.g., exit ticket) directly and accurately
measure the skill stated in the **Objective**?
- PASS: A student who masters the objective would succeed on the
  assessment, and vice versa.
- FAIL: The assessment tests a different skill, tests at a
  different cognitive level, or only partially covers the objective.

### 3. Activity-to-Objective Match
Do the **Guided Practice** and **Independent Practice** activities
build the EXACT skill described in the **Objective**?
- PASS: Both activities practice the target skill progressively.
- FAIL: Activities are topically related but do not practice the
  specific skill (e.g., the objective requires "analyze" but the
  activity only asks students to "identify").

### 4. Pacing Realism
Do the time estimates for all sections (Hook, Direct Instruction,
Guided Practice, Independent Practice, Assessment) sum to
approximately the stated lesson duration (within +/- 5 minutes),
accounting for 2-3 minutes of transition time?
- PASS: Times are realistic and sum correctly.
- FAIL: Times don't add up, or individual segments are unrealistically
  short/long for the described activity.

## OUTPUT FORMAT
You MUST respond with ONLY a valid JSON object (no markdown fences,
no commentary). Use this exact schema:

{
  "is_approved": <bool>,
  "criteria": [
    {
      "criterion": "Standards Alignment",
      "passed": <bool>,
      "reason": "<empty string if passed, specific actionable feedback if failed>"
    },
    {
      "criterion": "Objective-to-Assessment Match",
      "passed": <bool>,
      "reason": "<...>"
    },
    {
      "criterion": "Activity-to-Objective Match",
      "passed": <bool>,
      "reason": "<...>"
    },
    {
      "criterion": "Pacing Realism",
      "passed": <bool>,
      "reason": "<...>"
    }
  ],
  "failed_criteria": ["<names of failed criteria, empty list if all pass>"],
  "specific_feedback": "<One paragraph summary of all issues, or 'All checks passed.' if approved>"
}

## RULES
- "is_approved" is true ONLY if ALL four criteria pass.
- Every failed criterion MUST have a non-empty "reason" with a
  specific, actionable explanation.
- Be strict but fair.  Do not fail a criterion on stylistic
  preferences — only on genuine alignment gaps.
- Do NOT suggest improvements for passing criteria.
"""


# ==============================================================================
# Public API
# ==============================================================================

def review_lesson_plan(
    *,
    draft: str,
    standard_code: str,
    standard_text: str,
    duration: str,
    objective: str = "",
    model: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Evaluate a draft lesson plan against the alignment rubric.

    Parameters
    ----------
    draft : str
        The full draft lesson plan (Markdown).
    standard_code : str
        Confirmed standard code (e.g., "ELA.5.R.1.2").
    standard_text : str
        Full text of the confirmed standard.
    duration : str
        Stated lesson duration (e.g., "45 minutes").
    objective : str, optional
        If provided, the lesson objective is highlighted separately
        for the reviewer.  Otherwise extracted from the draft.
    model : str, optional
        OpenAI model override.
    temperature : float, optional
        Sampling temperature override.

    Returns
    -------
    dict[str, Any]
        Structured review with keys:
        - ``is_approved`` (bool)
        - ``criteria`` (list[dict]) — per-criterion results
        - ``failed_criteria`` (list[str]) — names of failed criteria
        - ``specific_feedback`` (str) — summary paragraph

    Raises
    ------
    ValueError
        If ``draft`` or ``standard_text`` is empty.
    RuntimeError
        If the LLM call fails or returns unparseable output.
    """
    # ── Validate inputs ──────────────────────────────────────────────
    if not draft.strip():
        raise ValueError("Cannot review an empty lesson plan draft.")
    if not standard_text.strip():
        raise ValueError("Cannot review without the confirmed standard text.")

    # ── Build the user prompt ────────────────────────────────────────
    user_prompt: str = _build_review_prompt(
        draft=draft,
        standard_code=standard_code,
        standard_text=standard_text,
        duration=duration,
        objective=objective,
    )

    # ── Call the LLM ─────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
    )

    logger.info(
        "review_agent: Evaluating lesson plan against rubric "
        "(standard=%s).",
        standard_code,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=REVIEW_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
    except Exception as exc:
        logger.error("review_agent: LLM call failed — %s", exc)
        raise RuntimeError(
            f"Alignment review failed: {exc}"
        ) from exc

    # ── Parse the structured JSON response ───────────────────────────
    raw_output: str = response.content.strip()
    result: dict[str, Any] = _parse_review_output(raw_output)

    logger.info(
        "review_agent: Review complete — approved=%s, failed=%s.",
        result["is_approved"],
        result.get("failed_criteria", []),
    )

    return result


# ==============================================================================
# Private helpers
# ==============================================================================

def _build_review_prompt(
    *,
    draft: str,
    standard_code: str,
    standard_text: str,
    duration: str,
    objective: str,
) -> str:
    """Assemble the review prompt with all context the evaluator needs."""
    sections: list[str] = [
        "## Confirmed Standard",
        f"**Code:** {standard_code}",
        f"**Full Text:** {standard_text}",
        "",
        f"## Stated Lesson Duration",
        f"{duration or 'Not specified'}",
    ]

    if objective.strip():
        sections.extend([
            "",
            "## Extracted Objective (for reference)",
            objective,
        ])

    sections.extend([
        "",
        "## Draft Lesson Plan to Evaluate",
        "```markdown",
        draft,
        "```",
        "",
        "---",
        "Evaluate this lesson plan against the four-point rubric. "
        "Return ONLY the JSON object.",
    ])

    return "\n".join(sections)


def _parse_review_output(raw: str) -> dict[str, Any]:
    """Parse the LLM's JSON response, handling common formatting issues.

    The LLM sometimes wraps JSON in markdown code fences or adds
    trailing commentary.  This function strips those artifacts before
    parsing.

    Parameters
    ----------
    raw : str
        Raw LLM output.

    Returns
    -------
    dict[str, Any]
        Parsed review result.

    Raises
    ------
    RuntimeError
        If the output cannot be parsed as valid JSON.
    """
    # Strip markdown code fences if present
    cleaned: str = raw
    if "```" in cleaned:
        # Extract content between first pair of fences
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)

    # Try to find a JSON object in the output
    cleaned = cleaned.strip()
    if not cleaned.startswith("{"):
        # Try to find the first { ... } block
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            cleaned = cleaned[start:end]

    try:
        result: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(
            "review_agent: Failed to parse LLM output as JSON.\n"
            "Raw output:\n%s",
            raw,
        )
        raise RuntimeError(
            f"Review agent returned unparseable output: {exc}\n"
            f"Raw: {raw[:500]}"
        ) from exc

    # ── Validate required keys ───────────────────────────────────────
    if "is_approved" not in result:
        # Derive from criteria if present
        criteria = result.get("criteria", [])
        result["is_approved"] = all(c.get("passed", False) for c in criteria)

    if "failed_criteria" not in result:
        result["failed_criteria"] = [
            c["criterion"]
            for c in result.get("criteria", [])
            if not c.get("passed", True)
        ]

    if "specific_feedback" not in result:
        result["specific_feedback"] = "See individual criteria for details."

    if "criteria" not in result:
        result["criteria"] = []

    return result
