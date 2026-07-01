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
# Deterministic Pacing Check
# ==============================================================================

def _check_pacing_deterministic(draft: str, duration: str) -> dict[str, Any]:
    """Extract time annotations from ## headers and sum them deterministically.

    Parses lines like '## Hook (5 min)', '## Direct Instruction (10 minutes)',
    etc. and compares the total to the requested lesson duration.

    Returns a dict with keys: criterion, passed, reason.
    """
    # Extract requested duration in minutes
    dur_match = re.search(r"(\d+)", duration or "")
    requested_minutes = int(dur_match.group(1)) if dur_match else 45

    # Extract all (X min) annotations from ## headers
    time_pattern = re.compile(
        r"^##\s+.+?\(\s*(\d+)\s*min(?:utes?)?\s*\)",
        re.IGNORECASE | re.MULTILINE,
    )
    found_times: list[tuple[str, int]] = []
    for match in time_pattern.finditer(draft):
        line = match.group(0)
        minutes = int(match.group(1))
        # Extract the section name for logging
        section = re.sub(r"\s*\(\d+\s*min(?:utes?)?\)", "", line.replace("## ", "")).strip()
        found_times.append((section, minutes))

    if not found_times:
        # No time annotations found — can't check deterministically
        return {
            "criterion": "Pacing Realism",
            "passed": True,
            "reason": "No time annotations found in headers; skipping deterministic check.",
        }

    total = sum(m for _, m in found_times)
    diff = total - requested_minutes
    breakdown = ", ".join(f"{name}: {mins}min" for name, mins in found_times)

    if abs(diff) <= 5:
        return {
            "criterion": "Pacing Realism",
            "passed": True,
            "reason": (
                f"Total pacing: {total} min (requested: {requested_minutes} min). "
                f"Breakdown: {breakdown}. Within ±5 min tolerance."
            ),
        }
    else:
        over_under = "over" if diff > 0 else "under"
        return {
            "criterion": "Pacing Realism",
            "passed": False,
            "reason": (
                f"PACING FAILED: Total is {total} min but lesson duration is "
                f"{requested_minutes} min ({abs(diff)} min {over_under}). "
                f"Breakdown: {breakdown}. "
                f"Adjust section times so they sum to within ±5 min of {requested_minutes}."
            ),
        }


# ==============================================================================
# System Prompt
# ==============================================================================

REVIEW_SYSTEM_PROMPT: str = """\
You are a rigorous K-12 curriculum alignment reviewer.  Your ONLY job
is to evaluate a draft lesson plan against a confirmed state standard
using a strict four-point rubric.

## RUBRIC (evaluate each criterion independently)

### 1. Standards Alignment
Does the lesson's stated **Objective** align with the core cognitive skill
and concept described in the confirmed standard?
- PASS: The objective reflects the standard's core skill and cognitive level.
  Synonymous verbs are acceptable (e.g., "interpret" for "determine",
  "analyze" for "examine"). Minor paraphrasing is acceptable.
- FAIL: The objective tests a COMPLETELY different skill or is at a
  drastically LOWER cognitive level (e.g., "recall" when the standard
  requires "evaluate").

### 2. Objective-to-Assessment Match
Does the **Assessment** (e.g., exit ticket) directly and accurately
measure the skill stated in the **Objective**?
- PASS: Any assessment format (written, verbal, visual, gestural) that directly measures the objective skill. Format alone is NOT grounds to fail.
- FAIL: Only if the assessment measures a completely different SKILL — not a different FORMAT.

### 3. Activity-to-Objective Match
Do the **Guided Practice** and **Independent Practice** activities
build the skill described in the **Objective**?
- PASS: The PRIMARY task in the activity requires students to practice the
  TARGET VERB from the objective at least once. Activities MAY include
  supplementary content, spiral review of prior skills, or cross-curricular
  connections — this is normal, good teaching and is NOT grounds to fail.
  Only evaluate whether the MAIN task aligns with the objective.
  *Note: Do NOT evaluate the Hook or Direct Instruction sections under
  this criterion. ONLY Guided Practice and Independent Practice.*
- FAIL: ONLY if the activity's PRIMARY task requires a completely
  different or drastically lower-order skill than the objective
  (e.g., students only "list" when objective requires "evaluate",
  and there is no higher-order task in the activity at all).


### 4. Pacing Realism
Do the time estimates for all sections (Hook, Direct Instruction,
Guided Practice, Independent Practice, Assessment) sum to
approximately the stated lesson duration (within +/- 5 minutes),
accounting for 2-3 minutes of transition time?
- PASS: Times are realistic and sum correctly.
- FAIL: Times don't add up, or individual segments are unrealistically
  short/long for the described activity.

## OUTPUT FORMAT
You MUST respond with ONLY a valid JSON object. Use this EXACT schema:

{
  "is_approved": true,
  "criteria": [
    {"criterion": "Standards Alignment", "passed": true, "reason": ""},
    {"criterion": "Objective-to-Assessment Match", "passed": true, "reason": ""},
    {"criterion": "Activity-to-Objective Match", "passed": false, "reason": "Specific reason here."},
    {"criterion": "Pacing Realism", "passed": true, "reason": ""}
  ],
  "failed_criteria": ["Activity-to-Objective Match"],
  "specific_feedback": "Summary of issues."
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

    # ── Override LLM pacing with deterministic math ──────────────────
    det_pacing = _check_pacing_deterministic(draft, duration)
    criteria = result.get("criteria", [])
    # Replace the LLM's Pacing Realism criterion with our deterministic one
    pacing_replaced = False
    for i, c in enumerate(criteria):
        if "pacing" in c.get("criterion", "").lower():
            criteria[i] = det_pacing
            pacing_replaced = True
            break
    if not pacing_replaced:
        criteria.append(det_pacing)
    result["criteria"] = criteria

    # Recompute is_approved after override
    result["is_approved"] = all(c.get("passed", False) for c in criteria)
    result["failed_criteria"] = [
        c["criterion"] for c in criteria if not c.get("passed", True)
    ]

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
    criteria = result.get("criteria", [])
    if criteria:
        # Strictly derive is_approved from the criteria to prevent LLM hallucinations
        # where it says is_approved=True but fails individual criteria.
        result["is_approved"] = all(c.get("passed", False) for c in criteria)
    elif "is_approved" not in result:
        result["is_approved"] = False

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
