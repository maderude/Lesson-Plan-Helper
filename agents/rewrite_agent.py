# Requires: pip install langchain-google-genai
"""
agents/rewrite_agent.py — Targeted Lesson Plan Rewrite Agent
=============================================================

Takes a draft lesson plan and specific feedback from the Review Agent,
and rewrites ONLY the flagged sections while leaving the rest of the
lesson intact.

The rewrite is constrained by the same alignment rules as the original
generation: the standard's exact verb and cognitive complexity must be
preserved, and time allocations must remain realistic.

Usage
-----
>>> from agents.rewrite_agent import rewrite_lesson_plan
>>> revised = rewrite_lesson_plan(
...     draft="# Lesson Plan: Main Idea ...",
...     standard_code="ELA.5.R.1.2",
...     standard_text="Explain how relevant details support...",
...     duration="45 minutes",
...     review_feedback=[
...         {
...             "criterion": "Objective-to-Assessment Match",
...             "passed": False,
...             "reason": "Assessment measures recall, not the analysis skill."
...         }
...     ],
... )
"""

from __future__ import annotations

import logging
import os
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
# Slightly lower temperature than planning for focused revisions
DEFAULT_TEMPERATURE: float = 0.5


# ==============================================================================
# System Prompt
# ==============================================================================

REWRITE_SYSTEM_PROMPT: str = """\
You are an expert K-12 curriculum revision specialist.  You are given
a draft lesson plan that FAILED one or more alignment review criteria,
along with the specific failure reasons.

## YOUR TASK
You must return the entire, complete lesson plan from start to finish.
For any sections that did not fail the review, copy their content word-for-word exactly as they are in the current draft.
For the sections that failed, rewrite them to address the feedback.

## RULES
1. **Preserve the overall structure.** The output must use the same
   Markdown heading structure as the input.
2. **Fix ONLY what's broken.** If the reviewer flagged "Assessment"
   but not "Hook", return the Hook section word-for-word unchanged.
3. **Output Completeness.** You MUST output the COMPLETE lesson plan including all unchanged sections. Do NOT truncate the output, do NOT omit unchanged sections, and do NOT return only the revised sections. DO NOT generate empty headers—you must preserve or generate the full content beneath each header.
4. **Maintain standard alignment.** Any rewritten section must use
   the EXACT cognitive verb and skill from the confirmed standard.
5. **Keep time allocations realistic.** If you change a section's
   content, verify its time estimate is still reasonable.  If Pacing
   was flagged, adjust times so they sum to the stated duration.
6. **Enforce Formatting Constraints.** If you rewrite the **Assessment**
   section, provide ONLY the question text (NO "exit ticket", NO "prompt").
   If you rewrite **Differentiation**, use plain labels (NO markdown bold, NO bullets).
   You MUST use EXACTLY these Markdown section headers if you rewrite or add them:
   ## Essential Question, ## Objective, ## Instructional Materials, 
   ## Teaching Strategies, ## Hook, ## Direct Instruction, ## Guided Practice, 
   ## Independent Practice, ## Assessment, ## Assignments, ## Homework Notes,
   ## Differentiation, ## Teacher Reflection.
   If you must add "Instructional Materials", provide a bulleted list of resources.
   If you must add "Teaching Strategies", list 2-3 specific pedagogical strategies.
7. **Do NOT add commentary.** Return ONLY the complete revised
   lesson plan in Markdown.  No preamble, no "Here's the revised
   version", no explanation of changes.

## HOW TO READ THE FEEDBACK
Each failed criterion has a "reason" field explaining exactly what
went wrong.  Use this to make a targeted fix.  Examples:
- "Objective uses 'identify' but the standard requires 'analyze'"
  → Rewrite only the Objective section with the correct verb.
- "Assessment measures recall, not the analysis skill"
  → Rewrite only the Assessment section.
- "Guided Practice asks students to list, not explain relationships"
  → Rewrite only the Guided Practice section.
"""


# ==============================================================================
# Public API
# ==============================================================================

def rewrite_lesson_plan(
    *,
    draft: str,
    standard_code: str,
    standard_text: str,
    duration: str,
    review_feedback: list[dict[str, Any]],
    model: str = "gpt-4o",
    temperature: float = 0.4,
) -> str:
    """Revise a lesson plan based on specific rubric failures.

    Parameters
    ----------
    draft : str
        The full draft lesson plan (Markdown) to revise.
    standard_code : str
        Confirmed standard code.
    standard_text : str
        Full text of the confirmed standard.
    duration : str
        Stated lesson duration.
    review_feedback : list[dict[str, Any]]
        List of rubric check results from the Review Agent.
        Each dict has keys: ``criterion``, ``passed``, ``reason``.
        Only items where ``passed`` is ``False`` will be addressed.
    model : str, optional
        OpenAI model override.
    temperature : float, optional
        Sampling temperature override.

    Returns
    -------
    str
        The revised lesson plan as Markdown.  Sections that were not
        flagged remain unchanged.

    Raises
    ------
    ValueError
        If ``draft`` is empty or no failed criteria are found.
    RuntimeError
        If the LLM call fails.
    """
    # ── Validate inputs ──────────────────────────────────────────────
    if not draft.strip():
        raise ValueError("Cannot rewrite an empty lesson plan draft.")

    # Extract only the failed criteria
    failed: list[dict[str, Any]] = [
        f for f in review_feedback
        if not f.get("passed", True)
    ]

    if not failed:
        logger.info(
            "rewrite_agent: No failed criteria — returning draft unchanged."
        )
        return draft

    user_prompt: str = _build_rewrite_prompt(
        draft=draft,
        standard_code=standard_code,
        standard_text=standard_text,
        duration=duration,
        failed_criteria=failed,
    )

    # ── Call the LLM ─────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
    )

    failed_names: list[str] = [f["criterion"] for f in failed]
    logger.info(
        "rewrite_agent: Revising lesson plan.  Failed criteria: %s.",
        failed_names,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
    except Exception as exc:
        logger.error("rewrite_agent: LLM call failed — %s", exc)
        raise RuntimeError(
            f"Lesson plan rewrite failed: {exc}"
        ) from exc

    revised_plan: str = response.content.strip()

    # Strip markdown code fences if the LLM wrapped the output
    if revised_plan.startswith("```"):
        lines = revised_plan.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        revised_plan = "\n".join(lines)

    logger.info(
        "rewrite_agent: Revision complete — %d characters "
        "(original: %d characters).",
        len(revised_plan),
        len(draft),
    )

    return revised_plan


# ==============================================================================
# Private helpers
# ==============================================================================

def _build_rewrite_prompt(
    *,
    draft: str,
    standard_code: str,
    standard_text: str,
    duration: str,
    failed_criteria: list[dict[str, Any]],
) -> str:
    """Assemble the rewrite prompt with the draft and failure details."""
    sections: list[str] = [
        "## Confirmed Standard",
        f"**Code:** {standard_code}",
        f"**Full Text:** {standard_text}",
        "",
        f"## Stated Duration: {duration}",
        "",
        "## Review Failures (fix ONLY these)",
    ]

    for i, failure in enumerate(failed_criteria, start=1):
        criterion: str = failure.get("criterion", "Unknown")
        reason: str = failure.get("reason", "No specific reason provided.")
        sections.append(f"### Failure {i}: {criterion}")
        sections.append(f"**Reason:** {reason}")
        sections.append("")

    sections.extend([
        "## Current Draft Lesson Plan",
        "```markdown",
        draft,
        "```",
        "",
        "---",
        "Rewrite the lesson plan, fixing ONLY the sections related to "
        "the failures above.  Leave all other sections unchanged.  "
        "Return the COMPLETE revised lesson plan in Markdown.",
    ])

    return "\n".join(sections)
