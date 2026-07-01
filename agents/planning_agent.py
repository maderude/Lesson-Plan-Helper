"""
agents/planning_agent.py — Lesson Plan Generation Agent
========================================================

Takes the teacher's confirmed lesson context and the approved standard
to generate a structured, standards-aligned daily lesson plan via an
LLM call.

The output is Markdown with mandatory sections:
    Objective, Essential Question, Instructional Materials,
    Teaching Strategies, Hook, Direct Instruction, Guided Practice,
    Independent Practice, Assessment, and Differentiation.

Usage
-----
>>> from agents.planning_agent import generate_lesson_plan
>>> plan = generate_lesson_plan(
...     topic="main idea and supporting details",
...     grade="5",
...     subject="ELA",
...     duration="45 minutes",
...     standard_code="ELA.5.R.1.2",
...     standard_text="Explain how relevant details support the central idea...",
...     accommodations="2 ELL students; 1 student with IEP for extended time",
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
DEFAULT_TEMPERATURE: float = 0.7  # Creative enough for engaging lessons


# ==============================================================================
# System Prompt
# ==============================================================================

PLANNING_SYSTEM_PROMPT: str = """\
You are an expert K-12 instructional designer and curriculum specialist.
Your job is to generate a structured, standards-aligned daily lesson plan
that is classroom-ready.

## ABSOLUTE RULES
1. The lesson objective MUST use the EXACT cognitive verb and skill
   described in the provided standard. Do not paraphrase or weaken it.
2. Every section must directly serve the stated objective.
3. Time allocations MUST sum to approximately the stated lesson duration,
   accounting for 2-3 minutes of transition time between segments.
4. The assessment MUST measure the EXACT skill in the objective — not a
   related or simplified skill.
5. Differentiation must address the specific accommodations provided.
6. You MUST generate ALL sections below in this exact order:
   ## Objective, ## Essential Question, ## Instructional Materials,
   ## Teaching Strategies, ## Hook, ## Direct Instruction,
   ## Guided Practice, ## Independent Practice, ## Assessment,
   ## Differentiation, ## Assignments, ## Homework Notes, ## Teacher Reflection.
   Do NOT skip any section. Do NOT leave any section empty.

## OUTPUT FORMAT (Markdown)
Return ONLY the lesson plan in the following Markdown structure.
Do NOT include any preamble, commentary, or explanation outside this
structure.

# Lesson Plan: [Topic]

**Date:** [date]
**Grade:** [grade] | **Subject:** [subject] | **Duration:** [duration]
**Standard:** [code] — [full standard text]

## Objective
[One measurable objective using the standard's exact verb and skill]

## Essential Question
[A single, open-ended driving question that frames the lesson for
students, sparks curiosity, and connects to the standard. It should be
grade-appropriate, debatable or thought-provoking, and answerable by
the end of the lesson.]

## Instructional Materials
[Generate a bulleted list of all materials needed for this lesson. You MUST include:
- Primary text or reading passage (name it specifically based on the topic)
- Any handouts or graphic organizers students will use
- Teacher display tools (projector, whiteboard, anchor chart, etc.)
- Any student tech required (Chromebooks, tablets, etc.)
Do NOT write "None" or leave this section vague.]

## Teaching Strategies
[List exactly 2-3 instructional strategies used in this lesson. Format EACH on its own
line exactly like this — Strategy Name: one sentence describing how it is applied in this lesson.
Choose from: Gradual Release of Responsibility, Think-Pair-Share, Think-Aloud Modeling,
Socratic Seminar, Jigsaw, Gallery Walk, Close Reading, Annotation, Graphic Organizer, etc.]

## Hook ([X] min)
[An engaging opening activity that activates prior knowledge and
connects to the topic. Be specific — name the activity, not just
"discuss" or "review."]

## Direct Instruction ([X] min)
[Step-by-step teacher-led instruction. Include what the teacher says,
models, or demonstrates. Reference specific examples or texts.]

## Guided Practice ([X] min)
[A structured collaborative activity where students practice the skill
with teacher support. Include specific directions and expected outputs.]

## Independent Practice ([X] min)
[A task students complete on their own to demonstrate the skill.
Include clear directions and success criteria.]

## Assessment ([X] min)
Students will respond to the following question:
[Write ONLY the exact question students will answer. The question must
directly measure the skill in the objective. Provide NO introductory text,
NO labels, NO "exit ticket", and NO "prompt" wording. Just the question itself.]

## Differentiation
[Specific modifications for the accommodations listed. Format each
accommodation group on its own line exactly like this (NO markdown bold `**`, NO bullets `-`):]

ELL Students: [specific supports — sentence frames, visual aids,
vocabulary scaffolds, etc.]

Students with IEP: [specific modifications — extended time,
simplified directions, alternative assessments, etc.]

[Add additional groups as needed based on the accommodations provided.]
## Assignments
[Describe any in-class work products, deliverables, or formative tasks students will complete during the lesson.]

## Homework Notes
[Provide actionable, family-friendly tasks for home that extend the lesson without requiring new instruction.]

## Teacher Reflection
[Generate 1-2 reflection questions or prompts to help the teacher evaluate the success of the lesson after instruction.]
"""


# ==============================================================================
# Public API
# ==============================================================================

def generate_lesson_plan(
    *,
    topic: str,
    grade: str,
    subject: str,
    duration: str,
    standard_code: str,
    standard_text: str,
    lesson_date: str = "",
    syllabus_text: str = "",
    materials: str = "",
    accommodations: str = "",
    teacher_notes: str = "",
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Generate a structured lesson plan grounded in the confirmed standard.

    Parameters
    ----------
    topic : str
        The lesson topic (e.g., "main idea and supporting details").
    grade : str
        Grade level (e.g., "5").
    subject : str
        Subject area (e.g., "ELA").
    duration : str
        Lesson duration (e.g., "45 minutes").
    standard_code : str
        Confirmed standard code (e.g., "ELA.5.R.1.2").
    standard_text : str
        Full text of the confirmed standard.
    lesson_date : str, optional
        Date the lesson will be taught.
    syllabus_text : str, optional
        Relevant syllabus context.
    materials : str, optional
        Available materials/resources.
    accommodations : str, optional
        Student accommodations or differentiation notes.
    teacher_notes : str, optional
        Additional teacher notes.
    model : str, optional
        OpenAI model override.  Defaults to ``OPENAI_MODEL`` env var
        or ``"gpt-4o"``.
    temperature : float, optional
        Sampling temperature override.

    Returns
    -------
    str
        The generated lesson plan as Markdown text.

    Raises
    ------
    ValueError
        If any of the four critical inputs (topic, grade, subject,
        standard_text) are empty.
    RuntimeError
        If the LLM call fails.
    """
    # ── Validate critical inputs ─────────────────────────────────────
    _validate_critical_inputs(
        topic=topic,
        grade=grade,
        subject=subject,
        standard_text=standard_text,
    )

    # ── Build the user prompt ────────────────────────────────────────
    user_prompt: str = _build_user_prompt(
        topic=topic,
        grade=grade,
        subject=subject,
        duration=duration,
        standard_code=standard_code,
        standard_text=standard_text,
        lesson_date=lesson_date,
        syllabus_text=syllabus_text,
        materials=materials,
        accommodations=accommodations,
        teacher_notes=teacher_notes,
    )

    # ── Call the LLM ─────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
    )

    logger.info(
        "planning_agent: Generating lesson plan for '%s' (grade %s, %s) "
        "aligned to %s.",
        topic,
        grade,
        subject,
        standard_code,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
    except Exception as exc:
        logger.error("planning_agent: LLM call failed — %s", exc)
        raise RuntimeError(
            f"Lesson plan generation failed: {exc}"
        ) from exc

    lesson_plan: str = response.content.strip()

    logger.info(
        "planning_agent: Generated %d-character lesson plan.",
        len(lesson_plan),
    )

    return lesson_plan


# ==============================================================================
# Private helpers
# ==============================================================================

def _validate_critical_inputs(
    *,
    topic: str,
    grade: str,
    subject: str,
    standard_text: str,
) -> None:
    """Raise ``ValueError`` if any critical input is empty."""
    missing: list[str] = []
    if not topic.strip():
        missing.append("topic")
    if not grade.strip():
        missing.append("grade")
    if not subject.strip():
        missing.append("subject")
    if not standard_text.strip():
        missing.append("standard_text")
    if missing:
        raise ValueError(
            f"Cannot generate lesson plan — missing: {', '.join(missing)}"
        )


def _build_user_prompt(
    *,
    topic: str,
    grade: str,
    subject: str,
    duration: str,
    standard_code: str,
    standard_text: str,
    lesson_date: str,
    syllabus_text: str,
    materials: str,
    accommodations: str,
    teacher_notes: str,
) -> str:
    """Assemble the user-facing prompt with all lesson context."""
    sections: list[str] = [
        f"## Lesson Context",
        f"- **Topic:** {topic}",
        f"- **Grade:** {grade}",
        f"- **Subject:** {subject}",
        f"- **Duration:** {duration or 'Not specified (assume 45 minutes)'}",
        f"- **Date:** {lesson_date or 'Not specified'}",
        "",
        f"## Confirmed Standard",
        f"- **Code:** {standard_code}",
        f"- **Full Text:** {standard_text}",
    ]

    if syllabus_text.strip():
        sections.extend(["", f"## Syllabus Context", syllabus_text])

    if materials.strip():
        sections.extend(["", f"## Available Materials", materials])

    if accommodations.strip():
        sections.extend([
            "",
            f"## Student Accommodations (MUST address in Differentiation)",
            accommodations,
        ])
    else:
        sections.extend([
            "",
            "## Student Accommodations",
            "None specified. Provide general differentiation for mixed-ability "
            "classrooms (struggling, on-level, advanced).",
        ])

    if teacher_notes.strip():
        sections.extend(["", f"## Teacher Notes", teacher_notes])

    sections.extend([
        "",
        "---",
        "Generate the complete lesson plan now, following the exact "
        "Markdown structure from your instructions.",
    ])

    return "\n".join(sections)
