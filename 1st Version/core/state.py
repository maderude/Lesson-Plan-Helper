"""
core/state.py — Workflow State Schema
======================================

Defines the data structures that flow between agents in the Lesson
Plan Helper multi-agent workflow.

Two complementary representations are provided:

1.  **Pydantic models** (``LessonInputs``, ``StandardResult``,
    ``RubricCheck``, ``LessonPlanDocument``) — used for input
    validation, serialisation, and API boundaries.

2.  **TypedDict** (``LessonPlanState``) — the canonical LangGraph
    state schema.  Every node reads from and writes to this dict.
    Fields use plain built-in types so LangGraph can diff/merge
    state updates without custom reducers.

Extending the state
-------------------
Add new keys to ``LessonPlanState`` and, if the field represents a
structured object, create a matching Pydantic model for validation
at the edges of the system (API ingress, final output, etc.).
"""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PYDANTIC MODELS — Validation & Serialisation                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ==============================================================================
# Teacher Inputs
# ==============================================================================

class LessonInputs(BaseModel):
    """Validated teacher-provided lesson context.

    All string fields default to empty strings so that the Intake
    Agent can report *which* fields are missing rather than raising
    a Pydantic ``ValidationError`` at the boundary.
    """

    lesson_date: str = Field(
        default="",
        description="Date the lesson will be taught (e.g. '2025-09-15').",
    )
    subject: str = Field(
        default="",
        description="Subject area (e.g. 'ELA', 'Math', 'Science').",
    )
    grade: str = Field(
        default="",
        description="Grade level as a string (e.g. '5', 'K', '11').",
    )
    duration: str = Field(
        default="",
        description="Lesson duration (e.g. '45 minutes', '1 hour').",
    )
    topic: str = Field(
        default="",
        description=(
            "Free-text topic the teacher intends to teach "
            "(e.g. 'main idea and supporting details')."
        ),
    )
    state: str = Field(
        default="",
        description=(
            "U.S. state whose standards framework should be used "
            "(e.g. 'Florida', 'TX').  Falls back to Common Core."
        ),
    )
    syllabus_text: str = Field(
        default="",
        description="Relevant syllabus excerpt or unit context.",
    )
    materials: str = Field(
        default="",
        description="Materials and resources available for the lesson.",
    )
    accommodations: str = Field(
        default="",
        description="Student accommodations or differentiation notes.",
    )
    teacher_notes: str = Field(
        default="",
        description="Any additional notes from the teacher.",
    )


# ==============================================================================
# Standard Result (from the retriever)
# ==============================================================================

class StandardResult(BaseModel):
    """A single standard returned by ``data.retriever.get_standards()``."""

    code: str = Field(
        ...,
        description="Official standard code (e.g. 'RI.5.1').",
    )
    description: str = Field(
        ...,
        description="Full text of the standard.",
    )
    strand: str = Field(
        default="",
        description="Curricular strand (e.g. 'Reading: Informational Text').",
    )
    source: str = Field(
        default="",
        description="Corpus key (e.g. 'florida', 'common_core').",
    )
    fallback: bool = Field(
        default=False,
        description="True if Common Core was used as a fallback.",
    )
    score: int = Field(
        default=0,
        description="Keyword-overlap relevance score.",
    )


# ==============================================================================
# Alignment Review — Rubric Check
# ==============================================================================

class RubricCheck(BaseModel):
    """Result of a single criterion in the Alignment Review rubric.

    The Alignment Review Agent evaluates four criteria.  Each produces
    one ``RubricCheck`` with a pass/fail verdict and, on failure, an
    actionable reason string.
    """

    criterion: str = Field(
        ...,
        description=(
            "Name of the rubric criterion.  One of: "
            "'Standard Match', 'Objective → Assessment Match', "
            "'Activity → Objective Match', 'Pacing Realism'."
        ),
    )
    passed: bool = Field(
        ...,
        description="True if the lesson satisfies this criterion.",
    )
    reason: str = Field(
        default="",
        description=(
            "If ``passed`` is False, a specific, actionable explanation "
            "of why the criterion failed (e.g. 'Assessment measures "
            "recall, not the analysis skill required by the standard')."
        ),
    )


# ==============================================================================
# Lesson Plan Document
# ==============================================================================

class LessonPlanDocument(BaseModel):
    """Structured representation of the final lesson plan output.

    Each section maps to a part of the classroom-ready document
    described in the project write-up.
    """

    objective: str = Field(default="", description="Measurable lesson objective.")
    standard_code: str = Field(default="", description="Aligned standard code.")
    standard_text: str = Field(default="", description="Full standard text.")
    hook: str = Field(default="", description="Opening hook / anticipatory set.")
    direct_instruction: str = Field(default="", description="Direct instruction segment.")
    guided_practice: str = Field(default="", description="Guided practice activity.")
    independent_practice: str = Field(default="", description="Independent practice activity.")
    assessment: str = Field(default="", description="Formative or summative assessment.")
    differentiation: str = Field(default="", description="Differentiation / accommodations.")
    reflection_prompt: str = Field(default="", description="Teacher reflection prompt.")
    pacing_notes: str = Field(default="", description="Pacing breakdown with time estimates.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LANGGRAPH STATE — TypedDict                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class LessonPlanState(TypedDict, total=False):
    """Canonical state schema for the LangGraph workflow.

    Every node receives this dict and returns a *partial* update dict
    containing only the keys it modifies.  LangGraph merges the update
    into the running state automatically.

    ``total=False`` makes every key optional so that:
      • The graph can be initialised with only the teacher inputs.
      • Each node only declares the keys it actually writes.

    Key Groups
    ----------
    - **Teacher Inputs** — populated at graph invocation.
    - **Intake Validation** — set by ``intake_node``.
    - **Standards Discovery** — set by ``discovery_node``.
    - **Human Confirmation** — set after the human-in-the-loop step.
    - **Lesson Generation** — set by ``planning_node``.
    - **Alignment Review** — set by ``review_node``.
    - **Revision Control** — tracks the rewrite loop.
    - **Final Output** — set when the review passes.
    - **Workflow Metadata** — error tracking and status.
    """

    # ── Teacher Inputs ──────────────────────────────────────────────────
    lesson_date: str
    subject: str
    grade: str
    duration: str
    topic: str
    state: str
    syllabus_text: str
    materials: str
    accommodations: str
    teacher_notes: str

    # ── Intake Validation ───────────────────────────────────────────────
    intake_valid: bool
    intake_errors: list[str]

    # ── Standards Discovery ─────────────────────────────────────────────
    retrieved_standards: list[dict[str, Any]]  # raw dicts from retriever
    selected_standard_code: str
    selected_standard_text: str
    selected_standard_strand: str
    selected_standard_source: str
    is_fallback: bool

    # ── Human Confirmation ──────────────────────────────────────────────
    standard_approved: bool

    # ── Lesson Generation ───────────────────────────────────────────────
    draft_lesson_plan: str

    # ── Alignment Review ────────────────────────────────────────────────
    review_passed: bool
    review_feedback: list[dict[str, Any]]  # serialised RubricCheck dicts

    # ── Revision Control ────────────────────────────────────────────────
    revision_count: int
    max_revisions: int

    # ── Final Output ────────────────────────────────────────────────────
    final_lesson_plan: str

    # ── Workflow Metadata ───────────────────────────────────────────────
    workflow_status: str   # "running" | "completed" | "failed"
    error_message: str
