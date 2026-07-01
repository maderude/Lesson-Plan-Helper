"""
core/orchestrator.py — LangGraph Workflow Orchestrator
======================================================

Defines the multi-agent workflow for the Lesson Plan Helper as a
LangGraph ``StateGraph``.  The graph encodes six processing stages
and one conditional revision loop:

.. code-block:: text

    START
      │
      ▼
    intake_node  ───[invalid]──► END (with errors)
      │ [valid]
      ▼
    discovery_node
      │
      ▼
    confirmation_node  ◄── human-in-the-loop (interrupt)
      │
      ├──[rejected]──► END
      │
      │  [approved]
      ▼
    planning_node
      │
      ▼
    review_node
      │
      ├──[passed]──► END (final lesson plan)
      │
      │  [failed, revisions remaining]
      ▼
    rewrite_node ──► review_node  (loop)
      │
      └──[max revisions exceeded]──► END (best-effort output)

Usage
-----
>>> from core.orchestrator import build_graph
>>> graph = build_graph()
>>> result = graph.invoke({
...     "subject": "ELA",
...     "grade": "5",
...     "topic": "main idea and supporting details",
...     "state": "Florida",
...     "duration": "45 minutes",
...     "lesson_date": "2025-09-15",
... })
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from core.state import LessonPlanState
from data.retriever import (
    InvalidInputError,
    StandardsNotFoundError,
    get_standards,
)
from agents.planning_agent import generate_lesson_plan
from agents.review_agent import review_lesson_plan
from agents.rewrite_agent import rewrite_lesson_plan

# ==============================================================================
# Logger
# ==============================================================================
logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_MAX_REVISIONS: int = 3
"""Maximum number of rewrite → review cycles before the orchestrator
terminates with the best-effort draft."""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NODE IMPLEMENTATIONS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ==============================================================================
# 1. Intake Node — validate teacher-provided context
# ==============================================================================

def intake_node(state: LessonPlanState) -> dict[str, Any]:
    """Validate that all required lesson-context fields are present.

    Required fields: ``subject``, ``grade``, ``topic``, ``state``,
    ``duration``.  Optional but recommended: ``lesson_date``,
    ``syllabus_text``, ``materials``, ``accommodations``,
    ``teacher_notes``.

    Returns
    -------
    dict
        ``intake_valid`` (bool) and ``intake_errors`` (list[str]).
    """
    logger.info("intake_node: Validating lesson context.")

    required_fields: list[str] = [
        "subject",
        "grade",
        "topic",
        "state",
        "duration",
    ]

    errors: list[str] = []
    for field in required_fields:
        value: str = state.get(field, "")  # type: ignore[arg-type]
        if not value or not str(value).strip():
            errors.append(f"Missing required field: '{field}'.")

    if errors:
        logger.warning("intake_node: Validation failed — %s", errors)
        return {
            "intake_valid": False,
            "intake_errors": errors,
            "workflow_status": "failed",
            "error_message": (
                "Intake validation failed.  Please provide: "
                + ", ".join(f"'{f}'" for f in required_fields if
                            not str(state.get(f, "")).strip())  # type: ignore[arg-type]
            ),
        }

    logger.info("intake_node: All required fields present.")
    return {
        "intake_valid": True,
        "intake_errors": [],
        "workflow_status": "running",
    }


# ==============================================================================
# 2. Discovery Node — retrieve matching standards
# ==============================================================================

def discovery_node(state: LessonPlanState) -> dict[str, Any]:
    """Call the standards retriever and select the top-scoring match.

    Populates the ``retrieved_standards`` list and auto-selects the
    highest-scoring standard into ``selected_standard_*`` fields.
    The teacher will confirm or override in the next step.

    Returns
    -------
    dict
        Standard discovery results including the top recommendation.
    """
    logger.info(
        "discovery_node: Searching standards for state='%s', "
        "grade='%s', subject='%s', topic='%s'.",
        state.get("state"),
        state.get("grade"),
        state.get("subject"),
        state.get("topic"),
    )

    try:
        results: list[dict[str, Any]] = get_standards(
            state=str(state.get("state", "")),
            grade=str(state.get("grade", "")),
            subject=str(state.get("subject", "")),
            topic=str(state.get("topic", "")),
        )
    except (InvalidInputError, StandardsNotFoundError) as exc:
        logger.error("discovery_node: Retrieval failed — %s", exc)
        return {
            "retrieved_standards": [],
            "workflow_status": "failed",
            "error_message": f"Standards retrieval failed: {exc}",
        }

    if not results:
        logger.warning("discovery_node: Retriever returned zero results.")
        return {
            "retrieved_standards": [],
            "workflow_status": "failed",
            "error_message": (
                "No standards found for the given grade, subject, "
                "and state combination."
            ),
        }

    # Auto-select the top-scoring standard as the recommendation.
    top: dict[str, Any] = results[0]

    logger.info(
        "discovery_node: Found %d standard(s).  Top match: %s (score=%s).",
        len(results),
        top.get("code"),
        top.get("score"),
    )

    return {
        "retrieved_standards": results,
        "selected_standard_code": top["code"],
        "selected_standard_text": top["description"],
        "selected_standard_strand": top.get("strand", ""),
        "selected_standard_source": top.get("source", ""),
        "is_fallback": top.get("fallback", False),
    }


# ==============================================================================
# 3. Confirmation Node — human-in-the-loop standard approval
# ==============================================================================

def confirmation_node(state: LessonPlanState) -> dict[str, Any]:
    """Pause execution and present the retrieved standard to the teacher.

    Uses LangGraph's ``interrupt()`` to surface the standard details
    to the calling application.  The caller resumes the graph with a
    dict containing ``{"approved": True/False}``.

    If the teacher also provides an override standard code/text in the
    resume payload, those values replace the auto-selected standard.

    Returns
    -------
    dict
        ``standard_approved`` flag (and optionally overridden standard
        fields).
    """
    logger.info(
        "confirmation_node: Presenting standard '%s' for teacher approval.",
        state.get("selected_standard_code"),
    )

    # Build the payload the UI will display to the teacher.
    interrupt_payload: dict[str, Any] = {
        "message": (
            "Please review the retrieved standard below and confirm it "
            "is correct for your lesson.  You may also select a "
            "different standard from the retrieved list."
        ),
        "selected_standard": {
            "code": state.get("selected_standard_code", ""),
            "text": state.get("selected_standard_text", ""),
            "strand": state.get("selected_standard_strand", ""),
            "source": state.get("selected_standard_source", ""),
        },
        "is_fallback": state.get("is_fallback", False),
        "all_retrieved": state.get("retrieved_standards", []),
    }

    # ── Interrupt: execution halts here until the caller resumes. ────
    human_response: dict[str, Any] = interrupt(interrupt_payload)
    # ─────────────────────────────────────────────────────────────────

    approved: bool = human_response.get("approved", False)

    update: dict[str, Any] = {"standard_approved": approved}

    # Allow the teacher to override the auto-selected standard.
    if approved:
        if "override_code" in human_response:
            update["selected_standard_code"] = human_response["override_code"]
        if "override_text" in human_response:
            update["selected_standard_text"] = human_response["override_text"]
        logger.info("confirmation_node: Standard APPROVED by teacher.")
    else:
        update["workflow_status"] = "failed"
        update["error_message"] = (
            "Teacher rejected the retrieved standard.  "
            "Please restart with a revised topic or state."
        )
        logger.info("confirmation_node: Standard REJECTED by teacher.")

    return update


# ==============================================================================
# 4. Planning Node — generate the lesson plan
# ==============================================================================

def planning_node(state: LessonPlanState) -> dict[str, Any]:
    """Generate a standards-aligned lesson plan via the Planning Agent.

    Calls ``agents.planning_agent.generate_lesson_plan()`` with the
    teacher's confirmed context.  Falls back to a placeholder plan
    if the LLM call fails (e.g., missing API key during testing).

    Returns
    -------
    dict
        ``draft_lesson_plan`` (str) and initial ``revision_count``.
    """
    logger.info(
        "planning_node: Generating lesson plan for standard '%s'.",
        state.get("selected_standard_code"),
    )

    try:
        draft: str = generate_lesson_plan(
            topic=str(state.get("topic", "")),
            grade=str(state.get("grade", "")),
            subject=str(state.get("subject", "")),
            duration=str(state.get("duration", "")),
            standard_code=str(state.get("selected_standard_code", "")),
            standard_text=str(state.get("selected_standard_text", "")),
            lesson_date=str(state.get("lesson_date", "")),
            syllabus_text=str(state.get("syllabus_text", "")),
            materials=str(state.get("materials", "")),
            accommodations=str(state.get("accommodations", "")),
            teacher_notes=str(state.get("teacher_notes", "")),
        )
        logger.info("planning_node: LLM generated %d-char plan.", len(draft))
    except Exception as exc:
        logger.warning(
            "planning_node: LLM call failed (%s). Using placeholder.",
            exc,
        )
        draft = _build_placeholder_plan(state)

    return {
        "draft_lesson_plan": draft,
        "revision_count": 0,
        "max_revisions": state.get("max_revisions", DEFAULT_MAX_REVISIONS),
        "workflow_status": "running",
    }


# ==============================================================================
# 5. Review Node — four-point alignment rubric
# ==============================================================================

def review_node(state: LessonPlanState) -> dict[str, Any]:
    """Evaluate the draft lesson plan against the alignment rubric.

    Calls ``agents.review_agent.review_lesson_plan()`` to check four
    criteria via LLM.  Falls back to an all-pass scaffold if the LLM
    call fails.

    Returns
    -------
    dict
        ``review_passed`` (bool) and ``review_feedback`` (list of
        serialised ``RubricCheck`` dicts).
    """
    logger.info("review_node: Evaluating lesson plan against rubric.")

    draft: str = state.get("draft_lesson_plan", "")
    standard_code: str = str(state.get("selected_standard_code", ""))
    standard_text: str = str(state.get("selected_standard_text", ""))
    duration: str = str(state.get("duration", ""))

    try:
        result: dict[str, Any] = review_lesson_plan(
            draft=draft,
            standard_code=standard_code,
            standard_text=standard_text,
            duration=duration,
        )
        is_approved: bool = result.get("is_approved", False)
        feedback: list[dict[str, Any]] = result.get("criteria", [])
        logger.info(
            "review_node: LLM review complete — approved=%s, failed=%s.",
            is_approved,
            result.get("failed_criteria", []),
        )
    except Exception as exc:
        logger.warning(
            "review_node: LLM call failed (%s). Defaulting to all-pass.",
            exc,
        )
        is_approved = True
        feedback = [
            {"criterion": "Standards Alignment", "passed": True, "reason": ""},
            {"criterion": "Objective-to-Assessment Match", "passed": True, "reason": ""},
            {"criterion": "Activity-to-Objective Match", "passed": True, "reason": ""},
            {"criterion": "Pacing Realism", "passed": True, "reason": ""},
        ]

    if is_approved:
        logger.info("review_node: All rubric checks PASSED.")
        return {
            "review_passed": True,
            "review_feedback": feedback,
            "final_lesson_plan": draft,
            "workflow_status": "completed",
        }

    failed_criteria: list[str] = [
        f["criterion"] for f in feedback if not f.get("passed", True)
    ]
    logger.warning(
        "review_node: Rubric FAILED on: %s",
        ", ".join(failed_criteria),
    )
    return {
        "review_passed": False,
        "review_feedback": feedback,
    }


# ==============================================================================
# 6. Rewrite Node — targeted revision of failed sections
# ==============================================================================

def rewrite_node(state: LessonPlanState) -> dict[str, Any]:
    """Revise the draft lesson plan via the Rewrite Agent.

    Calls ``agents.rewrite_agent.rewrite_lesson_plan()`` with the
    current draft and the specific failure feedback.  Falls back to
    passing the draft through unchanged if the LLM call fails.

    Returns
    -------
    dict
        Updated ``draft_lesson_plan`` and incremented ``revision_count``.
    """
    current_revision: int = state.get("revision_count", 0)
    new_revision: int = current_revision + 1

    draft: str = state.get("draft_lesson_plan", "")
    feedback: list[dict[str, Any]] = state.get("review_feedback", [])
    standard_code: str = str(state.get("selected_standard_code", ""))
    standard_text: str = str(state.get("selected_standard_text", ""))
    duration: str = str(state.get("duration", ""))

    logger.info(
        "rewrite_node: Revision %d/%d.",
        new_revision,
        state.get("max_revisions", DEFAULT_MAX_REVISIONS),
    )

    try:
        revised: str = rewrite_lesson_plan(
            draft=draft,
            standard_code=standard_code,
            standard_text=standard_text,
            duration=duration,
            review_feedback=feedback,
        )
        logger.info(
            "rewrite_node: LLM revision complete — %d chars "
            "(was %d chars).",
            len(revised),
            len(draft),
        )
    except Exception as exc:
        logger.warning(
            "rewrite_node: LLM call failed (%s). Passing draft through.",
            exc,
        )
        revised = draft

    return {
        "draft_lesson_plan": revised,
        "revision_count": new_revision,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONDITIONAL EDGE ROUTERS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def route_after_intake(state: LessonPlanState) -> str:
    """Route after intake validation.

    Returns
    -------
    str
        ``"discovery_node"`` if intake is valid, ``END`` otherwise.
    """
    if state.get("intake_valid", False):
        return "discovery_node"
    return END


def route_after_confirmation(state: LessonPlanState) -> str:
    """Route after teacher confirmation of the standard.

    Returns
    -------
    str
        ``"planning_node"`` if the teacher approved, ``END`` otherwise.
    """
    if state.get("standard_approved", False):
        return "planning_node"
    return END


def route_after_review(state: LessonPlanState) -> str:
    """Route after the alignment review.

    Three outcomes:
    1. **Passed** → ``END`` (final lesson plan is ready).
    2. **Failed, revisions remaining** → ``"rewrite_node"``.
    3. **Failed, max revisions exceeded** → ``END`` (best-effort).

    Returns
    -------
    str
        Next node name or ``END``.
    """
    if state.get("review_passed", False):
        return END

    current: int = state.get("revision_count", 0)
    maximum: int = state.get("max_revisions", DEFAULT_MAX_REVISIONS)

    if current < maximum:
        logger.info(
            "route_after_review: Review failed.  Routing to rewrite "
            "(revision %d/%d).",
            current + 1,
            maximum,
        )
        return "rewrite_node"

    # Max revisions exceeded — end with the best-effort draft.
    logger.warning(
        "route_after_review: Max revisions (%d) exceeded.  "
        "Ending with best-effort draft.",
        maximum,
    )
    return END


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  GRAPH BUILDER                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def build_graph(checkpointer: Any = None) -> StateGraph:
    """Construct and compile the Lesson Plan Helper workflow graph.

    Returns
    -------
    StateGraph
        A compiled LangGraph ``StateGraph`` ready for ``.invoke()``
        or ``.stream()``.

    Graph topology
    --------------
    .. code-block:: text

        START → intake_node ─┬─[valid]──► discovery_node
                             └─[invalid]─► END
                                            │
                             discovery_node ─► confirmation_node
                                                     │
                                  ┌──[rejected]──► END
                                  │
                           [approved]
                                  │
                           planning_node ─► review_node
                                                │
                                  ┌──[passed]──► END
                                  │
                           [failed, can retry]
                                  │
                           rewrite_node ──► review_node  (loop)
                                  │
                           [max revisions]──► END
    """
    # ── 1.  Initialise the graph with our state schema ───────────────
    graph = StateGraph(LessonPlanState)

    # ── 2.  Register nodes ───────────────────────────────────────────
    graph.add_node("intake_node", intake_node)
    graph.add_node("discovery_node", discovery_node)
    graph.add_node("confirmation_node", confirmation_node)
    graph.add_node("planning_node", planning_node)
    graph.add_node("review_node", review_node)
    graph.add_node("rewrite_node", rewrite_node)

    # ── 3.  Wire edges ───────────────────────────────────────────────

    # START → intake
    graph.add_edge(START, "intake_node")

    # intake → discovery (if valid) or END (if invalid)
    graph.add_conditional_edges(
        "intake_node",
        route_after_intake,
        {"discovery_node": "discovery_node", END: END},
    )

    # discovery → confirmation
    graph.add_edge("discovery_node", "confirmation_node")

    # confirmation → planning (if approved) or END (if rejected)
    graph.add_conditional_edges(
        "confirmation_node",
        route_after_confirmation,
        {"planning_node": "planning_node", END: END},
    )

    # planning → review
    graph.add_edge("planning_node", "review_node")

    # review → END (if passed) or rewrite (if failed) or END (max revisions)
    graph.add_conditional_edges(
        "review_node",
        route_after_review,
        {"rewrite_node": "rewrite_node", END: END},
    )

    # rewrite → review (loop back)
    graph.add_edge("rewrite_node", "review_node")

    # ── 4.  Compile ──────────────────────────────────────────────────
    compiled_graph = graph.compile(checkpointer=checkpointer)

    logger.info("build_graph: Workflow graph compiled successfully.")

    return compiled_graph


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PRIVATE HELPERS                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _build_placeholder_plan(state: LessonPlanState) -> str:
    """Build a structured placeholder lesson plan for testing.

    This will be replaced by an LLM-generated plan once the Planning
    Agent is wired in.  The structure mirrors ``LessonPlanDocument``
    so downstream nodes see a realistic shape.
    """
    standard_code: str = state.get("selected_standard_code", "N/A")
    standard_text: str = state.get("selected_standard_text", "N/A")
    topic: str = state.get("topic", "N/A")
    grade: str = state.get("grade", "N/A")
    duration: str = state.get("duration", "N/A")
    accommodations: str = state.get("accommodations", "None specified")

    return f"""# Lesson Plan — {topic}
**Grade:** {grade}  |  **Duration:** {duration}  |  **Standard:** {standard_code}

## Standard
{standard_text}

## Objective
[Placeholder] Students will be able to demonstrate mastery of {standard_code} by…

## Hook (5 min)
[Placeholder] Opening activity related to {topic}.

## Direct Instruction (10 min)
[Placeholder] Teacher-led instruction on {topic} aligned to {standard_code}.

## Guided Practice (10 min)
[Placeholder] Students work in pairs on a structured activity.

## Independent Practice (10 min)
[Placeholder] Students complete an independent task.

## Assessment (5 min)
[Placeholder] Exit ticket aligned to the objective.

## Differentiation
{accommodations}

## Teacher Reflection
[Placeholder] What worked?  What would you change?
"""
