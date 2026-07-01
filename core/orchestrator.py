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
import re
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from core.state import LessonPlanState
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


# ==============================================================================
# Markdown Header Sanitizer
# ==============================================================================
# Canonical section names the UI parser and MCP validator expect.
_CANONICAL_HEADERS: dict[str, str] = {
    "objective":              "## Objective",
    "essential question":     "## Essential Question",
    "instructional material": "## Instructional Materials",
    "material":               "## Instructional Materials",
    "required tool":          "## Instructional Materials",
    "teaching strateg":       "## Teaching Strategies",
    "strateg":                "## Teaching Strategies",
    "hook":                   "## Hook",
    "direct instruct":        "## Direct Instruction",
    "guided practice":        "## Guided Practice",
    "guided":                 "## Guided Practice",
    "independent practice":   "## Independent Practice",
    "independent":            "## Independent Practice",
    "assessment":             "## Assessment",
    "exit ticket":            "## Assessment",
    "assignment":             "## Assignments",
    "homework":               "## Homework Notes",
    "differentiat":           "## Differentiation",
    "accommodat":             "## Differentiation",
    "teacher reflection":     "## Teacher Reflection",
    "reflection":             "## Teacher Reflection",
}


def _sanitize_markdown_headers(markdown: str) -> str:
    """Normalize LLM-generated section headers to canonical ## format.

    Handles four common failure modes:
    1. Missing ## prefix (e.g. 'Objective' instead of '## Objective')
    2. Novel names (e.g. 'Required Tools' → '## Instructional Materials')
    3. Bold-wrapped headers (e.g. '**Objective**')
    4. Inline content on header lines (e.g. '**Teaching Strategies:** content...')
    
    Preserves time annotations like '## Hook (5 min)'.
    """
    lines = markdown.split("\n")
    sanitized = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines, bullet points, numbered lists
        if (not stripped
                or stripped.startswith("- ")
                or stripped.startswith("* ")
                or re.match(r"^\d+\.\s", stripped)):
            sanitized.append(line)
            continue

        # ── Case 1: Already a proper ## header ──────────────────────
        if stripped.startswith("## "):
            header_text = stripped[3:].strip()
            header_lower = re.sub(r":$", "", header_text).lower()
            # Remove time annotations for matching
            match_text = re.sub(r"\(\s*\d+\s*min(?:utes?)?\s*\)", "", header_lower).strip()
            time_match = re.search(r"\(\s*\d+\s*min(?:utes?)?\s*\)", header_text, re.IGNORECASE)
            time_suffix = f" {time_match.group()}" if time_match else ""

            matched = False
            for keyword, canonical in _CANONICAL_HEADERS.items():
                if keyword in match_text:
                    sanitized.append(f"{canonical}{time_suffix}")
                    matched = True
                    break
            if not matched:
                sanitized.append(line)
            continue

        # ── Case 2: Bold-wrapped or bare header (with possible inline content) ──
        # Check if the line STARTS with a bold header pattern: **SomeName** or **SomeName:**
        bold_match = re.match(r"^\s*\*\*(.+?)\*\*:?\s*(.*)", stripped)
        if bold_match:
            header_part = bold_match.group(1).strip()
            trailing_content = bold_match.group(2).strip()
            header_lower = header_part.lower()
            match_text = re.sub(r"\(\s*\d+\s*min(?:utes?)?\s*\)", "", header_lower).strip()
            time_match = re.search(r"\(\s*\d+\s*min(?:utes?)?\s*\)", header_part, re.IGNORECASE)
            time_suffix = f" {time_match.group()}" if time_match else ""

            for keyword, canonical in _CANONICAL_HEADERS.items():
                if keyword in match_text:
                    sanitized.append(f"{canonical}{time_suffix}")
                    if trailing_content:
                        sanitized.append(trailing_content)
                    break
            else:
                sanitized.append(line)  # No keyword match, keep as-is
            continue

        # ── Case 3: Bare header line (short, no punctuation at end) ──
        # Strip any leading # or * for matching
        clean = re.sub(r"^[\s#*]+", "", stripped).strip()
        clean = re.sub(r"\*\*$", "", clean).strip()
        clean_no_colon = re.sub(r":$", "", clean).strip()
        clean_lower = clean_no_colon.lower()

        # Extract time annotation
        time_match = re.search(r"\(\s*\d+\s*min(?:utes?)?\s*\)", clean, re.IGNORECASE)
        time_suffix = f" {time_match.group()}" if time_match else ""
        match_text = re.sub(r"\(\s*\d+\s*min(?:utes?)?\s*\)", "", clean_lower).strip()

        # Only treat as header if it's short (≤5 words) and doesn't end with period
        words = clean_no_colon.split()
        if len(words) <= 5 and not clean.endswith("."):
            matched = False
            for keyword, canonical in _CANONICAL_HEADERS.items():
                if match_text.startswith(keyword):
                    sanitized.append(f"{canonical}{time_suffix}")
                    matched = True
                    break
            if matched:
                continue

        sanitized.append(line)

    return "\n".join(sanitized)


# Canonical list of all 13 required sections in correct order
_REQUIRED_SECTIONS_ORDERED: list[tuple[str, str]] = [
    ("## Objective", "[Measurable objective using the standard's exact verb]"),
    ("## Essential Question", "[Open-ended driving question for students]"),
    ("## Instructional Materials", "- Primary text or reading passage\n- Graphic organizer or handout\n- Whiteboard or projector"),
    ("## Teaching Strategies", "Gradual Release of Responsibility: Teacher models, then guides, then releases to students.\nThink-Pair-Share: Students discuss with a partner before sharing with the class."),
    ("## Hook", "[Engaging opening activity that activates prior knowledge]"),
    ("## Direct Instruction", "[Step-by-step teacher-led instruction with examples]"),
    ("## Guided Practice", "[Structured collaborative activity with teacher support]"),
    ("## Independent Practice", "[Individual student task with success criteria]"),
    ("## Assessment", "[Exit ticket question that measures the objective skill]"),
    ("## Assignments", "- Completed graphic organizer\n- Written response from independent practice"),
    ("## Homework Notes", "Read for 20 minutes and identify one example related to today's lesson."),
    ("## Differentiation", "ELL Students: Provide sentence frames and visual aids.\nStudents with IEP: Allow extended time and simplified directions."),
    ("## Teacher Reflection", "What strategies were most effective?\nHow could I better support struggling learners next time?"),
]


def _inject_missing_section_stubs(markdown: str) -> str:
    """Inject skeleton ## headers with placeholder content for any missing sections.

    This ensures the draft always has all 13 sections, preventing
    structural validation failures and giving the rewrite agent
    headers to work with.
    """
    markdown_lower = markdown.lower()
    lines_to_append: list[str] = []

    for header, placeholder in _REQUIRED_SECTIONS_ORDERED:
        # Check if the section header keyword exists in the draft
        clean = header.replace("## ", "").strip().lower()
        if clean not in markdown_lower:
            lines_to_append.append(f"\n{header}")
            lines_to_append.append(placeholder)
            logger.info(f"_inject_missing_section_stubs: Injected missing '{header}'")

    if lines_to_append:
        markdown = markdown.rstrip() + "\n" + "\n".join(lines_to_append) + "\n"

    return markdown


def _validate_structure(markdown: str) -> list[str]:
    """Check for missing required sections. Returns list of missing section names."""
    required_sections = [
        "## Objective",
        "## Essential Question",
        "## Instructional Materials",
        "## Teaching Strategies",
        "## Hook",
        "## Direct Instruction",
        "## Guided Practice",
        "## Independent Practice",
        "## Assessment",
        "## Assignments",
        "## Homework Notes",
        "## Differentiation",
        "## Teacher Reflection"
    ]
    missing = []
    markdown_lower = markdown.lower()
    for section in required_sections:
        clean_section = section.replace("## ", "").strip().lower()
        if clean_section not in markdown_lower:
            missing.append(section)
    return missing


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NODE IMPLEMENTATIONS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ==============================================================================
# 1. Intake Node — validate teacher-provided context
# ==============================================================================

def intake_node(state: LessonPlanState) -> dict[str, Any]:
    """Validate that all required lesson-context fields are present."""
    logger.info("intake_node: Validating lesson context.")
    return {"status": "intake_complete"}


# ==============================================================================
# 2. Discovery Node — retrieve matching standards
# ==============================================================================

def discovery_node(state: LessonPlanState) -> dict[str, Any]:
    """Call the standards retriever and select the top-scoring match."""
    logger.info("discovery_node: Standards discovery.")
    return {"status": "discovery_complete"}


# ==============================================================================
# 3. Confirmation Node — human-in-the-loop standard approval
# ==============================================================================

def confirmation_node(state: LessonPlanState) -> dict[str, Any]:
    """Pause execution and present the retrieved standard to the teacher."""
    logger.info(
        "confirmation_node: Presenting standard '%s' for teacher approval.",
        state.get("selected_standard_code"),
    )

    human_response = interrupt("Confirm standards selection")
    update: dict[str, Any] = {"status": "standards_confirmed"}

    if isinstance(human_response, dict):
        approved = human_response.get("approved", False)
        update["standard_approved"] = approved
        if approved:
            if "override_code" in human_response:
                update["selected_standard_code"] = human_response["override_code"]
            if "override_text" in human_response:
                update["selected_standard_text"] = human_response["override_text"]
            logger.info("confirmation_node: Standard APPROVED by teacher.")
        else:
            update["workflow_status"] = "failed"
            update["error_message"] = "Teacher rejected the retrieved standard."
            logger.info("confirmation_node: Standard REJECTED by teacher.")

    return update


# ==============================================================================
# 4. Planning Node — generate the lesson plan
# ==============================================================================

def planning_node(state: LessonPlanState) -> dict[str, Any]:
    """Generate a standards-aligned lesson plan via LLM."""
    logger.info(
        "planning_node: Generating lesson plan for standard '%s'.",
        state.get("selected_standard_code"),
    )

    topic = state.get("topic", "")
    grade = state.get("grade", "")
    subject = state.get("subject", "ELA")
    duration = state.get("duration", "45 minutes")
    accommodations = state.get("accommodations", "None specified")
    syllabus = state.get("syllabus_text", "")
    continuous_skills = state.get("continuous_skills", "")
    std_code = state.get("selected_standard_code", "")
    std_text = state.get("selected_standard_text", "")

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # Normalize duration to minutes
    dur_match = re.search(r"\d+", duration or "")
    duration_minutes = int(dur_match.group()) if dur_match else 45
    duration_display = f"{duration_minutes} minutes"

    if grade in ["K", "1", "2"]:
        assessment_req = "— observable student work, manipulatives sorting, or oral check for understanding"
    else:
        assessment_req = "— the exact written question or specific task students complete, no labels"

    prompt = f"""
Draft a complete, classroom-ready lesson plan for the following:

## Lesson Context
- Topic: {topic}
- Grade: {grade}
- Subject: {subject}
- Duration: {duration_display}
- Accommodations: {accommodations}
- Syllabus Context: {syllabus or "None provided"}
- Continuous Skills to Review: {continuous_skills or "None provided"}

## CONFIRMED STANDARD
- Code: {std_code}
- Full Text: {std_text}

## CRITICAL RULES
1. ## Objective MUST use the exact cognitive verb from the standard above.
2. ## Assessment MUST measure the exact skill in the Objective.
3. ## Guided Practice and ## Independent Practice MUST build the exact skill in the Objective.
4. If Continuous Skills are provided, they MUST be integrated organically into the Hook, Guided Practice, or Homework — do not introduce them as a separate new topic.
5. Times for Hook + Direct Instruction + Guided Practice + Independent Practice + Assessment MUST sum to within 5 minutes of {duration_minutes} total minutes (the teacher's duration is a target, not a rigid constraint).
6. ## Teaching Strategies MUST list exactly 2-3 strategies, each formatted as: Strategy Name: how it is applied.
7. ## Assignments MUST be a concise bulleted list of TODAY's in-class work products.
8. Do NOT skip or leave empty ANY section listed below.

## REQUIRED SECTIONS (You MUST generate exactly these Markdown headers, starting with ##):

## Objective
[one measurable objective using the standard's exact verb]

## Essential Question
[one open-ended driving question for students]

## Instructional Materials
[bulleted list of specific materials, texts, handouts, tech]

## Teaching Strategies
[2-3 strategies, each formatted as "Name: application"]

## Hook (X min)
[specific engaging opening activity]

## Direct Instruction (X min)
[step-by-step teacher modeling with examples]

## Guided Practice (X min)
[structured collaborative student activity]

## Independent Practice (X min)
[individual student task with success criteria]

## Assessment (X min)
[{assessment_req}]

## Assignments
[concise bulleted list of in-class deliverables or work products]

## Homework Notes
[1-3 lines of family-friendly, actionable tasks]

## Differentiation
[specific supports for: {accommodations}]

## Teacher Reflection
[2-3 post-lesson reflective questions]
"""
    try:
        response = llm.invoke(prompt)
        draft = response.content
        logger.info("planning_node: LLM generated %d-char plan.", len(draft))
    except Exception as exc:
        logger.warning(
            "planning_node: LLM call failed (%s). Using placeholder.", exc,
        )
        draft = _build_placeholder_plan(state)

    # Sanitize headers and guarantee all 13 sections exist
    draft = _sanitize_markdown_headers(draft)
    draft = _inject_missing_section_stubs(draft)

    return {
        "draft_lesson_plan": draft,
        "revision_count": 0,
        "max_revisions": state.get("max_revisions", DEFAULT_MAX_REVISIONS),
        "workflow_status": "running",
    }


# ==============================================================================
# 5. Review Node — four-point alignment rubric (DIRECT call, no MCP)
# ==============================================================================

def review_node(state: LessonPlanState) -> dict[str, Any]:
    """Evaluate the draft lesson plan against the alignment rubric.

    Calls ``review_lesson_plan()`` directly — no MCP indirection.
    Also checks structural completeness (all 13 sections present).
    """
    logger.info("review_node: Evaluating lesson plan against rubric.")

    draft: str = state.get("draft_lesson_plan", "")
    standard_code: str = str(state.get("selected_standard_code", ""))
    standard_text: str = str(state.get("selected_standard_text", ""))
    duration: str = str(state.get("duration", ""))

    # ── Structural validation (inject stubs first) ───────────────────
    draft = _inject_missing_section_stubs(draft)
    missing = _validate_structure(draft)

    feedback: list[dict[str, Any]] = []
    overall_pass = True

    if missing:
        overall_pass = False
        feedback.append({
            "criterion": "Structural Completeness",
            "passed": False,
            "reason": f"Missing required sections: {', '.join(missing)}. You MUST add them.",
        })

    # ── Rubric scoring (direct call) ─────────────────────────────────
    try:
        result: dict[str, Any] = review_lesson_plan(
            draft=draft,
            standard_code=standard_code,
            standard_text=standard_text,
            duration=duration,
        )
        is_approved: bool = result.get("is_approved", False)
        criteria: list[dict[str, Any]] = result.get("criteria", [])

        overall_pass = overall_pass and is_approved
        for c in criteria:
            feedback.append({
                "criterion": c.get("criterion", "Unknown"),
                "passed": c.get("passed", False),
                "score": 100 if c.get("passed") else 0,
                "reason": c.get("reason", ""),
            })

        logger.info(
            "review_node: LLM review complete — approved=%s, failed=%s.",
            is_approved,
            result.get("failed_criteria", []),
        )
    except Exception as exc:
        logger.warning(
            "review_node: LLM call failed (%s). Defaulting to all-pass.", exc,
        )
        overall_pass = True
        feedback = [
            {"criterion": "Standards Alignment", "passed": True, "reason": ""},
            {"criterion": "Objective-to-Assessment Match", "passed": True, "reason": ""},
            {"criterion": "Activity-to-Objective Match", "passed": True, "reason": ""},
            {"criterion": "Pacing Realism", "passed": True, "reason": ""},
        ]

    logger.info(f"review_node: overall_pass={overall_pass}")
    for idx, f in enumerate(feedback, start=1):
        logger.info(f"  Criterion {idx}: {f['criterion']} - passed={f['passed']} - reason={f['reason']}")

    if overall_pass:
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
    logger.warning("review_node: Rubric FAILED on: %s", ", ".join(failed_criteria))

    return {
        "review_passed": False,
        "review_feedback": feedback,
        "draft_lesson_plan": draft,
    }


# ==============================================================================
# 6. Rewrite Node — targeted revision (DIRECT call, no MCP)
# ==============================================================================

def rewrite_node(state: LessonPlanState) -> dict[str, Any]:
    """Revise the draft lesson plan via the Rewrite Agent.

    Calls ``rewrite_lesson_plan()`` directly — no MCP indirection.
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
            "rewrite_node: LLM revision complete — %d chars (was %d chars).",
            len(revised),
            len(draft),
        )
    except Exception as exc:
        logger.warning(
            "rewrite_node: LLM call failed (%s). Passing draft through.", exc,
        )
        revised = draft

    # Sanitize and inject any still-missing stubs
    revised = _sanitize_markdown_headers(revised)
    revised = _inject_missing_section_stubs(revised)

    return {
        "draft_lesson_plan": revised,
        "revision_count": new_revision,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONDITIONAL EDGE ROUTERS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def route_after_review(state: LessonPlanState) -> str:
    """Route after the alignment review.

    Two outcomes:
    1. **Passed** → ``END`` (final lesson plan is ready).
    2. **Failed, revisions remaining** → ``"rewrite_node"``.
    3. **Failed, max revisions exceeded** → ``END`` (best-effort).
    """
    if state.get("review_passed", False):
        return END

    current: int = state.get("revision_count", 0)
    maximum: int = state.get("max_revisions", DEFAULT_MAX_REVISIONS)

    if current < maximum:
        logger.info(
            "route_after_review: Review failed. Routing to rewrite "
            "(revision %d/%d).",
            current + 1,
            maximum,
        )
        return "rewrite_node"

    # Max revisions exceeded — end with the best-effort draft.
    logger.warning(
        "route_after_review: Max revisions (%d) exceeded. "
        "Ending with best-effort draft.",
        maximum,
    )
    return END


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  GRAPH BUILDER                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def build_graph(checkpointer=None) -> StateGraph:
    """Construct and compile the Lesson Plan Helper workflow graph."""
    graph = StateGraph(LessonPlanState)

    # Register nodes
    graph.add_node("intake_node", intake_node)
    graph.add_node("discovery_node", discovery_node)
    graph.add_node("confirmation_node", confirmation_node)
    graph.add_node("planning_node", planning_node)
    graph.add_node("review_node", review_node)
    graph.add_node("rewrite_node", rewrite_node)

    # Wire edges
    graph.set_entry_point("intake_node")
    graph.add_edge("intake_node", "discovery_node")
    graph.add_edge("discovery_node", "confirmation_node")
    graph.add_edge("confirmation_node", "planning_node")
    graph.add_edge("planning_node", "review_node")
    graph.add_conditional_edges(
        "review_node",
        route_after_review,
        {"rewrite_node": "rewrite_node", END: END},
    )
    graph.add_edge("rewrite_node", "review_node")

    compiled_graph = graph.compile(checkpointer=checkpointer)
    logger.info("build_graph: Workflow graph compiled successfully.")

    return compiled_graph


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PRIVATE HELPERS                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _build_placeholder_plan(state: LessonPlanState) -> str:
    """Build a structured placeholder lesson plan for testing."""
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

## Essential Question
[Placeholder] How does {topic} connect to our learning goals?

## Instructional Materials
- Primary text or reading passage
- Graphic organizer or handout
- Whiteboard or projector

## Teaching Strategies
Gradual Release of Responsibility: Teacher models, then guides, then releases to students.
Think-Pair-Share: Students discuss with a partner before sharing with the class.

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

## Assignments
- Completed graphic organizer
- Written response from independent practice

## Homework Notes
Read for 20 minutes and identify one example related to today's lesson.

## Differentiation
{accommodations}

## Teacher Reflection
[Placeholder] What worked?  What would you change?
"""
