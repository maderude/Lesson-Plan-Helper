import logging
import uuid
import sys
import os
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP

# Internal imports (from our existing business logic)
from data.retriever import get_standards, InvalidInputError, StandardsNotFoundError
from data.database import STANDARDS_DB, STATE_KEY_MAP
from agents.review_agent import review_lesson_plan
from agents.rewrite_agent import rewrite_lesson_plan
from ui.export import markdown_to_pdf, markdown_to_docx

logger = logging.getLogger(__name__)

# ==============================================================================
# Initialization
# ==============================================================================
mcp = FastMCP("lesson-plan-helper-mcp")

# ==============================================================================
# Tools
# ==============================================================================

class SearchStandardsInput(BaseModel):
    grade: str
    subject: str
    state: str
    topic: str
    keywords: Optional[List[str]] = None

class StandardMatch(BaseModel):
    standard_id: str
    code: str
    description: str
    subject: str
    grade: str
    state: str
    confidence: float

class SearchStandardsOutput(BaseModel):
    matches: List[StandardMatch]

@mcp.tool()
def search_standards(payload: SearchStandardsInput) -> SearchStandardsOutput:
    """Finds candidate standards from grade/subject/state/topic."""
    try:
        results = get_standards(
            state=payload.state,
            grade=payload.grade,
            subject=payload.subject,
            topic=payload.topic,
        )
        
        matches = []
        for r in results:
            matches.append(StandardMatch(
                standard_id=r["code"], # Using code as ID
                code=r["code"],
                description=r["description"],
                subject=r["subject"],
                grade=r["grade"],
                state=r["state"],
                confidence=r.get("score", 1.0)
            ))
        return SearchStandardsOutput(matches=matches)
    except (InvalidInputError, StandardsNotFoundError) as e:
        logger.warning(f"Search standards failed: {e}")
        return SearchStandardsOutput(matches=[])


class GetStandardDetailsInput(BaseModel):
    standard_id: str

class GetStandardDetailsOutput(BaseModel):
    code: str
    description: str
    metadata: Dict[str, Any]

@mcp.tool()
def get_standard_details(payload: GetStandardDetailsInput) -> GetStandardDetailsOutput:
    """Returns full text and metadata for selected standards."""
    # Search all states and grades for the standard code
    for state, grades in STANDARDS_DB.items():
        for grade, subjects in grades.items():
            for subject, stds in subjects.items():
                for std in stds:
                    if std["code"] == payload.standard_id:
                        return GetStandardDetailsOutput(
                            code=std["code"],
                            description=std["description"],
                            metadata={"state": state, "grade": grade, "subject": subject}
                        )
    raise ValueError(f"Standard {payload.standard_id} not found.")


class ValidateLessonStructureInput(BaseModel):
    lesson_markdown: str

class ValidateLessonStructureOutput(BaseModel):
    is_valid: bool
    missing_sections: List[str]

@mcp.tool()
def validate_lesson_structure(payload: ValidateLessonStructureInput) -> ValidateLessonStructureOutput:
    """Checks required lesson sections and formatting rules."""
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
    markdown_lower = payload.lesson_markdown.lower()
    for section in required_sections:
        clean_section = section.replace("## ", "").strip().lower()
        if clean_section not in markdown_lower:
            missing.append(section)
    
    return ValidateLessonStructureOutput(
        is_valid=len(missing) == 0,
        missing_sections=missing
    )


class ScoreLessonRubricInput(BaseModel):
    lesson_markdown: str
    selected_standard_ids: List[str]
    duration: str = "45 minutes"
    rubric_version: str = "v1"

class RubricCriterion(BaseModel):
    model_config = {
        "populate_by_name": True
    }

    name: str
    score: int
    pass_: bool = Field(alias="pass")
    feedback: str

class ScoreLessonRubricOutput(BaseModel):
    overall_pass: bool
    criteria: List[RubricCriterion]
    revision_notes: List[str]

@mcp.tool()
def score_lesson_rubric(payload: ScoreLessonRubricInput) -> ScoreLessonRubricOutput:
    """Scores the draft on the 4 rubric criteria."""
    
    code = payload.selected_standard_ids[0] if payload.selected_standard_ids else "N/A"
    description = "Combined descriptions."
    
    try:
        details = get_standard_details(GetStandardDetailsInput(standard_id=code))
        description = details.description
    except ValueError:
        pass
        
    try:
        result = review_lesson_plan(
            draft=payload.lesson_markdown,
            standard_code=code,
            standard_text=description,
            duration=payload.duration
        )
    except (RuntimeError, ValueError) as e:
        logger.error(f"review_lesson_plan failed: {e}")
        # Return a graceful degraded result instead of crashing the MCP server
        return ScoreLessonRubricOutput(
            overall_pass=False,
            criteria=[RubricCriterion(**{"name": "Review Error", "score": 0,
                      "pass": False, "feedback": str(e)})],
            revision_notes=[str(e)]
        )
    
    criteria_list = []
    for crit in result.get("criteria", []):
        criteria_list.append(RubricCriterion(**{
            "name": crit.get("criterion", "Unknown"),
            "score": 100 if crit.get("passed") else 0,
            "pass": crit.get("passed", False),
            "feedback": crit.get("reason", "")
        }))
        
    return ScoreLessonRubricOutput(
        overall_pass=result.get("is_approved", False),
        criteria=criteria_list,
        revision_notes=[c.feedback for c in criteria_list if not c.pass_]
    )


class RewriteWithFeedbackInput(BaseModel):
    lesson_markdown: str
    selected_standard_ids: List[str]
    failure_reasons: List[str]
    duration: str = "45 minutes"
    formatting_rules_version: str = "v1"

class RewriteWithFeedbackOutput(BaseModel):
    revised_markdown: str
    changes_applied: List[str]

@mcp.tool()
def rewrite_with_feedback(payload: RewriteWithFeedbackInput) -> RewriteWithFeedbackOutput:
    """Rewrites based on structured failures."""
    
    code = payload.selected_standard_ids[0] if payload.selected_standard_ids else "N/A"
    description = "Combined descriptions."
    
    try:
        details = get_standard_details(GetStandardDetailsInput(standard_id=code))
        description = details.description
    except ValueError:
        pass
        
    fake_feedback_payload = [
        {
            "criterion": f"Failure {i+1}",
            "passed": False,
            "reason": f
        }
        for i, f in enumerate(payload.failure_reasons)
    ]
    
    revised = rewrite_lesson_plan(
        draft=payload.lesson_markdown,
        standard_code=code,
        standard_text=description,
        duration=payload.duration,
        review_feedback=fake_feedback_payload
    )
    
    return RewriteWithFeedbackOutput(
        revised_markdown=revised,
        changes_applied=["Applied rewrite agent feedback."]
    )


class ExportLessonPdfInput(BaseModel):
    lesson_markdown: str
    output_path: str

class ExportLessonPdfOutput(BaseModel):
    success: bool
    file_path: str

@mcp.tool()
def export_lesson_pdf(payload: ExportLessonPdfInput) -> ExportLessonPdfOutput:
    """Creates sanitized PDF output."""
    markdown_to_pdf(payload.lesson_markdown, payload.output_path)
    return ExportLessonPdfOutput(success=True, file_path=payload.output_path)


class ExportLessonDocxInput(BaseModel):
    lesson_markdown: str
    output_path: str

class ExportLessonDocxOutput(BaseModel):
    success: bool
    file_path: str

@mcp.tool()
def export_lesson_docx(payload: ExportLessonDocxInput) -> ExportLessonDocxOutput:
    """Creates Word output."""
    markdown_to_docx(payload.lesson_markdown, payload.output_path)
    return ExportLessonDocxOutput(success=True, file_path=payload.output_path)


class StartSessionInput(BaseModel):
    user_id: str
    previous_session_id: Optional[str] = None

class StartSessionOutput(BaseModel):
    session_id: str
    thread_id: str
    status: str

@mcp.tool()
def start_new_lesson_session(payload: StartSessionInput) -> StartSessionOutput:
    """Resets lesson session/thread context."""
    new_id = str(uuid.uuid4())
    return StartSessionOutput(
        session_id=new_id,
        thread_id=new_id,
        status="reset"
    )

# ==============================================================================
# Resources
# ==============================================================================

@mcp.resource("rubric://lesson-plan-helper/v1")
def rubric_resource() -> str:
    """The official 4-point review rubric for grading lesson plans."""
    return """
    Criteria:
    1. Standards Alignment (Score 1-4)
    2. Objective-to-Assessment Match (Score 1-4)
    3. Activity-to-Objective Match (Score 1-4)
    4. Time/Pacing (Score 1-4)
    Pass threshold: all criteria must pass (score >= 3).
    """

@mcp.resource("formatting://lesson-plan-helper/v1")
def formatting_resource() -> str:
    """The strict formatting rules for generating lesson plans."""
    return """
    Rules:
    - Differentiation must address the specific accommodations provided.
    - Differentiation labels must be on new lines, exactly formatted like `ELL Students: ` (no markdown bold **, no bullets -).
    - Assessment must contain ONLY the question itself. Do NOT prefix with 'Exit Ticket:', 'Prompt:', or any other label.
    - Use EXACT section headings: `## Objective`, `## Essential Question`, `## Instructional Materials`, `## Teaching Strategies`, `## Hook`, `## Direct Instruction`, `## Guided Practice`, `## Independent Practice`, `## Assessment`, `## Assignments`, `## Homework Notes`, `## Differentiation`, `## Teacher Reflection`.
    """

@mcp.resource("templates://lesson-plan/default-structure")
def templates_resource() -> str:
    """The Markdown skeleton template."""
    return """
    ## Objective
    ...
    ## Essential Question
    ...
    ## Instructional Materials
    ...
    ## Teaching Strategies
    ...
    ## Hook
    ...
    ## Direct Instruction
    ...
    ## Guided Practice
    ...
    ## Independent Practice
    ...
    ## Assessment
    ...
    ## Assignments
    ...
    ## Homework Notes
    ...
    ## Differentiation
    ...
    ## Teacher Reflection
    ...
    """

@mcp.resource("policy://export/unicode-sanitization")
def policy_resource() -> str:
    """The Unicode sanitization policy for exports."""
    return """
    Policy:
    - Replace smart quotes (“ ” ‘ ’) with straight quotes (" ').
    - Replace en-dash and em-dash (– —) with hyphens (- --).
    - Strip zero-width spaces and non-breaking spaces.
    - Encode to latin-1 and ignore unencodable characters.
    """

# ==============================================================================
# Prompts
# ==============================================================================

@mcp.prompt()
def draft_lesson_plan() -> str:
    """Prompt template for drafting a standards-aligned lesson plan."""
    return "Draft a standards-aligned lesson plan using the selected standards, incorporating the formatting rules from formatting://lesson-plan-helper/v1 and the structure from templates://lesson-plan/default-structure."

@mcp.prompt()
def review_lesson_plan_prompt() -> str:
    """Prompt template for reviewing a lesson plan."""
    return "Review the provided lesson plan against the rubric://lesson-plan-helper/v1. Ensure all criteria score a 3 or higher. Output structured pass/fail feedback."

@mcp.prompt()
def rewrite_lesson_plan_prompt() -> str:
    """Prompt template for rewriting a lesson plan."""
    return "Rewrite the provided lesson plan focusing ONLY on resolving the provided failure reasons. Maintain all other content exactly as is."

if __name__ == "__main__":
    # We will use stdio by default for ADK execution
    mcp.run(transport="stdio")
