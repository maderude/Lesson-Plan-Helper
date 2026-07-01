"""
ui/app.py — Streamlit Frontend for Lesson Plan Helper
======================================================

A polished, guided workspace that walks teachers through five steps:
1. Lesson Setup — enter lesson details in collapsible sidebar sections
2. Find Standards — match state standards to the topic
3. Generate Draft — multi-agent AI creates a full lesson plan
4. Differentiate — review and refine for ELL/IEP/gifted
5. Export — download the finished plan

The main panel shows 8 lesson-plan cards that start as placeholders
and progressively fill in as the workflow advances.
"""

from __future__ import annotations

import io
import logging
import os
import re
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from core.orchestrator import build_graph
from data.retriever import get_standards, InvalidInputError, StandardsNotFoundError
from ui.export import markdown_to_pdf, markdown_to_docx, render_copy_button_html
from ui.styles import (
    APP_CSS,
    CARD_DEFS,
    render_card_html,
    render_step_bar_html,
)

# Configure logging and load environment variables
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ==============================================================================
# Page Config
# ==============================================================================
st.set_page_config(
    page_title="Lesson Plan Helper",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject design system CSS
st.markdown(APP_CSS, unsafe_allow_html=True)

# ==============================================================================
# Session State Initialization
# ==============================================================================
_DEFAULTS = {
    "phase": "input",          # input | confirmed | generating | done | download
    "step": 1,                 # 1-5 workflow step
    "retrieved_standards": [],
    "selected_std_indices": [],  # list of selected standard indices
    "thread_id": str(uuid.uuid4()),
    "form_inputs": {},
    "final_state": {},
    "workflow_errors": [],
    "standards_found": False,
    "draft_generated": False,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "graph" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()
    st.session_state.graph = build_graph(checkpointer=st.session_state.checkpointer)


# ==============================================================================
# File Upload Helper
# ==============================================================================
def extract_text_from_file(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF, DOCX, or TXT file."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif name.endswith(".docx"):
            from docx import Document
            doc = Document(io.BytesIO(uploaded_file.read()))
            return "\n".join(p.text for p in doc.paragraphs)
        elif name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="replace")
        else:
            st.sidebar.warning(f"Unsupported file type: {name}")
            return ""
    except Exception as exc:
        st.sidebar.error(f"Could not read file: {exc}")
        return ""


# ==============================================================================
# Action Handlers
# ==============================================================================
def handle_find_standards(form_data: dict[str, Any]):
    """Call the retriever to fetch matching standards."""
    st.session_state.workflow_errors = []
    try:
        standards = get_standards(
            state=form_data["state"],
            grade=form_data["grade"],
            subject=form_data["subject"],
            topic=form_data["topic"],
        )
        if not standards:
            st.session_state.workflow_errors.append("No matching standards found.")
            return

        st.session_state.retrieved_standards = standards
        st.session_state.selected_std_indices = list(range(len(standards)))
        st.session_state.form_inputs = form_data
        st.session_state.phase = "confirmed"
        st.session_state.step = 2
        st.session_state.standards_found = True
    except (InvalidInputError, StandardsNotFoundError) as e:
        st.session_state.workflow_errors.append(str(e))
    except Exception as e:
        st.session_state.workflow_errors.append(f"An unexpected error occurred: {e}")


def handle_reset():
    """Reset application state to start a new lesson plan."""
    import uuid
    for key, default in _DEFAULTS.items():
        if key == "thread_id":
            st.session_state[key] = str(uuid.uuid4())
        else:
            st.session_state[key] = default
    st.session_state.thread_id = str(uuid.uuid4())


def _collect_sidebar_inputs() -> dict[str, Any]:
    """Read current values from sidebar session-state keys."""
    combined_syllabus = st.session_state.get("_syllabus_text", "").strip()
    uploaded = st.session_state.get("_syllabus_file")
    if uploaded is not None:
        file_text = extract_text_from_file(uploaded).strip()
        if file_text:
            combined_syllabus = f"{combined_syllabus}\n\n{file_text}".strip()
    return {
        "subject": st.session_state.get("_subject", "ELA"),
        "grade": st.session_state.get("_grade", "5"),
        "state": st.session_state.get("_state_fw", "Common Core"),
        "topic": st.session_state.get("_topic", ""),
        "duration": st.session_state.get("_duration", "45 minutes"),
        "lesson_date": str(st.session_state.get("_lesson_date", "")),
        "accommodations": st.session_state.get("_accommodations", ""),
        "materials": st.session_state.get("_materials", ""),
        "syllabus_text": combined_syllabus,
    }


# ==============================================================================
# Sidebar
# ==============================================================================
def render_sidebar():
    """Render the reorganized sidebar with collapsible sections."""

    # ── Brand ────────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">📚</div>
            <div>
                <div class="sidebar-brand-text">Lesson Plan Helper</div>
                <span class="sidebar-brand-sub">AI-Powered Planning</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 1. Lesson Basics ─────────────────────────────────────────────
    with st.sidebar.expander("📖  Lesson Basics", expanded=True):
        st.text_input(
            "Subject",
            value="ELA",
            help="e.g. ELA, Math, Science",
            key="_subject",
        )
        st.selectbox(
            "Grade Level",
            ["K"] + [str(i) for i in range(1, 13)],
            index=5,
            key="_grade",
        )
        st.text_input(
            "Topic / Focus Skill",
            value="Explain how relevant details support central idea",
            help="What do you want students to learn?",
            key="_topic",
        )

    # ── 2. Standards & Timing ────────────────────────────────────────
    with st.sidebar.expander("🕐  Standards & Timing", expanded=False):
        st.selectbox(
            "State Framework",
            ["Common Core", "Florida", "Texas", "Virginia"],
            index=1,
            key="_state_fw",
        )
        st.selectbox(
            "Duration",
            ["30 minutes", "45 minutes", "60 minutes", "90 minutes"],
            index=1,
            key="_duration",
        )
        st.date_input("Target Lesson Date", key="_lesson_date")

    # ── 3. Classroom Context ─────────────────────────────────────────
    with st.sidebar.expander("👥  Classroom Context", expanded=False):
        st.text_area(
            "Student Accommodations",
            value="2 ELL students; 1 student with IEP for extended time",
            placeholder="ELL, IEP, 504, gifted needs...",
            height=100,
            key="_accommodations",
        )

    # ── 4. Materials & Resources ─────────────────────────────────────
    with st.sidebar.expander("🧰  Materials & Resources", expanded=False):
        st.text_area(
            "Available Materials",
            placeholder="Textbooks, worksheets, manipulatives, tech...",
            height=90,
            key="_materials",
        )

    # ── 5. Unit Context ──────────────────────────────────────────────
    with st.sidebar.expander("📄  Unit Context", expanded=False):
        st.text_area(
            "Syllabus / Unit Plan",
            placeholder="Where does this lesson fit in your larger unit?",
            height=90,
            key="_syllabus_text",
        )
        st.file_uploader(
            "Upload file (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            help="Upload your syllabus, unit plan, or scope-and-sequence.",
            key="_syllabus_file",
        )

    # ── CTAs ─────────────────────────────────────────────────────────
    st.sidebar.markdown("---")

    # Secondary: Show Standards
    if st.sidebar.button(
        "🔍  Show Standards",
        use_container_width=True,
        type="secondary",
    ):
        form_data = _collect_sidebar_inputs()
        handle_find_standards(form_data)

    # Primary: Generate Lesson Draft (only enabled after standards found and at least one selected)
    can_generate = (
        st.session_state.phase == "confirmed"
        and len(st.session_state.selected_std_indices) > 0
    )
    if st.sidebar.button(
        "✨  Generate Lesson Draft",
        use_container_width=True,
        type="primary",
        disabled=not can_generate,
    ):
        st.session_state.phase = "generating"
        st.session_state.step = 3
        st.rerun()

    # Tertiary row (only visible after generation)
    if st.session_state.phase in ("done", "download"):
        st.sidebar.markdown("---")
        col_a, col_b = st.sidebar.columns(2)
        with col_a:
            if st.button("🔄 New Lesson", use_container_width=True):
                handle_reset()
                st.rerun()
        with col_b:
            final_state = st.session_state.final_state
            final_plan = final_state.get("final_lesson_plan", "") or final_state.get("draft_lesson_plan", "")
            if final_plan:
                if st.button("📥 Download", use_container_width=True):
                    st.session_state.phase = "download"
                    st.rerun()


# ==============================================================================
# Lesson Plan Section Parser
# ==============================================================================
def _parse_lesson_sections(markdown_text: str) -> dict[str, str]:
    """Parse the generated markdown into card-content sections.

    Returns a dict keyed by CARD_DEFS["key"] values with the
    corresponding section body text (raw markdown).
    """
    # Mapping from heading keywords to card keys
    heading_map = {
        "standard":        "standards",
        "objective":       "objective",
        "essential":       "essential_q",
        "outline":         "outline",
        "hook":            "outline",
        "direct instruct": "outline",
        "guided":          "guided",
        "independent":     "independent",
        "assessment":      "assessment",
        "exit ticket":     "assessment",
        "differentiat":    "differentiation",
        "accommodat":      "differentiation",
        "pacing":          "outline",
    }

    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in markdown_text.split("\n"):
        heading_match = re.match(r"^#{1,3}\s+(.+)", line)
        if heading_match:
            # Save previous section
            if current_key and current_lines:
                body = "\n".join(current_lines).strip()
                if current_key not in sections or len(body) > len(sections[current_key]):
                    sections[current_key] = body

            # Determine new section
            heading_text = heading_match.group(1).lower()
            current_key = None
            for keyword, card_key in heading_map.items():
                if keyword in heading_text:
                    current_key = card_key
                    break
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    # Flush last section
    if current_key and current_lines:
        body = "\n".join(current_lines).strip()
        if current_key not in sections or len(body) > len(sections[current_key]):
            sections[current_key] = body

    return sections


# ==============================================================================
# Main Panel Renderers
# ==============================================================================
def _render_empty_state():
    """Show the empty-state hero + 8 placeholder cards."""
    st.markdown(
        """
        <div class="empty-hero">
            <div class="empty-hero-icon">📝</div>
            <div class="empty-hero-title">Your lesson workspace</div>
            <div class="empty-hero-text">
                Fill out the lesson details in the sidebar, then click
                <strong>Find Standards</strong> to get started. Each card below
                will populate as your lesson plan takes shape.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_card_grid()


def _render_card_grid(populated: dict[str, str] | None = None):
    """Render the 8 lesson-plan cards in a 2-column layout."""
    populated = populated or {}

    cols = st.columns(2, gap="medium")
    for i, card in enumerate(CARD_DEFS):
        col = cols[i % 2]
        body = populated.get(card["key"], "")
        with col:
            st.markdown(
                render_card_html(
                    icon=card["icon"],
                    color=card["color"],
                    title=card["title"],
                    body=body,
                    placeholder=card["placeholder"],
                ),
                unsafe_allow_html=True,
            )


def _render_confirmed_state():
    """Standards found — show checkboxes for multi-select + card preview."""
    standards = st.session_state.retrieved_standards
    is_fallback = standards[0].get("fallback", False) if standards else False

    if is_fallback:
        st.warning(
            "**State Framework Fallback**: Your selected state's standards were "
            "not found. We've fallen back to **Common Core** defaults."
        )

    st.markdown("#### Select Standards")
    st.caption("All standards are selected by default. Uncheck any you want to exclude.")

    # Render checkboxes for each standard — pre-selected
    selected_indices: list[int] = []
    for i, std in enumerate(standards):
        default_checked = i in st.session_state.selected_std_indices
        checked = st.checkbox(
            f"**{std['code']}** — {std['description'][:100]}",
            value=default_checked,
            key=f"_std_cb_{i}",
        )
        if checked:
            selected_indices.append(i)

    st.session_state.selected_std_indices = selected_indices

    if not selected_indices:
        st.info("Select at least one standard to generate a lesson plan.")

    # Build standards alignment card content
    state_name = st.session_state.form_inputs.get("state", "Common Core")
    is_any_fallback = any(standards[i].get("fallback", False) for i in selected_indices)
    if is_any_fallback:
        state_label = "Common Core (Fallback)"
    else:
        state_label = state_name

    if selected_indices:
        list_items = ""
        for idx in selected_indices:
            std = standards[idx]
            list_items += f'<li><strong>{std["code"]}</strong> &mdash; {std["description"]}</li>\n'
        std_html = f"""<div class="std-card">
<div class="std-desc" style="margin-bottom:8px;">This lesson aligns with the <strong>{state_label}</strong> State Standards for:</div>
<ul style="margin:0 0 0 8px; padding-left:16px; color:#334155; line-height:1.7;">
{list_items}</ul>
</div>"""
    else:
        std_html = ""

    # Populate just the standards card, rest are placeholders
    _render_card_grid(populated={"standards": std_html})


def _render_generating_state():
    """Run the multi-agent pipeline with live streaming status."""
    with st.status("Preparing your lesson plan...", expanded=True) as status:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # Step 1: Run to interrupt
        status.update(label="Validating inputs and discovering standards...")
        st.write("✔ Intake validated inputs.")
        st.write("✔ Standards discovered.")

        st.session_state.graph.invoke(st.session_state.form_inputs, config=config)
        st.write("✔ Ready for approval.")

        # Step 2: Resume with approved standards (combine all selected)
        status.update(label="Generating lesson plan with AI agents...")
        selected_indices = st.session_state.selected_std_indices
        all_standards = st.session_state.retrieved_standards
        selected_stds = [all_standards[i] for i in selected_indices]

        # Combine codes and descriptions for the planning agent
        combined_code = "; ".join(s["code"] for s in selected_stds)
        combined_text = "\n".join(
            f"- {s['code']}: {s['description']}" for s in selected_stds
        )

        resume_payload = {
            "approved": True,
            "override_code": combined_code,
            "override_text": combined_text,
        }
        for s in selected_stds:
            st.write(f"Aligning to: **{s['code']}**")

        for event in st.session_state.graph.stream(
            Command(resume=resume_payload), config=config, stream_mode="updates"
        ):
            for node_name, node_update in event.items():
                if node_name == "planning_node":
                    st.write("⏳ **Planning Agent** — drafting lesson plan...")
                elif node_name == "review_node":
                    passed = node_update.get("review_passed", False)
                    if passed:
                        st.write("✔ **Review Agent** — rubric check **passed**.")
                    else:
                        st.write("⚠️ **Review Agent** — rubric check **failed**, revising...")
                elif node_name == "rewrite_node":
                    rev = node_update.get("revision_count", "?")
                    st.write(f"🔄 **Rewrite Agent** — revision #{rev}")

        status.update(label="Lesson plan generated!", state="complete")

    # Transition to done
    final_state = st.session_state.graph.get_state(config).values
    st.session_state.final_state = final_state
    st.session_state.phase = "done"
    st.session_state.step = 5
    st.session_state.draft_generated = True
    st.rerun()


def _render_done_state():
    """Display the completed lesson plan in populated cards."""
    final_state = st.session_state.final_state
    final_plan = final_state.get("final_lesson_plan", "") or final_state.get("draft_lesson_plan", "")
    review_feedback = final_state.get("review_feedback", [])
    revision_count = final_state.get("revision_count", 0)

    # Build the standards card content from state (may contain multiple)
    std_code = final_state.get("selected_standard_code", "")
    std_text = final_state.get("selected_standard_text", "")
    state_name = st.session_state.form_inputs.get("state", "Common Core")
    std_html = ""
    if std_code:
        codes = [c.strip() for c in std_code.split(";") if c.strip()]
        texts = [t.strip().lstrip("- ") for t in std_text.split("\n") if t.strip()]
        list_items = ""
        for j, code in enumerate(codes):
            desc = texts[j] if j < len(texts) else ""
            if desc.startswith(f"{code}: "):
                desc = desc[len(f"{code}: "):]
            list_items += f'<li><strong>{code}</strong> &mdash; {desc}</li>\n'
        std_html = f"""<div class="std-card">
<div class="std-desc" style="margin-bottom:8px;">This lesson aligns with the <strong>{state_name}</strong> State Standards for:</div>
<ul style="margin:0 0 0 8px; padding-left:16px; color:#334155; line-height:1.7;">
{list_items}</ul>
</div>"""

    # Parse lesson plan into sections
    sections = _parse_lesson_sections(final_plan) if final_plan else {}
    if std_html:
        sections["standards"] = std_html

    # Action row
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("➕  Plan Another Lesson", use_container_width=True):
            handle_reset()
            st.rerun()
    with col2:
        st.button(
            "♿  Refine for ELL/IEP",
            use_container_width=True,
            disabled=True,
            help="Coming soon — auto-differentiate for student needs.",
        )
    with col3:
        if final_plan:
            if st.button("📥  Download Lesson Plan", use_container_width=True):
                st.session_state.phase = "download"
                st.rerun()

    st.markdown("")

    # Rubric review
    with st.expander("🔍  Alignment Rubric Review", expanded=False):
        if revision_count > 0:
            st.caption(f"Revision cycles: {revision_count}")
        for check in review_feedback:
            passed = check.get("passed", False)
            criterion = check.get("criterion", "Unknown")
            reason = check.get("reason", "")
            if passed:
                st.markdown(
                    f'<div class="rubric-pass">✅  <strong>{criterion}</strong> — Passed</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="rubric-fail">❌  <strong>{criterion}</strong> — Failed</div>',
                    unsafe_allow_html=True,
                )
                if reason:
                    st.markdown(
                        f'<div class="rubric-reason">{reason}</div>',
                        unsafe_allow_html=True,
                    )

    # Populated card grid
    _render_card_grid(populated=sections)

    # Full markdown (collapsible)
    if final_plan:
        with st.expander("📄  View Full Lesson Plan", expanded=False):
            st.markdown(final_plan)


def _render_download_page():
    """Dedicated download page showing the lesson and download buttons."""
    final_state = st.session_state.final_state
    final_plan = final_state.get("final_lesson_plan", "") or final_state.get("draft_lesson_plan", "")

    if not final_plan:
        st.warning("No lesson plan found. Please generate a lesson first.")
        if st.button("🔄  Start New Lesson"):
            handle_reset()
            st.rerun()
        return

    # Header
    st.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0;">
<h2 style="margin:0; color: #1e293b;">✅  Your Lesson Plan is Ready!</h2>
<p style="color: #64748b; margin-top: 6px;">Review your lesson below, then download or start a new one.</p>
</div>
""", unsafe_allow_html=True)

    # ── Download buttons row ─────────────────────────────────────────
    subject = st.session_state.form_inputs.get("subject", "plan")
    grade = st.session_state.form_inputs.get("grade", "")
    file_base = f"lesson_{subject}_{grade}"

    col1, col2, col3 = st.columns(3)
    with col1:
        pdf_bytes = markdown_to_pdf(final_plan)
        st.download_button(
            "📄  Download PDF",
            data=pdf_bytes,
            file_name=f"{file_base}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_pdf_btn",
        )
    with col2:
        docx_bytes = markdown_to_docx(final_plan)
        st.download_button(
            "📝  Download Word",
            data=docx_bytes,
            file_name=f"{file_base}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key="dl_docx_btn",
        )
    with col3:
        if st.button("✨  New Lesson", use_container_width=True, type="primary"):
            handle_reset()
            st.rerun()

    st.markdown("---")

    # ── Lesson metadata ──────────────────────────────────────────────
    meta_cols = st.columns(4)
    std_code = final_state.get("selected_standard_code", "")
    with meta_cols[0]:
        st.metric("Subject", subject)
    with meta_cols[1]:
        st.metric("Grade", grade)
    with meta_cols[2]:
        st.metric("Duration", st.session_state.form_inputs.get("duration", ""))
    with meta_cols[3]:
        # Show first standard code or all if short
        codes = [c.strip() for c in std_code.split(";") if c.strip()]
        st.metric("Standards", f"{len(codes)} aligned" if len(codes) > 1 else (codes[0] if codes else "N/A"))

    st.markdown("")

    # ── Full lesson plan ─────────────────────────────────────────────
    st.markdown(final_plan)

    # ── Bottom action row ────────────────────────────────────────────
    st.markdown("---")
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        st.download_button(
            "📄  Download PDF",
            data=pdf_bytes,
            file_name=f"{file_base}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_pdf_btn_bottom",
        )
    with bcol2:
        st.download_button(
            "📝  Download Word",
            data=docx_bytes,
            file_name=f"{file_base}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key="dl_docx_btn_bottom",
        )
    with bcol3:
        if st.button("✨  New Lesson", use_container_width=True, type="primary", key="new_lesson_bottom"):
            handle_reset()
            st.rerun()

    # Back to cards view
    st.markdown("")
    if st.button("⬅  Back to Lesson Details", key="back_to_done"):
        st.session_state.phase = "done"
        st.rerun()


# ==============================================================================
# Main
# ==============================================================================
def main():
    # Sidebar
    render_sidebar()

    # ── Page Header ──────────────────────────────────────────────────
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">Lesson Planning Workspace</div>
            <div class="page-subtitle">Create standards-aligned, classroom-ready lessons in minutes.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Step Indicator ───────────────────────────────────────────────
    st.markdown(render_step_bar_html(st.session_state.step), unsafe_allow_html=True)

    # ── Errors ───────────────────────────────────────────────────────
    if st.session_state.workflow_errors:
        for err in st.session_state.workflow_errors:
            st.error(err)

    # ── API Key Check ────────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning(
            "**No OpenAI API Key Found** — The app will run in Mock Mode. "
            "Configure `OPENAI_API_KEY` in your `.env` to generate real lesson plans."
        )

    # ── Phase Router ─────────────────────────────────────────────────
    phase = st.session_state.phase

    if phase == "download":
        _render_download_page()
    elif phase == "input":
        _render_empty_state()
    elif phase == "confirmed":
        _render_confirmed_state()
    elif phase == "generating":
        _render_generating_state()
    elif phase == "done":
        _render_done_state()


if __name__ == "__main__":
    main()
