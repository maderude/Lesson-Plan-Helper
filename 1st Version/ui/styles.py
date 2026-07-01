"""
ui/styles.py — Design System & CSS for Lesson Plan Helper
==========================================================

All visual styling extracted into a single module for maintainability.
Provides a modern, SaaS-style design system with:
- Neutral color palette (slate grays, soft blue accent)
- System font stack
- Card system for lesson plan sections
- 5-step workflow indicator
- Sidebar overrides for collapsible sections
- Responsive adjustments
"""

from __future__ import annotations

# ==============================================================================
# Design Tokens
# ==============================================================================

COLORS = {
    "bg":           "#FFFFFF",
    "surface":      "#F8FAFC",
    "surface_alt":  "#F1F5F9",
    "border":       "#E2E8F0",
    "border_light": "#F1F5F9",
    "text":         "#1E293B",
    "text_muted":   "#64748B",
    "text_faint":   "#94A3B8",
    "primary":      "#2563EB",
    "primary_hover":"#1D4ED8",
    "primary_light":"#EFF6FF",
    "success":      "#16A34A",
    "success_bg":   "#F0FDF4",
    "warning":      "#D97706",
    "warning_bg":   "#FFFBEB",
    "error":        "#DC2626",
    "error_bg":     "#FEF2F2",
    "accent":       "#7C3AED",
}

# ==============================================================================
# Main Stylesheet
# ==============================================================================

APP_CSS = """
<style>
/* ── Global Reset & Typography ─────────────────────────────────────── */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', system-ui, -apple-system, "Segoe UI", sans-serif;
}

/* ── Sidebar ───────────────────────────────────────────────────────── */

section[data-testid="stSidebar"] {
    width: 22rem !important;
    background-color: #FAFBFC;
    border-right: 1px solid #E2E8F0;
}

section[data-testid="stSidebar"] .stExpander {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    margin-bottom: 0.5rem;
}

section[data-testid="stSidebar"] .stExpander details summary {
    font-weight: 600;
    font-size: 0.85rem;
    color: #334155;
    padding: 0.6rem 0.8rem;
}

section[data-testid="stSidebar"] .stExpander details[open] summary {
    border-bottom: 1px solid #F1F5F9;
}

section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextArea label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stFileUploader label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #475569;
}

/* Sidebar brand area */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.8rem 0 1.2rem 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 1rem;
}
.sidebar-brand-icon {
    font-size: 1.6rem;
    line-height: 1;
}
.sidebar-brand-text {
    font-size: 1rem;
    font-weight: 700;
    color: #1E293B;
    letter-spacing: -0.01em;
}
.sidebar-brand-sub {
    font-size: 0.7rem;
    font-weight: 400;
    color: #94A3B8;
    display: block;
    margin-top: 1px;
}

/* ── Step Indicator Bar ────────────────────────────────────────────── */

.step-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 1rem 0 1.5rem 0;
    margin-bottom: 0.5rem;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 0;
}
.step-circle {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
    flex-shrink: 0;
    transition: all 0.2s ease;
}
.step-circle.completed {
    background-color: #16A34A;
    color: #FFFFFF;
}
.step-circle.active {
    background-color: #2563EB;
    color: #FFFFFF;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
}
.step-circle.pending {
    background-color: #F1F5F9;
    color: #94A3B8;
    border: 1.5px solid #CBD5E1;
}
.step-label {
    font-size: 0.7rem;
    font-weight: 500;
    margin-top: 4px;
    white-space: nowrap;
}
.step-label.completed { color: #16A34A; }
.step-label.active { color: #2563EB; font-weight: 600; }
.step-label.pending { color: #94A3B8; }
.step-connector {
    width: 48px;
    height: 2px;
    margin: 0 4px;
    margin-bottom: 18px;
    flex-shrink: 0;
}
.step-connector.completed { background-color: #16A34A; }
.step-connector.active { background: linear-gradient(90deg, #16A34A, #2563EB); }
.step-connector.pending { background-color: #E2E8F0; }
.step-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
}

/* ── Page Header ───────────────────────────────────────────────────── */

.page-header {
    padding: 0.5rem 0 0 0;
}
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1E293B;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #64748B;
    margin: 0.25rem 0 0 0;
    font-weight: 400;
}

/* ── Lesson Plan Cards ─────────────────────────────────────────────── */

.lp-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.lp-card:hover {
    border-color: #CBD5E1;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}
.lp-card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
}
.lp-card-icon {
    font-size: 1.1rem;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    flex-shrink: 0;
}
.lp-card-icon.blue    { background: #EFF6FF; }
.lp-card-icon.green   { background: #F0FDF4; }
.lp-card-icon.purple  { background: #F5F3FF; }
.lp-card-icon.amber   { background: #FFFBEB; }
.lp-card-icon.rose    { background: #FFF1F2; }
.lp-card-icon.teal    { background: #F0FDFA; }
.lp-card-icon.indigo  { background: #EEF2FF; }
.lp-card-icon.orange  { background: #FFF7ED; }

.lp-card-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #334155;
    margin: 0;
}
.lp-card-body {
    font-size: 0.85rem;
    color: #64748B;
    line-height: 1.55;
}
.lp-card-body.populated {
    color: #334155;
}
.lp-card-placeholder {
    font-size: 0.8rem;
    color: #94A3B8;
    font-style: italic;
    line-height: 1.5;
    padding: 0.5rem 0;
}

/* ── Empty State Hero ──────────────────────────────────────────────── */

.empty-hero {
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 1.5rem;
}
.empty-hero-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    opacity: 0.7;
}
.empty-hero-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.4rem;
}
.empty-hero-text {
    font-size: 0.85rem;
    color: #94A3B8;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ── Action Bar ────────────────────────────────────────────────────── */

.action-bar {
    display: flex;
    gap: 0.5rem;
    padding: 0.75rem 0;
    flex-wrap: wrap;
}

/* ── Standard Card (confirmed) ─────────────────────────────────────── */

.std-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
}
.std-code {
    font-size: 0.95rem;
    font-weight: 700;
    color: #2563EB;
    margin-bottom: 0.3rem;
}
.std-desc {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.5;
}
.std-meta {
    display: flex;
    gap: 0.75rem;
    margin-top: 0.5rem;
    font-size: 0.75rem;
    color: #94A3B8;
}
.std-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    font-size: 0.7rem;
    font-weight: 600;
    border-radius: 4px;
    background-color: #EFF6FF;
    color: #2563EB;
}
.std-badge.fallback {
    background-color: #FFFBEB;
    color: #D97706;
}

/* ── Status / Generating  ──────────────────────────────────────────── */

.gen-status {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.25rem;
}

/* ── Rubric Check Items ────────────────────────────────────────────── */

.rubric-pass {
    padding: 0.5rem 0.75rem;
    background: #F0FDF4;
    border-left: 3px solid #16A34A;
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
    color: #166534;
}
.rubric-fail {
    padding: 0.5rem 0.75rem;
    background: #FEF2F2;
    border-left: 3px solid #DC2626;
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
    color: #991B1B;
}
.rubric-reason {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 0.2rem;
    padding-left: 0.25rem;
}

/* ── Responsive ────────────────────────────────────────────────────── */

@media (max-width: 768px) {
    .step-bar { flex-wrap: wrap; gap: 0.25rem; }
    .step-connector { width: 24px; }
    .step-label { font-size: 0.6rem; }
    .page-title { font-size: 1.3rem; }
}
</style>
"""


# ==============================================================================
# Step Indicator Renderer
# ==============================================================================

STEP_LABELS = [
    "Lesson Setup",
    "Find Standards",
    "Generate Draft",
    "Differentiate",
    "Export",
]

STEP_ICONS_DONE = "✓"


def render_step_bar_html(active_step: int) -> str:
    """Generate the HTML for the 5-step workflow indicator.

    Parameters
    ----------
    active_step : int
        The 1-indexed current step (1–5).

    Returns
    -------
    str
        HTML string to be rendered via st.markdown(unsafe_allow_html=True).
    """
    parts: list[str] = ['<div class="step-bar">']

    for i, label in enumerate(STEP_LABELS, start=1):
        # Determine state
        if i < active_step:
            state = "completed"
            circle_content = STEP_ICONS_DONE
        elif i == active_step:
            state = "active"
            circle_content = str(i)
        else:
            state = "pending"
            circle_content = str(i)

        parts.append(f'<div class="step-item">')
        parts.append(f'  <div class="step-wrapper">')
        parts.append(f'    <div class="step-circle {state}">{circle_content}</div>')
        parts.append(f'    <div class="step-label {state}">{label}</div>')
        parts.append(f'  </div>')

        # Connector (not after the last step)
        if i < len(STEP_LABELS):
            if i < active_step:
                conn_state = "completed"
            elif i == active_step:
                conn_state = "active"
            else:
                conn_state = "pending"
            parts.append(f'  <div class="step-connector {conn_state}"></div>')

        parts.append(f'</div>')

    parts.append('</div>')
    return "\n".join(parts)


# ==============================================================================
# Card Definitions (Empty State)
# ==============================================================================

CARD_DEFS = [
    {
        "key": "standards",
        "icon": "🎯",
        "color": "blue",
        "title": "Standards Alignment",
        "placeholder": "Your aligned state standards will appear here after you show standards.",
    },
    {
        "key": "objective",
        "icon": "📐",
        "color": "indigo",
        "title": "Learning Objective",
        "placeholder": "A measurable, student-centered learning objective will be generated.",
    },
    {
        "key": "essential_q",
        "icon": "💡",
        "color": "amber",
        "title": "Essential Question",
        "placeholder": "A driving question to frame the lesson and engage students.",
    },
    {
        "key": "outline",
        "icon": "📋",
        "color": "teal",
        "title": "Lesson Outline",
        "placeholder": "Hook, direct instruction, transitions — a full pacing guide.",
    },
    {
        "key": "guided",
        "icon": "🤝",
        "color": "green",
        "title": "Guided Practice",
        "placeholder": "Teacher-led practice activity with scaffolding and checks.",
    },
    {
        "key": "independent",
        "icon": "✏️",
        "color": "purple",
        "title": "Independent Practice",
        "placeholder": "Student-led activity to apply learning independently.",
    },
    {
        "key": "assessment",
        "icon": "📝",
        "color": "rose",
        "title": "Assessment",
        "placeholder": "A quick formative check to measure student understanding.",
    },
    {
        "key": "differentiation",
        "icon": "♿",
        "color": "orange",
        "title": "Differentiation",
        "placeholder": "ELL, IEP, gifted accommodations tailored to your classroom.",
    },
]


import markdown

def render_card_html(
    icon: str,
    color: str,
    title: str,
    body: str = "",
    placeholder: str = "",
) -> str:
    """Render a single lesson-plan card as HTML.

    If *body* is provided the card shows populated content;
    otherwise it shows the muted *placeholder* text.
    """
    if body:
        # Only parse as markdown if it's not already pre-rendered HTML (like the standards card)
        if not body.strip().startswith("<div"):
            body_html = markdown.markdown(body)
        else:
            body_html = body
        content = f'<div class="lp-card-body populated">{body_html}</div>'
    else:
        content = f'<div class="lp-card-placeholder">{placeholder}</div>'

    return f"""<div class="lp-card">
<div class="lp-card-header">
<div class="lp-card-icon {color}">{icon}</div>
<div class="lp-card-title">{title}</div>
</div>
{content}
</div>"""
