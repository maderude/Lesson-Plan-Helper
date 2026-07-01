"""
data/database.py — In-Memory Standards Document Store
=====================================================

Provides an in-memory dictionary (``STANDARDS_DB``) that acts as the
document store for state educational standards.  Each entry is keyed
by a normalised state identifier and contains grade → subject → list
of standard records.

Coverage (this initial seed):
    • Common Core (ELA) — default / fallback
    • Florida (B.E.S.T. ELA)
    • Texas (TEKS ELA)
    • Virginia (SOL English)

Each state corpus contains 3-4 **Grade 5 Reading: Informational Text**
standards.  Standard descriptions are paraphrased from published
frameworks to preserve intent, cognitive verbs, and complexity level
while avoiding verbatim copyright reproduction.

Extending the database:
    Add new states by inserting a top-level key (e.g., ``"california"``)
    following the same nested structure.  The retriever in
    ``data/retriever.py`` will pick it up automatically.
"""

from __future__ import annotations

from typing import Any

# ==============================================================================
# Type alias for readability
# ==============================================================================
# Structure:  state  →  grade  →  subject  →  [standard records]
StandardRecord = dict[str, Any]
StandardsDatabase = dict[str, dict[str, dict[str, list[StandardRecord]]]]


# ==============================================================================
# Helper — build a single standard record
# ==============================================================================
def _standard(
    code: str,
    description: str,
    strand: str,
    keywords: list[str],
) -> StandardRecord:
    """Return a consistently-shaped standard record dictionary.

    Parameters
    ----------
    code : str
        Official (or mock-official) standard code, e.g. ``"RI.5.1"``.
    description : str
        Full text of the standard.
    strand : str
        Curricular strand, e.g. ``"Reading: Informational Text"``.
    keywords : list[str]
        Topic keywords used for lightweight retrieval matching.

    Returns
    -------
    StandardRecord
        A flat dictionary with the four fields above.
    """
    return {
        "code": code,
        "description": description,
        "strand": strand,
        "keywords": keywords,
    }


# ==============================================================================
# STANDARDS DATABASE
# ==============================================================================

STANDARDS_DB: StandardsDatabase = {
    # ------------------------------------------------------------------
    # COMMON CORE  (default / fallback)
    # ------------------------------------------------------------------
    "common_core": {
        "5": {
            "ELA": [
                _standard(
                    code="RI.5.1",
                    description=(
                        "Quote accurately from a text and make relevant "
                        "inferences when explaining what the text says "
                        "explicitly and when drawing inferences from the text."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "quote", "accurately", "text evidence", "inference",
                        "explicit", "cite", "support",
                    ],
                ),
                _standard(
                    code="RI.5.2",
                    description=(
                        "Determine two or more main ideas of a text and "
                        "explain how they are supported by key details; "
                        "summarize the text."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "main idea", "key details", "summarize", "summary",
                        "central idea", "supporting details",
                    ],
                ),
                _standard(
                    code="RI.5.3",
                    description=(
                        "Explain the relationships or interactions between "
                        "two or more individuals, events, ideas, or concepts "
                        "in a historical, scientific, or technical text based "
                        "on specific information in the text."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "relationships", "interactions", "individuals",
                        "events", "ideas", "concepts", "historical",
                        "scientific", "technical", "compare",
                    ],
                ),
                _standard(
                    code="RI.5.4",
                    description=(
                        "Determine the meaning of general academic and "
                        "domain-specific words and phrases in a text relevant "
                        "to a grade 5 topic or subject area."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "vocabulary", "academic words", "domain-specific",
                        "context clues", "meaning", "words and phrases",
                    ],
                ),
            ],
        },
    },

    # ------------------------------------------------------------------
    # FLORIDA  (B.E.S.T. ELA Standards)
    # ------------------------------------------------------------------
    "florida": {
        "5": {
            "ELA": [
                _standard(
                    code="ELA.5.R.1.1",
                    description=(
                        "Analyze how setting, events, conflict, and "
                        "characterization contribute to the plot in a "
                        "literary text.  For informational texts, analyze "
                        "the purpose and key details the author uses to "
                        "support claims."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "analyze", "purpose", "key details", "claims",
                        "author", "support", "informational",
                    ],
                ),
                _standard(
                    code="ELA.5.R.1.2",
                    description=(
                        "Explain how relevant details support the central "
                        "idea or theme, identified with significant textual "
                        "evidence."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "central idea", "theme", "relevant details",
                        "textual evidence", "support", "main idea",
                    ],
                ),
                _standard(
                    code="ELA.5.R.1.3",
                    description=(
                        "Analyze the relationship between two or more "
                        "individuals, events, or ideas in an informational "
                        "text, explaining how the relationship is developed."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "relationships", "individuals", "events", "ideas",
                        "analyze", "interactions", "compare",
                    ],
                ),
                _standard(
                    code="ELA.5.R.2.1",
                    description=(
                        "Interpret figurative language and determine the "
                        "meaning of general academic and domain-specific "
                        "words and phrases used in grade-level content."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "figurative language", "vocabulary", "academic words",
                        "domain-specific", "meaning", "context clues",
                    ],
                ),
            ],
        },
    },

    # ------------------------------------------------------------------
    # TEXAS  (TEKS — Texas Essential Knowledge and Skills)
    # ------------------------------------------------------------------
    "texas": {
        "5": {
            "ELA": [
                _standard(
                    code="TEKS.5.6F",
                    description=(
                        "Make inferences and use evidence to support "
                        "understanding of informational text."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "inference", "evidence", "support", "understanding",
                        "informational", "cite",
                    ],
                ),
                _standard(
                    code="TEKS.5.6G",
                    description=(
                        "Evaluate details read to determine key ideas; "
                        "identify the central idea and supporting evidence "
                        "within each paragraph."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "key ideas", "central idea", "supporting evidence",
                        "evaluate", "main idea", "details", "summarize",
                    ],
                ),
                _standard(
                    code="TEKS.5.6H",
                    description=(
                        "Synthesize information from two texts on the same "
                        "topic to demonstrate understanding and produce a "
                        "written response."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "synthesize", "two texts", "same topic", "compare",
                        "written response", "combine",
                    ],
                ),
            ],
        },
    },

    # ------------------------------------------------------------------
    # VIRGINIA  (SOL — Standards of Learning)
    # ------------------------------------------------------------------
    "virginia": {
        "5": {
            "ELA": [
                _standard(
                    code="SOL.5.6a",
                    description=(
                        "Use text evidence to draw conclusions and make "
                        "inferences about informational texts."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "text evidence", "conclusions", "inference",
                        "draw conclusions", "informational", "cite",
                    ],
                ),
                _standard(
                    code="SOL.5.6b",
                    description=(
                        "Identify the main idea and supporting details in "
                        "nonfiction texts."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "main idea", "supporting details", "nonfiction",
                        "central idea", "summarize", "key details",
                    ],
                ),
                _standard(
                    code="SOL.5.6c",
                    description=(
                        "Summarize information found in nonfiction texts, "
                        "distinguishing between key ideas and minor details."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "summarize", "key ideas", "minor details",
                        "nonfiction", "distinguish", "summary",
                    ],
                ),
                _standard(
                    code="SOL.5.6d",
                    description=(
                        "Identify and explain the author's use of "
                        "organizational patterns in nonfiction, including "
                        "cause and effect, comparison and contrast, and "
                        "chronological order."
                    ),
                    strand="Reading: Informational Text",
                    keywords=[
                        "text structure", "organizational patterns",
                        "cause and effect", "compare and contrast",
                        "chronological order", "author",
                    ],
                ),
            ],
        },
    },
}


# ==============================================================================
# STATE NAME → DB KEY MAPPING
# ==============================================================================
# Maps common state name variants and abbreviations to the canonical key
# used in ``STANDARDS_DB``.  Extend this mapping as new states are added.
# ==============================================================================

STATE_KEY_MAP: dict[str, str] = {
    # Common Core (explicit requests)
    "common core": "common_core",
    "common_core": "common_core",
    "cc": "common_core",
    # Florida
    "florida": "florida",
    "fl": "florida",
    "best": "florida",
    "b.e.s.t.": "florida",
    # Texas
    "texas": "texas",
    "tx": "texas",
    "teks": "texas",
    # Virginia
    "virginia": "virginia",
    "va": "virginia",
    "sol": "virginia",
}
