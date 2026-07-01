"""
data/retriever.py — Standards Retrieval with State-Aware Routing
================================================================

Provides ``get_standards()``, the primary retrieval function for the
Lesson Plan Helper workflow.  It routes the teacher's request to the
correct state standards corpus, scores each standard by keyword
overlap with the teacher's topic, and returns results sorted by
relevance.

If the requested state is not in the database the function
transparently falls back to Common Core and flags every returned
record so the UI can warn the teacher.

Usage
-----
>>> from data.retriever import get_standards
>>> results = get_standards("Florida", "5", "ELA", "main idea and details")
>>> for r in results:
...     print(r["code"], r["score"], r["fallback"])
"""

from __future__ import annotations

import logging
import re
from typing import Any

from data.database import STANDARDS_DB, STATE_KEY_MAP

# ==============================================================================
# Logger
# ==============================================================================
logger = logging.getLogger(__name__)


# ==============================================================================
# Custom Exceptions
# ==============================================================================

class StandardsNotFoundError(Exception):
    """Raised when no standards match the requested grade + subject.

    This is intentionally raised instead of silently returning an empty
    list so that calling code can surface a clear message to the teacher
    (e.g., "We don't have Grade 8 Science standards yet").
    """


class InvalidInputError(Exception):
    """Raised when a required input field is empty or invalid."""


# ==============================================================================
# Internal helpers
# ==============================================================================

def _normalise_state(state: str) -> str:
    """Normalise a state string to its canonical database key.

    The lookup is case-insensitive and strips leading/trailing whitespace.
    If no explicit mapping exists the raw lower-cased value is tried as a
    direct key into ``STANDARDS_DB``.

    Parameters
    ----------
    state : str
        The raw state name or abbreviation provided by the teacher.

    Returns
    -------
    str
        A canonical key suitable for indexing into ``STANDARDS_DB``,
        or the lower-cased input if no mapping is found (which will
        trigger the Common Core fallback downstream).
    """
    cleaned: str = state.strip().lower()
    return STATE_KEY_MAP.get(cleaned, cleaned)


def _tokenise_topic(topic: str) -> set[str]:
    """Tokenise a topic string into a set of lower-case keywords.

    Splits on non-alphanumeric characters and discards tokens shorter
    than 2 characters (articles, single letters, etc.).

    Parameters
    ----------
    topic : str
        Free-text topic description provided by the teacher.

    Returns
    -------
    set[str]
        Unique lower-case keyword tokens.
    """
    tokens: list[str] = re.split(r"[^a-zA-Z0-9]+", topic.lower())
    return {t for t in tokens if len(t) >= 2}


def _score_standard(
    standard: dict[str, Any],
    topic_tokens: set[str],
) -> int:
    """Score a standard by keyword overlap with the topic tokens.

    Each keyword in the standard's ``keywords`` list is itself tokenised
    (multi-word keywords like ``"main idea"`` become ``{"main", "idea"}``)
    and then intersected with *topic_tokens*.

    Parameters
    ----------
    standard : dict[str, Any]
        A single standard record from the database.
    topic_tokens : set[str]
        Pre-tokenised topic keywords.

    Returns
    -------
    int
        The number of overlapping tokens (higher = more relevant).
    """
    # Flatten multi-word keywords into individual tokens
    keyword_tokens: set[str] = set()
    for kw in standard.get("keywords", []):
        keyword_tokens.update(kw.lower().split())

    return len(keyword_tokens & topic_tokens)


# ==============================================================================
# Public API
# ==============================================================================

def get_standards(
    state: str,
    grade: str,
    subject: str,
    topic: str,
) -> list[dict[str, Any]]:
    """Retrieve matching standards from the in-memory store.

    Routing logic
    -------------
    1. Normalise ``state`` to a database key via ``STATE_KEY_MAP``.
    2. If the state key exists in ``STANDARDS_DB``, search that corpus.
    3. If the state key is **missing**, fall back to ``"common_core"``
       and set ``fallback=True`` on every returned record.
    4. Filter by ``grade`` and ``subject`` (exact match after stripping
       and upper-casing the subject).
    5. Score remaining standards by keyword overlap with ``topic``.
    6. Return standards sorted by relevance (highest overlap first).
       If zero keywords match, return **all** standards for that
       grade/subject so the teacher can pick manually.

    Parameters
    ----------
    state : str
        State name, abbreviation, or framework name (e.g., ``"Florida"``,
        ``"TX"``, ``"B.E.S.T."``).
    grade : str
        Grade level as a string (e.g., ``"5"``).
    subject : str
        Subject area (e.g., ``"ELA"``, ``"Math"``).
    topic : str
        Free-text topic the teacher intends to teach (e.g.,
        ``"main idea and supporting details"``).

    Returns
    -------
    list[dict[str, Any]]
        Each dict contains:
        - ``code`` (str): Standard code.
        - ``description`` (str): Full standard text.
        - ``strand`` (str): Curricular strand.
        - ``keywords`` (list[str]): Original keyword list.
        - ``source`` (str): Corpus key used (e.g., ``"florida"``).
        - ``fallback`` (bool): ``True`` if Common Core was used.
        - ``score`` (int): Keyword overlap score.

    Raises
    ------
    InvalidInputError
        If any required parameter is empty or whitespace-only.
    StandardsNotFoundError
        If no standards match the requested grade + subject in the
        resolved corpus (including after fallback).
    """
    # ----- 0.  Validate inputs ------------------------------------------------
    _validate_inputs(state=state, grade=grade, subject=subject, topic=topic)

    # ----- 1.  Resolve state key ----------------------------------------------
    state_key: str = _normalise_state(state)
    is_fallback: bool = False

    if state_key not in STANDARDS_DB:
        logger.warning(
            "State '%s' (resolved key '%s') not found in STANDARDS_DB. "
            "Falling back to Common Core.",
            state,
            state_key,
        )
        state_key = "common_core"
        is_fallback = True

    corpus: dict[str, dict[str, list[dict[str, Any]]]] = STANDARDS_DB[state_key]

    # ----- 2.  Filter by grade ------------------------------------------------
    grade_clean: str = grade.strip()

    if grade_clean not in corpus:
        # If the grade doesn't exist in the state corpus either, try
        # Common Core before raising.
        if not is_fallback and grade_clean in STANDARDS_DB.get("common_core", {}):
            logger.warning(
                "Grade '%s' not found in '%s' corpus. "
                "Falling back to Common Core.",
                grade_clean,
                state_key,
            )
            corpus = STANDARDS_DB["common_core"]
            state_key = "common_core"
            is_fallback = True
        else:
            raise StandardsNotFoundError(
                f"No standards found for grade '{grade_clean}' in the "
                f"'{state_key}' corpus (or Common Core fallback)."
            )

    grade_corpus: dict[str, list[dict[str, Any]]] = corpus[grade_clean]

    # ----- 3.  Filter by subject ----------------------------------------------
    subject_clean: str = subject.strip().upper()

    if subject_clean not in grade_corpus:
        raise StandardsNotFoundError(
            f"No standards found for subject '{subject_clean}' in grade "
            f"'{grade_clean}' of the '{state_key}' corpus.  "
            f"Available subjects: {list(grade_corpus.keys())}."
        )

    standards_pool: list[dict[str, Any]] = grade_corpus[subject_clean]

    # ----- 4.  Score by topic keyword overlap ---------------------------------
    topic_tokens: set[str] = _tokenise_topic(topic)

    scored_results: list[dict[str, Any]] = []
    for std in standards_pool:
        score: int = _score_standard(std, topic_tokens)
        scored_results.append(
            {
                **std,
                "source": state_key,
                "fallback": is_fallback,
                "score": score,
            }
        )

    # ----- 5.  Sort by score (descending) ------------------------------------
    scored_results.sort(key=lambda r: r["score"], reverse=True)

    logger.info(
        "Retrieved %d standard(s) for state='%s' grade='%s' subject='%s' "
        "topic='%s' (fallback=%s).",
        len(scored_results),
        state,
        grade_clean,
        subject_clean,
        topic,
        is_fallback,
    )

    return scored_results


# ==============================================================================
# Input validation (private)
# ==============================================================================

def _validate_inputs(
    *,
    state: str,
    grade: str,
    subject: str,
    topic: str,
) -> None:
    """Raise ``InvalidInputError`` if any required field is blank.

    Parameters
    ----------
    state, grade, subject, topic : str
        The four required retrieval parameters.

    Raises
    ------
    InvalidInputError
        With a message listing every blank field.
    """
    missing: list[str] = []

    if not state or not state.strip():
        missing.append("state")
    if not grade or not grade.strip():
        missing.append("grade")
    if not subject or not subject.strip():
        missing.append("subject")
    if not topic or not topic.strip():
        missing.append("topic")

    if missing:
        raise InvalidInputError(
            f"The following required field(s) are missing or empty: "
            f"{', '.join(missing)}.  All four parameters (state, grade, "
            f"subject, topic) must be non-empty strings."
        )
