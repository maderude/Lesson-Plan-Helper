"""
ui/persistence.py — Local Lesson Plan Persistence
===================================================

Provides auto-save and resume functionality so that generated lesson
plans survive page refreshes and browser tab closures.

Lessons are saved as timestamped JSON files in a `saved_lessons/`
directory at the project root.

Usage
-----
>>> from ui.persistence import save_lesson, load_latest_lesson, list_saved_lessons
>>> path = save_lesson(final_state, form_inputs)
>>> latest = load_latest_lesson()  # Returns None if no saved lessons
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Directory where lessons are persisted
_SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_lessons"


def _ensure_dir() -> None:
    """Create the saved_lessons directory if it doesn't exist."""
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)


def save_lesson(
    final_state: dict[str, Any],
    form_inputs: dict[str, Any],
) -> str:
    """Save the completed lesson state and form inputs to a JSON file.

    Parameters
    ----------
    final_state : dict
        The full workflow state after lesson generation completes.
    form_inputs : dict
        The teacher's sidebar form inputs (subject, grade, topic, etc.).

    Returns
    -------
    str
        The absolute file path of the saved JSON file.
    """
    _ensure_dir()

    # Build a clean, serializable snapshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subject = form_inputs.get("subject", "lesson")
    grade = form_inputs.get("grade", "")
    filename = f"lesson_{subject}_{grade}_{timestamp}.json"

    # Extract only serializable keys from final_state
    safe_state = {}
    for key, value in final_state.items():
        try:
            json.dumps(value)
            safe_state[key] = value
        except (TypeError, ValueError):
            safe_state[key] = str(value)

    payload = {
        "saved_at": datetime.now().isoformat(),
        "form_inputs": form_inputs,
        "final_state": safe_state,
    }

    filepath = _SAVE_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("persistence: Saved lesson to %s", filepath)
    return str(filepath)


def load_latest_lesson() -> dict[str, Any] | None:
    """Load the most recently saved lesson.

    Returns
    -------
    dict or None
        The parsed JSON payload with keys 'saved_at', 'form_inputs',
        and 'final_state'.  Returns None if no saved lessons exist.
    """
    _ensure_dir()

    json_files = sorted(
        _SAVE_DIR.glob("lesson_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not json_files:
        return None

    latest = json_files[0]
    try:
        with open(latest, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("persistence: Loaded latest lesson from %s", latest)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("persistence: Failed to load %s — %s", latest, exc)
        return None


def list_saved_lessons() -> list[dict[str, Any]]:
    """List all saved lessons with metadata (filename, date, subject, grade).

    Returns
    -------
    list[dict]
        Each dict has keys: 'filename', 'path', 'saved_at', 'subject', 'grade'.
    """
    _ensure_dir()

    results = []
    for path in sorted(
        _SAVE_DIR.glob("lesson_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            form = data.get("form_inputs", {})
            results.append({
                "filename": path.name,
                "path": str(path),
                "saved_at": data.get("saved_at", "Unknown"),
                "subject": form.get("subject", ""),
                "grade": form.get("grade", ""),
                "topic": form.get("topic", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue

    return results
