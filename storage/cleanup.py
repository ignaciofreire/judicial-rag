"""
cleanup.py

Cleanup routines for temporary session data.
Deletes temporary directories and ChromaDB collections when a session
ends explicitly or when the configured inactivity timeout is exceeded.
Designed to run both on explicit logout and on a periodic schedule.
"""

import logging
from datetime import UTC, datetime

from config.settings import settings
from pipeline.embedder import delete_collection
from storage.session_manager import list_sessions, remove_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cleanup_session(session_id: str) -> None:
    """Delete all data for a single session.

    Removes the temporary PDF directory from disk and deletes the session's
    ChromaDB collection. Safe to call multiple times — both operations are
    idempotent.

    Args:
        session_id: Unique session identifier to clean up.
    """
    from storage.session_manager import get_session

    session = get_session(session_id)
    if session is None:
        logger.debug("Cleanup called for unknown session: %s", session_id)
        return

    logger.info("Cleaning up session: %s", session_id)

    # Delete ChromaDB collection first so no orphaned vectors remain
    # if the directory deletion fails for any reason
    try:
        delete_collection(session_id)
    except Exception:
        logger.exception("Failed to delete ChromaDB collection for %s", session_id)

    # Release the temporary directory — this deletes all uploaded PDFs
    try:
        session.temp_dir.cleanup()
        logger.debug("Temporary directory deleted: %s", session.pdf_dir.parent)
    except Exception:
        logger.exception("Failed to delete temp directory for %s", session_id)

    remove_session(session_id)
    logger.info("Session cleaned up: %s", session_id)


def cleanup_expired_sessions() -> int:
    """Delete all sessions that have exceeded the configured timeout.

    Compares each session's created_at timestamp against the current UTC
    time. Sessions older than SESSION_TIMEOUT_MINUTES are cleaned up.

    Called periodically by the Streamlit UI to prevent orphaned sessions
    from accumulating on long-running deployments.

    Returns:
        Number of sessions that were cleaned up.
    """
    now = datetime.now(UTC)
    timeout_minutes = settings.session_timeout_minutes
    cleaned = 0

    for session in list_sessions():
        age_minutes = (now - session.created_at).total_seconds() / 60
        if age_minutes >= timeout_minutes:
            logger.info(
                "Session expired: %s (age=%.1f min timeout=%d min)",
                session.session_id,
                age_minutes,
                timeout_minutes,
            )
            cleanup_session(session.session_id)
            cleaned += 1

    if cleaned:
        logger.info("Expired session cleanup: %d session(s) removed", cleaned)
    else:
        logger.debug("No expired sessions found")

    return cleaned
