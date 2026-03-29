"""
session_manager.py

Session lifecycle manager.
Creates and tracks isolated temporary directories for each user session.
Each session gets a unique ID generated at startup and a dedicated
temporary directory that stores uploaded PDFs during processing.
"""

import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session data
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """All state associated with a single user session.

    Attributes:
        session_id:  Unique identifier used for ChromaDB collection isolation
                     and temporary directory naming.
        temp_dir:    TemporaryDirectory instance. Holds the context manager
                     that controls the directory's lifetime on disk.
        pdf_dir:     Path to the directory where uploaded PDFs are saved.
                     Subdirectory of temp_dir to keep uploads isolated.
        created_at:  UTC timestamp of session creation. Used by cleanup.py
                     to detect sessions that have exceeded the timeout.
    """

    session_id: str
    temp_dir: tempfile.TemporaryDirectory
    pdf_dir: Path
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------
# Maps session_id → Session for all active sessions. A plain dict is safe
# here because Streamlit runs each user in a single thread.

_sessions: dict[str, Session] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_session() -> Session:
    """Create a new session with a unique ID and a temporary directory.

    The temporary directory is managed by Python's tempfile module, which
    guarantees cleanup on process exit even if cleanup.py is never called.
    The pdf_dir subdirectory is created inside it so uploaded files stay
    isolated from any other temporary files.

    Returns:
        Newly created Session instance registered in the active sessions dict.
    """
    session_id = str(uuid.uuid4())
    temp_dir = tempfile.TemporaryDirectory(prefix=f"judicial_rag_{session_id}_")
    pdf_dir = Path(temp_dir.name) / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    session = Session(
        session_id=session_id,
        temp_dir=temp_dir,
        pdf_dir=pdf_dir,
    )
    _sessions[session_id] = session

    logger.info("Session created: %s (pdf_dir=%s)", session_id, pdf_dir)
    return session


def get_session(session_id: str) -> Session | None:
    """Retrieve an active session by ID.

    Args:
        session_id: Unique session identifier.

    Returns:
        Session instance if found, None if the session does not exist or
        has already been cleaned up.
    """
    session = _sessions.get(session_id)
    if session is None:
        logger.debug("Session not found: %s", session_id)
    return session


def get_or_create_session(session_id: str) -> Session:
    """Retrieve an existing session or create a new one with the given ID.

    Used by the Streamlit UI to restore a session after a page rerun.
    Streamlit reruns the entire script on every interaction, so the session
    must persist across reruns via st.session_state.

    Args:
        session_id: Unique session identifier stored in st.session_state.

    Returns:
        Existing Session if found, otherwise a newly created one.
    """
    session = _sessions.get(session_id)
    if session is not None:
        logger.debug("Session restored: %s", session_id)
        return session

    # Session was lost — recreate it with the same ID so ChromaDB collections
    # remain accessible under the same name
    temp_dir = tempfile.TemporaryDirectory(prefix=f"judicial_rag_{session_id}_")
    pdf_dir = Path(temp_dir.name) / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    session = Session(
        session_id=session_id,
        temp_dir=temp_dir,
        pdf_dir=pdf_dir,
    )
    _sessions[session_id] = session

    logger.info("Session recreated: %s (pdf_dir=%s)", session_id, pdf_dir)
    return session


def list_sessions() -> list[Session]:
    """Return all currently active sessions.

    Used by cleanup.py to find sessions that have exceeded the timeout.

    Returns:
        List of active Session instances.
    """
    return list(_sessions.values())


def remove_session(session_id: str) -> None:
    """Remove a session from the registry without deleting its files.

    File deletion is handled by cleanup.py which calls temp_dir.cleanup()
    explicitly. This function only removes the session from the registry
    so it is no longer returned by list_sessions or get_session.

    Args:
        session_id: Unique session identifier to remove.
    """
    session = _sessions.pop(session_id, None)
    if session is None:
        logger.debug("Remove called for unknown session: %s", session_id)
        return
    logger.info("Session removed from registry: %s", session_id)
