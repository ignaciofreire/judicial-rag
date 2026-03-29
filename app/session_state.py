"""
session_state.py

Centralised Streamlit session state initialisation and access.
All st.session_state keys live here so they are never scattered across
components. Using string constants instead of raw keys prevents typos
and makes renaming straightforward.
"""

import uuid

import streamlit as st

from models.query import QuestionSchema
from storage.session_manager import get_or_create_session

# ---------------------------------------------------------------------------
# State keys
# ---------------------------------------------------------------------------
# One constant per key. Every component imports from here rather than
# using raw strings, so a rename is a single-line change.

_SESSION_ID = "session_id"
_SCHEMA = "schema"
_RESULTS = "results"
_IS_PROCESSING = "is_processing"
_UPLOADED_FILES = "uploaded_files"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init() -> None:
    """Initialise all session state keys on the first page load.

    Streamlit reruns the full script on every interaction. setdefault
    ensures existing values survive reruns — only absent keys are set.
    Called once at the top of app/main.py before any component renders.
    """
    if _SESSION_ID not in st.session_state:
        # Generate a stable UUID for this browser session. Used as the
        # ChromaDB collection name and temporary directory prefix so each
        # user's data stays fully isolated from other users.
        st.session_state[_SESSION_ID] = str(uuid.uuid4())

    st.session_state.setdefault(_SCHEMA, None)
    st.session_state.setdefault(_RESULTS, None)
    st.session_state.setdefault(_IS_PROCESSING, False)
    st.session_state.setdefault(_UPLOADED_FILES, [])

    # Ensure the backend Session object exists for this ID so the
    # temporary directory and ChromaDB collection are ready before
    # any component tries to use them
    get_or_create_session(st.session_state[_SESSION_ID])


# ---------------------------------------------------------------------------
# Accessors and mutators
# ---------------------------------------------------------------------------
# Thin wrappers around st.session_state. Components call these instead of
# accessing st.session_state directly so the key names stay encapsulated.


def session_id() -> str:
    """Return the UUID identifying this browser session.

    Returns:
        UUID string set at session initialisation.
    """
    return st.session_state[_SESSION_ID]


def schema() -> QuestionSchema | None:
    """Return the current question schema, or None if not yet defined.

    Returns:
        QuestionSchema instance, or None before the user saves a schema.
    """
    return st.session_state[_SCHEMA]


def set_schema(value: QuestionSchema) -> None:
    """Persist the question schema for the current session.

    Args:
        value: QuestionSchema built from the user's question form.
    """
    st.session_state[_SCHEMA] = value


def results() -> list | None:
    """Return the pipeline results, or None if the pipeline has not run.

    Returns:
        List of DocumentAnswers from the orchestrator, or None.
    """
    return st.session_state[_RESULTS]


def set_results(value: list) -> None:
    """Persist the pipeline results for display in the results viewer.

    Args:
        value: List of DocumentAnswers returned by orchestrator.run().
    """
    st.session_state[_RESULTS] = value


def is_processing() -> bool:
    """Return True while the pipeline is running.

    Used by the UI to disable buttons and show a spinner during processing.

    Returns:
        True if the pipeline is active, False otherwise.
    """
    return st.session_state[_IS_PROCESSING]


def set_processing(value: bool) -> None:
    """Set the processing flag.

    Args:
        value: True when the pipeline starts, False when it completes
            or fails.
    """
    st.session_state[_IS_PROCESSING] = value


def uploaded_files() -> list:
    """Return the list of files uploaded in the current session.

    Returns:
        List of Streamlit UploadedFile objects, empty before any upload.
    """
    return st.session_state[_UPLOADED_FILES]


def set_uploaded_files(value: list) -> None:
    """Persist the uploaded file objects for use by the pipeline.

    Args:
        value: List of Streamlit UploadedFile objects from st.file_uploader.
    """
    st.session_state[_UPLOADED_FILES] = value


def reset() -> None:
    """Clear results and uploads to prepare for a new pipeline run.

    Intentionally keeps the session ID and schema intact so the user
    does not need to redefine questions when uploading a new batch of PDFs.
    """
    st.session_state[_RESULTS] = None
    st.session_state[_UPLOADED_FILES] = []
    st.session_state[_IS_PROCESSING] = False
