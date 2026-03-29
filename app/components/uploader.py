"""
uploader.py

Streamlit component for PDF upload.
Validates uploaded files, saves accepted ones to the session temporary
directory and returns their absolute paths for the pipeline.
"""

import logging
from pathlib import Path

import streamlit as st

from app import session_state as state
from storage.session_manager import get_or_create_session

logger = logging.getLogger(__name__)

# PDFs larger than this are rejected before extraction to avoid memory
# exhaustion in Docling and the HF Inference API.
_MAX_MB = 20
_MAX_BYTES = _MAX_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render() -> list[Path]:
    """Render the PDF upload widget and return paths to accepted files.

    Validates each file for type (enforced by Streamlit) and size.
    Accepted files are written to the session temporary directory once
    and reused on subsequent reruns without re-writing.

    Returns:
        Absolute paths to the saved PDFs, ready for the pipeline.
        Empty list if no valid files have been uploaded yet.
    """
    st.subheader("Upload documents")

    uploaded = st.file_uploader(
        label="Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        disabled=state.is_processing(),
        help=f"Maximum {_MAX_MB} MB per file.",
    )

    if not uploaded:
        st.info("Upload at least one PDF to get started.")
        return []

    session = get_or_create_session(state.session_id())
    paths: list[Path] = []
    rejected: list[str] = []

    for file in uploaded:
        if file.size > _MAX_BYTES:
            rejected.append(
                f"{file.name} "
                f"({file.size / 1024 / 1024:.1f} MB — limit {_MAX_MB} MB)"
            )
            logger.warning(
                "Rejected oversized file: %s (%d bytes)", file.name, file.size
            )
            continue

        dest = session.pdf_dir / file.name
        if not dest.exists():
            # Write once per session — skip on Streamlit reruns to avoid
            # re-writing the same bytes on every user interaction
            dest.write_bytes(file.getvalue())
            logger.debug("Saved: %s", dest)

        paths.append(dest)

    if rejected:
        st.error(
            "Files rejected (size limit exceeded):\n"
            + "\n".join(f"- {r}" for r in rejected)
        )

    if paths:
        st.success(f"{len(paths)} file(s) ready: " + ", ".join(p.name for p in paths))

    state.set_uploaded_files(uploaded)
    return paths
