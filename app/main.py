"""
main.py

Entry point for the Streamlit application.
Initialises session state, renders sidebar components and orchestrates
the run button, progress display and results viewer.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app import session_state as state
from app.components import question_form, results_viewer, uploader
from models.query import QuestionSchema
from pipeline.orchestrator import ProgressEvent, Stage, run
from storage.cleanup import cleanup_expired_sessions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call in the script
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Judicial RAG",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in (
    "docling",
    "transformers",
    "huggingface_hub",
    "rapidocr",
    "httpx",
    "chromadb",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the full application."""
    state.init()
    cleanup_expired_sessions()

    st.title("⚖️ Judicial RAG")
    st.caption("Extract and classify variables from judicial PDF documents.")

    with st.sidebar:
        pdf_paths = uploader.render()
        st.divider()
        question_form.render()

    _render_run_section(pdf_paths)

    if state.results():
        results_viewer.render(state.results())


def _render_run_section(pdf_paths: list[Path]) -> None:
    """Render the run / reset buttons and the pipeline progress area.

    Args:
        pdf_paths: Validated PDF paths from the uploader component.
    """
    schema = state.schema()
    can_run = bool(pdf_paths) and schema is not None and not state.is_processing()

    col_run, col_reset = st.columns([2, 1])

    with col_run:
        if st.button(
            "Run pipeline",
            type="primary",
            disabled=not can_run,
            use_container_width=True,
            help="" if can_run else "Upload PDFs and save a schema first.",
        ):
            _run_pipeline(pdf_paths, schema)

    with col_reset:
        if st.button(
            "Reset",
            disabled=state.is_processing(),
            use_container_width=True,
        ):
            state.reset()
            st.rerun()


def _run_pipeline(pdf_paths: list[Path], schema: QuestionSchema) -> None:
    """Execute the full pipeline and stream progress updates to the UI.

    Uses asyncio.run() because Streamlit's execution model is synchronous
    at the script level. The pipeline itself is async internally to allow
    concurrent PDF indexing.

    Args:
        pdf_paths: Absolute paths to the PDFs to process.
        schema:    Question schema to apply to every document.
    """
    state.set_processing(True)
    state.set_results(None)

    bar = st.progress(0.0, text="Starting...")
    status = st.empty()

    def on_progress(event: ProgressEvent) -> None:
        # Update bar when we know the total, otherwise update status text
        if event.total > 0:
            bar.progress(
                min(event.current / event.total, 1.0),
                text=event.message,
            )
        else:
            status.info(event.message)

        if event.stage == Stage.ERROR:
            st.warning(f"⚠️ {event.message}")

    try:
        results = asyncio.run(
            run(
                pdf_paths=pdf_paths,
                schema=schema,
                session_id=state.session_id(),
                on_progress=on_progress,
                is_scanned=False,
            )
        )
        state.set_results(results)
        bar.progress(1.0, text="Complete.")
        st.success(f"Done — {len(results)} document(s) processed.")
        logger.info("Pipeline complete: %d documents", len(results))

    except Exception as e:
        logger.exception("Pipeline failed")
        st.error(f"Pipeline failed: {e}")

    finally:
        state.set_processing(False)
        st.rerun()


if __name__ == "__main__":
    main()
