"""
results_viewer.py

Streamlit component for displaying pipeline results.
Renders answers grouped by document with confidence badges, answer
provenance and the source citation fragment from the original text.
"""

import logging

import streamlit as st

from models.query import AgentAnswer, AnswerConfidence, AnswerSource, DocumentAnswers

logger = logging.getLogger(__name__)

# Streamlit colour names for each confidence level.
# Rendered as coloured text badges: :green[high], :orange[medium], etc.
_CONFIDENCE_COLOR: dict[AnswerConfidence, str] = {
    AnswerConfidence.HIGH: "green",
    AnswerConfidence.MEDIUM: "orange",
    AnswerConfidence.LOW: "red",
    AnswerConfidence.NOT_FOUND: "gray",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render(results: list[DocumentAnswers]) -> None:
    """Render pipeline results grouped by document.

    One expander per document, with all answers displayed inside it.
    The expander label shows how many questions were answered to give
    the user an immediate summary without opening every document.

    Args:
        results: List of DocumentAnswers returned by the orchestrator.
    """
    if not results:
        st.info("No results yet. Run the pipeline first.")
        return

    st.subheader("Results")

    for doc in results:
        _render_document(doc)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _render_document(doc: DocumentAnswers) -> None:
    """Render all answers for a single document inside an expander.

    Args:
        doc: Answers for one PDF document.
    """
    answered = sum(1 for a in doc.answers if a.confidence != AnswerConfidence.NOT_FOUND)
    total = len(doc.answers)

    with st.expander(
        label=f"{doc.document} — {answered}/{total} answered",
        expanded=True,
    ):
        for answer in doc.answers:
            _render_answer(answer)
            st.divider()


def _render_answer(answer: AgentAnswer) -> None:
    """Render a single answer with confidence badge, text and citation.

    Args:
        answer: One AgentAnswer from the RAG agent.
    """
    col_label, col_badge = st.columns([4, 1])

    with col_label:
        st.markdown(f"**{answer.question.label}**")

    with col_badge:
        color = _CONFIDENCE_COLOR.get(answer.confidence, "gray")
        st.markdown(
            f":{color}[{answer.confidence.value}]",
            help="Confidence assigned by the agent.",
        )

    if answer.answer:
        st.write(answer.answer)
    else:
        st.caption("No answer found.")

    if answer.answer_source:
        label = (
            "Extracted directly"
            if answer.answer_source == AnswerSource.DIRECT
            else "Inferred through reasoning"
        )
        st.caption(f"Source: {label}")

    if answer.citation:
        with st.container(border=True):
            st.caption(
                f"Page {answer.citation.page} — " f"score {answer.citation.score:.2f}"
            )
            st.text(answer.citation.text)
