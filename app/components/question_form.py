"""
question_form.py

Streamlit component for defining the question schema.
Renders a dynamic list of question editors supporting all four question
types. Classification questions get an extra category editor.
"""

import logging

import streamlit as st
from pydantic import ValidationError

from app import session_state as state
from models.query import Category, QuestionSchema, QuestionType, UserQuestion

logger = logging.getLogger(__name__)

# Human-readable labels for the type selectbox, mapped to QuestionType values.
# Ordered from most to least common for judicial extraction tasks.
_TYPE_OPTIONS: dict[str, QuestionType] = {
    "Extraction — copy a value directly from the text": QuestionType.EXTRACTION,
    "Calculation — derive a value from extracted data": QuestionType.CALCULATION,
    "Classification — assign to a category from a list": QuestionType.CLASSIFICATION,
    "Explanation — explain why or how something happened": QuestionType.EXPLANATION,
}
_TYPE_VALUES = list(_TYPE_OPTIONS.values())
_TYPE_KEYS = list(_TYPE_OPTIONS.keys())

# Internal session state key for the list of in-progress question drafts.
# Separate from the saved schema so edits do not affect a running pipeline.
_DRAFTS_KEY = "question_drafts"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render() -> QuestionSchema | None:
    """Render the question schema editor and return the saved schema.

    Manages a list of draft dicts in st.session_state. Each draft maps
    directly to one UserQuestion. The schema is only built and validated
    when the user clicks Save.

    Returns:
        The saved QuestionSchema, or None if not yet saved or invalid.
    """
    st.subheader("Define questions")

    if _DRAFTS_KEY not in st.session_state:
        st.session_state[_DRAFTS_KEY] = [_empty_draft()]

    drafts: list[dict] = st.session_state[_DRAFTS_KEY]

    for i, draft in enumerate(drafts):
        _render_editor(i, draft)

    col_add, col_save = st.columns([1, 3])

    with col_add:
        if st.button(
            "Add question",
            disabled=state.is_processing(),
            use_container_width=True,
        ):
            drafts.append(_empty_draft())
            st.rerun()

    with col_save:
        if st.button(
            "Save schema",
            type="primary",
            disabled=state.is_processing(),
            use_container_width=True,
        ):
            schema = _build_schema(drafts)
            if schema is not None:
                state.set_schema(schema)
                st.success(f"Schema saved — {len(schema.questions)} question(s).")
                logger.info("Schema saved: %d questions", len(schema.questions))
            return schema

    return state.schema()


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _render_editor(idx: int, draft: dict) -> None:
    """Render the editor widgets for a single question draft.

    All widgets write back to the draft dict directly, so changes
    persist across Streamlit reruns without explicit save actions.

    Args:
        idx:   Position of this question in the drafts list (0-based).
        draft: Mutable dict holding the current values for this question.
    """
    with st.expander(
        label=draft.get("label") or f"Question {idx + 1}",
        expanded=draft.get("expanded", True),
    ):
        col_label, col_type = st.columns(2)

        with col_label:
            draft["label"] = st.text_input(
                "Label",
                value=draft.get("label", ""),
                key=f"label_{idx}",
                placeholder="e.g. Sentencing date",
                disabled=state.is_processing(),
            )

        with col_type:
            selected_key = st.selectbox(
                "Type",
                options=_TYPE_KEYS,
                index=_TYPE_VALUES.index(
                    draft.get("question_type", QuestionType.EXTRACTION)
                ),
                key=f"type_{idx}",
                disabled=state.is_processing(),
            )
            draft["question_type"] = _TYPE_OPTIONS[selected_key]

        draft["question"] = st.text_area(
            "Question / extraction rule",
            value=draft.get("question", ""),
            key=f"question_{idx}",
            placeholder="e.g. What is the date of the sentence? Format: DD/MM/YYYY",
            disabled=state.is_processing(),
        )

        col_fmt, col_notes = st.columns(2)

        with col_fmt:
            draft["output_format"] = st.text_input(
                "Expected output format (optional)",
                value=draft.get("output_format", ""),
                key=f"fmt_{idx}",
                placeholder="e.g. DD/MM/YYYY, integer",
                disabled=state.is_processing(),
            )

        with col_notes:
            draft["notes"] = st.text_input(
                "Additional rules (optional)",
                value=draft.get("notes", ""),
                key=f"notes_{idx}",
                placeholder="e.g. Round to nearest integer",
                disabled=state.is_processing(),
            )

        if draft["question_type"] == QuestionType.CLASSIFICATION:
            _render_category_editor(idx, draft)

        if st.button(
            "Remove question",
            key=f"remove_{idx}",
            disabled=state.is_processing(),
        ):
            st.session_state[_DRAFTS_KEY].pop(idx)
            st.rerun()


def _render_category_editor(idx: int, draft: dict) -> None:
    """Render the category list editor for classification questions.

    Args:
        idx:   Question index for unique widget keys.
        draft: Mutable dict holding the current field values.
    """
    st.markdown("**Categories**")

    if "categories" not in draft:
        draft["categories"] = [{"code": "", "label": ""}]

    for j, cat in enumerate(draft["categories"]):
        col_code, col_label, col_rm = st.columns([1, 3, 1])

        with col_code:
            cat["code"] = st.text_input(
                "Code",
                value=cat.get("code", ""),
                key=f"cat_code_{idx}_{j}",
                placeholder="1",
                disabled=state.is_processing(),
            )

        with col_label:
            cat["label"] = st.text_input(
                "Label",
                value=cat.get("label", ""),
                key=f"cat_label_{idx}_{j}",
                placeholder="Conviction",
                disabled=state.is_processing(),
            )

        with col_rm:
            st.write("")
            if st.button("✕", key=f"rm_cat_{idx}_{j}"):
                draft["categories"].pop(j)
                st.rerun()

    if st.button(
        "Add category",
        key=f"add_cat_{idx}",
        disabled=state.is_processing(),
    ):
        draft["categories"].append({"code": "", "label": ""})
        st.rerun()


def _build_schema(drafts: list[dict]) -> QuestionSchema | None:
    """Validate draft dicts and build a QuestionSchema.

    Displays a user-facing error for the first invalid question rather
    than raising, so the user can correct the issue without losing their
    other question definitions.

    Args:
        drafts: List of draft dicts from the question editors.

    Returns:
        Valid QuestionSchema, or None if any draft fails validation.
    """
    questions: list[UserQuestion] = []

    for i, draft in enumerate(drafts):
        try:
            categories = None
            if draft.get("question_type") == QuestionType.CLASSIFICATION:
                categories = [
                    Category(code=c["code"], label=c["label"])
                    for c in draft.get("categories", [])
                    if c.get("code") and c.get("label")
                ] or None

            questions.append(
                UserQuestion(
                    label=draft.get("label", "").strip(),
                    question=draft.get("question", "").strip(),
                    question_type=draft.get("question_type", QuestionType.EXTRACTION),
                    categories=categories,
                    output_format=draft.get("output_format") or None,
                    notes=draft.get("notes") or None,
                )
            )

        except ValidationError as e:
            st.error(f"Question {i + 1}: {e.errors()[0]['msg']}")
            logger.warning("Question %d invalid: %s", i + 1, e)
            return None

    if not questions:
        st.error("Add at least one question before saving.")
        return None

    try:
        return QuestionSchema(name="User schema", questions=questions)
    except ValidationError as e:
        st.error(f"Schema error: {e.errors()[0]['msg']}")
        return None


def _empty_draft() -> dict:
    """Return a blank draft dict with default values for a new question.

    Returns:
        Dict with sensible defaults for every question editor field.
    """
    return {
        "label": "",
        "question": "",
        "question_type": QuestionType.EXTRACTION,
        "output_format": "",
        "notes": "",
        "categories": [],
        "expanded": True,
    }
