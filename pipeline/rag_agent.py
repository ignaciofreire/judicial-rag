"""
rag_agent.py

RAG agent for answering user-defined questions over judicial PDF chunks.
Retrieves relevant chunks from the vector store and calls the LLM to
produce a structured answer with confidence level, source citation and
answer provenance (direct extraction vs inference).

Each QuestionType maps to a dedicated prompt strategy:
    EXTRACTION     — locate and copy a specific value from the text.
    CALCULATION    — extract operands and compute a derived value.
    CLASSIFICATION — map evidence to one category from a closed list.
    EXPLANATION    — synthesise an explanation from multiple fragments.
"""

import json
import logging

from models.query import (
    AgentAnswer,
    AnswerConfidence,
    AnswerSource,
    Citation,
    QuestionType,
    UserQuestion,
)
from pipeline.vector_store import search

logger = logging.getLogger(__name__)

# Number of chunks retrieved per question. Higher values improve recall
# for questions that span multiple sections, at the cost of a larger
# context window sent to the LLM.
_N_RESULTS = 5

# Confidence and answer_source values the LLM is instructed to return.
# Defined as constants so prompt and parser stay in sync automatically.
_CONFIDENCE_HIGH = "high"
_CONFIDENCE_MEDIUM = "medium"
_CONFIDENCE_LOW = "low"
_CONFIDENCE_NOT_FOUND = "not_found"
_SOURCE_DIRECT = "direct"
_SOURCE_INFERRED = "inferred"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def answer_question(
    question: UserQuestion,
    session_id: str,
    source: str,
) -> AgentAnswer:
    """Answer a single user question for a specific document.

    Retrieves relevant chunks, builds a type-specific prompt, calls the LLM,
    and returns a structured answer with confidence, citation and provenance.

    Args:
        question: User-defined question including type, rules and categories.
        session_id: User session identifier for vector store isolation.
        source: PDF filename to restrict chunk retrieval to.

    Returns:
        AgentAnswer with answer text, citation and confidence level.
        Returns NOT_FOUND with None fields if no relevant chunks are found.
    """
    logger.info(
        "Answering %r (type=%s source=%s session=%s)",
        question.label,
        question.question_type.value,
        source,
        session_id,
    )

    chunks = search(
        question=question.question,
        session_id=session_id,
        source=source,
        n_results=_N_RESULTS,
    )

    if not chunks:
        logger.info("No chunks found for %r in %s", question.label, source)
        return AgentAnswer(
            question=question,
            document=source,
            confidence=AnswerConfidence.NOT_FOUND,
        )

    logger.debug(
        "Retrieved %d chunks for %r (top distance=%.4f)",
        len(chunks),
        question.label,
        chunks[0]["distance"],
    )

    raw = _call_llm(question, chunks)
    answer = _parse_response(raw, question, source, chunks)

    logger.info(
        "Answered %r — confidence=%s answer_source=%s",
        question.label,
        answer.confidence.value,
        answer.answer_source.value if answer.answer_source else "none",
    )
    return answer


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt(question: UserQuestion, chunks: list[dict]) -> str:
    """Build the LLM prompt tailored to the question type.

    The prompt has four sections: the retrieved context with headings,
    a type-specific task instruction, the question with optional notes
    and categories, and a strict JSON output specification.

    Args:
        question: User question with type, rules and optional categories.
        chunks: Retrieved chunks ordered by descending relevance.

    Returns:
        Complete prompt string ready to send to the LLM.
    """
    return "\n".join(
        filter(
            None,
            [
                "You are an expert judicial analyst extracting information from "
                "Spanish court rulings.",
                "",
                "## Retrieved fragments",
                "",
                _format_context(chunks),
                "## Task",
                "",
                _build_instruction(question),
                "",
                "## Question",
                "",
                question.question,
                "",
                _format_notes(question),
                _format_categories(question),
                _build_output_spec(),
            ],
        )
    )


def _build_instruction(question: UserQuestion) -> str:
    """Return the task instruction for the question type.

    Each instruction tells the LLM exactly how to approach the question
    and which answer_source value to use in the response.

    Args:
        question: User question with type.

    Returns:
        Instruction paragraph for the prompt.
    """
    instructions: dict[QuestionType, str] = {
        QuestionType.EXTRACTION: (
            "Extract the specific value requested directly from the text. "
            "Copy it verbatim where possible — do not paraphrase or interpret. "
            "Use answer_source='direct'."
        ),
        QuestionType.CALCULATION: (
            "Extract the operands needed from the text, compute the result "
            "following the rules provided, and show your reasoning in the "
            "answer field. Use answer_source='inferred'."
        ),
        QuestionType.CLASSIFICATION: (
            "Read the fragments and assign the content to exactly one category "
            "from the list below. Return only the category code in the answer "
            "field. Use answer_source='direct' if the text explicitly matches "
            "a category, or 'inferred' if you are mapping implicit evidence."
        ),
        QuestionType.EXPLANATION: (
            "Synthesise a concise explanation in Spanish from the retrieved "
            "fragments. You may combine information from multiple fragments. "
            "Use answer_source='inferred'."
        ),
    }
    return instructions.get(
        question.question_type,
        "Answer the question using only the retrieved fragments.",
    )


def _build_output_spec() -> str:
    """Return the JSON output specification block appended to every prompt.

    Centralised here so changes to the schema propagate to all question
    types automatically without touching the individual instruction strings.

    Returns:
        Output specification string including field rules and examples.
    """
    return f"""## Output format
Respond with a valid JSON object and nothing else. No markdown, no preamble.

{{
    "answer": "<your answer or null if not found>",
    "confidence": "<{_CONFIDENCE_HIGH} | {_CONFIDENCE_MEDIUM} |
      {_CONFIDENCE_LOW} | {_CONFIDENCE_NOT_FOUND}>",
    "answer_source": "<{_SOURCE_DIRECT} | {_SOURCE_INFERRED}>",
    "citation": "<exact fragment from the text that supports the answer, or null>"
}}

Confidence rules:
- {_CONFIDENCE_HIGH}: answer is explicitly and unambiguously stated in the text.
- {_CONFIDENCE_MEDIUM}: answer is present but requires some interpretation.
- {_CONFIDENCE_LOW}: answer is inferred from indirect or weak evidence.
- {_CONFIDENCE_NOT_FOUND}: no relevant information found in the fragments.

Answer source rules:
- {_SOURCE_DIRECT}: answer copied or minimally transformed from a specific fragment.
- {_SOURCE_INFERRED}: answer derived through reasoning, calculation or synthesis."""


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt.

    Each chunk is prefixed with its index, page number and section heading
    so the LLM knows where in the document each fragment comes from.

    Args:
        chunks: Retrieved chunks ordered by descending relevance.

    Returns:
        Formatted multi-line context string.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        heading = " > ".join(chunk["headings"]) if chunk["headings"] else "No section"
        lines.append(f"[{i}] Page {chunk['page']} — {heading}\n{chunk['text']}\n")
    return "\n".join(lines)


def _format_notes(question: UserQuestion) -> str:
    """Format the optional notes field as an additional rules block.

    Args:
        question: User question with optional notes.

    Returns:
        Formatted notes block or empty string if no notes are defined.
    """
    if not question.notes:
        return ""
    return f"## Additional rules\n\n{question.notes}\n"


def _format_categories(question: UserQuestion) -> str:
    """Format the category list for classification questions.

    Args:
        question: User question with optional categories.

    Returns:
        Formatted categories block or empty string for non-classification types.
    """
    if not question.categories:
        return ""
    lines = ["## Valid categories\n"]
    lines += [f"- {cat.code}: {cat.label}" for cat in question.categories]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _call_llm(question: UserQuestion, chunks: list[dict]) -> str:
    """Build the prompt and call the configured LLM provider.

    The LLM client is imported lazily to avoid circular imports and to keep
    this module loadable even when LLM credentials are not configured.

    Args:
        question: User question used to build the prompt.
        chunks: Retrieved chunks to include as context.

    Returns:
        Raw LLM response string expected to contain valid JSON.

    Raises:
        RuntimeError: If the LLM call fails for any reason.
    """
    from services.llm_client import call_llm

    prompt = _build_prompt(question, chunks)
    logger.debug(
        "Calling LLM (type=%s prompt_len=%d)",
        question.question_type.value,
        len(prompt),
    )

    try:
        response = call_llm(prompt)
        logger.debug("LLM response received (%d chars)", len(response))
        return response
    except Exception as e:
        logger.exception("LLM call failed for question %r", question.label)
        raise RuntimeError(f"LLM call failed: {e}") from e


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(
    raw: str,
    question: UserQuestion,
    source: str,
    chunks: list[dict],
) -> AgentAnswer:
    """Parse the LLM JSON response into a structured AgentAnswer.

    Falls back to NOT_FOUND on JSON parse errors rather than raising, since
    a malformed LLM response should not crash the pipeline for an entire
    document — the remaining questions still get processed.

    Args:
        raw: Raw LLM response string expected to contain valid JSON.
        question: Original user question.
        source: PDF filename this answer refers to.
        chunks: Retrieved chunks used to build the citation metadata.

    Returns:
        Structured AgentAnswer with confidence, citation and provenance.
    """
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        logger.warning(
            "Non-JSON LLM response for %r — defaulting to NOT_FOUND. Preview: %r",
            question.label,
            raw[:200],
        )
        return AgentAnswer(
            question=question,
            document=source,
            confidence=AnswerConfidence.NOT_FOUND,
        )

    citation = _build_citation(data.get("citation"), source, chunks)

    return AgentAnswer(
        question=question,
        answer=data.get("answer"),
        citation=citation,
        document=source,
        confidence=_parse_confidence(data.get("confidence", _CONFIDENCE_NOT_FOUND)),
        answer_source=_parse_answer_source(data.get("answer_source")),
    )


def _build_citation(
    citation_text: str | None,
    source: str,
    chunks: list[dict],
) -> Citation | None:
    """Build a Citation from the LLM-selected fragment and chunk metadata.

    The similarity score is derived from the top chunk's cosine distance
    (distance 0 = identical → score 1.0, distance 1 = orthogonal → score 0.0).

    Args:
        citation_text: Exact text fragment selected by the LLM, or None.
        source: PDF filename for the citation source field.
        chunks: Retrieved chunks; the first is used for page and score.

    Returns:
        Citation instance, or None if citation_text is absent.
    """
    if not citation_text or not chunks:
        return None

    best = chunks[0]
    return Citation(
        text=citation_text,
        source=source,
        page=best["page"],
        # Clamp to [0, 1] in case of floating point edge cases
        score=max(0.0, min(1.0, 1.0 - best["distance"])),
    )


def _parse_confidence(value: str) -> AnswerConfidence:
    """Parse a confidence string into an AnswerConfidence enum value.

    Falls back to NOT_FOUND on unrecognised values — a bad confidence label
    from the LLM should not raise, only log a warning.

    Args:
        value: Raw confidence string from the LLM JSON response.

    Returns:
        Matching AnswerConfidence member, or NOT_FOUND as fallback.
    """
    try:
        return AnswerConfidence(value.lower())
    except (ValueError, AttributeError):
        logger.warning("Unrecognised confidence %r — defaulting to NOT_FOUND", value)
        return AnswerConfidence.NOT_FOUND


def _parse_answer_source(value: str | None) -> AnswerSource | None:
    """Parse an answer_source string into an AnswerSource enum value.

    Args:
        value: Raw answer_source string from the LLM JSON response, or None.

    Returns:
        Matching AnswerSource member, or None if absent or unrecognised.
    """
    if not value:
        return None
    try:
        return AnswerSource(value.lower())
    except (ValueError, AttributeError):
        logger.warning("Unrecognised answer_source %r", value)
        return None
