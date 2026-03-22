"""
test_query.py

Unit tests for models/query.py.

Run all: uv run pytest tests/test_query.py -v
Run a class: uv run pytest tests/test_query.py::TestUserQuestion -v
"""

import pytest
from pydantic import ValidationError

from models.query import (
    AgentAnswer,
    AnswerConfidence,
    AnswerSource,
    Category,
    Citation,
    DocumentAnswers,
    QuestionSchema,
    QuestionType,
    UserQuestion,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extraction_question() -> UserQuestion:
    """Minimal valid extraction question. Used as a base object across tests."""
    return UserQuestion(
        label="Sentencing date",
        question="What is the date of the sentence?",
        question_type=QuestionType.EXTRACTION,
        output_format="DD/MM/YYYY",
    )


@pytest.fixture
def classification_question() -> UserQuestion:
    """Valid classification question with two categories. Tests the happy path
    for the categories validator."""
    return UserQuestion(
        label="Ruling type",
        question="What is the type of ruling?",
        question_type=QuestionType.CLASSIFICATION,
        categories=[
            Category(code="1", label="Conviction"),
            Category(code="2", label="Acquittal"),
        ],
    )


@pytest.fixture
def citation() -> Citation:
    """Valid citation with a realistic score and page number."""
    return Citation(
        text="En Madrid, a 3 de marzo de 2024, el tribunal...",
        source="sentencia_001.pdf",
        page=1,
        score=0.92,
    )


@pytest.fixture
def agent_answer(extraction_question: UserQuestion, citation: Citation) -> AgentAnswer:
    """Fully populated AgentAnswer. Depends on extraction_question and citation
    fixtures — pytest resolves and injects both automatically."""
    return AgentAnswer(
        question=extraction_question,
        answer="03/03/2024",
        citation=citation,
        document="sentencia_001.pdf",
        confidence=AnswerConfidence.HIGH,
        answer_source=AnswerSource.DIRECT,
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_all_enums_serialise_to_strings(self) -> None:
        # Enum values are stored in JSON and passed to the LLM.
        # Inheriting from str ensures serialisation works without
        # a custom serialiser — this test guards against accidental regressions
        # if the enum base class is changed.
        for enum_class in (QuestionType, AnswerConfidence, AnswerSource):
            for member in enum_class:
                assert isinstance(member.value, str)


# ---------------------------------------------------------------------------
# UserQuestion
# ---------------------------------------------------------------------------


class TestUserQuestion:
    @pytest.mark.parametrize("field", ["label", "question"])
    def test_empty_required_field_raises(self, field: str) -> None:
        # Both label and question drive the agent prompt — an empty value
        # would produce meaningless output without surfacing an obvious error.
        # parametrize runs this test twice, once per field, without duplication.
        with pytest.raises(ValidationError):
            UserQuestion(
                **{  # type: ignore[arg-type]
                    "label": "Sentencing date",
                    "question": "What is the date?",
                    "question_type": QuestionType.EXTRACTION,
                    field: "",
                }
            )

    def test_classification_requires_at_least_two_categories(self) -> None:
        # A single category makes classification trivial and likely a misconfiguration.
        # The validator catches this early so the error surfaces at schema definition
        # time rather than silently producing wrong outputs at inference time.
        with pytest.raises(ValidationError, match="at least two categories"):
            UserQuestion(
                label="Ruling type",
                question="What is the type of ruling?",
                question_type=QuestionType.CLASSIFICATION,
                categories=[Category(code="1", label="Conviction")],
            )

    def test_non_classification_with_categories_raises(self) -> None:
        # Defining categories on a non-classification question indicates a
        # misconfiguration that would confuse the agent. Rejected at validation
        # time to prevent silent prompt pollution.
        with pytest.raises(ValidationError, match="should not define categories"):
            UserQuestion(
                label="Sentencing date",
                question="What is the date?",
                question_type=QuestionType.EXTRACTION,
                categories=[
                    Category(code="1", label="Conviction"),
                    Category(code="2", label="Acquittal"),
                ],
            )

    def test_valid_classification_question(
        self, classification_question: UserQuestion
    ) -> None:
        # Confirms the happy path: a classification question with two categories
        # passes validation and retains both categories.
        assert classification_question.question_type == QuestionType.CLASSIFICATION
        assert len(classification_question.categories) == 2  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# QuestionSchema
# ---------------------------------------------------------------------------


class TestQuestionSchema:
    def test_empty_name_raises(self, extraction_question: UserQuestion) -> None:
        # An unnamed schema cannot be identified in the UI or in stored JSON.
        with pytest.raises(ValidationError):
            QuestionSchema(name="", questions=[extraction_question])

    def test_empty_questions_raises(self) -> None:
        # A schema with no questions would cause the pipeline to do nothing,
        # which is always a configuration error.
        with pytest.raises(ValidationError):
            QuestionSchema(name="Rulings schema v1", questions=[])


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


class TestCitation:
    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_score_out_of_range_raises(self, score: float) -> None:
        # Score represents cosine similarity normalised to [0, 1].
        # Values outside this range indicate a bug in the embedding pipeline.
        # parametrize tests both boundary violations in a single test definition.
        with pytest.raises(ValidationError):
            Citation(text="text", source="doc.pdf", page=1, score=score)

    def test_page_zero_raises(self) -> None:
        # The document model uses 1-indexed pages to match PDF conventions.
        # Page 0 would produce misleading source references in the UI.
        with pytest.raises(ValidationError):
            Citation(text="text", source="doc.pdf", page=0, score=0.9)


# ---------------------------------------------------------------------------
# AgentAnswer
# ---------------------------------------------------------------------------


class TestAgentAnswer:
    def test_valid_answer(self, agent_answer: AgentAnswer) -> None:
        # Confirms that a fully populated answer retains confidence and source.
        assert agent_answer.confidence == AnswerConfidence.HIGH
        assert agent_answer.answer_source == AnswerSource.DIRECT

    def test_defaults_when_no_answer_found(
        self, extraction_question: UserQuestion
    ) -> None:
        # When the agent finds no relevant content, only document and question
        # are required. All answer fields default to None and confidence to
        # NOT_FOUND, so the UI can render a "not found" state without extra logic.
        answer = AgentAnswer(
            question=extraction_question,
            document="sentencia_001.pdf",
        )
        assert answer.confidence == AnswerConfidence.NOT_FOUND
        assert answer.answer is None
        assert answer.citation is None
        assert answer.answer_source is None


# ---------------------------------------------------------------------------
# DocumentAnswers
# ---------------------------------------------------------------------------


class TestDocumentAnswers:
    def test_empty_answers_is_valid(self) -> None:
        # An empty answers list is intentionally allowed: it represents a document
        # where the agent found no relevant content for any question,
        # which is a legitimate outcome rather than a configuration error.
        doc = DocumentAnswers(document="sentencia_001.pdf", answers=[])
        assert doc.answers == []
