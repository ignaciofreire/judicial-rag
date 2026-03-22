"""
query.py

Pydantic models for query-related data structures.

Models:
    QuestionType: enum defining the four supported question strategies.
    AnswerConfidence: enum indicating how confident the agent is in its answer.
    AnswerSource: enum indicating whether the answer was extracted directly
        from the text or inferred through reasoning.
    Category: a single option in a classification question.
    UserQuestion: a question or variable defined by the user, with its type,
        rules and categories if applicable.
    QuestionSchema: the full set of questions defined by a user,
        persisted and reused across sessions.
    Citation: the source fragment the agent used to build an answer.
    AgentAnswer: the full agent response for one question on one document.
    DocumentAnswers: all answers for a single document.
"""

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class QuestionType(str, Enum):
    """Strategy the agent uses to answer a question.

    Each type maps to a dedicated skill in pipeline/skills/:

    Attributes:
        EXTRACTION: copy a value directly from the text (e.g. ID, date, name).
            Skill: skills/extraction.py
        CALCULATION: derive a value by operating on extracted data
            (e.g. months between two dates).
            Skill: skills/calculation.py
        CLASSIFICATION: assign content to a category from a closed list
            (e.g. sex of victim: 1=Woman, 2=Man).
            Skill: skills/classification.py
        EXPLANATION: explain why or how something happened, requiring
            reasoning over the text rather than direct extraction.
            Skill: skills/explanation.py
    """

    EXTRACTION = "extraction"
    CALCULATION = "calculation"
    CLASSIFICATION = "classification"
    EXPLANATION = "explanation"


class AnswerConfidence(str, Enum):
    """How confident the agent is in its answer.

    Assigned by the agent based on the quality of evidence found.
    Used by the UI to flag answers that may need manual review.

    Attributes:
        HIGH: answer is clearly and unambiguously stated in the text.
        MEDIUM: answer is present but requires some interpretation
            or the evidence is partial.
        LOW: answer is inferred from weak or indirect evidence.
        NOT_FOUND: no relevant information was found in the document.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOT_FOUND = "not_found"


class AnswerSource(str, Enum):
    """How the agent obtained the answer.

    Attributes:
        DIRECT: answer was copied verbatim or with minimal transformation
            from a specific text fragment. More reliable.
        INFERRED: answer was derived through reasoning, calculation, or
            combining information from multiple fragments. Should be
            reviewed more carefully.
    """

    DIRECT = "direct"
    INFERRED = "inferred"


class Category(BaseModel):
    """A single option in a classification question.

    The agent reasons using the label and outputs the code,
    keeping human-readable labels decoupled from stored values.

    Attributes:
        code: Value stored in the output (e.g. "1", "2").
        label: Human-readable description shown to the agent and in
            the UI (e.g. "Woman").
    """

    code: str
    label: str


class UserQuestion(BaseModel):
    """A question or variable the agent must answer for each document.

    Attributes:
        label: Short display name shown in the UI (e.g. "Sentencing date").
        question: Full instruction sent to the agent, including extraction
            rules and format requirements.
        question_type: Determines which skill is invoked in the pipeline.
        categories: Required when question_type is CLASSIFICATION. Must
            contain at least two options.
        output_format: Expected output format hint (e.g. "DD/MM/YYYY",
            "integer", "float").
        notes: Additional rules or edge cases for the agent
            (e.g. rounding rules for latency calculations).
    """

    label: str = Field(min_length=1)
    question: str = Field(min_length=1)
    question_type: QuestionType
    categories: list[Category] | None = None
    output_format: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_categories(self) -> "UserQuestion":
        # Classification requires at least two categories to be meaningful
        if self.question_type == QuestionType.CLASSIFICATION:
            if not self.categories or len(self.categories) < 2:
                raise ValueError(
                    f"Question '{self.label}' is of type CLASSIFICATION "
                    "and must define at least two categories."
                )
        # Other types must not define categories to avoid misconfiguration
        if self.question_type != QuestionType.CLASSIFICATION and self.categories:
            raise ValueError(
                f"Question '{self.label}' is of type {self.question_type.value} "
                "and should not define categories. "
                "Categories are only valid for CLASSIFICATION questions."
            )
        return self


class QuestionSchema(BaseModel):
    """Full set of questions defined by a user, persisted across sessions.

    Serialised to JSON for storage and reuse. Designed to support
    import from external files (Excel, JSON) in the future.

    Attributes:
        name: Descriptive name.
        description: Optional longer description of the schema purpose.
        questions: Ordered list of questions applied to each document.
    """

    name: str = Field(min_length=1)
    description: str | None = None
    questions: list[UserQuestion] = Field(min_length=1)


class Citation(BaseModel):
    """Source fragment the agent used to build its answer.

    Attributes:
        text: Exact text fragment retrieved from the document.
        source: PDF filename where the fragment was found.
        page: Page number where the fragment was found (1-indexed).
        score: Semantic similarity score between the question and this
            fragment (0-1). Higher means more relevant.
    """

    text: str
    source: str
    page: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)


class AgentAnswer(BaseModel):
    """Agent response for one question applied to one document.

    Attributes:
        question: The original question this answer responds to.
        answer: The agent answer text. None if no relevant content was found.
        citation: Source fragment used to generate the answer.
            None if no relevant content was found.
        document: Filename of the document this answer refers to.
        confidence: How confident the agent is in its answer.
            Helps the user decide whether to review the source manually.
        answer_source: Whether the answer was extracted directly or derived
            through reasoning. None if no answer was found.
    """

    question: UserQuestion
    answer: str | None = None
    citation: Citation | None = None
    document: str
    confidence: AnswerConfidence = AnswerConfidence.NOT_FOUND
    answer_source: AnswerSource | None = None


class DocumentAnswers(BaseModel):
    """All answers for a single document.

    Groups AgentAnswer objects by document so the UI can render
    results per PDF without filtering a flat list.

    Attributes:
        document: PDF filename.
        answers: One AgentAnswer per question in the schema.
    """

    document: str
    answers: list[AgentAnswer]
