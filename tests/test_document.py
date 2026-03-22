"""
test_document.py

Unit tests for models/document.py.

Run all:     uv run pytest tests/test_document.py -v
Run a class: uv run pytest tests/test_document.py::TestChunk -v
"""

import pytest
from pydantic import ValidationError

from models.document import Chunk, DocumentMetadata, DocumentResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunk() -> Chunk:
    """Minimal valid chunk without section headings."""
    return Chunk(
        text="En Barcelona, a treinta de enero de dos mil veinticuatro.",
        source="sentencia_001.pdf",
        page=1,
        chunk_index=0,
        chunk_id="sentencia_001.pdf_1_0",
    )


@pytest.fixture
def chunk_with_headings() -> Chunk:
    """Valid chunk with a heading breadcrumb from Docling's document hierarchy."""
    return Chunk(
        text="El día 20 de enero del año en curso se celebró el juicio oral.",
        source="sentencia_001.pdf",
        page=2,
        chunk_index=1,
        chunk_id="sentencia_001.pdf_2_1",
        headings=["FUNDAMENTOS DE DERECHO", "PRIMERO.-"],
    )


@pytest.fixture
def metadata() -> DocumentMetadata:
    """Valid document metadata for a native digital PDF."""
    return DocumentMetadata(
        filename="sentencia_001.pdf",
        total_pages=10,
        total_chunks=25,
    )


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_headings_default_to_empty_list(self, chunk: Chunk) -> None:
        # Chunks with no parent section are valid — the header section
        # of a ruling often has no enclosing heading in Docling's hierarchy
        assert chunk.headings == []

    def test_headings_are_retained(self, chunk_with_headings: Chunk) -> None:
        # Confirms the full heading breadcrumb is preserved as provided
        # by HybridChunker's document hierarchy metadata
        assert chunk_with_headings.headings == [
            "FUNDAMENTOS DE DERECHO",
            "PRIMERO.-",
        ]

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                text="",
                source="sentencia_001.pdf",
                page=1,
                chunk_index=0,
                chunk_id="sentencia_001.pdf_1_0",
            )

    def test_page_zero_raises(self) -> None:
        # Pages are 1-indexed to match PDF conventions
        with pytest.raises(ValidationError):
            Chunk(
                text="Some text.",
                source="sentencia_001.pdf",
                page=0,
                chunk_index=0,
                chunk_id="sentencia_001.pdf_1_0",
            )

    def test_negative_chunk_index_raises(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                text="Some text.",
                source="sentencia_001.pdf",
                page=1,
                chunk_index=-1,
                chunk_id="sentencia_001.pdf_1_0",
            )


# ---------------------------------------------------------------------------
# DocumentMetadata
# ---------------------------------------------------------------------------


class TestDocumentMetadata:
    def test_ocr_defaults_to_false(self, metadata: DocumentMetadata) -> None:
        # Most judicial PDFs are native digital — OCR should only be True
        # when Docling explicitly detected a scanned document
        assert metadata.ocr_applied is False

    def test_zero_total_pages_raises(self) -> None:
        with pytest.raises(ValidationError):
            DocumentMetadata(filename="doc.pdf", total_pages=0, total_chunks=0)


# ---------------------------------------------------------------------------
# DocumentResult
# ---------------------------------------------------------------------------


class TestDocumentResult:
    def test_valid_result(self, metadata: DocumentMetadata, chunk: Chunk) -> None:
        result = DocumentResult(metadata=metadata, chunks=[chunk])
        assert len(result.chunks) == 1

    def test_empty_chunks_is_valid(self, metadata: DocumentMetadata) -> None:
        # Empty chunk list is valid — it means the PDF had no extractable text
        result = DocumentResult(metadata=metadata, chunks=[])
        assert result.chunks == []
