"""
test_embedder.py

Unit tests for pipeline/embedder.py.
Tests embedding generation, ChromaDB collection management and metadata
serialisation without calling the real HF Inference API — all external
calls are mocked so tests run fast and offline.

Run all:     uv run pytest tests/test_embedder.py -v
Run a class: uv run pytest tests/test_embedder.py::TestCollectionManagement -v
"""

from unittest.mock import patch

import numpy as np
import pytest

from models.document import Chunk, DocumentMetadata, DocumentResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vector() -> list[float]:
    """A deterministic 1024-dimensional unit vector for testing."""
    rng = np.random.default_rng(seed=42)
    vector = rng.random(1024).astype(np.float32)
    return vector.tolist()


@pytest.fixture
def chunk() -> Chunk:
    """Minimal valid chunk representing a judicial text fragment."""
    return Chunk(
        text="El tribunal absolvió al acusado por falta de pruebas.",
        source="sentencia_001.pdf",
        page=1,
        chunk_index=0,
        chunk_id="sentencia_001.pdf_1_0",
        headings=["FALLO"],
    )


@pytest.fixture
def chunk_no_headings() -> Chunk:
    """Chunk with no section headings — tests empty headings serialisation."""
    return Chunk(
        text="Texto del encabezado de la sentencia.",
        source="sentencia_001.pdf",
        page=1,
        chunk_index=1,
        chunk_id="sentencia_001.pdf_1_1",
    )


@pytest.fixture
def document(chunk: Chunk, chunk_no_headings: Chunk) -> DocumentResult:
    """DocumentResult with two chunks for embedding tests."""
    return DocumentResult(
        metadata=DocumentMetadata(
            filename="sentencia_001.pdf",
            total_pages=3,
            total_chunks=2,
        ),
        chunks=[chunk, chunk_no_headings],
    )


@pytest.fixture
def empty_document() -> DocumentResult:
    """DocumentResult with no chunks — tests the early-exit path."""
    return DocumentResult(
        metadata=DocumentMetadata(
            filename="empty.pdf",
            total_pages=1,
            total_chunks=0,
        ),
        chunks=[],
    )


@pytest.fixture
def mock_inference(mock_vector: list[float]):
    """Patches the HF InferenceClient so no real API calls are made."""
    with patch("pipeline.embedder._inference") as mock:
        mock.feature_extraction.return_value = np.array(mock_vector)
        yield mock


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


class TestCollectionManagement:
    def test_get_collection_creates_new(self) -> None:
        # A collection is created on first access and returns a valid object
        from pipeline.embedder import get_collection

        collection = get_collection("session-new-abc")
        assert collection is not None
        assert collection.count() == 0

    def test_get_collection_is_idempotent(self) -> None:
        # Calling get_collection twice with the same session_id returns the
        # same collection without raising or creating a duplicate
        from pipeline.embedder import get_collection

        c1 = get_collection("session-idem-xyz")
        c2 = get_collection("session-idem-xyz")
        assert c1.name == c2.name

    def test_delete_collection_removes_it(self) -> None:
        from pipeline.embedder import delete_collection, get_collection

        session_id = "session-delete-test"
        get_collection(session_id)
        delete_collection(session_id)
        # After deletion a new empty collection is created on next access
        collection = get_collection(session_id)
        assert collection.count() == 0

    def test_delete_nonexistent_collection_does_not_raise(self) -> None:
        # Deleting a collection that never existed must not raise — a session
        # may end before any PDFs are processed
        from pipeline.embedder import delete_collection

        delete_collection("session-never-existed-99999")


# ---------------------------------------------------------------------------
# Metadata serialisation
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    def test_headings_serialised_as_pipe_separated_string(self, chunk: Chunk) -> None:
        from pipeline.embedder import _build_metadata

        meta = _build_metadata(chunk)
        assert meta["headings"] == "FALLO"

    def test_multiple_headings_joined_with_pipe(self) -> None:
        from pipeline.embedder import _build_metadata

        chunk = Chunk(
            text="Texto.",
            source="doc.pdf",
            page=2,
            chunk_index=0,
            chunk_id="doc.pdf_2_0",
            headings=["FUNDAMENTOS DE DERECHO", "PRIMERO.-"],
        )
        meta = _build_metadata(chunk)
        assert meta["headings"] == "FUNDAMENTOS DE DERECHO|PRIMERO.-"

    def test_empty_headings_serialised_as_empty_string(
        self, chunk_no_headings: Chunk
    ) -> None:
        # Empty string is the sentinel value — deserialise with guard:
        # headings = h.split("|") if h else []
        from pipeline.embedder import _build_metadata

        meta = _build_metadata(chunk_no_headings)
        assert meta["headings"] == ""

    def test_metadata_fields_present(self, chunk: Chunk) -> None:
        from pipeline.embedder import _build_metadata

        meta = _build_metadata(chunk)
        assert meta["source"] == chunk.source
        assert meta["page"] == chunk.page
        assert meta["chunk_index"] == chunk.chunk_index


# ---------------------------------------------------------------------------
# Vector generation
# ---------------------------------------------------------------------------


class TestToVector:
    def test_returns_list_of_floats(
        self, mock_inference, mock_vector: list[float]
    ) -> None:
        from pipeline.embedder import _to_vector

        result = _to_vector("passage: some text")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_vector_has_correct_dimension(
        self, mock_inference, mock_vector: list[float]
    ) -> None:
        from pipeline.embedder import _to_vector

        result = _to_vector("passage: some text")
        assert len(result) == 1024

    def test_passage_prefix_is_passed_to_api(self, mock_inference) -> None:
        from pipeline.embedder import _to_vector

        _to_vector("passage: texto judicial")
        mock_inference.feature_extraction.assert_called_once()
        call_args = mock_inference.feature_extraction.call_args
        assert call_args[0][0].startswith("passage:")

    def test_query_prefix_is_passed_to_api(self, mock_inference) -> None:
        from pipeline.embedder import embed_query

        embed_query("¿Cuál es el fallo?")
        mock_inference.feature_extraction.assert_called_once()
        call_args = mock_inference.feature_extraction.call_args
        assert call_args[0][0].startswith("query:")


# ---------------------------------------------------------------------------
# Document embedding
# ---------------------------------------------------------------------------


class TestEmbedDocument:
    @pytest.mark.asyncio
    async def test_embeds_all_chunks(
        self, mock_inference, document: DocumentResult
    ) -> None:
        from pipeline.embedder import embed_document, get_collection

        session_id = "session-embed-all"
        await embed_document(document, session_id)
        collection = get_collection(session_id)
        assert collection.count() == len(document.chunks)

    @pytest.mark.asyncio
    async def test_upsert_is_idempotent(
        self, mock_inference, document: DocumentResult
    ) -> None:
        # Embedding the same document twice must not create duplicates —
        # upsert deduplicates by chunk_id
        from pipeline.embedder import embed_document, get_collection

        session_id = "session-upsert-idem"
        await embed_document(document, session_id)
        await embed_document(document, session_id)
        collection = get_collection(session_id)
        assert collection.count() == len(document.chunks)

    @pytest.mark.asyncio
    async def test_empty_document_does_not_raise(
        self, mock_inference, empty_document: DocumentResult
    ) -> None:
        # A document with no chunks is a valid outcome — the extractor
        # may produce zero chunks for a blank or unreadable PDF
        from pipeline.embedder import embed_document

        await embed_document(empty_document, "session-empty")

    @pytest.mark.asyncio
    async def test_sessions_are_isolated(
        self, mock_inference, document: DocumentResult
    ) -> None:
        # Chunks embedded under session A must not appear in session B
        from pipeline.embedder import embed_document, get_collection

        await embed_document(document, "session-a")
        collection_b = get_collection("session-b")
        assert collection_b.count() == 0

    @pytest.mark.asyncio
    async def test_chunk_text_stored_without_prefix(
        self, mock_inference, document: DocumentResult
    ) -> None:
        # The raw chunk text (without "passage: " prefix) must be stored
        # in ChromaDB so it can be returned as a citation to the user
        from pipeline.embedder import embed_document, get_collection

        session_id = "session-text-check"
        await embed_document(document, session_id)
        collection = get_collection(session_id)
        results = collection.get(ids=[document.chunks[0].chunk_id])
        assert results["documents"][0] == document.chunks[0].text
