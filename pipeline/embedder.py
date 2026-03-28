"""
embedder.py

Embedding generation and vector store management.
Converts document chunks into vector representations using the
HF Inference API and stores them in ChromaDB with their metadata.
Each user session gets an isolated ChromaDB collection so that
documents from different users never mix.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import chromadb
import numpy as np
from huggingface_hub import InferenceClient

from config.settings import settings
from models.document import Chunk, DocumentResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
# Expensive resources are initialised once at import time and shared across
# all calls. Both ChromaDB and InferenceClient are thread-safe.

_chroma = chromadb.Client()
logger.debug("ChromaDB in-memory client initialised")

_inference = InferenceClient(
    provider="hf-inference",
    api_key=settings.huggingface_api_key,
)
logger.debug("HF InferenceClient initialised: %s", settings.embedding_model)

# I/O-bound embedding API calls run in a thread pool to avoid blocking
# the event loop while waiting for the HF Inference API to respond.
_executor = ThreadPoolExecutor(max_workers=settings.max_parallel_pdfs)

# multilingual-e5-large uses asymmetric prefixes for retrieval:
# "passage: " for documents at index time, "query: " for questions at
# search time. This distinction is central to the model's retrieval quality.
_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "

# Maximum number of chunks sent to the HF API in a single batch.
# Keeps individual requests small to stay within rate limits and timeouts.
_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def embed_document(document: DocumentResult, session_id: str) -> None:
    """Embed all chunks of a document and store them in ChromaDB.

    Each session gets its own isolated collection identified by session_id.
    Re-processing the same PDF is safe — upsert deduplicates by chunk_id.

    Args:
        document: Extraction result from extractor.py containing all chunks.
        session_id: Unique identifier for the user session. Used as the
            ChromaDB collection name to isolate data between users.
    """
    logger.info(
        "Embedding started: %s — %d chunks (session=%s)",
        document.metadata.filename,
        document.metadata.total_chunks,
        session_id,
    )
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _embed_sync, document, session_id)
    logger.info(
        "Embedding complete: %s (session=%s)",
        document.metadata.filename,
        session_id,
    )


def embed_query(text: str) -> list[float]:
    """Embed a user question for similarity search against stored chunks.

    Uses the "query: " prefix required by multilingual-e5-large for
    retrieval queries, which produces embeddings in the same vector space
    as the "passage: " embeddings stored during indexing.

    Args:
        text: The user question or search query.

    Returns:
        1024-dimensional float vector ready for ChromaDB similarity search.
    """
    logger.debug("Embedding query: %r", text[:80])
    return _to_vector(f"{_QUERY_PREFIX}{text}")


def get_collection(session_id: str) -> chromadb.Collection:
    """Retrieve or create the ChromaDB collection for a session.

    Args:
        session_id: Unique identifier for the user session.

    Returns:
        ChromaDB collection scoped to this session.
    """
    name = _collection_name(session_id)
    collection = _chroma.get_or_create_collection(
        name=name,
        # Cosine similarity is standard for sentence embedding retrieval.
        # It measures semantic angle between vectors independently of their
        # magnitude, which is the correct metric for this use case.
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug("Collection ready: %s (%d items)", name, collection.count())
    return collection


def delete_collection(session_id: str) -> None:
    """Delete the ChromaDB collection for a session.

    Called by storage/cleanup.py when a session ends or times out.
    Silently ignores missing collections — a session may have ended before
    any PDFs were processed.

    Args:
        session_id: Unique identifier for the user session.
    """
    name = _collection_name(session_id)
    try:
        _chroma.delete_collection(name)
        logger.info("Collection deleted: %s", name)
    except Exception:
        logger.debug("Collection not found for deletion (safe to ignore): %s", name)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _collection_name(session_id: str) -> str:
    """Build a ChromaDB-compatible collection name from a session ID.

    ChromaDB requires names to be 3-63 characters, start and end with an
    alphanumeric character, and contain only alphanumerics and hyphens.
    The "s-" prefix guarantees the name starts with a letter even when
    session_id begins with a digit.

    Args:
        session_id: Raw session identifier from Streamlit.

    Returns:
        Sanitised collection name safe for ChromaDB.
    """
    return f"s-{session_id[:50]}"


def _embed_sync(document: DocumentResult, session_id: str) -> None:
    """Embed all chunks synchronously and upsert them into ChromaDB.

    Processes chunks in fixed-size batches to respect API rate limits.
    Uses upsert so that re-processing the same PDF replaces existing
    vectors rather than creating duplicates.

    Args:
        document: Extraction result containing the chunks to embed.
        session_id: User session identifier for collection isolation.
    """
    chunks = document.chunks
    if not chunks:
        logger.warning("No chunks to embed for %s", document.metadata.filename)
        return

    collection = get_collection(session_id)
    total = len(chunks)

    for start in range(0, total, _BATCH_SIZE):
        batch = chunks[start : start + _BATCH_SIZE]
        logger.debug(
            "Embedding batch %d-%d / %d for %s",
            start + 1,
            start + len(batch),
            total,
            document.metadata.filename,
        )
        _upsert_batch(collection, batch)


def _upsert_batch(collection: chromadb.Collection, chunks: list[Chunk]) -> None:
    """Generate embeddings for a batch and upsert them into ChromaDB.

    Args:
        collection: Target ChromaDB collection.
        chunks: Batch of chunks to embed and store.
    """
    texts = [f"{_PASSAGE_PREFIX}{chunk.text}" for chunk in chunks]
    embeddings = [_to_vector(text) for text in texts]

    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=embeddings,
        # Store the raw text (without prefix) so the agent can return it
        # as the citation fragment shown to the user in the UI
        documents=[chunk.text for chunk in chunks],
        metadatas=[_build_metadata(chunk) for chunk in chunks],
    )
    logger.debug("Upserted %d chunks", len(chunks))


def _to_vector(text: str) -> list[float]:
    """Embed a single text string via the HF Inference API.

    multilingual-e5-large applies mean pooling internally and returns a
    flat (1024,) array, so no additional pooling is needed here.

    Args:
        text: Text to embed, including the appropriate prefix.

    Returns:
        1024-dimensional float vector as a Python list.
    """
    raw = _inference.feature_extraction(text, model=settings.embedding_model)
    return np.array(raw).tolist()


def _build_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Build the ChromaDB metadata dict for a chunk.

    ChromaDB metadata values must be strings, integers, floats or booleans.
    The headings list is serialised as a pipe-separated string because
    ChromaDB does not support list values. Deserialise with
    `headings.split("|")` at retrieval time, guarding for the empty string.

    Args:
        chunk: Source chunk to extract metadata from.

    Returns:
        Flat metadata dict compatible with ChromaDB's type constraints.
    """
    return {
        "source": chunk.source,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
        "headings": "|".join(chunk.headings) if chunk.headings else "",
    }
