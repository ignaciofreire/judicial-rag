"""
vector_store.py

Query interface for the ChromaDB vector store.
Provides semantic search over embedded document chunks, scoped both
to a user session and to a specific source document. The embedder
writes to ChromaDB; this module only reads from it.
"""

import logging

from pipeline.embedder import embed_query, get_collection

logger = logging.getLogger(__name__)

# Number of candidate chunks retrieved per query. The agent re-ranks and
# filters these before passing them to the LLM, so a higher value improves
# recall at the cost of a slightly larger context window sent to the model.
_DEFAULT_N_RESULTS = 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search(
    question: str,
    session_id: str,
    source: str,
    n_results: int = _DEFAULT_N_RESULTS,
) -> list[dict]:
    """Search for chunks relevant to a question within a single document.

    Embeds the question and queries ChromaDB with a `where` filter on the
    source filename so results never mix across PDFs. Returns chunks ordered
    by descending cosine similarity (ascending distance).

    Args:
        question: The user question or variable to extract.
        session_id: Selects the correct ChromaDB collection for this user.
        source: PDF filename to restrict the search to. Must match the
            `source` metadata field stored at index time.
        n_results: Maximum number of chunks to return.

    Returns:
        List of result dicts ordered by relevance, each containing:
            text        — raw chunk text returned as citation to the user.
            source      — PDF filename.
            page        — page number (1-indexed).
            headings    — section breadcrumb as a list of strings.
            chunk_index — position of the chunk within the document.
            distance    — cosine distance; lower means more similar.
    """
    logger.debug(
        "Search: question=%r source=%s n=%d session=%s",
        question[:80],
        source,
        n_results,
        session_id,
    )

    query_vector = embed_query(question)
    collection = get_collection(session_id)

    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        # Filtering by source before the similarity search is more efficient
        # than post-filtering and guarantees results stay within one PDF
        where={"source": source},
        include=["documents", "metadatas", "distances"],
    )

    results = _parse_query_results(raw)

    logger.debug(
        "Search returned %d results (top distance=%.4f)",
        len(results),
        results[0]["distance"] if results else float("nan"),
    )
    return results


def list_sources(session_id: str) -> list[str]:
    """Return the unique PDF filenames indexed in a session.

    Used by the orchestrator to iterate over all documents and run the
    agent's questions against each one independently.

    Args:
        session_id: User session identifier.

    Returns:
        Sorted list of unique source filenames. Empty if no documents
        have been indexed yet.
    """
    collection = get_collection(session_id)

    if collection.count() == 0:
        logger.debug("Collection empty for session=%s", session_id)
        return []

    # Fetch only metadata — skipping embeddings and documents keeps this fast
    items = collection.get(include=["metadatas"])
    sources = sorted(
        {meta["source"] for meta in items["metadatas"] if "source" in meta}
    )

    logger.debug(
        "%d unique source(s) in session=%s: %s",
        len(sources),
        session_id,
        sources,
    )
    return sources


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _parse_query_results(raw: dict) -> list[dict]:
    """Parse ChromaDB query output into a flat list of result dicts.

    ChromaDB returns parallel lists (documents, metadatas, distances) wrapped
    in an outer batch dimension. This function flattens that dimension and
    deserialises the pipe-separated headings string back into a list.

    Args:
        raw: Raw dict returned by collection.query().

    Returns:
        Flat list of result dicts ordered by ascending distance.
    """
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    return [
        {
            "text": text,
            "source": meta.get("source", ""),
            "page": meta.get("page", 1),
            # Deserialise headings from the pipe-separated sentinel stored at
            # index time — empty string means no section headings were detected
            "headings": meta["headings"].split("|") if meta.get("headings") else [],
            "chunk_index": meta.get("chunk_index", 0),
            "distance": distance,
        }
        for text, meta, distance in zip(documents, metadatas, distances, strict=False)
    ]
