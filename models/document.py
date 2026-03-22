"""
document.py

Pydantic models for document-related data structures.

Models:
    Chunk: a single text fragment with its metadata (page, position, source file).
    DocumentMetadata: high-level information about a processed PDF.
    DocumentResult: full extraction result for a single PDF.
"""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single text fragment extracted from a PDF.

    Attributes:
        text: The extracted text content of the fragment.
        source: Original PDF filename, used to trace answers back to their document.
        page: Page number where the chunk starts (1-indexed).
        chunk_index: Position of this chunk within the document (0-indexed).
        chunk_id: Unique key for storage, format: "{source}_{page}_{chunk_index}".
        headings: Breadcrumb of section headers this chunk falls under, as detected
            by Docling's document hierarchy. Empty list if the chunk has no parent
            section. Example: ["FUNDAMENTOS DE DERECHO", "PRIMERO.-"].
            Stored as ChromaDB metadata to enable section-aware retrieval.
    """

    text: str = Field(min_length=1)
    source: str
    page: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    chunk_id: str
    headings: list[str] = []


class DocumentMetadata(BaseModel):
    """High-level information about a processed PDF.

    Attributes:
        filename: Original PDF filename.
        total_pages: Total number of pages in the document.
        total_chunks: Number of chunks generated during extraction.
        ocr_applied: True if Docling used OCR during extraction.
        is_scanned: True if the user indicated this PDF is a scanned
            document. Determines whether OCR is applied at extraction time.
    """

    filename: str
    total_pages: int = Field(ge=1)
    total_chunks: int = Field(ge=0)
    ocr_applied: bool = False
    is_scanned: bool = False


class DocumentResult(BaseModel):
    """Full extraction result for a single PDF.

    Produced by extractor.py and consumed by embedder.py. Once the chunks
    are embedded and stored in ChromaDB, this object is discarded.

    Attributes:
        metadata: High-level document information.
        chunks: All text fragments extracted from the document.
    """

    metadata: DocumentMetadata
    chunks: list[Chunk]
