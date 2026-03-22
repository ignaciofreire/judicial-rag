"""
extractor.py

PDF content extraction using Docling and HybridChunker.
Handles both native digital PDFs and scanned documents (OCR).
Uses Docling's native document hierarchy to produce semantically
coherent chunks with section headings as metadata, avoiding the
need for regex-based section detection.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from config.settings import settings
from models.document import Chunk, DocumentMetadata, DocumentResult

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
# All expensive resources are initialised once at import time and shared
# across every extraction call. Instantiating these inside _extract_sync
# would reload Docling's layout models on every PDF call.

_executor = ThreadPoolExecutor(max_workers=settings.max_parallel_pdfs)

# Tokenizer used only to measure chunk sizes — not for generating embeddings.
_tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),
    max_tokens=512,
)


def _make_pipeline_options(*, ocr: bool) -> PdfPipelineOptions:
    """Build Docling pipeline options with only the features we need.

    Table structure analysis, page image generation and picture image
    generation are disabled unconditionally — judicial PDFs rarely contain
    tables worth parsing and we never use the rendered images. OCR is
    controlled by the caller based on whether the user indicated the PDF
    is a scanned document.

    Args:
        ocr: Whether to enable OCR for scanned PDFs.

    Returns:
        Configured PdfPipelineOptions instance.
    """
    options = PdfPipelineOptions()
    options.do_ocr = ocr
    options.do_table_structure = False
    options.generate_page_images = False
    options.generate_picture_images = False
    return options


def _make_converter(*, ocr: bool) -> DocumentConverter:
    """Build a DocumentConverter with minimal pipeline options.

    Args:
        ocr: Whether to enable OCR for scanned PDFs.

    Returns:
        Configured DocumentConverter instance.
    """
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=_make_pipeline_options(ocr=ocr)
            )
        }
    )


# Two converters cover all cases: digital PDFs (fast) and scanned PDFs (OCR).
# The caller selects between them based on the user-provided is_scanned flag.
_digital_converter = _make_converter(ocr=False)
_ocr_converter = _make_converter(ocr=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract(pdf_path: Path, is_scanned: bool = False) -> DocumentResult:
    """Extract text and metadata from a PDF asynchronously.

    Offloads Docling's synchronous conversion to a thread pool so multiple
    PDFs can be processed concurrently without blocking the event loop.

    Args:
        pdf_path: Absolute path to the PDF file.
        is_scanned: Whether the user indicated this PDF is a scanned
            document. When True, Docling uses OCR to extract text from
            page images instead of the native text layer.

    Returns:
        DocumentResult with all chunks and document metadata.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _extract_sync, pdf_path, is_scanned)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _extract_sync(pdf_path: Path, is_scanned: bool) -> DocumentResult:
    """Run Docling extraction and chunking synchronously.

    Separated from the async wrapper so it can run in a thread
    without carrying async context.

    Args:
        pdf_path: Absolute path to the PDF file.
        is_scanned: Whether OCR should be applied during extraction.

    Returns:
        DocumentResult with all chunks and document metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
        RuntimeError: If Docling fails to convert the document.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    converter = _ocr_converter if is_scanned else _digital_converter

    try:
        result = converter.convert(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Docling failed to convert {pdf_path.name}: {e}") from e

    chunker = HybridChunker(
        tokenizer=_tokenizer,
        # Merge consecutive undersized chunks that share the same headings
        # to avoid fragmenting short paragraphs across chunk boundaries
        merge_peers=True,
    )

    chunks = _build_chunks(chunker, result.document, pdf_path.name)

    return DocumentResult(
        metadata=DocumentMetadata(
            filename=pdf_path.name,
            total_pages=max((c.page for c in chunks), default=1),
            total_chunks=len(chunks),
            ocr_applied=is_scanned,
            is_scanned=is_scanned,
        ),
        chunks=chunks,
    )


def _build_chunks(
    chunker: HybridChunker,
    doc: object,
    filename: str,
) -> list[Chunk]:
    """Convert HybridChunker output into Chunk model instances.

    Uses chunker.contextualize() rather than the raw chunk text so that
    the heading breadcrumb is prepended to every chunk. This gives the
    embedding model the section context alongside the chunk content, which
    improves retrieval precision for section-specific questions like
    "what was the ruling?" or "what articles were cited?".

    Skips empty chunks that can appear as layout artefacts in Docling's
    output for pages that contain only images or decorative elements.

    Args:
        chunker: Configured HybridChunker instance.
        doc: DoclingDocument returned by DocumentConverter.
        filename: Source PDF filename used for chunk_id and source fields.

    Returns:
        Ordered list of Chunk instances, one per non-empty Docling chunk.
    """
    chunks: list[Chunk] = []

    for chunk_index, docling_chunk in enumerate(chunker.chunk(dl_doc=doc)):
        text = chunker.contextualize(chunk=docling_chunk)

        if not text.strip():
            continue

        page = _get_page(docling_chunk)
        chunks.append(
            Chunk(
                text=text,
                source=filename,
                page=page,
                chunk_index=chunk_index,
                chunk_id=f"{filename}_{page}_{chunk_index}",
                headings=list(docling_chunk.meta.headings or []),
            )
        )

    return chunks


def _get_page(docling_chunk: object) -> int:
    """Extract the page number from a Docling chunk's provenance metadata.

    Navigates the chain: chunk.meta.doc_items[0].prov[0].page_no.
    Each level can be absent depending on the Docling version and the
    type of content element, so every access is guarded by the try/except.

    Args:
        docling_chunk: A chunk produced by HybridChunker.

    Returns:
        1-indexed page number, or 1 as a safe fallback.
    """
    try:
        doc_items = docling_chunk.meta.doc_items
        if doc_items and doc_items[0].prov:
            return doc_items[0].prov[0].page_no
    except Exception:
        pass
    return 1
