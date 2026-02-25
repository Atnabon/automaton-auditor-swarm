"""
PDF ingestion and chunked querying tools for the Automaton Auditor.

Uses PyMuPDF (fitz) for fast extraction of text, images, and structural
elements.  Implements a RAG-lite approach: pages are chunked, embedded
descriptions are built, and keyword / semantic searches can filter relevant
passages for the DocAnalyst detective.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore[assignment]
    logger.warning("PyMuPDF not installed — PDF tools will be unavailable.")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PageChunk:
    """A chunk of text extracted from a single PDF page."""

    page_number: int
    text: str
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class ExtractedImage:
    """Metadata for an image extracted from the PDF."""

    page_number: int
    image_index: int
    width: int
    height: int
    image_bytes: bytes = field(repr=False)
    format: str = "png"


@dataclass
class PDFDocument:
    """Parsed PDF document with text chunks and extracted images."""

    path: str
    total_pages: int
    chunks: List[PageChunk] = field(default_factory=list)
    images: List[ExtractedImage] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PDF Ingestion
# ---------------------------------------------------------------------------


def ingest_pdf(pdf_path: str, extract_images: bool = True) -> Optional[PDFDocument]:
    """Load a PDF and extract per-page text chunks and optionally images.

    Args:
        pdf_path: Path to the PDF file.
        extract_images: Whether to extract embedded images.

    Returns:
        PDFDocument or None if an error occurs.
    """
    if fitz is None:
        logger.error("PyMuPDF is required for PDF ingestion.  Install with: pip install pymupdf")
        return None

    path = Path(pdf_path)
    if not path.is_file():
        logger.error("PDF not found: %s", pdf_path)
        return None

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", pdf_path, exc)
        return None

    pdf_doc = PDFDocument(
        path=str(path),
        total_pages=len(doc),
        metadata=dict(doc.metadata) if doc.metadata else {},
    )

    for page_num in range(len(doc)):
        page = doc[page_num]

        # --- Text extraction ---
        text = page.get_text("text")
        if text.strip():
            pdf_doc.chunks.append(PageChunk(page_number=page_num + 1, text=text.strip()))

        # --- Image extraction ---
        if extract_images:
            for img_index, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    if base_image:
                        pdf_doc.images.append(
                            ExtractedImage(
                                page_number=page_num + 1,
                                image_index=img_index,
                                width=base_image.get("width", 0),
                                height=base_image.get("height", 0),
                                image_bytes=base_image["image"],
                                format=base_image.get("ext", "png"),
                            )
                        )
                except Exception as exc:
                    logger.debug("Could not extract image xref=%d: %s", xref, exc)

    doc.close()
    logger.info(
        "Ingested PDF %s: %d pages, %d chunks, %d images",
        pdf_path,
        pdf_doc.total_pages,
        len(pdf_doc.chunks),
        len(pdf_doc.images),
    )
    return pdf_doc


# ---------------------------------------------------------------------------
# Chunked Querying (RAG-Lite)
# ---------------------------------------------------------------------------


def keyword_search(
    pdf_doc: PDFDocument,
    keywords: List[str],
    case_sensitive: bool = False,
) -> List[Dict]:
    """Search for keyword occurrences across all chunks.

    Returns a list of matches with page number, keyword, and surrounding context.
    """
    results: List[Dict] = []
    flags = 0 if case_sensitive else re.IGNORECASE

    for chunk in pdf_doc.chunks:
        for kw in keywords:
            for match in re.finditer(re.escape(kw), chunk.text, flags):
                start = max(0, match.start() - 200)
                end = min(len(chunk.text), match.end() + 200)
                context = chunk.text[start:end].strip()
                results.append(
                    {
                        "keyword": kw,
                        "page": chunk.page_number,
                        "context": context,
                        "match_start": match.start(),
                    }
                )
    return results


def extract_file_paths_from_text(text: str) -> List[str]:
    """Extract file-path-like strings from text.

    Looks for patterns such as `src/tools/ast_parser.py`, `graph.py`, etc.
    """
    # Match typical Python/project paths
    pattern = r"(?:src/|tests/|rubric/|reports/)?[\w/\-]+\.(?:py|json|toml|md|yaml|yml|csv|txt)"
    return sorted(set(re.findall(pattern, text)))


def extract_mentioned_paths(pdf_doc: PDFDocument) -> List[str]:
    """Scan all chunks and collect every file path mentioned in the PDF."""
    all_paths: List[str] = []
    for chunk in pdf_doc.chunks:
        all_paths.extend(extract_file_paths_from_text(chunk.text))
    return sorted(set(all_paths))


def get_full_text(pdf_doc: PDFDocument) -> str:
    """Concatenate all page chunks into a single string."""
    return "\n\n".join(chunk.text for chunk in pdf_doc.chunks)


def get_page_text(pdf_doc: PDFDocument, page_number: int) -> Optional[str]:
    """Retrieve text for a specific page (1-indexed)."""
    for chunk in pdf_doc.chunks:
        if chunk.page_number == page_number:
            return chunk.text
    return None


def search_context(
    pdf_doc: PDFDocument,
    query: str,
    window_chars: int = 500,
) -> List[Dict]:
    """Simple substring search with a context window around each hit.

    This is the 'RAG-lite' retrieval: instead of vector embeddings, we do
    exact substring matching with generous context windows so the LLM can
    interpret the surrounding text.
    """
    results: List[Dict] = []
    query_lower = query.lower()

    for chunk in pdf_doc.chunks:
        text_lower = chunk.text.lower()
        start = 0
        while True:
            idx = text_lower.find(query_lower, start)
            if idx == -1:
                break
            ctx_start = max(0, idx - window_chars)
            ctx_end = min(len(chunk.text), idx + len(query) + window_chars)
            results.append(
                {
                    "page": chunk.page_number,
                    "context": chunk.text[ctx_start:ctx_end].strip(),
                    "offset": idx,
                }
            )
            start = idx + 1  # advance past this match

    return results


# ---------------------------------------------------------------------------
# Semantic (Paragraph-Level) Chunking — RAG-lite upgrade
# ---------------------------------------------------------------------------


@dataclass
class SectionChunk:
    """A paragraph or logical section extracted from a PDF page.

    Produced by `chunk_by_paragraphs()`, which splits page text on blank-line
    boundaries (two or more newlines) so that each chunk contains a cohesive
    prose block rather than an entire page.
    """

    page_number: int
    section_index: int  # 0-based index within the page
    text: str
    char_count: int = 0
    word_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())


def chunk_by_paragraphs(
    pdf_doc: PDFDocument,
    min_chars: int = 80,
) -> List[SectionChunk]:
    """Split each page chunk into paragraph-level SectionChunks.

    Paragraphs are delineated by two or more consecutive newlines (the typical
    PDF text extraction pattern).  Very short fragments (< `min_chars`) are
    merged into the preceding chunk to avoid noisy single-line artefacts.

    Args:
        pdf_doc: Ingested PDFDocument from `ingest_pdf()`.
        min_chars: Minimum character count to keep a paragraph as its own chunk.

    Returns:
        Ordered list of SectionChunk objects, one per logical paragraph.
    """
    sections: List[SectionChunk] = []
    paragraph_sep = re.compile(r"\n{2,}")

    for page_chunk in pdf_doc.chunks:
        raw_paragraphs = paragraph_sep.split(page_chunk.text)
        idx = 0
        pending = ""

        for para in raw_paragraphs:
            cleaned = para.strip()
            if not cleaned:
                continue
            pending = (pending + " " + cleaned).strip() if pending else cleaned
            if len(pending) >= min_chars:
                sections.append(
                    SectionChunk(
                        page_number=page_chunk.page_number,
                        section_index=idx,
                        text=pending,
                    )
                )
                idx += 1
                pending = ""

        # Flush any remaining text
        if pending:
            if sections and sections[-1].page_number == page_chunk.page_number:
                # Merge short tail into last section of this page
                last = sections[-1]
                sections[-1] = SectionChunk(
                    page_number=last.page_number,
                    section_index=last.section_index,
                    text=last.text + " " + pending,
                )
            else:
                sections.append(
                    SectionChunk(
                        page_number=page_chunk.page_number,
                        section_index=idx,
                        text=pending,
                    )
                )

    return sections


def query_pdf(
    pdf_doc: PDFDocument,
    question: str,
    top_k: int = 5,
    min_chars: int = 80,
) -> List[Dict]:
    """RAG-lite targeted query against paragraph-level chunks.

    Scores each SectionChunk by term overlap with `question` (TF-style) and
    returns the `top_k` most relevant passages.  This allows detective nodes to
    ask specific questions (e.g., "What does the report say about reducers?")
    rather than consuming the full document text.

    Args:
        pdf_doc: The ingested PDFDocument.
        question: A natural-language question or set of terms to search for.
        top_k: Number of top passages to return.
        min_chars: Minimum paragraph size forwarded to `chunk_by_paragraphs`.

    Returns:
        List of dicts with keys: page, section_index, score, text.
    """
    sections = chunk_by_paragraphs(pdf_doc, min_chars=min_chars)
    if not sections:
        return []

    # Tokenise question into lower-case terms (strip punctuation)
    question_terms = set(re.sub(r"[^\w\s]", "", question.lower()).split())
    if not question_terms:
        return [{"page": s.page_number, "section_index": s.section_index, "score": 0.0, "text": s.text}
                for s in sections[:top_k]]

    scored: List[Tuple[float, SectionChunk]] = []
    for section in sections:
        section_terms = set(re.sub(r"[^\w\s]", "", section.text.lower()).split())
        overlap = len(question_terms & section_terms)
        # Normalise by question length so short questions don't always win
        score = overlap / max(len(question_terms), 1)
        scored.append((score, section))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "page": s.page_number,
            "section_index": s.section_index,
            "score": round(sc, 4),
            "text": s.text,
        }
        for sc, s in scored[:top_k]
    ]
