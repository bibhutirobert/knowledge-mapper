"""
Stage 1 + 2: PDF ingestion and text extraction.
Implements RESEARCH_LAYER stages 1-2 and SYSTEM_ARCHITECTURE PDFGateway + OCRWorker.
Fallback chain: pdfplumber → PyMuPDF → Tesseract OCR.
"""

from __future__ import annotations
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from utils.error_handler import retry, safe_stage, IngestionError, OCRError
from utils.logger import get_logger

logger = get_logger("pdf_ingestion")

CHUNK_SIZE = 20      # pages per processing window (SYSTEM_ARCHITECTURE spec)
CHUNK_OVERLAP = 3    # overlap pages between windows


@dataclass
class PageText:
    page_number: int
    text: str
    width: float = 0.0
    height: float = 0.0
    is_ocr: bool = False


@dataclass
class DocumentChunk:
    chunk_index: int
    start_page: int
    end_page: int
    pages: List[PageText] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


@dataclass
class IngestedDocument:
    filename: str
    total_pages: int
    chunks: List[DocumentChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        seen = set()
        texts = []
        for chunk in self.chunks:
            for page in chunk.pages:
                if page.page_number not in seen:
                    seen.add(page.page_number)
                    if page.text.strip():
                        texts.append(page.text)
        return "\n\n".join(texts)

    @property
    def all_pages(self) -> List[PageText]:
        seen = set()
        pages = []
        for chunk in self.chunks:
            for page in chunk.pages:
                if page.page_number not in seen:
                    seen.add(page.page_number)
                    pages.append(page)
        return sorted(pages, key=lambda p: p.page_number)


# ── Primary extractor: pdfplumber ──────────────────────────────────────────

def _extract_with_pdfplumber(file_bytes: bytes) -> List[PageText]:
    import pdfplumber
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageText(
                page_number=i,
                text=text.strip(),
                width=float(page.width or 0),
                height=float(page.height or 0),
            ))
    return pages


# ── Fallback extractor 1: PyMuPDF ─────────────────────────────────────────

def _extract_with_pymupdf(file_bytes: bytes) -> List[PageText]:
    import fitz  # PyMuPDF
    pages = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        rect = page.rect
        pages.append(PageText(
            page_number=i,
            text=text.strip(),
            width=float(rect.width),
            height=float(rect.height),
        ))
    doc.close()
    return pages


# ── Fallback extractor 2: Tesseract OCR ───────────────────────────────────

def _ocr_page(image_bytes: bytes, page_number: int) -> PageText:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, config="--psm 6")
        return PageText(page_number=page_number, text=text.strip(), is_ocr=True)
    except Exception as exc:
        raise OCRError(f"Tesseract failed on page {page_number}: {exc}") from exc


def _extract_with_ocr(file_bytes: bytes) -> List[PageText]:
    import fitz
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        mat = fitz.Matrix(2.0, 2.0)          # 2× upscale for OCR quality
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        try:
            pt = _ocr_page(img_bytes, i)
        except OCRError as exc:
            logger.warning(str(exc))
            pt = PageText(page_number=i, text="", is_ocr=True)
        pages.append(pt)
    doc.close()
    return pages


# ── Chunker ────────────────────────────────────────────────────────────────

def _build_chunks(pages: List[PageText]) -> List[DocumentChunk]:
    if not pages:
        return []
    chunks = []
    idx = 0
    chunk_index = 0
    total = len(pages)
    while idx < total:
        end = min(idx + CHUNK_SIZE, total)
        chunk_pages = pages[idx:end]
        chunks.append(DocumentChunk(
            chunk_index=chunk_index,
            start_page=chunk_pages[0].page_number,
            end_page=chunk_pages[-1].page_number,
            pages=chunk_pages,
        ))
        idx += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_index += 1
    return chunks


# ── Public API ─────────────────────────────────────────────────────────────

@retry(max_attempts=3, delay=0.5, exceptions=(Exception,))
def ingest_pdf(file_bytes: bytes, filename: str = "document.pdf") -> IngestedDocument:
    """
    Main ingestion entry point. Tries pdfplumber → PyMuPDF → OCR.
    Returns IngestedDocument with chunked pages.
    """
    logger.info(f"Ingesting '{filename}' ({len(file_bytes):,} bytes)")

    pages: Optional[List[PageText]] = None
    method_used = "unknown"

    # Primary: pdfplumber
    try:
        pages = _extract_with_pdfplumber(file_bytes)
        non_empty = sum(1 for p in pages if p.text.strip())
        if non_empty < max(1, len(pages) * 0.3):
            raise IngestionError("pdfplumber: < 30% pages have text — trying fallback")
        method_used = "pdfplumber"
        logger.info(f"pdfplumber extracted {len(pages)} pages ({non_empty} with text)")
    except Exception as exc:
        logger.warning(f"pdfplumber failed: {exc}")
        pages = None

    # Fallback 1: PyMuPDF
    if pages is None:
        try:
            pages = _extract_with_pymupdf(file_bytes)
            non_empty = sum(1 for p in pages if p.text.strip())
            if non_empty < max(1, len(pages) * 0.2):
                raise IngestionError("PyMuPDF: too little text — falling back to OCR")
            method_used = "pymupdf"
            logger.info(f"PyMuPDF extracted {len(pages)} pages")
        except Exception as exc:
            logger.warning(f"PyMuPDF failed: {exc}")
            pages = None

    # Fallback 2: Tesseract OCR
    if pages is None:
        logger.info("Falling back to Tesseract OCR")
        try:
            pages = _extract_with_ocr(file_bytes)
            method_used = "tesseract_ocr"
            logger.info(f"OCR extracted {len(pages)} pages")
        except Exception as exc:
            raise IngestionError(f"All extraction methods failed: {exc}") from exc

    if not pages:
        raise IngestionError("No pages extracted from document")

    chunks = _build_chunks(pages)
    doc = IngestedDocument(
        filename=filename,
        total_pages=len(pages),
        chunks=chunks,
        metadata={"extraction_method": method_used, "chunk_count": len(chunks)},
    )
    logger.info(
        f"Ingestion complete: {len(pages)} pages, {len(chunks)} chunks "
        f"(method: {method_used})"
    )
    return doc
