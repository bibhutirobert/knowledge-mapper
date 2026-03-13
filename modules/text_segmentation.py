"""
Stage 3: Semantic segmentation.
Detects chapters, sections, paragraphs using rule-based heading detection
and sentence-boundary analysis. Implements RESEARCH_LAYER Stage 3.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional

from modules.pdf_ingestion import IngestedDocument, PageText
from utils.error_handler import safe_stage
from utils.logger import get_logger

logger = get_logger("text_segmentation")

# Heading detection patterns (ordered strongest → weakest signal)
_HEADING_PATTERNS = [
    re.compile(r"^(chapter|part|unit|section)\s+[\divxlcm]+[\.\:\s]", re.I),
    re.compile(r"^\d+[\.\s]+[A-Z][A-Za-z\s]{3,60}$"),
    re.compile(r"^[A-Z][A-Z\s\-]{4,50}$"),          # ALL-CAPS headings
    re.compile(r"^(\d+\.){1,3}\d*\s+[A-Z]"),          # numbered 1.1.2 headings
]


@dataclass
class Segment:
    segment_id: str
    segment_type: str          # "chapter" | "section" | "paragraph" | "figure" | "table"
    title: str
    text: str
    start_page: int
    end_page: int
    depth: int = 0             # 0=chapter, 1=section, 2=subsection
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)


@dataclass
class SegmentationResult:
    segments: List[Segment]
    chapter_count: int
    section_count: int
    total_words: int


def _is_heading(line: str) -> tuple[bool, int]:
    """Returns (is_heading, depth)."""
    line = line.strip()
    if len(line) < 3 or len(line) > 120:
        return False, 0
    for depth, pattern in enumerate(_HEADING_PATTERNS):
        if pattern.match(line):
            return True, depth
    return False, 0


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text block into paragraph units."""
    paras = re.split(r"\n{2,}", text)
    return [p.strip() for p in paras if p.strip() and len(p.strip()) > 20]


def _infer_segment_type(text: str, depth: int) -> str:
    if depth == 0:
        return "chapter"
    if depth == 1:
        return "section"
    if re.search(r"(figure|fig\.|table|exhibit)\s*\d", text, re.I):
        return "figure"
    return "paragraph"


@safe_stage("text_segmentation", fallback_result=None)
def segment_document(doc: IngestedDocument) -> SegmentationResult:
    """
    Convert IngestedDocument into a list of typed Segments.
    Falls back to treating each page as a single segment if heading
    detection yields nothing.
    """
    logger.info(f"Segmenting document: {doc.filename} ({doc.total_pages} pages)")

    segments: List[Segment] = []
    seg_counter = 0
    current_chapter_id: Optional[str] = None
    current_section_id: Optional[str] = None

    for page in doc.all_pages:
        if not page.text.strip():
            continue

        lines = page.text.split("\n")
        buffer_lines: List[str] = []
        buffer_start_page = page.page_number

        for line in lines:
            is_head, depth = _is_heading(line)

            if is_head and buffer_lines:
                # Flush buffered paragraph text as a segment
                para_text = " ".join(buffer_lines).strip()
                if para_text and len(para_text) > 30:
                    seg_id = f"seg_{seg_counter:04d}"
                    seg_counter += 1
                    parent = current_section_id or current_chapter_id
                    seg = Segment(
                        segment_id=seg_id,
                        segment_type="paragraph",
                        title="",
                        text=para_text,
                        start_page=buffer_start_page,
                        end_page=page.page_number,
                        depth=2,
                        parent_id=parent,
                    )
                    segments.append(seg)
                    if parent:
                        for s in segments:
                            if s.segment_id == parent:
                                s.children_ids.append(seg_id)
                buffer_lines = []
                buffer_start_page = page.page_number

                # Register heading as a new segment
                seg_id = f"seg_{seg_counter:04d}"
                seg_counter += 1
                seg_type = _infer_segment_type(line, depth)
                parent_id = current_chapter_id if depth > 0 else None
                seg = Segment(
                    segment_id=seg_id,
                    segment_type=seg_type,
                    title=line.strip(),
                    text="",
                    start_page=page.page_number,
                    end_page=page.page_number,
                    depth=depth,
                    parent_id=parent_id,
                )
                segments.append(seg)
                if depth == 0:
                    current_chapter_id = seg_id
                    current_section_id = None
                elif depth == 1:
                    current_section_id = seg_id
            else:
                buffer_lines.append(line)

        # Flush remaining lines
        if buffer_lines:
            para_text = " ".join(buffer_lines).strip()
            if para_text and len(para_text) > 30:
                seg_id = f"seg_{seg_counter:04d}"
                seg_counter += 1
                parent = current_section_id or current_chapter_id
                segments.append(Segment(
                    segment_id=seg_id,
                    segment_type="paragraph",
                    title="",
                    text=para_text,
                    start_page=page.page_number,
                    end_page=page.page_number,
                    depth=2,
                    parent_id=parent,
                ))

    # Fallback: if heading detection found nothing, treat each page as one segment
    if not segments:
        logger.warning("No heading structure detected — falling back to page-level segments")
        for page in doc.all_pages:
            if page.text.strip():
                seg_id = f"seg_{seg_counter:04d}"
                seg_counter += 1
                segments.append(Segment(
                    segment_id=seg_id,
                    segment_type="section",
                    title=f"Page {page.page_number}",
                    text=page.text,
                    start_page=page.page_number,
                    end_page=page.page_number,
                    depth=1,
                ))

    chapter_count = sum(1 for s in segments if s.segment_type == "chapter")
    section_count = sum(1 for s in segments if s.segment_type == "section")
    total_words = sum(len(s.text.split()) for s in segments)

    logger.info(
        f"Segmentation complete: {len(segments)} segments, "
        f"{chapter_count} chapters, {section_count} sections, "
        f"{total_words:,} words"
    )
    return SegmentationResult(
        segments=segments,
        chapter_count=chapter_count,
        section_count=section_count,
        total_words=total_words,
    )
