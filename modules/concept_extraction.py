"""
Stage 4 + 5: Concept extraction and relationship detection.
Implements RESEARCH_LAYER ConceptNode schema and RelationshipEdge schema.
Uses spaCy NLP with pattern-rule primary and heuristic fallback.
"""

from __future__ import annotations
import re
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum

from modules.text_segmentation import SegmentationResult, Segment
from utils.error_handler import retry, safe_stage, ExtractionError
from utils.logger import get_logger

logger = get_logger("concept_extraction")


# ── RESEARCH_LAYER Enums ──────────────────────────────────────────────────

class ConceptType(str, Enum):
    DEFINITION  = "DEFINITION"
    PRINCIPLE   = "PRINCIPLE"
    THEOREM     = "THEOREM"
    EXAMPLE     = "EXAMPLE"
    ARGUMENT    = "ARGUMENT"
    CONCLUSION  = "CONCLUSION"
    INSIGHT     = "INSIGHT"


class RelationType(str, Enum):
    IS_A         = "IS_A"
    PART_OF      = "PART_OF"
    DEPENDS_ON   = "DEPENDS_ON"
    EXAMPLE_OF   = "EXAMPLE_OF"
    CONTRADICTS  = "CONTRADICTS"
    SUPPORTS     = "SUPPORTS"
    DERIVED_FROM = "DERIVED_FROM"


# ── RESEARCH_LAYER Schemas ────────────────────────────────────────────────

@dataclass
class ConceptNode:
    id: str
    title: str
    concept_type: ConceptType
    definition: str
    context: Dict
    source_page: int
    recurrence: List[int] = field(default_factory=list)
    salience: float = 0.5
    tags: List[str] = field(default_factory=list)
    low_confidence: bool = False

    @staticmethod
    def make_id(title: str, page: int) -> str:
        raw = f"{title.lower().strip()}:{page}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass
class RelationshipEdge:
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float
    evidence: str
    source_page: int
    bidirectional: bool = False
    weight: float = 1.0

    @staticmethod
    def make_id(src: str, tgt: str, rtype: RelationType) -> str:
        raw = f"{src}:{tgt}:{rtype.value}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass
class ExtractionResult:
    concepts: List[ConceptNode]
    edges: List[RelationshipEdge]
    stats: Dict = field(default_factory=dict)


# ── Linguistic patterns for concept detection ─────────────────────────────

_DEFINITION_PATTERNS = [
    re.compile(r"([A-Z][a-z\s]+)\s+(?:is defined as|is|refers to|means|denotes)\s+(.{20,200})", re.I),
    re.compile(r"(?:Definition|Def\.)\s*[\:\-]?\s*([A-Z][a-zA-Z\s\-]+)\s*[\:\-]\s*(.{20,200})", re.I),
]
_THEOREM_PATTERNS = [
    re.compile(r"(?:Theorem|Lemma|Corollary|Proposition)\s*[\d\.]*\s*[\:\-]?\s*(.{20,300})", re.I),
]
_PRINCIPLE_PATTERNS = [
    re.compile(r"(?:The\s+)?(?:principle|law|rule|property)\s+(?:of\s+)?([A-Z][a-z\s\-]{3,50})", re.I),
    re.compile(r"([A-Z][a-z\s]+)\s+(?:principle|theorem|law|effect|hypothesis)", re.I),
]
_CONCLUSION_PATTERNS = [
    re.compile(r"(?:Therefore|Thus|Hence|In conclusion|Consequently)[,\s]+(.{20,300})", re.I),
    re.compile(r"(?:We\s+conclude\s+that|This\s+shows\s+that|It\s+follows\s+that)\s+(.{20,300})", re.I),
]
_EXAMPLE_PATTERNS = [
    re.compile(r"(?:For\s+example|For\s+instance|e\.g\.|As\s+an\s+example)[,\s]+(.{20,200})", re.I),
]
_ARGUMENT_PATTERNS = [
    re.compile(r"(?:Because|Since|Given\s+that|If)\s+(.{20,200})", re.I),
]

# Relation signal words
_RELATION_SIGNALS: List[Tuple[RelationType, List[str]]] = [
    (RelationType.IS_A,         ["is a", "is an", "is a type of", "is a kind of", "is a form of"]),
    (RelationType.PART_OF,      ["is part of", "belongs to", "is contained in", "is a component of"]),
    (RelationType.DEPENDS_ON,   ["depends on", "requires", "relies on", "is based on", "needs"]),
    (RelationType.EXAMPLE_OF,   ["is an example of", "illustrates", "demonstrates", "is a case of"]),
    (RelationType.SUPPORTS,     ["supports", "confirms", "validates", "provides evidence for"]),
    (RelationType.CONTRADICTS,  ["contradicts", "opposes", "conflicts with", "refutes", "challenges"]),
    (RelationType.DERIVED_FROM, ["is derived from", "follows from", "stems from", "originates from"]),
]


# ── NLP loader (lazy, with fallback) ─────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded")
        except OSError:
            logger.warning("spaCy model not found — downloading en_core_web_sm")
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except ImportError:
        logger.warning("spaCy not available — using regex-only extraction")
        return None


# ── Concept extractors ────────────────────────────────────────────────────

def _extract_from_pattern(text: str, page: int, patterns, ctype: ConceptType) -> List[ConceptNode]:
    nodes = []
    for pat in patterns:
        for m in pat.finditer(text):
            groups = m.groups()
            if len(groups) >= 2:
                title, defn = groups[0].strip(), groups[1].strip()
            elif len(groups) == 1:
                title = groups[0].strip()[:60]
                defn = groups[0].strip()
            else:
                continue
            if len(title) < 3 or len(defn) < 10:
                continue
            cid = ConceptNode.make_id(title, page)
            nodes.append(ConceptNode(
                id=cid,
                title=title[:80],
                concept_type=ctype,
                definition=defn[:400],
                context={"text_snippet": text[:100]},
                source_page=page,
                salience=0.5,
            ))
    return nodes


def _extract_with_spacy(text: str, page: int, nlp) -> List[ConceptNode]:
    """Use NER to surface named entities as concept candidates."""
    doc = nlp(text[:5000])  # cap for speed
    nodes = []
    seen_titles = set()
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "PERCENT", "MONEY"):
            continue  # filter non-concept entity types
        title = ent.text.strip()
        if len(title) < 3 or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        # Extract surrounding sentence as definition
        sent_text = ent.sent.text.strip() if ent.sent else title
        cid = ConceptNode.make_id(title, page)
        nodes.append(ConceptNode(
            id=cid,
            title=title[:80],
            concept_type=ConceptType.DEFINITION,
            definition=sent_text[:400],
            context={"ner_label": ent.label_, "text_snippet": text[:100]},
            source_page=page,
            salience=0.4,
            low_confidence=True,
        ))
    return nodes


# ── Relationship detector ─────────────────────────────────────────────────

def _detect_relations(
    concepts: List[ConceptNode],
    segments: List[Segment],
) -> List[RelationshipEdge]:
    """
    Heuristic relation detection: scan segment text for signal phrases
    between pairs of concept titles.
    """
    edges: List[RelationshipEdge] = []
    concept_index = {c.title.lower(): c for c in concepts}

    for segment in segments:
        text_lower = segment.text.lower()
        titles = list(concept_index.keys())

        for i, t1 in enumerate(titles):
            if t1 not in text_lower:
                continue
            for t2 in titles[i + 1:]:
                if t2 not in text_lower:
                    continue
                # Check if both appear in same sentence and if a signal word connects them
                for rtype, signals in _RELATION_SIGNALS:
                    for signal in signals:
                        pattern = rf"{re.escape(t1)}.{{0,80}}{re.escape(signal)}.{{0,80}}{re.escape(t2)}"
                        if re.search(pattern, text_lower, re.I | re.S):
                            src = concept_index[t1]
                            tgt = concept_index[t2]
                            eid = RelationshipEdge.make_id(src.id, tgt.id, rtype)
                            edge = RelationshipEdge(
                                id=eid,
                                source_id=src.id,
                                target_id=tgt.id,
                                relation_type=rtype,
                                confidence=0.65,
                                evidence=segment.text[:200],
                                source_page=segment.start_page,
                                bidirectional=(rtype == RelationType.CONTRADICTS),
                            )
                            edges.append(edge)
    return edges


def _compute_salience(
    concept: ConceptNode,
    all_concepts: List[ConceptNode],
    edges: List[RelationshipEdge],
) -> float:
    """
    RESEARCH_LAYER salience = TF * recurrence weight * type weight * edge centrality.
    """
    type_weights = {
        ConceptType.THEOREM:    1.0,
        ConceptType.PRINCIPLE:  0.9,
        ConceptType.CONCLUSION: 0.8,
        ConceptType.DEFINITION: 0.7,
        ConceptType.ARGUMENT:   0.6,
        ConceptType.INSIGHT:    0.6,
        ConceptType.EXAMPLE:    0.3,
    }
    recurrence_score = min(len(concept.recurrence) / 5.0, 1.0)
    type_score = type_weights.get(concept.concept_type, 0.5)
    inbound = sum(1 for e in edges if e.target_id == concept.id)
    max_inbound = max((sum(1 for e in edges if e.target_id == c.id) for c in all_concepts), default=1)
    centrality = inbound / max(max_inbound, 1)
    return round(0.3 * concept.salience + 0.2 * recurrence_score + 0.1 * type_score + 0.4 * centrality, 4)


def _deduplicate_concepts(concepts: List[ConceptNode]) -> List[ConceptNode]:
    """Merge concepts with identical ids; track recurrence pages."""
    seen: Dict[str, ConceptNode] = {}
    for c in concepts:
        if c.id in seen:
            existing = seen[c.id]
            if c.source_page not in existing.recurrence:
                existing.recurrence.append(c.source_page)
        else:
            seen[c.id] = c
    return list(seen.values())


# ── Public API ────────────────────────────────────────────────────────────

@safe_stage("concept_extraction", fallback_result=ExtractionResult([], [], {}))
def extract_concepts(segmentation: SegmentationResult) -> ExtractionResult:
    """
    Extract ConceptNodes and RelationshipEdges from segmented document.
    Implements RESEARCH_LAYER Stages 4 + 5.
    """
    logger.info(f"Extracting concepts from {len(segmentation.segments)} segments")
    nlp = _get_nlp()
    all_concepts: List[ConceptNode] = []

    for segment in segmentation.segments:
        text = segment.text
        if not text.strip():
            continue
        page = segment.start_page

        # Pattern-based extraction
        all_concepts += _extract_from_pattern(text, page, _DEFINITION_PATTERNS, ConceptType.DEFINITION)
        all_concepts += _extract_from_pattern(text, page, _THEOREM_PATTERNS, ConceptType.THEOREM)
        all_concepts += _extract_from_pattern(text, page, _PRINCIPLE_PATTERNS, ConceptType.PRINCIPLE)
        all_concepts += _extract_from_pattern(text, page, _CONCLUSION_PATTERNS, ConceptType.CONCLUSION)
        all_concepts += _extract_from_pattern(text, page, _EXAMPLE_PATTERNS, ConceptType.EXAMPLE)
        all_concepts += _extract_from_pattern(text, page, _ARGUMENT_PATTERNS, ConceptType.ARGUMENT)

        # spaCy NER enrichment
        if nlp:
            all_concepts += _extract_with_spacy(text, page, nlp)

    # Fallback: if extraction found < 3 concepts, create concept per segment title
    if len(all_concepts) < 3:
        logger.warning("Few concepts found — creating segment-title concepts as fallback")
        for seg in segmentation.segments:
            if seg.title:
                cid = ConceptNode.make_id(seg.title, seg.start_page)
                all_concepts.append(ConceptNode(
                    id=cid,
                    title=seg.title[:80],
                    concept_type=ConceptType.PRINCIPLE,
                    definition=seg.text[:300] if seg.text else seg.title,
                    context={"segment_type": seg.segment_type},
                    source_page=seg.start_page,
                    salience=0.5,
                    low_confidence=True,
                ))

    concepts = _deduplicate_concepts(all_concepts)
    edges = _detect_relations(concepts, segmentation.segments)

    # Filter low-confidence edges
    edges = [e for e in edges if e.confidence >= 0.4]

    # Recompute salience with edge data
    for c in concepts:
        c.salience = _compute_salience(c, concepts, edges)

    stats = {
        "total_concepts": len(concepts),
        "total_edges": len(edges),
        "by_type": {t.value: sum(1 for c in concepts if c.concept_type == t) for t in ConceptType},
    }
    logger.info(f"Extraction complete: {len(concepts)} concepts, {len(edges)} edges")
    return ExtractionResult(concepts=concepts, edges=edges, stats=stats)
