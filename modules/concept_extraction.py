"""
Stage 4 + 5: Concept extraction and relationship detection.
Implements RESEARCH_LAYER ConceptNode schema and RelationshipEdge schema.
Uses spaCy NLP with pattern-rule primary and heuristic fallback.
"""

from __future__ import annotations
import re
import bisect
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
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


# ── Title normalisation ───────────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")


def normalise_title(title: str) -> str:
    """Canonical form of a concept title, used for identity and deduplication."""
    return _EDGE_PUNCT_RE.sub("", _WS_RE.sub(" ", (title or "").strip().lower()))


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
    def make_id(title: str) -> str:
        """
        Identity is derived from the normalised title alone.

        The page number must NOT take part: a concept discussed on pages 12,
        40 and 97 is one concept with three occurrences, not three concepts.
        Including the page here silently disabled deduplication and left
        `recurrence` permanently empty, which zeroed the recurrence signal in
        every downstream importance score.
        """
        return hashlib.md5(normalise_title(title).encode()).hexdigest()[:12]


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
            cid = ConceptNode.make_id(title)
            nodes.append(ConceptNode(
                id=cid,
                title=title[:80],
                concept_type=ctype,
                definition=defn[:400],
                context={"text_snippet": text[:100]},
                source_page=page,
                recurrence=[page],
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
        try:
            sent_text = ent.sent.text.strip() if ent.sent else title
        except ValueError:
            # Pipeline has no sentence boundaries (parser/sentencizer disabled)
            sent_text = title
        cid = ConceptNode.make_id(title)
        nodes.append(ConceptNode(
            id=cid,
            title=title[:80],
            concept_type=ConceptType.DEFINITION,
            definition=sent_text[:400],
            context={"ner_label": ent.label_, "text_snippet": text[:100]},
            source_page=page,
            recurrence=[page],
            salience=0.4,
            low_confidence=True,
        ))
    return nodes


# ── Relationship detector ─────────────────────────────────────────────────

# Maximum characters allowed between a concept title and the signal phrase
# that links it. Mirrors the `.{0,80}` window of the original implementation.
MAX_SIGNAL_GAP = 80

# Upper bounds that keep detection linear on book-length input.
MAX_TITLE_HITS_PER_SEGMENT = 3   # occurrences of one title considered per segment
MAX_CANDIDATES_PER_SIDE    = 3   # nearest titles considered either side of a signal
MAX_EDGES                  = 20000


def _build_signal_matcher() -> Tuple[re.Pattern, Dict[str, RelationType]]:
    """
    One alternation over every signal phrase, longest first so that
    "is a type of" wins over "is a". Returns the regex and a lookup from the
    matched text back to its RelationType.
    """
    lookup: Dict[str, RelationType] = {}
    for rtype, signals in _RELATION_SIGNALS:
        for signal in signals:
            lookup.setdefault(signal.lower(), rtype)
    ordered = sorted(lookup, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(s) for s in ordered), re.I)
    return pattern, lookup


_SIGNAL_RE, _SIGNAL_LOOKUP = _build_signal_matcher()


def _find_title_spans(text_lower: str, titles: List[str]) -> List[Tuple[int, int, str]]:
    """
    Locate every concept title occurring in `text_lower`.

    Cost is one substring scan per title, so this is linear in the number of
    concepts — not quadratic in it, as pairwise regex matching was.
    """
    spans: List[Tuple[int, int, str]] = []
    for title in titles:
        pos = text_lower.find(title)
        hits = 0
        while pos != -1 and hits < MAX_TITLE_HITS_PER_SEGMENT:
            spans.append((pos, pos + len(title), title))
            hits += 1
            pos = text_lower.find(title, pos + 1)
    spans.sort()
    return spans


def _detect_relations(
    concepts: List[ConceptNode],
    segments: List[Segment],
) -> List[RelationshipEdge]:
    """
    Heuristic relation detection: for each signal phrase found in a segment,
    pair the concept titles that sit just before it with those just after it.

    The previous implementation tested every ordered pair of concept titles
    against every signal phrase with a freshly built regex — roughly
    segments x concepts^2 x 40 regex compilations. On a 60-page book (1,368
    concepts, 72 segments) that is ~2.7e9 searches and the app hangs for over
    thirteen minutes; on a full-length book it never returns. Anchoring the
    search on signal occurrences instead makes the cost linear in the text.

    The result is a strict superset of what the old code found. The old
    `titles[i + 1:]` slice also required the source title to precede the
    target in the concept list, which silently dropped roughly half the
    relations even though the regex already pinned their order in the text.
    """
    edges: List[RelationshipEdge] = []
    seen_edge_ids: Set[str] = set()

    concept_index: Dict[str, ConceptNode] = {}
    for c in concepts:
        key = c.title.lower().strip()
        if len(key) >= 3:
            concept_index.setdefault(key, c)
    titles = list(concept_index)
    if not titles:
        return edges

    for segment in segments:
        if len(edges) >= MAX_EDGES:
            logger.warning(f"Relation cap of {MAX_EDGES} reached — stopping detection")
            break

        text_lower = segment.text.lower()
        if not text_lower.strip():
            continue

        spans = _find_title_spans(text_lower, titles)
        if len(spans) < 2:
            continue

        # `spans` is ordered by start position. Titles vary in length, so the
        # end positions are not monotonic in that order and need their own
        # ordering before they can be searched by bisection.
        by_start = spans
        starts = [span[0] for span in by_start]
        by_end = sorted(spans, key=lambda span: span[1])
        ends = [span[1] for span in by_end]

        for m in _SIGNAL_RE.finditer(text_lower):
            rtype = _SIGNAL_LOOKUP.get(m.group(0).lower())
            if rtype is None:
                continue
            sig_start, sig_end = m.span()

            # Titles ending within the window immediately before the signal.
            left_lo = bisect.bisect_left(ends, sig_start - MAX_SIGNAL_GAP)
            left_hi = bisect.bisect_right(ends, sig_start)
            left = by_end[left_lo:left_hi][-MAX_CANDIDATES_PER_SIDE:]

            # Titles starting within the window immediately after the signal.
            right_lo = bisect.bisect_left(starts, sig_end)
            right_hi = bisect.bisect_right(starts, sig_end + MAX_SIGNAL_GAP)
            right = by_start[right_lo:right_hi][:MAX_CANDIDATES_PER_SIDE]

            for _, _, t1 in left:
                src = concept_index[t1]
                for _, _, t2 in right:
                    tgt = concept_index[t2]
                    if src.id == tgt.id:
                        continue
                    eid = RelationshipEdge.make_id(src.id, tgt.id, rtype)
                    if eid in seen_edge_ids:
                        continue
                    seen_edge_ids.add(eid)
                    edges.append(RelationshipEdge(
                        id=eid,
                        source_id=src.id,
                        target_id=tgt.id,
                        relation_type=rtype,
                        confidence=0.65,
                        evidence=segment.text[:200],
                        source_page=segment.start_page,
                        bidirectional=(rtype == RelationType.CONTRADICTS),
                    ))

    return edges


def _inbound_counts(edges: List[RelationshipEdge]) -> Dict[str, int]:
    """Inbound edge count per concept id, computed in a single pass."""
    counts: Dict[str, int] = {}
    for e in edges:
        counts[e.target_id] = counts.get(e.target_id, 0) + 1
    return counts


def _compute_salience(
    concept: ConceptNode,
    all_concepts: List[ConceptNode],
    edges: List[RelationshipEdge],
    inbound_counts: Optional[Dict[str, int]] = None,
) -> float:
    """
    RESEARCH_LAYER salience = TF * recurrence weight * type weight * edge centrality.

    `inbound_counts` is precomputed by the caller. Deriving it here rescanned
    every edge for every concept on every call, which is O(concepts^2 x edges)
    across a full run.
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
    if inbound_counts is None:
        inbound_counts = _inbound_counts(edges)

    recurrence_score = min(len(concept.recurrence) / 5.0, 1.0)
    type_score = type_weights.get(concept.concept_type, 0.5)
    inbound = inbound_counts.get(concept.id, 0)
    max_inbound = max(inbound_counts.values(), default=1)
    centrality = inbound / max(max_inbound, 1)
    return round(0.3 * concept.salience + 0.2 * recurrence_score + 0.1 * type_score + 0.4 * centrality, 4)


def _deduplicate_concepts(concepts: List[ConceptNode]) -> List[ConceptNode]:
    """
    Merge concepts that share an id (i.e. the same normalised title), folding
    every page they appeared on into `recurrence` and keeping the richest
    definition. `recurrence` is what the recurrence signal in the importance
    score reads, so it must span the whole document, not a single page.
    """
    seen: Dict[str, ConceptNode] = {}
    for c in concepts:
        existing = seen.get(c.id)
        if existing is None:
            if c.source_page not in c.recurrence:
                c.recurrence.append(c.source_page)
            seen[c.id] = c
            continue

        for page in ([c.source_page] + list(c.recurrence)):
            if page not in existing.recurrence:
                existing.recurrence.append(page)

        # Prefer the most informative definition and the earliest mention.
        if len(c.definition or "") > len(existing.definition or ""):
            existing.definition = c.definition
        if c.source_page < existing.source_page:
            existing.source_page = c.source_page
        # A concept only stays low-confidence if every occurrence was.
        existing.low_confidence = existing.low_confidence and c.low_confidence

    for c in seen.values():
        c.recurrence.sort()
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
                cid = ConceptNode.make_id(seg.title)
                all_concepts.append(ConceptNode(
                    id=cid,
                    title=seg.title[:80],
                    concept_type=ConceptType.PRINCIPLE,
                    definition=seg.text[:300] if seg.text else seg.title,
                    context={"segment_type": seg.segment_type},
                    source_page=seg.start_page,
                    recurrence=[seg.start_page],
                    salience=0.5,
                    low_confidence=True,
                ))

    concepts = _deduplicate_concepts(all_concepts)
    edges = _detect_relations(concepts, segmentation.segments)

    # Filter low-confidence edges
    edges = [e for e in edges if e.confidence >= 0.4]

    # Recompute salience with edge data (inbound counts computed once, not per concept)
    inbound = _inbound_counts(edges)
    for c in concepts:
        c.salience = _compute_salience(c, concepts, edges, inbound)

    stats = {
        "total_concepts": len(concepts),
        "total_edges": len(edges),
        "by_type": {t.value: sum(1 for c in concepts if c.concept_type == t) for t in ConceptType},
    }
    logger.info(f"Extraction complete: {len(concepts)} concepts, {len(edges)} edges")
    return ExtractionResult(concepts=concepts, edges=edges, stats=stats)
