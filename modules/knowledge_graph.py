"""
Stage 6 + 7: Knowledge graph construction and hierarchy synthesis.
Implements RESEARCH_LAYER graph types, HierarchyEngine, InsightEngine,
and SemanticValidator using NetworkX.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from modules.concept_extraction import (
    ConceptNode, ConceptType, RelationshipEdge, RelationType, ExtractionResult,
)
from utils.error_handler import safe_stage, GraphError
from utils.logger import get_logger

logger = get_logger("knowledge_graph")

# ── Insight tags (RESEARCH_LAYER InsightEngine) ───────────────────────────

INSIGHT_CORE       = "CORE_CONCEPT"
INSIGHT_KEY        = "KEY_INSIGHT"
INSIGHT_SUPPORTING = "SUPPORTING_IDEA"
INSIGHT_REPEATED   = "REPEATED_EXPLANATION"

# ── RESEARCH_LAYER importance tiers (used by CompressionEngine) ───────────

TIER_CORE       = "CORE"
TIER_SUPPORTING = "SUPPORTING"
TIER_PRUNABLE   = "PRUNABLE"
TIER_REDUNDANT  = "REDUNDANT"


@dataclass
class KnowledgeGraph:
    concept_graph: nx.DiGraph          # all nodes + all typed edges
    hierarchy_tree: nx.DiGraph         # IS_A + PART_OF only, strict DAG
    dependency_graph: nx.DiGraph       # DEPENDS_ON + DERIVED_FROM
    insight_graph: nx.DiGraph          # SUPPORTS + CONTRADICTS
    concepts: Dict[str, ConceptNode]   # id → ConceptNode
    edges: List[RelationshipEdge]
    insight_tags: Dict[str, str]       # id → insight tag
    importance_scores: Dict[str, float]  # id → final importance score
    validation_report: Dict = field(default_factory=dict)


# ── Graph construction ────────────────────────────────────────────────────

def _build_concept_graph(
    concepts: List[ConceptNode],
    edges: List[RelationshipEdge],
) -> nx.DiGraph:
    G = nx.DiGraph()
    for c in concepts:
        G.add_node(c.id, **{
            "title": c.title,
            "concept_type": c.concept_type.value,
            "definition": c.definition,
            "salience": c.salience,
            "source_page": c.source_page,
            "recurrence": c.recurrence,
            "tags": c.tags,
        })
    for e in edges:
        if G.has_node(e.source_id) and G.has_node(e.target_id):
            G.add_edge(e.source_id, e.target_id, **{
                "relation_type": e.relation_type.value,
                "confidence": e.confidence,
                "weight": e.weight,
                "evidence": e.evidence,
            })
    return G


def _build_hierarchy_tree(
    concept_graph: nx.DiGraph,
    edges: List[RelationshipEdge],
) -> nx.DiGraph:
    """
    Strict DAG using IS_A and PART_OF edges only.
    Cycle detection applied per RESEARCH_LAYER HierarchyEngine spec.
    """
    H = nx.DiGraph()
    H.add_nodes_from(concept_graph.nodes(data=True))

    hierarchy_types = {RelationType.IS_A.value, RelationType.PART_OF.value}
    for e in edges:
        if e.relation_type.value not in hierarchy_types:
            continue
        if not (H.has_node(e.source_id) and H.has_node(e.target_id)):
            continue
        # Cycle guard: add edge only if it doesn't create a cycle
        H.add_edge(e.source_id, e.target_id, relation_type=e.relation_type.value)
        if not nx.is_directed_acyclic_graph(H):
            H.remove_edge(e.source_id, e.target_id)
            logger.debug(f"Cycle prevented: {e.source_id} → {e.target_id}")

    return H


def _build_dependency_graph(
    concept_graph: nx.DiGraph,
    edges: List[RelationshipEdge],
) -> nx.DiGraph:
    D = nx.DiGraph()
    D.add_nodes_from(concept_graph.nodes(data=True))
    dep_types = {RelationType.DEPENDS_ON.value, RelationType.DERIVED_FROM.value}
    for e in edges:
        if e.relation_type.value in dep_types and D.has_node(e.source_id) and D.has_node(e.target_id):
            D.add_edge(e.source_id, e.target_id, relation_type=e.relation_type.value)
    return D


def _build_insight_graph(
    concept_graph: nx.DiGraph,
    edges: List[RelationshipEdge],
) -> nx.DiGraph:
    I = nx.DiGraph()
    I.add_nodes_from(concept_graph.nodes(data=True))
    insight_types = {RelationType.SUPPORTS.value, RelationType.CONTRADICTS.value}
    for e in edges:
        if e.relation_type.value in insight_types and I.has_node(e.source_id) and I.has_node(e.target_id):
            I.add_edge(e.source_id, e.target_id, relation_type=e.relation_type.value)
    return I


# ── Importance scoring (COMPRESSION_ENGINE formula) ──────────────────────

def _compute_importance_scores(
    concepts: List[ConceptNode],
    concept_graph: nx.DiGraph,
    dependency_graph: nx.DiGraph,
    edges: List[RelationshipEdge],
) -> Dict[str, float]:
    """
    ImportanceScore = 0.30*S1 + 0.25*S2 + 0.20*S3 + 0.15*S4 + 0.10*S5
    """
    type_weights = {
        ConceptType.THEOREM:    1.0, ConceptType.PRINCIPLE:  0.9,
        ConceptType.CONCLUSION: 0.8, ConceptType.DEFINITION: 0.7,
        ConceptType.ARGUMENT:   0.6, ConceptType.INSIGHT:    0.6,
        ConceptType.EXAMPLE:    0.3,
    }
    max_recurrence = max((len(c.recurrence) for c in concepts), default=1) or 1
    max_inbound    = max((concept_graph.in_degree(c.id) for c in concepts), default=1) or 1
    max_dep_out    = max((dependency_graph.out_degree(c.id) for c in concepts), default=1) or 1
    max_support_in = max(
        (sum(1 for e in edges if e.target_id == c.id and e.relation_type in
             {RelationType.SUPPORTS, RelationType.DERIVED_FROM})
         for c in concepts), default=1) or 1

    scores = {}
    for c in concepts:
        s1 = c.salience
        in_deg  = concept_graph.in_degree(c.id)
        dep_out = dependency_graph.out_degree(c.id)
        s2 = min((in_deg + dep_out * 1.5) / max(max_inbound + max_dep_out * 1.5, 1), 1.0)
        s3 = len(c.recurrence) / max_recurrence
        support_in = sum(1 for e in edges if e.target_id == c.id and
                         e.relation_type in {RelationType.SUPPORTS, RelationType.DERIVED_FROM})
        s4 = support_in / max_support_in
        s5 = type_weights.get(c.concept_type, 0.5)
        score = 0.30 * s1 + 0.25 * s2 + 0.20 * s3 + 0.15 * s4 + 0.10 * s5
        scores[c.id] = round(min(score, 1.0), 4)
    return scores


# ── InsightEngine ─────────────────────────────────────────────────────────

def _run_insight_engine(
    concepts: List[ConceptNode],
    importance_scores: Dict[str, float],
    edges: List[RelationshipEdge],
) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for c in concepts:
        score = importance_scores.get(c.id, 0.0)
        inbound_support = sum(
            1 for e in edges if e.target_id == c.id and
            e.relation_type in {RelationType.SUPPORTS, RelationType.DERIVED_FROM}
        )
        outbound_support = sum(1 for e in edges if e.source_id == c.id and e.relation_type == RelationType.SUPPORTS)

        if score >= 0.75:
            tags[c.id] = INSIGHT_CORE
        elif c.concept_type == ConceptType.CONCLUSION and outbound_support >= 1:
            tags[c.id] = INSIGHT_KEY
        elif not any(e.target_id == c.id for e in edges) and outbound_support >= 1 and score < 0.5:
            tags[c.id] = INSIGHT_SUPPORTING
        else:
            tags[c.id] = INSIGHT_SUPPORTING  # default
    return tags


# ── Semantic Validator (3-pass) ───────────────────────────────────────────

def _semantic_validate(
    concepts: List[ConceptNode],
    edges: List[RelationshipEdge],
    concept_graph: nx.DiGraph,
    hierarchy_tree: nx.DiGraph,
) -> Tuple[List[ConceptNode], List[RelationshipEdge], Dict]:
    report = {"pass1": [], "pass2": [], "pass3": [], "repairs": 0}

    # Pass 1 — Concept structure
    repaired_concepts = []
    for c in concepts:
        issues = []
        if not c.title or len(c.title) > 80:
            c.title = (c.title or "Untitled")[:80]
            issues.append("title_truncated")
        if not c.definition or len(c.definition.split()) < 3:
            c.definition = c.title  # minimal fallback
            c.low_confidence = True
            issues.append("definition_missing")
        if not (0.0 <= c.salience <= 1.0):
            c.salience = 0.5
            issues.append("salience_clamped")
        if issues:
            report["pass1"].append({"id": c.id, "issues": issues})
            report["repairs"] += len(issues)
        repaired_concepts.append(c)

    # Pass 2 — Relationship consistency
    valid_ids = {c.id for c in repaired_concepts}
    repaired_edges = []
    for e in edges:
        if e.source_id not in valid_ids or e.target_id not in valid_ids:
            report["pass2"].append({"edge_id": e.id, "issue": "dangling_reference"})
            report["repairs"] += 1
            continue
        if e.source_id == e.target_id:
            report["pass2"].append({"edge_id": e.id, "issue": "self_loop"})
            report["repairs"] += 1
            continue
        if e.confidence < 0.4:
            report["pass2"].append({"edge_id": e.id, "issue": "quarantined_low_confidence"})
            report["repairs"] += 1
            continue
        repaired_edges.append(e)

    # Pass 3 — Graph integrity
    if not nx.is_directed_acyclic_graph(hierarchy_tree):
        # Remove cycle-forming edges
        cycles = list(nx.simple_cycles(hierarchy_tree))
        for cycle in cycles:
            for i in range(len(cycle)):
                src, tgt = cycle[i], cycle[(i + 1) % len(cycle)]
                if hierarchy_tree.has_edge(src, tgt):
                    hierarchy_tree.remove_edge(src, tgt)
                    report["pass3"].append({"issue": "cycle_removed", "edge": f"{src}→{tgt}"})
                    report["repairs"] += 1
                    break

    # Check connectivity — isolated nodes get tagged low_confidence
    isolated = list(nx.isolates(concept_graph))
    if isolated:
        report["pass3"].append({"issue": "isolated_nodes", "count": len(isolated)})
        for c in repaired_concepts:
            if c.id in isolated:
                c.low_confidence = True

    return repaired_concepts, repaired_edges, report


# ── Public API ────────────────────────────────────────────────────────────

@safe_stage("knowledge_graph", fallback_result=None)
def build_knowledge_graph(extraction: ExtractionResult) -> Optional[KnowledgeGraph]:
    """
    Build all four graph types from ExtractionResult.
    Implements RESEARCH_LAYER Stage 6 + 7 + InsightEngine + SemanticValidator.
    """
    concepts = extraction.concepts
    edges = extraction.edges

    if not concepts:
        raise GraphError("No concepts to build graph from")

    logger.info(f"Building graphs: {len(concepts)} concepts, {len(edges)} edges")

    # Build four graphs
    cg = _build_concept_graph(concepts, edges)
    ht = _build_hierarchy_tree(cg, edges)
    dg = _build_dependency_graph(cg, edges)
    ig = _build_insight_graph(cg, edges)

    # Importance scoring
    importance = _compute_importance_scores(concepts, cg, dg, edges)

    # Insight tagging
    insight_tags = _run_insight_engine(concepts, importance, edges)

    # Semantic validation (3 passes)
    concepts, edges, val_report = _semantic_validate(concepts, edges, cg, ht)

    # Rebuild graphs after validation repair
    cg = _build_concept_graph(concepts, edges)
    ht = _build_hierarchy_tree(cg, edges)
    dg = _build_dependency_graph(cg, edges)
    ig = _build_insight_graph(cg, edges)
    importance = _compute_importance_scores(concepts, cg, dg, edges)
    insight_tags = _run_insight_engine(concepts, importance, edges)

    logger.info(
        f"Graph built: {cg.number_of_nodes()} nodes, {cg.number_of_edges()} edges. "
        f"Validation repairs: {val_report['repairs']}"
    )

    return KnowledgeGraph(
        concept_graph=cg,
        hierarchy_tree=ht,
        dependency_graph=dg,
        insight_graph=ig,
        concepts={c.id: c for c in concepts},
        edges=edges,
        insight_tags=insight_tags,
        importance_scores=importance,
        validation_report=val_report,
    )


def get_tier(importance_score: float) -> str:
    if importance_score >= 0.75: return TIER_CORE
    if importance_score >= 0.45: return TIER_SUPPORTING
    if importance_score >= 0.25: return TIER_PRUNABLE
    return TIER_REDUNDANT
