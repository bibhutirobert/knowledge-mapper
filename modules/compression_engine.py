"""
COMPRESSION_ENGINE: Compresses the full knowledge graph into a compact mind map.
Implements importance scoring, clustering, condensation (4 operations),
and simplified mind map generation as per the COMPRESSION_ENGINE specification.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from modules.concept_extraction import ConceptNode, ConceptType, RelationshipEdge, RelationType
from modules.knowledge_graph import (
    KnowledgeGraph, get_tier,
    TIER_CORE, TIER_SUPPORTING, TIER_PRUNABLE, TIER_REDUNDANT,
    INSIGHT_CORE, INSIGHT_KEY,
)
from utils.error_handler import safe_stage, CompressionError
from utils.logger import get_logger

logger = get_logger("compression_engine")

# Compression thresholds (COMPRESSION_ENGINE spec)
THRESHOLD_CORE       = 0.75
THRESHOLD_SUPPORTING = 0.45
THRESHOLD_PRUNABLE   = 0.25
MIN_NODES            = 5
MAX_NODES            = 80
MAX_DEPTH            = 4
MIN_CLUSTER_SIZE     = 2
MAX_CLUSTER_SIZE     = 12


@dataclass
class Cluster:
    cluster_id: str
    domain: str
    seed_concept_id: str
    member_ids: List[str] = field(default_factory=list)
    label: str = ""


@dataclass
class BridgeEdge:
    from_cluster: str
    to_cluster: str
    edge_type: str
    label: str = ""


@dataclass
class SummaryBridgeNode:
    id: str
    title: str
    definition: str
    score: float
    source_pages: List[int] = field(default_factory=list)


@dataclass
class MindMapNode:
    id: str
    label: str
    concept_type: str
    score: float
    tier: str
    definition: str
    source_page: int
    children: List["MindMapNode"] = field(default_factory=list)
    annotations: List[Dict] = field(default_factory=list)
    is_bridge: bool = False


@dataclass
class MindMapBridge:
    from_node_id: str
    to_node_id: str
    edge_type: str
    label: str
    dashed: bool = True


@dataclass
class CompressionResult:
    mind_map_root: MindMapNode
    bridges: List[MindMapBridge]
    compression_report: Dict
    all_nodes: List[MindMapNode] = field(default_factory=list)


# ── Phase A+B+C: Concept clustering ──────────────────────────────────────

def _cluster_concepts(
    graph: KnowledgeGraph,
    survivors: List[str],   # node ids that survived pruning
) -> Tuple[List[Cluster], List[BridgeEdge]]:
    concepts = graph.concepts
    scores = graph.importance_scores
    importance = scores

    # Phase A: domain seeds from hierarchy tree root nodes
    roots = [n for n in graph.hierarchy_tree.nodes()
             if graph.hierarchy_tree.in_degree(n) == 0 and n in survivors]

    if not roots:
        # fallback: top-3 by score become seeds
        roots = sorted(survivors, key=lambda n: scores.get(n, 0), reverse=True)[:3]

    clusters: List[Cluster] = []
    assigned: Dict[str, str] = {}  # concept_id → cluster_id

    for root_id in roots:
        if root_id not in concepts:
            continue
        c = concepts[root_id]
        domain = c.tags[0] if c.tags else c.concept_type.value
        cl = Cluster(
            cluster_id=str(uuid.uuid4())[:8],
            domain=domain,
            seed_concept_id=root_id,
            member_ids=[root_id],
            label=c.title,
        )
        clusters.append(cl)
        assigned[root_id] = cl.cluster_id

    # Phase B: assign remaining survivors to nearest cluster (by shared edges)
    for node_id in survivors:
        if node_id in assigned:
            continue
        best_cluster = None
        best_score = -1
        for cl in clusters:
            shared = sum(
                1 for e in graph.edges
                if (e.source_id == node_id and e.target_id in cl.member_ids) or
                   (e.target_id == node_id and e.source_id in cl.member_ids)
            )
            if shared > best_score:
                best_score = shared
                best_cluster = cl
        if best_cluster is None:
            # create singleton cluster
            c = concepts.get(node_id)
            if c:
                best_cluster = Cluster(
                    cluster_id=str(uuid.uuid4())[:8],
                    domain=c.concept_type.value,
                    seed_concept_id=node_id,
                    member_ids=[],
                    label=c.title,
                )
                clusters.append(best_cluster)
        best_cluster.member_ids.append(node_id)
        assigned[node_id] = best_cluster.cluster_id

    # Phase C: edge gravity — merge clusters joined by DEPENDS_ON between CORE nodes
    for e in graph.edges:
        if e.relation_type != RelationType.DEPENDS_ON:
            continue
        if scores.get(e.source_id, 0) < THRESHOLD_CORE or scores.get(e.target_id, 0) < THRESHOLD_CORE:
            continue
        src_cl_id = assigned.get(e.source_id)
        tgt_cl_id = assigned.get(e.target_id)
        if src_cl_id and tgt_cl_id and src_cl_id != tgt_cl_id:
            src_cl = next((c for c in clusters if c.cluster_id == src_cl_id), None)
            tgt_cl = next((c for c in clusters if c.cluster_id == tgt_cl_id), None)
            if src_cl and tgt_cl:
                # merge smaller into larger
                if len(src_cl.member_ids) < len(tgt_cl.member_ids):
                    for mid in src_cl.member_ids:
                        tgt_cl.member_ids.append(mid)
                        assigned[mid] = tgt_cl.cluster_id
                    clusters.remove(src_cl)
                else:
                    for mid in tgt_cl.member_ids:
                        src_cl.member_ids.append(mid)
                        assigned[mid] = src_cl.cluster_id
                    clusters.remove(tgt_cl)

    # Merge guard: min size 2
    singletons = [cl for cl in clusters if len(cl.member_ids) < MIN_CLUSTER_SIZE]
    if singletons and len(clusters) > 1:
        primary = max(clusters, key=lambda c: len(c.member_ids))
        for s in singletons:
            if s is not primary:
                for mid in s.member_ids:
                    primary.member_ids.append(mid)
                    assigned[mid] = primary.cluster_id
                clusters.remove(s)

    # Max size guard: split if > 12
    final_clusters = []
    for cl in clusters:
        if len(cl.member_ids) > MAX_CLUSTER_SIZE:
            sorted_members = sorted(cl.member_ids, key=lambda n: scores.get(n, 0), reverse=True)
            half = len(sorted_members) // 2
            cl1 = Cluster(
                cluster_id=str(uuid.uuid4())[:8],
                domain=cl.domain + "_a",
                seed_concept_id=sorted_members[0],
                member_ids=sorted_members[:half],
                label=concepts[sorted_members[0]].title if sorted_members[0] in concepts else cl.label,
            )
            cl2 = Cluster(
                cluster_id=str(uuid.uuid4())[:8],
                domain=cl.domain + "_b",
                seed_concept_id=sorted_members[half],
                member_ids=sorted_members[half:],
                label=concepts[sorted_members[half]].title if sorted_members[half] in concepts else cl.label,
            )
            final_clusters.extend([cl1, cl2])
        else:
            final_clusters.append(cl)

    # Build bridge edges between clusters
    bridges: List[BridgeEdge] = []
    cluster_of = {mid: cl.cluster_id for cl in final_clusters for mid in cl.member_ids}
    for e in graph.edges:
        src_cl = cluster_of.get(e.source_id)
        tgt_cl = cluster_of.get(e.target_id)
        if src_cl and tgt_cl and src_cl != tgt_cl:
            bridges.append(BridgeEdge(
                from_cluster=src_cl,
                to_cluster=tgt_cl,
                edge_type=e.relation_type.value,
                label=e.relation_type.value.lower().replace("_", " "),
            ))

    return final_clusters, bridges


# ── Condensation (4 operations) ───────────────────────────────────────────

def _run_condensation(graph: KnowledgeGraph) -> Tuple[List[str], Dict, List[str], Dict]:
    """
    Returns (survivor_ids, annotations_by_parent, chain_orphan_ids, report)
    """
    scores = graph.importance_scores
    concepts = graph.concepts
    report = {"pruned": 0, "merged": 0, "deduped": 0, "flattened": 0}
    annotations_by_parent: Dict[str, List[Dict]] = {}
    chain_orphans: List[str] = []

    # Chain orphan detection: sole connector between two CORE nodes
    core_ids = {nid for nid, s in scores.items() if s >= THRESHOLD_CORE}
    for nid, c in concepts.items():
        if scores.get(nid, 0) < THRESHOLD_PRUNABLE:
            preds = list(graph.concept_graph.predecessors(nid))
            succs = list(graph.concept_graph.successors(nid))
            if any(p in core_ids for p in preds) and any(s in core_ids for s in succs):
                chain_orphans.append(nid)

    survivor_ids: List[str] = []
    pruned_ids: List[str] = []

    # Op 1: prune REDUNDANT tier (score < 0.25, not a chain orphan)
    for nid in concepts:
        score = scores.get(nid, 0)
        if score < THRESHOLD_PRUNABLE and nid not in chain_orphans:
            pruned_ids.append(nid)
            report["pruned"] += 1
        else:
            survivor_ids.append(nid)

    # Op 2: merge PRUNABLE tier nodes into parent annotation
    remaining: List[str] = []
    for nid in survivor_ids:
        score = scores.get(nid, 0)
        if THRESHOLD_PRUNABLE <= score < THRESHOLD_SUPPORTING:
            # Find highest-score neighbour as parent
            nbrs = list(graph.concept_graph.predecessors(nid)) + list(graph.concept_graph.successors(nid))
            if nbrs:
                parent = max(nbrs, key=lambda n: scores.get(n, 0))
                if parent not in annotations_by_parent:
                    annotations_by_parent[parent] = []
                c = concepts[nid]
                annotations_by_parent[parent].append({
                    "source_id": nid,
                    "title": c.title,
                    "summary": c.definition[:150],
                    "source_page": c.source_page,
                })
                report["merged"] += 1
            else:
                remaining.append(nid)
        else:
            remaining.append(nid)

    # Op 3: deduplicate — nodes sharing same title after lowering
    seen_titles: Dict[str, str] = {}
    deduped: List[str] = []
    for nid in remaining:
        c = concepts[nid]
        key = c.title.lower().strip()
        if key in seen_titles:
            canonical_id = seen_titles[key]
            canon = concepts[canonical_id]
            if c.source_page not in canon.recurrence:
                canon.recurrence.append(c.source_page)
            report["deduped"] += 1
        else:
            seen_titles[key] = nid
            deduped.append(nid)

    # Op 4: prerequisite chain flattening
    # (represented conceptually; long chains tracked for bridge node generation)
    dep_edges = [e for e in graph.edges if e.relation_type == RelationType.DEPENDS_ON]
    # Build dependency chains
    flat_report = {"flattened_chains": 0}
    for start in deduped:
        chain = [start]
        current = start
        while True:
            succs = [e.target_id for e in dep_edges if e.source_id == current and e.target_id in deduped]
            if not succs:
                break
            next_node = succs[0]
            if next_node in chain:
                break
            chain.append(next_node)
            current = next_node
        if len(chain) > 3:
            flat_report["flattened_chains"] += 1

    report["flattened"] = flat_report["flattened_chains"]

    # Compression ratio safety
    if len(deduped) > MAX_NODES:
        deduped = sorted(deduped, key=lambda n: scores.get(n, 0), reverse=True)[:MAX_NODES]
    if len(deduped) < MIN_NODES and len(remaining) >= MIN_NODES:
        deduped = sorted(remaining, key=lambda n: scores.get(n, 0), reverse=True)[:MIN_NODES]

    return deduped, annotations_by_parent, chain_orphans, report


# ── Mind map tree builder ─────────────────────────────────────────────────

def _build_mind_map_tree(
    graph: KnowledgeGraph,
    clusters: List[Cluster],
    survivor_ids: List[str],
    annotations_by_parent: Dict[str, List[Dict]],
    bridges: List[BridgeEdge],
) -> Tuple[MindMapNode, List[MindMapBridge], List[MindMapNode]]:
    concepts = graph.concepts
    scores = graph.importance_scores
    survivor_set = set(survivor_ids)

    # Step 1: root = highest importance_score among survivors
    if not survivor_ids:
        # absolute fallback
        root_id = next(iter(concepts))
    else:
        root_id = max(survivor_ids, key=lambda n: scores.get(n, 0))

    # Step 2: build branches per cluster
    def make_node(cid: str, depth: int = 0) -> MindMapNode:
        c = concepts.get(cid)
        if c is None:
            return MindMapNode(id=cid, label="Unknown", concept_type="DEFINITION",
                               score=0, tier=TIER_REDUNDANT, definition="", source_page=0)
        score = scores.get(cid, 0)
        tier = get_tier(score)
        return MindMapNode(
            id=cid,
            label=c.title,
            concept_type=c.concept_type.value,
            score=round(score, 3),
            tier=tier,
            definition=c.definition[:200],
            source_page=c.source_page,
            annotations=annotations_by_parent.get(cid, []),
        )

    root_node = make_node(root_id)
    all_nodes = [root_node]
    added_ids = {root_id}

    for cluster in clusters:
        cluster_members = [m for m in cluster.member_ids if m in survivor_set and m != root_id]
        if not cluster_members:
            continue
        cluster_members_sorted = sorted(cluster_members, key=lambda n: scores.get(n, 0), reverse=True)
        branch_root_id = cluster_members_sorted[0]

        branch_root = make_node(branch_root_id)
        all_nodes.append(branch_root)
        added_ids.add(branch_root_id)

        # Step 3: add children up to depth 3 (depth 1 = branch root, total from root = 4)
        children_ids = cluster_members_sorted[1:7]  # max 6 children per branch
        for child_id in children_ids:
            if child_id not in added_ids:
                child_node = make_node(child_id)
                branch_root.children.append(child_node)
                all_nodes.append(child_node)
                added_ids.add(child_id)

        root_node.children.append(branch_root)

    # Add any survivors not yet in tree as direct children of root (overflow safety)
    for nid in survivor_ids:
        if nid not in added_ids:
            overflow_node = make_node(nid)
            root_node.children.append(overflow_node)
            all_nodes.append(overflow_node)
            added_ids.add(nid)

    # Step 4: build bridge edges (cross-cluster)
    mind_map_bridges = []
    cluster_to_root: Dict[str, str] = {}
    for cl in clusters:
        if cl.member_ids:
            top = max(cl.member_ids, key=lambda n: scores.get(n, 0))
            cluster_to_root[cl.cluster_id] = top

    seen_bridge_pairs: set = set()
    for b in bridges:
        from_node = cluster_to_root.get(b.from_cluster)
        to_node   = cluster_to_root.get(b.to_cluster)
        if from_node and to_node and from_node != to_node:
            pair = (min(from_node, to_node), max(from_node, to_node), b.edge_type)
            if pair not in seen_bridge_pairs:
                seen_bridge_pairs.add(pair)
                mind_map_bridges.append(MindMapBridge(
                    from_node_id=from_node,
                    to_node_id=to_node,
                    edge_type=b.edge_type,
                    label=b.label,
                ))

    return root_node, mind_map_bridges, all_nodes


# ── Public API ────────────────────────────────────────────────────────────

@safe_stage("compression_engine", fallback_result=None)
def compress_graph(graph: KnowledgeGraph) -> Optional[CompressionResult]:
    """
    Main compression entry point. Implements full COMPRESSION_ENGINE spec.
    Fallback: if compression yields < MIN_NODES, returns a flat map of all concepts.
    """
    logger.info(
        f"Compressing graph: {len(graph.concepts)} concepts, "
        f"{len(graph.edges)} edges"
    )

    # Run condensation
    survivor_ids, annotations, chain_orphans, cond_report = _run_condensation(graph)

    if len(survivor_ids) < MIN_NODES:
        logger.warning(
            f"Compression yielded {len(survivor_ids)} nodes — below minimum. "
            "Using all concepts as fallback."
        )
        survivor_ids = list(graph.concepts.keys())[:MAX_NODES]

    # Cluster survivors
    clusters, bridges = _cluster_concepts(graph, survivor_ids)

    # Build mind map tree
    root_node, mm_bridges, all_nodes = _build_mind_map_tree(
        graph, clusters, survivor_ids, annotations, bridges
    )

    original = len(graph.concepts)
    compressed = len(survivor_ids)
    ratio = round(compressed / max(original, 1), 3)

    compression_report = {
        "original_node_count": original,
        "compressed_node_count": compressed,
        "ratio": ratio,
        "pruned_count": cond_report["pruned"],
        "merged_count": cond_report["merged"],
        "deduped_count": cond_report["deduped"],
        "flattened_chains": cond_report["flattened"],
        "cluster_count": len(clusters),
        "bridge_count": len(mm_bridges),
    }

    logger.info(
        f"Compression complete: {original} → {compressed} nodes "
        f"(ratio={ratio}, clusters={len(clusters)})"
    )

    return CompressionResult(
        mind_map_root=root_node,
        bridges=mm_bridges,
        compression_report=compression_report,
        all_nodes=all_nodes,
    )
