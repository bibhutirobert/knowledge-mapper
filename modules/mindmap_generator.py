"""
Stage 8: Mind map generation.
Converts CompressionResult into interactive visualisations using
pyvis (interactive HTML) and NetworkX (layout). Implements RESEARCH_LAYER Stage 8.
"""

from __future__ import annotations
import json
import textwrap
from typing import Dict, List, Optional, Tuple

from modules.compression_engine import (
    CompressionResult, MindMapNode, MindMapBridge,
)
from modules.knowledge_graph import TIER_CORE, TIER_SUPPORTING, TIER_PRUNABLE
from utils.error_handler import safe_stage
from utils.logger import get_logger

logger = get_logger("mindmap_generator")

# Visual tier colours (matches RESEARCH_LAYER tier taxonomy)
_TIER_COLORS = {
    TIER_CORE:       "#534AB7",   # purple-600
    TIER_SUPPORTING: "#1D9E75",   # teal-400
    TIER_PRUNABLE:   "#888780",   # gray-400
    "default":       "#888780",
}

_TYPE_SHAPE = {
    "THEOREM":    "diamond",
    "PRINCIPLE":  "dot",
    "CONCLUSION": "star",
    "DEFINITION": "dot",
    "ARGUMENT":   "dot",
    "EXAMPLE":    "square",
    "INSIGHT":    "dot",
}


def _all_nodes_flat(root: MindMapNode) -> List[MindMapNode]:
    """Flatten tree into ordered list (root first, BFS)."""
    result = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        result.append(node)
        queue.extend(node.children)
    return result


def _node_size(node: MindMapNode) -> int:
    base = {TIER_CORE: 30, TIER_SUPPORTING: 20}.get(node.tier, 12)
    boost = min(int(node.score * 20), 20)
    return base + boost


# ── pyvis interactive graph ───────────────────────────────────────────────

def generate_pyvis_html(result: CompressionResult) -> str:
    """Render an interactive pyvis network as HTML string."""
    try:
        from pyvis.network import Network
    except ImportError:
        logger.warning("pyvis not available — returning plain text summary")
        return _plain_text_fallback(result)

    net = Network(
        height="620px",
        width="100%",
        bgcolor="transparent",
        font_color="#2C2C2A",
        directed=True,
    )
    net.set_options(json.dumps({
        "nodes": {"font": {"size": 14, "face": "sans-serif"}},
        "edges": {"arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
                  "smooth": {"type": "cubicBezier"}},
        "physics": {
            "enabled": True,
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.01,
                "springLength": 120,
                "springConstant": 0.08,
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150},
        },
        "interaction": {"hover": True, "tooltipDelay": 200},
    }))

    all_nodes = _all_nodes_flat(result.mind_map_root)
    added_ids = set()

    for node in all_nodes:
        if node.id in added_ids:
            continue
        added_ids.add(node.id)
        color  = _TIER_COLORS.get(node.tier, _TIER_COLORS["default"])
        size   = _node_size(node)
        shape  = _TYPE_SHAPE.get(node.concept_type, "dot")
        tooltip = f"<b>{node.label}</b><br/>{node.definition[:120]}<br/><i>p.{node.source_page}</i>"
        net.add_node(
            node.id,
            label=textwrap.shorten(node.label, width=28, placeholder="…"),
            title=tooltip,
            color=color,
            size=size,
            shape=shape,
        )

    # Parent → child edges
    def add_edges(parent: MindMapNode):
        for child in parent.children:
            net.add_edge(parent.id, child.id, color="#B4B2A9", width=1.5)
            add_edges(child)

    add_edges(result.mind_map_root)

    # Cross-cluster bridges (dashed)
    for bridge in result.bridges:
        if bridge.from_node_id in added_ids and bridge.to_node_id in added_ids:
            net.add_edge(
                bridge.from_node_id,
                bridge.to_node_id,
                label=bridge.label[:20],
                color="#EF9F27",
                width=1,
                dashes=True,
            )

    return net.generate_html()


# ── Markdown outline ──────────────────────────────────────────────────────

def generate_markdown_outline(result: CompressionResult) -> str:
    """RESEARCH_LAYER MindMapGenerator markdown output."""
    lines = []

    def recurse(node: MindMapNode, depth: int):
        prefix = "#" * min(depth + 1, 4)
        score_str = f"(score: {node.score:.2f})"
        lines.append(f"{prefix} {node.label} {score_str}")
        if node.definition:
            lines.append(f"> {node.definition[:150]} — p.{node.source_page}")
        for ann in node.annotations:
            lines.append(f"  - *{ann['title']}*: {ann['summary'][:80]} — p.{ann['source_page']}")
        for child in node.children:
            recurse(child, depth + 1)

    recurse(result.mind_map_root, 0)

    if result.bridges:
        lines.append("\n---\n**Cross-references:**")
        for b in result.bridges:
            lines.append(f"- {b.from_node_id} ↔ {b.to_node_id} ({b.edge_type})")

    return "\n\n".join(lines)


# ── JSON export ───────────────────────────────────────────────────────────

def generate_json(result: CompressionResult) -> dict:
    """Serialisable dict for JSON export."""
    def node_to_dict(n: MindMapNode) -> dict:
        return {
            "id": n.id,
            "label": n.label,
            "concept_type": n.concept_type,
            "score": n.score,
            "tier": n.tier,
            "definition": n.definition,
            "source_page": n.source_page,
            "annotations": n.annotations,
            "children": [node_to_dict(c) for c in n.children],
        }
    return {
        "root": node_to_dict(result.mind_map_root),
        "bridges": [
            {"from": b.from_node_id, "to": b.to_node_id,
             "type": b.edge_type, "label": b.label}
            for b in result.bridges
        ],
        "compression_report": result.compression_report,
    }


# ── Plain text fallback ───────────────────────────────────────────────────

def _plain_text_fallback(result: CompressionResult) -> str:
    lines = ["<pre style='font-family:monospace;font-size:13px;'>"]
    lines.append("KNOWLEDGE MAP\n" + "=" * 50)

    def recurse(node: MindMapNode, indent: int):
        prefix = "  " * indent + ("• " if indent > 0 else "")
        lines.append(f"{prefix}{node.label} [{node.tier}] score={node.score:.2f}")
        for child in node.children:
            recurse(child, indent + 1)

    recurse(result.mind_map_root, 0)
    lines.append("</pre>")
    return "\n".join(lines)


# ── NetworkX layout for Streamlit static fallback ────────────────────────

def get_networkx_layout(result: CompressionResult):
    """Return (G, pos, node_attrs) for use with Streamlit matplotlib rendering."""
    import networkx as nx
    G = nx.DiGraph()
    all_nodes = _all_nodes_flat(result.mind_map_root)

    for node in all_nodes:
        G.add_node(node.id, label=node.label[:24], tier=node.tier,
                   score=node.score, concept_type=node.concept_type)

    def add_edges(parent: MindMapNode):
        for child in parent.children:
            G.add_edge(parent.id, child.id)
            add_edges(child)
    add_edges(result.mind_map_root)

    for bridge in result.bridges:
        if G.has_node(bridge.from_node_id) and G.has_node(bridge.to_node_id):
            G.add_edge(bridge.from_node_id, bridge.to_node_id, bridge=True)

    try:
        pos = nx.spring_layout(G, k=1.8, seed=42)
    except Exception:
        pos = nx.shell_layout(G)

    return G, pos


@safe_stage("mindmap_generator", fallback_result=("", {}, ""))
def generate_all_outputs(result: CompressionResult) -> Tuple[str, dict, str]:
    """Generate HTML, JSON, and Markdown from a CompressionResult."""
    html     = generate_pyvis_html(result)
    js_dict  = generate_json(result)
    markdown = generate_markdown_outline(result)
    return html, js_dict, markdown
