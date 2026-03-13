"""PDF export: renders the mind map as a structured PDF report using fpdf2."""

from __future__ import annotations
import io
from typing import Optional
from modules.compression_engine import CompressionResult, MindMapNode
from utils.logger import get_logger

logger = get_logger("pdf_exporter")


def export_to_pdf(result: CompressionResult, title: str = "Knowledge Map") -> bytes:
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("fpdf2 not installed — generating plain text PDF stub")
        return _text_stub(result, title)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, title, ln=True, align="C")
    pdf.ln(4)

    report = result.compression_report
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0, 8,
        f"Concepts: {report.get('compressed_node_count', '?')}  |  "
        f"Compression ratio: {report.get('ratio', 0):.0%}  |  "
        f"Clusters: {report.get('cluster_count', '?')}",
        ln=True, align="C",
    )
    pdf.ln(6)

    def write_node(node: MindMapNode, depth: int):
        if depth > 4:
            return
        indent = depth * 8
        if depth == 0:
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(83, 74, 183)  # purple
        elif depth == 1:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(29, 158, 117)  # teal
        else:
            pdf.set_font("Helvetica", size=10)
            pdf.set_text_color(60, 60, 60)

        bullet = ["◉", "▸", "–", "·"][min(depth, 3)]
        label = f"{bullet}  {node.label}"
        pdf.set_x(10 + indent)
        pdf.cell(0, 8, label, ln=True)

        if node.definition and depth <= 2:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(130, 130, 130)
            pdf.set_x(14 + indent)
            safe_def = node.definition[:120].replace("\n", " ")
            pdf.cell(0, 6, safe_def, ln=True)

        for child in node.children:
            write_node(child, depth + 1)

    write_node(result.mind_map_root, 0)

    if result.bridges:
        pdf.ln(8)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, "Cross-references", ln=True)
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(100, 100, 100)
        for b in result.bridges[:20]:
            pdf.cell(0, 6, f"  {b.from_node_id[:16]} ↔ {b.to_node_id[:16]}  [{b.edge_type}]", ln=True)

    return bytes(pdf.output())


def _text_stub(result: CompressionResult, title: str) -> bytes:
    lines = [title, "=" * len(title), ""]
    def walk(node: MindMapNode, depth: int):
        lines.append("  " * depth + f"• {node.label} (score={node.score:.2f})")
        for child in node.children:
            walk(child, depth + 1)
    walk(result.mind_map_root, 0)
    return "\n".join(lines).encode("utf-8")
