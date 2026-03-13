"""
pipeline.py — headless CLI runner for the Knowledge Mapper.

Usage:
    python pipeline.py input.pdf
    python pipeline.py input.pdf --output ./results --formats json html md
    python pipeline.py input.pdf --no-compress
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

from modules.pdf_ingestion import ingest_pdf
from modules.text_segmentation import segment_document
from modules.concept_extraction import extract_concepts
from modules.knowledge_graph import build_knowledge_graph
from modules.compression_engine import compress_graph
from modules.mindmap_generator import generate_all_outputs, generate_json
from export.pdf_exporter import export_to_pdf
from export.html_exporter import export_to_html
from export.json_exporter import export_to_json
from utils.logger import get_logger

logger = get_logger("pipeline")


def run(
    pdf_path: str,
    output_dir: str = "./output",
    formats: list[str] | None = None,
    compress: bool = True,
) -> dict:
    """
    Full headless pipeline. Returns summary dict.
    """
    if formats is None:
        formats = ["json", "html", "md"]

    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem

    logger.info(f"=== Knowledge Mapper pipeline starting: {pdf_path} ===")

    # Stage 1+2: Ingestion
    logger.info("Stage 1+2: PDF ingestion")
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    doc = ingest_pdf(file_bytes, Path(pdf_path).name)
    logger.info(f"  {doc.total_pages} pages, {len(doc.chunks)} chunks")

    # Stage 3: Segmentation
    logger.info("Stage 3: Segmentation")
    segments = segment_document(doc)
    if segments is None:
        logger.error("Segmentation failed")
        sys.exit(1)
    logger.info(f"  {len(segments.segments)} segments")

    # Stage 4+5: Extraction
    logger.info("Stage 4+5: Concept extraction")
    extraction = extract_concepts(segments)
    if extraction is None or not extraction.concepts:
        logger.error("Concept extraction failed")
        sys.exit(1)
    logger.info(f"  {len(extraction.concepts)} concepts, {len(extraction.edges)} edges")

    # Stage 6+7: Graph
    logger.info("Stage 6+7: Knowledge graph")
    graph = build_knowledge_graph(extraction)
    if graph is None:
        logger.error("Graph construction failed")
        sys.exit(1)
    repairs = graph.validation_report.get("repairs", 0)
    logger.info(f"  {graph.concept_graph.number_of_nodes()} nodes, {repairs} validation repairs")

    # Compression
    compression = None
    if compress:
        logger.info("Compression engine")
        compression = compress_graph(graph)
        if compression:
            r = compression.compression_report
            logger.info(
                f"  {r['original_node_count']} → {r['compressed_node_count']} nodes "
                f"(ratio={r['ratio']:.0%})"
            )

    # Stage 8: Outputs
    logger.info("Stage 8: Generating outputs")
    if compression:
        if "json" in formats:
            json_path = out / f"{stem}_knowledge_map.json"
            json_path.write_text(export_to_json(compression), encoding="utf-8")
            logger.info(f"  JSON → {json_path}")

        if "html" in formats:
            html_path = out / f"{stem}_knowledge_map.html"
            html_path.write_text(export_to_html(compression, title=stem), encoding="utf-8")
            logger.info(f"  HTML → {html_path}")

        if "pdf" in formats:
            pdf_out = out / f"{stem}_knowledge_map.pdf"
            pdf_out.write_bytes(export_to_pdf(compression, title=stem))
            logger.info(f"  PDF → {pdf_out}")

        if "md" in formats:
            from modules.mindmap_generator import generate_markdown_outline
            md_path = out / f"{stem}_knowledge_map.md"
            md_path.write_text(generate_markdown_outline(compression), encoding="utf-8")
            logger.info(f"  Markdown → {md_path}")

    elapsed = round(time.time() - t0, 2)
    summary = {
        "input": pdf_path,
        "pages": doc.total_pages,
        "concepts": len(extraction.concepts),
        "edges": len(extraction.edges),
        "compressed_nodes": compression.compression_report["compressed_node_count"] if compression else None,
        "elapsed_s": elapsed,
        "outputs": [str(p) for p in out.iterdir() if p.name.startswith(stem)],
    }
    logger.info(f"=== Done in {elapsed}s ===")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Knowledge Mapper — headless pipeline")
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument(
        "--formats", nargs="+", default=["json", "html", "md"],
        choices=["json", "html", "md", "pdf"],
        help="Output formats to generate",
    )
    parser.add_argument("--no-compress", action="store_true", help="Skip compression step")
    args = parser.parse_args()

    summary = run(
        pdf_path=args.pdf,
        output_dir=args.output,
        formats=args.formats,
        compress=not args.no_compress,
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
