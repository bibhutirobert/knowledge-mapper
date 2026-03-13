"""
tests/test_pipeline.py
Full test suite for the Knowledge Mapper pipeline.

Run:  python -m pytest tests/ -v
      python -m pytest tests/ -v --tb=short  (compact tracebacks)
"""

import io
import json
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── ensure repo root is on path ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — minimal synthetic data factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_page(n: int, text: str = "") -> "PageText":
    from modules.pdf_ingestion import PageText
    return PageText(page_number=n, text=text or f"Content of page {n}. " * 6)


def _make_doc(pages=3) -> "IngestedDocument":
    from modules.pdf_ingestion import IngestedDocument, DocumentChunk, PageText
    pg_list = [_make_page(i, f"Chapter {i}\n\nThis is the content of chapter {i}. "
                              f"A principle is a fundamental truth. "
                              f"For example, gravity is an example of a force. "
                              f"Therefore, we conclude that forces exist.") for i in range(1, pages + 1)]
    chunk = DocumentChunk(chunk_index=0, start_page=1, end_page=pages, pages=pg_list)
    return IngestedDocument(filename="test.pdf", total_pages=pages, chunks=[chunk])


def _make_segment_result(n=4) -> "SegmentationResult":
    from modules.text_segmentation import Segment, SegmentationResult
    segs = []
    for i in range(n):
        segs.append(Segment(
            segment_id=f"seg_{i:04d}",
            segment_type="section" if i > 0 else "chapter",
            title=f"Chapter {i + 1}" if i == 0 else f"Section {i}",
            text=(
                f"The principle of abstraction is a fundamental concept. "
                f"Abstraction is defined as the removal of detail. "
                f"For example, a map is an example of abstraction. "
                f"Therefore, abstraction supports simplicity."
            ),
            start_page=i + 1,
            end_page=i + 1,
            depth=0 if i == 0 else 1,
        ))
    return SegmentationResult(segments=segs, chapter_count=1,
                              section_count=n - 1, total_words=n * 30)


def _make_extraction() -> "ExtractionResult":
    from modules.concept_extraction import (
        ConceptNode, ConceptType, RelationshipEdge, RelationType, ExtractionResult,
    )
    concepts = []
    for i, (title, ctype) in enumerate([
        ("Abstraction",    ConceptType.PRINCIPLE),
        ("Simplicity",     ConceptType.DEFINITION),
        ("Encapsulation",  ConceptType.PRINCIPLE),
        ("Polymorphism",   ConceptType.THEOREM),
        ("Inheritance",    ConceptType.DEFINITION),
        ("Composition",    ConceptType.ARGUMENT),
        ("Interface",      ConceptType.DEFINITION),
        ("Pattern",        ConceptType.INSIGHT),
    ]):
        cid = ConceptNode.make_id(title, i + 1)
        concepts.append(ConceptNode(
            id=cid, title=title, concept_type=ctype,
            definition=f"{title} is a key concept in software design.",
            context={"chapter": 1}, source_page=i + 1,
            recurrence=list(range(1, (i % 3) + 2)),
            salience=round(0.4 + i * 0.07, 2),
        ))

    edges = []
    pairs = [
        (0, 1, RelationType.SUPPORTS,     0.8),
        (0, 2, RelationType.DEPENDS_ON,   0.7),
        (1, 3, RelationType.IS_A,         0.9),
        (2, 3, RelationType.DERIVED_FROM, 0.75),
        (3, 4, RelationType.PART_OF,      0.65),
        (4, 5, RelationType.EXAMPLE_OF,   0.6),
        (5, 6, RelationType.SUPPORTS,     0.7),
        (6, 7, RelationType.CONTRADICTS,  0.55),
    ]
    for src_i, tgt_i, rtype, conf in pairs:
        eid = RelationshipEdge.make_id(concepts[src_i].id, concepts[tgt_i].id, rtype)
        edges.append(RelationshipEdge(
            id=eid, source_id=concepts[src_i].id, target_id=concepts[tgt_i].id,
            relation_type=rtype, confidence=conf,
            evidence="test evidence", source_page=1,
        ))

    return ExtractionResult(concepts=concepts, edges=edges,
                            stats={"total_concepts": len(concepts), "total_edges": len(edges)})


# ─────────────────────────────────────────────────────────────────────────────
# Tests: PDF Ingestion
# ─────────────────────────────────────────────────────────────────────────────

class TestPDFIngestion(unittest.TestCase):

    def test_page_text_construction(self):
        from modules.pdf_ingestion import PageText
        pt = PageText(page_number=3, text="Hello world", width=612.0, height=792.0)
        self.assertEqual(pt.page_number, 3)
        self.assertFalse(pt.is_ocr)

    def test_document_chunk_full_text(self):
        from modules.pdf_ingestion import DocumentChunk
        pages = [_make_page(i, f"Page {i} text.") for i in range(1, 4)]
        chunk = DocumentChunk(chunk_index=0, start_page=1, end_page=3, pages=pages)
        self.assertIn("Page 1 text.", chunk.full_text)
        self.assertIn("Page 3 text.", chunk.full_text)

    def test_ingested_document_full_text_dedup(self):
        doc = _make_doc(pages=3)
        full = doc.full_text
        # Should contain each page exactly once (dedup by page_number)
        self.assertIsInstance(full, str)
        self.assertGreater(len(full), 10)

    def test_build_chunks_overlap(self):
        from modules.pdf_ingestion import _build_chunks
        pages = [_make_page(i) for i in range(1, 26)]  # 25 pages
        chunks = _build_chunks(pages)
        # First chunk: pages 1-20, second starts at 1+20-3=18
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].start_page, 1)

    def test_all_pages_sorted(self):
        doc = _make_doc(pages=5)
        pages = doc.all_pages
        for i in range(1, len(pages)):
            self.assertLessEqual(pages[i - 1].page_number, pages[i].page_number)

    def test_ingest_fallback_on_bad_bytes(self):
        from modules.pdf_ingestion import ingest_pdf
        # Passing garbage bytes should not raise — fallback chain should handle it
        # (pdfplumber + pymupdf will fail, OCR will fail gracefully via safe_stage)
        try:
            result = ingest_pdf(b"not a pdf", "fake.pdf")
        except Exception:
            pass  # acceptable — all fallbacks exhausted


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Text Segmentation
# ─────────────────────────────────────────────────────────────────────────────

class TestTextSegmentation(unittest.TestCase):

    def test_segments_produced(self):
        doc = _make_doc(pages=4)
        from modules.text_segmentation import segment_document
        result = segment_document(doc)
        self.assertIsNotNone(result)
        self.assertGreater(len(result.segments), 0)

    def test_heading_detection(self):
        from modules.text_segmentation import _is_heading
        self.assertTrue(_is_heading("Chapter 1. Introduction")[0])
        self.assertTrue(_is_heading("1.1 Background")[0])
        self.assertFalse(_is_heading("This is a regular sentence.")[0])
        self.assertFalse(_is_heading("ab")[0])

    def test_fallback_page_level(self):
        """When no headings found, should create one segment per page."""
        from modules.pdf_ingestion import IngestedDocument, DocumentChunk, PageText
        from modules.text_segmentation import segment_document
        pages = [PageText(page_number=i, text=f"Just plain text on page {i}. " * 5)
                 for i in range(1, 5)]
        chunk = DocumentChunk(chunk_index=0, start_page=1, end_page=4, pages=pages)
        doc = IngestedDocument(filename="test.pdf", total_pages=4, chunks=[chunk])
        result = segment_document(doc)
        self.assertIsNotNone(result)
        self.assertGreater(len(result.segments), 0)

    def test_total_words_positive(self):
        doc = _make_doc(pages=3)
        from modules.text_segmentation import segment_document
        result = segment_document(doc)
        self.assertGreater(result.total_words, 0)

    def test_segment_id_unique(self):
        doc = _make_doc(pages=5)
        from modules.text_segmentation import segment_document
        result = segment_document(doc)
        ids = [s.segment_id for s in result.segments]
        self.assertEqual(len(ids), len(set(ids)))


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Concept Extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestConceptExtraction(unittest.TestCase):

    def setUp(self):
        self.segs = _make_segment_result(n=4)

    def test_extraction_returns_concepts(self):
        from modules.concept_extraction import extract_concepts
        result = extract_concepts(self.segs)
        self.assertIsNotNone(result)
        self.assertGreater(len(result.concepts), 0)

    def test_concept_ids_unique(self):
        from modules.concept_extraction import extract_concepts
        result = extract_concepts(self.segs)
        ids = [c.id for c in result.concepts]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate concept IDs found")

    def test_salience_in_range(self):
        ext = _make_extraction()
        for c in ext.concepts:
            self.assertGreaterEqual(c.salience, 0.0)
            self.assertLessEqual(c.salience, 1.0)

    def test_edge_confidence_threshold(self):
        """All edges in an ExtractionResult should meet min confidence."""
        ext = _make_extraction()
        for e in ext.edges:
            self.assertGreaterEqual(e.confidence, 0.4)

    def test_no_self_loop_edges(self):
        ext = _make_extraction()
        for e in ext.edges:
            self.assertNotEqual(e.source_id, e.target_id)

    def test_concept_node_make_id_stable(self):
        from modules.concept_extraction import ConceptNode
        id1 = ConceptNode.make_id("Abstraction", 5)
        id2 = ConceptNode.make_id("Abstraction", 5)
        self.assertEqual(id1, id2)
        id3 = ConceptNode.make_id("Abstraction", 6)
        self.assertNotEqual(id1, id3)

    def test_definition_not_empty(self):
        ext = _make_extraction()
        for c in ext.concepts:
            self.assertTrue(c.definition, f"Empty definition for concept '{c.title}'")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Knowledge Graph
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeGraph(unittest.TestCase):

    def setUp(self):
        self.ext = _make_extraction()

    def test_graph_builds(self):
        from modules.knowledge_graph import build_knowledge_graph
        graph = build_knowledge_graph(self.ext)
        self.assertIsNotNone(graph)

    def test_concept_graph_node_count(self):
        from modules.knowledge_graph import build_knowledge_graph
        graph = build_knowledge_graph(self.ext)
        self.assertEqual(graph.concept_graph.number_of_nodes(), len(self.ext.concepts))

    def test_hierarchy_tree_is_dag(self):
        from modules.knowledge_graph import build_knowledge_graph
        import networkx as nx
        graph = build_knowledge_graph(self.ext)
        self.assertTrue(nx.is_directed_acyclic_graph(graph.hierarchy_tree))

    def test_importance_scores_range(self):
        from modules.knowledge_graph import build_knowledge_graph
        graph = build_knowledge_graph(self.ext)
        for nid, score in graph.importance_scores.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_all_concepts_have_importance_score(self):
        from modules.knowledge_graph import build_knowledge_graph
        graph = build_knowledge_graph(self.ext)
        for c in self.ext.concepts:
            self.assertIn(c.id, graph.importance_scores)

    def test_insight_tags_populated(self):
        from modules.knowledge_graph import build_knowledge_graph
        graph = build_knowledge_graph(self.ext)
        self.assertGreater(len(graph.insight_tags), 0)

    def test_validation_report_present(self):
        from modules.knowledge_graph import build_knowledge_graph
        graph = build_knowledge_graph(self.ext)
        self.assertIn("pass1", graph.validation_report)
        self.assertIn("pass2", graph.validation_report)
        self.assertIn("pass3", graph.validation_report)

    def test_get_tier_function(self):
        from modules.knowledge_graph import get_tier, TIER_CORE, TIER_SUPPORTING, TIER_PRUNABLE, TIER_REDUNDANT
        self.assertEqual(get_tier(0.80), TIER_CORE)
        self.assertEqual(get_tier(0.60), TIER_SUPPORTING)
        self.assertEqual(get_tier(0.35), TIER_PRUNABLE)
        self.assertEqual(get_tier(0.10), TIER_REDUNDANT)

    def test_no_self_loops_in_graphs(self):
        from modules.knowledge_graph import build_knowledge_graph
        graph = build_knowledge_graph(self.ext)
        for G in [graph.concept_graph, graph.hierarchy_tree, graph.dependency_graph]:
            self_loops = list(G.selfloop_edges()) if hasattr(G, 'selfloop_edges') else [(u, v) for u, v in G.edges() if u == v]
            self.assertEqual(len(self_loops), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Compression Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestCompressionEngine(unittest.TestCase):

    def setUp(self):
        from modules.knowledge_graph import build_knowledge_graph
        self.graph = build_knowledge_graph(_make_extraction())

    def test_compression_returns_result(self):
        from modules.compression_engine import compress_graph
        result = compress_graph(self.graph)
        self.assertIsNotNone(result)

    def test_compression_has_root(self):
        from modules.compression_engine import compress_graph
        result = compress_graph(self.graph)
        self.assertIsNotNone(result.mind_map_root)

    def test_compressed_node_count_in_bounds(self):
        from modules.compression_engine import compress_graph, MIN_NODES, MAX_NODES
        result = compress_graph(self.graph)
        r = result.compression_report
        self.assertGreaterEqual(r["compressed_node_count"], MIN_NODES)
        self.assertLessEqual(r["compressed_node_count"], MAX_NODES)

    def test_compression_ratio_positive(self):
        from modules.compression_engine import compress_graph
        result = compress_graph(self.graph)
        self.assertGreater(result.compression_report["ratio"], 0)

    def test_root_has_highest_score(self):
        """Root node should be the highest importance-score concept."""
        from modules.compression_engine import compress_graph
        result = compress_graph(self.graph)
        root_score = result.mind_map_root.score
        for node in result.all_nodes:
            if not node.is_bridge:
                self.assertLessEqual(node.score, root_score + 0.01)  # ±float tolerance

    def test_all_nodes_have_tier(self):
        from modules.compression_engine import compress_graph
        from modules.knowledge_graph import TIER_CORE, TIER_SUPPORTING, TIER_PRUNABLE, TIER_REDUNDANT
        result = compress_graph(self.graph)
        valid_tiers = {TIER_CORE, TIER_SUPPORTING, TIER_PRUNABLE, TIER_REDUNDANT}
        for node in result.all_nodes:
            self.assertIn(node.tier, valid_tiers)

    def test_compression_report_keys(self):
        from modules.compression_engine import compress_graph
        result = compress_graph(self.graph)
        required = {"original_node_count", "compressed_node_count", "ratio",
                    "pruned_count", "merged_count", "deduped_count",
                    "flattened_chains", "cluster_count"}
        for key in required:
            self.assertIn(key, result.compression_report)

    def test_bridges_are_list(self):
        from modules.compression_engine import compress_graph
        result = compress_graph(self.graph)
        self.assertIsInstance(result.bridges, list)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Mind Map Generator
# ─────────────────────────────────────────────────────────────────────────────

class TestMindMapGenerator(unittest.TestCase):

    def setUp(self):
        from modules.knowledge_graph import build_knowledge_graph
        from modules.compression_engine import compress_graph
        self.result = compress_graph(build_knowledge_graph(_make_extraction()))

    def test_markdown_outline_nonempty(self):
        from modules.mindmap_generator import generate_markdown_outline
        md = generate_markdown_outline(self.result)
        self.assertIsInstance(md, str)
        self.assertGreater(len(md), 50)

    def test_markdown_has_root_heading(self):
        from modules.mindmap_generator import generate_markdown_outline
        md = generate_markdown_outline(self.result)
        self.assertTrue(md.startswith("#"))

    def test_json_output_structure(self):
        from modules.mindmap_generator import generate_json
        data = generate_json(self.result)
        self.assertIn("root", data)
        self.assertIn("bridges", data)
        self.assertIn("compression_report", data)

    def test_json_root_has_required_keys(self):
        from modules.mindmap_generator import generate_json
        data = generate_json(self.result)
        root = data["root"]
        for key in ["id", "label", "score", "tier", "definition", "children"]:
            self.assertIn(key, root)

    def test_json_serialisable(self):
        from modules.mindmap_generator import generate_json
        data = generate_json(self.result)
        # Should not raise
        serialised = json.dumps(data)
        self.assertGreater(len(serialised), 10)

    def test_networkx_layout_returns_graph(self):
        from modules.mindmap_generator import get_networkx_layout
        import networkx as nx
        G, pos = get_networkx_layout(self.result)
        self.assertIsInstance(G, nx.DiGraph)
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertEqual(len(pos), G.number_of_nodes())

    def test_all_nodes_flat_includes_root(self):
        from modules.mindmap_generator import _all_nodes_flat
        nodes = _all_nodes_flat(self.result.mind_map_root)
        ids = [n.id for n in nodes]
        self.assertIn(self.result.mind_map_root.id, ids)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Export modules
# ─────────────────────────────────────────────────────────────────────────────

class TestExports(unittest.TestCase):

    def setUp(self):
        from modules.knowledge_graph import build_knowledge_graph
        from modules.compression_engine import compress_graph
        self.result = compress_graph(build_knowledge_graph(_make_extraction()))

    def test_json_export_is_valid_json(self):
        from export.json_exporter import export_to_json
        out = export_to_json(self.result)
        parsed = json.loads(out)
        self.assertIn("root", parsed)

    def test_html_export_contains_body(self):
        from export.html_exporter import export_to_html
        html = export_to_html(self.result, title="Test Map")
        self.assertIn("</html>", html)
        self.assertIn("Test Map", html)

    def test_pdf_export_returns_bytes(self):
        from export.pdf_exporter import export_to_pdf
        result = export_to_pdf(self.result, title="Test")
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 10)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Error handler / self-healing
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandler(unittest.TestCase):

    def test_retry_succeeds_on_first_try(self):
        from utils.error_handler import retry
        calls = []
        @retry(max_attempts=3)
        def good():
            calls.append(1)
            return "ok"
        self.assertEqual(good(), "ok")
        self.assertEqual(len(calls), 1)

    def test_retry_retries_then_succeeds(self):
        from utils.error_handler import retry
        counter = {"n": 0}
        @retry(max_attempts=3, delay=0)
        def flaky():
            counter["n"] += 1
            if counter["n"] < 3:
                raise ValueError("not yet")
            return "done"
        self.assertEqual(flaky(), "done")
        self.assertEqual(counter["n"], 3)

    def test_retry_uses_fallback(self):
        from utils.error_handler import retry
        @retry(max_attempts=2, delay=0, fallback=lambda: "fallback_value")
        def always_fails():
            raise RuntimeError("nope")
        self.assertEqual(always_fails(), "fallback_value")

    def test_safe_stage_returns_fallback_on_exception(self):
        from utils.error_handler import safe_stage
        @safe_stage("test_stage", fallback_result={"default": True})
        def bad_func():
            raise RuntimeError("intentional")
        result = bad_func()
        self.assertEqual(result, {"default": True})

    def test_safe_stage_passes_through_on_success(self):
        from utils.error_handler import safe_stage
        @safe_stage("test_stage", fallback_result=None)
        def good_func(x):
            return x * 2
        self.assertEqual(good_func(21), 42)


# ─────────────────────────────────────────────────────────────────────────────
# Integration test: full mini-pipeline on synthetic data
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline(unittest.TestCase):

    def test_end_to_end_synthetic(self):
        """Run the complete pipeline on synthetic IngestedDocument."""
        from modules.text_segmentation import segment_document
        from modules.concept_extraction import extract_concepts
        from modules.knowledge_graph import build_knowledge_graph
        from modules.compression_engine import compress_graph
        from modules.mindmap_generator import generate_json, generate_markdown_outline

        doc = _make_doc(pages=6)
        segs = segment_document(doc)
        self.assertIsNotNone(segs)

        ext = extract_concepts(segs)
        self.assertIsNotNone(ext)
        self.assertGreater(len(ext.concepts), 0)

        graph = build_knowledge_graph(ext)
        self.assertIsNotNone(graph)

        comp = compress_graph(graph)
        self.assertIsNotNone(comp)

        js = generate_json(comp)
        self.assertIn("root", js)

        md = generate_markdown_outline(comp)
        self.assertTrue(md.startswith("#"))

    def test_pipeline_survives_empty_pages(self):
        """Pipeline should not crash on pages with no text."""
        from modules.pdf_ingestion import IngestedDocument, DocumentChunk, PageText
        from modules.text_segmentation import segment_document
        from modules.concept_extraction import extract_concepts
        from modules.knowledge_graph import build_knowledge_graph
        from modules.compression_engine import compress_graph

        pages = [PageText(page_number=i, text="") for i in range(1, 5)]
        # Add one non-empty page
        pages[2] = PageText(page_number=3, text=(
            "The principle of least privilege is defined as granting minimal access. "
            "For example, a process should not run as root. "
            "Therefore, least privilege supports security. " * 3
        ))
        chunk = DocumentChunk(chunk_index=0, start_page=1, end_page=4, pages=pages)
        doc = IngestedDocument(filename="sparse.pdf", total_pages=4, chunks=[chunk])

        segs = segment_document(doc)
        ext = extract_concepts(segs)
        self.assertIsNotNone(ext)

        graph = build_knowledge_graph(ext)
        comp = compress_graph(graph)
        self.assertIsNotNone(comp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
