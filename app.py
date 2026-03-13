"""
Knowledge Mapper — Streamlit Application
Orchestrates the full pipeline: PDF → Concepts → Graph → Compression → Mind Map
"""

import io
import json
import time
import traceback
from typing import Optional

import streamlit as st
import networkx as nx

from modules.pdf_ingestion import ingest_pdf, IngestedDocument
from modules.text_segmentation import segment_document, SegmentationResult
from modules.concept_extraction import extract_concepts, ExtractionResult
from modules.knowledge_graph import build_knowledge_graph, KnowledgeGraph
from modules.compression_engine import compress_graph, CompressionResult
from modules.mindmap_generator import (
    generate_pyvis_html, generate_markdown_outline,
    generate_json, get_networkx_layout,
)
from export.pdf_exporter import export_to_pdf
from export.html_exporter import export_to_html
from export.json_exporter import export_to_json
from utils.logger import get_logger

logger = get_logger("app")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Mapper",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #fafaf9; }
.km-header { font-size: 1.7rem; font-weight: 500; color: #2C2C2A; margin: 0; }
.km-sub    { font-size: 0.9rem; color: #888780; margin-top: 2px; }
.metric-card {
    background: #f1efe8; border-radius: 8px; padding: 12px 16px;
    text-align: center; border: 0.5px solid #D3D1C7;
}
.metric-value { font-size: 1.6rem; font-weight: 500; color: #2C2C2A; }
.metric-label { font-size: 0.75rem; color: #888780; margin-top: 2px; }
.stage-ok   { color: #1D9E75; font-size: 0.8rem; }
.stage-fail { color: #E24B4A; font-size: 0.8rem; }
.stage-run  { color: #BA7517; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ───────────────────────────────────────────
for key in ["doc", "segments", "extraction", "graph", "compression", "stage_log"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "stage_log" not in st.session_state or st.session_state.stage_log is None:
    st.session_state.stage_log = []


# ── Helpers ────────────────────────────────────────────────────────────────

def log_stage(name: str, status: str, detail: str = ""):
    st.session_state.stage_log.append({"stage": name, "status": status, "detail": detail})


def reset_state():
    for key in ["doc", "segments", "extraction", "graph", "compression", "stage_log"]:
        st.session_state[key] = None
    st.session_state.stage_log = []


# ── Pipeline orchestrator ──────────────────────────────────────────────────

def run_pipeline(file_bytes: bytes, filename: str, progress_bar, status_text) -> bool:
    """
    Run all 8 stages with self-healing retries and fallbacks.
    Returns True on success, False on unrecoverable failure.
    """
    total_stages = 6
    stage_idx = 0

    def advance(label: str):
        nonlocal stage_idx
        stage_idx += 1
        progress_bar.progress(stage_idx / total_stages)
        status_text.markdown(f"<span class='stage-run'>⟳ {label}…</span>", unsafe_allow_html=True)

    # Stage 1+2: Ingestion
    advance("Ingesting PDF and extracting text")
    try:
        doc: IngestedDocument = ingest_pdf(file_bytes, filename)
        st.session_state.doc = doc
        log_stage("PDF Ingestion", "ok", f"{doc.total_pages} pages, {len(doc.chunks)} chunks")
    except Exception as exc:
        log_stage("PDF Ingestion", "fail", str(exc))
        status_text.markdown("<span class='stage-fail'>✗ PDF ingestion failed</span>", unsafe_allow_html=True)
        return False

    # Stage 3: Segmentation
    advance("Segmenting document structure")
    try:
        segments: SegmentationResult = segment_document(doc)
        if segments is None:
            raise ValueError("Segmentation returned None")
        st.session_state.segments = segments
        log_stage("Segmentation", "ok",
                  f"{len(segments.segments)} segments, {segments.chapter_count} chapters")
    except Exception as exc:
        # Self-heal: create one segment per page as fallback
        log_stage("Segmentation", "warn", f"Fallback: page-level segments ({exc})")
        from modules.text_segmentation import Segment, SegmentationResult as SR
        fallback_segs = []
        for page in doc.all_pages:
            if page.text.strip():
                fallback_segs.append(Segment(
                    segment_id=f"pg_{page.page_number}",
                    segment_type="section",
                    title=f"Page {page.page_number}",
                    text=page.text,
                    start_page=page.page_number,
                    end_page=page.page_number,
                    depth=0,
                ))
        segments = SR(segments=fallback_segs, chapter_count=0,
                      section_count=len(fallback_segs), total_words=sum(len(s.text.split()) for s in fallback_segs))
        st.session_state.segments = segments

    # Stage 4+5: Concept extraction
    advance("Extracting concepts and relationships")
    try:
        extraction: ExtractionResult = extract_concepts(segments)
        if extraction is None or not extraction.concepts:
            raise ValueError("No concepts extracted")
        st.session_state.extraction = extraction
        log_stage("Concept Extraction", "ok",
                  f"{len(extraction.concepts)} concepts, {len(extraction.edges)} edges")
    except Exception as exc:
        log_stage("Concept Extraction", "fail", str(exc))
        status_text.markdown("<span class='stage-fail'>✗ Concept extraction failed</span>", unsafe_allow_html=True)
        return False

    # Stage 6+7: Graph construction
    advance("Building knowledge graph")
    try:
        graph: KnowledgeGraph = build_knowledge_graph(extraction)
        if graph is None:
            raise GraphError("Graph returned None")
        st.session_state.graph = graph
        repairs = graph.validation_report.get("repairs", 0)
        log_stage("Knowledge Graph", "ok",
                  f"{graph.concept_graph.number_of_nodes()} nodes, {repairs} repairs")
    except Exception as exc:
        log_stage("Knowledge Graph", "fail", str(exc))
        status_text.markdown("<span class='stage-fail'>✗ Graph construction failed</span>", unsafe_allow_html=True)
        return False

    # Compression Engine
    advance("Compressing graph")
    try:
        compression: CompressionResult = compress_graph(graph)
        if compression is None:
            raise ValueError("Compression returned None")
        st.session_state.compression = compression
        r = compression.compression_report
        log_stage("Compression", "ok",
                  f"{r['original_node_count']} → {r['compressed_node_count']} nodes "
                  f"({r['ratio']:.0%})")
    except Exception as exc:
        log_stage("Compression", "fail", str(exc))
        status_text.markdown("<span class='stage-fail'>✗ Compression failed</span>", unsafe_allow_html=True)
        return False

    # Stage 8: Mind map generation
    advance("Generating mind map")
    try:
        # Pre-generate outputs and cache in session state
        html = generate_pyvis_html(compression)
        st.session_state["mm_html"] = html
        st.session_state["mm_json"] = generate_json(compression)
        st.session_state["mm_md"]   = generate_markdown_outline(compression)
        log_stage("Mind Map Generation", "ok", "HTML + JSON + Markdown ready")
    except Exception as exc:
        log_stage("Mind Map Generation", "warn", f"Partial output ({exc})")

    progress_bar.progress(1.0)
    status_text.markdown("<span class='stage-ok'>✓ Pipeline complete</span>", unsafe_allow_html=True)
    return True


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Upload")
    uploaded = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Supports native PDFs and scanned pages (OCR fallback automatic)",
    )

    st.divider()
    st.markdown("### Options")
    compression_hint = st.slider(
        "Compression level", min_value=1, max_value=5, value=3,
        help="Higher = fewer nodes in the mind map"
    )
    show_low_conf = st.checkbox("Show low-confidence concepts", value=False)

    st.divider()
    if st.button("Reset", use_container_width=True):
        reset_state()
        st.rerun()

    st.divider()
    st.markdown("### Pipeline log")
    if st.session_state.stage_log:
        for entry in st.session_state.stage_log:
            icon = {"ok": "✓", "warn": "⚠", "fail": "✗", "run": "⟳"}.get(entry["status"], "·")
            color = {"ok": "stage-ok", "warn": "stage-run", "fail": "stage-fail"}.get(
                entry["status"], "stage-ok"
            )
            st.markdown(
                f"<span class='{color}'>{icon} <b>{entry['stage']}</b></span><br/>"
                f"<span style='font-size:0.73rem;color:#888780;padding-left:14px'>"
                f"{entry['detail']}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<span style='color:#888780;font-size:0.8rem'>No pipeline run yet</span>",
                    unsafe_allow_html=True)


# ── Main header ────────────────────────────────────────────────────────────

st.markdown(
    "<p class='km-header'>◉ Knowledge Mapper</p>"
    "<p class='km-sub'>PDF → Concepts → Graph → Compressed Mind Map</p>",
    unsafe_allow_html=True,
)
st.divider()


# ── Upload + Run ────────────────────────────────────────────────────────────

if uploaded is not None and st.session_state.compression is None:
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown(f"**{uploaded.name}** · {uploaded.size:,} bytes")
    with col_btn:
        run_btn = st.button("Process book ↗", type="primary", use_container_width=True)

    if run_btn:
        reset_state()
        file_bytes = uploaded.read()
        prog = st.progress(0)
        status = st.empty()
        t0 = time.time()
        success = run_pipeline(file_bytes, uploaded.name, prog, status)
        elapsed = time.time() - t0
        if success:
            st.success(f"Processed in {elapsed:.1f}s — scroll down to explore the map.")
        else:
            st.error("Pipeline failed. Check the log in the sidebar for details.")

elif uploaded is None and st.session_state.compression is None:
    st.info("Upload a PDF in the sidebar to begin.")


# ── Results ────────────────────────────────────────────────────────────────

compression: Optional[CompressionResult] = st.session_state.compression
graph: Optional[KnowledgeGraph] = st.session_state.graph
extraction: Optional[ExtractionResult] = st.session_state.extraction

if compression and graph and extraction:

    # ── Metrics ──────────────────────────────────────────────────────────
    r = compression.compression_report
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, val, label in [
        (m1, graph.concept_graph.number_of_nodes(), "Concepts found"),
        (m2, r.get("compressed_node_count", "?"),   "In mind map"),
        (m3, f"{r.get('ratio', 0):.0%}",            "Compression"),
        (m4, r.get("cluster_count", "?"),            "Clusters"),
        (m5, len(extraction.edges),                  "Relationships"),
    ]:
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-label'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Tab layout ────────────────────────────────────────────────────────
    tab_map, tab_graph, tab_concepts, tab_export = st.tabs([
        "Mind map", "Concept graph", "Concept list", "Export",
    ])

    # ── Tab 1: Interactive Mind Map ────────────────────────────────────────
    with tab_map:
        mm_html = st.session_state.get("mm_html", "")
        if mm_html and len(mm_html) > 100:
            st.components.v1.html(mm_html, height=640, scrolling=False)
        else:
            st.warning("Interactive map not available — showing outline.")
            md = st.session_state.get("mm_md", "")
            st.markdown(md or "No outline generated.")

        with st.expander("Markdown outline"):
            md = st.session_state.get("mm_md", "")
            st.text_area("Outline", md, height=300, label_visibility="collapsed")

    # ── Tab 2: NetworkX concept graph ─────────────────────────────────────
    with tab_graph:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            G, pos = get_networkx_layout(compression)
            fig, ax = plt.subplots(figsize=(12, 7))
            fig.patch.set_facecolor("#fafaf9")
            ax.set_facecolor("#fafaf9")

            node_colors = []
            node_sizes  = []
            for nid in G.nodes():
                data = G.nodes[nid]
                tier = data.get("tier", "PRUNABLE")
                score = data.get("score", 0.3)
                color_map = {
                    "CORE":       "#534AB7",
                    "SUPPORTING": "#1D9E75",
                    "PRUNABLE":   "#888780",
                }
                node_colors.append(color_map.get(tier, "#888780"))
                node_sizes.append(200 + int(score * 600))

            labels = {n: G.nodes[n].get("label", n)[:18] for n in G.nodes()}
            nx.draw_networkx(
                G, pos=pos, ax=ax, labels=labels,
                node_color=node_colors, node_size=node_sizes,
                font_size=8, font_color="#2C2C2A", font_weight="normal",
                edge_color="#D3D1C7", width=0.8, arrows=True,
                arrowsize=12, connectionstyle="arc3,rad=0.1",
            )
            ax.set_axis_off()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except ImportError:
            st.info("Install matplotlib for static graph view: `pip install matplotlib`")
        except Exception as exc:
            st.error(f"Graph render error: {exc}")

        # Stats
        with st.expander("Validation report"):
            vr = graph.validation_report
            st.json(vr)

    # ── Tab 3: Concept list ──────────────────────────────────────────────
    with tab_concepts:
        concepts = graph.concepts
        scores   = graph.importance_scores
        tags     = graph.insight_tags

        rows = []
        for cid, c in concepts.items():
            score = scores.get(cid, 0)
            if not show_low_conf and c.low_confidence:
                continue
            from modules.knowledge_graph import get_tier
            rows.append({
                "Title":      c.title,
                "Type":       c.concept_type.value,
                "Score":      round(score, 3),
                "Tier":       get_tier(score),
                "Tag":        tags.get(cid, ""),
                "Page":       c.source_page,
                "Definition": c.definition[:120],
            })

        rows.sort(key=lambda x: x["Score"], reverse=True)

        # Filter controls
        col_type, col_tier = st.columns(2)
        with col_type:
            type_filter = st.multiselect(
                "Filter by type",
                options=list({r["Type"] for r in rows}),
                default=[],
            )
        with col_tier:
            tier_filter = st.multiselect(
                "Filter by tier",
                options=["CORE", "SUPPORTING", "PRUNABLE", "REDUNDANT"],
                default=["CORE", "SUPPORTING"],
            )

        filtered = [
            r for r in rows
            if (not type_filter or r["Type"] in type_filter)
            and (not tier_filter or r["Tier"] in tier_filter)
        ]

        st.caption(f"Showing {len(filtered)} of {len(rows)} concepts")
        if filtered:
            st.dataframe(
                filtered,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
                    "Definition": st.column_config.TextColumn("Definition", width="large"),
                },
                hide_index=True,
            )

    # ── Tab 4: Export ─────────────────────────────────────────────────────
    with tab_export:
        st.markdown("Download the knowledge map in your preferred format.")
        col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)

        with col_dl1:
            try:
                pdf_bytes = export_to_pdf(compression, title=st.session_state.doc.filename if st.session_state.doc else "Knowledge Map")
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name="knowledge_map.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF export error: {e}")

        with col_dl2:
            try:
                html_str = export_to_html(compression)
                st.download_button(
                    label="Download HTML",
                    data=html_str.encode("utf-8"),
                    file_name="knowledge_map.html",
                    mime="text/html",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"HTML export error: {e}")

        with col_dl3:
            try:
                json_str = export_to_json(compression)
                st.download_button(
                    label="Download JSON",
                    data=json_str.encode("utf-8"),
                    file_name="knowledge_map.json",
                    mime="application/json",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"JSON export error: {e}")

        with col_dl4:
            try:
                md_str = st.session_state.get("mm_md", "")
                st.download_button(
                    label="Download Markdown",
                    data=md_str.encode("utf-8"),
                    file_name="knowledge_map.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Markdown export error: {e}")

        st.divider()
        with st.expander("Compression report"):
            st.json(compression.compression_report)
