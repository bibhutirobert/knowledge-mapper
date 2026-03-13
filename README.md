# Knowledge Mapper

A research-grade system that converts PDF books into compressed knowledge graphs and interactive mind maps. Implements the RESEARCH_LAYER, SYSTEM_ARCHITECTURE, and COMPRESSION_ENGINE specifications.

---

## What it does

1. **Ingests** PDF books (native text or scanned) via a three-stage fallback pipeline: pdfplumber → PyMuPDF → Tesseract OCR
2. **Segments** the document into chapters, sections, and paragraphs
3. **Extracts** typed concept nodes (definitions, theorems, principles, arguments, conclusions, examples) and typed relationship edges (IS_A, DEPENDS_ON, SUPPORTS, CONTRADICTS, …)
4. **Builds** four knowledge graph types: concept graph, hierarchy tree, dependency graph, and insight graph — all validated by a three-pass semantic validator
5. **Compresses** the full graph using importance scoring, concept clustering, and four condensation operations
6. **Renders** an interactive mind map (pyvis), static NetworkX graph, Markdown outline, and structured exports

---

## Quick start

### Prerequisites

- Python 3.10 or higher
- Tesseract OCR installed on your system:
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/knowledge-mapper.git
cd knowledge-mapper

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the spaCy language model
python -m spacy download en_core_web_sm
```

### Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser, upload a PDF, and click **Process book**.

---

## Repository structure

```
knowledge_mapper/
│
├── app.py                     # Streamlit UI and pipeline orchestrator
│
├── modules/
│   ├── pdf_ingestion.py       # Stages 1+2: PDF parsing, OCR, chunking
│   ├── text_segmentation.py   # Stage 3: chapter/section/paragraph detection
│   ├── concept_extraction.py  # Stages 4+5: ConceptNode + RelationshipEdge extraction
│   ├── knowledge_graph.py     # Stages 6+7: graph construction, hierarchy, validation
│   ├── compression_engine.py  # COMPRESSION_ENGINE: importance scoring + clustering
│   └── mindmap_generator.py   # Stage 8: pyvis HTML, JSON, Markdown output
│
├── utils/
│   ├── logger.py              # Structured logging (console + file)
│   └── error_handler.py       # Self-healing decorators: retry, safe_stage
│
├── export/
│   ├── pdf_exporter.py        # PDF report (fpdf2)
│   ├── html_exporter.py       # Standalone HTML with interactive graph
│   └── json_exporter.py       # Structured JSON export
│
├── logs/                      # Auto-created on first run
├── requirements.txt
└── README.md
```

---

## Self-healing behaviour

Each pipeline stage has an automatic recovery strategy:

| Stage | Primary | Fallback 1 | Fallback 2 |
|-------|---------|-----------|-----------|
| PDF ingestion | pdfplumber | PyMuPDF | Tesseract OCR |
| Segmentation | Heading detection | Page-level segments | — |
| Concept extraction | Pattern + spaCy NER | Pattern-only | Segment-title concepts |
| Graph construction | Full four-graph build | Partial graph | — |
| Compression | Full condensation | Floor-enforced output | Raw concept list |
| Mind map | pyvis interactive | NetworkX static | Plain text outline |

All failures are logged to the sidebar pipeline log and to `logs/km_YYYYMMDD.log`.

---

## Exports

From the **Export** tab in the UI:

| Format | Contents |
|--------|----------|
| PDF | Structured report with hierarchy, scores, page refs |
| HTML | Standalone interactive pyvis mind map |
| JSON | Full tree structure with scores, tiers, annotations |
| Markdown | Hierarchical outline for note-taking tools |

---

## Architecture overview

The system implements three layered specifications:

- **RESEARCH_LAYER** — defines the intelligence: ConceptNode schema, RelationshipEdge schema, four graph types, HierarchyEngine (DAG + cycle guard), InsightEngine (five detectors), and three-pass SemanticValidator
- **SYSTEM_ARCHITECTURE** — modular Python pipeline with self-healing error recovery at each stage, staged fallback algorithms, and structured logging
- **COMPRESSION_ENGINE** — importance scoring (five signals), three-phase concept clustering, four condensation operations (prune/merge/dedup/flatten), and simplified mind map generation with cluster-based branching

---

## Configuration

Edit values at the top of each module file:

| File | Variable | Default | Effect |
|------|----------|---------|--------|
| `pdf_ingestion.py` | `CHUNK_SIZE` | 20 | Pages per processing window |
| `pdf_ingestion.py` | `CHUNK_OVERLAP` | 3 | Overlap between windows |
| `compression_engine.py` | `THRESHOLD_CORE` | 0.75 | Min score to be a core concept |
| `compression_engine.py` | `MAX_NODES` | 80 | Max nodes in compressed output |
| `compression_engine.py` | `MAX_CLUSTER_SIZE` | 12 | Max concepts per cluster |

---

## Troubleshooting

**`tesseract not found`** — ensure Tesseract is on your PATH: `tesseract --version`

**`en_core_web_sm not found`** — run `python -m spacy download en_core_web_sm`

**`pyvis not rendering`** — the interactive graph requires a browser with JavaScript. If running in a restricted environment the app automatically falls back to the static NetworkX graph.

**Very few concepts extracted** — this usually means the PDF is purely image-based. The app will automatically retry with Tesseract OCR. For best results, use a PDF with embedded text.

---

## Licence

MIT
