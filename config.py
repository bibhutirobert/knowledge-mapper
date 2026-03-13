"""
Central configuration for the Knowledge Mapper pipeline.
All tunable constants live here. Import via: from config import cfg
"""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class IngestionConfig:
    chunk_size: int    = 20     # pages per processing window
    chunk_overlap: int = 3      # overlap pages between windows
    ocr_dpi_scale: float = 2.0  # DPI multiplier for Tesseract upscale
    min_text_ratio: float = 0.3 # minimum fraction of pages needing text


@dataclass(frozen=True)
class ExtractionConfig:
    min_concept_title_len: int  = 3
    max_concept_title_len: int  = 80
    min_definition_tokens: int  = 3
    min_edge_confidence: float  = 0.4
    spacy_model: str            = "en_core_web_sm"
    spacy_text_cap: int         = 5000  # chars sent to spaCy per segment


@dataclass(frozen=True)
class CompressionConfig:
    threshold_core: float       = 0.75
    threshold_supporting: float = 0.45
    threshold_prunable: float   = 0.25
    min_nodes: int              = 5
    max_nodes: int              = 80
    max_depth: int              = 4
    min_cluster_size: int       = 2
    max_cluster_size: int       = 12
    dedup_cosine_threshold: float = 0.88

    # Score weights (must sum to 1.0)
    w_salience: float    = 0.30
    w_centrality: float  = 0.25
    w_recurrence: float  = 0.20
    w_authority: float   = 0.15
    w_type: float        = 0.10


@dataclass(frozen=True)
class GraphConfig:
    max_hierarchy_depth: int = 6
    min_edge_confidence: float = 0.4


@dataclass(frozen=True)
class AppConfig:
    ingestion:   IngestionConfig   = field(default_factory=IngestionConfig)
    extraction:  ExtractionConfig  = field(default_factory=ExtractionConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    graph:       GraphConfig       = field(default_factory=GraphConfig)
    log_level: str = "INFO"
    max_retries: int = 3
    retry_delay: float = 1.0


cfg = AppConfig()
