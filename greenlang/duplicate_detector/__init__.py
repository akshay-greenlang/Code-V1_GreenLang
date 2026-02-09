# -*- coding: utf-8 -*-
"""
GL-DATA-X-014: GreenLang Duplicate Detection Agent SDK
========================================================

This package provides record deduplication for GreenLang sustainability
datasets. It supports:

- Record fingerprinting (SHA-256, SimHash, MinHash)
- Blocking strategies for candidate pair generation
  (sorted neighborhood, standard, canopy)
- Field-level similarity comparison (exact, Levenshtein, Jaro-Winkler,
  Soundex, n-gram, TF-IDF cosine, numeric, date)
- Match classification (threshold-based + Fellegi-Sunter probabilistic)
- Cluster resolution (union-find, connected components)
- Record merging (keep_first, keep_latest, keep_most_complete,
  merge_fields, golden_record)
- End-to-end deduplication pipeline orchestration
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_DD_ env prefix

Key Components:
    - config: DuplicateDetectorConfig with GL_DD_ env prefix
    - record_fingerprinter: Record fingerprinting engine
    - blocking_engine: Blocking strategy engine
    - similarity_scorer: Similarity comparison engine
    - match_classifier: Match classification engine
    - cluster_resolver: Cluster resolution engine
    - merge_engine: Record merge engine
    - dedup_pipeline: End-to-end pipeline engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: DuplicateDetectorService facade

Example:
    >>> from greenlang.duplicate_detector import DuplicateDetectorService
    >>> service = DuplicateDetectorService()
    >>> result = service.fingerprint_records(
    ...     records=[{"name": "Alice", "email": "alice@co.com"}],
    ...     field_set=["name", "email"],
    ... )
    >>> print(result.total_records, result.unique_fingerprints)
    1 1

Agent ID: GL-DATA-X-014
Agent Name: Duplicate Detection Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-X-014"
__agent_name__ = "Duplicate Detection Agent"

# SDK availability flag
DUPLICATE_DETECTOR_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.duplicate_detector.config import (
    DuplicateDetectorConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.duplicate_detector.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.duplicate_detector.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    dd_jobs_processed_total,
    dd_records_fingerprinted_total,
    dd_blocks_created_total,
    dd_comparisons_performed_total,
    dd_matches_found_total,
    dd_clusters_formed_total,
    dd_merges_completed_total,
    dd_merge_conflicts_total,
    dd_processing_duration_seconds,
    dd_similarity_score,
    dd_active_jobs,
    dd_processing_errors_total,
    # Helper functions
    inc_jobs,
    inc_fingerprints,
    inc_blocks,
    inc_comparisons,
    inc_matches,
    inc_clusters,
    inc_merges,
    inc_conflicts,
    observe_duration,
    observe_similarity,
    set_active_jobs,
    inc_errors,
)

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.duplicate_detector.record_fingerprinter import (
        RecordFingerprinterEngine,
    )
except ImportError:
    RecordFingerprinterEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.duplicate_detector.blocking_engine import BlockingEngine
except ImportError:
    BlockingEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.duplicate_detector.similarity_scorer import (
        SimilarityScorerEngine,
    )
except ImportError:
    SimilarityScorerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.duplicate_detector.match_classifier import (
        MatchClassifierEngine,
    )
except ImportError:
    MatchClassifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.duplicate_detector.cluster_resolver import (
        ClusterResolverEngine,
    )
except ImportError:
    ClusterResolverEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.duplicate_detector.merge_engine import MergeEngine
except ImportError:
    MergeEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.duplicate_detector.dedup_pipeline import (
        DeduplicationPipelineEngine,
    )
except ImportError:
    DeduplicationPipelineEngine = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from normalizer (Entity Resolution)
# ---------------------------------------------------------------------------
try:
    from greenlang.normalizer.entity_resolver import EntityResolver
    from greenlang.normalizer.models import ConfidenceLevel, EntityMatch
except ImportError:
    EntityResolver = None  # type: ignore[assignment, misc]
    ConfidenceLevel = None  # type: ignore[assignment, misc]
    EntityMatch = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Service setup facade and models
# ---------------------------------------------------------------------------
from greenlang.duplicate_detector.setup import (
    DuplicateDetectorService,
    configure_duplicate_detector,
    get_duplicate_detector,
    get_router,
    # Models
    FingerprintResponse,
    BlockResponse,
    CompareResponse,
    ClassifyResponse,
    ClusterResponse,
    MergeResponse,
    PipelineResponse,
    StatsResponse,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "DUPLICATE_DETECTOR_SDK_AVAILABLE",
    # Configuration
    "DuplicateDetectorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Provenance
    "ProvenanceTracker",
    # Metric flag
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "dd_jobs_processed_total",
    "dd_records_fingerprinted_total",
    "dd_blocks_created_total",
    "dd_comparisons_performed_total",
    "dd_matches_found_total",
    "dd_clusters_formed_total",
    "dd_merges_completed_total",
    "dd_merge_conflicts_total",
    "dd_processing_duration_seconds",
    "dd_similarity_score",
    "dd_active_jobs",
    "dd_processing_errors_total",
    # Metric helper functions
    "inc_jobs",
    "inc_fingerprints",
    "inc_blocks",
    "inc_comparisons",
    "inc_matches",
    "inc_clusters",
    "inc_merges",
    "inc_conflicts",
    "observe_duration",
    "observe_similarity",
    "set_active_jobs",
    "inc_errors",
    # Core engines (Layer 2)
    "RecordFingerprinterEngine",
    "BlockingEngine",
    "SimilarityScorerEngine",
    "MatchClassifierEngine",
    "ClusterResolverEngine",
    "MergeEngine",
    "DeduplicationPipelineEngine",
    # Layer 1 re-exports
    "EntityResolver",
    "ConfidenceLevel",
    "EntityMatch",
    # Service setup facade
    "DuplicateDetectorService",
    "configure_duplicate_detector",
    "get_duplicate_detector",
    "get_router",
    # Response models
    "FingerprintResponse",
    "BlockResponse",
    "CompareResponse",
    "ClassifyResponse",
    "ClusterResponse",
    "MergeResponse",
    "PipelineResponse",
    "StatsResponse",
]
