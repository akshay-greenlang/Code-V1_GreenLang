# -*- coding: utf-8 -*-
"""
AGENT-EUDR-038: Reference Number Generator Agent

EUDR reference number generation, format validation, sequence management,
batch processing, collision detection, lifecycle management, and
verification services. Provides production-grade capabilities for generating
unique, format-compliant reference numbers for Due Diligence Statements
and related EUDR documentation per EU 2023/1115 Article 4(2) requirements.

The agent works in conjunction with the Documentation Generator (EUDR-030)
to assign unique, verifiable reference numbers to all DDS submissions,
ensuring traceability across the EU Information System.

Core capabilities:
    1. NumberGenerator           -- Generates unique reference numbers with
       configurable format patterns per EU member state requirements
    2. FormatValidator          -- Validates reference number format compliance
       across all 27 EU member states with checksum verification
    3. SequenceManager          -- Manages atomic sequence counters with
       rollover, exhaustion detection, and distributed lock support
    4. BatchProcessor           -- Processes batch reference number generation
       requests with concurrency control and deduplication
    5. CollisionDetector        -- Detects and resolves reference number
       collisions with configurable retry strategies
    6. LifecycleManager         -- Manages reference number lifecycle states
       including activation, expiration, revocation, and archival
    7. VerificationService      -- Verifies reference number authenticity,
       checksum validity, and current lifecycle status

Foundational modules:
    - config.py       -- ReferenceNumberGeneratorConfig with GL_EUDR_RNG_
      env var support (50+ settings)
    - models.py       -- Pydantic v2 data models with 12+ enumerations,
      15+ core models, and health/status types
    - provenance.py   -- SHA-256 chain-hashed audit trail tracking
    - metrics.py      -- 18 Prometheus self-monitoring metrics (gl_eudr_rng_)

Agent ID: GL-EUDR-RNG-038
Module: greenlang.agents.eudr.reference_number_generator
PRD: PRD-AGENT-EUDR-038
Regulation: EU 2023/1115 Articles 4, 9, 31, 33

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-038 Reference Number Generator (GL-EUDR-RNG-038)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-RNG-038"

# ---------------------------------------------------------------------------
# Public API listing
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Configuration --
    "ReferenceNumberGeneratorConfig",
    "get_config",
    "reset_config",
    # -- Enumerations (12+) --
    "MemberStateCode",
    "ReferenceStatus",
    "ReferenceType",
    "SequenceStatus",
    "BatchStatus",
    "FormatPattern",
    "ChecksumAlgorithm",
    "ValidationSeverity",
    "LifecycleAction",
    "VerificationResult",
    "CollisionStrategy",
    "EUDRCommodity",
    # -- Core Models (15+) --
    "ReferenceNumber",
    "SequenceCounter",
    "BatchRequest",
    "BatchResult",
    "FormatRule",
    "ValidationIssue",
    "ValidationReport",
    "CollisionRecord",
    "LifecycleEvent",
    "VerificationReport",
    "ReferenceQuery",
    "ReferenceStats",
    "HealthStatus",
    "GenerationRequest",
    "GenerationResponse",
    # -- Constants --
    "AGENT_ID",
    "AGENT_VERSION",
    "EU_MEMBER_STATES",
    # -- Provenance --
    "ProvenanceTracker",
    "GENESIS_HASH",
    # -- Metrics --
    "record_reference_generated",
    "record_batch_processed",
    "record_collision_detected",
    "record_verification_performed",
    "record_lifecycle_transition",
    "record_format_validation",
    "record_sequence_increment",
    "record_api_error",
    "observe_generation_duration",
    "observe_batch_duration",
    "observe_verification_duration",
    "observe_validation_duration",
    "observe_collision_resolution_duration",
    "set_active_references",
    "set_sequence_current_value",
    "set_pending_batches",
    "set_expired_references",
    "set_collision_rate",
    # -- Engines (7) --
    "NumberGenerator",
    "FormatValidator",
    "SequenceManager",
    "BatchProcessor",
    "CollisionDetector",
    "LifecycleManager",
    "VerificationService",
    # -- Service Facade --
    "ReferenceNumberGeneratorService",
    "get_service",
    "reset_service",
    "lifespan",
]


# ---------------------------------------------------------------------------
# Lazy import machinery
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Configuration
    "ReferenceNumberGeneratorConfig": ("config", "ReferenceNumberGeneratorConfig"),
    "get_config": ("config", "get_config"),
    "reset_config": ("config", "reset_config"),
    # Enumerations (12+)
    "MemberStateCode": ("models", "MemberStateCode"),
    "ReferenceStatus": ("models", "ReferenceStatus"),
    "ReferenceType": ("models", "ReferenceType"),
    "SequenceStatus": ("models", "SequenceStatus"),
    "BatchStatus": ("models", "BatchStatus"),
    "FormatPattern": ("models", "FormatPattern"),
    "ChecksumAlgorithm": ("models", "ChecksumAlgorithm"),
    "ValidationSeverity": ("models", "ValidationSeverity"),
    "LifecycleAction": ("models", "LifecycleAction"),
    "VerificationResult": ("models", "VerificationResult"),
    "CollisionStrategy": ("models", "CollisionStrategy"),
    "EUDRCommodity": ("models", "EUDRCommodity"),
    # Core Models (15+)
    "ReferenceNumber": ("models", "ReferenceNumber"),
    "SequenceCounter": ("models", "SequenceCounter"),
    "BatchRequest": ("models", "BatchRequest"),
    "BatchResult": ("models", "BatchResult"),
    "FormatRule": ("models", "FormatRule"),
    "ValidationIssue": ("models", "ValidationIssue"),
    "ValidationReport": ("models", "ValidationReport"),
    "CollisionRecord": ("models", "CollisionRecord"),
    "LifecycleEvent": ("models", "LifecycleEvent"),
    "VerificationReport": ("models", "VerificationReport"),
    "ReferenceQuery": ("models", "ReferenceQuery"),
    "ReferenceStats": ("models", "ReferenceStats"),
    "HealthStatus": ("models", "HealthStatus"),
    "GenerationRequest": ("models", "GenerationRequest"),
    "GenerationResponse": ("models", "GenerationResponse"),
    # Constants
    "AGENT_ID": ("models", "AGENT_ID"),
    "AGENT_VERSION": ("models", "AGENT_VERSION"),
    "EU_MEMBER_STATES": ("models", "EU_MEMBER_STATES"),
    # Provenance
    "ProvenanceTracker": ("provenance", "ProvenanceTracker"),
    "GENESIS_HASH": ("provenance", "GENESIS_HASH"),
    # Metrics (counters)
    "record_reference_generated": ("metrics", "record_reference_generated"),
    "record_batch_processed": ("metrics", "record_batch_processed"),
    "record_collision_detected": ("metrics", "record_collision_detected"),
    "record_verification_performed": ("metrics", "record_verification_performed"),
    "record_lifecycle_transition": ("metrics", "record_lifecycle_transition"),
    "record_format_validation": ("metrics", "record_format_validation"),
    "record_sequence_increment": ("metrics", "record_sequence_increment"),
    "record_api_error": ("metrics", "record_api_error"),
    # Metrics (histograms)
    "observe_generation_duration": ("metrics", "observe_generation_duration"),
    "observe_batch_duration": ("metrics", "observe_batch_duration"),
    "observe_verification_duration": ("metrics", "observe_verification_duration"),
    "observe_validation_duration": ("metrics", "observe_validation_duration"),
    "observe_collision_resolution_duration": ("metrics", "observe_collision_resolution_duration"),
    # Metrics (gauges)
    "set_active_references": ("metrics", "set_active_references"),
    "set_sequence_current_value": ("metrics", "set_sequence_current_value"),
    "set_pending_batches": ("metrics", "set_pending_batches"),
    "set_expired_references": ("metrics", "set_expired_references"),
    "set_collision_rate": ("metrics", "set_collision_rate"),
    # Engines (7)
    "NumberGenerator": ("number_generator", "NumberGenerator"),
    "FormatValidator": ("format_validator", "FormatValidator"),
    "SequenceManager": ("sequence_manager", "SequenceManager"),
    "BatchProcessor": ("batch_processor", "BatchProcessor"),
    "CollisionDetector": ("collision_detector", "CollisionDetector"),
    "LifecycleManager": ("lifecycle_manager", "LifecycleManager"),
    "VerificationService": ("verification_service", "VerificationService"),
    # Service Facade
    "ReferenceNumberGeneratorService": ("setup", "ReferenceNumberGeneratorService"),
    "get_service": ("setup", "get_service"),
    "reset_service": ("setup", "reset_service"),
    "lifespan": ("setup", "lifespan"),
}


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy imports."""
    if name in _LAZY_IMPORTS:
        module_suffix, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(
            f"greenlang.agents.eudr.reference_number_generator.{module_suffix}"
        )
        return getattr(mod, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the current module version string."""
    return __version__


def get_agent_info() -> dict:
    """Return agent identification and capability metadata."""
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Reference Number Generator",
        "prd": "PRD-AGENT-EUDR-038",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["4", "9", "31", "33"],
        "enforcement_date_large": "2025-12-30",
        "enforcement_date_sme": "2026-06-30",
        "engines": [
            "NumberGenerator",
            "FormatValidator",
            "SequenceManager",
            "BatchProcessor",
            "CollisionDetector",
            "LifecycleManager",
            "VerificationService",
        ],
        "engine_count": 7,
        "enum_count": 12,
        "core_model_count": 15,
        "metrics_count": 18,
        "member_states_supported": 27,
        "db_prefix": "gl_eudr_rng_",
        "metrics_prefix": "gl_eudr_rng_",
        "env_prefix": "GL_EUDR_RNG_",
    }
