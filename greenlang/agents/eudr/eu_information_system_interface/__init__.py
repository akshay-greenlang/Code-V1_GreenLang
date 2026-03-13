# -*- coding: utf-8 -*-
"""
AGENT-EUDR-036: EU Information System Interface Agent

EUDR Articles 4, 12, 13, 14, 31, and 33 compliant interface for the EU
Information System. Provides production-grade capabilities for submitting
Due Diligence Statements (DDS), registering operators, formatting geolocation
data, assembling document packages, tracking submission lifecycle status,
communicating with the EU IS API, and maintaining Article 31 audit trails.

The agent sits between the Documentation Generator (EUDR-030), Improvement
Plan Creator (EUDR-035), and the official EU Information System, consuming
prepared compliance documentation and submitting it through the regulated
EU IS API interface.

Core capabilities:
    1. DDSSubmitter             -- Creates, validates, and submits DDS per
       EUDR Articles 4 and 12 with full commodity line support
    2. OperatorRegistrar        -- Manages operator registration lifecycle
       per Articles 4-6 including EORI validation and renewal
    3. GeolocationFormatter     -- Formats coordinates to EU IS specs per
       Annex II with polygon simplification and area-based format selection
    4. PackageAssembler         -- Assembles complete document packages with
       regulatory ordering, size validation, and integrity hashing
    5. StatusTracker            -- Tracks submission lifecycle through EU IS
       status states with polling, caching, and timeline reporting
    6. APIClient                -- Manages all HTTP communication with EU IS
       including mTLS, retry logic, circuit breaking, and connection pooling
    7. AuditRecorder            -- Records Article 31 audit trail events with
       SHA-256 provenance hashes and 5-year retention compliance

Foundational modules:
    - config.py       -- EUInformationSystemInterfaceConfig with GL_EUDR_EUIS_
      env var support (60+ settings)
    - models.py       -- Pydantic v2 data models with 13 enumerations,
      15+ core models, and health/status types
    - provenance.py   -- SHA-256 chain-hashed audit trail tracking
    - metrics.py      -- 18 Prometheus self-monitoring metrics (gl_eudr_euis_)

Agent ID: GL-EUDR-EUIS-036
Module: greenlang.agents.eudr.eu_information_system_interface
PRD: PRD-AGENT-EUDR-036
Regulation: EU 2023/1115 Articles 4, 12, 13, 14, 31, 33

Example:
    >>> from greenlang.agents.eudr.eu_information_system_interface import (
    ...     EUInformationSystemInterfaceConfig,
    ...     get_config,
    ...     EUDRCommodity,
    ...     DDSStatus,
    ... )
    >>> cfg = get_config()
    >>> print(cfg.eu_api_base_url)
    https://eudr-is.europa.eu/api/v1

    >>> from greenlang.agents.eudr.eu_information_system_interface import (
    ...     EUInformationSystemInterfaceService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> health = await service.health_check()
    >>> assert health["status"] in ("healthy", "degraded")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-036 EU Information System Interface (GL-EUDR-EUIS-036)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-EUIS-036"

# ---------------------------------------------------------------------------
# Public API listing
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Configuration --
    "EUInformationSystemInterfaceConfig",
    "get_config",
    "reset_config",
    # -- Enumerations (13) --
    "EUDRCommodity",
    "OperatorType",
    "DDSType",
    "DDSStatus",
    "SubmissionStatus",
    "RegistrationStatus",
    "GeolocationFormat",
    "CoordinateSystem",
    "DocumentType",
    "AuditEventType",
    "CompetentAuthority",
    # -- Core Models (15+) --
    "Coordinate",
    "GeoPolygon",
    "GeolocationData",
    "OperatorRegistration",
    "DDSCommodityLine",
    "DueDiligenceStatement",
    "DocumentPackage",
    "SubmissionRequest",
    "StatusCheckResult",
    "AuditRecord",
    "APICallRecord",
    "DDSSummary",
    "SubmissionReport",
    "HealthStatus",
    # -- Constants --
    "AGENT_ID",
    "AGENT_VERSION",
    # -- Provenance --
    "ProvenanceTracker",
    "GENESIS_HASH",
    # -- Metrics (18) --
    "record_dds_submitted",
    "record_dds_accepted",
    "record_dds_rejected",
    "record_operator_registered",
    "record_package_assembled",
    "record_status_check",
    "record_api_call",
    "record_api_error",
    "observe_submission_duration",
    "observe_geolocation_format_duration",
    "observe_package_assembly_duration",
    "observe_api_call_duration",
    "observe_status_check_duration",
    "set_active_submissions",
    "set_pending_dds",
    "set_registered_operators",
    "set_eu_api_health",
    "set_audit_records_count",
    # -- Engines (7) --
    "DDSSubmitter",
    "OperatorRegistrar",
    "GeolocationFormatter",
    "PackageAssembler",
    "StatusTracker",
    "APIClient",
    "AuditRecorder",
    # -- Service Facade --
    "EUInformationSystemInterfaceService",
    "get_service",
    "reset_service",
    "lifespan",
]


# ---------------------------------------------------------------------------
# Lazy import machinery
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Configuration
    "EUInformationSystemInterfaceConfig": ("config", "EUInformationSystemInterfaceConfig"),
    "get_config": ("config", "get_config"),
    "reset_config": ("config", "reset_config"),
    # Enumerations
    "EUDRCommodity": ("models", "EUDRCommodity"),
    "OperatorType": ("models", "OperatorType"),
    "DDSType": ("models", "DDSType"),
    "DDSStatus": ("models", "DDSStatus"),
    "SubmissionStatus": ("models", "SubmissionStatus"),
    "RegistrationStatus": ("models", "RegistrationStatus"),
    "GeolocationFormat": ("models", "GeolocationFormat"),
    "CoordinateSystem": ("models", "CoordinateSystem"),
    "DocumentType": ("models", "DocumentType"),
    "AuditEventType": ("models", "AuditEventType"),
    "CompetentAuthority": ("models", "CompetentAuthority"),
    # Core Models
    "Coordinate": ("models", "Coordinate"),
    "GeoPolygon": ("models", "GeoPolygon"),
    "GeolocationData": ("models", "GeolocationData"),
    "OperatorRegistration": ("models", "OperatorRegistration"),
    "DDSCommodityLine": ("models", "DDSCommodityLine"),
    "DueDiligenceStatement": ("models", "DueDiligenceStatement"),
    "DocumentPackage": ("models", "DocumentPackage"),
    "SubmissionRequest": ("models", "SubmissionRequest"),
    "StatusCheckResult": ("models", "StatusCheckResult"),
    "AuditRecord": ("models", "AuditRecord"),
    "APICallRecord": ("models", "APICallRecord"),
    "DDSSummary": ("models", "DDSSummary"),
    "SubmissionReport": ("models", "SubmissionReport"),
    "HealthStatus": ("models", "HealthStatus"),
    # Constants
    "AGENT_ID": ("models", "AGENT_ID"),
    "AGENT_VERSION": ("models", "AGENT_VERSION"),
    # Provenance
    "ProvenanceTracker": ("provenance", "ProvenanceTracker"),
    "GENESIS_HASH": ("provenance", "GENESIS_HASH"),
    # Metrics (counters)
    "record_dds_submitted": ("metrics", "record_dds_submitted"),
    "record_dds_accepted": ("metrics", "record_dds_accepted"),
    "record_dds_rejected": ("metrics", "record_dds_rejected"),
    "record_operator_registered": ("metrics", "record_operator_registered"),
    "record_package_assembled": ("metrics", "record_package_assembled"),
    "record_status_check": ("metrics", "record_status_check"),
    "record_api_call": ("metrics", "record_api_call"),
    "record_api_error": ("metrics", "record_api_error"),
    # Metrics (histograms)
    "observe_submission_duration": ("metrics", "observe_submission_duration"),
    "observe_geolocation_format_duration": ("metrics", "observe_geolocation_format_duration"),
    "observe_package_assembly_duration": ("metrics", "observe_package_assembly_duration"),
    "observe_api_call_duration": ("metrics", "observe_api_call_duration"),
    "observe_status_check_duration": ("metrics", "observe_status_check_duration"),
    # Metrics (gauges)
    "set_active_submissions": ("metrics", "set_active_submissions"),
    "set_pending_dds": ("metrics", "set_pending_dds"),
    "set_registered_operators": ("metrics", "set_registered_operators"),
    "set_eu_api_health": ("metrics", "set_eu_api_health"),
    "set_audit_records_count": ("metrics", "set_audit_records_count"),
    # Engines
    "DDSSubmitter": ("dds_submitter", "DDSSubmitter"),
    "OperatorRegistrar": ("operator_registrar", "OperatorRegistrar"),
    "GeolocationFormatter": ("geolocation_formatter", "GeolocationFormatter"),
    "PackageAssembler": ("package_assembler", "PackageAssembler"),
    "StatusTracker": ("status_tracker", "StatusTracker"),
    "APIClient": ("api_client", "APIClient"),
    "AuditRecorder": ("audit_recorder", "AuditRecorder"),
    # Service Facade
    "EUInformationSystemInterfaceService": ("setup", "EUInformationSystemInterfaceService"),
    "get_service": ("setup", "get_service"),
    "reset_service": ("setup", "reset_service"),
    "lifespan": ("setup", "lifespan"),
}


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy imports.

    Enables ``from greenlang.agents.eudr.eu_information_system_interface import X``
    without eagerly loading all submodules at package import time.

    Args:
        name: Attribute name to look up.

    Returns:
        The lazily imported object.

    Raises:
        AttributeError: If the name is not a known export.
    """
    if name in _LAZY_IMPORTS:
        module_suffix, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(
            f"greenlang.agents.eudr.eu_information_system_interface.{module_suffix}"
        )
        return getattr(mod, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the current module version string.

    Returns:
        Version string in semver format (e.g. "1.0.0").

    Example:
        >>> get_version()
        '1.0.0'
    """
    return __version__


def get_agent_info() -> dict:
    """Return agent identification and capability metadata.

    Returns:
        Dictionary with agent_id, version, regulation references,
        engine listing, and capability summary.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-EUDR-EUIS-036'
        >>> info["engine_count"]
        7
    """
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "EU Information System Interface",
        "prd": "PRD-AGENT-EUDR-036",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["4", "12", "13", "14", "31", "33"],
        "enforcement_date_large": "2025-12-30",
        "enforcement_date_sme": "2026-06-30",
        "eudr_commodities": [
            "cattle", "cocoa", "coffee", "oil_palm",
            "rubber", "soya", "wood",
        ],
        "eu_member_states": [
            "DE", "FR", "IT", "ES", "NL", "BE", "AT", "PL",
            "SE", "DK", "FI", "PT", "IE", "CZ", "RO", "HU",
            "BG", "HR", "SK", "LT", "SI", "LV", "EE", "CY",
            "LU", "MT", "EL",
        ],
        "dds_types": ["placing", "making_available", "export"],
        "operator_types": ["operator", "trader", "sme_operator", "sme_trader"],
        "engines": [
            "DDSSubmitter",
            "OperatorRegistrar",
            "GeolocationFormatter",
            "PackageAssembler",
            "StatusTracker",
            "APIClient",
            "AuditRecorder",
        ],
        "engine_count": 7,
        "enum_count": 13,
        "core_model_count": 15,
        "metrics_count": 18,
        "db_prefix": "gl_eudr_euis_",
        "metrics_prefix": "gl_eudr_euis_",
        "env_prefix": "GL_EUDR_EUIS_",
    }
