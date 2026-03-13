# -*- coding: utf-8 -*-
"""
AGENT-EUDR-027: Information Gathering Agent

Collects, verifies, normalizes, and packages all information required by
EUDR Article 9 for due diligence statements. Orchestrates seven processing
engines across the full information gathering lifecycle:

    Engine 1: External Database Connector    -- 11 regulatory/trade databases
    Engine 2: Certification Verification     -- 6 certification body adapters
    Engine 3: Public Data Mining             -- 8 public dataset harvesters
    Engine 4: Supplier Information Aggregator -- Entity resolution + merging
    Engine 5: Completeness Validation        -- Article 9 element scoring
    Engine 6: Data Normalization             -- 8 normalization transforms
    Engine 7: Package Assembly               -- DDS evidence bundle compiler

Package Structure:
    Core Modules:
        - models.py          -- 12 enums, 16 core models
        - config.py          -- ~60 environment variables with GL_EUDR_IGA_ prefix
        - provenance.py      -- SHA-256 chain hashing
        - metrics.py         -- 18 Prometheus metrics with gl_eudr_iga_ prefix

    Processing Engines:
        - external_database_connector.py         -- Adapter-pattern DB connector
        - certification_verification_engine.py   -- Certificate verification
        - public_data_mining_engine.py           -- Public data harvesting
        - supplier_information_aggregator.py     -- Supplier profile aggregation
        - completeness_validation_engine.py      -- Article 9 completeness
        - data_normalization_engine.py           -- Data normalization
        - package_assembly_engine.py             -- Information package assembly

    Service Facade:
        - setup.py           -- InformationGatheringService facade
        - api.py             -- FastAPI router (11 endpoints)

Commodities Supported:
    cattle, cocoa, coffee, oil_palm, rubber, soya, wood

Regulatory References:
    - EU 2023/1115 (EUDR) Articles 4, 9, 10, 12, 13, 29, 31
    - Article 9: 10 mandatory information elements
    - Article 13: Simplified due diligence for low-risk countries
    - Article 31: 5-year record retention

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 Information Gathering Agent (GL-EUDR-IGA-027)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__: list[str] = [
    "__version__",
    # Service facade
    "InformationGatheringService",
    # Configuration
    "InformationGatheringConfig",
    "get_config",
    # Engines
    "ExternalDatabaseConnectorEngine",
    "CertificationVerificationEngine",
    "PublicDataMiningEngine",
    "SupplierInformationAggregator",
    "CompletenessValidationEngine",
    "DataNormalizationEngine",
    "PackageAssemblyEngine",
    # Provenance
    "ProvenanceTracker",
    # Models (enums)
    "ExternalDatabaseSource",
    "CertificationBody",
    "EUDRCommodity",
    "QueryStatus",
    "CertVerificationStatus",
    "CompletenessClassification",
    "NormalizationType",
    "GatheringOperationStatus",
    # Models (core)
    "QueryResult",
    "CertificateVerificationResult",
    "SupplierProfile",
    "CompletenessReport",
    "InformationPackage",
    "GatheringOperation",
]


def _lazy_import(name: str) -> object:
    """Lazy import to avoid circular imports at module load time.

    Args:
        name: Name of the attribute to import.

    Returns:
        The imported object.

    Raises:
        AttributeError: If the name is not in __all__.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Service facade
    if name == "InformationGatheringService":
        from greenlang.agents.eudr.information_gathering.setup import (
            InformationGatheringService,
        )
        return InformationGatheringService

    # Configuration
    if name == "InformationGatheringConfig":
        from greenlang.agents.eudr.information_gathering.config import (
            InformationGatheringConfig,
        )
        return InformationGatheringConfig
    if name == "get_config":
        from greenlang.agents.eudr.information_gathering.config import (
            get_config,
        )
        return get_config

    # Engines
    engine_map = {
        "ExternalDatabaseConnectorEngine": (
            "external_database_connector", "ExternalDatabaseConnectorEngine"
        ),
        "CertificationVerificationEngine": (
            "certification_verification_engine",
            "CertificationVerificationEngine",
        ),
        "PublicDataMiningEngine": (
            "public_data_mining_engine", "PublicDataMiningEngine"
        ),
        "SupplierInformationAggregator": (
            "supplier_information_aggregator",
            "SupplierInformationAggregator",
        ),
        "CompletenessValidationEngine": (
            "completeness_validation_engine",
            "CompletenessValidationEngine",
        ),
        "DataNormalizationEngine": (
            "data_normalization_engine", "DataNormalizationEngine"
        ),
        "PackageAssemblyEngine": (
            "package_assembly_engine", "PackageAssemblyEngine"
        ),
    }
    if name in engine_map:
        module_name, class_name = engine_map[name]
        import importlib
        mod = importlib.import_module(
            f"greenlang.agents.eudr.information_gathering.{module_name}"
        )
        return getattr(mod, class_name)

    # Provenance
    if name == "ProvenanceTracker":
        from greenlang.agents.eudr.information_gathering.provenance import (
            ProvenanceTracker,
        )
        return ProvenanceTracker

    # All models are in models.py
    from greenlang.agents.eudr.information_gathering import models
    if hasattr(models, name):
        return getattr(models, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy imports.

    Enables ``from greenlang.agents.eudr.information_gathering import X``
    without eagerly loading all submodules at package import time.

    Args:
        name: Attribute name to look up.

    Returns:
        The lazily imported object.
    """
    return _lazy_import(name)
