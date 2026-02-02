# -*- coding: utf-8 -*-
"""
GreenLang Shared Services
Core Infrastructure for Sustainability Applications

This module provides reusable services extracted from GreenLang applications
to enable consistent, high-quality sustainability calculations and data management.

Shared Services:
- Factor Broker: Multi-source emission factor resolution
- Entity MDM: ML-powered entity master data management
- Methodologies: Uncertainty quantification and data quality assessment
- PCF Exchange: Product Carbon Footprint exchange protocol support

Version: 1.0.0
License: Proprietary - GreenLang Platform
"""

# Factor Broker Service
from greenlang.services.factor_broker import (
    FactorBroker,
    FactorRequest,
    FactorResponse,
    FactorMetadata,
    DataQualityIndicator,
    FactorCache,
    FactorBrokerError,
    FactorNotFoundError,
    LicenseViolationError,
)

# Entity MDM Service
from greenlang.services.entity_mdm.ml.resolver import (
    EntityResolver,
    ResolutionStatus,
    MatchResult,
    ReviewItem,
)
from greenlang.services.entity_mdm.ml.vector_store import (
    VectorStore,
    SupplierEntity,
)

# Methodologies Service
from greenlang.services.methodologies import (
    PedigreeMatrixEvaluator,
    MonteCarloSimulator,
    DQICalculator,
    PedigreeScore,
    MonteCarloResult,
    UncertaintyResult,
)

# PCF Exchange Service
from greenlang.services.pcf_exchange import (
    PCFExchangeService,
    PACTPathfinderClient,
    CatenaXClient,
)

__version__ = "1.0.0"

__all__ = [
    # Factor Broker
    "FactorBroker",
    "FactorRequest",
    "FactorResponse",
    "FactorMetadata",
    "DataQualityIndicator",
    "FactorCache",
    "FactorBrokerError",
    "FactorNotFoundError",
    "LicenseViolationError",

    # Entity MDM
    "EntityResolver",
    "ResolutionStatus",
    "MatchResult",
    "ReviewItem",
    "VectorStore",
    "SupplierEntity",

    # Methodologies
    "PedigreeMatrixEvaluator",
    "MonteCarloSimulator",
    "DQICalculator",
    "PedigreeScore",
    "MonteCarloResult",
    "UncertaintyResult",

    # PCF Exchange
    "PCFExchangeService",
    "PACTPathfinderClient",
    "CatenaXClient",
]
