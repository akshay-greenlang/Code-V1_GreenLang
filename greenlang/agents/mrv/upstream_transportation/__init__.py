# -*- coding: utf-8 -*-
"""
Upstream Transportation & Distribution Agent Package - AGENT-MRV-017

GHG Protocol Scope 3, Category 4: Upstream Transportation and Distribution.
Calculates emissions from transportation/distribution of purchased products
in vehicles/facilities NOT owned or controlled by the reporting company.

Agent ID: GL-MRV-S3-004
Package: greenlang.agents.mrv.upstream_transportation
API: /api/v1/upstream-transportation
DB Migration: V068
Metrics Prefix: gl_uto_
Table Prefix: gl_uto_

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

__all__ = [
    "TransportDatabaseEngine",
    "DistanceBasedCalculatorEngine",
    "FuelBasedCalculatorEngine",
    "SpendBasedCalculatorEngine",
    "MultiLegCalculatorEngine",
    "ComplianceCheckerEngine",
    "TransportPipelineEngine",
]

VERSION: str = "1.0.0"

# Graceful imports - each engine with try/except
try:
    from greenlang.agents.mrv.upstream_transportation.transport_database import TransportDatabaseEngine
except ImportError:
    TransportDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.upstream_transportation.distance_based_calculator import DistanceBasedCalculatorEngine
except ImportError:
    DistanceBasedCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.upstream_transportation.fuel_based_calculator import FuelBasedCalculatorEngine
except ImportError:
    FuelBasedCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.upstream_transportation.spend_based_calculator import SpendBasedCalculatorEngine
except ImportError:
    SpendBasedCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.upstream_transportation.multi_leg_calculator import MultiLegCalculatorEngine
except ImportError:
    MultiLegCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.upstream_transportation.compliance_checker import ComplianceCheckerEngine
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.upstream_transportation.transport_pipeline import TransportPipelineEngine
except ImportError:
    TransportPipelineEngine = None  # type: ignore[assignment,misc]
