# -*- coding: utf-8 -*-
"""
Downstream Transportation & Distribution Agent Package - AGENT-MRV-022

GHG Protocol Scope 3, Category 9: Downstream Transportation and Distribution.
Calculates emissions from the transportation and distribution of products sold
by the reporting company in the reporting year between the reporting company's
operations and the end consumer (not paid for by the reporting company),
including retail and storage.

Agent ID: GL-MRV-S3-009
Package: greenlang.downstream_transportation
API: /api/v1/downstream-transportation
DB Migration: V073
Metrics Prefix: gl_dto_
Table Prefix: gl_dto_

Sub-Activities:
    - Outbound transportation (9a) -- post-sale transport per Incoterms
    - Outbound distribution (9b) -- distribution center / warehouse operations
    - Retail storage (9c) -- third-party retail energy consumption
    - Last-mile delivery (9d) -- final delivery to end consumer

Calculation Methods:
    - Distance-based (tonne-km x mode-specific EF)
    - Spend-based (EEIO factors with CPI deflation and currency conversion)
    - Average-data (industry average by distribution channel)
    - Supplier-specific (carrier-provided primary data)

7 Engines:
    1. DownstreamTransportDatabaseEngine -- EF lookup, vehicle/vessel types
    2. DistanceBasedCalculatorEngine -- tonne-km for all transport modes
    3. SpendBasedCalculatorEngine -- EEIO spend-based with CPI deflation
    4. AverageDataCalculatorEngine -- industry average by channel defaults
    5. WarehouseDistributionEngine -- DC, cold storage, retail storage
    6. ComplianceCheckerEngine -- 7-framework regulatory compliance
    7. DownstreamTransportPipelineEngine -- 10-stage orchestration pipeline

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

__all__ = [
    "DownstreamTransportDatabaseEngine",
    "DistanceBasedCalculatorEngine",
    "SpendBasedCalculatorEngine",
    "AverageDataCalculatorEngine",
    "WarehouseDistributionEngine",
    "ComplianceCheckerEngine",
    "DownstreamTransportPipelineEngine",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "METRICS_PREFIX",
    "API_PREFIX",
    "get_config",
]

AGENT_ID: str = "GL-MRV-S3-009"
AGENT_COMPONENT: str = "AGENT-MRV-022"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_dto_"
METRICS_PREFIX: str = "gl_dto_"
API_PREFIX: str = "/api/v1/downstream-transportation"

# ---------------------------------------------------------------------------
# Graceful imports -- each engine with try/except so the package loads even
# when optional dependencies are missing during lightweight imports or tests.
# ---------------------------------------------------------------------------

try:
    from greenlang.downstream_transportation.downstream_transport_database import (
        DownstreamTransportDatabaseEngine,
    )
except ImportError:
    DownstreamTransportDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_transportation.distance_based_calculator import (
        DistanceBasedCalculatorEngine,
    )
except ImportError:
    DistanceBasedCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_transportation.spend_based_calculator import (
        SpendBasedCalculatorEngine,
    )
except ImportError:
    SpendBasedCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_transportation.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
except ImportError:
    AverageDataCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_transportation.warehouse_distribution import (
        WarehouseDistributionEngine,
    )
except ImportError:
    WarehouseDistributionEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_transportation.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_transportation.downstream_transport_pipeline import (
        DownstreamTransportPipelineEngine,
    )
except ImportError:
    DownstreamTransportPipelineEngine = None  # type: ignore[assignment,misc]

# Export configuration helper
try:
    from greenlang.downstream_transportation.config import get_config
except ImportError:
    def get_config():  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None
