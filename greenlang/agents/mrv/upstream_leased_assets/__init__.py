# -*- coding: utf-8 -*-
"""
Upstream Leased Assets Agent Package - AGENT-MRV-021

GHG Protocol Scope 3, Category 8: Upstream Leased Assets.
Calculates emissions from the operation of assets that are leased by the
reporting company (lessee) and not included in Scope 1 or Scope 2,
reported by lessees using the financial control or equity share approach.

Agent ID: GL-MRV-S3-008
Package: greenlang.agents.mrv.upstream_leased_assets
API: /api/v1/upstream-leased-assets
DB Migration: V072
Metrics Prefix: gl_ula_
Table Prefix: gl_ula_

Supported Asset Categories:
    - Buildings (office, retail, warehouse, industrial, data center, hotel,
      healthcare, education - 8 types x 5 climate zones)
    - Vehicles (small/medium/large car, SUV, light/heavy van, light/heavy
      truck - 8 types x 7 fuel types)
    - Equipment (manufacturing, construction, generator, agricultural,
      mining, HVAC - 6 types with load factors)
    - IT Assets (server, network switch, storage, desktop, laptop,
      printer, copier - 7 types with PUE adjustment)

Calculation Methods:
    - Asset-specific (metered energy data per leased asset)
    - Lessor-specific (primary data from lessor/landlord)
    - Average-data (benchmark EUI/energy intensity by asset type)
    - Spend-based (EEIO factors with CPI deflation)

Frameworks:
    - GHG Protocol Scope 3 Category 8
    - ISO 14064-1:2018
    - CSRD ESRS E1
    - CDP Climate Change
    - SBTi Corporate Net-Zero
    - SB 253 (California Climate Disclosure)
    - GRI 305

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

__all__ = [
    "UpstreamLeasedDatabaseEngine",
    "BuildingCalculatorEngine",
    "VehicleFleetCalculatorEngine",
    "EquipmentCalculatorEngine",
    "ITAssetsCalculatorEngine",
    "ComplianceCheckerEngine",
    "UpstreamLeasedPipelineEngine",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "get_config",
]

AGENT_ID: str = "GL-MRV-S3-008"
AGENT_COMPONENT: str = "AGENT-MRV-021"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_ula_"

# Graceful imports - each engine with try/except
try:
    from greenlang.agents.mrv.upstream_leased_assets.upstream_leased_database import (
        UpstreamLeasedDatabaseEngine,
    )
except ImportError:
    UpstreamLeasedDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.upstream_leased_assets.building_calculator import (
        BuildingCalculatorEngine,
    )
except ImportError:
    BuildingCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.upstream_leased_assets.vehicle_fleet_calculator import (
        VehicleFleetCalculatorEngine,
    )
except ImportError:
    VehicleFleetCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.upstream_leased_assets.equipment_calculator import (
        EquipmentCalculatorEngine,
    )
except ImportError:
    EquipmentCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.upstream_leased_assets.it_assets_calculator import (
        ITAssetsCalculatorEngine,
    )
except ImportError:
    ITAssetsCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.upstream_leased_assets.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.upstream_leased_assets.upstream_leased_pipeline import (
        UpstreamLeasedPipelineEngine,
    )
except ImportError:
    UpstreamLeasedPipelineEngine = None  # type: ignore[assignment,misc]

# Export configuration helper
try:
    from greenlang.agents.mrv.upstream_leased_assets.config import get_config
except ImportError:
    def get_config():  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None
