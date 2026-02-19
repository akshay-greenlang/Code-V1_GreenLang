# -*- coding: utf-8 -*-
"""
GreenLang Land Use Emissions Agent - AGENT-MRV-006 (GL-MRV-SCOPE1-006)

Scope 1 land use, land-use change, and forestry (LULUCF) emission estimation
covering carbon stock changes across five carbon pools (above-ground biomass,
below-ground biomass, dead wood, litter, soil organic carbon), land-use
transitions between six IPCC land categories, soil organic carbon assessment,
fire and disturbance emissions, peatland emissions, and Monte Carlo
uncertainty quantification.

Engines:
    1. LandUseDatabaseEngine       - Land categories, emission factors, carbon stock defaults
    2. CarbonStockCalculatorEngine  - Stock-difference and gain-loss carbon calculations
    3. LandUseChangeTrackerEngine   - Land-use transition tracking and conversion emissions
    4. SoilOrganicCarbonEngine      - SOC reference stocks, land use/management/input factors
    5. UncertaintyQuantifierEngine   - Monte Carlo uncertainty and confidence intervals
    6. ComplianceCheckerEngine       - Multi-framework regulatory compliance validation
    7. LandUsePipelineEngine         - End-to-end orchestration of all engines

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Land Use Emissions (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

__all__ = [
    "LandUseDatabaseEngine",
    "CarbonStockCalculatorEngine",
    "LandUseChangeTrackerEngine",
    "SoilOrganicCarbonEngine",
    "UncertaintyQuantifierEngine",
    "ComplianceCheckerEngine",
    "LandUsePipelineEngine",
]

VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Graceful engine imports - engines may not yet be implemented
# ---------------------------------------------------------------------------

try:
    from greenlang.land_use_emissions.land_use_database import (
        LandUseDatabaseEngine,
    )
except ImportError:
    LandUseDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.land_use_emissions.carbon_stock_calculator import (
        CarbonStockCalculatorEngine,
    )
except ImportError:
    CarbonStockCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.land_use_emissions.land_use_change_tracker import (
        LandUseChangeTrackerEngine,
    )
except ImportError:
    LandUseChangeTrackerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.land_use_emissions.soil_organic_carbon import (
        SoilOrganicCarbonEngine,
    )
except ImportError:
    SoilOrganicCarbonEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.land_use_emissions.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.land_use_emissions.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.land_use_emissions.land_use_pipeline import (
        LandUsePipelineEngine,
    )
except ImportError:
    LandUsePipelineEngine = None  # type: ignore[assignment,misc]
