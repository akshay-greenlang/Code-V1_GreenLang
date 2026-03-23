# -*- coding: utf-8 -*-
"""
GreenLang Fuel & Energy Activities Agent - AGENT-MRV-016 (GL-MRV-S3-003)

Scope 3 Category 3 fuel and energy-related activities emission estimation
covering four sub-activities (3a upstream fuels WTT, 3b upstream electricity,
3c T&D losses, 3d utility resale), WTT emission factor database,
upstream fuel calculator, upstream electricity calculator, T&D loss calculator,
supplier-specific calculator, compliance checking against seven regulatory
frameworks, and pipeline orchestration.

Engines:
    1. WTTFuelDatabaseEngine              - WTT factors, fuel classification, unit conversion
    2. UpstreamFuelCalculatorEngine        - Activity 3a: fuel × WTT EF calculations
    3. UpstreamElectricityCalculatorEngine  - Activity 3b: electricity × upstream EF
    4. TDLossCalculatorEngine              - Activity 3c: T&D loss calculations
    5. SupplierSpecificCalculatorEngine    - Supplier-specific upstream data
    6. ComplianceCheckerEngine             - 7-framework regulatory compliance
    7. FuelEnergyPipelineEngine            - 10-stage pipeline orchestration

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-016 Fuel & Energy Activities (GL-MRV-S3-003)
Status: Production Ready
"""

from __future__ import annotations

__all__ = [
    "WTTFuelDatabaseEngine",
    "UpstreamFuelCalculatorEngine",
    "UpstreamElectricityCalculatorEngine",
    "TDLossCalculatorEngine",
    "SupplierSpecificCalculatorEngine",
    "ComplianceCheckerEngine",
    "FuelEnergyPipelineEngine",
]

VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Graceful engine imports - engines may not yet be implemented
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.fuel_energy_activities.wtt_fuel_database import (
        WTTFuelDatabaseEngine,
    )
except ImportError:
    WTTFuelDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.fuel_energy_activities.upstream_fuel_calculator import (
        UpstreamFuelCalculatorEngine,
    )
except ImportError:
    UpstreamFuelCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.fuel_energy_activities.upstream_electricity_calculator import (
        UpstreamElectricityCalculatorEngine,
    )
except ImportError:
    UpstreamElectricityCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.fuel_energy_activities.td_loss_calculator import (
        TDLossCalculatorEngine,
    )
except ImportError:
    TDLossCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.fuel_energy_activities.supplier_specific_calculator import (
        SupplierSpecificCalculatorEngine,
    )
except ImportError:
    SupplierSpecificCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.fuel_energy_activities.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.fuel_energy_activities.fuel_energy_pipeline import (
        FuelEnergyPipelineEngine,
    )
except ImportError:
    FuelEnergyPipelineEngine = None  # type: ignore[assignment,misc]
