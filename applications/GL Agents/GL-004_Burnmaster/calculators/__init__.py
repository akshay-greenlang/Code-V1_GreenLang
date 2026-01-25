"""
GL-004 BURNMASTER Calculators Module

Zero-hallucination calculation engine for burner optimization.
All calculations are deterministic, auditable, and bit-perfect reproducible.

This module provides specialized calculators for:
- Air-fuel ratio optimization
- Flame stability analysis
- Emissions estimation and compliance
- Turndown ratio management
- Key performance indicators (KPIs)

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from calculators.air_fuel_ratio_calculator import (
    AirFuelRatioCalculator,
    StabilityMetrics,
    RatioCalculationInput,
    RatioCalculationResult,
    O2TrimInput,
    O2TrimResult,
)

from calculators.stability_calculator import (
    FlameStabilityCalculator,
    OscillationResult,
    StabilityInput,
    StabilityResult,
    BlowoffRiskInput,
    FlashbackRiskInput,
    StabilityMarginResult,
)

from calculators.emissions_calculator import (
    EmissionsCalculator,
    ComplianceResult,
    NOxEstimateInput,
    COEstimateInput,
    EmissionRateInput,
    EmissionRateResult,
    RollingAverageResult,
)

from calculators.turndown_calculator import (
    TurndownCalculator,
    ValidationResult,
    StagingPlan,
    MinimumLoadInput,
    TurndownSetpointInput,
    StagingInput,
)

from calculators.kpi_calculator import (
    BurnerKPICalculator,
    ContributionMetrics,
    KPIDashboard,
    CombustionData,
    FuelIntensityInput,
    ThermalEfficiencyInput,
    AvailabilityInput,
)

__all__ = [
    # Air-Fuel Ratio Calculator
    "AirFuelRatioCalculator",
    "StabilityMetrics",
    "RatioCalculationInput",
    "RatioCalculationResult",
    "O2TrimInput",
    "O2TrimResult",
    # Flame Stability Calculator
    "FlameStabilityCalculator",
    "OscillationResult",
    "StabilityInput",
    "StabilityResult",
    "BlowoffRiskInput",
    "FlashbackRiskInput",
    "StabilityMarginResult",
    # Emissions Calculator
    "EmissionsCalculator",
    "ComplianceResult",
    "NOxEstimateInput",
    "COEstimateInput",
    "EmissionRateInput",
    "EmissionRateResult",
    "RollingAverageResult",
    # Turndown Calculator
    "TurndownCalculator",
    "ValidationResult",
    "StagingPlan",
    "MinimumLoadInput",
    "TurndownSetpointInput",
    "StagingInput",
    # KPI Calculator
    "BurnerKPICalculator",
    "ContributionMetrics",
    "KPIDashboard",
    "CombustionData",
    "FuelIntensityInput",
    "ThermalEfficiencyInput",
    "AvailabilityInput",
]

__version__ = "1.0.0"
__author__ = "GL-CalculatorEngineer"
