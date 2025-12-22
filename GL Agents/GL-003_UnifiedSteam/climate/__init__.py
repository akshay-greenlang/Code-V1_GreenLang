"""
GL-003 UNIFIEDSTEAM - Climate Intelligence Module

This module provides climate impact calculations, emissions factor management,
measurement and verification (M&V) methodology, and sustainability reporting
for steam system optimization recommendations.

Key Features:
    - Emission factor database (fuel-specific, grid-specific, regional)
    - M&V methodology aligned with IPMVP standards
    - CO2e calculations with Scope 1/2/3 breakdown
    - Baseline normalization for production and ambient conditions
    - Savings attribution using causal layer outputs
    - Uncertainty quantification for climate metrics
    - Audit-ready reporting with complete provenance

Reference Standards:
    - IPMVP: International Performance Measurement and Verification Protocol
    - GHG Protocol: Corporate Standard and Scope 3 guidance
    - ISO 14064: Greenhouse gases quantification and reporting
    - EPA emission factors and methodologies

Author: GL-003 Climate Intelligence Team
Version: 1.0.0
"""

from .emission_factors import (
    EmissionFactorDatabase,
    EmissionFactor,
    FuelType,
    GridRegion,
    EmissionScope,
)

from .m_and_v import (
    MVMethodology,
    BaselineManager,
    NormalizationEngine,
    SavingsCalculator,
    MVReport,
    MVOption,
)

from .co2e_calculator import (
    CO2eCalculator,
    ClimateImpactResult,
    EmissionsBreakdown,
    SteamCarbonIntensity,
    FuelConsumptionEstimate,
)

from .climate_reporter import (
    ClimateReporter,
    ClimateImpactSummary,
    SustainabilityDashboard,
    ComplianceReport,
)

from .savings_attribution import (
    SavingsAttributor,
    AttributedSavings,
    CausalSavingsLink,
    InterventionImpact,
)

__all__ = [
    # Emission Factors
    "EmissionFactorDatabase",
    "EmissionFactor",
    "FuelType",
    "GridRegion",
    "EmissionScope",
    # M&V
    "MVMethodology",
    "BaselineManager",
    "NormalizationEngine",
    "SavingsCalculator",
    "MVReport",
    "MVOption",
    # CO2e Calculator
    "CO2eCalculator",
    "ClimateImpactResult",
    "EmissionsBreakdown",
    "SteamCarbonIntensity",
    "FuelConsumptionEstimate",
    # Climate Reporter
    "ClimateReporter",
    "ClimateImpactSummary",
    "SustainabilityDashboard",
    "ComplianceReport",
    # Savings Attribution
    "SavingsAttributor",
    "AttributedSavings",
    "CausalSavingsLink",
    "InterventionImpact",
]
