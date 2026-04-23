"""
GL-012 STEAMQUAL - Calculators Module

Zero-hallucination calculation engines for steam quality management:
- Dryness fraction calculations (enthalpy, entropy, volume, throttling methods)
- Moisture carryover risk assessment
- Separator/scrubber efficiency modeling
- Moisture removal through drains and traps

All calculators implement:
- Deterministic formulas with zero hallucination
- SHA-256 provenance hashing
- Uncertainty bounds where applicable
- ASME and engineering standards compliance

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from .dryness_fraction_calculator import (
    DrynessFractionCalculator,
    DrynessFractionResult,
    ThrottlingResult,
    CombinedQualityResult,
    SteamConditions,
    SaturationProperties,
    CalculationMethod,
    QualityGrade,
    # Constants
    ATMOSPHERIC_PRESSURE_KPA,
    CRITICAL_PRESSURE_KPA,
    CRITICAL_TEMP_C,
    ENTHALPY_METHOD_UNCERTAINTY,
    ENTROPY_METHOD_UNCERTAINTY,
    VOLUME_METHOD_UNCERTAINTY,
    THROTTLING_METHOD_UNCERTAINTY,
)

from .carryover_risk_calculator import (
    CarryoverRiskCalculator,
    DrumLevelRiskResult,
    LoadSwingRiskResult,
    FoamingRiskResult,
    DropletEntrainmentRiskResult,
    TotalCarryoverRiskResult,
    DrumLevelData,
    LoadSwingData,
    WaterChemistryData,
    DropletEntrainmentData,
    RiskLevel,
    CarryoverType,
    # Constants
    DRUM_LEVEL_NORMAL,
    DRUM_LEVEL_HIGH_ALARM,
    DRUM_LEVEL_TRIP,
    LOAD_CHANGE_NORMAL,
    LOAD_CHANGE_MODERATE,
    LOAD_CHANGE_HIGH,
    TDS_LIMIT_LOW_PRESSURE,
    TDS_LIMIT_HIGH_PRESSURE,
)

from .separator_efficiency_calculator import (
    SeparatorEfficiencyCalculator,
    MassBalanceResult,
    EfficiencyEstimateResult,
    CapacityAnalysisResult,
    DropletSeparationResult,
    SeparatorPerformanceReport,
    SeparatorSpecs,
    SeparatorOperatingData,
    SeparatorType,
    OperatingStatus,
    MaintenanceStatus,
    # Constants
    SEPARATOR_EFFICIENCY_RANGES,
    SEPARATOR_VELOCITY_LIMITS,
    SEPARATOR_K_FACTORS,
    SEPARATOR_CUTOFF_SIZES,
)

from .moisture_removal_calculator import (
    MoistureRemovalCalculator,
    DrainCapacityResult,
    TrapSizingResult,
    FlashSteamResult,
    RemovalEfficiencyResult,
    EnergyLossResult,
    MoistureRemovalReport,
    DrainSpecs,
    DrainOperatingData,
    CondensateLoadData,
    TrapType,
    DrainCondition,
    RemovalStatus,
    # Constants
    TRAP_TYPES,
    SAFETY_FACTOR_STARTUP,
    SAFETY_FACTOR_NORMAL,
)

__version__ = "1.0.0"
__author__ = "GL-BackendDeveloper"

__all__ = [
    # =========================================================================
    # Dryness Fraction Calculator
    # =========================================================================
    "DrynessFractionCalculator",
    "DrynessFractionResult",
    "ThrottlingResult",
    "CombinedQualityResult",
    "SteamConditions",
    "SaturationProperties",
    "CalculationMethod",
    "QualityGrade",
    # Constants
    "ATMOSPHERIC_PRESSURE_KPA",
    "CRITICAL_PRESSURE_KPA",
    "CRITICAL_TEMP_C",
    "ENTHALPY_METHOD_UNCERTAINTY",
    "ENTROPY_METHOD_UNCERTAINTY",
    "VOLUME_METHOD_UNCERTAINTY",
    "THROTTLING_METHOD_UNCERTAINTY",

    # =========================================================================
    # Carryover Risk Calculator
    # =========================================================================
    "CarryoverRiskCalculator",
    "DrumLevelRiskResult",
    "LoadSwingRiskResult",
    "FoamingRiskResult",
    "DropletEntrainmentRiskResult",
    "TotalCarryoverRiskResult",
    "DrumLevelData",
    "LoadSwingData",
    "WaterChemistryData",
    "DropletEntrainmentData",
    "RiskLevel",
    "CarryoverType",
    # Constants
    "DRUM_LEVEL_NORMAL",
    "DRUM_LEVEL_HIGH_ALARM",
    "DRUM_LEVEL_TRIP",
    "LOAD_CHANGE_NORMAL",
    "LOAD_CHANGE_MODERATE",
    "LOAD_CHANGE_HIGH",
    "TDS_LIMIT_LOW_PRESSURE",
    "TDS_LIMIT_HIGH_PRESSURE",

    # =========================================================================
    # Separator Efficiency Calculator
    # =========================================================================
    "SeparatorEfficiencyCalculator",
    "MassBalanceResult",
    "EfficiencyEstimateResult",
    "CapacityAnalysisResult",
    "DropletSeparationResult",
    "SeparatorPerformanceReport",
    "SeparatorSpecs",
    "SeparatorOperatingData",
    "SeparatorType",
    "OperatingStatus",
    "MaintenanceStatus",
    # Constants
    "SEPARATOR_EFFICIENCY_RANGES",
    "SEPARATOR_VELOCITY_LIMITS",
    "SEPARATOR_K_FACTORS",
    "SEPARATOR_CUTOFF_SIZES",

    # =========================================================================
    # Moisture Removal Calculator
    # =========================================================================
    "MoistureRemovalCalculator",
    "DrainCapacityResult",
    "TrapSizingResult",
    "FlashSteamResult",
    "RemovalEfficiencyResult",
    "EnergyLossResult",
    "MoistureRemovalReport",
    "DrainSpecs",
    "DrainOperatingData",
    "CondensateLoadData",
    "TrapType",
    "DrainCondition",
    "RemovalStatus",
    # Constants
    "TRAP_TYPES",
    "SAFETY_FACTOR_STARTUP",
    "SAFETY_FACTOR_NORMAL",
]


def get_calculator_info() -> dict:
    """
    Get information about available calculators.

    Returns:
        Dictionary with calculator names, versions, and capabilities
    """
    return {
        "module": "GL-012_SteamQual.calculators",
        "version": __version__,
        "calculators": [
            {
                "name": "DrynessFractionCalculator",
                "version": DrynessFractionCalculator.VERSION,
                "description": "Steam quality (dryness fraction) calculations",
                "methods": [
                    "calculate_from_enthalpy",
                    "calculate_from_entropy",
                    "calculate_from_specific_volume",
                    "calculate_from_throttling_calorimeter",
                    "calculate_combined",
                    "get_saturation_properties",
                ],
            },
            {
                "name": "CarryoverRiskCalculator",
                "version": CarryoverRiskCalculator.VERSION,
                "description": "Moisture carryover risk assessment",
                "methods": [
                    "assess_drum_level_risk",
                    "assess_load_swing_risk",
                    "assess_foaming_risk",
                    "assess_droplet_entrainment_risk",
                    "assess_total_carryover_risk",
                ],
            },
            {
                "name": "SeparatorEfficiencyCalculator",
                "version": SeparatorEfficiencyCalculator.VERSION,
                "description": "Separator/scrubber performance modeling",
                "methods": [
                    "compute_mass_balance",
                    "estimate_efficiency_from_dp",
                    "analyze_capacity_constraints",
                    "calculate_droplet_separation",
                    "generate_performance_report",
                ],
            },
            {
                "name": "MoistureRemovalCalculator",
                "version": MoistureRemovalCalculator.VERSION,
                "description": "Drain/trap moisture removal calculations",
                "methods": [
                    "calculate_drain_capacity",
                    "calculate_trap_sizing",
                    "calculate_flash_steam",
                    "calculate_removal_efficiency",
                    "calculate_energy_loss",
                    "generate_removal_report",
                ],
            },
        ],
        "compliance": [
            "ASME PTC 19.11 (Steam Traps)",
            "ASME PTC 39 (Steam Traps)",
            "IAPWS-IF97 (Steam Properties)",
            "API Standard 521",
            "EPRI TR-102134 (Steam Purity)",
        ],
        "zero_hallucination": True,
        "provenance_tracking": "SHA-256",
    }
