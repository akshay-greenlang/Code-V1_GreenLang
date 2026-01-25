"""
GL-020 ECONOPULSE: Economizer Performance Monitoring Calculators

Zero-hallucination thermodynamic calculation engines for economizer
performance monitoring and fouling analysis.

This package provides comprehensive, deterministic calculations for:
- Heat transfer analysis (LMTD, U-value, NTU, effectiveness)
- Fouling factor calculation and trending
- Economizer efficiency metrics
- Thermal property calculations
- Soot blower optimization
- Complete provenance tracking

All calculations are:
- Deterministic (zero-hallucination guaranteed)
- Bit-perfect reproducible
- Fully auditable with SHA-256 provenance hashes
- Based on ASME PTC 4.3 and IAPWS-IF97 standards

Author: GL-CalculatorEngineer
Agent ID: GL-020
Codename: ECONOPULSE
Version: 1.1.0
Standards: ASME PTC 4.3, IAPWS-IF97, JANAF, TEMA
"""

from typing import Dict

__version__ = "1.1.0"
__agent_id__ = "GL-020"
__codename__ = "ECONOPULSE"
__author__ = "GL-CalculatorEngineer"

# =============================================================================
# PROVENANCE TRACKING
# =============================================================================
from .provenance import (
    # Classes
    CalculationType,
    CalculationStep,
    CalculationProvenance,
    ProvenanceTracker,
    AuditLogger,
    # Functions
    generate_calculation_hash,
    verify_calculation_reproducibility,
    get_default_tracker,
    get_default_logger,
)

# =============================================================================
# THERMAL PROPERTIES
# =============================================================================
from .thermal_properties import (
    # Water properties (IAPWS-IF97)
    get_water_cp,
    get_water_density,
    get_steam_cp,
    # Flue gas properties (JANAF)
    get_flue_gas_cp,
    get_flue_gas_density,
    DEFAULT_FLUE_GAS_COMPOSITION,
    # Tube material properties
    get_thermal_conductivity,
    TUBE_MATERIAL_CONDUCTIVITY,
    # Constants
    MOLECULAR_WEIGHTS,
    WATER_CRITICAL_TEMP_K,
    WATER_CRITICAL_PRESSURE_MPA,
    # Unit conversions
    fahrenheit_to_celsius,
    celsius_to_fahrenheit,
    fahrenheit_to_kelvin,
    kelvin_to_fahrenheit,
    psia_to_mpa,
    mpa_to_psia,
    btu_per_lb_f_to_kj_per_kg_k,
    kj_per_kg_k_to_btu_per_lb_f,
    # Validation
    ValidationError,
    validate_temperature_celsius,
    validate_temperature_fahrenheit,
    validate_pressure_psia,
    validate_composition,
)

# =============================================================================
# HEAT TRANSFER CALCULATIONS
# =============================================================================
from .heat_transfer_calculator import (
    # Enums
    FlowArrangement,
    # LMTD
    calculate_lmtd,
    # U-value
    calculate_overall_heat_transfer_coefficient,
    calculate_theoretical_u_value,
    # Heat duty
    calculate_heat_duty,
    calculate_water_side_heat_duty,
    calculate_gas_side_heat_duty,
    # Temperature differences
    calculate_approach_temperature,
    calculate_terminal_temperature_difference,
    # NTU-Effectiveness
    calculate_ntu,
    calculate_heat_capacity_ratio,
    calculate_effectiveness,
    calculate_effectiveness_from_temperatures,
)

# =============================================================================
# FOULING CALCULATIONS
# =============================================================================
from .fouling_calculator import (
    # Enums
    FoulingType,
    FoulingTrendModel,
    # Data classes
    FoulingDataPoint,
    # Fouling factor
    calculate_fouling_factor,
    calculate_fouling_factor_components,
    # Fouling rate
    calculate_fouling_rate,
    # Cleaning prediction
    predict_cleaning_time,
    predict_cleaning_date,
    # Cleanliness
    calculate_cleanliness_factor,
    # Efficiency loss
    calculate_efficiency_loss_from_fouling,
    # Fuel penalty
    estimate_fuel_penalty,
    # Assessment
    assess_fouling_severity,
    # Constants
    TEMA_FOULING_FACTORS,
    FOULING_THRESHOLDS,
)

# =============================================================================
# ECONOMIZER EFFICIENCY
# =============================================================================
from .economizer_efficiency_calculator import (
    # Data classes
    EconomizerOperatingPoint,
    DesignConditions,
    EfficiencyResult,
    # Effectiveness
    calculate_economizer_effectiveness,
    # Heat recovery
    calculate_heat_recovery_ratio,
    # Side-specific efficiency
    calculate_gas_side_efficiency,
    calculate_water_side_efficiency,
    # Overall efficiency
    calculate_overall_efficiency,
    # Design deviation
    calculate_design_deviation,
    # Comprehensive analysis
    calculate_comprehensive_efficiency,
    # Trending
    calculate_efficiency_trend,
)

# =============================================================================
# SOOT BLOWER OPTIMIZATION
# =============================================================================
from .soot_blower_optimizer import (
    # Enums
    SootBlowerType,
    BlowingMedium,
    EconomizerZone,
    # Data classes
    SootBlowerConfig,
    CleaningEvent,
    OptimizationResult,
    # Optimal interval
    calculate_optimal_cleaning_interval,
    calculate_dynamic_cleaning_interval,
    # Cleaning effectiveness
    calculate_cleaning_effectiveness,
    calculate_average_cleaning_effectiveness,
    # Media cost
    calculate_media_cost,
    calculate_total_cleaning_cost,
    # Sequence optimization
    optimize_soot_blower_sequence,
    # ROI
    calculate_cleaning_roi,
    calculate_annual_cleaning_economics,
    # Comprehensive optimization
    optimize_cleaning_program,
    # Constants
    DEFAULT_SOOT_BLOWER_PARAMS,
)
# =============================================================================
# ADVANCED FOULING ANALYSIS (NEW)
# =============================================================================
from .economizer_fouling_calculator import (
    # Enums
    FoulingSide,
    FoulingMechanism,
    CleaningMethod,
    TrendModel,
    # Data classes
    FoulingMeasurement,
    FoulingFactorResult,
    FoulingRateResult,
    HeatLossResult,
    FuelPenaltyResult,
    CarbonPenaltyResult,
    CleaningComparisonResult,
    CleaningIntervalResult,
    # Core calculations
    calculate_fouling_factor_from_u_values,
    calculate_cleanliness_trend,
    predict_fouling_rate,
    # Economic impact
    calculate_heat_loss_from_fouling,
    calculate_fuel_penalty,
    calculate_carbon_penalty,
    # Cleaning analysis
    compare_cleaning_effectiveness,
    optimize_cleaning_interval,
    # Constants
    ASME_REFERENCE_CONDITIONS,
    TEMA_FOULING_FACTORS as ADVANCED_TEMA_FOULING_FACTORS,
    FOULING_SEVERITY_THRESHOLDS,
    # Utility
    clear_calculation_cache,
)

# =============================================================================
# ADVANCED SOOT BLOWER OPTIMIZATION (NEW)
# =============================================================================
from .advanced_soot_blower_optimizer import (
    # Enums
    BlowerType,
    BlowingMedium as AdvancedBlowingMedium,
    EconomizerZone as AdvancedEconomizerZone,
    CleaningPriority,
    WearSeverity,
    # Data classes
    SootBlowerConfiguration,
    ZoneFoulingState,
    BlowingIntervalResult,
    ZonePriorityResult,
    MediaConsumptionResult,
    CleaningEffectivenessResult,
    ROIAnalysisResult,
    ErosionMonitorResult,
    SequentialScheduleResult,
    EnergyBalanceResult,
    # Core calculations
    calculate_optimal_blowing_interval,
    prioritize_cleaning_zones,
    track_media_consumption,
    measure_cleaning_effectiveness,
    # Economic analysis
    analyze_cleaning_roi,
    # Equipment monitoring
    monitor_erosion_wear,
    # Schedule optimization
    optimize_blowing_sequence,
    # Energy analysis
    calculate_soot_blowing_energy_balance,
    # Constants
    EPRI_GUIDELINES,
    ZONE_IMPORTANCE_WEIGHTS,
    STEAM_ENERGY_CONTENT,
    EROSION_RATES,
    # Utility
    clear_optimizer_cache,
)


# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # Version info
    "__version__",
    "__agent_id__",
    "__codename__",
    "__author__",

    # Provenance
    "CalculationType",
    "CalculationStep",
    "CalculationProvenance",
    "ProvenanceTracker",
    "AuditLogger",
    "generate_calculation_hash",
    "verify_calculation_reproducibility",
    "get_default_tracker",
    "get_default_logger",

    # Thermal Properties
    "get_water_cp",
    "get_water_density",
    "get_steam_cp",
    "get_flue_gas_cp",
    "get_flue_gas_density",
    "get_thermal_conductivity",
    "DEFAULT_FLUE_GAS_COMPOSITION",
    "TUBE_MATERIAL_CONDUCTIVITY",
    "MOLECULAR_WEIGHTS",
    "WATER_CRITICAL_TEMP_K",
    "WATER_CRITICAL_PRESSURE_MPA",
    "fahrenheit_to_celsius",
    "celsius_to_fahrenheit",
    "fahrenheit_to_kelvin",
    "kelvin_to_fahrenheit",
    "psia_to_mpa",
    "mpa_to_psia",
    "btu_per_lb_f_to_kj_per_kg_k",
    "kj_per_kg_k_to_btu_per_lb_f",
    "ValidationError",
    "validate_temperature_celsius",
    "validate_temperature_fahrenheit",
    "validate_pressure_psia",
    "validate_composition",

    # Heat Transfer
    "FlowArrangement",
    "calculate_lmtd",
    "calculate_overall_heat_transfer_coefficient",
    "calculate_theoretical_u_value",
    "calculate_heat_duty",
    "calculate_water_side_heat_duty",
    "calculate_gas_side_heat_duty",
    "calculate_approach_temperature",
    "calculate_terminal_temperature_difference",
    "calculate_ntu",
    "calculate_heat_capacity_ratio",
    "calculate_effectiveness",
    "calculate_effectiveness_from_temperatures",

    # Fouling
    "FoulingType",
    "FoulingTrendModel",
    "FoulingDataPoint",
    "calculate_fouling_factor",
    "calculate_fouling_factor_components",
    "calculate_fouling_rate",
    "predict_cleaning_time",
    "predict_cleaning_date",
    "calculate_cleanliness_factor",
    "calculate_efficiency_loss_from_fouling",
    "estimate_fuel_penalty",
    "assess_fouling_severity",
    "TEMA_FOULING_FACTORS",
    "FOULING_THRESHOLDS",

    # Efficiency
    "EconomizerOperatingPoint",
    "DesignConditions",
    "EfficiencyResult",
    "calculate_economizer_effectiveness",
    "calculate_heat_recovery_ratio",
    "calculate_gas_side_efficiency",
    "calculate_water_side_efficiency",
    "calculate_overall_efficiency",
    "calculate_design_deviation",
    "calculate_comprehensive_efficiency",
    "calculate_efficiency_trend",

    # Soot Blower
    "SootBlowerType",
    "BlowingMedium",
    "EconomizerZone",
    "SootBlowerConfig",
    "CleaningEvent",
    "OptimizationResult",
    "calculate_optimal_cleaning_interval",
    "calculate_dynamic_cleaning_interval",
    "calculate_cleaning_effectiveness",
    "calculate_average_cleaning_effectiveness",
    "calculate_media_cost",
    "calculate_total_cleaning_cost",
    "optimize_soot_blower_sequence",
    "calculate_cleaning_roi",
    "calculate_annual_cleaning_economics",
    "optimize_cleaning_program",
    "DEFAULT_SOOT_BLOWER_PARAMS",
    # Advanced Fouling Analysis (NEW)
    "FoulingSide",
    "FoulingMechanism",
    "CleaningMethod",
    "TrendModel",
    "FoulingMeasurement",
    "FoulingFactorResult",
    "FoulingRateResult",
    "HeatLossResult",
    "FuelPenaltyResult",
    "CarbonPenaltyResult",
    "CleaningComparisonResult",
    "CleaningIntervalResult",
    "calculate_fouling_factor_from_u_values",
    "calculate_cleanliness_trend",
    "predict_fouling_rate",
    "calculate_heat_loss_from_fouling",
    "calculate_fuel_penalty",
    "calculate_carbon_penalty",
    "compare_cleaning_effectiveness",
    "optimize_cleaning_interval",
    "ASME_REFERENCE_CONDITIONS",
    "ADVANCED_TEMA_FOULING_FACTORS",
    "FOULING_SEVERITY_THRESHOLDS",
    "clear_calculation_cache",

    # Advanced Soot Blower Optimization (NEW)
    "BlowerType",
    "AdvancedBlowingMedium",
    "AdvancedEconomizerZone",
    "CleaningPriority",
    "WearSeverity",
    "SootBlowerConfiguration",
    "ZoneFoulingState",
    "BlowingIntervalResult",
    "ZonePriorityResult",
    "MediaConsumptionResult",
    "CleaningEffectivenessResult",
    "ROIAnalysisResult",
    "ErosionMonitorResult",
    "SequentialScheduleResult",
    "EnergyBalanceResult",
    "calculate_optimal_blowing_interval",
    "prioritize_cleaning_zones",
    "track_media_consumption",
    "measure_cleaning_effectiveness",
    "analyze_cleaning_roi",
    "monitor_erosion_wear",
    "optimize_blowing_sequence",
    "calculate_soot_blowing_energy_balance",
    "EPRI_GUIDELINES",
    "ZONE_IMPORTANCE_WEIGHTS",
    "STEAM_ENERGY_CONTENT",
    "EROSION_RATES",
    "clear_optimizer_cache",

]


# =============================================================================
# MODULE VALIDATION
# =============================================================================
def validate_module_integrity() -> Dict[str, bool]:
    """
    Validate that all module components are properly loaded.

    Returns:
        Dictionary mapping component names to load status
    """
    validation_results = {}

    # Check provenance module
    try:
        from . import provenance
        validation_results["provenance"] = True
    except ImportError:
        validation_results["provenance"] = False

    # Check thermal properties module
    try:
        from . import thermal_properties
        validation_results["thermal_properties"] = True
    except ImportError:
        validation_results["thermal_properties"] = False

    # Check heat transfer module
    try:
        from . import heat_transfer_calculator
        validation_results["heat_transfer_calculator"] = True
    except ImportError:
        validation_results["heat_transfer_calculator"] = False

    # Check fouling module
    try:
        from . import fouling_calculator
        validation_results["fouling_calculator"] = True
    except ImportError:
        validation_results["fouling_calculator"] = False

    # Check efficiency module
    try:
        from . import economizer_efficiency_calculator
        validation_results["economizer_efficiency_calculator"] = True
    except ImportError:
        validation_results["economizer_efficiency_calculator"] = False

    # Check soot blower module
    try:
        from . import soot_blower_optimizer
        validation_results["soot_blower_optimizer"] = True
    except ImportError:
        validation_results["soot_blower_optimizer"] = False

    # Check advanced fouling module (NEW)
    try:
        from . import economizer_fouling_calculator
        validation_results["economizer_fouling_calculator"] = True
    except ImportError:
        validation_results["economizer_fouling_calculator"] = False

    # Check advanced soot blower module (NEW)
    try:
        from . import advanced_soot_blower_optimizer
        validation_results["advanced_soot_blower_optimizer"] = True
    except ImportError:
        validation_results["advanced_soot_blower_optimizer"] = False


    return validation_results


def get_module_info() -> Dict[str, str]:
    """
    Get module information.

    Returns:
        Dictionary with module metadata
    """
    return {
        "version": __version__,
        "agent_id": __agent_id__,
        "codename": __codename__,
        "author": __author__,
        "standards": "ASME PTC 4.3, IAPWS-IF97, JANAF, TEMA, EPRI",
        "description": "Zero-hallucination economizer performance monitoring calculators",
        "guarantee": "All calculations are deterministic, reproducible, and auditable",
        "modules": [
            "provenance",
            "thermal_properties",
            "heat_transfer_calculator",
            "fouling_calculator",
            "economizer_efficiency_calculator",
            "soot_blower_optimizer",
            "economizer_fouling_calculator",
            "advanced_soot_blower_optimizer",
        ]
    }
