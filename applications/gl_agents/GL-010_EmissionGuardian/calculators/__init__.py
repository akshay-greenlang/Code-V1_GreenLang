'''
EPA 40 CFR Part 75 Compliance Calculators

This module provides zero-hallucination calculation engines for EPA Part 75
CEMS compliance. All calculations are:
- 100% deterministic (same input produces same output)
- Use Decimal for precision (no floating-point errors)
- Include SHA-256 provenance hash for audit trail
- Include calculation trace for explainability
- Follow EPA methodologies exactly

Modules:
    emission_rate: Core emission rate calculations (ppm to lb/MMBtu, O2 correction, etc.)
    rata_calculator: RATA statistical calculations (relative accuracy, bias test)
    data_substitution: EPA Appendix D substitute data procedures
    averaging: Averaging period calculations (hourly, rolling, quarterly, annual)

Author: GL-CalculatorEngineer
Reference: 40 CFR Part 75
'''

# Emission Rate Calculations
from .emission_rate import (
    # Data classes
    CalculationTrace,
    EmissionRateResult,
    
    # Constants
    FuelType,
    F_FACTORS_O2,
    F_FACTORS_CO2,
    MOLECULAR_WEIGHTS,
    REFERENCE_O2_PERCENT,
    MOLAR_VOLUME_DSCF,
    
    # Core functions
    ppm_to_lb_per_mmbtu,
    o2_correction_factor,
    wet_to_dry_correction,
    mass_emission_rate,
    heat_input_rate,
    lb_per_mmbtu_to_lb_per_hr,
    calculate_total_emissions,
    get_f_factor,
    get_molecular_weight,
    calculate_provenance_hash,
)

# RATA Calculations
from .rata_calculator import (
    # Data classes
    RATAResult,
    
    # Constants
    T_VALUES_95,
    RATA_PASS_THRESHOLD_STANDARD,
    RATA_PASS_THRESHOLD_ABBREVIATED,
    
    # Core functions
    calculate_mean_difference,
    calculate_standard_deviation,
    calculate_confidence_coefficient,
    calculate_relative_accuracy,
    calculate_bias_test,
    calculate_bias_adjustment_factor,
    perform_rata,
)

# Data Substitution
from .data_substitution import (
    # Data classes
    SubstituteDataResult,
    DataAvailabilityResult,
    
    # Constants
    SubstitutionTier,
    TIER_1_MAX_HOURS,
    TIER_2_MAX_HOURS,
    LOOKBACK_PERIOD_HOURS,
    STANDARD_PERCENTILE,
    
    # Core functions
    calculate_percentile,
    determine_substitution_tier,
    calculate_standard_substitute,
    calculate_maximum_substitute,
    get_substitute_data,
    calculate_data_availability,
    get_lookback_period_values,
    count_consecutive_missing_hours,
)

# Averaging Calculations
from .averaging import (
    # Data classes
    AveragingResult,
    RollingAverageResult,
    
    # Constants
    AveragingPeriod,
    MIN_DATA_COMPLETENESS_HOUR,
    MIN_DATA_COMPLETENESS_DAY,
    MIN_DATA_COMPLETENESS_QUARTER,
    HOURS_PER_DAY,
    HOURS_PER_QUARTER,
    HOURS_PER_YEAR,
    
    # Core functions
    calculate_data_completeness,
    is_hour_valid,
    is_day_valid,
    calculate_hourly_average,
    calculate_daily_average,
    calculate_rolling_average,
    calculate_rolling_3_hour_average,
    calculate_rolling_24_hour_average,
    calculate_rolling_30_day_average,
    calculate_quarterly_average,
    calculate_annual_average,
    calculate_total_emissions_for_period,
)


__all__ = [
    # Emission Rate
    'CalculationTrace',
    'EmissionRateResult',
    'FuelType',
    'F_FACTORS_O2',
    'F_FACTORS_CO2',
    'MOLECULAR_WEIGHTS',
    'REFERENCE_O2_PERCENT',
    'MOLAR_VOLUME_DSCF',
    'ppm_to_lb_per_mmbtu',
    'o2_correction_factor',
    'wet_to_dry_correction',
    'mass_emission_rate',
    'heat_input_rate',
    'lb_per_mmbtu_to_lb_per_hr',
    'calculate_total_emissions',
    'get_f_factor',
    'get_molecular_weight',
    'calculate_provenance_hash',
    
    # RATA
    'RATAResult',
    'T_VALUES_95',
    'RATA_PASS_THRESHOLD_STANDARD',
    'RATA_PASS_THRESHOLD_ABBREVIATED',
    'calculate_mean_difference',
    'calculate_standard_deviation',
    'calculate_confidence_coefficient',
    'calculate_relative_accuracy',
    'calculate_bias_test',
    'calculate_bias_adjustment_factor',
    'perform_rata',
    
    # Data Substitution
    'SubstituteDataResult',
    'DataAvailabilityResult',
    'SubstitutionTier',
    'TIER_1_MAX_HOURS',
    'TIER_2_MAX_HOURS',
    'LOOKBACK_PERIOD_HOURS',
    'STANDARD_PERCENTILE',
    'calculate_percentile',
    'determine_substitution_tier',
    'calculate_standard_substitute',
    'calculate_maximum_substitute',
    'get_substitute_data',
    'calculate_data_availability',
    'get_lookback_period_values',
    'count_consecutive_missing_hours',
    
    # Averaging
    'AveragingResult',
    'RollingAverageResult',
    'AveragingPeriod',
    'MIN_DATA_COMPLETENESS_HOUR',
    'MIN_DATA_COMPLETENESS_DAY',
    'MIN_DATA_COMPLETENESS_QUARTER',
    'HOURS_PER_DAY',
    'HOURS_PER_QUARTER',
    'HOURS_PER_YEAR',
    'calculate_data_completeness',
    'is_hour_valid',
    'is_day_valid',
    'calculate_hourly_average',
    'calculate_daily_average',
    'calculate_rolling_average',
    'calculate_rolling_3_hour_average',
    'calculate_rolling_24_hour_average',
    'calculate_rolling_30_day_average',
    'calculate_quarterly_average',
    'calculate_annual_average',
    'calculate_total_emissions_for_period',
]

__version__ = '1.0.0'
__author__ = 'GL-CalculatorEngineer'
__reference__ = '40 CFR Part 75'
