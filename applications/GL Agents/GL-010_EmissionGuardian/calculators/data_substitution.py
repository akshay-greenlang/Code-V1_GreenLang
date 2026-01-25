'''
EPA 40 CFR Part 75 Data Substitution Calculator

This module implements EPA Appendix D substitute data procedures for
missing CEMS data. All calculations are:
- 100% deterministic (no random elements)
- Use Decimal for precision
- Include SHA-256 provenance hash for audit trail
- Include calculation trace for explainability

Reference: 40 CFR Part 75, Appendix D and Subpart D

Author: GL-CalculatorEngineer
'''

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from datetime import datetime, timedelta


# ==============================================================================
# CONSTANTS - EPA Data Substitution Rules
# ==============================================================================

class SubstitutionTier(Enum):
    '''Data substitution tiers per 40 CFR Part 75.'''
    TIER_1 = 'tier_1'  # Hours 1-720: 90th percentile
    TIER_2 = 'tier_2'  # Hours 721-2160: Maximum value
    TIER_3 = 'tier_3'  # Hours 2161+: Maximum or 200%

# Hour thresholds for substitution tiers
TIER_1_MAX_HOURS = 720
TIER_2_MAX_HOURS = 2160

# Data availability thresholds
MIN_DATA_AVAILABILITY = Decimal('90.0')  # 90% minimum
LOOKBACK_PERIOD_HOURS = 2160  # 90 days * 24 hours

# Percentile for standard substitute data
STANDARD_PERCENTILE = Decimal('90')


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class CalculationTrace:
    step_number: int
    description: str
    formula: str
    inputs: Dict[str, str]
    output: str
    output_value: Any
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SubstituteDataResult:
    '''Result of substitute data calculation.'''
    substitute_value: Decimal
    unit: str
    substitution_tier: SubstitutionTier
    method_used: str
    lookback_hours: int
    valid_data_count: int
    missing_hour_number: int
    calculation_trace: List[CalculationTrace]
    provenance_hash: str
    inputs: Dict[str, Any]
    formula_reference: str
    is_valid: bool = True
    validation_messages: List[str] = field(default_factory=list)


@dataclass
class DataAvailabilityResult:
    '''Result of data availability calculation.'''
    availability_percent: Decimal
    valid_hours: int
    total_hours: int
    meets_minimum: bool
    calculation_trace: List[CalculationTrace]
    provenance_hash: str
    inputs: Dict[str, Any]
    formula_reference: str


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def _decimal_str(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    elif isinstance(value, float):
        return str(Decimal(str(value)))
    elif isinstance(value, dict):
        return json.dumps({k: _decimal_str(v) for k, v in sorted(value.items())})
    elif isinstance(value, list):
        return json.dumps([_decimal_str(v) for v in value])
    elif isinstance(value, Enum):
        return value.value
    else:
        return str(value)


def calculate_provenance_hash(
    function_name: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    calculation_trace: List[CalculationTrace]
) -> str:
    provenance_data = {
        'function': function_name,
        'inputs': {k: _decimal_str(v) for k, v in sorted(inputs.items())},
        'outputs': {k: _decimal_str(v) for k, v in sorted(outputs.items())},
        'trace_steps': len(calculation_trace),
        'trace_checksums': [
            hashlib.sha256(
                f'{t.step_number}:{t.description}:{t.output_value}'.encode()
            ).hexdigest()[:16]
            for t in calculation_trace
        ]
    }
    provenance_str = json.dumps(provenance_data, sort_keys=True)
    return hashlib.sha256(provenance_str.encode()).hexdigest()


def _apply_precision(value: Decimal, decimal_places: int) -> Decimal:
    if decimal_places == 0:
        return value.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    quantize_str = '0.' + '0' * decimal_places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ==============================================================================
# CORE SUBSTITUTION CALCULATIONS
# ==============================================================================

def calculate_percentile(
    values: List[Decimal],
    percentile: Decimal
) -> Decimal:
    '''
    Calculate percentile value using EPA method (linear interpolation).
    
    Per 40 CFR Part 75, uses sorted data with linear interpolation
    between adjacent values when percentile falls between data points.
    '''
    if not values:
        raise ValueError('Cannot calculate percentile of empty list')
    
    sorted_values = sorted([Decimal(str(v)) for v in values])
    n = len(sorted_values)
    
    # Calculate rank (1-based)
    rank = (percentile / Decimal('100')) * Decimal(str(n + 1))
    
    if rank <= 1:
        return sorted_values[0]
    elif rank >= n:
        return sorted_values[-1]
    
    # Linear interpolation
    lower_idx = int(rank) - 1
    upper_idx = lower_idx + 1
    fraction = rank - Decimal(str(int(rank)))
    
    lower_val = sorted_values[lower_idx]
    upper_val = sorted_values[upper_idx]
    
    result = lower_val + fraction * (upper_val - lower_val)
    return result


def determine_substitution_tier(
    consecutive_missing_hours: int
) -> SubstitutionTier:
    '''
    Determine which substitution tier applies based on consecutive missing hours.
    
    Per 40 CFR Part 75:
    - Tier 1: Hours 1-720 (standard substitute data)
    - Tier 2: Hours 721-2160 (maximum substitute data)
    - Tier 3: Hours 2161+ (maximum or 200% substitute)
    '''
    if consecutive_missing_hours <= TIER_1_MAX_HOURS:
        return SubstitutionTier.TIER_1
    elif consecutive_missing_hours <= TIER_2_MAX_HOURS:
        return SubstitutionTier.TIER_2
    else:
        return SubstitutionTier.TIER_3


def calculate_standard_substitute(
    lookback_values: List[Decimal],
    decimal_precision: int = 3
) -> Tuple[Decimal, str]:
    '''
    Calculate standard substitute data (90th percentile).
    
    Per 40 CFR Part 75, Appendix D, standard substitute data is
    the 90th percentile value from the lookback period.
    '''
    if not lookback_values:
        raise ValueError('No lookback values available')
    
    percentile_90 = calculate_percentile(lookback_values, STANDARD_PERCENTILE)
    return _apply_precision(percentile_90, decimal_precision), '90th percentile'


def calculate_maximum_substitute(
    lookback_values: List[Decimal],
    decimal_precision: int = 3
) -> Tuple[Decimal, str]:
    '''
    Calculate maximum substitute data.
    
    Per 40 CFR Part 75, maximum substitute data is the maximum
    value from the lookback period.
    '''
    if not lookback_values:
        raise ValueError('No lookback values available')
    
    max_value = max([Decimal(str(v)) for v in lookback_values])
    return _apply_precision(max_value, decimal_precision), 'Maximum value'


def get_substitute_data(
    lookback_values: List[Decimal],
    consecutive_missing_hours: int,
    unit: str = 'lb/hr',
    decimal_precision: int = 3
) -> SubstituteDataResult:
    '''
    Get appropriate substitute data value based on EPA Part 75 rules.
    
    Args:
        lookback_values: Historical valid data from lookback period
        consecutive_missing_hours: Number of consecutive hours of missing data
        unit: Unit of measurement
        decimal_precision: Decimal places for result
        
    Returns:
        SubstituteDataResult with calculated substitute value
    '''
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []
    
    # Convert values
    lookback_values = [Decimal(str(v)) for v in lookback_values]
    
    # Validation
    if not lookback_values:
        validation_messages.append('No lookback data available')
        raise ValueError('No lookback data available for substitution')
    
    # Step 1: Determine substitution tier
    tier = determine_substitution_tier(consecutive_missing_hours)
    calculation_trace.append(CalculationTrace(
        step_number=1, description='Determine substitution tier',
        formula='Based on consecutive missing hours',
        inputs={'consecutive_missing_hours': str(consecutive_missing_hours)},
        output='substitution_tier', output_value=tier.value))
    
    # Step 2: Calculate substitute value based on tier
    if tier == SubstitutionTier.TIER_1:
        substitute_value, method = calculate_standard_substitute(
            lookback_values, decimal_precision)
        calculation_trace.append(CalculationTrace(
            step_number=2, description='Calculate 90th percentile (Tier 1)',
            formula='P90 = value at rank (0.90 * (n+1))',
            inputs={'n_values': str(len(lookback_values))},
            output='substitute_value', output_value=substitute_value))
    
    elif tier == SubstitutionTier.TIER_2:
        substitute_value, method = calculate_maximum_substitute(
            lookback_values, decimal_precision)
        calculation_trace.append(CalculationTrace(
            step_number=2, description='Calculate maximum value (Tier 2)',
            formula='Max = max(lookback_values)',
            inputs={'n_values': str(len(lookback_values))},
            output='substitute_value', output_value=substitute_value))
    
    else:  # TIER_3
        max_value, _ = calculate_maximum_substitute(lookback_values, decimal_precision)
        # Tier 3: Use maximum or report as bias-adjusted
        substitute_value = max_value
        method = 'Maximum value (Tier 3 - extended outage)'
        calculation_trace.append(CalculationTrace(
            step_number=2, description='Calculate maximum value (Tier 3)',
            formula='Max = max(lookback_values)',
            inputs={'n_values': str(len(lookback_values))},
            output='substitute_value', output_value=substitute_value))
    
    # Build provenance
    inputs_dict = {
        'lookback_values_count': len(lookback_values),
        'consecutive_missing_hours': consecutive_missing_hours,
        'unit': unit
    }
    
    provenance_hash = calculate_provenance_hash(
        function_name='get_substitute_data',
        inputs=inputs_dict,
        outputs={'substitute_value': substitute_value, 'tier': tier.value},
        calculation_trace=calculation_trace)
    
    return SubstituteDataResult(
        substitute_value=substitute_value,
        unit=unit,
        substitution_tier=tier,
        method_used=method,
        lookback_hours=len(lookback_values),
        valid_data_count=len(lookback_values),
        missing_hour_number=consecutive_missing_hours,
        calculation_trace=calculation_trace,
        provenance_hash=provenance_hash,
        inputs=inputs_dict,
        formula_reference='40 CFR Part 75, Appendix D',
        is_valid=len(validation_messages) == 0,
        validation_messages=validation_messages)


def calculate_data_availability(
    valid_hours: int,
    total_hours: int,
    decimal_precision: int = 2
) -> DataAvailabilityResult:
    '''
    Calculate data availability percentage per 40 CFR Part 75.
    
    Formula: Availability = (valid_hours / total_hours) * 100
    
    EPA requires minimum 90% data availability for CEMS.
    '''
    calculation_trace: List[CalculationTrace] = []
    
    if total_hours <= 0:
        raise ValueError('Total hours must be positive')
    
    # Calculate availability
    availability = (Decimal(str(valid_hours)) / Decimal(str(total_hours))) * Decimal('100')
    availability = _apply_precision(availability, decimal_precision)
    
    calculation_trace.append(CalculationTrace(
        step_number=1, description='Calculate data availability percentage',
        formula='Availability = (valid_hours / total_hours) * 100',
        inputs={'valid_hours': str(valid_hours), 'total_hours': str(total_hours)},
        output='availability_percent', output_value=availability))
    
    meets_minimum = availability >= MIN_DATA_AVAILABILITY
    calculation_trace.append(CalculationTrace(
        step_number=2, description='Check against minimum requirement',
        formula=f'Availability >= {MIN_DATA_AVAILABILITY}%',
        inputs={'availability': str(availability)},
        output='meets_minimum', output_value=meets_minimum))
    
    inputs_dict = {'valid_hours': valid_hours, 'total_hours': total_hours}
    
    provenance_hash = calculate_provenance_hash(
        function_name='calculate_data_availability',
        inputs=inputs_dict,
        outputs={'availability_percent': availability, 'meets_minimum': meets_minimum},
        calculation_trace=calculation_trace)
    
    return DataAvailabilityResult(
        availability_percent=availability,
        valid_hours=valid_hours,
        total_hours=total_hours,
        meets_minimum=meets_minimum,
        calculation_trace=calculation_trace,
        provenance_hash=provenance_hash,
        inputs=inputs_dict,
        formula_reference='40 CFR Part 75, Subpart D')


def get_lookback_period_values(
    hourly_data: List[Optional[Decimal]],
    current_hour_index: int,
    lookback_hours: int = LOOKBACK_PERIOD_HOURS
) -> List[Decimal]:
    '''
    Extract valid values from lookback period for substitution.
    
    Per 40 CFR Part 75, looks back up to 2160 hours (90 days)
    of quality-assured data.
    '''
    start_index = max(0, current_hour_index - lookback_hours)
    lookback_slice = hourly_data[start_index:current_hour_index]
    
    # Filter out None/invalid values
    valid_values = [Decimal(str(v)) for v in lookback_slice if v is not None]
    
    return valid_values


def count_consecutive_missing_hours(
    hourly_data: List[Optional[Decimal]],
    current_hour_index: int
) -> int:
    '''
    Count consecutive missing hours ending at current index.
    
    Looks backward from current index to find start of
    missing data period.
    '''
    count = 0
    index = current_hour_index
    
    while index >= 0 and hourly_data[index] is None:
        count += 1
        index -= 1
    
    return count
