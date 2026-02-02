'''
EPA 40 CFR Part 75 Averaging Period Calculator

This module implements EPA averaging period calculations including
hourly, rolling, and block averages. All calculations are:
- 100% deterministic (no random elements)
- Use Decimal for precision
- Include SHA-256 provenance hash for audit trail
- Include calculation trace for explainability

Reference: 40 CFR Part 75

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
# CONSTANTS - EPA Averaging Requirements
# ==============================================================================

class AveragingPeriod(Enum):
    '''Standard EPA averaging periods.'''
    HOURLY = 'hourly'
    ROLLING_3_HOUR = 'rolling_3_hour'
    ROLLING_24_HOUR = 'rolling_24_hour'
    ROLLING_30_DAY = 'rolling_30_day'
    BLOCK_24_HOUR = 'block_24_hour'
    QUARTERLY = 'quarterly'
    ANNUAL = 'annual'


# Data completeness thresholds
MIN_DATA_COMPLETENESS_HOUR = Decimal('75')  # 75% = 45 minutes of valid data
MIN_DATA_COMPLETENESS_DAY = Decimal('75')   # 75% = 18 hours of valid data
MIN_DATA_COMPLETENESS_QUARTER = Decimal('90')  # 90% for quarterly

# Period lengths in hours
HOURS_PER_DAY = 24
HOURS_PER_QUARTER = 2160  # 90 days
HOURS_PER_YEAR = 8760


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
class AveragingResult:
    '''Result of averaging calculation.'''
    average_value: Decimal
    unit: str
    period_type: AveragingPeriod
    data_completeness: Decimal
    is_valid: bool
    valid_data_points: int
    total_data_points: int
    period_start: Optional[str]
    period_end: Optional[str]
    calculation_trace: List[CalculationTrace]
    provenance_hash: str
    inputs: Dict[str, Any]
    formula_reference: str
    validation_messages: List[str] = field(default_factory=list)


@dataclass
class RollingAverageResult:
    '''Result of rolling average calculation.'''
    averages: List[Decimal]
    period_type: AveragingPeriod
    window_size: int
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
# DATA COMPLETENESS
# ==============================================================================

def calculate_data_completeness(
    valid_count: int,
    total_count: int,
    decimal_precision: int = 2
) -> Decimal:
    '''
    Calculate data completeness percentage.
    
    Formula: Completeness = (valid_count / total_count) * 100
    '''
    if total_count <= 0:
        return Decimal('0')
    
    completeness = (Decimal(str(valid_count)) / Decimal(str(total_count))) * Decimal('100')
    return _apply_precision(completeness, decimal_precision)


def is_hour_valid(
    minute_values: List[Optional[Decimal]],
    min_completeness: Decimal = MIN_DATA_COMPLETENESS_HOUR
) -> bool:
    '''
    Determine if an hour has sufficient valid data.
    
    Per 40 CFR Part 75, an hour is valid if at least 75%
    (45 of 60 minutes) of data is available.
    '''
    valid_count = sum(1 for v in minute_values if v is not None)
    completeness = calculate_data_completeness(valid_count, len(minute_values))
    return completeness >= min_completeness


def is_day_valid(
    hourly_values: List[Optional[Decimal]],
    min_completeness: Decimal = MIN_DATA_COMPLETENESS_DAY
) -> bool:
    '''
    Determine if a day has sufficient valid hourly data.
    
    Per 40 CFR Part 75, a day is valid if at least 75%
    (18 of 24 hours) of hourly data is available.
    '''
    valid_count = sum(1 for v in hourly_values if v is not None)
    completeness = calculate_data_completeness(valid_count, len(hourly_values))
    return completeness >= min_completeness


# ==============================================================================
# BLOCK AVERAGES
# ==============================================================================

def calculate_hourly_average(
    minute_values: List[Optional[Decimal]],
    unit: str = 'lb/hr',
    decimal_precision: int = 3,
    period_start: Optional[str] = None
) -> AveragingResult:
    '''
    Calculate hourly average from minute-level data.
    
    Per 40 CFR Part 75, hourly averages require at least 75%
    data completeness (45 minutes of valid data).
    '''
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []
    
    # Count valid data
    valid_values = [Decimal(str(v)) for v in minute_values if v is not None]
    valid_count = len(valid_values)
    total_count = len(minute_values)
    
    # Calculate completeness
    completeness = calculate_data_completeness(valid_count, total_count)
    calculation_trace.append(CalculationTrace(
        step_number=1, description='Calculate data completeness',
        formula='Completeness = (valid_minutes / 60) * 100',
        inputs={'valid_count': str(valid_count), 'total_count': str(total_count)},
        output='completeness_percent', output_value=completeness))
    
    # Check validity
    is_valid = completeness >= MIN_DATA_COMPLETENESS_HOUR
    if not is_valid:
        validation_messages.append(
            f'Insufficient data: {completeness}% < {MIN_DATA_COMPLETENESS_HOUR}% required')
    
    # Calculate average
    if valid_count > 0:
        average = sum(valid_values) / Decimal(str(valid_count))
        average = _apply_precision(average, decimal_precision)
    else:
        average = Decimal('0')
    
    calculation_trace.append(CalculationTrace(
        step_number=2, description='Calculate arithmetic mean',
        formula='Average = sum(values) / n',
        inputs={'n_values': str(valid_count)},
        output='hourly_average', output_value=average))
    
    inputs_dict = {
        'n_minute_values': total_count,
        'valid_count': valid_count,
        'period_start': period_start
    }
    
    provenance_hash = calculate_provenance_hash(
        function_name='calculate_hourly_average',
        inputs=inputs_dict,
        outputs={'average': average, 'is_valid': is_valid},
        calculation_trace=calculation_trace)
    
    return AveragingResult(
        average_value=average,
        unit=unit,
        period_type=AveragingPeriod.HOURLY,
        data_completeness=completeness,
        is_valid=is_valid,
        valid_data_points=valid_count,
        total_data_points=total_count,
        period_start=period_start,
        period_end=None,
        calculation_trace=calculation_trace,
        provenance_hash=provenance_hash,
        inputs=inputs_dict,
        formula_reference='40 CFR Part 75',
        validation_messages=validation_messages)


def calculate_daily_average(
    hourly_values: List[Optional[Decimal]],
    unit: str = 'lb/day',
    decimal_precision: int = 2,
    period_start: Optional[str] = None
) -> AveragingResult:
    '''
    Calculate 24-hour block average from hourly data.
    
    Per 40 CFR Part 75, daily averages require at least 75%
    data completeness (18 hours of valid data).
    '''
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []
    
    valid_values = [Decimal(str(v)) for v in hourly_values if v is not None]
    valid_count = len(valid_values)
    total_count = len(hourly_values)
    
    completeness = calculate_data_completeness(valid_count, total_count)
    calculation_trace.append(CalculationTrace(
        step_number=1, description='Calculate data completeness',
        formula='Completeness = (valid_hours / 24) * 100',
        inputs={'valid_count': str(valid_count), 'total_count': str(total_count)},
        output='completeness_percent', output_value=completeness))
    
    is_valid = completeness >= MIN_DATA_COMPLETENESS_DAY
    if not is_valid:
        validation_messages.append(
            f'Insufficient data: {completeness}% < {MIN_DATA_COMPLETENESS_DAY}% required')
    
    if valid_count > 0:
        average = sum(valid_values) / Decimal(str(valid_count))
        average = _apply_precision(average, decimal_precision)
    else:
        average = Decimal('0')
    
    calculation_trace.append(CalculationTrace(
        step_number=2, description='Calculate daily average',
        formula='Average = sum(hourly_values) / n_hours',
        inputs={'n_values': str(valid_count)},
        output='daily_average', output_value=average))
    
    inputs_dict = {
        'n_hourly_values': total_count,
        'valid_count': valid_count,
        'period_start': period_start
    }
    
    provenance_hash = calculate_provenance_hash(
        function_name='calculate_daily_average',
        inputs=inputs_dict,
        outputs={'average': average, 'is_valid': is_valid},
        calculation_trace=calculation_trace)
    
    return AveragingResult(
        average_value=average,
        unit=unit,
        period_type=AveragingPeriod.BLOCK_24_HOUR,
        data_completeness=completeness,
        is_valid=is_valid,
        valid_data_points=valid_count,
        total_data_points=total_count,
        period_start=period_start,
        period_end=None,
        calculation_trace=calculation_trace,
        provenance_hash=provenance_hash,
        inputs=inputs_dict,
        formula_reference='40 CFR Part 75',
        validation_messages=validation_messages)


# ==============================================================================
# ROLLING AVERAGES
# ==============================================================================

def calculate_rolling_average(
    values: List[Optional[Decimal]],
    window_size: int,
    period_type: AveragingPeriod,
    min_valid_points: Optional[int] = None,
    decimal_precision: int = 3
) -> RollingAverageResult:
    '''
    Calculate rolling average with specified window size.
    
    Args:
        values: List of values (None for missing data)
        window_size: Number of periods in rolling window
        period_type: Type of averaging period
        min_valid_points: Minimum valid points required (default: 75% of window)
        decimal_precision: Decimal places for results
        
    Returns:
        RollingAverageResult with list of rolling averages
    '''
    calculation_trace: List[CalculationTrace] = []
    
    if min_valid_points is None:
        min_valid_points = int(window_size * 0.75)
    
    calculation_trace.append(CalculationTrace(
        step_number=1, description='Set rolling window parameters',
        formula='min_valid = window_size * 0.75',
        inputs={'window_size': str(window_size)},
        output='min_valid_points', output_value=min_valid_points))
    
    rolling_averages: List[Decimal] = []
    
    for i in range(len(values)):
        if i < window_size - 1:
            # Not enough data points yet
            rolling_averages.append(None)
            continue
        
        window = values[i - window_size + 1:i + 1]
        valid_values = [Decimal(str(v)) for v in window if v is not None]
        
        if len(valid_values) >= min_valid_points:
            avg = sum(valid_values) / Decimal(str(len(valid_values)))
            rolling_averages.append(_apply_precision(avg, decimal_precision))
        else:
            rolling_averages.append(None)
    
    calculation_trace.append(CalculationTrace(
        step_number=2, description='Calculate rolling averages',
        formula='avg[i] = mean(values[i-window+1:i+1])',
        inputs={'n_values': str(len(values)), 'window': str(window_size)},
        output='n_averages', output_value=len([a for a in rolling_averages if a is not None])))
    
    inputs_dict = {
        'n_values': len(values),
        'window_size': window_size,
        'min_valid_points': min_valid_points
    }
    
    provenance_hash = calculate_provenance_hash(
        function_name='calculate_rolling_average',
        inputs=inputs_dict,
        outputs={'n_averages': len(rolling_averages)},
        calculation_trace=calculation_trace)
    
    return RollingAverageResult(
        averages=rolling_averages,
        period_type=period_type,
        window_size=window_size,
        calculation_trace=calculation_trace,
        provenance_hash=provenance_hash,
        inputs=inputs_dict,
        formula_reference='40 CFR Part 75')


def calculate_rolling_3_hour_average(
    hourly_values: List[Optional[Decimal]],
    decimal_precision: int = 3
) -> RollingAverageResult:
    '''Calculate 3-hour rolling average from hourly data.'''
    return calculate_rolling_average(
        values=hourly_values,
        window_size=3,
        period_type=AveragingPeriod.ROLLING_3_HOUR,
        min_valid_points=2,  # At least 2 of 3 hours
        decimal_precision=decimal_precision)


def calculate_rolling_24_hour_average(
    hourly_values: List[Optional[Decimal]],
    decimal_precision: int = 3
) -> RollingAverageResult:
    '''Calculate 24-hour rolling average from hourly data.'''
    return calculate_rolling_average(
        values=hourly_values,
        window_size=24,
        period_type=AveragingPeriod.ROLLING_24_HOUR,
        min_valid_points=18,  # At least 18 of 24 hours (75%)
        decimal_precision=decimal_precision)


def calculate_rolling_30_day_average(
    daily_values: List[Optional[Decimal]],
    decimal_precision: int = 2
) -> RollingAverageResult:
    '''Calculate 30-day rolling average from daily data.'''
    return calculate_rolling_average(
        values=daily_values,
        window_size=30,
        period_type=AveragingPeriod.ROLLING_30_DAY,
        min_valid_points=23,  # At least 23 of 30 days (75%)
        decimal_precision=decimal_precision)


# ==============================================================================
# QUARTERLY AND ANNUAL AVERAGES
# ==============================================================================

def calculate_quarterly_average(
    hourly_values: List[Optional[Decimal]],
    unit: str = 'lb/hr',
    decimal_precision: int = 2,
    quarter_label: Optional[str] = None
) -> AveragingResult:
    '''
    Calculate quarterly average from hourly data.
    
    Per 40 CFR Part 75, quarterly reporting requires 90%
    data availability.
    '''
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []
    
    valid_values = [Decimal(str(v)) for v in hourly_values if v is not None]
    valid_count = len(valid_values)
    total_count = len(hourly_values)
    
    completeness = calculate_data_completeness(valid_count, total_count)
    calculation_trace.append(CalculationTrace(
        step_number=1, description='Calculate quarterly data completeness',
        formula='Completeness = (valid_hours / total_hours) * 100',
        inputs={'valid_count': str(valid_count), 'total_count': str(total_count)},
        output='completeness_percent', output_value=completeness))
    
    is_valid = completeness >= MIN_DATA_COMPLETENESS_QUARTER
    if not is_valid:
        validation_messages.append(
            f'Insufficient data: {completeness}% < {MIN_DATA_COMPLETENESS_QUARTER}% required')
    
    if valid_count > 0:
        average = sum(valid_values) / Decimal(str(valid_count))
        average = _apply_precision(average, decimal_precision)
    else:
        average = Decimal('0')
    
    calculation_trace.append(CalculationTrace(
        step_number=2, description='Calculate quarterly average',
        formula='Average = sum(hourly_values) / n_hours',
        inputs={'n_values': str(valid_count)},
        output='quarterly_average', output_value=average))
    
    inputs_dict = {
        'n_hourly_values': total_count,
        'valid_count': valid_count,
        'quarter': quarter_label
    }
    
    provenance_hash = calculate_provenance_hash(
        function_name='calculate_quarterly_average',
        inputs=inputs_dict,
        outputs={'average': average, 'is_valid': is_valid},
        calculation_trace=calculation_trace)
    
    return AveragingResult(
        average_value=average,
        unit=unit,
        period_type=AveragingPeriod.QUARTERLY,
        data_completeness=completeness,
        is_valid=is_valid,
        valid_data_points=valid_count,
        total_data_points=total_count,
        period_start=quarter_label,
        period_end=None,
        calculation_trace=calculation_trace,
        provenance_hash=provenance_hash,
        inputs=inputs_dict,
        formula_reference='40 CFR Part 75',
        validation_messages=validation_messages)


def calculate_annual_average(
    hourly_values: List[Optional[Decimal]],
    unit: str = 'lb/hr',
    decimal_precision: int = 2,
    year_label: Optional[str] = None
) -> AveragingResult:
    '''
    Calculate annual average from hourly data.
    
    Per 40 CFR Part 75, annual reporting requires 90%
    data availability.
    '''
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []
    
    valid_values = [Decimal(str(v)) for v in hourly_values if v is not None]
    valid_count = len(valid_values)
    total_count = len(hourly_values)
    
    completeness = calculate_data_completeness(valid_count, total_count)
    calculation_trace.append(CalculationTrace(
        step_number=1, description='Calculate annual data completeness',
        formula='Completeness = (valid_hours / total_hours) * 100',
        inputs={'valid_count': str(valid_count), 'total_count': str(total_count)},
        output='completeness_percent', output_value=completeness))
    
    is_valid = completeness >= MIN_DATA_COMPLETENESS_QUARTER
    if not is_valid:
        validation_messages.append(
            f'Insufficient data: {completeness}% < {MIN_DATA_COMPLETENESS_QUARTER}% required')
    
    if valid_count > 0:
        average = sum(valid_values) / Decimal(str(valid_count))
        average = _apply_precision(average, decimal_precision)
    else:
        average = Decimal('0')
    
    calculation_trace.append(CalculationTrace(
        step_number=2, description='Calculate annual average',
        formula='Average = sum(hourly_values) / n_hours',
        inputs={'n_values': str(valid_count)},
        output='annual_average', output_value=average))
    
    inputs_dict = {
        'n_hourly_values': total_count,
        'valid_count': valid_count,
        'year': year_label
    }
    
    provenance_hash = calculate_provenance_hash(
        function_name='calculate_annual_average',
        inputs=inputs_dict,
        outputs={'average': average, 'is_valid': is_valid},
        calculation_trace=calculation_trace)
    
    return AveragingResult(
        average_value=average,
        unit=unit,
        period_type=AveragingPeriod.ANNUAL,
        data_completeness=completeness,
        is_valid=is_valid,
        valid_data_points=valid_count,
        total_data_points=total_count,
        period_start=year_label,
        period_end=None,
        calculation_trace=calculation_trace,
        provenance_hash=provenance_hash,
        inputs=inputs_dict,
        formula_reference='40 CFR Part 75',
        validation_messages=validation_messages)


def calculate_total_emissions_for_period(
    hourly_mass_rates: List[Optional[Decimal]],
    unit: str = 'tons',
    decimal_precision: int = 2
) -> Tuple[Decimal, int]:
    '''
    Calculate total emissions for a period from hourly mass rates.
    
    Formula: Total = sum(hourly_lb) / 2000 (convert to tons)
    
    Returns:
        Tuple of (total_emissions, valid_hours)
    '''
    valid_values = [Decimal(str(v)) for v in hourly_mass_rates if v is not None]
    
    if not valid_values:
        return Decimal('0'), 0
    
    # Sum hourly values (lb/hr * 1 hr = lb)
    total_lb = sum(valid_values)
    
    # Convert to tons
    total_tons = total_lb / Decimal('2000')
    
    return _apply_precision(total_tons, decimal_precision), len(valid_values)
