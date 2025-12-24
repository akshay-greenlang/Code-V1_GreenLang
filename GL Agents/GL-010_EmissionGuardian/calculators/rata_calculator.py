'''
EPA 40 CFR Part 75 RATA (Relative Accuracy Test Audit) Calculator

This module implements RATA statistical calculations per EPA 40 CFR Part 75,
Appendix A, Section 7. All calculations are:
- 100% deterministic (no random elements)
- Use Decimal for precision
- Include SHA-256 provenance hash for audit trail
- Include calculation trace for explainability

Reference: 40 CFR Part 75, Appendix A, Section 7

Author: GL-CalculatorEngineer
'''

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from datetime import datetime


# ==============================================================================
# CONSTANTS - EPA RATA Requirements
# ==============================================================================

# t-values for 95% confidence (two-tailed) from EPA tables
# Index is degrees of freedom (n-1)
T_VALUES_95: Dict[int, Decimal] = {
    1: Decimal('12.706'),
    2: Decimal('4.303'),
    3: Decimal('3.182'),
    4: Decimal('2.776'),
    5: Decimal('2.571'),
    6: Decimal('2.447'),
    7: Decimal('2.365'),
    8: Decimal('2.306'),
    9: Decimal('2.262'),
    10: Decimal('2.228'),
    11: Decimal('2.201'),
    12: Decimal('2.179'),
    13: Decimal('2.160'),
    14: Decimal('2.145'),
    15: Decimal('2.131'),
    16: Decimal('2.120'),
    17: Decimal('2.110'),
    18: Decimal('2.101'),
    19: Decimal('2.093'),
    20: Decimal('2.086'),
    21: Decimal('2.080'),
    22: Decimal('2.074'),
    23: Decimal('2.069'),
    24: Decimal('2.064'),
    25: Decimal('2.060'),
    26: Decimal('2.056'),
    27: Decimal('2.052'),
    28: Decimal('2.048'),
    29: Decimal('2.045'),
    30: Decimal('2.042'),
}

# RATA pass/fail thresholds per 40 CFR Part 75
RATA_PASS_THRESHOLD_STANDARD = Decimal('10.0')  # 10% for 9+ run tests
RATA_PASS_THRESHOLD_ABBREVIATED = Decimal('7.5')  # 7.5% for 3-run abbreviated

# Minimum reference method mean for alternative RA calculation
MIN_RM_MEAN_ALTERNATIVE = Decimal('0.0')

# Bias test alpha level
BIAS_TEST_ALPHA = Decimal('0.05')


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
class RATAResult:
    relative_accuracy: Decimal
    mean_difference: Decimal
    standard_deviation: Decimal
    confidence_coefficient: Decimal
    reference_method_mean: Decimal
    cems_mean: Decimal
    passed: bool
    test_type: str
    num_runs: int
    bias_test_passed: bool
    bias_adjustment_factor: Optional[Decimal]
    calculation_trace: List[CalculationTrace]
    provenance_hash: str
    inputs: Dict[str, Any]
    formula_reference: str
    is_valid: bool = True
    validation_messages: List[str] = field(default_factory=list)


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


def _get_t_value(degrees_of_freedom: int) -> Decimal:
    '''Get t-value for 95% confidence interval.'''
    if degrees_of_freedom in T_VALUES_95:
        return T_VALUES_95[degrees_of_freedom]
    elif degrees_of_freedom > 30:
        return Decimal('1.96')  # Normal distribution approximation
    else:
        raise ValueError(f'Invalid degrees of freedom: {degrees_of_freedom}')


def _sqrt_decimal(value: Decimal) -> Decimal:
    '''Calculate square root of Decimal with high precision.'''
    if value < 0:
        raise ValueError('Cannot take square root of negative number')
    if value == 0:
        return Decimal('0')
    return Decimal(str(math.sqrt(float(value))))


# ==============================================================================
# CORE RATA CALCULATIONS
# ==============================================================================

def calculate_mean_difference(
    cems_values: List[Decimal],
    rm_values: List[Decimal]
) -> Tuple[Decimal, List[Decimal]]:
    '''Calculate mean of differences (CEMS - RM) for paired data.'''
    if len(cems_values) != len(rm_values):
        raise ValueError('CEMS and RM lists must have same length')
    
    differences = [Decimal(str(c)) - Decimal(str(r)) for c, r in zip(cems_values, rm_values)]
    n = Decimal(str(len(differences)))
    mean_diff = sum(differences) / n
    return mean_diff, differences


def calculate_standard_deviation(
    differences: List[Decimal],
    mean_difference: Decimal
) -> Decimal:
    '''Calculate sample standard deviation of differences.'''
    n = len(differences)
    if n < 2:
        raise ValueError('Need at least 2 data points for standard deviation')
    
    sum_squared_deviations = sum(
        (d - mean_difference) ** 2 for d in differences
    )
    variance = sum_squared_deviations / Decimal(str(n - 1))
    return _sqrt_decimal(variance)


def calculate_confidence_coefficient(
    standard_deviation: Decimal,
    num_runs: int
) -> Decimal:
    '''
    Calculate confidence coefficient per 40 CFR Part 75.
    
    Formula: CC = t * Sd / sqrt(n)
    Where:
        t = t-value from t-distribution at alpha=0.05
        Sd = standard deviation of differences
        n = number of test runs
    '''
    degrees_of_freedom = num_runs - 1
    t_value = _get_t_value(degrees_of_freedom)
    sqrt_n = _sqrt_decimal(Decimal(str(num_runs)))
    
    cc = (t_value * standard_deviation) / sqrt_n
    return cc


def calculate_relative_accuracy(
    mean_difference: Decimal,
    confidence_coefficient: Decimal,
    rm_mean: Decimal,
    decimal_precision: int = 2
) -> Decimal:
    '''
    Calculate Relative Accuracy per 40 CFR Part 75, Appendix A.
    
    Formula: RA = (|d_bar| + CC) / RM_mean * 100
    Where:
        d_bar = mean of differences
        CC = confidence coefficient
        RM_mean = mean of reference method values
    '''
    if rm_mean == 0:
        raise ValueError('Reference method mean cannot be zero')
    
    abs_mean_diff = abs(mean_difference)
    numerator = abs_mean_diff + confidence_coefficient
    ra = (numerator / rm_mean) * Decimal('100')
    
    return _apply_precision(ra, decimal_precision)


def calculate_bias_test(
    differences: List[Decimal],
    mean_difference: Decimal,
    standard_deviation: Decimal,
    num_runs: int
) -> Tuple[bool, Decimal]:
    '''
    Perform bias test (paired t-test) per 40 CFR Part 75.
    
    Tests H0: mean difference = 0
    Uses t = d_bar / (Sd / sqrt(n))
    
    Returns:
        Tuple of (bias_significant: bool, t_statistic: Decimal)
    '''
    if standard_deviation == 0:
        return False, Decimal('0')
    
    sqrt_n = _sqrt_decimal(Decimal(str(num_runs)))
    t_statistic = mean_difference / (standard_deviation / sqrt_n)
    
    degrees_of_freedom = num_runs - 1
    t_critical = _get_t_value(degrees_of_freedom)
    
    bias_significant = abs(t_statistic) > t_critical
    return bias_significant, t_statistic


def calculate_bias_adjustment_factor(
    cems_mean: Decimal,
    rm_mean: Decimal
) -> Decimal:
    '''
    Calculate Bias Adjustment Factor (BAF) per 40 CFR Part 75.
    
    Formula: BAF = RM_mean / CEMS_mean
    
    Applied when bias is significant to adjust CEMS readings.
    '''
    if cems_mean == 0:
        raise ValueError('CEMS mean cannot be zero for BAF calculation')
    
    baf = rm_mean / cems_mean
    return _apply_precision(baf, 4)


def perform_rata(
    cems_values: List[Decimal],
    rm_values: List[Decimal],
    test_type: str = 'standard',
    decimal_precision: int = 2
) -> RATAResult:
    '''
    Perform complete RATA calculation per 40 CFR Part 75, Appendix A.
    
    Args:
        cems_values: List of CEMS measurements
        rm_values: List of Reference Method measurements
        test_type: 'standard' (9+ runs) or 'abbreviated' (3 runs)
        decimal_precision: Number of decimal places for results
        
    Returns:
        RATAResult with all calculated values and provenance
    '''
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []
    
    # Convert to Decimal
    cems_values = [Decimal(str(v)) for v in cems_values]
    rm_values = [Decimal(str(v)) for v in rm_values]
    num_runs = len(cems_values)
    
    # Validation
    if len(cems_values) != len(rm_values):
        raise ValueError('CEMS and RM lists must have same length')
    
    if test_type == 'standard' and num_runs < 9:
        validation_messages.append(f'Standard RATA requires 9+ runs, got {num_runs}')
    elif test_type == 'abbreviated' and num_runs != 3:
        validation_messages.append(f'Abbreviated RATA requires exactly 3 runs, got {num_runs}')
    
    # Step 1: Calculate means
    cems_mean = sum(cems_values) / Decimal(str(num_runs))
    rm_mean = sum(rm_values) / Decimal(str(num_runs))
    
    calculation_trace.append(CalculationTrace(
        step_number=1, description='Calculate CEMS mean',
        formula='CEMS_mean = sum(CEMS_i) / n',
        inputs={'cems_values': str(cems_values), 'n': str(num_runs)},
        output='cems_mean', output_value=cems_mean))
    
    calculation_trace.append(CalculationTrace(
        step_number=2, description='Calculate Reference Method mean',
        formula='RM_mean = sum(RM_i) / n',
        inputs={'rm_values': str(rm_values), 'n': str(num_runs)},
        output='rm_mean', output_value=rm_mean))
    
    # Step 2: Calculate mean difference
    mean_diff, differences = calculate_mean_difference(cems_values, rm_values)
    calculation_trace.append(CalculationTrace(
        step_number=3, description='Calculate mean difference',
        formula='d_bar = sum(CEMS_i - RM_i) / n',
        inputs={'differences': str(differences)},
        output='mean_difference', output_value=mean_diff))
    
    # Step 3: Calculate standard deviation
    std_dev = calculate_standard_deviation(differences, mean_diff)
    calculation_trace.append(CalculationTrace(
        step_number=4, description='Calculate standard deviation of differences',
        formula='Sd = sqrt(sum((d_i - d_bar)^2) / (n-1))',
        inputs={'mean_diff': str(mean_diff), 'n': str(num_runs)},
        output='standard_deviation', output_value=std_dev))
    
    # Step 4: Calculate confidence coefficient
    conf_coef = calculate_confidence_coefficient(std_dev, num_runs)
    calculation_trace.append(CalculationTrace(
        step_number=5, description='Calculate confidence coefficient',
        formula='CC = t * Sd / sqrt(n)',
        inputs={'std_dev': str(std_dev), 'n': str(num_runs), 
                't_value': str(_get_t_value(num_runs - 1))},
        output='confidence_coefficient', output_value=conf_coef))
    
    # Step 5: Calculate Relative Accuracy
    ra = calculate_relative_accuracy(mean_diff, conf_coef, rm_mean, decimal_precision)
    calculation_trace.append(CalculationTrace(
        step_number=6, description='Calculate Relative Accuracy',
        formula='RA = (|d_bar| + CC) / RM_mean * 100',
        inputs={'mean_diff': str(mean_diff), 'conf_coef': str(conf_coef), 
                'rm_mean': str(rm_mean)},
        output='relative_accuracy_percent', output_value=ra))
    
    # Step 6: Determine pass/fail
    threshold = RATA_PASS_THRESHOLD_ABBREVIATED if test_type == 'abbreviated' else RATA_PASS_THRESHOLD_STANDARD
    passed = ra <= threshold
    calculation_trace.append(CalculationTrace(
        step_number=7, description='Determine RATA pass/fail',
        formula=f'RA <= {threshold}%',
        inputs={'relative_accuracy': str(ra), 'threshold': str(threshold)},
        output='passed', output_value=passed))
    
    # Step 7: Bias test
    bias_significant, t_stat = calculate_bias_test(differences, mean_diff, std_dev, num_runs)
    bias_passed = not bias_significant
    calculation_trace.append(CalculationTrace(
        step_number=8, description='Perform bias test (paired t-test)',
        formula='t = d_bar / (Sd / sqrt(n))',
        inputs={'mean_diff': str(mean_diff), 'std_dev': str(std_dev)},
        output='bias_test_passed', output_value=bias_passed))
    
    # Step 8: Calculate BAF if bias significant
    baf = None
    if bias_significant and cems_mean != 0:
        baf = calculate_bias_adjustment_factor(cems_mean, rm_mean)
        calculation_trace.append(CalculationTrace(
            step_number=9, description='Calculate Bias Adjustment Factor',
            formula='BAF = RM_mean / CEMS_mean',
            inputs={'rm_mean': str(rm_mean), 'cems_mean': str(cems_mean)},
            output='bias_adjustment_factor', output_value=baf))
    
    # Build provenance
    inputs_dict = {
        'cems_values': [str(v) for v in cems_values],
        'rm_values': [str(v) for v in rm_values],
        'test_type': test_type,
        'num_runs': num_runs
    }
    
    outputs_dict = {
        'relative_accuracy': ra,
        'passed': passed,
        'bias_test_passed': bias_passed
    }
    
    provenance_hash = calculate_provenance_hash(
        function_name='perform_rata',
        inputs=inputs_dict,
        outputs=outputs_dict,
        calculation_trace=calculation_trace)
    
    return RATAResult(
        relative_accuracy=ra,
        mean_difference=_apply_precision(mean_diff, decimal_precision),
        standard_deviation=_apply_precision(std_dev, decimal_precision),
        confidence_coefficient=_apply_precision(conf_coef, decimal_precision),
        reference_method_mean=_apply_precision(rm_mean, decimal_precision),
        cems_mean=_apply_precision(cems_mean, decimal_precision),
        passed=passed,
        test_type=test_type,
        num_runs=num_runs,
        bias_test_passed=bias_passed,
        bias_adjustment_factor=baf,
        calculation_trace=calculation_trace,
        provenance_hash=provenance_hash,
        inputs=inputs_dict,
        formula_reference='40 CFR Part 75, Appendix A, Section 7',
        is_valid=len(validation_messages) == 0,
        validation_messages=validation_messages)
