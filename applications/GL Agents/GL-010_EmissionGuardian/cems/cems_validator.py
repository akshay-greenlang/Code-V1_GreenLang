# -*- coding: utf-8 -*-
"""
EPA 40 CFR Part 75 CEMS Data Validation Module

Comprehensive validation of Continuous Emission Monitoring System (CEMS) data
per EPA 40 CFR Part 75 quality assurance requirements.

Reference Documents:
- 40 CFR Part 75, Subpart D: Missing Data Substitution
- 40 CFR Part 75, Appendix A: CEMS Specifications
- 40 CFR Part 75, Appendix B: Quality Assurance Procedures
- 40 CFR Part 75, Appendix F: Conversion Procedures

Author: GL-CalculatorEngineer
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from datetime import datetime, timedelta


# ==============================================================================
# ENUMS AND CONSTANTS
# ==============================================================================

class CEMSParameter(Enum):
    """CEMS monitored parameters per 40 CFR Part 75."""
    SO2 = 'so2'
    NOX = 'nox'
    CO2 = 'co2'
    O2 = 'o2'
    FLOW = 'flow'
    OPACITY = 'opacity'


class ValidationStatus(Enum):
    """Data quality status per EPA requirements."""
    VALID = 'valid'
    MISSING = 'missing'
    SUBSTITUTED = 'substituted'
    OUT_OF_RANGE = 'out_of_range'
    CALIBRATION_ERROR = 'calibration_error'
    QA_FAILED = 'qa_failed'


class QATestType(Enum):
    """Quality assurance test types per Appendix B."""
    DAILY_CALIBRATION = 'daily_calibration'
    RATA = 'rata'
    CGA = 'cga'
    LINEARITY = 'linearity'
    BGVS = 'bias_test'


# EPA span and range limits per Appendix A
CEMS_RANGE_LIMITS: Dict[str, Dict[str, Decimal]] = {
    'SO2': {
        'low_range_max_ppm': Decimal('200'),
        'high_range_max_ppm': Decimal('2500'),
        'span_minimum_pct': Decimal('80'),
        'span_maximum_pct': Decimal('100'),
    },
    'NOX': {
        'low_range_max_ppm': Decimal('200'),
        'high_range_max_ppm': Decimal('2500'),
        'span_minimum_pct': Decimal('80'),
        'span_maximum_pct': Decimal('100'),
    },
    'CO2': {
        'range_max_pct': Decimal('18'),
        'span_minimum_pct': Decimal('80'),
    },
    'O2': {
        'range_max_pct': Decimal('25'),
        'span_value_pct': Decimal('25'),
    },
    'FLOW': {
        'span_minimum_pct': Decimal('80'),
        'span_maximum_pct': Decimal('100'),
    },
}

# Daily calibration error limits per Appendix A
CALIBRATION_ERROR_LIMITS: Dict[str, Decimal] = {
    'SO2': Decimal('2.5'),  # % of span
    'NOX': Decimal('2.5'),
    'CO2': Decimal('1.0'),  # % CO2
    'O2': Decimal('0.5'),   # % O2
    'FLOW': Decimal('3.0'),
}

# RATA performance criteria
RATA_LIMITS: Dict[str, Decimal] = {
    'standard_ra_limit': Decimal('10.0'),  # 9+ runs
    'abbreviated_ra_limit': Decimal('7.5'),  # 3 runs
    'bias_test_limit': Decimal('5.0'),
}

# CGA (Cylinder Gas Audit) limits
CGA_LIMITS: Dict[str, Decimal] = {
    'accuracy_limit': Decimal('5.0'),  # % of span
    'low_level_range': Decimal('0.2'),  # 20% of span
    'mid_level_range': Decimal('0.5'),  # 50% of span
    'high_level_range': Decimal('0.8'),  # 80% of span
}

# Linearity error limits
LINEARITY_LIMITS: Dict[str, Decimal] = {
    'SO2': Decimal('5.0'),  # % of reference
    'NOX': Decimal('5.0'),
    'FLOW': Decimal('5.0'),
}

# Data availability requirement
MIN_DATA_AVAILABILITY = Decimal('90.0')


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class CEMSReading:
    """Single CEMS reading with metadata."""
    timestamp: datetime
    parameter: CEMSParameter
    value: Decimal
    unit: str
    span_value: Decimal
    status: ValidationStatus = ValidationStatus.VALID
    calibration_drift: Optional[Decimal] = None
    qa_flags: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of CEMS data validation."""
    reading: CEMSReading
    is_valid: bool
    status: ValidationStatus
    validation_errors: List[str]
    validation_warnings: List[str]
    corrected_value: Optional[Decimal] = None
    provenance_hash: str = ''


@dataclass
class CalibrationCheck:
    """Daily calibration check result."""
    timestamp: datetime
    parameter: CEMSParameter
    zero_cal_error: Decimal
    upscale_cal_error: Decimal
    zero_passed: bool
    upscale_passed: bool
    overall_passed: bool
    provenance_hash: str = ''


@dataclass
class QATestResult:
    """Quality assurance test result."""
    test_type: QATestType
    test_date: datetime
    parameter: CEMSParameter
    result_value: Decimal
    limit_value: Decimal
    passed: bool
    expiration_date: datetime
    calculation_trace: List[Dict[str, Any]] = field(default_factory=list)
    provenance_hash: str = ''


@dataclass
class DataAvailabilityReport:
    """Data availability report per Subpart D."""
    parameter: CEMSParameter
    reporting_period: str
    total_hours: int
    valid_hours: int
    substitute_hours: int
    missing_hours: int
    availability_percent: Decimal
    meets_minimum: bool
    provenance_hash: str = ''


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def _apply_precision(value: Decimal, decimal_places: int) -> Decimal:
    """Apply precision with ROUND_HALF_UP."""
    if decimal_places == 0:
        return value.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    quantize_str = '0.' + '0' * decimal_places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def calculate_provenance_hash(
    operation: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any]
) -> str:
    """Calculate SHA-256 provenance hash for audit trail."""
    provenance_data = {
        'operation': operation,
        'inputs': {k: str(v) for k, v in sorted(inputs.items())},
        'outputs': {k: str(v) for k, v in sorted(outputs.items())},
        'timestamp': datetime.utcnow().isoformat(),
    }
    provenance_str = json.dumps(provenance_data, sort_keys=True)
    return hashlib.sha256(provenance_str.encode()).hexdigest()


# ==============================================================================
# CEMS DATA VALIDATOR CLASS
# ==============================================================================

class CEMSDataValidator:
    """
    EPA 40 CFR Part 75 compliant CEMS data validator.

    Validates CEMS readings against:
    - Range limits per Appendix A
    - Calibration error limits
    - QA test status
    - Data availability requirements
    """

    def __init__(self):
        """Initialize validator with EPA limits."""
        self.range_limits = CEMS_RANGE_LIMITS
        self.calibration_limits = CALIBRATION_ERROR_LIMITS
        self.rata_limits = RATA_LIMITS
        self.cga_limits = CGA_LIMITS
        self.linearity_limits = LINEARITY_LIMITS

    def validate_reading(
        self,
        reading: CEMSReading,
        qa_status: Optional[QATestResult] = None
    ) -> ValidationResult:
        """
        Validate a single CEMS reading per EPA requirements.

        Args:
            reading: CEMS reading to validate
            qa_status: Current QA test status (RATA, CGA, etc.)

        Returns:
            ValidationResult with validation status and any errors
        """
        errors: List[str] = []
        warnings: List[str] = []
        status = ValidationStatus.VALID
        corrected_value = None

        # Check for missing data
        if reading.value is None:
            status = ValidationStatus.MISSING
            errors.append('Missing data value')
            return ValidationResult(
                reading=reading,
                is_valid=False,
                status=status,
                validation_errors=errors,
                validation_warnings=warnings,
                provenance_hash=self._calculate_validation_hash(reading, errors)
            )

        # Validate range
        range_valid, range_msg = self._validate_range(reading)
        if not range_valid:
            status = ValidationStatus.OUT_OF_RANGE
            errors.append(range_msg)

        # Validate calibration drift
        if reading.calibration_drift is not None:
            cal_valid, cal_msg = self._validate_calibration_drift(reading)
            if not cal_valid:
                status = ValidationStatus.CALIBRATION_ERROR
                errors.append(cal_msg)

        # Check QA test status
        if qa_status is not None:
            if not qa_status.passed:
                status = ValidationStatus.QA_FAILED
                errors.append(f'{qa_status.test_type.value} test failed')
            elif qa_status.expiration_date < reading.timestamp:
                warnings.append(f'{qa_status.test_type.value} test expired')

        # Check span utilization
        span_pct = (reading.value / reading.span_value) * Decimal('100')
        if span_pct > Decimal('100'):
            warnings.append(f'Reading exceeds span ({span_pct:.1f}%)')
        elif span_pct < Decimal('20'):
            warnings.append(f'Reading below 20% of span ({span_pct:.1f}%)')

        is_valid = len(errors) == 0
        provenance_hash = self._calculate_validation_hash(reading, errors)

        return ValidationResult(
            reading=reading,
            is_valid=is_valid,
            status=status if not is_valid else ValidationStatus.VALID,
            validation_errors=errors,
            validation_warnings=warnings,
            corrected_value=corrected_value,
            provenance_hash=provenance_hash
        )

    def _validate_range(self, reading: CEMSReading) -> Tuple[bool, str]:
        """Validate reading against range limits."""
        param = reading.parameter.name
        if param not in self.range_limits:
            return True, ''

        limits = self.range_limits[param]

        # Check high range
        if 'high_range_max_ppm' in limits:
            max_val = limits['high_range_max_ppm']
            if reading.value > max_val:
                return False, f'{param} value {reading.value} exceeds maximum {max_val} ppm'

        if 'range_max_pct' in limits:
            max_val = limits['range_max_pct']
            if reading.value > max_val:
                return False, f'{param} value {reading.value} exceeds maximum {max_val}%'

        # Check for negative values
        if reading.value < Decimal('0'):
            return False, f'{param} value cannot be negative'

        return True, ''

    def _validate_calibration_drift(
        self,
        reading: CEMSReading
    ) -> Tuple[bool, str]:
        """Validate calibration drift against EPA limits."""
        param = reading.parameter.name
        if param not in self.calibration_limits:
            return True, ''

        limit = self.calibration_limits[param]
        if abs(reading.calibration_drift) > limit:
            return False, (
                f'{param} calibration drift {reading.calibration_drift}% '
                f'exceeds limit {limit}%'
            )
        return True, ''

    def _calculate_validation_hash(
        self,
        reading: CEMSReading,
        errors: List[str]
    ) -> str:
        """Calculate provenance hash for validation."""
        return calculate_provenance_hash(
            operation='validate_reading',
            inputs={
                'parameter': reading.parameter.value,
                'value': str(reading.value),
                'timestamp': reading.timestamp.isoformat(),
            },
            outputs={
                'error_count': len(errors),
                'errors': ','.join(errors),
            }
        )

    def validate_daily_calibration(
        self,
        parameter: CEMSParameter,
        zero_reference: Decimal,
        zero_response: Decimal,
        upscale_reference: Decimal,
        upscale_response: Decimal,
        span_value: Decimal
    ) -> CalibrationCheck:
        """
        Validate daily calibration check per Appendix B.

        Args:
            parameter: CEMS parameter being calibrated
            zero_reference: Zero gas reference value
            zero_response: CEMS response to zero gas
            upscale_reference: Upscale gas reference value
            upscale_response: CEMS response to upscale gas
            span_value: Analyzer span value

        Returns:
            CalibrationCheck result with pass/fail status
        """
        # Calculate calibration errors
        zero_error = abs(zero_response - zero_reference) / span_value * Decimal('100')
        upscale_error = abs(upscale_response - upscale_reference) / span_value * Decimal('100')

        # Get limit for this parameter
        limit = self.calibration_limits.get(parameter.name, Decimal('2.5'))

        zero_passed = zero_error <= limit
        upscale_passed = upscale_error <= limit
        overall_passed = zero_passed and upscale_passed

        provenance_hash = calculate_provenance_hash(
            operation='daily_calibration',
            inputs={
                'parameter': parameter.value,
                'zero_reference': str(zero_reference),
                'upscale_reference': str(upscale_reference),
                'span_value': str(span_value),
            },
            outputs={
                'zero_error': str(zero_error),
                'upscale_error': str(upscale_error),
                'passed': overall_passed,
            }
        )

        return CalibrationCheck(
            timestamp=datetime.utcnow(),
            parameter=parameter,
            zero_cal_error=_apply_precision(zero_error, 2),
            upscale_cal_error=_apply_precision(upscale_error, 2),
            zero_passed=zero_passed,
            upscale_passed=upscale_passed,
            overall_passed=overall_passed,
            provenance_hash=provenance_hash
        )

    def validate_cga(
        self,
        parameter: CEMSParameter,
        reference_values: List[Decimal],
        measured_values: List[Decimal],
        span_value: Decimal
    ) -> QATestResult:
        """
        Validate Cylinder Gas Audit (CGA) per Appendix B, Section 2.1.

        Args:
            parameter: CEMS parameter being audited
            reference_values: Reference gas concentrations (low, mid, high)
            measured_values: CEMS responses to reference gases
            span_value: Analyzer span value

        Returns:
            QATestResult with CGA pass/fail status
        """
        if len(reference_values) != 3 or len(measured_values) != 3:
            raise ValueError('CGA requires exactly 3 calibration levels')

        calculation_trace: List[Dict[str, Any]] = []
        max_error = Decimal('0')

        for i, (ref, meas) in enumerate(zip(reference_values, measured_values)):
            error = abs(meas - ref) / span_value * Decimal('100')
            level = ['low', 'mid', 'high'][i]

            calculation_trace.append({
                'step': i + 1,
                'level': level,
                'reference': str(ref),
                'measured': str(meas),
                'error_pct_span': str(_apply_precision(error, 2)),
            })

            max_error = max(max_error, error)

        limit = self.cga_limits['accuracy_limit']
        passed = max_error <= limit

        provenance_hash = calculate_provenance_hash(
            operation='cga_validation',
            inputs={
                'parameter': parameter.value,
                'reference_values': [str(v) for v in reference_values],
                'span_value': str(span_value),
            },
            outputs={
                'max_error': str(max_error),
                'passed': passed,
            }
        )

        # CGA valid for 1 quarter (3 months)
        expiration = datetime.utcnow() + timedelta(days=90)

        return QATestResult(
            test_type=QATestType.CGA,
            test_date=datetime.utcnow(),
            parameter=parameter,
            result_value=_apply_precision(max_error, 2),
            limit_value=limit,
            passed=passed,
            expiration_date=expiration,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def validate_linearity(
        self,
        parameter: CEMSParameter,
        reference_values: List[Decimal],
        measured_values: List[Decimal]
    ) -> QATestResult:
        """
        Validate linearity test per Appendix A, Section 6.

        Args:
            parameter: CEMS parameter
            reference_values: Reference concentrations (low, mid, high)
            measured_values: Averaged CEMS responses

        Returns:
            QATestResult with linearity pass/fail status
        """
        if len(reference_values) != 3 or len(measured_values) != 3:
            raise ValueError('Linearity requires exactly 3 levels')

        calculation_trace: List[Dict[str, Any]] = []
        max_error = Decimal('0')

        for i, (ref, meas) in enumerate(zip(reference_values, measured_values)):
            if ref == 0:
                error = abs(meas)
            else:
                error = abs(meas - ref) / ref * Decimal('100')

            level = ['low', 'mid', 'high'][i]

            calculation_trace.append({
                'step': i + 1,
                'level': level,
                'reference': str(ref),
                'measured': str(meas),
                'error_pct': str(_apply_precision(error, 2)),
            })

            max_error = max(max_error, error)

        limit = self.linearity_limits.get(parameter.name, Decimal('5.0'))
        passed = max_error <= limit

        provenance_hash = calculate_provenance_hash(
            operation='linearity_validation',
            inputs={
                'parameter': parameter.value,
                'reference_values': [str(v) for v in reference_values],
            },
            outputs={
                'max_error': str(max_error),
                'passed': passed,
            }
        )

        # Linearity valid for 1 quarter
        expiration = datetime.utcnow() + timedelta(days=90)

        return QATestResult(
            test_type=QATestType.LINEARITY,
            test_date=datetime.utcnow(),
            parameter=parameter,
            result_value=_apply_precision(max_error, 2),
            limit_value=limit,
            passed=passed,
            expiration_date=expiration,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def calculate_data_availability(
        self,
        parameter: CEMSParameter,
        hourly_statuses: List[ValidationStatus],
        reporting_period: str = 'quarter'
    ) -> DataAvailabilityReport:
        """
        Calculate data availability per Subpart D.

        Args:
            parameter: CEMS parameter
            hourly_statuses: Status for each hour in period
            reporting_period: 'quarter' or 'year'

        Returns:
            DataAvailabilityReport with availability percentage
        """
        total_hours = len(hourly_statuses)

        valid_hours = sum(
            1 for s in hourly_statuses
            if s == ValidationStatus.VALID
        )

        substitute_hours = sum(
            1 for s in hourly_statuses
            if s == ValidationStatus.SUBSTITUTED
        )

        missing_hours = sum(
            1 for s in hourly_statuses
            if s == ValidationStatus.MISSING
        )

        # Calculate availability (valid + substituted counts as available)
        available_hours = valid_hours + substitute_hours
        if total_hours > 0:
            availability = Decimal(available_hours) / Decimal(total_hours) * Decimal('100')
        else:
            availability = Decimal('0')

        availability = _apply_precision(availability, 2)
        meets_minimum = availability >= MIN_DATA_AVAILABILITY

        provenance_hash = calculate_provenance_hash(
            operation='data_availability',
            inputs={
                'parameter': parameter.value,
                'total_hours': total_hours,
                'reporting_period': reporting_period,
            },
            outputs={
                'availability_percent': str(availability),
                'meets_minimum': meets_minimum,
            }
        )

        return DataAvailabilityReport(
            parameter=parameter,
            reporting_period=reporting_period,
            total_hours=total_hours,
            valid_hours=valid_hours,
            substitute_hours=substitute_hours,
            missing_hours=missing_hours,
            availability_percent=availability,
            meets_minimum=meets_minimum,
            provenance_hash=provenance_hash
        )


# ==============================================================================
# BATCH VALIDATION FUNCTIONS
# ==============================================================================

def validate_hourly_data(
    readings: List[CEMSReading],
    validator: CEMSDataValidator,
    qa_status: Optional[Dict[CEMSParameter, QATestResult]] = None
) -> List[ValidationResult]:
    """
    Validate a batch of hourly CEMS readings.

    Args:
        readings: List of hourly readings
        validator: CEMSDataValidator instance
        qa_status: Current QA status by parameter

    Returns:
        List of ValidationResult for each reading
    """
    results: List[ValidationResult] = []

    for reading in readings:
        qa = None
        if qa_status and reading.parameter in qa_status:
            qa = qa_status[reading.parameter]

        result = validator.validate_reading(reading, qa)
        results.append(result)

    return results


def generate_validation_summary(
    results: List[ValidationResult]
) -> Dict[str, Any]:
    """
    Generate summary statistics from validation results.

    Args:
        results: List of validation results

    Returns:
        Summary dictionary with counts and percentages
    """
    total = len(results)
    if total == 0:
        return {'total': 0, 'valid_percent': Decimal('0')}

    valid_count = sum(1 for r in results if r.is_valid)
    by_status = {}
    for r in results:
        status = r.status.value
        by_status[status] = by_status.get(status, 0) + 1

    return {
        'total': total,
        'valid_count': valid_count,
        'valid_percent': _apply_precision(
            Decimal(valid_count) / Decimal(total) * Decimal('100'), 2
        ),
        'by_status': by_status,
        'error_count': sum(1 for r in results if not r.is_valid),
        'warning_count': sum(len(r.validation_warnings) for r in results),
    }


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'CEMSParameter',
    'ValidationStatus',
    'QATestType',
    'CEMSReading',
    'ValidationResult',
    'CalibrationCheck',
    'QATestResult',
    'DataAvailabilityReport',
    'CEMSDataValidator',
    'validate_hourly_data',
    'generate_validation_summary',
    'CEMS_RANGE_LIMITS',
    'CALIBRATION_ERROR_LIMITS',
    'RATA_LIMITS',
    'CGA_LIMITS',
    'LINEARITY_LIMITS',
    'MIN_DATA_AVAILABILITY',
]
