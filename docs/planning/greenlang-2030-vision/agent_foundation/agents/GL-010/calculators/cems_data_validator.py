"""
CEMS Data Validator Module for GL-010 EMISSIONWATCH.

This module provides comprehensive Continuous Emissions Monitoring System (CEMS)
data quality assurance per EPA 40 CFR Part 75 requirements. All validation
calculations are deterministic with full provenance tracking.

Features:
- Calibration error checking (daily, quarterly)
- Relative Accuracy Test Audit (RATA) validation
- Cylinder Gas Audit (CGA) validation
- Bias adjustment factor calculations
- Missing data substitution algorithms
- Daily calibration verification
- QA/QC status tracking and certification

Zero-Hallucination Guarantee:
- All calculations are deterministic (same input -> same output)
- No LLM involvement in calculation path
- Full provenance tracking with SHA-256 hashes
- Complete audit trails for regulatory compliance

References:
- EPA 40 CFR Part 75, Appendix A (CEMS specifications)
- EPA 40 CFR Part 75, Appendix B (QA/QC procedures)
- EPA 40 CFR Part 75, Appendix C (Missing data procedures)
- EPA 40 CFR Part 75, Appendix D (Fuel sampling and analysis)

Author: GreenLang GL-010 EMISSIONWATCH Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import json
import threading

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CEMSParameterType(str, Enum):
    """CEMS monitored parameter types per 40 CFR Part 75."""
    NOX = "nox"
    SO2 = "so2"
    CO2 = "co2"
    O2 = "o2"
    FLOW = "flow"
    MOISTURE = "moisture"
    OPACITY = "opacity"
    HCL = "hcl"
    HG = "hg"


class CalibrationGasLevel(str, Enum):
    """Calibration gas concentration levels."""
    ZERO = "zero"
    MID = "mid"
    HIGH = "high"
    SPAN = "span"


class QATestType(str, Enum):
    """QA/QC test types per 40 CFR Part 75."""
    DAILY_CALIBRATION = "daily_calibration"
    LINEARITY_CHECK = "linearity_check"
    RATA = "rata"  # Relative Accuracy Test Audit
    CGA = "cga"    # Cylinder Gas Audit
    LEAK_CHECK = "leak_check"
    BEAM_INTENSITY = "beam_intensity"


class DataQualityStatus(str, Enum):
    """Data quality status codes."""
    VALID = "valid"
    INVALID_CALIBRATION = "invalid_calibration"
    INVALID_RATA = "invalid_rata"
    SUBSTITUTE_STANDARD = "substitute_standard"
    SUBSTITUTE_MAXIMUM = "substitute_maximum"
    MISSING = "missing"
    OUT_OF_CONTROL = "out_of_control"


class MissingDataMethod(str, Enum):
    """Missing data substitution methods per 40 CFR Part 75 Appendix C."""
    STANDARD_90TH = "standard_90th"
    STANDARD_AVERAGE = "standard_average"
    MAXIMUM_VALUE = "maximum_value"
    FUEL_SPECIFIC = "fuel_specific"
    LOAD_SPECIFIC = "load_specific"


class BiasAdjustmentType(str, Enum):
    """Bias adjustment types."""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


# =============================================================================
# FROZEN DATACLASSES (Thread-Safe, Hashable)
# =============================================================================

@dataclass(frozen=True)
class CalibrationGasReference:
    """
    Certified calibration gas reference values.

    Attributes:
        parameter: Parameter type being calibrated
        gas_level: Calibration gas level (zero, mid, high)
        certified_value: Certified gas concentration
        uncertainty_percent: Certification uncertainty (%)
        expiration_date: Gas certification expiration
        cylinder_id: Gas cylinder identifier
    """
    parameter: CEMSParameterType
    gas_level: CalibrationGasLevel
    certified_value: Decimal
    uncertainty_percent: Decimal
    expiration_date: datetime
    cylinder_id: str


@dataclass(frozen=True)
class CalibrationResult:
    """
    Individual calibration test result.

    Attributes:
        test_datetime: When calibration was performed
        parameter: Parameter calibrated
        gas_level: Gas level used
        reference_value: Certified gas value
        measured_value: CEMS measured value
        calibration_error: Calculated error (%)
        passed: Whether test passed criteria
        error_limit: Applicable error limit (%)
        provenance_hash: SHA-256 hash for audit
    """
    test_datetime: datetime
    parameter: CEMSParameterType
    gas_level: CalibrationGasLevel
    reference_value: Decimal
    measured_value: Decimal
    calibration_error: Decimal
    passed: bool
    error_limit: Decimal
    provenance_hash: str


@dataclass(frozen=True)
class RATAResult:
    """
    Relative Accuracy Test Audit (RATA) result.

    Per 40 CFR Part 75 Appendix A Section 6.5.

    Attributes:
        test_date: Date of RATA test
        parameter: Parameter tested
        num_runs: Number of valid runs
        reference_method_values: Reference method results
        cems_values: Concurrent CEMS readings
        relative_accuracy: Calculated RA (%)
        bias: Mean difference (reference - CEMS)
        bias_adjustment_required: Whether BAF is required
        passed: Whether RATA passed
        ra_limit: Applicable RA limit (%)
        provenance_hash: SHA-256 hash for audit
    """
    test_date: datetime
    parameter: CEMSParameterType
    num_runs: int
    reference_method_values: Tuple[Decimal, ...]
    cems_values: Tuple[Decimal, ...]
    relative_accuracy: Decimal
    bias: Decimal
    bias_adjustment_required: bool
    passed: bool
    ra_limit: Decimal
    provenance_hash: str


@dataclass(frozen=True)
class CGAResult:
    """
    Cylinder Gas Audit (CGA) result.

    Per 40 CFR Part 75 Appendix B Section 2.2.

    Attributes:
        test_date: Date of CGA
        parameter: Parameter tested
        gas_level: Audit gas level used
        reference_value: Certified gas value
        measured_value: CEMS measured value
        error_percent: Calculated error (%)
        passed: Whether CGA passed
        error_limit: Applicable error limit (%)
        provenance_hash: SHA-256 hash for audit
    """
    test_date: datetime
    parameter: CEMSParameterType
    gas_level: CalibrationGasLevel
    reference_value: Decimal
    measured_value: Decimal
    error_percent: Decimal
    passed: bool
    error_limit: Decimal
    provenance_hash: str


@dataclass(frozen=True)
class BiasAdjustmentFactor:
    """
    Bias Adjustment Factor (BAF) per 40 CFR Part 75.

    Attributes:
        parameter: Parameter requiring adjustment
        adjustment_type: Additive or multiplicative
        factor_value: BAF value
        effective_date: When BAF became effective
        expiration_date: When BAF expires (next RATA)
        source_rata_date: RATA that determined BAF
        provenance_hash: SHA-256 hash for audit
    """
    parameter: CEMSParameterType
    adjustment_type: BiasAdjustmentType
    factor_value: Decimal
    effective_date: datetime
    expiration_date: Optional[datetime]
    source_rata_date: datetime
    provenance_hash: str


@dataclass(frozen=True)
class SubstitutedDataValue:
    """
    Substituted data value per missing data procedures.

    Attributes:
        timestamp: Hour for substituted value
        parameter: Parameter substituted
        original_status: Why data was missing/invalid
        substitution_method: Method used for substitution
        substituted_value: Calculated substitute value
        lookback_hours: Hours of data used in calculation
        data_availability_percent: Monitor availability (%)
        provenance_hash: SHA-256 hash for audit
    """
    timestamp: datetime
    parameter: CEMSParameterType
    original_status: DataQualityStatus
    substitution_method: MissingDataMethod
    substituted_value: Decimal
    lookback_hours: int
    data_availability_percent: Decimal
    provenance_hash: str


@dataclass(frozen=True)
class QAQCStatus:
    """
    QA/QC certification status for a CEMS parameter.

    Attributes:
        parameter: Parameter status
        is_certified: Currently certified (valid RATA)
        last_rata_date: Date of last valid RATA
        next_rata_deadline: Deadline for next RATA
        last_cga_date: Date of last valid CGA
        daily_cal_status: Current daily calibration status
        bias_adjustment_active: Whether BAF is applied
        data_availability_ytd: Year-to-date availability (%)
        status_code: Overall status code
        provenance_hash: SHA-256 hash for audit
    """
    parameter: CEMSParameterType
    is_certified: bool
    last_rata_date: Optional[datetime]
    next_rata_deadline: Optional[datetime]
    last_cga_date: Optional[datetime]
    daily_cal_status: DataQualityStatus
    bias_adjustment_active: bool
    data_availability_ytd: Decimal
    status_code: DataQualityStatus
    provenance_hash: str


@dataclass(frozen=True)
class ValidationResult:
    """
    Complete CEMS data validation result.

    Attributes:
        timestamp: Hour validated
        parameter: Parameter validated
        raw_value: Original measured value
        validated_value: Value after validation/adjustment
        is_valid: Whether value is valid
        quality_status: Quality status code
        bias_adjusted: Whether BAF was applied
        substituted: Whether value was substituted
        validation_steps: Audit trail of validation
        provenance_hash: SHA-256 hash for audit
    """
    timestamp: datetime
    parameter: CEMSParameterType
    raw_value: Decimal
    validated_value: Decimal
    is_valid: bool
    quality_status: DataQualityStatus
    bias_adjusted: bool
    substituted: bool
    validation_steps: Tuple[str, ...]
    provenance_hash: str


# =============================================================================
# INPUT MODELS (Pydantic)
# =============================================================================

class CalibrationTestInput(BaseModel):
    """Input for daily calibration error calculation."""
    parameter: CEMSParameterType = Field(description="Parameter being calibrated")
    reference_value: float = Field(ge=0, description="Certified gas value")
    measured_value: float = Field(description="CEMS measured value")
    span_value: float = Field(gt=0, description="Analyzer span setting")
    test_datetime: Optional[datetime] = Field(default=None, description="Test time")

    @field_validator("test_datetime", mode="before")
    @classmethod
    def set_datetime(cls, v):
        return v or datetime.utcnow()


class RATATestInput(BaseModel):
    """Input for RATA calculation per 40 CFR Part 75."""
    parameter: CEMSParameterType = Field(description="Parameter tested")
    reference_method_values: List[float] = Field(
        min_length=9,
        description="Reference method test run values (minimum 9)"
    )
    cems_values: List[float] = Field(
        min_length=9,
        description="Concurrent CEMS values"
    )
    test_date: Optional[datetime] = Field(default=None, description="RATA date")
    applicable_span: float = Field(gt=0, description="Applicable span for RA%")

    @field_validator("cems_values")
    @classmethod
    def validate_matching_length(cls, v, info):
        ref_values = info.data.get("reference_method_values", [])
        if len(v) != len(ref_values):
            raise ValueError("CEMS values must match reference method values count")
        return v


class CGATestInput(BaseModel):
    """Input for Cylinder Gas Audit calculation."""
    parameter: CEMSParameterType = Field(description="Parameter tested")
    gas_level: CalibrationGasLevel = Field(description="Audit gas level")
    reference_value: float = Field(ge=0, description="Certified gas value")
    measured_value: float = Field(description="CEMS measured value")
    span_value: float = Field(gt=0, description="Analyzer span setting")
    test_date: Optional[datetime] = Field(default=None, description="CGA date")


class MissingDataInput(BaseModel):
    """Input for missing data substitution calculation."""
    parameter: CEMSParameterType = Field(description="Parameter with missing data")
    missing_timestamp: datetime = Field(description="Hour needing substitution")
    quality_assured_hours: int = Field(
        ge=0, le=8760,
        description="Hours since last valid RATA/CGA"
    )
    operating_hours_lookback: int = Field(
        default=2160, ge=720, le=8760,
        description="Hours to look back for substitute data"
    )
    available_data: List[float] = Field(
        default=[],
        description="Available valid hourly values in lookback period"
    )
    monitor_availability: float = Field(
        default=100.0, ge=0, le=100,
        description="Monitor data availability (%)"
    )


class HourlyDataInput(BaseModel):
    """Input for hourly CEMS data validation."""
    timestamp: datetime = Field(description="Data hour (top of hour)")
    parameter: CEMSParameterType = Field(description="Parameter measured")
    measured_value: float = Field(description="Measured concentration/flow")
    data_status_code: Optional[str] = Field(default=None, description="Status code")
    daily_calibration_passed: bool = Field(default=True, description="Daily cal status")
    rata_valid: bool = Field(default=True, description="Current RATA validity")
    bias_adjustment_factor: Optional[float] = Field(
        default=None,
        description="Active BAF if applicable"
    )


# =============================================================================
# CEMS DATA VALIDATOR CLASS
# =============================================================================

class CEMSDataValidator:
    """
    CEMS Data Quality Assurance Validator per 40 CFR Part 75.

    Provides deterministic, zero-hallucination validation of CEMS data
    including calibration error checking, RATA validation, CGA validation,
    bias adjustment calculations, and missing data substitution.

    Thread Safety:
        All methods are thread-safe. Internal caches use thread locks.

    Example:
        >>> validator = CEMSDataValidator()
        >>> cal_result = validator.calculate_calibration_error(
        ...     CalibrationTestInput(
        ...         parameter=CEMSParameterType.NOX,
        ...         reference_value=100.0,
        ...         measured_value=102.5,
        ...         span_value=500.0
        ...     )
        ... )
        >>> print(f"Calibration error: {cal_result.calibration_error}%")
    """

    # Per 40 CFR Part 75 Appendix A - Calibration error limits
    CALIBRATION_ERROR_LIMITS: Dict[CEMSParameterType, Decimal] = {
        CEMSParameterType.NOX: Decimal("2.5"),
        CEMSParameterType.SO2: Decimal("2.5"),
        CEMSParameterType.CO2: Decimal("0.5"),
        CEMSParameterType.O2: Decimal("0.5"),
        CEMSParameterType.FLOW: Decimal("3.0"),
        CEMSParameterType.MOISTURE: Decimal("1.5"),
    }

    # Per 40 CFR Part 75 Appendix A - Relative accuracy limits
    RATA_LIMITS: Dict[CEMSParameterType, Decimal] = {
        CEMSParameterType.NOX: Decimal("10.0"),
        CEMSParameterType.SO2: Decimal("10.0"),
        CEMSParameterType.CO2: Decimal("10.0"),
        CEMSParameterType.O2: Decimal("1.0"),  # Absolute for diluent
        CEMSParameterType.FLOW: Decimal("10.0"),
    }

    # Per 40 CFR Part 75 Appendix B - CGA error limits
    CGA_ERROR_LIMITS: Dict[CEMSParameterType, Decimal] = {
        CEMSParameterType.NOX: Decimal("5.0"),
        CEMSParameterType.SO2: Decimal("5.0"),
        CEMSParameterType.CO2: Decimal("1.0"),
        CEMSParameterType.O2: Decimal("1.0"),
    }

    # Missing data lookback periods (hours)
    LOOKBACK_PERIODS = {
        "standard_90th": 2160,  # 90 days
        "quality_assured": 720,  # 30 days
    }

    def __init__(self):
        """Initialize CEMS Data Validator with thread-safe caching."""
        self._cache_lock = threading.Lock()
        self._rata_cache: Dict[str, RATAResult] = {}
        self._baf_cache: Dict[str, BiasAdjustmentFactor] = {}
        self._qa_status_cache: Dict[str, QAQCStatus] = {}

    def calculate_calibration_error(
        self,
        test_input: CalibrationTestInput,
        precision: int = 2
    ) -> CalibrationResult:
        """
        Calculate calibration error per 40 CFR Part 75 Appendix A.

        Formula: CE = |R - M| / S * 100
        Where:
        - CE = Calibration error (%)
        - R = Reference (certified gas) value
        - M = Monitor (CEMS) response
        - S = Span value

        Args:
            test_input: Calibration test parameters
            precision: Decimal places in result

        Returns:
            CalibrationResult with pass/fail status and provenance

        Reference:
            40 CFR Part 75, Appendix A, Section 6.3.1
        """
        ref = Decimal(str(test_input.reference_value))
        meas = Decimal(str(test_input.measured_value))
        span = Decimal(str(test_input.span_value))

        # Calculate calibration error
        error = abs(ref - meas) / span * Decimal("100")
        error = self._apply_precision(error, precision)

        # Get applicable limit
        limit = self.CALIBRATION_ERROR_LIMITS.get(
            test_input.parameter,
            Decimal("2.5")
        )

        # Determine pass/fail
        passed = error <= limit

        # Calculate provenance hash
        provenance_data = {
            "test_type": "daily_calibration",
            "parameter": test_input.parameter.value,
            "reference_value": str(ref),
            "measured_value": str(meas),
            "span_value": str(span),
            "calibration_error": str(error),
            "passed": passed,
            "timestamp": test_input.test_datetime.isoformat()
        }
        provenance_hash = self._calculate_hash(provenance_data)

        # Determine gas level from reference value relative to span
        ref_percent = ref / span * Decimal("100")
        if ref_percent < Decimal("5"):
            gas_level = CalibrationGasLevel.ZERO
        elif ref_percent < Decimal("60"):
            gas_level = CalibrationGasLevel.MID
        else:
            gas_level = CalibrationGasLevel.HIGH

        return CalibrationResult(
            test_datetime=test_input.test_datetime,
            parameter=test_input.parameter,
            gas_level=gas_level,
            reference_value=ref,
            measured_value=meas,
            calibration_error=error,
            passed=passed,
            error_limit=limit,
            provenance_hash=provenance_hash
        )

    def calculate_rata(
        self,
        test_input: RATATestInput,
        precision: int = 2
    ) -> RATAResult:
        """
        Calculate Relative Accuracy Test Audit per 40 CFR Part 75.

        Formula: RA = (|d_avg| + |CC|) / RM_avg * 100

        Where:
        - d_avg = Mean difference (RM - CEMS)
        - CC = Confidence coefficient = t * S_d / sqrt(n)
        - RM_avg = Mean reference method value
        - t = t-statistic for n-1 degrees of freedom (2-tailed, 0.025)
        - S_d = Standard deviation of differences

        Args:
            test_input: RATA test data with reference and CEMS values
            precision: Decimal places in result

        Returns:
            RATAResult with relative accuracy and bias determination

        Reference:
            40 CFR Part 75, Appendix A, Section 6.5
        """
        n = len(test_input.reference_method_values)

        # Convert to Decimal
        rm_values = tuple(Decimal(str(v)) for v in test_input.reference_method_values)
        cems_values = tuple(Decimal(str(v)) for v in test_input.cems_values)

        # Calculate differences (RM - CEMS)
        differences = tuple(rm - cems for rm, cems in zip(rm_values, cems_values))

        # Mean values
        rm_avg = sum(rm_values) / Decimal(str(n))
        cems_avg = sum(cems_values) / Decimal(str(n))
        d_avg = sum(differences) / Decimal(str(n))  # Mean bias

        # Standard deviation of differences
        if n > 1:
            variance = sum((d - d_avg) ** 2 for d in differences) / Decimal(str(n - 1))
            s_d = variance.sqrt()
        else:
            s_d = Decimal("0")

        # t-statistic lookup (two-tailed, alpha=0.05)
        t_values = {
            9: Decimal("2.306"),
            10: Decimal("2.262"),
            11: Decimal("2.228"),
            12: Decimal("2.201"),
            15: Decimal("2.145"),
            20: Decimal("2.093"),
        }
        t_stat = t_values.get(n, Decimal("2.0"))

        # Confidence coefficient
        cc = t_stat * s_d / Decimal(str(n)).sqrt() if n > 0 else Decimal("0")

        # Relative accuracy calculation
        if rm_avg > Decimal("0"):
            ra = (abs(d_avg) + abs(cc)) / rm_avg * Decimal("100")
        else:
            ra = Decimal("0")

        ra = self._apply_precision(ra, precision)

        # Get applicable limit
        limit = self.RATA_LIMITS.get(test_input.parameter, Decimal("10.0"))
        passed = ra <= limit

        # Determine if bias adjustment required
        # Per 40 CFR Part 75, BAF required if |d_avg| > |CC| AND RA <= limit
        bias = d_avg
        bias_adjustment_required = (
            passed and
            abs(d_avg) > abs(cc) and
            d_avg != Decimal("0")
        )

        # Calculate provenance hash
        provenance_data = {
            "test_type": "rata",
            "parameter": test_input.parameter.value,
            "num_runs": n,
            "rm_values": [str(v) for v in rm_values],
            "cems_values": [str(v) for v in cems_values],
            "rm_avg": str(rm_avg),
            "d_avg": str(d_avg),
            "s_d": str(s_d),
            "cc": str(cc),
            "ra": str(ra),
            "passed": passed,
            "bias_adjustment_required": bias_adjustment_required,
            "test_date": (test_input.test_date or datetime.utcnow()).isoformat()
        }
        provenance_hash = self._calculate_hash(provenance_data)

        result = RATAResult(
            test_date=test_input.test_date or datetime.utcnow(),
            parameter=test_input.parameter,
            num_runs=n,
            reference_method_values=rm_values,
            cems_values=cems_values,
            relative_accuracy=ra,
            bias=self._apply_precision(bias, precision + 2),
            bias_adjustment_required=bias_adjustment_required,
            passed=passed,
            ra_limit=limit,
            provenance_hash=provenance_hash
        )

        # Cache result
        cache_key = f"{test_input.parameter.value}_{result.test_date.isoformat()}"
        with self._cache_lock:
            self._rata_cache[cache_key] = result

        return result

    def calculate_cga(
        self,
        test_input: CGATestInput,
        precision: int = 2
    ) -> CGAResult:
        """
        Calculate Cylinder Gas Audit error per 40 CFR Part 75 Appendix B.

        Formula: CGA Error = |R - M| / S * 100

        Where:
        - R = Reference (certified audit gas) value
        - M = CEMS response
        - S = Span value

        Note: If reference < 5% of span, use alternative formula:
        CGA Error = |R - M| / R * 100 (percent of reference)

        Args:
            test_input: CGA test parameters
            precision: Decimal places in result

        Returns:
            CGAResult with pass/fail status and provenance

        Reference:
            40 CFR Part 75, Appendix B, Section 2.2.3
        """
        ref = Decimal(str(test_input.reference_value))
        meas = Decimal(str(test_input.measured_value))
        span = Decimal(str(test_input.span_value))

        # Determine calculation method
        ref_percent_span = ref / span * Decimal("100")

        if ref_percent_span < Decimal("5"):
            # Low-level alternative: percent of reference
            error = abs(ref - meas) / ref * Decimal("100") if ref > 0 else Decimal("0")
        else:
            # Standard: percent of span
            error = abs(ref - meas) / span * Decimal("100")

        error = self._apply_precision(error, precision)

        # Get applicable limit
        limit = self.CGA_ERROR_LIMITS.get(test_input.parameter, Decimal("5.0"))
        passed = error <= limit

        # Provenance hash
        provenance_data = {
            "test_type": "cga",
            "parameter": test_input.parameter.value,
            "gas_level": test_input.gas_level.value,
            "reference_value": str(ref),
            "measured_value": str(meas),
            "span_value": str(span),
            "error_percent": str(error),
            "passed": passed,
            "test_date": (test_input.test_date or datetime.utcnow()).isoformat()
        }
        provenance_hash = self._calculate_hash(provenance_data)

        return CGAResult(
            test_date=test_input.test_date or datetime.utcnow(),
            parameter=test_input.parameter,
            gas_level=test_input.gas_level,
            reference_value=ref,
            measured_value=meas,
            error_percent=error,
            passed=passed,
            error_limit=limit,
            provenance_hash=provenance_hash
        )

    def calculate_bias_adjustment_factor(
        self,
        rata_result: RATAResult,
        applicable_span: Union[float, Decimal],
        precision: int = 4
    ) -> Optional[BiasAdjustmentFactor]:
        """
        Calculate Bias Adjustment Factor from RATA results.

        Per 40 CFR Part 75, Appendix A, Section 7.6:
        - If |bias| > |CC| and RATA passes, BAF is required
        - BAF = 1 + (d_avg / CEMS_avg) for multiplicative
        - BAF = d_avg for additive (when CEMS_avg near zero)

        Args:
            rata_result: Completed RATA result
            applicable_span: Span value for BAF determination
            precision: Decimal places in result

        Returns:
            BiasAdjustmentFactor if required, None otherwise

        Reference:
            40 CFR Part 75, Appendix A, Section 7.6
        """
        if not rata_result.bias_adjustment_required:
            return None

        span = Decimal(str(applicable_span))
        bias = rata_result.bias

        # Calculate CEMS average
        cems_avg = sum(rata_result.cems_values) / Decimal(str(rata_result.num_runs))

        # Determine adjustment type
        # Use additive if CEMS average < 10% of span
        if cems_avg < span * Decimal("0.10"):
            adjustment_type = BiasAdjustmentType.ADDITIVE
            factor_value = bias
        else:
            adjustment_type = BiasAdjustmentType.MULTIPLICATIVE
            factor_value = Decimal("1") + (bias / cems_avg) if cems_avg != 0 else Decimal("1")

        factor_value = self._apply_precision(factor_value, precision)

        # Provenance hash
        provenance_data = {
            "source": "rata",
            "parameter": rata_result.parameter.value,
            "rata_date": rata_result.test_date.isoformat(),
            "bias": str(rata_result.bias),
            "cems_avg": str(cems_avg),
            "adjustment_type": adjustment_type.value,
            "factor_value": str(factor_value)
        }
        provenance_hash = self._calculate_hash(provenance_data)

        # BAF effective for one year or until next RATA
        effective_date = rata_result.test_date
        expiration_date = rata_result.test_date + timedelta(days=365)

        result = BiasAdjustmentFactor(
            parameter=rata_result.parameter,
            adjustment_type=adjustment_type,
            factor_value=factor_value,
            effective_date=effective_date,
            expiration_date=expiration_date,
            source_rata_date=rata_result.test_date,
            provenance_hash=provenance_hash
        )

        # Cache BAF
        cache_key = f"{rata_result.parameter.value}_baf"
        with self._cache_lock:
            self._baf_cache[cache_key] = result

        return result

    def calculate_missing_data_substitute(
        self,
        missing_input: MissingDataInput,
        precision: int = 2
    ) -> SubstitutedDataValue:
        """
        Calculate substitute data value per 40 CFR Part 75 Appendix C.

        Substitution hierarchy:
        1. If monitor availability >= 90%: Use 90th percentile of lookback data
        2. If availability 80-90%: Use average of maximum values from prior QA hours
        3. If availability < 80%: Use maximum potential value

        Args:
            missing_input: Missing data parameters
            precision: Decimal places in result

        Returns:
            SubstitutedDataValue with calculated substitute

        Reference:
            40 CFR Part 75, Appendix C, Section 2.4
        """
        availability = Decimal(str(missing_input.monitor_availability))
        data = [Decimal(str(v)) for v in missing_input.available_data]

        validation_steps = []

        # Determine substitution method based on availability
        if availability >= Decimal("90"):
            method = MissingDataMethod.STANDARD_90TH
            validation_steps.append(f"Availability {availability}% >= 90%: Using 90th percentile")

            if data:
                sorted_data = sorted(data)
                percentile_index = int(len(sorted_data) * 0.90)
                substitute_value = sorted_data[min(percentile_index, len(sorted_data) - 1)]
            else:
                substitute_value = Decimal("0")

        elif availability >= Decimal("80"):
            method = MissingDataMethod.STANDARD_AVERAGE
            validation_steps.append(f"Availability {availability}% 80-90%: Using average")

            if data:
                substitute_value = sum(data) / Decimal(str(len(data)))
            else:
                substitute_value = Decimal("0")

        else:
            method = MissingDataMethod.MAXIMUM_VALUE
            validation_steps.append(f"Availability {availability}% < 80%: Using maximum")

            if data:
                substitute_value = max(data)
            else:
                substitute_value = Decimal("0")

        substitute_value = self._apply_precision(substitute_value, precision)
        validation_steps.append(f"Substitute value calculated: {substitute_value}")

        # Provenance hash
        provenance_data = {
            "missing_timestamp": missing_input.missing_timestamp.isoformat(),
            "parameter": missing_input.parameter.value,
            "method": method.value,
            "availability": str(availability),
            "data_count": len(data),
            "substitute_value": str(substitute_value)
        }
        provenance_hash = self._calculate_hash(provenance_data)

        return SubstitutedDataValue(
            timestamp=missing_input.missing_timestamp,
            parameter=missing_input.parameter,
            original_status=DataQualityStatus.MISSING,
            substitution_method=method,
            substituted_value=substitute_value,
            lookback_hours=missing_input.operating_hours_lookback,
            data_availability_percent=availability,
            provenance_hash=provenance_hash
        )

    def validate_hourly_data(
        self,
        data_input: HourlyDataInput,
        active_baf: Optional[BiasAdjustmentFactor] = None,
        precision: int = 2
    ) -> ValidationResult:
        """
        Validate hourly CEMS data and apply necessary adjustments.

        Validation sequence:
        1. Check daily calibration status
        2. Check RATA validity
        3. Apply bias adjustment factor if active
        4. Determine final quality status

        Args:
            data_input: Hourly data to validate
            active_baf: Active bias adjustment factor (if any)
            precision: Decimal places in result

        Returns:
            ValidationResult with validated value and quality status
        """
        raw_value = Decimal(str(data_input.measured_value))
        validated_value = raw_value
        validation_steps = []
        is_valid = True
        bias_adjusted = False

        # Step 1: Check daily calibration
        if not data_input.daily_calibration_passed:
            validation_steps.append("Daily calibration FAILED - data flagged")
            quality_status = DataQualityStatus.INVALID_CALIBRATION
            is_valid = False
        else:
            validation_steps.append("Daily calibration: PASSED")
            quality_status = DataQualityStatus.VALID

        # Step 2: Check RATA validity
        if not data_input.rata_valid:
            validation_steps.append("RATA not valid - data flagged")
            quality_status = DataQualityStatus.INVALID_RATA
            is_valid = False
        else:
            validation_steps.append("RATA status: VALID")

        # Step 3: Apply bias adjustment if applicable
        if is_valid and active_baf is not None:
            if active_baf.adjustment_type == BiasAdjustmentType.MULTIPLICATIVE:
                validated_value = raw_value * active_baf.factor_value
                validation_steps.append(
                    f"BAF applied (multiplicative): {raw_value} * {active_baf.factor_value} = {validated_value}"
                )
            else:
                validated_value = raw_value + active_baf.factor_value
                validation_steps.append(
                    f"BAF applied (additive): {raw_value} + {active_baf.factor_value} = {validated_value}"
                )
            bias_adjusted = True

        # Apply precision
        validated_value = self._apply_precision(validated_value, precision)

        # Provenance hash
        provenance_data = {
            "timestamp": data_input.timestamp.isoformat(),
            "parameter": data_input.parameter.value,
            "raw_value": str(raw_value),
            "validated_value": str(validated_value),
            "is_valid": is_valid,
            "quality_status": quality_status.value,
            "bias_adjusted": bias_adjusted,
            "validation_steps": validation_steps
        }
        provenance_hash = self._calculate_hash(provenance_data)

        return ValidationResult(
            timestamp=data_input.timestamp,
            parameter=data_input.parameter,
            raw_value=raw_value,
            validated_value=validated_value,
            is_valid=is_valid,
            quality_status=quality_status,
            bias_adjusted=bias_adjusted,
            substituted=False,
            validation_steps=tuple(validation_steps),
            provenance_hash=provenance_hash
        )

    def get_qa_status(
        self,
        parameter: CEMSParameterType,
        last_rata: Optional[RATAResult] = None,
        last_cga: Optional[CGAResult] = None,
        last_calibration: Optional[CalibrationResult] = None,
        data_availability_ytd: float = 100.0
    ) -> QAQCStatus:
        """
        Get comprehensive QA/QC status for a CEMS parameter.

        Args:
            parameter: Parameter to check status
            last_rata: Most recent RATA result
            last_cga: Most recent CGA result
            last_calibration: Most recent daily calibration
            data_availability_ytd: Year-to-date availability (%)

        Returns:
            QAQCStatus with certification status and deadlines
        """
        now = datetime.utcnow()

        # Determine RATA status
        rata_valid = False
        last_rata_date = None
        next_rata_deadline = None

        if last_rata is not None and last_rata.passed:
            last_rata_date = last_rata.test_date
            # RATA valid for 4 quarters (approximately 1 year)
            next_rata_deadline = last_rata_date + timedelta(days=365)
            rata_valid = now < next_rata_deadline

        # Determine CGA status
        last_cga_date = last_cga.test_date if last_cga is not None else None

        # Determine daily calibration status
        daily_cal_status = DataQualityStatus.VALID
        if last_calibration is not None:
            if not last_calibration.passed:
                daily_cal_status = DataQualityStatus.INVALID_CALIBRATION
            elif (now - last_calibration.test_datetime).total_seconds() > 26 * 3600:
                # More than 26 hours since last calibration
                daily_cal_status = DataQualityStatus.OUT_OF_CONTROL
        else:
            daily_cal_status = DataQualityStatus.MISSING

        # Check for active BAF
        baf_key = f"{parameter.value}_baf"
        with self._cache_lock:
            active_baf = self._baf_cache.get(baf_key)
        bias_adjustment_active = (
            active_baf is not None and
            (active_baf.expiration_date is None or now < active_baf.expiration_date)
        )

        # Determine overall status
        if not rata_valid:
            overall_status = DataQualityStatus.INVALID_RATA
            is_certified = False
        elif daily_cal_status != DataQualityStatus.VALID:
            overall_status = daily_cal_status
            is_certified = rata_valid
        else:
            overall_status = DataQualityStatus.VALID
            is_certified = True

        # Provenance hash
        provenance_data = {
            "parameter": parameter.value,
            "check_time": now.isoformat(),
            "is_certified": is_certified,
            "rata_valid": rata_valid,
            "daily_cal_status": daily_cal_status.value,
            "overall_status": overall_status.value,
            "data_availability": str(data_availability_ytd)
        }
        provenance_hash = self._calculate_hash(provenance_data)

        return QAQCStatus(
            parameter=parameter,
            is_certified=is_certified,
            last_rata_date=last_rata_date,
            next_rata_deadline=next_rata_deadline,
            last_cga_date=last_cga_date,
            daily_cal_status=daily_cal_status,
            bias_adjustment_active=bias_adjustment_active,
            data_availability_ytd=Decimal(str(data_availability_ytd)),
            status_code=overall_status,
            provenance_hash=provenance_hash
        )

    def clear_cache(self) -> None:
        """Clear all internal caches (thread-safe)."""
        with self._cache_lock:
            self._rata_cache.clear()
            self._baf_cache.clear()
            self._qa_status_cache.clear()

    @staticmethod
    def _apply_precision(value: Decimal, precision: int) -> Decimal:
        """Apply decimal precision with ROUND_HALF_UP."""
        if precision < 0:
            raise ValueError(f"Precision must be non-negative, got {precision}")
        quantize_str = "0." + "0" * precision if precision > 0 else "1"
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    @staticmethod
    def _calculate_hash(data: Any) -> str:
        """Calculate SHA-256 hash of data for provenance."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global validator instance with thread-safe singleton pattern
_validator_instance: Optional[CEMSDataValidator] = None
_validator_lock = threading.Lock()


def get_cems_validator() -> CEMSDataValidator:
    """Get or create global CEMS validator instance (thread-safe)."""
    global _validator_instance
    if _validator_instance is None:
        with _validator_lock:
            if _validator_instance is None:
                _validator_instance = CEMSDataValidator()
    return _validator_instance


def validate_calibration(
    parameter: str,
    reference_value: float,
    measured_value: float,
    span_value: float
) -> CalibrationResult:
    """
    Convenience function to validate a calibration test.

    Args:
        parameter: Parameter type (nox, so2, co2, o2, flow)
        reference_value: Certified gas value
        measured_value: CEMS measured value
        span_value: Analyzer span setting

    Returns:
        CalibrationResult with pass/fail status
    """
    validator = get_cems_validator()
    test_input = CalibrationTestInput(
        parameter=CEMSParameterType(parameter.lower()),
        reference_value=reference_value,
        measured_value=measured_value,
        span_value=span_value
    )
    return validator.calculate_calibration_error(test_input)


def calculate_rata_result(
    parameter: str,
    reference_values: List[float],
    cems_values: List[float],
    applicable_span: float
) -> RATAResult:
    """
    Convenience function to calculate RATA result.

    Args:
        parameter: Parameter tested
        reference_values: Reference method test run values
        cems_values: Concurrent CEMS readings
        applicable_span: Span for relative accuracy calculation

    Returns:
        RATAResult with relative accuracy and bias info
    """
    validator = get_cems_validator()
    test_input = RATATestInput(
        parameter=CEMSParameterType(parameter.lower()),
        reference_method_values=reference_values,
        cems_values=cems_values,
        applicable_span=applicable_span
    )
    return validator.calculate_rata(test_input)


def get_substitute_value(
    parameter: str,
    missing_timestamp: datetime,
    available_data: List[float],
    monitor_availability: float
) -> SubstitutedDataValue:
    """
    Convenience function to get substitute data value.

    Args:
        parameter: Parameter with missing data
        missing_timestamp: Hour needing substitution
        available_data: Available valid hourly values
        monitor_availability: Monitor availability (%)

    Returns:
        SubstitutedDataValue with calculated substitute
    """
    validator = get_cems_validator()
    missing_input = MissingDataInput(
        parameter=CEMSParameterType(parameter.lower()),
        missing_timestamp=missing_timestamp,
        quality_assured_hours=720,
        available_data=available_data,
        monitor_availability=monitor_availability
    )
    return validator.calculate_missing_data_substitute(missing_input)
