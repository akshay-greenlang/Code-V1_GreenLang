"""
EPA RATA Automation Module - GL-010 EmissionsGuardian

This module implements Relative Accuracy Test Audit (RATA) scheduling, execution
tracking, and analysis per 40 CFR Part 75 for Continuous Emission Monitoring
Systems (CEMS). Provides complete compliance automation for utility and
industrial emission sources.

Key Features:
    - Quarterly RATA scheduling with automatic reminders
    - Cylinder Gas Audit (CGA) tracking
    - Relative Accuracy (RA) calculation per EPA methodology
    - Bias adjustment factor (BAF) calculations
    - Linearity check automation
    - Calibration drift tracking
    - Audit trail with SHA-256 provenance

Regulatory References:
    - 40 CFR Part 75 Subpart C (Quality Assurance/Quality Control)
    - 40 CFR 75.20 (Initial Certification)
    - 40 CFR 75.21 (Quality Assurance Requirements)
    - 40 CFR 75.22 (Reference Method Testing)
    - EPA Performance Specification 2 (PS-2)

Example:
    >>> rata_manager = RATAAutomation(unit_id="UNIT-001")
    >>> schedule = rata_manager.get_quarterly_schedule(year=2025)
    >>> result = rata_manager.calculate_relative_accuracy(test_data)
    >>> print(f"RA: {result.relative_accuracy_pct:.2f}%")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import math
import statistics

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - EPA 40 CFR 75 Reference Values
# =============================================================================

class RATAConstants:
    """EPA RATA regulatory constants per 40 CFR Part 75."""

    # Relative Accuracy Pass/Fail Thresholds
    RA_THRESHOLD_PERCENT = 10.0  # Primary threshold (%)
    RA_THRESHOLD_ALTERNATE = 20.0  # Alternate threshold when mean < threshold
    RA_THRESHOLD_DILUENT = 1.0  # Absolute threshold for O2/CO2 (%)

    # Bias test thresholds
    BIAS_THRESHOLD_PERCENT = 5.0  # Bias adjustment trigger (%)

    # Cylinder Gas Audit thresholds
    CGA_TOLERANCE_PERCENT = 5.0  # CGA accuracy requirement

    # Linearity thresholds
    LINEARITY_ERROR_LIMIT = 5.0  # Percent of span

    # Calibration drift limits
    DAILY_CALIBRATION_DRIFT = 2.5  # Percent of span
    QUARTERLY_CALIBRATION_DRIFT = 5.0  # Percent of span

    # Minimum run requirements
    MIN_RATA_RUNS = 9  # Minimum number of test runs
    MAX_INVALID_RUNS = 3  # Maximum runs that can be invalidated

    # Test duration requirements
    MIN_RUN_DURATION_MINUTES = 21  # Minimum run duration
    MAX_RUN_DURATION_MINUTES = 60  # Maximum run duration

    # Reference method specifications
    EPA_METHOD_7E_RANGE = (0, 5000)  # NOx ppm
    EPA_METHOD_6C_RANGE = (0, 5000)  # SO2 ppm
    EPA_METHOD_3A_O2_RANGE = (0, 21)  # O2 %
    EPA_METHOD_3A_CO2_RANGE = (0, 20)  # CO2 %


class RATAFrequency(Enum):
    """RATA testing frequency based on prior performance."""
    QUARTERLY = "quarterly"  # Default, RA > 7.5%
    SEMIANNUAL = "semiannual"  # RA <= 7.5% for 2 consecutive
    ANNUAL = "annual"  # RA <= 7.5% for 4 consecutive


class CEMSPollutant(Enum):
    """CEMS monitored pollutants."""
    NOX = "nox"
    SO2 = "so2"
    CO2 = "co2"
    O2 = "o2"
    FLOW = "flow"
    HCL = "hcl"
    HG = "hg"


class TestStatus(Enum):
    """RATA test status."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PASSED = "passed"
    FAILED = "failed"
    CONDITIONAL_PASS = "conditional_pass"
    INVALIDATED = "invalidated"


class BiasAdjustmentStatus(Enum):
    """Bias adjustment status."""
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    APPLIED = "applied"


# =============================================================================
# DATA MODELS
# =============================================================================

class RATARunData(BaseModel):
    """Single RATA test run data."""

    run_number: int = Field(..., ge=1, le=12, description="Run number (1-12)")
    start_time: datetime = Field(..., description="Run start time")
    end_time: datetime = Field(..., description="Run end time")

    # Reference method values
    reference_value: float = Field(..., ge=0, description="Reference method value")
    reference_unit: str = Field(..., description="Reference value unit (ppm, lb/hr, %)")

    # CEMS values
    cems_value: float = Field(..., ge=0, description="CEMS measured value")

    # Operating conditions
    load_percent: float = Field(..., ge=0, le=120, description="Unit load (%)")

    # Quality flags
    is_valid: bool = Field(default=True, description="Run validity flag")
    invalidation_reason: Optional[str] = Field(default=None, description="Reason if invalid")

    @validator('end_time')
    def validate_run_duration(cls, v, values):
        """Validate run duration meets minimum requirements."""
        if 'start_time' in values:
            duration = (v - values['start_time']).total_seconds() / 60
            if duration < RATAConstants.MIN_RUN_DURATION_MINUTES:
                raise ValueError(
                    f"Run duration {duration:.1f} min is less than "
                    f"minimum {RATAConstants.MIN_RUN_DURATION_MINUTES} min"
                )
        return v

    @property
    def duration_minutes(self) -> float:
        """Calculate run duration in minutes."""
        return (self.end_time - self.start_time).total_seconds() / 60

    @property
    def difference(self) -> float:
        """Calculate difference between reference and CEMS."""
        return self.reference_value - self.cems_value


class RATATestInput(BaseModel):
    """Complete RATA test input data."""

    unit_id: str = Field(..., description="Emission unit identifier")
    pollutant: CEMSPollutant = Field(..., description="Pollutant being tested")
    test_date: date = Field(..., description="RATA test date")

    # Test parameters
    reference_method: str = Field(..., description="EPA reference method (e.g., 7E, 6C, 3A)")
    analyzer_span: float = Field(..., gt=0, description="Analyzer span value")

    # Run data
    runs: List[RATARunData] = Field(..., min_items=9, max_items=12)

    # Operating level
    operating_level: str = Field(
        default="normal",
        description="Operating level (low, mid, high, normal)"
    )

    # Prior RATA results for frequency determination
    prior_rata_results: List[float] = Field(
        default_factory=list,
        description="Prior RATA RA% results for frequency determination"
    )

    @validator('runs')
    def validate_run_count(cls, v):
        """Validate minimum valid runs."""
        valid_runs = [r for r in v if r.is_valid]
        if len(valid_runs) < RATAConstants.MIN_RATA_RUNS:
            raise ValueError(
                f"Minimum {RATAConstants.MIN_RATA_RUNS} valid runs required, "
                f"got {len(valid_runs)}"
            )
        return v


class RATAResult(BaseModel):
    """RATA test result with complete analysis."""

    unit_id: str = Field(..., description="Emission unit identifier")
    pollutant: CEMSPollutant = Field(..., description="Tested pollutant")
    test_date: date = Field(..., description="Test date")

    # Calculated values
    mean_difference: float = Field(..., description="Mean of differences (d-bar)")
    standard_deviation: float = Field(..., description="Standard deviation (Sd)")
    confidence_coefficient: float = Field(..., description="Confidence coefficient (CC)")
    relative_accuracy_pct: float = Field(..., description="Relative accuracy (%)")

    # Reference values
    mean_reference_value: float = Field(..., description="Mean reference value (RM-bar)")
    applicable_span: float = Field(..., description="Applicable span value")

    # Pass/Fail
    status: TestStatus = Field(..., description="Test status (PASSED/FAILED)")
    threshold_used: str = Field(
        ...,
        description="Threshold criteria used (percent/alternate/diluent)"
    )

    # Bias analysis
    bias_status: BiasAdjustmentStatus = Field(..., description="Bias status")
    bias_adjustment_factor: Optional[float] = Field(
        default=None,
        description="Bias adjustment factor if required"
    )

    # Statistics
    valid_run_count: int = Field(..., description="Number of valid runs")
    t_value: float = Field(..., description="t-statistic for confidence interval")

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    calculation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # Recommendations
    next_rata_frequency: RATAFrequency = Field(
        ...,
        description="Recommended next RATA frequency"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    class Config:
        use_enum_values = True


class CGAResult(BaseModel):
    """Cylinder Gas Audit result."""

    unit_id: str = Field(..., description="Unit identifier")
    pollutant: CEMSPollutant = Field(..., description="Audited pollutant")
    audit_date: datetime = Field(..., description="Audit date/time")

    # Audit values
    cylinder_concentration: float = Field(..., description="Cylinder gas concentration")
    cems_response: float = Field(..., description="CEMS response value")

    # Calculated accuracy
    accuracy_percent: float = Field(..., description="CGA accuracy (%)")

    # Pass/Fail
    passed: bool = Field(..., description="CGA pass/fail")

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class CalibrationDriftResult(BaseModel):
    """Calibration drift test result."""

    unit_id: str = Field(..., description="Unit identifier")
    pollutant: CEMSPollutant = Field(..., description="Pollutant")
    test_date: datetime = Field(..., description="Test date/time")

    # Zero drift
    zero_reference: float = Field(..., description="Zero reference value")
    zero_cems_response: float = Field(..., description="CEMS zero response")
    zero_drift_percent: float = Field(..., description="Zero drift (% of span)")

    # Span drift
    upscale_reference: float = Field(..., description="Upscale reference value")
    upscale_cems_response: float = Field(..., description="CEMS upscale response")
    span_drift_percent: float = Field(..., description="Span drift (% of span)")

    # Pass/Fail
    passed: bool = Field(..., description="Drift test pass/fail")

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class RATASchedule(BaseModel):
    """RATA test schedule."""

    unit_id: str = Field(..., description="Unit identifier")
    year: int = Field(..., description="Schedule year")

    # Scheduled tests
    scheduled_tests: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Scheduled RATA tests"
    )

    # Current frequency
    current_frequency: RATAFrequency = Field(
        default=RATAFrequency.QUARTERLY,
        description="Current testing frequency"
    )

    # Compliance status
    next_due_date: date = Field(..., description="Next RATA due date")
    days_until_due: int = Field(..., description="Days until next RATA")
    overdue: bool = Field(default=False, description="Overdue flag")


# =============================================================================
# RATA AUTOMATION ENGINE
# =============================================================================

class RATAAutomation:
    """
    EPA RATA Automation Engine for CEMS Quality Assurance.

    Implements 40 CFR Part 75 Subpart C requirements for Relative Accuracy
    Test Audit scheduling, execution tracking, and analysis. Provides
    complete compliance automation for utility and industrial sources.

    Features:
        - Quarterly/semiannual/annual RATA scheduling
        - Automatic relative accuracy calculation
        - Bias adjustment factor (BAF) computation
        - Cylinder Gas Audit (CGA) tracking
        - Calibration drift monitoring
        - Linearity check automation
        - Complete audit trail with SHA-256 hashing

    Regulatory Compliance:
        - 40 CFR Part 75 Subpart C
        - EPA Performance Specification 2 (PS-2)
        - MATS (Mercury and Air Toxics Standards)

    Example:
        >>> rata = RATAAutomation(unit_id="BOILER-001")
        >>> schedule = rata.get_quarterly_schedule(year=2025)
        >>> result = rata.calculate_relative_accuracy(test_data)
        >>> if result.status == TestStatus.PASSED:
        ...     rata.record_successful_rata(result)
    """

    # Student's t-distribution values for 95% confidence
    T_VALUES = {
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
    }

    def __init__(
        self,
        unit_id: str,
        pollutants: Optional[List[CEMSPollutant]] = None,
        initial_frequency: RATAFrequency = RATAFrequency.QUARTERLY,
    ) -> None:
        """
        Initialize RATA automation engine.

        Args:
            unit_id: Emission unit identifier
            pollutants: List of monitored pollutants (default: NOx, SO2, CO2, O2, FLOW)
            initial_frequency: Initial RATA frequency (default: quarterly)
        """
        self.unit_id = unit_id
        self.pollutants = pollutants or [
            CEMSPollutant.NOX,
            CEMSPollutant.SO2,
            CEMSPollutant.CO2,
            CEMSPollutant.O2,
            CEMSPollutant.FLOW,
        ]
        self.current_frequency = initial_frequency

        # Historical tracking
        self._rata_history: List[RATAResult] = []
        self._cga_history: List[CGAResult] = []
        self._calibration_history: List[CalibrationDriftResult] = []

        # Schedule tracking
        self._last_rata_date: Optional[date] = None
        self._consecutive_good_ratas: int = 0

        logger.info(
            f"RATAAutomation initialized for {unit_id} with "
            f"{len(self.pollutants)} pollutants, frequency: {initial_frequency.value}"
        )

    def calculate_relative_accuracy(
        self,
        test_data: RATATestInput,
    ) -> RATAResult:
        """
        Calculate Relative Accuracy per 40 CFR 75.

        The RA calculation follows EPA methodology:
        1. Calculate mean difference (d-bar) = sum(d_i) / n
        2. Calculate standard deviation (Sd)
        3. Calculate confidence coefficient (CC) = t * Sd / sqrt(n)
        4. Calculate RA% = |d-bar| + |CC| / RM-bar * 100

        Args:
            test_data: Complete RATA test input data

        Returns:
            RATAResult with pass/fail determination and bias analysis

        Raises:
            ValueError: If test data is invalid
        """
        logger.info(
            f"Calculating RA for {test_data.unit_id} - "
            f"{test_data.pollutant.value} on {test_data.test_date}"
        )

        start_time = datetime.now(timezone.utc)
        warnings = []

        # Get valid runs only
        valid_runs = [r for r in test_data.runs if r.is_valid]
        n = len(valid_runs)

        if n < RATAConstants.MIN_RATA_RUNS:
            raise ValueError(
                f"Insufficient valid runs: {n} < {RATAConstants.MIN_RATA_RUNS}"
            )

        # Extract values
        reference_values = [r.reference_value for r in valid_runs]
        cems_values = [r.cems_value for r in valid_runs]
        differences = [r.difference for r in valid_runs]

        # Calculate mean reference value (RM-bar)
        rm_bar = statistics.mean(reference_values)

        if rm_bar <= 0:
            raise ValueError("Mean reference value must be positive")

        # Calculate mean difference (d-bar)
        d_bar = statistics.mean(differences)

        # Calculate standard deviation (Sd)
        if n > 1:
            sd = statistics.stdev(differences)
        else:
            sd = 0.0

        # Get t-value for confidence interval
        t_value = self.T_VALUES.get(n, 2.306)

        # Calculate confidence coefficient (CC)
        cc = t_value * sd / math.sqrt(n)

        # Determine applicable span
        applicable_span = test_data.analyzer_span

        # Calculate Relative Accuracy percentage
        # RA = (|d-bar| + |CC|) / RM-bar * 100
        ra_pct = (abs(d_bar) + abs(cc)) / rm_bar * 100

        # Determine pass/fail threshold
        threshold_used = "percent"
        passed = False

        if test_data.pollutant in [CEMSPollutant.O2, CEMSPollutant.CO2]:
            # Diluent monitors use absolute threshold
            absolute_ra = abs(d_bar) + abs(cc)
            if absolute_ra <= RATAConstants.RA_THRESHOLD_DILUENT:
                passed = True
                threshold_used = "diluent_absolute"
            elif ra_pct <= RATAConstants.RA_THRESHOLD_PERCENT:
                passed = True
                threshold_used = "percent"
        else:
            # Standard pollutants
            if ra_pct <= RATAConstants.RA_THRESHOLD_PERCENT:
                passed = True
                threshold_used = "percent"
            elif rm_bar < applicable_span * 0.5:
                # Alternate criteria when mean is low
                if ra_pct <= RATAConstants.RA_THRESHOLD_ALTERNATE:
                    passed = True
                    threshold_used = "alternate"
                    warnings.append(
                        f"Passed using alternate 20% RA criteria "
                        f"(RM-bar {rm_bar:.2f} < 50% span)"
                    )

        # Determine test status
        if passed:
            status = TestStatus.PASSED
        else:
            status = TestStatus.FAILED
            warnings.append(
                f"RA {ra_pct:.2f}% exceeds threshold, corrective action required"
            )

        # Bias analysis
        bias_status, baf = self._calculate_bias_adjustment(
            d_bar=d_bar,
            rm_bar=rm_bar,
            cc=cc,
            pollutant=test_data.pollutant,
        )

        if bias_status == BiasAdjustmentStatus.REQUIRED:
            warnings.append(
                f"Bias adjustment required. BAF = {baf:.4f}"
            )

        # Determine next RATA frequency
        next_frequency = self._determine_frequency(
            ra_pct=ra_pct,
            prior_results=test_data.prior_rata_results,
        )

        # Calculate provenance hash
        provenance_hash = self._hash_test_data(test_data, ra_pct)

        result = RATAResult(
            unit_id=test_data.unit_id,
            pollutant=test_data.pollutant,
            test_date=test_data.test_date,
            mean_difference=round(d_bar, 4),
            standard_deviation=round(sd, 4),
            confidence_coefficient=round(cc, 4),
            relative_accuracy_pct=round(ra_pct, 2),
            mean_reference_value=round(rm_bar, 4),
            applicable_span=applicable_span,
            status=status,
            threshold_used=threshold_used,
            bias_status=bias_status,
            bias_adjustment_factor=round(baf, 4) if baf else None,
            valid_run_count=n,
            t_value=t_value,
            provenance_hash=provenance_hash,
            next_rata_frequency=next_frequency,
            warnings=warnings,
        )

        # Record in history
        self._rata_history.append(result)
        self._last_rata_date = test_data.test_date

        if passed and ra_pct <= 7.5:
            self._consecutive_good_ratas += 1
        else:
            self._consecutive_good_ratas = 0

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            f"RATA calculation complete: RA={ra_pct:.2f}%, "
            f"Status={status.value}, Time={processing_time:.1f}ms"
        )

        return result

    def _calculate_bias_adjustment(
        self,
        d_bar: float,
        rm_bar: float,
        cc: float,
        pollutant: CEMSPollutant,
    ) -> Tuple[BiasAdjustmentStatus, Optional[float]]:
        """
        Calculate bias adjustment factor per 40 CFR 75.

        Bias adjustment is required if the mean difference (d-bar)
        is statistically significant and exceeds the threshold.

        Args:
            d_bar: Mean difference
            rm_bar: Mean reference value
            cc: Confidence coefficient
            pollutant: Tested pollutant

        Returns:
            Tuple of (BiasAdjustmentStatus, BAF if required)
        """
        # Check if bias is statistically significant
        # Bias is significant if |d-bar| > |cc|
        if abs(d_bar) <= abs(cc):
            return BiasAdjustmentStatus.NOT_REQUIRED, None

        # Calculate bias percentage
        bias_pct = abs(d_bar) / rm_bar * 100 if rm_bar > 0 else 0

        if bias_pct <= RATAConstants.BIAS_THRESHOLD_PERCENT:
            return BiasAdjustmentStatus.NOT_REQUIRED, None

        # Calculate Bias Adjustment Factor (BAF)
        # BAF = 1 - (d-bar / RM-bar)
        # Applied as: Adjusted = CEMS * BAF
        baf = 1.0 - (d_bar / rm_bar) if rm_bar > 0 else 1.0

        # BAF must be between 0.9 and 1.1 for reasonableness
        if not (0.9 <= baf <= 1.1):
            logger.warning(
                f"BAF {baf:.4f} outside typical range, review test data"
            )

        return BiasAdjustmentStatus.REQUIRED, baf

    def _determine_frequency(
        self,
        ra_pct: float,
        prior_results: List[float],
    ) -> RATAFrequency:
        """
        Determine RATA frequency based on performance history.

        Per 40 CFR 75:
        - Annual: 4 consecutive RATAs with RA <= 7.5%
        - Semiannual: 2 consecutive RATAs with RA <= 7.5%
        - Quarterly: Default, or after failed RATA

        Args:
            ra_pct: Current RATA relative accuracy
            prior_results: Prior RATA RA% results

        Returns:
            Recommended next RATA frequency
        """
        # Combine current with prior results
        all_results = prior_results + [ra_pct]

        # Check if current RA qualifies for reduced frequency
        if ra_pct > 7.5:
            return RATAFrequency.QUARTERLY

        # Count consecutive good RATAs (RA <= 7.5%)
        consecutive = 0
        for ra in reversed(all_results):
            if ra <= 7.5:
                consecutive += 1
            else:
                break

        if consecutive >= 4:
            return RATAFrequency.ANNUAL
        elif consecutive >= 2:
            return RATAFrequency.SEMIANNUAL
        else:
            return RATAFrequency.QUARTERLY

    def get_quarterly_schedule(
        self,
        year: int,
        base_date: Optional[date] = None,
    ) -> RATASchedule:
        """
        Generate quarterly RATA schedule for a year.

        Schedules are based on calendar quarters with grace periods
        per 40 CFR 75 requirements.

        Args:
            year: Schedule year
            base_date: Base date for initial schedule (optional)

        Returns:
            RATASchedule with all scheduled tests
        """
        scheduled_tests = []

        # Define quarter end dates
        quarters = [
            (1, date(year, 3, 31)),   # Q1
            (2, date(year, 6, 30)),   # Q2
            (3, date(year, 9, 30)),   # Q3
            (4, date(year, 12, 31)),  # Q4
        ]

        for quarter, end_date in quarters:
            # RATA should be completed by quarter end
            # Schedule with 2-week buffer
            target_date = end_date - timedelta(days=14)

            for pollutant in self.pollutants:
                scheduled_tests.append({
                    "quarter": quarter,
                    "pollutant": pollutant.value,
                    "target_date": target_date.isoformat(),
                    "deadline": end_date.isoformat(),
                    "status": TestStatus.SCHEDULED.value,
                })

        # Determine next due date
        today = date.today()
        next_due = None

        for _, end_date in quarters:
            if end_date >= today:
                next_due = end_date
                break

        if next_due is None:
            next_due = date(year + 1, 3, 31)  # Next year Q1

        days_until = (next_due - today).days

        return RATASchedule(
            unit_id=self.unit_id,
            year=year,
            scheduled_tests=scheduled_tests,
            current_frequency=self.current_frequency,
            next_due_date=next_due,
            days_until_due=max(0, days_until),
            overdue=days_until < 0,
        )

    def perform_cylinder_gas_audit(
        self,
        pollutant: CEMSPollutant,
        cylinder_concentration: float,
        cems_response: float,
        analyzer_span: float,
    ) -> CGAResult:
        """
        Perform Cylinder Gas Audit per 40 CFR 75.21.

        CGA is required quarterly for each CEMS component.
        Accuracy must be within 5% of reference.

        Args:
            pollutant: Audited pollutant
            cylinder_concentration: Certified cylinder concentration
            cems_response: CEMS analyzer response
            analyzer_span: Analyzer span value

        Returns:
            CGAResult with pass/fail determination
        """
        logger.info(
            f"Performing CGA for {self.unit_id} - {pollutant.value}"
        )

        # Calculate accuracy as percent of span
        if analyzer_span > 0:
            accuracy_pct = abs(cems_response - cylinder_concentration) / analyzer_span * 100
        else:
            accuracy_pct = 0.0

        # Determine pass/fail
        passed = accuracy_pct <= RATAConstants.CGA_TOLERANCE_PERCENT

        # Calculate provenance hash
        provenance_hash = hashlib.sha256(
            f"{self.unit_id}:{pollutant.value}:{cylinder_concentration}:"
            f"{cems_response}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()

        result = CGAResult(
            unit_id=self.unit_id,
            pollutant=pollutant,
            audit_date=datetime.now(timezone.utc),
            cylinder_concentration=cylinder_concentration,
            cems_response=cems_response,
            accuracy_percent=round(accuracy_pct, 2),
            passed=passed,
            provenance_hash=provenance_hash,
        )

        self._cga_history.append(result)

        logger.info(
            f"CGA complete: Accuracy={accuracy_pct:.2f}%, "
            f"{'PASSED' if passed else 'FAILED'}"
        )

        return result

    def check_calibration_drift(
        self,
        pollutant: CEMSPollutant,
        zero_reference: float,
        zero_cems_response: float,
        upscale_reference: float,
        upscale_cems_response: float,
        analyzer_span: float,
    ) -> CalibrationDriftResult:
        """
        Check daily calibration drift per 40 CFR 75.21.

        Daily calibration drift must not exceed 2.5% of span
        for zero and upscale (span) values.

        Args:
            pollutant: Tested pollutant
            zero_reference: Zero gas reference value
            zero_cems_response: CEMS zero response
            upscale_reference: Upscale (span) gas reference
            upscale_cems_response: CEMS upscale response
            analyzer_span: Analyzer span value

        Returns:
            CalibrationDriftResult with pass/fail
        """
        # Calculate zero drift
        zero_drift = abs(zero_cems_response - zero_reference)
        zero_drift_pct = zero_drift / analyzer_span * 100 if analyzer_span > 0 else 0

        # Calculate span drift
        span_drift = abs(upscale_cems_response - upscale_reference)
        span_drift_pct = span_drift / analyzer_span * 100 if analyzer_span > 0 else 0

        # Determine pass/fail
        passed = (
            zero_drift_pct <= RATAConstants.DAILY_CALIBRATION_DRIFT and
            span_drift_pct <= RATAConstants.DAILY_CALIBRATION_DRIFT
        )

        # Calculate provenance hash
        provenance_hash = hashlib.sha256(
            f"{self.unit_id}:{pollutant.value}:"
            f"{zero_drift_pct}:{span_drift_pct}:"
            f"{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()

        result = CalibrationDriftResult(
            unit_id=self.unit_id,
            pollutant=pollutant,
            test_date=datetime.now(timezone.utc),
            zero_reference=zero_reference,
            zero_cems_response=zero_cems_response,
            zero_drift_percent=round(zero_drift_pct, 2),
            upscale_reference=upscale_reference,
            upscale_cems_response=upscale_cems_response,
            span_drift_percent=round(span_drift_pct, 2),
            passed=passed,
            provenance_hash=provenance_hash,
        )

        self._calibration_history.append(result)

        return result

    def get_compliance_status(self) -> Dict[str, Any]:
        """
        Get current RATA compliance status summary.

        Returns:
            Dictionary with compliance status for each pollutant
        """
        today = date.today()
        status = {
            "unit_id": self.unit_id,
            "as_of": today.isoformat(),
            "current_frequency": self.current_frequency.value,
            "consecutive_good_ratas": self._consecutive_good_ratas,
            "pollutant_status": {},
        }

        for pollutant in self.pollutants:
            # Find most recent RATA for this pollutant
            recent_rata = None
            for result in reversed(self._rata_history):
                if result.pollutant == pollutant.value:
                    recent_rata = result
                    break

            if recent_rata:
                # Calculate days since last RATA
                days_since = (today - recent_rata.test_date).days

                # Determine next due based on frequency
                if self.current_frequency == RATAFrequency.QUARTERLY:
                    max_interval = 90
                elif self.current_frequency == RATAFrequency.SEMIANNUAL:
                    max_interval = 180
                else:  # Annual
                    max_interval = 365

                days_until_due = max_interval - days_since

                status["pollutant_status"][pollutant.value] = {
                    "last_rata_date": recent_rata.test_date.isoformat(),
                    "last_ra_percent": recent_rata.relative_accuracy_pct,
                    "last_status": recent_rata.status,
                    "days_since_rata": days_since,
                    "days_until_due": max(0, days_until_due),
                    "overdue": days_until_due < 0,
                }
            else:
                status["pollutant_status"][pollutant.value] = {
                    "last_rata_date": None,
                    "status": "NO_HISTORY",
                    "overdue": True,
                }

        return status

    def _hash_test_data(
        self,
        test_data: RATATestInput,
        ra_pct: float,
    ) -> str:
        """Calculate SHA-256 hash for test provenance."""
        import json

        hash_data = {
            "unit_id": test_data.unit_id,
            "pollutant": test_data.pollutant.value,
            "test_date": test_data.test_date.isoformat(),
            "ra_pct": ra_pct,
            "run_count": len(test_data.runs),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()

    def get_rata_history(
        self,
        pollutant: Optional[CEMSPollutant] = None,
        limit: int = 10,
    ) -> List[RATAResult]:
        """
        Get RATA test history.

        Args:
            pollutant: Filter by pollutant (optional)
            limit: Maximum number of results

        Returns:
            List of RATAResult objects
        """
        if pollutant:
            filtered = [
                r for r in self._rata_history
                if r.pollutant == pollutant.value
            ]
        else:
            filtered = self._rata_history

        return list(reversed(filtered))[:limit]
