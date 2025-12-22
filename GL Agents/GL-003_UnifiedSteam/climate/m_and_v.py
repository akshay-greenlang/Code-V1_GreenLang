"""
Measurement & Verification (M&V) Methodology for GL-003 UNIFIEDSTEAM

Implements IPMVP-aligned M&V methodology for quantifying steam system
energy savings with baseline normalization and uncertainty propagation.

Reference Standards:
    - IPMVP: International Performance Measurement and Verification Protocol
    - ASHRAE Guideline 14: Measurement of Energy, Demand, and Water Savings
    - ISO 50015: Energy management systems - M&V of energy performance

Author: GL-003 Climate Intelligence Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import logging
import statistics

logger = logging.getLogger(__name__)


class MVOption(Enum):
    """IPMVP M&V Options."""
    OPTION_A = "option_a"  # Partially Measured Retrofit Isolation
    OPTION_B = "option_b"  # Retrofit Isolation (full measurement)
    OPTION_C = "option_c"  # Whole Facility (utility analysis)
    OPTION_D = "option_d"  # Calibrated Simulation


class NormalizationMethod(Enum):
    """Methods for baseline normalization."""
    PRODUCTION_RATE = "production_rate"
    DEGREE_DAYS = "degree_days"
    COOLING_WATER_TEMP = "cooling_water_temp"
    AMBIENT_TEMP = "ambient_temp"
    OPERATING_HOURS = "operating_hours"
    MULTI_VARIABLE = "multi_variable"


class SavingsType(Enum):
    """Types of savings calculations."""
    ENERGY_GJ = "energy_gj"
    STEAM_TONNES = "steam_tonnes"
    FUEL_GJ = "fuel_gj"
    WATER_M3 = "water_m3"
    COST_USD = "cost_usd"
    CO2E_KG = "co2e_kg"


@dataclass
class BaselinePeriod:
    """
    Baseline period definition for M&V.

    Attributes:
        start_date: Start of baseline period
        end_date: End of baseline period
        production_avg: Average production rate
        energy_avg: Average energy consumption
        normalization_factors: Applied normalization factors
    """
    start_date: datetime
    end_date: datetime
    production_avg: Decimal
    energy_avg: Decimal
    steam_flow_avg: Decimal
    ambient_temp_avg: Optional[Decimal] = None
    cooling_water_temp_avg: Optional[Decimal] = None
    operating_hours: Optional[Decimal] = None
    normalization_factors: Dict[str, Decimal] = field(default_factory=dict)
    data_quality_score: Decimal = Decimal("1.0")
    notes: str = ""

    def duration_days(self) -> int:
        """Return baseline period duration in days."""
        return (self.end_date - self.start_date).days


@dataclass
class ReportingPeriod:
    """
    Reporting period definition for M&V.

    Attributes:
        start_date: Start of reporting period
        end_date: End of reporting period
        production_actual: Actual production rate
        energy_actual: Actual energy consumption
    """
    start_date: datetime
    end_date: datetime
    production_actual: Decimal
    energy_actual: Decimal
    steam_flow_actual: Decimal
    ambient_temp_actual: Optional[Decimal] = None
    cooling_water_temp_actual: Optional[Decimal] = None
    operating_hours_actual: Optional[Decimal] = None
    data_quality_score: Decimal = Decimal("1.0")


@dataclass
class NormalizationResult:
    """Result of baseline normalization."""
    baseline_adjusted: Decimal
    adjustment_factors: Dict[str, Decimal]
    method: NormalizationMethod
    r_squared: Optional[Decimal] = None
    cv_rmse: Optional[Decimal] = None
    notes: str = ""


@dataclass
class SavingsResult:
    """
    Result of savings calculation.

    Includes central estimate, uncertainty bounds, and provenance.
    """
    savings_type: SavingsType
    baseline_adjusted: Decimal
    actual: Decimal
    savings: Decimal
    savings_pct: Decimal
    uncertainty_lower: Decimal
    uncertainty_upper: Decimal
    confidence_level: Decimal  # e.g., 0.95 for 95%
    methodology: MVOption
    normalization_method: NormalizationMethod
    calculation_hash: str
    notes: str = ""


@dataclass
class MVReport:
    """
    Complete M&V report for a reporting period.

    Includes savings, methodology, data quality, and audit trail.
    """
    report_id: str
    created_at: datetime
    baseline: BaselinePeriod
    reporting: ReportingPeriod
    methodology: MVOption
    savings_results: Dict[SavingsType, SavingsResult]
    data_quality_assessment: Dict[str, Any]
    audit_trail: List[Dict[str, Any]]
    approval_status: str = "draft"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


class NormalizationEngine:
    """
    Engine for baseline normalization adjustments.

    Supports production-based, weather-based, and multi-variable
    normalization per IPMVP guidelines.
    """

    def __init__(self):
        """Initialize normalization engine."""
        self._model_coefficients: Dict[str, Decimal] = {}

    def normalize_for_production(
        self,
        baseline: BaselinePeriod,
        reporting: ReportingPeriod,
    ) -> NormalizationResult:
        """
        Normalize baseline for production rate changes.

        Uses simple ratio adjustment for production-based normalization.

        Args:
            baseline: Baseline period data
            reporting: Reporting period data

        Returns:
            NormalizationResult with adjusted baseline
        """
        # Calculate production ratio
        if baseline.production_avg == 0:
            raise ValueError("Baseline production average cannot be zero")

        production_ratio = reporting.production_actual / baseline.production_avg

        # Adjust baseline energy for production
        baseline_adjusted = baseline.energy_avg * production_ratio

        return NormalizationResult(
            baseline_adjusted=baseline_adjusted,
            adjustment_factors={"production_ratio": production_ratio},
            method=NormalizationMethod.PRODUCTION_RATE,
            notes="Linear production-based adjustment",
        )

    def normalize_multi_variable(
        self,
        baseline: BaselinePeriod,
        reporting: ReportingPeriod,
        variables: List[NormalizationMethod],
    ) -> NormalizationResult:
        """
        Multi-variable normalization using multiple factors.

        Args:
            baseline: Baseline period data
            reporting: Reporting period data
            variables: List of normalization variables to apply

        Returns:
            NormalizationResult with adjusted baseline
        """
        adjustment_factors: Dict[str, Decimal] = {}
        total_adjustment = Decimal("1.0")

        for var in variables:
            if var == NormalizationMethod.PRODUCTION_RATE:
                if baseline.production_avg > 0:
                    ratio = reporting.production_actual / baseline.production_avg
                    adjustment_factors["production"] = ratio
                    total_adjustment *= ratio

            elif var == NormalizationMethod.AMBIENT_TEMP:
                if baseline.ambient_temp_avg and reporting.ambient_temp_actual:
                    # Temperature adjustment (simplified)
                    # Assumes higher ambient temp = lower steam demand
                    temp_effect = (
                        baseline.ambient_temp_avg - reporting.ambient_temp_actual
                    ) * Decimal("0.005")  # 0.5% per degree
                    temp_factor = Decimal("1.0") + temp_effect
                    adjustment_factors["ambient_temp"] = temp_factor
                    total_adjustment *= temp_factor

            elif var == NormalizationMethod.COOLING_WATER_TEMP:
                if baseline.cooling_water_temp_avg and reporting.cooling_water_temp_actual:
                    # Cooling water adjustment for condensers
                    cw_effect = (
                        reporting.cooling_water_temp_actual -
                        baseline.cooling_water_temp_avg
                    ) * Decimal("0.003")  # 0.3% per degree
                    cw_factor = Decimal("1.0") + cw_effect
                    adjustment_factors["cooling_water"] = cw_factor
                    total_adjustment *= cw_factor

            elif var == NormalizationMethod.OPERATING_HOURS:
                if baseline.operating_hours and reporting.operating_hours_actual:
                    hours_ratio = (
                        reporting.operating_hours_actual / baseline.operating_hours
                    )
                    adjustment_factors["operating_hours"] = hours_ratio
                    total_adjustment *= hours_ratio

        baseline_adjusted = baseline.energy_avg * total_adjustment

        return NormalizationResult(
            baseline_adjusted=baseline_adjusted,
            adjustment_factors=adjustment_factors,
            method=NormalizationMethod.MULTI_VARIABLE,
            notes=f"Multi-variable adjustment with {len(variables)} factors",
        )


class SavingsCalculator:
    """
    Calculator for energy and emissions savings.

    Implements IPMVP savings calculations with uncertainty propagation.
    """

    def __init__(
        self,
        methodology: MVOption = MVOption.OPTION_B,
        confidence_level: Decimal = Decimal("0.95"),
    ):
        """
        Initialize savings calculator.

        Args:
            methodology: IPMVP M&V option
            confidence_level: Confidence level for uncertainty bounds
        """
        self.methodology = methodology
        self.confidence_level = confidence_level
        self._audit_log: List[Dict[str, Any]] = []

    def calculate_savings(
        self,
        baseline_adjusted: Decimal,
        actual: Decimal,
        savings_type: SavingsType,
        baseline_uncertainty_pct: Decimal = Decimal("5.0"),
        actual_uncertainty_pct: Decimal = Decimal("3.0"),
        normalization: Optional[NormalizationResult] = None,
    ) -> SavingsResult:
        """
        Calculate savings with uncertainty bounds.

        Savings = Baseline (adjusted) - Actual

        Args:
            baseline_adjusted: Normalized baseline value
            actual: Actual measured value
            savings_type: Type of savings being calculated
            baseline_uncertainty_pct: Baseline uncertainty (%)
            actual_uncertainty_pct: Actual measurement uncertainty (%)
            normalization: Optional normalization result

        Returns:
            SavingsResult with central estimate and bounds
        """
        # Calculate savings
        savings = baseline_adjusted - actual

        # Calculate savings percentage
        if baseline_adjusted != 0:
            savings_pct = (savings / baseline_adjusted) * Decimal("100")
        else:
            savings_pct = Decimal("0")

        # Propagate uncertainty (root sum of squares)
        baseline_unc = baseline_adjusted * (baseline_uncertainty_pct / Decimal("100"))
        actual_unc = actual * (actual_uncertainty_pct / Decimal("100"))

        combined_unc = (baseline_unc ** 2 + actual_unc ** 2).sqrt()

        # Apply confidence multiplier (1.96 for 95%)
        if self.confidence_level == Decimal("0.95"):
            z_score = Decimal("1.96")
        elif self.confidence_level == Decimal("0.90"):
            z_score = Decimal("1.645")
        else:
            z_score = Decimal("2.576")  # 99%

        uncertainty_lower = savings - combined_unc * z_score
        uncertainty_upper = savings + combined_unc * z_score

        # Compute hash for audit trail
        calc_hash = self._compute_hash(
            str(baseline_adjusted),
            str(actual),
            savings_type.value,
            self.methodology.value,
        )

        result = SavingsResult(
            savings_type=savings_type,
            baseline_adjusted=baseline_adjusted,
            actual=actual,
            savings=savings.quantize(Decimal("0.01"), ROUND_HALF_UP),
            savings_pct=savings_pct.quantize(Decimal("0.01"), ROUND_HALF_UP),
            uncertainty_lower=uncertainty_lower.quantize(Decimal("0.01"), ROUND_HALF_UP),
            uncertainty_upper=uncertainty_upper.quantize(Decimal("0.01"), ROUND_HALF_UP),
            confidence_level=self.confidence_level,
            methodology=self.methodology,
            normalization_method=(
                normalization.method if normalization
                else NormalizationMethod.PRODUCTION_RATE
            ),
            calculation_hash=calc_hash,
        )

        self._log_calculation(result)
        return result

    def calculate_multiple_savings(
        self,
        baseline: BaselinePeriod,
        reporting: ReportingPeriod,
        normalization_engine: NormalizationEngine,
        normalization_vars: List[NormalizationMethod],
    ) -> Dict[SavingsType, SavingsResult]:
        """
        Calculate multiple types of savings.

        Args:
            baseline: Baseline period data
            reporting: Reporting period data
            normalization_engine: Engine for normalization
            normalization_vars: Variables to use for normalization

        Returns:
            Dictionary of savings results by type
        """
        results: Dict[SavingsType, SavingsResult] = {}

        # Normalize baseline
        norm_result = normalization_engine.normalize_multi_variable(
            baseline, reporting, normalization_vars
        )

        # Energy savings
        results[SavingsType.ENERGY_GJ] = self.calculate_savings(
            baseline_adjusted=norm_result.baseline_adjusted,
            actual=reporting.energy_actual,
            savings_type=SavingsType.ENERGY_GJ,
            normalization=norm_result,
        )

        # Steam savings (using steam flow data)
        steam_norm = normalization_engine.normalize_for_production(baseline, reporting)
        steam_baseline_adj = baseline.steam_flow_avg * (
            steam_norm.adjustment_factors.get("production_ratio", Decimal("1.0"))
        )
        results[SavingsType.STEAM_TONNES] = self.calculate_savings(
            baseline_adjusted=steam_baseline_adj,
            actual=reporting.steam_flow_actual,
            savings_type=SavingsType.STEAM_TONNES,
            normalization=steam_norm,
        )

        return results

    def _compute_hash(self, *args) -> str:
        """Compute deterministic hash for audit."""
        data = "|".join(args)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _log_calculation(self, result: SavingsResult):
        """Log calculation to audit trail."""
        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "savings_type": result.savings_type.value,
            "savings": str(result.savings),
            "hash": result.calculation_hash,
        })

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return audit log entries."""
        return self._audit_log.copy()


class BaselineManager:
    """
    Manager for baseline periods and adjustments.

    Handles baseline creation, validation, and ongoing adjustments
    per M&V methodology requirements.
    """

    def __init__(self, min_baseline_days: int = 90):
        """
        Initialize baseline manager.

        Args:
            min_baseline_days: Minimum baseline period length
        """
        self.min_baseline_days = min_baseline_days
        self._baselines: Dict[str, BaselinePeriod] = {}
        self._audit_log: List[Dict[str, Any]] = []

    def create_baseline(
        self,
        baseline_id: str,
        start_date: datetime,
        end_date: datetime,
        production_data: List[Decimal],
        energy_data: List[Decimal],
        steam_flow_data: List[Decimal],
        ambient_temp_data: Optional[List[Decimal]] = None,
        cooling_water_data: Optional[List[Decimal]] = None,
        notes: str = "",
    ) -> BaselinePeriod:
        """
        Create a new baseline period from historical data.

        Args:
            baseline_id: Unique identifier for baseline
            start_date: Start of baseline period
            end_date: End of baseline period
            production_data: List of production values
            energy_data: List of energy consumption values
            steam_flow_data: List of steam flow values
            ambient_temp_data: Optional ambient temperature data
            cooling_water_data: Optional cooling water temperature data
            notes: Additional notes

        Returns:
            Created BaselinePeriod

        Raises:
            ValueError: If baseline period is too short or data is invalid
        """
        # Validate duration
        duration = (end_date - start_date).days
        if duration < self.min_baseline_days:
            raise ValueError(
                f"Baseline period ({duration} days) is shorter than "
                f"minimum ({self.min_baseline_days} days)"
            )

        # Validate data
        if not production_data or not energy_data:
            raise ValueError("Production and energy data are required")

        if len(production_data) != len(energy_data):
            raise ValueError("Production and energy data must have same length")

        # Calculate averages
        production_avg = Decimal(str(statistics.mean([float(x) for x in production_data])))
        energy_avg = Decimal(str(statistics.mean([float(x) for x in energy_data])))
        steam_flow_avg = Decimal(str(statistics.mean([float(x) for x in steam_flow_data])))

        ambient_temp_avg = None
        if ambient_temp_data:
            ambient_temp_avg = Decimal(
                str(statistics.mean([float(x) for x in ambient_temp_data]))
            )

        cooling_water_avg = None
        if cooling_water_data:
            cooling_water_avg = Decimal(
                str(statistics.mean([float(x) for x in cooling_water_data]))
            )

        # Calculate data quality score
        # Higher score for more data points and lower variance
        production_cv = Decimal(
            str(statistics.stdev([float(x) for x in production_data]) /
                statistics.mean([float(x) for x in production_data]))
        ) if len(production_data) > 1 else Decimal("0")

        data_quality = max(
            Decimal("0.5"),
            Decimal("1.0") - production_cv * Decimal("0.5")
        )

        baseline = BaselinePeriod(
            start_date=start_date,
            end_date=end_date,
            production_avg=production_avg,
            energy_avg=energy_avg,
            steam_flow_avg=steam_flow_avg,
            ambient_temp_avg=ambient_temp_avg,
            cooling_water_temp_avg=cooling_water_avg,
            data_quality_score=data_quality,
            notes=notes,
        )

        self._baselines[baseline_id] = baseline

        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "create_baseline",
            "baseline_id": baseline_id,
            "duration_days": duration,
            "data_points": len(production_data),
        })

        logger.info(f"Baseline '{baseline_id}' created with {duration} days of data")
        return baseline

    def get_baseline(self, baseline_id: str) -> BaselinePeriod:
        """Get baseline by ID."""
        if baseline_id not in self._baselines:
            raise KeyError(f"Baseline not found: {baseline_id}")
        return self._baselines[baseline_id]

    def list_baselines(self) -> List[str]:
        """List all baseline IDs."""
        return list(self._baselines.keys())

    def validate_baseline(
        self,
        baseline_id: str,
        criteria: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate baseline against M&V criteria.

        Args:
            baseline_id: Baseline to validate
            criteria: Optional validation criteria

        Returns:
            Validation results
        """
        baseline = self.get_baseline(baseline_id)
        criteria = criteria or {}

        results = {
            "baseline_id": baseline_id,
            "valid": True,
            "issues": [],
            "warnings": [],
        }

        # Check duration
        min_days = criteria.get("min_days", self.min_baseline_days)
        if baseline.duration_days() < min_days:
            results["valid"] = False
            results["issues"].append(
                f"Baseline duration ({baseline.duration_days()} days) "
                f"is less than minimum ({min_days} days)"
            )

        # Check data quality
        min_quality = criteria.get("min_quality", Decimal("0.7"))
        if baseline.data_quality_score < min_quality:
            results["warnings"].append(
                f"Data quality score ({baseline.data_quality_score}) "
                f"is below recommended minimum ({min_quality})"
            )

        return results


class MVMethodology:
    """
    Complete M&V methodology implementation.

    Coordinates baseline management, normalization, savings calculation,
    and reporting per IPMVP guidelines.
    """

    def __init__(
        self,
        option: MVOption = MVOption.OPTION_B,
        confidence_level: Decimal = Decimal("0.95"),
        min_baseline_days: int = 90,
    ):
        """
        Initialize M&V methodology.

        Args:
            option: IPMVP M&V option to use
            confidence_level: Confidence level for uncertainty bounds
            min_baseline_days: Minimum baseline period length
        """
        self.option = option
        self.confidence_level = confidence_level

        self.baseline_manager = BaselineManager(min_baseline_days)
        self.normalization_engine = NormalizationEngine()
        self.savings_calculator = SavingsCalculator(option, confidence_level)

        self._reports: Dict[str, MVReport] = {}
        self._audit_log: List[Dict[str, Any]] = []

    def generate_report(
        self,
        baseline_id: str,
        reporting: ReportingPeriod,
        normalization_vars: Optional[List[NormalizationMethod]] = None,
    ) -> MVReport:
        """
        Generate complete M&V report.

        Args:
            baseline_id: ID of baseline to use
            reporting: Reporting period data
            normalization_vars: Variables for normalization

        Returns:
            Complete MVReport
        """
        import uuid

        baseline = self.baseline_manager.get_baseline(baseline_id)

        # Default to production normalization
        normalization_vars = normalization_vars or [NormalizationMethod.PRODUCTION_RATE]

        # Calculate savings
        savings_results = self.savings_calculator.calculate_multiple_savings(
            baseline, reporting, self.normalization_engine, normalization_vars
        )

        # Assess data quality
        data_quality = {
            "baseline_quality": str(baseline.data_quality_score),
            "reporting_quality": str(reporting.data_quality_score),
            "combined_quality": str(
                (baseline.data_quality_score + reporting.data_quality_score) / 2
            ),
            "normalization_vars": [v.value for v in normalization_vars],
        }

        # Create report
        report_id = f"MV-{uuid.uuid4().hex[:8].upper()}"
        report = MVReport(
            report_id=report_id,
            created_at=datetime.now(timezone.utc),
            baseline=baseline,
            reporting=reporting,
            methodology=self.option,
            savings_results=savings_results,
            data_quality_assessment=data_quality,
            audit_trail=self._build_audit_trail(baseline_id, reporting),
        )

        self._reports[report_id] = report

        logger.info(
            f"M&V report generated: {report_id}, "
            f"Energy savings: {savings_results.get(SavingsType.ENERGY_GJ, 'N/A')}"
        )

        return report

    def _build_audit_trail(
        self,
        baseline_id: str,
        reporting: ReportingPeriod,
    ) -> List[Dict[str, Any]]:
        """Build audit trail for report."""
        return [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "report_generated",
                "baseline_id": baseline_id,
                "reporting_period": f"{reporting.start_date.date()} to {reporting.end_date.date()}",
                "methodology": self.option.value,
            }
        ]

    def get_report(self, report_id: str) -> MVReport:
        """Get report by ID."""
        if report_id not in self._reports:
            raise KeyError(f"Report not found: {report_id}")
        return self._reports[report_id]

    def approve_report(
        self,
        report_id: str,
        approver: str,
    ) -> MVReport:
        """
        Approve an M&V report.

        Args:
            report_id: Report to approve
            approver: Name/ID of approver

        Returns:
            Updated report
        """
        report = self.get_report(report_id)
        report.approval_status = "approved"
        report.approved_by = approver
        report.approved_at = datetime.now(timezone.utc)

        report.audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "report_approved",
            "approved_by": approver,
        })

        return report

    def export_report(self, report_id: str) -> Dict[str, Any]:
        """Export report as dictionary for serialization."""
        report = self.get_report(report_id)

        return {
            "report_id": report.report_id,
            "created_at": report.created_at.isoformat(),
            "methodology": report.methodology.value,
            "baseline": {
                "start_date": report.baseline.start_date.isoformat(),
                "end_date": report.baseline.end_date.isoformat(),
                "production_avg": str(report.baseline.production_avg),
                "energy_avg": str(report.baseline.energy_avg),
            },
            "reporting": {
                "start_date": report.reporting.start_date.isoformat(),
                "end_date": report.reporting.end_date.isoformat(),
                "production_actual": str(report.reporting.production_actual),
                "energy_actual": str(report.reporting.energy_actual),
            },
            "savings": {
                st.value: {
                    "savings": str(sr.savings),
                    "savings_pct": str(sr.savings_pct),
                    "uncertainty_lower": str(sr.uncertainty_lower),
                    "uncertainty_upper": str(sr.uncertainty_upper),
                }
                for st, sr in report.savings_results.items()
            },
            "data_quality": report.data_quality_assessment,
            "approval_status": report.approval_status,
            "approved_by": report.approved_by,
            "approved_at": report.approved_at.isoformat() if report.approved_at else None,
            "audit_trail": report.audit_trail,
        }
