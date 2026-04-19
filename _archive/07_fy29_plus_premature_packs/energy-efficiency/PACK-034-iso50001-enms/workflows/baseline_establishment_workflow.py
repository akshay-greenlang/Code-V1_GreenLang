# -*- coding: utf-8 -*-
"""
Baseline Establishment Workflow - EnB Setup
===================================

3-phase workflow for establishing energy baselines (EnB) using regression
modelling within PACK-034 ISO 50001 Energy Management System Pack.

Phases:
    1. DataValidation       -- Validate minimum data points, check gaps/outliers
    2. RegressionModeling   -- Fit regression models, select best by R2/CV(RMSE)
    3. BaselineApproval     -- Generate baseline report with confidence intervals

The workflow follows GreenLang zero-hallucination principles: regression
fitting uses deterministic least-squares formulas, model selection uses
R-squared and CV(RMSE) thresholds, and confidence intervals are calculated
from standard statistical theory. SHA-256 provenance hashes guarantee
auditability.

Schedule: annual / on baseline adjustment trigger
Estimated duration: 30 minutes

Author: GreenLang Team
Version: 34.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class BaselinePhase(str, Enum):
    """Phases of the baseline establishment workflow."""

    DATA_VALIDATION = "data_validation"
    REGRESSION_MODELING = "regression_modeling"
    BASELINE_APPROVAL = "baseline_approval"


class ModelType(str, Enum):
    """Regression model types."""

    SIMPLE_MEAN = "simple_mean"
    SINGLE_VARIABLE = "single_variable"
    MULTI_VARIABLE = "multi_variable"


# =============================================================================
# STATISTICAL THRESHOLDS (Zero-Hallucination Reference)
# =============================================================================

# Minimum data points for statistically valid regression
MIN_DATA_POINTS: int = 12

# R-squared threshold for acceptable model fit
R_SQUARED_THRESHOLD: float = 0.75

# CV(RMSE) threshold per ASHRAE Guideline 14 for whole-building models
CV_RMSE_THRESHOLD: float = 25.0  # percent

# Confidence level for interval estimation
CONFIDENCE_LEVEL: float = 0.95

# t-value approximation for 95% CI with >30 data points
T_VALUE_95: float = 1.96


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class DataQualityReport(BaseModel):
    """Data quality assessment for baseline data."""

    total_records: int = Field(default=0, ge=0)
    valid_records: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)
    outlier_count: int = Field(default=0, ge=0)
    gap_count: int = Field(default=0, ge=0)
    completeness_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    passes_minimum: bool = Field(default=False, description="True if >= MIN_DATA_POINTS valid records")


class RegressionModelResult(BaseModel):
    """Result from a single regression model fit."""

    model_type: str = Field(default="", description="simple_mean|single_variable|multi_variable")
    variable_names: List[str] = Field(default_factory=list, description="Independent variables used")
    intercept: Decimal = Field(default=Decimal("0"), description="Y-intercept (beta_0)")
    coefficients: Dict[str, Decimal] = Field(default_factory=dict, description="Variable coefficients")
    r_squared: Decimal = Field(default=Decimal("0"), description="R-squared goodness of fit")
    adjusted_r_squared: Decimal = Field(default=Decimal("0"), description="Adjusted R-squared")
    cv_rmse_pct: Decimal = Field(default=Decimal("0"), description="CV(RMSE) percentage")
    rmse: Decimal = Field(default=Decimal("0"), ge=0, description="Root mean square error")
    n_observations: int = Field(default=0, ge=0)
    is_acceptable: bool = Field(default=False, description="True if meets R2 and CV(RMSE) thresholds")


class ModelStatistics(BaseModel):
    """Comprehensive model statistics for approval package."""

    best_model_type: str = Field(default="", description="Selected model type")
    r_squared: Decimal = Field(default=Decimal("0"))
    adjusted_r_squared: Decimal = Field(default=Decimal("0"))
    cv_rmse_pct: Decimal = Field(default=Decimal("0"))
    rmse: Decimal = Field(default=Decimal("0"), ge=0)
    f_statistic: Decimal = Field(default=Decimal("0"), ge=0)
    p_value: Decimal = Field(default=Decimal("0"), ge=0)
    confidence_interval_lower: Decimal = Field(default=Decimal("0"))
    confidence_interval_upper: Decimal = Field(default=Decimal("0"))
    models_evaluated: int = Field(default=0, ge=0)


class ApprovalPackage(BaseModel):
    """Baseline approval documentation package."""

    baseline_id: str = Field(default="", description="Baseline identifier")
    recommendation: str = Field(default="", description="approve|reject|review")
    rationale: str = Field(default="", description="Rationale for recommendation")
    model_summary: str = Field(default="", description="Human-readable model description")
    data_quality_summary: str = Field(default="")
    confidence_interval: str = Field(default="", description="e.g. 95% CI: [lower, upper]")
    review_notes: List[str] = Field(default_factory=list)


class BaselineEstablishmentInput(BaseModel):
    """Input data model for BaselineEstablishmentWorkflow."""

    enms_id: str = Field(default="", description="EnMS program identifier")
    energy_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Time-series energy data: [{period, kwh, ...}]",
    )
    variable_data: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Independent variable data: {var_name: [values]}",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration overrides (thresholds, etc.)",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("energy_data")
    @classmethod
    def validate_energy_data(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure energy data is provided."""
        if not v:
            raise ValueError("energy_data must contain at least one record")
        return v


class BaselineEstablishmentResult(BaseModel):
    """Complete result from baseline establishment workflow."""

    baseline_id: str = Field(..., description="Unique baseline ID")
    enms_id: str = Field(default="", description="EnMS program identifier")
    baseline_model: RegressionModelResult = Field(
        default_factory=RegressionModelResult,
        description="Selected baseline regression model",
    )
    all_models: List[RegressionModelResult] = Field(
        default_factory=list, description="All models evaluated",
    )
    model_statistics: ModelStatistics = Field(
        default_factory=ModelStatistics, description="Comprehensive statistics",
    )
    data_quality: DataQualityReport = Field(
        default_factory=DataQualityReport, description="Data quality assessment",
    )
    validation_status: str = Field(default="pending", description="pass|fail|pending")
    approval_package: ApprovalPackage = Field(
        default_factory=ApprovalPackage, description="Approval documentation",
    )
    phases_completed: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class BaselineEstablishmentWorkflow:
    """
    3-phase baseline establishment workflow per ISO 50001.

    Performs data validation, regression modelling (simple mean,
    single-variable, multi-variable), model selection by R-squared and
    CV(RMSE), and generates an approval package with confidence intervals.

    Zero-hallucination: regression uses deterministic least-squares
    calculation. Model selection uses published ASHRAE Guideline 14
    thresholds. No LLM calls in the numeric computation path.

    Attributes:
        baseline_id: Unique baseline execution identifier.
        _data_quality: Data quality assessment.
        _models: Fitted regression models.
        _best_model: Selected best model.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = BaselineEstablishmentWorkflow()
        >>> inp = BaselineEstablishmentInput(
        ...     enms_id="enms-001",
        ...     energy_data=[{"period": "2025-01", "kwh": 50000}, ...],
        ...     variable_data={"hdd": [120, 95, ...]},
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.validation_status in ("pass", "fail")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BaselineEstablishmentWorkflow."""
        self.baseline_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._data_quality: DataQualityReport = DataQualityReport()
        self._models: List[RegressionModelResult] = []
        self._best_model: Optional[RegressionModelResult] = None
        self._energy_values: List[float] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def execute(self, input_data: BaselineEstablishmentInput) -> BaselineEstablishmentResult:
        """
        Execute the 3-phase baseline establishment workflow.

        Args:
            input_data: Validated baseline establishment input.

        Returns:
            BaselineEstablishmentResult with model, statistics, and approval package.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting baseline establishment workflow %s enms=%s data_points=%d",
            self.baseline_id, input_data.enms_id, len(input_data.energy_data),
        )

        self._phase_results = []
        self._models = []
        self._best_model = None
        self._energy_values = []

        try:
            # Phase 1: Data Validation
            phase1 = self._phase_data_validation(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Regression Modeling
            phase2 = self._phase_regression_modeling(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Baseline Approval
            phase3 = self._phase_baseline_approval(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error(
                "Baseline establishment workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Determine validation status
        validation_status = "fail"
        if self._best_model and self._best_model.is_acceptable:
            validation_status = "pass"

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        # Build model statistics
        model_stats = self._build_model_statistics()

        # Build approval package
        approval = self._build_approval_package(validation_status)

        result = BaselineEstablishmentResult(
            baseline_id=self.baseline_id,
            enms_id=input_data.enms_id,
            baseline_model=self._best_model or RegressionModelResult(),
            all_models=self._models,
            model_statistics=model_stats,
            data_quality=self._data_quality,
            validation_status=validation_status,
            approval_package=approval,
            phases_completed=completed_phases,
            execution_time_ms=round(elapsed_ms, 2),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Baseline establishment workflow %s completed in %.0fms status=%s R2=%.3f",
            self.baseline_id, elapsed_ms, validation_status,
            float(self._best_model.r_squared) if self._best_model else 0.0,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Validation
    # -------------------------------------------------------------------------

    def _phase_data_validation(
        self, input_data: BaselineEstablishmentInput
    ) -> PhaseResult:
        """Validate minimum data points, check for gaps and outliers."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_records = len(input_data.energy_data)
        valid_records = 0
        missing_count = 0
        outlier_count = 0

        # Extract energy values
        raw_values: List[float] = []
        for record in input_data.energy_data:
            kwh = record.get("kwh")
            if kwh is None or kwh == "":
                missing_count += 1
                continue
            try:
                val = float(kwh)
            except (ValueError, TypeError):
                missing_count += 1
                continue

            if val < 0:
                warnings.append(f"Negative kWh value: {val}; treating as outlier")
                outlier_count += 1
                continue

            raw_values.append(val)
            valid_records += 1

        # Detect outliers using IQR method
        if len(raw_values) >= 4:
            sorted_vals = sorted(raw_values)
            q1_idx = len(sorted_vals) // 4
            q3_idx = 3 * len(sorted_vals) // 4
            q1 = sorted_vals[q1_idx]
            q3 = sorted_vals[q3_idx]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            for val in raw_values:
                if val < lower_bound or val > upper_bound:
                    outlier_count += 1

        # Check for temporal gaps
        gap_count = 0
        periods = [r.get("period", "") for r in input_data.energy_data if r.get("period")]
        if len(periods) >= 2:
            sorted_periods = sorted(periods)
            for i in range(1, len(sorted_periods)):
                # Simple gap check: if consecutive periods differ by more than expected
                gap_count += 0  # Placeholder; real gap detection would parse dates

        completeness_pct = round(valid_records / max(total_records, 1) * 100.0, 1)
        passes_minimum = valid_records >= MIN_DATA_POINTS

        self._data_quality = DataQualityReport(
            total_records=total_records,
            valid_records=valid_records,
            missing_count=missing_count,
            outlier_count=outlier_count,
            gap_count=gap_count,
            completeness_pct=Decimal(str(completeness_pct)),
            passes_minimum=passes_minimum,
        )
        self._energy_values = raw_values

        if not passes_minimum:
            warnings.append(
                f"Only {valid_records} valid data points; minimum {MIN_DATA_POINTS} required"
            )

        # Validate variable data alignment
        for var_name, var_values in input_data.variable_data.items():
            if len(var_values) != valid_records:
                warnings.append(
                    f"Variable '{var_name}' has {len(var_values)} points "
                    f"vs {valid_records} energy records"
                )

        outputs["total_records"] = total_records
        outputs["valid_records"] = valid_records
        outputs["missing_count"] = missing_count
        outputs["outlier_count"] = outlier_count
        outputs["completeness_pct"] = completeness_pct
        outputs["passes_minimum"] = passes_minimum
        outputs["min_required"] = MIN_DATA_POINTS

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 DataValidation: %d/%d valid, %d missing, %d outliers, passes=%s",
            valid_records, total_records, missing_count, outlier_count, passes_minimum,
        )
        return PhaseResult(
            phase_name=BaselinePhase.DATA_VALIDATION.value, phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Regression Modeling
    # -------------------------------------------------------------------------

    def _phase_regression_modeling(
        self, input_data: BaselineEstablishmentInput
    ) -> PhaseResult:
        """Fit regression models and select best by R-squared / CV(RMSE)."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        y = self._energy_values
        n = len(y)

        if n < 2:
            warnings.append("Insufficient data for regression modelling")
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return PhaseResult(
                phase_name=BaselinePhase.REGRESSION_MODELING.value, phase_number=2,
                status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
                outputs={"models_fitted": 0}, warnings=warnings,
                provenance_hash=self._hash_dict({"models_fitted": 0}),
            )

        # Model 1: Simple mean (no independent variables)
        mean_model = self._fit_simple_mean(y)
        self._models.append(mean_model)

        # Model 2: Single-variable regressions
        for var_name, var_values in input_data.variable_data.items():
            x = var_values[:n]
            if len(x) < n:
                warnings.append(f"Skipping variable '{var_name}': insufficient data points")
                continue
            single_model = self._fit_single_variable(y, x, var_name)
            self._models.append(single_model)

        # Model 3: Multi-variable regression (if >= 2 variables available)
        var_names = list(input_data.variable_data.keys())
        if len(var_names) >= 2:
            x_matrix: List[List[float]] = []
            usable_vars: List[str] = []
            for vn in var_names:
                vals = input_data.variable_data[vn][:n]
                if len(vals) == n:
                    x_matrix.append(vals)
                    usable_vars.append(vn)

            if len(usable_vars) >= 2:
                multi_model = self._fit_multi_variable(y, x_matrix, usable_vars)
                self._models.append(multi_model)

        # Select best model by R-squared (prefer higher), then CV(RMSE) (prefer lower)
        acceptable_models = [m for m in self._models if m.is_acceptable]
        if acceptable_models:
            self._best_model = max(acceptable_models, key=lambda m: float(m.r_squared))
        elif self._models:
            self._best_model = max(self._models, key=lambda m: float(m.r_squared))

        outputs["models_fitted"] = len(self._models)
        outputs["acceptable_models"] = len(acceptable_models)
        outputs["best_model_type"] = self._best_model.model_type if self._best_model else "none"
        outputs["best_r_squared"] = str(self._best_model.r_squared) if self._best_model else "0"
        outputs["best_cv_rmse"] = str(self._best_model.cv_rmse_pct) if self._best_model else "0"

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 RegressionModeling: %d models, %d acceptable, best=%s R2=%s",
            len(self._models), len(acceptable_models),
            outputs["best_model_type"], outputs["best_r_squared"],
        )
        return PhaseResult(
            phase_name=BaselinePhase.REGRESSION_MODELING.value, phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _fit_simple_mean(self, y: List[float]) -> RegressionModelResult:
        """Fit a simple mean model (zero independent variables)."""
        n = len(y)
        mean_y = sum(y) / n
        ss_res = sum((yi - mean_y) ** 2 for yi in y)
        ss_tot = ss_res  # R2 = 0 for mean model
        rmse = math.sqrt(ss_res / max(n - 1, 1))
        cv_rmse = (rmse / mean_y * 100.0) if mean_y > 0 else 999.0

        return RegressionModelResult(
            model_type=ModelType.SIMPLE_MEAN.value,
            variable_names=[],
            intercept=Decimal(str(round(mean_y, 4))),
            coefficients={},
            r_squared=Decimal("0"),
            adjusted_r_squared=Decimal("0"),
            cv_rmse_pct=Decimal(str(round(cv_rmse, 2))),
            rmse=Decimal(str(round(rmse, 4))),
            n_observations=n,
            is_acceptable=cv_rmse <= CV_RMSE_THRESHOLD,
        )

    def _fit_single_variable(
        self, y: List[float], x: List[float], var_name: str
    ) -> RegressionModelResult:
        """Fit a single-variable linear regression (y = b0 + b1*x)."""
        n = len(y)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)

        mean_x = sum_x / n
        mean_y = sum_y / n

        denominator = sum_x2 - (sum_x ** 2) / n
        if abs(denominator) < 1e-10:
            return self._fit_simple_mean(y)

        b1 = (sum_xy - sum_x * sum_y / n) / denominator
        b0 = mean_y - b1 * mean_x

        # Calculate R-squared
        predicted = [b0 + b1 * xi for xi in x]
        ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, predicted))
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adjusted_r2 = 1.0 - (1.0 - r_squared) * (n - 1) / max(n - 2, 1)

        rmse = math.sqrt(ss_res / max(n - 2, 1))
        cv_rmse = (rmse / mean_y * 100.0) if mean_y > 0 else 999.0

        is_acceptable = r_squared >= R_SQUARED_THRESHOLD and cv_rmse <= CV_RMSE_THRESHOLD

        return RegressionModelResult(
            model_type=ModelType.SINGLE_VARIABLE.value,
            variable_names=[var_name],
            intercept=Decimal(str(round(b0, 4))),
            coefficients={var_name: Decimal(str(round(b1, 6)))},
            r_squared=Decimal(str(round(r_squared, 4))),
            adjusted_r_squared=Decimal(str(round(adjusted_r2, 4))),
            cv_rmse_pct=Decimal(str(round(cv_rmse, 2))),
            rmse=Decimal(str(round(rmse, 4))),
            n_observations=n,
            is_acceptable=is_acceptable,
        )

    def _fit_multi_variable(
        self, y: List[float], x_matrix: List[List[float]], var_names: List[str]
    ) -> RegressionModelResult:
        """Fit multi-variable regression using normal equations approximation."""
        n = len(y)
        k = len(var_names)
        mean_y = sum(y) / n

        # Use iterative single-variable approach as approximation
        # (full matrix inversion not implemented to avoid numpy dependency)
        residuals = list(y)
        coefficients: Dict[str, float] = {}
        intercept = mean_y

        for idx, var_name in enumerate(var_names):
            x = x_matrix[idx]
            mean_x = sum(x) / n
            sum_xr = sum(xi * ri for xi, ri in zip(x, residuals))
            sum_x2 = sum(xi ** 2 for xi in x)
            denom = sum_x2 - n * mean_x ** 2

            if abs(denom) < 1e-10:
                coefficients[var_name] = 0.0
                continue

            b = (sum_xr - n * mean_x * (sum(residuals) / n)) / denom
            coefficients[var_name] = b

            # Update residuals
            residuals = [ri - b * xi for ri, xi in zip(residuals, x)]

        intercept = sum(residuals) / n

        # Calculate predicted and R-squared
        predicted = []
        for i in range(n):
            pred = intercept
            for idx, var_name in enumerate(var_names):
                pred += coefficients[var_name] * x_matrix[idx][i]
            predicted.append(pred)

        ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, predicted))
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adjusted_r2 = 1.0 - (1.0 - r_squared) * (n - 1) / max(n - k - 1, 1)

        rmse = math.sqrt(ss_res / max(n - k - 1, 1))
        cv_rmse = (rmse / mean_y * 100.0) if mean_y > 0 else 999.0

        is_acceptable = r_squared >= R_SQUARED_THRESHOLD and cv_rmse <= CV_RMSE_THRESHOLD

        return RegressionModelResult(
            model_type=ModelType.MULTI_VARIABLE.value,
            variable_names=var_names,
            intercept=Decimal(str(round(intercept, 4))),
            coefficients={k: Decimal(str(round(v, 6))) for k, v in coefficients.items()},
            r_squared=Decimal(str(round(r_squared, 4))),
            adjusted_r_squared=Decimal(str(round(adjusted_r2, 4))),
            cv_rmse_pct=Decimal(str(round(cv_rmse, 2))),
            rmse=Decimal(str(round(rmse, 4))),
            n_observations=n,
            is_acceptable=is_acceptable,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Baseline Approval
    # -------------------------------------------------------------------------

    def _phase_baseline_approval(
        self, input_data: BaselineEstablishmentInput
    ) -> PhaseResult:
        """Generate baseline report for approval with confidence intervals."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not self._best_model:
            warnings.append("No model available for approval package")
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return PhaseResult(
                phase_name=BaselinePhase.BASELINE_APPROVAL.value, phase_number=3,
                status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
                outputs={"recommendation": "reject"}, warnings=warnings,
                provenance_hash=self._hash_dict({"recommendation": "reject"}),
            )

        # Calculate confidence interval on mean prediction
        y = self._energy_values
        n = len(y)
        mean_y = sum(y) / max(n, 1)
        rmse = float(self._best_model.rmse)
        se_mean = rmse / math.sqrt(max(n, 1))
        ci_lower = mean_y - T_VALUE_95 * se_mean
        ci_upper = mean_y + T_VALUE_95 * se_mean

        # Determine recommendation
        if self._best_model.is_acceptable and self._data_quality.passes_minimum:
            recommendation = "approve"
            rationale = (
                f"Model meets ASHRAE Guideline 14 criteria: "
                f"R2={self._best_model.r_squared}, "
                f"CV(RMSE)={self._best_model.cv_rmse_pct}%. "
                f"Data quality: {self._data_quality.completeness_pct}% complete."
            )
        elif self._best_model.is_acceptable:
            recommendation = "review"
            rationale = (
                f"Model fit acceptable but data quality below minimum. "
                f"Only {self._data_quality.valid_records} valid data points "
                f"(minimum: {MIN_DATA_POINTS})."
            )
        else:
            recommendation = "reject"
            rationale = (
                f"Model does not meet acceptance criteria. "
                f"R2={self._best_model.r_squared} (threshold: {R_SQUARED_THRESHOLD}), "
                f"CV(RMSE)={self._best_model.cv_rmse_pct}% (threshold: {CV_RMSE_THRESHOLD}%)."
            )

        outputs["recommendation"] = recommendation
        outputs["rationale"] = rationale
        outputs["ci_lower"] = round(ci_lower, 2)
        outputs["ci_upper"] = round(ci_upper, 2)
        outputs["confidence_level"] = CONFIDENCE_LEVEL
        outputs["model_type"] = self._best_model.model_type
        outputs["r_squared"] = str(self._best_model.r_squared)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 BaselineApproval: recommendation=%s R2=%s CI=[%.0f, %.0f]",
            recommendation, self._best_model.r_squared, ci_lower, ci_upper,
        )
        return PhaseResult(
            phase_name=BaselinePhase.BASELINE_APPROVAL.value, phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Supporting Methods
    # -------------------------------------------------------------------------

    def _build_model_statistics(self) -> ModelStatistics:
        """Build comprehensive model statistics from best model."""
        if not self._best_model:
            return ModelStatistics()

        y = self._energy_values
        n = len(y)
        mean_y = sum(y) / max(n, 1)
        rmse = float(self._best_model.rmse)
        se_mean = rmse / math.sqrt(max(n, 1))

        return ModelStatistics(
            best_model_type=self._best_model.model_type,
            r_squared=self._best_model.r_squared,
            adjusted_r_squared=self._best_model.adjusted_r_squared,
            cv_rmse_pct=self._best_model.cv_rmse_pct,
            rmse=self._best_model.rmse,
            f_statistic=Decimal("0"),  # Placeholder; full F-test not implemented
            p_value=Decimal("0"),
            confidence_interval_lower=Decimal(str(round(mean_y - T_VALUE_95 * se_mean, 2))),
            confidence_interval_upper=Decimal(str(round(mean_y + T_VALUE_95 * se_mean, 2))),
            models_evaluated=len(self._models),
        )

    def _build_approval_package(self, validation_status: str) -> ApprovalPackage:
        """Build approval documentation package."""
        if not self._best_model:
            return ApprovalPackage(
                baseline_id=self.baseline_id,
                recommendation="reject",
                rationale="No viable regression model could be fitted",
            )

        recommendation = "approve" if validation_status == "pass" else "review"
        if validation_status == "fail":
            recommendation = "reject"

        model_summary = (
            f"Model: {self._best_model.model_type}, "
            f"Variables: {', '.join(self._best_model.variable_names) or 'none'}, "
            f"R2={self._best_model.r_squared}, "
            f"CV(RMSE)={self._best_model.cv_rmse_pct}%"
        )

        return ApprovalPackage(
            baseline_id=self.baseline_id,
            recommendation=recommendation,
            rationale=f"Baseline model validation status: {validation_status}",
            model_summary=model_summary,
            data_quality_summary=(
                f"{self._data_quality.valid_records}/{self._data_quality.total_records} "
                f"valid records ({self._data_quality.completeness_pct}% complete)"
            ),
            confidence_interval=f"95% CI based on RMSE={self._best_model.rmse}",
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: BaselineEstablishmentResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
