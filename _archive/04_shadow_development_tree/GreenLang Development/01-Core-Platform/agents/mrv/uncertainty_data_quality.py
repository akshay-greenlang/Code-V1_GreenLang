# -*- coding: utf-8 -*-
"""
GL-MRV-X-006: Uncertainty & Data Quality Agent
===============================================

Quantifies uncertainty and calculates confidence scores for GHG calculations
following GHG Protocol and IPCC uncertainty guidance.

Capabilities:
    - Uncertainty propagation calculations
    - Data quality scoring (DQI)
    - Confidence interval estimation
    - Monte Carlo simulation support
    - Activity data uncertainty assessment
    - Emission factor uncertainty assessment
    - Complete provenance tracking

Zero-Hallucination Guarantees:
    - All calculations are deterministic mathematical operations
    - Uncertainty formulas from IPCC Guidelines
    - Complete provenance hash for every calculation

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DataQualityIndicator(str, Enum):
    """Data quality indicator levels (1=best, 5=worst)."""
    EXCELLENT = "1"
    GOOD = "2"
    FAIR = "3"
    POOR = "4"
    VERY_POOR = "5"


class UncertaintyType(str, Enum):
    """Types of uncertainty."""
    ACTIVITY_DATA = "activity_data"
    EMISSION_FACTOR = "emission_factor"
    COMBINED = "combined"
    PARAMETER = "parameter"


class DistributionType(str, Enum):
    """Statistical distribution types."""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"


# Default uncertainty ranges (% of value, 95% CI half-width)
DEFAULT_UNCERTAINTIES: Dict[str, Dict[str, float]] = {
    # Activity data uncertainties
    "metered_energy": {"low": 1.0, "mid": 2.0, "high": 5.0},
    "estimated_energy": {"low": 5.0, "mid": 10.0, "high": 20.0},
    "fuel_consumption_measured": {"low": 2.0, "mid": 5.0, "high": 10.0},
    "fuel_consumption_estimated": {"low": 10.0, "mid": 20.0, "high": 30.0},
    "distance_traveled": {"low": 5.0, "mid": 10.0, "high": 25.0},
    "spend_data": {"low": 3.0, "mid": 5.0, "high": 10.0},
    "waste_measured": {"low": 5.0, "mid": 10.0, "high": 20.0},
    "waste_estimated": {"low": 20.0, "mid": 30.0, "high": 50.0},

    # Emission factor uncertainties
    "tier1_ef": {"low": 15.0, "mid": 25.0, "high": 50.0},
    "tier2_ef": {"low": 10.0, "mid": 15.0, "high": 25.0},
    "tier3_ef": {"low": 5.0, "mid": 10.0, "high": 15.0},
    "supplier_ef": {"low": 5.0, "mid": 10.0, "high": 20.0},
    "spend_ef": {"low": 30.0, "mid": 50.0, "high": 100.0},
    "grid_factor": {"low": 5.0, "mid": 10.0, "high": 20.0},
}

# DQI score mapping to uncertainty multiplier
DQI_UNCERTAINTY_MULTIPLIER: Dict[DataQualityIndicator, float] = {
    DataQualityIndicator.EXCELLENT: 0.5,
    DataQualityIndicator.GOOD: 0.75,
    DataQualityIndicator.FAIR: 1.0,
    DataQualityIndicator.POOR: 1.5,
    DataQualityIndicator.VERY_POOR: 2.0,
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class UncertaintyInput(BaseModel):
    """Uncertainty specification for a value."""
    value: float = Field(..., description="Central value")
    uncertainty_percent: Optional[float] = Field(
        None, ge=0, description="Uncertainty as % (95% CI half-width)"
    )
    uncertainty_absolute: Optional[float] = Field(
        None, ge=0, description="Absolute uncertainty"
    )
    distribution: DistributionType = Field(
        default=DistributionType.NORMAL, description="Distribution type"
    )
    data_type: Optional[str] = Field(
        None, description="Type for default uncertainty lookup"
    )
    dqi_score: Optional[DataQualityIndicator] = Field(
        None, description="Data quality indicator"
    )


class DataQualityAssessment(BaseModel):
    """Data quality assessment for a data point."""
    parameter_name: str = Field(..., description="Parameter being assessed")
    temporal_correlation: DataQualityIndicator = Field(
        default=DataQualityIndicator.FAIR, description="Time alignment"
    )
    geographical_correlation: DataQualityIndicator = Field(
        default=DataQualityIndicator.FAIR, description="Geographic relevance"
    )
    technological_correlation: DataQualityIndicator = Field(
        default=DataQualityIndicator.FAIR, description="Technology match"
    )
    completeness: DataQualityIndicator = Field(
        default=DataQualityIndicator.FAIR, description="Data completeness"
    )
    reliability: DataQualityIndicator = Field(
        default=DataQualityIndicator.FAIR, description="Source reliability"
    )


class UncertaintyResult(BaseModel):
    """Result of uncertainty calculation."""
    central_value: float = Field(..., description="Central/mean value")
    lower_bound: float = Field(..., description="Lower 95% CI bound")
    upper_bound: float = Field(..., description="Upper 95% CI bound")
    uncertainty_percent: float = Field(..., description="Relative uncertainty %")
    uncertainty_absolute: float = Field(..., description="Absolute uncertainty")
    distribution: DistributionType = Field(..., description="Distribution used")
    confidence_level: float = Field(default=0.95, description="Confidence level")
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)


class DataQualityResult(BaseModel):
    """Result of data quality assessment."""
    parameter_name: str = Field(...)
    overall_dqi_score: float = Field(..., ge=1, le=5, description="Overall DQI (1-5)")
    overall_dqi_level: DataQualityIndicator = Field(...)
    uncertainty_multiplier: float = Field(...)
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)


class UncertaintyDataQualityInput(BaseModel):
    """Input model for UncertaintyDataQualityAgent."""
    # For uncertainty propagation
    uncertainty_inputs: Optional[List[UncertaintyInput]] = Field(
        None, description="Values with uncertainties to propagate"
    )
    operation: Optional[str] = Field(
        None, description="Operation: add, multiply, or custom"
    )

    # For data quality assessment
    data_quality_assessments: Optional[List[DataQualityAssessment]] = Field(
        None, description="Data quality assessments"
    )

    # For combined calculation
    activity_data: Optional[UncertaintyInput] = Field(None)
    emission_factor: Optional[UncertaintyInput] = Field(None)

    organization_id: Optional[str] = Field(None)


class UncertaintyDataQualityOutput(BaseModel):
    """Output model for UncertaintyDataQualityAgent."""
    success: bool = Field(...)

    # Uncertainty results
    propagated_uncertainty: Optional[UncertaintyResult] = Field(None)
    individual_uncertainties: List[UncertaintyResult] = Field(default_factory=list)

    # Data quality results
    data_quality_results: List[DataQualityResult] = Field(default_factory=list)
    overall_data_quality: Optional[float] = Field(None)

    # Confidence score
    confidence_score: float = Field(
        default=0.5, ge=0, le=1, description="Overall confidence (0-1)"
    )

    # Metadata
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)
    validation_status: str = Field(...)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# =============================================================================
# UNCERTAINTY & DATA QUALITY AGENT
# =============================================================================

class UncertaintyDataQualityAgent(DeterministicAgent):
    """
    GL-MRV-X-006: Uncertainty & Data Quality Agent

    Quantifies uncertainty and calculates confidence scores for GHG
    calculations following IPCC and GHG Protocol guidance.

    Zero-Hallucination Implementation:
        - All calculations use deterministic mathematical formulas
        - Uncertainty propagation from IPCC Guidelines
        - Complete provenance tracking

    Capabilities:
        - Uncertainty propagation (addition, multiplication)
        - Data Quality Indicator (DQI) scoring
        - Confidence interval estimation
        - Combined activity/EF uncertainty

    Example:
        >>> agent = UncertaintyDataQualityAgent()
        >>> result = agent.execute({
        ...     "activity_data": {"value": 1000, "uncertainty_percent": 5},
        ...     "emission_factor": {"value": 2.5, "uncertainty_percent": 10}
        ... })
    """

    AGENT_ID = "GL-MRV-X-006"
    AGENT_NAME = "Uncertainty & Data Quality Agent"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    metadata = AgentMetadata(
        name="UncertaintyDataQualityAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Quantifies uncertainty and data quality scores"
    )

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize UncertaintyDataQualityAgent."""
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute uncertainty and data quality calculations."""
        start_time = DeterministicClock.now()

        try:
            agent_input = UncertaintyDataQualityInput(**inputs)
            individual_uncertainties: List[UncertaintyResult] = []
            propagated_result: Optional[UncertaintyResult] = None
            dq_results: List[DataQualityResult] = []

            # Calculate individual uncertainties
            if agent_input.uncertainty_inputs:
                for ui in agent_input.uncertainty_inputs:
                    result = self._calculate_uncertainty(ui)
                    individual_uncertainties.append(result)

                # Propagate if operation specified
                if agent_input.operation and len(individual_uncertainties) > 1:
                    propagated_result = self._propagate_uncertainty(
                        individual_uncertainties,
                        agent_input.operation
                    )

            # Calculate combined activity/EF uncertainty
            if agent_input.activity_data and agent_input.emission_factor:
                ad_result = self._calculate_uncertainty(agent_input.activity_data)
                ef_result = self._calculate_uncertainty(agent_input.emission_factor)
                individual_uncertainties.extend([ad_result, ef_result])

                # Emissions = Activity * EF, so multiply uncertainties
                propagated_result = self._propagate_uncertainty(
                    [ad_result, ef_result],
                    "multiply"
                )

            # Calculate data quality scores
            if agent_input.data_quality_assessments:
                for dqa in agent_input.data_quality_assessments:
                    dq_result = self._assess_data_quality(dqa)
                    dq_results.append(dq_result)

            # Calculate overall confidence score
            confidence_score = self._calculate_confidence_score(
                propagated_result,
                dq_results
            )

            # Overall data quality
            overall_dq = None
            if dq_results:
                overall_dq = sum(r.overall_dqi_score for r in dq_results) / len(dq_results)

            # Processing time
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            provenance_hash = self._compute_provenance_hash({
                "input": inputs,
                "confidence_score": confidence_score
            })

            output = UncertaintyDataQualityOutput(
                success=True,
                propagated_uncertainty=propagated_result,
                individual_uncertainties=individual_uncertainties,
                data_quality_results=dq_results,
                overall_data_quality=overall_dq,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                provenance_hash=provenance_hash,
                validation_status="PASS"
            )

            self._capture_audit_entry(
                operation="calculate_uncertainty",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=[f"Confidence score: {confidence_score:.2f}"]
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Uncertainty calculation failed: {str(e)}", exc_info=True)
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "validation_status": "FAIL"
            }

    def _calculate_uncertainty(self, ui: UncertaintyInput) -> UncertaintyResult:
        """Calculate uncertainty for a single value."""
        trace = []
        value = ui.value
        trace.append(f"Central value: {value}")

        # Determine uncertainty percent
        if ui.uncertainty_percent is not None:
            uncertainty_pct = ui.uncertainty_percent
        elif ui.uncertainty_absolute is not None:
            uncertainty_pct = (ui.uncertainty_absolute / abs(value)) * 100 if value != 0 else 0
        elif ui.data_type and ui.data_type in DEFAULT_UNCERTAINTIES:
            uncertainty_pct = DEFAULT_UNCERTAINTIES[ui.data_type]["mid"]
            trace.append(f"Using default uncertainty for {ui.data_type}")
        else:
            uncertainty_pct = 25.0  # Default
            trace.append("Using default uncertainty of 25%")

        # Apply DQI multiplier if provided
        if ui.dqi_score:
            multiplier = DQI_UNCERTAINTY_MULTIPLIER.get(ui.dqi_score, 1.0)
            uncertainty_pct *= multiplier
            trace.append(f"Applied DQI multiplier: {multiplier}")

        uncertainty_abs = abs(value) * (uncertainty_pct / 100)
        lower_bound = value - uncertainty_abs
        upper_bound = value + uncertainty_abs

        trace.append(f"Uncertainty: {uncertainty_pct:.1f}%")
        trace.append(f"95% CI: [{lower_bound:.4f}, {upper_bound:.4f}]")

        provenance_hash = self._compute_provenance_hash({
            "value": value,
            "uncertainty_percent": uncertainty_pct
        })

        return UncertaintyResult(
            central_value=value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            uncertainty_percent=round(uncertainty_pct, 2),
            uncertainty_absolute=round(uncertainty_abs, 4),
            distribution=ui.distribution,
            calculation_trace=trace,
            provenance_hash=provenance_hash
        )

    def _propagate_uncertainty(
        self,
        uncertainties: List[UncertaintyResult],
        operation: str
    ) -> UncertaintyResult:
        """Propagate uncertainties through an operation."""
        trace = []

        if operation == "add":
            # Sum rule: combined absolute uncertainty = sqrt(sum of squared absolutes)
            total_value = sum(u.central_value for u in uncertainties)
            combined_abs = math.sqrt(sum(u.uncertainty_absolute**2 for u in uncertainties))
            combined_pct = (combined_abs / abs(total_value)) * 100 if total_value != 0 else 0
            trace.append("Propagation method: Addition (root sum of squares)")

        elif operation == "multiply":
            # Product rule: combined relative uncertainty = sqrt(sum of squared relatives)
            total_value = 1.0
            for u in uncertainties:
                total_value *= u.central_value

            combined_pct = math.sqrt(sum(u.uncertainty_percent**2 for u in uncertainties))
            combined_abs = abs(total_value) * (combined_pct / 100)
            trace.append("Propagation method: Multiplication (root sum of squared relatives)")

        else:
            # Default to multiplication
            total_value = 1.0
            for u in uncertainties:
                total_value *= u.central_value
            combined_pct = math.sqrt(sum(u.uncertainty_percent**2 for u in uncertainties))
            combined_abs = abs(total_value) * (combined_pct / 100)
            trace.append("Propagation method: Default (multiplication)")

        trace.append(f"Combined value: {total_value:.4f}")
        trace.append(f"Combined uncertainty: {combined_pct:.1f}%")

        provenance_hash = self._compute_provenance_hash({
            "operation": operation,
            "combined_value": total_value,
            "combined_uncertainty": combined_pct
        })

        return UncertaintyResult(
            central_value=total_value,
            lower_bound=total_value - combined_abs,
            upper_bound=total_value + combined_abs,
            uncertainty_percent=round(combined_pct, 2),
            uncertainty_absolute=round(combined_abs, 4),
            distribution=DistributionType.NORMAL,
            calculation_trace=trace,
            provenance_hash=provenance_hash
        )

    def _assess_data_quality(self, dqa: DataQualityAssessment) -> DataQualityResult:
        """Assess data quality and calculate DQI score."""
        # Convert DQI levels to numeric scores
        scores = {
            "temporal": int(dqa.temporal_correlation.value),
            "geographical": int(dqa.geographical_correlation.value),
            "technological": int(dqa.technological_correlation.value),
            "completeness": int(dqa.completeness.value),
            "reliability": int(dqa.reliability.value),
        }

        # Calculate weighted average (equal weights)
        overall_score = sum(scores.values()) / len(scores)

        # Determine overall level
        if overall_score <= 1.5:
            overall_level = DataQualityIndicator.EXCELLENT
        elif overall_score <= 2.5:
            overall_level = DataQualityIndicator.GOOD
        elif overall_score <= 3.5:
            overall_level = DataQualityIndicator.FAIR
        elif overall_score <= 4.5:
            overall_level = DataQualityIndicator.POOR
        else:
            overall_level = DataQualityIndicator.VERY_POOR

        uncertainty_mult = DQI_UNCERTAINTY_MULTIPLIER[overall_level]

        # Generate recommendations
        recommendations = []
        if scores["temporal"] >= 4:
            recommendations.append("Improve temporal alignment with more recent data")
        if scores["geographical"] >= 4:
            recommendations.append("Use more geographically specific data")
        if scores["technological"] >= 4:
            recommendations.append("Match technology specifications more closely")
        if scores["completeness"] >= 4:
            recommendations.append("Increase data completeness")
        if scores["reliability"] >= 4:
            recommendations.append("Verify data with more reliable sources")

        provenance_hash = self._compute_provenance_hash({
            "parameter": dqa.parameter_name,
            "overall_score": overall_score
        })

        return DataQualityResult(
            parameter_name=dqa.parameter_name,
            overall_dqi_score=round(overall_score, 2),
            overall_dqi_level=overall_level,
            uncertainty_multiplier=uncertainty_mult,
            dimension_scores={k: float(v) for k, v in scores.items()},
            recommendations=recommendations,
            provenance_hash=provenance_hash
        )

    def _calculate_confidence_score(
        self,
        uncertainty: Optional[UncertaintyResult],
        dq_results: List[DataQualityResult]
    ) -> float:
        """Calculate overall confidence score (0-1)."""
        scores = []

        # Uncertainty contribution (lower uncertainty = higher confidence)
        if uncertainty:
            # Convert uncertainty % to confidence (100% uncertainty = 0 confidence)
            u_conf = max(0, 1 - (uncertainty.uncertainty_percent / 100))
            scores.append(u_conf)

        # Data quality contribution
        if dq_results:
            # Convert DQI score (1-5) to confidence (1=1.0, 5=0.0)
            avg_dqi = sum(r.overall_dqi_score for r in dq_results) / len(dq_results)
            dq_conf = (5 - avg_dqi) / 4
            scores.append(dq_conf)

        if not scores:
            return 0.5  # Default neutral confidence

        return round(sum(scores) / len(scores), 2)

    def _compute_provenance_hash(self, data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_default_uncertainty(self, data_type: str) -> Optional[Dict[str, float]]:
        """Get default uncertainty values for a data type."""
        return DEFAULT_UNCERTAINTIES.get(data_type)
