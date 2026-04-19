# -*- coding: utf-8 -*-
"""
Waste Adaptation Base Module
=============================

This module provides base classes and common functionality for all
Waste & Circularity Climate Adaptation agents.

Design Principles:
- Zero-hallucination guarantee (no LLM in calculation path)
- Full audit trail with SHA-256 provenance hashing
- TCFD and TNFD aligned risk assessment
- Climate scenario analysis (SSP/RCP)
- Infrastructure resilience focus

Reference Standards:
- TCFD Physical Risk Assessment
- IPCC AR6 Climate Scenarios
- ISO 14090 Climate Adaptation
- ASCE Infrastructure Resilience
- EPA Climate Resilience Evaluation

Author: GreenLang Framework Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE VARIABLES
# =============================================================================

InputT = TypeVar("InputT", bound="WasteAdaptInput")
OutputT = TypeVar("OutputT", bound="WasteAdaptOutput")


# =============================================================================
# ENUMS
# =============================================================================

class ClimateScenario(str, Enum):
    """IPCC climate scenarios."""
    SSP1_19 = "ssp1_1.9"  # Very low emissions
    SSP1_26 = "ssp1_2.6"  # Low emissions
    SSP2_45 = "ssp2_4.5"  # Intermediate
    SSP3_70 = "ssp3_7.0"  # High emissions
    SSP5_85 = "ssp5_8.5"  # Very high emissions


class TimeHorizon(str, Enum):
    """Time horizons for adaptation planning."""
    SHORT_TERM = "short_term"  # 2030
    MEDIUM_TERM = "medium_term"  # 2050
    LONG_TERM = "long_term"  # 2100


class RiskLevel(str, Enum):
    """Climate risk levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AdaptationMeasureType(str, Enum):
    """Types of adaptation measures."""
    STRUCTURAL = "structural"  # Physical infrastructure
    OPERATIONAL = "operational"  # Process changes
    NATURE_BASED = "nature_based"  # Natural solutions
    INSTITUTIONAL = "institutional"  # Policy/governance
    FINANCIAL = "financial"  # Insurance/risk transfer


class UrgencyLevel(str, Enum):
    """Urgency for adaptation action."""
    IMMEDIATE = "immediate"  # Within 1 year
    NEAR_TERM = "near_term"  # 1-3 years
    MEDIUM_TERM = "medium_term"  # 3-10 years
    LONG_TERM = "long_term"  # 10+ years


# =============================================================================
# DATA MODELS
# =============================================================================

class ClimateHazard(BaseModel):
    """Climate hazard definition."""
    hazard_type: str = Field(..., description="Type of hazard")
    current_frequency: str = Field(..., description="Current return period")
    projected_frequency: str = Field(..., description="Projected return period")
    intensity_change_pct: Decimal = Field(Decimal("0"), description="Change in intensity")
    confidence: str = Field("medium", description="Projection confidence")


class VulnerabilityAssessment(BaseModel):
    """Vulnerability assessment results."""
    exposure_score: Decimal = Field(Decimal("0"), ge=0, le=5)
    sensitivity_score: Decimal = Field(Decimal("0"), ge=0, le=5)
    adaptive_capacity_score: Decimal = Field(Decimal("0"), ge=0, le=5)
    overall_vulnerability: RiskLevel = Field(RiskLevel.MODERATE)


class AdaptationMeasure(BaseModel):
    """Individual adaptation measure."""
    measure_id: str = Field(..., description="Unique measure identifier")
    measure_type: AdaptationMeasureType = Field(..., description="Type of measure")
    description: str = Field(..., description="Measure description")
    urgency: UrgencyLevel = Field(UrgencyLevel.MEDIUM_TERM)

    # Risk reduction
    risk_reduction_pct: Decimal = Field(Decimal("0"), ge=0, le=100)
    residual_risk: RiskLevel = Field(RiskLevel.LOW)

    # Costs and benefits
    implementation_cost_usd: Decimal = Field(Decimal("0"), ge=0)
    annual_maintenance_usd: Decimal = Field(Decimal("0"), ge=0)
    avoided_damages_usd: Decimal = Field(Decimal("0"), ge=0)
    benefit_cost_ratio: Decimal = Field(Decimal("0"), ge=0)

    # Co-benefits
    co_benefits: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class AdaptationPlan(BaseModel):
    """Complete adaptation plan."""
    plan_id: str = Field(..., description="Plan identifier")
    name: str = Field(..., description="Plan name")
    time_horizon: TimeHorizon = Field(..., description="Planning horizon")
    climate_scenario: ClimateScenario = Field(..., description="Climate scenario")

    # Risk context
    key_hazards: List[ClimateHazard] = Field(default_factory=list)
    vulnerability: Optional[VulnerabilityAssessment] = None
    baseline_risk: RiskLevel = Field(RiskLevel.MODERATE)

    # Measures
    adaptation_measures: List[AdaptationMeasure] = Field(default_factory=list)

    # Summary
    total_investment_usd: Decimal = Field(Decimal("0"))
    total_avoided_damages_usd: Decimal = Field(Decimal("0"))
    portfolio_bcr: Decimal = Field(Decimal("0"))
    residual_risk: RiskLevel = Field(RiskLevel.LOW)


# =============================================================================
# BASE INPUT/OUTPUT MODELS
# =============================================================================

class WasteAdaptInput(BaseModel):
    """Base input model for waste adaptation agents."""

    # Identification
    organization_id: str = Field(..., description="Organization identifier")
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    request_id: Optional[str] = Field(None, description="Unique request ID")

    # Location
    latitude: Optional[Decimal] = Field(None, ge=-90, le=90)
    longitude: Optional[Decimal] = Field(None, ge=-180, le=180)
    region: str = Field("global", description="Geographic region")
    country: Optional[str] = Field(None, description="Country code")

    # Scenario
    climate_scenario: ClimateScenario = Field(
        ClimateScenario.SSP2_45, description="Climate scenario"
    )
    time_horizon: TimeHorizon = Field(
        TimeHorizon.MEDIUM_TERM, description="Planning horizon"
    )

    # Assessment year
    baseline_year: int = Field(2024, ge=2020, le=2030)
    target_year: Optional[int] = Field(None, ge=2030, le=2100)

    class Config:
        use_enum_values = True


class WasteAdaptOutput(BaseModel):
    """Base output model for waste adaptation agents."""

    # Agent identification
    agent_id: str = Field(..., description="Agent identifier")
    agent_version: str = Field("1.0.0", description="Agent version")

    # Risk assessment
    current_risk_level: RiskLevel = Field(RiskLevel.MODERATE)
    projected_risk_level: RiskLevel = Field(RiskLevel.HIGH)
    risk_change: str = Field("", description="Description of risk change")

    # Adaptation plan
    adaptation_plan: Optional[AdaptationPlan] = None

    # Summary
    total_investment_required_usd: Decimal = Field(Decimal("0"))
    projected_avoided_damages_usd: Decimal = Field(Decimal("0"))
    benefit_cost_ratio: Decimal = Field(Decimal("0"))
    adaptation_deficit_usd: Decimal = Field(Decimal("0"))

    # Audit trail
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    # Timestamps
    calculation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    calculation_duration_ms: float = Field(0.0)

    # Status
    status: str = Field("success")
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


# =============================================================================
# CLIMATE PROJECTION DATA
# =============================================================================

# Temperature change by scenario and time horizon (degrees C)
TEMPERATURE_PROJECTIONS: Dict[str, Dict[str, Decimal]] = {
    ClimateScenario.SSP1_19.value: {
        TimeHorizon.SHORT_TERM.value: Decimal("1.2"),
        TimeHorizon.MEDIUM_TERM.value: Decimal("1.4"),
        TimeHorizon.LONG_TERM.value: Decimal("1.4"),
    },
    ClimateScenario.SSP2_45.value: {
        TimeHorizon.SHORT_TERM.value: Decimal("1.5"),
        TimeHorizon.MEDIUM_TERM.value: Decimal("2.1"),
        TimeHorizon.LONG_TERM.value: Decimal("2.7"),
    },
    ClimateScenario.SSP5_85.value: {
        TimeHorizon.SHORT_TERM.value: Decimal("1.7"),
        TimeHorizon.MEDIUM_TERM.value: Decimal("2.9"),
        TimeHorizon.LONG_TERM.value: Decimal("4.4"),
    },
}

# Precipitation change factors (multiplier)
PRECIPITATION_PROJECTIONS: Dict[str, Dict[str, Decimal]] = {
    ClimateScenario.SSP1_19.value: {
        TimeHorizon.MEDIUM_TERM.value: Decimal("1.05"),
        TimeHorizon.LONG_TERM.value: Decimal("1.08"),
    },
    ClimateScenario.SSP2_45.value: {
        TimeHorizon.MEDIUM_TERM.value: Decimal("1.08"),
        TimeHorizon.LONG_TERM.value: Decimal("1.15"),
    },
    ClimateScenario.SSP5_85.value: {
        TimeHorizon.MEDIUM_TERM.value: Decimal("1.12"),
        TimeHorizon.LONG_TERM.value: Decimal("1.25"),
    },
}


# =============================================================================
# BASE WASTE ADAPTATION AGENT
# =============================================================================

class BaseWasteAdaptAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for waste adaptation agents.

    All waste adaptation agents inherit from this class and implement
    the assess() method with hazard-specific logic.

    Key Guarantees:
    - ZERO HALLUCINATION: No LLM calls in calculation path
    - DETERMINISTIC: Same input always produces same output
    - AUDITABLE: Complete SHA-256 provenance tracking
    - TCFD ALIGNED: Physical risk assessment methodology

    Attributes:
        AGENT_ID: Unique agent identifier (e.g., GL-ADAPT-WST-001)
        AGENT_NAME: Human-readable agent name
        AGENT_VERSION: Semantic version string
        HAZARD_TYPE: Primary climate hazard addressed
    """

    AGENT_ID: str = "GL-ADAPT-WST-000"
    AGENT_NAME: str = "Base Waste Adaptation Agent"
    AGENT_VERSION: str = "1.0.0"
    HAZARD_TYPE: str = "general"

    def __init__(self):
        """Initialize the waste adaptation agent."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._temperature_projections = TEMPERATURE_PROJECTIONS
        self._precipitation_projections = PRECIPITATION_PROJECTIONS
        self.logger.info(f"Initialized {self.AGENT_ID} v{self.AGENT_VERSION}")

    @abstractmethod
    def assess(self, input_data: InputT) -> OutputT:
        """
        Conduct climate risk assessment and adaptation planning.

        Args:
            input_data: Assessment input data

        Returns:
            Complete assessment with adaptation recommendations
        """
        pass

    def _get_temperature_change(
        self,
        scenario: ClimateScenario,
        horizon: TimeHorizon,
    ) -> Decimal:
        """Get projected temperature change."""
        scenario_data = self._temperature_projections.get(
            scenario.value,
            self._temperature_projections[ClimateScenario.SSP2_45.value]
        )
        return scenario_data.get(horizon.value, Decimal("2.0"))

    def _get_precipitation_factor(
        self,
        scenario: ClimateScenario,
        horizon: TimeHorizon,
    ) -> Decimal:
        """Get precipitation change factor."""
        scenario_data = self._precipitation_projections.get(
            scenario.value,
            self._precipitation_projections[ClimateScenario.SSP2_45.value]
        )
        return scenario_data.get(horizon.value, Decimal("1.1"))

    def _calculate_risk_level(
        self,
        exposure: Decimal,
        sensitivity: Decimal,
        adaptive_capacity: Decimal,
    ) -> RiskLevel:
        """Calculate overall risk level from vulnerability components."""
        # Risk = (Exposure * Sensitivity) / Adaptive Capacity
        if adaptive_capacity == Decimal("0"):
            adaptive_capacity = Decimal("0.1")

        risk_score = (exposure * sensitivity / adaptive_capacity).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )

        if risk_score < Decimal("1"):
            return RiskLevel.VERY_LOW
        elif risk_score < Decimal("2"):
            return RiskLevel.LOW
        elif risk_score < Decimal("3"):
            return RiskLevel.MODERATE
        elif risk_score < Decimal("4"):
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _create_adaptation_measure(
        self,
        measure_id: str,
        measure_type: AdaptationMeasureType,
        description: str,
        risk_reduction_pct: Decimal,
        cost_usd: Decimal,
        avoided_damages_usd: Decimal,
    ) -> AdaptationMeasure:
        """Create an adaptation measure with calculated BCR."""
        bcr = (avoided_damages_usd / cost_usd).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        ) if cost_usd > 0 else Decimal("0")

        return AdaptationMeasure(
            measure_id=measure_id,
            measure_type=measure_type,
            description=description,
            risk_reduction_pct=risk_reduction_pct,
            implementation_cost_usd=cost_usd,
            avoided_damages_usd=avoided_damages_usd,
            benefit_cost_ratio=bcr,
        )

    def _generate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> str:
        """Generate SHA-256 provenance hash."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "input": {
                k: str(v) if isinstance(v, Decimal) else v
                for k, v in input_data.items()
            },
            "output": {
                k: str(v) if isinstance(v, Decimal) else v
                for k, v in output_data.items()
                if k not in ["provenance_hash", "calculation_timestamp"]
            },
        }
        data_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
