# -*- coding: utf-8 -*-
"""
Waste Decarbonization Base Module
==================================

This module provides base classes and common functionality for all
Waste & Circularity Decarbonization agents.

Design Principles:
- Zero-hallucination guarantee (no LLM in calculation path)
- Full audit trail with SHA-256 provenance hashing
- Circular economy aligned planning
- Science-based target methodology support
- EPA Waste Reduction Model (WARM) compatible

Reference Standards:
- Science Based Targets Initiative (SBTi)
- Ellen MacArthur Foundation Circular Economy
- EPA WARM Model
- EU Circular Economy Action Plan
- Zero Waste International Alliance

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

InputT = TypeVar("InputT", bound="WasteDecarbInput")
OutputT = TypeVar("OutputT", bound="WasteDecarbOutput")


# =============================================================================
# ENUMS
# =============================================================================

class DecarbonizationStrategy(str, Enum):
    """Waste decarbonization strategies."""
    WASTE_REDUCTION = "waste_reduction"  # Source reduction
    REUSE = "reuse"
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"
    LANDFILL_GAS_CAPTURE = "landfill_gas_capture"
    WASTE_TO_ENERGY = "waste_to_energy"
    INDUSTRIAL_SYMBIOSIS = "industrial_symbiosis"
    EXTENDED_PRODUCER_RESPONSIBILITY = "epr"
    CIRCULAR_DESIGN = "circular_design"


class ImplementationTimeline(str, Enum):
    """Implementation timeline categories."""
    IMMEDIATE = "immediate"  # 0-1 years
    SHORT_TERM = "short_term"  # 1-3 years
    MEDIUM_TERM = "medium_term"  # 3-5 years
    LONG_TERM = "long_term"  # 5-10 years


class CostCategory(str, Enum):
    """Cost categories for interventions."""
    NO_COST = "no_cost"
    LOW_COST = "low_cost"  # <$10/tCO2e
    MODERATE_COST = "moderate_cost"  # $10-50/tCO2e
    HIGH_COST = "high_cost"  # $50-100/tCO2e
    VERY_HIGH_COST = "very_high_cost"  # >$100/tCO2e


class ConfidenceLevel(str, Enum):
    """Confidence level for projections."""
    HIGH = "high"  # >90% confidence
    MEDIUM = "medium"  # 70-90% confidence
    LOW = "low"  # 50-70% confidence
    VERY_LOW = "very_low"  # <50% confidence


# =============================================================================
# DATA MODELS
# =============================================================================

class DecarbonizationIntervention(BaseModel):
    """Individual decarbonization intervention."""
    intervention_id: str = Field(..., description="Unique intervention identifier")
    strategy: DecarbonizationStrategy = Field(..., description="Decarbonization strategy")
    description: str = Field(..., description="Intervention description")
    timeline: ImplementationTimeline = Field(..., description="Implementation timeline")
    cost_category: CostCategory = Field(..., description="Cost category")

    # Reduction potential
    reduction_potential_kg_co2e: Decimal = Field(
        Decimal("0"), description="Annual reduction potential in kg CO2e"
    )
    reduction_potential_pct: Decimal = Field(
        Decimal("0"), ge=0, le=100, description="Reduction as percentage of baseline"
    )

    # Cost
    capex_usd: Decimal = Field(Decimal("0"), ge=0, description="Capital expenditure")
    annual_opex_usd: Decimal = Field(Decimal("0"), description="Annual operating expense")
    payback_years: Optional[Decimal] = Field(None, ge=0, description="Payback period")

    # Co-benefits
    co_benefits: List[str] = Field(default_factory=list, description="List of co-benefits")

    # Confidence
    confidence: ConfidenceLevel = Field(
        ConfidenceLevel.MEDIUM, description="Confidence in estimates"
    )

    class Config:
        use_enum_values = True


class DecarbonizationPathway(BaseModel):
    """Complete decarbonization pathway."""
    pathway_id: str = Field(..., description="Unique pathway identifier")
    name: str = Field(..., description="Pathway name")
    description: str = Field(..., description="Pathway description")

    # Interventions
    interventions: List[DecarbonizationIntervention] = Field(
        default_factory=list, description="List of interventions"
    )

    # Totals
    total_reduction_kg_co2e: Decimal = Field(Decimal("0"))
    total_capex_usd: Decimal = Field(Decimal("0"))
    total_annual_opex_usd: Decimal = Field(Decimal("0"))

    # Timeline
    start_year: int = Field(..., ge=2020, le=2100)
    target_year: int = Field(..., ge=2020, le=2100)

    # Achievement
    baseline_emissions_kg_co2e: Decimal = Field(Decimal("0"))
    target_emissions_kg_co2e: Decimal = Field(Decimal("0"))
    reduction_pct: Decimal = Field(Decimal("0"))


# =============================================================================
# BASE INPUT/OUTPUT MODELS
# =============================================================================

class WasteDecarbInput(BaseModel):
    """Base input model for waste decarbonization agents."""

    # Identification
    organization_id: str = Field(..., description="Organization identifier")
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    request_id: Optional[str] = Field(None, description="Unique request ID")

    # Baseline
    baseline_year: int = Field(..., ge=2015, le=2030, description="Baseline year")
    baseline_emissions_kg_co2e: Decimal = Field(
        ..., ge=0, description="Baseline emissions"
    )

    # Targets
    target_year: int = Field(..., ge=2025, le=2060, description="Target year")
    target_reduction_pct: Decimal = Field(
        Decimal("50"), ge=0, le=100, description="Target reduction percentage"
    )

    # Current state
    current_year: int = Field(..., ge=2020, le=2030, description="Current year")
    current_emissions_kg_co2e: Optional[Decimal] = Field(
        None, ge=0, description="Current emissions"
    )

    # Constraints
    max_capex_usd: Optional[Decimal] = Field(None, ge=0, description="Maximum CAPEX")
    max_payback_years: Optional[Decimal] = Field(None, ge=0, description="Max payback period")

    class Config:
        use_enum_values = True


class WasteDecarbOutput(BaseModel):
    """Base output model for waste decarbonization agents."""

    # Agent identification
    agent_id: str = Field(..., description="Agent identifier")
    agent_version: str = Field("1.0.0", description="Agent version")

    # Pathways
    recommended_pathway: Optional[DecarbonizationPathway] = Field(
        None, description="Recommended pathway"
    )
    alternative_pathways: List[DecarbonizationPathway] = Field(
        default_factory=list, description="Alternative pathways"
    )

    # Summary
    baseline_emissions_kg_co2e: Decimal = Field(Decimal("0"))
    achievable_reduction_kg_co2e: Decimal = Field(Decimal("0"))
    achievable_reduction_pct: Decimal = Field(Decimal("0"))
    gap_to_target_kg_co2e: Decimal = Field(Decimal("0"))

    # Costs
    total_capex_usd: Decimal = Field(Decimal("0"))
    total_annual_opex_usd: Decimal = Field(Decimal("0"))
    average_abatement_cost_usd_per_tco2e: Decimal = Field(Decimal("0"))

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
# REDUCTION POTENTIAL DATABASE
# =============================================================================

# Typical reduction potentials by strategy (% of baseline)
REDUCTION_POTENTIALS: Dict[str, Dict[str, Any]] = {
    DecarbonizationStrategy.WASTE_REDUCTION.value: {
        "typical_pct": Decimal("20"),
        "range_low": Decimal("10"),
        "range_high": Decimal("40"),
        "cost_category": CostCategory.LOW_COST,
        "timeline": ImplementationTimeline.SHORT_TERM,
    },
    DecarbonizationStrategy.REUSE.value: {
        "typical_pct": Decimal("10"),
        "range_low": Decimal("5"),
        "range_high": Decimal("20"),
        "cost_category": CostCategory.LOW_COST,
        "timeline": ImplementationTimeline.SHORT_TERM,
    },
    DecarbonizationStrategy.RECYCLING.value: {
        "typical_pct": Decimal("25"),
        "range_low": Decimal("15"),
        "range_high": Decimal("40"),
        "cost_category": CostCategory.MODERATE_COST,
        "timeline": ImplementationTimeline.MEDIUM_TERM,
    },
    DecarbonizationStrategy.COMPOSTING.value: {
        "typical_pct": Decimal("15"),
        "range_low": Decimal("8"),
        "range_high": Decimal("25"),
        "cost_category": CostCategory.MODERATE_COST,
        "timeline": ImplementationTimeline.SHORT_TERM,
    },
    DecarbonizationStrategy.LANDFILL_GAS_CAPTURE.value: {
        "typical_pct": Decimal("60"),
        "range_low": Decimal("40"),
        "range_high": Decimal("85"),
        "cost_category": CostCategory.MODERATE_COST,
        "timeline": ImplementationTimeline.MEDIUM_TERM,
    },
    DecarbonizationStrategy.WASTE_TO_ENERGY.value: {
        "typical_pct": Decimal("70"),
        "range_low": Decimal("50"),
        "range_high": Decimal("85"),
        "cost_category": CostCategory.HIGH_COST,
        "timeline": ImplementationTimeline.LONG_TERM,
    },
    DecarbonizationStrategy.INDUSTRIAL_SYMBIOSIS.value: {
        "typical_pct": Decimal("30"),
        "range_low": Decimal("15"),
        "range_high": Decimal("50"),
        "cost_category": CostCategory.MODERATE_COST,
        "timeline": ImplementationTimeline.MEDIUM_TERM,
    },
}

# Abatement costs (USD per tCO2e reduced)
ABATEMENT_COSTS: Dict[str, Decimal] = {
    DecarbonizationStrategy.WASTE_REDUCTION.value: Decimal("-50"),  # Net savings
    DecarbonizationStrategy.REUSE.value: Decimal("-30"),
    DecarbonizationStrategy.RECYCLING.value: Decimal("20"),
    DecarbonizationStrategy.COMPOSTING.value: Decimal("15"),
    DecarbonizationStrategy.LANDFILL_GAS_CAPTURE.value: Decimal("25"),
    DecarbonizationStrategy.WASTE_TO_ENERGY.value: Decimal("80"),
    DecarbonizationStrategy.INDUSTRIAL_SYMBIOSIS.value: Decimal("10"),
}


# =============================================================================
# BASE WASTE DECARBONIZATION AGENT
# =============================================================================

class BaseWasteDecarbAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for waste decarbonization agents.

    All waste decarbonization agents inherit from this class and implement
    the plan() method with strategy-specific logic.

    Key Guarantees:
    - ZERO HALLUCINATION: No LLM calls in calculation path
    - DETERMINISTIC: Same input always produces same output
    - AUDITABLE: Complete SHA-256 provenance tracking
    - SCIENCE-BASED: Aligned with SBTi and circular economy principles

    Attributes:
        AGENT_ID: Unique agent identifier (e.g., GL-DECARB-WST-001)
        AGENT_NAME: Human-readable agent name
        AGENT_VERSION: Semantic version string
        STRATEGY: Primary decarbonization strategy
    """

    AGENT_ID: str = "GL-DECARB-WST-000"
    AGENT_NAME: str = "Base Waste Decarbonization Agent"
    AGENT_VERSION: str = "1.0.0"
    STRATEGY: DecarbonizationStrategy = DecarbonizationStrategy.WASTE_REDUCTION

    def __init__(self):
        """Initialize the waste decarbonization agent."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._reduction_potentials = REDUCTION_POTENTIALS
        self._abatement_costs = ABATEMENT_COSTS
        self.logger.info(f"Initialized {self.AGENT_ID} v{self.AGENT_VERSION}")

    @abstractmethod
    def plan(self, input_data: InputT) -> OutputT:
        """
        Generate decarbonization plan.

        Args:
            input_data: Planning input data

        Returns:
            Complete plan with pathways and recommendations
        """
        pass

    def _get_reduction_potential(
        self,
        strategy: DecarbonizationStrategy,
        baseline_kg: Decimal,
    ) -> Dict[str, Decimal]:
        """Get reduction potential for a strategy."""
        potential = self._reduction_potentials.get(
            strategy.value,
            self._reduction_potentials[DecarbonizationStrategy.WASTE_REDUCTION.value]
        )

        typical_kg = (baseline_kg * potential["typical_pct"] / Decimal("100")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        low_kg = (baseline_kg * potential["range_low"] / Decimal("100")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        high_kg = (baseline_kg * potential["range_high"] / Decimal("100")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        return {
            "typical_kg": typical_kg,
            "low_kg": low_kg,
            "high_kg": high_kg,
            "typical_pct": potential["typical_pct"],
        }

    def _calculate_abatement_cost(
        self,
        strategy: DecarbonizationStrategy,
        reduction_kg: Decimal,
    ) -> Decimal:
        """Calculate abatement cost for a strategy."""
        cost_per_tonne = self._abatement_costs.get(
            strategy.value, Decimal("50")
        )
        reduction_tonnes = reduction_kg / Decimal("1000")
        return (cost_per_tonne * reduction_tonnes).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    def _create_intervention(
        self,
        intervention_id: str,
        strategy: DecarbonizationStrategy,
        description: str,
        baseline_kg: Decimal,
        reduction_pct: Optional[Decimal] = None,
    ) -> DecarbonizationIntervention:
        """Create a decarbonization intervention."""
        potential = self._reduction_potentials.get(
            strategy.value,
            self._reduction_potentials[DecarbonizationStrategy.WASTE_REDUCTION.value]
        )

        pct = reduction_pct or potential["typical_pct"]
        reduction_kg = (baseline_kg * pct / Decimal("100")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        cost_per_tonne = self._abatement_costs.get(strategy.value, Decimal("50"))
        reduction_tonnes = reduction_kg / Decimal("1000")
        annual_cost = (cost_per_tonne * reduction_tonnes).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return DecarbonizationIntervention(
            intervention_id=intervention_id,
            strategy=strategy,
            description=description,
            timeline=potential["timeline"],
            cost_category=potential["cost_category"],
            reduction_potential_kg_co2e=reduction_kg,
            reduction_potential_pct=pct,
            annual_opex_usd=annual_cost if annual_cost > 0 else Decimal("0"),
            confidence=ConfidenceLevel.MEDIUM,
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

    def _kg_to_metric_tons(self, kg: Decimal) -> Decimal:
        """Convert kg to metric tons."""
        return (kg / Decimal("1000")).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
