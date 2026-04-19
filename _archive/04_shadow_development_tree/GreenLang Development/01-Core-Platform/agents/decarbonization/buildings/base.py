# -*- coding: utf-8 -*-
"""
GreenLang Buildings Decarbonization Base Agent
================================================

Base class for all building sector decarbonization agents.
Provides common functionality for retrofit planning, technology recommendations,
and decarbonization pathway analysis.

Design Principles:
    - Recommendation pathway: AI-enhanced analysis with deterministic savings
    - Technology-agnostic: Support for multiple decarbonization technologies
    - Financial integration: NPV, payback, and ROI calculations
    - Standards-aligned: ASHRAE, LEED, passive house certifications

Author: GreenLang Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE VARIABLES
# =============================================================================

InputT = TypeVar("InputT", bound="DecarbonizationInput")
OutputT = TypeVar("OutputT", bound="DecarbonizationOutput")


# =============================================================================
# ENUMS
# =============================================================================

class TechnologyCategory(str, Enum):
    """Decarbonization technology categories."""
    HVAC = "hvac"
    LIGHTING = "lighting"
    ENVELOPE = "envelope"
    RENEWABLES = "renewables"
    STORAGE = "storage"
    CONTROLS = "controls"
    ELECTRIFICATION = "electrification"
    MATERIALS = "materials"
    OPERATIONS = "operations"


class RecommendationPriority(str, Enum):
    """Recommendation priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class ImplementationPhase(str, Enum):
    """Implementation timeline phases."""
    IMMEDIATE = "immediate"  # 0-1 years
    SHORT_TERM = "short_term"  # 1-3 years
    MEDIUM_TERM = "medium_term"  # 3-5 years
    LONG_TERM = "long_term"  # 5-10 years


class RiskLevel(str, Enum):
    """Implementation risk levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =============================================================================
# DATA MODELS
# =============================================================================

class TechnologySpec(BaseModel):
    """Technology specification for decarbonization measure."""
    technology_id: str
    category: TechnologyCategory
    name: str
    description: str
    efficiency_improvement_percent: Optional[Decimal] = None
    energy_savings_kwh_per_year: Optional[Decimal] = None
    emission_reduction_kgco2e_per_year: Optional[Decimal] = None
    lifespan_years: int = Field(default=15, ge=1, le=50)


class FinancialMetrics(BaseModel):
    """Financial analysis for decarbonization measure."""
    capital_cost_usd: Decimal = Field(..., ge=0)
    annual_operating_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    annual_savings_usd: Decimal = Field(default=Decimal("0"), ge=0)
    simple_payback_years: Optional[Decimal] = None
    npv_usd: Optional[Decimal] = None
    irr_percent: Optional[Decimal] = None
    roi_percent: Optional[Decimal] = None
    available_incentives_usd: Decimal = Field(default=Decimal("0"), ge=0)


class DecarbonizationMeasure(BaseModel):
    """Individual decarbonization measure recommendation."""
    measure_id: str
    name: str
    description: str
    technology: TechnologySpec
    priority: RecommendationPriority
    phase: ImplementationPhase
    risk_level: RiskLevel
    financial: FinancialMetrics

    # Impact metrics
    annual_energy_savings_kwh: Decimal = Field(default=Decimal("0"))
    annual_emission_reduction_kgco2e: Decimal = Field(default=Decimal("0"))
    lifetime_emission_reduction_kgco2e: Decimal = Field(default=Decimal("0"))

    # Implementation details
    implementation_notes: Optional[str] = None
    prerequisites: List[str] = Field(default_factory=list)
    co_benefits: List[str] = Field(default_factory=list)


class DecarbonizationPathway(BaseModel):
    """Complete decarbonization pathway with phases."""
    pathway_id: str
    name: str
    description: str
    target_year: int
    target_reduction_percent: Decimal

    # Measures by phase
    immediate_measures: List[DecarbonizationMeasure] = Field(default_factory=list)
    short_term_measures: List[DecarbonizationMeasure] = Field(default_factory=list)
    medium_term_measures: List[DecarbonizationMeasure] = Field(default_factory=list)
    long_term_measures: List[DecarbonizationMeasure] = Field(default_factory=list)

    # Totals
    total_capital_cost_usd: Decimal = Field(default=Decimal("0"))
    total_annual_savings_usd: Decimal = Field(default=Decimal("0"))
    total_emission_reduction_kgco2e: Decimal = Field(default=Decimal("0"))


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class BuildingBaseline(BaseModel):
    """Building baseline data for decarbonization analysis."""
    building_id: str
    building_type: str
    gross_floor_area_sqm: Decimal = Field(..., gt=0)
    year_built: Optional[int] = None

    # Current performance
    current_energy_kwh_per_year: Decimal = Field(..., ge=0)
    current_emissions_kgco2e_per_year: Decimal = Field(..., ge=0)
    current_eui_kwh_per_sqm: Optional[Decimal] = None

    # Energy sources
    electricity_percent: Decimal = Field(default=Decimal("100"), ge=0, le=100)
    natural_gas_percent: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    other_fuel_percent: Decimal = Field(default=Decimal("0"), ge=0, le=100)


class DecarbonizationTarget(BaseModel):
    """Decarbonization target specification."""
    target_year: int = Field(..., ge=2025, le=2100)
    target_type: str = Field(default="net_zero")  # net_zero, percent_reduction, absolute
    target_reduction_percent: Optional[Decimal] = Field(None, ge=0, le=100)
    target_emissions_kgco2e: Optional[Decimal] = Field(None, ge=0)
    interim_targets: Dict[int, Decimal] = Field(default_factory=dict)


class DecarbonizationInput(BaseModel):
    """Base input model for decarbonization agents."""
    building_baseline: BuildingBaseline
    target: DecarbonizationTarget

    # Constraints
    budget_usd: Optional[Decimal] = Field(None, ge=0)
    max_payback_years: Optional[Decimal] = Field(None, ge=0, le=50)

    # Preferences
    technology_preferences: List[TechnologyCategory] = Field(default_factory=list)
    excluded_technologies: List[str] = Field(default_factory=list)

    # Economic parameters
    discount_rate_percent: Decimal = Field(default=Decimal("5"), ge=0, le=50)
    electricity_cost_per_kwh: Decimal = Field(default=Decimal("0.12"), ge=0)
    gas_cost_per_therm: Decimal = Field(default=Decimal("1.50"), ge=0)
    carbon_price_per_tonne: Optional[Decimal] = Field(None, ge=0)


class DecarbonizationOutput(BaseModel):
    """Base output model for decarbonization agents."""
    # Identification
    analysis_id: str
    agent_id: str
    agent_version: str
    timestamp: str

    # Building summary
    building_id: str
    baseline_emissions_kgco2e: Decimal
    target_emissions_kgco2e: Decimal
    target_year: int

    # Recommended pathway
    pathway: Optional[DecarbonizationPathway] = None

    # Summary metrics
    total_reduction_kgco2e: Decimal = Field(default=Decimal("0"))
    total_reduction_percent: Decimal = Field(default=Decimal("0"))
    total_investment_usd: Decimal = Field(default=Decimal("0"))
    total_annual_savings_usd: Decimal = Field(default=Decimal("0"))
    average_payback_years: Optional[Decimal] = None

    # Feasibility
    target_achievable: bool = Field(default=False)
    gap_to_target_kgco2e: Decimal = Field(default=Decimal("0"))

    # Provenance
    provenance_hash: str = Field(default="")

    # Validation
    is_valid: bool = Field(default=True)
    validation_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# BASE AGENT
# =============================================================================

class BuildingDecarbonizationBaseAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for building decarbonization agents.

    These agents analyze buildings and recommend decarbonization measures
    to achieve emission reduction targets.

    Attributes:
        AGENT_ID: Unique agent identifier
        AGENT_VERSION: Semantic version string
        TECHNOLOGY_FOCUS: Primary technology category
    """

    AGENT_ID: str = "GL-DECARB-BLD-BASE"
    AGENT_VERSION: str = "1.0.0"
    TECHNOLOGY_FOCUS: Optional[TechnologyCategory] = None

    def __init__(self):
        """Initialize the decarbonization agent."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._technology_database: Dict[str, TechnologySpec] = {}
        self._initialize()

    def _initialize(self) -> None:
        """Initialize agent resources."""
        self._load_technology_database()

    @abstractmethod
    def _load_technology_database(self) -> None:
        """Load technology specifications. Must be implemented."""
        pass

    @abstractmethod
    def analyze(self, input_data: InputT) -> OutputT:
        """
        Analyze building and recommend decarbonization measures.

        Args:
            input_data: Building baseline and targets

        Returns:
            Decarbonization pathway and recommendations
        """
        pass

    def process(self, input_data: InputT) -> OutputT:
        """
        Main processing method with lifecycle management.

        Args:
            input_data: Input data for analysis

        Returns:
            Complete decarbonization output
        """
        start_time = datetime.now(timezone.utc)

        try:
            self.logger.info(
                f"{self.AGENT_ID} analyzing: building={input_data.building_baseline.building_id}"
            )

            output = self.analyze(input_data)

            # Calculate provenance hash
            output.provenance_hash = self._calculate_hash({
                "input": input_data.model_dump(),
                "output_summary": {
                    "total_reduction": str(output.total_reduction_kgco2e),
                    "total_investment": str(output.total_investment_usd)
                }
            })

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.info(f"{self.AGENT_ID} completed in {duration_ms:.2f}ms")

            return output

        except Exception as e:
            self.logger.error(f"{self.AGENT_ID} failed: {str(e)}", exc_info=True)
            raise

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _to_decimal(self, value: Any) -> Decimal:
        """Convert value to Decimal."""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    def _round_financial(self, value: Decimal, precision: int = 2) -> Decimal:
        """Round financial values."""
        quantize_str = "0." + "0" * precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _round_emissions(self, value: Decimal, precision: int = 4) -> Decimal:
        """Round emission values."""
        quantize_str = "0." + "0" * precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _generate_analysis_id(self, building_id: str) -> str:
        """Generate unique analysis ID."""
        data = f"{self.AGENT_ID}:{building_id}:{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _get_timestamp(self) -> str:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash."""
        def convert(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, Enum):
                return obj.value
            return obj

        converted = convert(data)
        json_str = json.dumps(converted, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _calculate_simple_payback(
        self,
        capital_cost: Decimal,
        annual_savings: Decimal
    ) -> Optional[Decimal]:
        """Calculate simple payback period."""
        if annual_savings <= 0:
            return None
        return self._round_financial(capital_cost / annual_savings, 1)

    def _calculate_npv(
        self,
        capital_cost: Decimal,
        annual_savings: Decimal,
        annual_operating_cost: Decimal,
        lifespan_years: int,
        discount_rate: Decimal
    ) -> Decimal:
        """Calculate Net Present Value."""
        npv = -capital_cost
        annual_net_savings = annual_savings - annual_operating_cost

        for year in range(1, lifespan_years + 1):
            discount_factor = Decimal("1") / ((1 + discount_rate / 100) ** year)
            npv += annual_net_savings * discount_factor

        return self._round_financial(npv)

    def _create_measure(
        self,
        measure_id: str,
        name: str,
        description: str,
        technology: TechnologySpec,
        capital_cost: Decimal,
        annual_savings: Decimal,
        energy_savings_kwh: Decimal,
        emission_reduction: Decimal,
        priority: RecommendationPriority = RecommendationPriority.MEDIUM,
        phase: ImplementationPhase = ImplementationPhase.SHORT_TERM,
        risk: RiskLevel = RiskLevel.LOW,
        discount_rate: Decimal = Decimal("5")
    ) -> DecarbonizationMeasure:
        """Create a decarbonization measure with financial analysis."""
        payback = self._calculate_simple_payback(capital_cost, annual_savings)
        npv = self._calculate_npv(
            capital_cost,
            annual_savings,
            Decimal("0"),
            technology.lifespan_years,
            discount_rate
        )

        lifetime_reduction = emission_reduction * Decimal(str(technology.lifespan_years))

        return DecarbonizationMeasure(
            measure_id=measure_id,
            name=name,
            description=description,
            technology=technology,
            priority=priority,
            phase=phase,
            risk_level=risk,
            financial=FinancialMetrics(
                capital_cost_usd=capital_cost,
                annual_savings_usd=annual_savings,
                simple_payback_years=payback,
                npv_usd=npv
            ),
            annual_energy_savings_kwh=energy_savings_kwh,
            annual_emission_reduction_kgco2e=emission_reduction,
            lifetime_emission_reduction_kgco2e=lifetime_reduction
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.AGENT_ID}, version={self.AGENT_VERSION})"


# =============================================================================
# COMMON TECHNOLOGY SPECIFICATIONS
# =============================================================================

# Heat pump efficiency (COP)
HEAT_PUMP_COP = {
    "air_source": Decimal("3.5"),
    "ground_source": Decimal("4.5"),
    "water_source": Decimal("4.0"),
}

# LED lighting savings vs fluorescent
LED_SAVINGS_PERCENT = Decimal("50")

# Building envelope improvements
ENVELOPE_SAVINGS = {
    "window_upgrade": Decimal("15"),  # % energy savings
    "insulation_upgrade": Decimal("20"),
    "air_sealing": Decimal("10"),
    "cool_roof": Decimal("8"),
}

# Solar PV capacity factors by region
SOLAR_CAPACITY_FACTOR = {
    "excellent": Decimal("0.22"),  # Southwest US
    "good": Decimal("0.18"),       # Southeast US
    "moderate": Decimal("0.14"),   # Northeast US
    "poor": Decimal("0.10"),       # Pacific Northwest
}
