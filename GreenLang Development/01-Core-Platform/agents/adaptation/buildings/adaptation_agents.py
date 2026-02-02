# -*- coding: utf-8 -*-
"""
GreenLang Buildings Adaptation Agents
======================================

Climate adaptation agents for building sector resilience.
GL-ADAPT-BLD-001 through GL-ADAPT-BLD-008.

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

InputT = TypeVar("InputT", bound="AdaptationInput")
OutputT = TypeVar("OutputT", bound="AdaptationOutput")


# =============================================================================
# ENUMS
# =============================================================================

class ClimateHazard(str, Enum):
    """Climate hazard types."""
    EXTREME_HEAT = "extreme_heat"
    FLOODING = "flooding"
    SEA_LEVEL_RISE = "sea_level_rise"
    WIND_STORM = "wind_storm"
    WILDFIRE = "wildfire"
    DROUGHT = "drought"
    FREEZE = "freeze"
    HAIL = "hail"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TimeHorizon(str, Enum):
    """Climate scenario time horizons."""
    CURRENT = "current"
    NEAR_TERM = "2030"
    MID_TERM = "2050"
    LONG_TERM = "2100"


# =============================================================================
# DATA MODELS
# =============================================================================

class RiskAssessment(BaseModel):
    """Climate risk assessment result."""
    hazard: ClimateHazard
    current_risk: RiskLevel
    future_risk_2050: RiskLevel
    probability_percent: Decimal = Field(ge=0, le=100)
    potential_damage_usd: Optional[Decimal] = None
    annual_expected_loss_usd: Optional[Decimal] = None


class AdaptationMeasure(BaseModel):
    """Recommended adaptation measure."""
    measure_id: str
    name: str
    description: str
    hazard: ClimateHazard
    risk_reduction_percent: Decimal = Field(ge=0, le=100)
    capital_cost_usd: Decimal = Field(ge=0)
    annual_maintenance_usd: Decimal = Field(default=Decimal("0"), ge=0)
    implementation_years: int = Field(default=1, ge=1)
    effectiveness_years: int = Field(default=30, ge=1)


class BuildingVulnerability(BaseModel):
    """Building vulnerability profile."""
    building_id: str
    year_built: Optional[int] = None
    building_type: str
    construction_type: Optional[str] = None
    stories: Optional[int] = None
    basement: bool = Field(default=False)
    flat_roof: bool = Field(default=False)


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class AdaptationInput(BaseModel):
    """Base input for adaptation agents."""
    building_id: str
    building_type: str
    gross_floor_area_sqm: Decimal = Field(..., gt=0)
    location_lat: Optional[Decimal] = None
    location_lon: Optional[Decimal] = None
    postal_code: Optional[str] = None
    country_code: str = Field(default="US")

    vulnerability: Optional[BuildingVulnerability] = None
    time_horizon: TimeHorizon = Field(default=TimeHorizon.MID_TERM)
    climate_scenario: str = Field(default="RCP4.5")


class AdaptationOutput(BaseModel):
    """Base output for adaptation agents."""
    analysis_id: str
    agent_id: str
    agent_version: str
    timestamp: str
    building_id: str

    risk_assessments: List[RiskAssessment] = Field(default_factory=list)
    adaptation_measures: List[AdaptationMeasure] = Field(default_factory=list)

    overall_risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    total_potential_damage_usd: Decimal = Field(default=Decimal("0"))
    total_adaptation_cost_usd: Decimal = Field(default=Decimal("0"))
    benefit_cost_ratio: Optional[Decimal] = None

    provenance_hash: str = Field(default="")
    is_valid: bool = Field(default=True)
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# BASE AGENT
# =============================================================================

class BuildingAdaptationBaseAgent(ABC, Generic[InputT, OutputT]):
    """Base class for building adaptation agents."""

    AGENT_ID: str = "GL-ADAPT-BLD-BASE"
    AGENT_VERSION: str = "1.0.0"
    PRIMARY_HAZARD: Optional[ClimateHazard] = None

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def analyze(self, input_data: InputT) -> OutputT:
        pass

    def process(self, input_data: InputT) -> OutputT:
        start_time = datetime.now(timezone.utc)
        try:
            self.logger.info(f"{self.AGENT_ID} analyzing: building={input_data.building_id}")
            output = self.analyze(input_data)
            output.provenance_hash = self._calculate_hash({"input": input_data.model_dump()})
            return output
        except Exception as e:
            self.logger.error(f"{self.AGENT_ID} failed: {str(e)}", exc_info=True)
            raise

    def _round_financial(self, value: Decimal, precision: int = 2) -> Decimal:
        quantize_str = "0." + "0" * precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _generate_analysis_id(self, building_id: str) -> str:
        data = f"{self.AGENT_ID}:{building_id}:{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _get_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
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


# =============================================================================
# GL-ADAPT-BLD-001: Heat Resilience Agent
# =============================================================================

class HeatResilienceInput(AdaptationInput):
    """Input for heat resilience analysis."""
    has_central_cooling: bool = Field(default=True)
    cooling_capacity_kw: Optional[Decimal] = None
    has_cool_roof: bool = Field(default=False)
    has_shade_structures: bool = Field(default=False)


class HeatResilienceOutput(AdaptationOutput):
    """Output for heat resilience analysis."""
    heat_risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    cooling_adequacy_percent: Optional[Decimal] = None
    heat_island_effect: bool = Field(default=False)


class HeatResilienceAgent(BuildingAdaptationBaseAgent[HeatResilienceInput, HeatResilienceOutput]):
    """GL-ADAPT-BLD-001: Heat Resilience Agent."""

    AGENT_ID = "GL-ADAPT-BLD-001"
    AGENT_VERSION = "1.0.0"
    PRIMARY_HAZARD = ClimateHazard.EXTREME_HEAT

    def analyze(self, input_data: HeatResilienceInput) -> HeatResilienceOutput:
        measures = []

        # Assess heat risk based on cooling capacity
        risk_level = RiskLevel.LOW
        if not input_data.has_central_cooling:
            risk_level = RiskLevel.HIGH
            measures.append(AdaptationMeasure(
                measure_id="HEAT-001",
                name="Install Central Cooling",
                description="Install high-efficiency central air conditioning",
                hazard=ClimateHazard.EXTREME_HEAT,
                risk_reduction_percent=Decimal("60"),
                capital_cost_usd=input_data.gross_floor_area_sqm * Decimal("100")
            ))

        if not input_data.has_cool_roof:
            measures.append(AdaptationMeasure(
                measure_id="HEAT-002",
                name="Cool Roof Coating",
                description="Apply reflective roof coating",
                hazard=ClimateHazard.EXTREME_HEAT,
                risk_reduction_percent=Decimal("15"),
                capital_cost_usd=input_data.gross_floor_area_sqm * Decimal("15")
            ))

        total_cost = sum(m.capital_cost_usd for m in measures)

        return HeatResilienceOutput(
            analysis_id=self._generate_analysis_id(input_data.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=input_data.building_id,
            risk_assessments=[RiskAssessment(
                hazard=ClimateHazard.EXTREME_HEAT,
                current_risk=risk_level,
                future_risk_2050=RiskLevel.HIGH if risk_level != RiskLevel.LOW else RiskLevel.MODERATE,
                probability_percent=Decimal("40")
            )],
            adaptation_measures=measures,
            overall_risk_level=risk_level,
            total_adaptation_cost_usd=self._round_financial(total_cost),
            heat_risk_level=risk_level,
            is_valid=True
        )


# =============================================================================
# GL-ADAPT-BLD-002: Flood Resilience Agent
# =============================================================================

class FloodResilienceInput(AdaptationInput):
    """Input for flood resilience analysis."""
    flood_zone: Optional[str] = Field(None, description="FEMA flood zone")
    elevation_m: Optional[Decimal] = None
    has_flood_barriers: bool = Field(default=False)
    has_sump_pump: bool = Field(default=False)


class FloodResilienceOutput(AdaptationOutput):
    """Output for flood resilience analysis."""
    flood_risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    base_flood_elevation_m: Optional[Decimal] = None
    freeboard_m: Optional[Decimal] = None


class FloodResilienceAgent(BuildingAdaptationBaseAgent[FloodResilienceInput, FloodResilienceOutput]):
    """GL-ADAPT-BLD-002: Flood Resilience Agent."""

    AGENT_ID = "GL-ADAPT-BLD-002"
    AGENT_VERSION = "1.0.0"
    PRIMARY_HAZARD = ClimateHazard.FLOODING

    def analyze(self, input_data: FloodResilienceInput) -> FloodResilienceOutput:
        measures = []

        # Assess flood risk
        risk_level = RiskLevel.LOW
        if input_data.flood_zone in ["A", "AE", "V", "VE"]:
            risk_level = RiskLevel.HIGH
        elif input_data.flood_zone in ["X500", "B"]:
            risk_level = RiskLevel.MODERATE

        if risk_level in [RiskLevel.HIGH, RiskLevel.MODERATE] and not input_data.has_flood_barriers:
            measures.append(AdaptationMeasure(
                measure_id="FLOOD-001",
                name="Flood Barriers",
                description="Install deployable flood barriers",
                hazard=ClimateHazard.FLOODING,
                risk_reduction_percent=Decimal("40"),
                capital_cost_usd=Decimal("25000")
            ))

        if not input_data.has_sump_pump:
            measures.append(AdaptationMeasure(
                measure_id="FLOOD-002",
                name="Sump Pump System",
                description="Install backup sump pump with battery",
                hazard=ClimateHazard.FLOODING,
                risk_reduction_percent=Decimal("20"),
                capital_cost_usd=Decimal("5000")
            ))

        total_cost = sum(m.capital_cost_usd for m in measures)

        return FloodResilienceOutput(
            analysis_id=self._generate_analysis_id(input_data.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=input_data.building_id,
            risk_assessments=[RiskAssessment(
                hazard=ClimateHazard.FLOODING,
                current_risk=risk_level,
                future_risk_2050=risk_level,
                probability_percent=Decimal("20") if risk_level == RiskLevel.HIGH else Decimal("5")
            )],
            adaptation_measures=measures,
            overall_risk_level=risk_level,
            total_adaptation_cost_usd=self._round_financial(total_cost),
            flood_risk_level=risk_level,
            is_valid=True
        )


# =============================================================================
# GL-ADAPT-BLD-003: Wind Resilience Agent
# =============================================================================

class WindResilienceInput(AdaptationInput):
    """Input for wind resilience analysis."""
    wind_zone: Optional[str] = Field(None, description="ASCE 7 wind zone")
    roof_age_years: Optional[int] = None
    has_impact_windows: bool = Field(default=False)


class WindResilienceOutput(AdaptationOutput):
    """Output for wind resilience analysis."""
    wind_risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    design_wind_speed_mph: Optional[Decimal] = None


class WindResilienceAgent(BuildingAdaptationBaseAgent[WindResilienceInput, WindResilienceOutput]):
    """GL-ADAPT-BLD-003: Wind Resilience Agent."""

    AGENT_ID = "GL-ADAPT-BLD-003"
    AGENT_VERSION = "1.0.0"
    PRIMARY_HAZARD = ClimateHazard.WIND_STORM

    def analyze(self, input_data: WindResilienceInput) -> WindResilienceOutput:
        measures = []
        risk_level = RiskLevel.LOW

        # Assess wind risk
        if input_data.wind_zone in ["hurricane", "high"]:
            risk_level = RiskLevel.HIGH

        if not input_data.has_impact_windows and risk_level == RiskLevel.HIGH:
            measures.append(AdaptationMeasure(
                measure_id="WIND-001",
                name="Impact-Resistant Windows",
                description="Install hurricane-rated windows",
                hazard=ClimateHazard.WIND_STORM,
                risk_reduction_percent=Decimal("35"),
                capital_cost_usd=input_data.gross_floor_area_sqm * Decimal("50")
            ))

        total_cost = sum(m.capital_cost_usd for m in measures)

        return WindResilienceOutput(
            analysis_id=self._generate_analysis_id(input_data.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=input_data.building_id,
            risk_assessments=[RiskAssessment(
                hazard=ClimateHazard.WIND_STORM,
                current_risk=risk_level,
                future_risk_2050=risk_level,
                probability_percent=Decimal("15")
            )],
            adaptation_measures=measures,
            overall_risk_level=risk_level,
            total_adaptation_cost_usd=self._round_financial(total_cost),
            wind_risk_level=risk_level,
            is_valid=True
        )


# =============================================================================
# GL-ADAPT-BLD-004: Wildfire Resilience Agent
# =============================================================================

class WildfireResilienceInput(AdaptationInput):
    """Input for wildfire resilience analysis."""
    fire_hazard_severity_zone: Optional[str] = None
    defensible_space_m: Optional[Decimal] = None
    roof_class: Optional[str] = Field(None, description="A, B, or C rating")


class WildfireResilienceOutput(AdaptationOutput):
    """Output for wildfire resilience analysis."""
    wildfire_risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    ember_exposure_risk: bool = Field(default=False)


class WildfireResilienceAgent(BuildingAdaptationBaseAgent[WildfireResilienceInput, WildfireResilienceOutput]):
    """GL-ADAPT-BLD-004: Wildfire Resilience Agent."""

    AGENT_ID = "GL-ADAPT-BLD-004"
    AGENT_VERSION = "1.0.0"
    PRIMARY_HAZARD = ClimateHazard.WILDFIRE

    def analyze(self, input_data: WildfireResilienceInput) -> WildfireResilienceOutput:
        measures = []
        risk_level = RiskLevel.LOW

        if input_data.fire_hazard_severity_zone in ["very_high", "high"]:
            risk_level = RiskLevel.HIGH

        if input_data.roof_class != "A" and risk_level != RiskLevel.LOW:
            measures.append(AdaptationMeasure(
                measure_id="FIRE-001",
                name="Fire-Resistant Roofing",
                description="Install Class A fire-rated roof",
                hazard=ClimateHazard.WILDFIRE,
                risk_reduction_percent=Decimal("30"),
                capital_cost_usd=input_data.gross_floor_area_sqm * Decimal("80")
            ))

        total_cost = sum(m.capital_cost_usd for m in measures)

        return WildfireResilienceOutput(
            analysis_id=self._generate_analysis_id(input_data.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=input_data.building_id,
            risk_assessments=[RiskAssessment(
                hazard=ClimateHazard.WILDFIRE,
                current_risk=risk_level,
                future_risk_2050=RiskLevel.HIGH if risk_level == RiskLevel.MODERATE else risk_level,
                probability_percent=Decimal("10")
            )],
            adaptation_measures=measures,
            overall_risk_level=risk_level,
            total_adaptation_cost_usd=self._round_financial(total_cost),
            wildfire_risk_level=risk_level,
            is_valid=True
        )


# =============================================================================
# GL-ADAPT-BLD-005: Sea Level Rise Agent
# =============================================================================

class SeaLevelRiseInput(AdaptationInput):
    """Input for sea level rise analysis."""
    distance_to_coast_m: Optional[Decimal] = None
    current_elevation_m: Optional[Decimal] = None
    storm_surge_zone: bool = Field(default=False)


class SeaLevelRiseOutput(AdaptationOutput):
    """Output for sea level rise analysis."""
    slr_risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    years_until_at_risk: Optional[int] = None


class SeaLevelRiseAgent(BuildingAdaptationBaseAgent[SeaLevelRiseInput, SeaLevelRiseOutput]):
    """GL-ADAPT-BLD-005: Sea Level Rise Agent."""

    AGENT_ID = "GL-ADAPT-BLD-005"
    AGENT_VERSION = "1.0.0"
    PRIMARY_HAZARD = ClimateHazard.SEA_LEVEL_RISE

    def analyze(self, input_data: SeaLevelRiseInput) -> SeaLevelRiseOutput:
        measures = []
        risk_level = RiskLevel.LOW

        if input_data.distance_to_coast_m and input_data.distance_to_coast_m < 500:
            risk_level = RiskLevel.HIGH
        elif input_data.storm_surge_zone:
            risk_level = RiskLevel.MODERATE

        total_cost = sum(m.capital_cost_usd for m in measures)

        return SeaLevelRiseOutput(
            analysis_id=self._generate_analysis_id(input_data.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=input_data.building_id,
            risk_assessments=[RiskAssessment(
                hazard=ClimateHazard.SEA_LEVEL_RISE,
                current_risk=risk_level,
                future_risk_2050=RiskLevel.VERY_HIGH if risk_level == RiskLevel.HIGH else risk_level,
                probability_percent=Decimal("30") if risk_level == RiskLevel.HIGH else Decimal("5")
            )],
            adaptation_measures=measures,
            overall_risk_level=risk_level,
            total_adaptation_cost_usd=self._round_financial(total_cost),
            slr_risk_level=risk_level,
            is_valid=True
        )


# =============================================================================
# GL-ADAPT-BLD-006: Drought Resilience Agent
# =============================================================================

class DroughtResilienceInput(AdaptationInput):
    """Input for drought resilience analysis."""
    water_source: Optional[str] = None
    has_water_recycling: bool = Field(default=False)
    has_rainwater_harvesting: bool = Field(default=False)


class DroughtResilienceOutput(AdaptationOutput):
    """Output for drought resilience analysis."""
    drought_risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    water_security_score: Optional[int] = Field(None, ge=0, le=100)


class DroughtResilienceAgent(BuildingAdaptationBaseAgent[DroughtResilienceInput, DroughtResilienceOutput]):
    """GL-ADAPT-BLD-006: Drought Resilience Agent."""

    AGENT_ID = "GL-ADAPT-BLD-006"
    AGENT_VERSION = "1.0.0"
    PRIMARY_HAZARD = ClimateHazard.DROUGHT

    def analyze(self, input_data: DroughtResilienceInput) -> DroughtResilienceOutput:
        measures = []
        risk_level = RiskLevel.MODERATE

        if not input_data.has_water_recycling:
            measures.append(AdaptationMeasure(
                measure_id="DROUGHT-001",
                name="Greywater Recycling",
                description="Install greywater treatment and reuse system",
                hazard=ClimateHazard.DROUGHT,
                risk_reduction_percent=Decimal("25"),
                capital_cost_usd=Decimal("20000")
            ))

        total_cost = sum(m.capital_cost_usd for m in measures)

        return DroughtResilienceOutput(
            analysis_id=self._generate_analysis_id(input_data.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=input_data.building_id,
            adaptation_measures=measures,
            overall_risk_level=risk_level,
            total_adaptation_cost_usd=self._round_financial(total_cost),
            drought_risk_level=risk_level,
            is_valid=True
        )


# =============================================================================
# GL-ADAPT-BLD-007: Building Envelope Resilience Agent
# =============================================================================

class EnvelopeResilienceInput(AdaptationInput):
    """Input for envelope resilience analysis."""
    envelope_age_years: Optional[int] = None
    has_weather_resistant_barrier: bool = Field(default=False)


class EnvelopeResilienceOutput(AdaptationOutput):
    """Output for envelope resilience analysis."""
    envelope_condition: str = Field(default="unknown")


class EnvelopeResilienceAgent(BuildingAdaptationBaseAgent[EnvelopeResilienceInput, EnvelopeResilienceOutput]):
    """GL-ADAPT-BLD-007: Building Envelope Resilience Agent."""

    AGENT_ID = "GL-ADAPT-BLD-007"
    AGENT_VERSION = "1.0.0"

    def analyze(self, input_data: EnvelopeResilienceInput) -> EnvelopeResilienceOutput:
        return EnvelopeResilienceOutput(
            analysis_id=self._generate_analysis_id(input_data.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=input_data.building_id,
            is_valid=True
        )


# =============================================================================
# GL-ADAPT-BLD-008: Grid Resilience Agent
# =============================================================================

class GridResilienceInput(AdaptationInput):
    """Input for grid resilience analysis."""
    has_backup_generator: bool = Field(default=False)
    has_battery_storage: bool = Field(default=False)
    critical_load_kw: Optional[Decimal] = None


class GridResilienceOutput(AdaptationOutput):
    """Output for grid resilience analysis."""
    grid_risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    backup_duration_hours: Optional[Decimal] = None


class GridResilienceAgent(BuildingAdaptationBaseAgent[GridResilienceInput, GridResilienceOutput]):
    """GL-ADAPT-BLD-008: Grid Resilience Agent."""

    AGENT_ID = "GL-ADAPT-BLD-008"
    AGENT_VERSION = "1.0.0"

    def analyze(self, input_data: GridResilienceInput) -> GridResilienceOutput:
        measures = []
        risk_level = RiskLevel.MODERATE

        if not input_data.has_backup_generator and not input_data.has_battery_storage:
            risk_level = RiskLevel.HIGH
            measures.append(AdaptationMeasure(
                measure_id="GRID-001",
                name="Battery Backup System",
                description="Install battery storage for critical loads",
                hazard=ClimateHazard.EXTREME_HEAT,
                risk_reduction_percent=Decimal("50"),
                capital_cost_usd=Decimal("30000")
            ))

        total_cost = sum(m.capital_cost_usd for m in measures)

        return GridResilienceOutput(
            analysis_id=self._generate_analysis_id(input_data.building_id),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=input_data.building_id,
            adaptation_measures=measures,
            overall_risk_level=risk_level,
            total_adaptation_cost_usd=self._round_financial(total_cost),
            grid_risk_level=risk_level,
            is_valid=True
        )
