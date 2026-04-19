# -*- coding: utf-8 -*-
"""
Adaptation Energy Sector - Common Schemas and Data Models

This module defines Pydantic models for climate adaptation planning
in the energy sector.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ClimateHazard(str, Enum):
    """Climate hazards affecting energy infrastructure."""
    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"
    FLOODING = "flooding"
    DROUGHT = "drought"
    WILDFIRE = "wildfire"
    HURRICANE = "hurricane"
    SEA_LEVEL_RISE = "sea_level_rise"
    ICE_STORM = "ice_storm"
    STORM_SURGE = "storm_surge"


class ClimateScenario(str, Enum):
    """IPCC climate scenarios."""
    SSP1_26 = "ssp1_26"  # Low emissions
    SSP2_45 = "ssp2_45"  # Medium emissions
    SSP3_70 = "ssp3_70"  # High emissions
    SSP5_85 = "ssp5_85"  # Very high emissions


class InfrastructureType(str, Enum):
    """Energy infrastructure types."""
    POWER_PLANT = "power_plant"
    TRANSMISSION_LINE = "transmission_line"
    SUBSTATION = "substation"
    DISTRIBUTION_NETWORK = "distribution_network"
    SOLAR_FARM = "solar_farm"
    WIND_FARM = "wind_farm"
    HYDROPOWER = "hydropower"
    OFFSHORE_PLATFORM = "offshore_platform"
    LNG_TERMINAL = "lng_terminal"
    PIPELINE = "pipeline"


class VulnerabilityLevel(str, Enum):
    """Infrastructure vulnerability levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class AdaptationBaseInput(BaseModel):
    """Base input for adaptation planning agents."""

    organization_id: str = Field(...)
    region: str = Field(...)
    climate_scenario: ClimateScenario = Field(default=ClimateScenario.SSP2_45)
    planning_horizon_years: int = Field(default=30, ge=10, le=100)
    infrastructure_assets: List[Dict[str, Any]] = Field(default_factory=list)
    historical_events: List[Dict[str, Any]] = Field(default_factory=list)
    budget_million_usd: Optional[float] = Field(None, ge=0)


class AdaptationBaseOutput(BaseModel):
    """Base output for adaptation planning agents."""

    organization_id: str = Field(...)
    agent_id: str = Field(...)
    calculation_timestamp: datetime = Field(...)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)

    # Risk assessment
    hazards_assessed: List[str] = Field(...)
    vulnerability_level: str = Field(...)
    risk_score: float = Field(..., ge=0, le=100)

    # Adaptation measures
    recommended_measures: List[Dict[str, Any]] = Field(...)
    total_adaptation_cost_million_usd: float = Field(...)
    risk_reduction_pct: float = Field(..., ge=0, le=100)

    # Cost-benefit
    benefit_cost_ratio: float = Field(...)
    avoided_losses_million_usd: float = Field(...)

    confidence_level: float = Field(..., ge=0, le=1)
    key_uncertainties: List[str] = Field(default_factory=list)
