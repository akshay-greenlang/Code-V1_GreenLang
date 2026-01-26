# -*- coding: utf-8 -*-
"""
GreenLang Industrial Decarbonization Base Agent
================================================

Base class for all industrial sector decarbonization planning and tracking agents.
Provides common functionality for pathway modeling, technology assessment, and
decarbonization roadmap generation.

Features:
    - Decarbonization pathway modeling
    - Technology option assessment
    - Investment requirement analysis
    - Timeline and milestone tracking
    - Abatement cost calculations

Sources:
    - IEA Net Zero by 2050 Roadmap
    - Mission Possible Partnership Sector Transition Strategies
    - SBTi Sectoral Decarbonization Approach
    - McKinsey Decarbonization Pathways

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


InputT = TypeVar("InputT", bound="DecarbonizationInput")
OutputT = TypeVar("OutputT", bound="DecarbonizationOutput")


# =============================================================================
# ENUMS
# =============================================================================

class TechnologyReadiness(str, Enum):
    """Technology Readiness Level classification."""
    COMMERCIAL = "COMMERCIAL"  # TRL 9 - widely deployed
    EARLY_ADOPTION = "EARLY_ADOPTION"  # TRL 7-8 - first commercial
    DEMONSTRATION = "DEMONSTRATION"  # TRL 5-6 - pilot scale
    DEVELOPMENT = "DEVELOPMENT"  # TRL 3-4 - R&D
    CONCEPT = "CONCEPT"  # TRL 1-2 - basic research


class DecarbonizationLever(str, Enum):
    """Decarbonization levers for industrial sectors."""
    ENERGY_EFFICIENCY = "ENERGY_EFFICIENCY"
    FUEL_SWITCHING = "FUEL_SWITCHING"
    ELECTRIFICATION = "ELECTRIFICATION"
    HYDROGEN = "HYDROGEN"
    CCUS = "CCUS"  # Carbon Capture, Utilization, Storage
    PROCESS_CHANGE = "PROCESS_CHANGE"
    CIRCULAR_ECONOMY = "CIRCULAR_ECONOMY"
    RENEWABLE_ENERGY = "RENEWABLE_ENERGY"
    MATERIAL_EFFICIENCY = "MATERIAL_EFFICIENCY"
    DEMAND_REDUCTION = "DEMAND_REDUCTION"


class TimeHorizon(str, Enum):
    """Time horizons for decarbonization planning."""
    SHORT_TERM = "SHORT_TERM"  # 0-5 years
    MEDIUM_TERM = "MEDIUM_TERM"  # 5-15 years
    LONG_TERM = "LONG_TERM"  # 15-30 years


# =============================================================================
# DATA MODELS
# =============================================================================

class Technology(BaseModel):
    """Decarbonization technology option."""
    technology_id: str = Field(..., description="Unique technology identifier")
    name: str = Field(..., description="Technology name")
    description: str = Field(default="")
    lever: DecarbonizationLever
    readiness: TechnologyReadiness

    # Emissions impact
    abatement_potential_pct: Decimal = Field(ge=0, le=100)
    abatement_cost_usd_per_tco2: Decimal  # Can be negative (cost savings)

    # Investment
    capex_usd_per_annual_tonne: Optional[Decimal] = None
    opex_change_pct: Optional[Decimal] = None  # Change vs baseline

    # Timeline
    deployment_year_earliest: int = Field(ge=2024, le=2050)
    ramp_up_years: int = Field(default=3, ge=1, le=15)


class Milestone(BaseModel):
    """Decarbonization milestone."""
    year: int = Field(ge=2024, le=2100)
    target_reduction_pct: Decimal = Field(ge=0, le=100)
    target_intensity_tco2e_per_t: Optional[Decimal] = None
    description: str = Field(default="")
    technologies: List[str] = Field(default_factory=list)


class DecarbonizationPathway(BaseModel):
    """Complete decarbonization pathway."""
    pathway_id: str
    name: str
    baseline_emissions_tco2e: Decimal
    baseline_year: int = Field(default=2023)
    target_year: int = Field(default=2050)
    target_reduction_pct: Decimal = Field(ge=0, le=100)

    # Projected trajectory
    annual_trajectory: Dict[int, Decimal] = Field(default_factory=dict)  # Year -> tCO2e

    # Technologies and investments
    technologies: List[Technology] = Field(default_factory=list)
    milestones: List[Milestone] = Field(default_factory=list)

    # Financial
    total_capex_usd: Decimal = Field(default=Decimal("0"))
    average_abatement_cost_usd_per_tco2: Decimal = Field(default=Decimal("0"))

    # Provenance
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class DecarbonizationInput(BaseModel):
    """Base input for decarbonization agents."""
    facility_id: str = Field(..., description="Unique facility identifier")
    sector: str = Field(..., description="Industrial sector")

    # Baseline
    baseline_production_tonnes: Decimal = Field(gt=0)
    baseline_emissions_tco2e: Decimal = Field(gt=0)
    baseline_year: int = Field(default=2023, ge=2020, le=2030)

    # Targets
    target_year: int = Field(default=2050, ge=2030, le=2100)
    target_reduction_pct: Decimal = Field(default=Decimal("90"), ge=0, le=100)

    # Constraints
    budget_capex_usd: Optional[Decimal] = Field(None, ge=0)
    max_abatement_cost_usd_per_tco2: Optional[Decimal] = Field(None)

    # Preferences
    preferred_levers: List[DecarbonizationLever] = Field(default_factory=list)
    excluded_levers: List[DecarbonizationLever] = Field(default_factory=list)

    class Config:
        json_encoders = {Decimal: str}


class DecarbonizationOutput(BaseModel):
    """Base output for decarbonization agents."""
    # Identification
    calculation_id: str
    agent_id: str
    agent_version: str
    timestamp: str

    # Input summary
    facility_id: str
    sector: str
    baseline_emissions_tco2e: Decimal

    # Pathway
    recommended_pathway: Optional[DecarbonizationPathway] = None
    alternative_pathways: List[DecarbonizationPathway] = Field(default_factory=list)

    # Summary metrics
    total_abatement_tco2e: Decimal = Field(default=Decimal("0"))
    abatement_pct: Decimal = Field(default=Decimal("0"))
    total_capex_usd: Decimal = Field(default=Decimal("0"))
    levelized_abatement_cost_usd_per_tco2: Decimal = Field(default=Decimal("0"))

    # Timeline
    key_milestones: List[Milestone] = Field(default_factory=list)
    first_technology_deployment_year: int = Field(default=2025)

    # Provenance
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    provenance_hash: str = Field(default="")

    # Validation
    is_valid: bool = Field(default=True)
    warnings: List[str] = Field(default_factory=list)

    class Config:
        json_encoders = {Decimal: str}


# =============================================================================
# BASE AGENT
# =============================================================================

class IndustrialDecarbonizationBaseAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for industrial decarbonization agents.

    Each sector-specific agent implements technology options, abatement costs,
    and pathway generation logic specific to that industrial sector.
    """

    AGENT_ID: str = "GL-DECARB-IND-BASE"
    AGENT_VERSION: str = "1.0.0"
    SECTOR: str = "Industrial"

    PRECISION_EMISSIONS: int = 6
    PRECISION_COST: int = 2

    def __init__(self):
        """Initialize the decarbonization agent."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._technologies: List[Technology] = []
        self._initialize()

    def _initialize(self) -> None:
        """Initialize agent resources."""
        self._load_technologies()

    @abstractmethod
    def _load_technologies(self) -> None:
        """Load sector-specific decarbonization technologies."""
        pass

    @abstractmethod
    def generate_pathway(self, input_data: InputT) -> OutputT:
        """Generate decarbonization pathway for the sector."""
        pass

    def process(self, input_data: InputT) -> OutputT:
        """Main processing method."""
        start_time = datetime.now(timezone.utc)

        try:
            self.logger.info(
                f"{self.AGENT_ID} generating pathway: facility={input_data.facility_id}"
            )

            output = self.generate_pathway(input_data)

            # Calculate hashes
            output.input_hash = self._calculate_hash(input_data.model_dump())
            output.output_hash = self._calculate_hash({
                "total_abatement": str(output.total_abatement_tco2e),
                "total_capex": str(output.total_capex_usd)
            })
            output.provenance_hash = self._calculate_provenance_hash(
                output.input_hash, output.output_hash
            )

            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.info(f"{self.AGENT_ID} completed in {duration_ms:.2f}ms")

            return output

        except Exception as e:
            self.logger.error(f"{self.AGENT_ID} failed: {str(e)}", exc_info=True)
            raise

    def _generate_calculation_id(self, facility_id: str) -> str:
        """Generate unique calculation ID."""
        data = f"{self.AGENT_ID}:{facility_id}:{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _get_timestamp(self) -> str:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _round_emissions(self, value: Decimal) -> Decimal:
        """Round emission values."""
        return value.quantize(
            Decimal("0." + "0" * self.PRECISION_EMISSIONS),
            rounding=ROUND_HALF_UP
        )

    def _round_cost(self, value: Decimal) -> Decimal:
        """Round cost values."""
        return value.quantize(
            Decimal("0." + "0" * self.PRECISION_COST),
            rounding=ROUND_HALF_UP
        )

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

    def _calculate_provenance_hash(self, input_hash: str, output_hash: str) -> str:
        """Calculate provenance hash."""
        data = {
            "agent_id": self.AGENT_ID,
            "version": self.AGENT_VERSION,
            "input_hash": input_hash,
            "output_hash": output_hash
        }
        return self._calculate_hash(data)

    def _calculate_trajectory(
        self,
        baseline_emissions: Decimal,
        baseline_year: int,
        target_year: int,
        technologies: List[Technology]
    ) -> Dict[int, Decimal]:
        """Calculate annual emissions trajectory."""
        trajectory = {}
        current_emissions = baseline_emissions

        for year in range(baseline_year, target_year + 1):
            # Apply technologies that are active this year
            year_reduction = Decimal("0")
            for tech in technologies:
                if tech.deployment_year_earliest <= year:
                    # Ramp up factor
                    years_since_deploy = year - tech.deployment_year_earliest
                    ramp_factor = min(
                        Decimal(str(years_since_deploy)) / Decimal(str(tech.ramp_up_years)),
                        Decimal("1")
                    )
                    year_reduction += (
                        tech.abatement_potential_pct / Decimal("100") * ramp_factor
                    )

            # Cap reduction at 100%
            year_reduction = min(year_reduction, Decimal("1"))
            trajectory[year] = self._round_emissions(
                baseline_emissions * (Decimal("1") - year_reduction)
            )

        return trajectory

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.AGENT_ID})"
