# -*- coding: utf-8 -*-
"""
Agriculture MRV Base Module
===========================

This module provides base classes and common functionality for all
agriculture MRV (Monitoring, Reporting, Verification) agents.

Design Principles:
- Zero-hallucination guarantee (no LLM in calculation path)
- Full audit trail with SHA-256 provenance hashing
- IPCC 2006/2019 Guidelines compliant
- GHG Protocol Agricultural Guidance aligned
- Pydantic models for type safety

Reference Standards:
- IPCC 2006 Guidelines for National GHG Inventories, Volume 4 (Agriculture)
- IPCC 2019 Refinement to 2006 Guidelines
- GHG Protocol Agricultural Guidance
- FAO Guidelines for GHG Emissions from Agriculture
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class AgricultureSector(str, Enum):
    """Agriculture sub-sectors for emissions classification."""
    CROP_PRODUCTION = "crop_production"
    LIVESTOCK = "livestock"
    FERTILIZER = "fertilizer"
    LAND_USE_CHANGE = "land_use_change"
    RICE_CULTIVATION = "rice_cultivation"
    AGRICULTURAL_MACHINERY = "agricultural_machinery"
    IRRIGATION = "irrigation"
    FOOD_PROCESSING = "food_processing"


class EmissionScope(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class DataQualityTier(str, Enum):
    """Data quality tiers per IPCC/GHG Protocol."""
    TIER_1 = "tier_1"  # Default factors
    TIER_2 = "tier_2"  # Country-specific factors
    TIER_3 = "tier_3"  # Site-specific measurement


class EmissionFactorSource(str, Enum):
    """Sources for emission factors."""
    IPCC_2006 = "ipcc_2006"
    IPCC_2019 = "ipcc_2019"
    FAO = "fao"
    DEFRA = "defra"
    EPA = "epa"
    NATIONAL = "national"
    SITE_SPECIFIC = "site_specific"


class ClimateZone(str, Enum):
    """Climate zones for emission factor selection."""
    TROPICAL_WET = "tropical_wet"
    TROPICAL_MOIST = "tropical_moist"
    TROPICAL_DRY = "tropical_dry"
    WARM_TEMPERATE_MOIST = "warm_temperate_moist"
    WARM_TEMPERATE_DRY = "warm_temperate_dry"
    COOL_TEMPERATE_MOIST = "cool_temperate_moist"
    COOL_TEMPERATE_DRY = "cool_temperate_dry"
    POLAR_MOIST = "polar_moist"
    POLAR_DRY = "polar_dry"
    BOREAL_MOIST = "boreal_moist"
    BOREAL_DRY = "boreal_dry"


class SoilType(str, Enum):
    """Soil types for emission calculations."""
    HIGH_ACTIVITY_CLAY = "high_activity_clay"
    LOW_ACTIVITY_CLAY = "low_activity_clay"
    SANDY = "sandy"
    SPODIC = "spodic"
    VOLCANIC = "volcanic"
    WETLAND = "wetland"
    ORGANIC = "organic"


# =============================================================================
# Emission Factor Models
# =============================================================================

class EmissionFactor(BaseModel):
    """Emission factor record with provenance."""

    factor_id: str = Field(..., description="Unique factor identifier")
    factor_value: Decimal = Field(..., ge=0, description="Emission factor value")
    factor_unit: str = Field(..., description="Factor unit")
    source: EmissionFactorSource = Field(..., description="Factor source")
    source_uri: str = Field("", description="URI to source documentation")
    version: str = Field(..., description="Factor version/year")
    last_updated: str = Field(..., description="Last update date")
    uncertainty_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Uncertainty percentage"
    )
    data_quality_tier: DataQualityTier = Field(
        DataQualityTier.TIER_1, description="Data quality tier"
    )
    geographic_scope: str = Field("global", description="Geographic applicability")
    climate_zone: Optional[ClimateZone] = Field(None, description="Climate zone")

    class Config:
        use_enum_values = True


# =============================================================================
# Global Warming Potentials (IPCC AR5 100-year)
# =============================================================================

GWP_AR5 = {
    "CO2": Decimal("1"),
    "CH4": Decimal("28"),  # IPCC AR5
    "N2O": Decimal("265"),  # IPCC AR5
}

GWP_AR6 = {
    "CO2": Decimal("1"),
    "CH4": Decimal("27.9"),  # IPCC AR6
    "N2O": Decimal("273"),  # IPCC AR6
}


# =============================================================================
# Calculation Step Model
# =============================================================================

class CalculationStep(BaseModel):
    """Single step in the calculation audit trail."""

    step_number: int = Field(..., ge=1, description="Step sequence number")
    description: str = Field(..., description="Step description")
    formula: Optional[str] = Field(None, description="Formula applied")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values")
    output: Optional[str] = Field(None, description="Output value")
    emission_factor: Optional[EmissionFactor] = Field(
        None, description="Emission factor used"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Step timestamp"
    )


# =============================================================================
# Base Input/Output Models
# =============================================================================

class AgricultureMRVInput(BaseModel):
    """Base input model for all agriculture MRV agents."""

    # Identification
    organization_id: str = Field(..., description="Organization identifier")
    facility_id: Optional[str] = Field(None, description="Farm/facility identifier")
    request_id: Optional[str] = Field(None, description="Unique request ID")

    # Reporting period
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting year")
    reporting_period_start: Optional[datetime] = Field(
        None, description="Period start date"
    )
    reporting_period_end: Optional[datetime] = Field(
        None, description="Period end date"
    )

    # Geographic context
    region: str = Field("global", description="Geographic region")
    country: Optional[str] = Field(None, description="Country code (ISO 3166-1)")
    climate_zone: ClimateZone = Field(
        ClimateZone.WARM_TEMPERATE_MOIST, description="Climate zone"
    )

    # Soil context
    soil_type: SoilType = Field(
        SoilType.HIGH_ACTIVITY_CLAY, description="Predominant soil type"
    )

    class Config:
        use_enum_values = True

    @validator("reporting_year")
    def validate_reporting_year(cls, v: int) -> int:
        """Validate reporting year is reasonable."""
        current_year = datetime.now().year
        if v > current_year + 1:
            raise ValueError(f"Reporting year {v} is too far in the future")
        return v


class AgricultureMRVOutput(BaseModel):
    """Base output model for all agriculture MRV agents."""

    # Agent identification
    agent_id: str = Field(..., description="Agent identifier")
    agent_version: str = Field("1.0.0", description="Agent version")

    # Emission results (kg)
    total_emissions_kg_co2e: Decimal = Field(
        ..., ge=0, description="Total emissions in kg CO2e"
    )
    total_emissions_mt_co2e: Decimal = Field(
        ..., ge=0, description="Total emissions in metric tons CO2e"
    )

    # Gas breakdown (kg)
    co2_kg: Decimal = Field(Decimal("0"), ge=0, description="CO2 in kg")
    ch4_kg: Decimal = Field(Decimal("0"), ge=0, description="CH4 in kg")
    n2o_kg: Decimal = Field(Decimal("0"), ge=0, description="N2O in kg")

    # CO2e breakdown
    co2_co2e_kg: Decimal = Field(Decimal("0"), ge=0, description="CO2 as CO2e")
    ch4_co2e_kg: Decimal = Field(Decimal("0"), ge=0, description="CH4 as CO2e")
    n2o_co2e_kg: Decimal = Field(Decimal("0"), ge=0, description="N2O as CO2e")

    # Scope classification
    scope: EmissionScope = Field(..., description="GHG Protocol scope")
    sector: AgricultureSector = Field(..., description="Agriculture sector")

    # Audit trail
    calculation_steps: List[CalculationStep] = Field(
        default_factory=list, description="Calculation audit trail"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    # Data quality
    data_quality_tier: DataQualityTier = Field(
        DataQualityTier.TIER_1, description="Overall data quality"
    )
    uncertainty_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Uncertainty percentage"
    )

    # Timestamps
    calculation_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Calculation timestamp"
    )
    calculation_duration_ms: float = Field(
        0.0, ge=0, description="Calculation duration in ms"
    )

    # Status
    status: str = Field("success", description="Calculation status")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: List[str] = Field(default_factory=list, description="Error messages")

    # Metadata
    emission_factors_used: List[EmissionFactor] = Field(
        default_factory=list, description="Emission factors applied"
    )
    activity_data_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary of activity data"
    )

    # GWP version used
    gwp_version: str = Field("AR5", description="GWP version used")

    class Config:
        use_enum_values = True
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


# =============================================================================
# Base Agriculture MRV Agent
# =============================================================================

class BaseAgricultureMRVAgent(ABC):
    """
    Abstract base class for agriculture MRV agents.

    All agriculture MRV agents inherit from this class and implement
    the calculate() method with sector-specific logic.

    Key Guarantees:
    - ZERO HALLUCINATION: No LLM calls in calculation path
    - DETERMINISTIC: Same input always produces same output
    - AUDITABLE: Complete SHA-256 provenance tracking
    - IPCC COMPLIANT: 2006/2019 Guidelines aligned
    """

    # Class attributes to be overridden by subclasses
    AGENT_ID: str = "GL-MRV-AGR-000"
    AGENT_NAME: str = "Base Agriculture MRV Agent"
    AGENT_VERSION: str = "1.0.0"
    SECTOR: AgricultureSector = AgricultureSector.CROP_PRODUCTION
    DEFAULT_SCOPE: EmissionScope = EmissionScope.SCOPE_1

    def __init__(self, gwp_version: str = "AR5"):
        """
        Initialize the agriculture MRV agent.

        Args:
            gwp_version: GWP version to use (AR5 or AR6)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.gwp_version = gwp_version
        self.gwp = GWP_AR5 if gwp_version == "AR5" else GWP_AR6
        self.logger.info(f"Initialized {self.AGENT_ID} v{self.AGENT_VERSION}")

    @abstractmethod
    def calculate(self, input_data: AgricultureMRVInput) -> AgricultureMRVOutput:
        """
        Execute the emission calculation.

        Args:
            input_data: Agriculture-specific input data

        Returns:
            Complete calculation result with audit trail

        Raises:
            ValueError: If input validation fails
            CalculationError: If calculation fails
        """
        pass

    def _convert_to_co2e(
        self,
        co2_kg: Decimal,
        ch4_kg: Decimal,
        n2o_kg: Decimal,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        """
        Convert individual gases to CO2 equivalent.

        Args:
            co2_kg: CO2 in kg
            ch4_kg: CH4 in kg
            n2o_kg: N2O in kg

        Returns:
            Tuple of (total_co2e, co2_co2e, ch4_co2e, n2o_co2e)
        """
        co2_co2e = (co2_kg * self.gwp["CO2"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        ch4_co2e = (ch4_kg * self.gwp["CH4"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        n2o_co2e = (n2o_kg * self.gwp["N2O"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        total_co2e = co2_co2e + ch4_co2e + n2o_co2e

        return total_co2e, co2_co2e, ch4_co2e, n2o_co2e

    def _kg_to_metric_tons(self, kg: Decimal) -> Decimal:
        """Convert kg to metric tons."""
        return (kg / Decimal("1000")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

    def _generate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        steps: List[CalculationStep],
    ) -> str:
        """
        Generate SHA-256 provenance hash for audit trail.

        Args:
            input_data: Input data dictionary
            output_data: Output data dictionary
            steps: Calculation steps

        Returns:
            SHA-256 hex digest
        """
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "input": input_data,
            "output": {
                k: str(v) if isinstance(v, Decimal) else v
                for k, v in output_data.items()
                if k not in ["provenance_hash", "calculation_timestamp"]
            },
            "steps": [
                {
                    "step_number": s.step_number,
                    "description": s.description,
                    "formula": s.formula,
                    "inputs": s.inputs,
                    "output": s.output,
                }
                for s in steps
            ],
        }
        data_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _create_output(
        self,
        co2_kg: Decimal,
        ch4_kg: Decimal,
        n2o_kg: Decimal,
        steps: List[CalculationStep],
        emission_factors: List[EmissionFactor],
        activity_summary: Dict[str, Any],
        start_time: datetime,
        scope: Optional[EmissionScope] = None,
        warnings: Optional[List[str]] = None,
    ) -> AgricultureMRVOutput:
        """
        Create standardized output with provenance.

        Args:
            co2_kg: CO2 in kg
            ch4_kg: CH4 in kg
            n2o_kg: N2O in kg
            steps: Calculation steps
            emission_factors: Factors used
            activity_summary: Summary of activity data
            start_time: Calculation start time
            scope: Emission scope (optional, uses default)
            warnings: Warning messages

        Returns:
            Complete AgricultureMRVOutput
        """
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Convert to CO2e
        total_co2e, co2_co2e, ch4_co2e, n2o_co2e = self._convert_to_co2e(
            co2_kg, ch4_kg, n2o_kg
        )

        output_data = {
            "total_emissions_kg_co2e": total_co2e,
            "co2_kg": co2_kg,
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
            "co2_co2e_kg": co2_co2e,
            "ch4_co2e_kg": ch4_co2e,
            "n2o_co2e_kg": n2o_co2e,
        }

        provenance_hash = self._generate_provenance_hash(
            input_data=activity_summary,
            output_data=output_data,
            steps=steps,
        )

        # Determine overall data quality
        if emission_factors:
            quality_tiers = [ef.data_quality_tier for ef in emission_factors]
            if DataQualityTier.TIER_1 in quality_tiers:
                overall_quality = DataQualityTier.TIER_1
            elif DataQualityTier.TIER_2 in quality_tiers:
                overall_quality = DataQualityTier.TIER_2
            else:
                overall_quality = DataQualityTier.TIER_3
        else:
            overall_quality = DataQualityTier.TIER_1

        return AgricultureMRVOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            total_emissions_kg_co2e=total_co2e,
            total_emissions_mt_co2e=self._kg_to_metric_tons(total_co2e),
            co2_kg=co2_kg,
            ch4_kg=ch4_kg,
            n2o_kg=n2o_kg,
            co2_co2e_kg=co2_co2e,
            ch4_co2e_kg=ch4_co2e,
            n2o_co2e_kg=n2o_co2e,
            scope=scope or self.DEFAULT_SCOPE,
            sector=self.SECTOR,
            calculation_steps=steps,
            provenance_hash=provenance_hash,
            data_quality_tier=overall_quality,
            calculation_timestamp=end_time,
            calculation_duration_ms=duration_ms,
            emission_factors_used=emission_factors,
            activity_data_summary=activity_summary,
            gwp_version=self.gwp_version,
            warnings=warnings or [],
        )
