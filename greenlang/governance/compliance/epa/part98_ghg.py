# -*- coding: utf-8 -*-
"""
EPA Part 98 GHG Reporting - Subpart C (General Stationary Fuel Combustion)

This module implements EPA 40 CFR Part 98 Subpart C requirements for calculating
and reporting greenhouse gas emissions from stationary fuel combustion sources.

References:
    - 40 CFR Part 98 Subpart C - General Stationary Fuel Combustion
    - EPA Greenhouse Gas Reporting Program (GHGRP) Guidelines
    - Table C-1: CO2 Emission Factors by Fuel Type
    - Table C-2: CH4 and N2O Emission Factors

Zero-hallucination principle: All calculations use deterministic formulas,
EPA-published emission factors, and material-specific data. No LLM usage
in calculation path.

Example:
    >>> config = Part98Config(facility_id="FAC123", epa_ghgrp_id="123456789")
    >>> reporter = Part98Reporter(config)
    >>> fuel_data = FuelCombustionData(fuel_type="Natural Gas", heat_input_mmbtu=5000)
    >>> result = reporter.calculate_subpart_c(fuel_data)
    >>> assert result.total_co2_metric_tons > 0
    >>> assert result.validation_status == "PASS"
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging
import json
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)


class FuelType(str, Enum):
    """EPA-approved fuel types for Subpart C reporting."""
    NATURAL_GAS = "natural_gas"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUBBITUMINOUS = "coal_subbituminous"
    COAL_LIGNITE = "coal_lignite"
    COAL_ANTHRACITE = "coal_anthracite"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    PROPANE = "propane"
    KEROSENE = "kerosene"
    GASOLINE = "gasoline"
    DIESEL = "diesel"
    BIOMASS = "biomass"
    LANDFILL_GAS = "landfill_gas"
    COAL_COKE = "coal_coke"


class TierLevel(str, Enum):
    """Calculation methodologies per 40 CFR 98.33."""
    TIER1 = "tier1"  # Default emission factors
    TIER2 = "tier2"  # Fuel-specific higher heating value + carbon content
    TIER3 = "tier3"  # Continuous emissions monitoring systems (CEMS)


class CO2_EMISSION_FACTORS(BaseModel):
    """
    EPA Table C-1: CO2 Emission Factors (kg CO2/MMBtu)

    These are default factors for Tier 1 calculations.
    Source: 40 CFR Part 98, Table C-1
    """
    natural_gas: float = 53.06
    coal_bituminous: float = 93.69
    coal_subbituminous: float = 96.86
    coal_lignite: float = 97.41
    coal_anthracite: float = 98.32
    fuel_oil_no2: float = 73.96
    fuel_oil_no6: float = 77.59
    propane: float = 62.10
    kerosene: float = 75.13
    gasoline: float = 69.24
    diesel: float = 74.58
    biomass: float = 96.00  # Varies, approximation
    landfill_gas: float = 53.06
    coal_coke: float = 101.33

    class Config:
        frozen = True

    @classmethod
    def get_factor(cls, fuel_type: FuelType) -> float:
        """Retrieve CO2 emission factor for fuel type."""
        factors = cls()
        return getattr(factors, fuel_type.value)


class CH4_N2O_FACTORS(BaseModel):
    """
    EPA Table C-2: CH4 and N2O Emission Factors (kg/MMBtu)

    Source: 40 CFR Part 98, Table C-2
    """
    # CH4 emission factors
    ch4_natural_gas: float = 0.0022
    ch4_coal: float = 0.0005
    ch4_oil: float = 0.0010
    ch4_biomass: float = 0.0021

    # N2O emission factors
    n2o_natural_gas: float = 0.0001
    n2o_coal: float = 0.0001
    n2o_oil: float = 0.0005
    n2o_biomass: float = 0.0006

    class Config:
        frozen = True

    @classmethod
    def get_ch4_factor(cls, fuel_type: FuelType) -> float:
        """Get CH4 emission factor (kg/MMBtu)."""
        factors = cls()
        if fuel_type in [FuelType.NATURAL_GAS, FuelType.LANDFILL_GAS]:
            return factors.ch4_natural_gas
        elif "coal" in fuel_type.value:
            return factors.ch4_coal
        elif "oil" in fuel_type.value or fuel_type in [FuelType.DIESEL, FuelType.GASOLINE]:
            return factors.ch4_oil
        elif fuel_type == FuelType.BIOMASS:
            return factors.ch4_biomass
        else:
            return 0.0010  # Default

    @classmethod
    def get_n2o_factor(cls, fuel_type: FuelType) -> float:
        """Get N2O emission factor (kg/MMBtu)."""
        factors = cls()
        if fuel_type in [FuelType.NATURAL_GAS, FuelType.LANDFILL_GAS]:
            return factors.n2o_natural_gas
        elif "coal" in fuel_type.value:
            return factors.n2o_coal
        elif "oil" in fuel_type.value or fuel_type in [FuelType.DIESEL, FuelType.GASOLINE]:
            return factors.n2o_oil
        elif fuel_type == FuelType.BIOMASS:
            return factors.n2o_biomass
        else:
            return 0.0005  # Default


class FuelCombustionData(BaseModel):
    """Input data for Subpart C fuel combustion calculations."""

    fuel_type: FuelType = Field(..., description="Type of fuel combusted")
    heat_input_mmbtu: float = Field(..., ge=0, description="Heat input in MMBtu")

    # Tier 2 and Tier 3 parameters
    fuel_quantity: Optional[float] = Field(None, ge=0, description="Quantity of fuel consumed")
    fuel_unit: Optional[str] = Field("kg", description="Unit of fuel quantity (kg, lbs, gal, scf)")
    higher_heating_value: Optional[float] = Field(None, ge=0, description="HHV (BTU/kg or BTU/lb)")
    carbon_content: Optional[float] = Field(None, ge=0, le=100, description="Carbon content as % by weight")

    # Process identification
    facility_id: str = Field(..., description="EPA GHGRP Facility ID")
    process_id: Optional[str] = Field(None, description="Source category ID")
    reporting_year: int = Field(..., description="Calendar year of reporting")

    # Optional metadata
    equipment_type: Optional[str] = Field(None, description="Type of equipment (boiler, turbine, etc.)")
    is_co_fired: bool = Field(default=False, description="Is this a co-fired process?")
    co_fired_fuels: Optional[List[str]] = Field(None, description="Other fuels in co-fired process")

    @validator('heat_input_mmbtu')
    def validate_heat_input(cls, v):
        """Validate heat input is reasonable."""
        if v > 100000000:  # > 100M MMBtu is unreasonable
            raise ValueError("Heat input exceeds reasonable range")
        return v

    @validator('carbon_content')
    def validate_carbon_content(cls, v):
        """Validate carbon content percentage."""
        if v is not None and (v < 0 or v > 100):
            raise ValueError("Carbon content must be 0-100%")
        return v


class CO2Calculation(BaseModel):
    """CO2 emissions calculation result."""

    fuel_type: FuelType
    calculation_tier: TierLevel
    heat_input_mmbtu: float = Field(..., description="Heat input in MMBtu")
    co2_kg: float = Field(..., description="CO2 in kilograms")
    co2_metric_tons: float = Field(..., description="CO2 in metric tons")
    emission_factor_used: float = Field(..., description="Emission factor (kg CO2/MMBtu)")

    # Tier 2+ specifics
    fuel_quantity: Optional[float] = None
    fuel_unit: Optional[str] = None
    higher_heating_value: Optional[float] = None
    carbon_content: Optional[float] = None


class CH4N2OCalculation(BaseModel):
    """CH4 and N2O emissions calculation result."""

    fuel_type: FuelType
    heat_input_mmbtu: float
    ch4_kg: float = Field(..., description="CH4 in kilograms")
    ch4_metric_tons: float = Field(..., description="CH4 in metric tons")
    n2o_kg: float = Field(..., description="N2O in kilograms")
    n2o_metric_tons: float = Field(..., description="N2O in metric tons")
    ch4_factor: float = Field(..., description="CH4 factor (kg/MMBtu)")
    n2o_factor: float = Field(..., description="N2O factor (kg/MMBtu)")


class SubpartCResult(BaseModel):
    """Complete Subpart C reporting result for a fuel combustion source."""

    facility_id: str
    process_id: Optional[str] = None
    reporting_year: int
    fuel_type: FuelType

    # CO2 calculations
    co2_calculation: CO2Calculation

    # CH4 and N2O calculations
    ch4n2o_calculation: CH4N2OCalculation

    # Total CO2e (GWP: CO2=1, CH4=28, N2O=265 over 100 years)
    total_co2_metric_tons: float = Field(..., description="Total CO2 emissions (metric tons)")
    total_ch4_metric_tons: float
    total_n2o_metric_tons: float
    total_co2e_metric_tons: float = Field(..., description="Total CO2e using AR5 GWP")

    # Regulatory thresholds
    exceeds_threshold: bool = Field(..., description="Exceeds 25,000 MT CO2e threshold")
    requires_reporting: bool = Field(..., description="Facility must report to EPA GHGRP")

    # Validation and provenance
    validation_status: str = Field(..., description="PASS or FAIL")
    validation_errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., description="Processing duration")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Part98Config(BaseModel):
    """Configuration for Part 98 reporter."""

    facility_id: str = Field(..., description="EPA GHGRP Facility ID")
    epa_ghgrp_id: Optional[str] = Field(None, description="GHGRP submission ID")
    facility_name: Optional[str] = Field(None)
    facility_address: Optional[str] = Field(None)
    threshold_mtco2e: float = Field(default=25000.0, description="GHGRP reporting threshold")
    use_gwp_ar5: bool = Field(default=True, description="Use AR5 GWP values")
    enable_tier2_3: bool = Field(default=False, description="Allow Tier 2 & 3 calculations")
    enable_validation: bool = Field(default=True, description="Enable data validation")
    enable_provenance: bool = Field(default=True, description="Track provenance hashes")


class Part98Reporter:
    """
    EPA Part 98 Subpart C GHG Reporter.

    Calculates greenhouse gas emissions from stationary fuel combustion
    per 40 CFR Part 98 requirements. Implements three calculation tiers
    with increasing precision requirements.

    Attributes:
        config: Reporter configuration
        co2_factors: CO2 emission factors (Table C-1)
        ch4_n2o_factors: CH4/N2O emission factors (Table C-2)
    """

    # Global warming potential values (AR5, 100-year)
    GWP_CH4_AR5 = 28
    GWP_N2O_AR5 = 265
    GWP_CH4_AR4 = 25
    GWP_N2O_AR4 = 298

    def __init__(self, config: Part98Config):
        """Initialize Part 98 reporter."""
        self.config = config
        self.co2_factors = CO2_EMISSION_FACTORS()
        self.ch4_n2o_factors = CH4_N2O_FACTORS()
        logger.info(f"Part98Reporter initialized for facility {config.facility_id}")

    def calculate_subpart_c(
        self,
        fuel_data: FuelCombustionData,
        tier: Optional[TierLevel] = None
    ) -> SubpartCResult:
        """
        Calculate Subpart C emissions from fuel combustion.

        Args:
            fuel_data: Fuel combustion input data
            tier: Calculation tier (auto-selected if not specified)

        Returns:
            Complete Subpart C calculation result

        Raises:
            ValueError: If input validation fails
        """
        from datetime import datetime
        start_time = datetime.utcnow()

        try:
            # Step 1: Determine calculation tier
            if tier is None:
                tier = self._select_tier(fuel_data)

            # Step 2: Validate input data
            validation_errors = self._validate_fuel_data(fuel_data, tier)
            if validation_errors and self.config.enable_validation:
                raise ValueError(f"Input validation failed: {validation_errors}")

            # Step 3: Calculate CO2 emissions
            co2_calc = self._calculate_co2(fuel_data, tier)

            # Step 4: Calculate CH4 and N2O emissions
            ch4n2o_calc = self._calculate_ch4_n2o(fuel_data)

            # Step 5: Calculate CO2e and check thresholds
            gwp_ch4 = self.GWP_CH4_AR5 if self.config.use_gwp_ar5 else self.GWP_CH4_AR4
            gwp_n2o = self.GWP_N2O_AR5 if self.config.use_gwp_ar5 else self.GWP_N2O_AR4

            total_co2e = (
                co2_calc.co2_metric_tons +
                (ch4n2o_calc.ch4_metric_tons * gwp_ch4) +
                (ch4n2o_calc.n2o_metric_tons * gwp_n2o)
            )

            exceeds_threshold = total_co2e >= self.config.threshold_mtco2e

            # Step 6: Generate output
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            result = SubpartCResult(
                facility_id=fuel_data.facility_id,
                process_id=fuel_data.process_id,
                reporting_year=fuel_data.reporting_year,
                fuel_type=fuel_data.fuel_type,
                co2_calculation=co2_calc,
                ch4n2o_calculation=ch4n2o_calc,
                total_co2_metric_tons=co2_calc.co2_metric_tons,
                total_ch4_metric_tons=ch4n2o_calc.ch4_metric_tons,
                total_n2o_metric_tons=ch4n2o_calc.n2o_metric_tons,
                total_co2e_metric_tons=total_co2e,
                exceeds_threshold=exceeds_threshold,
                requires_reporting=exceeds_threshold,
                validation_status="PASS" if not validation_errors else "FAIL",
                validation_errors=validation_errors,
                provenance_hash=self._calculate_provenance(fuel_data, co2_calc) if self.config.enable_provenance else "",
                processing_time_ms=processing_time
            )

            logger.info(
                f"Subpart C calculation complete: {fuel_data.facility_id} "
                f"CO2e={total_co2e:.2f} MT, Reporting={exceeds_threshold}"
            )

            return result

        except Exception as e:
            logger.error(f"Subpart C calculation failed: {str(e)}", exc_info=True)
            raise

    def calculate_co2_tier1(
        self,
        fuel_quantity: float,
        emission_factor: float
    ) -> CO2Calculation:
        """
        Tier 1: CO2 = Fuel Heat Input (MMBtu) x Default Emission Factor

        This is the simplest method using EPA Table C-1 default factors.

        Args:
            fuel_quantity: Heat input in MMBtu
            emission_factor: CO2 emission factor in kg CO2/MMBtu

        Returns:
            CO2 calculation result
        """
        co2_kg = fuel_quantity * emission_factor
        co2_metric_tons = co2_kg / 1000.0

        return CO2Calculation(
            fuel_type=FuelType.NATURAL_GAS,  # Placeholder
            calculation_tier=TierLevel.TIER1,
            heat_input_mmbtu=fuel_quantity,
            co2_kg=co2_kg,
            co2_metric_tons=co2_metric_tons,
            emission_factor_used=emission_factor
        )

    def calculate_co2_tier2(
        self,
        fuel_quantity: float,
        higher_heating_value: float,
        carbon_content: float,
        fuel_type: FuelType
    ) -> CO2Calculation:
        """
        Tier 2: CO2 = Fuel Quantity x HHV x Carbon Content x Fraction to CO2

        More precise than Tier 1 using fuel-specific data.
        Formula: CO2 (kg) = Fuel Qty x HHV x C% x 3.6667
        (3.6667 = 44/12, converting carbon to CO2)

        Args:
            fuel_quantity: Fuel consumed (kg)
            higher_heating_value: HHV in BTU/kg
            carbon_content: Carbon content as % by weight
            fuel_type: Type of fuel

        Returns:
            CO2 calculation result
        """
        # Convert HHV and carbon content to MMBtu heat input
        heat_input_mmbtu = (fuel_quantity * higher_heating_value) / 1_000_000.0

        # Tier 2 calculation: CO2 = Qty * HHV * C% * 3.6667
        co2_kg = fuel_quantity * higher_heating_value * (carbon_content / 100.0) * 3.6667
        co2_metric_tons = co2_kg / 1000.0

        implied_factor = co2_kg / heat_input_mmbtu if heat_input_mmbtu > 0 else 0

        return CO2Calculation(
            fuel_type=fuel_type,
            calculation_tier=TierLevel.TIER2,
            heat_input_mmbtu=heat_input_mmbtu,
            co2_kg=co2_kg,
            co2_metric_tons=co2_metric_tons,
            emission_factor_used=implied_factor,
            fuel_quantity=fuel_quantity,
            higher_heating_value=higher_heating_value,
            carbon_content=carbon_content
        )

    def calculate_co2_tier3(
        self,
        fuel_quantity: float,
        higher_heating_value: float,
        carbon_content: float,
        fuel_type: FuelType,
        cems_co2_measured_kg: Optional[float] = None
    ) -> CO2Calculation:
        """
        Tier 3: CO2 based on Continuous Emissions Monitoring Systems (CEMS)

        Uses measured CO2 from CEMS, or falls back to Tier 2 calc if not available.
        Most precise but requires CEMS equipment.

        Args:
            fuel_quantity: Fuel consumed (kg)
            higher_heating_value: HHV in BTU/kg
            carbon_content: Carbon content as % by weight
            fuel_type: Type of fuel
            cems_co2_measured_kg: Measured CO2 from CEMS (kg)

        Returns:
            CO2 calculation result
        """
        if cems_co2_measured_kg is not None:
            # Use measured CEMS data
            co2_kg = cems_co2_measured_kg
            heat_input_mmbtu = (fuel_quantity * higher_heating_value) / 1_000_000.0
            implied_factor = co2_kg / heat_input_mmbtu if heat_input_mmbtu > 0 else 0
        else:
            # Fall back to Tier 2 calculation
            tier2_result = self.calculate_co2_tier2(
                fuel_quantity, higher_heating_value, carbon_content, fuel_type
            )
            return CO2Calculation(
                **{**tier2_result.dict(), "calculation_tier": TierLevel.TIER3}
            )

        co2_metric_tons = co2_kg / 1000.0
        heat_input_mmbtu = (fuel_quantity * higher_heating_value) / 1_000_000.0

        return CO2Calculation(
            fuel_type=fuel_type,
            calculation_tier=TierLevel.TIER3,
            heat_input_mmbtu=heat_input_mmbtu,
            co2_kg=co2_kg,
            co2_metric_tons=co2_metric_tons,
            emission_factor_used=implied_factor,
            fuel_quantity=fuel_quantity,
            higher_heating_value=higher_heating_value,
            carbon_content=carbon_content
        )

    def calculate_ch4_n2o(
        self,
        fuel_type: FuelType,
        heat_input_mmbtu: float
    ) -> CH4N2OCalculation:
        """
        Calculate CH4 and N2O emissions using Table C-2 factors.

        CH4 and N2O are calculated using default emission factors
        regardless of tier (Tier 1 only for these gases per EPA guidance).

        Args:
            fuel_type: Type of fuel
            heat_input_mmbtu: Heat input in MMBtu

        Returns:
            CH4 and N2O calculation result
        """
        ch4_factor = CH4_N2O_FACTORS.get_ch4_factor(fuel_type)
        n2o_factor = CH4_N2O_FACTORS.get_n2o_factor(fuel_type)

        ch4_kg = heat_input_mmbtu * ch4_factor
        n2o_kg = heat_input_mmbtu * n2o_factor

        ch4_metric_tons = ch4_kg / 1000.0
        n2o_metric_tons = n2o_kg / 1000.0

        return CH4N2OCalculation(
            fuel_type=fuel_type,
            heat_input_mmbtu=heat_input_mmbtu,
            ch4_kg=ch4_kg,
            ch4_metric_tons=ch4_metric_tons,
            n2o_kg=n2o_kg,
            n2o_metric_tons=n2o_metric_tons,
            ch4_factor=ch4_factor,
            n2o_factor=n2o_factor
        )

    def generate_annual_report(
        self,
        facility_data: List[SubpartCResult]
    ) -> Dict[str, Any]:
        """
        Generate annual facility-level GHG report for GHGRP submission.

        Aggregates all source category emissions and formats for
        EPA GHGRP XML submission.

        Args:
            facility_data: List of all Subpart C results for facility

        Returns:
            Annual report dictionary ready for GHGRP submission
        """
        if not facility_data:
            raise ValueError("No facility data provided")

        # Validate all records are from same facility
        facility_ids = set(r.facility_id for r in facility_data)
        if len(facility_ids) > 1:
            raise ValueError(f"Multiple facilities in data: {facility_ids}")

        facility_id = facility_data[0].facility_id
        reporting_year = facility_data[0].reporting_year

        # Aggregate emissions
        total_co2 = sum(r.total_co2_metric_tons for r in facility_data)
        total_ch4 = sum(r.total_ch4_metric_tons for r in facility_data)
        total_n2o = sum(r.total_n2o_metric_tons for r in facility_data)

        gwp_ch4 = self.GWP_CH4_AR5 if self.config.use_gwp_ar5 else self.GWP_CH4_AR4
        gwp_n2o = self.GWP_N2O_AR5 if self.config.use_gwp_ar5 else self.GWP_N2O_AR4

        total_co2e = (
            total_co2 +
            (total_ch4 * gwp_ch4) +
            (total_n2o * gwp_n2o)
        )

        # Determine reporting requirements
        exceeds_threshold = total_co2e >= self.config.threshold_mtco2e

        report = {
            "facility_id": facility_id,
            "epa_ghgrp_id": self.config.epa_ghgrp_id,
            "facility_name": self.config.facility_name,
            "reporting_year": reporting_year,
            "report_date": datetime.utcnow().isoformat(),

            # Emissions summary
            "emissions_summary": {
                "total_co2_metric_tons": round(total_co2, 2),
                "total_ch4_metric_tons": round(total_ch4, 4),
                "total_n2o_metric_tons": round(total_n2o, 4),
                "total_co2e_metric_tons": round(total_co2e, 2),
                "gwp_ch4": gwp_ch4,
                "gwp_n2o": gwp_n2o,
            },

            # Regulatory compliance
            "threshold_mtco2e": self.config.threshold_mtco2e,
            "exceeds_threshold": exceeds_threshold,
            "requires_reporting": exceeds_threshold,
            "reporting_status": "REQUIRED" if exceeds_threshold else "NOT_REQUIRED",

            # Source category details
            "source_categories": [
                {
                    "process_id": r.process_id or "UNKNOWN",
                    "fuel_type": r.fuel_type.value,
                    "co2_metric_tons": round(r.total_co2_metric_tons, 2),
                    "ch4_metric_tons": round(r.total_ch4_metric_tons, 4),
                    "n2o_metric_tons": round(r.total_n2o_metric_tons, 4),
                    "co2e_metric_tons": round(r.total_co2e_metric_tons, 2),
                    "calculation_tier": r.co2_calculation.calculation_tier.value,
                }
                for r in facility_data
            ],

            # Validation status
            "validation_status": "PASS" if all(r.validation_status == "PASS" for r in facility_data) else "FAIL",
            "total_records": len(facility_data),
            "processing_time_ms": sum(r.processing_time_ms for r in facility_data),
        }

        logger.info(
            f"Annual report generated: {facility_id} reporting_year={reporting_year} "
            f"total_co2e={total_co2e:.2f} MT exceeds_threshold={exceeds_threshold}"
        )

        return report

    # Private helper methods

    def _select_tier(self, fuel_data: FuelCombustionData) -> TierLevel:
        """Auto-select appropriate calculation tier based on available data."""
        if fuel_data.carbon_content is not None and fuel_data.higher_heating_value is not None:
            if self.config.enable_tier2_3:
                return TierLevel.TIER2
        return TierLevel.TIER1

    def _validate_fuel_data(self, fuel_data: FuelCombustionData, tier: TierLevel) -> List[str]:
        """Validate fuel data for selected tier."""
        errors = []

        if not fuel_data.fuel_type:
            errors.append("Fuel type is required")

        if fuel_data.heat_input_mmbtu < 0:
            errors.append("Heat input must be non-negative")

        if tier in [TierLevel.TIER2, TierLevel.TIER3]:
            if fuel_data.higher_heating_value is None:
                errors.append(f"{tier.value} requires higher heating value")
            if fuel_data.carbon_content is None:
                errors.append(f"{tier.value} requires carbon content")

        return errors

    def _calculate_co2(self, fuel_data: FuelCombustionData, tier: TierLevel) -> CO2Calculation:
        """Calculate CO2 using appropriate tier."""
        if tier == TierLevel.TIER1:
            factor = self.co2_factors.get_factor(fuel_data.fuel_type)
            return self.calculate_co2_tier1(fuel_data.heat_input_mmbtu, factor)

        elif tier == TierLevel.TIER2:
            return self.calculate_co2_tier2(
                fuel_data.fuel_quantity or fuel_data.heat_input_mmbtu,
                fuel_data.higher_heating_value or 0,
                fuel_data.carbon_content or 0,
                fuel_data.fuel_type
            )

        else:  # TIER3
            return self.calculate_co2_tier3(
                fuel_data.fuel_quantity or fuel_data.heat_input_mmbtu,
                fuel_data.higher_heating_value or 0,
                fuel_data.carbon_content or 0,
                fuel_data.fuel_type
            )

    def _calculate_ch4_n2o(self, fuel_data: FuelCombustionData) -> CH4N2OCalculation:
        """Calculate CH4 and N2O emissions."""
        return self.calculate_ch4_n2o(fuel_data.fuel_type, fuel_data.heat_input_mmbtu)

    def _calculate_provenance(self, input_data: Any, output_data: Any) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_str = f"{input_data.json()}{output_data.json()}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()
