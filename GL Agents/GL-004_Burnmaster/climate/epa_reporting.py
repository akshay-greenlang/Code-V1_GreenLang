"""
GL-004 BURNMASTER - EPA 40 CFR Part 98 Reporting

EPA Greenhouse Gas Reporting Program compliance for Subpart C (Stationary Combustion).

References:
    - 40 CFR Part 98 Subpart C - General Stationary Fuel Combustion Sources
    - EPA GHG Reporting Tool (e-GGRT)
    - EPA Emission Factors for Greenhouse Gas Inventories

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TierMethodology(str, Enum):
    """EPA Subpart C calculation tiers."""
    TIER_1 = "tier_1"  # Default emission factors
    TIER_2 = "tier_2"  # Facility-specific HHV
    TIER_3 = "tier_3"  # Carbon content + fuel analysis
    TIER_4 = "tier_4"  # CEMS-based continuous measurement


class FuelClassification(str, Enum):
    """EPA fuel classifications for Subpart C."""
    COAL_AND_COKE = "coal_and_coke"
    NATURAL_GAS = "natural_gas"
    PETROLEUM_PRODUCTS = "petroleum_products"
    BIOMASS = "biomass"
    OTHER_SOLID = "other_solid"
    OTHER_LIQUID = "other_liquid"
    OTHER_GASEOUS = "other_gaseous"


class CombustionUnitType(str, Enum):
    """Types of combustion units for EPA reporting."""
    BOILER = "boiler"
    PROCESS_HEATER = "process_heater"
    TURBINE = "turbine"
    IC_ENGINE = "ic_engine"
    OTHER = "other"


class SubpartCReport(BaseModel):
    """EPA Subpart C annual emissions report."""
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Facility identification
    facility_id: str = Field(..., description="EPA Facility ID")
    facility_name: str = Field(default="")
    reporting_year: int = Field(..., description="Reporting year")
    ghg_reporter_id: Optional[str] = Field(None, description="e-GGRT ID")

    # Total emissions (metric tonnes)
    total_co2_mt: Decimal = Field(..., ge=0, description="Total CO2 (metric tonnes)")
    total_ch4_mt: Decimal = Field(..., ge=0, description="Total CH4 (metric tonnes)")
    total_n2o_mt: Decimal = Field(..., ge=0, description="Total N2O (metric tonnes)")
    total_co2e_mt: Decimal = Field(..., ge=0, description="Total CO2e (metric tonnes)")

    # Biogenic emissions (reported separately)
    biogenic_co2_mt: Decimal = Field(default=Decimal("0"), ge=0)

    # Emissions by unit
    emissions_by_unit: Dict[str, Dict[str, Decimal]] = Field(default_factory=dict)

    # Emissions by fuel
    emissions_by_fuel: Dict[str, Dict[str, Decimal]] = Field(default_factory=dict)

    # Methodology
    calculation_tier: TierMethodology = Field(default=TierMethodology.TIER_1)
    missing_data_procedures_used: bool = Field(default=False)
    missing_data_hours: int = Field(default=0, ge=0)

    # Activity data (fuel consumption)
    fuel_consumption: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Quality assurance
    data_verification_status: str = Field(default="unverified")
    third_party_verification: bool = Field(default=False)

    provenance_hash: str = Field(default="")


# EPA 40 CFR Part 98 Table C-1 Default CO2 Emission Factors
# Units: kg CO2 per mmBtu
EPA_CO2_FACTORS_MMBTU: Dict[str, Decimal] = {
    "anthracite_coal": Decimal("103.69"),
    "bituminous_coal": Decimal("93.28"),
    "subbituminous_coal": Decimal("97.17"),
    "lignite": Decimal("97.72"),
    "coal_coke": Decimal("113.67"),
    "mixed_industrial": Decimal("90.63"),
    "municipal_solid_waste": Decimal("90.7"),
    "petroleum_coke": Decimal("102.41"),
    "plastics": Decimal("75.0"),
    "tires": Decimal("85.97"),
    "natural_gas": Decimal("53.06"),
    "landfill_gas": Decimal("52.07"),
    "other_biomass_gas": Decimal("52.07"),
    "propane": Decimal("62.87"),
    "ethane": Decimal("59.60"),
    "butane": Decimal("65.15"),
    "isobutane": Decimal("64.94"),
    "pentanes_plus": Decimal("70.02"),
    "lpg": Decimal("61.71"),
    "ngl": Decimal("68.02"),
    "still_gas": Decimal("64.20"),
    "fuel_gas": Decimal("59.00"),
    "distillate_fuel_oil_no1": Decimal("73.25"),
    "distillate_fuel_oil_no2": Decimal("73.96"),
    "distillate_fuel_oil_no4": Decimal("75.04"),
    "residual_fuel_oil_no5": Decimal("72.93"),
    "residual_fuel_oil_no6": Decimal("75.10"),
    "kerosene": Decimal("72.31"),
    "kerosene_jet_fuel": Decimal("72.22"),
    "aviation_gasoline": Decimal("69.25"),
    "motor_gasoline": Decimal("70.22"),
    "asphalt_road_oil": Decimal("75.36"),
    "crude_oil": Decimal("74.54"),
    "lubricants": Decimal("74.27"),
    "naphtha": Decimal("68.02"),
    "petrochemical_feedstocks": Decimal("71.02"),
    "special_naphtha": Decimal("72.34"),
    "unfinished_oils": Decimal("74.49"),
    "waxes": Decimal("72.64"),
    "wood_residuals": Decimal("93.80"),
    "agricultural_byproducts": Decimal("118.17"),
    "peat": Decimal("111.84"),
    "solid_byproducts": Decimal("105.51"),
}

# EPA Table C-2 Default CH4 and N2O Emission Factors
# Units: kg per mmBtu
EPA_CH4_N2O_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "coal": {"ch4": Decimal("0.011"), "n2o": Decimal("0.0016")},
    "natural_gas": {"ch4": Decimal("0.001"), "n2o": Decimal("0.0001")},
    "petroleum": {"ch4": Decimal("0.003"), "n2o": Decimal("0.0006")},
    "biomass": {"ch4": Decimal("0.032"), "n2o": Decimal("0.0042")},
    "other": {"ch4": Decimal("0.003"), "n2o": Decimal("0.0006")},
}

# EPA GWP values (per 40 CFR Part 98 Subpart A Table A-1)
# These are IPCC AR4 values as required by EPA
EPA_GWP = {
    "co2": 1,
    "ch4": 25,
    "n2o": 298,
}


class EPAReporter:
    """
    EPA 40 CFR Part 98 Subpart C emissions reporter.

    Implements the EPA Greenhouse Gas Reporting Program requirements
    for stationary combustion sources.

    Example:
        >>> reporter = EPAReporter()
        >>> report = reporter.generate_subpart_c_report(
        ...     facility_id="EPA-12345",
        ...     reporting_year=2024,
        ...     fuel_data=[
        ...         {"fuel": "natural_gas", "quantity_mmbtu": Decimal("50000")}
        ...     ]
        ... )
    """

    def __init__(self, precision: int = 4):
        """Initialize EPA reporter."""
        self.precision = precision
        self._quantize_str = "0." + "0" * precision
        self._custom_factors: Dict[str, Decimal] = {}
        logger.info("EPAReporter initialized")

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash."""
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def set_custom_factor(self, fuel_type: str, co2_kg_per_mmbtu: Decimal) -> None:
        """Set custom emission factor for a fuel."""
        self._custom_factors[fuel_type] = co2_kg_per_mmbtu
        logger.info(f"Custom factor set for {fuel_type}: {co2_kg_per_mmbtu} kg/mmBtu")

    def get_co2_factor(self, fuel_type: str) -> Decimal:
        """Get CO2 emission factor (kg/mmBtu)."""
        if fuel_type in self._custom_factors:
            return self._custom_factors[fuel_type]
        return EPA_CO2_FACTORS_MMBTU.get(fuel_type, Decimal("60.0"))

    def get_ch4_n2o_factors(self, fuel_type: str) -> Dict[str, Decimal]:
        """Get CH4 and N2O factors based on fuel classification."""
        fuel_lower = fuel_type.lower()

        if any(c in fuel_lower for c in ["coal", "coke", "lignite", "anthracite"]):
            return EPA_CH4_N2O_FACTORS["coal"]
        elif any(g in fuel_lower for g in ["natural_gas", "landfill_gas"]):
            return EPA_CH4_N2O_FACTORS["natural_gas"]
        elif any(b in fuel_lower for b in ["wood", "bio", "agricultural", "peat"]):
            return EPA_CH4_N2O_FACTORS["biomass"]
        elif any(p in fuel_lower for p in ["oil", "diesel", "gasoline", "fuel", "petroleum"]):
            return EPA_CH4_N2O_FACTORS["petroleum"]
        else:
            return EPA_CH4_N2O_FACTORS["other"]

    def calculate_tier1_emissions(
        self,
        fuel_type: str,
        quantity_mmbtu: Decimal,
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal, bool]:
        """
        Calculate emissions using Tier 1 methodology.

        Tier 1: Uses default emission factors from EPA tables.

        Args:
            fuel_type: EPA fuel type identifier
            quantity_mmbtu: Fuel consumption in mmBtu

        Returns:
            Tuple of (co2_kg, ch4_kg, n2o_kg, co2e_kg, is_biogenic)
        """
        # Get emission factors
        co2_factor = self.get_co2_factor(fuel_type)
        ch4_n2o = self.get_ch4_n2o_factors(fuel_type)

        # Calculate emissions (DETERMINISTIC per 40 CFR 98.33)
        co2_kg = self._quantize(quantity_mmbtu * co2_factor)
        ch4_kg = self._quantize(quantity_mmbtu * ch4_n2o["ch4"])
        n2o_kg = self._quantize(quantity_mmbtu * ch4_n2o["n2o"])

        # Calculate CO2e using EPA GWP values
        co2e_kg = self._quantize(
            co2_kg * EPA_GWP["co2"] +
            ch4_kg * EPA_GWP["ch4"] +
            n2o_kg * EPA_GWP["n2o"]
        )

        # Check if biogenic
        is_biogenic = any(
            b in fuel_type.lower()
            for b in ["wood", "bio", "agricultural", "peat", "landfill"]
        )

        return co2_kg, ch4_kg, n2o_kg, co2e_kg, is_biogenic

    def calculate_tier2_emissions(
        self,
        fuel_type: str,
        quantity_fuel_units: Decimal,
        hhv_mmbtu_per_unit: Decimal,
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal, bool]:
        """
        Calculate emissions using Tier 2 methodology.

        Tier 2: Uses facility-specific HHV with default emission factors.

        Args:
            fuel_type: EPA fuel type identifier
            quantity_fuel_units: Fuel consumption in fuel-specific units
            hhv_mmbtu_per_unit: High heating value (mmBtu per unit)

        Returns:
            Tuple of (co2_kg, ch4_kg, n2o_kg, co2e_kg, is_biogenic)
        """
        # Convert to mmBtu using facility-specific HHV
        quantity_mmbtu = self._quantize(quantity_fuel_units * hhv_mmbtu_per_unit)

        # Use Tier 1 calculation with derived mmBtu
        return self.calculate_tier1_emissions(fuel_type, quantity_mmbtu)

    def generate_subpart_c_report(
        self,
        facility_id: str,
        reporting_year: int,
        fuel_data: List[Dict[str, Any]],
        facility_name: str = "",
        ghg_reporter_id: Optional[str] = None,
        tier: TierMethodology = TierMethodology.TIER_1,
        units_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> SubpartCReport:
        """
        Generate EPA Subpart C annual emissions report.

        Args:
            facility_id: EPA facility identifier
            reporting_year: Year being reported
            fuel_data: List of fuel consumption records, each with:
                - fuel: str (EPA fuel type)
                - quantity_mmbtu: Decimal (for Tier 1)
                - quantity_units: Decimal (for Tier 2)
                - hhv: Decimal (for Tier 2)
                - unit_id: str (optional - for unit-level tracking)
            facility_name: Facility display name
            ghg_reporter_id: e-GGRT ID
            tier: Calculation tier methodology
            units_data: Optional unit-level metadata

        Returns:
            SubpartCReport with EPA-compliant emissions data
        """
        # Initialize accumulators
        total_co2_kg = Decimal("0")
        total_ch4_kg = Decimal("0")
        total_n2o_kg = Decimal("0")
        total_co2e_kg = Decimal("0")
        total_biogenic_kg = Decimal("0")

        emissions_by_fuel: Dict[str, Dict[str, Decimal]] = {}
        emissions_by_unit: Dict[str, Dict[str, Decimal]] = {}
        fuel_consumption: Dict[str, Dict[str, Any]] = {}

        # Process each fuel record
        for fd in fuel_data:
            fuel = fd.get("fuel", "natural_gas")
            unit_id = fd.get("unit_id", "UNIT-001")

            # Calculate based on tier
            if tier == TierMethodology.TIER_1:
                quantity_mmbtu = Decimal(str(fd.get("quantity_mmbtu", 0)))
                co2_kg, ch4_kg, n2o_kg, co2e_kg, is_bio = self.calculate_tier1_emissions(
                    fuel, quantity_mmbtu
                )
            elif tier == TierMethodology.TIER_2:
                quantity_units = Decimal(str(fd.get("quantity_units", 0)))
                hhv = Decimal(str(fd.get("hhv", 1.0)))
                co2_kg, ch4_kg, n2o_kg, co2e_kg, is_bio = self.calculate_tier2_emissions(
                    fuel, quantity_units, hhv
                )
                quantity_mmbtu = self._quantize(quantity_units * hhv)
            else:
                # Tier 3/4 - would require additional data
                quantity_mmbtu = Decimal(str(fd.get("quantity_mmbtu", 0)))
                co2_kg, ch4_kg, n2o_kg, co2e_kg, is_bio = self.calculate_tier1_emissions(
                    fuel, quantity_mmbtu
                )

            # Accumulate
            if is_bio:
                total_biogenic_kg += co2_kg
            else:
                total_co2_kg += co2_kg

            total_ch4_kg += ch4_kg
            total_n2o_kg += n2o_kg
            total_co2e_kg += co2e_kg

            # Track by fuel
            if fuel not in emissions_by_fuel:
                emissions_by_fuel[fuel] = {
                    "co2_kg": Decimal("0"),
                    "ch4_kg": Decimal("0"),
                    "n2o_kg": Decimal("0"),
                    "co2e_kg": Decimal("0"),
                }
                fuel_consumption[fuel] = {"quantity_mmbtu": Decimal("0")}

            emissions_by_fuel[fuel]["co2_kg"] += co2_kg
            emissions_by_fuel[fuel]["ch4_kg"] += ch4_kg
            emissions_by_fuel[fuel]["n2o_kg"] += n2o_kg
            emissions_by_fuel[fuel]["co2e_kg"] += co2e_kg
            fuel_consumption[fuel]["quantity_mmbtu"] += quantity_mmbtu

            # Track by unit
            if unit_id not in emissions_by_unit:
                emissions_by_unit[unit_id] = {
                    "co2_kg": Decimal("0"),
                    "ch4_kg": Decimal("0"),
                    "n2o_kg": Decimal("0"),
                    "co2e_kg": Decimal("0"),
                }

            emissions_by_unit[unit_id]["co2_kg"] += co2_kg
            emissions_by_unit[unit_id]["ch4_kg"] += ch4_kg
            emissions_by_unit[unit_id]["n2o_kg"] += n2o_kg
            emissions_by_unit[unit_id]["co2e_kg"] += co2e_kg

        # Convert to metric tonnes
        total_co2_mt = self._quantize(total_co2_kg / Decimal("1000"))
        total_ch4_mt = self._quantize(total_ch4_kg / Decimal("1000"))
        total_n2o_mt = self._quantize(total_n2o_kg / Decimal("1000"))
        total_co2e_mt = self._quantize(total_co2e_kg / Decimal("1000"))
        biogenic_co2_mt = self._quantize(total_biogenic_kg / Decimal("1000"))

        # Compute provenance
        provenance = self._compute_hash({
            "facility_id": facility_id,
            "reporting_year": reporting_year,
            "tier": tier.value,
            "total_co2e_mt": str(total_co2e_mt),
        })

        return SubpartCReport(
            facility_id=facility_id,
            facility_name=facility_name,
            reporting_year=reporting_year,
            ghg_reporter_id=ghg_reporter_id,
            total_co2_mt=total_co2_mt,
            total_ch4_mt=total_ch4_mt,
            total_n2o_mt=total_n2o_mt,
            total_co2e_mt=total_co2e_mt,
            biogenic_co2_mt=biogenic_co2_mt,
            emissions_by_unit=emissions_by_unit,
            emissions_by_fuel=emissions_by_fuel,
            fuel_consumption=fuel_consumption,
            calculation_tier=tier,
            provenance_hash=provenance,
        )

    def get_reporting_threshold(self) -> Decimal:
        """Get EPA GHG reporting threshold (25,000 metric tonnes CO2e)."""
        return Decimal("25000")

    def check_reporting_requirement(self, total_co2e_mt: Decimal) -> bool:
        """Check if facility exceeds EPA reporting threshold."""
        return total_co2e_mt >= self.get_reporting_threshold()
