"""
GL-004 BURNMASTER - EU ETS Reporting

EU Emissions Trading System compliance for stationary combustion installations.

References:
    - EU ETS Directive 2003/87/EC (amended)
    - Commission Regulation (EU) 2018/2066 (MRR)
    - Commission Regulation (EU) 2018/2067 (AVR)

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


class InstallationCategory(str, Enum):
    """EU ETS installation categories by emissions."""
    CATEGORY_A = "category_a"  # < 50,000 tCO2/year
    CATEGORY_B = "category_b"  # 50,000 - 500,000 tCO2/year
    CATEGORY_C = "category_c"  # > 500,000 tCO2/year


class ActivityType(str, Enum):
    """EU ETS Annex I activity types."""
    COMBUSTION_INSTALLATIONS = "combustion_installations"
    REFINING = "refining"
    COKE_OVENS = "coke_ovens"
    METAL_ORE_ROASTING = "metal_ore_roasting"
    PIG_IRON_STEEL = "pig_iron_steel"
    CEMENT_CLINKER = "cement_clinker"
    GLASS = "glass"
    CERAMICS = "ceramics"
    PULP_PAPER = "pulp_paper"
    CHEMICALS = "chemicals"


class CalculationTier(str, Enum):
    """EU ETS calculation tiers per MRR."""
    TIER_1 = "tier_1"  # Default factors
    TIER_2A = "tier_2a"  # Country-specific factors
    TIER_2B = "tier_2b"  # Installation-specific analysis
    TIER_3 = "tier_3"  # Full fuel analysis
    TIER_4 = "tier_4"  # CEMS


class FuelStream(BaseModel):
    """EU ETS fuel stream definition per MRR."""
    stream_id: str = Field(default_factory=lambda: str(uuid4()))
    stream_name: str = Field(..., description="Fuel stream name")
    fuel_type: str = Field(..., description="Fuel type")
    activity_data_tier: CalculationTier = Field(default=CalculationTier.TIER_2A)
    emission_factor_tier: CalculationTier = Field(default=CalculationTier.TIER_2A)
    oxidation_factor_tier: CalculationTier = Field(default=CalculationTier.TIER_1)


class ETSReport(BaseModel):
    """EU ETS annual emissions report (AER)."""
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Installation identification
    installation_id: str = Field(..., description="EU ETS installation ID")
    installation_name: str = Field(default="")
    permit_number: str = Field(default="")
    reporting_year: int = Field(..., description="Reporting year")
    member_state: str = Field(default="", description="EU member state code")

    # Installation category
    category: InstallationCategory = Field(default=InstallationCategory.CATEGORY_B)
    activity_type: ActivityType = Field(default=ActivityType.COMBUSTION_INSTALLATIONS)

    # Total emissions (tonnes CO2)
    total_emissions_tco2: Decimal = Field(..., ge=0, description="Total CO2 emissions")
    fossil_emissions_tco2: Decimal = Field(..., ge=0, description="Fossil CO2")
    biomass_emissions_tco2: Decimal = Field(default=Decimal("0"), ge=0)
    transferred_co2_tco2: Decimal = Field(default=Decimal("0"), ge=0)

    # Emissions by source stream
    emissions_by_stream: Dict[str, Decimal] = Field(default_factory=dict)

    # Activity data
    fuel_consumption_by_stream: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Calculation methodology
    calculation_approach: str = Field(default="standard")
    tiers_applied: Dict[str, str] = Field(default_factory=dict)

    # Free allocation and surrender
    free_allocation_tco2: Optional[Decimal] = Field(None, description="Free allowances")
    allowances_to_surrender: Optional[Decimal] = Field(None)

    # Verification
    verification_status: str = Field(default="unverified")
    verifier_name: Optional[str] = Field(None)
    verification_opinion: Optional[str] = Field(None)

    # Uncertainties (per MRR)
    overall_uncertainty_percent: float = Field(default=2.5, ge=0)

    provenance_hash: str = Field(default="")


# EU ETS default emission factors (tonnes CO2 per TJ)
# Source: MRR Annex VI
EU_ETS_EMISSION_FACTORS_TJ: Dict[str, Decimal] = {
    "hard_coal": Decimal("94.6"),
    "coking_coal": Decimal("94.6"),
    "sub_bituminous_coal": Decimal("96.1"),
    "lignite": Decimal("101.0"),
    "anthracite": Decimal("98.3"),
    "coal_coke": Decimal("107.0"),
    "petroleum_coke": Decimal("97.5"),
    "natural_gas": Decimal("56.1"),
    "liquefied_natural_gas": Decimal("56.1"),
    "compressed_natural_gas": Decimal("56.1"),
    "refinery_gas": Decimal("57.6"),
    "lpg": Decimal("63.1"),
    "ethane": Decimal("61.6"),
    "naphtha": Decimal("73.3"),
    "gasoline": Decimal("69.3"),
    "kerosene": Decimal("71.9"),
    "gas_diesel_oil": Decimal("74.1"),
    "heavy_fuel_oil": Decimal("77.4"),
    "waste_oil": Decimal("73.3"),
    "peat": Decimal("106.0"),
    "wood_pellets": Decimal("0.0"),  # Biomass - zero rated
    "biodiesel": Decimal("0.0"),     # Biomass - zero rated
    "biogas": Decimal("0.0"),        # Biomass - zero rated
}

# Default net calorific values (TJ per 1000 tonnes)
EU_ETS_NCV: Dict[str, Decimal] = {
    "hard_coal": Decimal("25.8"),
    "coking_coal": Decimal("28.2"),
    "sub_bituminous_coal": Decimal("18.9"),
    "lignite": Decimal("11.9"),
    "anthracite": Decimal("26.7"),
    "natural_gas": Decimal("48.0"),  # TJ per million Nm3
    "lpg": Decimal("47.3"),
    "gas_diesel_oil": Decimal("43.0"),
    "heavy_fuel_oil": Decimal("40.4"),
    "petroleum_coke": Decimal("32.5"),
}

# Default oxidation factors
DEFAULT_OXIDATION_FACTOR = Decimal("1.0")
COAL_OXIDATION_FACTOR = Decimal("0.99")


class EUETSReporter:
    """
    EU ETS emissions reporter for stationary combustion.

    Implements EU Monitoring and Reporting Regulation (MRR) requirements
    for annual emissions reports submitted to national authorities.

    Example:
        >>> reporter = EUETSReporter()
        >>> report = reporter.generate_annual_report(
        ...     installation_id="EU-ETS-DE-12345",
        ...     reporting_year=2024,
        ...     fuel_streams=[
        ...         {"stream_name": "Boiler NG", "fuel_type": "natural_gas",
        ...          "quantity_tj": Decimal("500")}
        ...     ]
        ... )
    """

    def __init__(self, precision: int = 3):
        """Initialize EU ETS reporter."""
        self.precision = precision
        self._quantize_str = "0." + "0" * precision
        self._custom_factors: Dict[str, Decimal] = {}
        self._custom_ncv: Dict[str, Decimal] = {}
        logger.info("EUETSReporter initialized")

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding per MRR requirements."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash."""
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def set_installation_specific_factor(
        self,
        fuel_type: str,
        emission_factor_tco2_per_tj: Decimal,
        ncv_tj_per_kt: Optional[Decimal] = None,
    ) -> None:
        """Set installation-specific emission factor (Tier 2b/3)."""
        self._custom_factors[fuel_type] = emission_factor_tco2_per_tj
        if ncv_tj_per_kt:
            self._custom_ncv[fuel_type] = ncv_tj_per_kt
        logger.info(f"Custom ETS factor set for {fuel_type}")

    def get_emission_factor(self, fuel_type: str) -> Decimal:
        """Get emission factor (tonnes CO2 per TJ)."""
        if fuel_type in self._custom_factors:
            return self._custom_factors[fuel_type]
        return EU_ETS_EMISSION_FACTORS_TJ.get(fuel_type, Decimal("74.0"))

    def get_ncv(self, fuel_type: str) -> Decimal:
        """Get net calorific value (TJ per 1000 tonnes)."""
        if fuel_type in self._custom_ncv:
            return self._custom_ncv[fuel_type]
        return EU_ETS_NCV.get(fuel_type, Decimal("40.0"))

    def get_oxidation_factor(self, fuel_type: str) -> Decimal:
        """Get oxidation factor for fuel type."""
        if "coal" in fuel_type.lower() or "coke" in fuel_type.lower():
            return COAL_OXIDATION_FACTOR
        return DEFAULT_OXIDATION_FACTOR

    def calculate_stream_emissions(
        self,
        fuel_type: str,
        quantity_tj: Optional[Decimal] = None,
        quantity_tonnes: Optional[Decimal] = None,
    ) -> Tuple[Decimal, bool]:
        """
        Calculate emissions for a fuel stream per MRR methodology.

        CO2 = Activity Data x Emission Factor x Oxidation Factor

        Args:
            fuel_type: Fuel type identifier
            quantity_tj: Energy consumption in TJ (if known)
            quantity_tonnes: Mass consumption in tonnes (converted via NCV)

        Returns:
            Tuple of (emissions_tco2, is_biomass)
        """
        # Convert mass to energy if needed
        if quantity_tj is None and quantity_tonnes is not None:
            ncv = self.get_ncv(fuel_type)
            quantity_tj = self._quantize(quantity_tonnes * ncv / Decimal("1000"))
        elif quantity_tj is None:
            quantity_tj = Decimal("0")

        # Get factors
        ef = self.get_emission_factor(fuel_type)
        of = self.get_oxidation_factor(fuel_type)

        # Calculate emissions (DETERMINISTIC per MRR)
        emissions_tco2 = self._quantize(quantity_tj * ef * of)

        # Check if biomass (zero-rated under EU ETS)
        is_biomass = any(
            b in fuel_type.lower()
            for b in ["wood", "bio", "pellet", "biodiesel", "biogas"]
        )

        return emissions_tco2, is_biomass

    def determine_installation_category(
        self,
        total_emissions_tco2: Decimal
    ) -> InstallationCategory:
        """Determine installation category based on emissions."""
        if total_emissions_tco2 < Decimal("50000"):
            return InstallationCategory.CATEGORY_A
        elif total_emissions_tco2 < Decimal("500000"):
            return InstallationCategory.CATEGORY_B
        else:
            return InstallationCategory.CATEGORY_C

    def generate_annual_report(
        self,
        installation_id: str,
        reporting_year: int,
        fuel_streams: List[Dict[str, Any]],
        installation_name: str = "",
        permit_number: str = "",
        member_state: str = "",
        activity_type: ActivityType = ActivityType.COMBUSTION_INSTALLATIONS,
        free_allocation_tco2: Optional[Decimal] = None,
    ) -> ETSReport:
        """
        Generate EU ETS Annual Emissions Report.

        Args:
            installation_id: EU ETS installation identifier
            reporting_year: Year being reported
            fuel_streams: List of fuel stream data, each with:
                - stream_name: str
                - fuel_type: str
                - quantity_tj: Decimal (optional)
                - quantity_tonnes: Decimal (optional)
            installation_name: Installation display name
            permit_number: GHG permit number
            member_state: EU member state code (DE, FR, etc.)
            activity_type: Annex I activity type
            free_allocation_tco2: Free allowances allocated

        Returns:
            ETSReport with MRR-compliant emissions data
        """
        # Initialize accumulators
        total_fossil_tco2 = Decimal("0")
        total_biomass_tco2 = Decimal("0")
        emissions_by_stream: Dict[str, Decimal] = {}
        fuel_consumption: Dict[str, Dict[str, Any]] = {}
        tiers_applied: Dict[str, str] = {}

        # Process each fuel stream
        for stream in fuel_streams:
            stream_name = stream.get("stream_name", "Stream")
            fuel_type = stream.get("fuel_type", "natural_gas")
            quantity_tj = stream.get("quantity_tj")
            quantity_tonnes = stream.get("quantity_tonnes")

            if quantity_tj is not None:
                quantity_tj = Decimal(str(quantity_tj))
            if quantity_tonnes is not None:
                quantity_tonnes = Decimal(str(quantity_tonnes))

            # Calculate emissions
            stream_emissions, is_biomass = self.calculate_stream_emissions(
                fuel_type, quantity_tj, quantity_tonnes
            )

            # Track by stream
            emissions_by_stream[stream_name] = stream_emissions

            # Accumulate
            if is_biomass:
                total_biomass_tco2 += stream_emissions
            else:
                total_fossil_tco2 += stream_emissions

            # Track fuel consumption
            fuel_consumption[stream_name] = {
                "fuel_type": fuel_type,
                "quantity_tj": str(quantity_tj) if quantity_tj else None,
                "quantity_tonnes": str(quantity_tonnes) if quantity_tonnes else None,
            }

            # Record tier (simplified - would come from monitoring plan)
            tiers_applied[stream_name] = CalculationTier.TIER_2A.value

        # Total emissions (fossil only for compliance)
        total_emissions_tco2 = total_fossil_tco2

        # Determine category
        category = self.determine_installation_category(total_emissions_tco2)

        # Calculate allowances to surrender
        allowances_to_surrender = None
        if free_allocation_tco2 is not None:
            allowances_to_surrender = max(
                Decimal("0"),
                total_emissions_tco2 - free_allocation_tco2
            )

        # Compute provenance
        provenance = self._compute_hash({
            "installation_id": installation_id,
            "reporting_year": reporting_year,
            "total_emissions_tco2": str(total_emissions_tco2),
            "streams_count": len(fuel_streams),
        })

        return ETSReport(
            installation_id=installation_id,
            installation_name=installation_name,
            permit_number=permit_number,
            reporting_year=reporting_year,
            member_state=member_state,
            category=category,
            activity_type=activity_type,
            total_emissions_tco2=total_emissions_tco2,
            fossil_emissions_tco2=total_fossil_tco2,
            biomass_emissions_tco2=total_biomass_tco2,
            emissions_by_stream=emissions_by_stream,
            fuel_consumption_by_stream=fuel_consumption,
            tiers_applied=tiers_applied,
            free_allocation_tco2=free_allocation_tco2,
            allowances_to_surrender=allowances_to_surrender,
            provenance_hash=provenance,
        )

    def get_required_uncertainty(self, category: InstallationCategory) -> float:
        """Get required uncertainty threshold per MRR for category."""
        thresholds = {
            InstallationCategory.CATEGORY_A: 7.5,  # +/- 7.5%
            InstallationCategory.CATEGORY_B: 5.0,  # +/- 5.0%
            InstallationCategory.CATEGORY_C: 2.5,  # +/- 2.5%
        }
        return thresholds.get(category, 5.0)

    def get_verification_frequency(self, category: InstallationCategory) -> str:
        """Get required verification frequency per AVR."""
        frequencies = {
            InstallationCategory.CATEGORY_A: "simplified",  # May use simplified
            InstallationCategory.CATEGORY_B: "annual",
            InstallationCategory.CATEGORY_C: "annual",
        }
        return frequencies.get(category, "annual")
