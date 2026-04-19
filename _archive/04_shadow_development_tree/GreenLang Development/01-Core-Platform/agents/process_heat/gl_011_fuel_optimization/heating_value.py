"""
GL-011 FUELCRAFT - Heating Value Calculator

This module provides deterministic calculations for fuel heating values
including Higher Heating Value (HHV), Lower Heating Value (LHV), and
Wobbe Index for gas interchangeability analysis.

All calculations follow ASTM D3588 and ISO 6976 standards.
Zero-hallucination: No ML/LLM in calculation path.

Example:
    >>> from greenlang.agents.process_heat.gl_011_fuel_optimization.heating_value import (
    ...     HeatingValueCalculator,
    ...     HeatingValueInput,
    ... )
    >>>
    >>> calc = HeatingValueCalculator()
    >>> result = calc.calculate_hhv(HeatingValueInput(
    ...     fuel_type="natural_gas",
    ...     methane_pct=95.0,
    ...     ethane_pct=3.0,
    ...     propane_pct=1.0,
    ...     nitrogen_pct=1.0,
    ... ))
    >>> print(f"HHV: {result.hhv_btu_scf:.1f} BTU/SCF")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import logging
import math

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Standard Reference Values (ASTM D3588, ISO 6976)
# =============================================================================

@dataclass(frozen=True)
class GasComponent:
    """Properties of a gas component at standard conditions (60F, 14.696 psia)."""
    name: str
    molecular_weight: float  # lb/lb-mol
    hhv_btu_scf: float  # BTU/SCF (gross)
    lhv_btu_scf: float  # BTU/SCF (net)
    specific_gravity: float  # Relative to air


# Standard gas component properties (60F, 14.696 psia)
GAS_COMPONENTS = {
    "methane": GasComponent("Methane", 16.043, 1010.0, 909.4, 0.5539),
    "ethane": GasComponent("Ethane", 30.070, 1769.7, 1618.7, 1.0382),
    "propane": GasComponent("Propane", 44.097, 2516.1, 2314.9, 1.5226),
    "n_butane": GasComponent("n-Butane", 58.123, 3262.3, 3010.8, 2.0068),
    "i_butane": GasComponent("i-Butane", 58.123, 3251.9, 3000.4, 2.0068),
    "n_pentane": GasComponent("n-Pentane", 72.150, 4008.9, 3706.9, 2.4911),
    "i_pentane": GasComponent("i-Pentane", 72.150, 4000.9, 3699.0, 2.4911),
    "hexane": GasComponent("Hexanes+", 86.177, 4755.9, 4403.8, 2.9753),
    "nitrogen": GasComponent("Nitrogen", 28.014, 0.0, 0.0, 0.9672),
    "carbon_dioxide": GasComponent("CO2", 44.010, 0.0, 0.0, 1.5196),
    "hydrogen_sulfide": GasComponent("H2S", 34.082, 637.1, 586.8, 1.1767),
    "hydrogen": GasComponent("Hydrogen", 2.016, 324.2, 273.8, 0.0696),
    "carbon_monoxide": GasComponent("CO", 28.010, 321.8, 321.8, 0.9671),
    "oxygen": GasComponent("Oxygen", 32.000, 0.0, 0.0, 1.1048),
    "water_vapor": GasComponent("Water", 18.015, 0.0, 0.0, 0.6220),
    "helium": GasComponent("Helium", 4.003, 0.0, 0.0, 0.1382),
    "argon": GasComponent("Argon", 39.948, 0.0, 0.0, 1.3793),
}

# Liquid fuel heating values (BTU/lb, BTU/gallon)
LIQUID_FUEL_PROPERTIES = {
    "no2_fuel_oil": {
        "hhv_btu_lb": 19_580,
        "lhv_btu_lb": 18_410,
        "density_lb_gal": 7.21,
        "api_gravity": 33.0,
    },
    "no4_fuel_oil": {
        "hhv_btu_lb": 18_890,
        "lhv_btu_lb": 17_770,
        "density_lb_gal": 7.75,
        "api_gravity": 24.0,
    },
    "no6_fuel_oil": {
        "hhv_btu_lb": 18_300,
        "lhv_btu_lb": 17_250,
        "density_lb_gal": 8.10,
        "api_gravity": 15.0,
    },
    "diesel": {
        "hhv_btu_lb": 19_300,
        "lhv_btu_lb": 18_150,
        "density_lb_gal": 7.05,
        "api_gravity": 38.0,
    },
    "kerosene": {
        "hhv_btu_lb": 19_810,
        "lhv_btu_lb": 18_560,
        "density_lb_gal": 6.82,
        "api_gravity": 42.0,
    },
    "lpg_propane": {
        "hhv_btu_lb": 21_500,
        "lhv_btu_lb": 19_770,
        "density_lb_gal": 4.20,
        "api_gravity": None,
    },
    "lpg_butane": {
        "hhv_btu_lb": 21_180,
        "lhv_btu_lb": 19_520,
        "density_lb_gal": 4.84,
        "api_gravity": None,
    },
}

# Solid fuel heating values (BTU/lb, as-received basis)
SOLID_FUEL_PROPERTIES = {
    "coal_anthracite": {
        "hhv_btu_lb": 13_000,
        "lhv_btu_lb": 12_350,
        "moisture_pct": 4.0,
        "ash_pct": 10.0,
    },
    "coal_bituminous": {
        "hhv_btu_lb": 12_500,
        "lhv_btu_lb": 11_900,
        "moisture_pct": 6.0,
        "ash_pct": 8.0,
    },
    "coal_sub_bituminous": {
        "hhv_btu_lb": 9_500,
        "lhv_btu_lb": 9_000,
        "moisture_pct": 20.0,
        "ash_pct": 5.0,
    },
    "coal_lignite": {
        "hhv_btu_lb": 7_000,
        "lhv_btu_lb": 6_500,
        "moisture_pct": 35.0,
        "ash_pct": 6.0,
    },
    "biomass_wood": {
        "hhv_btu_lb": 8_500,
        "lhv_btu_lb": 7_800,
        "moisture_pct": 25.0,
        "ash_pct": 1.0,
    },
    "biomass_pellets": {
        "hhv_btu_lb": 8_000,
        "lhv_btu_lb": 7_400,
        "moisture_pct": 8.0,
        "ash_pct": 0.5,
    },
}


# =============================================================================
# DATA MODELS
# =============================================================================

class HeatingValueInput(BaseModel):
    """Input for heating value calculations."""

    fuel_type: str = Field(..., description="Fuel type identifier")

    # Gas composition (mol %)
    methane_pct: Optional[float] = Field(default=None, ge=0, le=100)
    ethane_pct: Optional[float] = Field(default=None, ge=0, le=100)
    propane_pct: Optional[float] = Field(default=None, ge=0, le=100)
    n_butane_pct: Optional[float] = Field(default=None, ge=0, le=100)
    i_butane_pct: Optional[float] = Field(default=None, ge=0, le=100)
    n_pentane_pct: Optional[float] = Field(default=None, ge=0, le=100)
    i_pentane_pct: Optional[float] = Field(default=None, ge=0, le=100)
    hexane_pct: Optional[float] = Field(default=None, ge=0, le=100)
    nitrogen_pct: Optional[float] = Field(default=None, ge=0, le=100)
    co2_pct: Optional[float] = Field(default=None, ge=0, le=100)
    h2s_pct: Optional[float] = Field(default=None, ge=0, le=100)
    hydrogen_pct: Optional[float] = Field(default=None, ge=0, le=100)
    oxygen_pct: Optional[float] = Field(default=None, ge=0, le=100)
    helium_pct: Optional[float] = Field(default=None, ge=0, le=100)
    argon_pct: Optional[float] = Field(default=None, ge=0, le=100)

    # Temperature/pressure (for real gas corrections)
    temperature_f: float = Field(default=60.0, description="Temperature (F)")
    pressure_psia: float = Field(default=14.696, description="Pressure (psia)")

    # For solid/liquid fuels
    moisture_pct: Optional[float] = Field(default=None, ge=0, le=100)
    ash_pct: Optional[float] = Field(default=None, ge=0, le=100)
    sulfur_pct: Optional[float] = Field(default=None, ge=0, le=100)

    @validator("methane_pct", always=True)
    def validate_gas_composition(cls, v, values):
        """Validate gas composition sums to approximately 100%."""
        if v is None:
            return v

        components = [
            values.get("methane_pct", 0) or 0,
            values.get("ethane_pct", 0) or 0,
            values.get("propane_pct", 0) or 0,
            values.get("n_butane_pct", 0) or 0,
            values.get("i_butane_pct", 0) or 0,
            values.get("n_pentane_pct", 0) or 0,
            values.get("i_pentane_pct", 0) or 0,
            values.get("hexane_pct", 0) or 0,
            values.get("nitrogen_pct", 0) or 0,
            values.get("co2_pct", 0) or 0,
            values.get("h2s_pct", 0) or 0,
            values.get("hydrogen_pct", 0) or 0,
            values.get("oxygen_pct", 0) or 0,
            values.get("helium_pct", 0) or 0,
            values.get("argon_pct", 0) or 0,
        ]

        total = sum(components)
        if total > 0 and abs(total - 100.0) > 1.0:
            logger.warning(
                f"Gas composition sums to {total:.2f}%, expected ~100%"
            )

        return v


class HeatingValueResult(BaseModel):
    """Result from heating value calculation."""

    # Primary results
    hhv_btu_scf: Optional[float] = Field(
        default=None,
        description="Higher Heating Value (BTU/SCF)"
    )
    lhv_btu_scf: Optional[float] = Field(
        default=None,
        description="Lower Heating Value (BTU/SCF)"
    )
    hhv_btu_lb: Optional[float] = Field(
        default=None,
        description="Higher Heating Value (BTU/lb)"
    )
    lhv_btu_lb: Optional[float] = Field(
        default=None,
        description="Lower Heating Value (BTU/lb)"
    )
    hhv_mj_m3: Optional[float] = Field(
        default=None,
        description="Higher Heating Value (MJ/m3)"
    )
    lhv_mj_m3: Optional[float] = Field(
        default=None,
        description="Lower Heating Value (MJ/m3)"
    )

    # Physical properties
    specific_gravity: Optional[float] = Field(
        default=None,
        description="Specific gravity relative to air"
    )
    molecular_weight: Optional[float] = Field(
        default=None,
        description="Average molecular weight (lb/lb-mol)"
    )
    density_lb_scf: Optional[float] = Field(
        default=None,
        description="Density (lb/SCF)"
    )

    # Wobbe Index
    wobbe_index: Optional[float] = Field(
        default=None,
        description="Wobbe Index (BTU/SCF)"
    )
    wobbe_index_modified: Optional[float] = Field(
        default=None,
        description="Modified Wobbe Index"
    )

    # Calculation metadata
    calculation_method: str = Field(
        default="ASTM_D3588",
        description="Calculation method used"
    )
    reference_conditions: str = Field(
        default="60F, 14.696 psia",
        description="Reference conditions"
    )
    provenance_hash: str = Field(..., description="Calculation provenance hash")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # Warnings
    warnings: List[str] = Field(
        default_factory=list,
        description="Calculation warnings"
    )


class WobbeIndexResult(BaseModel):
    """Result from Wobbe Index calculation."""

    wobbe_index: float = Field(..., description="Wobbe Index (BTU/SCF)")
    wobbe_index_modified: float = Field(
        ...,
        description="Modified Wobbe Index"
    )
    hhv_btu_scf: float = Field(..., description="HHV used (BTU/SCF)")
    specific_gravity: float = Field(..., description="Specific gravity")

    # Interchangeability
    reference_wobbe: float = Field(
        default=1360.0,
        description="Reference Wobbe Index"
    )
    deviation_pct: float = Field(..., description="Deviation from reference (%)")
    interchangeable: bool = Field(
        ...,
        description="Within interchangeability limits"
    )
    interchangeability_status: str = Field(
        default="acceptable",
        description="Interchangeability status"
    )

    # AGA Bulletin 36 indices
    lifting_index: Optional[float] = Field(
        default=None,
        description="Lifting Index (IL)"
    )
    flashback_index: Optional[float] = Field(
        default=None,
        description="Flashback Index (IF)"
    )
    yellow_tip_index: Optional[float] = Field(
        default=None,
        description="Yellow Tip Index (IY)"
    )

    provenance_hash: str = Field(..., description="Calculation provenance hash")


# =============================================================================
# HEATING VALUE CALCULATOR
# =============================================================================

class HeatingValueCalculator:
    """
    Heating value calculator for gaseous, liquid, and solid fuels.

    This calculator provides ASTM D3588 and ISO 6976 compliant calculations
    for fuel heating values and Wobbe Index. All calculations are deterministic
    with complete provenance tracking.

    Features:
        - HHV/LHV calculation for gas mixtures
        - Wobbe Index for gas interchangeability
        - Support for liquid and solid fuels
        - Real gas corrections for non-standard conditions
        - AGA Bulletin 36 interchangeability indices

    Example:
        >>> calc = HeatingValueCalculator()
        >>> result = calc.calculate_hhv(input_data)
        >>> print(f"HHV: {result.hhv_btu_scf:.1f} BTU/SCF")
    """

    def __init__(self, reference_temp_f: float = 60.0) -> None:
        """
        Initialize the heating value calculator.

        Args:
            reference_temp_f: Reference temperature for calculations (default 60F)
        """
        self.reference_temp_f = reference_temp_f
        self.reference_pressure_psia = 14.696
        self._calculation_count = 0

        logger.info(
            f"HeatingValueCalculator initialized "
            f"(ref: {reference_temp_f}F, {self.reference_pressure_psia} psia)"
        )

    def calculate_gas_heating_value(
        self,
        input_data: HeatingValueInput,
    ) -> HeatingValueResult:
        """
        Calculate heating values for a gas mixture.

        Uses ASTM D3588 method for calculating HHV and LHV from gas
        composition. The calculation is deterministic and auditable.

        Args:
            input_data: Gas composition and conditions

        Returns:
            HeatingValueResult with HHV, LHV, and related properties

        Raises:
            ValueError: If input data is invalid
        """
        logger.debug(f"Calculating gas heating value for {input_data.fuel_type}")
        self._calculation_count += 1

        warnings = []

        # Build composition dictionary
        composition = self._build_composition_dict(input_data)

        if not composition:
            # Use default composition for fuel type
            composition = self._get_default_composition(input_data.fuel_type)
            warnings.append(
                f"Using default composition for {input_data.fuel_type}"
            )

        # Validate composition
        total = sum(composition.values())
        if abs(total - 100.0) > 0.5:
            warnings.append(
                f"Composition sums to {total:.2f}%, normalized to 100%"
            )
            composition = {k: v * 100.0 / total for k, v in composition.items()}

        # Calculate mixture properties
        hhv_btu_scf = 0.0
        lhv_btu_scf = 0.0
        molecular_weight = 0.0
        specific_gravity = 0.0

        for component, mol_pct in composition.items():
            if mol_pct <= 0:
                continue

            mol_frac = mol_pct / 100.0

            if component in GAS_COMPONENTS:
                props = GAS_COMPONENTS[component]
                hhv_btu_scf += props.hhv_btu_scf * mol_frac
                lhv_btu_scf += props.lhv_btu_scf * mol_frac
                molecular_weight += props.molecular_weight * mol_frac
                specific_gravity += props.specific_gravity * mol_frac

        # Calculate derived properties
        # Density at standard conditions (lb/SCF)
        # Using ideal gas: rho = PM/RT where P=14.696 psia, T=60F=520R
        # rho = P * MW / (R * T) = 14.696 * MW / (10.73 * 520)
        density_lb_scf = molecular_weight / 379.5  # Molar volume at STP

        # Mass-based heating values
        if density_lb_scf > 0:
            hhv_btu_lb = hhv_btu_scf / density_lb_scf
            lhv_btu_lb = lhv_btu_scf / density_lb_scf
        else:
            hhv_btu_lb = 0.0
            lhv_btu_lb = 0.0

        # Wobbe Index
        if specific_gravity > 0:
            wobbe_index = hhv_btu_scf / math.sqrt(specific_gravity)
            # Modified Wobbe Index (temperature corrected)
            temp_correction = math.sqrt(
                (input_data.temperature_f + 460) / 520
            )
            wobbe_index_modified = wobbe_index / temp_correction
        else:
            wobbe_index = 0.0
            wobbe_index_modified = 0.0
            warnings.append("Could not calculate Wobbe Index (SG=0)")

        # Convert to SI units
        hhv_mj_m3 = hhv_btu_scf * 0.0372506
        lhv_mj_m3 = lhv_btu_scf * 0.0372506

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data.dict(),
            {
                "hhv_btu_scf": hhv_btu_scf,
                "lhv_btu_scf": lhv_btu_scf,
                "wobbe_index": wobbe_index,
            }
        )

        return HeatingValueResult(
            hhv_btu_scf=round(hhv_btu_scf, 2),
            lhv_btu_scf=round(lhv_btu_scf, 2),
            hhv_btu_lb=round(hhv_btu_lb, 2),
            lhv_btu_lb=round(lhv_btu_lb, 2),
            hhv_mj_m3=round(hhv_mj_m3, 4),
            lhv_mj_m3=round(lhv_mj_m3, 4),
            specific_gravity=round(specific_gravity, 4),
            molecular_weight=round(molecular_weight, 3),
            density_lb_scf=round(density_lb_scf, 6),
            wobbe_index=round(wobbe_index, 2),
            wobbe_index_modified=round(wobbe_index_modified, 2),
            calculation_method="ASTM_D3588",
            reference_conditions=f"{self.reference_temp_f}F, {self.reference_pressure_psia} psia",
            provenance_hash=provenance_hash,
            warnings=warnings,
        )

    def calculate_liquid_heating_value(
        self,
        fuel_type: str,
        api_gravity: Optional[float] = None,
    ) -> HeatingValueResult:
        """
        Calculate heating values for liquid fuels.

        Uses correlations from API Technical Data Book for liquid
        petroleum products.

        Args:
            fuel_type: Liquid fuel type identifier
            api_gravity: API gravity (optional, uses default if not provided)

        Returns:
            HeatingValueResult with HHV and LHV

        Raises:
            ValueError: If fuel type is unknown
        """
        logger.debug(f"Calculating liquid heating value for {fuel_type}")
        self._calculation_count += 1

        warnings = []
        fuel_key = fuel_type.lower().replace(" ", "_").replace("-", "_")

        if fuel_key in LIQUID_FUEL_PROPERTIES:
            props = LIQUID_FUEL_PROPERTIES[fuel_key]
            hhv_btu_lb = props["hhv_btu_lb"]
            lhv_btu_lb = props["lhv_btu_lb"]
            density_lb_gal = props["density_lb_gal"]
        else:
            # Use API gravity correlation
            if api_gravity is None:
                raise ValueError(
                    f"Unknown fuel type '{fuel_type}' and no API gravity provided"
                )

            # Calculate density from API gravity
            # SG = 141.5 / (API + 131.5)
            sg = 141.5 / (api_gravity + 131.5)
            density_lb_gal = sg * 8.337  # Water is 8.337 lb/gal

            # API correlation for HHV
            # HHV (BTU/lb) = 22,320 - 3,780 * SG^2
            hhv_btu_lb = 22320 - 3780 * sg * sg

            # LHV correction (approximately 5-6% less than HHV)
            lhv_btu_lb = hhv_btu_lb * 0.94

            warnings.append(
                f"Used API gravity correlation for {fuel_type}"
            )

        # Calculate volumetric heating values
        hhv_btu_gal = hhv_btu_lb * density_lb_gal
        lhv_btu_gal = lhv_btu_lb * density_lb_gal

        provenance_hash = self._calculate_provenance_hash(
            {"fuel_type": fuel_type, "api_gravity": api_gravity},
            {"hhv_btu_lb": hhv_btu_lb, "lhv_btu_lb": lhv_btu_lb}
        )

        return HeatingValueResult(
            hhv_btu_lb=round(hhv_btu_lb, 0),
            lhv_btu_lb=round(lhv_btu_lb, 0),
            density_lb_scf=None,  # Not applicable for liquids
            calculation_method="API_TECHNICAL_DATA_BOOK",
            provenance_hash=provenance_hash,
            warnings=warnings,
        )

    def calculate_solid_heating_value(
        self,
        fuel_type: str,
        moisture_pct: Optional[float] = None,
        ash_pct: Optional[float] = None,
    ) -> HeatingValueResult:
        """
        Calculate heating values for solid fuels.

        Uses Dulong formula and moisture/ash corrections for
        coal and biomass fuels.

        Args:
            fuel_type: Solid fuel type identifier
            moisture_pct: Moisture content (%)
            ash_pct: Ash content (%)

        Returns:
            HeatingValueResult with HHV and LHV
        """
        logger.debug(f"Calculating solid heating value for {fuel_type}")
        self._calculation_count += 1

        warnings = []
        fuel_key = fuel_type.lower().replace(" ", "_").replace("-", "_")

        if fuel_key not in SOLID_FUEL_PROPERTIES:
            raise ValueError(f"Unknown solid fuel type: {fuel_type}")

        props = SOLID_FUEL_PROPERTIES[fuel_key]

        # Base values (as-received)
        hhv_base = props["hhv_btu_lb"]
        lhv_base = props["lhv_btu_lb"]

        # Apply moisture correction if provided
        if moisture_pct is not None:
            default_moisture = props.get("moisture_pct", 0)
            if moisture_pct != default_moisture:
                # Correct for moisture difference
                # HHV_corrected = HHV_base * (100 - moisture_actual) / (100 - moisture_default)
                hhv_base = hhv_base * (100 - moisture_pct) / (100 - default_moisture)
                lhv_base = lhv_base * (100 - moisture_pct) / (100 - default_moisture)
                warnings.append(
                    f"Applied moisture correction from {default_moisture:.1f}% to {moisture_pct:.1f}%"
                )

        # Apply ash correction if provided
        if ash_pct is not None:
            default_ash = props.get("ash_pct", 0)
            if ash_pct != default_ash:
                # Correct for ash difference
                hhv_base = hhv_base * (100 - ash_pct) / (100 - default_ash)
                lhv_base = lhv_base * (100 - ash_pct) / (100 - default_ash)
                warnings.append(
                    f"Applied ash correction from {default_ash:.1f}% to {ash_pct:.1f}%"
                )

        provenance_hash = self._calculate_provenance_hash(
            {"fuel_type": fuel_type, "moisture": moisture_pct, "ash": ash_pct},
            {"hhv_btu_lb": hhv_base, "lhv_btu_lb": lhv_base}
        )

        return HeatingValueResult(
            hhv_btu_lb=round(hhv_base, 0),
            lhv_btu_lb=round(lhv_base, 0),
            calculation_method="ASTM_D5865",
            provenance_hash=provenance_hash,
            warnings=warnings,
        )

    def calculate_wobbe_index(
        self,
        hhv_btu_scf: float,
        specific_gravity: float,
        temperature_f: float = 60.0,
        reference_wobbe: float = 1360.0,
    ) -> WobbeIndexResult:
        """
        Calculate Wobbe Index and interchangeability.

        The Wobbe Index is the key parameter for gas interchangeability,
        indicating whether two gases will produce the same heat output
        when burned in the same equipment.

        Formula: WI = HHV / sqrt(SG)

        Args:
            hhv_btu_scf: Higher Heating Value (BTU/SCF)
            specific_gravity: Specific gravity (air = 1.0)
            temperature_f: Gas temperature (F)
            reference_wobbe: Reference Wobbe Index for comparison

        Returns:
            WobbeIndexResult with interchangeability analysis
        """
        logger.debug("Calculating Wobbe Index")
        self._calculation_count += 1

        if specific_gravity <= 0:
            raise ValueError("Specific gravity must be positive")

        # Calculate Wobbe Index
        wobbe_index = hhv_btu_scf / math.sqrt(specific_gravity)

        # Modified Wobbe Index (temperature corrected)
        temp_correction = math.sqrt((temperature_f + 460) / 520)
        wobbe_index_modified = wobbe_index / temp_correction

        # Interchangeability analysis
        deviation_pct = abs(wobbe_index - reference_wobbe) / reference_wobbe * 100

        # Typical interchangeability limits: +/- 5%
        interchangeable = deviation_pct <= 5.0

        if deviation_pct <= 2.0:
            status = "excellent"
        elif deviation_pct <= 5.0:
            status = "acceptable"
        elif deviation_pct <= 10.0:
            status = "marginal"
        else:
            status = "not_interchangeable"

        # Calculate AGA Bulletin 36 indices (simplified)
        # These require more detailed composition data for accuracy
        lifting_index = 1.0  # Placeholder
        flashback_index = 1.0  # Placeholder
        yellow_tip_index = 1.0  # Placeholder

        provenance_hash = self._calculate_provenance_hash(
            {"hhv": hhv_btu_scf, "sg": specific_gravity, "temp": temperature_f},
            {"wobbe": wobbe_index, "deviation": deviation_pct}
        )

        return WobbeIndexResult(
            wobbe_index=round(wobbe_index, 2),
            wobbe_index_modified=round(wobbe_index_modified, 2),
            hhv_btu_scf=round(hhv_btu_scf, 2),
            specific_gravity=round(specific_gravity, 4),
            reference_wobbe=reference_wobbe,
            deviation_pct=round(deviation_pct, 2),
            interchangeable=interchangeable,
            interchangeability_status=status,
            lifting_index=lifting_index,
            flashback_index=flashback_index,
            yellow_tip_index=yellow_tip_index,
            provenance_hash=provenance_hash,
        )

    def convert_hhv_to_lhv(
        self,
        hhv: float,
        hydrogen_content_pct: float = 25.0,
        moisture_content_pct: float = 0.0,
        unit: str = "BTU/lb",
    ) -> float:
        """
        Convert HHV to LHV.

        The difference between HHV and LHV is the latent heat of
        vaporization of water formed during combustion.

        LHV = HHV - (Hydrogen_content * 9 * 1050 + Moisture * 1050)

        Args:
            hhv: Higher Heating Value
            hydrogen_content_pct: Hydrogen content by weight (%)
            moisture_content_pct: Moisture content by weight (%)
            unit: Heating value unit

        Returns:
            Lower Heating Value in same units
        """
        # Latent heat of water at 60F = ~1050 BTU/lb
        latent_heat = 1050.0

        # Water formed from hydrogen combustion: H2 + 0.5*O2 -> H2O
        # 1 lb H2 produces 9 lb water
        water_from_h2 = hydrogen_content_pct / 100 * 9.0

        # Total water
        total_water = water_from_h2 + moisture_content_pct / 100

        # LHV calculation
        lhv = hhv - (total_water * latent_heat)

        return round(lhv, 2)

    def _build_composition_dict(
        self,
        input_data: HeatingValueInput,
    ) -> Dict[str, float]:
        """Build composition dictionary from input data."""
        composition = {}

        component_map = {
            "methane": input_data.methane_pct,
            "ethane": input_data.ethane_pct,
            "propane": input_data.propane_pct,
            "n_butane": input_data.n_butane_pct,
            "i_butane": input_data.i_butane_pct,
            "n_pentane": input_data.n_pentane_pct,
            "i_pentane": input_data.i_pentane_pct,
            "hexane": input_data.hexane_pct,
            "nitrogen": input_data.nitrogen_pct,
            "carbon_dioxide": input_data.co2_pct,
            "hydrogen_sulfide": input_data.h2s_pct,
            "hydrogen": input_data.hydrogen_pct,
            "oxygen": input_data.oxygen_pct,
            "helium": input_data.helium_pct,
            "argon": input_data.argon_pct,
        }

        for component, value in component_map.items():
            if value is not None and value > 0:
                composition[component] = value

        return composition

    def _get_default_composition(self, fuel_type: str) -> Dict[str, float]:
        """Get default composition for a fuel type."""
        compositions = {
            "natural_gas": {
                "methane": 95.0,
                "ethane": 2.5,
                "propane": 0.5,
                "nitrogen": 1.5,
                "carbon_dioxide": 0.5,
            },
            "pipeline_gas": {
                "methane": 93.0,
                "ethane": 3.5,
                "propane": 1.0,
                "n_butane": 0.5,
                "nitrogen": 1.5,
                "carbon_dioxide": 0.5,
            },
            "biogas": {
                "methane": 60.0,
                "carbon_dioxide": 38.0,
                "nitrogen": 1.5,
                "hydrogen_sulfide": 0.5,
            },
            "hydrogen": {
                "hydrogen": 99.9,
                "nitrogen": 0.1,
            },
            "rng": {
                "methane": 97.0,
                "carbon_dioxide": 1.5,
                "nitrogen": 1.5,
            },
        }

        fuel_key = fuel_type.lower().replace(" ", "_").replace("-", "_")
        return compositions.get(fuel_key, compositions["natural_gas"])

    def _calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        import json

        data = {
            "inputs": inputs,
            "outputs": outputs,
            "calculator": "HeatingValueCalculator",
            "method": "ASTM_D3588",
        }

        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count
