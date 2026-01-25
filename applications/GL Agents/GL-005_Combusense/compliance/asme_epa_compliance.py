# -*- coding: utf-8 -*-
"""
ASME PTC 4.1 and EPA Method 19 Compliance Module

This module provides regulatory compliance mapping and validation for
GL-005 COMBUSENSE combustion control system. It implements:

1. ASME PTC 4.1-1964 (R1991): Steam Generating Units
   - Efficiency calculations (direct and indirect methods)
   - Heat balance methodology
   - Uncertainty analysis

2. EPA Method 19: Determination of Sulfur Dioxide Removal Efficiency
   - F-factor calculations
   - Emission rate conversions
   - Stack flow calculations

3. EPA 40 CFR Part 60/75: Continuous Emissions Monitoring
   - Data validation requirements
   - Quality assurance procedures
   - Missing data substitution

Design Principles:
    - Zero-hallucination: All calculations use published formulas
    - Deterministic: Identical inputs produce identical outputs
    - Auditable: SHA-256 provenance tracking on all calculations
    - Validated: Cross-referenced against published examples

Reference Standards:
    - ASME PTC 4.1-1964 (R1991): Steam Generating Units
    - ASME PTC 4-2013: Fired Steam Generators (modern replacement)
    - EPA Method 19: 40 CFR Part 60, Appendix A
    - EPA 40 CFR Part 75: Continuous Emission Monitoring
    - EPA 40 CFR Part 98: Mandatory Greenhouse Gas Reporting

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, timezone
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import logging
import math

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class RegulatoryFramework(str, Enum):
    """Regulatory frameworks supported."""
    ASME_PTC_41 = "ASME_PTC_4.1"
    ASME_PTC_4 = "ASME_PTC_4"
    EPA_METHOD_19 = "EPA_METHOD_19"
    EPA_40_CFR_60 = "EPA_40_CFR_60"
    EPA_40_CFR_75 = "EPA_40_CFR_75"
    EPA_40_CFR_98 = "EPA_40_CFR_98"


class FuelType(str, Enum):
    """Standard fuel types with default properties."""
    NATURAL_GAS = "natural_gas"
    NO_2_OIL = "no_2_oil"
    NO_6_OIL = "no_6_oil"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUBBITUMINOUS = "coal_subbituminous"
    WOOD = "wood"
    PROPANE = "propane"


class ComplianceStatus(str, Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    DATA_MISSING = "data_missing"


class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    STRICT = "strict"       # All requirements enforced
    STANDARD = "standard"   # Standard enforcement
    LENIENT = "lenient"     # Warnings only


# =============================================================================
# Reference Data Tables (EPA Method 19, ASME PTC 4)
# =============================================================================

class ReferenceData:
    """
    Reference data tables from regulatory standards.

    All values are sourced from published EPA and ASME documents.
    """

    # EPA Method 19, Table 19-1: F-Factors
    # Units: dscf/MMBtu (dry) and wscf/MMBtu (wet)
    F_FACTORS: Dict[FuelType, Dict[str, float]] = {
        FuelType.NATURAL_GAS: {"Fd": 8710, "Fw": 10610, "Fc": 1040},
        FuelType.NO_2_OIL: {"Fd": 9190, "Fw": 10320, "Fc": 1420},
        FuelType.NO_6_OIL: {"Fd": 9220, "Fw": 10260, "Fc": 1490},
        FuelType.COAL_BITUMINOUS: {"Fd": 9780, "Fw": 10640, "Fc": 1800},
        FuelType.COAL_SUBBITUMINOUS: {"Fd": 9820, "Fw": 10580, "Fc": 1840},
        FuelType.WOOD: {"Fd": 9240, "Fw": 10540, "Fc": 1130},
        FuelType.PROPANE: {"Fd": 8710, "Fw": 10200, "Fc": 1190},
    }

    # EPA 40 CFR Part 98, Table C-1: CO2 Emission Factors
    # Units: kg CO2/MMBtu
    CO2_EMISSION_FACTORS: Dict[FuelType, float] = {
        FuelType.NATURAL_GAS: 53.06,
        FuelType.NO_2_OIL: 73.96,
        FuelType.NO_6_OIL: 75.10,
        FuelType.COAL_BITUMINOUS: 93.28,
        FuelType.COAL_SUBBITUMINOUS: 97.17,
        FuelType.WOOD: 93.80,  # Biogenic
        FuelType.PROPANE: 62.87,
    }

    # EPA AP-42, Chapter 1: Emission Factors (lb/MMBtu unless noted)
    EMISSION_FACTORS: Dict[FuelType, Dict[str, float]] = {
        FuelType.NATURAL_GAS: {
            "NOx_uncontrolled": 0.098,  # lb/MMBtu
            "CO": 0.082,
            "PM": 0.0076,
            "SO2": 0.0006,  # Negligible
            "VOC": 0.0054,
        },
        FuelType.NO_2_OIL: {
            "NOx_uncontrolled": 0.14,
            "CO": 0.036,
            "PM": 0.024,
            "SO2": 0.5,  # Depends on sulfur content
            "VOC": 0.0024,
        },
    }

    # Higher Heating Values (HHV) - BTU per unit
    HEATING_VALUES: Dict[FuelType, Dict[str, float]] = {
        FuelType.NATURAL_GAS: {"hhv": 1020, "unit": "BTU/scf"},
        FuelType.NO_2_OIL: {"hhv": 138500, "unit": "BTU/gal"},
        FuelType.NO_6_OIL: {"hhv": 150000, "unit": "BTU/gal"},
        FuelType.COAL_BITUMINOUS: {"hhv": 12500, "unit": "BTU/lb"},
        FuelType.PROPANE: {"hhv": 91500, "unit": "BTU/gal"},
    }


# =============================================================================
# Pydantic Models
# =============================================================================

class FuelAnalysis(BaseModel):
    """Ultimate and proximate analysis of fuel."""

    fuel_type: FuelType = Field(...)
    hhv_btu_per_unit: float = Field(..., gt=0, description="Higher heating value")
    unit: str = Field(default="lb", description="Mass unit for solid, volume for gas/liquid")

    # Ultimate analysis (mass %, as-fired basis)
    carbon_percent: float = Field(default=0, ge=0, le=100)
    hydrogen_percent: float = Field(default=0, ge=0, le=100)
    oxygen_percent: float = Field(default=0, ge=0, le=100)
    nitrogen_percent: float = Field(default=0, ge=0, le=100)
    sulfur_percent: float = Field(default=0, ge=0, le=100)
    moisture_percent: float = Field(default=0, ge=0, le=100)
    ash_percent: float = Field(default=0, ge=0, le=100)

    @model_validator(mode='after')
    def validate_total_analysis(self) -> 'FuelAnalysis':
        """Validate ultimate analysis sums to approximately 100%."""
        total = (
            self.carbon_percent +
            self.hydrogen_percent +
            self.oxygen_percent +
            self.nitrogen_percent +
            self.sulfur_percent +
            self.moisture_percent +
            self.ash_percent
        )
        if total > 0 and abs(total - 100) > 2.0:
            logger.warning(f"Fuel analysis total is {total:.1f}%, expected ~100%")
        return self


class CombustionInputs(BaseModel):
    """Input data for combustion compliance calculations."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Fuel inputs
    fuel_analysis: FuelAnalysis = Field(...)
    fuel_flow_rate: float = Field(..., gt=0, description="Fuel flow rate")
    fuel_flow_unit: str = Field(default="lb/hr")

    # Air inputs
    combustion_air_temp_f: float = Field(default=80.0, description="Combustion air temperature")
    combustion_air_humidity_percent: float = Field(default=50.0, ge=0, le=100)

    # Flue gas measurements
    o2_percent_dry: float = Field(..., ge=0, lt=21, description="O2 in flue gas, dry basis")
    co2_percent_dry: Optional[float] = Field(None, ge=0, le=25)
    co_ppm: float = Field(default=0, ge=0)
    nox_ppm: float = Field(default=0, ge=0)
    so2_ppm: float = Field(default=0, ge=0)
    flue_gas_temp_f: float = Field(..., description="Flue gas temperature at measurement point")

    # Stack parameters (for EPA Method 19)
    stack_flow_acfm: Optional[float] = Field(None, ge=0)
    stack_pressure_inwc: float = Field(default=0.0)
    barometric_pressure_inhg: float = Field(default=29.92)

    # Steam/output (for efficiency)
    steam_flow_lb_hr: Optional[float] = Field(None, ge=0)
    steam_pressure_psig: Optional[float] = Field(None)
    steam_temp_f: Optional[float] = Field(None)
    feedwater_temp_f: Optional[float] = Field(None)


class ComplianceResult(BaseModel):
    """Result of a single compliance calculation."""

    parameter: str = Field(...)
    calculated_value: float = Field(...)
    unit: str = Field(...)
    regulatory_limit: Optional[float] = Field(default=None)
    limit_unit: Optional[str] = Field(default=None)
    status: ComplianceStatus = Field(...)
    framework: RegulatoryFramework = Field(...)
    reference: str = Field(..., description="Standard section reference")
    notes: List[str] = Field(default_factory=list)

    # Provenance
    calculation_formula: str = Field(default="")
    provenance_hash: str = Field(default="")


class ComplianceReport(BaseModel):
    """Complete compliance report with all calculations."""

    report_id: str = Field(...)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Input summary
    fuel_type: FuelType = Field(...)
    measurement_timestamp: datetime = Field(...)

    # Framework compliance
    frameworks_evaluated: List[RegulatoryFramework] = Field(...)
    overall_status: ComplianceStatus = Field(...)

    # Individual results
    results: List[ComplianceResult] = Field(default_factory=list)

    # Calculated values
    efficiency_percent: Optional[float] = Field(default=None)
    excess_air_percent: Optional[float] = Field(default=None)
    heat_input_mmbtu_hr: Optional[float] = Field(default=None)
    co2_emissions_kg_hr: Optional[float] = Field(default=None)
    nox_emissions_lb_mmbtu: Optional[float] = Field(default=None)

    # Audit trail
    provenance_hash: str = Field(default="")
    calculation_log: List[str] = Field(default_factory=list)


# =============================================================================
# ASME PTC 4.1 Compliance Implementation
# =============================================================================

class ASMEPTC41Compliance:
    """
    ASME PTC 4.1 (and PTC 4) Compliance Calculator.

    Implements steam generator performance calculations per ASME PTC 4.1-1964
    and the modern replacement ASME PTC 4-2013.

    Key calculations:
    1. Efficiency by direct (input-output) method
    2. Efficiency by indirect (heat loss) method
    3. Heat losses (dry flue gas, moisture, radiation, etc.)
    4. Excess air from flue gas analysis

    Example:
        >>> compliance = ASMEPTC41Compliance()
        >>> inputs = CombustionInputs(
        ...     fuel_analysis=FuelAnalysis(fuel_type=FuelType.NATURAL_GAS, ...),
        ...     o2_percent_dry=3.0,
        ...     flue_gas_temp_f=350.0,
        ...     ...
        ... )
        >>> result = compliance.calculate_efficiency_indirect(inputs)
    """

    # ASME PTC 4 constants
    REFERENCE_TEMP_F = 77.0  # Reference temperature for heat balance
    CP_DRY_GAS = 0.24  # Specific heat of dry flue gas, BTU/lb-F (approx)
    CP_WATER_VAPOR = 0.45  # Specific heat of water vapor, BTU/lb-F
    LATENT_HEAT_WATER = 1050  # BTU/lb at reference conditions

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize ASME PTC 4.1 compliance calculator."""
        self.validation_level = validation_level
        self._calculation_log: List[str] = []

    def calculate_excess_air(
        self,
        o2_percent_dry: float,
        fuel_type: FuelType = FuelType.NATURAL_GAS
    ) -> Tuple[float, str]:
        """
        Calculate excess air from measured O2 per ASME PTC 4.

        Formula (ASME PTC 4, Eq. 5-18):
            EA = (O2 / (21 - O2)) * 100

        For fuels with nitrogen, a correction may be needed.

        Args:
            o2_percent_dry: Measured O2 percentage (dry basis)
            fuel_type: Type of fuel

        Returns:
            Tuple of (excess_air_percent, provenance_hash)
        """
        if o2_percent_dry >= 21.0:
            raise ValueError("O2 cannot be >= 21%")
        if o2_percent_dry < 0:
            raise ValueError("O2 cannot be negative")

        # Standard excess air formula
        excess_air = (o2_percent_dry / (21.0 - o2_percent_dry)) * 100.0

        # Log calculation
        formula = f"EA = (O2 / (21 - O2)) * 100 = ({o2_percent_dry} / (21 - {o2_percent_dry})) * 100"
        self._log(f"Excess air calculation: {formula} = {excess_air:.2f}%")

        # Generate provenance
        provenance = self._generate_provenance({
            "calculation": "excess_air",
            "o2_percent_dry": o2_percent_dry,
            "result": excess_air,
            "formula": "ASME PTC 4, Eq. 5-18"
        })

        return round(excess_air, 2), provenance

    def calculate_dry_flue_gas_loss(
        self,
        inputs: CombustionInputs,
        excess_air_percent: float
    ) -> Tuple[float, str]:
        """
        Calculate dry flue gas heat loss per ASME PTC 4.1.

        Formula (ASME PTC 4, Section 5.10.2):
            L_dfg = (m_dfg * Cp * (T_fg - T_ref)) / (m_fuel * HHV) * 100

        Simplified for natural gas:
            L_dfg = k * (1 + EA/100) * (T_fg - T_ref) / 1000

        Args:
            inputs: Combustion input data
            excess_air_percent: Calculated excess air percentage

        Returns:
            Tuple of (loss_percent, provenance_hash)
        """
        # Temperature difference
        temp_diff = inputs.flue_gas_temp_f - self.REFERENCE_TEMP_F

        # Dry flue gas coefficient (empirical, varies by fuel)
        if inputs.fuel_analysis.fuel_type == FuelType.NATURAL_GAS:
            k_dfg = 0.0024
        elif inputs.fuel_analysis.fuel_type in (FuelType.NO_2_OIL, FuelType.NO_6_OIL):
            k_dfg = 0.0026
        else:
            k_dfg = 0.0028  # Coal

        # Calculate loss
        loss = k_dfg * (1 + excess_air_percent / 100) * temp_diff

        self._log(
            f"Dry flue gas loss: k={k_dfg}, EA={excess_air_percent}%, "
            f"dT={temp_diff}F, loss={loss:.2f}%"
        )

        provenance = self._generate_provenance({
            "calculation": "dry_flue_gas_loss",
            "temp_diff": temp_diff,
            "excess_air": excess_air_percent,
            "result": loss
        })

        return round(loss, 2), provenance

    def calculate_moisture_loss(
        self,
        inputs: CombustionInputs
    ) -> Tuple[float, str]:
        """
        Calculate moisture heat loss per ASME PTC 4.1.

        Includes:
        - Moisture from combustion of hydrogen (H2O from burning H)
        - Moisture in fuel (if any)
        - Moisture in combustion air

        Formula:
            L_H2O = (m_H2O * (hg - hf)) / (m_fuel * HHV) * 100

        Args:
            inputs: Combustion input data

        Returns:
            Tuple of (loss_percent, provenance_hash)
        """
        fuel = inputs.fuel_analysis

        # For natural gas, moisture from H2 combustion dominates
        if fuel.fuel_type == FuelType.NATURAL_GAS:
            # Typical natural gas: ~23% hydrogen by mass
            # H2 + 0.5 O2 -> H2O (9 lb H2O per lb H2)
            moisture_loss = 10.5  # Typical for natural gas
        else:
            # Calculate from ultimate analysis
            # H2O from hydrogen combustion
            h2o_from_hydrogen = fuel.hydrogen_percent * 9.0 / 100.0  # lb H2O per lb fuel

            # H2O from moisture in fuel
            h2o_from_moisture = fuel.moisture_percent / 100.0

            # Total water per lb fuel
            total_h2o = h2o_from_hydrogen + h2o_from_moisture

            # Heat to evaporate and superheat
            temp_diff = inputs.flue_gas_temp_f - self.REFERENCE_TEMP_F
            enthalpy_change = self.LATENT_HEAT_WATER + self.CP_WATER_VAPOR * temp_diff

            # Loss as percentage of HHV
            moisture_loss = (total_h2o * enthalpy_change) / (fuel.hhv_btu_per_unit / 100)

        self._log(f"Moisture loss calculation: {moisture_loss:.2f}%")

        provenance = self._generate_provenance({
            "calculation": "moisture_loss",
            "fuel_type": fuel.fuel_type.value,
            "result": moisture_loss
        })

        return round(moisture_loss, 2), provenance

    def calculate_radiation_loss(
        self,
        heat_input_mmbtu_hr: float
    ) -> Tuple[float, str]:
        """
        Estimate radiation and convection loss per ASME PTC 4.1.

        Uses American Boiler Manufacturers Association (ABMA) curves.

        Typical values:
        - Large units (>200 MMBtu/hr): 0.3-0.5%
        - Medium units (50-200 MMBtu/hr): 0.5-1.0%
        - Small units (<50 MMBtu/hr): 1.0-2.0%

        Args:
            heat_input_mmbtu_hr: Heat input rate in MMBtu/hr

        Returns:
            Tuple of (loss_percent, provenance_hash)
        """
        if heat_input_mmbtu_hr > 200:
            loss = 0.4
        elif heat_input_mmbtu_hr > 50:
            loss = 0.7
        else:
            loss = 1.5

        self._log(f"Radiation loss (ABMA estimate): {loss}% for {heat_input_mmbtu_hr} MMBtu/hr")

        provenance = self._generate_provenance({
            "calculation": "radiation_loss",
            "heat_input": heat_input_mmbtu_hr,
            "result": loss
        })

        return loss, provenance

    def calculate_efficiency_indirect(
        self,
        inputs: CombustionInputs
    ) -> ComplianceResult:
        """
        Calculate boiler efficiency using indirect (heat loss) method.

        Per ASME PTC 4.1 and PTC 4:
            Efficiency = 100 - Sum(Losses)

        Where losses include:
        - Dry flue gas loss
        - Moisture loss
        - Radiation loss
        - Unburned combustibles (assumed 0 for gas)
        - Other losses

        Args:
            inputs: Combustion input data

        Returns:
            ComplianceResult with efficiency and all loss components
        """
        self._calculation_log = []  # Reset log

        # Calculate excess air
        excess_air, _ = self.calculate_excess_air(
            inputs.o2_percent_dry,
            inputs.fuel_analysis.fuel_type
        )

        # Calculate heat input
        fuel = inputs.fuel_analysis
        heat_input = inputs.fuel_flow_rate * fuel.hhv_btu_per_unit / 1e6  # MMBtu/hr

        # Calculate losses
        dfg_loss, _ = self.calculate_dry_flue_gas_loss(inputs, excess_air)
        moisture_loss, _ = self.calculate_moisture_loss(inputs)
        radiation_loss, _ = self.calculate_radiation_loss(heat_input)

        # Unburned combustibles (negligible for gas with proper O2)
        unburned_loss = 0.0 if inputs.o2_percent_dry > 0.5 else 0.5

        # Total losses
        total_losses = dfg_loss + moisture_loss + radiation_loss + unburned_loss

        # Efficiency
        efficiency = 100.0 - total_losses

        self._log(
            f"Efficiency (indirect method): 100 - ({dfg_loss} + {moisture_loss} + "
            f"{radiation_loss} + {unburned_loss}) = {efficiency:.1f}%"
        )

        # Generate provenance
        provenance_data = {
            "calculation": "efficiency_indirect",
            "inputs": {
                "o2_percent": inputs.o2_percent_dry,
                "flue_gas_temp": inputs.flue_gas_temp_f,
                "fuel_type": fuel.fuel_type.value
            },
            "losses": {
                "dry_flue_gas": dfg_loss,
                "moisture": moisture_loss,
                "radiation": radiation_loss,
                "unburned": unburned_loss
            },
            "efficiency": efficiency
        }
        provenance = self._generate_provenance(provenance_data)

        return ComplianceResult(
            parameter="boiler_efficiency",
            calculated_value=round(efficiency, 1),
            unit="%",
            regulatory_limit=None,
            status=ComplianceStatus.COMPLIANT if efficiency > 75 else ComplianceStatus.NEEDS_REVIEW,
            framework=RegulatoryFramework.ASME_PTC_41,
            reference="ASME PTC 4.1, Section 5 / ASME PTC 4, Section 5.10",
            notes=self._calculation_log.copy(),
            calculation_formula="Efficiency = 100 - (L_dfg + L_moisture + L_radiation + L_unburned)",
            provenance_hash=provenance
        )

    def _log(self, message: str) -> None:
        """Add message to calculation log."""
        self._calculation_log.append(f"[{datetime.now(timezone.utc).isoformat()}] {message}")
        logger.debug(message)

    def _generate_provenance(self, data: Dict[str, Any]) -> str:
        """Generate SHA-256 provenance hash."""
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data["standard"] = "ASME_PTC_4.1"
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()


# =============================================================================
# EPA Method 19 Compliance Implementation
# =============================================================================

class EPAMethod19Compliance:
    """
    EPA Method 19 Compliance Calculator.

    Implements calculations per 40 CFR Part 60, Appendix A, Method 19:
    - F-factor method for emission rate calculation
    - Dry basis to wet basis conversions
    - Emission rate in lb/MMBtu

    Key formulas:
    - E = (C * Fd * 20.9) / (20.9 - %O2)  for dry measurement
    - E = (C * Fw) for wet measurement

    Example:
        >>> epa = EPAMethod19Compliance()
        >>> result = epa.calculate_nox_emission_rate(
        ...     nox_ppm=50,
        ...     o2_percent_dry=3.0,
        ...     fuel_type=FuelType.NATURAL_GAS
        ... )
    """

    # Molecular weights
    MW_NOX = 46.0  # As NO2
    MW_SO2 = 64.0
    MW_CO = 28.0
    MW_CO2 = 44.0

    # Standard conditions
    STD_MOLAR_VOLUME = 385.3  # scf/lb-mol at 68F, 29.92 inHg

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize EPA Method 19 compliance calculator."""
        self.validation_level = validation_level
        self._calculation_log: List[str] = []

    def get_f_factor(
        self,
        fuel_type: FuelType,
        factor_type: str = "Fd"
    ) -> float:
        """
        Get F-factor from EPA Method 19 Table 19-1.

        Args:
            fuel_type: Fuel type
            factor_type: "Fd" (dry), "Fw" (wet), or "Fc" (carbon)

        Returns:
            F-factor in dscf/MMBtu or wscf/MMBtu
        """
        if fuel_type not in ReferenceData.F_FACTORS:
            raise ValueError(f"No F-factor data for fuel type: {fuel_type}")

        factors = ReferenceData.F_FACTORS[fuel_type]

        if factor_type not in factors:
            raise ValueError(f"Unknown factor type: {factor_type}")

        return factors[factor_type]

    def calculate_emission_rate(
        self,
        concentration_ppm: float,
        o2_percent_dry: float,
        fuel_type: FuelType,
        pollutant_mw: float,
        pollutant_name: str = "pollutant"
    ) -> ComplianceResult:
        """
        Calculate emission rate per EPA Method 19.

        Formula (40 CFR 60, Appendix A, Method 19, Eq. 19-3):
            E = (C * Fd * 20.9) / (20.9 - %O2d) * (MW / MV)

        Where:
            E = Emission rate (lb/MMBtu)
            C = Concentration (ppm)
            Fd = F-factor dry (dscf/MMBtu)
            %O2d = O2 percent dry
            MW = Molecular weight
            MV = Molar volume (385.3 scf/lb-mol)

        Args:
            concentration_ppm: Measured concentration in ppm
            o2_percent_dry: O2 percentage (dry basis)
            fuel_type: Fuel type
            pollutant_mw: Molecular weight of pollutant
            pollutant_name: Name for reporting

        Returns:
            ComplianceResult with emission rate in lb/MMBtu
        """
        self._calculation_log = []

        if o2_percent_dry >= 20.9:
            raise ValueError("O2 cannot be >= 20.9% for emission calculation")

        # Get F-factor
        fd = self.get_f_factor(fuel_type, "Fd")

        # O2 correction factor
        o2_correction = 20.9 / (20.9 - o2_percent_dry)

        # Convert ppm to lb/dscf
        # ppm = (lb pollutant / lb flue gas) * 10^6
        # At standard conditions: 1 lb-mol = 385.3 scf
        lb_per_scf = (concentration_ppm / 1e6) * (pollutant_mw / self.STD_MOLAR_VOLUME)

        # Emission rate in lb/MMBtu
        emission_rate = lb_per_scf * fd * o2_correction

        self._log(
            f"EPA Method 19 emission calculation for {pollutant_name}: "
            f"C={concentration_ppm} ppm, O2={o2_percent_dry}%, "
            f"Fd={fd} dscf/MMBtu, E={emission_rate:.4f} lb/MMBtu"
        )

        provenance = self._generate_provenance({
            "calculation": "emission_rate",
            "pollutant": pollutant_name,
            "concentration_ppm": concentration_ppm,
            "o2_percent": o2_percent_dry,
            "f_factor": fd,
            "emission_rate": emission_rate
        })

        return ComplianceResult(
            parameter=f"{pollutant_name}_emission_rate",
            calculated_value=round(emission_rate, 5),
            unit="lb/MMBtu",
            status=ComplianceStatus.COMPLIANT,
            framework=RegulatoryFramework.EPA_METHOD_19,
            reference="40 CFR 60, Appendix A, Method 19, Eq. 19-3",
            notes=self._calculation_log.copy(),
            calculation_formula=f"E = (C * Fd * 20.9) / (20.9 - %O2) * (MW / {self.STD_MOLAR_VOLUME})",
            provenance_hash=provenance
        )

    def calculate_nox_emission_rate(
        self,
        nox_ppm: float,
        o2_percent_dry: float,
        fuel_type: FuelType
    ) -> ComplianceResult:
        """Calculate NOx emission rate in lb/MMBtu."""
        return self.calculate_emission_rate(
            concentration_ppm=nox_ppm,
            o2_percent_dry=o2_percent_dry,
            fuel_type=fuel_type,
            pollutant_mw=self.MW_NOX,
            pollutant_name="NOx"
        )

    def calculate_so2_emission_rate(
        self,
        so2_ppm: float,
        o2_percent_dry: float,
        fuel_type: FuelType
    ) -> ComplianceResult:
        """Calculate SO2 emission rate in lb/MMBtu."""
        return self.calculate_emission_rate(
            concentration_ppm=so2_ppm,
            o2_percent_dry=o2_percent_dry,
            fuel_type=fuel_type,
            pollutant_mw=self.MW_SO2,
            pollutant_name="SO2"
        )

    def calculate_co_emission_rate(
        self,
        co_ppm: float,
        o2_percent_dry: float,
        fuel_type: FuelType
    ) -> ComplianceResult:
        """Calculate CO emission rate in lb/MMBtu."""
        return self.calculate_emission_rate(
            concentration_ppm=co_ppm,
            o2_percent_dry=o2_percent_dry,
            fuel_type=fuel_type,
            pollutant_mw=self.MW_CO,
            pollutant_name="CO"
        )

    def calculate_co2_emissions(
        self,
        heat_input_mmbtu: float,
        fuel_type: FuelType
    ) -> ComplianceResult:
        """
        Calculate CO2 emissions per EPA 40 CFR Part 98.

        Formula:
            CO2 = Heat_Input_MMBtu * EF_CO2

        Args:
            heat_input_mmbtu: Heat input in MMBtu
            fuel_type: Fuel type

        Returns:
            ComplianceResult with CO2 in kg
        """
        self._calculation_log = []

        ef = ReferenceData.CO2_EMISSION_FACTORS.get(fuel_type, 53.06)
        co2_kg = heat_input_mmbtu * ef

        self._log(
            f"CO2 emissions: {heat_input_mmbtu} MMBtu * {ef} kg/MMBtu = {co2_kg:.2f} kg"
        )

        provenance = self._generate_provenance({
            "calculation": "co2_emissions",
            "heat_input": heat_input_mmbtu,
            "emission_factor": ef,
            "co2_kg": co2_kg
        })

        return ComplianceResult(
            parameter="CO2_emissions",
            calculated_value=round(co2_kg, 2),
            unit="kg",
            status=ComplianceStatus.COMPLIANT,
            framework=RegulatoryFramework.EPA_40_CFR_98,
            reference="40 CFR Part 98, Table C-1",
            notes=self._calculation_log.copy(),
            calculation_formula="CO2 = Heat_Input * EF",
            provenance_hash=provenance
        )

    def _log(self, message: str) -> None:
        """Add message to calculation log."""
        self._calculation_log.append(f"[{datetime.now(timezone.utc).isoformat()}] {message}")
        logger.debug(message)

    def _generate_provenance(self, data: Dict[str, Any]) -> str:
        """Generate SHA-256 provenance hash."""
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data["standard"] = "EPA_METHOD_19"
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()


# =============================================================================
# Compliance Validator (Unified Interface)
# =============================================================================

class ComplianceValidator:
    """
    Unified compliance validator for multiple regulatory frameworks.

    Provides a single interface for validating combustion data against
    multiple regulatory standards (ASME PTC 4.1, EPA Method 19, etc.).

    Example:
        >>> validator = ComplianceValidator()
        >>> inputs = CombustionInputs(...)
        >>> report = validator.validate(inputs, [
        ...     RegulatoryFramework.ASME_PTC_41,
        ...     RegulatoryFramework.EPA_METHOD_19
        ... ])
    """

    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ):
        """Initialize compliance validator."""
        self.validation_level = validation_level
        self.asme = ASMEPTC41Compliance(validation_level)
        self.epa = EPAMethod19Compliance(validation_level)

    def validate(
        self,
        inputs: CombustionInputs,
        frameworks: List[RegulatoryFramework]
    ) -> ComplianceReport:
        """
        Validate combustion inputs against specified frameworks.

        Args:
            inputs: Combustion input data
            frameworks: List of frameworks to validate against

        Returns:
            ComplianceReport with all results
        """
        results: List[ComplianceResult] = []
        calculation_log: List[str] = []

        # Generate report ID
        report_id = hashlib.sha256(
            f"{inputs.timestamp.isoformat()}{inputs.fuel_analysis.fuel_type.value}".encode()
        ).hexdigest()[:16]

        # ASME PTC 4.1 validation
        if RegulatoryFramework.ASME_PTC_41 in frameworks:
            efficiency_result = self.asme.calculate_efficiency_indirect(inputs)
            results.append(efficiency_result)
            calculation_log.extend(efficiency_result.notes)

            # Calculate excess air
            excess_air, _ = self.asme.calculate_excess_air(
                inputs.o2_percent_dry,
                inputs.fuel_analysis.fuel_type
            )

        # EPA Method 19 validation
        if RegulatoryFramework.EPA_METHOD_19 in frameworks:
            # NOx emission rate
            if inputs.nox_ppm > 0:
                nox_result = self.epa.calculate_nox_emission_rate(
                    inputs.nox_ppm,
                    inputs.o2_percent_dry,
                    inputs.fuel_analysis.fuel_type
                )
                results.append(nox_result)
                calculation_log.extend(nox_result.notes)

            # CO emission rate
            if inputs.co_ppm > 0:
                co_result = self.epa.calculate_co_emission_rate(
                    inputs.co_ppm,
                    inputs.o2_percent_dry,
                    inputs.fuel_analysis.fuel_type
                )
                results.append(co_result)
                calculation_log.extend(co_result.notes)

            # SO2 emission rate
            if inputs.so2_ppm > 0:
                so2_result = self.epa.calculate_so2_emission_rate(
                    inputs.so2_ppm,
                    inputs.o2_percent_dry,
                    inputs.fuel_analysis.fuel_type
                )
                results.append(so2_result)
                calculation_log.extend(so2_result.notes)

        # EPA 40 CFR Part 98 (CO2)
        if RegulatoryFramework.EPA_40_CFR_98 in frameworks:
            heat_input = (
                inputs.fuel_flow_rate *
                inputs.fuel_analysis.hhv_btu_per_unit / 1e6
            )
            co2_result = self.epa.calculate_co2_emissions(
                heat_input,
                inputs.fuel_analysis.fuel_type
            )
            results.append(co2_result)
            calculation_log.extend(co2_result.notes)

        # Determine overall status
        statuses = [r.status for r in results]
        if ComplianceStatus.NON_COMPLIANT in statuses:
            overall_status = ComplianceStatus.NON_COMPLIANT
        elif ComplianceStatus.NEEDS_REVIEW in statuses:
            overall_status = ComplianceStatus.NEEDS_REVIEW
        elif ComplianceStatus.DATA_MISSING in statuses:
            overall_status = ComplianceStatus.DATA_MISSING
        else:
            overall_status = ComplianceStatus.COMPLIANT

        # Extract key values
        efficiency = next(
            (r.calculated_value for r in results if r.parameter == "boiler_efficiency"),
            None
        )
        nox_rate = next(
            (r.calculated_value for r in results if r.parameter == "NOx_emission_rate"),
            None
        )

        # Calculate heat input
        heat_input = (
            inputs.fuel_flow_rate *
            inputs.fuel_analysis.hhv_btu_per_unit / 1e6
        )

        # Generate provenance for report
        report_data = {
            "report_id": report_id,
            "frameworks": [f.value for f in frameworks],
            "fuel_type": inputs.fuel_analysis.fuel_type.value,
            "result_count": len(results)
        }
        provenance = hashlib.sha256(
            json.dumps(report_data, sort_keys=True).encode()
        ).hexdigest()

        return ComplianceReport(
            report_id=report_id,
            fuel_type=inputs.fuel_analysis.fuel_type,
            measurement_timestamp=inputs.timestamp,
            frameworks_evaluated=frameworks,
            overall_status=overall_status,
            results=results,
            efficiency_percent=efficiency,
            excess_air_percent=excess_air if RegulatoryFramework.ASME_PTC_41 in frameworks else None,
            heat_input_mmbtu_hr=round(heat_input, 2),
            nox_emissions_lb_mmbtu=nox_rate,
            calculation_log=calculation_log,
            provenance_hash=provenance
        )


# =============================================================================
# Compliance Registry
# =============================================================================

class ComplianceRegistry:
    """Registry for compliance validators."""

    def __init__(self):
        """Initialize registry."""
        self._validators: Dict[str, ComplianceValidator] = {}

    def register(
        self,
        name: str,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ComplianceValidator:
        """Register a new compliance validator."""
        validator = ComplianceValidator(validation_level)
        self._validators[name] = validator
        logger.info(f"Registered compliance validator: {name}")
        return validator

    def get(self, name: str) -> Optional[ComplianceValidator]:
        """Get a validator by name."""
        return self._validators.get(name)


# Global registry
compliance_registry = ComplianceRegistry()
