"""
GL-018 FLUEFLOW - Emissions Calculator

Zero-hallucination, deterministic calculations for emissions analysis
and concentration conversions following EPA standards.

This module provides:
- NOx, CO, SO2 concentration conversions (ppm to mg/Nm³)
- Emission factors calculation
- EPA compliance checks
- CO/CO2 ratio analysis

Standards Reference:
- EPA Method 19 - Determination of SO2 Removal Efficiency
- EPA 40 CFR Part 60 - Standards of Performance for New Stationary Sources
- EN 14181 - Quality Assurance of Automated Measuring Systems
- ISO 10780 - Stationary Source Emissions - Measurement of Velocity

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import math

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

# Molecular weights (g/mol)
MW_O2 = 32.0
MW_N2 = 28.0
MW_CO2 = 44.0
MW_H2O = 18.0
MW_CO = 28.0
MW_SO2 = 64.0
MW_NOX = 46.0  # As NO2 equivalent
MW_NO = 30.0
MW_NO2 = 46.0

# Standard conditions for emissions
STANDARD_TEMP_K = 273.15  # 0°C
STANDARD_PRESSURE_PA = 101325  # 1 atm

# Molar volume at STP
MOLAR_VOLUME_NM3_KMOL = 22.414  # Nm³/kmol

# Reference oxygen for emissions corrections
REFERENCE_O2_PERCENT = {
    "combustion_turbines": 15.0,
    "boilers": 3.0,
    "incinerators": 11.0,
    "glass_furnaces": 8.0,
}

# EPA emission limits (example values, ppm dry @ 3% O2)
EPA_LIMITS_PPM = {
    "NOx_boiler": 200.0,
    "CO_boiler": 400.0,
    "SO2_boiler": 500.0,
}


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class EmissionsInput:
    """
    Input parameters for emissions calculations.

    Attributes:
        NOx_ppm: NOx concentration (ppm, dry basis)
        CO_ppm: CO concentration (ppm, dry basis)
        SO2_ppm: SO2 concentration (ppm, dry basis)
        CO2_pct: CO2 concentration (%, dry basis)
        O2_pct: O2 concentration (%, dry basis)
        flue_gas_temp_c: Flue gas temperature (°C)
        flue_gas_pressure_pa: Flue gas pressure (Pa)
        flue_gas_flow_nm3_hr: Flue gas flow rate (Nm³/hr, dry)
        fuel_type: Type of fuel being burned
        reference_O2_pct: Reference O2 for corrections (default 3%)
        moisture_pct: Moisture content in flue gas (%)
    """
    NOx_ppm: float
    CO_ppm: float
    SO2_ppm: float
    CO2_pct: float
    O2_pct: float
    flue_gas_temp_c: float
    flue_gas_flow_nm3_hr: float
    fuel_type: str
    flue_gas_pressure_pa: float = STANDARD_PRESSURE_PA
    reference_O2_pct: float = 3.0
    moisture_pct: float = 10.0


@dataclass(frozen=True)
class EmissionsOutput:
    """
    Output results from emissions calculations.

    Attributes:
        NOx_mg_nm3: NOx concentration (mg/Nm³, dry)
        CO_mg_nm3: CO concentration (mg/Nm³, dry)
        SO2_mg_nm3: SO2 concentration (mg/Nm³, dry)
        NOx_mg_nm3_corrected: NOx at reference O2 (mg/Nm³)
        CO_mg_nm3_corrected: CO at reference O2 (mg/Nm³)
        SO2_mg_nm3_corrected: SO2 at reference O2 (mg/Nm³)
        NOx_kg_hr: NOx mass emission rate (kg/hr)
        CO_kg_hr: CO mass emission rate (kg/hr)
        SO2_kg_hr: SO2 mass emission rate (kg/hr)
        CO_CO2_ratio: CO/CO2 ratio (indicator of combustion quality)
        NOx_compliance_status: EPA compliance status for NOx
        CO_compliance_status: EPA compliance status for CO
        SO2_compliance_status: EPA compliance status for SO2
        emission_factor_NOx_g_GJ: NOx emission factor (g/GJ)
        emission_factor_CO_g_GJ: CO emission factor (g/GJ)
        emission_factor_SO2_g_GJ: SO2 emission factor (g/GJ)
    """
    NOx_mg_nm3: float
    CO_mg_nm3: float
    SO2_mg_nm3: float
    NOx_mg_nm3_corrected: float
    CO_mg_nm3_corrected: float
    SO2_mg_nm3_corrected: float
    NOx_kg_hr: float
    CO_kg_hr: float
    SO2_kg_hr: float
    CO_CO2_ratio: float
    NOx_compliance_status: str
    CO_compliance_status: str
    SO2_compliance_status: str
    emission_factor_NOx_g_GJ: float
    emission_factor_CO_g_GJ: float
    emission_factor_SO2_g_GJ: float


# =============================================================================
# EMISSIONS CALCULATOR CLASS
# =============================================================================

class EmissionsCalculator:
    """
    Zero-hallucination emissions calculator.

    Implements deterministic calculations following EPA Method 19
    and EN standards for emissions analysis. All calculations produce
    bit-perfect reproducible results with complete provenance.

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> calculator = EmissionsCalculator()
        >>> inputs = EmissionsInput(
        ...     NOx_ppm=150.0,
        ...     CO_ppm=50.0,
        ...     SO2_ppm=100.0,
        ...     CO2_pct=12.0,
        ...     O2_pct=3.5,
        ...     flue_gas_temp_c=180.0,
        ...     flue_gas_flow_nm3_hr=50000.0,
        ...     fuel_type="Natural Gas"
        ... )
        >>> result, provenance = calculator.calculate(inputs)
        >>> print(f"NOx: {result.NOx_mg_nm3:.1f} mg/Nm³")
    """

    VERSION = "1.0.0"
    NAME = "EmissionsCalculator"

    def __init__(self):
        """Initialize the emissions calculator."""
        self._tracker: Optional[ProvenanceTracker] = None

    def calculate(
        self,
        inputs: EmissionsInput
    ) -> Tuple[EmissionsOutput, ProvenanceRecord]:
        """
        Perform complete emissions analysis.

        Args:
            inputs: EmissionsInput with all required parameters

        Returns:
            Tuple of (EmissionsOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["EPA Method 19", "EN 14181", "40 CFR Part 60"],
                "domain": "Emissions Analysis"
            }
        )

        # Set inputs for provenance
        input_dict = {
            "NOx_ppm": inputs.NOx_ppm,
            "CO_ppm": inputs.CO_ppm,
            "SO2_ppm": inputs.SO2_ppm,
            "CO2_pct": inputs.CO2_pct,
            "O2_pct": inputs.O2_pct,
            "flue_gas_temp_c": inputs.flue_gas_temp_c,
            "flue_gas_pressure_pa": inputs.flue_gas_pressure_pa,
            "flue_gas_flow_nm3_hr": inputs.flue_gas_flow_nm3_hr,
            "fuel_type": inputs.fuel_type,
            "reference_O2_pct": inputs.reference_O2_pct,
            "moisture_pct": inputs.moisture_pct
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Step 1: Convert ppm to mg/Nm³
        NOx_mg_nm3 = self._convert_ppm_to_mg_nm3(inputs.NOx_ppm, MW_NOX, "NOx")
        CO_mg_nm3 = self._convert_ppm_to_mg_nm3(inputs.CO_ppm, MW_CO, "CO")
        SO2_mg_nm3 = self._convert_ppm_to_mg_nm3(inputs.SO2_ppm, MW_SO2, "SO2")

        # Step 2: Correct to reference O2
        NOx_corrected = self._correct_to_reference_O2(
            NOx_mg_nm3,
            inputs.O2_pct,
            inputs.reference_O2_pct,
            "NOx"
        )

        CO_corrected = self._correct_to_reference_O2(
            CO_mg_nm3,
            inputs.O2_pct,
            inputs.reference_O2_pct,
            "CO"
        )

        SO2_corrected = self._correct_to_reference_O2(
            SO2_mg_nm3,
            inputs.O2_pct,
            inputs.reference_O2_pct,
            "SO2"
        )

        # Step 3: Calculate mass emission rates
        NOx_kg_hr = self._calculate_mass_emission_rate(
            NOx_mg_nm3,
            inputs.flue_gas_flow_nm3_hr,
            "NOx"
        )

        CO_kg_hr = self._calculate_mass_emission_rate(
            CO_mg_nm3,
            inputs.flue_gas_flow_nm3_hr,
            "CO"
        )

        SO2_kg_hr = self._calculate_mass_emission_rate(
            SO2_mg_nm3,
            inputs.flue_gas_flow_nm3_hr,
            "SO2"
        )

        # Step 4: Calculate CO/CO2 ratio
        co_co2_ratio = self._calculate_CO_CO2_ratio(
            inputs.CO_ppm,
            inputs.CO2_pct
        )

        # Step 5: Check EPA compliance
        NOx_compliance = self._check_compliance(
            NOx_corrected,
            EPA_LIMITS_PPM.get("NOx_boiler", 200.0) * (MW_NOX / MOLAR_VOLUME_NM3_KMOL),
            "NOx"
        )

        CO_compliance = self._check_compliance(
            CO_corrected,
            EPA_LIMITS_PPM.get("CO_boiler", 400.0) * (MW_CO / MOLAR_VOLUME_NM3_KMOL),
            "CO"
        )

        SO2_compliance = self._check_compliance(
            SO2_corrected,
            EPA_LIMITS_PPM.get("SO2_boiler", 500.0) * (MW_SO2 / MOLAR_VOLUME_NM3_KMOL),
            "SO2"
        )

        # Step 6: Calculate emission factors (g/GJ)
        # Simplified - would need fuel flow and LHV for accurate calculation
        ef_NOx = (NOx_kg_hr / 50.0) * 1000.0  # Assuming 50 GJ/hr typical
        ef_CO = (CO_kg_hr / 50.0) * 1000.0
        ef_SO2 = (SO2_kg_hr / 50.0) * 1000.0

        # Create output
        output = EmissionsOutput(
            NOx_mg_nm3=round(NOx_mg_nm3, 2),
            CO_mg_nm3=round(CO_mg_nm3, 2),
            SO2_mg_nm3=round(SO2_mg_nm3, 2),
            NOx_mg_nm3_corrected=round(NOx_corrected, 2),
            CO_mg_nm3_corrected=round(CO_corrected, 2),
            SO2_mg_nm3_corrected=round(SO2_corrected, 2),
            NOx_kg_hr=round(NOx_kg_hr, 3),
            CO_kg_hr=round(CO_kg_hr, 3),
            SO2_kg_hr=round(SO2_kg_hr, 3),
            CO_CO2_ratio=round(co_co2_ratio, 4),
            NOx_compliance_status=NOx_compliance,
            CO_compliance_status=CO_compliance,
            SO2_compliance_status=SO2_compliance,
            emission_factor_NOx_g_GJ=round(ef_NOx, 1),
            emission_factor_CO_g_GJ=round(ef_CO, 1),
            emission_factor_SO2_g_GJ=round(ef_SO2, 1)
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "NOx_mg_nm3": output.NOx_mg_nm3,
            "CO_mg_nm3": output.CO_mg_nm3,
            "SO2_mg_nm3": output.SO2_mg_nm3,
            "NOx_mg_nm3_corrected": output.NOx_mg_nm3_corrected,
            "CO_mg_nm3_corrected": output.CO_mg_nm3_corrected,
            "SO2_mg_nm3_corrected": output.SO2_mg_nm3_corrected,
            "NOx_kg_hr": output.NOx_kg_hr,
            "CO_kg_hr": output.CO_kg_hr,
            "SO2_kg_hr": output.SO2_kg_hr,
            "CO_CO2_ratio": output.CO_CO2_ratio,
            "NOx_compliance_status": output.NOx_compliance_status,
            "CO_compliance_status": output.CO_compliance_status,
            "SO2_compliance_status": output.SO2_compliance_status,
            "emission_factor_NOx_g_GJ": output.emission_factor_NOx_g_GJ,
            "emission_factor_CO_g_GJ": output.emission_factor_CO_g_GJ,
            "emission_factor_SO2_g_GJ": output.emission_factor_SO2_g_GJ
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _validate_inputs(self, inputs: EmissionsInput) -> None:
        """Validate input parameters."""
        if inputs.NOx_ppm < 0:
            raise ValueError("NOx concentration cannot be negative")

        if inputs.CO_ppm < 0:
            raise ValueError("CO concentration cannot be negative")

        if inputs.SO2_ppm < 0:
            raise ValueError("SO2 concentration cannot be negative")

        if inputs.O2_pct < 0 or inputs.O2_pct > 21:
            raise ValueError(f"O2 concentration out of range: {inputs.O2_pct}%")

        if inputs.flue_gas_flow_nm3_hr <= 0:
            raise ValueError("Flue gas flow rate must be positive")

    def _convert_ppm_to_mg_nm3(
        self,
        concentration_ppm: float,
        molecular_weight: float,
        species: str
    ) -> float:
        """
        Convert concentration from ppm to mg/Nm³.

        Formula (at STP):
            C[mg/Nm³] = C[ppm] × MW / Vm

        Where:
            MW = molecular weight (g/mol)
            Vm = molar volume at STP = 22.414 Nm³/kmol

        Args:
            concentration_ppm: Concentration in ppm (volumetric)
            molecular_weight: Molecular weight (g/mol)
            species: Name of species (for tracking)

        Returns:
            Concentration in mg/Nm³
        """
        # Convert ppm to mg/Nm³
        # ppm = cm³/m³ = 10^-6 m³/m³
        # At STP: 1 kmol = 22.414 Nm³
        # So: C[mg/Nm³] = C[ppm] × MW[g/mol] / 22.414[Nm³/kmol] × 1000[mg/g] / 1000[mol/kmol]

        concentration_mg_nm3 = concentration_ppm * molecular_weight / MOLAR_VOLUME_NM3_KMOL

        self._tracker.add_step(
            step_number=len(self._tracker._steps) + 1,
            description=f"Convert {species} from ppm to mg/Nm³",
            operation="unit_conversion",
            inputs={
                f"{species}_ppm": concentration_ppm,
                "molecular_weight": molecular_weight,
                "molar_volume_nm3_kmol": MOLAR_VOLUME_NM3_KMOL
            },
            output_value=concentration_mg_nm3,
            output_name=f"{species}_mg_nm3",
            formula="C[mg/Nm³] = C[ppm] × MW / 22.414"
        )

        return concentration_mg_nm3

    def _correct_to_reference_O2(
        self,
        concentration_mg_nm3: float,
        measured_O2_pct: float,
        reference_O2_pct: float,
        species: str
    ) -> float:
        """
        Correct concentration to reference O2 level.

        Formula (EPA Method 19):
            C_ref = C_measured × (21 - O2_ref) / (21 - O2_measured)

        This correction normalizes emissions to a standard O2 level,
        accounting for dilution effects of excess air.

        Args:
            concentration_mg_nm3: Measured concentration (mg/Nm³)
            measured_O2_pct: Measured O2 (%)
            reference_O2_pct: Reference O2 for correction (%)
            species: Name of species

        Returns:
            Corrected concentration at reference O2 (mg/Nm³)
        """
        correction_factor = (21.0 - reference_O2_pct) / (21.0 - measured_O2_pct)
        concentration_corrected = concentration_mg_nm3 * correction_factor

        self._tracker.add_step(
            step_number=len(self._tracker._steps) + 1,
            description=f"Correct {species} to reference O2",
            operation="O2_correction",
            inputs={
                f"{species}_mg_nm3": concentration_mg_nm3,
                "measured_O2_pct": measured_O2_pct,
                "reference_O2_pct": reference_O2_pct,
                "correction_factor": correction_factor
            },
            output_value=concentration_corrected,
            output_name=f"{species}_mg_nm3_corrected",
            formula="C_ref = C × (21 - O2_ref) / (21 - O2_meas)"
        )

        return concentration_corrected

    def _calculate_mass_emission_rate(
        self,
        concentration_mg_nm3: float,
        flow_rate_nm3_hr: float,
        species: str
    ) -> float:
        """
        Calculate mass emission rate.

        Formula:
            E[kg/hr] = C[mg/Nm³] × Q[Nm³/hr] / 1,000,000

        Args:
            concentration_mg_nm3: Concentration (mg/Nm³)
            flow_rate_nm3_hr: Flue gas flow rate (Nm³/hr)
            species: Name of species

        Returns:
            Mass emission rate (kg/hr)
        """
        # Convert mg/hr to kg/hr
        emission_rate_kg_hr = (concentration_mg_nm3 * flow_rate_nm3_hr) / 1_000_000.0

        self._tracker.add_step(
            step_number=len(self._tracker._steps) + 1,
            description=f"Calculate {species} mass emission rate",
            operation="mass_flow_calc",
            inputs={
                f"{species}_mg_nm3": concentration_mg_nm3,
                "flow_rate_nm3_hr": flow_rate_nm3_hr
            },
            output_value=emission_rate_kg_hr,
            output_name=f"{species}_kg_hr",
            formula="E[kg/hr] = C[mg/Nm³] × Q[Nm³/hr] / 1,000,000"
        )

        return emission_rate_kg_hr

    def _calculate_CO_CO2_ratio(
        self,
        CO_ppm: float,
        CO2_pct: float
    ) -> float:
        """
        Calculate CO/CO2 ratio (combustion quality indicator).

        Low ratio indicates good combustion efficiency.
        Typical values:
        - Excellent: < 0.001 (< 10 ppm CO @ 1% CO2)
        - Good: 0.001 - 0.004
        - Fair: 0.004 - 0.01
        - Poor: > 0.01

        Args:
            CO_ppm: CO concentration (ppm)
            CO2_pct: CO2 concentration (%)

        Returns:
            CO/CO2 ratio (dimensionless)
        """
        # Convert both to same basis (ppm)
        CO2_ppm = CO2_pct * 10000.0

        # Calculate ratio
        if CO2_ppm > 0:
            ratio = CO_ppm / CO2_ppm
        else:
            ratio = 0.0

        self._tracker.add_step(
            step_number=len(self._tracker._steps) + 1,
            description="Calculate CO/CO2 ratio",
            operation="ratio_calc",
            inputs={
                "CO_ppm": CO_ppm,
                "CO2_pct": CO2_pct,
                "CO2_ppm": CO2_ppm
            },
            output_value=ratio,
            output_name="CO_CO2_ratio",
            formula="Ratio = CO[ppm] / CO2[ppm]"
        )

        return ratio

    def _check_compliance(
        self,
        concentration_mg_nm3: float,
        limit_mg_nm3: float,
        species: str
    ) -> str:
        """
        Check EPA compliance status.

        Args:
            concentration_mg_nm3: Measured concentration (mg/Nm³)
            limit_mg_nm3: Regulatory limit (mg/Nm³)
            species: Name of species

        Returns:
            Compliance status string
        """
        if concentration_mg_nm3 <= limit_mg_nm3 * 0.8:
            status = "Compliant (Good Margin)"
        elif concentration_mg_nm3 <= limit_mg_nm3:
            status = "Compliant"
        elif concentration_mg_nm3 <= limit_mg_nm3 * 1.1:
            status = "Near Limit (Warning)"
        else:
            status = "Non-Compliant"

        self._tracker.add_step(
            step_number=len(self._tracker._steps) + 1,
            description=f"Check {species} EPA compliance",
            operation="compliance_check",
            inputs={
                f"{species}_mg_nm3": concentration_mg_nm3,
                "limit_mg_nm3": limit_mg_nm3,
                "threshold_80pct": limit_mg_nm3 * 0.8,
                "threshold_110pct": limit_mg_nm3 * 1.1
            },
            output_value=status,
            output_name=f"{species}_compliance_status",
            formula="Status based on percentage of limit"
        )

        return status


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def convert_ppm_to_mg_nm3(
    ppm: float,
    molecular_weight: float
) -> float:
    """
    Convert ppm to mg/Nm³ at standard conditions.

    Formula:
        C[mg/Nm³] = C[ppm] × MW / 22.414

    Args:
        ppm: Concentration in ppm
        molecular_weight: Molecular weight (g/mol)

    Returns:
        Concentration in mg/Nm³

    Example:
        >>> nox_mg = convert_ppm_to_mg_nm3(100, 46.0)
        >>> print(f"NOx: {nox_mg:.1f} mg/Nm³")  # ~205 mg/Nm³
    """
    return ppm * molecular_weight / MOLAR_VOLUME_NM3_KMOL


def convert_mg_nm3_to_ppm(
    mg_nm3: float,
    molecular_weight: float
) -> float:
    """
    Convert mg/Nm³ to ppm at standard conditions.

    Formula:
        C[ppm] = C[mg/Nm³] × 22.414 / MW

    Args:
        mg_nm3: Concentration in mg/Nm³
        molecular_weight: Molecular weight (g/mol)

    Returns:
        Concentration in ppm
    """
    return mg_nm3 * MOLAR_VOLUME_NM3_KMOL / molecular_weight


def correct_to_reference_O2(
    concentration: float,
    measured_O2_pct: float,
    reference_O2_pct: float
) -> float:
    """
    Correct concentration to reference O2 level (EPA Method 19).

    Formula:
        C_ref = C × (21 - O2_ref) / (21 - O2_measured)

    Args:
        concentration: Measured concentration (any units)
        measured_O2_pct: Measured O2 (%)
        reference_O2_pct: Reference O2 (%)

    Returns:
        Corrected concentration (same units as input)

    Example:
        >>> nox_corrected = correct_to_reference_O2(150, 5.0, 3.0)
        >>> print(f"NOx @ 3% O2: {nox_corrected:.1f} ppm")
    """
    correction_factor = (21.0 - reference_O2_pct) / (21.0 - measured_O2_pct)
    return concentration * correction_factor


def calculate_CO_CO2_ratio(CO_ppm: float, CO2_pct: float) -> float:
    """
    Calculate CO/CO2 ratio (combustion quality indicator).

    Args:
        CO_ppm: CO concentration (ppm)
        CO2_pct: CO2 concentration (%)

    Returns:
        CO/CO2 ratio

    Example:
        >>> ratio = calculate_CO_CO2_ratio(50, 12.0)
        >>> print(f"CO/CO2: {ratio:.4f}")  # 0.0004 (good)
    """
    CO2_ppm = CO2_pct * 10000.0
    return CO_ppm / CO2_ppm if CO2_ppm > 0 else 0.0
