"""
EPA 40 CFR Part 75 Emission Rate Calculations

This module implements the core emission rate calculations required for
EPA Part 75 CEMS compliance. All calculations are:
- 100% deterministic (no random elements)
- Use Decimal for precision
- Include SHA-256 provenance hash for audit trail
- Include calculation trace for explainability

Reference: 40 CFR Part 75, Appendix A and F

Author: GL-CalculatorEngineer
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from datetime import datetime


# ==============================================================================
# CONSTANTS - EPA 40 CFR Part 75
# ==============================================================================

class FuelType(Enum):
    """Fuel types with corresponding F-factors per 40 CFR Part 75 Appendix F."""
    COAL = "coal"
    OIL = "oil"
    NATURAL_GAS = "natural_gas"
    WOOD = "wood"
    REFUSE = "refuse"


# F-factors (dscf/MMBtu) from 40 CFR Part 75 Appendix F, Table 1
# These are dry-basis F-factors for O2-based emission rate calculations
F_FACTORS_O2: Dict[FuelType, Decimal] = {
    FuelType.COAL: Decimal("9780"),
    FuelType.OIL: Decimal("9190"),
    FuelType.NATURAL_GAS: Decimal("8710"),
    FuelType.WOOD: Decimal("9240"),
    FuelType.REFUSE: Decimal("9570"),
}

# F-factors (dscf/MMBtu) for CO2-based emission rate calculations
F_FACTORS_CO2: Dict[FuelType, Decimal] = {
    FuelType.COAL: Decimal("1800"),
    FuelType.OIL: Decimal("1420"),
    FuelType.NATURAL_GAS: Decimal("1040"),
    FuelType.WOOD: Decimal("1830"),
    FuelType.REFUSE: Decimal("1820"),
}

# Molecular weights (lb/lb-mole) - standard values
MOLECULAR_WEIGHTS: Dict[str, Decimal] = {
    "NOx": Decimal("46.01"),   # as NO2
    "SO2": Decimal("64.06"),
    "CO": Decimal("28.01"),
    "CO2": Decimal("44.01"),
    "O2": Decimal("32.00"),
    "N2": Decimal("28.01"),
    "H2O": Decimal("18.02"),
}

# Reference O2 percentage (ambient air)
REFERENCE_O2_PERCENT = Decimal("20.9")

# Molar volume at standard conditions (385.3 dscf/lb-mole at 68F, 29.92 inHg)
MOLAR_VOLUME_DSCF = Decimal("385.3")

# Conversion constants
LB_PER_KG = Decimal("2.20462")
BTU_PER_MMBTU = Decimal("1000000")


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class CalculationTrace:
    """Trace of calculation steps for explainability."""
    step_number: int
    description: str
    formula: str
    inputs: Dict[str, str]
    output: str
    output_value: Decimal
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EmissionRateResult:
    """Result of emission rate calculation with provenance."""
    value: Decimal
    unit: str
    calculation_trace: List[CalculationTrace]
    provenance_hash: str
    inputs: Dict[str, Any]
    formula_reference: str
    is_valid: bool = True
    validation_messages: List[str] = field(default_factory=list)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def _decimal_str(value: Any) -> str:
    """Convert value to string for hashing, preserving Decimal precision."""
    if isinstance(value, Decimal):
        return str(value)
    elif isinstance(value, float):
        return str(Decimal(str(value)))
    elif isinstance(value, dict):
        return json.dumps({k: _decimal_str(v) for k, v in sorted(value.items())})
    elif isinstance(value, list):
        return json.dumps([_decimal_str(v) for v in value])
    elif isinstance(value, Enum):
        return value.value
    else:
        return str(value)


def calculate_provenance_hash(
    function_name: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    calculation_trace: List[CalculationTrace]
) -> str:
    """
    Calculate SHA-256 provenance hash for audit trail.

    This provides cryptographic proof of the calculation inputs and outputs.
    """
    provenance_data = {
        "function": function_name,
        "inputs": {k: _decimal_str(v) for k, v in sorted(inputs.items())},
        "outputs": {k: _decimal_str(v) for k, v in sorted(outputs.items())},
        "trace_steps": len(calculation_trace),
        "trace_checksums": [
            hashlib.sha256(
                f"{t.step_number}:{t.description}:{t.output_value}".encode()
            ).hexdigest()[:16]
            for t in calculation_trace
        ]
    }
    provenance_str = json.dumps(provenance_data, sort_keys=True)
    return hashlib.sha256(provenance_str.encode()).hexdigest()


def _apply_precision(value: Decimal, decimal_places: int) -> Decimal:
    """Apply regulatory rounding (ROUND_HALF_UP) to specified decimal places."""
    if decimal_places == 0:
        return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    quantize_str = "0." + "0" * decimal_places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ==============================================================================
# CORE EMISSION RATE CALCULATIONS
# ==============================================================================

def ppm_to_lb_per_mmbtu(
    concentration_ppm: Decimal,
    pollutant: str,
    fuel_type: FuelType,
    o2_percent: Optional[Decimal] = None,
    apply_o2_correction: bool = True,
    decimal_precision: int = 4
) -> EmissionRateResult:
    """
    Convert pollutant concentration (ppm) to emission rate (lb/MMBtu).

    Per 40 CFR Part 75, Appendix F, Equation F-5:
    E = K * C * Fd * (20.9 / (20.9 - %O2d))

    Where:
        E  = emission rate (lb/MMBtu)
        K  = 1.194 x 10^-7 (lb-dscf)/(ppm-scf-lb-mole) * MW
        C  = pollutant concentration (ppm, dry basis)
        Fd = F-factor (dscf/MMBtu)
        20.9/(20.9 - %O2d) = O2 correction factor

    Args:
        concentration_ppm: Pollutant concentration in ppm (dry basis)
        pollutant: Pollutant name (NOx, SO2, CO, CO2)
        fuel_type: Type of fuel being burned
        o2_percent: Measured O2 percentage (dry basis), required if apply_o2_correction=True
        apply_o2_correction: Whether to apply O2 correction factor
        decimal_precision: Number of decimal places for result

    Returns:
        EmissionRateResult with value in lb/MMBtu
    """
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []

    # Input validation
    concentration_ppm = Decimal(str(concentration_ppm))
    if concentration_ppm < 0:
        validation_messages.append("Concentration cannot be negative")

    if pollutant not in MOLECULAR_WEIGHTS:
        raise ValueError(f"Unknown pollutant: {pollutant}. Valid: {list(MOLECULAR_WEIGHTS.keys())}")

    if fuel_type not in F_FACTORS_O2:
        raise ValueError(f"Unknown fuel type: {fuel_type}. Valid: {list(F_FACTORS_O2.keys())}")

    # Step 1: Get molecular weight
    mw = MOLECULAR_WEIGHTS[pollutant]
    calculation_trace.append(CalculationTrace(
        step_number=1,
        description=f"Retrieve molecular weight for {pollutant}",
        formula="MW lookup from EPA standard values",
        inputs={"pollutant": pollutant},
        output="molecular_weight",
        output_value=mw
    ))

    # Step 2: Get F-factor for fuel type
    fd = F_FACTORS_O2[fuel_type]
    calculation_trace.append(CalculationTrace(
        step_number=2,
        description=f"Retrieve F-factor (Fd) for {fuel_type.value}",
        formula="Fd lookup from 40 CFR Part 75, Appendix F, Table 1",
        inputs={"fuel_type": fuel_type.value},
        output="f_factor_dscf_per_mmbtu",
        output_value=fd
    ))

    # Step 3: Calculate K factor
    # K = (1/MOLAR_VOLUME) * MW * 10^-6 (conversion from ppm)
    # K = MW / (385.3 * 10^6) lb/dscf per ppm
    k_factor = mw / (MOLAR_VOLUME_DSCF * Decimal("1000000"))
    calculation_trace.append(CalculationTrace(
        step_number=3,
        description="Calculate K factor (ppm to lb/dscf conversion)",
        formula="K = MW / (385.3 * 10^6)",
        inputs={"molecular_weight": str(mw), "molar_volume": str(MOLAR_VOLUME_DSCF)},
        output="k_factor",
        output_value=k_factor
    ))

    # Step 4: Calculate base emission rate (without O2 correction)
    base_emission_rate = k_factor * concentration_ppm * fd
    calculation_trace.append(CalculationTrace(
        step_number=4,
        description="Calculate base emission rate (before O2 correction)",
        formula="E_base = K * C * Fd",
        inputs={
            "k_factor": str(k_factor),
            "concentration_ppm": str(concentration_ppm),
            "f_factor": str(fd)
        },
        output="base_emission_rate_lb_per_mmbtu",
        output_value=base_emission_rate
    ))

    # Step 5: Apply O2 correction if requested
    final_emission_rate = base_emission_rate
    if apply_o2_correction:
        if o2_percent is None:
            raise ValueError("O2 percentage required when apply_o2_correction=True")

        o2_percent = Decimal(str(o2_percent))

        # Validation: O2 should be between 0 and 20.9%
        if o2_percent < 0 or o2_percent >= REFERENCE_O2_PERCENT:
            validation_messages.append(
                f"O2 percentage ({o2_percent}%) outside valid range (0 to {REFERENCE_O2_PERCENT}%)"
            )

        o2_correction = REFERENCE_O2_PERCENT / (REFERENCE_O2_PERCENT - o2_percent)
        calculation_trace.append(CalculationTrace(
            step_number=5,
            description="Calculate O2 correction factor",
            formula="O2_correction = 20.9 / (20.9 - %O2)",
            inputs={
                "reference_o2": str(REFERENCE_O2_PERCENT),
                "measured_o2": str(o2_percent)
            },
            output="o2_correction_factor",
            output_value=o2_correction
        ))

        final_emission_rate = base_emission_rate * o2_correction
        calculation_trace.append(CalculationTrace(
            step_number=6,
            description="Apply O2 correction to emission rate",
            formula="E = E_base * O2_correction",
            inputs={
                "base_emission_rate": str(base_emission_rate),
                "o2_correction": str(o2_correction)
            },
            output="final_emission_rate_lb_per_mmbtu",
            output_value=final_emission_rate
        ))

    # Apply precision
    final_value = _apply_precision(final_emission_rate, decimal_precision)

    # Build inputs dictionary for provenance
    inputs_dict = {
        "concentration_ppm": concentration_ppm,
        "pollutant": pollutant,
        "fuel_type": fuel_type.value,
        "apply_o2_correction": apply_o2_correction,
        "decimal_precision": decimal_precision
    }
    if o2_percent is not None:
        inputs_dict["o2_percent"] = o2_percent

    # Calculate provenance hash
    provenance_hash = calculate_provenance_hash(
        function_name="ppm_to_lb_per_mmbtu",
        inputs=inputs_dict,
        outputs={"emission_rate_lb_per_mmbtu": final_value},
        calculation_trace=calculation_trace
    )

    return EmissionRateResult(
        value=final_value,
        unit="lb/MMBtu",
        calculation_trace=calculation_trace,
        provenance_hash=provenance_hash,
        inputs=inputs_dict,
        formula_reference="40 CFR Part 75, Appendix F, Equation F-5",
        is_valid=len(validation_messages) == 0,
        validation_messages=validation_messages
    )


def o2_correction_factor(
    o2_percent: Decimal,
    decimal_precision: int = 6
) -> EmissionRateResult:
    """
    Calculate O2 correction factor per 40 CFR Part 75.

    Formula: O2_correction = 20.9 / (20.9 - %O2)

    This corrects emission concentrations to a standard O2 reference level
    to allow comparison between sources operating at different excess air levels.

    Args:
        o2_percent: Measured O2 percentage (dry basis)
        decimal_precision: Number of decimal places for result

    Returns:
        EmissionRateResult with O2 correction factor (dimensionless)
    """
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []

    o2_percent = Decimal(str(o2_percent))

    # Validation
    if o2_percent < 0:
        validation_messages.append("O2 percentage cannot be negative")
    if o2_percent >= REFERENCE_O2_PERCENT:
        validation_messages.append(
            f"O2 percentage ({o2_percent}%) must be less than reference ({REFERENCE_O2_PERCENT}%)"
        )

    # Calculate correction factor
    denominator = REFERENCE_O2_PERCENT - o2_percent

    calculation_trace.append(CalculationTrace(
        step_number=1,
        description="Calculate denominator (20.9 - %O2)",
        formula="denominator = 20.9 - %O2",
        inputs={
            "reference_o2": str(REFERENCE_O2_PERCENT),
            "measured_o2": str(o2_percent)
        },
        output="denominator",
        output_value=denominator
    ))

    if denominator <= 0:
        # Avoid division by zero
        correction_factor = Decimal("999.999")  # Max reasonable value
        validation_messages.append("Division by zero avoided - O2 at or above reference")
    else:
        correction_factor = REFERENCE_O2_PERCENT / denominator

    calculation_trace.append(CalculationTrace(
        step_number=2,
        description="Calculate O2 correction factor",
        formula="O2_correction = 20.9 / (20.9 - %O2)",
        inputs={
            "reference_o2": str(REFERENCE_O2_PERCENT),
            "denominator": str(denominator)
        },
        output="o2_correction_factor",
        output_value=correction_factor
    ))

    final_value = _apply_precision(correction_factor, decimal_precision)

    inputs_dict = {
        "o2_percent": o2_percent,
        "decimal_precision": decimal_precision
    }

    provenance_hash = calculate_provenance_hash(
        function_name="o2_correction_factor",
        inputs=inputs_dict,
        outputs={"o2_correction_factor": final_value},
        calculation_trace=calculation_trace
    )

    return EmissionRateResult(
        value=final_value,
        unit="dimensionless",
        calculation_trace=calculation_trace,
        provenance_hash=provenance_hash,
        inputs=inputs_dict,
        formula_reference="40 CFR Part 75, Appendix F",
        is_valid=len(validation_messages) == 0,
        validation_messages=validation_messages
    )


def wet_to_dry_correction(
    moisture_percent: Decimal,
    decimal_precision: int = 6
) -> EmissionRateResult:
    """
    Calculate wet-to-dry moisture correction factor.

    Formula: Bws = 1 / (1 - (H2O% / 100))

    This converts wet-basis concentrations to dry-basis for regulatory reporting.

    Args:
        moisture_percent: Moisture content in percentage (H2O%)
        decimal_precision: Number of decimal places for result

    Returns:
        EmissionRateResult with moisture correction factor (dimensionless)
    """
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []

    moisture_percent = Decimal(str(moisture_percent))

    # Validation
    if moisture_percent < 0:
        validation_messages.append("Moisture percentage cannot be negative")
    if moisture_percent >= Decimal("100"):
        validation_messages.append("Moisture percentage must be less than 100%")

    # Calculate moisture fraction
    moisture_fraction = moisture_percent / Decimal("100")

    calculation_trace.append(CalculationTrace(
        step_number=1,
        description="Convert moisture percentage to fraction",
        formula="moisture_fraction = H2O% / 100",
        inputs={"moisture_percent": str(moisture_percent)},
        output="moisture_fraction",
        output_value=moisture_fraction
    ))

    # Calculate dry gas fraction
    dry_fraction = Decimal("1") - moisture_fraction

    calculation_trace.append(CalculationTrace(
        step_number=2,
        description="Calculate dry gas fraction",
        formula="dry_fraction = 1 - moisture_fraction",
        inputs={"moisture_fraction": str(moisture_fraction)},
        output="dry_fraction",
        output_value=dry_fraction
    ))

    # Calculate correction factor
    if dry_fraction <= 0:
        correction_factor = Decimal("999.999")
        validation_messages.append("Division by zero avoided - moisture at or above 100%")
    else:
        correction_factor = Decimal("1") / dry_fraction

    calculation_trace.append(CalculationTrace(
        step_number=3,
        description="Calculate wet-to-dry correction factor",
        formula="Bws = 1 / (1 - H2O%/100)",
        inputs={"dry_fraction": str(dry_fraction)},
        output="wet_to_dry_factor",
        output_value=correction_factor
    ))

    final_value = _apply_precision(correction_factor, decimal_precision)

    inputs_dict = {
        "moisture_percent": moisture_percent,
        "decimal_precision": decimal_precision
    }

    provenance_hash = calculate_provenance_hash(
        function_name="wet_to_dry_correction",
        inputs=inputs_dict,
        outputs={"wet_to_dry_factor": final_value},
        calculation_trace=calculation_trace
    )

    return EmissionRateResult(
        value=final_value,
        unit="dimensionless",
        calculation_trace=calculation_trace,
        provenance_hash=provenance_hash,
        inputs=inputs_dict,
        formula_reference="40 CFR Part 75, Appendix A",
        is_valid=len(validation_messages) == 0,
        validation_messages=validation_messages
    )


def mass_emission_rate(
    concentration_ppm: Decimal,
    pollutant: str,
    stack_flow_scfh: Decimal,
    moisture_percent: Optional[Decimal] = None,
    concentration_basis: str = "dry",
    decimal_precision: int = 3
) -> EmissionRateResult:
    """
    Calculate mass emission rate (lb/hr) from concentration and flow.

    Per 40 CFR Part 75, Appendix F, Equation F-1:
    E_mass = C * Q * MW / V_mol

    Where:
        E_mass = mass emission rate (lb/hr)
        C = concentration (ppm converted to fraction)
        Q = volumetric flow rate (scf/hr)
        MW = molecular weight (lb/lb-mole)
        V_mol = molar volume at standard conditions (385.3 scf/lb-mole)

    Args:
        concentration_ppm: Pollutant concentration in ppm
        pollutant: Pollutant name (NOx, SO2, CO, CO2)
        stack_flow_scfh: Stack gas flow rate (scf/hr at standard conditions)
        moisture_percent: Moisture percentage if concentration is wet-basis
        concentration_basis: "dry" or "wet" basis for concentration
        decimal_precision: Number of decimal places for result

    Returns:
        EmissionRateResult with mass emission rate in lb/hr
    """
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []

    concentration_ppm = Decimal(str(concentration_ppm))
    stack_flow_scfh = Decimal(str(stack_flow_scfh))

    # Input validation
    if concentration_ppm < 0:
        validation_messages.append("Concentration cannot be negative")
    if stack_flow_scfh < 0:
        validation_messages.append("Stack flow cannot be negative")
    if pollutant not in MOLECULAR_WEIGHTS:
        raise ValueError(f"Unknown pollutant: {pollutant}")

    # Step 1: Get molecular weight
    mw = MOLECULAR_WEIGHTS[pollutant]
    calculation_trace.append(CalculationTrace(
        step_number=1,
        description=f"Retrieve molecular weight for {pollutant}",
        formula="MW lookup",
        inputs={"pollutant": pollutant},
        output="molecular_weight",
        output_value=mw
    ))

    # Step 2: Convert ppm to fraction
    concentration_fraction = concentration_ppm / Decimal("1000000")
    calculation_trace.append(CalculationTrace(
        step_number=2,
        description="Convert ppm to volume fraction",
        formula="C_fraction = C_ppm / 10^6",
        inputs={"concentration_ppm": str(concentration_ppm)},
        output="concentration_fraction",
        output_value=concentration_fraction
    ))

    # Step 3: Apply wet-to-dry correction if needed
    working_concentration = concentration_fraction
    if concentration_basis == "wet" and moisture_percent is not None:
        moisture_percent = Decimal(str(moisture_percent))
        wet_dry_result = wet_to_dry_correction(moisture_percent)
        wet_dry_factor = wet_dry_result.value
        working_concentration = concentration_fraction * wet_dry_factor

        calculation_trace.append(CalculationTrace(
            step_number=3,
            description="Apply wet-to-dry correction",
            formula="C_dry = C_wet * Bws",
            inputs={
                "concentration_wet": str(concentration_fraction),
                "wet_dry_factor": str(wet_dry_factor)
            },
            output="concentration_dry",
            output_value=working_concentration
        ))

    # Step 4: Calculate mass emission rate
    # E = C * Q * MW / V_mol
    mass_rate = (working_concentration * stack_flow_scfh * mw) / MOLAR_VOLUME_DSCF

    calculation_trace.append(CalculationTrace(
        step_number=4 if concentration_basis == "dry" else 5,
        description="Calculate mass emission rate",
        formula="E_mass = C * Q * MW / V_mol",
        inputs={
            "concentration_fraction": str(working_concentration),
            "stack_flow_scfh": str(stack_flow_scfh),
            "molecular_weight": str(mw),
            "molar_volume": str(MOLAR_VOLUME_DSCF)
        },
        output="mass_emission_rate_lb_per_hr",
        output_value=mass_rate
    ))

    final_value = _apply_precision(mass_rate, decimal_precision)

    inputs_dict = {
        "concentration_ppm": concentration_ppm,
        "pollutant": pollutant,
        "stack_flow_scfh": stack_flow_scfh,
        "concentration_basis": concentration_basis,
        "decimal_precision": decimal_precision
    }
    if moisture_percent is not None:
        inputs_dict["moisture_percent"] = moisture_percent

    provenance_hash = calculate_provenance_hash(
        function_name="mass_emission_rate",
        inputs=inputs_dict,
        outputs={"mass_emission_rate_lb_hr": final_value},
        calculation_trace=calculation_trace
    )

    return EmissionRateResult(
        value=final_value,
        unit="lb/hr",
        calculation_trace=calculation_trace,
        provenance_hash=provenance_hash,
        inputs=inputs_dict,
        formula_reference="40 CFR Part 75, Appendix F, Equation F-1",
        is_valid=len(validation_messages) == 0,
        validation_messages=validation_messages
    )


def heat_input_rate(
    fuel_flow_rate: Decimal,
    fuel_flow_unit: str,
    gross_calorific_value: Decimal,
    gcv_unit: str,
    decimal_precision: int = 2
) -> EmissionRateResult:
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []
    fuel_flow_rate = Decimal(str(fuel_flow_rate))
    gross_calorific_value = Decimal(str(gross_calorific_value))
    if fuel_flow_rate < 0:
        validation_messages.append('Fuel flow rate cannot be negative')
    if gross_calorific_value <= 0:
        validation_messages.append('Gross calorific value must be positive')
    calculation_trace.append(CalculationTrace(
        step_number=1, description='Input fuel flow rate', formula='Given',
        inputs={'fuel_flow_rate': str(fuel_flow_rate), 'unit': fuel_flow_unit},
        output='fuel_flow_rate', output_value=fuel_flow_rate))
    calculation_trace.append(CalculationTrace(
        step_number=2, description='Input gross calorific value', formula='Given',
        inputs={'gross_calorific_value': str(gross_calorific_value), 'unit': gcv_unit},
        output='gross_calorific_value', output_value=gross_calorific_value))
    heat_input_btu = fuel_flow_rate * gross_calorific_value
    calculation_trace.append(CalculationTrace(
        step_number=3, description='Calculate heat input in Btu/hr',
        formula='HI_btu = fuel_flow * GCV',
        inputs={'fuel_flow_rate': str(fuel_flow_rate), 'gross_calorific_value': str(gross_calorific_value)},
        output='heat_input_btu_hr', output_value=heat_input_btu))
    heat_input_mmbtu = heat_input_btu / BTU_PER_MMBTU
    calculation_trace.append(CalculationTrace(
        step_number=4, description='Convert to MMBtu/hr', formula='HI_mmbtu = HI_btu / 10^6',
        inputs={'heat_input_btu': str(heat_input_btu)},
        output='heat_input_mmbtu_hr', output_value=heat_input_mmbtu))
    final_value = _apply_precision(heat_input_mmbtu, decimal_precision)
    inputs_dict = {'fuel_flow_rate': fuel_flow_rate, 'fuel_flow_unit': fuel_flow_unit,
        'gross_calorific_value': gross_calorific_value, 'gcv_unit': gcv_unit,
        'decimal_precision': decimal_precision}
    provenance_hash = calculate_provenance_hash(function_name='heat_input_rate',
        inputs=inputs_dict, outputs={'heat_input_mmbtu_hr': final_value},
        calculation_trace=calculation_trace)
    return EmissionRateResult(value=final_value, unit='MMBtu/hr',
        calculation_trace=calculation_trace, provenance_hash=provenance_hash,
        inputs=inputs_dict, formula_reference='40 CFR Part 75, Appendix D',
        is_valid=len(validation_messages) == 0, validation_messages=validation_messages)


def lb_per_mmbtu_to_lb_per_hr(
    emission_rate_lb_mmbtu: Decimal,
    heat_input_mmbtu_hr: Decimal,
    decimal_precision: int = 3
) -> EmissionRateResult:
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []
    emission_rate_lb_mmbtu = Decimal(str(emission_rate_lb_mmbtu))
    heat_input_mmbtu_hr = Decimal(str(heat_input_mmbtu_hr))
    if emission_rate_lb_mmbtu < 0:
        validation_messages.append('Emission rate cannot be negative')
    if heat_input_mmbtu_hr < 0:
        validation_messages.append('Heat input cannot be negative')
    mass_rate = emission_rate_lb_mmbtu * heat_input_mmbtu_hr
    calculation_trace.append(CalculationTrace(
        step_number=1, description='Calculate mass emission rate from emission rate and heat input',
        formula='E_mass = E_rate * HI',
        inputs={'emission_rate_lb_mmbtu': str(emission_rate_lb_mmbtu),
                'heat_input_mmbtu_hr': str(heat_input_mmbtu_hr)},
        output='mass_emission_rate_lb_hr', output_value=mass_rate))
    final_value = _apply_precision(mass_rate, decimal_precision)
    inputs_dict = {'emission_rate_lb_mmbtu': emission_rate_lb_mmbtu,
        'heat_input_mmbtu_hr': heat_input_mmbtu_hr, 'decimal_precision': decimal_precision}
    provenance_hash = calculate_provenance_hash(function_name='lb_per_mmbtu_to_lb_per_hr',
        inputs=inputs_dict, outputs={'mass_emission_rate_lb_hr': final_value},
        calculation_trace=calculation_trace)
    return EmissionRateResult(value=final_value, unit='lb/hr',
        calculation_trace=calculation_trace, provenance_hash=provenance_hash,
        inputs=inputs_dict, formula_reference='40 CFR Part 75',
        is_valid=len(validation_messages) == 0, validation_messages=validation_messages)


def calculate_total_emissions(
    mass_emission_rate_lb_hr: Decimal,
    operating_hours: Decimal,
    decimal_precision: int = 2
) -> EmissionRateResult:
    calculation_trace: List[CalculationTrace] = []
    validation_messages: List[str] = []
    mass_emission_rate_lb_hr = Decimal(str(mass_emission_rate_lb_hr))
    operating_hours = Decimal(str(operating_hours))
    if mass_emission_rate_lb_hr < 0:
        validation_messages.append('Mass emission rate cannot be negative')
    if operating_hours < 0:
        validation_messages.append('Operating hours cannot be negative')
    total_lb = mass_emission_rate_lb_hr * operating_hours
    calculation_trace.append(CalculationTrace(
        step_number=1, description='Calculate total emissions in pounds',
        formula='E_lb = E_rate * hours',
        inputs={'mass_emission_rate_lb_hr': str(mass_emission_rate_lb_hr),
                'operating_hours': str(operating_hours)},
        output='total_emissions_lb', output_value=total_lb))
    total_tons = total_lb / Decimal('2000')
    calculation_trace.append(CalculationTrace(
        step_number=2, description='Convert pounds to tons', formula='E_tons = E_lb / 2000',
        inputs={'total_emissions_lb': str(total_lb)},
        output='total_emissions_tons', output_value=total_tons))
    final_value = _apply_precision(total_tons, decimal_precision)
    inputs_dict = {'mass_emission_rate_lb_hr': mass_emission_rate_lb_hr,
        'operating_hours': operating_hours, 'decimal_precision': decimal_precision}
    provenance_hash = calculate_provenance_hash(function_name='calculate_total_emissions',
        inputs=inputs_dict, outputs={'total_emissions_tons': final_value},
        calculation_trace=calculation_trace)
    return EmissionRateResult(value=final_value, unit='tons',
        calculation_trace=calculation_trace, provenance_hash=provenance_hash,
        inputs=inputs_dict, formula_reference='40 CFR Part 75',
        is_valid=len(validation_messages) == 0, validation_messages=validation_messages)


def get_f_factor(fuel_type: FuelType, diluent_type: str = 'O2') -> Tuple[Decimal, str]:
    if diluent_type == 'O2':
        return F_FACTORS_O2[fuel_type], '40 CFR Part 75, Appendix F, Table 1 (Fd)'
    elif diluent_type == 'CO2':
        return F_FACTORS_CO2[fuel_type], '40 CFR Part 75, Appendix F, Table 1 (Fc)'
    else:
        raise ValueError(f'Invalid diluent type: {diluent_type}')


def get_molecular_weight(pollutant: str) -> Decimal:
    if pollutant not in MOLECULAR_WEIGHTS:
        raise ValueError(f'Unknown pollutant: {pollutant}. Valid: {list(MOLECULAR_WEIGHTS.keys())}')
    return MOLECULAR_WEIGHTS[pollutant]
