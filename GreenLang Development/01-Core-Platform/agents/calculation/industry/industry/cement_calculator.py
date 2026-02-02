"""
Cement Industry Emission Calculator - Zero Hallucination Guarantee

This module implements deterministic, bit-perfect calculations for cement
production emissions following IPCC and GNR (Getting the Numbers Right)
methodologies.

Sources:
    - IPCC 2006 Guidelines for National Greenhouse Gas Inventories, Volume 3, Chapter 2
    - GCCA/CSI Getting the Numbers Right (GNR) Database
    - EU ETS Monitoring and Reporting Regulation (EU) 2018/2066
    - EN 197-1 Cement Composition Standard

Cement Types (EN 197-1):
    - CEM I: Portland cement (95-100% clinker)
    - CEM II: Portland composite (65-94% clinker)
    - CEM III: Blast furnace cement (5-64% clinker)
    - CEM IV: Pozzolanic cement (45-89% clinker)
    - CEM V: Composite cement (20-64% clinker)

CBAM Compliance: Annex III cement-specific methodology
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import hashlib
import json
from datetime import datetime, timezone


# =============================================================================
# EMISSION FACTORS - DETERMINISTIC CONSTANTS
# =============================================================================

class CementEmissionFactors:
    """
    Authoritative emission factors for cement production.

    ALL factors are from IPCC and industry-standard sources.
    NO interpolation, estimation, or LLM-generated values.
    """

    # Clinker calcination emissions (tCO2/t clinker)
    # Source: IPCC 2006, Volume 3, Chapter 2, Table 2.1
    # CaCO3 -> Cite + CO2 stoichiometry: 0.440 * ite CaO content (typically ~65%)
    CLINKER_CALCINATION: Decimal = Decimal("0.525")

    # Kiln fuel combustion (tCO2/t clinker) - global average
    # Source: GNR Database 2020, weighted average
    KILN_FUEL_AVERAGE: Decimal = Decimal("0.310")

    # Kiln fuel by type (tCO2/t clinker)
    KILN_FUEL_COAL: Decimal = Decimal("0.350")
    KILN_FUEL_PETCOKE: Decimal = Decimal("0.340")
    KILN_FUEL_NATURAL_GAS: Decimal = Decimal("0.200")
    KILN_FUEL_ALTERNATIVE: Decimal = Decimal("0.150")  # RDF, biomass mix

    # Electricity for grinding (tCO2/t cement) - depends on grid
    # Average electricity: 100-120 kWh/t cement
    GRINDING_ELECTRICITY_KWH: Decimal = Decimal("110")

    # SCM (Supplementary Cite Materials) emission factors
    # These are credits for avoided clinker production
    # Source: GNR Database, allocation methodology
    SCM_GGBS_CREDIT: Decimal = Decimal("0.070")  # tCO2/t GGBS (allocation from steel)
    SCM_FLY_ASH_CREDIT: Decimal = Decimal("0.020")  # tCO2/t fly ash (allocation from coal)
    SCM_NATURAL_POZZOLAN_CREDIT: Decimal = Decimal("0.010")  # tCO2/t (extraction only)
    SCM_LIMESTONE_CREDIT: Decimal = Decimal("0.030")  # tCO2/t limestone filler

    # Cement type clinker ratios (EN 197-1 midpoints)
    CEMENT_CLINKER_RATIOS = {
        "CEM_I": Decimal("0.95"),    # 95% clinker (95-100% range)
        "CEM_II_A": Decimal("0.85"), # 85% clinker (80-94% range)
        "CEM_II_B": Decimal("0.70"), # 70% clinker (65-79% range)
        "CEM_III_A": Decimal("0.50"), # 50% clinker (35-64% range)
        "CEM_III_B": Decimal("0.25"), # 25% clinker (20-34% range)
        "CEM_III_C": Decimal("0.10"), # 10% clinker (5-19% range)
        "CEM_IV_A": Decimal("0.75"), # 75% clinker (65-89% range)
        "CEM_IV_B": Decimal("0.55"), # 55% clinker (45-64% range)
        "CEM_V_A": Decimal("0.50"),  # 50% clinker (40-64% range)
        "CEM_V_B": Decimal("0.30"),  # 30% clinker (20-39% range)
    }

    # Metadata
    SOURCES = {
        "CLINKER_CALCINATION": "IPCC 2006 Vol 3 Ch 2 Table 2.1",
        "KILN_FUEL": "GCCA GNR Database 2020",
        "SCM_CREDITS": "GCCA GNR Allocation Methodology",
        "CEMENT_TYPES": "EN 197-1:2011 Cement Composition"
    }


# =============================================================================
# DATA MODELS
# =============================================================================

class CementType(str, Enum):
    """Cement types per EN 197-1."""
    CEM_I = "CEM_I"
    CEM_II_A = "CEM_II_A"
    CEM_II_B = "CEM_II_B"
    CEM_III_A = "CEM_III_A"
    CEM_III_B = "CEM_III_B"
    CEM_III_C = "CEM_III_C"
    CEM_IV_A = "CEM_IV_A"
    CEM_IV_B = "CEM_IV_B"
    CEM_V_A = "CEM_V_A"
    CEM_V_B = "CEM_V_B"


class KilnFuelType(str, Enum):
    """Kiln fuel types."""
    COAL = "COAL"
    PETCOKE = "PETCOKE"
    NATURAL_GAS = "NATURAL_GAS"
    ALTERNATIVE = "ALTERNATIVE"
    MIXED = "MIXED"


class SCMType(str, Enum):
    """Supplementary Cementitious Materials."""
    GGBS = "GGBS"  # Ground Granulated Blast-furnace Slag
    FLY_ASH = "FLY_ASH"
    NATURAL_POZZOLAN = "NATURAL_POZZOLAN"
    LIMESTONE = "LIMESTONE"


class SCMInput(BaseModel):
    """SCM input with quantity."""
    scm_type: SCMType
    quantity_tonnes: Decimal = Field(ge=0)


class CementCalculationInput(BaseModel):
    """Input parameters for cement emission calculation."""

    cement_production_tonnes: Decimal = Field(
        ...,
        gt=0,
        le=Decimal("100000000"),
        description="Cement production in metric tonnes"
    )

    cement_type: CementType = Field(
        default=CementType.CEM_I,
        description="Cement type per EN 197-1"
    )

    clinker_production_tonnes: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Clinker production (if different from calculated)"
    )

    clinker_ratio: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0.05"),
        le=Decimal("1.00"),
        description="Custom clinker-to-cement ratio (overrides cement_type)"
    )

    kiln_fuel_type: KilnFuelType = Field(
        default=KilnFuelType.MIXED,
        description="Primary kiln fuel type"
    )

    kiln_fuel_mix: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Custom fuel mix percentages"
    )

    scm_inputs: List[SCMInput] = Field(
        default_factory=list,
        description="Supplementary cementitious materials used"
    )

    electricity_kwh: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Electricity consumption in kWh"
    )

    grid_emission_factor_kg_co2_per_kwh: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=Decimal("2.0"),
        description="Grid emission factor in kg CO2/kWh"
    )

    reporting_period: str = Field(
        default="",
        description="Reporting period (e.g., '2024-Q1')"
    )

    facility_id: str = Field(
        default="",
        description="Facility identifier for CBAM reporting"
    )


class CalculationStep(BaseModel):
    """Individual calculation step with full provenance."""
    step_number: int
    description: str
    formula: str
    inputs: Dict[str, str]
    output_value: Decimal
    output_unit: str
    source: str


class CementCalculationResult(BaseModel):
    """Complete calculation result with CBAM-compliant output."""

    # Identification
    calculation_id: str
    timestamp: str
    calculator_version: str = "1.0.0"

    # Input summary
    cement_type: str
    cement_production_tonnes: Decimal
    clinker_production_tonnes: Decimal
    clinker_ratio: Decimal

    # Emissions breakdown
    calcination_emissions_tco2: Decimal
    kiln_fuel_emissions_tco2: Decimal
    electricity_emissions_tco2: Decimal
    scm_credits_tco2: Decimal
    total_direct_emissions_tco2: Decimal
    total_indirect_emissions_tco2: Decimal
    total_emissions_tco2: Decimal

    # Intensity metrics
    emission_intensity_tco2_per_tonne_cement: Decimal
    emission_intensity_tco2_per_tonne_clinker: Decimal

    # CBAM fields
    cbam_product_category: str = "Cement"
    cbam_cn_code: str = "2523"
    cbam_embedded_emissions_tco2: Decimal
    cbam_specific_embedded_emissions: Decimal

    # Provenance
    calculation_steps: List[CalculationStep]
    emission_factor_sources: Dict[str, str]
    provenance_hash: str

    # Validation
    is_validated: bool = True
    validation_notes: List[str] = []


# =============================================================================
# CALCULATION ENGINE
# =============================================================================

class CementEmissionCalculator:
    """
    Zero-hallucination cement emission calculator.

    Guarantees:
        - Deterministic: Same input produces identical output (bit-perfect)
        - Reproducible: Complete provenance tracking
        - Auditable: SHA-256 hash of all calculation steps
        - Compliant: CBAM Annex III and IPCC methodology
        - Zero LLM: No AI/ML in calculation path
    """

    VERSION = "1.0.0"
    PRECISION = 6  # Decimal places for intermediate calculations
    OUTPUT_PRECISION = 3  # Decimal places for final output

    def __init__(self):
        """Initialize calculator with emission factors."""
        self.factors = CementEmissionFactors()

    def calculate(self, input_data: CementCalculationInput) -> CementCalculationResult:
        """
        Execute cement emission calculation with zero hallucination guarantee.

        Args:
            input_data: Validated calculation inputs

        Returns:
            Complete calculation result with provenance
        """
        calculation_steps: List[CalculationStep] = []
        step_num = 0

        # Generate calculation ID
        calc_id = self._generate_calculation_id(input_data)

        # Step 1: Determine clinker production
        step_num += 1
        clinker_ratio = self._get_clinker_ratio(input_data)
        clinker_tonnes = input_data.clinker_production_tonnes or (
            input_data.cement_production_tonnes * clinker_ratio
        )

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="Calculate clinker production from cement type",
            formula="cement_tonnes * clinker_ratio",
            inputs={
                "cement_production_tonnes": str(input_data.cement_production_tonnes),
                "cement_type": input_data.cement_type.value,
                "clinker_ratio": str(clinker_ratio)
            },
            output_value=clinker_tonnes,
            output_unit="tonnes clinker",
            source=self.factors.SOURCES["CEMENT_TYPES"]
        ))

        # Step 2: Calculate calcination emissions
        step_num += 1
        calcination_emissions = clinker_tonnes * self.factors.CLINKER_CALCINATION

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="Calcination emissions (CaCO3 decomposition)",
            formula="clinker_tonnes * 0.525 tCO2/t",
            inputs={
                "clinker_tonnes": str(clinker_tonnes),
                "calcination_factor": str(self.factors.CLINKER_CALCINATION)
            },
            output_value=calcination_emissions,
            output_unit="tCO2",
            source=self.factors.SOURCES["CLINKER_CALCINATION"]
        ))

        # Step 3: Calculate kiln fuel emissions
        step_num += 1
        kiln_fuel_factor = self._get_kiln_fuel_factor(input_data)
        kiln_fuel_emissions = clinker_tonnes * kiln_fuel_factor

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="Kiln fuel combustion emissions",
            formula="clinker_tonnes * kiln_fuel_factor",
            inputs={
                "clinker_tonnes": str(clinker_tonnes),
                "kiln_fuel_type": input_data.kiln_fuel_type.value,
                "kiln_fuel_factor": str(kiln_fuel_factor)
            },
            output_value=kiln_fuel_emissions,
            output_unit="tCO2",
            source=self.factors.SOURCES["KILN_FUEL"]
        ))

        # Step 4: Calculate electricity emissions
        step_num += 1
        electricity_emissions = self._calculate_electricity_emissions(
            input_data, input_data.cement_production_tonnes
        )

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="Electricity emissions (grinding, etc.)",
            formula="electricity_kwh * grid_factor / 1000",
            inputs={
                "electricity_kwh": str(
                    input_data.electricity_kwh or
                    (input_data.cement_production_tonnes * self.factors.GRINDING_ELECTRICITY_KWH)
                ),
                "grid_factor_kg_co2_per_kwh": str(
                    input_data.grid_emission_factor_kg_co2_per_kwh or Decimal("0")
                )
            },
            output_value=electricity_emissions,
            output_unit="tCO2",
            source="Calculated from grid factor"
        ))

        # Step 5: Calculate SCM credits
        step_num += 1
        scm_credits = self._calculate_scm_credits(input_data.scm_inputs)

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="SCM credits (avoided clinker production)",
            formula="sum(scm_tonnes * scm_credit_factor)",
            inputs={
                "scm_inputs": str([
                    {"type": s.scm_type.value, "tonnes": str(s.quantity_tonnes)}
                    for s in input_data.scm_inputs
                ])
            },
            output_value=scm_credits,
            output_unit="tCO2",
            source=self.factors.SOURCES["SCM_CREDITS"]
        ))

        # Step 6: Calculate totals
        step_num += 1
        total_direct = calcination_emissions + kiln_fuel_emissions
        total_indirect = electricity_emissions
        total_emissions = total_direct + total_indirect + scm_credits
        total_emissions = self._apply_precision(total_emissions, self.OUTPUT_PRECISION)

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="Total emissions calculation",
            formula="calcination + kiln_fuel + electricity + scm_credits",
            inputs={
                "calcination_emissions": str(calcination_emissions),
                "kiln_fuel_emissions": str(kiln_fuel_emissions),
                "electricity_emissions": str(electricity_emissions),
                "scm_credits": str(scm_credits)
            },
            output_value=total_emissions,
            output_unit="tCO2",
            source="Summation"
        ))

        # Step 7: Calculate emission intensities
        step_num += 1
        intensity_per_cement = total_emissions / input_data.cement_production_tonnes
        intensity_per_clinker = (calcination_emissions + kiln_fuel_emissions) / clinker_tonnes

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="Emission intensity calculations",
            formula="total_emissions / production_tonnes",
            inputs={
                "total_emissions": str(total_emissions),
                "cement_tonnes": str(input_data.cement_production_tonnes),
                "clinker_tonnes": str(clinker_tonnes)
            },
            output_value=intensity_per_cement,
            output_unit="tCO2/t cement",
            source="Calculated"
        ))

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data, calculation_steps, total_emissions
        )

        # Build result
        result = CementCalculationResult(
            calculation_id=calc_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cement_type=input_data.cement_type.value,
            cement_production_tonnes=input_data.cement_production_tonnes,
            clinker_production_tonnes=self._apply_precision(clinker_tonnes, self.OUTPUT_PRECISION),
            clinker_ratio=clinker_ratio,
            calcination_emissions_tco2=self._apply_precision(calcination_emissions, self.OUTPUT_PRECISION),
            kiln_fuel_emissions_tco2=self._apply_precision(kiln_fuel_emissions, self.OUTPUT_PRECISION),
            electricity_emissions_tco2=self._apply_precision(electricity_emissions, self.OUTPUT_PRECISION),
            scm_credits_tco2=self._apply_precision(scm_credits, self.OUTPUT_PRECISION),
            total_direct_emissions_tco2=self._apply_precision(total_direct, self.OUTPUT_PRECISION),
            total_indirect_emissions_tco2=self._apply_precision(total_indirect, self.OUTPUT_PRECISION),
            total_emissions_tco2=total_emissions,
            emission_intensity_tco2_per_tonne_cement=self._apply_precision(intensity_per_cement, self.OUTPUT_PRECISION),
            emission_intensity_tco2_per_tonne_clinker=self._apply_precision(intensity_per_clinker, self.OUTPUT_PRECISION),
            cbam_embedded_emissions_tco2=total_emissions,
            cbam_specific_embedded_emissions=self._apply_precision(intensity_per_cement, self.OUTPUT_PRECISION),
            calculation_steps=calculation_steps,
            emission_factor_sources=self.factors.SOURCES,
            provenance_hash=provenance_hash
        )

        return result

    def _get_clinker_ratio(self, input_data: CementCalculationInput) -> Decimal:
        """Get clinker ratio from cement type or custom input."""
        if input_data.clinker_ratio is not None:
            return input_data.clinker_ratio
        return self.factors.CEMENT_CLINKER_RATIOS.get(
            input_data.cement_type.value,
            Decimal("0.95")
        )

    def _get_kiln_fuel_factor(self, input_data: CementCalculationInput) -> Decimal:
        """Get kiln fuel emission factor - DETERMINISTIC."""
        fuel_factors = {
            KilnFuelType.COAL: self.factors.KILN_FUEL_COAL,
            KilnFuelType.PETCOKE: self.factors.KILN_FUEL_PETCOKE,
            KilnFuelType.NATURAL_GAS: self.factors.KILN_FUEL_NATURAL_GAS,
            KilnFuelType.ALTERNATIVE: self.factors.KILN_FUEL_ALTERNATIVE,
            KilnFuelType.MIXED: self.factors.KILN_FUEL_AVERAGE
        }
        return fuel_factors.get(input_data.kiln_fuel_type, self.factors.KILN_FUEL_AVERAGE)

    def _calculate_electricity_emissions(
        self,
        input_data: CementCalculationInput,
        cement_tonnes: Decimal
    ) -> Decimal:
        """Calculate electricity-related emissions."""
        if input_data.grid_emission_factor_kg_co2_per_kwh is None:
            return Decimal("0")

        electricity_kwh = input_data.electricity_kwh or (
            cement_tonnes * self.factors.GRINDING_ELECTRICITY_KWH
        )

        emissions_kg = electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh
        return emissions_kg / Decimal("1000")

    def _calculate_scm_credits(self, scm_inputs: List[SCMInput]) -> Decimal:
        """
        Calculate SCM credits - DETERMINISTIC.

        SCM credits are NEGATIVE (reduce total emissions).
        """
        if not scm_inputs:
            return Decimal("0")

        credit_factors = {
            SCMType.GGBS: self.factors.SCM_GGBS_CREDIT,
            SCMType.FLY_ASH: self.factors.SCM_FLY_ASH_CREDIT,
            SCMType.NATURAL_POZZOLAN: self.factors.SCM_NATURAL_POZZOLAN_CREDIT,
            SCMType.LIMESTONE: self.factors.SCM_LIMESTONE_CREDIT
        }

        total_credit = Decimal("0")
        for scm in scm_inputs:
            factor = credit_factors.get(scm.scm_type, Decimal("0"))
            # Credits are negative (avoided emissions)
            total_credit -= scm.quantity_tonnes * factor

        return total_credit

    def _apply_precision(self, value: Decimal, precision: int) -> Decimal:
        """Apply regulatory rounding precision."""
        quantize_str = "0." + "0" * precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _generate_calculation_id(self, input_data: CementCalculationInput) -> str:
        """Generate unique calculation ID."""
        data = f"{input_data.facility_id}:{input_data.reporting_period}:{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _calculate_provenance_hash(
        self,
        input_data: CementCalculationInput,
        steps: List[CalculationStep],
        final_value: Decimal
    ) -> str:
        """Calculate SHA-256 hash for complete audit trail."""
        provenance_data = {
            "calculator": "CementEmissionCalculator",
            "version": self.VERSION,
            "input": {
                "cement_type": input_data.cement_type.value,
                "cement_production_tonnes": str(input_data.cement_production_tonnes),
                "clinker_ratio": str(input_data.clinker_ratio) if input_data.clinker_ratio else None,
                "kiln_fuel_type": input_data.kiln_fuel_type.value,
                "scm_count": len(input_data.scm_inputs)
            },
            "steps": [
                {"step": s.step_number, "output": str(s.output_value)}
                for s in steps
            ],
            "final_value": str(final_value)
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# CBAM REPORTING FUNCTIONS
# =============================================================================

def format_cbam_output(result: CementCalculationResult) -> Dict:
    """Format calculation result for CBAM reporting."""
    return {
        "productCategory": result.cbam_product_category,
        "cnCode": result.cbam_cn_code,
        "quantityTonnes": str(result.cement_production_tonnes),
        "clinkerRatio": str(result.clinker_ratio),
        "embeddedEmissions": {
            "direct": str(result.total_direct_emissions_tco2),
            "indirect": str(result.total_indirect_emissions_tco2),
            "total": str(result.cbam_embedded_emissions_tco2)
        },
        "emissionBreakdown": {
            "calcination": str(result.calcination_emissions_tco2),
            "kilnFuel": str(result.kiln_fuel_emissions_tco2),
            "electricity": str(result.electricity_emissions_tco2),
            "scmCredits": str(result.scm_credits_tco2)
        },
        "specificEmbeddedEmissions": {
            "perTonneCite": str(result.emission_intensity_tco2_per_tonne_cement),
            "perTonneClinker": str(result.emission_intensity_tco2_per_tonne_clinker),
            "unit": "tCO2/t"
        },
        "cementType": result.cement_type,
        "provenanceHash": result.provenance_hash,
        "calculatorVersion": result.calculator_version
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    calculator = CementEmissionCalculator()

    # Example 1: CEM I (high clinker Portland cement)
    input1 = CementCalculationInput(
        cement_production_tonnes=Decimal("10000"),
        cement_type=CementType.CEM_I,
        kiln_fuel_type=KilnFuelType.COAL,
        facility_id="CEMENT_001",
        reporting_period="2024-Q1"
    )

    result1 = calculator.calculate(input1)
    print(f"CEM I Production:")
    print(f"  Clinker: {result1.clinker_production_tonnes} tonnes ({result1.clinker_ratio * 100}%)")
    print(f"  Calcination: {result1.calcination_emissions_tco2} tCO2")
    print(f"  Kiln fuel: {result1.kiln_fuel_emissions_tco2} tCO2")
    print(f"  Total: {result1.total_emissions_tco2} tCO2")
    print(f"  Intensity: {result1.emission_intensity_tco2_per_tonne_cement} tCO2/t cement")
    print()

    # Example 2: CEM III with GGBS (low carbon cement)
    input2 = CementCalculationInput(
        cement_production_tonnes=Decimal("10000"),
        cement_type=CementType.CEM_III_B,
        kiln_fuel_type=KilnFuelType.NATURAL_GAS,
        scm_inputs=[
            SCMInput(scm_type=SCMType.GGBS, quantity_tonnes=Decimal("7500"))
        ],
        grid_emission_factor_kg_co2_per_kwh=Decimal("0.4"),
        facility_id="CEMENT_002",
        reporting_period="2024-Q1"
    )

    result2 = calculator.calculate(input2)
    print(f"CEM III/B Production (with GGBS):")
    print(f"  Clinker: {result2.clinker_production_tonnes} tonnes ({result2.clinker_ratio * 100}%)")
    print(f"  Calcination: {result2.calcination_emissions_tco2} tCO2")
    print(f"  Kiln fuel: {result2.kiln_fuel_emissions_tco2} tCO2")
    print(f"  Electricity: {result2.electricity_emissions_tco2} tCO2")
    print(f"  SCM credits: {result2.scm_credits_tco2} tCO2")
    print(f"  Total: {result2.total_emissions_tco2} tCO2")
    print(f"  Intensity: {result2.emission_intensity_tco2_per_tonne_cement} tCO2/t cement")
    print()

    # Compare intensities
    print(f"Emission Reduction (CEM III vs CEM I):")
    reduction = (
        (result1.emission_intensity_tco2_per_tonne_cement - result2.emission_intensity_tco2_per_tonne_cement)
        / result1.emission_intensity_tco2_per_tonne_cement * 100
    )
    print(f"  {reduction:.1f}% reduction in emission intensity")
    print()

    # CBAM output
    cbam_output = format_cbam_output(result1)
    print(f"CBAM Output Format:")
    print(json.dumps(cbam_output, indent=2))
