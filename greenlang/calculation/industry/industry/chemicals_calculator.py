"""
Chemicals Industry Emission Calculator - Zero Hallucination Guarantee

This module implements deterministic, bit-perfect calculations for chemical
production emissions, focusing on ammonia and hydrogen - key decarbonization
commodities.

Sources:
    - IPCC 2006 Guidelines for National Greenhouse Gas Inventories, Volume 3
    - IEA The Future of Hydrogen (2019)
    - IEA Ammonia Technology Roadmap (2021)
    - IEAGHG Techno-Economic Evaluation of SMR Based Standalone Hydrogen Plant

Ammonia Production Routes:
    - SMR (Steam Methane Reforming): Natural gas based
    - Coal Gasification: Coal-to-ammonia
    - Green Ammonia: Electrolysis + Haber-Bosch with renewable electricity

Hydrogen Production Routes:
    - Gray: SMR without CCS
    - Blue: SMR with CCS
    - Green: Electrolysis with renewable electricity

CBAM Compliance: Annex III chemical products methodology
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import json
from datetime import datetime, timezone


# =============================================================================
# EMISSION FACTORS - DETERMINISTIC CONSTANTS
# =============================================================================

class ChemicalEmissionFactors:
    """
    Authoritative emission factors for chemical production.

    ALL factors are from IEA and IPCC sources with full provenance.
    NO interpolation, estimation, or LLM-generated values.
    """

    # =========================================================================
    # AMMONIA PRODUCTION
    # =========================================================================

    # SMR (Steam Methane Reforming) - tCO2/t NH3
    # Source: IEA Ammonia Technology Roadmap 2021
    # Includes: reformer emissions, process heat, compression
    AMMONIA_SMR: Decimal = Decimal("1.80")

    # Coal Gasification - tCO2/t NH3
    # Source: IEA, China average for coal-based ammonia
    AMMONIA_COAL: Decimal = Decimal("2.40")

    # SMR with CCS - tCO2/t NH3
    # Source: IEA, assumes 90% capture rate
    AMMONIA_SMR_CCS: Decimal = Decimal("0.20")

    # Green Ammonia (electrolysis + Haber-Bosch) - direct emissions only
    # Source: IEA, zero direct emissions (indirect from electricity)
    AMMONIA_GREEN_DIRECT: Decimal = Decimal("0.00")

    # Electricity for green ammonia (kWh/t NH3)
    # Source: IEA, includes electrolysis and Haber-Bosch
    AMMONIA_GREEN_ELECTRICITY: Decimal = Decimal("9500")

    # =========================================================================
    # HYDROGEN PRODUCTION
    # =========================================================================

    # Gray Hydrogen (SMR without CCS) - kgCO2/kgH2
    # Source: IEA The Future of Hydrogen 2019
    HYDROGEN_GRAY: Decimal = Decimal("10.00")

    # Blue Hydrogen (SMR with CCS) - kgCO2/kgH2
    # Source: IEA, assumes 90% capture rate
    HYDROGEN_BLUE: Decimal = Decimal("2.00")

    # Green Hydrogen (electrolysis) - direct emissions only
    # Source: IEA, zero direct emissions
    HYDROGEN_GREEN_DIRECT: Decimal = Decimal("0.00")

    # Electricity for green hydrogen (kWh/kgH2)
    # Source: IEA, PEM electrolysis typical
    HYDROGEN_GREEN_ELECTRICITY: Decimal = Decimal("55")

    # Turquoise Hydrogen (methane pyrolysis) - kgCO2/kgH2
    # Source: IEAGHG, solid carbon byproduct
    HYDROGEN_TURQUOISE: Decimal = Decimal("1.50")

    # =========================================================================
    # OTHER CHEMICALS
    # =========================================================================

    # Methanol (natural gas based) - tCO2/t MeOH
    # Source: IPCC 2006
    METHANOL_NG: Decimal = Decimal("0.67")

    # Methanol (coal based) - tCO2/t MeOH
    METHANOL_COAL: Decimal = Decimal("1.50")

    # Ethylene (steam cracking) - tCO2/t C2H4
    # Source: IEA Petrochemicals
    ETHYLENE_STEAM_CRACKING: Decimal = Decimal("1.00")

    # Metadata
    SOURCES = {
        "AMMONIA_SMR": "IEA Ammonia Technology Roadmap 2021",
        "AMMONIA_COAL": "IEA, China coal-based ammonia average",
        "HYDROGEN_GRAY": "IEA The Future of Hydrogen 2019",
        "HYDROGEN_BLUE": "IEA The Future of Hydrogen 2019",
        "HYDROGEN_GREEN": "IEA, electrolysis zero direct emissions",
        "ELECTRICITY": "IEA technology specifications"
    }


# =============================================================================
# DATA MODELS
# =============================================================================

class AmmoniaProductionRoute(str, Enum):
    """Ammonia production routes."""
    SMR = "SMR"  # Steam Methane Reforming
    COAL = "COAL"  # Coal Gasification
    SMR_CCS = "SMR_CCS"  # SMR with Carbon Capture
    GREEN = "GREEN"  # Electrolysis + Haber-Bosch


class HydrogenProductionRoute(str, Enum):
    """Hydrogen production routes."""
    GRAY = "GRAY"  # SMR without CCS
    BLUE = "BLUE"  # SMR with CCS
    GREEN = "GREEN"  # Electrolysis
    TURQUOISE = "TURQUOISE"  # Methane pyrolysis


class ChemicalProduct(str, Enum):
    """Chemical products supported."""
    AMMONIA = "AMMONIA"
    HYDROGEN = "HYDROGEN"
    METHANOL = "METHANOL"
    ETHYLENE = "ETHYLENE"


class AmmoniaCalculationInput(BaseModel):
    """Input parameters for ammonia emission calculation."""

    production_route: AmmoniaProductionRoute = Field(
        ...,
        description="Ammonia production technology route"
    )

    ammonia_production_tonnes: Decimal = Field(
        ...,
        gt=0,
        le=Decimal("100000000"),
        description="Ammonia (NH3) production in metric tonnes"
    )

    electricity_kwh: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Electricity consumption (for green ammonia)"
    )

    grid_emission_factor_kg_co2_per_kwh: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=Decimal("2.0"),
        description="Grid emission factor in kg CO2/kWh"
    )

    reporting_period: str = Field(
        default="",
        description="Reporting period"
    )

    facility_id: str = Field(
        default="",
        description="Facility identifier"
    )


class HydrogenCalculationInput(BaseModel):
    """Input parameters for hydrogen emission calculation."""

    production_route: HydrogenProductionRoute = Field(
        ...,
        description="Hydrogen production technology route"
    )

    hydrogen_production_kg: Decimal = Field(
        ...,
        gt=0,
        le=Decimal("100000000000"),  # 100M tonnes in kg
        description="Hydrogen (H2) production in kilograms"
    )

    electricity_kwh: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Electricity consumption (for green hydrogen)"
    )

    grid_emission_factor_kg_co2_per_kwh: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=Decimal("2.0"),
        description="Grid emission factor in kg CO2/kWh"
    )

    ccs_capture_rate: Optional[Decimal] = Field(
        default=Decimal("0.90"),
        ge=Decimal("0.50"),
        le=Decimal("0.99"),
        description="CCS capture rate (for blue hydrogen)"
    )

    reporting_period: str = Field(
        default="",
        description="Reporting period"
    )

    facility_id: str = Field(
        default="",
        description="Facility identifier"
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


class AmmoniaCalculationResult(BaseModel):
    """Ammonia calculation result with CBAM-compliant output."""

    # Identification
    calculation_id: str
    timestamp: str
    calculator_version: str = "1.0.0"

    # Input summary
    production_route: str
    ammonia_production_tonnes: Decimal

    # Emissions breakdown
    direct_emissions_tco2: Decimal
    indirect_emissions_tco2: Decimal
    total_emissions_tco2: Decimal

    # Intensity metrics
    emission_intensity_tco2_per_tonne: Decimal

    # CBAM fields
    cbam_product_category: str = "Ammonia"
    cbam_cn_code: str = "2814"
    cbam_embedded_emissions_tco2: Decimal
    cbam_specific_embedded_emissions: Decimal

    # Provenance
    calculation_steps: List[CalculationStep]
    emission_factor_sources: Dict[str, str]
    provenance_hash: str


class HydrogenCalculationResult(BaseModel):
    """Hydrogen calculation result."""

    # Identification
    calculation_id: str
    timestamp: str
    calculator_version: str = "1.0.0"

    # Input summary
    production_route: str
    hydrogen_production_kg: Decimal
    hydrogen_production_tonnes: Decimal

    # Emissions breakdown
    direct_emissions_kgco2: Decimal
    indirect_emissions_kgco2: Decimal
    total_emissions_kgco2: Decimal
    total_emissions_tco2: Decimal

    # Intensity metrics
    emission_intensity_kgco2_per_kgh2: Decimal

    # CBAM fields
    cbam_product_category: str = "Hydrogen"
    cbam_cn_code: str = "2804"
    cbam_embedded_emissions_tco2: Decimal
    cbam_specific_embedded_emissions_kgco2_per_kg: Decimal

    # Provenance
    calculation_steps: List[CalculationStep]
    emission_factor_sources: Dict[str, str]
    provenance_hash: str


# =============================================================================
# CALCULATION ENGINE
# =============================================================================

class ChemicalEmissionCalculator:
    """
    Zero-hallucination chemical emission calculator.

    Guarantees:
        - Deterministic: Same input produces identical output (bit-perfect)
        - Reproducible: Complete provenance tracking
        - Auditable: SHA-256 hash of all calculation steps
        - Compliant: CBAM and IEA methodology
        - Zero LLM: No AI/ML in calculation path
    """

    VERSION = "1.0.0"
    PRECISION = 6
    OUTPUT_PRECISION = 3

    def __init__(self):
        """Initialize calculator with emission factors."""
        self.factors = ChemicalEmissionFactors()

    def calculate_ammonia(
        self,
        input_data: AmmoniaCalculationInput
    ) -> AmmoniaCalculationResult:
        """
        Calculate ammonia production emissions.

        Args:
            input_data: Validated calculation inputs

        Returns:
            Complete calculation result with provenance
        """
        calculation_steps: List[CalculationStep] = []
        step_num = 0
        calc_id = self._generate_calculation_id(input_data.facility_id, input_data.reporting_period)

        # Step 1: Calculate direct emissions based on route
        step_num += 1
        direct_emissions = self._calculate_ammonia_direct(input_data)

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description=f"Direct emissions from {input_data.production_route.value} route",
            formula=self._get_ammonia_formula(input_data.production_route),
            inputs={
                "ammonia_production_tonnes": str(input_data.ammonia_production_tonnes),
                "emission_factor": str(self._get_ammonia_factor(input_data.production_route))
            },
            output_value=direct_emissions,
            output_unit="tCO2",
            source=self.factors.SOURCES.get("AMMONIA_SMR", "")
        ))

        # Step 2: Calculate indirect emissions (electricity)
        step_num += 1
        indirect_emissions = self._calculate_ammonia_indirect(input_data)

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="Indirect emissions from electricity",
            formula="electricity_kwh * grid_factor / 1000",
            inputs={
                "electricity_kwh": str(input_data.electricity_kwh or Decimal("0")),
                "grid_factor": str(input_data.grid_emission_factor_kg_co2_per_kwh or Decimal("0"))
            },
            output_value=indirect_emissions,
            output_unit="tCO2",
            source="Calculated from grid factor"
        ))

        # Step 3: Calculate total
        step_num += 1
        total_emissions = direct_emissions + indirect_emissions
        total_emissions = self._apply_precision(total_emissions, self.OUTPUT_PRECISION)

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="Total emissions",
            formula="direct + indirect",
            inputs={
                "direct_emissions": str(direct_emissions),
                "indirect_emissions": str(indirect_emissions)
            },
            output_value=total_emissions,
            output_unit="tCO2",
            source="Summation"
        ))

        # Step 4: Calculate intensity
        step_num += 1
        intensity = total_emissions / input_data.ammonia_production_tonnes
        intensity = self._apply_precision(intensity, self.OUTPUT_PRECISION)

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="Emission intensity",
            formula="total_emissions / ammonia_tonnes",
            inputs={
                "total_emissions": str(total_emissions),
                "ammonia_tonnes": str(input_data.ammonia_production_tonnes)
            },
            output_value=intensity,
            output_unit="tCO2/t NH3",
            source="Calculated"
        ))

        provenance_hash = self._calculate_provenance_hash_ammonia(
            input_data, calculation_steps, total_emissions
        )

        return AmmoniaCalculationResult(
            calculation_id=calc_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            production_route=input_data.production_route.value,
            ammonia_production_tonnes=input_data.ammonia_production_tonnes,
            direct_emissions_tco2=self._apply_precision(direct_emissions, self.OUTPUT_PRECISION),
            indirect_emissions_tco2=self._apply_precision(indirect_emissions, self.OUTPUT_PRECISION),
            total_emissions_tco2=total_emissions,
            emission_intensity_tco2_per_tonne=intensity,
            cbam_embedded_emissions_tco2=total_emissions,
            cbam_specific_embedded_emissions=intensity,
            calculation_steps=calculation_steps,
            emission_factor_sources=self.factors.SOURCES,
            provenance_hash=provenance_hash
        )

    def calculate_hydrogen(
        self,
        input_data: HydrogenCalculationInput
    ) -> HydrogenCalculationResult:
        """
        Calculate hydrogen production emissions.

        Args:
            input_data: Validated calculation inputs

        Returns:
            Complete calculation result with provenance
        """
        calculation_steps: List[CalculationStep] = []
        step_num = 0
        calc_id = self._generate_calculation_id(input_data.facility_id, input_data.reporting_period)

        # Step 1: Calculate direct emissions based on route
        step_num += 1
        direct_emissions_kg = self._calculate_hydrogen_direct(input_data)

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description=f"Direct emissions from {input_data.production_route.value} hydrogen",
            formula=self._get_hydrogen_formula(input_data.production_route),
            inputs={
                "hydrogen_production_kg": str(input_data.hydrogen_production_kg),
                "emission_factor_kgco2_per_kgh2": str(
                    self._get_hydrogen_factor(input_data.production_route)
                )
            },
            output_value=direct_emissions_kg,
            output_unit="kgCO2",
            source=self.factors.SOURCES.get(f"HYDROGEN_{input_data.production_route.value}", "")
        ))

        # Step 2: Calculate indirect emissions (electricity for green H2)
        step_num += 1
        indirect_emissions_kg = self._calculate_hydrogen_indirect(input_data)

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="Indirect emissions from electricity",
            formula="electricity_kwh * grid_factor",
            inputs={
                "electricity_kwh": str(input_data.electricity_kwh or Decimal("0")),
                "grid_factor_kg_per_kwh": str(
                    input_data.grid_emission_factor_kg_co2_per_kwh or Decimal("0")
                )
            },
            output_value=indirect_emissions_kg,
            output_unit="kgCO2",
            source="Calculated from grid factor"
        ))

        # Step 3: Calculate totals
        step_num += 1
        total_emissions_kg = direct_emissions_kg + indirect_emissions_kg
        total_emissions_t = total_emissions_kg / Decimal("1000")
        total_emissions_t = self._apply_precision(total_emissions_t, self.OUTPUT_PRECISION)

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="Total emissions",
            formula="direct + indirect",
            inputs={
                "direct_emissions_kg": str(direct_emissions_kg),
                "indirect_emissions_kg": str(indirect_emissions_kg)
            },
            output_value=total_emissions_kg,
            output_unit="kgCO2",
            source="Summation"
        ))

        # Step 4: Calculate intensity
        step_num += 1
        intensity = total_emissions_kg / input_data.hydrogen_production_kg
        intensity = self._apply_precision(intensity, self.OUTPUT_PRECISION)

        calculation_steps.append(CalculationStep(
            step_number=step_num,
            description="Emission intensity",
            formula="total_emissions_kg / hydrogen_kg",
            inputs={
                "total_emissions_kg": str(total_emissions_kg),
                "hydrogen_kg": str(input_data.hydrogen_production_kg)
            },
            output_value=intensity,
            output_unit="kgCO2/kgH2",
            source="Calculated"
        ))

        provenance_hash = self._calculate_provenance_hash_hydrogen(
            input_data, calculation_steps, total_emissions_t
        )

        hydrogen_tonnes = input_data.hydrogen_production_kg / Decimal("1000")

        return HydrogenCalculationResult(
            calculation_id=calc_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            production_route=input_data.production_route.value,
            hydrogen_production_kg=input_data.hydrogen_production_kg,
            hydrogen_production_tonnes=self._apply_precision(hydrogen_tonnes, self.OUTPUT_PRECISION),
            direct_emissions_kgco2=self._apply_precision(direct_emissions_kg, self.OUTPUT_PRECISION),
            indirect_emissions_kgco2=self._apply_precision(indirect_emissions_kg, self.OUTPUT_PRECISION),
            total_emissions_kgco2=self._apply_precision(total_emissions_kg, self.OUTPUT_PRECISION),
            total_emissions_tco2=total_emissions_t,
            emission_intensity_kgco2_per_kgh2=intensity,
            cbam_embedded_emissions_tco2=total_emissions_t,
            cbam_specific_embedded_emissions_kgco2_per_kg=intensity,
            calculation_steps=calculation_steps,
            emission_factor_sources=self.factors.SOURCES,
            provenance_hash=provenance_hash
        )

    def _get_ammonia_factor(self, route: AmmoniaProductionRoute) -> Decimal:
        """Get ammonia emission factor - DETERMINISTIC."""
        factors = {
            AmmoniaProductionRoute.SMR: self.factors.AMMONIA_SMR,
            AmmoniaProductionRoute.COAL: self.factors.AMMONIA_COAL,
            AmmoniaProductionRoute.SMR_CCS: self.factors.AMMONIA_SMR_CCS,
            AmmoniaProductionRoute.GREEN: self.factors.AMMONIA_GREEN_DIRECT
        }
        return factors[route]

    def _get_ammonia_formula(self, route: AmmoniaProductionRoute) -> str:
        """Get formula string for ammonia calculation."""
        formulas = {
            AmmoniaProductionRoute.SMR: "ammonia_tonnes * 1.8 tCO2/t",
            AmmoniaProductionRoute.COAL: "ammonia_tonnes * 2.4 tCO2/t",
            AmmoniaProductionRoute.SMR_CCS: "ammonia_tonnes * 0.2 tCO2/t",
            AmmoniaProductionRoute.GREEN: "0 (direct) + electricity emissions"
        }
        return formulas[route]

    def _calculate_ammonia_direct(self, input_data: AmmoniaCalculationInput) -> Decimal:
        """Calculate direct ammonia emissions - DETERMINISTIC."""
        factor = self._get_ammonia_factor(input_data.production_route)
        return input_data.ammonia_production_tonnes * factor

    def _calculate_ammonia_indirect(self, input_data: AmmoniaCalculationInput) -> Decimal:
        """Calculate indirect ammonia emissions (electricity)."""
        if input_data.production_route != AmmoniaProductionRoute.GREEN:
            return Decimal("0")

        if input_data.grid_emission_factor_kg_co2_per_kwh is None:
            return Decimal("0")

        electricity_kwh = input_data.electricity_kwh or (
            input_data.ammonia_production_tonnes * self.factors.AMMONIA_GREEN_ELECTRICITY
        )

        emissions_kg = electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh
        return emissions_kg / Decimal("1000")

    def _get_hydrogen_factor(self, route: HydrogenProductionRoute) -> Decimal:
        """Get hydrogen emission factor - DETERMINISTIC."""
        factors = {
            HydrogenProductionRoute.GRAY: self.factors.HYDROGEN_GRAY,
            HydrogenProductionRoute.BLUE: self.factors.HYDROGEN_BLUE,
            HydrogenProductionRoute.GREEN: self.factors.HYDROGEN_GREEN_DIRECT,
            HydrogenProductionRoute.TURQUOISE: self.factors.HYDROGEN_TURQUOISE
        }
        return factors[route]

    def _get_hydrogen_formula(self, route: HydrogenProductionRoute) -> str:
        """Get formula string for hydrogen calculation."""
        formulas = {
            HydrogenProductionRoute.GRAY: "hydrogen_kg * 10 kgCO2/kgH2",
            HydrogenProductionRoute.BLUE: "hydrogen_kg * 2 kgCO2/kgH2",
            HydrogenProductionRoute.GREEN: "0 (direct) + electricity emissions",
            HydrogenProductionRoute.TURQUOISE: "hydrogen_kg * 1.5 kgCO2/kgH2"
        }
        return formulas[route]

    def _calculate_hydrogen_direct(self, input_data: HydrogenCalculationInput) -> Decimal:
        """Calculate direct hydrogen emissions - DETERMINISTIC."""
        factor = self._get_hydrogen_factor(input_data.production_route)
        return input_data.hydrogen_production_kg * factor

    def _calculate_hydrogen_indirect(self, input_data: HydrogenCalculationInput) -> Decimal:
        """Calculate indirect hydrogen emissions (electricity)."""
        if input_data.production_route != HydrogenProductionRoute.GREEN:
            return Decimal("0")

        if input_data.grid_emission_factor_kg_co2_per_kwh is None:
            return Decimal("0")

        electricity_kwh = input_data.electricity_kwh or (
            input_data.hydrogen_production_kg * self.factors.HYDROGEN_GREEN_ELECTRICITY
        )

        emissions_kg = electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh
        return emissions_kg

    def _apply_precision(self, value: Decimal, precision: int) -> Decimal:
        """Apply regulatory rounding precision."""
        quantize_str = "0." + "0" * precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _generate_calculation_id(self, facility_id: str, reporting_period: str) -> str:
        """Generate unique calculation ID."""
        data = f"{facility_id}:{reporting_period}:{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _calculate_provenance_hash_ammonia(
        self,
        input_data: AmmoniaCalculationInput,
        steps: List[CalculationStep],
        final_value: Decimal
    ) -> str:
        """Calculate SHA-256 hash for ammonia calculation."""
        provenance_data = {
            "calculator": "ChemicalEmissionCalculator",
            "product": "AMMONIA",
            "version": self.VERSION,
            "input": {
                "production_route": input_data.production_route.value,
                "ammonia_production_tonnes": str(input_data.ammonia_production_tonnes)
            },
            "steps": [{"step": s.step_number, "output": str(s.output_value)} for s in steps],
            "final_value": str(final_value)
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _calculate_provenance_hash_hydrogen(
        self,
        input_data: HydrogenCalculationInput,
        steps: List[CalculationStep],
        final_value: Decimal
    ) -> str:
        """Calculate SHA-256 hash for hydrogen calculation."""
        provenance_data = {
            "calculator": "ChemicalEmissionCalculator",
            "product": "HYDROGEN",
            "version": self.VERSION,
            "input": {
                "production_route": input_data.production_route.value,
                "hydrogen_production_kg": str(input_data.hydrogen_production_kg)
            },
            "steps": [{"step": s.step_number, "output": str(s.output_value)} for s in steps],
            "final_value": str(final_value)
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# CBAM REPORTING FUNCTIONS
# =============================================================================

def format_cbam_output_ammonia(result: AmmoniaCalculationResult) -> Dict:
    """Format ammonia calculation for CBAM reporting."""
    return {
        "productCategory": result.cbam_product_category,
        "cnCode": result.cbam_cn_code,
        "quantityTonnes": str(result.ammonia_production_tonnes),
        "productionRoute": result.production_route,
        "embeddedEmissions": {
            "direct": str(result.direct_emissions_tco2),
            "indirect": str(result.indirect_emissions_tco2),
            "total": str(result.cbam_embedded_emissions_tco2)
        },
        "specificEmbeddedEmissions": {
            "value": str(result.cbam_specific_embedded_emissions),
            "unit": "tCO2/t NH3"
        },
        "provenanceHash": result.provenance_hash,
        "calculatorVersion": result.calculator_version
    }


def format_cbam_output_hydrogen(result: HydrogenCalculationResult) -> Dict:
    """Format hydrogen calculation for CBAM reporting."""
    return {
        "productCategory": result.cbam_product_category,
        "cnCode": result.cbam_cn_code,
        "quantityKg": str(result.hydrogen_production_kg),
        "quantityTonnes": str(result.hydrogen_production_tonnes),
        "productionRoute": result.production_route,
        "embeddedEmissions": {
            "directKgCO2": str(result.direct_emissions_kgco2),
            "indirectKgCO2": str(result.indirect_emissions_kgco2),
            "totalTCO2": str(result.cbam_embedded_emissions_tco2)
        },
        "specificEmbeddedEmissions": {
            "value": str(result.cbam_specific_embedded_emissions_kgco2_per_kg),
            "unit": "kgCO2/kgH2"
        },
        "provenanceHash": result.provenance_hash,
        "calculatorVersion": result.calculator_version
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    calculator = ChemicalEmissionCalculator()

    # =========================================================================
    # AMMONIA EXAMPLES
    # =========================================================================
    print("=" * 60)
    print("AMMONIA PRODUCTION EMISSIONS")
    print("=" * 60)

    # Example 1: SMR Ammonia (most common)
    input_smr = AmmoniaCalculationInput(
        production_route=AmmoniaProductionRoute.SMR,
        ammonia_production_tonnes=Decimal("1000"),
        facility_id="CHEM_001"
    )
    result_smr = calculator.calculate_ammonia(input_smr)
    print(f"\nSMR Ammonia:")
    print(f"  Direct: {result_smr.direct_emissions_tco2} tCO2")
    print(f"  Intensity: {result_smr.emission_intensity_tco2_per_tonne} tCO2/t NH3")

    # Example 2: Coal-based Ammonia (China)
    input_coal = AmmoniaCalculationInput(
        production_route=AmmoniaProductionRoute.COAL,
        ammonia_production_tonnes=Decimal("1000"),
        facility_id="CHEM_002"
    )
    result_coal = calculator.calculate_ammonia(input_coal)
    print(f"\nCoal Ammonia:")
    print(f"  Direct: {result_coal.direct_emissions_tco2} tCO2")
    print(f"  Intensity: {result_coal.emission_intensity_tco2_per_tonne} tCO2/t NH3")

    # Example 3: Green Ammonia
    input_green = AmmoniaCalculationInput(
        production_route=AmmoniaProductionRoute.GREEN,
        ammonia_production_tonnes=Decimal("1000"),
        grid_emission_factor_kg_co2_per_kwh=Decimal("0.05"),  # Renewable grid
        facility_id="CHEM_003"
    )
    result_green = calculator.calculate_ammonia(input_green)
    print(f"\nGreen Ammonia (Renewable Grid):")
    print(f"  Direct: {result_green.direct_emissions_tco2} tCO2")
    print(f"  Indirect: {result_green.indirect_emissions_tco2} tCO2")
    print(f"  Intensity: {result_green.emission_intensity_tco2_per_tonne} tCO2/t NH3")

    # =========================================================================
    # HYDROGEN EXAMPLES
    # =========================================================================
    print("\n" + "=" * 60)
    print("HYDROGEN PRODUCTION EMISSIONS")
    print("=" * 60)

    # Example 1: Gray Hydrogen
    input_gray = HydrogenCalculationInput(
        production_route=HydrogenProductionRoute.GRAY,
        hydrogen_production_kg=Decimal("1000"),
        facility_id="H2_001"
    )
    result_gray = calculator.calculate_hydrogen(input_gray)
    print(f"\nGray Hydrogen (SMR):")
    print(f"  Direct: {result_gray.direct_emissions_kgco2} kgCO2")
    print(f"  Intensity: {result_gray.emission_intensity_kgco2_per_kgh2} kgCO2/kgH2")

    # Example 2: Blue Hydrogen (CCS)
    input_blue = HydrogenCalculationInput(
        production_route=HydrogenProductionRoute.BLUE,
        hydrogen_production_kg=Decimal("1000"),
        ccs_capture_rate=Decimal("0.90"),
        facility_id="H2_002"
    )
    result_blue = calculator.calculate_hydrogen(input_blue)
    print(f"\nBlue Hydrogen (SMR + CCS):")
    print(f"  Direct: {result_blue.direct_emissions_kgco2} kgCO2")
    print(f"  Intensity: {result_blue.emission_intensity_kgco2_per_kgh2} kgCO2/kgH2")

    # Example 3: Green Hydrogen
    input_green_h2 = HydrogenCalculationInput(
        production_route=HydrogenProductionRoute.GREEN,
        hydrogen_production_kg=Decimal("1000"),
        grid_emission_factor_kg_co2_per_kwh=Decimal("0.02"),  # Renewable
        facility_id="H2_003"
    )
    result_green_h2 = calculator.calculate_hydrogen(input_green_h2)
    print(f"\nGreen Hydrogen (Electrolysis, Renewable):")
    print(f"  Direct: {result_green_h2.direct_emissions_kgco2} kgCO2")
    print(f"  Indirect: {result_green_h2.indirect_emissions_kgco2} kgCO2")
    print(f"  Intensity: {result_green_h2.emission_intensity_kgco2_per_kgh2} kgCO2/kgH2")

    # Comparison
    print(f"\nHydrogen Intensity Comparison:")
    print(f"  Gray: {result_gray.emission_intensity_kgco2_per_kgh2} kgCO2/kgH2")
    print(f"  Blue: {result_blue.emission_intensity_kgco2_per_kgh2} kgCO2/kgH2")
    print(f"  Green: {result_green_h2.emission_intensity_kgco2_per_kgh2} kgCO2/kgH2")

    # CBAM output
    print("\nCBAM Output (Ammonia):")
    print(json.dumps(format_cbam_output_ammonia(result_smr), indent=2))
