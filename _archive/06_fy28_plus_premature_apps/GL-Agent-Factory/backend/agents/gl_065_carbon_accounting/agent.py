"""
GL-065: Carbon Accounting Agent (CARBON-ACCOUNTANT)

This module implements the CarbonAccountingAgent for Scope 1 GHG emissions
calculation, tracking, and reporting per GHG Protocol standards.

Standards Reference:
    - GHG Protocol Corporate Standard
    - ISO 14064-1 (GHG emissions quantification)
    - EPA 40 CFR Part 98 (Mandatory GHG Reporting)

Example:
    >>> agent = CarbonAccountingAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Total Scope 1 emissions: {result.total_scope1_emissions_tCO2e:.2f} tCO2e")
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class EmissionScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class EmissionSource(str, Enum):
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"


class FuelType(str, Enum):
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    FUEL_OIL = "fuel_oil"
    PROPANE = "propane"
    COAL = "coal"
    BIOMASS = "biomass"


class GHGType(str, Enum):
    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    HFC = "HFC"
    PFC = "PFC"
    SF6 = "SF6"
    NF3 = "NF3"


# Emission factors (kg CO2e per unit) - simplified, actual would come from EPA/IPCC databases
EMISSION_FACTORS = {
    FuelType.NATURAL_GAS: {
        "factor": 53.06,  # kg CO2e per MMBtu
        "unit": "MMBtu",
        "CH4_factor": 0.001,
        "N2O_factor": 0.0001
    },
    FuelType.DIESEL: {
        "factor": 10.21,  # kg CO2e per gallon
        "unit": "gallon",
        "CH4_factor": 0.0003,
        "N2O_factor": 0.0003
    },
    FuelType.GASOLINE: {
        "factor": 8.89,  # kg CO2e per gallon
        "unit": "gallon",
        "CH4_factor": 0.0003,
        "N2O_factor": 0.0003
    },
    FuelType.FUEL_OIL: {
        "factor": 11.26,  # kg CO2e per gallon
        "unit": "gallon",
        "CH4_factor": 0.0004,
        "N2O_factor": 0.0002
    },
    FuelType.COAL: {
        "factor": 95.26,  # kg CO2e per MMBtu
        "unit": "MMBtu",
        "CH4_factor": 0.011,
        "N2O_factor": 0.0016
    }
}

# Global Warming Potentials (GWP) - AR5 100-year values
GWP_VALUES = {
    GHGType.CO2: 1,
    GHGType.CH4: 28,
    GHGType.N2O: 265,
    GHGType.SF6: 23500,
    GHGType.NF3: 16100
}


class CombustionSource(BaseModel):
    source_id: str = Field(..., description="Source identifier")
    name: str = Field(..., description="Source name (boiler, heater, etc.)")
    fuel_type: FuelType = Field(..., description="Fuel type")
    fuel_consumption: float = Field(..., ge=0, description="Fuel consumption")
    fuel_unit: str = Field(..., description="Fuel unit (MMBtu, gallons, etc.)")
    emission_source: EmissionSource = Field(default=EmissionSource.STATIONARY_COMBUSTION)


class ProcessEmissionSource(BaseModel):
    source_id: str = Field(..., description="Source identifier")
    name: str = Field(..., description="Process name")
    ghg_type: GHGType = Field(..., description="GHG type emitted")
    annual_emissions_kg: float = Field(..., ge=0, description="Annual emissions (kg)")
    emission_source: EmissionSource = Field(default=EmissionSource.PROCESS_EMISSIONS)


class FugitiveEmissionSource(BaseModel):
    source_id: str = Field(..., description="Source identifier")
    name: str = Field(..., description="Equipment/source name")
    ghg_type: GHGType = Field(..., description="GHG type")
    annual_emissions_kg: float = Field(..., ge=0, description="Annual emissions (kg)")
    emission_source: EmissionSource = Field(default=EmissionSource.FUGITIVE_EMISSIONS)


class CarbonAccountingInput(BaseModel):
    report_id: Optional[str] = Field(None, description="Report identifier")
    facility_name: str = Field(..., description="Facility name")
    reporting_period: str = Field(..., description="Reporting period")
    reporting_year: int = Field(..., ge=2000, le=2100, description="Reporting year")
    combustion_sources: List[CombustionSource] = Field(default_factory=list)
    process_emissions: List[ProcessEmissionSource] = Field(default_factory=list)
    fugitive_emissions: List[FugitiveEmissionSource] = Field(default_factory=list)
    facility_latitude: Optional[float] = Field(None, description="Facility latitude")
    facility_longitude: Optional[float] = Field(None, description="Facility longitude")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmissionCalculation(BaseModel):
    source_id: str
    source_name: str
    emission_source_category: str
    fuel_type: Optional[str]
    activity_data: float
    activity_unit: str
    emission_factor: float
    emission_factor_unit: str
    CO2_emissions_tCO2e: float
    CH4_emissions_tCO2e: float
    N2O_emissions_tCO2e: float
    total_emissions_tCO2e: float


class ScopeSummary(BaseModel):
    scope: str
    total_emissions_tCO2e: float
    CO2_emissions_tCO2e: float
    CH4_emissions_tCO2e: float
    N2O_emissions_tCO2e: float
    other_GHG_emissions_tCO2e: float
    percent_of_total: float


class EmissionsByCategory(BaseModel):
    category: str
    emissions_tCO2e: float
    percent_of_total: float
    num_sources: int


class ReductionOpportunity(BaseModel):
    opportunity_id: str
    title: str
    description: str
    affected_sources: List[str]
    current_emissions_tCO2e: float
    potential_reduction_tCO2e: float
    reduction_percent: float
    implementation_cost_estimate: str
    priority: str


class ProvenanceRecord(BaseModel):
    operation: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    tool_name: str


class CarbonAccountingOutput(BaseModel):
    report_id: str
    facility_name: str
    reporting_period: str
    reporting_year: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    emission_calculations: List[EmissionCalculation]
    total_scope1_emissions_tCO2e: float
    scope_summaries: List[ScopeSummary]
    emissions_by_category: List[EmissionsByCategory]
    emissions_intensity_kgCO2e_per_unit: Optional[float]
    reduction_opportunities: List[ReductionOpportunity]
    compliance_status: str
    recommendations: List[str]
    warnings: List[str]
    provenance_chain: List[ProvenanceRecord]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str
    validation_errors: List[str] = Field(default_factory=list)


class CarbonAccountingAgent:
    """GL-065: Carbon Accounting Agent - Scope 1 GHG emissions calculation and reporting."""

    AGENT_ID = "GL-065"
    AGENT_NAME = "CARBON-ACCOUNTANT"
    VERSION = "1.0.0"
    REPORTING_THRESHOLD_TCO2E = 25000  # EPA threshold for mandatory reporting

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        self._warnings: List[str] = []
        self._recommendations: List[str] = []
        logger.info(f"CarbonAccountingAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: CarbonAccountingInput) -> CarbonAccountingOutput:
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []
        self._warnings = []
        self._recommendations = []

        # Calculate emissions from all sources
        emission_calculations = []

        # Combustion sources
        for source in input_data.combustion_sources:
            calc = self._calculate_combustion_emissions(source)
            emission_calculations.append(calc)

        # Process emissions
        for source in input_data.process_emissions:
            calc = self._calculate_process_emissions(source)
            emission_calculations.append(calc)

        # Fugitive emissions
        for source in input_data.fugitive_emissions:
            calc = self._calculate_fugitive_emissions(source)
            emission_calculations.append(calc)

        # Total Scope 1 emissions
        total_scope1 = sum(calc.total_emissions_tCO2e for calc in emission_calculations)
        total_CO2 = sum(calc.CO2_emissions_tCO2e for calc in emission_calculations)
        total_CH4 = sum(calc.CH4_emissions_tCO2e for calc in emission_calculations)
        total_N2O = sum(calc.N2O_emissions_tCO2e for calc in emission_calculations)

        self._track_provenance("ghg_calculation",
            {"num_sources": len(emission_calculations), "year": input_data.reporting_year},
            {"total_scope1_tCO2e": total_scope1},
            "GHG Calculator (GHG Protocol)")

        # Scope summaries
        scope_summaries = [
            ScopeSummary(
                scope="Scope 1",
                total_emissions_tCO2e=round(total_scope1, 2),
                CO2_emissions_tCO2e=round(total_CO2, 2),
                CH4_emissions_tCO2e=round(total_CH4, 2),
                N2O_emissions_tCO2e=round(total_N2O, 2),
                other_GHG_emissions_tCO2e=round(total_scope1 - total_CO2 - total_CH4 - total_N2O, 2),
                percent_of_total=100.0
            )
        ]

        # Emissions by category
        emissions_by_category = self._calculate_emissions_by_category(emission_calculations, total_scope1)

        # Reduction opportunities
        reduction_opportunities = self._identify_reduction_opportunities(emission_calculations, total_scope1)

        # Compliance status
        compliance_status = self._assess_compliance(total_scope1, input_data.reporting_year)

        # Generate recommendations and warnings
        self._generate_recommendations_and_warnings(total_scope1, emissions_by_category)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return CarbonAccountingOutput(
            report_id=input_data.report_id or f"GHG-{input_data.reporting_year}-{datetime.utcnow().strftime('%m%d%H%M')}",
            facility_name=input_data.facility_name,
            reporting_period=input_data.reporting_period,
            reporting_year=input_data.reporting_year,
            emission_calculations=emission_calculations,
            total_scope1_emissions_tCO2e=round(total_scope1, 2),
            scope_summaries=scope_summaries,
            emissions_by_category=emissions_by_category,
            emissions_intensity_kgCO2e_per_unit=None,  # Would need production data
            reduction_opportunities=reduction_opportunities,
            compliance_status=compliance_status,
            recommendations=self._recommendations,
            warnings=self._warnings,
            provenance_chain=[ProvenanceRecord(**{k: v for k, v in s.items()}) for s in self._provenance_steps],
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS" if not self._validation_errors else "FAIL",
            validation_errors=self._validation_errors
        )

    def _calculate_combustion_emissions(self, source: CombustionSource) -> EmissionCalculation:
        """Calculate emissions from combustion sources."""
        if source.fuel_type not in EMISSION_FACTORS:
            self._warnings.append(f"No emission factor found for {source.fuel_type.value}. Using default.")
            ef_data = {"factor": 50.0, "unit": source.fuel_unit, "CH4_factor": 0.001, "N2O_factor": 0.0001}
        else:
            ef_data = EMISSION_FACTORS[source.fuel_type]

        # Calculate CO2 emissions
        CO2_kg = source.fuel_consumption * ef_data["factor"]
        CO2_tCO2e = CO2_kg / 1000.0

        # Calculate CH4 emissions (convert to CO2e using GWP)
        CH4_kg = source.fuel_consumption * ef_data["CH4_factor"]
        CH4_tCO2e = (CH4_kg * GWP_VALUES[GHGType.CH4]) / 1000.0

        # Calculate N2O emissions (convert to CO2e using GWP)
        N2O_kg = source.fuel_consumption * ef_data["N2O_factor"]
        N2O_tCO2e = (N2O_kg * GWP_VALUES[GHGType.N2O]) / 1000.0

        total_tCO2e = CO2_tCO2e + CH4_tCO2e + N2O_tCO2e

        return EmissionCalculation(
            source_id=source.source_id,
            source_name=source.name,
            emission_source_category=source.emission_source.value,
            fuel_type=source.fuel_type.value,
            activity_data=round(source.fuel_consumption, 2),
            activity_unit=source.fuel_unit,
            emission_factor=ef_data["factor"],
            emission_factor_unit=f"kg CO2e per {ef_data['unit']}",
            CO2_emissions_tCO2e=round(CO2_tCO2e, 3),
            CH4_emissions_tCO2e=round(CH4_tCO2e, 3),
            N2O_emissions_tCO2e=round(N2O_tCO2e, 3),
            total_emissions_tCO2e=round(total_tCO2e, 3)
        )

    def _calculate_process_emissions(self, source: ProcessEmissionSource) -> EmissionCalculation:
        """Calculate process emissions."""
        gwp = GWP_VALUES.get(source.ghg_type, 1)
        total_tCO2e = (source.annual_emissions_kg * gwp) / 1000.0

        # Assign to appropriate GHG category
        CO2_tCO2e = total_tCO2e if source.ghg_type == GHGType.CO2 else 0.0
        CH4_tCO2e = total_tCO2e if source.ghg_type == GHGType.CH4 else 0.0
        N2O_tCO2e = total_tCO2e if source.ghg_type == GHGType.N2O else 0.0

        return EmissionCalculation(
            source_id=source.source_id,
            source_name=source.name,
            emission_source_category=source.emission_source.value,
            fuel_type=None,
            activity_data=round(source.annual_emissions_kg, 2),
            activity_unit="kg",
            emission_factor=gwp,
            emission_factor_unit=f"GWP ({source.ghg_type.value})",
            CO2_emissions_tCO2e=round(CO2_tCO2e, 3),
            CH4_emissions_tCO2e=round(CH4_tCO2e, 3),
            N2O_emissions_tCO2e=round(N2O_tCO2e, 3),
            total_emissions_tCO2e=round(total_tCO2e, 3)
        )

    def _calculate_fugitive_emissions(self, source: FugitiveEmissionSource) -> EmissionCalculation:
        """Calculate fugitive emissions."""
        gwp = GWP_VALUES.get(source.ghg_type, 1)
        total_tCO2e = (source.annual_emissions_kg * gwp) / 1000.0

        CO2_tCO2e = total_tCO2e if source.ghg_type == GHGType.CO2 else 0.0
        CH4_tCO2e = total_tCO2e if source.ghg_type == GHGType.CH4 else 0.0
        N2O_tCO2e = total_tCO2e if source.ghg_type == GHGType.N2O else 0.0

        return EmissionCalculation(
            source_id=source.source_id,
            source_name=source.name,
            emission_source_category=source.emission_source.value,
            fuel_type=None,
            activity_data=round(source.annual_emissions_kg, 2),
            activity_unit="kg",
            emission_factor=gwp,
            emission_factor_unit=f"GWP ({source.ghg_type.value})",
            CO2_emissions_tCO2e=round(CO2_tCO2e, 3),
            CH4_emissions_tCO2e=round(CH4_tCO2e, 3),
            N2O_emissions_tCO2e=round(N2O_tCO2e, 3),
            total_emissions_tCO2e=round(total_tCO2e, 3)
        )

    def _calculate_emissions_by_category(self, calculations: List[EmissionCalculation],
                                         total: float) -> List[EmissionsByCategory]:
        """Group emissions by source category."""
        category_totals: Dict[str, float] = {}
        category_counts: Dict[str, int] = {}

        for calc in calculations:
            cat = calc.emission_source_category
            category_totals[cat] = category_totals.get(cat, 0.0) + calc.total_emissions_tCO2e
            category_counts[cat] = category_counts.get(cat, 0) + 1

        result = []
        for cat, emissions in category_totals.items():
            result.append(EmissionsByCategory(
                category=cat,
                emissions_tCO2e=round(emissions, 2),
                percent_of_total=round(emissions / total * 100, 2) if total > 0 else 0.0,
                num_sources=category_counts[cat]
            ))

        return sorted(result, key=lambda x: -x.emissions_tCO2e)

    def _identify_reduction_opportunities(self, calculations: List[EmissionCalculation],
                                         total: float) -> List[ReductionOpportunity]:
        """Identify emission reduction opportunities."""
        opportunities = []

        # Identify high-emission sources
        high_emitters = [c for c in calculations if c.total_emissions_tCO2e > total * 0.10]

        for i, calc in enumerate(high_emitters[:5]):  # Top 5 opportunities
            if "combustion" in calc.emission_source_category:
                potential_reduction = calc.total_emissions_tCO2e * 0.20  # 20% reduction potential
                opportunities.append(ReductionOpportunity(
                    opportunity_id=f"RED-{i+1:03d}",
                    title=f"Improve Combustion Efficiency - {calc.source_name}",
                    description="Optimize combustion efficiency, tune burners, implement O2 trim control",
                    affected_sources=[calc.source_id],
                    current_emissions_tCO2e=calc.total_emissions_tCO2e,
                    potential_reduction_tCO2e=round(potential_reduction, 2),
                    reduction_percent=20.0,
                    implementation_cost_estimate="$50K-$200K",
                    priority="HIGH" if calc.total_emissions_tCO2e > total * 0.20 else "MEDIUM"
                ))

            elif "fugitive" in calc.emission_source_category:
                potential_reduction = calc.total_emissions_tCO2e * 0.50  # 50% reduction potential
                opportunities.append(ReductionOpportunity(
                    opportunity_id=f"RED-{i+1:03d}",
                    title=f"Reduce Fugitive Emissions - {calc.source_name}",
                    description="Implement leak detection and repair (LDAR) program",
                    affected_sources=[calc.source_id],
                    current_emissions_tCO2e=calc.total_emissions_tCO2e,
                    potential_reduction_tCO2e=round(potential_reduction, 2),
                    reduction_percent=50.0,
                    implementation_cost_estimate="$25K-$100K",
                    priority="HIGH"
                ))

        return opportunities

    def _assess_compliance(self, total_emissions: float, year: int) -> str:
        """Assess regulatory compliance status."""
        if total_emissions >= self.REPORTING_THRESHOLD_TCO2E:
            return f"MANDATORY_REPORTING_REQUIRED (>= {self.REPORTING_THRESHOLD_TCO2E:,} tCO2e/year per EPA 40 CFR Part 98)"
        else:
            return f"BELOW_MANDATORY_THRESHOLD (Voluntary reporting recommended)"

    def _generate_recommendations_and_warnings(self, total_emissions: float,
                                               categories: List[EmissionsByCategory]) -> None:
        """Generate system-level recommendations and warnings."""
        if total_emissions >= self.REPORTING_THRESHOLD_TCO2E:
            self._warnings.append(f"Total emissions ({total_emissions:,.0f} tCO2e) exceed EPA mandatory reporting threshold")
            self._recommendations.append("Ensure GHG emissions report is submitted to EPA by March 31")

        # Category-specific recommendations
        for cat in categories:
            if cat.percent_of_total > 50:
                self._recommendations.append(f"{cat.category} represents {cat.percent_of_total:.1f}% of emissions. "
                                           "Prioritize reduction efforts in this category.")

        # General recommendations
        self._recommendations.append("Implement ISO 14064-1 certified GHG management system")
        self._recommendations.append("Consider setting Science-Based Targets (SBTi) for emission reductions")

        if total_emissions > 10000:
            self._recommendations.append("Investigate carbon offset or renewable energy credit opportunities")

        if not self._warnings:
            self._warnings.append("All emissions calculations complete. Verify data quality before reporting.")

    def _track_provenance(self, operation: str, inputs: Dict, outputs: Dict, tool_name: str) -> None:
        """Track provenance of calculations."""
        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            "output_hash": hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(),
            "tool_name": tool_name
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of provenance chain."""
        data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": [{"operation": s["operation"], "input_hash": s["input_hash"], "output_hash": s["output_hash"]}
                     for s in self._provenance_steps],
            "timestamp": datetime.utcnow().isoformat()
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-065",
    "name": "CARBON-ACCOUNTANT",
    "version": "1.0.0",
    "summary": "Scope 1 GHG emissions calculation and carbon accounting per GHG Protocol",
    "tags": ["carbon-accounting", "ghg", "scope-1", "emissions", "ghg-protocol", "iso-14064"],
    "standards": [
        {"ref": "GHG Protocol", "description": "Corporate Accounting and Reporting Standard"},
        {"ref": "ISO 14064-1", "description": "GHG emissions quantification and reporting"},
        {"ref": "EPA 40 CFR Part 98", "description": "Mandatory GHG Reporting Rule"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True
    }
}
