"""
GreenLang Framework - Engineering Rationale Generator

Generates rule-based engineering explanations with citations to authoritative
standards including ASME, EPA, NIST, IAPWS, ISO, and ASHRAE.

Features:
- Rule-based explanation generation (zero-hallucination)
- Citations to engineering standards and regulatory documents
- Thermodynamic principle references with formulas
- Structured explanation format with provenance tracking
- Domain-specific explanation templates
- Multi-framework compliance mapping

Standards Coverage:
- ASME: Boiler and Pressure Vessel Code, Performance Test Codes
- EPA: Emission factors, environmental regulations
- NIST: Physical constants, measurement standards
- IAPWS: Steam and water property formulations
- ISO: Quality and environmental management
- ASHRAE: HVAC and refrigeration standards
- API: Petroleum industry standards
- IPCC: Climate change emission factors

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import json

from .explanation_schemas import (
    EngineeringRationale,
    StandardCitation,
    ThermodynamicPrinciple,
    StandardSource,
    PredictionType,
)

logger = logging.getLogger(__name__)


class CalculationType(Enum):
    """Types of engineering calculations."""
    HEAT_TRANSFER = "heat_transfer"
    COMBUSTION = "combustion"
    STEAM_PROPERTIES = "steam_properties"
    EMISSION_CALCULATION = "emission_calculation"
    EFFICIENCY_CALCULATION = "efficiency_calculation"
    PINCH_ANALYSIS = "pinch_analysis"
    PRESSURE_DROP = "pressure_drop"
    HEAT_EXCHANGER = "heat_exchanger"
    INSULATION = "insulation"
    STEAM_TRAP = "steam_trap"
    WATER_TREATMENT = "water_treatment"
    FUEL_OPTIMIZATION = "fuel_optimization"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"


class ThermodynamicLaw(Enum):
    """Fundamental thermodynamic laws."""
    FIRST_LAW = "first_law"
    SECOND_LAW = "second_law"
    ZEROTH_LAW = "zeroth_law"
    THIRD_LAW = "third_law"


# Standard Citations Database
STANDARD_CITATIONS: Dict[str, StandardCitation] = {
    # ASME Standards
    "ASME_PTC_4": StandardCitation(
        source=StandardSource.ASME,
        standard_id="PTC 4",
        section=None,
        year=2023,
        title="Fired Steam Generators Performance Test Code",
        url="https://www.asme.org/codes-standards/find-codes-standards/ptc-4"
    ),
    "ASME_PTC_4_1": StandardCitation(
        source=StandardSource.ASME,
        standard_id="PTC 4.1",
        year=1964,
        title="Steam Generating Units",
        url=None
    ),
    "ASME_PTC_12_5": StandardCitation(
        source=StandardSource.ASME,
        standard_id="PTC 12.5",
        year=2000,
        title="Single Phase Heat Exchangers",
        url=None
    ),
    "ASME_BPVC_I": StandardCitation(
        source=StandardSource.ASME,
        standard_id="BPVC Section I",
        year=2023,
        title="Rules for Construction of Power Boilers",
        url=None
    ),
    "ASME_BPVC_VIII": StandardCitation(
        source=StandardSource.ASME,
        standard_id="BPVC Section VIII",
        year=2023,
        title="Rules for Construction of Pressure Vessels",
        url=None
    ),

    # EPA Standards
    "EPA_AP42": StandardCitation(
        source=StandardSource.EPA,
        standard_id="AP-42",
        section="Chapter 1",
        year=2023,
        title="Compilation of Air Pollutant Emission Factors",
        url="https://www.epa.gov/air-emissions-factors-and-quantification/ap-42"
    ),
    "EPA_GHGRP": StandardCitation(
        source=StandardSource.EPA,
        standard_id="40 CFR Part 98",
        year=2023,
        title="Greenhouse Gas Reporting Program",
        url="https://www.epa.gov/ghgreporting"
    ),
    "EPA_EGRID": StandardCitation(
        source=StandardSource.EPA,
        standard_id="eGRID",
        year=2023,
        title="Emissions & Generation Resource Integrated Database",
        url="https://www.epa.gov/egrid"
    ),

    # NIST Standards
    "NIST_SP811": StandardCitation(
        source=StandardSource.NIST,
        standard_id="SP 811",
        year=2008,
        title="Guide for the Use of the International System of Units (SI)",
        url="https://www.nist.gov/pml/special-publication-811"
    ),
    "NIST_CODATA": StandardCitation(
        source=StandardSource.NIST,
        standard_id="CODATA 2018",
        year=2019,
        title="CODATA Internationally Recommended Values of Fundamental Physical Constants",
        url="https://physics.nist.gov/cuu/Constants/"
    ),

    # IAPWS Standards
    "IAPWS_IF97": StandardCitation(
        source=StandardSource.IAPWS,
        standard_id="IF-97",
        year=1997,
        title="Industrial Formulation 1997 for Thermodynamic Properties of Water and Steam",
        url="http://www.iapws.org/relguide/IF97.html"
    ),
    "IAPWS_2011": StandardCitation(
        source=StandardSource.IAPWS,
        standard_id="R7-97(2012)",
        year=2012,
        title="Revised Release on the IAPWS Industrial Formulation 1997",
        url=None
    ),
    "IAPWS_G5": StandardCitation(
        source=StandardSource.IAPWS,
        standard_id="G5-01(2020)",
        year=2020,
        title="Guideline on the Use of Fundamental Physical Constants and Basic Constants of Water",
        url=None
    ),

    # ISO Standards
    "ISO_14064": StandardCitation(
        source=StandardSource.ISO,
        standard_id="ISO 14064",
        year=2018,
        title="Greenhouse gases - Part 1: Specification for organizational GHG emissions and removals",
        url=None
    ),
    "ISO_50001": StandardCitation(
        source=StandardSource.ISO,
        standard_id="ISO 50001",
        year=2018,
        title="Energy management systems - Requirements with guidance for use",
        url=None
    ),
    "ISO_50006": StandardCitation(
        source=StandardSource.ISO,
        standard_id="ISO 50006",
        year=2014,
        title="Energy management systems - Measuring energy performance using energy baselines",
        url=None
    ),

    # ASHRAE Standards
    "ASHRAE_FUNDAMENTALS": StandardCitation(
        source=StandardSource.ASHRAE,
        standard_id="Handbook - Fundamentals",
        year=2021,
        title="ASHRAE Handbook - Fundamentals",
        url=None
    ),
    "ASHRAE_HVAC": StandardCitation(
        source=StandardSource.ASHRAE,
        standard_id="Handbook - HVAC Systems",
        year=2020,
        title="ASHRAE Handbook - HVAC Systems and Equipment",
        url=None
    ),

    # IPCC Guidelines
    "IPCC_2006": StandardCitation(
        source=StandardSource.IPCC,
        standard_id="2006 Guidelines",
        section="Volume 2",
        year=2006,
        title="IPCC Guidelines for National Greenhouse Gas Inventories",
        url="https://www.ipcc-nggip.iges.or.jp/public/2006gl/"
    ),
    "IPCC_AR6": StandardCitation(
        source=StandardSource.IPCC,
        standard_id="AR6",
        year=2021,
        title="IPCC Sixth Assessment Report",
        url=None
    ),

    # API Standards
    "API_560": StandardCitation(
        source=StandardSource.API,
        standard_id="API 560",
        year=2016,
        title="Fired Heaters for General Refinery Service",
        url=None
    ),
    "API_2000": StandardCitation(
        source=StandardSource.API,
        standard_id="API 2000",
        year=2014,
        title="Venting Atmospheric and Low-Pressure Storage Tanks",
        url=None
    ),

    # GHG Protocol
    "GHG_CORPORATE": StandardCitation(
        source=StandardSource.GHG_PROTOCOL,
        standard_id="Corporate Standard",
        year=2015,
        title="GHG Protocol Corporate Accounting and Reporting Standard",
        url="https://ghgprotocol.org/corporate-standard"
    ),
    "GHG_SCOPE3": StandardCitation(
        source=StandardSource.GHG_PROTOCOL,
        standard_id="Scope 3 Standard",
        year=2011,
        title="Corporate Value Chain (Scope 3) Accounting and Reporting Standard",
        url=None
    ),

    # DEFRA
    "DEFRA_EF": StandardCitation(
        source=StandardSource.DEFRA,
        standard_id="GHG Conversion Factors",
        year=2023,
        title="UK Government GHG Conversion Factors for Company Reporting",
        url="https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting"
    ),
}


# Thermodynamic Principles Database
THERMODYNAMIC_PRINCIPLES: Dict[str, ThermodynamicPrinciple] = {
    "first_law_energy_balance": ThermodynamicPrinciple(
        name="First Law of Thermodynamics - Energy Balance",
        formula="Q - W = Delta_U",
        description="Energy cannot be created or destroyed, only converted from one form to another. For a control volume: Q_in - Q_out + W_in - W_out = Delta_E_system",
        variables={
            "Q": "Heat transfer (J or kW)",
            "W": "Work done (J or kW)",
            "Delta_U": "Change in internal energy (J)",
            "Delta_E_system": "Change in system energy (J)"
        },
        citations=[STANDARD_CITATIONS["ASHRAE_FUNDAMENTALS"]]
    ),
    "heat_transfer_conduction": ThermodynamicPrinciple(
        name="Fourier's Law of Heat Conduction",
        formula="Q = -k * A * (dT/dx)",
        description="Heat transfer by conduction is proportional to temperature gradient and thermal conductivity",
        variables={
            "Q": "Heat transfer rate (W)",
            "k": "Thermal conductivity (W/m-K)",
            "A": "Cross-sectional area (m^2)",
            "dT/dx": "Temperature gradient (K/m)"
        },
        citations=[STANDARD_CITATIONS["ASHRAE_FUNDAMENTALS"]]
    ),
    "heat_transfer_convection": ThermodynamicPrinciple(
        name="Newton's Law of Cooling",
        formula="Q = h * A * (T_s - T_inf)",
        description="Heat transfer by convection is proportional to surface-fluid temperature difference",
        variables={
            "Q": "Heat transfer rate (W)",
            "h": "Convective heat transfer coefficient (W/m^2-K)",
            "A": "Surface area (m^2)",
            "T_s": "Surface temperature (K)",
            "T_inf": "Fluid temperature (K)"
        },
        citations=[STANDARD_CITATIONS["ASHRAE_FUNDAMENTALS"]]
    ),
    "heat_transfer_radiation": ThermodynamicPrinciple(
        name="Stefan-Boltzmann Law",
        formula="Q = epsilon * sigma * A * (T_s^4 - T_sur^4)",
        description="Heat transfer by radiation between a surface and its surroundings",
        variables={
            "Q": "Heat transfer rate (W)",
            "epsilon": "Surface emissivity (dimensionless, 0-1)",
            "sigma": "Stefan-Boltzmann constant (5.67e-8 W/m^2-K^4)",
            "A": "Surface area (m^2)",
            "T_s": "Surface temperature (K)",
            "T_sur": "Surrounding temperature (K)"
        },
        citations=[STANDARD_CITATIONS["NIST_CODATA"]]
    ),
    "combustion_efficiency": ThermodynamicPrinciple(
        name="Combustion Efficiency",
        formula="eta_c = (HHV - L_stack - L_unburned) / HHV * 100",
        description="Ratio of heat released to fuel higher heating value, accounting for stack and unburned losses",
        variables={
            "eta_c": "Combustion efficiency (%)",
            "HHV": "Higher heating value of fuel (kJ/kg)",
            "L_stack": "Stack heat losses (kJ/kg)",
            "L_unburned": "Unburned fuel losses (kJ/kg)"
        },
        citations=[STANDARD_CITATIONS["ASME_PTC_4"]]
    ),
    "boiler_efficiency": ThermodynamicPrinciple(
        name="Boiler Efficiency - Input-Output Method",
        formula="eta_b = (Q_out / Q_in) * 100",
        description="Ratio of useful heat output to fuel heat input",
        variables={
            "eta_b": "Boiler efficiency (%)",
            "Q_out": "Heat absorbed by steam/water (kJ)",
            "Q_in": "Heat input from fuel (kJ)"
        },
        citations=[STANDARD_CITATIONS["ASME_PTC_4"]]
    ),
    "steam_enthalpy": ThermodynamicPrinciple(
        name="Steam Enthalpy from IAPWS-IF97",
        formula="h = h(P, T) or h = h(P, x)",
        description="Specific enthalpy of steam as function of pressure and temperature (or quality)",
        variables={
            "h": "Specific enthalpy (kJ/kg)",
            "P": "Pressure (MPa)",
            "T": "Temperature (K)",
            "x": "Steam quality (dimensionless, 0-1)"
        },
        citations=[STANDARD_CITATIONS["IAPWS_IF97"]]
    ),
    "lmtd": ThermodynamicPrinciple(
        name="Log Mean Temperature Difference (LMTD)",
        formula="LMTD = (Delta_T1 - Delta_T2) / ln(Delta_T1 / Delta_T2)",
        description="Effective temperature difference for heat exchanger design",
        variables={
            "LMTD": "Log mean temperature difference (K)",
            "Delta_T1": "Temperature difference at one end (K)",
            "Delta_T2": "Temperature difference at other end (K)"
        },
        citations=[STANDARD_CITATIONS["ASME_PTC_12_5"]]
    ),
    "emission_factor": ThermodynamicPrinciple(
        name="Emission Factor Calculation",
        formula="E = A * EF",
        description="Total emissions calculated from activity data and emission factor",
        variables={
            "E": "Total emissions (kg CO2e)",
            "A": "Activity data (unit of activity)",
            "EF": "Emission factor (kg CO2e/unit)"
        },
        citations=[STANDARD_CITATIONS["EPA_AP42"], STANDARD_CITATIONS["IPCC_2006"]]
    ),
    "pinch_temperature": ThermodynamicPrinciple(
        name="Pinch Analysis - Minimum Temperature Approach",
        formula="Delta_T_min = T_hot - T_cold (at pinch point)",
        description="Minimum temperature difference between hot and cold streams determines maximum heat recovery",
        variables={
            "Delta_T_min": "Minimum temperature approach (K)",
            "T_hot": "Hot stream temperature at pinch (K)",
            "T_cold": "Cold stream temperature at pinch (K)"
        },
        citations=[STANDARD_CITATIONS["ISO_50001"]]
    ),
}


# Calculation-specific methodologies
CALCULATION_METHODOLOGIES: Dict[CalculationType, List[str]] = {
    CalculationType.HEAT_TRANSFER: [
        "Identify heat transfer mode(s): conduction, convection, and/or radiation",
        "Apply appropriate heat transfer equation for each mode",
        "Calculate thermal resistance for composite systems",
        "Sum heat transfer rates for parallel paths",
        "Account for transient effects if applicable"
    ],
    CalculationType.COMBUSTION: [
        "Determine fuel composition and heating value (HHV/LHV)",
        "Calculate stoichiometric air requirement",
        "Apply excess air factor based on equipment type",
        "Compute flue gas composition and temperature",
        "Calculate combustion efficiency per ASME PTC 4"
    ],
    CalculationType.STEAM_PROPERTIES: [
        "Identify thermodynamic state: subcooled, saturated, or superheated",
        "Apply IAPWS-IF97 formulation for region determination",
        "Calculate specific properties (h, s, v, u) from state equations",
        "Verify thermodynamic consistency using Maxwell relations",
        "Apply quality calculations for two-phase regions"
    ],
    CalculationType.EMISSION_CALCULATION: [
        "Identify emission scope (1, 2, or 3) per GHG Protocol",
        "Select appropriate emission factor from authoritative source",
        "Collect activity data with proper units",
        "Apply emission factor: Emissions = Activity x EF",
        "Convert to CO2 equivalent using GWP values from IPCC"
    ],
    CalculationType.EFFICIENCY_CALCULATION: [
        "Define system boundary and energy flows",
        "Identify all energy inputs (fuel, electricity, etc.)",
        "Identify all useful energy outputs",
        "Calculate efficiency: eta = Output / Input x 100%",
        "Compare against benchmark or design values"
    ],
    CalculationType.PINCH_ANALYSIS: [
        "Extract hot and cold stream data (temperatures, heat capacity rates)",
        "Construct composite curves for hot and cold streams",
        "Identify pinch point and minimum approach temperature",
        "Calculate minimum heating and cooling utilities",
        "Design heat exchanger network respecting pinch rules"
    ],
    CalculationType.HEAT_EXCHANGER: [
        "Determine heat exchanger type and configuration",
        "Calculate LMTD or use effectiveness-NTU method",
        "Apply correction factor for shell-and-tube configurations",
        "Calculate overall heat transfer coefficient (U)",
        "Size exchanger: Q = U x A x LMTD"
    ],
    CalculationType.INSULATION: [
        "Identify pipe/equipment geometry and surface conditions",
        "Determine ambient conditions (temperature, wind speed)",
        "Calculate heat loss without insulation",
        "Apply thermal resistance of insulation material",
        "Calculate heat loss reduction and energy savings"
    ],
    CalculationType.STEAM_TRAP: [
        "Identify trap type and operating conditions",
        "Measure inlet and outlet temperatures",
        "Assess trap condition: working, failed open, failed closed, or bypassing",
        "Calculate steam loss rate for failed traps",
        "Estimate annual energy and cost impact"
    ],
    CalculationType.WATER_TREATMENT: [
        "Analyze feedwater and makeup water quality",
        "Determine required cycles of concentration",
        "Calculate blowdown rate for dissolved solids control",
        "Size chemical treatment requirements",
        "Calculate energy loss from blowdown"
    ],
    CalculationType.FUEL_OPTIMIZATION: [
        "Analyze fuel composition and properties",
        "Calculate theoretical and actual combustion air",
        "Optimize excess air for efficiency vs. emissions",
        "Evaluate fuel switching opportunities",
        "Calculate cost and emission impacts of alternatives"
    ],
    CalculationType.PREDICTIVE_MAINTENANCE: [
        "Collect sensor data (vibration, temperature, etc.)",
        "Extract relevant features from time-series data",
        "Apply trained ML model for remaining useful life (RUL)",
        "Validate prediction against historical failure patterns",
        "Generate maintenance recommendations with confidence intervals"
    ],
}


# Calculation-specific citations
CALCULATION_CITATIONS: Dict[CalculationType, List[str]] = {
    CalculationType.HEAT_TRANSFER: [
        "ASHRAE_FUNDAMENTALS", "NIST_CODATA"
    ],
    CalculationType.COMBUSTION: [
        "ASME_PTC_4", "EPA_AP42", "API_560"
    ],
    CalculationType.STEAM_PROPERTIES: [
        "IAPWS_IF97", "IAPWS_2011", "IAPWS_G5"
    ],
    CalculationType.EMISSION_CALCULATION: [
        "EPA_AP42", "EPA_GHGRP", "IPCC_2006", "GHG_CORPORATE", "DEFRA_EF"
    ],
    CalculationType.EFFICIENCY_CALCULATION: [
        "ASME_PTC_4", "ISO_50001", "ISO_50006"
    ],
    CalculationType.PINCH_ANALYSIS: [
        "ISO_50001", "ASHRAE_FUNDAMENTALS"
    ],
    CalculationType.HEAT_EXCHANGER: [
        "ASME_PTC_12_5", "ASHRAE_HVAC"
    ],
    CalculationType.INSULATION: [
        "ASHRAE_FUNDAMENTALS", "ISO_50001"
    ],
    CalculationType.STEAM_TRAP: [
        "ASME_PTC_4_1", "ISO_50001"
    ],
    CalculationType.WATER_TREATMENT: [
        "ASME_BPVC_I", "ASHRAE_HVAC"
    ],
    CalculationType.FUEL_OPTIMIZATION: [
        "ASME_PTC_4", "EPA_AP42", "ISO_50001"
    ],
    CalculationType.PREDICTIVE_MAINTENANCE: [
        "ISO_50001", "API_560"
    ],
}


@dataclass
class RationaleConfig:
    """Configuration for engineering rationale generation."""
    include_formulas: bool = True
    include_citations: bool = True
    include_assumptions: bool = True
    include_limitations: bool = True
    include_principles: bool = True
    max_citations: int = 10
    max_assumptions: int = 10
    confidence_level: float = 0.95


class EngineeringRationaleGenerator:
    """
    Generates engineering rationales with standard citations.

    Provides rule-based, zero-hallucination explanations for
    engineering calculations with full traceability to
    authoritative standards.

    Example:
        >>> generator = EngineeringRationaleGenerator()
        >>> rationale = generator.generate_rationale(
        ...     calculation_type=CalculationType.COMBUSTION,
        ...     inputs={"fuel_type": "natural_gas", "mass_flow": 100},
        ...     outputs={"efficiency": 0.85, "emissions": 1200},
        ...     custom_assumptions=["Steady-state operation"]
        ... )
        >>> print(rationale.summary)
    """

    def __init__(
        self,
        config: Optional[RationaleConfig] = None,
        agent_id: str = "GL-FRAMEWORK",
        agent_version: str = "1.0.0"
    ):
        """
        Initialize engineering rationale generator.

        Args:
            config: Rationale generation configuration
            agent_id: Agent identifier for provenance tracking
            agent_version: Agent version for provenance tracking
        """
        self.config = config or RationaleConfig()
        self.agent_id = agent_id
        self.agent_version = agent_version

        logger.info(
            f"EngineeringRationaleGenerator initialized: agent={agent_id}"
        )

    def generate_rationale(
        self,
        calculation_type: CalculationType,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        custom_assumptions: Optional[List[str]] = None,
        custom_limitations: Optional[List[str]] = None,
        additional_citations: Optional[List[StandardCitation]] = None
    ) -> EngineeringRationale:
        """
        Generate engineering rationale for a calculation.

        Args:
            calculation_type: Type of engineering calculation
            inputs: Input parameters and values
            outputs: Output results and values
            custom_assumptions: Additional assumptions to include
            custom_limitations: Additional limitations to include
            additional_citations: Additional standard citations

        Returns:
            EngineeringRationale with complete documentation
        """
        # Generate rationale ID
        rationale_id = self._generate_rationale_id(calculation_type, inputs)

        # Get methodology
        methodology = self._get_methodology(calculation_type)

        # Get principles
        principles = self._get_principles(calculation_type)

        # Get citations
        citations = self._get_citations(calculation_type, additional_citations)

        # Get assumptions
        assumptions = self._get_assumptions(calculation_type, custom_assumptions)

        # Get limitations
        limitations = self._get_limitations(calculation_type, custom_limitations)

        # Generate summary
        summary = self._generate_summary(calculation_type, inputs, outputs)

        # Determine validation status
        validation_status = self._determine_validation_status(inputs, outputs)

        rationale = EngineeringRationale(
            rationale_id=rationale_id,
            calculation_type=calculation_type.value,
            summary=summary,
            methodology=methodology,
            principles=principles,
            citations=citations,
            assumptions=assumptions,
            limitations=limitations,
            input_parameters=inputs,
            output_results=outputs,
            validation_status=validation_status,
            confidence_level=self.config.confidence_level,
            timestamp=datetime.now(timezone.utc)
        )

        logger.info(
            f"Engineering rationale generated: id={rationale_id[:8]}, "
            f"type={calculation_type.value}"
        )

        return rationale

    def generate_thermodynamic_explanation(
        self,
        principle_key: str,
        inputs: Dict[str, Any],
        result: float
    ) -> Dict[str, Any]:
        """
        Generate explanation based on thermodynamic principle.

        Args:
            principle_key: Key from THERMODYNAMIC_PRINCIPLES
            inputs: Input values for the formula
            result: Calculated result

        Returns:
            Dictionary with explanation details
        """
        if principle_key not in THERMODYNAMIC_PRINCIPLES:
            raise ValueError(f"Unknown principle: {principle_key}")

        principle = THERMODYNAMIC_PRINCIPLES[principle_key]

        return {
            "principle_name": principle.name,
            "formula": principle.formula,
            "description": principle.description,
            "input_values": inputs,
            "result": result,
            "variables": principle.variables,
            "citations": [c.format_citation() for c in principle.citations]
        }

    def get_standard_citation(self, citation_key: str) -> Optional[StandardCitation]:
        """
        Get a standard citation by key.

        Args:
            citation_key: Key from STANDARD_CITATIONS

        Returns:
            StandardCitation or None if not found
        """
        return STANDARD_CITATIONS.get(citation_key)

    def get_principle(self, principle_key: str) -> Optional[ThermodynamicPrinciple]:
        """
        Get a thermodynamic principle by key.

        Args:
            principle_key: Key from THERMODYNAMIC_PRINCIPLES

        Returns:
            ThermodynamicPrinciple or None if not found
        """
        return THERMODYNAMIC_PRINCIPLES.get(principle_key)

    def format_citation_list(self, citations: List[StandardCitation]) -> str:
        """
        Format a list of citations as a reference string.

        Args:
            citations: List of citations

        Returns:
            Formatted citation string
        """
        return "; ".join(c.format_citation() for c in citations)

    def generate_compliance_mapping(
        self,
        calculation_type: CalculationType
    ) -> Dict[str, List[str]]:
        """
        Generate mapping of calculation to compliance frameworks.

        Args:
            calculation_type: Type of calculation

        Returns:
            Dictionary mapping frameworks to relevant sections
        """
        framework_mapping = {
            "GHG Protocol": [],
            "ISO 14064": [],
            "ISO 50001": [],
            "EPA GHGRP": [],
            "EU ETS": []
        }

        if calculation_type == CalculationType.EMISSION_CALCULATION:
            framework_mapping["GHG Protocol"] = [
                "Scope 1: Direct emissions",
                "Scope 2: Indirect emissions from purchased energy",
                "Scope 3: Other indirect emissions"
            ]
            framework_mapping["ISO 14064"] = [
                "Part 1: Organization level quantification",
                "Part 2: Project level quantification",
                "Part 3: Validation and verification"
            ]
            framework_mapping["EPA GHGRP"] = [
                "Subpart C: General Stationary Fuel Combustion",
                "Subpart D: Electricity Generation"
            ]

        elif calculation_type == CalculationType.EFFICIENCY_CALCULATION:
            framework_mapping["ISO 50001"] = [
                "4.4.3: Energy review",
                "4.4.5: Energy baseline",
                "4.4.6: Energy performance indicators"
            ]

        elif calculation_type == CalculationType.COMBUSTION:
            framework_mapping["EPA GHGRP"] = [
                "Subpart C: General Stationary Fuel Combustion"
            ]
            framework_mapping["ISO 50001"] = [
                "4.4.3.a: Past and present energy use"
            ]

        return framework_mapping

    # Private methods

    def _generate_rationale_id(
        self,
        calculation_type: CalculationType,
        inputs: Dict[str, Any]
    ) -> str:
        """Generate unique rationale ID."""
        id_data = {
            "type": calculation_type.value,
            "inputs": str(inputs),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uuid": str(uuid.uuid4())
        }
        id_str = json.dumps(id_data, sort_keys=True)
        return hashlib.sha256(id_str.encode()).hexdigest()[:16]

    def _get_methodology(self, calculation_type: CalculationType) -> List[str]:
        """Get methodology steps for calculation type."""
        return CALCULATION_METHODOLOGIES.get(calculation_type, [
            "Apply standard engineering calculation methodology",
            "Validate inputs against physical constraints",
            "Compute results using verified formulas",
            "Verify outputs are within expected ranges"
        ])

    def _get_principles(
        self,
        calculation_type: CalculationType
    ) -> List[ThermodynamicPrinciple]:
        """Get relevant thermodynamic principles."""
        if not self.config.include_principles:
            return []

        principle_map = {
            CalculationType.HEAT_TRANSFER: [
                "first_law_energy_balance",
                "heat_transfer_conduction",
                "heat_transfer_convection",
                "heat_transfer_radiation"
            ],
            CalculationType.COMBUSTION: [
                "first_law_energy_balance",
                "combustion_efficiency"
            ],
            CalculationType.STEAM_PROPERTIES: [
                "steam_enthalpy"
            ],
            CalculationType.EFFICIENCY_CALCULATION: [
                "first_law_energy_balance",
                "boiler_efficiency"
            ],
            CalculationType.EMISSION_CALCULATION: [
                "emission_factor"
            ],
            CalculationType.HEAT_EXCHANGER: [
                "heat_transfer_convection",
                "lmtd"
            ],
            CalculationType.PINCH_ANALYSIS: [
                "first_law_energy_balance",
                "pinch_temperature"
            ],
            CalculationType.INSULATION: [
                "heat_transfer_conduction",
                "heat_transfer_convection",
                "heat_transfer_radiation"
            ]
        }

        principle_keys = principle_map.get(calculation_type, [])
        principles = []

        for key in principle_keys:
            if key in THERMODYNAMIC_PRINCIPLES:
                principles.append(THERMODYNAMIC_PRINCIPLES[key])

        return principles

    def _get_citations(
        self,
        calculation_type: CalculationType,
        additional: Optional[List[StandardCitation]] = None
    ) -> List[StandardCitation]:
        """Get relevant standard citations."""
        if not self.config.include_citations:
            return []

        citation_keys = CALCULATION_CITATIONS.get(calculation_type, [])
        citations = []

        for key in citation_keys:
            if key in STANDARD_CITATIONS:
                citations.append(STANDARD_CITATIONS[key])

        if additional:
            citations.extend(additional)

        # Limit to max citations
        return citations[:self.config.max_citations]

    def _get_assumptions(
        self,
        calculation_type: CalculationType,
        custom: Optional[List[str]] = None
    ) -> List[str]:
        """Get assumptions for calculation type."""
        if not self.config.include_assumptions:
            return []

        base_assumptions = {
            CalculationType.HEAT_TRANSFER: [
                "Steady-state heat transfer conditions",
                "Material properties are temperature-independent over operating range",
                "No heat generation within the control volume"
            ],
            CalculationType.COMBUSTION: [
                "Complete combustion of fuel",
                "Fuel composition as specified or typical values used",
                "Air is at standard atmospheric conditions"
            ],
            CalculationType.STEAM_PROPERTIES: [
                "Steam is in thermodynamic equilibrium",
                "IAPWS-IF97 formulation applicable to operating conditions"
            ],
            CalculationType.EMISSION_CALCULATION: [
                "Emission factors are applicable to specific fuel and equipment",
                "Activity data is accurate and representative",
                "GWP values from latest IPCC assessment used"
            ],
            CalculationType.EFFICIENCY_CALCULATION: [
                "System boundary is clearly defined",
                "All energy flows are accounted for",
                "Measurement uncertainties are within acceptable limits"
            ],
            CalculationType.PINCH_ANALYSIS: [
                "Streams can exchange heat within defined constraints",
                "Minimum temperature approach is maintained",
                "No phase changes within streams unless specified"
            ]
        }

        assumptions = base_assumptions.get(calculation_type, [
            "Standard engineering assumptions apply",
            "Input data is representative of actual conditions"
        ])

        if custom:
            assumptions.extend(custom)

        return assumptions[:self.config.max_assumptions]

    def _get_limitations(
        self,
        calculation_type: CalculationType,
        custom: Optional[List[str]] = None
    ) -> List[str]:
        """Get limitations for calculation type."""
        if not self.config.include_limitations:
            return []

        base_limitations = {
            CalculationType.HEAT_TRANSFER: [
                "Accuracy depends on accurate material property data",
                "Transient effects not captured in steady-state analysis",
                "Surface conditions may vary from assumed values"
            ],
            CalculationType.COMBUSTION: [
                "Actual combustion may deviate from theoretical",
                "Emission factors have inherent uncertainty",
                "Equipment-specific variations not captured"
            ],
            CalculationType.STEAM_PROPERTIES: [
                "IAPWS-IF97 has defined validity ranges",
                "Near critical point accuracy may be reduced",
                "Impurities in actual steam not accounted for"
            ],
            CalculationType.EMISSION_CALCULATION: [
                "Emission factors are averages with uncertainty",
                "Activity data quality affects accuracy",
                "Scope 3 emissions may have higher uncertainty"
            ]
        }

        limitations = base_limitations.get(calculation_type, [
            "Results are estimates based on engineering calculations",
            "Actual performance may vary from calculated values"
        ])

        if custom:
            limitations.extend(custom)

        return limitations

    def _generate_summary(
        self,
        calculation_type: CalculationType,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> str:
        """Generate summary text for rationale."""
        summaries = {
            CalculationType.HEAT_TRANSFER: (
                f"Heat transfer calculation performed using standard engineering methods. "
                f"Results computed based on {len(inputs)} input parameters."
            ),
            CalculationType.COMBUSTION: (
                f"Combustion analysis performed per ASME PTC 4 methodology. "
                f"Efficiency and emissions calculated from fuel and operating data."
            ),
            CalculationType.STEAM_PROPERTIES: (
                f"Steam properties calculated using IAPWS-IF97 industrial formulation. "
                f"Thermodynamic state determined from provided conditions."
            ),
            CalculationType.EMISSION_CALCULATION: (
                f"Greenhouse gas emissions calculated per GHG Protocol methodology. "
                f"Emission factors from authoritative sources applied to activity data."
            ),
            CalculationType.EFFICIENCY_CALCULATION: (
                f"Energy efficiency calculated using input-output method. "
                f"Results per ISO 50001 energy management requirements."
            ),
            CalculationType.PINCH_ANALYSIS: (
                f"Pinch analysis performed to identify heat recovery opportunities. "
                f"Minimum utility requirements and pinch point determined."
            ),
            CalculationType.HEAT_EXCHANGER: (
                f"Heat exchanger analysis using LMTD method per ASME PTC 12.5. "
                f"Overall heat transfer and sizing calculated."
            )
        }

        return summaries.get(calculation_type, (
            f"Engineering calculation performed for {calculation_type.value}. "
            f"Results computed from {len(inputs)} inputs with {len(outputs)} outputs."
        ))

    def _determine_validation_status(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> str:
        """Determine validation status based on inputs/outputs."""
        # Basic validation checks
        if not inputs:
            return "FAIL"
        if not outputs:
            return "FAIL"

        # Check for None or NaN values in outputs
        for key, value in outputs.items():
            if value is None:
                return "FAIL"
            if isinstance(value, float) and (value != value):  # NaN check
                return "FAIL"

        return "PASS"


# Utility functions

def get_all_standard_sources() -> List[str]:
    """Get list of all available standard sources."""
    return [s.value for s in StandardSource]


def get_citations_by_source(source: StandardSource) -> List[StandardCitation]:
    """Get all citations from a specific source."""
    return [c for c in STANDARD_CITATIONS.values() if c.source == source]


def format_principle_as_markdown(principle: ThermodynamicPrinciple) -> str:
    """Format thermodynamic principle as Markdown."""
    lines = [
        f"### {principle.name}",
        "",
        f"**Formula:** `{principle.formula}`",
        "",
        principle.description,
        "",
        "**Variables:**"
    ]

    for var, desc in principle.variables.items():
        lines.append(f"- `{var}`: {desc}")

    if principle.citations:
        lines.append("")
        lines.append("**References:**")
        for citation in principle.citations:
            lines.append(f"- {citation.format_citation()}")

    return "\n".join(lines)


def format_rationale_as_markdown(rationale: EngineeringRationale) -> str:
    """Format engineering rationale as Markdown."""
    lines = [
        f"# Engineering Rationale: {rationale.calculation_type}",
        "",
        f"**ID:** {rationale.rationale_id}",
        f"**Timestamp:** {rationale.timestamp.isoformat()}",
        f"**Status:** {rationale.validation_status}",
        f"**Confidence:** {rationale.confidence_level:.0%}",
        "",
        "## Summary",
        rationale.summary,
        "",
        "## Methodology"
    ]

    for i, step in enumerate(rationale.methodology, 1):
        lines.append(f"{i}. {step}")

    lines.extend(["", "## Input Parameters"])
    for key, value in rationale.input_parameters.items():
        lines.append(f"- **{key}:** {value}")

    lines.extend(["", "## Output Results"])
    for key, value in rationale.output_results.items():
        lines.append(f"- **{key}:** {value}")

    if rationale.assumptions:
        lines.extend(["", "## Assumptions"])
        for assumption in rationale.assumptions:
            lines.append(f"- {assumption}")

    if rationale.limitations:
        lines.extend(["", "## Limitations"])
        for limitation in rationale.limitations:
            lines.append(f"- {limitation}")

    if rationale.citations:
        lines.extend(["", "## References"])
        for citation in rationale.citations:
            lines.append(f"- {citation.format_citation()}")

    lines.extend(["", f"**Provenance Hash:** `{rationale.provenance_hash}`"])

    return "\n".join(lines)
