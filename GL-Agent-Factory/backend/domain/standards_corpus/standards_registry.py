# -*- coding: utf-8 -*-
"""
Standards Registry for Process Heat Systems
============================================

Central registry of industry standards (ASME, API, NFPA, IEC, ISO)
with section-level indexing, cross-references, and equipment mappings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class StandardCategory(str, Enum):
    """Categories of standards."""
    PRESSURE_VESSEL = "pressure_vessel"
    PIPING = "piping"
    COMBUSTION_SAFETY = "combustion_safety"
    FIRED_EQUIPMENT = "fired_equipment"
    HEAT_EXCHANGER = "heat_exchanger"
    INSTRUMENTATION = "instrumentation"
    ELECTRICAL = "electrical"
    EMISSIONS = "emissions"
    SAFETY_SYSTEMS = "safety_systems"
    TESTING = "testing"
    MATERIALS = "materials"


class StandardsBody(str, Enum):
    """Standards organizations."""
    ASME = "ASME"
    API = "API"
    NFPA = "NFPA"
    IEC = "IEC"
    ISA = "ISA"
    ISO = "ISO"
    EPA = "EPA"
    EN = "EN"
    OSHA = "OSHA"


class ComplianceLevel(str, Enum):
    """Compliance requirement levels."""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    BEST_PRACTICE = "best_practice"


@dataclass
class StandardSection:
    """Section within a standard."""
    section_id: str
    title: str
    description: str
    parent_section: Optional[str] = None
    subsections: List[str] = field(default_factory=list)
    key_requirements: List[str] = field(default_factory=list)
    referenced_formulas: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)


@dataclass
class StandardFormula:
    """Formula or calculation method from a standard."""
    formula_id: str
    name: str
    description: str
    equation: str
    variables: Dict[str, str]  # variable: description
    units: Dict[str, str]  # variable: unit
    source_standard: str
    source_section: str
    application: str
    limitations: List[str] = field(default_factory=list)


@dataclass
class CrossReference:
    """Cross-reference between standards."""
    source_standard: str
    source_section: str
    target_standard: str
    target_section: str
    relationship: str  # "supersedes", "references", "complements", "conflicts"
    notes: str = ""


@dataclass
class EquipmentMapping:
    """Mapping of equipment to applicable standards."""
    equipment_type: str
    applicable_standards: List[str]
    mandatory_sections: List[str]
    design_standards: List[str]
    testing_standards: List[str]
    inspection_standards: List[str]


@dataclass
class ComplianceRequirement:
    """Specific compliance requirement."""
    requirement_id: str
    standard: str
    section: str
    description: str
    compliance_level: ComplianceLevel
    verification_method: str
    documentation_required: List[str] = field(default_factory=list)
    frequency: str = ""  # inspection/test frequency


@dataclass
class Standard:
    """
    Complete standard definition.

    Attributes:
        code: Standard code (e.g., "ASME BPVC Section I")
        title: Full title
        body: Standards organization
        category: Standard category
        description: Brief description
        current_edition: Current edition year
        sections: List of sections
        scope: Scope statement
        key_requirements: Key requirements summary
    """
    code: str
    title: str
    body: StandardsBody
    category: StandardCategory
    description: str
    current_edition: str
    scope: str = ""
    sections: List[StandardSection] = field(default_factory=list)
    key_requirements: List[str] = field(default_factory=list)
    formulas: List[StandardFormula] = field(default_factory=list)
    equipment_types: List[str] = field(default_factory=list)
    related_standards: List[str] = field(default_factory=list)
    supersedes: List[str] = field(default_factory=list)
    effective_date: Optional[date] = None


# =============================================================================
# ASME Standards
# =============================================================================

ASME_STANDARDS = {
    "ASME_BPVC_I": Standard(
        code="ASME BPVC Section I",
        title="Rules for Construction of Power Boilers",
        body=StandardsBody.ASME,
        category=StandardCategory.PRESSURE_VESSEL,
        description="Rules for construction of power boilers, electric boilers, and high temperature water boilers",
        current_edition="2023",
        scope="Power boilers and high pressure/temperature water boilers over 15 psig",
        sections=[
            StandardSection("PG", "General Requirements", "General requirements for all boilers",
                          key_requirements=["Materials", "Design", "Fabrication", "Inspection", "Testing"]),
            StandardSection("PW", "Requirements for Boilers Fabricated by Welding", "Welding requirements"),
            StandardSection("PFT", "Requirements for Firetube Boilers", "Firetube boiler specific rules"),
            StandardSection("PWT", "Requirements for Watertube Boilers", "Watertube boiler specific rules"),
            StandardSection("PMB", "Requirements for Miniature Boilers", "Small boiler requirements"),
            StandardSection("PEB", "Requirements for Electric Boilers", "Electric boiler requirements"),
            StandardSection("PVG", "Requirements for Organic Fluid Vaporizers", "Organic Rankine cycle"),
        ],
        key_requirements=[
            "Maximum Allowable Working Pressure (MAWP) calculation",
            "Material specifications per Section II",
            "Welding per Section IX",
            "NDE per Section V",
            "Pressure relief device sizing",
            "Hydrostatic testing",
        ],
        formulas=[
            StandardFormula(
                formula_id="ASME_I_PG27_1",
                name="Cylindrical Shell Thickness",
                description="Minimum required thickness for cylindrical shells under internal pressure",
                equation="t = (P × R) / (S × E - 0.6 × P)",
                variables={"t": "required thickness", "P": "design pressure", "R": "inside radius",
                          "S": "maximum allowable stress", "E": "joint efficiency"},
                units={"t": "in", "P": "psi", "R": "in", "S": "psi", "E": "dimensionless"},
                source_standard="ASME BPVC Section I",
                source_section="PG-27.2.2",
                application="Cylindrical shells and drums",
            ),
        ],
        equipment_types=["boiler", "power_boiler", "hrsg", "water_tube_boiler", "fire_tube_boiler"],
    ),

    "ASME_BPVC_VIII_1": Standard(
        code="ASME BPVC Section VIII Division 1",
        title="Rules for Construction of Pressure Vessels",
        body=StandardsBody.ASME,
        category=StandardCategory.PRESSURE_VESSEL,
        description="Rules for construction of pressure vessels",
        current_edition="2023",
        scope="Pressure vessels with design pressures exceeding 15 psig",
        key_requirements=[
            "Design by rule calculations",
            "Material certification",
            "Welding qualifications",
            "NDE requirements",
            "Pressure testing",
            "ASME stamp and U-1 data report",
        ],
        equipment_types=["pressure_vessel", "heat_exchanger", "deaerator", "flash_tank"],
    ),

    "ASME_B31_1": Standard(
        code="ASME B31.1",
        title="Power Piping",
        body=StandardsBody.ASME,
        category=StandardCategory.PIPING,
        description="Requirements for piping in electric generating stations and industrial plants",
        current_edition="2022",
        scope="Piping for steam, water, oil, gas, air for power generation",
        key_requirements=[
            "Pipe wall thickness calculation",
            "Flexibility analysis",
            "Support design",
            "Welding requirements",
            "Examination and testing",
        ],
        equipment_types=["steam_piping", "water_piping", "condensate_piping"],
    ),

    "ASME_PTC_4": Standard(
        code="ASME PTC 4",
        title="Fired Steam Generators",
        body=StandardsBody.ASME,
        category=StandardCategory.TESTING,
        description="Performance test code for fired steam generators",
        current_edition="2023",
        scope="Performance testing of fired steam generators",
        key_requirements=[
            "Input-output efficiency method",
            "Energy balance efficiency method",
            "Heat loss methods",
            "Uncertainty analysis",
            "Test procedure requirements",
        ],
        formulas=[
            StandardFormula(
                formula_id="ASME_PTC4_EFF",
                name="Boiler Efficiency (Input-Output)",
                description="Boiler efficiency by input-output method",
                equation="η = (Q_out / Q_in) × 100",
                variables={"η": "efficiency", "Q_out": "output energy", "Q_in": "input energy"},
                units={"η": "%", "Q_out": "Btu/hr", "Q_in": "Btu/hr"},
                source_standard="ASME PTC 4",
                source_section="Section 5",
                application="Boiler efficiency determination",
            ),
        ],
        equipment_types=["boiler", "steam_generator"],
    ),
}


# =============================================================================
# API Standards
# =============================================================================

API_STANDARDS = {
    "API_530": Standard(
        code="API 530",
        title="Calculation of Heater-Tube Thickness in Petroleum Refineries",
        body=StandardsBody.API,
        category=StandardCategory.FIRED_EQUIPMENT,
        description="Methods for calculating heater tube thickness considering creep",
        current_edition="2022",
        scope="Heater tubes in petroleum refinery service",
        key_requirements=[
            "Minimum tube thickness calculation",
            "Allowable stress at temperature",
            "Corrosion/erosion allowance",
            "Creep-rupture considerations",
            "Remaining life assessment",
        ],
        formulas=[
            StandardFormula(
                formula_id="API_530_THICKNESS",
                name="Tube Minimum Thickness",
                description="Minimum thickness for heater tubes",
                equation="t_min = (P × D_o) / (2 × σ + P) + CA",
                variables={"t_min": "minimum thickness", "P": "design pressure",
                          "D_o": "outside diameter", "σ": "allowable stress", "CA": "corrosion allowance"},
                units={"t_min": "in", "P": "psi", "D_o": "in", "σ": "psi", "CA": "in"},
                source_standard="API 530",
                source_section="Section 4",
                application="Heater tube design",
            ),
        ],
        equipment_types=["fired_heater", "process_furnace"],
    ),

    "API_560": Standard(
        code="API 560",
        title="Fired Heaters for General Refinery Service",
        body=StandardsBody.API,
        category=StandardCategory.FIRED_EQUIPMENT,
        description="Minimum requirements for fired heaters",
        current_edition="2022",
        scope="Fired heaters for petroleum and chemical industries",
        key_requirements=[
            "Heater design requirements",
            "Tube material selection",
            "Burner specifications",
            "Structural design",
            "Instrumentation requirements",
            "Safety interlocks",
        ],
        equipment_types=["fired_heater", "process_furnace", "reformer"],
    ),

    "API_579": Standard(
        code="API 579-1/ASME FFS-1",
        title="Fitness-For-Service",
        body=StandardsBody.API,
        category=StandardCategory.PRESSURE_VESSEL,
        description="Assessment procedures for equipment with flaws or damage",
        current_edition="2021",
        scope="Fitness-for-service assessments of pressure equipment",
        key_requirements=[
            "Level 1, 2, 3 assessment procedures",
            "Remaining life calculation",
            "General metal loss assessment",
            "Local metal loss assessment",
            "Crack-like flaw assessment",
            "Creep damage assessment",
        ],
        equipment_types=["pressure_vessel", "piping", "fired_heater", "boiler"],
    ),

    "API_580": Standard(
        code="API 580",
        title="Risk-Based Inspection",
        body=StandardsBody.API,
        category=StandardCategory.TESTING,
        description="Basis for risk-based inspection programs",
        current_edition="2016",
        scope="Risk-based inspection planning",
        key_requirements=[
            "Probability of failure assessment",
            "Consequence of failure assessment",
            "Risk ranking",
            "Inspection planning",
            "Risk mitigation strategies",
        ],
        equipment_types=["all"],
    ),

    "API_660": Standard(
        code="API 660",
        title="Shell-and-Tube Heat Exchangers",
        body=StandardsBody.API,
        category=StandardCategory.HEAT_EXCHANGER,
        description="Requirements for shell-and-tube heat exchangers",
        current_edition="2022",
        scope="Shell-and-tube heat exchangers for petroleum and chemical",
        key_requirements=[
            "Thermal design",
            "Mechanical design",
            "Material requirements",
            "Fabrication requirements",
            "Testing requirements",
        ],
        equipment_types=["shell_tube_hx", "heat_exchanger"],
    ),

    "API_661": Standard(
        code="API 661",
        title="Air-Cooled Heat Exchangers",
        body=StandardsBody.API,
        category=StandardCategory.HEAT_EXCHANGER,
        description="Requirements for air-cooled heat exchangers",
        current_edition="2021",
        scope="Air-cooled heat exchangers for petroleum and chemical",
        key_requirements=[
            "Thermal design",
            "Tube bundle design",
            "Header box design",
            "Fan selection",
            "Structural design",
        ],
        equipment_types=["air_cooled_hx"],
    ),
}


# =============================================================================
# NFPA Standards
# =============================================================================

NFPA_STANDARDS = {
    "NFPA_85": Standard(
        code="NFPA 85",
        title="Boiler and Combustion Systems Hazards Code",
        body=StandardsBody.NFPA,
        category=StandardCategory.COMBUSTION_SAFETY,
        description="Requirements for prevention of explosions in boilers",
        current_edition="2023",
        scope="Boilers with heat input > 12.5 MMBtu/hr",
        sections=[
            StandardSection("Ch4", "Single Burner Boilers", "Requirements for single burner boilers",
                          key_requirements=["Purge requirements", "Flame supervision", "Safety shutoff valves"]),
            StandardSection("Ch5", "Multiple Burner Boilers", "Multiple burner requirements"),
            StandardSection("Ch6", "Pulverized Fuel Systems", "Coal-fired boiler requirements"),
            StandardSection("Ch7", "Atmospheric Fluidized-Bed Boilers", "AFBC requirements"),
            StandardSection("Ch8", "Stokers", "Stoker-fired boiler requirements"),
            StandardSection("Ch9", "Heat Recovery Steam Generators", "HRSG specific requirements"),
        ],
        key_requirements=[
            "Pre-purge: 4 air changes minimum",
            "Post-purge requirements",
            "Flame failure response time",
            "Safety shutoff valve requirements",
            "Combustion air proving",
            "Fuel gas leak test",
            "Emergency shutdown procedures",
        ],
        formulas=[
            StandardFormula(
                formula_id="NFPA85_PURGE_TIME",
                name="Purge Time Calculation",
                description="Minimum purge time for 4 air changes",
                equation="t_purge = (4 × V_furnace) / Q_air",
                variables={"t_purge": "purge time", "V_furnace": "furnace volume", "Q_air": "air flow rate"},
                units={"t_purge": "min", "V_furnace": "ft³", "Q_air": "ft³/min"},
                source_standard="NFPA 85",
                source_section="4.6.2",
                application="Pre-ignition purge",
            ),
        ],
        equipment_types=["boiler", "hrsg", "duct_burner"],
    ),

    "NFPA_86": Standard(
        code="NFPA 86",
        title="Standard for Ovens and Furnaces",
        body=StandardsBody.NFPA,
        category=StandardCategory.COMBUSTION_SAFETY,
        description="Requirements for ovens and furnaces",
        current_edition="2023",
        scope="Class A, B, C, D ovens and furnaces",
        sections=[
            StandardSection("Ch8", "Class A Ovens", "Ovens < 750°F with flammable materials"),
            StandardSection("Ch9", "Class B Ovens", "Ovens < 750°F, no flammable materials"),
            StandardSection("Ch10", "Class C Furnaces", "Furnaces with special atmospheres"),
            StandardSection("Ch11", "Class D Furnaces", "Vacuum furnaces"),
        ],
        key_requirements=[
            "Safety equipment requirements",
            "Purge requirements",
            "Ventilation requirements",
            "Electrical classification",
            "Emergency shutdown",
            "Flame safeguard systems",
        ],
        equipment_types=["furnace", "oven", "dryer", "kiln"],
    ),

    "NFPA_87": Standard(
        code="NFPA 87",
        title="Recommended Practice for Fluid Heaters",
        body=StandardsBody.NFPA,
        category=StandardCategory.FIRED_EQUIPMENT,
        description="Guidelines for fluid heaters",
        current_edition="2021",
        scope="Fired heaters for heating process fluids",
        key_requirements=[
            "Design considerations",
            "Safety systems",
            "Operating procedures",
            "Emergency procedures",
        ],
        equipment_types=["fired_heater", "thermal_oil_heater"],
    ),
}


# =============================================================================
# IEC/ISA Standards
# =============================================================================

IEC_STANDARDS = {
    "IEC_61511": Standard(
        code="IEC 61511",
        title="Functional Safety - Safety Instrumented Systems for Process Industry",
        body=StandardsBody.IEC,
        category=StandardCategory.SAFETY_SYSTEMS,
        description="Requirements for safety instrumented systems",
        current_edition="2016",
        scope="SIS for process industry sector",
        sections=[
            StandardSection("Part1", "Framework, Definitions", "General requirements"),
            StandardSection("Part2", "Guidelines", "Application guidelines"),
            StandardSection("Part3", "Guidance", "Guidance for SIL determination"),
        ],
        key_requirements=[
            "SIL determination methods",
            "Safety requirements specification",
            "SIS design requirements",
            "Verification and validation",
            "Operation and maintenance",
            "Proof test requirements",
        ],
        equipment_types=["safety_instrumented_system", "interlock"],
    ),

    "ISA_84": Standard(
        code="ISA 84",
        title="Application of Safety Instrumented Systems for the Process Industries",
        body=StandardsBody.ISA,
        category=StandardCategory.SAFETY_SYSTEMS,
        description="US adoption of IEC 61511",
        current_edition="2016",
        scope="Safety instrumented systems in process industry",
        key_requirements=[
            "Complements IEC 61511",
            "US-specific guidance",
            "OSHA compliance path",
        ],
        equipment_types=["safety_instrumented_system"],
    ),

    "ISA_5_1": Standard(
        code="ISA 5.1",
        title="Instrumentation Symbols and Identification",
        body=StandardsBody.ISA,
        category=StandardCategory.INSTRUMENTATION,
        description="Standard symbols for P&IDs",
        current_edition="2022",
        scope="Instrumentation symbology",
        key_requirements=[
            "Tag number format",
            "Functional identification letters",
            "Symbol standards",
            "Line symbology",
        ],
        equipment_types=["instrument", "all"],
    ),
}


# =============================================================================
# Equipment to Standards Mappings
# =============================================================================

EQUIPMENT_MAPPINGS = {
    "boiler": EquipmentMapping(
        equipment_type="boiler",
        applicable_standards=["ASME_BPVC_I", "NFPA_85", "ASME_PTC_4", "API_579", "API_580"],
        mandatory_sections=["ASME_BPVC_I:PG", "NFPA_85:Ch4", "NFPA_85:Ch5"],
        design_standards=["ASME_BPVC_I"],
        testing_standards=["ASME_PTC_4"],
        inspection_standards=["API_579", "API_580"],
    ),
    "fired_heater": EquipmentMapping(
        equipment_type="fired_heater",
        applicable_standards=["API_560", "API_530", "NFPA_87", "API_579", "API_580"],
        mandatory_sections=["API_560:all", "NFPA_87:all"],
        design_standards=["API_560", "API_530"],
        testing_standards=["API_560"],
        inspection_standards=["API_579", "API_580"],
    ),
    "heat_exchanger": EquipmentMapping(
        equipment_type="heat_exchanger",
        applicable_standards=["API_660", "ASME_BPVC_VIII_1", "TEMA"],
        mandatory_sections=["API_660:all"],
        design_standards=["API_660", "TEMA"],
        testing_standards=["ASME_BPVC_VIII_1"],
        inspection_standards=["API_579"],
    ),
    "furnace": EquipmentMapping(
        equipment_type="furnace",
        applicable_standards=["NFPA_86", "API_560", "EN_746"],
        mandatory_sections=["NFPA_86:Ch8", "NFPA_86:Ch9", "NFPA_86:Ch10"],
        design_standards=["API_560", "EN_746"],
        testing_standards=["NFPA_86"],
        inspection_standards=["API_579"],
    ),
    "safety_instrumented_system": EquipmentMapping(
        equipment_type="safety_instrumented_system",
        applicable_standards=["IEC_61511", "ISA_84"],
        mandatory_sections=["IEC_61511:Part1", "IEC_61511:Part2"],
        design_standards=["IEC_61511"],
        testing_standards=["IEC_61511"],
        inspection_standards=["IEC_61511"],
    ),
}


# =============================================================================
# Standards Registry
# =============================================================================

class StandardsRegistry:
    """
    Central registry for industry standards.

    Provides:
    - Standard lookup by code
    - Equipment-to-standard mappings
    - Cross-reference lookup
    - Formula search
    - Compliance requirement tracking
    """

    def __init__(self):
        self.standards: Dict[str, Standard] = {}
        self.equipment_mappings: Dict[str, EquipmentMapping] = {}
        self.cross_references: List[CrossReference] = []
        self.compliance_requirements: Dict[str, ComplianceRequirement] = {}
        self._initialize_registry()

    def _initialize_registry(self):
        """Initialize with standard definitions."""
        # Load all standards
        self.standards.update(ASME_STANDARDS)
        self.standards.update(API_STANDARDS)
        self.standards.update(NFPA_STANDARDS)
        self.standards.update(IEC_STANDARDS)

        # Load equipment mappings
        self.equipment_mappings = EQUIPMENT_MAPPINGS.copy()

        # Create cross-references
        self._create_cross_references()

    def _create_cross_references(self):
        """Create standard cross-references."""
        self.cross_references = [
            CrossReference(
                source_standard="ASME_BPVC_I",
                source_section="PG",
                target_standard="NFPA_85",
                target_section="all",
                relationship="references",
                notes="NFPA 85 required for combustion safety",
            ),
            CrossReference(
                source_standard="API_560",
                source_section="all",
                target_standard="API_530",
                target_section="all",
                relationship="references",
                notes="API 530 for tube thickness calculations",
            ),
            CrossReference(
                source_standard="IEC_61511",
                source_section="all",
                target_standard="ISA_84",
                target_section="all",
                relationship="complements",
                notes="ISA 84 is US adoption with additional guidance",
            ),
        ]

    def get_standard(self, code: str) -> Optional[Standard]:
        """Get standard by code."""
        # Try exact match first
        if code in self.standards:
            return self.standards[code]
        # Try normalized lookup
        normalized = code.replace(" ", "_").replace("-", "_").upper()
        for key, std in self.standards.items():
            if key.upper() == normalized or std.code.replace(" ", "_").upper() == normalized:
                return std
        return None

    def get_standards_for_equipment(self, equipment_type: str) -> List[Standard]:
        """Get all standards applicable to an equipment type."""
        mapping = self.equipment_mappings.get(equipment_type)
        if mapping:
            return [
                self.standards[code] for code in mapping.applicable_standards
                if code in self.standards
            ]
        return []

    def get_mandatory_standards(self, equipment_type: str) -> List[Standard]:
        """Get mandatory standards for an equipment type."""
        mapping = self.equipment_mappings.get(equipment_type)
        if mapping:
            return [self.standards[code] for code in mapping.design_standards if code in self.standards]
        return []

    def get_standards_by_category(self, category: StandardCategory) -> List[Standard]:
        """Get standards by category."""
        return [s for s in self.standards.values() if s.category == category]

    def get_standards_by_body(self, body: StandardsBody) -> List[Standard]:
        """Get standards by issuing body."""
        return [s for s in self.standards.values() if s.body == body]

    def search_standards(self, query: str) -> List[Standard]:
        """Search standards by title or description."""
        query_lower = query.lower()
        return [
            s for s in self.standards.values()
            if query_lower in s.title.lower() or query_lower in s.description.lower()
        ]

    def get_formulas(self, standard_code: str = None) -> List[StandardFormula]:
        """Get formulas from standards."""
        if standard_code:
            std = self.get_standard(standard_code)
            return std.formulas if std else []
        # Return all formulas
        all_formulas = []
        for std in self.standards.values():
            all_formulas.extend(std.formulas)
        return all_formulas

    def get_cross_references(self, standard_code: str) -> List[CrossReference]:
        """Get cross-references for a standard."""
        return [
            xref for xref in self.cross_references
            if xref.source_standard == standard_code or xref.target_standard == standard_code
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_standards": len(self.standards),
            "asme_standards": len(self.get_standards_by_body(StandardsBody.ASME)),
            "api_standards": len(self.get_standards_by_body(StandardsBody.API)),
            "nfpa_standards": len(self.get_standards_by_body(StandardsBody.NFPA)),
            "iec_standards": len(self.get_standards_by_body(StandardsBody.IEC)),
            "equipment_mappings": len(self.equipment_mappings),
            "cross_references": len(self.cross_references),
            "total_formulas": len(self.get_formulas()),
        }


# Module-level singleton
_registry_instance: Optional[StandardsRegistry] = None

def get_standards_registry() -> StandardsRegistry:
    """Get or create the global standards registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = StandardsRegistry()
    return _registry_instance
