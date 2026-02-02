# -*- coding: utf-8 -*-
"""
Process Heat Domain Ontology - Core Implementation
==================================================

OWL/RDF-based semantic model for industrial process heat systems.
Supports reasoning, inference, and knowledge graph population.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field


# =============================================================================
# Ontology Namespaces
# =============================================================================

class OntologyNamespace(str, Enum):
    """Standard ontology namespaces for process heat domain."""

    # GreenLang namespaces
    GL = "https://greenlang.io/ontology#"
    GL_EQUIP = "https://greenlang.io/ontology/equipment#"
    GL_PROCESS = "https://greenlang.io/ontology/process#"
    GL_MEASURE = "https://greenlang.io/ontology/measurement#"
    GL_SAFETY = "https://greenlang.io/ontology/safety#"
    GL_EMISSION = "https://greenlang.io/ontology/emission#"

    # Standard ontologies
    RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    RDFS = "http://www.w3.org/2000/01/rdf-schema#"
    OWL = "http://www.w3.org/2002/07/owl#"
    XSD = "http://www.w3.org/2001/XMLSchema#"
    SKOS = "http://www.w3.org/2004/02/skos/core#"

    # Domain ontologies
    QUDT = "http://qudt.org/schema/qudt/"
    QUDT_UNIT = "http://qudt.org/vocab/unit/"
    SAREF = "https://saref.etsi.org/core/"
    SSN = "http://www.w3.org/ns/ssn/"
    SOSA = "http://www.w3.org/ns/sosa/"

    # Industrial ontologies
    ISO_15926 = "http://standards.iso.org/iso/15926/"
    IEC_CDD = "http://cdd.iec.ch/cdd/iec61360/"


# =============================================================================
# Ontology Class Definitions
# =============================================================================

class OntologyClassType(str, Enum):
    """Types of ontology classes."""
    EQUIPMENT = "equipment"
    PROCESS = "process"
    MEASUREMENT = "measurement"
    SAFETY = "safety"
    EMISSION = "emission"
    MATERIAL = "material"
    LOCATION = "location"
    ORGANIZATION = "organization"
    DOCUMENT = "document"
    REGULATION = "regulation"


@dataclass
class OntologyClass:
    """
    Represents an OWL class in the process heat ontology.

    Attributes:
        uri: Unique identifier for the class
        label: Human-readable label
        definition: Class definition
        parent_classes: List of parent class URIs
        equivalent_classes: Equivalent class expressions
        disjoint_classes: Disjoint class URIs
        properties: Associated property restrictions
    """
    uri: str
    label: str
    definition: str
    class_type: OntologyClassType
    parent_classes: List[str] = field(default_factory=list)
    equivalent_classes: List[str] = field(default_factory=list)
    disjoint_classes: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def get_full_uri(self) -> str:
        """Get fully qualified URI."""
        if self.uri.startswith("http"):
            return self.uri
        return f"{OntologyNamespace.GL.value}{self.uri}"

    def to_owl_xml(self) -> str:
        """Generate OWL/XML representation."""
        full_uri = self.get_full_uri()
        xml_parts = [f'<owl:Class rdf:about="{full_uri}">']
        xml_parts.append(f'  <rdfs:label>{self.label}</rdfs:label>')
        xml_parts.append(f'  <rdfs:comment>{self.definition}</rdfs:comment>')

        for parent in self.parent_classes:
            xml_parts.append(f'  <rdfs:subClassOf rdf:resource="{parent}"/>')

        xml_parts.append('</owl:Class>')
        return '\n'.join(xml_parts)


@dataclass
class OntologyProperty:
    """
    Represents an OWL property (object or datatype).

    Attributes:
        uri: Property URI
        label: Human-readable label
        definition: Property definition
        domain: Domain class URIs
        range: Range class/datatype URIs
        is_functional: Whether property is functional
        is_inverse_functional: Whether property is inverse functional
    """
    uri: str
    label: str
    definition: str
    property_type: str  # "object" or "datatype"
    domain: List[str] = field(default_factory=list)
    range: List[str] = field(default_factory=list)
    is_functional: bool = False
    is_inverse_functional: bool = False
    inverse_of: Optional[str] = None
    parent_properties: List[str] = field(default_factory=list)

    def get_full_uri(self) -> str:
        """Get fully qualified URI."""
        if self.uri.startswith("http"):
            return self.uri
        return f"{OntologyNamespace.GL.value}{self.uri}"


@dataclass
class OntologyIndividual:
    """
    Represents an OWL individual (instance).

    Attributes:
        uri: Individual URI
        label: Human-readable label
        class_types: Class URIs this individual belongs to
        properties: Property values
    """
    uri: str
    label: str
    class_types: List[str]
    properties: Dict[str, Any] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def get_full_uri(self) -> str:
        """Get fully qualified URI."""
        if self.uri.startswith("http"):
            return self.uri
        return f"{OntologyNamespace.GL.value}{self.uri}"


# =============================================================================
# Process Heat Equipment Taxonomy
# =============================================================================

# Equipment class hierarchy
EQUIPMENT_HIERARCHY = {
    "ProcessHeatEquipment": {
        "definition": "Equipment used in industrial process heat applications",
        "children": {
            "Boiler": {
                "definition": "Vessel for generating steam or heating water",
                "children": {
                    "FireTubeBoiler": {"definition": "Boiler with combustion gases inside tubes"},
                    "WaterTubeBoiler": {"definition": "Boiler with water inside tubes"},
                    "ElectricBoiler": {"definition": "Boiler using electrical resistance heating"},
                    "WasteHeatBoiler": {"definition": "Boiler recovering heat from process streams"},
                    "HRSGBoiler": {"definition": "Heat Recovery Steam Generator"},
                }
            },
            "Furnace": {
                "definition": "Enclosed structure for high-temperature heating",
                "children": {
                    "ProcessFurnace": {"definition": "Furnace for process heating applications"},
                    "ReheatFurnace": {"definition": "Furnace for reheating steel products"},
                    "AnnealingFurnace": {"definition": "Furnace for heat treatment"},
                    "MeltingFurnace": {"definition": "Furnace for melting metals"},
                    "CremationFurnace": {"definition": "Furnace for high-temperature oxidation"},
                }
            },
            "HeatExchanger": {
                "definition": "Device for heat transfer between fluids",
                "children": {
                    "ShellAndTubeHX": {"definition": "Heat exchanger with shell and tube bundle"},
                    "PlateHX": {"definition": "Heat exchanger with corrugated plates"},
                    "AirCooledHX": {"definition": "Heat exchanger with air cooling"},
                    "DoubleP lateHX": {"definition": "Heat exchanger with double-walled plates"},
                    "SpiralHX": {"definition": "Heat exchanger with spiral flow paths"},
                    "RegenerativeHX": {"definition": "Heat exchanger with regenerative heat storage"},
                }
            },
            "Heater": {
                "definition": "Equipment for adding heat to process streams",
                "children": {
                    "FiredHeater": {"definition": "Heater using combustion"},
                    "ElectricHeater": {"definition": "Heater using electrical resistance"},
                    "InductionHeater": {"definition": "Heater using electromagnetic induction"},
                    "MicrowaveHeater": {"definition": "Heater using microwave radiation"},
                    "InfraredHeater": {"definition": "Heater using infrared radiation"},
                }
            },
            "Dryer": {
                "definition": "Equipment for removing moisture",
                "children": {
                    "RotaryDryer": {"definition": "Dryer with rotating drum"},
                    "FluidBedDryer": {"definition": "Dryer with fluidized bed"},
                    "SprayDryer": {"definition": "Dryer using spray atomization"},
                    "ConveyorDryer": {"definition": "Dryer with conveyor belt"},
                    "FlashDryer": {"definition": "Dryer with pneumatic conveying"},
                }
            },
            "Kiln": {
                "definition": "Furnace for processing materials at high temperature",
                "children": {
                    "RotaryKiln": {"definition": "Kiln with rotating cylinder"},
                    "ShaftKiln": {"definition": "Kiln with vertical shaft"},
                    "TunnelKiln": {"definition": "Kiln with continuous tunnel"},
                    "RollerKiln": {"definition": "Kiln with roller conveyor"},
                }
            },
            "SteamTrap": {
                "definition": "Device for discharging condensate while retaining steam",
                "children": {
                    "MechanicalTrap": {"definition": "Steam trap using mechanical operation"},
                    "ThermodynamicTrap": {"definition": "Steam trap using thermodynamic principles"},
                    "ThermostaticTrap": {"definition": "Steam trap using temperature sensing"},
                }
            },
            "Condenser": {
                "definition": "Heat exchanger for condensing vapor",
                "children": {
                    "SurfaceCondenser": {"definition": "Condenser with surface heat transfer"},
                    "JetCondenser": {"definition": "Condenser with direct contact"},
                    "AirCooledCondenser": {"definition": "Condenser with air cooling"},
                }
            },
            "Economizer": {
                "definition": "Heat recovery device for preheating feedwater",
                "children": {
                    "CastIronEconomizer": {"definition": "Economizer with cast iron construction"},
                    "SteelEconomizer": {"definition": "Economizer with steel construction"},
                    "CondensingEconomizer": {"definition": "Economizer recovering latent heat"},
                }
            },
            "AirPreheater": {
                "definition": "Device for preheating combustion air",
                "children": {
                    "RecuperativeAPH": {"definition": "Air preheater with recuperative heat transfer"},
                    "RegenerativeAPH": {"definition": "Air preheater with regenerative heat storage"},
                    "HeatPipeAPH": {"definition": "Air preheater using heat pipes"},
                }
            },
        }
    }
}


# =============================================================================
# Physical Quantities and Units
# =============================================================================

PHYSICAL_QUANTITIES = {
    "Temperature": {
        "definition": "Measure of thermal energy",
        "si_unit": "kelvin",
        "common_units": ["celsius", "fahrenheit", "rankine"],
        "symbol": "T",
        "qudt_uri": "http://qudt.org/vocab/quantitykind/Temperature"
    },
    "Pressure": {
        "definition": "Force per unit area",
        "si_unit": "pascal",
        "common_units": ["bar", "psi", "atm", "mmHg"],
        "symbol": "P",
        "qudt_uri": "http://qudt.org/vocab/quantitykind/Pressure"
    },
    "MassFlowRate": {
        "definition": "Mass per unit time",
        "si_unit": "kilogram_per_second",
        "common_units": ["kg_per_hour", "tonne_per_hour", "lb_per_hour"],
        "symbol": "m_dot",
        "qudt_uri": "http://qudt.org/vocab/quantitykind/MassFlowRate"
    },
    "HeatFlowRate": {
        "definition": "Thermal energy per unit time",
        "si_unit": "watt",
        "common_units": ["kilowatt", "megawatt", "btu_per_hour", "mmbtu_per_hour"],
        "symbol": "Q_dot",
        "qudt_uri": "http://qudt.org/vocab/quantitykind/HeatFlowRate"
    },
    "ThermalEfficiency": {
        "definition": "Ratio of useful heat output to heat input",
        "si_unit": "dimensionless",
        "common_units": ["percent"],
        "symbol": "eta",
        "qudt_uri": "http://qudt.org/vocab/quantitykind/Efficiency"
    },
    "SpecificEnthalpy": {
        "definition": "Enthalpy per unit mass",
        "si_unit": "joule_per_kilogram",
        "common_units": ["kJ_per_kg", "btu_per_lb"],
        "symbol": "h",
        "qudt_uri": "http://qudt.org/vocab/quantitykind/SpecificEnthalpy"
    },
    "HeatTransferCoefficient": {
        "definition": "Heat flux per unit temperature difference",
        "si_unit": "watt_per_square_meter_kelvin",
        "common_units": ["btu_per_hour_ft2_F"],
        "symbol": "U",
        "qudt_uri": "http://qudt.org/vocab/quantitykind/HeatTransferCoefficient"
    },
    "ThermalConductivity": {
        "definition": "Ability to conduct heat",
        "si_unit": "watt_per_meter_kelvin",
        "common_units": ["btu_per_hour_ft_F"],
        "symbol": "k",
        "qudt_uri": "http://qudt.org/vocab/quantitykind/ThermalConductivity"
    },
    "Emissivity": {
        "definition": "Ratio of emitted radiation to blackbody radiation",
        "si_unit": "dimensionless",
        "common_units": [],
        "symbol": "epsilon",
        "qudt_uri": "http://qudt.org/vocab/quantitykind/Emissivity"
    },
    "ExcessAir": {
        "definition": "Air supplied above stoichiometric requirement",
        "si_unit": "percent",
        "common_units": [],
        "symbol": "EA",
        "qudt_uri": None
    },
    "StackTemperature": {
        "definition": "Temperature of flue gas at stack exit",
        "si_unit": "kelvin",
        "common_units": ["celsius", "fahrenheit"],
        "symbol": "T_stack",
        "qudt_uri": None
    },
    "CO2Concentration": {
        "definition": "Carbon dioxide concentration in flue gas",
        "si_unit": "percent",
        "common_units": ["ppm"],
        "symbol": "CO2",
        "qudt_uri": None
    },
    "NOxConcentration": {
        "definition": "Nitrogen oxides concentration",
        "si_unit": "ppm",
        "common_units": ["mg_per_Nm3"],
        "symbol": "NOx",
        "qudt_uri": None
    },
}


# =============================================================================
# Process Heat Ontology Manager
# =============================================================================

class ProcessHeatOntology:
    """
    Main manager for the Process Heat Domain Ontology.

    Provides methods for:
    - Loading and saving ontology
    - Creating and querying classes, properties, individuals
    - SPARQL query execution
    - Knowledge graph population
    - Reasoning and inference
    """

    def __init__(self, base_uri: str = None):
        """Initialize the ontology manager."""
        self.base_uri = base_uri or OntologyNamespace.GL.value
        self.classes: Dict[str, OntologyClass] = {}
        self.properties: Dict[str, OntologyProperty] = {}
        self.individuals: Dict[str, OntologyIndividual] = {}
        self._initialize_core_ontology()

    def _initialize_core_ontology(self):
        """Initialize core ontology classes and properties."""
        # Create equipment hierarchy
        self._create_equipment_classes()

        # Create measurement classes
        self._create_measurement_classes()

        # Create core properties
        self._create_core_properties()

    def _create_equipment_classes(self):
        """Create equipment class hierarchy from taxonomy."""
        def create_class_recursive(name: str, data: dict, parent: str = None):
            class_uri = f"{OntologyNamespace.GL_EQUIP.value}{name}"
            parent_uris = [f"{OntologyNamespace.GL_EQUIP.value}{parent}"] if parent else []

            ontology_class = OntologyClass(
                uri=class_uri,
                label=name,
                definition=data.get("definition", ""),
                class_type=OntologyClassType.EQUIPMENT,
                parent_classes=parent_uris,
            )
            self.classes[class_uri] = ontology_class

            if "children" in data:
                for child_name, child_data in data["children"].items():
                    create_class_recursive(child_name, child_data, name)

        for root_name, root_data in EQUIPMENT_HIERARCHY.items():
            create_class_recursive(root_name, root_data)

    def _create_measurement_classes(self):
        """Create measurement/quantity classes from definitions."""
        for quantity_name, quantity_data in PHYSICAL_QUANTITIES.items():
            class_uri = f"{OntologyNamespace.GL_MEASURE.value}{quantity_name}"

            annotations = {
                "symbol": quantity_data.get("symbol", ""),
                "si_unit": quantity_data.get("si_unit", ""),
            }
            if quantity_data.get("qudt_uri"):
                annotations["qudt_equivalent"] = quantity_data["qudt_uri"]

            ontology_class = OntologyClass(
                uri=class_uri,
                label=quantity_name,
                definition=quantity_data.get("definition", ""),
                class_type=OntologyClassType.MEASUREMENT,
                annotations=annotations,
            )
            self.classes[class_uri] = ontology_class

    def _create_core_properties(self):
        """Create core ontology properties."""
        core_properties = [
            # Equipment relationships
            OntologyProperty(
                uri=f"{OntologyNamespace.GL.value}hasComponent",
                label="has component",
                definition="Equipment has a component",
                property_type="object",
                domain=[f"{OntologyNamespace.GL_EQUIP.value}ProcessHeatEquipment"],
                range=[f"{OntologyNamespace.GL_EQUIP.value}ProcessHeatEquipment"],
            ),
            OntologyProperty(
                uri=f"{OntologyNamespace.GL.value}connectedTo",
                label="connected to",
                definition="Equipment is physically connected to another equipment",
                property_type="object",
                domain=[f"{OntologyNamespace.GL_EQUIP.value}ProcessHeatEquipment"],
                range=[f"{OntologyNamespace.GL_EQUIP.value}ProcessHeatEquipment"],
            ),
            OntologyProperty(
                uri=f"{OntologyNamespace.GL.value}feedsInto",
                label="feeds into",
                definition="Process stream feeds into equipment",
                property_type="object",
            ),
            # Measurement properties
            OntologyProperty(
                uri=f"{OntologyNamespace.GL.value}hasValue",
                label="has value",
                definition="Measurement has a numeric value",
                property_type="datatype",
                range=[f"{OntologyNamespace.XSD.value}double"],
                is_functional=True,
            ),
            OntologyProperty(
                uri=f"{OntologyNamespace.GL.value}hasUnit",
                label="has unit",
                definition="Measurement has a unit of measure",
                property_type="object",
                range=[f"{OntologyNamespace.QUDT_UNIT.value}Unit"],
            ),
            OntologyProperty(
                uri=f"{OntologyNamespace.GL.value}measuredAt",
                label="measured at",
                definition="Timestamp of measurement",
                property_type="datatype",
                range=[f"{OntologyNamespace.XSD.value}dateTime"],
            ),
            # Safety properties
            OntologyProperty(
                uri=f"{OntologyNamespace.GL.value}hasSILLevel",
                label="has SIL level",
                definition="Safety Integrity Level",
                property_type="datatype",
                range=[f"{OntologyNamespace.XSD.value}integer"],
            ),
            OntologyProperty(
                uri=f"{OntologyNamespace.GL.value}protectedBy",
                label="protected by",
                definition="Equipment is protected by safety device",
                property_type="object",
            ),
        ]

        for prop in core_properties:
            self.properties[prop.uri] = prop

    def add_class(self, ontology_class: OntologyClass) -> str:
        """Add a class to the ontology."""
        self.classes[ontology_class.uri] = ontology_class
        return ontology_class.uri

    def add_property(self, ontology_property: OntologyProperty) -> str:
        """Add a property to the ontology."""
        self.properties[ontology_property.uri] = ontology_property
        return ontology_property.uri

    def add_individual(self, individual: OntologyIndividual) -> str:
        """Add an individual to the ontology."""
        self.individuals[individual.uri] = individual
        return individual.uri

    def get_class(self, uri: str) -> Optional[OntologyClass]:
        """Get a class by URI."""
        return self.classes.get(uri)

    def get_property(self, uri: str) -> Optional[OntologyProperty]:
        """Get a property by URI."""
        return self.properties.get(uri)

    def get_individual(self, uri: str) -> Optional[OntologyIndividual]:
        """Get an individual by URI."""
        return self.individuals.get(uri)

    def get_subclasses(self, class_uri: str) -> List[OntologyClass]:
        """Get all subclasses of a class."""
        subclasses = []
        for cls in self.classes.values():
            if class_uri in cls.parent_classes:
                subclasses.append(cls)
        return subclasses

    def get_equipment_by_type(self, equipment_type: str) -> List[OntologyIndividual]:
        """Get all equipment individuals of a specific type."""
        type_uri = f"{OntologyNamespace.GL_EQUIP.value}{equipment_type}"
        return [
            ind for ind in self.individuals.values()
            if type_uri in ind.class_types
        ]

    def export_owl_xml(self, filepath: Union[str, Path]) -> None:
        """Export ontology to OWL/XML format."""
        owl_content = self._generate_owl_xml()
        Path(filepath).write_text(owl_content, encoding="utf-8")

    def _generate_owl_xml(self) -> str:
        """Generate OWL/XML representation of the ontology."""
        lines = [
            '<?xml version="1.0"?>',
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
            '         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"',
            '         xmlns:owl="http://www.w3.org/2002/07/owl#"',
            '         xmlns:xsd="http://www.w3.org/2001/XMLSchema#"',
            f'         xmlns:gl="{OntologyNamespace.GL.value}">',
            '',
            f'  <owl:Ontology rdf:about="{self.base_uri}">',
            '    <rdfs:label>GreenLang Process Heat Ontology</rdfs:label>',
            '    <owl:versionInfo>1.0.0</owl:versionInfo>',
            '  </owl:Ontology>',
            '',
        ]

        # Add classes
        for cls in self.classes.values():
            lines.append(cls.to_owl_xml())
            lines.append('')

        lines.append('</rdf:RDF>')
        return '\n'.join(lines)

    def get_statistics(self) -> Dict[str, int]:
        """Get ontology statistics."""
        return {
            "total_classes": len(self.classes),
            "total_properties": len(self.properties),
            "total_individuals": len(self.individuals),
            "equipment_classes": len([c for c in self.classes.values()
                                      if c.class_type == OntologyClassType.EQUIPMENT]),
            "measurement_classes": len([c for c in self.classes.values()
                                        if c.class_type == OntologyClassType.MEASUREMENT]),
        }


# Module-level singleton
_ontology_instance: Optional[ProcessHeatOntology] = None

def get_ontology() -> ProcessHeatOntology:
    """Get or create the global ontology instance."""
    global _ontology_instance
    if _ontology_instance is None:
        _ontology_instance = ProcessHeatOntology()
    return _ontology_instance
