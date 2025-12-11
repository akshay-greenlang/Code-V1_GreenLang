# -*- coding: utf-8 -*-
"""
GreenLang Process Heat Domain Ontology
======================================

OWL/RDF-based ontology for industrial process heat systems.
Provides semantic modeling for equipment, processes, measurements,
and regulatory compliance in industrial decarbonization.

Key Classes:
- ProcessHeatOntology: Main ontology manager
- EquipmentClass: Equipment type hierarchy
- MeasurementClass: Physical measurements and units
- ProcessClass: Industrial processes and operations
- SafetyClass: Safety interlocks and compliance
"""

from .process_heat_ontology import (
    ProcessHeatOntology,
    OntologyNamespace,
    OntologyClass,
    OntologyProperty,
    OntologyIndividual,
)
from .equipment_taxonomy import (
    EquipmentTaxonomy,
    EquipmentClass,
    EquipmentRelation,
    EquipmentAttribute,
)
from .measurement_ontology import (
    MeasurementOntology,
    PhysicalQuantity,
    UnitOfMeasure,
    MeasurementScale,
)
from .process_ontology import (
    ProcessOntology,
    ThermalProcess,
    CombustionProcess,
    HeatTransferProcess,
)
from .safety_ontology import (
    SafetyOntology,
    SafetyInterlock,
    ProtectionLayer,
    HazardClass,
)

__all__ = [
    # Core ontology
    "ProcessHeatOntology",
    "OntologyNamespace",
    "OntologyClass",
    "OntologyProperty",
    "OntologyIndividual",
    # Equipment
    "EquipmentTaxonomy",
    "EquipmentClass",
    "EquipmentRelation",
    "EquipmentAttribute",
    # Measurements
    "MeasurementOntology",
    "PhysicalQuantity",
    "UnitOfMeasure",
    "MeasurementScale",
    # Processes
    "ProcessOntology",
    "ThermalProcess",
    "CombustionProcess",
    "HeatTransferProcess",
    # Safety
    "SafetyOntology",
    "SafetyInterlock",
    "ProtectionLayer",
    "HazardClass",
]

__version__ = "1.0.0"
