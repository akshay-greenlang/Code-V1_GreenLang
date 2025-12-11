# -*- coding: utf-8 -*-
"""
Safety Ontology for Process Heat Systems
========================================

Safety classifications, interlocks, protection layers,
and hazard definitions for industrial process heat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SILLevel(int, Enum):
    """Safety Integrity Level per IEC 61511."""
    SIL_NONE = 0
    SIL_1 = 1
    SIL_2 = 2
    SIL_3 = 3
    SIL_4 = 4


class HazardSeverity(str, Enum):
    """Hazard severity classification."""
    CATASTROPHIC = "catastrophic"
    CRITICAL = "critical"
    MARGINAL = "marginal"
    NEGLIGIBLE = "negligible"


class HazardLikelihood(str, Enum):
    """Hazard likelihood classification."""
    FREQUENT = "frequent"
    PROBABLE = "probable"
    OCCASIONAL = "occasional"
    REMOTE = "remote"
    IMPROBABLE = "improbable"


class ProtectionLayerType(str, Enum):
    """Types of protection layers."""
    BPCS = "basic_process_control_system"
    ALARM = "alarm_and_operator_response"
    SIS = "safety_instrumented_system"
    PSV = "pressure_safety_valve"
    RUPTURE_DISK = "rupture_disk"
    PHYSICAL_BARRIER = "physical_barrier"
    EMERGENCY_RESPONSE = "emergency_response"


class InterlockType(str, Enum):
    """Types of safety interlocks."""
    HIGH_TEMPERATURE = "high_temperature"
    LOW_TEMPERATURE = "low_temperature"
    HIGH_PRESSURE = "high_pressure"
    LOW_PRESSURE = "low_pressure"
    HIGH_LEVEL = "high_level"
    LOW_LEVEL = "low_level"
    HIGH_FLOW = "high_flow"
    LOW_FLOW = "low_flow"
    FLAME_FAILURE = "flame_failure"
    COMBUSTION_AIR = "combustion_air"
    PURGE = "purge"
    LOSS_OF_FLAME = "loss_of_flame"


@dataclass
class HazardClass:
    """
    Hazard classification definition.

    Attributes:
        id: Hazard identifier
        name: Hazard name
        description: Detailed description
        severity: Hazard severity
        likelihood: Hazard likelihood without safeguards
        consequences: Potential consequences
        applicable_equipment: Equipment types where hazard applies
        mitigation_measures: Standard mitigation measures
    """
    id: str
    name: str
    description: str
    severity: HazardSeverity
    likelihood: HazardLikelihood
    consequences: List[str] = field(default_factory=list)
    applicable_equipment: List[str] = field(default_factory=list)
    mitigation_measures: List[str] = field(default_factory=list)
    regulatory_references: List[str] = field(default_factory=list)


@dataclass
class SafetyInterlock:
    """
    Safety interlock definition.

    Attributes:
        id: Interlock identifier
        name: Interlock name
        interlock_type: Type of interlock
        trigger_condition: Condition that triggers interlock
        action: Action taken when triggered
        sil_level: Required SIL level
        response_time_s: Required response time
        test_frequency: Required proof test frequency
    """
    id: str
    name: str
    interlock_type: InterlockType
    trigger_condition: str
    action: str
    sil_level: SILLevel = SILLevel.SIL_NONE
    response_time_s: float = 1.0
    test_frequency: str = "annually"
    applicable_standards: List[str] = field(default_factory=list)
    voting_logic: str = "1oo1"  # 1 out of 1, 2oo3, etc.


@dataclass
class ProtectionLayer:
    """
    Independent Protection Layer (IPL) definition.

    Attributes:
        id: Layer identifier
        name: Layer name
        layer_type: Type of protection layer
        pfd: Probability of Failure on Demand
        independence: Is layer independent
        specificity: Does layer protect against specific hazard
        auditability: Can layer effectiveness be verified
    """
    id: str
    name: str
    layer_type: ProtectionLayerType
    pfd: float  # Probability of Failure on Demand
    independence: bool = True
    specificity: bool = True
    auditability: bool = True
    description: str = ""
    response_time_s: Optional[float] = None


# =============================================================================
# Standard Hazard Definitions
# =============================================================================

PROCESS_HEAT_HAZARDS = {
    "boiler_explosion": HazardClass(
        id="boiler_explosion",
        name="Boiler Explosion",
        description="Catastrophic failure of boiler pressure vessel",
        severity=HazardSeverity.CATASTROPHIC,
        likelihood=HazardLikelihood.REMOTE,
        consequences=[
            "Multiple fatalities",
            "Major property damage",
            "Extended shutdown",
            "Environmental release",
        ],
        applicable_equipment=["boiler", "fire_tube_boiler", "water_tube_boiler"],
        mitigation_measures=[
            "Pressure relief devices",
            "Low water cutoff",
            "Regular inspection",
            "Proper water treatment",
        ],
        regulatory_references=["ASME BPVC Section I", "NFPA 85"],
    ),
    "furnace_explosion": HazardClass(
        id="furnace_explosion",
        name="Furnace Explosion",
        description="Explosion due to fuel accumulation in furnace enclosure",
        severity=HazardSeverity.CATASTROPHIC,
        likelihood=HazardLikelihood.OCCASIONAL,
        consequences=[
            "Fatalities",
            "Major equipment damage",
            "Fire propagation",
            "Production loss",
        ],
        applicable_equipment=["furnace", "fired_heater", "process_furnace"],
        mitigation_measures=[
            "Pre-purge before ignition",
            "Flame monitoring",
            "Fuel shutoff valves",
            "Combustion safeguards",
        ],
        regulatory_references=["NFPA 86", "NFPA 87", "API 556"],
    ),
    "tube_rupture": HazardClass(
        id="tube_rupture",
        name="Heater Tube Rupture",
        description="Failure of process tube in fired heater",
        severity=HazardSeverity.CRITICAL,
        likelihood=HazardLikelihood.OCCASIONAL,
        consequences=[
            "Fire",
            "Process release",
            "Equipment damage",
            "Production loss",
        ],
        applicable_equipment=["fired_heater", "process_furnace"],
        mitigation_measures=[
            "Tube wall thickness monitoring",
            "Operating within design limits",
            "Emergency depressuring",
            "Fire suppression",
        ],
        regulatory_references=["API 530", "API 579", "API 580"],
    ),
    "steam_release": HazardClass(
        id="steam_release",
        name="Uncontrolled Steam Release",
        description="Release of high-pressure steam causing burns",
        severity=HazardSeverity.CRITICAL,
        likelihood=HazardLikelihood.PROBABLE,
        consequences=[
            "Severe burns",
            "Fatality",
            "Equipment damage",
        ],
        applicable_equipment=["boiler", "steam_trap", "prv", "deaerator"],
        mitigation_measures=[
            "Proper valve operation",
            "Steam leak detection",
            "Thermal insulation",
            "PPE requirements",
        ],
        regulatory_references=["OSHA 1910.219", "ASME B31.1"],
    ),
    "thermal_runaway": HazardClass(
        id="thermal_runaway",
        name="Thermal Runaway",
        description="Uncontrolled temperature increase in exothermic process",
        severity=HazardSeverity.CRITICAL,
        likelihood=HazardLikelihood.OCCASIONAL,
        consequences=[
            "Fire",
            "Explosion",
            "Toxic release",
            "Equipment damage",
        ],
        applicable_equipment=["reactor", "furnace", "dryer"],
        mitigation_measures=[
            "Temperature monitoring",
            "Emergency cooling",
            "Reaction quench system",
            "Process interlocks",
        ],
        regulatory_references=["IEC 61511", "ISA 84"],
    ),
    "low_water": HazardClass(
        id="low_water",
        name="Low Water Condition",
        description="Insufficient water level in boiler causing overheating",
        severity=HazardSeverity.CATASTROPHIC,
        likelihood=HazardLikelihood.OCCASIONAL,
        consequences=[
            "Boiler damage",
            "Tube failure",
            "Potential explosion",
        ],
        applicable_equipment=["boiler", "fire_tube_boiler", "water_tube_boiler"],
        mitigation_measures=[
            "Low water cutoff",
            "Water level monitoring",
            "Automatic feedwater control",
            "Alarm systems",
        ],
        regulatory_references=["ASME CSD-1", "NFPA 85"],
    ),
}


# =============================================================================
# Standard Safety Interlocks
# =============================================================================

STANDARD_INTERLOCKS = {
    "llc_boiler": SafetyInterlock(
        id="llc_boiler",
        name="Low-Low Water Cutoff",
        interlock_type=InterlockType.LOW_LEVEL,
        trigger_condition="Drum level < LLLL setpoint",
        action="Trip fuel, close main steam valve",
        sil_level=SILLevel.SIL_2,
        response_time_s=2.0,
        test_frequency="weekly",
        applicable_standards=["ASME CSD-1", "NFPA 85"],
        voting_logic="1oo2",
    ),
    "hhp_boiler": SafetyInterlock(
        id="hhp_boiler",
        name="High-High Pressure Trip",
        interlock_type=InterlockType.HIGH_PRESSURE,
        trigger_condition="Steam pressure > HHP setpoint",
        action="Trip fuel, open vent valve",
        sil_level=SILLevel.SIL_2,
        response_time_s=1.0,
        test_frequency="monthly",
        applicable_standards=["ASME BPVC", "NFPA 85"],
        voting_logic="2oo3",
    ),
    "flame_failure": SafetyInterlock(
        id="flame_failure",
        name="Flame Failure Trip",
        interlock_type=InterlockType.FLAME_FAILURE,
        trigger_condition="Loss of flame signal",
        action="Close fuel valves, alarm",
        sil_level=SILLevel.SIL_2,
        response_time_s=4.0,  # per NFPA 85 for large boilers
        test_frequency="annually",
        applicable_standards=["NFPA 85", "NFPA 86"],
        voting_logic="1oo1",
    ),
    "combustion_air_low": SafetyInterlock(
        id="combustion_air_low",
        name="Low Combustion Air Trip",
        interlock_type=InterlockType.COMBUSTION_AIR,
        trigger_condition="Combustion air flow < minimum",
        action="Trip fuel",
        sil_level=SILLevel.SIL_1,
        response_time_s=2.0,
        applicable_standards=["NFPA 85", "NFPA 86"],
    ),
    "purge_interlock": SafetyInterlock(
        id="purge_interlock",
        name="Pre-Ignition Purge Interlock",
        interlock_type=InterlockType.PURGE,
        trigger_condition="Purge not complete",
        action="Block fuel ignition",
        sil_level=SILLevel.SIL_2,
        response_time_s=0.5,
        applicable_standards=["NFPA 85", "NFPA 86"],
    ),
    "hht_heater": SafetyInterlock(
        id="hht_heater",
        name="High-High Temperature Trip",
        interlock_type=InterlockType.HIGH_TEMPERATURE,
        trigger_condition="Process outlet temp > HHT setpoint",
        action="Trip fuel, increase process flow",
        sil_level=SILLevel.SIL_1,
        response_time_s=5.0,
        applicable_standards=["API 556"],
    ),
    "low_flow_heater": SafetyInterlock(
        id="low_flow_heater",
        name="Low Process Flow Trip",
        interlock_type=InterlockType.LOW_FLOW,
        trigger_condition="Process flow < minimum",
        action="Trip fuel",
        sil_level=SILLevel.SIL_2,
        response_time_s=2.0,
        applicable_standards=["API 560"],
    ),
}


# =============================================================================
# Standard Protection Layers
# =============================================================================

PROTECTION_LAYERS = {
    "bpcs": ProtectionLayer(
        id="bpcs",
        name="Basic Process Control System",
        layer_type=ProtectionLayerType.BPCS,
        pfd=0.1,  # 10^-1
        description="DCS/PLC-based process control for normal operation",
    ),
    "alarm_response": ProtectionLayer(
        id="alarm_response",
        name="Alarm and Operator Response",
        layer_type=ProtectionLayerType.ALARM,
        pfd=0.1,  # 10^-1, varies with complexity
        description="Process alarm with trained operator response",
        response_time_s=600,  # 10 minutes typical
    ),
    "sis_sil1": ProtectionLayer(
        id="sis_sil1",
        name="SIL 1 Safety Instrumented System",
        layer_type=ProtectionLayerType.SIS,
        pfd=0.01,  # 10^-2
        description="Safety instrumented function meeting SIL 1",
    ),
    "sis_sil2": ProtectionLayer(
        id="sis_sil2",
        name="SIL 2 Safety Instrumented System",
        layer_type=ProtectionLayerType.SIS,
        pfd=0.001,  # 10^-3
        description="Safety instrumented function meeting SIL 2",
    ),
    "sis_sil3": ProtectionLayer(
        id="sis_sil3",
        name="SIL 3 Safety Instrumented System",
        layer_type=ProtectionLayerType.SIS,
        pfd=0.0001,  # 10^-4
        description="Safety instrumented function meeting SIL 3",
    ),
    "psv": ProtectionLayer(
        id="psv",
        name="Pressure Safety Valve",
        layer_type=ProtectionLayerType.PSV,
        pfd=0.01,  # 10^-2 typical
        description="Spring-loaded pressure relief valve",
        response_time_s=0.1,
    ),
    "rupture_disk": ProtectionLayer(
        id="rupture_disk",
        name="Rupture Disk",
        layer_type=ProtectionLayerType.RUPTURE_DISK,
        pfd=0.001,  # 10^-3
        description="Pressure relief rupture disk",
        response_time_s=0.001,
    ),
}


# =============================================================================
# Safety Ontology Manager
# =============================================================================

class SafetyOntology:
    """
    Manager for process heat safety ontology.

    Provides access to hazards, interlocks, and protection layers.
    """

    def __init__(self):
        self.hazards = PROCESS_HEAT_HAZARDS.copy()
        self.interlocks = STANDARD_INTERLOCKS.copy()
        self.protection_layers = PROTECTION_LAYERS.copy()

    def get_hazard(self, hazard_id: str) -> Optional[HazardClass]:
        """Get hazard by ID."""
        return self.hazards.get(hazard_id)

    def get_interlock(self, interlock_id: str) -> Optional[SafetyInterlock]:
        """Get interlock by ID."""
        return self.interlocks.get(interlock_id)

    def get_protection_layer(self, layer_id: str) -> Optional[ProtectionLayer]:
        """Get protection layer by ID."""
        return self.protection_layers.get(layer_id)

    def get_hazards_for_equipment(self, equipment_type: str) -> List[HazardClass]:
        """Get hazards applicable to an equipment type."""
        return [
            h for h in self.hazards.values()
            if equipment_type in h.applicable_equipment
        ]

    def get_interlocks_by_type(self, interlock_type: InterlockType) -> List[SafetyInterlock]:
        """Get interlocks of a specific type."""
        return [i for i in self.interlocks.values() if i.interlock_type == interlock_type]

    def get_interlocks_by_sil(self, sil_level: SILLevel) -> List[SafetyInterlock]:
        """Get interlocks requiring a specific SIL level."""
        return [i for i in self.interlocks.values() if i.sil_level == sil_level]

    def get_required_interlocks(self, equipment_type: str) -> List[SafetyInterlock]:
        """Get required interlocks for equipment type."""
        # Map equipment types to relevant interlocks
        equipment_interlocks = {
            "boiler": ["llc_boiler", "hhp_boiler", "flame_failure", "combustion_air_low", "purge_interlock"],
            "fired_heater": ["hht_heater", "low_flow_heater", "flame_failure", "purge_interlock"],
            "furnace": ["hht_heater", "flame_failure", "combustion_air_low", "purge_interlock"],
        }
        interlock_ids = equipment_interlocks.get(equipment_type, [])
        return [self.interlocks[iid] for iid in interlock_ids if iid in self.interlocks]

    def calculate_risk_reduction(self, protection_layer_ids: List[str]) -> float:
        """Calculate combined risk reduction from protection layers."""
        combined_pfd = 1.0
        for layer_id in protection_layer_ids:
            layer = self.protection_layers.get(layer_id)
            if layer:
                combined_pfd *= layer.pfd
        return combined_pfd

    def determine_sil_requirement(
        self,
        severity: HazardSeverity,
        likelihood: HazardLikelihood,
        target_pfd: float = 0.001
    ) -> SILLevel:
        """Determine SIL requirement based on risk matrix."""
        # Simplified risk matrix
        risk_matrix = {
            (HazardSeverity.CATASTROPHIC, HazardLikelihood.FREQUENT): SILLevel.SIL_4,
            (HazardSeverity.CATASTROPHIC, HazardLikelihood.PROBABLE): SILLevel.SIL_3,
            (HazardSeverity.CATASTROPHIC, HazardLikelihood.OCCASIONAL): SILLevel.SIL_3,
            (HazardSeverity.CATASTROPHIC, HazardLikelihood.REMOTE): SILLevel.SIL_2,
            (HazardSeverity.CRITICAL, HazardLikelihood.FREQUENT): SILLevel.SIL_3,
            (HazardSeverity.CRITICAL, HazardLikelihood.PROBABLE): SILLevel.SIL_2,
            (HazardSeverity.CRITICAL, HazardLikelihood.OCCASIONAL): SILLevel.SIL_2,
            (HazardSeverity.CRITICAL, HazardLikelihood.REMOTE): SILLevel.SIL_1,
            (HazardSeverity.MARGINAL, HazardLikelihood.FREQUENT): SILLevel.SIL_2,
            (HazardSeverity.MARGINAL, HazardLikelihood.PROBABLE): SILLevel.SIL_1,
            (HazardSeverity.MARGINAL, HazardLikelihood.OCCASIONAL): SILLevel.SIL_1,
        }
        return risk_matrix.get((severity, likelihood), SILLevel.SIL_NONE)

    def get_statistics(self) -> Dict[str, int]:
        """Get ontology statistics."""
        return {
            "total_hazards": len(self.hazards),
            "total_interlocks": len(self.interlocks),
            "total_protection_layers": len(self.protection_layers),
            "sil2_interlocks": len(self.get_interlocks_by_sil(SILLevel.SIL_2)),
            "sil3_interlocks": len(self.get_interlocks_by_sil(SILLevel.SIL_3)),
        }


# Module-level singleton
_safety_ontology: Optional[SafetyOntology] = None

def get_safety_ontology() -> SafetyOntology:
    """Get or create the safety ontology instance."""
    global _safety_ontology
    if _safety_ontology is None:
        _safety_ontology = SafetyOntology()
    return _safety_ontology
