# -*- coding: utf-8 -*-
"""
Process Ontology for Industrial Heat Systems
=============================================

Thermal processes, combustion processes, and heat transfer processes
for industrial decarbonization applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ProcessType(str, Enum):
    """Types of industrial processes."""
    COMBUSTION = "combustion"
    HEAT_TRANSFER = "heat_transfer"
    STEAM_GENERATION = "steam_generation"
    DRYING = "drying"
    MELTING = "melting"
    CALCINATION = "calcination"
    DISTILLATION = "distillation"
    EVAPORATION = "evaporation"
    HEAT_TREATMENT = "heat_treatment"


class HeatTransferMode(str, Enum):
    """Modes of heat transfer."""
    CONDUCTION = "conduction"
    CONVECTION = "convection"
    RADIATION = "radiation"
    COMBINED = "combined"


class FlowPattern(str, Enum):
    """Flow patterns in heat exchangers."""
    COUNTER_FLOW = "counter_flow"
    PARALLEL_FLOW = "parallel_flow"
    CROSS_FLOW = "cross_flow"
    MIXED_FLOW = "mixed_flow"
    SHELL_SIDE = "shell_side"


@dataclass
class ProcessParameter:
    """Parameter definition for a process."""
    name: str
    symbol: str
    unit: str
    description: str
    typical_range: Optional[tuple] = None
    is_input: bool = True


@dataclass
class ThermalProcess:
    """
    Thermal process definition.

    Attributes:
        id: Process identifier
        name: Process name
        process_type: Type of process
        description: Process description
        input_parameters: Input parameters
        output_parameters: Output parameters
        governing_equations: Key equations
        applicable_equipment: Equipment types used
        efficiency_factors: Factors affecting efficiency
    """
    id: str
    name: str
    process_type: ProcessType
    description: str
    input_parameters: List[ProcessParameter] = field(default_factory=list)
    output_parameters: List[ProcessParameter] = field(default_factory=list)
    governing_equations: List[str] = field(default_factory=list)
    applicable_equipment: List[str] = field(default_factory=list)
    efficiency_factors: List[str] = field(default_factory=list)
    temperature_range_c: Optional[tuple] = None
    pressure_range_bar: Optional[tuple] = None


@dataclass
class CombustionProcess:
    """
    Combustion process definition.

    Attributes:
        id: Process identifier
        name: Process name
        fuel_type: Type of fuel
        oxidizer: Oxidizer (air, oxygen, etc.)
        stoichiometric_air: Stoichiometric air-fuel ratio
        excess_air_range: Typical excess air range
        adiabatic_flame_temp: Adiabatic flame temperature
        products: Combustion products
    """
    id: str
    name: str
    fuel_type: str
    oxidizer: str = "air"
    stoichiometric_air: float = 0.0  # kg air / kg fuel
    excess_air_range: tuple = (0.05, 0.30)  # 5-30%
    adiabatic_flame_temp_c: float = 0.0
    products: List[str] = field(default_factory=list)
    emissions: Dict[str, str] = field(default_factory=dict)
    lower_heating_value_mj_kg: float = 0.0
    higher_heating_value_mj_kg: float = 0.0


@dataclass
class HeatTransferProcess:
    """
    Heat transfer process definition.

    Attributes:
        id: Process identifier
        name: Process name
        transfer_mode: Mode of heat transfer
        hot_fluid: Hot side fluid
        cold_fluid: Cold side fluid
        flow_pattern: Flow arrangement
        typical_htc: Typical heat transfer coefficient range
    """
    id: str
    name: str
    transfer_mode: HeatTransferMode
    hot_fluid: str
    cold_fluid: str
    flow_pattern: FlowPattern = FlowPattern.COUNTER_FLOW
    typical_htc_range: tuple = (0, 0)  # W/m²K
    fouling_factor_range: tuple = (0, 0)  # m²K/W
    effectiveness_range: tuple = (0.5, 0.95)


# =============================================================================
# Standard Process Definitions
# =============================================================================

COMBUSTION_PROCESSES = {
    "natural_gas_combustion": CombustionProcess(
        id="natural_gas_combustion",
        name="Natural Gas Combustion",
        fuel_type="natural_gas",
        stoichiometric_air=17.2,  # kg air / kg CH4
        excess_air_range=(0.05, 0.20),
        adiabatic_flame_temp_c=1950,
        products=["CO2", "H2O", "N2", "O2"],
        emissions={"CO2": "2.75 kg/kg fuel", "NOx": "varies"},
        lower_heating_value_mj_kg=50.0,
        higher_heating_value_mj_kg=55.5,
    ),
    "fuel_oil_combustion": CombustionProcess(
        id="fuel_oil_combustion",
        name="Fuel Oil Combustion",
        fuel_type="fuel_oil",
        stoichiometric_air=14.5,
        excess_air_range=(0.10, 0.25),
        adiabatic_flame_temp_c=2030,
        products=["CO2", "H2O", "N2", "O2", "SO2"],
        emissions={"CO2": "3.15 kg/kg fuel", "SO2": "varies"},
        lower_heating_value_mj_kg=42.5,
        higher_heating_value_mj_kg=45.0,
    ),
    "hydrogen_combustion": CombustionProcess(
        id="hydrogen_combustion",
        name="Hydrogen Combustion",
        fuel_type="hydrogen",
        stoichiometric_air=34.3,
        excess_air_range=(0.05, 0.15),
        adiabatic_flame_temp_c=2110,
        products=["H2O", "N2", "O2"],
        emissions={"CO2": "0 kg/kg fuel", "NOx": "varies with temp"},
        lower_heating_value_mj_kg=120.0,
        higher_heating_value_mj_kg=142.0,
    ),
    "biomass_combustion": CombustionProcess(
        id="biomass_combustion",
        name="Biomass Combustion",
        fuel_type="biomass",
        stoichiometric_air=6.0,  # varies with moisture
        excess_air_range=(0.20, 0.50),
        adiabatic_flame_temp_c=1400,
        products=["CO2", "H2O", "N2", "O2", "ash"],
        emissions={"CO2": "biogenic", "particulates": "varies"},
        lower_heating_value_mj_kg=15.0,
        higher_heating_value_mj_kg=18.0,
    ),
}

HEAT_TRANSFER_PROCESSES = {
    "shell_tube_hx": HeatTransferProcess(
        id="shell_tube_hx",
        name="Shell and Tube Heat Exchange",
        transfer_mode=HeatTransferMode.CONVECTION,
        hot_fluid="process_stream",
        cold_fluid="cooling_water",
        flow_pattern=FlowPattern.COUNTER_FLOW,
        typical_htc_range=(300, 1500),
        fouling_factor_range=(0.0001, 0.0005),
        effectiveness_range=(0.7, 0.95),
    ),
    "plate_hx": HeatTransferProcess(
        id="plate_hx",
        name="Plate Heat Exchange",
        transfer_mode=HeatTransferMode.CONVECTION,
        hot_fluid="liquid",
        cold_fluid="liquid",
        flow_pattern=FlowPattern.COUNTER_FLOW,
        typical_htc_range=(3000, 7000),
        fouling_factor_range=(0.00005, 0.0002),
        effectiveness_range=(0.8, 0.98),
    ),
    "air_cooled_hx": HeatTransferProcess(
        id="air_cooled_hx",
        name="Air Cooled Heat Exchange",
        transfer_mode=HeatTransferMode.COMBINED,
        hot_fluid="process_stream",
        cold_fluid="air",
        flow_pattern=FlowPattern.CROSS_FLOW,
        typical_htc_range=(20, 60),
        effectiveness_range=(0.5, 0.85),
    ),
    "radiation_furnace": HeatTransferProcess(
        id="radiation_furnace",
        name="Radiant Furnace Heat Transfer",
        transfer_mode=HeatTransferMode.RADIATION,
        hot_fluid="flame",
        cold_fluid="process_tubes",
        typical_htc_range=(50, 150),  # effective
        effectiveness_range=(0.4, 0.7),
    ),
    "convection_section": HeatTransferProcess(
        id="convection_section",
        name="Convection Section Heat Transfer",
        transfer_mode=HeatTransferMode.CONVECTION,
        hot_fluid="flue_gas",
        cold_fluid="process_stream",
        flow_pattern=FlowPattern.CROSS_FLOW,
        typical_htc_range=(30, 80),
        effectiveness_range=(0.6, 0.85),
    ),
    "economizer": HeatTransferProcess(
        id="economizer",
        name="Economizer Heat Recovery",
        transfer_mode=HeatTransferMode.CONVECTION,
        hot_fluid="flue_gas",
        cold_fluid="feedwater",
        flow_pattern=FlowPattern.COUNTER_FLOW,
        typical_htc_range=(40, 100),
        effectiveness_range=(0.5, 0.8),
    ),
    "air_preheater": HeatTransferProcess(
        id="air_preheater",
        name="Air Preheater Heat Recovery",
        transfer_mode=HeatTransferMode.CONVECTION,
        hot_fluid="flue_gas",
        cold_fluid="combustion_air",
        flow_pattern=FlowPattern.COUNTER_FLOW,
        typical_htc_range=(20, 50),
        effectiveness_range=(0.5, 0.75),
    ),
}

THERMAL_PROCESSES = {
    "steam_generation": ThermalProcess(
        id="steam_generation",
        name="Steam Generation",
        process_type=ProcessType.STEAM_GENERATION,
        description="Generation of steam from water using heat from combustion or other sources",
        input_parameters=[
            ProcessParameter("feedwater_flow", "ṁ_fw", "kg/h", "Feedwater mass flow rate"),
            ProcessParameter("feedwater_temp", "T_fw", "°C", "Feedwater temperature"),
            ProcessParameter("fuel_flow", "ṁ_fuel", "kg/h", "Fuel mass flow rate"),
            ProcessParameter("excess_air", "EA", "%", "Excess air percentage"),
        ],
        output_parameters=[
            ProcessParameter("steam_flow", "ṁ_s", "kg/h", "Steam mass flow rate", is_input=False),
            ProcessParameter("steam_temp", "T_s", "°C", "Steam temperature", is_input=False),
            ProcessParameter("steam_pressure", "P_s", "bar", "Steam pressure", is_input=False),
            ProcessParameter("efficiency", "η", "%", "Boiler efficiency", is_input=False),
        ],
        governing_equations=[
            "Q_in = ṁ_fuel × LHV",
            "Q_out = ṁ_s × (h_s - h_fw)",
            "η = Q_out / Q_in × 100%",
        ],
        applicable_equipment=["boiler", "hrsg", "waste_heat_boiler"],
        efficiency_factors=["excess_air", "stack_temperature", "blowdown_rate", "radiation_losses"],
        temperature_range_c=(100, 540),
        pressure_range_bar=(1, 200),
    ),
    "fired_heating": ThermalProcess(
        id="fired_heating",
        name="Fired Heating",
        process_type=ProcessType.HEAT_TRANSFER,
        description="Direct heating of process fluids using combustion",
        input_parameters=[
            ProcessParameter("process_flow", "ṁ_p", "kg/h", "Process fluid flow rate"),
            ProcessParameter("inlet_temp", "T_in", "°C", "Process inlet temperature"),
            ProcessParameter("fuel_flow", "ṁ_fuel", "kg/h", "Fuel flow rate"),
        ],
        output_parameters=[
            ProcessParameter("outlet_temp", "T_out", "°C", "Process outlet temperature", is_input=False),
            ProcessParameter("heat_duty", "Q", "kW", "Heat duty", is_input=False),
            ProcessParameter("efficiency", "η", "%", "Heater efficiency", is_input=False),
        ],
        governing_equations=[
            "Q = ṁ_p × Cp × (T_out - T_in)",
            "Q_in = ṁ_fuel × LHV",
            "η = Q / Q_in × 100%",
        ],
        applicable_equipment=["fired_heater", "process_furnace"],
        efficiency_factors=["excess_air", "bridgewall_temp", "convection_section_design"],
        temperature_range_c=(200, 900),
    ),
    "industrial_drying": ThermalProcess(
        id="industrial_drying",
        name="Industrial Drying",
        process_type=ProcessType.DRYING,
        description="Removal of moisture from materials using thermal energy",
        input_parameters=[
            ProcessParameter("material_flow", "ṁ_m", "kg/h", "Wet material flow rate"),
            ProcessParameter("inlet_moisture", "X_in", "%", "Inlet moisture content"),
            ProcessParameter("air_temp", "T_air", "°C", "Drying air temperature"),
            ProcessParameter("air_flow", "ṁ_air", "kg/h", "Air flow rate"),
        ],
        output_parameters=[
            ProcessParameter("outlet_moisture", "X_out", "%", "Outlet moisture content", is_input=False),
            ProcessParameter("evaporation_rate", "ṁ_evap", "kg/h", "Water evaporation rate", is_input=False),
            ProcessParameter("specific_energy", "SEC", "kJ/kg", "Specific energy consumption", is_input=False),
        ],
        governing_equations=[
            "ṁ_evap = ṁ_m × (X_in - X_out) / (100 - X_out)",
            "Q = ṁ_evap × (h_fg + Cp_v × ΔT)",
            "SEC = Q / ṁ_evap",
        ],
        applicable_equipment=["rotary_dryer", "fluid_bed_dryer", "spray_dryer"],
        efficiency_factors=["air_recirculation", "exhaust_temperature", "heat_recovery"],
        temperature_range_c=(50, 400),
    ),
}


# =============================================================================
# Process Ontology Manager
# =============================================================================

class ProcessOntology:
    """
    Manager for process heat ontology.

    Provides access to thermal, combustion, and heat transfer process definitions.
    """

    def __init__(self):
        self.thermal_processes = THERMAL_PROCESSES.copy()
        self.combustion_processes = COMBUSTION_PROCESSES.copy()
        self.heat_transfer_processes = HEAT_TRANSFER_PROCESSES.copy()

    def get_thermal_process(self, process_id: str) -> Optional[ThermalProcess]:
        """Get thermal process by ID."""
        return self.thermal_processes.get(process_id)

    def get_combustion_process(self, process_id: str) -> Optional[CombustionProcess]:
        """Get combustion process by ID."""
        return self.combustion_processes.get(process_id)

    def get_heat_transfer_process(self, process_id: str) -> Optional[HeatTransferProcess]:
        """Get heat transfer process by ID."""
        return self.heat_transfer_processes.get(process_id)

    def get_processes_by_type(self, process_type: ProcessType) -> List[ThermalProcess]:
        """Get all processes of a specific type."""
        return [p for p in self.thermal_processes.values() if p.process_type == process_type]

    def get_processes_for_equipment(self, equipment_type: str) -> List[ThermalProcess]:
        """Get processes applicable to an equipment type."""
        return [
            p for p in self.thermal_processes.values()
            if equipment_type in p.applicable_equipment
        ]

    def get_combustion_for_fuel(self, fuel_type: str) -> List[CombustionProcess]:
        """Get combustion processes for a fuel type."""
        return [p for p in self.combustion_processes.values() if fuel_type.lower() in p.fuel_type.lower()]

    def calculate_combustion_air(self, fuel_type: str, fuel_flow_kg_h: float, excess_air_pct: float) -> float:
        """Calculate required combustion air flow."""
        combustion = self.combustion_processes.get(f"{fuel_type}_combustion")
        if combustion:
            stoich_air = combustion.stoichiometric_air * fuel_flow_kg_h
            return stoich_air * (1 + excess_air_pct / 100)
        return 0.0

    def estimate_adiabatic_flame_temp(self, fuel_type: str, excess_air_pct: float = 10.0) -> float:
        """Estimate adiabatic flame temperature."""
        combustion = self.combustion_processes.get(f"{fuel_type}_combustion")
        if combustion:
            # Simple correction for excess air
            correction = 1 - (excess_air_pct / 100) * 0.5
            return combustion.adiabatic_flame_temp_c * correction
        return 0.0

    def get_statistics(self) -> Dict[str, int]:
        """Get ontology statistics."""
        return {
            "thermal_processes": len(self.thermal_processes),
            "combustion_processes": len(self.combustion_processes),
            "heat_transfer_processes": len(self.heat_transfer_processes),
        }


# Module-level singleton
_process_ontology: Optional[ProcessOntology] = None

def get_process_ontology() -> ProcessOntology:
    """Get or create the process ontology instance."""
    global _process_ontology
    if _process_ontology is None:
        _process_ontology = ProcessOntology()
    return _process_ontology
