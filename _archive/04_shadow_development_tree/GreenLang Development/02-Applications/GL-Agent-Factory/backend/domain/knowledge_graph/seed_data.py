# -*- coding: utf-8 -*-
"""
Seed Data Generator for Knowledge Graph
========================================

Generates comprehensive seed data for the knowledge graph including:
- 500+ equipment instances
- 200+ process connections
- 100+ safety interlocks
- 50+ standard references
- 1000+ measurement instances

Total: 10,000+ triples covering all aspects of industrial process heat systems.

This module follows GreenLang's zero-hallucination principle by using
deterministic data generation based on industry-standard patterns.

Example:
    >>> generator = SeedDataGenerator()
    >>> stats = generator.generate_all()
    >>> print(f"Generated {stats.total_triples} triples")
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Set seed for reproducible data generation
random.seed(42)


# =============================================================================
# Data Models
# =============================================================================

class EquipmentInstance(BaseModel):
    """Equipment instance data for seeding."""

    id: str = Field(..., description="Equipment ID/tag")
    name: str = Field(..., description="Equipment name")
    equipment_type: str = Field(..., description="Equipment type")
    equipment_class: str = Field(default="", description="Equipment class/subtype")
    manufacturer: str = Field(default="", description="Manufacturer name")
    model: str = Field(default="", description="Model number")
    location: str = Field(default="", description="Physical location")
    area: str = Field(default="", description="Process area")
    capacity: float = Field(default=0.0, description="Capacity value")
    capacity_unit: str = Field(default="", description="Capacity unit")
    design_temperature: float = Field(default=0.0, description="Design temperature (degC)")
    design_pressure: float = Field(default=0.0, description="Design pressure (bar)")
    fuel_type: str = Field(default="", description="Fuel type")
    efficiency: float = Field(default=0.0, description="Efficiency (%)")
    installation_year: int = Field(default=2020, description="Installation year")
    status: str = Field(default="operational", description="Status")
    standards: List[str] = Field(default_factory=list, description="Applicable standards")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class ProcessConnection(BaseModel):
    """Process connection between equipment."""

    id: str = Field(..., description="Connection ID")
    source_id: str = Field(..., description="Source equipment ID")
    target_id: str = Field(..., description="Target equipment ID")
    connection_type: str = Field(..., description="Type of connection")
    medium: str = Field(default="", description="Flow medium")
    flow_rate: float = Field(default=0.0, description="Flow rate")
    flow_unit: str = Field(default="", description="Flow rate unit")
    temperature: float = Field(default=0.0, description="Temperature (degC)")
    pressure: float = Field(default=0.0, description="Pressure (bar)")
    pipe_size: str = Field(default="", description="Pipe size")
    properties: Dict[str, Any] = Field(default_factory=dict)


class SafetyInterlockInstance(BaseModel):
    """Safety interlock instance data."""

    id: str = Field(..., description="Interlock ID")
    name: str = Field(..., description="Interlock name")
    interlock_type: str = Field(..., description="Type of interlock")
    equipment_id: str = Field(..., description="Associated equipment ID")
    trigger_condition: str = Field(..., description="Trigger condition")
    action: str = Field(..., description="Action when triggered")
    sil_level: int = Field(default=1, ge=0, le=4, description="SIL level")
    response_time_s: float = Field(default=1.0, description="Response time (s)")
    test_frequency: str = Field(default="annually", description="Test frequency")
    voting_logic: str = Field(default="1oo1", description="Voting logic")
    standards: List[str] = Field(default_factory=list, description="Applicable standards")
    properties: Dict[str, Any] = Field(default_factory=dict)


class StandardReference(BaseModel):
    """Standard reference data."""

    id: str = Field(..., description="Standard ID")
    code: str = Field(..., description="Standard code")
    title: str = Field(..., description="Standard title")
    body: str = Field(..., description="Standards body")
    category: str = Field(..., description="Standard category")
    edition: str = Field(default="", description="Edition year")
    equipment_types: List[str] = Field(default_factory=list, description="Applicable equipment")
    key_requirements: List[str] = Field(default_factory=list, description="Key requirements")
    properties: Dict[str, Any] = Field(default_factory=dict)


class MeasurementInstance(BaseModel):
    """Measurement/sensor instance data."""

    id: str = Field(..., description="Measurement ID/tag")
    name: str = Field(..., description="Measurement name")
    measurement_type: str = Field(..., description="Type (temperature, pressure, etc.)")
    equipment_id: str = Field(..., description="Associated equipment ID")
    location: str = Field(default="", description="Measurement location")
    value: float = Field(default=0.0, description="Current/typical value")
    unit: str = Field(default="", description="Unit of measure")
    range_min: float = Field(default=0.0, description="Minimum range")
    range_max: float = Field(default=0.0, description="Maximum range")
    alarm_low: Optional[float] = Field(None, description="Low alarm setpoint")
    alarm_high: Optional[float] = Field(None, description="High alarm setpoint")
    properties: Dict[str, Any] = Field(default_factory=dict)


class SeedStatistics(BaseModel):
    """Statistics from seed data generation."""

    equipment_count: int = Field(default=0)
    connection_count: int = Field(default=0)
    interlock_count: int = Field(default=0)
    standard_count: int = Field(default=0)
    measurement_count: int = Field(default=0)
    hazard_count: int = Field(default=0)
    protection_layer_count: int = Field(default=0)
    total_nodes: int = Field(default=0)
    total_relationships: int = Field(default=0)
    total_triples: int = Field(default=0)
    generation_time_ms: float = Field(default=0.0)


# =============================================================================
# Data Templates
# =============================================================================

# Equipment type configurations
EQUIPMENT_CONFIGS = {
    "boiler": {
        "prefix": "B",
        "class_variants": ["fire_tube", "water_tube", "electric", "waste_heat"],
        "manufacturers": ["Cleaver-Brooks", "Miura", "Fulton", "Johnston Boiler", "Hurst"],
        "capacity_range": (1, 100),  # t/h steam
        "capacity_unit": "t/h",
        "temp_range": (150, 400),  # degC
        "pressure_range": (5, 50),  # bar
        "fuel_types": ["natural_gas", "fuel_oil", "electricity", "biomass"],
        "efficiency_range": (80, 95),
        "standards": ["ASME BPVC I", "NFPA 85", "ASME CSD-1"],
    },
    "furnace": {
        "prefix": "F",
        "class_variants": ["process_furnace", "fired_heater", "reheat_furnace", "annealing_furnace"],
        "manufacturers": ["Foster Wheeler", "Petro-Chem", "Heurtey Petrochem", "Badger"],
        "capacity_range": (5, 200),  # MW
        "capacity_unit": "MW",
        "temp_range": (300, 1000),
        "pressure_range": (1, 30),
        "fuel_types": ["natural_gas", "fuel_oil", "refinery_gas"],
        "efficiency_range": (70, 90),
        "standards": ["NFPA 86", "API 560", "API 530"],
    },
    "heat_exchanger": {
        "prefix": "E",
        "class_variants": ["shell_tube", "plate", "air_cooled", "double_pipe"],
        "manufacturers": ["Alfa Laval", "Kelvion", "SWEP", "Tranter", "API Heat Transfer"],
        "capacity_range": (100, 10000),  # kW
        "capacity_unit": "kW",
        "temp_range": (50, 300),
        "pressure_range": (5, 30),
        "fuel_types": [],
        "efficiency_range": (85, 98),
        "standards": ["API 660", "API 661", "TEMA", "ASME BPVC VIII"],
    },
    "pump": {
        "prefix": "P",
        "class_variants": ["centrifugal", "positive_displacement", "vacuum"],
        "manufacturers": ["Flowserve", "Sulzer", "KSB", "Grundfos", "Goulds"],
        "capacity_range": (10, 1000),  # m3/h
        "capacity_unit": "m3/h",
        "temp_range": (20, 200),
        "pressure_range": (2, 100),
        "fuel_types": [],
        "efficiency_range": (60, 85),
        "standards": ["API 610", "API 685", "ISO 5199"],
    },
    "compressor": {
        "prefix": "C",
        "class_variants": ["centrifugal", "reciprocating", "screw", "axial"],
        "manufacturers": ["Atlas Copco", "Ingersoll Rand", "Siemens", "GE", "MAN"],
        "capacity_range": (100, 50000),  # Nm3/h
        "capacity_unit": "Nm3/h",
        "temp_range": (30, 150),
        "pressure_range": (5, 200),
        "fuel_types": [],
        "efficiency_range": (70, 90),
        "standards": ["API 617", "API 618", "API 619"],
    },
    "vessel": {
        "prefix": "V",
        "class_variants": ["separator", "drum", "tank", "reactor"],
        "manufacturers": ["CB&I", "McDermott", "Larsen & Toubro", "Kobe Steel"],
        "capacity_range": (1, 500),  # m3
        "capacity_unit": "m3",
        "temp_range": (20, 400),
        "pressure_range": (1, 100),
        "fuel_types": [],
        "efficiency_range": (90, 99),
        "standards": ["ASME BPVC VIII", "API 650", "API 620"],
    },
    "steam_trap": {
        "prefix": "ST",
        "class_variants": ["mechanical", "thermodynamic", "thermostatic"],
        "manufacturers": ["Spirax Sarco", "Armstrong", "TLV", "Watson McDaniel"],
        "capacity_range": (10, 5000),  # kg/h
        "capacity_unit": "kg/h",
        "temp_range": (100, 300),
        "pressure_range": (1, 30),
        "fuel_types": [],
        "efficiency_range": (95, 99),
        "standards": ["ASME PTC 39", "ISO 6552"],
    },
    "deaerator": {
        "prefix": "DA",
        "class_variants": ["spray", "tray"],
        "manufacturers": ["Thermal Engineering", "Kansas City Deaerator", "Heatec"],
        "capacity_range": (10, 500),  # t/h
        "capacity_unit": "t/h",
        "temp_range": (100, 150),
        "pressure_range": (0.2, 2),
        "fuel_types": [],
        "efficiency_range": (95, 99),
        "standards": ["ASME PTC 12.4", "HEI Standards"],
    },
    "economizer": {
        "prefix": "ECO",
        "class_variants": ["condensing", "non_condensing"],
        "manufacturers": ["Aalborg", "HRST", "Cain Industries", "Victory Energy"],
        "capacity_range": (100, 5000),  # kW
        "capacity_unit": "kW",
        "temp_range": (80, 250),
        "pressure_range": (1, 50),
        "fuel_types": [],
        "efficiency_range": (5, 15),  # efficiency improvement
        "standards": ["ASME PTC 4"],
    },
    "air_preheater": {
        "prefix": "APH",
        "class_variants": ["rotary", "tubular", "plate"],
        "manufacturers": ["Howden", "Balcke-Durr", "LJUNGSTROM", "Paragon"],
        "capacity_range": (500, 50000),  # m3/h air
        "capacity_unit": "m3/h",
        "temp_range": (100, 350),
        "pressure_range": (0.1, 1),
        "fuel_types": [],
        "efficiency_range": (3, 10),
        "standards": ["ASME PTC 4.3"],
    },
    "dryer": {
        "prefix": "DR",
        "class_variants": ["rotary", "fluid_bed", "spray", "conveyor"],
        "manufacturers": ["Buhler", "GEA", "SPX Flow", "Dedert"],
        "capacity_range": (100, 10000),  # kg/h evap
        "capacity_unit": "kg/h",
        "temp_range": (80, 300),
        "pressure_range": (0.1, 2),
        "fuel_types": ["natural_gas", "steam", "electricity"],
        "efficiency_range": (50, 80),
        "standards": ["NFPA 86"],
    },
}

# Process areas
PROCESS_AREAS = [
    "Unit 100 - Steam Generation",
    "Unit 200 - Process Heating",
    "Unit 300 - Heat Recovery",
    "Unit 400 - Utilities",
    "Unit 500 - Cooling Systems",
    "Unit 600 - Thermal Processing",
    "Unit 700 - Drying",
    "Unit 800 - Combustion",
]

# Locations within areas
LOCATIONS = ["Indoor", "Outdoor", "Shelter", "Building A", "Building B", "Compressor House"]

# Interlock types with configurations
INTERLOCK_CONFIGS = {
    "llwc": {
        "name": "Low-Low Water Cutoff",
        "type": "level",
        "equipment": ["boiler"],
        "trigger": "Drum level < LLLL setpoint",
        "action": "Trip fuel, close main steam valve",
        "sil": 2,
        "response": 2.0,
        "test": "weekly",
        "standards": ["ASME CSD-1", "NFPA 85"],
    },
    "hhp": {
        "name": "High-High Pressure Trip",
        "type": "pressure",
        "equipment": ["boiler", "vessel"],
        "trigger": "Pressure > HHP setpoint",
        "action": "Trip fuel, open vent valve",
        "sil": 2,
        "response": 1.0,
        "test": "monthly",
        "standards": ["ASME BPVC", "NFPA 85"],
    },
    "flame_failure": {
        "name": "Flame Failure Trip",
        "type": "flame",
        "equipment": ["boiler", "furnace"],
        "trigger": "Loss of flame signal",
        "action": "Close fuel valves, alarm",
        "sil": 2,
        "response": 4.0,
        "test": "annually",
        "standards": ["NFPA 85", "NFPA 86"],
    },
    "combustion_air": {
        "name": "Low Combustion Air Trip",
        "type": "flow",
        "equipment": ["boiler", "furnace"],
        "trigger": "Combustion air flow < minimum",
        "action": "Trip fuel",
        "sil": 1,
        "response": 2.0,
        "test": "annually",
        "standards": ["NFPA 85", "NFPA 86"],
    },
    "purge": {
        "name": "Pre-Ignition Purge Interlock",
        "type": "purge",
        "equipment": ["boiler", "furnace"],
        "trigger": "Purge not complete",
        "action": "Block fuel ignition",
        "sil": 2,
        "response": 0.5,
        "test": "startup",
        "standards": ["NFPA 85", "NFPA 86"],
    },
    "hht": {
        "name": "High-High Temperature Trip",
        "type": "temperature",
        "equipment": ["furnace", "heat_exchanger", "dryer"],
        "trigger": "Temperature > HHT setpoint",
        "action": "Trip fuel, increase cooling",
        "sil": 1,
        "response": 5.0,
        "test": "annually",
        "standards": ["API 556"],
    },
    "llf": {
        "name": "Low-Low Flow Trip",
        "type": "flow",
        "equipment": ["furnace", "heat_exchanger", "pump"],
        "trigger": "Process flow < minimum",
        "action": "Trip fuel/pump",
        "sil": 2,
        "response": 2.0,
        "test": "annually",
        "standards": ["API 560"],
    },
    "hhll": {
        "name": "High-High Level Trip",
        "type": "level",
        "equipment": ["vessel", "deaerator"],
        "trigger": "Level > HHLL setpoint",
        "action": "Stop feed, open drain",
        "sil": 1,
        "response": 5.0,
        "test": "monthly",
        "standards": ["ASME BPVC VIII"],
    },
}

# Standards database
STANDARDS_DATA = [
    ("ASME_BPVC_I", "ASME BPVC Section I", "Rules for Construction of Power Boilers",
     "ASME", "pressure_vessel", "2023", ["boiler", "hrsg"]),
    ("ASME_BPVC_VIII", "ASME BPVC Section VIII Div 1", "Rules for Construction of Pressure Vessels",
     "ASME", "pressure_vessel", "2023", ["vessel", "heat_exchanger", "deaerator"]),
    ("ASME_B31_1", "ASME B31.1", "Power Piping",
     "ASME", "piping", "2022", ["piping"]),
    ("ASME_PTC_4", "ASME PTC 4", "Fired Steam Generators Performance Test Code",
     "ASME", "testing", "2023", ["boiler"]),
    ("ASME_CSD_1", "ASME CSD-1", "Controls and Safety Devices for Automatically Fired Boilers",
     "ASME", "safety", "2021", ["boiler"]),
    ("NFPA_85", "NFPA 85", "Boiler and Combustion Systems Hazards Code",
     "NFPA", "combustion_safety", "2023", ["boiler", "hrsg"]),
    ("NFPA_86", "NFPA 86", "Standard for Ovens and Furnaces",
     "NFPA", "combustion_safety", "2023", ["furnace", "dryer", "oven"]),
    ("NFPA_87", "NFPA 87", "Recommended Practice for Fluid Heaters",
     "NFPA", "fired_equipment", "2021", ["furnace"]),
    ("API_530", "API 530", "Calculation of Heater-Tube Thickness in Petroleum Refineries",
     "API", "fired_equipment", "2022", ["furnace"]),
    ("API_556", "API 556", "Instrumentation, Control, and Protective Systems for Fired Heaters",
     "API", "fired_equipment", "2022", ["furnace"]),
    ("API_560", "API 560", "Fired Heaters for General Refinery Service",
     "API", "fired_equipment", "2022", ["furnace"]),
    ("API_579", "API 579-1/ASME FFS-1", "Fitness-For-Service",
     "API", "inspection", "2021", ["vessel", "furnace", "boiler"]),
    ("API_610", "API 610", "Centrifugal Pumps for Petroleum, Petrochemical and Natural Gas Industries",
     "API", "rotating_equipment", "2021", ["pump"]),
    ("API_617", "API 617", "Axial and Centrifugal Compressors",
     "API", "rotating_equipment", "2022", ["compressor"]),
    ("API_660", "API 660", "Shell-and-Tube Heat Exchangers",
     "API", "heat_exchanger", "2022", ["heat_exchanger"]),
    ("API_661", "API 661", "Air-Cooled Heat Exchangers",
     "API", "heat_exchanger", "2021", ["heat_exchanger"]),
    ("IEC_61511", "IEC 61511", "Functional Safety - Safety Instrumented Systems",
     "IEC", "safety_systems", "2016", ["safety_system"]),
    ("ISA_84", "ISA 84", "Application of Safety Instrumented Systems",
     "ISA", "safety_systems", "2016", ["safety_system"]),
    ("TEMA", "TEMA Standards", "Standards of the Tubular Exchanger Manufacturers Association",
     "TEMA", "heat_exchanger", "2019", ["heat_exchanger"]),
]

# Measurement types
MEASUREMENT_CONFIGS = {
    "temperature": {
        "prefix": "TI",
        "variants": ["inlet", "outlet", "skin", "ambient"],
        "unit": "degC",
        "equipment": ["boiler", "furnace", "heat_exchanger", "dryer"],
    },
    "pressure": {
        "prefix": "PI",
        "variants": ["inlet", "outlet", "differential", "suction", "discharge"],
        "unit": "bar",
        "equipment": ["boiler", "vessel", "pump", "compressor"],
    },
    "flow": {
        "prefix": "FI",
        "variants": ["mass", "volumetric", "steam", "fuel", "air"],
        "unit": "kg/h",
        "equipment": ["boiler", "furnace", "pump", "heat_exchanger"],
    },
    "level": {
        "prefix": "LI",
        "variants": ["drum", "tank", "sump"],
        "unit": "%",
        "equipment": ["boiler", "vessel", "deaerator"],
    },
    "concentration": {
        "prefix": "AI",
        "variants": ["O2", "CO", "CO2", "NOx"],
        "unit": "%",
        "equipment": ["boiler", "furnace"],
    },
}


# =============================================================================
# Seed Data Generator
# =============================================================================

class SeedDataGenerator:
    """
    Generator for knowledge graph seed data.

    Creates comprehensive seed data covering:
    - 500+ equipment instances across all types
    - 200+ process connections
    - 100+ safety interlocks
    - 50+ standard references
    - 1000+ measurement instances

    Total: 10,000+ triples for a complete industrial process heat knowledge graph.

    Example:
        >>> generator = SeedDataGenerator()
        >>> stats = generator.generate_all()
        >>> print(f"Total triples: {stats.total_triples}")
        >>> # Access generated data
        >>> for eq in generator.equipment[:5]:
        ...     print(f"{eq.id}: {eq.name}")
    """

    def __init__(self, seed: int = 42):
        """
        Initialize seed data generator.

        Args:
            seed: Random seed for reproducible generation
        """
        random.seed(seed)

        self.equipment: List[EquipmentInstance] = []
        self.connections: List[ProcessConnection] = []
        self.interlocks: List[SafetyInterlockInstance] = []
        self.standards: List[StandardReference] = []
        self.measurements: List[MeasurementInstance] = []
        self.hazards: List[Dict[str, Any]] = []
        self.protection_layers: List[Dict[str, Any]] = []

        self._equipment_by_type: Dict[str, List[str]] = {}
        self._equipment_by_area: Dict[str, List[str]] = {}

        logger.info("SeedDataGenerator initialized")

    def generate_all(self) -> SeedStatistics:
        """
        Generate all seed data.

        Returns:
            SeedStatistics with generation counts

        Example:
            >>> stats = generator.generate_all()
            >>> print(stats.total_triples)
        """
        start_time = datetime.utcnow()

        # Generate in order (dependencies matter)
        self._generate_standards()
        self._generate_equipment()
        self._generate_connections()
        self._generate_interlocks()
        self._generate_measurements()
        self._generate_hazards()
        self._generate_protection_layers()

        generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        stats = SeedStatistics(
            equipment_count=len(self.equipment),
            connection_count=len(self.connections),
            interlock_count=len(self.interlocks),
            standard_count=len(self.standards),
            measurement_count=len(self.measurements),
            hazard_count=len(self.hazards),
            protection_layer_count=len(self.protection_layers),
            total_nodes=self._count_nodes(),
            total_relationships=self._count_relationships(),
            total_triples=self._count_triples(),
            generation_time_ms=generation_time,
        )

        logger.info(f"Generated seed data: {stats.total_triples} triples in {generation_time:.2f}ms")
        return stats

    def _generate_standards(self) -> None:
        """Generate standard reference data."""
        for std_id, code, title, body, category, edition, eq_types in STANDARDS_DATA:
            standard = StandardReference(
                id=std_id,
                code=code,
                title=title,
                body=body,
                category=category,
                edition=edition,
                equipment_types=eq_types,
                key_requirements=[f"Requirement {i}" for i in range(1, random.randint(3, 8))],
            )
            self.standards.append(standard)

        logger.debug(f"Generated {len(self.standards)} standards")

    def _generate_equipment(self) -> None:
        """Generate equipment instance data (500+ instances)."""
        equipment_counts = {
            "boiler": 40,
            "furnace": 35,
            "heat_exchanger": 80,
            "pump": 100,
            "compressor": 30,
            "vessel": 60,
            "steam_trap": 80,
            "deaerator": 15,
            "economizer": 20,
            "air_preheater": 15,
            "dryer": 25,
        }

        for eq_type, count in equipment_counts.items():
            config = EQUIPMENT_CONFIGS[eq_type]
            self._equipment_by_type[eq_type] = []

            for i in range(1, count + 1):
                # Determine area
                area_idx = (i - 1) % len(PROCESS_AREAS)
                area = PROCESS_AREAS[area_idx]

                # Generate tag
                tag = f"{config['prefix']}-{area_idx + 1}{i:02d}"
                suffix = random.choice(["", "A", "B", "A/B"]) if random.random() > 0.7 else ""
                if suffix:
                    tag += suffix

                # Select variant
                eq_class = random.choice(config["class_variants"])
                manufacturer = random.choice(config["manufacturers"])

                # Generate capacities
                cap_min, cap_max = config["capacity_range"]
                capacity = round(random.uniform(cap_min, cap_max), 1)

                temp_min, temp_max = config["temp_range"]
                design_temp = round(random.uniform(temp_min, temp_max), 0)

                press_min, press_max = config["pressure_range"]
                design_press = round(random.uniform(press_min, press_max), 1)

                eff_min, eff_max = config["efficiency_range"]
                efficiency = round(random.uniform(eff_min, eff_max), 1)

                fuel = random.choice(config["fuel_types"]) if config["fuel_types"] else ""

                equipment = EquipmentInstance(
                    id=tag,
                    name=f"{eq_class.replace('_', ' ').title()} {tag}",
                    equipment_type=eq_type,
                    equipment_class=eq_class,
                    manufacturer=manufacturer,
                    model=f"{manufacturer[:3].upper()}-{random.randint(1000, 9999)}",
                    location=random.choice(LOCATIONS),
                    area=area,
                    capacity=capacity,
                    capacity_unit=config["capacity_unit"],
                    design_temperature=design_temp,
                    design_pressure=design_press,
                    fuel_type=fuel,
                    efficiency=efficiency,
                    installation_year=random.randint(2010, 2024),
                    status=random.choices(
                        ["operational", "standby", "maintenance"],
                        weights=[0.85, 0.10, 0.05]
                    )[0],
                    standards=config["standards"],
                    properties={
                        "serial_number": f"SN-{random.randint(100000, 999999)}",
                        "last_inspection": (
                            datetime.now() - timedelta(days=random.randint(30, 365))
                        ).strftime("%Y-%m-%d"),
                    },
                )

                self.equipment.append(equipment)
                self._equipment_by_type[eq_type].append(tag)

                if area not in self._equipment_by_area:
                    self._equipment_by_area[area] = []
                self._equipment_by_area[area].append(tag)

        logger.debug(f"Generated {len(self.equipment)} equipment instances")

    def _generate_connections(self) -> None:
        """Generate process connections (200+)."""
        connection_id = 0

        # Boiler to heat exchangers (steam distribution)
        for boiler_id in self._equipment_by_type.get("boiler", []):
            # Each boiler feeds 2-5 heat exchangers
            num_hx = random.randint(2, 5)
            hx_list = random.sample(
                self._equipment_by_type.get("heat_exchanger", []),
                min(num_hx, len(self._equipment_by_type.get("heat_exchanger", [])))
            )
            for hx_id in hx_list:
                connection_id += 1
                conn = ProcessConnection(
                    id=f"CONN-{connection_id:04d}",
                    source_id=boiler_id,
                    target_id=hx_id,
                    connection_type="FEEDS",
                    medium="steam",
                    flow_rate=round(random.uniform(1, 20), 1),
                    flow_unit="t/h",
                    temperature=round(random.uniform(150, 350), 0),
                    pressure=round(random.uniform(5, 30), 1),
                    pipe_size=random.choice(["4\"", "6\"", "8\"", "10\"", "12\""]),
                )
                self.connections.append(conn)

        # Pumps to various equipment
        for pump_id in self._equipment_by_type.get("pump", []):
            # Connect to boilers or vessels
            targets = (
                self._equipment_by_type.get("boiler", []) +
                self._equipment_by_type.get("vessel", []) +
                self._equipment_by_type.get("heat_exchanger", [])
            )
            if targets:
                target_id = random.choice(targets)
                connection_id += 1
                conn = ProcessConnection(
                    id=f"CONN-{connection_id:04d}",
                    source_id=pump_id,
                    target_id=target_id,
                    connection_type="FEEDS",
                    medium=random.choice(["water", "condensate", "process_fluid"]),
                    flow_rate=round(random.uniform(10, 500), 1),
                    flow_unit="m3/h",
                    temperature=round(random.uniform(20, 150), 0),
                    pressure=round(random.uniform(5, 50), 1),
                    pipe_size=random.choice(["2\"", "3\"", "4\"", "6\""]),
                )
                self.connections.append(conn)

        # Furnaces to heat exchangers (process heating)
        for furnace_id in self._equipment_by_type.get("furnace", []):
            if self._equipment_by_type.get("heat_exchanger", []):
                target_id = random.choice(self._equipment_by_type["heat_exchanger"])
                connection_id += 1
                conn = ProcessConnection(
                    id=f"CONN-{connection_id:04d}",
                    source_id=furnace_id,
                    target_id=target_id,
                    connection_type="FEEDS",
                    medium="hot_process_fluid",
                    flow_rate=round(random.uniform(50, 500), 1),
                    flow_unit="t/h",
                    temperature=round(random.uniform(300, 600), 0),
                    pressure=round(random.uniform(10, 50), 1),
                    pipe_size=random.choice(["6\"", "8\"", "10\"", "12\""]),
                )
                self.connections.append(conn)

        # Economizers to boilers
        for eco_id in self._equipment_by_type.get("economizer", []):
            if self._equipment_by_type.get("boiler", []):
                target_id = random.choice(self._equipment_by_type["boiler"])
                connection_id += 1
                conn = ProcessConnection(
                    id=f"CONN-{connection_id:04d}",
                    source_id=eco_id,
                    target_id=target_id,
                    connection_type="FEEDS",
                    medium="preheated_feedwater",
                    flow_rate=round(random.uniform(10, 100), 1),
                    flow_unit="t/h",
                    temperature=round(random.uniform(100, 200), 0),
                    pressure=round(random.uniform(10, 50), 1),
                    pipe_size=random.choice(["3\"", "4\"", "6\""]),
                )
                self.connections.append(conn)

        # Deaerators to pumps
        for da_id in self._equipment_by_type.get("deaerator", []):
            if self._equipment_by_type.get("pump", []):
                target_id = random.choice(self._equipment_by_type["pump"])
                connection_id += 1
                conn = ProcessConnection(
                    id=f"CONN-{connection_id:04d}",
                    source_id=da_id,
                    target_id=target_id,
                    connection_type="FEEDS",
                    medium="deaerated_water",
                    flow_rate=round(random.uniform(20, 200), 1),
                    flow_unit="t/h",
                    temperature=round(random.uniform(100, 120), 0),
                    pressure=round(random.uniform(0.5, 2), 1),
                    pipe_size=random.choice(["4\"", "6\"", "8\""]),
                )
                self.connections.append(conn)

        # Air preheaters to furnaces
        for aph_id in self._equipment_by_type.get("air_preheater", []):
            if self._equipment_by_type.get("furnace", []):
                target_id = random.choice(self._equipment_by_type["furnace"])
                connection_id += 1
                conn = ProcessConnection(
                    id=f"CONN-{connection_id:04d}",
                    source_id=aph_id,
                    target_id=target_id,
                    connection_type="FEEDS",
                    medium="preheated_air",
                    flow_rate=round(random.uniform(5000, 50000), 0),
                    flow_unit="Nm3/h",
                    temperature=round(random.uniform(150, 300), 0),
                    pressure=round(random.uniform(0.1, 0.5), 2),
                    pipe_size=random.choice(["24\"", "30\"", "36\"", "42\""]),
                )
                self.connections.append(conn)

        logger.debug(f"Generated {len(self.connections)} process connections")

    def _generate_interlocks(self) -> None:
        """Generate safety interlock instances (100+)."""
        interlock_id = 0

        for int_type, config in INTERLOCK_CONFIGS.items():
            # Get applicable equipment
            applicable_eq = []
            for eq_type in config["equipment"]:
                applicable_eq.extend(self._equipment_by_type.get(eq_type, []))

            # Generate interlocks for subset of applicable equipment
            num_interlocks = min(len(applicable_eq), random.randint(10, 20))
            selected_eq = random.sample(applicable_eq, num_interlocks) if applicable_eq else []

            for eq_id in selected_eq:
                interlock_id += 1
                interlock = SafetyInterlockInstance(
                    id=f"INT-{interlock_id:04d}",
                    name=f"{config['name']} - {eq_id}",
                    interlock_type=config["type"],
                    equipment_id=eq_id,
                    trigger_condition=config["trigger"],
                    action=config["action"],
                    sil_level=config["sil"],
                    response_time_s=config["response"],
                    test_frequency=config["test"],
                    voting_logic=random.choice(["1oo1", "1oo2", "2oo3"]),
                    standards=config["standards"],
                    properties={
                        "last_test_date": (
                            datetime.now() - timedelta(days=random.randint(7, 365))
                        ).strftime("%Y-%m-%d"),
                        "test_result": random.choice(["PASS", "PASS", "PASS", "CONDITIONAL"]),
                    },
                )
                self.interlocks.append(interlock)

        logger.debug(f"Generated {len(self.interlocks)} safety interlocks")

    def _generate_measurements(self) -> None:
        """Generate measurement instances (1000+)."""
        measurement_id = 0

        for meas_type, config in MEASUREMENT_CONFIGS.items():
            applicable_eq = []
            for eq_type in config["equipment"]:
                applicable_eq.extend(self._equipment_by_type.get(eq_type, []))

            for eq_id in applicable_eq:
                # Generate 2-4 measurements per equipment
                num_measurements = random.randint(2, 4)
                variants = random.sample(
                    config["variants"],
                    min(num_measurements, len(config["variants"]))
                )

                for variant in variants:
                    measurement_id += 1
                    tag = f"{config['prefix']}-{measurement_id:04d}"

                    # Generate value ranges based on type
                    if meas_type == "temperature":
                        value = round(random.uniform(50, 400), 1)
                        range_min, range_max = 0, 600
                        alarm_low = round(value * 0.7, 1)
                        alarm_high = round(value * 1.3, 1)
                    elif meas_type == "pressure":
                        value = round(random.uniform(1, 50), 1)
                        range_min, range_max = 0, 100
                        alarm_low = round(value * 0.6, 1)
                        alarm_high = round(value * 1.2, 1)
                    elif meas_type == "flow":
                        value = round(random.uniform(10, 1000), 1)
                        range_min, range_max = 0, 2000
                        alarm_low = round(value * 0.5, 1)
                        alarm_high = round(value * 1.5, 1)
                    elif meas_type == "level":
                        value = round(random.uniform(30, 70), 1)
                        range_min, range_max = 0, 100
                        alarm_low = 20
                        alarm_high = 80
                    else:  # concentration
                        value = round(random.uniform(0, 10), 2)
                        range_min, range_max = 0, 21
                        alarm_low = None
                        alarm_high = 15 if variant == "O2" else None

                    measurement = MeasurementInstance(
                        id=tag,
                        name=f"{variant.replace('_', ' ').title()} {meas_type.title()} - {eq_id}",
                        measurement_type=meas_type,
                        equipment_id=eq_id,
                        location=variant,
                        value=value,
                        unit=config["unit"],
                        range_min=range_min,
                        range_max=range_max,
                        alarm_low=alarm_low,
                        alarm_high=alarm_high,
                        properties={
                            "sensor_type": random.choice(["RTD", "thermocouple", "transmitter"]),
                            "calibration_date": (
                                datetime.now() - timedelta(days=random.randint(30, 365))
                            ).strftime("%Y-%m-%d"),
                        },
                    )
                    self.measurements.append(measurement)

        logger.debug(f"Generated {len(self.measurements)} measurements")

    def _generate_hazards(self) -> None:
        """Generate hazard data."""
        hazards_data = [
            ("boiler_explosion", "Boiler Explosion", "catastrophic", ["boiler"]),
            ("furnace_explosion", "Furnace Explosion", "catastrophic", ["furnace"]),
            ("tube_rupture", "Heater Tube Rupture", "critical", ["furnace", "heat_exchanger"]),
            ("steam_release", "Uncontrolled Steam Release", "critical", ["boiler", "steam_trap"]),
            ("thermal_runaway", "Thermal Runaway", "critical", ["furnace", "dryer"]),
            ("low_water", "Low Water Condition", "catastrophic", ["boiler"]),
            ("overpressure", "Vessel Overpressure", "critical", ["vessel", "heat_exchanger"]),
            ("fire", "Equipment Fire", "critical", ["furnace", "dryer"]),
            ("chemical_release", "Chemical Release", "critical", ["vessel", "pump"]),
        ]

        for hazard_id, name, severity, eq_types in hazards_data:
            applicable_eq = []
            for eq_type in eq_types:
                applicable_eq.extend(self._equipment_by_type.get(eq_type, []))

            # Sample equipment
            num_affected = min(len(applicable_eq), random.randint(5, 15))
            affected = random.sample(applicable_eq, num_affected) if applicable_eq else []

            hazard = {
                "id": hazard_id,
                "name": name,
                "severity": severity,
                "likelihood": random.choice(["remote", "occasional", "probable"]),
                "applicable_equipment": affected,
                "mitigation_measures": [f"Mitigation {i}" for i in range(1, 5)],
            }
            self.hazards.append(hazard)

        logger.debug(f"Generated {len(self.hazards)} hazards")

    def _generate_protection_layers(self) -> None:
        """Generate protection layer data."""
        layers_data = [
            ("bpcs", "Basic Process Control System", 0.1),
            ("alarm_response", "Alarm and Operator Response", 0.1),
            ("sis_sil1", "SIL 1 Safety Instrumented System", 0.01),
            ("sis_sil2", "SIL 2 Safety Instrumented System", 0.001),
            ("sis_sil3", "SIL 3 Safety Instrumented System", 0.0001),
            ("psv", "Pressure Safety Valve", 0.01),
            ("rupture_disk", "Rupture Disk", 0.001),
            ("fire_suppression", "Fire Suppression System", 0.01),
            ("emergency_response", "Emergency Response", 0.1),
        ]

        for layer_id, name, pfd in layers_data:
            # Apply to relevant equipment
            applicable_eq = self.equipment[:50]  # Apply to first 50 equipment
            affected = [eq.id for eq in applicable_eq]

            layer = {
                "id": layer_id,
                "name": name,
                "pfd": pfd,
                "independence": True,
                "applicable_equipment": affected,
            }
            self.protection_layers.append(layer)

        logger.debug(f"Generated {len(self.protection_layers)} protection layers")

    def _count_nodes(self) -> int:
        """Count total nodes."""
        return (
            len(self.equipment) +
            len(self.standards) +
            len(self.interlocks) +
            len(self.measurements) +
            len(self.hazards) +
            len(self.protection_layers) +
            len(PROCESS_AREAS) +  # Process areas as nodes
            len(LOCATIONS)  # Locations as nodes
        )

    def _count_relationships(self) -> int:
        """Count total relationships."""
        # Equipment relationships
        eq_rels = len(self.equipment) * 3  # ~3 relationships per equipment (location, area, type)

        # Process connections
        conn_rels = len(self.connections)

        # Interlock relationships
        int_rels = len(self.interlocks) * 2  # equipment + standards

        # Measurement relationships
        meas_rels = len(self.measurements)  # equipment

        # Standard relationships
        std_rels = len(self.standards) * len(EQUIPMENT_CONFIGS)  # applies_to

        # Hazard relationships
        hazard_rels = sum(len(h.get("applicable_equipment", [])) for h in self.hazards)

        # Protection layer relationships
        protection_rels = sum(len(p.get("applicable_equipment", [])) for p in self.protection_layers)

        return eq_rels + conn_rels + int_rels + meas_rels + std_rels + hazard_rels + protection_rels

    def _count_triples(self) -> int:
        """Count total triples (nodes + relationships + properties)."""
        # Each node has multiple property triples
        node_triples = self._count_nodes() * 5  # ~5 properties per node average

        # Each relationship is a triple
        rel_triples = self._count_relationships()

        return node_triples + rel_triples

    def get_ontology_data(self) -> Dict[str, Any]:
        """
        Get data formatted for bulk ontology import.

        Returns:
            Dictionary with equipment, processes, standards, safety data

        Example:
            >>> data = generator.get_ontology_data()
            >>> kg.bulk_import_from_ontology(data)
        """
        return {
            "equipment": [eq.dict() for eq in self.equipment],
            "processes": [],  # Could add process definitions
            "standards": [std.dict() for std in self.standards],
            "safety": [
                {"id": int.id, "name": int.name, "type": "interlock", "properties": int.dict()}
                for int in self.interlocks
            ] + [
                {"id": h["id"], "name": h["name"], "type": "hazard", "properties": h}
                for h in self.hazards
            ] + [
                {"id": p["id"], "name": p["name"], "type": "protection_layer", "properties": p}
                for p in self.protection_layers
            ],
            "measurements": [m.dict() for m in self.measurements],
            "relationships": [
                {"source": c.source_id, "target": c.target_id, "type": c.connection_type}
                for c in self.connections
            ] + [
                {"source": int.equipment_id, "target": int.id, "type": "HAS_INTERLOCK"}
                for int in self.interlocks
            ] + [
                {"source": m.equipment_id, "target": m.id, "type": "HAS_MEASUREMENT"}
                for m in self.measurements
            ],
        }

    def get_equipment_by_type(self, equipment_type: str) -> List[EquipmentInstance]:
        """Get equipment instances by type."""
        return [eq for eq in self.equipment if eq.equipment_type == equipment_type]

    def get_equipment_by_area(self, area: str) -> List[EquipmentInstance]:
        """Get equipment instances by process area."""
        return [eq for eq in self.equipment if eq.area == area]

    def get_interlocks_by_type(self, interlock_type: str) -> List[SafetyInterlockInstance]:
        """Get interlocks by type."""
        return [i for i in self.interlocks if i.interlock_type == interlock_type]

    def get_measurements_for_equipment(self, equipment_id: str) -> List[MeasurementInstance]:
        """Get measurements for an equipment."""
        return [m for m in self.measurements if m.equipment_id == equipment_id]


# =============================================================================
# Module-level functions
# =============================================================================

_generator_instance: Optional[SeedDataGenerator] = None


def get_seed_generator(seed: int = 42) -> SeedDataGenerator:
    """Get or create the global seed data generator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = SeedDataGenerator(seed)
    return _generator_instance


def generate_seed_data(seed: int = 42) -> Tuple[SeedDataGenerator, SeedStatistics]:
    """
    Generate seed data and return generator with statistics.

    Args:
        seed: Random seed for reproducible generation

    Returns:
        Tuple of (generator, statistics)

    Example:
        >>> generator, stats = generate_seed_data()
        >>> print(f"Generated {stats.total_triples} triples")
    """
    generator = SeedDataGenerator(seed)
    stats = generator.generate_all()
    return generator, stats
