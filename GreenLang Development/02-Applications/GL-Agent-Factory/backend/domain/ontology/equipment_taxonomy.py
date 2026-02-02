# -*- coding: utf-8 -*-
"""
Equipment Taxonomy for Process Heat Systems
============================================

Comprehensive taxonomy of industrial process heat equipment
with hierarchical classification, attributes, and relationships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class EquipmentCategory(str, Enum):
    """Top-level equipment categories."""
    HEAT_GENERATION = "heat_generation"
    HEAT_TRANSFER = "heat_transfer"
    HEAT_RECOVERY = "heat_recovery"
    STEAM_SYSTEM = "steam_system"
    COMBUSTION = "combustion"
    DRYING = "drying"
    THERMAL_PROCESSING = "thermal_processing"
    INSTRUMENTATION = "instrumentation"


class EquipmentStatus(str, Enum):
    """Equipment operational status."""
    OPERATIONAL = "operational"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    OUT_OF_SERVICE = "out_of_service"
    DECOMMISSIONED = "decommissioned"


class FuelType(str, Enum):
    """Types of fuel for combustion equipment."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    BIOMASS = "biomass"
    HYDROGEN = "hydrogen"
    BIOGAS = "biogas"
    PROPANE = "propane"
    ELECTRICITY = "electricity"
    WASTE_HEAT = "waste_heat"
    SOLAR_THERMAL = "solar_thermal"


@dataclass
class EquipmentAttribute:
    """Attribute definition for equipment class."""
    name: str
    description: str
    data_type: str  # string, float, int, bool, enum
    unit: Optional[str] = None
    required: bool = False
    default_value: Any = None
    valid_range: Optional[tuple] = None
    enum_values: Optional[List[str]] = None


@dataclass
class EquipmentRelation:
    """Relationship between equipment classes."""
    name: str
    description: str
    source_class: str
    target_class: str
    cardinality: str  # "1:1", "1:N", "N:N"
    is_required: bool = False
    inverse_name: Optional[str] = None


@dataclass
class EquipmentClass:
    """
    Equipment class definition in the taxonomy.

    Attributes:
        id: Unique identifier
        name: Human-readable name
        category: Equipment category
        description: Detailed description
        parent_class: Parent class ID (for hierarchy)
        attributes: List of equipment attributes
        applicable_standards: List of applicable standards
        typical_efficiency_range: Typical efficiency range (min, max)
        typical_temperature_range: Operating temperature range (min, max) in Celsius
        fuel_types: Applicable fuel types
    """
    id: str
    name: str
    category: EquipmentCategory
    description: str
    parent_class: Optional[str] = None
    attributes: List[EquipmentAttribute] = field(default_factory=list)
    applicable_standards: List[str] = field(default_factory=list)
    typical_efficiency_range: Optional[tuple] = None
    typical_temperature_range: Optional[tuple] = None
    fuel_types: List[FuelType] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


# =============================================================================
# Equipment Attribute Definitions
# =============================================================================

COMMON_ATTRIBUTES = [
    EquipmentAttribute("tag", "Equipment tag/identifier", "string", required=True),
    EquipmentAttribute("name", "Equipment name", "string", required=True),
    EquipmentAttribute("manufacturer", "Equipment manufacturer", "string"),
    EquipmentAttribute("model", "Model number", "string"),
    EquipmentAttribute("serial_number", "Serial number", "string"),
    EquipmentAttribute("installation_date", "Installation date", "date"),
    EquipmentAttribute("design_life_years", "Design life in years", "int", valid_range=(1, 100)),
    EquipmentAttribute("status", "Operational status", "enum", enum_values=[s.value for s in EquipmentStatus]),
]

THERMAL_ATTRIBUTES = [
    EquipmentAttribute("design_temperature_c", "Design temperature", "float", "degC"),
    EquipmentAttribute("max_temperature_c", "Maximum operating temperature", "float", "degC"),
    EquipmentAttribute("design_pressure_bar", "Design pressure", "float", "bar"),
    EquipmentAttribute("max_pressure_bar", "Maximum operating pressure", "float", "bar"),
    EquipmentAttribute("thermal_capacity_kw", "Thermal capacity", "float", "kW"),
    EquipmentAttribute("rated_efficiency", "Rated efficiency", "float", "%", valid_range=(0, 100)),
]

COMBUSTION_ATTRIBUTES = [
    EquipmentAttribute("fuel_type", "Primary fuel type", "enum", enum_values=[f.value for f in FuelType]),
    EquipmentAttribute("burner_type", "Burner type", "string"),
    EquipmentAttribute("max_firing_rate_kw", "Maximum firing rate", "float", "kW"),
    EquipmentAttribute("turndown_ratio", "Turndown ratio", "float"),
    EquipmentAttribute("nox_emissions_mg_nm3", "NOx emissions at 3% O2", "float", "mg/Nm3"),
    EquipmentAttribute("co_emissions_ppm", "CO emissions", "float", "ppm"),
]


# =============================================================================
# Equipment Taxonomy Definition
# =============================================================================

class EquipmentTaxonomy:
    """
    Complete equipment taxonomy for process heat systems.

    Provides hierarchical classification of equipment types
    with associated attributes, standards, and relationships.
    """

    def __init__(self):
        self.classes: Dict[str, EquipmentClass] = {}
        self.relations: Dict[str, EquipmentRelation] = {}
        self._initialize_taxonomy()

    def _initialize_taxonomy(self):
        """Initialize the equipment taxonomy."""
        self._create_boiler_classes()
        self._create_furnace_classes()
        self._create_heat_exchanger_classes()
        self._create_steam_system_classes()
        self._create_heat_recovery_classes()
        self._create_dryer_classes()
        self._create_relations()

    def _create_boiler_classes(self):
        """Create boiler equipment classes."""
        # Root boiler class
        self.classes["boiler"] = EquipmentClass(
            id="boiler",
            name="Boiler",
            category=EquipmentCategory.HEAT_GENERATION,
            description="Vessel for generating steam or hot water by combustion or electric heating",
            attributes=COMMON_ATTRIBUTES + THERMAL_ATTRIBUTES + COMBUSTION_ATTRIBUTES + [
                EquipmentAttribute("steam_capacity_tph", "Steam generation capacity", "float", "t/h"),
                EquipmentAttribute("steam_pressure_bar", "Steam pressure", "float", "bar"),
                EquipmentAttribute("steam_temperature_c", "Steam temperature", "float", "degC"),
                EquipmentAttribute("feedwater_temperature_c", "Feedwater temperature", "float", "degC"),
                EquipmentAttribute("blowdown_rate_pct", "Blowdown rate", "float", "%"),
            ],
            applicable_standards=["ASME BPVC Section I", "NFPA 85", "EN 12952", "EN 12953"],
            typical_efficiency_range=(80, 95),
            typical_temperature_range=(100, 540),
            fuel_types=[FuelType.NATURAL_GAS, FuelType.FUEL_OIL, FuelType.BIOMASS, FuelType.ELECTRICITY],
        )

        # Boiler subtypes
        boiler_subtypes = [
            ("fire_tube_boiler", "Fire Tube Boiler", "Boiler with combustion gases flowing inside tubes", (78, 88)),
            ("water_tube_boiler", "Water Tube Boiler", "Boiler with water flowing inside tubes", (82, 92)),
            ("electric_boiler", "Electric Boiler", "Boiler using electrical resistance heating", (95, 99)),
            ("waste_heat_boiler", "Waste Heat Boiler", "Boiler recovering heat from process exhaust", (70, 85)),
            ("hrsg", "Heat Recovery Steam Generator", "Boiler recovering heat from gas turbine exhaust", (75, 90)),
            ("once_through_boiler", "Once-Through Boiler", "Boiler without steam drum", (85, 94)),
        ]

        for subtype_id, name, description, eff_range in boiler_subtypes:
            self.classes[subtype_id] = EquipmentClass(
                id=subtype_id,
                name=name,
                category=EquipmentCategory.HEAT_GENERATION,
                description=description,
                parent_class="boiler",
                attributes=self.classes["boiler"].attributes.copy(),
                applicable_standards=self.classes["boiler"].applicable_standards.copy(),
                typical_efficiency_range=eff_range,
            )
            self.classes["boiler"].children.append(subtype_id)

    def _create_furnace_classes(self):
        """Create furnace equipment classes."""
        self.classes["furnace"] = EquipmentClass(
            id="furnace",
            name="Furnace",
            category=EquipmentCategory.THERMAL_PROCESSING,
            description="Enclosed structure for high-temperature heating of materials or fluids",
            attributes=COMMON_ATTRIBUTES + THERMAL_ATTRIBUTES + COMBUSTION_ATTRIBUTES + [
                EquipmentAttribute("furnace_volume_m3", "Furnace volume", "float", "m3"),
                EquipmentAttribute("heat_flux_kw_m2", "Heat flux", "float", "kW/m2"),
                EquipmentAttribute("residence_time_s", "Residence time", "float", "s"),
                EquipmentAttribute("atmosphere", "Furnace atmosphere", "string"),
            ],
            applicable_standards=["NFPA 86", "API 560", "EN 746"],
            typical_efficiency_range=(60, 85),
            typical_temperature_range=(200, 1200),
        )

        furnace_subtypes = [
            ("process_furnace", "Process Furnace", "Furnace for heating process fluids", (70, 90)),
            ("fired_heater", "Fired Heater", "Direct-fired heater for process streams", (75, 92)),
            ("reheat_furnace", "Reheat Furnace", "Furnace for reheating steel products", (50, 70)),
            ("annealing_furnace", "Annealing Furnace", "Furnace for heat treatment", (60, 80)),
            ("melting_furnace", "Melting Furnace", "Furnace for melting metals", (40, 70)),
            ("calcining_furnace", "Calcining Furnace", "Furnace for calcination reactions", (50, 75)),
        ]

        for subtype_id, name, description, eff_range in furnace_subtypes:
            self.classes[subtype_id] = EquipmentClass(
                id=subtype_id,
                name=name,
                category=EquipmentCategory.THERMAL_PROCESSING,
                description=description,
                parent_class="furnace",
                applicable_standards=self.classes["furnace"].applicable_standards.copy(),
                typical_efficiency_range=eff_range,
            )
            self.classes["furnace"].children.append(subtype_id)

    def _create_heat_exchanger_classes(self):
        """Create heat exchanger equipment classes."""
        self.classes["heat_exchanger"] = EquipmentClass(
            id="heat_exchanger",
            name="Heat Exchanger",
            category=EquipmentCategory.HEAT_TRANSFER,
            description="Device for transferring heat between two fluids",
            attributes=COMMON_ATTRIBUTES + [
                EquipmentAttribute("heat_duty_kw", "Heat duty", "float", "kW"),
                EquipmentAttribute("lmtd_c", "Log mean temperature difference", "float", "degC"),
                EquipmentAttribute("overall_htc_w_m2k", "Overall heat transfer coefficient", "float", "W/m2K"),
                EquipmentAttribute("heat_transfer_area_m2", "Heat transfer area", "float", "m2"),
                EquipmentAttribute("hot_side_inlet_temp_c", "Hot side inlet temperature", "float", "degC"),
                EquipmentAttribute("hot_side_outlet_temp_c", "Hot side outlet temperature", "float", "degC"),
                EquipmentAttribute("cold_side_inlet_temp_c", "Cold side inlet temperature", "float", "degC"),
                EquipmentAttribute("cold_side_outlet_temp_c", "Cold side outlet temperature", "float", "degC"),
                EquipmentAttribute("fouling_factor_m2k_w", "Fouling factor", "float", "m2K/W"),
            ],
            applicable_standards=["TEMA", "ASME BPVC Section VIII", "API 660", "API 661"],
            typical_efficiency_range=(85, 98),
        )

        hx_subtypes = [
            ("shell_tube_hx", "Shell and Tube Heat Exchanger", "HX with shell and tube bundle"),
            ("plate_hx", "Plate Heat Exchanger", "HX with corrugated plates"),
            ("air_cooled_hx", "Air Cooled Heat Exchanger", "HX with air-side cooling"),
            ("double_pipe_hx", "Double Pipe Heat Exchanger", "HX with concentric pipes"),
            ("spiral_hx", "Spiral Heat Exchanger", "HX with spiral flow paths"),
            ("printed_circuit_hx", "Printed Circuit Heat Exchanger", "Compact diffusion-bonded HX"),
        ]

        for subtype_id, name, description in hx_subtypes:
            self.classes[subtype_id] = EquipmentClass(
                id=subtype_id,
                name=name,
                category=EquipmentCategory.HEAT_TRANSFER,
                description=description,
                parent_class="heat_exchanger",
            )
            self.classes["heat_exchanger"].children.append(subtype_id)

    def _create_steam_system_classes(self):
        """Create steam system equipment classes."""
        steam_equipment = [
            ("steam_trap", "Steam Trap", "Device for discharging condensate while retaining steam",
             ["ASME PTC 39", "ISO 6552"], [
                 EquipmentAttribute("trap_type", "Steam trap type", "enum",
                                    enum_values=["mechanical", "thermodynamic", "thermostatic"]),
                 EquipmentAttribute("max_pressure_bar", "Maximum pressure", "float", "bar"),
                 EquipmentAttribute("max_backpressure_pct", "Maximum backpressure", "float", "%"),
                 EquipmentAttribute("condensate_capacity_kg_h", "Condensate capacity", "float", "kg/h"),
             ]),
            ("deaerator", "Deaerator", "Device for removing dissolved gases from feedwater",
             ["ASME PTC 12.4"], [
                 EquipmentAttribute("operating_pressure_bar", "Operating pressure", "float", "bar"),
                 EquipmentAttribute("storage_capacity_m3", "Storage capacity", "float", "m3"),
                 EquipmentAttribute("oxygen_guarantee_ppb", "Oxygen guarantee", "float", "ppb"),
             ]),
            ("feedwater_pump", "Feedwater Pump", "Pump for boiler feedwater",
             ["API 610"], [
                 EquipmentAttribute("flow_rate_m3_h", "Flow rate", "float", "m3/h"),
                 EquipmentAttribute("discharge_pressure_bar", "Discharge pressure", "float", "bar"),
                 EquipmentAttribute("pump_efficiency_pct", "Pump efficiency", "float", "%"),
             ]),
            ("prv", "Pressure Reducing Valve", "Valve for reducing steam pressure",
             ["ISA 75", "IEC 60534"], [
                 EquipmentAttribute("inlet_pressure_bar", "Inlet pressure", "float", "bar"),
                 EquipmentAttribute("outlet_pressure_bar", "Outlet pressure", "float", "bar"),
                 EquipmentAttribute("cv_value", "Cv value", "float"),
             ]),
        ]

        for eq_id, name, description, standards, attrs in steam_equipment:
            self.classes[eq_id] = EquipmentClass(
                id=eq_id,
                name=name,
                category=EquipmentCategory.STEAM_SYSTEM,
                description=description,
                attributes=COMMON_ATTRIBUTES + attrs,
                applicable_standards=standards,
            )

    def _create_heat_recovery_classes(self):
        """Create heat recovery equipment classes."""
        hr_equipment = [
            ("economizer", "Economizer", "Device for preheating boiler feedwater using flue gas",
             ["ASME PTC 4"], (5, 15)),
            ("air_preheater", "Air Preheater", "Device for preheating combustion air",
             ["ASME PTC 4.3"], (2, 8)),
            ("heat_pipe", "Heat Pipe", "Passive heat transfer device using phase change",
             [], (3, 10)),
            ("rotary_regenerator", "Rotary Regenerator", "Rotating heat storage device",
             ["ASME PTC 4.3"], (5, 15)),
            ("run_around_coil", "Run-Around Coil", "Indirect heat recovery using intermediate fluid",
             [], (3, 8)),
        ]

        for eq_id, name, description, standards, eff_improvement in hr_equipment:
            self.classes[eq_id] = EquipmentClass(
                id=eq_id,
                name=name,
                category=EquipmentCategory.HEAT_RECOVERY,
                description=description,
                applicable_standards=standards,
                attributes=COMMON_ATTRIBUTES + [
                    EquipmentAttribute("heat_recovery_kw", "Heat recovery", "float", "kW"),
                    EquipmentAttribute("efficiency_improvement_pct", "Efficiency improvement", "float", "%",
                                       valid_range=eff_improvement),
                ],
            )

    def _create_dryer_classes(self):
        """Create dryer equipment classes."""
        self.classes["dryer"] = EquipmentClass(
            id="dryer",
            name="Industrial Dryer",
            category=EquipmentCategory.DRYING,
            description="Equipment for removing moisture from materials",
            attributes=COMMON_ATTRIBUTES + THERMAL_ATTRIBUTES + [
                EquipmentAttribute("evaporation_capacity_kg_h", "Evaporation capacity", "float", "kg/h"),
                EquipmentAttribute("inlet_moisture_pct", "Inlet moisture content", "float", "%"),
                EquipmentAttribute("outlet_moisture_pct", "Outlet moisture content", "float", "%"),
                EquipmentAttribute("air_temperature_c", "Drying air temperature", "float", "degC"),
                EquipmentAttribute("specific_energy_kj_kg", "Specific energy consumption", "float", "kJ/kg water"),
            ],
            applicable_standards=["NFPA 86"],
            typical_efficiency_range=(50, 80),
        )

        dryer_subtypes = [
            ("rotary_dryer", "Rotary Dryer", "Dryer with rotating drum"),
            ("fluid_bed_dryer", "Fluid Bed Dryer", "Dryer with fluidized particle bed"),
            ("spray_dryer", "Spray Dryer", "Dryer using spray atomization"),
            ("conveyor_dryer", "Conveyor Dryer", "Dryer with belt conveyor"),
            ("flash_dryer", "Flash Dryer", "Dryer with pneumatic conveying"),
            ("tray_dryer", "Tray Dryer", "Batch dryer with trays"),
        ]

        for subtype_id, name, description in dryer_subtypes:
            self.classes[subtype_id] = EquipmentClass(
                id=subtype_id,
                name=name,
                category=EquipmentCategory.DRYING,
                description=description,
                parent_class="dryer",
            )
            self.classes["dryer"].children.append(subtype_id)

    def _create_relations(self):
        """Create equipment relationships."""
        relations = [
            EquipmentRelation(
                name="feeds_steam_to",
                description="Equipment feeds steam to another equipment",
                source_class="boiler",
                target_class="*",
                cardinality="1:N",
                inverse_name="receives_steam_from",
            ),
            EquipmentRelation(
                name="has_heat_recovery",
                description="Equipment has associated heat recovery device",
                source_class="boiler",
                target_class="economizer",
                cardinality="1:N",
            ),
            EquipmentRelation(
                name="preheats_air_for",
                description="Air preheater serves combustion equipment",
                source_class="air_preheater",
                target_class="furnace",
                cardinality="N:1",
            ),
            EquipmentRelation(
                name="discharges_to",
                description="Steam trap discharges condensate to",
                source_class="steam_trap",
                target_class="*",
                cardinality="N:1",
            ),
        ]

        for rel in relations:
            self.relations[rel.name] = rel

    def get_class(self, class_id: str) -> Optional[EquipmentClass]:
        """Get equipment class by ID."""
        return self.classes.get(class_id)

    def get_children(self, class_id: str) -> List[EquipmentClass]:
        """Get child classes of a parent class."""
        parent = self.classes.get(class_id)
        if parent:
            return [self.classes[child_id] for child_id in parent.children if child_id in self.classes]
        return []

    def get_ancestors(self, class_id: str) -> List[EquipmentClass]:
        """Get all ancestor classes."""
        ancestors = []
        current = self.classes.get(class_id)
        while current and current.parent_class:
            parent = self.classes.get(current.parent_class)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors

    def get_all_leaf_classes(self) -> List[EquipmentClass]:
        """Get all equipment classes without children."""
        return [cls for cls in self.classes.values() if not cls.children]

    def get_by_category(self, category: EquipmentCategory) -> List[EquipmentClass]:
        """Get all equipment classes in a category."""
        return [cls for cls in self.classes.values() if cls.category == category]

    def get_by_standard(self, standard: str) -> List[EquipmentClass]:
        """Get equipment classes that reference a specific standard."""
        return [
            cls for cls in self.classes.values()
            if any(standard.lower() in s.lower() for s in cls.applicable_standards)
        ]

    def search_classes(self, query: str) -> List[EquipmentClass]:
        """Search equipment classes by name or description."""
        query_lower = query.lower()
        return [
            cls for cls in self.classes.values()
            if query_lower in cls.name.lower() or query_lower in cls.description.lower()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get taxonomy statistics."""
        return {
            "total_classes": len(self.classes),
            "total_relations": len(self.relations),
            "leaf_classes": len(self.get_all_leaf_classes()),
            "categories": {cat.value: len(self.get_by_category(cat)) for cat in EquipmentCategory},
        }


# Module-level singleton
_taxonomy_instance: Optional[EquipmentTaxonomy] = None

def get_taxonomy() -> EquipmentTaxonomy:
    """Get or create the global taxonomy instance."""
    global _taxonomy_instance
    if _taxonomy_instance is None:
        _taxonomy_instance = EquipmentTaxonomy()
    return _taxonomy_instance
