# -*- coding: utf-8 -*-
"""
ProcessEnergyMappingEngine - PACK-031 Industrial Energy Audit Engine 3
=======================================================================

Maps energy flows through industrial processes, identifies losses,
calculates process efficiency, and scores optimisation opportunities.
Produces Sankey diagram data for energy flow visualisation, energy
balance per production line, loss quantification by type, process
heat cascade analysis, and energy intensity per product unit.

Process Analysis Methods:
    - Energy flow mapping with Sankey-ready data output
    - Unit operation efficiency calculation
    - Energy balance per production line (inputs = outputs + losses)
    - Loss identification by type (thermal, mechanical, electrical, conversion)
    - Pinch analysis foundation for heat cascade optimisation
    - Energy intensity metrics (kWh/unit, kWh/kg, kWh/tonne)

Standards Alignment:
    - ISO 50001:2018 Clause 6.3 (Energy review)
    - EN 16247-3:2022 (Energy audits - Processes)
    - EU BAT Reference Documents (BREF) for process industries
    - IEA Industrial Energy Efficiency benchmarks

Temperature Grades (for heat recovery):
    - High Grade: > 400 C (process furnaces, kilns)
    - Medium Grade: 100-400 C (steam, exhaust gases)
    - Low Grade: 40-100 C (cooling water, warm air)
    - Ambient: < 40 C (generally not recoverable)

Zero-Hallucination:
    - All efficiency calculations use deterministic Decimal arithmetic
    - Process benchmarks from published BREF / IEA data
    - Loss quantification via energy balance (inputs - outputs)
    - SHA-256 provenance hashing on every result
    - No LLM involvement in any numeric calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-031 Industrial Energy Audit
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float."""
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))


def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _round1(value: float) -> float:
    """Round to 1 decimal place using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProcessType(str, Enum):
    """Industrial process types for energy mapping.

    Covers major unit operations in manufacturing and process industries.
    """
    HEATING_FURNACE = "heating_furnace"
    BOILER = "boiler"
    KILN = "kiln"
    OVEN = "oven"
    DRYER = "dryer"
    HEAT_EXCHANGER = "heat_exchanger"
    DISTILLATION = "distillation"
    EVAPORATOR = "evaporator"
    REACTOR = "reactor"
    COMPRESSOR = "compressor"
    PUMP = "pump"
    FAN = "fan"
    MOTOR_DRIVE = "motor_drive"
    CONVEYOR = "conveyor"
    CRUSHER_GRINDER = "crusher_grinder"
    MIXER = "mixer"
    CHILLER = "chiller"
    COOLING_TOWER = "cooling_tower"
    TRANSFORMER = "transformer"
    RECTIFIER = "rectifier"
    WELDING = "welding"
    ELECTROLYSIS = "electrolysis"
    COMPRESSED_AIR_SYSTEM = "compressed_air_system"
    STEAM_SYSTEM = "steam_system"
    LIGHTING_SYSTEM = "lighting_system"
    HVAC_SYSTEM = "hvac_system"
    PACKAGING = "packaging"
    ASSEMBLY = "assembly"
    OTHER = "other"


class EnergyType(str, Enum):
    """Energy types flowing between process nodes.

    Classifies energy by form for proper loss categorisation.
    """
    ELECTRICAL = "electrical"
    THERMAL_STEAM = "thermal_steam"
    THERMAL_HOT_WATER = "thermal_hot_water"
    THERMAL_HOT_AIR = "thermal_hot_air"
    THERMAL_RADIATION = "thermal_radiation"
    MECHANICAL = "mechanical"
    CHEMICAL = "chemical"
    COMPRESSED_AIR = "compressed_air"
    FUEL_GAS = "fuel_gas"
    FUEL_OIL = "fuel_oil"
    COOLING = "cooling"
    PRODUCT_ENTHALPY = "product_enthalpy"
    WASTE_HEAT = "waste_heat"


class LossType(str, Enum):
    """Types of energy losses in industrial processes.

    Classification per EN 16247-3 and BREF documents.
    """
    THERMAL_RADIATION = "thermal_radiation"
    THERMAL_CONVECTION = "thermal_convection"
    THERMAL_CONDUCTION = "thermal_conduction"
    THERMAL_FLUE_GAS = "thermal_flue_gas"
    THERMAL_PRODUCT = "thermal_product"
    MECHANICAL_FRICTION = "mechanical_friction"
    MECHANICAL_VIBRATION = "mechanical_vibration"
    ELECTRICAL_TRANSFORMER = "electrical_transformer"
    ELECTRICAL_CABLE = "electrical_cable"
    ELECTRICAL_MOTOR = "electrical_motor"
    CONVERSION_INEFFICIENCY = "conversion_inefficiency"
    COMPRESSED_AIR_LEAKS = "compressed_air_leaks"
    STEAM_LEAKS = "steam_leaks"
    STEAM_TRAP_FAILURE = "steam_trap_failure"
    COOLING_TOWER_REJECTION = "cooling_tower_rejection"
    EXHAUST_GAS = "exhaust_gas"
    STANDBY_LOSSES = "standby_losses"
    PROCESS_WASTE = "process_waste"
    OTHER = "other"


class TemperatureGrade(str, Enum):
    """Temperature grades for heat recovery classification.

    Based on process integration / pinch analysis conventions.
    """
    HIGH = "high"          # > 400 C
    MEDIUM = "medium"      # 100-400 C
    LOW = "low"            # 40-100 C
    AMBIENT = "ambient"    # < 40 C


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Process efficiency benchmarks by process type.
# Values are typical, good practice, and best available technology (BAT).
# Sources: EU BREF documents, IEA Industrial Energy, Carbon Trust.
PROCESS_EFFICIENCY_BENCHMARKS: Dict[str, Dict[str, float]] = {
    ProcessType.HEATING_FURNACE: {
        "typical_pct": 40.0,
        "good_practice_pct": 55.0,
        "bat_pct": 70.0,
        "source": "EU BREF Smitheries and Foundries (2005), updated IEA 2022",
    },
    ProcessType.BOILER: {
        "typical_pct": 80.0,
        "good_practice_pct": 88.0,
        "bat_pct": 95.0,
        "source": "EU BREF Large Combustion Plants, ASME PTC 4",
    },
    ProcessType.KILN: {
        "typical_pct": 35.0,
        "good_practice_pct": 50.0,
        "bat_pct": 65.0,
        "source": "EU BREF Cement, Lime and Magnesium Oxide (2013)",
    },
    ProcessType.OVEN: {
        "typical_pct": 45.0,
        "good_practice_pct": 60.0,
        "bat_pct": 75.0,
        "source": "Carbon Trust Industrial Ovens guide",
    },
    ProcessType.DRYER: {
        "typical_pct": 35.0,
        "good_practice_pct": 50.0,
        "bat_pct": 65.0,
        "source": "EU BREF Food, Drink and Milk Industries (2019)",
    },
    ProcessType.HEAT_EXCHANGER: {
        "typical_pct": 70.0,
        "good_practice_pct": 82.0,
        "bat_pct": 92.0,
        "source": "TEMA Standards, HTRI design guidelines",
    },
    ProcessType.DISTILLATION: {
        "typical_pct": 25.0,
        "good_practice_pct": 40.0,
        "bat_pct": 55.0,
        "source": "EU BREF Organic Fine Chemicals (2006)",
    },
    ProcessType.EVAPORATOR: {
        "typical_pct": 50.0,
        "good_practice_pct": 65.0,
        "bat_pct": 80.0,
        "source": "EU BREF Food, Drink and Milk Industries (2019)",
    },
    ProcessType.REACTOR: {
        "typical_pct": 55.0,
        "good_practice_pct": 70.0,
        "bat_pct": 85.0,
        "source": "EU BREF Chemical sector (various)",
    },
    ProcessType.COMPRESSOR: {
        "typical_pct": 60.0,
        "good_practice_pct": 75.0,
        "bat_pct": 88.0,
        "source": "Compressed Air & Gas Institute, ISO 1217",
    },
    ProcessType.PUMP: {
        "typical_pct": 55.0,
        "good_practice_pct": 70.0,
        "bat_pct": 85.0,
        "source": "Hydraulic Institute / Europump guidelines",
    },
    ProcessType.FAN: {
        "typical_pct": 55.0,
        "good_practice_pct": 70.0,
        "bat_pct": 85.0,
        "source": "AMCA International, EN 327",
    },
    ProcessType.MOTOR_DRIVE: {
        "typical_pct": 85.0,
        "good_practice_pct": 92.0,
        "bat_pct": 96.0,
        "source": "EU Regulation 2019/1781 (IE3/IE4/IE5), IEC 60034-30-1",
    },
    ProcessType.CONVEYOR: {
        "typical_pct": 75.0,
        "good_practice_pct": 85.0,
        "bat_pct": 92.0,
        "source": "CEMA Engineering Conference papers",
    },
    ProcessType.CRUSHER_GRINDER: {
        "typical_pct": 5.0,
        "good_practice_pct": 10.0,
        "bat_pct": 15.0,
        "source": "Bond Work Index, EU BREF Mineral Industries",
    },
    ProcessType.CHILLER: {
        "typical_pct": 65.0,
        "good_practice_pct": 78.0,
        "bat_pct": 90.0,
        "source": "ASHRAE 90.1, ARI 550/590 (COP-based)",
    },
    ProcessType.COOLING_TOWER: {
        "typical_pct": 70.0,
        "good_practice_pct": 80.0,
        "bat_pct": 90.0,
        "source": "CTI (Cooling Technology Institute) standards",
    },
    ProcessType.TRANSFORMER: {
        "typical_pct": 96.0,
        "good_practice_pct": 98.0,
        "bat_pct": 99.5,
        "source": "EU Regulation 2019/1783 (Ecodesign transformers)",
    },
    ProcessType.COMPRESSED_AIR_SYSTEM: {
        "typical_pct": 10.0,
        "good_practice_pct": 18.0,
        "bat_pct": 25.0,
        "source": "Carbon Trust Compressed Air guide, only ~10% of input energy "
                  "does useful work in a typical compressed air system",
    },
    ProcessType.STEAM_SYSTEM: {
        "typical_pct": 55.0,
        "good_practice_pct": 70.0,
        "bat_pct": 82.0,
        "source": "US DOE Steam System Assessment Tool, EU BREF",
    },
    ProcessType.LIGHTING_SYSTEM: {
        "typical_pct": 20.0,
        "good_practice_pct": 45.0,
        "bat_pct": 65.0,
        "source": "IEA Lighting policy, EN 12464-1, LED vs fluorescent basis",
    },
    ProcessType.HVAC_SYSTEM: {
        "typical_pct": 60.0,
        "good_practice_pct": 75.0,
        "bat_pct": 88.0,
        "source": "ASHRAE 90.1, CIBSE Guide F",
    },
    ProcessType.OTHER: {
        "typical_pct": 50.0,
        "good_practice_pct": 65.0,
        "bat_pct": 80.0,
        "source": "Generic industrial average",
    },
}


# Thermal loss factors by surface type (W per m2 per degree C temperature
# difference above ambient).  Used for estimating radiation/convection losses.
# Source: CIBSE Guide C, EN ISO 12241 (thermal insulation).
THERMAL_LOSS_FACTORS: Dict[str, float] = {
    "uninsulated_pipe": 15.0,
    "insulated_pipe": 3.0,
    "uninsulated_vessel": 12.0,
    "insulated_vessel": 2.5,
    "uninsulated_duct": 10.0,
    "insulated_duct": 2.0,
    "furnace_wall_unlined": 25.0,
    "furnace_wall_refractory": 5.0,
    "boiler_shell": 4.0,
    "kiln_shell": 8.0,
    "oven_wall": 6.0,
    "default": 10.0,
}
"""Surface heat loss factors in W/m2/K.
Multiply by surface area (m2) and temperature difference (K) to get loss in watts."""


# Motor/drive efficiency standards by class (%).
# Source: IEC 60034-30-1:2014, EU Regulation 2019/1781.
MOTOR_DRIVE_EFFICIENCY_STANDARDS: Dict[str, Dict[str, float]] = {
    "IE1_standard": {
        "0.75kW": 72.1, "1.1kW": 75.0, "2.2kW": 79.6, "4kW": 82.6,
        "7.5kW": 85.7, "11kW": 87.6, "15kW": 88.7, "22kW": 89.6,
        "37kW": 90.9, "55kW": 91.8, "75kW": 92.4, "90kW": 92.7,
        "110kW": 93.0, "160kW": 93.5, "200kW": 93.8,
    },
    "IE2_high": {
        "0.75kW": 77.4, "1.1kW": 79.6, "2.2kW": 84.3, "4kW": 86.6,
        "7.5kW": 89.1, "11kW": 90.4, "15kW": 91.2, "22kW": 91.9,
        "37kW": 92.7, "55kW": 93.3, "75kW": 93.7, "90kW": 93.9,
        "110kW": 94.1, "160kW": 94.6, "200kW": 94.8,
    },
    "IE3_premium": {
        "0.75kW": 80.7, "1.1kW": 82.7, "2.2kW": 86.7, "4kW": 88.6,
        "7.5kW": 90.4, "11kW": 91.4, "15kW": 92.1, "22kW": 92.9,
        "37kW": 93.7, "55kW": 94.0, "75kW": 94.3, "90kW": 94.6,
        "110kW": 94.8, "160kW": 95.2, "200kW": 95.4,
    },
    "IE4_super_premium": {
        "0.75kW": 82.5, "1.1kW": 84.1, "2.2kW": 87.7, "4kW": 89.6,
        "7.5kW": 91.4, "11kW": 92.4, "15kW": 93.0, "22kW": 93.6,
        "37kW": 94.4, "55kW": 94.7, "75kW": 95.0, "90kW": 95.2,
        "110kW": 95.4, "160kW": 95.8, "200kW": 96.0,
    },
    "IE5_ultra_premium": {
        "0.75kW": 84.0, "1.1kW": 85.6, "2.2kW": 89.0, "4kW": 90.8,
        "7.5kW": 92.5, "11kW": 93.4, "15kW": 93.9, "22kW": 94.4,
        "37kW": 95.1, "55kW": 95.4, "75kW": 95.7, "90kW": 95.9,
        "110kW": 96.1, "160kW": 96.4, "200kW": 96.6,
    },
}


# Temperature grade boundaries (Celsius).
TEMPERATURE_GRADE_BOUNDARIES: Dict[str, Tuple[float, float]] = {
    TemperatureGrade.HIGH: (400.0, 2000.0),
    TemperatureGrade.MEDIUM: (100.0, 400.0),
    TemperatureGrade.LOW: (40.0, 100.0),
    TemperatureGrade.AMBIENT: (0.0, 40.0),
}


# Typical heat recovery potential by temperature grade (%).
HEAT_RECOVERY_POTENTIAL: Dict[str, float] = {
    TemperatureGrade.HIGH: 60.0,
    TemperatureGrade.MEDIUM: 50.0,
    TemperatureGrade.LOW: 30.0,
    TemperatureGrade.AMBIENT: 5.0,
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class ProcessNode(BaseModel):
    """A single process node (unit operation) in the energy flow map.

    Attributes:
        node_id: Unique identifier for this process node.
        name: Human-readable name.
        process_type: Type of process/equipment.
        input_energy_kwh: Total energy input to this node (kWh).
        output_energy_kwh: Useful energy output from this node (kWh).
        losses_kwh: Energy losses at this node (kWh).
        efficiency_pct: Process efficiency (output / input * 100).
        temperature_in_c: Input temperature (C) for thermal processes.
        temperature_out_c: Output temperature (C) for thermal processes.
        rated_capacity_kw: Rated capacity of equipment (kW).
        operating_hours: Annual operating hours.
        load_factor_pct: Average load as percentage of rated capacity.
    """
    node_id: str = Field(..., min_length=1, description="Node identifier")
    name: str = Field(..., min_length=1, description="Node name")
    process_type: ProcessType = Field(..., description="Process type")
    input_energy_kwh: float = Field(..., ge=0, description="Energy input (kWh)")
    output_energy_kwh: float = Field(
        default=0.0, ge=0, description="Useful output (kWh)"
    )
    losses_kwh: Optional[float] = Field(
        None, ge=0, description="Losses (kWh), auto-calculated if None"
    )
    efficiency_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Efficiency (%), auto-calculated if None"
    )
    temperature_in_c: Optional[float] = Field(
        None, description="Input temperature (C)"
    )
    temperature_out_c: Optional[float] = Field(
        None, description="Output temperature (C)"
    )
    rated_capacity_kw: Optional[float] = Field(
        None, ge=0, description="Rated capacity (kW)"
    )
    operating_hours: Optional[float] = Field(
        None, ge=0, le=8784, description="Operating hours/year"
    )
    load_factor_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Load factor (%)"
    )

    @model_validator(mode="after")
    def auto_calculate_fields(self) -> "ProcessNode":
        """Auto-calculate losses and efficiency if not provided."""
        inp = _decimal(self.input_energy_kwh)
        out = _decimal(self.output_energy_kwh)

        if self.losses_kwh is None:
            computed_loss = inp - out
            if computed_loss < Decimal("0"):
                computed_loss = Decimal("0")
            self.losses_kwh = float(computed_loss)

        if self.efficiency_pct is None and inp > Decimal("0"):
            eff = _safe_pct(out, inp)
            self.efficiency_pct = float(eff)
        elif self.efficiency_pct is None:
            self.efficiency_pct = 0.0

        return self


class EnergyFlow(BaseModel):
    """Energy flow between two process nodes.

    Attributes:
        source_node: ID of the source (upstream) node.
        target_node: ID of the target (downstream) node.
        energy_kwh: Energy transferred (kWh).
        energy_type: Type of energy being transferred.
        temperature_c: Temperature of the energy stream (C).
    """
    source_node: str = Field(..., min_length=1, description="Source node ID")
    target_node: str = Field(..., min_length=1, description="Target node ID")
    energy_kwh: float = Field(..., ge=0, description="Energy flow (kWh)")
    energy_type: EnergyType = Field(..., description="Energy type")
    temperature_c: Optional[float] = Field(
        None, description="Stream temperature (C)"
    )


class ProductionLine(BaseModel):
    """A production line composed of process nodes.

    Attributes:
        line_id: Unique line identifier.
        name: Production line name.
        nodes: List of process nodes in this line.
        total_input_kwh: Total energy input to the line.
        total_output_kwh: Total useful energy output.
        total_losses_kwh: Total losses across the line.
        line_efficiency_pct: Overall line efficiency.
        annual_production_units: Annual production volume.
        production_unit: Unit label (kg, tonne, pieces, etc.).
    """
    line_id: str = Field(..., min_length=1, description="Line identifier")
    name: str = Field(..., min_length=1, description="Line name")
    nodes: List[ProcessNode] = Field(
        default_factory=list, description="Process nodes"
    )
    total_input_kwh: Optional[float] = Field(
        None, ge=0, description="Total input (kWh)"
    )
    total_output_kwh: Optional[float] = Field(
        None, ge=0, description="Total output (kWh)"
    )
    total_losses_kwh: Optional[float] = Field(
        None, ge=0, description="Total losses (kWh)"
    )
    line_efficiency_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Line efficiency (%)"
    )
    annual_production_units: Optional[float] = Field(
        None, ge=0, description="Annual production (units)"
    )
    production_unit: str = Field(
        default="unit", description="Production unit label"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class SankeyNode(BaseModel):
    """Sankey diagram node for visualisation."""
    node_id: str = Field(..., description="Node identifier")
    name: str = Field(default="", description="Display name")
    value_kwh: float = Field(default=0.0, description="Energy value (kWh)")
    node_type: str = Field(default="process", description="Type: source/process/loss/output")


class SankeyLink(BaseModel):
    """Sankey diagram link (flow) for visualisation."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    value_kwh: float = Field(default=0.0, description="Flow value (kWh)")
    energy_type: str = Field(default="", description="Energy type")


class SankeyData(BaseModel):
    """Complete Sankey diagram data for energy flow visualisation.

    Attributes:
        nodes: List of Sankey nodes.
        links: List of Sankey links (flows).
        total_input: Total energy entering the system (kWh).
        total_output: Total useful output (kWh).
        total_losses: Total losses (kWh).
    """
    nodes: List[SankeyNode] = Field(default_factory=list, description="Nodes")
    links: List[SankeyLink] = Field(default_factory=list, description="Links")
    total_input: float = Field(default=0.0, description="Total input (kWh)")
    total_output: float = Field(default=0.0, description="Total useful output (kWh)")
    total_losses: float = Field(default=0.0, description="Total losses (kWh)")


class LossBreakdown(BaseModel):
    """Detailed loss identification and quantification.

    Attributes:
        loss_type: Classification of the loss.
        location: Where in the process the loss occurs.
        node_id: Associated process node ID.
        quantity_kwh: Loss quantity (kWh).
        temperature_c: Temperature of the waste stream (C).
        temperature_grade: Temperature grade classification.
        recoverable: Whether the loss is potentially recoverable.
        recovery_potential_pct: Estimated recovery potential (%).
        recovery_potential_kwh: Estimated recoverable energy (kWh).
    """
    loss_type: str = Field(..., description="Loss type")
    location: str = Field(default="", description="Loss location")
    node_id: str = Field(default="", description="Process node ID")
    quantity_kwh: float = Field(default=0.0, description="Loss quantity (kWh)")
    temperature_c: Optional[float] = Field(None, description="Temperature (C)")
    temperature_grade: str = Field(default="", description="Temperature grade")
    recoverable: bool = Field(default=False, description="Recoverable flag")
    recovery_potential_pct: float = Field(
        default=0.0, description="Recovery potential (%)"
    )
    recovery_potential_kwh: float = Field(
        default=0.0, description="Recoverable energy (kWh)"
    )


class ProcessNodeResult(BaseModel):
    """Enriched process node result with benchmark comparison."""
    node_id: str = Field(..., description="Node identifier")
    name: str = Field(default="", description="Node name")
    process_type: str = Field(default="", description="Process type")
    input_energy_kwh: float = Field(default=0.0, description="Input (kWh)")
    output_energy_kwh: float = Field(default=0.0, description="Output (kWh)")
    losses_kwh: float = Field(default=0.0, description="Losses (kWh)")
    efficiency_pct: float = Field(default=0.0, description="Efficiency (%)")
    benchmark_typical_pct: Optional[float] = Field(
        None, description="Typical benchmark efficiency"
    )
    benchmark_good_pct: Optional[float] = Field(
        None, description="Good practice benchmark"
    )
    benchmark_bat_pct: Optional[float] = Field(
        None, description="BAT benchmark"
    )
    gap_vs_bat_pct: float = Field(
        default=0.0, description="Gap vs BAT (%)"
    )
    performance_label: str = Field(default="", description="Performance label")
    improvement_potential_kwh: float = Field(
        default=0.0, description="Improvement potential (kWh)"
    )


class IntensityMetric(BaseModel):
    """Energy intensity metric per product or area."""
    metric_name: str = Field(..., description="Metric name")
    value: float = Field(default=0.0, description="Metric value")
    unit: str = Field(default="", description="Metric unit")
    line_id: Optional[str] = Field(None, description="Production line ID")


class OptimisationOpportunity(BaseModel):
    """Scored optimisation opportunity from process analysis.

    Attributes:
        opportunity_id: Unique identifier.
        description: Description of the opportunity.
        affected_node: Process node ID.
        current_efficiency_pct: Current efficiency.
        target_efficiency_pct: Target efficiency (BAT or good practice).
        savings_kwh: Estimated annual savings (kWh).
        savings_pct: Savings as percentage of node input.
        priority_score: Priority score (0-100, higher = more impactful).
        category: Opportunity category (thermal, mechanical, etc.).
    """
    opportunity_id: str = Field(default_factory=_new_uuid, description="ID")
    description: str = Field(default="", description="Description")
    affected_node: str = Field(default="", description="Node ID")
    current_efficiency_pct: float = Field(default=0.0, description="Current eff (%)")
    target_efficiency_pct: float = Field(default=0.0, description="Target eff (%)")
    savings_kwh: float = Field(default=0.0, description="Savings (kWh/year)")
    savings_pct: float = Field(default=0.0, description="Savings (%)")
    priority_score: float = Field(default=0.0, description="Priority (0-100)")
    category: str = Field(default="", description="Category")


class ProcessEnergyResult(BaseModel):
    """Complete process energy mapping result with full provenance.

    Contains production line analysis, Sankey diagram data, energy balance,
    loss breakdown, intensity metrics, and optimisation opportunities.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Calc timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")

    facility_id: str = Field(default="", description="Facility identifier")

    production_lines: List[Dict[str, Any]] = Field(
        default_factory=list, description="Production line summaries"
    )
    process_nodes: List[ProcessNodeResult] = Field(
        default_factory=list, description="Enriched process node results"
    )
    sankey_data: Optional[SankeyData] = Field(None, description="Sankey diagram data")

    total_input_kwh: float = Field(default=0.0, description="Total input (kWh)")
    total_output_kwh: float = Field(default=0.0, description="Total output (kWh)")
    total_losses_kwh: float = Field(default=0.0, description="Total losses (kWh)")
    overall_efficiency_pct: float = Field(default=0.0, description="Overall efficiency (%)")

    losses: List[LossBreakdown] = Field(
        default_factory=list, description="Loss breakdown"
    )
    total_recoverable_kwh: float = Field(
        default=0.0, description="Total recoverable energy (kWh)"
    )

    intensity_metrics: List[IntensityMetric] = Field(
        default_factory=list, description="Energy intensity metrics"
    )

    optimization_opportunities: List[OptimisationOpportunity] = Field(
        default_factory=list, description="Optimisation opportunities"
    )
    total_optimization_potential_kwh: float = Field(
        default=0.0, description="Total optimisation potential (kWh)"
    )

    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------


class ProcessEnergyMappingEngine:
    """Process energy flow mapping and optimisation engine.

    Provides deterministic, zero-hallucination calculations for:
    - Energy flow mapping with Sankey diagram data output
    - Process efficiency calculation per unit operation
    - Energy balance per production line (inputs = outputs + losses)
    - Loss identification and quantification by type
    - Temperature-based heat recovery opportunity assessment
    - Energy intensity per product unit (kWh/unit, kWh/kg, kWh/tonne)
    - Process optimisation opportunity scoring

    All calculations are bit-perfect reproducible. No LLM is used
    in any calculation path.

    Usage::

        engine = ProcessEnergyMappingEngine()
        result = engine.map_process_energy(
            facility_id="PLANT-001",
            production_lines=[line1, line2],
            energy_flows=[flow1, flow2, flow3],
        )
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise the process energy mapping engine."""
        self._benchmarks = PROCESS_EFFICIENCY_BENCHMARKS
        self._thermal_loss_factors = THERMAL_LOSS_FACTORS
        self._temp_grades = TEMPERATURE_GRADE_BOUNDARIES
        self._recovery_potential = HEAT_RECOVERY_POTENTIAL

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def map_process_energy(
        self,
        facility_id: str,
        production_lines: List[ProductionLine],
        energy_flows: Optional[List[EnergyFlow]] = None,
    ) -> ProcessEnergyResult:
        """Map energy flows through industrial processes.

        Analyses production lines, calculates efficiency per node,
        identifies losses, generates Sankey diagram data, computes
        intensity metrics, and scores optimisation opportunities.

        Args:
            facility_id: Facility identifier.
            production_lines: List of production lines with process nodes.
            energy_flows: Optional explicit energy flows between nodes.

        Returns:
            ProcessEnergyResult with complete analysis and provenance.

        Raises:
            ValueError: If production_lines is empty.
        """
        t0 = time.perf_counter()

        if not production_lines:
            raise ValueError("At least one ProductionLine is required")

        logger.info(
            "Mapping process energy for facility %s (%d lines)",
            facility_id, len(production_lines),
        )

        # Step 1: Enrich all nodes with benchmarks
        all_node_results: List[ProcessNodeResult] = []
        for line in production_lines:
            for node in line.nodes:
                node_result = self._enrich_node(node)
                all_node_results.append(node_result)

        # Step 2: Calculate production line summaries
        line_summaries = self._calculate_line_summaries(production_lines)

        # Step 3: Calculate overall totals
        total_input = Decimal("0")
        total_output = Decimal("0")
        total_losses = Decimal("0")
        for nr in all_node_results:
            total_input += _decimal(nr.input_energy_kwh)
            total_output += _decimal(nr.output_energy_kwh)
            total_losses += _decimal(nr.losses_kwh)

        overall_eff = _safe_pct(total_output, total_input)

        # Step 4: Build Sankey data
        sankey = self._build_sankey_data(
            all_node_results, energy_flows, total_input, total_output, total_losses,
        )

        # Step 5: Identify and quantify losses
        losses = self._identify_losses(all_node_results)
        total_recoverable = Decimal("0")
        for loss in losses:
            total_recoverable += _decimal(loss.recovery_potential_kwh)

        # Step 6: Calculate intensity metrics
        intensity_metrics = self._calculate_intensity_metrics(
            production_lines, total_input,
        )

        # Step 7: Score optimisation opportunities
        opportunities = self._score_optimisation_opportunities(all_node_results)
        total_opt_kwh = Decimal("0")
        for opp in opportunities:
            total_opt_kwh += _decimal(opp.savings_kwh)

        # Step 8: Recommendations
        recommendations = self._generate_recommendations(
            all_node_results, losses, opportunities, overall_eff,
            total_recoverable, total_input,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ProcessEnergyResult(
            facility_id=facility_id,
            production_lines=line_summaries,
            process_nodes=all_node_results,
            sankey_data=sankey,
            total_input_kwh=_round_val(total_input, 2),
            total_output_kwh=_round_val(total_output, 2),
            total_losses_kwh=_round_val(total_losses, 2),
            overall_efficiency_pct=_round_val(overall_eff, 1),
            losses=losses,
            total_recoverable_kwh=_round_val(total_recoverable, 2),
            intensity_metrics=intensity_metrics,
            optimization_opportunities=opportunities,
            total_optimization_potential_kwh=_round_val(total_opt_kwh, 2),
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def calculate_node_efficiency(
        self, node: ProcessNode,
    ) -> ProcessNodeResult:
        """Calculate efficiency for a single process node with benchmarking.

        Args:
            node: Process node data.

        Returns:
            ProcessNodeResult with efficiency and benchmark comparison.
        """
        return self._enrich_node(node)

    def calculate_line_efficiency(
        self, line: ProductionLine,
    ) -> Dict[str, Any]:
        """Calculate overall efficiency for a production line.

        Args:
            line: Production line data.

        Returns:
            Dict with line efficiency metrics and provenance hash.
        """
        summaries = self._calculate_line_summaries([line])
        if summaries:
            summary = summaries[0]
            summary["provenance_hash"] = _compute_hash(summary)
            return summary
        return {"error": "No line data", "provenance_hash": ""}

    def identify_waste_heat_streams(
        self, nodes: List[ProcessNode],
    ) -> List[LossBreakdown]:
        """Identify waste heat streams suitable for recovery.

        Args:
            nodes: List of process nodes to analyse.

        Returns:
            List of LossBreakdown for waste heat streams, sorted by
            recovery potential.
        """
        enriched = [self._enrich_node(n) for n in nodes]
        all_losses = self._identify_losses(enriched)
        # Filter to recoverable losses only
        recoverable = [l for l in all_losses if l.recoverable]
        recoverable.sort(key=lambda l: l.recovery_potential_kwh, reverse=True)
        return recoverable

    def calculate_energy_intensity(
        self,
        total_energy_kwh: float,
        production_quantity: float,
        production_unit: str = "tonne",
    ) -> Dict[str, Any]:
        """Calculate energy intensity per production unit.

        Args:
            total_energy_kwh: Total energy consumed (kWh).
            production_quantity: Total production (in production_unit).
            production_unit: Unit of production (kg, tonne, piece, etc.).

        Returns:
            Dict with intensity metrics and provenance hash.
        """
        energy = _decimal(total_energy_kwh)
        production = _decimal(production_quantity)
        intensity = _safe_divide(energy, production)

        result = {
            "total_energy_kwh": _round_val(energy, 2),
            "production_quantity": _round_val(production, 2),
            "production_unit": production_unit,
            "energy_intensity": _round_val(intensity, 4),
            "energy_intensity_unit": f"kWh/{production_unit}",
            "provenance_hash": _compute_hash({
                "energy": str(energy),
                "production": str(production),
            }),
        }

        # Add common conversions
        if production_unit == "kg":
            intensity_per_tonne = intensity * Decimal("1000")
            result["energy_intensity_per_tonne"] = _round_val(intensity_per_tonne, 2)
            result["energy_intensity_per_tonne_unit"] = "kWh/tonne"
        elif production_unit == "tonne":
            intensity_per_kg = _safe_divide(intensity, Decimal("1000"))
            result["energy_intensity_per_kg"] = _round_val(intensity_per_kg, 4)
            result["energy_intensity_per_kg_unit"] = "kWh/kg"

        return result

    # -------------------------------------------------------------------
    # Internal: Node Enrichment
    # -------------------------------------------------------------------

    def _enrich_node(self, node: ProcessNode) -> ProcessNodeResult:
        """Enrich a process node with benchmark comparison.

        Args:
            node: Raw process node data.

        Returns:
            ProcessNodeResult with benchmarks and gap analysis.
        """
        inp = _decimal(node.input_energy_kwh)
        out = _decimal(node.output_energy_kwh)
        losses = _decimal(node.losses_kwh or 0)
        eff = _decimal(node.efficiency_pct or 0)

        bench = self._benchmarks.get(node.process_type, self._benchmarks.get(ProcessType.OTHER, {}))
        typical = bench.get("typical_pct")
        good = bench.get("good_practice_pct")
        bat = bench.get("bat_pct")

        # Performance classification
        eff_float = float(eff)
        if bat is not None and eff_float >= bat:
            label = "BAT (Best Available Technology)"
        elif good is not None and eff_float >= good:
            label = "Good Practice"
        elif typical is not None and eff_float >= typical:
            label = "Typical"
        elif typical is not None:
            label = "Below Typical"
        else:
            label = "Unknown"

        # Gap vs BAT
        gap = Decimal("0")
        improvement_kwh = Decimal("0")
        if bat is not None:
            gap = _decimal(bat) - eff
            if gap > Decimal("0") and inp > Decimal("0"):
                # If node were at BAT efficiency, output would be higher
                # meaning less input needed for same output
                bat_input = _safe_divide(out, _decimal(bat) / Decimal("100"))
                improvement_kwh = inp - bat_input
                if improvement_kwh < Decimal("0"):
                    improvement_kwh = Decimal("0")

        return ProcessNodeResult(
            node_id=node.node_id,
            name=node.name,
            process_type=node.process_type.value,
            input_energy_kwh=_round_val(inp, 2),
            output_energy_kwh=_round_val(out, 2),
            losses_kwh=_round_val(losses, 2),
            efficiency_pct=_round_val(eff, 1),
            benchmark_typical_pct=typical,
            benchmark_good_pct=good,
            benchmark_bat_pct=bat,
            gap_vs_bat_pct=_round_val(gap, 1),
            performance_label=label,
            improvement_potential_kwh=_round_val(improvement_kwh, 2),
        )

    # -------------------------------------------------------------------
    # Internal: Line Summaries
    # -------------------------------------------------------------------

    def _calculate_line_summaries(
        self, lines: List[ProductionLine],
    ) -> List[Dict[str, Any]]:
        """Calculate summary metrics for each production line.

        Args:
            lines: List of production lines.

        Returns:
            List of line summary dicts.
        """
        summaries: List[Dict[str, Any]] = []

        for line in lines:
            total_in = Decimal("0")
            total_out = Decimal("0")
            total_loss = Decimal("0")

            for node in line.nodes:
                total_in += _decimal(node.input_energy_kwh)
                total_out += _decimal(node.output_energy_kwh)
                total_loss += _decimal(node.losses_kwh or 0)

            # Override with explicit values if provided
            if line.total_input_kwh is not None:
                total_in = _decimal(line.total_input_kwh)
            if line.total_output_kwh is not None:
                total_out = _decimal(line.total_output_kwh)
            if line.total_losses_kwh is not None:
                total_loss = _decimal(line.total_losses_kwh)

            line_eff = _safe_pct(total_out, total_in)

            # Energy intensity per production unit
            intensity = Decimal("0")
            if line.annual_production_units and line.annual_production_units > 0:
                intensity = _safe_divide(
                    total_in, _decimal(line.annual_production_units)
                )

            summaries.append({
                "line_id": line.line_id,
                "name": line.name,
                "node_count": len(line.nodes),
                "total_input_kwh": _round_val(total_in, 2),
                "total_output_kwh": _round_val(total_out, 2),
                "total_losses_kwh": _round_val(total_loss, 2),
                "line_efficiency_pct": _round_val(line_eff, 1),
                "energy_intensity": _round_val(intensity, 4),
                "energy_intensity_unit": f"kWh/{line.production_unit}",
                "annual_production": line.annual_production_units,
                "production_unit": line.production_unit,
            })

        return summaries

    # -------------------------------------------------------------------
    # Internal: Sankey Data
    # -------------------------------------------------------------------

    def _build_sankey_data(
        self,
        node_results: List[ProcessNodeResult],
        flows: Optional[List[EnergyFlow]],
        total_input: Decimal,
        total_output: Decimal,
        total_losses: Decimal,
    ) -> SankeyData:
        """Build Sankey diagram data for energy flow visualisation.

        Creates nodes for each process, plus source (energy input) and
        sink (useful output / losses) nodes.

        Args:
            node_results: Enriched process node results.
            flows: Explicit energy flows (optional).
            total_input: Total system input (kWh).
            total_output: Total useful output (kWh).
            total_losses: Total losses (kWh).

        Returns:
            SankeyData with nodes and links.
        """
        sankey_nodes: List[SankeyNode] = []
        sankey_links: List[SankeyLink] = []

        # Source node (energy supply)
        sankey_nodes.append(SankeyNode(
            node_id="SOURCE",
            name="Energy Supply",
            value_kwh=_round_val(total_input, 2),
            node_type="source",
        ))

        # Process nodes
        for nr in node_results:
            sankey_nodes.append(SankeyNode(
                node_id=nr.node_id,
                name=nr.name,
                value_kwh=nr.input_energy_kwh,
                node_type="process",
            ))

            # Loss node for this process
            if nr.losses_kwh > 0:
                loss_id = f"LOSS_{nr.node_id}"
                sankey_nodes.append(SankeyNode(
                    node_id=loss_id,
                    name=f"Losses ({nr.name})",
                    value_kwh=nr.losses_kwh,
                    node_type="loss",
                ))
                sankey_links.append(SankeyLink(
                    source=nr.node_id,
                    target=loss_id,
                    value_kwh=nr.losses_kwh,
                    energy_type="waste_heat",
                ))

        # Output node (useful energy)
        sankey_nodes.append(SankeyNode(
            node_id="OUTPUT",
            name="Useful Output",
            value_kwh=_round_val(total_output, 2),
            node_type="output",
        ))

        # If explicit flows are provided, use them for links
        if flows:
            for f in flows:
                sankey_links.append(SankeyLink(
                    source=f.source_node,
                    target=f.target_node,
                    value_kwh=f.energy_kwh,
                    energy_type=f.energy_type.value,
                ))
        else:
            # Auto-generate links: SOURCE -> each node, each node -> OUTPUT
            for nr in node_results:
                sankey_links.append(SankeyLink(
                    source="SOURCE",
                    target=nr.node_id,
                    value_kwh=nr.input_energy_kwh,
                    energy_type="electrical",
                ))
                if nr.output_energy_kwh > 0:
                    sankey_links.append(SankeyLink(
                        source=nr.node_id,
                        target="OUTPUT",
                        value_kwh=nr.output_energy_kwh,
                        energy_type="product_enthalpy",
                    ))

        return SankeyData(
            nodes=sankey_nodes,
            links=sankey_links,
            total_input=_round_val(total_input, 2),
            total_output=_round_val(total_output, 2),
            total_losses=_round_val(total_losses, 2),
        )

    # -------------------------------------------------------------------
    # Internal: Loss Identification
    # -------------------------------------------------------------------

    def _identify_losses(
        self, node_results: List[ProcessNodeResult],
    ) -> List[LossBreakdown]:
        """Identify and classify losses from all process nodes.

        For each node with losses > 0, classifies the loss type based
        on the process type and assigns temperature grades and recovery
        potential.

        Args:
            node_results: Enriched process node results.

        Returns:
            List of LossBreakdown sorted by quantity descending.
        """
        losses: List[LossBreakdown] = []

        # Map process types to typical loss types
        thermal_processes = {
            "heating_furnace", "boiler", "kiln", "oven", "dryer",
            "heat_exchanger", "distillation", "evaporator", "reactor",
            "steam_system",
        }
        mechanical_processes = {
            "compressor", "pump", "fan", "motor_drive", "conveyor",
            "crusher_grinder", "compressed_air_system",
        }
        electrical_processes = {
            "transformer", "rectifier", "lighting_system",
        }

        for nr in node_results:
            if nr.losses_kwh <= 0:
                continue

            proc = nr.process_type
            loss_kwh = _decimal(nr.losses_kwh)

            if proc in thermal_processes:
                # Split thermal losses into flue gas + radiation
                flue_gas_fraction = Decimal("0.6")
                radiation_fraction = Decimal("0.4")

                # Estimate temperature based on process type
                temp_map: Dict[str, float] = {
                    "heating_furnace": 500.0,
                    "kiln": 600.0,
                    "oven": 250.0,
                    "boiler": 180.0,
                    "dryer": 120.0,
                    "heat_exchanger": 80.0,
                    "distillation": 150.0,
                    "evaporator": 100.0,
                    "reactor": 200.0,
                    "steam_system": 150.0,
                }
                temp = temp_map.get(proc, 150.0)
                grade = self._classify_temperature(temp)
                recovery_pct = _decimal(
                    self._recovery_potential.get(grade, 10.0)
                )

                # Flue gas / exhaust loss
                flue_kwh = loss_kwh * flue_gas_fraction
                flue_recovery = flue_kwh * recovery_pct / Decimal("100")
                losses.append(LossBreakdown(
                    loss_type=LossType.THERMAL_FLUE_GAS,
                    location=f"{nr.name} exhaust",
                    node_id=nr.node_id,
                    quantity_kwh=_round_val(flue_kwh, 2),
                    temperature_c=temp,
                    temperature_grade=grade,
                    recoverable=temp > 60.0,
                    recovery_potential_pct=_round_val(recovery_pct, 1),
                    recovery_potential_kwh=_round_val(flue_recovery, 2),
                ))

                # Radiation / surface loss
                rad_kwh = loss_kwh * radiation_fraction
                rad_recovery = rad_kwh * Decimal("0.1")  # Limited recovery
                losses.append(LossBreakdown(
                    loss_type=LossType.THERMAL_RADIATION,
                    location=f"{nr.name} surface",
                    node_id=nr.node_id,
                    quantity_kwh=_round_val(rad_kwh, 2),
                    temperature_c=temp * 0.5,  # Surface temp lower
                    temperature_grade=self._classify_temperature(temp * 0.5),
                    recoverable=False,
                    recovery_potential_pct=_round1(10.0),
                    recovery_potential_kwh=_round_val(rad_recovery, 2),
                ))

            elif proc in mechanical_processes:
                temp = 40.0
                grade = self._classify_temperature(temp)

                if proc == "compressed_air_system":
                    # Split into leaks + heat
                    leak_fraction = Decimal("0.3")
                    heat_fraction = Decimal("0.7")

                    leak_kwh = loss_kwh * leak_fraction
                    losses.append(LossBreakdown(
                        loss_type=LossType.COMPRESSED_AIR_LEAKS,
                        location=f"{nr.name} leakage",
                        node_id=nr.node_id,
                        quantity_kwh=_round_val(leak_kwh, 2),
                        recoverable=True,
                        recovery_potential_pct=_round1(80.0),
                        recovery_potential_kwh=_round_val(
                            leak_kwh * Decimal("0.8"), 2
                        ),
                    ))

                    heat_kwh = loss_kwh * heat_fraction
                    losses.append(LossBreakdown(
                        loss_type=LossType.CONVERSION_INEFFICIENCY,
                        location=f"{nr.name} compression heat",
                        node_id=nr.node_id,
                        quantity_kwh=_round_val(heat_kwh, 2),
                        temperature_c=80.0,
                        temperature_grade=TemperatureGrade.LOW,
                        recoverable=True,
                        recovery_potential_pct=_round1(30.0),
                        recovery_potential_kwh=_round_val(
                            heat_kwh * Decimal("0.3"), 2
                        ),
                    ))
                else:
                    losses.append(LossBreakdown(
                        loss_type=LossType.MECHANICAL_FRICTION,
                        location=f"{nr.name}",
                        node_id=nr.node_id,
                        quantity_kwh=_round_val(loss_kwh, 2),
                        temperature_c=temp,
                        temperature_grade=grade,
                        recoverable=False,
                        recovery_potential_pct=0.0,
                        recovery_potential_kwh=0.0,
                    ))

            elif proc in electrical_processes:
                losses.append(LossBreakdown(
                    loss_type=LossType.ELECTRICAL_TRANSFORMER
                    if proc == "transformer" else LossType.CONVERSION_INEFFICIENCY,
                    location=f"{nr.name}",
                    node_id=nr.node_id,
                    quantity_kwh=_round_val(loss_kwh, 2),
                    temperature_c=35.0,
                    temperature_grade=TemperatureGrade.AMBIENT,
                    recoverable=False,
                    recovery_potential_pct=0.0,
                    recovery_potential_kwh=0.0,
                ))

            else:
                # Generic loss
                losses.append(LossBreakdown(
                    loss_type=LossType.OTHER,
                    location=f"{nr.name}",
                    node_id=nr.node_id,
                    quantity_kwh=_round_val(loss_kwh, 2),
                    recoverable=False,
                    recovery_potential_pct=0.0,
                    recovery_potential_kwh=0.0,
                ))

        losses.sort(key=lambda l: l.quantity_kwh, reverse=True)
        return losses

    def _classify_temperature(self, temp_c: float) -> str:
        """Classify temperature into a grade.

        Args:
            temp_c: Temperature in Celsius.

        Returns:
            TemperatureGrade value string.
        """
        if temp_c > 400.0:
            return TemperatureGrade.HIGH
        elif temp_c > 100.0:
            return TemperatureGrade.MEDIUM
        elif temp_c > 40.0:
            return TemperatureGrade.LOW
        else:
            return TemperatureGrade.AMBIENT

    # -------------------------------------------------------------------
    # Internal: Intensity Metrics
    # -------------------------------------------------------------------

    def _calculate_intensity_metrics(
        self,
        lines: List[ProductionLine],
        total_input: Decimal,
    ) -> List[IntensityMetric]:
        """Calculate energy intensity metrics per production line.

        Args:
            lines: Production lines with production data.
            total_input: Total facility energy input.

        Returns:
            List of IntensityMetric.
        """
        metrics: List[IntensityMetric] = []

        for line in lines:
            line_input = Decimal("0")
            for node in line.nodes:
                line_input += _decimal(node.input_energy_kwh)

            if line.total_input_kwh is not None:
                line_input = _decimal(line.total_input_kwh)

            if line.annual_production_units and line.annual_production_units > 0:
                prod = _decimal(line.annual_production_units)
                intensity = _safe_divide(line_input, prod)

                metrics.append(IntensityMetric(
                    metric_name=f"Energy Intensity - {line.name}",
                    value=_round_val(intensity, 4),
                    unit=f"kWh/{line.production_unit}",
                    line_id=line.line_id,
                ))

                # If unit is kg or tonne, add conversions
                if line.production_unit == "kg":
                    per_tonne = intensity * Decimal("1000")
                    metrics.append(IntensityMetric(
                        metric_name=f"Energy Intensity - {line.name} (per tonne)",
                        value=_round_val(per_tonne, 2),
                        unit="kWh/tonne",
                        line_id=line.line_id,
                    ))
                elif line.production_unit == "tonne":
                    per_kg = _safe_divide(intensity, Decimal("1000"))
                    metrics.append(IntensityMetric(
                        metric_name=f"Energy Intensity - {line.name} (per kg)",
                        value=_round_val(per_kg, 4),
                        unit="kWh/kg",
                        line_id=line.line_id,
                    ))

        return metrics

    # -------------------------------------------------------------------
    # Internal: Optimisation Scoring
    # -------------------------------------------------------------------

    def _score_optimisation_opportunities(
        self, node_results: List[ProcessNodeResult],
    ) -> List[OptimisationOpportunity]:
        """Score optimisation opportunities for each process node.

        Priority scoring (0-100) considers:
        - Energy savings magnitude (40% weight)
        - Gap vs BAT efficiency (30% weight)
        - Absolute loss quantity (30% weight)

        Args:
            node_results: Enriched process node results.

        Returns:
            List of OptimisationOpportunity sorted by priority descending.
        """
        opportunities: List[OptimisationOpportunity] = []

        # Find max values for normalisation
        max_improvement = max(
            (nr.improvement_potential_kwh for nr in node_results),
            default=1.0,
        )
        max_loss = max(
            (nr.losses_kwh for nr in node_results),
            default=1.0,
        )
        max_gap = max(
            (nr.gap_vs_bat_pct for nr in node_results),
            default=1.0,
        )

        if max_improvement <= 0:
            max_improvement = 1.0
        if max_loss <= 0:
            max_loss = 1.0
        if max_gap <= 0:
            max_gap = 1.0

        for nr in node_results:
            if nr.improvement_potential_kwh <= 0 and nr.gap_vs_bat_pct <= 0:
                continue

            # Normalised scores (0-100)
            savings_score = (nr.improvement_potential_kwh / max_improvement) * 100.0
            gap_score = (nr.gap_vs_bat_pct / max_gap) * 100.0
            loss_score = (nr.losses_kwh / max_loss) * 100.0

            priority = (
                savings_score * 0.4 + gap_score * 0.3 + loss_score * 0.3
            )

            savings_pct = Decimal("0")
            if _decimal(nr.input_energy_kwh) > Decimal("0"):
                savings_pct = _safe_pct(
                    _decimal(nr.improvement_potential_kwh),
                    _decimal(nr.input_energy_kwh),
                )

            target_eff = nr.benchmark_bat_pct or nr.benchmark_good_pct or 0.0

            # Determine category
            thermal_types = {
                "heating_furnace", "boiler", "kiln", "oven", "dryer",
                "heat_exchanger", "distillation", "evaporator", "reactor",
                "steam_system",
            }
            mech_types = {
                "compressor", "pump", "fan", "motor_drive", "conveyor",
                "crusher_grinder", "compressed_air_system",
            }
            if nr.process_type in thermal_types:
                category = "thermal"
            elif nr.process_type in mech_types:
                category = "mechanical"
            else:
                category = "other"

            desc = (
                f"Improve {nr.name} efficiency from {nr.efficiency_pct}% "
                f"to {target_eff}% ({nr.performance_label} -> BAT). "
                f"Potential savings: {nr.improvement_potential_kwh:.0f} kWh/year."
            )

            opportunities.append(OptimisationOpportunity(
                description=desc,
                affected_node=nr.node_id,
                current_efficiency_pct=nr.efficiency_pct,
                target_efficiency_pct=target_eff,
                savings_kwh=nr.improvement_potential_kwh,
                savings_pct=_round_val(savings_pct, 1),
                priority_score=_round1(priority),
                category=category,
            ))

        opportunities.sort(key=lambda o: o.priority_score, reverse=True)
        return opportunities

    # -------------------------------------------------------------------
    # Internal: Recommendations
    # -------------------------------------------------------------------

    def _generate_recommendations(
        self,
        node_results: List[ProcessNodeResult],
        losses: List[LossBreakdown],
        opportunities: List[OptimisationOpportunity],
        overall_eff: Decimal,
        total_recoverable: Decimal,
        total_input: Decimal,
    ) -> List[str]:
        """Generate actionable recommendations based on process analysis.

        All recommendations are deterministic threshold-based rules.

        Args:
            node_results: Enriched process node results.
            losses: Loss breakdown.
            opportunities: Scored optimisation opportunities.
            overall_eff: Overall system efficiency.
            total_recoverable: Total recoverable energy.
            total_input: Total system input.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # R1: Overall efficiency
        eff_float = float(overall_eff)
        if eff_float < 30.0:
            recs.append(
                f"Overall process efficiency is {eff_float}%, which is very low. "
                f"A comprehensive process integration study (pinch analysis) is "
                f"recommended to identify systemic efficiency improvements."
            )
        elif eff_float < 50.0:
            recs.append(
                f"Overall process efficiency is {eff_float}%. There is significant "
                f"room for improvement. Focus on the largest loss points and "
                f"heat recovery opportunities."
            )

        # R2: Heat recovery potential
        recoverable_pct = _safe_pct(total_recoverable, total_input)
        if float(total_recoverable) > 0:
            recs.append(
                f"Total recoverable waste energy: {float(total_recoverable):,.0f} kWh/year "
                f"({float(recoverable_pct):.1f}% of total input). "
                f"Implement heat recovery systems (heat exchangers, economisers, "
                f"heat pumps) to capture this energy."
            )

        # R3: Top opportunities
        if opportunities:
            top = opportunities[0]
            recs.append(
                f"Highest priority optimisation: {top.affected_node} - "
                f"potential savings of {top.savings_kwh:,.0f} kWh/year by "
                f"improving efficiency from {top.current_efficiency_pct}% "
                f"to {top.target_efficiency_pct}% (BAT level)."
            )

        # R4: Below-typical nodes
        below_typical = [
            nr for nr in node_results
            if nr.performance_label == "Below Typical"
        ]
        if below_typical:
            names = ", ".join(nr.name for nr in below_typical[:3])
            recs.append(
                f"Process nodes operating below sector typical efficiency: "
                f"{names}. These should be prioritised for immediate "
                f"maintenance and optimisation."
            )

        # R5: Compressed air leaks
        ca_leaks = [
            l for l in losses
            if l.loss_type == LossType.COMPRESSED_AIR_LEAKS
        ]
        if ca_leaks:
            total_leak_kwh = sum(l.quantity_kwh for l in ca_leaks)
            recs.append(
                f"Compressed air leakage losses total {total_leak_kwh:,.0f} kWh/year. "
                f"Conduct an ultrasonic leak survey and repair all leaks. "
                f"This is typically a no-cost/low-cost measure with immediate payback."
            )

        # R6: High-grade waste heat
        high_grade = [
            l for l in losses
            if l.temperature_grade in (TemperatureGrade.HIGH, TemperatureGrade.MEDIUM)
            and l.recoverable
        ]
        if high_grade:
            total_hg_kwh = sum(l.recovery_potential_kwh for l in high_grade)
            recs.append(
                f"High and medium grade waste heat recovery potential: "
                f"{total_hg_kwh:,.0f} kWh/year from {len(high_grade)} sources. "
                f"Consider waste heat boilers, economisers, or organic "
                f"Rankine cycle (ORC) systems for conversion to electricity."
            )

        # R7: Sub-metering for process nodes
        if len(node_results) < 5:
            recs.append(
                "Limited process nodes mapped. Install energy sub-metering on "
                "all major unit operations to enable detailed process energy "
                "analysis and continuous monitoring per EN 16247-3."
            )

        return recs
