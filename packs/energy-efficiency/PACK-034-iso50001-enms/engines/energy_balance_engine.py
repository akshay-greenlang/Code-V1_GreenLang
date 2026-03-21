# -*- coding: utf-8 -*-
"""
EnergyBalanceEngine - PACK-034 ISO 50001 EnMS Engine 6
======================================================

Facility energy balance analysis engine providing Sankey diagram generation,
sub-metering reconciliation, loss estimation, and end-use breakdown.
Implements energy conservation validation (first law of thermodynamics)
across metering hierarchies with discrepancy detection.

Calculation Methodology:
    Energy Balance (Conservation):
        total_input  = sum(flow.value_kwh  for flow in flows  if flow.flow_type == INPUT)
        total_output = sum(flow.value_kwh  for flow in flows  if flow.flow_type == OUTPUT)
        total_losses = sum(flow.value_kwh  for flow in flows  if flow.flow_type in (LOSS, REJECTED))
        unaccounted  = total_input - total_output - total_losses
        balance_efficiency_pct = (total_output / total_input) * 100

    Meter Reconciliation:
        difference     = parent_reading - sum(child_readings)
        difference_pct = abs(difference) / parent_reading * 100
        status:
            balanced           : difference_pct <= 2.0%
            minor_discrepancy  : 2.0% < difference_pct <= 5.0%
            major_discrepancy  : 5.0% < difference_pct <= 10.0%
            unreconciled       : difference_pct > 10.0%

    Loss Estimation:
        distribution_loss = total_input * typical_loss_factor[system]
        transformation_loss = conversion_flow * (1 - conversion_efficiency)
        unmetered_load    = total_input - metered_output - known_losses

    Sankey Diagram:
        Nodes: source -> conversion -> end-use -> losses/rejected
        Links: proportional energy flows between nodes
        Colors: assigned by energy source/type per standard palette

Regulatory References:
    - ISO 50001:2018 - Energy management systems (Clause 6.3, 6.6)
    - ISO 50006:2014 - Measuring energy performance using EnPIs and EnBs
    - ISO 50015:2014 - Measurement and verification of energy performance
    - IEC 62053 series - Electricity metering accuracy classes
    - EN 16247-1:2022 - Energy audits (general requirements)
    - ASHRAE Guideline 14-2014 - Measurement of Energy Savings

Zero-Hallucination:
    - All formulas are standard energy engineering calculations
    - Loss factors from published engineering references
    - Meter accuracy classes per IEC 62053 standards
    - Conversion factors from NIST / IEA published data
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-034 ISO 50001 EnMS
Engine:  6 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
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
    """Safely convert a value to Decimal."""
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


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EnergyFlowType(str, Enum):
    """Type of energy flow within the facility balance boundary.

    INPUT:      Energy entering the facility boundary.
    OUTPUT:     Useful energy delivered to end-uses.
    LOSS:       Unavoidable system losses (distribution, transformation).
    CONVERSION: Energy undergoing form change (e.g. fuel to steam).
    STORAGE:    Energy entering or leaving storage (batteries, thermal).
    REJECTED:   Waste heat or energy rejected to the environment.
    """
    INPUT = "input"
    OUTPUT = "output"
    LOSS = "loss"
    CONVERSION = "conversion"
    STORAGE = "storage"
    REJECTED = "rejected"


class EnergySource(str, Enum):
    """Primary energy source or carrier.

    Covers grid electricity, fossil fuels, district energy, and renewables.
    """
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    LPG = "lpg"
    STEAM = "steam"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    SOLAR_PV = "solar_pv"
    SOLAR_THERMAL = "solar_thermal"
    WIND = "wind"
    BIOMASS = "biomass"
    BIOGAS = "biogas"
    DIESEL = "diesel"
    PETROL = "petrol"
    HYDROGEN = "hydrogen"
    OTHER = "other"


class EndUseCategory(str, Enum):
    """End-use category for energy consumption within a facility.

    Follows ISO 50001 Annex A end-use classification with extensions.
    """
    HVAC_HEATING = "hvac_heating"
    HVAC_COOLING = "hvac_cooling"
    HVAC_VENTILATION = "hvac_ventilation"
    LIGHTING = "lighting"
    MOTORS_DRIVES = "motors_drives"
    COMPRESSED_AIR = "compressed_air"
    PROCESS_HEAT = "process_heat"
    PROCESS_COOLING = "process_cooling"
    REFRIGERATION = "refrigeration"
    DOMESTIC_HOT_WATER = "domestic_hot_water"
    PLUG_LOADS = "plug_loads"
    IT_EQUIPMENT = "it_equipment"
    TRANSPORT = "transport"
    COOKING = "cooking"
    LAUNDRY = "laundry"
    OTHER = "other"


class MeterType(str, Enum):
    """Type of energy meter in the metering hierarchy.

    MAIN:       Revenue / billing meter at the facility boundary.
    SUB:        Sub-meter measuring a portion of the main supply.
    CHECK:      Temporary / portable meter for verification.
    VIRTUAL:    Software-calculated meter from other meter readings.
    CALCULATED: Meter value derived from engineering calculation.
    """
    MAIN = "main"
    SUB = "sub"
    CHECK = "check"
    VIRTUAL = "virtual"
    CALCULATED = "calculated"


class ReconciliationStatus(str, Enum):
    """Status of meter reconciliation between parent and children.

    BALANCED:            Difference <= 2% (within normal tolerance).
    MINOR_DISCREPANCY:   2% < difference <= 5% (investigation recommended).
    MAJOR_DISCREPANCY:   5% < difference <= 10% (action required).
    UNRECONCILED:        Difference > 10% (metering fault likely).
    """
    BALANCED = "balanced"
    MINOR_DISCREPANCY = "minor_discrepancy"
    MAJOR_DISCREPANCY = "major_discrepancy"
    UNRECONCILED = "unreconciled"


# ---------------------------------------------------------------------------
# Constants / Reference Data
# ---------------------------------------------------------------------------

# Typical system loss factors (fraction of energy input lost).
# Sources: ASHRAE Handbooks, DOE Industrial Assessment Center data,
# IEA Energy Efficiency Indicators methodology.
TYPICAL_LOSS_FACTORS: Dict[str, Decimal] = {
    "electrical_distribution": Decimal("0.03"),
    "transformer": Decimal("0.02"),
    "motor_efficiency": Decimal("0.08"),
    "compressed_air_system": Decimal("0.25"),
    "steam_distribution": Decimal("0.10"),
    "steam_condensate_losses": Decimal("0.05"),
    "boiler_flue_gas": Decimal("0.15"),
    "boiler_radiation": Decimal("0.02"),
    "boiler_blowdown": Decimal("0.03"),
    "chiller_condenser_heat": Decimal("0.30"),
    "cooling_tower_evaporation": Decimal("0.05"),
    "hvac_duct_leakage": Decimal("0.10"),
    "building_envelope": Decimal("0.15"),
    "lighting_heat_gain": Decimal("0.90"),
    "pipe_insulation": Decimal("0.05"),
    "thermal_storage": Decimal("0.03"),
    "power_factor_losses": Decimal("0.02"),
    "harmonic_losses": Decimal("0.01"),
    "standby_consumption": Decimal("0.05"),
    "vsd_conversion_losses": Decimal("0.03"),
}

# Energy conversion factors (to kWh equivalents).
# Sources: NIST SP 811, IEA Statistics Manual.
ENERGY_CONVERSION_FACTORS: Dict[str, Decimal] = {
    "kwh_to_mj": Decimal("3.6"),
    "mj_to_kwh": Decimal("0.277778"),
    "therms_to_kwh": Decimal("29.3001"),
    "kwh_to_therms": Decimal("0.034130"),
    "gj_to_kwh": Decimal("277.778"),
    "kwh_to_gj": Decimal("0.0036"),
    "mmbtu_to_kwh": Decimal("293.071"),
    "kwh_to_mmbtu": Decimal("0.003412"),
    "btu_to_kwh": Decimal("0.000293071"),
    "kwh_to_btu": Decimal("3412.14"),
    "mwh_to_kwh": Decimal("1000"),
    "kwh_to_mwh": Decimal("0.001"),
    "kg_coal_to_kwh": Decimal("8.141"),
    "litre_diesel_to_kwh": Decimal("10.7"),
    "litre_petrol_to_kwh": Decimal("9.5"),
    "m3_natural_gas_to_kwh": Decimal("10.55"),
    "kg_lpg_to_kwh": Decimal("13.8"),
    "kg_biomass_to_kwh": Decimal("4.5"),
    "m3_biogas_to_kwh": Decimal("6.0"),
    "kg_hydrogen_to_kwh": Decimal("33.33"),
    "litre_fuel_oil_to_kwh": Decimal("10.35"),
}

# Meter accuracy classes per IEC 62053 series.
# Class: maximum percentage error at unity power factor, rated current.
METER_ACCURACY_STANDARDS: Dict[str, Dict[str, Decimal]] = {
    "class_0_2": {
        "accuracy_pct": Decimal("0.2"),
        "description_code": "IEC 62053-22",
        "application": "revenue_metering_high_accuracy",
    },
    "class_0_5": {
        "accuracy_pct": Decimal("0.5"),
        "description_code": "IEC 62053-22",
        "application": "revenue_metering_standard",
    },
    "class_1_0": {
        "accuracy_pct": Decimal("1.0"),
        "description_code": "IEC 62053-21",
        "application": "sub_metering_standard",
    },
    "class_2_0": {
        "accuracy_pct": Decimal("2.0"),
        "description_code": "IEC 62053-21",
        "application": "sub_metering_general",
    },
    "class_3_0": {
        "accuracy_pct": Decimal("3.0"),
        "description_code": "IEC 62053-21",
        "application": "check_metering_indicative",
    },
}

# Standard color palette for Sankey diagram nodes by energy source.
SANKEY_COLOR_PALETTE: Dict[str, str] = {
    EnergySource.ELECTRICITY.value: "#FFD700",
    EnergySource.NATURAL_GAS.value: "#FF6347",
    EnergySource.FUEL_OIL.value: "#8B4513",
    EnergySource.LPG.value: "#FF8C00",
    EnergySource.STEAM.value: "#B0C4DE",
    EnergySource.DISTRICT_HEATING.value: "#DC143C",
    EnergySource.DISTRICT_COOLING.value: "#00CED1",
    EnergySource.SOLAR_PV.value: "#FFD700",
    EnergySource.SOLAR_THERMAL.value: "#FFA500",
    EnergySource.WIND.value: "#87CEEB",
    EnergySource.BIOMASS.value: "#228B22",
    EnergySource.BIOGAS.value: "#32CD32",
    EnergySource.DIESEL.value: "#696969",
    EnergySource.PETROL.value: "#A0522D",
    EnergySource.HYDROGEN.value: "#E0FFFF",
    EnergySource.OTHER.value: "#C0C0C0",
    # End-use category colors.
    EndUseCategory.HVAC_HEATING.value: "#FF4500",
    EndUseCategory.HVAC_COOLING.value: "#1E90FF",
    EndUseCategory.HVAC_VENTILATION.value: "#87CEEB",
    EndUseCategory.LIGHTING.value: "#FFFF00",
    EndUseCategory.MOTORS_DRIVES.value: "#4682B4",
    EndUseCategory.COMPRESSED_AIR.value: "#00BFFF",
    EndUseCategory.PROCESS_HEAT.value: "#FF6347",
    EndUseCategory.PROCESS_COOLING.value: "#00CED1",
    EndUseCategory.REFRIGERATION.value: "#4169E1",
    EndUseCategory.DOMESTIC_HOT_WATER.value: "#FF7F50",
    EndUseCategory.PLUG_LOADS.value: "#9370DB",
    EndUseCategory.IT_EQUIPMENT.value: "#6A5ACD",
    EndUseCategory.TRANSPORT.value: "#808080",
    EndUseCategory.COOKING.value: "#D2691E",
    EndUseCategory.LAUNDRY.value: "#DDA0DD",
    EndUseCategory.OTHER.value: "#A9A9A9",
    # Special nodes.
    "losses": "#FF0000",
    "rejected": "#8B0000",
    "unaccounted": "#FF69B4",
    "storage": "#9ACD32",
    "conversion": "#DAA520",
}

# Reconciliation thresholds (percentage).
_BALANCED_THRESHOLD: Decimal = Decimal("2.0")
_MINOR_THRESHOLD: Decimal = Decimal("5.0")
_MAJOR_THRESHOLD: Decimal = Decimal("10.0")

# Default conversion efficiencies by system type.
_CONVERSION_EFFICIENCIES: Dict[str, Decimal] = {
    "boiler_natural_gas": Decimal("0.85"),
    "boiler_fuel_oil": Decimal("0.82"),
    "boiler_biomass": Decimal("0.75"),
    "boiler_biogas": Decimal("0.80"),
    "chiller_electric": Decimal("3.50"),
    "chiller_absorption": Decimal("1.20"),
    "heat_pump": Decimal("3.00"),
    "cogeneration_chp": Decimal("0.80"),
    "solar_pv_inverter": Decimal("0.96"),
    "transformer": Decimal("0.98"),
    "vsd_drive": Decimal("0.97"),
    "motor_ie3": Decimal("0.92"),
    "motor_ie4": Decimal("0.95"),
    "steam_turbine": Decimal("0.35"),
    "gas_turbine": Decimal("0.38"),
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class EnergyFlow(BaseModel):
    """Single energy flow within the facility balance boundary.

    Attributes:
        flow_id: Unique flow identifier.
        source: Energy source / carrier.
        flow_type: Classification of this flow (input, output, loss, etc.).
        end_use: End-use category (optional, for output flows).
        value_kwh: Energy quantity (kWh).
        value_cost: Associated cost (currency units, optional).
        percentage_of_total: Flow as percentage of total facility input.
        parent_flow_id: ID of the parent flow (for hierarchical flows).
        meter_id: ID of the meter measuring this flow (optional).
    """
    flow_id: str = Field(
        default_factory=_new_uuid, description="Unique flow identifier"
    )
    source: EnergySource = Field(
        default=EnergySource.ELECTRICITY, description="Energy source"
    )
    flow_type: EnergyFlowType = Field(
        default=EnergyFlowType.INPUT, description="Flow type classification"
    )
    end_use: Optional[EndUseCategory] = Field(
        default=None, description="End-use category (for output flows)"
    )
    value_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Energy quantity (kWh)"
    )
    value_cost: Optional[Decimal] = Field(
        default=None, ge=0, description="Associated cost (currency units)"
    )
    percentage_of_total: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Flow as percentage of total input"
    )
    parent_flow_id: Optional[str] = Field(
        default=None, description="Parent flow ID for hierarchical flows"
    )
    meter_id: Optional[str] = Field(
        default=None, description="Meter ID measuring this flow"
    )

    @field_validator("flow_id")
    @classmethod
    def validate_flow_id(cls, v: str) -> str:
        """Ensure flow_id is non-empty."""
        if not v or not v.strip():
            return _new_uuid()
        return v


class MeterNode(BaseModel):
    """Meter node in the facility metering hierarchy.

    Attributes:
        meter_id: Unique meter identifier.
        meter_name: Human-readable meter name.
        meter_type: Classification (main, sub, check, virtual, calculated).
        parent_meter_id: ID of the parent meter in the hierarchy.
        energy_source: Energy source measured by this meter.
        location: Physical location description.
        accuracy_class: Meter accuracy class (percentage error).
        last_calibration: Date of last calibration.
        reading_kwh: Current period meter reading (kWh).
        children: List of child meter IDs.
    """
    meter_id: str = Field(
        default_factory=_new_uuid, description="Unique meter identifier"
    )
    meter_name: str = Field(
        default="", max_length=500, description="Human-readable meter name"
    )
    meter_type: MeterType = Field(
        default=MeterType.SUB, description="Meter type classification"
    )
    parent_meter_id: Optional[str] = Field(
        default=None, description="Parent meter ID in hierarchy"
    )
    energy_source: EnergySource = Field(
        default=EnergySource.ELECTRICITY, description="Energy source measured"
    )
    location: str = Field(
        default="", max_length=500, description="Physical location"
    )
    accuracy_class: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=Decimal("10.0"),
        description="Accuracy class (percentage error)"
    )
    last_calibration: Optional[date] = Field(
        default=None, description="Date of last calibration"
    )
    reading_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Current period reading (kWh)"
    )
    children: List[str] = Field(
        default_factory=list, description="List of child meter IDs"
    )

    @field_validator("meter_id")
    @classmethod
    def validate_meter_id(cls, v: str) -> str:
        """Ensure meter_id is non-empty."""
        if not v or not v.strip():
            return _new_uuid()
        return v


class SankeyNode(BaseModel):
    """Node in a Sankey energy flow diagram.

    Attributes:
        node_id: Unique node identifier.
        label: Display label for the node.
        value_kwh: Total energy flowing through this node (kWh).
        level: Hierarchy level (0 = source, 1 = conversion, 2 = end-use, 3 = losses).
        color: Hex color code for rendering.
    """
    node_id: str = Field(
        default_factory=_new_uuid, description="Unique node identifier"
    )
    label: str = Field(
        default="", max_length=200, description="Display label"
    )
    value_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Energy through node (kWh)"
    )
    level: int = Field(
        default=0, ge=0, le=10, description="Hierarchy level"
    )
    color: str = Field(
        default="#C0C0C0", max_length=20, description="Hex color code"
    )


class SankeyLink(BaseModel):
    """Link connecting two nodes in a Sankey diagram.

    Attributes:
        source_id: Source node identifier.
        target_id: Target node identifier.
        value_kwh: Energy flowing through this link (kWh).
        label: Optional display label for the link.
    """
    source_id: str = Field(
        default="", description="Source node ID"
    )
    target_id: str = Field(
        default="", description="Target node ID"
    )
    value_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Energy through link (kWh)"
    )
    label: Optional[str] = Field(
        default=None, max_length=200, description="Display label"
    )


class SankeyDiagram(BaseModel):
    """Complete Sankey diagram data structure for energy flows.

    Attributes:
        nodes: List of all diagram nodes.
        links: List of all diagram links.
        title: Diagram title.
        total_input_kwh: Total energy input (kWh).
        total_output_kwh: Total useful energy output (kWh).
        unaccounted_kwh: Unaccounted energy (kWh).
    """
    nodes: List[SankeyNode] = Field(
        default_factory=list, description="Diagram nodes"
    )
    links: List[SankeyLink] = Field(
        default_factory=list, description="Diagram links"
    )
    title: str = Field(
        default="Facility Energy Balance Sankey Diagram",
        max_length=500, description="Diagram title"
    )
    total_input_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total energy input (kWh)"
    )
    total_output_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total useful output (kWh)"
    )
    unaccounted_kwh: Decimal = Field(
        default=Decimal("0"), description="Unaccounted energy (kWh)"
    )


class MeterReconciliation(BaseModel):
    """Result of reconciling a parent meter against its sub-meters.

    Attributes:
        parent_meter: Parent meter node.
        child_meters: List of child meter nodes.
        parent_reading: Parent meter reading (kWh).
        sum_children: Sum of child meter readings (kWh).
        difference: Absolute difference (parent - children) (kWh).
        difference_pct: Difference as percentage of parent reading.
        status: Reconciliation status classification.
        estimated_losses: Estimated distribution losses in the gap (kWh).
    """
    parent_meter: MeterNode = Field(
        ..., description="Parent meter node"
    )
    child_meters: List[MeterNode] = Field(
        default_factory=list, description="Child meter nodes"
    )
    parent_reading: Decimal = Field(
        default=Decimal("0"), ge=0, description="Parent reading (kWh)"
    )
    sum_children: Decimal = Field(
        default=Decimal("0"), ge=0, description="Sum of children (kWh)"
    )
    difference: Decimal = Field(
        default=Decimal("0"), description="Parent minus children (kWh)"
    )
    difference_pct: Decimal = Field(
        default=Decimal("0"), description="Difference as pct of parent"
    )
    status: ReconciliationStatus = Field(
        default=ReconciliationStatus.UNRECONCILED,
        description="Reconciliation status"
    )
    estimated_losses: Decimal = Field(
        default=Decimal("0"), ge=0, description="Estimated losses (kWh)"
    )


class LossEstimate(BaseModel):
    """Estimated energy loss for a specific category.

    Attributes:
        loss_category: Description of the loss category.
        estimated_kwh: Estimated loss (kWh).
        estimation_method: Method used to estimate the loss.
        confidence_pct: Confidence in the estimate (0-100).
        notes: Additional notes or assumptions.
    """
    loss_category: str = Field(
        default="", max_length=200, description="Loss category"
    )
    estimated_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Estimated loss (kWh)"
    )
    estimation_method: str = Field(
        default="engineering_estimate", max_length=200,
        description="Estimation method"
    )
    confidence_pct: Decimal = Field(
        default=Decimal("70"), ge=0, le=Decimal("100"),
        description="Confidence percentage"
    )
    notes: str = Field(
        default="", max_length=1000, description="Additional notes"
    )


class EnergyBalanceResult(BaseModel):
    """Complete energy balance result for a facility and period.

    Attributes:
        balance_id: Unique balance calculation identifier.
        facility_id: Facility identifier.
        period_start: Start of the analysis period.
        period_end: End of the analysis period.
        total_input_kwh: Total energy input (kWh).
        total_output_kwh: Total useful energy output (kWh).
        total_losses_kwh: Total identified losses (kWh).
        unaccounted_kwh: Energy not accounted for (kWh).
        unaccounted_pct: Unaccounted as percentage of input.
        energy_flows: All energy flows in the balance.
        sankey_diagram: Sankey diagram data structure.
        meter_reconciliations: Meter reconciliation results.
        loss_estimates: Itemised loss estimates.
        balance_efficiency_pct: Overall balance efficiency (output / input * 100).
        provenance_hash: SHA-256 audit trail hash.
        calculation_time_ms: Processing duration (milliseconds).
    """
    balance_id: str = Field(
        default_factory=_new_uuid, description="Unique balance ID"
    )
    facility_id: str = Field(
        default="", description="Facility identifier"
    )
    period_start: date = Field(
        ..., description="Analysis period start"
    )
    period_end: date = Field(
        ..., description="Analysis period end"
    )
    total_input_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total energy input (kWh)"
    )
    total_output_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total useful output (kWh)"
    )
    total_losses_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total losses (kWh)"
    )
    unaccounted_kwh: Decimal = Field(
        default=Decimal("0"), description="Unaccounted energy (kWh)"
    )
    unaccounted_pct: Decimal = Field(
        default=Decimal("0"), description="Unaccounted as pct of input"
    )
    energy_flows: List[EnergyFlow] = Field(
        default_factory=list, description="All energy flows"
    )
    sankey_diagram: SankeyDiagram = Field(
        default_factory=SankeyDiagram, description="Sankey diagram data"
    )
    meter_reconciliations: List[MeterReconciliation] = Field(
        default_factory=list, description="Meter reconciliation results"
    )
    loss_estimates: List[LossEstimate] = Field(
        default_factory=list, description="Itemised loss estimates"
    )
    balance_efficiency_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Balance efficiency (output / input * 100)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )
    calculation_time_ms: int = Field(
        default=0, ge=0, description="Processing time (ms)"
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EnergyBalanceEngine:
    """Facility energy balance analysis with Sankey diagrams and metering.

    Provides complete energy balance calculations including:
    - Energy conservation validation (first law of thermodynamics)
    - Sankey diagram generation for energy flow visualisation
    - Sub-metering hierarchy reconciliation with discrepancy detection
    - Loss estimation and unmetered load identification
    - End-use breakdown analysis and chart data generation

    All calculations use deterministic Decimal arithmetic with SHA-256
    provenance hashing for audit-grade traceability per ISO 50001:2018.

    Usage::

        engine = EnergyBalanceEngine()
        flows = [
            EnergyFlow(
                source=EnergySource.ELECTRICITY,
                flow_type=EnergyFlowType.INPUT,
                value_kwh=Decimal("500000"),
            ),
            EnergyFlow(
                source=EnergySource.ELECTRICITY,
                flow_type=EnergyFlowType.OUTPUT,
                end_use=EndUseCategory.LIGHTING,
                value_kwh=Decimal("80000"),
            ),
        ]
        meters = [
            MeterNode(meter_id="main-1", meter_type=MeterType.MAIN, reading_kwh=Decimal("500000")),
        ]
        result = engine.calculate_energy_balance(flows, meters, "FAC-001")
        print(f"Balance efficiency: {result.balance_efficiency_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise EnergyBalanceEngine.

        Args:
            config: Optional configuration overrides.  Supported keys:
                - balanced_threshold (Decimal): reconciliation balanced pct
                - minor_threshold (Decimal): minor discrepancy pct
                - major_threshold (Decimal): major discrepancy pct
                - custom_loss_factors (dict): override typical loss factors
                - custom_conversion_efficiencies (dict): override efficiencies
                - default_facility_id (str): default facility identifier
        """
        self.config = config or {}
        self._balanced_threshold = _decimal(
            self.config.get("balanced_threshold", _BALANCED_THRESHOLD)
        )
        self._minor_threshold = _decimal(
            self.config.get("minor_threshold", _MINOR_THRESHOLD)
        )
        self._major_threshold = _decimal(
            self.config.get("major_threshold", _MAJOR_THRESHOLD)
        )
        self._custom_loss_factors: Dict[str, Decimal] = {
            k: _decimal(v) for k, v in
            self.config.get("custom_loss_factors", {}).items()
        }
        self._custom_efficiencies: Dict[str, Decimal] = {
            k: _decimal(v) for k, v in
            self.config.get("custom_conversion_efficiencies", {}).items()
        }
        self._default_facility_id = self.config.get(
            "default_facility_id", ""
        )
        logger.info(
            "EnergyBalanceEngine v%s initialised (balanced_thr=%.1f%%, "
            "minor_thr=%.1f%%, major_thr=%.1f%%)",
            self.engine_version,
            float(self._balanced_threshold),
            float(self._minor_threshold),
            float(self._major_threshold),
        )

    # ------------------------------------------------------------------ #
    # Public API: calculate_energy_balance                                #
    # ------------------------------------------------------------------ #

    def calculate_energy_balance(
        self,
        flows: List[EnergyFlow],
        meters: List[MeterNode],
        facility_id: str = "",
        period_start: Optional[date] = None,
        period_end: Optional[date] = None,
    ) -> EnergyBalanceResult:
        """Calculate complete energy balance for a facility.

        Sums all inputs, outputs, and losses.  Validates energy conservation
        (inputs >= outputs + losses).  Generates Sankey diagram, reconciles
        meters, and estimates losses.

        Args:
            flows: List of energy flows within the facility boundary.
            meters: List of meter nodes in the metering hierarchy.
            facility_id: Facility identifier (uses config default if empty).
            period_start: Analysis period start date.
            period_end: Analysis period end date.

        Returns:
            EnergyBalanceResult with full balance, diagram, and reconciliation.

        Raises:
            ValueError: If no flows are provided.
        """
        t0 = time.perf_counter()
        fac_id = facility_id or self._default_facility_id or _new_uuid()
        p_start = period_start or date.today().replace(month=1, day=1)
        p_end = period_end or date.today()

        logger.info(
            "Calculating energy balance: facility=%s, flows=%d, meters=%d, "
            "period=%s to %s",
            fac_id, len(flows), len(meters), p_start, p_end,
        )

        if not flows:
            raise ValueError("At least one energy flow is required.")

        # Step 1: Classify and sum flows
        total_input, total_output, total_losses = self._sum_flows(flows)

        # Step 2: Calculate unaccounted energy
        unaccounted = total_input - total_output - total_losses
        unaccounted_pct = _safe_pct(abs(unaccounted), total_input)

        # Step 3: Calculate percentages on each flow
        enriched_flows = self._enrich_flow_percentages(flows, total_input)

        # Step 4: Balance efficiency
        balance_efficiency = _safe_pct(total_output, total_input)

        # Step 5: Generate Sankey diagram
        sankey = self.generate_sankey_diagram(enriched_flows)

        # Step 6: Reconcile meters
        reconciliations = self.reconcile_meters(meters)

        # Step 7: Build preliminary result for loss estimation
        preliminary = EnergyBalanceResult(
            facility_id=fac_id,
            period_start=p_start,
            period_end=p_end,
            total_input_kwh=_round_val(total_input, 2),
            total_output_kwh=_round_val(total_output, 2),
            total_losses_kwh=_round_val(total_losses, 2),
            unaccounted_kwh=_round_val(unaccounted, 2),
            unaccounted_pct=_round_val(unaccounted_pct, 2),
            energy_flows=enriched_flows,
            sankey_diagram=sankey,
            meter_reconciliations=reconciliations,
            loss_estimates=[],
            balance_efficiency_pct=_round_val(balance_efficiency, 2),
        )

        # Step 8: Estimate losses
        loss_estimates = self.estimate_losses(preliminary)

        # Step 9: Final result
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        result = EnergyBalanceResult(
            facility_id=fac_id,
            period_start=p_start,
            period_end=p_end,
            total_input_kwh=_round_val(total_input, 2),
            total_output_kwh=_round_val(total_output, 2),
            total_losses_kwh=_round_val(total_losses, 2),
            unaccounted_kwh=_round_val(unaccounted, 2),
            unaccounted_pct=_round_val(unaccounted_pct, 2),
            energy_flows=enriched_flows,
            sankey_diagram=sankey,
            meter_reconciliations=reconciliations,
            loss_estimates=loss_estimates,
            balance_efficiency_pct=_round_val(balance_efficiency, 2),
            calculation_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Energy balance complete: facility=%s, input=%.0f kWh, "
            "output=%.0f kWh, losses=%.0f kWh, unaccounted=%.0f kWh (%.1f%%), "
            "efficiency=%.1f%%, hash=%s, %d ms",
            fac_id, float(total_input), float(total_output),
            float(total_losses), float(unaccounted),
            float(unaccounted_pct), float(balance_efficiency),
            result.provenance_hash[:16], elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Public API: generate_sankey_diagram                                 #
    # ------------------------------------------------------------------ #

    def generate_sankey_diagram(
        self,
        flows: List[EnergyFlow],
    ) -> SankeyDiagram:
        """Generate Sankey diagram data from energy flows.

        Creates a node hierarchy: source (level 0) -> conversion (level 1)
        -> end-use (level 2) -> losses/rejected (level 3).  Generates
        links with proportional values and assigns colors by energy type.

        Args:
            flows: List of energy flows with types and values.

        Returns:
            SankeyDiagram with nodes, links, and summary totals.
        """
        t0 = time.perf_counter()
        logger.info("Generating Sankey diagram from %d flows", len(flows))

        nodes: List[SankeyNode] = []
        links: List[SankeyLink] = []
        node_map: Dict[str, SankeyNode] = {}

        total_input = Decimal("0")
        total_output = Decimal("0")

        # Pass 1: Create source nodes (level 0) from input flows
        source_nodes: Dict[str, Decimal] = {}
        for flow in flows:
            if flow.flow_type == EnergyFlowType.INPUT:
                key = f"source_{flow.source.value}"
                source_nodes[key] = source_nodes.get(key, Decimal("0")) + flow.value_kwh
                total_input += flow.value_kwh

        for key, value in source_nodes.items():
            source_name = key.replace("source_", "").replace("_", " ").title()
            color = SANKEY_COLOR_PALETTE.get(
                key.replace("source_", ""), "#C0C0C0"
            )
            node = SankeyNode(
                node_id=key, label=source_name,
                value_kwh=_round_val(value, 2), level=0, color=color,
            )
            nodes.append(node)
            node_map[key] = node

        # Pass 2: Create conversion nodes (level 1)
        conversion_nodes: Dict[str, Decimal] = {}
        for flow in flows:
            if flow.flow_type == EnergyFlowType.CONVERSION:
                key = f"conv_{flow.source.value}"
                conversion_nodes[key] = (
                    conversion_nodes.get(key, Decimal("0")) + flow.value_kwh
                )

        for key, value in conversion_nodes.items():
            conv_name = key.replace("conv_", "").replace("_", " ").title()
            node = SankeyNode(
                node_id=key, label=f"{conv_name} Conversion",
                value_kwh=_round_val(value, 2), level=1,
                color=SANKEY_COLOR_PALETTE.get("conversion", "#DAA520"),
            )
            nodes.append(node)
            node_map[key] = node

        # Pass 3: Create end-use nodes (level 2) from output flows
        enduse_nodes: Dict[str, Decimal] = {}
        for flow in flows:
            if flow.flow_type == EnergyFlowType.OUTPUT:
                eu = flow.end_use.value if flow.end_use else EndUseCategory.OTHER.value
                key = f"enduse_{eu}"
                enduse_nodes[key] = (
                    enduse_nodes.get(key, Decimal("0")) + flow.value_kwh
                )
                total_output += flow.value_kwh

        for key, value in enduse_nodes.items():
            eu_name = key.replace("enduse_", "").replace("_", " ").title()
            eu_color_key = key.replace("enduse_", "")
            color = SANKEY_COLOR_PALETTE.get(eu_color_key, "#A9A9A9")
            node = SankeyNode(
                node_id=key, label=eu_name,
                value_kwh=_round_val(value, 2), level=2, color=color,
            )
            nodes.append(node)
            node_map[key] = node

        # Pass 4: Create loss and rejected nodes (level 3)
        total_loss_kwh = Decimal("0")
        total_rejected_kwh = Decimal("0")
        for flow in flows:
            if flow.flow_type == EnergyFlowType.LOSS:
                total_loss_kwh += flow.value_kwh
            elif flow.flow_type == EnergyFlowType.REJECTED:
                total_rejected_kwh += flow.value_kwh

        if total_loss_kwh > Decimal("0"):
            loss_node = SankeyNode(
                node_id="losses", label="System Losses",
                value_kwh=_round_val(total_loss_kwh, 2), level=3,
                color=SANKEY_COLOR_PALETTE.get("losses", "#FF0000"),
            )
            nodes.append(loss_node)
            node_map["losses"] = loss_node

        if total_rejected_kwh > Decimal("0"):
            rejected_node = SankeyNode(
                node_id="rejected", label="Rejected Energy",
                value_kwh=_round_val(total_rejected_kwh, 2), level=3,
                color=SANKEY_COLOR_PALETTE.get("rejected", "#8B0000"),
            )
            nodes.append(rejected_node)
            node_map["rejected"] = rejected_node

        # Pass 5: Create storage node if present
        storage_kwh = Decimal("0")
        for flow in flows:
            if flow.flow_type == EnergyFlowType.STORAGE:
                storage_kwh += flow.value_kwh

        if storage_kwh > Decimal("0"):
            storage_node = SankeyNode(
                node_id="storage", label="Energy Storage",
                value_kwh=_round_val(storage_kwh, 2), level=1,
                color=SANKEY_COLOR_PALETTE.get("storage", "#9ACD32"),
            )
            nodes.append(storage_node)
            node_map["storage"] = storage_node

        # Pass 6: Unaccounted node
        unaccounted = total_input - total_output - total_loss_kwh - total_rejected_kwh - storage_kwh
        if unaccounted > Decimal("0"):
            ua_node = SankeyNode(
                node_id="unaccounted", label="Unaccounted",
                value_kwh=_round_val(unaccounted, 2), level=3,
                color=SANKEY_COLOR_PALETTE.get("unaccounted", "#FF69B4"),
            )
            nodes.append(ua_node)
            node_map["unaccounted"] = ua_node

        # Generate links
        links = self._generate_sankey_links(
            flows, node_map, source_nodes, conversion_nodes,
            enduse_nodes, total_loss_kwh, total_rejected_kwh,
            storage_kwh, unaccounted,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Sankey diagram generated: %d nodes, %d links, "
            "input=%.0f kWh, output=%.0f kWh (%.1f ms)",
            len(nodes), len(links),
            float(total_input), float(total_output), elapsed_ms,
        )

        return SankeyDiagram(
            nodes=nodes,
            links=links,
            title="Facility Energy Balance Sankey Diagram",
            total_input_kwh=_round_val(total_input, 2),
            total_output_kwh=_round_val(total_output, 2),
            unaccounted_kwh=_round_val(max(unaccounted, Decimal("0")), 2),
        )

    # ------------------------------------------------------------------ #
    # Public API: reconcile_meters                                        #
    # ------------------------------------------------------------------ #

    def reconcile_meters(
        self,
        meter_tree: List[MeterNode],
    ) -> List[MeterReconciliation]:
        """Reconcile parent meters against their sub-meters.

        Compares each parent meter reading to the sum of its child meter
        readings.  Classifies discrepancies as balanced (<= 2%),
        minor (2-5%), major (5-10%), or unreconciled (> 10%).

        Args:
            meter_tree: List of meter nodes (parent-child via IDs).

        Returns:
            List of MeterReconciliation results for each parent.
        """
        t0 = time.perf_counter()
        logger.info("Reconciling %d meters", len(meter_tree))

        if not meter_tree:
            return []

        # Build lookup
        meter_by_id: Dict[str, MeterNode] = {
            m.meter_id: m for m in meter_tree
        }

        # Find parent meters (those with children)
        parents_with_children: Dict[str, List[MeterNode]] = {}
        for meter in meter_tree:
            if meter.children:
                child_nodes = []
                for child_id in meter.children:
                    if child_id in meter_by_id:
                        child_nodes.append(meter_by_id[child_id])
                    else:
                        logger.warning(
                            "Child meter '%s' not found for parent '%s'",
                            child_id, meter.meter_id,
                        )
                if child_nodes:
                    parents_with_children[meter.meter_id] = child_nodes

        # Also find parents via parent_meter_id references
        for meter in meter_tree:
            if meter.parent_meter_id and meter.parent_meter_id in meter_by_id:
                parent = meter_by_id[meter.parent_meter_id]
                if parent.meter_id not in parents_with_children:
                    parents_with_children[parent.meter_id] = []
                # Avoid duplicates
                existing_ids = {
                    m.meter_id for m in
                    parents_with_children[parent.meter_id]
                }
                if meter.meter_id not in existing_ids:
                    parents_with_children[parent.meter_id].append(meter)

        # Perform reconciliation for each parent
        results: List[MeterReconciliation] = []
        for parent_id, children in parents_with_children.items():
            parent = meter_by_id[parent_id]
            recon = self._reconcile_single_parent(parent, children)
            results.append(recon)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Meter reconciliation complete: %d parents, %d balanced, "
            "%d minor, %d major, %d unreconciled (%.1f ms)",
            len(results),
            sum(1 for r in results if r.status == ReconciliationStatus.BALANCED),
            sum(1 for r in results if r.status == ReconciliationStatus.MINOR_DISCREPANCY),
            sum(1 for r in results if r.status == ReconciliationStatus.MAJOR_DISCREPANCY),
            sum(1 for r in results if r.status == ReconciliationStatus.UNRECONCILED),
            elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------ #
    # Public API: validate_meter_hierarchy                                #
    # ------------------------------------------------------------------ #

    def validate_meter_hierarchy(
        self,
        meters: List[MeterNode],
    ) -> Dict[str, Any]:
        """Validate completeness and correctness of the meter hierarchy.

        Checks for:
        - Tree completeness (all referenced parents/children exist)
        - No orphan meters (meters with no parent and not a MAIN type)
        - No circular references
        - Coverage percentage (sub-metered fraction of main meter)

        Args:
            meters: List of meter nodes.

        Returns:
            Dict with is_valid, warnings, errors, orphans, coverage_pct.
        """
        t0 = time.perf_counter()
        logger.info("Validating meter hierarchy: %d meters", len(meters))

        errors: List[str] = []
        warnings: List[str] = []
        orphans: List[str] = []
        meter_ids = {m.meter_id for m in meters}
        meter_by_id = {m.meter_id: m for m in meters}

        # Check 1: Find main meters
        main_meters = [m for m in meters if m.meter_type == MeterType.MAIN]
        if not main_meters:
            errors.append("No MAIN meter found in the hierarchy.")

        # Check 2: Verify parent references exist
        for meter in meters:
            if meter.parent_meter_id:
                if meter.parent_meter_id not in meter_ids:
                    errors.append(
                        f"Meter '{meter.meter_id}' references non-existent "
                        f"parent '{meter.parent_meter_id}'."
                    )

        # Check 3: Verify child references exist
        for meter in meters:
            for child_id in meter.children:
                if child_id not in meter_ids:
                    errors.append(
                        f"Meter '{meter.meter_id}' references non-existent "
                        f"child '{child_id}'."
                    )

        # Check 4: Find orphan meters (not MAIN and no parent)
        for meter in meters:
            if meter.meter_type != MeterType.MAIN and not meter.parent_meter_id:
                # Also check if any other meter lists this as a child
                is_child = any(
                    meter.meter_id in m.children for m in meters
                )
                if not is_child:
                    orphans.append(meter.meter_id)
                    warnings.append(
                        f"Meter '{meter.meter_id}' ({meter.meter_name}) is an "
                        f"orphan (no parent and not a MAIN meter)."
                    )

        # Check 5: Detect circular references
        circular = self._detect_circular_references(meters)
        if circular:
            for chain in circular:
                errors.append(
                    f"Circular reference detected: {' -> '.join(chain)}."
                )

        # Check 6: Calculate coverage percentage per main meter
        coverage_pct = Decimal("0")
        coverage_details: Dict[str, Decimal] = {}
        for main in main_meters:
            children = self._collect_direct_children(main.meter_id, meters)
            child_sum = sum(
                (meter_by_id[cid].reading_kwh for cid in children
                 if cid in meter_by_id), Decimal("0")
            )
            if main.reading_kwh > Decimal("0"):
                cov = _safe_pct(child_sum, main.reading_kwh)
                coverage_details[main.meter_id] = _round_val(cov, 2)
            else:
                coverage_details[main.meter_id] = Decimal("0")

        if coverage_details:
            coverage_pct = _safe_divide(
                sum(coverage_details.values()),
                _decimal(len(coverage_details)),
            )

        # Check 7: Calibration warnings
        for meter in meters:
            if meter.last_calibration:
                days_since = (date.today() - meter.last_calibration).days
                if days_since > 365:
                    warnings.append(
                        f"Meter '{meter.meter_id}' ({meter.meter_name}) "
                        f"calibration is {days_since} days old (> 12 months)."
                    )

        is_valid = len(errors) == 0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Meter hierarchy validation: valid=%s, errors=%d, warnings=%d, "
            "orphans=%d, coverage=%.1f%% (%.1f ms)",
            is_valid, len(errors), len(warnings), len(orphans),
            float(coverage_pct), elapsed_ms,
        )

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "orphans": orphans,
            "total_meters": len(meters),
            "main_meters": len(main_meters),
            "coverage_pct": str(_round_val(coverage_pct, 2)),
            "coverage_details": {
                k: str(v) for k, v in coverage_details.items()
            },
            "provenance_hash": _compute_hash({
                "total_meters": len(meters),
                "is_valid": is_valid,
                "error_count": len(errors),
                "orphan_count": len(orphans),
            }),
        }

    # ------------------------------------------------------------------ #
    # Public API: estimate_losses                                         #
    # ------------------------------------------------------------------ #

    def estimate_losses(
        self,
        balance: EnergyBalanceResult,
    ) -> List[LossEstimate]:
        """Estimate energy losses by category for the facility.

        Uses typical loss factors for common systems (distribution,
        transformation, unmetered loads) and attributes the unaccounted
        energy to likely loss categories.

        Args:
            balance: Preliminary EnergyBalanceResult (may have empty losses).

        Returns:
            List of LossEstimate objects for each identified category.
        """
        t0 = time.perf_counter()
        logger.info(
            "Estimating losses: total_input=%.0f kWh, unaccounted=%.0f kWh",
            float(balance.total_input_kwh), float(balance.unaccounted_kwh),
        )

        estimates: List[LossEstimate] = []
        total_input = balance.total_input_kwh

        if total_input <= Decimal("0"):
            return estimates

        # Identify energy sources present in flows
        sources_present = self._identify_sources(balance.energy_flows)

        # Electrical distribution losses
        if EnergySource.ELECTRICITY.value in sources_present:
            elec_input = sources_present[EnergySource.ELECTRICITY.value]
            dist_factor = self._get_loss_factor("electrical_distribution")
            dist_loss = elec_input * dist_factor
            estimates.append(LossEstimate(
                loss_category="Electrical distribution losses",
                estimated_kwh=_round_val(dist_loss, 2),
                estimation_method="typical_loss_factor",
                confidence_pct=Decimal("75"),
                notes=f"Estimated at {float(dist_factor) * 100:.1f}% of "
                      f"electrical input ({_round_val(elec_input, 0)} kWh). "
                      f"Source: ASHRAE/DOE reference data.",
            ))

            # Transformer losses
            tx_factor = self._get_loss_factor("transformer")
            tx_loss = elec_input * tx_factor
            estimates.append(LossEstimate(
                loss_category="Transformer losses",
                estimated_kwh=_round_val(tx_loss, 2),
                estimation_method="typical_loss_factor",
                confidence_pct=Decimal("80"),
                notes=f"Estimated at {float(tx_factor) * 100:.1f}% of "
                      f"electrical input. IEC 60076 reference.",
            ))

            # Power factor losses
            pf_factor = self._get_loss_factor("power_factor_losses")
            pf_loss = elec_input * pf_factor
            estimates.append(LossEstimate(
                loss_category="Power factor losses",
                estimated_kwh=_round_val(pf_loss, 2),
                estimation_method="typical_loss_factor",
                confidence_pct=Decimal("65"),
                notes=f"Estimated at {float(pf_factor) * 100:.1f}% of "
                      f"electrical input. Assumes uncorrected PF ~0.85.",
            ))

        # Gas / steam boiler losses
        gas_sources = {
            EnergySource.NATURAL_GAS.value,
            EnergySource.FUEL_OIL.value,
            EnergySource.LPG.value,
            EnergySource.BIOMASS.value,
            EnergySource.BIOGAS.value,
        }
        gas_input = sum(
            sources_present.get(s, Decimal("0")) for s in gas_sources
        )
        if gas_input > Decimal("0"):
            flue_factor = self._get_loss_factor("boiler_flue_gas")
            flue_loss = gas_input * flue_factor
            estimates.append(LossEstimate(
                loss_category="Boiler flue gas losses",
                estimated_kwh=_round_val(flue_loss, 2),
                estimation_method="typical_loss_factor",
                confidence_pct=Decimal("70"),
                notes=f"Estimated at {float(flue_factor) * 100:.1f}% of "
                      f"thermal fuel input ({_round_val(gas_input, 0)} kWh). "
                      f"Assumes conventional boiler efficiency.",
            ))

            rad_factor = self._get_loss_factor("boiler_radiation")
            rad_loss = gas_input * rad_factor
            estimates.append(LossEstimate(
                loss_category="Boiler radiation and convection losses",
                estimated_kwh=_round_val(rad_loss, 2),
                estimation_method="typical_loss_factor",
                confidence_pct=Decimal("70"),
                notes=f"Estimated at {float(rad_factor) * 100:.1f}% of "
                      f"thermal fuel input.",
            ))

            blow_factor = self._get_loss_factor("boiler_blowdown")
            blow_loss = gas_input * blow_factor
            estimates.append(LossEstimate(
                loss_category="Boiler blowdown losses",
                estimated_kwh=_round_val(blow_loss, 2),
                estimation_method="typical_loss_factor",
                confidence_pct=Decimal("65"),
                notes=f"Estimated at {float(blow_factor) * 100:.1f}% of "
                      f"thermal fuel input.",
            ))

        # Steam distribution losses
        if (EnergySource.STEAM.value in sources_present
                or gas_input > Decimal("0")):
            steam_base = sources_present.get(
                EnergySource.STEAM.value, gas_input
            )
            steam_dist_factor = self._get_loss_factor("steam_distribution")
            steam_loss = steam_base * steam_dist_factor
            estimates.append(LossEstimate(
                loss_category="Steam distribution losses",
                estimated_kwh=_round_val(steam_loss, 2),
                estimation_method="typical_loss_factor",
                confidence_pct=Decimal("65"),
                notes=f"Estimated at {float(steam_dist_factor) * 100:.1f}% "
                      f"of steam/thermal input. Includes pipe insulation "
                      f"and condensate losses.",
            ))

        # HVAC duct leakage
        has_hvac = any(
            f.end_use in (
                EndUseCategory.HVAC_HEATING,
                EndUseCategory.HVAC_COOLING,
                EndUseCategory.HVAC_VENTILATION,
            )
            for f in balance.energy_flows
            if f.end_use is not None
        )
        if has_hvac:
            hvac_output = sum(
                f.value_kwh for f in balance.energy_flows
                if f.flow_type == EnergyFlowType.OUTPUT
                and f.end_use in (
                    EndUseCategory.HVAC_HEATING,
                    EndUseCategory.HVAC_COOLING,
                    EndUseCategory.HVAC_VENTILATION,
                )
            )
            duct_factor = self._get_loss_factor("hvac_duct_leakage")
            duct_loss = hvac_output * duct_factor
            estimates.append(LossEstimate(
                loss_category="HVAC duct leakage losses",
                estimated_kwh=_round_val(duct_loss, 2),
                estimation_method="typical_loss_factor",
                confidence_pct=Decimal("60"),
                notes=f"Estimated at {float(duct_factor) * 100:.1f}% of "
                      f"HVAC output ({_round_val(hvac_output, 0)} kWh). "
                      f"Per ASHRAE 90.1 duct leakage reference.",
            ))

        # Standby / parasitic consumption
        standby_factor = self._get_loss_factor("standby_consumption")
        standby_loss = total_input * standby_factor
        estimates.append(LossEstimate(
            loss_category="Standby and parasitic consumption",
            estimated_kwh=_round_val(standby_loss, 2),
            estimation_method="typical_loss_factor",
            confidence_pct=Decimal("55"),
            notes=f"Estimated at {float(standby_factor) * 100:.1f}% of "
                  f"total input. Includes standby power, controls, "
                  f"and building management systems.",
        ))

        # Unmetered loads (residual)
        estimated_total = sum(e.estimated_kwh for e in estimates)
        residual = balance.unaccounted_kwh - estimated_total
        if residual > Decimal("0"):
            estimates.append(LossEstimate(
                loss_category="Unmetered / unidentified loads",
                estimated_kwh=_round_val(residual, 2),
                estimation_method="residual_calculation",
                confidence_pct=Decimal("40"),
                notes="Residual energy after deducting estimated losses "
                      "from unaccounted total. May include unmetered "
                      "equipment, meter inaccuracies, or data gaps.",
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Loss estimation complete: %d categories, total=%.0f kWh (%.1f ms)",
            len(estimates),
            float(sum(e.estimated_kwh for e in estimates)),
            elapsed_ms,
        )
        return estimates

    # ------------------------------------------------------------------ #
    # Public API: calculate_end_use_breakdown                             #
    # ------------------------------------------------------------------ #

    def calculate_end_use_breakdown(
        self,
        flows: List[EnergyFlow],
    ) -> Dict[str, Any]:
        """Calculate end-use energy breakdown from output flows.

        Groups all output flows by EndUseCategory and computes the
        percentage of each relative to total output.

        Args:
            flows: List of energy flows.

        Returns:
            Dict with categories, values, percentages, and provenance hash.
        """
        t0 = time.perf_counter()
        logger.info("Calculating end-use breakdown from %d flows", len(flows))

        output_flows = [
            f for f in flows if f.flow_type == EnergyFlowType.OUTPUT
        ]
        total_output = sum(
            (f.value_kwh for f in output_flows), Decimal("0")
        )

        breakdown: Dict[str, Decimal] = {}
        for flow in output_flows:
            eu = flow.end_use.value if flow.end_use else EndUseCategory.OTHER.value
            breakdown[eu] = breakdown.get(eu, Decimal("0")) + flow.value_kwh

        # Calculate percentages
        categories: List[Dict[str, Any]] = []
        for eu_key, value in sorted(
            breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            pct = _safe_pct(value, total_output)
            categories.append({
                "category": eu_key,
                "label": eu_key.replace("_", " ").title(),
                "value_kwh": str(_round_val(value, 2)),
                "percentage": str(_round_val(pct, 2)),
                "color": SANKEY_COLOR_PALETTE.get(eu_key, "#A9A9A9"),
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "End-use breakdown: %d categories, total=%.0f kWh (%.1f ms)",
            len(categories), float(total_output), elapsed_ms,
        )

        result = {
            "total_output_kwh": str(_round_val(total_output, 2)),
            "categories": categories,
            "category_count": len(categories),
            "top_category": categories[0]["category"] if categories else "",
            "top_category_pct": categories[0]["percentage"] if categories else "0",
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Public API: identify_unmetered_loads                                #
    # ------------------------------------------------------------------ #

    def identify_unmetered_loads(
        self,
        total_input: Decimal,
        metered_output: Decimal,
        known_losses: Decimal,
    ) -> Dict[str, Any]:
        """Identify and characterise unmetered loads in the facility.

        Calculates the gap between total input and the sum of metered output
        and known losses.  Provides breakdown of likely unmetered load
        categories with confidence estimates.

        Args:
            total_input: Total energy entering the facility (kWh).
            metered_output: Total metered end-use output (kWh).
            known_losses: Total known/estimated losses (kWh).

        Returns:
            Dict with unmetered load characterisation and recommendations.
        """
        t0 = time.perf_counter()
        logger.info(
            "Identifying unmetered loads: input=%.0f, metered=%.0f, "
            "losses=%.0f kWh",
            float(total_input), float(metered_output), float(known_losses),
        )

        unmetered_total = total_input - metered_output - known_losses
        unmetered_pct = _safe_pct(abs(unmetered_total), total_input)
        metered_pct = _safe_pct(metered_output, total_input)

        # Characterise likely unmetered loads
        likely_loads: List[Dict[str, Any]] = []
        if unmetered_total > Decimal("0"):
            # Plug loads / miscellaneous: typically 15-25% of unmetered
            plug_est = unmetered_total * Decimal("0.20")
            likely_loads.append({
                "category": "plug_loads_miscellaneous",
                "estimated_kwh": str(_round_val(plug_est, 2)),
                "fraction": "0.20",
                "confidence_pct": "50",
                "notes": "Desk equipment, chargers, vending machines.",
            })

            # Process equipment: typically 25-35% of unmetered
            process_est = unmetered_total * Decimal("0.30")
            likely_loads.append({
                "category": "unmetered_process_equipment",
                "estimated_kwh": str(_round_val(process_est, 2)),
                "fraction": "0.30",
                "confidence_pct": "45",
                "notes": "Production equipment without dedicated meters.",
            })

            # Building services: typically 10-20% of unmetered
            bldg_est = unmetered_total * Decimal("0.15")
            likely_loads.append({
                "category": "building_services",
                "estimated_kwh": str(_round_val(bldg_est, 2)),
                "fraction": "0.15",
                "confidence_pct": "50",
                "notes": "Lifts, fire systems, security, BMS.",
            })

            # Exterior / site: typically 5-10% of unmetered
            ext_est = unmetered_total * Decimal("0.10")
            likely_loads.append({
                "category": "exterior_site_loads",
                "estimated_kwh": str(_round_val(ext_est, 2)),
                "fraction": "0.10",
                "confidence_pct": "55",
                "notes": "Car park lighting, signage, irrigation.",
            })

            # Meter drift / inaccuracy: typically 5-15% of unmetered
            drift_est = unmetered_total * Decimal("0.10")
            likely_loads.append({
                "category": "meter_inaccuracy_drift",
                "estimated_kwh": str(_round_val(drift_est, 2)),
                "fraction": "0.10",
                "confidence_pct": "60",
                "notes": "Accumulated meter drift and accuracy tolerances.",
            })

            # Residual uncharacterised
            characterised = plug_est + process_est + bldg_est + ext_est + drift_est
            residual = unmetered_total - characterised
            if residual > Decimal("0"):
                likely_loads.append({
                    "category": "residual_uncharacterised",
                    "estimated_kwh": str(_round_val(residual, 2)),
                    "fraction": str(_round_val(
                        _safe_divide(residual, unmetered_total), 4
                    )),
                    "confidence_pct": "30",
                    "notes": "Remaining uncharacterised energy gap.",
                })

        # Recommendations
        recommendations: List[str] = []
        if unmetered_pct > Decimal("20"):
            recommendations.append(
                "CRITICAL: Unmetered load exceeds 20% of input. "
                "Install additional sub-metering immediately."
            )
        elif unmetered_pct > Decimal("10"):
            recommendations.append(
                "HIGH: Unmetered load exceeds 10%. Prioritise sub-metering "
                "for largest unmetered areas."
            )
        elif unmetered_pct > Decimal("5"):
            recommendations.append(
                "MODERATE: Unmetered load is 5-10%. Consider targeted "
                "sub-metering for process equipment."
            )
        else:
            recommendations.append(
                "GOOD: Unmetered load is below 5%. Metering coverage is "
                "adequate for ISO 50001 compliance."
            )

        if metered_pct < Decimal("80"):
            recommendations.append(
                "ISO 50001 Clause 6.3 recommends metering coverage of at "
                "least 80% of total energy use. Current coverage: "
                f"{_round_val(metered_pct, 1)}%."
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result: Dict[str, Any] = {
            "total_input_kwh": str(_round_val(total_input, 2)),
            "metered_output_kwh": str(_round_val(metered_output, 2)),
            "known_losses_kwh": str(_round_val(known_losses, 2)),
            "unmetered_total_kwh": str(_round_val(
                max(unmetered_total, Decimal("0")), 2
            )),
            "unmetered_pct": str(_round_val(unmetered_pct, 2)),
            "metered_coverage_pct": str(_round_val(metered_pct, 2)),
            "likely_loads": likely_loads,
            "recommendations": recommendations,
            "processing_time_ms": round(elapsed_ms),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Unmetered load identification: unmetered=%.0f kWh (%.1f%%), "
            "coverage=%.1f%% (%.1f ms)",
            float(max(unmetered_total, Decimal("0"))),
            float(unmetered_pct), float(metered_pct), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Public API: generate_balance_chart_data                             #
    # ------------------------------------------------------------------ #

    def generate_balance_chart_data(
        self,
        result: EnergyBalanceResult,
    ) -> Dict[str, Any]:
        """Generate chart data for visualising the energy balance.

        Produces data suitable for three chart types:
        - Pie chart: end-use breakdown
        - Bar chart: energy source comparison (input vs output by source)
        - Waterfall chart: balance flow (input -> output -> losses -> unaccounted)

        Args:
            result: Completed EnergyBalanceResult.

        Returns:
            Dict with pie_chart, bar_chart, waterfall_chart data and provenance.
        """
        t0 = time.perf_counter()
        logger.info("Generating balance chart data for facility=%s", result.facility_id)

        # -- Pie chart: end-use breakdown --
        pie_data = self._build_pie_chart_data(result.energy_flows)

        # -- Bar chart: source comparison --
        bar_data = self._build_bar_chart_data(result.energy_flows)

        # -- Waterfall chart: balance flow --
        waterfall_data = self._build_waterfall_chart_data(result)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        chart_result: Dict[str, Any] = {
            "facility_id": result.facility_id,
            "period_start": str(result.period_start),
            "period_end": str(result.period_end),
            "pie_chart": pie_data,
            "bar_chart": bar_data,
            "waterfall_chart": waterfall_data,
            "processing_time_ms": round(elapsed_ms),
        }
        chart_result["provenance_hash"] = _compute_hash(chart_result)

        logger.info(
            "Chart data generated: %d pie slices, %d bar groups, "
            "%d waterfall steps (%.1f ms)",
            len(pie_data.get("slices", [])),
            len(bar_data.get("groups", [])),
            len(waterfall_data.get("steps", [])),
            elapsed_ms,
        )
        return chart_result

    # ------------------------------------------------------------------ #
    # Internal: Flow Summation                                            #
    # ------------------------------------------------------------------ #

    def _sum_flows(
        self,
        flows: List[EnergyFlow],
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Sum energy flows by type: input, output, losses.

        Args:
            flows: List of energy flows.

        Returns:
            Tuple of (total_input, total_output, total_losses).
        """
        total_input = Decimal("0")
        total_output = Decimal("0")
        total_losses = Decimal("0")

        for flow in flows:
            if flow.flow_type == EnergyFlowType.INPUT:
                total_input += flow.value_kwh
            elif flow.flow_type == EnergyFlowType.OUTPUT:
                total_output += flow.value_kwh
            elif flow.flow_type in (
                EnergyFlowType.LOSS, EnergyFlowType.REJECTED,
            ):
                total_losses += flow.value_kwh
            # CONVERSION and STORAGE are internal transfers,
            # not added to input/output/loss totals.

        return total_input, total_output, total_losses

    # ------------------------------------------------------------------ #
    # Internal: Enrich Flow Percentages                                   #
    # ------------------------------------------------------------------ #

    def _enrich_flow_percentages(
        self,
        flows: List[EnergyFlow],
        total_input: Decimal,
    ) -> List[EnergyFlow]:
        """Update percentage_of_total on each flow relative to total input.

        Args:
            flows: Original flows.
            total_input: Total energy input (kWh).

        Returns:
            New list of flows with updated percentages.
        """
        enriched: List[EnergyFlow] = []
        for flow in flows:
            pct = _safe_pct(flow.value_kwh, total_input)
            enriched_flow = flow.model_copy(update={
                "percentage_of_total": _round_val(pct, 2),
            })
            enriched.append(enriched_flow)
        return enriched

    # ------------------------------------------------------------------ #
    # Internal: Sankey Link Generation                                    #
    # ------------------------------------------------------------------ #

    def _generate_sankey_links(
        self,
        flows: List[EnergyFlow],
        node_map: Dict[str, SankeyNode],
        source_nodes: Dict[str, Decimal],
        conversion_nodes: Dict[str, Decimal],
        enduse_nodes: Dict[str, Decimal],
        total_loss: Decimal,
        total_rejected: Decimal,
        storage_kwh: Decimal,
        unaccounted: Decimal,
    ) -> List[SankeyLink]:
        """Generate links between Sankey diagram nodes.

        Creates proportional links from sources to conversions/end-uses,
        and from end-uses to losses/rejected/unaccounted.

        Args:
            flows: All energy flows.
            node_map: Node ID to SankeyNode mapping.
            source_nodes: Source node values.
            conversion_nodes: Conversion node values.
            enduse_nodes: End-use node values.
            total_loss: Total system losses (kWh).
            total_rejected: Total rejected energy (kWh).
            storage_kwh: Total storage flow (kWh).
            unaccounted: Unaccounted energy (kWh).

        Returns:
            List of SankeyLink objects.
        """
        links: List[SankeyLink] = []
        total_source = sum(source_nodes.values())

        # Source -> Conversion links (if conversion nodes exist)
        if conversion_nodes:
            for src_key, src_val in source_nodes.items():
                source_type = src_key.replace("source_", "")
                conv_key = f"conv_{source_type}"
                if conv_key in node_map:
                    links.append(SankeyLink(
                        source_id=src_key,
                        target_id=conv_key,
                        value_kwh=_round_val(
                            min(src_val, node_map[conv_key].value_kwh), 2
                        ),
                        label=f"{source_type.replace('_', ' ').title()} to conversion",
                    ))

        # Source -> End-Use links (direct supply, proportional allocation)
        for src_key, src_val in source_nodes.items():
            source_type = src_key.replace("source_", "")
            conv_key = f"conv_{source_type}"
            # Allocate remaining source energy to end-uses
            allocated_to_conv = Decimal("0")
            if conv_key in node_map:
                allocated_to_conv = min(src_val, node_map[conv_key].value_kwh)

            remaining = src_val - allocated_to_conv
            if remaining <= Decimal("0"):
                continue

            total_enduse = sum(enduse_nodes.values())
            if total_enduse <= Decimal("0"):
                continue

            for eu_key, eu_val in enduse_nodes.items():
                proportion = _safe_divide(eu_val, total_enduse)
                link_value = remaining * proportion
                if link_value > Decimal("0"):
                    links.append(SankeyLink(
                        source_id=src_key,
                        target_id=eu_key,
                        value_kwh=_round_val(link_value, 2),
                        label=f"{source_type.replace('_', ' ').title()} "
                              f"to {eu_key.replace('enduse_', '').replace('_', ' ').title()}",
                    ))

        # Conversion -> End-Use links
        for conv_key, conv_val in conversion_nodes.items():
            total_enduse = sum(enduse_nodes.values())
            if total_enduse <= Decimal("0"):
                continue
            for eu_key, eu_val in enduse_nodes.items():
                proportion = _safe_divide(eu_val, total_enduse)
                link_value = conv_val * proportion
                if link_value > Decimal("0"):
                    links.append(SankeyLink(
                        source_id=conv_key,
                        target_id=eu_key,
                        value_kwh=_round_val(link_value, 2),
                        label=f"Converted to "
                              f"{eu_key.replace('enduse_', '').replace('_', ' ').title()}",
                    ))

        # Source -> Losses links (proportional to source contribution)
        if total_loss > Decimal("0") and "losses" in node_map:
            for src_key, src_val in source_nodes.items():
                proportion = _safe_divide(src_val, total_source)
                link_val = total_loss * proportion
                if link_val > Decimal("0"):
                    links.append(SankeyLink(
                        source_id=src_key,
                        target_id="losses",
                        value_kwh=_round_val(link_val, 2),
                        label="System losses",
                    ))

        # Source -> Rejected links
        if total_rejected > Decimal("0") and "rejected" in node_map:
            for src_key, src_val in source_nodes.items():
                proportion = _safe_divide(src_val, total_source)
                link_val = total_rejected * proportion
                if link_val > Decimal("0"):
                    links.append(SankeyLink(
                        source_id=src_key,
                        target_id="rejected",
                        value_kwh=_round_val(link_val, 2),
                        label="Rejected energy",
                    ))

        # Source -> Storage links
        if storage_kwh > Decimal("0") and "storage" in node_map:
            for src_key, src_val in source_nodes.items():
                proportion = _safe_divide(src_val, total_source)
                link_val = storage_kwh * proportion
                if link_val > Decimal("0"):
                    links.append(SankeyLink(
                        source_id=src_key,
                        target_id="storage",
                        value_kwh=_round_val(link_val, 2),
                        label="To storage",
                    ))

        # Source -> Unaccounted links
        if unaccounted > Decimal("0") and "unaccounted" in node_map:
            for src_key, src_val in source_nodes.items():
                proportion = _safe_divide(src_val, total_source)
                link_val = unaccounted * proportion
                if link_val > Decimal("0"):
                    links.append(SankeyLink(
                        source_id=src_key,
                        target_id="unaccounted",
                        value_kwh=_round_val(link_val, 2),
                        label="Unaccounted",
                    ))

        return links

    # ------------------------------------------------------------------ #
    # Internal: Single Parent Reconciliation                              #
    # ------------------------------------------------------------------ #

    def _reconcile_single_parent(
        self,
        parent: MeterNode,
        children: List[MeterNode],
    ) -> MeterReconciliation:
        """Reconcile a single parent meter against its child meters.

        Args:
            parent: Parent meter node.
            children: List of child meter nodes.

        Returns:
            MeterReconciliation result.
        """
        parent_reading = parent.reading_kwh
        sum_children = sum((c.reading_kwh for c in children), Decimal("0"))
        difference = parent_reading - sum_children
        difference_pct = _safe_pct(abs(difference), parent_reading)

        # Determine status
        if difference_pct <= self._balanced_threshold:
            status = ReconciliationStatus.BALANCED
        elif difference_pct <= self._minor_threshold:
            status = ReconciliationStatus.MINOR_DISCREPANCY
        elif difference_pct <= self._major_threshold:
            status = ReconciliationStatus.MAJOR_DISCREPANCY
        else:
            status = ReconciliationStatus.UNRECONCILED

        # Estimate distribution losses from discrepancy
        dist_factor = self._get_loss_factor("electrical_distribution")
        estimated_losses = parent_reading * dist_factor

        logger.debug(
            "Reconciliation: parent=%s (%.0f kWh), children_sum=%.0f kWh, "
            "diff=%.0f kWh (%.1f%%), status=%s",
            parent.meter_id, float(parent_reading), float(sum_children),
            float(difference), float(difference_pct), status.value,
        )

        return MeterReconciliation(
            parent_meter=parent,
            child_meters=children,
            parent_reading=_round_val(parent_reading, 2),
            sum_children=_round_val(sum_children, 2),
            difference=_round_val(difference, 2),
            difference_pct=_round_val(difference_pct, 2),
            status=status,
            estimated_losses=_round_val(estimated_losses, 2),
        )

    # ------------------------------------------------------------------ #
    # Internal: Circular Reference Detection                              #
    # ------------------------------------------------------------------ #

    def _detect_circular_references(
        self,
        meters: List[MeterNode],
    ) -> List[List[str]]:
        """Detect circular references in the meter hierarchy.

        Args:
            meters: List of meter nodes.

        Returns:
            List of circular reference chains (empty if none found).
        """
        meter_by_id = {m.meter_id: m for m in meters}
        circular_chains: List[List[str]] = []

        for meter in meters:
            visited: set = set()
            chain: List[str] = []
            current = meter

            while current:
                if current.meter_id in visited:
                    chain.append(current.meter_id)
                    circular_chains.append(chain[:])
                    break
                visited.add(current.meter_id)
                chain.append(current.meter_id)
                if current.parent_meter_id and current.parent_meter_id in meter_by_id:
                    current = meter_by_id[current.parent_meter_id]
                else:
                    break

        return circular_chains

    # ------------------------------------------------------------------ #
    # Internal: Collect Direct Children                                   #
    # ------------------------------------------------------------------ #

    def _collect_direct_children(
        self,
        parent_id: str,
        meters: List[MeterNode],
    ) -> List[str]:
        """Collect all direct child meter IDs for a parent.

        Args:
            parent_id: Parent meter identifier.
            meters: All meter nodes.

        Returns:
            List of child meter IDs.
        """
        children: List[str] = []
        meter_by_id = {m.meter_id: m for m in meters}

        # From parent's children list
        if parent_id in meter_by_id:
            parent = meter_by_id[parent_id]
            children.extend(parent.children)

        # From child's parent_meter_id reference
        for meter in meters:
            if meter.parent_meter_id == parent_id:
                if meter.meter_id not in children:
                    children.append(meter.meter_id)

        return children

    # ------------------------------------------------------------------ #
    # Internal: Loss Factor Lookup                                        #
    # ------------------------------------------------------------------ #

    def _get_loss_factor(self, system: str) -> Decimal:
        """Get the loss factor for a system, checking custom overrides first.

        Args:
            system: System name key.

        Returns:
            Loss factor as Decimal (0.0 to 1.0).
        """
        if system in self._custom_loss_factors:
            return self._custom_loss_factors[system]
        return TYPICAL_LOSS_FACTORS.get(system, Decimal("0.05"))

    # ------------------------------------------------------------------ #
    # Internal: Identify Sources from Flows                               #
    # ------------------------------------------------------------------ #

    def _identify_sources(
        self,
        flows: List[EnergyFlow],
    ) -> Dict[str, Decimal]:
        """Identify energy sources and their input quantities.

        Args:
            flows: List of energy flows.

        Returns:
            Dict mapping source value to total input kWh.
        """
        sources: Dict[str, Decimal] = {}
        for flow in flows:
            if flow.flow_type == EnergyFlowType.INPUT:
                key = flow.source.value
                sources[key] = sources.get(key, Decimal("0")) + flow.value_kwh
        return sources

    # ------------------------------------------------------------------ #
    # Internal: Pie Chart Data                                            #
    # ------------------------------------------------------------------ #

    def _build_pie_chart_data(
        self,
        flows: List[EnergyFlow],
    ) -> Dict[str, Any]:
        """Build pie chart data for end-use breakdown.

        Args:
            flows: Energy flows.

        Returns:
            Dict with slices, labels, values, colors.
        """
        output_flows = [
            f for f in flows if f.flow_type == EnergyFlowType.OUTPUT
        ]
        total_output = sum(
            (f.value_kwh for f in output_flows), Decimal("0")
        )

        breakdown: Dict[str, Decimal] = {}
        for flow in output_flows:
            eu = flow.end_use.value if flow.end_use else EndUseCategory.OTHER.value
            breakdown[eu] = breakdown.get(eu, Decimal("0")) + flow.value_kwh

        slices: List[Dict[str, Any]] = []
        for eu_key, value in sorted(
            breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            pct = _safe_pct(value, total_output)
            slices.append({
                "label": eu_key.replace("_", " ").title(),
                "value_kwh": str(_round_val(value, 2)),
                "percentage": str(_round_val(pct, 2)),
                "color": SANKEY_COLOR_PALETTE.get(eu_key, "#A9A9A9"),
            })

        return {
            "chart_type": "pie",
            "title": "Energy End-Use Breakdown",
            "total_kwh": str(_round_val(total_output, 2)),
            "slices": slices,
        }

    # ------------------------------------------------------------------ #
    # Internal: Bar Chart Data                                            #
    # ------------------------------------------------------------------ #

    def _build_bar_chart_data(
        self,
        flows: List[EnergyFlow],
    ) -> Dict[str, Any]:
        """Build bar chart data for energy source comparison.

        Groups flows by source and shows input vs output for each.

        Args:
            flows: Energy flows.

        Returns:
            Dict with groups, each containing input and output bars.
        """
        source_input: Dict[str, Decimal] = {}
        source_output: Dict[str, Decimal] = {}

        for flow in flows:
            key = flow.source.value
            if flow.flow_type == EnergyFlowType.INPUT:
                source_input[key] = (
                    source_input.get(key, Decimal("0")) + flow.value_kwh
                )
            elif flow.flow_type == EnergyFlowType.OUTPUT:
                source_output[key] = (
                    source_output.get(key, Decimal("0")) + flow.value_kwh
                )

        all_sources = set(source_input.keys()) | set(source_output.keys())
        groups: List[Dict[str, Any]] = []
        for src in sorted(all_sources):
            inp = source_input.get(src, Decimal("0"))
            out = source_output.get(src, Decimal("0"))
            groups.append({
                "source": src.replace("_", " ").title(),
                "input_kwh": str(_round_val(inp, 2)),
                "output_kwh": str(_round_val(out, 2)),
                "efficiency_pct": str(_round_val(_safe_pct(out, inp), 2)),
                "color": SANKEY_COLOR_PALETTE.get(src, "#C0C0C0"),
            })

        return {
            "chart_type": "bar",
            "title": "Energy Source Input vs Output",
            "groups": groups,
        }

    # ------------------------------------------------------------------ #
    # Internal: Waterfall Chart Data                                      #
    # ------------------------------------------------------------------ #

    def _build_waterfall_chart_data(
        self,
        result: EnergyBalanceResult,
    ) -> Dict[str, Any]:
        """Build waterfall chart data showing the energy balance flow.

        Steps: Total Input -> Output Uses -> Losses -> Rejected -> Unaccounted.

        Args:
            result: EnergyBalanceResult.

        Returns:
            Dict with ordered steps for a waterfall chart.
        """
        steps: List[Dict[str, Any]] = []

        # Starting point: total input
        steps.append({
            "label": "Total Energy Input",
            "value_kwh": str(_round_val(result.total_input_kwh, 2)),
            "cumulative_kwh": str(_round_val(result.total_input_kwh, 2)),
            "type": "total",
            "color": "#2196F3",
        })

        # Subtract output by end-use category
        output_breakdown: Dict[str, Decimal] = {}
        for flow in result.energy_flows:
            if flow.flow_type == EnergyFlowType.OUTPUT:
                eu = flow.end_use.value if flow.end_use else EndUseCategory.OTHER.value
                output_breakdown[eu] = (
                    output_breakdown.get(eu, Decimal("0")) + flow.value_kwh
                )

        running = result.total_input_kwh
        for eu_key, value in sorted(
            output_breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            running -= value
            steps.append({
                "label": eu_key.replace("_", " ").title(),
                "value_kwh": str(_round_val(-value, 2)),
                "cumulative_kwh": str(_round_val(running, 2)),
                "type": "decrease",
                "color": SANKEY_COLOR_PALETTE.get(eu_key, "#4CAF50"),
            })

        # Subtract losses
        loss_kwh = result.total_losses_kwh
        if loss_kwh > Decimal("0"):
            running -= loss_kwh
            steps.append({
                "label": "System Losses",
                "value_kwh": str(_round_val(-loss_kwh, 2)),
                "cumulative_kwh": str(_round_val(running, 2)),
                "type": "decrease",
                "color": "#FF5722",
            })

        # Unaccounted
        if result.unaccounted_kwh > Decimal("0"):
            running -= result.unaccounted_kwh
            steps.append({
                "label": "Unaccounted Energy",
                "value_kwh": str(_round_val(-result.unaccounted_kwh, 2)),
                "cumulative_kwh": str(_round_val(running, 2)),
                "type": "decrease",
                "color": "#FF69B4",
            })
        elif result.unaccounted_kwh < Decimal("0"):
            running -= result.unaccounted_kwh
            steps.append({
                "label": "Measurement Surplus",
                "value_kwh": str(_round_val(-result.unaccounted_kwh, 2)),
                "cumulative_kwh": str(_round_val(running, 2)),
                "type": "increase",
                "color": "#FF69B4",
            })

        # Final balance (should be near zero)
        steps.append({
            "label": "Balance Check",
            "value_kwh": str(_round_val(running, 2)),
            "cumulative_kwh": str(_round_val(running, 2)),
            "type": "total",
            "color": "#607D8B",
        })

        return {
            "chart_type": "waterfall",
            "title": "Energy Balance Waterfall",
            "steps": steps,
        }

    # ------------------------------------------------------------------ #
    # Utility: Energy Unit Conversion                                     #
    # ------------------------------------------------------------------ #

    def convert_energy(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """Convert energy value between units via kWh intermediate.

        Supported units: kwh, mj, gj, therms, mmbtu, btu, mwh.

        Args:
            value: Energy value in source units.
            from_unit: Source unit (lowercase).
            to_unit: Target unit (lowercase).

        Returns:
            Converted value, rounded to 6 decimal places.
        """
        if from_unit == to_unit:
            return value

        # Convert from source to kWh
        to_kwh_key = f"{from_unit}_to_kwh"
        if from_unit == "kwh":
            kwh_value = value
        elif to_kwh_key in ENERGY_CONVERSION_FACTORS:
            kwh_value = value * ENERGY_CONVERSION_FACTORS[to_kwh_key]
        else:
            logger.warning(
                "Unknown source unit '%s'; returning original value.", from_unit
            )
            return value

        # Convert from kWh to target
        from_kwh_key = f"kwh_to_{to_unit}"
        if to_unit == "kwh":
            return _round_val(kwh_value, 6)
        elif from_kwh_key in ENERGY_CONVERSION_FACTORS:
            return _round_val(
                kwh_value * ENERGY_CONVERSION_FACTORS[from_kwh_key], 6
            )
        else:
            logger.warning(
                "Unknown target unit '%s'; returning kWh value.", to_unit
            )
            return _round_val(kwh_value, 6)

    # ------------------------------------------------------------------ #
    # Utility: Conversion Efficiency Lookup                               #
    # ------------------------------------------------------------------ #

    def get_conversion_efficiency(
        self,
        system_type: str,
    ) -> Decimal:
        """Look up typical conversion efficiency for a system type.

        Checks custom overrides first, then default reference data.

        Args:
            system_type: System type key (e.g. 'boiler_natural_gas').

        Returns:
            Efficiency factor (Decimal), or 1.0 if not found.
        """
        if system_type in self._custom_efficiencies:
            return self._custom_efficiencies[system_type]
        return _CONVERSION_EFFICIENCIES.get(system_type, Decimal("1.0"))

    # ------------------------------------------------------------------ #
    # Utility: Validate Energy Conservation                               #
    # ------------------------------------------------------------------ #

    def validate_conservation(
        self,
        flows: List[EnergyFlow],
        tolerance_pct: Decimal = Decimal("5.0"),
    ) -> Dict[str, Any]:
        """Validate energy conservation (first law of thermodynamics).

        Checks that total input >= total output + total losses, within
        the specified tolerance.

        Args:
            flows: List of energy flows.
            tolerance_pct: Acceptable imbalance percentage.

        Returns:
            Dict with is_valid, imbalance_kwh, imbalance_pct, details.
        """
        total_input, total_output, total_losses = self._sum_flows(flows)
        total_demand = total_output + total_losses
        imbalance = total_input - total_demand
        imbalance_pct = _safe_pct(abs(imbalance), total_input)

        is_valid = imbalance_pct <= tolerance_pct
        is_surplus = imbalance >= Decimal("0")

        details: List[str] = [
            f"Total input: {_round_val(total_input, 2)} kWh",
            f"Total output: {_round_val(total_output, 2)} kWh",
            f"Total losses: {_round_val(total_losses, 2)} kWh",
            f"Imbalance: {_round_val(imbalance, 2)} kWh "
            f"({'surplus' if is_surplus else 'deficit'})",
            f"Imbalance: {_round_val(imbalance_pct, 2)}% "
            f"(tolerance: {_round_val(tolerance_pct, 2)}%)",
        ]

        if not is_valid:
            if is_surplus:
                details.append(
                    "WARNING: Energy surplus exceeds tolerance. Possible "
                    "causes: missing output flows, overestimated inputs, "
                    "or significant unmetered loads."
                )
            else:
                details.append(
                    "WARNING: Energy deficit exceeds tolerance. Possible "
                    "causes: missing input flows, overestimated outputs, "
                    "or generation sources not captured."
                )

        result = {
            "is_valid": is_valid,
            "total_input_kwh": str(_round_val(total_input, 2)),
            "total_output_kwh": str(_round_val(total_output, 2)),
            "total_losses_kwh": str(_round_val(total_losses, 2)),
            "imbalance_kwh": str(_round_val(imbalance, 2)),
            "imbalance_pct": str(_round_val(imbalance_pct, 2)),
            "tolerance_pct": str(_round_val(tolerance_pct, 2)),
            "is_surplus": is_surplus,
            "details": details,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Utility: Meter Accuracy Impact                                      #
    # ------------------------------------------------------------------ #

    def calculate_meter_accuracy_impact(
        self,
        meters: List[MeterNode],
    ) -> Dict[str, Any]:
        """Calculate the combined accuracy impact of the metering system.

        Determines the worst-case measurement uncertainty based on
        individual meter accuracy classes (root-sum-square combination).

        Args:
            meters: List of meter nodes.

        Returns:
            Dict with combined accuracy, per-meter details, and recommendations.
        """
        if not meters:
            return {
                "combined_accuracy_pct": "0",
                "meter_details": [],
                "recommendations": [],
                "provenance_hash": _compute_hash({"empty": True}),
            }

        details: List[Dict[str, Any]] = []
        sum_sq_error = Decimal("0")
        total_energy = Decimal("0")

        for meter in meters:
            accuracy = meter.accuracy_class
            error_kwh = meter.reading_kwh * accuracy / Decimal("100")
            sum_sq_error += error_kwh * error_kwh
            total_energy += meter.reading_kwh

            # Find matching IEC class
            iec_class = "unknown"
            for class_key, class_data in METER_ACCURACY_STANDARDS.items():
                if class_data["accuracy_pct"] == accuracy:
                    iec_class = class_key
                    break

            details.append({
                "meter_id": meter.meter_id,
                "meter_name": meter.meter_name,
                "accuracy_class": str(accuracy),
                "iec_standard": iec_class,
                "reading_kwh": str(_round_val(meter.reading_kwh, 2)),
                "max_error_kwh": str(_round_val(error_kwh, 2)),
            })

        # Root-sum-square combined uncertainty
        combined_error_kwh = _decimal(math.sqrt(float(sum_sq_error)))
        combined_pct = _safe_pct(combined_error_kwh, total_energy)

        recommendations: List[str] = []
        for meter in meters:
            if meter.accuracy_class > Decimal("2.0"):
                recommendations.append(
                    f"Meter '{meter.meter_id}' ({meter.meter_name}) has "
                    f"accuracy class {meter.accuracy_class}%. Consider "
                    f"upgrading to IEC Class 1.0 or better."
                )
            if meter.meter_type == MeterType.MAIN and meter.accuracy_class > Decimal("0.5"):
                recommendations.append(
                    f"MAIN meter '{meter.meter_id}' should be Class 0.5 or "
                    f"better for revenue-grade accuracy. Current: "
                    f"{meter.accuracy_class}%."
                )

        if combined_pct > Decimal("5.0"):
            recommendations.append(
                "Combined metering uncertainty exceeds 5%. Reconciliation "
                "results may be unreliable. Prioritise meter upgrades."
            )

        result = {
            "combined_accuracy_pct": str(_round_val(combined_pct, 2)),
            "combined_error_kwh": str(_round_val(combined_error_kwh, 2)),
            "total_metered_kwh": str(_round_val(total_energy, 2)),
            "meter_count": len(meters),
            "meter_details": details,
            "recommendations": recommendations,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Utility: Summary Statistics                                         #
    # ------------------------------------------------------------------ #

    def summarise_balance(
        self,
        result: EnergyBalanceResult,
    ) -> Dict[str, Any]:
        """Produce a summary dictionary from an energy balance result.

        Args:
            result: Completed EnergyBalanceResult.

        Returns:
            Summary dict with key metrics, status, and recommendations.
        """
        # Determine overall health
        health = "GOOD"
        if result.unaccounted_pct > Decimal("10"):
            health = "POOR"
        elif result.unaccounted_pct > Decimal("5"):
            health = "FAIR"

        # Count reconciliation issues
        recon_issues = sum(
            1 for r in result.meter_reconciliations
            if r.status in (
                ReconciliationStatus.MAJOR_DISCREPANCY,
                ReconciliationStatus.UNRECONCILED,
            )
        )

        # Top loss category
        top_loss = ""
        top_loss_kwh = Decimal("0")
        for loss in result.loss_estimates:
            if loss.estimated_kwh > top_loss_kwh:
                top_loss_kwh = loss.estimated_kwh
                top_loss = loss.loss_category

        recommendations: List[str] = []
        if health == "POOR":
            recommendations.append(
                "Unaccounted energy exceeds 10% of input. Conduct a "
                "detailed sub-metering audit."
            )
        if recon_issues > 0:
            recommendations.append(
                f"{recon_issues} meter reconciliation issue(s) detected. "
                f"Investigate discrepancies and verify meter calibration."
            )
        if result.balance_efficiency_pct < Decimal("60"):
            recommendations.append(
                "Overall balance efficiency is below 60%. Review major "
                "loss categories for improvement opportunities."
            )

        summary = {
            "balance_id": result.balance_id,
            "facility_id": result.facility_id,
            "period": f"{result.period_start} to {result.period_end}",
            "total_input_kwh": str(_round_val(result.total_input_kwh, 2)),
            "total_output_kwh": str(_round_val(result.total_output_kwh, 2)),
            "total_losses_kwh": str(_round_val(result.total_losses_kwh, 2)),
            "unaccounted_kwh": str(_round_val(result.unaccounted_kwh, 2)),
            "unaccounted_pct": str(_round_val(result.unaccounted_pct, 2)),
            "balance_efficiency_pct": str(
                _round_val(result.balance_efficiency_pct, 2)
            ),
            "health": health,
            "flow_count": len(result.energy_flows),
            "reconciliation_count": len(result.meter_reconciliations),
            "reconciliation_issues": recon_issues,
            "loss_estimate_count": len(result.loss_estimates),
            "top_loss_category": top_loss,
            "top_loss_kwh": str(_round_val(top_loss_kwh, 2)),
            "recommendations": recommendations,
            "engine_version": self.engine_version,
            "calculation_time_ms": result.calculation_time_ms,
        }
        summary["provenance_hash"] = _compute_hash(summary)
        return summary


# ---------------------------------------------------------------------------
# Pydantic v2 model rebuilds (required with `from __future__ import annotations`)
# ---------------------------------------------------------------------------

EnergyFlow.model_rebuild()
MeterNode.model_rebuild()
SankeyNode.model_rebuild()
SankeyLink.model_rebuild()
SankeyDiagram.model_rebuild()
MeterReconciliation.model_rebuild()
LossEstimate.model_rebuild()
EnergyBalanceResult.model_rebuild()
