# -*- coding: utf-8 -*-
"""
MeterRegistryEngine - PACK-039 Energy Monitoring Engine 1
==========================================================

Comprehensive meter asset management engine for energy monitoring programmes.
Handles meter registration, hierarchical topology (site > building > floor >
system > circuit), calibration tracking, CT/PT ratio management, virtual
meter formula definition, and data channel configuration.

Calculation Methodology:
    Virtual Meter Formula Evaluation:
        result = SUM(component_meter_values * coefficient)
        Example: virtual_kW = meter_A_kW + meter_B_kW - meter_C_kW

    CT/PT Ratio Scaling:  scaled_value = raw_value * CT_ratio * PT_ratio
    Calibration Drift:    drift_pct = |actual - reference| / reference * 100
    Hierarchy Completeness: meters_with_parent / total_meters * 100
    Coverage Ratio:       sum(submeter_capacity) / main_meter_capacity * 100

Regulatory References:
    ANSI C12.1, IEC 62053, ASHRAE Guideline 14-2014, ISO 50001:2018,
    IPMVP Volume I, IEC 61968/CIM, ANSI C12.20

Zero-Hallucination:
    All calculations use deterministic Decimal arithmetic.  No LLM
    involvement in any calculation path.  Virtual meter formulas
    evaluated from stored coefficients only.  SHA-256 provenance hash
    on every result.

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-039 Energy Monitoring
Engine:  1 of 5
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
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


class MeterType(str, Enum):
    """Classification of metering device purpose.

    REVENUE:    Utility revenue-grade meter for billing.
    CHECK:      Check meter for billing verification.
    SUBMETER:   Sub-metering for internal allocation.
    VIRTUAL:    Calculated meter from formula of other meters.
    TEMPORARY:  Temporary measurement campaign meter.
    """
    REVENUE = "revenue"
    CHECK = "check"
    SUBMETER = "submeter"
    VIRTUAL = "virtual"
    TEMPORARY = "temporary"


class MeterProtocol(str, Enum):
    """Communication protocol for meter data retrieval."""
    MODBUS_RTU = "modbus_rtu"
    MODBUS_TCP = "modbus_tcp"
    BACNET_IP = "bacnet_ip"
    BACNET_MSTP = "bacnet_mstp"
    MQTT = "mqtt"
    OPCUA = "opcua"
    PULSE = "pulse"
    MANUAL = "manual"


class EnergyType(str, Enum):
    """Type of energy commodity metered."""
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    STEAM = "steam"
    CHILLED_WATER = "chilled_water"
    HOT_WATER = "hot_water"
    FUEL_OIL = "fuel_oil"
    PROPANE = "propane"
    DISTRICT_HEAT = "district_heat"


class MeterStatus(str, Enum):
    """Operational status of a meter asset.

    ACTIVE:            Meter is operational and reporting data.
    INACTIVE:          Meter is installed but not currently active.
    CALIBRATION_DUE:   Calibration period has expired.
    FAULT:             Meter is reporting errors or offline.
    DECOMMISSIONED:    Meter has been permanently removed.
    """
    ACTIVE = "active"
    INACTIVE = "inactive"
    CALIBRATION_DUE = "calibration_due"
    FAULT = "fault"
    DECOMMISSIONED = "decommissioned"


class ChannelType(str, Enum):
    """Measurement channel type available on a meter."""
    KW = "kw"
    KWH = "kwh"
    KVAR = "kvar"
    KVARH = "kvarh"
    KVA = "kva"
    VOLTAGE = "voltage"
    CURRENT = "current"
    PF = "pf"
    THERM = "therm"
    M3 = "m3"
    FLOW = "flow"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"


class HierarchyLevel(str, Enum):
    """Level within the meter hierarchy tree.

    SITE:      Top-level site / campus.
    BUILDING:  Individual building.
    FLOOR:     Building floor / level.
    SYSTEM:    Mechanical / electrical system.
    CIRCUIT:   Individual electrical circuit.
    """
    SITE = "site"
    BUILDING = "building"
    FLOOR = "floor"
    SYSTEM = "system"
    CIRCUIT = "circuit"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ANSI C12.20 accuracy class limits (percentage error).
ACCURACY_CLASS_LIMITS: Dict[str, Decimal] = {
    "0.1": Decimal("0.1"),
    "0.2": Decimal("0.2"),
    "0.5": Decimal("0.5"),
    "1.0": Decimal("1.0"),
    "2.0": Decimal("2.0"),
}

# Default calibration interval in months by meter type.
CALIBRATION_INTERVAL_MONTHS: Dict[str, int] = {
    MeterType.REVENUE.value: 12,
    MeterType.CHECK.value: 24,
    MeterType.SUBMETER.value: 36,
    MeterType.VIRTUAL.value: 0,
    MeterType.TEMPORARY.value: 6,
}

# Default CT/PT ratios for common transformer sizes.
COMMON_CT_RATIOS: List[Decimal] = [
    Decimal("100"), Decimal("200"), Decimal("400"),
    Decimal("600"), Decimal("800"), Decimal("1000"),
    Decimal("1200"), Decimal("1500"), Decimal("2000"),
    Decimal("3000"), Decimal("4000"), Decimal("5000"),
]

COMMON_PT_RATIOS: List[Decimal] = [
    Decimal("1"), Decimal("2"), Decimal("20"),
    Decimal("40"), Decimal("60"), Decimal("100"),
    Decimal("200"), Decimal("350"),
]

# Hierarchy depth limits.
MAX_HIERARCHY_DEPTH: int = 10

# Maximum channels per meter.
MAX_CHANNELS_PER_METER: int = 32


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class MeterChannel(BaseModel):
    """Configuration of a single measurement channel on a meter.

    Attributes:
        channel_id:   Unique channel identifier.
        channel_type: Measurement type (kW, kWh, etc.).
        unit:         Engineering unit string.
        multiplier:   Scaling multiplier (CT*PT ratio).
        offset:       Calibration offset.
        min_value:    Minimum valid reading.
        max_value:    Maximum valid reading.
        is_cumulative: Whether channel reports cumulative values.
        register_address: Protocol register address.
    """
    channel_id: str = Field(default_factory=_new_uuid, description="Channel ID")
    channel_type: ChannelType = Field(
        default=ChannelType.KWH, description="Channel measurement type"
    )
    unit: str = Field(default="kWh", max_length=32, description="Engineering unit")
    multiplier: Decimal = Field(
        default=Decimal("1"), description="CT*PT scaling multiplier"
    )
    offset: Decimal = Field(
        default=Decimal("0"), description="Calibration offset"
    )
    min_value: Decimal = Field(
        default=Decimal("0"), description="Minimum valid reading"
    )
    max_value: Decimal = Field(
        default=Decimal("999999999"), description="Maximum valid reading"
    )
    is_cumulative: bool = Field(
        default=False, description="True if cumulative register"
    )
    register_address: int = Field(
        default=0, ge=0, description="Protocol register address"
    )


class MeterConfig(BaseModel):
    """Complete configuration for a metering device.

    Attributes:
        meter_id:           Unique meter identifier.
        meter_name:         Human-readable meter name.
        meter_type:         Classification of meter purpose.
        protocol:           Communication protocol.
        energy_type:        Energy commodity metered.
        status:             Operational status.
        serial_number:      Manufacturer serial number.
        manufacturer:       Meter manufacturer.
        model:              Meter model number.
        accuracy_class:     ANSI accuracy class.
        ct_ratio:           Current transformer ratio.
        pt_ratio:           Potential transformer ratio.
        rated_capacity_kw:  Rated maximum capacity (kW).
        location:           Physical installation location.
        parent_meter_id:    Parent meter ID in hierarchy.
        hierarchy_level:    Level in meter hierarchy.
        channels:           List of data channels.
        install_date:       Installation date.
        last_calibration:   Last calibration date.
        calibration_interval_months: Months between calibrations.
        notes:              Additional notes.
    """
    meter_id: str = Field(default_factory=_new_uuid, description="Meter ID")
    meter_name: str = Field(default="", max_length=200, description="Meter name")
    meter_type: MeterType = Field(
        default=MeterType.SUBMETER, description="Meter type"
    )
    protocol: MeterProtocol = Field(
        default=MeterProtocol.MODBUS_TCP, description="Protocol"
    )
    energy_type: EnergyType = Field(
        default=EnergyType.ELECTRICITY, description="Energy type"
    )
    status: MeterStatus = Field(
        default=MeterStatus.ACTIVE, description="Status"
    )
    serial_number: str = Field(default="", max_length=100, description="Serial")
    manufacturer: str = Field(default="", max_length=200, description="Manufacturer")
    model: str = Field(default="", max_length=200, description="Model")
    accuracy_class: str = Field(default="0.5", max_length=10, description="Accuracy")
    ct_ratio: Decimal = Field(default=Decimal("1"), description="CT ratio")
    pt_ratio: Decimal = Field(default=Decimal("1"), description="PT ratio")
    rated_capacity_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Rated capacity kW"
    )
    location: str = Field(default="", max_length=500, description="Location")
    parent_meter_id: Optional[str] = Field(
        default=None, description="Parent meter ID"
    )
    hierarchy_level: HierarchyLevel = Field(
        default=HierarchyLevel.BUILDING, description="Hierarchy level"
    )
    channels: List[MeterChannel] = Field(
        default_factory=list, description="Channels"
    )
    install_date: Optional[datetime] = Field(
        default=None, description="Install date"
    )
    last_calibration: Optional[datetime] = Field(
        default=None, description="Last calibration"
    )
    calibration_interval_months: int = Field(
        default=36, ge=0, description="Calibration interval months"
    )
    notes: str = Field(default="", max_length=2000, description="Notes")


class CalibrationRecord(BaseModel):
    """Record of a meter calibration event.

    Attributes:
        calibration_id:      Unique calibration identifier.
        meter_id:            Meter being calibrated.
        calibration_date:    Date of calibration.
        next_due_date:       Next calibration due date.
        technician:          Calibrating technician.
        reference_standard:  Reference standard used.
        test_points:         Test point results (load% -> error%).
        overall_error_pct:   Overall measured error percentage.
        accuracy_class:      Accuracy class verified.
        passed:              Whether calibration passed.
        drift_pct:           Drift since last calibration.
        ct_ratio_verified:   CT ratio verified value.
        pt_ratio_verified:   PT ratio verified value.
        certificate_number:  Calibration certificate number.
        notes:               Calibration notes.
        provenance_hash:     SHA-256 audit hash.
    """
    calibration_id: str = Field(default_factory=_new_uuid)
    meter_id: str = Field(default="")
    calibration_date: datetime = Field(default_factory=_utcnow)
    next_due_date: Optional[datetime] = Field(default=None)
    technician: str = Field(default="", max_length=200)
    reference_standard: str = Field(default="", max_length=200)
    test_points: Dict[str, Decimal] = Field(default_factory=dict)
    overall_error_pct: Decimal = Field(default=Decimal("0"))
    accuracy_class: str = Field(default="0.5")
    passed: bool = Field(default=True)
    drift_pct: Decimal = Field(default=Decimal("0"))
    ct_ratio_verified: Decimal = Field(default=Decimal("1"))
    pt_ratio_verified: Decimal = Field(default=Decimal("1"))
    certificate_number: str = Field(default="", max_length=100)
    notes: str = Field(default="", max_length=2000)
    provenance_hash: str = Field(default="")


class MeterHierarchy(BaseModel):
    """Representation of the meter hierarchy tree.

    Attributes:
        hierarchy_id:        Unique hierarchy identifier.
        site_id:             Site this hierarchy belongs to.
        site_name:           Site name.
        total_meters:        Total meters in hierarchy.
        active_meters:       Number of active meters.
        hierarchy_depth:     Maximum depth of tree.
        coverage_pct:        Sub-metering coverage percentage.
        tree:                Nested hierarchy structure.
        orphan_meters:       Meters without a valid parent.
        completeness_pct:    Hierarchy completeness percentage.
        calculated_at:       Calculation timestamp.
        provenance_hash:     SHA-256 audit hash.
    """
    hierarchy_id: str = Field(default_factory=_new_uuid)
    site_id: str = Field(default="")
    site_name: str = Field(default="", max_length=200)
    total_meters: int = Field(default=0, ge=0)
    active_meters: int = Field(default=0, ge=0)
    hierarchy_depth: int = Field(default=0, ge=0)
    coverage_pct: Decimal = Field(default=Decimal("0"))
    tree: Dict[str, Any] = Field(default_factory=dict)
    orphan_meters: List[str] = Field(default_factory=list)
    completeness_pct: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class VirtualMeterDefinition(BaseModel):
    """Definition of a virtual (calculated) meter.

    Attributes:
        virtual_meter_id:  The virtual meter's ID.
        formula_name:      Human-readable formula name.
        components:        Dict of meter_id -> coefficient.
        operator:          Aggregation operator (SUM / DIFF / AVG).
        description:       Formula description.
        created_at:        Creation timestamp.
        provenance_hash:   SHA-256 audit hash.
    """
    virtual_meter_id: str = Field(default_factory=_new_uuid)
    formula_name: str = Field(default="", max_length=200)
    components: Dict[str, Decimal] = Field(default_factory=dict)
    operator: str = Field(default="SUM", max_length=10)
    description: str = Field(default="", max_length=1000)
    created_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class MeterRegistryResult(BaseModel):
    """Complete meter registry analysis result.

    Attributes:
        result_id:              Unique result identifier.
        site_id:                Site identifier.
        site_name:              Site name.
        total_meters:           Total registered meters.
        active_meters:          Active meters count.
        virtual_meters:         Virtual meters count.
        meters_calibration_due: Meters with calibration overdue.
        hierarchy:              Computed hierarchy.
        registered_meters:      List of registered meter configs.
        virtual_definitions:    List of virtual meter definitions.
        calibration_records:    Recent calibration records.
        coverage_pct:           Sub-metering coverage percentage.
        energy_types_covered:   Energy types with metering.
        protocol_summary:       Count of meters by protocol.
        warnings:               List of warnings.
        recommendations:        List of recommendations.
        processing_time_ms:     Processing duration milliseconds.
        calculated_at:          Calculation timestamp.
        provenance_hash:        SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    site_id: str = Field(default="")
    site_name: str = Field(default="", max_length=500)
    total_meters: int = Field(default=0, ge=0)
    active_meters: int = Field(default=0, ge=0)
    virtual_meters: int = Field(default=0, ge=0)
    meters_calibration_due: int = Field(default=0, ge=0)
    hierarchy: MeterHierarchy = Field(default_factory=MeterHierarchy)
    registered_meters: List[MeterConfig] = Field(default_factory=list)
    virtual_definitions: List[VirtualMeterDefinition] = Field(default_factory=list)
    calibration_records: List[CalibrationRecord] = Field(default_factory=list)
    coverage_pct: Decimal = Field(default=Decimal("0"))
    energy_types_covered: List[str] = Field(default_factory=list)
    protocol_summary: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MeterRegistryEngine:
    """Meter asset management engine for energy monitoring programmes.

    Manages meter registration, hierarchical topology, calibration tracking,
    CT/PT ratio management, virtual meter formulas, and data channel
    configuration.  All calculations use deterministic Decimal arithmetic
    with SHA-256 provenance hashing.

    Usage::

        engine = MeterRegistryEngine()
        meter = engine.register_meter(
            meter_name="Main Switchboard",
            meter_type=MeterType.REVENUE,
            energy_type=EnergyType.ELECTRICITY,
            ct_ratio=Decimal("400"),
            pt_ratio=Decimal("1"),
        )
        channels = engine.configure_channels(meter, [ChannelType.KW, ChannelType.KWH])
        hierarchy = engine.build_hierarchy("SITE-001", "HQ Campus", [meter])
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise MeterRegistryEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - default_accuracy_class (str): default accuracy class
                - default_calibration_months (int): default calibration interval
                - max_channels (int): maximum channels per meter
        """
        self.config = config or {}
        self._default_accuracy = self.config.get("default_accuracy_class", "0.5")
        self._default_cal_months = int(
            self.config.get("default_calibration_months", 36)
        )
        self._max_channels = int(
            self.config.get("max_channels", MAX_CHANNELS_PER_METER)
        )
        self._meters: Dict[str, MeterConfig] = {}
        self._calibrations: Dict[str, List[CalibrationRecord]] = {}
        self._virtual_meters: Dict[str, VirtualMeterDefinition] = {}
        logger.info(
            "MeterRegistryEngine v%s initialised (accuracy=%s, cal_months=%d)",
            self.engine_version, self._default_accuracy, self._default_cal_months,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def register_meter(
        self,
        meter_name: str,
        meter_type: MeterType = MeterType.SUBMETER,
        energy_type: EnergyType = EnergyType.ELECTRICITY,
        protocol: MeterProtocol = MeterProtocol.MODBUS_TCP,
        ct_ratio: Decimal = Decimal("1"),
        pt_ratio: Decimal = Decimal("1"),
        rated_capacity_kw: Decimal = Decimal("0"),
        serial_number: str = "",
        manufacturer: str = "",
        model: str = "",
        location: str = "",
        parent_meter_id: Optional[str] = None,
        hierarchy_level: HierarchyLevel = HierarchyLevel.BUILDING,
        accuracy_class: Optional[str] = None,
        install_date: Optional[datetime] = None,
        notes: str = "",
    ) -> MeterConfig:
        """Register a new meter in the registry.

        Creates a new MeterConfig with validated parameters and stores it
        in the internal registry.  Assigns a unique meter_id and sets
        the calibration interval based on meter type.

        Args:
            meter_name:        Human-readable name.
            meter_type:        Meter classification.
            energy_type:       Energy commodity being metered.
            protocol:          Communication protocol.
            ct_ratio:          Current transformer ratio.
            pt_ratio:          Potential transformer ratio.
            rated_capacity_kw: Rated capacity in kW.
            serial_number:     Manufacturer serial number.
            manufacturer:      Manufacturer name.
            model:             Model number.
            location:          Physical location description.
            parent_meter_id:   Parent meter for hierarchy.
            hierarchy_level:   Position in hierarchy.
            accuracy_class:    Accuracy class override.
            install_date:      Date of installation.
            notes:             Additional notes.

        Returns:
            MeterConfig with all fields populated.
        """
        t0 = time.perf_counter()
        logger.info("Registering meter: %s (type=%s)", meter_name, meter_type.value)

        acc_class = accuracy_class or self._default_accuracy
        cal_months = CALIBRATION_INTERVAL_MONTHS.get(
            meter_type.value, self._default_cal_months
        )

        meter = MeterConfig(
            meter_name=meter_name,
            meter_type=meter_type,
            energy_type=energy_type,
            protocol=protocol,
            ct_ratio=ct_ratio,
            pt_ratio=pt_ratio,
            rated_capacity_kw=rated_capacity_kw,
            serial_number=serial_number,
            manufacturer=manufacturer,
            model=model,
            accuracy_class=acc_class,
            location=location,
            parent_meter_id=parent_meter_id,
            hierarchy_level=hierarchy_level,
            install_date=install_date or _utcnow(),
            calibration_interval_months=cal_months,
            notes=notes,
        )

        self._meters[meter.meter_id] = meter
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Meter registered: %s (id=%s, type=%s, CT=%s, PT=%s) (%.1f ms)",
            meter_name, meter.meter_id[:12], meter_type.value,
            str(ct_ratio), str(pt_ratio), elapsed,
        )
        return meter

    def configure_channels(
        self,
        meter: MeterConfig,
        channel_types: List[ChannelType],
        ct_ratio: Optional[Decimal] = None,
        pt_ratio: Optional[Decimal] = None,
    ) -> MeterConfig:
        """Configure data channels on a meter.

        Creates channel definitions for each requested channel type and
        computes the combined CT*PT multiplier for scaling raw values.

        Args:
            meter:         Meter to configure.
            channel_types: List of channel types to add.
            ct_ratio:      Override CT ratio (uses meter default if None).
            pt_ratio:      Override PT ratio (uses meter default if None).

        Returns:
            Updated MeterConfig with channels added.
        """
        t0 = time.perf_counter()
        logger.info(
            "Configuring %d channels on meter %s",
            len(channel_types), meter.meter_id[:12],
        )

        effective_ct = ct_ratio if ct_ratio is not None else meter.ct_ratio
        effective_pt = pt_ratio if pt_ratio is not None else meter.pt_ratio
        multiplier = _round_val(effective_ct * effective_pt, 6)

        new_channels: List[MeterChannel] = list(meter.channels)
        existing_types = {ch.channel_type for ch in new_channels}

        for ch_type in channel_types:
            if len(new_channels) >= self._max_channels:
                logger.warning(
                    "Max channels (%d) reached on meter %s",
                    self._max_channels, meter.meter_id[:12],
                )
                break
            if ch_type in existing_types:
                logger.info(
                    "Channel %s already exists on meter %s, skipping",
                    ch_type.value, meter.meter_id[:12],
                )
                continue

            unit = self._channel_type_to_unit(ch_type)
            is_cumulative = ch_type in (
                ChannelType.KWH, ChannelType.KVARH, ChannelType.M3,
            )
            min_val, max_val = self._channel_type_limits(ch_type, multiplier)

            channel = MeterChannel(
                channel_type=ch_type,
                unit=unit,
                multiplier=multiplier,
                is_cumulative=is_cumulative,
                min_value=min_val,
                max_value=max_val,
                register_address=len(new_channels) * 2,
            )
            new_channels.append(channel)
            existing_types.add(ch_type)

        meter.channels = new_channels

        if meter.meter_id in self._meters:
            self._meters[meter.meter_id] = meter

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Channels configured: %d total on %s (multiplier=%s) (%.1f ms)",
            len(new_channels), meter.meter_id[:12], str(multiplier), elapsed,
        )
        return meter

    def build_hierarchy(
        self,
        site_id: str,
        site_name: str,
        meters: List[MeterConfig],
    ) -> MeterHierarchy:
        """Build a hierarchical tree from a list of meters.

        Constructs a nested tree structure based on parent-child meter
        relationships, computes coverage percentage, identifies orphan
        meters (missing parent), and calculates hierarchy completeness.

        Args:
            site_id:   Unique site identifier.
            site_name: Site name.
            meters:    List of MeterConfig objects.

        Returns:
            MeterHierarchy with tree structure and metrics.
        """
        t0 = time.perf_counter()
        logger.info(
            "Building hierarchy for site %s (%d meters)",
            site_name, len(meters),
        )

        if not meters:
            result = MeterHierarchy(site_id=site_id, site_name=site_name)
            result.provenance_hash = _compute_hash(result)
            return result

        meter_map: Dict[str, MeterConfig] = {m.meter_id: m for m in meters}
        children_map: Dict[Optional[str], List[str]] = {}
        orphans: List[str] = []

        for m in meters:
            parent = m.parent_meter_id
            if parent is not None and parent not in meter_map:
                orphans.append(m.meter_id)
                parent = None
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(m.meter_id)

        # Build tree recursively
        tree = self._build_tree_node(None, children_map, meter_map, 0)

        # Calculate depth
        depth = self._tree_depth(tree)

        # Active meters
        active = sum(1 for m in meters if m.status == MeterStatus.ACTIVE)
        total = len(meters)

        # Coverage: sum(submeter_capacity) / main_meter_capacity
        root_meters = [
            m for m in meters if m.parent_meter_id is None
        ]
        root_capacity = sum(
            (_decimal(m.rated_capacity_kw) for m in root_meters), Decimal("0")
        )
        child_meters = [
            m for m in meters if m.parent_meter_id is not None
        ]
        child_capacity = sum(
            (_decimal(m.rated_capacity_kw) for m in child_meters), Decimal("0")
        )
        coverage = _safe_pct(child_capacity, root_capacity)

        # Completeness: meters with valid parent / total
        meters_with_parent = sum(
            1 for m in meters
            if m.parent_meter_id is None or m.parent_meter_id in meter_map
        )
        completeness = _safe_pct(
            _decimal(meters_with_parent), _decimal(total)
        )

        result = MeterHierarchy(
            site_id=site_id,
            site_name=site_name,
            total_meters=total,
            active_meters=active,
            hierarchy_depth=depth,
            coverage_pct=_round_val(coverage, 2),
            tree=tree,
            orphan_meters=orphans,
            completeness_pct=_round_val(completeness, 2),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Hierarchy built: %d meters, depth=%d, coverage=%.1f%%, "
            "orphans=%d, hash=%s (%.1f ms)",
            total, depth, float(coverage), len(orphans),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def track_calibration(
        self,
        meter_id: str,
        test_points: Dict[str, Decimal],
        technician: str = "",
        reference_standard: str = "",
        certificate_number: str = "",
        notes: str = "",
    ) -> CalibrationRecord:
        """Record a meter calibration event.

        Computes overall error, drift since last calibration, and
        determines pass/fail against the meter's accuracy class.

        Args:
            meter_id:            Meter being calibrated.
            test_points:         Dict of load% -> measured error%.
            technician:          Calibrating technician name.
            reference_standard:  Reference standard used.
            certificate_number:  Certificate number.
            notes:               Additional notes.

        Returns:
            CalibrationRecord with pass/fail determination.
        """
        t0 = time.perf_counter()
        logger.info("Tracking calibration for meter %s", meter_id[:12])

        meter = self._meters.get(meter_id)
        accuracy_limit = ACCURACY_CLASS_LIMITS.get(
            meter.accuracy_class if meter else self._default_accuracy,
            Decimal("0.5"),
        )

        # Compute overall error (average of absolute errors)
        if test_points:
            abs_errors = [abs(v) for v in test_points.values()]
            overall_error = _safe_divide(
                sum(abs_errors, Decimal("0")),
                _decimal(len(abs_errors)),
            )
        else:
            overall_error = Decimal("0")

        passed = overall_error <= accuracy_limit

        # Compute drift from last calibration
        drift = Decimal("0")
        prev_records = self._calibrations.get(meter_id, [])
        if prev_records:
            last_error = prev_records[-1].overall_error_pct
            drift = abs(overall_error - last_error)

        # Calculate next due date
        cal_months = self._default_cal_months
        if meter:
            cal_months = meter.calibration_interval_months

        cal_date = _utcnow()
        next_due = self._add_months(cal_date, cal_months) if cal_months > 0 else None

        record = CalibrationRecord(
            meter_id=meter_id,
            calibration_date=cal_date,
            next_due_date=next_due,
            technician=technician,
            reference_standard=reference_standard,
            test_points=test_points,
            overall_error_pct=_round_val(overall_error, 4),
            accuracy_class=meter.accuracy_class if meter else self._default_accuracy,
            passed=passed,
            drift_pct=_round_val(drift, 4),
            ct_ratio_verified=meter.ct_ratio if meter else Decimal("1"),
            pt_ratio_verified=meter.pt_ratio if meter else Decimal("1"),
            certificate_number=certificate_number,
            notes=notes,
        )
        record.provenance_hash = _compute_hash(record)

        # Store record
        if meter_id not in self._calibrations:
            self._calibrations[meter_id] = []
        self._calibrations[meter_id].append(record)

        # Update meter status
        if meter:
            meter.last_calibration = cal_date
            if not passed:
                meter.status = MeterStatus.FAULT

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Calibration recorded: meter=%s, error=%.4f%%, passed=%s, "
            "drift=%.4f%%, hash=%s (%.1f ms)",
            meter_id[:12], float(overall_error), str(passed),
            float(drift), record.provenance_hash[:16], elapsed,
        )
        return record

    def create_virtual_meter(
        self,
        formula_name: str,
        components: Dict[str, Decimal],
        operator: str = "SUM",
        description: str = "",
        energy_type: EnergyType = EnergyType.ELECTRICITY,
    ) -> MeterConfig:
        """Create a virtual (calculated) meter from a formula.

        A virtual meter derives its value from a formula applied to
        one or more physical meters.  The components dict maps
        meter_id -> coefficient (positive for addition, negative for
        subtraction).

        Example:
            components = {"M-001": Decimal("1"), "M-002": Decimal("1"),
                          "M-003": Decimal("-1")}
            Means: virtual_value = M-001 + M-002 - M-003

        Args:
            formula_name: Name for the formula.
            components:   Dict of meter_id -> coefficient.
            operator:     Aggregation operator (SUM / DIFF / AVG).
            description:  Formula description.
            energy_type:  Energy type for the virtual meter.

        Returns:
            MeterConfig for the virtual meter.
        """
        t0 = time.perf_counter()
        logger.info(
            "Creating virtual meter: %s (%d components)",
            formula_name, len(components),
        )

        # Register the virtual meter
        meter = self.register_meter(
            meter_name=f"VIRTUAL: {formula_name}",
            meter_type=MeterType.VIRTUAL,
            energy_type=energy_type,
            protocol=MeterProtocol.MANUAL,
            hierarchy_level=HierarchyLevel.SYSTEM,
            notes=f"Virtual meter: {description}",
        )

        # Store virtual definition
        definition = VirtualMeterDefinition(
            virtual_meter_id=meter.meter_id,
            formula_name=formula_name,
            components=components,
            operator=operator,
            description=description,
        )
        definition.provenance_hash = _compute_hash(definition)
        self._virtual_meters[meter.meter_id] = definition

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Virtual meter created: %s (id=%s, components=%d) (%.1f ms)",
            formula_name, meter.meter_id[:12], len(components), elapsed,
        )
        return meter

    def evaluate_virtual_meter(
        self,
        virtual_meter_id: str,
        component_values: Dict[str, Decimal],
    ) -> Decimal:
        """Evaluate a virtual meter formula with given component values.

        Args:
            virtual_meter_id: ID of the virtual meter.
            component_values: Dict of meter_id -> current reading value.

        Returns:
            Calculated virtual meter value.
        """
        t0 = time.perf_counter()
        definition = self._virtual_meters.get(virtual_meter_id)
        if definition is None:
            logger.error("Virtual meter %s not found", virtual_meter_id[:12])
            return Decimal("0")

        result = Decimal("0")
        for meter_id, coefficient in definition.components.items():
            value = component_values.get(meter_id, Decimal("0"))
            result += _decimal(value) * _decimal(coefficient)

        if definition.operator == "AVG" and len(definition.components) > 0:
            result = _safe_divide(result, _decimal(len(definition.components)))

        result = _round_val(result, 4)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Virtual meter %s evaluated: result=%s (%s) (%.1f ms)",
            virtual_meter_id[:12], str(result),
            definition.formula_name, elapsed,
        )
        return result

    def get_registry_summary(
        self,
        site_id: str,
        site_name: str,
    ) -> MeterRegistryResult:
        """Generate a comprehensive registry summary for a site.

        Args:
            site_id:   Site identifier.
            site_name: Site name.

        Returns:
            MeterRegistryResult with complete registry analysis.
        """
        t0 = time.perf_counter()
        logger.info("Generating registry summary for site %s", site_name)

        all_meters = list(self._meters.values())

        # Basic counts
        total = len(all_meters)
        active = sum(1 for m in all_meters if m.status == MeterStatus.ACTIVE)
        virtual = sum(1 for m in all_meters if m.meter_type == MeterType.VIRTUAL)

        # Calibration due check
        now = _utcnow()
        cal_due = 0
        for m in all_meters:
            if m.meter_type == MeterType.VIRTUAL:
                continue
            if m.last_calibration is None:
                cal_due += 1
                continue
            next_due = self._add_months(m.last_calibration, m.calibration_interval_months)
            if next_due is not None and next_due <= now:
                cal_due += 1

        # Build hierarchy
        hierarchy = self.build_hierarchy(site_id, site_name, all_meters)

        # Energy types covered
        energy_types = sorted(set(m.energy_type.value for m in all_meters))

        # Protocol summary
        proto_summary: Dict[str, int] = {}
        for m in all_meters:
            pv = m.protocol.value
            proto_summary[pv] = proto_summary.get(pv, 0) + 1

        # Collect calibration records
        all_cals: List[CalibrationRecord] = []
        for cal_list in self._calibrations.values():
            all_cals.extend(cal_list)

        # Virtual definitions
        virt_defs = list(self._virtual_meters.values())

        # Warnings
        warnings = self._generate_warnings(all_meters, cal_due, hierarchy)

        # Recommendations
        recommendations = self._generate_recommendations(
            all_meters, hierarchy, cal_due, energy_types,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = MeterRegistryResult(
            site_id=site_id,
            site_name=site_name,
            total_meters=total,
            active_meters=active,
            virtual_meters=virtual,
            meters_calibration_due=cal_due,
            hierarchy=hierarchy,
            registered_meters=all_meters,
            virtual_definitions=virt_defs,
            calibration_records=all_cals[-50:],
            coverage_pct=hierarchy.coverage_pct,
            energy_types_covered=energy_types,
            protocol_summary=proto_summary,
            warnings=warnings,
            recommendations=recommendations,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Registry summary: %d meters (%d active, %d virtual), "
            "cal_due=%d, coverage=%.1f%%, hash=%s (%.1f ms)",
            total, active, virtual, cal_due,
            float(hierarchy.coverage_pct),
            result.provenance_hash[:16], float(elapsed_ms),
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _channel_type_to_unit(self, ch_type: ChannelType) -> str:
        """Map a channel type to its default engineering unit.

        Args:
            ch_type: Channel type.

        Returns:
            Unit string.
        """
        unit_map: Dict[ChannelType, str] = {
            ChannelType.KW: "kW",
            ChannelType.KWH: "kWh",
            ChannelType.KVAR: "kVAR",
            ChannelType.KVARH: "kVARh",
            ChannelType.KVA: "kVA",
            ChannelType.VOLTAGE: "V",
            ChannelType.CURRENT: "A",
            ChannelType.PF: "pf",
            ChannelType.THERM: "therm",
            ChannelType.M3: "m3",
            ChannelType.FLOW: "m3/h",
            ChannelType.TEMPERATURE: "degC",
            ChannelType.PRESSURE: "kPa",
        }
        return unit_map.get(ch_type, "units")

    def _channel_type_limits(
        self,
        ch_type: ChannelType,
        multiplier: Decimal,
    ) -> Tuple[Decimal, Decimal]:
        """Return (min_value, max_value) limits for a channel type.

        Args:
            ch_type:    Channel type.
            multiplier: CT*PT scaling factor.

        Returns:
            Tuple of (min, max) Decimal values.
        """
        limits: Dict[ChannelType, Tuple[Decimal, Decimal]] = {
            ChannelType.KW: (Decimal("0"), Decimal("100000") * multiplier),
            ChannelType.KWH: (Decimal("0"), Decimal("999999999")),
            ChannelType.KVAR: (Decimal("-100000") * multiplier, Decimal("100000") * multiplier),
            ChannelType.KVARH: (Decimal("0"), Decimal("999999999")),
            ChannelType.KVA: (Decimal("0"), Decimal("100000") * multiplier),
            ChannelType.VOLTAGE: (Decimal("0"), Decimal("500000")),
            ChannelType.CURRENT: (Decimal("0"), Decimal("100000")),
            ChannelType.PF: (Decimal("-1"), Decimal("1")),
            ChannelType.THERM: (Decimal("0"), Decimal("999999999")),
            ChannelType.M3: (Decimal("0"), Decimal("999999999")),
            ChannelType.FLOW: (Decimal("0"), Decimal("100000")),
            ChannelType.TEMPERATURE: (Decimal("-273"), Decimal("1500")),
            ChannelType.PRESSURE: (Decimal("0"), Decimal("100000")),
        }
        return limits.get(ch_type, (Decimal("0"), Decimal("999999999")))

    def _build_tree_node(
        self,
        parent_id: Optional[str],
        children_map: Dict[Optional[str], List[str]],
        meter_map: Dict[str, MeterConfig],
        depth: int,
    ) -> Dict[str, Any]:
        """Recursively build a hierarchy tree node.

        Args:
            parent_id:    Current parent meter ID.
            children_map: Map of parent_id -> list of child meter IDs.
            meter_map:    Map of meter_id -> MeterConfig.
            depth:        Current recursion depth.

        Returns:
            Dict representing the tree node.
        """
        if depth > MAX_HIERARCHY_DEPTH:
            return {}

        children_ids = children_map.get(parent_id, [])
        node: Dict[str, Any] = {}

        for child_id in children_ids:
            meter = meter_map.get(child_id)
            if meter is None:
                continue
            child_node: Dict[str, Any] = {
                "meter_id": child_id,
                "meter_name": meter.meter_name,
                "meter_type": meter.meter_type.value,
                "energy_type": meter.energy_type.value,
                "status": meter.status.value,
                "level": meter.hierarchy_level.value,
                "capacity_kw": str(meter.rated_capacity_kw),
                "children": self._build_tree_node(
                    child_id, children_map, meter_map, depth + 1
                ),
            }
            node[child_id] = child_node

        return node

    def _tree_depth(self, tree: Dict[str, Any], current: int = 0) -> int:
        """Calculate the maximum depth of a hierarchy tree.

        Args:
            tree:    Tree dict.
            current: Current depth.

        Returns:
            Maximum depth.
        """
        if not tree:
            return current
        max_depth = current
        for node in tree.values():
            if isinstance(node, dict) and "children" in node:
                child_depth = self._tree_depth(node["children"], current + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth

    def _add_months(
        self,
        dt: datetime,
        months: int,
    ) -> Optional[datetime]:
        """Add months to a datetime.

        Args:
            dt:     Base datetime.
            months: Months to add.

        Returns:
            New datetime or None if months is 0.
        """
        if months <= 0:
            return None
        month = dt.month + months
        year = dt.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        day = min(dt.day, 28)
        return dt.replace(year=year, month=month, day=day)

    def _generate_warnings(
        self,
        meters: List[MeterConfig],
        cal_due: int,
        hierarchy: MeterHierarchy,
    ) -> List[str]:
        """Generate warnings based on registry analysis.

        Args:
            meters:    All registered meters.
            cal_due:   Number of meters with calibration due.
            hierarchy: Computed hierarchy.

        Returns:
            List of warning strings.
        """
        warnings: List[str] = []

        if cal_due > 0:
            warnings.append(
                f"{cal_due} meter(s) have calibration overdue or never calibrated."
            )

        fault_meters = sum(1 for m in meters if m.status == MeterStatus.FAULT)
        if fault_meters > 0:
            warnings.append(
                f"{fault_meters} meter(s) are in FAULT status."
            )

        if hierarchy.orphan_meters:
            warnings.append(
                f"{len(hierarchy.orphan_meters)} orphan meter(s) found "
                "with invalid parent references."
            )

        no_channel_meters = sum(1 for m in meters if not m.channels and m.meter_type != MeterType.VIRTUAL)
        if no_channel_meters > 0:
            warnings.append(
                f"{no_channel_meters} physical meter(s) have no channels configured."
            )

        return warnings

    def _generate_recommendations(
        self,
        meters: List[MeterConfig],
        hierarchy: MeterHierarchy,
        cal_due: int,
        energy_types: List[str],
    ) -> List[str]:
        """Generate recommendations based on registry state.

        Args:
            meters:       All registered meters.
            hierarchy:    Computed hierarchy.
            cal_due:      Number of meters with calibration due.
            energy_types: Energy types with metering.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if hierarchy.coverage_pct < Decimal("80"):
            recs.append(
                "Sub-metering coverage is below 80%. Install additional "
                "sub-meters to improve energy allocation accuracy."
            )

        if cal_due > 0:
            recs.append(
                "Schedule calibration for overdue meters to maintain "
                "measurement accuracy per ANSI C12.20."
            )

        rev_meters = sum(1 for m in meters if m.meter_type == MeterType.REVENUE)
        check_meters = sum(1 for m in meters if m.meter_type == MeterType.CHECK)
        if rev_meters > 0 and check_meters == 0:
            recs.append(
                "No check meters installed. Consider adding check meters "
                "for revenue meter verification per ANSI C12.1."
            )

        if EnergyType.ELECTRICITY.value in energy_types:
            has_pf = any(
                any(ch.channel_type == ChannelType.PF for ch in m.channels)
                for m in meters
            )
            if not has_pf:
                recs.append(
                    "No power factor channels configured. Add PF metering "
                    "to monitor reactive power and avoid utility penalties."
                )

        manual_meters = sum(
            1 for m in meters if m.protocol == MeterProtocol.MANUAL
            and m.meter_type != MeterType.VIRTUAL
        )
        if manual_meters > 2:
            recs.append(
                f"{manual_meters} meters use manual reading. Consider "
                "upgrading to automated protocols (Modbus/BACnet) for "
                "real-time data acquisition."
            )

        if not recs:
            recs.append(
                "Meter registry is well-configured. Continue with "
                "routine calibration and monitoring."
            )

        return recs
