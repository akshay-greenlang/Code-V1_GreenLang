# -*- coding: utf-8 -*-
"""
Meter Setup Workflow
===================================

4-phase workflow for registering energy meters, configuring measurement
channels, establishing metering hierarchies, and commissioning within
PACK-039 Energy Monitoring Pack.

Phases:
    1. MeterRegistration       -- Register meter assets with protocol specs
    2. ChannelConfiguration    -- Configure measurement channels and units
    3. HierarchySetup          -- Build metering hierarchy (site > building > floor)
    4. Commissioning           -- Verify communication, validate first reads

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - IEC 62053-21/22 (electricity metering equipment)
    - IEC 61850 (substation communication)
    - ISO 50001:2018 (energy management systems - metering plan)
    - EN 15232 (building automation impact on energy)

Schedule: on-demand / project commissioning
Estimated duration: 20 minutes

Author: GreenLang Team
Version: 39.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.utcnow()


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class MeterType(str, Enum):
    """Meter hardware type classification."""

    ELECTRICAL = "electrical"
    GAS = "gas"
    WATER = "water"
    STEAM = "steam"
    THERMAL = "thermal"
    COMPRESSED_AIR = "compressed_air"
    VIRTUAL = "virtual"


class CommissionStatus(str, Enum):
    """Meter commissioning status."""

    PENDING = "pending"
    COMMUNICATING = "communicating"
    VALIDATED = "validated"
    FAILED = "failed"
    BYPASSED = "bypassed"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

METER_PROTOCOL_SPECS: Dict[str, Dict[str, Any]] = {
    "modbus_rtu": {
        "description": "Modbus RTU serial protocol",
        "baud_rates": [9600, 19200, 38400, 57600, 115200],
        "default_baud": 9600,
        "addressing": "slave_id_1_247",
        "max_devices_per_bus": 247,
        "data_types": ["int16", "int32", "float32", "float64"],
        "register_types": ["holding", "input"],
        "typical_poll_interval_s": 15,
        "max_registers_per_read": 125,
    },
    "modbus_tcp": {
        "description": "Modbus TCP/IP Ethernet protocol",
        "baud_rates": [],
        "default_baud": 0,
        "addressing": "ip_address_unit_id",
        "max_devices_per_bus": 247,
        "data_types": ["int16", "int32", "float32", "float64"],
        "register_types": ["holding", "input"],
        "typical_poll_interval_s": 5,
        "max_registers_per_read": 125,
    },
    "bacnet_ip": {
        "description": "BACnet/IP building automation protocol",
        "baud_rates": [],
        "default_baud": 0,
        "addressing": "device_instance_object",
        "max_devices_per_bus": 4194303,
        "data_types": ["real", "unsigned", "signed", "double"],
        "register_types": ["analog_input", "analog_output", "analog_value"],
        "typical_poll_interval_s": 60,
        "max_registers_per_read": 50,
    },
    "bacnet_mstp": {
        "description": "BACnet MS/TP serial protocol",
        "baud_rates": [9600, 19200, 38400, 76800],
        "default_baud": 38400,
        "addressing": "mac_address_0_127",
        "max_devices_per_bus": 128,
        "data_types": ["real", "unsigned", "signed", "double"],
        "register_types": ["analog_input", "analog_output", "analog_value"],
        "typical_poll_interval_s": 60,
        "max_registers_per_read": 50,
    },
    "mbus": {
        "description": "M-Bus (EN 13757) metering bus protocol",
        "baud_rates": [300, 2400, 9600],
        "default_baud": 2400,
        "addressing": "primary_1_250",
        "max_devices_per_bus": 250,
        "data_types": ["bcd", "int8", "int16", "int32", "float32"],
        "register_types": ["variable_data_block"],
        "typical_poll_interval_s": 300,
        "max_registers_per_read": 64,
    },
    "dlms_cosem": {
        "description": "DLMS/COSEM (IEC 62056) utility metering standard",
        "baud_rates": [300, 9600, 19200],
        "default_baud": 9600,
        "addressing": "obis_code",
        "max_devices_per_bus": 1,
        "data_types": ["unsigned", "double_long_unsigned", "float32", "float64"],
        "register_types": ["register", "demand_register", "profile_generic"],
        "typical_poll_interval_s": 900,
        "max_registers_per_read": 32,
    },
    "iec_61850": {
        "description": "IEC 61850 substation automation protocol",
        "baud_rates": [],
        "default_baud": 0,
        "addressing": "logical_node_data_object",
        "max_devices_per_bus": 65535,
        "data_types": ["float32", "float64", "int32", "boolean"],
        "register_types": ["goose", "mms", "sampled_values"],
        "typical_poll_interval_s": 1,
        "max_registers_per_read": 256,
    },
    "pulse_output": {
        "description": "Pulse/contact closure output (digital input)",
        "baud_rates": [],
        "default_baud": 0,
        "addressing": "digital_input_channel",
        "max_devices_per_bus": 32,
        "data_types": ["pulse_count", "frequency"],
        "register_types": ["digital_input"],
        "typical_poll_interval_s": 1,
        "max_registers_per_read": 1,
    },
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class MeterDefinition(BaseModel):
    """Definition of a meter to register."""

    meter_name: str = Field(..., min_length=1, description="Meter display name")
    meter_type: str = Field(default="electrical", description="Meter type key")
    protocol: str = Field(default="modbus_tcp", description="Communication protocol key")
    address: str = Field(default="", description="Protocol-specific address")
    manufacturer: str = Field(default="", description="Meter manufacturer")
    model: str = Field(default="", description="Meter model number")
    serial_number: str = Field(default="", description="Meter serial number")
    ct_ratio: float = Field(default=1.0, gt=0, description="Current transformer ratio")
    pt_ratio: float = Field(default=1.0, gt=0, description="Potential transformer ratio")
    pulse_weight: float = Field(default=1.0, gt=0, description="Pulse weight for pulse meters")
    channels: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Channel definitions: name, unit, register, data_type",
    )
    parent_meter_id: str = Field(default="", description="Parent meter ID in hierarchy")
    location: str = Field(default="", description="Physical location description")
    tags: List[str] = Field(default_factory=list, description="Classification tags")


class MeterSetupInput(BaseModel):
    """Input data model for MeterSetupWorkflow."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(..., min_length=1, description="Facility name")
    meters: List[MeterDefinition] = Field(
        default_factory=list,
        description="List of meters to register and configure",
    )
    hierarchy_rules: Dict[str, Any] = Field(
        default_factory=lambda: {
            "levels": ["site", "building", "floor", "zone", "panel"],
            "auto_assign": True,
        },
        description="Hierarchy configuration rules",
    )
    commissioning_checks: List[str] = Field(
        default_factory=lambda: [
            "communication_test",
            "register_read",
            "value_range_check",
            "timestamp_sync",
        ],
        description="Commissioning validation checks to perform",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facility_name")
    @classmethod
    def validate_facility_name(cls, v: str) -> str:
        """Ensure facility name is non-empty after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("facility_name must not be blank")
        return stripped


class MeterSetupResult(BaseModel):
    """Complete result from meter setup workflow."""

    setup_id: str = Field(..., description="Unique setup execution ID")
    facility_id: str = Field(default="", description="Facility identifier")
    meters_registered: int = Field(default=0, ge=0)
    channels_configured: int = Field(default=0, ge=0)
    hierarchy_nodes: int = Field(default=0, ge=0)
    hierarchy_depth: int = Field(default=0, ge=0)
    meters_commissioned: int = Field(default=0, ge=0)
    meters_failed: int = Field(default=0, ge=0)
    commission_pass_rate_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    protocols_used: List[str] = Field(default_factory=list)
    meter_registry: List[Dict[str, Any]] = Field(default_factory=list)
    hierarchy_tree: Dict[str, Any] = Field(default_factory=dict)
    setup_duration_ms: int = Field(default=0, ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class MeterSetupWorkflow:
    """
    4-phase meter setup workflow for energy monitoring system commissioning.

    Registers meter assets, configures measurement channels, builds metering
    hierarchies, and verifies communication through commissioning checks.

    Zero-hallucination: all meter specifications and protocol parameters are
    sourced from validated reference data. No LLM calls in the configuration
    or verification path.

    Attributes:
        setup_id: Unique setup execution identifier.
        _registry: Registered meter records.
        _channels: Configured channel records.
        _hierarchy: Hierarchy tree structure.
        _commission: Commissioning results.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = MeterSetupWorkflow()
        >>> meter = MeterDefinition(meter_name="Main Incomer", protocol="modbus_tcp")
        >>> inp = MeterSetupInput(facility_name="HQ Building", meters=[meter])
        >>> result = wf.run(inp)
        >>> assert result.meters_registered > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MeterSetupWorkflow."""
        self.setup_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._registry: List[Dict[str, Any]] = []
        self._channels: List[Dict[str, Any]] = []
        self._hierarchy: Dict[str, Any] = {}
        self._commission: Dict[str, Any] = {}
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: MeterSetupInput) -> MeterSetupResult:
        """
        Execute the 4-phase meter setup workflow.

        Args:
            input_data: Validated meter setup input.

        Returns:
            MeterSetupResult with registration, channels, hierarchy, and
            commissioning outcomes.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = _utcnow()
        self.logger.info(
            "Starting meter setup workflow %s for facility=%s meters=%d",
            self.setup_id, input_data.facility_name, len(input_data.meters),
        )

        self._phase_results = []
        self._registry = []
        self._channels = []
        self._hierarchy = {}
        self._commission = {}

        try:
            # Phase 1: Meter Registration
            phase1 = self._phase_meter_registration(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Channel Configuration
            phase2 = self._phase_channel_configuration(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Hierarchy Setup
            phase3 = self._phase_hierarchy_setup(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Commissioning
            phase4 = self._phase_commissioning(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error("Meter setup workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        # Aggregate protocols used
        protocols_used = list(set(
            m.get("protocol", "unknown") for m in self._registry
        ))

        # Commission stats
        commissioned = self._commission.get("passed", 0)
        failed = self._commission.get("failed", 0)
        total_comm = commissioned + failed
        pass_rate = Decimal(str(
            round(commissioned / max(total_comm, 1) * 100, 1)
        ))

        result = MeterSetupResult(
            setup_id=self.setup_id,
            facility_id=input_data.facility_id,
            meters_registered=len(self._registry),
            channels_configured=len(self._channels),
            hierarchy_nodes=self._hierarchy.get("total_nodes", 0),
            hierarchy_depth=self._hierarchy.get("depth", 0),
            meters_commissioned=commissioned,
            meters_failed=failed,
            commission_pass_rate_pct=pass_rate,
            protocols_used=protocols_used,
            meter_registry=self._registry,
            hierarchy_tree=self._hierarchy,
            setup_duration_ms=int(elapsed_ms),
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Meter setup workflow %s completed in %dms meters=%d channels=%d "
            "commissioned=%d/%d (%.0f%%)",
            self.setup_id, int(elapsed_ms), len(self._registry),
            len(self._channels), commissioned, total_comm, float(pass_rate),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Meter Registration
    # -------------------------------------------------------------------------

    def _phase_meter_registration(
        self, input_data: MeterSetupInput
    ) -> PhaseResult:
        """Register meter assets with protocol specifications."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not input_data.meters:
            warnings.append("No meters provided; creating default main incomer")
            input_data.meters.append(MeterDefinition(
                meter_name=f"{input_data.facility_name} - Main Incomer",
                meter_type="electrical",
                protocol="modbus_tcp",
            ))

        for idx, meter_def in enumerate(input_data.meters):
            meter_id = f"mtr-{_new_uuid()[:8]}"
            protocol = meter_def.protocol

            # Validate protocol exists
            spec = METER_PROTOCOL_SPECS.get(protocol)
            if not spec:
                warnings.append(
                    f"Unknown protocol '{protocol}' for meter '{meter_def.meter_name}'; "
                    f"defaulting to modbus_tcp"
                )
                protocol = "modbus_tcp"
                spec = METER_PROTOCOL_SPECS["modbus_tcp"]

            # Determine baud rate for serial protocols
            baud_rate = spec["default_baud"]
            if spec["baud_rates"]:
                baud_rate = spec["default_baud"] or spec["baud_rates"][0]

            record = {
                "meter_id": meter_id,
                "meter_name": meter_def.meter_name,
                "meter_type": meter_def.meter_type,
                "protocol": protocol,
                "protocol_description": spec["description"],
                "address": meter_def.address or f"unit-{idx + 1}",
                "baud_rate": baud_rate,
                "addressing_scheme": spec["addressing"],
                "data_types": spec["data_types"],
                "manufacturer": meter_def.manufacturer,
                "model": meter_def.model,
                "serial_number": meter_def.serial_number,
                "ct_ratio": meter_def.ct_ratio,
                "pt_ratio": meter_def.pt_ratio,
                "pulse_weight": meter_def.pulse_weight,
                "parent_meter_id": meter_def.parent_meter_id,
                "location": meter_def.location,
                "tags": meter_def.tags,
                "poll_interval_s": spec["typical_poll_interval_s"],
                "max_registers_per_read": spec["max_registers_per_read"],
                "registered_at": _utcnow().isoformat() + "Z",
                "sequence": idx + 1,
            }
            self._registry.append(record)

        outputs["meters_registered"] = len(self._registry)
        outputs["protocols"] = list(set(m["protocol"] for m in self._registry))
        outputs["facility_id"] = input_data.facility_id

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 MeterRegistration: %d meters registered for facility=%s",
            len(self._registry), input_data.facility_name,
        )
        return PhaseResult(
            phase_name="meter_registration", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Channel Configuration
    # -------------------------------------------------------------------------

    def _phase_channel_configuration(
        self, input_data: MeterSetupInput
    ) -> PhaseResult:
        """Configure measurement channels and units for each meter."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        default_channels = self._get_default_channels()

        for meter_rec in self._registry:
            meter_id = meter_rec["meter_id"]
            meter_type = meter_rec.get("meter_type", "electrical")

            # Find matching meter definition to get user-specified channels
            matching_def = self._find_meter_def(
                input_data.meters, meter_rec["meter_name"]
            )
            user_channels = matching_def.channels if matching_def else []

            if user_channels:
                for ch_def in user_channels:
                    channel_rec = {
                        "channel_id": f"ch-{_new_uuid()[:8]}",
                        "meter_id": meter_id,
                        "channel_name": ch_def.get("name", ""),
                        "unit": ch_def.get("unit", "kWh"),
                        "register": ch_def.get("register", 0),
                        "data_type": ch_def.get("data_type", "float32"),
                        "scaling_factor": ch_def.get("scaling_factor", 1.0),
                        "accumulation": ch_def.get("accumulation", "incremental"),
                        "ct_ratio_applied": meter_rec.get("ct_ratio", 1.0),
                        "pt_ratio_applied": meter_rec.get("pt_ratio", 1.0),
                    }
                    self._channels.append(channel_rec)
            else:
                # Apply default channels for the meter type
                defaults = default_channels.get(meter_type, default_channels["electrical"])
                for ch_idx, ch_template in enumerate(defaults):
                    channel_rec = {
                        "channel_id": f"ch-{_new_uuid()[:8]}",
                        "meter_id": meter_id,
                        "channel_name": ch_template["name"],
                        "unit": ch_template["unit"],
                        "register": ch_template.get("register_offset", 0) + ch_idx * 2,
                        "data_type": ch_template.get("data_type", "float32"),
                        "scaling_factor": ch_template.get("scaling_factor", 1.0),
                        "accumulation": ch_template.get("accumulation", "incremental"),
                        "ct_ratio_applied": meter_rec.get("ct_ratio", 1.0),
                        "pt_ratio_applied": meter_rec.get("pt_ratio", 1.0),
                    }
                    self._channels.append(channel_rec)
                warnings.append(
                    f"Applied default channels for meter '{meter_rec['meter_name']}' "
                    f"(type={meter_type})"
                )

        outputs["channels_configured"] = len(self._channels)
        outputs["meters_with_channels"] = len(set(
            ch["meter_id"] for ch in self._channels
        ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 ChannelConfiguration: %d channels configured across %d meters",
            len(self._channels), outputs["meters_with_channels"],
        )
        return PhaseResult(
            phase_name="channel_configuration", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Hierarchy Setup
    # -------------------------------------------------------------------------

    def _phase_hierarchy_setup(
        self, input_data: MeterSetupInput
    ) -> PhaseResult:
        """Build metering hierarchy (site > building > floor > zone)."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        levels = input_data.hierarchy_rules.get(
            "levels", ["site", "building", "floor", "zone", "panel"]
        )
        auto_assign = input_data.hierarchy_rules.get("auto_assign", True)

        # Build tree from meter parent relationships
        root_node = {
            "node_id": f"node-{_new_uuid()[:8]}",
            "node_name": input_data.facility_name,
            "level": levels[0] if levels else "site",
            "meter_ids": [],
            "children": [],
        }

        parent_map: Dict[str, List[Dict[str, Any]]] = {}
        orphan_meters: List[Dict[str, Any]] = []

        for meter_rec in self._registry:
            parent_id = meter_rec.get("parent_meter_id", "")
            if parent_id:
                parent_map.setdefault(parent_id, []).append(meter_rec)
            else:
                orphan_meters.append(meter_rec)

        # Assign orphans to root
        if auto_assign:
            for meter_rec in orphan_meters:
                root_node["meter_ids"].append(meter_rec["meter_id"])

                # Build children from parent relationships
                child_meters = parent_map.get(meter_rec["meter_id"], [])
                if child_meters:
                    child_level = levels[1] if len(levels) > 1 else "sub"
                    child_node = {
                        "node_id": f"node-{_new_uuid()[:8]}",
                        "node_name": meter_rec["meter_name"],
                        "level": child_level,
                        "meter_ids": [
                            cm["meter_id"] for cm in child_meters
                        ],
                        "children": [],
                    }
                    root_node["children"].append(child_node)

        # Count total nodes
        total_nodes = self._count_nodes(root_node)
        max_depth = self._tree_depth(root_node)

        self._hierarchy = {
            "tree": root_node,
            "total_nodes": total_nodes,
            "depth": max_depth,
            "levels_defined": levels,
            "auto_assigned": auto_assign,
        }

        outputs["total_nodes"] = total_nodes
        outputs["depth"] = max_depth
        outputs["levels"] = levels
        outputs["root_meters"] = len(root_node["meter_ids"])

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 HierarchySetup: %d nodes, depth=%d, levels=%s",
            total_nodes, max_depth, levels,
        )
        return PhaseResult(
            phase_name="hierarchy_setup", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Commissioning
    # -------------------------------------------------------------------------

    def _phase_commissioning(
        self, input_data: MeterSetupInput
    ) -> PhaseResult:
        """Verify communication and validate first meter reads."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        checks = input_data.commissioning_checks
        passed = 0
        failed = 0
        check_results: List[Dict[str, Any]] = []

        for meter_rec in self._registry:
            meter_id = meter_rec["meter_id"]
            meter_name = meter_rec["meter_name"]
            protocol = meter_rec["protocol"]
            spec = METER_PROTOCOL_SPECS.get(protocol, METER_PROTOCOL_SPECS["modbus_tcp"])

            meter_checks: List[Dict[str, Any]] = []
            meter_passed = True

            for check in checks:
                check_passed = self._run_commission_check(
                    check, meter_rec, spec
                )
                meter_checks.append({
                    "check": check,
                    "passed": check_passed,
                    "timestamp": _utcnow().isoformat() + "Z",
                })
                if not check_passed:
                    meter_passed = False
                    warnings.append(
                        f"Meter '{meter_name}' failed check '{check}'"
                    )

            if meter_passed:
                passed += 1
                status = CommissionStatus.VALIDATED.value
            else:
                failed += 1
                status = CommissionStatus.FAILED.value

            check_results.append({
                "meter_id": meter_id,
                "meter_name": meter_name,
                "status": status,
                "checks": meter_checks,
                "checks_passed": sum(1 for c in meter_checks if c["passed"]),
                "checks_total": len(meter_checks),
            })

        self._commission = {
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
            "check_results": check_results,
        }

        outputs["meters_commissioned"] = passed
        outputs["meters_failed"] = failed
        outputs["pass_rate_pct"] = round(
            passed / max(passed + failed, 1) * 100, 1
        )
        outputs["checks_per_meter"] = len(checks)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 Commissioning: %d/%d passed (%.0f%%)",
            passed, passed + failed, outputs["pass_rate_pct"],
        )
        return PhaseResult(
            phase_name="commissioning", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_default_channels(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return default channel templates by meter type."""
        return {
            "electrical": [
                {"name": "active_energy", "unit": "kWh", "register_offset": 0,
                 "data_type": "float32", "accumulation": "incremental"},
                {"name": "reactive_energy", "unit": "kVARh", "register_offset": 2,
                 "data_type": "float32", "accumulation": "incremental"},
                {"name": "active_power", "unit": "kW", "register_offset": 4,
                 "data_type": "float32", "accumulation": "instantaneous"},
                {"name": "voltage_avg", "unit": "V", "register_offset": 6,
                 "data_type": "float32", "accumulation": "instantaneous"},
                {"name": "current_avg", "unit": "A", "register_offset": 8,
                 "data_type": "float32", "accumulation": "instantaneous"},
                {"name": "power_factor", "unit": "pf", "register_offset": 10,
                 "data_type": "float32", "accumulation": "instantaneous"},
            ],
            "gas": [
                {"name": "volume", "unit": "m3", "register_offset": 0,
                 "data_type": "float32", "accumulation": "incremental"},
                {"name": "flow_rate", "unit": "m3/h", "register_offset": 2,
                 "data_type": "float32", "accumulation": "instantaneous"},
                {"name": "temperature", "unit": "degC", "register_offset": 4,
                 "data_type": "float32", "accumulation": "instantaneous"},
                {"name": "pressure", "unit": "kPa", "register_offset": 6,
                 "data_type": "float32", "accumulation": "instantaneous"},
            ],
            "water": [
                {"name": "volume", "unit": "m3", "register_offset": 0,
                 "data_type": "float32", "accumulation": "incremental"},
                {"name": "flow_rate", "unit": "m3/h", "register_offset": 2,
                 "data_type": "float32", "accumulation": "instantaneous"},
            ],
            "steam": [
                {"name": "mass_flow", "unit": "kg/h", "register_offset": 0,
                 "data_type": "float32", "accumulation": "instantaneous"},
                {"name": "energy", "unit": "GJ", "register_offset": 2,
                 "data_type": "float32", "accumulation": "incremental"},
                {"name": "temperature", "unit": "degC", "register_offset": 4,
                 "data_type": "float32", "accumulation": "instantaneous"},
                {"name": "pressure", "unit": "kPa", "register_offset": 6,
                 "data_type": "float32", "accumulation": "instantaneous"},
            ],
            "thermal": [
                {"name": "energy", "unit": "kWh_th", "register_offset": 0,
                 "data_type": "float32", "accumulation": "incremental"},
                {"name": "power", "unit": "kW_th", "register_offset": 2,
                 "data_type": "float32", "accumulation": "instantaneous"},
                {"name": "supply_temp", "unit": "degC", "register_offset": 4,
                 "data_type": "float32", "accumulation": "instantaneous"},
                {"name": "return_temp", "unit": "degC", "register_offset": 6,
                 "data_type": "float32", "accumulation": "instantaneous"},
            ],
            "compressed_air": [
                {"name": "volume", "unit": "Nm3", "register_offset": 0,
                 "data_type": "float32", "accumulation": "incremental"},
                {"name": "flow_rate", "unit": "Nm3/h", "register_offset": 2,
                 "data_type": "float32", "accumulation": "instantaneous"},
                {"name": "pressure", "unit": "bar", "register_offset": 4,
                 "data_type": "float32", "accumulation": "instantaneous"},
            ],
            "virtual": [
                {"name": "calculated_energy", "unit": "kWh", "register_offset": 0,
                 "data_type": "float32", "accumulation": "incremental"},
            ],
        }

    def _find_meter_def(
        self, meters: List[MeterDefinition], name: str
    ) -> Optional[MeterDefinition]:
        """Find a MeterDefinition by name."""
        for m in meters:
            if m.meter_name == name:
                return m
        return None

    def _run_commission_check(
        self, check: str, meter_rec: Dict[str, Any], spec: Dict[str, Any],
    ) -> bool:
        """Simulate a commissioning check (deterministic validation)."""
        if check == "communication_test":
            # Validate protocol and address are set
            return bool(meter_rec.get("protocol")) and bool(meter_rec.get("address"))
        elif check == "register_read":
            # Validate data types are supported
            return len(spec.get("data_types", [])) > 0
        elif check == "value_range_check":
            # Validate CT/PT ratios are sensible
            ct = meter_rec.get("ct_ratio", 1.0)
            pt = meter_rec.get("pt_ratio", 1.0)
            return 0.001 <= ct <= 10000 and 0.001 <= pt <= 10000
        elif check == "timestamp_sync":
            # Always passes in simulation (NTP sync assumed)
            return True
        return True

    def _count_nodes(self, node: Dict[str, Any]) -> int:
        """Count total nodes in hierarchy tree."""
        count = 1
        for child in node.get("children", []):
            count += self._count_nodes(child)
        return count

    def _tree_depth(self, node: Dict[str, Any]) -> int:
        """Calculate maximum depth of hierarchy tree."""
        children = node.get("children", [])
        if not children:
            return 1
        return 1 + max(self._tree_depth(c) for c in children)

    def _compute_provenance(self, result: MeterSetupResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
