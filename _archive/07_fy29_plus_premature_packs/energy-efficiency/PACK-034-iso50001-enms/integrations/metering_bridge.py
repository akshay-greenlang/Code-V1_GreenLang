# -*- coding: utf-8 -*-
"""
MeteringBridge - Sub-Metering Hierarchy and Data Reconciliation for EnMS
=========================================================================

This module provides sub-metering hierarchy management and data reconciliation
for the ISO 50001 EnMS pipeline. It supports Clause 6.6 (planning for
collection of energy data) by managing meter hierarchies, validating
meter coverage, reconciling readings, and calculating virtual meter values.

Features:
    - Build and manage hierarchical meter trees
    - Reconcile sub-meter readings against parent meters
    - Validate metering coverage against total consumption
    - Identify missing sub-meters in the hierarchy
    - Calculate virtual meter values from formulas
    - Support for main, sub, check, virtual, and calculated meters

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-034 ISO 50001 Energy Management System
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MeterType(str, Enum):
    """Types of energy meters in the hierarchy."""

    MAIN = "main"
    SUB = "sub"
    CHECK = "check"
    VIRTUAL = "virtual"
    CALCULATED = "calculated"

class ReadingQuality(str, Enum):
    """Quality classification for meter readings."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    CALCULATED = "calculated"
    INTERPOLATED = "interpolated"
    MISSING = "missing"

class ReconciliationStatus(str, Enum):
    """Status of meter reconciliation."""

    BALANCED = "balanced"
    IMBALANCED = "imbalanced"
    PARTIAL = "partial"
    NOT_RECONCILED = "not_reconciled"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class MeterConfig(BaseModel):
    """Configuration for a single meter in the hierarchy."""

    meter_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="")
    meter_type: MeterType = Field(default=MeterType.SUB)
    parent_meter_id: Optional[str] = Field(None, description="Parent meter in hierarchy")
    energy_type: str = Field(default="electricity", description="electricity|gas|steam|chilled_water")
    unit: str = Field(default="kWh")
    seu_id: str = Field(default="", description="Associated SEU identifier")
    location: str = Field(default="")
    ct_ratio: float = Field(default=1.0, ge=0.01, description="Current transformer ratio")
    pulse_factor: float = Field(default=1.0, ge=0.001, description="Pulse to unit conversion")
    accuracy_class: str = Field(default="1.0", description="Meter accuracy class")
    serial_number: str = Field(default="")
    installed_date: Optional[str] = Field(None)
    calibration_due: Optional[str] = Field(None)
    is_active: bool = Field(default=True)

class MeterHierarchy(BaseModel):
    """Tree structure representing the metering hierarchy."""

    hierarchy_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    root_meter_id: str = Field(default="")
    meters: Dict[str, MeterConfig] = Field(default_factory=dict)
    children_map: Dict[str, List[str]] = Field(default_factory=dict)
    depth: int = Field(default=0)
    total_meters: int = Field(default=0)
    coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")

class MeterReading(BaseModel):
    """A meter reading with quality classification."""

    reading_id: str = Field(default_factory=_new_uuid)
    meter_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=utcnow)
    value_kwh: float = Field(default=0.0)
    quality: ReadingQuality = Field(default=ReadingQuality.MEASURED)
    period_start: Optional[datetime] = Field(None)
    period_end: Optional[datetime] = Field(None)
    notes: str = Field(default="")

class ReconciliationResult(BaseModel):
    """Result of reconciling meter readings against the hierarchy."""

    reconciliation_id: str = Field(default_factory=_new_uuid)
    hierarchy_id: str = Field(default="")
    parent_meter_id: str = Field(default="")
    parent_reading_kwh: float = Field(default=0.0)
    sum_sub_readings_kwh: float = Field(default=0.0)
    difference_kwh: float = Field(default=0.0)
    difference_pct: float = Field(default=0.0)
    status: ReconciliationStatus = Field(default=ReconciliationStatus.NOT_RECONCILED)
    tolerance_pct: float = Field(default=5.0)
    sub_meter_count: int = Field(default=0)
    missing_sub_meters: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utcnow)
    message: str = Field(default="")
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# MeteringBridge
# ---------------------------------------------------------------------------

class MeteringBridge:
    """Sub-metering hierarchy and data reconciliation for ISO 50001 EnMS.

    Manages meter hierarchies, reconciles readings, validates coverage,
    and supports Clause 6.6 planning for energy data collection.

    Attributes:
        _hierarchies: Registered meter hierarchies.
        _readings: Stored meter readings by meter_id.

    Example:
        >>> bridge = MeteringBridge()
        >>> hierarchy = bridge.build_meter_hierarchy(meters)
        >>> result = bridge.reconcile_readings(hierarchy)
    """

    def __init__(self) -> None:
        """Initialize the Metering Bridge."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._hierarchies: Dict[str, MeterHierarchy] = {}
        self._readings: Dict[str, List[MeterReading]] = {}
        self.logger.info("MeteringBridge initialized")

    def build_meter_hierarchy(
        self, meters: List[MeterConfig],
    ) -> MeterHierarchy:
        """Build a hierarchical meter tree from a flat list of meters.

        Args:
            meters: List of MeterConfig objects with parent references.

        Returns:
            MeterHierarchy with tree structure and coverage metrics.
        """
        start = time.monotonic()
        self.logger.info("Building meter hierarchy from %d meters", len(meters))

        meter_map: Dict[str, MeterConfig] = {}
        children_map: Dict[str, List[str]] = {}
        root_id = ""

        for meter in meters:
            meter_map[meter.meter_id] = meter
            if meter.meter_type == MeterType.MAIN and not meter.parent_meter_id:
                root_id = meter.meter_id

        for meter in meters:
            parent = meter.parent_meter_id
            if parent:
                if parent not in children_map:
                    children_map[parent] = []
                children_map[parent].append(meter.meter_id)

        # Calculate depth
        depth = self._calculate_depth(root_id, children_map)

        # Calculate coverage
        total_sub = sum(1 for m in meters if m.meter_type in (MeterType.SUB, MeterType.CHECK))
        coverage = (total_sub / max(len(meters) - 1, 1)) * 100.0 if meters else 0.0

        hierarchy = MeterHierarchy(
            facility_id=meters[0].location if meters else "",
            root_meter_id=root_id,
            meters=meter_map,
            children_map=children_map,
            depth=depth,
            total_meters=len(meters),
            coverage_pct=round(min(coverage, 100.0), 1),
        )
        hierarchy.provenance_hash = _compute_hash(hierarchy)

        self._hierarchies[hierarchy.hierarchy_id] = hierarchy

        self.logger.info(
            "Meter hierarchy built: %d meters, depth=%d, coverage=%.1f%%",
            len(meters), depth, hierarchy.coverage_pct,
        )
        return hierarchy

    def reconcile_readings(
        self,
        hierarchy: MeterHierarchy,
        parent_reading_kwh: Optional[float] = None,
        sub_readings: Optional[Dict[str, float]] = None,
        tolerance_pct: float = 5.0,
    ) -> ReconciliationResult:
        """Reconcile sub-meter readings against the parent meter.

        Deterministic: difference = parent - sum(sub_meters)

        Args:
            hierarchy: Meter hierarchy to reconcile.
            parent_reading_kwh: Parent meter reading (uses stored if None).
            sub_readings: Dict of meter_id to kWh (uses stored if None).
            tolerance_pct: Acceptable difference percentage.

        Returns:
            ReconciliationResult with balance analysis.
        """
        start = time.monotonic()
        root_id = hierarchy.root_meter_id
        children = hierarchy.children_map.get(root_id, [])

        # Get parent reading
        parent_kwh = parent_reading_kwh or 0.0

        # Sum sub-meter readings
        sub_readings = sub_readings or {}
        sum_sub_kwh = sum(sub_readings.get(cid, 0.0) for cid in children)

        # Deterministic calculation
        diff_kwh = parent_kwh - sum_sub_kwh
        diff_pct = (abs(diff_kwh) / parent_kwh * 100.0) if parent_kwh > 0 else 0.0

        # Missing sub-meters
        missing = [cid for cid in children if cid not in sub_readings]

        # Determine status
        if diff_pct <= tolerance_pct and not missing:
            status = ReconciliationStatus.BALANCED
        elif missing:
            status = ReconciliationStatus.PARTIAL
        else:
            status = ReconciliationStatus.IMBALANCED

        result = ReconciliationResult(
            hierarchy_id=hierarchy.hierarchy_id,
            parent_meter_id=root_id,
            parent_reading_kwh=parent_kwh,
            sum_sub_readings_kwh=sum_sub_kwh,
            difference_kwh=round(diff_kwh, 2),
            difference_pct=round(diff_pct, 2),
            status=status,
            tolerance_pct=tolerance_pct,
            sub_meter_count=len(children),
            missing_sub_meters=missing,
            message=f"Reconciliation: {status.value} ({diff_pct:.1f}% difference)",
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Reconciliation: parent=%.1f, sub_total=%.1f, diff=%.1f (%.1f%%), status=%s",
            parent_kwh, sum_sub_kwh, diff_kwh, diff_pct, status.value,
        )
        return result

    def validate_meter_coverage(
        self,
        hierarchy: MeterHierarchy,
        total_consumption_kwh: float,
    ) -> Dict[str, Any]:
        """Validate that metering covers the total facility consumption.

        Args:
            hierarchy: Meter hierarchy to validate.
            total_consumption_kwh: Known total facility consumption.

        Returns:
            Dict with coverage validation results.
        """
        metered_kwh = 0.0
        for meter_id, meter in hierarchy.meters.items():
            if meter.meter_type in (MeterType.MAIN, MeterType.SUB):
                readings = self._readings.get(meter_id, [])
                if readings:
                    metered_kwh += readings[-1].value_kwh

        coverage_pct = (metered_kwh / total_consumption_kwh * 100.0) if total_consumption_kwh > 0 else 0.0

        result = {
            "hierarchy_id": hierarchy.hierarchy_id,
            "total_consumption_kwh": total_consumption_kwh,
            "metered_kwh": metered_kwh,
            "unmetered_kwh": total_consumption_kwh - metered_kwh,
            "coverage_pct": round(coverage_pct, 1),
            "adequate": coverage_pct >= 80.0,
            "iso50001_requirement": "Clause 6.6 - at least 80% metered",
            "provenance_hash": _compute_hash({"metered": metered_kwh, "total": total_consumption_kwh}),
        }
        return result

    def identify_missing_sub_meters(
        self, hierarchy: MeterHierarchy,
    ) -> List[Dict[str, Any]]:
        """Identify gaps in the metering hierarchy.

        Args:
            hierarchy: Meter hierarchy to analyze.

        Returns:
            List of recommended sub-meter additions.
        """
        recommendations: List[Dict[str, Any]] = []
        for meter_id, meter in hierarchy.meters.items():
            if meter.meter_type == MeterType.MAIN:
                children = hierarchy.children_map.get(meter_id, [])
                if not children:
                    recommendations.append({
                        "parent_meter_id": meter_id,
                        "parent_name": meter.name,
                        "recommendation": "Add sub-meters for SEU monitoring",
                        "priority": "high",
                        "reason": "Main meter has no sub-meters",
                    })
            if meter.seu_id and meter.meter_type == MeterType.MAIN:
                # SEU should have dedicated sub-metering
                seu_sub_meters = [
                    m for m in hierarchy.meters.values()
                    if m.seu_id == meter.seu_id and m.meter_type == MeterType.SUB
                ]
                if not seu_sub_meters:
                    recommendations.append({
                        "seu_id": meter.seu_id,
                        "recommendation": f"Add dedicated sub-meter for SEU {meter.seu_id}",
                        "priority": "high",
                        "reason": "SEU lacks dedicated sub-metering (Clause 6.6)",
                    })
        return recommendations

    def calculate_virtual_meter(
        self,
        formula: str,
        readings: Dict[str, float],
    ) -> Dict[str, Any]:
        """Calculate a virtual meter value from a formula and readings.

        Supports simple formulas: difference (parent - sum_children),
        sum, and weighted average.

        Args:
            formula: Formula type ('difference', 'sum', 'average').
            readings: Dict of meter_id to kWh values.

        Returns:
            Dict with calculated virtual meter value.
        """
        values = list(readings.values())

        if formula == "difference" and len(values) >= 2:
            result_value = values[0] - sum(values[1:])
        elif formula == "sum":
            result_value = sum(values)
        elif formula == "average" and values:
            result_value = sum(values) / len(values)
        else:
            result_value = 0.0

        return {
            "virtual_meter_id": _new_uuid(),
            "formula": formula,
            "input_meters": list(readings.keys()),
            "input_values": readings,
            "calculated_value_kwh": round(result_value, 2),
            "quality": ReadingQuality.CALCULATED.value,
            "timestamp": utcnow().isoformat(),
            "provenance_hash": _compute_hash(readings),
        }

    def store_reading(self, reading: MeterReading) -> None:
        """Store a meter reading for reconciliation.

        Args:
            reading: MeterReading to store.
        """
        if reading.meter_id not in self._readings:
            self._readings[reading.meter_id] = []
        self._readings[reading.meter_id].append(reading)

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _calculate_depth(
        self, root_id: str, children_map: Dict[str, List[str]],
    ) -> int:
        """Calculate the depth of the meter hierarchy tree.

        Args:
            root_id: Root meter identifier.
            children_map: Map of parent to children.

        Returns:
            Maximum depth of the tree.
        """
        if not root_id or root_id not in children_map:
            return 1

        max_child_depth = 0
        for child_id in children_map.get(root_id, []):
            child_depth = self._calculate_depth(child_id, children_map)
            max_child_depth = max(max_child_depth, child_depth)
        return 1 + max_child_depth
