# -*- coding: utf-8 -*-
"""
UtilityMeteringBridge - Smart Meter and Sub-Metering Integration for PACK-031
================================================================================

This module provides integration with smart metering infrastructure (AMI) and
sub-metering systems. It manages meter hierarchies, interval data processing,
virtual meter calculations, utility bill reconciliation, and demand profile
analysis.

Capabilities:
    - AMI (Advanced Metering Infrastructure) data ingestion
    - Sub-meter hierarchy: facility > building > floor > circuit
    - Interval data processing: 15-min, 30-min, hourly
    - Virtual meter calculations (aggregation and disaggregation)
    - Utility bill reconciliation against meter data
    - Demand profile analysis (peak, base load, load factor)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-031 Industrial Energy Audit
Status: Production Ready
"""

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
    """Types of energy meters."""

    MAIN_UTILITY = "main_utility"
    SUB_METER = "sub_meter"
    CHECK_METER = "check_meter"
    VIRTUAL = "virtual"
    AMI_SMART = "ami_smart"

class EnergyCarrier(str, Enum):
    """Energy carriers measured by meters."""

    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    STEAM = "steam"
    CHILLED_WATER = "chilled_water"
    HOT_WATER = "hot_water"
    COMPRESSED_AIR = "compressed_air"
    FUEL_OIL = "fuel_oil"
    LPG = "lpg"
    DIESEL = "diesel"
    WATER = "water"

class IntervalResolution(str, Enum):
    """Metering interval resolutions."""

    FIFTEEN_MIN = "15min"
    THIRTY_MIN = "30min"
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"

class HierarchyLevel(str, Enum):
    """Sub-meter hierarchy levels."""

    FACILITY = "facility"
    BUILDING = "building"
    FLOOR = "floor"
    ZONE = "zone"
    CIRCUIT = "circuit"
    EQUIPMENT = "equipment"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class MeterRegistration(BaseModel):
    """Registration record for a physical or virtual meter."""

    meter_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="")
    meter_type: MeterType = Field(default=MeterType.SUB_METER)
    energy_carrier: EnergyCarrier = Field(default=EnergyCarrier.ELECTRICITY)
    hierarchy_level: HierarchyLevel = Field(default=HierarchyLevel.FACILITY)
    parent_meter_id: Optional[str] = Field(None, description="Parent meter in hierarchy")
    child_meter_ids: List[str] = Field(default_factory=list)
    location: str = Field(default="")
    ct_ratio: float = Field(default=1.0, description="Current transformer ratio")
    multiplier: float = Field(default=1.0, description="Meter pulse multiplier")
    unit: str = Field(default="kWh")
    interval: IntervalResolution = Field(default=IntervalResolution.FIFTEEN_MIN)
    is_active: bool = Field(default=True)
    commissioned_date: Optional[str] = Field(None)
    equipment_ids: List[str] = Field(default_factory=list, description="Associated equipment")

class IntervalReading(BaseModel):
    """A single interval meter reading."""

    reading_id: str = Field(default_factory=_new_uuid)
    meter_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=utcnow)
    value: float = Field(default=0.0)
    unit: str = Field(default="kWh")
    interval: str = Field(default="15min")
    quality: str = Field(default="actual", description="actual|estimated|substituted")
    is_validated: bool = Field(default=False)

class VirtualMeterDefinition(BaseModel):
    """Definition of a virtual meter computed from physical meters."""

    virtual_meter_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="")
    formula: str = Field(default="sum", description="sum|difference|weighted_sum")
    source_meter_ids: List[str] = Field(default_factory=list)
    weights: List[float] = Field(default_factory=list, description="For weighted_sum")
    subtract_meter_ids: List[str] = Field(default_factory=list, description="For difference")
    unit: str = Field(default="kWh")

class BillReconciliationResult(BaseModel):
    """Result of reconciling utility bills against meter data."""

    reconciliation_id: str = Field(default_factory=_new_uuid)
    meter_id: str = Field(default="")
    billing_period_start: Optional[str] = Field(None)
    billing_period_end: Optional[str] = Field(None)
    billed_kwh: float = Field(default=0.0)
    metered_kwh: float = Field(default=0.0)
    variance_kwh: float = Field(default=0.0)
    variance_pct: float = Field(default=0.0)
    within_tolerance: bool = Field(default=True)
    tolerance_pct: float = Field(default=2.0)
    billed_cost: float = Field(default=0.0)
    issues: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class DemandProfile(BaseModel):
    """Demand profile analysis result."""

    profile_id: str = Field(default_factory=_new_uuid)
    meter_id: str = Field(default="")
    analysis_period: str = Field(default="")
    peak_demand_kw: float = Field(default=0.0)
    base_load_kw: float = Field(default=0.0)
    average_demand_kw: float = Field(default=0.0)
    load_factor_pct: float = Field(default=0.0)
    peak_to_average_ratio: float = Field(default=0.0)
    peak_hours: List[int] = Field(default_factory=list, description="Hours of peak demand")
    off_peak_hours: List[int] = Field(default_factory=list)
    weekend_base_load_kw: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class UtilityMeteringBridgeConfig(BaseModel):
    """Configuration for the Utility Metering Bridge."""

    pack_id: str = Field(default="PACK-031")
    enable_provenance: bool = Field(default=True)
    default_interval: IntervalResolution = Field(default=IntervalResolution.FIFTEEN_MIN)
    bill_reconciliation_tolerance_pct: float = Field(default=2.0, ge=0.0, le=10.0)
    max_readings_per_import: int = Field(default=500000, ge=1000)

# ---------------------------------------------------------------------------
# UtilityMeteringBridge
# ---------------------------------------------------------------------------

class UtilityMeteringBridge:
    """Smart meter and sub-metering integration.

    Manages meter hierarchies, interval data, virtual meters, bill
    reconciliation, and demand profiles.

    Attributes:
        config: Bridge configuration.
        _meters: Registered meter records.
        _virtual_meters: Virtual meter definitions.
        _readings: Buffered interval readings.

    Example:
        >>> bridge = UtilityMeteringBridge()
        >>> meter = bridge.register_meter(MeterRegistration(
        ...     name="Main Incomer", meter_type=MeterType.MAIN_UTILITY
        ... ))
        >>> bridge.ingest_interval_data(meter.meter_id, readings)
    """

    def __init__(self, config: Optional[UtilityMeteringBridgeConfig] = None) -> None:
        """Initialize the Utility Metering Bridge."""
        self.config = config or UtilityMeteringBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._meters: Dict[str, MeterRegistration] = {}
        self._virtual_meters: Dict[str, VirtualMeterDefinition] = {}
        self._readings: Dict[str, List[IntervalReading]] = {}
        self.logger.info("UtilityMeteringBridge initialized")

    # -------------------------------------------------------------------------
    # Meter Registration
    # -------------------------------------------------------------------------

    def register_meter(self, meter: MeterRegistration) -> MeterRegistration:
        """Register a physical or virtual meter.

        Args:
            meter: Meter registration data.

        Returns:
            Registered MeterRegistration.
        """
        self._meters[meter.meter_id] = meter
        self._readings[meter.meter_id] = []

        # Update parent's child list
        if meter.parent_meter_id and meter.parent_meter_id in self._meters:
            parent = self._meters[meter.parent_meter_id]
            if meter.meter_id not in parent.child_meter_ids:
                parent.child_meter_ids.append(meter.meter_id)

        self.logger.info(
            "Meter registered: %s (%s, %s, %s)",
            meter.name, meter.meter_type.value,
            meter.energy_carrier.value, meter.hierarchy_level.value,
        )
        return meter

    def get_meter_hierarchy(self) -> Dict[str, Any]:
        """Get the full meter hierarchy tree.

        Returns:
            Nested dict representing meter hierarchy.
        """
        root_meters = [
            m for m in self._meters.values() if m.parent_meter_id is None
        ]
        return {
            "facility_meters": [self._build_hierarchy_node(m) for m in root_meters],
            "total_meters": len(self._meters),
            "virtual_meters": len(self._virtual_meters),
        }

    def _build_hierarchy_node(self, meter: MeterRegistration) -> Dict[str, Any]:
        """Build a hierarchy tree node recursively."""
        children = [
            self._build_hierarchy_node(self._meters[cid])
            for cid in meter.child_meter_ids
            if cid in self._meters
        ]
        return {
            "meter_id": meter.meter_id,
            "name": meter.name,
            "type": meter.meter_type.value,
            "carrier": meter.energy_carrier.value,
            "level": meter.hierarchy_level.value,
            "children": children,
        }

    # -------------------------------------------------------------------------
    # Interval Data Ingestion
    # -------------------------------------------------------------------------

    def ingest_interval_data(
        self, meter_id: str, readings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Ingest interval meter readings.

        Args:
            meter_id: Meter identifier.
            readings: List of dicts with timestamp, value, quality fields.

        Returns:
            Dict with ingestion summary.
        """
        if meter_id not in self._meters:
            return {"meter_id": meter_id, "success": False, "message": "Meter not found"}

        start = time.monotonic()
        meter = self._meters[meter_id]
        ingested = 0

        for r in readings:
            reading = IntervalReading(
                meter_id=meter_id,
                timestamp=r.get("timestamp", utcnow()),
                value=r.get("value", 0.0) * meter.multiplier * meter.ct_ratio,
                unit=meter.unit,
                interval=meter.interval.value,
                quality=r.get("quality", "actual"),
            )
            self._readings[meter_id].append(reading)
            ingested += 1

        elapsed = (time.monotonic() - start) * 1000
        self.logger.info(
            "Interval data ingested: meter=%s, readings=%d in %.1fms",
            meter.name, ingested, elapsed,
        )
        return {
            "meter_id": meter_id,
            "success": True,
            "readings_ingested": ingested,
            "duration_ms": round(elapsed, 1),
        }

    # -------------------------------------------------------------------------
    # Virtual Meters
    # -------------------------------------------------------------------------

    def create_virtual_meter(
        self, definition: VirtualMeterDefinition,
    ) -> VirtualMeterDefinition:
        """Create a virtual meter from physical meter sources.

        Args:
            definition: Virtual meter definition.

        Returns:
            Registered VirtualMeterDefinition.
        """
        self._virtual_meters[definition.virtual_meter_id] = definition
        self.logger.info(
            "Virtual meter created: %s (formula=%s, sources=%d)",
            definition.name, definition.formula, len(definition.source_meter_ids),
        )
        return definition

    def calculate_virtual_meter(
        self, virtual_meter_id: str,
    ) -> Dict[str, Any]:
        """Calculate virtual meter value from source meters.

        Deterministic formula evaluation:
            sum: sum of all source meter latest values
            difference: first source minus all subtract meters
            weighted_sum: sum of (source_value * weight)

        Args:
            virtual_meter_id: Virtual meter identifier.

        Returns:
            Dict with calculated value and metadata.
        """
        vm = self._virtual_meters.get(virtual_meter_id)
        if vm is None:
            return {"virtual_meter_id": virtual_meter_id, "found": False}

        if vm.formula == "sum":
            total = 0.0
            for sid in vm.source_meter_ids:
                readings = self._readings.get(sid, [])
                if readings:
                    total += readings[-1].value
            return {"virtual_meter_id": virtual_meter_id, "value": total, "unit": vm.unit}

        elif vm.formula == "difference":
            # First source minus subtract meters
            main_value = 0.0
            if vm.source_meter_ids:
                main_readings = self._readings.get(vm.source_meter_ids[0], [])
                if main_readings:
                    main_value = main_readings[-1].value
            subtract_total = 0.0
            for sid in vm.subtract_meter_ids:
                readings = self._readings.get(sid, [])
                if readings:
                    subtract_total += readings[-1].value
            return {
                "virtual_meter_id": virtual_meter_id,
                "value": main_value - subtract_total,
                "unit": vm.unit,
            }

        elif vm.formula == "weighted_sum":
            total = 0.0
            weights = vm.weights or [1.0] * len(vm.source_meter_ids)
            for sid, w in zip(vm.source_meter_ids, weights):
                readings = self._readings.get(sid, [])
                if readings:
                    total += readings[-1].value * w
            return {"virtual_meter_id": virtual_meter_id, "value": total, "unit": vm.unit}

        return {"virtual_meter_id": virtual_meter_id, "error": f"Unknown formula: {vm.formula}"}

    # -------------------------------------------------------------------------
    # Bill Reconciliation
    # -------------------------------------------------------------------------

    def reconcile_bill(
        self,
        meter_id: str,
        billed_kwh: float,
        billed_cost: float,
        billing_period_start: str,
        billing_period_end: str,
    ) -> BillReconciliationResult:
        """Reconcile a utility bill against metered data.

        Deterministic calculation:
            variance = metered - billed
            variance_pct = (variance / billed) * 100

        Args:
            meter_id: Meter identifier.
            billed_kwh: Energy quantity on the bill.
            billed_cost: Cost on the bill.
            billing_period_start: Period start (ISO date string).
            billing_period_end: Period end (ISO date string).

        Returns:
            BillReconciliationResult with variance analysis.
        """
        readings = self._readings.get(meter_id, [])
        metered_kwh = sum(r.value for r in readings)

        # Deterministic variance calculation
        variance_kwh = metered_kwh - billed_kwh
        variance_pct = (variance_kwh / billed_kwh * 100.0) if billed_kwh > 0 else 0.0
        tolerance = self.config.bill_reconciliation_tolerance_pct
        within = abs(variance_pct) <= tolerance

        issues: List[str] = []
        if not within:
            issues.append(
                f"Variance {variance_pct:.1f}% exceeds tolerance {tolerance}%"
            )
        if metered_kwh == 0:
            issues.append("No metered data available for reconciliation")

        result = BillReconciliationResult(
            meter_id=meter_id,
            billing_period_start=billing_period_start,
            billing_period_end=billing_period_end,
            billed_kwh=billed_kwh,
            metered_kwh=metered_kwh,
            variance_kwh=round(variance_kwh, 2),
            variance_pct=round(variance_pct, 2),
            within_tolerance=within,
            tolerance_pct=tolerance,
            billed_cost=billed_cost,
            issues=issues,
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    # -------------------------------------------------------------------------
    # Demand Profile Analysis
    # -------------------------------------------------------------------------

    def analyze_demand_profile(
        self, meter_id: str, period: str = "",
    ) -> DemandProfile:
        """Analyze demand profile for a meter.

        Deterministic calculations:
            load_factor = (average / peak) * 100
            peak_to_average = peak / average

        Args:
            meter_id: Meter identifier.
            period: Description of analysis period.

        Returns:
            DemandProfile with load analysis.
        """
        readings = self._readings.get(meter_id, [])
        values = [r.value for r in readings if r.value > 0]

        if not values:
            return DemandProfile(meter_id=meter_id, analysis_period=period)

        peak = max(values)
        base_load = min(values)
        average = sum(values) / len(values)
        load_factor = (average / peak * 100.0) if peak > 0 else 0.0
        peak_to_avg = (peak / average) if average > 0 else 0.0

        profile = DemandProfile(
            meter_id=meter_id,
            analysis_period=period,
            peak_demand_kw=round(peak, 2),
            base_load_kw=round(base_load, 2),
            average_demand_kw=round(average, 2),
            load_factor_pct=round(load_factor, 1),
            peak_to_average_ratio=round(peak_to_avg, 2),
        )
        if self.config.enable_provenance:
            profile.provenance_hash = _compute_hash(profile)

        return profile

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    def check_health(self) -> Dict[str, Any]:
        """Check metering system health.

        Returns:
            Dict with health metrics.
        """
        total_readings = sum(len(r) for r in self._readings.values())
        active_meters = sum(1 for m in self._meters.values() if m.is_active)

        return {
            "total_meters": len(self._meters),
            "active_meters": active_meters,
            "virtual_meters": len(self._virtual_meters),
            "total_readings_buffered": total_readings,
            "status": "healthy",
        }
