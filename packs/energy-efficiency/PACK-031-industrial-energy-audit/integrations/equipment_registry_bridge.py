# -*- coding: utf-8 -*-
"""
EquipmentRegistryBridge - Asset Management and CMMS Integration for PACK-031
===============================================================================

This module provides integration with asset management systems and Computerized
Maintenance Management Systems (CMMS). It handles equipment master data import,
nameplate data extraction, maintenance schedule integration, run-hour tracking,
equipment lifecycle management, and replacement planning.

Equipment Categories for Energy Audits:
    - Motors and drives (pumps, fans, compressors)
    - Boilers and furnaces
    - Chillers and cooling towers
    - Air compressors
    - HVAC systems
    - Lighting systems
    - Transformers
    - Heat exchangers
    - Steam traps

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
from datetime import datetime, date, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class EquipmentCategory(str, Enum):
    """Major equipment categories relevant to energy audits."""

    MOTOR = "motor"
    PUMP = "pump"
    FAN = "fan"
    COMPRESSOR = "compressor"
    BOILER = "boiler"
    FURNACE = "furnace"
    CHILLER = "chiller"
    COOLING_TOWER = "cooling_tower"
    AIR_COMPRESSOR = "air_compressor"
    HVAC_AHU = "hvac_ahu"
    LIGHTING = "lighting"
    TRANSFORMER = "transformer"
    HEAT_EXCHANGER = "heat_exchanger"
    STEAM_TRAP = "steam_trap"
    VFD = "variable_frequency_drive"
    HEAT_PUMP = "heat_pump"
    CHP = "chp_cogeneration"
    DRYER = "dryer"
    KILN = "kiln"
    OVEN = "oven"


class EquipmentCondition(str, Enum):
    """Equipment condition assessment grades."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"
    DECOMMISSIONED = "decommissioned"


class EfficiencyRating(str, Enum):
    """Equipment efficiency rating (IE class for motors, etc.)."""

    IE1_STANDARD = "IE1"
    IE2_HIGH = "IE2"
    IE3_PREMIUM = "IE3"
    IE4_SUPER_PREMIUM = "IE4"
    IE5_ULTRA_PREMIUM = "IE5"
    NOT_RATED = "not_rated"
    CUSTOM = "custom"


class MaintenanceType(str, Enum):
    """Types of maintenance activities."""

    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    CONDITION_BASED = "condition_based"
    OVERHAUL = "overhaul"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class NameplateData(BaseModel):
    """Equipment nameplate data for energy audit assessment."""

    rated_power_kw: float = Field(default=0.0, ge=0)
    rated_voltage_v: float = Field(default=0.0, ge=0)
    rated_current_a: float = Field(default=0.0, ge=0)
    rated_speed_rpm: Optional[float] = Field(None, ge=0)
    rated_efficiency_pct: float = Field(default=0.0, ge=0, le=100)
    power_factor: float = Field(default=0.0, ge=0, le=1)
    frequency_hz: float = Field(default=50.0)
    fuel_type: Optional[str] = Field(None)
    thermal_capacity_kw: Optional[float] = Field(None, ge=0)
    cop_or_eer: Optional[float] = Field(None, ge=0, description="COP or EER for chillers/heat pumps")
    flow_rate_m3h: Optional[float] = Field(None, ge=0)
    pressure_bar: Optional[float] = Field(None, ge=0)


class EquipmentRecord(BaseModel):
    """Equipment master data record."""

    equipment_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="")
    category: EquipmentCategory = Field(default=EquipmentCategory.MOTOR)
    manufacturer: str = Field(default="")
    model: str = Field(default="")
    serial_number: str = Field(default="")
    installation_date: Optional[date] = Field(None)
    nameplate: NameplateData = Field(default_factory=NameplateData)
    efficiency_rating: EfficiencyRating = Field(default=EfficiencyRating.NOT_RATED)
    condition: EquipmentCondition = Field(default=EquipmentCondition.GOOD)
    location: str = Field(default="")
    zone: str = Field(default="")
    process: str = Field(default="", description="Production process this serves")
    run_hours: float = Field(default=0.0, ge=0)
    annual_run_hours: float = Field(default=0.0, ge=0)
    load_factor_pct: float = Field(default=100.0, ge=0, le=100)
    meter_id: Optional[str] = Field(None, description="Associated energy meter")
    parent_equipment_id: Optional[str] = Field(None, description="Parent assembly")
    is_critical: bool = Field(default=False)
    replacement_candidate: bool = Field(default=False)
    estimated_annual_kwh: float = Field(default=0.0, ge=0)


class MaintenanceRecord(BaseModel):
    """Maintenance activity record."""

    record_id: str = Field(default_factory=_new_uuid)
    equipment_id: str = Field(default="")
    maintenance_type: MaintenanceType = Field(default=MaintenanceType.PREVENTIVE)
    description: str = Field(default="")
    scheduled_date: Optional[date] = Field(None)
    completed_date: Optional[date] = Field(None)
    technician: str = Field(default="")
    cost_eur: float = Field(default=0.0, ge=0)
    run_hours_at_service: float = Field(default=0.0, ge=0)
    findings: str = Field(default="")
    energy_impact_notes: str = Field(default="")


class ReplacementAssessment(BaseModel):
    """Equipment replacement assessment for energy savings."""

    assessment_id: str = Field(default_factory=_new_uuid)
    equipment_id: str = Field(default="")
    current_efficiency_pct: float = Field(default=0.0)
    proposed_efficiency_pct: float = Field(default=0.0)
    annual_energy_current_kwh: float = Field(default=0.0)
    annual_energy_proposed_kwh: float = Field(default=0.0)
    annual_savings_kwh: float = Field(default=0.0)
    annual_savings_eur: float = Field(default=0.0)
    replacement_cost_eur: float = Field(default=0.0)
    simple_payback_years: float = Field(default=0.0)
    recommended: bool = Field(default=False)
    provenance_hash: str = Field(default="")


class EquipmentRegistryBridgeConfig(BaseModel):
    """Configuration for the Equipment Registry Bridge."""

    pack_id: str = Field(default="PACK-031")
    enable_provenance: bool = Field(default=True)
    replacement_payback_threshold_years: float = Field(
        default=5.0, ge=0.5, description="Max payback for recommendation"
    )
    default_electricity_cost_eur_per_kwh: float = Field(default=0.15, ge=0.0)


# ---------------------------------------------------------------------------
# EquipmentRegistryBridge
# ---------------------------------------------------------------------------


class EquipmentRegistryBridge:
    """Asset management and CMMS integration for energy audits.

    Manages equipment master data, nameplate information, maintenance
    schedules, run-hour tracking, and replacement assessments.

    Attributes:
        config: Bridge configuration.
        _equipment: Equipment registry.
        _maintenance: Maintenance records.

    Example:
        >>> bridge = EquipmentRegistryBridge()
        >>> eq = bridge.register_equipment(EquipmentRecord(
        ...     name="Pump P-101", category=EquipmentCategory.PUMP,
        ...     nameplate=NameplateData(rated_power_kw=75)
        ... ))
        >>> assessment = bridge.assess_replacement(eq.equipment_id, 95.0, 80000)
    """

    def __init__(self, config: Optional[EquipmentRegistryBridgeConfig] = None) -> None:
        """Initialize the Equipment Registry Bridge."""
        self.config = config or EquipmentRegistryBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._equipment: Dict[str, EquipmentRecord] = {}
        self._maintenance: Dict[str, List[MaintenanceRecord]] = {}
        self.logger.info("EquipmentRegistryBridge initialized")

    # -------------------------------------------------------------------------
    # Equipment Registration
    # -------------------------------------------------------------------------

    def register_equipment(self, record: EquipmentRecord) -> EquipmentRecord:
        """Register equipment in the registry.

        Auto-calculates estimated annual kWh if not provided:
            annual_kwh = rated_power_kw * annual_run_hours * (load_factor / 100)

        Args:
            record: Equipment data.

        Returns:
            Registered EquipmentRecord with calculated fields.
        """
        # Deterministic annual energy estimation
        if record.estimated_annual_kwh == 0 and record.nameplate.rated_power_kw > 0:
            record.estimated_annual_kwh = round(
                record.nameplate.rated_power_kw
                * record.annual_run_hours
                * (record.load_factor_pct / 100.0),
                2,
            )

        self._equipment[record.equipment_id] = record
        self._maintenance[record.equipment_id] = []

        self.logger.info(
            "Equipment registered: %s (%s, %.1f kW, est. %.0f kWh/yr)",
            record.name, record.category.value,
            record.nameplate.rated_power_kw, record.estimated_annual_kwh,
        )
        return record

    def import_equipment_batch(
        self, records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Import a batch of equipment records.

        Args:
            records: List of equipment data dicts.

        Returns:
            Dict with import summary.
        """
        start = time.monotonic()
        imported = 0
        failed = 0

        for data in records:
            try:
                nameplate_data = data.pop("nameplate", {})
                record = EquipmentRecord(
                    nameplate=NameplateData(**nameplate_data),
                    **data,
                )
                self.register_equipment(record)
                imported += 1
            except Exception as exc:
                self.logger.warning("Equipment import failed: %s", exc)
                failed += 1

        elapsed = (time.monotonic() - start) * 1000
        return {
            "total": len(records),
            "imported": imported,
            "failed": failed,
            "duration_ms": round(elapsed, 1),
        }

    # -------------------------------------------------------------------------
    # Equipment Queries
    # -------------------------------------------------------------------------

    def get_equipment_by_category(
        self, category: EquipmentCategory,
    ) -> List[Dict[str, Any]]:
        """Get equipment filtered by category.

        Args:
            category: Equipment category.

        Returns:
            List of equipment summaries.
        """
        return [
            {
                "equipment_id": eq.equipment_id,
                "name": eq.name,
                "rated_power_kw": eq.nameplate.rated_power_kw,
                "efficiency_rating": eq.efficiency_rating.value,
                "condition": eq.condition.value,
                "annual_kwh": eq.estimated_annual_kwh,
                "run_hours": eq.annual_run_hours,
                "location": eq.location,
            }
            for eq in self._equipment.values()
            if eq.category == category
        ]

    def get_top_consumers(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Get top energy consuming equipment.

        Args:
            top_n: Number of top consumers to return.

        Returns:
            List sorted by estimated annual kWh descending.
        """
        sorted_eq = sorted(
            self._equipment.values(),
            key=lambda e: e.estimated_annual_kwh,
            reverse=True,
        )
        return [
            {
                "equipment_id": eq.equipment_id,
                "name": eq.name,
                "category": eq.category.value,
                "annual_kwh": eq.estimated_annual_kwh,
                "rated_power_kw": eq.nameplate.rated_power_kw,
                "condition": eq.condition.value,
            }
            for eq in sorted_eq[:top_n]
        ]

    # -------------------------------------------------------------------------
    # Maintenance Integration
    # -------------------------------------------------------------------------

    def add_maintenance_record(self, record: MaintenanceRecord) -> MaintenanceRecord:
        """Add a maintenance record for equipment.

        Args:
            record: Maintenance record data.

        Returns:
            Recorded MaintenanceRecord.
        """
        if record.equipment_id not in self._maintenance:
            self._maintenance[record.equipment_id] = []
        self._maintenance[record.equipment_id].append(record)

        # Update run hours on equipment
        if record.run_hours_at_service > 0 and record.equipment_id in self._equipment:
            self._equipment[record.equipment_id].run_hours = record.run_hours_at_service

        return record

    def get_maintenance_history(
        self, equipment_id: str,
    ) -> List[Dict[str, Any]]:
        """Get maintenance history for equipment.

        Args:
            equipment_id: Equipment identifier.

        Returns:
            List of maintenance record summaries.
        """
        records = self._maintenance.get(equipment_id, [])
        return [
            {
                "record_id": r.record_id,
                "type": r.maintenance_type.value,
                "description": r.description,
                "scheduled_date": r.scheduled_date.isoformat() if r.scheduled_date else None,
                "completed_date": r.completed_date.isoformat() if r.completed_date else None,
                "cost_eur": r.cost_eur,
            }
            for r in records
        ]

    # -------------------------------------------------------------------------
    # Replacement Assessment
    # -------------------------------------------------------------------------

    def assess_replacement(
        self,
        equipment_id: str,
        proposed_efficiency_pct: float,
        replacement_cost_eur: float,
        electricity_cost_override: Optional[float] = None,
    ) -> Optional[ReplacementAssessment]:
        """Assess equipment replacement for energy savings.

        Deterministic calculations:
            savings_kwh = current_kwh - proposed_kwh
            savings_eur = savings_kwh * electricity_cost
            payback = replacement_cost / savings_eur

        Args:
            equipment_id: Equipment identifier.
            proposed_efficiency_pct: Efficiency of replacement equipment.
            replacement_cost_eur: Cost of replacement.
            electricity_cost_override: Override electricity cost per kWh.

        Returns:
            ReplacementAssessment, or None if equipment not found.
        """
        eq = self._equipment.get(equipment_id)
        if eq is None:
            return None

        current_eff = eq.nameplate.rated_efficiency_pct or 80.0
        elec_cost = electricity_cost_override or self.config.default_electricity_cost_eur_per_kwh

        # Deterministic calculations
        current_kwh = eq.estimated_annual_kwh
        if current_eff > 0 and proposed_efficiency_pct > 0:
            # Energy proportional to 1/efficiency
            proposed_kwh = current_kwh * (current_eff / proposed_efficiency_pct)
        else:
            proposed_kwh = current_kwh

        savings_kwh = current_kwh - proposed_kwh
        savings_eur = savings_kwh * elec_cost
        payback = (replacement_cost_eur / savings_eur) if savings_eur > 0 else 999.0

        assessment = ReplacementAssessment(
            equipment_id=equipment_id,
            current_efficiency_pct=current_eff,
            proposed_efficiency_pct=proposed_efficiency_pct,
            annual_energy_current_kwh=round(current_kwh, 2),
            annual_energy_proposed_kwh=round(proposed_kwh, 2),
            annual_savings_kwh=round(savings_kwh, 2),
            annual_savings_eur=round(savings_eur, 2),
            replacement_cost_eur=replacement_cost_eur,
            simple_payback_years=round(payback, 2),
            recommended=payback <= self.config.replacement_payback_threshold_years,
        )
        if self.config.enable_provenance:
            assessment.provenance_hash = _compute_hash(assessment)

        return assessment

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    def check_health(self) -> Dict[str, Any]:
        """Check equipment registry health.

        Returns:
            Dict with health metrics.
        """
        total = len(self._equipment)
        poor_condition = sum(
            1 for eq in self._equipment.values()
            if eq.condition in (EquipmentCondition.POOR, EquipmentCondition.CRITICAL)
        )
        replacement_candidates = sum(
            1 for eq in self._equipment.values() if eq.replacement_candidate
        )

        return {
            "total_equipment": total,
            "poor_condition": poor_condition,
            "replacement_candidates": replacement_candidates,
            "total_maintenance_records": sum(len(r) for r in self._maintenance.values()),
            "status": "healthy",
        }
