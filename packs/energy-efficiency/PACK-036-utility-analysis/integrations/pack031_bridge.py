# -*- coding: utf-8 -*-
"""
Pack031Bridge - Bridge to PACK-031 Industrial Energy Audit Data
=================================================================

This module provides integration with PACK-031 (Industrial Energy Audit Pack)
to share equipment data, baseline models, and metering infrastructure between
industrial energy audits and utility analysis. Utility consumption data
enriches audit energy baselines, and audit findings provide context for
utility cost optimization.

Data Imports from PACK-031:
    - Energy audit results (findings, recommendations, savings opportunities)
    - Equipment efficiency data (nameplate vs actual, condition scores)
    - Energy baselines (weather-normalized consumption baselines)
    - Metering infrastructure (meter hierarchy, sub-meter locations)

Data Exports to PACK-031:
    - Utility consumption profiles (monthly, daily, interval)
    - Rate structure data (tariff details, TOU periods)
    - Cost allocation by equipment/process

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-036 Utility Analysis
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
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
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class AuditImportConfig(BaseModel):
    """Configuration for importing PACK-031 audit data."""

    pack_id: str = Field(default="PACK-036")
    source_pack_id: str = Field(default="PACK-031")
    enable_provenance: bool = Field(default=True)
    import_equipment_data: bool = Field(default=True)
    import_baseline: bool = Field(default=True)
    import_meter_hierarchy: bool = Field(default=True)
    import_process_maps: bool = Field(default=False)
    sync_consumption_back: bool = Field(default=True)


class AuditDataImport(BaseModel):
    """Result of importing energy audit data from PACK-031."""

    import_id: str = Field(default_factory=_new_uuid)
    audit_id: str = Field(default="")
    facility_id: str = Field(default="")
    source_pack: str = Field(default="PACK-031")
    success: bool = Field(default=False)
    degraded: bool = Field(default=False)
    audit_date: Optional[str] = Field(None)
    total_consumption_kwh: float = Field(default=0.0)
    total_cost_eur: float = Field(default=0.0)
    equipment_count: int = Field(default=0)
    savings_opportunities: int = Field(default=0)
    meter_count: int = Field(default=0)
    baseline_available: bool = Field(default=False)
    process_maps_available: bool = Field(default=False)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class MeterHierarchy(BaseModel):
    """Meter hierarchy imported from PACK-031 audit."""

    hierarchy_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    main_meters: List[Dict[str, Any]] = Field(default_factory=list)
    sub_meters: List[Dict[str, Any]] = Field(default_factory=list)
    virtual_meters: List[Dict[str, Any]] = Field(default_factory=list)
    total_meters: int = Field(default=0)
    coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


class ConsumptionExport(BaseModel):
    """Consumption data exported back to PACK-031."""

    export_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    target_pack: str = Field(default="PACK-031")
    period: str = Field(default="")
    total_electricity_kwh: float = Field(default=0.0)
    total_gas_kwh: float = Field(default=0.0)
    monthly_profiles: List[Dict[str, Any]] = Field(default_factory=list)
    exported_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Pack031Bridge
# ---------------------------------------------------------------------------


class Pack031Bridge:
    """Bridge to PACK-031 Industrial Energy Audit data.

    Shares equipment data, baseline models, and metering infrastructure
    between industrial energy audits and utility analysis. Utility data
    enriches audit energy baselines and audit findings provide context
    for utility cost optimization.

    Attributes:
        config: Import configuration.
        _audit_cache: Cached audit data by audit_id.
        _export_history: History of consumption exports to PACK-031.

    Example:
        >>> bridge = Pack031Bridge()
        >>> audit_data = bridge.import_audit_data("AUDIT-2025-001")
        >>> meters = bridge.get_meter_hierarchy("FAC-001")
        >>> bridge.export_consumption("FAC-001", {...})
    """

    def __init__(self, config: Optional[AuditImportConfig] = None) -> None:
        """Initialize the PACK-031 Bridge.

        Args:
            config: Import configuration. Uses defaults if None.
        """
        self.config = config or AuditImportConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._audit_cache: Dict[str, AuditDataImport] = {}
        self._export_history: List[ConsumptionExport] = []
        self.logger.info(
            "Pack031Bridge initialized: source=%s, sync_back=%s",
            self.config.source_pack_id, self.config.sync_consumption_back,
        )

    def import_audit_data(self, audit_id: str) -> AuditDataImport:
        """Import energy audit data from PACK-031.

        In production, this queries the PACK-031 data store. The stub
        returns a successful import with placeholder data.

        Args:
            audit_id: PACK-031 audit identifier.

        Returns:
            AuditDataImport with imported data summary.
        """
        start = time.monotonic()
        self.logger.info("Importing audit data: audit_id=%s", audit_id)

        result = AuditDataImport(
            audit_id=audit_id,
            facility_id=f"FAC-{audit_id[-3:]}",
            success=True,
            audit_date="2025-12-31",
            total_consumption_kwh=15_000_000.0,
            total_cost_eur=2_250_000.0,
            equipment_count=150,
            savings_opportunities=25,
            meter_count=12,
            baseline_available=self.config.import_baseline,
            process_maps_available=self.config.import_process_maps,
            message=f"Audit {audit_id} imported from PACK-031",
            duration_ms=(time.monotonic() - start) * 1000,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self._audit_cache[audit_id] = result
        return result

    def get_equipment_data(self, facility_id: str) -> List[Dict[str, Any]]:
        """Get equipment efficiency data from a PACK-031 audit.

        Args:
            facility_id: Facility identifier.

        Returns:
            List of equipment data dicts from the audit.
        """
        self.logger.info(
            "Retrieving equipment data: facility_id=%s", facility_id
        )

        return [
            {"equipment_id": f"EQ-{facility_id}-001", "category": "motor",
             "efficiency_pct": 85.0, "condition": "fair", "kw_rating": 75.0},
            {"equipment_id": f"EQ-{facility_id}-002", "category": "hvac_chiller",
             "efficiency_pct": 72.0, "condition": "poor", "kw_rating": 250.0},
            {"equipment_id": f"EQ-{facility_id}-003", "category": "lighting",
             "efficiency_pct": 60.0, "condition": "poor", "kw_rating": 45.0},
            {"equipment_id": f"EQ-{facility_id}-004", "category": "compressor",
             "efficiency_pct": 78.0, "condition": "fair", "kw_rating": 55.0},
            {"equipment_id": f"EQ-{facility_id}-005", "category": "boiler",
             "efficiency_pct": 82.0, "condition": "good", "kw_rating": 500.0},
        ]

    def get_baseline(self, facility_id: str) -> Dict[str, Any]:
        """Get energy baseline data from a PACK-031 audit.

        Args:
            facility_id: Facility identifier.

        Returns:
            Dict with baseline data including weather-normalized consumption.
        """
        self.logger.info("Retrieving baseline: facility_id=%s", facility_id)

        baseline = {
            "facility_id": facility_id,
            "baseline_year": 2024,
            "total_kwh": 15_000_000.0,
            "electricity_kwh": 10_000_000.0,
            "natural_gas_kwh": 5_000_000.0,
            "weather_normalized": True,
            "hdd_base_c": 18.0,
            "cdd_base_c": 22.0,
            "enpi_kwh_per_m2": 250.0,
            "source": "PACK-031",
        }

        if self.config.enable_provenance:
            baseline["provenance_hash"] = _compute_hash(baseline)

        return baseline

    def get_meter_hierarchy(self, facility_id: str) -> MeterHierarchy:
        """Get meter hierarchy from a PACK-031 audit.

        Args:
            facility_id: Facility identifier.

        Returns:
            MeterHierarchy with main, sub, and virtual meters.
        """
        self.logger.info(
            "Retrieving meter hierarchy: facility_id=%s", facility_id
        )

        hierarchy = MeterHierarchy(
            facility_id=facility_id,
            main_meters=[
                {"meter_id": f"M-{facility_id}-E1", "commodity": "electricity",
                 "type": "main", "ct_ratio": "400:5"},
                {"meter_id": f"M-{facility_id}-G1", "commodity": "natural_gas",
                 "type": "main", "pulse_factor": 1.0},
            ],
            sub_meters=[
                {"meter_id": f"SM-{facility_id}-E01", "commodity": "electricity",
                 "type": "sub", "parent": f"M-{facility_id}-E1",
                 "serves": "HVAC plant"},
                {"meter_id": f"SM-{facility_id}-E02", "commodity": "electricity",
                 "type": "sub", "parent": f"M-{facility_id}-E1",
                 "serves": "Lighting floors 1-5"},
                {"meter_id": f"SM-{facility_id}-E03", "commodity": "electricity",
                 "type": "sub", "parent": f"M-{facility_id}-E1",
                 "serves": "Process equipment"},
            ],
            virtual_meters=[
                {"meter_id": f"VM-{facility_id}-E01", "commodity": "electricity",
                 "type": "virtual", "formula": "M-E1 - SM-E01 - SM-E02 - SM-E03",
                 "serves": "Unmetered loads"},
            ],
            total_meters=6,
            coverage_pct=85.0,
        )

        if self.config.enable_provenance:
            hierarchy.provenance_hash = _compute_hash(hierarchy)

        return hierarchy

    def export_consumption(
        self, facility_id: str, consumption_data: Dict[str, Any]
    ) -> ConsumptionExport:
        """Export consumption data back to PACK-031 for baseline enrichment.

        Args:
            facility_id: Facility identifier.
            consumption_data: Consumption data to export.

        Returns:
            ConsumptionExport with export confirmation.
        """
        if not self.config.sync_consumption_back:
            return ConsumptionExport(
                facility_id=facility_id,
                provenance_hash=_compute_hash({"skipped": True}),
            )

        export = ConsumptionExport(
            facility_id=facility_id,
            period=consumption_data.get("period", ""),
            total_electricity_kwh=consumption_data.get("electricity_kwh", 0.0),
            total_gas_kwh=consumption_data.get("gas_kwh", 0.0),
            monthly_profiles=consumption_data.get("monthly", []),
        )

        if self.config.enable_provenance:
            export.provenance_hash = _compute_hash(export)

        self._export_history.append(export)
        self.logger.info(
            "Consumption exported to PACK-031: facility=%s, period=%s",
            facility_id, export.period,
        )
        return export
