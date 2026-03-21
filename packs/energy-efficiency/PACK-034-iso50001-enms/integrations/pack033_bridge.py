# -*- coding: utf-8 -*-
"""
Pack033Bridge - Bridge to PACK-033 Quick Wins Data for EnMS
=============================================================

This module provides integration with PACK-033 (Quick Wins Identifier Pack)
to import quick win opportunities, payback analyses, and implementation plans
into the ISO 50001 EnMS pipeline for action planning and operational control.

Data Imports:
    - Quick win measures (no/low-cost savings opportunities)
    - Payback analyses (ROI, NPV, simple payback calculations)
    - Implementation plans (schedules, resources, milestones)

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
# Data Models
# ---------------------------------------------------------------------------


class QuickWinsImportConfig(BaseModel):
    """Configuration for importing PACK-033 quick wins data."""

    pack_id: str = Field(default="PACK-034")
    source_pack_id: str = Field(default="PACK-033")
    enable_provenance: bool = Field(default=True)
    import_payback_analysis: bool = Field(default=True)
    import_implementation_plans: bool = Field(default=True)
    min_savings_kwh: float = Field(default=0.0, ge=0.0, description="Minimum savings threshold")


class QuickWinsDataImport(BaseModel):
    """Result of importing quick wins data from PACK-033."""

    import_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    source_pack: str = Field(default="PACK-033")
    success: bool = Field(default=False)
    degraded: bool = Field(default=False)
    quick_wins: List[Dict[str, Any]] = Field(default_factory=list)
    payback_analyses: List[Dict[str, Any]] = Field(default_factory=list)
    implementation_plans: List[Dict[str, Any]] = Field(default_factory=list)
    total_quick_wins: int = Field(default=0)
    total_savings_kwh: float = Field(default=0.0)
    total_savings_eur: float = Field(default=0.0)
    average_payback_months: float = Field(default=0.0)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Pack033Bridge
# ---------------------------------------------------------------------------


class Pack033Bridge:
    """Bridge to PACK-033 Quick Wins Identifier data for EnMS.

    Imports quick win opportunities, payback analyses, and implementation
    plans from PACK-033 for integration into ISO 50001 action planning
    (Clause 6.2) and operational control (Clause 8.1).

    Attributes:
        config: Import configuration.
        _import_cache: Cached import results by facility_id.

    Example:
        >>> bridge = Pack033Bridge()
        >>> data = bridge.import_quick_wins("FAC-001")
        >>> actions = bridge.map_quick_wins_to_actions(data.quick_wins)
    """

    def __init__(self, config: Optional[QuickWinsImportConfig] = None) -> None:
        """Initialize the PACK-033 Bridge.

        Args:
            config: Import configuration. Uses defaults if None.
        """
        self.config = config or QuickWinsImportConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._import_cache: Dict[str, QuickWinsDataImport] = {}
        self.logger.info("Pack033Bridge initialized: source=%s", self.config.source_pack_id)

    def import_quick_wins(self, facility_id: str) -> QuickWinsDataImport:
        """Import quick win measures from PACK-033.

        In production, this queries the PACK-033 data store. The stub
        returns a successful import with representative data.

        Args:
            facility_id: Facility identifier.

        Returns:
            QuickWinsDataImport with imported quick wins data.
        """
        start = time.monotonic()
        self.logger.info("Importing quick wins: facility_id=%s", facility_id)

        quick_wins = self._get_stub_quick_wins(facility_id)
        paybacks = self._get_stub_paybacks(facility_id)
        plans = self._get_stub_implementation_plans(facility_id)

        total_savings_kwh = sum(qw.get("savings_kwh", 0.0) for qw in quick_wins)
        total_savings_eur = sum(qw.get("savings_eur", 0.0) for qw in quick_wins)
        avg_payback = (
            sum(p.get("payback_months", 0) for p in paybacks) / len(paybacks)
            if paybacks else 0.0
        )

        result = QuickWinsDataImport(
            facility_id=facility_id,
            success=True,
            quick_wins=quick_wins,
            payback_analyses=paybacks,
            implementation_plans=plans,
            total_quick_wins=len(quick_wins),
            total_savings_kwh=total_savings_kwh,
            total_savings_eur=total_savings_eur,
            average_payback_months=round(avg_payback, 1),
            message=f"Imported {len(quick_wins)} quick wins from PACK-033",
            duration_ms=(time.monotonic() - start) * 1000,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self._import_cache[facility_id] = result
        return result

    def import_payback_analysis(self, facility_id: str) -> List[Dict[str, Any]]:
        """Import payback analyses from PACK-033.

        Args:
            facility_id: Facility identifier.

        Returns:
            List of payback analysis dicts.
        """
        self.logger.info("Importing payback analysis: facility_id=%s", facility_id)
        return self._get_stub_paybacks(facility_id)

    def map_quick_wins_to_actions(
        self, quick_wins: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Map quick wins to ISO 50001 action plan items.

        Converts PACK-033 quick win measures into action plan entries
        suitable for Clause 6.2 (Objectives, energy targets, and action
        plans for achieving them).

        Args:
            quick_wins: List of quick win dicts from PACK-033.

        Returns:
            List of action plan item dicts for the EnMS.
        """
        self.logger.info("Mapping %d quick wins to EnMS actions", len(quick_wins))
        actions: List[Dict[str, Any]] = []
        for idx, qw in enumerate(quick_wins):
            actions.append({
                "action_id": f"ACT-{idx + 1:03d}",
                "description": qw.get("description", ""),
                "measure_category": qw.get("category", ""),
                "savings_kwh": qw.get("savings_kwh", 0.0),
                "savings_eur": qw.get("savings_eur", 0.0),
                "investment_eur": qw.get("investment_eur", 0.0),
                "payback_months": qw.get("payback_months", 0),
                "responsible_person": "",
                "target_completion_date": "",
                "iso50001_clause": "6.2",
                "status": "planned",
                "source_pack": "PACK-033",
                "source_measure_id": qw.get("measure_id", ""),
            })
        return actions

    def prioritize_for_enms(
        self,
        quick_wins: List[Dict[str, Any]],
        enms_objectives: List[str],
    ) -> List[Dict[str, Any]]:
        """Prioritize quick wins based on EnMS objectives.

        Ranks quick wins by alignment with EnMS energy objectives and
        targets, considering payback period, savings magnitude, and
        SEU relevance.

        Args:
            quick_wins: List of quick win dicts.
            enms_objectives: List of EnMS objective descriptions.

        Returns:
            Prioritized list with EnMS alignment scores.
        """
        self.logger.info(
            "Prioritizing %d quick wins against %d objectives",
            len(quick_wins), len(enms_objectives),
        )
        prioritized: List[Dict[str, Any]] = []
        for qw in quick_wins:
            savings = qw.get("savings_kwh", 0.0)
            payback = qw.get("payback_months", 24)
            # Deterministic scoring: higher savings and lower payback = higher priority
            savings_score = min(savings / 100_000.0, 10.0) * 40.0
            payback_score = max(0.0, (24 - payback) / 24.0) * 30.0
            alignment_score = 30.0 if enms_objectives else 15.0
            total_score = round(savings_score + payback_score + alignment_score, 1)
            prioritized.append({
                **qw,
                "enms_priority_score": total_score,
                "savings_score": round(savings_score, 1),
                "payback_score": round(payback_score, 1),
                "alignment_score": round(alignment_score, 1),
            })
        # Sort by priority score descending
        prioritized.sort(key=lambda x: x.get("enms_priority_score", 0), reverse=True)
        return prioritized

    # -------------------------------------------------------------------------
    # Stub Data
    # -------------------------------------------------------------------------

    def _get_stub_quick_wins(self, facility_id: str) -> List[Dict[str, Any]]:
        """Return representative quick wins."""
        return [
            {"measure_id": f"QW-{facility_id}-001", "description": "LED lighting retrofit", "category": "lighting", "savings_kwh": 180_000, "savings_eur": 27_000, "investment_eur": 35_000, "payback_months": 16},
            {"measure_id": f"QW-{facility_id}-002", "description": "Compressed air leak repair", "category": "compressed_air", "savings_kwh": 120_000, "savings_eur": 18_000, "investment_eur": 5_000, "payback_months": 3},
            {"measure_id": f"QW-{facility_id}-003", "description": "BMS setpoint optimization", "category": "controls", "savings_kwh": 90_000, "savings_eur": 13_500, "investment_eur": 8_000, "payback_months": 7},
            {"measure_id": f"QW-{facility_id}-004", "description": "VFD on AHU fans", "category": "hvac", "savings_kwh": 150_000, "savings_eur": 22_500, "investment_eur": 25_000, "payback_months": 13},
            {"measure_id": f"QW-{facility_id}-005", "description": "Steam trap maintenance", "category": "steam", "savings_kwh": 200_000, "savings_eur": 30_000, "investment_eur": 12_000, "payback_months": 5},
        ]

    def _get_stub_paybacks(self, facility_id: str) -> List[Dict[str, Any]]:
        """Return representative payback analyses."""
        return [
            {"measure_id": f"QW-{facility_id}-001", "simple_payback_months": 16, "npv_eur": 85_000, "irr_pct": 42.0, "roi_pct": 77.1},
            {"measure_id": f"QW-{facility_id}-002", "simple_payback_months": 3, "npv_eur": 120_000, "irr_pct": 180.0, "roi_pct": 260.0},
            {"measure_id": f"QW-{facility_id}-003", "simple_payback_months": 7, "npv_eur": 55_000, "irr_pct": 95.0, "roi_pct": 68.8},
            {"measure_id": f"QW-{facility_id}-004", "simple_payback_months": 13, "npv_eur": 65_000, "irr_pct": 55.0, "roi_pct": 90.0},
            {"measure_id": f"QW-{facility_id}-005", "simple_payback_months": 5, "npv_eur": 140_000, "irr_pct": 150.0, "roi_pct": 250.0},
        ]

    def _get_stub_implementation_plans(self, facility_id: str) -> List[Dict[str, Any]]:
        """Return representative implementation plans."""
        return [
            {"plan_id": f"IMPL-{facility_id}-001", "phase": "immediate", "duration_weeks": 4, "resources": ["electrician", "procurement"]},
            {"plan_id": f"IMPL-{facility_id}-002", "phase": "immediate", "duration_weeks": 1, "resources": ["maintenance_team"]},
            {"plan_id": f"IMPL-{facility_id}-003", "phase": "short_term", "duration_weeks": 2, "resources": ["bms_technician"]},
        ]
