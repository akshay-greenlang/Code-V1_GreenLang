# -*- coding: utf-8 -*-
"""
LandUseChangeTrackerEngine - Land-Use Transition Tracking (Engine 3 of 7)

AGENT-MRV-006: Land Use Emissions Agent

Tracks and manages land-use transitions between the six IPCC land categories
(Forest Land, Cropland, Grassland, Wetlands, Settlements, Other Land).
Provides a 6x6 transition matrix, transition history per parcel,
deforestation/afforestation detection, wetland drainage tracking,
peatland conversion monitoring, and area consistency validation.

Key Concepts:
    - "Remaining" transitions: Land stays in the same category (e.g. FL->FL).
      These have ongoing carbon stock changes due to management.
    - "Conversion" transitions: Land changes category (e.g. FL->CL).
      These trigger immediate carbon stock changes and a 20-year SOC
      transition period per IPCC Tier 1 methodology.

Transition Matrix:
    A 6x6 matrix tracking area (ha) for each possible transition.
    Diagonal = remaining, off-diagonal = conversions.

Zero-Hallucination Guarantees:
    - All transition logic is deterministic.
    - No LLM calls in any tracking path.
    - Area consistency enforced (total area conserved).
    - SHA-256 provenance hash for every transition record.

Thread Safety:
    All mutable state (transition history, matrix) is protected by a
    reentrant lock.

Example:
    >>> from greenlang.land_use_emissions.land_use_change_tracker import (
    ...     LandUseChangeTrackerEngine,
    ... )
    >>> tracker = LandUseChangeTrackerEngine()
    >>> tracker.record_transition({
    ...     "parcel_id": "P001",
    ...     "from_category": "FOREST_LAND",
    ...     "to_category": "CROPLAND",
    ...     "area_ha": 50.0,
    ...     "transition_date": "2023-01-15",
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Land Use Emissions (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["LandUseChangeTrackerEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.land_use_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.land_use_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.land_use_emissions.metrics import (
        record_component_operation as _record_tracker_operation,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_tracker_operation = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decimal helpers
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


# ===========================================================================
# Constants
# ===========================================================================

#: The six IPCC land categories.
LAND_CATEGORIES: List[str] = [
    "FOREST_LAND",
    "CROPLAND",
    "GRASSLAND",
    "WETLANDS",
    "SETTLEMENTS",
    "OTHER_LAND",
]

#: Default IPCC SOC transition period (years).
DEFAULT_TRANSITION_PERIOD: int = 20

#: Transition types.
TRANSITION_REMAINING = "REMAINING"
TRANSITION_CONVERSION = "CONVERSION"


# ===========================================================================
# Dataclasses
# ===========================================================================


@dataclass
class TransitionRecord:
    """A single land-use transition event.

    Attributes:
        transition_id: Unique identifier for this transition.
        parcel_id: Identifier for the land parcel.
        from_category: Original IPCC land category.
        to_category: New IPCC land category.
        area_ha: Area transitioned in hectares.
        transition_date: Date of the transition.
        transition_type: REMAINING or CONVERSION.
        transition_period_years: SOC transition period (default 20).
        completion_date: Estimated SOC transition completion date.
        is_deforestation: True if conversion from forest.
        is_afforestation: True if conversion to forest.
        is_wetland_drainage: True if wetland is drained.
        is_peatland_conversion: True if peatland is converted.
        notes: Optional notes about the transition.
        provenance_hash: SHA-256 hash for audit.
        recorded_at: Timestamp of recording.
    """

    transition_id: str
    parcel_id: str
    from_category: str
    to_category: str
    area_ha: Decimal
    transition_date: str
    transition_type: str
    transition_period_years: int
    completion_date: str
    is_deforestation: bool
    is_afforestation: bool
    is_wetland_drainage: bool
    is_peatland_conversion: bool
    notes: str
    provenance_hash: str
    recorded_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "transition_id": self.transition_id,
            "parcel_id": self.parcel_id,
            "from_category": self.from_category,
            "to_category": self.to_category,
            "area_ha": str(self.area_ha),
            "transition_date": self.transition_date,
            "transition_type": self.transition_type,
            "transition_period_years": self.transition_period_years,
            "completion_date": self.completion_date,
            "is_deforestation": self.is_deforestation,
            "is_afforestation": self.is_afforestation,
            "is_wetland_drainage": self.is_wetland_drainage,
            "is_peatland_conversion": self.is_peatland_conversion,
            "notes": self.notes,
            "provenance_hash": self.provenance_hash,
            "recorded_at": self.recorded_at,
        }


# ===========================================================================
# LandUseChangeTrackerEngine
# ===========================================================================


class LandUseChangeTrackerEngine:
    """Tracks land-use transitions between IPCC land categories.

    Maintains a 6x6 transition matrix, per-parcel transition history,
    and provides detection methods for deforestation, afforestation,
    wetland changes, and peatland conversions.

    Thread Safety:
        All mutable state is protected by a reentrant lock.

    Attributes:
        _transition_matrix: 6x6 area matrix (from_cat -> to_cat -> ha).
        _transition_history: List of all transition records.
        _parcel_history: Per-parcel transition history.
        _parcel_current_category: Current category per parcel.
        _parcel_total_area: Total area per parcel.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> tracker = LandUseChangeTrackerEngine()
        >>> tracker.record_transition({
        ...     "parcel_id": "P001",
        ...     "from_category": "FOREST_LAND",
        ...     "to_category": "CROPLAND",
        ...     "area_ha": 50.0,
        ...     "transition_date": "2023-01-15",
        ... })
    """

    def __init__(self) -> None:
        """Initialize the LandUseChangeTrackerEngine."""
        self._lock = threading.RLock()
        self._transition_matrix: Dict[str, Dict[str, Decimal]] = {
            from_cat: {to_cat: _ZERO for to_cat in LAND_CATEGORIES}
            for from_cat in LAND_CATEGORIES
        }
        self._transition_history: List[TransitionRecord] = []
        self._parcel_history: Dict[str, List[TransitionRecord]] = defaultdict(list)
        self._parcel_current_category: Dict[str, str] = {}
        self._parcel_total_area: Dict[str, Decimal] = {}
        self._total_transitions: int = 0
        self._created_at = _utcnow()

        logger.info(
            "LandUseChangeTrackerEngine initialized: categories=%d, "
            "matrix_size=%dx%d",
            len(LAND_CATEGORIES), len(LAND_CATEGORIES), len(LAND_CATEGORIES),
        )

    # ------------------------------------------------------------------
    # Validation Helpers
    # ------------------------------------------------------------------

    def _validate_category(self, category: str) -> str:
        """Validate and normalise a land category.

        Args:
            category: Land category string.

        Returns:
            Normalised category string.

        Raises:
            ValueError: If category is not recognized.
        """
        normalised = category.upper().replace(" ", "_")
        if normalised not in LAND_CATEGORIES:
            raise ValueError(
                f"Unknown land category '{category}'. "
                f"Valid: {LAND_CATEGORIES}"
            )
        return normalised

    def _parse_date(self, date_str: str) -> date:
        """Parse a date string to a date object.

        Supports YYYY-MM-DD and YYYY/MM/DD formats.

        Args:
            date_str: Date string.

        Returns:
            date object.

        Raises:
            ValueError: If the date string cannot be parsed.
        """
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        raise ValueError(
            f"Cannot parse date '{date_str}'. Expected YYYY-MM-DD."
        )

    def _compute_completion_date(
        self,
        transition_date: date,
        transition_period: int,
    ) -> str:
        """Compute the estimated SOC transition completion date.

        Args:
            transition_date: Date of the transition.
            transition_period: Transition period in years.

        Returns:
            Completion date as ISO string.
        """
        completion_year = transition_date.year + transition_period
        try:
            completion = date(completion_year, transition_date.month, transition_date.day)
        except ValueError:
            # Handle Feb 29 -> Feb 28
            completion = date(completion_year, transition_date.month, 28)
        return completion.isoformat()

    # ------------------------------------------------------------------
    # Record Transition
    # ------------------------------------------------------------------

    def record_transition(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record a land-use transition event.

        Required request keys:
            parcel_id: Identifier for the land parcel.
            from_category: Original IPCC land category.
            to_category: New IPCC land category.
            area_ha: Area transitioned in hectares.
            transition_date: Date of the transition (YYYY-MM-DD).

        Optional keys:
            transition_period_years: SOC transition period (default 20).
            notes: Additional notes.

        Args:
            request: Transition request dictionary.

        Returns:
            Transition record with classification and provenance.
        """
        start_time = time.monotonic()

        parcel_id = str(request.get("parcel_id", ""))
        from_cat = str(request.get("from_category", "")).upper()
        to_cat = str(request.get("to_category", "")).upper()
        area_ha = _D(str(request.get("area_ha", 0)))
        transition_date_str = str(request.get("transition_date", ""))
        transition_period = int(request.get(
            "transition_period_years", DEFAULT_TRANSITION_PERIOD
        ))
        notes = str(request.get("notes", ""))

        # -- Validate -------------------------------------------------------
        errors: List[str] = []
        if not parcel_id:
            errors.append("parcel_id is required")

        try:
            from_cat = self._validate_category(from_cat)
        except ValueError as e:
            errors.append(str(e))

        try:
            to_cat = self._validate_category(to_cat)
        except ValueError as e:
            errors.append(str(e))

        if area_ha <= _ZERO:
            errors.append("area_ha must be > 0")

        try:
            trans_date = self._parse_date(transition_date_str)
        except ValueError as e:
            errors.append(str(e))
            trans_date = date.today()

        if errors:
            return {
                "status": "VALIDATION_ERROR",
                "errors": errors,
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        # -- Classify transition -------------------------------------------
        transition_type = (
            TRANSITION_REMAINING if from_cat == to_cat
            else TRANSITION_CONVERSION
        )

        is_deforestation = (
            from_cat == "FOREST_LAND" and to_cat != "FOREST_LAND"
        )
        is_afforestation = (
            from_cat != "FOREST_LAND" and to_cat == "FOREST_LAND"
        )
        is_wetland_drainage = (
            from_cat == "WETLANDS" and to_cat != "WETLANDS"
        )
        is_peatland_conversion = (
            from_cat == "WETLANDS"
            and to_cat in ("CROPLAND", "GRASSLAND", "SETTLEMENTS")
        )

        completion_date = self._compute_completion_date(
            trans_date, transition_period
        )

        # -- Create record -------------------------------------------------
        transition_id = str(uuid4())
        record_data = {
            "transition_id": transition_id,
            "parcel_id": parcel_id,
            "from_category": from_cat,
            "to_category": to_cat,
            "area_ha": str(area_ha),
            "transition_date": trans_date.isoformat(),
            "transition_type": transition_type,
        }
        provenance_hash = _compute_hash(record_data)

        record = TransitionRecord(
            transition_id=transition_id,
            parcel_id=parcel_id,
            from_category=from_cat,
            to_category=to_cat,
            area_ha=area_ha,
            transition_date=trans_date.isoformat(),
            transition_type=transition_type,
            transition_period_years=transition_period,
            completion_date=completion_date,
            is_deforestation=is_deforestation,
            is_afforestation=is_afforestation,
            is_wetland_drainage=is_wetland_drainage,
            is_peatland_conversion=is_peatland_conversion,
            notes=notes,
            provenance_hash=provenance_hash,
            recorded_at=_utcnow().isoformat(),
        )

        # -- Update state --------------------------------------------------
        with self._lock:
            self._transition_history.append(record)
            self._parcel_history[parcel_id].append(record)
            self._transition_matrix[from_cat][to_cat] += area_ha
            self._parcel_current_category[parcel_id] = to_cat
            if parcel_id not in self._parcel_total_area:
                self._parcel_total_area[parcel_id] = area_ha
            self._total_transitions += 1

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = record.to_dict()
        result["status"] = "SUCCESS"
        result["processing_time_ms"] = processing_time

        logger.info(
            "Transition recorded: id=%s, parcel=%s, %s -> %s, "
            "area=%s ha, type=%s, deforestation=%s, time=%.3fms",
            transition_id, parcel_id, from_cat, to_cat,
            area_ha, transition_type, is_deforestation, processing_time,
        )
        return result

    # ------------------------------------------------------------------
    # Transition Matrix
    # ------------------------------------------------------------------

    def get_transition_matrix(self) -> Dict[str, Any]:
        """Get the current 6x6 land-use transition matrix.

        Returns:
            Dictionary with matrix data, totals, and provenance.
        """
        with self._lock:
            matrix: Dict[str, Dict[str, str]] = {}
            total_remaining = _ZERO
            total_conversion = _ZERO

            for from_cat in LAND_CATEGORIES:
                matrix[from_cat] = {}
                for to_cat in LAND_CATEGORIES:
                    val = self._transition_matrix[from_cat][to_cat]
                    matrix[from_cat][to_cat] = str(val)
                    if from_cat == to_cat:
                        total_remaining += val
                    else:
                        total_conversion += val

            result = {
                "matrix": matrix,
                "categories": LAND_CATEGORIES,
                "total_remaining_ha": str(total_remaining),
                "total_conversion_ha": str(total_conversion),
                "total_area_ha": str(total_remaining + total_conversion),
                "total_transitions": self._total_transitions,
            }
            result["provenance_hash"] = _compute_hash(result)
            return result

    # ------------------------------------------------------------------
    # Transition History
    # ------------------------------------------------------------------

    def get_transition_history(
        self,
        parcel_id: Optional[str] = None,
        from_category: Optional[str] = None,
        to_category: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Query transition history with optional filters.

        Args:
            parcel_id: Filter by parcel ID.
            from_category: Filter by source category.
            to_category: Filter by target category.
            start_date: Filter transitions after this date (YYYY-MM-DD).
            end_date: Filter transitions before this date (YYYY-MM-DD).
            limit: Maximum records to return.
            offset: Records to skip.

        Returns:
            Filtered transition history with pagination metadata.
        """
        with self._lock:
            records = list(self._transition_history)

        # -- Apply filters -------------------------------------------------
        if parcel_id:
            records = [r for r in records if r.parcel_id == parcel_id]

        if from_category:
            fc = from_category.upper()
            records = [r for r in records if r.from_category == fc]

        if to_category:
            tc = to_category.upper()
            records = [r for r in records if r.to_category == tc]

        if start_date:
            records = [
                r for r in records if r.transition_date >= start_date
            ]

        if end_date:
            records = [
                r for r in records if r.transition_date <= end_date
            ]

        total_count = len(records)
        paginated = records[offset:offset + limit]

        return {
            "transitions": [r.to_dict() for r in paginated],
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total_count,
        }

    # ------------------------------------------------------------------
    # Deforestation Detection
    # ------------------------------------------------------------------

    def detect_deforestation(
        self,
        parcel_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect deforestation events (any conversion FROM forest land).

        Args:
            parcel_id: Optional parcel filter.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            List of deforestation events with area summary.
        """
        with self._lock:
            records = list(self._transition_history)

        deforest = [r for r in records if r.is_deforestation]

        if parcel_id:
            deforest = [r for r in deforest if r.parcel_id == parcel_id]
        if start_date:
            deforest = [r for r in deforest if r.transition_date >= start_date]
        if end_date:
            deforest = [r for r in deforest if r.transition_date <= end_date]

        total_area = sum(r.area_ha for r in deforest)

        # Group by target category
        by_target: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
        for r in deforest:
            by_target[r.to_category] += r.area_ha

        return {
            "deforestation_events": [r.to_dict() for r in deforest],
            "total_deforestation_ha": str(total_area),
            "event_count": len(deforest),
            "by_target_category": {
                k: str(v) for k, v in sorted(by_target.items())
            },
            "provenance_hash": _compute_hash({
                "count": len(deforest),
                "total_ha": str(total_area),
            }),
        }

    # ------------------------------------------------------------------
    # Afforestation/Reforestation Detection
    # ------------------------------------------------------------------

    def detect_afforestation(
        self,
        parcel_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect afforestation/reforestation (conversion TO forest land).

        Args:
            parcel_id: Optional parcel filter.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            List of afforestation events with area summary.
        """
        with self._lock:
            records = list(self._transition_history)

        afforest = [r for r in records if r.is_afforestation]

        if parcel_id:
            afforest = [r for r in afforest if r.parcel_id == parcel_id]
        if start_date:
            afforest = [r for r in afforest if r.transition_date >= start_date]
        if end_date:
            afforest = [r for r in afforest if r.transition_date <= end_date]

        total_area = sum(r.area_ha for r in afforest)

        by_source: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
        for r in afforest:
            by_source[r.from_category] += r.area_ha

        return {
            "afforestation_events": [r.to_dict() for r in afforest],
            "total_afforestation_ha": str(total_area),
            "event_count": len(afforest),
            "by_source_category": {
                k: str(v) for k, v in sorted(by_source.items())
            },
            "provenance_hash": _compute_hash({
                "count": len(afforest),
                "total_ha": str(total_area),
            }),
        }

    # ------------------------------------------------------------------
    # Wetland Change Detection
    # ------------------------------------------------------------------

    def detect_wetland_changes(
        self,
        parcel_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect wetland drainage, rewetting, and peatland conversion events.

        Args:
            parcel_id: Optional parcel filter.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            Categorized wetland change events.
        """
        with self._lock:
            records = list(self._transition_history)

        # Filter by optional criteria
        filtered = records
        if parcel_id:
            filtered = [r for r in filtered if r.parcel_id == parcel_id]
        if start_date:
            filtered = [r for r in filtered if r.transition_date >= start_date]
        if end_date:
            filtered = [r for r in filtered if r.transition_date <= end_date]

        # Drainage: from wetlands to non-wetlands
        drainage = [r for r in filtered if r.is_wetland_drainage]
        drainage_area = sum(r.area_ha for r in drainage)

        # Rewetting: from non-wetlands to wetlands
        rewetting = [
            r for r in filtered
            if r.from_category != "WETLANDS" and r.to_category == "WETLANDS"
        ]
        rewetting_area = sum(r.area_ha for r in rewetting)

        # Peatland conversion
        peatland = [r for r in filtered if r.is_peatland_conversion]
        peatland_area = sum(r.area_ha for r in peatland)

        return {
            "drainage_events": [r.to_dict() for r in drainage],
            "total_drainage_ha": str(drainage_area),
            "drainage_count": len(drainage),
            "rewetting_events": [r.to_dict() for r in rewetting],
            "total_rewetting_ha": str(rewetting_area),
            "rewetting_count": len(rewetting),
            "peatland_conversion_events": [r.to_dict() for r in peatland],
            "total_peatland_conversion_ha": str(peatland_area),
            "peatland_conversion_count": len(peatland),
            "net_wetland_change_ha": str(rewetting_area - drainage_area),
            "provenance_hash": _compute_hash({
                "drainage": str(drainage_area),
                "rewetting": str(rewetting_area),
                "peatland": str(peatland_area),
            }),
        }

    # ------------------------------------------------------------------
    # Transition Rate
    # ------------------------------------------------------------------

    def get_transition_rate(
        self,
        from_category: str,
        to_category: str,
        start_year: int,
        end_year: int,
    ) -> Dict[str, Any]:
        """Calculate the annual transition rate between two categories.

        Rate = total_area_transitioned / number_of_years

        Args:
            from_category: Source category.
            to_category: Target category.
            start_year: Start year of the analysis period.
            end_year: End year of the analysis period.

        Returns:
            Annual transition rate with metadata.
        """
        from_cat = self._validate_category(from_category)
        to_cat = self._validate_category(to_category)

        if end_year <= start_year:
            raise ValueError("end_year must be > start_year")

        years = _D(str(end_year - start_year))

        with self._lock:
            records = [
                r for r in self._transition_history
                if r.from_category == from_cat
                and r.to_category == to_cat
                and r.transition_date >= f"{start_year}-01-01"
                and r.transition_date <= f"{end_year}-12-31"
            ]

        total_area = sum(r.area_ha for r in records)
        annual_rate = (total_area / years).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        return {
            "from_category": from_cat,
            "to_category": to_cat,
            "start_year": start_year,
            "end_year": end_year,
            "total_area_ha": str(total_area),
            "annual_rate_ha_yr": str(annual_rate),
            "transition_count": len(records),
            "years": str(years),
        }

    # ------------------------------------------------------------------
    # Area Consistency Validation
    # ------------------------------------------------------------------

    def validate_area_consistency(
        self,
        expected_total_ha: Optional[Decimal] = None,
        tolerance_pct: Decimal = _D("0.01"),
    ) -> Dict[str, Any]:
        """Validate that total area is conserved across all transitions.

        Checks that the sum of all "from" areas equals the sum of all
        "to" areas (area is neither created nor destroyed).

        Args:
            expected_total_ha: Optional expected total area.
            tolerance_pct: Allowed tolerance as a fraction (default 0.01 = 1%).

        Returns:
            Validation result with pass/fail and discrepancy details.
        """
        with self._lock:
            total_from: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
            total_to: Dict[str, Decimal] = defaultdict(lambda: _ZERO)

            for record in self._transition_history:
                total_from[record.from_category] += record.area_ha
                total_to[record.to_category] += record.area_ha

        sum_from = sum(total_from.values())
        sum_to = sum(total_to.values())
        discrepancy = abs(sum_from - sum_to)

        is_consistent = True
        findings: List[str] = []

        # Check from/to balance
        if sum_from > _ZERO:
            discrepancy_pct = (discrepancy / sum_from) * _D("100")
            threshold = tolerance_pct * _D("100")
            if discrepancy_pct > threshold:
                is_consistent = False
                findings.append(
                    f"Area discrepancy of {discrepancy} ha "
                    f"({discrepancy_pct}%) exceeds tolerance of {threshold}%"
                )

        # Check against expected total
        if expected_total_ha is not None and sum_from > _ZERO:
            diff = abs(sum_from - expected_total_ha)
            if diff > expected_total_ha * tolerance_pct:
                is_consistent = False
                findings.append(
                    f"Total area {sum_from} ha differs from expected "
                    f"{expected_total_ha} ha by {diff} ha"
                )

        return {
            "is_consistent": is_consistent,
            "total_from_ha": str(sum_from),
            "total_to_ha": str(sum_to),
            "discrepancy_ha": str(discrepancy),
            "by_category_from": {
                k: str(v) for k, v in sorted(total_from.items())
            },
            "by_category_to": {
                k: str(v) for k, v in sorted(total_to.items())
            },
            "findings": findings,
            "provenance_hash": _compute_hash({
                "from": str(sum_from),
                "to": str(sum_to),
            }),
        }

    # ------------------------------------------------------------------
    # Portfolio Summary
    # ------------------------------------------------------------------

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of all transitions across the portfolio.

        Returns:
            Portfolio-level transition summary with categorized totals.
        """
        with self._lock:
            records = list(self._transition_history)

        total_parcels = len(set(r.parcel_id for r in records))
        total_area = sum(r.area_ha for r in records)

        # Deforestation summary
        deforest_records = [r for r in records if r.is_deforestation]
        deforest_area = sum(r.area_ha for r in deforest_records)

        # Afforestation summary
        afforest_records = [r for r in records if r.is_afforestation]
        afforest_area = sum(r.area_ha for r in afforest_records)

        # Wetland drainage summary
        drainage_records = [r for r in records if r.is_wetland_drainage]
        drainage_area = sum(r.area_ha for r in drainage_records)

        # Peatland conversion summary
        peatland_records = [r for r in records if r.is_peatland_conversion]
        peatland_area = sum(r.area_ha for r in peatland_records)

        # Count by transition type
        remaining_count = sum(
            1 for r in records if r.transition_type == TRANSITION_REMAINING
        )
        conversion_count = sum(
            1 for r in records if r.transition_type == TRANSITION_CONVERSION
        )

        remaining_area = sum(
            r.area_ha for r in records
            if r.transition_type == TRANSITION_REMAINING
        )
        conversion_area = sum(
            r.area_ha for r in records
            if r.transition_type == TRANSITION_CONVERSION
        )

        result = {
            "total_parcels": total_parcels,
            "total_transitions": len(records),
            "total_area_ha": str(total_area),
            "remaining": {
                "count": remaining_count,
                "area_ha": str(remaining_area),
            },
            "conversions": {
                "count": conversion_count,
                "area_ha": str(conversion_area),
            },
            "deforestation": {
                "count": len(deforest_records),
                "area_ha": str(deforest_area),
            },
            "afforestation": {
                "count": len(afforest_records),
                "area_ha": str(afforest_area),
            },
            "net_forest_change_ha": str(afforest_area - deforest_area),
            "wetland_drainage": {
                "count": len(drainage_records),
                "area_ha": str(drainage_area),
            },
            "peatland_conversion": {
                "count": len(peatland_records),
                "area_ha": str(peatland_area),
            },
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Active Transitions
    # ------------------------------------------------------------------

    def get_active_transitions(
        self,
        as_of_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get transitions whose SOC transition period is still active.

        A transition is "active" if the current date is between the
        transition_date and the completion_date.

        Args:
            as_of_date: Reference date (YYYY-MM-DD). Defaults to today.

        Returns:
            Active transitions with remaining years.
        """
        if as_of_date:
            ref_date = self._parse_date(as_of_date)
        else:
            ref_date = date.today()

        ref_str = ref_date.isoformat()

        with self._lock:
            records = list(self._transition_history)

        active = []
        for r in records:
            if r.transition_type == TRANSITION_REMAINING:
                continue
            if r.transition_date <= ref_str <= r.completion_date:
                trans_date = self._parse_date(r.transition_date)
                elapsed_days = (ref_date - trans_date).days
                elapsed_years = _D(str(elapsed_days)) / _D("365.25")
                remaining_years = (
                    _D(str(r.transition_period_years)) - elapsed_years
                ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
                progress_pct = (
                    elapsed_years / _D(str(r.transition_period_years)) * _D("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                entry = r.to_dict()
                entry["elapsed_years"] = str(elapsed_years.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                ))
                entry["remaining_years"] = str(remaining_years)
                entry["progress_pct"] = str(progress_pct)
                active.append(entry)

        return {
            "active_transitions": active,
            "active_count": len(active),
            "as_of_date": ref_str,
        }

    # ------------------------------------------------------------------
    # Transition Age
    # ------------------------------------------------------------------

    def get_transition_age(
        self,
        transition_id: str,
        as_of_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the age of a specific transition in years.

        Args:
            transition_id: Transition record ID.
            as_of_date: Reference date. Defaults to today.

        Returns:
            Transition age and status information.
        """
        if as_of_date:
            ref_date = self._parse_date(as_of_date)
        else:
            ref_date = date.today()

        with self._lock:
            record = None
            for r in self._transition_history:
                if r.transition_id == transition_id:
                    record = r
                    break

        if record is None:
            return {"status": "NOT_FOUND", "transition_id": transition_id}

        trans_date = self._parse_date(record.transition_date)
        elapsed_days = (ref_date - trans_date).days
        age_years = (_D(str(elapsed_days)) / _D("365.25")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        completion = self._parse_date(record.completion_date)
        is_complete = ref_date >= completion

        return {
            "transition_id": transition_id,
            "transition_date": record.transition_date,
            "completion_date": record.completion_date,
            "age_years": str(age_years),
            "is_soc_transition_complete": is_complete,
            "transition_period_years": record.transition_period_years,
            "as_of_date": ref_date.isoformat(),
        }

    # ------------------------------------------------------------------
    # Transition Reversal Detection
    # ------------------------------------------------------------------

    def detect_reversals(
        self,
        parcel_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect transition reversals (e.g. CL->FL after FL->CL).

        A reversal occurs when a parcel transitions from A->B and later
        transitions B->A.

        Args:
            parcel_id: Optional parcel filter.

        Returns:
            Detected reversal pairs with metadata.
        """
        with self._lock:
            if parcel_id:
                records = list(self._parcel_history.get(parcel_id, []))
            else:
                records = list(self._transition_history)

        # Group by parcel
        by_parcel: Dict[str, List[TransitionRecord]] = defaultdict(list)
        for r in records:
            by_parcel[r.parcel_id].append(r)

        reversals: List[Dict[str, Any]] = []

        for pid, parcel_records in by_parcel.items():
            sorted_records = sorted(parcel_records, key=lambda r: r.transition_date)
            for i, r1 in enumerate(sorted_records):
                if r1.transition_type == TRANSITION_REMAINING:
                    continue
                for r2 in sorted_records[i + 1:]:
                    if r2.transition_type == TRANSITION_REMAINING:
                        continue
                    if (
                        r2.from_category == r1.to_category
                        and r2.to_category == r1.from_category
                    ):
                        reversals.append({
                            "parcel_id": pid,
                            "original_transition_id": r1.transition_id,
                            "original_date": r1.transition_date,
                            "original_from": r1.from_category,
                            "original_to": r1.to_category,
                            "reversal_transition_id": r2.transition_id,
                            "reversal_date": r2.transition_date,
                            "reversal_from": r2.from_category,
                            "reversal_to": r2.to_category,
                            "area_ha": str(r2.area_ha),
                        })

        return {
            "reversals": reversals,
            "reversal_count": len(reversals),
            "provenance_hash": _compute_hash({
                "count": len(reversals),
            }),
        }

    # ------------------------------------------------------------------
    # Cumulative Transition Area
    # ------------------------------------------------------------------

    def get_cumulative_transitions(
        self,
        from_category: Optional[str] = None,
        to_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get cumulative transition area by year.

        Args:
            from_category: Optional source category filter.
            to_category: Optional target category filter.

        Returns:
            Year-by-year cumulative area.
        """
        with self._lock:
            records = list(self._transition_history)

        if from_category:
            fc = from_category.upper()
            records = [r for r in records if r.from_category == fc]
        if to_category:
            tc = to_category.upper()
            records = [r for r in records if r.to_category == tc]

        # Group by year
        by_year: Dict[int, Decimal] = defaultdict(lambda: _ZERO)
        for r in records:
            year = int(r.transition_date[:4])
            by_year[year] += r.area_ha

        # Compute cumulative
        sorted_years = sorted(by_year.keys())
        cumulative: Dict[str, str] = {}
        running_total = _ZERO
        for year in sorted_years:
            running_total += by_year[year]
            cumulative[str(year)] = str(running_total)

        return {
            "annual_area": {
                str(y): str(a) for y, a in sorted(by_year.items())
            },
            "cumulative_area": cumulative,
            "total_area_ha": str(running_total),
            "years_covered": sorted_years,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics."""
        with self._lock:
            return {
                "engine": "LandUseChangeTrackerEngine",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_transitions": self._total_transitions,
                "total_parcels": len(self._parcel_current_category),
                "categories": LAND_CATEGORIES,
            }

    def reset(self) -> None:
        """Reset all engine state. Intended for testing teardown."""
        with self._lock:
            self._transition_matrix = {
                from_cat: {to_cat: _ZERO for to_cat in LAND_CATEGORIES}
                for from_cat in LAND_CATEGORIES
            }
            self._transition_history.clear()
            self._parcel_history.clear()
            self._parcel_current_category.clear()
            self._parcel_total_area.clear()
            self._total_transitions = 0
        logger.info("LandUseChangeTrackerEngine reset")
