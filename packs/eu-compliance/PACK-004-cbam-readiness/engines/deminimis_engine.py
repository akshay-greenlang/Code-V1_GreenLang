# -*- coding: utf-8 -*-
"""
DeMinimisEngine - PACK-004 CBAM Readiness Engine 5
====================================================

De minimis threshold tracking engine for CBAM compliance. Tracks cumulative
import volumes per sector group against the 50-tonne exemption threshold,
provides alert levels, and runs annual exemption assessments.

De Minimis Rule (CBAM Regulation Article 2(4)):
    Goods are exempt from CBAM obligations if the total quantity imported
    per sector group is below 50 tonnes per calendar year. This threshold
    is applied per CBAM goods category (not per CN code or per supplier).

Threshold Tracking:
    - Cumulative tracking by sector group and calendar year
    - Real-time utilization percentage calculation
    - Multi-level alert system (SAFE, APPROACHING, WARNING, CRITICAL, EXCEEDED)
    - Annual projection based on historical import patterns

Alert Levels:
    - SAFE       : <80% utilization
    - APPROACHING: 80-90% utilization
    - WARNING    : 90-95% utilization
    - CRITICAL   : 95-100% utilization
    - EXCEEDED   : >100% utilization (CBAM obligations apply)

Zero-Hallucination:
    - All threshold calculations use deterministic arithmetic
    - No LLM involvement in any exemption determination
    - SHA-256 provenance hashing on every assessment

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
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


def _round_val(value: Decimal, places: int = 4) -> float:
    """Round a Decimal to specified places and return float."""
    rounded = value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return float(rounded)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# De minimis threshold per sector group per calendar year (tonnes)
DE_MINIMIS_THRESHOLD_TONNES: float = 50.0

# Alert level thresholds (percentage of threshold utilization)
ALERT_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "SAFE": (0.0, 80.0),
    "APPROACHING": (80.0, 90.0),
    "WARNING": (90.0, 95.0),
    "CRITICAL": (95.0, 100.0),
    "EXCEEDED": (100.0, float("inf")),
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SectorGroup(str, Enum):
    """CBAM sector groups for de minimis threshold tracking.

    Each sector group corresponds to a CBAM Annex I goods category.
    The de minimis threshold is applied independently per sector group.
    """

    CEMENT = "cement"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    FERTILIZERS = "fertilizers"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"


class AlertLevel(str, Enum):
    """Alert level for threshold utilization."""

    SAFE = "SAFE"
    APPROACHING = "APPROACHING"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EXCEEDED = "EXCEEDED"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ThresholdStatus(BaseModel):
    """Status of a single sector group against the de minimis threshold.

    Provides real-time tracking of cumulative imports against the 50-tonne
    threshold with alert level and exemption status.
    """

    sector_group: SectorGroup = Field(
        ..., description="CBAM sector group",
    )
    year: int = Field(
        ..., ge=2023, le=2050,
        description="Calendar year being tracked",
    )
    cumulative_tonnes: float = Field(
        ..., ge=0,
        description="Cumulative import quantity (tonnes) year-to-date",
    )
    threshold_tonnes: float = Field(
        DE_MINIMIS_THRESHOLD_TONNES,
        description="De minimis threshold (tonnes)",
    )
    utilization_pct: float = Field(
        ..., ge=0,
        description="Threshold utilization percentage",
    )
    remaining_tonnes: float = Field(
        ..., description="Tonnes remaining before threshold (can be negative)",
    )
    alert_level: AlertLevel = Field(
        ..., description="Current alert level",
    )
    exempt: bool = Field(
        ..., description="Whether the sector is currently exempt from CBAM",
    )
    last_import_date: Optional[datetime] = Field(
        None, description="Date of the most recent import tracked",
    )
    import_count: int = Field(
        0, ge=0, description="Number of imports tracked",
    )


class DeMinimisAssessment(BaseModel):
    """Annual de minimis assessment across all sector groups.

    Provides a complete view of the importer's de minimis status for
    a given year, including per-sector thresholds and overall exemption.
    """

    assessment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique assessment identifier",
    )
    reporting_year: int = Field(
        ..., ge=2023, le=2050,
        description="Calendar year assessed",
    )
    sector_statuses: List[ThresholdStatus] = Field(
        default_factory=list,
        description="Per-sector threshold status",
    )
    sectors_exempt: int = Field(
        0, ge=0,
        description="Number of sectors currently exempt",
    )
    sectors_non_exempt: int = Field(
        0, ge=0,
        description="Number of sectors that have exceeded the threshold",
    )
    overall_exempt: bool = Field(
        ..., description="True if ALL sector groups are below threshold",
    )
    assessment_date: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of assessment",
    )
    provenance_hash: str = Field(
        "", description="SHA-256 hash for audit trail",
    )


class ImportRecord(BaseModel):
    """Internal record of a tracked import for de minimis accounting."""

    record_id: str = Field(
        default_factory=_new_uuid,
        description="Record identifier",
    )
    sector_group: SectorGroup = Field(
        ..., description="CBAM sector group",
    )
    quantity_tonnes: float = Field(
        ..., gt=0,
        description="Import quantity in tonnes",
    )
    year: int = Field(
        ..., ge=2023, le=2050,
        description="Calendar year of import",
    )
    cn_code: str = Field(
        "", description="CN code of imported goods",
    )
    country_of_origin: str = Field(
        "", max_length=2,
        description="Country of origin",
    )
    recorded_at: datetime = Field(
        default_factory=_utcnow,
        description="When the import was recorded",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DeMinimisEngine:
    """De minimis threshold tracking engine for CBAM compliance.

    Tracks cumulative import volumes per sector group against the 50-tonne
    threshold, provides alert levels, runs annual assessments, and projects
    annual volumes from historical data.

    Zero-Hallucination Guarantees:
        - All threshold calculations use deterministic arithmetic
        - Alert levels determined by fixed percentage ranges
        - No LLM involvement in any exemption determination
        - SHA-256 provenance hashing on every assessment

    Example:
        >>> engine = DeMinimisEngine()
        >>> status = engine.track_import(SectorGroup.CEMENT, 25.0, year=2027)
        >>> assert status.exempt is True
        >>> assert status.alert_level == AlertLevel.SAFE
    """

    def __init__(self) -> None:
        """Initialize DeMinimisEngine."""
        # Cumulative tracking: {year: {sector: Decimal}}
        self._cumulative: Dict[int, Dict[SectorGroup, Decimal]] = defaultdict(
            lambda: defaultdict(Decimal)
        )
        # Import records for audit trail
        self._records: Dict[int, Dict[SectorGroup, List[ImportRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Import counts
        self._counts: Dict[int, Dict[SectorGroup, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Last import dates
        self._last_import: Dict[int, Dict[SectorGroup, datetime]] = defaultdict(dict)

        logger.info("DeMinimisEngine initialized (v%s)", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track_import(
        self,
        sector_group: SectorGroup,
        quantity_tonnes: float,
        year: Optional[int] = None,
        cn_code: str = "",
        country_of_origin: str = "",
    ) -> ThresholdStatus:
        """Track an import and return updated threshold status.

        Adds the import quantity to the cumulative total for the sector
        group and year, then returns the current threshold status including
        alert level and exemption determination.

        Args:
            sector_group: CBAM sector group.
            quantity_tonnes: Import quantity in tonnes (must be > 0).
            year: Calendar year. Defaults to current year.
            cn_code: Optional CN code for record keeping.
            country_of_origin: Optional country code.

        Returns:
            Updated ThresholdStatus for the sector group.

        Raises:
            ValueError: If quantity_tonnes <= 0.
        """
        if quantity_tonnes <= 0:
            raise ValueError(f"Import quantity must be > 0, got {quantity_tonnes}")

        target_year = year or _utcnow().year
        qty = _decimal(quantity_tonnes)

        # Update cumulative total
        self._cumulative[target_year][sector_group] += qty
        self._counts[target_year][sector_group] += 1
        self._last_import[target_year][sector_group] = _utcnow()

        # Store record
        record = ImportRecord(
            sector_group=sector_group,
            quantity_tonnes=quantity_tonnes,
            year=target_year,
            cn_code=cn_code,
            country_of_origin=country_of_origin,
        )
        self._records[target_year][sector_group].append(record)

        # Build status
        status = self._build_status(sector_group, target_year)

        logger.info(
            "Import tracked [%s/%d]: +%.2f t, cumulative=%.2f t, alert=%s, exempt=%s",
            sector_group.value,
            target_year,
            quantity_tonnes,
            status.cumulative_tonnes,
            status.alert_level.value,
            status.exempt,
        )

        return status

    def get_cumulative(
        self,
        sector_group: SectorGroup,
        year: Optional[int] = None,
    ) -> float:
        """Get the cumulative import quantity for a sector group and year.

        Args:
            sector_group: CBAM sector group.
            year: Calendar year. Defaults to current year.

        Returns:
            Cumulative tonnes imported year-to-date.
        """
        target_year = year or _utcnow().year
        return _round_val(self._cumulative[target_year][sector_group], 4)

    def assess_threshold(
        self,
        sector_group: SectorGroup,
        year: Optional[int] = None,
    ) -> ThresholdStatus:
        """Assess the current threshold status for a sector group.

        Args:
            sector_group: CBAM sector group.
            year: Calendar year. Defaults to current year.

        Returns:
            Current ThresholdStatus.
        """
        target_year = year or _utcnow().year
        return self._build_status(sector_group, target_year)

    def run_annual_assessment(
        self,
        year: Optional[int] = None,
    ) -> DeMinimisAssessment:
        """Run a full de minimis assessment across all sector groups.

        Evaluates every CBAM sector group against the threshold for the
        specified year and produces a comprehensive assessment.

        Args:
            year: Calendar year to assess. Defaults to current year.

        Returns:
            DeMinimisAssessment with per-sector statuses.
        """
        target_year = year or _utcnow().year

        statuses: List[ThresholdStatus] = []
        for sector in SectorGroup:
            status = self._build_status(sector, target_year)
            statuses.append(status)

        exempt_count = sum(1 for s in statuses if s.exempt)
        non_exempt_count = len(statuses) - exempt_count
        overall_exempt = exempt_count == len(statuses)

        assessment = DeMinimisAssessment(
            reporting_year=target_year,
            sector_statuses=statuses,
            sectors_exempt=exempt_count,
            sectors_non_exempt=non_exempt_count,
            overall_exempt=overall_exempt,
        )
        assessment.provenance_hash = _compute_hash(assessment)

        logger.info(
            "Annual assessment [%d]: %d/%d sectors exempt, overall=%s",
            target_year,
            exempt_count,
            len(statuses),
            "EXEMPT" if overall_exempt else "NOT EXEMPT",
        )

        return assessment

    def project_annual_volume(
        self,
        sector_group: SectorGroup,
        historical_data: Optional[Dict[int, float]] = None,
    ) -> float:
        """Project the annual import volume for a sector group.

        Uses historical data or tracked imports to project the full-year
        volume based on year-to-date data. If the current year is partially
        elapsed, the projection extrapolates linearly.

        Args:
            sector_group: CBAM sector group.
            historical_data: Optional dict of {year: annual_tonnes}.
                If None, uses internally tracked data.

        Returns:
            Projected annual volume in tonnes.
        """
        now = _utcnow()
        current_year = now.year

        if historical_data:
            # Use average of historical data
            if not historical_data:
                return 0.0
            avg = sum(historical_data.values()) / len(historical_data)
            return round(avg, 4)

        # Use current year-to-date with linear extrapolation
        ytd = self._cumulative[current_year][sector_group]
        if ytd <= 0:
            return 0.0

        # Fraction of year elapsed
        day_of_year = now.timetuple().tm_yday
        fraction = Decimal(str(day_of_year)) / Decimal("365")

        if fraction > 0:
            projected = ytd / fraction
            return _round_val(projected, 2)

        return _round_val(ytd, 2)

    def check_exemption(
        self,
        sector_group: SectorGroup,
        year: Optional[int] = None,
    ) -> bool:
        """Check whether a sector group is exempt from CBAM.

        A sector is exempt if the cumulative import volume for the year
        is below the de minimis threshold (50 tonnes).

        Args:
            sector_group: CBAM sector group.
            year: Calendar year. Defaults to current year.

        Returns:
            True if exempt (below threshold), False if obligations apply.
        """
        target_year = year or _utcnow().year
        cumulative = self._cumulative[target_year][sector_group]
        threshold = _decimal(DE_MINIMIS_THRESHOLD_TONNES)
        return cumulative < threshold

    def get_alert_level(
        self,
        utilization_pct: float,
    ) -> str:
        """Determine the alert level for a given utilization percentage.

        Alert levels:
            - SAFE       : < 80%
            - APPROACHING: 80% - 90%
            - WARNING    : 90% - 95%
            - CRITICAL   : 95% - 100%
            - EXCEEDED   : > 100%

        Args:
            utilization_pct: Threshold utilization as a percentage.

        Returns:
            Alert level string.
        """
        pct = float(utilization_pct)
        for level, (low, high) in ALERT_THRESHOLDS.items():
            if low <= pct < high:
                return level
        return "EXCEEDED"

    def get_import_records(
        self,
        sector_group: SectorGroup,
        year: Optional[int] = None,
    ) -> List[ImportRecord]:
        """Retrieve all import records for a sector group and year.

        Args:
            sector_group: CBAM sector group.
            year: Calendar year. Defaults to current year.

        Returns:
            List of ImportRecord sorted by date.
        """
        target_year = year or _utcnow().year
        records = self._records[target_year][sector_group]
        return sorted(records, key=lambda r: r.recorded_at)

    def reset_year(
        self,
        year: int,
        sector_group: Optional[SectorGroup] = None,
    ) -> None:
        """Reset tracking data for a year (optionally per sector).

        Used for corrections or re-initialization. Clears cumulative totals,
        import counts, and records.

        Args:
            year: Calendar year to reset.
            sector_group: Optional sector. If None, resets all sectors for the year.
        """
        if sector_group:
            self._cumulative[year][sector_group] = Decimal("0")
            self._counts[year][sector_group] = 0
            self._records[year][sector_group] = []
            self._last_import[year].pop(sector_group, None)
            logger.info("Reset tracking for %s/%d", sector_group.value, year)
        else:
            self._cumulative[year] = defaultdict(Decimal)
            self._counts[year] = defaultdict(int)
            self._records[year] = defaultdict(list)
            self._last_import[year] = {}
            logger.info("Reset all tracking for year %d", year)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def tracked_years(self) -> List[int]:
        """List of years with tracked data."""
        return sorted(self._cumulative.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_status(
        self,
        sector_group: SectorGroup,
        year: int,
    ) -> ThresholdStatus:
        """Build a ThresholdStatus for a sector group and year."""
        cumulative = self._cumulative[year][sector_group]
        threshold = _decimal(DE_MINIMIS_THRESHOLD_TONNES)
        utilization = (
            (cumulative / threshold * Decimal("100"))
            if threshold > 0
            else Decimal("0")
        )
        remaining = threshold - cumulative
        exempt = cumulative < threshold

        alert_str = self.get_alert_level(float(utilization))
        alert = AlertLevel(alert_str)

        last_import = self._last_import.get(year, {}).get(sector_group)
        count = self._counts[year][sector_group]

        return ThresholdStatus(
            sector_group=sector_group,
            year=year,
            cumulative_tonnes=_round_val(cumulative, 4),
            threshold_tonnes=DE_MINIMIS_THRESHOLD_TONNES,
            utilization_pct=_round_val(utilization, 2),
            remaining_tonnes=_round_val(remaining, 4),
            alert_level=alert,
            exempt=exempt,
            last_import_date=last_import,
            import_count=count,
        )
