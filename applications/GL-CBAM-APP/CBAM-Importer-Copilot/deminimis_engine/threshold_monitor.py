# -*- coding: utf-8 -*-
"""
ThresholdMonitorEngine - 50-tonne annual de minimis threshold monitor.

Tracks cumulative CBAM imports per importer per year against the 50-tonne
threshold established by the Omnibus Simplification Package (Oct 2025).
Thread-safe singleton implementation ensures consistent state across
concurrent API requests.

Electricity (CN 2716) and hydrogen (CN 2804) imports are excluded from
the threshold calculation per Article 2(3a) of the amended CBAM Regulation.

Example:
    >>> engine = ThresholdMonitorEngine.get_instance()
    >>> status = engine.add_import("IMP-001", 2026, "72011000", Decimal("15.0"))
    >>> print(f"{status.percentage}% of threshold used")
    30.0% of threshold used

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import hashlib
import logging
import threading
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DE_MINIMIS_THRESHOLD_MT = Decimal("50")
ALERT_THRESHOLDS_PCT: Tuple[int, ...] = (80, 90, 95, 100)

# CN-code prefixes that are excluded from threshold calculation
_EXCLUDED_CN_PREFIXES: FrozenSet[str] = frozenset({"2716", "2804"})

# Mapping of CN-code prefixes to CBAM sector groups
_CN_SECTOR_MAP: Dict[str, str] = {
    "2507": "cement",
    "2523": "cement",
    "7201": "iron_steel",
    "7202": "iron_steel",
    "7203": "iron_steel",
    "7204": "iron_steel",
    "7205": "iron_steel",
    "7206": "iron_steel",
    "7207": "iron_steel",
    "7208": "iron_steel",
    "7209": "iron_steel",
    "7210": "iron_steel",
    "7211": "iron_steel",
    "7212": "iron_steel",
    "7213": "iron_steel",
    "7214": "iron_steel",
    "7215": "iron_steel",
    "7216": "iron_steel",
    "7217": "iron_steel",
    "7218": "iron_steel",
    "7219": "iron_steel",
    "7220": "iron_steel",
    "7221": "iron_steel",
    "7222": "iron_steel",
    "7223": "iron_steel",
    "7224": "iron_steel",
    "7225": "iron_steel",
    "7226": "iron_steel",
    "7227": "iron_steel",
    "7228": "iron_steel",
    "7229": "iron_steel",
    "7601": "aluminium",
    "7602": "aluminium",
    "7603": "aluminium",
    "7604": "aluminium",
    "7605": "aluminium",
    "7606": "aluminium",
    "7607": "aluminium",
    "7608": "aluminium",
    "7609": "aluminium",
    "7610": "aluminium",
    "7611": "aluminium",
    "7612": "aluminium",
    "7613": "aluminium",
    "7614": "aluminium",
    "7615": "aluminium",
    "7616": "aluminium",
    "2808": "fertilisers",
    "2814": "fertilisers",
    "3102": "fertilisers",
    "3103": "fertilisers",
    "3104": "fertilisers",
    "3105": "fertilisers",
    "2716": "electricity",
    "2804": "hydrogen",
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ImportRecord(BaseModel):
    """Individual import event recorded against the threshold."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    importer_id: str = Field(..., description="EORI or internal importer identifier")
    year: int = Field(..., ge=2026, description="CBAM reporting year")
    cn_code: str = Field(..., min_length=4, max_length=10, description="Combined Nomenclature code")
    quantity_mt: Decimal = Field(..., ge=Decimal("0"), description="Import quantity in metric tonnes")
    sector: str = Field(default="unknown", description="Resolved CBAM sector")
    eligible_for_threshold: bool = Field(default=True, description="Counts toward 50t threshold")
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(default="", description="SHA-256 of record data")

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("cn_code")
    @classmethod
    def normalize_cn_code(cls, v: str) -> str:
        """Strip spaces and non-digit characters from CN code."""
        return "".join(c for c in v if c.isdigit())


class SectorBreakdown(BaseModel):
    """Cumulative imports broken down by CBAM sector."""

    sector: str = Field(..., description="CBAM sector name")
    cumulative_mt: Decimal = Field(default=Decimal("0"), description="Metric tonnes imported")
    record_count: int = Field(default=0, description="Number of import records")
    percentage_of_total: Decimal = Field(default=Decimal("0"), description="Sector share of total imports")


class ThresholdAlert(BaseModel):
    """Alert generated when an importer approaches or exceeds the threshold."""

    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    importer_id: str
    year: int
    alert_level: int = Field(..., description="Percentage trigger (80/90/95/100)")
    current_percentage: Decimal
    cumulative_mt: Decimal
    threshold_mt: Decimal = DE_MINIMIS_THRESHOLD_MT
    message: str = ""
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}


class ThresholdStatus(BaseModel):
    """Complete threshold status for an importer-year pair."""

    importer_id: str
    year: int
    cumulative_mt: Decimal = Field(default=Decimal("0"))
    threshold_mt: Decimal = Field(default=DE_MINIMIS_THRESHOLD_MT)
    percentage: Decimal = Field(default=Decimal("0"))
    exempt: bool = Field(default=True)
    alert_level: Optional[int] = Field(default=None, description="Highest alert triggered")
    sector_breakdown: List[SectorBreakdown] = Field(default_factory=list)
    projected_breach_date: Optional[date] = Field(default=None)
    import_velocity_mt_per_month: Decimal = Field(default=Decimal("0"))
    total_records: int = Field(default=0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Engine Implementation
# ---------------------------------------------------------------------------

class ThresholdMonitorEngine:
    """
    Thread-safe singleton that monitors cumulative CBAM imports against the
    50-metric-tonne annual de minimis threshold.

    The engine stores import records in memory (production deployments should
    back this with a persistent store via the repository adapter pattern).

    Thread safety is guaranteed by an ``RLock`` that serialises all mutations
    to the internal state dictionaries.

    Attributes:
        _records: Per-importer-year list of ImportRecord objects.
        _alerts_fired: Set of (importer_id, year, level) tuples already triggered.
    """

    _instance: Optional["ThresholdMonitorEngine"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialise the engine (called once via get_instance)."""
        self._lock: threading.RLock = threading.RLock()
        self._records: Dict[str, Dict[int, List[ImportRecord]]] = {}
        self._alerts_fired: set = set()
        self._alert_listeners: List[Any] = []
        logger.info("ThresholdMonitorEngine initialised (threshold=%s MT)", DE_MINIMIS_THRESHOLD_MT)

    @classmethod
    def get_instance(cls) -> "ThresholdMonitorEngine":
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing only)."""
        with cls._singleton_lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_threshold(self, importer_id: str, year: int) -> ThresholdStatus:
        """
        Check the current cumulative imports versus the 50-tonne threshold.

        Args:
            importer_id: EORI or internal identifier.
            year: Calendar year (>= 2026).

        Returns:
            ThresholdStatus with current cumulative total and exemption flag.
        """
        with self._lock:
            return self._build_status(importer_id, year)

    def add_import(
        self,
        importer_id: str,
        year: int,
        cn_code: str,
        quantity_mt: Decimal,
    ) -> ThresholdStatus:
        """
        Record an import and update the running total.

        Args:
            importer_id: EORI or internal identifier.
            year: Calendar year (>= 2026).
            cn_code: 4-10 digit Combined Nomenclature code.
            quantity_mt: Quantity in metric tonnes (>= 0).

        Returns:
            Updated ThresholdStatus after recording the import.

        Raises:
            ValueError: If quantity_mt is negative or year < 2026.
        """
        if quantity_mt < 0:
            raise ValueError(f"quantity_mt must be >= 0, got {quantity_mt}")
        if year < 2026:
            raise ValueError(f"year must be >= 2026, got {year}")

        normalized_cn = "".join(c for c in cn_code if c.isdigit())
        eligible = self._is_threshold_eligible(normalized_cn)
        sector = self._resolve_sector(normalized_cn)

        record = ImportRecord(
            importer_id=importer_id,
            year=year,
            cn_code=normalized_cn,
            quantity_mt=quantity_mt,
            sector=sector,
            eligible_for_threshold=eligible,
        )
        record.provenance_hash = self._hash_record(record)

        with self._lock:
            self._store_record(record)
            status = self._build_status(importer_id, year)
            self._evaluate_alerts(importer_id, year, status)
            logger.info(
                "Recorded import: importer=%s year=%d cn=%s qty=%s MT eligible=%s cumulative=%s MT (%s%%)",
                importer_id, year, normalized_cn, quantity_mt,
                eligible, status.cumulative_mt, status.percentage,
            )
            return status

    def get_cumulative_imports(self, importer_id: str, year: int) -> Dict[str, Any]:
        """
        Return cumulative imports broken down by sector.

        Args:
            importer_id: EORI or internal identifier.
            year: Calendar year.

        Returns:
            Dictionary with total_mt, eligible_mt, excluded_mt, and per-sector detail.
        """
        with self._lock:
            records = self._get_records(importer_id, year)
            total_mt = sum(r.quantity_mt for r in records)
            eligible_mt = sum(r.quantity_mt for r in records if r.eligible_for_threshold)
            excluded_mt = total_mt - eligible_mt
            sectors: Dict[str, Decimal] = {}
            for r in records:
                sectors[r.sector] = sectors.get(r.sector, Decimal("0")) + r.quantity_mt
            return {
                "importer_id": importer_id,
                "year": year,
                "total_mt": total_mt,
                "eligible_mt": eligible_mt,
                "excluded_mt": excluded_mt,
                "sectors": {k: float(v) for k, v in sectors.items()},
                "record_count": len(records),
            }

    def get_threshold_percentage(self, importer_id: str, year: int) -> Decimal:
        """
        Return the percentage of the 50-tonne threshold consumed.

        Args:
            importer_id: EORI or internal identifier.
            year: Calendar year.

        Returns:
            Decimal percentage (0-100+). Values above 100 indicate threshold breach.
        """
        with self._lock:
            eligible_mt = self._get_eligible_total(importer_id, year)
            return self._calc_percentage(eligible_mt)

    def forecast_threshold_breach(
        self, importer_id: str, year: int
    ) -> Optional[date]:
        """
        Predict when the importer will breach the 50-tonne threshold based
        on current import velocity (linear extrapolation).

        Args:
            importer_id: EORI or internal identifier.
            year: Calendar year.

        Returns:
            Projected breach date, or None if velocity is zero or already breached.
        """
        with self._lock:
            eligible_mt = self._get_eligible_total(importer_id, year)
            if eligible_mt >= DE_MINIMIS_THRESHOLD_MT:
                return None  # Already breached

            velocity = self._calc_velocity(importer_id, year)
            if velocity <= 0:
                return None  # No import activity

            remaining_mt = DE_MINIMIS_THRESHOLD_MT - eligible_mt
            months_remaining = remaining_mt / velocity
            days_remaining = int((months_remaining * Decimal("30.44")).to_integral_value(rounding=ROUND_HALF_UP))
            today = date.today()
            projected = today + timedelta(days=days_remaining)

            # Cap at end of year
            year_end = date(year, 12, 31)
            if projected > year_end:
                return None  # Will not breach within the year
            return projected

    def get_import_velocity(self, importer_id: str, year: int) -> Decimal:
        """
        Calculate average import velocity in metric tonnes per month.

        Uses the span from the first recorded import to today (or end of year
        if year is in the past).

        Args:
            importer_id: EORI or internal identifier.
            year: Calendar year.

        Returns:
            Decimal representing MT/month average, rounded to 3 decimal places.
        """
        with self._lock:
            return self._calc_velocity(importer_id, year)

    def is_exempt(self, importer_id: str, year: int) -> bool:
        """
        Return True if the importer is currently exempt from full CBAM
        reporting for the given year (cumulative eligible < 50 MT).

        Args:
            importer_id: EORI or internal identifier.
            year: Calendar year.

        Returns:
            Boolean exemption status.
        """
        with self._lock:
            eligible_mt = self._get_eligible_total(importer_id, year)
            return eligible_mt < DE_MINIMIS_THRESHOLD_MT

    def get_sector_breakdown(self, importer_id: str, year: int) -> Dict[str, SectorBreakdown]:
        """
        Return imports aggregated by CBAM sector with percentage shares.

        Args:
            importer_id: EORI or internal identifier.
            year: Calendar year.

        Returns:
            Dictionary keyed by sector name containing SectorBreakdown models.
        """
        with self._lock:
            records = self._get_records(importer_id, year)
            sectors: Dict[str, SectorBreakdown] = {}
            total = sum(r.quantity_mt for r in records) or Decimal("1")

            for r in records:
                if r.sector not in sectors:
                    sectors[r.sector] = SectorBreakdown(sector=r.sector)
                sb = sectors[r.sector]
                sb.cumulative_mt += r.quantity_mt
                sb.record_count += 1

            for sb in sectors.values():
                sb.percentage_of_total = (sb.cumulative_mt / total * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

            return sectors

    def trigger_threshold_alert(
        self, importer_id: str, pct: int
    ) -> ThresholdAlert:
        """
        Manually trigger an alert at the specified percentage level.

        Args:
            importer_id: EORI or internal identifier.
            pct: Alert percentage level (typically 80, 90, 95, or 100).

        Returns:
            ThresholdAlert that was created and dispatched.
        """
        with self._lock:
            status = self._build_status(importer_id, status_year := date.today().year)
            alert = ThresholdAlert(
                importer_id=importer_id,
                year=status_year,
                alert_level=pct,
                current_percentage=status.percentage,
                cumulative_mt=status.cumulative_mt,
                message=self._alert_message(pct, status.percentage, status.cumulative_mt),
            )
            alert.provenance_hash = self._hash_alert(alert)
            self._dispatch_alert(alert)
            return alert

    def register_alert_listener(self, listener: Any) -> None:
        """Register a callable to be invoked when alerts fire."""
        self._alert_listeners.append(listener)

    def get_all_records(self, importer_id: str, year: int) -> List[ImportRecord]:
        """Return a copy of all import records for the importer-year pair."""
        with self._lock:
            return list(self._get_records(importer_id, year))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_threshold_eligible(self, cn_code: str) -> bool:
        """
        Determine if a CN code counts toward the 50-tonne threshold.

        Electricity (2716) and hydrogen (2804) are excluded per the Omnibus
        Simplification Package.

        Args:
            cn_code: Normalised CN code (digits only).

        Returns:
            True if the import counts toward the threshold.
        """
        for prefix in _EXCLUDED_CN_PREFIXES:
            if cn_code.startswith(prefix):
                return False
        return True

    def _resolve_sector(self, cn_code: str) -> str:
        """Resolve a CN code to its CBAM sector group."""
        for prefix_len in (4,):
            prefix = cn_code[:prefix_len]
            if prefix in _CN_SECTOR_MAP:
                return _CN_SECTOR_MAP[prefix]
        return "unknown"

    def _store_record(self, record: ImportRecord) -> None:
        """Persist an import record in the in-memory store (lock held)."""
        imp = record.importer_id
        yr = record.year
        if imp not in self._records:
            self._records[imp] = {}
        if yr not in self._records[imp]:
            self._records[imp][yr] = []
        self._records[imp][yr].append(record)

    def _get_records(self, importer_id: str, year: int) -> List[ImportRecord]:
        """Retrieve import records (lock must be held by caller)."""
        return self._records.get(importer_id, {}).get(year, [])

    def _get_eligible_total(self, importer_id: str, year: int) -> Decimal:
        """Sum eligible import quantities (lock must be held by caller)."""
        records = self._get_records(importer_id, year)
        return sum(
            (r.quantity_mt for r in records if r.eligible_for_threshold),
            Decimal("0"),
        )

    def _calc_percentage(self, eligible_mt: Decimal) -> Decimal:
        """Calculate percentage of threshold consumed."""
        if DE_MINIMIS_THRESHOLD_MT == 0:
            return Decimal("100")
        return (eligible_mt / DE_MINIMIS_THRESHOLD_MT * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    def _calc_velocity(self, importer_id: str, year: int) -> Decimal:
        """
        Calculate MT/month velocity from first import to reference date.

        Lock must be held by caller.
        """
        records = [r for r in self._get_records(importer_id, year) if r.eligible_for_threshold]
        if not records:
            return Decimal("0")

        eligible_mt = sum(r.quantity_mt for r in records)
        first_import = min(r.recorded_at for r in records)
        reference = datetime.utcnow()
        elapsed_days = max((reference - first_import).days, 1)
        months_elapsed = Decimal(str(elapsed_days)) / Decimal("30.44")
        if months_elapsed <= 0:
            months_elapsed = Decimal("1")

        velocity = (eligible_mt / months_elapsed).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        return velocity

    def _build_status(self, importer_id: str, year: int) -> ThresholdStatus:
        """Build a complete ThresholdStatus (lock must be held)."""
        records = self._get_records(importer_id, year)
        eligible_mt = self._get_eligible_total(importer_id, year)
        percentage = self._calc_percentage(eligible_mt)
        velocity = self._calc_velocity(importer_id, year)

        # Sector breakdown
        sector_map: Dict[str, SectorBreakdown] = {}
        total_all = sum(r.quantity_mt for r in records) or Decimal("1")
        for r in records:
            if r.sector not in sector_map:
                sector_map[r.sector] = SectorBreakdown(sector=r.sector)
            sb = sector_map[r.sector]
            sb.cumulative_mt += r.quantity_mt
            sb.record_count += 1
        for sb in sector_map.values():
            sb.percentage_of_total = (sb.cumulative_mt / total_all * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Determine highest alert level crossed
        alert_level: Optional[int] = None
        for level in ALERT_THRESHOLDS_PCT:
            if percentage >= Decimal(str(level)):
                alert_level = level

        # Projected breach
        projected: Optional[date] = None
        if eligible_mt < DE_MINIMIS_THRESHOLD_MT and velocity > 0:
            remaining = DE_MINIMIS_THRESHOLD_MT - eligible_mt
            months_to_breach = remaining / velocity
            days_to_breach = int(
                (months_to_breach * Decimal("30.44")).to_integral_value(rounding=ROUND_HALF_UP)
            )
            candidate = date.today() + timedelta(days=days_to_breach)
            if candidate <= date(year, 12, 31):
                projected = candidate

        status = ThresholdStatus(
            importer_id=importer_id,
            year=year,
            cumulative_mt=eligible_mt,
            threshold_mt=DE_MINIMIS_THRESHOLD_MT,
            percentage=percentage,
            exempt=eligible_mt < DE_MINIMIS_THRESHOLD_MT,
            alert_level=alert_level,
            sector_breakdown=list(sector_map.values()),
            projected_breach_date=projected,
            import_velocity_mt_per_month=velocity,
            total_records=len(records),
            last_updated=datetime.utcnow(),
        )
        status.provenance_hash = self._hash_status(status)
        return status

    def _evaluate_alerts(
        self, importer_id: str, year: int, status: ThresholdStatus
    ) -> None:
        """Fire alerts for newly crossed thresholds (lock must be held)."""
        for level in ALERT_THRESHOLDS_PCT:
            key = (importer_id, year, level)
            if key in self._alerts_fired:
                continue
            if status.percentage >= Decimal(str(level)):
                self._alerts_fired.add(key)
                alert = ThresholdAlert(
                    importer_id=importer_id,
                    year=year,
                    alert_level=level,
                    current_percentage=status.percentage,
                    cumulative_mt=status.cumulative_mt,
                    message=self._alert_message(level, status.percentage, status.cumulative_mt),
                )
                alert.provenance_hash = self._hash_alert(alert)
                self._dispatch_alert(alert)
                logger.warning(
                    "THRESHOLD ALERT: importer=%s year=%d level=%d%% cumulative=%s MT",
                    importer_id, year, level, status.cumulative_mt,
                )

    def _dispatch_alert(self, alert: ThresholdAlert) -> None:
        """Invoke all registered alert listeners."""
        for listener in self._alert_listeners:
            try:
                listener(alert)
            except Exception:
                logger.exception("Alert listener failed for alert %s", alert.alert_id)

    @staticmethod
    def _alert_message(level: int, pct: Decimal, cumulative_mt: Decimal) -> str:
        """Build a human-readable alert message."""
        if level >= 100:
            return (
                f"CBAM de minimis threshold BREACHED: {cumulative_mt} MT imported "
                f"({pct}% of 50 MT). Full CBAM reporting obligations now apply."
            )
        return (
            f"CBAM de minimis threshold warning ({level}%): {cumulative_mt} MT "
            f"imported ({pct}% of 50 MT threshold). "
            f"Review import plans to determine if full CBAM compliance is needed."
        )

    # ------------------------------------------------------------------
    # Provenance hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_record(record: ImportRecord) -> str:
        """Compute SHA-256 provenance hash for an ImportRecord."""
        payload = (
            f"{record.importer_id}|{record.year}|{record.cn_code}|"
            f"{record.quantity_mt}|{record.sector}|{record.eligible_for_threshold}|"
            f"{record.recorded_at.isoformat()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_status(status: ThresholdStatus) -> str:
        """Compute SHA-256 provenance hash for a ThresholdStatus."""
        payload = (
            f"{status.importer_id}|{status.year}|{status.cumulative_mt}|"
            f"{status.threshold_mt}|{status.percentage}|{status.exempt}|"
            f"{status.total_records}|{status.last_updated.isoformat()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_alert(alert: ThresholdAlert) -> str:
        """Compute SHA-256 provenance hash for a ThresholdAlert."""
        payload = (
            f"{alert.importer_id}|{alert.year}|{alert.alert_level}|"
            f"{alert.current_percentage}|{alert.cumulative_mt}|"
            f"{alert.triggered_at.isoformat()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
