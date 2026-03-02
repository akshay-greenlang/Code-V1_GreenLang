# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Supplier Portal - Supplier Dashboard Service

Aggregation service providing dashboard data for the supplier self-service
portal.  Combines data from SupplierRegistryEngine and
EmissionsSubmissionEngine to present a unified view of supplier status,
submissions, data quality, deadlines, and emissions trends.

All aggregations are deterministic (zero-hallucination) and use Decimal
for emissions arithmetic.

Version: 1.1.0
Author: GreenLang CBAM Team
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from calendar import monthrange
from collections import Counter, defaultdict
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from supplier_portal.models import (
    CBAMSector,
    DataQualityScore,
    Deadline,
    EmissionsDataSubmission,
    SubmissionStatus,
    SupplierDashboard,
    SupplierProfile,
    SupplierStatus,
    VerificationStatus,
    _quantize,
    _utc_now,
)

logger = logging.getLogger(__name__)


# ============================================================================
# JSON ENCODER
# ============================================================================


class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder that converts Decimal and datetime for hashing."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


# ============================================================================
# SUPPLIER DASHBOARD SERVICE
# ============================================================================


class SupplierDashboardService:
    """
    Aggregation service for the CBAM Supplier Portal dashboard.

    Combines data from the registry and submission engines to provide
    comprehensive dashboard views including submission summaries, data
    quality overviews, upcoming deadlines, verification timelines,
    emissions trends, and linked importer information.

    Thread Safety:
      Read operations acquire locks from the underlying engines.
      Aggregation logic is stateless and thread-safe.

    Args:
        registry: SupplierRegistryEngine instance.
        submissions: EmissionsSubmissionEngine instance.

    Example:
        >>> service = SupplierDashboardService(registry, submissions)
        >>> dashboard = service.get_dashboard("SUP-ABC123")
        >>> print(dashboard.total_submissions)
    """

    def __init__(self, registry: Any, submissions: Any) -> None:
        """Initialize with registry and submission engines."""
        self._registry = registry
        self._submissions = submissions
        self._lock = threading.RLock()
        logger.info("SupplierDashboardService initialized")

    def get_dashboard(self, supplier_id: str) -> SupplierDashboard:
        """
        Build and return the full supplier dashboard.

        Aggregates all dashboard sections into a single SupplierDashboard
        model.  Each section is independently computed and failures in
        one section do not block others.

        Args:
            supplier_id: Unique supplier identifier.

        Returns:
            SupplierDashboard with all aggregated data.

        Raises:
            Exception: Propagates SupplierNotFoundError from registry.
        """
        start_time = datetime.now(timezone.utc)

        with self._lock:
            profile = self._registry.get_supplier(supplier_id)
            installations = self._registry.get_installations(supplier_id)

            # Gather all submissions for this supplier
            all_submissions = self._get_supplier_submissions(supplier_id)

            # Build each section
            submission_summary = self._build_submission_summary(all_submissions)
            period_summary = self._build_period_summary(all_submissions)
            dq_overview = self._build_data_quality_overview(all_submissions)
            deadlines = self._build_upcoming_deadlines(
                supplier_id, installations, all_submissions
            )
            verification_timeline = self._build_verification_timeline(
                supplier_id, installations
            )
            emissions_trend = self._build_emissions_trend(all_submissions)
            linked_summary = self._build_linked_importers_summary(supplier_id)
            recent_activity = self._build_recent_activity(supplier_id)

            dashboard = SupplierDashboard(
                supplier_id=supplier_id,
                company_name=profile.company_name,
                status=profile.status,
                installations_count=len(installations),
                total_submissions=len(all_submissions),
                submission_summary=submission_summary,
                period_summary=period_summary,
                data_quality_overview=dq_overview,
                upcoming_deadlines=deadlines,
                verification_timeline=verification_timeline,
                emissions_trend=emissions_trend,
                linked_importers_count=len(profile.linked_importers),
                recent_activity=recent_activity,
                generated_at=datetime.now(timezone.utc),
            )

            # Compute provenance
            dashboard = dashboard.model_copy(
                update={
                    "provenance_hash": self._compute_provenance_hash(
                        dashboard.model_dump(mode="json")
                    )
                }
            )

            duration_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            logger.info(
                "Dashboard built for supplier %s in %.1f ms (%d submissions)",
                supplier_id,
                duration_ms,
                len(all_submissions),
            )

        return dashboard

    def get_submission_summary(self, supplier_id: str) -> Dict[str, Any]:
        """
        Get submission count by status and period for a supplier.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Dictionary with 'by_status' and 'by_period' breakdowns.
        """
        with self._lock:
            submissions = self._get_supplier_submissions(supplier_id)
            by_status = self._build_submission_summary(submissions)
            by_period = self._build_period_summary(submissions)

            return {
                "supplier_id": supplier_id,
                "total": len(submissions),
                "by_status": by_status,
                "by_period": by_period,
            }

    def get_data_quality_overview(self, supplier_id: str) -> Dict[str, Any]:
        """
        Get average data quality scores across all submissions.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Dictionary with average scores per dimension and per installation.
        """
        with self._lock:
            submissions = self._get_supplier_submissions(supplier_id)
            dq = self._build_data_quality_overview(submissions)

            # Per-installation breakdown
            installation_scores: Dict[str, List[DataQualityScore]] = defaultdict(list)
            for sub in submissions:
                if sub.data_quality_score:
                    installation_scores[sub.installation_id].append(
                        sub.data_quality_score
                    )

            inst_averages = {}
            for inst_id, scores in installation_scores.items():
                inst_averages[inst_id] = self._average_quality_scores(scores)

            return {
                "supplier_id": supplier_id,
                "overall": dq.model_dump(mode="json") if dq else None,
                "by_installation": {
                    k: v.model_dump(mode="json")
                    for k, v in inst_averages.items()
                },
                "submission_count": len(submissions),
            }

    def get_upcoming_deadlines(self, supplier_id: str) -> List[Deadline]:
        """
        Get upcoming submission and verification deadlines.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            List of Deadline objects sorted by due date.
        """
        with self._lock:
            installations = self._registry.get_installations(supplier_id)
            submissions = self._get_supplier_submissions(supplier_id)
            return self._build_upcoming_deadlines(
                supplier_id, installations, submissions
            )

    def get_verification_timeline(self, supplier_id: str) -> Dict[str, Any]:
        """
        Get verification status and timeline for all installations.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Dictionary with verification status per installation and
            next scheduled visit dates.
        """
        with self._lock:
            installations = self._registry.get_installations(supplier_id)
            return self._build_verification_timeline(supplier_id, installations)

    def get_emissions_trend(
        self,
        supplier_id: str,
        periods: Optional[int] = 8,
    ) -> Dict[str, Any]:
        """
        Get emissions trend data across reporting periods.

        Args:
            supplier_id: Supplier identifier.
            periods: Number of recent periods to include (default 8).

        Returns:
            Dictionary with period-by-period emissions breakdown.
        """
        with self._lock:
            submissions = self._get_supplier_submissions(supplier_id)

            # Filter to only accepted/submitted
            active = [
                s
                for s in submissions
                if s.submission_status in {
                    SubmissionStatus.ACCEPTED,
                    SubmissionStatus.SUBMITTED,
                }
            ]

            trend = self._build_emissions_trend(active)

            # Limit to most recent N periods
            if periods and "periods" in trend:
                period_keys = sorted(trend["periods"].keys())
                if len(period_keys) > periods:
                    recent_keys = period_keys[-periods:]
                    trend["periods"] = {
                        k: trend["periods"][k] for k in recent_keys
                    }

            return {
                "supplier_id": supplier_id,
                **trend,
            }

    def get_linked_importers_summary(
        self, supplier_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get summary information about linked importers.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            List of importer summary dictionaries.
        """
        with self._lock:
            return self._build_linked_importers_summary(supplier_id)

    # ------------------------------------------------------------------
    # Private: Section Builders
    # ------------------------------------------------------------------

    def _get_supplier_submissions(
        self, supplier_id: str
    ) -> List[EmissionsDataSubmission]:
        """Retrieve all submissions for the supplier's installations."""
        try:
            return self._submissions.get_submissions(supplier_id=supplier_id)
        except Exception as e:
            logger.warning(
                "Could not fetch submissions for %s: %s", supplier_id, e
            )
            return []

    def _build_submission_summary(
        self, submissions: List[EmissionsDataSubmission]
    ) -> Dict[str, int]:
        """Count submissions by status."""
        counter = Counter(s.submission_status.value for s in submissions)
        return dict(counter)

    def _build_period_summary(
        self, submissions: List[EmissionsDataSubmission]
    ) -> Dict[str, int]:
        """Count submissions by reporting period."""
        counter = Counter(s.reporting_period for s in submissions)
        return dict(sorted(counter.items()))

    def _build_data_quality_overview(
        self, submissions: List[EmissionsDataSubmission]
    ) -> Optional[DataQualityScore]:
        """Compute average data quality across all scored submissions."""
        scores = [
            s.data_quality_score
            for s in submissions
            if s.data_quality_score is not None
        ]

        if not scores:
            return None

        return self._average_quality_scores(scores)

    def _build_upcoming_deadlines(
        self,
        supplier_id: str,
        installations: List[Any],
        submissions: List[EmissionsDataSubmission],
    ) -> List[Deadline]:
        """
        Build list of upcoming deadlines.

        Generates deadlines for:
          - Next quarterly submission (for each installation)
          - Verification expiry warnings
        """
        deadlines: List[Deadline] = []
        today = date.today()

        # Determine the current reporting quarter
        current_year = today.year
        current_quarter = (today.month - 1) // 3 + 1

        # Submitted periods by installation
        submitted_periods: Dict[str, set] = defaultdict(set)
        for sub in submissions:
            submitted_periods[sub.installation_id].add(sub.reporting_period)

        for inst in installations:
            if not inst.is_active:
                continue

            # Check next submission deadline
            next_period = f"{current_year}Q{current_quarter}"
            if next_period not in submitted_periods.get(inst.installation_id, set()):
                # Deadline is end of month following quarter end
                quarter_end_month = current_quarter * 3
                if quarter_end_month >= 12:
                    dl_year = current_year + 1
                    dl_month = 1
                else:
                    dl_year = current_year
                    dl_month = quarter_end_month + 1

                _, last_day = monthrange(dl_year, dl_month)
                due = date(dl_year, dl_month, last_day)

                deadlines.append(
                    Deadline(
                        deadline_type="submission_due",
                        description=(
                            f"Submit emissions data for {inst.name} - {next_period}"
                        ),
                        due_date=due,
                        installation_id=inst.installation_id,
                        reporting_period=next_period,
                        is_overdue=due < today,
                        priority="high" if due < today else "medium",
                    )
                )

            # Check verification expiry
            if inst.verified_until:
                days_to_expiry = (inst.verified_until - today).days
                if days_to_expiry <= 90:
                    deadlines.append(
                        Deadline(
                            deadline_type="verification_expiry",
                            description=(
                                f"Verification expiry for {inst.name}: "
                                f"{inst.verified_until.isoformat()}"
                            ),
                            due_date=inst.verified_until,
                            installation_id=inst.installation_id,
                            is_overdue=inst.verified_until < today,
                            priority=(
                                "high" if days_to_expiry <= 30 else "medium"
                            ),
                        )
                    )

        # Sort by due_date
        deadlines.sort(key=lambda d: d.due_date)

        return deadlines

    def _build_verification_timeline(
        self, supplier_id: str, installations: List[Any]
    ) -> Dict[str, Any]:
        """Build verification status and timeline for all installations."""
        today = date.today()
        timeline: Dict[str, Any] = {
            "supplier_status": None,
            "installations": [],
            "next_expiry": None,
            "all_verified": False,
        }

        try:
            profile = self._registry.get_supplier(supplier_id)
            timeline["supplier_status"] = profile.status.value
        except Exception:
            pass

        all_verified = True
        nearest_expiry: Optional[date] = None

        for inst in installations:
            if not inst.is_active:
                continue

            is_verified = inst.verification_status == VerificationStatus.VERIFIED
            if not is_verified:
                all_verified = False

            entry = {
                "installation_id": inst.installation_id,
                "name": inst.name,
                "verification_status": inst.verification_status.value,
                "verified_until": (
                    inst.verified_until.isoformat()
                    if inst.verified_until
                    else None
                ),
                "is_current": (
                    inst.verified_until > today if inst.verified_until else False
                ),
                "days_remaining": (
                    (inst.verified_until - today).days
                    if inst.verified_until
                    else None
                ),
            }
            timeline["installations"].append(entry)

            if inst.verified_until:
                if nearest_expiry is None or inst.verified_until < nearest_expiry:
                    nearest_expiry = inst.verified_until

        timeline["all_verified"] = all_verified and len(installations) > 0
        timeline["next_expiry"] = (
            nearest_expiry.isoformat() if nearest_expiry else None
        )

        return timeline

    def _build_emissions_trend(
        self, submissions: List[EmissionsDataSubmission]
    ) -> Dict[str, Any]:
        """
        Build period-by-period emissions trend data.

        Groups submissions by reporting period and aggregates direct,
        indirect, and total emissions along with production volumes.
        """
        periods_data: Dict[str, Dict[str, Decimal]] = {}

        for sub in submissions:
            period = sub.reporting_period
            if period not in periods_data:
                periods_data[period] = {
                    "direct": Decimal("0"),
                    "indirect": Decimal("0"),
                    "total": Decimal("0"),
                    "volume": Decimal("0"),
                    "count": Decimal("0"),
                }

            pd = periods_data[period]
            pd["direct"] += sub.direct_emissions_tCO2e_per_mt * sub.quantity_mt
            pd["indirect"] += (
                sub.indirect_emissions_tCO2e_per_mt * sub.quantity_mt
            )
            pd["total"] += (
                sub.total_embedded_emissions_tCO2e_per_mt * sub.quantity_mt
            )
            pd["volume"] += sub.quantity_mt
            pd["count"] += Decimal("1")

        # Format for output
        trend: Dict[str, Any] = {
            "periods": {},
            "period_list": sorted(periods_data.keys()),
        }

        for period_key in sorted(periods_data.keys()):
            pd = periods_data[period_key]
            volume = pd["volume"]
            trend["periods"][period_key] = {
                "direct_emissions_tco2e": str(_quantize(pd["direct"])),
                "indirect_emissions_tco2e": str(_quantize(pd["indirect"])),
                "total_emissions_tco2e": str(_quantize(pd["total"])),
                "production_volume_mt": str(_quantize(volume)),
                "intensity_tco2e_per_mt": (
                    str(_quantize(pd["total"] / volume))
                    if volume > Decimal("0")
                    else "0"
                ),
                "submission_count": int(pd["count"]),
            }

        return trend

    def _build_linked_importers_summary(
        self, supplier_id: str
    ) -> List[Dict[str, Any]]:
        """Build summary of linked importers."""
        try:
            importer_ids = self._registry.get_linked_importers(supplier_id)
        except Exception:
            return []

        summary = []
        for imp_id in importer_ids:
            summary.append({
                "importer_id": imp_id,
                "linked": True,
                "access_status": "active",
            })

        return summary

    def _build_recent_activity(
        self, supplier_id: str
    ) -> List[Dict[str, Any]]:
        """Build recent activity log from audit trails."""
        activity: List[Dict[str, Any]] = []

        # Registry audit trail
        try:
            registry_audit = self._registry.get_audit_trail(
                supplier_id=supplier_id, limit=10
            )
            for entry in registry_audit:
                activity.append({
                    "timestamp": entry.get("timestamp"),
                    "action": entry.get("action"),
                    "resource_type": entry.get("resource_type"),
                    "resource_id": entry.get("resource_id"),
                    "source": "registry",
                })
        except Exception:
            pass

        # Sort by timestamp descending
        activity.sort(
            key=lambda a: a.get("timestamp", ""),
            reverse=True,
        )

        return activity[:20]

    # ------------------------------------------------------------------
    # Private: Utility Helpers
    # ------------------------------------------------------------------

    def _average_quality_scores(
        self, scores: List[DataQualityScore]
    ) -> DataQualityScore:
        """Compute weighted average of multiple DataQualityScore objects."""
        if not scores:
            return DataQualityScore(
                completeness=Decimal("0"),
                consistency=Decimal("0"),
                timeliness=Decimal("0"),
                accuracy=Decimal("0"),
                overall=Decimal("0"),
            )

        n = Decimal(len(scores))
        avg_completeness = _quantize(
            sum(s.completeness for s in scores) / n
        )
        avg_consistency = _quantize(
            sum(s.consistency for s in scores) / n
        )
        avg_timeliness = _quantize(
            sum(s.timeliness for s in scores) / n
        )
        avg_accuracy = _quantize(
            sum(s.accuracy for s in scores) / n
        )
        avg_overall = _quantize(
            avg_completeness * Decimal("0.30")
            + avg_consistency * Decimal("0.25")
            + avg_timeliness * Decimal("0.20")
            + avg_accuracy * Decimal("0.25")
        )

        return DataQualityScore(
            completeness=avg_completeness,
            consistency=avg_consistency,
            timeliness=avg_timeliness,
            accuracy=avg_accuracy,
            overall=avg_overall,
        )

    def _compute_provenance_hash(self, data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        serialized = json.dumps(data, sort_keys=True, cls=_DecimalEncoder)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
