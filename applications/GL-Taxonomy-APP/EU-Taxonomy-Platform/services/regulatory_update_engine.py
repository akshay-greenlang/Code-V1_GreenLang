"""
Regulatory Update Engine -- Delegated Act Version Tracking & Transition

Implements version tracking for EU Taxonomy Delegated Acts, Technical
Screening Criteria (TSC) threshold updates, Omnibus simplification impact
assessment, transition planning between regulatory versions, update
timeline management, version compliance checking, and regulatory summary
generation.

Tracks:
    - Climate Delegated Act 2021/2139
    - Environmental Delegated Act 2023/2486
    - Complementary Delegated Act 2022/1214 (Nuclear & Gas)
    - Simplification Delegated Act 2025
    - Omnibus Regulation changes (2025-2026)

All logic is deterministic (zero-hallucination).

Reference:
    - Regulation (EU) 2020/852 (Taxonomy Regulation)
    - Delegated Regulation (EU) 2021/2139 (Climate DA)
    - Delegated Regulation (EU) 2023/2486 (Environmental DA)
    - Delegated Regulation (EU) 2022/1214 (Complementary DA)
    - EU Commission Omnibus Proposal (February 2025)
    - Platform on Sustainable Finance -- Recommendations (2022-2024)

Example:
    >>> from services.config import TaxonomyAppConfig
    >>> engine = RegulatoryUpdateEngine(TaxonomyAppConfig())
    >>> version = engine.get_applicable_version("4.1", "2025-06-15")
    >>> print(version.act_name)
    'Climate Delegated Act'
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    TAXONOMY_ACTIVITIES,
    TaxonomyAppConfig,
)
from .models import (
    _new_id,
    _now,
    _sha256,
)


# ---------------------------------------------------------------------------
# Local Delegated Act registry (seeded at engine init)
# ---------------------------------------------------------------------------

_DELEGATED_ACTS: Dict[str, Dict[str, Any]] = {
    "EU_2021_2139": {
        "name": "Climate Delegated Act",
        "type": "climate",
        "version": "1.0",
        "regulation": "Delegated Regulation (EU) 2021/2139",
        "effective_date": "2022-01-01",
        "objectives": ["climate_mitigation", "climate_adaptation"],
    },
    "EU_2023_2486": {
        "name": "Environmental Delegated Act",
        "type": "environmental",
        "version": "1.0",
        "regulation": "Delegated Regulation (EU) 2023/2486",
        "effective_date": "2024-01-01",
        "objectives": [
            "water_marine", "circular_economy",
            "pollution_prevention", "biodiversity_ecosystems",
        ],
    },
    "EU_2022_1214": {
        "name": "Complementary Climate Delegated Act (Nuclear & Gas)",
        "type": "complementary",
        "version": "1.0",
        "regulation": "Delegated Regulation (EU) 2022/1214",
        "effective_date": "2023-01-01",
        "objectives": ["climate_mitigation"],
    },
    "EU_2025_SIMPLIFICATION": {
        "name": "Taxonomy Simplification Delegated Act",
        "type": "simplification",
        "version": "draft",
        "regulation": "Proposed Simplification DA 2025",
        "effective_date": "2026-01-01",
        "objectives": [
            "climate_mitigation", "climate_adaptation",
            "water_marine", "circular_economy",
            "pollution_prevention", "biodiversity_ecosystems",
        ],
    },
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class RegulatoryVersion(BaseModel):
    """A specific version of a Delegated Act applicable to an activity."""

    id: str = Field(default_factory=_new_id)
    act_id: str = Field(...)
    act_name: str = Field(default="")
    act_type: str = Field(default="climate")
    version: str = Field(default="1.0")
    regulation_ref: str = Field(default="")
    effective_date: str = Field(default="")
    activities_affected: List[str] = Field(default_factory=list)
    is_current: bool = Field(default=True)
    superseded_by: Optional[str] = Field(None)
    notes: str = Field(default="")


class TSCUpdate(BaseModel):
    """A Technical Screening Criteria threshold update."""

    id: str = Field(default_factory=_new_id)
    activity_code: str = Field(...)
    objective: str = Field(...)
    old_threshold: str = Field(default="")
    new_threshold: str = Field(default="")
    effective_date: str = Field(...)
    source_act: str = Field(default="")
    impact_description: str = Field(default="")
    registered_at: datetime = Field(default_factory=_now)


class OmnibusImpact(BaseModel):
    """Impact assessment of the Omnibus simplification on an organization."""

    org_id: str = Field(...)
    affected_activities: List[Dict[str, Any]] = Field(default_factory=list)
    total_activities_affected: int = Field(default=0)
    threshold_changes: List[Dict[str, Any]] = Field(default_factory=list)
    simplification_benefits: List[str] = Field(default_factory=list)
    risk_areas: List[str] = Field(default_factory=list)
    estimated_effort_reduction_pct: float = Field(default=0.0)
    reporting_changes: List[str] = Field(default_factory=list)
    effective_date: str = Field(default="2026-01-01")
    recommendation: str = Field(default="")
    provenance_hash: str = Field(default="")
    assessed_at: datetime = Field(default_factory=_now)


class TransitionPlan(BaseModel):
    """Plan for transitioning between regulatory versions."""

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    from_version: str = Field(...)
    to_version: str = Field(...)
    affected_activities: List[Dict[str, Any]] = Field(default_factory=list)
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    timeline_weeks: int = Field(default=0)
    estimated_effort_days: int = Field(default=0)
    risk_level: str = Field(default="medium")
    status: str = Field(default="draft")
    created_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class UpdateTimeline(BaseModel):
    """Timeline of upcoming and past regulatory changes."""

    upcoming: List[Dict[str, Any]] = Field(default_factory=list)
    past: List[Dict[str, Any]] = Field(default_factory=list)
    next_effective_date: Optional[str] = Field(None)
    total_upcoming: int = Field(default=0)
    total_past: int = Field(default=0)
    generated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# RegulatoryUpdateEngine
# ---------------------------------------------------------------------------

class RegulatoryUpdateEngine:
    """
    Delegated Act version tracking and transition management engine.

    Registers and tracks EU Taxonomy Delegated Act versions, TSC
    threshold updates, assesses Omnibus simplification impacts, plans
    transitions between versions, and provides regulatory timelines.

    Attributes:
        config: Application configuration.
        _acts: In-memory Delegated Act registry keyed by act_id.
        _tsc_updates: TSC threshold updates keyed by (activity_code, objective).
        _org_activities: Organization activity lists keyed by org_id.

    Example:
        >>> engine = RegulatoryUpdateEngine(TaxonomyAppConfig())
        >>> engine.register_delegated_act("EU_2021_2139", ...)
        >>> version = engine.get_applicable_version("4.1", "2025-06-15")
    """

    def __init__(self, config: Optional[TaxonomyAppConfig] = None) -> None:
        """Initialize the RegulatoryUpdateEngine."""
        self.config = config or TaxonomyAppConfig()
        self._acts: Dict[str, RegulatoryVersion] = {}
        self._tsc_updates: Dict[str, List[TSCUpdate]] = {}
        self._org_activities: Dict[str, List[Dict[str, Any]]] = {}

        # Seed from config registry
        self._seed_from_config()
        logger.info("RegulatoryUpdateEngine initialized with %d acts", len(self._acts))

    # ------------------------------------------------------------------
    # Seed from Config
    # ------------------------------------------------------------------

    def _seed_from_config(self) -> None:
        """Pre-populate the Delegated Act registry from config constants."""
        for act_id, act_data in _DELEGATED_ACTS.items():
            version = RegulatoryVersion(
                act_id=act_id,
                act_name=act_data["name"],
                act_type=act_data["type"],
                version=act_data["version"],
                regulation_ref=act_data["regulation"],
                effective_date=act_data["effective_date"],
                activities_affected=[],
                is_current=act_data["version"] != "draft",
            )
            self._acts[act_id] = version

    # ------------------------------------------------------------------
    # Register Delegated Act
    # ------------------------------------------------------------------

    def register_delegated_act(
        self,
        act_id: str,
        name: str,
        version: str,
        effective_date: str,
        activities_affected: Optional[List[str]] = None,
    ) -> str:
        """
        Register or update a Delegated Act version.

        Args:
            act_id: Unique identifier for the Delegated Act.
            name: Human-readable name.
            version: Version string (e.g. "1.0", "2.0").
            effective_date: Effective date (YYYY-MM-DD).
            activities_affected: List of activity codes affected.

        Returns:
            The act_id of the registered Delegated Act.
        """
        act = RegulatoryVersion(
            act_id=act_id,
            act_name=name,
            version=version,
            effective_date=effective_date,
            activities_affected=activities_affected or [],
            is_current=True,
        )

        # If updating an existing act, mark prior as superseded
        existing = self._acts.get(act_id)
        if existing and existing.version != version:
            existing.is_current = False
            existing.superseded_by = f"{act_id}_v{version}"
            # Store old version under versioned key
            old_key = f"{act_id}_v{existing.version}"
            self._acts[old_key] = existing

        self._acts[act_id] = act

        logger.info(
            "Registered Delegated Act %s v%s (effective %s, %d activities)",
            act_id, version, effective_date, len(activities_affected or []),
        )
        return act_id

    # ------------------------------------------------------------------
    # Track TSC Update
    # ------------------------------------------------------------------

    def track_tsc_update(
        self,
        activity_code: str,
        objective: str,
        old_threshold: str,
        new_threshold: str,
        effective_date: str,
    ) -> str:
        """
        Register a Technical Screening Criteria threshold change.

        Tracks the change from old to new threshold for a specific
        activity and objective, with the effective date.

        Args:
            activity_code: EU Taxonomy activity code.
            objective: Environmental objective.
            old_threshold: Previous threshold description.
            new_threshold: Updated threshold description.
            effective_date: Date the new threshold takes effect.

        Returns:
            Update identifier string.
        """
        update = TSCUpdate(
            activity_code=activity_code,
            objective=objective,
            old_threshold=old_threshold,
            new_threshold=new_threshold,
            effective_date=effective_date,
        )

        key = f"{activity_code}:{objective}"
        self._tsc_updates.setdefault(key, []).append(update)

        logger.info(
            "TSC update for %s (%s): '%s' -> '%s' effective %s",
            activity_code, objective, old_threshold, new_threshold, effective_date,
        )
        return update.id

    # ------------------------------------------------------------------
    # Omnibus Impact Assessment
    # ------------------------------------------------------------------

    def assess_omnibus_impact(self, org_id: str) -> OmnibusImpact:
        """
        Assess the impact of the Omnibus simplification on an organization.

        The 2025 Omnibus Regulation proposes simplification of EU Taxonomy
        reporting requirements, including reduced disclosure scope for
        smaller entities, streamlined templates, and extended transition
        periods.

        Args:
            org_id: Organization identifier.

        Returns:
            OmnibusImpact with affected activities and simplification benefits.
        """
        start = datetime.utcnow()

        org_acts = self._org_activities.get(org_id, [])

        affected: List[Dict[str, Any]] = []
        threshold_changes: List[Dict[str, Any]] = []

        for act in org_acts:
            code = act.get("activity_code", "")
            # Check if any TSC updates affect this activity
            for key, updates in self._tsc_updates.items():
                if key.startswith(code):
                    for upd in updates:
                        affected.append({
                            "activity_code": code,
                            "objective": upd.objective,
                            "change": f"{upd.old_threshold} -> {upd.new_threshold}",
                            "effective_date": upd.effective_date,
                        })
                        threshold_changes.append({
                            "activity_code": code,
                            "old": upd.old_threshold,
                            "new": upd.new_threshold,
                        })

        # Simplification benefits
        benefits = [
            "Reduced reporting burden for SMEs under Omnibus threshold exemptions",
            "Streamlined Article 8 templates with fewer mandatory rows",
            "Extended phase-in for environmental objectives (water, CE, pollution, biodiversity)",
            "Simplified DNSH documentation requirements",
            "Voluntary BTAR reporting with simplified methodology",
        ]

        # Risk areas
        risks = [
            "Transition period may require parallel reporting under old and new rules",
            "TSC threshold changes may reclassify currently-aligned activities",
            "Reduced scope may limit comparability with peers maintaining full disclosure",
        ]

        # Reporting changes
        reporting_changes = [
            "Simplified template format for companies below 750 employees",
            "Reduced qualitative disclosure requirements",
            "Consolidated KPI template (single table instead of three)",
        ]

        effort_reduction = 15.0 if not org_acts else min(
            30.0, len(threshold_changes) * 5.0 + 10.0,
        )

        recommendation = (
            "Review TSC threshold changes against current activity data. "
            "Plan for dual reporting during transition. "
            "Engage auditors on updated requirements."
        )

        provenance = _sha256(
            f"omnibus:{org_id}:{len(affected)}:{len(threshold_changes)}"
        )

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Omnibus impact for %s: %d affected, %d threshold changes in %.1f ms",
            org_id, len(affected), len(threshold_changes), elapsed,
        )

        return OmnibusImpact(
            org_id=org_id,
            affected_activities=affected,
            total_activities_affected=len(affected),
            threshold_changes=threshold_changes,
            simplification_benefits=benefits,
            risk_areas=risks,
            estimated_effort_reduction_pct=round(effort_reduction, 2),
            reporting_changes=reporting_changes,
            recommendation=recommendation,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Applicable Version
    # ------------------------------------------------------------------

    def get_applicable_version(
        self, activity_code: str, assessment_date: str,
    ) -> RegulatoryVersion:
        """
        Determine the applicable Delegated Act version for an activity.

        Finds the most recent Delegated Act that covers the activity
        and was effective on or before the assessment date.

        Args:
            activity_code: EU Taxonomy activity code.
            assessment_date: Date of assessment (YYYY-MM-DD).

        Returns:
            RegulatoryVersion representing the applicable act.
        """
        try:
            assess_dt = datetime.strptime(assessment_date, "%Y-%m-%d").date()
        except ValueError:
            assess_dt = date.today()

        # Find activity info
        act_info = TAXONOMY_ACTIVITIES.get(activity_code, {})
        activity_objectives = act_info.get("objectives", ["climate_mitigation"])

        # Find applicable act
        candidates: List[RegulatoryVersion] = []
        for act_id, act in self._acts.items():
            try:
                eff_date = datetime.strptime(act.effective_date, "%Y-%m-%d").date()
            except ValueError:
                continue

            if eff_date > assess_dt:
                continue

            # Check if act covers any of the activity's objectives
            act_config = _DELEGATED_ACTS.get(act.act_id, {})
            da_objectives = act_config.get("objectives", [])
            if any(obj in da_objectives for obj in activity_objectives):
                candidates.append(act)

        if not candidates:
            return RegulatoryVersion(
                act_id="none",
                act_name="No applicable Delegated Act found",
                notes=(
                    f"No Delegated Act effective on {assessment_date} "
                    f"covers activity {activity_code}"
                ),
            )

        # Return the most recent effective version
        best = max(
            candidates,
            key=lambda a: a.effective_date,
        )

        logger.debug(
            "Applicable version for %s on %s: %s v%s",
            activity_code, assessment_date, best.act_name, best.version,
        )
        return best

    # ------------------------------------------------------------------
    # Update Timeline
    # ------------------------------------------------------------------

    def get_update_timeline(self) -> UpdateTimeline:
        """
        Generate a timeline of past and upcoming regulatory changes.

        Categorizes all registered Delegated Acts and TSC updates into
        past (already effective) and upcoming (future effective date).

        Returns:
            UpdateTimeline with past and upcoming changes.
        """
        today = date.today()
        upcoming: List[Dict[str, Any]] = []
        past: List[Dict[str, Any]] = []

        # Delegated Acts
        for act_id, act in self._acts.items():
            try:
                eff_date = datetime.strptime(act.effective_date, "%Y-%m-%d").date()
            except ValueError:
                continue

            entry = {
                "type": "delegated_act",
                "act_id": act.act_id,
                "name": act.act_name,
                "version": act.version,
                "effective_date": act.effective_date,
                "is_current": act.is_current,
            }

            if eff_date > today:
                upcoming.append(entry)
            else:
                past.append(entry)

        # TSC updates
        for key, updates in self._tsc_updates.items():
            for upd in updates:
                try:
                    eff_date = datetime.strptime(upd.effective_date, "%Y-%m-%d").date()
                except ValueError:
                    continue

                entry = {
                    "type": "tsc_update",
                    "activity_code": upd.activity_code,
                    "objective": upd.objective,
                    "change": f"{upd.old_threshold} -> {upd.new_threshold}",
                    "effective_date": upd.effective_date,
                }

                if eff_date > today:
                    upcoming.append(entry)
                else:
                    past.append(entry)

        # Sort
        upcoming.sort(key=lambda x: x["effective_date"])
        past.sort(key=lambda x: x["effective_date"], reverse=True)

        next_date = upcoming[0]["effective_date"] if upcoming else None

        return UpdateTimeline(
            upcoming=upcoming,
            past=past,
            next_effective_date=next_date,
            total_upcoming=len(upcoming),
            total_past=len(past),
        )

    # ------------------------------------------------------------------
    # Transition Planning
    # ------------------------------------------------------------------

    def plan_transition(
        self, org_id: str, from_version: str, to_version: str,
    ) -> TransitionPlan:
        """
        Create a transition plan between two regulatory versions.

        Identifies affected activities, generates action items, and
        estimates the timeline and effort required for the transition.

        Args:
            org_id: Organization identifier.
            from_version: Source version identifier.
            to_version: Target version identifier.

        Returns:
            TransitionPlan with actions and timeline.
        """
        start = datetime.utcnow()

        from_act = self._acts.get(from_version)
        to_act = self._acts.get(to_version)

        # Identify affected activities
        org_acts = self._org_activities.get(org_id, [])
        affected: List[Dict[str, Any]] = []
        action_items: List[Dict[str, Any]] = []

        for act in org_acts:
            code = act.get("activity_code", "")
            affected.append({
                "activity_code": code,
                "current_status": act.get("alignment_status", "unknown"),
                "action_needed": "review_tsc",
            })

        # Standard transition action items
        action_items.extend([
            {
                "priority": 1,
                "action": "Review updated TSC thresholds for all activities",
                "responsible": "Sustainability team",
                "deadline_weeks": 4,
                "status": "pending",
            },
            {
                "priority": 2,
                "action": "Reassess DNSH criteria under new version",
                "responsible": "Environmental compliance",
                "deadline_weeks": 6,
                "status": "pending",
            },
            {
                "priority": 3,
                "action": "Update Article 8 reporting templates",
                "responsible": "Finance/reporting team",
                "deadline_weeks": 8,
                "status": "pending",
            },
            {
                "priority": 4,
                "action": "Engage external auditor on updated requirements",
                "responsible": "Internal audit",
                "deadline_weeks": 10,
                "status": "pending",
            },
            {
                "priority": 5,
                "action": "Update internal training materials",
                "responsible": "HR/training",
                "deadline_weeks": 12,
                "status": "pending",
            },
        ])

        # Estimate effort
        base_effort = max(len(affected) * 2, 10)
        timeline_weeks = 12

        # Risk assessment
        risk_level = "low"
        if len(affected) > 10:
            risk_level = "medium"
        if len(affected) > 20:
            risk_level = "high"

        provenance = _sha256(
            f"transition:{org_id}:{from_version}:{to_version}:{len(affected)}"
        )

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Transition plan for %s (%s -> %s): %d affected, risk=%s in %.1f ms",
            org_id, from_version, to_version, len(affected), risk_level, elapsed,
        )

        return TransitionPlan(
            org_id=org_id,
            from_version=from_version,
            to_version=to_version,
            affected_activities=affected,
            action_items=action_items,
            timeline_weeks=timeline_weeks,
            estimated_effort_days=base_effort,
            risk_level=risk_level,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Version History
    # ------------------------------------------------------------------

    def get_version_history(
        self, delegated_act: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the version history of a Delegated Act.

        Args:
            delegated_act: Delegated Act base identifier (e.g. "EU_2021_2139").

        Returns:
            List of version dicts sorted by effective date descending.
        """
        versions: List[Dict[str, Any]] = []

        for act_id, act in self._acts.items():
            if act.act_id == delegated_act or act_id.startswith(delegated_act):
                versions.append({
                    "act_id": act.act_id,
                    "version": act.version,
                    "effective_date": act.effective_date,
                    "is_current": act.is_current,
                    "superseded_by": act.superseded_by,
                    "act_name": act.act_name,
                })

        versions.sort(key=lambda v: v["effective_date"], reverse=True)

        logger.info(
            "Version history for %s: %d versions", delegated_act, len(versions),
        )
        return versions

    # ------------------------------------------------------------------
    # Version Compliance Check
    # ------------------------------------------------------------------

    def check_version_compliance(
        self, org_id: str, assessment_date: str,
    ) -> Dict[str, Any]:
        """
        Check if an organization's assessments use the correct regulatory version.

        Verifies that each activity's alignment was assessed under the
        Delegated Act version applicable on the assessment date.

        Args:
            org_id: Organization identifier.
            assessment_date: Date of the assessment (YYYY-MM-DD).

        Returns:
            Dict with compliance status, compliant/non-compliant counts,
            and specific non-compliance details.
        """
        org_acts = self._org_activities.get(org_id, [])

        compliant = 0
        non_compliant = 0
        details: List[Dict[str, Any]] = []

        for act in org_acts:
            code = act.get("activity_code", "")
            assessed_under = act.get("assessed_under_version", "")

            applicable = self.get_applicable_version(code, assessment_date)

            is_compliant = (
                assessed_under == applicable.act_id
                or assessed_under == ""
                or applicable.act_id == "none"
            )

            if is_compliant:
                compliant += 1
            else:
                non_compliant += 1
                details.append({
                    "activity_code": code,
                    "assessed_under": assessed_under,
                    "applicable_version": applicable.act_id,
                    "message": (
                        f"Activity {code} was assessed under {assessed_under} "
                        f"but should use {applicable.act_id} "
                        f"(effective {applicable.effective_date})"
                    ),
                })

        total = len(org_acts)
        compliance_pct = (compliant / total * 100.0) if total > 0 else 100.0

        return {
            "org_id": org_id,
            "assessment_date": assessment_date,
            "total_activities": total,
            "compliant_count": compliant,
            "non_compliant_count": non_compliant,
            "compliance_pct": round(compliance_pct, 2),
            "is_fully_compliant": non_compliant == 0,
            "non_compliant_details": details,
        }

    # ------------------------------------------------------------------
    # Regulatory Summary
    # ------------------------------------------------------------------

    def get_regulatory_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the regulatory landscape.

        Provides an overview of all registered Delegated Acts, their
        status, total TSC updates, and upcoming changes.

        Returns:
            Dict with regulatory landscape overview.
        """
        current_acts = [a for a in self._acts.values() if a.is_current]
        draft_acts = [a for a in self._acts.values() if a.version == "draft"]

        total_tsc_updates = sum(len(v) for v in self._tsc_updates.values())
        timeline = self.get_update_timeline()

        return {
            "total_delegated_acts": len(self._acts),
            "current_acts": len(current_acts),
            "draft_acts": len(draft_acts),
            "total_tsc_updates": total_tsc_updates,
            "upcoming_changes": timeline.total_upcoming,
            "next_effective_date": timeline.next_effective_date,
            "acts": [
                {
                    "act_id": a.act_id,
                    "name": a.act_name,
                    "type": a.act_type,
                    "version": a.version,
                    "effective_date": a.effective_date,
                    "is_current": a.is_current,
                }
                for a in self._acts.values()
            ],
            "generated_at": _now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Organization Activity Registration
    # ------------------------------------------------------------------

    def register_org_activities(
        self, org_id: str, activities: List[Dict[str, Any]],
    ) -> None:
        """
        Register an organization's activities for regulatory tracking.

        Args:
            org_id: Organization identifier.
            activities: List of activity dicts with activity_code and metadata.
        """
        self._org_activities[org_id] = activities
        logger.info(
            "Registered %d activities for org %s", len(activities), org_id,
        )
