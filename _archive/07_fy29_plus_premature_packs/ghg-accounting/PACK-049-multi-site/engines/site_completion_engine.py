# -*- coding: utf-8 -*-
"""
PACK-049 GHG Multi-Site Management Pack - Site Completion Engine
=================================================================

Tracks the completeness of GHG data collection across a multi-site
portfolio.  Scores each facility's submission against required scopes
and categories, manages collection deadlines with reminder escalation,
detects data gaps at site, scope, and category level, and calculates
aggregate coverage metrics for the portfolio.

Completeness Score:
    site_score = (measured_items + estimated_items) / required_items * 100
    A score of 100 means all required categories have been reported.
    Estimated items count toward completeness but lower quality scores.

Coverage Metrics:
    sites_reporting_pct    = reporting_sites / total_sites * 100
    emissions_covered_pct  = sum(reporting_emissions) / total_emissions * 100
    floor_area_covered_pct = sum(reporting_area) / total_area * 100

Gap Types:
    MISSING_SITE:     Site registered but no submission received
    MISSING_SCOPE:    Site submitted but scope not reported
    MISSING_CATEGORY: Scope submitted but required category absent
    STALE_DATA:       Submission older than configured freshness window
    ESTIMATION_HEAVY: >50% of site data is estimated rather than measured

Escalation:
    Level 0: Standard reminder (days_before deadline)
    Level 1: Escalation to site manager (deadline passed)
    Level 2: Escalation to regional lead (7 days overdue)
    Level 3: Escalation to portfolio admin (14 days overdue)

Provenance:
    SHA-256 hash on every CompletionResult.

Regulatory References:
    - GHG Protocol Corporate Standard (2004, rev 2015) Ch 6 - Identifying and calculating
    - ISO 14064-1:2018 Clause 5 - Completeness requirements
    - EU CSRD / ESRS E1 - Disclosure completeness
    - GHG Protocol Scope 3 Standard Ch 7 - Data management
    - PCAF v3 2024 - Portfolio completeness and data availability

Zero-Hallucination:
    - All scores and metrics are deterministic Decimal arithmetic
    - No LLM involvement in scoring or gap detection
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-049 GHG Multi-Site Management
Engine:  8 of 10
Status:  Production Ready
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_FIFTY = Decimal("50")
_DP6 = Decimal("0.000001")
_DP2 = Decimal("0.01")

def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Divide with zero-guard."""
    if denominator == _ZERO:
        return _ZERO
    return (numerator / denominator).quantize(_DP6, rounding=ROUND_HALF_UP)

def _quantise(value: Decimal, precision: Decimal = _DP6) -> Decimal:
    """Quantise a Decimal value."""
    return value.quantize(precision, rounding=ROUND_HALF_UP)

def _today() -> date:
    """Return current UTC date (mockable for testing)."""
    return datetime.now(timezone.utc).date()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GapType(str, Enum):
    """Types of data gaps detected in portfolio completeness."""
    MISSING_SITE = "MISSING_SITE"
    MISSING_SCOPE = "MISSING_SCOPE"
    MISSING_CATEGORY = "MISSING_CATEGORY"
    STALE_DATA = "STALE_DATA"
    ESTIMATION_HEAVY = "ESTIMATION_HEAVY"

class SubmissionStatus(str, Enum):
    """Status of a site's data submission."""
    NOT_STARTED = "NOT_STARTED"
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RESUBMITTED = "RESUBMITTED"
    OVERDUE = "OVERDUE"

class EscalationLevel(int, Enum):
    """Escalation levels for overdue submissions."""
    NONE = 0
    SITE_MANAGER = 1
    REGIONAL_LEAD = 2
    PORTFOLIO_ADMIN = 3

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class SiteSubmission(BaseModel):
    """A data submission record from a single site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_id: str = Field(..., description="Site identifier")
    round_id: str = Field("", description="Collection round identifier")
    status: str = Field(
        SubmissionStatus.NOT_STARTED.value,
        description="Current submission status",
    )
    submitted_at: Optional[datetime] = Field(None, description="Submission timestamp")
    scopes_reported: List[str] = Field(
        default_factory=list, description="Scopes reported (e.g. ['SCOPE_1','SCOPE_2'])"
    )
    categories_reported: List[str] = Field(
        default_factory=list,
        description="Categories/sources reported within each scope",
    )
    measured_items: List[str] = Field(
        default_factory=list, description="Items with measured/actual data"
    )
    estimated_items: List[str] = Field(
        default_factory=list, description="Items with estimated data"
    )
    total_emissions: Decimal = Field(_ZERO, description="Total reported emissions (tCO2e)")
    floor_area_m2: Decimal = Field(_ZERO, description="Site floor area (m2)")
    evidence_count: int = Field(0, ge=0, description="Number of evidence attachments")

class CompletenessScore(BaseModel):
    """Completeness score for a single site / scope / category."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_id: str = Field(..., description="Site identifier")
    scope: str = Field(..., description="Scope (e.g. SCOPE_1, SCOPE_2)")
    category: Optional[str] = Field(None, description="Category within scope (if applicable)")
    period: str = Field("", description="Reporting period label")
    score: Decimal = Field(
        _ZERO, ge=_ZERO, le=_HUNDRED,
        description="Completeness score (0-100%)",
    )
    missing_items: List[str] = Field(
        default_factory=list, description="Items still missing"
    )
    estimated_items: List[str] = Field(
        default_factory=list, description="Items that are estimated rather than measured"
    )
    measured_items: List[str] = Field(
        default_factory=list, description="Items with measured/actual data"
    )

class SubmissionTracker(BaseModel):
    """Tracks a site's submission against a collection deadline."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_id: str = Field(..., description="Site identifier")
    round_id: str = Field(..., description="Collection round identifier")
    status: str = Field(
        SubmissionStatus.NOT_STARTED.value, description="Submission status"
    )
    deadline: date = Field(..., description="Submission deadline")
    days_remaining: int = Field(0, description="Days until deadline (negative = overdue)")
    reminder_sent: bool = Field(False, description="Whether a reminder has been sent")
    escalation_level: int = Field(
        0, ge=0, le=3, description="Current escalation level (0=none, 3=portfolio admin)"
    )

class GapDetection(BaseModel):
    """A detected data gap in the portfolio."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    gap_id: str = Field(default_factory=_new_uuid, description="Gap identifier")
    gap_type: str = Field(..., description="Type of gap detected")
    site_id: Optional[str] = Field(None, description="Affected site (if site-level gap)")
    scope: Optional[str] = Field(None, description="Affected scope (if scope-level gap)")
    category: Optional[str] = Field(None, description="Affected category (if category-level)")
    details: str = Field("", description="Human-readable gap description")
    estimated_impact: Decimal = Field(
        _ZERO, description="Estimated emission impact of the gap (tCO2e)"
    )

    @field_validator("gap_type")
    @classmethod
    def validate_gap_type(cls, v: str) -> str:
        """Validate gap type."""
        allowed = {e.value for e in GapType}
        if v.upper() not in allowed:
            raise ValueError(f"gap_type must be one of {sorted(allowed)}, got '{v}'")
        return v.upper()

class CompletionResult(BaseModel):
    """Overall portfolio completeness assessment."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    round_id: str = Field("", description="Collection round identifier")
    overall_completeness: Decimal = Field(
        _ZERO, ge=_ZERO, le=_HUNDRED,
        description="Overall portfolio completeness score (0-100%)",
    )
    sites_reporting_pct: Decimal = Field(
        _ZERO, description="Percentage of sites that have submitted data"
    )
    emissions_covered_pct: Decimal = Field(
        _ZERO, description="Percentage of portfolio emissions covered by submissions"
    )
    floor_area_covered_pct: Decimal = Field(
        _ZERO, description="Percentage of portfolio floor area covered"
    )
    gaps: List[GapDetection] = Field(default_factory=list, description="Detected gaps")
    overdue_sites: List[str] = Field(
        default_factory=list, description="Site IDs with overdue submissions"
    )
    created_at: datetime = Field(default_factory=utcnow, description="Assessment timestamp")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.result_id}|{self.round_id}|{self.overall_completeness}|"
                f"{self.sites_reporting_pct}|{self.emissions_covered_pct}|"
                f"{self.floor_area_covered_pct}|{len(self.gaps)}|"
                f"{len(self.overdue_sites)}"
            )
            self.provenance_hash = _compute_hash(payload)

class CollectionRound(BaseModel):
    """Definition of a data collection round."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    round_id: str = Field(default_factory=_new_uuid, description="Round identifier")
    period: str = Field("", description="Reporting period label (e.g. '2025')")
    deadline: date = Field(..., description="Submission deadline")
    required_scopes: List[str] = Field(
        default_factory=lambda: ["SCOPE_1", "SCOPE_2"],
        description="Scopes required in this round",
    )
    required_categories: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Required categories per scope (scope -> [categories])",
    )
    target_site_ids: List[str] = Field(
        default_factory=list, description="Sites targeted for this round"
    )
    reminder_days: List[int] = Field(
        default_factory=lambda: [14, 7, 3, 1],
        description="Days before deadline to send reminders",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SiteCompletionEngine:
    """
    Tracks data collection completeness across a multi-site portfolio.

    Provides:
        - Per-site completeness scoring
        - Submission deadline tracking with escalation
        - Gap detection at site, scope, and category level
        - Aggregate coverage metrics (sites, emissions, floor area)
        - Reminder and escalation management

    All calculations use Decimal arithmetic.  Every result carries a
    SHA-256 provenance hash.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        escalation_days: Optional[Dict[int, int]] = None,
        rounding_precision: Decimal = _DP6,
    ) -> None:
        """
        Initialise the SiteCompletionEngine.

        Args:
            escalation_days: Mapping of escalation level -> days overdue threshold.
                             Defaults to {1: 0, 2: 7, 3: 14}.
            rounding_precision: Decimal quantisation precision.
        """
        self._precision = rounding_precision
        self._escalation_days = escalation_days or {1: 0, 2: 7, 3: 14}
        logger.info(
            "SiteCompletionEngine v%s initialised (escalation=%s)",
            _MODULE_VERSION, self._escalation_days,
        )

    # ----------------------------------------------- assess completeness
    def assess_completeness(
        self,
        collection_round: CollectionRound,
        submissions: List[SiteSubmission],
        total_sites: int,
        total_floor_area: Decimal = _ZERO,
    ) -> CompletionResult:
        """
        Assess overall portfolio completeness for a collection round.

        Steps:
            1. Calculate sites reporting percentage
            2. Calculate emissions and floor area coverage
            3. Detect gaps (missing sites, scopes, categories)
            4. Identify overdue sites
            5. Compute overall completeness score

        Args:
            collection_round: The collection round definition.
            submissions: List of site submissions received.
            total_sites: Total number of sites in the portfolio.
            total_floor_area: Total portfolio floor area (m2).

        Returns:
            CompletionResult with scores, gaps, and provenance hash.

        Raises:
            ValueError: If total_sites is zero or negative.
        """
        logger.info(
            "Assessing completeness: round=%s submissions=%d total_sites=%d",
            collection_round.round_id, len(submissions), total_sites,
        )
        if total_sites <= 0:
            raise ValueError("total_sites must be positive")

        # Track submissions by site
        submission_map: Dict[str, SiteSubmission] = {}
        for sub in submissions:
            submission_map[sub.site_id] = sub

        # Sites reporting
        reporting_sites = [
            s for s in submissions
            if s.status not in (
                SubmissionStatus.NOT_STARTED.value,
                SubmissionStatus.OVERDUE.value,
            )
        ]
        sites_reporting_pct = _quantise(
            Decimal(str(len(reporting_sites))) / Decimal(str(total_sites)) * _HUNDRED,
            self._precision,
        )

        # Emissions coverage
        total_reported_emissions = sum(s.total_emissions for s in reporting_sites)
        total_portfolio_emissions = sum(s.total_emissions for s in submissions)
        emissions_covered_pct = (
            _quantise(
                _safe_divide(total_reported_emissions, total_portfolio_emissions) * _HUNDRED,
                self._precision,
            )
            if total_portfolio_emissions > _ZERO
            else _ZERO
        )

        # Floor area coverage
        total_reported_area = sum(s.floor_area_m2 for s in reporting_sites)
        floor_area_covered_pct = (
            _quantise(
                _safe_divide(total_reported_area, total_floor_area) * _HUNDRED,
                self._precision,
            )
            if total_floor_area > _ZERO
            else _ZERO
        )

        # Detect gaps
        gaps = self.detect_gaps(
            collection_round,
            submissions,
            collection_round.target_site_ids or [],
            collection_round.required_scopes,
        )

        # Overdue sites
        today = _today()
        overdue_sites: List[str] = []
        for target_id in collection_round.target_site_ids:
            sub = submission_map.get(target_id)
            if sub is None or sub.status in (
                SubmissionStatus.NOT_STARTED.value,
                SubmissionStatus.OVERDUE.value,
            ):
                if today > collection_round.deadline:
                    overdue_sites.append(target_id)

        # Overall completeness = average of the three coverage metrics
        overall = _quantise(
            (sites_reporting_pct + emissions_covered_pct + floor_area_covered_pct)
            / Decimal("3"),
            self._precision,
        )

        result = CompletionResult(
            round_id=collection_round.round_id,
            overall_completeness=min(overall, _HUNDRED),
            sites_reporting_pct=sites_reporting_pct,
            emissions_covered_pct=emissions_covered_pct,
            floor_area_covered_pct=floor_area_covered_pct,
            gaps=gaps,
            overdue_sites=sorted(overdue_sites),
        )

        logger.info(
            "Completeness assessed: overall=%s%% sites=%s%% emissions=%s%% area=%s%% gaps=%d",
            overall, sites_reporting_pct, emissions_covered_pct,
            floor_area_covered_pct, len(gaps),
        )
        return result

    # ----------------------------------------------- site completeness
    def score_site_completeness(
        self,
        submission: SiteSubmission,
        required_categories: Dict[str, List[str]],
    ) -> List[CompletenessScore]:
        """
        Score a single site's completeness against required categories.

        For each scope:
            required = set of required categories
            reported = set of categories reported
            measured = items with measured data
            estimated = items with estimated data
            missing = required - reported
            score = (len(reported) / len(required)) * 100

        Args:
            submission: The site's submission data.
            required_categories: Required categories per scope.

        Returns:
            List of CompletenessScore, one per scope.
        """
        logger.info(
            "Scoring site completeness: site=%s scopes=%s",
            submission.site_id, list(required_categories.keys()),
        )
        scores: List[CompletenessScore] = []

        for scope, required_cats in required_categories.items():
            if not required_cats:
                scores.append(CompletenessScore(
                    site_id=submission.site_id,
                    scope=scope,
                    period=submission.round_id,
                    score=_HUNDRED,
                ))
                continue

            required_set = set(required_cats)

            # Determine which categories were reported for this scope
            reported_set = set()
            for cat in submission.categories_reported:
                # Categories are assumed to be prefixed: "SCOPE_1::stationary_combustion"
                if cat.startswith(f"{scope}::"):
                    reported_set.add(cat.split("::")[-1])
                elif cat in required_set:
                    reported_set.add(cat)

            missing = sorted(required_set - reported_set)
            reported = sorted(required_set & reported_set)

            # Split reported into measured vs estimated
            measured = [
                item for item in reported
                if f"{scope}::{item}" in submission.measured_items
                or item in submission.measured_items
            ]
            estimated = [
                item for item in reported
                if item not in measured and (
                    f"{scope}::{item}" in submission.estimated_items
                    or item in submission.estimated_items
                )
            ]
            # Items reported but not in measured or estimated lists
            unclassified = [
                item for item in reported
                if item not in measured and item not in estimated
            ]
            measured.extend(unclassified)

            n_required = Decimal(str(len(required_set)))
            n_reported = Decimal(str(len(reported)))
            score = _quantise(
                _safe_divide(n_reported, n_required) * _HUNDRED, self._precision
            )

            scores.append(CompletenessScore(
                site_id=submission.site_id,
                scope=scope,
                period=submission.round_id,
                score=score,
                missing_items=missing,
                estimated_items=estimated,
                measured_items=measured,
            ))

        return scores

    # ----------------------------------------------- submission tracking
    def track_submissions(
        self,
        collection_round: CollectionRound,
        submissions: List[SiteSubmission],
    ) -> List[SubmissionTracker]:
        """
        Create submission trackers for all target sites in a round.

        For each target site, calculates days remaining to deadline
        and determines whether reminders are due.

        Args:
            collection_round: Collection round definition.
            submissions: List of received submissions.

        Returns:
            List of SubmissionTracker for all target sites.
        """
        logger.info(
            "Tracking submissions: round=%s targets=%d received=%d",
            collection_round.round_id, len(collection_round.target_site_ids),
            len(submissions),
        )
        today = _today()
        submission_map: Dict[str, SiteSubmission] = {
            s.site_id: s for s in submissions
        }

        trackers: List[SubmissionTracker] = []
        for site_id in sorted(collection_round.target_site_ids):
            sub = submission_map.get(site_id)
            days_remaining = (collection_round.deadline - today).days

            if sub is None:
                status = (
                    SubmissionStatus.OVERDUE.value
                    if days_remaining < 0
                    else SubmissionStatus.NOT_STARTED.value
                )
            else:
                status = sub.status
                if status == SubmissionStatus.NOT_STARTED.value and days_remaining < 0:
                    status = SubmissionStatus.OVERDUE.value

            # Determine if a reminder is due
            reminder_sent = False
            for reminder_day in collection_round.reminder_days:
                if days_remaining == reminder_day:
                    reminder_sent = True
                    break

            # Determine escalation level
            escalation_level = 0
            if days_remaining < 0 and status in (
                SubmissionStatus.NOT_STARTED.value,
                SubmissionStatus.OVERDUE.value,
                SubmissionStatus.DRAFT.value,
            ):
                overdue_days = abs(days_remaining)
                for level in sorted(self._escalation_days.keys(), reverse=True):
                    if overdue_days >= self._escalation_days[level]:
                        escalation_level = level
                        break

            tracker = SubmissionTracker(
                site_id=site_id,
                round_id=collection_round.round_id,
                status=status,
                deadline=collection_round.deadline,
                days_remaining=days_remaining,
                reminder_sent=reminder_sent,
                escalation_level=escalation_level,
            )
            trackers.append(tracker)

        return trackers

    # ----------------------------------------------- gap detection
    def detect_gaps(
        self,
        collection_round: CollectionRound,
        submissions: List[SiteSubmission],
        all_site_ids: List[str],
        required_scopes: List[str],
    ) -> List[GapDetection]:
        """
        Detect data gaps across the portfolio.

        Gap Types:
            MISSING_SITE:       Registered site with no submission
            MISSING_SCOPE:      Submission received but required scope absent
            MISSING_CATEGORY:   Scope reported but required category absent
            ESTIMATION_HEAVY:   >50% of items are estimated rather than measured

        Args:
            collection_round: Collection round definition.
            submissions: List of received submissions.
            all_site_ids: All registered site IDs.
            required_scopes: Scopes required in this round.

        Returns:
            List of GapDetection objects.
        """
        logger.info(
            "Detecting gaps: sites=%d submissions=%d scopes=%s",
            len(all_site_ids), len(submissions), required_scopes,
        )
        gaps: List[GapDetection] = []
        submission_map: Dict[str, SiteSubmission] = {
            s.site_id: s for s in submissions
        }

        # 1. MISSING_SITE gaps
        for site_id in sorted(all_site_ids):
            sub = submission_map.get(site_id)
            if sub is None or sub.status in (
                SubmissionStatus.NOT_STARTED.value,
                SubmissionStatus.OVERDUE.value,
            ):
                gap = GapDetection(
                    gap_type=GapType.MISSING_SITE.value,
                    site_id=site_id,
                    details=f"Site {site_id} has no valid submission for round {collection_round.round_id}",
                )
                gaps.append(gap)
                continue

            # 2. MISSING_SCOPE gaps
            reported_scopes = set(sub.scopes_reported)
            for scope in required_scopes:
                if scope not in reported_scopes:
                    gap = GapDetection(
                        gap_type=GapType.MISSING_SCOPE.value,
                        site_id=site_id,
                        scope=scope,
                        details=f"Site {site_id} missing {scope} data",
                    )
                    gaps.append(gap)
                    continue

                # 3. MISSING_CATEGORY gaps
                required_cats = collection_round.required_categories.get(scope, [])
                for cat in required_cats:
                    full_cat = f"{scope}::{cat}"
                    if full_cat not in sub.categories_reported and cat not in sub.categories_reported:
                        gap = GapDetection(
                            gap_type=GapType.MISSING_CATEGORY.value,
                            site_id=site_id,
                            scope=scope,
                            category=cat,
                            details=f"Site {site_id} missing category {cat} under {scope}",
                        )
                        gaps.append(gap)

            # 4. ESTIMATION_HEAVY check
            total_items = len(sub.measured_items) + len(sub.estimated_items)
            if total_items > 0:
                estimated_pct = _safe_divide(
                    Decimal(str(len(sub.estimated_items))),
                    Decimal(str(total_items)),
                ) * _HUNDRED
                if estimated_pct > _FIFTY:
                    gap = GapDetection(
                        gap_type=GapType.ESTIMATION_HEAVY.value,
                        site_id=site_id,
                        details=(
                            f"Site {site_id} has {estimated_pct}% estimated data "
                            f"({len(sub.estimated_items)}/{total_items} items)"
                        ),
                    )
                    gaps.append(gap)

        logger.info("Gap detection complete: %d gaps found", len(gaps))
        return gaps

    # ----------------------------------------------- reminders
    def get_reminders_due(
        self,
        trackers: List[SubmissionTracker],
        days_before: Optional[List[int]] = None,
    ) -> List[SubmissionTracker]:
        """
        Filter trackers to those requiring a reminder notification.

        A reminder is due when:
            - days_remaining matches one of the configured reminder days
            - status is NOT_STARTED, DRAFT, or REJECTED

        Args:
            trackers: List of SubmissionTracker.
            days_before: Override reminder day thresholds. If None uses tracker's reminder_sent.

        Returns:
            List of trackers that need a reminder sent.
        """
        logger.info("Checking reminders due: %d trackers", len(trackers))
        remindable_statuses = {
            SubmissionStatus.NOT_STARTED.value,
            SubmissionStatus.DRAFT.value,
            SubmissionStatus.REJECTED.value,
        }

        reminders: List[SubmissionTracker] = []
        for tracker in trackers:
            if tracker.status not in remindable_statuses:
                continue

            if days_before is not None:
                if tracker.days_remaining in days_before:
                    reminders.append(tracker)
            elif tracker.reminder_sent:
                reminders.append(tracker)

        logger.info("Reminders due: %d trackers", len(reminders))
        return reminders

    # ----------------------------------------------- escalation
    def escalate_overdue(
        self,
        trackers: List[SubmissionTracker],
    ) -> List[SubmissionTracker]:
        """
        Escalate overdue submissions based on days overdue.

        Updates escalation_level and status for overdue trackers.
        Does NOT modify submitted/approved/under_review trackers.

        Args:
            trackers: List of SubmissionTracker to evaluate.

        Returns:
            List of trackers that were escalated (subset of input).
        """
        logger.info("Escalating overdue trackers: %d total", len(trackers))
        escalated: List[SubmissionTracker] = []

        non_escalatable = {
            SubmissionStatus.SUBMITTED.value,
            SubmissionStatus.UNDER_REVIEW.value,
            SubmissionStatus.APPROVED.value,
            SubmissionStatus.RESUBMITTED.value,
        }

        for tracker in trackers:
            if tracker.status in non_escalatable:
                continue
            if tracker.days_remaining >= 0:
                continue

            overdue_days = abs(tracker.days_remaining)
            new_level = tracker.escalation_level

            for level in sorted(self._escalation_days.keys(), reverse=True):
                if overdue_days >= self._escalation_days[level]:
                    new_level = max(new_level, level)
                    break

            if new_level > tracker.escalation_level:
                tracker.escalation_level = new_level
                tracker.status = SubmissionStatus.OVERDUE.value
                escalated.append(tracker)
                logger.warning(
                    "Escalated: site=%s round=%s level=%d overdue=%d days",
                    tracker.site_id, tracker.round_id, new_level, overdue_days,
                )

        logger.info("Escalation complete: %d trackers escalated", len(escalated))
        return escalated

    # ----------------------------------------------- coverage metrics
    def get_coverage_metrics(
        self,
        submissions: List[SiteSubmission],
        total_sites: int,
        total_emissions: Decimal,
        total_floor_area: Decimal,
    ) -> Dict[str, Any]:
        """
        Calculate portfolio coverage metrics.

        Metrics:
            sites_reporting_pct:    % of sites with valid submissions
            emissions_covered_pct:  % of total emissions covered
            floor_area_covered_pct: % of total floor area covered
            measured_pct:           % of data points that are measured (not estimated)
            avg_completeness:       Average site completeness score

        Args:
            submissions: List of site submissions.
            total_sites: Total portfolio sites.
            total_emissions: Total portfolio emissions estimate (tCO2e).
            total_floor_area: Total portfolio floor area (m2).

        Returns:
            Dict with coverage metric values and provenance hash.
        """
        logger.info(
            "Calculating coverage: submissions=%d total_sites=%d",
            len(submissions), total_sites,
        )
        valid_statuses = {
            SubmissionStatus.SUBMITTED.value,
            SubmissionStatus.UNDER_REVIEW.value,
            SubmissionStatus.APPROVED.value,
            SubmissionStatus.RESUBMITTED.value,
        }

        valid_submissions = [s for s in submissions if s.status in valid_statuses]
        n_valid = len(valid_submissions)

        sites_pct = _quantise(
            Decimal(str(n_valid)) / Decimal(str(max(total_sites, 1))) * _HUNDRED,
            self._precision,
        )

        reported_emissions = sum(s.total_emissions for s in valid_submissions)
        emissions_pct = (
            _quantise(
                _safe_divide(reported_emissions, total_emissions) * _HUNDRED,
                self._precision,
            )
            if total_emissions > _ZERO
            else _ZERO
        )

        reported_area = sum(s.floor_area_m2 for s in valid_submissions)
        area_pct = (
            _quantise(
                _safe_divide(reported_area, total_floor_area) * _HUNDRED,
                self._precision,
            )
            if total_floor_area > _ZERO
            else _ZERO
        )

        total_measured = sum(len(s.measured_items) for s in valid_submissions)
        total_estimated = sum(len(s.estimated_items) for s in valid_submissions)
        total_data_points = total_measured + total_estimated
        measured_pct = (
            _quantise(
                Decimal(str(total_measured)) / Decimal(str(total_data_points)) * _HUNDRED,
                self._precision,
            )
            if total_data_points > 0
            else _ZERO
        )

        payload = (
            f"coverage|{n_valid}|{total_sites}|{reported_emissions}|"
            f"{total_emissions}|{reported_area}|{total_floor_area}"
        )
        provenance = _compute_hash(payload)

        return {
            "sites_reporting_count": n_valid,
            "total_sites": total_sites,
            "sites_reporting_pct": sites_pct,
            "reported_emissions_tco2e": reported_emissions,
            "total_emissions_tco2e": total_emissions,
            "emissions_covered_pct": emissions_pct,
            "reported_floor_area_m2": reported_area,
            "total_floor_area_m2": total_floor_area,
            "floor_area_covered_pct": area_pct,
            "total_measured_items": total_measured,
            "total_estimated_items": total_estimated,
            "measured_pct": measured_pct,
            "provenance_hash": provenance,
        }

    # ----------------------------------------------- gap impact estimation
    def estimate_gap_impact(
        self,
        gaps: List[GapDetection],
        avg_site_emissions: Decimal,
    ) -> Decimal:
        """
        Estimate the emission impact of detected gaps.

        Estimation rules:
            MISSING_SITE:       avg_site_emissions per site
            MISSING_SCOPE:      avg_site_emissions * 0.5 per scope
            MISSING_CATEGORY:   avg_site_emissions * 0.1 per category
            ESTIMATION_HEAVY:   0 (data exists but quality is low)
            STALE_DATA:         avg_site_emissions * 0.2

        Args:
            gaps: List of detected gaps.
            avg_site_emissions: Average site emissions for estimation (tCO2e).

        Returns:
            Total estimated impact (tCO2e).
        """
        logger.info("Estimating gap impact: %d gaps, avg=%s", len(gaps), avg_site_emissions)

        impact_factors: Dict[str, Decimal] = {
            GapType.MISSING_SITE.value: _ONE,
            GapType.MISSING_SCOPE.value: Decimal("0.5"),
            GapType.MISSING_CATEGORY.value: Decimal("0.1"),
            GapType.STALE_DATA.value: Decimal("0.2"),
            GapType.ESTIMATION_HEAVY.value: _ZERO,
        }

        total_impact = _ZERO
        for gap in gaps:
            factor = impact_factors.get(gap.gap_type, _ZERO)
            gap_impact = _quantise(avg_site_emissions * factor, self._precision)
            gap.estimated_impact = gap_impact
            total_impact += gap_impact

        total_impact = _quantise(total_impact, self._precision)
        logger.info("Estimated gap impact: %s tCO2e", total_impact)
        return total_impact

    # ----------------------------------------------- completeness summary
    def get_completeness_summary(
        self,
        scores: List[CompletenessScore],
    ) -> Dict[str, Any]:
        """
        Aggregate completeness scores into a summary.

        Args:
            scores: List of CompletenessScore from multiple sites/scopes.

        Returns:
            Summary dict with aggregated metrics and per-scope breakdown.
        """
        if not scores:
            return {
                "overall_score": _ZERO,
                "by_scope": {},
                "sites_at_100_pct": 0,
                "sites_below_50_pct": 0,
                "total_missing_items": 0,
                "total_estimated_items": 0,
            }

        # Per-scope aggregation
        by_scope: Dict[str, List[Decimal]] = {}
        for s in scores:
            by_scope.setdefault(s.scope, []).append(s.score)

        scope_averages: Dict[str, Decimal] = {}
        for scope, scope_scores in by_scope.items():
            avg = _quantise(
                sum(scope_scores) / Decimal(str(len(scope_scores))),
                self._precision,
            )
            scope_averages[scope] = avg

        # Overall average across all scores
        all_scores_vals = [s.score for s in scores]
        overall = _quantise(
            sum(all_scores_vals) / Decimal(str(len(all_scores_vals))),
            self._precision,
        )

        # Per-site aggregation for thresholds
        site_scores: Dict[str, List[Decimal]] = {}
        for s in scores:
            site_scores.setdefault(s.site_id, []).append(s.score)

        sites_at_100 = 0
        sites_below_50 = 0
        for site_id, s_scores in site_scores.items():
            site_avg = sum(s_scores) / Decimal(str(len(s_scores)))
            if site_avg >= _HUNDRED:
                sites_at_100 += 1
            if site_avg < _FIFTY:
                sites_below_50 += 1

        total_missing = sum(len(s.missing_items) for s in scores)
        total_estimated = sum(len(s.estimated_items) for s in scores)

        payload = f"completeness_summary|{overall}|{len(scores)}"
        provenance = _compute_hash(payload)

        return {
            "overall_score": overall,
            "by_scope": scope_averages,
            "sites_at_100_pct": sites_at_100,
            "sites_below_50_pct": sites_below_50,
            "total_missing_items": total_missing,
            "total_estimated_items": total_estimated,
            "total_scores_evaluated": len(scores),
            "provenance_hash": provenance,
        }

    # ----------------------------------------------- historical tracking
    def track_completeness_history(
        self,
        historical_results: List[CompletionResult],
    ) -> Dict[str, Any]:
        """
        Track completeness over time from historical results.

        Args:
            historical_results: List of CompletionResult from past rounds.

        Returns:
            Trend dict with period-over-period completeness and direction.
        """
        if not historical_results:
            return {
                "periods": [],
                "direction": "STABLE",
                "change_pct": _ZERO,
            }

        sorted_results = sorted(historical_results, key=lambda r: r.round_id)
        periods: List[Dict[str, Any]] = []
        for r in sorted_results:
            periods.append({
                "round_id": r.round_id,
                "overall_completeness": r.overall_completeness,
                "sites_reporting_pct": r.sites_reporting_pct,
                "gap_count": len(r.gaps),
                "overdue_count": len(r.overdue_sites),
            })

        first_val = sorted_results[0].overall_completeness
        last_val = sorted_results[-1].overall_completeness

        if first_val == _ZERO:
            change_pct = _HUNDRED if last_val > _ZERO else _ZERO
        else:
            change_pct = _quantise(
                (last_val - first_val) / first_val * _HUNDRED,
                self._precision,
            )

        if last_val > first_val:
            direction = "IMPROVING"
        elif last_val < first_val:
            direction = "WORSENING"
        else:
            direction = "STABLE"

        return {
            "periods": periods,
            "direction": direction,
            "change_pct": change_pct,
            "first_period": sorted_results[0].round_id,
            "last_period": sorted_results[-1].round_id,
        }

# ---------------------------------------------------------------------------
# Pydantic v2 model rebuild (required with `from __future__ import annotations`)
# ---------------------------------------------------------------------------

SiteSubmission.model_rebuild()
CompletenessScore.model_rebuild()
SubmissionTracker.model_rebuild()
GapDetection.model_rebuild()
CompletionResult.model_rebuild()
CollectionRound.model_rebuild()
