"""
CDP Verification Tracker -- Third-Party Verification Management

This module implements per-scope verification status tracking for CDP
Climate Change disclosures.  Third-party verification is critical for CDP
scoring -- A-level requires 100% Scope 1+2 verification (AREQ03) and >=70%
coverage on at least one Scope 3 category (AREQ04).

Key capabilities:
  - Per-scope verification record management (Scope 1, 2, 3, 3 by category)
  - Coverage percentage calculation and tracking
  - Verifier details and credential management
  - Assurance level tracking (limited / reasonable)
  - A-level verification requirement assessment (AREQ03, AREQ04)
  - Verification gap identification
  - Verification evidence linking

Verification standards supported:
  - ISO 14064-3:2019
  - ISAE 3000 (Revised)
  - ISAE 3410
  - AA1000AS v3

Example:
    >>> tracker = VerificationTracker(config)
    >>> tracker.add_verification("q-1", "org-1", scope="scope_1",
    ...     assurance_level=VerificationAssurance.REASONABLE, coverage_pct=100)
    >>> status = tracker.get_verification_status("q-1", "org-1")
    >>> status["a_requirements"]["AREQ03"]["met"]
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import CDPAppConfig, VerificationAssurance
from .models import (
    VerificationRecord,
    _new_id,
    _now,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recognized verification standards
# ---------------------------------------------------------------------------

VERIFICATION_STANDARDS: Dict[str, Dict[str, str]] = {
    "ISO14064-3": {
        "name": "ISO 14064-3:2019",
        "description": "Specification with guidance for the verification and validation of greenhouse gas statements",
        "body": "ISO/TC 207",
    },
    "ISAE3000": {
        "name": "ISAE 3000 (Revised)",
        "description": "Assurance engagements other than audits or reviews of historical financial information",
        "body": "IAASB",
    },
    "ISAE3410": {
        "name": "ISAE 3410",
        "description": "Assurance engagements on greenhouse gas statements",
        "body": "IAASB",
    },
    "AA1000AS": {
        "name": "AA1000AS v3",
        "description": "AccountAbility Assurance Standard",
        "body": "AccountAbility",
    },
}


# ---------------------------------------------------------------------------
# Scopes that can be verified independently
# ---------------------------------------------------------------------------

ALL_SCOPES = [
    "scope_1",
    "scope_2",
    "scope_3",
]

SCOPE_3_CATEGORIES = [f"scope_3_cat_{i}" for i in range(1, 16)]


class VerificationTracker:
    """
    CDP Verification Tracker -- manages third-party verification records.

    Tracks verification per scope, calculates coverage, assesses A-level
    eligibility, and identifies verification gaps.

    Attributes:
        config: Application configuration.
        _records: Verification record store keyed by record ID.
        _by_questionnaire: Questionnaire ID -> list of record IDs.

    Example:
        >>> tracker = VerificationTracker(config)
        >>> rec = tracker.add_verification("q-1", "org-1", "scope_1",
        ...     assurance_level=VerificationAssurance.REASONABLE, coverage_pct=100)
        >>> print(rec.id)
    """

    def __init__(self, config: CDPAppConfig) -> None:
        """Initialize the Verification Tracker."""
        self.config = config
        self._records: Dict[str, VerificationRecord] = {}
        self._by_questionnaire: Dict[str, List[str]] = {}
        logger.info("VerificationTracker initialized")

    # ------------------------------------------------------------------
    # Add / Update / Remove Verification Records
    # ------------------------------------------------------------------

    def add_verification(
        self,
        questionnaire_id: str,
        org_id: str,
        scope: str,
        assurance_level: VerificationAssurance = VerificationAssurance.LIMITED,
        coverage_pct: Decimal = Decimal("100"),
        verifier_name: Optional[str] = None,
        verifier_organization: Optional[str] = None,
        verifier_accreditation: Optional[str] = None,
        verification_standard: Optional[str] = None,
        statement_date: Optional[date] = None,
        statement_file_id: Optional[str] = None,
        emissions_verified_tco2e: Optional[Decimal] = None,
        year: int = 2026,
    ) -> VerificationRecord:
        """
        Add a new verification record for a specific scope.

        Args:
            questionnaire_id: Associated questionnaire ID.
            org_id: Organization ID.
            scope: Scope verified (scope_1, scope_2, scope_3, scope_3_cat_N).
            assurance_level: Limited or reasonable assurance.
            coverage_pct: Percentage of emissions in scope that are verified.
            verifier_name: Name of the lead verifier.
            verifier_organization: Verification firm name.
            verifier_accreditation: Verifier accreditation body/number.
            verification_standard: Standard used (e.g. ISO14064-3).
            statement_date: Date of verification statement.
            statement_file_id: Evidence attachment ID.
            emissions_verified_tco2e: Amount of emissions verified.
            year: Reporting year.

        Returns:
            Created VerificationRecord.
        """
        self._validate_scope(scope)

        record = VerificationRecord(
            questionnaire_id=questionnaire_id,
            org_id=org_id,
            scope=scope,
            assurance_level=assurance_level,
            coverage_pct=min(coverage_pct, Decimal("100")),
            verifier_name=verifier_name,
            verifier_organization=verifier_organization,
            verifier_accreditation=verifier_accreditation,
            verification_standard=verification_standard,
            statement_date=statement_date,
            statement_file_id=statement_file_id,
            emissions_verified_tco2e=emissions_verified_tco2e,
            year=year,
        )

        self._records[record.id] = record
        if questionnaire_id not in self._by_questionnaire:
            self._by_questionnaire[questionnaire_id] = []
        self._by_questionnaire[questionnaire_id].append(record.id)

        logger.info(
            "Added %s verification for %s: %s assurance, %.1f%% coverage, verifier=%s",
            scope, questionnaire_id, assurance_level.value,
            float(coverage_pct), verifier_organization or "N/A",
        )
        return record

    def update_verification(
        self,
        record_id: str,
        **updates: Any,
    ) -> Optional[VerificationRecord]:
        """
        Update an existing verification record.

        Args:
            record_id: Record ID to update.
            **updates: Fields to update (assurance_level, coverage_pct, etc.).

        Returns:
            Updated record or None if not found.
        """
        record = self._records.get(record_id)
        if not record:
            return None

        allowed_fields = {
            "assurance_level", "coverage_pct", "verifier_name",
            "verifier_organization", "verifier_accreditation",
            "verification_standard", "statement_date",
            "statement_file_id", "emissions_verified_tco2e",
        }

        for key, value in updates.items():
            if key in allowed_fields and hasattr(record, key):
                setattr(record, key, value)

        record.updated_at = _now()
        logger.info("Updated verification record %s", record_id)
        return record

    def remove_verification(self, record_id: str) -> bool:
        """Remove a verification record."""
        record = self._records.pop(record_id, None)
        if not record:
            return False

        qid = record.questionnaire_id
        if qid in self._by_questionnaire:
            ids = self._by_questionnaire[qid]
            if record_id in ids:
                ids.remove(record_id)

        logger.info(
            "Removed verification record %s for scope %s",
            record_id, record.scope,
        )
        return True

    # ------------------------------------------------------------------
    # Query Verification Records
    # ------------------------------------------------------------------

    def get_record(self, record_id: str) -> Optional[VerificationRecord]:
        """Get a single verification record by ID."""
        return self._records.get(record_id)

    def get_records_for_questionnaire(
        self,
        questionnaire_id: str,
    ) -> List[VerificationRecord]:
        """Get all verification records for a questionnaire."""
        record_ids = self._by_questionnaire.get(questionnaire_id, [])
        return [
            self._records[rid]
            for rid in record_ids
            if rid in self._records
        ]

    def get_records_by_scope(
        self,
        questionnaire_id: str,
        scope: str,
    ) -> List[VerificationRecord]:
        """Get verification records for a specific scope."""
        records = self.get_records_for_questionnaire(questionnaire_id)
        return [r for r in records if r.scope == scope]

    # ------------------------------------------------------------------
    # Coverage Calculation
    # ------------------------------------------------------------------

    def get_scope_coverage(
        self,
        questionnaire_id: str,
        scope: str,
    ) -> Dict[str, Any]:
        """
        Calculate verification coverage for a given scope.

        Returns the highest assurance level and maximum coverage percentage
        among all records for the scope.

        Args:
            questionnaire_id: Questionnaire ID.
            scope: Scope to check.

        Returns:
            Coverage information.
        """
        records = self.get_records_by_scope(questionnaire_id, scope)

        if not records:
            return {
                "scope": scope,
                "verified": False,
                "assurance_level": VerificationAssurance.NOT_VERIFIED.value,
                "coverage_pct": 0.0,
                "verifier_count": 0,
                "records": [],
            }

        # Highest assurance level
        assurance_order = {
            VerificationAssurance.NOT_VERIFIED: 0,
            VerificationAssurance.LIMITED: 1,
            VerificationAssurance.REASONABLE: 2,
        }
        best_assurance = max(records, key=lambda r: assurance_order.get(r.assurance_level, 0))

        # Maximum coverage (take highest single record or sum if non-overlapping)
        max_coverage = max(float(r.coverage_pct) for r in records)

        return {
            "scope": scope,
            "verified": True,
            "assurance_level": best_assurance.assurance_level.value,
            "coverage_pct": max_coverage,
            "verifier_count": len(records),
            "records": [
                {
                    "id": r.id,
                    "verifier": r.verifier_organization or r.verifier_name or "Unknown",
                    "assurance_level": r.assurance_level.value,
                    "coverage_pct": float(r.coverage_pct),
                    "standard": r.verification_standard or "Not specified",
                    "statement_date": r.statement_date.isoformat() if r.statement_date else None,
                }
                for r in records
            ],
        }

    def get_all_coverage(
        self,
        questionnaire_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate verification coverage across all scopes.

        Returns:
            Coverage summary for Scope 1, 2, 3, and individual Scope 3 categories.
        """
        result: Dict[str, Any] = {
            "questionnaire_id": questionnaire_id,
            "scopes": {},
            "overall_verified": False,
            "scope_3_categories_verified": [],
        }

        for scope in ALL_SCOPES:
            coverage = self.get_scope_coverage(questionnaire_id, scope)
            result["scopes"][scope] = coverage

        # Check individual Scope 3 categories
        for cat_scope in SCOPE_3_CATEGORIES:
            coverage = self.get_scope_coverage(questionnaire_id, cat_scope)
            if coverage["verified"]:
                result["scope_3_categories_verified"].append({
                    "category": cat_scope,
                    "coverage_pct": coverage["coverage_pct"],
                    "assurance_level": coverage["assurance_level"],
                })

        # Overall verified if Scope 1 and 2 are both verified
        s1 = result["scopes"].get("scope_1", {})
        s2 = result["scopes"].get("scope_2", {})
        result["overall_verified"] = (
            s1.get("verified", False) and s2.get("verified", False)
        )

        return result

    # ------------------------------------------------------------------
    # A-Level Requirement Assessment
    # ------------------------------------------------------------------

    def check_a_level_requirements(
        self,
        questionnaire_id: str,
    ) -> Dict[str, Any]:
        """
        Assess verification status against CDP A-level requirements.

        AREQ03: Third-party verification of 100% Scope 1 and Scope 2 emissions.
        AREQ04: Third-party verification of >= 70% of at least one Scope 3 category.

        Returns:
            A-level requirement assessment results.
        """
        coverage = self.get_all_coverage(questionnaire_id)

        # AREQ03: Scope 1+2 verification at 100% coverage
        s1 = coverage["scopes"].get("scope_1", {})
        s2 = coverage["scopes"].get("scope_2", {})

        s1_met = s1.get("verified", False) and s1.get("coverage_pct", 0) >= 100.0
        s2_met = s2.get("verified", False) and s2.get("coverage_pct", 0) >= 100.0
        areq03_met = s1_met and s2_met

        areq03_details = {
            "scope_1_verified": s1.get("verified", False),
            "scope_1_coverage_pct": s1.get("coverage_pct", 0),
            "scope_1_assurance": s1.get("assurance_level", "not_verified"),
            "scope_2_verified": s2.get("verified", False),
            "scope_2_coverage_pct": s2.get("coverage_pct", 0),
            "scope_2_assurance": s2.get("assurance_level", "not_verified"),
        }

        # AREQ04: >= 70% coverage on at least one Scope 3 category
        areq04_met = False
        qualifying_categories: List[Dict[str, Any]] = []

        for cat_info in coverage.get("scope_3_categories_verified", []):
            if cat_info.get("coverage_pct", 0) >= 70.0:
                areq04_met = True
                qualifying_categories.append(cat_info)

        # Also check aggregate scope_3
        s3 = coverage["scopes"].get("scope_3", {})
        if s3.get("verified", False) and s3.get("coverage_pct", 0) >= 70.0:
            areq04_met = True

        areq04_details = {
            "scope_3_aggregate_verified": s3.get("verified", False),
            "scope_3_aggregate_coverage_pct": s3.get("coverage_pct", 0),
            "qualifying_categories": qualifying_categories,
        }

        return {
            "questionnaire_id": questionnaire_id,
            "a_requirements": {
                "AREQ03": {
                    "name": "Scope 1+2 verification",
                    "description": "Third-party verification of 100% Scope 1 and Scope 2 emissions",
                    "met": areq03_met,
                    "details": areq03_details,
                },
                "AREQ04": {
                    "name": "Scope 3 verification",
                    "description": "Third-party verification of >= 70% of at least one Scope 3 category",
                    "met": areq04_met,
                    "details": areq04_details,
                },
            },
            "both_met": areq03_met and areq04_met,
        }

    # ------------------------------------------------------------------
    # Verification Gap Identification
    # ------------------------------------------------------------------

    def identify_verification_gaps(
        self,
        questionnaire_id: str,
    ) -> Dict[str, Any]:
        """
        Identify verification gaps that prevent A-level scoring.

        Returns:
            Verification gaps with recommended actions.
        """
        a_check = self.check_a_level_requirements(questionnaire_id)
        coverage = self.get_all_coverage(questionnaire_id)
        gaps: List[Dict[str, Any]] = []

        # Check Scope 1 gaps
        s1 = coverage["scopes"].get("scope_1", {})
        if not s1.get("verified", False):
            gaps.append({
                "scope": "scope_1",
                "gap_type": "no_verification",
                "severity": "critical",
                "current_coverage_pct": 0.0,
                "required_coverage_pct": 100.0,
                "recommendation": "Engage an accredited third-party verifier for Scope 1 emissions",
                "a_requirement": "AREQ03",
            })
        elif s1.get("coverage_pct", 0) < 100.0:
            gaps.append({
                "scope": "scope_1",
                "gap_type": "partial_coverage",
                "severity": "high",
                "current_coverage_pct": s1["coverage_pct"],
                "required_coverage_pct": 100.0,
                "recommendation": f"Extend Scope 1 verification from {s1['coverage_pct']:.0f}% to 100% coverage",
                "a_requirement": "AREQ03",
            })

        # Check Scope 2 gaps
        s2 = coverage["scopes"].get("scope_2", {})
        if not s2.get("verified", False):
            gaps.append({
                "scope": "scope_2",
                "gap_type": "no_verification",
                "severity": "critical",
                "current_coverage_pct": 0.0,
                "required_coverage_pct": 100.0,
                "recommendation": "Engage an accredited third-party verifier for Scope 2 emissions",
                "a_requirement": "AREQ03",
            })
        elif s2.get("coverage_pct", 0) < 100.0:
            gaps.append({
                "scope": "scope_2",
                "gap_type": "partial_coverage",
                "severity": "high",
                "current_coverage_pct": s2["coverage_pct"],
                "required_coverage_pct": 100.0,
                "recommendation": f"Extend Scope 2 verification from {s2['coverage_pct']:.0f}% to 100% coverage",
                "a_requirement": "AREQ03",
            })

        # Check Scope 3 gaps (AREQ04)
        areq04 = a_check["a_requirements"]["AREQ04"]
        if not areq04["met"]:
            s3_cats = coverage.get("scope_3_categories_verified", [])
            best_cat_coverage = 0.0
            best_cat = None
            for cat in s3_cats:
                if cat.get("coverage_pct", 0) > best_cat_coverage:
                    best_cat_coverage = cat["coverage_pct"]
                    best_cat = cat["category"]

            if best_cat:
                gaps.append({
                    "scope": best_cat,
                    "gap_type": "insufficient_coverage",
                    "severity": "high",
                    "current_coverage_pct": best_cat_coverage,
                    "required_coverage_pct": 70.0,
                    "recommendation": f"Increase {best_cat} verification coverage from {best_cat_coverage:.0f}% to >= 70%",
                    "a_requirement": "AREQ04",
                })
            else:
                gaps.append({
                    "scope": "scope_3",
                    "gap_type": "no_verification",
                    "severity": "critical",
                    "current_coverage_pct": 0.0,
                    "required_coverage_pct": 70.0,
                    "recommendation": "Verify at least one Scope 3 category (>= 70% coverage). Start with your largest category.",
                    "a_requirement": "AREQ04",
                })

        # Check assurance levels (reasonable is better for scoring)
        for scope in ALL_SCOPES:
            scope_data = coverage["scopes"].get(scope, {})
            if (
                scope_data.get("verified", False)
                and scope_data.get("assurance_level") == VerificationAssurance.LIMITED.value
            ):
                gaps.append({
                    "scope": scope,
                    "gap_type": "limited_assurance",
                    "severity": "low",
                    "current_coverage_pct": scope_data.get("coverage_pct", 0),
                    "required_coverage_pct": scope_data.get("coverage_pct", 0),
                    "recommendation": f"Consider upgrading {scope} from limited to reasonable assurance for higher scoring",
                    "a_requirement": None,
                })

        return {
            "questionnaire_id": questionnaire_id,
            "total_gaps": len(gaps),
            "critical_gaps": sum(1 for g in gaps if g["severity"] == "critical"),
            "high_gaps": sum(1 for g in gaps if g["severity"] == "high"),
            "gaps": gaps,
            "a_level_eligible": a_check["both_met"],
        }

    # ------------------------------------------------------------------
    # Verification Summary for Scoring
    # ------------------------------------------------------------------

    def get_verification_status(
        self,
        questionnaire_id: str,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Get comprehensive verification status for scoring and dashboard.

        Returns:
            Verification status with coverage, A-level assessment, and gaps.
        """
        records = self.get_records_for_questionnaire(questionnaire_id)
        coverage = self.get_all_coverage(questionnaire_id)
        a_check = self.check_a_level_requirements(questionnaire_id)
        gaps = self.identify_verification_gaps(questionnaire_id)

        # Summary statistics
        total_records = len(records)
        scopes_verified = sum(
            1 for scope in ALL_SCOPES
            if coverage["scopes"].get(scope, {}).get("verified", False)
        )

        # Verification completeness percentage
        # Scope 1 (33.3%), Scope 2 (33.3%), Scope 3 (33.3%) = 100%
        completeness = 0.0
        for scope in ALL_SCOPES:
            scope_data = coverage["scopes"].get(scope, {})
            if scope_data.get("verified", False):
                completeness += (scope_data.get("coverage_pct", 0) / 100.0) * 33.33

        return {
            "questionnaire_id": questionnaire_id,
            "org_id": org_id,
            "total_records": total_records,
            "scopes_verified": scopes_verified,
            "scopes_total": len(ALL_SCOPES),
            "completeness_pct": round(min(completeness, 100.0), 1),
            "coverage": coverage,
            "a_level_assessment": a_check,
            "verification_gaps": gaps,
            "standards_used": self._get_standards_used(records),
            "verifiers": self._get_unique_verifiers(records),
        }

    # ------------------------------------------------------------------
    # Verifier Management
    # ------------------------------------------------------------------

    def get_verifier_details(
        self,
        questionnaire_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get details of all verifiers for a questionnaire.

        Returns:
            List of unique verifiers with their scopes and standards.
        """
        records = self.get_records_for_questionnaire(questionnaire_id)
        verifier_map: Dict[str, Dict[str, Any]] = {}

        for rec in records:
            verifier_key = rec.verifier_organization or rec.verifier_name or "Unknown"
            if verifier_key not in verifier_map:
                verifier_map[verifier_key] = {
                    "name": rec.verifier_name or "",
                    "organization": rec.verifier_organization or "",
                    "accreditation": rec.verifier_accreditation or "",
                    "scopes_verified": [],
                    "standards_used": set(),
                    "records_count": 0,
                }
            entry = verifier_map[verifier_key]
            entry["scopes_verified"].append(rec.scope)
            if rec.verification_standard:
                entry["standards_used"].add(rec.verification_standard)
            entry["records_count"] += 1

        # Convert sets to lists for serialization
        result = []
        for key, info in verifier_map.items():
            info["standards_used"] = sorted(info["standards_used"])
            result.append(info)

        return result

    # ------------------------------------------------------------------
    # Bulk Operations
    # ------------------------------------------------------------------

    def bulk_add_verification(
        self,
        questionnaire_id: str,
        org_id: str,
        verifications: List[Dict[str, Any]],
    ) -> List[VerificationRecord]:
        """
        Add multiple verification records at once.

        Args:
            questionnaire_id: Questionnaire ID.
            org_id: Organization ID.
            verifications: List of verification data dicts.

        Returns:
            List of created VerificationRecords.
        """
        results = []
        for v in verifications:
            assurance = v.get("assurance_level", VerificationAssurance.LIMITED)
            if isinstance(assurance, str):
                assurance = VerificationAssurance(assurance)

            record = self.add_verification(
                questionnaire_id=questionnaire_id,
                org_id=org_id,
                scope=v.get("scope", "scope_1"),
                assurance_level=assurance,
                coverage_pct=Decimal(str(v.get("coverage_pct", 100))),
                verifier_name=v.get("verifier_name"),
                verifier_organization=v.get("verifier_organization"),
                verifier_accreditation=v.get("verifier_accreditation"),
                verification_standard=v.get("verification_standard"),
                emissions_verified_tco2e=v.get("emissions_verified_tco2e"),
                year=v.get("year", self.config.default_questionnaire_year),
            )
            results.append(record)

        logger.info(
            "Bulk added %d verification records for questionnaire %s",
            len(results), questionnaire_id,
        )
        return results

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _validate_scope(self, scope: str) -> None:
        """Validate that a scope string is recognized."""
        valid_scopes = set(ALL_SCOPES + SCOPE_3_CATEGORIES)
        if scope not in valid_scopes:
            raise ValueError(
                f"Invalid scope '{scope}'. Must be one of: "
                f"{', '.join(sorted(valid_scopes))}"
            )

    def _get_standards_used(
        self,
        records: List[VerificationRecord],
    ) -> List[str]:
        """Extract unique verification standards used."""
        standards = set()
        for r in records:
            if r.verification_standard:
                standards.add(r.verification_standard)
        return sorted(standards)

    def _get_unique_verifiers(
        self,
        records: List[VerificationRecord],
    ) -> List[str]:
        """Extract unique verifier organizations."""
        verifiers = set()
        for r in records:
            org = r.verifier_organization or r.verifier_name
            if org:
                verifiers.add(org)
        return sorted(verifiers)
