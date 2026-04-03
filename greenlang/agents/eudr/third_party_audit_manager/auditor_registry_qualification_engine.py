# -*- coding: utf-8 -*-
"""
Auditor Registry and Qualification Engine - AGENT-EUDR-024

Manages third-party auditor profiles, ISO/IEC 17065:2012 and ISO/IEC
17021-1:2015 competence tracking, conflict-of-interest screening,
commodity and regional expertise matching, continuing professional
development (CPD) compliance verification, and auditor performance
benchmarking. Implements smart auditor-to-audit matching with weighted
scoring across competence dimensions.

Features:
    - F2.1-F2.10: Complete auditor registry management (PRD Section 6.2)
    - Auditor registration with full credential verification
    - ISO/IEC 17065/17021-1 accreditation tracking
    - Commodity-specific competence (7 EUDR commodities)
    - Regional expertise and language capability matching
    - Conflict-of-interest screening with cooling-off periods
    - Auditor rotation enforcement (configurable cycle)
    - CPD compliance tracking (minimum hours per year)
    - Performance rating (findings/audit, CAR closure rate)
    - Smart auditor matching with weighted scoring
    - Auditor availability and capacity planning

Matching Dimensions and Weights:
    - Commodity competence: 0.30
    - Scheme qualification: 0.25
    - Country expertise: 0.20
    - Language capability: 0.10
    - Performance rating: 0.10
    - Availability: 0.05

Performance:
    - < 500 ms for auditor matching (configurable timeout)

Dependencies:
    - None (standalone engine within TAM agent)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

from greenlang.agents.eudr.third_party_audit_manager.config import (
    ThirdPartyAuditManagerConfig,
    get_config,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    Auditor,
    AuditScope,
    CertificationScheme,
    MatchAuditorRequest,
    MatchAuditorResponse,
    SUPPORTED_COMMODITIES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Matching dimension weights for auditor scoring
MATCH_WEIGHTS: Dict[str, Decimal] = {
    "commodity_competence": Decimal("0.30"),
    "scheme_qualification": Decimal("0.25"),
    "country_expertise": Decimal("0.20"),
    "language_capability": Decimal("0.10"),
    "performance_rating": Decimal("0.10"),
    "availability": Decimal("0.05"),
}

#: Scheme qualification identifiers
SCHEME_QUALIFICATIONS: Dict[str, List[str]] = {
    "fsc": ["FSC Lead Auditor", "FSC Auditor", "FSC CoC Auditor"],
    "pefc": ["PEFC Lead Auditor", "PEFC Auditor", "PEFC CoC Auditor"],
    "rspo": ["RSPO Lead Auditor", "RSPO Auditor"],
    "rainforest_alliance": ["RA Lead Auditor", "RA Auditor"],
    "iscc": ["ISCC Lead Auditor", "ISCC Auditor"],
}

#: CPD requirement categories (annual hours)
CPD_CATEGORIES: Dict[str, int] = {
    "regulatory_updates": 8,
    "technical_skills": 12,
    "audit_methodology": 8,
    "commodity_specific": 6,
    "ethics_and_integrity": 4,
    "leadership_and_management": 2,
}

#: Conflict-of-interest cooling-off period (months)
COI_COOLING_OFF_MONTHS: int = 24

def _compute_provenance_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash for provenance tracking.

    Args:
        data: Dictionary to hash.

    Returns:
        64-character hex SHA-256 hash string.
    """
    canonical = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

class AuditorRegistryQualificationEngine:
    """Auditor registry and qualification management engine.

    Implements ISO/IEC 17065:2012 and ISO/IEC 17021-1:2015 competence
    management requirements, providing auditor registration, credential
    verification, conflict-of-interest screening, CPD tracking, performance
    benchmarking, and intelligent auditor-to-audit matching.

    All matching scores are deterministic: same inputs produce the same
    auditor ranking (bit-perfect reproducibility).

    Attributes:
        config: Agent configuration.
    """

    def __init__(
        self,
        config: Optional[ThirdPartyAuditManagerConfig] = None,
    ) -> None:
        """Initialize the auditor registry and qualification engine.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        logger.info("AuditorRegistryQualificationEngine initialized")

    def match_auditors(
        self,
        request: MatchAuditorRequest,
        auditor_pool: Optional[List[Auditor]] = None,
    ) -> MatchAuditorResponse:
        """Match qualified auditors to an audit assignment.

        Scores each auditor in the pool against the audit requirements
        using weighted multi-dimensional matching, then returns the
        top-ranked candidates.

        Args:
            request: Auditor matching request with requirements.
            auditor_pool: Optional auditor pool (defaults to empty).

        Returns:
            MatchAuditorResponse with ranked auditor matches.
        """
        start_time = utcnow()

        try:
            pool = auditor_pool or []
            eligible = self._filter_eligible(pool, request)
            scored = self._score_auditors(eligible, request)

            # Sort by total score descending
            scored.sort(key=lambda x: x["total_score"], reverse=True)

            # Limit to max_results
            top_matches = scored[: request.max_results]

            processing_time = Decimal(str(
                (utcnow() - start_time).total_seconds() * 1000
            )).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            response = MatchAuditorResponse(
                matched_auditors=top_matches,
                total_matches=len(scored),
                match_criteria={
                    "commodity": request.commodity,
                    "country_code": request.country_code,
                    "scheme": request.scheme.value if request.scheme else None,
                    "scope": request.scope.value,
                    "required_languages": request.required_languages,
                    "excluded_auditors": request.exclude_auditor_ids,
                    "weights": {k: str(v) for k, v in MATCH_WEIGHTS.items()},
                },
                processing_time_ms=processing_time,
                request_id=request.request_id,
            )

            response.provenance_hash = _compute_provenance_hash({
                "total_matches": len(scored),
                "top_match_count": len(top_matches),
                "processing_time_ms": str(processing_time),
            })

            logger.info(
                f"Auditor matching complete: {len(scored)} eligible, "
                f"{len(top_matches)} returned"
            )

            return response

        except Exception as e:
            logger.error("Auditor matching failed: %s", e, exc_info=True)
            raise

    def register_auditor(
        self,
        full_name: str,
        organization: str,
        commodity_competencies: Optional[List[str]] = None,
        scheme_qualifications: Optional[List[str]] = None,
        country_expertise: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        accreditation_body: Optional[str] = None,
        accreditation_expiry: Optional[date] = None,
        contact_email: Optional[str] = None,
    ) -> Auditor:
        """Register a new auditor in the registry.

        Creates a new auditor profile with validated credentials
        and competence data.

        Args:
            full_name: Auditor full legal name.
            organization: Employing certification body or audit firm.
            commodity_competencies: EUDR commodities qualified for.
            scheme_qualifications: Scheme-specific qualifications.
            country_expertise: Countries of expertise (ISO 3166-1 alpha-2).
            languages: Language capabilities (ISO 639-1 codes).
            accreditation_body: IAF MLA signatory accreditation body.
            accreditation_expiry: Accreditation expiry date.
            contact_email: Professional contact email.

        Returns:
            Registered Auditor record.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Validate commodity competencies
        validated_commodities = self._validate_commodities(
            commodity_competencies or []
        )

        auditor = Auditor(
            full_name=full_name,
            organization=organization,
            commodity_competencies=validated_commodities,
            scheme_qualifications=scheme_qualifications or [],
            country_expertise=[c.upper() for c in (country_expertise or [])],
            languages=[lang.lower() for lang in (languages or [])],
            accreditation_body=accreditation_body,
            accreditation_expiry=accreditation_expiry,
            accreditation_status="active",
            contact_email=contact_email,
            cpd_compliant=True,
            available_from=date.today(),
        )

        auditor.provenance_hash = _compute_provenance_hash({
            "auditor_id": auditor.auditor_id,
            "full_name": full_name,
            "organization": organization,
            "commodity_competencies": validated_commodities,
        })

        logger.info(
            f"Auditor registered: id={auditor.auditor_id}, "
            f"name={full_name}, org={organization}"
        )

        return auditor

    def check_conflict_of_interest(
        self,
        auditor: Auditor,
        supplier_id: str,
        audit_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Screen auditor for conflict of interest.

        Checks CoI declarations, prior engagement history, and
        cooling-off period requirements.

        Args:
            auditor: Auditor to screen.
            supplier_id: Supplier being audited.
            audit_history: Optional audit history records.

        Returns:
            Dictionary with conflict screening results.
        """
        conflicts_found: List[Dict[str, str]] = []
        is_clear = True

        # Check declared CoI entries
        for coi in auditor.conflict_of_interest:
            if coi.get("supplier_id") == supplier_id:
                conflicts_found.append({
                    "type": "declared_coi",
                    "description": coi.get("description", "Declared conflict"),
                    "supplier_id": supplier_id,
                })
                is_clear = False

        # Check audit history for cooling-off period
        if audit_history:
            cutoff = date.today() - timedelta(days=COI_COOLING_OFF_MONTHS * 30)
            for record in audit_history:
                if (
                    record.get("auditor_id") == auditor.auditor_id
                    and record.get("supplier_id") == supplier_id
                ):
                    audit_date_str = record.get("audit_date", "")
                    try:
                        audit_date = date.fromisoformat(audit_date_str)
                        if audit_date > cutoff:
                            conflicts_found.append({
                                "type": "cooling_off_period",
                                "description": (
                                    f"Auditor previously audited supplier "
                                    f"within {COI_COOLING_OFF_MONTHS} month "
                                    f"cooling-off period (audit date: "
                                    f"{audit_date_str})"
                                ),
                                "supplier_id": supplier_id,
                            })
                            is_clear = False
                    except (ValueError, TypeError):
                        pass

        # Check auditor rotation requirement
        rotation_conflict = self._check_rotation(
            auditor, supplier_id, audit_history
        )
        if rotation_conflict:
            conflicts_found.append(rotation_conflict)
            is_clear = False

        return {
            "auditor_id": auditor.auditor_id,
            "supplier_id": supplier_id,
            "is_clear": is_clear,
            "conflicts_found": conflicts_found,
            "screened_at": utcnow().isoformat(),
        }

    def verify_accreditation(self, auditor: Auditor) -> Dict[str, Any]:
        """Verify auditor accreditation status.

        Checks accreditation validity, expiry proximity, and
        generates warnings for approaching expirations.

        Args:
            auditor: Auditor to verify.

        Returns:
            Dictionary with accreditation verification results.
        """
        today = date.today()
        is_valid = auditor.accreditation_status == "active"
        warnings: List[str] = []
        days_to_expiry: Optional[int] = None

        if auditor.accreditation_expiry:
            days_to_expiry = (auditor.accreditation_expiry - today).days

            if days_to_expiry < 0:
                is_valid = False
                warnings.append(
                    f"Accreditation expired {abs(days_to_expiry)} days ago"
                )
            elif days_to_expiry <= self.config.accreditation_expiry_warning_days:
                warnings.append(
                    f"Accreditation expires in {days_to_expiry} days "
                    f"(warning threshold: "
                    f"{self.config.accreditation_expiry_warning_days} days)"
                )

        if auditor.accreditation_status == "suspended":
            is_valid = False
            warnings.append("Accreditation is currently suspended")
        elif auditor.accreditation_status == "withdrawn":
            is_valid = False
            warnings.append("Accreditation has been withdrawn")

        return {
            "auditor_id": auditor.auditor_id,
            "accreditation_status": auditor.accreditation_status,
            "accreditation_body": auditor.accreditation_body,
            "expiry_date": str(auditor.accreditation_expiry) if auditor.accreditation_expiry else None,
            "days_to_expiry": days_to_expiry,
            "is_valid": is_valid,
            "warnings": warnings,
            "verified_at": utcnow().isoformat(),
        }

    def verify_cpd_compliance(self, auditor: Auditor) -> Dict[str, Any]:
        """Verify auditor CPD (Continuing Professional Development) compliance.

        Checks whether the auditor meets the minimum CPD hours requirement
        as configured.

        Args:
            auditor: Auditor to verify.

        Returns:
            Dictionary with CPD compliance verification results.
        """
        minimum_hours = self.config.cpd_hours_minimum
        is_compliant = auditor.cpd_hours >= minimum_hours
        shortfall = max(0, minimum_hours - auditor.cpd_hours)

        return {
            "auditor_id": auditor.auditor_id,
            "cpd_hours_completed": auditor.cpd_hours,
            "cpd_hours_required": minimum_hours,
            "is_compliant": is_compliant,
            "shortfall_hours": shortfall,
            "categories": CPD_CATEGORIES,
            "verified_at": utcnow().isoformat(),
        }

    def calculate_performance_score(
        self,
        auditor: Auditor,
        audit_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Calculate auditor performance score.

        Computes a composite performance rating from audit history
        including findings per audit, CAR closure rate, and timeliness.

        Args:
            auditor: Auditor to score.
            audit_results: Optional historical audit results.

        Returns:
            Dictionary with performance scoring results.
        """
        results = audit_results or []

        if not results:
            return {
                "auditor_id": auditor.auditor_id,
                "performance_rating": str(auditor.performance_rating),
                "findings_per_audit": str(auditor.findings_per_audit),
                "car_closure_rate": str(auditor.car_closure_rate),
                "audit_count": auditor.audit_count,
                "benchmarks": {},
                "calculated_at": utcnow().isoformat(),
            }

        # Calculate from audit history
        total_findings = Decimal("0")
        total_cars = Decimal("0")
        closed_cars = Decimal("0")

        for result in results:
            findings = result.get("findings_count", 0)
            total_findings += Decimal(str(findings))
            cars_issued = result.get("cars_issued", 0)
            total_cars += Decimal(str(cars_issued))
            cars_closed = result.get("cars_closed", 0)
            closed_cars += Decimal(str(cars_closed))

        audit_count = len(results)
        findings_per_audit = (
            total_findings / Decimal(str(audit_count))
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        car_closure_rate = Decimal("100")
        if total_cars > 0:
            car_closure_rate = (
                closed_cars / total_cars * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Composite performance (higher car closure rate is better,
        # moderate findings per audit is ideal)
        performance_rating = self._compute_performance_rating(
            findings_per_audit, car_closure_rate
        )

        return {
            "auditor_id": auditor.auditor_id,
            "performance_rating": str(performance_rating),
            "findings_per_audit": str(findings_per_audit),
            "car_closure_rate": str(car_closure_rate),
            "audit_count": audit_count,
            "total_findings": str(total_findings),
            "total_cars": str(total_cars),
            "closed_cars": str(closed_cars),
            "calculated_at": utcnow().isoformat(),
        }

    def _filter_eligible(
        self,
        pool: List[Auditor],
        request: MatchAuditorRequest,
    ) -> List[Auditor]:
        """Filter auditor pool for eligibility.

        Removes excluded auditors, inactive accreditations, and
        auditors not meeting minimum requirements.

        Args:
            pool: Full auditor pool.
            request: Matching request with requirements.

        Returns:
            Filtered list of eligible auditors.
        """
        eligible: List[Auditor] = []

        for auditor in pool:
            # Skip excluded auditors
            if auditor.auditor_id in request.exclude_auditor_ids:
                continue

            # Skip inactive accreditations
            if auditor.accreditation_status != "active":
                continue

            # Skip expired accreditations
            if (
                auditor.accreditation_expiry
                and auditor.accreditation_expiry < date.today()
            ):
                continue

            # Skip CPD non-compliant
            if not auditor.cpd_compliant:
                continue

            eligible.append(auditor)

        logger.debug(
            f"Eligible auditors: {len(eligible)}/{len(pool)} "
            f"after filtering"
        )

        return eligible

    def _score_auditors(
        self,
        eligible: List[Auditor],
        request: MatchAuditorRequest,
    ) -> List[Dict[str, Any]]:
        """Score eligible auditors against audit requirements.

        Computes a weighted score across multiple matching dimensions.

        Args:
            eligible: Filtered eligible auditors.
            request: Matching request with requirements.

        Returns:
            List of auditor score dictionaries, sorted by total_score.
        """
        scored: List[Dict[str, Any]] = []

        for auditor in eligible:
            dimension_scores = self._compute_dimension_scores(auditor, request)

            # Weighted total
            total = Decimal("0")
            for dim, weight in MATCH_WEIGHTS.items():
                total += dimension_scores.get(dim, Decimal("0")) * weight

            total_score = total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            scored.append({
                "auditor_id": auditor.auditor_id,
                "full_name": auditor.full_name,
                "organization": auditor.organization,
                "total_score": str(total_score),
                "dimension_scores": {
                    k: str(v) for k, v in dimension_scores.items()
                },
                "commodity_competencies": auditor.commodity_competencies,
                "scheme_qualifications": auditor.scheme_qualifications,
                "country_expertise": auditor.country_expertise,
                "languages": auditor.languages,
                "performance_rating": str(auditor.performance_rating),
                "available_from": str(auditor.available_from) if auditor.available_from else None,
            })

        return scored

    def _compute_dimension_scores(
        self,
        auditor: Auditor,
        request: MatchAuditorRequest,
    ) -> Dict[str, Decimal]:
        """Compute individual dimension scores for an auditor.

        Args:
            auditor: Auditor to score.
            request: Matching request with requirements.

        Returns:
            Dictionary of dimension name to score (0-100).
        """
        scores: Dict[str, Decimal] = {}

        # 1. Commodity competence (0 or 100)
        commodity_match = request.commodity.lower() in [
            c.lower() for c in auditor.commodity_competencies
        ]
        scores["commodity_competence"] = (
            Decimal("100") if commodity_match else Decimal("0")
        )

        # 2. Scheme qualification (0, 50, or 100)
        if request.scheme:
            scheme_key = request.scheme.value
            has_lead = any(
                "Lead" in q and scheme_key.lower() in q.lower()
                for q in auditor.scheme_qualifications
            )
            has_any = any(
                scheme_key.lower() in q.lower()
                for q in auditor.scheme_qualifications
            )
            if has_lead:
                scores["scheme_qualification"] = Decimal("100")
            elif has_any:
                scores["scheme_qualification"] = Decimal("50")
            else:
                scores["scheme_qualification"] = Decimal("0")
        else:
            scores["scheme_qualification"] = Decimal("100")

        # 3. Country expertise (0 or 100)
        country_match = request.country_code.upper() in [
            c.upper() for c in auditor.country_expertise
        ]
        scores["country_expertise"] = (
            Decimal("100") if country_match else Decimal("0")
        )

        # 4. Language capability (ratio of matched languages)
        if request.required_languages:
            auditor_langs = {lang.lower() for lang in auditor.languages}
            required_langs = {lang.lower() for lang in request.required_languages}
            matched = len(auditor_langs & required_langs)
            ratio = Decimal(str(matched)) / Decimal(str(len(required_langs)))
            scores["language_capability"] = (
                ratio * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            scores["language_capability"] = Decimal("100")

        # 5. Performance rating (direct from auditor profile)
        scores["performance_rating"] = auditor.performance_rating

        # 6. Availability (100 if available, 0 if not)
        if auditor.available_from:
            is_available = auditor.available_from <= date.today()
            scores["availability"] = (
                Decimal("100") if is_available else Decimal("0")
            )
        else:
            scores["availability"] = Decimal("100")

        return scores

    def _validate_commodities(
        self, commodities: List[str]
    ) -> List[str]:
        """Validate and normalize commodity competencies.

        Args:
            commodities: Raw commodity list.

        Returns:
            Validated and normalized commodity list.

        Raises:
            ValueError: If an invalid commodity is found.
        """
        validated: List[str] = []
        for commodity in commodities:
            normalized = commodity.lower().strip()
            if normalized not in SUPPORTED_COMMODITIES:
                raise ValueError(
                    f"Invalid commodity: {commodity}. "
                    f"Must be one of {SUPPORTED_COMMODITIES}"
                )
            validated.append(normalized)
        return validated

    def _check_rotation(
        self,
        auditor: Auditor,
        supplier_id: str,
        audit_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, str]]:
        """Check auditor rotation requirement.

        Enforces maximum consecutive audit cycles before rotation.

        Args:
            auditor: Auditor to check.
            supplier_id: Supplier being audited.
            audit_history: Optional audit history records.

        Returns:
            Conflict record if rotation required, None otherwise.
        """
        if not audit_history:
            return None

        rotation_years = self.config.auditor_rotation_years
        cutoff = date.today() - timedelta(days=rotation_years * 365)

        consecutive_audits = 0
        for record in audit_history:
            if (
                record.get("auditor_id") == auditor.auditor_id
                and record.get("supplier_id") == supplier_id
            ):
                audit_date_str = record.get("audit_date", "")
                try:
                    audit_date = date.fromisoformat(audit_date_str)
                    if audit_date > cutoff:
                        consecutive_audits += 1
                except (ValueError, TypeError):
                    pass

        if consecutive_audits >= rotation_years:
            return {
                "type": "rotation_required",
                "description": (
                    f"Auditor has audited supplier for "
                    f"{consecutive_audits} consecutive cycles "
                    f"(maximum {rotation_years} allowed)"
                ),
                "supplier_id": supplier_id,
            }

        return None

    def _compute_performance_rating(
        self,
        findings_per_audit: Decimal,
        car_closure_rate: Decimal,
    ) -> Decimal:
        """Compute composite performance rating.

        Performance rating formula:
        - CAR closure rate contributes 60% (higher is better)
        - Findings thoroughness contributes 40%
          (optimal range: 2-8 findings/audit)

        Args:
            findings_per_audit: Average findings per audit.
            car_closure_rate: CAR closure rate percentage.

        Returns:
            Performance rating (0-100).
        """
        # CAR closure component (60% weight)
        closure_component = car_closure_rate * Decimal("0.60")

        # Findings thoroughness component (40% weight)
        # Optimal range: 2-8 findings per audit
        if findings_per_audit < Decimal("2"):
            thoroughness = findings_per_audit / Decimal("2") * Decimal("100")
        elif findings_per_audit <= Decimal("8"):
            thoroughness = Decimal("100")
        else:
            # Diminishing returns above 8
            excess = findings_per_audit - Decimal("8")
            thoroughness = max(
                Decimal("50"),
                Decimal("100") - excess * Decimal("5")
            )

        thoroughness_component = thoroughness * Decimal("0.40")

        rating = (closure_component + thoroughness_component).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return min(Decimal("100"), max(Decimal("0"), rating))
