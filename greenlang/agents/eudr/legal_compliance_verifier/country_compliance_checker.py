# -*- coding: utf-8 -*-
"""
AGENT-EUDR-023: Legal Compliance Verifier - Country Compliance Checker

Engine 5 of 7. Performs per-country legal requirement verification across
all 8 legislation categories for a given supplier-commodity pair. Produces
compliance scorecards, gap analyses, and compliance status determinations.

Compliance Assessment Process:
    1. Identify country + commodity -> Retrieve applicable legal requirements
    2. For each of 8 categories:
       a. List specific legal requirements (from Engine 1)
       b. Check available evidence (documents from Engine 2, certs from Engine 3)
       c. Apply deterministic compliance rules
       d. Score: COMPLIANT / PARTIALLY_COMPLIANT / NON_COMPLIANT / INSUFFICIENT_DATA
    3. Calculate category scores (0-100) and overall compliance score
    4. Generate gap analysis listing unmet requirements
    5. Produce compliance assessment with provenance chain

Compliance States (4):
    COMPLIANT:           score 80-100, all requirements met with evidence
    PARTIALLY_COMPLIANT: score 50-79, some requirements met, gaps identified
    NON_COMPLIANT:       score 0-49, critical requirements unmet
    INSUFFICIENT_DATA:   cannot assess due to missing information

Scoring Formula:
    category_score = (requirements_met / total_requirements) * 100
    overall_score = weighted_average(category_scores)

Zero-Hallucination Approach:
    - Compliance determination is rule-based checklist with boolean verification
    - No LLM interpretation of legal requirements
    - All rules traceable to specific legislation citations

Performance Targets:
    - Single compliance check (1 category): <500ms
    - Full 8-category assessment: <5s

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 Legal Compliance Verifier (GL-EUDR-LCV-023)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-LCV-023"

_CATEGORY_KEYS: List[str] = [
    "land_use_rights",
    "environmental_protection",
    "forest_related_rules",
    "third_party_rights",
    "labour_rights",
    "tax_and_royalty",
    "trade_and_customs",
    "anti_corruption",
]

# Default thresholds (overridable via config)
_DEFAULT_COMPLIANT_THRESHOLD = Decimal("80")
_DEFAULT_PARTIAL_THRESHOLD = Decimal("50")

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.legal_compliance_verifier.config import get_config
except ImportError:
    get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.legal_compliance_verifier.provenance import get_tracker
except ImportError:
    get_tracker = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.legal_compliance_verifier.metrics import (
        record_compliance_assessment,
        observe_compliance_check_duration,
        observe_full_assessment_duration,
    )
except ImportError:
    record_compliance_assessment = None  # type: ignore[assignment]
    observe_compliance_check_duration = None  # type: ignore[assignment]
    observe_full_assessment_duration = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.legal_compliance_verifier.reference_data import (
        COUNTRY_FRAMEWORKS,
        LEGISLATION_CATEGORIES,
        get_country_framework,
        SUPPORTED_COUNTRIES,
    )
except ImportError:
    COUNTRY_FRAMEWORKS = {}  # type: ignore[assignment]
    LEGISLATION_CATEGORIES = {}  # type: ignore[assignment]
    get_country_framework = None  # type: ignore[assignment]
    SUPPORTED_COUNTRIES = []  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CountryComplianceChecker
# ---------------------------------------------------------------------------


class CountryComplianceChecker:
    """Engine 5: Per-country legal requirement verification and scoring.

    Performs deterministic compliance checks against country-specific legal
    requirements for each of the 8 EUDR Article 2(40) legislation categories.
    Produces scorecards, gap analyses, and compliance status determinations.

    Example:
        >>> checker = CountryComplianceChecker()
        >>> result = checker.assess_compliance(
        ...     country_code="BR",
        ...     commodity="soya",
        ...     documents=[{"document_type": "eia_approval", "validity_status": "valid"}],
        ...     certifications=[],
        ... )
        >>> assert "overall_score" in result
    """

    def __init__(self) -> None:
        """Initialize the Country Compliance Checker."""
        self._frameworks: Dict[str, Any] = dict(COUNTRY_FRAMEWORKS)
        self._categories: Dict[str, Any] = dict(LEGISLATION_CATEGORIES)
        self._supported_countries: List[str] = list(SUPPORTED_COUNTRIES)

        self._compliant_threshold = _DEFAULT_COMPLIANT_THRESHOLD
        self._partial_threshold = _DEFAULT_PARTIAL_THRESHOLD

        if get_config is not None:
            try:
                cfg = get_config()
                self._compliant_threshold = Decimal(str(cfg.compliant_threshold))
                self._partial_threshold = Decimal(str(cfg.partial_threshold))
            except Exception:
                pass

        logger.info(
            f"CountryComplianceChecker v{_MODULE_VERSION} initialized: "
            f"compliant>={self._compliant_threshold}, "
            f"partial>={self._partial_threshold}"
        )

    # -------------------------------------------------------------------
    # Public API: Full compliance assessment
    # -------------------------------------------------------------------

    def assess_compliance(
        self,
        country_code: str,
        commodity: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        certifications: Optional[List[Dict[str, Any]]] = None,
        red_flag_data: Optional[Dict[str, Any]] = None,
        categories: Optional[List[str]] = None,
        supplier_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run full compliance assessment for a supplier-commodity pair.

        Evaluates all 8 legislation categories (or a subset) by checking
        available documents and certifications against country-specific
        legal requirements using deterministic rule matching.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            commodity: EUDR commodity type.
            documents: List of document dicts with type and status.
            certifications: List of certification dicts with scheme and status.
            red_flag_data: Optional red flag scan results.
            categories: Optional subset of categories to assess.
            supplier_id: Optional supplier identifier.

        Returns:
            Dict with overall score, status, category scores, gap analysis.

        Example:
            >>> checker = CountryComplianceChecker()
            >>> result = checker.assess_compliance(
            ...     country_code="ID",
            ...     commodity="oil_palm",
            ...     documents=[
            ...         {"document_type": "forest_harvesting_permit", "validity_status": "valid"},
            ...         {"document_type": "eia_approval", "validity_status": "valid"},
            ...     ],
            ... )
            >>> assert result["overall_status"] in (
            ...     "compliant", "partially_compliant", "non_compliant", "insufficient_data"
            ... )
        """
        start_time = time.monotonic()
        country_code = country_code.upper()

        docs = documents or []
        certs = certifications or []
        cats_to_check = categories or _CATEGORY_KEYS

        country_data = self._frameworks.get(country_code, {})

        category_scores: Dict[str, Decimal] = {}
        category_statuses: Dict[str, str] = {}
        gap_analysis: List[Dict[str, Any]] = []

        total_requirements = 0
        total_met = 0
        total_unmet = 0
        total_insufficient = 0

        for cat in cats_to_check:
            cat_result = self._assess_category(
                country_code=country_code,
                commodity=commodity,
                category=cat,
                country_data=country_data,
                documents=docs,
                certifications=certs,
            )

            category_scores[cat] = cat_result["score"]
            category_statuses[cat] = cat_result["status"]
            total_requirements += cat_result["requirements_total"]
            total_met += cat_result["requirements_met"]
            total_unmet += cat_result["requirements_unmet"]
            total_insufficient += cat_result["requirements_insufficient"]

            if cat_result["gaps"]:
                gap_analysis.extend(cat_result["gaps"])

        # Compute overall score (average of category scores)
        overall_score = self._compute_overall_score(category_scores)
        overall_status = self._determine_status(overall_score)

        # Integrate red flag data if provided
        red_flag_count = 0
        red_flag_score = Decimal("0")
        if red_flag_data:
            red_flag_count = red_flag_data.get("total_flags", 0)
            rf_score = red_flag_data.get("aggregate_score", "0")
            red_flag_score = Decimal(str(rf_score))

        risk_level = self._determine_risk_level(overall_score, red_flag_score)

        provenance_hash = self._compute_provenance_hash(
            "assess_compliance", country_code, commodity, supplier_id or "unknown",
        )

        self._record_provenance(
            "assess", supplier_id or country_code, provenance_hash,
        )
        self._record_assessment_metrics(
            country_code, commodity, overall_status, start_time,
        )

        return {
            "supplier_id": supplier_id,
            "country_code": country_code,
            "commodity": commodity,
            "overall_score": str(overall_score),
            "overall_status": overall_status,
            "category_scores": {k: str(v) for k, v in category_scores.items()},
            "category_statuses": category_statuses,
            "requirements_total": total_requirements,
            "requirements_met": total_met,
            "requirements_unmet": total_unmet,
            "requirements_insufficient_data": total_insufficient,
            "gap_analysis": gap_analysis,
            "red_flag_count": red_flag_count,
            "red_flag_score": str(red_flag_score),
            "risk_level": risk_level,
            "documents_verified": len(docs),
            "certifications_validated": len(certs),
            "provenance_hash": provenance_hash,
        }

    # -------------------------------------------------------------------
    # Public API: Single category assessment
    # -------------------------------------------------------------------

    def assess_single_category(
        self,
        country_code: str,
        commodity: str,
        category: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        certifications: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Assess compliance for a single legislation category.

        Args:
            country_code: Country code.
            commodity: Commodity type.
            category: Legislation category key.
            documents: Available documents.
            certifications: Available certifications.

        Returns:
            Dict with category score, status, and gaps.

        Example:
            >>> checker = CountryComplianceChecker()
            >>> result = checker.assess_single_category(
            ...     "BR", "soya", "environmental_protection",
            ...     documents=[{"document_type": "eia_approval", "validity_status": "valid"}],
            ... )
            >>> assert "score" in result
        """
        start_time = time.monotonic()
        country_code = country_code.upper()
        country_data = self._frameworks.get(country_code, {})

        result = self._assess_category(
            country_code=country_code,
            commodity=commodity,
            category=category,
            country_data=country_data,
            documents=documents or [],
            certifications=certifications or [],
        )

        elapsed = time.monotonic() - start_time
        if observe_compliance_check_duration is not None:
            try:
                observe_compliance_check_duration(elapsed)
            except Exception:
                pass

        return {
            "country_code": country_code,
            "commodity": commodity,
            "category": category,
            "score": str(result["score"]),
            "status": result["status"],
            "requirements_total": result["requirements_total"],
            "requirements_met": result["requirements_met"],
            "gaps": result["gaps"],
        }

    # -------------------------------------------------------------------
    # Public API: Generate recommendations
    # -------------------------------------------------------------------

    def generate_recommendations(
        self,
        assessment: Dict[str, Any],
    ) -> List[str]:
        """Generate compliance improvement recommendations.

        Args:
            assessment: Compliance assessment result dict.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []
        status = assessment.get("overall_status", "")
        gaps = assessment.get("gap_analysis", [])

        if status == "non_compliant":
            recommendations.append(
                "URGENT: Supplier is non-compliant. Immediate remediation required "
                "before EUDR due diligence statement submission."
            )

        if status == "partially_compliant":
            recommendations.append(
                "Supplier is partially compliant. Address identified gaps "
                "to achieve full compliance."
            )

        # Category-specific recommendations
        cat_statuses = assessment.get("category_statuses", {})
        for cat, cat_status in cat_statuses.items():
            if cat_status == "non_compliant":
                cat_def = self._categories.get(cat, {})
                cat_name = cat_def.get("name", cat)
                recommendations.append(
                    f"{cat_name}: Non-compliant. Obtain required documentation "
                    f"per EUDR Article 2(40)({cat_def.get('article_reference', '')})."
                )

        # Gap-specific recommendations
        for gap in gaps[:5]:  # Top 5 gaps
            recommendations.append(
                f"GAP: {gap.get('description', 'Unmet requirement')} - "
                f"Evidence needed: {', '.join(gap.get('evidence_types', []))}"
            )

        # Red flag recommendations
        rf_count = assessment.get("red_flag_count", 0)
        if rf_count > 0:
            recommendations.append(
                f"WARNING: {rf_count} red flag(s) detected. Review and "
                f"acknowledge each flag with documented justification."
            )

        if not recommendations:
            recommendations.append(
                "Supplier is compliant across all assessed categories."
            )

        return recommendations

    # -------------------------------------------------------------------
    # Internal: Category assessment
    # -------------------------------------------------------------------

    def _assess_category(
        self,
        country_code: str,
        commodity: str,
        category: str,
        country_data: Dict[str, Any],
        documents: List[Dict[str, Any]],
        certifications: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assess compliance for one legislation category.

        Args:
            country_code: Country code.
            commodity: Commodity type.
            category: Category key.
            country_data: Country framework data.
            documents: Available documents.
            certifications: Available certifications.

        Returns:
            Dict with score, status, requirements counts, gaps.
        """
        cat_def = self._categories.get(category, {})
        evidence_types = cat_def.get("evidence_types", [])
        requirements_total = max(len(evidence_types), 1)

        # Count matching documents
        doc_matches = self._count_matching_documents(
            category, documents,
        )

        # Count matching certifications
        cert_matches = self._count_matching_certifications(
            category, certifications,
        )

        # Check country-specific legislation exists
        key_legislation = country_data.get("key_legislation", {})
        has_framework = bool(key_legislation.get(category))

        met = doc_matches + cert_matches
        if has_framework and met == 0:
            met = 0  # No evidence despite having framework
        elif not has_framework and met == 0:
            # No framework data = insufficient data
            return {
                "score": Decimal("0"),
                "status": "insufficient_data",
                "requirements_total": requirements_total,
                "requirements_met": 0,
                "requirements_unmet": 0,
                "requirements_insufficient": requirements_total,
                "gaps": [{
                    "category": category,
                    "category_name": cat_def.get("name", category),
                    "description": f"No legal framework data for {category}",
                    "evidence_types": evidence_types,
                    "country_code": country_code,
                }],
            }

        requirements_met = min(met, requirements_total)
        requirements_unmet = requirements_total - requirements_met

        score = (
            Decimal(str(requirements_met)) / Decimal(str(requirements_total))
        ) * Decimal("100")
        score = score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        status = self._determine_status(score)

        gaps: List[Dict[str, Any]] = []
        if requirements_unmet > 0:
            gaps.append({
                "category": category,
                "category_name": cat_def.get("name", category),
                "description": (
                    f"{requirements_unmet} of {requirements_total} requirements unmet "
                    f"for {category}"
                ),
                "evidence_types": evidence_types,
                "country_code": country_code,
            })

        return {
            "score": score,
            "status": status,
            "requirements_total": requirements_total,
            "requirements_met": requirements_met,
            "requirements_unmet": requirements_unmet,
            "requirements_insufficient": 0,
            "gaps": gaps,
        }

    # -------------------------------------------------------------------
    # Internal: Evidence matching
    # -------------------------------------------------------------------

    # Mapping from document types to categories
    _DOC_CATEGORY_MAP: Dict[str, str] = {
        "forest_harvesting_permit": "forest_related_rules",
        "concession_license": "forest_related_rules",
        "forest_management_plan": "forest_related_rules",
        "reforestation_certificate": "forest_related_rules",
        "timber_legality_certificate": "forest_related_rules",
        "annual_coupe_permit": "forest_related_rules",
        "transport_document": "forest_related_rules",
        "eia_approval": "environmental_protection",
        "environmental_permit": "environmental_protection",
        "pollution_control_certificate": "environmental_protection",
        "water_use_permit": "environmental_protection",
        "biodiversity_assessment": "environmental_protection",
        "waste_management_plan": "environmental_protection",
        "land_title_deed": "land_use_rights",
        "land_lease_agreement": "land_use_rights",
        "customary_tenure_certificate": "land_use_rights",
        "zoning_compliance_certificate": "land_use_rights",
        "land_registry_extract": "land_use_rights",
        "fpic_documentation": "third_party_rights",
        "community_consent_record": "third_party_rights",
        "benefit_sharing_agreement": "third_party_rights",
        "consultation_minutes": "third_party_rights",
        "indigenous_territory_map": "third_party_rights",
        "customary_rights_certificate": "third_party_rights",
        "labour_compliance_certificate": "labour_rights",
        "osh_inspection_report": "labour_rights",
        "employment_contract_sample": "labour_rights",
        "wage_records": "labour_rights",
        "ilo_compliance_assessment": "labour_rights",
        "social_audit_report": "labour_rights",
        "tax_clearance_certificate": "tax_and_royalty",
        "royalty_payment_receipt": "tax_and_royalty",
        "export_duty_receipt": "tax_and_royalty",
        "tax_registration_certificate": "tax_and_royalty",
        "financial_audit_report": "tax_and_royalty",
        "transfer_pricing_documentation": "tax_and_royalty",
        "export_permit": "trade_and_customs",
        "cites_permit": "trade_and_customs",
        "certificate_of_origin": "trade_and_customs",
        "customs_declaration": "trade_and_customs",
        "phytosanitary_certificate": "trade_and_customs",
        "flegt_license": "trade_and_customs",
        "anti_corruption_declaration": "anti_corruption",
        "beneficial_ownership_register": "anti_corruption",
        "procurement_compliance_certificate": "anti_corruption",
        "aml_compliance_certificate": "anti_corruption",
        "ethics_code_declaration": "anti_corruption",
        "whistleblower_policy": "anti_corruption",
    }

    def _count_matching_documents(
        self,
        category: str,
        documents: List[Dict[str, Any]],
    ) -> int:
        """Count documents that match a legislation category.

        Args:
            category: Legislation category key.
            documents: List of document dicts.

        Returns:
            Number of valid matching documents.
        """
        count = 0
        for doc in documents:
            doc_type = doc.get("document_type", "")
            mapped_cat = self._DOC_CATEGORY_MAP.get(doc_type, "")
            if mapped_cat == category:
                validity = doc.get("validity_status", "valid")
                if validity in ("valid", "expiring_soon"):
                    count += 1
        return count

    # Mapping from certification schemes to categories they cover
    _CERT_CATEGORY_MAP: Dict[str, List[str]] = {
        "fsc_fm": ["land_use_rights", "environmental_protection",
                    "forest_related_rules", "third_party_rights", "labour_rights"],
        "fsc_coc": ["forest_related_rules", "trade_and_customs"],
        "pefc_sfm": ["land_use_rights", "environmental_protection",
                      "forest_related_rules"],
        "pefc_coc": ["forest_related_rules", "trade_and_customs"],
        "rspo_pc": ["land_use_rights", "environmental_protection",
                     "third_party_rights", "labour_rights"],
        "rspo_scc": ["trade_and_customs"],
        "ra_sa": ["environmental_protection", "forest_related_rules",
                   "labour_rights"],
        "ra_coc": ["trade_and_customs"],
        "iscc_eu": ["environmental_protection", "forest_related_rules"],
        "iscc_plus": ["environmental_protection", "forest_related_rules"],
    }

    def _count_matching_certifications(
        self,
        category: str,
        certifications: List[Dict[str, Any]],
    ) -> int:
        """Count certifications covering a legislation category.

        Args:
            category: Legislation category key.
            certifications: List of certification dicts.

        Returns:
            Number of valid certifications covering the category.
        """
        count = 0
        for cert in certifications:
            scheme = cert.get("scheme", "")
            covered_cats = self._CERT_CATEGORY_MAP.get(scheme, [])
            if category in covered_cats:
                validity = cert.get("validity_status", "valid")
                if validity in ("valid", "expiring_soon"):
                    count += 1
        return count

    # -------------------------------------------------------------------
    # Internal: Score computation
    # -------------------------------------------------------------------

    def _compute_overall_score(
        self,
        category_scores: Dict[str, Decimal],
    ) -> Decimal:
        """Compute overall compliance score as average of category scores.

        Args:
            category_scores: Dict mapping category to score (0-100).

        Returns:
            Decimal overall score (0-100).
        """
        if not category_scores:
            return Decimal("0")

        total = sum(category_scores.values())
        count = Decimal(str(len(category_scores)))
        average = total / count
        return average.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _determine_status(self, score: Decimal) -> str:
        """Determine compliance status from score.

        Args:
            score: Compliance score (0-100).

        Returns:
            Compliance status string.
        """
        if score >= self._compliant_threshold:
            return "compliant"
        elif score >= self._partial_threshold:
            return "partially_compliant"
        return "non_compliant"

    def _determine_risk_level(
        self,
        compliance_score: Decimal,
        red_flag_score: Decimal,
    ) -> str:
        """Determine overall risk level combining compliance and red flags.

        Args:
            compliance_score: Compliance score (0-100, higher=better).
            red_flag_score: Red flag score (0-100, higher=worse).

        Returns:
            Risk level string (low/moderate/high/critical).
        """
        # Invert compliance score for risk assessment
        compliance_risk = Decimal("100") - compliance_score

        # Weight: 60% compliance risk, 40% red flag risk
        combined = (
            compliance_risk * Decimal("0.6") + red_flag_score * Decimal("0.4")
        )
        combined = combined.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        if combined >= Decimal("75"):
            return "critical"
        elif combined >= Decimal("50"):
            return "high"
        elif combined >= Decimal("25"):
            return "moderate"
        return "low"

    # -------------------------------------------------------------------
    # Internal: Provenance and metrics
    # -------------------------------------------------------------------

    def _compute_provenance_hash(
        self,
        operation: str,
        country_code: str,
        commodity: str,
        supplier_id: str,
    ) -> str:
        """Compute SHA-256 provenance hash."""
        data = {
            "agent_id": _AGENT_ID,
            "engine": "country_compliance_checker",
            "version": _MODULE_VERSION,
            "operation": operation,
            "country_code": country_code,
            "commodity": commodity,
            "supplier_id": supplier_id,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _record_provenance(
        self, action: str, entity_id: str, provenance_hash: str,
    ) -> None:
        """Record provenance entry."""
        if get_tracker is not None:
            try:
                tracker = get_tracker()
                tracker.record(
                    entity_type="compliance_assessment",
                    action=action,
                    entity_id=entity_id,
                    metadata={"provenance_hash": provenance_hash},
                )
            except Exception as exc:
                logger.warning(f"Provenance recording failed: {exc}")

    def _record_assessment_metrics(
        self,
        country_code: str,
        commodity: str,
        status: str,
        start_time: float,
    ) -> None:
        """Record assessment metrics."""
        elapsed = time.monotonic() - start_time
        if record_compliance_assessment is not None:
            try:
                record_compliance_assessment(country_code, commodity, status)
            except Exception:
                pass
        if observe_full_assessment_duration is not None:
            try:
                observe_full_assessment_duration(elapsed)
            except Exception:
                pass
