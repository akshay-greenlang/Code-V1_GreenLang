# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - AGENT-MRV-029 Engine 6

This module implements regulatory compliance checking for Scope 3 category
mapping completeness against 8 regulatory frameworks.

Unlike the category-specific compliance checkers (in MRV-014 through MRV-028),
this engine validates that the *overall Scope 3 category inventory* meets
framework requirements for screening, completeness, exclusion justification,
data quality, and assurance readiness.

Regulatory Frameworks (8):
1. GHG Protocol Scope 3 Standard -- screen all 15, report material
2. ISO 14064-1:2018 -- documented methodology, limited assurance
3. CSRD / ESRS E1 -- double materiality, limited moving to reasonable
4. CDP Climate Change -- methodology documented, encouraged assurance
5. SBTi -- 67% emission coverage, mandatory screening
6. SB 253 (California) -- all material, limited then reasonable assurance
7. SEC Climate Disclosure -- material only, voluntary Scope 3
8. ISSB IFRS S2 -- all material, limited assurance

Zero-Hallucination Guarantee:
    - Framework requirements are hardcoded from official publications.
    - Compliance scoring is deterministic (no LLM/ML).
    - All findings are traceable to specific regulation references.

Example:
    >>> from greenlang.agents.mrv.scope3_category_mapper.compliance_checker import (
    ...     ComplianceCheckerEngine,
    ... )
    >>> engine = ComplianceCheckerEngine.get_instance()
    >>> assessment = engine.assess_compliance(
    ...     framework=ComplianceFramework.GHG_PROTOCOL,
    ...     company_type=CompanyType.MANUFACTURER,
    ...     categories_reported=[
    ...         Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
    ...     ],
    ...     completeness_report=completeness_report,
    ... )
    >>> print(f"Score: {assessment.score}, Status: {assessment.status}")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-040
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.agents.mrv.scope3_category_mapper.models import (
    ALL_SCOPE3_CATEGORIES,
    SCOPE3_CATEGORY_NAMES,
    SCOPE3_CATEGORY_NUMBERS,
    CategoryCompletenessEntry,
    CategoryRelevance,
    CompanyType,
    ComplianceAssessment,
    ComplianceFinding,
    ComplianceFramework,
    ComplianceSeverity,
    ComplianceStatus,
    CompletenessReport,
    DetailedComplianceAssessment,
    Scope3Category,
    ScreeningResult,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "gl_scm_compliance_checker_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-X-040"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_4DP: Decimal = Decimal("0.0001")
ROUNDING: str = ROUND_HALF_UP

# SBTi coverage threshold: must cover >= 67% of total Scope 3 emissions
SBTI_COVERAGE_THRESHOLD_PCT: Decimal = Decimal("67.00")

# SB 253 materiality threshold: include if > 1% of total Scope 3
SB253_MATERIALITY_THRESHOLD_PCT: Decimal = Decimal("1.00")

# All 15 categories count for screening
TOTAL_SCOPE3_CATEGORIES: int = 15


# ==============================================================================
# FRAMEWORK REQUIREMENTS
# ==============================================================================

FRAMEWORK_REQUIREMENTS: Dict[ComplianceFramework, Dict[str, Any]] = {
    ComplianceFramework.GHG_PROTOCOL: {
        "min_categories": 1,
        "mandatory_screening": True,
        "requires_justification_for_exclusion": True,
        "data_quality_requirement": None,
        "assurance_level": None,
        "description": "GHG Protocol Scope 3 Standard",
        "regulation_ref": "GHG Protocol Scope 3 Standard, Chapter 2-5",
    },
    ComplianceFramework.ISO_14064: {
        "min_categories": 1,
        "mandatory_screening": True,
        "requires_justification_for_exclusion": True,
        "data_quality_requirement": "documented_methodology",
        "assurance_level": "limited",
        "description": "ISO 14064-1:2018",
        "regulation_ref": "ISO 14064-1:2018, Clause 5.2",
    },
    ComplianceFramework.CSRD_ESRS: {
        "min_categories": 1,
        "mandatory_screening": True,
        "requires_justification_for_exclusion": True,
        "data_quality_requirement": "double_materiality",
        "assurance_level": "limited_moving_to_reasonable",
        "description": "CSRD / ESRS E1",
        "regulation_ref": "ESRS E1-6, Disclosure Requirement E1-6",
    },
    ComplianceFramework.CDP: {
        "min_categories": 1,
        "mandatory_screening": True,
        "requires_justification_for_exclusion": True,
        "data_quality_requirement": "methodology_documented",
        "assurance_level": "encouraged",
        "description": "CDP Climate Change",
        "regulation_ref": "CDP Climate Change Questionnaire, C6.5",
    },
    ComplianceFramework.SBTI: {
        "min_categories": 1,
        "mandatory_screening": True,
        "requires_justification_for_exclusion": True,
        "data_quality_requirement": "67_pct_coverage",
        "assurance_level": None,
        "description": "SBTi / SBTi-FI",
        "regulation_ref": "SBTi Corporate Manual v2.1, Section 6",
    },
    ComplianceFramework.SB_253: {
        "min_categories": 1,
        "mandatory_screening": True,
        "requires_justification_for_exclusion": True,
        "data_quality_requirement": "all_material",
        "assurance_level": "limited_then_reasonable",
        "description": "California SB 253",
        "regulation_ref": "SB 253 Climate Corporate Data Accountability Act",
    },
    ComplianceFramework.SEC_CLIMATE: {
        "min_categories": 0,
        "mandatory_screening": False,
        "requires_justification_for_exclusion": False,
        "data_quality_requirement": "material_only",
        "assurance_level": None,
        "description": "SEC Climate Disclosure",
        "regulation_ref": "SEC Climate-Related Disclosures, S7-10-22",
    },
}

# NOTE: The existing models.py ComplianceFramework uses EU_TAXONOMY instead
# of ISSB_S2. We map ISSB S2 requirements under EU_TAXONOMY enum value
# for forward compatibility, or handle it gracefully if the enum member
# exists.
_ISSB_S2_KEY: Optional[ComplianceFramework] = None
try:
    _ISSB_S2_KEY = ComplianceFramework("issb_s2")
except ValueError:
    # If ISSB_S2 is not in the enum, use EU_TAXONOMY as placeholder
    try:
        _ISSB_S2_KEY = ComplianceFramework.EU_TAXONOMY
    except AttributeError:
        pass

if _ISSB_S2_KEY is not None:
    FRAMEWORK_REQUIREMENTS[_ISSB_S2_KEY] = {
        "min_categories": 1,
        "mandatory_screening": True,
        "requires_justification_for_exclusion": True,
        "data_quality_requirement": "all_material",
        "assurance_level": "limited",
        "description": "ISSB IFRS S2",
        "regulation_ref": "IFRS S2 Climate-related Disclosures, para 29(a)",
    }

# Framework scoring weights (relative importance)
FRAMEWORK_WEIGHTS: Dict[ComplianceFramework, Decimal] = {
    ComplianceFramework.GHG_PROTOCOL: Decimal("1.00"),
    ComplianceFramework.ISO_14064: Decimal("0.90"),
    ComplianceFramework.CSRD_ESRS: Decimal("0.95"),
    ComplianceFramework.CDP: Decimal("0.85"),
    ComplianceFramework.SBTI: Decimal("0.90"),
    ComplianceFramework.SB_253: Decimal("0.85"),
    ComplianceFramework.SEC_CLIMATE: Decimal("0.70"),
}
if _ISSB_S2_KEY is not None:
    FRAMEWORK_WEIGHTS[_ISSB_S2_KEY] = Decimal("0.90")


# ==============================================================================
# SERIALIZATION UTILITIES
# ==============================================================================


def _serialize_for_hash(obj: Any) -> str:
    """
    Serialize an object to a deterministic JSON string for hashing.

    Args:
        obj: Object to serialize.

    Returns:
        Deterministic JSON string.
    """

    def _default_handler(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=_default_handler)


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Data to hash.

    Returns:
        Lowercase hex SHA-256 hash string.
    """
    serialized = _serialize_for_hash(data)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _severity_rank(severity: ComplianceSeverity) -> int:
    """
    Return numeric rank for severity (higher = more severe).

    Args:
        severity: ComplianceSeverity enum value.

    Returns:
        Integer rank (5=CRITICAL, 1=INFO).
    """
    rank_map: Dict[ComplianceSeverity, int] = {
        ComplianceSeverity.CRITICAL: 5,
        ComplianceSeverity.HIGH: 4,
        ComplianceSeverity.MEDIUM: 3,
        ComplianceSeverity.LOW: 2,
        ComplianceSeverity.INFO: 1,
    }
    return rank_map.get(severity, 0)


# ==============================================================================
# INTERNAL CHECK STATE ACCUMULATOR
# ==============================================================================


@dataclass
class _FrameworkCheckState:
    """
    Internal state accumulator for compliance checks against a single
    framework.

    Tracks passed/failed/warning checks and their associated findings.
    Used internally by ComplianceCheckerEngine to build assessment output.
    """

    framework: ComplianceFramework
    findings: List[ComplianceFinding] = field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    total_checks: int = 0

    def add_pass(self, rule_code: str, description: str) -> None:
        """Record a passed check."""
        self.passed_checks += 1
        self.total_checks += 1

    def add_fail(
        self,
        rule_code: str,
        description: str,
        severity: ComplianceSeverity = ComplianceSeverity.HIGH,
        recommendation: Optional[str] = None,
        regulation_reference: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a failed check with a finding."""
        self.failed_checks += 1
        self.total_checks += 1
        self.findings.append(
            ComplianceFinding(
                rule_code=rule_code,
                description=description,
                severity=severity,
                framework=self.framework.value,
                status=ComplianceStatus.FAIL,
                recommendation=recommendation,
                regulation_reference=regulation_reference,
                details=details,
            )
        )

    def add_warning(
        self,
        rule_code: str,
        description: str,
        severity: ComplianceSeverity = ComplianceSeverity.MEDIUM,
        recommendation: Optional[str] = None,
        regulation_reference: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a warning check with a finding."""
        self.warning_checks += 1
        self.total_checks += 1
        self.findings.append(
            ComplianceFinding(
                rule_code=rule_code,
                description=description,
                severity=severity,
                framework=self.framework.value,
                status=ComplianceStatus.WARNING,
                recommendation=recommendation,
                regulation_reference=regulation_reference,
                details=details,
            )
        )

    def compute_score(self) -> Decimal:
        """
        Compute compliance score (0-100).

        Penalties: each failure reduces score proportionally, each warning
        reduces at 50% of a failure weight.

        Returns:
            Decimal score clamped to [0, 100].
        """
        if self.total_checks == 0:
            return Decimal("100.00")

        penalty = (
            Decimal(str(self.failed_checks))
            + Decimal(str(self.warning_checks)) * Decimal("0.5")
        )
        max_points = Decimal(str(self.total_checks))
        score = (
            (max_points - penalty) / max_points * Decimal("100")
        ).quantize(_QUANT_2DP, rounding=ROUNDING)

        if score < Decimal("0"):
            score = Decimal("0.00")
        if score > Decimal("100"):
            score = Decimal("100.00")

        return score

    def compute_status(self) -> ComplianceStatus:
        """Compute overall status from findings."""
        if self.failed_checks > 0:
            return ComplianceStatus.FAIL
        if self.warning_checks > 0:
            return ComplianceStatus.WARNING
        return ComplianceStatus.PASS

    def to_detailed_assessment(
        self,
        provenance_hash: str,
        processing_time_ms: float,
    ) -> DetailedComplianceAssessment:
        """
        Convert accumulated state to a DetailedComplianceAssessment.

        Args:
            provenance_hash: SHA-256 provenance hash.
            processing_time_ms: Processing time in milliseconds.

        Returns:
            DetailedComplianceAssessment Pydantic model.
        """
        fw_req = FRAMEWORK_REQUIREMENTS.get(self.framework, {})
        recommendations = [
            f.recommendation
            for f in self.findings
            if f.recommendation is not None
        ]

        return DetailedComplianceAssessment(
            framework=self.framework,
            framework_description=fw_req.get("description", ""),
            status=self.compute_status(),
            score=self.compute_score(),
            findings=list(self.findings),
            recommendations=recommendations,
            passed_checks=self.passed_checks,
            failed_checks=self.failed_checks,
            warning_checks=self.warning_checks,
            total_checks=self.total_checks,
            provenance_hash=provenance_hash,
            assessed_at=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=round(processing_time_ms, 2),
        )


# ==============================================================================
# ComplianceCheckerEngine
# ==============================================================================


class ComplianceCheckerEngine:
    """
    ComplianceCheckerEngine - validates category mapping compliance.

    This engine validates that a company's Scope 3 category inventory meets
    regulatory requirements for 8 compliance frameworks. It checks screening
    completeness, exclusion justifications, data quality requirements, and
    assurance readiness.

    All framework requirements are hardcoded from official regulatory
    publications. No LLM or ML models are used (zero-hallucination guarantee).

    Thread-Safe: Singleton pattern with lock for concurrent access.

    Attributes:
        _instance: Singleton instance.
        _lock: Thread lock for singleton creation.

    Example:
        >>> engine = ComplianceCheckerEngine.get_instance()
        >>> assessment = engine.assess_compliance(
        ...     framework=ComplianceFramework.GHG_PROTOCOL,
        ...     company_type=CompanyType.MANUFACTURER,
        ...     categories_reported=[
        ...         Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        ...     ],
        ...     completeness_report=report,
        ... )
        >>> print(f"{assessment.framework}: {assessment.status}")
    """

    _instance: Optional["ComplianceCheckerEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize ComplianceCheckerEngine."""
        logger.info(
            "ComplianceCheckerEngine initialized (version=%s, frameworks=%d)",
            ENGINE_VERSION,
            len(FRAMEWORK_REQUIREMENTS),
        )

    @classmethod
    def get_instance(cls) -> "ComplianceCheckerEngine":
        """
        Get singleton instance of ComplianceCheckerEngine (thread-safe).

        Returns:
            Singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    # =========================================================================
    # MAIN ENTRY POINTS
    # =========================================================================

    def assess_compliance(
        self,
        framework: ComplianceFramework,
        company_type: CompanyType,
        categories_reported: List[Scope3Category],
        completeness_report: CompletenessReport,
        justifications: Optional[Dict[Scope3Category, str]] = None,
        emission_pcts: Optional[Dict[Scope3Category, Decimal]] = None,
    ) -> DetailedComplianceAssessment:
        """
        Assess compliance against a single framework.

        Runs all compliance checks for the specified framework, including
        screening requirements, category coverage, exclusion justifications,
        data quality, and assurance readiness.

        Args:
            framework: Compliance framework to assess.
            company_type: Company type for relevance lookup.
            categories_reported: Categories with reported data.
            completeness_report: Completeness screening report.
            justifications: Optional exclusion justifications by category.
            emission_pcts: Optional emission percentages by category
                (for SBTi 67% coverage check).

        Returns:
            DetailedComplianceAssessment with findings, score, and
            recommendations.
        """
        start_time = time.monotonic()
        logger.info(
            "Assessing compliance: framework=%s, company_type=%s, "
            "categories=%d",
            framework.value,
            company_type.value,
            len(categories_reported),
        )

        if justifications is None:
            justifications = {}
        if emission_pcts is None:
            emission_pcts = {}

        fw_req = FRAMEWORK_REQUIREMENTS.get(framework)
        if fw_req is None:
            raise ValueError(
                f"Unknown compliance framework: {framework.value}"
            )

        state = _FrameworkCheckState(framework=framework)

        # Run standard checks
        self._check_screening(state, fw_req, completeness_report)
        self._check_minimum_categories(
            state, fw_req, categories_reported
        )
        self._check_material_coverage(
            state, fw_req, completeness_report
        )
        self._check_exclusion_justifications(
            state, fw_req, completeness_report, justifications
        )
        self._check_data_quality(state, fw_req, completeness_report)
        self._check_assurance(state, fw_req, completeness_report)

        # Framework-specific checks
        if framework == ComplianceFramework.SBTI:
            self._check_sbti_specific(
                state, categories_reported, emission_pcts
            )
        elif framework == ComplianceFramework.SB_253:
            self._check_sb253_specific(state, completeness_report)
        elif framework == ComplianceFramework.SEC_CLIMATE:
            self._check_sec_specific(state, completeness_report)
        elif framework == ComplianceFramework.CSRD_ESRS:
            self._check_csrd_specific(state, completeness_report)
        elif _ISSB_S2_KEY is not None and framework == _ISSB_S2_KEY:
            self._check_issb_specific(state, completeness_report)

        processing_time = (time.monotonic() - start_time) * 1000

        provenance_hash = self._compute_provenance_hash({
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "framework": framework.value,
            "company_type": company_type.value,
            "categories_reported": [c.value for c in categories_reported],
            "score": str(state.compute_score()),
            "status": state.compute_status().value,
        })

        assessment = state.to_detailed_assessment(
            provenance_hash, processing_time
        )

        logger.info(
            "Compliance assessment done: framework=%s, status=%s, "
            "score=%s, checks=%d, time=%.1fms",
            framework.value,
            assessment.status.value,
            assessment.score,
            assessment.total_checks,
            processing_time,
        )
        return assessment

    def assess_all_frameworks(
        self,
        company_type: CompanyType,
        categories_reported: List[Scope3Category],
        completeness_report: CompletenessReport,
        justifications: Optional[Dict[Scope3Category, str]] = None,
        emission_pcts: Optional[Dict[Scope3Category, Decimal]] = None,
    ) -> List[DetailedComplianceAssessment]:
        """
        Assess compliance against all supported regulatory frameworks.

        Args:
            company_type: Company type for relevance lookup.
            categories_reported: Categories with reported data.
            completeness_report: Completeness screening report.
            justifications: Optional exclusion justifications by category.
            emission_pcts: Optional emission percentages by category.

        Returns:
            List of DetailedComplianceAssessment, one per framework.
        """
        logger.info(
            "Assessing all frameworks: company_type=%s, categories=%d",
            company_type.value,
            len(categories_reported),
        )

        assessments: List[DetailedComplianceAssessment] = []
        for framework in FRAMEWORK_REQUIREMENTS:
            try:
                assessment = self.assess_compliance(
                    framework=framework,
                    company_type=company_type,
                    categories_reported=categories_reported,
                    completeness_report=completeness_report,
                    justifications=justifications,
                    emission_pcts=emission_pcts,
                )
                assessments.append(assessment)
            except Exception as e:
                logger.error(
                    "Error assessing %s: %s",
                    framework.value,
                    str(e),
                    exc_info=True,
                )
                assessments.append(
                    DetailedComplianceAssessment(
                        framework=framework,
                        framework_description=FRAMEWORK_REQUIREMENTS.get(
                            framework, {}
                        ).get("description", ""),
                        status=ComplianceStatus.FAIL,
                        score=Decimal("0.00"),
                        findings=[
                            ComplianceFinding(
                                rule_code="CHECK_ERROR",
                                description=(
                                    f"Compliance check failed: {str(e)}"
                                ),
                                severity=ComplianceSeverity.CRITICAL,
                                framework=framework.value,
                                status=ComplianceStatus.FAIL,
                            )
                        ],
                        assessed_at=datetime.now(timezone.utc).isoformat(),
                    )
                )

        return assessments

    # =========================================================================
    # PUBLIC UTILITY METHODS
    # =========================================================================

    def check_screening_requirement(
        self,
        framework: ComplianceFramework,
        categories_screened: int,
    ) -> bool:
        """
        Check if the screening requirement is met for a framework.

        Args:
            framework: Compliance framework.
            categories_screened: Number of categories that were screened.

        Returns:
            True if screening requirement is satisfied.
        """
        fw_req = FRAMEWORK_REQUIREMENTS.get(framework)
        if fw_req is None:
            raise ValueError(
                f"Unknown compliance framework: {framework.value}"
            )

        if not fw_req["mandatory_screening"]:
            return True

        return categories_screened >= TOTAL_SCOPE3_CATEGORIES

    def check_exclusion_justification(
        self,
        framework: ComplianceFramework,
        excluded_categories: List[Scope3Category],
        justifications: Dict[Scope3Category, str],
    ) -> List[str]:
        """
        Check that exclusion justifications are provided where required.

        Args:
            framework: Compliance framework.
            excluded_categories: Categories that were excluded.
            justifications: Justification text by category.

        Returns:
            List of violation messages for categories missing justification.
        """
        fw_req = FRAMEWORK_REQUIREMENTS.get(framework)
        if fw_req is None:
            raise ValueError(
                f"Unknown compliance framework: {framework.value}"
            )

        violations: List[str] = []

        if not fw_req["requires_justification_for_exclusion"]:
            return violations

        for cat in excluded_categories:
            justification = justifications.get(cat, "").strip()
            if not justification:
                cat_name = SCOPE3_CATEGORY_NAMES.get(cat, cat.value)
                cat_num = SCOPE3_CATEGORY_NUMBERS.get(cat, 0)
                violations.append(
                    f"Category {cat_num} ({cat_name}) excluded without "
                    f"justification. {framework.value} requires documented "
                    f"reasoning for all excluded categories."
                )

        return violations

    def calculate_compliance_score(
        self,
        assessment: DetailedComplianceAssessment,
    ) -> Decimal:
        """
        Calculate or recalculate compliance score for an assessment.

        This is a convenience method that recalculates from check counts
        for verification.

        Args:
            assessment: Compliance assessment.

        Returns:
            Decimal score in range [0, 100].
        """
        total = assessment.total_checks
        if total == 0:
            return Decimal("100.00")

        penalty = (
            Decimal(str(assessment.failed_checks))
            + Decimal(str(assessment.warning_checks)) * Decimal("0.5")
        )
        max_points = Decimal(str(total))
        score = (
            (max_points - penalty) / max_points * Decimal("100")
        ).quantize(_QUANT_2DP, rounding=ROUNDING)

        if score < Decimal("0"):
            score = Decimal("0.00")
        if score > Decimal("100"):
            score = Decimal("100.00")

        return score

    def get_framework_requirements(
        self,
        framework: ComplianceFramework,
    ) -> Dict[str, Any]:
        """
        Get the requirements for a specific framework.

        Args:
            framework: Compliance framework.

        Returns:
            Dictionary of framework requirements.

        Raises:
            ValueError: If framework is not recognized.
        """
        if framework not in FRAMEWORK_REQUIREMENTS:
            raise ValueError(
                f"Unknown compliance framework: {framework.value}"
            )
        return dict(FRAMEWORK_REQUIREMENTS[framework])

    def get_improvement_recommendations(
        self,
        assessments: List[DetailedComplianceAssessment],
    ) -> List[str]:
        """
        Generate cross-framework improvement recommendations.

        Analyzes all framework assessments and produces a deduplicated,
        priority-ordered list of recommendations.

        Args:
            assessments: List of DetailedComplianceAssessment from
                multiple frameworks.

        Returns:
            Priority-ordered list of recommendation strings.
        """
        rec_priority: Dict[str, ComplianceSeverity] = {}

        for assessment in assessments:
            for finding in assessment.findings:
                if finding.recommendation:
                    existing = rec_priority.get(finding.recommendation)
                    if (
                        existing is None
                        or _severity_rank(finding.severity)
                        > _severity_rank(existing)
                    ):
                        rec_priority[finding.recommendation] = (
                            finding.severity
                        )

        sorted_recs = sorted(
            rec_priority.items(),
            key=lambda kv: _severity_rank(kv[1]),
            reverse=True,
        )

        result: List[str] = []
        for rec_text, severity in sorted_recs:
            result.append(f"[{severity.value}] {rec_text}")

        avg_score = self._compute_average_score(assessments)
        if avg_score < Decimal("50"):
            result.append(
                "[GENERAL] Overall compliance is below 50%. "
                "Conduct a comprehensive Scope 3 screening exercise "
                "across all 15 categories as a priority."
            )
        elif avg_score < Decimal("70"):
            result.append(
                "[GENERAL] Overall compliance is moderate. "
                "Focus on filling material category gaps and "
                "documenting exclusion justifications."
            )
        elif avg_score < Decimal("90"):
            result.append(
                "[GENERAL] Good compliance foundation. "
                "Improve data quality from spend-based to "
                "activity-based methods for highest-impact categories."
            )

        return result

    def check_sbti_coverage(
        self,
        categories_reported: List[Scope3Category],
        emission_pcts: Dict[Scope3Category, Decimal],
    ) -> bool:
        """
        Check SBTi 67% Scope 3 emission coverage requirement.

        SBTi requires that reported categories cover at least 67% of
        total Scope 3 emissions (by mass).

        Args:
            categories_reported: Categories with reported data.
            emission_pcts: Percentage of total Scope 3 by category.

        Returns:
            True if reported categories cover >= 67% of total Scope 3.
        """
        reported_set = set(categories_reported)
        total_covered = Decimal("0.00")

        for cat, pct in emission_pcts.items():
            if cat in reported_set:
                total_covered += pct

        result = total_covered >= SBTI_COVERAGE_THRESHOLD_PCT
        logger.info(
            "SBTi coverage check: covered=%s%%, threshold=%s%%, pass=%s",
            total_covered.quantize(_QUANT_2DP, rounding=ROUNDING),
            SBTI_COVERAGE_THRESHOLD_PCT,
            result,
        )
        return result

    # =========================================================================
    # PROVENANCE
    # =========================================================================

    def _compute_provenance_hash(self, data: Any) -> str:
        """
        Compute SHA-256 provenance hash for audit trail.

        Args:
            data: Data to hash.

        Returns:
            Lowercase hex SHA-256 hash string.
        """
        return _compute_hash(data)

    # =========================================================================
    # INTERNAL CHECK METHODS
    # =========================================================================

    def _check_screening(
        self,
        state: _FrameworkCheckState,
        fw_req: Dict[str, Any],
        report: CompletenessReport,
    ) -> None:
        """
        Check if all 15 categories have been screened.

        Args:
            state: Accumulator for check results.
            fw_req: Framework requirements dict.
            report: Completeness report.
        """
        rule_code = f"{state.framework.value.upper()}-SCR-001"
        regulation_ref = fw_req.get("regulation_ref", "")

        if not fw_req["mandatory_screening"]:
            state.add_pass(
                rule_code,
                "Screening not mandatory for this framework",
            )
            return

        screened_count = len(report.entries)
        if screened_count >= TOTAL_SCOPE3_CATEGORIES:
            state.add_pass(
                rule_code,
                f"All {TOTAL_SCOPE3_CATEGORIES} categories screened",
            )
        else:
            state.add_fail(
                rule_code=rule_code,
                description=(
                    f"Only {screened_count} of {TOTAL_SCOPE3_CATEGORIES} "
                    f"categories screened. Framework requires screening "
                    f"of all 15 Scope 3 categories."
                ),
                severity=ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Complete screening of all 15 Scope 3 categories "
                    "using the GHG Protocol category relevance assessment."
                ),
                regulation_reference=regulation_ref,
            )

    def _check_minimum_categories(
        self,
        state: _FrameworkCheckState,
        fw_req: Dict[str, Any],
        categories_reported: List[Scope3Category],
    ) -> None:
        """
        Check minimum number of reported categories.

        Args:
            state: Accumulator for check results.
            fw_req: Framework requirements dict.
            categories_reported: Reported categories.
        """
        rule_code = f"{state.framework.value.upper()}-MIN-001"
        min_required = fw_req["min_categories"]
        actual = len(set(categories_reported))

        if actual >= min_required:
            state.add_pass(
                rule_code,
                f"Reported {actual} categories (minimum: {min_required})",
            )
        else:
            state.add_fail(
                rule_code=rule_code,
                description=(
                    f"Only {actual} categories reported. "
                    f"Framework requires at least {min_required}."
                ),
                severity=ComplianceSeverity.HIGH,
                recommendation=(
                    f"Report data for at least {min_required} Scope 3 "
                    f"categories. Start with the most material categories."
                ),
                regulation_reference=fw_req.get("regulation_ref", ""),
            )

    def _check_material_coverage(
        self,
        state: _FrameworkCheckState,
        fw_req: Dict[str, Any],
        report: CompletenessReport,
    ) -> None:
        """
        Check that all material categories are reported.

        Args:
            state: Accumulator for check results.
            fw_req: Framework requirements dict.
            report: Completeness report.
        """
        rule_code = f"{state.framework.value.upper()}-MAT-001"

        if report.categories_material == 0:
            state.add_pass(
                rule_code,
                "No material categories identified",
            )
            return

        # Count material categories that are reported
        material_reported = sum(
            1 for e in report.entries
            if e.relevance == CategoryRelevance.MATERIAL and e.data_available
        )

        if material_reported >= report.categories_material:
            state.add_pass(
                rule_code,
                f"All {report.categories_material} material categories "
                f"reported",
            )
        else:
            missing = report.categories_material - material_reported
            missing_cats = [
                e for e in report.entries
                if e.relevance == CategoryRelevance.MATERIAL
                and not e.data_available
            ]
            missing_names = [
                f"Cat {SCOPE3_CATEGORY_NUMBERS.get(e.category, '?')} "
                f"({SCOPE3_CATEGORY_NAMES.get(e.category, e.category.value)})"
                for e in missing_cats
            ]
            state.add_fail(
                rule_code=rule_code,
                description=(
                    f"{missing} material category(ies) not reported: "
                    f"{', '.join(missing_names)}."
                ),
                severity=ComplianceSeverity.CRITICAL,
                recommendation=(
                    f"Collect data for missing material categories: "
                    f"{', '.join(missing_names)}. These are expected to "
                    f"represent a significant portion of total Scope 3."
                ),
                regulation_reference=fw_req.get("regulation_ref", ""),
                details={
                    "missing_material_categories": [
                        e.category.value for e in missing_cats
                    ]
                },
            )

    def _check_exclusion_justifications(
        self,
        state: _FrameworkCheckState,
        fw_req: Dict[str, Any],
        report: CompletenessReport,
        justifications: Dict[Scope3Category, str],
    ) -> None:
        """
        Check that excluded categories have justification where required.

        Args:
            state: Accumulator for check results.
            fw_req: Framework requirements dict.
            report: Completeness report.
            justifications: Justification text by excluded category.
        """
        rule_code = f"{state.framework.value.upper()}-EXC-001"

        if not fw_req["requires_justification_for_exclusion"]:
            state.add_pass(
                rule_code,
                "Exclusion justification not required by this framework",
            )
            return

        excluded_needing_just: List[Scope3Category] = []
        for entry in report.entries:
            if (
                not entry.data_available
                and entry.relevance in (
                    CategoryRelevance.MATERIAL,
                    CategoryRelevance.RELEVANT,
                )
            ):
                excluded_needing_just.append(entry.category)

        if not excluded_needing_just:
            state.add_pass(
                rule_code,
                "No excluded relevant/material categories requiring "
                "justification",
            )
            return

        missing_just: List[str] = []
        for cat in excluded_needing_just:
            justification = justifications.get(cat, "").strip()
            if not justification:
                cat_name = SCOPE3_CATEGORY_NAMES.get(cat, cat.value)
                cat_num = SCOPE3_CATEGORY_NUMBERS.get(cat, 0)
                missing_just.append(f"Cat {cat_num} ({cat_name})")

        if missing_just:
            state.add_fail(
                rule_code=rule_code,
                description=(
                    f"Missing exclusion justification for: "
                    f"{', '.join(missing_just)}."
                ),
                severity=ComplianceSeverity.HIGH,
                recommendation=(
                    "Document justification for excluding each relevant "
                    "or material category. Justifications must explain "
                    "why the category is not relevant or data is not "
                    "available."
                ),
                regulation_reference=fw_req.get("regulation_ref", ""),
                details={
                    "categories_missing_justification": missing_just
                },
            )
        else:
            state.add_pass(
                rule_code,
                "All excluded categories have documented justification",
            )

    def _check_data_quality(
        self,
        state: _FrameworkCheckState,
        fw_req: Dict[str, Any],
        report: CompletenessReport,
    ) -> None:
        """
        Check data quality requirements for the framework.

        Args:
            state: Accumulator for check results.
            fw_req: Framework requirements dict.
            report: Completeness report.
        """
        rule_code = f"{state.framework.value.upper()}-DQ-001"
        dq_req = fw_req.get("data_quality_requirement")

        if dq_req is None:
            state.add_pass(
                rule_code,
                "No specific data quality requirement for this framework",
            )
            return

        partial_entries = [
            e for e in report.entries
            if e.data_available
            and e.screening_result == ScreeningResult.PARTIAL
        ]

        if partial_entries:
            partial_names = [
                f"Cat {SCOPE3_CATEGORY_NUMBERS.get(e.category, '?')} "
                f"({SCOPE3_CATEGORY_NAMES.get(e.category, e.category.value)})"
                for e in partial_entries
            ]
            state.add_warning(
                rule_code=rule_code,
                description=(
                    f"Data quality concerns in {len(partial_entries)} "
                    f"categories: {', '.join(partial_names)}. "
                    f"Framework requires: {dq_req}."
                ),
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Improve data quality for flagged categories by "
                    "upgrading from estimated/spend-based to activity-based "
                    "or supplier-specific methods."
                ),
                regulation_reference=fw_req.get("regulation_ref", ""),
            )
        else:
            state.add_pass(
                rule_code,
                f"Data quality meets {dq_req} requirement",
            )

    def _check_assurance(
        self,
        state: _FrameworkCheckState,
        fw_req: Dict[str, Any],
        report: CompletenessReport,
    ) -> None:
        """
        Check assurance readiness requirements.

        Args:
            state: Accumulator for check results.
            fw_req: Framework requirements dict.
            report: Completeness report.
        """
        rule_code = f"{state.framework.value.upper()}-ASR-001"
        assurance_req = fw_req.get("assurance_level")

        if assurance_req is None:
            state.add_pass(
                rule_code,
                "No assurance requirement for this framework",
            )
            return

        material_reported = sum(
            1 for e in report.entries
            if e.relevance == CategoryRelevance.MATERIAL and e.data_available
        )
        is_ready = (
            report.overall_score >= Decimal("80")
            and material_reported == report.categories_material
        )

        if is_ready:
            state.add_pass(
                rule_code,
                f"Data appears ready for {assurance_req} assurance",
            )
        else:
            material_gap = report.categories_material - material_reported
            state.add_warning(
                rule_code=rule_code,
                description=(
                    f"Data may not be ready for {assurance_req} assurance. "
                    f"Completeness score is {report.overall_score}% "
                    f"and {material_gap} material categories are missing."
                ),
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    f"Improve data completeness to support {assurance_req} "
                    f"assurance engagement. Target completeness score "
                    f"of 80%+."
                ),
                regulation_reference=fw_req.get("regulation_ref", ""),
            )

    # =========================================================================
    # FRAMEWORK-SPECIFIC CHECKS
    # =========================================================================

    def _check_sbti_specific(
        self,
        state: _FrameworkCheckState,
        categories_reported: List[Scope3Category],
        emission_pcts: Dict[Scope3Category, Decimal],
    ) -> None:
        """
        SBTi-specific: check 67% Scope 3 emission coverage.

        Args:
            state: Accumulator for check results.
            categories_reported: Reported categories.
            emission_pcts: Emission percentages by category.
        """
        rule_code = "SBTI-COV-001"

        if not emission_pcts:
            state.add_warning(
                rule_code=rule_code,
                description=(
                    "Cannot verify SBTi 67% coverage requirement: "
                    "no emission percentages provided."
                ),
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Provide emission percentage breakdown by category "
                    "to verify SBTi 67% Scope 3 coverage threshold."
                ),
                regulation_reference=(
                    "SBTi Corporate Manual v2.1, Section 6"
                ),
            )
            return

        covered = self.check_sbti_coverage(
            categories_reported, emission_pcts
        )
        reported_set = set(categories_reported)
        total_pct = sum(
            pct for cat, pct in emission_pcts.items()
            if cat in reported_set
        )

        if covered:
            state.add_pass(
                rule_code,
                f"SBTi 67% coverage met: "
                f"{total_pct.quantize(_QUANT_2DP, rounding=ROUNDING)}% "
                f"covered",
            )
        else:
            state.add_fail(
                rule_code=rule_code,
                description=(
                    f"SBTi 67% coverage NOT met: only "
                    f"{total_pct.quantize(_QUANT_2DP, rounding=ROUNDING)}% "
                    f"of Scope 3 emissions covered by reported categories."
                ),
                severity=ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report additional high-impact Scope 3 categories "
                    "to reach 67% coverage threshold required by SBTi."
                ),
                regulation_reference=(
                    "SBTi Corporate Manual v2.1, Section 6"
                ),
                details={
                    "covered_pct": str(
                        total_pct.quantize(_QUANT_2DP, rounding=ROUNDING)
                    ),
                    "threshold_pct": str(SBTI_COVERAGE_THRESHOLD_PCT),
                },
            )

    def _check_sb253_specific(
        self,
        state: _FrameworkCheckState,
        report: CompletenessReport,
    ) -> None:
        """
        SB 253-specific: all material categories must be reported.

        Args:
            state: Accumulator for check results.
            report: Completeness report.
        """
        rule_code = "SB253-MAT-001"
        material_reported = sum(
            1 for e in report.entries
            if e.relevance == CategoryRelevance.MATERIAL and e.data_available
        )

        if material_reported >= report.categories_material:
            state.add_pass(
                rule_code,
                "All material categories reported (SB 253 compliance)",
            )
        else:
            gap = report.categories_material - material_reported
            state.add_fail(
                rule_code=rule_code,
                description=(
                    f"SB 253 requires all material categories. "
                    f"{gap} material categories are missing."
                ),
                severity=ComplianceSeverity.CRITICAL,
                recommendation=(
                    "SB 253 mandates reporting of all material Scope 3 "
                    "categories with third-party assurance. Address all "
                    "material category gaps immediately."
                ),
                regulation_reference=(
                    "SB 253 Climate Corporate Data Accountability Act"
                ),
            )

    def _check_sec_specific(
        self,
        state: _FrameworkCheckState,
        report: CompletenessReport,
    ) -> None:
        """
        SEC Climate Disclosure-specific checks.

        Args:
            state: Accumulator for check results.
            report: Completeness report.
        """
        rule_code = "SEC-MAT-001"

        if report.categories_reported > 0:
            state.add_pass(
                rule_code,
                f"Scope 3 data reported for "
                f"{report.categories_reported} categories",
            )
        else:
            state.add_warning(
                rule_code=rule_code,
                description=(
                    "No Scope 3 categories reported. If the registrant "
                    "has set a GHG reduction target that includes Scope 3, "
                    "material categories must be disclosed."
                ),
                severity=ComplianceSeverity.LOW,
                recommendation=(
                    "Consider disclosing material Scope 3 categories if "
                    "your organization has set GHG reduction targets."
                ),
                regulation_reference=(
                    "SEC Climate-Related Disclosures, S7-10-22"
                ),
            )

    def _check_csrd_specific(
        self,
        state: _FrameworkCheckState,
        report: CompletenessReport,
    ) -> None:
        """
        CSRD/ESRS E1-specific: double materiality perspective.

        Args:
            state: Accumulator for check results.
            report: Completeness report.
        """
        rule_code = "CSRD-DM-001"

        total_material_relevant = sum(
            1 for e in report.entries
            if e.relevance in (
                CategoryRelevance.MATERIAL,
                CategoryRelevance.RELEVANT,
            )
        )
        reported_material_relevant = sum(
            1 for e in report.entries
            if e.data_available and e.relevance in (
                CategoryRelevance.MATERIAL,
                CategoryRelevance.RELEVANT,
            )
        )

        if reported_material_relevant >= total_material_relevant:
            state.add_pass(
                rule_code,
                "All material and relevant categories reported "
                "(CSRD/ESRS E1)",
            )
        else:
            gap = total_material_relevant - reported_material_relevant
            state.add_warning(
                rule_code=rule_code,
                description=(
                    f"CSRD/ESRS E1 requires reporting of all categories "
                    f"identified as material in double materiality "
                    f"assessment. {gap} material/relevant categories are "
                    f"not yet reported."
                ),
                severity=ComplianceSeverity.HIGH,
                recommendation=(
                    "Complete double materiality assessment for Scope 3 "
                    "categories and report all categories identified as "
                    "material from both impact and financial perspectives."
                ),
                regulation_reference=(
                    "ESRS E1-6, Disclosure Requirement E1-6"
                ),
            )

    def _check_issb_specific(
        self,
        state: _FrameworkCheckState,
        report: CompletenessReport,
    ) -> None:
        """
        ISSB S2-specific: all material categories with limited assurance.

        Args:
            state: Accumulator for check results.
            report: Completeness report.
        """
        rule_code = "ISSB-MAT-001"
        material_reported = sum(
            1 for e in report.entries
            if e.relevance == CategoryRelevance.MATERIAL and e.data_available
        )

        if material_reported >= report.categories_material:
            state.add_pass(
                rule_code,
                "All material categories reported (ISSB S2 compliance)",
            )
        else:
            gap = report.categories_material - material_reported
            state.add_fail(
                rule_code=rule_code,
                description=(
                    f"ISSB S2 requires all material Scope 3 categories. "
                    f"{gap} material categories are missing."
                ),
                severity=ComplianceSeverity.HIGH,
                recommendation=(
                    "Report all material Scope 3 categories as required "
                    "by IFRS S2 paragraph 29(a). Limited assurance will "
                    "be required."
                ),
                regulation_reference=(
                    "IFRS S2 Climate-related Disclosures, para 29(a)"
                ),
            )

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _compute_average_score(
        self,
        assessments: List[DetailedComplianceAssessment],
    ) -> Decimal:
        """
        Compute weighted average compliance score across assessments.

        Args:
            assessments: List of compliance assessments.

        Returns:
            Weighted average score (0-100).
        """
        if not assessments:
            return Decimal("0.00")

        total_weighted = Decimal("0")
        total_weight = Decimal("0")

        for assessment in assessments:
            weight = FRAMEWORK_WEIGHTS.get(
                assessment.framework, Decimal("1.00")
            )
            total_weighted += assessment.score * weight
            total_weight += weight

        if total_weight == 0:
            return Decimal("0.00")

        return (total_weighted / total_weight).quantize(
            _QUANT_2DP, rounding=ROUNDING
        )
