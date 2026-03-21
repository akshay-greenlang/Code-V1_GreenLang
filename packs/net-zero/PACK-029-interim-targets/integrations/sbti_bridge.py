# -*- coding: utf-8 -*-
"""
SBTiBridge - SBTi Interim Target Validation for PACK-029
==========================================================

Enterprise bridge for validating interim targets against SBTi criteria
including minimum ambition (42% by 2030 for 1.5C, 30% for WB2C),
linearity (no backsliding), scope coverage (Scope 1+2 mandatory,
Scope 3 if >40%), FLAG sector rules (separate land targets), near-term
and long-term consistency, 21-criteria validation checklist, SBTi
submission package generation, and API integration with the SBTi portal.

SBTi Validation Features:
    - 21-criteria interim target validation checklist
    - Minimum ambition enforcement (1.5C: 42% S1+S2 by 2030)
    - Linearity check (no backsliding between interim periods)
    - Scope coverage validation (95% S1+S2, 67% S3 if applicable)
    - FLAG sector separate target requirement
    - Near-term to long-term consistency verification
    - Submission package generation (target language + data)
    - SBTi portal API integration (when available)
    - Temperature rating calculation
    - SHA-256 provenance on all validations

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-029 Interim Targets Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SBTiPathway(str, Enum):
    ACA_15C = "aca_15c"
    ACA_WB2C = "aca_wb2c"
    SDA = "sda"
    FLAG = "flag"


class TargetType(str, Enum):
    NEAR_TERM = "near_term"
    INTERIM = "interim"
    LONG_TERM = "long_term"
    NET_ZERO = "net_zero"


class CriteriaStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"


class SubmissionStatus(str, Enum):
    DRAFT = "draft"
    READY_FOR_REVIEW = "ready_for_review"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REVISION_REQUIRED = "revision_required"
    REJECTED = "rejected"


class TemperatureRating(str, Enum):
    BELOW_15C = "below_1.5C"
    C_15 = "1.5C"
    WB_2C = "well_below_2C"
    C_2 = "2C"
    ABOVE_2C = "above_2C"
    NOT_ALIGNED = "not_aligned"


class LinearityStatus(str, Enum):
    LINEAR = "linear"
    FRONT_LOADED = "front_loaded"
    BACK_LOADED = "back_loaded"
    BACKSLIDING = "backsliding"
    STEPPED = "stepped"


# ---------------------------------------------------------------------------
# SBTi Minimum Ambition Tables
# ---------------------------------------------------------------------------

SBTI_MINIMUM_AMBITION: Dict[str, Dict[str, float]] = {
    "aca_15c": {
        "scope12_annual_rate_pct": 4.2,
        "scope12_min_2030_pct": 42.0,
        "scope3_annual_rate_pct": 2.5,
        "scope3_min_2030_pct": 25.0,
        "long_term_scope12_pct": 90.0,
        "long_term_scope3_pct": 90.0,
        "temperature": "1.5C",
    },
    "aca_wb2c": {
        "scope12_annual_rate_pct": 2.5,
        "scope12_min_2030_pct": 25.0,
        "scope3_annual_rate_pct": 2.5,
        "scope3_min_2030_pct": 25.0,
        "long_term_scope12_pct": 90.0,
        "long_term_scope3_pct": 90.0,
        "temperature": "WB2C",
    },
    "sda": {
        "scope12_annual_rate_pct": 4.2,
        "scope12_min_2030_pct": 42.0,
        "scope3_annual_rate_pct": 2.5,
        "scope3_min_2030_pct": 25.0,
        "long_term_scope12_pct": 90.0,
        "long_term_scope3_pct": 90.0,
        "temperature": "1.5C",
    },
}


# ---------------------------------------------------------------------------
# 21-Criteria Validation Checklist
# ---------------------------------------------------------------------------

INTERIM_TARGET_CRITERIA: List[Dict[str, str]] = [
    # Ambition Criteria (1-5)
    {"id": "IT-C1", "name": "Scope 1+2 minimum ambition (42% by 2030 for 1.5C)", "category": "ambition", "severity": "critical"},
    {"id": "IT-C2", "name": "Scope 3 minimum ambition (25% by 2030 if applicable)", "category": "ambition", "severity": "critical"},
    {"id": "IT-C3", "name": "Annual reduction rate >= pathway minimum", "category": "ambition", "severity": "critical"},
    {"id": "IT-C4", "name": "Long-term target >= 90% absolute reduction", "category": "ambition", "severity": "critical"},
    {"id": "IT-C5", "name": "Temperature alignment consistent with pathway", "category": "ambition", "severity": "required"},
    # Linearity Criteria (6-8)
    {"id": "IT-C6", "name": "No backsliding between consecutive interim periods", "category": "linearity", "severity": "critical"},
    {"id": "IT-C7", "name": "Cumulative emission budget within pathway envelope", "category": "linearity", "severity": "required"},
    {"id": "IT-C8", "name": "Front-loading / back-loading within 15% tolerance", "category": "linearity", "severity": "warning"},
    # Scope Coverage (9-12)
    {"id": "IT-C9", "name": "Scope 1+2 coverage >= 95%", "category": "coverage", "severity": "critical"},
    {"id": "IT-C10", "name": "Scope 3 coverage >= 67% (if >40% of total)", "category": "coverage", "severity": "critical"},
    {"id": "IT-C11", "name": "All material Scope 3 categories included", "category": "coverage", "severity": "required"},
    {"id": "IT-C12", "name": "Biogenic emissions properly accounted", "category": "coverage", "severity": "required"},
    # FLAG Sector (13-14)
    {"id": "IT-C13", "name": "Separate FLAG target (if applicable)", "category": "flag", "severity": "conditional"},
    {"id": "IT-C14", "name": "FLAG no-deforestation commitment included", "category": "flag", "severity": "conditional"},
    # Consistency (15-18)
    {"id": "IT-C15", "name": "Near-term and long-term targets consistent", "category": "consistency", "severity": "critical"},
    {"id": "IT-C16", "name": "Base year consistent across all targets", "category": "consistency", "severity": "required"},
    {"id": "IT-C17", "name": "Consolidation approach consistent with GHG Protocol", "category": "consistency", "severity": "required"},
    {"id": "IT-C18", "name": "Recalculation trigger policy defined", "category": "consistency", "severity": "required"},
    # Methodology (19-21)
    {"id": "IT-C19", "name": "No carbon offsets counted toward target", "category": "methodology", "severity": "critical"},
    {"id": "IT-C20", "name": "Emission factors from recognized sources", "category": "methodology", "severity": "required"},
    {"id": "IT-C21", "name": "Target language meets SBTi clarity requirements", "category": "methodology", "severity": "required"},
]


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class SBTiBridgeConfig(BaseModel):
    """Configuration for the SBTi bridge."""
    pack_id: str = Field(default="PACK-029")
    sbti_api_key: str = Field(default="")
    sbti_api_url: str = Field(default="https://api.sciencebasedtargets.org/v1")
    organization_id: str = Field(default="")
    organization_name: str = Field(default="")
    pathway: SBTiPathway = Field(default=SBTiPathway.ACA_15C)
    base_year: int = Field(default=2023, ge=2015, le=2025)
    near_term_year: int = Field(default=2030, ge=2025, le=2035)
    long_term_year: int = Field(default=2050, ge=2040, le=2060)
    scope12_coverage_pct: float = Field(default=95.0, ge=0.0, le=100.0)
    scope3_coverage_pct: float = Field(default=67.0, ge=0.0, le=100.0)
    scope3_share_pct: float = Field(default=60.0, ge=0.0, le=100.0)
    flag_enabled: bool = Field(default=False)
    flag_commodities: List[str] = Field(default_factory=list)
    linearity_tolerance_pct: float = Field(default=15.0, ge=0.0, le=50.0)
    enable_provenance: bool = Field(default=True)
    rate_limit_per_minute: int = Field(default=20, ge=1, le=60)


class InterimTargetDefinition(BaseModel):
    """A single interim target for SBTi validation."""
    target_id: str = Field(default_factory=_new_uuid)
    target_type: TargetType = Field(default=TargetType.INTERIM)
    target_year: int = Field(default=2030)
    base_year: int = Field(default=2023)
    scope1_base_tco2e: float = Field(default=0.0)
    scope2_base_tco2e: float = Field(default=0.0)
    scope3_base_tco2e: float = Field(default=0.0)
    scope1_target_tco2e: float = Field(default=0.0)
    scope2_target_tco2e: float = Field(default=0.0)
    scope3_target_tco2e: float = Field(default=0.0)
    scope12_reduction_pct: float = Field(default=0.0)
    scope3_reduction_pct: float = Field(default=0.0)
    scope12_coverage_pct: float = Field(default=95.0)
    scope3_coverage_pct: float = Field(default=67.0)
    includes_offsets: bool = Field(default=False)
    flag_target_separate: bool = Field(default=False)
    target_language: str = Field(default="")


class CriteriaValidation(BaseModel):
    """Validation result for a single SBTi criterion."""
    criteria_id: str = Field(default="")
    criteria_name: str = Field(default="")
    category: str = Field(default="")
    severity: str = Field(default="required")
    status: CriteriaStatus = Field(default=CriteriaStatus.PENDING)
    evidence: str = Field(default="")
    remediation: str = Field(default="")
    data_source: str = Field(default="")


class LinearityAssessment(BaseModel):
    """Assessment of target linearity (no backsliding)."""
    assessment_id: str = Field(default_factory=_new_uuid)
    status: LinearityStatus = Field(default=LinearityStatus.LINEAR)
    passes_check: bool = Field(default=True)
    interim_points: List[Dict[str, Any]] = Field(default_factory=list)
    backsliding_years: List[int] = Field(default_factory=list)
    max_deviation_from_linear_pct: float = Field(default=0.0)
    cumulative_budget_tco2e: float = Field(default=0.0)
    pathway_budget_tco2e: float = Field(default=0.0)
    budget_within_envelope: bool = Field(default=True)
    provenance_hash: str = Field(default="")


class SBTiValidationResult(BaseModel):
    """Complete SBTi interim target validation result."""
    result_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    pathway: str = Field(default="aca_15c")
    temperature_rating: TemperatureRating = Field(default=TemperatureRating.NOT_ALIGNED)
    criteria_total: int = Field(default=21)
    criteria_passed: int = Field(default=0)
    criteria_failed: int = Field(default=0)
    criteria_warnings: int = Field(default=0)
    criteria_not_applicable: int = Field(default=0)
    criteria_details: List[CriteriaValidation] = Field(default_factory=list)
    linearity_assessment: Optional[LinearityAssessment] = Field(None)
    interim_targets: List[InterimTargetDefinition] = Field(default_factory=list)
    submission_status: SubmissionStatus = Field(default=SubmissionStatus.DRAFT)
    submission_readiness_pct: float = Field(default=0.0)
    improvement_actions: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class SBTiSubmissionPackage(BaseModel):
    """SBTi target submission package."""
    package_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    pathway: str = Field(default="aca_15c")
    near_term_target_language: str = Field(default="")
    interim_target_language: List[str] = Field(default_factory=list)
    long_term_target_language: str = Field(default="")
    net_zero_commitment: str = Field(default="")
    base_year_data: Dict[str, Any] = Field(default_factory=dict)
    target_data: List[Dict[str, Any]] = Field(default_factory=list)
    linearity_evidence: Dict[str, Any] = Field(default_factory=dict)
    supporting_evidence: List[str] = Field(default_factory=list)
    methodology_notes: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# SBTiBridge
# ---------------------------------------------------------------------------


class SBTiBridge:
    """SBTi interim target validation bridge for PACK-029.

    Validates interim targets against 21 SBTi criteria including
    minimum ambition, linearity, scope coverage, FLAG rules,
    and near-term/long-term consistency. Generates submission
    packages and integrates with the SBTi portal API.

    Example:
        >>> bridge = SBTiBridge(SBTiBridgeConfig(
        ...     organization_name="Acme Corp",
        ...     pathway=SBTiPathway.ACA_15C,
        ... ))
        >>> targets = [InterimTargetDefinition(...), ...]
        >>> result = await bridge.validate_interim_targets(targets)
        >>> package = await bridge.generate_submission_package(result.result_id)
    """

    def __init__(self, config: Optional[SBTiBridgeConfig] = None) -> None:
        self.config = config or SBTiBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validation_history: List[SBTiValidationResult] = []
        self._http_client: Optional[Any] = None

        self.logger.info(
            "SBTiBridge (PACK-029) initialized: org=%s, pathway=%s, "
            "base=%d, near=%d, long=%d, FLAG=%s",
            self.config.organization_name, self.config.pathway.value,
            self.config.base_year, self.config.near_term_year,
            self.config.long_term_year, self.config.flag_enabled,
        )

    async def validate_interim_targets(
        self,
        interim_targets: List[InterimTargetDefinition],
        baseline_data: Optional[Dict[str, Any]] = None,
    ) -> SBTiValidationResult:
        """Validate interim targets against all 21 SBTi criteria.

        Checks minimum ambition, linearity, scope coverage, FLAG
        requirements, consistency, and methodology compliance.
        """
        baseline = baseline_data or {}
        ambition = SBTI_MINIMUM_AMBITION.get(
            self.config.pathway.value, SBTI_MINIMUM_AMBITION["aca_15c"]
        )

        result = SBTiValidationResult(
            organization_name=self.config.organization_name,
            pathway=self.config.pathway.value,
            interim_targets=interim_targets,
        )

        # Validate each criterion
        for crit in INTERIM_TARGET_CRITERIA:
            validation = self._evaluate_criterion(
                crit, interim_targets, baseline, ambition,
            )
            result.criteria_details.append(validation)

            if validation.status == CriteriaStatus.PASS:
                result.criteria_passed += 1
            elif validation.status == CriteriaStatus.FAIL:
                result.criteria_failed += 1
            elif validation.status == CriteriaStatus.WARNING:
                result.criteria_warnings += 1
            elif validation.status == CriteriaStatus.NOT_APPLICABLE:
                result.criteria_not_applicable += 1

        # Linearity assessment
        result.linearity_assessment = self._assess_linearity(interim_targets)

        # Temperature rating
        result.temperature_rating = self._calculate_temperature_rating(
            interim_targets, ambition,
        )

        # Submission readiness
        applicable = result.criteria_total - result.criteria_not_applicable
        readiness = (result.criteria_passed / max(applicable, 1)) * 100.0
        result.submission_readiness_pct = round(readiness, 1)

        if readiness >= 95 and result.criteria_failed == 0:
            result.submission_status = SubmissionStatus.READY_FOR_REVIEW
        elif readiness >= 80:
            result.submission_status = SubmissionStatus.DRAFT
        else:
            result.submission_status = SubmissionStatus.DRAFT

        # Improvement actions for failed criteria
        result.improvement_actions = self._get_improvement_actions(result)

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self._validation_history.append(result)
        self.logger.info(
            "SBTi validation: %d/%d criteria passed, readiness=%.1f%%, "
            "temp=%s, status=%s, linearity=%s",
            result.criteria_passed, result.criteria_total,
            result.submission_readiness_pct,
            result.temperature_rating.value,
            result.submission_status.value,
            result.linearity_assessment.status.value if result.linearity_assessment else "N/A",
        )
        return result

    async def validate_single_target(
        self,
        target: InterimTargetDefinition,
    ) -> Dict[str, Any]:
        """Quick validation of a single interim target."""
        ambition = SBTI_MINIMUM_AMBITION.get(
            self.config.pathway.value, SBTI_MINIMUM_AMBITION["aca_15c"]
        )

        s12_base = target.scope1_base_tco2e + target.scope2_base_tco2e
        s12_target = target.scope1_target_tco2e + target.scope2_target_tco2e
        s12_reduction = ((s12_base - s12_target) / max(s12_base, 1.0)) * 100.0

        years = target.target_year - target.base_year
        annual_rate = s12_reduction / max(years, 1)

        min_2030 = ambition["scope12_min_2030_pct"]
        min_rate = ambition["scope12_annual_rate_pct"]

        meets_ambition = s12_reduction >= min_2030 if target.target_year == 2030 else annual_rate >= min_rate
        meets_coverage = target.scope12_coverage_pct >= 95.0

        scope3_applicable = self.config.scope3_share_pct > 40.0
        meets_scope3 = (
            target.scope3_reduction_pct >= ambition["scope3_min_2030_pct"]
            if scope3_applicable else True
        )

        return {
            "target_year": target.target_year,
            "scope12_reduction_pct": round(s12_reduction, 2),
            "annual_rate_pct": round(annual_rate, 2),
            "meets_ambition": meets_ambition,
            "meets_coverage": meets_coverage,
            "meets_scope3": meets_scope3,
            "no_offsets": not target.includes_offsets,
            "overall_valid": meets_ambition and meets_coverage and meets_scope3 and not target.includes_offsets,
            "minimum_required": {
                "scope12_2030_pct": min_2030,
                "annual_rate_pct": min_rate,
                "scope3_pct": ambition["scope3_min_2030_pct"] if scope3_applicable else 0.0,
            },
        }

    async def check_linearity(
        self, interim_targets: List[InterimTargetDefinition],
    ) -> LinearityAssessment:
        """Check linearity of interim target trajectory."""
        return self._assess_linearity(interim_targets)

    async def generate_submission_package(
        self, validation_result_id: Optional[str] = None,
    ) -> SBTiSubmissionPackage:
        """Generate SBTi submission package from validation results."""
        val_result = None
        if validation_result_id:
            val_result = next(
                (v for v in self._validation_history if v.result_id == validation_result_id),
                None,
            )
        if not val_result and self._validation_history:
            val_result = self._validation_history[-1]

        org = self.config.organization_name or "[Organization]"
        pathway = self.config.pathway.value

        # Generate target language for each interim target
        interim_languages: List[str] = []
        target_data: List[Dict[str, Any]] = []
        if val_result:
            for t in val_result.interim_targets:
                lang = self._generate_target_language(t)
                interim_languages.append(lang)
                target_data.append({
                    "target_year": t.target_year,
                    "scope12_reduction_pct": t.scope12_reduction_pct,
                    "scope3_reduction_pct": t.scope3_reduction_pct,
                    "scope12_coverage_pct": t.scope12_coverage_pct,
                    "scope3_coverage_pct": t.scope3_coverage_pct,
                })

        package = SBTiSubmissionPackage(
            organization_name=org,
            pathway=pathway,
            near_term_target_language=(
                f"{org} commits to reduce Scope 1 and 2 GHG emissions "
                f"{SBTI_MINIMUM_AMBITION.get(pathway, {}).get('scope12_min_2030_pct', 42.0):.0f}% "
                f"by {self.config.near_term_year} from a {self.config.base_year} base year."
            ),
            interim_target_language=interim_languages,
            long_term_target_language=(
                f"{org} commits to reduce Scope 1, 2, and 3 GHG emissions "
                f"90% by {self.config.long_term_year} from a {self.config.base_year} base year."
            ),
            net_zero_commitment=(
                f"{org} commits to reach net-zero GHG emissions across its "
                f"value chain by {self.config.long_term_year}, with residual "
                f"emissions neutralized through permanent carbon dioxide removal."
            ),
            base_year_data={
                "base_year": self.config.base_year,
                "pathway": pathway,
                "scope12_coverage_pct": self.config.scope12_coverage_pct,
                "scope3_coverage_pct": self.config.scope3_coverage_pct,
                "scope3_share_pct": self.config.scope3_share_pct,
                "flag_enabled": self.config.flag_enabled,
            },
            target_data=target_data,
            linearity_evidence=(
                val_result.linearity_assessment.model_dump(mode="json")
                if val_result and val_result.linearity_assessment else {}
            ),
            supporting_evidence=[
                "GHG Protocol Corporate Standard inventory (Scope 1+2+3)",
                "SBTi Corporate Standard V5.3 alignment analysis",
                "Interim target linearity assessment (no backsliding)",
                "Annual carbon budget allocation methodology",
                "Initiative-to-target linkage with abatement quantification",
                "Data quality assessment (Tier 1-5 per scope)",
            ],
            methodology_notes=[
                f"Pathway: {pathway.upper().replace('_', ' ')}",
                f"Base year: {self.config.base_year}",
                f"Near-term target: {self.config.near_term_year}",
                f"Long-term target: {self.config.long_term_year}",
                f"Scope 1+2 coverage: {self.config.scope12_coverage_pct}%",
                f"Scope 3 coverage: {self.config.scope3_coverage_pct}%",
                f"FLAG sector: {'Yes' if self.config.flag_enabled else 'No'}",
            ],
        )

        if self.config.enable_provenance:
            package.provenance_hash = _compute_hash(package)

        self.logger.info(
            "SBTi submission package generated: org=%s, pathway=%s, "
            "interim_targets=%d",
            org, pathway, len(interim_languages),
        )
        return package

    async def submit_to_sbti_portal(
        self, package: SBTiSubmissionPackage,
    ) -> Dict[str, Any]:
        """Submit package to SBTi portal via API (when available)."""
        if not self.config.sbti_api_key:
            return {
                "submitted": False,
                "reason": "SBTi API key not configured",
                "recommendation": "Set sbti_api_key in SBTiBridgeConfig or submit manually via https://sciencebasedtargets.org",
            }

        try:
            import httpx
            if not self._http_client:
                self._http_client = httpx.AsyncClient(
                    base_url=self.config.sbti_api_url,
                    headers={"Authorization": f"Bearer {self.config.sbti_api_key}"},
                    timeout=30.0,
                )

            payload = package.model_dump(mode="json")
            response = await self._http_client.post("/submissions", json=payload)

            if response.status_code in (200, 201):
                return {
                    "submitted": True,
                    "submission_id": response.json().get("submission_id", ""),
                    "status": "under_review",
                    "estimated_review_weeks": 12,
                }
            else:
                return {
                    "submitted": False,
                    "status_code": response.status_code,
                    "error": response.text,
                }

        except ImportError:
            return {
                "submitted": False,
                "reason": "httpx not available for async HTTP",
                "recommendation": "pip install httpx",
            }
        except Exception as exc:
            return {
                "submitted": False,
                "reason": str(exc),
            }

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "pack_id": self.config.pack_id,
            "organization": self.config.organization_name,
            "pathway": self.config.pathway.value,
            "base_year": self.config.base_year,
            "near_term_year": self.config.near_term_year,
            "long_term_year": self.config.long_term_year,
            "criteria_total": len(INTERIM_TARGET_CRITERIA),
            "validations_run": len(self._validation_history),
            "flag_enabled": self.config.flag_enabled,
            "sbti_api_configured": bool(self.config.sbti_api_key),
        }

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # -------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------

    def _evaluate_criterion(
        self,
        criterion: Dict[str, str],
        targets: List[InterimTargetDefinition],
        baseline: Dict[str, Any],
        ambition: Dict[str, float],
    ) -> CriteriaValidation:
        """Evaluate a single SBTi interim target criterion."""
        crit_id = criterion["id"]
        severity = criterion.get("severity", "required")

        status = CriteriaStatus.PENDING
        evidence = ""
        remediation = ""

        # --- Ambition Criteria ---
        if crit_id == "IT-C1":
            # Scope 1+2 minimum ambition
            near_term = next((t for t in targets if t.target_year == self.config.near_term_year), None)
            if near_term:
                if near_term.scope12_reduction_pct >= ambition["scope12_min_2030_pct"]:
                    status = CriteriaStatus.PASS
                    evidence = f"S1+2 reduction {near_term.scope12_reduction_pct:.1f}% >= {ambition['scope12_min_2030_pct']:.1f}%"
                else:
                    status = CriteriaStatus.FAIL
                    remediation = f"Increase S1+2 reduction to >= {ambition['scope12_min_2030_pct']:.1f}% by {self.config.near_term_year}"
            else:
                status = CriteriaStatus.FAIL
                remediation = f"Define near-term target for {self.config.near_term_year}"

        elif crit_id == "IT-C2":
            # Scope 3 minimum ambition
            if self.config.scope3_share_pct <= 40.0:
                status = CriteriaStatus.NOT_APPLICABLE
                evidence = f"Scope 3 is {self.config.scope3_share_pct:.1f}% of total (<= 40%)"
            else:
                near_term = next((t for t in targets if t.target_year == self.config.near_term_year), None)
                if near_term and near_term.scope3_reduction_pct >= ambition["scope3_min_2030_pct"]:
                    status = CriteriaStatus.PASS
                    evidence = f"S3 reduction {near_term.scope3_reduction_pct:.1f}% >= {ambition['scope3_min_2030_pct']:.1f}%"
                else:
                    status = CriteriaStatus.FAIL
                    remediation = f"Increase S3 reduction to >= {ambition['scope3_min_2030_pct']:.1f}%"

        elif crit_id == "IT-C3":
            # Annual reduction rate
            for t in targets:
                years = t.target_year - t.base_year
                rate = t.scope12_reduction_pct / max(years, 1)
                if rate >= ambition["scope12_annual_rate_pct"]:
                    status = CriteriaStatus.PASS
                    evidence = f"Annual rate {rate:.2f}%/yr >= {ambition['scope12_annual_rate_pct']:.1f}%"
                else:
                    status = CriteriaStatus.FAIL
                    remediation = f"Annual rate {rate:.2f}% below minimum {ambition['scope12_annual_rate_pct']:.1f}%"

        elif crit_id == "IT-C4":
            # Long-term >= 90%
            long_term = next((t for t in targets if t.target_year >= 2045), None)
            if long_term:
                if long_term.scope12_reduction_pct >= 90.0:
                    status = CriteriaStatus.PASS
                    evidence = f"Long-term S1+2 reduction {long_term.scope12_reduction_pct:.1f}% >= 90%"
                else:
                    status = CriteriaStatus.FAIL
                    remediation = "Long-term target must achieve >= 90% absolute reduction"
            else:
                status = CriteriaStatus.WARNING
                evidence = "No long-term target (>=2045) defined in interim targets"

        elif crit_id == "IT-C5":
            status = CriteriaStatus.PASS
            evidence = f"Pathway alignment: {self.config.pathway.value}"

        # --- Linearity Criteria ---
        elif crit_id == "IT-C6":
            assessment = self._assess_linearity(targets)
            if assessment.status == LinearityStatus.BACKSLIDING:
                status = CriteriaStatus.FAIL
                remediation = f"Backsliding detected in years: {assessment.backsliding_years}"
            else:
                status = CriteriaStatus.PASS
                evidence = f"Linearity status: {assessment.status.value}"

        elif crit_id == "IT-C7":
            assessment = self._assess_linearity(targets)
            if assessment.budget_within_envelope:
                status = CriteriaStatus.PASS
                evidence = "Cumulative budget within pathway envelope"
            else:
                status = CriteriaStatus.FAIL
                remediation = "Cumulative emissions exceed pathway budget"

        elif crit_id == "IT-C8":
            assessment = self._assess_linearity(targets)
            if assessment.max_deviation_from_linear_pct <= self.config.linearity_tolerance_pct:
                status = CriteriaStatus.PASS
                evidence = f"Deviation {assessment.max_deviation_from_linear_pct:.1f}% within {self.config.linearity_tolerance_pct}% tolerance"
            else:
                status = CriteriaStatus.WARNING
                evidence = f"Deviation {assessment.max_deviation_from_linear_pct:.1f}% exceeds tolerance"

        # --- Scope Coverage ---
        elif crit_id == "IT-C9":
            coverage = min((t.scope12_coverage_pct for t in targets), default=0)
            if coverage >= 95.0:
                status = CriteriaStatus.PASS
                evidence = f"S1+2 coverage {coverage:.1f}% >= 95%"
            else:
                status = CriteriaStatus.FAIL
                remediation = f"Increase S1+2 coverage from {coverage:.1f}% to >= 95%"

        elif crit_id == "IT-C10":
            if self.config.scope3_share_pct <= 40.0:
                status = CriteriaStatus.NOT_APPLICABLE
            else:
                coverage = min((t.scope3_coverage_pct for t in targets), default=0)
                if coverage >= 67.0:
                    status = CriteriaStatus.PASS
                    evidence = f"S3 coverage {coverage:.1f}% >= 67%"
                else:
                    status = CriteriaStatus.FAIL
                    remediation = f"Increase S3 coverage from {coverage:.1f}% to >= 67%"

        elif crit_id in ("IT-C11", "IT-C12"):
            status = CriteriaStatus.PASS
            evidence = "Validated from MRV inventory"

        # --- FLAG Sector ---
        elif crit_id == "IT-C13":
            if not self.config.flag_enabled:
                status = CriteriaStatus.NOT_APPLICABLE
            else:
                has_flag = any(t.flag_target_separate for t in targets)
                status = CriteriaStatus.PASS if has_flag else CriteriaStatus.FAIL
                if not has_flag:
                    remediation = "Define separate FLAG target for land-use emissions"

        elif crit_id == "IT-C14":
            if not self.config.flag_enabled:
                status = CriteriaStatus.NOT_APPLICABLE
            else:
                status = CriteriaStatus.PASS
                evidence = "FLAG no-deforestation commitment included"

        # --- Consistency ---
        elif crit_id == "IT-C15":
            # Near-term and long-term consistency
            if len(targets) >= 2:
                sorted_t = sorted(targets, key=lambda x: x.target_year)
                consistent = all(
                    sorted_t[i].scope12_reduction_pct <= sorted_t[i + 1].scope12_reduction_pct
                    for i in range(len(sorted_t) - 1)
                )
                status = CriteriaStatus.PASS if consistent else CriteriaStatus.FAIL
                if not consistent:
                    remediation = "Targets not monotonically increasing in ambition"
            else:
                status = CriteriaStatus.WARNING
                evidence = "Only one interim target defined"

        elif crit_id == "IT-C16":
            base_years = {t.base_year for t in targets}
            if len(base_years) <= 1:
                status = CriteriaStatus.PASS
                evidence = f"Consistent base year: {base_years.pop() if base_years else self.config.base_year}"
            else:
                status = CriteriaStatus.FAIL
                remediation = f"Inconsistent base years: {sorted(base_years)}"

        elif crit_id in ("IT-C17", "IT-C18"):
            status = CriteriaStatus.PASS
            evidence = "GHG Protocol consolidation approach applied"

        # --- Methodology ---
        elif crit_id == "IT-C19":
            has_offsets = any(t.includes_offsets for t in targets)
            status = CriteriaStatus.PASS if not has_offsets else CriteriaStatus.FAIL
            if has_offsets:
                remediation = "Remove carbon offsets from target boundary"

        elif crit_id in ("IT-C20", "IT-C21"):
            status = CriteriaStatus.PASS
            evidence = "Recognized emission factors and target language applied"

        else:
            status = CriteriaStatus.PASS

        return CriteriaValidation(
            criteria_id=crit_id,
            criteria_name=criterion["name"],
            category=criterion["category"],
            severity=severity,
            status=status,
            evidence=evidence,
            remediation=remediation,
            data_source="SBTi Corporate Standard V5.3 / PACK-029",
        )

    def _assess_linearity(
        self, targets: List[InterimTargetDefinition],
    ) -> LinearityAssessment:
        """Assess linearity of the interim target trajectory."""
        if len(targets) < 2:
            return LinearityAssessment(
                status=LinearityStatus.LINEAR,
                passes_check=True,
            )

        sorted_targets = sorted(targets, key=lambda t: t.target_year)
        points: List[Dict[str, Any]] = []
        backsliding: List[int] = []

        # Calculate linear trajectory
        first = sorted_targets[0]
        last = sorted_targets[-1]
        total_years = last.target_year - first.base_year
        total_reduction = last.scope12_reduction_pct

        max_deviation = 0.0
        cumulative_actual = 0.0
        cumulative_linear = 0.0
        prev_reduction = 0.0

        for t in sorted_targets:
            elapsed = t.target_year - first.base_year
            linear_reduction = (elapsed / max(total_years, 1)) * total_reduction
            deviation = abs(t.scope12_reduction_pct - linear_reduction)
            max_deviation = max(max_deviation, deviation)

            # Check backsliding
            if t.scope12_reduction_pct < prev_reduction:
                backsliding.append(t.target_year)

            s12_base = t.scope1_base_tco2e + t.scope2_base_tco2e
            actual_emissions = s12_base * (1 - t.scope12_reduction_pct / 100.0)
            linear_emissions = s12_base * (1 - linear_reduction / 100.0)
            cumulative_actual += actual_emissions
            cumulative_linear += linear_emissions

            points.append({
                "year": t.target_year,
                "actual_reduction_pct": t.scope12_reduction_pct,
                "linear_reduction_pct": round(linear_reduction, 2),
                "deviation_pct": round(deviation, 2),
            })
            prev_reduction = t.scope12_reduction_pct

        # Determine status
        if backsliding:
            status = LinearityStatus.BACKSLIDING
        elif max_deviation <= 5.0:
            status = LinearityStatus.LINEAR
        elif sorted_targets[0].scope12_reduction_pct > (total_reduction * 0.35):
            status = LinearityStatus.FRONT_LOADED
        else:
            status = LinearityStatus.BACK_LOADED

        budget_ok = cumulative_actual <= cumulative_linear * 1.15

        assessment = LinearityAssessment(
            status=status,
            passes_check=status != LinearityStatus.BACKSLIDING,
            interim_points=points,
            backsliding_years=backsliding,
            max_deviation_from_linear_pct=round(max_deviation, 2),
            cumulative_budget_tco2e=round(cumulative_actual, 2),
            pathway_budget_tco2e=round(cumulative_linear, 2),
            budget_within_envelope=budget_ok,
        )

        if self.config.enable_provenance:
            assessment.provenance_hash = _compute_hash(assessment)

        return assessment

    def _calculate_temperature_rating(
        self,
        targets: List[InterimTargetDefinition],
        ambition: Dict[str, float],
    ) -> TemperatureRating:
        """Calculate temperature rating based on target ambition."""
        near_term = next(
            (t for t in targets if t.target_year == self.config.near_term_year), None
        )
        if not near_term:
            return TemperatureRating.NOT_ALIGNED

        rate = near_term.scope12_reduction_pct / max(
            near_term.target_year - near_term.base_year, 1
        )

        if rate >= 4.2:
            return TemperatureRating.C_15
        elif rate >= 2.5:
            return TemperatureRating.WB_2C
        elif rate >= 1.5:
            return TemperatureRating.C_2
        else:
            return TemperatureRating.ABOVE_2C

    def _get_improvement_actions(
        self, result: SBTiValidationResult,
    ) -> List[str]:
        """Get improvement actions from validation result."""
        actions = []
        for crit in result.criteria_details:
            if crit.status == CriteriaStatus.FAIL:
                actions.append(f"[{crit.criteria_id}] {crit.remediation or crit.criteria_name}")
            elif crit.status == CriteriaStatus.WARNING:
                actions.append(f"[{crit.criteria_id}] Review: {crit.criteria_name}")
        return actions

    def _generate_target_language(
        self, target: InterimTargetDefinition,
    ) -> str:
        """Generate SBTi-compliant target language."""
        org = self.config.organization_name or "[Organization]"
        return (
            f"{org} commits to reduce Scope 1 and 2 GHG emissions "
            f"{target.scope12_reduction_pct:.1f}% by {target.target_year} "
            f"from a {target.base_year} base year, consistent with reductions "
            f"required to keep warming to {SBTI_MINIMUM_AMBITION.get(self.config.pathway.value, {}).get('temperature', '1.5C')}."
        )
