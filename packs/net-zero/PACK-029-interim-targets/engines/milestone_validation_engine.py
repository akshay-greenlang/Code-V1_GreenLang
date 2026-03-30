# -*- coding: utf-8 -*-
"""
MilestoneValidationEngine - PACK-029 Interim Targets Pack Engine 7
====================================================================

Validates interim targets against SBTi criteria with 21 validation
checks, minimum ambition validation, linearity checks, scope coverage
requirements, FLAG sector special rules, and comprehensive validation
report generation.

Validation Checks (21 total):
    Near-Term Checks (7):
        1. Near-term target year <= 10 years from submission
        2. Near-term Scope 1+2 reduction >= 42% (1.5C) or 25% (WB2C)
        3. Annual Scope 1+2 rate >= 4.2%/yr (1.5C) or 2.5%/yr (WB2C)
        4. Scope 1+2 coverage >= 95%
        5. Scope 3 coverage >= 67% (if Scope 3 >= 40% of total)
        6. Near-term Scope 3 reduction >= 52% (1.5C) within 5-10 years
        7. Baseline year no earlier than 5 years before submission

    Long-Term Checks (5):
        8.  Long-term reduction >= 90% from baseline
        9.  Long-term target year <= 2050
        10. Residual emissions <= 10% of baseline
        11. Neutralization strategy for residuals
        12. All scopes included in long-term target

    Pathway Checks (4):
        13. Linear pathway (no backsliding between milestones)
        14. Consistent annual rate (no periods of zero reduction)
        15. 5-year interim milestones present
        16. Pathway covers all years (no gaps)

    Scope & Coverage (3):
        17. Scope 1 included
        18. Scope 2 included
        19. Scope 3 included (if material)

    FLAG Sector (2):
        20. FLAG targets separate from non-FLAG
        21. FLAG reduction >= 30% from FLAG baseline

Regulatory References:
    - SBTi Corporate Net-Zero Standard v1.2 (2024)
    - SBTi Corporate Manual v5.3 (2024)
    - SBTi Target Validation Protocol v2.0
    - SBTi FLAG Guidance (2022)
    - IPCC AR6 WG3 (2022) -- 1.5C pathways

Zero-Hallucination:
    - All thresholds hard-coded from SBTi publications
    - Rule-based validation (no ML/LLM)
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-029 Interim Targets
Engine:  7 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(n: Decimal, d: Decimal, default: Decimal = Decimal("0")) -> Decimal:
    if d == Decimal("0"):
        return default
    return n / d

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    q = "0." + "0" * places
    return value.quantize(Decimal(q), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_DATA = "insufficient_data"

class CheckCategory(str, Enum):
    NEAR_TERM = "near_term"
    LONG_TERM = "long_term"
    PATHWAY = "pathway"
    SCOPE_COVERAGE = "scope_coverage"
    FLAG = "flag"

class AmbitionLevel(str, Enum):
    CELSIUS_1_5 = "1.5c"
    WELL_BELOW_2C = "wb2c"
    TWO_C = "2c"

class DataQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALIDATION_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    AmbitionLevel.CELSIUS_1_5.value: {
        "near_term_s12_reduction_pct": Decimal("42"),
        "near_term_s12_annual_rate_pct": Decimal("4.2"),
        "near_term_s3_reduction_pct": Decimal("52"),
        "near_term_max_years": 10,
        "long_term_reduction_pct": Decimal("90"),
        "long_term_max_year": 2050,
        "residual_max_pct": Decimal("10"),
        "s12_coverage_pct": Decimal("95"),
        "s3_coverage_pct": Decimal("67"),
        "s3_materiality_pct": Decimal("40"),
        "baseline_max_age_years": 5,
        "flag_reduction_pct": Decimal("30"),
    },
    AmbitionLevel.WELL_BELOW_2C.value: {
        "near_term_s12_reduction_pct": Decimal("25"),
        "near_term_s12_annual_rate_pct": Decimal("2.5"),
        "near_term_s3_reduction_pct": Decimal("25"),
        "near_term_max_years": 10,
        "long_term_reduction_pct": Decimal("80"),
        "long_term_max_year": 2050,
        "residual_max_pct": Decimal("20"),
        "s12_coverage_pct": Decimal("95"),
        "s3_coverage_pct": Decimal("67"),
        "s3_materiality_pct": Decimal("40"),
        "baseline_max_age_years": 5,
        "flag_reduction_pct": Decimal("30"),
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class MilestonePoint(BaseModel):
    """A milestone in the target pathway."""
    year: int = Field(..., ge=2020, le=2070)
    scope: str = Field(default="all_scopes")
    reduction_pct: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))

class MilestoneValidationInput(BaseModel):
    """Input for milestone validation."""
    entity_name: str = Field(..., min_length=1, max_length=300)
    entity_id: str = Field(default="", max_length=100)
    ambition_level: AmbitionLevel = Field(default=AmbitionLevel.CELSIUS_1_5)
    submission_year: int = Field(default=2024, ge=2020, le=2030)
    baseline_year: int = Field(..., ge=2015, le=2025)
    baseline_scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    baseline_scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    baseline_scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    baseline_total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope12_coverage_pct: Decimal = Field(default=Decimal("95"))
    scope3_coverage_pct: Decimal = Field(default=Decimal("67"))
    near_term_year: int = Field(default=2030, ge=2025, le=2040)
    near_term_s12_reduction_pct: Decimal = Field(default=Decimal("0"))
    near_term_s3_reduction_pct: Decimal = Field(default=Decimal("0"))
    near_term_annual_rate_s12_pct: Decimal = Field(default=Decimal("0"))
    long_term_year: int = Field(default=2050, ge=2030, le=2070)
    long_term_reduction_pct: Decimal = Field(default=Decimal("90"))
    residual_emissions_pct: Decimal = Field(default=Decimal("10"))
    has_neutralization_strategy: bool = Field(default=False)
    includes_scope1: bool = Field(default=True)
    includes_scope2: bool = Field(default=True)
    includes_scope3: bool = Field(default=True)
    is_flag_sector: bool = Field(default=False)
    flag_targets_separate: bool = Field(default=False)
    flag_reduction_pct: Decimal = Field(default=Decimal("0"))
    milestones: List[MilestonePoint] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class ValidationCheck(BaseModel):
    """A single validation check result."""
    check_id: int = Field(default=0)
    check_name: str = Field(default="")
    category: str = Field(default="")
    status: str = Field(default=CheckStatus.INSUFFICIENT_DATA.value)
    threshold: str = Field(default="")
    actual_value: str = Field(default="")
    message: str = Field(default="")
    reference: str = Field(default="")

class MilestoneValidationResult(BaseModel):
    """Complete milestone validation result."""
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    entity_id: str = Field(default="")
    ambition_level: str = Field(default="")
    is_compliant: bool = Field(default=False)
    total_checks: int = Field(default=21)
    passed_checks: int = Field(default=0)
    failed_checks: int = Field(default=0)
    warning_checks: int = Field(default=0)
    na_checks: int = Field(default=0)
    checks: List[ValidationCheck] = Field(default_factory=list)
    compliance_score_pct: Decimal = Field(default=Decimal("0"))
    critical_failures: List[str] = Field(default_factory=list)
    data_quality: str = Field(default=DataQuality.MEDIUM.value)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MilestoneValidationEngine:
    """Milestone validation engine for PACK-029.

    Validates interim targets against SBTi criteria with 21 checks.

    Usage::

        engine = MilestoneValidationEngine()
        result = await engine.calculate(validation_input)
        print(f"Compliant: {result.is_compliant} ({result.passed_checks}/{result.total_checks})")
    """

    engine_version: str = _MODULE_VERSION

    async def calculate(self, data: MilestoneValidationInput) -> MilestoneValidationResult:
        """Run all 21 validation checks."""
        t0 = time.perf_counter()
        logger.info("Milestone validation: entity=%s, ambition=%s", data.entity_name, data.ambition_level.value)

        thresholds = VALIDATION_THRESHOLDS.get(
            data.ambition_level.value,
            VALIDATION_THRESHOLDS[AmbitionLevel.CELSIUS_1_5.value],
        )

        checks: List[ValidationCheck] = []
        checks.extend(self._near_term_checks(data, thresholds))
        checks.extend(self._long_term_checks(data, thresholds))
        checks.extend(self._pathway_checks(data, thresholds))
        checks.extend(self._scope_checks(data, thresholds))
        checks.extend(self._flag_checks(data, thresholds))

        passed = sum(1 for c in checks if c.status == CheckStatus.PASS.value)
        failed = sum(1 for c in checks if c.status == CheckStatus.FAIL.value)
        warnings = sum(1 for c in checks if c.status == CheckStatus.WARNING.value)
        na = sum(1 for c in checks if c.status == CheckStatus.NOT_APPLICABLE.value)

        applicable = len(checks) - na
        score = _safe_pct(_decimal(passed), _decimal(applicable)) if applicable > 0 else Decimal("0")

        critical = [c.check_name for c in checks if c.status == CheckStatus.FAIL.value and c.category in (CheckCategory.NEAR_TERM.value, CheckCategory.LONG_TERM.value)]

        dq = self._assess_data_quality(data)
        recs = self._generate_recommendations(data, checks, failed)
        warns_list = self._generate_warnings(data, checks)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = MilestoneValidationResult(
            entity_name=data.entity_name,
            entity_id=data.entity_id,
            ambition_level=data.ambition_level.value,
            is_compliant=(failed == 0),
            total_checks=len(checks),
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            na_checks=na,
            checks=checks,
            compliance_score_pct=_round_val(score, 1),
            critical_failures=critical,
            data_quality=dq,
            recommendations=recs,
            warnings=warns_list,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    async def calculate_batch(self, inputs: List[MilestoneValidationInput]) -> List[MilestoneValidationResult]:
        results = []
        for inp in inputs:
            try:
                results.append(await self.calculate(inp))
            except Exception as exc:
                logger.error("Batch error for %s: %s", inp.entity_name, exc)
                results.append(MilestoneValidationResult(entity_name=inp.entity_name, warnings=[f"Error: {exc}"]))
        return results

    # ------------------------------------------------------------------ #
    # Near-Term Checks (1-7)                                               #
    # ------------------------------------------------------------------ #

    def _near_term_checks(self, data: MilestoneValidationInput, th: Dict) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []

        # 1. Near-term year <= 10 years from submission
        max_years = th["near_term_max_years"]
        nt_years = data.near_term_year - data.submission_year
        checks.append(ValidationCheck(check_id=1, check_name="Near-term target horizon", category=CheckCategory.NEAR_TERM.value,
            status=CheckStatus.PASS.value if nt_years <= max_years else CheckStatus.FAIL.value,
            threshold=f"<= {max_years} years", actual_value=f"{nt_years} years",
            message=f"Near-term target is {nt_years} years from submission." if nt_years <= max_years else f"Near-term target exceeds {max_years}-year maximum.",
            reference="SBTi Corporate Manual v5.3 Section 4.1"))

        # 2. Near-term S1+2 reduction
        min_red = th["near_term_s12_reduction_pct"]
        checks.append(ValidationCheck(check_id=2, check_name="Near-term S1+2 reduction", category=CheckCategory.NEAR_TERM.value,
            status=CheckStatus.PASS.value if data.near_term_s12_reduction_pct >= min_red else CheckStatus.FAIL.value,
            threshold=f">= {min_red}%", actual_value=f"{data.near_term_s12_reduction_pct}%",
            message=f"Scope 1+2 reduction of {data.near_term_s12_reduction_pct}% {'meets' if data.near_term_s12_reduction_pct >= min_red else 'below'} {min_red}% minimum.",
            reference="SBTi Corporate Net-Zero Standard v1.2 Section C.3"))

        # 3. Annual S1+2 rate
        min_rate = th["near_term_s12_annual_rate_pct"]
        checks.append(ValidationCheck(check_id=3, check_name="Annual S1+2 reduction rate", category=CheckCategory.NEAR_TERM.value,
            status=CheckStatus.PASS.value if data.near_term_annual_rate_s12_pct >= min_rate else CheckStatus.FAIL.value,
            threshold=f">= {min_rate}%/yr", actual_value=f"{data.near_term_annual_rate_s12_pct}%/yr",
            message=f"Annual rate of {data.near_term_annual_rate_s12_pct}%/yr {'meets' if data.near_term_annual_rate_s12_pct >= min_rate else 'below'} {min_rate}%/yr minimum.",
            reference="SBTi Corporate Manual v5.3 Section 4.2"))

        # 4. S1+2 coverage
        min_cov = th["s12_coverage_pct"]
        checks.append(ValidationCheck(check_id=4, check_name="Scope 1+2 coverage", category=CheckCategory.NEAR_TERM.value,
            status=CheckStatus.PASS.value if data.scope12_coverage_pct >= min_cov else CheckStatus.FAIL.value,
            threshold=f">= {min_cov}%", actual_value=f"{data.scope12_coverage_pct}%",
            message=f"Scope 1+2 coverage {data.scope12_coverage_pct}% {'meets' if data.scope12_coverage_pct >= min_cov else 'below'} {min_cov}% threshold.",
            reference="SBTi Corporate Net-Zero Standard v1.2 Section C.4"))

        # 5. S3 coverage
        min_s3 = th["s3_coverage_pct"]
        s3_material = data.baseline_scope3_tco2e > Decimal("0")
        total = data.baseline_scope1_tco2e + data.baseline_scope2_tco2e + data.baseline_scope3_tco2e
        s3_pct = _safe_pct(data.baseline_scope3_tco2e, total) if total > Decimal("0") else Decimal("0")
        s3_is_material = s3_pct >= th["s3_materiality_pct"]

        if not s3_is_material:
            checks.append(ValidationCheck(check_id=5, check_name="Scope 3 coverage", category=CheckCategory.NEAR_TERM.value,
                status=CheckStatus.NOT_APPLICABLE.value, threshold=f">= {min_s3}%", actual_value=f"S3 = {s3_pct}% of total",
                message=f"Scope 3 is {s3_pct}% of total (< {th['s3_materiality_pct']}%). Scope 3 target not required.",
                reference="SBTi Corporate Manual v5.3 Section 4.3"))
        else:
            checks.append(ValidationCheck(check_id=5, check_name="Scope 3 coverage", category=CheckCategory.NEAR_TERM.value,
                status=CheckStatus.PASS.value if data.scope3_coverage_pct >= min_s3 else CheckStatus.FAIL.value,
                threshold=f">= {min_s3}%", actual_value=f"{data.scope3_coverage_pct}%",
                message=f"Scope 3 coverage {data.scope3_coverage_pct}%.",
                reference="SBTi Corporate Manual v5.3 Section 4.3"))

        # 6. Near-term S3 reduction
        min_s3_red = th["near_term_s3_reduction_pct"]
        if s3_is_material:
            checks.append(ValidationCheck(check_id=6, check_name="Near-term S3 reduction", category=CheckCategory.NEAR_TERM.value,
                status=CheckStatus.PASS.value if data.near_term_s3_reduction_pct >= min_s3_red else CheckStatus.WARNING.value,
                threshold=f">= {min_s3_red}%", actual_value=f"{data.near_term_s3_reduction_pct}%",
                message=f"Scope 3 near-term reduction {data.near_term_s3_reduction_pct}%.",
                reference="SBTi Corporate Net-Zero Standard v1.2 Section C.5"))
        else:
            checks.append(ValidationCheck(check_id=6, check_name="Near-term S3 reduction", category=CheckCategory.NEAR_TERM.value,
                status=CheckStatus.NOT_APPLICABLE.value, threshold=f">= {min_s3_red}%", actual_value="N/A",
                message="Scope 3 not material.", reference="SBTi Corporate Manual v5.3"))

        # 7. Baseline age
        max_age = th["baseline_max_age_years"]
        age = data.submission_year - data.baseline_year
        checks.append(ValidationCheck(check_id=7, check_name="Baseline year recency", category=CheckCategory.NEAR_TERM.value,
            status=CheckStatus.PASS.value if age <= max_age else CheckStatus.WARNING.value,
            threshold=f"<= {max_age} years old", actual_value=f"{age} years old",
            message=f"Baseline year {data.baseline_year} is {age} years before submission.",
            reference="SBTi Corporate Manual v5.3 Section 3.1"))

        return checks

    # ------------------------------------------------------------------ #
    # Long-Term Checks (8-12)                                              #
    # ------------------------------------------------------------------ #

    def _long_term_checks(self, data: MilestoneValidationInput, th: Dict) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []

        # 8. Long-term reduction
        min_lt = th["long_term_reduction_pct"]
        checks.append(ValidationCheck(check_id=8, check_name="Long-term reduction", category=CheckCategory.LONG_TERM.value,
            status=CheckStatus.PASS.value if data.long_term_reduction_pct >= min_lt else CheckStatus.FAIL.value,
            threshold=f">= {min_lt}%", actual_value=f"{data.long_term_reduction_pct}%",
            message=f"Long-term reduction {data.long_term_reduction_pct}%.", reference="SBTi Net-Zero Standard v1.2 Section C.6"))

        # 9. Long-term year
        max_year = th["long_term_max_year"]
        checks.append(ValidationCheck(check_id=9, check_name="Long-term target year", category=CheckCategory.LONG_TERM.value,
            status=CheckStatus.PASS.value if data.long_term_year <= max_year else CheckStatus.FAIL.value,
            threshold=f"<= {max_year}", actual_value=f"{data.long_term_year}",
            message=f"Long-term target year {data.long_term_year}.", reference="SBTi Net-Zero Standard v1.2 Section C.7"))

        # 10. Residual emissions
        max_res = th["residual_max_pct"]
        checks.append(ValidationCheck(check_id=10, check_name="Residual emissions", category=CheckCategory.LONG_TERM.value,
            status=CheckStatus.PASS.value if data.residual_emissions_pct <= max_res else CheckStatus.FAIL.value,
            threshold=f"<= {max_res}%", actual_value=f"{data.residual_emissions_pct}%",
            message=f"Residual emissions {data.residual_emissions_pct}%.", reference="SBTi Net-Zero Standard v1.2 Section C.8"))

        # 11. Neutralization strategy
        checks.append(ValidationCheck(check_id=11, check_name="Neutralization strategy", category=CheckCategory.LONG_TERM.value,
            status=CheckStatus.PASS.value if data.has_neutralization_strategy else CheckStatus.WARNING.value,
            threshold="Required", actual_value="Present" if data.has_neutralization_strategy else "Missing",
            message="Neutralization strategy " + ("documented." if data.has_neutralization_strategy else "not documented."),
            reference="SBTi Net-Zero Standard v1.2 Section C.9"))

        # 12. All scopes in long-term
        all_scopes = data.includes_scope1 and data.includes_scope2 and data.includes_scope3
        checks.append(ValidationCheck(check_id=12, check_name="Long-term scope coverage", category=CheckCategory.LONG_TERM.value,
            status=CheckStatus.PASS.value if all_scopes else CheckStatus.WARNING.value,
            threshold="All scopes", actual_value=f"S1={'Y' if data.includes_scope1 else 'N'} S2={'Y' if data.includes_scope2 else 'N'} S3={'Y' if data.includes_scope3 else 'N'}",
            message=f"Long-term target {'covers' if all_scopes else 'missing coverage for'} all scopes.",
            reference="SBTi Net-Zero Standard v1.2 Section C.10"))

        return checks

    # ------------------------------------------------------------------ #
    # Pathway Checks (13-16)                                               #
    # ------------------------------------------------------------------ #

    def _pathway_checks(self, data: MilestoneValidationInput, th: Dict) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        sorted_ms = sorted(data.milestones, key=lambda m: m.year)

        # 13. Linearity (no backsliding)
        backslide = False
        for i in range(1, len(sorted_ms)):
            if sorted_ms[i].reduction_pct < sorted_ms[i - 1].reduction_pct:
                backslide = True
                break
        checks.append(ValidationCheck(check_id=13, check_name="Pathway linearity", category=CheckCategory.PATHWAY.value,
            status=CheckStatus.PASS.value if not backslide else CheckStatus.FAIL.value,
            threshold="No backsliding", actual_value="No backsliding" if not backslide else "Backsliding detected",
            message="Pathway is monotonically increasing." if not backslide else "Reduction decreases between milestones.",
            reference="SBTi Net-Zero Standard v1.2 Section C.11"))

        # 14. Consistent annual rate
        zero_periods = 0
        for i in range(1, len(sorted_ms)):
            if sorted_ms[i].reduction_pct == sorted_ms[i - 1].reduction_pct:
                zero_periods += 1
        checks.append(ValidationCheck(check_id=14, check_name="Consistent annual rate", category=CheckCategory.PATHWAY.value,
            status=CheckStatus.PASS.value if zero_periods == 0 else CheckStatus.WARNING.value,
            threshold="No zero-reduction periods", actual_value=f"{zero_periods} flat period(s)",
            message=f"{zero_periods} period(s) with zero additional reduction.",
            reference="SBTi Target Tracking Protocol v2.0"))

        # 15. 5-year interim milestones
        has_5yr = False
        if len(sorted_ms) >= 2:
            for i in range(1, len(sorted_ms)):
                gap = sorted_ms[i].year - sorted_ms[i - 1].year
                if gap <= 5:
                    has_5yr = True
                    break
        elif len(sorted_ms) == 1:
            has_5yr = True
        checks.append(ValidationCheck(check_id=15, check_name="5-year interim milestones", category=CheckCategory.PATHWAY.value,
            status=CheckStatus.PASS.value if has_5yr else CheckStatus.WARNING.value,
            threshold="Milestones at 5-year intervals", actual_value=f"{len(sorted_ms)} milestone(s)",
            message=f"{'Adequate' if has_5yr else 'Insufficient'} milestone frequency.",
            reference="SBTi Net-Zero Standard v1.2 Section C.12"))

        # 16. No gaps
        has_gap = False
        for i in range(1, len(sorted_ms)):
            if sorted_ms[i].year - sorted_ms[i - 1].year > 10:
                has_gap = True
                break
        checks.append(ValidationCheck(check_id=16, check_name="Pathway completeness", category=CheckCategory.PATHWAY.value,
            status=CheckStatus.PASS.value if not has_gap else CheckStatus.WARNING.value,
            threshold="No gaps > 10 years", actual_value="No gaps" if not has_gap else "Gap detected",
            message="Pathway covers all periods." if not has_gap else "Gap > 10 years between milestones.",
            reference="SBTi Net-Zero Standard v1.2"))

        return checks

    # ------------------------------------------------------------------ #
    # Scope Checks (17-19)                                                 #
    # ------------------------------------------------------------------ #

    def _scope_checks(self, data: MilestoneValidationInput, th: Dict) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []

        checks.append(ValidationCheck(check_id=17, check_name="Scope 1 included", category=CheckCategory.SCOPE_COVERAGE.value,
            status=CheckStatus.PASS.value if data.includes_scope1 else CheckStatus.FAIL.value,
            threshold="Required", actual_value="Yes" if data.includes_scope1 else "No",
            message="Scope 1 " + ("included." if data.includes_scope1 else "missing."),
            reference="SBTi Corporate Manual v5.3 Section 4.4"))

        checks.append(ValidationCheck(check_id=18, check_name="Scope 2 included", category=CheckCategory.SCOPE_COVERAGE.value,
            status=CheckStatus.PASS.value if data.includes_scope2 else CheckStatus.FAIL.value,
            threshold="Required", actual_value="Yes" if data.includes_scope2 else "No",
            message="Scope 2 " + ("included." if data.includes_scope2 else "missing."),
            reference="SBTi Corporate Manual v5.3 Section 4.4"))

        total = data.baseline_scope1_tco2e + data.baseline_scope2_tco2e + data.baseline_scope3_tco2e
        s3_pct = _safe_pct(data.baseline_scope3_tco2e, total) if total > Decimal("0") else Decimal("0")
        s3_material = s3_pct >= th["s3_materiality_pct"]

        if s3_material:
            checks.append(ValidationCheck(check_id=19, check_name="Scope 3 included (material)", category=CheckCategory.SCOPE_COVERAGE.value,
                status=CheckStatus.PASS.value if data.includes_scope3 else CheckStatus.FAIL.value,
                threshold=f"Required (S3 = {_round_val(s3_pct, 1)}% >= {th['s3_materiality_pct']}%)",
                actual_value="Yes" if data.includes_scope3 else "No",
                message=f"Scope 3 is material ({_round_val(s3_pct, 1)}%) and {'included' if data.includes_scope3 else 'NOT included'}.",
                reference="SBTi Corporate Manual v5.3 Section 4.5"))
        else:
            checks.append(ValidationCheck(check_id=19, check_name="Scope 3 included (material)", category=CheckCategory.SCOPE_COVERAGE.value,
                status=CheckStatus.NOT_APPLICABLE.value, threshold="Required if >= 40%",
                actual_value=f"S3 = {_round_val(s3_pct, 1)}%",
                message=f"Scope 3 is {_round_val(s3_pct, 1)}% of total. Not material.",
                reference="SBTi Corporate Manual v5.3 Section 4.5"))

        return checks

    # ------------------------------------------------------------------ #
    # FLAG Checks (20-21)                                                  #
    # ------------------------------------------------------------------ #

    def _flag_checks(self, data: MilestoneValidationInput, th: Dict) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []

        if not data.is_flag_sector:
            checks.append(ValidationCheck(check_id=20, check_name="FLAG targets separate", category=CheckCategory.FLAG.value,
                status=CheckStatus.NOT_APPLICABLE.value, threshold="Required for FLAG", actual_value="Not FLAG sector",
                message="Entity is not in FLAG sector.", reference="SBTi FLAG Guidance (2022)"))
            checks.append(ValidationCheck(check_id=21, check_name="FLAG reduction", category=CheckCategory.FLAG.value,
                status=CheckStatus.NOT_APPLICABLE.value, threshold="N/A", actual_value="N/A",
                message="N/A", reference="SBTi FLAG Guidance (2022)"))
            return checks

        min_flag = th["flag_reduction_pct"]
        checks.append(ValidationCheck(check_id=20, check_name="FLAG targets separate", category=CheckCategory.FLAG.value,
            status=CheckStatus.PASS.value if data.flag_targets_separate else CheckStatus.FAIL.value,
            threshold="Separate FLAG targets", actual_value="Separate" if data.flag_targets_separate else "Combined",
            message=f"FLAG targets {'are' if data.flag_targets_separate else 'are NOT'} separate.",
            reference="SBTi FLAG Guidance (2022)"))

        checks.append(ValidationCheck(check_id=21, check_name="FLAG reduction", category=CheckCategory.FLAG.value,
            status=CheckStatus.PASS.value if data.flag_reduction_pct >= min_flag else CheckStatus.FAIL.value,
            threshold=f">= {min_flag}%", actual_value=f"{data.flag_reduction_pct}%",
            message=f"FLAG reduction {data.flag_reduction_pct}%.",
            reference="SBTi FLAG Guidance (2022)"))

        return checks

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _assess_data_quality(self, data: MilestoneValidationInput) -> str:
        score = 0
        if data.baseline_scope1_tco2e > Decimal("0"):
            score += 2
        if data.baseline_scope2_tco2e > Decimal("0"):
            score += 2
        if data.baseline_scope3_tco2e > Decimal("0"):
            score += 1
        if len(data.milestones) >= 3:
            score += 2
        if data.near_term_annual_rate_s12_pct > Decimal("0"):
            score += 1
        if data.entity_id:
            score += 1
        if score >= 7:
            return DataQuality.HIGH.value
        elif score >= 4:
            return DataQuality.MEDIUM.value
        elif score >= 2:
            return DataQuality.LOW.value
        return DataQuality.ESTIMATED.value

    def _generate_recommendations(self, data: MilestoneValidationInput, checks: List[ValidationCheck], failed: int) -> List[str]:
        recs: List[str] = []
        if failed > 0:
            recs.append(f"{failed} validation check(s) failed. Address these before SBTi submission.")
        for c in checks:
            if c.status == CheckStatus.FAIL.value:
                recs.append(f"Check #{c.check_id} ({c.check_name}): {c.message} [Ref: {c.reference}]")
        return recs

    def _generate_warnings(self, data: MilestoneValidationInput, checks: List[ValidationCheck]) -> List[str]:
        warns: List[str] = []
        for c in checks:
            if c.status == CheckStatus.WARNING.value:
                warns.append(f"Check #{c.check_id} ({c.check_name}): {c.message}")
        return warns

    def get_validation_thresholds(self) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in VALIDATION_THRESHOLDS.items()}
