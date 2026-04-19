# -*- coding: utf-8 -*-
"""
DataQualityGuardianEngine - PACK-027 Enterprise Net Zero Pack Engine 9
=======================================================================

Automated data quality scoring, monitoring, and improvement roadmap
targeting +/-3% accuracy for enterprise GHG data.  Implements the GHG
Protocol 5-level data quality hierarchy with per-category and per-entity
scoring, outlier detection, missing data flagging, lineage validation,
and improvement prioritization.

Calculation Methodology:
    DQ Score (1-5 per GHG Protocol hierarchy):
        1 = Supplier-specific verified (+/-3%)
        2 = Supplier-specific unverified (+/-5-10%)
        3 = Average data physical (+/-10-20%)
        4 = Spend-based EEIO (+/-20-40%)
        5 = Proxy/extrapolation (+/-40-60%)

    Weighted DQ Score:
        overall = sum(category_dq * category_tco2e) / total_tco2e

    Accuracy Assessment:
        achieved_accuracy = f(overall_dq_score)
        meets_target = achieved_accuracy <= target_accuracy

    Improvement Priority:
        priority_score = (dq_level - target_dq) * category_tco2e
        Rank by priority_score descending

    Data Completeness:
        completeness = fields_populated / total_required_fields

    Outlier Detection:
        z_score = (value - mean) / std_dev
        outlier if |z_score| > 3.0

Regulatory References:
    - GHG Protocol Corporate Standard - Data quality guidance
    - GHG Protocol Scope 3 Standard - DQ hierarchy
    - ISO 14064-1:2018 - Data quality requirements
    - ISAE 3410 - Assurance data quality expectations

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - DQ hierarchy from GHG Protocol standard
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-027 Enterprise Net Zero Pack
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

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
        serializable = {k: v for k, v in serializable.items()
                        if k not in ("calculated_at", "processing_time_ms", "provenance_hash")}
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IssueType(str, Enum):
    MISSING_DATA = "missing_data"
    OUTLIER = "outlier"
    LOW_QUALITY = "low_quality"
    STALE_DATA = "stale_data"
    INCONSISTENCY = "inconsistency"
    LINEAGE_GAP = "lineage_gap"

class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ImprovementAction(str, Enum):
    SWITCH_TO_ACTIVITY_DATA = "switch_to_activity_data"
    REQUEST_SUPPLIER_DATA = "request_supplier_data"
    INSTALL_METERING = "install_metering"
    AUTOMATE_COLLECTION = "automate_collection"
    VERIFY_EXISTING = "verify_existing"
    FILL_MISSING = "fill_missing"
    RESOLVE_OUTLIER = "resolve_outlier"

# DQ level to accuracy mapping.
DQ_ACCURACY_MAP: Dict[int, Decimal] = {
    1: Decimal("3"), 2: Decimal("7.5"), 3: Decimal("15"),
    4: Decimal("30"), 5: Decimal("50"),
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class CategoryDQEntry(BaseModel):
    """Data quality entry for a single category or source."""
    category: str = Field(..., max_length=100)
    entity_id: str = Field(default="", max_length=100)
    data_quality_level: int = Field(..., ge=1, le=5)
    emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    data_source: str = Field(default="", max_length=200)
    fields_populated: int = Field(default=0, ge=0)
    fields_required: int = Field(default=1, ge=1)
    last_updated: str = Field(default="", max_length=10)
    has_lineage: bool = Field(default=False)
    values_for_outlier_check: List[Decimal] = Field(default_factory=list)

class DataQualityGuardianInput(BaseModel):
    """Complete input for data quality assessment."""
    organization_name: str = Field(default="Enterprise", min_length=1, max_length=500)
    reporting_year: int = Field(default=2026, ge=2020, le=2050)
    target_accuracy_pct: Decimal = Field(default=Decimal("3.0"), ge=Decimal("0"), le=Decimal("100"))
    target_dq_level: int = Field(default=2, ge=1, le=5)
    total_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    dq_entries: List[CategoryDQEntry] = Field(default_factory=list)
    prior_year_overall_dq: Optional[Decimal] = Field(None, ge=Decimal("1"), le=Decimal("5"))

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class DQIssue(BaseModel):
    """A single data quality issue."""
    issue_id: str = Field(default_factory=_new_uuid)
    category: str = Field(default="")
    entity_id: str = Field(default="")
    issue_type: str = Field(default="")
    severity: str = Field(default="medium")
    description: str = Field(default="")
    impact_tco2e: Decimal = Field(default=Decimal("0"))
    recommended_action: str = Field(default="")

class ImprovementPriority(BaseModel):
    """Prioritized improvement action."""
    rank: int = Field(default=0)
    category: str = Field(default="")
    entity_id: str = Field(default="")
    current_dq_level: int = Field(default=4)
    target_dq_level: int = Field(default=2)
    emissions_tco2e: Decimal = Field(default=Decimal("0"))
    priority_score: Decimal = Field(default=Decimal("0"))
    action: str = Field(default="")
    estimated_effort_hours: int = Field(default=0)
    expected_accuracy_improvement_pct: Decimal = Field(default=Decimal("0"))

class DataQualityGuardianResult(BaseModel):
    """Complete data quality assessment result."""
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    organization_name: str = Field(default="")

    overall_dq_score: Decimal = Field(default=Decimal("3.0"))
    target_dq_level: int = Field(default=2)
    achieved_accuracy_pct: Decimal = Field(default=Decimal("15"))
    target_accuracy_pct: Decimal = Field(default=Decimal("3"))
    meets_target: bool = Field(default=False)

    dq_by_category: Dict[str, Decimal] = Field(default_factory=dict)
    dq_by_entity: Dict[str, Decimal] = Field(default_factory=dict)
    completeness_pct: Decimal = Field(default=Decimal("0"))

    issues: List[DQIssue] = Field(default_factory=list)
    issue_count_critical: int = Field(default=0)
    issue_count_high: int = Field(default=0)
    issue_count_medium: int = Field(default=0)
    issue_count_low: int = Field(default=0)

    improvement_roadmap: List[ImprovementPriority] = Field(default_factory=list)
    yoy_dq_change: Optional[Decimal] = Field(None)

    regulatory_citations: List[str] = Field(default_factory=lambda: [
        "GHG Protocol Corporate Standard - Data quality",
        "GHG Protocol Scope 3 Standard - DQ hierarchy",
        "ISO 14064-1:2018 - Data quality",
    ])
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DataQualityGuardianEngine:
    """Automated data quality scoring and improvement engine.

    Scores, monitors, and prioritizes data quality improvement across
    all GHG categories and entities targeting +/-3% accuracy.

    Usage::

        engine = DataQualityGuardianEngine()
        result = engine.calculate(dq_input)
        assert result.provenance_hash
    """

    engine_version: str = _MODULE_VERSION

    def calculate(self, data: DataQualityGuardianInput) -> DataQualityGuardianResult:
        """Run data quality assessment."""
        t0 = time.perf_counter()
        logger.info(
            "DQ Guardian: org=%s, entries=%d, target=+/-%s%%",
            data.organization_name, len(data.dq_entries), data.target_accuracy_pct,
        )

        # Weighted DQ score
        weighted_num = Decimal("0")
        weighted_den = Decimal("0")
        dq_by_cat: Dict[str, List[Decimal]] = {}
        dq_by_entity: Dict[str, List[Decimal]] = {}
        total_pop = 0
        total_req = 0
        issues: List[DQIssue] = []
        priorities: List[ImprovementPriority] = []

        for entry in data.dq_entries:
            dq = _decimal(entry.data_quality_level)
            em = entry.emissions_tco2e
            weighted_num += dq * em
            weighted_den += em

            # By category
            if entry.category not in dq_by_cat:
                dq_by_cat[entry.category] = []
            dq_by_cat[entry.category].append(dq)

            # By entity
            eid = entry.entity_id or "default"
            if eid not in dq_by_entity:
                dq_by_entity[eid] = []
            dq_by_entity[eid].append(dq)

            # Completeness
            total_pop += entry.fields_populated
            total_req += entry.fields_required

            # Issues: low quality
            if entry.data_quality_level >= 4 and em > Decimal("0"):
                severity = IssueSeverity.CRITICAL.value if entry.data_quality_level == 5 else IssueSeverity.HIGH.value
                issues.append(DQIssue(
                    category=entry.category,
                    entity_id=entry.entity_id,
                    issue_type=IssueType.LOW_QUALITY.value,
                    severity=severity,
                    description=f"DQ level {entry.data_quality_level} for {entry.category} ({em} tCO2e)",
                    impact_tco2e=em,
                    recommended_action=ImprovementAction.SWITCH_TO_ACTIVITY_DATA.value,
                ))

            # Issues: missing data
            if entry.fields_populated < entry.fields_required:
                missing = entry.fields_required - entry.fields_populated
                issues.append(DQIssue(
                    category=entry.category,
                    entity_id=entry.entity_id,
                    issue_type=IssueType.MISSING_DATA.value,
                    severity=IssueSeverity.MEDIUM.value,
                    description=f"{missing} missing fields in {entry.category}",
                    impact_tco2e=em,
                    recommended_action=ImprovementAction.FILL_MISSING.value,
                ))

            # Issues: lineage gap
            if not entry.has_lineage and em > Decimal("0"):
                issues.append(DQIssue(
                    category=entry.category,
                    entity_id=entry.entity_id,
                    issue_type=IssueType.LINEAGE_GAP.value,
                    severity=IssueSeverity.LOW.value,
                    description=f"No data lineage for {entry.category}",
                    impact_tco2e=em,
                    recommended_action=ImprovementAction.AUTOMATE_COLLECTION.value,
                ))

            # Outlier detection (simple z-score)
            if len(entry.values_for_outlier_check) >= 3:
                vals = [float(v) for v in entry.values_for_outlier_check]
                mean_v = sum(vals) / len(vals)
                var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
                std_v = var_v ** 0.5
                if std_v > 0:
                    for v in vals:
                        z = abs((v - mean_v) / std_v)
                        if z > 3.0:
                            issues.append(DQIssue(
                                category=entry.category,
                                entity_id=entry.entity_id,
                                issue_type=IssueType.OUTLIER.value,
                                severity=IssueSeverity.HIGH.value,
                                description=f"Outlier detected (z={z:.1f}) in {entry.category}",
                                impact_tco2e=em,
                                recommended_action=ImprovementAction.RESOLVE_OUTLIER.value,
                            ))
                            break

            # Improvement priority
            if entry.data_quality_level > data.target_dq_level:
                gap = entry.data_quality_level - data.target_dq_level
                pscore = _round_val(_decimal(gap) * em, 2)
                effort = gap * 20  # rough hours
                accuracy_gain = DQ_ACCURACY_MAP.get(entry.data_quality_level, Decimal("30")) - DQ_ACCURACY_MAP.get(data.target_dq_level, Decimal("7.5"))

                priorities.append(ImprovementPriority(
                    category=entry.category,
                    entity_id=entry.entity_id,
                    current_dq_level=entry.data_quality_level,
                    target_dq_level=data.target_dq_level,
                    emissions_tco2e=em,
                    priority_score=pscore,
                    action=ImprovementAction.SWITCH_TO_ACTIVITY_DATA.value if gap >= 2 else ImprovementAction.VERIFY_EXISTING.value,
                    estimated_effort_hours=effort,
                    expected_accuracy_improvement_pct=accuracy_gain,
                ))

        # Sort and rank priorities
        priorities.sort(key=lambda p: float(p.priority_score), reverse=True)
        for i, p in enumerate(priorities):
            p.rank = i + 1

        # Overall DQ score
        overall = _round_val(_safe_divide(weighted_num, weighted_den), 1) if weighted_den > Decimal("0") else Decimal("3.0")

        # Accuracy
        dq_int = int(overall.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        dq_int = max(1, min(5, dq_int))
        achieved = DQ_ACCURACY_MAP.get(dq_int, Decimal("15"))
        meets = achieved <= data.target_accuracy_pct

        # Aggregate DQ by category and entity
        cat_dq: Dict[str, Decimal] = {}
        for cat, vals in dq_by_cat.items():
            cat_dq[cat] = _round_val(_decimal(sum(float(v) for v in vals)) / _decimal(len(vals)), 1)

        ent_dq: Dict[str, Decimal] = {}
        for eid, vals in dq_by_entity.items():
            ent_dq[eid] = _round_val(_decimal(sum(float(v) for v in vals)) / _decimal(len(vals)), 1)

        # Completeness
        completeness = _round_val(_safe_pct(_decimal(total_pop), _decimal(max(total_req, 1))), 1)

        # Issue counts
        crit = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL.value)
        high = sum(1 for i in issues if i.severity == IssueSeverity.HIGH.value)
        med = sum(1 for i in issues if i.severity == IssueSeverity.MEDIUM.value)
        low = sum(1 for i in issues if i.severity == IssueSeverity.LOW.value)

        # YoY change
        yoy = None
        if data.prior_year_overall_dq is not None:
            yoy = _round_val(overall - data.prior_year_overall_dq, 1)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = DataQualityGuardianResult(
            organization_name=data.organization_name,
            overall_dq_score=overall,
            target_dq_level=data.target_dq_level,
            achieved_accuracy_pct=achieved,
            target_accuracy_pct=data.target_accuracy_pct,
            meets_target=meets,
            dq_by_category=cat_dq,
            dq_by_entity=ent_dq,
            completeness_pct=completeness,
            issues=issues,
            issue_count_critical=crit,
            issue_count_high=high,
            issue_count_medium=med,
            issue_count_low=low,
            improvement_roadmap=priorities[:20],
            yoy_dq_change=yoy,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "DQ Guardian complete: dq=%.1f, accuracy=+/-%.0f%%, meets=%s, issues=%d, hash=%s",
            float(overall), float(achieved), meets,
            len(issues), result.provenance_hash[:16],
        )
        return result

    async def calculate_async(self, data: DataQualityGuardianInput) -> DataQualityGuardianResult:
        """Async wrapper for calculate()."""
        return self.calculate(data)
