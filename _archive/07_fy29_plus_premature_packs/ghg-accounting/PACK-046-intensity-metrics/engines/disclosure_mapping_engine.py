# -*- coding: utf-8 -*-
"""
DisclosureMappingEngine - PACK-046 Intensity Metrics Engine 9
====================================================================

Maps intensity metrics to regulatory framework disclosure requirements.
Implements completeness checks, gap analysis, and data table generation
for ESRS, CDP, SEC, SBTi, ISO 14064, TCFD, GRI, and IFRS S2.

Calculation Methodology:
    Completeness Check:
        For each framework F:
            required_fields(F) = set of mandatory data points
            populated_fields(F) = set of data points with values
            completeness_pct(F) = |populated| / |required| * 100
            is_compliant(F) = completeness_pct == 100

    Gap Analysis:
        For each missing field:
            gap = {field_name, framework, requirement_text, remediation_guidance}
        Priority scoring:
            mandatory = priority 1 (critical)
            recommended = priority 2 (important)
            optional = priority 3 (nice to have)

    Cross-Framework Mapping:
        Single intensity metric can satisfy multiple frameworks.
        Map: metric_type -> [list of framework requirements satisfied]

Regulatory Framework Requirements:
    ESRS E1-6:
        - Scope 1+2 intensity per net revenue
        - Scope 1+2+3 intensity per net revenue (if Scope 3 material)
        - Intensity per sector-specific metric where applicable
        - Base year and target year intensities

    CDP C6.10:
        - At least one intensity metric (mandatory)
        - Numerator: Scope 1+2, or Scope 3, or all scopes
        - Denominator: revenue, production, FTE, etc.
        - Sector-specific metrics in C-XX modules

    SEC Item 1504:
        - Scope 1+2 intensity per unit of revenue
        - Scope 3 intensity per unit of revenue (if disclosed)
        - Year-over-year comparison

    SBTi:
        - Sector-specific intensity metric per SDA guidance
        - Base year intensity and target year intensity
        - Annual reduction rate

    ISO 14064-1:
        - GHG intensity ratio per Clause 5.3.4
        - At least one physical and one economic intensity

    TCFD:
        - Cross-industry: GHG intensity per revenue
        - Sector-specific: as per supplemental guidance
        - Trend over time

    GRI 305-4:
        - GHG emissions intensity ratio
        - Denominator chosen by organisation
        - Scope coverage specified

    IFRS S2:
        - GHG intensity per unit of revenue
        - Cross-industry metric (Scope 1+2 per revenue)

Zero-Hallucination:
    - All framework mappings from published regulatory texts
    - No LLM involvement in completeness or gap assessment
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-046 Intensity Metrics
Engine:  9 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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

def _round2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Framework(str, Enum):
    """Regulatory framework identifier."""
    ESRS_E1_6 = "ESRS_E1_6"
    CDP_C6_10 = "CDP_C6_10"
    CDP_SECTOR = "CDP_SECTOR"
    SEC_1504 = "SEC_1504"
    SBTI = "SBTi"
    ISO_14064 = "ISO_14064"
    TCFD = "TCFD"
    GRI_305_4 = "GRI_305_4"
    IFRS_S2 = "IFRS_S2"

class RequirementLevel(str, Enum):
    """Level of requirement."""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"

class FieldStatus(str, Enum):
    """Status of a disclosure field."""
    POPULATED = "populated"
    MISSING = "missing"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"

# ---------------------------------------------------------------------------
# Constants -- Disclosure Requirements
# ---------------------------------------------------------------------------

DISCLOSURE_REQUIREMENTS: Dict[str, List[Dict[str, Any]]] = {
    Framework.ESRS_E1_6.value: [
        {
            "field_id": "esrs_e1_6_scope12_revenue",
            "field_name": "Scope 1+2 intensity per net revenue",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "intensity_metric",
            "scope": "scope_1_2",
            "denominator": "revenue",
            "reference": "ESRS E1-6 para 53",
        },
        {
            "field_id": "esrs_e1_6_scope123_revenue",
            "field_name": "Scope 1+2+3 intensity per net revenue",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "intensity_metric",
            "scope": "scope_1_2_3",
            "denominator": "revenue",
            "reference": "ESRS E1-6 para 53(b)",
        },
        {
            "field_id": "esrs_e1_6_sector_specific",
            "field_name": "Sector-specific intensity metric",
            "level": RequirementLevel.RECOMMENDED.value,
            "data_type": "intensity_metric",
            "scope": "scope_1_2",
            "denominator": "sector_specific",
            "reference": "ESRS E1-6 para 54",
        },
        {
            "field_id": "esrs_e1_6_base_year",
            "field_name": "Base year intensity values",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "base_year",
            "reference": "ESRS E1-6 para 55",
        },
        {
            "field_id": "esrs_e1_6_target_year",
            "field_name": "Target year intensity values",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "target",
            "reference": "ESRS E1-4 para 34",
        },
    ],
    Framework.CDP_C6_10.value: [
        {
            "field_id": "cdp_c6_10_metric1",
            "field_name": "Emissions intensity metric 1",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "intensity_metric",
            "scope": "any",
            "denominator": "any",
            "reference": "CDP C6.10",
        },
        {
            "field_id": "cdp_c6_10_metric2",
            "field_name": "Emissions intensity metric 2",
            "level": RequirementLevel.RECOMMENDED.value,
            "data_type": "intensity_metric",
            "scope": "any",
            "denominator": "any",
            "reference": "CDP C6.10",
        },
        {
            "field_id": "cdp_c6_10_previous_year",
            "field_name": "Previous year intensity for comparison",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "time_series",
            "reference": "CDP C6.10",
        },
        {
            "field_id": "cdp_c6_10_change_reason",
            "field_name": "Reason for intensity change",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "narrative",
            "reference": "CDP C6.10",
        },
    ],
    Framework.CDP_SECTOR.value: [
        {
            "field_id": "cdp_sector_metric",
            "field_name": "Sector-specific intensity metric",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "intensity_metric",
            "scope": "sector_specific",
            "denominator": "sector_specific",
            "reference": "CDP Sector Module",
        },
    ],
    Framework.SEC_1504.value: [
        {
            "field_id": "sec_1504_scope12_revenue",
            "field_name": "Scope 1+2 intensity per unit of revenue",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "intensity_metric",
            "scope": "scope_1_2",
            "denominator": "revenue",
            "reference": "SEC Final Rule 33-11275, Item 1504(c)(1)",
        },
        {
            "field_id": "sec_1504_scope3_revenue",
            "field_name": "Scope 3 intensity per unit of revenue",
            "level": RequirementLevel.RECOMMENDED.value,
            "data_type": "intensity_metric",
            "scope": "scope_3",
            "denominator": "revenue",
            "reference": "SEC Final Rule 33-11275, Item 1504(c)(1)",
        },
        {
            "field_id": "sec_1504_yoy",
            "field_name": "Year-over-year intensity comparison",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "time_series",
            "reference": "SEC Final Rule 33-11275, Item 1504(c)(2)",
        },
    ],
    Framework.SBTI.value: [
        {
            "field_id": "sbti_base_intensity",
            "field_name": "Base year sector intensity",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "base_year",
            "reference": "SBTi Corporate Manual v2.1, Section 4",
        },
        {
            "field_id": "sbti_target_intensity",
            "field_name": "Target year intensity",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "target",
            "reference": "SBTi Corporate Manual v2.1, Section 5",
        },
        {
            "field_id": "sbti_current_intensity",
            "field_name": "Current year intensity",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "intensity_metric",
            "reference": "SBTi Monitoring Report",
        },
        {
            "field_id": "sbti_annual_reduction",
            "field_name": "Annual reduction rate",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "metric",
            "reference": "SBTi SDA Tool",
        },
    ],
    Framework.ISO_14064.value: [
        {
            "field_id": "iso_physical_intensity",
            "field_name": "Physical intensity ratio",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "intensity_metric",
            "scope": "scope_1_2",
            "denominator": "physical",
            "reference": "ISO 14064-1:2018 Clause 5.3.4",
        },
        {
            "field_id": "iso_economic_intensity",
            "field_name": "Economic intensity ratio",
            "level": RequirementLevel.RECOMMENDED.value,
            "data_type": "intensity_metric",
            "scope": "scope_1_2",
            "denominator": "economic",
            "reference": "ISO 14064-1:2018 Clause 5.3.4",
        },
    ],
    Framework.TCFD.value: [
        {
            "field_id": "tcfd_intensity_revenue",
            "field_name": "GHG intensity per revenue",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "intensity_metric",
            "scope": "scope_1_2",
            "denominator": "revenue",
            "reference": "TCFD Recommendations, Metrics (b)",
        },
        {
            "field_id": "tcfd_sector_intensity",
            "field_name": "Sector-specific intensity (supplemental)",
            "level": RequirementLevel.RECOMMENDED.value,
            "data_type": "intensity_metric",
            "scope": "sector_specific",
            "denominator": "sector_specific",
            "reference": "TCFD Supplemental Guidance",
        },
        {
            "field_id": "tcfd_trend",
            "field_name": "Intensity trend over time",
            "level": RequirementLevel.RECOMMENDED.value,
            "data_type": "time_series",
            "reference": "TCFD Metrics (b)",
        },
    ],
    Framework.GRI_305_4.value: [
        {
            "field_id": "gri_305_4_ratio",
            "field_name": "GHG emissions intensity ratio",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "intensity_metric",
            "scope": "any",
            "denominator": "any",
            "reference": "GRI 305-4",
        },
        {
            "field_id": "gri_305_4_denominator",
            "field_name": "Organisation-specific denominator",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "denominator",
            "reference": "GRI 305-4",
        },
        {
            "field_id": "gri_305_4_scope_coverage",
            "field_name": "Scope coverage specification",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "narrative",
            "reference": "GRI 305-4",
        },
    ],
    Framework.IFRS_S2.value: [
        {
            "field_id": "ifrs_s2_intensity_revenue",
            "field_name": "GHG intensity per unit of revenue",
            "level": RequirementLevel.MANDATORY.value,
            "data_type": "intensity_metric",
            "scope": "scope_1_2",
            "denominator": "revenue",
            "reference": "IFRS S2 para 29(f)",
        },
        {
            "field_id": "ifrs_s2_industry_metric",
            "field_name": "Industry-based metric",
            "level": RequirementLevel.RECOMMENDED.value,
            "data_type": "intensity_metric",
            "scope": "sector_specific",
            "denominator": "sector_specific",
            "reference": "IFRS S2 Appendix B",
        },
    ],
}

# Remediation guidance templates
REMEDIATION_GUIDANCE: Dict[str, str] = {
    "intensity_metric": "Calculate the required intensity metric using the IntensityCalculationEngine.",
    "base_year": "Establish a base year using the BaseYearSelectionEngine (PACK-045).",
    "target": "Set an intensity target using the TargetPathwayEngine.",
    "time_series": "Provide at least two periods of data for year-over-year comparison.",
    "narrative": "Provide qualitative narrative describing the metric context.",
    "denominator": "Define and document the denominator using the DenominatorRegistryEngine.",
    "metric": "Calculate the required metric using the appropriate engine.",
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class AvailableMetric(BaseModel):
    """An intensity metric available for disclosure.

    Attributes:
        metric_id:          Metric identifier.
        scope_coverage:     Scope coverage (e.g. 'scope_1_2').
        denominator_type:   Denominator type (e.g. 'revenue', 'physical').
        denominator_id:     Specific denominator ID.
        intensity_value:    Intensity value.
        intensity_unit:     Intensity unit.
        period:             Reporting period.
        has_base_year:      Whether base year data is available.
        has_target:         Whether target data is available.
        has_time_series:    Whether multi-period data is available.
        has_narrative:      Whether qualitative narrative is available.
    """
    metric_id: str = Field(default="", description="Metric ID")
    scope_coverage: str = Field(default="scope_1_2", description="Scope coverage")
    denominator_type: str = Field(default="revenue", description="Denominator type")
    denominator_id: str = Field(default="", description="Denominator ID")
    intensity_value: Optional[Decimal] = Field(default=None, description="Intensity value")
    intensity_unit: str = Field(default="tCO2e/unit", description="Intensity unit")
    period: str = Field(default="2024", description="Reporting period")
    has_base_year: bool = Field(default=False, description="Has base year data")
    has_target: bool = Field(default=False, description="Has target data")
    has_time_series: bool = Field(default=False, description="Has time series")
    has_narrative: bool = Field(default=False, description="Has narrative")

class DisclosureInput(BaseModel):
    """Input for disclosure mapping analysis.

    Attributes:
        organisation_id:     Organisation identifier.
        target_frameworks:   Frameworks to assess.
        available_metrics:   Available intensity metrics.
        sector:              Organisation sector.
        output_precision:    Output decimal places.
    """
    organisation_id: str = Field(default="", description="Organisation ID")
    target_frameworks: List[Framework] = Field(..., min_length=1, description="Target frameworks")
    available_metrics: List[AvailableMetric] = Field(
        default_factory=list, description="Available metrics"
    )
    sector: str = Field(default="", description="Sector")
    output_precision: int = Field(default=2, ge=0, le=12, description="Precision")

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class FieldAssessment(BaseModel):
    """Assessment of a single disclosure field.

    Attributes:
        field_id:           Field identifier.
        field_name:         Field name.
        framework:          Framework.
        level:              Requirement level.
        status:             Field status.
        matched_metric_id:  ID of matching metric (if populated).
        remediation:        Remediation guidance (if missing).
        reference:          Regulatory reference.
    """
    field_id: str = Field(..., description="Field ID")
    field_name: str = Field(default="", description="Field name")
    framework: str = Field(default="", description="Framework")
    level: RequirementLevel = Field(default=RequirementLevel.MANDATORY, description="Level")
    status: FieldStatus = Field(default=FieldStatus.MISSING, description="Status")
    matched_metric_id: str = Field(default="", description="Matched metric ID")
    remediation: str = Field(default="", description="Remediation")
    reference: str = Field(default="", description="Reference")

class FrameworkCompleteness(BaseModel):
    """Completeness assessment for a single framework.

    Attributes:
        framework:              Framework identifier.
        total_requirements:     Total number of requirements.
        mandatory_count:        Number of mandatory requirements.
        populated_count:        Number of populated fields.
        mandatory_populated:    Number of mandatory fields populated.
        completeness_pct:       Overall completeness (%).
        mandatory_completeness_pct: Mandatory completeness (%).
        is_compliant:           Whether all mandatory fields are populated.
        field_assessments:      Per-field assessments.
    """
    framework: str = Field(..., description="Framework")
    total_requirements: int = Field(default=0, description="Total requirements")
    mandatory_count: int = Field(default=0, description="Mandatory count")
    populated_count: int = Field(default=0, description="Populated count")
    mandatory_populated: int = Field(default=0, description="Mandatory populated")
    completeness_pct: Decimal = Field(default=Decimal("0"), description="Completeness (%)")
    mandatory_completeness_pct: Decimal = Field(
        default=Decimal("0"), description="Mandatory completeness (%)"
    )
    is_compliant: bool = Field(default=False, description="Is compliant")
    field_assessments: List[FieldAssessment] = Field(
        default_factory=list, description="Field assessments"
    )

class FrameworkMapping(BaseModel):
    """Cross-framework mapping for a single metric.

    Attributes:
        metric_id:           Metric identifier.
        satisfies:           List of framework requirements satisfied.
    """
    metric_id: str = Field(..., description="Metric ID")
    satisfies: List[str] = Field(default_factory=list, description="Satisfied requirements")

class CompletenessReport(BaseModel):
    """Summary completeness report across all frameworks.

    Attributes:
        overall_completeness_pct:  Average completeness across frameworks.
        compliant_frameworks:      Number of fully compliant frameworks.
        total_gaps:                Total number of gaps.
        critical_gaps:             Number of critical (mandatory) gaps.
    """
    overall_completeness_pct: Decimal = Field(default=Decimal("0"), description="Overall (%)")
    compliant_frameworks: int = Field(default=0, description="Compliant count")
    total_gaps: int = Field(default=0, description="Total gaps")
    critical_gaps: int = Field(default=0, description="Critical gaps")

class DisclosureResult(BaseModel):
    """Result of disclosure mapping analysis.

    Attributes:
        result_id:             Unique result identifier.
        organisation_id:       Organisation identifier.
        framework_assessments: Per-framework completeness.
        cross_framework_map:   Cross-framework metric mappings.
        completeness_report:   Summary completeness report.
        gap_list:              List of all gaps with remediation.
        warnings:              Warnings.
        calculated_at:         Timestamp.
        processing_time_ms:    Processing time (ms).
        provenance_hash:       SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    organisation_id: str = Field(default="", description="Organisation ID")
    framework_assessments: List[FrameworkCompleteness] = Field(
        default_factory=list, description="Framework assessments"
    )
    cross_framework_map: List[FrameworkMapping] = Field(
        default_factory=list, description="Cross-framework map"
    )
    completeness_report: CompletenessReport = Field(
        default_factory=CompletenessReport, description="Completeness report"
    )
    gap_list: List[FieldAssessment] = Field(default_factory=list, description="Gaps")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DisclosureMappingEngine:
    """Maps intensity metrics to regulatory disclosure requirements.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: All mappings from published regulatory texts.
        - Zero-Hallucination: No LLM in any assessment path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("DisclosureMappingEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: DisclosureInput) -> DisclosureResult:
        """Perform disclosure mapping and completeness analysis.

        Args:
            input_data: Disclosure mapping input.

        Returns:
            DisclosureResult with completeness and gap analysis.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        framework_assessments: List[FrameworkCompleteness] = []
        all_gaps: List[FieldAssessment] = []
        total_compliant = 0

        for fw in input_data.target_frameworks:
            assessment = self._assess_framework(
                fw.value, input_data.available_metrics
            )
            framework_assessments.append(assessment)
            if assessment.is_compliant:
                total_compliant += 1

            gaps = [
                fa for fa in assessment.field_assessments
                if fa.status in (FieldStatus.MISSING, FieldStatus.PARTIAL)
            ]
            all_gaps.extend(gaps)

        # Cross-framework mapping
        cross_map = self._build_cross_framework_map(
            input_data.available_metrics, input_data.target_frameworks
        )

        # Completeness report
        total_pcts = [a.completeness_pct for a in framework_assessments]
        overall_pct = Decimal("0")
        if total_pcts:
            overall_pct = (
                sum(total_pcts) / Decimal(str(len(total_pcts)))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        critical_gaps = sum(
            1 for g in all_gaps if g.level == RequirementLevel.MANDATORY
        )

        report = CompletenessReport(
            overall_completeness_pct=overall_pct,
            compliant_frameworks=total_compliant,
            total_gaps=len(all_gaps),
            critical_gaps=critical_gaps,
        )

        if critical_gaps > 0:
            warnings.append(
                f"{critical_gaps} critical (mandatory) gap(s) identified across "
                f"{len(input_data.target_frameworks)} framework(s)."
            )

        if not input_data.available_metrics:
            warnings.append(
                "No intensity metrics provided. All framework fields are gaps."
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = DisclosureResult(
            organisation_id=input_data.organisation_id,
            framework_assessments=framework_assessments,
            cross_framework_map=cross_map,
            completeness_report=report,
            gap_list=all_gaps,
            warnings=warnings,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def get_framework_requirements(
        self, framework: str,
    ) -> List[Dict[str, Any]]:
        """Return disclosure requirements for a framework.

        Args:
            framework: Framework identifier.

        Returns:
            List of requirement definitions.
        """
        return list(DISCLOSURE_REQUIREMENTS.get(framework, []))

    def get_available_frameworks(self) -> List[str]:
        """Return list of supported frameworks."""
        return sorted(DISCLOSURE_REQUIREMENTS.keys())

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _assess_framework(
        self,
        framework: str,
        metrics: List[AvailableMetric],
    ) -> FrameworkCompleteness:
        """Assess completeness for a single framework."""
        requirements = DISCLOSURE_REQUIREMENTS.get(framework, [])
        if not requirements:
            return FrameworkCompleteness(framework=framework)

        field_assessments: List[FieldAssessment] = []
        mandatory_count = 0
        mandatory_populated = 0
        populated_count = 0

        for req in requirements:
            level = RequirementLevel(req.get("level", RequirementLevel.MANDATORY.value))
            if level == RequirementLevel.MANDATORY:
                mandatory_count += 1

            # Try to match
            status, matched_id = self._match_requirement(req, metrics)

            remediation = ""
            if status in (FieldStatus.MISSING, FieldStatus.PARTIAL):
                data_type = req.get("data_type", "metric")
                remediation = REMEDIATION_GUIDANCE.get(data_type, "")

            if status == FieldStatus.POPULATED:
                populated_count += 1
                if level == RequirementLevel.MANDATORY:
                    mandatory_populated += 1

            field_assessments.append(FieldAssessment(
                field_id=req.get("field_id", ""),
                field_name=req.get("field_name", ""),
                framework=framework,
                level=level,
                status=status,
                matched_metric_id=matched_id,
                remediation=remediation,
                reference=req.get("reference", ""),
            ))

        total = len(requirements)
        completeness = Decimal("0")
        if total > 0:
            completeness = (
                Decimal(str(populated_count)) / Decimal(str(total)) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        mandatory_comp = Decimal("0")
        if mandatory_count > 0:
            mandatory_comp = (
                Decimal(str(mandatory_populated)) / Decimal(str(mandatory_count)) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        return FrameworkCompleteness(
            framework=framework,
            total_requirements=total,
            mandatory_count=mandatory_count,
            populated_count=populated_count,
            mandatory_populated=mandatory_populated,
            completeness_pct=completeness,
            mandatory_completeness_pct=mandatory_comp,
            is_compliant=(mandatory_populated == mandatory_count),
            field_assessments=field_assessments,
        )

    def _match_requirement(
        self,
        requirement: Dict[str, Any],
        metrics: List[AvailableMetric],
    ) -> Tuple[FieldStatus, str]:
        """Try to match a requirement against available metrics.

        Returns (status, matched_metric_id).
        """
        data_type = requirement.get("data_type", "")
        req_scope = requirement.get("scope", "any")
        req_denom = requirement.get("denominator", "any")

        for metric in metrics:
            # Check data type
            if data_type == "intensity_metric":
                if metric.intensity_value is None:
                    continue
                # Check scope match
                if req_scope not in ("any", "sector_specific"):
                    if not self._scope_matches(req_scope, metric.scope_coverage):
                        continue
                # Check denominator match
                if req_denom not in ("any", "sector_specific"):
                    if not self._denominator_matches(req_denom, metric.denominator_type):
                        continue
                return FieldStatus.POPULATED, metric.metric_id

            elif data_type == "base_year":
                if metric.has_base_year:
                    return FieldStatus.POPULATED, metric.metric_id

            elif data_type == "target":
                if metric.has_target:
                    return FieldStatus.POPULATED, metric.metric_id

            elif data_type == "time_series":
                if metric.has_time_series:
                    return FieldStatus.POPULATED, metric.metric_id

            elif data_type == "narrative":
                if metric.has_narrative:
                    return FieldStatus.POPULATED, metric.metric_id

            elif data_type == "denominator":
                if metric.denominator_id:
                    return FieldStatus.POPULATED, metric.metric_id

            elif data_type == "metric":
                if metric.intensity_value is not None:
                    return FieldStatus.POPULATED, metric.metric_id

        return FieldStatus.MISSING, ""

    def _scope_matches(self, required: str, available: str) -> bool:
        """Check if available scope matches requirement."""
        # scope_1_2_3 satisfies scope_1_2 requirement
        if required == "scope_1_2" and available in ("scope_1_2", "scope_1_2_3", "scope_1_2_location", "scope_1_2_market"):
            return True
        if required == "scope_1_2_3" and available == "scope_1_2_3":
            return True
        if required == "scope_3" and available in ("scope_3", "scope_1_2_3"):
            return True
        if required == available:
            return True
        return False

    def _denominator_matches(self, required: str, available: str) -> bool:
        """Check if available denominator matches requirement."""
        if required == "revenue" and available in ("revenue", "economic"):
            return True
        if required == "physical" and available == "physical":
            return True
        if required == "economic" and available in ("revenue", "economic"):
            return True
        if required == available:
            return True
        return False

    def _build_cross_framework_map(
        self,
        metrics: List[AvailableMetric],
        frameworks: List[Framework],
    ) -> List[FrameworkMapping]:
        """Build cross-framework mapping showing which metrics satisfy which requirements."""
        mappings: List[FrameworkMapping] = []

        for metric in metrics:
            satisfies: List[str] = []
            for fw in frameworks:
                requirements = DISCLOSURE_REQUIREMENTS.get(fw.value, [])
                for req in requirements:
                    status, matched = self._match_requirement(req, [metric])
                    if status == FieldStatus.POPULATED:
                        satisfies.append(f"{fw.value}:{req.get('field_id', '')}")

            if satisfies:
                mappings.append(FrameworkMapping(
                    metric_id=metric.metric_id,
                    satisfies=satisfies,
                ))

        return mappings

    def get_version(self) -> str:
        return self._version

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "Framework",
    "RequirementLevel",
    "FieldStatus",
    # Input Models
    "AvailableMetric",
    "DisclosureInput",
    # Output Models
    "FieldAssessment",
    "FrameworkCompleteness",
    "FrameworkMapping",
    "CompletenessReport",
    "DisclosureResult",
    # Engine
    "DisclosureMappingEngine",
    # Constants
    "DISCLOSURE_REQUIREMENTS",
    "REMEDIATION_GUIDANCE",
]
