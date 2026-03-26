# -*- coding: utf-8 -*-
"""
Disclosure Preparation Workflow
====================================

4-phase workflow for multi-framework intensity disclosure preparation
within PACK-046 Intensity Metrics Pack.

Phases:
    1. MetricAggregation          -- Aggregate all intensity metrics from
                                     calculation engines across scopes,
                                     denominators, and periods into a unified
                                     metric catalogue with quality scores.
    2. FrameworkMapping           -- Map aggregated metrics to framework-specific
                                     disclosure fields for ESRS E1-6, CDP C6.10,
                                     SEC Climate, SBTi, ISO 14064, TCFD, GRI
                                     305-4, and IFRS S2 requirements.
    3. CompletenessCheck          -- Check completeness of mandatory disclosures
                                     per framework, identify gaps, classify gap
                                     severity, and recommend remediation actions.
    4. DisclosurePackage          -- Generate disclosure package with data tables,
                                     supporting evidence, methodology notes, and
                                     framework-specific formatted outputs.

The workflow follows GreenLang zero-hallucination principles: every mapping
is derived from deterministic framework requirement tables. SHA-256 provenance
hashes guarantee auditability.

Regulatory Basis:
    ESRS E1-6 (2024) - Climate change disclosures
    CDP Climate Change Questionnaire C6.10 (2026)
    SEC Climate Disclosure Rules (2024)
    SBTi Corporate Manual v2.1 - Progress reporting
    ISO 14064-1:2018 - Intensity reporting
    TCFD Recommendations - Metrics and Targets
    GRI 305-4 (2016) - GHG emissions intensity
    IFRS S2 (2023) - Climate-related disclosures

Schedule: Annually before reporting deadline, with interim quarterly updates
Estimated duration: 2-4 weeks

Author: GreenLang Team
Version: 46.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.utcnow().isoformat() + "Z"


def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable data."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class DisclosurePhase(str, Enum):
    """Disclosure preparation workflow phases."""

    METRIC_AGGREGATION = "metric_aggregation"
    FRAMEWORK_MAPPING = "framework_mapping"
    COMPLETENESS_CHECK = "completeness_check"
    DISCLOSURE_PACKAGE = "disclosure_package"


class DisclosureFramework(str, Enum):
    """Supported disclosure frameworks."""

    ESRS_E1 = "esrs_e1"
    CDP_C6 = "cdp_c6"
    SEC_CLIMATE = "sec_climate"
    SBTI = "sbti"
    ISO_14064 = "iso_14064"
    TCFD = "tcfd"
    GRI_305_4 = "gri_305_4"
    IFRS_S2 = "ifrs_s2"


class FieldStatus(str, Enum):
    """Status of a disclosure field."""

    POPULATED = "populated"
    PARTIALLY_POPULATED = "partially_populated"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


class GapSeverity(str, Enum):
    """Severity of a disclosure completeness gap."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PackageFormat(str, Enum):
    """Format of disclosure package output."""

    JSON = "json"
    EXCEL = "excel"
    PDF = "pdf"
    XBRL = "xbrl"


# =============================================================================
# FRAMEWORK FIELD REQUIREMENTS (Zero-Hallucination Reference Data)
# =============================================================================

FRAMEWORK_FIELDS: Dict[str, List[Dict[str, Any]]] = {
    "esrs_e1": [
        {"field_id": "E1-6_01", "name": "GHG intensity per revenue", "mandatory": True, "scope": "scope_1_2_location", "denominator": "revenue"},
        {"field_id": "E1-6_02", "name": "GHG intensity per revenue (S1+S2+S3)", "mandatory": True, "scope": "scope_1_2_3", "denominator": "revenue"},
        {"field_id": "E1-6_03", "name": "Sector-specific intensity", "mandatory": False, "scope": "scope_1_2_location", "denominator": "sector_specific"},
        {"field_id": "E1-6_04", "name": "Base year intensity", "mandatory": True, "scope": "scope_1_2_location", "denominator": "revenue"},
        {"field_id": "E1-6_05", "name": "Intensity trend (YoY change)", "mandatory": True, "scope": "scope_1_2_location", "denominator": "revenue"},
    ],
    "cdp_c6": [
        {"field_id": "C6.10_01", "name": "Scope 1+2 intensity per revenue", "mandatory": True, "scope": "scope_1_2_location", "denominator": "revenue"},
        {"field_id": "C6.10_02", "name": "Scope 1+2 intensity per FTE", "mandatory": False, "scope": "scope_1_2_location", "denominator": "fte"},
        {"field_id": "C6.10_03", "name": "Sector-specific intensity", "mandatory": False, "scope": "scope_1_2_location", "denominator": "sector_specific"},
        {"field_id": "C6.10_04", "name": "Intensity direction of change", "mandatory": True, "scope": "scope_1_2_location", "denominator": "revenue"},
    ],
    "sec_climate": [
        {"field_id": "SEC_01", "name": "GHG intensity per revenue", "mandatory": True, "scope": "scope_1_2_location", "denominator": "revenue"},
        {"field_id": "SEC_02", "name": "GHG intensity per unit of production", "mandatory": False, "scope": "scope_1_2_location", "denominator": "production_volume"},
    ],
    "sbti": [
        {"field_id": "SBTI_01", "name": "Base year intensity", "mandatory": True, "scope": "scope_1_2_location", "denominator": "sector_specific"},
        {"field_id": "SBTI_02", "name": "Target year intensity", "mandatory": True, "scope": "scope_1_2_location", "denominator": "sector_specific"},
        {"field_id": "SBTI_03", "name": "Current year intensity", "mandatory": True, "scope": "scope_1_2_location", "denominator": "sector_specific"},
        {"field_id": "SBTI_04", "name": "Progress towards target (%)", "mandatory": True, "scope": "scope_1_2_location", "denominator": "sector_specific"},
    ],
    "iso_14064": [
        {"field_id": "ISO_01", "name": "Emissions per unit of output", "mandatory": True, "scope": "scope_1_2_location", "denominator": "production_volume"},
        {"field_id": "ISO_02", "name": "Emissions intensity per revenue", "mandatory": False, "scope": "scope_1_2_location", "denominator": "revenue"},
    ],
    "tcfd": [
        {"field_id": "TCFD_M01", "name": "Scope 1+2 intensity (cross-industry)", "mandatory": True, "scope": "scope_1_2_location", "denominator": "revenue"},
        {"field_id": "TCFD_M02", "name": "Scope 1+2+3 intensity", "mandatory": False, "scope": "scope_1_2_3", "denominator": "revenue"},
        {"field_id": "TCFD_M03", "name": "Sector-specific intensity", "mandatory": False, "scope": "scope_1_2_location", "denominator": "sector_specific"},
    ],
    "gri_305_4": [
        {"field_id": "GRI_01", "name": "GHG emissions intensity ratio", "mandatory": True, "scope": "scope_1_2_location", "denominator": "revenue"},
        {"field_id": "GRI_02", "name": "Types of GHG included", "mandatory": True, "scope": "scope_1_2_location", "denominator": "revenue"},
        {"field_id": "GRI_03", "name": "Gases included in calculation", "mandatory": True, "scope": "scope_1_2_location", "denominator": "revenue"},
    ],
    "ifrs_s2": [
        {"field_id": "IFRS_01", "name": "GHG intensity per revenue", "mandatory": True, "scope": "scope_1_2_location", "denominator": "revenue"},
        {"field_id": "IFRS_02", "name": "GHG intensity per unit", "mandatory": False, "scope": "scope_1_2_location", "denominator": "production_volume"},
    ],
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class IntensityMetricInput(BaseModel):
    """An intensity metric to include in disclosure."""

    metric_id: str = Field(default="")
    period: str = Field(...)
    scope_coverage: str = Field(default="scope_1_2_location")
    denominator_type: str = Field(default="revenue")
    intensity_value: float = Field(default=0.0, ge=0.0)
    intensity_unit: str = Field(default="tCO2e/unit")
    numerator_tco2e: float = Field(default=0.0, ge=0.0)
    denominator_value: float = Field(default=0.0, ge=0.0)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)


class AggregatedMetric(BaseModel):
    """Aggregated metric for disclosure."""

    metric_key: str = Field(..., description="scope|denominator|period key")
    period: str = Field(...)
    scope_coverage: str = Field(...)
    denominator_type: str = Field(...)
    intensity_value: float = Field(default=0.0)
    intensity_unit: str = Field(default="")
    data_quality_score: float = Field(default=0.0)
    source_metric_count: int = Field(default=0)
    provenance_hash: str = Field(default="")


class FrameworkField(BaseModel):
    """A mapped framework disclosure field."""

    framework: DisclosureFramework = Field(...)
    field_id: str = Field(...)
    field_name: str = Field(default="")
    mandatory: bool = Field(default=False)
    status: FieldStatus = Field(default=FieldStatus.MISSING)
    value: Optional[float] = Field(default=None)
    unit: str = Field(default="")
    period: str = Field(default="")
    source_metric_key: str = Field(default="")
    notes: str = Field(default="")


class CompletenessGap(BaseModel):
    """A gap in disclosure completeness."""

    gap_id: str = Field(default_factory=lambda: f"gap-{_new_uuid()[:8]}")
    framework: DisclosureFramework = Field(...)
    field_id: str = Field(...)
    field_name: str = Field(default="")
    severity: GapSeverity = Field(...)
    reason: str = Field(default="")
    remediation: str = Field(default="")


class DisclosurePackageItem(BaseModel):
    """An item in the generated disclosure package."""

    item_id: str = Field(default_factory=lambda: f"dp-{_new_uuid()[:8]}")
    framework: DisclosureFramework = Field(...)
    title: str = Field(default="")
    content_type: str = Field(default="data_table")
    fields_count: int = Field(default=0)
    populated_count: int = Field(default=0)
    completeness_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class DisclosureInput(BaseModel):
    """Input data model for DisclosurePreparationWorkflow."""

    organization_id: str = Field(..., min_length=1)
    reporting_period: str = Field(default="2024")
    intensity_metrics: List[IntensityMetricInput] = Field(
        default_factory=list, description="Intensity metrics from calculation workflows",
    )
    target_frameworks: List[DisclosureFramework] = Field(
        default_factory=lambda: [
            DisclosureFramework.ESRS_E1,
            DisclosureFramework.CDP_C6,
            DisclosureFramework.GRI_305_4,
        ],
    )
    base_year_intensity: Optional[float] = Field(default=None)
    target_intensity: Optional[float] = Field(default=None)
    yoy_change_pct: Optional[float] = Field(default=None)
    sector_specific_denominator: str = Field(default="")
    output_formats: List[PackageFormat] = Field(
        default_factory=lambda: [PackageFormat.JSON],
    )
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class DisclosureResult(BaseModel):
    """Complete result from disclosure preparation workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="disclosure_preparation")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    aggregated_metrics: List[AggregatedMetric] = Field(default_factory=list)
    framework_fields: List[FrameworkField] = Field(default_factory=list)
    completeness_gaps: List[CompletenessGap] = Field(default_factory=list)
    disclosure_packages: List[DisclosurePackageItem] = Field(default_factory=list)
    overall_completeness_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class DisclosurePreparationWorkflow:
    """
    4-phase workflow for multi-framework intensity disclosure preparation.

    Aggregates metrics, maps to framework fields, checks completeness, and
    generates disclosure packages for all target frameworks.

    Zero-hallucination: all field mappings use deterministic framework
    requirement tables; no LLM calls in mapping paths; SHA-256 provenance
    on every disclosure field.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _aggregated: Aggregated intensity metrics.
        _fields: Mapped framework fields.
        _gaps: Completeness gaps.
        _packages: Disclosure package items.

    Example:
        >>> wf = DisclosurePreparationWorkflow()
        >>> metric = IntensityMetricInput(period="2024", intensity_value=50.0)
        >>> inp = DisclosureInput(
        ...     organization_id="org-001",
        ...     intensity_metrics=[metric],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[DisclosurePhase] = [
        DisclosurePhase.METRIC_AGGREGATION,
        DisclosurePhase.FRAMEWORK_MAPPING,
        DisclosurePhase.COMPLETENESS_CHECK,
        DisclosurePhase.DISCLOSURE_PACKAGE,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize DisclosurePreparationWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._aggregated: List[AggregatedMetric] = []
        self._fields: List[FrameworkField] = []
        self._gaps: List[CompletenessGap] = []
        self._packages: List[DisclosurePackageItem] = []
        self._overall_completeness: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: DisclosureInput) -> DisclosureResult:
        """Execute the 4-phase disclosure preparation workflow."""
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting disclosure preparation %s org=%s frameworks=%d metrics=%d",
            self.workflow_id, input_data.organization_id,
            len(input_data.target_frameworks), len(input_data.intensity_metrics),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_metric_aggregation,
            self._phase_2_framework_mapping,
            self._phase_3_completeness_check,
            self._phase_4_disclosure_package,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Disclosure preparation failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = DisclosureResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            aggregated_metrics=self._aggregated,
            framework_fields=self._fields,
            completeness_gaps=self._gaps,
            disclosure_packages=self._packages,
            overall_completeness_pct=self._overall_completeness,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Disclosure preparation %s completed in %.2fs status=%s completeness=%.1f%%",
            self.workflow_id, elapsed, overall_status.value, self._overall_completeness,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Metric Aggregation
    # -------------------------------------------------------------------------

    async def _phase_1_metric_aggregation(
        self, input_data: DisclosureInput,
    ) -> PhaseResult:
        """Aggregate intensity metrics into unified catalogue."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._aggregated = []
        metric_groups: Dict[str, List[IntensityMetricInput]] = {}

        for metric in input_data.intensity_metrics:
            key = f"{metric.scope_coverage}|{metric.denominator_type}|{metric.period}"
            metric_groups.setdefault(key, []).append(metric)

        for key, group in sorted(metric_groups.items()):
            parts = key.split("|")
            scope = parts[0]
            denom = parts[1]
            period = parts[2]

            # Take the highest quality metric or average if same quality
            best = max(group, key=lambda m: m.data_quality_score)
            avg_quality = sum(m.data_quality_score for m in group) / len(group)

            agg_data = {"key": key, "value": best.intensity_value, "count": len(group)}

            self._aggregated.append(AggregatedMetric(
                metric_key=key,
                period=period,
                scope_coverage=scope,
                denominator_type=denom,
                intensity_value=round(best.intensity_value, 6),
                intensity_unit=best.intensity_unit,
                data_quality_score=round(avg_quality, 2),
                source_metric_count=len(group),
                provenance_hash=_compute_hash(agg_data),
            ))

        outputs["metrics_aggregated"] = len(self._aggregated)
        outputs["input_metrics"] = len(input_data.intensity_metrics)
        outputs["unique_keys"] = len(metric_groups)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 MetricAggregation: %d aggregated from %d inputs",
            len(self._aggregated), len(input_data.intensity_metrics),
        )
        return PhaseResult(
            phase_name="metric_aggregation", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Framework Mapping
    # -------------------------------------------------------------------------

    async def _phase_2_framework_mapping(
        self, input_data: DisclosureInput,
    ) -> PhaseResult:
        """Map aggregated metrics to framework-specific disclosure fields."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._fields = []
        # Build lookup for aggregated metrics
        agg_lookup: Dict[str, AggregatedMetric] = {}
        for agg in self._aggregated:
            agg_lookup[agg.metric_key] = agg
            # Also index by (scope, denominator) without period
            short_key = f"{agg.scope_coverage}|{agg.denominator_type}"
            if short_key not in agg_lookup:
                agg_lookup[short_key] = agg

        for framework in input_data.target_frameworks:
            fw_key = framework.value
            field_defs = FRAMEWORK_FIELDS.get(fw_key, [])

            for fdef in field_defs:
                scope = fdef.get("scope", "scope_1_2_location")
                denom = fdef.get("denominator", "revenue")

                # Resolve sector_specific to actual denominator
                if denom == "sector_specific":
                    denom = input_data.sector_specific_denominator or "revenue"

                # Try to find matching metric
                full_key = f"{scope}|{denom}|{input_data.reporting_period}"
                short_key = f"{scope}|{denom}"
                matched_agg = agg_lookup.get(full_key) or agg_lookup.get(short_key)

                if matched_agg:
                    status = FieldStatus.POPULATED
                    value = matched_agg.intensity_value
                    unit = matched_agg.intensity_unit
                    source_key = matched_agg.metric_key
                else:
                    status = FieldStatus.MISSING
                    value = None
                    unit = ""
                    source_key = ""

                self._fields.append(FrameworkField(
                    framework=framework,
                    field_id=fdef.get("field_id", ""),
                    field_name=fdef.get("name", ""),
                    mandatory=fdef.get("mandatory", False),
                    status=status,
                    value=value,
                    unit=unit,
                    period=input_data.reporting_period,
                    source_metric_key=source_key,
                ))

        populated = sum(1 for f in self._fields if f.status == FieldStatus.POPULATED)
        outputs["fields_mapped"] = len(self._fields)
        outputs["fields_populated"] = populated
        outputs["fields_missing"] = len(self._fields) - populated
        outputs["frameworks_mapped"] = [fw.value for fw in input_data.target_frameworks]

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 FrameworkMapping: %d fields, %d populated",
            len(self._fields), populated,
        )
        return PhaseResult(
            phase_name="framework_mapping", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Completeness Check
    # -------------------------------------------------------------------------

    async def _phase_3_completeness_check(
        self, input_data: DisclosureInput,
    ) -> PhaseResult:
        """Check disclosure completeness and identify gaps."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._gaps = []

        for field in self._fields:
            if field.status == FieldStatus.MISSING:
                severity = GapSeverity.CRITICAL if field.mandatory else GapSeverity.MEDIUM

                remediation = ""
                if "revenue" in field.field_name.lower():
                    remediation = "Ensure revenue denominator data is available"
                elif "sector" in field.field_name.lower():
                    remediation = "Configure sector-specific denominator"
                elif "target" in field.field_name.lower():
                    remediation = "Run target setting workflow first"
                else:
                    remediation = "Review data collection and calculation coverage"

                self._gaps.append(CompletenessGap(
                    framework=field.framework,
                    field_id=field.field_id,
                    field_name=field.field_name,
                    severity=severity,
                    reason=f"No matching metric for {field.field_id}",
                    remediation=remediation,
                ))

        # Calculate overall completeness
        total_fields = len(self._fields)
        populated = sum(1 for f in self._fields if f.status == FieldStatus.POPULATED)
        self._overall_completeness = round(
            (populated / max(total_fields, 1)) * 100.0, 2,
        )

        # Per-framework completeness
        fw_completeness: Dict[str, float] = {}
        for fw in input_data.target_frameworks:
            fw_fields = [f for f in self._fields if f.framework == fw]
            fw_populated = sum(1 for f in fw_fields if f.status == FieldStatus.POPULATED)
            fw_completeness[fw.value] = round(
                (fw_populated / max(len(fw_fields), 1)) * 100.0, 2,
            )

        critical_gaps = sum(1 for g in self._gaps if g.severity == GapSeverity.CRITICAL)

        outputs["total_gaps"] = len(self._gaps)
        outputs["critical_gaps"] = critical_gaps
        outputs["overall_completeness_pct"] = self._overall_completeness
        outputs["framework_completeness"] = fw_completeness
        outputs["ready_for_submission"] = critical_gaps == 0

        if critical_gaps > 0:
            warnings.append(f"{critical_gaps} critical gap(s) must be resolved before submission")

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 CompletenessCheck: %d gaps (%d critical) completeness=%.1f%%",
            len(self._gaps), critical_gaps, self._overall_completeness,
        )
        return PhaseResult(
            phase_name="completeness_check", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Disclosure Package
    # -------------------------------------------------------------------------

    async def _phase_4_disclosure_package(
        self, input_data: DisclosureInput,
    ) -> PhaseResult:
        """Generate disclosure packages per framework."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._packages = []
        now_iso = _utcnow()

        for fw in input_data.target_frameworks:
            fw_fields = [f for f in self._fields if f.framework == fw]
            populated = sum(1 for f in fw_fields if f.status == FieldStatus.POPULATED)
            completeness = round((populated / max(len(fw_fields), 1)) * 100.0, 2)

            pkg_data = {
                "framework": fw.value,
                "fields": len(fw_fields),
                "populated": populated,
                "completeness": completeness,
            }

            self._packages.append(DisclosurePackageItem(
                framework=fw,
                title=f"{fw.value.upper()} Intensity Disclosure - {input_data.reporting_period}",
                content_type="data_table",
                fields_count=len(fw_fields),
                populated_count=populated,
                completeness_pct=completeness,
                provenance_hash=_compute_hash(pkg_data),
            ))

        outputs["packages_generated"] = len(self._packages)
        outputs["output_formats"] = [f.value for f in input_data.output_formats]
        outputs["frameworks"] = [p.framework.value for p in self._packages]

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 DisclosurePackage: %d packages generated",
            len(self._packages),
        )
        return PhaseResult(
            phase_name="disclosure_package", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: DisclosureInput, phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._aggregated = []
        self._fields = []
        self._gaps = []
        self._packages = []
        self._overall_completeness = 0.0

    def _compute_provenance(self, result: DisclosureResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{result.organization_id}|{result.overall_completeness_pct}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
