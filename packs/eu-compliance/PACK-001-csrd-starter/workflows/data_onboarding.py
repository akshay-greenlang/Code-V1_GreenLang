# -*- coding: utf-8 -*-
"""
Data Onboarding Workflow
========================

Guided first-time data import and validation workflow for new CSRD
implementations. Walks the user through data source discovery, schema
mapping, sample validation, gap analysis, and full import with the
GreenLang data quality pipeline.

Designed for initial onboarding when an organization first connects to
GreenLang for CSRD reporting. Also usable for adding new data sources
mid-cycle.

Steps:
    1. Data source discovery (auto-detect available formats)
    2. Schema mapping (auto-map fields to ESRS data points)
    3. Sample validation (validate first 100 records)
    4. Gap analysis (which ESRS data points are missing)
    5. Data quality baseline
    6. Recommended actions report
    7. Full import with quality pipeline

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class OnboardingStepStatus(str, Enum):
    """Status of a data onboarding step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_INPUT = "needs_input"


class DataSourceType(str, Enum):
    """Supported data source types for onboarding."""
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    ERP_SAP = "erp_sap"
    ERP_ORACLE = "erp_oracle"
    ERP_DYNAMICS = "erp_dynamics"
    API_REST = "api_rest"
    DATABASE = "database"
    MANUAL_ENTRY = "manual_entry"
    UTILITY_BILL = "utility_bill"


class MappingConfidence(str, Enum):
    """Confidence level for auto-detected schema mappings."""
    HIGH = "high"          # >= 0.9
    MEDIUM = "medium"      # >= 0.7
    LOW = "low"            # >= 0.5
    UNMAPPED = "unmapped"  # < 0.5


class GapSeverity(str, Enum):
    """Severity of a missing ESRS data point."""
    CRITICAL = "critical"    # Mandatory disclosure, no data at all
    HIGH = "high"            # Mandatory with incomplete data
    MEDIUM = "medium"        # Recommended disclosure, missing
    LOW = "low"              # Optional, nice-to-have


class ActionPriority(str, Enum):
    """Priority for recommended onboarding actions."""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


# =============================================================================
# DATA MODELS
# =============================================================================


class DiscoveredSource(BaseModel):
    """A data source discovered during the discovery step."""
    source_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: DataSourceType = Field(...)
    source_name: str = Field(...)
    location: str = Field(..., description="File path, URL, or connection string")
    detected_format: str = Field(default="", description="Detected file/data format")
    estimated_records: int = Field(default=0, description="Estimated number of records")
    columns_detected: List[str] = Field(default_factory=list)
    usable: bool = Field(default=True, description="Whether this source is usable")
    issues: List[str] = Field(default_factory=list)


class FieldMapping(BaseModel):
    """Mapping between a source field and an ESRS data point."""
    source_field: str = Field(..., description="Field name in source data")
    esrs_data_point: str = Field(..., description="Target ESRS data point ID")
    esrs_label: str = Field(default="", description="Human-readable ESRS label")
    confidence: MappingConfidence = Field(...)
    confidence_score: float = Field(..., ge=0, le=1)
    transformation: Optional[str] = Field(
        None, description="Transformation to apply: unit_convert, date_parse, etc."
    )
    needs_review: bool = Field(default=False)


class DataGap(BaseModel):
    """An ESRS data point that has no source data mapped to it."""
    esrs_data_point: str = Field(...)
    esrs_label: str = Field(default="")
    esrs_standard: str = Field(default="")
    severity: GapSeverity = Field(...)
    is_mandatory: bool = Field(default=False)
    suggested_sources: List[str] = Field(
        default_factory=list, description="Suggested data sources to fill gap"
    )


class RecommendedAction(BaseModel):
    """A recommended action to improve data completeness or quality."""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(...)
    description: str = Field(...)
    priority: ActionPriority = Field(...)
    category: str = Field(default="data_collection")
    estimated_effort_hours: float = Field(default=0.0)
    affected_data_points: List[str] = Field(default_factory=list)


class DataOnboardingInput(BaseModel):
    """Input configuration for the data onboarding workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=2024, le=2050)
    source_paths: List[str] = Field(
        default_factory=list,
        description="File paths, URLs, or connection IDs to discover"
    )
    source_types: List[DataSourceType] = Field(
        default_factory=list,
        description="Specific source types to look for; empty = auto-detect all"
    )
    esrs_standards: List[str] = Field(
        default_factory=lambda: [
            "ESRS_E1", "ESRS_E2", "ESRS_E3", "ESRS_E4", "ESRS_E5",
            "ESRS_S1", "ESRS_S2", "ESRS_S3", "ESRS_S4",
            "ESRS_G1", "ESRS_G2",
        ],
        description="ESRS standards to map against"
    )
    sample_size: int = Field(
        default=100, ge=10, le=10000,
        description="Number of records to validate in sample step"
    )
    auto_import: bool = Field(
        default=False,
        description="Automatically proceed to full import if sample passes"
    )
    quality_threshold: float = Field(
        default=0.8, ge=0, le=1,
        description="Minimum quality score to proceed with auto-import"
    )

    @field_validator("source_paths")
    @classmethod
    def validate_source_paths(cls, v: List[str]) -> List[str]:
        """Ensure source paths are non-empty strings."""
        for path in v:
            if not path.strip():
                raise ValueError("Source paths must be non-empty strings")
        return v


class StepResult(BaseModel):
    """Result from a single onboarding step."""
    step_name: str = Field(...)
    status: OnboardingStepStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class DataOnboardingResult(BaseModel):
    """Complete result from the data onboarding workflow."""
    workflow_id: str = Field(...)
    status: OnboardingStepStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    steps: List[StepResult] = Field(default_factory=list)
    discovered_sources: List[DiscoveredSource] = Field(default_factory=list)
    field_mappings: List[FieldMapping] = Field(default_factory=list)
    data_gaps: List[DataGap] = Field(default_factory=list)
    recommended_actions: List[RecommendedAction] = Field(default_factory=list)
    quality_baseline: Dict[str, Any] = Field(default_factory=dict)
    import_summary: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# ESRS DATA POINT CATALOGUE (SUBSET)
# =============================================================================

ESRS_MANDATORY_DATA_POINTS: Dict[str, Dict[str, Any]] = {
    "E1_DP001": {"label": "GHG Scope 1 emissions", "standard": "ESRS_E1", "mandatory": True},
    "E1_DP002": {"label": "GHG Scope 2 emissions", "standard": "ESRS_E1", "mandatory": True},
    "E1_DP003": {"label": "GHG Scope 3 emissions", "standard": "ESRS_E1", "mandatory": True},
    "E1_DP004": {"label": "Total GHG emissions", "standard": "ESRS_E1", "mandatory": True},
    "E1_DP005": {"label": "GHG intensity (revenue)", "standard": "ESRS_E1", "mandatory": True},
    "E1_DP006": {"label": "Energy consumption total", "standard": "ESRS_E1", "mandatory": True},
    "E1_DP007": {"label": "Renewable energy share", "standard": "ESRS_E1", "mandatory": True},
    "E1_DP008": {"label": "Climate transition plan", "standard": "ESRS_E1", "mandatory": True},
    "E2_DP001": {"label": "Pollutants to air", "standard": "ESRS_E2", "mandatory": True},
    "E2_DP002": {"label": "Pollutants to water", "standard": "ESRS_E2", "mandatory": True},
    "E3_DP001": {"label": "Water consumption", "standard": "ESRS_E3", "mandatory": True},
    "E3_DP002": {"label": "Water withdrawal", "standard": "ESRS_E3", "mandatory": True},
    "E4_DP001": {"label": "Impact on biodiversity", "standard": "ESRS_E4", "mandatory": True},
    "E5_DP001": {"label": "Waste generated", "standard": "ESRS_E5", "mandatory": True},
    "E5_DP002": {"label": "Circular material rate", "standard": "ESRS_E5", "mandatory": True},
    "S1_DP001": {"label": "Total headcount", "standard": "ESRS_S1", "mandatory": True},
    "S1_DP002": {"label": "Gender pay gap", "standard": "ESRS_S1", "mandatory": True},
    "S1_DP003": {"label": "Training hours per employee", "standard": "ESRS_S1", "mandatory": True},
    "S1_DP004": {"label": "Work-related incidents", "standard": "ESRS_S1", "mandatory": True},
    "G1_DP001": {"label": "Corruption convictions", "standard": "ESRS_G1", "mandatory": True},
    "G1_DP002": {"label": "Lobbying expenditures", "standard": "ESRS_G1", "mandatory": True},
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class DataOnboardingWorkflow:
    """
    Guided first-time data import and validation workflow.

    Walks new users through a structured onboarding process that discovers
    available data sources, auto-maps fields to ESRS data points, validates
    a sample, identifies gaps, and performs the full import with quality
    assurance.

    Attributes:
        workflow_id: Unique execution identifier.
        _cancelled: Cancellation flag.
        _progress_callback: Optional callback for progress updates.

    Example:
        >>> wf = DataOnboardingWorkflow()
        >>> inp = DataOnboardingInput(
        ...     organization_id="org-123",
        ...     reporting_year=2025,
        ...     source_paths=["data/emissions_2025.xlsx", "data/hr_data.csv"],
        ... )
        >>> result = await wf.execute(inp)
        >>> print(f"Gaps found: {len(result.data_gaps)}")
    """

    STEPS = [
        "source_discovery",
        "schema_mapping",
        "sample_validation",
        "gap_analysis",
        "quality_baseline",
        "recommended_actions",
        "full_import",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the data onboarding workflow.

        Args:
            progress_callback: Optional callback(step_name, message, pct_complete).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._cancelled: bool = False
        self._progress_callback = progress_callback
        self._step_results: Dict[str, StepResult] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: DataOnboardingInput) -> DataOnboardingResult:
        """
        Execute the data onboarding workflow.

        Args:
            input_data: Validated onboarding input.

        Returns:
            DataOnboardingResult with discovered sources, mappings, gaps, actions.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting data onboarding %s for org=%s year=%d",
            self.workflow_id, input_data.organization_id, input_data.reporting_year,
        )
        self._notify("workflow", "Data onboarding started", 0.0)

        completed_steps: List[StepResult] = []
        overall_status = OnboardingStepStatus.RUNNING
        discovered_sources: List[DiscoveredSource] = []
        field_mappings: List[FieldMapping] = []
        data_gaps: List[DataGap] = []
        recommended_actions: List[RecommendedAction] = []
        quality_baseline: Dict[str, Any] = {}
        import_summary: Dict[str, Any] = {}

        step_handlers = [
            ("source_discovery", self._step_source_discovery),
            ("schema_mapping", self._step_schema_mapping),
            ("sample_validation", self._step_sample_validation),
            ("gap_analysis", self._step_gap_analysis),
            ("quality_baseline", self._step_quality_baseline),
            ("recommended_actions", self._step_recommended_actions),
            ("full_import", self._step_full_import),
        ]

        try:
            for idx, (step_name, handler) in enumerate(step_handlers):
                if self._cancelled:
                    overall_status = OnboardingStepStatus.SKIPPED
                    break

                # Skip full_import if auto_import is disabled and sample had issues
                if step_name == "full_import" and not input_data.auto_import:
                    sample_step = self._step_results.get("sample_validation")
                    quality_step = self._step_results.get("quality_baseline")
                    if quality_step and quality_step.artifacts:
                        score = quality_step.artifacts.get("overall_score", 0.0)
                        if score < input_data.quality_threshold:
                            skip_result = StepResult(
                                step_name=step_name,
                                status=OnboardingStepStatus.NEEDS_INPUT,
                                artifacts={
                                    "reason": (
                                        f"Quality score {score:.2f} below threshold "
                                        f"{input_data.quality_threshold}. "
                                        "Review recommended actions before importing."
                                    )
                                },
                                provenance_hash=self._hash({"skipped": True}),
                            )
                            completed_steps.append(skip_result)
                            self._step_results[step_name] = skip_result
                            continue

                pct = idx / len(step_handlers)
                self._notify(step_name, f"Starting: {step_name}", pct)
                step_started = datetime.utcnow()

                try:
                    step_result = await handler(input_data, pct)
                    step_result.started_at = step_started
                    step_result.completed_at = datetime.utcnow()
                    step_result.duration_seconds = (
                        step_result.completed_at - step_started
                    ).total_seconds()
                except Exception as exc:
                    logger.error("Step '%s' failed: %s", step_name, exc, exc_info=True)
                    step_result = StepResult(
                        step_name=step_name,
                        status=OnboardingStepStatus.FAILED,
                        started_at=step_started,
                        completed_at=datetime.utcnow(),
                        duration_seconds=(datetime.utcnow() - step_started).total_seconds(),
                        errors=[str(exc)],
                        provenance_hash=self._hash({"error": str(exc)}),
                    )

                completed_steps.append(step_result)
                self._step_results[step_name] = step_result

                # Extract typed outputs
                if step_name == "source_discovery" and step_result.artifacts:
                    raw = step_result.artifacts.get("sources", [])
                    discovered_sources = [
                        DiscoveredSource(**s) for s in raw if isinstance(s, dict)
                    ]
                if step_name == "schema_mapping" and step_result.artifacts:
                    raw = step_result.artifacts.get("mappings", [])
                    field_mappings = [
                        FieldMapping(**m) for m in raw if isinstance(m, dict)
                    ]
                if step_name == "gap_analysis" and step_result.artifacts:
                    raw = step_result.artifacts.get("gaps", [])
                    data_gaps = [DataGap(**g) for g in raw if isinstance(g, dict)]
                if step_name == "quality_baseline" and step_result.artifacts:
                    quality_baseline = step_result.artifacts
                if step_name == "recommended_actions" and step_result.artifacts:
                    raw = step_result.artifacts.get("actions", [])
                    recommended_actions = [
                        RecommendedAction(**a) for a in raw if isinstance(a, dict)
                    ]
                if step_name == "full_import" and step_result.artifacts:
                    import_summary = step_result.artifacts.get("import_summary", {})

                # Stop on failure for critical steps
                if step_result.status == OnboardingStepStatus.FAILED:
                    if step_name in ("source_discovery", "schema_mapping"):
                        overall_status = OnboardingStepStatus.FAILED
                        break

            if overall_status == OnboardingStepStatus.RUNNING:
                needs_input = any(
                    s.status == OnboardingStepStatus.NEEDS_INPUT for s in completed_steps
                )
                has_failure = any(
                    s.status == OnboardingStepStatus.FAILED for s in completed_steps
                )
                if needs_input:
                    overall_status = OnboardingStepStatus.NEEDS_INPUT
                elif has_failure:
                    overall_status = OnboardingStepStatus.FAILED
                else:
                    overall_status = OnboardingStepStatus.COMPLETED

        except Exception as exc:
            logger.critical("Data onboarding %s failed: %s", self.workflow_id, exc, exc_info=True)
            overall_status = OnboardingStepStatus.FAILED

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        metrics = {
            "sources_discovered": len(discovered_sources),
            "fields_mapped": len(field_mappings),
            "high_confidence_mappings": sum(
                1 for m in field_mappings if m.confidence == MappingConfidence.HIGH
            ),
            "data_gaps": len(data_gaps),
            "critical_gaps": sum(1 for g in data_gaps if g.severity == GapSeverity.CRITICAL),
            "recommended_actions": len(recommended_actions),
            "steps_completed": sum(
                1 for s in completed_steps if s.status == OnboardingStepStatus.COMPLETED
            ),
        }
        artifacts = {s.step_name: s.artifacts for s in completed_steps if s.artifacts}
        provenance = self._hash({
            "workflow_id": self.workflow_id,
            "steps": [s.provenance_hash for s in completed_steps],
        })

        self._notify("workflow", f"Onboarding {overall_status.value}", 1.0)

        return DataOnboardingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            steps=completed_steps,
            discovered_sources=discovered_sources,
            field_mappings=field_mappings,
            data_gaps=data_gaps,
            recommended_actions=recommended_actions,
            quality_baseline=quality_baseline,
            import_summary=import_summary,
            metrics=metrics,
            artifacts=artifacts,
            provenance_hash=provenance,
        )

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        logger.info("Cancellation requested for onboarding %s", self.workflow_id)
        self._cancelled = True

    # -------------------------------------------------------------------------
    # Step 1: Source Discovery
    # -------------------------------------------------------------------------

    async def _step_source_discovery(
        self, input_data: DataOnboardingInput, pct_base: float
    ) -> StepResult:
        """
        Auto-detect available data sources from provided paths and connections.

        Agents invoked:
            - greenlang.agents.data.document_ingestion_agent (file format detection)
            - greenlang.agents.data.erp_connector_agent (ERP connection probing)
            - greenlang.excel_csv_normalizer (spreadsheet structure detection)
        """
        step_name = "source_discovery"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Probing data sources", pct_base + 0.02)

        discovered: List[Dict[str, Any]] = []

        for source_path in input_data.source_paths:
            try:
                source_info = await self._probe_data_source(source_path)
                discovered.append(source_info)
            except Exception as exc:
                warnings.append(f"Could not probe source '{source_path}': {exc}")

        # If source types specified, also probe for those connection types
        if input_data.source_types:
            for src_type in input_data.source_types:
                type_sources = await self._discover_by_type(
                    input_data.organization_id, src_type
                )
                discovered.extend(type_sources)

        artifacts["sources"] = discovered
        artifacts["total_discovered"] = len(discovered)
        artifacts["usable_sources"] = sum(1 for s in discovered if s.get("usable", True))

        if not discovered:
            errors.append("No data sources discovered. Provide valid file paths or connection IDs.")

        provenance = self._hash(artifacts)
        status = OnboardingStepStatus.COMPLETED if not errors else OnboardingStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=len(discovered),
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 2: Schema Mapping
    # -------------------------------------------------------------------------

    async def _step_schema_mapping(
        self, input_data: DataOnboardingInput, pct_base: float
    ) -> StepResult:
        """
        Auto-map source fields to ESRS data points using semantic matching.

        Agents invoked:
            - greenlang.agents.foundation.schema_compiler (ESRS schema reference)
            - greenlang.agents.intelligence (semantic field matching via LLM)
            - greenlang.agents.foundation.unit_normalizer (unit detection)
        """
        step_name = "schema_mapping"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Auto-mapping fields to ESRS data points", pct_base + 0.02)

        discovery_step = self._step_results.get("source_discovery")
        sources = []
        if discovery_step and discovery_step.artifacts:
            sources = discovery_step.artifacts.get("sources", [])

        all_mappings: List[Dict[str, Any]] = []
        unmapped_fields: List[str] = []

        for source in sources:
            if not source.get("usable", True):
                continue
            columns = source.get("columns_detected", [])
            for col in columns:
                mapping = await self._auto_map_field(
                    col, source.get("source_type", ""), input_data.esrs_standards
                )
                if mapping:
                    all_mappings.append(mapping)
                    if mapping.get("confidence") == MappingConfidence.UNMAPPED.value:
                        unmapped_fields.append(col)
                else:
                    unmapped_fields.append(col)

        artifacts["mappings"] = all_mappings
        artifacts["total_mappings"] = len(all_mappings)
        artifacts["unmapped_fields"] = unmapped_fields

        # Statistics by confidence level
        conf_counts: Dict[str, int] = {c.value: 0 for c in MappingConfidence}
        for m in all_mappings:
            conf = m.get("confidence", MappingConfidence.UNMAPPED.value)
            conf_counts[conf] = conf_counts.get(conf, 0) + 1
        artifacts["confidence_distribution"] = conf_counts

        needs_review_count = sum(1 for m in all_mappings if m.get("needs_review", False))
        if needs_review_count > 0:
            warnings.append(
                f"{needs_review_count} field mapping(s) need manual review."
            )

        provenance = self._hash(artifacts)
        status = OnboardingStepStatus.COMPLETED if not errors else OnboardingStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=len(all_mappings),
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 3: Sample Validation
    # -------------------------------------------------------------------------

    async def _step_sample_validation(
        self, input_data: DataOnboardingInput, pct_base: float
    ) -> StepResult:
        """
        Validate the first N records (default 100) against ESRS schema and
        data quality rules.

        Agents invoked:
            - greenlang.agents.foundation.schema_compiler (type/format validation)
            - greenlang.validation_rule_engine (business rule checks)
            - greenlang.agents.foundation.unit_normalizer (unit consistency)
            - greenlang.outlier_detection (anomaly flagging)
        """
        step_name = "sample_validation"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, f"Validating {input_data.sample_size} sample records", pct_base + 0.02)

        discovery_step = self._step_results.get("source_discovery")
        mapping_step = self._step_results.get("schema_mapping")

        sources = []
        if discovery_step and discovery_step.artifacts:
            sources = discovery_step.artifacts.get("sources", [])

        mappings = []
        if mapping_step and mapping_step.artifacts:
            mappings = mapping_step.artifacts.get("mappings", [])

        # Validate sample from each source
        total_validated = 0
        total_errors = 0
        total_warnings_count = 0
        validation_details: List[Dict[str, Any]] = []

        for source in sources:
            if not source.get("usable", True):
                continue

            sample_result = await self._validate_sample(
                source, mappings, input_data.sample_size
            )
            total_validated += sample_result.get("records_validated", 0)
            total_errors += sample_result.get("validation_errors", 0)
            total_warnings_count += sample_result.get("validation_warnings", 0)
            validation_details.append({
                "source_name": source.get("source_name", ""),
                **sample_result,
            })

        # Compute pass rate
        pass_rate = (
            (total_validated - total_errors) / max(total_validated, 1)
        )
        artifacts["validation_details"] = validation_details
        artifacts["total_records_validated"] = total_validated
        artifacts["total_errors"] = total_errors
        artifacts["total_warnings"] = total_warnings_count
        artifacts["pass_rate"] = round(pass_rate, 4)

        if pass_rate < 0.5:
            warnings.append(
                f"Sample pass rate is {pass_rate*100:.1f}%. Significant data quality "
                "issues detected. Review field mappings and source data."
            )
        elif pass_rate < 0.8:
            warnings.append(
                f"Sample pass rate is {pass_rate*100:.1f}%. Some corrections recommended."
            )

        # Check for type mismatches, unit inconsistencies
        type_issues = await self._check_type_consistency(mappings, validation_details)
        artifacts["type_issues"] = type_issues

        provenance = self._hash(artifacts)
        status = OnboardingStepStatus.COMPLETED if not errors else OnboardingStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=total_validated,
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 4: Gap Analysis
    # -------------------------------------------------------------------------

    async def _step_gap_analysis(
        self, input_data: DataOnboardingInput, pct_base: float
    ) -> StepResult:
        """
        Identify which ESRS data points have no mapped source data.
        Cross-reference against the ESRS mandatory data point catalogue.
        """
        step_name = "gap_analysis"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Analyzing data gaps against ESRS catalogue", pct_base + 0.02)

        mapping_step = self._step_results.get("schema_mapping")
        mapped_points: set = set()
        if mapping_step and mapping_step.artifacts:
            for m in mapping_step.artifacts.get("mappings", []):
                conf = m.get("confidence", "")
                if conf != MappingConfidence.UNMAPPED.value:
                    mapped_points.add(m.get("esrs_data_point", ""))

        # Check all mandatory data points
        gaps: List[Dict[str, Any]] = []
        for dp_id, dp_info in ESRS_MANDATORY_DATA_POINTS.items():
            # Only check data points for included standards
            standard = dp_info.get("standard", "")
            if standard not in input_data.esrs_standards:
                continue

            if dp_id not in mapped_points:
                severity = GapSeverity.CRITICAL if dp_info.get("mandatory") else GapSeverity.MEDIUM
                suggested = await self._suggest_sources_for_gap(dp_id, dp_info)
                gaps.append({
                    "esrs_data_point": dp_id,
                    "esrs_label": dp_info.get("label", ""),
                    "esrs_standard": standard,
                    "severity": severity.value,
                    "is_mandatory": dp_info.get("mandatory", False),
                    "suggested_sources": suggested,
                })

        artifacts["gaps"] = gaps
        artifacts["total_gaps"] = len(gaps)
        artifacts["critical_gaps"] = sum(
            1 for g in gaps if g.get("severity") == GapSeverity.CRITICAL.value
        )
        artifacts["coverage_pct"] = round(
            len(mapped_points) / max(len(ESRS_MANDATORY_DATA_POINTS), 1) * 100, 1
        )

        if artifacts["critical_gaps"] > 0:
            warnings.append(
                f"{artifacts['critical_gaps']} critical ESRS data point(s) have no source data."
            )

        provenance = self._hash(artifacts)
        status = OnboardingStepStatus.COMPLETED if not errors else OnboardingStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=len(gaps),
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 5: Data Quality Baseline
    # -------------------------------------------------------------------------

    async def _step_quality_baseline(
        self, input_data: DataOnboardingInput, pct_base: float
    ) -> StepResult:
        """
        Establish a data quality baseline across all discovered sources.

        Agents invoked:
            - greenlang.data_quality_profiler (comprehensive profiling)
            - greenlang.duplicate_detection (dedup check)
            - greenlang.outlier_detection (statistical outlier detection)
        """
        step_name = "quality_baseline"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Establishing quality baseline", pct_base + 0.02)

        discovery_step = self._step_results.get("source_discovery")
        sources = []
        if discovery_step and discovery_step.artifacts:
            sources = discovery_step.artifacts.get("sources", [])

        # Run quality profiling
        quality_profile = await self._run_quality_profiling(
            input_data.organization_id, sources
        )

        # Run duplicate detection
        dedup_result = await self._run_duplicate_detection(
            input_data.organization_id, sources
        )

        # Run outlier detection
        outlier_result = await self._run_outlier_detection(
            input_data.organization_id, sources
        )

        # Compute overall quality score (weighted average)
        completeness = quality_profile.get("completeness_pct", 0.0) / 100.0
        accuracy = quality_profile.get("accuracy_score", 0.0)
        consistency = 1.0 - dedup_result.get("duplicate_rate", 0.0)
        validity = 1.0 - (outlier_result.get("outlier_rate", 0.0))

        # Weighted quality score
        overall = (
            completeness * 0.35 +
            accuracy * 0.25 +
            consistency * 0.20 +
            validity * 0.20
        )
        overall = round(min(max(overall, 0.0), 1.0), 4)

        artifacts["completeness"] = round(completeness, 4)
        artifacts["accuracy"] = round(accuracy, 4)
        artifacts["consistency"] = round(consistency, 4)
        artifacts["validity"] = round(validity, 4)
        artifacts["overall_score"] = overall
        artifacts["quality_profile"] = quality_profile
        artifacts["duplicates"] = dedup_result
        artifacts["outliers"] = outlier_result

        if overall < 0.5:
            warnings.append(
                f"Overall data quality score is {overall:.2f} (poor). "
                "Significant data cleaning required before import."
            )
        elif overall < 0.8:
            warnings.append(
                f"Overall data quality score is {overall:.2f} (fair). "
                "Some data quality improvements recommended."
            )

        provenance = self._hash(artifacts)
        status = OnboardingStepStatus.COMPLETED if not errors else OnboardingStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 6: Recommended Actions
    # -------------------------------------------------------------------------

    async def _step_recommended_actions(
        self, input_data: DataOnboardingInput, pct_base: float
    ) -> StepResult:
        """
        Generate a prioritized list of actions to improve data completeness
        and quality before proceeding with full import.
        """
        step_name = "recommended_actions"
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Generating recommended actions", pct_base + 0.02)

        actions: List[Dict[str, Any]] = []

        # Actions from gap analysis
        gap_step = self._step_results.get("gap_analysis")
        if gap_step and gap_step.artifacts:
            for gap in gap_step.artifacts.get("gaps", []):
                if gap.get("severity") in (GapSeverity.CRITICAL.value, GapSeverity.HIGH.value):
                    actions.append({
                        "action_id": str(uuid.uuid4()),
                        "title": f"Fill data gap: {gap.get('esrs_label', '')}",
                        "description": (
                            f"ESRS data point {gap.get('esrs_data_point', '')} "
                            f"({gap.get('esrs_label', '')}) has no source data. "
                            f"Suggested sources: {', '.join(gap.get('suggested_sources', []))}"
                        ),
                        "priority": ActionPriority.IMMEDIATE.value if gap.get("severity") == GapSeverity.CRITICAL.value else ActionPriority.SHORT_TERM.value,
                        "category": "data_collection",
                        "estimated_effort_hours": 4.0,
                        "affected_data_points": [gap.get("esrs_data_point", "")],
                    })

        # Actions from quality baseline
        quality_step = self._step_results.get("quality_baseline")
        if quality_step and quality_step.artifacts:
            qa = quality_step.artifacts
            if qa.get("completeness", 1.0) < 0.8:
                actions.append({
                    "action_id": str(uuid.uuid4()),
                    "title": "Improve data completeness",
                    "description": (
                        f"Data completeness is {qa.get('completeness', 0)*100:.1f}%. "
                        "Identify and import missing records."
                    ),
                    "priority": ActionPriority.IMMEDIATE.value,
                    "category": "data_quality",
                    "estimated_effort_hours": 8.0,
                    "affected_data_points": [],
                })
            dupes = qa.get("duplicates", {})
            if dupes.get("duplicate_rate", 0) > 0.05:
                actions.append({
                    "action_id": str(uuid.uuid4()),
                    "title": "Resolve duplicate records",
                    "description": (
                        f"Duplicate rate is {dupes.get('duplicate_rate', 0)*100:.1f}%. "
                        "Run deduplication before import."
                    ),
                    "priority": ActionPriority.SHORT_TERM.value,
                    "category": "data_quality",
                    "estimated_effort_hours": 4.0,
                    "affected_data_points": [],
                })

        # Actions from mapping review
        mapping_step = self._step_results.get("schema_mapping")
        if mapping_step and mapping_step.artifacts:
            needs_review = [
                m for m in mapping_step.artifacts.get("mappings", [])
                if m.get("needs_review", False)
            ]
            if needs_review:
                actions.append({
                    "action_id": str(uuid.uuid4()),
                    "title": f"Review {len(needs_review)} field mappings",
                    "description": (
                        f"{len(needs_review)} field mapping(s) have low confidence and "
                        "require manual verification."
                    ),
                    "priority": ActionPriority.IMMEDIATE.value,
                    "category": "schema_mapping",
                    "estimated_effort_hours": 2.0,
                    "affected_data_points": [
                        m.get("esrs_data_point", "") for m in needs_review
                    ],
                })

        # Sort by priority
        priority_order = {
            ActionPriority.IMMEDIATE.value: 0,
            ActionPriority.SHORT_TERM.value: 1,
            ActionPriority.MEDIUM_TERM.value: 2,
            ActionPriority.LONG_TERM.value: 3,
        }
        actions.sort(key=lambda a: priority_order.get(a.get("priority", ""), 99))

        artifacts["actions"] = actions
        artifacts["total_actions"] = len(actions)
        artifacts["immediate_actions"] = sum(
            1 for a in actions if a.get("priority") == ActionPriority.IMMEDIATE.value
        )
        artifacts["estimated_total_hours"] = sum(
            a.get("estimated_effort_hours", 0) for a in actions
        )

        provenance = self._hash(artifacts)

        return StepResult(
            step_name=step_name,
            status=OnboardingStepStatus.COMPLETED,
            artifacts=artifacts,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 7: Full Import
    # -------------------------------------------------------------------------

    async def _step_full_import(
        self, input_data: DataOnboardingInput, pct_base: float
    ) -> StepResult:
        """
        Perform full data import with the GreenLang data quality pipeline.

        Agents invoked:
            - greenlang.agents.data.document_ingestion_agent (document import)
            - greenlang.excel_csv_normalizer (spreadsheet normalization)
            - greenlang.agents.data.erp_connector_agent (ERP sync)
            - greenlang.validation_rule_engine (full validation)
            - greenlang.data_lineage_tracker (lineage registration)
            - greenlang.duplicate_detection (dedup enforcement)
            - greenlang.missing_value_imputer (gap filling)
        """
        step_name = "full_import"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Starting full data import", pct_base + 0.02)

        discovery_step = self._step_results.get("source_discovery")
        mapping_step = self._step_results.get("schema_mapping")

        sources = []
        if discovery_step and discovery_step.artifacts:
            sources = discovery_step.artifacts.get("sources", [])

        mappings = []
        if mapping_step and mapping_step.artifacts:
            mappings = mapping_step.artifacts.get("mappings", [])

        total_imported = 0
        total_rejected = 0
        import_details: List[Dict[str, Any]] = []

        for source in sources:
            if not source.get("usable", True):
                continue

            self._notify(
                step_name,
                f"Importing: {source.get('source_name', '')}",
                pct_base + 0.04,
            )

            try:
                result = await self._import_source(
                    input_data.organization_id,
                    source,
                    mappings,
                    input_data.reporting_year,
                )
                total_imported += result.get("records_imported", 0)
                total_rejected += result.get("records_rejected", 0)
                import_details.append({
                    "source_name": source.get("source_name", ""),
                    **result,
                })
            except Exception as exc:
                errors.append(f"Import failed for '{source.get('source_name')}': {exc}")

        # Register lineage
        lineage_id = await self._register_import_lineage(
            input_data.organization_id, import_details
        )

        import_summary = {
            "total_records_imported": total_imported,
            "total_records_rejected": total_rejected,
            "sources_imported": len(import_details),
            "lineage_id": lineage_id,
            "import_timestamp": datetime.utcnow().isoformat(),
        }
        artifacts["import_summary"] = import_summary
        artifacts["import_details"] = import_details

        if total_rejected > 0:
            warnings.append(
                f"{total_rejected} record(s) rejected during import. "
                "Review rejected records for data quality issues."
            )

        provenance = self._hash(artifacts)
        status = OnboardingStepStatus.COMPLETED if not errors else OnboardingStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=total_imported + total_rejected,
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Agent Invocation Helpers
    # -------------------------------------------------------------------------

    async def _probe_data_source(self, source_path: str) -> Dict[str, Any]:
        """Probe a data source to detect format, columns, and record count."""
        logger.info("Probing data source: %s", source_path)
        await asyncio.sleep(0)
        # Detect type from extension
        lower = source_path.lower()
        if lower.endswith(".csv"):
            src_type = DataSourceType.CSV.value
        elif lower.endswith((".xlsx", ".xls")):
            src_type = DataSourceType.EXCEL.value
        elif lower.endswith(".pdf"):
            src_type = DataSourceType.PDF.value
        else:
            src_type = DataSourceType.MANUAL_ENTRY.value

        return {
            "source_id": str(uuid.uuid4()),
            "source_type": src_type,
            "source_name": source_path.split("/")[-1].split("\\")[-1],
            "location": source_path,
            "detected_format": src_type,
            "estimated_records": 0,
            "columns_detected": [],
            "usable": True,
            "issues": [],
        }

    async def _discover_by_type(
        self, org_id: str, src_type: DataSourceType
    ) -> List[Dict[str, Any]]:
        """Discover available sources of a specific type."""
        await asyncio.sleep(0)
        return []

    async def _auto_map_field(
        self, column_name: str, source_type: str, standards: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Auto-map a source field to an ESRS data point using semantic matching."""
        await asyncio.sleep(0)
        return {
            "source_field": column_name,
            "esrs_data_point": "",
            "esrs_label": "",
            "confidence": MappingConfidence.UNMAPPED.value,
            "confidence_score": 0.0,
            "transformation": None,
            "needs_review": True,
        }

    async def _validate_sample(
        self, source: Dict[str, Any], mappings: List[Dict[str, Any]], sample_size: int
    ) -> Dict[str, Any]:
        """Validate a sample of records from a source."""
        await asyncio.sleep(0)
        return {
            "records_validated": 0,
            "validation_errors": 0,
            "validation_warnings": 0,
        }

    async def _check_type_consistency(
        self, mappings: List[Dict[str, Any]], validation_details: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check for type mismatches in mapped fields."""
        await asyncio.sleep(0)
        return []

    async def _suggest_sources_for_gap(
        self, dp_id: str, dp_info: Dict[str, Any]
    ) -> List[str]:
        """Suggest potential data sources to fill a gap."""
        await asyncio.sleep(0)
        label = dp_info.get("label", "")
        suggestions = []
        if "GHG" in label or "emissions" in label.lower():
            suggestions = ["ERP energy module", "Utility bills", "Fleet management system"]
        elif "workforce" in label.lower() or "employee" in label.lower():
            suggestions = ["HR/HCM system", "Payroll system"]
        elif "water" in label.lower():
            suggestions = ["Utility bills", "SCADA/BMS"]
        elif "waste" in label.lower():
            suggestions = ["Waste hauler reports", "ERP waste module"]
        else:
            suggestions = ["Manual data collection", "Supplier questionnaire"]
        return suggestions

    async def _run_quality_profiling(
        self, org_id: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run comprehensive quality profiling on all sources."""
        await asyncio.sleep(0)
        return {"completeness_pct": 0.0, "accuracy_score": 0.0}

    async def _run_duplicate_detection(
        self, org_id: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run duplicate detection across all sources."""
        await asyncio.sleep(0)
        return {"duplicate_rate": 0.0, "duplicates_found": 0}

    async def _run_outlier_detection(
        self, org_id: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run outlier detection on numeric fields."""
        await asyncio.sleep(0)
        return {"outlier_rate": 0.0, "outliers_found": 0}

    async def _import_source(
        self, org_id: str, source: Dict[str, Any],
        mappings: List[Dict[str, Any]], year: int,
    ) -> Dict[str, Any]:
        """Import a single data source using the quality pipeline."""
        await asyncio.sleep(0)
        return {"records_imported": 0, "records_rejected": 0, "status": "completed"}

    async def _register_import_lineage(
        self, org_id: str, import_details: List[Dict[str, Any]]
    ) -> str:
        """Register import lineage for audit trail."""
        await asyncio.sleep(0)
        return str(uuid.uuid4())

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _notify(self, step: str, message: str, pct: float) -> None:
        """Send progress notification."""
        if self._progress_callback:
            try:
                self._progress_callback(step, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for step=%s", step)

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
