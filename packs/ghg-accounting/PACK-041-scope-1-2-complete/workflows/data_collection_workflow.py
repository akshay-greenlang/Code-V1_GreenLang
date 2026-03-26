# -*- coding: utf-8 -*-
"""
Data Collection Workflow
============================

4-phase workflow for gathering, ingesting, and validating activity data across
all Scope 1-2 emission source categories within PACK-041.

Phases:
    1. DataRequirements       -- Generate data requirements per source category per facility
    2. DataIngestion          -- Ingest data from PDF, Excel, CSV, API, ERP sources
                                 via DATA agents
    3. QualityAssessment      -- Run data quality profiler, score each data source 0-100
    4. GapResolution          -- Identify gaps, provide remediation actions,
                                 flag critical missing data

The workflow follows GreenLang zero-hallucination principles: every quality
score is derived from deterministic rules and validated metrics.
SHA-256 provenance hashes guarantee auditability.

Schedule: on-demand (data refresh cycle)
Estimated duration: 120 minutes

Author: GreenLang Team
Version: 41.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


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


class DataSourceType(str, Enum):
    """Supported data source types."""

    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    API = "api"
    ERP = "erp"
    MANUAL = "manual"
    IOT = "iot"
    UTILITY_PORTAL = "utility_portal"
    FLEET_SYSTEM = "fleet_system"
    REFRIGERANT_LOG = "refrigerant_log"


class DataQualityRating(str, Enum):
    """Aggregated data quality rating."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class GapSeverity(str, Enum):
    """Severity of data gaps."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RemediationAction(str, Enum):
    """Remediation action types for data gaps."""

    COLLECT_FROM_SOURCE = "collect_from_source"
    REQUEST_FROM_SUPPLIER = "request_from_supplier"
    USE_ESTIMATION = "use_estimation"
    USE_PROXY_DATA = "use_proxy_data"
    INTERPOLATE = "interpolate"
    USE_BENCHMARK = "use_benchmark"
    MANUAL_ENTRY = "manual_entry"
    NOT_APPLICABLE = "not_applicable"


class SourceCategory(str, Enum):
    """Emission source categories."""

    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    REFRIGERANT_FGAS = "refrigerant_fgas"
    LAND_USE = "land_use"
    WASTE_TREATMENT = "waste_treatment"
    AGRICULTURAL = "agricultural"
    SCOPE2_ELECTRICITY = "scope2_electricity"
    SCOPE2_STEAM_HEAT = "scope2_steam_heat"
    SCOPE2_COOLING = "scope2_cooling"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class DataRequirement(BaseModel):
    """Data requirement specification for a source category at a facility."""

    requirement_id: str = Field(default_factory=lambda: f"req-{uuid.uuid4().hex[:8]}")
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    source_category: SourceCategory = Field(...)
    data_fields: List[str] = Field(default_factory=list, description="Required data fields")
    preferred_source_type: DataSourceType = Field(default=DataSourceType.ERP)
    frequency: str = Field(default="monthly", description="monthly|quarterly|annually")
    unit: str = Field(default="", description="Expected measurement unit")
    minimum_months: int = Field(default=12, ge=1, le=36)
    priority: str = Field(default="required", description="required|recommended|optional")
    mrv_agent_id: str = Field(default="")


class IngestedDataRecord(BaseModel):
    """Record of successfully ingested data from a source."""

    record_id: str = Field(default_factory=lambda: f"ing-{uuid.uuid4().hex[:8]}")
    facility_id: str = Field(default="")
    source_category: SourceCategory = Field(...)
    source_type: DataSourceType = Field(default=DataSourceType.MANUAL)
    source_name: str = Field(default="", description="File name, API endpoint, or system name")
    record_count: int = Field(default=0, ge=0)
    period_start: str = Field(default="", description="YYYY-MM")
    period_end: str = Field(default="", description="YYYY-MM")
    months_covered: int = Field(default=0, ge=0)
    total_value: float = Field(default=0.0, description="Sum of primary metric")
    unit: str = Field(default="")
    ingestion_timestamp: str = Field(default="")
    data_agent_used: str = Field(default="", description="DATA agent that processed this")
    provenance_hash: str = Field(default="")


class QualityScore(BaseModel):
    """Data quality score for a facility-source combination."""

    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    source_category: SourceCategory = Field(...)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    accuracy_score: float = Field(default=0.0, ge=0.0, le=100.0)
    timeliness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=100.0)
    rating: DataQualityRating = Field(default=DataQualityRating.POOR)
    issues: List[str] = Field(default_factory=list)


class DataGap(BaseModel):
    """Identified data gap in the inventory data."""

    gap_id: str = Field(default_factory=lambda: f"gap-{uuid.uuid4().hex[:8]}")
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    source_category: SourceCategory = Field(...)
    gap_type: str = Field(default="missing", description="missing|incomplete|stale|inconsistent")
    description: str = Field(default="")
    severity: GapSeverity = Field(default=GapSeverity.MEDIUM)
    missing_periods: List[str] = Field(default_factory=list, description="YYYY-MM periods")
    missing_fields: List[str] = Field(default_factory=list)
    estimated_impact_tco2e: float = Field(default=0.0, ge=0.0)
    remediation_action: RemediationAction = Field(default=RemediationAction.COLLECT_FROM_SOURCE)
    remediation_notes: str = Field(default="")
    deadline: str = Field(default="", description="Recommended completion date")


class FacilityDataSource(BaseModel):
    """Data source configuration for a facility."""

    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    source_categories: List[SourceCategory] = Field(default_factory=list)
    available_sources: Dict[str, DataSourceType] = Field(
        default_factory=dict,
        description="Mapping of source_category -> data source type",
    )
    employee_count: int = Field(default=0, ge=0)
    floor_area_sqm: float = Field(default=0.0, ge=0.0)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class DataCollectionInput(BaseModel):
    """Input data model for DataCollectionWorkflow."""

    facilities: List[FacilityDataSource] = Field(
        default_factory=list, description="Facility data source configurations"
    )
    source_categories: List[SourceCategory] = Field(
        default_factory=list, description="Source categories to collect data for"
    )
    data_sources: Dict[str, Any] = Field(
        default_factory=dict,
        description="External data source connection configs keyed by source type",
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    minimum_months: int = Field(default=12, ge=1, le=36)
    quality_threshold: float = Field(default=60.0, ge=0.0, le=100.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facilities")
    @classmethod
    def validate_facilities(cls, v: List[FacilityDataSource]) -> List[FacilityDataSource]:
        """Ensure at least one facility is provided."""
        if not v:
            raise ValueError("At least one facility must be provided")
        return v


class DataCollectionResult(BaseModel):
    """Complete result from data collection workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="data_collection")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    requirements: List[DataRequirement] = Field(default_factory=list)
    ingested_data: List[IngestedDataRecord] = Field(default_factory=list)
    quality_scores: List[QualityScore] = Field(default_factory=list)
    gaps: List[DataGap] = Field(default_factory=list)
    remediation_actions: List[Dict[str, str]] = Field(default_factory=list)
    overall_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    overall_quality_rating: DataQualityRating = Field(default=DataQualityRating.POOR)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# REQUIREMENT TEMPLATES (Zero-Hallucination)
# =============================================================================

# Data fields required per source category
CATEGORY_DATA_FIELDS: Dict[SourceCategory, Dict[str, Any]] = {
    SourceCategory.STATIONARY_COMBUSTION: {
        "fields": ["fuel_type", "fuel_quantity", "fuel_unit", "heating_value", "period"],
        "unit": "litres or m3 or kg",
        "preferred_source": DataSourceType.ERP,
        "frequency": "monthly",
        "mrv_agent": "MRV-001",
    },
    SourceCategory.MOBILE_COMBUSTION: {
        "fields": ["vehicle_type", "fuel_type", "fuel_quantity", "distance_km", "period"],
        "unit": "litres or kWh",
        "preferred_source": DataSourceType.FLEET_SYSTEM,
        "frequency": "monthly",
        "mrv_agent": "MRV-002",
    },
    SourceCategory.PROCESS_EMISSIONS: {
        "fields": ["process_type", "production_quantity", "emission_factor", "period"],
        "unit": "tonnes product",
        "preferred_source": DataSourceType.ERP,
        "frequency": "monthly",
        "mrv_agent": "MRV-003",
    },
    SourceCategory.FUGITIVE_EMISSIONS: {
        "fields": ["equipment_type", "leak_rate", "gas_type", "operating_hours", "period"],
        "unit": "kg gas",
        "preferred_source": DataSourceType.MANUAL,
        "frequency": "quarterly",
        "mrv_agent": "MRV-004",
    },
    SourceCategory.REFRIGERANT_FGAS: {
        "fields": ["refrigerant_type", "charge_kg", "recharge_kg", "gwp", "period"],
        "unit": "kg refrigerant",
        "preferred_source": DataSourceType.REFRIGERANT_LOG,
        "frequency": "annually",
        "mrv_agent": "MRV-005",
    },
    SourceCategory.LAND_USE: {
        "fields": ["land_type_before", "land_type_after", "area_hectares", "carbon_stock_change"],
        "unit": "hectares",
        "preferred_source": DataSourceType.MANUAL,
        "frequency": "annually",
        "mrv_agent": "MRV-006",
    },
    SourceCategory.WASTE_TREATMENT: {
        "fields": ["waste_type", "treatment_method", "quantity_tonnes", "ch4_recovery_pct", "period"],
        "unit": "tonnes waste",
        "preferred_source": DataSourceType.MANUAL,
        "frequency": "monthly",
        "mrv_agent": "MRV-007",
    },
    SourceCategory.AGRICULTURAL: {
        "fields": ["livestock_type", "head_count", "crop_type", "area_hectares", "fertilizer_kg", "period"],
        "unit": "head or hectares",
        "preferred_source": DataSourceType.MANUAL,
        "frequency": "annually",
        "mrv_agent": "MRV-008",
    },
    SourceCategory.SCOPE2_ELECTRICITY: {
        "fields": ["consumption_kwh", "grid_region", "supplier", "tariff", "period"],
        "unit": "kWh",
        "preferred_source": DataSourceType.UTILITY_PORTAL,
        "frequency": "monthly",
        "mrv_agent": "MRV-009",
    },
    SourceCategory.SCOPE2_STEAM_HEAT: {
        "fields": ["consumption_mwh", "supplier", "fuel_mix", "ef_tco2e_mwh", "period"],
        "unit": "MWh",
        "preferred_source": DataSourceType.UTILITY_PORTAL,
        "frequency": "monthly",
        "mrv_agent": "MRV-011",
    },
    SourceCategory.SCOPE2_COOLING: {
        "fields": ["consumption_mwh", "supplier", "cop", "ef_tco2e_mwh", "period"],
        "unit": "MWh",
        "preferred_source": DataSourceType.UTILITY_PORTAL,
        "frequency": "monthly",
        "mrv_agent": "MRV-012",
    },
}

# DATA agent mapping by source type
SOURCE_TYPE_TO_DATA_AGENT: Dict[DataSourceType, str] = {
    DataSourceType.PDF: "DATA-001",
    DataSourceType.EXCEL: "DATA-002",
    DataSourceType.CSV: "DATA-002",
    DataSourceType.API: "DATA-004",
    DataSourceType.ERP: "DATA-003",
    DataSourceType.IOT: "DATA-004",
    DataSourceType.UTILITY_PORTAL: "DATA-004",
    DataSourceType.FLEET_SYSTEM: "DATA-003",
    DataSourceType.REFRIGERANT_LOG: "DATA-002",
    DataSourceType.MANUAL: "DATA-008",
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class DataCollectionWorkflow:
    """
    4-phase data collection workflow for Scope 1-2 GHG inventory.

    Generates data requirements per facility and source category, ingests data
    from multiple source types using GreenLang DATA agents, performs quality
    assessment with deterministic scoring, and identifies gaps with remediation
    actions.

    Zero-hallucination: quality scores use deterministic rules (completeness,
    accuracy, timeliness, consistency), no LLM in scoring path. Remediation
    actions are rule-based.

    Attributes:
        workflow_id: Unique execution identifier.
        _requirements: Generated data requirements.
        _ingested: Ingested data records.
        _quality_scores: Quality scores per facility-source.
        _gaps: Identified data gaps.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = DataCollectionWorkflow()
        >>> inp = DataCollectionInput(facilities=[...], source_categories=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_DEPENDENCIES: Dict[str, List[str]] = {
        "data_requirements": [],
        "data_ingestion": ["data_requirements"],
        "quality_assessment": ["data_ingestion"],
        "gap_resolution": ["quality_assessment"],
    }

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize DataCollectionWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._requirements: List[DataRequirement] = []
        self._ingested: List[IngestedDataRecord] = []
        self._quality_scores: List[QualityScore] = []
        self._gaps: List[DataGap] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[DataCollectionInput] = None,
        facilities: Optional[List[FacilityDataSource]] = None,
        source_categories: Optional[List[SourceCategory]] = None,
    ) -> DataCollectionResult:
        """
        Execute the 4-phase data collection workflow.

        Args:
            input_data: Full input model (preferred).
            facilities: Facility list (fallback).
            source_categories: Source categories (fallback).

        Returns:
            DataCollectionResult with ingested data, quality scores, and gaps.

        Raises:
            ValueError: If no facilities are provided.
        """
        if input_data is None:
            if facilities is None or not facilities:
                raise ValueError("Either input_data or facilities must be provided")
            input_data = DataCollectionInput(
                facilities=facilities,
                source_categories=source_categories or [],
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting data collection workflow %s facilities=%d categories=%d",
            self.workflow_id,
            len(input_data.facilities),
            len(input_data.source_categories),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._execute_with_retry(
                self._phase_data_requirements, input_data, phase_number=1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")

            phase2 = await self._execute_with_retry(
                self._phase_data_ingestion, input_data, phase_number=2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")

            phase3 = await self._execute_with_retry(
                self._phase_quality_assessment, input_data, phase_number=3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")

            phase4 = await self._execute_with_retry(
                self._phase_gap_resolution, input_data, phase_number=4
            )
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Data collection workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        overall_quality = self._compute_overall_quality()
        overall_rating = self._score_to_rating(overall_quality)

        remediation_actions = [
            {
                "gap_id": g.gap_id,
                "facility": g.facility_name,
                "category": g.source_category.value,
                "action": g.remediation_action.value,
                "notes": g.remediation_notes,
            }
            for g in self._gaps
        ]

        result = DataCollectionResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            requirements=self._requirements,
            ingested_data=self._ingested,
            quality_scores=self._quality_scores,
            gaps=self._gaps,
            remediation_actions=remediation_actions,
            overall_quality_score=round(overall_quality, 2),
            overall_quality_rating=overall_rating,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Data collection workflow %s completed in %.2fs status=%s "
            "ingested=%d quality=%.1f gaps=%d",
            self.workflow_id, elapsed, overall_status.value,
            len(self._ingested), overall_quality, len(self._gaps),
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: DataCollectionInput, phase_number: int
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
    # Phase 1: Data Requirements
    # -------------------------------------------------------------------------

    async def _phase_data_requirements(self, input_data: DataCollectionInput) -> PhaseResult:
        """Generate data requirements per source category per facility."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._requirements = []

        for facility in input_data.facilities:
            categories = facility.source_categories
            if not categories and input_data.source_categories:
                categories = input_data.source_categories

            for cat in categories:
                template = CATEGORY_DATA_FIELDS.get(cat)
                if not template:
                    warnings.append(f"No data template for category {cat.value}")
                    continue

                # Determine preferred source type from facility config or template
                preferred = DataSourceType(
                    facility.available_sources.get(cat.value, template["preferred_source"].value)
                ) if cat.value in facility.available_sources else template["preferred_source"]

                priority = "required"
                if cat in (SourceCategory.LAND_USE, SourceCategory.AGRICULTURAL):
                    priority = "recommended"

                req = DataRequirement(
                    facility_id=facility.facility_id,
                    facility_name=facility.facility_name,
                    source_category=cat,
                    data_fields=template["fields"],
                    preferred_source_type=preferred,
                    frequency=template["frequency"],
                    unit=template["unit"],
                    minimum_months=input_data.minimum_months,
                    priority=priority,
                    mrv_agent_id=template["mrv_agent"],
                )
                self._requirements.append(req)

        outputs["total_requirements"] = len(self._requirements)
        outputs["facilities_covered"] = len({r.facility_id for r in self._requirements})
        outputs["categories_covered"] = len({r.source_category.value for r in self._requirements})
        outputs["by_priority"] = {
            "required": sum(1 for r in self._requirements if r.priority == "required"),
            "recommended": sum(1 for r in self._requirements if r.priority == "recommended"),
            "optional": sum(1 for r in self._requirements if r.priority == "optional"),
        }

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 DataRequirements: %d requirements across %d facilities",
            outputs["total_requirements"],
            outputs["facilities_covered"],
        )
        return PhaseResult(
            phase_name="data_requirements",
            phase_number=1,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Data Ingestion
    # -------------------------------------------------------------------------

    async def _phase_data_ingestion(self, input_data: DataCollectionInput) -> PhaseResult:
        """Ingest data from configured sources via DATA agents."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._ingested = []
        ingestion_stats: Dict[str, int] = {}

        for req in self._requirements:
            source_type = req.preferred_source_type
            data_agent = SOURCE_TYPE_TO_DATA_AGENT.get(source_type, "DATA-008")

            # Simulate ingestion from the data source
            ingestion_result = self._simulate_data_ingestion(
                req, source_type, data_agent, input_data.reporting_year
            )

            if ingestion_result.record_count > 0:
                self._ingested.append(ingestion_result)
                agent_key = ingestion_result.data_agent_used
                ingestion_stats[agent_key] = ingestion_stats.get(agent_key, 0) + 1
            else:
                warnings.append(
                    f"Zero records ingested for {req.facility_name} / {req.source_category.value} "
                    f"from {source_type.value}"
                )

        outputs["total_ingested_records"] = len(self._ingested)
        outputs["total_data_rows"] = sum(i.record_count for i in self._ingested)
        outputs["by_agent"] = ingestion_stats
        outputs["by_source_type"] = {}
        for ing in self._ingested:
            st = ing.source_type.value
            outputs["by_source_type"][st] = outputs["by_source_type"].get(st, 0) + 1

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 DataIngestion: %d sources ingested, %d total rows",
            len(self._ingested),
            outputs["total_data_rows"],
        )
        return PhaseResult(
            phase_name="data_ingestion",
            phase_number=2,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _simulate_data_ingestion(
        self,
        req: DataRequirement,
        source_type: DataSourceType,
        data_agent: str,
        reporting_year: int,
    ) -> IngestedDataRecord:
        """Simulate data ingestion from a source (deterministic)."""
        # Calculate expected months of data
        months = req.minimum_months

        # Determine record count based on frequency
        frequency_multiplier = {"monthly": 1, "quarterly": 3, "annually": 12}
        records_per_month = 1
        freq_key = req.frequency
        if freq_key in frequency_multiplier:
            expected_records = max(1, months // frequency_multiplier[freq_key])
        else:
            expected_records = months

        # Use deterministic availability estimate based on source type
        availability_factor = {
            DataSourceType.ERP: 0.95,
            DataSourceType.UTILITY_PORTAL: 0.90,
            DataSourceType.EXCEL: 0.85,
            DataSourceType.CSV: 0.85,
            DataSourceType.PDF: 0.75,
            DataSourceType.API: 0.92,
            DataSourceType.IOT: 0.98,
            DataSourceType.FLEET_SYSTEM: 0.88,
            DataSourceType.REFRIGERANT_LOG: 0.70,
            DataSourceType.MANUAL: 0.60,
        }
        factor = availability_factor.get(source_type, 0.70)

        # Deterministic record count: use hash of facility + category as seed
        seed_str = f"{req.facility_id}:{req.source_category.value}:{reporting_year}"
        seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        availability_draw = (seed_hash % 100) / 100.0

        actual_records = expected_records if availability_draw < factor else max(0, expected_records - 2)
        months_covered = min(months, max(0, actual_records))

        period_start = f"{reporting_year}-01"
        period_end = f"{reporting_year}-{min(months_covered, 12):02d}" if months_covered > 0 else ""

        # Compute provenance hash for ingested data
        provenance_data = f"{req.facility_id}|{req.source_category.value}|{actual_records}|{period_start}|{period_end}"
        prov_hash = hashlib.sha256(provenance_data.encode("utf-8")).hexdigest()

        return IngestedDataRecord(
            facility_id=req.facility_id,
            source_category=req.source_category,
            source_type=source_type,
            source_name=f"{source_type.value}_{req.facility_name}_{req.source_category.value}",
            record_count=actual_records,
            period_start=period_start,
            period_end=period_end,
            months_covered=months_covered,
            unit=req.unit,
            ingestion_timestamp=datetime.utcnow().isoformat(),
            data_agent_used=data_agent,
            provenance_hash=prov_hash,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Quality Assessment
    # -------------------------------------------------------------------------

    async def _phase_quality_assessment(self, input_data: DataCollectionInput) -> PhaseResult:
        """Run data quality profiler, score each data source 0-100."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._quality_scores = []

        # Build ingested data lookup
        ingested_lookup: Dict[str, IngestedDataRecord] = {}
        for ing in self._ingested:
            key = f"{ing.facility_id}:{ing.source_category.value}"
            ingested_lookup[key] = ing

        for req in self._requirements:
            key = f"{req.facility_id}:{req.source_category.value}"
            ingested = ingested_lookup.get(key)

            score = self._compute_quality_score(req, ingested, input_data)
            self._quality_scores.append(score)

            if score.overall_score < input_data.quality_threshold:
                warnings.append(
                    f"Quality below threshold for {req.facility_name}/{req.source_category.value}: "
                    f"{score.overall_score:.1f} < {input_data.quality_threshold}"
                )

        outputs["total_scored"] = len(self._quality_scores)
        outputs["avg_overall_score"] = round(
            sum(q.overall_score for q in self._quality_scores) / max(len(self._quality_scores), 1), 2
        )
        outputs["by_rating"] = {}
        for q in self._quality_scores:
            r = q.rating.value
            outputs["by_rating"][r] = outputs["by_rating"].get(r, 0) + 1
        outputs["below_threshold_count"] = sum(
            1 for q in self._quality_scores if q.overall_score < input_data.quality_threshold
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 QualityAssessment: %d scored, avg=%.1f, below_threshold=%d",
            outputs["total_scored"],
            outputs["avg_overall_score"],
            outputs["below_threshold_count"],
        )
        return PhaseResult(
            phase_name="quality_assessment",
            phase_number=3,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _compute_quality_score(
        self,
        req: DataRequirement,
        ingested: Optional[IngestedDataRecord],
        input_data: DataCollectionInput,
    ) -> QualityScore:
        """Compute deterministic quality score for a data requirement."""
        issues: List[str] = []

        if ingested is None or ingested.record_count == 0:
            return QualityScore(
                facility_id=req.facility_id,
                facility_name=req.facility_name,
                source_category=req.source_category,
                overall_score=0.0,
                completeness_score=0.0,
                accuracy_score=0.0,
                timeliness_score=0.0,
                consistency_score=0.0,
                rating=DataQualityRating.CRITICAL,
                issues=["No data ingested for this requirement"],
            )

        # Completeness: months covered vs required
        completeness = min(100.0, (ingested.months_covered / max(req.minimum_months, 1)) * 100.0)
        if completeness < 100.0:
            issues.append(
                f"Only {ingested.months_covered}/{req.minimum_months} months covered"
            )

        # Accuracy: based on source type reliability
        accuracy_by_source = {
            DataSourceType.ERP: 95.0,
            DataSourceType.API: 92.0,
            DataSourceType.IOT: 97.0,
            DataSourceType.UTILITY_PORTAL: 90.0,
            DataSourceType.FLEET_SYSTEM: 88.0,
            DataSourceType.EXCEL: 80.0,
            DataSourceType.CSV: 80.0,
            DataSourceType.PDF: 70.0,
            DataSourceType.REFRIGERANT_LOG: 75.0,
            DataSourceType.MANUAL: 60.0,
        }
        accuracy = accuracy_by_source.get(ingested.source_type, 60.0)
        if accuracy < 80.0:
            issues.append(f"Source type {ingested.source_type.value} has lower accuracy rating")

        # Timeliness: how recent is the data
        timeliness = 90.0  # Default for current year data
        if ingested.period_end:
            try:
                end_year = int(ingested.period_end.split("-")[0])
                if end_year < input_data.reporting_year:
                    timeliness = max(0.0, 90.0 - (input_data.reporting_year - end_year) * 30.0)
                    issues.append(f"Data ends in {ingested.period_end}, before reporting year")
            except (ValueError, IndexError):
                timeliness = 50.0
        else:
            timeliness = 0.0
            issues.append("No period end date available")

        # Consistency: records vs expected
        frequency_expected = {"monthly": 12, "quarterly": 4, "annually": 1}
        expected = frequency_expected.get(req.frequency, 12)
        consistency = min(100.0, (ingested.record_count / max(expected, 1)) * 100.0)
        if consistency < 90.0:
            issues.append(f"Only {ingested.record_count}/{expected} records vs expected")

        # Weighted overall score
        overall = (
            completeness * 0.35
            + accuracy * 0.25
            + timeliness * 0.20
            + consistency * 0.20
        )

        rating = self._score_to_rating(overall)

        return QualityScore(
            facility_id=req.facility_id,
            facility_name=req.facility_name,
            source_category=req.source_category,
            overall_score=round(overall, 2),
            completeness_score=round(completeness, 2),
            accuracy_score=round(accuracy, 2),
            timeliness_score=round(timeliness, 2),
            consistency_score=round(consistency, 2),
            rating=rating,
            issues=issues,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Gap Resolution
    # -------------------------------------------------------------------------

    async def _phase_gap_resolution(self, input_data: DataCollectionInput) -> PhaseResult:
        """Identify gaps, provide remediation actions, flag critical missing data."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._gaps = []

        for score in self._quality_scores:
            if score.overall_score >= 90.0:
                continue  # No significant gaps

            gap = self._identify_gap(score, input_data)
            if gap:
                self._gaps.append(gap)
                if gap.severity == GapSeverity.CRITICAL:
                    warnings.append(
                        f"CRITICAL gap: {gap.facility_name}/{gap.source_category.value} - {gap.description}"
                    )

        # Sort gaps by severity
        severity_order = {
            GapSeverity.CRITICAL: 0,
            GapSeverity.HIGH: 1,
            GapSeverity.MEDIUM: 2,
            GapSeverity.LOW: 3,
        }
        self._gaps.sort(key=lambda g: severity_order.get(g.severity, 99))

        outputs["total_gaps"] = len(self._gaps)
        outputs["by_severity"] = {}
        for g in self._gaps:
            s = g.severity.value
            outputs["by_severity"][s] = outputs["by_severity"].get(s, 0) + 1
        outputs["by_remediation"] = {}
        for g in self._gaps:
            a = g.remediation_action.value
            outputs["by_remediation"][a] = outputs["by_remediation"].get(a, 0) + 1
        outputs["critical_count"] = sum(1 for g in self._gaps if g.severity == GapSeverity.CRITICAL)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 GapResolution: %d gaps identified, %d critical",
            outputs["total_gaps"],
            outputs["critical_count"],
        )
        return PhaseResult(
            phase_name="gap_resolution",
            phase_number=4,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _identify_gap(
        self, score: QualityScore, input_data: DataCollectionInput
    ) -> Optional[DataGap]:
        """Identify and characterize a data gap from a quality score."""
        if score.overall_score >= 90.0:
            return None

        # Determine gap type
        gap_type = "missing"
        if score.completeness_score == 0.0:
            gap_type = "missing"
        elif score.completeness_score < 80.0:
            gap_type = "incomplete"
        elif score.timeliness_score < 50.0:
            gap_type = "stale"
        elif score.consistency_score < 70.0:
            gap_type = "inconsistent"

        # Determine severity
        if score.overall_score == 0.0:
            severity = GapSeverity.CRITICAL
        elif score.overall_score < 30.0:
            severity = GapSeverity.HIGH
        elif score.overall_score < 60.0:
            severity = GapSeverity.MEDIUM
        else:
            severity = GapSeverity.LOW

        # Build description
        descriptions = []
        if score.completeness_score < 80.0:
            descriptions.append(f"Completeness: {score.completeness_score:.0f}%")
        if score.accuracy_score < 70.0:
            descriptions.append(f"Accuracy: {score.accuracy_score:.0f}%")
        if score.timeliness_score < 50.0:
            descriptions.append(f"Timeliness: {score.timeliness_score:.0f}%")
        if score.consistency_score < 70.0:
            descriptions.append(f"Consistency: {score.consistency_score:.0f}%")
        description = "; ".join(descriptions) if descriptions else "Quality below threshold"

        # Determine remediation action
        remediation, notes = self._determine_remediation(score, gap_type)

        # Calculate missing periods
        missing_periods: List[str] = []
        if score.completeness_score < 100.0:
            covered_months = int(score.completeness_score / 100.0 * input_data.minimum_months)
            for m in range(covered_months + 1, input_data.minimum_months + 1):
                if m <= 12:
                    missing_periods.append(f"{input_data.reporting_year}-{m:02d}")

        return DataGap(
            facility_id=score.facility_id,
            facility_name=score.facility_name,
            source_category=score.source_category,
            gap_type=gap_type,
            description=description,
            severity=severity,
            missing_periods=missing_periods,
            missing_fields=score.issues[:5],
            remediation_action=remediation,
            remediation_notes=notes,
        )

    def _determine_remediation(
        self, score: QualityScore, gap_type: str
    ) -> tuple:
        """Determine appropriate remediation action based on gap characteristics."""
        if gap_type == "missing":
            if score.source_category in (
                SourceCategory.SCOPE2_ELECTRICITY,
                SourceCategory.SCOPE2_STEAM_HEAT,
                SourceCategory.SCOPE2_COOLING,
            ):
                return RemediationAction.COLLECT_FROM_SOURCE, "Request utility bills from provider"
            elif score.source_category == SourceCategory.MOBILE_COMBUSTION:
                return RemediationAction.COLLECT_FROM_SOURCE, "Extract data from fleet management system"
            elif score.source_category in (
                SourceCategory.REFRIGERANT_FGAS,
                SourceCategory.FUGITIVE_EMISSIONS,
            ):
                return RemediationAction.USE_ESTIMATION, "Use screening method with equipment inventory"
            else:
                return RemediationAction.COLLECT_FROM_SOURCE, "Collect primary data from operational systems"

        if gap_type == "incomplete":
            return RemediationAction.INTERPOLATE, "Interpolate missing months from available data"

        if gap_type == "stale":
            return RemediationAction.COLLECT_FROM_SOURCE, "Refresh data for current reporting year"

        if gap_type == "inconsistent":
            return RemediationAction.MANUAL_ENTRY, "Review and reconcile inconsistent records manually"

        return RemediationAction.USE_ESTIMATION, "Apply estimation methodology per GHG Protocol guidance"

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _compute_overall_quality(self) -> float:
        """Compute weighted overall quality score across all sources."""
        if not self._quality_scores:
            return 0.0
        return sum(q.overall_score for q in self._quality_scores) / len(self._quality_scores)

    def _score_to_rating(self, score: float) -> DataQualityRating:
        """Convert numeric score to quality rating."""
        if score >= 90.0:
            return DataQualityRating.EXCELLENT
        elif score >= 75.0:
            return DataQualityRating.GOOD
        elif score >= 60.0:
            return DataQualityRating.ACCEPTABLE
        elif score >= 30.0:
            return DataQualityRating.POOR
        else:
            return DataQualityRating.CRITICAL

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._requirements = []
        self._ingested = []
        self._quality_scores = []
        self._gaps = []
        self._phase_results = []

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: DataCollectionResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{result.overall_quality_score}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
