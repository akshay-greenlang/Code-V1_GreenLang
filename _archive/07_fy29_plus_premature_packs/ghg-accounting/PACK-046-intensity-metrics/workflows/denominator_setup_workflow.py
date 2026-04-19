# -*- coding: utf-8 -*-
"""
Denominator Setup Workflow
====================================

4-phase workflow for denominator identification, selection, data collection,
and validation within PACK-046 Intensity Metrics Pack.

Phases:
    1. SectorIdentification       -- Identify the reporting entity's sector,
                                     sub-sector, and applicable regulatory
                                     frameworks to determine which denominators
                                     are relevant (e.g. revenue, FTE, floor area,
                                     MWh generated, tonnes produced).
    2. DenominatorSelection       -- Run DenominatorRegistryEngine recommendations,
                                     rank candidates by framework applicability and
                                     data availability, select primary and secondary
                                     denominators for each scope/framework combination.
    3. DataCollection             -- Collect denominator values for all selected
                                     denominators across all reporting periods,
                                     normalise units, and reconcile across sources.
    4. Validation                 -- Validate data quality, completeness, temporal
                                     consistency, and cross-source agreement; flag
                                     anomalies and produce a readiness assessment.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GRI 305-4 (2016) - GHG emissions intensity denominator requirements
    ESRS E1-6 - Sector-specific denominator specifications
    CDP C6.10 (2026) - Denominator selection guidance
    SBTi SDA v2.0 - Sector pathway denominator definitions
    ISO 14064-1:2018 Clause 5 - Quantification per unit of output

Schedule: Once during initial setup, annually for new denominators
Estimated duration: 1-3 weeks depending on data availability

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
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas.enums import ValidationSeverity

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

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

class SetupPhase(str, Enum):
    """Denominator setup workflow phases."""

    SECTOR_IDENTIFICATION = "sector_identification"
    DENOMINATOR_SELECTION = "denominator_selection"
    DATA_COLLECTION = "data_collection"
    VALIDATION = "validation"

class SectorClassification(str, Enum):
    """High-level sector classification for denominator mapping."""

    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    TRANSPORTATION = "transportation"
    OTHER = "other"

class DenominatorType(str, Enum):
    """Type of denominator used for intensity metrics."""

    REVENUE = "revenue"
    FTE = "fte"
    FLOOR_AREA = "floor_area"
    PRODUCTION_VOLUME = "production_volume"
    ENERGY_GENERATED = "energy_generated"
    TONNE_PRODUCT = "tonne_product"
    PASSENGER_KM = "passenger_km"
    TONNE_KM = "tonne_km"
    BEDS = "beds"
    ROOMS = "rooms"
    ASSETS_UNDER_MANAGEMENT = "assets_under_management"
    CUSTOM = "custom"

class DenominatorUnit(str, Enum):
    """Unit of measurement for denominators."""

    USD_MILLION = "USD_million"
    EUR_MILLION = "EUR_million"
    GBP_MILLION = "GBP_million"
    HEADCOUNT = "headcount"
    SQM = "sqm"
    SQFT = "sqft"
    TONNES = "tonnes"
    MWH = "MWh"
    GWH = "GWh"
    PASSENGER_KM = "passenger_km"
    TONNE_KM = "tonne_km"
    UNIT_COUNT = "unit_count"
    CUSTOM = "custom"

class DataQualityGrade(str, Enum):
    """Data quality grade for collected denominator values."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"

# =============================================================================
# SECTOR DENOMINATOR MAPPING (Zero-Hallucination Reference Data)
# =============================================================================

SECTOR_DENOMINATOR_MAP: Dict[str, List[str]] = {
    "energy": ["energy_generated", "revenue", "fte"],
    "materials": ["tonne_product", "revenue", "fte"],
    "industrials": ["revenue", "production_volume", "fte"],
    "consumer_discretionary": ["revenue", "fte", "floor_area"],
    "consumer_staples": ["tonne_product", "revenue", "fte"],
    "healthcare": ["revenue", "beds", "fte"],
    "financials": ["revenue", "assets_under_management", "fte"],
    "information_technology": ["revenue", "fte", "floor_area"],
    "communication_services": ["revenue", "fte", "floor_area"],
    "utilities": ["energy_generated", "revenue", "fte"],
    "real_estate": ["floor_area", "revenue", "rooms"],
    "transportation": ["passenger_km", "tonne_km", "revenue"],
    "other": ["revenue", "fte"],
}

FRAMEWORK_REQUIRED_DENOMINATORS: Dict[str, List[str]] = {
    "esrs_e1": ["revenue", "sector_specific"],
    "cdp_c6": ["revenue", "sector_specific"],
    "sec_climate": ["revenue"],
    "sbti_sda": ["sector_specific"],
    "iso_14064": ["revenue", "production_volume"],
    "tcfd": ["revenue", "sector_specific"],
    "gri_305_4": ["revenue", "production_volume"],
    "ifrs_s2": ["revenue"],
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

class SectorProfile(BaseModel):
    """Identified sector profile for the reporting entity."""

    sector: SectorClassification = Field(...)
    sub_sector: str = Field(default="", description="Sub-sector or NACE/SIC code")
    applicable_frameworks: List[str] = Field(default_factory=list)
    recommended_denominators: List[str] = Field(default_factory=list)
    sector_specific_denominator: str = Field(default="")
    notes: str = Field(default="")

class DenominatorCandidate(BaseModel):
    """A candidate denominator recommended for the entity."""

    denominator_type: DenominatorType = Field(...)
    unit: DenominatorUnit = Field(default=DenominatorUnit.CUSTOM)
    relevance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    frameworks_requiring: List[str] = Field(default_factory=list)
    data_available: bool = Field(default=False)
    selected: bool = Field(default=False)
    selection_reason: str = Field(default="")

class DenominatorRecord(BaseModel):
    """Collected denominator value for a specific period."""

    denominator_type: DenominatorType = Field(...)
    unit: DenominatorUnit = Field(...)
    period: str = Field(..., description="Reporting period, e.g. 2024")
    value: float = Field(..., ge=0.0)
    data_source: str = Field(default="")
    quality_grade: DataQualityGrade = Field(default=DataQualityGrade.MEDIUM)
    notes: str = Field(default="")
    provenance_hash: str = Field(default="")

class ValidationFinding(BaseModel):
    """A finding from denominator data validation."""

    finding_id: str = Field(default_factory=lambda: f"vf-{_new_uuid()[:8]}")
    severity: ValidationSeverity = Field(...)
    denominator_type: str = Field(default="")
    period: str = Field(default="")
    message: str = Field(default="")
    recommendation: str = Field(default="")

# =============================================================================
# INPUT / OUTPUT
# =============================================================================

class DenominatorSetupInput(BaseModel):
    """Input data model for DenominatorSetupWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organization identifier")
    sector: SectorClassification = Field(
        default=SectorClassification.OTHER,
        description="Primary sector classification",
    )
    sub_sector: str = Field(default="", description="Sub-sector code or description")
    applicable_frameworks: List[str] = Field(
        default_factory=lambda: ["esrs_e1", "cdp_c6", "gri_305_4"],
        description="Frameworks requiring intensity disclosures",
    )
    reporting_periods: List[str] = Field(
        default_factory=lambda: ["2024", "2025"],
        description="Reporting periods for data collection",
    )
    available_data: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Available denominator data: {denominator_type: {period: value}}",
    )
    custom_denominators: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Custom denominator definitions",
    )
    minimum_quality_grade: DataQualityGrade = Field(
        default=DataQualityGrade.MEDIUM,
        description="Minimum acceptable data quality",
    )
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class DenominatorSetupResult(BaseModel):
    """Complete result from denominator setup workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="denominator_setup")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    sector_profile: Optional[SectorProfile] = Field(default=None)
    denominator_candidates: List[DenominatorCandidate] = Field(default_factory=list)
    selected_denominators: List[DenominatorCandidate] = Field(default_factory=list)
    collected_records: List[DenominatorRecord] = Field(default_factory=list)
    validation_findings: List[ValidationFinding] = Field(default_factory=list)
    readiness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class DenominatorSetupWorkflow:
    """
    4-phase workflow for denominator identification, selection, data collection,
    and validation.

    Identifies the entity's sector and applicable frameworks, recommends
    denominators, collects values across reporting periods, and validates
    data quality and completeness for intensity metric computation.

    Zero-hallucination: denominator recommendations use a deterministic
    sector-to-denominator mapping table, no LLM calls in selection path;
    SHA-256 provenance on every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _sector_profile: Identified sector profile.
        _candidates: Denominator candidates with scores.
        _selected: Selected denominators.
        _records: Collected denominator records.
        _findings: Validation findings.

    Example:
        >>> wf = DenominatorSetupWorkflow()
        >>> inp = DenominatorSetupInput(
        ...     organization_id="org-001",
        ...     sector=SectorClassification.INDUSTRIALS,
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[SetupPhase] = [
        SetupPhase.SECTOR_IDENTIFICATION,
        SetupPhase.DENOMINATOR_SELECTION,
        SetupPhase.DATA_COLLECTION,
        SetupPhase.VALIDATION,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize DenominatorSetupWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._sector_profile: Optional[SectorProfile] = None
        self._candidates: List[DenominatorCandidate] = []
        self._selected: List[DenominatorCandidate] = []
        self._records: List[DenominatorRecord] = []
        self._findings: List[ValidationFinding] = []
        self._readiness_score: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: DenominatorSetupInput) -> DenominatorSetupResult:
        """
        Execute the 4-phase denominator setup workflow.

        Args:
            input_data: Sector, frameworks, reporting periods, and available data.

        Returns:
            DenominatorSetupResult with selected denominators and validation.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting denominator setup %s org=%s sector=%s",
            self.workflow_id, input_data.organization_id, input_data.sector.value,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_sector_identification,
            self._phase_2_denominator_selection,
            self._phase_3_data_collection,
            self._phase_4_validation,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Denominator setup failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = DenominatorSetupResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            sector_profile=self._sector_profile,
            denominator_candidates=self._candidates,
            selected_denominators=self._selected,
            collected_records=self._records,
            validation_findings=self._findings,
            readiness_score=self._readiness_score,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Denominator setup %s completed in %.2fs status=%s selected=%d readiness=%.1f%%",
            self.workflow_id, elapsed, overall_status.value,
            len(self._selected), self._readiness_score,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Sector Identification
    # -------------------------------------------------------------------------

    async def _phase_1_sector_identification(
        self, input_data: DenominatorSetupInput,
    ) -> PhaseResult:
        """Identify sector, sub-sector, and applicable frameworks."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        sector = input_data.sector
        sub_sector = input_data.sub_sector
        frameworks = list(input_data.applicable_frameworks)

        # Determine recommended denominators from sector mapping
        sector_key = sector.value
        recommended = list(SECTOR_DENOMINATOR_MAP.get(sector_key, ["revenue", "fte"]))

        # Determine sector-specific denominator
        sector_specific = ""
        if sector_key in ("energy", "utilities"):
            sector_specific = "energy_generated"
        elif sector_key in ("materials", "consumer_staples"):
            sector_specific = "tonne_product"
        elif sector_key == "transportation":
            sector_specific = "passenger_km"
        elif sector_key == "real_estate":
            sector_specific = "floor_area"
        elif sector_key == "financials":
            sector_specific = "assets_under_management"
        elif sector_key == "healthcare":
            sector_specific = "beds"

        # Validate frameworks
        valid_frameworks = [
            fw for fw in frameworks
            if fw in FRAMEWORK_REQUIRED_DENOMINATORS
        ]
        invalid_frameworks = [
            fw for fw in frameworks
            if fw not in FRAMEWORK_REQUIRED_DENOMINATORS
        ]
        if invalid_frameworks:
            warnings.append(f"Unknown frameworks ignored: {invalid_frameworks}")

        self._sector_profile = SectorProfile(
            sector=sector,
            sub_sector=sub_sector,
            applicable_frameworks=valid_frameworks,
            recommended_denominators=recommended,
            sector_specific_denominator=sector_specific,
            notes=f"Classified via PACK-046 sector mapping v{_MODULE_VERSION}",
        )

        outputs["sector"] = sector.value
        outputs["sub_sector"] = sub_sector
        outputs["applicable_frameworks"] = valid_frameworks
        outputs["recommended_denominators"] = recommended
        outputs["sector_specific_denominator"] = sector_specific

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 SectorIdentification: sector=%s frameworks=%d denominators=%d",
            sector.value, len(valid_frameworks), len(recommended),
        )
        return PhaseResult(
            phase_name="sector_identification", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Denominator Selection
    # -------------------------------------------------------------------------

    async def _phase_2_denominator_selection(
        self, input_data: DenominatorSetupInput,
    ) -> PhaseResult:
        """Rank and select denominators based on framework requirements."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not self._sector_profile:
            raise RuntimeError("Sector profile not available from Phase 1")

        self._candidates = []
        recommended = self._sector_profile.recommended_denominators
        frameworks = self._sector_profile.applicable_frameworks

        # Build candidate list with relevance scoring
        all_denominator_types = set(recommended)
        for fw in frameworks:
            req = FRAMEWORK_REQUIRED_DENOMINATORS.get(fw, [])
            for r in req:
                if r == "sector_specific":
                    sp = self._sector_profile.sector_specific_denominator
                    if sp:
                        all_denominator_types.add(sp)
                else:
                    all_denominator_types.add(r)

        for dtype_str in all_denominator_types:
            try:
                dtype = DenominatorType(dtype_str)
            except ValueError:
                warnings.append(f"Unknown denominator type: {dtype_str}")
                continue

            # Calculate relevance score
            relevance = 0.0
            requiring_frameworks: List[str] = []

            # Score for being in recommended list
            if dtype_str in recommended:
                relevance += 40.0

            # Score for framework requirements
            for fw in frameworks:
                req = FRAMEWORK_REQUIRED_DENOMINATORS.get(fw, [])
                if dtype_str in req:
                    relevance += 15.0
                    requiring_frameworks.append(fw)
                elif "sector_specific" in req and dtype_str == self._sector_profile.sector_specific_denominator:
                    relevance += 15.0
                    requiring_frameworks.append(fw)

            # Score for data availability
            data_available = dtype_str in input_data.available_data
            if data_available:
                relevance += 20.0

            relevance = min(relevance, 100.0)

            # Determine default unit
            unit = self._default_unit_for_type(dtype)

            self._candidates.append(DenominatorCandidate(
                denominator_type=dtype,
                unit=unit,
                relevance_score=round(relevance, 2),
                frameworks_requiring=requiring_frameworks,
                data_available=data_available,
            ))

        # Sort by relevance descending and select top candidates
        self._candidates.sort(key=lambda c: c.relevance_score, reverse=True)

        # Select: all with relevance >= 40, or at least revenue
        self._selected = []
        for candidate in self._candidates:
            if candidate.relevance_score >= 40.0:
                candidate.selected = True
                candidate.selection_reason = (
                    f"Relevance score {candidate.relevance_score:.1f} "
                    f"meets threshold (40.0)"
                )
                self._selected.append(candidate)

        # Ensure at least revenue is selected
        if not self._selected:
            for candidate in self._candidates:
                if candidate.denominator_type == DenominatorType.REVENUE:
                    candidate.selected = True
                    candidate.selection_reason = "Default selection: revenue"
                    self._selected.append(candidate)
                    break
            if not self._selected and self._candidates:
                top = self._candidates[0]
                top.selected = True
                top.selection_reason = "Highest relevance fallback"
                self._selected.append(top)

        if not self._selected:
            warnings.append("No denominators could be selected")

        outputs["candidates_evaluated"] = len(self._candidates)
        outputs["denominators_selected"] = len(self._selected)
        outputs["selected_types"] = [s.denominator_type.value for s in self._selected]
        outputs["relevance_range"] = {
            "min": round(min((c.relevance_score for c in self._candidates), default=0.0), 2),
            "max": round(max((c.relevance_score for c in self._candidates), default=0.0), 2),
        }

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 DenominatorSelection: %d candidates, %d selected",
            len(self._candidates), len(self._selected),
        )
        return PhaseResult(
            phase_name="denominator_selection", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Data Collection
    # -------------------------------------------------------------------------

    async def _phase_3_data_collection(
        self, input_data: DenominatorSetupInput,
    ) -> PhaseResult:
        """Collect denominator values for all selected denominators and periods."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._records = []
        collected_count = 0
        missing_count = 0

        for denom in self._selected:
            dtype_str = denom.denominator_type.value
            period_data = input_data.available_data.get(dtype_str, {})

            for period in input_data.reporting_periods:
                value = period_data.get(period)

                if value is not None and value >= 0:
                    # Determine quality grade based on source
                    quality = DataQualityGrade.HIGH if value > 0 else DataQualityGrade.LOW

                    record_data = {
                        "type": dtype_str, "period": period,
                        "value": value, "unit": denom.unit.value,
                    }

                    self._records.append(DenominatorRecord(
                        denominator_type=denom.denominator_type,
                        unit=denom.unit,
                        period=period,
                        value=value,
                        data_source="available_data",
                        quality_grade=quality,
                        provenance_hash=_compute_hash(record_data),
                    ))
                    collected_count += 1
                else:
                    missing_count += 1
                    warnings.append(
                        f"Missing data for {dtype_str} period {period}"
                    )

        # Handle custom denominators
        for custom in input_data.custom_denominators:
            c_type = custom.get("type", "custom")
            c_unit = custom.get("unit", "custom")
            for period in input_data.reporting_periods:
                c_value = custom.get("values", {}).get(period)
                if c_value is not None:
                    record_data = {
                        "type": c_type, "period": period,
                        "value": c_value, "unit": c_unit,
                    }
                    self._records.append(DenominatorRecord(
                        denominator_type=DenominatorType.CUSTOM,
                        unit=DenominatorUnit.CUSTOM,
                        period=period,
                        value=float(c_value),
                        data_source="custom_input",
                        quality_grade=DataQualityGrade.MEDIUM,
                        notes=f"Custom: {c_type}",
                        provenance_hash=_compute_hash(record_data),
                    ))
                    collected_count += 1

        total_expected = len(self._selected) * len(input_data.reporting_periods)
        coverage_pct = (collected_count / max(total_expected, 1)) * 100.0

        outputs["records_collected"] = collected_count
        outputs["records_missing"] = missing_count
        outputs["total_expected"] = total_expected
        outputs["coverage_pct"] = round(coverage_pct, 2)
        outputs["periods"] = input_data.reporting_periods
        outputs["denominator_types"] = [d.denominator_type.value for d in self._selected]

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 DataCollection: %d collected, %d missing, %.1f%% coverage",
            collected_count, missing_count, coverage_pct,
        )
        return PhaseResult(
            phase_name="data_collection", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Validation
    # -------------------------------------------------------------------------

    async def _phase_4_validation(
        self, input_data: DenominatorSetupInput,
    ) -> PhaseResult:
        """Validate data quality, completeness, and consistency."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._findings = []

        # Check 1: Completeness - all selected denominators have data
        for denom in self._selected:
            dtype = denom.denominator_type.value
            periods_with_data = [
                r.period for r in self._records
                if r.denominator_type == denom.denominator_type
            ]
            missing_periods = [
                p for p in input_data.reporting_periods
                if p not in periods_with_data
            ]
            if missing_periods:
                self._findings.append(ValidationFinding(
                    severity=ValidationSeverity.ERROR,
                    denominator_type=dtype,
                    message=f"Missing data for periods: {missing_periods}",
                    recommendation="Collect denominator data or use estimation",
                ))

        # Check 2: Quality grade threshold
        quality_order = {
            DataQualityGrade.HIGH: 4,
            DataQualityGrade.MEDIUM: 3,
            DataQualityGrade.LOW: 2,
            DataQualityGrade.ESTIMATED: 1,
            DataQualityGrade.UNAVAILABLE: 0,
        }
        min_quality_val = quality_order.get(input_data.minimum_quality_grade, 3)

        for record in self._records:
            record_quality_val = quality_order.get(record.quality_grade, 0)
            if record_quality_val < min_quality_val:
                self._findings.append(ValidationFinding(
                    severity=ValidationSeverity.WARNING,
                    denominator_type=record.denominator_type.value,
                    period=record.period,
                    message=(
                        f"Quality grade {record.quality_grade.value} below "
                        f"minimum {input_data.minimum_quality_grade.value}"
                    ),
                    recommendation="Improve data source or document estimation method",
                ))

        # Check 3: Temporal consistency - no large year-over-year swings (>50%)
        for denom in self._selected:
            dtype_records = sorted(
                [r for r in self._records if r.denominator_type == denom.denominator_type],
                key=lambda r: r.period,
            )
            for i in range(1, len(dtype_records)):
                prev_val = dtype_records[i - 1].value
                curr_val = dtype_records[i].value
                if prev_val > 0:
                    change_pct = abs((curr_val - prev_val) / prev_val) * 100.0
                    if change_pct > 50.0:
                        self._findings.append(ValidationFinding(
                            severity=ValidationSeverity.WARNING,
                            denominator_type=denom.denominator_type.value,
                            period=dtype_records[i].period,
                            message=(
                                f"Year-over-year change of {change_pct:.1f}% "
                                f"exceeds 50% threshold"
                            ),
                            recommendation="Verify denominator data and document reason for change",
                        ))

        # Check 4: Zero or negative values
        for record in self._records:
            if record.value <= 0:
                self._findings.append(ValidationFinding(
                    severity=ValidationSeverity.ERROR,
                    denominator_type=record.denominator_type.value,
                    period=record.period,
                    message=f"Zero or negative value: {record.value}",
                    recommendation="Denominators must be positive for intensity calculation",
                ))

        # Calculate readiness score
        error_count = sum(1 for f in self._findings if f.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for f in self._findings if f.severity == ValidationSeverity.WARNING)
        total_checks = max(len(self._records) * 3, 1)  # 3 checks per record approx
        deductions = (error_count * 15.0) + (warning_count * 5.0)
        self._readiness_score = round(max(100.0 - deductions, 0.0), 2)

        outputs["total_findings"] = len(self._findings)
        outputs["errors"] = error_count
        outputs["warnings"] = warning_count
        outputs["info"] = sum(1 for f in self._findings if f.severity == ValidationSeverity.INFO)
        outputs["readiness_score"] = self._readiness_score
        outputs["ready_for_calculation"] = error_count == 0

        if error_count > 0:
            warnings.append(f"{error_count} validation error(s) must be resolved")

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 Validation: %d findings (%d errors, %d warnings) readiness=%.1f%%",
            len(self._findings), error_count, warning_count, self._readiness_score,
        )
        return PhaseResult(
            phase_name="validation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: DenominatorSetupInput, phase_number: int,
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
        self._sector_profile = None
        self._candidates = []
        self._selected = []
        self._records = []
        self._findings = []
        self._readiness_score = 0.0

    def _default_unit_for_type(self, dtype: DenominatorType) -> DenominatorUnit:
        """Return the default unit for a denominator type."""
        unit_map: Dict[DenominatorType, DenominatorUnit] = {
            DenominatorType.REVENUE: DenominatorUnit.USD_MILLION,
            DenominatorType.FTE: DenominatorUnit.HEADCOUNT,
            DenominatorType.FLOOR_AREA: DenominatorUnit.SQM,
            DenominatorType.PRODUCTION_VOLUME: DenominatorUnit.TONNES,
            DenominatorType.ENERGY_GENERATED: DenominatorUnit.MWH,
            DenominatorType.TONNE_PRODUCT: DenominatorUnit.TONNES,
            DenominatorType.PASSENGER_KM: DenominatorUnit.PASSENGER_KM,
            DenominatorType.TONNE_KM: DenominatorUnit.TONNE_KM,
            DenominatorType.BEDS: DenominatorUnit.UNIT_COUNT,
            DenominatorType.ROOMS: DenominatorUnit.UNIT_COUNT,
            DenominatorType.ASSETS_UNDER_MANAGEMENT: DenominatorUnit.USD_MILLION,
            DenominatorType.CUSTOM: DenominatorUnit.CUSTOM,
        }
        return unit_map.get(dtype, DenominatorUnit.CUSTOM)

    def _compute_provenance(self, result: DenominatorSetupResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{result.organization_id}|{len(result.selected_denominators)}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
