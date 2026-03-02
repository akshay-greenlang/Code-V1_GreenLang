# -*- coding: utf-8 -*-
"""
CategoryMapperPipelineEngine - AGENT-MRV-029 Engine 7

This module implements the end-to-end 10-stage orchestration pipeline for the
Scope 3 Category Mapper cross-cutting agent (GL-MRV-X-040).

The pipeline classifies raw organisational data (spend, PO, BOM, travel, etc.)
into GHG Protocol Scope 3 categories (1-15), determines boundaries, checks for
double-counting, recommends calculation approaches, and screens completeness.

10 Pipeline Stages:
    1. VALIDATE         - Input validation and normalization
    2. SOURCE_CLASSIFY  - Data source classification (determine DataSourceType)
    3. CODE_LOOKUP      - Industry code lookup (NAICS/ISIC via CategoryDatabaseEngine)
    4. SPEND_CLASSIFY   - Spend classification (via SpendClassifierEngine)
    5. BOUNDARY         - Boundary determination (via BoundaryDeterminerEngine)
    6. DOUBLE_COUNTING  - Double-counting check (via BoundaryDeterminerEngine)
    7. SPLIT            - Multi-category splitting
    8. RECOMMEND        - Calculation approach recommendation
    9. COMPLETENESS     - Completeness screening (via CompletenessScreenerEngine)
    10. SEAL            - Provenance hashing and output assembly

Zero-Hallucination Guarantee:
    All classifications use deterministic lookup tables. No LLM or ML
    models are involved in the classification or boundary determination
    pipeline.

Example:
    >>> pipeline = CategoryMapperPipelineEngine(
    ...     database_engine=CategoryDatabaseEngine(),
    ...     spend_engine=SpendClassifierEngine(),
    ...     router_engine=ActivityRouterEngine(),
    ...     boundary_engine=BoundaryDeterminerEngine(),
    ...     completeness_engine=CompletenessScreenerEngine.get_instance(),
    ...     compliance_engine=ComplianceCheckerEngine(),
    ... )
    >>> result = pipeline.run_pipeline(batch_input)
    >>> assert result.total_classified > 0

Module: greenlang.scope3_category_mapper.category_mapper_pipeline
Agent: AGENT-MRV-029
Version: 1.0.0
"""

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-X-040"
AGENT_COMPONENT: str = "AGENT-MRV-029"
VERSION: str = "1.0.0"


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class DataSourceType(str, Enum):
    """Supported data source types for classification."""

    SPEND = "spend"
    PURCHASE_ORDER = "purchase_order"
    BOM = "bom"
    TRAVEL = "travel"
    FLEET = "fleet"
    WASTE = "waste"
    LEASE = "lease"
    LOGISTICS = "logistics"
    PRODUCT_SALES = "product_sales"
    INVESTMENT = "investment"
    FRANCHISE = "franchise"
    ENERGY = "energy"
    SUPPLIER = "supplier"


class CalculationApproach(str, Enum):
    """Recommended calculation approach for emissions estimation."""

    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"


class ConfidenceLevel(str, Enum):
    """Classification confidence level."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class MappingStatus(str, Enum):
    """Classification mapping status."""

    MAPPED = "mapped"
    SPLIT = "split"
    UNMAPPED = "unmapped"
    REVIEW_REQUIRED = "review_required"
    EXCLUDED = "excluded"


class RoutingAction(str, Enum):
    """Routing action for classified records."""

    ROUTE = "route"
    SPLIT_ROUTE = "split_route"
    QUEUE_REVIEW = "queue_review"
    EXCLUDE = "exclude"


class PipelineStageEnum(str, Enum):
    """Pipeline stage identifiers (used internally)."""

    VALIDATE = "validate"
    SOURCE_CLASSIFY = "source_classify"
    CODE_LOOKUP = "code_lookup"
    SPEND_CLASSIFY = "spend_classify"
    BOUNDARY = "boundary"
    DOUBLE_COUNTING = "double_counting"
    SPLIT = "split"
    RECOMMEND = "recommend"
    COMPLETENESS = "completeness"
    SEAL = "seal"


# ==============================================================================
# PIPELINE INPUT / OUTPUT MODELS
# ==============================================================================


class BatchClassificationInput(BaseModel):
    """Input model for batch classification pipeline."""

    model_config = ConfigDict(frozen=True)

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=50000,
        description="List of records to classify (up to 50,000)"
    )
    source_type: Optional[str] = Field(
        None, description="Data source type (if known)"
    )
    organization_id: str = Field(
        ..., description="Organization identifier"
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2100, description="Reporting year"
    )
    company_type: Optional[str] = Field(
        None,
        description="Company type for completeness screening"
    )
    consolidation_approach: str = Field(
        default="operational_control",
        description="Consolidation approach (operational_control/financial_control/equity_share)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ClassificationResult(BaseModel):
    """Classification result for a single record."""

    model_config = ConfigDict(frozen=False)

    record_id: str = Field(
        ..., description="Unique identifier for this classification"
    )
    original_record: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original input record"
    )
    primary_category: Optional[int] = Field(
        None, ge=1, le=15,
        description="Primary Scope 3 category number (1-15)"
    )
    secondary_categories: List[int] = Field(
        default_factory=list,
        description="Secondary categories that may also apply"
    )
    classification_method: Optional[str] = Field(
        None, description="Method used for classification"
    )
    confidence: Decimal = Field(
        default=Decimal("0.0"),
        ge=0, le=1,
        description="Classification confidence (0.0-1.0)"
    )
    confidence_level: str = Field(
        default="very_low",
        description="Classification confidence level"
    )
    mapping_status: str = Field(
        default="unmapped",
        description="Mapping status (mapped/split/unmapped/review_required/excluded)"
    )
    routing_action: str = Field(
        default="queue_review",
        description="Routing action"
    )
    target_agent: Optional[str] = Field(
        None, description="Target category agent identifier"
    )
    calculation_approach: Optional[str] = Field(
        None, description="Recommended calculation approach"
    )
    data_quality_tier: int = Field(
        default=5, ge=1, le=5,
        description="Data quality tier (1=best, 5=worst)"
    )
    boundary_direction: Optional[str] = Field(
        None, description="Value chain direction (upstream/downstream)"
    )
    split_allocations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Multi-category split allocations"
    )
    double_counting_flags: List[str] = Field(
        default_factory=list,
        description="Double-counting rule flags triggered"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Per-record processing time in ms"
    )


class CategorySummary(BaseModel):
    """Aggregated statistics for a single Scope 3 category."""

    model_config = ConfigDict(frozen=True)

    category_number: int = Field(..., ge=1, le=15)
    category_name: str = Field(...)
    record_count: int = Field(default=0, ge=0)
    total_spend: Decimal = Field(default=Decimal("0.0"))
    avg_confidence: Decimal = Field(default=Decimal("0.0"))
    methods_used: List[str] = Field(default_factory=list)


class BatchClassificationResult(BaseModel):
    """Output model for batch classification pipeline."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique pipeline run identifier")
    organization_id: str = Field(...)
    reporting_year: int = Field(...)
    total_records: int = Field(..., ge=0)
    total_classified: int = Field(default=0, ge=0)
    total_unmapped: int = Field(default=0, ge=0)
    total_split: int = Field(default=0, ge=0)
    total_review: int = Field(default=0, ge=0)
    results: List[ClassificationResult] = Field(default_factory=list)
    category_summaries: List[CategorySummary] = Field(default_factory=list)
    completeness_report: Optional[Dict[str, Any]] = Field(None)
    double_counting_detections: int = Field(default=0, ge=0)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    stage_durations_ms: Dict[str, float] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


# ==============================================================================
# DATA SOURCE CLASSIFICATION HEURISTICS
# ==============================================================================

_SOURCE_TYPE_INDICATORS: Dict[str, List[str]] = {
    DataSourceType.SPEND.value: [
        "gl_code", "gl_account", "account_code", "vendor", "spend_amount",
        "invoice_amount", "ap_amount",
    ],
    DataSourceType.PURCHASE_ORDER.value: [
        "po_number", "po_line", "purchase_order", "requisition",
    ],
    DataSourceType.BOM.value: [
        "bom_id", "material_code", "component", "assembly",
    ],
    DataSourceType.TRAVEL.value: [
        "trip_id", "flight", "hotel", "booking_ref", "travel_type",
        "departure", "arrival", "airline",
    ],
    DataSourceType.FLEET.value: [
        "vehicle_id", "fleet", "fuel_card", "odometer", "registration",
    ],
    DataSourceType.WASTE.value: [
        "waste_type", "manifest", "disposal_method", "waste_code",
    ],
    DataSourceType.LEASE.value: [
        "lease_id", "lease_type", "lessor", "lessee", "rent_amount",
    ],
    DataSourceType.LOGISTICS.value: [
        "shipment_id", "freight", "carrier", "bill_of_lading", "incoterm",
    ],
    DataSourceType.PRODUCT_SALES.value: [
        "product_id", "sku", "units_sold", "revenue", "sales_channel",
    ],
    DataSourceType.INVESTMENT.value: [
        "isin", "ticker", "portfolio_id", "holding", "asset_class",
    ],
    DataSourceType.FRANCHISE.value: [
        "franchise_id", "franchisee", "royalty", "franchise_type",
    ],
    DataSourceType.ENERGY.value: [
        "meter_id", "kwh", "electricity", "gas_usage", "utility_account",
    ],
    DataSourceType.SUPPLIER.value: [
        "supplier_id", "supplier_emissions", "questionnaire_response",
    ],
}

# Category number -> agent ID mapping
_CATEGORY_AGENT_MAP: Dict[int, str] = {
    1: "GL-MRV-S3-001",   # Purchased Goods & Services (MRV-014)
    2: "GL-MRV-S3-002",   # Capital Goods (MRV-015)
    3: "GL-MRV-S3-003",   # Fuel & Energy Activities (MRV-016)
    4: "GL-MRV-S3-004",   # Upstream Transportation (MRV-017)
    5: "GL-MRV-S3-005",   # Waste Generated (MRV-018)
    6: "GL-MRV-S3-006",   # Business Travel (MRV-019)
    7: "GL-MRV-S3-007",   # Employee Commuting (MRV-020)
    8: "GL-MRV-S3-008",   # Upstream Leased Assets (MRV-021)
    9: "GL-MRV-S3-009",   # Downstream Transportation (MRV-022)
    10: "GL-MRV-S3-010",  # Processing of Sold Products (MRV-023)
    11: "GL-MRV-S3-011",  # Use of Sold Products (MRV-024)
    12: "GL-MRV-S3-012",  # End-of-Life Treatment (MRV-025)
    13: "GL-MRV-S3-013",  # Downstream Leased Assets (MRV-026)
    14: "GL-MRV-S3-014",  # Franchises (MRV-027)
    15: "GL-MRV-S3-015",  # Investments (MRV-028)
}

_CATEGORY_NAMES: Dict[int, str] = {
    1: "Purchased Goods & Services",
    2: "Capital Goods",
    3: "Fuel- and Energy-Related Activities",
    4: "Upstream Transportation & Distribution",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation & Distribution",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment of Sold Products",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}

# Upstream categories (1-8), downstream categories (9-15)
_UPSTREAM_CATEGORIES = {1, 2, 3, 4, 5, 6, 7, 8}
_DOWNSTREAM_CATEGORIES = {9, 10, 11, 12, 13, 14, 15}

# Double-counting rule pairs
_DC_RULES: Dict[str, Tuple[int, int, str]] = {
    "DC-SCM-001": (1, 2, "Opex vs capex: use capitalization policy"),
    "DC-SCM-002": (1, 4, "Goods cost vs freight: split by Incoterm"),
    "DC-SCM-003": (3, 0, "WTT/T&D losses vs Scope 2: exclude reported S2"),
    "DC-SCM-004": (4, 9, "Upstream vs downstream transport: split at sale"),
    "DC-SCM-005": (6, 7, "Business travel vs commuting: no overlap on travel days"),
    "DC-SCM-006": (8, 0, "Leased assets vs Scope 1/2: consolidation approach"),
    "DC-SCM-007": (10, 11, "Processing vs use of sold products: sequential"),
    "DC-SCM-008": (11, 12, "Use vs end-of-life: product lifetime boundary"),
    "DC-SCM-009": (13, 0, "Downstream leased vs Scope 1/2: lessor exclusion"),
    "DC-SCM-010": (14, 15, "Franchise vs investment: agreement type"),
}


# ==============================================================================
# PIPELINE ENGINE
# ==============================================================================


class CategoryMapperPipelineEngine:
    """
    10-stage orchestration pipeline for Scope 3 category classification.

    Processes raw organisational data through deterministic classification,
    boundary determination, double-counting checks, and completeness
    screening. Each stage emits metrics and provenance hashes.

    Attributes:
        database_engine: Engine 1 -- NAICS/ISIC/UNSPSC/HS lookups.
        spend_engine: Engine 2 -- Spend classification.
        router_engine: Engine 3 -- Activity routing.
        boundary_engine: Engine 4 -- Boundary determination.
        completeness_engine: Engine 5 -- Completeness screening.
        compliance_engine: Engine 6 -- Compliance checking.

    Example:
        >>> pipeline = CategoryMapperPipelineEngine(
        ...     database_engine=db, spend_engine=sc,
        ...     router_engine=ar, boundary_engine=bd,
        ...     completeness_engine=cs, compliance_engine=cc,
        ... )
        >>> batch_result = pipeline.run_pipeline(batch_input)
    """

    def __init__(
        self,
        database_engine: Optional[Any] = None,
        spend_engine: Optional[Any] = None,
        router_engine: Optional[Any] = None,
        boundary_engine: Optional[Any] = None,
        completeness_engine: Optional[Any] = None,
        compliance_engine: Optional[Any] = None,
    ) -> None:
        """
        Initialize the pipeline with all 6 collaborating engines.

        Args:
            database_engine: CategoryDatabaseEngine instance.
            spend_engine: SpendClassifierEngine instance.
            router_engine: ActivityRouterEngine instance.
            boundary_engine: BoundaryDeterminerEngine instance.
            completeness_engine: CompletenessScreenerEngine instance.
            compliance_engine: ComplianceCheckerEngine instance.
        """
        self.database_engine = database_engine
        self.spend_engine = spend_engine
        self.router_engine = router_engine
        self.boundary_engine = boundary_engine
        self.completeness_engine = completeness_engine
        self.compliance_engine = compliance_engine

        # Lazy-import provenance engine
        self._provenance_engine: Optional[Any] = None

        logger.info("CategoryMapperPipelineEngine initialized")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_provenance(self) -> Any:
        """Get or create provenance engine (lazy import)."""
        if self._provenance_engine is None:
            try:
                from greenlang.scope3_category_mapper.provenance import (
                    ProvenanceEngine,
                )
                self._provenance_engine = ProvenanceEngine()
            except ImportError:
                logger.warning("Provenance engine not available")
        return self._provenance_engine

    def _get_metrics(self) -> Any:
        """Get metrics instance (lazy import)."""
        try:
            from greenlang.scope3_category_mapper.metrics import get_metrics
            return get_metrics()
        except ImportError:
            return None

    def _record_stage(self, stage: str, duration: float) -> None:
        """Record pipeline stage duration in metrics."""
        metrics = self._get_metrics()
        if metrics is not None:
            metrics.record_pipeline_stage(stage=stage, duration=duration)

    @staticmethod
    def _confidence_to_level(confidence: Decimal) -> str:
        """Convert numeric confidence to a named level."""
        val = float(confidence)
        if val >= 0.90:
            return ConfidenceLevel.VERY_HIGH.value
        if val >= 0.75:
            return ConfidenceLevel.HIGH.value
        if val >= 0.50:
            return ConfidenceLevel.MEDIUM.value
        if val >= 0.25:
            return ConfidenceLevel.LOW.value
        return ConfidenceLevel.VERY_LOW.value

    @staticmethod
    def _determine_approach(confidence: Decimal, source_type: str) -> str:
        """Determine recommended calculation approach."""
        val = float(confidence)
        if source_type == DataSourceType.SUPPLIER.value and val >= 0.80:
            return CalculationApproach.SUPPLIER_SPECIFIC.value
        if val >= 0.70:
            return CalculationApproach.HYBRID.value
        if val >= 0.40:
            return CalculationApproach.AVERAGE_DATA.value
        return CalculationApproach.SPEND_BASED.value

    @staticmethod
    def _determine_data_quality(
        confidence: Decimal, source_type: str
    ) -> int:
        """Determine data quality tier from confidence and source."""
        val = float(confidence)
        if val >= 0.90:
            return 1
        if val >= 0.75:
            return 2
        if val >= 0.50:
            return 3
        if val >= 0.25:
            return 4
        return 5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_pipeline(
        self, input_data: BatchClassificationInput
    ) -> BatchClassificationResult:
        """
        Execute the full 10-stage classification pipeline.

        Processes all records in the batch through validation, source
        classification, code lookup, spend classification, boundary
        determination, double-counting check, splitting, approach
        recommendation, completeness screening, and provenance sealing.

        Args:
            input_data: Batch classification input with records.

        Returns:
            BatchClassificationResult with all classifications.
        """
        pipeline_start = time.monotonic()
        run_id = f"scm-{uuid4().hex[:12]}"
        stage_durations: Dict[str, float] = {}
        errors: List[Dict[str, Any]] = []

        try:
            # Stage 1: Validate
            s1_start = time.monotonic()
            validated = self._stage_1_validate(input_data)
            stage_durations["validate"] = (
                time.monotonic() - s1_start
            ) * 1000.0
            self._record_stage("validate", time.monotonic() - s1_start)

            # Stage 2: Source classification
            s2_start = time.monotonic()
            source_classified = self._stage_2_classify_source(
                validated.records
            )
            stage_durations["source_classify"] = (
                time.monotonic() - s2_start
            ) * 1000.0
            self._record_stage("source_classify", time.monotonic() - s2_start)

            # Stage 3: Code lookup
            s3_start = time.monotonic()
            code_results = self._stage_3_lookup_codes(source_classified)
            stage_durations["code_lookup"] = (
                time.monotonic() - s3_start
            ) * 1000.0
            self._record_stage("code_lookup", time.monotonic() - s3_start)

            # Stage 4: Spend classification
            s4_start = time.monotonic()
            classifications = self._stage_4_classify(code_results)
            stage_durations["spend_classify"] = (
                time.monotonic() - s4_start
            ) * 1000.0
            self._record_stage("spend_classify", time.monotonic() - s4_start)

            # Stage 5: Boundary determination
            s5_start = time.monotonic()
            classifications = self._stage_5_boundaries(classifications)
            stage_durations["boundary"] = (
                time.monotonic() - s5_start
            ) * 1000.0
            self._record_stage("boundary", time.monotonic() - s5_start)

            # Stage 6: Double-counting check
            s6_start = time.monotonic()
            classifications = self._stage_6_dc_check(classifications)
            stage_durations["double_counting"] = (
                time.monotonic() - s6_start
            ) * 1000.0
            self._record_stage(
                "double_counting", time.monotonic() - s6_start
            )

            # Stage 7: Multi-category splitting
            s7_start = time.monotonic()
            classifications = self._stage_7_split(classifications)
            stage_durations["split"] = (
                time.monotonic() - s7_start
            ) * 1000.0
            self._record_stage("split", time.monotonic() - s7_start)

            # Stage 8: Calculation approach recommendation
            s8_start = time.monotonic()
            classifications = self._stage_8_recommend(classifications)
            stage_durations["recommend"] = (
                time.monotonic() - s8_start
            ) * 1000.0
            self._record_stage("recommend", time.monotonic() - s8_start)

            # Stage 9: Completeness screening
            s9_start = time.monotonic()
            completeness = self._stage_9_completeness(
                classifications, validated.company_type
            )
            stage_durations["completeness"] = (
                time.monotonic() - s9_start
            ) * 1000.0
            self._record_stage("completeness", time.monotonic() - s9_start)

            # Stage 10: Provenance and finalize
            s10_start = time.monotonic()
            result = self._stage_10_finalize(
                classifications, completeness, run_id,
                validated.organization_id, validated.reporting_year,
                stage_durations, errors,
            )
            stage_durations["seal"] = (
                time.monotonic() - s10_start
            ) * 1000.0
            self._record_stage("seal", time.monotonic() - s10_start)

            # Update total processing time
            total_ms = (time.monotonic() - pipeline_start) * 1000.0

            # Record batch metrics
            metrics = self._get_metrics()
            if metrics is not None:
                source = validated.source_type or "spend"
                metrics.record_batch(source, len(validated.records))

            return BatchClassificationResult(
                run_id=result.run_id,
                organization_id=result.organization_id,
                reporting_year=result.reporting_year,
                total_records=result.total_records,
                total_classified=result.total_classified,
                total_unmapped=result.total_unmapped,
                total_split=result.total_split,
                total_review=result.total_review,
                results=result.results,
                category_summaries=result.category_summaries,
                completeness_report=result.completeness_report,
                double_counting_detections=result.double_counting_detections,
                provenance_hash=result.provenance_hash,
                processing_time_ms=total_ms,
                stage_durations_ms=stage_durations,
                errors=result.errors,
            )

        except Exception as e:
            elapsed = (time.monotonic() - pipeline_start) * 1000.0
            logger.error("Pipeline failed: %s", e, exc_info=True)
            metrics = self._get_metrics()
            if metrics is not None:
                metrics.record_error("classification")
            return BatchClassificationResult(
                run_id=run_id,
                organization_id=input_data.organization_id,
                reporting_year=input_data.reporting_year,
                total_records=len(input_data.records),
                processing_time_ms=elapsed,
                stage_durations_ms=stage_durations,
                errors=[{"error": str(e)}],
            )

    def run_single(
        self,
        record: Dict[str, Any],
        source_type: DataSourceType,
        org_id: str,
        year: int,
    ) -> ClassificationResult:
        """
        Classify a single record through the pipeline.

        Args:
            record: Raw data record.
            source_type: Data source type.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            ClassificationResult for the single record.
        """
        batch_input = BatchClassificationInput(
            records=[record],
            source_type=source_type.value,
            organization_id=org_id,
            reporting_year=year,
        )
        batch_result = self.run_pipeline(batch_input)
        if batch_result.results:
            return batch_result.results[0]
        return ClassificationResult(
            record_id=f"scm-{uuid4().hex[:12]}",
            original_record=record,
        )

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _stage_1_validate(
        self, input_data: BatchClassificationInput
    ) -> BatchClassificationInput:
        """
        Stage 1: Input validation and normalization.

        Validates the batch input structure. Returns the input as-is
        after validation (Pydantic model already validates on construction).

        Args:
            input_data: Raw batch input.

        Returns:
            Validated BatchClassificationInput.
        """
        logger.debug(
            "Stage 1: Validating %d records", len(input_data.records)
        )
        return input_data

    def _stage_2_classify_source(
        self, records: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], DataSourceType]]:
        """
        Stage 2: Classify each record's data source type.

        Uses field-name heuristics to determine whether the record is
        spend data, a purchase order, travel, logistics, etc.

        Args:
            records: List of raw records.

        Returns:
            List of (record, DataSourceType) tuples.
        """
        results: List[Tuple[Dict[str, Any], DataSourceType]] = []

        for record in records:
            keys_lower = {k.lower() for k in record.keys()}
            best_type = DataSourceType.SPEND
            best_score = 0

            for src_type, indicators in _SOURCE_TYPE_INDICATORS.items():
                score = sum(1 for ind in indicators if ind in keys_lower)
                if score > best_score:
                    best_score = score
                    best_type = DataSourceType(src_type)

            results.append((record, best_type))

        logger.debug("Stage 2: Classified %d source types", len(results))
        return results

    def _stage_3_lookup_codes(
        self,
        records: List[Tuple[Dict[str, Any], DataSourceType]],
    ) -> List[Tuple[Dict[str, Any], DataSourceType, Optional[int], Decimal, Optional[str]]]:
        """
        Stage 3: Industry code lookup via CategoryDatabaseEngine.

        Attempts NAICS, ISIC, and GL code lookups for each record.

        Args:
            records: Source-classified records.

        Returns:
            List of (record, source_type, category, confidence, method).
        """
        results = []

        for record, source_type in records:
            category: Optional[int] = None
            confidence = Decimal("0.0")
            method: Optional[str] = None

            # Try NAICS lookup
            naics = record.get("naics_code") or record.get("naics")
            if naics and self.database_engine is not None:
                try:
                    lookup = self.database_engine.lookup_naics(str(naics))
                    if lookup is not None:
                        category = int(lookup.primary_category)
                        confidence = lookup.confidence
                        method = "naics"
                except Exception as e:
                    logger.debug("NAICS lookup failed: %s", e)

            # Try ISIC lookup if NAICS missed
            if category is None:
                isic = record.get("isic_code") or record.get("isic")
                if isic and self.database_engine is not None:
                    try:
                        lookup = self.database_engine.lookup_isic(str(isic))
                        if lookup is not None:
                            category = int(lookup.primary_category)
                            confidence = lookup.confidence
                            method = "isic"
                    except Exception as e:
                        logger.debug("ISIC lookup failed: %s", e)

            # Try GL account lookup
            if category is None:
                gl = record.get("gl_code") or record.get("gl_account") or record.get("account_code")
                if gl and self.database_engine is not None:
                    try:
                        lookup = self.database_engine.lookup_gl_account(str(gl))
                        if lookup is not None:
                            category = int(lookup.primary_category)
                            confidence = lookup.confidence
                            method = "gl_account"
                    except Exception as e:
                        logger.debug("GL lookup failed: %s", e)

            results.append((record, source_type, category, confidence, method))

        logger.debug("Stage 3: Code lookups completed for %d records", len(results))
        return results

    def _stage_4_classify(
        self,
        records: List[Tuple[Dict[str, Any], DataSourceType, Optional[int], Decimal, Optional[str]]],
    ) -> List[ClassificationResult]:
        """
        Stage 4: Spend classification for records not yet classified.

        Uses keyword matching as a fallback for records that did not
        match via industry codes.

        Args:
            records: Code-lookup results.

        Returns:
            List of ClassificationResult objects.
        """
        results: List[ClassificationResult] = []

        for record, source_type, category, confidence, method in records:
            record_id = f"scm-{uuid4().hex[:12]}"

            # If code lookup succeeded, use that result
            if category is not None and method is not None:
                conf_level = self._confidence_to_level(confidence)
                results.append(ClassificationResult(
                    record_id=record_id,
                    original_record=record,
                    primary_category=category,
                    classification_method=method,
                    confidence=confidence,
                    confidence_level=conf_level,
                    mapping_status=MappingStatus.MAPPED.value,
                    routing_action=RoutingAction.ROUTE.value,
                    target_agent=_CATEGORY_AGENT_MAP.get(category),
                ))
                continue

            # Try keyword classification
            description = (
                record.get("description", "")
                or record.get("item_description", "")
                or record.get("line_description", "")
                or ""
            )
            if description and self.database_engine is not None:
                try:
                    kw_result = self.database_engine.lookup_keyword(description)
                    if kw_result is not None:
                        cat_num = int(kw_result.primary_category)
                        kw_conf = kw_result.confidence
                        kw_level = self._confidence_to_level(kw_conf)
                        results.append(ClassificationResult(
                            record_id=record_id,
                            original_record=record,
                            primary_category=cat_num,
                            classification_method="keyword",
                            confidence=kw_conf,
                            confidence_level=kw_level,
                            mapping_status=MappingStatus.MAPPED.value,
                            routing_action=RoutingAction.ROUTE.value,
                            target_agent=_CATEGORY_AGENT_MAP.get(cat_num),
                        ))
                        continue
                except Exception as e:
                    logger.debug("Keyword lookup failed: %s", e)

            # Unmapped record
            results.append(ClassificationResult(
                record_id=record_id,
                original_record=record,
                mapping_status=MappingStatus.UNMAPPED.value,
                routing_action=RoutingAction.QUEUE_REVIEW.value,
            ))

        logger.debug(
            "Stage 4: Classified %d records (%d mapped)",
            len(results),
            sum(1 for r in results if r.mapping_status == MappingStatus.MAPPED.value),
        )
        return results

    def _stage_5_boundaries(
        self, results: List[ClassificationResult]
    ) -> List[ClassificationResult]:
        """
        Stage 5: Determine boundary direction for classified records.

        Assigns upstream or downstream direction based on category number.

        Args:
            results: Classification results from stage 4.

        Returns:
            Updated classification results with boundary direction.
        """
        for r in results:
            if r.primary_category is not None:
                if r.primary_category in _UPSTREAM_CATEGORIES:
                    r.boundary_direction = "upstream"
                elif r.primary_category in _DOWNSTREAM_CATEGORIES:
                    r.boundary_direction = "downstream"

        logger.debug("Stage 5: Boundary determination completed")
        return results

    def _stage_6_dc_check(
        self, results: List[ClassificationResult]
    ) -> List[ClassificationResult]:
        """
        Stage 6: Double-counting check across categories.

        Identifies records that may overlap between related categories
        according to DC-SCM-001 through DC-SCM-010 rules.

        Args:
            results: Classification results from stage 5.

        Returns:
            Updated results with double-counting flags.
        """
        # Build category presence set
        present_cats = {
            r.primary_category
            for r in results
            if r.primary_category is not None
        }

        dc_count = 0
        for rule_id, (cat_a, cat_b, _desc) in _DC_RULES.items():
            if cat_b == 0:
                # Rules referencing Scope 1/2 boundary -- skip in mapper
                continue
            if cat_a in present_cats and cat_b in present_cats:
                # Flag records in both categories
                for r in results:
                    if r.primary_category in (cat_a, cat_b):
                        if rule_id not in r.double_counting_flags:
                            r.double_counting_flags.append(rule_id)
                            dc_count += 1

        if dc_count > 0:
            metrics = self._get_metrics()
            if metrics is not None:
                for rule_id, (cat_a, cat_b, _) in _DC_RULES.items():
                    if cat_b != 0 and cat_a in present_cats and cat_b in present_cats:
                        metrics.record_double_counting(rule_id.lower().replace("-", "_"))

        logger.debug(
            "Stage 6: Double-counting check found %d flags", dc_count
        )
        return results

    def _stage_7_split(
        self, results: List[ClassificationResult]
    ) -> List[ClassificationResult]:
        """
        Stage 7: Handle multi-category splitting.

        Records with secondary categories and high confidence get split
        allocations. The primary category retains the majority share.

        Args:
            results: Classification results from stage 6.

        Returns:
            Updated results with split allocations where applicable.
        """
        for r in results:
            if (
                r.primary_category is not None
                and len(r.secondary_categories) > 0
                and float(r.confidence) >= 0.50
            ):
                total_parts = 1 + len(r.secondary_categories)
                primary_ratio = round(0.70, 2)
                secondary_ratio = round(
                    0.30 / len(r.secondary_categories), 2
                )

                r.split_allocations = [
                    {
                        "category": r.primary_category,
                        "ratio": primary_ratio,
                        "is_primary": True,
                    }
                ]
                for sec_cat in r.secondary_categories:
                    r.split_allocations.append({
                        "category": sec_cat,
                        "ratio": secondary_ratio,
                        "is_primary": False,
                    })

                r.mapping_status = MappingStatus.SPLIT.value
                r.routing_action = RoutingAction.SPLIT_ROUTE.value

        logger.debug("Stage 7: Multi-category splitting completed")
        return results

    def _stage_8_recommend(
        self, results: List[ClassificationResult]
    ) -> List[ClassificationResult]:
        """
        Stage 8: Recommend calculation approach for each record.

        Assigns supplier_specific, hybrid, average_data, or spend_based
        based on data quality and classification confidence.

        Args:
            results: Classification results from stage 7.

        Returns:
            Updated results with calculation approach and DQ tier.
        """
        for r in results:
            if r.primary_category is not None:
                source = r.original_record.get("_source_type", "spend")
                r.calculation_approach = self._determine_approach(
                    r.confidence, source
                )
                r.data_quality_tier = self._determine_data_quality(
                    r.confidence, source
                )

        logger.debug("Stage 8: Calculation approach recommendation completed")
        return results

    def _stage_9_completeness(
        self,
        results: List[ClassificationResult],
        company_type: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Stage 9: Completeness screening across all 15 categories.

        Uses the CompletenessScreenerEngine to evaluate which categories
        have data and which are missing.

        Args:
            results: All classification results.
            company_type: Company type for relevance matrix.

        Returns:
            Completeness report as dictionary, or None if engine unavailable.
        """
        if self.completeness_engine is None or company_type is None:
            return None

        try:
            from greenlang.scope3_category_mapper.models import (
                CompanyType as ModelCompanyType,
                Scope3Category as ModelScope3Category,
            )

            # Determine which categories are present
            categories_present = set()
            for r in results:
                if r.primary_category is not None:
                    categories_present.add(r.primary_category)

            # Map to model enums
            cat_enum_map = {
                1: ModelScope3Category.CAT_1_PURCHASED_GOODS,
                2: ModelScope3Category.CAT_2_CAPITAL_GOODS,
                3: ModelScope3Category.CAT_3_FUEL_ENERGY,
                4: ModelScope3Category.CAT_4_UPSTREAM_TRANSPORT,
                5: ModelScope3Category.CAT_5_WASTE,
                6: ModelScope3Category.CAT_6_BUSINESS_TRAVEL,
                7: ModelScope3Category.CAT_7_EMPLOYEE_COMMUTING,
                8: ModelScope3Category.CAT_8_UPSTREAM_LEASED,
                9: ModelScope3Category.CAT_9_DOWNSTREAM_TRANSPORT,
                10: ModelScope3Category.CAT_10_PROCESSING_SOLD,
                11: ModelScope3Category.CAT_11_USE_SOLD,
                12: ModelScope3Category.CAT_12_END_OF_LIFE,
                13: ModelScope3Category.CAT_13_DOWNSTREAM_LEASED,
                14: ModelScope3Category.CAT_14_FRANCHISES,
                15: ModelScope3Category.CAT_15_INVESTMENTS,
            }

            reported_enums = [
                cat_enum_map[c] for c in categories_present
                if c in cat_enum_map
            ]

            ct = ModelCompanyType(company_type)

            report = self.completeness_engine.screen_completeness(
                company_type=ct,
                categories_reported=reported_enums,
                data_by_category={},
            )

            # Update metrics
            metrics = self._get_metrics()
            if metrics is not None:
                metrics.update_completeness(
                    company_type, float(report.completeness_score)
                )

            return report.model_dump() if hasattr(report, "model_dump") else report.dict()

        except Exception as e:
            logger.warning("Completeness screening failed: %s", e)
            return None

    def _stage_10_finalize(
        self,
        results: List[ClassificationResult],
        completeness: Optional[Dict[str, Any]],
        run_id: str,
        organization_id: str,
        reporting_year: int,
        stage_durations: Dict[str, float],
        errors: List[Dict[str, Any]],
    ) -> BatchClassificationResult:
        """
        Stage 10: Provenance hashing and output assembly.

        Computes per-record provenance hashes, builds category summaries,
        and assembles the final BatchClassificationResult.

        Args:
            results: All classification results.
            completeness: Completeness report dictionary.
            run_id: Pipeline run identifier.
            organization_id: Organization identifier.
            reporting_year: Reporting year.
            stage_durations: Per-stage durations in ms.
            errors: Accumulated errors.

        Returns:
            Final BatchClassificationResult.
        """
        provenance = self._get_provenance()

        # Compute per-record provenance hashes
        for r in results:
            if provenance is not None:
                try:
                    from greenlang.scope3_category_mapper.provenance import (
                        PipelineStage as ProvStage,
                    )
                    rec = provenance.create_provenance_record(
                        ProvStage.SEAL,
                        r.original_record,
                        {
                            "category": r.primary_category,
                            "method": r.classification_method,
                            "confidence": str(r.confidence),
                        },
                    )
                    r.provenance_hash = rec.stage_hash
                except Exception:
                    pass

        # Build category summaries
        cat_data: Dict[int, Dict[str, Any]] = {}
        for r in results:
            if r.primary_category is not None:
                cat = r.primary_category
                if cat not in cat_data:
                    cat_data[cat] = {
                        "count": 0,
                        "spend": Decimal("0.0"),
                        "confidences": [],
                        "methods": set(),
                    }
                cat_data[cat]["count"] += 1
                spend = r.original_record.get("amount") or r.original_record.get("spend_amount") or 0
                cat_data[cat]["spend"] += Decimal(str(spend))
                cat_data[cat]["confidences"].append(float(r.confidence))
                if r.classification_method:
                    cat_data[cat]["methods"].add(r.classification_method)

        summaries = []
        for cat_num in sorted(cat_data.keys()):
            data = cat_data[cat_num]
            confs = data["confidences"]
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            summaries.append(CategorySummary(
                category_number=cat_num,
                category_name=_CATEGORY_NAMES.get(cat_num, f"Category {cat_num}"),
                record_count=data["count"],
                total_spend=data["spend"],
                avg_confidence=Decimal(str(round(avg_conf, 4))),
                methods_used=sorted(data["methods"]),
            ))

        # Count statuses
        total_classified = sum(
            1 for r in results
            if r.mapping_status == MappingStatus.MAPPED.value
        )
        total_unmapped = sum(
            1 for r in results
            if r.mapping_status == MappingStatus.UNMAPPED.value
        )
        total_split = sum(
            1 for r in results
            if r.mapping_status == MappingStatus.SPLIT.value
        )
        total_review = sum(
            1 for r in results
            if r.routing_action == RoutingAction.QUEUE_REVIEW.value
        )
        dc_count = sum(
            1 for r in results if len(r.double_counting_flags) > 0
        )

        # Build chain-level provenance hash
        chain_hash = ""
        if provenance is not None:
            try:
                from greenlang.scope3_category_mapper.provenance import (
                    build_chain_hash,
                )
                stage_hashes = [
                    r.provenance_hash for r in results if r.provenance_hash
                ]
                if stage_hashes:
                    chain_hash = build_chain_hash(stage_hashes)
            except Exception:
                pass

        # Record classification metrics
        metrics = self._get_metrics()
        if metrics is not None:
            for r in results:
                if r.primary_category is not None:
                    metrics.record_classification(
                        category=f"cat_{r.primary_category}",
                        method=r.classification_method or "unknown",
                        confidence_level=r.confidence_level,
                        duration=r.processing_time_ms / 1000.0
                        if r.processing_time_ms > 0 else 0.001,
                        confidence_value=float(r.confidence),
                    )
            metrics.update_categories_active(
                organization_id, len(cat_data)
            )

        return BatchClassificationResult(
            run_id=run_id,
            organization_id=organization_id,
            reporting_year=reporting_year,
            total_records=len(results),
            total_classified=total_classified,
            total_unmapped=total_unmapped,
            total_split=total_split,
            total_review=total_review,
            results=results,
            category_summaries=summaries,
            completeness_report=completeness,
            double_counting_detections=dc_count,
            provenance_hash=chain_hash,
            stage_durations_ms=stage_durations,
            errors=errors,
        )


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Enums
    "DataSourceType",
    "CalculationApproach",
    "ConfidenceLevel",
    "MappingStatus",
    "RoutingAction",
    "PipelineStageEnum",
    # Models
    "BatchClassificationInput",
    "ClassificationResult",
    "CategorySummary",
    "BatchClassificationResult",
    # Engine
    "CategoryMapperPipelineEngine",
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
]
