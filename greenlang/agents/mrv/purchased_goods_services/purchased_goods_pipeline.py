"""
PurchasedGoodsPipelineEngine - Purchased Goods & Services Pipeline Orchestrator

This module implements Engine 7 of 7 for AGENT-MRV-014 (Purchased Goods & Services).
It orchestrates the complete 10-stage pipeline from ingestion through export.

Pipeline Stages:
    1. INGEST - Parse and validate procurement items
    2. CLASSIFY - Assign NAICS/NACE/ISIC/UNSPSC codes and material categories
    3. BOUNDARY_CHECK - Filter excluded items (Cat 2-8, intercompany, credits)
    4. SPEND_CALC - Execute spend-based calculation via Engine 2
    5. AVGDATA_CALC - Execute average-data calculation via Engine 3
    6. SUPPLIER_CALC - Execute supplier-specific calculation via Engine 4
    7. AGGREGATE - Combine results via Engine 5 hybrid aggregation
    8. DQI_SCORE - Score data quality (GHG Protocol DQI framework)
    9. COMPLIANCE_CHECK - Validate via Engine 6
    10. EXPORT - Format results (JSON/CSV/Excel)

Example:
    >>> pipeline = PurchasedGoodsPipelineEngine()
    >>> items = [
    ...     {"description": "Steel beams", "spend_usd": 50000, "quantity": 100, "unit": "t"},
    ...     {"description": "IT services", "spend_usd": 30000}
    ... ]
    >>> result = pipeline.run_pipeline(
    ...     items=items,
    ...     method=CalculationMethod.HYBRID,
    ...     frameworks=[ComplianceFramework.GHG_PROTOCOL_SCOPE3]
    ... )
    >>> assert result["status"] == BatchStatus.COMPLETED
    >>> assert result["hybrid_result"].total_emissions_tco2e > Decimal("0")

Author: GL-BackendDeveloper
Agent: AGENT-MRV-014 (Purchased Goods & Services)
Created: 2026-02-25
"""

import threading
import logging
import json
import csv
import io
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone

from greenlang.agents.mrv.purchased_goods_services.models import (
    AGENT_ID,
    VERSION,
    TABLE_PREFIX,
    ZERO,
    ONE,
    ONE_HUNDRED,
    ONE_THOUSAND,
    DECIMAL_PLACES,
    CalculationMethod,
    PipelineStage,
    ExportFormat,
    BatchStatus,
    ComplianceFramework,
    CoverageLevel,
    EEIODatabase,
    PhysicalEFSource,
    ProcurementItem,
    SpendRecord,
    PhysicalRecord,
    SupplierRecord,
    SpendBasedResult,
    AverageDataResult,
    SupplierSpecificResult,
    HybridResult,
    ComplianceCheckResult,
    BatchRequest,
    BatchResult,
    ExportRequest,
    PipelineContext,
    CategoryBoundaryCheck,
)
from greenlang.agents.mrv.purchased_goods_services.config import PurchasedGoodsServicesConfig
from greenlang.agents.mrv.purchased_goods_services.metrics import PurchasedGoodsServicesMetrics
from greenlang.agents.mrv.purchased_goods_services.provenance import PurchasedGoodsProvenanceTracker
from greenlang.agents.mrv.purchased_goods_services.procurement_database import (
    ProcurementDatabaseEngine,
)
from greenlang.agents.mrv.purchased_goods_services.spend_based_calculator import (
    SpendBasedCalculatorEngine,
)
from greenlang.agents.mrv.purchased_goods_services.average_data_calculator import (
    AverageDataCalculatorEngine,
)
from greenlang.agents.mrv.purchased_goods_services.supplier_specific_calculator import (
    SupplierSpecificCalculatorEngine,
)
from greenlang.agents.mrv.purchased_goods_services.hybrid_aggregator import HybridAggregatorEngine
from greenlang.agents.mrv.purchased_goods_services.compliance_checker import (
    ComplianceCheckerEngine,
)

logger = logging.getLogger(__name__)

__all__ = ["PurchasedGoodsPipelineEngine"]


class PurchasedGoodsPipelineEngine:
    """
    Pipeline orchestrator for purchased goods & services emissions.

    This engine coordinates the complete calculation flow from raw procurement
    data through to compliance-checked results and formatted exports.

    Thread-safe singleton pattern ensures consistent state across the application.

    Attributes:
        config: Configuration settings
        metrics: Performance metrics tracker
        provenance_tracker: Data lineage tracker
        db_engine: Procurement database engine (Engine 1)
        spend_calc: Spend-based calculator (Engine 2)
        avgdata_calc: Average-data calculator (Engine 3)
        supplier_calc: Supplier-specific calculator (Engine 4)
        hybrid_agg: Hybrid aggregator (Engine 5)
        compliance_checker: Compliance checker (Engine 6)

    Example:
        >>> pipeline = PurchasedGoodsPipelineEngine()
        >>> result = pipeline.run_pipeline(items, method=CalculationMethod.SPEND_BASED)
        >>> assert result["status"] == BatchStatus.COMPLETED
    """

    _instance: Optional["PurchasedGoodsPipelineEngine"] = None
    _lock = threading.RLock()

    def __new__(cls) -> "PurchasedGoodsPipelineEngine":
        """Thread-safe singleton constructor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        """Initialize pipeline engine (singleton-safe)."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.config = PurchasedGoodsServicesConfig()
            self.metrics = PurchasedGoodsServicesMetrics()
            self.provenance_tracker = PurchasedGoodsProvenanceTracker()

            # Initialize all 6 engines
            self.db_engine = ProcurementDatabaseEngine()
            self.spend_calc = SpendBasedCalculatorEngine()
            self.avgdata_calc = AverageDataCalculatorEngine()
            self.supplier_calc = SupplierSpecificCalculatorEngine()
            self.hybrid_agg = HybridAggregatorEngine()
            self.compliance_checker = ComplianceCheckerEngine()

            self._initialized = True
            logger.info(
                f"{AGENT_ID} v{VERSION} - PurchasedGoodsPipelineEngine initialized"
            )

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def run_pipeline(
        self,
        items: List[Dict[str, Any]],
        method: CalculationMethod = CalculationMethod.HYBRID,
        frameworks: Optional[List[ComplianceFramework]] = None,
        disclosures: Optional[List[str]] = None,
        export_format: ExportFormat = ExportFormat.JSON,
        supplier_records: Optional[List[Dict[str, Any]]] = None,
        eeio_database: EEIODatabase = EEIODatabase.EPA_USEEIO,
        cpi_ratio: Decimal = ONE,
    ) -> Dict[str, Any]:
        """
        Execute the complete 10-stage pipeline.

        Args:
            items: Raw procurement items (dicts or ProcurementItem objects)
            method: Calculation method (SPEND_BASED, AVERAGE_DATA, SUPPLIER_SPECIFIC, HYBRID)
            frameworks: Compliance frameworks to check (optional)
            disclosures: Required disclosures (optional)
            export_format: Output format (JSON, CSV, EXCEL)
            supplier_records: Supplier-specific data (optional)
            eeio_database: EEIO database to use for spend-based
            cpi_ratio: CPI adjustment ratio (current_year / base_year)

        Returns:
            Complete pipeline result with all stage outputs

        Raises:
            ValueError: If input validation fails
            RuntimeError: If critical pipeline stage fails
        """
        start_time = datetime.now(timezone.utc)
        logger.info(
            f"Starting pipeline: {len(items)} items, method={method.value}, "
            f"frameworks={[f.value for f in (frameworks or [])]}"
        )

        # Create pipeline context
        ctx = PipelineContext(
            pipeline_id=self._generate_pipeline_id(),
            calculation_method=method,
            start_time=start_time,
            current_stage=PipelineStage.INGEST,
            total_items=len(items),
            eeio_database=eeio_database,
            cpi_ratio=cpi_ratio,
        )

        try:
            # Stage 1: INGEST
            with self.metrics.timer("pipeline.stage.ingest"):
                ingested_items = self._stage_ingest(items, ctx)
                ctx.current_stage = PipelineStage.CLASSIFY
                logger.info(f"Stage 1 INGEST: {len(ingested_items)} items validated")

            # Stage 2: CLASSIFY
            with self.metrics.timer("pipeline.stage.classify"):
                classified_items = self._stage_classify(ingested_items, ctx)
                ctx.current_stage = PipelineStage.BOUNDARY_CHECK
                logger.info(f"Stage 2 CLASSIFY: {len(classified_items)} items classified")

            # Stage 3: BOUNDARY_CHECK
            with self.metrics.timer("pipeline.stage.boundary_check"):
                in_scope_items, excluded_items = self._stage_boundary_check(
                    classified_items, ctx
                )
                ctx.current_stage = PipelineStage.SPEND_CALC
                ctx.excluded_items = len(excluded_items)
                logger.info(
                    f"Stage 3 BOUNDARY_CHECK: {len(in_scope_items)} in-scope, "
                    f"{len(excluded_items)} excluded"
                )

            # Stages 4-6: Calculation (conditional on method)
            spend_results: List[SpendBasedResult] = []
            avgdata_results: List[AverageDataResult] = []
            supplier_results: List[SupplierSpecificResult] = []

            if method in (CalculationMethod.SPEND_BASED, CalculationMethod.HYBRID):
                # Stage 4: SPEND_CALC
                with self.metrics.timer("pipeline.stage.spend_calc"):
                    spend_results = self._stage_spend_calc(in_scope_items, ctx)
                    ctx.current_stage = PipelineStage.AVGDATA_CALC
                    logger.info(
                        f"Stage 4 SPEND_CALC: {len(spend_results)} results"
                    )

            if method in (CalculationMethod.AVERAGE_DATA, CalculationMethod.HYBRID):
                # Stage 5: AVGDATA_CALC
                with self.metrics.timer("pipeline.stage.avgdata_calc"):
                    avgdata_results = self._stage_avgdata_calc(in_scope_items, ctx)
                    ctx.current_stage = PipelineStage.SUPPLIER_CALC
                    logger.info(
                        f"Stage 5 AVGDATA_CALC: {len(avgdata_results)} results"
                    )

            if method in (
                CalculationMethod.SUPPLIER_SPECIFIC,
                CalculationMethod.HYBRID,
            ):
                # Stage 6: SUPPLIER_CALC
                if supplier_records:
                    with self.metrics.timer("pipeline.stage.supplier_calc"):
                        supplier_results = self._stage_supplier_calc(
                            supplier_records, ctx
                        )
                        ctx.current_stage = PipelineStage.AGGREGATE
                        logger.info(
                            f"Stage 6 SUPPLIER_CALC: {len(supplier_results)} results"
                        )
                else:
                    logger.warning(
                        "Supplier-specific method requested but no supplier_records provided"
                    )

            # Calculate total spend for hybrid aggregation
            total_spend = sum(
                (
                    Decimal(str(item.get("spend_usd", 0)))
                    if isinstance(item, dict)
                    else item.spend_usd
                )
                for item in in_scope_items
            )

            # Stage 7: AGGREGATE
            with self.metrics.timer("pipeline.stage.aggregate"):
                hybrid_result = self._stage_aggregate(
                    spend_results,
                    avgdata_results,
                    supplier_results,
                    in_scope_items,
                    total_spend,
                    ctx,
                )
                ctx.current_stage = PipelineStage.DQI_SCORE
                logger.info(
                    f"Stage 7 AGGREGATE: "
                    f"{hybrid_result.total_emissions_tco2e:.2f} tCO2e total"
                )

            # Stage 8: DQI_SCORE
            with self.metrics.timer("pipeline.stage.dqi_score"):
                dqi_scores = self._stage_dqi_score(hybrid_result, ctx)
                ctx.current_stage = PipelineStage.COMPLIANCE_CHECK
                logger.info(
                    f"Stage 8 DQI_SCORE: Overall score = {dqi_scores['overall_score']:.2f}"
                )

            # Stage 9: COMPLIANCE_CHECK
            compliance_results = []
            if frameworks:
                with self.metrics.timer("pipeline.stage.compliance_check"):
                    compliance_results = self._stage_compliance_check(
                        hybrid_result, disclosures, frameworks, ctx
                    )
                    ctx.current_stage = PipelineStage.EXPORT
                    logger.info(
                        f"Stage 9 COMPLIANCE_CHECK: {len(compliance_results)} frameworks checked"
                    )

            # Stage 10: EXPORT
            with self.metrics.timer("pipeline.stage.export"):
                export_data = self._stage_export(hybrid_result, export_format, ctx)
                logger.info(f"Stage 10 EXPORT: Format = {export_format.value}")

            # Build final result
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            result = {
                "pipeline_id": ctx.pipeline_id,
                "status": BatchStatus.COMPLETED,
                "calculation_method": method.value,
                "total_items": len(items),
                "in_scope_items": len(in_scope_items),
                "excluded_items": len(excluded_items),
                "spend_results_count": len(spend_results),
                "avgdata_results_count": len(avgdata_results),
                "supplier_results_count": len(supplier_results),
                "hybrid_result": hybrid_result,
                "dqi_scores": dqi_scores,
                "compliance_results": compliance_results,
                "export_data": export_data,
                "export_format": export_format.value,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_ms": duration_ms,
                "provenance_hash": self.provenance_tracker.calculate_provenance(
                    {
                        "items": items,
                        "method": method.value,
                        "frameworks": [f.value for f in (frameworks or [])],
                    },
                    hybrid_result.dict(),
                ),
            }

            # Record provenance
            self.provenance_tracker.record_event(
                event_type="pipeline_completed",
                data=result,
                agent_id=AGENT_ID,
            )

            # Record metrics
            self.metrics.record_counter("pipeline.completed", 1)
            self.metrics.record_histogram("pipeline.duration_ms", duration_ms)
            self.metrics.record_histogram("pipeline.total_emissions_tco2e", float(hybrid_result.total_emissions_tco2e))

            logger.info(
                f"Pipeline completed: {duration_ms:.0f}ms, "
                f"{hybrid_result.total_emissions_tco2e:.2f} tCO2e"
            )

            return result

        except Exception as e:
            logger.error(f"Pipeline failed at stage {ctx.current_stage.value}: {e}", exc_info=True)
            self.metrics.record_counter("pipeline.failed", 1)
            raise RuntimeError(f"Pipeline failed at {ctx.current_stage.value}: {e}") from e

    def _stage_ingest(
        self, items: List[Dict[str, Any]], ctx: PipelineContext
    ) -> List[ProcurementItem]:
        """
        Stage 1: INGEST - Parse and validate procurement items.

        Args:
            items: Raw procurement items
            ctx: Pipeline context

        Returns:
            List of validated ProcurementItem objects

        Raises:
            ValueError: If any item fails validation
        """
        ingested: List[ProcurementItem] = []
        errors: List[str] = []

        for idx, item_data in enumerate(items):
            try:
                # Convert dict to ProcurementItem if needed
                if isinstance(item_data, ProcurementItem):
                    item = item_data
                else:
                    # Convert numeric fields to Decimal
                    if "spend_usd" in item_data and item_data["spend_usd"] is not None:
                        item_data["spend_usd"] = Decimal(str(item_data["spend_usd"]))
                    if "quantity" in item_data and item_data["quantity"] is not None:
                        item_data["quantity"] = Decimal(str(item_data["quantity"]))

                    item = ProcurementItem(**item_data)

                # Validate spend > 0 or quantity > 0
                if item.spend_usd <= ZERO and (item.quantity is None or item.quantity <= ZERO):
                    errors.append(
                        f"Item {idx}: Must have either spend_usd > 0 or quantity > 0"
                    )
                    continue

                ingested.append(item)

            except Exception as e:
                errors.append(f"Item {idx}: Validation failed - {e}")

        if errors:
            error_msg = "\n".join(errors[:10])  # Limit to first 10 errors
            if len(errors) > 10:
                error_msg += f"\n... and {len(errors) - 10} more errors"
            raise ValueError(f"Ingestion failed:\n{error_msg}")

        return ingested

    def _stage_classify(
        self, items: List[ProcurementItem], ctx: PipelineContext
    ) -> List[ProcurementItem]:
        """
        Stage 2: CLASSIFY - Assign industry codes and material categories.

        Uses the procurement database engine to lookup or infer codes.

        Args:
            items: Ingested procurement items
            ctx: Pipeline context

        Returns:
            List of classified items with codes assigned
        """
        classified: List[ProcurementItem] = []

        for item in items:
            # If item already has codes, keep them
            if item.naics_code and item.nace_code:
                classified.append(item)
                continue

            # Otherwise, lookup by description
            classification = self.db_engine.classify_item(
                description=item.description,
                material_name=item.material_name,
            )

            # Create updated item with classification
            classified_item = item.copy(
                update={
                    "naics_code": classification.get("naics_code", item.naics_code),
                    "nace_code": classification.get("nace_code", item.nace_code),
                    "isic_code": classification.get("isic_code", item.isic_code),
                    "unspsc_code": classification.get("unspsc_code", item.unspsc_code),
                    "material_category": classification.get(
                        "material_category", item.material_category
                    ),
                }
            )
            classified.append(classified_item)

        return classified

    def _stage_boundary_check(
        self, items: List[ProcurementItem], ctx: PipelineContext
    ) -> Tuple[List[ProcurementItem], List[Dict[str, Any]]]:
        """
        Stage 3: BOUNDARY_CHECK - Filter out excluded items.

        Excluded items include:
        - Items that belong to other Scope 3 categories (Cat 2-8)
        - Intercompany transactions
        - Carbon offsets/credits
        - Items explicitly marked as excluded

        Args:
            items: Classified procurement items
            ctx: Pipeline context

        Returns:
            Tuple of (in-scope items, excluded items with reasons)
        """
        in_scope: List[ProcurementItem] = []
        excluded: List[Dict[str, Any]] = []

        for item in items:
            # Check if explicitly excluded
            if hasattr(item, "is_excluded") and item.is_excluded:
                excluded.append(
                    {
                        "item": item,
                        "reason": "Explicitly marked as excluded",
                        "category": "EXCLUDED",
                    }
                )
                continue

            # Perform boundary check
            boundary_check = self.db_engine.check_category_boundary(
                naics_code=item.naics_code,
                description=item.description,
                supplier_id=item.supplier_id,
                material_category=item.material_category,
            )

            if boundary_check.is_in_scope:
                in_scope.append(item)
            else:
                excluded.append(
                    {
                        "item": item,
                        "reason": boundary_check.exclusion_reason,
                        "category": boundary_check.excluded_category,
                    }
                )

        return in_scope, excluded

    def _stage_spend_calc(
        self, items: List[ProcurementItem], ctx: PipelineContext
    ) -> List[SpendBasedResult]:
        """
        Stage 4: SPEND_CALC - Execute spend-based calculation.

        Args:
            items: In-scope procurement items
            ctx: Pipeline context

        Returns:
            List of spend-based results
        """
        results: List[SpendBasedResult] = []

        for item in items:
            try:
                spend_record = SpendRecord(
                    item_id=item.item_id,
                    description=item.description,
                    naics_code=item.naics_code,
                    nace_code=item.nace_code,
                    isic_code=item.isic_code,
                    spend_usd=item.spend_usd,
                    supplier_country=item.supplier_country,
                    purchase_date=item.purchase_date,
                )

                result = self.spend_calc.calculate_spend_based(
                    record=spend_record,
                    database=ctx.eeio_database,
                    cpi_ratio=ctx.cpi_ratio,
                )
                results.append(result)

            except Exception as e:
                logger.warning(
                    f"Spend-based calc failed for item {item.item_id}: {e}"
                )
                # Continue with other items

        return results

    def _stage_avgdata_calc(
        self, items: List[ProcurementItem], ctx: PipelineContext
    ) -> List[AverageDataResult]:
        """
        Stage 5: AVGDATA_CALC - Execute average-data calculation.

        Args:
            items: In-scope procurement items
            ctx: Pipeline context

        Returns:
            List of average-data results
        """
        results: List[AverageDataResult] = []

        for item in items:
            # Only calculate if we have quantity and material info
            if item.quantity is None or item.quantity <= ZERO:
                continue
            if not item.material_name and not item.material_category:
                continue

            try:
                physical_record = PhysicalRecord(
                    item_id=item.item_id,
                    material_name=item.material_name or "",
                    material_category=item.material_category or "",
                    quantity=item.quantity,
                    unit=item.unit or "kg",
                    supplier_country=item.supplier_country,
                    purchase_date=item.purchase_date,
                )

                result = self.avgdata_calc.calculate_average_data(
                    record=physical_record,
                    preferred_source=PhysicalEFSource.ECOINVENT,
                )
                results.append(result)

            except Exception as e:
                logger.warning(
                    f"Average-data calc failed for item {item.item_id}: {e}"
                )
                # Continue with other items

        return results

    def _stage_supplier_calc(
        self, records: List[Dict[str, Any]], ctx: PipelineContext
    ) -> List[SupplierSpecificResult]:
        """
        Stage 6: SUPPLIER_CALC - Execute supplier-specific calculation.

        Args:
            records: Supplier-specific records (PCFs, LCAs, etc.)
            ctx: Pipeline context

        Returns:
            List of supplier-specific results
        """
        results: List[SupplierSpecificResult] = []

        for record_data in records:
            try:
                # Convert dict to SupplierRecord
                if "emissions_tco2e" in record_data:
                    record_data["emissions_tco2e"] = Decimal(
                        str(record_data["emissions_tco2e"])
                    )
                if "quantity" in record_data:
                    record_data["quantity"] = Decimal(str(record_data["quantity"]))

                supplier_record = SupplierRecord(**record_data)

                result = self.supplier_calc.calculate_supplier_specific(
                    record=supplier_record
                )
                results.append(result)

            except Exception as e:
                logger.warning(f"Supplier-specific calc failed for record: {e}")
                # Continue with other records

        return results

    def _stage_aggregate(
        self,
        spend_results: List[SpendBasedResult],
        avgdata_results: List[AverageDataResult],
        supplier_results: List[SupplierSpecificResult],
        items: List[ProcurementItem],
        total_spend: Decimal,
        ctx: PipelineContext,
    ) -> HybridResult:
        """
        Stage 7: AGGREGATE - Combine all calculation results.

        Uses Engine 5 (Hybrid Aggregator) to apply hierarchical preference:
        1. Supplier-specific (highest quality)
        2. Average-data (medium quality)
        3. Spend-based (lowest quality, fallback)

        Args:
            spend_results: Spend-based results
            avgdata_results: Average-data results
            supplier_results: Supplier-specific results
            items: Original procurement items
            total_spend: Total spend across all items
            ctx: Pipeline context

        Returns:
            Aggregated hybrid result
        """
        return self.hybrid_agg.aggregate_hybrid(
            spend_results=spend_results,
            avgdata_results=avgdata_results,
            supplier_results=supplier_results,
            total_spend=total_spend,
        )

    def _stage_dqi_score(
        self, hybrid_result: HybridResult, ctx: PipelineContext
    ) -> Dict[str, Any]:
        """
        Stage 8: DQI_SCORE - Calculate data quality indicators.

        Implements GHG Protocol Corporate Value Chain (Scope 3) Accounting
        and Reporting Standard DQI framework.

        DQI dimensions:
        - Technological representativeness (TnR)
        - Temporal representativeness (TiR)
        - Geographical representativeness (GR)
        - Completeness (C)
        - Reliability (R)

        Args:
            hybrid_result: Aggregated hybrid result
            ctx: Pipeline context

        Returns:
            DQI scores dictionary with overall score and dimension scores
        """
        # Extract coverage from hybrid result
        supplier_coverage = hybrid_result.supplier_specific_coverage_pct
        avgdata_coverage = hybrid_result.average_data_coverage_pct
        spend_coverage = hybrid_result.spend_based_coverage_pct

        # Score each dimension (1-5 scale, 1 = best)
        # Based on GHG Protocol Scope 3 standard guidance

        # Technological representativeness
        # Primary data (supplier-specific) = 1, secondary (average-data) = 3, EEIO = 4
        tnr_score = (
            (supplier_coverage * Decimal("1.0"))
            + (avgdata_coverage * Decimal("3.0"))
            + (spend_coverage * Decimal("4.0"))
        ) / ONE_HUNDRED

        # Temporal representativeness
        # For now, assume recent data (score = 2)
        # In production, calculate from purchase_date vs current date
        tir_score = Decimal("2.0")

        # Geographical representativeness
        # For now, assume appropriate geography (score = 2)
        # In production, match supplier_country to EF geography
        gr_score = Decimal("2.0")

        # Completeness
        # All items covered = 1, >90% = 2, >75% = 3, >50% = 4, <50% = 5
        total_coverage = supplier_coverage + avgdata_coverage + spend_coverage
        if total_coverage >= Decimal("99.9"):
            c_score = Decimal("1.0")
        elif total_coverage >= Decimal("90.0"):
            c_score = Decimal("2.0")
        elif total_coverage >= Decimal("75.0"):
            c_score = Decimal("3.0")
        elif total_coverage >= Decimal("50.0"):
            c_score = Decimal("4.0")
        else:
            c_score = Decimal("5.0")

        # Reliability
        # Verified data = 1, audited = 2, unverified = 3, estimates = 4
        # For now, assume unverified (score = 3)
        r_score = Decimal("3.0")

        # Overall DQI score (geometric mean)
        overall_score = (tnr_score * tir_score * gr_score * c_score * r_score) ** (
            Decimal("0.2")
        )

        dqi_scores = {
            "overall_score": float(overall_score.quantize(Decimal("0.01"))),
            "technological_representativeness": float(tnr_score.quantize(Decimal("0.01"))),
            "temporal_representativeness": float(tir_score.quantize(Decimal("0.01"))),
            "geographical_representativeness": float(gr_score.quantize(Decimal("0.01"))),
            "completeness": float(c_score.quantize(Decimal("0.01"))),
            "reliability": float(r_score.quantize(Decimal("0.01"))),
            "interpretation": self._interpret_dqi_score(overall_score),
        }

        return dqi_scores

    def _interpret_dqi_score(self, score: Decimal) -> str:
        """Interpret DQI score into quality level."""
        if score <= Decimal("2.0"):
            return "Excellent - High quality, fit for public disclosure"
        elif score <= Decimal("3.0"):
            return "Good - Acceptable for reporting, some uncertainty"
        elif score <= Decimal("4.0"):
            return "Fair - Significant uncertainty, use with caution"
        else:
            return "Poor - High uncertainty, not recommended for disclosure"

    def _stage_compliance_check(
        self,
        hybrid_result: HybridResult,
        disclosures: Optional[List[str]],
        frameworks: List[ComplianceFramework],
        ctx: PipelineContext,
    ) -> List[ComplianceCheckResult]:
        """
        Stage 9: COMPLIANCE_CHECK - Validate against regulatory frameworks.

        Args:
            hybrid_result: Aggregated result to validate
            disclosures: Required disclosures (optional)
            frameworks: Frameworks to check against
            ctx: Pipeline context

        Returns:
            List of compliance check results
        """
        results: List[ComplianceCheckResult] = []

        for framework in frameworks:
            try:
                result = self.compliance_checker.check_compliance(
                    hybrid_result=hybrid_result,
                    framework=framework,
                    required_disclosures=disclosures,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Compliance check failed for {framework.value}: {e}")
                # Continue with other frameworks

        return results

    def _stage_export(
        self,
        hybrid_result: HybridResult,
        export_format: ExportFormat,
        ctx: PipelineContext,
    ) -> Dict[str, Any]:
        """
        Stage 10: EXPORT - Format results for output.

        Args:
            hybrid_result: Result to export
            export_format: Desired output format
            ctx: Pipeline context

        Returns:
            Export data dictionary with formatted content
        """
        if export_format == ExportFormat.JSON:
            content = self.export_json(hybrid_result)
        elif export_format == ExportFormat.CSV:
            content = self.export_csv(hybrid_result)
        elif export_format == ExportFormat.EXCEL:
            content = self.export_excel(hybrid_result)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        return {
            "format": export_format.value,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def run_batch(self, batch: BatchRequest) -> BatchResult:
        """
        Run a batch of procurement items through the pipeline.

        Wrapper around run_pipeline() that returns a BatchResult object.

        Args:
            batch: BatchRequest with items and configuration

        Returns:
            BatchResult with status and results
        """
        try:
            result = self.run_pipeline(
                items=batch.items,
                method=batch.calculation_method,
                frameworks=batch.compliance_frameworks,
                disclosures=batch.required_disclosures,
                export_format=batch.export_format,
                supplier_records=batch.supplier_records,
                eeio_database=batch.eeio_database,
                cpi_ratio=batch.cpi_ratio,
            )

            return BatchResult(
                batch_id=batch.batch_id,
                status=result["status"],
                total_items=result["total_items"],
                in_scope_items=result["in_scope_items"],
                excluded_items=result["excluded_items"],
                total_emissions_tco2e=result["hybrid_result"].total_emissions_tco2e,
                spend_based_emissions_tco2e=result[
                    "hybrid_result"
                ].spend_based_emissions_tco2e,
                average_data_emissions_tco2e=result[
                    "hybrid_result"
                ].average_data_emissions_tco2e,
                supplier_specific_emissions_tco2e=result[
                    "hybrid_result"
                ].supplier_specific_emissions_tco2e,
                coverage_level=self._determine_coverage_level(result["hybrid_result"]),
                dqi_scores=result["dqi_scores"],
                compliance_results=result["compliance_results"],
                export_data=result["export_data"],
                start_time=result["start_time"],
                end_time=result["end_time"],
                duration_ms=result["duration_ms"],
                provenance_hash=result["provenance_hash"],
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            return BatchResult(
                batch_id=batch.batch_id,
                status=BatchStatus.FAILED,
                total_items=len(batch.items),
                in_scope_items=0,
                excluded_items=0,
                total_emissions_tco2e=ZERO,
                spend_based_emissions_tco2e=ZERO,
                average_data_emissions_tco2e=ZERO,
                supplier_specific_emissions_tco2e=ZERO,
                coverage_level=CoverageLevel.NONE,
                dqi_scores={},
                compliance_results=[],
                export_data={},
                start_time=datetime.now(timezone.utc).isoformat(),
                end_time=datetime.now(timezone.utc).isoformat(),
                duration_ms=0.0,
                error_message=str(e),
                provenance_hash="",
            )

    def run_multi_period(
        self, periods: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run pipeline across multiple time periods.

        Useful for year-over-year comparisons and trend analysis.

        Args:
            periods: List of period configs, each with items and period metadata

        Returns:
            List of period results
        """
        results: List[Dict[str, Any]] = []

        for period_config in periods:
            period_label = period_config.get("period_label", "Unknown")
            logger.info(f"Processing period: {period_label}")

            try:
                result = self.run_pipeline(
                    items=period_config["items"],
                    method=period_config.get(
                        "method", CalculationMethod.HYBRID
                    ),
                    frameworks=period_config.get("frameworks"),
                    disclosures=period_config.get("disclosures"),
                    export_format=period_config.get(
                        "export_format", ExportFormat.JSON
                    ),
                    supplier_records=period_config.get("supplier_records"),
                    eeio_database=period_config.get(
                        "eeio_database", EEIODatabase.EPA_USEEIO
                    ),
                    cpi_ratio=Decimal(str(period_config.get("cpi_ratio", "1.0"))),
                )

                result["period_label"] = period_label
                results.append(result)

            except Exception as e:
                logger.error(f"Period {period_label} failed: {e}")
                results.append(
                    {
                        "period_label": period_label,
                        "status": BatchStatus.FAILED,
                        "error": str(e),
                    }
                )

        return results

    def export_json(self, result: HybridResult) -> str:
        """
        Export result as JSON string.

        Args:
            result: HybridResult to export

        Returns:
            JSON string
        """
        return result.json(indent=2)

    def export_csv(self, result: HybridResult) -> str:
        """
        Export result as CSV string.

        Creates a flat CSV with summary rows for each calculation method.

        Args:
            result: HybridResult to export

        Returns:
            CSV string
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "Calculation Method",
                "Total Emissions (tCO2e)",
                "Coverage (%)",
                "Items Count",
            ]
        )

        # Spend-based row
        writer.writerow(
            [
                "Spend-Based (EEIO)",
                f"{result.spend_based_emissions_tco2e:.{DECIMAL_PLACES}f}",
                f"{result.spend_based_coverage_pct:.2f}",
                result.spend_based_items_count,
            ]
        )

        # Average-data row
        writer.writerow(
            [
                "Average-Data (Physical EF)",
                f"{result.average_data_emissions_tco2e:.{DECIMAL_PLACES}f}",
                f"{result.average_data_coverage_pct:.2f}",
                result.average_data_items_count,
            ]
        )

        # Supplier-specific row
        writer.writerow(
            [
                "Supplier-Specific (Primary)",
                f"{result.supplier_specific_emissions_tco2e:.{DECIMAL_PLACES}f}",
                f"{result.supplier_specific_coverage_pct:.2f}",
                result.supplier_specific_items_count,
            ]
        )

        # Total row
        writer.writerow(
            [
                "TOTAL (Hybrid)",
                f"{result.total_emissions_tco2e:.{DECIMAL_PLACES}f}",
                "100.00",
                result.total_items_count,
            ]
        )

        return output.getvalue()

    def export_excel(self, result: HybridResult) -> bytes:
        """
        Export result as Excel file (bytes).

        Note: This is a placeholder. In production, use openpyxl or xlsxwriter.

        Args:
            result: HybridResult to export

        Returns:
            Excel file as bytes
        """
        # For now, return CSV as bytes
        # In production, implement proper Excel export with formatting
        csv_content = self.export_csv(result)
        return csv_content.encode("utf-8")

    def get_pipeline_status(self, ctx: PipelineContext) -> Dict[str, Any]:
        """
        Get current pipeline execution status.

        Args:
            ctx: Pipeline context

        Returns:
            Status dictionary
        """
        return {
            "pipeline_id": ctx.pipeline_id,
            "current_stage": ctx.current_stage.value,
            "calculation_method": ctx.calculation_method.value,
            "total_items": ctx.total_items,
            "excluded_items": ctx.excluded_items,
            "start_time": ctx.start_time.isoformat(),
            "elapsed_ms": (datetime.now(timezone.utc) - ctx.start_time).total_seconds()
            * 1000,
        }

    def validate_request(self, items: List[Dict[str, Any]]) -> List[str]:
        """
        Validate pipeline request before execution.

        Args:
            items: Procurement items to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        if not items:
            errors.append("Items list cannot be empty")
            return errors

        if len(items) > 10000:
            errors.append(f"Too many items ({len(items)}). Maximum is 10,000 per batch.")

        # Sample first 100 items for validation
        sample_size = min(100, len(items))
        for idx, item in enumerate(items[:sample_size]):
            if not isinstance(item, dict):
                errors.append(f"Item {idx} is not a dictionary")
                continue

            if "description" not in item:
                errors.append(f"Item {idx} missing required field: description")

            if "spend_usd" not in item and "quantity" not in item:
                errors.append(
                    f"Item {idx} must have either spend_usd or quantity"
                )

            # Validate numeric fields
            for field in ["spend_usd", "quantity"]:
                if field in item and item[field] is not None:
                    try:
                        value = Decimal(str(item[field]))
                        if value < ZERO:
                            errors.append(f"Item {idx} {field} cannot be negative")
                    except (InvalidOperation, ValueError):
                        errors.append(f"Item {idx} {field} is not a valid number")

        return errors

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on pipeline and all engines.

        Returns:
            Health status dictionary
        """
        health = {
            "pipeline": "healthy",
            "engines": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Check all engines
        engines = {
            "db_engine": self.db_engine,
            "spend_calc": self.spend_calc,
            "avgdata_calc": self.avgdata_calc,
            "supplier_calc": self.supplier_calc,
            "hybrid_agg": self.hybrid_agg,
            "compliance_checker": self.compliance_checker,
        }

        for name, engine in engines.items():
            try:
                engine_health = engine.health_check()
                health["engines"][name] = engine_health
            except Exception as e:
                health["engines"][name] = {"status": "unhealthy", "error": str(e)}
                health["pipeline"] = "degraded"

        return health

    def _generate_pipeline_id(self) -> str:
        """Generate unique pipeline ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{TABLE_PREFIX}pipeline_{timestamp}"

    def _determine_coverage_level(self, result: HybridResult) -> CoverageLevel:
        """
        Determine overall coverage level from hybrid result.

        Args:
            result: HybridResult with coverage percentages

        Returns:
            CoverageLevel enum value
        """
        # Prioritize supplier-specific coverage
        supplier_pct = result.supplier_specific_coverage_pct
        avgdata_pct = result.average_data_coverage_pct

        if supplier_pct >= Decimal("75.0"):
            return CoverageLevel.SUPPLIER_SPECIFIC_DOMINANT
        elif avgdata_pct >= Decimal("50.0"):
            return CoverageLevel.AVERAGE_DATA_DOMINANT
        elif supplier_pct + avgdata_pct >= Decimal("25.0"):
            return CoverageLevel.HYBRID_BALANCED
        else:
            return CoverageLevel.SPEND_BASED_ONLY
