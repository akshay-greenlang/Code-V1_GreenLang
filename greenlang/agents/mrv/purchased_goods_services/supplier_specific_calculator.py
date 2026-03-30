# -*- coding: utf-8 -*-
"""
SupplierSpecificCalculatorEngine - Engine 4 of 7: Purchased Goods & Services
Agent (AGENT-MRV-014)

Implements the supplier-specific calculation method for GHG Protocol Scope 3
Category 1 emissions using primary data from suppliers.  This engine provides
the highest-accuracy emission calculations (+/- 10-30%) by consuming verified
supplier data from six source types:

    1. Environmental Product Declarations (EPDs) per ISO 14025 / EN 15804
    2. Product Carbon Footprints (PCFs) per ISO 14067
    3. CDP Supply Chain programme Scope 1+2 data with revenue allocation
    4. EcoVadis sustainability platform scores
    5. PACT Network (WBCSD) product-level data exchange
    6. Direct disclosure / measurement from suppliers

Calculation Methods:
    * Product-Level:   Emissions = Quantity x Product_EF_kgco2e_per_unit
    * Facility-Level:  Emissions = Supplier_Total_tCO2e x Allocation_Factor
    * CDP Integration: Emissions = (Scope1 + Scope2) x (CustomerRev / TotalRev)
    * EPD Integration: Emissions = Quantity x (GWP_Total / DeclaredUnit)

Five allocation methods are supported for facility-level data:
    - revenue_based   : customer revenue / total revenue
    - mass_based       : mass purchased / total mass output
    - economic         : economic value purchased / total economic output
    - physical_unit    : units purchased / total unit output
    - equal            : 1 / number_of_customers

System boundary verification ensures only cradle-to-gate data is used
directly; cradle-to-grave data is flagged for manual review unless the
use-phase and end-of-life components can be removed.

Data Quality Indicator (DQI) scoring follows GHG Protocol Scope 3 Standard
Chapter 7, with supplier-specific data receiving the highest quality scores
(temporal 1-2, geographical 1-2, technological 1, completeness 1-2,
reliability 1-2) resulting in composite DQI 1.0-2.0.

Supplier engagement is scored on a 5-level maturity model:
    Level 1 (Unengaged):  No supplier-specific data available
    Level 2 (Initial):    Self-reported unverified data
    Level 3 (Developing): Third-party assessed (EcoVadis/CDP)
    Level 4 (Established): Verified PCF or EPD with cradle-to-gate boundary
    Level 5 (Leading):    Product-level EPD, 3rd-party verified, annual update

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal with configurable precision
    - No LLM calls in the numeric calculation path
    - Every step is recorded in the calculation trace
    - SHA-256 provenance hash for every result
    - Deterministic allocation factor arithmetic

Thread Safety:
    - Singleton pattern with threading.RLock
    - All mutable state protected by locks
    - Stateless calculation methods are inherently thread-safe

Example:
    >>> from greenlang.agents.mrv.purchased_goods_services.supplier_specific_calculator import (
    ...     SupplierSpecificCalculatorEngine,
    ... )
    >>> from greenlang.agents.mrv.purchased_goods_services.models import (
    ...     SupplierRecord, ProcurementItem, SupplierDataSource,
    ...     AllocationMethod,
    ... )
    >>> from decimal import Decimal
    >>> engine = SupplierSpecificCalculatorEngine()
    >>> result = engine.calculate_product_level(
    ...     quantity=Decimal("1000"),
    ...     product_ef=Decimal("2.33"),
    ...     item_id="ITEM-001",
    ...     supplier_id="SUP-001",
    ...     data_source=SupplierDataSource.EPD,
    ... )
    >>> assert result.emissions_kgco2e == Decimal("2330.00000000")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-014 Purchased Goods & Services (GL-MRV-S3-001)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

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
    SupplierDataSource,
    AllocationMethod,
    DQIDimension,
    DQIScore,
    EmissionGas,
    ProcurementType,
    DQI_SCORE_VALUES,
    UNCERTAINTY_RANGES,
    PEDIGREE_UNCERTAINTY_FACTORS,
    ProcurementItem,
    SupplierRecord,
    SupplierSpecificResult,
    SupplierEF,
    DQIAssessment,
    DQI_QUALITY_TIERS,
)
from greenlang.agents.mrv.purchased_goods_services.config import PurchasedGoodsServicesConfig
from greenlang.agents.mrv.purchased_goods_services.metrics import PurchasedGoodsServicesMetrics
from greenlang.schemas import utcnow
from greenlang.agents.mrv.purchased_goods_services.provenance import (
    PurchasedGoodsProvenanceTracker,
    ProvenanceStage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "SupplierSpecificCalculatorEngine",
]

#: Decimal precision quantizer derived from config.
_DEFAULT_PRECISION = Decimal("0." + "0" * DECIMAL_PLACES)

#: Conversion factor: tonnes to kilograms.
_TONNES_TO_KG = Decimal("1000")

#: Conversion factor: kilograms to tonnes.
_KG_TO_TONNES = Decimal("0.001")

#: Maximum EPD age in years before it is considered expired.
_EPD_MAX_AGE_YEARS: int = 5

#: Valid system boundary values for cradle-to-gate acceptance.
_VALID_CTG_BOUNDARIES: frozenset = frozenset({
    "cradle_to_gate",
    "cradle-to-gate",
    "c2g",
    "ctg",
})

#: System boundary values that require adjustment before use.
_ADJUSTABLE_BOUNDARIES: frozenset = frozenset({
    "cradle_to_grave",
    "cradle-to-grave",
    "c2gr",
    "ctgr",
})

#: CDP score quality mapping: letter grade to numeric weight (0-1).
_CDP_SCORE_QUALITY: Dict[str, Decimal] = {
    "A":  Decimal("1.00"),
    "A-": Decimal("0.95"),
    "B":  Decimal("0.85"),
    "B-": Decimal("0.80"),
    "C":  Decimal("0.70"),
    "C-": Decimal("0.65"),
    "D":  Decimal("0.50"),
    "D-": Decimal("0.45"),
    "F":  Decimal("0.20"),
}

#: Supplier engagement level descriptions.
_ENGAGEMENT_LEVEL_NAMES: Dict[int, str] = {
    1: "Unengaged",
    2: "Initial",
    3: "Developing",
    4: "Established",
    5: "Leading",
}

#: Minimum allocation factor threshold to prevent near-zero allocations.
_MIN_ALLOCATION_FACTOR = Decimal("0.00000001")

#: Maximum number of records in a single batch calculation.
_MAX_BATCH_SIZE: int = 50_000

#: Number of equal-allocation customers when no count is provided.
_DEFAULT_EQUAL_CUSTOMERS: int = 10

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _quantize(value: Decimal, precision: Optional[Decimal] = None) -> Decimal:
    """Quantize a Decimal value to the configured precision.

    Args:
        value: The Decimal value to quantize.
        precision: Optional override for the quantization precision.
            Defaults to ``_DEFAULT_PRECISION``.

    Returns:
        The quantized Decimal value.

    Raises:
        InvalidOperation: If the value cannot be quantized.
    """
    prec = precision if precision is not None else _DEFAULT_PRECISION
    try:
        return value.quantize(prec, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        logger.warning(
            "Failed to quantize value=%s with precision=%s; "
            "returning original value",
            value, prec,
        )
        return value

def _compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash of arbitrary data.

    Serializes data to a canonical JSON string (sorted keys, no indent)
    and returns the hexadecimal digest. Handles Decimal, datetime, and
    Enum types via a custom encoder.

    Args:
        data: Any JSON-serializable data structure.

    Returns:
        64-character hexadecimal SHA-256 digest string.
    """
    class _Encoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, Decimal):
                return str(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            if hasattr(obj, "value"):
                return obj.value
            return super().default(obj)

    canonical = json.dumps(data, cls=_Encoder, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return when denominator is zero.

    Returns:
        The quotient or *default*.
    """
    if denominator == ZERO:
        return default
    return _quantize(numerator / denominator)

def _get_quality_tier(composite_score: Decimal) -> str:
    """Map a composite DQI score to a quality tier label.

    Args:
        composite_score: Arithmetic mean of the five DQI dimension scores.

    Returns:
        Human-readable quality tier label.
    """
    for tier_name, (lo, hi) in DQI_QUALITY_TIERS.items():
        if lo <= composite_score < hi:
            return tier_name
    return "Very Poor"

def _current_reporting_year() -> int:
    """Return the current calendar year for reporting-year checks."""
    return utcnow().year

# ===========================================================================
# SupplierSpecificCalculatorEngine
# ===========================================================================

class SupplierSpecificCalculatorEngine:
    """Engine 4: Supplier-specific calculation engine for Scope 3 Category 1.

    Implements the highest-accuracy supplier-specific method using primary
    data from suppliers including EPDs, PCFs, CDP Supply Chain data,
    EcoVadis scores, PACT Network feeds, and direct disclosures.

    The engine supports two primary calculation paths:

    1. **Product-level**: ``Emissions = Quantity * Product_EF_kgco2e_per_unit``
       Used when the supplier provides product-level emission factors from
       an EPD, PCF, or direct measurement.

    2. **Facility-level with allocation**: ``Emissions = Supplier_Total *
       Allocation_Factor``
       Used when only facility-level (total) emission data is available and
       the reporting company must allocate a share using one of five
       allocation methods.

    Thread Safety:
        Singleton with ``threading.RLock`` for double-checked locking.
        Calculation methods are stateless and inherently thread-safe.
        Only the singleton constructor and ``reset`` mutate class state.

    Attributes:
        _config: Configuration singleton for decimal precision, DQI weights,
            coverage targets, and reporting year.
        _metrics: Prometheus metrics singleton for recording calculation
            operations.
        _provenance: SHA-256 provenance tracker singleton for audit chains.
        _precision: Decimal quantization precision derived from config.
        _calculation_count: Running count of calculations performed.

    Example:
        >>> engine = SupplierSpecificCalculatorEngine()
        >>> result = engine.calculate_product_level(
        ...     quantity=Decimal("500"),
        ...     product_ef=Decimal("2.33"),
        ...     item_id="ITEM-001",
        ...     supplier_id="SUP-001",
        ...     data_source=SupplierDataSource.EPD,
        ... )
        >>> result.emissions_kgco2e
        Decimal('1165.00000000')
    """

    # -- Class-level singleton state ----------------------------------------
    _instance: Optional[SupplierSpecificCalculatorEngine] = None
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(cls) -> SupplierSpecificCalculatorEngine:
        """Return the singleton instance, creating it on first call.

        Uses a threading RLock for double-checked locking so that
        concurrent threads do not create multiple instances.

        Returns:
            The singleton SupplierSpecificCalculatorEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize engine internals on first call only.

        Loads configuration, metrics, and provenance singletons.
        Subsequent calls are guarded by the ``_initialized`` flag.
        """
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return

            self._config: PurchasedGoodsServicesConfig = (
                PurchasedGoodsServicesConfig()
            )
            self._metrics: PurchasedGoodsServicesMetrics = (
                PurchasedGoodsServicesMetrics()
            )
            self._provenance: PurchasedGoodsProvenanceTracker = (
                PurchasedGoodsProvenanceTracker.get_instance()
            )

            # Decimal precision from config
            dp = getattr(self._config, "decimal_places", DECIMAL_PLACES)
            self._precision: Decimal = Decimal("0." + "0" * dp)

            # Running counters
            self._calculation_count: int = 0
            self._batch_count: int = 0
            self._error_count: int = 0

            self._initialized = True

            logger.info(
                "SupplierSpecificCalculatorEngine initialized: "
                "agent=%s, version=%s, decimal_places=%d",
                AGENT_ID,
                VERSION,
                dp,
            )

    # ------------------------------------------------------------------
    # Singleton reset (for testing)
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance.

        Intended for use in unit tests to ensure a clean state between
        test runs. Not safe to call in production while calculations
        are in flight.
        """
        with cls._lock:
            cls._instance = None
            logger.info("SupplierSpecificCalculatorEngine singleton reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _q(self, value: Decimal) -> Decimal:
        """Quantize a value using the engine's configured precision.

        Args:
            value: Decimal value to quantize.

        Returns:
            Quantized Decimal.
        """
        return _quantize(value, self._precision)

    def _provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute a SHA-256 provenance hash for the given data.

        Args:
            data: Dictionary of calculation inputs and outputs.

        Returns:
            64-character hexadecimal SHA-256 digest.
        """
        data_with_agent = {
            "agent_id": AGENT_ID,
            "version": VERSION,
            "engine": "supplier_specific_calculator",
            "timestamp": utcnow().isoformat(),
            **data,
        }
        return _compute_sha256(data_with_agent)

    def _record_metrics(
        self,
        status: str,
        duration_s: float,
        emissions_kgco2e: Optional[float] = None,
        spend_usd: Optional[float] = None,
        data_source: Optional[str] = None,
    ) -> None:
        """Record a calculation to Prometheus metrics.

        Args:
            status: Calculation status (success/failed).
            duration_s: Duration in seconds.
            emissions_kgco2e: Emissions calculated (optional).
            spend_usd: Spend amount (optional).
            data_source: Supplier data source name (optional).
        """
        try:
            tenant_id = getattr(self._config, "default_tenant", "default")
            self._metrics.record_calculation(
                tenant_id=tenant_id,
                method="supplier_specific",
                status=status,
                duration_s=duration_s,
                emissions_kgco2e=emissions_kgco2e,
                spend_usd=spend_usd,
            )
            if data_source:
                self._metrics.record_supplier_assessed(
                    data_source=data_source,
                    count=1,
                )
        except Exception as exc:
            logger.warning(
                "Failed to record supplier-specific metrics: %s", exc,
            )

    def _build_result(
        self,
        *,
        item_id: str,
        emissions_kgco2e: Decimal,
        supplier_id: str,
        supplier_name: Optional[str],
        data_source: SupplierDataSource,
        allocation_method: AllocationMethod,
        allocation_factor: Decimal,
        supplier_total_tco2e: Optional[Decimal],
        product_ef_kgco2e_per_unit: Optional[Decimal],
        quantity: Optional[Decimal],
        verification_status: str,
        provenance_data: Dict[str, Any],
    ) -> SupplierSpecificResult:
        """Construct a SupplierSpecificResult with derived fields.

        Computes emissions in tonnes CO2e from kgCO2e, generates
        the provenance hash, and assembles the frozen Pydantic model.

        Args:
            item_id: Reference to the source procurement item.
            emissions_kgco2e: Calculated emissions in kgCO2e.
            supplier_id: Unique identifier of the supplier.
            supplier_name: Human-readable supplier name.
            data_source: Source of the supplier emission data.
            allocation_method: Allocation method used.
            allocation_factor: Allocation factor applied (0-1).
            supplier_total_tco2e: Total supplier facility emissions.
            product_ef_kgco2e_per_unit: Product-level EF if used.
            quantity: Quantity purchased.
            verification_status: Verification status string.
            provenance_data: Data dictionary for provenance hashing.

        Returns:
            Frozen SupplierSpecificResult Pydantic model.
        """
        emissions_kgco2e_q = self._q(emissions_kgco2e)
        emissions_tco2e_q = self._q(emissions_kgco2e_q * _KG_TO_TONNES)

        prov_hash = self._provenance_hash({
            **provenance_data,
            "emissions_kgco2e": str(emissions_kgco2e_q),
            "emissions_tco2e": str(emissions_tco2e_q),
        })

        return SupplierSpecificResult(
            item_id=item_id,
            emissions_kgco2e=emissions_kgco2e_q,
            emissions_tco2e=emissions_tco2e_q,
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            data_source=data_source,
            allocation_method=allocation_method,
            allocation_factor=self._q(allocation_factor),
            supplier_total_tco2e=(
                self._q(supplier_total_tco2e)
                if supplier_total_tco2e is not None else None
            ),
            product_ef_kgco2e_per_unit=(
                self._q(product_ef_kgco2e_per_unit)
                if product_ef_kgco2e_per_unit is not None else None
            ),
            quantity=(
                self._q(quantity)
                if quantity is not None else None
            ),
            verification_status=verification_status,
            provenance_hash=prov_hash,
        )

    # ==================================================================
    # Public API: Single-Item Calculations
    # ==================================================================

    def calculate_single(
        self,
        item: ProcurementItem,
        supplier_data: SupplierRecord,
    ) -> SupplierSpecificResult:
        """Calculate supplier-specific emissions for a single procurement item.

        Automatically selects the appropriate calculation path:
        - If the supplier provides a product-level EF (product_ef_kgco2e_per_unit
          is set and quantity is available), uses product-level calculation.
        - If the supplier provides a PCF value, uses that as a product-level EF.
        - Otherwise, falls back to facility-level calculation with the
          allocation method and factor from the supplier record.

        Args:
            item: The procurement item with quantity and supplier info.
            supplier_data: Supplier-specific data record with emission factors,
                allocation settings, and verification status.

        Returns:
            SupplierSpecificResult with calculated emissions.

        Raises:
            ValueError: If neither product-level EF nor facility-level
                total emissions are available.
            ValueError: If quantity is required but not provided.
        """
        start_time = time.monotonic()

        try:
            # ---- Validate boundary ----
            boundary_info = self.verify_boundary(supplier_data.boundary)
            if not boundary_info["is_acceptable"]:
                raise ValueError(
                    f"System boundary '{supplier_data.boundary}' is not "
                    f"acceptable for supplier-specific calculation. "
                    f"Reason: {boundary_info['reason']}"
                )

            # ---- Determine calculation path ----
            has_product_ef = (
                supplier_data.product_ef_kgco2e_per_unit is not None
                and supplier_data.product_ef_kgco2e_per_unit > ZERO
            )
            has_pcf = (
                supplier_data.pcf_value_kgco2e is not None
                and supplier_data.pcf_value_kgco2e > ZERO
            )
            has_facility_total = (
                supplier_data.supplier_emissions_tco2e is not None
                and supplier_data.supplier_emissions_tco2e > ZERO
            )
            has_quantity = (
                item.quantity is not None
                and item.quantity > ZERO
            )

            # Path 1: Product-level EF with quantity
            if has_product_ef and has_quantity:
                result = self.calculate_product_level(
                    quantity=item.quantity,
                    product_ef=supplier_data.product_ef_kgco2e_per_unit,
                    item_id=item.item_id,
                    supplier_id=item.supplier_id or "UNKNOWN",
                    data_source=supplier_data.data_source,
                    supplier_name=item.supplier_name,
                    verification_status=supplier_data.verification_status,
                )
                duration = time.monotonic() - start_time
                self._record_metrics(
                    status="success",
                    duration_s=duration,
                    emissions_kgco2e=float(result.emissions_kgco2e),
                    data_source=supplier_data.data_source.value,
                )
                self._calculation_count += 1
                return result

            # Path 2: PCF value with quantity
            if has_pcf and has_quantity:
                result = self.calculate_product_level(
                    quantity=item.quantity,
                    product_ef=supplier_data.pcf_value_kgco2e,
                    item_id=item.item_id,
                    supplier_id=item.supplier_id or "UNKNOWN",
                    data_source=SupplierDataSource.PACT_NETWORK,
                    supplier_name=item.supplier_name,
                    verification_status=supplier_data.verification_status,
                )
                duration = time.monotonic() - start_time
                self._record_metrics(
                    status="success",
                    duration_s=duration,
                    emissions_kgco2e=float(result.emissions_kgco2e),
                    data_source=supplier_data.data_source.value,
                )
                self._calculation_count += 1
                return result

            # Path 3: Facility-level with allocation
            if has_facility_total:
                result = self.calculate_facility_allocation(
                    supplier_total_tco2e=supplier_data.supplier_emissions_tco2e,
                    allocation_factor=supplier_data.allocation_factor,
                    item_id=item.item_id,
                    supplier_id=item.supplier_id or "UNKNOWN",
                    allocation_method=supplier_data.allocation_method,
                    supplier_name=item.supplier_name,
                    verification_status=supplier_data.verification_status,
                )
                duration = time.monotonic() - start_time
                self._record_metrics(
                    status="success",
                    duration_s=duration,
                    emissions_kgco2e=float(result.emissions_kgco2e),
                    data_source=supplier_data.data_source.value,
                )
                self._calculation_count += 1
                return result

            # No valid calculation path
            raise ValueError(
                f"Insufficient supplier data for item '{item.item_id}': "
                f"need either (product_ef + quantity) or "
                f"(facility_emissions + allocation_factor). "
                f"product_ef={supplier_data.product_ef_kgco2e_per_unit}, "
                f"pcf={supplier_data.pcf_value_kgco2e}, "
                f"facility_total={supplier_data.supplier_emissions_tco2e}, "
                f"quantity={item.quantity}"
            )

        except ValueError:
            duration = time.monotonic() - start_time
            self._record_metrics(status="failed", duration_s=duration)
            self._error_count += 1
            raise

        except Exception as exc:
            duration = time.monotonic() - start_time
            self._record_metrics(status="failed", duration_s=duration)
            self._error_count += 1
            logger.error(
                "Unexpected error in calculate_single for item=%s: %s",
                item.item_id, exc, exc_info=True,
            )
            raise

    def calculate_from_record(
        self,
        record: SupplierRecord,
    ) -> SupplierSpecificResult:
        """Calculate supplier-specific emissions from a SupplierRecord.

        Convenience method that extracts the ProcurementItem from the
        SupplierRecord and delegates to ``calculate_single``.

        Args:
            record: A SupplierRecord containing both the ProcurementItem
                and supplier-specific emission data.

        Returns:
            SupplierSpecificResult with calculated emissions.

        Raises:
            ValueError: If the record lacks sufficient data.
        """
        return self.calculate_single(
            item=record.item,
            supplier_data=record,
        )

    # ==================================================================
    # Public API: Batch Calculations
    # ==================================================================

    def calculate_batch(
        self,
        records: List[SupplierRecord],
    ) -> List[SupplierSpecificResult]:
        """Calculate supplier-specific emissions for a batch of records.

        Processes each record independently, collecting results and logging
        failures. Individual record failures do not abort the batch.

        Args:
            records: List of SupplierRecord objects to process.

        Returns:
            List of SupplierSpecificResult objects (one per successful
            calculation). Failed records are logged and skipped.

        Raises:
            ValueError: If the batch exceeds the maximum size.
        """
        if len(records) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(records)} exceeds maximum "
                f"of {_MAX_BATCH_SIZE}"
            )

        start_time = time.monotonic()
        results: List[SupplierSpecificResult] = []
        error_count = 0

        logger.info(
            "Starting supplier-specific batch calculation: "
            "%d records", len(records),
        )

        for idx, record in enumerate(records):
            try:
                result = self.calculate_from_record(record)
                results.append(result)
            except Exception as exc:
                error_count += 1
                logger.warning(
                    "Batch record %d/%d failed (item=%s): %s",
                    idx + 1, len(records),
                    record.item.item_id, exc,
                )

        duration = time.monotonic() - start_time
        self._batch_count += 1

        logger.info(
            "Supplier-specific batch complete: %d/%d successful, "
            "%d failed, duration=%.3fs",
            len(results), len(records), error_count, duration,
        )

        try:
            self._metrics.record_batch_job(
                status="completed" if error_count == 0 else "partial",
                record_count=len(records),
                duration_s=duration,
            )
        except Exception as exc:
            logger.warning("Failed to record batch metrics: %s", exc)

        return results

    # ==================================================================
    # Public API: Product-Level Calculation
    # ==================================================================

    def calculate_product_level(
        self,
        quantity: Decimal,
        product_ef: Decimal,
        item_id: str,
        supplier_id: str,
        data_source: SupplierDataSource,
        supplier_name: Optional[str] = None,
        verification_status: str = "unverified",
    ) -> SupplierSpecificResult:
        """Calculate emissions using product-level emission factors.

        Implements the formula:
            Emissions_kgCO2e = Quantity x Product_EF_kgCO2e_per_unit

        This is the preferred and most accurate calculation path, used
        when the supplier provides product-level emission factors from
        an EPD, PCF, or direct measurement.

        Args:
            quantity: Physical quantity of the product purchased (units
                matching the EF denominator).
            product_ef: Product-level emission factor in kgCO2e per unit.
            item_id: Unique identifier for the procurement item.
            supplier_id: Unique identifier for the supplier.
            data_source: Source of the supplier emission data.
            supplier_name: Optional human-readable supplier name.
            verification_status: Verification status of the data.

        Returns:
            SupplierSpecificResult with product-level emissions.

        Raises:
            ValueError: If quantity is zero or negative.
            ValueError: If product_ef is negative.
        """
        # ---- Validate inputs ----
        if quantity <= ZERO:
            raise ValueError(
                f"Quantity must be positive, got {quantity} "
                f"for item={item_id}"
            )
        if product_ef < ZERO:
            raise ValueError(
                f"Product emission factor must be non-negative, "
                f"got {product_ef} for item={item_id}"
            )

        # ---- Zero EF shortcut ----
        if product_ef == ZERO:
            logger.info(
                "Product EF is zero for item=%s supplier=%s; "
                "emissions will be zero",
                item_id, supplier_id,
            )

        # ---- Calculate emissions (ZERO-HALLUCINATION) ----
        emissions_kgco2e = self._q(quantity * product_ef)

        # ---- Build provenance data ----
        provenance_data = {
            "calculation_path": "product_level",
            "formula": "Emissions = Quantity * Product_EF",
            "item_id": item_id,
            "supplier_id": supplier_id,
            "quantity": str(quantity),
            "product_ef_kgco2e_per_unit": str(product_ef),
            "data_source": data_source.value,
            "verification_status": verification_status,
        }

        logger.debug(
            "Product-level calculation: item=%s, supplier=%s, "
            "qty=%s, ef=%s, emissions=%s kgCO2e",
            item_id, supplier_id, quantity, product_ef, emissions_kgco2e,
        )

        return self._build_result(
            item_id=item_id,
            emissions_kgco2e=emissions_kgco2e,
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            data_source=data_source,
            allocation_method=AllocationMethod.PHYSICAL_UNIT,
            allocation_factor=ONE,
            supplier_total_tco2e=None,
            product_ef_kgco2e_per_unit=product_ef,
            quantity=quantity,
            verification_status=verification_status,
            provenance_data=provenance_data,
        )

    # ==================================================================
    # Public API: Facility-Level Allocation Calculation
    # ==================================================================

    def calculate_facility_allocation(
        self,
        supplier_total_tco2e: Decimal,
        allocation_factor: Decimal,
        item_id: str,
        supplier_id: str,
        allocation_method: AllocationMethod,
        supplier_name: Optional[str] = None,
        verification_status: str = "unverified",
    ) -> SupplierSpecificResult:
        """Calculate emissions using facility-level data with allocation.

        Implements the formula:
            Emissions_kgCO2e = Supplier_Total_tCO2e * Allocation_Factor * 1000

        The allocation factor represents the reporting company's share
        of the supplier's total facility emissions, computed using one of
        five allocation methods.

        Args:
            supplier_total_tco2e: Total supplier facility emissions in
                tonnes CO2e.
            allocation_factor: Share of total emissions attributable to
                the reporting company (0 to 1 inclusive).
            item_id: Unique identifier for the procurement item.
            supplier_id: Unique identifier for the supplier.
            allocation_method: Method used to compute the allocation factor.
            supplier_name: Optional human-readable supplier name.
            verification_status: Verification status of the data.

        Returns:
            SupplierSpecificResult with facility-allocated emissions.

        Raises:
            ValueError: If supplier_total_tco2e is negative.
            ValueError: If allocation_factor is outside [0, 1].
        """
        # ---- Validate inputs ----
        if supplier_total_tco2e < ZERO:
            raise ValueError(
                f"Supplier total emissions must be non-negative, "
                f"got {supplier_total_tco2e} for item={item_id}"
            )
        if allocation_factor < ZERO or allocation_factor > ONE:
            raise ValueError(
                f"Allocation factor must be in [0, 1], "
                f"got {allocation_factor} for item={item_id}"
            )

        # ---- Zero allocation shortcut ----
        if allocation_factor == ZERO or supplier_total_tco2e == ZERO:
            logger.info(
                "Zero allocation or zero total for item=%s supplier=%s; "
                "emissions will be zero",
                item_id, supplier_id,
            )

        # ---- Calculate emissions (ZERO-HALLUCINATION) ----
        allocated_tco2e = self._q(supplier_total_tco2e * allocation_factor)
        emissions_kgco2e = self._q(allocated_tco2e * _TONNES_TO_KG)

        # ---- Build provenance data ----
        provenance_data = {
            "calculation_path": "facility_allocation",
            "formula": "Emissions = Supplier_Total * Allocation_Factor * 1000",
            "item_id": item_id,
            "supplier_id": supplier_id,
            "supplier_total_tco2e": str(supplier_total_tco2e),
            "allocation_factor": str(allocation_factor),
            "allocated_tco2e": str(allocated_tco2e),
            "allocation_method": allocation_method.value,
            "verification_status": verification_status,
        }

        logger.debug(
            "Facility allocation: item=%s, supplier=%s, "
            "total=%s tCO2e, factor=%s, allocated=%s tCO2e, "
            "emissions=%s kgCO2e",
            item_id, supplier_id, supplier_total_tco2e,
            allocation_factor, allocated_tco2e, emissions_kgco2e,
        )

        return self._build_result(
            item_id=item_id,
            emissions_kgco2e=emissions_kgco2e,
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            data_source=SupplierDataSource.DIRECT_MEASUREMENT,
            allocation_method=allocation_method,
            allocation_factor=allocation_factor,
            supplier_total_tco2e=supplier_total_tco2e,
            product_ef_kgco2e_per_unit=None,
            quantity=None,
            verification_status=verification_status,
            provenance_data=provenance_data,
        )

    # ==================================================================
    # Public API: Allocation Factor Computation
    # ==================================================================

    def compute_allocation_factor(
        self,
        method: AllocationMethod,
        customer_value: Decimal,
        total_value: Decimal,
    ) -> Decimal:
        """Compute the allocation factor for facility-level data.

        Given a customer's share value and the total facility value,
        computes the allocation factor as a ratio between 0 and 1.

        For the ``EQUAL`` allocation method, ``customer_value`` represents
        the number of customers and the factor is ``1 / customer_value``.

        Args:
            method: The allocation method to use.
            customer_value: For revenue/mass/economic/physical methods,
                the customer's share of the total. For EQUAL, the number
                of customers.
            total_value: The total facility value (revenue, mass, output).
                Ignored for the EQUAL method.

        Returns:
            Allocation factor as a Decimal between 0 and 1.

        Raises:
            ValueError: If customer_value or total_value is negative.
            ValueError: If customer_value exceeds total_value (except EQUAL).
            ValueError: If total_value is zero (except EQUAL).
        """
        if customer_value < ZERO:
            raise ValueError(
                f"Customer value must be non-negative, got {customer_value}"
            )

        # ---- EQUAL allocation ----
        if method == AllocationMethod.EQUAL:
            num_customers = int(customer_value) if customer_value > ZERO else _DEFAULT_EQUAL_CUSTOMERS
            if num_customers <= 0:
                num_customers = _DEFAULT_EQUAL_CUSTOMERS
            factor = self._q(ONE / Decimal(str(num_customers)))
            logger.debug(
                "Equal allocation: %d customers, factor=%s",
                num_customers, factor,
            )
            return factor

        # ---- Ratio-based allocation ----
        if total_value < ZERO:
            raise ValueError(
                f"Total value must be non-negative, got {total_value}"
            )
        if total_value == ZERO:
            raise ValueError(
                "Total value cannot be zero for ratio-based allocation"
            )
        if customer_value > total_value:
            raise ValueError(
                f"Customer value ({customer_value}) exceeds "
                f"total value ({total_value})"
            )

        factor = self._q(customer_value / total_value)

        # Clamp to [0, 1]
        if factor < ZERO:
            factor = ZERO
        elif factor > ONE:
            factor = ONE

        logger.debug(
            "Allocation factor computed: method=%s, "
            "customer=%s, total=%s, factor=%s",
            method.value, customer_value, total_value, factor,
        )

        return factor

    # ==================================================================
    # Public API: EPD Integration
    # ==================================================================

    def process_epd_data(
        self,
        epd_number: str,
        declared_unit: str,
        gwp_total: Decimal,
        quantity: Decimal,
        item_id: str,
        supplier_id: str,
        reporting_year: Optional[int] = None,
        boundary: str = "cradle_to_gate",
        verification_status: str = "third_party_verified",
    ) -> SupplierSpecificResult:
        """Process Environmental Product Declaration (EPD) data.

        Parses EPD data conforming to ISO 14025 / EN 15804 and
        calculates emissions based on the declared unit's GWP total.

        The formula is:
            Product_EF = GWP_Total / DeclaredUnit_quantity (implied 1)
            Emissions_kgCO2e = Quantity * Product_EF

        EPDs are validated for:
        - Non-negative GWP total
        - Acceptable system boundary (cradle-to-gate)
        - Validity period (max 5 years from reporting year)

        Args:
            epd_number: EPD registration number (e.g. "S-P-01234").
            declared_unit: Declared unit description (e.g. "1 kg").
            gwp_total: GWP total per declared unit in kgCO2e.
            quantity: Quantity purchased in declared units.
            item_id: Unique identifier for the procurement item.
            supplier_id: Unique identifier for the supplier.
            reporting_year: Year the EPD was published.
            boundary: System boundary of the EPD.
            verification_status: Verification status.

        Returns:
            SupplierSpecificResult with EPD-based emissions.

        Raises:
            ValueError: If the EPD fails validation.
        """
        # ---- Validate EPD ----
        validation = self.validate_epd(
            epd_number=epd_number,
            gwp_total=gwp_total,
            boundary=boundary,
            reporting_year=reporting_year,
        )
        if not validation["is_valid"]:
            raise ValueError(
                f"EPD validation failed for {epd_number}: "
                f"{validation['errors']}"
            )

        # ---- Validate quantity ----
        if quantity <= ZERO:
            raise ValueError(
                f"Quantity must be positive for EPD calculation, "
                f"got {quantity}"
            )

        # ---- Calculate using product-level path ----
        # GWP_Total is per declared unit, so it IS the product EF
        product_ef = gwp_total

        result = self.calculate_product_level(
            quantity=quantity,
            product_ef=product_ef,
            item_id=item_id,
            supplier_id=supplier_id,
            data_source=SupplierDataSource.EPD,
            verification_status=verification_status,
        )

        logger.info(
            "EPD processed: epd=%s, declared_unit=%s, gwp=%s, "
            "qty=%s, emissions=%s kgCO2e",
            epd_number, declared_unit, gwp_total,
            quantity, result.emissions_kgco2e,
        )

        return result

    def validate_epd(
        self,
        epd_number: str,
        gwp_total: Decimal,
        boundary: str,
        reporting_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validate an Environmental Product Declaration.

        Checks the EPD for compliance with ISO 14025 requirements:
        1. EPD number is non-empty
        2. GWP total is non-negative
        3. System boundary is cradle-to-gate
        4. EPD is within validity period (5 years)

        Args:
            epd_number: EPD registration number.
            gwp_total: GWP total per declared unit in kgCO2e.
            boundary: System boundary of the EPD.
            reporting_year: Year the EPD was published.

        Returns:
            Dictionary with keys:
                - is_valid (bool): Whether the EPD passes all checks.
                - errors (List[str]): List of validation error messages.
                - warnings (List[str]): List of validation warnings.
                - boundary_status (str): Boundary verification result.
                - age_years (Optional[int]): Age of EPD in years.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # ---- Check EPD number ----
        if not epd_number or not epd_number.strip():
            errors.append("EPD number is empty or blank")

        # ---- Check GWP total ----
        if gwp_total < ZERO:
            errors.append(
                f"GWP total must be non-negative, got {gwp_total}"
            )
        elif gwp_total == ZERO:
            warnings.append(
                "GWP total is zero; verify this is correct for the product"
            )

        # ---- Verify boundary ----
        boundary_info = self.verify_boundary(boundary)
        if not boundary_info["is_acceptable"]:
            errors.append(
                f"System boundary '{boundary}' is not acceptable: "
                f"{boundary_info['reason']}"
            )
        if boundary_info.get("requires_adjustment", False):
            warnings.append(
                f"System boundary '{boundary}' requires adjustment: "
                f"{boundary_info.get('adjustment_note', '')}"
            )

        # ---- Check validity period ----
        age_years: Optional[int] = None
        if reporting_year is not None:
            current_year = _current_reporting_year()
            age_years = current_year - reporting_year
            if age_years > _EPD_MAX_AGE_YEARS:
                errors.append(
                    f"EPD is {age_years} years old (published {reporting_year}); "
                    f"maximum validity is {_EPD_MAX_AGE_YEARS} years"
                )
            elif age_years > _EPD_MAX_AGE_YEARS - 1:
                warnings.append(
                    f"EPD will expire within 1 year (age: {age_years} years)"
                )
            if age_years < 0:
                warnings.append(
                    f"EPD reporting year {reporting_year} is in the future"
                )

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "boundary_status": boundary_info.get("status", "unknown"),
            "age_years": age_years,
        }

    # ==================================================================
    # Public API: CDP Supply Chain Integration
    # ==================================================================

    def process_cdp_data(
        self,
        supplier_id: str,
        supplier_scope1_tco2e: Decimal,
        supplier_scope2_tco2e: Decimal,
        supplier_total_revenue: Decimal,
        customer_revenue: Decimal,
        cdp_score: str,
        supplier_name: Optional[str] = None,
        item_id: Optional[str] = None,
        verification_status: str = "unverified",
    ) -> SupplierSpecificResult:
        """Process CDP Supply Chain programme data.

        Calculates Scope 3 Category 1 emissions using the supplier's
        reported Scope 1+2 emissions allocated by revenue share:

            Emissions = (Scope1 + Scope2) * (CustomerRevenue / TotalRevenue)

        CDP score quality affects the DQI reliability dimension but does
        not modify the emission calculation itself (zero-hallucination).

        Args:
            supplier_id: Unique identifier for the supplier.
            supplier_scope1_tco2e: Supplier's Scope 1 emissions in tCO2e.
            supplier_scope2_tco2e: Supplier's Scope 2 emissions in tCO2e.
            supplier_total_revenue: Supplier's total annual revenue (USD).
            customer_revenue: Revenue from the reporting company (USD).
            cdp_score: CDP letter grade (A, A-, B, B-, C, C-, D, D-, F).
            supplier_name: Optional human-readable supplier name.
            item_id: Optional procurement item ID. Generated if not provided.
            verification_status: Verification status of the data.

        Returns:
            SupplierSpecificResult with CDP-based emissions.

        Raises:
            ValueError: If Scope 1 or Scope 2 is negative.
            ValueError: If revenue values are invalid.
        """
        # ---- Generate item_id if not provided ----
        if item_id is None:
            item_id = f"CDP-{supplier_id}-{str(uuid.uuid4())[:8]}"

        # ---- Validate inputs ----
        if supplier_scope1_tco2e < ZERO:
            raise ValueError(
                f"Scope 1 emissions must be non-negative, "
                f"got {supplier_scope1_tco2e}"
            )
        if supplier_scope2_tco2e < ZERO:
            raise ValueError(
                f"Scope 2 emissions must be non-negative, "
                f"got {supplier_scope2_tco2e}"
            )
        if supplier_total_revenue <= ZERO:
            raise ValueError(
                f"Total revenue must be positive, "
                f"got {supplier_total_revenue}"
            )
        if customer_revenue < ZERO:
            raise ValueError(
                f"Customer revenue must be non-negative, "
                f"got {customer_revenue}"
            )
        if customer_revenue > supplier_total_revenue:
            raise ValueError(
                f"Customer revenue ({customer_revenue}) exceeds "
                f"total revenue ({supplier_total_revenue})"
            )

        # ---- Validate CDP score ----
        cdp_score_upper = cdp_score.strip().upper()
        cdp_quality = _CDP_SCORE_QUALITY.get(
            cdp_score_upper, Decimal("0.50")
        )

        # ---- Calculate (ZERO-HALLUCINATION) ----
        total_scope12_tco2e = self._q(
            supplier_scope1_tco2e + supplier_scope2_tco2e
        )
        revenue_allocation = self._q(
            customer_revenue / supplier_total_revenue
        )
        allocated_tco2e = self._q(total_scope12_tco2e * revenue_allocation)
        emissions_kgco2e = self._q(allocated_tco2e * _TONNES_TO_KG)

        # ---- Build provenance data ----
        provenance_data = {
            "calculation_path": "cdp_supply_chain",
            "formula": "(Scope1 + Scope2) * (CustomerRev / TotalRev)",
            "item_id": item_id,
            "supplier_id": supplier_id,
            "scope1_tco2e": str(supplier_scope1_tco2e),
            "scope2_tco2e": str(supplier_scope2_tco2e),
            "total_scope12_tco2e": str(total_scope12_tco2e),
            "total_revenue": str(supplier_total_revenue),
            "customer_revenue": str(customer_revenue),
            "revenue_allocation": str(revenue_allocation),
            "cdp_score": cdp_score_upper,
            "cdp_quality_weight": str(cdp_quality),
        }

        logger.info(
            "CDP calculation: supplier=%s, scope12=%s tCO2e, "
            "rev_alloc=%s, allocated=%s tCO2e, "
            "emissions=%s kgCO2e, score=%s",
            supplier_id, total_scope12_tco2e, revenue_allocation,
            allocated_tco2e, emissions_kgco2e, cdp_score_upper,
        )

        return self._build_result(
            item_id=item_id,
            emissions_kgco2e=emissions_kgco2e,
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            data_source=SupplierDataSource.CDP_SUPPLY_CHAIN,
            allocation_method=AllocationMethod.REVENUE_BASED,
            allocation_factor=revenue_allocation,
            supplier_total_tco2e=total_scope12_tco2e,
            product_ef_kgco2e_per_unit=None,
            quantity=None,
            verification_status=verification_status,
            provenance_data=provenance_data,
        )

    # ==================================================================
    # Public API: Boundary Verification
    # ==================================================================

    def verify_boundary(self, boundary: str) -> Dict[str, Any]:
        """Verify the system boundary of supplier emission data.

        Ensures only cradle-to-gate data is used directly. Cradle-to-grave
        data is flagged for manual adjustment because it includes use-phase
        and end-of-life emissions that are not part of Category 1.

        Args:
            boundary: System boundary string from the supplier data.

        Returns:
            Dictionary with keys:
                - is_acceptable (bool): True if boundary can be used.
                - status (str): "accepted", "requires_adjustment",
                    or "rejected".
                - reason (str): Explanation of the decision.
                - boundary_normalized (str): Normalized boundary value.
                - requires_adjustment (bool): Whether manual adjustment
                    is needed.
                - adjustment_note (str): Guidance for adjustment if needed.
        """
        if boundary is None:
            return {
                "is_acceptable": False,
                "status": "rejected",
                "reason": "Boundary is None; cannot determine scope",
                "boundary_normalized": "",
                "requires_adjustment": False,
                "adjustment_note": "",
            }

        boundary_lower = boundary.strip().lower()

        # ---- Cradle-to-gate: accepted ----
        if boundary_lower in _VALID_CTG_BOUNDARIES:
            return {
                "is_acceptable": True,
                "status": "accepted",
                "reason": "Cradle-to-gate boundary is acceptable for Category 1",
                "boundary_normalized": "cradle_to_gate",
                "requires_adjustment": False,
                "adjustment_note": "",
            }

        # ---- Cradle-to-grave: requires adjustment ----
        if boundary_lower in _ADJUSTABLE_BOUNDARIES:
            return {
                "is_acceptable": False,
                "status": "requires_adjustment",
                "reason": (
                    "Cradle-to-grave boundary includes use-phase and "
                    "end-of-life emissions not attributable to Category 1. "
                    "Remove use-phase and end-of-life components to obtain "
                    "cradle-to-gate value."
                ),
                "boundary_normalized": "cradle_to_grave",
                "requires_adjustment": True,
                "adjustment_note": (
                    "Subtract use-phase (B1-B7) and end-of-life "
                    "(C1-C4, D) modules from the total GWP to obtain "
                    "cradle-to-gate GWP (A1-A3)."
                ),
            }

        # ---- Gate-to-gate: accepted with caveat ----
        if boundary_lower in {"gate_to_gate", "gate-to-gate", "g2g", "gtg"}:
            return {
                "is_acceptable": True,
                "status": "accepted",
                "reason": (
                    "Gate-to-gate boundary captures direct supplier "
                    "operations. Note: upstream emissions of the "
                    "supplier's own inputs are excluded."
                ),
                "boundary_normalized": "gate_to_gate",
                "requires_adjustment": False,
                "adjustment_note": (
                    "Gate-to-gate may understate total cradle-to-gate "
                    "emissions. Consider adding upstream supply chain "
                    "emissions for completeness."
                ),
            }

        # ---- Unknown boundary: rejected ----
        return {
            "is_acceptable": False,
            "status": "rejected",
            "reason": (
                f"Unknown system boundary '{boundary}'. "
                f"Acceptable values: cradle_to_gate, gate_to_gate."
            ),
            "boundary_normalized": boundary_lower,
            "requires_adjustment": False,
            "adjustment_note": "",
        }

    # ==================================================================
    # Public API: Supplier Engagement Scoring
    # ==================================================================

    def score_supplier_engagement(
        self,
        supplier_data: SupplierRecord,
    ) -> int:
        """Score supplier engagement maturity on a 1-5 scale.

        Evaluates the depth and quality of supplier data disclosure
        to assess engagement maturity:

        - Level 5 (Leading): Product-level EPD, third-party verified,
          reporting year is current or prior year.
        - Level 4 (Established): Verified PCF or EPD with
          cradle-to-gate boundary.
        - Level 3 (Developing): Third-party assessed via CDP or
          EcoVadis, but may lack product-level detail.
        - Level 2 (Initial): Self-reported unverified data from
          sustainability reports or direct disclosure.
        - Level 1 (Unengaged): No supplier-specific data, or data
          fails validation.

        Args:
            supplier_data: SupplierRecord containing emission data,
                source, verification status, and boundary information.

        Returns:
            Integer engagement level (1-5).
        """
        # ---- Level 5: Leading ----
        # Requires: EPD/PCF with product-level EF, verified, recent
        has_product_ef = (
            supplier_data.product_ef_kgco2e_per_unit is not None
            and supplier_data.product_ef_kgco2e_per_unit > ZERO
        )
        has_epd = supplier_data.epd_number is not None and supplier_data.epd_number.strip()
        is_verified = supplier_data.verification_status.lower() in {
            "verified", "third_party_verified", "third-party-verified",
            "iso_verified", "epd_verified",
        }
        is_recent = False
        if supplier_data.reporting_year is not None:
            age = _current_reporting_year() - supplier_data.reporting_year
            is_recent = age <= 1
        is_ctg = supplier_data.boundary.strip().lower() in _VALID_CTG_BOUNDARIES

        if has_product_ef and has_epd and is_verified and is_recent and is_ctg:
            return 5

        # ---- Level 4: Established ----
        # Requires: Verified EF (product or PCF) with CTG boundary
        has_pcf = (
            supplier_data.pcf_value_kgco2e is not None
            and supplier_data.pcf_value_kgco2e > ZERO
        )
        if (has_product_ef or has_pcf) and is_verified and is_ctg:
            return 4

        # ---- Level 3: Developing ----
        # Requires: CDP or EcoVadis data source
        if supplier_data.data_source in {
            SupplierDataSource.CDP_SUPPLY_CHAIN,
            SupplierDataSource.ECOVADIS,
        }:
            return 3

        # ---- Level 2: Initial ----
        # Requires: Any data from sustainability report or direct disclosure
        has_any_data = (
            has_product_ef
            or has_pcf
            or (
                supplier_data.supplier_emissions_tco2e is not None
                and supplier_data.supplier_emissions_tco2e > ZERO
            )
        )
        if has_any_data:
            return 2

        # ---- Level 1: Unengaged ----
        return 1

    def get_engagement_recommendations(
        self,
        engagement_level: int,
    ) -> List[str]:
        """Get actionable recommendations based on engagement level.

        Provides specific next steps to improve supplier engagement
        and data quality from the current level to the next.

        Args:
            engagement_level: Current engagement level (1-5).

        Returns:
            List of recommendation strings.

        Raises:
            ValueError: If engagement_level is not 1-5.
        """
        if engagement_level < 1 or engagement_level > 5:
            raise ValueError(
                f"Engagement level must be 1-5, got {engagement_level}"
            )

        recommendations: Dict[int, List[str]] = {
            1: [
                "Initiate supplier engagement programme for this supplier.",
                "Request basic carbon footprint data via supplier questionnaire.",
                "Include carbon disclosure requirements in procurement contracts.",
                "Prioritize engagement if the supplier represents >1% of spend.",
                "Encourage the supplier to respond to CDP Supply Chain questionnaire.",
                "Provide supplier with GHG accounting guidance and templates.",
            ],
            2: [
                "Request third-party verification of reported emission data.",
                "Encourage supplier to register with CDP or EcoVadis.",
                "Ask supplier to provide cradle-to-gate boundary data.",
                "Request product-level emission factors instead of facility totals.",
                "Include data quality requirements in supplier scorecards.",
                "Provide feedback on data completeness and suggest improvements.",
            ],
            3: [
                "Request product-level EPD or PCF data from the supplier.",
                "Encourage supplier to obtain third-party verification.",
                "Ask supplier to provide annual updates to emission data.",
                "Collaborate on joint emission reduction initiatives.",
                "Include supplier in SBTi supplier engagement targets.",
                "Review CDP/EcoVadis scores for improvement opportunities.",
            ],
            4: [
                "Encourage supplier to publish EPDs via a programme operator.",
                "Request annual data updates to maintain data currency.",
                "Collaborate on product-level emission reduction targets.",
                "Consider PACT Network integration for automated data exchange.",
                "Joint development of low-carbon product alternatives.",
                "Share best practices and recognition for leading disclosure.",
            ],
            5: [
                "Maintain annual data update cadence with the supplier.",
                "Recognize supplier as a best-practice example in reporting.",
                "Co-develop science-based targets with the supplier.",
                "Explore Scope 3 reduction opportunities jointly.",
                "Share supplier success story in sustainability report.",
                "Integrate supplier data into real-time monitoring systems.",
            ],
        }

        return recommendations.get(engagement_level, [])

    # ==================================================================
    # Public API: DQI Scoring (Supplier-Specific)
    # ==================================================================

    def score_dqi_supplier_specific(
        self,
        record: SupplierRecord,
        result: SupplierSpecificResult,
    ) -> DQIAssessment:
        """Score data quality for a supplier-specific calculation result.

        Assesses data quality across the five GHG Protocol dimensions
        (temporal, geographical, technological, completeness, reliability)
        on a 1-5 scale. Supplier-specific data typically scores 1-2 on
        most dimensions, making it the highest-quality method.

        Scoring criteria:

        **Temporal (1-5)**:
            1 = Reporting year matches current or prior year
            2 = Data is 2-3 years old
            3 = Data is 4-5 years old
            4 = Data is 6-10 years old
            5 = Data is >10 years old or no year specified

        **Geographical (1-5)**:
            1 = Supplier-specific data from the actual facility
            2 = Data from supplier's reported region
            3 = Country-level supplier data
            4 = Regional average data
            5 = Global average data

        **Technological (1-5)**:
            1 = Product-level EPD/PCF for exact product
            2 = Product-group level emission factor
            3 = Industry-average factor from supplier category
            4 = Cross-industry average factor
            5 = Generic emission factor

        **Completeness (1-5)**:
            1 = All Scope 1+2+3 included with product EF
            2 = Scope 1+2 with product EF
            3 = Scope 1+2 with facility allocation
            4 = Scope 1 only or partial data
            5 = Estimate or placeholder data

        **Reliability (1-5)**:
            1 = Third-party verified EPD/PCF
            2 = Third-party verified facility data
            3 = Audited self-reported data (CDP, EcoVadis)
            4 = Unaudited self-reported data
            5 = Estimate or assumption

        Args:
            record: The SupplierRecord used for the calculation.
            result: The SupplierSpecificResult to score.

        Returns:
            DQIAssessment with scores, composite, tier, and findings.
        """
        findings: List[str] = []

        # ---- Temporal score ----
        temporal_score = self._score_temporal(record, findings)

        # ---- Geographical score ----
        geographical_score = self._score_geographical(record, findings)

        # ---- Technological score ----
        technological_score = self._score_technological(record, result, findings)

        # ---- Completeness score ----
        completeness_score = self._score_completeness(record, result, findings)

        # ---- Reliability score ----
        reliability_score = self._score_reliability(record, findings)

        # ---- Composite score (arithmetic mean) ----
        scores = [
            temporal_score,
            geographical_score,
            technological_score,
            completeness_score,
            reliability_score,
        ]
        composite_raw = sum(scores) / Decimal("5")
        composite_score = self._q(composite_raw)

        # Clamp to [1.0, 5.0]
        if composite_score < Decimal("1.0"):
            composite_score = Decimal("1.0")
        elif composite_score > Decimal("5.0"):
            composite_score = Decimal("5.0")

        # ---- Quality tier ----
        quality_tier = _get_quality_tier(composite_score)

        # ---- Uncertainty factor (pedigree matrix) ----
        uncertainty_factor = self._compute_pedigree_uncertainty(scores)

        # ---- EF hierarchy level ----
        ef_hierarchy_level = self._determine_ef_hierarchy_level(record)

        logger.debug(
            "DQI assessment: item=%s, temporal=%s, geo=%s, tech=%s, "
            "complete=%s, reliable=%s, composite=%s, tier=%s",
            result.item_id, temporal_score, geographical_score,
            technological_score, completeness_score, reliability_score,
            composite_score, quality_tier,
        )

        return DQIAssessment(
            item_id=result.item_id,
            calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
            temporal_score=temporal_score,
            geographical_score=geographical_score,
            technological_score=technological_score,
            completeness_score=completeness_score,
            reliability_score=reliability_score,
            composite_score=composite_score,
            quality_tier=quality_tier,
            uncertainty_factor=uncertainty_factor,
            findings=findings,
            ef_hierarchy_level=ef_hierarchy_level,
        )

    def _score_temporal(
        self,
        record: SupplierRecord,
        findings: List[str],
    ) -> Decimal:
        """Score temporal representativeness (1-5).

        Args:
            record: The supplier record to score.
            findings: List to append findings to.

        Returns:
            Temporal DQI score as Decimal.
        """
        if record.reporting_year is None:
            findings.append(
                "No reporting year specified; temporal score is worst-case (5)."
            )
            return Decimal("5.0")

        current_year = _current_reporting_year()
        age = current_year - record.reporting_year

        if age <= 1:
            findings.append(
                f"Data is current (reporting year {record.reporting_year}); "
                f"temporal score 1."
            )
            return Decimal("1.0")
        elif age <= 3:
            findings.append(
                f"Data is {age} years old; temporal score 2."
            )
            return Decimal("2.0")
        elif age <= 5:
            findings.append(
                f"Data is {age} years old; temporal score 3. "
                f"Consider requesting updated data from supplier."
            )
            return Decimal("3.0")
        elif age <= 10:
            findings.append(
                f"Data is {age} years old; temporal score 4. "
                f"Data should be updated to improve quality."
            )
            return Decimal("4.0")
        else:
            findings.append(
                f"Data is {age} years old; temporal score 5. "
                f"Data is outdated and should be replaced."
            )
            return Decimal("5.0")

    def _score_geographical(
        self,
        record: SupplierRecord,
        findings: List[str],
    ) -> Decimal:
        """Score geographical representativeness (1-5).

        For supplier-specific data, geographical representativeness is
        inherently high because the data comes from the specific supplier.

        Args:
            record: The supplier record to score.
            findings: List to append findings to.

        Returns:
            Geographical DQI score as Decimal.
        """
        # Supplier-specific data is from the actual supplier/facility
        has_product_ef = (
            record.product_ef_kgco2e_per_unit is not None
            and record.product_ef_kgco2e_per_unit > ZERO
        )
        has_facility_data = (
            record.supplier_emissions_tco2e is not None
            and record.supplier_emissions_tco2e > ZERO
        )

        if has_product_ef:
            findings.append(
                "Product-level EF from specific supplier; "
                "geographical score 1."
            )
            return Decimal("1.0")
        elif has_facility_data:
            findings.append(
                "Facility-level data from specific supplier; "
                "geographical score 1."
            )
            return Decimal("1.0")
        elif record.data_source == SupplierDataSource.CDP_SUPPLY_CHAIN:
            findings.append(
                "CDP Supply Chain data; geographical score 2 "
                "(corporate-level, not facility-specific)."
            )
            return Decimal("2.0")
        elif record.data_source == SupplierDataSource.ECOVADIS:
            findings.append(
                "EcoVadis data; geographical score 2 "
                "(company-level assessment)."
            )
            return Decimal("2.0")
        else:
            findings.append(
                "Limited geographical information; geographical score 3."
            )
            return Decimal("3.0")

    def _score_technological(
        self,
        record: SupplierRecord,
        result: SupplierSpecificResult,
        findings: List[str],
    ) -> Decimal:
        """Score technological representativeness (1-5).

        Args:
            record: The supplier record to score.
            result: The calculation result.
            findings: List to append findings to.

        Returns:
            Technological DQI score as Decimal.
        """
        has_product_ef = (
            result.product_ef_kgco2e_per_unit is not None
            and result.product_ef_kgco2e_per_unit > ZERO
        )
        has_epd = (
            record.epd_number is not None
            and record.epd_number.strip()
        )

        if has_product_ef and has_epd:
            findings.append(
                "Product-level EPD with specific technology; "
                "technological score 1."
            )
            return Decimal("1.0")
        elif has_product_ef:
            findings.append(
                "Product-level EF (non-EPD source); "
                "technological score 1."
            )
            return Decimal("1.0")
        elif record.data_source in {
            SupplierDataSource.PACT_NETWORK,
            SupplierDataSource.DIRECT_MEASUREMENT,
        }:
            findings.append(
                f"Direct supplier data ({record.data_source.value}); "
                f"technological score 2."
            )
            return Decimal("2.0")
        elif record.data_source in {
            SupplierDataSource.CDP_SUPPLY_CHAIN,
            SupplierDataSource.ECOVADIS,
        }:
            findings.append(
                f"Third-party platform data ({record.data_source.value}); "
                f"technological score 3."
            )
            return Decimal("3.0")
        elif record.data_source == SupplierDataSource.SUSTAINABILITY_REPORT:
            findings.append(
                "Sustainability report data; technological score 3."
            )
            return Decimal("3.0")
        else:
            findings.append(
                "Generic supplier data; technological score 4."
            )
            return Decimal("4.0")

    def _score_completeness(
        self,
        record: SupplierRecord,
        result: SupplierSpecificResult,
        findings: List[str],
    ) -> Decimal:
        """Score data completeness (1-5).

        Args:
            record: The supplier record to score.
            result: The calculation result.
            findings: List to append findings to.

        Returns:
            Completeness DQI score as Decimal.
        """
        has_product_ef = (
            result.product_ef_kgco2e_per_unit is not None
            and result.product_ef_kgco2e_per_unit > ZERO
        )
        has_facility_total = (
            result.supplier_total_tco2e is not None
            and result.supplier_total_tco2e > ZERO
        )
        is_ctg = record.boundary.strip().lower() in _VALID_CTG_BOUNDARIES

        if has_product_ef and is_ctg:
            findings.append(
                "Product-level cradle-to-gate data; completeness score 1."
            )
            return Decimal("1.0")
        elif has_product_ef:
            findings.append(
                "Product-level data (boundary not confirmed CTG); "
                "completeness score 2."
            )
            return Decimal("2.0")
        elif has_facility_total and is_ctg:
            findings.append(
                "Facility-level cradle-to-gate data with allocation; "
                "completeness score 2."
            )
            return Decimal("2.0")
        elif has_facility_total:
            findings.append(
                "Facility-level data with allocation; completeness score 3."
            )
            return Decimal("3.0")
        elif record.data_source == SupplierDataSource.CDP_SUPPLY_CHAIN:
            findings.append(
                "CDP Scope 1+2 data with revenue allocation; "
                "completeness score 3. Scope 3 upstream excluded."
            )
            return Decimal("3.0")
        else:
            findings.append(
                "Limited completeness information; completeness score 4."
            )
            return Decimal("4.0")

    def _score_reliability(
        self,
        record: SupplierRecord,
        findings: List[str],
    ) -> Decimal:
        """Score data reliability (1-5).

        Args:
            record: The supplier record to score.
            findings: List to append findings to.

        Returns:
            Reliability DQI score as Decimal.
        """
        verification_lower = record.verification_status.lower().strip()

        if verification_lower in {
            "verified", "third_party_verified",
            "third-party-verified", "third_party_verified_epd",
            "iso_verified", "epd_verified",
        }:
            findings.append(
                "Third-party verified data; reliability score 1."
            )
            return Decimal("1.0")
        elif verification_lower in {
            "audited", "internally_audited", "limited_assurance",
        }:
            findings.append(
                "Audited/internally verified data; reliability score 2."
            )
            return Decimal("2.0")
        elif record.data_source in {
            SupplierDataSource.CDP_SUPPLY_CHAIN,
            SupplierDataSource.ECOVADIS,
        }:
            findings.append(
                f"Platform-assessed data ({record.data_source.value}); "
                f"reliability score 2."
            )
            return Decimal("2.0")
        elif verification_lower in {"self_reported", "unverified"}:
            findings.append(
                "Self-reported unverified data; reliability score 4."
            )
            return Decimal("4.0")
        elif verification_lower in {"estimated", "assumed", "proxy"}:
            findings.append(
                "Estimated/assumed data; reliability score 5."
            )
            return Decimal("5.0")
        else:
            findings.append(
                f"Unknown verification status '{record.verification_status}'; "
                f"reliability score 3."
            )
            return Decimal("3.0")

    def _compute_pedigree_uncertainty(
        self,
        scores: List[Decimal],
    ) -> Decimal:
        """Compute combined pedigree uncertainty factor from DQI scores.

        Uses the ecoinvent pedigree matrix methodology: each dimension
        score maps to a basic uncertainty factor, and the combined
        factor is the product of all individual factors.

        Args:
            scores: List of five DQI dimension scores.

        Returns:
            Combined pedigree uncertainty factor (>= 1.0).
        """
        combined = ONE
        for score in scores:
            # Map numeric score to DQIScore enum
            if score <= Decimal("1.5"):
                pedigree = PEDIGREE_UNCERTAINTY_FACTORS.get(
                    DQIScore.VERY_GOOD, Decimal("1.00")
                )
            elif score <= Decimal("2.5"):
                pedigree = PEDIGREE_UNCERTAINTY_FACTORS.get(
                    DQIScore.GOOD, Decimal("1.05")
                )
            elif score <= Decimal("3.5"):
                pedigree = PEDIGREE_UNCERTAINTY_FACTORS.get(
                    DQIScore.FAIR, Decimal("1.10")
                )
            elif score <= Decimal("4.5"):
                pedigree = PEDIGREE_UNCERTAINTY_FACTORS.get(
                    DQIScore.POOR, Decimal("1.20")
                )
            else:
                pedigree = PEDIGREE_UNCERTAINTY_FACTORS.get(
                    DQIScore.VERY_POOR, Decimal("1.50")
                )
            combined = self._q(combined * pedigree)

        return combined

    def _determine_ef_hierarchy_level(
        self,
        record: SupplierRecord,
    ) -> int:
        """Determine the EF hierarchy level for a supplier record.

        Per GHG Protocol Scope 3 Technical Guidance Section 1.4:
            1 = supplier_epd_verified
            2 = supplier_cdp_unverified
            3 = product_lca_ecoinvent
            4 = material_avg_ice_defra
            5 = industry_avg_physical
            6 = regional_eeio_exiobase
            7 = national_eeio_useeio
            8 = global_avg_eeio_fallback

        Args:
            record: The supplier record to classify.

        Returns:
            EF hierarchy level (1-8).
        """
        has_epd = (
            record.epd_number is not None
            and record.epd_number.strip()
        )
        is_verified = record.verification_status.lower().strip() in {
            "verified", "third_party_verified",
            "third-party-verified", "epd_verified", "iso_verified",
        }
        has_product_ef = (
            record.product_ef_kgco2e_per_unit is not None
            and record.product_ef_kgco2e_per_unit > ZERO
        )

        if has_epd and is_verified:
            return 1
        elif record.data_source == SupplierDataSource.CDP_SUPPLY_CHAIN:
            return 2
        elif has_product_ef and is_verified:
            return 1
        elif has_product_ef:
            return 2
        elif record.data_source in {
            SupplierDataSource.PACT_NETWORK,
            SupplierDataSource.DIRECT_MEASUREMENT,
        }:
            return 2
        elif record.data_source == SupplierDataSource.ECOVADIS:
            return 3
        elif record.data_source == SupplierDataSource.SUSTAINABILITY_REPORT:
            return 3
        else:
            return 4

    # ==================================================================
    # Public API: Aggregation
    # ==================================================================

    def aggregate_by_supplier(
        self,
        results: List[SupplierSpecificResult],
    ) -> Dict[str, Dict[str, Decimal]]:
        """Aggregate supplier-specific results by supplier.

        Groups results by supplier_id and computes totals for each
        supplier: total emissions (kgCO2e and tCO2e), item count,
        and average allocation factor.

        Args:
            results: List of SupplierSpecificResult objects.

        Returns:
            Dictionary keyed by supplier_id, each containing:
                - total_emissions_kgco2e (Decimal)
                - total_emissions_tco2e (Decimal)
                - item_count (Decimal)
                - avg_allocation_factor (Decimal)
                - max_emissions_kgco2e (Decimal)
                - min_emissions_kgco2e (Decimal)
        """
        if not results:
            return {}

        aggregation: Dict[str, Dict[str, Decimal]] = {}

        for result in results:
            sid = result.supplier_id

            if sid not in aggregation:
                aggregation[sid] = {
                    "total_emissions_kgco2e": ZERO,
                    "total_emissions_tco2e": ZERO,
                    "item_count": ZERO,
                    "sum_allocation_factor": ZERO,
                    "max_emissions_kgco2e": ZERO,
                    "min_emissions_kgco2e": Decimal("Infinity"),
                }

            agg = aggregation[sid]
            agg["total_emissions_kgco2e"] = self._q(
                agg["total_emissions_kgco2e"] + result.emissions_kgco2e
            )
            agg["total_emissions_tco2e"] = self._q(
                agg["total_emissions_tco2e"] + result.emissions_tco2e
            )
            agg["item_count"] = agg["item_count"] + ONE
            agg["sum_allocation_factor"] = self._q(
                agg["sum_allocation_factor"] + result.allocation_factor
            )

            if result.emissions_kgco2e > agg["max_emissions_kgco2e"]:
                agg["max_emissions_kgco2e"] = result.emissions_kgco2e
            if result.emissions_kgco2e < agg["min_emissions_kgco2e"]:
                agg["min_emissions_kgco2e"] = result.emissions_kgco2e

        # Compute averages and finalize
        for sid, agg in aggregation.items():
            item_count = agg["item_count"]
            if item_count > ZERO:
                agg["avg_allocation_factor"] = self._q(
                    agg["sum_allocation_factor"] / item_count
                )
            else:
                agg["avg_allocation_factor"] = ZERO

            # Clean up internal key
            del agg["sum_allocation_factor"]

            # Replace infinity sentinel
            if agg["min_emissions_kgco2e"] == Decimal("Infinity"):
                agg["min_emissions_kgco2e"] = ZERO

        logger.info(
            "Aggregated %d results across %d suppliers",
            len(results), len(aggregation),
        )

        return aggregation

    # ==================================================================
    # Public API: Coverage Analysis
    # ==================================================================

    def compute_coverage(
        self,
        results: List[SupplierSpecificResult],
        total_spend_usd: Decimal,
    ) -> Dict[str, Decimal]:
        """Compute supplier-specific method coverage statistics.

        Calculates the percentage of total procurement spend covered
        by the supplier-specific method, along with supplier count
        and emission totals.

        Coverage is evaluated against the GHG Protocol recommended
        thresholds:
            - High: >= 95% of spend covered by any method
            - Medium: >= 90%
            - Low: >= 80%
            - Minimal: < 80%

        For the supplier-specific method specifically, the SBTi
        recommends targeting 20%+ of spend covered by
        supplier-specific data.

        Args:
            results: List of SupplierSpecificResult objects.
            total_spend_usd: Total procurement spend in USD across
                all calculation methods.

        Returns:
            Dictionary with coverage metrics:
                - supplier_specific_emissions_kgco2e (Decimal)
                - supplier_specific_emissions_tco2e (Decimal)
                - supplier_specific_item_count (Decimal)
                - supplier_count (Decimal)
                - supplier_specific_coverage_pct (Decimal)
                - meets_sbti_target (Decimal): 1 if >= 20%, else 0
        """
        if not results or total_spend_usd <= ZERO:
            return {
                "supplier_specific_emissions_kgco2e": ZERO,
                "supplier_specific_emissions_tco2e": ZERO,
                "supplier_specific_item_count": ZERO,
                "supplier_count": ZERO,
                "supplier_specific_coverage_pct": ZERO,
                "meets_sbti_target": ZERO,
            }

        total_kgco2e = ZERO
        total_tco2e = ZERO
        suppliers: set = set()

        for result in results:
            total_kgco2e = self._q(total_kgco2e + result.emissions_kgco2e)
            total_tco2e = self._q(total_tco2e + result.emissions_tco2e)
            suppliers.add(result.supplier_id)

        # Coverage percentage is based on item count for supplier-specific
        # since spend is not directly available on SupplierSpecificResult.
        # A more accurate coverage requires integration with spend data.
        # Here we estimate coverage as fraction of total results.
        item_count = Decimal(str(len(results)))
        supplier_count = Decimal(str(len(suppliers)))

        # Compute spend coverage placeholder (requires external spend data)
        # For now, coverage is 0% unless caller integrates with spend data
        coverage_pct = ZERO

        sbti_target = getattr(
            self._config, "supplier_specific_target", Decimal("20")
        )
        meets_sbti = ONE if coverage_pct >= sbti_target else ZERO

        return {
            "supplier_specific_emissions_kgco2e": total_kgco2e,
            "supplier_specific_emissions_tco2e": total_tco2e,
            "supplier_specific_item_count": item_count,
            "supplier_count": supplier_count,
            "supplier_specific_coverage_pct": coverage_pct,
            "meets_sbti_target": meets_sbti,
        }

    # ==================================================================
    # Public API: Uncertainty Estimation
    # ==================================================================

    def estimate_uncertainty(
        self,
        result: SupplierSpecificResult,
    ) -> Dict[str, Decimal]:
        """Estimate uncertainty range for a supplier-specific result.

        Uses the GHG Protocol uncertainty guidance for the
        supplier-specific method, which has the lowest uncertainty
        range of all four methods: +/- 10-30%.

        The specific uncertainty depends on:
        - Data source (EPD/verified = lower, self-reported = higher)
        - Allocation method (product-level = lower, facility = higher)
        - Verification status (verified = lower)

        Args:
            result: The SupplierSpecificResult to estimate uncertainty for.

        Returns:
            Dictionary with uncertainty metrics:
                - lower_bound_kgco2e (Decimal): Emissions minus uncertainty
                - upper_bound_kgco2e (Decimal): Emissions plus uncertainty
                - lower_bound_tco2e (Decimal)
                - upper_bound_tco2e (Decimal)
                - uncertainty_pct_low (Decimal): Lower uncertainty %
                - uncertainty_pct_high (Decimal): Upper uncertainty %
                - uncertainty_method (str): Method used
                - confidence_level_pct (Decimal): Confidence level
        """
        # ---- Determine base uncertainty range ----
        base_low, base_high = UNCERTAINTY_RANGES.get(
            CalculationMethod.SUPPLIER_SPECIFIC,
            (Decimal("10"), Decimal("30")),
        )

        # ---- Adjust for data quality factors ----
        # Product-level EF: lower uncertainty
        has_product_ef = (
            result.product_ef_kgco2e_per_unit is not None
            and result.product_ef_kgco2e_per_unit > ZERO
        )

        # Verified data: lower uncertainty
        is_verified = result.verification_status.lower().strip() in {
            "verified", "third_party_verified",
            "third-party-verified", "epd_verified",
        }

        # EPD source: lowest uncertainty
        is_epd = result.data_source == SupplierDataSource.EPD

        # Facility-level with allocation: higher uncertainty
        is_facility_allocation = (
            result.supplier_total_tco2e is not None
            and result.allocation_factor < ONE
        )

        # ---- Apply adjustments ----
        uncertainty_low = base_low
        uncertainty_high = base_high

        if is_epd and is_verified and has_product_ef:
            # Best case: verified EPD with product-level EF
            uncertainty_low = self._q(base_low * Decimal("0.8"))
            uncertainty_high = self._q(base_high * Decimal("0.7"))
        elif has_product_ef and is_verified:
            # Verified product-level data
            uncertainty_low = self._q(base_low * Decimal("0.9"))
            uncertainty_high = self._q(base_high * Decimal("0.8"))
        elif has_product_ef:
            # Unverified product-level data
            pass  # Use base range
        elif is_facility_allocation and is_verified:
            # Verified facility allocation
            uncertainty_low = self._q(base_low * Decimal("1.1"))
            uncertainty_high = self._q(base_high * Decimal("1.1"))
        elif is_facility_allocation:
            # Unverified facility allocation
            uncertainty_low = self._q(base_low * Decimal("1.2"))
            uncertainty_high = self._q(base_high * Decimal("1.3"))

        # ---- Calculate bounds ----
        low_factor = ONE - (uncertainty_low / ONE_HUNDRED)
        high_factor = ONE + (uncertainty_high / ONE_HUNDRED)

        lower_kgco2e = self._q(result.emissions_kgco2e * low_factor)
        upper_kgco2e = self._q(result.emissions_kgco2e * high_factor)
        lower_tco2e = self._q(lower_kgco2e * _KG_TO_TONNES)
        upper_tco2e = self._q(upper_kgco2e * _KG_TO_TONNES)

        # Ensure non-negative lower bound
        if lower_kgco2e < ZERO:
            lower_kgco2e = ZERO
        if lower_tco2e < ZERO:
            lower_tco2e = ZERO

        logger.debug(
            "Uncertainty estimate: item=%s, range=[-%s%%, +%s%%], "
            "bounds=[%s, %s] kgCO2e",
            result.item_id, uncertainty_low, uncertainty_high,
            lower_kgco2e, upper_kgco2e,
        )

        return {
            "lower_bound_kgco2e": lower_kgco2e,
            "upper_bound_kgco2e": upper_kgco2e,
            "lower_bound_tco2e": lower_tco2e,
            "upper_bound_tco2e": upper_tco2e,
            "uncertainty_pct_low": self._q(uncertainty_low),
            "uncertainty_pct_high": self._q(uncertainty_high),
            "uncertainty_method": "ghg_protocol_supplier_specific",
            "confidence_level_pct": Decimal("95.0"),
        }

    # ==================================================================
    # Public API: Health Check
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the engine.

        Validates that the engine is properly initialized and all
        dependencies are accessible.

        Returns:
            Dictionary with health check results:
                - status (str): "healthy" or "degraded"
                - engine (str): Engine name
                - agent_id (str): Agent identifier
                - version (str): Agent version
                - calculation_count (int): Total calculations performed
                - batch_count (int): Total batches processed
                - error_count (int): Total errors encountered
                - config_loaded (bool): Whether config is loaded
                - metrics_available (bool): Whether metrics are available
                - provenance_available (bool): Whether provenance is available
                - timestamp (str): UTC timestamp
                - uptime_info (str): Information about engine uptime
        """
        status = "healthy"
        issues: List[str] = []

        # ---- Check config ----
        config_loaded = False
        try:
            config_loaded = self._config is not None and self._config.enabled
        except Exception as exc:
            issues.append(f"Config check failed: {exc}")
            status = "degraded"

        # ---- Check metrics ----
        metrics_available = False
        try:
            metrics_available = self._metrics is not None
        except Exception as exc:
            issues.append(f"Metrics check failed: {exc}")
            status = "degraded"

        # ---- Check provenance ----
        provenance_available = False
        try:
            provenance_available = self._provenance is not None
        except Exception as exc:
            issues.append(f"Provenance check failed: {exc}")
            status = "degraded"

        # ---- Compute error rate ----
        total_ops = self._calculation_count + self._batch_count
        error_rate = (
            self._error_count / total_ops * 100 if total_ops > 0 else 0.0
        )
        if error_rate > 10.0:
            issues.append(
                f"High error rate: {error_rate:.1f}%"
            )
            status = "degraded"

        return {
            "status": status,
            "engine": "SupplierSpecificCalculatorEngine",
            "agent_id": AGENT_ID,
            "version": VERSION,
            "calculation_count": self._calculation_count,
            "batch_count": self._batch_count,
            "error_count": self._error_count,
            "error_rate_pct": round(error_rate, 2),
            "config_loaded": config_loaded,
            "metrics_available": metrics_available,
            "provenance_available": provenance_available,
            "timestamp": utcnow().isoformat(),
            "issues": issues,
            "supported_data_sources": [
                ds.value for ds in SupplierDataSource
            ],
            "supported_allocation_methods": [
                am.value for am in AllocationMethod
            ],
            "engagement_levels": _ENGAGEMENT_LEVEL_NAMES,
        }

# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def get_supplier_specific_calculator() -> SupplierSpecificCalculatorEngine:
    """Return the singleton SupplierSpecificCalculatorEngine instance.

    Convenience factory function for callers that prefer a functional
    style over direct class instantiation.

    Returns:
        The singleton engine instance.

    Example:
        >>> engine = get_supplier_specific_calculator()
        >>> result = engine.calculate_product_level(...)
    """
    return SupplierSpecificCalculatorEngine()
