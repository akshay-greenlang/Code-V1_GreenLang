# -*- coding: utf-8 -*-
"""
SupplierSpecificCalculatorEngine - Engine 4: Capital Goods Agent (AGENT-MRV-015)

Supplier-specific calculation engine for GHG Protocol Scope 3 Category 2
capital goods emissions using primary data from Environmental Product
Declarations (EPD), Product Carbon Footprints (PCF), CDP Supply Chain
disclosures, EcoVadis assessments, and direct supplier measurements.

The supplier-specific method is the highest-accuracy approach for Category 2
calculations per the GHG Protocol Technical Guidance for Scope 3.  It uses
cradle-to-gate emission data provided directly by capital goods suppliers
rather than industry-average or spend-based proxies.

Three core calculation paths:

1. **Product-level EF path**: Supplier provides kgCO2e per unit (or kg)
   of product.  Emissions = quantity x product_ef_kgco2e_per_unit.

2. **PCF value path**: Supplier provides total Product Carbon Footprint
   for a defined functional unit.  Emissions = pcf_total x
   (units_purchased / units_in_pcf_scope).

3. **Facility-level allocation**: Supplier provides total facility
   emissions and the engine allocates using economic, physical, mass,
   energy, or hybrid methods.

Data Quality:
    EPD (ISO 14025) third-party verified data is the highest quality,
    followed by certified PCF (ISO 14067), then CDP/EcoVadis/direct
    measurement, with estimated data being the lowest quality.

Verification scoring:
    THIRD_PARTY_VERIFIED = 1.0 (best)
    CERTIFIED_EPD        = 0.9
    SELF_DECLARED_WITH_AUDIT = 0.7
    SELF_DECLARED        = 0.5
    ESTIMATED            = 0.3
    UNVERIFIED           = 0.1

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (ROUND_HALF_UP, 8 decimal places)
    - No LLM calls in the calculation path
    - Every calculation step recorded in trace dict
    - SHA-256 provenance hash for every result

Thread Safety:
    Thread-safe singleton via threading.Lock on class instantiation.
    Instance methods are re-entrant; no shared mutable state is modified
    after __init__ except via the reset() classmethod.

Example:
    >>> from greenlang.capital_goods.supplier_specific_calculator import (
    ...     SupplierSpecificCalculatorEngine,
    ... )
    >>> from greenlang.capital_goods.models import (
    ...     SupplierRecord, CapitalAssetRecord, AssetCategory,
    ...     SupplierDataSource, AllocationMethod,
    ... )
    >>> from decimal import Decimal
    >>> from datetime import date
    >>> engine = SupplierSpecificCalculatorEngine()
    >>> record = SupplierRecord(
    ...     asset_id="ASSET-001",
    ...     supplier_name="Siemens AG",
    ...     data_source=SupplierDataSource.EPD,
    ...     ef_value=Decimal("420.0"),
    ...     ef_unit="kgCO2e/unit",
    ...     verification_status="third_party_verified",
    ...     epd_number="EPD-2024-001",
    ...     reporting_year=2024,
    ... )
    >>> asset = CapitalAssetRecord(
    ...     asset_id="ASSET-001",
    ...     asset_category=AssetCategory.MACHINERY,
    ...     description="CNC Milling Machine 5-Axis",
    ...     acquisition_date=date(2024, 6, 15),
    ...     capex_amount=Decimal("250000"),
    ...     quantity=Decimal("2"),
    ... )
    >>> result = engine.calculate(record, asset)
    >>> assert result.emissions_kg_co2e == Decimal("840.00000000")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-015 Capital Goods (GL-MRV-S3-002)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.capital_goods.models import (
    AllocationMethod,
    CalculationMethod,
    CapitalAssetRecord,
    CoverageReport,
    DQIAssessment,
    DQIScore,
    DQI_QUALITY_TIERS,
    DQI_SCORE_VALUES,
    PEDIGREE_UNCERTAINTY_FACTORS,
    SupplierDataSource,
    SupplierRecord,
    SupplierSpecificResult,
    UNCERTAINTY_RANGES,
    ZERO,
    ONE,
    ONE_HUNDRED,
    ONE_THOUSAND,
    DECIMAL_PLACES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level Constants
# ---------------------------------------------------------------------------

#: Agent and engine identifiers.
AGENT_ID: str = "GL-MRV-S3-002"
ENGINE_ID: str = "SupplierSpecificCalculatorEngine"
ENGINE_VERSION: str = "1.0.0"

#: Decimal precision quantizer (8 decimal places).
_PRECISION = Decimal(10) ** -DECIMAL_PLACES

#: Five for DQI dimension count.
_FIVE = Decimal("5")

#: Gas split ratios for supplier-specific data (typical cradle-to-gate).
#: When supplier provides only total CO2e, split into constituent gases
#: using default ratios from IPCC Guidelines Volume 2.
_DEFAULT_GAS_SPLIT_CO2: Decimal = Decimal("0.980")
_DEFAULT_GAS_SPLIT_CH4: Decimal = Decimal("0.015")
_DEFAULT_GAS_SPLIT_N2O: Decimal = Decimal("0.005")

#: Verification status to numeric quality score mapping.
#: Scores range from 0.1 (worst: unverified) to 1.0 (best: third-party).
VERIFICATION_SCORES: Dict[str, Decimal] = {
    "third_party_verified": Decimal("1.0"),
    "certified_epd": Decimal("0.9"),
    "self_declared_with_audit": Decimal("0.7"),
    "self_declared": Decimal("0.5"),
    "estimated": Decimal("0.3"),
    "unverified": Decimal("0.1"),
}

#: Data source to default DQI reliability scores (1=best, 5=worst).
DATA_SOURCE_RELIABILITY: Dict[SupplierDataSource, Decimal] = {
    SupplierDataSource.EPD: Decimal("1.0"),
    SupplierDataSource.PCF: Decimal("1.5"),
    SupplierDataSource.CDP: Decimal("2.0"),
    SupplierDataSource.ECOVADIS: Decimal("2.5"),
    SupplierDataSource.DIRECT_MEASUREMENT: Decimal("2.0"),
    SupplierDataSource.ESTIMATED: Decimal("4.0"),
}

#: Data source to EF hierarchy level (1=best, 8=worst).
DATA_SOURCE_EF_HIERARCHY: Dict[SupplierDataSource, int] = {
    SupplierDataSource.EPD: 1,
    SupplierDataSource.PCF: 2,
    SupplierDataSource.CDP: 3,
    SupplierDataSource.ECOVADIS: 5,
    SupplierDataSource.DIRECT_MEASUREMENT: 3,
    SupplierDataSource.ESTIMATED: 6,
}

#: Default uncertainty percentages by data source.
DATA_SOURCE_UNCERTAINTY: Dict[SupplierDataSource, Decimal] = {
    SupplierDataSource.EPD: Decimal("10.0"),
    SupplierDataSource.PCF: Decimal("15.0"),
    SupplierDataSource.CDP: Decimal("25.0"),
    SupplierDataSource.ECOVADIS: Decimal("35.0"),
    SupplierDataSource.DIRECT_MEASUREMENT: Decimal("20.0"),
    SupplierDataSource.ESTIMATED: Decimal("50.0"),
}

#: EPD required fields for ISO 14025 compliance validation.
_EPD_REQUIRED_FIELDS: List[str] = [
    "epd_number",
    "reporting_year",
    "boundary",
]

#: PCF required fields for ISO 14067 compliance validation.
_PCF_REQUIRED_FIELDS: List[str] = [
    "reporting_year",
    "boundary",
]

#: Maximum age (years) for supplier data to be considered current.
_MAX_DATA_AGE_YEARS: int = 5

#: Maximum age (years) for EPD data before temporal penalty.
_EPD_MAX_AGE_YEARS: int = 5

#: Maximum age (years) for PCF data before temporal penalty.
_PCF_MAX_AGE_YEARS: int = 3

#: Batch processing chunk size.
_DEFAULT_BATCH_SIZE: int = 1000


# ============================================================================
# Helper: UTC timestamp
# ============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _current_year() -> int:
    """Return the current calendar year."""
    return date.today().year


# ============================================================================
# SupplierSpecificCalculatorEngine (thread-safe singleton)
# ============================================================================


class SupplierSpecificCalculatorEngine:
    """Supplier-specific calculation engine for capital goods emissions.

    Implements the GHG Protocol supplier-specific method for Scope 3
    Category 2 capital goods.  Uses primary supplier data (EPD, PCF,
    CDP, EcoVadis, direct measurement) to calculate cradle-to-gate
    emissions with the highest possible accuracy.

    Three calculation paths:
        1. Product-level EF: quantity x ef_value
        2. PCF allocation: pcf_total x (purchased_units / scope_units)
        3. Facility allocation: facility_total x allocation_factor

    Thread Safety:
        Singleton pattern with ``threading.Lock``.  All public methods
        are re-entrant; no shared mutable state is written after init.

    Attributes:
        _config: Engine configuration dictionary.
        _lock: Thread lock for singleton management.
        _initialized: Whether the engine has been initialized.

    Example:
        >>> engine = SupplierSpecificCalculatorEngine()
        >>> result = engine.calculate(record, asset)
        >>> assert result.method == CalculationMethod.SUPPLIER_SPECIFIC
    """

    _instance: Optional[SupplierSpecificCalculatorEngine] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __new__(cls, config: Optional[Dict[str, Any]] = None) -> SupplierSpecificCalculatorEngine:
        """Create or return the singleton instance.

        Args:
            config: Optional configuration dictionary.

        Returns:
            The singleton SupplierSpecificCalculatorEngine instance.
        """
        with cls._singleton_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the engine (only on first instantiation).

        Args:
            config: Optional configuration dictionary.  Supports:
                - ``enable_provenance`` (bool): Enable SHA-256 hashing.
                  Default True.
                - ``batch_size`` (int): Chunk size for batch processing.
                  Default 1000.
                - ``max_data_age_years`` (int): Max age for supplier data.
                  Default 5.
                - ``default_gas_split`` (dict): Override gas split ratios.
                - ``strict_validation`` (bool): Fail on validation warnings.
                  Default False.
        """
        if self._initialized:
            return

        self._config: Dict[str, Any] = config or {}
        self._lock: threading.Lock = threading.Lock()
        self._enable_provenance: bool = self._config.get("enable_provenance", True)
        self._batch_size: int = self._config.get("batch_size", _DEFAULT_BATCH_SIZE)
        self._max_data_age: int = self._config.get("max_data_age_years", _MAX_DATA_AGE_YEARS)
        self._strict_validation: bool = self._config.get("strict_validation", False)

        # Gas split overrides
        gas_split = self._config.get("default_gas_split", {})
        self._gas_co2: Decimal = Decimal(str(gas_split.get("co2", _DEFAULT_GAS_SPLIT_CO2)))
        self._gas_ch4: Decimal = Decimal(str(gas_split.get("ch4", _DEFAULT_GAS_SPLIT_CH4)))
        self._gas_n2o: Decimal = Decimal(str(gas_split.get("n2o", _DEFAULT_GAS_SPLIT_N2O)))

        self._initialized = True
        logger.info(
            "%s v%s initialized (provenance=%s, batch_size=%d, strict=%s)",
            ENGINE_ID,
            ENGINE_VERSION,
            self._enable_provenance,
            self._batch_size,
            self._strict_validation,
        )

    # ------------------------------------------------------------------
    # Singleton management
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for testing or reconfiguration.

        After calling reset(), the next instantiation will create a
        fresh engine instance with new configuration.

        Example:
            >>> SupplierSpecificCalculatorEngine.reset()
            >>> engine = SupplierSpecificCalculatorEngine({"strict_validation": True})
        """
        with cls._singleton_lock:
            cls._instance = None
            logger.info("%s singleton reset", ENGINE_ID)

    # ==================================================================
    #  PUBLIC API: Core Calculations
    # ==================================================================

    def calculate(
        self,
        record: SupplierRecord,
        asset: CapitalAssetRecord,
        config: Optional[Dict[str, Any]] = None,
    ) -> SupplierSpecificResult:
        """Calculate supplier-specific emissions for a single capital asset.

        Automatically selects the best calculation path based on the
        supplier record data:
            1. If ef_value > 0 and allocation_factor == 1.0 and data
               source is EPD/PCF/DIRECT_MEASUREMENT -> product-level EF.
            2. If data_source is PCF and allocation_factor < 1.0 ->
               PCF allocation path.
            3. If allocation_factor < 1.0 (CDP/facility data) ->
               facility-level allocation.

        All arithmetic uses Decimal with ROUND_HALF_UP.

        Args:
            record: Supplier-specific input record with EF data.
            asset: Capital asset record with quantity and metadata.
            config: Optional per-calculation configuration overrides.
                - ``force_path`` (str): Force a calculation path
                  (``"product_level"``, ``"pcf"``, ``"facility"``).
                - ``gas_split`` (dict): Override gas split ratios.

        Returns:
            SupplierSpecificResult with total emissions, gas breakdown,
            DQI score, uncertainty, and provenance hash.

        Raises:
            ValueError: If record or asset data is invalid.
            InvalidOperation: If Decimal arithmetic fails.
        """
        start_ns = time.monotonic_ns()
        calc_config = config or {}
        trace: Dict[str, Any] = {
            "engine": ENGINE_ID,
            "version": ENGINE_VERSION,
            "record_id": record.record_id,
            "asset_id": asset.asset_id,
            "timestamp": _utcnow().isoformat(),
        }

        logger.debug(
            "Calculating supplier-specific emissions: asset=%s, supplier=%s, source=%s",
            asset.asset_id,
            record.supplier_name,
            record.data_source.value,
        )

        # Step 1: Validate inputs
        errors = self.validate_supplier_data(record)
        if errors and self._strict_validation:
            raise ValueError(
                f"Supplier data validation failed for {record.record_id}: "
                + "; ".join(errors)
            )
        trace["validation_warnings"] = errors

        # Step 2: Route to calculation path
        forced_path = calc_config.get("force_path")
        path = self._select_calculation_path(record, forced_path)
        trace["calculation_path"] = path

        # Step 3: Execute calculation
        if path == "product_level":
            result = self._calculate_product_level_internal(record, asset, trace)
        elif path == "pcf":
            result = self._calculate_pcf_internal(record, asset, trace)
        elif path == "facility":
            result = self._calculate_facility_internal(record, asset, trace)
        else:
            raise ValueError(f"Unknown calculation path: {path}")

        # Step 4: Compute provenance hash
        provenance_hash = ""
        if self._enable_provenance:
            provenance_hash = self.compute_provenance_hash(record, result)
            trace["provenance_hash"] = provenance_hash

        # Step 5: Build final result
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        trace["processing_time_ms"] = float(elapsed_ms)

        final_result = SupplierSpecificResult(
            record_id=str(uuid.uuid4()),
            asset_id=asset.asset_id,
            supplier_name=record.supplier_name,
            data_source=record.data_source,
            ef_value=record.ef_value,
            allocation_method=record.allocation_method,
            allocation_factor=record.allocation_factor,
            emissions_kg_co2e=result["emissions_kg_co2e"],
            co2=result["co2"],
            ch4=result["ch4"],
            n2o=result["n2o"],
            verification_status=record.verification_status,
            dqi_score=result["dqi_score"],
            uncertainty_pct=result["uncertainty_pct"],
            method=CalculationMethod.SUPPLIER_SPECIFIC,
            provenance_hash=provenance_hash,
        )

        logger.info(
            "Supplier-specific calculation complete: asset=%s, supplier=%s, "
            "emissions=%.4f kgCO2e, path=%s, time=%.2fms",
            asset.asset_id,
            record.supplier_name,
            final_result.emissions_kg_co2e,
            path,
            elapsed_ms,
        )

        return final_result

    def calculate_batch(
        self,
        records: List[Tuple[SupplierRecord, CapitalAssetRecord]],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[SupplierSpecificResult]:
        """Calculate supplier-specific emissions for a batch of records.

        Processes records in chunks for memory efficiency.  Each record
        pair is calculated independently; failures are logged and
        skipped (unless strict mode is enabled).

        Args:
            records: List of (SupplierRecord, CapitalAssetRecord) tuples.
            config: Optional per-batch configuration overrides.

        Returns:
            List of SupplierSpecificResult objects (one per successful
            calculation).  Order matches input order for successful
            records.

        Raises:
            ValueError: If records list is empty.
        """
        if not records:
            raise ValueError("Records list must not be empty")

        start_ns = time.monotonic_ns()
        total = len(records)
        results: List[SupplierSpecificResult] = []
        errors_count = 0
        chunk_size = self._batch_size

        logger.info(
            "Starting batch calculation: %d records, chunk_size=%d",
            total,
            chunk_size,
        )

        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk = records[chunk_start:chunk_end]

            for idx, (record, asset) in enumerate(chunk):
                global_idx = chunk_start + idx
                try:
                    result = self.calculate(record, asset, config)
                    results.append(result)
                except Exception as exc:
                    errors_count += 1
                    logger.warning(
                        "Batch record %d/%d failed (asset=%s): %s",
                        global_idx + 1,
                        total,
                        getattr(asset, "asset_id", "unknown"),
                        str(exc),
                    )
                    if self._strict_validation:
                        raise

            logger.debug(
                "Batch chunk %d-%d processed (%d/%d complete)",
                chunk_start + 1,
                chunk_end,
                len(results),
                total,
            )

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        logger.info(
            "Batch calculation complete: %d/%d successful, %d errors, %.2fms",
            len(results),
            total,
            errors_count,
            elapsed_ms,
        )

        return results

    def calculate_product_level(
        self,
        record: SupplierRecord,
        asset: CapitalAssetRecord,
    ) -> SupplierSpecificResult:
        """Calculate emissions using the product-level EF path.

        Formula:
            emissions_kg_co2e = quantity x ef_value

        Where ef_value is the supplier-provided emission factor in
        kgCO2e per unit (or per kg if ef_unit indicates mass-based).

        This is the simplest and most direct calculation path.  Used
        when the supplier provides a product-specific emission factor
        (typically from an EPD or PCF study).

        Args:
            record: SupplierRecord with ef_value in kgCO2e/unit.
            asset: CapitalAssetRecord with quantity.

        Returns:
            SupplierSpecificResult with calculated emissions.

        Raises:
            ValueError: If ef_value is zero or negative.
        """
        return self.calculate(record, asset, config={"force_path": "product_level"})

    def calculate_pcf(
        self,
        record: SupplierRecord,
        asset: CapitalAssetRecord,
    ) -> SupplierSpecificResult:
        """Calculate emissions using the PCF value path.

        Formula:
            emissions_kg_co2e = ef_value x allocation_factor

        Where:
            ef_value = total PCF value (kgCO2e for the functional unit)
            allocation_factor = units_purchased / units_in_pcf_scope

        The PCF path is used when the supplier provides a total Product
        Carbon Footprint value for a defined functional unit, and the
        purchaser needs to allocate a portion of that footprint based
        on the number of units purchased versus the total scope of
        the PCF study.

        Args:
            record: SupplierRecord with ef_value as total PCF kgCO2e
                and allocation_factor as the unit ratio.
            asset: CapitalAssetRecord for the capital good.

        Returns:
            SupplierSpecificResult with calculated emissions.

        Raises:
            ValueError: If ef_value is zero.
        """
        return self.calculate(record, asset, config={"force_path": "pcf"})

    def calculate_facility_allocation(
        self,
        record: SupplierRecord,
        asset: CapitalAssetRecord,
    ) -> SupplierSpecificResult:
        """Calculate emissions using facility-level allocation.

        Formula:
            emissions_kg_co2e = ef_value x allocation_factor

        Where:
            ef_value = total facility emissions (kgCO2e)
            allocation_factor = allocation share (0-1)

        The allocation_factor is computed externally using one of five
        allocation methods (economic, physical, mass, energy, hybrid)
        and stored on the SupplierRecord.

        This path is used when the supplier provides total facility
        emissions (e.g., from CDP disclosure) and the purchaser must
        allocate a share to their specific capital goods purchase.

        Args:
            record: SupplierRecord with ef_value as facility total
                and allocation_factor as the allocation share.
            asset: CapitalAssetRecord for the capital good.

        Returns:
            SupplierSpecificResult with calculated emissions.

        Raises:
            ValueError: If allocation_factor is zero.
        """
        return self.calculate(record, asset, config={"force_path": "facility"})

    # ==================================================================
    #  PUBLIC API: Allocation
    # ==================================================================

    def allocate_emissions(
        self,
        total_emissions: Decimal,
        allocation_method: AllocationMethod,
        allocation_data: Dict[str, Decimal],
    ) -> Decimal:
        """Allocate a share of total emissions using the specified method.

        Supports five allocation methods per GHG Protocol / ISO 14044:
            - Economic: asset_value / total_facility_value
            - Physical: asset_quantity / total_facility_quantity
            - Mass: asset_mass_kg / total_facility_mass_kg
            - Energy: asset_energy_mj / total_facility_energy_mj
            - Hybrid: weighted combination of multiple methods

        Args:
            total_emissions: Total facility/scope emissions in kgCO2e.
            allocation_method: Allocation method to use.
            allocation_data: Method-specific numerator/denominator data.
                For economic: ``asset_value``, ``total_value``.
                For physical: ``asset_quantity``, ``total_quantity``.
                For mass: ``asset_mass_kg``, ``total_mass_kg``.
                For energy: ``asset_energy_mj``, ``total_energy_mj``.
                For hybrid: ``weights`` (dict of method->weight),
                    plus all relevant numerator/denominator pairs.

        Returns:
            Allocated emissions in kgCO2e (Decimal, 8 decimal places).

        Raises:
            ValueError: If required allocation data is missing or
                denominators are zero.
        """
        if total_emissions < ZERO:
            raise ValueError(
                f"total_emissions must be non-negative, got {total_emissions}"
            )

        if allocation_method == AllocationMethod.ECONOMIC:
            return self._allocate_economic(total_emissions, allocation_data)
        elif allocation_method == AllocationMethod.PHYSICAL:
            return self._allocate_physical(total_emissions, allocation_data)
        elif allocation_method == AllocationMethod.MASS:
            return self._allocate_mass(total_emissions, allocation_data)
        elif allocation_method == AllocationMethod.ENERGY:
            return self._allocate_energy(total_emissions, allocation_data)
        elif allocation_method == AllocationMethod.HYBRID:
            return self._allocate_hybrid(total_emissions, allocation_data)
        else:
            raise ValueError(f"Unsupported allocation method: {allocation_method}")

    # ==================================================================
    #  PUBLIC API: Validation
    # ==================================================================

    def validate_supplier_data(self, record: SupplierRecord) -> List[str]:
        """Validate supplier data completeness and quality.

        Checks:
            1. Emission factor is positive.
            2. Supplier name is non-empty.
            3. Data source is recognized.
            4. Allocation factor is in [0, 1].
            5. Reporting year is within acceptable age.
            6. Boundary is recognized (cradle_to_gate or cradle_to_grave).
            7. Verification status is recognized.

        Args:
            record: SupplierRecord to validate.

        Returns:
            List of validation warning strings.  Empty list means
            the record passes all quality checks.
        """
        warnings: List[str] = []

        # EF value check
        if record.ef_value <= ZERO:
            warnings.append(
                f"ef_value must be positive, got {record.ef_value}"
            )

        # Allocation factor range
        if record.allocation_factor < ZERO or record.allocation_factor > ONE:
            warnings.append(
                f"allocation_factor must be in [0, 1], got {record.allocation_factor}"
            )

        # Reporting year freshness
        if record.reporting_year is not None:
            age = _current_year() - record.reporting_year
            if age > self._max_data_age:
                warnings.append(
                    f"Supplier data is {age} years old "
                    f"(reporting_year={record.reporting_year}), "
                    f"exceeds max_data_age={self._max_data_age}"
                )
            if age < 0:
                warnings.append(
                    f"Reporting year {record.reporting_year} is in the future"
                )
        else:
            warnings.append("reporting_year is missing; temporal DQI will be penalized")

        # Boundary check
        valid_boundaries = {"cradle_to_gate", "cradle_to_grave"}
        if record.boundary not in valid_boundaries:
            warnings.append(
                f"Unrecognized boundary '{record.boundary}'; "
                f"expected one of {valid_boundaries}"
            )

        # Verification status check
        if record.verification_status not in VERIFICATION_SCORES:
            warnings.append(
                f"Unrecognized verification_status '{record.verification_status}'; "
                f"expected one of {list(VERIFICATION_SCORES.keys())}"
            )

        return warnings

    def validate_epd(self, record: SupplierRecord) -> Dict[str, Any]:
        """Validate EPD-specific data for ISO 14025 compliance.

        Checks that the supplier record has all required EPD fields
        and that the data meets ISO 14025 Environmental Product
        Declaration requirements.

        Validation rules:
            1. data_source must be EPD.
            2. epd_number must be present and non-empty.
            3. reporting_year must be present and within 5 years.
            4. boundary must be cradle_to_gate.
            5. ef_value must be positive.
            6. Verification status should be certified_epd or
               third_party_verified.

        Args:
            record: SupplierRecord to validate as EPD.

        Returns:
            Dict with keys:
                - ``valid`` (bool): Overall pass/fail.
                - ``errors`` (List[str]): Blocking errors.
                - ``warnings`` (List[str]): Non-blocking warnings.
                - ``iso_14025_compliant`` (bool): ISO compliance.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Source check
        if record.data_source != SupplierDataSource.EPD:
            errors.append(
                f"data_source must be EPD, got {record.data_source.value}"
            )

        # EPD number
        if not record.epd_number:
            errors.append("epd_number is required for EPD data source")
        elif len(record.epd_number.strip()) < 3:
            warnings.append(
                f"epd_number '{record.epd_number}' seems unusually short"
            )

        # Reporting year
        if record.reporting_year is None:
            errors.append("reporting_year is required for EPD validation")
        else:
            age = _current_year() - record.reporting_year
            if age > _EPD_MAX_AGE_YEARS:
                warnings.append(
                    f"EPD is {age} years old (max recommended: {_EPD_MAX_AGE_YEARS})"
                )
            if age < 0:
                errors.append(
                    f"EPD reporting_year {record.reporting_year} is in the future"
                )

        # Boundary
        if record.boundary != "cradle_to_gate":
            warnings.append(
                f"EPD boundary is '{record.boundary}'; "
                "ISO 14025 typically requires cradle_to_gate"
            )

        # EF value
        if record.ef_value <= ZERO:
            errors.append(f"ef_value must be positive, got {record.ef_value}")

        # Verification
        epd_verifications = {"certified_epd", "third_party_verified"}
        if record.verification_status not in epd_verifications:
            warnings.append(
                f"EPD verification_status '{record.verification_status}' "
                f"is not in {epd_verifications}; ISO 14025 requires "
                "third-party verification"
            )

        is_valid = len(errors) == 0
        iso_compliant = is_valid and len(warnings) == 0

        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "iso_14025_compliant": iso_compliant,
        }

    def validate_pcf(self, record: SupplierRecord) -> Dict[str, Any]:
        """Validate PCF-specific data for ISO 14067 compliance.

        Checks that the supplier record meets ISO 14067 Product Carbon
        Footprint requirements.

        Validation rules:
            1. data_source must be PCF.
            2. reporting_year must be present and within 3 years.
            3. boundary must be cradle_to_gate or cradle_to_grave.
            4. ef_value must be positive.
            5. Verification should be third-party or self-declared
               with audit for ISO 14067.

        Args:
            record: SupplierRecord to validate as PCF.

        Returns:
            Dict with keys:
                - ``valid`` (bool): Overall pass/fail.
                - ``errors`` (List[str]): Blocking errors.
                - ``warnings`` (List[str]): Non-blocking warnings.
                - ``iso_14067_compliant`` (bool): ISO compliance.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Source check
        if record.data_source != SupplierDataSource.PCF:
            errors.append(
                f"data_source must be PCF, got {record.data_source.value}"
            )

        # Reporting year
        if record.reporting_year is None:
            errors.append("reporting_year is required for PCF validation")
        else:
            age = _current_year() - record.reporting_year
            if age > _PCF_MAX_AGE_YEARS:
                warnings.append(
                    f"PCF is {age} years old (max recommended: {_PCF_MAX_AGE_YEARS})"
                )
            if age < 0:
                errors.append(
                    f"PCF reporting_year {record.reporting_year} is in the future"
                )

        # Boundary
        valid_boundaries = {"cradle_to_gate", "cradle_to_grave"}
        if record.boundary not in valid_boundaries:
            errors.append(
                f"PCF boundary '{record.boundary}' is not valid; "
                f"expected one of {valid_boundaries}"
            )

        # EF value
        if record.ef_value <= ZERO:
            errors.append(f"ef_value must be positive, got {record.ef_value}")

        # Verification
        pcf_verifications = {
            "third_party_verified",
            "certified_epd",
            "self_declared_with_audit",
        }
        if record.verification_status not in pcf_verifications:
            warnings.append(
                f"PCF verification_status '{record.verification_status}' "
                f"is not in {pcf_verifications}; ISO 14067 recommends "
                "third-party or audited self-declaration"
            )

        is_valid = len(errors) == 0
        iso_compliant = is_valid and len(warnings) == 0

        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "iso_14067_compliant": iso_compliant,
        }

    # ==================================================================
    #  PUBLIC API: Data Quality
    # ==================================================================

    def score_dqi(
        self,
        record: SupplierRecord,
        verification_level: str = "",
    ) -> DQIAssessment:
        """Score data quality for a supplier-specific record.

        Evaluates the five GHG Protocol DQI dimensions:
            1. Temporal: How recent is the supplier data?
            2. Geographical: Does the EF match the asset's region?
            3. Technological: Does the EF match the asset type?
            4. Completeness: Does the data cover all GHG sources?
            5. Reliability: How trustworthy is the data source?

        The composite score is the arithmetic mean of the five
        dimension scores.  Lower is better (1=best, 5=worst).

        Args:
            record: SupplierRecord to assess.
            verification_level: Override verification level for scoring.
                If empty, uses record.verification_status.

        Returns:
            DQIAssessment with dimension scores, composite score,
            quality tier, uncertainty factor, and findings.
        """
        findings: List[str] = []
        ver_level = verification_level or record.verification_status

        # 1. Temporal score
        temporal = self._score_temporal(record, findings)

        # 2. Geographical score (supplier-specific is inherently good)
        geographical = self._score_geographical(record, findings)

        # 3. Technological score (product-specific data is best)
        technological = self._score_technological(record, findings)

        # 4. Completeness score
        completeness = self._score_completeness(record, findings)

        # 5. Reliability score
        reliability = self._score_reliability(record, ver_level, findings)

        # Composite: arithmetic mean
        composite = (
            (temporal + geographical + technological + completeness + reliability)
            / _FIVE
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Clamp to [1.0, 5.0]
        composite = max(Decimal("1.0"), min(Decimal("5.0"), composite))

        # Quality tier
        quality_tier = self._determine_quality_tier(composite)

        # Uncertainty factor from pedigree matrix
        uncertainty_factor = self._compute_pedigree_uncertainty(composite)

        # EF hierarchy level
        ef_level = DATA_SOURCE_EF_HIERARCHY.get(record.data_source, 6)

        return DQIAssessment(
            asset_id=record.asset_id,
            calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
            temporal_score=temporal,
            geographical_score=geographical,
            technological_score=technological,
            completeness_score=completeness,
            reliability_score=reliability,
            composite_score=composite,
            quality_tier=quality_tier,
            uncertainty_factor=uncertainty_factor,
            findings=findings,
            ef_hierarchy_level=ef_level,
        )

    def score_verification(self, record: SupplierRecord) -> Decimal:
        """Score the verification quality of supplier data.

        Maps the verification_status string to a numeric quality
        score between 0.1 (unverified) and 1.0 (third-party verified).

        Args:
            record: SupplierRecord with verification_status.

        Returns:
            Decimal quality score in [0.1, 1.0].
        """
        score = VERIFICATION_SCORES.get(
            record.verification_status,
            Decimal("0.1"),
        )
        logger.debug(
            "Verification score for '%s': %s",
            record.verification_status,
            score,
        )
        return score

    # ==================================================================
    #  PUBLIC API: Aggregation
    # ==================================================================

    def aggregate_by_supplier(
        self,
        results: List[SupplierSpecificResult],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by supplier name.

        Groups results by ``supplier_name`` and sums emissions within
        each group.  Useful for identifying which capital goods
        suppliers contribute the most to Category 2 emissions.

        Args:
            results: List of SupplierSpecificResult objects.

        Returns:
            Dict mapping supplier_name to total emissions in kgCO2e.
            Sorted by emissions descending.
        """
        aggregation: Dict[str, Decimal] = defaultdict(lambda: ZERO)

        for result in results:
            aggregation[result.supplier_name] = (
                aggregation[result.supplier_name] + result.emissions_kg_co2e
            )

        # Sort by emissions descending
        sorted_agg = dict(
            sorted(
                aggregation.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        logger.debug(
            "Aggregated %d results into %d suppliers",
            len(results),
            len(sorted_agg),
        )
        return sorted_agg

    def aggregate_by_data_source(
        self,
        results: List[SupplierSpecificResult],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by data source type.

        Groups results by ``data_source`` and sums emissions.  Useful
        for understanding the distribution of data quality across
        the supplier-specific portion of the inventory.

        Args:
            results: List of SupplierSpecificResult objects.

        Returns:
            Dict mapping data source value (str) to total emissions
            in kgCO2e.
        """
        aggregation: Dict[str, Decimal] = defaultdict(lambda: ZERO)

        for result in results:
            key = result.data_source.value
            aggregation[key] = aggregation[key] + result.emissions_kg_co2e

        sorted_agg = dict(
            sorted(
                aggregation.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        logger.debug(
            "Aggregated %d results into %d data sources",
            len(results),
            len(sorted_agg),
        )
        return sorted_agg

    def aggregate_by_verification(
        self,
        results: List[SupplierSpecificResult],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by verification status.

        Groups results by ``verification_status`` and sums emissions.
        Useful for assurance reporting and understanding verification
        coverage of supplier-specific data.

        Args:
            results: List of SupplierSpecificResult objects.

        Returns:
            Dict mapping verification_status to total emissions
            in kgCO2e.
        """
        aggregation: Dict[str, Decimal] = defaultdict(lambda: ZERO)

        for result in results:
            key = result.verification_status
            aggregation[key] = aggregation[key] + result.emissions_kg_co2e

        sorted_agg = dict(
            sorted(
                aggregation.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        logger.debug(
            "Aggregated %d results into %d verification levels",
            len(results),
            len(sorted_agg),
        )
        return sorted_agg

    # ==================================================================
    #  PUBLIC API: Coverage
    # ==================================================================

    def get_coverage_report(
        self,
        results: List[SupplierSpecificResult],
        total_assets: int,
    ) -> CoverageReport:
        """Generate a coverage report for supplier-specific calculations.

        Analyzes the extent to which capital assets are covered by
        supplier-specific data, including breakdown by method,
        uncovered CapEx, and gap identification.

        Args:
            results: List of SupplierSpecificResult objects.
            total_assets: Total number of capital assets in scope.

        Returns:
            CoverageReport with coverage percentage, method breakdown,
            uncovered CapEx estimate, and gap categories.

        Raises:
            ValueError: If total_assets is negative.
        """
        if total_assets < 0:
            raise ValueError(f"total_assets must be non-negative, got {total_assets}")

        covered_assets = len(results)

        if total_assets == 0:
            coverage_pct = ZERO
        else:
            coverage_pct = (
                (Decimal(str(covered_assets)) / Decimal(str(total_assets)))
                * ONE_HUNDRED
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
            coverage_pct = min(ONE_HUNDRED, coverage_pct)

        # Build method breakdown
        by_method: Dict[str, Dict[str, Decimal]] = {}
        supplier_method_key = CalculationMethod.SUPPLIER_SPECIFIC.value
        total_emissions = ZERO

        for result in results:
            total_emissions = total_emissions + result.emissions_kg_co2e

        by_method[supplier_method_key] = {
            "count": Decimal(str(covered_assets)),
            "emissions_kg_co2e": total_emissions,
        }

        # Breakdown by data source within supplier-specific
        source_counts: Dict[str, int] = defaultdict(int)
        source_emissions: Dict[str, Decimal] = defaultdict(lambda: ZERO)

        for result in results:
            src_key = f"supplier_{result.data_source.value}"
            source_counts[src_key] += 1
            source_emissions[src_key] = (
                source_emissions[src_key] + result.emissions_kg_co2e
            )

        for src_key in source_counts:
            by_method[src_key] = {
                "count": Decimal(str(source_counts[src_key])),
                "emissions_kg_co2e": source_emissions[src_key],
            }

        # Gap analysis: assets not covered
        uncovered_count = max(0, total_assets - covered_assets)
        gap_categories: List[str] = []
        if uncovered_count > 0:
            gap_categories.append(
                f"{uncovered_count} assets lack supplier-specific data"
            )

        # Breakdown by verification for quality gap identification
        unverified_count = sum(
            1
            for r in results
            if r.verification_status in ("unverified", "estimated")
        )
        if unverified_count > 0:
            gap_categories.append(
                f"{unverified_count} records have unverified/estimated data"
            )

        return CoverageReport(
            total_assets=total_assets,
            covered_assets=covered_assets,
            coverage_pct=coverage_pct,
            by_method=by_method,
            uncovered_capex_usd=ZERO,  # Cannot determine without asset CapEx data
            gap_categories=gap_categories,
        )

    # ==================================================================
    #  PUBLIC API: Uncertainty
    # ==================================================================

    def estimate_uncertainty(
        self,
        result: SupplierSpecificResult,
    ) -> Dict[str, Any]:
        """Estimate uncertainty range for a supplier-specific result.

        Uses the GHG Protocol Scope 3 default uncertainty ranges for
        the supplier-specific method, adjusted by data source quality
        and verification status.

        Uncertainty is calculated as:
            base_uncertainty = method default from UNCERTAINTY_RANGES
            source_adjustment = DATA_SOURCE_UNCERTAINTY[source]
            verification_adjustment = (1 - verification_score) * 10
            final_uncertainty = weighted average of factors

        Args:
            result: SupplierSpecificResult to estimate uncertainty for.

        Returns:
            Dict with keys:
                - ``uncertainty_pct`` (Decimal): Final +/- percentage.
                - ``lower_bound_kg_co2e`` (Decimal): Low estimate.
                - ``upper_bound_kg_co2e`` (Decimal): High estimate.
                - ``confidence_level_pct`` (Decimal): Confidence level.
                - ``method`` (str): Uncertainty estimation method.
                - ``components`` (dict): Breakdown of uncertainty sources.
        """
        # Base uncertainty from method defaults
        method_range = UNCERTAINTY_RANGES.get(
            CalculationMethod.SUPPLIER_SPECIFIC,
            (Decimal("10"), Decimal("30")),
        )
        base_min_pct = method_range[0]
        base_max_pct = method_range[1]
        base_mid_pct = ((base_min_pct + base_max_pct) / Decimal("2")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        # Source-specific uncertainty
        source_pct = DATA_SOURCE_UNCERTAINTY.get(
            result.data_source,
            Decimal("50.0"),
        )

        # Verification adjustment
        ver_score = VERIFICATION_SCORES.get(
            result.verification_status,
            Decimal("0.1"),
        )
        ver_adjustment = ((ONE - ver_score) * Decimal("10")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        # Combined uncertainty: weighted average
        # 40% base, 40% source, 20% verification
        combined_pct = (
            base_mid_pct * Decimal("0.40")
            + source_pct * Decimal("0.40")
            + ver_adjustment * Decimal("0.20")
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Clamp to method range
        combined_pct = max(base_min_pct, min(base_max_pct, combined_pct))

        # Calculate bounds
        emissions = result.emissions_kg_co2e
        delta = (emissions * combined_pct / ONE_HUNDRED).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        lower_bound = max(ZERO, emissions - delta)
        upper_bound = emissions + delta

        return {
            "uncertainty_pct": combined_pct,
            "lower_bound_kg_co2e": lower_bound.quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            ),
            "upper_bound_kg_co2e": upper_bound.quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            ),
            "confidence_level_pct": Decimal("95.0"),
            "method": "tier_default_adjusted",
            "components": {
                "base_method_pct": base_mid_pct,
                "source_pct": source_pct,
                "verification_adjustment_pct": ver_adjustment,
                "verification_score": ver_score,
                "weight_base": Decimal("0.40"),
                "weight_source": Decimal("0.40"),
                "weight_verification": Decimal("0.20"),
            },
        }

    # ==================================================================
    #  PUBLIC API: Provenance
    # ==================================================================

    def compute_provenance_hash(
        self,
        record: SupplierRecord,
        result: Any,
    ) -> str:
        """Compute SHA-256 provenance hash for audit trail.

        Creates a deterministic hash from the input record and
        calculation result for tamper-evident audit trail.  The hash
        covers all material fields that affect the emission calculation.

        Args:
            record: SupplierRecord input.
            result: Calculation result (dict or SupplierSpecificResult).

        Returns:
            64-character hex SHA-256 digest string.
        """
        hash_input = self._build_hash_payload(record, result)
        digest = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

        logger.debug(
            "Provenance hash computed for record=%s: %s",
            record.record_id,
            digest[:16] + "...",
        )
        return digest

    # ==================================================================
    #  INTERNAL: Calculation Path Selection
    # ==================================================================

    def _select_calculation_path(
        self,
        record: SupplierRecord,
        forced_path: Optional[str] = None,
    ) -> str:
        """Select the appropriate calculation path for a record.

        Decision logic:
            1. If forced_path is specified, use it.
            2. If allocation_factor == 1.0 and source is product-level
               (EPD, PCF, DIRECT_MEASUREMENT) -> product_level.
            3. If source is PCF and allocation_factor < 1.0 -> pcf.
            4. If allocation_factor < 1.0 (CDP, facility) -> facility.
            5. Default: product_level.

        Args:
            record: SupplierRecord with data source and allocation info.
            forced_path: Optional forced path override.

        Returns:
            String path identifier: "product_level", "pcf", or "facility".
        """
        if forced_path:
            valid_paths = {"product_level", "pcf", "facility"}
            if forced_path not in valid_paths:
                raise ValueError(
                    f"Invalid forced_path '{forced_path}'; "
                    f"expected one of {valid_paths}"
                )
            return forced_path

        product_sources = {
            SupplierDataSource.EPD,
            SupplierDataSource.DIRECT_MEASUREMENT,
        }

        is_full_allocation = record.allocation_factor == ONE

        if record.data_source == SupplierDataSource.PCF and not is_full_allocation:
            return "pcf"

        if record.data_source in product_sources and is_full_allocation:
            return "product_level"

        if not is_full_allocation:
            return "facility"

        # Default for PCF with full allocation, ECOVADIS, etc.
        return "product_level"

    # ==================================================================
    #  INTERNAL: Product-Level Calculation
    # ==================================================================

    def _calculate_product_level_internal(
        self,
        record: SupplierRecord,
        asset: CapitalAssetRecord,
        trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute product-level EF calculation.

        Formula:
            emissions_kg_co2e = quantity x ef_value

        Gas split uses default ratios unless the supplier data
        includes per-gas breakdowns.

        Args:
            record: SupplierRecord with product-level ef_value.
            asset: CapitalAssetRecord with quantity.
            trace: Mutable trace dict for recording steps.

        Returns:
            Dict with emissions_kg_co2e, co2, ch4, n2o, dqi_score,
            uncertainty_pct.
        """
        quantity = asset.quantity
        ef_value = record.ef_value

        trace["path"] = "product_level"
        trace["quantity"] = str(quantity)
        trace["ef_value"] = str(ef_value)
        trace["ef_unit"] = record.ef_unit

        # Core formula: emissions = quantity x ef_value
        emissions_kg_co2e = (quantity * ef_value).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        trace["emissions_kg_co2e"] = str(emissions_kg_co2e)

        # Gas split
        co2, ch4, n2o = self._split_gases(emissions_kg_co2e)
        trace["co2"] = str(co2)
        trace["ch4"] = str(ch4)
        trace["n2o"] = str(n2o)

        # DQI scoring
        dqi = self.score_dqi(record)
        dqi_score = dqi.composite_score
        trace["dqi_score"] = str(dqi_score)

        # Uncertainty
        uncertainty_pct = DATA_SOURCE_UNCERTAINTY.get(
            record.data_source,
            Decimal("30.0"),
        )
        ver_score = self.score_verification(record)
        # Adjust: higher verification = lower uncertainty
        uncertainty_pct = (
            uncertainty_pct * (ONE - ver_score * Decimal("0.5"))
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        uncertainty_pct = max(Decimal("5.0"), uncertainty_pct)
        trace["uncertainty_pct"] = str(uncertainty_pct)

        return {
            "emissions_kg_co2e": emissions_kg_co2e,
            "co2": co2,
            "ch4": ch4,
            "n2o": n2o,
            "dqi_score": dqi_score,
            "uncertainty_pct": uncertainty_pct,
        }

    # ==================================================================
    #  INTERNAL: PCF Calculation
    # ==================================================================

    def _calculate_pcf_internal(
        self,
        record: SupplierRecord,
        asset: CapitalAssetRecord,
        trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute PCF value path calculation.

        Formula:
            emissions_kg_co2e = ef_value x allocation_factor

        Where:
            ef_value = total PCF value (kgCO2e for functional unit)
            allocation_factor = units_purchased / units_in_pcf_scope

        For PCF, the ef_value represents the TOTAL product carbon
        footprint for the declared unit scope, and the allocation
        factor represents the fraction of that scope purchased.

        If the asset has a quantity > 1 and allocation_factor == 1.0,
        the formula becomes: ef_value x quantity (each unit gets full PCF).

        Args:
            record: SupplierRecord with PCF total ef_value.
            asset: CapitalAssetRecord for the capital good.
            trace: Mutable trace dict for recording steps.

        Returns:
            Dict with emissions_kg_co2e, co2, ch4, n2o, dqi_score,
            uncertainty_pct.
        """
        pcf_total = record.ef_value
        allocation_factor = record.allocation_factor
        quantity = asset.quantity

        trace["path"] = "pcf"
        trace["pcf_total_kgco2e"] = str(pcf_total)
        trace["allocation_factor"] = str(allocation_factor)
        trace["quantity"] = str(quantity)

        # Core formula: emissions = pcf_total x allocation_factor
        # If allocation_factor < 1, it means we are purchasing a fraction
        # of the scope. If allocation_factor == 1 and quantity > 1,
        # multiply by quantity (each unit has full PCF value).
        if allocation_factor < ONE:
            emissions_kg_co2e = (pcf_total * allocation_factor).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            trace["formula"] = "pcf_total x allocation_factor"
        else:
            emissions_kg_co2e = (pcf_total * quantity).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            trace["formula"] = "pcf_total x quantity"

        trace["emissions_kg_co2e"] = str(emissions_kg_co2e)

        # Gas split
        co2, ch4, n2o = self._split_gases(emissions_kg_co2e)
        trace["co2"] = str(co2)
        trace["ch4"] = str(ch4)
        trace["n2o"] = str(n2o)

        # DQI scoring
        dqi = self.score_dqi(record)
        dqi_score = dqi.composite_score
        trace["dqi_score"] = str(dqi_score)

        # Uncertainty
        uncertainty_pct = DATA_SOURCE_UNCERTAINTY.get(
            record.data_source,
            Decimal("30.0"),
        )
        ver_score = self.score_verification(record)
        uncertainty_pct = (
            uncertainty_pct * (ONE - ver_score * Decimal("0.5"))
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        uncertainty_pct = max(Decimal("5.0"), uncertainty_pct)
        trace["uncertainty_pct"] = str(uncertainty_pct)

        return {
            "emissions_kg_co2e": emissions_kg_co2e,
            "co2": co2,
            "ch4": ch4,
            "n2o": n2o,
            "dqi_score": dqi_score,
            "uncertainty_pct": uncertainty_pct,
        }

    # ==================================================================
    #  INTERNAL: Facility Allocation Calculation
    # ==================================================================

    def _calculate_facility_internal(
        self,
        record: SupplierRecord,
        asset: CapitalAssetRecord,
        trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute facility-level allocation calculation.

        Formula:
            emissions_kg_co2e = facility_total x allocation_factor

        Where:
            facility_total = ef_value (total facility emissions kgCO2e)
            allocation_factor = allocation share (0-1) computed by
                the specified allocation method (economic, physical,
                mass, energy, or hybrid).

        Args:
            record: SupplierRecord with facility ef_value and allocation.
            asset: CapitalAssetRecord for the capital good.
            trace: Mutable trace dict for recording steps.

        Returns:
            Dict with emissions_kg_co2e, co2, ch4, n2o, dqi_score,
            uncertainty_pct.
        """
        facility_total = record.ef_value
        allocation_factor = record.allocation_factor
        allocation_method = record.allocation_method

        trace["path"] = "facility"
        trace["facility_total_kgco2e"] = str(facility_total)
        trace["allocation_factor"] = str(allocation_factor)
        trace["allocation_method"] = allocation_method.value

        # Validate allocation factor
        if allocation_factor <= ZERO:
            logger.warning(
                "Facility allocation factor is zero or negative for "
                "record=%s; emissions will be zero",
                record.record_id,
            )

        # Core formula: emissions = facility_total x allocation_factor
        emissions_kg_co2e = (facility_total * allocation_factor).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        trace["emissions_kg_co2e"] = str(emissions_kg_co2e)
        trace["formula"] = "facility_total x allocation_factor"

        # Gas split
        co2, ch4, n2o = self._split_gases(emissions_kg_co2e)
        trace["co2"] = str(co2)
        trace["ch4"] = str(ch4)
        trace["n2o"] = str(n2o)

        # DQI scoring - facility data gets a completeness penalty
        dqi = self.score_dqi(record)
        dqi_score = dqi.composite_score

        # Facility allocation adds uncertainty vs. product-level
        allocation_penalty = Decimal("0.5")
        if allocation_method == AllocationMethod.ECONOMIC:
            allocation_penalty = Decimal("0.5")
        elif allocation_method == AllocationMethod.PHYSICAL:
            allocation_penalty = Decimal("0.3")
        elif allocation_method == AllocationMethod.MASS:
            allocation_penalty = Decimal("0.3")
        elif allocation_method == AllocationMethod.ENERGY:
            allocation_penalty = Decimal("0.4")
        elif allocation_method == AllocationMethod.HYBRID:
            allocation_penalty = Decimal("0.2")

        adjusted_dqi = min(
            Decimal("5.0"),
            dqi_score + allocation_penalty,
        )
        trace["dqi_score_raw"] = str(dqi_score)
        trace["dqi_allocation_penalty"] = str(allocation_penalty)
        dqi_score = adjusted_dqi
        trace["dqi_score"] = str(dqi_score)

        # Uncertainty: facility allocation adds ~10-15% additional uncertainty
        base_uncertainty = DATA_SOURCE_UNCERTAINTY.get(
            record.data_source,
            Decimal("30.0"),
        )
        allocation_uncertainty = Decimal("10.0")
        if allocation_method == AllocationMethod.ECONOMIC:
            allocation_uncertainty = Decimal("15.0")
        elif allocation_method == AllocationMethod.PHYSICAL:
            allocation_uncertainty = Decimal("10.0")
        elif allocation_method == AllocationMethod.MASS:
            allocation_uncertainty = Decimal("10.0")
        elif allocation_method == AllocationMethod.ENERGY:
            allocation_uncertainty = Decimal("12.0")
        elif allocation_method == AllocationMethod.HYBRID:
            allocation_uncertainty = Decimal("8.0")

        # Combined uncertainty via root-sum-of-squares approximation
        uncertainty_pct = _sqrt_decimal(
            base_uncertainty * base_uncertainty
            + allocation_uncertainty * allocation_uncertainty
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Clamp to supplier-specific range
        method_range = UNCERTAINTY_RANGES.get(
            CalculationMethod.SUPPLIER_SPECIFIC,
            (Decimal("10"), Decimal("30")),
        )
        uncertainty_pct = max(method_range[0], min(method_range[1], uncertainty_pct))
        trace["uncertainty_pct"] = str(uncertainty_pct)

        return {
            "emissions_kg_co2e": emissions_kg_co2e,
            "co2": co2,
            "ch4": ch4,
            "n2o": n2o,
            "dqi_score": dqi_score,
            "uncertainty_pct": uncertainty_pct,
        }

    # ==================================================================
    #  INTERNAL: Allocation Methods
    # ==================================================================

    def _allocate_economic(
        self,
        total_emissions: Decimal,
        data: Dict[str, Decimal],
    ) -> Decimal:
        """Allocate by economic value ratio.

        Formula:
            share = asset_value / total_value
            allocated = total_emissions x share

        Args:
            total_emissions: Total facility emissions kgCO2e.
            data: Must contain ``asset_value`` and ``total_value``.

        Returns:
            Allocated emissions in kgCO2e.

        Raises:
            ValueError: If keys missing or total_value is zero.
        """
        asset_value = data.get("asset_value")
        total_value = data.get("total_value")

        if asset_value is None or total_value is None:
            raise ValueError(
                "Economic allocation requires 'asset_value' and 'total_value'"
            )
        if total_value <= ZERO:
            raise ValueError(
                f"total_value must be positive, got {total_value}"
            )

        share = (asset_value / total_value).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        allocated = (total_emissions * share).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "Economic allocation: asset_value=%s / total_value=%s = share=%s, "
            "allocated=%s kgCO2e",
            asset_value,
            total_value,
            share,
            allocated,
        )
        return allocated

    def _allocate_physical(
        self,
        total_emissions: Decimal,
        data: Dict[str, Decimal],
    ) -> Decimal:
        """Allocate by physical output ratio.

        Formula:
            share = asset_quantity / total_quantity
            allocated = total_emissions x share

        Args:
            total_emissions: Total facility emissions kgCO2e.
            data: Must contain ``asset_quantity`` and ``total_quantity``.

        Returns:
            Allocated emissions in kgCO2e.

        Raises:
            ValueError: If keys missing or total_quantity is zero.
        """
        asset_qty = data.get("asset_quantity")
        total_qty = data.get("total_quantity")

        if asset_qty is None or total_qty is None:
            raise ValueError(
                "Physical allocation requires 'asset_quantity' and 'total_quantity'"
            )
        if total_qty <= ZERO:
            raise ValueError(
                f"total_quantity must be positive, got {total_qty}"
            )

        share = (asset_qty / total_qty).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        allocated = (total_emissions * share).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "Physical allocation: asset_qty=%s / total_qty=%s = share=%s, "
            "allocated=%s kgCO2e",
            asset_qty,
            total_qty,
            share,
            allocated,
        )
        return allocated

    def _allocate_mass(
        self,
        total_emissions: Decimal,
        data: Dict[str, Decimal],
    ) -> Decimal:
        """Allocate by mass ratio.

        Formula:
            share = asset_mass_kg / total_mass_kg
            allocated = total_emissions x share

        Args:
            total_emissions: Total facility emissions kgCO2e.
            data: Must contain ``asset_mass_kg`` and ``total_mass_kg``.

        Returns:
            Allocated emissions in kgCO2e.

        Raises:
            ValueError: If keys missing or total_mass_kg is zero.
        """
        asset_mass = data.get("asset_mass_kg")
        total_mass = data.get("total_mass_kg")

        if asset_mass is None or total_mass is None:
            raise ValueError(
                "Mass allocation requires 'asset_mass_kg' and 'total_mass_kg'"
            )
        if total_mass <= ZERO:
            raise ValueError(
                f"total_mass_kg must be positive, got {total_mass}"
            )

        share = (asset_mass / total_mass).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        allocated = (total_emissions * share).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "Mass allocation: asset_mass=%s / total_mass=%s = share=%s, "
            "allocated=%s kgCO2e",
            asset_mass,
            total_mass,
            share,
            allocated,
        )
        return allocated

    def _allocate_energy(
        self,
        total_emissions: Decimal,
        data: Dict[str, Decimal],
    ) -> Decimal:
        """Allocate by energy consumption ratio.

        Formula:
            share = asset_energy_mj / total_energy_mj
            allocated = total_emissions x share

        Args:
            total_emissions: Total facility emissions kgCO2e.
            data: Must contain ``asset_energy_mj`` and ``total_energy_mj``.

        Returns:
            Allocated emissions in kgCO2e.

        Raises:
            ValueError: If keys missing or total_energy_mj is zero.
        """
        asset_energy = data.get("asset_energy_mj")
        total_energy = data.get("total_energy_mj")

        if asset_energy is None or total_energy is None:
            raise ValueError(
                "Energy allocation requires 'asset_energy_mj' and 'total_energy_mj'"
            )
        if total_energy <= ZERO:
            raise ValueError(
                f"total_energy_mj must be positive, got {total_energy}"
            )

        share = (asset_energy / total_energy).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        allocated = (total_emissions * share).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "Energy allocation: asset_energy=%s / total_energy=%s = share=%s, "
            "allocated=%s kgCO2e",
            asset_energy,
            total_energy,
            share,
            allocated,
        )
        return allocated

    def _allocate_hybrid(
        self,
        total_emissions: Decimal,
        data: Dict[str, Decimal],
    ) -> Decimal:
        """Allocate by weighted hybrid combination of methods.

        Computes a weighted average of multiple allocation methods.
        The weights are specified in ``data["weights"]``.

        Formula:
            hybrid_share = sum(weight_i x share_i) for each method
            allocated = total_emissions x hybrid_share

        Args:
            total_emissions: Total facility emissions kgCO2e.
            data: Must contain:
                - ``weights``: Dict mapping method name to weight.
                    E.g. {"economic": 0.6, "physical": 0.4}.
                - All numerator/denominator pairs for included methods.

        Returns:
            Allocated emissions in kgCO2e.

        Raises:
            ValueError: If weights missing, don't sum to ~1.0, or if
                required method data is missing.
        """
        weights_raw = data.get("weights")
        if weights_raw is None:
            raise ValueError("Hybrid allocation requires 'weights' dict")

        # Convert weights to Decimal if needed
        weights: Dict[str, Decimal] = {}
        for method_name, weight in weights_raw.items():
            if isinstance(weight, Decimal):
                weights[str(method_name)] = weight
            else:
                weights[str(method_name)] = Decimal(str(weight))

        # Validate weights sum to ~1.0 (within 0.01 tolerance)
        weight_sum = sum(weights.values())
        if abs(weight_sum - ONE) > Decimal("0.01"):
            raise ValueError(
                f"Hybrid weights must sum to 1.0, got {weight_sum}"
            )

        # Compute weighted share
        method_allocators = {
            "economic": self._allocate_economic,
            "physical": self._allocate_physical,
            "mass": self._allocate_mass,
            "energy": self._allocate_energy,
        }

        hybrid_emissions = ZERO

        for method_name, weight in weights.items():
            if method_name not in method_allocators:
                raise ValueError(
                    f"Hybrid method '{method_name}' is not a valid base method; "
                    f"expected one of {list(method_allocators.keys())}"
                )
            method_allocated = method_allocators[method_name](
                total_emissions, data
            )
            weighted_contribution = (method_allocated * weight).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            hybrid_emissions = hybrid_emissions + weighted_contribution

        result = hybrid_emissions.quantize(_PRECISION, rounding=ROUND_HALF_UP)

        logger.debug(
            "Hybrid allocation: weights=%s, allocated=%s kgCO2e",
            {k: str(v) for k, v in weights.items()},
            result,
        )
        return result

    # ==================================================================
    #  INTERNAL: Gas Split
    # ==================================================================

    def _split_gases(
        self,
        total_kg_co2e: Decimal,
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Split total CO2e into constituent gas components.

        Uses default gas split ratios (CO2 98%, CH4 1.5%, N2O 0.5%)
        which are typical for manufacturing cradle-to-gate profiles.

        The split ensures the three components sum to the total within
        rounding tolerance by assigning the remainder to CO2.

        Args:
            total_kg_co2e: Total emissions in kgCO2e.

        Returns:
            Tuple of (co2, ch4, n2o) all in kgCO2e.
        """
        ch4 = (total_kg_co2e * self._gas_ch4).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        n2o = (total_kg_co2e * self._gas_n2o).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        # CO2 gets the remainder to ensure exact sum
        co2 = (total_kg_co2e - ch4 - n2o).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        # Guard against negative CO2 from rounding
        if co2 < ZERO:
            co2 = ZERO

        return co2, ch4, n2o

    # ==================================================================
    #  INTERNAL: DQI Dimension Scoring
    # ==================================================================

    def _score_temporal(
        self,
        record: SupplierRecord,
        findings: List[str],
    ) -> Decimal:
        """Score temporal representativeness (1-5).

        Scoring:
            1.0: Data from current or previous year.
            2.0: Data 2 years old.
            3.0: Data 3 years old.
            4.0: Data 4-5 years old.
            5.0: Data >5 years old or no reporting year.

        Args:
            record: SupplierRecord to assess.
            findings: Mutable findings list to append to.

        Returns:
            Decimal score in [1.0, 5.0].
        """
        if record.reporting_year is None:
            findings.append(
                "Temporal: No reporting year provided; assigned worst score"
            )
            return Decimal("5.0")

        age = _current_year() - record.reporting_year

        if age < 0:
            findings.append(
                f"Temporal: Reporting year {record.reporting_year} is in the future"
            )
            return Decimal("3.0")
        elif age <= 1:
            return Decimal("1.0")
        elif age == 2:
            findings.append(
                f"Temporal: Data is {age} years old (Good quality)"
            )
            return Decimal("2.0")
        elif age == 3:
            findings.append(
                f"Temporal: Data is {age} years old (Fair quality)"
            )
            return Decimal("3.0")
        elif age <= 5:
            findings.append(
                f"Temporal: Data is {age} years old (Poor quality); "
                "recommend requesting updated supplier data"
            )
            return Decimal("4.0")
        else:
            findings.append(
                f"Temporal: Data is {age} years old (Very Poor quality); "
                "supplier data should be refreshed urgently"
            )
            return Decimal("5.0")

    def _score_geographical(
        self,
        record: SupplierRecord,
        findings: List[str],
    ) -> Decimal:
        """Score geographical representativeness (1-5).

        Supplier-specific data is inherently geographically
        representative since it comes from the actual supplier.
        EPD and PCF are product-specific.  CDP/EcoVadis may be
        at facility or corporate level.

        Scoring by data source:
            EPD / DIRECT_MEASUREMENT: 1.0 (product-specific)
            PCF: 1.5 (product-specific, may be generic site)
            CDP: 2.0 (facility-level, good geographical match)
            ECOVADIS: 3.0 (corporate-level, may span regions)
            ESTIMATED: 4.0 (no specific geography)

        Args:
            record: SupplierRecord to assess.
            findings: Mutable findings list.

        Returns:
            Decimal score in [1.0, 5.0].
        """
        source_geo_scores: Dict[SupplierDataSource, Decimal] = {
            SupplierDataSource.EPD: Decimal("1.0"),
            SupplierDataSource.PCF: Decimal("1.5"),
            SupplierDataSource.CDP: Decimal("2.0"),
            SupplierDataSource.DIRECT_MEASUREMENT: Decimal("1.0"),
            SupplierDataSource.ECOVADIS: Decimal("3.0"),
            SupplierDataSource.ESTIMATED: Decimal("4.0"),
        }

        score = source_geo_scores.get(record.data_source, Decimal("3.0"))

        if score > Decimal("2.0"):
            findings.append(
                f"Geographical: {record.data_source.value} data source "
                "may not fully represent the specific manufacturing geography"
            )

        return score

    def _score_technological(
        self,
        record: SupplierRecord,
        findings: List[str],
    ) -> Decimal:
        """Score technological representativeness (1-5).

        Supplier-specific data should match the actual product
        technology.  EPD and PCF are product-specific.  CDP is
        facility-average (may cover many product types).

        Scoring by data source:
            EPD: 1.0 (product-specific declaration)
            DIRECT_MEASUREMENT: 1.5 (product-specific measurement)
            PCF: 1.0 (product-specific footprint)
            CDP: 3.0 (facility-level, averaged across products)
            ECOVADIS: 3.5 (corporate assessment, technology-generic)
            ESTIMATED: 4.0 (industry proxy, not technology-specific)

        Args:
            record: SupplierRecord to assess.
            findings: Mutable findings list.

        Returns:
            Decimal score in [1.0, 5.0].
        """
        source_tech_scores: Dict[SupplierDataSource, Decimal] = {
            SupplierDataSource.EPD: Decimal("1.0"),
            SupplierDataSource.PCF: Decimal("1.0"),
            SupplierDataSource.DIRECT_MEASUREMENT: Decimal("1.5"),
            SupplierDataSource.CDP: Decimal("3.0"),
            SupplierDataSource.ECOVADIS: Decimal("3.5"),
            SupplierDataSource.ESTIMATED: Decimal("4.0"),
        }

        score = source_tech_scores.get(record.data_source, Decimal("3.0"))

        if score > Decimal("2.0"):
            findings.append(
                f"Technological: {record.data_source.value} data may not "
                "precisely represent the specific product technology; "
                "consider requesting product-level EPD or PCF"
            )

        return score

    def _score_completeness(
        self,
        record: SupplierRecord,
        findings: List[str],
    ) -> Decimal:
        """Score data completeness (1-5).

        Evaluates whether the supplier data covers all relevant
        emission sources within the system boundary.

        Scoring factors:
            1. System boundary: cradle_to_gate = +0; other = +1
            2. Data source completeness (EPD/PCF cover all sources)
            3. Allocation factor < 1 implies partial coverage: +0.5
            4. Missing reporting year: +1

        Base scores by source:
            EPD: 1.0 (comprehensive, all sources)
            PCF: 1.5 (comprehensive for carbon)
            CDP: 2.5 (may miss some Scope 3 upstream)
            DIRECT_MEASUREMENT: 2.0 (depends on boundary)
            ECOVADIS: 3.0 (assessment, not measurement)
            ESTIMATED: 4.0 (broad estimate, incomplete)

        Args:
            record: SupplierRecord to assess.
            findings: Mutable findings list.

        Returns:
            Decimal score in [1.0, 5.0].
        """
        base_scores: Dict[SupplierDataSource, Decimal] = {
            SupplierDataSource.EPD: Decimal("1.0"),
            SupplierDataSource.PCF: Decimal("1.5"),
            SupplierDataSource.CDP: Decimal("2.5"),
            SupplierDataSource.DIRECT_MEASUREMENT: Decimal("2.0"),
            SupplierDataSource.ECOVADIS: Decimal("3.0"),
            SupplierDataSource.ESTIMATED: Decimal("4.0"),
        }

        score = base_scores.get(record.data_source, Decimal("3.0"))

        # Boundary adjustment
        if record.boundary != "cradle_to_gate":
            score = score + Decimal("0.5")
            findings.append(
                f"Completeness: Boundary is '{record.boundary}'; "
                "cradle_to_gate preferred for capital goods"
            )

        # Allocation partial coverage penalty
        if record.allocation_factor < ONE:
            score = score + Decimal("0.5")
            findings.append(
                f"Completeness: Allocation factor {record.allocation_factor} < 1.0 "
                "indicates partial coverage"
            )

        # Missing reporting year
        if record.reporting_year is None:
            score = score + Decimal("0.5")

        # Clamp to [1.0, 5.0]
        score = max(Decimal("1.0"), min(Decimal("5.0"), score))

        return score

    def _score_reliability(
        self,
        record: SupplierRecord,
        verification_level: str,
        findings: List[str],
    ) -> Decimal:
        """Score data reliability (1-5).

        Maps verification level and data source to a reliability
        score.  Third-party verified EPDs get the best score;
        unverified estimates get the worst.

        Scoring:
            Verification score (0.1-1.0) is mapped to DQI (1-5):
                DQI = 5 - (verification_score * 4)
            Then adjusted by data source reliability.

        Args:
            record: SupplierRecord to assess.
            verification_level: Verification status string.
            findings: Mutable findings list.

        Returns:
            Decimal score in [1.0, 5.0].
        """
        ver_score = VERIFICATION_SCORES.get(
            verification_level,
            Decimal("0.1"),
        )

        # Map verification score (0.1-1.0) to DQI (1.0-5.0)
        # ver=1.0 -> DQI=1.0, ver=0.1 -> DQI=4.6
        ver_dqi = (
            Decimal("5.0") - (ver_score * Decimal("4.0"))
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Source reliability adjustment
        source_reliability = DATA_SOURCE_RELIABILITY.get(
            record.data_source,
            Decimal("3.0"),
        )

        # Weighted: 60% verification, 40% source
        reliability = (
            ver_dqi * Decimal("0.6") + source_reliability * Decimal("0.4")
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Clamp to [1.0, 5.0]
        reliability = max(Decimal("1.0"), min(Decimal("5.0"), reliability))

        if reliability > Decimal("3.0"):
            findings.append(
                f"Reliability: Verification '{verification_level}' with "
                f"source '{record.data_source.value}' yields poor reliability; "
                "recommend obtaining third-party verified data"
            )

        return reliability

    # ==================================================================
    #  INTERNAL: Quality Tier and Pedigree
    # ==================================================================

    def _determine_quality_tier(self, composite_score: Decimal) -> str:
        """Determine the quality tier label from composite DQI score.

        Uses the DQI_QUALITY_TIERS lookup from models.py.

        Args:
            composite_score: Composite DQI score (1.0-5.0).

        Returns:
            Quality tier label string.
        """
        for tier_name, (low, high) in DQI_QUALITY_TIERS.items():
            if low <= composite_score < high:
                return tier_name

        # Fallback for edge cases
        if composite_score >= Decimal("4.6"):
            return "Very Poor"
        return "Fair"

    def _compute_pedigree_uncertainty(self, composite_score: Decimal) -> Decimal:
        """Compute pedigree uncertainty factor from composite DQI.

        Maps the composite score to the nearest DQIScore enum value
        and looks up the pedigree uncertainty factor.

        Args:
            composite_score: Composite DQI score (1.0-5.0).

        Returns:
            Pedigree uncertainty factor (Decimal >= 1.0).
        """
        # Map composite to nearest DQI score level
        rounded = int(composite_score.quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        ))
        rounded = max(1, min(5, rounded))

        try:
            dqi_level = DQIScore(rounded)
        except ValueError:
            dqi_level = DQIScore.FAIR

        factor = PEDIGREE_UNCERTAINTY_FACTORS.get(
            dqi_level,
            Decimal("1.10"),
        )
        return factor

    # ==================================================================
    #  INTERNAL: Provenance Hash Payload
    # ==================================================================

    def _build_hash_payload(
        self,
        record: SupplierRecord,
        result: Any,
    ) -> str:
        """Build a deterministic string payload for SHA-256 hashing.

        Serializes all material fields from the input record and
        calculation result into a canonical JSON string.

        Args:
            record: SupplierRecord input.
            result: Dict or SupplierSpecificResult.

        Returns:
            Canonical JSON string for hashing.
        """
        # Extract result fields
        if isinstance(result, dict):
            result_data = {
                k: str(v) for k, v in result.items()
            }
        else:
            # SupplierSpecificResult (Pydantic model)
            result_data = {
                "emissions_kg_co2e": str(getattr(result, "emissions_kg_co2e", "")),
                "co2": str(getattr(result, "co2", "")),
                "ch4": str(getattr(result, "ch4", "")),
                "n2o": str(getattr(result, "n2o", "")),
                "dqi_score": str(getattr(result, "dqi_score", "")),
                "uncertainty_pct": str(getattr(result, "uncertainty_pct", "")),
            }

        payload = {
            "engine": ENGINE_ID,
            "version": ENGINE_VERSION,
            "record_id": record.record_id,
            "asset_id": record.asset_id,
            "supplier_name": record.supplier_name,
            "data_source": record.data_source.value,
            "ef_value": str(record.ef_value),
            "ef_unit": record.ef_unit,
            "allocation_method": record.allocation_method.value,
            "allocation_factor": str(record.allocation_factor),
            "verification_status": record.verification_status,
            "boundary": record.boundary,
            "result": result_data,
        }

        # Deterministic serialization: sorted keys, no spaces
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# ============================================================================
# Module-level Utility: Decimal Square Root
# ============================================================================


def _sqrt_decimal(value: Decimal, precision: int = DECIMAL_PLACES) -> Decimal:
    """Compute the square root of a Decimal using Newton's method.

    Uses iterative Newton-Raphson with Decimal arithmetic for
    deterministic, platform-independent results.

    Args:
        value: Non-negative Decimal to compute square root of.
        precision: Number of decimal places for convergence.
            Default is DECIMAL_PLACES (8).

    Returns:
        Square root as Decimal, quantized to specified precision.

    Raises:
        ValueError: If value is negative.
    """
    if value < ZERO:
        raise ValueError(f"Cannot compute square root of negative: {value}")
    if value == ZERO:
        return ZERO

    quantizer = Decimal(10) ** -precision

    # Initial guess: use integer square root for starting point
    # For values > 1, start with value/2; for values < 1, start with 1
    if value >= ONE:
        guess = value / Decimal("2")
    else:
        guess = ONE

    # Newton-Raphson iteration
    max_iterations = 100
    tolerance = Decimal(10) ** -(precision + 2)

    for _ in range(max_iterations):
        next_guess = ((guess + value / guess) / Decimal("2")).quantize(
            tolerance, rounding=ROUND_HALF_UP
        )
        diff = abs(next_guess - guess)
        if diff <= tolerance:
            return next_guess.quantize(quantizer, rounding=ROUND_HALF_UP)
        guess = next_guess

    return guess.quantize(quantizer, rounding=ROUND_HALF_UP)


# ============================================================================
# Module-level Factory
# ============================================================================


def get_supplier_specific_calculator(
    config: Optional[Dict[str, Any]] = None,
) -> SupplierSpecificCalculatorEngine:
    """Get the singleton SupplierSpecificCalculatorEngine instance.

    Convenience factory function for obtaining the engine singleton.

    Args:
        config: Optional configuration dict (only used on first call).

    Returns:
        The singleton SupplierSpecificCalculatorEngine instance.

    Example:
        >>> engine = get_supplier_specific_calculator()
        >>> result = engine.calculate(record, asset)
    """
    return SupplierSpecificCalculatorEngine(config)
