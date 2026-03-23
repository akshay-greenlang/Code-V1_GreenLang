# -*- coding: utf-8 -*-
"""
WasteTypeSpecificCalculatorEngine - Waste-type-specific EOL emission calculator.

This module implements Engine 2 for AGENT-MRV-025 (GL-MRV-S3-012)
End-of-Life Treatment of Sold Products. It calculates greenhouse gas emissions
from the end-of-life treatment of products sold by the reporting company using
the waste-type-specific method (Method A per GHG Protocol).

Core Formula:
    E_total = SUM_products [ SUM_materials [ SUM_treatments [
        units_sold x weight_per_unit x material_fraction x treatment_fraction x treatment_EF
    ]]]

Key Differentiator from Category 5 (Waste Generated in Operations):
    Category 5 covers waste from the reporting company's own operations.
    Category 12 covers end-of-life treatment of products SOLD by the company,
    disposed of by downstream consumers and third parties.

Treatment Pathway Calculations:
    1. Landfill - IPCC First Order Decay (FOD) model
    2. Incineration - IPCC mass balance (fossil CO2 + biogenic CO2 + N2O)
    3. Recycling - Cut-off approach (processing emissions + separate avoided credits)
    4. Composting - Aerobic decomposition (CH4 + N2O)
    5. Anaerobic Digestion - Fugitive methane from biogas leakage
    6. Open Burning - IPCC uncontrolled burning factors

Reporting Rules:
    - Biogenic CO2 ALWAYS reported separately from fossil CO2
    - Avoided emissions (recycling credits) ALWAYS reported separately
    - Energy recovery credits reported as avoided, NEVER netted from total
    - All arithmetic uses Decimal with ROUND_HALF_UP for regulatory precision

Zero-Hallucination Guarantees:
    - All calculations use deterministic Python Decimal arithmetic
    - No LLM calls anywhere in the calculation path
    - Every intermediate step recorded in calculation trace
    - SHA-256 provenance hash on every result
    - All emission factors sourced from EOLProductDatabaseEngine

References:
    - GHG Protocol Scope 3 Standard, Category 12 Technical Guidance
    - IPCC 2006 Guidelines for National GHG Inventories Vol 5
    - IPCC 2019 Refinement to the 2006 Guidelines
    - EPA WARM v16, DEFRA/DESNZ 2024

Example:
    >>> from greenlang.agents.mrv.end_of_life_treatment.waste_type_specific_calculator import (
    ...     WasteTypeSpecificCalculatorEngine,
    ... )
    >>> from decimal import Decimal
    >>> engine = WasteTypeSpecificCalculatorEngine()
    >>> products = [{
    ...     "product_id": "PROD-001",
    ...     "product_category": "electronics",
    ...     "units_sold": 10000,
    ...     "weight_per_unit_kg": Decimal("2.5"),
    ...     "region": "US",
    ... }]
    >>> result = engine.calculate(products, "ORG-001", 2025)
    >>> result["total_co2e_kg"] > Decimal("0")
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-012
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
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.agents.mrv.end_of_life_treatment.eol_product_database import (
    EOLProductDatabaseEngine,
    AGENT_ID,
    ENGINE_VERSION as DB_ENGINE_VERSION,
    VALID_MATERIALS,
    VALID_TREATMENTS,
    VALID_PRODUCT_CATEGORIES,
    VALID_REGIONS,
    GWP_VALUES,
    DEFAULT_N2O_INCINERATION_EF,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# ENGINE METADATA
# ==============================================================================

ENGINE_ID: str = "waste_type_specific_calculator_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_COMPONENT: str = "AGENT-MRV-025"

# ==============================================================================
# DECIMAL PRECISION AND CONSTANTS
# ==============================================================================

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWELVE = Decimal("12")
_SIXTEEN = Decimal("16")
_FORTY_FOUR = Decimal("44")
_THOUSAND = Decimal("1000")

# Molecular weight ratios
_CH4_C_RATIO = _SIXTEEN / _TWELVE         # 16/12 = 1.33333...
_CO2_C_RATIO = _FORTY_FOUR / _TWELVE      # 44/12 = 3.66666...

# MJ to kWh conversion
_MJ_TO_KWH = Decimal("1") / Decimal("3.6")

# Maximum batch size
_MAX_BATCH_SIZE: int = 10000

# Default GWP version (AR5 per GHG Protocol)
DEFAULT_GWP_VERSION: str = "AR5"

# Default N2O EF for composting (kg N2O / kg waste)
DEFAULT_COMPOSTING_N2O_EF = Decimal("0.0003")
DEFAULT_COMPOSTING_CH4_EF = Decimal("0.004")

# Default GWP values (AR5 100-year)
GWP_CH4_AR5 = Decimal("28")
GWP_N2O_AR5 = Decimal("265")

# Natural log of 2 for half-life calculations
_LN2 = Decimal(str(math.log(2)))


# ==============================================================================
# RESULT DATA CLASSES
# ==============================================================================


class EOLCalculationResult:
    """
    Complete result from waste-type-specific EOL emission calculation.

    Contains fossil CO2e total, biogenic CO2 (separate), avoided emissions
    (separate), per-treatment breakdown, per-product breakdown, DQI scores,
    uncertainty ranges, and provenance hash.

    Attributes:
        calculation_id: Unique calculation identifier.
        org_id: Organization identifier.
        reporting_year: Reporting year.
        total_co2e_kg: Total fossil CO2e emissions (kgCO2e).
        total_biogenic_co2_kg: Total biogenic CO2 (kgCO2, memo item).
        total_avoided_co2e_kg: Total avoided emissions from recycling (kgCO2e, separate).
        total_energy_recovery_credit_kg: Energy recovery credit (kgCO2e, separate).
        treatment_breakdown: Per-treatment pathway emissions.
        product_breakdown: Per-product emissions.
        material_breakdown: Per-material emissions.
        total_mass_kg: Total product mass processed.
        gwp_version: GWP assessment report version used.
        dqi_score: Data quality indicator score (1-5).
        uncertainty_pct: Uncertainty range (percentage, +/-).
        calculation_trace: Detailed calculation steps.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: Calculation timestamp (UTC).
    """

    def __init__(self) -> None:
        """Initialize with zero values."""
        self.calculation_id: str = str(uuid.uuid4())
        self.org_id: str = ""
        self.reporting_year: int = 0
        self.total_co2e_kg: Decimal = _ZERO
        self.total_biogenic_co2_kg: Decimal = _ZERO
        self.total_avoided_co2e_kg: Decimal = _ZERO
        self.total_energy_recovery_credit_kg: Decimal = _ZERO
        self.treatment_breakdown: Dict[str, Decimal] = {}
        self.product_breakdown: Dict[str, Dict[str, Decimal]] = {}
        self.material_breakdown: Dict[str, Decimal] = {}
        self.total_mass_kg: Decimal = _ZERO
        self.gwp_version: str = DEFAULT_GWP_VERSION
        self.dqi_score: Decimal = Decimal("3.0")
        self.uncertainty_pct: Decimal = Decimal("30.0")
        self.calculation_trace: List[Dict[str, Any]] = []
        self.provenance_hash: str = ""
        self.processing_time_ms: float = 0.0
        self.timestamp: str = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "calculation_id": self.calculation_id,
            "org_id": self.org_id,
            "reporting_year": self.reporting_year,
            "total_co2e_kg": str(self.total_co2e_kg),
            "total_biogenic_co2_kg": str(self.total_biogenic_co2_kg),
            "total_avoided_co2e_kg": str(self.total_avoided_co2e_kg),
            "total_energy_recovery_credit_kg": str(self.total_energy_recovery_credit_kg),
            "treatment_breakdown": {k: str(v) for k, v in self.treatment_breakdown.items()},
            "product_breakdown": {
                k: {kk: str(vv) for kk, vv in v.items()}
                for k, v in self.product_breakdown.items()
            },
            "material_breakdown": {k: str(v) for k, v in self.material_breakdown.items()},
            "total_mass_kg": str(self.total_mass_kg),
            "gwp_version": self.gwp_version,
            "dqi_score": str(self.dqi_score),
            "uncertainty_pct": str(self.uncertainty_pct),
            "calculation_trace_count": len(self.calculation_trace),
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
        }


class TreatmentResult:
    """
    Result from a single treatment pathway calculation.

    Attributes:
        treatment: Treatment pathway name.
        fossil_co2e_kg: Fossil CO2e emissions (kgCO2e).
        biogenic_co2_kg: Biogenic CO2 (kgCO2, memo item).
        ch4_kg: Methane emissions (kg CH4).
        n2o_kg: N2O emissions (kg N2O).
        avoided_co2e_kg: Avoided emissions from recycling/energy recovery (kgCO2e).
        mass_kg: Mass of waste treated (kg).
        trace: Calculation trace steps.
    """

    def __init__(self, treatment: str) -> None:
        """Initialize with zero values for a given treatment."""
        self.treatment: str = treatment
        self.fossil_co2e_kg: Decimal = _ZERO
        self.biogenic_co2_kg: Decimal = _ZERO
        self.ch4_kg: Decimal = _ZERO
        self.n2o_kg: Decimal = _ZERO
        self.avoided_co2e_kg: Decimal = _ZERO
        self.energy_recovery_credit_kg: Decimal = _ZERO
        self.mass_kg: Decimal = _ZERO
        self.trace: List[Dict[str, Any]] = []


class MaterialStreamResult:
    """
    Result from decomposing a product into material streams.

    Attributes:
        material: Material type.
        mass_kg: Mass of this material stream (kg).
        treatment_results: Per-treatment results for this material.
    """

    def __init__(self, material: str, mass_kg: Decimal) -> None:
        """Initialize for a material stream."""
        self.material: str = material
        self.mass_kg: Decimal = mass_kg
        self.treatment_results: Dict[str, TreatmentResult] = {}


# ==============================================================================
# WASTE TYPE SPECIFIC CALCULATOR ENGINE - SINGLETON
# ==============================================================================


class WasteTypeSpecificCalculatorEngine:
    """
    Thread-safe singleton engine for waste-type-specific EOL emission calculations.

    Implements Method A (waste-type-specific) from GHG Protocol Scope 3
    Category 12 Technical Guidance. Calculates emissions by decomposing
    sold products into material streams, applying regional treatment mix
    fractions, and computing treatment-specific emissions for each pathway.

    Treatment Pathways:
        1. Landfill - IPCC FOD model (DDOCm, CH4 generation, recovery, oxidation)
        2. Incineration - IPCC mass balance (fossil CO2 + biogenic CO2 + N2O)
        3. Recycling - Cut-off (processing EFs; avoided credits separate)
        4. Composting - CH4 + N2O from aerobic decomposition
        5. Anaerobic Digestion - Fugitive CH4 from biogas leakage
        6. Open Burning - CO2 + CH4 + N2O per material type

    Thread Safety:
        Uses __new__ singleton with threading.Lock() for instance creation
        and threading.RLock() for all calculation state.

    Zero-Hallucination:
        All calculations use deterministic Decimal arithmetic.
        Emission factors from EOLProductDatabaseEngine only.
        No LLM calls anywhere in the calculation pipeline.

    Provenance:
        SHA-256 hash computed on inputs + outputs for every calculation.
        Complete calculation trace recorded for audit trails.

    Attributes:
        _db: EOLProductDatabaseEngine singleton for factor lookups.
        _calc_count: Total calculations performed.

    Example:
        >>> engine = WasteTypeSpecificCalculatorEngine()
        >>> products = [{"product_category": "electronics", "units_sold": 1000, ...}]
        >>> result = engine.calculate(products, "ORG-001", 2025)
        >>> assert result["total_co2e_kg"] > Decimal("0")
    """

    _instance: Optional["WasteTypeSpecificCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "WasteTypeSpecificCalculatorEngine":
        """Thread-safe singleton instantiation via double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the WasteTypeSpecificCalculatorEngine (called once)."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        start_ts = time.monotonic()
        self._initialized: bool = True
        self._rlock: threading.RLock = threading.RLock()
        self._db: EOLProductDatabaseEngine = EOLProductDatabaseEngine()
        self._calc_count: int = 0
        self._last_calculation: Optional[datetime] = None
        self._default_gwp_version: str = DEFAULT_GWP_VERSION

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        logger.info(
            "WasteTypeSpecificCalculatorEngine initialized: "
            "agent=%s, engine=%s, version=%s, gwp=%s, elapsed_ms=%.2f",
            AGENT_ID, ENGINE_ID, ENGINE_VERSION,
            self._default_gwp_version, elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Singleton management
    # ------------------------------------------------------------------

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        This method is intended exclusively for unit tests that need
        a fresh engine instance. It should never be called in production.
        """
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment_calc_count(self) -> None:
        """Thread-safe increment of calculation counter."""
        with self._rlock:
            self._calc_count += 1
            self._last_calculation = datetime.now(timezone.utc)

    def _quantize(self, value: Decimal, precision: str = "0.00000001") -> Decimal:
        """
        Quantize a Decimal to the specified precision with ROUND_HALF_UP.

        Args:
            value: Decimal value to quantize.
            precision: Precision string (default 8 decimal places).

        Returns:
            Quantized Decimal.
        """
        return value.quantize(Decimal(precision), rounding=ROUND_HALF_UP)

    def _safe_decimal(self, value: Any, field_name: str = "value") -> Decimal:
        """
        Safely convert a value to Decimal.

        Args:
            value: Value to convert (str, int, float, Decimal).
            field_name: Name of the field for error messages.

        Returns:
            Decimal value.

        Raises:
            ValueError: If value cannot be converted.
        """
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert {field_name}={value!r} to Decimal: {e}"
            ) from e

    # ==================================================================
    # 1. MAIN CALCULATION ENTRY POINT
    # ==================================================================

    def calculate(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        reporting_year: int,
        gwp_version: Optional[str] = None,
        region_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate total EOL emissions for a list of sold products.

        This is the primary entry point for waste-type-specific calculations.
        Each product is decomposed into material streams, then each stream
        is allocated to treatment pathways based on the regional mix.

        Formula:
            E_total = SUM_products [ SUM_materials [ SUM_treatments [
                units * weight * mat_fraction * treatment_fraction * EF
            ]]]

        Args:
            products: List of product dicts, each containing:
                - product_id (str): Unique product identifier.
                - product_category (str): One of VALID_PRODUCT_CATEGORIES.
                - units_sold (int): Number of units sold in reporting year.
                - weight_per_unit_kg (Decimal|str|float): Weight per unit in kg.
                  Optional: uses default weight if not provided.
                - region (str): Region code for treatment mix. Optional:
                  uses region_override or "GLOBAL" if not provided.
                - composition (Dict[str,Decimal]): Custom material composition.
                  Optional: uses default composition if not provided.
                - treatment_mix (Dict[str,Decimal]): Custom treatment mix.
                  Optional: uses regional default if not provided.
            org_id: Organization identifier.
            reporting_year: Reporting year (e.g., 2025).
            gwp_version: GWP assessment report version ("AR4","AR5","AR6").
                        Defaults to AR5.
            region_override: Override region for all products.

        Returns:
            Dict containing:
                - total_co2e_kg: Total fossil CO2e (kgCO2e)
                - total_biogenic_co2_kg: Biogenic CO2 (kgCO2, memo)
                - total_avoided_co2e_kg: Avoided emissions (kgCO2e, separate)
                - total_energy_recovery_credit_kg: Energy recovery (kgCO2e, separate)
                - treatment_breakdown: Per-treatment pathway totals
                - product_breakdown: Per-product totals
                - material_breakdown: Per-material totals
                - total_mass_kg: Total waste mass (kg)
                - calculation_id: Unique ID
                - provenance_hash: SHA-256 audit hash
                - processing_time_ms: Duration
                - dqi_score: Data quality score
                - uncertainty_pct: Uncertainty range

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If calculation encounters an error.

        Example:
            >>> result = engine.calculate(products, "ORG-001", 2025)
            >>> result["total_co2e_kg"]
            Decimal('12345.678')
        """
        start_ts = time.monotonic()
        self._increment_calc_count()
        effective_gwp = gwp_version or self._default_gwp_version
        calc_id = str(uuid.uuid4())

        logger.info(
            "calculate() calc_id=%s, org=%s, year=%d, products=%d, gwp=%s",
            calc_id, org_id, reporting_year, len(products), effective_gwp,
        )

        try:
            # Step 1: Validate inputs
            validated_products = self._validate_products(products)

            # Step 2: Initialize result accumulator
            result = EOLCalculationResult()
            result.org_id = org_id
            result.reporting_year = reporting_year
            result.gwp_version = effective_gwp
            result.calculation_id = calc_id

            # Step 3: Process each product
            for product in validated_products:
                product_result = self._calculate_product(
                    product=product,
                    gwp_version=effective_gwp,
                    region_override=region_override,
                )
                self._accumulate_product_result(result, product_result, product)

            # Step 4: Compute DQI score
            result.dqi_score = self._compute_dqi_score(validated_products)
            result.uncertainty_pct = self._compute_uncertainty(validated_products, result)

            # Step 5: Quantize final values
            result.total_co2e_kg = self._quantize(result.total_co2e_kg)
            result.total_biogenic_co2_kg = self._quantize(result.total_biogenic_co2_kg)
            result.total_avoided_co2e_kg = self._quantize(result.total_avoided_co2e_kg)
            result.total_energy_recovery_credit_kg = self._quantize(
                result.total_energy_recovery_credit_kg
            )
            result.total_mass_kg = self._quantize(result.total_mass_kg)

            # Step 6: Compute provenance hash
            result.provenance_hash = self._compute_provenance_hash(
                products, result, org_id, reporting_year,
            )

            # Step 7: Record processing time
            elapsed_ms = (time.monotonic() - start_ts) * 1000.0
            result.processing_time_ms = round(elapsed_ms, 2)

            logger.info(
                "calculate() COMPLETE: calc_id=%s, total_co2e=%s kg, "
                "biogenic=%s kg, avoided=%s kg, mass=%s kg, "
                "products=%d, elapsed_ms=%.2f",
                calc_id, result.total_co2e_kg, result.total_biogenic_co2_kg,
                result.total_avoided_co2e_kg, result.total_mass_kg,
                len(products), elapsed_ms,
            )

            return result.to_dict()

        except ValueError as e:
            logger.error("Validation error in calculate(): %s", e)
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in calculate(): calc_id=%s, error=%s",
                calc_id, e, exc_info=True,
            )
            raise RuntimeError(
                f"EOL calculation failed for calc_id={calc_id}: {e}"
            ) from e

    # ==================================================================
    # 2. BATCH PROCESSING
    # ==================================================================

    def calculate_batch(
        self,
        product_batches: List[Dict[str, Any]],
        batch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple organization/year combinations in batches.

        Each batch item should contain:
            - org_id (str): Organization identifier.
            - reporting_year (int): Reporting year.
            - products (List[Dict]): List of product dicts.
            - gwp_version (str, optional): GWP version override.
            - region_override (str, optional): Region override.

        Args:
            product_batches: List of batch items.
            batch_size: Maximum items per processing chunk.

        Returns:
            List of result dicts, one per batch item.
            Failed items include an "error" key.

        Example:
            >>> batches = [
            ...     {"org_id": "ORG-001", "reporting_year": 2025, "products": [...]},
            ...     {"org_id": "ORG-002", "reporting_year": 2025, "products": [...]},
            ... ]
            >>> results = engine.calculate_batch(batches)
        """
        start_ts = time.monotonic()
        total = len(product_batches)

        if total > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {total} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        logger.info(
            "calculate_batch(): items=%d, batch_size=%d",
            total, batch_size,
        )

        results: List[Dict[str, Any]] = []
        success_count = 0
        error_count = 0

        for i in range(0, total, batch_size):
            chunk = product_batches[i : i + batch_size]
            for item in chunk:
                try:
                    org_id = item.get("org_id", "UNKNOWN")
                    year = item.get("reporting_year", 0)
                    products = item.get("products", [])
                    gwp = item.get("gwp_version")
                    region = item.get("region_override")

                    result = self.calculate(
                        products=products,
                        org_id=org_id,
                        reporting_year=year,
                        gwp_version=gwp,
                        region_override=region,
                    )
                    results.append(result)
                    success_count += 1

                except Exception as e:
                    logger.error(
                        "Batch item failed: org=%s, year=%s, error=%s",
                        item.get("org_id"), item.get("reporting_year"), e,
                    )
                    results.append({
                        "org_id": item.get("org_id", "UNKNOWN"),
                        "reporting_year": item.get("reporting_year", 0),
                        "error": str(e),
                        "status": "failed",
                    })
                    error_count += 1

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        logger.info(
            "calculate_batch() COMPLETE: total=%d, success=%d, errors=%d, "
            "elapsed_ms=%.2f",
            total, success_count, error_count, elapsed_ms,
        )

        return results

    # ==================================================================
    # 3. INPUT VALIDATION
    # ==================================================================

    def _validate_products(
        self,
        products: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Validate and normalize product inputs.

        Each product must have at minimum product_category and units_sold.
        Weight, composition, and treatment mix can be defaulted from database.

        Args:
            products: Raw product input list.

        Returns:
            Validated and normalized product list.

        Raises:
            ValueError: If any product fails validation.
        """
        if not products:
            raise ValueError("Products list cannot be empty")

        if len(products) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Products count {len(products)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        validated: List[Dict[str, Any]] = []
        errors: List[str] = []

        for idx, product in enumerate(products):
            product_errors = self._validate_single_product(product, idx)
            if product_errors:
                errors.extend(product_errors)
            else:
                normalized = self._normalize_product(product)
                validated.append(normalized)

        if errors:
            raise ValueError(
                f"Product validation failed with {len(errors)} error(s): "
                + "; ".join(errors[:10])
            )

        return validated

    def _validate_single_product(
        self,
        product: Dict[str, Any],
        index: int,
    ) -> List[str]:
        """
        Validate a single product input.

        Args:
            product: Product dict.
            index: Index in the products list (for error messages).

        Returns:
            List of error messages. Empty if valid.
        """
        errors: List[str] = []
        prefix = f"Product[{index}]"

        # Required: product_category
        category = product.get("product_category", "")
        if not category:
            errors.append(f"{prefix}: 'product_category' is required")
        elif category.lower().strip() not in VALID_PRODUCT_CATEGORIES:
            errors.append(
                f"{prefix}: Invalid product_category '{category}'. "
                f"Must be one of: {sorted(VALID_PRODUCT_CATEGORIES)}"
            )

        # Required: units_sold
        units = product.get("units_sold")
        if units is None:
            errors.append(f"{prefix}: 'units_sold' is required")
        else:
            try:
                units_int = int(units)
                if units_int <= 0:
                    errors.append(f"{prefix}: 'units_sold' must be > 0, got {units_int}")
            except (ValueError, TypeError):
                errors.append(f"{prefix}: 'units_sold' must be an integer, got {units!r}")

        # Optional: weight_per_unit_kg (if provided, must be positive)
        weight = product.get("weight_per_unit_kg")
        if weight is not None:
            try:
                w = Decimal(str(weight))
                if w <= _ZERO:
                    errors.append(f"{prefix}: 'weight_per_unit_kg' must be > 0, got {w}")
            except (InvalidOperation, ValueError, TypeError):
                errors.append(
                    f"{prefix}: 'weight_per_unit_kg' must be numeric, got {weight!r}"
                )

        # Optional: region
        region = product.get("region")
        if region is not None:
            if region.upper().strip() not in VALID_REGIONS:
                errors.append(
                    f"{prefix}: Invalid region '{region}'. "
                    f"Must be one of: {sorted(VALID_REGIONS)}"
                )

        # Optional: composition (if provided, must sum to 1.0 and use valid materials)
        composition = product.get("composition")
        if composition is not None:
            if not isinstance(composition, dict):
                errors.append(f"{prefix}: 'composition' must be a dict")
            else:
                total = _ZERO
                for mat, frac in composition.items():
                    if mat.lower().strip() not in VALID_MATERIALS:
                        errors.append(f"{prefix}: Invalid material '{mat}' in composition")
                    try:
                        total += Decimal(str(frac))
                    except (InvalidOperation, ValueError, TypeError):
                        errors.append(
                            f"{prefix}: Invalid fraction for material '{mat}': {frac!r}"
                        )
                if abs(total - _ONE) > Decimal("0.01"):
                    errors.append(
                        f"{prefix}: Composition fractions sum to {total}, expected 1.0"
                    )

        # Optional: treatment_mix (if provided, must sum to 1.0)
        treatment_mix = product.get("treatment_mix")
        if treatment_mix is not None:
            if not isinstance(treatment_mix, dict):
                errors.append(f"{prefix}: 'treatment_mix' must be a dict")
            else:
                total = _ZERO
                for trt, frac in treatment_mix.items():
                    if trt.lower().strip() not in VALID_TREATMENTS:
                        errors.append(
                            f"{prefix}: Invalid treatment '{trt}' in treatment_mix"
                        )
                    try:
                        total += Decimal(str(frac))
                    except (InvalidOperation, ValueError, TypeError):
                        errors.append(
                            f"{prefix}: Invalid fraction for treatment '{trt}': {frac!r}"
                        )
                if abs(total - _ONE) > Decimal("0.01"):
                    errors.append(
                        f"{prefix}: Treatment mix fractions sum to {total}, expected 1.0"
                    )

        return errors

    def _normalize_product(
        self,
        product: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize product input, filling in defaults from database.

        Args:
            product: Validated product dict.

        Returns:
            Normalized product dict with all fields populated.
        """
        category = product["product_category"].lower().strip()
        region = product.get("region", "GLOBAL").upper().strip()

        # Get weight: user-provided or default
        if product.get("weight_per_unit_kg") is not None:
            weight = self._safe_decimal(product["weight_per_unit_kg"], "weight_per_unit_kg")
        else:
            weight = self._db.get_product_weight(category)

        # Get composition: user-provided or default
        if product.get("composition") is not None:
            composition = {
                k.lower().strip(): self._safe_decimal(v, f"composition.{k}")
                for k, v in product["composition"].items()
            }
        else:
            composition = self._db.get_product_composition(category)

        # Get treatment mix: user-provided or regional default
        if product.get("treatment_mix") is not None:
            treatment_mix = {
                k.lower().strip(): self._safe_decimal(v, f"treatment_mix.{k}")
                for k, v in product["treatment_mix"].items()
            }
        else:
            treatment_mix = self._db.get_regional_treatment_mix(region)

        return {
            "product_id": product.get("product_id", str(uuid.uuid4())),
            "product_category": category,
            "units_sold": int(product["units_sold"]),
            "weight_per_unit_kg": weight,
            "region": region,
            "composition": composition,
            "treatment_mix": treatment_mix,
            "climate_zone": product.get("climate_zone", "temperate_wet"),
            "landfill_type": product.get("landfill_type", "managed_anaerobic"),
            "has_energy_recovery": product.get("has_energy_recovery", False),
        }

    # ==================================================================
    # 4. PER-PRODUCT CALCULATION
    # ==================================================================

    def _calculate_product(
        self,
        product: Dict[str, Any],
        gwp_version: str,
        region_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate EOL emissions for a single product.

        Decomposes the product into material streams, then calculates
        emissions for each material under each treatment pathway.

        Args:
            product: Normalized product dict.
            gwp_version: GWP version.
            region_override: Optional region override.

        Returns:
            Dict with per-material, per-treatment emissions.
        """
        start_ts = time.monotonic()
        region = region_override or product["region"]
        units = Decimal(str(product["units_sold"]))
        weight = product["weight_per_unit_kg"]
        total_mass_kg = units * weight

        # Decompose into material streams
        material_streams = self._decompose_product(product)

        # Calculate per-material per-treatment
        product_co2e = _ZERO
        product_biogenic = _ZERO
        product_avoided = _ZERO
        product_energy_credit = _ZERO
        product_treatment_breakdown: Dict[str, Decimal] = {}
        product_material_breakdown: Dict[str, Decimal] = {}
        product_trace: List[Dict[str, Any]] = []

        for stream in material_streams:
            material = stream["material"]
            material_mass = stream["mass_kg"]
            material_co2e = _ZERO

            for treatment, treatment_fraction in product["treatment_mix"].items():
                treatment_mass = material_mass * treatment_fraction
                if treatment_mass <= _ZERO:
                    continue

                # Calculate treatment-specific emissions
                trt_result = self._calculate_treatment_emissions(
                    material=material,
                    treatment=treatment,
                    mass_kg=treatment_mass,
                    gwp_version=gwp_version,
                    region=region,
                    climate_zone=product.get("climate_zone", "temperate_wet"),
                    landfill_type=product.get("landfill_type", "managed_anaerobic"),
                    has_energy_recovery=product.get("has_energy_recovery", False),
                )

                # Accumulate results
                product_co2e += trt_result.fossil_co2e_kg
                product_biogenic += trt_result.biogenic_co2_kg
                product_avoided += trt_result.avoided_co2e_kg
                product_energy_credit += trt_result.energy_recovery_credit_kg
                material_co2e += trt_result.fossil_co2e_kg

                # Treatment breakdown
                if treatment not in product_treatment_breakdown:
                    product_treatment_breakdown[treatment] = _ZERO
                product_treatment_breakdown[treatment] += trt_result.fossil_co2e_kg

                # Trace
                product_trace.extend(trt_result.trace)

            # Material breakdown
            product_material_breakdown[material] = material_co2e

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0

        return {
            "product_id": product["product_id"],
            "product_category": product["product_category"],
            "units_sold": product["units_sold"],
            "total_mass_kg": total_mass_kg,
            "fossil_co2e_kg": product_co2e,
            "biogenic_co2_kg": product_biogenic,
            "avoided_co2e_kg": product_avoided,
            "energy_recovery_credit_kg": product_energy_credit,
            "treatment_breakdown": product_treatment_breakdown,
            "material_breakdown": product_material_breakdown,
            "trace": product_trace,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    # ==================================================================
    # 5. PRODUCT DECOMPOSITION INTO MATERIAL STREAMS
    # ==================================================================

    def decompose_product(
        self,
        product: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Public API: Decompose a product into material streams.

        Given a product with units_sold, weight, and composition,
        returns a list of material streams with calculated mass.

        Args:
            product: Product dict with product_category, units_sold,
                    weight_per_unit_kg (optional), composition (optional).

        Returns:
            List of material stream dicts with material and mass_kg.

        Example:
            >>> streams = engine.decompose_product({
            ...     "product_category": "electronics",
            ...     "units_sold": 1000,
            ...     "weight_per_unit_kg": Decimal("2.5"),
            ... })
            >>> streams[0]
            {'material': 'plastic', 'mass_kg': Decimal('1000.000')}
        """
        normalized = self._normalize_product(product)
        return self._decompose_product(normalized)

    def _decompose_product(
        self,
        product: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Internal: Decompose product into material streams.

        Args:
            product: Normalized product dict.

        Returns:
            List of material stream dicts: [{"material": str, "mass_kg": Decimal}]
        """
        units = Decimal(str(product["units_sold"]))
        weight = product["weight_per_unit_kg"]
        composition = product["composition"]
        total_mass = units * weight

        streams: List[Dict[str, Any]] = []
        for material, fraction in composition.items():
            mass_kg = self._quantize(total_mass * fraction)
            if mass_kg > _ZERO:
                streams.append({
                    "material": material,
                    "mass_kg": mass_kg,
                    "fraction": fraction,
                })

        logger.debug(
            "decompose_product: category=%s, total_mass=%.2f kg, streams=%d",
            product["product_category"], total_mass, len(streams),
        )
        return streams

    # ==================================================================
    # 6. APPLY REGIONAL TREATMENT MIX
    # ==================================================================

    def apply_regional_treatment_mix(
        self,
        product: Dict[str, Any],
        region: str,
    ) -> Dict[str, Any]:
        """
        Apply regional treatment mix to a product.

        Overrides any existing treatment_mix on the product with
        the regional default from the database.

        Args:
            product: Product dict.
            region: Region code (e.g., "US", "EU").

        Returns:
            Updated product dict with regional treatment mix.

        Example:
            >>> product = {"product_category": "electronics", "units_sold": 100}
            >>> updated = engine.apply_regional_treatment_mix(product, "JP")
            >>> updated["treatment_mix"]["incineration"]
            Decimal('0.79')
        """
        mix = self._db.get_regional_treatment_mix(region)
        product_copy = dict(product)
        product_copy["treatment_mix"] = mix
        product_copy["region"] = region.upper().strip()
        return product_copy

    # ==================================================================
    # 7. TREATMENT-SPECIFIC EMISSION CALCULATIONS
    # ==================================================================

    def _calculate_treatment_emissions(
        self,
        material: str,
        treatment: str,
        mass_kg: Decimal,
        gwp_version: str,
        region: str = "GLOBAL",
        climate_zone: str = "temperate_wet",
        landfill_type: str = "managed_anaerobic",
        has_energy_recovery: bool = False,
    ) -> TreatmentResult:
        """
        Route to treatment-specific calculation method.

        Args:
            material: Material type.
            treatment: Treatment pathway.
            mass_kg: Mass of waste in kg.
            gwp_version: GWP version.
            region: Region code for energy recovery.
            climate_zone: IPCC climate zone for landfill.
            landfill_type: Landfill management type.
            has_energy_recovery: Whether WtE credits apply.

        Returns:
            TreatmentResult with emissions breakdown.
        """
        if treatment == "landfill":
            return self._calculate_landfill(
                material, mass_kg, gwp_version, climate_zone, landfill_type,
            )
        elif treatment == "incineration":
            return self._calculate_incineration(
                material, mass_kg, gwp_version, region, has_energy_recovery,
            )
        elif treatment == "recycling":
            return self._calculate_recycling(material, mass_kg, gwp_version)
        elif treatment == "composting":
            return self._calculate_composting(material, mass_kg, gwp_version)
        elif treatment == "anaerobic_digestion":
            return self._calculate_anaerobic_digestion(material, mass_kg, gwp_version)
        elif treatment == "open_burning":
            return self._calculate_open_burning(material, mass_kg, gwp_version)
        else:
            # Fallback: use simple EF lookup
            return self._calculate_simple_ef(material, treatment, mass_kg, gwp_version)

    # ------------------------------------------------------------------
    # 7a. LANDFILL - IPCC First Order Decay
    # ------------------------------------------------------------------

    def _calculate_landfill(
        self,
        material: str,
        mass_kg: Decimal,
        gwp_version: str,
        climate_zone: str,
        landfill_type: str,
    ) -> TreatmentResult:
        """
        Calculate landfill emissions using IPCC FOD model.

        Steps:
            1. DDOCm = W x DOC x DOCf x MCF
            2. CH4_generated = DDOCm x F x (16/12)
            3. CH4_recovered = CH4_generated x gas_collection_efficiency
            4. CH4_emitted = (CH4_generated - CH4_recovered) x (1 - OX)
            5. CO2e = CH4_emitted x GWP_CH4

        Args:
            material: Material type.
            mass_kg: Mass of waste (kg).
            gwp_version: GWP version.
            climate_zone: IPCC climate zone.
            landfill_type: Landfill management type.

        Returns:
            TreatmentResult with CH4-based CO2e.
        """
        result = TreatmentResult("landfill")
        result.mass_kg = mass_kg

        # Get FOD parameters from database
        fod_params = self._db.get_landfill_fod_params(material, climate_zone)
        doc = fod_params["doc"]
        docf = fod_params["docf"]
        f_ch4 = fod_params["f_ch4"]
        ox = fod_params["ox_with_cover"]  # Assume managed with cover

        # Get MCF and gas collection
        mcf = self._db.get_mcf(landfill_type)
        gas_eff = self._db.get_gas_collection_efficiency(landfill_type)

        # Get GWP
        gwp_ch4 = self._db.get_gwp_ch4(gwp_version)

        # Step 1: DDOCm - mass of decomposable DOC deposited
        # Convert mass from kg to kg (no conversion needed for per-kg factors)
        ddocm = mass_kg * doc * docf * mcf
        ddocm = self._quantize(ddocm)

        # For first-year simplified calculation (steady-state assumption):
        # Assume all DDOCm decomposes in the assessment period.
        ddocm_decomposed = ddocm

        # Step 2: CH4 generated
        ch4_generated = ddocm_decomposed * f_ch4 * _CH4_C_RATIO
        ch4_generated = self._quantize(ch4_generated)

        # Step 3: CH4 recovered
        ch4_recovered = ch4_generated * gas_eff
        ch4_recovered = self._quantize(ch4_recovered)

        # Step 4: CH4 emitted
        ch4_emitted = (ch4_generated - ch4_recovered) * (_ONE - ox)
        ch4_emitted = self._quantize(ch4_emitted)

        # Step 5: CO2e
        co2e = ch4_emitted * gwp_ch4
        co2e = self._quantize(co2e)

        result.ch4_kg = ch4_emitted
        result.fossil_co2e_kg = co2e

        # Biogenic CO2 from landfill gas (CO2 portion)
        # CO2 in landfill gas = DDOCm_decomposed x (1-F) x (44/12)
        biogenic_co2 = ddocm_decomposed * (_ONE - f_ch4) * _CO2_C_RATIO
        result.biogenic_co2_kg = self._quantize(biogenic_co2)

        # Trace
        result.trace.append({
            "step": "landfill_fod",
            "material": material,
            "mass_kg": str(mass_kg),
            "doc": str(doc),
            "docf": str(docf),
            "mcf": str(mcf),
            "f_ch4": str(f_ch4),
            "ox": str(ox),
            "gas_collection_eff": str(gas_eff),
            "ddocm": str(ddocm),
            "ch4_generated_kg": str(ch4_generated),
            "ch4_recovered_kg": str(ch4_recovered),
            "ch4_emitted_kg": str(ch4_emitted),
            "gwp_ch4": str(gwp_ch4),
            "fossil_co2e_kg": str(co2e),
            "biogenic_co2_kg": str(result.biogenic_co2_kg),
            "climate_zone": climate_zone,
            "landfill_type": landfill_type,
            "gwp_version": gwp_version,
        })

        return result

    # ------------------------------------------------------------------
    # 7b. INCINERATION - IPCC Mass Balance
    # ------------------------------------------------------------------

    def _calculate_incineration(
        self,
        material: str,
        mass_kg: Decimal,
        gwp_version: str,
        region: str,
        has_energy_recovery: bool,
    ) -> TreatmentResult:
        """
        Calculate incineration emissions using IPCC mass balance.

        Steps:
            1. CO2_fossil = mass x dm x CF x FCF x OF x (44/12)
            2. CO2_biogenic = mass x dm x CF x (1-FCF) x OF x (44/12) (memo)
            3. N2O = mass x N2O_ef (default 0.00004 for MSW)
            4. CO2e = CO2_fossil + (N2O x GWP_N2O)
            5. Energy_recovery_credit = mass x CV x WtE_eff x grid_EF (separate)

        Args:
            material: Material type.
            mass_kg: Mass of waste (kg).
            gwp_version: GWP version.
            region: Region for energy recovery factors.
            has_energy_recovery: Whether to calculate WtE credits.

        Returns:
            TreatmentResult with fossil CO2, biogenic CO2, N2O, and
            optional energy recovery credit (all separate).
        """
        result = TreatmentResult("incineration")
        result.mass_kg = mass_kg

        # Get incineration parameters
        params = self._db.get_incineration_params(material)
        dm = params["dry_matter_fraction"]
        cf = params["carbon_fraction"]
        fcf = params["fossil_carbon_fraction"]
        of = params["oxidation_factor"]

        # Get GWP
        gwp_n2o = self._db.get_gwp_n2o(gwp_version)

        # Step 1: Fossil CO2
        co2_fossil = mass_kg * dm * cf * fcf * of * _CO2_C_RATIO
        co2_fossil = self._quantize(co2_fossil)

        # Step 2: Biogenic CO2 (memo item, reported separately)
        co2_biogenic = mass_kg * dm * cf * (_ONE - fcf) * of * _CO2_C_RATIO
        co2_biogenic = self._quantize(co2_biogenic)

        # Step 3: N2O emissions
        n2o_ef = self._db.get_n2o_incineration_ef()
        n2o_kg = mass_kg * n2o_ef
        n2o_kg = self._quantize(n2o_kg)

        # Step 4: Total fossil CO2e (fossil CO2 + N2O as CO2e)
        n2o_co2e = n2o_kg * gwp_n2o
        total_co2e = co2_fossil + n2o_co2e
        total_co2e = self._quantize(total_co2e)

        result.fossil_co2e_kg = total_co2e
        result.biogenic_co2_kg = co2_biogenic
        result.n2o_kg = n2o_kg

        # Step 5: Energy recovery credit (reported separately, NOT netted)
        energy_credit = _ZERO
        if has_energy_recovery:
            energy_credit = self._calculate_energy_recovery_credit(
                material, mass_kg, region,
            )
            result.energy_recovery_credit_kg = energy_credit

        # Trace
        result.trace.append({
            "step": "incineration_mass_balance",
            "material": material,
            "mass_kg": str(mass_kg),
            "dry_matter": str(dm),
            "carbon_fraction": str(cf),
            "fossil_carbon_fraction": str(fcf),
            "oxidation_factor": str(of),
            "co2_fossil_kg": str(co2_fossil),
            "co2_biogenic_kg": str(co2_biogenic),
            "n2o_kg": str(n2o_kg),
            "n2o_co2e_kg": str(n2o_co2e),
            "total_fossil_co2e_kg": str(total_co2e),
            "energy_recovery_credit_kg": str(energy_credit),
            "gwp_n2o": str(gwp_n2o),
            "gwp_version": gwp_version,
            "region": region,
            "has_energy_recovery": has_energy_recovery,
        })

        return result

    def _calculate_energy_recovery_credit(
        self,
        material: str,
        mass_kg: Decimal,
        region: str,
    ) -> Decimal:
        """
        Calculate energy recovery credit from waste-to-energy incineration.

        Credit = mass x calorific_value x WtE_efficiency x displaced_grid_EF
        Reported separately, NEVER netted from total.

        Args:
            material: Material type.
            mass_kg: Mass of waste (kg).
            region: Region for grid EF.

        Returns:
            Energy recovery credit in kgCO2e (positive value = avoided emissions).
        """
        cv = self._db.get_calorific_value(material)
        if cv <= _ZERO:
            return _ZERO

        recovery_factors = self._db.get_energy_recovery_factor(region)
        wte_eff = recovery_factors["wte_efficiency"]
        grid_ef = recovery_factors["displaced_grid_ef"]

        # Energy in MJ -> kWh -> multiply by grid EF
        energy_mj = mass_kg * cv
        energy_kwh = energy_mj * _MJ_TO_KWH
        electricity_kwh = energy_kwh * wte_eff
        credit = electricity_kwh * grid_ef

        return self._quantize(credit)

    # ------------------------------------------------------------------
    # 7c. RECYCLING - Cut-off approach
    # ------------------------------------------------------------------

    def _calculate_recycling(
        self,
        material: str,
        mass_kg: Decimal,
        gwp_version: str,
    ) -> TreatmentResult:
        """
        Calculate recycling emissions using the GHG Protocol cut-off approach.

        Under the cut-off approach:
        - Processing_emissions = mass x (transport_ef + mrf_ef)
        - Avoided_emissions = mass x avoided_ef (reported SEPARATELY)

        Args:
            material: Material type.
            mass_kg: Mass of waste (kg).
            gwp_version: GWP version (not used for recycling, but kept for consistency).

        Returns:
            TreatmentResult with processing emissions and separate avoided credits.
        """
        result = TreatmentResult("recycling")
        result.mass_kg = mass_kg

        # Processing emissions (transport + MRF sorting)
        processing_ef = self._db.get_recycling_processing_ef(material)
        processing_emissions = mass_kg * processing_ef
        processing_emissions = self._quantize(processing_emissions)

        result.fossil_co2e_kg = processing_emissions

        # Avoided emissions (virgin material substitution) - SEPARATE
        avoided_ef = self._db.get_avoided_emission_factor(material)
        avoided_emissions = mass_kg * abs(avoided_ef)  # Store as positive value
        avoided_emissions = self._quantize(avoided_emissions)

        result.avoided_co2e_kg = avoided_emissions

        # Trace
        result.trace.append({
            "step": "recycling_cutoff",
            "material": material,
            "mass_kg": str(mass_kg),
            "processing_ef_kg_per_kg": str(processing_ef),
            "processing_emissions_kg": str(processing_emissions),
            "avoided_ef_kg_per_kg": str(avoided_ef),
            "avoided_emissions_kg": str(avoided_emissions),
            "note": "Avoided emissions reported SEPARATELY per GHG Protocol",
        })

        return result

    # ------------------------------------------------------------------
    # 7d. COMPOSTING - Aerobic decomposition
    # ------------------------------------------------------------------

    def _calculate_composting(
        self,
        material: str,
        mass_kg: Decimal,
        gwp_version: str,
    ) -> TreatmentResult:
        """
        Calculate composting emissions from aerobic decomposition.

        CH4 = mass x CH4_ef (0.004 kg CH4/kg waste)
        N2O = mass x N2O_ef (0.0003 kg N2O/kg waste)
        CO2e = CH4 x GWP_CH4 + N2O x GWP_N2O

        Biogenic CO2 from composting is not counted per GHG Protocol
        (carbon neutral cycle for biogenic materials).

        Args:
            material: Material type.
            mass_kg: Mass of waste (kg).
            gwp_version: GWP version.

        Returns:
            TreatmentResult with CH4 and N2O emissions.
        """
        result = TreatmentResult("composting")
        result.mass_kg = mass_kg

        # Check if material is compostable
        composting_efs = self._db.get_composting_ef()
        ch4_ef = composting_efs["ch4_ef_kg_per_kg"]
        n2o_ef = composting_efs["n2o_ef_kg_per_kg"]

        # Non-compostable materials (metals, glass, etc.) have zero composting emissions
        non_compostable = {"plastic", "metal", "aluminum", "steel", "glass",
                          "electronics", "rubber", "ceramic", "concrete"}
        if material in non_compostable:
            result.trace.append({
                "step": "composting",
                "material": material,
                "mass_kg": str(mass_kg),
                "note": f"Material '{material}' is not compostable; zero emissions",
                "fossil_co2e_kg": "0",
            })
            return result

        # Get GWP values
        gwp_ch4 = self._db.get_gwp_ch4(gwp_version)
        gwp_n2o = self._db.get_gwp_n2o(gwp_version)

        # Calculate CH4 and N2O
        ch4_kg = mass_kg * ch4_ef
        ch4_kg = self._quantize(ch4_kg)

        n2o_kg = mass_kg * n2o_ef
        n2o_kg = self._quantize(n2o_kg)

        # CO2e
        ch4_co2e = ch4_kg * gwp_ch4
        n2o_co2e = n2o_kg * gwp_n2o
        total_co2e = ch4_co2e + n2o_co2e
        total_co2e = self._quantize(total_co2e)

        result.ch4_kg = ch4_kg
        result.n2o_kg = n2o_kg
        result.fossil_co2e_kg = total_co2e

        # Trace
        result.trace.append({
            "step": "composting",
            "material": material,
            "mass_kg": str(mass_kg),
            "ch4_ef_kg_per_kg": str(ch4_ef),
            "n2o_ef_kg_per_kg": str(n2o_ef),
            "ch4_kg": str(ch4_kg),
            "n2o_kg": str(n2o_kg),
            "ch4_co2e_kg": str(ch4_co2e),
            "n2o_co2e_kg": str(n2o_co2e),
            "total_co2e_kg": str(total_co2e),
            "gwp_ch4": str(gwp_ch4),
            "gwp_n2o": str(gwp_n2o),
            "gwp_version": gwp_version,
        })

        return result

    # ------------------------------------------------------------------
    # 7e. ANAEROBIC DIGESTION - Fugitive methane
    # ------------------------------------------------------------------

    def _calculate_anaerobic_digestion(
        self,
        material: str,
        mass_kg: Decimal,
        gwp_version: str,
    ) -> TreatmentResult:
        """
        Calculate anaerobic digestion emissions from fugitive methane.

        Biogas = mass x biogas_yield
        CH4_in_biogas = Biogas x CH4_fraction
        CH4_fugitive = CH4_in_biogas x (1 - capture_efficiency)
        CO2e = CH4_fugitive x GWP_CH4

        Args:
            material: Material type.
            mass_kg: Mass of waste (kg).
            gwp_version: GWP version.

        Returns:
            TreatmentResult with fugitive CH4 emissions.
        """
        result = TreatmentResult("anaerobic_digestion")
        result.mass_kg = mass_kg

        # Non-digestable materials
        non_digestable = {"plastic", "metal", "aluminum", "steel", "glass",
                         "electronics", "rubber", "ceramic", "concrete"}
        if material in non_digestable:
            result.trace.append({
                "step": "anaerobic_digestion",
                "material": material,
                "mass_kg": str(mass_kg),
                "note": f"Material '{material}' is not suitable for AD; zero emissions",
                "fossil_co2e_kg": "0",
            })
            return result

        # Get AD parameters
        ad_params = self._db.get_ad_ef()
        biogas_yield = ad_params["biogas_yield_m3_per_kg"]
        ch4_fraction = ad_params["ch4_fraction_in_biogas"]
        capture_eff = ad_params["default_capture_efficiency"]

        # Get GWP
        gwp_ch4 = self._db.get_gwp_ch4(gwp_version)

        # Calculate biogas production
        biogas_m3 = mass_kg * biogas_yield
        biogas_m3 = self._quantize(biogas_m3)

        # CH4 in biogas (convert m3 to kg: ~0.717 kg/m3 at STP)
        ch4_density = Decimal("0.717")  # kg CH4 per m3 at STP
        ch4_total_kg = biogas_m3 * ch4_fraction * ch4_density
        ch4_total_kg = self._quantize(ch4_total_kg)

        # Fugitive CH4 (leakage)
        ch4_fugitive = ch4_total_kg * (_ONE - capture_eff)
        ch4_fugitive = self._quantize(ch4_fugitive)

        # CO2e
        co2e = ch4_fugitive * gwp_ch4
        co2e = self._quantize(co2e)

        result.ch4_kg = ch4_fugitive
        result.fossil_co2e_kg = co2e

        # Trace
        result.trace.append({
            "step": "anaerobic_digestion",
            "material": material,
            "mass_kg": str(mass_kg),
            "biogas_yield_m3_per_kg": str(biogas_yield),
            "biogas_m3": str(biogas_m3),
            "ch4_fraction": str(ch4_fraction),
            "ch4_density_kg_per_m3": str(ch4_density),
            "ch4_total_kg": str(ch4_total_kg),
            "capture_efficiency": str(capture_eff),
            "ch4_fugitive_kg": str(ch4_fugitive),
            "gwp_ch4": str(gwp_ch4),
            "fossil_co2e_kg": str(co2e),
            "gwp_version": gwp_version,
        })

        return result

    # ------------------------------------------------------------------
    # 7f. OPEN BURNING - IPCC emission factors
    # ------------------------------------------------------------------

    def _calculate_open_burning(
        self,
        material: str,
        mass_kg: Decimal,
        gwp_version: str,
    ) -> TreatmentResult:
        """
        Calculate open burning emissions using IPCC factors.

        CO2_fossil + CH4 + N2O per material type.
        Biogenic CO2 reported separately.

        Args:
            material: Material type.
            mass_kg: Mass of waste (kg).
            gwp_version: GWP version.

        Returns:
            TreatmentResult with fossil CO2, biogenic CO2, CH4, N2O.
        """
        result = TreatmentResult("open_burning")
        result.mass_kg = mass_kg

        # Get open burning EFs
        ob_efs = self._db.get_open_burning_ef(material)

        # Get GWP values
        gwp_ch4 = self._db.get_gwp_ch4(gwp_version)
        gwp_n2o = self._db.get_gwp_n2o(gwp_version)

        # Calculate gas-by-gas
        co2_fossil = mass_kg * ob_efs.get("co2_fossil_kg_per_kg", _ZERO)
        co2_fossil = self._quantize(co2_fossil)

        co2_biogenic = mass_kg * ob_efs.get("co2_biogenic_kg_per_kg", _ZERO)
        co2_biogenic = self._quantize(co2_biogenic)

        ch4_kg = mass_kg * ob_efs.get("ch4_kg_per_kg", _ZERO)
        ch4_kg = self._quantize(ch4_kg)

        n2o_kg = mass_kg * ob_efs.get("n2o_kg_per_kg", _ZERO)
        n2o_kg = self._quantize(n2o_kg)

        # CO2e total (fossil only; biogenic separate)
        ch4_co2e = ch4_kg * gwp_ch4
        n2o_co2e = n2o_kg * gwp_n2o
        total_co2e = co2_fossil + ch4_co2e + n2o_co2e
        total_co2e = self._quantize(total_co2e)

        result.fossil_co2e_kg = total_co2e
        result.biogenic_co2_kg = co2_biogenic
        result.ch4_kg = ch4_kg
        result.n2o_kg = n2o_kg

        # Trace
        result.trace.append({
            "step": "open_burning",
            "material": material,
            "mass_kg": str(mass_kg),
            "co2_fossil_kg": str(co2_fossil),
            "co2_biogenic_kg": str(co2_biogenic),
            "ch4_kg": str(ch4_kg),
            "n2o_kg": str(n2o_kg),
            "ch4_co2e_kg": str(ch4_co2e),
            "n2o_co2e_kg": str(n2o_co2e),
            "total_fossil_co2e_kg": str(total_co2e),
            "gwp_ch4": str(gwp_ch4),
            "gwp_n2o": str(gwp_n2o),
            "gwp_version": gwp_version,
        })

        return result

    # ------------------------------------------------------------------
    # 7g. SIMPLE EF LOOKUP (fallback)
    # ------------------------------------------------------------------

    def _calculate_simple_ef(
        self,
        material: str,
        treatment: str,
        mass_kg: Decimal,
        gwp_version: str,
    ) -> TreatmentResult:
        """
        Fallback: Calculate emissions using simple EF lookup.

        CO2e = mass x EF

        Used when a treatment pathway does not have a specialized
        calculation method (e.g., emerging treatment technologies).

        Args:
            material: Material type.
            treatment: Treatment pathway.
            mass_kg: Mass of waste (kg).
            gwp_version: GWP version (for consistency).

        Returns:
            TreatmentResult with simple EF-based emissions.
        """
        result = TreatmentResult(treatment)
        result.mass_kg = mass_kg

        try:
            ef = self._db.get_material_ef(material, treatment)
        except ValueError:
            logger.warning(
                "No EF for material=%s, treatment=%s; using zero.",
                material, treatment,
            )
            ef = _ZERO

        # Only count positive EFs as emissions; negatives are avoided
        if ef > _ZERO:
            co2e = mass_kg * ef
            result.fossil_co2e_kg = self._quantize(co2e)
        elif ef < _ZERO:
            avoided = mass_kg * abs(ef)
            result.avoided_co2e_kg = self._quantize(avoided)

        result.trace.append({
            "step": "simple_ef_lookup",
            "material": material,
            "treatment": treatment,
            "mass_kg": str(mass_kg),
            "ef_kgco2e_per_kg": str(ef),
            "fossil_co2e_kg": str(result.fossil_co2e_kg),
            "avoided_co2e_kg": str(result.avoided_co2e_kg),
        })

        return result

    # ==================================================================
    # 8. RESULT ACCUMULATION
    # ==================================================================

    def _accumulate_product_result(
        self,
        total_result: EOLCalculationResult,
        product_result: Dict[str, Any],
        product: Dict[str, Any],
    ) -> None:
        """
        Accumulate a single product's results into the total.

        Args:
            total_result: The accumulating total result.
            product_result: Single product result dict.
            product: Original product dict (for ID/category).
        """
        total_result.total_co2e_kg += product_result["fossil_co2e_kg"]
        total_result.total_biogenic_co2_kg += product_result["biogenic_co2_kg"]
        total_result.total_avoided_co2e_kg += product_result["avoided_co2e_kg"]
        total_result.total_energy_recovery_credit_kg += product_result["energy_recovery_credit_kg"]
        total_result.total_mass_kg += product_result["total_mass_kg"]

        # Treatment breakdown
        for treatment, co2e in product_result["treatment_breakdown"].items():
            if treatment not in total_result.treatment_breakdown:
                total_result.treatment_breakdown[treatment] = _ZERO
            total_result.treatment_breakdown[treatment] += co2e

        # Product breakdown
        pid = product_result["product_id"]
        total_result.product_breakdown[pid] = {
            "product_category": product_result["product_category"],
            "units_sold": str(product_result["units_sold"]),
            "total_mass_kg": str(product_result["total_mass_kg"]),
            "fossil_co2e_kg": str(product_result["fossil_co2e_kg"]),
            "biogenic_co2_kg": str(product_result["biogenic_co2_kg"]),
            "avoided_co2e_kg": str(product_result["avoided_co2e_kg"]),
        }

        # Material breakdown
        for material, co2e in product_result["material_breakdown"].items():
            if material not in total_result.material_breakdown:
                total_result.material_breakdown[material] = _ZERO
            total_result.material_breakdown[material] += co2e

        # Calculation trace
        total_result.calculation_trace.extend(product_result.get("trace", []))

    # ==================================================================
    # 9. DATA QUALITY INDICATOR (DQI)
    # ==================================================================

    def compute_dqi_score(
        self,
        products: List[Dict[str, Any]],
    ) -> Decimal:
        """
        Public API: Compute Data Quality Indicator score.

        Scores range from 1 (very good) to 5 (very poor) across five
        dimensions: temporal, geographical, technological, completeness,
        reliability.

        Args:
            products: List of product dicts.

        Returns:
            Overall DQI score (Decimal, 1.0-5.0).
        """
        return self._compute_dqi_score(products)

    def _compute_dqi_score(
        self,
        products: List[Dict[str, Any]],
    ) -> Decimal:
        """
        Internal: Compute Data Quality Indicator score.

        Scoring criteria:
        - Custom composition provided: temporal=2, else temporal=3
        - Custom treatment mix provided: geographical=2, else geographical=3
        - Custom weight provided: technological=2, else technological=3
        - All products have IDs: completeness=2, else completeness=3
        - More than 10 products: reliability=2, else reliability=3

        Returns:
            Average of 5 dimension scores.
        """
        temporal_scores: List[int] = []
        geographical_scores: List[int] = []
        technological_scores: List[int] = []
        completeness_scores: List[int] = []
        reliability_scores: List[int] = []

        for product in products:
            # Temporal: how current is the composition data?
            if product.get("composition"):
                temporal_scores.append(2)
            else:
                temporal_scores.append(3)

            # Geographical: region-specific treatment mix?
            if product.get("treatment_mix"):
                geographical_scores.append(2)
            else:
                geographical_scores.append(3)

            # Technological: product-specific weight?
            if product.get("weight_per_unit_kg"):
                technological_scores.append(2)
            else:
                technological_scores.append(3)

            # Completeness: has product ID?
            if product.get("product_id"):
                completeness_scores.append(2)
            else:
                completeness_scores.append(3)

            # Reliability: has region specified?
            if product.get("region"):
                reliability_scores.append(2)
            else:
                reliability_scores.append(3)

        def _avg(scores: List[int]) -> Decimal:
            if not scores:
                return Decimal("3")
            return Decimal(str(sum(scores))) / Decimal(str(len(scores)))

        overall = (
            _avg(temporal_scores)
            + _avg(geographical_scores)
            + _avg(technological_scores)
            + _avg(completeness_scores)
            + _avg(reliability_scores)
        ) / Decimal("5")

        return self._quantize(overall, "0.1")

    # ==================================================================
    # 10. UNCERTAINTY QUANTIFICATION
    # ==================================================================

    def compute_uncertainty(
        self,
        products: List[Dict[str, Any]],
        result: Optional[Dict[str, Any]] = None,
    ) -> Decimal:
        """
        Public API: Compute uncertainty range as percentage.

        Args:
            products: Product inputs.
            result: Optional calculation result for method-specific uncertainty.

        Returns:
            Uncertainty as +/- percentage (e.g., Decimal("30.0")).
        """
        # Wrap result dict in a simple object if needed
        class _MockResult:
            def __init__(self, data: Optional[Dict[str, Any]]) -> None:
                self.treatment_breakdown = {}
                if data:
                    for k, v in data.get("treatment_breakdown", {}).items():
                        self.treatment_breakdown[k] = Decimal(str(v))

        mock = _MockResult(result)
        return self._compute_uncertainty(products, mock)

    def _compute_uncertainty(
        self,
        products: List[Dict[str, Any]],
        result: Any,
    ) -> Decimal:
        """
        Internal: Compute method-specific uncertainty.

        Uncertainty ranges (IPCC Tier 1 defaults, +/-):
        - Landfill: +/- 40% (FOD model parameters)
        - Incineration: +/- 25% (well-characterized process)
        - Recycling: +/- 30% (processing only)
        - Composting: +/- 50% (variable conditions)
        - AD: +/- 35% (leakage uncertainty)
        - Open burning: +/- 60% (uncontrolled process)

        Overall uncertainty is the weighted average based on treatment
        fractions in the result.
        """
        treatment_uncertainties = {
            "landfill": Decimal("40"),
            "incineration": Decimal("25"),
            "recycling": Decimal("30"),
            "composting": Decimal("50"),
            "anaerobic_digestion": Decimal("35"),
            "open_burning": Decimal("60"),
        }

        if not hasattr(result, "treatment_breakdown") or not result.treatment_breakdown:
            return Decimal("30.0")

        total_emissions = sum(
            abs(v) for v in result.treatment_breakdown.values()
        )
        if total_emissions <= _ZERO:
            return Decimal("30.0")

        weighted_uncertainty = _ZERO
        for treatment, emissions in result.treatment_breakdown.items():
            fraction = abs(emissions) / total_emissions
            uncertainty = treatment_uncertainties.get(treatment, Decimal("30"))
            weighted_uncertainty += fraction * uncertainty

        return self._quantize(weighted_uncertainty, "0.1")

    # ==================================================================
    # 11. INPUT VALIDATION (Public API)
    # ==================================================================

    def validate_inputs(
        self,
        products: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate product inputs without performing calculations.

        Args:
            products: Product input list.

        Returns:
            Dict with "valid" (bool), "errors" (list), and "warnings" (list).

        Example:
            >>> validation = engine.validate_inputs([{"product_category": "electronics", ...}])
            >>> validation["valid"]
            True
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not products:
            errors.append("Products list cannot be empty")
            return {"valid": False, "errors": errors, "warnings": warnings}

        if len(products) > _MAX_BATCH_SIZE:
            errors.append(
                f"Products count {len(products)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        for idx, product in enumerate(products):
            product_errors = self._validate_single_product(product, idx)
            errors.extend(product_errors)

            # Warnings for missing optional fields
            prefix = f"Product[{idx}]"
            if not product.get("weight_per_unit_kg"):
                warnings.append(
                    f"{prefix}: No weight_per_unit_kg provided; "
                    f"using default for category"
                )
            if not product.get("region"):
                warnings.append(
                    f"{prefix}: No region provided; using GLOBAL defaults"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "product_count": len(products),
        }

    # ==================================================================
    # 12. PROVENANCE HASHING
    # ==================================================================

    def _compute_provenance_hash(
        self,
        products: List[Dict[str, Any]],
        result: EOLCalculationResult,
        org_id: str,
        reporting_year: int,
    ) -> str:
        """
        Compute SHA-256 provenance hash for the calculation.

        Hash covers: agent metadata, inputs, outputs, and timestamp.
        Creates an immutable audit record.

        Args:
            products: Input products.
            result: Calculation result.
            org_id: Organization ID.
            reporting_year: Reporting year.

        Returns:
            SHA-256 hex digest string (64 characters).
        """
        def _decimal_default(obj: Any) -> str:
            if isinstance(obj, Decimal):
                return str(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # Build provenance payload
        input_summary = []
        for p in products:
            input_summary.append({
                "product_id": p.get("product_id", ""),
                "product_category": p.get("product_category", ""),
                "units_sold": p.get("units_sold", 0),
            })

        payload = json.dumps(
            {
                "agent_id": AGENT_ID,
                "engine_id": ENGINE_ID,
                "engine_version": ENGINE_VERSION,
                "org_id": org_id,
                "reporting_year": reporting_year,
                "calculation_id": result.calculation_id,
                "input_count": len(products),
                "input_summary": input_summary,
                "total_co2e_kg": str(result.total_co2e_kg),
                "total_biogenic_co2_kg": str(result.total_biogenic_co2_kg),
                "total_avoided_co2e_kg": str(result.total_avoided_co2e_kg),
                "total_mass_kg": str(result.total_mass_kg),
                "gwp_version": result.gwp_version,
                "timestamp": result.timestamp,
            },
            sort_keys=True,
            default=_decimal_default,
        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ==================================================================
    # 13. STATISTICS / METRICS
    # ==================================================================

    def get_calculation_count(self) -> int:
        """
        Get total number of calculations performed.

        Returns:
            Count of calculations since engine initialization.
        """
        with self._rlock:
            return self._calc_count

    def get_last_calculation_time(self) -> Optional[datetime]:
        """
        Get timestamp of last calculation.

        Returns:
            UTC datetime of last calculation, or None if never calculated.
        """
        with self._rlock:
            return self._last_calculation

    # ==================================================================
    # 14. HEALTH CHECK
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on the calculator engine.

        Validates:
        - Singleton initialization
        - Database engine connectivity
        - Sample calculation success
        - Treatment pathway coverage

        Returns:
            Dict with status ("healthy" or "unhealthy"), checks performed,
            and any errors encountered.

        Example:
            >>> result = engine.health_check()
            >>> result["status"]
            'healthy'
        """
        start_ts = time.monotonic()
        checks: Dict[str, Any] = {}
        errors: List[str] = []

        # Check 1: Singleton initialized
        checks["singleton_initialized"] = self._initialized
        if not self._initialized:
            errors.append("Engine not initialized")

        # Check 2: Database engine health
        try:
            db_health = self._db.health_check()
            checks["database_engine_healthy"] = db_health["status"] == "healthy"
            if db_health["status"] != "healthy":
                errors.append(f"Database engine unhealthy: {db_health.get('errors', [])}")
        except Exception as e:
            checks["database_engine_healthy"] = False
            errors.append(f"Database engine health check failed: {e}")

        # Check 3: Sample calculation - simple product
        try:
            sample_products = [{
                "product_id": "HEALTH-CHECK-001",
                "product_category": "packaging",
                "units_sold": 100,
                "weight_per_unit_kg": "0.15",
                "region": "GLOBAL",
            }]
            sample_result = self.calculate(
                sample_products, "HEALTH-CHECK", 2025,
            )
            checks["sample_calculation"] = (
                Decimal(sample_result["total_co2e_kg"]) >= _ZERO
            )
        except Exception as e:
            checks["sample_calculation"] = False
            errors.append(f"Sample calculation failed: {e}")

        # Check 4: All treatment pathways callable
        try:
            for treatment in VALID_TREATMENTS:
                trt_result = self._calculate_treatment_emissions(
                    material="paper",
                    treatment=treatment,
                    mass_kg=Decimal("1.0"),
                    gwp_version="AR5",
                )
                if trt_result is None:
                    errors.append(f"Treatment '{treatment}' returned None")
            checks["all_treatments_callable"] = True
        except Exception as e:
            checks["all_treatments_callable"] = False
            errors.append(f"Treatment pathway test failed: {e}")

        # Check 5: Landfill FOD model
        try:
            landfill_result = self._calculate_landfill(
                "paper", Decimal("100.0"), "AR5", "temperate_wet", "managed_anaerobic",
            )
            checks["landfill_fod_operational"] = landfill_result.fossil_co2e_kg > _ZERO
        except Exception as e:
            checks["landfill_fod_operational"] = False
            errors.append(f"Landfill FOD test failed: {e}")

        # Check 6: Incineration model
        try:
            incin_result = self._calculate_incineration(
                "plastic", Decimal("100.0"), "AR5", "GLOBAL", False,
            )
            checks["incineration_operational"] = incin_result.fossil_co2e_kg > _ZERO
        except Exception as e:
            checks["incineration_operational"] = False
            errors.append(f"Incineration test failed: {e}")

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        status = "healthy" if len(errors) == 0 else "unhealthy"

        result = {
            "status": status,
            "agent_id": AGENT_ID,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "checks": checks,
            "errors": errors,
            "calculation_count": self.get_calculation_count(),
            "elapsed_ms": round(elapsed_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if status == "healthy":
            logger.info(
                "Health check PASSED: engine=%s, checks=%d, elapsed_ms=%.2f",
                ENGINE_ID, len(checks), elapsed_ms,
            )
        else:
            logger.warning(
                "Health check FAILED: engine=%s, errors=%s",
                ENGINE_ID, errors,
            )

        return result

    # ==================================================================
    # 15. STRING REPRESENTATION
    # ==================================================================

    def __repr__(self) -> str:
        """Return string representation of the engine."""
        return (
            f"WasteTypeSpecificCalculatorEngine("
            f"agent_id={AGENT_ID!r}, "
            f"engine_version={ENGINE_VERSION!r}, "
            f"gwp={self._default_gwp_version!r}, "
            f"calculations={self.get_calculation_count()}"
            f")"
        )

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"WasteTypeSpecificCalculatorEngine v{ENGINE_VERSION} "
            f"({AGENT_ID}): "
            f"GWP={self._default_gwp_version}, "
            f"calculations={self.get_calculation_count()}"
        )
