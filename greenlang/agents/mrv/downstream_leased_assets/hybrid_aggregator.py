# -*- coding: utf-8 -*-
"""
HybridAggregatorEngine - AGENT-MRV-026 Engine 5

GHG Protocol Scope 3 Category 13 hybrid aggregation engine implementing
method waterfall, portfolio aggregation, multi-tenant allocation, and
hot-spot analysis for downstream leased assets.

This engine orchestrates the three calculation methods available for
Category 13 (Downstream Leased Assets) in a prioritized waterfall:

    1. **Asset-Specific (Tier 1)** -- Metered energy data from tenants.
       Uncertainty +/-10%. Highest accuracy. Requires tenant cooperation.

    2. **Average-Data (Tier 2)** -- Building EUI benchmarks, fleet distances,
       equipment load factors. Uncertainty +/-30%. Uses lazy-loaded
       AverageDataCalculatorEngine.

    3. **Spend-Based (Tier 3)** -- EEIO factors x lease revenue.
       Uncertainty +/-50%. Uses lazy-loaded SpendBasedCalculatorEngine.

The engine selects the best available method per asset, aggregates results
across the portfolio, supports multi-tenant allocation, handles vacancy
periods, performs hot-spot (Pareto 80/20) analysis, and enforces the
operational control boundary check.

Key Features:
    - Lazy-loaded child engines with graceful fallback
    - Method waterfall with per-asset override
    - Portfolio aggregation by category, building type, vehicle type, region
    - 6 allocation methods: floor_area, headcount, revenue, FTE,
      equal_share, custom
    - Multi-tenant shared building allocation
    - Vacancy period handling with base-load fraction
    - DQI-weighted blending across methods
    - Blended portfolio uncertainty (inverse-variance weighting)
    - Hot-spot analysis (Pareto 80/20 across assets)
    - Operational control boundary enforcement

All calculations use Decimal arithmetic with ROUND_HALF_UP for regulatory
precision. Thread-safe singleton pattern.

References:
    - GHG Protocol Scope 3 Standard, Chapter 10 (Downstream Leased Assets)
    - GHG Protocol Scope 3 Technical Guidance, Category 13
    - IFRS 16 / ASC 842 Lease Classification (operating vs finance)
    - GHG Protocol Scope 2 Guidance (for operational control boundary)

Example:
    >>> engine = get_hybrid_aggregator()
    >>> result = engine.calculate([
    ...     {
    ...         "asset_id": "B001",
    ...         "asset_category": "building",
    ...         "building_type": "office",
    ...         "climate_zone": "temperate",
    ...         "floor_area_sqm": Decimal("5000"),
    ...         "region": "US_AVERAGE",
    ...         "lease_share": Decimal("1.0"),
    ...     },
    ...     {
    ...         "asset_id": "V001",
    ...         "asset_category": "vehicle",
    ...         "vehicle_type": "medium_car",
    ...         "fleet_count": 10,
    ...     },
    ... ])
    >>> result["total_co2e_kg"] > Decimal("0")
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-013
"""

import hashlib
import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "hybrid_aggregator_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-013"

PRECISION: int = 8
ROUNDING: str = ROUND_HALF_UP
_QUANT_8DP: Decimal = Decimal("0.00000001")
_QUANT_4DP: Decimal = Decimal("0.0001")
_QUANT_2DP: Decimal = Decimal("0.01")
_ZERO: Decimal = Decimal("0")
_ONE: Decimal = Decimal("1")
_HUNDRED: Decimal = Decimal("100")

# Tier uncertainty defaults
TIER_UNCERTAINTIES: Dict[str, Decimal] = {
    "tier_1": Decimal("0.10"),   # Asset-specific: +/-10%
    "tier_2": Decimal("0.30"),   # Average-data: +/-30%
    "tier_3": Decimal("0.50"),   # Spend-based: +/-50%
}

# Method waterfall priority (lower index = higher priority)
METHOD_WATERFALL: List[str] = [
    "asset_specific",   # Tier 1
    "average_data",     # Tier 2
    "spend_based",      # Tier 3
]

# Pareto threshold for hot-spot analysis
PARETO_THRESHOLD: Decimal = Decimal("0.80")


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class AllocationMethod(str, Enum):
    """Allocation methods for multi-tenant buildings."""

    FLOOR_AREA = "floor_area"
    HEADCOUNT = "headcount"
    REVENUE = "revenue"
    FTE = "fte"
    EQUAL_SHARE = "equal_share"
    CUSTOM = "custom"


class CalculationMethod(str, Enum):
    """Calculation methods for category 13 emissions."""

    ASSET_SPECIFIC = "asset_specific"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"
    HYBRID = "hybrid"


class AssetCategory(str, Enum):
    """Top-level asset categories."""

    BUILDING = "building"
    VEHICLE = "vehicle"
    EQUIPMENT = "equipment"
    IT_ASSET = "it_asset"


class OperationalControlStatus(str, Enum):
    """Operational control boundary status."""

    LESSOR_CONTROL = "lessor_control"       # Reporter retains control -> Scope 1/2
    TENANT_CONTROL = "tenant_control"       # Tenant has control -> Cat 13
    SHARED_CONTROL = "shared_control"       # Shared -> partial Cat 13
    UNDETERMINED = "undetermined"           # Needs assessment


# ==============================================================================
# DQI WEIGHTS
# ==============================================================================

DQI_WEIGHTS_PORTFOLIO: Dict[str, Decimal] = {
    "representativeness": Decimal("0.30"),
    "completeness": Decimal("0.25"),
    "temporal": Decimal("0.15"),
    "geographical": Decimal("0.15"),
    "technological": Decimal("0.15"),
}


# ==============================================================================
# PROVENANCE HELPER
# ==============================================================================


def _calculate_provenance_hash(*inputs: Any) -> str:
    """
    Calculate SHA-256 provenance hash from variable inputs.

    Args:
        *inputs: Variable number of input objects to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    hash_input = ""
    for inp in inputs:
        if isinstance(inp, dict):
            hash_input += json.dumps(inp, sort_keys=True, default=str)
        elif isinstance(inp, Decimal):
            hash_input += str(inp.quantize(_QUANT_8DP, rounding=ROUNDING))
        elif isinstance(inp, (list, tuple)):
            hash_input += json.dumps(
                [str(x) if isinstance(x, Decimal) else x for x in inp],
                sort_keys=True,
                default=str,
            )
        else:
            hash_input += str(inp)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


# ==============================================================================
# HybridAggregatorEngine
# ==============================================================================


class HybridAggregatorEngine:
    """
    Hybrid aggregation engine for downstream leased assets (Category 13).

    Orchestrates the method waterfall (asset-specific > average-data >
    spend-based), aggregates portfolio results, supports multi-tenant
    allocation, and performs hot-spot analysis.

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.

    Method Waterfall:
        For each asset, the engine selects the highest-accuracy method
        for which sufficient data is available:
            1. asset_specific (Tier 1, +/-10%) -- requires metered data
            2. average_data (Tier 2, +/-30%) -- requires building/vehicle type
            3. spend_based (Tier 3, +/-50%) -- requires lease revenue + NAICS

    Child Engines:
        AverageDataCalculatorEngine and SpendBasedCalculatorEngine are
        lazy-loaded on first use. If either is unavailable (import error),
        the engine gracefully skips that tier.

    Operational Control Boundary:
        If the lessor retains operational control of the leased asset,
        emissions belong in Scope 1/2, NOT Category 13. This engine
        flags such assets with a boundary warning.

    Attributes:
        _avg_engine: Lazy-loaded AverageDataCalculatorEngine
        _spend_engine: Lazy-loaded SpendBasedCalculatorEngine
        _calculation_count: Running count of portfolio calculations
        _asset_count: Running count of individual asset calculations

    Example:
        >>> engine = HybridAggregatorEngine.get_instance()
        >>> result = engine.calculate([
        ...     {"asset_id": "B001", "asset_category": "building", ...},
        ... ])
        >>> result["total_co2e_kg"] > Decimal("0")
        True
    """

    _instance: Optional["HybridAggregatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize HybridAggregatorEngine with lazy-loaded child engines."""
        self._avg_engine: Optional[Any] = None
        self._spend_engine: Optional[Any] = None
        self._avg_available: Optional[bool] = None
        self._spend_available: Optional[bool] = None
        self._calculation_count: int = 0
        self._asset_count: int = 0
        self._initialized_at: str = datetime.now(timezone.utc).isoformat()

        logger.info(
            "HybridAggregatorEngine initialized: version=%s, agent=%s",
            ENGINE_VERSION,
            AGENT_ID,
        )

    @classmethod
    def get_instance(cls) -> "HybridAggregatorEngine":
        """
        Get singleton instance (thread-safe double-checked locking).

        Returns:
            HybridAggregatorEngine singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (for testing only).

        Thread Safety:
            Protected by the class-level lock.
        """
        with cls._lock:
            cls._instance = None
            logger.info("HybridAggregatorEngine singleton reset")

    # ==========================================================================
    # Lazy Engine Loading
    # ==========================================================================

    def _get_avg_engine(self) -> Optional[Any]:
        """
        Lazy-load AverageDataCalculatorEngine.

        Returns:
            AverageDataCalculatorEngine instance, or None if unavailable.
        """
        if self._avg_available is None:
            try:
                from greenlang.agents.mrv.downstream_leased_assets.average_data_calculator import (
                    AverageDataCalculatorEngine,
                )
                self._avg_engine = AverageDataCalculatorEngine.get_instance()
                self._avg_available = True
                logger.info("AverageDataCalculatorEngine loaded successfully")
            except ImportError as e:
                self._avg_available = False
                logger.warning(
                    "AverageDataCalculatorEngine not available: %s", str(e)
                )
        return self._avg_engine if self._avg_available else None

    def _get_spend_engine(self) -> Optional[Any]:
        """
        Lazy-load SpendBasedCalculatorEngine.

        Returns:
            SpendBasedCalculatorEngine instance, or None if unavailable.
        """
        if self._spend_available is None:
            try:
                from greenlang.agents.mrv.downstream_leased_assets.spend_based_calculator import (
                    SpendBasedCalculatorEngine,
                )
                self._spend_engine = SpendBasedCalculatorEngine.get_instance()
                self._spend_available = True
                logger.info("SpendBasedCalculatorEngine loaded successfully")
            except ImportError as e:
                self._spend_available = False
                logger.warning(
                    "SpendBasedCalculatorEngine not available: %s", str(e)
                )
        return self._spend_engine if self._spend_available else None

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    def calculate(
        self, assets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a portfolio of downstream leased assets.

        For each asset, selects the best available method via the waterfall,
        aggregates results, and returns portfolio-level totals.

        Args:
            assets: List of asset dictionaries. Each must contain at least
                asset_id and enough data for at least one calculation method.
                Fields:
                    - asset_id (str): Unique asset identifier
                    - asset_category (str): building/vehicle/equipment/it_asset
                    - preferred_method (str, optional): Force a specific method
                    - operational_control (str, optional): Control status
                    - Plus method-specific fields

        Returns:
            Dictionary with total_co2e_kg, asset_results (list),
            by_category (dict), by_method (dict), portfolio_dqi,
            portfolio_uncertainty, provenance_hash, hot_spots (list).

        Raises:
            ValueError: If assets list is empty.

        Example:
            >>> result = engine.calculate([
            ...     {"asset_id": "B001", "asset_category": "building", ...},
            ... ])
        """
        if not assets:
            raise ValueError("Assets list cannot be empty")

        start_time = time.monotonic()

        asset_results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        by_category: Dict[str, Decimal] = {}
        by_method: Dict[str, Decimal] = {}
        by_building_type: Dict[str, Decimal] = {}
        by_vehicle_type: Dict[str, Decimal] = {}
        by_region: Dict[str, Decimal] = {}
        total_co2e = _ZERO
        boundary_warnings: List[Dict[str, Any]] = []

        logger.info(
            "Starting hybrid portfolio calculation: %d assets", len(assets)
        )

        for idx, asset_data in enumerate(assets):
            try:
                # Check operational control boundary
                boundary_check = self._check_operational_control(asset_data)
                if boundary_check["status"] == OperationalControlStatus.LESSOR_CONTROL.value:
                    boundary_warnings.append({
                        "asset_id": asset_data.get("asset_id", f"idx_{idx}"),
                        "warning": (
                            "Lessor retains operational control. Emissions "
                            "belong in Scope 1/2, not Category 13."
                        ),
                        "status": boundary_check["status"],
                    })
                    logger.warning(
                        "Asset %s excluded: lessor retains operational control",
                        asset_data.get("asset_id", f"idx_{idx}"),
                    )
                    continue

                # Select best method and calculate
                result = self._calculate_single_asset(asset_data)
                asset_results.append(result)

                co2e = result.get("co2e_kg", _ZERO)
                if isinstance(co2e, str):
                    co2e = Decimal(co2e)
                total_co2e += co2e

                # Aggregate by category
                cat = result.get("asset_category", "unknown")
                by_category[cat] = by_category.get(cat, _ZERO) + co2e

                # Aggregate by method
                method = result.get("selected_method", "unknown")
                by_method[method] = by_method.get(method, _ZERO) + co2e

                # Aggregate by building type
                bt = result.get("building_type")
                if bt:
                    by_building_type[bt] = by_building_type.get(bt, _ZERO) + co2e

                # Aggregate by vehicle type
                vt = result.get("vehicle_type")
                if vt:
                    by_vehicle_type[vt] = by_vehicle_type.get(vt, _ZERO) + co2e

                # Aggregate by region
                region = result.get("region")
                if region:
                    by_region[region] = by_region.get(region, _ZERO) + co2e

                self._asset_count += 1

            except (ValueError, InvalidOperation, KeyError) as e:
                errors.append({
                    "index": idx,
                    "asset_id": asset_data.get("asset_id", f"idx_{idx}"),
                    "error": str(e),
                })
                logger.error(
                    "Asset %d (%s) failed: %s",
                    idx,
                    asset_data.get("asset_id", f"idx_{idx}"),
                    str(e),
                )

        total_co2e = total_co2e.quantize(_QUANT_8DP, rounding=ROUNDING)

        # Portfolio-level metrics
        portfolio_dqi = self.compute_portfolio_dqi(asset_results)
        portfolio_uncertainty = self.compute_portfolio_uncertainty(asset_results)
        hot_spots = self.hotspot_analysis(asset_results, total_co2e)

        # Portfolio provenance hash
        provenance_hash = _calculate_provenance_hash(
            str(total_co2e),
            str(len(asset_results)),
            json.dumps(
                {k: str(v) for k, v in sorted(by_category.items())},
                sort_keys=True,
            ),
            json.dumps(
                {k: str(v) for k, v in sorted(by_method.items())},
                sort_keys=True,
            ),
        )

        duration = time.monotonic() - start_time
        self._calculation_count += 1

        logger.info(
            "Hybrid portfolio complete: %d/%d assets, total_co2e=%s kgCO2e, "
            "%d boundary warnings, %d errors, duration=%.4fs",
            len(asset_results),
            len(assets),
            total_co2e,
            len(boundary_warnings),
            len(errors),
            duration,
        )

        return {
            "total_co2e_kg": total_co2e,
            "asset_count": len(asset_results),
            "error_count": len(errors),
            "boundary_warning_count": len(boundary_warnings),
            "asset_results": asset_results,
            "errors": errors,
            "boundary_warnings": boundary_warnings,
            "by_category": {k: str(v) for k, v in sorted(by_category.items())},
            "by_method": {k: str(v) for k, v in sorted(by_method.items())},
            "by_building_type": {
                k: str(v) for k, v in sorted(by_building_type.items())
            },
            "by_vehicle_type": {
                k: str(v) for k, v in sorted(by_vehicle_type.items())
            },
            "by_region": {k: str(v) for k, v in sorted(by_region.items())},
            "portfolio_dqi": portfolio_dqi,
            "portfolio_uncertainty": portfolio_uncertainty,
            "hot_spots": hot_spots,
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "processing_time_ms": round(duration * 1000, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def select_best_method(
        self, asset_data: Dict[str, Any]
    ) -> str:
        """
        Select the best available calculation method for an asset.

        Follows the GHG Protocol method waterfall:
            1. asset_specific -- if metered_energy_kwh is provided
            2. average_data -- if asset_category + type fields are provided
            3. spend_based -- if lease_revenue + naics_code are provided

        A preferred_method field can force a specific method.

        Args:
            asset_data: Asset dictionary with available data fields.

        Returns:
            Selected method name string.

        Raises:
            ValueError: If no method has sufficient data.

        Example:
            >>> method = engine.select_best_method({
            ...     "asset_category": "building",
            ...     "building_type": "office",
            ...     "climate_zone": "temperate",
            ...     "floor_area_sqm": Decimal("5000"),
            ... })
            >>> method
            'average_data'
        """
        # Check for forced method
        preferred = asset_data.get("preferred_method", "").lower()
        if preferred and preferred in METHOD_WATERFALL:
            logger.debug(
                "Using preferred method: %s for asset %s",
                preferred,
                asset_data.get("asset_id", "unknown"),
            )
            return preferred

        # Tier 1: Asset-specific (metered data)
        if self._has_asset_specific_data(asset_data):
            return CalculationMethod.ASSET_SPECIFIC.value

        # Tier 2: Average-data (benchmark)
        if self._has_average_data(asset_data):
            avg_engine = self._get_avg_engine()
            if avg_engine is not None:
                return CalculationMethod.AVERAGE_DATA.value

        # Tier 3: Spend-based (EEIO)
        if self._has_spend_data(asset_data):
            spend_engine = self._get_spend_engine()
            if spend_engine is not None:
                return CalculationMethod.SPEND_BASED.value

        raise ValueError(
            f"No calculation method has sufficient data for asset "
            f"'{asset_data.get('asset_id', 'unknown')}'. Provide either "
            f"metered_energy_kwh (Tier 1), asset type fields (Tier 2), "
            f"or lease_revenue + naics_code (Tier 3)."
        )

    def aggregate_by_category(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Aggregate emissions by asset category.

        Args:
            results: List of asset result dictionaries.

        Returns:
            Dictionary mapping category to total kgCO2e string.
        """
        aggregation: Dict[str, Decimal] = {}
        for r in results:
            cat = r.get("asset_category", "unknown")
            co2e = self._extract_co2e(r)
            aggregation[cat] = aggregation.get(cat, _ZERO) + co2e

        return {k: str(v.quantize(_QUANT_8DP, rounding=ROUNDING))
                for k, v in sorted(aggregation.items())}

    def aggregate_by_building_type(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Aggregate emissions by building type.

        Args:
            results: List of asset result dictionaries.

        Returns:
            Dictionary mapping building type to total kgCO2e string.
        """
        aggregation: Dict[str, Decimal] = {}
        for r in results:
            bt = r.get("building_type")
            if bt:
                co2e = self._extract_co2e(r)
                aggregation[bt] = aggregation.get(bt, _ZERO) + co2e

        return {k: str(v.quantize(_QUANT_8DP, rounding=ROUNDING))
                for k, v in sorted(aggregation.items())}

    def aggregate_by_vehicle_type(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Aggregate emissions by vehicle type.

        Args:
            results: List of asset result dictionaries.

        Returns:
            Dictionary mapping vehicle type to total kgCO2e string.
        """
        aggregation: Dict[str, Decimal] = {}
        for r in results:
            vt = r.get("vehicle_type")
            if vt:
                co2e = self._extract_co2e(r)
                aggregation[vt] = aggregation.get(vt, _ZERO) + co2e

        return {k: str(v.quantize(_QUANT_8DP, rounding=ROUNDING))
                for k, v in sorted(aggregation.items())}

    def aggregate_by_region(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Aggregate emissions by region.

        Args:
            results: List of asset result dictionaries.

        Returns:
            Dictionary mapping region to total kgCO2e string.
        """
        aggregation: Dict[str, Decimal] = {}
        for r in results:
            region = r.get("region")
            if region:
                co2e = self._extract_co2e(r)
                aggregation[region] = aggregation.get(region, _ZERO) + co2e

        return {k: str(v.quantize(_QUANT_8DP, rounding=ROUNDING))
                for k, v in sorted(aggregation.items())}

    def allocate_to_tenants(
        self,
        total_co2e_kg: Decimal,
        tenants: List[Dict[str, Any]],
        method: str = "floor_area",
    ) -> List[Dict[str, Any]]:
        """
        Allocate building emissions to multiple tenants.

        Supports 6 allocation methods:
            - floor_area: Proportional to each tenant's leased area
            - headcount: Proportional to number of employees
            - revenue: Proportional to tenant revenue
            - fte: Proportional to full-time-equivalent employees
            - equal_share: Equal allocation across all tenants
            - custom: Use tenant-provided allocation weights

        Args:
            total_co2e_kg: Total building emissions to allocate.
            tenants: List of tenant dictionaries, each containing:
                - tenant_id (str): Unique tenant identifier
                - floor_area_sqm (Decimal, optional): For floor_area method
                - headcount (int, optional): For headcount method
                - revenue (Decimal, optional): For revenue method
                - fte (Decimal, optional): For FTE method
                - weight (Decimal, optional): For custom method
            method: Allocation method name.

        Returns:
            List of dictionaries with tenant_id, allocated_co2e_kg,
            share_pct, method, and provenance_hash.

        Raises:
            ValueError: If method is unsupported or tenant data is insufficient.

        Example:
            >>> allocations = engine.allocate_to_tenants(
            ...     Decimal("10000"),
            ...     [
            ...         {"tenant_id": "T1", "floor_area_sqm": Decimal("3000")},
            ...         {"tenant_id": "T2", "floor_area_sqm": Decimal("2000")},
            ...     ],
            ...     method="floor_area",
            ... )
            >>> allocations[0]["share_pct"]
            '60.00000000'
        """
        if not tenants:
            raise ValueError("Tenants list cannot be empty")

        method_lower = method.lower()
        if method_lower not in [m.value for m in AllocationMethod]:
            raise ValueError(
                f"Allocation method '{method}' not supported. "
                f"Valid: {[m.value for m in AllocationMethod]}"
            )

        # Compute allocation shares
        shares = self._compute_allocation_shares(tenants, method_lower)

        # Allocate emissions
        allocations: List[Dict[str, Any]] = []
        for tenant, share in zip(tenants, shares):
            allocated = (total_co2e_kg * share).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )
            share_pct = (share * _HUNDRED).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )

            provenance_hash = _calculate_provenance_hash(
                tenant.get("tenant_id", "unknown"),
                str(total_co2e_kg),
                str(share),
                str(allocated),
                method_lower,
            )

            allocations.append({
                "tenant_id": tenant.get("tenant_id", "unknown"),
                "allocated_co2e_kg": str(allocated),
                "share_pct": str(share_pct),
                "share_fraction": str(share),
                "method": method_lower,
                "provenance_hash": provenance_hash,
            })

        logger.info(
            "Tenant allocation complete: %d tenants, method=%s, "
            "total=%s kgCO2e",
            len(allocations), method_lower, total_co2e_kg,
        )

        return allocations

    def apply_vacancy_adjustment(
        self,
        co2e_kg: Decimal,
        vacancy_fraction: Decimal,
        base_load_fraction: Decimal = Decimal("0.30"),
    ) -> Dict[str, str]:
        """
        Apply vacancy adjustment to building emissions.

        Formula:
            adjusted = co2e_kg x (1 - vacancy_fraction x (1 - base_load_fraction))

        During vacancy, base-load energy (HVAC standby, security lighting)
        continues but occupant-driven loads stop.

        Args:
            co2e_kg: Total emissions before vacancy adjustment.
            vacancy_fraction: Fraction of the year the asset is vacant (0-1).
            base_load_fraction: Fraction of energy consumed as base load (0-1).

        Returns:
            Dictionary with original_co2e_kg, adjusted_co2e_kg,
            vacancy_fraction, base_load_fraction, adjustment_factor.

        Raises:
            ValueError: If fractions are out of range.
        """
        if vacancy_fraction < _ZERO or vacancy_fraction > _ONE:
            raise ValueError(
                f"vacancy_fraction must be between 0 and 1, "
                f"got {vacancy_fraction}"
            )
        if base_load_fraction < _ZERO or base_load_fraction > _ONE:
            raise ValueError(
                f"base_load_fraction must be between 0 and 1, "
                f"got {base_load_fraction}"
            )

        adjustment_factor = _ONE - (
            vacancy_fraction * (_ONE - base_load_fraction)
        )
        adjusted = (co2e_kg * adjustment_factor).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        return {
            "original_co2e_kg": str(co2e_kg),
            "adjusted_co2e_kg": str(adjusted),
            "vacancy_fraction": str(vacancy_fraction),
            "base_load_fraction": str(base_load_fraction),
            "adjustment_factor": str(adjustment_factor),
        }

    def hotspot_analysis(
        self,
        results: List[Dict[str, Any]],
        total_co2e_kg: Optional[Decimal] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform Pareto (80/20) hot-spot analysis across assets.

        Identifies the assets contributing to the top 80% of total emissions,
        sorted by descending emissions.

        Args:
            results: List of asset result dictionaries.
            total_co2e_kg: Pre-computed total (optional; computed if None).

        Returns:
            List of hot-spot dictionaries with asset_id, co2e_kg,
            cumulative_pct, is_hotspot, rank.

        Example:
            >>> spots = engine.hotspot_analysis(results)
            >>> spots[0]["rank"]
            1
        """
        if not results:
            return []

        # Compute total if not provided
        if total_co2e_kg is None:
            total_co2e_kg = _ZERO
            for r in results:
                total_co2e_kg += self._extract_co2e(r)

        if total_co2e_kg == _ZERO:
            return []

        # Sort by emissions descending
        sorted_assets = sorted(
            results,
            key=lambda r: self._extract_co2e(r),
            reverse=True,
        )

        hot_spots: List[Dict[str, Any]] = []
        cumulative = _ZERO

        for rank, r in enumerate(sorted_assets, start=1):
            co2e = self._extract_co2e(r)
            cumulative += co2e
            pct = (cumulative / total_co2e_kg * _HUNDRED).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )
            share_pct = (co2e / total_co2e_kg * _HUNDRED).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )

            is_hotspot = cumulative <= (
                total_co2e_kg * PARETO_THRESHOLD
            ) or rank == 1

            hot_spots.append({
                "rank": rank,
                "asset_id": r.get("asset_id", f"rank_{rank}"),
                "asset_category": r.get("asset_category", "unknown"),
                "co2e_kg": str(co2e),
                "share_pct": str(share_pct),
                "cumulative_pct": str(pct),
                "is_hotspot": is_hotspot,
            })

        return hot_spots

    def compute_portfolio_dqi(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute portfolio-level DQI by emissions-weighted averaging of
        individual asset DQI scores.

        Args:
            results: List of asset result dictionaries.

        Returns:
            Dictionary with composite_score, classification, method_mix,
            and tier.
        """
        if not results:
            return {
                "composite_score": _ZERO,
                "classification": "Very Poor",
                "method_mix": {},
                "tier": "tier_3",
            }

        total_weight = _ZERO
        weighted_score = _ZERO
        method_counts: Dict[str, int] = {}

        for r in results:
            co2e = self._extract_co2e(r)
            dqi = r.get("dqi_score", {})
            composite = dqi.get("composite_score", Decimal("2"))
            if isinstance(composite, str):
                composite = Decimal(composite)

            weighted_score += composite * co2e
            total_weight += co2e

            method = r.get("selected_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

        if total_weight > _ZERO:
            portfolio_composite = (weighted_score / total_weight).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )
        else:
            portfolio_composite = Decimal("2.00000000")

        # Classification
        if portfolio_composite >= Decimal("4.5"):
            classification = "Excellent"
        elif portfolio_composite >= Decimal("3.5"):
            classification = "Good"
        elif portfolio_composite >= Decimal("2.5"):
            classification = "Fair"
        elif portfolio_composite >= Decimal("1.5"):
            classification = "Poor"
        else:
            classification = "Very Poor"

        # Determine dominant tier
        tier = "tier_3"
        if method_counts.get("asset_specific", 0) > len(results) // 2:
            tier = "tier_1"
        elif method_counts.get("average_data", 0) > len(results) // 2:
            tier = "tier_2"

        return {
            "composite_score": portfolio_composite,
            "classification": classification,
            "method_mix": method_counts,
            "tier": tier,
        }

    def compute_portfolio_uncertainty(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute portfolio-level uncertainty using inverse-variance weighting.

        Blends per-asset uncertainties weighted by their emissions contribution.
        Lower-uncertainty assets (Tier 1) receive more weight.

        Args:
            results: List of asset result dictionaries.

        Returns:
            Dictionary with blended_uncertainty_pct, lower_bound_kg,
            upper_bound_kg, total_co2e_kg, method.
        """
        if not results:
            return {
                "blended_uncertainty_pct": str(TIER_UNCERTAINTIES["tier_3"]),
                "lower_bound_kg": "0",
                "upper_bound_kg": "0",
                "total_co2e_kg": "0",
                "method": "inverse_variance_weighted",
            }

        total_co2e = _ZERO
        weighted_var_sum = _ZERO

        for r in results:
            co2e = self._extract_co2e(r)
            unc_str = r.get("uncertainty_pct", "0.30")
            if isinstance(unc_str, Decimal):
                unc = unc_str
            else:
                unc = Decimal(str(unc_str))

            total_co2e += co2e
            # Variance contribution = (co2e * unc)^2
            contribution = co2e * unc
            weighted_var_sum += contribution * contribution

        if total_co2e > _ZERO:
            # Portfolio uncertainty = sqrt(sum of variances) / total
            portfolio_std = Decimal(
                str(math.sqrt(float(weighted_var_sum)))
            )
            blended_pct = (portfolio_std / total_co2e).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )
        else:
            blended_pct = TIER_UNCERTAINTIES["tier_3"]

        total_co2e = total_co2e.quantize(_QUANT_8DP, rounding=ROUNDING)
        lower = (total_co2e * (_ONE - blended_pct)).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        upper = (total_co2e * (_ONE + blended_pct)).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        return {
            "blended_uncertainty_pct": str(blended_pct),
            "lower_bound_kg": str(lower),
            "upper_bound_kg": str(upper),
            "total_co2e_kg": str(total_co2e),
            "method": "inverse_variance_weighted",
        }

    def validate_inputs(
        self, assets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate portfolio inputs and return structured result.

        Args:
            assets: List of asset dictionaries to validate.

        Returns:
            Dictionary with is_valid, errors, warnings, and per-asset details.
        """
        errors: List[str] = []
        warnings: List[str] = []
        asset_validations: List[Dict[str, Any]] = []

        if not assets:
            errors.append("Assets list cannot be empty")
            return {"is_valid": False, "errors": errors, "warnings": warnings}

        for idx, asset in enumerate(assets):
            asset_id = asset.get("asset_id", f"idx_{idx}")
            asset_errors: List[str] = []
            asset_warnings: List[str] = []

            # Check required fields
            if "asset_category" not in asset and "naics_code" not in asset:
                asset_errors.append(
                    "Either asset_category or naics_code is required"
                )

            # Check operational control
            oc = asset.get("operational_control", "").lower()
            if oc == OperationalControlStatus.LESSOR_CONTROL.value:
                asset_warnings.append(
                    "Lessor retains operational control: emissions may "
                    "belong in Scope 1/2, not Category 13"
                )

            # Try to select method
            try:
                method = self.select_best_method(asset)
            except ValueError as e:
                asset_errors.append(str(e))

            asset_validations.append({
                "asset_id": asset_id,
                "is_valid": len(asset_errors) == 0,
                "errors": asset_errors,
                "warnings": asset_warnings,
            })

            errors.extend([f"Asset {asset_id}: {e}" for e in asset_errors])
            warnings.extend([f"Asset {asset_id}: {w}" for w in asset_warnings])

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "asset_validations": asset_validations,
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Return engine health status and child engine availability.

        Returns:
            Dictionary with engine_id, engine_version, status, stats,
            child_engines, and method_waterfall.
        """
        avg_engine = self._get_avg_engine()
        spend_engine = self._get_spend_engine()

        avg_health = None
        if avg_engine is not None:
            try:
                avg_health = avg_engine.health_check()
            except Exception as e:
                avg_health = {"status": "error", "error": str(e)}

        spend_health = None
        if spend_engine is not None:
            try:
                spend_health = spend_engine.health_check()
            except Exception as e:
                spend_health = {"status": "error", "error": str(e)}

        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "status": "healthy",
            "stats": {
                "calculation_count": self._calculation_count,
                "asset_count": self._asset_count,
            },
            "child_engines": {
                "average_data": {
                    "available": self._avg_available or False,
                    "health": avg_health,
                },
                "spend_based": {
                    "available": self._spend_available or False,
                    "health": spend_health,
                },
            },
            "method_waterfall": METHOD_WATERFALL,
            "tier_uncertainties": {
                k: str(v) for k, v in TIER_UNCERTAINTIES.items()
            },
            "initialized_at": self._initialized_at,
        }

    # ==========================================================================
    # Internal Helpers - Calculation
    # ==========================================================================

    def _calculate_single_asset(
        self, asset_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a single asset using the best method.

        Args:
            asset_data: Asset dictionary.

        Returns:
            Result dictionary with co2e_kg and selected_method.
        """
        method = self.select_best_method(asset_data)
        asset_id = asset_data.get("asset_id", "unknown")

        logger.debug(
            "Calculating asset %s using method: %s", asset_id, method
        )

        if method == CalculationMethod.ASSET_SPECIFIC.value:
            result = self._calculate_asset_specific(asset_data)
        elif method == CalculationMethod.AVERAGE_DATA.value:
            result = self._calculate_average_data(asset_data)
        elif method == CalculationMethod.SPEND_BASED.value:
            result = self._calculate_spend_based(asset_data)
        else:
            raise ValueError(f"Unknown calculation method: {method}")

        result["asset_id"] = asset_id
        result["selected_method"] = method

        return result

    def _calculate_asset_specific(
        self, asset_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate using asset-specific (Tier 1) metered data.

        Formula: E = metered_energy_kwh x grid_EF x lease_share

        Args:
            asset_data: Asset dictionary with metered_energy_kwh.

        Returns:
            Result dictionary.
        """
        metered_kwh = Decimal(str(asset_data["metered_energy_kwh"]))
        region = asset_data.get("region", "GLOBAL_AVERAGE")
        lease_share = Decimal(str(asset_data.get("lease_share", "1.0")))

        # Get grid EF
        avg_engine = self._get_avg_engine()
        if avg_engine is not None:
            grid_ef = avg_engine.get_benchmark_ef(region)
        else:
            # Fallback to US average
            grid_ef = Decimal("0.3937")

        co2e_kg = (metered_kwh * grid_ef * lease_share).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        uncertainty_pct = TIER_UNCERTAINTIES["tier_1"]

        provenance_hash = _calculate_provenance_hash(
            str(metered_kwh), str(grid_ef), str(lease_share),
            region, str(co2e_kg),
        )

        return {
            "asset_category": asset_data.get("asset_category", "unknown"),
            "building_type": asset_data.get("building_type"),
            "vehicle_type": asset_data.get("vehicle_type"),
            "region": region,
            "metered_energy_kwh": str(metered_kwh),
            "grid_ef_kg_per_kwh": str(grid_ef),
            "lease_share": str(lease_share),
            "co2e_kg": co2e_kg,
            "calculation_method": "asset_specific",
            "data_quality_tier": "tier_1",
            "uncertainty_pct": str(uncertainty_pct),
            "uncertainty_lower_kg": str(
                (co2e_kg * (_ONE - uncertainty_pct)).quantize(
                    _QUANT_8DP, rounding=ROUNDING
                )
            ),
            "uncertainty_upper_kg": str(
                (co2e_kg * (_ONE + uncertainty_pct)).quantize(
                    _QUANT_8DP, rounding=ROUNDING
                )
            ),
            "dqi_score": {
                "composite_score": Decimal("4.00000000"),
                "classification": "Good",
                "tier": "tier_1",
            },
            "provenance_hash": provenance_hash,
        }

    def _calculate_average_data(
        self, asset_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate using average-data (Tier 2) engine.

        Args:
            asset_data: Asset dictionary with category-specific fields.

        Returns:
            Result dictionary from AverageDataCalculatorEngine.
        """
        avg_engine = self._get_avg_engine()
        if avg_engine is None:
            raise ValueError("AverageDataCalculatorEngine not available")

        result = avg_engine.calculate(asset_data)
        return result

    def _calculate_spend_based(
        self, asset_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate using spend-based (Tier 3) engine.

        Args:
            asset_data: Asset dictionary with lease_revenue and naics_code.

        Returns:
            Result dictionary from SpendBasedCalculatorEngine.
        """
        spend_engine = self._get_spend_engine()
        if spend_engine is None:
            raise ValueError("SpendBasedCalculatorEngine not available")

        result = spend_engine.calculate(asset_data)
        return result

    # ==========================================================================
    # Internal Helpers - Method Selection
    # ==========================================================================

    def _has_asset_specific_data(self, asset_data: Dict[str, Any]) -> bool:
        """
        Check if asset has metered energy data (Tier 1).

        Args:
            asset_data: Asset dictionary.

        Returns:
            True if metered_energy_kwh is present and positive.
        """
        metered = asset_data.get("metered_energy_kwh")
        if metered is None:
            return False
        try:
            val = Decimal(str(metered))
            return val > _ZERO
        except (InvalidOperation, ValueError):
            return False

    def _has_average_data(self, asset_data: Dict[str, Any]) -> bool:
        """
        Check if asset has sufficient data for average-data method (Tier 2).

        Requires asset_category plus category-specific fields.

        Args:
            asset_data: Asset dictionary.

        Returns:
            True if sufficient average-data fields are present.
        """
        category = asset_data.get("asset_category", "").lower()

        if category == AssetCategory.BUILDING.value:
            return all(
                asset_data.get(f)
                for f in ["building_type", "climate_zone", "floor_area_sqm"]
            )
        elif category == AssetCategory.VEHICLE.value:
            return bool(asset_data.get("vehicle_type"))
        elif category == AssetCategory.EQUIPMENT.value:
            return all(
                asset_data.get(f)
                for f in ["equipment_type", "rated_power_kw"]
            )
        elif category == AssetCategory.IT_ASSET.value:
            return bool(asset_data.get("it_asset_type"))

        return False

    def _has_spend_data(self, asset_data: Dict[str, Any]) -> bool:
        """
        Check if asset has sufficient data for spend-based method (Tier 3).

        Requires lease_revenue and naics_code (or classifiable description).

        Args:
            asset_data: Asset dictionary.

        Returns:
            True if spend-based fields are present.
        """
        has_revenue = asset_data.get("lease_revenue") is not None
        has_naics = bool(asset_data.get("naics_code"))
        has_description = bool(asset_data.get("description"))

        return has_revenue and (has_naics or has_description)

    # ==========================================================================
    # Internal Helpers - Allocation
    # ==========================================================================

    def _compute_allocation_shares(
        self,
        tenants: List[Dict[str, Any]],
        method: str,
    ) -> List[Decimal]:
        """
        Compute allocation shares for tenants.

        Args:
            tenants: List of tenant dictionaries.
            method: Allocation method name.

        Returns:
            List of share fractions (sum to 1.0).

        Raises:
            ValueError: If required data is missing for the method.
        """
        if method == AllocationMethod.EQUAL_SHARE.value:
            n = len(tenants)
            share = (_ONE / Decimal(str(n))).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )
            return [share] * n

        if method == AllocationMethod.CUSTOM.value:
            weights = []
            for t in tenants:
                w = t.get("weight")
                if w is None:
                    raise ValueError(
                        f"Tenant '{t.get('tenant_id', '?')}' missing "
                        f"'weight' for custom allocation"
                    )
                weights.append(Decimal(str(w)))
            total = sum(weights)
            if total == _ZERO:
                raise ValueError("Total custom weights cannot be zero")
            return [
                (w / total).quantize(_QUANT_8DP, rounding=ROUNDING)
                for w in weights
            ]

        # Proportional methods: floor_area, headcount, revenue, fte
        field_map = {
            AllocationMethod.FLOOR_AREA.value: "floor_area_sqm",
            AllocationMethod.HEADCOUNT.value: "headcount",
            AllocationMethod.REVENUE.value: "revenue",
            AllocationMethod.FTE.value: "fte",
        }

        field_name = field_map.get(method)
        if field_name is None:
            raise ValueError(f"Unknown allocation method: {method}")

        values: List[Decimal] = []
        for t in tenants:
            val = t.get(field_name)
            if val is None:
                raise ValueError(
                    f"Tenant '{t.get('tenant_id', '?')}' missing "
                    f"'{field_name}' for {method} allocation"
                )
            values.append(Decimal(str(val)))

        total = sum(values)
        if total == _ZERO:
            raise ValueError(
                f"Total {field_name} across tenants cannot be zero"
            )

        return [
            (v / total).quantize(_QUANT_8DP, rounding=ROUNDING)
            for v in values
        ]

    # ==========================================================================
    # Internal Helpers - Boundary Check
    # ==========================================================================

    def _check_operational_control(
        self, asset_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Check operational control boundary for an asset.

        If the lessor (reporter) retains operational control of the asset,
        emissions belong in Scope 1/2, not Category 13. This method checks
        the 'operational_control' field for explicit designation.

        Args:
            asset_data: Asset dictionary with optional operational_control.

        Returns:
            Dictionary with status and description.
        """
        oc = asset_data.get("operational_control", "").lower()

        if oc == OperationalControlStatus.LESSOR_CONTROL.value:
            return {
                "status": OperationalControlStatus.LESSOR_CONTROL.value,
                "description": (
                    "Reporter retains operational control. Emissions should "
                    "be reported under Scope 1/2, not Scope 3 Category 13."
                ),
            }
        elif oc == OperationalControlStatus.SHARED_CONTROL.value:
            return {
                "status": OperationalControlStatus.SHARED_CONTROL.value,
                "description": (
                    "Shared operational control. Partial emissions may be "
                    "allocated to Category 13 based on control assessment."
                ),
            }
        elif oc == OperationalControlStatus.TENANT_CONTROL.value:
            return {
                "status": OperationalControlStatus.TENANT_CONTROL.value,
                "description": (
                    "Tenant has operational control. Emissions are correctly "
                    "reported under Scope 3 Category 13."
                ),
            }

        return {
            "status": OperationalControlStatus.UNDETERMINED.value,
            "description": (
                "Operational control status not specified. Defaulting to "
                "Category 13 inclusion. Recommend formal assessment."
            ),
        }

    # ==========================================================================
    # Internal Helpers - Utilities
    # ==========================================================================

    def _extract_co2e(self, result: Dict[str, Any]) -> Decimal:
        """
        Extract co2e_kg from a result dictionary.

        Handles both Decimal and string representations.

        Args:
            result: Result dictionary.

        Returns:
            Emissions as Decimal.
        """
        co2e = result.get("co2e_kg", _ZERO)
        if isinstance(co2e, str):
            try:
                return Decimal(co2e)
            except (InvalidOperation, ValueError):
                return _ZERO
        if isinstance(co2e, Decimal):
            return co2e
        try:
            return Decimal(str(co2e))
        except (InvalidOperation, ValueError):
            return _ZERO

    # ==========================================================================
    # Additional Public Methods - Portfolio Analysis
    # ==========================================================================

    def method_coverage_report(
        self, assets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze method coverage across a portfolio without calculating.

        Reports which method each asset would use and identifies assets
        that lack data for any method. Useful for data collection planning.

        Args:
            assets: List of asset dictionaries.

        Returns:
            Dictionary with method_counts, missing_data_assets,
            coverage_pct, and upgrade_opportunities.

        Example:
            >>> report = engine.method_coverage_report([
            ...     {"asset_id": "B001", "asset_category": "building", ...},
            ...     {"asset_id": "V001", "asset_category": "vehicle", ...},
            ... ])
            >>> report["coverage_pct"] >= 0
            True
        """
        method_counts: Dict[str, int] = {
            "asset_specific": 0,
            "average_data": 0,
            "spend_based": 0,
            "no_data": 0,
        }
        missing_data: List[Dict[str, Any]] = []
        upgrade_opportunities: List[Dict[str, Any]] = []

        for asset in assets:
            asset_id = asset.get("asset_id", "unknown")
            try:
                method = self.select_best_method(asset)
                method_counts[method] = method_counts.get(method, 0) + 1

                # Identify upgrade opportunities
                if method == CalculationMethod.SPEND_BASED.value:
                    upgrade_opportunities.append({
                        "asset_id": asset_id,
                        "current_method": "spend_based",
                        "recommended_method": "average_data",
                        "data_needed": self._identify_missing_avg_fields(asset),
                        "uncertainty_reduction": "50% -> 30%",
                    })
                elif method == CalculationMethod.AVERAGE_DATA.value:
                    upgrade_opportunities.append({
                        "asset_id": asset_id,
                        "current_method": "average_data",
                        "recommended_method": "asset_specific",
                        "data_needed": ["metered_energy_kwh"],
                        "uncertainty_reduction": "30% -> 10%",
                    })

            except ValueError:
                method_counts["no_data"] += 1
                missing_data.append({
                    "asset_id": asset_id,
                    "category": asset.get("asset_category", "unknown"),
                    "available_fields": [
                        k for k, v in asset.items()
                        if v is not None and k != "asset_id"
                    ],
                })

        total = len(assets)
        covered = total - method_counts["no_data"]
        coverage_pct = (
            (Decimal(str(covered)) / Decimal(str(total)) * _HUNDRED).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )
            if total > 0
            else _ZERO
        )

        # Tier distribution
        tier_1_pct = _ZERO
        tier_2_pct = _ZERO
        tier_3_pct = _ZERO
        if covered > 0:
            covered_dec = Decimal(str(covered))
            tier_1_pct = (
                Decimal(str(method_counts["asset_specific"])) / covered_dec
                * _HUNDRED
            ).quantize(_QUANT_2DP, rounding=ROUNDING)
            tier_2_pct = (
                Decimal(str(method_counts["average_data"])) / covered_dec
                * _HUNDRED
            ).quantize(_QUANT_2DP, rounding=ROUNDING)
            tier_3_pct = (
                Decimal(str(method_counts["spend_based"])) / covered_dec
                * _HUNDRED
            ).quantize(_QUANT_2DP, rounding=ROUNDING)

        return {
            "total_assets": total,
            "covered_assets": covered,
            "coverage_pct": str(coverage_pct),
            "method_counts": method_counts,
            "tier_distribution": {
                "tier_1_pct": str(tier_1_pct),
                "tier_2_pct": str(tier_2_pct),
                "tier_3_pct": str(tier_3_pct),
            },
            "missing_data_assets": missing_data,
            "upgrade_opportunities": upgrade_opportunities[:20],
        }

    def generate_disclosure_summary(
        self, portfolio_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a GHG Protocol-compliant disclosure summary from
        portfolio calculation results.

        Produces the narrative and data elements required for Scope 3
        Category 13 disclosure per GHG Protocol, CSRD, CDP, and SBTi.

        Args:
            portfolio_result: Result from calculate().

        Returns:
            Dictionary with narrative sections, data tables, methodology
            description, exclusions, and data quality assessment.

        Example:
            >>> result = engine.calculate([...])
            >>> summary = engine.generate_disclosure_summary(result)
            >>> summary["total_co2e_tonnes"] > 0
            True
        """
        total_kg = portfolio_result.get("total_co2e_kg", _ZERO)
        if isinstance(total_kg, str):
            total_kg = Decimal(total_kg)

        total_tonnes = (total_kg / Decimal("1000")).quantize(
            _QUANT_4DP, rounding=ROUNDING
        )

        asset_count = portfolio_result.get("asset_count", 0)
        by_category = portfolio_result.get("by_category", {})
        by_method = portfolio_result.get("by_method", {})
        dqi = portfolio_result.get("portfolio_dqi", {})
        uncertainty = portfolio_result.get("portfolio_uncertainty", {})
        hot_spots = portfolio_result.get("hot_spots", [])

        # Category breakdown in tonnes
        category_tonnes = {}
        for cat, kg_str in by_category.items():
            kg = Decimal(kg_str)
            category_tonnes[cat] = str(
                (kg / Decimal("1000")).quantize(_QUANT_4DP, rounding=ROUNDING)
            )

        # Method mix description
        method_desc = []
        total_by_method = _ZERO
        for m, kg_str in by_method.items():
            kg = Decimal(kg_str)
            total_by_method += kg
        for m, kg_str in sorted(by_method.items()):
            kg = Decimal(kg_str)
            pct = (
                (kg / total_by_method * _HUNDRED).quantize(
                    _QUANT_2DP, rounding=ROUNDING
                )
                if total_by_method > _ZERO
                else _ZERO
            )
            method_desc.append({
                "method": m,
                "co2e_tonnes": str(
                    (kg / Decimal("1000")).quantize(
                        _QUANT_4DP, rounding=ROUNDING
                    )
                ),
                "share_pct": str(pct),
            })

        # Top hot-spot assets
        top_hotspots = [
            hs for hs in hot_spots if hs.get("is_hotspot", False)
        ][:5]

        # Narrative
        methodology = (
            f"Category 13 emissions were calculated using a hybrid approach "
            f"across {asset_count} downstream leased assets. "
            f"The method waterfall prioritized asset-specific metered data "
            f"(Tier 1, +/-10%), followed by average-data benchmarks "
            f"(Tier 2, +/-30%), and spend-based EEIO factors "
            f"(Tier 3, +/-50%) as a fallback."
        )

        data_quality_note = (
            f"Portfolio data quality classification: "
            f"{dqi.get('classification', 'Unknown')} "
            f"(composite score: {dqi.get('composite_score', 'N/A')}). "
            f"Blended uncertainty: +/-"
            f"{uncertainty.get('blended_uncertainty_pct', 'N/A')} "
            f"(inverse-variance weighted)."
        )

        return {
            "category": "Scope 3 Category 13 - Downstream Leased Assets",
            "total_co2e_kg": str(total_kg),
            "total_co2e_tonnes": str(total_tonnes),
            "asset_count": asset_count,
            "category_breakdown_tonnes": category_tonnes,
            "method_mix": method_desc,
            "methodology": methodology,
            "data_quality_note": data_quality_note,
            "dqi": dqi,
            "uncertainty": uncertainty,
            "top_hotspots": top_hotspots,
            "boundary": (
                "Emissions included are from assets owned by the reporting "
                "company and leased to other entities where the tenant has "
                "operational control. Assets where the reporter retains "
                "operational control are excluded (reported under Scope 1/2)."
            ),
            "exclusions": portfolio_result.get("boundary_warnings", []),
            "reporting_standard": "GHG Protocol Corporate Value Chain "
                                  "(Scope 3) Standard, Category 13",
        }

    def compare_allocation_methods(
        self,
        total_co2e_kg: Decimal,
        tenants: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare multiple allocation methods for the same set of tenants.

        Runs all applicable allocation methods and shows how results
        differ, helping organizations select the most appropriate method.

        Args:
            total_co2e_kg: Total building emissions to allocate.
            tenants: List of tenant dictionaries with all possible
                allocation data (floor_area_sqm, headcount, revenue,
                fte, weight).

        Returns:
            Dictionary with method results, variance analysis, and
            recommendation.
        """
        method_results: Dict[str, List[Dict[str, Any]]] = {}
        applicable_methods: List[str] = []

        # Try each method
        for method in AllocationMethod:
            try:
                result = self.allocate_to_tenants(
                    total_co2e_kg, tenants, method.value
                )
                method_results[method.value] = result
                applicable_methods.append(method.value)
            except ValueError:
                # Method not applicable (missing data)
                pass

        # Compute variance across methods for each tenant
        tenant_variance: List[Dict[str, Any]] = []
        for i, tenant in enumerate(tenants):
            tenant_id = tenant.get("tenant_id", f"tenant_{i}")
            allocations: Dict[str, Decimal] = {}
            for method_name, results in method_results.items():
                if i < len(results):
                    alloc_str = results[i].get("allocated_co2e_kg", "0")
                    allocations[method_name] = Decimal(alloc_str)

            if allocations:
                values = list(allocations.values())
                min_alloc = min(values)
                max_alloc = max(values)
                range_val = max_alloc - min_alloc
                mean_val = sum(values) / Decimal(str(len(values)))

                tenant_variance.append({
                    "tenant_id": tenant_id,
                    "allocations_by_method": {
                        k: str(v) for k, v in allocations.items()
                    },
                    "min_kg": str(min_alloc),
                    "max_kg": str(max_alloc),
                    "range_kg": str(range_val),
                    "mean_kg": str(
                        mean_val.quantize(_QUANT_8DP, rounding=ROUNDING)
                    ),
                })

        recommendation = (
            "Floor area allocation is recommended as the default method "
            "for buildings. It provides the most direct physical relationship "
            "between space usage and energy consumption. Headcount is "
            "recommended for office environments where occupancy drives "
            "energy use. Revenue-based allocation should be used only when "
            "physical data is unavailable."
        )

        return {
            "total_co2e_kg": str(total_co2e_kg),
            "tenant_count": len(tenants),
            "applicable_methods": applicable_methods,
            "method_results": {
                k: v for k, v in method_results.items()
            },
            "tenant_variance": tenant_variance,
            "recommendation": recommendation,
        }

    def _identify_missing_avg_fields(
        self, asset_data: Dict[str, Any]
    ) -> List[str]:
        """
        Identify which fields are missing for average-data method.

        Args:
            asset_data: Asset dictionary.

        Returns:
            List of missing field names.
        """
        category = asset_data.get("asset_category", "").lower()
        missing: List[str] = []

        if category == AssetCategory.BUILDING.value:
            for field in ["building_type", "climate_zone", "floor_area_sqm"]:
                if not asset_data.get(field):
                    missing.append(field)
        elif category == AssetCategory.VEHICLE.value:
            if not asset_data.get("vehicle_type"):
                missing.append("vehicle_type")
        elif category == AssetCategory.EQUIPMENT.value:
            for field in ["equipment_type", "rated_power_kw"]:
                if not asset_data.get(field):
                    missing.append(field)
        elif category == AssetCategory.IT_ASSET.value:
            if not asset_data.get("it_asset_type"):
                missing.append("it_asset_type")
        else:
            missing.append("asset_category")

        return missing


# ==============================================================================
# MODULE-LEVEL ACCESSOR
# ==============================================================================


def get_hybrid_aggregator() -> HybridAggregatorEngine:
    """
    Get the HybridAggregatorEngine singleton instance.

    Convenience function that delegates to the class-level get_instance().

    Returns:
        HybridAggregatorEngine singleton.

    Example:
        >>> engine = get_hybrid_aggregator()
        >>> engine.health_check()["status"]
        'healthy'
    """
    return HybridAggregatorEngine.get_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Constants
    "ENGINE_ID",
    "ENGINE_VERSION",
    "AGENT_ID",
    "TIER_UNCERTAINTIES",
    "METHOD_WATERFALL",
    "PARETO_THRESHOLD",
    # Enums
    "AllocationMethod",
    "CalculationMethod",
    "AssetCategory",
    "OperationalControlStatus",
    # Engine
    "HybridAggregatorEngine",
    "get_hybrid_aggregator",
]
