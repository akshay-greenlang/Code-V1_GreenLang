# -*- coding: utf-8 -*-
"""
AssetSpecificCalculatorEngine - Asset-specific (Tier 1) calculation engine.

This module implements the AssetSpecificCalculatorEngine for AGENT-MRV-026
(Downstream Leased Assets, GHG Protocol Scope 3 Category 13). It performs
Tier 1 calculations using metered energy data collected from tenants of
assets OWNED by the reporting company and LEASED OUT to others.

Category 13 Distinction (Reporter is LESSOR):
    The reporter owns the assets and leases them to tenants. Tier 1
    calculations require actual metered energy consumption data from
    tenant operations. This engine handles tenant data collection tracking,
    common area vs tenant area allocation, vacancy period base load
    adjustments, green lease clause impacts, and sub-metering coverage.

Supported Asset Categories:
    - Buildings: Multi-energy-type metered consumption with tenant allocation
    - Vehicles: Fuel-based and distance-based with fleet aggregation
    - Equipment: Fuel consumption with operating hours and load factor
    - IT Assets: PUE-adjusted power consumption with quantity scaling

Zero-Hallucination Guarantees:
    - All calculations use deterministic arithmetic formulas
    - No LLM calls in any calculation path
    - All emission factors from DownstreamAssetDatabaseEngine
    - Decimal arithmetic with ROUND_HALF_UP for regulatory precision
    - SHA-256 provenance hash on every result

Core Formulas:
    Building:  E = SUM [ metered_energy x EF x lease_share x alloc_factor ]
    Vehicle:   E = fuel_consumed x fuel_EF x lease_share (fuel-based)
               E = distance x vehicle_EF x lease_share (distance-based)
    Equipment: E = fuel_consumed x fuel_EF x lease_share x op_hours_frac
    IT Asset:  E = power_kw x PUE x hours x grid_EF x lease_share x qty

Example:
    >>> from greenlang.downstream_leased_assets.asset_specific_calculator import (
    ...     AssetSpecificCalculatorEngine,
    ... )
    >>> engine = AssetSpecificCalculatorEngine()
    >>> result = engine.calculate_building({
    ...     "building_type": "office",
    ...     "floor_area_m2": 5000,
    ...     "region": "US",
    ...     "energy_consumption": {
    ...         "electricity": {"kwh": 450000},
    ...         "natural_gas": {"m3": 25000},
    ...     },
    ...     "lease_share": 1.0,
    ...     "allocation_factor": 0.35,
    ... })
    >>> result["total_emissions_kgco2e"] > 0
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-013
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.downstream_leased_assets.downstream_asset_database import (
    DownstreamAssetDatabaseEngine,
    AGENT_ID,
    VERSION,
    TABLE_PREFIX,
    FUEL_EMISSION_FACTORS,
    ENERGY_TYPE_DEFAULTS,
    DQI_DIMENSION_WEIGHTS,
)

logger = logging.getLogger(__name__)

# =============================================================================
# QUANTIZATION CONSTANTS
# =============================================================================

_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")
_QUANT_4DP = Decimal("0.0001")

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")

# =============================================================================
# PROMETHEUS METRICS (graceful import)
# =============================================================================

try:
    from prometheus_client import Counter, Histogram, Gauge
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

    class _NoOpMetric:
        """No-op metric stub when prometheus_client is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "_NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def dec(self, amount: float = 1) -> None:
            pass

        def set(self, value: float = 0) -> None:
            pass

        def observe(self, value: float = 0) -> None:
            pass

    Counter = _NoOpMetric  # type: ignore[assignment,misc]
    Histogram = _NoOpMetric  # type: ignore[assignment,misc]
    Gauge = _NoOpMetric  # type: ignore[assignment,misc]


# Module-level singleton metrics
_CALCULATIONS_TOTAL = Counter(
    "gl_dla_asset_specific_calculations_total",
    "Total asset-specific calculations",
    ["asset_category", "status"],
)
_EMISSIONS_TOTAL = Counter(
    "gl_dla_asset_specific_emissions_kgco2e",
    "Total emissions from asset-specific calculations",
    ["asset_category"],
)
_CALC_DURATION = Histogram(
    "gl_dla_asset_specific_duration_seconds",
    "Asset-specific calculation duration",
    ["asset_category"],
)
_BATCH_SIZE = Histogram(
    "gl_dla_asset_specific_batch_size",
    "Batch sizes for asset-specific calculations",
)
_ACTIVE_CALCS = Gauge(
    "gl_dla_asset_specific_active",
    "Currently active asset-specific calculations",
)


# =============================================================================
# HELPER: DECIMAL CONVERSION
# =============================================================================


def _to_decimal(value: Any) -> Decimal:
    """
    Convert a value to Decimal safely.

    Handles int, float, str, and Decimal inputs. Floats are converted
    via string representation to avoid floating-point precision issues.

    Args:
        value: Value to convert to Decimal.

    Returns:
        Decimal representation of the value.

    Raises:
        TypeError: If value cannot be converted to Decimal.
        ValueError: If value string is not a valid decimal.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        return Decimal(value.strip())
    raise TypeError(
        f"Cannot convert {type(value).__name__} to Decimal: {value!r}"
    )


def _quantize(value: Decimal, precision: Decimal = _QUANT_8DP) -> Decimal:
    """
    Quantize a Decimal value with ROUND_HALF_UP.

    Args:
        value: Decimal to quantize.
        precision: Quantization precision.

    Returns:
        Quantized Decimal.
    """
    return value.quantize(precision, rounding=ROUND_HALF_UP)


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash for provenance tracking.

    Args:
        data: Data to hash (JSON-serialized with Decimal support).

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(
        data, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


# =============================================================================
# ENGINE CLASS
# =============================================================================


class AssetSpecificCalculatorEngine:
    """
    Thread-safe singleton for asset-specific (Tier 1) emissions calculations.

    Implements the highest-accuracy calculation method using actual metered
    energy consumption data from tenant operations of downstream leased
    assets. Supports buildings, vehicles, equipment, and IT assets.

    Category 13 Lessor-Specific Features:
        - Tenant energy data collection tracking
        - Common area vs tenant area allocation
        - Vacancy period base load handling
        - Green lease clause DQI impact tracking
        - Sub-metering coverage percentage tracking
        - Multi-tenant building allocation methods
        - Fleet aggregation for leased vehicle portfolios

    Thread Safety:
        Uses the __new__ singleton pattern with threading.Lock to ensure
        only one instance is created across all threads.

    Zero-Hallucination:
        All calculations use deterministic Python Decimal arithmetic.
        Emission factors are retrieved from DownstreamAssetDatabaseEngine
        which contains only validated, frozen constant tables.

    Attributes:
        _db: Reference to DownstreamAssetDatabaseEngine singleton
        _calc_count: Total number of calculations performed
        _error_count: Total number of calculation errors

    Example:
        >>> engine = AssetSpecificCalculatorEngine()
        >>> result = engine.calculate({
        ...     "asset_category": "building",
        ...     "building_type": "office",
        ...     "floor_area_m2": 5000,
        ...     "region": "US",
        ...     "energy_consumption": {"electricity": {"kwh": 450000}},
        ...     "lease_share": 1.0,
        ... })
        >>> result["total_emissions_kgco2e"] > 0
        True
    """

    _instance: Optional["AssetSpecificCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "AssetSpecificCalculatorEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the calculator engine (only once due to singleton)."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized: bool = True
        self._db: DownstreamAssetDatabaseEngine = DownstreamAssetDatabaseEngine()
        self._calc_count: int = 0
        self._error_count: int = 0
        self._calc_lock: threading.Lock = threading.Lock()
        self._created_at: datetime = datetime.now(timezone.utc)

        logger.info(
            "AssetSpecificCalculatorEngine initialized: agent=%s, version=%s",
            AGENT_ID, VERSION,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_calc(self) -> None:
        """Increment the calculation counter in a thread-safe manner."""
        with self._calc_lock:
            self._calc_count += 1

    def _increment_error(self) -> None:
        """Increment the error counter in a thread-safe manner."""
        with self._calc_lock:
            self._error_count += 1

    def _record_calc(
        self, asset_category: str, status: str, duration: float, emissions: Decimal
    ) -> None:
        """
        Record calculation metrics in Prometheus.

        Args:
            asset_category: Category of asset calculated.
            status: Outcome ("success" or "error").
            duration: Calculation duration in seconds.
            emissions: Total emissions in kgCO2e.
        """
        try:
            _CALCULATIONS_TOTAL.labels(
                asset_category=asset_category, status=status
            ).inc()
            _CALC_DURATION.labels(asset_category=asset_category).observe(duration)
            if status == "success" and emissions > _ZERO:
                _EMISSIONS_TOTAL.labels(
                    asset_category=asset_category
                ).inc(float(emissions))
        except Exception as exc:
            logger.warning("Failed to record calculation metric: %s", exc)

    @staticmethod
    def _validate_positive(name: str, value: Decimal) -> None:
        """
        Validate that a value is non-negative.

        Args:
            name: Name of the field for error messages.
            value: Value to validate.

        Raises:
            ValueError: If value is negative.
        """
        if value < _ZERO:
            raise ValueError(f"{name} must be non-negative, got {value}")

    @staticmethod
    def _validate_fraction(name: str, value: Decimal) -> None:
        """
        Validate that a value is in [0, 1] range.

        Args:
            name: Name of the field for error messages.
            value: Value to validate.

        Raises:
            ValueError: If value is not in [0, 1].
        """
        if value < _ZERO or value > _ONE:
            raise ValueError(
                f"{name} must be between 0 and 1, got {value}"
            )

    # =========================================================================
    # MAIN DISPATCH: calculate()
    # =========================================================================

    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route calculation to the appropriate asset-category-specific method.

        This is the main entry point for all asset-specific calculations.
        It inspects the "asset_category" field and dispatches to the
        corresponding calculate_building(), calculate_vehicle(),
        calculate_equipment(), or calculate_it_asset() method.

        Args:
            inputs: Calculation input dict. Must include "asset_category"
                plus all fields required by the category-specific method.

        Returns:
            Calculation result dict with emissions, provenance, DQI, and
            uncertainty.

        Raises:
            ValueError: If asset_category is missing or unknown.
            KeyError: If required input fields are missing.

        Example:
            >>> engine = AssetSpecificCalculatorEngine()
            >>> result = engine.calculate({
            ...     "asset_category": "vehicle",
            ...     "vehicle_type": "medium_car",
            ...     "fuel_type": "diesel",
            ...     "fuel_consumed_litres": 500,
            ...     "lease_share": 1.0,
            ...     "region": "US",
            ... })
            >>> result["total_emissions_kgco2e"] > 0
            True
        """
        category = inputs.get("asset_category")
        if category is None:
            raise ValueError(
                "Missing required field 'asset_category'. "
                "Must be one of: building, vehicle, equipment, it_asset"
            )

        cat_lower = str(category).strip().lower()

        if cat_lower == "building":
            return self.calculate_building(inputs)
        elif cat_lower == "vehicle":
            return self.calculate_vehicle(inputs)
        elif cat_lower == "equipment":
            return self.calculate_equipment(inputs)
        elif cat_lower == "it_asset":
            return self.calculate_it_asset(inputs)
        else:
            raise ValueError(
                f"Unknown asset_category '{category}'. "
                f"Must be one of: building, vehicle, equipment, it_asset"
            )

    # =========================================================================
    # BUILDING CALCULATIONS
    # =========================================================================

    def calculate_building(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate emissions for a leased building using metered energy data.

        Implements the building formula:
            E = SUM_energy_types [ metered_energy x EF x lease_share x alloc_factor ]

        Handles multiple energy types (electricity, natural_gas, steam,
        chilled_water), optional refrigerant leakage, common area allocation,
        and vacancy period adjustments.

        Args:
            inputs: Dict with required keys:
                - building_type (str): Building type for vacancy defaults.
                - region (str): Country code or eGRID subregion for grid EF.
                - energy_consumption (dict): Mapping of energy_type to
                    consumption dict. Each consumption dict has "kwh" or "m3".
                - lease_share (float/Decimal): Fraction of building leased
                    (0-1, where 1.0 = entire building).
                Optional keys:
                - floor_area_m2 (float/Decimal): Total floor area.
                - allocation_factor (float/Decimal): Common area allocation
                    fraction (0-1). Default 1.0 (no common area).
                - vacancy_fraction (float/Decimal): Fraction of year the
                    space was vacant (0-1). Default 0.0.
                - vacancy_adjustment (bool): Whether to apply vacancy base
                    load adjustment. Default True.
                - refrigerant_type (str): Refrigerant designation for leakage.
                - refrigerant_charge_kg (float/Decimal): System charge in kg.
                - refrigerant_leak_rate (float/Decimal): Annual leak rate (0-1).
                - sub_metering_coverage (float/Decimal): Fraction of space
                    sub-metered (0-1). Affects DQI score.
                - green_lease_clauses (list[str]): Active green lease clauses.
                - tenant_count (int): Number of tenants sharing the building.
                - tenant_id (str): Identifier for the tenant.

        Returns:
            Dict with keys:
                - total_emissions_kgco2e: Total CO2e emissions.
                - emissions_by_energy_type: Breakdown by energy source.
                - refrigerant_emissions_kgco2e: Refrigerant leakage emissions.
                - lease_share: Applied lease share.
                - allocation_factor: Applied allocation factor.
                - vacancy_fraction: Applied vacancy fraction.
                - vacancy_adjustment_kgco2e: Vacancy base load amount.
                - calculation_method: "asset_specific".
                - calculation_tier: "tier_1".
                - dqi_score: Data quality indicator score.
                - uncertainty_pct: Uncertainty percentage (+/-).
                - provenance_hash: SHA-256 hash.
                - processing_time_ms: Processing duration.

        Raises:
            ValueError: If required fields are missing or invalid.

        Example:
            >>> engine = AssetSpecificCalculatorEngine()
            >>> result = engine.calculate_building({
            ...     "building_type": "office",
            ...     "region": "US",
            ...     "energy_consumption": {
            ...         "electricity": {"kwh": 450000},
            ...         "natural_gas": {"m3": 25000},
            ...     },
            ...     "lease_share": 0.5,
            ...     "allocation_factor": 0.8,
            ... })
            >>> result["total_emissions_kgco2e"] > 0
            True
        """
        start_time = time.monotonic()
        self._increment_calc()

        try:
            _ACTIVE_CALCS.inc()

            # ---- Extract and validate inputs ----
            building_type = str(inputs.get("building_type", "")).strip().lower()
            if not building_type:
                raise ValueError("Missing required field 'building_type'.")

            region = str(inputs.get("region", "")).strip()
            if not region:
                raise ValueError("Missing required field 'region'.")

            energy_consumption = inputs.get("energy_consumption")
            if not energy_consumption or not isinstance(energy_consumption, dict):
                raise ValueError(
                    "Missing or invalid 'energy_consumption'. "
                    "Must be a dict mapping energy_type to consumption."
                )

            lease_share = _to_decimal(inputs.get("lease_share", 1))
            self._validate_fraction("lease_share", lease_share)

            allocation_factor = _to_decimal(inputs.get("allocation_factor", 1))
            self._validate_fraction("allocation_factor", allocation_factor)

            vacancy_fraction = _to_decimal(inputs.get("vacancy_fraction", 0))
            self._validate_fraction("vacancy_fraction", vacancy_fraction)

            apply_vacancy_adj = inputs.get("vacancy_adjustment", True)

            floor_area = inputs.get("floor_area_m2")
            if floor_area is not None:
                floor_area = _to_decimal(floor_area)
                self._validate_positive("floor_area_m2", floor_area)

            sub_metering_coverage = inputs.get("sub_metering_coverage")
            if sub_metering_coverage is not None:
                sub_metering_coverage = _to_decimal(sub_metering_coverage)
                self._validate_fraction("sub_metering_coverage", sub_metering_coverage)

            green_lease_clauses = inputs.get("green_lease_clauses", [])
            tenant_count = inputs.get("tenant_count")
            tenant_id = inputs.get("tenant_id", "unknown")

            # ---- Get grid EF for electricity ----
            grid_ef = self._db.get_grid_ef(region)

            # ---- Calculate emissions by energy type ----
            emissions_by_type: Dict[str, Dict[str, Decimal]] = {}
            total_emissions = _ZERO

            for energy_type, consumption in energy_consumption.items():
                et_lower = energy_type.strip().lower()
                type_emissions = self._calculate_building_energy_type(
                    et_lower, consumption, grid_ef
                )
                emissions_by_type[et_lower] = {
                    "consumption": type_emissions["consumption"],
                    "consumption_unit": type_emissions["consumption_unit"],
                    "ef_applied": type_emissions["ef_applied"],
                    "ef_unit": type_emissions["ef_unit"],
                    "raw_emissions_kgco2e": type_emissions["emissions_kgco2e"],
                }
                total_emissions += type_emissions["emissions_kgco2e"]

            # ---- Apply lease share and allocation ----
            allocated_emissions = _quantize(
                total_emissions * lease_share * allocation_factor
            )

            # ---- Apply vacancy adjustment ----
            vacancy_adj_kgco2e = _ZERO
            if apply_vacancy_adj and vacancy_fraction > _ZERO:
                vacancy_base = self._db.get_vacancy_base_load(building_type)
                vacancy_adj_kgco2e = _quantize(
                    allocated_emissions * vacancy_fraction * (vacancy_base - _ONE)
                )
                # Vacancy reduces actual consumption below full-occupancy
                # base load fraction means this portion still consumed
                allocated_emissions = _quantize(
                    allocated_emissions * (
                        _ONE - vacancy_fraction + vacancy_fraction * vacancy_base
                    )
                )

            # ---- Refrigerant leakage emissions ----
            refrigerant_emissions = _ZERO
            ref_type = inputs.get("refrigerant_type")
            if ref_type:
                ref_charge = _to_decimal(inputs.get("refrigerant_charge_kg", 0))
                ref_leak_rate = _to_decimal(inputs.get("refrigerant_leak_rate", Decimal("0.05")))
                self._validate_positive("refrigerant_charge_kg", ref_charge)
                self._validate_fraction("refrigerant_leak_rate", ref_leak_rate)

                gwp_data = self._db.get_refrigerant_gwp(ref_type)
                gwp = gwp_data["gwp_100"]
                refrigerant_emissions = _quantize(
                    ref_charge * ref_leak_rate * gwp * lease_share
                )

            # ---- Total with refrigerant ----
            final_emissions = _quantize(allocated_emissions + refrigerant_emissions)

            # ---- DQI Score ----
            dqi = self.compute_dqi_score(
                sub_metering_coverage=sub_metering_coverage,
                green_lease_clauses=green_lease_clauses,
                has_metered_data=True,
            )

            # ---- Uncertainty ----
            uncertainty = self.compute_uncertainty(final_emissions)

            # ---- Build result ----
            duration_ms = (time.monotonic() - start_time) * 1000

            result: Dict[str, Any] = {
                "asset_category": "building",
                "building_type": building_type,
                "region": region,
                "total_emissions_kgco2e": final_emissions,
                "emissions_by_energy_type": emissions_by_type,
                "refrigerant_emissions_kgco2e": refrigerant_emissions,
                "lease_share": _quantize(lease_share),
                "allocation_factor": _quantize(allocation_factor),
                "vacancy_fraction": _quantize(vacancy_fraction),
                "vacancy_adjustment_kgco2e": vacancy_adj_kgco2e,
                "calculation_method": "asset_specific",
                "calculation_tier": "tier_1",
                "dqi_score": dqi,
                "uncertainty_pct": uncertainty["uncertainty_pct"],
                "uncertainty_lower_kgco2e": uncertainty["lower"],
                "uncertainty_upper_kgco2e": uncertainty["upper"],
                "tenant_id": tenant_id,
                "processing_time_ms": round(duration_ms, 2),
            }

            result["provenance_hash"] = _compute_hash(result)

            self._record_calc("building", "success", duration_ms / 1000, final_emissions)

            logger.info(
                "Building calculation: type=%s, region=%s, emissions=%s kgCO2e, "
                "lease_share=%s, alloc=%s, vacancy=%s, duration=%.1fms",
                building_type, region, final_emissions,
                lease_share, allocation_factor, vacancy_fraction, duration_ms,
            )
            return result

        except Exception as exc:
            self._increment_error()
            duration_ms = (time.monotonic() - start_time) * 1000
            self._record_calc("building", "error", duration_ms / 1000, _ZERO)
            logger.error(
                "Building calculation failed: %s", exc, exc_info=True
            )
            raise
        finally:
            _ACTIVE_CALCS.dec()

    def _calculate_building_energy_type(
        self,
        energy_type: str,
        consumption: Dict[str, Any],
        grid_ef: Decimal,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a single building energy type.

        Applies the appropriate emission factor based on energy type:
        electricity uses grid EF, natural gas uses fuel EF, steam and
        chilled water use default EFs.

        Args:
            energy_type: Type of energy (electricity, natural_gas, steam,
                chilled_water).
            consumption: Dict with consumption quantity (kwh or m3 key).
            grid_ef: Grid emission factor for electricity (kgCO2e/kWh).

        Returns:
            Dict with consumption, ef_applied, emissions_kgco2e.

        Raises:
            ValueError: If energy type is unknown or consumption invalid.
        """
        et_defaults = ENERGY_TYPE_DEFAULTS.get(energy_type)
        if et_defaults is None:
            raise ValueError(
                f"Unknown energy type '{energy_type}'. "
                f"Available: {sorted(ENERGY_TYPE_DEFAULTS.keys())}"
            )

        # Extract consumption value
        if "kwh" in consumption:
            amount = _to_decimal(consumption["kwh"])
            unit = "kWh"
        elif "m3" in consumption:
            amount = _to_decimal(consumption["m3"])
            unit = "m3"
        elif "amount" in consumption:
            amount = _to_decimal(consumption["amount"])
            unit = consumption.get("unit", et_defaults["unit"])
        else:
            raise ValueError(
                f"Energy consumption for '{energy_type}' must include "
                f"'kwh', 'm3', or 'amount' key."
            )
        self._validate_positive(f"consumption_{energy_type}", amount)

        # Determine emission factor
        if et_defaults.get("requires_grid_ef"):
            ef = grid_ef
            ef_unit = "kgCO2e/kWh"
        elif et_defaults.get("requires_fuel_ef"):
            fuel_key = et_defaults.get("fuel_ef_key", energy_type)
            fuel_data = self._db.get_fuel_ef(fuel_key)
            ef = fuel_data["co2e_per_unit"]
            ef_unit = f"kgCO2e/{fuel_data['unit']}"
        else:
            ef = et_defaults.get("default_ef_kgco2e_per_kwh", _ZERO)
            ef_unit = "kgCO2e/kWh"

        emissions = _quantize(amount * ef)

        return {
            "consumption": _quantize(amount),
            "consumption_unit": unit,
            "ef_applied": _quantize(ef),
            "ef_unit": ef_unit,
            "emissions_kgco2e": emissions,
        }

    # =========================================================================
    # VEHICLE CALCULATIONS
    # =========================================================================

    def calculate_vehicle(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate emissions for a leased vehicle using fuel or distance data.

        Supports two calculation approaches:
        - Fuel-based: E = fuel_consumed x fuel_EF x lease_share
        - Distance-based: E = distance x vehicle_EF x lease_share

        For BEV (battery electric vehicles), uses grid EF and energy consumed.

        Args:
            inputs: Dict with required keys:
                - vehicle_type (str): Vehicle type from reference data.
                - fuel_type (str): Fuel type (gasoline, diesel, lpg, cng,
                    hybrid, phev, bev).
                - region (str): Country/eGRID for BEV grid EF.
                - lease_share (float/Decimal): Fraction leased (0-1).
                And at least one of:
                - fuel_consumed_litres (float/Decimal): Fuel consumed in L.
                - fuel_consumed_m3 (float/Decimal): Fuel consumed in m3 (CNG).
                - distance_km (float/Decimal): Distance driven in km.
                Optional keys:
                - bev_energy_kwh (float/Decimal): BEV energy consumed in kWh.
                - tenant_id (str): Tenant identifier.

        Returns:
            Dict with total_emissions_kgco2e, calculation details, DQI,
            uncertainty, and provenance_hash.

        Raises:
            ValueError: If required fields missing or invalid.

        Example:
            >>> engine = AssetSpecificCalculatorEngine()
            >>> result = engine.calculate_vehicle({
            ...     "vehicle_type": "medium_car",
            ...     "fuel_type": "diesel",
            ...     "fuel_consumed_litres": 500,
            ...     "lease_share": 1.0,
            ...     "region": "US",
            ... })
            >>> result["total_emissions_kgco2e"] > 0
            True
        """
        start_time = time.monotonic()
        self._increment_calc()

        try:
            _ACTIVE_CALCS.inc()

            # ---- Extract and validate ----
            vehicle_type = str(inputs.get("vehicle_type", "")).strip().lower()
            if not vehicle_type:
                raise ValueError("Missing required field 'vehicle_type'.")

            fuel_type = str(inputs.get("fuel_type", "")).strip().lower()
            if not fuel_type:
                raise ValueError("Missing required field 'fuel_type'.")

            region = str(inputs.get("region", "")).strip()
            if not region:
                raise ValueError("Missing required field 'region'.")

            lease_share = _to_decimal(inputs.get("lease_share", 1))
            self._validate_fraction("lease_share", lease_share)

            tenant_id = inputs.get("tenant_id", "unknown")

            # ---- Determine calculation approach ----
            fuel_litres = inputs.get("fuel_consumed_litres")
            fuel_m3 = inputs.get("fuel_consumed_m3")
            distance_km = inputs.get("distance_km")
            bev_kwh = inputs.get("bev_energy_kwh")

            emissions = _ZERO
            calc_approach = "unknown"
            approach_details: Dict[str, Any] = {}

            if fuel_type == "bev":
                # BEV: use grid EF and energy consumed
                calc_approach = "bev_energy"
                grid_ef = self._db.get_grid_ef(region)

                if bev_kwh is not None:
                    energy = _to_decimal(bev_kwh)
                elif distance_km is not None:
                    # Estimate energy from distance: ~0.2 kWh/km average
                    dist = _to_decimal(distance_km)
                    self._validate_positive("distance_km", dist)
                    energy = _quantize(dist * Decimal("0.2"))
                else:
                    raise ValueError(
                        "BEV requires 'bev_energy_kwh' or 'distance_km'."
                    )
                self._validate_positive("bev_energy", energy)

                emissions = _quantize(energy * grid_ef * lease_share)
                approach_details = {
                    "bev_energy_kwh": _quantize(energy),
                    "grid_ef": _quantize(grid_ef),
                    "grid_ef_unit": "kgCO2e/kWh",
                }

            elif fuel_litres is not None or fuel_m3 is not None:
                # Fuel-based calculation
                calc_approach = "fuel_based"

                if fuel_litres is not None:
                    fuel_amount = _to_decimal(fuel_litres)
                    fuel_unit = "L"
                else:
                    fuel_amount = _to_decimal(fuel_m3)
                    fuel_unit = "m3"

                self._validate_positive("fuel_consumed", fuel_amount)

                fuel_data = self._db.get_fuel_ef(fuel_type)
                fuel_ef = fuel_data["co2e_per_unit"]

                emissions = _quantize(fuel_amount * fuel_ef * lease_share)
                approach_details = {
                    "fuel_consumed": _quantize(fuel_amount),
                    "fuel_unit": fuel_unit,
                    "fuel_ef": _quantize(fuel_ef),
                    "fuel_ef_unit": f"kgCO2e/{fuel_data['unit']}",
                }

            elif distance_km is not None:
                # Distance-based calculation
                calc_approach = "distance_based"
                dist = _to_decimal(distance_km)
                self._validate_positive("distance_km", dist)

                vehicle_ef = self._db.get_vehicle_ef(vehicle_type, fuel_type)
                emissions = _quantize(dist * vehicle_ef * lease_share)
                approach_details = {
                    "distance_km": _quantize(dist),
                    "vehicle_ef": _quantize(vehicle_ef),
                    "vehicle_ef_unit": "kgCO2e/km",
                }

            else:
                raise ValueError(
                    "Vehicle calculation requires at least one of: "
                    "fuel_consumed_litres, fuel_consumed_m3, distance_km, "
                    "bev_energy_kwh."
                )

            # ---- DQI and uncertainty ----
            dqi = self.compute_dqi_score(has_metered_data=True)
            uncertainty = self.compute_uncertainty(emissions)

            # ---- Build result ----
            duration_ms = (time.monotonic() - start_time) * 1000

            result: Dict[str, Any] = {
                "asset_category": "vehicle",
                "vehicle_type": vehicle_type,
                "fuel_type": fuel_type,
                "region": region,
                "calculation_approach": calc_approach,
                "total_emissions_kgco2e": emissions,
                "lease_share": _quantize(lease_share),
                "approach_details": approach_details,
                "calculation_method": "asset_specific",
                "calculation_tier": "tier_1",
                "dqi_score": dqi,
                "uncertainty_pct": uncertainty["uncertainty_pct"],
                "uncertainty_lower_kgco2e": uncertainty["lower"],
                "uncertainty_upper_kgco2e": uncertainty["upper"],
                "tenant_id": tenant_id,
                "processing_time_ms": round(duration_ms, 2),
            }

            result["provenance_hash"] = _compute_hash(result)

            self._record_calc("vehicle", "success", duration_ms / 1000, emissions)

            logger.info(
                "Vehicle calculation: type=%s, fuel=%s, approach=%s, "
                "emissions=%s kgCO2e, lease=%s, duration=%.1fms",
                vehicle_type, fuel_type, calc_approach,
                emissions, lease_share, duration_ms,
            )
            return result

        except Exception as exc:
            self._increment_error()
            duration_ms = (time.monotonic() - start_time) * 1000
            self._record_calc("vehicle", "error", duration_ms / 1000, _ZERO)
            logger.error(
                "Vehicle calculation failed: %s", exc, exc_info=True
            )
            raise
        finally:
            _ACTIVE_CALCS.dec()

    # =========================================================================
    # EQUIPMENT CALCULATIONS
    # =========================================================================

    def calculate_equipment(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate emissions for leased equipment using fuel consumption data.

        Implements the equipment formula:
            E = fuel_consumed x fuel_EF x lease_share x op_hours_fraction

        For electric equipment (e.g., HVAC), uses:
            E = rated_power_kw x load_factor x hours x grid_EF x lease_share

        Args:
            inputs: Dict with required keys:
                - equipment_type (str): Equipment type from reference data.
                - region (str): Country/eGRID for electric equipment grid EF.
                - lease_share (float/Decimal): Fraction leased (0-1).
                And at least one of:
                - fuel_consumed_litres (float/Decimal): Actual fuel consumed.
                - operating_hours (float/Decimal): Hours operated in period.
                Optional keys:
                - operating_hours_fraction (float/Decimal): Fraction of total
                    hours operated (0-1). Default 1.0.
                - load_factor_override (float/Decimal): Override default load
                    factor (0-1).
                - tenant_id (str): Tenant identifier.

        Returns:
            Dict with total_emissions_kgco2e, calculation details, DQI,
            uncertainty, and provenance_hash.

        Raises:
            ValueError: If required fields missing or invalid.

        Example:
            >>> engine = AssetSpecificCalculatorEngine()
            >>> result = engine.calculate_equipment({
            ...     "equipment_type": "generator",
            ...     "fuel_consumed_litres": 1200,
            ...     "lease_share": 1.0,
            ...     "region": "US",
            ... })
            >>> result["total_emissions_kgco2e"] > 0
            True
        """
        start_time = time.monotonic()
        self._increment_calc()

        try:
            _ACTIVE_CALCS.inc()

            # ---- Extract and validate ----
            equipment_type = str(inputs.get("equipment_type", "")).strip().lower()
            if not equipment_type:
                raise ValueError("Missing required field 'equipment_type'.")

            region = str(inputs.get("region", "")).strip()
            if not region:
                raise ValueError("Missing required field 'region'.")

            lease_share = _to_decimal(inputs.get("lease_share", 1))
            self._validate_fraction("lease_share", lease_share)

            op_hours_fraction = _to_decimal(
                inputs.get("operating_hours_fraction", 1)
            )
            self._validate_fraction("operating_hours_fraction", op_hours_fraction)

            tenant_id = inputs.get("tenant_id", "unknown")

            # ---- Get equipment profile ----
            profile = self._db.get_equipment_fuel(equipment_type)
            eq_fuel_type = profile["fuel_type"]

            load_factor = profile["load_factor"]
            if inputs.get("load_factor_override") is not None:
                load_factor = _to_decimal(inputs["load_factor_override"])
                self._validate_fraction("load_factor_override", load_factor)

            emissions = _ZERO
            calc_approach = "unknown"
            approach_details: Dict[str, Any] = {
                "equipment_profile": {
                    "rated_power_kw": profile["rated_power_kw"],
                    "default_fuel_consumption_lph": profile["fuel_consumption_lph"],
                    "load_factor": _quantize(load_factor),
                    "fuel_type": eq_fuel_type,
                },
            }

            if eq_fuel_type == "electric":
                # Electric equipment: power x load_factor x hours x grid_ef
                calc_approach = "electric_metered"
                grid_ef = self._db.get_grid_ef(region)

                operating_hours = inputs.get("operating_hours")
                if operating_hours is None:
                    raise ValueError(
                        "Electric equipment requires 'operating_hours'."
                    )
                hours = _to_decimal(operating_hours)
                self._validate_positive("operating_hours", hours)

                rated_power = profile["rated_power_kw"]
                emissions = _quantize(
                    rated_power * load_factor * hours * grid_ef
                    * lease_share * op_hours_fraction
                )
                approach_details.update({
                    "operating_hours": _quantize(hours),
                    "grid_ef": _quantize(grid_ef),
                    "grid_ef_unit": "kgCO2e/kWh",
                })

            else:
                # Fuel-based equipment
                fuel_litres = inputs.get("fuel_consumed_litres")
                operating_hours = inputs.get("operating_hours")

                if fuel_litres is not None:
                    # Direct fuel measurement (most accurate)
                    calc_approach = "fuel_metered"
                    fuel_amount = _to_decimal(fuel_litres)
                    self._validate_positive("fuel_consumed_litres", fuel_amount)

                    fuel_data = self._db.get_fuel_ef(eq_fuel_type)
                    fuel_ef = fuel_data["co2e_per_unit"]

                    emissions = _quantize(
                        fuel_amount * fuel_ef * lease_share * op_hours_fraction
                    )
                    approach_details.update({
                        "fuel_consumed_litres": _quantize(fuel_amount),
                        "fuel_ef": _quantize(fuel_ef),
                        "fuel_ef_unit": f"kgCO2e/{fuel_data['unit']}",
                    })

                elif operating_hours is not None:
                    # Estimate fuel from operating hours and default consumption
                    calc_approach = "hours_estimated"
                    hours = _to_decimal(operating_hours)
                    self._validate_positive("operating_hours", hours)

                    default_lph = profile["fuel_consumption_lph"]
                    estimated_fuel = _quantize(
                        hours * default_lph * load_factor
                    )

                    fuel_data = self._db.get_fuel_ef(eq_fuel_type)
                    fuel_ef = fuel_data["co2e_per_unit"]

                    emissions = _quantize(
                        estimated_fuel * fuel_ef * lease_share * op_hours_fraction
                    )
                    approach_details.update({
                        "operating_hours": _quantize(hours),
                        "estimated_fuel_litres": estimated_fuel,
                        "fuel_ef": _quantize(fuel_ef),
                        "fuel_ef_unit": f"kgCO2e/{fuel_data['unit']}",
                    })

                else:
                    raise ValueError(
                        "Equipment calculation requires 'fuel_consumed_litres' "
                        "or 'operating_hours'."
                    )

            # ---- DQI and uncertainty ----
            dqi = self.compute_dqi_score(has_metered_data=True)
            uncertainty = self.compute_uncertainty(emissions)

            # ---- Build result ----
            duration_ms = (time.monotonic() - start_time) * 1000

            result: Dict[str, Any] = {
                "asset_category": "equipment",
                "equipment_type": equipment_type,
                "region": region,
                "calculation_approach": calc_approach,
                "total_emissions_kgco2e": emissions,
                "lease_share": _quantize(lease_share),
                "operating_hours_fraction": _quantize(op_hours_fraction),
                "approach_details": approach_details,
                "calculation_method": "asset_specific",
                "calculation_tier": "tier_1",
                "dqi_score": dqi,
                "uncertainty_pct": uncertainty["uncertainty_pct"],
                "uncertainty_lower_kgco2e": uncertainty["lower"],
                "uncertainty_upper_kgco2e": uncertainty["upper"],
                "tenant_id": tenant_id,
                "processing_time_ms": round(duration_ms, 2),
            }

            result["provenance_hash"] = _compute_hash(result)

            self._record_calc("equipment", "success", duration_ms / 1000, emissions)

            logger.info(
                "Equipment calculation: type=%s, approach=%s, "
                "emissions=%s kgCO2e, lease=%s, duration=%.1fms",
                equipment_type, calc_approach, emissions,
                lease_share, duration_ms,
            )
            return result

        except Exception as exc:
            self._increment_error()
            duration_ms = (time.monotonic() - start_time) * 1000
            self._record_calc("equipment", "error", duration_ms / 1000, _ZERO)
            logger.error(
                "Equipment calculation failed: %s", exc, exc_info=True
            )
            raise
        finally:
            _ACTIVE_CALCS.dec()

    # =========================================================================
    # IT ASSET CALCULATIONS
    # =========================================================================

    def calculate_it_asset(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate emissions for leased IT assets using power data.

        Implements the IT asset formula:
            E = power_kw x PUE x hours x grid_EF x lease_share x quantity

        Data center assets (servers, switches, storage) apply PUE > 1.0 to
        account for cooling, lighting, and UPS overhead. Office assets
        (desktops, laptops, printers, copiers) use PUE = 1.0.

        Args:
            inputs: Dict with required keys:
                - it_type (str): IT asset type from reference data.
                - region (str): Country/eGRID for grid EF.
                - lease_share (float/Decimal): Fraction leased (0-1).
                Optional keys:
                - quantity (int): Number of assets. Default 1.
                - power_kw_override (float/Decimal): Override rated power.
                - pue_override (float/Decimal): Override default PUE.
                - hours_override (float/Decimal): Override annual hours.
                - utilization (float/Decimal): Server utilization (0-1).
                    Scales power consumption. Default 1.0.
                - tenant_id (str): Tenant identifier.

        Returns:
            Dict with total_emissions_kgco2e, per_unit_emissions, calculation
            details, DQI, uncertainty, and provenance_hash.

        Raises:
            ValueError: If required fields missing or invalid.

        Example:
            >>> engine = AssetSpecificCalculatorEngine()
            >>> result = engine.calculate_it_asset({
            ...     "it_type": "server",
            ...     "region": "US",
            ...     "quantity": 10,
            ...     "lease_share": 1.0,
            ... })
            >>> result["total_emissions_kgco2e"] > 0
            True
        """
        start_time = time.monotonic()
        self._increment_calc()

        try:
            _ACTIVE_CALCS.inc()

            # ---- Extract and validate ----
            it_type = str(inputs.get("it_type", "")).strip().lower()
            if not it_type:
                raise ValueError("Missing required field 'it_type'.")

            region = str(inputs.get("region", "")).strip()
            if not region:
                raise ValueError("Missing required field 'region'.")

            lease_share = _to_decimal(inputs.get("lease_share", 1))
            self._validate_fraction("lease_share", lease_share)

            quantity = int(inputs.get("quantity", 1))
            if quantity < 1:
                raise ValueError(f"quantity must be >= 1, got {quantity}")
            qty_decimal = Decimal(quantity)

            tenant_id = inputs.get("tenant_id", "unknown")

            # ---- Get IT profile and overrides ----
            profile = self._db.get_it_power(it_type)

            power_kw = profile["power_kw"]
            if inputs.get("power_kw_override") is not None:
                power_kw = _to_decimal(inputs["power_kw_override"])
                self._validate_positive("power_kw_override", power_kw)

            pue = profile["default_pue"]
            if inputs.get("pue_override") is not None:
                pue = _to_decimal(inputs["pue_override"])
                if pue < _ONE:
                    raise ValueError(f"PUE must be >= 1.0, got {pue}")

            hours = profile["hours_per_year"]
            if inputs.get("hours_override") is not None:
                hours = _to_decimal(inputs["hours_override"])
                self._validate_positive("hours_override", hours)

            utilization = _to_decimal(inputs.get("utilization", 1))
            self._validate_fraction("utilization", utilization)

            # ---- Get grid EF ----
            grid_ef = self._db.get_grid_ef(region)

            # ---- Calculate: E = power x PUE x hours x grid_ef x lease x qty x util ----
            per_unit_emissions = _quantize(
                power_kw * pue * hours * grid_ef * utilization
            )
            total_emissions = _quantize(
                per_unit_emissions * lease_share * qty_decimal
            )

            # ---- DQI and uncertainty ----
            dqi = self.compute_dqi_score(has_metered_data=True)
            uncertainty = self.compute_uncertainty(total_emissions)

            # ---- Build result ----
            duration_ms = (time.monotonic() - start_time) * 1000

            result: Dict[str, Any] = {
                "asset_category": "it_asset",
                "it_type": it_type,
                "region": region,
                "total_emissions_kgco2e": total_emissions,
                "per_unit_emissions_kgco2e": per_unit_emissions,
                "quantity": quantity,
                "lease_share": _quantize(lease_share),
                "approach_details": {
                    "power_kw": _quantize(power_kw),
                    "pue": _quantize(pue),
                    "hours_per_year": _quantize(hours),
                    "grid_ef": _quantize(grid_ef),
                    "grid_ef_unit": "kgCO2e/kWh",
                    "utilization": _quantize(utilization),
                },
                "calculation_method": "asset_specific",
                "calculation_tier": "tier_1",
                "dqi_score": dqi,
                "uncertainty_pct": uncertainty["uncertainty_pct"],
                "uncertainty_lower_kgco2e": uncertainty["lower"],
                "uncertainty_upper_kgco2e": uncertainty["upper"],
                "tenant_id": tenant_id,
                "processing_time_ms": round(duration_ms, 2),
            }

            result["provenance_hash"] = _compute_hash(result)

            self._record_calc("it_asset", "success", duration_ms / 1000, total_emissions)

            logger.info(
                "IT asset calculation: type=%s, qty=%d, region=%s, "
                "emissions=%s kgCO2e, per_unit=%s, duration=%.1fms",
                it_type, quantity, region,
                total_emissions, per_unit_emissions, duration_ms,
            )
            return result

        except Exception as exc:
            self._increment_error()
            duration_ms = (time.monotonic() - start_time) * 1000
            self._record_calc("it_asset", "error", duration_ms / 1000, _ZERO)
            logger.error(
                "IT asset calculation failed: %s", exc, exc_info=True
            )
            raise
        finally:
            _ACTIVE_CALCS.dec()

    # =========================================================================
    # BATCH CALCULATION
    # =========================================================================

    def calculate_batch(
        self,
        records: List[Dict[str, Any]],
        batch_size: int = 100,
        continue_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a batch of asset-specific calculations.

        Iterates through a list of calculation inputs, dispatching each to
        the appropriate calculate_* method. Supports continuing on individual
        record errors and aggregates results with summary statistics.

        Args:
            records: List of input dicts, each with "asset_category" and
                category-specific fields.
            batch_size: Processing chunk size for memory efficiency. Default 100.
            continue_on_error: If True, skip failed records and continue.
                If False, raise on first error. Default True.

        Returns:
            Dict with keys:
                - results: List of successful calculation results.
                - errors: List of error dicts (index, error message).
                - summary: Aggregation with total_emissions, count, by_category.
                - provenance_hash: SHA-256 of entire batch result.
                - processing_time_ms: Total batch duration.

        Example:
            >>> engine = AssetSpecificCalculatorEngine()
            >>> batch = engine.calculate_batch([
            ...     {"asset_category": "vehicle", "vehicle_type": "small_car",
            ...      "fuel_type": "gasoline", "fuel_consumed_litres": 100,
            ...      "lease_share": 1.0, "region": "US"},
            ...     {"asset_category": "it_asset", "it_type": "server",
            ...      "region": "US", "quantity": 5, "lease_share": 1.0},
            ... ])
            >>> len(batch["results"]) == 2
            True
        """
        start_time = time.monotonic()

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total_emissions = _ZERO
        by_category: Dict[str, Dict[str, Any]] = {}

        try:
            _BATCH_SIZE.observe(len(records))
        except Exception:
            pass

        for idx, record in enumerate(records):
            try:
                result = self.calculate(record)
                results.append(result)

                # Aggregate
                category = result.get("asset_category", "unknown")
                record_emissions = result.get("total_emissions_kgco2e", _ZERO)
                total_emissions += record_emissions

                if category not in by_category:
                    by_category[category] = {
                        "count": 0,
                        "total_emissions_kgco2e": _ZERO,
                    }
                by_category[category]["count"] += 1
                by_category[category]["total_emissions_kgco2e"] += record_emissions

            except Exception as exc:
                error_entry = {
                    "index": idx,
                    "error": str(exc),
                    "asset_category": record.get("asset_category", "unknown"),
                }
                errors.append(error_entry)

                if not continue_on_error:
                    raise ValueError(
                        f"Batch calculation failed at record {idx}: {exc}"
                    ) from exc

                logger.warning(
                    "Batch record %d failed (continuing): %s", idx, exc
                )

        # Quantize aggregated values
        for cat_data in by_category.values():
            cat_data["total_emissions_kgco2e"] = _quantize(
                cat_data["total_emissions_kgco2e"]
            )

        duration_ms = (time.monotonic() - start_time) * 1000

        summary = {
            "total_records": len(records),
            "successful": len(results),
            "failed": len(errors),
            "total_emissions_kgco2e": _quantize(total_emissions),
            "by_category": by_category,
        }

        batch_result: Dict[str, Any] = {
            "results": results,
            "errors": errors,
            "summary": summary,
            "processing_time_ms": round(duration_ms, 2),
        }
        batch_result["provenance_hash"] = _compute_hash({
            "summary": summary,
            "record_hashes": [
                r.get("provenance_hash", "") for r in results
            ],
        })

        logger.info(
            "Batch calculation complete: total=%d, success=%d, "
            "failed=%d, emissions=%s kgCO2e, duration=%.1fms",
            len(records), len(results), len(errors),
            _quantize(total_emissions), duration_ms,
        )
        return batch_result

    # =========================================================================
    # ALLOCATION
    # =========================================================================

    def apply_allocation(
        self,
        total_emissions_kgco2e: Union[Decimal, float, int, str],
        method: str,
        tenant_value: Union[Decimal, float, int, str],
        total_value: Union[Decimal, float, int, str],
        common_area_fraction: Union[Decimal, float, int, str] = 0,
    ) -> Dict[str, Any]:
        """
        Apply allocation to distribute emissions among tenants.

        Calculates the tenant's share of emissions using the specified
        allocation method. Supports floor area, headcount, revenue, equal
        share, and metered allocation. Common area emissions can be
        allocated separately.

        Args:
            total_emissions_kgco2e: Total building/asset emissions.
            method: Allocation method (floor_area, headcount, revenue,
                equal_share, metered).
            tenant_value: Tenant's value for the allocation metric
                (e.g., tenant floor area, headcount, metered consumption).
            total_value: Total value across all tenants/building.
            common_area_fraction: Fraction of emissions attributable to
                common areas (0-1). Default 0.

        Returns:
            Dict with keys:
                - tenant_emissions_kgco2e: Emissions allocated to tenant.
                - common_area_emissions_kgco2e: Common area allocation.
                - allocation_ratio: Tenant share ratio.
                - method: Allocation method used.
                - provenance_hash: SHA-256 hash.

        Raises:
            ValueError: If method is unknown or values invalid.

        Example:
            >>> engine = AssetSpecificCalculatorEngine()
            >>> result = engine.apply_allocation(
            ...     total_emissions_kgco2e=1000,
            ...     method="floor_area",
            ...     tenant_value=500,
            ...     total_value=2000,
            ...     common_area_fraction=0.15,
            ... )
            >>> result["tenant_emissions_kgco2e"]
            Decimal('250.00000000')
        """
        total_em = _to_decimal(total_emissions_kgco2e)
        self._validate_positive("total_emissions_kgco2e", total_em)

        t_val = _to_decimal(tenant_value)
        self._validate_positive("tenant_value", t_val)

        tot_val = _to_decimal(total_value)
        if tot_val <= _ZERO:
            raise ValueError(f"total_value must be positive, got {tot_val}")

        ca_frac = _to_decimal(common_area_fraction)
        self._validate_fraction("common_area_fraction", ca_frac)

        method_lower = method.strip().lower()
        valid_methods = {"floor_area", "headcount", "revenue", "equal_share", "metered"}
        if method_lower not in valid_methods:
            raise ValueError(
                f"Unknown allocation method '{method}'. "
                f"Valid: {sorted(valid_methods)}"
            )

        # Separate common area emissions
        common_area_em = _quantize(total_em * ca_frac)
        tenant_pool_em = _quantize(total_em - common_area_em)

        # Calculate tenant ratio
        if method_lower == "equal_share":
            # total_value represents number of tenants
            alloc_ratio = _quantize(
                _ONE / tot_val, _QUANT_8DP
            )
        else:
            alloc_ratio = _quantize(t_val / tot_val, _QUANT_8DP)

        # Allocate
        tenant_em = _quantize(tenant_pool_em * alloc_ratio)

        # Common area share (proportional to tenant ratio)
        tenant_ca = _quantize(common_area_em * alloc_ratio)
        total_tenant = _quantize(tenant_em + tenant_ca)

        result: Dict[str, Any] = {
            "tenant_emissions_kgco2e": total_tenant,
            "tenant_private_emissions_kgco2e": tenant_em,
            "common_area_emissions_kgco2e": tenant_ca,
            "allocation_ratio": alloc_ratio,
            "method": method_lower,
            "common_area_fraction": _quantize(ca_frac),
            "total_building_emissions_kgco2e": _quantize(total_em),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.debug(
            "Allocation: method=%s, ratio=%s, tenant_em=%s, "
            "common_area=%s, total_tenant=%s",
            method_lower, alloc_ratio, tenant_em, tenant_ca, total_tenant,
        )
        return result

    # =========================================================================
    # VACANCY ADJUSTMENT
    # =========================================================================

    def apply_vacancy_adjustment(
        self,
        emissions_kgco2e: Union[Decimal, float, int, str],
        building_type: str,
        vacancy_fraction: Union[Decimal, float, int, str],
    ) -> Dict[str, Any]:
        """
        Apply vacancy period adjustment to building emissions.

        During vacancy periods, a base load fraction of energy continues
        to be consumed (HVAC setbacks, security lighting, etc.). This
        method adjusts full-occupancy emissions to account for partial
        vacancy.

        Formula:
            adjusted = emissions x (1 - vacancy + vacancy x base_load)

        Args:
            emissions_kgco2e: Full-occupancy emissions.
            building_type: Building type for base load lookup.
            vacancy_fraction: Fraction of period the space was vacant (0-1).

        Returns:
            Dict with adjusted emissions, vacancy details, and provenance.

        Raises:
            ValueError: If building_type or vacancy_fraction invalid.

        Example:
            >>> engine = AssetSpecificCalculatorEngine()
            >>> result = engine.apply_vacancy_adjustment(
            ...     emissions_kgco2e=10000,
            ...     building_type="office",
            ...     vacancy_fraction=0.25,
            ... )
            >>> result["adjusted_emissions_kgco2e"] < Decimal("10000")
            True
        """
        em = _to_decimal(emissions_kgco2e)
        self._validate_positive("emissions_kgco2e", em)

        vac = _to_decimal(vacancy_fraction)
        self._validate_fraction("vacancy_fraction", vac)

        base_load = self._db.get_vacancy_base_load(building_type)

        # adjusted = em x (1 - vac + vac * base_load)
        # = em x (1 - vac * (1 - base_load))
        adjustment_factor = _ONE - vac * (_ONE - base_load)
        adjusted = _quantize(em * adjustment_factor)
        reduction = _quantize(em - adjusted)

        result: Dict[str, Any] = {
            "original_emissions_kgco2e": _quantize(em),
            "adjusted_emissions_kgco2e": adjusted,
            "reduction_kgco2e": reduction,
            "vacancy_fraction": _quantize(vac),
            "base_load_fraction": _quantize(base_load),
            "adjustment_factor": _quantize(adjustment_factor),
            "building_type": building_type,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.debug(
            "Vacancy adjustment: type=%s, vacancy=%s, base_load=%s, "
            "original=%s, adjusted=%s, reduction=%s",
            building_type, vac, base_load, em, adjusted, reduction,
        )
        return result

    # =========================================================================
    # DATA QUALITY INDICATOR (DQI)
    # =========================================================================

    def compute_dqi_score(
        self,
        sub_metering_coverage: Optional[Union[Decimal, float]] = None,
        green_lease_clauses: Optional[List[str]] = None,
        has_metered_data: bool = True,
        temporal_score: Optional[int] = None,
        geographical_score: Optional[int] = None,
        technological_score: Optional[int] = None,
        completeness_score: Optional[int] = None,
        reliability_score: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute data quality indicator score for asset-specific calculations.

        Uses the GHG Protocol Scope 3 5-dimension DQI framework with
        adjustments for sub-metering coverage and green lease clauses.
        Asset-specific (Tier 1) calculations start with the best default
        scores.

        Dimensions (each scored 1=highest to 5=lowest):
            - Temporal: Data recency and reporting period match
            - Geographical: Regional specificity of emission factors
            - Technological: Technology representativeness
            - Completeness: Completeness of data coverage
            - Reliability: Data verification level

        Args:
            sub_metering_coverage: Fraction of space sub-metered (0-1).
                Improves completeness and reliability if high.
            green_lease_clauses: Active green lease clauses from tenant.
                Each clause provides a DQI improvement.
            has_metered_data: Whether calculation uses metered data.
            temporal_score: Override temporal dimension (1-5).
            geographical_score: Override geographical dimension (1-5).
            technological_score: Override technological dimension (1-5).
            completeness_score: Override completeness dimension (1-5).
            reliability_score: Override reliability dimension (1-5).

        Returns:
            Dict with dimension scores, weighted_average, and quality_level.

        Example:
            >>> engine = AssetSpecificCalculatorEngine()
            >>> dqi = engine.compute_dqi_score(
            ...     sub_metering_coverage=0.85,
            ...     green_lease_clauses=["energy_data_sharing", "sub_metering"],
            ... )
            >>> dqi["quality_level"]
            'high'
        """
        # Start with defaults for asset-specific tier
        tier = "asset_specific" if has_metered_data else "average_data"
        defaults = self._db.get_dqi_defaults(tier)

        scores = {
            "temporal": temporal_score or defaults["temporal"],
            "geographical": geographical_score or defaults["geographical"],
            "technological": technological_score or defaults["technological"],
            "completeness": completeness_score or defaults["completeness"],
            "reliability": reliability_score or defaults["reliability"],
        }

        # Clamp scores to [1, 5]
        for dim in scores:
            scores[dim] = max(1, min(5, scores[dim]))

        # Adjust for sub-metering coverage
        if sub_metering_coverage is not None:
            smc = float(sub_metering_coverage)
            if smc >= 0.9:
                scores["completeness"] = max(1, scores["completeness"] - 1)
                scores["reliability"] = max(1, scores["reliability"] - 1)
            elif smc >= 0.7:
                scores["completeness"] = max(1, scores["completeness"] - 1)
            elif smc < 0.3:
                scores["completeness"] = min(5, scores["completeness"] + 1)

        # Adjust for green lease clauses
        if green_lease_clauses:
            clause_db = self._db.get_green_lease_clauses()
            for clause_id in green_lease_clauses:
                if clause_id in clause_db:
                    clause = clause_db[clause_id]
                    if clause["tier_upgrade"]:
                        scores["reliability"] = max(
                            1, scores["reliability"] - 1
                        )

        # Weighted average
        weighted_sum = Decimal("0")
        for dim, score in scores.items():
            weight = DQI_DIMENSION_WEIGHTS.get(dim, Decimal("0.20"))
            weighted_sum += weight * Decimal(score)
        weighted_avg = _quantize(weighted_sum, _QUANT_2DP)

        # Quality level classification
        avg_float = float(weighted_avg)
        if avg_float <= 1.5:
            quality_level = "very_high"
        elif avg_float <= 2.5:
            quality_level = "high"
        elif avg_float <= 3.5:
            quality_level = "medium"
        elif avg_float <= 4.5:
            quality_level = "low"
        else:
            quality_level = "very_low"

        return {
            "scores": scores,
            "weighted_average": weighted_avg,
            "quality_level": quality_level,
            "tier": tier,
        }

    # =========================================================================
    # UNCERTAINTY
    # =========================================================================

    def compute_uncertainty(
        self,
        emissions_kgco2e: Union[Decimal, float, int, str],
        uncertainty_pct: Optional[Union[Decimal, float, int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute uncertainty range for asset-specific calculations.

        Asset-specific (Tier 1) calculations have +/-10% default uncertainty
        per IPCC 2006 Guidelines. Custom uncertainty can be specified.

        Args:
            emissions_kgco2e: Central emissions estimate.
            uncertainty_pct: Override uncertainty percentage. Default 10.

        Returns:
            Dict with uncertainty_pct, lower, upper, range_kgco2e.

        Example:
            >>> engine = AssetSpecificCalculatorEngine()
            >>> u = engine.compute_uncertainty(Decimal("1000"))
            >>> u["lower"]
            Decimal('900.00000000')
            >>> u["upper"]
            Decimal('1100.00000000')
        """
        em = _to_decimal(emissions_kgco2e)

        if uncertainty_pct is not None:
            pct = _to_decimal(uncertainty_pct)
        else:
            pct = self._db.get_uncertainty_default("asset_specific")

        fraction = _quantize(pct / _HUNDRED)
        lower = _quantize(em * (_ONE - fraction))
        upper = _quantize(em * (_ONE + fraction))
        range_val = _quantize(upper - lower)

        return {
            "uncertainty_pct": _quantize(pct),
            "lower": lower,
            "upper": upper,
            "range_kgco2e": range_val,
        }

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    def validate_inputs(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate calculation inputs before processing.

        Performs comprehensive validation of input fields including type
        checking, range validation, and asset-type-specific requirements
        without actually running the calculation.

        Args:
            inputs: Calculation input dict to validate.

        Returns:
            Dict with keys: valid (bool), errors (list), warnings (list).

        Example:
            >>> engine = AssetSpecificCalculatorEngine()
            >>> v = engine.validate_inputs({
            ...     "asset_category": "building",
            ...     "building_type": "office",
            ...     "region": "US",
            ...     "energy_consumption": {"electricity": {"kwh": 100000}},
            ...     "lease_share": 0.5,
            ... })
            >>> v["valid"]
            True
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Category check
        category = inputs.get("asset_category")
        if category is None:
            errors.append("Missing required field 'asset_category'.")
        else:
            cat_lower = str(category).strip().lower()
            valid_cats = {"building", "vehicle", "equipment", "it_asset"}
            if cat_lower not in valid_cats:
                errors.append(
                    f"Invalid asset_category '{category}'. "
                    f"Must be one of: {sorted(valid_cats)}"
                )

        # Region check
        region = inputs.get("region")
        if region is None:
            errors.append("Missing required field 'region'.")
        else:
            try:
                self._db.get_grid_ef(str(region))
            except ValueError as exc:
                errors.append(f"Invalid region: {exc}")

        # Lease share check
        lease_share = inputs.get("lease_share")
        if lease_share is not None:
            try:
                ls = _to_decimal(lease_share)
                if ls < _ZERO or ls > _ONE:
                    errors.append(
                        f"lease_share must be 0-1, got {ls}"
                    )
            except (TypeError, ValueError) as exc:
                errors.append(f"Invalid lease_share: {exc}")
        else:
            warnings.append("lease_share not provided, will default to 1.0.")

        # Category-specific validation
        if category is not None:
            cat_lower = str(category).strip().lower()

            if cat_lower == "building":
                if not inputs.get("building_type"):
                    errors.append("Missing building_type for building asset.")
                else:
                    try:
                        bt = str(inputs["building_type"]).strip().lower()
                        self._db.get_vacancy_base_load(bt)
                    except ValueError as exc:
                        errors.append(f"Invalid building_type: {exc}")

                if not inputs.get("energy_consumption"):
                    errors.append(
                        "Missing energy_consumption for building asset."
                    )

            elif cat_lower == "vehicle":
                if not inputs.get("vehicle_type"):
                    errors.append("Missing vehicle_type for vehicle asset.")
                if not inputs.get("fuel_type"):
                    errors.append("Missing fuel_type for vehicle asset.")
                has_consumption = any(
                    inputs.get(k) is not None for k in [
                        "fuel_consumed_litres", "fuel_consumed_m3",
                        "distance_km", "bev_energy_kwh",
                    ]
                )
                if not has_consumption:
                    errors.append(
                        "Vehicle requires fuel_consumed_litres, "
                        "fuel_consumed_m3, distance_km, or bev_energy_kwh."
                    )

            elif cat_lower == "equipment":
                if not inputs.get("equipment_type"):
                    errors.append("Missing equipment_type for equipment asset.")
                has_data = inputs.get("fuel_consumed_litres") is not None or \
                           inputs.get("operating_hours") is not None
                if not has_data:
                    errors.append(
                        "Equipment requires fuel_consumed_litres or "
                        "operating_hours."
                    )

            elif cat_lower == "it_asset":
                if not inputs.get("it_type"):
                    errors.append("Missing it_type for IT asset.")

        is_valid = len(errors) == 0
        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
        }

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the calculator engine.

        Verifies database engine connectivity, runs a sample calculation,
        and returns engine statistics.

        Returns:
            Dict with status, db_health, sample_calc, calc_count,
            error_count, created_at.

        Example:
            >>> engine = AssetSpecificCalculatorEngine()
            >>> health = engine.health_check()
            >>> health["status"]
            'healthy'
        """
        db_health = self._db.health_check()

        # Run a sample calculation
        sample_ok = False
        sample_error = None
        try:
            sample_result = self.calculate_it_asset({
                "it_type": "server",
                "region": "US",
                "quantity": 1,
                "lease_share": 1.0,
            })
            sample_ok = sample_result.get("total_emissions_kgco2e", _ZERO) > _ZERO
        except Exception as exc:
            sample_error = str(exc)

        status = "healthy" if (
            db_health.get("status") == "healthy" and sample_ok
        ) else "degraded"

        return {
            "status": status,
            "agent_id": AGENT_ID,
            "engine": "AssetSpecificCalculatorEngine",
            "version": VERSION,
            "db_health": db_health.get("status", "unknown"),
            "sample_calculation_ok": sample_ok,
            "sample_calculation_error": sample_error,
            "calc_count": self._calc_count,
            "error_count": self._error_count,
            "created_at": self._created_at.isoformat(),
        }

    # =========================================================================
    # RESET (for testing)
    # =========================================================================

    @classmethod
    def _reset_singleton(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        WARNING: This method is intended for unit testing only. Do not call
        in production code.
        """
        with cls._lock:
            cls._instance = None
        logger.warning("AssetSpecificCalculatorEngine singleton reset.")

    # =========================================================================
    # REPR / STR
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation of the engine."""
        return (
            f"AssetSpecificCalculatorEngine("
            f"agent_id='{AGENT_ID}', "
            f"version='{VERSION}', "
            f"calcs={self._calc_count}, "
            f"errors={self._error_count})"
        )

    def __str__(self) -> str:
        """Return human-readable string of the engine."""
        return (
            f"AssetSpecificCalculatorEngine v{VERSION} "
            f"({self._calc_count} calculations, {self._error_count} errors)"
        )
