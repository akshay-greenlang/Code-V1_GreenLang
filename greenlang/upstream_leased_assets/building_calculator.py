# -*- coding: utf-8 -*-
"""
BuildingCalculatorEngine - Engine 2: Upstream Leased Assets Agent (AGENT-MRV-021)

Calculates GHG emissions from leased buildings (office, retail, warehouse,
industrial, data center, hotel, healthcare, education) using four methods:
asset-specific (metered energy), average-data (benchmark EUI), lessor-specific
(primary data from landlord), and spend-based (EEIO).

Primary Formulae:
    Asset-Specific:
        elec_co2e      = electricity_kwh x grid_ef
        gas_co2e       = gas_kwh x gas_ef
        heating_co2e   = heating_kwh x dh_ef
        cooling_co2e   = cooling_kwh x dc_ef
        refrigerant_co2e = leakage_kg x gwp
        wtt_co2e       = elec x wtt_grid + gas x wtt_gas + heat x wtt_dh + cool x wtt_dc
        total_co2e     = (elec + gas + heat + cool + refrig + wtt) x allocation x (months/12)

    Average-Data:
        total_energy   = floor_area_sqm x eui_kwh_sqm
        elec_energy    = total_energy x (1 - gas_fraction)
        gas_energy     = total_energy x gas_fraction
        elec_co2e      = elec_energy x grid_ef
        gas_co2e       = gas_energy x gas_ef
        gas_ef_adj     = gas_fraction x (gas_ef / grid_ef - 1)
        total_co2e     = floor_area x eui x grid_ef x (1 + gas_ef_adj) x alloc x (months/12)

    Lessor-Specific:
        total_co2e     = lessor_co2e_kg x allocation_factor x (lease_months / 12)

    Spend-Based:
        usd_amount     = amount x currency_rate
        deflated       = usd_amount / cpi_deflator(reporting_year) x cpi_deflator(2021)
        total_co2e     = deflated x eeio_factor

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate value is deterministic and reproducible
    - SHA-256 provenance hash on every result
    - Emission factors sourced from DEFRA 2024 / EPA 2024 / IEA 2024

Supports:
    - 8 building types x 5 climate zones
    - Asset-specific with full energy breakdown (elec/gas/heating/cooling/refrig)
    - Average-data with climate-zone-adjusted EUI
    - Lessor-specific with validation flag
    - Spend-based with CPI deflation and currency conversion
    - Multi-tenant allocation (floor_area, headcount, revenue, equal_share)
    - Lease period proration (months/12)
    - WTT (well-to-tank) upstream emissions
    - Refrigerant leakage emissions
    - Batch processing for multiple buildings
    - Building comparison and ranking by CO2e intensity
    - Quick annual emissions estimate
    - Data Quality Indicator (DQI) scoring
    - Provenance hash integration for audit trails

Example:
    >>> from greenlang.upstream_leased_assets.building_calculator import (
    ...     get_building_calculator,
    ... )
    >>> from decimal import Decimal
    >>> engine = get_building_calculator()
    >>> result = engine.calculate_building_asset_specific(
    ...     building_type="office",
    ...     floor_area_sqm=Decimal("5000"),
    ...     electricity_kwh=Decimal("750000"),
    ...     gas_kwh=Decimal("250000"),
    ...     country_code="US",
    ... )
    >>> assert result["co2e_kg"] > Decimal("0")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-021 Upstream Leased Assets (GL-MRV-S3-008)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.upstream_leased_assets.upstream_leased_database import (
    UpstreamLeasedDatabaseEngine,
    get_database_engine,
    BUILDING_EUI_DATA,
    GRID_EMISSION_FACTORS,
    FUEL_EMISSION_FACTORS,
    EEIO_FACTORS,
    DISTRICT_HEATING_FACTORS,
    ENGINE_ID as DB_ENGINE_ID,
)

logger = logging.getLogger(__name__)

# =============================================================================
# ENGINE METADATA
# =============================================================================

ENGINE_ID: str = "building_calculator_engine"
ENGINE_VERSION: str = "1.0.0"

# =============================================================================
# DECIMAL PRECISION & CONSTANTS
# =============================================================================

_QUANT_8DP = Decimal("0.00000001")
_PRECISION = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWELVE = Decimal("12")
ROUNDING = ROUND_HALF_UP

# Batch processing limits
_MAX_BATCH_SIZE = 10000

# DQI scoring weights (sum to 1.0)
_DQI_METHOD_WEIGHT = Decimal("0.40")
_DQI_TEMPORAL_WEIGHT = Decimal("0.20")
_DQI_GEO_WEIGHT = Decimal("0.20")
_DQI_COMPLETENESS_WEIGHT = Decimal("0.20")

# DQI method scores (higher is better, 1-5 scale)
_DQI_METHOD_SCORES: Dict[str, Decimal] = {
    "asset_specific": Decimal("5.0"),
    "lessor_specific": Decimal("4.0"),
    "average_data": Decimal("3.0"),
    "spend_based": Decimal("2.0"),
}

# Calculation method labels
_METHOD_ASSET_SPECIFIC = "asset_specific"
_METHOD_AVERAGE_DATA = "average_data"
_METHOD_LESSOR_SPECIFIC = "lessor_specific"
_METHOD_SPEND_BASED = "spend_based"

# EEIO base year
_EEIO_BASE_YEAR = 2021


# =============================================================================
# PROVENANCE HASH UTILITY
# =============================================================================


def _calculate_provenance_hash(data: Dict[str, Any]) -> str:
    """
    Calculate SHA-256 provenance hash for audit trail.

    Serializes the input dictionary to a canonical JSON string
    (sorted keys, Decimal values converted to strings) and
    returns its SHA-256 hex digest.

    Args:
        data: Dictionary of values to hash.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    def _serialize_value(v: Any) -> Any:
        """Convert Decimal and datetime to serializable form."""
        if isinstance(v, Decimal):
            return str(v)
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, dict):
            return {k: _serialize_value(val) for k, val in sorted(v.items())}
        if isinstance(v, (list, tuple)):
            return [_serialize_value(item) for item in v]
        return v

    serializable = _serialize_value(data)
    json_str = json.dumps(serializable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


# =============================================================================
# INPUT VALIDATION HELPERS
# =============================================================================


def _to_decimal(
    value: Union[Decimal, float, int, str],
    field_name: str,
) -> Decimal:
    """
    Convert a value to Decimal with validation.

    Args:
        value: Input value to convert.
        field_name: Field name for error messages.

    Returns:
        Decimal representation of the value.

    Raises:
        ValueError: If the value cannot be converted to Decimal.
        TypeError: If the value type is not supported.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(
                f"Cannot convert {field_name}={value} to Decimal: {exc}"
            ) from exc
    if isinstance(value, str):
        try:
            return Decimal(value)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(
                f"Cannot convert {field_name}='{value}' to Decimal: {exc}"
            ) from exc
    raise TypeError(
        f"Unsupported type for {field_name}: {type(value).__name__}. "
        f"Expected Decimal, float, int, or str."
    )


def _validate_positive(
    value: Decimal,
    field_name: str,
    allow_zero: bool = True,
) -> Decimal:
    """
    Validate that a Decimal value is non-negative (or strictly positive).

    Args:
        value: Decimal value to validate.
        field_name: Field name for error messages.
        allow_zero: If True, zero is acceptable. If False, must be > 0.

    Returns:
        The validated Decimal value.

    Raises:
        ValueError: If the value fails validation.
    """
    if allow_zero and value < _ZERO:
        raise ValueError(f"{field_name} must be >= 0, got {value}")
    if not allow_zero and value <= _ZERO:
        raise ValueError(f"{field_name} must be > 0, got {value}")
    return value


def _validate_allocation_factor(value: Decimal) -> Decimal:
    """
    Validate allocation factor is between 0 and 1 (inclusive).

    Args:
        value: Allocation factor to validate.

    Returns:
        The validated Decimal value.

    Raises:
        ValueError: If outside [0, 1] range.
    """
    if value < _ZERO or value > _ONE:
        raise ValueError(
            f"allocation_factor must be between 0.0 and 1.0, got {value}"
        )
    return value


def _validate_lease_months(value: int) -> int:
    """
    Validate lease months is between 1 and 12.

    Args:
        value: Number of lease months in the reporting period.

    Returns:
        The validated integer value.

    Raises:
        ValueError: If outside [1, 12] range.
    """
    if value < 1 or value > 12:
        raise ValueError(
            f"lease_months must be between 1 and 12, got {value}"
        )
    return value


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal value to 8 decimal places with ROUND_HALF_UP."""
    return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


# =============================================================================
# DQI SCORING
# =============================================================================


def _calculate_dqi_score(
    method: str,
    has_country_specific_ef: bool = True,
    has_wtt: bool = True,
    has_refrigerant: bool = False,
    has_all_energy_streams: bool = True,
) -> Decimal:
    """
    Calculate Data Quality Indicator (DQI) score on a 1-5 scale.

    The DQI score is a weighted average of:
    - Method quality (40%): asset_specific=5, lessor=4, average=3, spend=2
    - Temporal relevance (20%): always 4.0 (current-year factors)
    - Geographic specificity (20%): 5.0 if country-specific, 3.0 if global
    - Completeness (20%): based on presence of WTT, refrigerant, all streams

    Args:
        method: Calculation method used.
        has_country_specific_ef: Whether country-specific EF was used.
        has_wtt: Whether WTT emissions are included.
        has_refrigerant: Whether refrigerant emissions are included.
        has_all_energy_streams: Whether all energy streams were provided.

    Returns:
        DQI score as Decimal (1.0-5.0 scale), quantized to 8dp.
    """
    # Method quality score
    method_score = _DQI_METHOD_SCORES.get(method, Decimal("2.0"))

    # Temporal relevance (current year factors = 4.0)
    temporal_score = Decimal("4.0")

    # Geographic specificity
    geo_score = Decimal("5.0") if has_country_specific_ef else Decimal("3.0")

    # Completeness score
    completeness_items = 0
    completeness_max = 3
    if has_wtt:
        completeness_items += 1
    if has_refrigerant:
        completeness_items += 1
    if has_all_energy_streams:
        completeness_items += 1
    completeness_score = (
        Decimal("2.0")
        + Decimal("3.0") * Decimal(str(completeness_items)) / Decimal(str(completeness_max))
    )

    # Weighted average
    dqi = (
        method_score * _DQI_METHOD_WEIGHT
        + temporal_score * _DQI_TEMPORAL_WEIGHT
        + geo_score * _DQI_GEO_WEIGHT
        + completeness_score * _DQI_COMPLETENESS_WEIGHT
    )

    return _quantize(dqi)


# =============================================================================
# ENGINE CLASS
# =============================================================================


class BuildingCalculatorEngine:
    """
    Thread-safe singleton engine for leased building emissions calculations.

    Provides four calculation methods for upstream leased building assets:
    asset-specific (metered energy data), average-data (benchmark EUI by type
    and climate zone), lessor-specific (primary data from landlord), and
    spend-based (EEIO with CPI deflation and currency conversion).

    Also provides multi-tenant allocation, batch processing, building comparison,
    and quick annual estimation capabilities.

    This engine does NOT perform any LLM calls. All calculations use deterministic
    Python Decimal arithmetic with emission factors from the
    UpstreamLeasedDatabaseEngine.

    Thread Safety:
        Uses the __new__ singleton pattern with threading.Lock to ensure
        only one instance is created across all threads.

    Attributes:
        ENGINE_ID: Unique engine identifier string.
        ENGINE_VERSION: Semantic version of the engine.
        _db: Reference to the UpstreamLeasedDatabaseEngine singleton.
        _calc_count: Total number of calculations performed.

    Example:
        >>> engine = BuildingCalculatorEngine()
        >>> result = engine.calculate_building_asset_specific(
        ...     building_type="office",
        ...     floor_area_sqm=Decimal("5000"),
        ...     electricity_kwh=Decimal("750000"),
        ...     country_code="US",
        ... )
        >>> result["co2e_kg"] > Decimal("0")
        True
    """

    ENGINE_ID: str = ENGINE_ID
    ENGINE_VERSION: str = ENGINE_VERSION

    _instance: Optional["BuildingCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "BuildingCalculatorEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the building calculator engine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._db: UpstreamLeasedDatabaseEngine = get_database_engine()
        self._calc_count: int = 0
        self._calc_lock: threading.Lock = threading.Lock()

        logger.info(
            "BuildingCalculatorEngine initialized: engine_id=%s, version=%s, "
            "db_engine=%s",
            self.ENGINE_ID,
            self.ENGINE_VERSION,
            self._db.ENGINE_ID,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_calc(self) -> None:
        """Increment the calculation counter in a thread-safe manner."""
        with self._calc_lock:
            self._calc_count += 1

    def get_calc_count(self) -> int:
        """
        Get the total number of calculations performed.

        Returns:
            Integer count of calculations.
        """
        with self._calc_lock:
            return self._calc_count

    def _build_result(
        self,
        method: str,
        building_type: str,
        floor_area_sqm: Decimal,
        elec_co2e_kg: Decimal,
        gas_co2e_kg: Decimal,
        heating_co2e_kg: Decimal,
        cooling_co2e_kg: Decimal,
        refrigerant_co2e_kg: Decimal,
        wtt_co2e_kg: Decimal,
        allocation_factor: Decimal,
        lease_months: int,
        ef_source: str,
        country_code: str = "US",
        dqi_score: Optional[Decimal] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a standardized result dictionary with provenance hash.

        Calculates the total CO2e from component emissions and applies
        allocation factor and lease proration. Generates SHA-256 provenance
        hash over all inputs and outputs.

        Args:
            method: Calculation method used.
            building_type: Building type string.
            floor_area_sqm: Floor area in square meters.
            elec_co2e_kg: Electricity emissions in kgCO2e.
            gas_co2e_kg: Natural gas emissions in kgCO2e.
            heating_co2e_kg: District heating emissions in kgCO2e.
            cooling_co2e_kg: District cooling emissions in kgCO2e.
            refrigerant_co2e_kg: Refrigerant leakage emissions in kgCO2e.
            wtt_co2e_kg: Well-to-tank upstream emissions in kgCO2e.
            allocation_factor: Multi-tenant allocation factor (0-1).
            lease_months: Number of months in reporting period (1-12).
            ef_source: Emission factor source string.
            country_code: Country code for the building location.
            dqi_score: Pre-calculated DQI score, or None to auto-calculate.
            extra_fields: Additional fields to include in the result.

        Returns:
            Standardized result dictionary with all required fields.
        """
        # Calculate subtotal before allocation and proration
        subtotal = (
            elec_co2e_kg
            + gas_co2e_kg
            + heating_co2e_kg
            + cooling_co2e_kg
            + refrigerant_co2e_kg
            + wtt_co2e_kg
        )

        # Apply allocation factor and lease period proration
        lease_fraction = Decimal(str(lease_months)) / _TWELVE
        co2e_kg = _quantize(subtotal * allocation_factor * lease_fraction)

        # Calculate DQI if not provided
        if dqi_score is None:
            has_country_ef = country_code.upper() in GRID_EMISSION_FACTORS
            has_wtt = wtt_co2e_kg > _ZERO
            has_refrig = refrigerant_co2e_kg > _ZERO
            has_all_streams = (
                elec_co2e_kg > _ZERO
                or gas_co2e_kg > _ZERO
                or heating_co2e_kg > _ZERO
            )
            dqi_score = _calculate_dqi_score(
                method=method,
                has_country_specific_ef=has_country_ef,
                has_wtt=has_wtt,
                has_refrigerant=has_refrig,
                has_all_energy_streams=has_all_streams,
            )

        # Build the result dictionary
        result: Dict[str, Any] = {
            "co2e_kg": co2e_kg,
            "method": method,
            "building_type": building_type,
            "floor_area_sqm": _quantize(floor_area_sqm),
            "elec_co2e_kg": _quantize(elec_co2e_kg),
            "gas_co2e_kg": _quantize(gas_co2e_kg),
            "heating_co2e_kg": _quantize(heating_co2e_kg),
            "cooling_co2e_kg": _quantize(cooling_co2e_kg),
            "refrigerant_co2e_kg": _quantize(refrigerant_co2e_kg),
            "wtt_co2e_kg": _quantize(wtt_co2e_kg),
            "allocation_factor": _quantize(allocation_factor),
            "lease_months": lease_months,
            "ef_source": ef_source,
            "dqi_score": dqi_score,
            "country_code": country_code.upper(),
            "engine_id": self.ENGINE_ID,
            "engine_version": self.ENGINE_VERSION,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add extra fields if provided
        if extra_fields:
            result.update(extra_fields)

        # Calculate provenance hash over the result (excluding the hash itself)
        result["provenance_hash"] = _calculate_provenance_hash(result)

        return result

    # =========================================================================
    # 1. ASSET-SPECIFIC CALCULATION
    # =========================================================================

    def calculate_building_asset_specific(
        self,
        building_type: str,
        floor_area_sqm: Union[Decimal, float, int, str],
        electricity_kwh: Union[Decimal, float, int, str],
        gas_kwh: Union[Decimal, float, int, str] = 0,
        heating_kwh: Union[Decimal, float, int, str] = 0,
        cooling_kwh: Union[Decimal, float, int, str] = 0,
        country_code: str = "US",
        allocation_factor: Union[Decimal, float, str] = 1.0,
        lease_months: int = 12,
        include_wtt: bool = True,
        include_refrigerants: bool = False,
        refrigerant: Optional[str] = None,
        leakage_kg: Union[Decimal, float, int, str] = 0,
    ) -> Dict[str, Any]:
        """
        Calculate building emissions using asset-specific metered energy data.

        This is the highest-quality calculation method (GHG Protocol preferred)
        using actual metered energy consumption data for the leased building.

        Formula:
            elec_co2e      = electricity_kwh x grid_ef(country)
            gas_co2e       = gas_kwh x gas_ef
            heating_co2e   = heating_kwh x district_heating_ef
            cooling_co2e   = cooling_kwh x district_cooling_ef
            refrig_co2e    = leakage_kg x gwp(refrigerant)
            wtt_co2e       = sum of WTT for each energy stream
            total_co2e     = sum(all) x allocation_factor x (lease_months / 12)

        Args:
            building_type: Building type for reporting (office, retail, etc.).
            floor_area_sqm: Total floor area of the leased space (sqm).
            electricity_kwh: Annual electricity consumption (kWh).
            gas_kwh: Annual natural gas consumption (kWh). Defaults to 0.
            heating_kwh: Annual district heating consumption (kWh). Defaults to 0.
            cooling_kwh: Annual district cooling consumption (kWh). Defaults to 0.
            country_code: ISO country code for grid EF selection. Defaults to "US".
            allocation_factor: Lessee share of emissions (0.0-1.0). Defaults to 1.0.
            lease_months: Months of lease in reporting period (1-12). Defaults to 12.
            include_wtt: Include well-to-tank upstream emissions. Defaults to True.
            include_refrigerants: Include refrigerant leakage. Defaults to False.
            refrigerant: Refrigerant type (e.g., "R-410A"). Required if
                include_refrigerants is True.
            leakage_kg: Refrigerant leakage amount in kg. Defaults to 0.

        Returns:
            Dict with co2e_kg, method, building_type, floor_area_sqm,
            elec_co2e_kg, gas_co2e_kg, heating_co2e_kg, cooling_co2e_kg,
            refrigerant_co2e_kg, wtt_co2e_kg, allocation_factor, lease_months,
            provenance_hash, ef_source, dqi_score.

        Raises:
            ValueError: If input validation fails.
            TypeError: If input types cannot be converted.

        Example:
            >>> engine = BuildingCalculatorEngine()
            >>> result = engine.calculate_building_asset_specific(
            ...     building_type="office",
            ...     floor_area_sqm=Decimal("5000"),
            ...     electricity_kwh=Decimal("750000"),
            ...     gas_kwh=Decimal("250000"),
            ...     country_code="US",
            ... )
            >>> result["method"]
            'asset_specific'
        """
        start_time = time.monotonic()
        self._increment_calc()

        # Input conversion and validation
        floor_area = _validate_positive(
            _to_decimal(floor_area_sqm, "floor_area_sqm"),
            "floor_area_sqm",
            allow_zero=False,
        )
        elec_kwh = _validate_positive(
            _to_decimal(electricity_kwh, "electricity_kwh"),
            "electricity_kwh",
        )
        gas = _validate_positive(
            _to_decimal(gas_kwh, "gas_kwh"),
            "gas_kwh",
        )
        heating = _validate_positive(
            _to_decimal(heating_kwh, "heating_kwh"),
            "heating_kwh",
        )
        cooling = _validate_positive(
            _to_decimal(cooling_kwh, "cooling_kwh"),
            "cooling_kwh",
        )
        alloc = _validate_allocation_factor(
            _to_decimal(allocation_factor, "allocation_factor")
        )
        _validate_lease_months(lease_months)
        leak_kg = _validate_positive(
            _to_decimal(leakage_kg, "leakage_kg"),
            "leakage_kg",
        )

        btype = building_type.lower().strip().replace(" ", "_")
        country = country_code.upper().strip()

        # Look up emission factors
        grid_ef = self._db.get_grid_emission_factor(country)
        gas_ef = self._db.get_fuel_emission_factor("natural_gas")
        dh_ef = self._db.get_fuel_emission_factor("district_heating")
        dc_ef = self._db.get_fuel_emission_factor("district_cooling")

        # Calculate electricity emissions
        elec_co2e = _quantize(elec_kwh * grid_ef["co2e_per_kwh"])

        # Calculate gas emissions
        gas_co2e = _quantize(gas * gas_ef["ef_per_kwh"])

        # Calculate district heating emissions
        heating_co2e = _quantize(heating * dh_ef["ef_per_kwh"])

        # Calculate district cooling emissions
        cooling_co2e = _quantize(cooling * dc_ef["ef_per_kwh"])

        # Calculate refrigerant emissions
        refrigerant_co2e = _ZERO
        if include_refrigerants and refrigerant is not None and leak_kg > _ZERO:
            gwp_data = self._db.get_refrigerant_gwp(refrigerant)
            refrigerant_co2e = _quantize(leak_kg * gwp_data["gwp_ar6"])

        # Calculate WTT emissions
        wtt_co2e = _ZERO
        if include_wtt:
            wtt_elec = _quantize(elec_kwh * grid_ef["wtt_per_kwh"])
            wtt_gas = _quantize(gas * gas_ef["wtt_per_kwh"])
            wtt_heating = _quantize(heating * dh_ef["wtt_per_kwh"])
            wtt_cooling = _quantize(cooling * dc_ef["wtt_per_kwh"])
            wtt_co2e = _quantize(wtt_elec + wtt_gas + wtt_heating + wtt_cooling)

        # Build EF source string
        ef_sources = [grid_ef["source"]]
        if gas > _ZERO:
            ef_sources.append(gas_ef["source"])
        if heating > _ZERO:
            ef_sources.append(dh_ef["source"])
        if cooling > _ZERO:
            ef_sources.append(dc_ef["source"])
        ef_source = " / ".join(sorted(set(ef_sources)))

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Asset-specific calculation: building=%s, area=%s sqm, "
            "country=%s, elec=%s kWh, gas=%s kWh, elapsed=%.2f ms",
            btype, floor_area, country, elec_kwh, gas, elapsed_ms,
        )

        return self._build_result(
            method=_METHOD_ASSET_SPECIFIC,
            building_type=btype,
            floor_area_sqm=floor_area,
            elec_co2e_kg=elec_co2e,
            gas_co2e_kg=gas_co2e,
            heating_co2e_kg=heating_co2e,
            cooling_co2e_kg=cooling_co2e,
            refrigerant_co2e_kg=refrigerant_co2e,
            wtt_co2e_kg=wtt_co2e,
            allocation_factor=alloc,
            lease_months=lease_months,
            ef_source=ef_source,
            country_code=country,
            extra_fields={
                "electricity_kwh": _quantize(elec_kwh),
                "gas_kwh": _quantize(gas),
                "heating_kwh": _quantize(heating),
                "cooling_kwh": _quantize(cooling),
                "include_wtt": include_wtt,
                "include_refrigerants": include_refrigerants,
                "refrigerant": refrigerant,
                "leakage_kg": _quantize(leak_kg),
                "processing_time_ms": round(elapsed_ms, 2),
            },
        )

    # =========================================================================
    # 2. AVERAGE-DATA CALCULATION
    # =========================================================================

    def calculate_building_average_data(
        self,
        building_type: str,
        floor_area_sqm: Union[Decimal, float, int, str],
        climate_zone: str = "temperate",
        country_code: str = "US",
        allocation_factor: Union[Decimal, float, str] = 1.0,
        lease_months: int = 12,
    ) -> Dict[str, Any]:
        """
        Calculate building emissions using average-data benchmark EUI.

        Uses building type and climate zone to determine Energy Use Intensity
        (EUI, kWh/sqm/year), then applies grid and gas emission factors.

        Formula:
            total_energy   = floor_area_sqm x eui_kwh_sqm
            gas_energy     = total_energy x gas_fraction
            elec_energy    = total_energy x (1 - gas_fraction)
            elec_co2e      = elec_energy x grid_ef
            gas_co2e       = gas_energy x gas_ef
            wtt_co2e       = elec_energy x wtt_grid + gas_energy x wtt_gas
            total_co2e     = (elec + gas + wtt) x allocation x (months/12)

        Simplified (equivalent):
            gas_ef_adj     = gas_fraction x (gas_ef / grid_ef - 1)
            total_co2e     = floor_area x eui x grid_ef x (1 + gas_ef_adj) x alloc x (m/12)

        Args:
            building_type: Building type (office, retail, warehouse, industrial,
                data_center, hotel, healthcare, education).
            floor_area_sqm: Total floor area of the leased space (sqm).
            climate_zone: Climate zone (tropical, arid, temperate, continental,
                polar). Defaults to "temperate".
            country_code: ISO country code for grid EF selection. Defaults to "US".
            allocation_factor: Lessee share of emissions (0.0-1.0). Defaults to 1.0.
            lease_months: Months of lease in reporting period (1-12). Defaults to 12.

        Returns:
            Dict with co2e_kg, method, building_type, floor_area_sqm,
            elec_co2e_kg, gas_co2e_kg, heating_co2e_kg, cooling_co2e_kg,
            refrigerant_co2e_kg, wtt_co2e_kg, allocation_factor, lease_months,
            provenance_hash, ef_source, dqi_score, plus climate_zone and
            eui_kwh_sqm.

        Raises:
            ValueError: If input validation fails.

        Example:
            >>> engine = BuildingCalculatorEngine()
            >>> result = engine.calculate_building_average_data(
            ...     building_type="office",
            ...     floor_area_sqm=Decimal("5000"),
            ...     climate_zone="temperate",
            ...     country_code="US",
            ... )
            >>> result["method"]
            'average_data'
        """
        start_time = time.monotonic()
        self._increment_calc()

        # Input conversion and validation
        floor_area = _validate_positive(
            _to_decimal(floor_area_sqm, "floor_area_sqm"),
            "floor_area_sqm",
            allow_zero=False,
        )
        alloc = _validate_allocation_factor(
            _to_decimal(allocation_factor, "allocation_factor")
        )
        _validate_lease_months(lease_months)

        btype = building_type.lower().strip().replace(" ", "_")
        zone = climate_zone.lower().strip()
        country = country_code.upper().strip()

        # Look up EUI and emission factors
        eui_data = self._db.get_building_eui(btype, zone)
        grid_ef = self._db.get_grid_emission_factor(country)
        gas_ef = self._db.get_fuel_emission_factor("natural_gas")

        eui = eui_data["eui_kwh_sqm"]
        gas_fraction = eui_data["gas_fraction"]

        # Calculate total energy
        total_energy = _quantize(floor_area * eui)

        # Split into electricity and gas
        gas_energy = _quantize(total_energy * gas_fraction)
        elec_energy = _quantize(total_energy - gas_energy)

        # Calculate emissions
        elec_co2e = _quantize(elec_energy * grid_ef["co2e_per_kwh"])
        gas_co2e = _quantize(gas_energy * gas_ef["ef_per_kwh"])

        # Calculate WTT emissions
        wtt_elec = _quantize(elec_energy * grid_ef["wtt_per_kwh"])
        wtt_gas = _quantize(gas_energy * gas_ef["wtt_per_kwh"])
        wtt_co2e = _quantize(wtt_elec + wtt_gas)

        ef_source = f"{eui_data['source']} / {grid_ef['source']} / {gas_ef['source']}"

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Average-data calculation: building=%s, area=%s sqm, "
            "zone=%s, country=%s, eui=%s, elapsed=%.2f ms",
            btype, floor_area, zone, country, eui, elapsed_ms,
        )

        return self._build_result(
            method=_METHOD_AVERAGE_DATA,
            building_type=btype,
            floor_area_sqm=floor_area,
            elec_co2e_kg=elec_co2e,
            gas_co2e_kg=gas_co2e,
            heating_co2e_kg=_ZERO,
            cooling_co2e_kg=_ZERO,
            refrigerant_co2e_kg=_ZERO,
            wtt_co2e_kg=wtt_co2e,
            allocation_factor=alloc,
            lease_months=lease_months,
            ef_source=ef_source,
            country_code=country,
            extra_fields={
                "climate_zone": zone,
                "eui_kwh_sqm": _quantize(eui),
                "gas_fraction": _quantize(gas_fraction),
                "total_energy_kwh": total_energy,
                "elec_energy_kwh": elec_energy,
                "gas_energy_kwh": gas_energy,
                "processing_time_ms": round(elapsed_ms, 2),
            },
        )

    # =========================================================================
    # 3. LESSOR-SPECIFIC CALCULATION
    # =========================================================================

    def calculate_building_lessor(
        self,
        lessor_co2e_kg: Union[Decimal, float, int, str],
        allocation_factor: Union[Decimal, float, str] = 1.0,
        lease_months: int = 12,
        validated: bool = False,
        building_type: str = "unknown",
        floor_area_sqm: Union[Decimal, float, int, str] = 0,
    ) -> Dict[str, Any]:
        """
        Calculate building emissions using lessor-provided primary data.

        The simplest method: takes the total CO2e value reported by the lessor
        (landlord) and applies the lessee's allocation factor and lease proration.

        Formula:
            total_co2e = lessor_co2e_kg x allocation_factor x (lease_months / 12)

        Args:
            lessor_co2e_kg: Total annual CO2e reported by lessor (kgCO2e).
            allocation_factor: Lessee share (0.0-1.0). Defaults to 1.0.
            lease_months: Months of lease in reporting period (1-12). Defaults to 12.
            validated: Whether the lessor data has been externally validated
                (e.g., third-party verified). Defaults to False.
            building_type: Building type for reporting. Defaults to "unknown".
            floor_area_sqm: Floor area for reporting. Defaults to 0.

        Returns:
            Dict with co2e_kg, method, building_type, floor_area_sqm,
            and all standard fields. Lessor-specific results have all
            energy-stream fields set to zero except co2e_kg.

        Raises:
            ValueError: If input validation fails.

        Example:
            >>> engine = BuildingCalculatorEngine()
            >>> result = engine.calculate_building_lessor(
            ...     lessor_co2e_kg=Decimal("150000"),
            ...     allocation_factor=Decimal("0.5"),
            ...     validated=True,
            ... )
            >>> result["co2e_kg"]
            Decimal('75000.00000000')
        """
        start_time = time.monotonic()
        self._increment_calc()

        # Input conversion and validation
        lessor_co2e = _validate_positive(
            _to_decimal(lessor_co2e_kg, "lessor_co2e_kg"),
            "lessor_co2e_kg",
        )
        alloc = _validate_allocation_factor(
            _to_decimal(allocation_factor, "allocation_factor")
        )
        _validate_lease_months(lease_months)
        floor_area = _validate_positive(
            _to_decimal(floor_area_sqm, "floor_area_sqm"),
            "floor_area_sqm",
        )

        btype = building_type.lower().strip().replace(" ", "_")

        # DQI score: higher if validated
        dqi_base = _DQI_METHOD_SCORES[_METHOD_LESSOR_SPECIFIC]
        if validated:
            dqi_score = _quantize(dqi_base + Decimal("0.5"))
        else:
            dqi_score = _quantize(dqi_base - Decimal("0.5"))
        # Clamp to 1-5
        dqi_score = max(Decimal("1.0"), min(Decimal("5.0"), dqi_score))
        dqi_score = _quantize(dqi_score)

        ef_source = "Lessor-provided data"
        if validated:
            ef_source += " (third-party verified)"

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Lessor-specific calculation: building=%s, lessor_co2e=%s kg, "
            "validated=%s, elapsed=%.2f ms",
            btype, lessor_co2e, validated, elapsed_ms,
        )

        # For lessor-specific, the entire CO2e is assigned to elec_co2e_kg
        # as a single bucket (breakdown not available from lessor)
        return self._build_result(
            method=_METHOD_LESSOR_SPECIFIC,
            building_type=btype,
            floor_area_sqm=floor_area,
            elec_co2e_kg=lessor_co2e,
            gas_co2e_kg=_ZERO,
            heating_co2e_kg=_ZERO,
            cooling_co2e_kg=_ZERO,
            refrigerant_co2e_kg=_ZERO,
            wtt_co2e_kg=_ZERO,
            allocation_factor=alloc,
            lease_months=lease_months,
            ef_source=ef_source,
            country_code="GLOBAL",
            dqi_score=dqi_score,
            extra_fields={
                "lessor_co2e_kg": _quantize(lessor_co2e),
                "validated": validated,
                "processing_time_ms": round(elapsed_ms, 2),
            },
        )

    # =========================================================================
    # 4. SPEND-BASED CALCULATION
    # =========================================================================

    def calculate_building_spend(
        self,
        naics_code: str,
        amount: Union[Decimal, float, int, str],
        currency: str = "USD",
        reporting_year: int = 2024,
        building_type: str = "unknown",
        floor_area_sqm: Union[Decimal, float, int, str] = 0,
    ) -> Dict[str, Any]:
        """
        Calculate building emissions using spend-based EEIO method.

        This is the lowest-quality fallback method. Converts the lease spend
        amount to USD, deflates to the EEIO base year (2021), and multiplies
        by the NAICS-specific EEIO factor.

        Formula:
            usd_amount     = amount x currency_rate
            deflated       = usd_amount / cpi(reporting_year) x cpi(2021)
            total_co2e     = deflated x eeio_factor

        Args:
            naics_code: NAICS industry code for EEIO factor lookup.
            amount: Lease spend amount in the given currency.
            currency: ISO 4217 currency code. Defaults to "USD".
            reporting_year: Year of the spend data (for CPI deflation).
                Defaults to 2024.
            building_type: Building type for reporting. Defaults to "unknown".
            floor_area_sqm: Floor area for reporting. Defaults to 0.

        Returns:
            Dict with co2e_kg and all standard fields.

        Raises:
            ValueError: If NAICS code not found or input validation fails.

        Example:
            >>> engine = BuildingCalculatorEngine()
            >>> result = engine.calculate_building_spend(
            ...     naics_code="531120",
            ...     amount=Decimal("120000"),
            ...     currency="USD",
            ...     reporting_year=2024,
            ... )
            >>> result["method"]
            'spend_based'
        """
        start_time = time.monotonic()
        self._increment_calc()

        # Input conversion and validation
        spend_amount = _validate_positive(
            _to_decimal(amount, "amount"),
            "amount",
            allow_zero=False,
        )
        floor_area = _validate_positive(
            _to_decimal(floor_area_sqm, "floor_area_sqm"),
            "floor_area_sqm",
        )

        btype = building_type.lower().strip().replace(" ", "_")
        curr = currency.upper().strip()

        # Convert to USD
        currency_rate = self._db.get_currency_rate(curr)
        usd_amount = _quantize(spend_amount * currency_rate)

        # Deflate to EEIO base year
        cpi_reporting = self._db.get_cpi_deflator(reporting_year)
        cpi_base = self._db.get_cpi_deflator(_EEIO_BASE_YEAR)
        deflated_amount = _quantize(usd_amount / cpi_reporting * cpi_base)

        # Look up EEIO factor
        eeio_data = self._db.get_eeio_factor(naics_code)
        eeio_ef = eeio_data["co2e_per_usd"]

        # Calculate emissions
        total_co2e = _quantize(deflated_amount * eeio_ef)

        ef_source = f"{eeio_data['source']} (NAICS {naics_code})"

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Spend-based calculation: naics=%s, amount=%s %s, "
            "usd=%s, deflated=%s, co2e=%s kg, elapsed=%.2f ms",
            naics_code, spend_amount, curr, usd_amount,
            deflated_amount, total_co2e, elapsed_ms,
        )

        return self._build_result(
            method=_METHOD_SPEND_BASED,
            building_type=btype,
            floor_area_sqm=floor_area,
            elec_co2e_kg=total_co2e,
            gas_co2e_kg=_ZERO,
            heating_co2e_kg=_ZERO,
            cooling_co2e_kg=_ZERO,
            refrigerant_co2e_kg=_ZERO,
            wtt_co2e_kg=_ZERO,
            allocation_factor=_ONE,
            lease_months=12,
            ef_source=ef_source,
            country_code="GLOBAL",
            extra_fields={
                "naics_code": naics_code,
                "original_amount": _quantize(spend_amount),
                "original_currency": curr,
                "usd_amount": usd_amount,
                "deflated_amount": deflated_amount,
                "eeio_factor": _quantize(eeio_ef),
                "reporting_year": reporting_year,
                "eeio_base_year": _EEIO_BASE_YEAR,
                "currency_rate": _quantize(currency_rate),
                "cpi_reporting": cpi_reporting,
                "cpi_base": cpi_base,
                "eeio_description": eeio_data["description"],
                "processing_time_ms": round(elapsed_ms, 2),
            },
        )

    # =========================================================================
    # 5. MULTI-TENANT ALLOCATION
    # =========================================================================

    def calculate_multi_tenant_allocation(
        self,
        total_co2e: Union[Decimal, float, int, str],
        allocation_method: str,
        leased_area: Optional[Union[Decimal, float, int, str]] = None,
        total_area: Optional[Union[Decimal, float, int, str]] = None,
        headcount_lessee: Optional[Union[Decimal, float, int, str]] = None,
        headcount_total: Optional[Union[Decimal, float, int, str]] = None,
        revenue_lessee: Optional[Union[Decimal, float, int, str]] = None,
        revenue_total: Optional[Union[Decimal, float, int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate multi-tenant allocation of total building emissions.

        Supports four allocation methods per GHG Protocol Scope 3 guidance:
        - floor_area: leased_area / total_area
        - headcount: headcount_lessee / headcount_total
        - revenue: revenue_lessee / revenue_total
        - equal_share: 1 / total_tenants (derived from headcount or defaults)

        Args:
            total_co2e: Total building CO2e before allocation (kgCO2e).
            allocation_method: Method key (floor_area, headcount, revenue,
                equal_share).
            leased_area: Leased floor area. Required for floor_area method.
            total_area: Total building floor area. Required for floor_area method.
            headcount_lessee: Lessee headcount. Required for headcount method.
            headcount_total: Total building headcount. Required for headcount method.
            revenue_lessee: Lessee revenue. Required for revenue method.
            revenue_total: Total building revenue. Required for revenue method.

        Returns:
            Dict with allocated_co2e_kg, allocation_factor, allocation_method,
            total_co2e_kg, and provenance_hash.

        Raises:
            ValueError: If required parameters for the chosen method are missing
                or the method is unrecognized.

        Example:
            >>> engine = BuildingCalculatorEngine()
            >>> result = engine.calculate_multi_tenant_allocation(
            ...     total_co2e=Decimal("500000"),
            ...     allocation_method="floor_area",
            ...     leased_area=Decimal("2000"),
            ...     total_area=Decimal("10000"),
            ... )
            >>> result["allocation_factor"]
            Decimal('0.20000000')
            >>> result["allocated_co2e_kg"]
            Decimal('100000.00000000')
        """
        start_time = time.monotonic()
        self._increment_calc()

        total_emissions = _validate_positive(
            _to_decimal(total_co2e, "total_co2e"),
            "total_co2e",
        )

        method = allocation_method.lower().strip().replace(" ", "_")

        # Calculate allocation factor based on method
        if method == "floor_area":
            if leased_area is None or total_area is None:
                raise ValueError(
                    "floor_area allocation requires both leased_area and total_area"
                )
            la = _validate_positive(
                _to_decimal(leased_area, "leased_area"),
                "leased_area",
                allow_zero=False,
            )
            ta = _validate_positive(
                _to_decimal(total_area, "total_area"),
                "total_area",
                allow_zero=False,
            )
            if la > ta:
                raise ValueError(
                    f"leased_area ({la}) cannot exceed total_area ({ta})"
                )
            alloc_factor = _quantize(la / ta)

        elif method == "headcount":
            if headcount_lessee is None or headcount_total is None:
                raise ValueError(
                    "headcount allocation requires both headcount_lessee "
                    "and headcount_total"
                )
            hl = _validate_positive(
                _to_decimal(headcount_lessee, "headcount_lessee"),
                "headcount_lessee",
                allow_zero=False,
            )
            ht = _validate_positive(
                _to_decimal(headcount_total, "headcount_total"),
                "headcount_total",
                allow_zero=False,
            )
            if hl > ht:
                raise ValueError(
                    f"headcount_lessee ({hl}) cannot exceed headcount_total ({ht})"
                )
            alloc_factor = _quantize(hl / ht)

        elif method == "revenue":
            if revenue_lessee is None or revenue_total is None:
                raise ValueError(
                    "revenue allocation requires both revenue_lessee "
                    "and revenue_total"
                )
            rl = _validate_positive(
                _to_decimal(revenue_lessee, "revenue_lessee"),
                "revenue_lessee",
                allow_zero=False,
            )
            rt = _validate_positive(
                _to_decimal(revenue_total, "revenue_total"),
                "revenue_total",
                allow_zero=False,
            )
            if rl > rt:
                raise ValueError(
                    f"revenue_lessee ({rl}) cannot exceed revenue_total ({rt})"
                )
            alloc_factor = _quantize(rl / rt)

        elif method == "equal_share":
            # For equal share, derive number of tenants from headcount_total
            # or default to 2 tenants
            if headcount_total is not None:
                ht = _validate_positive(
                    _to_decimal(headcount_total, "headcount_total"),
                    "headcount_total",
                    allow_zero=False,
                )
                # Assume equal share means 1/N where N = headcount_total
                # Typically N = number of tenants; if headcount_total given,
                # interpret as tenant count
                alloc_factor = _quantize(_ONE / ht)
            else:
                # Default: assume 2 tenants = 50% share
                alloc_factor = _quantize(Decimal("0.50"))
                logger.info(
                    "equal_share allocation with no headcount_total; "
                    "defaulting to 50%% share (2 tenants)"
                )
        else:
            raise ValueError(
                f"Unrecognized allocation method: '{method}'. "
                f"Supported methods: floor_area, headcount, revenue, equal_share"
            )

        # Calculate allocated emissions
        allocated_co2e = _quantize(total_emissions * alloc_factor)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Build result
        result: Dict[str, Any] = {
            "allocated_co2e_kg": allocated_co2e,
            "allocation_factor": alloc_factor,
            "allocation_method": method,
            "total_co2e_kg": _quantize(total_emissions),
            "engine_id": self.ENGINE_ID,
            "engine_version": self.ENGINE_VERSION,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(elapsed_ms, 2),
        }

        # Add method-specific detail fields
        if method == "floor_area" and leased_area is not None and total_area is not None:
            result["leased_area_sqm"] = _quantize(
                _to_decimal(leased_area, "leased_area")
            )
            result["total_area_sqm"] = _quantize(
                _to_decimal(total_area, "total_area")
            )
        elif method == "headcount" and headcount_lessee is not None:
            result["headcount_lessee"] = _quantize(
                _to_decimal(headcount_lessee, "headcount_lessee")
            )
            result["headcount_total"] = _quantize(
                _to_decimal(headcount_total, "headcount_total")
            )
        elif method == "revenue" and revenue_lessee is not None:
            result["revenue_lessee"] = _quantize(
                _to_decimal(revenue_lessee, "revenue_lessee")
            )
            result["revenue_total"] = _quantize(
                _to_decimal(revenue_total, "revenue_total")
            )

        result["provenance_hash"] = _calculate_provenance_hash(result)

        logger.info(
            "Multi-tenant allocation: method=%s, factor=%s, "
            "total=%s kg, allocated=%s kg, elapsed=%.2f ms",
            method, alloc_factor, total_emissions, allocated_co2e, elapsed_ms,
        )

        return result

    # =========================================================================
    # 6. BATCH CALCULATION
    # =========================================================================

    def calculate_batch(
        self,
        buildings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a batch of buildings.

        Processes each building individually and aggregates results.
        Each building dict must have a "method" key indicating which
        calculation method to use, plus the required parameters for that
        method.

        Supported method values:
        - "asset_specific": requires building_type, floor_area_sqm,
          electricity_kwh, plus optional gas/heating/cooling/refrigerant params
        - "average_data": requires building_type, floor_area_sqm, plus
          optional climate_zone, country_code
        - "lessor_specific" / "lessor": requires lessor_co2e_kg
        - "spend_based" / "spend": requires naics_code, amount

        Args:
            buildings: List of building parameter dictionaries.

        Returns:
            Dict with:
            - total_co2e_kg: Sum of all building emissions
            - building_count: Number of successfully processed buildings
            - results: List of individual building results
            - errors: List of error dicts for failed buildings
            - processing_time_ms: Total processing time

        Raises:
            ValueError: If buildings list exceeds MAX_BATCH_SIZE.

        Example:
            >>> engine = BuildingCalculatorEngine()
            >>> result = engine.calculate_batch([
            ...     {
            ...         "method": "asset_specific",
            ...         "building_type": "office",
            ...         "floor_area_sqm": 5000,
            ...         "electricity_kwh": 750000,
            ...         "country_code": "US",
            ...     },
            ...     {
            ...         "method": "average_data",
            ...         "building_type": "retail",
            ...         "floor_area_sqm": 3000,
            ...     },
            ... ])
            >>> result["building_count"]
            2
        """
        start_time = time.monotonic()

        if len(buildings) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(buildings)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total_co2e = _ZERO

        for idx, building in enumerate(buildings):
            try:
                method = building.get("method", "").lower().strip()
                result = self._dispatch_single_building(method, building)
                results.append(result)
                total_co2e += result["co2e_kg"]
            except Exception as exc:
                error_entry = {
                    "index": idx,
                    "building": building,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                errors.append(error_entry)
                logger.warning(
                    "Batch building %d failed: %s: %s",
                    idx, type(exc).__name__, exc,
                )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        batch_result: Dict[str, Any] = {
            "total_co2e_kg": _quantize(total_co2e),
            "building_count": len(results),
            "error_count": len(errors),
            "total_buildings_submitted": len(buildings),
            "results": results,
            "errors": errors,
            "engine_id": self.ENGINE_ID,
            "engine_version": self.ENGINE_VERSION,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(elapsed_ms, 2),
        }

        batch_result["provenance_hash"] = _calculate_provenance_hash({
            "total_co2e_kg": str(total_co2e),
            "building_count": len(results),
            "error_count": len(errors),
        })

        logger.info(
            "Batch calculation complete: %d buildings, %d errors, "
            "total_co2e=%s kg, elapsed=%.2f ms",
            len(results), len(errors), total_co2e, elapsed_ms,
        )

        return batch_result

    def _dispatch_single_building(
        self,
        method: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Dispatch a single building calculation based on method.

        Args:
            method: Calculation method string.
            params: Building parameters dictionary.

        Returns:
            Calculation result dictionary.

        Raises:
            ValueError: If method is unrecognized or required params missing.
        """
        if method in ("asset_specific", "asset-specific"):
            return self.calculate_building_asset_specific(
                building_type=params.get("building_type", "unknown"),
                floor_area_sqm=params.get("floor_area_sqm", 0),
                electricity_kwh=params.get("electricity_kwh", 0),
                gas_kwh=params.get("gas_kwh", 0),
                heating_kwh=params.get("heating_kwh", 0),
                cooling_kwh=params.get("cooling_kwh", 0),
                country_code=params.get("country_code", "US"),
                allocation_factor=params.get("allocation_factor", 1.0),
                lease_months=params.get("lease_months", 12),
                include_wtt=params.get("include_wtt", True),
                include_refrigerants=params.get("include_refrigerants", False),
                refrigerant=params.get("refrigerant"),
                leakage_kg=params.get("leakage_kg", 0),
            )
        elif method in ("average_data", "average-data", "average"):
            return self.calculate_building_average_data(
                building_type=params.get("building_type", "office"),
                floor_area_sqm=params.get("floor_area_sqm", 0),
                climate_zone=params.get("climate_zone", "temperate"),
                country_code=params.get("country_code", "US"),
                allocation_factor=params.get("allocation_factor", 1.0),
                lease_months=params.get("lease_months", 12),
            )
        elif method in ("lessor_specific", "lessor-specific", "lessor"):
            return self.calculate_building_lessor(
                lessor_co2e_kg=params.get("lessor_co2e_kg", 0),
                allocation_factor=params.get("allocation_factor", 1.0),
                lease_months=params.get("lease_months", 12),
                validated=params.get("validated", False),
                building_type=params.get("building_type", "unknown"),
                floor_area_sqm=params.get("floor_area_sqm", 0),
            )
        elif method in ("spend_based", "spend-based", "spend"):
            return self.calculate_building_spend(
                naics_code=params.get("naics_code", "531120"),
                amount=params.get("amount", 0),
                currency=params.get("currency", "USD"),
                reporting_year=params.get("reporting_year", 2024),
                building_type=params.get("building_type", "unknown"),
                floor_area_sqm=params.get("floor_area_sqm", 0),
            )
        else:
            raise ValueError(
                f"Unrecognized calculation method: '{method}'. "
                f"Supported: asset_specific, average_data, lessor_specific, spend_based"
            )

    # =========================================================================
    # 7. BUILDING COMPARISON
    # =========================================================================

    def compare_buildings(
        self,
        buildings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare multiple buildings by CO2e intensity (kgCO2e per sqm).

        Calculates emissions for each building, then ranks them by intensity
        (CO2e per square meter of floor area), identifying the most and
        least efficient buildings.

        Args:
            buildings: List of building parameter dictionaries (same format
                as calculate_batch). Each must include floor_area_sqm > 0.

        Returns:
            Dict with:
            - rankings: List of buildings sorted by intensity (ascending)
            - most_efficient: Building with lowest CO2e/sqm
            - least_efficient: Building with highest CO2e/sqm
            - average_intensity: Average CO2e/sqm across all buildings
            - total_co2e_kg: Sum of all building emissions
            - building_count: Number of successfully compared buildings

        Example:
            >>> engine = BuildingCalculatorEngine()
            >>> result = engine.compare_buildings([
            ...     {
            ...         "method": "average_data",
            ...         "building_type": "office",
            ...         "floor_area_sqm": 5000,
            ...         "country_code": "US",
            ...     },
            ...     {
            ...         "method": "average_data",
            ...         "building_type": "warehouse",
            ...         "floor_area_sqm": 10000,
            ...         "country_code": "US",
            ...     },
            ... ])
            >>> result["most_efficient"]["building_type"]
            'warehouse'
        """
        start_time = time.monotonic()

        if not buildings:
            raise ValueError("At least one building is required for comparison")

        if len(buildings) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Comparison size {len(buildings)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        ranked_buildings: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for idx, building in enumerate(buildings):
            try:
                method = building.get("method", "").lower().strip()
                result = self._dispatch_single_building(method, building)

                floor_area = result.get("floor_area_sqm", _ZERO)
                if floor_area <= _ZERO:
                    raise ValueError(
                        f"Building at index {idx} has zero or negative "
                        f"floor_area_sqm; cannot compute intensity"
                    )

                intensity = _quantize(result["co2e_kg"] / floor_area)

                ranked_buildings.append({
                    "index": idx,
                    "building_type": result["building_type"],
                    "floor_area_sqm": result["floor_area_sqm"],
                    "co2e_kg": result["co2e_kg"],
                    "intensity_kg_per_sqm": intensity,
                    "method": result["method"],
                    "country_code": result.get("country_code", "GLOBAL"),
                    "dqi_score": result.get("dqi_score", _ZERO),
                })
            except Exception as exc:
                errors.append({
                    "index": idx,
                    "building": building,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                })
                logger.warning(
                    "Comparison building %d failed: %s: %s",
                    idx, type(exc).__name__, exc,
                )

        # Sort by intensity (ascending = most efficient first)
        ranked_buildings.sort(key=lambda b: b["intensity_kg_per_sqm"])

        # Assign rank
        for rank, bldg in enumerate(ranked_buildings, start=1):
            bldg["rank"] = rank

        # Calculate summary statistics
        total_co2e = sum(
            b["co2e_kg"] for b in ranked_buildings
        )
        total_area = sum(
            b["floor_area_sqm"] for b in ranked_buildings
        )

        if total_area > _ZERO:
            average_intensity = _quantize(total_co2e / total_area)
        else:
            average_intensity = _ZERO

        elapsed_ms = (time.monotonic() - start_time) * 1000

        comparison_result: Dict[str, Any] = {
            "rankings": ranked_buildings,
            "building_count": len(ranked_buildings),
            "error_count": len(errors),
            "errors": errors,
            "total_co2e_kg": _quantize(total_co2e),
            "total_area_sqm": _quantize(total_area),
            "average_intensity_kg_per_sqm": average_intensity,
            "engine_id": self.ENGINE_ID,
            "engine_version": self.ENGINE_VERSION,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(elapsed_ms, 2),
        }

        if ranked_buildings:
            comparison_result["most_efficient"] = ranked_buildings[0]
            comparison_result["least_efficient"] = ranked_buildings[-1]
        else:
            comparison_result["most_efficient"] = None
            comparison_result["least_efficient"] = None

        comparison_result["provenance_hash"] = _calculate_provenance_hash({
            "total_co2e_kg": str(total_co2e),
            "building_count": len(ranked_buildings),
            "average_intensity": str(average_intensity),
        })

        logger.info(
            "Building comparison: %d buildings ranked, "
            "avg intensity=%s kg/sqm, elapsed=%.2f ms",
            len(ranked_buildings), average_intensity, elapsed_ms,
        )

        return comparison_result

    # =========================================================================
    # 8. QUICK ANNUAL ESTIMATE
    # =========================================================================

    def estimate_annual_emissions(
        self,
        building_type: str,
        floor_area_sqm: Union[Decimal, float, int, str],
        country_code: str = "US",
    ) -> Dict[str, Any]:
        """
        Quick convenience estimate of annual building emissions.

        Uses average-data method with auto-detected climate zone for the
        given country, full allocation (1.0), and full year (12 months).
        Intended for quick portfolio screening, not formal reporting.

        Args:
            building_type: Building type.
            floor_area_sqm: Floor area in square meters.
            country_code: ISO country code. Defaults to "US".

        Returns:
            Dict with co2e_kg and all standard result fields.

        Example:
            >>> engine = BuildingCalculatorEngine()
            >>> result = engine.estimate_annual_emissions(
            ...     building_type="office",
            ...     floor_area_sqm=Decimal("5000"),
            ...     country_code="US",
            ... )
            >>> result["co2e_kg"] > Decimal("0")
            True
        """
        # Auto-detect climate zone from country
        climate_zone = self._db.get_climate_zone(country_code)

        return self.calculate_building_average_data(
            building_type=building_type,
            floor_area_sqm=floor_area_sqm,
            climate_zone=climate_zone,
            country_code=country_code,
            allocation_factor=1.0,
            lease_months=12,
        )

    # =========================================================================
    # ENGINE SUMMARY
    # =========================================================================

    def get_engine_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the engine's capabilities and state.

        Returns:
            Dict with engine metadata, supported methods, asset types,
            and calculation count.
        """
        return {
            "engine_id": self.ENGINE_ID,
            "engine_version": self.ENGINE_VERSION,
            "supported_methods": [
                _METHOD_ASSET_SPECIFIC,
                _METHOD_AVERAGE_DATA,
                _METHOD_LESSOR_SPECIFIC,
                _METHOD_SPEND_BASED,
            ],
            "supported_building_types": self._db.get_all_building_types(),
            "supported_allocation_methods": sorted(
                ["floor_area", "headcount", "revenue", "equal_share"]
            ),
            "total_calculations": self.get_calc_count(),
            "db_engine_id": self._db.ENGINE_ID,
            "db_engine_version": self._db.ENGINE_VERSION,
            "max_batch_size": _MAX_BATCH_SIZE,
        }

    # =========================================================================
    # RESET (TESTING ONLY)
    # =========================================================================

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        Warning: This method is intended for use in test fixtures only.
        Do not call in production code.
        """
        with cls._lock:
            cls._instance = None


# =============================================================================
# MODULE-LEVEL ACCESSORS
# =============================================================================

_calculator_instance: Optional[BuildingCalculatorEngine] = None
_calculator_lock: threading.Lock = threading.Lock()


def get_building_calculator() -> BuildingCalculatorEngine:
    """
    Get the singleton BuildingCalculatorEngine instance.

    Thread-safe accessor for the global building calculator instance.

    Returns:
        BuildingCalculatorEngine singleton instance.

    Example:
        >>> engine = get_building_calculator()
        >>> result = engine.calculate_building_average_data(
        ...     building_type="office",
        ...     floor_area_sqm=5000,
        ... )
    """
    global _calculator_instance
    with _calculator_lock:
        if _calculator_instance is None:
            _calculator_instance = BuildingCalculatorEngine()
        return _calculator_instance


def reset_building_calculator() -> None:
    """
    Reset the module-level calculator instance (for testing only).

    Warning: This function is intended for use in test fixtures only.
    Do not call in production code.
    """
    global _calculator_instance
    with _calculator_lock:
        _calculator_instance = None
    BuildingCalculatorEngine.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "BuildingCalculatorEngine",
    "get_building_calculator",
    "reset_building_calculator",
    "ENGINE_ID",
    "ENGINE_VERSION",
]
