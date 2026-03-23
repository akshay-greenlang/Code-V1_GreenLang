# -*- coding: utf-8 -*-
"""
ITAssetsCalculatorEngine - Engine 5: Upstream Leased Assets Agent (AGENT-MRV-021)

Core calculation engine for leased IT asset emissions covering servers, network
switches, storage arrays, desktops, laptops, printers, copiers, and data center
allocations. Supports individual asset, portfolio, batch, and annual estimation
methods.

This engine implements deterministic Decimal-based emissions calculations
for all leased IT asset categories, following EPA eGRID 2024 / IEA 2024
emission factors and the GHG Protocol Scope 3 Category 8 methodology.

Primary Formulae:
    Server:
        co2e = power_kw x PUE x utilization x operating_hours x count x grid_ef

    Network / Storage:
        co2e = power_kw x operating_hours x count x grid_ef

    Desktop / Laptop:
        co2e = power_kw x operating_hours x count x grid_ef

    Printer / Copier (active + standby):
        co2e = (power_kw x active_hours + standby_power_kw x standby_hours) x count x grid_ef

    Data Center Allocation:
        co2e = allocated_kw x PUE x 8760 x grid_ef

    IT Portfolio:
        Iterate per-asset, sum to portfolio totals

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate value is deterministic and reproducible
    - SHA-256 provenance hash on every result
    - Emission factors sourced from EPA eGRID 2024, IEA 2024

Supports:
    - 7 IT asset types (server, network_switch, storage, desktop,
      laptop, printer, copier)
    - Data center power allocation method
    - PUE (Power Usage Effectiveness) adjustment for servers and data centers
    - Server utilization factor
    - Active + standby power modes for printers and copiers
    - Country-specific and eGRID subregion grid emission factors
    - IT portfolio aggregation with per-asset breakdown
    - Batch processing for multiple IT asset calculations
    - Quick annual emission estimation
    - Input validation with detailed error messages
    - SHA-256 provenance hash integration for audit trails

Example:
    >>> from greenlang.agents.mrv.upstream_leased_assets.it_assets_calculator import (
    ...     get_it_assets_calculator,
    ... )
    >>> from decimal import Decimal
    >>> engine = get_it_assets_calculator()
    >>> result = engine.calculate_server(
    ...     power_kw=Decimal("0.500"),
    ...     pue=Decimal("1.58"),
    ...     utilization=Decimal("0.30"),
    ...     count=10,
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

logger = logging.getLogger(__name__)

# ==============================================================================
# ENGINE METADATA
# ==============================================================================

ENGINE_ID: str = "it_assets_calculator_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-008"
AGENT_COMPONENT: str = "AGENT-MRV-021"
VERSION: str = "1.0.0"

# ==============================================================================
# DECIMAL PRECISION & CONSTANTS
# ==============================================================================

_QUANT_8DP = Decimal("0.00000001")
_PRECISION = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
ROUNDING = ROUND_HALF_UP

# Batch processing limits
_MAX_BATCH_SIZE = 10000

# Hours in a year
_HOURS_PER_YEAR = Decimal("8760")

# ==============================================================================
# IT ASSET POWER RATINGS (kW) - Industry standard defaults
# Source: Energy Star, SPECpower, EPA ENERGY STAR
# ==============================================================================

IT_POWER_RATINGS: Dict[str, Decimal] = {
    "server": Decimal("0.500"),
    "network_switch": Decimal("0.150"),
    "storage": Decimal("0.200"),
    "desktop": Decimal("0.150"),
    "laptop": Decimal("0.045"),
    "printer": Decimal("0.200"),
    "copier": Decimal("0.300"),
}

# ==============================================================================
# DEFAULT OPERATING HOURS by IT asset type
# Source: Industry benchmarks, EPA ENERGY STAR
# ==============================================================================

IT_OPERATING_HOURS: Dict[str, Decimal] = {
    "server": Decimal("8760"),
    "network_switch": Decimal("8760"),
    "storage": Decimal("8760"),
    "desktop": Decimal("2500"),
    "laptop": Decimal("2000"),
    "printer": Decimal("1000"),
    "copier": Decimal("800"),
}

# ==============================================================================
# STANDBY PARAMETERS for printers and copiers
# Source: EPA ENERGY STAR typical energy usage
# ==============================================================================

IT_STANDBY_PARAMS: Dict[str, Dict[str, Decimal]] = {
    "printer": {
        "standby_power_kw": Decimal("0.020"),
        "standby_hours": Decimal("3760"),
    },
    "copier": {
        "standby_power_kw": Decimal("0.030"),
        "standby_hours": Decimal("3960"),
    },
}

# ==============================================================================
# PUE (Power Usage Effectiveness) DEFAULTS
# Source: Uptime Institute 2024 Global Data Center Survey
# ==============================================================================

PUE_DEFAULTS: Dict[str, Decimal] = {
    "data_center": Decimal("1.58"),
    "colocation": Decimal("1.40"),
    "edge": Decimal("1.80"),
}

# ==============================================================================
# DEFAULT SERVER UTILIZATION
# Source: NRDC Data Center Efficiency Assessment 2024
# ==============================================================================

DEFAULT_SERVER_UTILIZATION: Decimal = Decimal("0.30")

# ==============================================================================
# VALID IT ASSET TYPES
# ==============================================================================

VALID_IT_TYPES: List[str] = [
    "server",
    "network_switch",
    "storage",
    "desktop",
    "laptop",
    "printer",
    "copier",
]

# ==============================================================================
# GRID EMISSION FACTORS (kgCO2e per kWh) by country
# Source: IEA 2024, EPA eGRID 2024
# ==============================================================================

GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.37120"),
    "GB": Decimal("0.20700"),
    "DE": Decimal("0.33800"),
    "FR": Decimal("0.05200"),
    "JP": Decimal("0.47100"),
    "CN": Decimal("0.55500"),
    "IN": Decimal("0.71600"),
    "AU": Decimal("0.65600"),
    "CA": Decimal("0.12000"),
    "BR": Decimal("0.07500"),
    "KR": Decimal("0.42200"),
    "IT": Decimal("0.25800"),
    "ES": Decimal("0.18100"),
    "NL": Decimal("0.33000"),
    "SE": Decimal("0.01200"),
    "NO": Decimal("0.00800"),
    "DK": Decimal("0.14000"),
    "FI": Decimal("0.07300"),
    "PL": Decimal("0.63500"),
    "AT": Decimal("0.09400"),
    "BE": Decimal("0.16200"),
    "CH": Decimal("0.01100"),
    "IE": Decimal("0.29600"),
    "PT": Decimal("0.18400"),
    "NZ": Decimal("0.08700"),
    "SG": Decimal("0.40800"),
    "ZA": Decimal("0.92800"),
    "MX": Decimal("0.42300"),
    "GLOBAL": Decimal("0.43200"),
}

# ==============================================================================
# eGRID SUBREGION EMISSION FACTORS (kgCO2e per kWh) - EPA eGRID 2024
# ==============================================================================

EGRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "AKGD": Decimal("0.41100"),
    "AKMS": Decimal("0.20300"),
    "AZNM": Decimal("0.36300"),
    "CAMX": Decimal("0.21500"),
    "ERCT": Decimal("0.36200"),
    "FRCC": Decimal("0.36200"),
    "HIMS": Decimal("0.55400"),
    "HIOA": Decimal("0.62100"),
    "MROE": Decimal("0.51700"),
    "MROW": Decimal("0.40500"),
    "NEWE": Decimal("0.18400"),
    "NWPP": Decimal("0.25100"),
    "NYCW": Decimal("0.22100"),
    "NYLI": Decimal("0.38300"),
    "NYUP": Decimal("0.09200"),
    "PRMS": Decimal("0.68200"),
    "RFCE": Decimal("0.30300"),
    "RFCM": Decimal("0.48600"),
    "RFCW": Decimal("0.44500"),
    "RMPA": Decimal("0.47700"),
    "SPNO": Decimal("0.42300"),
    "SPSO": Decimal("0.39100"),
    "SRMV": Decimal("0.33800"),
    "SRMW": Decimal("0.62100"),
    "SRSO": Decimal("0.38000"),
    "SRTV": Decimal("0.37900"),
    "SRVC": Decimal("0.27500"),
}

# ==============================================================================
# DQI SCORE BY METHOD
# GHG Protocol data quality indicators (1=best, 5=worst)
# ==============================================================================

DQI_SCORES: Dict[str, Decimal] = {
    "server": Decimal("2.0"),
    "network": Decimal("2.0"),
    "storage": Decimal("2.0"),
    "desktop": Decimal("2.5"),
    "laptop": Decimal("2.5"),
    "printer": Decimal("2.5"),
    "copier": Decimal("2.5"),
    "data_center_allocation": Decimal("1.5"),
    "it_portfolio": Decimal("2.0"),
    "estimate": Decimal("4.0"),
}

# ==============================================================================
# SINGLETON INSTANCE MANAGEMENT
# ==============================================================================

_instance: Optional["ITAssetsCalculatorEngine"] = None
_instance_lock: threading.Lock = threading.Lock()


# ==============================================================================
# HELPER: Quantize a Decimal to 8 decimal places
# ==============================================================================

def _q(value: Decimal) -> Decimal:
    """
    Quantize a Decimal value to 8 decimal places with ROUND_HALF_UP.

    Args:
        value: The Decimal value to quantize.

    Returns:
        Quantized Decimal with exactly 8 decimal places.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


def _safe_decimal(value: Any) -> Decimal:
    """
    Safely convert a value to Decimal.

    Args:
        value: Value to convert (str, int, float, or Decimal).

    Returns:
        Decimal representation of the value.

    Raises:
        ValueError: If value cannot be converted to Decimal.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot convert '{value}' (type={type(value).__name__}) to Decimal"
        ) from exc


def _calculate_provenance_hash(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> str:
    """
    Calculate SHA-256 provenance hash for audit trail.

    Args:
        inputs: Input parameters as a dict.
        outputs: Output values as a dict.

    Returns:
        SHA-256 hex digest string.
    """
    payload = json.dumps(
        {"inputs": inputs, "outputs": outputs, "engine": ENGINE_ID, "version": ENGINE_VERSION},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ==============================================================================
# MODULE-LEVEL SINGLETON ACCESSORS
# ==============================================================================

def get_it_assets_calculator() -> "ITAssetsCalculatorEngine":
    """
    Get the singleton ITAssetsCalculatorEngine instance.

    Returns:
        ITAssetsCalculatorEngine singleton.
    """
    return ITAssetsCalculatorEngine.get_instance()


def reset_it_assets_calculator() -> None:
    """
    Reset the singleton ITAssetsCalculatorEngine instance (testing only).
    """
    ITAssetsCalculatorEngine.reset_instance()


# ==============================================================================
# ITAssetsCalculatorEngine
# ==============================================================================


class ITAssetsCalculatorEngine:
    """
    Engine 5: IT assets emissions calculator for upstream leased assets.

    Implements deterministic emissions calculations for leased IT equipment
    including servers (with PUE and utilization), network switches, storage
    arrays, desktops, laptops, printers, copiers (with standby power), and
    data center power allocations using EPA eGRID 2024 / IEA 2024 emission
    factors aligned with GHG Protocol Scope 3 Category 8 methodology.

    The engine follows GreenLang's zero-hallucination principle by using only
    deterministic Decimal arithmetic with EPA/IEA-sourced parameters. No LLM
    calls are made anywhere in the calculation pipeline.

    Thread Safety:
        This engine is fully thread-safe. A reentrant lock protects shared
        state during calculations. The singleton instance is created lazily
        with double-checked locking.

    Attributes:
        _lock: Reentrant lock for thread safety.
        _calculation_count: Running count of calculations performed.
        _initialized_at: Timestamp when the engine was initialized.

    Example:
        >>> engine = get_it_assets_calculator()
        >>> result = engine.calculate_server(
        ...     power_kw=Decimal("0.500"),
        ...     pue=Decimal("1.58"),
        ...     utilization=Decimal("0.30"),
        ...     count=10,
        ... )
        >>> assert result["co2e_kg"] > Decimal("0")
    """

    # ------------------------------------------------------------------
    # Singleton Access
    # ------------------------------------------------------------------

    @staticmethod
    def get_instance() -> "ITAssetsCalculatorEngine":
        """
        Get or create the singleton ITAssetsCalculatorEngine instance.

        Thread-safe lazy initialization using double-checked locking.

        Returns:
            Singleton ITAssetsCalculatorEngine instance.
        """
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:
                    _instance = ITAssetsCalculatorEngine()
        return _instance

    @staticmethod
    def reset_instance() -> None:
        """
        Reset the singleton instance (for testing only).

        This method is intended exclusively for unit tests that need
        a fresh engine instance. It should never be called in production.
        """
        global _instance
        with _instance_lock:
            _instance = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialize the ITAssetsCalculatorEngine."""
        self._lock: threading.RLock = threading.RLock()
        self._calculation_count: int = 0
        self._initialized_at: str = datetime.now(timezone.utc).isoformat()

        logger.info(
            "ITAssetsCalculatorEngine initialized: "
            "engine=%s, version=%s, agent=%s",
            ENGINE_ID,
            ENGINE_VERSION,
            AGENT_ID,
        )

    # ==================================================================
    # PROPERTY: calculation_count
    # ==================================================================

    @property
    def calculation_count(self) -> int:
        """Return the total number of calculations performed by this engine."""
        return self._calculation_count

    @property
    def engine_id(self) -> str:
        """Return the engine identifier."""
        return ENGINE_ID

    @property
    def engine_version(self) -> str:
        """Return the engine version."""
        return ENGINE_VERSION

    # ==================================================================
    # VALIDATION HELPERS
    # ==================================================================

    def _validate_positive_decimal(
        self, value: Any, field_name: str
    ) -> Decimal:
        """
        Validate that a value is a positive Decimal.

        Args:
            value: Value to validate and convert.
            field_name: Name of the field for error messages.

        Returns:
            Validated Decimal value.

        Raises:
            ValueError: If value is not positive.
        """
        dec_val = _safe_decimal(value)
        if dec_val <= _ZERO:
            raise ValueError(
                f"{field_name} must be positive, got {dec_val}"
            )
        return dec_val

    def _validate_non_negative_decimal(
        self, value: Any, field_name: str
    ) -> Decimal:
        """
        Validate that a value is a non-negative Decimal.

        Args:
            value: Value to validate and convert.
            field_name: Name of the field for error messages.

        Returns:
            Validated Decimal value.

        Raises:
            ValueError: If value is negative.
        """
        dec_val = _safe_decimal(value)
        if dec_val < _ZERO:
            raise ValueError(
                f"{field_name} must be non-negative, got {dec_val}"
            )
        return dec_val

    def _validate_count(self, count: int) -> int:
        """
        Validate asset count is a positive integer.

        Args:
            count: Number of assets.

        Returns:
            Validated count.

        Raises:
            ValueError: If count is not a positive integer.
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError(
                f"Asset count must be a positive integer, got {count}"
            )
        return count

    def _validate_country_code(self, country_code: str) -> str:
        """
        Validate and normalize country code.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Normalized country code string (uppercase, stripped).

        Raises:
            ValueError: If country_code is not recognized.
        """
        normalized = country_code.upper().strip()
        if normalized not in GRID_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown country_code '{country_code}'. "
                f"Available: {list(GRID_EMISSION_FACTORS.keys())}"
            )
        return normalized

    def _validate_pue(self, pue: Any) -> Decimal:
        """
        Validate PUE is at least 1.0.

        PUE (Power Usage Effectiveness) is always >= 1.0 since total
        facility power includes IT load plus overhead.

        Args:
            pue: PUE value to validate.

        Returns:
            Validated Decimal PUE.

        Raises:
            ValueError: If PUE < 1.0.
        """
        dec_pue = _safe_decimal(pue)
        if dec_pue < _ONE:
            raise ValueError(
                f"PUE must be >= 1.0, got {dec_pue}"
            )
        return dec_pue

    def _validate_utilization(self, utilization: Any) -> Decimal:
        """
        Validate server utilization is between 0 (exclusive) and 1 (inclusive).

        Args:
            utilization: Utilization factor to validate.

        Returns:
            Validated Decimal utilization.

        Raises:
            ValueError: If utilization out of range.
        """
        dec_util = _safe_decimal(utilization)
        if dec_util <= _ZERO or dec_util > _ONE:
            raise ValueError(
                f"Utilization must be in range (0.0, 1.0], got {dec_util}"
            )
        return dec_util

    def _get_grid_ef(
        self, country_code: str, egrid_subregion: Optional[str] = None
    ) -> Tuple[Decimal, str]:
        """
        Get grid emission factor and source label for a location.

        eGRID subregion takes precedence for US-based calculations.

        Args:
            country_code: ISO country code (uppercase).
            egrid_subregion: Optional EPA eGRID subregion code.

        Returns:
            Tuple of (grid_ef in kgCO2e/kWh, ef_source label).

        Raises:
            ValueError: If egrid_subregion is provided but not recognized.
        """
        if egrid_subregion is not None:
            subregion = egrid_subregion.upper().strip()
            if subregion not in EGRID_EMISSION_FACTORS:
                raise ValueError(
                    f"Unknown eGRID subregion '{egrid_subregion}'. "
                    f"Available: {list(EGRID_EMISSION_FACTORS.keys())}"
                )
            return EGRID_EMISSION_FACTORS[subregion], "EPA_eGRID_2024"
        return GRID_EMISSION_FACTORS[country_code], "IEA_2024"

    # ==================================================================
    # 1. calculate_server
    # ==================================================================

    def calculate_server(
        self,
        power_kw: Union[Decimal, int, float, str] = Decimal("0.500"),
        pue: Union[Decimal, int, float, str] = Decimal("1.58"),
        utilization: Union[Decimal, int, float, str] = Decimal("0.30"),
        operating_hours: Union[Decimal, int, float, str] = Decimal("8760"),
        country_code: str = "US",
        count: int = 1,
        egrid_subregion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate server emissions using energy-based method with PUE.

        Computes annual emissions from leased servers accounting for power
        draw, data center PUE overhead, server utilization, and operating
        hours.

        Formula:
            energy_kwh = power_kw x PUE x utilization x operating_hours x count
            co2e_kg    = energy_kwh x grid_ef

        Args:
            power_kw: Server power draw in kW (default 0.500).
            pue: Power Usage Effectiveness (default 1.58).
            utilization: Server utilization factor 0-1 (default 0.30).
            operating_hours: Annual operating hours (default 8760).
            country_code: ISO country code for grid EF (default "US").
            count: Number of servers (default 1).
            egrid_subregion: Optional EPA eGRID subregion code.

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg,
            energy_kwh, power_kw, pue, utilization, operating_hours,
            grid_ef, method, it_type, count, country_code,
            ef_source, dqi_score, provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If any input parameter is invalid.

        Example:
            >>> result = engine.calculate_server(
            ...     power_kw=Decimal("0.500"),
            ...     pue=Decimal("1.58"),
            ...     utilization=Decimal("0.30"),
            ...     count=10,
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            pwr = self._validate_positive_decimal(power_kw, "power_kw")
            pue_val = self._validate_pue(pue)
            util = self._validate_utilization(utilization)
            hours = self._validate_positive_decimal(operating_hours, "operating_hours")
            c_code = self._validate_country_code(country_code)
            cnt = self._validate_count(count)

            # Step 2: Resolve grid EF (ZERO HALLUCINATION)
            grid_ef, ef_source = self._get_grid_ef(c_code, egrid_subregion)

            # Step 3: Calculate energy and emissions (Decimal only)
            count_dec = _safe_decimal(cnt)
            energy_kwh = _q(pwr * pue_val * util * hours * count_dec)
            co2e_kg = _q(energy_kwh * grid_ef)

            # Grid-based: all upstream, no tailpipe
            ttw_co2e = _ZERO
            wtt_co2e = co2e_kg

            # Step 4: Build provenance hash
            input_data = {
                "it_type": "server",
                "power_kw": str(pwr),
                "pue": str(pue_val),
                "utilization": str(util),
                "operating_hours": str(hours),
                "country_code": c_code,
                "egrid_subregion": egrid_subregion,
                "count": cnt,
            }
            output_data = {
                "co2e_kg": str(co2e_kg),
                "energy_kwh": str(energy_kwh),
            }
            provenance_hash = _calculate_provenance_hash(input_data, output_data)

            # Step 5: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": co2e_kg,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "energy_kwh": energy_kwh,
                "power_kw": pwr,
                "pue": pue_val,
                "utilization": util,
                "operating_hours": hours,
                "grid_ef": grid_ef,
                "method": "server",
                "it_type": "server",
                "count": cnt,
                "country_code": c_code,
                "egrid_subregion": egrid_subregion,
                "ef_source": ef_source,
                "dqi_score": DQI_SCORES["server"],
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "Server calculation complete: power=%s kW, pue=%s, "
                "util=%s, hours=%s, count=%d, kWh=%s, co2e=%s kg, "
                "duration=%.4fs",
                pwr, pue_val, util, hours, cnt,
                energy_kwh, co2e_kg, duration,
            )

            return result

    # ==================================================================
    # 2. calculate_network
    # ==================================================================

    def calculate_network(
        self,
        power_kw: Union[Decimal, int, float, str] = Decimal("0.150"),
        operating_hours: Union[Decimal, int, float, str] = Decimal("8760"),
        country_code: str = "US",
        count: int = 1,
        egrid_subregion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate network switch/router emissions using energy-based method.

        Network equipment typically runs 24/7 with no utilization scaling
        (always-on infrastructure).

        Formula:
            energy_kwh = power_kw x operating_hours x count
            co2e_kg    = energy_kwh x grid_ef

        Args:
            power_kw: Switch power draw in kW (default 0.150).
            operating_hours: Annual operating hours (default 8760).
            country_code: ISO country code for grid EF (default "US").
            count: Number of network devices (default 1).
            egrid_subregion: Optional EPA eGRID subregion code.

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg,
            energy_kwh, power_kw, operating_hours, grid_ef, method,
            it_type, count, country_code, ef_source, dqi_score,
            provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If any input parameter is invalid.

        Example:
            >>> result = engine.calculate_network(
            ...     power_kw=Decimal("0.150"),
            ...     count=20,
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        return self._calculate_simple_it_asset(
            it_type="network_switch",
            method_label="network",
            power_kw=power_kw,
            operating_hours=operating_hours,
            country_code=country_code,
            count=count,
            egrid_subregion=egrid_subregion,
        )

    # ==================================================================
    # 3. calculate_storage
    # ==================================================================

    def calculate_storage(
        self,
        power_kw: Union[Decimal, int, float, str] = Decimal("0.200"),
        operating_hours: Union[Decimal, int, float, str] = Decimal("8760"),
        country_code: str = "US",
        count: int = 1,
        egrid_subregion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate storage array emissions using energy-based method.

        Storage arrays typically run 24/7 with no utilization scaling.

        Formula:
            energy_kwh = power_kw x operating_hours x count
            co2e_kg    = energy_kwh x grid_ef

        Args:
            power_kw: Storage array power draw in kW (default 0.200).
            operating_hours: Annual operating hours (default 8760).
            country_code: ISO country code for grid EF (default "US").
            count: Number of storage arrays (default 1).
            egrid_subregion: Optional EPA eGRID subregion code.

        Returns:
            Dict with calculation results and provenance.

        Raises:
            ValueError: If any input parameter is invalid.

        Example:
            >>> result = engine.calculate_storage(count=5)
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        return self._calculate_simple_it_asset(
            it_type="storage",
            method_label="storage",
            power_kw=power_kw,
            operating_hours=operating_hours,
            country_code=country_code,
            count=count,
            egrid_subregion=egrid_subregion,
        )

    # ==================================================================
    # 4. calculate_desktop
    # ==================================================================

    def calculate_desktop(
        self,
        power_kw: Union[Decimal, int, float, str] = Decimal("0.150"),
        operating_hours: Union[Decimal, int, float, str] = Decimal("2500"),
        country_code: str = "US",
        count: int = 1,
        egrid_subregion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate desktop computer emissions using energy-based method.

        Desktop PCs typically operate during business hours with some
        extended usage.

        Formula:
            energy_kwh = power_kw x operating_hours x count
            co2e_kg    = energy_kwh x grid_ef

        Args:
            power_kw: Desktop power draw in kW (default 0.150).
            operating_hours: Annual operating hours (default 2500).
            country_code: ISO country code for grid EF (default "US").
            count: Number of desktops (default 1).
            egrid_subregion: Optional EPA eGRID subregion code.

        Returns:
            Dict with calculation results and provenance.

        Raises:
            ValueError: If any input parameter is invalid.

        Example:
            >>> result = engine.calculate_desktop(count=100)
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        return self._calculate_simple_it_asset(
            it_type="desktop",
            method_label="desktop",
            power_kw=power_kw,
            operating_hours=operating_hours,
            country_code=country_code,
            count=count,
            egrid_subregion=egrid_subregion,
        )

    # ==================================================================
    # 5. calculate_laptop
    # ==================================================================

    def calculate_laptop(
        self,
        power_kw: Union[Decimal, int, float, str] = Decimal("0.045"),
        operating_hours: Union[Decimal, int, float, str] = Decimal("2000"),
        country_code: str = "US",
        count: int = 1,
        egrid_subregion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate laptop computer emissions using energy-based method.

        Laptops have lower power draw and fewer operating hours than desktops.

        Formula:
            energy_kwh = power_kw x operating_hours x count
            co2e_kg    = energy_kwh x grid_ef

        Args:
            power_kw: Laptop power draw in kW (default 0.045).
            operating_hours: Annual operating hours (default 2000).
            country_code: ISO country code for grid EF (default "US").
            count: Number of laptops (default 1).
            egrid_subregion: Optional EPA eGRID subregion code.

        Returns:
            Dict with calculation results and provenance.

        Raises:
            ValueError: If any input parameter is invalid.

        Example:
            >>> result = engine.calculate_laptop(count=200)
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        return self._calculate_simple_it_asset(
            it_type="laptop",
            method_label="laptop",
            power_kw=power_kw,
            operating_hours=operating_hours,
            country_code=country_code,
            count=count,
            egrid_subregion=egrid_subregion,
        )

    # ==================================================================
    # 6. calculate_printer
    # ==================================================================

    def calculate_printer(
        self,
        power_kw: Union[Decimal, int, float, str] = Decimal("0.200"),
        active_hours: Union[Decimal, int, float, str] = Decimal("1000"),
        standby_hours: Union[Decimal, int, float, str] = Decimal("3760"),
        standby_power_kw: Union[Decimal, int, float, str] = Decimal("0.020"),
        country_code: str = "US",
        count: int = 1,
        egrid_subregion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate printer emissions including active and standby modes.

        Printers have dual power states: active printing and standby (idle).
        Both states consume electricity and generate emissions.

        Formula:
            active_kwh  = power_kw x active_hours x count
            standby_kwh = standby_power_kw x standby_hours x count
            energy_kwh  = active_kwh + standby_kwh
            co2e_kg     = energy_kwh x grid_ef

        Args:
            power_kw: Active printing power in kW (default 0.200).
            active_hours: Annual active printing hours (default 1000).
            standby_hours: Annual standby hours (default 3760).
            standby_power_kw: Standby power in kW (default 0.020).
            country_code: ISO country code for grid EF (default "US").
            count: Number of printers (default 1).
            egrid_subregion: Optional EPA eGRID subregion code.

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg,
            energy_kwh, active_kwh, standby_kwh, power_kw,
            standby_power_kw, active_hours, standby_hours, grid_ef,
            method, it_type, count, country_code, ef_source, dqi_score,
            provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If any input parameter is invalid.

        Example:
            >>> result = engine.calculate_printer(count=10)
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        return self._calculate_dual_mode_asset(
            it_type="printer",
            method_label="printer",
            power_kw=power_kw,
            active_hours=active_hours,
            standby_hours=standby_hours,
            standby_power_kw=standby_power_kw,
            country_code=country_code,
            count=count,
            egrid_subregion=egrid_subregion,
        )

    # ==================================================================
    # 7. calculate_copier
    # ==================================================================

    def calculate_copier(
        self,
        power_kw: Union[Decimal, int, float, str] = Decimal("0.300"),
        active_hours: Union[Decimal, int, float, str] = Decimal("800"),
        standby_hours: Union[Decimal, int, float, str] = Decimal("3960"),
        standby_power_kw: Union[Decimal, int, float, str] = Decimal("0.030"),
        country_code: str = "US",
        count: int = 1,
        egrid_subregion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate copier emissions including active and standby modes.

        Copiers have dual power states: active copying and standby (idle).
        Both states consume electricity and generate emissions.

        Formula:
            active_kwh  = power_kw x active_hours x count
            standby_kwh = standby_power_kw x standby_hours x count
            energy_kwh  = active_kwh + standby_kwh
            co2e_kg     = energy_kwh x grid_ef

        Args:
            power_kw: Active copying power in kW (default 0.300).
            active_hours: Annual active copying hours (default 800).
            standby_hours: Annual standby hours (default 3960).
            standby_power_kw: Standby power in kW (default 0.030).
            country_code: ISO country code for grid EF (default "US").
            count: Number of copiers (default 1).
            egrid_subregion: Optional EPA eGRID subregion code.

        Returns:
            Dict with calculation results including active/standby breakdown.

        Raises:
            ValueError: If any input parameter is invalid.

        Example:
            >>> result = engine.calculate_copier(count=5)
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        return self._calculate_dual_mode_asset(
            it_type="copier",
            method_label="copier",
            power_kw=power_kw,
            active_hours=active_hours,
            standby_hours=standby_hours,
            standby_power_kw=standby_power_kw,
            country_code=country_code,
            count=count,
            egrid_subregion=egrid_subregion,
        )

    # ==================================================================
    # 8. calculate_data_center_allocation
    # ==================================================================

    def calculate_data_center_allocation(
        self,
        allocated_kw: Union[Decimal, int, float, str],
        pue: Union[Decimal, int, float, str] = Decimal("1.58"),
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate data center emissions from allocated power capacity.

        For organizations that lease data center space by power allocation
        (e.g., kW allocated for a cage, rack, or suite), this method
        calculates the full-year emissions based on allocated capacity.

        Formula:
            energy_kwh = allocated_kw x PUE x 8760
            co2e_kg    = energy_kwh x grid_ef

        Args:
            allocated_kw: Allocated IT power capacity in kW.
            pue: Data center PUE (default 1.58).
            country_code: ISO country code for grid EF (default "US").
            egrid_subregion: Optional EPA eGRID subregion code.

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg,
            energy_kwh, allocated_kw, pue, annual_hours, grid_ef,
            method, it_type, country_code, ef_source, dqi_score,
            provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If allocated_kw <= 0 or PUE < 1.0.

        Example:
            >>> result = engine.calculate_data_center_allocation(
            ...     allocated_kw=Decimal("50"),
            ...     pue=Decimal("1.40"),
            ...     country_code="US",
            ...     egrid_subregion="CAMX",
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            alloc_kw = self._validate_positive_decimal(allocated_kw, "allocated_kw")
            pue_val = self._validate_pue(pue)
            c_code = self._validate_country_code(country_code)

            # Step 2: Resolve grid EF (ZERO HALLUCINATION)
            grid_ef, ef_source = self._get_grid_ef(c_code, egrid_subregion)

            # Step 3: Calculate energy and emissions (Decimal only)
            energy_kwh = _q(alloc_kw * pue_val * _HOURS_PER_YEAR)
            co2e_kg = _q(energy_kwh * grid_ef)

            # Grid-based: all upstream
            ttw_co2e = _ZERO
            wtt_co2e = co2e_kg

            # Step 4: Build provenance hash
            input_data = {
                "it_type": "data_center_allocation",
                "allocated_kw": str(alloc_kw),
                "pue": str(pue_val),
                "country_code": c_code,
                "egrid_subregion": egrid_subregion,
            }
            output_data = {
                "co2e_kg": str(co2e_kg),
                "energy_kwh": str(energy_kwh),
            }
            provenance_hash = _calculate_provenance_hash(input_data, output_data)

            # Step 5: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": co2e_kg,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "energy_kwh": energy_kwh,
                "allocated_kw": alloc_kw,
                "pue": pue_val,
                "annual_hours": _HOURS_PER_YEAR,
                "grid_ef": grid_ef,
                "method": "data_center_allocation",
                "it_type": "data_center_allocation",
                "country_code": c_code,
                "egrid_subregion": egrid_subregion,
                "ef_source": ef_source,
                "dqi_score": DQI_SCORES["data_center_allocation"],
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "Data center allocation complete: alloc=%s kW, pue=%s, "
                "kWh=%s, co2e=%s kg, duration=%.4fs",
                alloc_kw, pue_val, energy_kwh, co2e_kg, duration,
            )

            return result

    # ==================================================================
    # 9. calculate_it_portfolio
    # ==================================================================

    def calculate_it_portfolio(
        self,
        assets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate aggregate IT portfolio emissions from multiple asset types.

        Iterates over a list of IT asset specifications, calculates emissions
        for each asset using the appropriate method, and aggregates totals
        for the entire IT portfolio.

        Each asset dict should contain:
            - it_type (str): Asset type (server, network_switch, storage,
              desktop, laptop, printer, copier, data_center_allocation).
            - Plus type-specific parameters (power_kw, count, etc.).

        Args:
            assets: List of IT asset specification dicts.

        Returns:
            Dict with keys: portfolio_co2e_kg, portfolio_ttw_co2e_kg,
            portfolio_wtt_co2e_kg, total_energy_kwh, total_assets,
            asset_results (list), errors (list), method,
            provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If assets list is empty or exceeds _MAX_BATCH_SIZE.

        Example:
            >>> result = engine.calculate_it_portfolio([
            ...     {"it_type": "server", "power_kw": 0.500, "count": 10},
            ...     {"it_type": "laptop", "count": 200},
            ...     {"it_type": "printer", "count": 15},
            ... ])
            >>> assert result["portfolio_co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        if not assets:
            raise ValueError("Assets list must not be empty")
        if len(assets) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Portfolio size {len(assets)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        asset_results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        portfolio_co2e = _ZERO
        portfolio_ttw = _ZERO
        portfolio_wtt = _ZERO
        total_energy = _ZERO
        total_assets = 0

        for idx, spec in enumerate(assets):
            try:
                asset_result = self._dispatch_it_asset(idx, spec)
                asset_results.append(asset_result)
                portfolio_co2e = _q(portfolio_co2e + asset_result.get("co2e_kg", _ZERO))
                portfolio_ttw = _q(portfolio_ttw + asset_result.get("ttw_co2e_kg", _ZERO))
                portfolio_wtt = _q(portfolio_wtt + asset_result.get("wtt_co2e_kg", _ZERO))
                total_energy = _q(total_energy + asset_result.get("energy_kwh", _ZERO))
                total_assets += asset_result.get("count", 1)
            except Exception as exc:
                logger.warning(
                    "IT portfolio asset %d failed: %s", idx, str(exc)
                )
                errors.append({
                    "index": idx,
                    "error": str(exc),
                    "asset_spec": spec,
                })

        # Build provenance hash
        input_data = {
            "asset_types": len(assets),
            "total_assets": total_assets,
        }
        output_data = {
            "portfolio_co2e_kg": str(portfolio_co2e),
            "portfolio_ttw_co2e_kg": str(portfolio_ttw),
            "portfolio_wtt_co2e_kg": str(portfolio_wtt),
            "total_energy_kwh": str(total_energy),
        }
        provenance_hash = _calculate_provenance_hash(input_data, output_data)

        timestamp = datetime.now(timezone.utc).isoformat()
        result: Dict[str, Any] = {
            "portfolio_co2e_kg": portfolio_co2e,
            "portfolio_ttw_co2e_kg": portfolio_ttw,
            "portfolio_wtt_co2e_kg": portfolio_wtt,
            "total_energy_kwh": total_energy,
            "total_assets": total_assets,
            "asset_types_processed": len(asset_results),
            "asset_types_failed": len(errors),
            "asset_results": asset_results,
            "errors": errors,
            "method": "it_portfolio",
            "dqi_score": DQI_SCORES["it_portfolio"],
            "provenance_hash": provenance_hash,
            "calculation_timestamp": timestamp,
        }

        duration = time.monotonic() - start_time
        logger.info(
            "IT portfolio calculation complete: types=%d, assets=%d, "
            "co2e=%s kg, kWh=%s, errors=%d, duration=%.4fs",
            len(asset_results), total_assets,
            portfolio_co2e, total_energy, len(errors), duration,
        )

        return result

    def _dispatch_it_asset(
        self, idx: int, spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Dispatch a single IT asset calculation based on it_type.

        Args:
            idx: Asset index for logging.
            spec: IT asset specification dict with 'it_type' key.

        Returns:
            Calculation result dict.

        Raises:
            ValueError: If it_type is unknown or required params missing.
        """
        it_type = spec.get("it_type", "").lower().strip()

        if it_type == "server":
            return self.calculate_server(
                power_kw=spec.get("power_kw", Decimal("0.500")),
                pue=spec.get("pue", Decimal("1.58")),
                utilization=spec.get("utilization", Decimal("0.30")),
                operating_hours=spec.get("operating_hours", Decimal("8760")),
                country_code=spec.get("country_code", "US"),
                count=spec.get("count", 1),
                egrid_subregion=spec.get("egrid_subregion"),
            )
        elif it_type in ("network_switch", "network"):
            return self.calculate_network(
                power_kw=spec.get("power_kw", Decimal("0.150")),
                operating_hours=spec.get("operating_hours", Decimal("8760")),
                country_code=spec.get("country_code", "US"),
                count=spec.get("count", 1),
                egrid_subregion=spec.get("egrid_subregion"),
            )
        elif it_type == "storage":
            return self.calculate_storage(
                power_kw=spec.get("power_kw", Decimal("0.200")),
                operating_hours=spec.get("operating_hours", Decimal("8760")),
                country_code=spec.get("country_code", "US"),
                count=spec.get("count", 1),
                egrid_subregion=spec.get("egrid_subregion"),
            )
        elif it_type == "desktop":
            return self.calculate_desktop(
                power_kw=spec.get("power_kw", Decimal("0.150")),
                operating_hours=spec.get("operating_hours", Decimal("2500")),
                country_code=spec.get("country_code", "US"),
                count=spec.get("count", 1),
                egrid_subregion=spec.get("egrid_subregion"),
            )
        elif it_type == "laptop":
            return self.calculate_laptop(
                power_kw=spec.get("power_kw", Decimal("0.045")),
                operating_hours=spec.get("operating_hours", Decimal("2000")),
                country_code=spec.get("country_code", "US"),
                count=spec.get("count", 1),
                egrid_subregion=spec.get("egrid_subregion"),
            )
        elif it_type == "printer":
            return self.calculate_printer(
                power_kw=spec.get("power_kw", Decimal("0.200")),
                active_hours=spec.get("active_hours", Decimal("1000")),
                standby_hours=spec.get("standby_hours", Decimal("3760")),
                standby_power_kw=spec.get("standby_power_kw", Decimal("0.020")),
                country_code=spec.get("country_code", "US"),
                count=spec.get("count", 1),
                egrid_subregion=spec.get("egrid_subregion"),
            )
        elif it_type == "copier":
            return self.calculate_copier(
                power_kw=spec.get("power_kw", Decimal("0.300")),
                active_hours=spec.get("active_hours", Decimal("800")),
                standby_hours=spec.get("standby_hours", Decimal("3960")),
                standby_power_kw=spec.get("standby_power_kw", Decimal("0.030")),
                country_code=spec.get("country_code", "US"),
                count=spec.get("count", 1),
                egrid_subregion=spec.get("egrid_subregion"),
            )
        elif it_type in ("data_center_allocation", "data_center"):
            return self.calculate_data_center_allocation(
                allocated_kw=spec["allocated_kw"],
                pue=spec.get("pue", Decimal("1.58")),
                country_code=spec.get("country_code", "US"),
                egrid_subregion=spec.get("egrid_subregion"),
            )
        else:
            raise ValueError(
                f"Unknown IT asset type '{it_type}' at index {idx}. "
                f"Supported: {VALID_IT_TYPES + ['data_center_allocation']}"
            )

    # ==================================================================
    # 10. calculate_batch
    # ==================================================================

    def calculate_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process multiple IT asset calculations in a single batch.

        Each item dict must contain an 'it_type' key indicating the asset
        type plus asset-specific parameters. Failed calculations do not
        halt the batch.

        Args:
            items: List of dicts, each with 'it_type' and asset-specific params.

        Returns:
            Dict with keys: total_co2e_kg, total_ttw_co2e_kg,
            total_wtt_co2e_kg, total_energy_kwh, items_processed,
            items_failed, results (list), errors (list),
            provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If items list exceeds _MAX_BATCH_SIZE.

        Example:
            >>> batch_result = engine.calculate_batch([
            ...     {"it_type": "server", "count": 10},
            ...     {"it_type": "laptop", "count": 100},
            ... ])
            >>> assert batch_result["total_co2e_kg"] > Decimal("0")
        """
        if len(items) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total_co2e = _ZERO
        total_ttw = _ZERO
        total_wtt = _ZERO
        total_energy = _ZERO

        for idx, item in enumerate(items):
            try:
                calc_result = self._dispatch_it_asset(idx, item)
                results.append({
                    "index": idx,
                    "status": "success",
                    "result": calc_result,
                })
                total_co2e = _q(total_co2e + calc_result.get("co2e_kg", _ZERO))
                total_ttw = _q(total_ttw + calc_result.get("ttw_co2e_kg", _ZERO))
                total_wtt = _q(total_wtt + calc_result.get("wtt_co2e_kg", _ZERO))
                total_energy = _q(total_energy + calc_result.get("energy_kwh", _ZERO))
            except Exception as exc:
                logger.warning(
                    "Batch item %d failed: %s", idx, str(exc)
                )
                errors.append({
                    "index": idx,
                    "status": "error",
                    "error": str(exc),
                    "item": item,
                })

        # Build provenance hash
        input_data = {
            "batch_size": len(items),
            "items_processed": len(results),
            "items_failed": len(errors),
        }
        output_data = {
            "total_co2e_kg": str(total_co2e),
            "total_ttw_co2e_kg": str(total_ttw),
            "total_wtt_co2e_kg": str(total_wtt),
            "total_energy_kwh": str(total_energy),
        }
        provenance_hash = _calculate_provenance_hash(input_data, output_data)

        timestamp = datetime.now(timezone.utc).isoformat()
        batch_result: Dict[str, Any] = {
            "total_co2e_kg": total_co2e,
            "total_ttw_co2e_kg": total_ttw,
            "total_wtt_co2e_kg": total_wtt,
            "total_energy_kwh": total_energy,
            "items_processed": len(results),
            "items_failed": len(errors),
            "results": results,
            "errors": errors,
            "provenance_hash": provenance_hash,
            "calculation_timestamp": timestamp,
        }

        duration = time.monotonic() - start_time
        logger.info(
            "Batch calculation complete: total=%d, success=%d, "
            "errors=%d, co2e=%s kg, kWh=%s, duration=%.4fs",
            len(items), len(results), len(errors),
            total_co2e, total_energy, duration,
        )

        return batch_result

    # ==================================================================
    # 11. estimate_annual_emissions
    # ==================================================================

    def estimate_annual_emissions(
        self,
        it_type: str,
        count: int = 1,
        country_code: str = "US",
    ) -> Dict[str, Any]:
        """
        Quick annual emission estimation using default parameters.

        Provides a rapid estimate for screening-level assessments when
        detailed operating data is not available. Uses standard power
        ratings and operating hours for the specified IT asset type.

        Args:
            it_type: IT asset type key (server, network_switch, storage,
                     desktop, laptop, printer, copier).
            count: Number of assets (default 1).
            country_code: ISO country code for grid EF (default "US").

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg,
            energy_kwh, method, it_type, count, ef_source,
            dqi_score, provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If it_type is invalid or country_code unknown.

        Example:
            >>> result = engine.estimate_annual_emissions(
            ...     it_type="server",
            ...     count=50,
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        normalized_type = it_type.lower().strip()

        if normalized_type == "server":
            return self.calculate_server(
                power_kw=IT_POWER_RATINGS["server"],
                pue=PUE_DEFAULTS["data_center"],
                utilization=DEFAULT_SERVER_UTILIZATION,
                operating_hours=IT_OPERATING_HOURS["server"],
                country_code=country_code,
                count=count,
            )
        elif normalized_type in ("network_switch", "network"):
            return self.calculate_network(
                power_kw=IT_POWER_RATINGS["network_switch"],
                operating_hours=IT_OPERATING_HOURS["network_switch"],
                country_code=country_code,
                count=count,
            )
        elif normalized_type == "storage":
            return self.calculate_storage(
                power_kw=IT_POWER_RATINGS["storage"],
                operating_hours=IT_OPERATING_HOURS["storage"],
                country_code=country_code,
                count=count,
            )
        elif normalized_type == "desktop":
            return self.calculate_desktop(
                power_kw=IT_POWER_RATINGS["desktop"],
                operating_hours=IT_OPERATING_HOURS["desktop"],
                country_code=country_code,
                count=count,
            )
        elif normalized_type == "laptop":
            return self.calculate_laptop(
                power_kw=IT_POWER_RATINGS["laptop"],
                operating_hours=IT_OPERATING_HOURS["laptop"],
                country_code=country_code,
                count=count,
            )
        elif normalized_type == "printer":
            return self.calculate_printer(
                power_kw=IT_POWER_RATINGS["printer"],
                active_hours=IT_OPERATING_HOURS["printer"],
                standby_hours=IT_STANDBY_PARAMS["printer"]["standby_hours"],
                standby_power_kw=IT_STANDBY_PARAMS["printer"]["standby_power_kw"],
                country_code=country_code,
                count=count,
            )
        elif normalized_type == "copier":
            return self.calculate_copier(
                power_kw=IT_POWER_RATINGS["copier"],
                active_hours=IT_OPERATING_HOURS["copier"],
                standby_hours=IT_STANDBY_PARAMS["copier"]["standby_hours"],
                standby_power_kw=IT_STANDBY_PARAMS["copier"]["standby_power_kw"],
                country_code=country_code,
                count=count,
            )
        else:
            raise ValueError(
                f"Unknown IT asset type '{it_type}'. "
                f"Available: {VALID_IT_TYPES}"
            )

    # ==================================================================
    # INTERNAL: Simple IT Asset Calculator (network, storage, desktop, laptop)
    # ==================================================================

    def _calculate_simple_it_asset(
        self,
        it_type: str,
        method_label: str,
        power_kw: Union[Decimal, int, float, str],
        operating_hours: Union[Decimal, int, float, str],
        country_code: str,
        count: int,
        egrid_subregion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a simple IT asset (no PUE, no standby).

        Internal helper shared by network, storage, desktop, and laptop.

        Formula:
            energy_kwh = power_kw x operating_hours x count
            co2e_kg    = energy_kwh x grid_ef

        Args:
            it_type: IT asset type key for result metadata.
            method_label: Method label for result metadata.
            power_kw: Asset power draw in kW.
            operating_hours: Annual operating hours.
            country_code: ISO country code.
            count: Number of assets.
            egrid_subregion: Optional eGRID subregion code.

        Returns:
            Dict with calculation results and provenance.
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            pwr = self._validate_positive_decimal(power_kw, "power_kw")
            hours = self._validate_positive_decimal(operating_hours, "operating_hours")
            c_code = self._validate_country_code(country_code)
            cnt = self._validate_count(count)

            # Step 2: Resolve grid EF (ZERO HALLUCINATION)
            grid_ef, ef_source = self._get_grid_ef(c_code, egrid_subregion)

            # Step 3: Calculate energy and emissions (Decimal only)
            count_dec = _safe_decimal(cnt)
            energy_kwh = _q(pwr * hours * count_dec)
            co2e_kg = _q(energy_kwh * grid_ef)

            # Grid-based: all upstream
            ttw_co2e = _ZERO
            wtt_co2e = co2e_kg

            # Step 4: Build provenance hash
            input_data = {
                "it_type": it_type,
                "power_kw": str(pwr),
                "operating_hours": str(hours),
                "country_code": c_code,
                "egrid_subregion": egrid_subregion,
                "count": cnt,
            }
            output_data = {
                "co2e_kg": str(co2e_kg),
                "energy_kwh": str(energy_kwh),
            }
            provenance_hash = _calculate_provenance_hash(input_data, output_data)

            # Step 5: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            dqi_key = method_label if method_label in DQI_SCORES else "estimate"
            result: Dict[str, Any] = {
                "co2e_kg": co2e_kg,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "energy_kwh": energy_kwh,
                "power_kw": pwr,
                "operating_hours": hours,
                "grid_ef": grid_ef,
                "method": method_label,
                "it_type": it_type,
                "count": cnt,
                "country_code": c_code,
                "egrid_subregion": egrid_subregion,
                "ef_source": ef_source,
                "dqi_score": DQI_SCORES.get(dqi_key, DQI_SCORES["estimate"]),
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "%s calculation complete: power=%s kW, hours=%s, "
                "count=%d, kWh=%s, co2e=%s kg, duration=%.4fs",
                it_type, pwr, hours, cnt,
                energy_kwh, co2e_kg, duration,
            )

            return result

    # ==================================================================
    # INTERNAL: Dual-Mode IT Asset Calculator (printer, copier)
    # ==================================================================

    def _calculate_dual_mode_asset(
        self,
        it_type: str,
        method_label: str,
        power_kw: Union[Decimal, int, float, str],
        active_hours: Union[Decimal, int, float, str],
        standby_hours: Union[Decimal, int, float, str],
        standby_power_kw: Union[Decimal, int, float, str],
        country_code: str,
        count: int,
        egrid_subregion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a dual-mode IT asset (active + standby).

        Internal helper shared by printer and copier.

        Formula:
            active_kwh  = power_kw x active_hours x count
            standby_kwh = standby_power_kw x standby_hours x count
            energy_kwh  = active_kwh + standby_kwh
            co2e_kg     = energy_kwh x grid_ef

        Args:
            it_type: IT asset type key.
            method_label: Method label for result metadata.
            power_kw: Active power draw in kW.
            active_hours: Annual active hours.
            standby_hours: Annual standby hours.
            standby_power_kw: Standby power draw in kW.
            country_code: ISO country code.
            count: Number of assets.
            egrid_subregion: Optional eGRID subregion code.

        Returns:
            Dict with calculation results including active/standby breakdown.
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            pwr = self._validate_positive_decimal(power_kw, "power_kw")
            a_hours = self._validate_non_negative_decimal(active_hours, "active_hours")
            s_hours = self._validate_non_negative_decimal(standby_hours, "standby_hours")
            s_power = self._validate_non_negative_decimal(standby_power_kw, "standby_power_kw")
            c_code = self._validate_country_code(country_code)
            cnt = self._validate_count(count)

            # Step 2: Resolve grid EF (ZERO HALLUCINATION)
            grid_ef, ef_source = self._get_grid_ef(c_code, egrid_subregion)

            # Step 3: Calculate energy and emissions (Decimal only)
            count_dec = _safe_decimal(cnt)
            active_kwh = _q(pwr * a_hours * count_dec)
            standby_kwh = _q(s_power * s_hours * count_dec)
            energy_kwh = _q(active_kwh + standby_kwh)
            co2e_kg = _q(energy_kwh * grid_ef)

            # Grid-based: all upstream
            ttw_co2e = _ZERO
            wtt_co2e = co2e_kg

            # Step 4: Build provenance hash
            input_data = {
                "it_type": it_type,
                "power_kw": str(pwr),
                "active_hours": str(a_hours),
                "standby_hours": str(s_hours),
                "standby_power_kw": str(s_power),
                "country_code": c_code,
                "egrid_subregion": egrid_subregion,
                "count": cnt,
            }
            output_data = {
                "co2e_kg": str(co2e_kg),
                "energy_kwh": str(energy_kwh),
                "active_kwh": str(active_kwh),
                "standby_kwh": str(standby_kwh),
            }
            provenance_hash = _calculate_provenance_hash(input_data, output_data)

            # Step 5: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            dqi_key = method_label if method_label in DQI_SCORES else "estimate"
            result: Dict[str, Any] = {
                "co2e_kg": co2e_kg,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "energy_kwh": energy_kwh,
                "active_kwh": active_kwh,
                "standby_kwh": standby_kwh,
                "power_kw": pwr,
                "standby_power_kw": s_power,
                "active_hours": a_hours,
                "standby_hours": s_hours,
                "grid_ef": grid_ef,
                "method": method_label,
                "it_type": it_type,
                "count": cnt,
                "country_code": c_code,
                "egrid_subregion": egrid_subregion,
                "ef_source": ef_source,
                "dqi_score": DQI_SCORES.get(dqi_key, DQI_SCORES["estimate"]),
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "%s calculation complete: power=%s kW, active=%s h, "
                "standby=%s h, count=%d, kWh=%s (active=%s + standby=%s), "
                "co2e=%s kg, duration=%.4fs",
                it_type, pwr, a_hours, s_hours, cnt,
                energy_kwh, active_kwh, standby_kwh, co2e_kg, duration,
            )

            return result

    # ==================================================================
    # EMISSION FACTOR ACCESSORS (read-only)
    # ==================================================================

    @staticmethod
    def get_it_power_ratings() -> Dict[str, str]:
        """
        Return all IT asset power ratings as a serializable dict.

        Returns:
            Dict mapping IT type to power rating in kW as string.
        """
        return {it_type: str(power) for it_type, power in IT_POWER_RATINGS.items()}

    @staticmethod
    def get_it_operating_hours() -> Dict[str, str]:
        """
        Return all IT asset operating hours as a serializable dict.

        Returns:
            Dict mapping IT type to operating hours as string.
        """
        return {it_type: str(hours) for it_type, hours in IT_OPERATING_HOURS.items()}

    @staticmethod
    def get_pue_defaults() -> Dict[str, str]:
        """
        Return all PUE defaults as a serializable dict.

        Returns:
            Dict mapping facility type to PUE as string.
        """
        return {ftype: str(pue) for ftype, pue in PUE_DEFAULTS.items()}

    @staticmethod
    def get_grid_emission_factors() -> Dict[str, str]:
        """
        Return all grid emission factors as a serializable dict.

        Returns:
            Dict mapping country code to kgCO2e/kWh string.
        """
        return {code: str(ef) for code, ef in GRID_EMISSION_FACTORS.items()}

    @staticmethod
    def get_egrid_emission_factors() -> Dict[str, str]:
        """
        Return all eGRID subregion emission factors as a serializable dict.

        Returns:
            Dict mapping subregion code to kgCO2e/kWh string.
        """
        return {code: str(ef) for code, ef in EGRID_EMISSION_FACTORS.items()}

    @staticmethod
    def get_standby_params() -> Dict[str, Dict[str, str]]:
        """
        Return standby parameters for printers and copiers.

        Returns:
            Dict mapping IT type to standby params.
        """
        return {
            it_type: {k: str(v) for k, v in params.items()}
            for it_type, params in IT_STANDBY_PARAMS.items()
        }

    @staticmethod
    def get_supported_it_types() -> List[str]:
        """
        Return list of supported IT asset types.

        Returns:
            List of IT type key strings.
        """
        return list(VALID_IT_TYPES)

    # ==================================================================
    # ENGINE INFO
    # ==================================================================

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Return engine metadata and status information.

        Returns:
            Dict with engine_id, version, agent_id, calculation_count,
            initialized_at, it_types, pue_defaults.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "calculation_count": self._calculation_count,
            "initialized_at": self._initialized_at,
            "it_types": VALID_IT_TYPES,
            "pue_defaults": {k: str(v) for k, v in PUE_DEFAULTS.items()},
            "grid_countries": list(GRID_EMISSION_FACTORS.keys()),
            "egrid_subregions": list(EGRID_EMISSION_FACTORS.keys()),
        }


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "ITAssetsCalculatorEngine",
    "get_it_assets_calculator",
    "reset_it_assets_calculator",
    "ENGINE_ID",
    "ENGINE_VERSION",
    "IT_POWER_RATINGS",
    "IT_OPERATING_HOURS",
    "IT_STANDBY_PARAMS",
    "PUE_DEFAULTS",
    "DEFAULT_SERVER_UTILIZATION",
    "GRID_EMISSION_FACTORS",
    "EGRID_EMISSION_FACTORS",
    "DQI_SCORES",
    "VALID_IT_TYPES",
]
