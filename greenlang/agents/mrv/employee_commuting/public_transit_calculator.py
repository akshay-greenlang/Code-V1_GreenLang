# -*- coding: utf-8 -*-
"""
Public Transit Calculator Engine - Engine 3: Employee Commuting Agent (AGENT-MRV-020)

Calculates GHG emissions from public transit commuting including local/express bus,
coach, commuter rail, subway/metro, light rail, tram/streetcar, and ferry services.
Uses passenger-kilometer (pkm) based emission factors from DEFRA 2024.

Primary Formulae:
    Single-mode transit commute:
        daily_distance  = one_way_distance x (2 if round_trip else 1) x trips_per_day
        annual_distance = daily_distance x working_days x (1 - wfh_fraction)

        ttw_co2e  = annual_distance x ef.co2e_per_pkm
        wtt_co2e  = annual_distance x ef.wtt_per_pkm   (if include_wtt)
        total_co2e = ttw_co2e + wtt_co2e

        co2  = annual_distance x ef.co2_per_pkm
        ch4  = annual_distance x ef.ch4_per_pkm
        n2o  = annual_distance x ef.n2o_per_pkm

    Multi-modal transit commute:
        For each segment i in segments:
            segment_annual_distance_i = segment_distance_i x (2 if round_trip else 1)
                                        x working_days x (1 - wfh_fraction)
            segment_co2e_i = segment_annual_distance_i x ef_i.co2e_per_pkm
            ...
        total_co2e = SUM(segment_co2e_i)

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate value is deterministic and reproducible
    - SHA-256 provenance hash on every result
    - Emission factors sourced from DEFRA 2024

Supports:
    - 9 transit types (local bus, express bus, coach, commuter rail,
      subway/metro, light rail, tram/streetcar, ferry, water taxi)
    - Per-gas breakdown (CO2, CH4, N2O) in addition to CO2e
    - Well-to-tank (WTT) upstream fuel cycle emissions
    - Work-from-home (WFH) fraction adjustment
    - Round-trip and one-way distance modes
    - Multiple trips per day
    - Multi-modal (multi-segment) transit commutes
    - Convenience methods for bus, rail, and ferry
    - Batch processing for multiple employees
    - Quick annual estimates with sensible defaults
    - Transit mode comparison / ranking
    - Input validation with detailed error messages
    - Provenance hash integration for audit trails
    - Prometheus metrics integration

Example:
    >>> from greenlang.agents.mrv.employee_commuting.public_transit_calculator import (
    ...     get_public_transit_calculator,
    ... )
    >>> from decimal import Decimal
    >>> engine = get_public_transit_calculator()
    >>> result = engine.calculate_transit_commute(
    ...     transit_type="commuter_rail",
    ...     one_way_distance_km=Decimal("25"),
    ...     working_days=225,
    ... )
    >>> assert result["total_co2e_kg"] > Decimal("0")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-020 Employee Commuting (GL-MRV-S3-007)
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

from greenlang.agents.mrv.employee_commuting.config import get_config
from greenlang.agents.mrv.employee_commuting.metrics import get_metrics
from greenlang.agents.mrv.employee_commuting.provenance import get_provenance_tracker

logger = logging.getLogger(__name__)

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-007"
AGENT_COMPONENT: str = "AGENT-MRV-020"
ENGINE_ID: str = "public_transit_calculator_engine"
ENGINE_VERSION: str = "1.0.0"

# ==============================================================================
# DECIMAL PRECISION & CONSTANTS
# ==============================================================================

_QUANT_8DP = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
ROUNDING = ROUND_HALF_UP

# Batch processing limits
_MAX_BATCH_SIZE = 10000

# Default working days (global default; callers should prefer regional values)
_DEFAULT_WORKING_DAYS = 240

# ==============================================================================
# TRANSIT EMISSION FACTORS (kg CO2e per passenger-km) - DEFRA 2024
#
# Each entry provides:
#   co2e_per_pkm: Tank-to-wheel CO2-equivalent per passenger-km
#   co2_per_pkm:  CO2-only component per passenger-km
#   ch4_per_pkm:  CH4 component per passenger-km
#   n2o_per_pkm:  N2O component per passenger-km
#   wtt_per_pkm:  Well-to-tank upstream emissions per passenger-km
#   source:       Data provenance label
# ==============================================================================

TRANSIT_EFS: Dict[str, Dict[str, Any]] = {
    "local_bus": {
        "co2e_per_pkm": Decimal("0.10312"),
        "co2_per_pkm": Decimal("0.10189"),
        "ch4_per_pkm": Decimal("0.00002"),
        "n2o_per_pkm": Decimal("0.00121"),
        "wtt_per_pkm": Decimal("0.02473"),
        "source": "DEFRA_2024",
    },
    "express_bus": {
        "co2e_per_pkm": Decimal("0.08956"),
        "co2_per_pkm": Decimal("0.08842"),
        "ch4_per_pkm": Decimal("0.00002"),
        "n2o_per_pkm": Decimal("0.00112"),
        "wtt_per_pkm": Decimal("0.02149"),
        "source": "DEFRA_2024",
    },
    "coach": {
        "co2e_per_pkm": Decimal("0.02732"),
        "co2_per_pkm": Decimal("0.02699"),
        "ch4_per_pkm": Decimal("0.00001"),
        "n2o_per_pkm": Decimal("0.00032"),
        "wtt_per_pkm": Decimal("0.00656"),
        "source": "DEFRA_2024",
    },
    "commuter_rail": {
        "co2e_per_pkm": Decimal("0.04115"),
        "co2_per_pkm": Decimal("0.04062"),
        "ch4_per_pkm": Decimal("0.00001"),
        "n2o_per_pkm": Decimal("0.00052"),
        "wtt_per_pkm": Decimal("0.00988"),
        "source": "DEFRA_2024",
    },
    "subway_metro": {
        "co2e_per_pkm": Decimal("0.03071"),
        "co2_per_pkm": Decimal("0.03033"),
        "ch4_per_pkm": Decimal("0.00001"),
        "n2o_per_pkm": Decimal("0.00037"),
        "wtt_per_pkm": Decimal("0.00737"),
        "source": "DEFRA_2024",
    },
    "light_rail": {
        "co2e_per_pkm": Decimal("0.02904"),
        "co2_per_pkm": Decimal("0.02869"),
        "ch4_per_pkm": Decimal("0.00001"),
        "n2o_per_pkm": Decimal("0.00034"),
        "wtt_per_pkm": Decimal("0.00697"),
        "source": "DEFRA_2024",
    },
    "tram_streetcar": {
        "co2e_per_pkm": Decimal("0.02940"),
        "co2_per_pkm": Decimal("0.02904"),
        "ch4_per_pkm": Decimal("0.00001"),
        "n2o_per_pkm": Decimal("0.00035"),
        "wtt_per_pkm": Decimal("0.00706"),
        "source": "DEFRA_2024",
    },
    "ferry_boat": {
        "co2e_per_pkm": Decimal("0.11318"),
        "co2_per_pkm": Decimal("0.11182"),
        "ch4_per_pkm": Decimal("0.00003"),
        "n2o_per_pkm": Decimal("0.00133"),
        "wtt_per_pkm": Decimal("0.02716"),
        "source": "DEFRA_2024",
    },
    "water_taxi": {
        "co2e_per_pkm": Decimal("0.14782"),
        "co2_per_pkm": Decimal("0.14601"),
        "ch4_per_pkm": Decimal("0.00004"),
        "n2o_per_pkm": Decimal("0.00177"),
        "wtt_per_pkm": Decimal("0.03548"),
        "source": "DEFRA_2024",
    },
}

# Transit type groupings for convenience methods
_BUS_TYPES = {"local_bus", "express_bus", "coach"}
_RAIL_TYPES = {"commuter_rail", "subway_metro", "light_rail", "tram_streetcar"}
_FERRY_TYPES = {"ferry_boat", "water_taxi"}

# ==============================================================================
# SINGLETON INSTANCE MANAGEMENT
# ==============================================================================

_instance: Optional["PublicTransitCalculatorEngine"] = None
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
    return value.quantize(_QUANT_8DP, rounding=ROUNDING)


def _to_decimal(value: Any, name: str = "value") -> Decimal:
    """
    Safely convert a value to Decimal.

    Args:
        value: Value to convert (str, int, float, or Decimal).
        name: Parameter name for error messages.

    Returns:
        Decimal representation of the value.

    Raises:
        ValueError: If value cannot be converted to Decimal.
        TypeError: If value is of unsupported type.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, str)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            raise ValueError(
                f"Cannot convert {name}={value!r} to Decimal"
            )
    if isinstance(value, float):
        return Decimal(str(value))
    raise TypeError(
        f"Cannot convert {name} of type {type(value).__name__} to Decimal"
    )


def _serialize_for_hash(obj: Any) -> str:
    """
    Serialize object to deterministic JSON for SHA-256 hashing.

    Converts Decimal to string, sorts dict keys, and handles nested
    structures deterministically so the same logical input always
    produces the same hash.

    Args:
        obj: Object to serialize.

    Returns:
        Deterministic JSON string.
    """

    def default_handler(o: Any) -> Any:
        """Handle non-JSON-serializable types."""
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=default_handler)


# ==============================================================================
# PublicTransitCalculatorEngine
# ==============================================================================


class PublicTransitCalculatorEngine:
    """
    Engine 3: Public transit emissions calculator for employee commuting.

    Implements deterministic emissions calculations for public transit modes
    (bus, rail, subway, tram, ferry) using DEFRA 2024 passenger-km emission
    factors. Calculates annual commute emissions based on one-way distance,
    working days, work-from-home fraction, and transit mode selection.

    The engine follows GreenLang's zero-hallucination principle by using only
    deterministic Decimal arithmetic with DEFRA-sourced parameters. No LLM
    calls are made anywhere in the calculation pipeline.

    Thread Safety:
        This engine is fully thread-safe. A reentrant lock protects shared
        state during calculations. The singleton instance is created lazily
        with double-checked locking.

    Attributes:
        _config: Employee commuting configuration singleton.
        _metrics: Prometheus metrics collector for monitoring.
        _provenance: SHA-256 provenance tracker for audit trails.
        _lock: Reentrant lock for thread safety.
        _calculation_count: Running count of calculations performed.

    Example:
        >>> engine = PublicTransitCalculatorEngine.get_instance()
        >>> result = engine.calculate_transit_commute(
        ...     transit_type="subway_metro",
        ...     one_way_distance_km=Decimal("12"),
        ...     working_days=225,
        ... )
        >>> assert result["total_co2e_kg"] > Decimal("0")
    """

    # ------------------------------------------------------------------
    # Singleton Access
    # ------------------------------------------------------------------

    @staticmethod
    def get_instance() -> "PublicTransitCalculatorEngine":
        """
        Get or create the singleton PublicTransitCalculatorEngine instance.

        Thread-safe lazy initialization using double-checked locking.

        Returns:
            Singleton PublicTransitCalculatorEngine instance.
        """
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:
                    _instance = PublicTransitCalculatorEngine()
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
        """Initialise the PublicTransitCalculatorEngine."""
        self._config = get_config()
        self._metrics = get_metrics()
        self._provenance = get_provenance_tracker()
        self._lock: threading.RLock = threading.RLock()
        self._calculation_count: int = 0

        logger.info(
            "PublicTransitCalculatorEngine initialised: engine=%s, version=%s, "
            "agent=%s, transit_types=%d",
            ENGINE_ID,
            ENGINE_VERSION,
            AGENT_ID,
            len(TRANSIT_EFS),
        )

    # ==================================================================
    # PROPERTY: calculation_count
    # ==================================================================

    @property
    def calculation_count(self) -> int:
        """Return the total number of calculations performed by this engine."""
        return self._calculation_count

    # ==================================================================
    # 1. calculate_transit_commute - Core single-mode calculation
    # ==================================================================

    def calculate_transit_commute(
        self,
        transit_type: str,
        one_way_distance_km: Union[Decimal, str, int, float],
        working_days: int = _DEFAULT_WORKING_DAYS,
        wfh_fraction: Union[Decimal, str, int, float] = _ZERO,
        round_trip: bool = True,
        include_wtt: bool = True,
        trips_per_day: int = 1,
    ) -> Dict[str, Any]:
        """
        Calculate annual public transit commute emissions for a single mode.

        Formula:
            daily_distance  = one_way_distance x (2 if round_trip else 1) x trips_per_day
            annual_distance = daily_distance x working_days x (1 - wfh_fraction)

            ttw_co2e  = annual_distance x ef.co2e_per_pkm
            wtt_co2e  = annual_distance x ef.wtt_per_pkm  (if include_wtt)
            total_co2e = ttw_co2e + wtt_co2e

            co2 = annual_distance x ef.co2_per_pkm
            ch4 = annual_distance x ef.ch4_per_pkm
            n2o = annual_distance x ef.n2o_per_pkm

        Args:
            transit_type: Public transit type key (e.g., "local_bus",
                "commuter_rail", "subway_metro"). Case-insensitive,
                spaces/hyphens normalized to underscores.
            one_way_distance_km: One-way commute distance in km. Must be > 0.
            working_days: Annual commuting days. Must be > 0. Default 240.
            wfh_fraction: Fraction of working days spent working from home
                (0.0 = fully in-office, 1.0 = fully remote). Default 0.
            round_trip: If True (default), double the one-way distance for
                daily commute. If False, use one-way distance only.
            include_wtt: If True (default), include well-to-tank upstream
                emissions. If False, only tank-to-wheel.
            trips_per_day: Number of transit trips per commute day
                (default 1). Useful for mid-day transit trips.

        Returns:
            Dict with keys:
                co2e_kg (Decimal): Total annual CO2e including WTT if enabled
                ttw_co2e_kg (Decimal): Tank-to-wheel CO2e
                wtt_co2e_kg (Decimal): Well-to-tank CO2e (0 if not included)
                co2_kg (Decimal): Annual CO2 emissions
                ch4_kg (Decimal): Annual CH4 emissions
                n2o_kg (Decimal): Annual N2O emissions
                annual_distance_km (Decimal): Total annual passenger-km
                daily_distance_km (Decimal): Daily passenger-km
                transit_type (str): Normalized transit type key
                ef_used (Dict): Emission factor values used
                ef_source (str): Emission factor source label
                method (str): "distance_based_transit"
                working_days (int): Working days used
                wfh_fraction (Decimal): WFH fraction used
                round_trip (bool): Whether round-trip was applied
                trips_per_day (int): Trips per day used
                include_wtt (bool): Whether WTT was included
                provenance_hash (str): SHA-256 provenance hash

        Raises:
            ValueError: If transit_type is unknown, distance <= 0,
                working_days <= 0, wfh_fraction not in [0, 1],
                or trips_per_day < 1.

        Example:
            >>> result = engine.calculate_transit_commute(
            ...     transit_type="commuter_rail",
            ...     one_way_distance_km=Decimal("25"),
            ...     working_days=225,
            ...     wfh_fraction=Decimal("0.20"),
            ... )
            >>> assert result["total_co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate and normalize inputs
            transit_type_norm = self._validate_transit_type(transit_type)
            distance = _to_decimal(one_way_distance_km, "one_way_distance_km")
            wfh = _to_decimal(wfh_fraction, "wfh_fraction")
            errors = self._validate_inputs({
                "one_way_distance_km": distance,
                "working_days": working_days,
                "wfh_fraction": wfh,
                "trips_per_day": trips_per_day,
            })
            if errors:
                raise ValueError(
                    f"Input validation failed: {'; '.join(errors)}"
                )

            # Step 2: Resolve emission factor
            ef = self._get_ef(transit_type_norm)

            # Step 3: Calculate distances (ZERO HALLUCINATION - Decimal only)
            multiplier = _TWO if round_trip else _ONE
            trips = Decimal(str(trips_per_day))
            daily_distance = _q(distance * multiplier * trips)
            commute_fraction = _q(_ONE - wfh)
            annual_distance = _q(
                daily_distance * Decimal(str(working_days)) * commute_fraction
            )

            # Step 4: Calculate emissions (ZERO HALLUCINATION - Decimal only)
            ttw_co2e = _q(annual_distance * ef["co2e_per_pkm"])
            wtt_co2e = (
                _q(annual_distance * ef["wtt_per_pkm"]) if include_wtt else _ZERO
            )
            total_co2e = _q(ttw_co2e + wtt_co2e)

            # Per-gas breakdown (tank-to-wheel only)
            co2 = _q(annual_distance * ef["co2_per_pkm"])
            ch4 = _q(annual_distance * ef["ch4_per_pkm"])
            n2o = _q(annual_distance * ef["n2o_per_pkm"])

            # Step 5: Build EF summary for provenance
            ef_used = {
                "co2e_per_pkm": str(ef["co2e_per_pkm"]),
                "co2_per_pkm": str(ef["co2_per_pkm"]),
                "ch4_per_pkm": str(ef["ch4_per_pkm"]),
                "n2o_per_pkm": str(ef["n2o_per_pkm"]),
                "wtt_per_pkm": str(ef["wtt_per_pkm"]),
            }

            # Step 6: Compute provenance hash
            provenance_hash = self._calculate_provenance_hash(
                {
                    "transit_type": transit_type_norm,
                    "one_way_distance_km": str(distance),
                    "working_days": working_days,
                    "wfh_fraction": str(wfh),
                    "round_trip": round_trip,
                    "include_wtt": include_wtt,
                    "trips_per_day": trips_per_day,
                },
                {
                    "total_co2e_kg": str(total_co2e),
                    "ttw_co2e_kg": str(ttw_co2e),
                    "wtt_co2e_kg": str(wtt_co2e),
                    "co2_kg": str(co2),
                    "ch4_kg": str(ch4),
                    "n2o_kg": str(n2o),
                    "annual_distance_km": str(annual_distance),
                },
            )

            # Step 7: Build result
            result: Dict[str, Any] = {
                "co2e_kg": total_co2e,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "co2_kg": co2,
                "ch4_kg": ch4,
                "n2o_kg": n2o,
                "annual_distance_km": annual_distance,
                "daily_distance_km": daily_distance,
                "transit_type": transit_type_norm,
                "ef_used": ef_used,
                "ef_source": ef["source"],
                "method": "distance_based_transit",
                "working_days": working_days,
                "wfh_fraction": wfh,
                "round_trip": round_trip,
                "trips_per_day": trips_per_day,
                "include_wtt": include_wtt,
                "provenance_hash": provenance_hash,
            }

            # Step 8: Record metrics
            duration = time.monotonic() - start_time
            self._record_metrics(
                mode=transit_type_norm,
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "Transit commute calculation complete: type=%s, dist=%s km/day, "
                "annual=%s km, ttw=%s kg, wtt=%s kg, total=%s kg, hash=%s",
                transit_type_norm,
                daily_distance,
                annual_distance,
                ttw_co2e,
                wtt_co2e,
                total_co2e,
                provenance_hash[:16],
            )

            return result

    # ==================================================================
    # 2. calculate_multi_modal_transit - Multi-segment transit
    # ==================================================================

    def calculate_multi_modal_transit(
        self,
        segments: List[Dict[str, Any]],
        working_days: int = _DEFAULT_WORKING_DAYS,
        wfh_fraction: Union[Decimal, str, int, float] = _ZERO,
        round_trip: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a multi-modal transit commute.

        Processes each segment independently then aggregates totals.
        Each segment represents one leg of the commute (e.g., bus to
        subway station, then subway to office).

        Args:
            segments: List of segment dicts, each containing:
                - transit_type (str): Transit mode for this segment
                - distance_km (Decimal|str|int|float): Segment distance in km
                - include_wtt (bool, optional): Include WTT. Default True.
            working_days: Annual commuting days. Default 240.
            wfh_fraction: Fraction of days working from home. Default 0.
            round_trip: If True, double each segment distance. Default True.

        Returns:
            Dict with keys:
                total_co2e_kg (Decimal): Aggregate total CO2e
                total_ttw_co2e_kg (Decimal): Aggregate TTW CO2e
                total_wtt_co2e_kg (Decimal): Aggregate WTT CO2e
                total_co2_kg (Decimal): Aggregate CO2
                total_ch4_kg (Decimal): Aggregate CH4
                total_n2o_kg (Decimal): Aggregate N2O
                total_distance_km (Decimal): Total annual distance
                segment_count (int): Number of segments
                segments (List[Dict]): Per-segment results
                working_days (int): Working days used
                wfh_fraction (Decimal): WFH fraction used
                round_trip (bool): Whether round-trip was applied
                method (str): "multi_modal_transit"
                provenance_hash (str): SHA-256 hash of aggregate result

        Raises:
            ValueError: If segments is empty, or any segment is invalid.

        Example:
            >>> result = engine.calculate_multi_modal_transit(
            ...     segments=[
            ...         {"transit_type": "local_bus", "distance_km": Decimal("3")},
            ...         {"transit_type": "subway_metro", "distance_km": Decimal("8")},
            ...     ],
            ...     working_days=225,
            ... )
            >>> assert result["segment_count"] == 2
        """
        start_time = time.monotonic()

        with self._lock:
            # Validate segments list
            if not segments:
                raise ValueError("segments list must not be empty")
            if len(segments) > 50:
                raise ValueError(
                    f"segments count {len(segments)} exceeds maximum 50"
                )

            wfh = _to_decimal(wfh_fraction, "wfh_fraction")

            # Process each segment
            segment_results: List[Dict[str, Any]] = []
            agg_co2e = _ZERO
            agg_ttw = _ZERO
            agg_wtt = _ZERO
            agg_co2 = _ZERO
            agg_ch4 = _ZERO
            agg_n2o = _ZERO
            agg_distance = _ZERO

            for idx, seg in enumerate(segments):
                seg_type = seg.get("transit_type")
                seg_dist = seg.get("distance_km")
                seg_wtt = seg.get("include_wtt", True)

                if seg_type is None:
                    raise ValueError(
                        f"Segment {idx}: missing 'transit_type'"
                    )
                if seg_dist is None:
                    raise ValueError(
                        f"Segment {idx}: missing 'distance_km'"
                    )

                seg_result = self.calculate_transit_commute(
                    transit_type=seg_type,
                    one_way_distance_km=seg_dist,
                    working_days=working_days,
                    wfh_fraction=wfh,
                    round_trip=round_trip,
                    include_wtt=seg_wtt,
                    trips_per_day=1,
                )

                segment_results.append({
                    "segment_index": idx,
                    "transit_type": seg_result["transit_type"],
                    "segment_distance_km": _to_decimal(seg_dist, "distance_km"),
                    "annual_distance_km": seg_result["annual_distance_km"],
                    "co2e_kg": seg_result["co2e_kg"],
                    "ttw_co2e_kg": seg_result["ttw_co2e_kg"],
                    "wtt_co2e_kg": seg_result["wtt_co2e_kg"],
                    "co2_kg": seg_result["co2_kg"],
                    "ch4_kg": seg_result["ch4_kg"],
                    "n2o_kg": seg_result["n2o_kg"],
                    "ef_source": seg_result["ef_source"],
                    "provenance_hash": seg_result["provenance_hash"],
                })

                agg_co2e = _q(agg_co2e + seg_result["co2e_kg"])
                agg_ttw = _q(agg_ttw + seg_result["ttw_co2e_kg"])
                agg_wtt = _q(agg_wtt + seg_result["wtt_co2e_kg"])
                agg_co2 = _q(agg_co2 + seg_result["co2_kg"])
                agg_ch4 = _q(agg_ch4 + seg_result["ch4_kg"])
                agg_n2o = _q(agg_n2o + seg_result["n2o_kg"])
                agg_distance = _q(agg_distance + seg_result["annual_distance_km"])

            # Provenance hash for aggregate
            provenance_hash = self._calculate_provenance_hash(
                {
                    "method": "multi_modal_transit",
                    "segment_count": len(segments),
                    "working_days": working_days,
                    "wfh_fraction": str(wfh),
                    "round_trip": round_trip,
                    "segment_hashes": [
                        s["provenance_hash"] for s in segment_results
                    ],
                },
                {
                    "total_co2e_kg": str(agg_co2e),
                    "total_distance_km": str(agg_distance),
                },
            )

            result: Dict[str, Any] = {
                "total_co2e_kg": agg_co2e,
                "total_ttw_co2e_kg": agg_ttw,
                "total_wtt_co2e_kg": agg_wtt,
                "total_co2_kg": agg_co2,
                "total_ch4_kg": agg_ch4,
                "total_n2o_kg": agg_n2o,
                "total_distance_km": agg_distance,
                "segment_count": len(segment_results),
                "segments": segment_results,
                "working_days": working_days,
                "wfh_fraction": wfh,
                "round_trip": round_trip,
                "method": "multi_modal_transit",
                "provenance_hash": provenance_hash,
            }

            duration = time.monotonic() - start_time
            logger.info(
                "Multi-modal transit calculation complete: segments=%d, "
                "total_co2e=%s kg, total_dist=%s km, duration=%.4fs",
                len(segment_results),
                agg_co2e,
                agg_distance,
                duration,
            )

            return result

    # ==================================================================
    # 3. calculate_bus_commute - Convenience method for bus
    # ==================================================================

    def calculate_bus_commute(
        self,
        one_way_distance_km: Union[Decimal, str, int, float],
        bus_type: str = "local_bus",
        working_days: int = _DEFAULT_WORKING_DAYS,
        wfh_fraction: Union[Decimal, str, int, float] = _ZERO,
        round_trip: bool = True,
        include_wtt: bool = True,
        trips_per_day: int = 1,
    ) -> Dict[str, Any]:
        """
        Calculate bus commute emissions (convenience wrapper).

        Delegates to calculate_transit_commute with bus-specific defaults.

        Args:
            one_way_distance_km: One-way commute distance in km.
            bus_type: Bus type key. One of "local_bus", "express_bus",
                "coach". Default "local_bus".
            working_days: Annual commuting days. Default 240.
            wfh_fraction: WFH fraction [0, 1]. Default 0.
            round_trip: If True, double distance. Default True.
            include_wtt: If True, include WTT. Default True.
            trips_per_day: Number of bus trips per day. Default 1.

        Returns:
            Dict with same structure as calculate_transit_commute.

        Raises:
            ValueError: If bus_type is not a valid bus transit type, or
                if other input parameters are invalid.

        Example:
            >>> result = engine.calculate_bus_commute(
            ...     one_way_distance_km=Decimal("5.5"),
            ...     bus_type="local_bus",
            ...     working_days=225,
            ... )
            >>> assert result["transit_type"] == "local_bus"
        """
        normalized = self._validate_transit_type(bus_type)
        if normalized not in _BUS_TYPES:
            raise ValueError(
                f"bus_type '{bus_type}' is not a bus type. "
                f"Valid bus types: {sorted(_BUS_TYPES)}"
            )

        return self.calculate_transit_commute(
            transit_type=normalized,
            one_way_distance_km=one_way_distance_km,
            working_days=working_days,
            wfh_fraction=wfh_fraction,
            round_trip=round_trip,
            include_wtt=include_wtt,
            trips_per_day=trips_per_day,
        )

    # ==================================================================
    # 4. calculate_rail_commute - Convenience method for rail
    # ==================================================================

    def calculate_rail_commute(
        self,
        one_way_distance_km: Union[Decimal, str, int, float],
        rail_type: str = "commuter_rail",
        working_days: int = _DEFAULT_WORKING_DAYS,
        wfh_fraction: Union[Decimal, str, int, float] = _ZERO,
        round_trip: bool = True,
        include_wtt: bool = True,
        trips_per_day: int = 1,
    ) -> Dict[str, Any]:
        """
        Calculate rail commute emissions (convenience wrapper).

        Delegates to calculate_transit_commute with rail-specific defaults.

        Args:
            one_way_distance_km: One-way commute distance in km.
            rail_type: Rail type key. One of "commuter_rail",
                "subway_metro", "light_rail", "tram_streetcar".
                Default "commuter_rail".
            working_days: Annual commuting days. Default 240.
            wfh_fraction: WFH fraction [0, 1]. Default 0.
            round_trip: If True, double distance. Default True.
            include_wtt: If True, include WTT. Default True.
            trips_per_day: Number of rail trips per day. Default 1.

        Returns:
            Dict with same structure as calculate_transit_commute.

        Raises:
            ValueError: If rail_type is not a valid rail transit type, or
                if other input parameters are invalid.

        Example:
            >>> result = engine.calculate_rail_commute(
            ...     one_way_distance_km=Decimal("18.0"),
            ...     rail_type="subway_metro",
            ...     working_days=212,
            ... )
            >>> assert result["transit_type"] == "subway_metro"
        """
        normalized = self._validate_transit_type(rail_type)
        if normalized not in _RAIL_TYPES:
            raise ValueError(
                f"rail_type '{rail_type}' is not a rail type. "
                f"Valid rail types: {sorted(_RAIL_TYPES)}"
            )

        return self.calculate_transit_commute(
            transit_type=normalized,
            one_way_distance_km=one_way_distance_km,
            working_days=working_days,
            wfh_fraction=wfh_fraction,
            round_trip=round_trip,
            include_wtt=include_wtt,
            trips_per_day=trips_per_day,
        )

    # ==================================================================
    # 5. calculate_ferry_commute - Convenience method for ferry
    # ==================================================================

    def calculate_ferry_commute(
        self,
        one_way_distance_km: Union[Decimal, str, int, float],
        ferry_type: str = "ferry_boat",
        working_days: int = _DEFAULT_WORKING_DAYS,
        wfh_fraction: Union[Decimal, str, int, float] = _ZERO,
        round_trip: bool = True,
        include_wtt: bool = True,
        trips_per_day: int = 1,
    ) -> Dict[str, Any]:
        """
        Calculate ferry commute emissions (convenience wrapper).

        Delegates to calculate_transit_commute with ferry-specific defaults.

        Args:
            one_way_distance_km: One-way commute distance in km.
            ferry_type: Ferry type key. One of "ferry_boat", "water_taxi".
                Default "ferry_boat".
            working_days: Annual commuting days. Default 240.
            wfh_fraction: WFH fraction [0, 1]. Default 0.
            round_trip: If True, double distance. Default True.
            include_wtt: If True, include WTT. Default True.
            trips_per_day: Number of ferry trips per day. Default 1.

        Returns:
            Dict with same structure as calculate_transit_commute.

        Raises:
            ValueError: If ferry_type is not a valid ferry transit type, or
                if other input parameters are invalid.

        Example:
            >>> result = engine.calculate_ferry_commute(
            ...     one_way_distance_km=Decimal("6.0"),
            ...     ferry_type="water_taxi",
            ...     working_days=225,
            ... )
            >>> assert result["transit_type"] == "water_taxi"
        """
        normalized = self._validate_transit_type(ferry_type)
        if normalized not in _FERRY_TYPES:
            raise ValueError(
                f"ferry_type '{ferry_type}' is not a ferry type. "
                f"Valid ferry types: {sorted(_FERRY_TYPES)}"
            )

        return self.calculate_transit_commute(
            transit_type=normalized,
            one_way_distance_km=one_way_distance_km,
            working_days=working_days,
            wfh_fraction=wfh_fraction,
            round_trip=round_trip,
            include_wtt=include_wtt,
            trips_per_day=trips_per_day,
        )

    # ==================================================================
    # 6. calculate_batch - Batch processing for multiple items
    # ==================================================================

    def calculate_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process multiple transit commute calculations in a single batch.

        Each item dict must contain at least 'transit_type' and
        'one_way_distance_km'. All other parameters from
        calculate_transit_commute are optional and use defaults if
        not specified.

        Failed calculations do not halt the batch. Each item result
        includes a 'status' field ("success" or "error").

        Args:
            items: List of dicts, each containing transit commute
                parameters. Required keys: transit_type, one_way_distance_km.
                Optional keys: working_days, wfh_fraction, round_trip,
                include_wtt, trips_per_day.

        Returns:
            Dict with keys:
                total_items (int): Number of items in batch
                success_count (int): Number of successful calculations
                error_count (int): Number of failed calculations
                total_co2e_kg (Decimal): Aggregate CO2e of successful items
                total_distance_km (Decimal): Aggregate distance of successes
                results (List[Dict]): Per-item results
                processing_time_ms (float): Total batch processing time
                provenance_hash (str): SHA-256 hash of batch result

        Raises:
            ValueError: If items list exceeds _MAX_BATCH_SIZE or is empty.

        Example:
            >>> batch = engine.calculate_batch([
            ...     {"transit_type": "subway_metro",
            ...      "one_way_distance_km": "12"},
            ...     {"transit_type": "local_bus",
            ...      "one_way_distance_km": "5.5"},
            ... ])
            >>> assert batch["success_count"] == 2
        """
        if not items:
            raise ValueError("items list must not be empty")
        if len(items) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        agg_co2e = _ZERO
        agg_distance = _ZERO
        success_count = 0
        error_count = 0

        for idx, item in enumerate(items):
            try:
                calc_result = self.calculate_transit_commute(
                    transit_type=item.get("transit_type", ""),
                    one_way_distance_km=item.get("one_way_distance_km", _ZERO),
                    working_days=int(item.get("working_days", _DEFAULT_WORKING_DAYS)),
                    wfh_fraction=item.get("wfh_fraction", _ZERO),
                    round_trip=item.get("round_trip", True),
                    include_wtt=item.get("include_wtt", True),
                    trips_per_day=int(item.get("trips_per_day", 1)),
                )
                results.append({
                    "index": idx,
                    "status": "success",
                    "result": calc_result,
                })
                agg_co2e = _q(agg_co2e + calc_result["co2e_kg"])
                agg_distance = _q(
                    agg_distance + calc_result["annual_distance_km"]
                )
                success_count += 1

            except Exception as exc:
                logger.warning(
                    "Batch item %d failed: %s", idx, str(exc),
                )
                results.append({
                    "index": idx,
                    "status": "error",
                    "error": str(exc),
                })
                error_count += 1

        duration_ms = (time.monotonic() - start_time) * 1000

        # Provenance hash for batch
        provenance_hash = self._calculate_provenance_hash(
            {
                "method": "batch_transit",
                "total_items": len(items),
                "success_count": success_count,
            },
            {
                "total_co2e_kg": str(agg_co2e),
                "total_distance_km": str(agg_distance),
                "error_count": error_count,
            },
        )

        batch_result: Dict[str, Any] = {
            "total_items": len(items),
            "success_count": success_count,
            "error_count": error_count,
            "total_co2e_kg": agg_co2e,
            "total_distance_km": agg_distance,
            "results": results,
            "processing_time_ms": round(duration_ms, 4),
            "provenance_hash": provenance_hash,
        }

        logger.info(
            "Batch transit calculation complete: total=%d, success=%d, "
            "errors=%d, co2e=%s kg, duration=%.2fms",
            len(items),
            success_count,
            error_count,
            agg_co2e,
            duration_ms,
        )

        return batch_result

    # ==================================================================
    # 7. estimate_annual_emissions - Quick estimate with defaults
    # ==================================================================

    def estimate_annual_emissions(
        self,
        transit_type: str,
        one_way_distance_km: Union[Decimal, str, int, float],
        working_days: int = _DEFAULT_WORKING_DAYS,
    ) -> Dict[str, Any]:
        """
        Quick estimate of annual transit commute emissions with defaults.

        Uses sensible defaults: round-trip, WTT included, no WFH,
        1 trip per day. Intended for screening-level estimates and
        dashboard widgets.

        Args:
            transit_type: Transit type key.
            one_way_distance_km: One-way commute distance in km.
            working_days: Annual commuting days. Default 240.

        Returns:
            Dict with keys:
                transit_type (str): Normalized transit type
                one_way_distance_km (Decimal): Input distance
                annual_co2e_kg (Decimal): Total annual CO2e
                annual_distance_km (Decimal): Annual passenger-km
                co2e_per_day_kg (Decimal): Average daily CO2e
                co2e_per_trip_kg (Decimal): CO2e per one-way trip
                ef_source (str): Emission factor source label
                provenance_hash (str): SHA-256 hash

        Raises:
            ValueError: If transit_type is unknown or distance <= 0.

        Example:
            >>> est = engine.estimate_annual_emissions(
            ...     "subway_metro", Decimal("10"), 225
            ... )
            >>> print(f"Annual CO2e: {est['annual_co2e_kg']} kg")
        """
        result = self.calculate_transit_commute(
            transit_type=transit_type,
            one_way_distance_km=one_way_distance_km,
            working_days=working_days,
            wfh_fraction=_ZERO,
            round_trip=True,
            include_wtt=True,
            trips_per_day=1,
        )

        distance = _to_decimal(one_way_distance_km, "one_way_distance_km")
        wd_decimal = Decimal(str(working_days))
        co2e_per_day = (
            _q(result["co2e_kg"] / wd_decimal) if working_days > 0
            else _ZERO
        )
        co2e_per_trip = (
            _q(result["co2e_kg"] / (wd_decimal * _TWO)) if working_days > 0
            else _ZERO
        )

        return {
            "transit_type": result["transit_type"],
            "one_way_distance_km": distance,
            "annual_co2e_kg": result["co2e_kg"],
            "annual_distance_km": result["annual_distance_km"],
            "co2e_per_day_kg": co2e_per_day,
            "co2e_per_trip_kg": co2e_per_trip,
            "ef_source": result["ef_source"],
            "provenance_hash": result["provenance_hash"],
        }

    # ==================================================================
    # 8. compare_transit_modes - Rank modes by emissions
    # ==================================================================

    def compare_transit_modes(
        self,
        one_way_distance_km: Union[Decimal, str, int, float],
        working_days: int = _DEFAULT_WORKING_DAYS,
    ) -> Dict[str, Any]:
        """
        Compare emissions for all transit types for the same commute route.

        Calculates annual emissions for each transit type at the same
        distance and working days, then returns a ranked list from
        lowest to highest total CO2e.

        Args:
            one_way_distance_km: One-way commute distance in km.
            working_days: Annual commuting days. Default 240.

        Returns:
            Dict with keys:
                one_way_distance_km (Decimal): Input distance
                working_days (int): Working days used
                rankings (List[Dict]): Ranked list from lowest to highest
                    CO2e. Each dict has transit_type, annual_co2e_kg,
                    annual_distance_km, ef_source, rank.
                lowest_emission (Dict): The transit type with lowest CO2e
                highest_emission (Dict): The transit type with highest CO2e
                savings_vs_highest_kg (Decimal): CO2e savings of lowest
                    vs highest
                provenance_hash (str): SHA-256 hash

        Raises:
            ValueError: If distance <= 0 or working_days <= 0.

        Example:
            >>> comparison = engine.compare_transit_modes(
            ...     one_way_distance_km=Decimal("15"), working_days=225
            ... )
            >>> print(comparison["lowest_emission"]["transit_type"])
        """
        distance = _to_decimal(one_way_distance_km, "one_way_distance_km")
        if distance <= _ZERO:
            raise ValueError(
                f"one_way_distance_km must be positive, got {distance}"
            )
        if working_days <= 0:
            raise ValueError(
                f"working_days must be positive, got {working_days}"
            )

        mode_results: List[Dict[str, Any]] = []
        for transit_type in sorted(TRANSIT_EFS.keys()):
            try:
                result = self.calculate_transit_commute(
                    transit_type=transit_type,
                    one_way_distance_km=distance,
                    working_days=working_days,
                    wfh_fraction=_ZERO,
                    round_trip=True,
                    include_wtt=True,
                    trips_per_day=1,
                )
                mode_results.append({
                    "transit_type": transit_type,
                    "annual_co2e_kg": result["co2e_kg"],
                    "annual_distance_km": result["annual_distance_km"],
                    "ef_source": result["ef_source"],
                })
            except Exception as exc:
                logger.warning(
                    "Failed to calculate comparison for %s: %s",
                    transit_type,
                    str(exc),
                )

        # Sort by annual_co2e_kg ascending (lowest first)
        mode_results.sort(key=lambda x: x["annual_co2e_kg"])

        # Add rank
        for rank_idx, entry in enumerate(mode_results, start=1):
            entry["rank"] = rank_idx

        lowest = mode_results[0] if mode_results else {}
        highest = mode_results[-1] if mode_results else {}

        savings = _ZERO
        if lowest and highest:
            savings = _q(
                highest["annual_co2e_kg"] - lowest["annual_co2e_kg"]
            )

        provenance_hash = self._calculate_provenance_hash(
            {
                "method": "compare_transit_modes",
                "one_way_distance_km": str(distance),
                "working_days": working_days,
                "mode_count": len(mode_results),
            },
            {
                "lowest_type": lowest.get("transit_type", ""),
                "lowest_co2e": str(lowest.get("annual_co2e_kg", _ZERO)),
                "highest_type": highest.get("transit_type", ""),
                "highest_co2e": str(highest.get("annual_co2e_kg", _ZERO)),
            },
        )

        return {
            "one_way_distance_km": distance,
            "working_days": working_days,
            "rankings": mode_results,
            "lowest_emission": lowest,
            "highest_emission": highest,
            "savings_vs_highest_kg": savings,
            "provenance_hash": provenance_hash,
        }

    # ==================================================================
    # HELPER: _get_ef - Retrieve emission factor for transit type
    # ==================================================================

    def _get_ef(self, transit_type: str) -> Dict[str, Any]:
        """
        Get emission factor dict for a validated transit type.

        Args:
            transit_type: Normalized transit type key (must be pre-validated).

        Returns:
            Dict with co2e_per_pkm, co2_per_pkm, ch4_per_pkm,
            n2o_per_pkm, wtt_per_pkm, source.

        Raises:
            KeyError: If transit_type not found in TRANSIT_EFS.
        """
        ef = TRANSIT_EFS.get(transit_type)
        if ef is None:
            raise KeyError(
                f"Emission factor not found for transit_type '{transit_type}'. "
                f"Available: {sorted(TRANSIT_EFS.keys())}"
            )
        logger.debug(
            "Resolved transit EF: type=%s, co2e=%s, wtt=%s, source=%s",
            transit_type,
            ef["co2e_per_pkm"],
            ef["wtt_per_pkm"],
            ef["source"],
        )
        return ef

    # ==================================================================
    # HELPER: _validate_transit_type - Normalize and validate type key
    # ==================================================================

    def _validate_transit_type(self, transit_type: str) -> str:
        """
        Validate and normalize a transit type key.

        Normalizes by lowercasing, stripping whitespace, and replacing
        hyphens/spaces with underscores. Then checks that the normalized
        key exists in TRANSIT_EFS.

        Args:
            transit_type: Raw transit type string.

        Returns:
            Normalized transit type key.

        Raises:
            ValueError: If transit_type is empty or not found in TRANSIT_EFS
                after normalization.
        """
        if not transit_type or not isinstance(transit_type, str):
            raise ValueError(
                "transit_type must be a non-empty string"
            )

        normalized = transit_type.strip().lower().replace("-", "_").replace(" ", "_")

        if normalized not in TRANSIT_EFS:
            raise ValueError(
                f"Unknown transit_type '{transit_type}' (normalized: "
                f"'{normalized}'). Available: {sorted(TRANSIT_EFS.keys())}"
            )

        return normalized

    # ==================================================================
    # HELPER: _validate_inputs - Validate numeric parameters
    # ==================================================================

    def _validate_inputs(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate numeric input parameters.

        Checks:
            - one_way_distance_km > 0
            - working_days > 0
            - wfh_fraction in [0, 1]
            - trips_per_day >= 1

        Args:
            params: Dict with parameter names as keys and values.

        Returns:
            List of error messages. Empty list means all valid.
        """
        errors: List[str] = []

        distance = params.get("one_way_distance_km")
        if distance is not None and distance <= _ZERO:
            errors.append(
                f"one_way_distance_km must be positive, got {distance}"
            )

        working_days = params.get("working_days")
        if working_days is not None and working_days <= 0:
            errors.append(
                f"working_days must be positive, got {working_days}"
            )

        wfh = params.get("wfh_fraction")
        if wfh is not None:
            if wfh < _ZERO or wfh > _ONE:
                errors.append(
                    f"wfh_fraction must be between 0 and 1, got {wfh}"
                )

        trips = params.get("trips_per_day")
        if trips is not None and trips < 1:
            errors.append(
                f"trips_per_day must be >= 1, got {trips}"
            )

        return errors

    # ==================================================================
    # HELPER: _calculate_provenance_hash - SHA-256 hash
    # ==================================================================

    def _calculate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> str:
        """
        Calculate SHA-256 provenance hash for input+output pair.

        Creates a deterministic hash from the combined input and result
        data, ensuring reproducibility and audit trail integrity.

        Args:
            input_data: Calculation input parameters.
            result_data: Calculation result values.

        Returns:
            64-character lowercase hex SHA-256 hash.
        """
        combined = {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "input": input_data,
            "output": result_data,
        }
        serialized = _serialize_for_hash(combined)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ==================================================================
    # ACCESSOR: get_supported_transit_types
    # ==================================================================

    def get_supported_transit_types(self) -> List[str]:
        """
        Get list of all supported transit type keys.

        Returns:
            Sorted list of transit type key strings.

        Example:
            >>> types = engine.get_supported_transit_types()
            >>> assert "subway_metro" in types
        """
        return sorted(TRANSIT_EFS.keys())

    # ==================================================================
    # ACCESSOR: get_emission_factor_summary
    # ==================================================================

    def get_emission_factor_summary(self) -> Dict[str, Dict[str, str]]:
        """
        Get a read-only summary of all transit emission factors.

        Returns a dict keyed by transit type with each EF value
        converted to string for safe serialization.

        Returns:
            Dict mapping transit type to its emission factor summary.

        Example:
            >>> summary = engine.get_emission_factor_summary()
            >>> assert "local_bus" in summary
            >>> assert "co2e_per_pkm" in summary["local_bus"]
        """
        summary: Dict[str, Dict[str, str]] = {}
        for transit_type, ef in TRANSIT_EFS.items():
            summary[transit_type] = {
                "co2e_per_pkm": str(ef["co2e_per_pkm"]),
                "co2_per_pkm": str(ef["co2_per_pkm"]),
                "ch4_per_pkm": str(ef["ch4_per_pkm"]),
                "n2o_per_pkm": str(ef["n2o_per_pkm"]),
                "wtt_per_pkm": str(ef["wtt_per_pkm"]),
                "source": ef["source"],
            }
        return summary

    # ==================================================================
    # ACCESSOR: get_bus_types
    # ==================================================================

    def get_bus_types(self) -> List[str]:
        """
        Get list of bus transit type keys.

        Returns:
            Sorted list of bus type key strings.

        Example:
            >>> bus_types = engine.get_bus_types()
            >>> assert "local_bus" in bus_types
        """
        return sorted(_BUS_TYPES)

    # ==================================================================
    # ACCESSOR: get_rail_types
    # ==================================================================

    def get_rail_types(self) -> List[str]:
        """
        Get list of rail transit type keys.

        Returns:
            Sorted list of rail type key strings.

        Example:
            >>> rail_types = engine.get_rail_types()
            >>> assert "commuter_rail" in rail_types
        """
        return sorted(_RAIL_TYPES)

    # ==================================================================
    # ACCESSOR: get_ferry_types
    # ==================================================================

    def get_ferry_types(self) -> List[str]:
        """
        Get list of ferry transit type keys.

        Returns:
            Sorted list of ferry type key strings.

        Example:
            >>> ferry_types = engine.get_ferry_types()
            >>> assert "ferry_boat" in ferry_types
        """
        return sorted(_FERRY_TYPES)

    # ==================================================================
    # ACCESSOR: get_engine_info
    # ==================================================================

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get engine metadata and status information.

        Returns:
            Dict with engine_id, engine_version, agent_id,
            calculation_count, transit_type_count, and timestamp.

        Example:
            >>> info = engine.get_engine_info()
            >>> assert info["engine_id"] == "public_transit_calculator_engine"
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "calculation_count": self._calculation_count,
            "transit_type_count": len(TRANSIT_EFS),
            "bus_type_count": len(_BUS_TYPES),
            "rail_type_count": len(_RAIL_TYPES),
            "ferry_type_count": len(_FERRY_TYPES),
            "max_batch_size": _MAX_BATCH_SIZE,
            "default_working_days": _DEFAULT_WORKING_DAYS,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ==================================================================
    # Internal: Metrics Recording
    # ==================================================================

    def _record_metrics(
        self,
        mode: str,
        co2e: float,
        duration: float,
    ) -> None:
        """
        Record transit calculation metrics to Prometheus.

        Wraps calls to the EmployeeCommutingMetrics singleton to record
        calculation throughput, emissions, and duration. All calls are
        wrapped in try/except to prevent metrics failures from
        disrupting calculations.

        Args:
            mode: Transit mode label (e.g., "subway_metro").
            co2e: Emissions in kgCO2e for emissions counter.
            duration: Calculation duration in seconds.
        """
        try:
            self._metrics.record_calculation(
                method="distance_based",
                mode=mode,
                status="success",
                duration=duration,
                co2e=co2e,
                category="transport",
            )
            self._metrics.record_commute(
                mode=mode,
                vehicle_type="public_transit",
            )
        except Exception as exc:
            logger.warning(
                "Failed to record transit metrics: %s", str(exc)
            )


# ==============================================================================
# MODULE-LEVEL SINGLETON ACCESSORS
# ==============================================================================


def get_public_transit_calculator() -> PublicTransitCalculatorEngine:
    """
    Get the singleton PublicTransitCalculatorEngine instance.

    Thread-safe singleton pattern. Returns the same engine instance
    across all callers in the process.

    Returns:
        PublicTransitCalculatorEngine singleton instance.

    Example:
        >>> engine = get_public_transit_calculator()
        >>> result = engine.calculate_transit_commute(
        ...     "subway_metro", Decimal("10"), 225,
        ... )
    """
    return PublicTransitCalculatorEngine.get_instance()


def reset_public_transit_calculator() -> None:
    """
    Reset the singleton PublicTransitCalculatorEngine instance.

    Intended exclusively for unit tests. Creates a fresh engine
    instance on the next call to get_public_transit_calculator().

    Example:
        >>> reset_public_transit_calculator()
        >>> engine = get_public_transit_calculator()  # Fresh instance
    """
    PublicTransitCalculatorEngine.reset_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Engine
    "PublicTransitCalculatorEngine",
    # Singleton accessors
    "get_public_transit_calculator",
    "reset_public_transit_calculator",
    # Constants
    "ENGINE_ID",
    "ENGINE_VERSION",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "TRANSIT_EFS",
]
