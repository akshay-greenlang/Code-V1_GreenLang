# -*- coding: utf-8 -*-
"""
HotelStayCalculatorEngine - Engine 4: Business Travel Agent (AGENT-MRV-019)

Core calculation engine for hotel accommodation emissions by country and
hotel class, following DEFRA 2024 and Cornell HCMI room-night emission
factors.

This engine implements deterministic Decimal-based emissions calculations
for hotel stays during business travel, as defined in GHG Protocol Scope 3
Category 6.

Primary Formula:
    base_co2e     = room_nights x ef_per_room_night
    class_co2e    = base_co2e x class_multiplier
    discount      = 0.85 if room_nights >= extended_stay_threshold else 1.0
    total_co2e    = class_co2e x discount

Hotel EFs embed well-to-tank (WTT) emissions within the room-night factor
since hotel energy consumption is a blend of electricity (grid), heating
fuel, and cooling. DEFRA 2024 factors already include upstream energy
production emissions.

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate value is deterministic and reproducible
    - SHA-256 provenance hash on every result
    - Emission factors sourced from DEFRA 2024 / Cornell HCMI

Supports:
    - 16 country-specific room-night EFs (GB, US, CA, FR, DE, ES, IT, NL,
      JP, CN, IN, AU, BR, SG, AE, GLOBAL fallback)
    - 4 hotel classes (budget, standard, upscale, luxury) with multipliers
    - Extended stay discount (configurable threshold, default 14 nights)
    - Batch processing for multiple hotel inputs
    - Input validation with detailed error messages
    - Provenance hash integration for audit trails
    - Prometheus metrics integration (hotel_nights_total counter)

Example:
    >>> from greenlang.business_travel.hotel_stay_calculator import (
    ...     HotelStayCalculatorEngine,
    ... )
    >>> from greenlang.business_travel.models import (
    ...     HotelInput, HotelClass,
    ... )
    >>> engine = HotelStayCalculatorEngine.get_instance()
    >>> hotel_input = HotelInput(
    ...     country_code="GB",
    ...     room_nights=3,
    ...     hotel_class=HotelClass.UPSCALE,
    ... )
    >>> result = engine.calculate(hotel_input)
    >>> assert result.total_co2e > Decimal("0")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-019 Business Travel (GL-MRV-S3-006)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional

from greenlang.business_travel.models import (
    AGENT_COMPONENT,
    AGENT_ID,
    VERSION,
    HotelInput,
    HotelResult,
    HotelClass,
    EFSource,
    HOTEL_EMISSION_FACTORS,
    HOTEL_CLASS_MULTIPLIERS,
    calculate_provenance_hash,
)
from greenlang.business_travel.metrics import BusinessTravelMetrics, get_metrics
from greenlang.business_travel.config import get_config
from greenlang.business_travel.provenance import get_provenance_tracker

logger = logging.getLogger(__name__)

# ==============================================================================
# DECIMAL PRECISION & CONSTANTS
# ==============================================================================

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")

# Extended stay discount parameters
_DEFAULT_EXTENDED_STAY_THRESHOLD = 14  # nights
_DEFAULT_EXTENDED_STAY_DISCOUNT = Decimal("0.85")  # 15% discount

# Batch processing limits
_MAX_BATCH_SIZE = 10000

# ==============================================================================
# SINGLETON INSTANCE MANAGEMENT
# ==============================================================================

_instance: Optional["HotelStayCalculatorEngine"] = None
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


# ==============================================================================
# HotelStayCalculatorEngine
# ==============================================================================


class HotelStayCalculatorEngine:
    """
    Engine 4: Hotel accommodation emissions calculator.

    Implements deterministic emissions calculations for hotel stays using
    DEFRA 2024 and Cornell HCMI country-specific room-night emission
    factors, adjusted by hotel class multiplier and optional extended
    stay discount.

    The engine follows GreenLang's zero-hallucination principle by using only
    deterministic Decimal arithmetic with published emission factors. No
    LLM calls are made anywhere in the calculation pipeline.

    Thread Safety:
        This engine is fully thread-safe. A reentrant lock protects shared
        state during calculations. The singleton instance is created lazily
        with double-checked locking.

    Attributes:
        _config: Business travel configuration singleton.
        _metrics: Prometheus metrics collector for monitoring.
        _provenance: SHA-256 provenance tracker for audit trails.
        _lock: Reentrant lock for thread safety.
        _calculation_count: Running count of calculations performed.
        _extended_stay_threshold: Minimum nights to qualify for extended
                                  stay discount.
        _extended_stay_discount: Discount multiplier for extended stays.

    Example:
        >>> engine = HotelStayCalculatorEngine.get_instance()
        >>> result = engine.calculate(hotel_input)
        >>> assert result.total_co2e > Decimal("0")
    """

    # ------------------------------------------------------------------
    # Singleton Access
    # ------------------------------------------------------------------

    @staticmethod
    def get_instance(
        metrics: Optional[BusinessTravelMetrics] = None,
        extended_stay_threshold: Optional[int] = None,
        extended_stay_discount: Optional[Decimal] = None,
    ) -> "HotelStayCalculatorEngine":
        """
        Get or create the singleton HotelStayCalculatorEngine instance.

        Thread-safe lazy initialization using double-checked locking.

        Args:
            metrics: Optional Prometheus metrics collector. A default
                     instance is obtained via get_metrics() if None.
            extended_stay_threshold: Minimum nights for extended stay
                                     discount. Defaults to 14.
            extended_stay_discount: Discount multiplier for extended stays
                                    (e.g. 0.85 = 15% discount). Defaults
                                    to 0.85.

        Returns:
            Singleton HotelStayCalculatorEngine instance.
        """
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:
                    _instance = HotelStayCalculatorEngine(
                        metrics=metrics,
                        extended_stay_threshold=extended_stay_threshold,
                        extended_stay_discount=extended_stay_discount,
                    )
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

    def __init__(
        self,
        metrics: Optional[BusinessTravelMetrics] = None,
        extended_stay_threshold: Optional[int] = None,
        extended_stay_discount: Optional[Decimal] = None,
    ) -> None:
        """
        Initialise the HotelStayCalculatorEngine.

        Args:
            metrics: Optional Prometheus metrics collector. A default
                     instance is obtained via get_metrics() if None.
            extended_stay_threshold: Minimum nights for extended stay
                                     discount. Defaults to 14.
            extended_stay_discount: Discount multiplier for extended stays.
                                    Defaults to 0.85 (15% reduction).
        """
        self._config = get_config()
        self._metrics: BusinessTravelMetrics = metrics or get_metrics()
        self._provenance = get_provenance_tracker()
        self._lock: threading.RLock = threading.RLock()
        self._calculation_count: int = 0

        # Extended stay parameters
        self._extended_stay_threshold: int = (
            extended_stay_threshold
            if extended_stay_threshold is not None
            else _DEFAULT_EXTENDED_STAY_THRESHOLD
        )
        self._extended_stay_discount: Decimal = (
            extended_stay_discount
            if extended_stay_discount is not None
            else _DEFAULT_EXTENDED_STAY_DISCOUNT
        )

        # Validate extended stay parameters
        if self._extended_stay_threshold < 1:
            raise ValueError(
                f"extended_stay_threshold must be >= 1, "
                f"got {self._extended_stay_threshold}"
            )
        if self._extended_stay_discount <= _ZERO or self._extended_stay_discount > _ONE:
            raise ValueError(
                f"extended_stay_discount must be > 0 and <= 1, "
                f"got {self._extended_stay_discount}"
            )

        logger.info(
            "HotelStayCalculatorEngine initialised: agent=%s, version=%s, "
            "extended_stay_threshold=%d nights, discount=%s",
            AGENT_ID,
            VERSION,
            self._extended_stay_threshold,
            self._extended_stay_discount,
        )

    # ==================================================================
    # PROPERTY: calculation_count
    # ==================================================================

    @property
    def calculation_count(self) -> int:
        """Return the total number of calculations performed by this engine."""
        return self._calculation_count

    @property
    def extended_stay_threshold(self) -> int:
        """Return the configured extended stay threshold in nights."""
        return self._extended_stay_threshold

    @property
    def extended_stay_discount(self) -> Decimal:
        """Return the configured extended stay discount multiplier."""
        return self._extended_stay_discount

    # ==================================================================
    # 1. calculate - Single hotel stay calculation
    # ==================================================================

    def calculate(
        self,
        hotel_input: HotelInput,
    ) -> HotelResult:
        """
        Calculate hotel accommodation emissions for a single stay.

        Formula:
            base_ef        = HOTEL_EMISSION_FACTORS[country_code] (or GLOBAL)
            class_mult     = HOTEL_CLASS_MULTIPLIERS[hotel_class]
            base_co2e      = room_nights x base_ef
            class_co2e     = base_co2e x class_mult
            extended_disc  = 0.85 if room_nights >= threshold else 1.0
            total_co2e     = class_co2e x extended_disc

        WTT is embedded in room-night factors (DEFRA methodology), so
        there is no separate wtt_co2e field. The 'co2e' field in the
        result contains the base (pre-class-multiplier) emissions.

        Args:
            hotel_input: Validated HotelInput with country_code, room_nights,
                         and hotel_class.

        Returns:
            HotelResult with co2e (base), total_co2e (with class multiplier
            and discount), class_multiplier, and provenance hash.

        Raises:
            ValueError: If room_nights <= 0.

        Example:
            >>> result = engine.calculate(HotelInput(
            ...     country_code="GB",
            ...     room_nights=3,
            ...     hotel_class=HotelClass.UPSCALE,
            ... ))
            >>> assert result.total_co2e > Decimal("0")
            >>> assert result.class_multiplier == Decimal("1.35")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate input
            self._validate_hotel_input(hotel_input)

            # Step 2: Resolve country emission factor
            country_code = hotel_input.country_code.upper()
            ef_per_room_night = self._resolve_country_ef(country_code)

            # Step 3: Resolve hotel class multiplier
            class_multiplier = self._resolve_class_multiplier(
                hotel_input.hotel_class
            )

            # Step 4: Determine extended stay discount
            room_nights = hotel_input.room_nights
            extended_discount = self._calculate_extended_stay_discount(
                room_nights
            )

            # Step 5: Calculate emissions (ZERO HALLUCINATION - Decimal only)
            room_nights_dec = Decimal(str(room_nights))

            # Base CO2e: room_nights x ef_per_room_night
            base_co2e = _q(room_nights_dec * ef_per_room_night)

            # Apply class multiplier
            class_co2e = _q(base_co2e * class_multiplier)

            # Apply extended stay discount
            total_co2e = _q(class_co2e * extended_discount)

            # Step 6: Provenance hash
            provenance_hash = calculate_provenance_hash(
                hotel_input,
                ef_per_room_night,
                class_multiplier,
                extended_discount,
                base_co2e,
                total_co2e,
                "hotel",
                AGENT_ID,
            )

            # Step 7: Build result
            result = HotelResult(
                country_code=country_code,
                room_nights=room_nights,
                hotel_class=hotel_input.hotel_class,
                class_multiplier=class_multiplier,
                co2e=base_co2e,
                total_co2e=total_co2e,
                ef_source=EFSource.DEFRA,
                provenance_hash=provenance_hash,
            )

            # Step 8: Record metrics
            duration = time.monotonic() - start_time
            self._record_hotel_metrics(
                country_code=country_code,
                nights=room_nights,
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "Hotel calculation complete: country=%s, nights=%d, "
                "class=%s, multiplier=%s, discount=%s, base_co2e=%s kg, "
                "total_co2e=%s kg",
                country_code,
                room_nights,
                hotel_input.hotel_class.value,
                class_multiplier,
                extended_discount,
                base_co2e,
                total_co2e,
            )

            return result

    # ==================================================================
    # 2. calculate_batch - Batch hotel stay processing
    # ==================================================================

    def calculate_batch(
        self,
        inputs: List[HotelInput],
    ) -> List[HotelResult]:
        """
        Calculate hotel accommodation emissions for multiple stays.

        Each input is processed independently. Failed calculations are
        logged as warnings but do not halt the batch. Successful results
        are returned in a list (preserving input order for successful
        items only; callers can also use calculate_batch_with_errors for
        full error tracking).

        Args:
            inputs: List of HotelInput objects to process.

        Returns:
            List of HotelResult objects for successfully processed inputs.

        Raises:
            ValueError: If inputs list exceeds _MAX_BATCH_SIZE.

        Example:
            >>> results = engine.calculate_batch([
            ...     HotelInput(country_code="US", room_nights=2),
            ...     HotelInput(country_code="GB", room_nights=5,
            ...                hotel_class=HotelClass.LUXURY),
            ... ])
            >>> assert len(results) == 2
        """
        if len(inputs) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(inputs)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        results: List[HotelResult] = []
        errors: int = 0

        for idx, hotel_input in enumerate(inputs):
            try:
                result = self.calculate(hotel_input)
                results.append(result)
            except Exception as exc:
                errors += 1
                logger.warning(
                    "Batch hotel item %d failed: %s", idx, str(exc)
                )

        logger.info(
            "Batch hotel calculation complete: total=%d, success=%d, "
            "errors=%d",
            len(inputs),
            len(results),
            errors,
        )

        return results

    def calculate_batch_with_errors(
        self,
        inputs: List[HotelInput],
    ) -> List[Dict[str, Any]]:
        """
        Calculate hotel emissions for multiple stays with error tracking.

        Unlike calculate_batch which only returns successful results, this
        method returns a list of dicts that includes both successes and
        failures, preserving input order.

        Args:
            inputs: List of HotelInput objects to process.

        Returns:
            List of dicts, each containing 'index', 'status' ('success'
            or 'error'), and either 'result' (HotelResult) or 'error'
            (error message string).

        Raises:
            ValueError: If inputs list exceeds _MAX_BATCH_SIZE.
        """
        if len(inputs) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(inputs)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        results: List[Dict[str, Any]] = []

        for idx, hotel_input in enumerate(inputs):
            try:
                result = self.calculate(hotel_input)
                results.append({
                    "index": idx,
                    "status": "success",
                    "result": result,
                })
            except Exception as exc:
                logger.warning(
                    "Batch hotel item %d failed: %s", idx, str(exc)
                )
                results.append({
                    "index": idx,
                    "status": "error",
                    "error": str(exc),
                })

        return results

    # ==================================================================
    # 3. get_available_countries - List countries with emission factors
    # ==================================================================

    @staticmethod
    def get_available_countries() -> List[Dict[str, Any]]:
        """
        List all countries that have hotel emission factors.

        Returns a list of dicts, each containing the country code and
        the emission factor in kgCO2e per room-night.

        Returns:
            List of dicts with 'country_code' and 'ef_per_room_night' keys.

        Example:
            >>> countries = HotelStayCalculatorEngine.get_available_countries()
            >>> assert len(countries) >= 16
            >>> gb = next(c for c in countries if c['country_code'] == 'GB')
            >>> assert gb['ef_per_room_night'] == Decimal('12.32')
        """
        countries: List[Dict[str, Any]] = []
        for code, ef in HOTEL_EMISSION_FACTORS.items():
            countries.append({
                "country_code": code,
                "ef_per_room_night": ef,
            })

        # Sort by country code for deterministic ordering
        countries.sort(key=lambda c: c["country_code"])
        return countries

    # ==================================================================
    # 4. get_hotel_class_multipliers - Return all class multipliers
    # ==================================================================

    @staticmethod
    def get_hotel_class_multipliers() -> Dict[str, Decimal]:
        """
        Return all hotel class multipliers as a dict keyed by class value.

        The multiplier adjusts the base room-night emission factor:
            - BUDGET: 0.75 (smaller room, fewer amenities)
            - STANDARD: 1.0 (baseline)
            - UPSCALE: 1.35 (larger room, more services)
            - LUXURY: 1.80 (suite, full services)

        Returns:
            Dict mapping hotel class string to its Decimal multiplier.

        Example:
            >>> mults = HotelStayCalculatorEngine.get_hotel_class_multipliers()
            >>> mults['luxury']
            Decimal('1.80')
        """
        return {
            hc.value: mult
            for hc, mult in HOTEL_CLASS_MULTIPLIERS.items()
        }

    # ==================================================================
    # Internal: Validation
    # ==================================================================

    def _validate_hotel_input(self, hotel_input: HotelInput) -> None:
        """
        Validate hotel input parameters.

        Args:
            hotel_input: HotelInput to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if hotel_input.room_nights <= 0:
            raise ValueError(
                f"Hotel room_nights must be positive, got {hotel_input.room_nights}"
            )
        if not hotel_input.country_code:
            raise ValueError("Hotel country_code cannot be empty")
        if hotel_input.hotel_class not in HOTEL_CLASS_MULTIPLIERS:
            raise ValueError(
                f"Unknown hotel_class '{hotel_input.hotel_class.value}'. "
                f"Available: {list(HOTEL_CLASS_MULTIPLIERS.keys())}"
            )

    # ==================================================================
    # Internal: Emission Factor Resolution
    # ==================================================================

    def _resolve_country_ef(self, country_code: str) -> Decimal:
        """
        Resolve the hotel emission factor for a given country code.

        Falls back to the 'GLOBAL' factor if the country code is not
        found in HOTEL_EMISSION_FACTORS.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (uppercase).

        Returns:
            Emission factor in kgCO2e per room-night.

        Example:
            >>> engine._resolve_country_ef("GB")
            Decimal('12.32')
            >>> engine._resolve_country_ef("ZZ")  # falls back to GLOBAL
            Decimal('20.90')
        """
        code = country_code.upper()
        ef = HOTEL_EMISSION_FACTORS.get(code)

        if ef is not None:
            logger.debug(
                "Resolved hotel EF: country=%s, ef=%s kgCO2e/room-night",
                code,
                ef,
            )
            return ef

        # Fall back to GLOBAL
        global_ef = HOTEL_EMISSION_FACTORS["GLOBAL"]
        logger.info(
            "Country '%s' not found in HOTEL_EMISSION_FACTORS, "
            "falling back to GLOBAL (%s kgCO2e/room-night)",
            code,
            global_ef,
        )
        return global_ef

    def _resolve_class_multiplier(self, hotel_class: HotelClass) -> Decimal:
        """
        Resolve the hotel class multiplier.

        Args:
            hotel_class: The HotelClass enum value.

        Returns:
            Class multiplier as a Decimal.

        Raises:
            KeyError: If hotel_class is not in HOTEL_CLASS_MULTIPLIERS.
        """
        mult = HOTEL_CLASS_MULTIPLIERS.get(hotel_class)
        if mult is None:
            raise KeyError(
                f"Hotel class multiplier not found for '{hotel_class.value}'"
            )

        logger.debug(
            "Resolved hotel class multiplier: class=%s, multiplier=%s",
            hotel_class.value,
            mult,
        )
        return mult

    # ==================================================================
    # Internal: Extended Stay Discount
    # ==================================================================

    def _calculate_extended_stay_discount(
        self, room_nights: int
    ) -> Decimal:
        """
        Calculate the extended stay discount multiplier.

        If room_nights >= extended_stay_threshold, apply the discount.
        Otherwise return 1.0 (no discount).

        The rationale: extended stays have lower per-night emissions due
        to reduced linen changes, fewer check-in/check-out overheads,
        and more efficient energy use.

        Args:
            room_nights: Number of room-nights in the stay.

        Returns:
            Decimal discount multiplier (e.g. 0.85 for extended, 1.0 otherwise).
        """
        if room_nights >= self._extended_stay_threshold:
            logger.debug(
                "Extended stay discount applied: nights=%d >= threshold=%d, "
                "discount=%s",
                room_nights,
                self._extended_stay_threshold,
                self._extended_stay_discount,
            )
            return self._extended_stay_discount

        return _ONE

    # ==================================================================
    # Internal: Metrics Recording
    # ==================================================================

    def _record_hotel_metrics(
        self,
        country_code: str,
        nights: int,
        co2e: float,
        duration: float,
    ) -> None:
        """
        Record hotel calculation metrics to Prometheus.

        Wraps calls to the BusinessTravelMetrics singleton to record
        calculation throughput, emissions, duration, and hotel night
        counters. All calls are wrapped in try/except to prevent
        metrics failures from disrupting calculations.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            nights: Number of hotel nights.
            co2e: Emissions in kgCO2e.
            duration: Calculation duration in seconds.
        """
        try:
            # Record the primary calculation counter
            self._metrics.record_calculation(
                method="distance_based",
                mode="hotel",
                status="success",
                duration=duration,
                co2e=co2e,
                rf_option="without_rf",
            )

            # Record hotel nights counter
            self._metrics.record_hotel(
                country=country_code,
                nights=nights,
            )
        except Exception as exc:
            logger.warning(
                "Failed to record hotel metrics: %s", str(exc)
            )


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "HotelStayCalculatorEngine",
]
