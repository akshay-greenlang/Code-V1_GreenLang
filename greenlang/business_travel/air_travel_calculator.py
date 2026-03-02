# -*- coding: utf-8 -*-
"""
AirTravelCalculatorEngine - Aviation emissions with GCD, uplift, cabin class, RF.

This module implements the AirTravelCalculatorEngine for AGENT-MRV-019
(Business Travel, GHG Protocol Scope 3 Category 6). It provides thread-safe
singleton aviation emissions calculations using great-circle distance (GCD),
distance uplift, cabin class multipliers, and radiative forcing (RF) options.

Calculation Pipeline:
    1. Validate input (IATA codes exist in AIRPORT_DATABASE)
    2. Look up airport coordinates
    3. Calculate great-circle distance (Haversine formula)
    4. Apply uplift factor (default 8% for routing inefficiency)
    5. Classify distance band (domestic < 500 km, short-haul < 3700 km, long-haul >= 3700 km)
    6. Get cabin class multiplier (economy=1.0, premium=1.6, business=2.9, first=4.0)
    7. Get emission factors for distance band (DEFRA 2024)
    8. Calculate CO2e without RF = distance x passengers x class_multiplier x EF_without_rf
    9. Calculate CO2e with RF = distance x passengers x class_multiplier x EF_with_rf
    10. Calculate WTT = distance x passengers x class_multiplier x WTT_EF
    11. Calculate total based on rf_option
    12. If round_trip, double everything
    13. Record provenance hash (SHA-256)
    14. Record Prometheus metrics
    15. Return FlightResult

Radiative Forcing (RF):
    Aviation emissions at altitude produce non-CO2 climate effects including
    contrails, NOx, and water vapour. The DEFRA methodology includes an RF
    multiplier within the "with_rf" emission factor. The GHG Protocol
    recommends disclosing emissions both with and without RF.

Thread Safety:
    Uses __new__ singleton pattern with threading.Lock for thread-safe
    instantiation. All mutable state is protected by locks.

Example:
    >>> from greenlang.business_travel.models import FlightInput, CabinClass
    >>> engine = AirTravelCalculatorEngine()
    >>> flight = FlightInput(
    ...     origin_iata="JFK",
    ...     destination_iata="LHR",
    ...     cabin_class=CabinClass.BUSINESS,
    ...     passengers=1,
    ...     round_trip=True
    ... )
    >>> result = engine.calculate(flight)
    >>> result.distance_band
    <FlightDistanceBand.LONG_HAUL: 'long_haul'>

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-006
"""

import logging
import math
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

from greenlang.business_travel.models import (
    FlightInput,
    FlightResult,
    CabinClass,
    FlightDistanceBand,
    RFOption,
    EFSource,
    GWPVersion,
    AIR_EMISSION_FACTORS,
    CABIN_CLASS_MULTIPLIERS,
    AIRPORT_DATABASE,
    GWP_VALUES,
    calculate_provenance_hash,
)
from greenlang.business_travel.config import get_config
from greenlang.business_travel.metrics import get_metrics
from greenlang.business_travel.business_travel_database import (
    BusinessTravelDatabaseEngine,
    get_database_engine,
)

logger = logging.getLogger(__name__)

# Quantization constant: 8 decimal places
_QUANT_8DP = Decimal("0.00000001")

# Earth radius in kilometres for Haversine formula
_EARTH_RADIUS_KM = Decimal("6371.0")

# Default distance band thresholds in km
_DOMESTIC_THRESHOLD_KM = Decimal("500")
_SHORT_HAUL_THRESHOLD_KM = Decimal("3700")

# Default uplift factor (8% for routing inefficiency)
_DEFAULT_UPLIFT_FACTOR = Decimal("0.08")


# =============================================================================
# ENGINE CLASS
# =============================================================================


class AirTravelCalculatorEngine:
    """
    Thread-safe singleton engine for aviation emissions calculations.

    Implements the complete flight emissions calculation pipeline per GHG
    Protocol Scope 3 Category 6 and DEFRA 2024 methodology. All arithmetic
    uses Python Decimal with ROUND_HALF_UP quantization to 8 decimal places
    for regulatory precision.

    This engine is ZERO-HALLUCINATION: all calculations are deterministic
    Python arithmetic. No LLM calls are used for any numeric computation.

    Thread Safety:
        Uses __new__ singleton pattern with threading.Lock. The
        _calculation_count attribute is protected by a dedicated lock.

    Attributes:
        _config: Singleton configuration from get_config()
        _metrics: Singleton metrics from get_metrics()
        _database_engine: BusinessTravelDatabaseEngine for factor lookups
        _calculation_count: Total number of calculations performed

    Example:
        >>> engine = AirTravelCalculatorEngine()
        >>> flight = FlightInput(
        ...     origin_iata="JFK",
        ...     destination_iata="LHR",
        ...     cabin_class=CabinClass.ECONOMY,
        ...     passengers=1,
        ... )
        >>> result = engine.calculate(flight)
        >>> assert result.provenance_hash  # SHA-256 hash present
    """

    _instance: Optional["AirTravelCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "AirTravelCalculatorEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the air travel calculator engine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._config = get_config()
        self._metrics = get_metrics()
        self._database_engine: Optional[BusinessTravelDatabaseEngine] = None
        self._calculation_count: int = 0
        self._count_lock: threading.Lock = threading.Lock()

        logger.info(
            "AirTravelCalculatorEngine initialized: "
            "uplift_factor=%s, domestic_threshold=%s km, "
            "short_haul_threshold=%s km, earth_radius=%s km",
            _DEFAULT_UPLIFT_FACTOR,
            _DOMESTIC_THRESHOLD_KM,
            _SHORT_HAUL_THRESHOLD_KM,
            _EARTH_RADIUS_KM,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _get_database_engine(self) -> BusinessTravelDatabaseEngine:
        """
        Lazy-load the database engine singleton.

        Returns:
            BusinessTravelDatabaseEngine instance.
        """
        if self._database_engine is None:
            self._database_engine = get_database_engine()
        return self._database_engine

    def _increment_calculation_count(self) -> int:
        """
        Increment and return the calculation counter in a thread-safe manner.

        Returns:
            Updated calculation count.
        """
        with self._count_lock:
            self._calculation_count += 1
            return self._calculation_count

    def _quantize(self, value: Decimal) -> Decimal:
        """
        Quantize a Decimal value to 8 decimal places with ROUND_HALF_UP.

        Args:
            value: Decimal value to quantize.

        Returns:
            Quantized Decimal value.
        """
        return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

    # =========================================================================
    # HAVERSINE GREAT-CIRCLE DISTANCE
    # =========================================================================

    def calculate_great_circle_distance(
        self,
        lat1: Decimal,
        lon1: Decimal,
        lat2: Decimal,
        lon2: Decimal,
    ) -> Decimal:
        """
        Calculate great-circle distance between two geographic points.

        Uses the Haversine formula:
            a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            d = R * c

        Where R = 6371.0 km (mean Earth radius).

        Trigonometric functions use Python math module (float), with
        results immediately converted back to Decimal for precision.

        Args:
            lat1: Latitude of point 1 in decimal degrees.
            lon1: Longitude of point 1 in decimal degrees.
            lat2: Latitude of point 2 in decimal degrees.
            lon2: Longitude of point 2 in decimal degrees.

        Returns:
            Great-circle distance in kilometres, quantized to 8 decimal places.

        Example:
            >>> engine = AirTravelCalculatorEngine()
            >>> dist = engine.calculate_great_circle_distance(
            ...     Decimal("40.6413"), Decimal("-73.7781"),  # JFK
            ...     Decimal("51.4700"), Decimal("-0.4543"),   # LHR
            ... )
            >>> dist  # ~5541 km
        """
        # Convert Decimal degrees to float radians for trig functions
        lat1_rad = math.radians(float(lat1))
        lon1_rad = math.radians(float(lon1))
        lat2_rad = math.radians(float(lat2))
        lon2_rad = math.radians(float(lon2))

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(lat1_rad)
            * math.cos(lat2_rad)
            * math.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

        # Convert result back to Decimal immediately
        c_decimal = Decimal(str(c))
        distance = _EARTH_RADIUS_KM * c_decimal

        result = self._quantize(distance)

        logger.debug(
            "Great-circle distance: (%s, %s) -> (%s, %s) = %s km",
            lat1, lon1, lat2, lon2, result,
        )

        return result

    # =========================================================================
    # UPLIFT FACTOR
    # =========================================================================

    def apply_uplift(
        self,
        distance_km: Decimal,
        uplift_factor: Decimal = _DEFAULT_UPLIFT_FACTOR,
    ) -> Decimal:
        """
        Apply distance uplift factor to account for routing inefficiency.

        Airlines do not fly perfect great-circle routes due to airspace
        restrictions, weather avoidance, and ATC routing. The DEFRA
        methodology recommends an 8% uplift to compensate.

        Formula: uplifted_distance = distance * (1 + uplift_factor)

        Args:
            distance_km: Great-circle distance in kilometres.
            uplift_factor: Uplift factor (default 0.08 = 8%).

        Returns:
            Uplifted distance in kilometres, quantized to 8 decimal places.

        Example:
            >>> engine = AirTravelCalculatorEngine()
            >>> engine.apply_uplift(Decimal("5541.0"))
            Decimal('5984.28000000')
        """
        uplifted = distance_km * (Decimal("1") + uplift_factor)
        result = self._quantize(uplifted)

        logger.debug(
            "Uplift applied: %s km * (1 + %s) = %s km",
            distance_km, uplift_factor, result,
        )

        return result

    # =========================================================================
    # DISTANCE BAND CLASSIFICATION
    # =========================================================================

    def classify_distance_band(
        self,
        distance_km: Decimal,
    ) -> FlightDistanceBand:
        """
        Classify a flight distance into DEFRA distance bands.

        Thresholds:
        - DOMESTIC: < 500 km
        - SHORT_HAUL: >= 500 km and < 3700 km
        - LONG_HAUL: >= 3700 km

        Args:
            distance_km: Flight distance in kilometres (after uplift).

        Returns:
            FlightDistanceBand classification.

        Example:
            >>> engine = AirTravelCalculatorEngine()
            >>> engine.classify_distance_band(Decimal("400"))
            <FlightDistanceBand.DOMESTIC: 'domestic'>
            >>> engine.classify_distance_band(Decimal("2000"))
            <FlightDistanceBand.SHORT_HAUL: 'short_haul'>
            >>> engine.classify_distance_band(Decimal("8000"))
            <FlightDistanceBand.LONG_HAUL: 'long_haul'>
        """
        if distance_km < _DOMESTIC_THRESHOLD_KM:
            band = FlightDistanceBand.DOMESTIC
        elif distance_km < _SHORT_HAUL_THRESHOLD_KM:
            band = FlightDistanceBand.SHORT_HAUL
        else:
            band = FlightDistanceBand.LONG_HAUL

        logger.debug(
            "Distance band classification: %s km -> %s",
            distance_km, band.value,
        )

        return band

    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================

    def calculate(
        self,
        flight_input: FlightInput,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> FlightResult:
        """
        Calculate aviation emissions for a single flight.

        Full calculation pipeline:
            1. Validate IATA codes exist in AIRPORT_DATABASE
            2. Look up airport coordinates (lat, lon)
            3. Calculate great-circle distance (Haversine)
            4. Apply uplift factor (default 8%)
            5. Classify distance band (domestic/short-haul/long-haul)
            6. Get cabin class multiplier
            7. Get emission factors for distance band (DEFRA 2024)
            8. CO2e_without_rf = distance x passengers x class_multiplier x EF_without_rf
            9. CO2e_with_rf = distance x passengers x class_multiplier x EF_with_rf
            10. WTT = distance x passengers x class_multiplier x WTT_EF
            11. Total = based on rf_option (with_rf, without_rf, or both->with_rf)
            12. If round_trip, double all values
            13. Record provenance hash
            14. Record Prometheus metrics
            15. Return FlightResult

        Args:
            flight_input: Validated FlightInput with IATA codes, cabin class,
                          passengers, round_trip flag, and RF option.
            gwp_version: IPCC GWP assessment report version (default AR5).

        Returns:
            FlightResult with emissions breakdown, provenance hash, and metadata.

        Raises:
            ValueError: If origin or destination IATA code is not in the database.
            ValueError: If emission factors are not available for the distance band.

        Example:
            >>> engine = AirTravelCalculatorEngine()
            >>> flight = FlightInput(
            ...     origin_iata="JFK",
            ...     destination_iata="LHR",
            ...     cabin_class=CabinClass.BUSINESS,
            ...     passengers=1,
            ...     round_trip=True,
            ... )
            >>> result = engine.calculate(flight)
            >>> result.distance_band == FlightDistanceBand.LONG_HAUL
            True
        """
        start_time = time.monotonic()
        calc_number = self._increment_calculation_count()

        logger.info(
            "Flight calculation #%d: %s -> %s, class=%s, pax=%d, "
            "round_trip=%s, rf=%s, gwp=%s",
            calc_number,
            flight_input.origin_iata,
            flight_input.destination_iata,
            flight_input.cabin_class.value,
            flight_input.passengers,
            flight_input.round_trip,
            flight_input.rf_option.value,
            gwp_version.value,
        )

        # Step 1: Validate IATA codes exist
        origin = self._validate_airport(flight_input.origin_iata, "origin")
        destination = self._validate_airport(
            flight_input.destination_iata, "destination"
        )

        # Step 2: Extract coordinates
        origin_lat = origin["lat"]
        origin_lon = origin["lon"]
        dest_lat = destination["lat"]
        dest_lon = destination["lon"]

        # Step 3: Calculate great-circle distance
        gcd_km = self.calculate_great_circle_distance(
            origin_lat, origin_lon, dest_lat, dest_lon
        )

        # Step 4: Apply uplift factor
        uplift_factor = self._get_uplift_factor()
        distance_km = self.apply_uplift(gcd_km, uplift_factor)

        # Step 5: Classify distance band
        distance_band = self.classify_distance_band(distance_km)

        # Step 6: Get cabin class multiplier
        db_engine = self._get_database_engine()
        class_multiplier = db_engine.get_cabin_class_multiplier(
            flight_input.cabin_class
        )

        # Step 7: Get emission factors for distance band
        ef_data = db_engine.get_air_emission_factor(
            distance_band,
            flight_input.cabin_class,
            EFSource.DEFRA,
        )

        ef_without_rf = ef_data["without_rf"]
        ef_with_rf = ef_data["with_rf"]
        ef_wtt = ef_data["wtt"]

        # Step 8-10: Calculate emissions (ZERO-HALLUCINATION deterministic math)
        passengers_dec = Decimal(str(flight_input.passengers))

        co2e_without_rf = self._calculate_emissions_component(
            distance_km, passengers_dec, class_multiplier, ef_without_rf
        )

        co2e_with_rf = self._calculate_emissions_component(
            distance_km, passengers_dec, class_multiplier, ef_with_rf
        )

        wtt_co2e = self._calculate_emissions_component(
            distance_km, passengers_dec, class_multiplier, ef_wtt
        )

        # Step 11: Calculate total based on rf_option
        total_co2e = self._calculate_total_by_rf_option(
            co2e_without_rf,
            co2e_with_rf,
            wtt_co2e,
            flight_input.rf_option,
        )

        # Step 12: Double for round trip
        round_trip_multiplier = Decimal("2") if flight_input.round_trip else Decimal("1")
        co2e_without_rf = self._quantize(co2e_without_rf * round_trip_multiplier)
        co2e_with_rf = self._quantize(co2e_with_rf * round_trip_multiplier)
        wtt_co2e = self._quantize(wtt_co2e * round_trip_multiplier)
        total_co2e = self._quantize(total_co2e * round_trip_multiplier)

        # Also double the distance for reporting if round trip
        reported_distance = self._quantize(
            distance_km * round_trip_multiplier
        )

        # Step 13: Record provenance hash
        provenance_hash = calculate_provenance_hash(
            flight_input,
            distance_km,
            distance_band.value,
            class_multiplier,
            ef_without_rf,
            ef_with_rf,
            ef_wtt,
            co2e_without_rf,
            co2e_with_rf,
            wtt_co2e,
            total_co2e,
        )

        # Step 14: Record Prometheus metrics
        duration = time.monotonic() - start_time
        self._record_metrics(
            flight_input=flight_input,
            distance_band=distance_band,
            distance_km=float(distance_km),
            total_co2e=float(total_co2e),
            duration=duration,
        )

        # Step 15: Build and return FlightResult
        result = FlightResult(
            origin_iata=flight_input.origin_iata,
            destination_iata=flight_input.destination_iata,
            distance_km=reported_distance,
            distance_band=distance_band,
            cabin_class=flight_input.cabin_class,
            passengers=flight_input.passengers,
            class_multiplier=class_multiplier,
            co2e_without_rf=co2e_without_rf,
            co2e_with_rf=co2e_with_rf,
            wtt_co2e=wtt_co2e,
            total_co2e=total_co2e,
            ef_source=EFSource.DEFRA,
            rf_option=flight_input.rf_option,
            provenance_hash=provenance_hash,
        )

        logger.info(
            "Flight calculation #%d complete: %s -> %s, distance=%s km, "
            "band=%s, total_co2e=%s kgCO2e, duration=%.3fs, "
            "provenance=%s...%s",
            calc_number,
            flight_input.origin_iata,
            flight_input.destination_iata,
            reported_distance,
            distance_band.value,
            total_co2e,
            duration,
            provenance_hash[:8],
            provenance_hash[-8:],
        )

        return result

    # =========================================================================
    # EMISSIONS COMPONENT CALCULATION
    # =========================================================================

    def _calculate_emissions_component(
        self,
        distance_km: Decimal,
        passengers: Decimal,
        class_multiplier: Decimal,
        emission_factor: Decimal,
    ) -> Decimal:
        """
        Calculate a single emissions component.

        Formula: emissions = distance_km x passengers x class_multiplier x EF

        All operands are Decimal. Result is quantized to 8 decimal places.

        Args:
            distance_km: Flight distance in km (after uplift).
            passengers: Number of passengers as Decimal.
            class_multiplier: Cabin class multiplier.
            emission_factor: Emission factor per passenger-km (kgCO2e/pkm).

        Returns:
            Emissions in kgCO2e, quantized to 8 decimal places.
        """
        emissions = distance_km * passengers * class_multiplier * emission_factor
        return self._quantize(emissions)

    def _calculate_total_by_rf_option(
        self,
        co2e_without_rf: Decimal,
        co2e_with_rf: Decimal,
        wtt_co2e: Decimal,
        rf_option: RFOption,
    ) -> Decimal:
        """
        Calculate total emissions based on the radiative forcing option.

        RF Options:
        - WITH_RF: total = co2e_with_rf + wtt_co2e
        - WITHOUT_RF: total = co2e_without_rf + wtt_co2e
        - BOTH: total = co2e_with_rf + wtt_co2e (conservative, reports both)

        Args:
            co2e_without_rf: CO2e without radiative forcing.
            co2e_with_rf: CO2e with radiative forcing.
            wtt_co2e: Well-to-tank emissions.
            rf_option: Radiative forcing reporting option.

        Returns:
            Total emissions in kgCO2e.
        """
        if rf_option == RFOption.WITHOUT_RF:
            total = co2e_without_rf + wtt_co2e
        elif rf_option == RFOption.WITH_RF:
            total = co2e_with_rf + wtt_co2e
        else:
            # BOTH: use with_rf as the primary total (both values are in result)
            total = co2e_with_rf + wtt_co2e

        return self._quantize(total)

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_airport(
        self,
        iata_code: str,
        role: str,
    ) -> Dict[str, object]:
        """
        Validate that an IATA code exists in the airport database.

        Args:
            iata_code: 3-letter IATA code to validate.
            role: Descriptive role ("origin" or "destination") for error messages.

        Returns:
            Airport data dict with name, lat, lon, country.

        Raises:
            ValueError: If IATA code is not found in AIRPORT_DATABASE.
        """
        code = iata_code.upper().strip()
        airport_data = AIRPORT_DATABASE.get(code)

        if airport_data is None:
            available_codes = sorted(AIRPORT_DATABASE.keys())
            raise ValueError(
                f"{role.capitalize()} airport IATA code '{code}' not found "
                f"in database. Available codes ({len(available_codes)}): "
                f"{available_codes[:10]}... (use search_airports for full list)"
            )

        logger.debug(
            "Validated %s airport: %s (%s, %s)",
            role,
            code,
            airport_data["name"],
            airport_data["country"],
        )

        return airport_data

    def _get_uplift_factor(self) -> Decimal:
        """
        Get the configured uplift factor.

        Falls back to the default 8% if configuration is unavailable.

        Returns:
            Uplift factor as Decimal.
        """
        try:
            return self._config.general.default_uplift_factor
        except Exception:
            logger.warning(
                "Could not read uplift factor from config, using default %s",
                _DEFAULT_UPLIFT_FACTOR,
            )
            return _DEFAULT_UPLIFT_FACTOR

    # =========================================================================
    # METRICS RECORDING
    # =========================================================================

    def _record_metrics(
        self,
        flight_input: FlightInput,
        distance_band: FlightDistanceBand,
        distance_km: float,
        total_co2e: float,
        duration: float,
    ) -> None:
        """
        Record Prometheus metrics for a completed flight calculation.

        Records:
        - Calculation counter (method=distance_based, mode=air, status=success)
        - Flight counter (distance_band, cabin_class)
        - Emissions counter (mode=air, rf_option)
        - Duration histogram
        - Distance counter

        Args:
            flight_input: Original flight input.
            distance_band: Classified distance band.
            distance_km: Total distance in km.
            total_co2e: Total emissions in kgCO2e.
            duration: Calculation duration in seconds.
        """
        try:
            # Record main calculation metric
            self._metrics.record_calculation(
                method="distance_based",
                mode="air",
                status="success",
                duration=duration,
                co2e=total_co2e,
                rf_option=flight_input.rf_option.value,
            )

            # Record flight-specific metric
            self._metrics.record_flight(
                distance_band=distance_band.value,
                cabin_class=flight_input.cabin_class.value,
                distance_km=distance_km,
            )

        except Exception as exc:
            logger.warning(
                "Failed to record flight metrics: %s", exc
            )

    # =========================================================================
    # BATCH CALCULATION
    # =========================================================================

    def calculate_batch(
        self,
        flights: List[FlightInput],
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> List[FlightResult]:
        """
        Calculate emissions for a batch of flights.

        Processes each flight independently. Failed calculations are logged
        but do not halt the batch. Results are returned in the same order
        as the input list; failed calculations are omitted from the output.

        Args:
            flights: List of FlightInput objects to process.
            gwp_version: IPCC GWP version for all calculations (default AR5).

        Returns:
            List of FlightResult objects for successfully calculated flights.

        Example:
            >>> engine = AirTravelCalculatorEngine()
            >>> flights = [
            ...     FlightInput(origin_iata="JFK", destination_iata="LHR"),
            ...     FlightInput(origin_iata="LAX", destination_iata="SFO"),
            ... ]
            >>> results = engine.calculate_batch(flights)
            >>> len(results)
            2
        """
        start_time = time.monotonic()
        results: List[FlightResult] = []
        error_count = 0

        logger.info(
            "Batch flight calculation started: %d flights, gwp=%s",
            len(flights),
            gwp_version.value,
        )

        for i, flight_input in enumerate(flights):
            try:
                result = self.calculate(flight_input, gwp_version)
                results.append(result)
            except Exception as exc:
                error_count += 1
                logger.error(
                    "Batch flight #%d failed (%s -> %s): %s",
                    i + 1,
                    flight_input.origin_iata,
                    flight_input.destination_iata,
                    exc,
                    exc_info=True,
                )

        total_duration = time.monotonic() - start_time

        # Record batch metrics
        try:
            batch_status = "completed" if error_count == 0 else "partial"
            self._metrics.record_batch(
                status=batch_status,
                size=len(flights),
            )
        except Exception as exc:
            logger.warning("Failed to record batch metrics: %s", exc)

        logger.info(
            "Batch flight calculation complete: %d/%d succeeded, "
            "%d errors, duration=%.3fs",
            len(results),
            len(flights),
            error_count,
            total_duration,
        )

        return results

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def get_distance_between_airports(
        self,
        origin_iata: str,
        destination_iata: str,
    ) -> Decimal:
        """
        Calculate the distance between two airports (GCD + uplift).

        Convenience method that looks up airport coordinates, calculates
        the great-circle distance, and applies the default uplift factor.

        Args:
            origin_iata: Origin airport IATA code.
            destination_iata: Destination airport IATA code.

        Returns:
            Distance in kilometres (after uplift), quantized to 8 decimal places.

        Raises:
            ValueError: If either IATA code is not in the airport database.

        Example:
            >>> engine = AirTravelCalculatorEngine()
            >>> dist = engine.get_distance_between_airports("JFK", "LHR")
            >>> dist > Decimal("5000")
            True
        """
        # Validate and look up airports
        origin = self._validate_airport(origin_iata, "origin")
        destination = self._validate_airport(destination_iata, "destination")

        # Calculate great-circle distance
        gcd_km = self.calculate_great_circle_distance(
            origin["lat"],
            origin["lon"],
            destination["lat"],
            destination["lon"],
        )

        # Apply uplift
        uplift_factor = self._get_uplift_factor()
        distance_km = self.apply_uplift(gcd_km, uplift_factor)

        logger.info(
            "Distance %s -> %s: GCD=%s km, uplifted=%s km (factor=%s)",
            origin_iata.upper(),
            destination_iata.upper(),
            gcd_km,
            distance_km,
            uplift_factor,
        )

        return distance_km

    # =========================================================================
    # SUMMARY AND STATS
    # =========================================================================

    def get_calculation_count(self) -> int:
        """
        Get the total number of calculations performed.

        Returns:
            Integer count of calculations.
        """
        with self._count_lock:
            return self._calculation_count

    def get_engine_summary(self) -> Dict[str, object]:
        """
        Get a summary of the engine state and configuration.

        Returns:
            Dict with engine metadata and statistics.

        Example:
            >>> engine = AirTravelCalculatorEngine()
            >>> summary = engine.get_engine_summary()
            >>> summary["uplift_factor"]
            '0.08'
        """
        return {
            "engine": "AirTravelCalculatorEngine",
            "agent_id": "GL-MRV-S3-006",
            "version": "1.0.0",
            "calculation_count": self.get_calculation_count(),
            "uplift_factor": str(_DEFAULT_UPLIFT_FACTOR),
            "earth_radius_km": str(_EARTH_RADIUS_KM),
            "domestic_threshold_km": str(_DOMESTIC_THRESHOLD_KM),
            "short_haul_threshold_km": str(_SHORT_HAUL_THRESHOLD_KM),
            "cabin_classes": len(CABIN_CLASS_MULTIPLIERS),
            "distance_bands": len(AIR_EMISSION_FACTORS),
            "airports": len(AIRPORT_DATABASE),
        }

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
# MODULE-LEVEL ACCESSOR
# =============================================================================

_calculator_instance: Optional[AirTravelCalculatorEngine] = None
_calculator_lock: threading.Lock = threading.Lock()


def get_air_travel_calculator() -> AirTravelCalculatorEngine:
    """
    Get the singleton AirTravelCalculatorEngine instance.

    Thread-safe accessor for the global air travel calculator instance.

    Returns:
        AirTravelCalculatorEngine singleton instance.

    Example:
        >>> calculator = get_air_travel_calculator()
        >>> result = calculator.calculate(flight_input)
    """
    global _calculator_instance
    with _calculator_lock:
        if _calculator_instance is None:
            _calculator_instance = AirTravelCalculatorEngine()
        return _calculator_instance


def reset_air_travel_calculator() -> None:
    """
    Reset the module-level calculator instance (for testing only).

    Warning: This function is intended for use in test fixtures only.
    Do not call in production code.
    """
    global _calculator_instance
    with _calculator_lock:
        _calculator_instance = None
    AirTravelCalculatorEngine.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "AirTravelCalculatorEngine",
    "get_air_travel_calculator",
    "reset_air_travel_calculator",
]
