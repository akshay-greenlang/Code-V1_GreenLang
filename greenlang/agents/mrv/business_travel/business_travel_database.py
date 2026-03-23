# -*- coding: utf-8 -*-
"""
BusinessTravelDatabaseEngine - Emission factor database and classification engine.

This module implements the BusinessTravelDatabaseEngine for AGENT-MRV-019
(Business Travel, GHG Protocol Scope 3 Category 6). It provides thread-safe
singleton access to emission factor databases, airport lookups, cabin class
multipliers, EEIO factors, currency conversion, and CPI deflation.

Features:
- Air emission factors by distance band and cabin class (DEFRA 2024)
- Rail emission factors for 8 rail types with WTT
- Road vehicle emission factors for 13 vehicle types
- Fuel-based emission factors for 5 fuel types
- Bus emission factors (local/coach)
- Ferry emission factors (foot/car passenger)
- Hotel room-night emission factors for 16 countries with class multipliers
- EEIO spend-based factors for 10 NAICS codes
- 50 major airport database with IATA code lookup and search
- Currency conversion (12 currencies to USD)
- CPI deflation (2015-2025)
- Transport mode classification from trip data
- Thread-safe singleton pattern with __new__
- Zero-hallucination factor retrieval
- Provenance tracking via SHA-256 hashes
- Prometheus metrics recording for all lookups

Example:
    >>> engine = BusinessTravelDatabaseEngine()
    >>> factor = engine.get_air_emission_factor(
    ...     FlightDistanceBand.LONG_HAUL,
    ...     CabinClass.BUSINESS,
    ... )
    >>> factor["with_rf"]
    Decimal('0.21932')

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-006
"""

import logging
import threading
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.agents.mrv.business_travel.models import (
    TransportMode,
    FlightDistanceBand,
    CabinClass,
    RailType,
    RoadVehicleType,
    FuelType,
    BusType,
    FerryType,
    HotelClass,
    EFSource,
    AIR_EMISSION_FACTORS,
    CABIN_CLASS_MULTIPLIERS,
    RAIL_EMISSION_FACTORS,
    ROAD_VEHICLE_EMISSION_FACTORS,
    FUEL_EMISSION_FACTORS,
    BUS_EMISSION_FACTORS,
    FERRY_EMISSION_FACTORS,
    HOTEL_EMISSION_FACTORS,
    HOTEL_CLASS_MULTIPLIERS,
    EEIO_FACTORS,
    AIRPORT_DATABASE,
    CURRENCY_RATES,
    CPI_DEFLATORS,
    CurrencyCode,
    calculate_provenance_hash,
)
from greenlang.agents.mrv.business_travel.config import get_config
from greenlang.agents.mrv.business_travel.metrics import get_metrics

logger = logging.getLogger(__name__)

# Quantization constant: 8 decimal places
_QUANT_8DP = Decimal("0.00000001")


# =============================================================================
# ENGINE CLASS
# =============================================================================


class BusinessTravelDatabaseEngine:
    """
    Thread-safe singleton engine for emission factor lookups and classification.

    Provides deterministic, zero-hallucination factor retrieval for all
    business travel transport modes and accommodation. Every lookup is
    recorded via Prometheus metrics (gl_bt_factor_selections_total) and
    returns data suitable for provenance hashing.

    This engine does NOT perform any LLM calls. All factors are retrieved
    from validated, frozen constant tables defined in models.py.

    Thread Safety:
        Uses the __new__ singleton pattern with threading.Lock to ensure
        only one instance is created across all threads.

    Attributes:
        _config: Singleton configuration from get_config()
        _metrics: Singleton metrics from get_metrics()
        _lookup_count: Total number of factor lookups performed

    Example:
        >>> engine = BusinessTravelDatabaseEngine()
        >>> air_ef = engine.get_air_emission_factor(FlightDistanceBand.LONG_HAUL)
        >>> rail_ef = engine.get_rail_emission_factor(RailType.EUROSTAR)
        >>> hotel_ef = engine.get_hotel_emission_factor("GB", HotelClass.UPSCALE)
    """

    _instance: Optional["BusinessTravelDatabaseEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "BusinessTravelDatabaseEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the database engine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._config = get_config()
        self._metrics = get_metrics()
        self._lookup_count: int = 0
        self._lookup_lock: threading.Lock = threading.Lock()

        logger.info(
            "BusinessTravelDatabaseEngine initialized: "
            "airports=%d, air_bands=%d, rail_types=%d, road_types=%d, "
            "fuel_types=%d, bus_types=%d, ferry_types=%d, hotel_countries=%d, "
            "eeio_codes=%d",
            len(AIRPORT_DATABASE),
            len(AIR_EMISSION_FACTORS),
            len(RAIL_EMISSION_FACTORS),
            len(ROAD_VEHICLE_EMISSION_FACTORS),
            len(FUEL_EMISSION_FACTORS),
            len(BUS_EMISSION_FACTORS),
            len(FERRY_EMISSION_FACTORS),
            len(HOTEL_EMISSION_FACTORS),
            len(EEIO_FACTORS),
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_lookup(self) -> None:
        """Increment the lookup counter in a thread-safe manner."""
        with self._lookup_lock:
            self._lookup_count += 1

    def _record_factor_selection(self, source: str, mode: str) -> None:
        """
        Record a factor selection in Prometheus metrics.

        Args:
            source: EF source identifier (e.g., "defra", "epa", "eeio")
            mode: Transport mode (e.g., "air", "rail", "road", "hotel")
        """
        try:
            self._metrics.record_factor_selection(source=source, mode=mode)
        except Exception as exc:
            logger.warning(
                "Failed to record factor selection metric: %s", exc
            )

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
    # AIR EMISSION FACTORS
    # =========================================================================

    def get_air_emission_factor(
        self,
        distance_band: FlightDistanceBand,
        cabin_class: CabinClass = CabinClass.ECONOMY,
        source: EFSource = EFSource.DEFRA,
    ) -> Dict[str, Decimal]:
        """
        Get air emission factors for a given distance band and cabin class.

        Returns emission factors per passenger-km (kgCO2e/pkm) including:
        - without_rf: CO2-only combustion factor (no radiative forcing)
        - with_rf: Factor including radiative forcing uplift (DEFRA default)
        - wtt: Well-to-tank upstream factor
        - class_multiplier: Cabin class multiplier relative to economy

        Args:
            distance_band: DEFRA flight distance band classification.
            cabin_class: Aircraft cabin class (default ECONOMY).
            source: Emission factor source (default DEFRA).

        Returns:
            Dict with keys: without_rf, with_rf, wtt, class_multiplier.

        Raises:
            ValueError: If distance_band is not found in the factor database.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> ef = engine.get_air_emission_factor(
            ...     FlightDistanceBand.LONG_HAUL,
            ...     CabinClass.BUSINESS,
            ... )
            >>> ef["class_multiplier"]
            Decimal('2.9')
        """
        self._increment_lookup()

        band_factors = AIR_EMISSION_FACTORS.get(distance_band)
        if band_factors is None:
            raise ValueError(
                f"Air emission factors not found for distance band "
                f"'{distance_band.value}'. Available bands: "
                f"{[b.value for b in FlightDistanceBand]}"
            )

        class_multiplier = CABIN_CLASS_MULTIPLIERS.get(
            cabin_class, Decimal("1.0")
        )

        result = {
            "without_rf": self._quantize(band_factors["without_rf"]),
            "with_rf": self._quantize(band_factors["with_rf"]),
            "wtt": self._quantize(band_factors["wtt"]),
            "class_multiplier": self._quantize(class_multiplier),
        }

        self._record_factor_selection(source.value, "air")

        logger.debug(
            "Air EF lookup: band=%s, class=%s, source=%s, "
            "without_rf=%s, with_rf=%s, wtt=%s, multiplier=%s",
            distance_band.value,
            cabin_class.value,
            source.value,
            result["without_rf"],
            result["with_rf"],
            result["wtt"],
            result["class_multiplier"],
        )

        return result

    # =========================================================================
    # RAIL EMISSION FACTORS
    # =========================================================================

    def get_rail_emission_factor(
        self,
        rail_type: RailType,
        source: EFSource = EFSource.DEFRA,
    ) -> Dict[str, Decimal]:
        """
        Get rail emission factors for a given rail service type.

        Returns emission factors per passenger-km (kgCO2e/pkm) including:
        - ttw: Tank-to-wheel direct emissions
        - wtt: Well-to-tank upstream emissions

        Args:
            rail_type: Type of rail service.
            source: Emission factor source (default DEFRA).

        Returns:
            Dict with keys: ttw, wtt.

        Raises:
            ValueError: If rail_type is not found in the factor database.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> ef = engine.get_rail_emission_factor(RailType.EUROSTAR)
            >>> ef["ttw"]
            Decimal('0.00446000')
        """
        self._increment_lookup()

        rail_factors = RAIL_EMISSION_FACTORS.get(rail_type)
        if rail_factors is None:
            raise ValueError(
                f"Rail emission factors not found for rail type "
                f"'{rail_type.value}'. Available types: "
                f"{[r.value for r in RailType]}"
            )

        result = {
            "ttw": self._quantize(rail_factors["ttw"]),
            "wtt": self._quantize(rail_factors["wtt"]),
        }

        self._record_factor_selection(source.value, "rail")

        logger.debug(
            "Rail EF lookup: type=%s, source=%s, ttw=%s, wtt=%s",
            rail_type.value,
            source.value,
            result["ttw"],
            result["wtt"],
        )

        return result

    # =========================================================================
    # ROAD VEHICLE EMISSION FACTORS
    # =========================================================================

    def get_road_emission_factor(
        self,
        vehicle_type: RoadVehicleType,
        source: EFSource = EFSource.DEFRA,
    ) -> Dict[str, Decimal]:
        """
        Get road vehicle emission factors for a given vehicle type.

        Returns emission factors (kgCO2e) including:
        - ef_per_vkm: Emissions per vehicle-km
        - ef_per_pkm: Emissions per passenger-km
        - wtt_per_vkm: Well-to-tank per vehicle-km
        - occupancy: Average occupancy factor

        Args:
            vehicle_type: Road vehicle type / fuel / size category.
            source: Emission factor source (default DEFRA).

        Returns:
            Dict with keys: ef_per_vkm, ef_per_pkm, wtt_per_vkm, occupancy.

        Raises:
            ValueError: If vehicle_type is not found in the factor database.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> ef = engine.get_road_emission_factor(
            ...     RoadVehicleType.CAR_MEDIUM_PETROL
            ... )
            >>> ef["ef_per_vkm"]
            Decimal('0.25594000')
        """
        self._increment_lookup()

        road_factors = ROAD_VEHICLE_EMISSION_FACTORS.get(vehicle_type)
        if road_factors is None:
            raise ValueError(
                f"Road emission factors not found for vehicle type "
                f"'{vehicle_type.value}'. Available types: "
                f"{[v.value for v in RoadVehicleType]}"
            )

        result = {
            "ef_per_vkm": self._quantize(road_factors["ef_per_vkm"]),
            "ef_per_pkm": self._quantize(road_factors["ef_per_pkm"]),
            "wtt_per_vkm": self._quantize(road_factors["wtt_per_vkm"]),
            "occupancy": self._quantize(road_factors["occupancy"]),
        }

        self._record_factor_selection(source.value, "road")

        logger.debug(
            "Road EF lookup: type=%s, source=%s, ef_vkm=%s, ef_pkm=%s, "
            "wtt_vkm=%s, occupancy=%s",
            vehicle_type.value,
            source.value,
            result["ef_per_vkm"],
            result["ef_per_pkm"],
            result["wtt_per_vkm"],
            result["occupancy"],
        )

        return result

    # =========================================================================
    # FUEL EMISSION FACTORS
    # =========================================================================

    def get_fuel_emission_factor(
        self,
        fuel_type: FuelType,
    ) -> Dict[str, Decimal]:
        """
        Get fuel-based emission factors for a given fuel type.

        Returns emission factors per litre (kgCO2e/litre) or per kg for CNG:
        - ef_per_litre: Direct combustion emissions per litre (or per kg)
        - wtt_per_litre: Well-to-tank upstream emissions per litre (or per kg)

        Args:
            fuel_type: Fuel type consumed.

        Returns:
            Dict with keys: ef_per_litre, wtt_per_litre.

        Raises:
            ValueError: If fuel_type is not found in the factor database.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> ef = engine.get_fuel_emission_factor(FuelType.DIESEL)
            >>> ef["ef_per_litre"]
            Decimal('2.70370000')
        """
        self._increment_lookup()

        fuel_factors = FUEL_EMISSION_FACTORS.get(fuel_type)
        if fuel_factors is None:
            raise ValueError(
                f"Fuel emission factors not found for fuel type "
                f"'{fuel_type.value}'. Available types: "
                f"{[f.value for f in FuelType]}"
            )

        result = {
            "ef_per_litre": self._quantize(fuel_factors["ef_per_litre"]),
            "wtt_per_litre": self._quantize(fuel_factors["wtt_per_litre"]),
        }

        self._record_factor_selection("defra", "road")

        logger.debug(
            "Fuel EF lookup: type=%s, ef_per_litre=%s, wtt_per_litre=%s",
            fuel_type.value,
            result["ef_per_litre"],
            result["wtt_per_litre"],
        )

        return result

    # =========================================================================
    # BUS EMISSION FACTORS
    # =========================================================================

    def get_bus_emission_factor(
        self,
        bus_type: BusType,
    ) -> Dict[str, Decimal]:
        """
        Get bus emission factors for a given bus service type.

        Returns emission factors per passenger-km (kgCO2e/pkm):
        - ef: Direct emissions per passenger-km
        - wtt: Well-to-tank upstream emissions per passenger-km

        Args:
            bus_type: Type of bus service (local or coach).

        Returns:
            Dict with keys: ef, wtt.

        Raises:
            ValueError: If bus_type is not found in the factor database.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> ef = engine.get_bus_emission_factor(BusType.COACH)
            >>> ef["ef"]
            Decimal('0.02732000')
        """
        self._increment_lookup()

        bus_factors = BUS_EMISSION_FACTORS.get(bus_type)
        if bus_factors is None:
            raise ValueError(
                f"Bus emission factors not found for bus type "
                f"'{bus_type.value}'. Available types: "
                f"{[b.value for b in BusType]}"
            )

        result = {
            "ef": self._quantize(bus_factors["ef"]),
            "wtt": self._quantize(bus_factors["wtt"]),
        }

        self._record_factor_selection("defra", "bus")

        logger.debug(
            "Bus EF lookup: type=%s, ef=%s, wtt=%s",
            bus_type.value,
            result["ef"],
            result["wtt"],
        )

        return result

    # =========================================================================
    # FERRY EMISSION FACTORS
    # =========================================================================

    def get_ferry_emission_factor(
        self,
        ferry_type: FerryType,
    ) -> Dict[str, Decimal]:
        """
        Get ferry emission factors for a given ferry passenger type.

        Returns emission factors per passenger-km (kgCO2e/pkm):
        - ef: Direct emissions per passenger-km
        - wtt: Well-to-tank upstream emissions per passenger-km

        Args:
            ferry_type: Ferry passenger type (foot or car passenger).

        Returns:
            Dict with keys: ef, wtt.

        Raises:
            ValueError: If ferry_type is not found in the factor database.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> ef = engine.get_ferry_emission_factor(FerryType.FOOT_PASSENGER)
            >>> ef["ef"]
            Decimal('0.01877000')
        """
        self._increment_lookup()

        ferry_factors = FERRY_EMISSION_FACTORS.get(ferry_type)
        if ferry_factors is None:
            raise ValueError(
                f"Ferry emission factors not found for ferry type "
                f"'{ferry_type.value}'. Available types: "
                f"{[f.value for f in FerryType]}"
            )

        result = {
            "ef": self._quantize(ferry_factors["ef"]),
            "wtt": self._quantize(ferry_factors["wtt"]),
        }

        self._record_factor_selection("defra", "ferry")

        logger.debug(
            "Ferry EF lookup: type=%s, ef=%s, wtt=%s",
            ferry_type.value,
            result["ef"],
            result["wtt"],
        )

        return result

    # =========================================================================
    # HOTEL EMISSION FACTORS
    # =========================================================================

    def get_hotel_emission_factor(
        self,
        country_code: str,
        hotel_class: HotelClass = HotelClass.STANDARD,
    ) -> Dict[str, Decimal]:
        """
        Get hotel emission factors for a given country and hotel class.

        Falls back to GLOBAL average if the country code is not found in
        the hotel emission factor database.

        Returns:
        - ef_per_room_night: Base emission factor per room-night (kgCO2e)
        - class_multiplier: Hotel class multiplier relative to standard

        Args:
            country_code: ISO 3166-1 alpha-2 country code (or "GLOBAL").
            hotel_class: Hotel class / tier (default STANDARD).

        Returns:
            Dict with keys: ef_per_room_night, class_multiplier.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> ef = engine.get_hotel_emission_factor("GB", HotelClass.UPSCALE)
            >>> ef["ef_per_room_night"]
            Decimal('12.32000000')
            >>> ef["class_multiplier"]
            Decimal('1.35000000')
        """
        self._increment_lookup()

        code = country_code.upper().strip()

        # Look up country-specific EF, fall back to GLOBAL
        base_ef = HOTEL_EMISSION_FACTORS.get(code)
        fallback_used = False
        if base_ef is None:
            base_ef = HOTEL_EMISSION_FACTORS["GLOBAL"]
            fallback_used = True
            logger.info(
                "Hotel EF: country '%s' not found, falling back to GLOBAL "
                "(ef=%s kgCO2e/room-night)",
                code,
                base_ef,
            )

        class_multiplier = HOTEL_CLASS_MULTIPLIERS.get(
            hotel_class, Decimal("1.0")
        )

        result = {
            "ef_per_room_night": self._quantize(base_ef),
            "class_multiplier": self._quantize(class_multiplier),
        }

        self._record_factor_selection("defra", "hotel")

        logger.debug(
            "Hotel EF lookup: country=%s (fallback=%s), class=%s, "
            "ef=%s, multiplier=%s",
            code,
            fallback_used,
            hotel_class.value,
            result["ef_per_room_night"],
            result["class_multiplier"],
        )

        return result

    # =========================================================================
    # EEIO FACTORS (SPEND-BASED)
    # =========================================================================

    def get_eeio_factor(
        self,
        naics_code: str,
    ) -> Dict[str, Any]:
        """
        Get EEIO emission factor for a given NAICS code.

        EEIO (Environmentally Extended Input-Output) factors are used for
        spend-based calculation when activity data is unavailable.

        Returns:
        - name: Human-readable name for the NAICS category
        - ef_per_usd: Emission factor in kgCO2e per USD (2021 base year)

        Args:
            naics_code: NAICS industry classification code string.

        Returns:
            Dict with keys: name, ef_per_usd.

        Raises:
            ValueError: If naics_code is not found in the EEIO factor database.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> ef = engine.get_eeio_factor("481000")
            >>> ef["name"]
            'Air transportation'
            >>> ef["ef_per_usd"]
            Decimal('0.47700000')
        """
        self._increment_lookup()

        eeio_entry = EEIO_FACTORS.get(naics_code)
        if eeio_entry is None:
            raise ValueError(
                f"EEIO factor not found for NAICS code '{naics_code}'. "
                f"Available codes: {sorted(EEIO_FACTORS.keys())}"
            )

        result = {
            "name": eeio_entry["name"],
            "ef_per_usd": self._quantize(eeio_entry["ef"]),
        }

        self._record_factor_selection("eeio", "road")

        logger.debug(
            "EEIO factor lookup: naics=%s, name=%s, ef_per_usd=%s",
            naics_code,
            result["name"],
            result["ef_per_usd"],
        )

        return result

    # =========================================================================
    # AIRPORT DATABASE
    # =========================================================================

    def lookup_airport(
        self,
        iata_code: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Look up airport metadata by IATA code.

        Args:
            iata_code: 3-letter IATA airport code (case-insensitive).

        Returns:
            Dict with keys: name, lat, lon, country; or None if not found.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> airport = engine.lookup_airport("JFK")
            >>> airport["name"]
            'John F. Kennedy International'
            >>> airport["lat"]
            Decimal('40.6413')
        """
        code = iata_code.upper().strip()
        airport_data = AIRPORT_DATABASE.get(code)

        if airport_data is None:
            logger.warning("Airport not found for IATA code '%s'", code)
            return None

        result = {
            "name": airport_data["name"],
            "lat": airport_data["lat"],
            "lon": airport_data["lon"],
            "country": airport_data["country"],
        }

        logger.debug(
            "Airport lookup: code=%s, name=%s, country=%s",
            code,
            result["name"],
            result["country"],
        )

        return result

    def search_airports(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Search airports by name or IATA code (case-insensitive substring).

        Args:
            query: Search query (partial airport name or IATA code).

        Returns:
            List of matching airport dicts, each with keys:
            iata, name, lat, lon, country. Empty list if no matches.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> results = engine.search_airports("london")
            >>> len(results) >= 2  # LHR and LGW
            True
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        matches: List[Dict[str, Any]] = []

        for iata_code, airport_data in AIRPORT_DATABASE.items():
            name_lower = airport_data["name"].lower()
            code_lower = iata_code.lower()

            if query_lower in name_lower or query_lower in code_lower:
                matches.append({
                    "iata": iata_code,
                    "name": airport_data["name"],
                    "lat": airport_data["lat"],
                    "lon": airport_data["lon"],
                    "country": airport_data["country"],
                })

        logger.debug(
            "Airport search: query='%s', matches=%d",
            query,
            len(matches),
        )

        return matches

    # =========================================================================
    # CABIN CLASS MULTIPLIER
    # =========================================================================

    def get_cabin_class_multiplier(
        self,
        cabin_class: CabinClass,
    ) -> Decimal:
        """
        Get the cabin class multiplier relative to economy class.

        Cabin class multipliers allocate per-passenger emissions based on
        the floor space occupied by each seat class. Business and first
        class seats occupy more space, resulting in higher per-passenger
        emissions.

        Args:
            cabin_class: Aircraft cabin class.

        Returns:
            Decimal multiplier (1.0 for economy).

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> engine.get_cabin_class_multiplier(CabinClass.BUSINESS)
            Decimal('2.90000000')
        """
        multiplier = CABIN_CLASS_MULTIPLIERS.get(
            cabin_class, Decimal("1.0")
        )
        return self._quantize(multiplier)

    # =========================================================================
    # CURRENCY CONVERSION
    # =========================================================================

    def get_currency_rate(
        self,
        currency: CurrencyCode,
    ) -> Decimal:
        """
        Get the exchange rate from a given currency to USD.

        Args:
            currency: ISO 4217 currency code.

        Returns:
            Exchange rate to USD (e.g., 1.265 for GBP means 1 GBP = 1.265 USD).

        Raises:
            ValueError: If currency is not found in the rate table.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> engine.get_currency_rate(CurrencyCode.GBP)
            Decimal('1.26500000')
        """
        rate = CURRENCY_RATES.get(currency)
        if rate is None:
            raise ValueError(
                f"Currency rate not found for '{currency.value}'. "
                f"Available currencies: "
                f"{[c.value for c in CurrencyCode]}"
            )
        return self._quantize(rate)

    # =========================================================================
    # CPI DEFLATION
    # =========================================================================

    def get_cpi_deflator(
        self,
        year: int,
    ) -> Decimal:
        """
        Get the CPI deflator for a given year (base year 2021 = 1.0).

        Used to convert nominal spend values to real (2021 base year) USD
        for consistent EEIO factor application across reporting years.

        Args:
            year: Year of the spend data.

        Returns:
            CPI deflator value.

        Raises:
            ValueError: If year is not found in the CPI deflator table.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> engine.get_cpi_deflator(2024)
            Decimal('1.14900000')
        """
        deflator = CPI_DEFLATORS.get(year)
        if deflator is None:
            raise ValueError(
                f"CPI deflator not available for year {year}. "
                f"Available years: {sorted(CPI_DEFLATORS.keys())}"
            )
        return self._quantize(deflator)

    # =========================================================================
    # AVAILABLE OPTIONS
    # =========================================================================

    def get_available_transport_modes(self) -> List[str]:
        """
        Get list of all available transport modes.

        Returns:
            List of transport mode value strings.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> modes = engine.get_available_transport_modes()
            >>> "air" in modes
            True
        """
        return [mode.value for mode in TransportMode]

    def get_available_cabin_classes(self) -> List[Dict[str, Any]]:
        """
        Get list of all available cabin classes with their multipliers.

        Returns:
            List of dicts, each with keys: cabin_class, multiplier.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> classes = engine.get_available_cabin_classes()
            >>> len(classes)
            4
        """
        result: List[Dict[str, Any]] = []
        for cabin_class in CabinClass:
            multiplier = CABIN_CLASS_MULTIPLIERS.get(
                cabin_class, Decimal("1.0")
            )
            result.append({
                "cabin_class": cabin_class.value,
                "multiplier": self._quantize(multiplier),
            })
        return result

    # =========================================================================
    # TRANSPORT MODE CLASSIFICATION
    # =========================================================================

    def classify_transport_mode(
        self,
        trip_data: dict,
    ) -> TransportMode:
        """
        Classify the transport mode from trip input data.

        Examines the trip_data dictionary for mode-identifying keys:
        - "origin_iata" or "destination_iata" -> AIR
        - "rail_type" -> RAIL
        - "bus_type" -> BUS
        - "ferry_type" -> FERRY
        - "taxi_type" -> TAXI
        - "vehicle_type" containing "motorcycle" -> MOTORCYCLE
        - "vehicle_type" -> ROAD
        - "room_nights" or "hotel_class" -> HOTEL
        - "mode" field (explicit) -> direct mapping

        Args:
            trip_data: Dictionary of trip input fields.

        Returns:
            Classified TransportMode enum value.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> mode = engine.classify_transport_mode(
            ...     {"origin_iata": "JFK", "destination_iata": "LHR"}
            ... )
            >>> mode == TransportMode.AIR
            True
        """
        if not trip_data:
            logger.warning("Empty trip_data for classification, defaulting to ROAD")
            return TransportMode.ROAD

        # Check for explicit mode field
        explicit_mode = trip_data.get("mode")
        if explicit_mode is not None:
            mode_str = str(explicit_mode).lower().strip()
            for transport_mode in TransportMode:
                if transport_mode.value == mode_str:
                    logger.debug(
                        "Classified transport mode from explicit field: %s",
                        transport_mode.value,
                    )
                    return transport_mode

        # Check for air travel indicators
        if "origin_iata" in trip_data or "destination_iata" in trip_data:
            logger.debug("Classified transport mode as AIR (IATA codes present)")
            return TransportMode.AIR

        # Check for rail indicators
        if "rail_type" in trip_data:
            logger.debug("Classified transport mode as RAIL (rail_type present)")
            return TransportMode.RAIL

        # Check for bus indicators
        if "bus_type" in trip_data:
            logger.debug("Classified transport mode as BUS (bus_type present)")
            return TransportMode.BUS

        # Check for ferry indicators
        if "ferry_type" in trip_data:
            logger.debug("Classified transport mode as FERRY (ferry_type present)")
            return TransportMode.FERRY

        # Check for taxi indicators
        if "taxi_type" in trip_data:
            logger.debug("Classified transport mode as TAXI (taxi_type present)")
            return TransportMode.TAXI

        # Check for hotel indicators
        if "room_nights" in trip_data or "hotel_class" in trip_data:
            logger.debug("Classified transport mode as HOTEL (hotel fields present)")
            return TransportMode.HOTEL

        # Check for motorcycle in vehicle_type
        vehicle_type = trip_data.get("vehicle_type", "")
        if isinstance(vehicle_type, str) and "motorcycle" in vehicle_type.lower():
            logger.debug(
                "Classified transport mode as MOTORCYCLE (vehicle_type contains motorcycle)"
            )
            return TransportMode.MOTORCYCLE

        # Check for generic vehicle_type (road)
        if "vehicle_type" in trip_data:
            logger.debug(
                "Classified transport mode as ROAD (vehicle_type present)"
            )
            return TransportMode.ROAD

        # Check for fuel-based road indicators
        if "fuel_type" in trip_data or "litres" in trip_data:
            logger.debug(
                "Classified transport mode as ROAD (fuel data present)"
            )
            return TransportMode.ROAD

        # Default to ROAD
        logger.info(
            "Could not classify transport mode from trip_data keys %s, "
            "defaulting to ROAD",
            list(trip_data.keys()),
        )
        return TransportMode.ROAD

    # =========================================================================
    # SUMMARY AND STATS
    # =========================================================================

    def get_lookup_count(self) -> int:
        """
        Get the total number of factor lookups performed.

        Returns:
            Integer count of lookups.
        """
        with self._lookup_lock:
            return self._lookup_count

    def get_database_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the database contents.

        Returns:
            Dict with counts of all factor categories.

        Example:
            >>> engine = BusinessTravelDatabaseEngine()
            >>> summary = engine.get_database_summary()
            >>> summary["airport_count"]
            50
        """
        return {
            "airport_count": len(AIRPORT_DATABASE),
            "air_distance_bands": len(AIR_EMISSION_FACTORS),
            "cabin_classes": len(CABIN_CLASS_MULTIPLIERS),
            "rail_types": len(RAIL_EMISSION_FACTORS),
            "road_vehicle_types": len(ROAD_VEHICLE_EMISSION_FACTORS),
            "fuel_types": len(FUEL_EMISSION_FACTORS),
            "bus_types": len(BUS_EMISSION_FACTORS),
            "ferry_types": len(FERRY_EMISSION_FACTORS),
            "hotel_countries": len(HOTEL_EMISSION_FACTORS),
            "hotel_classes": len(HOTEL_CLASS_MULTIPLIERS),
            "eeio_naics_codes": len(EEIO_FACTORS),
            "currencies": len(CURRENCY_RATES),
            "cpi_years": len(CPI_DEFLATORS),
            "transport_modes": len(TransportMode),
            "total_lookups": self.get_lookup_count(),
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

_engine_instance: Optional[BusinessTravelDatabaseEngine] = None
_engine_lock: threading.Lock = threading.Lock()


def get_database_engine() -> BusinessTravelDatabaseEngine:
    """
    Get the singleton BusinessTravelDatabaseEngine instance.

    Thread-safe accessor for the global database engine instance.

    Returns:
        BusinessTravelDatabaseEngine singleton instance.

    Example:
        >>> engine = get_database_engine()
        >>> ef = engine.get_air_emission_factor(FlightDistanceBand.LONG_HAUL)
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = BusinessTravelDatabaseEngine()
        return _engine_instance


def reset_database_engine() -> None:
    """
    Reset the module-level engine instance (for testing only).

    Warning: This function is intended for use in test fixtures only.
    Do not call in production code.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    BusinessTravelDatabaseEngine.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "BusinessTravelDatabaseEngine",
    "get_database_engine",
    "reset_database_engine",
]
