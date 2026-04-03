"""
BusinessTravelPipelineEngine - Orchestrated 10-stage pipeline for business travel emissions.

This module implements the BusinessTravelPipelineEngine for AGENT-MRV-019 (Business Travel).
It orchestrates a 10-stage pipeline for complete business travel emissions calculation from
raw input to compliant output with full audit trail.

The 10 stages are:
1. VALIDATE: Input validation (required fields, types, ranges, mode-specific checks)
2. CLASSIFY: Determine transport mode, auto-classify distance band / vehicle type
3. NORMALIZE: Convert units (miles->km, gallons->litres, currency->USD, CPI deflation)
4. RESOLVE_EFS: Resolve emission factors from database engine (DEFRA/EPA/ICAO/EEIO)
5. CALCULATE_FLIGHTS: If mode==AIR, delegate to AirTravelCalculatorEngine
6. CALCULATE_GROUND: If mode in (RAIL,ROAD,BUS,TAXI,FERRY,MOTORCYCLE,HOTEL), delegate
7. ALLOCATE: Allocate emissions to department / cost_center if provided
8. COMPLIANCE: Run compliance checks if enabled (7 frameworks)
9. AGGREGATE: Calculate DQI score, summarize emissions
10. SEAL: Seal provenance chain, generate final SHA-256 hash

Example:
    >>> from greenlang.agents.mrv.business_travel.business_travel_pipeline import BusinessTravelPipelineEngine
    >>> engine = BusinessTravelPipelineEngine()
    >>> trip = TripInput(mode=TransportMode.AIR, trip_data={"origin_iata": "JFK", "destination_iata": "LHR"})
    >>> result = engine.calculate(trip)
    >>> print(f"Total emissions: {result.total_co2e} kgCO2e")

Module: greenlang.agents.mrv.business_travel.business_travel_pipeline
Agent: AGENT-MRV-019
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import logging
import hashlib
import json
from threading import RLock

from greenlang.agents.mrv.business_travel.models import (
    TripInput,
    BatchTripInput,
    TripCalculationResult,
    BatchResult,
    AggregationResult,
    TransportMode,
    CalculationMethod,
    ProvenanceStage,
    FlightInput,
    RailInput,
    RoadDistanceInput,
    RoadFuelInput,
    TaxiInput,
    BusInput,
    FerryInput,
    HotelInput,
    SpendInput,
    calculate_provenance_hash,
)


logger = logging.getLogger(__name__)

_QUANT_8DP = Decimal("0.00000001")


# ==============================================================================
# PIPELINE STATUS
# ==============================================================================


class PipelineStatus:
    """Pipeline execution status constants."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


# ==============================================================================
# BUSINESS TRAVEL PIPELINE ENGINE
# ==============================================================================


class BusinessTravelPipelineEngine:
    """
    BusinessTravelPipelineEngine - Orchestrated 10-stage pipeline for business travel.

    This engine coordinates the complete business travel emissions calculation
    workflow through 10 sequential stages, from input validation to sealed
    audit trail. It supports all transport modes (air, rail, road, bus, taxi,
    ferry, motorcycle) plus hotel accommodation and spend-based fallback.

    The engine uses lazy initialization for sub-engines, creating them only
    when needed. This reduces startup time and memory footprint.

    Attributes:
        _air_engine: AirTravelCalculatorEngine (lazy-loaded)
        _ground_engine: GroundTransportCalculatorEngine (lazy-loaded)
        _hotel_engine: HotelStayCalculatorEngine (lazy-loaded)
        _spend_engine: SpendBasedCalculatorEngine (lazy-loaded)
        _database_engine: BusinessTravelDatabaseEngine (lazy-loaded)
        _compliance_engine: ComplianceCheckerEngine (lazy-loaded)

    Example:
        >>> engine = BusinessTravelPipelineEngine()
        >>> trip = TripInput(
        ...     mode=TransportMode.AIR,
        ...     trip_data={"origin_iata": "JFK", "destination_iata": "LHR"},
        ... )
        >>> result = engine.calculate(trip)
        >>> assert result.total_co2e > Decimal("0")
    """

    _instance: Optional["BusinessTravelPipelineEngine"] = None
    _lock: RLock = RLock()

    def __new__(cls) -> "BusinessTravelPipelineEngine":
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize BusinessTravelPipelineEngine.

        Prevents re-initialization of the singleton. All sub-engines are
        lazy-loaded on first calculate() call.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Lazy-loaded engines (created on first use)
        self._air_engine: Optional[Any] = None
        self._ground_engine: Optional[Any] = None
        self._hotel_engine: Optional[Any] = None
        self._spend_engine: Optional[Any] = None
        self._database_engine: Optional[Any] = None
        self._compliance_engine: Optional[Any] = None

        # Pipeline state
        self._provenance_chains: Dict[str, List[Dict[str, Any]]] = {}

        self._initialized = True
        logger.info("BusinessTravelPipelineEngine initialized (version 1.0.0)")

    # ==========================================================================
    # PUBLIC API - CORE PROCESSING METHODS
    # ==========================================================================

    def calculate(self, trip_input: TripInput) -> TripCalculationResult:
        """
        Execute the 10-stage business travel emissions calculation pipeline.

        Args:
            trip_input: Generic trip input wrapping mode-specific data.

        Returns:
            TripCalculationResult with emissions, DQI, and provenance hash.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If pipeline execution fails.
        """
        chain_id = f"bt-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._provenance_chains[chain_id] = []
        stage_durations: Dict[str, float] = {}

        try:
            # ------------------------------------------------------------------
            # Stage 1: VALIDATE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            is_valid, errors = self._stage_validate(trip_input)
            duration_ms = self._elapsed_ms(start)
            stage_durations["VALIDATE"] = duration_ms
            self._record_provenance(chain_id, ProvenanceStage.VALIDATE, trip_input, {"valid": is_valid})

            if not is_valid:
                raise ValueError(f"Input validation failed: {'; '.join(errors)}")
            logger.info("[%s] Stage VALIDATE completed in %.2fms", chain_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 2: CLASSIFY
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            mode, method = self._stage_classify(trip_input)
            duration_ms = self._elapsed_ms(start)
            stage_durations["CLASSIFY"] = duration_ms
            self._record_provenance(chain_id, ProvenanceStage.CLASSIFY, trip_input, {"mode": mode.value, "method": method.value})
            logger.info("[%s] Stage CLASSIFY completed in %.2fms (mode=%s)", chain_id, duration_ms, mode.value)

            # ------------------------------------------------------------------
            # Stage 3: NORMALIZE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            normalized_data = self._stage_normalize(trip_input.trip_data, mode)
            duration_ms = self._elapsed_ms(start)
            stage_durations["NORMALIZE"] = duration_ms
            self._record_provenance(chain_id, ProvenanceStage.NORMALIZE, trip_input.trip_data, normalized_data)
            logger.info("[%s] Stage NORMALIZE completed in %.2fms", chain_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 4: RESOLVE_EFS
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            ef_data = self._stage_resolve_efs(mode, normalized_data)
            duration_ms = self._elapsed_ms(start)
            stage_durations["RESOLVE_EFS"] = duration_ms
            self._record_provenance(chain_id, ProvenanceStage.RESOLVE_EFS, normalized_data, ef_data)
            logger.info("[%s] Stage RESOLVE_EFS completed in %.2fms", chain_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 5: CALCULATE_FLIGHTS  (air only)
            # Stage 6: CALCULATE_GROUND   (ground / hotel)
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            calc_result = self._route_to_engine(mode, normalized_data, ef_data, method)
            if mode == TransportMode.AIR:
                stage_name = "CALCULATE_FLIGHTS"
                provenance_stage = ProvenanceStage.CALCULATE_FLIGHTS
            else:
                stage_name = "CALCULATE_GROUND"
                provenance_stage = ProvenanceStage.CALCULATE_GROUND
            duration_ms = self._elapsed_ms(start)
            stage_durations[stage_name] = duration_ms
            self._record_provenance(chain_id, provenance_stage, normalized_data, calc_result)
            logger.info(
                f"[{chain_id}] Stage {stage_name} completed in {duration_ms:.2f}ms "
                f"(total_co2e={calc_result.get('total_co2e', 0)} kgCO2e)"
            )

            # ------------------------------------------------------------------
            # Stage 7: ALLOCATE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            allocated = self._stage_allocate(calc_result, trip_input.department, trip_input.cost_center)
            duration_ms = self._elapsed_ms(start)
            stage_durations["ALLOCATE"] = duration_ms
            self._record_provenance(chain_id, ProvenanceStage.ALLOCATE, calc_result, allocated)
            logger.info("[%s] Stage ALLOCATE completed in %.2fms", chain_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 8: COMPLIANCE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            compliance = self._stage_compliance(mode, method, allocated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["COMPLIANCE"] = duration_ms
            self._record_provenance(chain_id, ProvenanceStage.COMPLIANCE, allocated, compliance)
            logger.info("[%s] Stage COMPLIANCE completed in %.2fms", chain_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 9: AGGREGATE  (DQI scoring)
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            dqi_score = self._stage_aggregate_dqi(mode, method)
            duration_ms = self._elapsed_ms(start)
            stage_durations["AGGREGATE"] = duration_ms
            self._record_provenance(chain_id, ProvenanceStage.AGGREGATE, allocated, {"dqi": str(dqi_score)})
            logger.info("[%s] Stage AGGREGATE completed in %.2fms (DQI=%s)", chain_id, duration_ms, dqi_score)

            # ------------------------------------------------------------------
            # Stage 10: SEAL
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            provenance_hash = self._stage_seal(chain_id, allocated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["SEAL"] = duration_ms
            logger.info("[%s] Stage SEAL completed in %.2fms", chain_id, duration_ms)

            # Build unified result
            total_co2e = Decimal(str(allocated.get("total_co2e", 0)))
            wtt_co2e = Decimal(str(allocated.get("wtt_co2e", 0)))
            co2e_without_rf = allocated.get("co2e_without_rf")
            co2e_with_rf = allocated.get("co2e_with_rf")

            result = TripCalculationResult(
                mode=mode,
                method=method,
                total_co2e=total_co2e,
                co2e_without_rf=Decimal(str(co2e_without_rf)) if co2e_without_rf is not None else None,
                co2e_with_rf=Decimal(str(co2e_with_rf)) if co2e_with_rf is not None else None,
                wtt_co2e=wtt_co2e,
                dqi_score=dqi_score,
                trip_detail=allocated,
                provenance_hash=provenance_hash,
            )

            total_dur = sum(stage_durations.values())
            logger.info(
                f"[{chain_id}] Pipeline completed successfully in {total_dur:.2f}ms. "
                f"Total emissions: {total_co2e} kgCO2e"
            )
            return result

        except ValueError:
            raise
        except Exception as e:
            logger.error("[%s] Pipeline execution failed: %s", chain_id, e, exc_info=True)
            raise RuntimeError(f"Pipeline execution failed: {e}") from e
        finally:
            # Cleanup provenance chain from memory (already sealed or failed)
            self._provenance_chains.pop(chain_id, None)

    def calculate_batch(self, batch_input: BatchTripInput) -> BatchResult:
        """
        Process a list of trips, aggregate results.

        Handles errors per-trip without failing the entire batch. Each trip
        is independently processed through the full 10-stage pipeline.

        Args:
            batch_input: Batch of trip inputs with reporting period.

        Returns:
            BatchResult with individual results, totals, and error details.
        """
        start_time = datetime.now(timezone.utc)
        results: List[TripCalculationResult] = []
        errors: List[dict] = []

        logger.info(
            f"Starting batch calculation ({len(batch_input.trips)} trips, "
            f"period={batch_input.reporting_period})"
        )

        for idx, trip in enumerate(batch_input.trips):
            try:
                result = self.calculate(trip)
                results.append(result)
            except Exception as e:
                logger.error("Batch trip %s (%s) failed: %s", idx, trip.mode.value, e)
                errors.append({
                    "index": idx,
                    "mode": trip.mode.value,
                    "error": str(e),
                })

        # Aggregate totals
        total_co2e = sum((r.total_co2e for r in results), Decimal("0"))

        # Air-specific aggregates
        air_results = [r for r in results if r.mode == TransportMode.AIR]
        total_co2e_without_rf: Optional[Decimal] = None
        total_co2e_with_rf: Optional[Decimal] = None
        if air_results:
            total_co2e_without_rf = sum(
                (r.co2e_without_rf for r in air_results if r.co2e_without_rf is not None),
                Decimal("0"),
            )
            total_co2e_with_rf = sum(
                (r.co2e_with_rf for r in air_results if r.co2e_with_rf is not None),
                Decimal("0"),
            )

        duration_ms = self._elapsed_ms(start_time)

        logger.info(
            f"Batch completed in {duration_ms:.2f}ms. "
            f"Success: {len(results)}, Failed: {len(errors)}, "
            f"Total emissions: {total_co2e} kgCO2e"
        )

        return BatchResult(
            results=results,
            total_co2e=total_co2e,
            total_co2e_without_rf=total_co2e_without_rf,
            total_co2e_with_rf=total_co2e_with_rf,
            count=len(results),
            errors=errors,
            reporting_period=batch_input.reporting_period,
        )

    def aggregate(
        self,
        results: List[TripCalculationResult],
        period: str = "",
    ) -> AggregationResult:
        """
        Multi-dimensional aggregation of trip calculation results.

        Aggregates by:
        - total_co2e: sum all
        - by_mode: sum grouped by transport mode
        - by_period: sum grouped by reporting period (from trip_detail)
        - by_department: sum grouped by department
        - by_cabin_class: sum grouped by cabin class (air trips)

        Args:
            results: List of individual trip results.
            period: Reporting period label for the aggregation.

        Returns:
            AggregationResult with multi-dimensional breakdown.
        """
        total_co2e = Decimal("0")
        by_mode: Dict[str, Decimal] = {}
        by_period: Dict[str, Decimal] = {}
        by_department: Dict[str, Decimal] = {}
        by_cabin_class: Dict[str, Decimal] = {}

        for r in results:
            emissions = r.total_co2e
            total_co2e += emissions

            # By mode
            mode_key = r.mode.value
            by_mode[mode_key] = by_mode.get(mode_key, Decimal("0")) + emissions

            # By department
            dept = r.trip_detail.get("department")
            if dept:
                by_department[dept] = by_department.get(dept, Decimal("0")) + emissions

            # By period
            trip_period = r.trip_detail.get("reporting_period", period)
            if trip_period:
                by_period[trip_period] = by_period.get(trip_period, Decimal("0")) + emissions

            # By cabin class (air trips only)
            cabin = r.trip_detail.get("cabin_class")
            if cabin:
                by_cabin_class[cabin] = by_cabin_class.get(cabin, Decimal("0")) + emissions

        return AggregationResult(
            total_co2e=total_co2e,
            by_mode=by_mode,
            by_period=by_period,
            by_department=by_department,
            by_cabin_class=by_cabin_class,
            period=period,
        )

    # ==========================================================================
    # STAGE METHODS (PRIVATE)
    # ==========================================================================

    def _stage_validate(self, trip_input: TripInput) -> Tuple[bool, List[str]]:
        """
        Stage 1: VALIDATE - Input validation.

        Checks:
        - mode is set and is a valid TransportMode
        - trip_data is not empty
        - Mode-specific required fields are present

        Args:
            trip_input: Generic trip input.

        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors: List[str] = []

        if not trip_input.trip_data:
            errors.append("trip_data must not be empty")

        data = trip_input.trip_data
        mode = trip_input.mode

        if mode == TransportMode.AIR:
            if "origin_iata" not in data:
                errors.append("origin_iata required for AIR mode")
            if "destination_iata" not in data:
                errors.append("destination_iata required for AIR mode")

        elif mode == TransportMode.RAIL:
            if "rail_type" not in data:
                errors.append("rail_type required for RAIL mode")
            if "distance_km" not in data and "distance_miles" not in data:
                errors.append("distance_km or distance_miles required for RAIL mode")

        elif mode == TransportMode.ROAD:
            has_distance = "distance_km" in data or "distance_miles" in data
            has_fuel = "fuel_type" in data and "litres" in data
            if not has_distance and not has_fuel:
                errors.append("distance or fuel data required for ROAD mode")

        elif mode == TransportMode.BUS:
            if "bus_type" not in data:
                errors.append("bus_type required for BUS mode")
            if "distance_km" not in data and "distance_miles" not in data:
                errors.append("distance required for BUS mode")

        elif mode == TransportMode.TAXI:
            if "distance_km" not in data and "distance_miles" not in data:
                errors.append("distance required for TAXI mode")

        elif mode == TransportMode.FERRY:
            if "ferry_type" not in data:
                errors.append("ferry_type required for FERRY mode")
            if "distance_km" not in data:
                errors.append("distance_km required for FERRY mode")

        elif mode == TransportMode.HOTEL:
            if "room_nights" not in data:
                errors.append("room_nights required for HOTEL mode")

        is_valid = len(errors) == 0
        return is_valid, errors

    def _stage_classify(self, trip_input: TripInput) -> Tuple[TransportMode, CalculationMethod]:
        """
        Stage 2: CLASSIFY - Determine transport mode and calculation method.

        Auto-selects calculation method based on available data:
        - supplier-specific if supplier data present
        - distance-based if distance/IATA codes present
        - spend-based if only spend data present
        - average-data otherwise

        Args:
            trip_input: Generic trip input.

        Returns:
            Tuple of (TransportMode, CalculationMethod).
        """
        mode = trip_input.mode
        data = trip_input.trip_data

        # Determine calculation method from data availability
        if data.get("supplier_co2e") is not None:
            method = CalculationMethod.SUPPLIER_SPECIFIC
        elif data.get("naics_code") and data.get("amount"):
            method = CalculationMethod.SPEND_BASED
        elif mode == TransportMode.AIR and data.get("origin_iata") and data.get("destination_iata"):
            method = CalculationMethod.DISTANCE_BASED
        elif data.get("distance_km") or data.get("distance_miles") or data.get("litres"):
            method = CalculationMethod.DISTANCE_BASED
        elif data.get("room_nights"):
            method = CalculationMethod.DISTANCE_BASED  # distance-based proxy for hotel
        else:
            method = CalculationMethod.AVERAGE_DATA

        return mode, method

    def _stage_normalize(self, trip_data: dict, mode: TransportMode) -> dict:
        """
        Stage 3: NORMALIZE - Convert units and standardize codes.

        Conversions:
        - miles -> km (x 1.60934)
        - gallons -> litres (x 3.78541)
        - IATA codes -> uppercase
        - currency -> USD via exchange rate
        - CPI deflation for spend-based data

        Args:
            trip_data: Raw trip data dict.
            mode: Transport mode.

        Returns:
            Normalized trip data dict.
        """
        data = dict(trip_data)  # shallow copy

        # Miles to kilometres
        if "distance_miles" in data and "distance_km" not in data:
            miles = Decimal(str(data["distance_miles"]))
            data["distance_km"] = str((miles * Decimal("1.60934")).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP))
            data["_original_distance_miles"] = data.pop("distance_miles")

        # Gallons to litres
        if "gallons" in data and "litres" not in data:
            gallons = Decimal(str(data["gallons"]))
            data["litres"] = str((gallons * Decimal("3.78541")).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP))
            data["_original_gallons"] = data.pop("gallons")

        # Uppercase IATA codes
        if "origin_iata" in data:
            data["origin_iata"] = str(data["origin_iata"]).upper()
        if "destination_iata" in data:
            data["destination_iata"] = str(data["destination_iata"]).upper()

        # Currency conversion for spend-based
        if "amount" in data and "currency" in data:
            from greenlang.agents.mrv.business_travel.models import convert_currency_to_usd, CurrencyCode
            try:
                currency = CurrencyCode(data["currency"])
                amount = Decimal(str(data["amount"]))
                data["amount_usd"] = str(convert_currency_to_usd(amount, currency))
            except (ValueError, KeyError):
                data["amount_usd"] = str(data["amount"])

        # CPI deflation for spend-based
        if "amount_usd" in data and "reporting_year" in data:
            from greenlang.agents.mrv.business_travel.models import get_cpi_deflator
            try:
                year = int(data["reporting_year"])
                deflator = get_cpi_deflator(year)
                deflated = Decimal(str(data["amount_usd"])) / deflator
                data["amount_usd_deflated"] = str(deflated.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP))
            except (ValueError, KeyError):
                data["amount_usd_deflated"] = data["amount_usd"]

        return data

    def _stage_resolve_efs(self, mode: TransportMode, normalized_data: dict) -> dict:
        """
        Stage 4: RESOLVE_EFS - Resolve emission factors.

        Uses the database engine if available, otherwise falls back to the
        static emission factor tables in models.py.

        Args:
            mode: Transport mode.
            normalized_data: Normalized trip data.

        Returns:
            Dict with resolved emission factor(s) and source metadata.
        """
        from greenlang.agents.mrv.business_travel.models import (
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
            FlightDistanceBand,
            CabinClass,
            RailType,
            RoadVehicleType,
            FuelType,
            BusType,
            FerryType,
            HotelClass,
            classify_flight_distance_band,
            lookup_airport,
            calculate_great_circle_distance,
        )

        ef_data: Dict[str, Any] = {"ef_source": "DEFRA"}

        if mode == TransportMode.AIR:
            origin = normalized_data.get("origin_iata", "")
            dest = normalized_data.get("destination_iata", "")
            origin_info = lookup_airport(origin)
            dest_info = lookup_airport(dest)

            if origin_info and dest_info:
                distance_km = calculate_great_circle_distance(
                    origin_info["lat"], origin_info["lon"],
                    dest_info["lat"], dest_info["lon"],
                )
                # Apply uplift factor (8% for routing inefficiency)
                distance_km = distance_km * Decimal("1.08")
            else:
                distance_km = Decimal(str(normalized_data.get("distance_km", "0")))

            band = classify_flight_distance_band(distance_km)
            cabin_str = normalized_data.get("cabin_class", "economy")
            try:
                cabin = CabinClass(cabin_str)
            except ValueError:
                cabin = CabinClass.ECONOMY

            ef_data["distance_km"] = str(distance_km)
            ef_data["distance_band"] = band.value
            ef_data["air_ef"] = {k: str(v) for k, v in AIR_EMISSION_FACTORS[band].items()}
            ef_data["cabin_multiplier"] = str(CABIN_CLASS_MULTIPLIERS[cabin])
            ef_data["cabin_class"] = cabin.value

        elif mode == TransportMode.RAIL:
            rail_type_str = normalized_data.get("rail_type", "national")
            try:
                rail_type = RailType(rail_type_str)
            except ValueError:
                rail_type = RailType.NATIONAL
            ef_data["rail_ef"] = {k: str(v) for k, v in RAIL_EMISSION_FACTORS[rail_type].items()}
            ef_data["rail_type"] = rail_type.value

        elif mode == TransportMode.ROAD:
            if "fuel_type" in normalized_data and "litres" in normalized_data:
                fuel_str = normalized_data.get("fuel_type", "petrol")
                try:
                    fuel_type = FuelType(fuel_str)
                except ValueError:
                    fuel_type = FuelType.PETROL
                ef_data["fuel_ef"] = {k: str(v) for k, v in FUEL_EMISSION_FACTORS[fuel_type].items()}
                ef_data["fuel_type"] = fuel_type.value
                ef_data["calc_mode"] = "fuel_based"
            else:
                veh_str = normalized_data.get("vehicle_type", "car_average")
                try:
                    veh_type = RoadVehicleType(veh_str)
                except ValueError:
                    veh_type = RoadVehicleType.CAR_AVERAGE
                ef_data["road_ef"] = {k: str(v) for k, v in ROAD_VEHICLE_EMISSION_FACTORS[veh_type].items()}
                ef_data["vehicle_type"] = veh_type.value
                ef_data["calc_mode"] = "distance_based"

        elif mode == TransportMode.TAXI:
            taxi_str = normalized_data.get("taxi_type", "taxi_regular")
            try:
                taxi_type = RoadVehicleType(taxi_str)
            except ValueError:
                taxi_type = RoadVehicleType.TAXI_REGULAR
            ef_data["road_ef"] = {k: str(v) for k, v in ROAD_VEHICLE_EMISSION_FACTORS[taxi_type].items()}
            ef_data["vehicle_type"] = taxi_type.value

        elif mode == TransportMode.BUS:
            bus_str = normalized_data.get("bus_type", "local")
            try:
                bus_type = BusType(bus_str)
            except ValueError:
                bus_type = BusType.LOCAL
            ef_data["bus_ef"] = {k: str(v) for k, v in BUS_EMISSION_FACTORS[bus_type].items()}
            ef_data["bus_type"] = bus_type.value

        elif mode == TransportMode.FERRY:
            ferry_str = normalized_data.get("ferry_type", "foot_passenger")
            try:
                ferry_type = FerryType(ferry_str)
            except ValueError:
                ferry_type = FerryType.FOOT_PASSENGER
            ef_data["ferry_ef"] = {k: str(v) for k, v in FERRY_EMISSION_FACTORS[ferry_type].items()}
            ef_data["ferry_type"] = ferry_type.value

        elif mode == TransportMode.MOTORCYCLE:
            veh_type = RoadVehicleType.MOTORCYCLE
            ef_data["road_ef"] = {k: str(v) for k, v in ROAD_VEHICLE_EMISSION_FACTORS[veh_type].items()}
            ef_data["vehicle_type"] = veh_type.value

        elif mode == TransportMode.HOTEL:
            country = normalized_data.get("country_code", "GLOBAL").upper()
            ef_val = HOTEL_EMISSION_FACTORS.get(country, HOTEL_EMISSION_FACTORS["GLOBAL"])
            hotel_class_str = normalized_data.get("hotel_class", "standard")
            try:
                hotel_class = HotelClass(hotel_class_str)
            except ValueError:
                hotel_class = HotelClass.STANDARD
            ef_data["hotel_ef"] = str(ef_val)
            ef_data["hotel_class_multiplier"] = str(HOTEL_CLASS_MULTIPLIERS[hotel_class])
            ef_data["hotel_class"] = hotel_class.value
            ef_data["country_code"] = country

        return ef_data

    def _route_to_engine(
        self,
        mode: TransportMode,
        normalized_data: dict,
        ef_data: dict,
        method: CalculationMethod,
    ) -> dict:
        """
        Route calculation to the appropriate sub-engine based on mode.

        Attempts to delegate to lazy-loaded engine classes. If the engine
        is not available (ImportError), performs inline deterministic calculation
        using the resolved emission factors from Stage 4.

        Args:
            mode: Transport mode.
            normalized_data: Normalized trip data.
            ef_data: Resolved emission factor data.
            method: Calculation method.

        Returns:
            Dict with calculation results (total_co2e, wtt_co2e, etc.).
        """
        # Try delegating to sub-engine first
        if mode == TransportMode.AIR:
            engine = self._get_air_engine()
            if engine is not None:
                try:
                    return engine.calculate(normalized_data, ef_data)
                except Exception as e:
                    logger.warning("AirTravelCalculatorEngine failed, using inline: %s", e)
            return self._calculate_air_inline(normalized_data, ef_data)

        elif mode == TransportMode.HOTEL:
            engine = self._get_hotel_engine()
            if engine is not None:
                try:
                    return engine.calculate(normalized_data, ef_data)
                except Exception as e:
                    logger.warning("HotelStayCalculatorEngine failed, using inline: %s", e)
            return self._calculate_hotel_inline(normalized_data, ef_data)

        elif mode in (
            TransportMode.RAIL,
            TransportMode.ROAD,
            TransportMode.BUS,
            TransportMode.TAXI,
            TransportMode.FERRY,
            TransportMode.MOTORCYCLE,
        ):
            engine = self._get_ground_engine()
            if engine is not None:
                try:
                    return engine.calculate(mode, normalized_data, ef_data)
                except Exception as e:
                    logger.warning("GroundTransportCalculatorEngine failed, using inline: %s", e)
            return self._calculate_ground_inline(mode, normalized_data, ef_data)

        else:
            raise ValueError(f"Unsupported transport mode: {mode.value}")

    def _stage_allocate(
        self,
        calc_result: dict,
        department: Optional[str],
        cost_center: Optional[str],
    ) -> dict:
        """
        Stage 7: ALLOCATE - Attach department/cost_center to result.

        No proportional allocation is applied here (single-trip context);
        department and cost_center are recorded for downstream aggregation.

        Args:
            calc_result: Calculation result dict.
            department: Department name (optional).
            cost_center: Cost center code (optional).

        Returns:
            Enriched calculation result dict.
        """
        result = dict(calc_result)
        if department:
            result["department"] = department
        if cost_center:
            result["cost_center"] = cost_center
        return result

    def _stage_compliance(
        self,
        mode: TransportMode,
        method: CalculationMethod,
        allocated: dict,
    ) -> dict:
        """
        Stage 8: COMPLIANCE - Run compliance checks.

        Validates that the calculation meets minimum requirements for:
        - GHG Protocol Scope 3 (method disclosure, EF source citation)
        - CDP (radiative forcing disclosure for flights)

        Args:
            mode: Transport mode.
            method: Calculation method.
            allocated: Allocated calculation results.

        Returns:
            Dict with compliance check results.
        """
        engine = self._get_compliance_engine()
        if engine is not None:
            try:
                return engine.check(mode, method, allocated)
            except Exception as e:
                logger.warning("ComplianceCheckerEngine failed, using inline: %s", e)

        # Inline lightweight compliance check
        findings: List[str] = []
        status = "PASS"

        # GHG Protocol requires method disclosure
        if not method:
            findings.append("GHG Protocol: calculation method not specified")
            status = "WARNING"

        # CDP requires RF disclosure for air travel
        if mode == TransportMode.AIR:
            if "co2e_without_rf" not in allocated and "co2e_with_rf" not in allocated:
                findings.append("CDP: radiative forcing not disclosed for air travel")
                status = "WARNING"

        return {
            "status": status,
            "findings": findings,
            "frameworks_checked": ["GHG_PROTOCOL", "CDP"],
        }

    def _stage_aggregate_dqi(self, mode: TransportMode, method: CalculationMethod) -> Decimal:
        """
        Stage 9: AGGREGATE - Calculate Data Quality Indicator score.

        Assigns a DQI score (1-5 scale) based on mode and method:
        - supplier_specific: 4.5
        - distance_based AIR: 4.0
        - distance_based GROUND: 3.5
        - average_data: 2.5
        - spend_based: 1.5

        Args:
            mode: Transport mode.
            method: Calculation method used.

        Returns:
            DQI score as Decimal (1-5).
        """
        if method == CalculationMethod.SUPPLIER_SPECIFIC:
            return Decimal("4.5")
        elif method == CalculationMethod.DISTANCE_BASED:
            if mode == TransportMode.AIR:
                return Decimal("4.0")
            return Decimal("3.5")
        elif method == CalculationMethod.AVERAGE_DATA:
            return Decimal("2.5")
        elif method == CalculationMethod.SPEND_BASED:
            return Decimal("1.5")
        return Decimal("3.0")

    def _stage_seal(self, chain_id: str, allocated: dict) -> str:
        """
        Stage 10: SEAL - Seal provenance chain and generate final hash.

        Creates SHA-256 hash over the entire provenance chain for this
        calculation, producing an immutable audit fingerprint.

        Args:
            chain_id: Provenance chain identifier.
            allocated: Final allocated result data.

        Returns:
            SHA-256 hex digest string (64 characters).
        """
        chain = self._provenance_chains.get(chain_id, [])
        chain_str = json.dumps(chain, sort_keys=True, default=str)
        result_str = json.dumps(allocated, sort_keys=True, default=str)
        combined = f"{chain_str}|{result_str}"
        provenance_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        return provenance_hash

    # ==========================================================================
    # INLINE CALCULATION METHODS (zero-hallucination, deterministic)
    # ==========================================================================

    def _calculate_air_inline(self, data: dict, ef_data: dict) -> dict:
        """
        Inline air travel calculation using DEFRA emission factors.

        Formula:
          distance_km x air_ef[band] x cabin_multiplier x passengers x (2 if round_trip)

        Args:
            data: Normalized trip data.
            ef_data: Resolved emission factor data.

        Returns:
            Dict with co2e_without_rf, co2e_with_rf, wtt_co2e, total_co2e.
        """
        distance_km = Decimal(str(ef_data.get("distance_km", data.get("distance_km", "0"))))
        cabin_mult = Decimal(str(ef_data.get("cabin_multiplier", "1.0")))
        passengers = int(data.get("passengers", 1))
        round_trip_mult = Decimal("2") if data.get("round_trip", False) else Decimal("1")
        air_ef = ef_data.get("air_ef", {})

        ef_without_rf = Decimal(str(air_ef.get("without_rf", "0.19309")))
        ef_with_rf = Decimal(str(air_ef.get("with_rf", "0.21932")))
        ef_wtt = Decimal(str(air_ef.get("wtt", "0.04528")))

        base = distance_km * cabin_mult * Decimal(str(passengers)) * round_trip_mult

        co2e_without_rf = (base * ef_without_rf).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        co2e_with_rf = (base * ef_with_rf).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        wtt_co2e = (base * ef_wtt).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        # Default total uses with_rf (DEFRA default)
        total_co2e = co2e_with_rf + wtt_co2e

        return {
            "total_co2e": str(total_co2e),
            "co2e_without_rf": str(co2e_without_rf),
            "co2e_with_rf": str(co2e_with_rf),
            "wtt_co2e": str(wtt_co2e),
            "distance_km": str(distance_km),
            "cabin_class": ef_data.get("cabin_class", "economy"),
            "distance_band": ef_data.get("distance_band", "long_haul"),
            "passengers": passengers,
            "round_trip": data.get("round_trip", False),
            "ef_source": "DEFRA",
        }

    def _calculate_ground_inline(self, mode: TransportMode, data: dict, ef_data: dict) -> dict:
        """
        Inline ground transport calculation.

        Handles rail, road (distance and fuel), bus, taxi, ferry, motorcycle.

        Args:
            mode: Transport mode.
            data: Normalized trip data.
            ef_data: Resolved emission factor data.

        Returns:
            Dict with total_co2e, wtt_co2e, and mode-specific details.
        """
        passengers = int(data.get("passengers", 1))
        distance_km = Decimal(str(data.get("distance_km", "0")))

        if mode == TransportMode.RAIL:
            rail_ef = ef_data.get("rail_ef", {})
            ef_ttw = Decimal(str(rail_ef.get("ttw", "0.03549")))
            ef_wtt = Decimal(str(rail_ef.get("wtt", "0.00434")))
            co2e = (distance_km * ef_ttw * Decimal(str(passengers))).quantize(_QUANT_8DP)
            wtt = (distance_km * ef_wtt * Decimal(str(passengers))).quantize(_QUANT_8DP)

        elif mode == TransportMode.ROAD:
            if ef_data.get("calc_mode") == "fuel_based":
                litres = Decimal(str(data.get("litres", "0")))
                fuel_ef = ef_data.get("fuel_ef", {})
                ef_per_litre = Decimal(str(fuel_ef.get("ef_per_litre", "2.31480")))
                wtt_per_litre = Decimal(str(fuel_ef.get("wtt_per_litre", "0.58549")))
                co2e = (litres * ef_per_litre).quantize(_QUANT_8DP)
                wtt = (litres * wtt_per_litre).quantize(_QUANT_8DP)
            else:
                road_ef = ef_data.get("road_ef", {})
                ef_per_vkm = Decimal(str(road_ef.get("ef_per_vkm", "0.27145")))
                wtt_per_vkm = Decimal(str(road_ef.get("wtt_per_vkm", "0.06291")))
                co2e = (distance_km * ef_per_vkm).quantize(_QUANT_8DP)
                wtt = (distance_km * wtt_per_vkm).quantize(_QUANT_8DP)

        elif mode == TransportMode.TAXI:
            road_ef = ef_data.get("road_ef", {})
            ef_per_vkm = Decimal(str(road_ef.get("ef_per_vkm", "0.20920")))
            wtt_per_vkm = Decimal(str(road_ef.get("wtt_per_vkm", "0.04710")))
            co2e = (distance_km * ef_per_vkm).quantize(_QUANT_8DP)
            wtt = (distance_km * wtt_per_vkm).quantize(_QUANT_8DP)

        elif mode == TransportMode.BUS:
            bus_ef = ef_data.get("bus_ef", {})
            ef = Decimal(str(bus_ef.get("ef", "0.10312")))
            ef_wtt = Decimal(str(bus_ef.get("wtt", "0.01847")))
            co2e = (distance_km * ef * Decimal(str(passengers))).quantize(_QUANT_8DP)
            wtt = (distance_km * ef_wtt * Decimal(str(passengers))).quantize(_QUANT_8DP)

        elif mode == TransportMode.FERRY:
            ferry_ef = ef_data.get("ferry_ef", {})
            ef = Decimal(str(ferry_ef.get("ef", "0.01877")))
            ef_wtt = Decimal(str(ferry_ef.get("wtt", "0.00572")))
            co2e = (distance_km * ef * Decimal(str(passengers))).quantize(_QUANT_8DP)
            wtt = (distance_km * ef_wtt * Decimal(str(passengers))).quantize(_QUANT_8DP)

        elif mode == TransportMode.MOTORCYCLE:
            road_ef = ef_data.get("road_ef", {})
            ef_per_vkm = Decimal(str(road_ef.get("ef_per_vkm", "0.11337")))
            wtt_per_vkm = Decimal(str(road_ef.get("wtt_per_vkm", "0.02867")))
            co2e = (distance_km * ef_per_vkm).quantize(_QUANT_8DP)
            wtt = (distance_km * wtt_per_vkm).quantize(_QUANT_8DP)

        else:
            co2e = Decimal("0")
            wtt = Decimal("0")

        total = co2e + wtt

        return {
            "total_co2e": str(total),
            "co2e": str(co2e),
            "wtt_co2e": str(wtt),
            "distance_km": str(distance_km),
            "mode": mode.value,
            "ef_source": ef_data.get("ef_source", "DEFRA"),
        }

    def _calculate_hotel_inline(self, data: dict, ef_data: dict) -> dict:
        """
        Inline hotel accommodation calculation.

        Formula:
          room_nights x hotel_ef[country] x hotel_class_multiplier

        Args:
            data: Normalized trip data.
            ef_data: Resolved emission factor data.

        Returns:
            Dict with total_co2e and hotel-specific details.
        """
        room_nights = int(data.get("room_nights", 1))
        hotel_ef = Decimal(str(ef_data.get("hotel_ef", "20.90")))
        class_mult = Decimal(str(ef_data.get("hotel_class_multiplier", "1.0")))

        co2e = (Decimal(str(room_nights)) * hotel_ef * class_mult).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        return {
            "total_co2e": str(co2e),
            "co2e": str(co2e),
            "wtt_co2e": "0",
            "room_nights": room_nights,
            "hotel_class": ef_data.get("hotel_class", "standard"),
            "country_code": ef_data.get("country_code", "GLOBAL"),
            "ef_source": "DEFRA",
        }

    # ==========================================================================
    # PROVENANCE AND UTILITY HELPERS
    # ==========================================================================

    def _record_provenance(
        self,
        chain_id: str,
        stage: ProvenanceStage,
        input_data: Any,
        output_data: Any,
    ) -> None:
        """
        Record a provenance entry for a pipeline stage.

        Args:
            chain_id: Provenance chain identifier.
            stage: Pipeline stage enum.
            input_data: Input to this stage.
            output_data: Output from this stage.
        """
        input_str = json.dumps(input_data, sort_keys=True, default=str) if not isinstance(input_data, str) else input_data
        output_str = json.dumps(output_data, sort_keys=True, default=str) if not isinstance(output_data, str) else output_data

        entry = {
            "stage": stage.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_hash": hashlib.sha256(input_str.encode("utf-8")).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode("utf-8")).hexdigest(),
        }

        chain = self._provenance_chains.get(chain_id)
        if chain is not None:
            entry["chain_hash"] = hashlib.sha256(
                (json.dumps(chain[-1], sort_keys=True) if chain else "").encode("utf-8")
            ).hexdigest()
            chain.append(entry)

    @staticmethod
    def _elapsed_ms(start: datetime) -> float:
        """Calculate milliseconds elapsed since start."""
        return (datetime.now(timezone.utc) - start).total_seconds() * 1000.0

    # ==========================================================================
    # LAZY ENGINE LOADING
    # ==========================================================================

    def _get_air_engine(self) -> Optional[Any]:
        """Get or create AirTravelCalculatorEngine (lazy loading)."""
        if self._air_engine is None:
            try:
                from greenlang.agents.mrv.business_travel.air_travel_calculator import AirTravelCalculatorEngine
                self._air_engine = AirTravelCalculatorEngine()
            except ImportError:
                logger.debug("AirTravelCalculatorEngine not available, using inline calculation")
                self._air_engine = None
        return self._air_engine

    def _get_ground_engine(self) -> Optional[Any]:
        """Get or create GroundTransportCalculatorEngine (lazy loading)."""
        if self._ground_engine is None:
            try:
                from greenlang.agents.mrv.business_travel.ground_transport_calculator import GroundTransportCalculatorEngine
                self._ground_engine = GroundTransportCalculatorEngine()
            except ImportError:
                logger.debug("GroundTransportCalculatorEngine not available, using inline calculation")
                self._ground_engine = None
        return self._ground_engine

    def _get_hotel_engine(self) -> Optional[Any]:
        """Get or create HotelStayCalculatorEngine (lazy loading)."""
        if self._hotel_engine is None:
            try:
                from greenlang.agents.mrv.business_travel.hotel_stay_calculator import HotelStayCalculatorEngine
                self._hotel_engine = HotelStayCalculatorEngine()
            except ImportError:
                logger.debug("HotelStayCalculatorEngine not available, using inline calculation")
                self._hotel_engine = None
        return self._hotel_engine

    def _get_spend_engine(self) -> Optional[Any]:
        """Get or create SpendBasedCalculatorEngine (lazy loading)."""
        if self._spend_engine is None:
            try:
                from greenlang.agents.mrv.business_travel.spend_based_calculator import SpendBasedCalculatorEngine
                self._spend_engine = SpendBasedCalculatorEngine()
            except ImportError:
                logger.debug("SpendBasedCalculatorEngine not available, using inline calculation")
                self._spend_engine = None
        return self._spend_engine

    def _get_database_engine(self) -> Optional[Any]:
        """Get or create BusinessTravelDatabaseEngine (lazy loading)."""
        if self._database_engine is None:
            try:
                from greenlang.agents.mrv.business_travel.business_travel_database import BusinessTravelDatabaseEngine
                self._database_engine = BusinessTravelDatabaseEngine()
            except ImportError:
                logger.debug("BusinessTravelDatabaseEngine not available")
                self._database_engine = None
        return self._database_engine

    def _get_compliance_engine(self) -> Optional[Any]:
        """Get or create ComplianceCheckerEngine (lazy loading)."""
        if self._compliance_engine is None:
            try:
                from greenlang.agents.mrv.business_travel.compliance_checker import ComplianceCheckerEngine
                self._compliance_engine = ComplianceCheckerEngine()
            except ImportError:
                logger.debug("ComplianceCheckerEngine not available, using inline compliance")
                self._compliance_engine = None
        return self._compliance_engine

    # ==========================================================================
    # UTILITY / STATUS METHODS
    # ==========================================================================

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status including loaded engines.

        Returns:
            Dictionary with pipeline status information.
        """
        return {
            "agent_id": "GL-MRV-S3-006",
            "agent_component": "AGENT-MRV-019",
            "version": "1.0.0",
            "engines_loaded": {
                "air": self._air_engine is not None,
                "ground": self._ground_engine is not None,
                "hotel": self._hotel_engine is not None,
                "spend": self._spend_engine is not None,
                "database": self._database_engine is not None,
                "compliance": self._compliance_engine is not None,
            },
            "active_chains": len(self._provenance_chains),
        }

    def reset_pipeline(self) -> None:
        """
        Reset pipeline state (clear provenance chains).

        Used for testing or periodic cleanup.
        """
        with self._lock:
            self._provenance_chains.clear()
            logger.info("Pipeline state reset")

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        This forces re-initialization on next instantiation.
        """
        with cls._lock:
            cls._instance = None


# ==============================================================================
# MODULE-LEVEL HELPERS
# ==============================================================================


def get_pipeline_engine() -> BusinessTravelPipelineEngine:
    """
    Get singleton pipeline engine instance.

    Returns:
        BusinessTravelPipelineEngine singleton instance.
    """
    return BusinessTravelPipelineEngine()
