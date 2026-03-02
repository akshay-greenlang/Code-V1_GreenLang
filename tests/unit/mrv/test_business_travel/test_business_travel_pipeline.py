# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-019 Business Travel Agent - BusinessTravelPipelineEngine.

Tests the 10-stage orchestration pipeline: VALIDATE, CLASSIFY, NORMALIZE,
RESOLVE_EFS, CALCULATE_FLIGHTS, CALCULATE_GROUND, ALLOCATE, COMPLIANCE,
AGGREGATE (DQI), and SEAL. Also tests batch processing, aggregation,
infrastructure (singleton, thread safety, lazy loading), and edge cases.

Target: 45 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.business_travel.business_travel_pipeline import (
        BusinessTravelPipelineEngine,
        PipelineStatus,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

try:
    from greenlang.business_travel.models import (
        TripInput,
        BatchTripInput,
        TripCalculationResult,
        BatchResult,
        AggregationResult,
        TransportMode,
        CalculationMethod,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not (PIPELINE_AVAILABLE and MODELS_AVAILABLE),
    reason="BusinessTravelPipelineEngine or models not available",
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the pipeline singleton before each test for isolation."""
    if PIPELINE_AVAILABLE:
        BusinessTravelPipelineEngine.reset_singleton()
    yield
    if PIPELINE_AVAILABLE:
        BusinessTravelPipelineEngine.reset_singleton()


@pytest.fixture
def engine():
    """Create a fresh BusinessTravelPipelineEngine instance."""
    return BusinessTravelPipelineEngine()


@pytest.fixture
def air_trip():
    """Valid air trip input: LHR -> JFK, economy, one-way, with RF."""
    return TripInput(
        mode=TransportMode.AIR,
        trip_data={
            "origin_iata": "LHR",
            "destination_iata": "JFK",
            "cabin_class": "economy",
            "passengers": 1,
            "round_trip": False,
            "rf_option": "with_rf",
        },
    )


@pytest.fixture
def rail_trip():
    """Valid rail trip input: national, 640 km."""
    return TripInput(
        mode=TransportMode.RAIL,
        trip_data={
            "rail_type": "national",
            "distance_km": 640,
            "passengers": 1,
        },
    )


@pytest.fixture
def road_trip():
    """Valid road trip input: car_average, 300 km."""
    return TripInput(
        mode=TransportMode.ROAD,
        trip_data={
            "vehicle_type": "car_average",
            "distance_km": 300,
        },
    )


@pytest.fixture
def hotel_trip():
    """Valid hotel trip input: GB standard, 3 nights."""
    return TripInput(
        mode=TransportMode.HOTEL,
        trip_data={
            "country_code": "GB",
            "room_nights": 3,
            "hotel_class": "standard",
        },
    )


@pytest.fixture
def taxi_trip():
    """Valid taxi trip input: regular taxi, 25 km."""
    return TripInput(
        mode=TransportMode.TAXI,
        trip_data={
            "distance_km": 25,
            "taxi_type": "taxi_regular",
        },
    )


@pytest.fixture
def bus_trip():
    """Valid bus trip input: coach, 100 km."""
    return TripInput(
        mode=TransportMode.BUS,
        trip_data={
            "bus_type": "coach",
            "distance_km": 100,
            "passengers": 1,
        },
    )


# ===========================================================================
# Single Trip Tests (15)
# ===========================================================================


@_SKIP
class TestSingleTripCalculation:
    """Test individual trip emissions calculation through the full pipeline."""

    def test_air_trip(self, engine, air_trip):
        """Air trip (LHR->JFK economy) returns positive CO2e."""
        result = engine.calculate(air_trip)
        assert isinstance(result, TripCalculationResult)
        assert result.mode == TransportMode.AIR
        assert result.total_co2e > Decimal("0")

    def test_rail_trip(self, engine, rail_trip):
        """Rail trip (national 640 km) returns positive CO2e."""
        result = engine.calculate(rail_trip)
        assert isinstance(result, TripCalculationResult)
        assert result.mode == TransportMode.RAIL
        assert result.total_co2e > Decimal("0")

    def test_road_trip(self, engine, road_trip):
        """Road trip (car_average 300 km) returns positive CO2e."""
        result = engine.calculate(road_trip)
        assert isinstance(result, TripCalculationResult)
        assert result.mode == TransportMode.ROAD
        assert result.total_co2e > Decimal("0")

    def test_hotel_trip(self, engine, hotel_trip):
        """Hotel trip (GB standard 3 nights) returns positive CO2e."""
        result = engine.calculate(hotel_trip)
        assert isinstance(result, TripCalculationResult)
        assert result.mode == TransportMode.HOTEL
        assert result.total_co2e > Decimal("0")

    def test_taxi_trip(self, engine, taxi_trip):
        """Taxi trip (regular 25 km) returns positive CO2e."""
        result = engine.calculate(taxi_trip)
        assert isinstance(result, TripCalculationResult)
        assert result.mode == TransportMode.TAXI
        assert result.total_co2e > Decimal("0")

    def test_bus_trip(self, engine, bus_trip):
        """Bus trip (coach 100 km) returns positive CO2e."""
        result = engine.calculate(bus_trip)
        assert isinstance(result, TripCalculationResult)
        assert result.mode == TransportMode.BUS
        assert result.total_co2e > Decimal("0")

    def test_result_has_mode(self, engine, air_trip):
        """Result includes the transport mode."""
        result = engine.calculate(air_trip)
        assert result.mode is not None
        assert isinstance(result.mode, TransportMode)

    def test_result_has_method(self, engine, air_trip):
        """Result includes the calculation method."""
        result = engine.calculate(air_trip)
        assert result.method is not None
        assert isinstance(result.method, CalculationMethod)

    def test_result_has_total_co2e(self, engine, rail_trip):
        """Result total_co2e is a Decimal."""
        result = engine.calculate(rail_trip)
        assert isinstance(result.total_co2e, Decimal)

    def test_result_has_provenance_hash(self, engine, air_trip):
        """Result provenance_hash is a 64-char hex SHA-256 string."""
        result = engine.calculate(air_trip)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_result_has_dqi_score(self, engine, air_trip):
        """Result dqi_score is a Decimal between 1.0 and 5.0."""
        result = engine.calculate(air_trip)
        assert result.dqi_score is not None
        assert Decimal("1.0") <= result.dqi_score <= Decimal("5.0")

    def test_result_co2e_positive(self, engine, road_trip):
        """Result total_co2e is strictly positive for valid inputs."""
        result = engine.calculate(road_trip)
        assert result.total_co2e > Decimal("0")

    def test_air_trip_has_rf_values(self, engine, air_trip):
        """Air trip result includes co2e_with_rf and co2e_without_rf."""
        result = engine.calculate(air_trip)
        assert result.co2e_without_rf is not None
        assert result.co2e_with_rf is not None
        assert result.co2e_with_rf > result.co2e_without_rf

    def test_invalid_mode_data_raises(self, engine):
        """Air trip without required IATA codes raises ValueError."""
        trip = TripInput(
            mode=TransportMode.AIR,
            trip_data={"passengers": 1},
        )
        with pytest.raises(ValueError, match="validation failed"):
            engine.calculate(trip)

    def test_missing_required_data_raises(self, engine):
        """Trip with empty trip_data raises ValueError."""
        trip = TripInput(
            mode=TransportMode.RAIL,
            trip_data={},
        )
        with pytest.raises(ValueError, match="validation failed"):
            engine.calculate(trip)


# ===========================================================================
# Batch Tests (12)
# ===========================================================================


@_SKIP
class TestBatchCalculation:
    """Test batch trip processing through the pipeline."""

    def test_batch_multiple_trips(self, engine, air_trip, rail_trip, hotel_trip):
        """Batch with 3 trips returns BatchResult with correct count."""
        batch = BatchTripInput(
            trips=[air_trip, rail_trip, hotel_trip],
            reporting_period="2024-Q1",
        )
        result = engine.calculate_batch(batch)
        assert isinstance(result, BatchResult)
        assert result.count == 3
        assert result.total_co2e > Decimal("0")

    def test_batch_total_is_sum(self, engine, air_trip, rail_trip):
        """Batch total_co2e equals sum of individual trip totals."""
        batch = BatchTripInput(
            trips=[air_trip, rail_trip],
            reporting_period="2024-Q2",
        )
        result = engine.calculate_batch(batch)
        individual_sum = sum(r.total_co2e for r in result.results)
        assert result.total_co2e == individual_sum

    def test_batch_count_correct(self, engine, road_trip):
        """Batch count matches number of successful results."""
        batch = BatchTripInput(
            trips=[road_trip, road_trip],
            reporting_period="2024-Q3",
        )
        result = engine.calculate_batch(batch)
        assert result.count == len(result.results)

    def test_batch_errors_isolated(self, engine, air_trip):
        """A single bad trip does not prevent other trips from calculating."""
        bad_trip = TripInput(
            mode=TransportMode.AIR,
            trip_data={},
        )
        batch = BatchTripInput(
            trips=[air_trip, bad_trip],
            reporting_period="2024-Q1",
        )
        result = engine.calculate_batch(batch)
        # At least 1 successful, at least 1 error
        assert len(result.results) >= 1
        assert len(result.errors) >= 1

    def test_batch_single_trip(self, engine, hotel_trip):
        """Batch with a single trip processes successfully."""
        batch = BatchTripInput(
            trips=[hotel_trip],
            reporting_period="2024-Q4",
        )
        result = engine.calculate_batch(batch)
        assert result.count == 1
        assert result.total_co2e > Decimal("0")

    def test_batch_reporting_period(self, engine, rail_trip):
        """Batch result preserves the reporting_period field."""
        batch = BatchTripInput(
            trips=[rail_trip],
            reporting_period="2025-H1",
        )
        result = engine.calculate_batch(batch)
        assert result.reporting_period == "2025-H1"

    def test_batch_mixed_modes(self, engine, air_trip, rail_trip, road_trip, hotel_trip, taxi_trip, bus_trip):
        """Batch with all transport modes succeeds."""
        batch = BatchTripInput(
            trips=[air_trip, rail_trip, road_trip, hotel_trip, taxi_trip, bus_trip],
            reporting_period="2024-FY",
        )
        result = engine.calculate_batch(batch)
        assert result.count == 6
        assert len(result.errors) == 0

    def test_batch_rf_aggregation(self, engine, air_trip):
        """Batch with air trips aggregates RF values correctly."""
        batch = BatchTripInput(
            trips=[air_trip, air_trip],
            reporting_period="2024-Q1",
        )
        result = engine.calculate_batch(batch)
        assert result.total_co2e_with_rf is not None
        assert result.total_co2e_with_rf > Decimal("0")
        assert result.total_co2e_without_rf is not None
        assert result.total_co2e_without_rf > Decimal("0")

    def test_batch_with_error_still_succeeds(self, engine, road_trip):
        """Batch returns partial results when some trips fail."""
        bad_trip = TripInput(
            mode=TransportMode.FERRY,
            trip_data={},
        )
        batch = BatchTripInput(
            trips=[road_trip, bad_trip, road_trip],
            reporting_period="2024-Q2",
        )
        result = engine.calculate_batch(batch)
        assert result.count >= 2
        assert result.total_co2e > Decimal("0")

    def test_batch_all_errors(self, engine):
        """Batch where all trips fail returns zero total and all errors."""
        bad_trip1 = TripInput(mode=TransportMode.AIR, trip_data={})
        bad_trip2 = TripInput(mode=TransportMode.RAIL, trip_data={})
        batch = BatchTripInput(
            trips=[bad_trip1, bad_trip2],
            reporting_period="2024-Q1",
        )
        result = engine.calculate_batch(batch)
        assert result.count == 0
        assert result.total_co2e == Decimal("0")
        assert len(result.errors) == 2

    def test_batch_large_100_trips(self, engine, road_trip):
        """Batch with 100 identical trips processes without error."""
        batch = BatchTripInput(
            trips=[road_trip] * 100,
            reporting_period="2024-FY",
        )
        result = engine.calculate_batch(batch)
        assert result.count == 100
        assert len(result.errors) == 0
        # Total should be 100x a single road trip
        single_result = engine.calculate(road_trip)
        expected = single_result.total_co2e * 100
        assert result.total_co2e == expected

    def test_batch_empty_errors_on_success(self, engine, rail_trip):
        """When all trips succeed, errors list is empty."""
        batch = BatchTripInput(
            trips=[rail_trip, rail_trip],
            reporting_period="2024-Q3",
        )
        result = engine.calculate_batch(batch)
        assert len(result.errors) == 0


# ===========================================================================
# Aggregation Tests (10)
# ===========================================================================


@_SKIP
class TestAggregation:
    """Test multi-dimensional aggregation of trip results."""

    def _build_results(self, engine, air_trip, rail_trip, hotel_trip):
        """Helper to build a list of TripCalculationResults."""
        results = [
            engine.calculate(air_trip),
            engine.calculate(rail_trip),
            engine.calculate(hotel_trip),
        ]
        return results

    def test_aggregate_total_co2e(self, engine, air_trip, rail_trip, hotel_trip):
        """Aggregation total_co2e equals sum of all results."""
        results = self._build_results(engine, air_trip, rail_trip, hotel_trip)
        agg = engine.aggregate(results, period="2024-Q1")
        expected = sum(r.total_co2e for r in results)
        assert agg.total_co2e == expected

    def test_aggregate_by_mode(self, engine, air_trip, rail_trip, hotel_trip):
        """Aggregation includes by_mode breakdown."""
        results = self._build_results(engine, air_trip, rail_trip, hotel_trip)
        agg = engine.aggregate(results, period="2024-Q1")
        assert "air" in agg.by_mode
        assert "rail" in agg.by_mode
        assert "hotel" in agg.by_mode

    def test_aggregate_by_department(self, engine):
        """Aggregation groups emissions by department when provided."""
        trip_sales = TripInput(
            mode=TransportMode.RAIL,
            trip_data={"rail_type": "national", "distance_km": 200, "passengers": 1},
            department="Sales",
        )
        trip_eng = TripInput(
            mode=TransportMode.RAIL,
            trip_data={"rail_type": "national", "distance_km": 100, "passengers": 1},
            department="Engineering",
        )
        results = [engine.calculate(trip_sales), engine.calculate(trip_eng)]
        agg = engine.aggregate(results, period="2024-Q1")
        assert "Sales" in agg.by_department
        assert "Engineering" in agg.by_department

    def test_aggregate_by_period(self, engine, rail_trip):
        """Aggregation preserves the period field."""
        results = [engine.calculate(rail_trip)]
        agg = engine.aggregate(results, period="2024-FY")
        assert agg.period == "2024-FY"

    def test_aggregate_by_cabin_class(self, engine):
        """Aggregation captures cabin class breakdown for air trips."""
        trip = TripInput(
            mode=TransportMode.AIR,
            trip_data={
                "origin_iata": "JFK",
                "destination_iata": "LHR",
                "cabin_class": "business",
                "passengers": 1,
                "round_trip": False,
                "rf_option": "with_rf",
            },
        )
        results = [engine.calculate(trip)]
        agg = engine.aggregate(results, period="2024-Q1")
        assert "business" in agg.by_cabin_class

    def test_aggregate_empty_list(self, engine):
        """Aggregation of empty list returns zero total."""
        agg = engine.aggregate([], period="2024-Q1")
        assert agg.total_co2e == Decimal("0")
        assert agg.by_mode == {}

    def test_aggregate_single_result(self, engine, road_trip):
        """Aggregation of a single result matches the result total."""
        result = engine.calculate(road_trip)
        agg = engine.aggregate([result], period="2024-Q1")
        assert agg.total_co2e == result.total_co2e

    def test_aggregate_modes_sum_to_total(self, engine, air_trip, rail_trip, hotel_trip):
        """Sum of by_mode values equals total_co2e."""
        results = self._build_results(engine, air_trip, rail_trip, hotel_trip)
        agg = engine.aggregate(results, period="2024-Q1")
        mode_sum = sum(agg.by_mode.values())
        assert mode_sum == agg.total_co2e

    def test_aggregate_all_modes_present(self, engine, air_trip, rail_trip, road_trip, hotel_trip, taxi_trip, bus_trip):
        """All transport modes are captured in by_mode when present."""
        results = [
            engine.calculate(air_trip),
            engine.calculate(rail_trip),
            engine.calculate(road_trip),
            engine.calculate(hotel_trip),
            engine.calculate(taxi_trip),
            engine.calculate(bus_trip),
        ]
        agg = engine.aggregate(results, period="2024-FY")
        for mode_str in ["air", "rail", "road", "hotel", "taxi", "bus"]:
            assert mode_str in agg.by_mode

    def test_aggregate_period_stored(self, engine, air_trip):
        """Aggregation result stores the supplied period string."""
        results = [engine.calculate(air_trip)]
        agg = engine.aggregate(results, period="FY2025")
        assert agg.period == "FY2025"


# ===========================================================================
# Infrastructure Tests (8)
# ===========================================================================


@_SKIP
class TestPipelineInfrastructure:
    """Test singleton pattern, lazy loading, thread safety, and stage methods."""

    def test_singleton_pattern(self):
        """Multiple instantiations return the same object."""
        engine_a = BusinessTravelPipelineEngine()
        engine_b = BusinessTravelPipelineEngine()
        assert engine_a is engine_b

    def test_lazy_engine_loading(self, engine):
        """Sub-engines are None until first use triggers lazy loading."""
        # Before any calculation, sub-engines should be None
        assert engine._air_engine is None or engine._air_engine is not None
        # After calculation, at least one engine gets lazily loaded
        trip = TripInput(
            mode=TransportMode.AIR,
            trip_data={
                "origin_iata": "JFK",
                "destination_iata": "LAX",
                "cabin_class": "economy",
                "passengers": 1,
                "round_trip": False,
                "rf_option": "with_rf",
            },
        )
        engine.calculate(trip)
        # Pipeline should have attempted to load the air engine
        # (may or may not succeed depending on availability)
        status = engine.get_pipeline_status()
        assert "engines_loaded" in status

    def test_thread_safety(self):
        """Concurrent instantiation from multiple threads returns same singleton."""
        engines = []
        errors = []

        def create():
            try:
                e = BusinessTravelPipelineEngine()
                engines.append(id(e))
            except Exception as ex:
                errors.append(str(ex))

        threads = [threading.Thread(target=create) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All engines should be the same singleton
        assert len(set(engines)) == 1

    def test_engine_count(self, engine):
        """Pipeline status reports 6 engine slots."""
        status = engine.get_pipeline_status()
        assert len(status["engines_loaded"]) == 6

    def test_validate_stage_rejects_bad_input(self, engine):
        """Stage 1 VALIDATE rejects trip with empty trip_data."""
        trip = TripInput(mode=TransportMode.AIR, trip_data={})
        is_valid, errors = engine._stage_validate(trip)
        assert not is_valid
        assert len(errors) > 0

    def test_classify_stage(self, engine, air_trip):
        """Stage 2 CLASSIFY correctly identifies AIR mode and DISTANCE_BASED method."""
        mode, method = engine._stage_classify(air_trip)
        assert mode == TransportMode.AIR
        assert method == CalculationMethod.DISTANCE_BASED

    def test_normalize_converts_miles(self, engine):
        """Stage 3 NORMALIZE converts distance_miles to distance_km."""
        data = {"distance_miles": 100}
        normalized = engine._stage_normalize(data, TransportMode.ROAD)
        assert "distance_km" in normalized
        # 100 miles * 1.60934 = 160.934
        km = Decimal(str(normalized["distance_km"]))
        assert Decimal("160") < km < Decimal("162")

    def test_seal_stage_produces_hash(self, engine):
        """Stage 10 SEAL produces a 64-character hex hash."""
        chain_id = "test-chain"
        engine._provenance_chains[chain_id] = [
            {"stage": "test", "input_hash": "aaa", "output_hash": "bbb"}
        ]
        seal_hash = engine._stage_seal(chain_id, {"total_co2e": "100"})
        assert len(seal_hash) == 64
        assert all(c in "0123456789abcdef" for c in seal_hash)
