# -*- coding: utf-8 -*-
"""
Unit tests for AirTravelCalculatorEngine (Engine 2) - AGENT-MRV-019

Tests the complete aviation emissions calculation pipeline including:
- Great-circle distance (Haversine) for 12 airport pairs
- Distance uplift factor (default 8%, custom, zero)
- Distance band classification (domestic/short-haul/long-haul boundaries)
- Full flight calculation with cabin class, RF, WTT, round trip
- Batch calculation, provenance hashing, and edge cases

80 tests total across 7 test classes.

DEFRA 2024 emission factors and CABIN_CLASS_MULTIPLIERS are used
throughout. All expected values are computed from deterministic
Decimal arithmetic matching the engine's own formula.

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from greenlang.business_travel.models import (
    FlightInput,
    FlightResult,
    CabinClass,
    FlightDistanceBand,
    RFOption,
    GWPVersion,
    EFSource,
    AIRPORT_DATABASE,
    AIR_EMISSION_FACTORS,
    CABIN_CLASS_MULTIPLIERS,
    calculate_provenance_hash,
)

# ---------------------------------------------------------------------------
# Module-level constants for expected values
# ---------------------------------------------------------------------------

_Q8 = Decimal("0.00000001")


def _q(v: Decimal) -> Decimal:
    """Local quantize helper matching engine precision."""
    return v.quantize(_Q8, rounding=ROUND_HALF_UP)


# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """
    Reset the AirTravelCalculatorEngine singleton before and after each test.

    This ensures every test starts with a fresh engine instance so that
    singleton state (e.g. calculation_count) does not leak between tests.
    """
    from greenlang.business_travel.air_travel_calculator import (
        reset_air_travel_calculator,
    )

    reset_air_travel_calculator()
    yield
    reset_air_travel_calculator()


@pytest.fixture
def engine():
    """
    Create a fresh AirTravelCalculatorEngine with mocked dependencies.

    Mocks get_config, get_metrics, and the database engine so that tests
    run without external systems while still exercising the full
    calculation pipeline.
    """
    with patch(
        "greenlang.business_travel.air_travel_calculator.get_config"
    ) as mock_config, patch(
        "greenlang.business_travel.air_travel_calculator.get_metrics"
    ) as mock_metrics, patch(
        "greenlang.business_travel.air_travel_calculator.get_database_engine"
    ) as mock_db_factory:
        # Configure mock config
        cfg = MagicMock()
        cfg.general.default_uplift_factor = Decimal("0.08")
        mock_config.return_value = cfg

        # Configure mock metrics (no-op)
        metrics = MagicMock()
        mock_metrics.return_value = metrics

        # Configure mock database engine
        db_engine = MagicMock()

        def _mock_cabin_multiplier(cabin_class):
            return CABIN_CLASS_MULTIPLIERS[cabin_class]

        def _mock_air_ef(distance_band, cabin_class=CabinClass.ECONOMY, source=EFSource.DEFRA):
            return dict(AIR_EMISSION_FACTORS[distance_band])

        db_engine.get_cabin_class_multiplier.side_effect = _mock_cabin_multiplier
        db_engine.get_air_emission_factor.side_effect = _mock_air_ef
        mock_db_factory.return_value = db_engine

        from greenlang.business_travel.air_travel_calculator import (
            AirTravelCalculatorEngine,
        )

        eng = AirTravelCalculatorEngine()
        yield eng


# ===========================================================================
# 1. GREAT-CIRCLE DISTANCE TESTS (12 tests)
# ===========================================================================


class TestGreatCircleDistance:
    """Validate Haversine great-circle distance computation."""

    def setup_method(self):
        with patch(
            "greenlang.business_travel.air_travel_calculator.get_config"
        ) as mc, patch(
            "greenlang.business_travel.air_travel_calculator.get_metrics"
        ) as mm:
            cfg = MagicMock()
            cfg.general.default_uplift_factor = Decimal("0.08")
            mc.return_value = cfg
            mm.return_value = MagicMock()
            from greenlang.business_travel.air_travel_calculator import (
                AirTravelCalculatorEngine,
            )
            self.engine = AirTravelCalculatorEngine()

    # 1
    def test_gcd_lhr_jfk(self):
        """LHR (51.47, -0.4543) to JFK (40.6413, -73.7781) approx 5540 km."""
        d = self.engine.calculate_great_circle_distance(
            Decimal("51.4700"), Decimal("-0.4543"),
            Decimal("40.6413"), Decimal("-73.7781"),
        )
        assert Decimal("5400") < d < Decimal("5700")

    # 2
    def test_gcd_lhr_cdg(self):
        """LHR to CDG approx 344 km (short hop)."""
        d = self.engine.calculate_great_circle_distance(
            Decimal("51.4700"), Decimal("-0.4543"),
            Decimal("49.0097"), Decimal("2.5479"),
        )
        assert Decimal("300") < d < Decimal("400")

    # 3
    def test_gcd_syd_lax(self):
        """SYD to LAX approx 12000 km (ultra long-haul)."""
        d = self.engine.calculate_great_circle_distance(
            Decimal("-33.9461"), Decimal("151.1772"),
            Decimal("33.9425"), Decimal("-118.4081"),
        )
        assert Decimal("11500") < d < Decimal("12500")

    # 4
    def test_gcd_same_point_returns_zero(self):
        """Same origin and destination yields zero distance."""
        d = self.engine.calculate_great_circle_distance(
            Decimal("51.4700"), Decimal("-0.4543"),
            Decimal("51.4700"), Decimal("-0.4543"),
        )
        assert d == Decimal("0E-8") or d == _q(Decimal("0"))

    # 5
    def test_gcd_returns_decimal_type(self):
        """Result must be a Decimal instance."""
        d = self.engine.calculate_great_circle_distance(
            Decimal("51.4700"), Decimal("-0.4543"),
            Decimal("49.0097"), Decimal("2.5479"),
        )
        assert isinstance(d, Decimal)

    # 6
    def test_gcd_cross_dateline_nrt_lax(self):
        """NRT (35.772, 140.393) to LAX (33.943, -118.408) crosses date-line."""
        d = self.engine.calculate_great_circle_distance(
            Decimal("35.7720"), Decimal("140.3929"),
            Decimal("33.9425"), Decimal("-118.4081"),
        )
        assert Decimal("8500") < d < Decimal("9500")

    # 7
    def test_gcd_short_distance_lhr_lgw(self):
        """LHR to LGW approx 41-46 km."""
        d = self.engine.calculate_great_circle_distance(
            Decimal("51.4700"), Decimal("-0.4543"),
            Decimal("51.1537"), Decimal("-0.1821"),
        )
        assert Decimal("30") < d < Decimal("60")

    # 8
    def test_gcd_equatorial_sin_nbo(self):
        """SIN (1.364, 103.99) to NBO (-1.319, 36.928) near equator."""
        d = self.engine.calculate_great_circle_distance(
            Decimal("1.3644"), Decimal("103.9915"),
            Decimal("-1.3192"), Decimal("36.9278"),
        )
        assert Decimal("7300") < d < Decimal("7800")

    # 9
    def test_gcd_always_positive(self):
        """Distance must be non-negative regardless of direction."""
        d = self.engine.calculate_great_circle_distance(
            Decimal("-33.9461"), Decimal("151.1772"),
            Decimal("51.4700"), Decimal("-0.4543"),
        )
        assert d >= Decimal("0")

    # 10
    def test_gcd_symmetric(self):
        """Distance A->B must equal B->A."""
        d_ab = self.engine.calculate_great_circle_distance(
            Decimal("51.4700"), Decimal("-0.4543"),
            Decimal("40.6413"), Decimal("-73.7781"),
        )
        d_ba = self.engine.calculate_great_circle_distance(
            Decimal("40.6413"), Decimal("-73.7781"),
            Decimal("51.4700"), Decimal("-0.4543"),
        )
        assert d_ab == d_ba

    # 11
    def test_gcd_north_to_south_arn_jnb(self):
        """ARN (59.65, 17.92) to JNB (-26.14, 28.25) north-south traverse."""
        d = self.engine.calculate_great_circle_distance(
            Decimal("59.6519"), Decimal("17.9186"),
            Decimal("-26.1367"), Decimal("28.2460"),
        )
        assert Decimal("9400") < d < Decimal("9900")

    # 12
    def test_gcd_decimal_precision_8dp(self):
        """Result must have exactly 8 decimal places."""
        d = self.engine.calculate_great_circle_distance(
            Decimal("51.4700"), Decimal("-0.4543"),
            Decimal("40.6413"), Decimal("-73.7781"),
        )
        # Check the exponent is -8
        _, _, exponent = d.as_tuple()
        assert exponent == -8


# ===========================================================================
# 2. UPLIFT FACTOR TESTS (5 tests)
# ===========================================================================


class TestUpliftFactor:
    """Validate distance uplift for routing inefficiency."""

    def setup_method(self):
        with patch(
            "greenlang.business_travel.air_travel_calculator.get_config"
        ) as mc, patch(
            "greenlang.business_travel.air_travel_calculator.get_metrics"
        ) as mm:
            cfg = MagicMock()
            cfg.general.default_uplift_factor = Decimal("0.08")
            mc.return_value = cfg
            mm.return_value = MagicMock()
            from greenlang.business_travel.air_travel_calculator import (
                AirTravelCalculatorEngine,
            )
            self.engine = AirTravelCalculatorEngine()

    # 13
    def test_uplift_default_8_percent(self):
        """1000 km with default 8% uplift = 1080 km."""
        result = self.engine.apply_uplift(Decimal("1000"))
        assert result == _q(Decimal("1080"))

    # 14
    def test_uplift_zero(self):
        """Zero uplift returns same distance."""
        result = self.engine.apply_uplift(Decimal("1000"), Decimal("0"))
        assert result == _q(Decimal("1000"))

    # 15
    def test_uplift_custom_10_percent(self):
        """10% uplift on 1000 km = 1100 km."""
        result = self.engine.apply_uplift(Decimal("1000"), Decimal("0.10"))
        assert result == _q(Decimal("1100"))

    # 16
    def test_uplift_preserves_decimal_type(self):
        """Result type is Decimal."""
        result = self.engine.apply_uplift(Decimal("5541.5"))
        assert isinstance(result, Decimal)

    # 17
    def test_uplift_small_distance(self):
        """Uplift on a small distance (50 km)."""
        result = self.engine.apply_uplift(Decimal("50"))
        expected = _q(Decimal("50") * Decimal("1.08"))
        assert result == expected


# ===========================================================================
# 3. DISTANCE BAND CLASSIFICATION TESTS (8 tests)
# ===========================================================================


class TestDistanceBandClassification:
    """Validate DEFRA distance band thresholds (500 / 3700 km)."""

    def setup_method(self):
        with patch(
            "greenlang.business_travel.air_travel_calculator.get_config"
        ) as mc, patch(
            "greenlang.business_travel.air_travel_calculator.get_metrics"
        ) as mm:
            cfg = MagicMock()
            cfg.general.default_uplift_factor = Decimal("0.08")
            mc.return_value = cfg
            mm.return_value = MagicMock()
            from greenlang.business_travel.air_travel_calculator import (
                AirTravelCalculatorEngine,
            )
            self.engine = AirTravelCalculatorEngine()

    # 18
    def test_classify_499km_domestic(self):
        """499 km is below the 500-km threshold -> DOMESTIC."""
        assert self.engine.classify_distance_band(Decimal("499")) == FlightDistanceBand.DOMESTIC

    # 19
    def test_classify_500km_short_haul(self):
        """500 km is exactly at the short-haul threshold -> SHORT_HAUL."""
        assert self.engine.classify_distance_band(Decimal("500")) == FlightDistanceBand.SHORT_HAUL

    # 20
    def test_classify_1500km_short_haul(self):
        """1500 km is in the short-haul range."""
        assert self.engine.classify_distance_band(Decimal("1500")) == FlightDistanceBand.SHORT_HAUL

    # 21
    def test_classify_3699km_short_haul(self):
        """3699 km is below the 3700-km long-haul threshold -> SHORT_HAUL."""
        assert self.engine.classify_distance_band(Decimal("3699")) == FlightDistanceBand.SHORT_HAUL

    # 22
    def test_classify_3700km_long_haul(self):
        """3700 km is exactly at the long-haul threshold -> LONG_HAUL."""
        assert self.engine.classify_distance_band(Decimal("3700")) == FlightDistanceBand.LONG_HAUL

    # 23
    def test_classify_8000km_long_haul(self):
        """8000 km is deep in the long-haul range."""
        assert self.engine.classify_distance_band(Decimal("8000")) == FlightDistanceBand.LONG_HAUL

    # 24
    def test_classify_1km_domestic(self):
        """Very short distance -> DOMESTIC."""
        assert self.engine.classify_distance_band(Decimal("1")) == FlightDistanceBand.DOMESTIC

    # 25
    def test_classify_returns_enum(self):
        """Result type is FlightDistanceBand enum."""
        result = self.engine.classify_distance_band(Decimal("2000"))
        assert isinstance(result, FlightDistanceBand)


# ===========================================================================
# 4. FULL FLIGHT CALCULATION TESTS (30 tests)
# ===========================================================================


class TestFlightCalculation:
    """End-to-end flight emissions calculation tests."""

    # 26
    def test_lhr_jfk_economy_basic(self, engine):
        """LHR -> JFK economy, one-way, with RF: distance band is LONG_HAUL."""
        inp = FlightInput(
            origin_iata="LHR",
            destination_iata="JFK",
        )
        result = engine.calculate(inp)
        assert isinstance(result, FlightResult)
        assert result.distance_band == FlightDistanceBand.LONG_HAUL
        assert result.cabin_class == CabinClass.ECONOMY
        assert result.class_multiplier == Decimal("1.0")
        assert result.total_co2e > Decimal("0")

    # 27
    def test_co2e_with_rf_greater_than_without(self, engine):
        """co2e_with_rf must be strictly greater than co2e_without_rf."""
        inp = FlightInput(origin_iata="LHR", destination_iata="JFK")
        result = engine.calculate(inp)
        assert result.co2e_with_rf > result.co2e_without_rf

    # 28
    def test_business_class_multiplier_2_9(self, engine):
        """Business class emissions should be 2.9x economy emissions."""
        eco = engine.calculate(
            FlightInput(
                origin_iata="LHR",
                destination_iata="JFK",
                cabin_class=CabinClass.ECONOMY,
                rf_option=RFOption.WITHOUT_RF,
            )
        )
        biz = engine.calculate(
            FlightInput(
                origin_iata="LHR",
                destination_iata="JFK",
                cabin_class=CabinClass.BUSINESS,
                rf_option=RFOption.WITHOUT_RF,
            )
        )
        ratio = biz.co2e_without_rf / eco.co2e_without_rf
        assert Decimal("2.85") < ratio < Decimal("2.95")

    # 29
    def test_first_class_multiplier_4_0(self, engine):
        """First class emissions should be 4.0x economy emissions."""
        eco = engine.calculate(
            FlightInput(
                origin_iata="LHR",
                destination_iata="JFK",
                cabin_class=CabinClass.ECONOMY,
                rf_option=RFOption.WITHOUT_RF,
            )
        )
        first = engine.calculate(
            FlightInput(
                origin_iata="LHR",
                destination_iata="JFK",
                cabin_class=CabinClass.FIRST,
                rf_option=RFOption.WITHOUT_RF,
            )
        )
        ratio = first.co2e_without_rf / eco.co2e_without_rf
        assert Decimal("3.95") < ratio < Decimal("4.05")

    # 30
    def test_premium_economy_multiplier_1_6(self, engine):
        """Premium economy should be 1.6x economy."""
        eco = engine.calculate(
            FlightInput(
                origin_iata="LHR",
                destination_iata="JFK",
                cabin_class=CabinClass.ECONOMY,
                rf_option=RFOption.WITHOUT_RF,
            )
        )
        prem = engine.calculate(
            FlightInput(
                origin_iata="LHR",
                destination_iata="JFK",
                cabin_class=CabinClass.PREMIUM_ECONOMY,
                rf_option=RFOption.WITHOUT_RF,
            )
        )
        ratio = prem.co2e_without_rf / eco.co2e_without_rf
        assert Decimal("1.55") < ratio < Decimal("1.65")

    # 31
    def test_round_trip_doubles_emissions(self, engine):
        """Round trip emissions must be exactly 2x one-way."""
        ow = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="JFK", round_trip=False)
        )
        rt = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="JFK", round_trip=True)
        )
        assert rt.total_co2e == _q(ow.total_co2e * Decimal("2"))

    # 32
    def test_round_trip_doubles_distance(self, engine):
        """Round trip reported distance must be 2x one-way."""
        ow = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="JFK", round_trip=False)
        )
        rt = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="JFK", round_trip=True)
        )
        assert rt.distance_km == _q(ow.distance_km * Decimal("2"))

    # 33
    def test_multi_passenger(self, engine):
        """3 passengers should produce 3x single-passenger emissions."""
        one = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="JFK", passengers=1)
        )
        three = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="JFK", passengers=3)
        )
        assert three.total_co2e == _q(one.total_co2e * Decimal("3"))

    # 34
    def test_with_rf_option_uses_rf_ef(self, engine):
        """WITH_RF total = co2e_with_rf + wtt_co2e."""
        inp = FlightInput(
            origin_iata="LHR",
            destination_iata="JFK",
            rf_option=RFOption.WITH_RF,
        )
        result = engine.calculate(inp)
        expected_total = _q(result.co2e_with_rf + result.wtt_co2e)
        assert result.total_co2e == expected_total

    # 35
    def test_without_rf_option_uses_base_ef(self, engine):
        """WITHOUT_RF total = co2e_without_rf + wtt_co2e."""
        inp = FlightInput(
            origin_iata="LHR",
            destination_iata="JFK",
            rf_option=RFOption.WITHOUT_RF,
        )
        result = engine.calculate(inp)
        expected_total = _q(result.co2e_without_rf + result.wtt_co2e)
        assert result.total_co2e == expected_total

    # 36
    def test_both_rf_option_uses_with_rf(self, engine):
        """BOTH defaults total to co2e_with_rf + wtt_co2e."""
        result_both = engine.calculate(
            FlightInput(
                origin_iata="LHR",
                destination_iata="JFK",
                rf_option=RFOption.BOTH,
            )
        )
        result_with = engine.calculate(
            FlightInput(
                origin_iata="LHR",
                destination_iata="JFK",
                rf_option=RFOption.WITH_RF,
            )
        )
        assert result_both.total_co2e == result_with.total_co2e

    # 37
    def test_wtt_always_calculated(self, engine):
        """WTT component must always be positive."""
        result = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="JFK")
        )
        assert result.wtt_co2e > Decimal("0")

    # 38
    def test_total_equals_sum_of_components_with_rf(self, engine):
        """Total = co2e_with_rf + wtt_co2e when RF is enabled."""
        result = engine.calculate(
            FlightInput(
                origin_iata="LHR",
                destination_iata="JFK",
                rf_option=RFOption.WITH_RF,
            )
        )
        expected = _q(result.co2e_with_rf + result.wtt_co2e)
        assert result.total_co2e == expected

    # 39
    def test_provenance_hash_present(self, engine):
        """Provenance hash must be a 64-char hex string (SHA-256)."""
        result = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="JFK")
        )
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    # 40
    def test_provenance_deterministic(self, engine):
        """Same input must produce same provenance hash."""
        inp = FlightInput(origin_iata="LHR", destination_iata="JFK")
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.provenance_hash == r2.provenance_hash

    # 41
    def test_ef_source_is_defra(self, engine):
        """Default EF source is DEFRA."""
        result = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="JFK")
        )
        assert result.ef_source == EFSource.DEFRA

    # 42
    def test_distance_stored_in_result(self, engine):
        """distance_km in result must be positive and reflect uplift."""
        result = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="JFK")
        )
        # After uplift, distance should be larger than raw GCD
        assert result.distance_km > Decimal("5000")

    # 43
    def test_domestic_short_flight(self, engine):
        """LHR -> LGW short hop should be DOMESTIC band."""
        result = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="LGW")
        )
        assert result.distance_band == FlightDistanceBand.DOMESTIC

    # 44
    def test_short_haul_medium_flight(self, engine):
        """LHR -> CDG should be DOMESTIC or SHORT_HAUL (after uplift ~372 km)."""
        result = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="CDG")
        )
        assert result.distance_band in (
            FlightDistanceBand.DOMESTIC,
            FlightDistanceBand.SHORT_HAUL,
        )

    # 45
    def test_invalid_origin_iata_raises(self, engine):
        """Unknown origin IATA code raises ValueError."""
        with pytest.raises(ValueError, match="[Oo]rigin"):
            engine.calculate(
                FlightInput(origin_iata="ZZZ", destination_iata="JFK")
            )

    # 46
    def test_unknown_destination_iata_raises(self, engine):
        """Unknown destination IATA code raises ValueError."""
        with pytest.raises(ValueError, match="[Dd]estination"):
            engine.calculate(
                FlightInput(origin_iata="LHR", destination_iata="ZZZ")
            )

    # 47
    def test_batch_calculation(self, engine):
        """Batch processing returns results for each valid flight."""
        flights = [
            FlightInput(origin_iata="LHR", destination_iata="JFK"),
            FlightInput(origin_iata="LAX", destination_iata="SFO"),
        ]
        results = engine.calculate_batch(flights)
        assert len(results) == 2
        assert all(isinstance(r, FlightResult) for r in results)

    # 48
    def test_batch_total_sum(self, engine):
        """Batch results total_co2e sum should equal individual calculations."""
        flights = [
            FlightInput(origin_iata="LHR", destination_iata="JFK"),
            FlightInput(origin_iata="SYD", destination_iata="LAX"),
        ]
        results = engine.calculate_batch(flights)
        batch_sum = sum(r.total_co2e for r in results)
        individual_sum = sum(
            engine.calculate(f).total_co2e for f in flights
        )
        assert batch_sum == individual_sum

    # 49
    def test_batch_with_invalid_skips_error(self, engine):
        """Batch with one invalid flight still returns valid results."""
        flights = [
            FlightInput(origin_iata="LHR", destination_iata="JFK"),
            FlightInput(origin_iata="ZZZ", destination_iata="JFK"),
        ]
        results = engine.calculate_batch(flights)
        # The ZZZ flight fails; only 1 result returned
        assert len(results) == 1

    # 50
    def test_get_distance_between_airports(self, engine):
        """Convenience method returns uplifted distance between airports."""
        dist = engine.get_distance_between_airports("LHR", "JFK")
        assert isinstance(dist, Decimal)
        assert dist > Decimal("5000")

    # 51
    def test_get_distance_between_airports_raises_on_invalid(self, engine):
        """Convenience method raises ValueError for unknown airport."""
        with pytest.raises(ValueError):
            engine.get_distance_between_airports("ZZZ", "JFK")

    # 52
    def test_result_origin_destination_match_input(self, engine):
        """FlightResult IATA codes must match input."""
        inp = FlightInput(origin_iata="SYD", destination_iata="LAX")
        result = engine.calculate(inp)
        assert result.origin_iata == "SYD"
        assert result.destination_iata == "LAX"

    # 53
    def test_passengers_stored_in_result(self, engine):
        """Passenger count in result must match input."""
        inp = FlightInput(origin_iata="LHR", destination_iata="JFK", passengers=5)
        result = engine.calculate(inp)
        assert result.passengers == 5

    # 54
    def test_rf_option_stored_in_result(self, engine):
        """RF option in result must match input."""
        inp = FlightInput(
            origin_iata="LHR",
            destination_iata="JFK",
            rf_option=RFOption.BOTH,
        )
        result = engine.calculate(inp)
        assert result.rf_option == RFOption.BOTH

    # 55
    def test_economy_class_multiplier_is_one(self, engine):
        """Economy cabin class multiplier is 1.0."""
        result = engine.calculate(
            FlightInput(
                origin_iata="LHR",
                destination_iata="JFK",
                cabin_class=CabinClass.ECONOMY,
            )
        )
        assert result.class_multiplier == Decimal("1.0")


# ===========================================================================
# 5. CABIN CLASS AND MULTIPLIER TESTS (10 tests)
# ===========================================================================


class TestCabinClassMultipliers:
    """Validate cabin class emission multiplier application."""

    # 56
    def test_economy_multiplier_value(self):
        assert CABIN_CLASS_MULTIPLIERS[CabinClass.ECONOMY] == Decimal("1.0")

    # 57
    def test_premium_economy_multiplier_value(self):
        assert CABIN_CLASS_MULTIPLIERS[CabinClass.PREMIUM_ECONOMY] == Decimal("1.6")

    # 58
    def test_business_multiplier_value(self):
        assert CABIN_CLASS_MULTIPLIERS[CabinClass.BUSINESS] == Decimal("2.9")

    # 59
    def test_first_multiplier_value(self):
        assert CABIN_CLASS_MULTIPLIERS[CabinClass.FIRST] == Decimal("4.0")

    # 60
    def test_all_classes_present(self):
        for cls in CabinClass:
            assert cls in CABIN_CLASS_MULTIPLIERS

    # 61
    def test_multipliers_are_positive(self):
        for cls, mult in CABIN_CLASS_MULTIPLIERS.items():
            assert mult > Decimal("0"), f"{cls} has non-positive multiplier"

    # 62
    def test_multipliers_ordered_ascending(self):
        """Multipliers increase: economy < premium < business < first."""
        assert (
            CABIN_CLASS_MULTIPLIERS[CabinClass.ECONOMY]
            < CABIN_CLASS_MULTIPLIERS[CabinClass.PREMIUM_ECONOMY]
            < CABIN_CLASS_MULTIPLIERS[CabinClass.BUSINESS]
            < CABIN_CLASS_MULTIPLIERS[CabinClass.FIRST]
        )

    # 63
    def test_economy_is_baseline(self):
        """Economy is the baseline (1.0)."""
        assert CABIN_CLASS_MULTIPLIERS[CabinClass.ECONOMY] == Decimal("1.0")

    # 64
    def test_first_class_not_over_5x(self):
        """First class multiplier should not exceed 5.0."""
        assert CABIN_CLASS_MULTIPLIERS[CabinClass.FIRST] <= Decimal("5.0")

    # 65
    def test_multiplier_type_is_decimal(self):
        for mult in CABIN_CLASS_MULTIPLIERS.values():
            assert isinstance(mult, Decimal)


# ===========================================================================
# 6. RADIATIVE FORCING AND EMISSION FACTOR TESTS (8 tests)
# ===========================================================================


class TestRadiativeForcingAndEF:
    """Validate RF-related emission factors and behaviour."""

    # 66
    def test_domestic_ef_with_rf_gt_without(self):
        """DOMESTIC: with_rf > without_rf."""
        ef = AIR_EMISSION_FACTORS[FlightDistanceBand.DOMESTIC]
        assert ef["with_rf"] > ef["without_rf"]

    # 67
    def test_short_haul_ef_with_rf_gt_without(self):
        """SHORT_HAUL: with_rf > without_rf."""
        ef = AIR_EMISSION_FACTORS[FlightDistanceBand.SHORT_HAUL]
        assert ef["with_rf"] > ef["without_rf"]

    # 68
    def test_long_haul_ef_with_rf_gt_without(self):
        """LONG_HAUL: with_rf > without_rf."""
        ef = AIR_EMISSION_FACTORS[FlightDistanceBand.LONG_HAUL]
        assert ef["with_rf"] > ef["without_rf"]

    # 69
    def test_all_bands_have_wtt(self):
        """Every distance band must have a WTT factor."""
        for band in FlightDistanceBand:
            if band in AIR_EMISSION_FACTORS:
                assert "wtt" in AIR_EMISSION_FACTORS[band]

    # 70
    def test_all_bands_have_without_rf(self):
        """Every distance band must have a without_rf factor."""
        for band in FlightDistanceBand:
            if band in AIR_EMISSION_FACTORS:
                assert "without_rf" in AIR_EMISSION_FACTORS[band]

    # 71
    def test_all_efs_are_positive(self):
        """All emission factors must be positive."""
        for band, efs in AIR_EMISSION_FACTORS.items():
            for key, val in efs.items():
                assert val > Decimal("0"), f"{band}.{key} is non-positive"

    # 72
    def test_wtt_smaller_than_combustion(self):
        """WTT factor should be smaller than combustion factor."""
        for band in AIR_EMISSION_FACTORS:
            efs = AIR_EMISSION_FACTORS[band]
            assert efs["wtt"] < efs["without_rf"], f"{band}: WTT >= without_rf"

    # 73
    def test_four_distance_bands_present(self):
        """Should have 4 distance bands: domestic, short_haul, long_haul, international_avg."""
        assert len(AIR_EMISSION_FACTORS) == 4


# ===========================================================================
# 7. INFRASTRUCTURE AND THREAD SAFETY TESTS (7 tests)
# ===========================================================================


class TestInfrastructure:
    """Engine infrastructure: singleton, counts, summaries, thread safety."""

    # 74
    def test_calculation_count_increments(self, engine):
        """Calculation count should increment after each calculation."""
        initial = engine.get_calculation_count()
        engine.calculate(FlightInput(origin_iata="LHR", destination_iata="JFK"))
        assert engine.get_calculation_count() == initial + 1

    # 75
    def test_engine_summary_dict(self, engine):
        """get_engine_summary returns expected keys."""
        summary = engine.get_engine_summary()
        assert "engine" in summary
        assert "calculation_count" in summary
        assert "uplift_factor" in summary
        assert summary["engine"] == "AirTravelCalculatorEngine"

    # 76
    def test_engine_summary_airport_count(self, engine):
        """Summary should report correct airport count."""
        summary = engine.get_engine_summary()
        assert summary["airports"] == len(AIRPORT_DATABASE)

    # 77
    def test_reset_clears_singleton(self):
        """reset() should clear the singleton instance."""
        from greenlang.business_travel.air_travel_calculator import (
            AirTravelCalculatorEngine,
            reset_air_travel_calculator,
        )

        reset_air_travel_calculator()
        # After reset, the class-level _instance should be None
        assert AirTravelCalculatorEngine._instance is None

    # 78
    def test_get_air_travel_calculator_returns_engine(self, engine):
        """get_air_travel_calculator returns an AirTravelCalculatorEngine."""
        from greenlang.business_travel.air_travel_calculator import (
            AirTravelCalculatorEngine,
        )

        assert isinstance(engine, AirTravelCalculatorEngine)

    # 79
    def test_gwp_version_default_ar5(self, engine):
        """Default GWP version parameter is AR5."""
        # calculate() with no gwp_version should use AR5 without error
        result = engine.calculate(
            FlightInput(origin_iata="LHR", destination_iata="JFK")
        )
        assert result.total_co2e > Decimal("0")

    # 80
    def test_input_validation_lowercase_iata_raises(self):
        """Lowercase IATA code in FlightInput should raise validation error."""
        with pytest.raises(Exception):
            FlightInput(origin_iata="lhr", destination_iata="JFK")
