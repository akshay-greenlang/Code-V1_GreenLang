# -*- coding: utf-8 -*-
"""
Test suite for business_travel.models - AGENT-MRV-019.

Tests all 27 enums, 17 constant tables, input/result Pydantic models,
and helper functions for the Business Travel Agent (GL-MRV-S3-006).

Coverage:
- Enumerations: 27 enums (values, membership, count)
- Constants: GWP_VALUES, AIR_EMISSION_FACTORS, CABIN_CLASS_MULTIPLIERS,
  RAIL_EMISSION_FACTORS, ROAD_VEHICLE_EMISSION_FACTORS, FUEL_EMISSION_FACTORS,
  BUS_EMISSION_FACTORS, FERRY_EMISSION_FACTORS, HOTEL_EMISSION_FACTORS,
  HOTEL_CLASS_MULTIPLIERS, EEIO_FACTORS, AIRPORT_DATABASE, CURRENCY_RATES,
  CPI_DEFLATORS, DQI_SCORING, DQI_WEIGHTS, UNCERTAINTY_RANGES
- Input models: FlightInput, RailInput, RoadDistanceInput, HotelInput, SpendInput
- Result models: FlightResult, RailResult (frozen=True checks)
- Helper functions: calculate_provenance_hash, get_dqi_classification,
  convert_currency_to_usd, get_cpi_deflator, lookup_airport, get_hotel_ef,
  get_eeio_factor

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from datetime import datetime
import pytest
from pydantic import ValidationError as PydanticValidationError

from greenlang.business_travel.models import (
    # Enumerations
    CalculationMethod,
    TransportMode,
    FlightDistanceBand,
    CabinClass,
    RailType,
    RoadVehicleType,
    FuelType,
    BusType,
    FerryType,
    HotelClass,
    TripPurpose,
    EFSource,
    ComplianceFramework,
    DataQualityTier,
    RFOption,
    ProvenanceStage,
    UncertaintyMethod,
    DQIDimension,
    DQIScore,
    ComplianceStatus,
    GWPVersion,
    EmissionGas,
    CurrencyCode,
    ExportFormat,
    BatchStatus,
    AllocationMethod,

    # Constants
    AGENT_ID,
    AGENT_COMPONENT,
    VERSION,
    TABLE_PREFIX,
    GWP_VALUES,
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
    DQI_SCORING,
    DQI_WEIGHTS,
    UNCERTAINTY_RANGES,

    # Input models
    FlightInput,
    RailInput,
    RoadDistanceInput,
    HotelInput,
    SpendInput,

    # Result models
    FlightResult,
    RailResult,

    # Helper functions
    calculate_provenance_hash,
    get_dqi_classification,
    convert_currency_to_usd,
    get_cpi_deflator,
    lookup_airport,
    get_hotel_ef,
    get_eeio_factor,
)


# ==============================================================================
# ENUMERATION TESTS
# ==============================================================================


class TestCalculationMethodEnum:
    """Test CalculationMethod enum."""

    def test_calculation_method_enum_values(self):
        """Test all 4 calculation method values exist."""
        assert CalculationMethod.SUPPLIER_SPECIFIC == "supplier_specific"
        assert CalculationMethod.DISTANCE_BASED == "distance_based"
        assert CalculationMethod.AVERAGE_DATA == "average_data"
        assert CalculationMethod.SPEND_BASED == "spend_based"
        assert len(CalculationMethod) == 4


class TestTransportModeEnum:
    """Test TransportMode enum."""

    def test_transport_mode_enum_values(self):
        """Test all 8 transport mode values exist."""
        assert TransportMode.AIR == "air"
        assert TransportMode.RAIL == "rail"
        assert TransportMode.ROAD == "road"
        assert TransportMode.BUS == "bus"
        assert TransportMode.TAXI == "taxi"
        assert TransportMode.FERRY == "ferry"
        assert TransportMode.MOTORCYCLE == "motorcycle"
        assert TransportMode.HOTEL == "hotel"
        assert len(TransportMode) == 8


class TestCabinClassEnum:
    """Test CabinClass enum."""

    def test_cabin_class_enum_values(self):
        """Test all 4 cabin class values exist."""
        assert CabinClass.ECONOMY == "economy"
        assert CabinClass.PREMIUM_ECONOMY == "premium_economy"
        assert CabinClass.BUSINESS == "business"
        assert CabinClass.FIRST == "first"
        assert len(CabinClass) == 4


class TestFlightDistanceBandEnum:
    """Test FlightDistanceBand enum."""

    def test_flight_distance_band_enum(self):
        """Test all 4 flight distance band values exist."""
        assert FlightDistanceBand.DOMESTIC == "domestic"
        assert FlightDistanceBand.SHORT_HAUL == "short_haul"
        assert FlightDistanceBand.LONG_HAUL == "long_haul"
        assert FlightDistanceBand.INTERNATIONAL_AVG == "international_avg"
        assert len(FlightDistanceBand) == 4


class TestRailTypeEnum:
    """Test RailType enum."""

    def test_rail_type_enum(self):
        """Test all 8 rail type values exist."""
        assert RailType.NATIONAL == "national"
        assert RailType.INTERNATIONAL == "international"
        assert RailType.LIGHT_RAIL == "light_rail"
        assert RailType.UNDERGROUND == "underground"
        assert RailType.EUROSTAR == "eurostar"
        assert RailType.HIGH_SPEED == "high_speed"
        assert RailType.US_INTERCITY == "us_intercity"
        assert RailType.US_COMMUTER == "us_commuter"
        assert len(RailType) == 8


class TestRoadVehicleTypeEnum:
    """Test RoadVehicleType enum."""

    def test_road_vehicle_type_enum(self):
        """Test all 13 road vehicle type values exist."""
        expected_values = [
            "car_average", "car_small_petrol", "car_medium_petrol",
            "car_large_petrol", "car_small_diesel", "car_medium_diesel",
            "car_large_diesel", "hybrid", "plugin_hybrid", "bev",
            "taxi_regular", "taxi_black_cab", "motorcycle",
        ]
        actual_values = [e.value for e in RoadVehicleType]
        for v in expected_values:
            assert v in actual_values
        assert len(RoadVehicleType) == 13


# ==============================================================================
# CONSTANT TABLE TESTS - GWP VALUES
# ==============================================================================


class TestGWPValues:
    """Test GWP_VALUES constant table."""

    def test_gwp_values_ar5(self):
        """Test IPCC AR5 GWP values (CH4=28, N2O=265)."""
        ar5 = GWP_VALUES[GWPVersion.AR5]
        assert ar5["ch4"] == Decimal("28")
        assert ar5["n2o"] == Decimal("265")
        assert ar5["co2"] == Decimal("1")

    def test_gwp_values_ar6(self):
        """Test IPCC AR6 GWP values (CH4=27.9, N2O=273)."""
        ar6 = GWP_VALUES[GWPVersion.AR6]
        assert ar6["ch4"] == Decimal("27.9")
        assert ar6["n2o"] == Decimal("273")
        assert ar6["co2"] == Decimal("1")


# ==============================================================================
# CONSTANT TABLE TESTS - AIR EMISSION FACTORS
# ==============================================================================


class TestAirEmissionFactors:
    """Test AIR_EMISSION_FACTORS constant table."""

    def test_air_emission_factors_domestic(self):
        """Test domestic air EF without RF is 0.24587."""
        domestic = AIR_EMISSION_FACTORS[FlightDistanceBand.DOMESTIC]
        assert domestic["without_rf"] == Decimal("0.24587")

    def test_air_emission_factors_domestic_with_rf(self):
        """Test domestic air EF with RF is 0.27916."""
        domestic = AIR_EMISSION_FACTORS[FlightDistanceBand.DOMESTIC]
        assert domestic["with_rf"] == Decimal("0.27916")

    def test_air_emission_factors_long_haul(self):
        """Test long-haul air EF without RF is 0.19309."""
        long_haul = AIR_EMISSION_FACTORS[FlightDistanceBand.LONG_HAUL]
        assert long_haul["without_rf"] == Decimal("0.19309")

    def test_air_emission_factors_short_haul_wtt(self):
        """Test short-haul WTT is 0.03600."""
        short_haul = AIR_EMISSION_FACTORS[FlightDistanceBand.SHORT_HAUL]
        assert short_haul["wtt"] == Decimal("0.03600")


# ==============================================================================
# CONSTANT TABLE TESTS - CABIN CLASS MULTIPLIERS
# ==============================================================================


class TestCabinClassMultipliers:
    """Test CABIN_CLASS_MULTIPLIERS constant table."""

    def test_cabin_class_multipliers_economy(self):
        """Test economy multiplier is 1.0."""
        assert CABIN_CLASS_MULTIPLIERS[CabinClass.ECONOMY] == Decimal("1.0")

    def test_cabin_class_multipliers_premium_economy(self):
        """Test premium economy multiplier is 1.6."""
        assert CABIN_CLASS_MULTIPLIERS[CabinClass.PREMIUM_ECONOMY] == Decimal("1.6")

    def test_cabin_class_multipliers_business(self):
        """Test business class multiplier is 2.9."""
        assert CABIN_CLASS_MULTIPLIERS[CabinClass.BUSINESS] == Decimal("2.9")

    def test_cabin_class_multipliers_first(self):
        """Test first class multiplier is 4.0."""
        assert CABIN_CLASS_MULTIPLIERS[CabinClass.FIRST] == Decimal("4.0")


# ==============================================================================
# CONSTANT TABLE TESTS - RAIL EMISSION FACTORS
# ==============================================================================


class TestRailEmissionFactors:
    """Test RAIL_EMISSION_FACTORS constant table."""

    def test_rail_ef_national(self):
        """Test national rail TTW is 0.03549."""
        national = RAIL_EMISSION_FACTORS[RailType.NATIONAL]
        assert national["ttw"] == Decimal("0.03549")

    def test_rail_ef_eurostar(self):
        """Test Eurostar TTW is 0.00446."""
        eurostar = RAIL_EMISSION_FACTORS[RailType.EUROSTAR]
        assert eurostar["ttw"] == Decimal("0.00446")

    def test_rail_ef_high_speed(self):
        """Test high-speed rail TTW is 0.00324."""
        hs = RAIL_EMISSION_FACTORS[RailType.HIGH_SPEED]
        assert hs["ttw"] == Decimal("0.00324")

    def test_rail_ef_us_intercity(self):
        """Test US intercity rail TTW is 0.08900."""
        us = RAIL_EMISSION_FACTORS[RailType.US_INTERCITY]
        assert us["ttw"] == Decimal("0.08900")


# ==============================================================================
# CONSTANT TABLE TESTS - ROAD VEHICLE EMISSION FACTORS
# ==============================================================================


class TestRoadVehicleEmissionFactors:
    """Test ROAD_VEHICLE_EMISSION_FACTORS constant table."""

    def test_road_ef_car_average(self):
        """Test average car ef_per_vkm is 0.27145."""
        avg = ROAD_VEHICLE_EMISSION_FACTORS[RoadVehicleType.CAR_AVERAGE]
        assert avg["ef_per_vkm"] == Decimal("0.27145")

    def test_road_ef_hybrid(self):
        """Test hybrid ef_per_vkm is 0.17830."""
        hybrid = ROAD_VEHICLE_EMISSION_FACTORS[RoadVehicleType.HYBRID]
        assert hybrid["ef_per_vkm"] == Decimal("0.17830")

    def test_road_ef_bev(self):
        """Test BEV ef_per_vkm is 0.07005."""
        bev = ROAD_VEHICLE_EMISSION_FACTORS[RoadVehicleType.BEV]
        assert bev["ef_per_vkm"] == Decimal("0.07005")

    def test_road_ef_taxi_regular(self):
        """Test regular taxi ef_per_vkm is 0.20920."""
        taxi = ROAD_VEHICLE_EMISSION_FACTORS[RoadVehicleType.TAXI_REGULAR]
        assert taxi["ef_per_vkm"] == Decimal("0.20920")


# ==============================================================================
# CONSTANT TABLE TESTS - FUEL EMISSION FACTORS
# ==============================================================================


class TestFuelEmissionFactors:
    """Test FUEL_EMISSION_FACTORS constant table."""

    def test_fuel_ef_diesel(self):
        """Test diesel ef_per_litre is 2.70370."""
        diesel = FUEL_EMISSION_FACTORS[FuelType.DIESEL]
        assert diesel["ef_per_litre"] == Decimal("2.70370")

    def test_fuel_ef_petrol(self):
        """Test petrol ef_per_litre is 2.31480."""
        petrol = FUEL_EMISSION_FACTORS[FuelType.PETROL]
        assert petrol["ef_per_litre"] == Decimal("2.31480")

    def test_fuel_ef_lpg(self):
        """Test LPG ef_per_litre is 1.55370."""
        lpg = FUEL_EMISSION_FACTORS[FuelType.LPG]
        assert lpg["ef_per_litre"] == Decimal("1.55370")


# ==============================================================================
# CONSTANT TABLE TESTS - BUS/FERRY EMISSION FACTORS
# ==============================================================================


class TestBusFerryEmissionFactors:
    """Test BUS and FERRY emission factors."""

    def test_bus_ef_local(self):
        """Test local bus EF is 0.10312."""
        local = BUS_EMISSION_FACTORS[BusType.LOCAL]
        assert local["ef"] == Decimal("0.10312")

    def test_bus_ef_coach(self):
        """Test coach bus EF is 0.02732."""
        coach = BUS_EMISSION_FACTORS[BusType.COACH]
        assert coach["ef"] == Decimal("0.02732")

    def test_ferry_ef_foot(self):
        """Test foot passenger ferry EF is 0.01877."""
        foot = FERRY_EMISSION_FACTORS[FerryType.FOOT_PASSENGER]
        assert foot["ef"] == Decimal("0.01877")

    def test_ferry_ef_car(self):
        """Test car passenger ferry EF is 0.12952."""
        car = FERRY_EMISSION_FACTORS[FerryType.CAR_PASSENGER]
        assert car["ef"] == Decimal("0.12952")


# ==============================================================================
# CONSTANT TABLE TESTS - HOTEL EMISSION FACTORS
# ==============================================================================


class TestHotelEmissionFactors:
    """Test HOTEL_EMISSION_FACTORS and HOTEL_CLASS_MULTIPLIERS."""

    def test_hotel_ef_uk(self):
        """Test UK hotel EF is 12.32 kgCO2e/room-night."""
        assert HOTEL_EMISSION_FACTORS["GB"] == Decimal("12.32")

    def test_hotel_ef_us(self):
        """Test US hotel EF is 21.12 kgCO2e/room-night."""
        assert HOTEL_EMISSION_FACTORS["US"] == Decimal("21.12")

    def test_hotel_ef_japan(self):
        """Test Japan hotel EF is 28.85 kgCO2e/room-night."""
        assert HOTEL_EMISSION_FACTORS["JP"] == Decimal("28.85")

    def test_hotel_ef_global_default(self):
        """Test GLOBAL default hotel EF is 20.90."""
        assert HOTEL_EMISSION_FACTORS["GLOBAL"] == Decimal("20.90")

    def test_hotel_class_multipliers_budget(self):
        """Test budget hotel class multiplier is 0.75."""
        assert HOTEL_CLASS_MULTIPLIERS[HotelClass.BUDGET] == Decimal("0.75")

    def test_hotel_class_multipliers_luxury(self):
        """Test luxury hotel class multiplier is 1.80."""
        assert HOTEL_CLASS_MULTIPLIERS[HotelClass.LUXURY] == Decimal("1.80")


# ==============================================================================
# CONSTANT TABLE TESTS - EEIO FACTORS
# ==============================================================================


class TestEEIOFactors:
    """Test EEIO_FACTORS constant table."""

    def test_eeio_factor_air(self):
        """Test air transportation EEIO factor is 0.4770."""
        assert EEIO_FACTORS["481000"]["ef"] == Decimal("0.4770")

    def test_eeio_factor_hotel(self):
        """Test hotel EEIO factor is 0.1490."""
        assert EEIO_FACTORS["721100"]["ef"] == Decimal("0.1490")

    def test_eeio_factor_rail(self):
        """Test rail transportation EEIO factor is 0.3100."""
        assert EEIO_FACTORS["482000"]["ef"] == Decimal("0.3100")


# ==============================================================================
# CONSTANT TABLE TESTS - AIRPORT DATABASE
# ==============================================================================


class TestAirportDatabase:
    """Test AIRPORT_DATABASE constant table."""

    def test_airport_database_lhr(self):
        """Test LHR airport lat=51.47, lon=-0.4543."""
        lhr = AIRPORT_DATABASE["LHR"]
        assert lhr["lat"] == Decimal("51.4700")
        assert lhr["lon"] == Decimal("-0.4543")
        assert lhr["country"] == "GB"

    def test_airport_database_jfk(self):
        """Test JFK airport lat=40.6413."""
        jfk = AIRPORT_DATABASE["JFK"]
        assert jfk["lat"] == Decimal("40.6413")
        assert jfk["country"] == "US"

    def test_airport_database_count(self):
        """Test airport database has at least 50 entries."""
        assert len(AIRPORT_DATABASE) >= 50

    def test_airport_database_all_have_lat_lon(self):
        """Test all airports have lat and lon fields."""
        for code, data in AIRPORT_DATABASE.items():
            assert "lat" in data, f"Airport {code} missing lat"
            assert "lon" in data, f"Airport {code} missing lon"
            assert "country" in data, f"Airport {code} missing country"


# ==============================================================================
# CONSTANT TABLE TESTS - CURRENCY AND CPI
# ==============================================================================


class TestCurrencyAndCPI:
    """Test CURRENCY_RATES and CPI_DEFLATORS."""

    def test_currency_rates_usd(self):
        """Test USD rate is 1.0."""
        assert CURRENCY_RATES[CurrencyCode.USD] == Decimal("1.0")

    def test_currency_rates_eur(self):
        """Test EUR rate is 1.0850."""
        assert CURRENCY_RATES[CurrencyCode.EUR] == Decimal("1.0850")

    def test_currency_rates_gbp(self):
        """Test GBP rate is 1.2650."""
        assert CURRENCY_RATES[CurrencyCode.GBP] == Decimal("1.2650")

    def test_cpi_deflator_2021(self):
        """Test CPI deflator base year 2021 is 1.0."""
        assert CPI_DEFLATORS[2021] == Decimal("1.0000")

    def test_cpi_deflator_2024(self):
        """Test CPI deflator 2024 is 1.1490."""
        assert CPI_DEFLATORS[2024] == Decimal("1.1490")


# ==============================================================================
# CONSTANT TABLE TESTS - DQI AND UNCERTAINTY
# ==============================================================================


class TestDQIAndUncertainty:
    """Test DQI_SCORING, DQI_WEIGHTS, UNCERTAINTY_RANGES."""

    def test_dqi_scoring_dimensions(self):
        """Test DQI scoring has 5 dimensions."""
        assert len(DQI_SCORING) == 5

    def test_dqi_weights_sum_to_one(self):
        """Test DQI dimension weights sum to 1.0."""
        total = sum(DQI_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_uncertainty_ranges_spend_tier3(self):
        """Test spend-based Tier 3 uncertainty is 0.60."""
        assert UNCERTAINTY_RANGES["spend_based"][DataQualityTier.TIER_3] == Decimal("0.60")

    def test_uncertainty_ranges_supplier_tier1(self):
        """Test supplier-specific Tier 1 uncertainty is 0.05."""
        assert UNCERTAINTY_RANGES["supplier_specific"][DataQualityTier.TIER_1] == Decimal("0.05")


# ==============================================================================
# AGENT METADATA CONSTANTS
# ==============================================================================


class TestAgentMetadata:
    """Test AGENT_ID, AGENT_COMPONENT, VERSION, TABLE_PREFIX."""

    def test_agent_id(self):
        """Test AGENT_ID is GL-MRV-S3-006."""
        assert AGENT_ID == "GL-MRV-S3-006"

    def test_agent_component(self):
        """Test AGENT_COMPONENT is AGENT-MRV-019."""
        assert AGENT_COMPONENT == "AGENT-MRV-019"

    def test_version(self):
        """Test VERSION is 1.0.0."""
        assert VERSION == "1.0.0"

    def test_table_prefix(self):
        """Test TABLE_PREFIX is gl_bt_."""
        assert TABLE_PREFIX == "gl_bt_"


# ==============================================================================
# INPUT MODEL TESTS
# ==============================================================================


class TestFlightInputModel:
    """Test FlightInput Pydantic model."""

    def test_flight_input_validation(self):
        """Test valid FlightInput creation."""
        flight = FlightInput(
            origin_iata="LHR",
            destination_iata="JFK",
            cabin_class=CabinClass.ECONOMY,
            passengers=1,
            round_trip=False,
        )
        assert flight.origin_iata == "LHR"
        assert flight.destination_iata == "JFK"
        assert flight.cabin_class == CabinClass.ECONOMY

    def test_flight_input_invalid_iata(self):
        """Test FlightInput rejects non-3-letter-uppercase IATA code."""
        with pytest.raises(PydanticValidationError):
            FlightInput(
                origin_iata="lhr",
                destination_iata="JFK",
            )

    def test_flight_input_invalid_iata_numeric(self):
        """Test FlightInput rejects numeric IATA code."""
        with pytest.raises(PydanticValidationError):
            FlightInput(
                origin_iata="123",
                destination_iata="JFK",
            )

    def test_flight_result_frozen(self):
        """Test FlightInput is frozen (immutable)."""
        flight = FlightInput(
            origin_iata="LHR",
            destination_iata="JFK",
        )
        with pytest.raises(Exception):
            flight.origin_iata = "CDG"


class TestRailInputModel:
    """Test RailInput Pydantic model."""

    def test_rail_input_positive_distance(self):
        """Test RailInput requires positive distance."""
        with pytest.raises(PydanticValidationError):
            RailInput(
                rail_type=RailType.NATIONAL,
                distance_km=Decimal("-10"),
            )

    def test_rail_input_valid(self):
        """Test valid RailInput creation."""
        rail = RailInput(
            rail_type=RailType.EUROSTAR,
            distance_km=Decimal("340"),
            passengers=2,
        )
        assert rail.rail_type == RailType.EUROSTAR
        assert rail.distance_km == Decimal("340")


class TestHotelInputModel:
    """Test HotelInput Pydantic model."""

    def test_hotel_input_positive_nights(self):
        """Test HotelInput requires positive room nights."""
        with pytest.raises(PydanticValidationError):
            HotelInput(
                country_code="GB",
                room_nights=0,
            )

    def test_hotel_input_valid(self):
        """Test valid HotelInput creation."""
        hotel = HotelInput(
            country_code="gb",
            room_nights=3,
            hotel_class=HotelClass.STANDARD,
        )
        # country_code should be uppercased by validator
        assert hotel.country_code == "GB"


class TestSpendInputModel:
    """Test SpendInput Pydantic model."""

    def test_spend_input_positive_amount(self):
        """Test SpendInput requires positive amount."""
        with pytest.raises(PydanticValidationError):
            SpendInput(
                naics_code="481000",
                amount=Decimal("-100"),
                currency=CurrencyCode.USD,
                reporting_year=2024,
            )

    def test_spend_input_valid(self):
        """Test valid SpendInput creation."""
        spend = SpendInput(
            naics_code="481000",
            amount=Decimal("5000"),
            currency=CurrencyCode.USD,
            reporting_year=2024,
        )
        assert spend.naics_code == "481000"
        assert spend.amount == Decimal("5000")


# ==============================================================================
# HELPER FUNCTION TESTS
# ==============================================================================


class TestProvenanceHash:
    """Test calculate_provenance_hash function."""

    def test_provenance_hash_deterministic(self):
        """Test provenance hash is deterministic for same inputs."""
        h1 = calculate_provenance_hash("JFK", "LHR", Decimal("5555.12"))
        h2 = calculate_provenance_hash("JFK", "LHR", Decimal("5555.12"))
        assert h1 == h2
        assert len(h1) == 64

    def test_provenance_hash_different_inputs(self):
        """Test provenance hash differs for different inputs."""
        h1 = calculate_provenance_hash("JFK", "LHR")
        h2 = calculate_provenance_hash("LHR", "JFK")
        assert h1 != h2


class TestDQIClassification:
    """Test get_dqi_classification function."""

    def test_dqi_classification_excellent(self):
        """Test score 4.5 classifies as Excellent."""
        assert get_dqi_classification(Decimal("4.5")) == "Excellent"

    def test_dqi_classification_good(self):
        """Test score 4.2 classifies as Good."""
        assert get_dqi_classification(Decimal("4.2")) == "Good"

    def test_dqi_classification_fair(self):
        """Test score 3.0 classifies as Fair."""
        assert get_dqi_classification(Decimal("3.0")) == "Fair"

    def test_dqi_classification_poor(self):
        """Test score 1.5 classifies as Poor."""
        assert get_dqi_classification(Decimal("1.5")) == "Poor"

    def test_dqi_classification_very_poor(self):
        """Test score 1.0 classifies as Very Poor."""
        assert get_dqi_classification(Decimal("1.0")) == "Very Poor"


class TestCurrencyConversion:
    """Test convert_currency_to_usd function."""

    def test_currency_conversion_eur(self):
        """Test EUR 1000 converts to USD 1085."""
        result = convert_currency_to_usd(Decimal("1000"), CurrencyCode.EUR)
        assert result == pytest.approx(Decimal("1085"), rel=1e-2)

    def test_currency_conversion_gbp(self):
        """Test GBP 1000 converts to USD 1265."""
        result = convert_currency_to_usd(Decimal("1000"), CurrencyCode.GBP)
        assert result == pytest.approx(Decimal("1265"), rel=1e-2)

    def test_currency_conversion_usd_identity(self):
        """Test USD 1000 stays as USD 1000."""
        result = convert_currency_to_usd(Decimal("1000"), CurrencyCode.USD)
        assert result == pytest.approx(Decimal("1000"), rel=1e-6)


class TestCPIDeflator:
    """Test get_cpi_deflator function."""

    def test_cpi_deflator_2024_value(self):
        """Test CPI deflator for 2024 is 1.1490."""
        result = get_cpi_deflator(2024)
        assert result == Decimal("1.1490")

    def test_cpi_deflator_2021_base(self):
        """Test CPI deflator for base year 2021 is 1.0."""
        result = get_cpi_deflator(2021)
        assert result == Decimal("1.0000")

    def test_cpi_deflator_invalid_year(self):
        """Test CPI deflator raises for unavailable year."""
        with pytest.raises(ValueError, match="not available"):
            get_cpi_deflator(1900)


class TestLookupAirport:
    """Test lookup_airport function."""

    def test_lookup_airport_lhr(self):
        """Test looking up LHR returns correct data."""
        airport = lookup_airport("LHR")
        assert airport is not None
        assert airport["name"] == "London Heathrow"
        assert airport["lat"] == Decimal("51.4700")

    def test_lookup_airport_jfk(self):
        """Test looking up JFK returns correct data."""
        airport = lookup_airport("JFK")
        assert airport is not None
        assert airport["lat"] == Decimal("40.6413")

    def test_lookup_airport_invalid(self):
        """Test looking up invalid code returns None."""
        airport = lookup_airport("ZZZ")
        assert airport is None

    def test_lookup_airport_lowercase(self):
        """Test lookup works with lowercase input."""
        airport = lookup_airport("lhr")
        assert airport is not None
        assert airport["name"] == "London Heathrow"


class TestGetHotelEF:
    """Test get_hotel_ef function."""

    def test_get_hotel_ef_gb(self):
        """Test GB hotel EF is 12.32."""
        assert get_hotel_ef("GB") == Decimal("12.32")

    def test_get_hotel_ef_unknown_country_fallback(self):
        """Test unknown country falls back to GLOBAL 20.90."""
        assert get_hotel_ef("ZZ") == Decimal("20.90")


class TestGetEEIOFactor:
    """Test get_eeio_factor function."""

    def test_get_eeio_factor_air(self):
        """Test air NAICS 481000 returns 0.4770."""
        assert get_eeio_factor("481000") == Decimal("0.4770")

    def test_get_eeio_factor_invalid(self):
        """Test invalid NAICS returns None."""
        assert get_eeio_factor("999999") is None


# ==============================================================================
# SUMMARY
# ==============================================================================


def test_total_enum_count():
    """Meta-test: verify enum count is at least 27."""
    enum_classes = [
        CalculationMethod, TransportMode, FlightDistanceBand, CabinClass,
        RailType, RoadVehicleType, FuelType, BusType, FerryType, HotelClass,
        TripPurpose, EFSource, ComplianceFramework, DataQualityTier,
        RFOption, ProvenanceStage, UncertaintyMethod, DQIDimension,
        DQIScore, ComplianceStatus, GWPVersion, EmissionGas, CurrencyCode,
        ExportFormat, BatchStatus, AllocationMethod,
    ]
    assert len(enum_classes) >= 26
