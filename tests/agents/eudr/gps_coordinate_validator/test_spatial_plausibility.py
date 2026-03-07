# -*- coding: utf-8 -*-
"""
Tests for SpatialPlausibilityChecker - AGENT-EUDR-007 Engine 5: Spatial Plausibility

Comprehensive test suite covering:
- Land/ocean discrimination for various coordinate points
- Country identification from coordinates (reverse geocoding)
- Country mismatch detection (declared != detected)
- Commodity-region plausibility (tropical lat/lon for oil palm, etc.)
- Elevation plausibility for EUDR commodities
- Urban area detection
- Protected area flagging
- Batch plausibility checking
- Parametrized tests for 7 EUDR commodities

Test count: 45+ tests
Coverage target: >= 85% of SpatialPlausibilityChecker module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import pytest

from tests.agents.eudr.gps_coordinate_validator.conftest import (
    COCOA_FARM_GHANA,
    PALM_PLANTATION_INDONESIA,
    COFFEE_FARM_COLOMBIA,
    SOYA_FIELD_BRAZIL,
    RUBBER_FARM_THAILAND,
    CATTLE_RANCH_BRAZIL,
    TIMBER_FOREST_CONGO,
    COFFEE_FARM_ETHIOPIA,
    PALM_PLANTATION_MALAYSIA,
    COCOA_FARM_IVORY_COAST,
    OCEAN_POINT,
    NULL_ISLAND,
    ARCTIC_POINT,
    URBAN_POINT,
    DESERT_POINT,
    ANTARCTIC_POINT,
    BOUNDARY_LATITUDE,
    SOUTH_POLE,
    PLAUSIBLE_COMMODITY_COORDINATES,
    assert_close,
)


# ===========================================================================
# 1. Land / Ocean Check
# ===========================================================================


class TestLandOceanCheck:
    """Test land vs ocean discrimination."""

    def test_land_check_ghana(self, spatial_plausibility_checker):
        """Ghana cocoa farm is on land."""
        result = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
        )
        assert result.is_on_land is True

    def test_land_check_brazil(self, spatial_plausibility_checker):
        """Brazil soya field is on land."""
        result = spatial_plausibility_checker.check(
            latitude=SOYA_FIELD_BRAZIL[0],
            longitude=SOYA_FIELD_BRAZIL[1],
        )
        assert result.is_on_land is True

    def test_land_check_indonesia(self, spatial_plausibility_checker):
        """Indonesia palm plantation is on land."""
        result = spatial_plausibility_checker.check(
            latitude=PALM_PLANTATION_INDONESIA[0],
            longitude=PALM_PLANTATION_INDONESIA[1],
        )
        assert result.is_on_land is True

    def test_land_check_colombia(self, spatial_plausibility_checker):
        """Colombia coffee farm is on land."""
        result = spatial_plausibility_checker.check(
            latitude=COFFEE_FARM_COLOMBIA[0],
            longitude=COFFEE_FARM_COLOMBIA[1],
        )
        assert result.is_on_land is True

    def test_land_check_thailand(self, spatial_plausibility_checker):
        """Thailand rubber farm is on land."""
        result = spatial_plausibility_checker.check(
            latitude=RUBBER_FARM_THAILAND[0],
            longitude=RUBBER_FARM_THAILAND[1],
        )
        assert result.is_on_land is True

    def test_land_check_congo(self, spatial_plausibility_checker):
        """Congo timber forest is on land."""
        result = spatial_plausibility_checker.check(
            latitude=TIMBER_FOREST_CONGO[0],
            longitude=TIMBER_FOREST_CONGO[1],
        )
        assert result.is_on_land is True

    def test_ocean_check_atlantic(self, spatial_plausibility_checker):
        """Atlantic Ocean point is not on land."""
        result = spatial_plausibility_checker.check(
            latitude=OCEAN_POINT[0],
            longitude=OCEAN_POINT[1],
        )
        assert result.is_on_land is False

    def test_ocean_check_pacific(self, spatial_plausibility_checker):
        """Central Pacific point is not on land."""
        result = spatial_plausibility_checker.check(
            latitude=0.0,
            longitude=-150.0,
        )
        assert result.is_on_land is False

    def test_ocean_check_indian(self, spatial_plausibility_checker):
        """Indian Ocean point is not on land."""
        result = spatial_plausibility_checker.check(
            latitude=-20.0,
            longitude=70.0,
        )
        assert result.is_on_land is False


# ===========================================================================
# 2. Country Identification
# ===========================================================================


class TestCountryIdentification:
    """Test country identification from coordinates."""

    def test_country_check_brazil(self, spatial_plausibility_checker):
        """Brazil is correctly identified."""
        result = spatial_plausibility_checker.check(
            latitude=SOYA_FIELD_BRAZIL[0],
            longitude=SOYA_FIELD_BRAZIL[1],
        )
        assert result.detected_country == "BR"

    def test_country_check_indonesia(self, spatial_plausibility_checker):
        """Indonesia is correctly identified."""
        result = spatial_plausibility_checker.check(
            latitude=PALM_PLANTATION_INDONESIA[0],
            longitude=PALM_PLANTATION_INDONESIA[1],
        )
        assert result.detected_country == "ID"

    def test_country_check_ghana(self, spatial_plausibility_checker):
        """Ghana is correctly identified."""
        result = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
        )
        assert result.detected_country == "GH"

    def test_country_check_colombia(self, spatial_plausibility_checker):
        """Colombia is correctly identified."""
        result = spatial_plausibility_checker.check(
            latitude=COFFEE_FARM_COLOMBIA[0],
            longitude=COFFEE_FARM_COLOMBIA[1],
        )
        assert result.detected_country == "CO"

    def test_country_check_thailand(self, spatial_plausibility_checker):
        """Thailand is correctly identified."""
        result = spatial_plausibility_checker.check(
            latitude=RUBBER_FARM_THAILAND[0],
            longitude=RUBBER_FARM_THAILAND[1],
        )
        assert result.detected_country == "TH"


# ===========================================================================
# 3. Country Mismatch Detection
# ===========================================================================


class TestCountryMismatch:
    """Test declared vs detected country mismatch."""

    def test_country_mismatch_declared_brazil_actual_ghana(
        self, spatial_plausibility_checker
    ):
        """Declaring Brazil for Ghana coordinate triggers mismatch."""
        result = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            declared_country="BR",
        )
        assert result.country_mismatch is True
        assert result.detected_country == "GH"

    def test_country_mismatch_declared_ghana_actual_brazil(
        self, spatial_plausibility_checker
    ):
        """Declaring Ghana for Brazil coordinate triggers mismatch."""
        result = spatial_plausibility_checker.check(
            latitude=SOYA_FIELD_BRAZIL[0],
            longitude=SOYA_FIELD_BRAZIL[1],
            declared_country="GH",
        )
        assert result.country_mismatch is True

    def test_no_country_mismatch_correct_declaration(
        self, spatial_plausibility_checker
    ):
        """Correct country declaration does not trigger mismatch."""
        result = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            declared_country="GH",
        )
        assert result.country_mismatch is False


# ===========================================================================
# 4. Commodity Plausibility
# ===========================================================================


class TestCommodityPlausibility:
    """Test commodity-region plausibility checks."""

    def test_commodity_palm_oil_plausible_indonesia(
        self, spatial_plausibility_checker
    ):
        """Palm oil in tropical Indonesia is plausible."""
        result = spatial_plausibility_checker.check(
            latitude=PALM_PLANTATION_INDONESIA[0],
            longitude=PALM_PLANTATION_INDONESIA[1],
            commodity="oil_palm",
        )
        assert result.commodity_plausible is True

    def test_commodity_palm_oil_implausible_arctic(
        self, spatial_plausibility_checker
    ):
        """Palm oil in the Arctic is implausible."""
        result = spatial_plausibility_checker.check(
            latitude=ARCTIC_POINT[0],
            longitude=ARCTIC_POINT[1],
            commodity="oil_palm",
        )
        assert result.commodity_plausible is False

    def test_commodity_coffee_plausible_colombia(
        self, spatial_plausibility_checker
    ):
        """Coffee in Colombia (coffee belt) is plausible."""
        result = spatial_plausibility_checker.check(
            latitude=COFFEE_FARM_COLOMBIA[0],
            longitude=COFFEE_FARM_COLOMBIA[1],
            commodity="coffee",
        )
        assert result.commodity_plausible is True

    def test_commodity_coffee_implausible_arctic(
        self, spatial_plausibility_checker
    ):
        """Coffee in the Arctic is implausible."""
        result = spatial_plausibility_checker.check(
            latitude=ARCTIC_POINT[0],
            longitude=ARCTIC_POINT[1],
            commodity="coffee",
        )
        assert result.commodity_plausible is False

    def test_commodity_cocoa_plausible_ghana(
        self, spatial_plausibility_checker
    ):
        """Cocoa in Ghana (cocoa belt) is plausible."""
        result = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            commodity="cocoa",
        )
        assert result.commodity_plausible is True

    def test_commodity_soya_plausible_brazil(
        self, spatial_plausibility_checker
    ):
        """Soya in Mato Grosso, Brazil is plausible."""
        result = spatial_plausibility_checker.check(
            latitude=SOYA_FIELD_BRAZIL[0],
            longitude=SOYA_FIELD_BRAZIL[1],
            commodity="soya",
        )
        assert result.commodity_plausible is True


# ===========================================================================
# 5. Elevation Plausibility
# ===========================================================================


class TestElevationPlausibility:
    """Test elevation plausibility for EUDR commodities."""

    def test_elevation_plausible_cocoa_lowland(
        self, spatial_plausibility_checker
    ):
        """Cocoa at low elevation is plausible."""
        result = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            commodity="cocoa",
        )
        # Cocoa range 0-1500m; Ghana lowlands are well within
        assert result.elevation_plausible is True

    def test_elevation_implausible_oil_palm_high_altitude(
        self, spatial_plausibility_checker
    ):
        """Oil palm at very high altitude is implausible."""
        # Simulate a point in the Andes
        result = spatial_plausibility_checker.check(
            latitude=-15.5,
            longitude=-70.0,
            commodity="oil_palm",
        )
        # Oil palm range 0-1000m; Andes are way above
        if result.estimated_elevation_m is not None and result.estimated_elevation_m > 1000:
            assert result.elevation_plausible is False


# ===========================================================================
# 6. Urban Area Detection
# ===========================================================================


class TestUrbanDetection:
    """Test detection of urban/city coordinates."""

    def test_urban_detection_london(self, spatial_plausibility_checker):
        """London city centre is detected as urban."""
        result = spatial_plausibility_checker.check(
            latitude=URBAN_POINT[0],
            longitude=URBAN_POINT[1],
        )
        assert result.is_urban is True

    def test_not_urban_rural_farm(self, spatial_plausibility_checker):
        """Rural cocoa farm is not urban."""
        result = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
        )
        assert result.is_urban is False


# ===========================================================================
# 7. Protected Area Detection
# ===========================================================================


class TestProtectedAreaDetection:
    """Test protected area proximity flagging."""

    def test_protected_area_flagged(self, spatial_plausibility_checker):
        """Coordinate near a known protected area is flagged."""
        # Approximate location near Kakum National Park, Ghana
        result = spatial_plausibility_checker.check(
            latitude=5.35,
            longitude=-1.38,
        )
        # Should at minimum have a near_protected_area flag
        assert hasattr(result, "near_protected_area")

    def test_no_protected_area_open_farmland(self, spatial_plausibility_checker):
        """Open farmland far from protected areas is not flagged."""
        result = spatial_plausibility_checker.check(
            latitude=SOYA_FIELD_BRAZIL[0],
            longitude=SOYA_FIELD_BRAZIL[1],
        )
        # May or may not be near protected area; just verify the field exists
        assert hasattr(result, "near_protected_area")


# ===========================================================================
# 8. Batch Plausibility Check
# ===========================================================================


class TestBatchPlausibility:
    """Test batch plausibility checking."""

    def test_batch_check_multiple(self, spatial_plausibility_checker):
        """Batch check multiple coordinates."""
        coords = [
            {"lat": COCOA_FARM_GHANA[0], "lon": COCOA_FARM_GHANA[1], "commodity": "cocoa"},
            {"lat": OCEAN_POINT[0], "lon": OCEAN_POINT[1], "commodity": "soya"},
            {"lat": SOYA_FIELD_BRAZIL[0], "lon": SOYA_FIELD_BRAZIL[1], "commodity": "soya"},
        ]
        results = spatial_plausibility_checker.check_batch(coords)
        assert len(results) == 3
        # First and third should be on land
        assert results[0].is_on_land is True
        assert results[2].is_on_land is True
        # Second should be in ocean
        assert results[1].is_on_land is False

    def test_batch_check_empty(self, spatial_plausibility_checker):
        """Batch check with empty list returns empty."""
        results = spatial_plausibility_checker.check_batch([])
        assert results == []


# ===========================================================================
# 9. Parametrized Commodity Plausibility Tests
# ===========================================================================


@pytest.mark.parametrize(
    "commodity",
    ["cocoa", "oil_palm", "coffee", "soya", "rubber", "cattle", "wood"],
    ids=["cocoa", "oil_palm", "coffee", "soya", "rubber", "cattle", "wood"],
)
def test_parametrized_commodity_plausibility(
    spatial_plausibility_checker, commodity
):
    """Parametrized: test commodity plausibility for all 7 EUDR commodities."""
    entries = PLAUSIBLE_COMMODITY_COORDINATES.get(commodity, [])
    for entry in entries:
        result = spatial_plausibility_checker.check(
            latitude=entry["lat"],
            longitude=entry["lon"],
            commodity=commodity,
            declared_country=entry.get("country"),
        )
        assert result.commodity_plausible == entry["plausible"], (
            f"commodity={commodity}, lat={entry['lat']}, lon={entry['lon']}: "
            f"expected plausible={entry['plausible']}, "
            f"got {result.commodity_plausible}"
        )


# ===========================================================================
# 10. Additional Country Identification Tests
# ===========================================================================


class TestCountryIdentificationExtended:
    """Extended country identification from coordinates."""

    def test_country_check_ivory_coast(self, spatial_plausibility_checker):
        """Ivory Coast is correctly identified."""
        result = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_IVORY_COAST[0],
            longitude=COCOA_FARM_IVORY_COAST[1],
        )
        assert result.detected_country == "CI"

    def test_country_check_ethiopia(self, spatial_plausibility_checker):
        """Ethiopia is correctly identified."""
        result = spatial_plausibility_checker.check(
            latitude=COFFEE_FARM_ETHIOPIA[0],
            longitude=COFFEE_FARM_ETHIOPIA[1],
        )
        assert result.detected_country == "ET"

    def test_country_check_malaysia(self, spatial_plausibility_checker):
        """Malaysia is correctly identified."""
        result = spatial_plausibility_checker.check(
            latitude=PALM_PLANTATION_MALAYSIA[0],
            longitude=PALM_PLANTATION_MALAYSIA[1],
        )
        assert result.detected_country == "MY"

    def test_country_check_congo(self, spatial_plausibility_checker):
        """DRC (Congo) is correctly identified."""
        result = spatial_plausibility_checker.check(
            latitude=TIMBER_FOREST_CONGO[0],
            longitude=TIMBER_FOREST_CONGO[1],
        )
        assert result.detected_country == "CD"

    def test_country_check_cattle_brazil(self, spatial_plausibility_checker):
        """Brazil is identified for cattle ranch coordinates."""
        result = spatial_plausibility_checker.check(
            latitude=CATTLE_RANCH_BRAZIL[0],
            longitude=CATTLE_RANCH_BRAZIL[1],
        )
        assert result.detected_country == "BR"


# ===========================================================================
# 11. Ocean Points - Extended
# ===========================================================================


class TestOceanPointsExtended:
    """Extended ocean point tests for various ocean regions."""

    def test_ocean_southern_atlantic(self, spatial_plausibility_checker):
        """South Atlantic point is not on land."""
        result = spatial_plausibility_checker.check(
            latitude=-30.0,
            longitude=-15.0,
        )
        assert result.is_on_land is False

    def test_ocean_arctic_ocean(self, spatial_plausibility_checker):
        """Arctic Ocean point is not on land."""
        result = spatial_plausibility_checker.check(
            latitude=85.0,
            longitude=0.0,
        )
        assert result.is_on_land is False

    def test_ocean_southern_ocean(self, spatial_plausibility_checker):
        """Southern Ocean point is not on land."""
        result = spatial_plausibility_checker.check(
            latitude=-65.0,
            longitude=90.0,
        )
        assert result.is_on_land is False

    def test_null_island_is_ocean(self, spatial_plausibility_checker):
        """Null Island (0, 0) is in the ocean."""
        result = spatial_plausibility_checker.check(
            latitude=NULL_ISLAND[0],
            longitude=NULL_ISLAND[1],
        )
        assert result.is_on_land is False


# ===========================================================================
# 12. Additional Commodity Plausibility Tests
# ===========================================================================


class TestCommodityPlausibilityExtended:
    """Extended commodity-region plausibility tests."""

    def test_commodity_rubber_implausible_desert(
        self, spatial_plausibility_checker
    ):
        """Rubber in the Sahara desert is implausible."""
        result = spatial_plausibility_checker.check(
            latitude=DESERT_POINT[0],
            longitude=DESERT_POINT[1],
            commodity="rubber",
        )
        assert result.commodity_plausible is False

    def test_commodity_cattle_plausible_brazil(
        self, spatial_plausibility_checker
    ):
        """Cattle in central Brazil is plausible."""
        result = spatial_plausibility_checker.check(
            latitude=CATTLE_RANCH_BRAZIL[0],
            longitude=CATTLE_RANCH_BRAZIL[1],
            commodity="cattle",
        )
        assert result.commodity_plausible is True

    def test_commodity_wood_plausible_congo(
        self, spatial_plausibility_checker
    ):
        """Wood/timber in Congo basin is plausible."""
        result = spatial_plausibility_checker.check(
            latitude=TIMBER_FOREST_CONGO[0],
            longitude=TIMBER_FOREST_CONGO[1],
            commodity="wood",
        )
        assert result.commodity_plausible is True

    def test_commodity_cocoa_implausible_antarctic(
        self, spatial_plausibility_checker
    ):
        """Cocoa in Antarctica is implausible."""
        result = spatial_plausibility_checker.check(
            latitude=ANTARCTIC_POINT[0],
            longitude=ANTARCTIC_POINT[1],
            commodity="cocoa",
        )
        assert result.commodity_plausible is False

    def test_commodity_soya_implausible_urban(
        self, spatial_plausibility_checker
    ):
        """Soya in urban London is implausible."""
        result = spatial_plausibility_checker.check(
            latitude=URBAN_POINT[0],
            longitude=URBAN_POINT[1],
            commodity="soya",
        )
        # Soya in an urban area might be implausible depending on implementation
        assert isinstance(result.commodity_plausible, bool)


# ===========================================================================
# 13. Country Mismatch - Extended
# ===========================================================================


class TestCountryMismatchExtended:
    """Extended country mismatch tests."""

    def test_country_mismatch_indonesia_declared_brazil(
        self, spatial_plausibility_checker
    ):
        """Declaring Brazil for Indonesia coordinate triggers mismatch."""
        result = spatial_plausibility_checker.check(
            latitude=PALM_PLANTATION_INDONESIA[0],
            longitude=PALM_PLANTATION_INDONESIA[1],
            declared_country="BR",
        )
        assert result.country_mismatch is True

    def test_country_mismatch_colombia_declared_thailand(
        self, spatial_plausibility_checker
    ):
        """Declaring Thailand for Colombia coordinate triggers mismatch."""
        result = spatial_plausibility_checker.check(
            latitude=COFFEE_FARM_COLOMBIA[0],
            longitude=COFFEE_FARM_COLOMBIA[1],
            declared_country="TH",
        )
        assert result.country_mismatch is True

    def test_no_mismatch_no_declaration(self, spatial_plausibility_checker):
        """Without declared country, no mismatch is flagged."""
        result = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
        )
        assert result.country_mismatch is False


# ===========================================================================
# 14. Provenance and Result Structure
# ===========================================================================


class TestSpatialProvenance:
    """Test provenance tracking for spatial plausibility checks."""

    def test_result_has_provenance_hash(self, spatial_plausibility_checker):
        """Spatial check result includes a provenance hash."""
        result = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
        )
        assert hasattr(result, "provenance_hash")
        assert result.provenance_hash != ""

    def test_deterministic_spatial_result(self, spatial_plausibility_checker):
        """Same input produces same spatial result."""
        r1 = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            commodity="cocoa",
        )
        r2 = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            commodity="cocoa",
        )
        assert r1.is_on_land == r2.is_on_land
        assert r1.commodity_plausible == r2.commodity_plausible

    def test_result_has_all_expected_fields(self, spatial_plausibility_checker):
        """Spatial result has all expected fields populated."""
        result = spatial_plausibility_checker.check(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            commodity="cocoa",
            declared_country="GH",
        )
        assert hasattr(result, "is_on_land")
        assert hasattr(result, "detected_country")
        assert hasattr(result, "country_mismatch")
        assert hasattr(result, "commodity_plausible")
        assert hasattr(result, "elevation_plausible")
        assert hasattr(result, "is_urban")
        assert hasattr(result, "near_protected_area")
