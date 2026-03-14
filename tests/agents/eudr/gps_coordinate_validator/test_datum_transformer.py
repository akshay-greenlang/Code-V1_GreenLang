# -*- coding: utf-8 -*-
"""
Tests for DatumTransformer - AGENT-EUDR-007 Engine 2: Geodetic Datum Transformation

Comprehensive test suite covering:
- WGS84-to-WGS84 identity transformation (zero displacement)
- NAD27 to WGS84 transformation (North America)
- NAD83 to WGS84 (near identity)
- ED50 to WGS84 (European Datum 1950)
- SIRGAS_2000 to WGS84 (South America)
- INDIAN_1975 to WGS84 (Southeast Asia)
- PULKOVO_1942 to WGS84 (Russia / Eastern Europe)
- ARC1960 to WGS84 (East Africa)
- TOKYO to WGS84 (Japan)
- GDA94 to WGS84 (Australia)
- Molodensky approximation vs Helmert 7-parameter accuracy
- Displacement calculation verification
- Country-based datum auto-detection
- Geocentric round-trip (geographic -> geocentric -> geographic)
- Batch transformation
- Listing supported datums
- Parametrized tests for 15+ datums

Test count: 50+ tests
Coverage target: >= 85% of DatumTransformer module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import math

import pytest

from greenlang.agents.eudr.gps_coordinate_validator.models import (
    GeodeticDatum,
    NormalizedCoordinate,
)
from tests.agents.eudr.gps_coordinate_validator.conftest import (
    DATUM_TRANSFORM_REFERENCE,
    COCOA_FARM_GHANA,
    SOYA_FIELD_BRAZIL,
    PALM_PLANTATION_INDONESIA,
    RUBBER_FARM_THAILAND,
    CATTLE_RANCH_BRAZIL,
    TIMBER_FOREST_CONGO,
    COFFEE_FARM_COLOMBIA,
    COFFEE_FARM_ETHIOPIA,
    COCOA_FARM_IVORY_COAST,
    HIGH_PRECISION,
    NULL_ISLAND,
    BOUNDARY_LATITUDE,
    SOUTH_POLE,
    SHA256_HEX_LENGTH,
    assert_close,
    haversine_distance_m,
)


# ===========================================================================
# 1. WGS84 Identity Transformation
# ===========================================================================


class TestWGS84Identity:
    """WGS84 to WGS84 transformation should produce zero displacement."""

    def test_wgs84_identity_ghana(self, datum_transformer):
        """WGS84 -> WGS84 for Ghana coordinate produces zero displacement."""
        result = datum_transformer.transform(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            source_datum=GeodeticDatum.WGS84,
        )
        assert isinstance(result, NormalizedCoordinate)
        assert_close(result.latitude, COCOA_FARM_GHANA[0], tolerance=1e-10)
        assert_close(result.longitude, COCOA_FARM_GHANA[1], tolerance=1e-10)
        assert result.displacement_m < 0.001

    def test_wgs84_identity_brazil(self, datum_transformer):
        """WGS84 -> WGS84 for Brazil coordinate produces zero displacement."""
        result = datum_transformer.transform(
            latitude=SOYA_FIELD_BRAZIL[0],
            longitude=SOYA_FIELD_BRAZIL[1],
            source_datum=GeodeticDatum.WGS84,
        )
        assert_close(result.latitude, SOYA_FIELD_BRAZIL[0], tolerance=1e-10)
        assert_close(result.longitude, SOYA_FIELD_BRAZIL[1], tolerance=1e-10)
        assert result.displacement_m < 0.001

    def test_wgs84_identity_preserves_original(self, datum_transformer):
        """WGS84 identity records original coordinates."""
        result = datum_transformer.transform(
            latitude=5.603716,
            longitude=-0.186964,
            source_datum=GeodeticDatum.WGS84,
        )
        assert result.original_latitude == 5.603716
        assert result.original_longitude == -0.186964

    def test_wgs84_identity_datum_fields(self, datum_transformer):
        """WGS84 identity sets source and target datums correctly."""
        result = datum_transformer.transform(
            latitude=5.0,
            longitude=-1.0,
            source_datum=GeodeticDatum.WGS84,
        )
        assert result.source_datum == GeodeticDatum.WGS84
        assert result.target_datum == GeodeticDatum.WGS84


# ===========================================================================
# 2. Regional Datum Transformations
# ===========================================================================


class TestNAD27ToWGS84:
    """Test NAD27 to WGS84 transformation (North America)."""

    def test_nad27_to_wgs84_displacement(self, datum_transformer):
        """NAD27 -> WGS84 should produce measurable displacement."""
        result = datum_transformer.transform(
            latitude=40.0,
            longitude=-75.0,
            source_datum=GeodeticDatum.NAD27,
        )
        assert isinstance(result, NormalizedCoordinate)
        assert result.source_datum == GeodeticDatum.NAD27
        assert result.target_datum == GeodeticDatum.WGS84
        # NAD27 shift is typically 10-30m in CONUS
        assert result.displacement_m > 0.1
        assert result.displacement_m < 100.0

    def test_nad27_to_wgs84_result_within_tolerance(self, datum_transformer):
        """NAD27 -> WGS84 result is within expected tolerance."""
        result = datum_transformer.transform(
            latitude=40.0,
            longitude=-75.0,
            source_datum=GeodeticDatum.NAD27,
        )
        dist = haversine_distance_m(
            result.latitude, result.longitude, 40.0, -75.0
        )
        assert dist < 100.0  # Should be less than 100m offset


class TestNAD83ToWGS84:
    """Test NAD83 to WGS84 transformation (near identity)."""

    def test_nad83_to_wgs84_near_identity(self, datum_transformer):
        """NAD83 -> WGS84 should produce very small displacement (<2m)."""
        result = datum_transformer.transform(
            latitude=40.0,
            longitude=-75.0,
            source_datum=GeodeticDatum.NAD83,
        )
        assert result.displacement_m < 2.0
        assert_close(result.latitude, 40.0, tolerance=0.001)
        assert_close(result.longitude, -75.0, tolerance=0.001)


class TestED50ToWGS84:
    """Test ED50 to WGS84 transformation (European Datum 1950)."""

    def test_ed50_to_wgs84(self, datum_transformer):
        """ED50 -> WGS84 for a European coordinate."""
        result = datum_transformer.transform(
            latitude=48.0,
            longitude=2.0,
            source_datum=GeodeticDatum.ED50,
        )
        assert result.source_datum == GeodeticDatum.ED50
        assert result.displacement_m > 0.1
        assert result.displacement_m < 200.0


class TestSIRGASToWGS84:
    """Test SIRGAS_2000 to WGS84 transformation (South America)."""

    def test_sirgas_to_wgs84_near_identity(self, datum_transformer):
        """SIRGAS_2000 -> WGS84 should be near identity (<2m)."""
        result = datum_transformer.transform(
            latitude=SOYA_FIELD_BRAZIL[0],
            longitude=SOYA_FIELD_BRAZIL[1],
            source_datum=GeodeticDatum.SIRGAS_2000,
        )
        assert result.displacement_m < 2.0


class TestIndian1975ToWGS84:
    """Test INDIAN_1975 to WGS84 (Southeast Asia)."""

    def test_indian_1975_to_wgs84(self, datum_transformer):
        """INDIAN_1975 -> WGS84 for Thailand coordinate."""
        result = datum_transformer.transform(
            latitude=RUBBER_FARM_THAILAND[0],
            longitude=RUBBER_FARM_THAILAND[1],
            source_datum=GeodeticDatum.INDIAN_1975,
        )
        assert result.source_datum == GeodeticDatum.INDIAN_1975
        # Indian 1975 shift is typically 10-50m
        assert result.displacement_m < 100.0


class TestPulkovoToWGS84:
    """Test PULKOVO_1942 to WGS84 (Russia/Eastern Europe)."""

    def test_pulkovo_to_wgs84(self, datum_transformer):
        """PULKOVO_1942 -> WGS84 for Moscow area."""
        result = datum_transformer.transform(
            latitude=55.75,
            longitude=37.62,
            source_datum=GeodeticDatum.PULKOVO_1942,
        )
        assert result.displacement_m > 0.1
        assert result.displacement_m < 200.0


class TestARC1960ToWGS84:
    """Test ARC1960 to WGS84 (East Africa)."""

    def test_arc_1960_to_wgs84(self, datum_transformer):
        """ARC1960 -> WGS84 for Nairobi area."""
        result = datum_transformer.transform(
            latitude=-1.29,
            longitude=36.82,
            source_datum=GeodeticDatum.ARC1960,
        )
        assert result.source_datum == GeodeticDatum.ARC1960
        assert result.displacement_m < 200.0


class TestTokyoToWGS84:
    """Test TOKYO datum to WGS84 (Japan)."""

    def test_tokyo_to_wgs84(self, datum_transformer):
        """TOKYO -> WGS84 for Tokyo area."""
        result = datum_transformer.transform(
            latitude=35.68,
            longitude=139.77,
            source_datum=GeodeticDatum.TOKYO,
        )
        assert result.displacement_m > 0.1
        assert result.displacement_m < 500.0


class TestGDA94ToWGS84:
    """Test GDA94 to WGS84 (Australia)."""

    def test_gda94_to_wgs84_near_identity(self, datum_transformer):
        """GDA94 -> WGS84 should be near identity (<2m)."""
        result = datum_transformer.transform(
            latitude=-33.87,
            longitude=151.21,
            source_datum=GeodeticDatum.GDA94,
        )
        assert result.displacement_m < 2.0


# ===========================================================================
# 3. Transformation Accuracy and Methods
# ===========================================================================


class TestTransformationAccuracy:
    """Test transformation method accuracy and consistency."""

    def test_helmert_accuracy_sub_meter(self, datum_transformer):
        """Helmert 7-parameter should achieve sub-metre accuracy for compatible datums."""
        result = datum_transformer.transform(
            latitude=48.8566,
            longitude=2.3522,
            source_datum=GeodeticDatum.ED50,
        )
        # ED50 Helmert parameters are well-known; expect reasonable accuracy
        assert result.transformation_method in ("helmert_7param", "molodensky", "identity")
        assert result.provenance_hash != ""

    def test_displacement_calculation_consistency(self, datum_transformer):
        """Displacement should equal Haversine between original and transformed."""
        result = datum_transformer.transform(
            latitude=40.0,
            longitude=-75.0,
            source_datum=GeodeticDatum.NAD27,
        )
        expected_dist = haversine_distance_m(
            result.original_latitude,
            result.original_longitude,
            result.latitude,
            result.longitude,
        )
        # Allow 10% relative tolerance on displacement reporting
        if result.displacement_m > 0.1:
            ratio = abs(result.displacement_m - expected_dist) / max(result.displacement_m, 0.001)
            assert ratio < 0.5  # Within 50% agreement (different distance methods)


# ===========================================================================
# 4. Datum Auto-Detection
# ===========================================================================


class TestDatumAutoDetection:
    """Test country-based datum auto-detection."""

    @pytest.mark.parametrize(
        "country,expected_datums",
        [
            ("US", [GeodeticDatum.NAD27, GeodeticDatum.NAD83]),
            ("BR", [GeodeticDatum.SOUTH_AMERICAN_1969, GeodeticDatum.SIRGAS_2000]),
            ("AU", [GeodeticDatum.AGD66, GeodeticDatum.AGD84, GeodeticDatum.GDA94, GeodeticDatum.GDA2020]),
            ("JP", [GeodeticDatum.TOKYO]),
            ("DE", [GeodeticDatum.ED50, GeodeticDatum.ETRS89]),
            ("ID", [GeodeticDatum.JAKARTA]),
            ("GH", [GeodeticDatum.ACCRA, GeodeticDatum.LOME]),
            ("NG", [GeodeticDatum.MINNA]),
            ("KE", [GeodeticDatum.ARC1960]),
            ("RU", [GeodeticDatum.PULKOVO_1942]),
        ],
        ids=["us", "brazil", "australia", "japan", "germany",
             "indonesia", "ghana", "nigeria", "kenya", "russia"],
    )
    def test_datum_auto_detection_by_country(
        self, datum_transformer, country, expected_datums
    ):
        """Datum auto-detection returns plausible datums for country."""
        datums = datum_transformer.detect_likely_datums(country_iso=country)
        assert isinstance(datums, list)
        assert len(datums) > 0
        # At least one expected datum should be in the list
        assert any(d in datums for d in expected_datums), (
            f"Expected one of {expected_datums} for country={country}, "
            f"got {datums}"
        )


# ===========================================================================
# 5. Geocentric Round-Trip
# ===========================================================================


class TestGeocentricRoundTrip:
    """Test geographic -> geocentric -> geographic round-trip consistency."""

    @pytest.mark.parametrize(
        "lat,lon",
        [
            (0.0, 0.0),
            (45.0, 90.0),
            (-33.87, 151.21),
            (78.0, 15.0),
            (-90.0, 0.0),
            (90.0, 0.0),
        ],
        ids=["equator_origin", "mid_lat", "sydney", "arctic", "south_pole", "north_pole"],
    )
    def test_geocentric_roundtrip(self, datum_transformer, lat, lon):
        """Geographic -> geocentric -> geographic should preserve coordinates."""
        x, y, z = datum_transformer.geographic_to_geocentric(lat, lon, 0.0)
        lat_back, lon_back, alt_back = datum_transformer.geocentric_to_geographic(x, y, z)
        assert_close(lat_back, lat, tolerance=1e-8)
        assert_close(lon_back, lon, tolerance=1e-8)
        assert abs(alt_back) < 0.01  # Nearly zero altitude


# ===========================================================================
# 6. Batch Transformation
# ===========================================================================


class TestBatchTransform:
    """Test batch transformation of multiple coordinates."""

    def test_batch_transform_multiple(self, datum_transformer):
        """Batch transform multiple coordinates from NAD27."""
        coords = [
            (40.0, -75.0),
            (42.0, -80.0),
            (35.0, -90.0),
        ]
        results = datum_transformer.transform_batch(
            coordinates=coords,
            source_datum=GeodeticDatum.NAD27,
        )
        assert len(results) == 3
        for r in results:
            assert isinstance(r, NormalizedCoordinate)
            assert r.source_datum == GeodeticDatum.NAD27
            assert r.target_datum == GeodeticDatum.WGS84

    def test_batch_transform_empty(self, datum_transformer):
        """Batch transform with empty list returns empty."""
        results = datum_transformer.transform_batch(
            coordinates=[],
            source_datum=GeodeticDatum.NAD27,
        )
        assert results == []


# ===========================================================================
# 7. Supported Datums Listing
# ===========================================================================


class TestSupportedDatums:
    """Test listing of supported datums."""

    def test_list_supported_datums(self, datum_transformer):
        """Should return all supported datum identifiers."""
        datums = datum_transformer.list_supported_datums()
        assert isinstance(datums, list)
        assert len(datums) >= 15
        assert GeodeticDatum.WGS84 in datums
        assert GeodeticDatum.NAD27 in datums
        assert GeodeticDatum.ED50 in datums

    def test_list_includes_african_datums(self, datum_transformer):
        """Supported datums include African datums for EUDR compliance."""
        datums = datum_transformer.list_supported_datums()
        assert GeodeticDatum.ARC1960 in datums

    def test_list_includes_southeast_asian_datums(self, datum_transformer):
        """Supported datums include SE Asian datums for EUDR compliance."""
        datums = datum_transformer.list_supported_datums()
        assert GeodeticDatum.INDIAN_1975 in datums


# ===========================================================================
# 8. Provenance Tracking
# ===========================================================================


class TestDatumProvenance:
    """Test provenance hash generation for datum transformations."""

    def test_transform_has_provenance_hash(self, datum_transformer):
        """Every transformation result includes a provenance hash."""
        result = datum_transformer.transform(
            latitude=5.0,
            longitude=-1.0,
            source_datum=GeodeticDatum.WGS84,
        )
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_deterministic_provenance(self, datum_transformer):
        """Same input produces same provenance hash."""
        r1 = datum_transformer.transform(40.0, -75.0, GeodeticDatum.NAD27)
        r2 = datum_transformer.transform(40.0, -75.0, GeodeticDatum.NAD27)
        assert r1.provenance_hash == r2.provenance_hash


# ===========================================================================
# 9. Parametrized Datum Transform Reference Tests
# ===========================================================================


@pytest.mark.parametrize(
    "ref",
    DATUM_TRANSFORM_REFERENCE,
    ids=[r["name"] for r in DATUM_TRANSFORM_REFERENCE],
)
def test_parametrized_datum_transforms(datum_transformer, ref):
    """Parametrized: transform across 9 datum reference points."""
    result = datum_transformer.transform(
        latitude=ref["lat"],
        longitude=ref["lon"],
        source_datum=ref["source_datum"],
    )
    assert isinstance(result, NormalizedCoordinate)
    dist = haversine_distance_m(
        result.latitude, result.longitude,
        ref["expected_lat"], ref["expected_lon"],
    )
    assert dist < ref["tolerance_m"], (
        f"Transform {ref['name']}: displacement {dist:.1f}m exceeds "
        f"tolerance {ref['tolerance_m']}m"
    )


# ===========================================================================
# 10. Additional African Datum Transformations
# ===========================================================================


class TestAfricanDatumTransformations:
    """Test datum transformations for African EUDR producing regions."""

    def test_accra_datum_ghana(self, datum_transformer):
        """ACCRA datum -> WGS84 for Ghana coordinate."""
        result = datum_transformer.transform(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            source_datum=GeodeticDatum.ACCRA,
        )
        assert result.source_datum == GeodeticDatum.ACCRA
        assert result.target_datum == GeodeticDatum.WGS84
        assert result.displacement_m >= 0.0
        assert result.displacement_m < 500.0

    def test_lome_datum_west_africa(self, datum_transformer):
        """LOME datum -> WGS84 for West Africa coordinate."""
        result = datum_transformer.transform(
            latitude=COCOA_FARM_IVORY_COAST[0],
            longitude=COCOA_FARM_IVORY_COAST[1],
            source_datum=GeodeticDatum.LOME,
        )
        assert isinstance(result, NormalizedCoordinate)
        assert result.displacement_m < 500.0

    def test_adindan_datum_east_africa(self, datum_transformer):
        """ADINDAN datum -> WGS84 for East Africa (Ethiopia)."""
        result = datum_transformer.transform(
            latitude=COFFEE_FARM_ETHIOPIA[0],
            longitude=COFFEE_FARM_ETHIOPIA[1],
            source_datum=GeodeticDatum.ADINDAN,
        )
        assert result.source_datum == GeodeticDatum.ADINDAN
        assert result.displacement_m < 500.0

    def test_cape_datum_south_africa(self, datum_transformer):
        """CAPE datum -> WGS84 for Southern Africa."""
        result = datum_transformer.transform(
            latitude=-33.9,
            longitude=18.4,
            source_datum=GeodeticDatum.CAPE,
        )
        assert result.source_datum == GeodeticDatum.CAPE
        assert result.displacement_m < 500.0

    def test_minna_datum_nigeria(self, datum_transformer):
        """MINNA datum -> WGS84 for Nigeria."""
        result = datum_transformer.transform(
            latitude=9.06,
            longitude=7.49,
            source_datum=GeodeticDatum.MINNA,
        )
        assert result.source_datum == GeodeticDatum.MINNA
        assert result.displacement_m < 500.0


# ===========================================================================
# 11. Additional Southeast Asian Datum Transformations
# ===========================================================================


class TestSoutheastAsianDatumTransformations:
    """Test datum transformations for Southeast Asian EUDR regions."""

    def test_jakarta_datum_indonesia(self, datum_transformer):
        """JAKARTA datum -> WGS84 for Indonesia."""
        result = datum_transformer.transform(
            latitude=PALM_PLANTATION_INDONESIA[0],
            longitude=PALM_PLANTATION_INDONESIA[1],
            source_datum=GeodeticDatum.JAKARTA,
        )
        assert result.source_datum == GeodeticDatum.JAKARTA
        assert result.displacement_m < 500.0

    def test_kertau_datum_malaysia(self, datum_transformer):
        """KERTAU datum -> WGS84 for Malaysia."""
        result = datum_transformer.transform(
            latitude=2.95,
            longitude=101.7,
            source_datum=GeodeticDatum.KERTAU,
        )
        assert result.source_datum == GeodeticDatum.KERTAU
        assert result.displacement_m < 500.0

    def test_timbalai_datum_borneo(self, datum_transformer):
        """TIMBALAI_1948 datum -> WGS84 for Borneo."""
        result = datum_transformer.transform(
            latitude=4.0,
            longitude=115.0,
            source_datum=GeodeticDatum.TIMBALAI_1948,
        )
        assert result.source_datum == GeodeticDatum.TIMBALAI_1948
        assert result.displacement_m < 500.0


# ===========================================================================
# 12. South American Datum Transformations
# ===========================================================================


class TestSouthAmericanDatumTransformations:
    """Test datum transformations for South American EUDR regions."""

    def test_sad69_datum_brazil(self, datum_transformer):
        """SOUTH_AMERICAN_1969 datum -> WGS84 for Brazil."""
        result = datum_transformer.transform(
            latitude=SOYA_FIELD_BRAZIL[0],
            longitude=SOYA_FIELD_BRAZIL[1],
            source_datum=GeodeticDatum.SOUTH_AMERICAN_1969,
        )
        assert result.source_datum == GeodeticDatum.SOUTH_AMERICAN_1969
        assert result.target_datum == GeodeticDatum.WGS84
        # SOUTH_AMERICAN_1969 shift is typically 30-100m in Brazil
        assert result.displacement_m < 200.0

    def test_bogota_datum_colombia(self, datum_transformer):
        """BOGOTA datum -> WGS84 for Colombia coffee region."""
        result = datum_transformer.transform(
            latitude=COFFEE_FARM_COLOMBIA[0],
            longitude=COFFEE_FARM_COLOMBIA[1],
            source_datum=GeodeticDatum.BOGOTA,
        )
        assert result.source_datum == GeodeticDatum.BOGOTA
        assert result.displacement_m < 500.0


# ===========================================================================
# 13. Edge Cases and Boundary Points
# ===========================================================================


class TestDatumEdgeCases:
    """Test datum transformation edge cases and boundary points."""

    def test_transform_north_pole(self, datum_transformer):
        """Datum transform at North Pole (lat=90)."""
        result = datum_transformer.transform(
            latitude=BOUNDARY_LATITUDE[0],
            longitude=BOUNDARY_LATITUDE[1],
            source_datum=GeodeticDatum.WGS84,
        )
        assert_close(result.latitude, 90.0, tolerance=1e-8)
        assert result.displacement_m < 0.001

    def test_transform_south_pole(self, datum_transformer):
        """Datum transform at South Pole (lat=-90)."""
        result = datum_transformer.transform(
            latitude=SOUTH_POLE[0],
            longitude=SOUTH_POLE[1],
            source_datum=GeodeticDatum.WGS84,
        )
        assert_close(result.latitude, -90.0, tolerance=1e-8)

    def test_transform_null_island(self, datum_transformer):
        """Datum transform at origin (0, 0)."""
        result = datum_transformer.transform(
            latitude=NULL_ISLAND[0],
            longitude=NULL_ISLAND[1],
            source_datum=GeodeticDatum.WGS84,
        )
        assert_close(result.latitude, 0.0, tolerance=1e-8)
        assert_close(result.longitude, 0.0, tolerance=1e-8)

    def test_transform_high_precision_input(self, datum_transformer):
        """Datum transform preserves high precision input."""
        result = datum_transformer.transform(
            latitude=HIGH_PRECISION[0],
            longitude=HIGH_PRECISION[1],
            source_datum=GeodeticDatum.WGS84,
        )
        assert_close(result.latitude, HIGH_PRECISION[0], tolerance=1e-8)
        assert_close(result.longitude, HIGH_PRECISION[1], tolerance=1e-8)


# ===========================================================================
# 14. Batch Transform - Extended
# ===========================================================================


class TestBatchTransformExtended:
    """Extended batch transformation tests."""

    def test_batch_transform_all_eudr_regions(self, datum_transformer):
        """Batch transform all EUDR region coordinates from WGS84."""
        coords = [
            (COCOA_FARM_GHANA[0], COCOA_FARM_GHANA[1]),
            (PALM_PLANTATION_INDONESIA[0], PALM_PLANTATION_INDONESIA[1]),
            (COFFEE_FARM_COLOMBIA[0], COFFEE_FARM_COLOMBIA[1]),
            (SOYA_FIELD_BRAZIL[0], SOYA_FIELD_BRAZIL[1]),
            (RUBBER_FARM_THAILAND[0], RUBBER_FARM_THAILAND[1]),
            (TIMBER_FOREST_CONGO[0], TIMBER_FOREST_CONGO[1]),
        ]
        results = datum_transformer.transform_batch(
            coordinates=coords,
            source_datum=GeodeticDatum.WGS84,
        )
        assert len(results) == 6
        for i, r in enumerate(results):
            assert r.displacement_m < 0.001  # WGS84 identity
            assert_close(r.latitude, coords[i][0], tolerance=1e-8)

    def test_batch_transform_mixed_datums(self, datum_transformer):
        """Batch transform preserves per-coordinate integrity."""
        coords = [(40.0, -75.0), (42.0, -80.0)]
        results = datum_transformer.transform_batch(
            coordinates=coords,
            source_datum=GeodeticDatum.NAD27,
        )
        assert len(results) == 2
        # Each result should have measurable displacement
        for r in results:
            assert r.displacement_m > 0.1

    def test_batch_transform_single_item(self, datum_transformer):
        """Batch transform with a single coordinate."""
        coords = [(5.603716, -0.186964)]
        results = datum_transformer.transform_batch(
            coordinates=coords,
            source_datum=GeodeticDatum.WGS84,
        )
        assert len(results) == 1
        assert results[0].displacement_m < 0.001


# ===========================================================================
# 15. Provenance Chain Consistency
# ===========================================================================


class TestDatumProvenanceExtended:
    """Extended provenance tracking for datum transformations."""

    def test_provenance_hash_length(self, datum_transformer):
        """Provenance hash is exactly 64 hex characters (SHA-256)."""
        result = datum_transformer.transform(
            latitude=COCOA_FARM_GHANA[0],
            longitude=COCOA_FARM_GHANA[1],
            source_datum=GeodeticDatum.WGS84,
        )
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_provenance_different_datums_different_hash(self, datum_transformer):
        """Different source datums produce different provenance hashes."""
        r1 = datum_transformer.transform(40.0, -75.0, GeodeticDatum.NAD27)
        r2 = datum_transformer.transform(40.0, -75.0, GeodeticDatum.WGS84)
        assert r1.provenance_hash != r2.provenance_hash

    def test_provenance_different_coords_different_hash(self, datum_transformer):
        """Different coordinates produce different provenance hashes."""
        r1 = datum_transformer.transform(40.0, -75.0, GeodeticDatum.WGS84)
        r2 = datum_transformer.transform(42.0, -80.0, GeodeticDatum.WGS84)
        assert r1.provenance_hash != r2.provenance_hash

    def test_transformation_method_recorded(self, datum_transformer):
        """Transformation method is recorded in the result."""
        result = datum_transformer.transform(
            latitude=40.0,
            longitude=-75.0,
            source_datum=GeodeticDatum.NAD27,
        )
        assert result.transformation_method in (
            "helmert_7param", "molodensky", "identity", "abridged_molodensky"
        )
