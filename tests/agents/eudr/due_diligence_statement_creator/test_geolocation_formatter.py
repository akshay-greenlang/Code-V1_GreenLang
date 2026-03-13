# -*- coding: utf-8 -*-
"""
Unit tests for GeolocationFormatter - AGENT-EUDR-037

Tests geolocation formatting, batch processing, validation, GeoJSON generation.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.due_diligence_statement_creator.config import DDSCreatorConfig
from greenlang.agents.eudr.due_diligence_statement_creator.geolocation_formatter import GeolocationFormatter
from greenlang.agents.eudr.due_diligence_statement_creator.models import GeolocationData, GeolocationMethod


@pytest.fixture
def config():
    return DDSCreatorConfig()


@pytest.fixture
def formatter(config):
    return GeolocationFormatter(config=config)


class TestFormatGeolocation:
    @pytest.mark.asyncio
    async def test_returns_geolocation_data(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=5.123456, longitude=-3.456789)
        assert isinstance(geo, GeolocationData)

    @pytest.mark.asyncio
    async def test_plot_id_preserved(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-XYZ", latitude=0.0, longitude=0.0)
        assert geo.plot_id == "PLT-XYZ"

    @pytest.mark.asyncio
    async def test_latitude_rounded(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=5.1234567890, longitude=0.0)
        lat_str = str(geo.latitude)
        if "." in lat_str:
            decimals = len(lat_str.split(".")[1])
            assert decimals <= 6

    @pytest.mark.asyncio
    async def test_longitude_rounded(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=0.0, longitude=-3.4567890123)
        lon_str = str(geo.longitude)
        if "." in lon_str:
            decimals = len(lon_str.split(".")[1])
            assert decimals <= 6

    @pytest.mark.asyncio
    async def test_area_hectares_set(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=0.0, longitude=0.0, area_hectares=2.5)
        assert geo.area_hectares == Decimal("2.50")

    @pytest.mark.asyncio
    async def test_country_code_set(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=0.0, longitude=0.0, country_code="CI")
        assert geo.country_code == "CI"

    @pytest.mark.asyncio
    async def test_default_method(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=0.0, longitude=0.0)
        assert geo.collection_method == GeolocationMethod.GPS_FIELD_SURVEY

    @pytest.mark.asyncio
    async def test_custom_method(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=0.0, longitude=0.0,
            method="satellite_derived")
        assert geo.collection_method == GeolocationMethod.SATELLITE_DERIVED

    @pytest.mark.asyncio
    async def test_invalid_method_defaults(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=0.0, longitude=0.0,
            method="invalid_method")
        assert geo.collection_method == GeolocationMethod.GPS_FIELD_SURVEY

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=5.0, longitude=-3.0)
        assert len(geo.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_polygon_coordinates_formatted(self, formatter):
        poly = [[5.19, -3.49], [5.21, -3.49], [5.21, -3.51], [5.19, -3.51]]
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=5.2, longitude=-3.5,
            polygon_coordinates=poly)
        assert len(geo.polygon_coordinates) >= 4

    @pytest.mark.asyncio
    async def test_out_of_bounds_latitude_raises(self, formatter):
        with pytest.raises(ValueError, match="Latitude"):
            await formatter.format_geolocation(
                plot_id="PLT-001", latitude=91.0, longitude=0.0)

    @pytest.mark.asyncio
    async def test_out_of_bounds_longitude_raises(self, formatter):
        with pytest.raises(ValueError, match="Longitude"):
            await formatter.format_geolocation(
                plot_id="PLT-001", latitude=0.0, longitude=181.0)

    @pytest.mark.asyncio
    async def test_negative_out_of_bounds_latitude_raises(self, formatter):
        with pytest.raises(ValueError, match="Latitude"):
            await formatter.format_geolocation(
                plot_id="PLT-001", latitude=-91.0, longitude=0.0)

    @pytest.mark.asyncio
    async def test_formatted_count_increments(self, formatter):
        await formatter.format_geolocation(plot_id="PLT-001", latitude=0.0, longitude=0.0)
        await formatter.format_geolocation(plot_id="PLT-002", latitude=1.0, longitude=1.0)
        health = await formatter.health_check()
        assert health["plots_formatted"] == 2


class TestFormatBatch:
    @pytest.mark.asyncio
    async def test_batch_returns_list(self, formatter):
        plots = [
            {"plot_id": "P1", "latitude": 5.0, "longitude": -3.0},
            {"plot_id": "P2", "latitude": 6.0, "longitude": -4.0},
        ]
        results = await formatter.format_batch(plots)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_batch_empty_list(self, formatter):
        results = await formatter.format_batch([])
        assert results == []


class TestValidateGeolocation:
    @pytest.mark.asyncio
    async def test_validate_valid_geo(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=5.123456, longitude=-3.456789,
            country_code="CI")
        result = await formatter.validate_geolocation(geo)
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_missing_country(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=5.0, longitude=-3.0)
        result = await formatter.validate_geolocation(geo)
        assert result["valid"] is False
        assert any("country" in i.lower() for i in result["issues"])

    @pytest.mark.asyncio
    async def test_validate_large_plot_without_polygon(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=5.0, longitude=-3.0,
            area_hectares=10.0, country_code="CI")
        result = await formatter.validate_geolocation(geo)
        assert any("polygon" in i.lower() for i in result["issues"])


class TestGenerateGeoJSON:
    @pytest.mark.asyncio
    async def test_geojson_returns_feature_collection(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=5.0, longitude=-3.0)
        geojson = await formatter.generate_geojson([geo])
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 1

    @pytest.mark.asyncio
    async def test_geojson_point_geometry(self, formatter):
        geo = await formatter.format_geolocation(
            plot_id="PLT-001", latitude=5.0, longitude=-3.0, area_hectares=1.0)
        geojson = await formatter.generate_geojson([geo])
        feature = geojson["features"][0]
        assert feature["geometry"]["type"] == "Point"

    @pytest.mark.asyncio
    async def test_geojson_empty_list(self, formatter):
        geojson = await formatter.generate_geojson([])
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 0


class TestGeolocationHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, formatter):
        health = await formatter.health_check()
        assert health["engine"] == "GeolocationFormatter"
        assert health["status"] == "healthy"
        assert health["precision_digits"] == 6
