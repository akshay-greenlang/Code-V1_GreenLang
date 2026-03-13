# -*- coding: utf-8 -*-
"""
Unit tests for GeolocationFormatter engine - AGENT-EUDR-036

Tests coordinate formatting, polygon handling, centroid calculation,
area threshold detection, and multipolygon formatting.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.eu_information_system_interface.config import (
    EUInformationSystemInterfaceConfig,
)
from greenlang.agents.eudr.eu_information_system_interface.geolocation_formatter import (
    GeolocationFormatter,
)
from greenlang.agents.eudr.eu_information_system_interface.models import (
    GeolocationFormat,
)
from greenlang.agents.eudr.eu_information_system_interface.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def formatter() -> GeolocationFormatter:
    """Create a GeolocationFormatter instance."""
    config = EUInformationSystemInterfaceConfig()
    return GeolocationFormatter(config=config, provenance=ProvenanceTracker())


class TestFormatGeolocation:
    """Test GeolocationFormatter.format_geolocation()."""

    @pytest.mark.asyncio
    async def test_single_point(self, formatter, single_coordinate):
        result = await formatter.format_geolocation(
            coordinates=single_coordinate,
            country_code="CO",
        )
        assert result.format == GeolocationFormat.POINT
        assert result.point is not None
        assert result.polygon is None
        assert result.formatted_for_eu is True
        assert result.country_code == "CO"

    @pytest.mark.asyncio
    async def test_polygon_format(self, formatter, sample_coordinates):
        result = await formatter.format_geolocation(
            coordinates=sample_coordinates,
            country_code="GH",
            area_hectares=Decimal("10.0"),
        )
        assert result.format == GeolocationFormat.POLYGON
        assert result.polygon is not None
        assert result.point is None

    @pytest.mark.asyncio
    async def test_small_area_uses_point(self, formatter, sample_coordinates):
        result = await formatter.format_geolocation(
            coordinates=sample_coordinates,
            country_code="GH",
            area_hectares=Decimal("2.0"),
        )
        assert result.format == GeolocationFormat.POINT
        assert result.point is not None

    @pytest.mark.asyncio
    async def test_country_code_uppercased(self, formatter, single_coordinate):
        result = await formatter.format_geolocation(
            coordinates=single_coordinate,
            country_code="co",
        )
        assert result.country_code == "CO"

    @pytest.mark.asyncio
    async def test_region_included(self, formatter, single_coordinate):
        result = await formatter.format_geolocation(
            coordinates=single_coordinate,
            country_code="CO",
            region="Antioquia",
        )
        assert result.region == "Antioquia"

    @pytest.mark.asyncio
    async def test_empty_coordinates_raises(self, formatter):
        with pytest.raises(ValueError, match="At least one coordinate"):
            await formatter.format_geolocation(
                coordinates=[],
                country_code="GH",
            )

    @pytest.mark.asyncio
    async def test_two_coordinates_uses_centroid(self, formatter):
        coords = [
            {"lat": 6.688, "lng": -1.624},
            {"lat": 6.691, "lng": -1.620},
        ]
        result = await formatter.format_geolocation(
            coordinates=coords,
            country_code="GH",
        )
        assert result.format == GeolocationFormat.POINT
        assert result.point is not None

    @pytest.mark.asyncio
    async def test_coordinate_precision(self, formatter):
        coords = [{"lat": 6.123456789, "lng": -1.987654321}]
        result = await formatter.format_geolocation(
            coordinates=coords,
            country_code="GH",
        )
        # Default precision is 6 decimal places
        lat_str = str(result.point.latitude)
        # Check precision is applied (max 6 decimal digits after the dot)
        parts = lat_str.split(".")
        if len(parts) == 2:
            assert len(parts[1]) <= 6


class TestFormatMultipolygon:
    """Test GeolocationFormatter.format_multipolygon()."""

    @pytest.mark.asyncio
    async def test_multipolygon(self, formatter):
        groups = [
            [
                {"lat": 6.0, "lng": -1.0},
                {"lat": 6.1, "lng": -1.1},
                {"lat": 6.2, "lng": -1.0},
            ],
            [
                {"lat": 7.0, "lng": -2.0},
                {"lat": 7.1, "lng": -2.1},
                {"lat": 7.2, "lng": -2.0},
            ],
        ]
        result = await formatter.format_multipolygon(
            polygon_groups=groups,
            country_code="GH",
        )
        assert result.format == GeolocationFormat.MULTIPOLYGON
        assert len(result.polygons) == 2

    @pytest.mark.asyncio
    async def test_multipolygon_empty_raises(self, formatter):
        with pytest.raises(ValueError, match="At least one polygon group"):
            await formatter.format_multipolygon(
                polygon_groups=[],
                country_code="GH",
            )

    @pytest.mark.asyncio
    async def test_multipolygon_skips_small_groups(self, formatter):
        groups = [
            [{"lat": 6.0, "lng": -1.0}, {"lat": 6.1, "lng": -1.1}],  # < 3 coords
            [
                {"lat": 7.0, "lng": -2.0},
                {"lat": 7.1, "lng": -2.1},
                {"lat": 7.2, "lng": -2.0},
            ],
        ]
        result = await formatter.format_multipolygon(
            polygon_groups=groups,
            country_code="GH",
        )
        assert len(result.polygons) == 1


class TestValidateCoordinates:
    """Test GeolocationFormatter.validate_coordinates()."""

    @pytest.mark.asyncio
    async def test_valid_coordinates(self, formatter, sample_coordinates):
        result = await formatter.validate_coordinates(sample_coordinates)
        assert result["valid"] is True
        assert result["coordinate_count"] == 4

    @pytest.mark.asyncio
    async def test_invalid_latitude(self, formatter):
        coords = [{"lat": 100.0, "lng": -1.0}]
        result = await formatter.validate_coordinates(coords)
        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_invalid_longitude(self, formatter):
        coords = [{"lat": 6.0, "lng": -200.0}]
        result = await formatter.validate_coordinates(coords)
        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_missing_lat_lng(self, formatter):
        coords = [{"x": 6.0, "y": -1.0}]
        result = await formatter.validate_coordinates(coords)
        assert result["valid"] is False


class TestHealthCheck:
    """Test GeolocationFormatter.health_check()."""

    @pytest.mark.asyncio
    async def test_health_check(self, formatter):
        health = await formatter.health_check()
        assert health["engine"] == "GeolocationFormatter"
        assert health["status"] == "available"
        assert "precision" in health["config"]
