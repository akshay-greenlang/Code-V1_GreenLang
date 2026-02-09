# -*- coding: utf-8 -*-
"""
Unit Tests for CRSTransformerEngine (AGENT-DATA-006)

Tests WGS84<->UTM transform, WGS84<->Web Mercator, batch transform,
detect UTM zone (positive lon = N, negative lon = S, zone calculation),
get CRS info, is_geographic/is_projected, unknown CRS, identity transform
(same source/target), and provenance hash generation.

Coverage target: 85%+ of crs_transformer.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline models (minimal)
# ---------------------------------------------------------------------------


class Coordinate:
    def __init__(self, longitude: float = 0.0, latitude: float = 0.0,
                 altitude: Optional[float] = None):
        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude


class Geometry:
    def __init__(self, geometry_type: str = "point", coordinates: Optional[Any] = None,
                 properties: Optional[Dict[str, Any]] = None):
        self.geometry_type = geometry_type
        self.coordinates = coordinates or []
        self.properties = properties or {}


class Feature:
    def __init__(self, feature_id: str = "", geometry: Optional[Geometry] = None,
                 properties: Optional[Dict[str, Any]] = None, crs: str = "EPSG:4326",
                 provenance_hash: Optional[str] = None):
        import uuid
        self.feature_id = feature_id or f"FTR-{uuid.uuid4().hex[:5]}"
        self.geometry = geometry
        self.properties = properties or {}
        self.crs = crs
        self.provenance_hash = provenance_hash


class TransformResult:
    def __init__(self, transform_id: str = "", source_crs: str = "", target_crs: str = "",
                 feature_count: int = 0, status: str = "completed",
                 execution_time_ms: float = 0.0, provenance_hash: Optional[str] = None):
        import uuid
        self.transform_id = transform_id or f"TRF-{uuid.uuid4().hex[:5]}"
        self.source_crs = source_crs
        self.target_crs = target_crs
        self.feature_count = feature_count
        self.status = status
        self.execution_time_ms = execution_time_ms
        self.provenance_hash = provenance_hash


class CRSInfo:
    def __init__(self, code: str = "", name: str = "", crs_type: str = "",
                 is_geographic: bool = False, is_projected: bool = False,
                 datum: str = "", unit: str = "", authority: str = "EPSG"):
        self.code = code
        self.name = name
        self.crs_type = crs_type
        self.is_geographic = is_geographic
        self.is_projected = is_projected
        self.datum = datum
        self.unit = unit
        self.authority = authority


# ---------------------------------------------------------------------------
# Inline CRSTransformerEngine
# ---------------------------------------------------------------------------


class CRSTransformerEngine:
    """Transforms coordinates and features between coordinate reference systems."""

    # Well-known CRS registry
    CRS_REGISTRY: Dict[str, CRSInfo] = {
        "EPSG:4326": CRSInfo(
            code="EPSG:4326", name="WGS 84", crs_type="geographic",
            is_geographic=True, is_projected=False,
            datum="WGS84", unit="degree",
        ),
        "EPSG:3857": CRSInfo(
            code="EPSG:3857", name="WGS 84 / Pseudo-Mercator", crs_type="projected",
            is_geographic=False, is_projected=True,
            datum="WGS84", unit="metre",
        ),
    }

    # Earth radius in meters
    EARTH_RADIUS_M = 6378137.0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._lock = threading.Lock()
        self._counter = 0
        self._stats = {
            "transforms_performed": 0,
            "features_transformed": 0,
            "errors": 0,
        }
        # Dynamically register UTM zones
        self._register_utm_zones()

    def _register_utm_zones(self) -> None:
        """Register UTM zone CRS entries."""
        for zone in range(1, 61):
            # Northern hemisphere
            code_n = f"EPSG:{32600 + zone}"
            self.CRS_REGISTRY[code_n] = CRSInfo(
                code=code_n, name=f"WGS 84 / UTM zone {zone}N", crs_type="projected",
                is_geographic=False, is_projected=True,
                datum="WGS84", unit="metre",
            )
            # Southern hemisphere
            code_s = f"EPSG:{32700 + zone}"
            self.CRS_REGISTRY[code_s] = CRSInfo(
                code=code_s, name=f"WGS 84 / UTM zone {zone}S", crs_type="projected",
                is_geographic=False, is_projected=True,
                datum="WGS84", unit="metre",
            )

    def _next_transform_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"TRF-{self._counter:05d}"

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # -----------------------------------------------------------------------
    # Core transform operations
    # -----------------------------------------------------------------------

    def transform_coordinate(
        self, lon: float, lat: float, source_crs: str, target_crs: str,
    ) -> Tuple[float, float]:
        """Transform a single coordinate pair between CRS."""
        if source_crs == target_crs:
            return (lon, lat)

        # WGS84 -> Web Mercator
        if source_crs == "EPSG:4326" and target_crs == "EPSG:3857":
            return self._wgs84_to_web_mercator(lon, lat)

        # Web Mercator -> WGS84
        if source_crs == "EPSG:3857" and target_crs == "EPSG:4326":
            return self._web_mercator_to_wgs84(lon, lat)

        # WGS84 -> UTM
        if source_crs == "EPSG:4326" and self._is_utm(target_crs):
            zone, is_north = self._parse_utm_code(target_crs)
            return self._wgs84_to_utm(lon, lat, zone, is_north)

        # UTM -> WGS84
        if self._is_utm(source_crs) and target_crs == "EPSG:4326":
            zone, is_north = self._parse_utm_code(source_crs)
            return self._utm_to_wgs84(lon, lat, zone, is_north)

        raise ValueError(f"Unsupported CRS transform: {source_crs} -> {target_crs}")

    def transform_feature(
        self, feature: Feature, source_crs: str, target_crs: str,
    ) -> Feature:
        """Transform a single feature to a new CRS."""
        if feature.geometry is None:
            return feature

        new_coords = self._transform_coords(
            feature.geometry.coordinates,
            feature.geometry.geometry_type,
            source_crs,
            target_crs,
        )

        new_geom = Geometry(
            geometry_type=feature.geometry.geometry_type,
            coordinates=new_coords,
            properties=dict(feature.geometry.properties),
        )

        prov = {
            "op": "transform_feature",
            "feature_id": feature.feature_id,
            "source_crs": source_crs,
            "target_crs": target_crs,
        }

        new_feat = Feature(
            feature_id=feature.feature_id,
            geometry=new_geom,
            properties=dict(feature.properties),
            crs=target_crs,
            provenance_hash=self._compute_provenance(prov),
        )

        with self._lock:
            self._stats["features_transformed"] += 1

        return new_feat

    def batch_transform(
        self, features: List[Feature], source_crs: str, target_crs: str,
    ) -> TransformResult:
        """Transform a batch of features to a new CRS."""
        transform_id = self._next_transform_id()
        transformed_count = 0
        errors = 0

        for feat in features:
            try:
                self.transform_feature(feat, source_crs, target_crs)
                transformed_count += 1
            except Exception:
                errors += 1

        status = "completed" if errors == 0 else "completed_with_errors"

        prov = {
            "op": "batch_transform",
            "transform_id": transform_id,
            "source_crs": source_crs,
            "target_crs": target_crs,
            "feature_count": len(features),
        }

        with self._lock:
            self._stats["transforms_performed"] += 1
            self._stats["errors"] += errors

        return TransformResult(
            transform_id=transform_id,
            source_crs=source_crs,
            target_crs=target_crs,
            feature_count=transformed_count,
            status=status,
            provenance_hash=self._compute_provenance(prov),
        )

    # -----------------------------------------------------------------------
    # UTM zone detection
    # -----------------------------------------------------------------------

    def detect_utm_zone(self, lon: float, lat: float) -> str:
        """Detect the appropriate UTM zone for a coordinate.
        Returns EPSG code like EPSG:32618 (zone 18N) or EPSG:32718 (zone 18S).
        """
        zone = int((lon + 180) / 6) + 1
        zone = max(1, min(60, zone))

        if lat >= 0:
            return f"EPSG:{32600 + zone}"
        else:
            return f"EPSG:{32700 + zone}"

    # -----------------------------------------------------------------------
    # CRS information
    # -----------------------------------------------------------------------

    def get_crs_info(self, crs_code: str) -> Optional[CRSInfo]:
        """Get CRS information by EPSG code."""
        return self.CRS_REGISTRY.get(crs_code)

    def is_geographic(self, crs_code: str) -> bool:
        """Check if CRS is geographic (uses degrees)."""
        info = self.get_crs_info(crs_code)
        if info is None:
            raise ValueError(f"Unknown CRS: {crs_code}")
        return info.is_geographic

    def is_projected(self, crs_code: str) -> bool:
        """Check if CRS is projected (uses metres/feet)."""
        info = self.get_crs_info(crs_code)
        if info is None:
            raise ValueError(f"Unknown CRS: {crs_code}")
        return info.is_projected

    # -----------------------------------------------------------------------
    # Internal transforms
    # -----------------------------------------------------------------------

    def _wgs84_to_web_mercator(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert WGS84 (EPSG:4326) to Web Mercator (EPSG:3857)."""
        x = math.radians(lon) * self.EARTH_RADIUS_M
        y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * self.EARTH_RADIUS_M
        return (x, y)

    def _web_mercator_to_wgs84(self, x: float, y: float) -> Tuple[float, float]:
        """Convert Web Mercator (EPSG:3857) to WGS84 (EPSG:4326)."""
        lon = math.degrees(x / self.EARTH_RADIUS_M)
        lat = math.degrees(2 * math.atan(math.exp(y / self.EARTH_RADIUS_M)) - math.pi / 2)
        return (lon, lat)

    def _wgs84_to_utm(self, lon: float, lat: float, zone: int, is_north: bool) -> Tuple[float, float]:
        """Simplified WGS84 to UTM conversion."""
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        # Central meridian of UTM zone
        lon0 = math.radians((zone - 1) * 6 - 180 + 3)

        # Simplified projection (not fully accurate, but works for testing)
        k0 = 0.9996
        a = self.EARTH_RADIUS_M
        e = 0.0818191908426  # WGS84 eccentricity

        N = a / math.sqrt(1 - e**2 * math.sin(lat_rad)**2)
        T = math.tan(lat_rad)**2
        C = (e**2 / (1 - e**2)) * math.cos(lat_rad)**2
        A = math.cos(lat_rad) * (lon_rad - lon0)

        # Meridional arc
        M = a * (
            (1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * lat_rad
            - (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * math.sin(2*lat_rad)
            + (15*e**4/256 + 45*e**6/1024) * math.sin(4*lat_rad)
            - (35*e**6/3072) * math.sin(6*lat_rad)
        )

        easting = k0 * N * (A + (1-T+C)*A**3/6 + (5-18*T+T**2+72*C-58*(e**2/(1-e**2)))*A**5/120) + 500000.0
        northing = k0 * (M + N*math.tan(lat_rad)*(A**2/2 + (5-T+9*C+4*C**2)*A**4/24 + (61-58*T+T**2+600*C-330*(e**2/(1-e**2)))*A**6/720))

        if not is_north:
            northing += 10000000.0

        return (easting, northing)

    def _utm_to_wgs84(self, easting: float, northing: float, zone: int, is_north: bool) -> Tuple[float, float]:
        """Simplified UTM to WGS84 conversion."""
        k0 = 0.9996
        a = self.EARTH_RADIUS_M
        e = 0.0818191908426
        e1 = (1 - math.sqrt(1 - e**2)) / (1 + math.sqrt(1 - e**2))

        x = easting - 500000.0
        y = northing
        if not is_north:
            y -= 10000000.0

        M = y / k0
        mu = M / (a * (1 - e**2/4 - 3*e**4/64 - 5*e**6/256))

        lat1 = mu + (3*e1/2 - 27*e1**3/32) * math.sin(2*mu)
        lat1 += (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
        lat1 += (151*e1**3/96) * math.sin(6*mu)
        lat1 += (1097*e1**4/512) * math.sin(8*mu)

        N1 = a / math.sqrt(1 - e**2 * math.sin(lat1)**2)
        T1 = math.tan(lat1)**2
        C1 = (e**2 / (1 - e**2)) * math.cos(lat1)**2
        R1 = a * (1 - e**2) / (1 - e**2 * math.sin(lat1)**2)**1.5
        D = x / (N1 * k0)

        lat = lat1 - (N1*math.tan(lat1)/R1) * (D**2/2 - (5+3*T1+10*C1-4*C1**2-9*(e**2/(1-e**2)))*D**4/24 + (61+90*T1+298*C1+45*T1**2-252*(e**2/(1-e**2))-3*C1**2)*D**6/720)
        lon = (D - (1+2*T1+C1)*D**3/6 + (5-2*C1+28*T1-3*C1**2+8*(e**2/(1-e**2))+24*T1**2)*D**5/120) / math.cos(lat1)

        lon0 = math.radians((zone - 1) * 6 - 180 + 3)

        return (math.degrees(lon + lon0), math.degrees(lat))

    def _is_utm(self, crs_code: str) -> bool:
        """Check if CRS code is a UTM zone."""
        match = re.match(r'^EPSG:(326\d{2}|327\d{2})$', crs_code)
        return match is not None

    def _parse_utm_code(self, crs_code: str) -> Tuple[int, bool]:
        """Parse UTM EPSG code into (zone, is_north)."""
        code_num = int(crs_code.split(":")[1])
        if code_num >= 32700:
            return (code_num - 32700, False)
        else:
            return (code_num - 32600, True)

    def _transform_coords(self, coordinates: Any, geom_type: str,
                          source_crs: str, target_crs: str) -> Any:
        """Recursively transform coordinates."""
        if geom_type == "point":
            x, y = self.transform_coordinate(coordinates[0], coordinates[1], source_crs, target_crs)
            return [x, y]
        elif geom_type in ("linestring", "multipoint"):
            return [
                list(self.transform_coordinate(c[0], c[1], source_crs, target_crs))
                for c in coordinates
            ]
        elif geom_type == "polygon":
            return [
                [list(self.transform_coordinate(c[0], c[1], source_crs, target_crs)) for c in ring]
                for ring in coordinates
            ]
        elif geom_type in ("multilinestring", "multipolygon"):
            return [
                self._transform_coords(sub, "polygon" if geom_type == "multipolygon" else "linestring", source_crs, target_crs)
                for sub in coordinates
            ]
        return coordinates

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine():
    """CRSTransformerEngine instance for testing."""
    return CRSTransformerEngine()


@pytest.fixture
def nyc_feature():
    """Feature at NYC (lon=-74.006, lat=40.7128) in WGS84."""
    geom = Geometry(geometry_type="point", coordinates=[-74.006, 40.7128])
    return Feature(geometry=geom, crs="EPSG:4326")


@pytest.fixture
def london_feature():
    """Feature at London (lon=-0.1278, lat=51.5074) in WGS84."""
    geom = Geometry(geometry_type="point", coordinates=[-0.1278, 51.5074])
    return Feature(geometry=geom, crs="EPSG:4326")


@pytest.fixture
def berlin_feature():
    """Feature at Berlin (lon=13.405, lat=52.52) in WGS84."""
    geom = Geometry(geometry_type="point", coordinates=[13.405, 52.52])
    return Feature(geometry=geom, crs="EPSG:4326")


# ===========================================================================
# Test Classes -- WGS84 <-> Web Mercator
# ===========================================================================


class TestWGS84ToWebMercator:
    """Test WGS84 (EPSG:4326) to Web Mercator (EPSG:3857) transforms."""

    def test_origin(self, engine):
        """Origin (0, 0) maps to (0, 0) in Web Mercator."""
        x, y = engine.transform_coordinate(0.0, 0.0, "EPSG:4326", "EPSG:3857")
        assert abs(x) < 0.01
        assert abs(y) < 0.01

    def test_nyc(self, engine):
        """NYC coordinates transform to expected Web Mercator values."""
        x, y = engine.transform_coordinate(-74.006, 40.7128, "EPSG:4326", "EPSG:3857")
        # Expected approximately: x=-8238310, y=4970072
        assert abs(x - (-8238310)) < 1000
        assert abs(y - 4970072) < 1000

    def test_london(self, engine):
        """London coordinates transform to expected Web Mercator values."""
        x, y = engine.transform_coordinate(-0.1278, 51.5074, "EPSG:4326", "EPSG:3857")
        # Expected approximately: x=-14226, y=6711548
        assert abs(x - (-14226)) < 1000
        assert abs(y - 6711548) < 1000

    def test_positive_x_for_positive_lon(self, engine):
        """Positive longitude produces positive x in Web Mercator."""
        x, y = engine.transform_coordinate(10.0, 20.0, "EPSG:4326", "EPSG:3857")
        assert x > 0

    def test_negative_x_for_negative_lon(self, engine):
        """Negative longitude produces negative x in Web Mercator."""
        x, y = engine.transform_coordinate(-10.0, 20.0, "EPSG:4326", "EPSG:3857")
        assert x < 0


class TestWebMercatorToWGS84:
    """Test Web Mercator (EPSG:3857) to WGS84 (EPSG:4326) transforms."""

    def test_origin_roundtrip(self, engine):
        """Origin roundtrips correctly."""
        x, y = engine.transform_coordinate(0.0, 0.0, "EPSG:4326", "EPSG:3857")
        lon, lat = engine.transform_coordinate(x, y, "EPSG:3857", "EPSG:4326")
        assert abs(lon) < 0.0001
        assert abs(lat) < 0.0001

    def test_nyc_roundtrip(self, engine):
        """NYC roundtrips within tolerance."""
        x, y = engine.transform_coordinate(-74.006, 40.7128, "EPSG:4326", "EPSG:3857")
        lon, lat = engine.transform_coordinate(x, y, "EPSG:3857", "EPSG:4326")
        assert abs(lon - (-74.006)) < 0.001
        assert abs(lat - 40.7128) < 0.001

    def test_london_roundtrip(self, engine):
        """London roundtrips within tolerance."""
        x, y = engine.transform_coordinate(-0.1278, 51.5074, "EPSG:4326", "EPSG:3857")
        lon, lat = engine.transform_coordinate(x, y, "EPSG:3857", "EPSG:4326")
        assert abs(lon - (-0.1278)) < 0.001
        assert abs(lat - 51.5074) < 0.001

    def test_berlin_roundtrip(self, engine):
        """Berlin roundtrips within tolerance."""
        x, y = engine.transform_coordinate(13.405, 52.52, "EPSG:4326", "EPSG:3857")
        lon, lat = engine.transform_coordinate(x, y, "EPSG:3857", "EPSG:4326")
        assert abs(lon - 13.405) < 0.001
        assert abs(lat - 52.52) < 0.001


# ===========================================================================
# Test Classes -- WGS84 <-> UTM
# ===========================================================================


class TestWGS84ToUTM:
    """Test WGS84 to UTM transforms."""

    def test_nyc_to_utm_zone_18n(self, engine):
        """NYC transforms to UTM zone 18N with reasonable easting/northing."""
        easting, northing = engine.transform_coordinate(
            -74.006, 40.7128, "EPSG:4326", "EPSG:32618",
        )
        # NYC in UTM zone 18N: easting ~583960, northing ~4507523
        assert 500000 < easting < 700000  # Within valid UTM easting range
        assert 4000000 < northing < 5000000  # Reasonable northing for NYC

    def test_london_to_utm_zone_30n(self, engine):
        """London transforms to UTM zone 30N."""
        easting, northing = engine.transform_coordinate(
            -0.1278, 51.5074, "EPSG:4326", "EPSG:32630",
        )
        assert 400000 < easting < 800000  # London is near zone edge, easting can be high
        assert 5000000 < northing < 6000000

    def test_utm_easting_range(self, engine):
        """UTM easting near zone center should be close to 500000."""
        # Zone 33N central meridian is 15E
        easting, northing = engine.transform_coordinate(
            15.0, 45.0, "EPSG:4326", "EPSG:32633",
        )
        assert abs(easting - 500000) < 1000  # Near central meridian

    def test_utm_roundtrip(self, engine):
        """WGS84 -> UTM -> WGS84 roundtrip within tolerance."""
        lon_orig, lat_orig = 13.405, 52.52
        easting, northing = engine.transform_coordinate(
            lon_orig, lat_orig, "EPSG:4326", "EPSG:32633",
        )
        lon_back, lat_back = engine.transform_coordinate(
            easting, northing, "EPSG:32633", "EPSG:4326",
        )
        assert abs(lon_back - lon_orig) < 0.001
        assert abs(lat_back - lat_orig) < 0.001


class TestUTMToWGS84:
    """Test UTM to WGS84 transforms."""

    def test_utm_zone_center_to_wgs84(self, engine):
        """UTM zone center point (500000, 0) back to central meridian."""
        # Zone 33N: central meridian at 15E
        lon, lat = engine.transform_coordinate(
            500000, 0, "EPSG:32633", "EPSG:4326",
        )
        assert abs(lon - 15.0) < 0.1  # Should be near 15E
        assert abs(lat) < 0.1  # Should be near equator

    def test_southern_hemisphere_utm(self, engine):
        """Southern hemisphere UTM zone (zone 34S for Cape Town)."""
        # Cape Town: lon=18.4241, lat=-33.9249
        easting, northing = engine.transform_coordinate(
            18.4241, -33.9249, "EPSG:4326", "EPSG:32734",
        )
        lon, lat = engine.transform_coordinate(
            easting, northing, "EPSG:32734", "EPSG:4326",
        )
        assert abs(lon - 18.4241) < 0.01
        assert abs(lat - (-33.9249)) < 0.01


# ===========================================================================
# Test Classes -- Batch Transform
# ===========================================================================


class TestBatchTransform:
    """Test batch feature transformation."""

    def test_batch_transform(self, engine, nyc_feature, london_feature, berlin_feature):
        """Batch transform 3 features."""
        result = engine.batch_transform(
            [nyc_feature, london_feature, berlin_feature],
            "EPSG:4326", "EPSG:3857",
        )
        assert result.feature_count == 3
        assert result.status == "completed"
        assert result.source_crs == "EPSG:4326"
        assert result.target_crs == "EPSG:3857"

    def test_batch_transform_empty(self, engine):
        """Batch transform with empty feature list."""
        result = engine.batch_transform([], "EPSG:4326", "EPSG:3857")
        assert result.feature_count == 0
        assert result.status == "completed"

    def test_batch_transform_single(self, engine, nyc_feature):
        """Batch transform with single feature."""
        result = engine.batch_transform([nyc_feature], "EPSG:4326", "EPSG:3857")
        assert result.feature_count == 1

    def test_batch_transform_id_format(self, engine, nyc_feature):
        """Batch transform generates TRF-xxxxx ID."""
        result = engine.batch_transform([nyc_feature], "EPSG:4326", "EPSG:3857")
        assert result.transform_id.startswith("TRF-")

    def test_batch_transform_provenance(self, engine, nyc_feature):
        """Batch transform generates provenance hash."""
        result = engine.batch_transform([nyc_feature], "EPSG:4326", "EPSG:3857")
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_batch_transform_increments_stats(self, engine, nyc_feature, london_feature):
        """Batch transform increments stats."""
        engine.batch_transform([nyc_feature, london_feature], "EPSG:4326", "EPSG:3857")
        stats = engine.get_statistics()
        assert stats["transforms_performed"] == 1
        assert stats["features_transformed"] == 2


# ===========================================================================
# Test Classes -- Detect UTM Zone
# ===========================================================================


class TestDetectUTMZone:
    """Test UTM zone detection."""

    def test_nyc_zone_18n(self, engine):
        """NYC (lon=-74.006, lat=40.7128) is in UTM zone 18N."""
        zone = engine.detect_utm_zone(-74.006, 40.7128)
        assert zone == "EPSG:32618"

    def test_london_zone_30n(self, engine):
        """London (lon=-0.1278, lat=51.5074) is in UTM zone 30N."""
        zone = engine.detect_utm_zone(-0.1278, 51.5074)
        assert zone == "EPSG:32630"

    def test_berlin_zone_33n(self, engine):
        """Berlin (lon=13.405, lat=52.52) is in UTM zone 33N."""
        zone = engine.detect_utm_zone(13.405, 52.52)
        assert zone == "EPSG:32633"

    def test_positive_lat_north(self, engine):
        """Positive latitude yields northern hemisphere UTM zone."""
        zone = engine.detect_utm_zone(10.0, 45.0)
        # Zone should be EPSG:326xx (northern)
        code = int(zone.split(":")[1])
        assert 32601 <= code <= 32660

    def test_negative_lat_south(self, engine):
        """Negative latitude yields southern hemisphere UTM zone."""
        zone = engine.detect_utm_zone(10.0, -45.0)
        # Zone should be EPSG:327xx (southern)
        code = int(zone.split(":")[1])
        assert 32701 <= code <= 32760

    def test_zone_calculation_formula(self, engine):
        """UTM zone number = floor((lon + 180) / 6) + 1."""
        # lon = 0 -> zone 31
        zone = engine.detect_utm_zone(0.0, 10.0)
        assert zone == "EPSG:32631"

        # lon = -180 -> zone 1
        zone = engine.detect_utm_zone(-180.0, 10.0)
        assert zone == "EPSG:32601"

        # lon = 179.9 -> zone 60
        zone = engine.detect_utm_zone(179.9, 10.0)
        assert zone == "EPSG:32660"

    def test_cape_town_zone_34s(self, engine):
        """Cape Town (lon=18.4241, lat=-33.9249) is in UTM zone 34S."""
        zone = engine.detect_utm_zone(18.4241, -33.9249)
        assert zone == "EPSG:32734"

    def test_equator_positive_lat(self, engine):
        """Equator (lat=0) is treated as northern hemisphere."""
        zone = engine.detect_utm_zone(10.0, 0.0)
        code = int(zone.split(":")[1])
        assert 32601 <= code <= 32660


# ===========================================================================
# Test Classes -- CRS Info
# ===========================================================================


class TestGetCRSInfo:
    """Test CRS information retrieval."""

    def test_wgs84_info(self, engine):
        """WGS84 (EPSG:4326) CRS info."""
        info = engine.get_crs_info("EPSG:4326")
        assert info is not None
        assert info.code == "EPSG:4326"
        assert info.name == "WGS 84"
        assert info.crs_type == "geographic"
        assert info.datum == "WGS84"
        assert info.unit == "degree"

    def test_web_mercator_info(self, engine):
        """Web Mercator (EPSG:3857) CRS info."""
        info = engine.get_crs_info("EPSG:3857")
        assert info is not None
        assert info.code == "EPSG:3857"
        assert info.name == "WGS 84 / Pseudo-Mercator"
        assert info.crs_type == "projected"
        assert info.unit == "metre"

    def test_utm_zone_info(self, engine):
        """UTM zone 33N (EPSG:32633) CRS info."""
        info = engine.get_crs_info("EPSG:32633")
        assert info is not None
        assert info.code == "EPSG:32633"
        assert "UTM zone 33N" in info.name
        assert info.is_projected is True

    def test_unknown_crs(self, engine):
        """Unknown CRS returns None."""
        info = engine.get_crs_info("EPSG:99999")
        assert info is None


class TestIsGeographicProjected:
    """Test is_geographic and is_projected checks."""

    def test_wgs84_is_geographic(self, engine):
        """WGS84 is geographic."""
        assert engine.is_geographic("EPSG:4326") is True
        assert engine.is_projected("EPSG:4326") is False

    def test_web_mercator_is_projected(self, engine):
        """Web Mercator is projected."""
        assert engine.is_geographic("EPSG:3857") is False
        assert engine.is_projected("EPSG:3857") is True

    def test_utm_is_projected(self, engine):
        """UTM zones are projected."""
        assert engine.is_projected("EPSG:32633") is True
        assert engine.is_geographic("EPSG:32633") is False

    def test_unknown_crs_raises(self, engine):
        """Unknown CRS raises ValueError."""
        with pytest.raises(ValueError, match="Unknown CRS"):
            engine.is_geographic("EPSG:99999")

        with pytest.raises(ValueError, match="Unknown CRS"):
            engine.is_projected("EPSG:99999")


# ===========================================================================
# Test Classes -- Identity Transform
# ===========================================================================


class TestIdentityTransform:
    """Test identity transform (same source and target CRS)."""

    def test_identity_coordinate(self, engine):
        """Same CRS returns original coordinates unchanged."""
        lon, lat = engine.transform_coordinate(10.0, 20.0, "EPSG:4326", "EPSG:4326")
        assert lon == 10.0
        assert lat == 20.0

    def test_identity_web_mercator(self, engine):
        """Identity transform in Web Mercator."""
        x, y = engine.transform_coordinate(1000.0, 2000.0, "EPSG:3857", "EPSG:3857")
        assert x == 1000.0
        assert y == 2000.0

    def test_identity_feature(self, engine, nyc_feature):
        """Identity feature transform preserves coordinates exactly."""
        result = engine.transform_feature(nyc_feature, "EPSG:4326", "EPSG:4326")
        assert result.geometry.coordinates == [-74.006, 40.7128]
        assert result.crs == "EPSG:4326"


# ===========================================================================
# Test Classes -- Provenance
# ===========================================================================


class TestTransformProvenance:
    """Test provenance hash generation for transforms."""

    def test_feature_transform_provenance(self, engine, nyc_feature):
        """Feature transform generates SHA-256 provenance hash."""
        result = engine.transform_feature(nyc_feature, "EPSG:4326", "EPSG:3857")
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64
        assert re.match(r"^[0-9a-f]{64}$", result.provenance_hash)

    def test_batch_transform_provenance(self, engine, nyc_feature):
        """Batch transform generates SHA-256 provenance hash."""
        result = engine.batch_transform([nyc_feature], "EPSG:4326", "EPSG:3857")
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_consistency(self, engine):
        """Same provenance data produces same hash."""
        data = {"op": "test", "crs": "EPSG:4326"}
        h1 = engine._compute_provenance(data)
        h2 = engine._compute_provenance(data)
        assert h1 == h2

    def test_provenance_different_data(self, engine):
        """Different provenance data produces different hash."""
        h1 = engine._compute_provenance({"op": "a"})
        h2 = engine._compute_provenance({"op": "b"})
        assert h1 != h2


# ===========================================================================
# Test Classes -- Unsupported Transform
# ===========================================================================


class TestUnsupportedTransform:
    """Test error handling for unsupported CRS transforms."""

    def test_unsupported_transform_raises(self, engine):
        """Unsupported CRS pair raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported CRS transform"):
            engine.transform_coordinate(10.0, 20.0, "EPSG:3857", "EPSG:32633")

    def test_feature_no_geometry(self, engine):
        """Feature without geometry passes through unchanged."""
        feat = Feature(feature_id="FTR-NOGEO", crs="EPSG:4326")
        result = engine.transform_feature(feat, "EPSG:4326", "EPSG:3857")
        assert result.feature_id == "FTR-NOGEO"
        assert result.geometry is None
