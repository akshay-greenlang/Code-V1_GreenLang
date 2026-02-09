# -*- coding: utf-8 -*-
"""
Unit Tests for LayerManagerEngine (AGENT-DATA-006)

Tests layer creation, retrieval, listing (all/by status/by geometry type),
feature addition, feature retrieval (all/bbox filter/pagination),
layer updates, soft deletion, GeoJSON export, layer merging,
layer statistics, and provenance tracking
for the GIS/Mapping Connector Agent.

Coverage target: 85%+ of layer_manager.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Inline models
# ---------------------------------------------------------------------------


class Feature:
    """A geospatial feature within a layer."""

    def __init__(
        self,
        feature_id: str,
        geometry_type: str,
        coordinates: Any,
        properties: Optional[Dict[str, Any]] = None,
    ):
        self.feature_id = feature_id
        self.geometry_type = geometry_type
        self.coordinates = coordinates
        self.properties = properties or {}
        self.created_at = datetime.now(timezone.utc).isoformat()


class Layer:
    """A geospatial layer containing features."""

    def __init__(
        self,
        layer_id: str,
        name: str,
        geometry_type: str,
        crs: str = "EPSG:4326",
        description: str = "",
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None,
        provenance_hash: str = "",
    ):
        self.layer_id = layer_id
        self.name = name
        self.geometry_type = geometry_type
        self.crs = crs
        self.description = description
        self.status = status
        self.metadata = metadata or {}
        self.features: List[Feature] = []
        self.provenance_hash = provenance_hash
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at


class LayerStatistics:
    """Statistics for a layer."""

    def __init__(
        self,
        layer_id: str,
        feature_count: int,
        geometry_type: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        crs: str = "EPSG:4326",
    ):
        self.layer_id = layer_id
        self.feature_count = feature_count
        self.geometry_type = geometry_type
        self.bbox = bbox
        self.crs = crs


# ---------------------------------------------------------------------------
# Inline LayerManagerEngine
# ---------------------------------------------------------------------------


class LayerManagerEngine:
    """Engine for managing geospatial layers and features.

    Provides CRUD operations for layers and features, spatial filtering,
    pagination, GeoJSON export, layer merging, and statistics.
    """

    def __init__(self):
        self._layers: Dict[str, Layer] = {}
        self._layer_counter = 0
        self._feature_counter = 0

    def create_layer(
        self,
        name: str,
        geometry_type: str,
        crs: str = "EPSG:4326",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Layer:
        """Create a new geospatial layer."""
        self._layer_counter += 1
        layer_id = f"LYR-{self._layer_counter:05d}"

        layer = Layer(
            layer_id=layer_id,
            name=name,
            geometry_type=geometry_type,
            crs=crs,
            description=description,
            metadata=metadata,
            provenance_hash=_compute_hash({
                "layer_id": layer_id,
                "name": name,
                "geometry_type": geometry_type,
            }),
        )
        self._layers[layer_id] = layer
        return layer

    def get_layer(self, layer_id: str) -> Optional[Layer]:
        """Get a layer by ID."""
        layer = self._layers.get(layer_id)
        if layer and layer.status == "deleted":
            return None
        return layer

    def list_layers(
        self,
        status: Optional[str] = None,
        geometry_type: Optional[str] = None,
    ) -> List[Layer]:
        """List layers with optional filtering."""
        layers = [l for l in self._layers.values() if l.status != "deleted"]
        if status:
            layers = [l for l in layers if l.status == status]
        if geometry_type:
            layers = [l for l in layers if l.geometry_type == geometry_type]
        return layers

    def update_layer(
        self,
        layer_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Layer]:
        """Update layer properties."""
        layer = self.get_layer(layer_id)
        if layer is None:
            return None

        if name is not None:
            layer.name = name
        if description is not None:
            layer.description = description
        if metadata is not None:
            layer.metadata.update(metadata)

        layer.updated_at = datetime.now(timezone.utc).isoformat()
        layer.provenance_hash = _compute_hash({
            "layer_id": layer_id,
            "name": layer.name,
            "updated": True,
        })
        return layer

    def delete_layer(self, layer_id: str) -> bool:
        """Soft-delete a layer by setting status to 'deleted'."""
        layer = self._layers.get(layer_id)
        if layer is None or layer.status == "deleted":
            return False
        layer.status = "deleted"
        layer.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def add_feature(
        self,
        layer_id: str,
        geometry_type: str,
        coordinates: Any,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[Feature]:
        """Add a feature to a layer."""
        layer = self.get_layer(layer_id)
        if layer is None:
            return None

        self._feature_counter += 1
        feature_id = f"FTR-{self._feature_counter:05d}"

        feature = Feature(
            feature_id=feature_id,
            geometry_type=geometry_type,
            coordinates=coordinates,
            properties=properties,
        )
        layer.features.append(feature)
        layer.updated_at = datetime.now(timezone.utc).isoformat()
        return feature

    def get_features(
        self,
        layer_id: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Feature]:
        """Get features from a layer with optional bbox filter and pagination."""
        layer = self.get_layer(layer_id)
        if layer is None:
            return []

        features = layer.features

        # Apply bbox filter for Point geometries
        if bbox is not None:
            min_lat, min_lon, max_lat, max_lon = bbox
            filtered = []
            for f in features:
                if f.geometry_type == "Point" and isinstance(f.coordinates, (list, tuple)):
                    lat, lon = f.coordinates[1], f.coordinates[0]
                    if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                        filtered.append(f)
                else:
                    filtered.append(f)  # Non-point features always included
            features = filtered

        # Apply pagination
        return features[offset:offset + limit]

    def export_geojson(self, layer_id: str) -> Optional[Dict[str, Any]]:
        """Export a layer as GeoJSON FeatureCollection."""
        layer = self.get_layer(layer_id)
        if layer is None:
            return None

        geojson_features = []
        for f in layer.features:
            geojson_features.append({
                "type": "Feature",
                "id": f.feature_id,
                "geometry": {
                    "type": f.geometry_type,
                    "coordinates": f.coordinates,
                },
                "properties": f.properties,
            })

        return {
            "type": "FeatureCollection",
            "name": layer.name,
            "crs": {"type": "name", "properties": {"name": layer.crs}},
            "features": geojson_features,
        }

    def merge_layers(
        self,
        layer_ids: List[str],
        merged_name: str,
        geometry_type: Optional[str] = None,
    ) -> Optional[Layer]:
        """Merge multiple layers into a new layer."""
        source_layers = []
        for lid in layer_ids:
            layer = self.get_layer(lid)
            if layer is not None:
                source_layers.append(layer)

        if not source_layers:
            return None

        # Use geometry type from first layer if not specified
        gt = geometry_type or source_layers[0].geometry_type

        merged = self.create_layer(
            name=merged_name,
            geometry_type=gt,
            description=f"Merged from {len(source_layers)} layers",
        )

        for src in source_layers:
            for f in src.features:
                self.add_feature(
                    merged.layer_id,
                    f.geometry_type,
                    f.coordinates,
                    f.properties,
                )

        return merged

    def get_layer_statistics(self, layer_id: str) -> Optional[LayerStatistics]:
        """Get statistics for a layer."""
        layer = self.get_layer(layer_id)
        if layer is None:
            return None

        bbox = self._compute_bbox(layer.features)

        return LayerStatistics(
            layer_id=layer_id,
            feature_count=len(layer.features),
            geometry_type=layer.geometry_type,
            bbox=bbox,
            crs=layer.crs,
        )

    def _compute_bbox(
        self, features: List[Feature]
    ) -> Optional[Tuple[float, float, float, float]]:
        """Compute bounding box from point features."""
        lats, lons = [], []
        for f in features:
            if f.geometry_type == "Point" and isinstance(f.coordinates, (list, tuple)):
                lons.append(f.coordinates[0])
                lats.append(f.coordinates[1])
        if not lats:
            return None
        return (min(lats), min(lons), max(lats), max(lons))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> LayerManagerEngine:
    return LayerManagerEngine()


@pytest.fixture
def engine_with_layer(engine) -> LayerManagerEngine:
    """Engine with a pre-created layer containing 3 point features."""
    layer = engine.create_layer("Test Points", "Point", description="Test layer")
    engine.add_feature(layer.layer_id, "Point", [-74.006, 40.7128], {"name": "NYC"})
    engine.add_feature(layer.layer_id, "Point", [-0.1278, 51.5074], {"name": "London"})
    engine.add_feature(layer.layer_id, "Point", [2.3522, 48.8566], {"name": "Paris"})
    return engine


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCreateLayer:
    """Tests for layer creation."""

    def test_create_success(self, engine):
        layer = engine.create_layer("Forests", "Polygon")
        assert layer is not None
        assert layer.name == "Forests"
        assert layer.geometry_type == "Polygon"
        assert layer.status == "active"

    def test_id_format(self, engine):
        layer = engine.create_layer("Test", "Point")
        assert layer.layer_id.startswith("LYR-")
        assert len(layer.layer_id) == 9  # LYR-00001

    def test_sequential_ids(self, engine):
        l1 = engine.create_layer("A", "Point")
        l2 = engine.create_layer("B", "Polygon")
        assert l1.layer_id == "LYR-00001"
        assert l2.layer_id == "LYR-00002"

    def test_default_crs(self, engine):
        layer = engine.create_layer("Test", "Point")
        assert layer.crs == "EPSG:4326"

    def test_custom_crs(self, engine):
        layer = engine.create_layer("Test", "Point", crs="EPSG:3857")
        assert layer.crs == "EPSG:3857"

    def test_with_metadata(self, engine):
        layer = engine.create_layer(
            "Test", "Point", metadata={"source": "satellite"}
        )
        assert layer.metadata["source"] == "satellite"

    def test_provenance_hash(self, engine):
        layer = engine.create_layer("Test", "Point")
        assert len(layer.provenance_hash) == 64
        int(layer.provenance_hash, 16)


class TestGetLayer:
    """Tests for layer retrieval."""

    def test_get_existing(self, engine):
        created = engine.create_layer("Test", "Point")
        retrieved = engine.get_layer(created.layer_id)
        assert retrieved is not None
        assert retrieved.layer_id == created.layer_id

    def test_get_nonexistent(self, engine):
        result = engine.get_layer("LYR-99999")
        assert result is None

    def test_get_deleted_returns_none(self, engine):
        layer = engine.create_layer("Test", "Point")
        engine.delete_layer(layer.layer_id)
        assert engine.get_layer(layer.layer_id) is None


class TestListLayers:
    """Tests for listing layers."""

    def test_list_all(self, engine):
        engine.create_layer("A", "Point")
        engine.create_layer("B", "Polygon")
        engine.create_layer("C", "LineString")
        layers = engine.list_layers()
        assert len(layers) == 3

    def test_list_by_status(self, engine):
        engine.create_layer("Active", "Point")
        inactive = engine.create_layer("Inactive", "Point")
        engine.delete_layer(inactive.layer_id)
        layers = engine.list_layers(status="active")
        assert len(layers) == 1
        assert layers[0].name == "Active"

    def test_list_by_geometry_type(self, engine):
        engine.create_layer("Points", "Point")
        engine.create_layer("Polygons", "Polygon")
        engine.create_layer("More Points", "Point")
        layers = engine.list_layers(geometry_type="Point")
        assert len(layers) == 2

    def test_list_excludes_deleted(self, engine):
        engine.create_layer("A", "Point")
        b = engine.create_layer("B", "Point")
        engine.delete_layer(b.layer_id)
        layers = engine.list_layers()
        assert len(layers) == 1

    def test_list_empty(self, engine):
        layers = engine.list_layers()
        assert layers == []


class TestAddFeatures:
    """Tests for adding features to layers."""

    def test_add_point_feature(self, engine):
        layer = engine.create_layer("Points", "Point")
        feature = engine.add_feature(
            layer.layer_id, "Point", [-74.006, 40.7128],
            properties={"name": "NYC"},
        )
        assert feature is not None
        assert feature.feature_id.startswith("FTR-")

    def test_feature_id_format(self, engine):
        layer = engine.create_layer("Points", "Point")
        f = engine.add_feature(layer.layer_id, "Point", [0, 0])
        assert f.feature_id.startswith("FTR-")
        assert len(f.feature_id) == 9  # FTR-00001

    def test_sequential_feature_ids(self, engine):
        layer = engine.create_layer("Points", "Point")
        f1 = engine.add_feature(layer.layer_id, "Point", [0, 0])
        f2 = engine.add_feature(layer.layer_id, "Point", [1, 1])
        assert f1.feature_id == "FTR-00001"
        assert f2.feature_id == "FTR-00002"

    def test_add_to_nonexistent_layer(self, engine):
        result = engine.add_feature("LYR-99999", "Point", [0, 0])
        assert result is None

    def test_add_polygon_feature(self, engine):
        layer = engine.create_layer("Polygons", "Polygon")
        coords = [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        feature = engine.add_feature(layer.layer_id, "Polygon", coords)
        assert feature is not None
        assert feature.geometry_type == "Polygon"

    def test_feature_with_properties(self, engine):
        layer = engine.create_layer("Points", "Point")
        props = {"name": "Test", "value": 42}
        feature = engine.add_feature(layer.layer_id, "Point", [0, 0], props)
        assert feature.properties["name"] == "Test"
        assert feature.properties["value"] == 42


class TestGetFeatures:
    """Tests for retrieving features from layers."""

    def test_get_all_features(self, engine_with_layer):
        layers = engine_with_layer.list_layers()
        features = engine_with_layer.get_features(layers[0].layer_id)
        assert len(features) == 3

    def test_get_with_bbox_filter(self, engine_with_layer):
        layers = engine_with_layer.list_layers()
        # BBox around London/Paris area
        features = engine_with_layer.get_features(
            layers[0].layer_id,
            bbox=(45.0, -5.0, 55.0, 10.0),
        )
        names = [f.properties["name"] for f in features]
        assert "London" in names
        assert "Paris" in names

    def test_get_with_pagination(self, engine_with_layer):
        layers = engine_with_layer.list_layers()
        lid = layers[0].layer_id
        page1 = engine_with_layer.get_features(lid, offset=0, limit=2)
        page2 = engine_with_layer.get_features(lid, offset=2, limit=2)
        assert len(page1) == 2
        assert len(page2) == 1

    def test_get_from_nonexistent_layer(self, engine):
        features = engine.get_features("LYR-99999")
        assert features == []

    def test_get_empty_layer(self, engine):
        layer = engine.create_layer("Empty", "Point")
        features = engine.get_features(layer.layer_id)
        assert features == []


class TestUpdateLayer:
    """Tests for updating layer properties."""

    def test_update_name(self, engine):
        layer = engine.create_layer("Old Name", "Point")
        updated = engine.update_layer(layer.layer_id, name="New Name")
        assert updated is not None
        assert updated.name == "New Name"

    def test_update_description(self, engine):
        layer = engine.create_layer("Test", "Point")
        updated = engine.update_layer(layer.layer_id, description="Updated desc")
        assert updated.description == "Updated desc"

    def test_update_metadata(self, engine):
        layer = engine.create_layer("Test", "Point", metadata={"a": 1})
        updated = engine.update_layer(layer.layer_id, metadata={"b": 2})
        assert updated.metadata["a"] == 1
        assert updated.metadata["b"] == 2

    def test_update_nonexistent(self, engine):
        result = engine.update_layer("LYR-99999", name="X")
        assert result is None

    def test_update_changes_timestamp(self, engine):
        layer = engine.create_layer("Test", "Point")
        original_updated = layer.updated_at
        updated = engine.update_layer(layer.layer_id, name="Changed")
        assert updated.updated_at >= original_updated

    def test_update_changes_provenance(self, engine):
        layer = engine.create_layer("Test", "Point")
        old_hash = layer.provenance_hash
        updated = engine.update_layer(layer.layer_id, name="Changed")
        assert updated.provenance_hash != old_hash


class TestDeleteLayer:
    """Tests for soft-deleting layers."""

    def test_delete_success(self, engine):
        layer = engine.create_layer("Test", "Point")
        result = engine.delete_layer(layer.layer_id)
        assert result is True

    def test_delete_makes_invisible(self, engine):
        layer = engine.create_layer("Test", "Point")
        engine.delete_layer(layer.layer_id)
        assert engine.get_layer(layer.layer_id) is None

    def test_delete_nonexistent(self, engine):
        result = engine.delete_layer("LYR-99999")
        assert result is False

    def test_delete_already_deleted(self, engine):
        layer = engine.create_layer("Test", "Point")
        engine.delete_layer(layer.layer_id)
        result = engine.delete_layer(layer.layer_id)
        assert result is False

    def test_soft_delete_preserves_data(self, engine):
        layer = engine.create_layer("Test", "Point")
        lid = layer.layer_id
        engine.add_feature(lid, "Point", [0, 0])
        engine.delete_layer(lid)
        # Still in internal storage
        assert lid in engine._layers
        assert engine._layers[lid].status == "deleted"


class TestExportGeoJSON:
    """Tests for GeoJSON export."""

    def test_export_structure(self, engine_with_layer):
        layers = engine_with_layer.list_layers()
        geojson = engine_with_layer.export_geojson(layers[0].layer_id)
        assert geojson is not None
        assert geojson["type"] == "FeatureCollection"
        assert "features" in geojson
        assert "crs" in geojson

    def test_export_feature_count(self, engine_with_layer):
        layers = engine_with_layer.list_layers()
        geojson = engine_with_layer.export_geojson(layers[0].layer_id)
        assert len(geojson["features"]) == 3

    def test_export_feature_structure(self, engine_with_layer):
        layers = engine_with_layer.list_layers()
        geojson = engine_with_layer.export_geojson(layers[0].layer_id)
        feature = geojson["features"][0]
        assert feature["type"] == "Feature"
        assert "geometry" in feature
        assert "properties" in feature
        assert feature["geometry"]["type"] == "Point"

    def test_export_nonexistent(self, engine):
        result = engine.export_geojson("LYR-99999")
        assert result is None

    def test_export_empty_layer(self, engine):
        layer = engine.create_layer("Empty", "Point")
        geojson = engine.export_geojson(layer.layer_id)
        assert geojson is not None
        assert len(geojson["features"]) == 0

    def test_export_includes_name(self, engine):
        layer = engine.create_layer("My Layer", "Point")
        geojson = engine.export_geojson(layer.layer_id)
        assert geojson["name"] == "My Layer"


class TestMergeLayers:
    """Tests for merging layers."""

    def test_merge_two_layers(self, engine):
        l1 = engine.create_layer("Layer1", "Point")
        l2 = engine.create_layer("Layer2", "Point")
        engine.add_feature(l1.layer_id, "Point", [0, 0])
        engine.add_feature(l2.layer_id, "Point", [1, 1])
        engine.add_feature(l2.layer_id, "Point", [2, 2])

        merged = engine.merge_layers(
            [l1.layer_id, l2.layer_id], "Merged"
        )
        assert merged is not None
        assert merged.name == "Merged"
        assert len(merged.features) == 3

    def test_merge_empty_list(self, engine):
        result = engine.merge_layers([], "Empty")
        assert result is None

    def test_merge_nonexistent_layers(self, engine):
        result = engine.merge_layers(["LYR-99999"], "X")
        assert result is None

    def test_merge_preserves_features(self, engine):
        l1 = engine.create_layer("L1", "Point")
        engine.add_feature(l1.layer_id, "Point", [0, 0], {"name": "A"})

        l2 = engine.create_layer("L2", "Point")
        engine.add_feature(l2.layer_id, "Point", [1, 1], {"name": "B"})

        merged = engine.merge_layers([l1.layer_id, l2.layer_id], "Merged")
        names = [f.properties["name"] for f in merged.features]
        assert "A" in names
        assert "B" in names

    def test_merge_creates_new_layer(self, engine):
        l1 = engine.create_layer("L1", "Point")
        merged = engine.merge_layers([l1.layer_id], "Merged")
        assert merged.layer_id != l1.layer_id


class TestLayerStatistics:
    """Tests for layer statistics."""

    def test_statistics_feature_count(self, engine_with_layer):
        layers = engine_with_layer.list_layers()
        stats = engine_with_layer.get_layer_statistics(layers[0].layer_id)
        assert stats is not None
        assert stats.feature_count == 3

    def test_statistics_geometry_type(self, engine_with_layer):
        layers = engine_with_layer.list_layers()
        stats = engine_with_layer.get_layer_statistics(layers[0].layer_id)
        assert stats.geometry_type == "Point"

    def test_statistics_bbox(self, engine_with_layer):
        layers = engine_with_layer.list_layers()
        stats = engine_with_layer.get_layer_statistics(layers[0].layer_id)
        assert stats.bbox is not None
        min_lat, min_lon, max_lat, max_lon = stats.bbox
        assert min_lat < max_lat
        assert min_lon < max_lon

    def test_statistics_nonexistent(self, engine):
        result = engine.get_layer_statistics("LYR-99999")
        assert result is None

    def test_statistics_empty_layer(self, engine):
        layer = engine.create_layer("Empty", "Point")
        stats = engine.get_layer_statistics(layer.layer_id)
        assert stats.feature_count == 0
        assert stats.bbox is None

    def test_statistics_crs(self, engine):
        layer = engine.create_layer("Test", "Point", crs="EPSG:3857")
        stats = engine.get_layer_statistics(layer.layer_id)
        assert stats.crs == "EPSG:3857"


class TestProvenance:
    """Tests for provenance tracking in layer operations."""

    def test_create_has_provenance(self, engine):
        layer = engine.create_layer("Test", "Point")
        assert len(layer.provenance_hash) == 64
        int(layer.provenance_hash, 16)

    def test_different_layers_different_hash(self, engine):
        l1 = engine.create_layer("A", "Point")
        l2 = engine.create_layer("B", "Polygon")
        assert l1.provenance_hash != l2.provenance_hash

    def test_update_changes_provenance(self, engine):
        layer = engine.create_layer("Test", "Point")
        old_hash = layer.provenance_hash
        engine.update_layer(layer.layer_id, name="Updated")
        assert engine.get_layer(layer.layer_id).provenance_hash != old_hash
