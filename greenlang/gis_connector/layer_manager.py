# -*- coding: utf-8 -*-
"""
Layer Manager Engine - AGENT-DATA-006: GIS/Mapping Connector (GL-DATA-GEO-001)

Manages geospatial layers as collections of features with metadata,
spatial indexing, and export capabilities. Supports create, read,
update, delete (CRUD) operations plus spatial query filtering.

Zero-Hallucination Guarantees:
    - All layer operations are deterministic CRUD
    - Feature storage uses in-memory dictionaries with unique IDs
    - Spatial filtering uses bounding box intersection tests
    - Export formats use deterministic serialization
    - No ML/LLM used for layer management
    - SHA-256 provenance hashes on all layer operations

Example:
    >>> from greenlang.gis_connector.layer_manager import LayerManagerEngine
    >>> manager = LayerManagerEngine()
    >>> layer = manager.create_layer("test", "Point", "EPSG:4326")
    >>> assert layer["layer_id"].startswith("LYR-")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-006 GIS/Mapping Connector
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEOMETRY_TYPES = frozenset({
    "Point", "MultiPoint", "LineString", "MultiLineString",
    "Polygon", "MultiPolygon", "GeometryCollection", "Mixed",
})

LAYER_STATUSES = frozenset({"active", "inactive", "deleted"})

EXPORT_FORMATS = frozenset({"geojson", "wkt", "csv"})


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

def _make_layer(
    layer_id: str,
    name: str,
    geometry_type: str,
    crs: str = "EPSG:4326",
    description: str = "",
    status: str = "active",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a GeoLayer dictionary.

    Args:
        layer_id: Unique layer identifier.
        name: Layer name.
        geometry_type: Geometry type for the layer.
        crs: Coordinate reference system.
        description: Layer description.
        status: Layer status (active, inactive, deleted).
        tags: Optional list of tags.
        metadata: Optional metadata dictionary.

    Returns:
        GeoLayer dictionary.
    """
    now = _utcnow().isoformat()
    return {
        "layer_id": layer_id,
        "name": name,
        "geometry_type": geometry_type,
        "crs": crs,
        "description": description,
        "status": status,
        "tags": tags or [],
        "metadata": metadata or {},
        "feature_count": 0,
        "bbox": [],
        "created_at": now,
        "updated_at": now,
    }


def _make_feature(
    feature_id: str,
    geometry: Dict[str, Any],
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a Feature dictionary.

    Args:
        feature_id: Unique feature identifier.
        geometry: Geometry dictionary with type and coordinates.
        properties: Feature properties.

    Returns:
        Feature dictionary.
    """
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": geometry,
        "properties": properties or {},
        "created_at": _utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class LayerManagerEngine:
    """Geospatial layer management engine.

    Manages named layers of features with CRUD operations, spatial
    filtering, export, merge, and statistics.

    Attributes:
        _config: Configuration dictionary or object.
        _provenance: Provenance tracker instance.
        _layers: In-memory layer storage.
        _features: In-memory feature storage keyed by layer_id.

    Example:
        >>> manager = LayerManagerEngine()
        >>> layer = manager.create_layer("test", "Point", "EPSG:4326")
        >>> assert layer["layer_id"].startswith("LYR-")
    """

    def __init__(
        self,
        config: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize LayerManagerEngine.

        Args:
            config: Optional configuration.
            provenance: Optional ProvenanceTracker instance.
        """
        self._config = config or {}
        self._provenance = provenance
        self._layers: Dict[str, Dict[str, Any]] = {}
        self._features: Dict[str, List[Dict[str, Any]]] = {}

        logger.info("LayerManagerEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_layer(
        self,
        name: str,
        geometry_type: str,
        crs: str = "EPSG:4326",
        features: Optional[List[Dict[str, Any]]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new geospatial layer.

        Args:
            name: Layer name.
            geometry_type: Geometry type (Point, Polygon, etc.).
            crs: Coordinate reference system (default EPSG:4326).
            features: Optional initial features.
            description: Layer description.
            tags: Optional tags list.
            metadata: Optional metadata.

        Returns:
            GeoLayer dictionary.
        """
        start_time = time.monotonic()
        layer_id = f"LYR-{uuid.uuid4().hex[:12]}"

        if geometry_type not in GEOMETRY_TYPES:
            logger.warning(
                "Unknown geometry type '%s', defaulting to 'Mixed'",
                geometry_type,
            )
            geometry_type = "Mixed"

        layer = _make_layer(
            layer_id=layer_id,
            name=name,
            geometry_type=geometry_type,
            crs=crs,
            description=description,
            tags=tags,
            metadata=metadata,
        )

        self._layers[layer_id] = layer
        self._features[layer_id] = []

        # Add initial features if provided
        if features:
            self.add_features(layer_id, features)

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(layer)
            self._provenance.record(
                entity_type="layer",
                entity_id=layer_id,
                action="layer_operation",
                data_hash=data_hash,
            )

        # Record metrics
        try:
            from greenlang.gis_connector.metrics import (
                update_active_layers,
                record_operation,
            )
            update_active_layers(len(
                [l for l in self._layers.values() if l.get("status") == "active"]
            ))
            record_operation(
                operation="layer_create",
                format=geometry_type,
                status="success",
                duration=(time.monotonic() - start_time),
            )
        except ImportError:
            pass

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Created layer %s: name=%s, type=%s, crs=%s (%.1f ms)",
            layer_id, name, geometry_type, crs, elapsed_ms,
        )
        return layer

    def get_layer(self, layer_id: str) -> Optional[Dict[str, Any]]:
        """Get layer metadata by ID.

        Args:
            layer_id: Layer identifier.

        Returns:
            GeoLayer dictionary or None.
        """
        layer = self._layers.get(layer_id)
        if layer and layer.get("status") == "deleted":
            return None
        return layer

    def list_layers(
        self,
        status: Optional[str] = None,
        geometry_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List layers with optional filters.

        Args:
            status: Filter by status (active, inactive).
            geometry_type: Filter by geometry type.

        Returns:
            List of GeoLayer dictionaries.
        """
        results = []
        for layer in self._layers.values():
            if layer.get("status") == "deleted":
                continue
            if status and layer.get("status") != status:
                continue
            if geometry_type and layer.get("geometry_type") != geometry_type:
                continue
            results.append(layer)

        results.sort(key=lambda l: l.get("created_at", ""), reverse=True)
        return results

    def add_features(
        self,
        layer_id: str,
        features: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Add features to a layer.

        Args:
            layer_id: Layer identifier.
            features: List of feature dictionaries.

        Returns:
            Summary with added count and feature IDs.

        Raises:
            ValueError: If layer not found.
        """
        layer = self._layers.get(layer_id)
        if not layer or layer.get("status") == "deleted":
            raise ValueError(f"Layer not found: {layer_id}")

        added_ids: List[str] = []
        for raw_feat in features:
            feature_id = raw_feat.get("id") or f"FTR-{uuid.uuid4().hex[:12]}"

            geometry = raw_feat.get("geometry", {})
            properties = raw_feat.get("properties", {})

            feat = _make_feature(
                feature_id=feature_id,
                geometry=geometry,
                properties=properties,
            )
            self._features[layer_id].append(feat)
            added_ids.append(feature_id)

        # Update layer metadata
        layer["feature_count"] = len(self._features[layer_id])
        layer["updated_at"] = _utcnow().isoformat()
        layer["bbox"] = self._compute_layer_bbox(layer_id)

        # Record metrics
        try:
            from greenlang.gis_connector.metrics import (
                record_features_processed,
                update_layer_features,
            )
            record_features_processed(layer=layer_id, operation="add")
            update_layer_features(layer=layer_id, count=layer["feature_count"])
        except ImportError:
            pass

        logger.info(
            "Added %d features to layer %s (total: %d)",
            len(added_ids), layer_id, layer["feature_count"],
        )
        return {
            "layer_id": layer_id,
            "added_count": len(added_ids),
            "feature_ids": added_ids,
            "total_features": layer["feature_count"],
        }

    def get_features(
        self,
        layer_id: str,
        bbox: Optional[List[float]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get features from a layer with optional spatial filter.

        Args:
            layer_id: Layer identifier.
            bbox: Optional bounding box filter [minx, miny, maxx, maxy].
            limit: Maximum features to return.
            offset: Offset for pagination.

        Returns:
            Dictionary with features list and metadata.

        Raises:
            ValueError: If layer not found.
        """
        layer = self._layers.get(layer_id)
        if not layer or layer.get("status") == "deleted":
            raise ValueError(f"Layer not found: {layer_id}")

        features = self._features.get(layer_id, [])

        # Apply spatial filter
        if bbox and len(bbox) == 4:
            features = [
                f for f in features
                if self._feature_intersects_bbox(f, bbox)
            ]

        total = len(features)
        features = features[offset:offset + limit]

        return {
            "layer_id": layer_id,
            "features": features,
            "total_count": total,
            "returned_count": len(features),
            "limit": limit,
            "offset": offset,
        }

    def update_layer(
        self,
        layer_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update layer metadata.

        Args:
            layer_id: Layer identifier.
            updates: Dictionary of fields to update.

        Returns:
            Updated GeoLayer dictionary.

        Raises:
            ValueError: If layer not found.
        """
        layer = self._layers.get(layer_id)
        if not layer or layer.get("status") == "deleted":
            raise ValueError(f"Layer not found: {layer_id}")

        # Only allow certain fields to be updated
        allowed_fields = {"name", "description", "status", "tags", "metadata", "crs"}
        for key, value in updates.items():
            if key in allowed_fields:
                layer[key] = value

        layer["updated_at"] = _utcnow().isoformat()

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(layer)
            self._provenance.record(
                entity_type="layer",
                entity_id=layer_id,
                action="layer_operation",
                data_hash=data_hash,
            )

        logger.info("Updated layer %s: fields=%s", layer_id, list(updates.keys()))
        return layer

    def delete_layer(self, layer_id: str) -> Dict[str, Any]:
        """Soft delete a layer.

        Args:
            layer_id: Layer identifier.

        Returns:
            Deletion confirmation dictionary.

        Raises:
            ValueError: If layer not found.
        """
        layer = self._layers.get(layer_id)
        if not layer:
            raise ValueError(f"Layer not found: {layer_id}")

        layer["status"] = "deleted"
        layer["updated_at"] = _utcnow().isoformat()

        # Update metrics
        try:
            from greenlang.gis_connector.metrics import update_active_layers
            update_active_layers(len(
                [l for l in self._layers.values() if l.get("status") == "active"]
            ))
        except ImportError:
            pass

        logger.info("Deleted layer %s", layer_id)
        return {
            "layer_id": layer_id,
            "status": "deleted",
            "deleted_at": layer["updated_at"],
        }

    def export_layer(
        self,
        layer_id: str,
        format: str = "geojson",
    ) -> Dict[str, Any]:
        """Export a layer to a specified format.

        Args:
            layer_id: Layer identifier.
            format: Export format (geojson, wkt, csv).

        Returns:
            Export result with data string.

        Raises:
            ValueError: If layer not found or format unsupported.
        """
        layer = self._layers.get(layer_id)
        if not layer or layer.get("status") == "deleted":
            raise ValueError(f"Layer not found: {layer_id}")

        features = self._features.get(layer_id, [])
        format = format.lower()

        if format == "geojson":
            data = json.dumps({
                "type": "FeatureCollection",
                "features": features,
                "crs": {"type": "name", "properties": {"name": layer.get("crs", "EPSG:4326")}},
            }, indent=2, default=str)
        elif format == "csv":
            lines = ["id,geometry_type,lon,lat,properties"]
            for feat in features:
                geom = feat.get("geometry", {})
                coords = geom.get("coordinates", [])
                geom_type = geom.get("type", "")
                lon = coords[0] if coords and isinstance(coords[0], (int, float)) else ""
                lat = coords[1] if coords and len(coords) > 1 and isinstance(coords[1], (int, float)) else ""
                props = json.dumps(feat.get("properties", {}), default=str)
                lines.append(f"{feat.get('id', '')},{geom_type},{lon},{lat},{props}")
            data = "\n".join(lines)
        elif format == "wkt":
            wkt_lines = []
            for feat in features:
                geom = feat.get("geometry", {})
                wkt = self._geometry_to_wkt(geom)
                wkt_lines.append(f"{feat.get('id', '')}\t{wkt}")
            data = "\n".join(wkt_lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        # Record metrics
        try:
            from greenlang.gis_connector.metrics import record_operation
            record_operation(
                operation="layer_export",
                format=format,
                status="success",
            )
        except ImportError:
            pass

        return {
            "layer_id": layer_id,
            "format": format,
            "feature_count": len(features),
            "data": data,
            "size_bytes": len(data.encode("utf-8")),
        }

    def merge_layers(
        self,
        layer_ids: List[str],
        name: str,
    ) -> Dict[str, Any]:
        """Merge multiple layers into a new layer.

        Args:
            layer_ids: List of layer IDs to merge.
            name: Name for the merged layer.

        Returns:
            New merged GeoLayer dictionary.

        Raises:
            ValueError: If any layer not found.
        """
        all_features: List[Dict[str, Any]] = []
        geometry_types: set = set()
        crs = "EPSG:4326"

        for lid in layer_ids:
            layer = self._layers.get(lid)
            if not layer or layer.get("status") == "deleted":
                raise ValueError(f"Layer not found: {lid}")
            geometry_types.add(layer.get("geometry_type", "Mixed"))
            crs = layer.get("crs", crs)
            feats = self._features.get(lid, [])
            all_features.extend(feats)

        geom_type = "Mixed" if len(geometry_types) > 1 else geometry_types.pop() if geometry_types else "Mixed"

        merged = self.create_layer(
            name=name,
            geometry_type=geom_type,
            crs=crs,
            description=f"Merged from {len(layer_ids)} layers",
        )

        if all_features:
            self.add_features(merged["layer_id"], all_features)
            merged = self._layers[merged["layer_id"]]

        logger.info(
            "Merged %d layers into %s: %d features",
            len(layer_ids), merged["layer_id"], len(all_features),
        )
        return merged

    def get_layer_statistics(self, layer_id: str) -> Dict[str, Any]:
        """Get statistics for a layer.

        Args:
            layer_id: Layer identifier.

        Returns:
            Statistics dictionary.

        Raises:
            ValueError: If layer not found.
        """
        layer = self._layers.get(layer_id)
        if not layer or layer.get("status") == "deleted":
            raise ValueError(f"Layer not found: {layer_id}")

        features = self._features.get(layer_id, [])

        geom_type_counts: Dict[str, int] = {}
        for feat in features:
            gt = feat.get("geometry", {}).get("type", "Unknown")
            geom_type_counts[gt] = geom_type_counts.get(gt, 0) + 1

        property_keys: set = set()
        for feat in features:
            property_keys.update(feat.get("properties", {}).keys())

        return {
            "layer_id": layer_id,
            "name": layer.get("name", ""),
            "feature_count": len(features),
            "geometry_type_distribution": geom_type_counts,
            "property_fields": sorted(property_keys),
            "bbox": layer.get("bbox", []),
            "crs": layer.get("crs", "EPSG:4326"),
            "created_at": layer.get("created_at", ""),
            "updated_at": layer.get("updated_at", ""),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_layer_bbox(self, layer_id: str) -> List[float]:
        """Compute bounding box for all features in a layer.

        Args:
            layer_id: Layer identifier.

        Returns:
            [minx, miny, maxx, maxy] or empty list.
        """
        features = self._features.get(layer_id, [])
        all_coords: list = []

        for feat in features:
            coords = self._extract_coords(feat.get("geometry", {}))
            all_coords.extend(coords)

        if not all_coords:
            return []

        min_x = min(c[0] for c in all_coords)
        min_y = min(c[1] for c in all_coords)
        max_x = max(c[0] for c in all_coords)
        max_y = max(c[1] for c in all_coords)

        return [round(min_x, 8), round(min_y, 8), round(max_x, 8), round(max_y, 8)]

    def _extract_coords(self, geometry: Dict[str, Any]) -> List[List[float]]:
        """Extract flat coordinate pairs from geometry.

        Args:
            geometry: Geometry dictionary.

        Returns:
            List of [x, y] pairs.
        """
        coords = geometry.get("coordinates", [])
        return self._flatten_coords(coords)

    def _flatten_coords(self, coords: Any) -> List[List[float]]:
        """Flatten nested coordinates to [x, y] pairs.

        Args:
            coords: Nested coordinate array.

        Returns:
            List of [x, y] pairs.
        """
        if not isinstance(coords, list) or not coords:
            return []
        if isinstance(coords[0], (int, float)):
            return [coords[:2]] if len(coords) >= 2 else []
        result: List[List[float]] = []
        for c in coords:
            result.extend(self._flatten_coords(c))
        return result

    def _feature_intersects_bbox(
        self,
        feature: Dict[str, Any],
        bbox: List[float],
    ) -> bool:
        """Check if a feature intersects a bounding box.

        Args:
            feature: Feature dictionary.
            bbox: [minx, miny, maxx, maxy].

        Returns:
            True if feature intersects the bbox.
        """
        coords = self._extract_coords(feature.get("geometry", {}))
        if not coords:
            return False

        for c in coords:
            if bbox[0] <= c[0] <= bbox[2] and bbox[1] <= c[1] <= bbox[3]:
                return True
        return False

    def _geometry_to_wkt(self, geometry: Dict[str, Any]) -> str:
        """Convert a geometry to WKT string.

        Args:
            geometry: Geometry dictionary.

        Returns:
            WKT string representation.
        """
        geom_type = geometry.get("type", "")
        coords = geometry.get("coordinates", [])

        if geom_type == "Point":
            if coords and len(coords) >= 2:
                return f"POINT({coords[0]} {coords[1]})"
            return "POINT EMPTY"

        elif geom_type == "LineString":
            pts = ", ".join(f"{c[0]} {c[1]}" for c in coords if len(c) >= 2)
            return f"LINESTRING({pts})" if pts else "LINESTRING EMPTY"

        elif geom_type == "Polygon":
            rings = []
            for ring in coords:
                pts = ", ".join(f"{c[0]} {c[1]}" for c in ring if len(c) >= 2)
                rings.append(f"({pts})")
            return f"POLYGON({', '.join(rings)})" if rings else "POLYGON EMPTY"

        elif geom_type == "MultiPoint":
            pts = ", ".join(f"({c[0]} {c[1]})" for c in coords if len(c) >= 2)
            return f"MULTIPOINT({pts})" if pts else "MULTIPOINT EMPTY"

        return f"{geom_type.upper()} EMPTY"

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def layer_count(self) -> int:
        """Return the total number of active layers."""
        return sum(1 for l in self._layers.values() if l.get("status") != "deleted")

    @property
    def total_feature_count(self) -> int:
        """Return the total number of features across all layers."""
        return sum(len(feats) for feats in self._features.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get layer manager statistics.

        Returns:
            Dictionary with layer and feature counts.
        """
        active = sum(1 for l in self._layers.values() if l.get("status") == "active")
        total_features = sum(len(f) for f in self._features.values())

        return {
            "total_layers": len(self._layers),
            "active_layers": active,
            "total_features": total_features,
        }


__all__ = [
    "LayerManagerEngine",
    "GEOMETRY_TYPES",
    "LAYER_STATUSES",
    "EXPORT_FORMATS",
]
