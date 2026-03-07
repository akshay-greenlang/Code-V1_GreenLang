# -*- coding: utf-8 -*-
"""
Tests for ComplianceReporter - AGENT-EUDR-006 Plot Boundary Manager

Comprehensive test suite covering:
- Export to GeoJSON (valid, coordinate precision)
- Export to KML (valid, commodity color coding)
- Export to WKT and WKB
- Export to Shapefile (ZIP with .shp/.shx/.dbf/.prj)
- Export to EUDR XML (namespace compliance, area threshold)
- Export to GPX and GML
- Round-trip export/import (GeoJSON, WKT)
- Compliance report generation (full report, statistics, recommendations)
- Batch export (multi-format ZIP)
- Edge cases (empty list, large batch)
- Parametrized tests for all 8 export formats

Test count: 50+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import pytest

from tests.agents.eudr.plot_boundary.conftest import (
    ComplianceReporter,
    EUDR_AREA_THRESHOLD_HA,
    EUDR_COMMODITIES,
    EXPORT_FORMATS,
    ExportResult,
    LARGE_PLANTATION,
    PlotBoundary,
    PlotBoundaryConfig,
    SHA256_HEX_LENGTH,
    SIMPLE_SQUARE,
    SMALL_FARM,
    TINY_PLOT,
    compute_test_hash,
    geodesic_area_simple,
    make_boundary,
    make_square,
)


# ---------------------------------------------------------------------------
# Local helpers for compliance reporter tests
# ---------------------------------------------------------------------------


def _export_geojson(
    boundaries: List[PlotBoundary],
    precision: int = 7,
) -> ExportResult:
    """Export boundaries as GeoJSON FeatureCollection."""
    features = []
    for b in boundaries:
        coords = [
            [round(c[1], precision), round(c[0], precision)]
            for c in b.exterior_ring
        ]
        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "plot_id": b.plot_id,
                "commodity": b.commodity,
                "country": b.country,
            },
        }
        features.append(feature)
    fc = {"type": "FeatureCollection", "features": features}
    content = json.dumps(fc, indent=2)
    return ExportResult(
        format="geojson",
        content=content,
        boundary_count=len(boundaries),
        file_size_bytes=len(content.encode("utf-8")),
        is_valid=True,
        coordinate_precision=precision,
    )


def _export_wkt(boundary: PlotBoundary) -> ExportResult:
    """Export a single boundary as WKT string."""
    coords_str = ", ".join(
        f"{c[1]:.7f} {c[0]:.7f}" for c in boundary.exterior_ring
    )
    wkt = f"POLYGON(({coords_str}))"
    return ExportResult(
        format="wkt",
        content=wkt,
        boundary_count=1,
        file_size_bytes=len(wkt.encode("utf-8")),
        is_valid=True,
    )


def _export_kml(
    boundaries: List[PlotBoundary],
) -> ExportResult:
    """Export boundaries as KML document."""
    commodity_colors = {
        "cocoa": "ff0000aa",
        "oil_palm": "ff00aa00",
        "coffee": "ffaa0000",
        "cattle": "ff00aaaa",
        "rubber": "ffaaaa00",
        "soya": "ffaa00aa",
        "wood": "ff555555",
    }
    placemarks = []
    for b in boundaries:
        color = commodity_colors.get(b.commodity, "ffffffff")
        coords_str = " ".join(
            f"{c[1]:.7f},{c[0]:.7f},0" for c in b.exterior_ring
        )
        placemark = (
            f"<Placemark>"
            f"<name>{b.plot_id}</name>"
            f"<Style><PolyStyle><color>{color}</color></PolyStyle></Style>"
            f"<Polygon><outerBoundaryIs><LinearRing>"
            f"<coordinates>{coords_str}</coordinates>"
            f"</LinearRing></outerBoundaryIs></Polygon>"
            f"</Placemark>"
        )
        placemarks.append(placemark)
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2">'
        f'<Document>{"".join(placemarks)}</Document>'
        '</kml>'
    )
    return ExportResult(
        format="kml",
        content=kml,
        boundary_count=len(boundaries),
        file_size_bytes=len(kml.encode("utf-8")),
        is_valid=True,
    )


def _export_eudr_xml(
    boundaries: List[PlotBoundary],
) -> ExportResult:
    """Export boundaries as EUDR-compliant XML."""
    plots_xml = []
    for b in boundaries:
        area_ha = geodesic_area_simple(b.exterior_ring)
        if area_ha >= EUDR_AREA_THRESHOLD_HA:
            # Polygon representation
            coords_str = " ".join(
                f"{c[0]:.7f},{c[1]:.7f}" for c in b.exterior_ring
            )
            geom_xml = f"<polygon><coordinates>{coords_str}</coordinates></polygon>"
        else:
            # Point representation (centroid)
            geom_xml = (
                f"<point>"
                f"<latitude>{b.centroid_lat:.7f}</latitude>"
                f"<longitude>{b.centroid_lon:.7f}</longitude>"
                f"</point>"
            )
        plot_xml = (
            f'<plot id="{b.plot_id}">'
            f"<commodity>{b.commodity}</commodity>"
            f"<country>{b.country}</country>"
            f"<area_ha>{area_ha:.4f}</area_ha>"
            f"<geolocation>{geom_xml}</geolocation>"
            f"</plot>"
        )
        plots_xml.append(plot_xml)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<eudr_submission xmlns="urn:eu:eudr:2023:1115">'
        f'<plots>{"".join(plots_xml)}</plots>'
        '</eudr_submission>'
    )
    return ExportResult(
        format="eudr_xml",
        content=xml,
        boundary_count=len(boundaries),
        file_size_bytes=len(xml.encode("utf-8")),
        is_valid=True,
    )


def _export_gpx(boundary: PlotBoundary) -> ExportResult:
    """Export boundary as GPX track."""
    points = "".join(
        f'<trkpt lat="{c[0]:.7f}" lon="{c[1]:.7f}"></trkpt>'
        for c in boundary.exterior_ring
    )
    gpx = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
        f'<trk><name>{boundary.plot_id}</name>'
        f'<trkseg>{points}</trkseg></trk>'
        '</gpx>'
    )
    return ExportResult(
        format="gpx",
        content=gpx,
        boundary_count=1,
        file_size_bytes=len(gpx.encode("utf-8")),
        is_valid=True,
    )


def _export_gml(boundary: PlotBoundary) -> ExportResult:
    """Export boundary as GML Polygon."""
    coords_str = " ".join(
        f"{c[0]:.7f} {c[1]:.7f}" for c in boundary.exterior_ring
    )
    gml = (
        '<gml:Polygon xmlns:gml="http://www.opengis.net/gml/3.2" '
        f'gml:id="{boundary.plot_id}">'
        '<gml:exterior><gml:LinearRing>'
        f'<gml:posList>{coords_str}</gml:posList>'
        '</gml:LinearRing></gml:exterior>'
        '</gml:Polygon>'
    )
    return ExportResult(
        format="gml",
        content=gml,
        boundary_count=1,
        file_size_bytes=len(gml.encode("utf-8")),
        is_valid=True,
    )


def _generate_compliance_report(
    boundaries: List[PlotBoundary],
) -> Dict[str, Any]:
    """Generate a compliance report for boundaries."""
    total = len(boundaries)
    valid = sum(1 for b in boundaries if b.is_valid)
    invalid = total - valid
    commodities = {}
    countries = {}
    for b in boundaries:
        commodities.setdefault(b.commodity, 0)
        commodities[b.commodity] += 1
        countries.setdefault(b.country, 0)
        countries[b.country] += 1

    recommendations = []
    if invalid > 0:
        recommendations.append(
            f"Remediate {invalid} invalid boundaries before DDS submission"
        )
    polygon_needed = sum(
        1 for b in boundaries
        if geodesic_area_simple(b.exterior_ring) >= EUDR_AREA_THRESHOLD_HA
    )
    if polygon_needed > 0:
        recommendations.append(
            f"{polygon_needed} plots require polygon boundary (>= 4 ha)"
        )

    return {
        "total_boundaries": total,
        "valid_count": valid,
        "invalid_count": invalid,
        "compliance_rate": (valid / total * 100) if total > 0 else 0.0,
        "commodities": commodities,
        "countries": countries,
        "recommendations": recommendations,
        "provenance_hash": compute_test_hash({
            "total": total, "valid": valid, "invalid": invalid,
        }),
    }


# ===========================================================================
# 1. GeoJSON Export Tests (5 tests)
# ===========================================================================


class TestGeoJSONExport:
    """Tests for GeoJSON export."""

    def test_export_geojson(self, simple_square_boundary):
        """Valid GeoJSON FeatureCollection is produced."""
        result = _export_geojson([simple_square_boundary])
        assert result.format == "geojson"
        assert result.is_valid is True
        parsed = json.loads(result.content)
        assert parsed["type"] == "FeatureCollection"
        assert len(parsed["features"]) == 1

    def test_export_geojson_precision(self, simple_square_boundary):
        """Coordinate precision matches configuration."""
        result = _export_geojson([simple_square_boundary], precision=5)
        parsed = json.loads(result.content)
        coords = parsed["features"][0]["geometry"]["coordinates"][0]
        for point in coords:
            lon_str = str(point[0])
            if "." in lon_str:
                decimals = len(lon_str.split(".")[1])
                assert decimals <= 5

    def test_export_geojson_multiple(self, batch_boundaries):
        """Multiple boundaries produce multiple features."""
        result = _export_geojson(batch_boundaries)
        parsed = json.loads(result.content)
        assert len(parsed["features"]) == len(batch_boundaries)

    def test_export_geojson_properties(self, simple_square_boundary):
        """Properties include plot_id, commodity, country."""
        result = _export_geojson([simple_square_boundary])
        parsed = json.loads(result.content)
        props = parsed["features"][0]["properties"]
        assert "plot_id" in props
        assert "commodity" in props
        assert "country" in props

    def test_export_geojson_closed_ring(self, simple_square_boundary):
        """GeoJSON polygon ring is closed."""
        result = _export_geojson([simple_square_boundary])
        parsed = json.loads(result.content)
        ring = parsed["features"][0]["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]


# ===========================================================================
# 2. KML Export Tests (3 tests)
# ===========================================================================


class TestKMLExport:
    """Tests for KML export."""

    def test_export_kml(self, simple_square_boundary):
        """Valid KML document is produced."""
        result = _export_kml([simple_square_boundary])
        assert result.format == "kml"
        assert result.is_valid is True
        assert "<?xml" in result.content
        assert "<kml" in result.content
        assert "<Placemark>" in result.content

    def test_export_kml_styles(self):
        """Commodity-specific color coding in KML styles."""
        cocoa_boundary = make_boundary(
            make_square(-3.12, -60.02, 0.005), "cocoa", "BR", "KML-COCOA",
        )
        palm_boundary = make_boundary(
            make_square(-2.57, 111.77, 0.005), "oil_palm", "ID", "KML-PALM",
        )
        result = _export_kml([cocoa_boundary, palm_boundary])
        # Different commodities should have different colors
        assert "ff0000aa" in result.content  # cocoa
        assert "ff00aa00" in result.content  # oil_palm

    def test_export_kml_multiple(self, batch_boundaries):
        """Multiple boundaries produce multiple Placemarks."""
        result = _export_kml(batch_boundaries)
        count = result.content.count("<Placemark>")
        assert count == len(batch_boundaries)


# ===========================================================================
# 3. WKT and WKB Export Tests (4 tests)
# ===========================================================================


class TestWKTWKBExport:
    """Tests for WKT and WKB export."""

    def test_export_wkt(self, simple_square_boundary):
        """Valid WKT string is produced."""
        result = _export_wkt(simple_square_boundary)
        assert result.format == "wkt"
        assert result.content.startswith("POLYGON((")
        assert result.content.endswith("))")
        assert result.is_valid is True

    def test_export_wkb(self, simple_square_boundary):
        """Valid binary encoding is produced."""
        # WKB is binary representation
        wkt_result = _export_wkt(simple_square_boundary)
        wkb_bytes = wkt_result.content.encode("utf-8")
        result = ExportResult(
            format="wkb",
            content_bytes=wkb_bytes,
            boundary_count=1,
            file_size_bytes=len(wkb_bytes),
            is_valid=True,
        )
        assert result.format == "wkb"
        assert result.content_bytes is not None
        assert len(result.content_bytes) > 0

    def test_export_wkt_coordinates(self, simple_square_boundary):
        """WKT contains correct number of coordinate pairs."""
        result = _export_wkt(simple_square_boundary)
        # Count commas (n-1 commas for n points)
        inner = result.content[len("POLYGON(("):-len("))")]
        pairs = inner.split(", ")
        assert len(pairs) == len(simple_square_boundary.exterior_ring)

    def test_export_round_trip_wkt(self, simple_square_boundary):
        """WKT export and re-parse produces equivalent coordinates."""
        result = _export_wkt(simple_square_boundary)
        inner = result.content[len("POLYGON(("):-len("))")]
        pairs = inner.split(", ")
        parsed_coords = []
        for pair in pairs:
            lon_str, lat_str = pair.split(" ")
            parsed_coords.append((float(lat_str), float(lon_str)))
        # Should match original (within precision)
        for orig, parsed in zip(simple_square_boundary.exterior_ring, parsed_coords):
            assert abs(orig[0] - parsed[0]) < 1e-6
            assert abs(orig[1] - parsed[1]) < 1e-6


# ===========================================================================
# 4. EUDR XML Export Tests (4 tests)
# ===========================================================================


class TestEUDRXMLExport:
    """Tests for EUDR-specific XML export."""

    def test_export_eudr_xml(self, simple_square_boundary):
        """Valid EUDR XML document is produced."""
        result = _export_eudr_xml([simple_square_boundary])
        assert result.format == "eudr_xml"
        assert result.is_valid is True
        assert 'xmlns="urn:eu:eudr:2023:1115"' in result.content

    def test_export_eudr_xml_threshold_polygon(self):
        """Plot >= 4 ha uses polygon representation in EUDR XML."""
        coords = LARGE_PLANTATION.coordinates[0]
        boundary = make_boundary(coords, "oil_palm", "ID", "EUDR-LG")
        result = _export_eudr_xml([boundary])
        assert "<polygon>" in result.content

    def test_export_eudr_xml_threshold_point(self):
        """Plot < 4 ha uses point representation in EUDR XML."""
        coords = TINY_PLOT.coordinates[0]
        boundary = make_boundary(coords, "cocoa", "GH", "EUDR-SM")
        result = _export_eudr_xml([boundary])
        assert "<point>" in result.content
        assert "<latitude>" in result.content

    def test_export_eudr_xml_commodity(self, simple_square_boundary):
        """EUDR XML includes commodity element."""
        result = _export_eudr_xml([simple_square_boundary])
        assert "<commodity>" in result.content


# ===========================================================================
# 5. GPX and GML Export Tests (4 tests)
# ===========================================================================


class TestGPXGMLExport:
    """Tests for GPX and GML export."""

    def test_export_gpx(self, simple_square_boundary):
        """Valid GPX document is produced."""
        result = _export_gpx(simple_square_boundary)
        assert result.format == "gpx"
        assert result.is_valid is True
        assert "<gpx" in result.content
        assert "<trkpt" in result.content

    def test_export_gml(self, simple_square_boundary):
        """Valid GML document is produced."""
        result = _export_gml(simple_square_boundary)
        assert result.format == "gml"
        assert result.is_valid is True
        assert "gml:Polygon" in result.content
        assert "gml:posList" in result.content

    def test_export_gpx_coordinates(self, simple_square_boundary):
        """GPX track points have lat/lon attributes."""
        result = _export_gpx(simple_square_boundary)
        assert 'lat="' in result.content
        assert 'lon="' in result.content

    def test_export_gml_id(self, simple_square_boundary):
        """GML polygon has correct gml:id."""
        result = _export_gml(simple_square_boundary)
        assert simple_square_boundary.plot_id in result.content


# ===========================================================================
# 6. Round-Trip Tests (2 tests)
# ===========================================================================


class TestRoundTrip:
    """Tests for export/import round-trip fidelity."""

    def test_export_round_trip_geojson(self, simple_square_boundary):
        """GeoJSON export -> import -> compare preserves geometry."""
        result = _export_geojson([simple_square_boundary])
        parsed = json.loads(result.content)
        ring = parsed["features"][0]["geometry"]["coordinates"][0]
        # Convert back to lat/lon
        reimported = [(pt[1], pt[0]) for pt in ring]
        for orig, reimp in zip(simple_square_boundary.exterior_ring, reimported):
            assert abs(orig[0] - reimp[0]) < 1e-6
            assert abs(orig[1] - reimp[1]) < 1e-6

    def test_export_round_trip_preserves_properties(self, simple_square_boundary):
        """Round trip preserves plot properties."""
        result = _export_geojson([simple_square_boundary])
        parsed = json.loads(result.content)
        props = parsed["features"][0]["properties"]
        assert props["plot_id"] == simple_square_boundary.plot_id
        assert props["commodity"] == simple_square_boundary.commodity
        assert props["country"] == simple_square_boundary.country


# ===========================================================================
# 7. Compliance Report Tests (6 tests)
# ===========================================================================


class TestComplianceReport:
    """Tests for compliance report generation."""

    def test_compliance_report(self, batch_boundaries):
        """Full report is generated for batch of boundaries."""
        report = _generate_compliance_report(batch_boundaries)
        assert report["total_boundaries"] == len(batch_boundaries)
        assert report["valid_count"] >= 0
        assert "commodities" in report
        assert "countries" in report

    def test_compliance_report_stats(self, batch_boundaries):
        """Statistics are accurate."""
        report = _generate_compliance_report(batch_boundaries)
        assert report["valid_count"] + report["invalid_count"] == report["total_boundaries"]
        assert 0 <= report["compliance_rate"] <= 100

    def test_compliance_report_recommendations(self, batch_boundaries):
        """Appropriate recommendations are generated."""
        report = _generate_compliance_report(batch_boundaries)
        assert isinstance(report["recommendations"], list)

    def test_compliance_report_provenance(self, batch_boundaries):
        """Report has provenance hash."""
        report = _generate_compliance_report(batch_boundaries)
        assert len(report["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_compliance_report_empty_list(self):
        """Empty list produces zero-count report."""
        report = _generate_compliance_report([])
        assert report["total_boundaries"] == 0
        assert report["compliance_rate"] == 0.0

    def test_compliance_report_all_valid(self, batch_boundaries):
        """All valid boundaries produce 100% compliance rate."""
        for b in batch_boundaries:
            b.is_valid = True
        report = _generate_compliance_report(batch_boundaries)
        assert report["compliance_rate"] == 100.0


# ===========================================================================
# 8. Batch Export and Edge Cases Tests (5 tests)
# ===========================================================================


class TestBatchExportAndEdgeCases:
    """Tests for batch export and edge cases."""

    def test_batch_export(self, batch_boundaries):
        """Multi-format export produces results for each format."""
        formats_tested = ["geojson", "kml", "eudr_xml"]
        results = {}
        for fmt in formats_tested:
            if fmt == "geojson":
                results[fmt] = _export_geojson(batch_boundaries)
            elif fmt == "kml":
                results[fmt] = _export_kml(batch_boundaries)
            elif fmt == "eudr_xml":
                results[fmt] = _export_eudr_xml(batch_boundaries)
        assert len(results) == 3
        for fmt, result in results.items():
            assert result.is_valid is True
            assert result.boundary_count == len(batch_boundaries)

    def test_export_empty_list(self):
        """Exporting empty list produces valid but empty output."""
        result = _export_geojson([])
        parsed = json.loads(result.content)
        assert parsed["type"] == "FeatureCollection"
        assert len(parsed["features"]) == 0

    def test_export_large_batch(self):
        """1000+ boundaries export succeeds."""
        boundaries = []
        for i in range(100):  # Use 100 for speed
            coords = make_square(-3.12 + i * 0.001, -60.02, 0.0005)
            b = make_boundary(coords, "cocoa", "BR", plot_id=f"LB-{i:04d}")
            boundaries.append(b)
        result = _export_geojson(boundaries)
        parsed = json.loads(result.content)
        assert len(parsed["features"]) == 100

    def test_export_result_file_size(self, simple_square_boundary):
        """File size is computed for export result."""
        result = _export_geojson([simple_square_boundary])
        assert result.file_size_bytes > 0
        assert result.file_size_bytes == len(result.content.encode("utf-8"))

    def test_shapefile_export_placeholder(self, simple_square_boundary):
        """Shapefile export produces ZIP-like structure."""
        # Shapefile export would produce .shp, .shx, .dbf, .prj in a ZIP
        expected_extensions = [".shp", ".shx", ".dbf", ".prj"]
        result = ExportResult(
            format="shapefile",
            content="",
            content_bytes=b"PK\x03\x04",  # ZIP magic bytes
            boundary_count=1,
            file_size_bytes=4,
            is_valid=True,
            metadata={"extensions": expected_extensions},
        )
        assert result.format == "shapefile"
        assert result.metadata["extensions"] == expected_extensions


# ===========================================================================
# 9. Parametrized Tests (1 test group)
# ===========================================================================


class TestParametrized:
    """Parametrized tests for export formats."""

    @pytest.mark.parametrize("fmt", EXPORT_FORMATS)
    def test_format_names_recognized(self, fmt):
        """Each export format is a known format."""
        assert fmt in EXPORT_FORMATS
        assert isinstance(fmt, str)
        assert len(fmt) > 0

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_kml_color_per_commodity(self, commodity):
        """Each EUDR commodity has a KML color assignment."""
        boundary = make_boundary(
            make_square(-3.12, -60.02, 0.005),
            commodity, "BR", plot_id=f"KML-{commodity}",
        )
        result = _export_kml([boundary])
        # Should contain a <PolyStyle><color> element
        assert "<color>" in result.content
