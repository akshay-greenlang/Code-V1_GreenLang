# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack: Geolocation Verification Report Template
======================================================================

Generates a plot geolocation verification report covering validation
summaries, coordinate quality checks, polygon analysis, overlap
detection, country determination, Article 9 plot-size compliance,
cutoff date verification, and GeoJSON map output data.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

PACK_ID = "PACK-006-eudr-starter"
TEMPLATE_NAME = "geolocation_report"
TEMPLATE_VERSION = "1.0.0"

# EUDR cutoff date per Article 2
EUDR_CUTOFF_DATE = date(2020, 12, 31)


# =============================================================================
# ENUMS
# =============================================================================

class ValidationStatus(str, Enum):
    """Geolocation validation status."""
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"
    PENDING = "PENDING"


class PrecisionGrade(str, Enum):
    """Coordinate precision grade."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class PolygonValidity(str, Enum):
    """Polygon geometric validity."""
    VALID = "VALID"
    INVALID = "INVALID"
    SELF_INTERSECTING = "SELF_INTERSECTING"
    UNCLOSED = "UNCLOSED"


class DeforestationStatus(str, Enum):
    """Deforestation-free status relative to cutoff date."""
    DEFORESTATION_FREE = "DEFORESTATION_FREE"
    DEFORESTATION_DETECTED = "DEFORESTATION_DETECTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"


class ComplianceResult(str, Enum):
    """Article 9 plot size compliance."""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    WARNING = "WARNING"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ValidationSummary(BaseModel):
    """Section 1: Overall validation summary."""
    total_plots: int = Field(0, ge=0, description="Total plots")
    validated: int = Field(0, ge=0, description="Plots validated")
    failed: int = Field(0, ge=0, description="Plots failed")
    pending: int = Field(0, ge=0, description="Plots pending")
    validation_date: date = Field(
        default_factory=date.today, description="Date of validation"
    )


class CoordinateQualityEntry(BaseModel):
    """Coordinate quality check result for a single plot."""
    plot_id: str = Field(..., description="Plot identifier")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude")
    precision_meters: Optional[float] = Field(None, ge=0, description="Precision (m)")
    precision_grade: PrecisionGrade = Field(
        PrecisionGrade.MEDIUM, description="Precision grade"
    )
    wgs84_compliant: bool = Field(True, description="WGS84 CRS compliant")
    format_valid: bool = Field(True, description="Coordinate format valid")
    original_format: Optional[str] = Field(
        None, description="Original format (DD/DMS/UTM)"
    )
    normalized: bool = Field(True, description="Whether format was normalized")
    issues: List[str] = Field(default_factory=list, description="Quality issues found")


class PolygonAnalysisEntry(BaseModel):
    """Polygon analysis result."""
    plot_id: str = Field(..., description="Plot identifier")
    vertex_count: int = Field(0, ge=0, description="Number of vertices")
    area_hectares: float = Field(0.0, ge=0.0, description="Calculated area (ha)")
    perimeter_km: Optional[float] = Field(None, ge=0, description="Perimeter (km)")
    validity: PolygonValidity = Field(
        PolygonValidity.VALID, description="Geometric validity"
    )
    is_closed: bool = Field(True, description="Whether polygon is closed")
    topology_issues: List[str] = Field(
        default_factory=list, description="Topology issues"
    )


class OverlapEntry(BaseModel):
    """Overlap detection result."""
    plot_id_a: str = Field(..., description="First plot ID")
    plot_id_b: str = Field(..., description="Second plot ID")
    overlap_area_ha: float = Field(0.0, ge=0.0, description="Overlap area (ha)")
    overlap_pct: float = Field(0.0, ge=0.0, le=100.0, description="Overlap %")
    severity: str = Field("LOW", description="Overlap severity")


class CountryDetermination(BaseModel):
    """Reverse geocode country determination."""
    plot_id: str = Field(..., description="Plot identifier")
    determined_country_iso: str = Field(..., description="Determined country ISO")
    determined_country_name: str = Field(..., description="Determined country name")
    declared_country_iso: Optional[str] = Field(
        None, description="Declared country ISO"
    )
    match: bool = Field(True, description="Whether determined matches declared")
    confidence: float = Field(
        0.0, ge=0.0, le=100.0, description="Geocode confidence %"
    )


class PlotSizeCompliance(BaseModel):
    """Article 9 plot size compliance check."""
    plot_id: str = Field(..., description="Plot identifier")
    area_hectares: float = Field(0.0, ge=0.0, description="Area in hectares")
    geometry_type: str = Field(
        "POINT", description="POINT (<4ha) or POLYGON (>=4ha)"
    )
    required_geometry: str = Field(
        "POINT", description="Required geometry per Article 9"
    )
    compliant: ComplianceResult = Field(
        ComplianceResult.COMPLIANT, description="Compliance result"
    )
    issue: Optional[str] = Field(None, description="Compliance issue if any")


class CutoffDateEntry(BaseModel):
    """Deforestation cutoff date verification."""
    plot_id: str = Field(..., description="Plot identifier")
    cutoff_date: date = Field(
        default=EUDR_CUTOFF_DATE, description="EUDR cutoff date"
    )
    verification_date: Optional[date] = Field(
        None, description="Date of satellite verification"
    )
    status: DeforestationStatus = Field(
        DeforestationStatus.PENDING_VERIFICATION, description="Status"
    )
    data_source: Optional[str] = Field(
        None, description="Satellite data source"
    )
    forest_cover_baseline_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Baseline forest cover %"
    )
    forest_cover_current_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Current forest cover %"
    )
    notes: Optional[str] = Field(None, description="Verification notes")


class GeoJSONFeature(BaseModel):
    """GeoJSON feature for map data output."""
    plot_id: str = Field(..., description="Plot identifier")
    geometry_type: str = Field(..., description="Point or Polygon")
    coordinates: Any = Field(..., description="GeoJSON coordinates")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Feature properties"
    )


class MapData(BaseModel):
    """Section 8: Map data for visualization."""
    features: List[GeoJSONFeature] = Field(
        default_factory=list, description="GeoJSON features"
    )
    bounding_box: Optional[List[float]] = Field(
        None, description="[min_lon, min_lat, max_lon, max_lat]"
    )
    centroid: Optional[List[float]] = Field(
        None, description="[longitude, latitude]"
    )
    total_area_ha: float = Field(0.0, ge=0.0, description="Total area")


class GeolocationReportInput(BaseModel):
    """Complete input for the Geolocation Verification Report."""
    company_name: str = Field(..., description="Reporting entity")
    report_date: date = Field(
        default_factory=date.today, description="Report date"
    )
    validation_summary: ValidationSummary = Field(
        default_factory=ValidationSummary, description="Validation summary"
    )
    coordinate_quality: List[CoordinateQualityEntry] = Field(
        default_factory=list, description="Coordinate quality checks"
    )
    polygon_analysis: List[PolygonAnalysisEntry] = Field(
        default_factory=list, description="Polygon analysis results"
    )
    overlaps: List[OverlapEntry] = Field(
        default_factory=list, description="Overlap detections"
    )
    country_determinations: List[CountryDetermination] = Field(
        default_factory=list, description="Country determinations"
    )
    plot_size_compliance: List[PlotSizeCompliance] = Field(
        default_factory=list, description="Article 9 compliance"
    )
    cutoff_date_checks: List[CutoffDateEntry] = Field(
        default_factory=list, description="Cutoff date verifications"
    )
    map_data: MapData = Field(
        default_factory=MapData, description="Map visualization data"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _status_badge(status: ValidationStatus) -> str:
    """Text badge for validation status."""
    return f"[{status.value}]"


def _precision_badge(grade: PrecisionGrade) -> str:
    """Text badge for precision grade."""
    return f"[{grade.value}]"


def _deforestation_badge(status: DeforestationStatus) -> str:
    """Text badge for deforestation status."""
    return f"[{status.value}]"


def _compliance_badge(result: ComplianceResult) -> str:
    """Text badge for compliance result."""
    return f"[{result.value}]"


def _compliance_css(result: ComplianceResult) -> str:
    """Inline CSS for compliance result."""
    mapping = {
        ComplianceResult.COMPLIANT: "color:#1a7f37;font-weight:bold;",
        ComplianceResult.NON_COMPLIANT: "color:#cf222e;font-weight:bold;",
        ComplianceResult.WARNING: "color:#b08800;font-weight:bold;",
    }
    return mapping.get(result, "")


def _deforestation_css(status: DeforestationStatus) -> str:
    """Inline CSS for deforestation status."""
    mapping = {
        DeforestationStatus.DEFORESTATION_FREE: "color:#1a7f37;font-weight:bold;",
        DeforestationStatus.DEFORESTATION_DETECTED: "color:#cf222e;font-weight:bold;",
        DeforestationStatus.INCONCLUSIVE: "color:#b08800;font-weight:bold;",
        DeforestationStatus.PENDING_VERIFICATION: "color:#888;",
    }
    return mapping.get(status, "")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class GeolocationReport:
    """Generate plot geolocation verification report.

    Sections:
        1. Validation Summary - Total, validated, failed, pending
        2. Coordinate Quality - Precision, WGS84, format checks
        3. Polygon Analysis - Validity, area, topology
        4. Overlap Detection - Overlapping plot pairs
        5. Country Determination - Reverse geocode matching
        6. Plot Size Compliance - Article 9 (<4ha point, >=4ha polygon)
        7. Cutoff Date Status - Deforestation-free vs Dec 31, 2020
        8. Map Data - GeoJSON output, bounding box, centroid

    Example:
        >>> report = GeolocationReport()
        >>> data = GeolocationReportInput(...)
        >>> md = report.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize the Geolocation Report template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: GeolocationReportInput) -> str:
        """Render as Markdown.

        Args:
            data: Validated geolocation report input data.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_validation_summary(data),
            self._md_coordinate_quality(data),
            self._md_polygon_analysis(data),
            self._md_overlap_detection(data),
            self._md_country_determination(data),
            self._md_plot_size_compliance(data),
            self._md_cutoff_date(data),
            self._md_map_data(data),
            self._md_provenance(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: GeolocationReportInput) -> str:
        """Render as HTML.

        Args:
            data: Validated geolocation report input data.

        Returns:
            Complete HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_validation_summary(data),
            self._html_coordinate_quality(data),
            self._html_polygon_analysis(data),
            self._html_overlap_detection(data),
            self._html_country_determination(data),
            self._html_plot_size_compliance(data),
            self._html_cutoff_date(data),
            self._html_map_data(data),
            self._html_provenance(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: GeolocationReportInput) -> Dict[str, Any]:
        """Render as JSON-serializable dictionary.

        Args:
            data: Validated geolocation report input data.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance_hash = self._compute_provenance_hash(data)

        geojson_collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": f.plot_id,
                    "geometry": {
                        "type": f.geometry_type,
                        "coordinates": f.coordinates,
                    },
                    "properties": f.properties,
                }
                for f in data.map_data.features
            ],
        }

        return {
            "metadata": {
                "pack_id": PACK_ID,
                "template_name": TEMPLATE_NAME,
                "version": TEMPLATE_VERSION,
                "generated_at": self._render_timestamp.isoformat(),
                "provenance_hash": provenance_hash,
            },
            "company_name": data.company_name,
            "report_date": data.report_date.isoformat(),
            "validation_summary": data.validation_summary.model_dump(mode="json"),
            "coordinate_quality": [
                c.model_dump(mode="json") for c in data.coordinate_quality
            ],
            "polygon_analysis": [
                p.model_dump(mode="json") for p in data.polygon_analysis
            ],
            "overlaps": [o.model_dump(mode="json") for o in data.overlaps],
            "country_determinations": [
                c.model_dump(mode="json") for c in data.country_determinations
            ],
            "plot_size_compliance": [
                p.model_dump(mode="json") for p in data.plot_size_compliance
            ],
            "cutoff_date_checks": [
                c.model_dump(mode="json") for c in data.cutoff_date_checks
            ],
            "map_data": {
                "geojson": geojson_collection,
                "bounding_box": data.map_data.bounding_box,
                "centroid": data.map_data.centroid,
                "total_area_ha": data.map_data.total_area_ha,
            },
        }

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance_hash(self, data: GeolocationReportInput) -> str:
        """Compute SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: GeolocationReportInput) -> str:
        """Report header."""
        return (
            f"# Geolocation Verification Report - {data.company_name}\n"
            f"**Report Date:** {data.report_date.isoformat()}\n\n---"
        )

    def _md_validation_summary(self, data: GeolocationReportInput) -> str:
        """Section 1: Validation Summary."""
        vs = data.validation_summary
        pct = (vs.validated / vs.total_plots * 100) if vs.total_plots > 0 else 0.0
        return (
            "## 1. Validation Summary\n\n"
            "| Metric | Count |\n"
            "|--------|-------|\n"
            f"| Total Plots | {vs.total_plots} |\n"
            f"| Validated | {vs.validated} ({pct:.1f}%) |\n"
            f"| Failed | {vs.failed} |\n"
            f"| Pending | {vs.pending} |\n"
            f"| Validation Date | {vs.validation_date.isoformat()} |"
        )

    def _md_coordinate_quality(self, data: GeolocationReportInput) -> str:
        """Section 2: Coordinate Quality."""
        lines = [
            "## 2. Coordinate Quality\n",
            "| Plot ID | Lat | Lon | Precision | Grade | WGS84 | Format | Issues |",
            "|---------|-----|-----|-----------|-------|-------|--------|--------|",
        ]
        for c in data.coordinate_quality:
            precision = f"{c.precision_meters:.1f}m" if c.precision_meters is not None else "N/A"
            wgs84 = "Yes" if c.wgs84_compliant else "No"
            fmt = "Valid" if c.format_valid else "Invalid"
            issues = "; ".join(c.issues) if c.issues else "None"
            lines.append(
                f"| {c.plot_id} | {c.latitude:.6f} | {c.longitude:.6f} "
                f"| {precision} | {_precision_badge(c.precision_grade)} "
                f"| {wgs84} | {fmt} | {issues} |"
            )
        if not data.coordinate_quality:
            lines.append("| - | No coordinate data | - | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_polygon_analysis(self, data: GeolocationReportInput) -> str:
        """Section 3: Polygon Analysis."""
        lines = [
            "## 3. Polygon Analysis\n",
            "| Plot ID | Vertices | Area (ha) | Perimeter (km) | Validity | Closed | Issues |",
            "|---------|----------|-----------|----------------|----------|--------|--------|",
        ]
        for p in data.polygon_analysis:
            perimeter = f"{p.perimeter_km:.2f}" if p.perimeter_km is not None else "N/A"
            closed = "Yes" if p.is_closed else "No"
            issues = "; ".join(p.topology_issues) if p.topology_issues else "None"
            lines.append(
                f"| {p.plot_id} | {p.vertex_count} | {p.area_hectares:.2f} "
                f"| {perimeter} | [{p.validity.value}] | {closed} | {issues} |"
            )
        if not data.polygon_analysis:
            lines.append("| - | No polygon data | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_overlap_detection(self, data: GeolocationReportInput) -> str:
        """Section 4: Overlap Detection."""
        if not data.overlaps:
            return "## 4. Overlap Detection\n\nNo overlapping plots detected."
        lines = [
            "## 4. Overlap Detection\n",
            f"**{len(data.overlaps)} overlap(s) detected**\n",
            "| Plot A | Plot B | Overlap (ha) | Overlap (%) | Severity |",
            "|--------|--------|-------------|-------------|----------|",
        ]
        for o in data.overlaps:
            lines.append(
                f"| {o.plot_id_a} | {o.plot_id_b} | {o.overlap_area_ha:.3f} "
                f"| {o.overlap_pct:.1f}% | [{o.severity}] |"
            )
        return "\n".join(lines)

    def _md_country_determination(self, data: GeolocationReportInput) -> str:
        """Section 5: Country Determination."""
        lines = [
            "## 5. Country Determination\n",
            "| Plot ID | Determined | Declared | Match | Confidence |",
            "|---------|-----------|----------|-------|------------|",
        ]
        for c in data.country_determinations:
            declared = c.declared_country_iso or "N/A"
            match_str = "Yes" if c.match else "**NO**"
            lines.append(
                f"| {c.plot_id} | {c.determined_country_name} "
                f"({c.determined_country_iso}) | {declared} "
                f"| {match_str} | {c.confidence:.1f}% |"
            )
        if not data.country_determinations:
            lines.append("| - | No determinations | - | - | - |")
        return "\n".join(lines)

    def _md_plot_size_compliance(self, data: GeolocationReportInput) -> str:
        """Section 6: Plot Size Compliance (Article 9)."""
        lines = [
            "## 6. Plot Size Compliance (Article 9)\n",
            "> Plots <4ha: point coordinate required. "
            "Plots >=4ha: polygon boundary required.\n",
            "| Plot ID | Area (ha) | Geometry | Required | Compliance | Issue |",
            "|---------|-----------|----------|----------|------------|-------|",
        ]
        for p in data.plot_size_compliance:
            issue = p.issue or "None"
            lines.append(
                f"| {p.plot_id} | {p.area_hectares:.2f} | {p.geometry_type} "
                f"| {p.required_geometry} | {_compliance_badge(p.compliant)} "
                f"| {issue} |"
            )
        if not data.plot_size_compliance:
            lines.append("| - | No compliance data | - | - | - | - |")
        return "\n".join(lines)

    def _md_cutoff_date(self, data: GeolocationReportInput) -> str:
        """Section 7: Cutoff Date Status."""
        lines = [
            "## 7. Cutoff Date Status\n",
            f"> EUDR cutoff date: {EUDR_CUTOFF_DATE.isoformat()} "
            "(December 31, 2020)\n",
            "| Plot ID | Status | Verification Date | Source "
            "| Baseline (%) | Current (%) | Notes |",
            "|---------|--------|-------------------|--------"
            "|--------------|-------------|-------|",
        ]
        for c in data.cutoff_date_checks:
            ver_date = c.verification_date.isoformat() if c.verification_date else "N/A"
            source = c.data_source or "N/A"
            baseline = f"{c.forest_cover_baseline_pct:.1f}" if c.forest_cover_baseline_pct is not None else "N/A"
            current = f"{c.forest_cover_current_pct:.1f}" if c.forest_cover_current_pct is not None else "N/A"
            notes = c.notes or "N/A"
            lines.append(
                f"| {c.plot_id} | {_deforestation_badge(c.status)} "
                f"| {ver_date} | {source} | {baseline} | {current} | {notes} |"
            )
        if not data.cutoff_date_checks:
            lines.append("| - | No cutoff date data | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_map_data(self, data: GeolocationReportInput) -> str:
        """Section 8: Map Data."""
        md = data.map_data
        bbox = (
            f"[{md.bounding_box[0]:.6f}, {md.bounding_box[1]:.6f}, "
            f"{md.bounding_box[2]:.6f}, {md.bounding_box[3]:.6f}]"
            if md.bounding_box and len(md.bounding_box) == 4
            else "N/A"
        )
        centroid = (
            f"[{md.centroid[0]:.6f}, {md.centroid[1]:.6f}]"
            if md.centroid and len(md.centroid) == 2
            else "N/A"
        )
        return (
            "## 8. Map Data\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Features | {len(md.features)} |\n"
            f"| Total Area | {md.total_area_ha:,.2f} ha |\n"
            f"| Bounding Box | {bbox} |\n"
            f"| Centroid | {centroid} |\n\n"
            f"*GeoJSON FeatureCollection available in JSON output.*"
        )

    def _md_provenance(self, data: GeolocationReportInput) -> str:
        """Provenance footer."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang EUDR Starter Pack v{TEMPLATE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, data: GeolocationReportInput, body: str) -> str:
        """Wrap body in HTML document."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Geolocation Report - {data.company_name}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "max-width:1200px;color:#222;line-height:1.5;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "h1{color:#1a365d;border-bottom:3px solid #2b6cb0;padding-bottom:0.5rem;}\n"
            "h2{color:#2b6cb0;margin-top:2rem;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".status-validated{color:#1a7f37;font-weight:bold;}\n"
            ".status-failed{color:#cf222e;font-weight:bold;}\n"
            ".status-pending{color:#b08800;}\n"
            ".compliant{color:#1a7f37;font-weight:bold;}\n"
            ".non-compliant{color:#cf222e;font-weight:bold;}\n"
            ".warning{color:#b08800;font-weight:bold;}\n"
            ".deforestation-free{color:#1a7f37;font-weight:bold;}\n"
            ".deforestation-detected{color:#cf222e;font-weight:bold;}\n"
            ".note-box{background:#f0f4f8;border-left:4px solid #2b6cb0;"
            "padding:0.8rem;margin:1rem 0;}\n"
            ".provenance{font-size:0.85rem;color:#666;}\n"
            "code{background:#f5f5f5;padding:0.2rem 0.4rem;border-radius:3px;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: GeolocationReportInput) -> str:
        """HTML header."""
        return (
            '<div class="section">\n'
            f"<h1>Geolocation Verification Report &mdash; {data.company_name}</h1>\n"
            f"<p><strong>Report Date:</strong> "
            f"{data.report_date.isoformat()}</p>\n<hr>\n</div>"
        )

    def _html_validation_summary(self, data: GeolocationReportInput) -> str:
        """HTML Section 1: Validation Summary."""
        vs = data.validation_summary
        pct = (vs.validated / vs.total_plots * 100) if vs.total_plots > 0 else 0.0
        return (
            '<div class="section">\n<h2>1. Validation Summary</h2>\n'
            "<table><tbody>"
            f"<tr><th>Total Plots</th><td>{vs.total_plots}</td></tr>"
            f'<tr><th>Validated</th><td class="status-validated">'
            f"{vs.validated} ({pct:.1f}%)</td></tr>"
            f'<tr><th>Failed</th><td class="status-failed">{vs.failed}</td></tr>'
            f'<tr><th>Pending</th><td class="status-pending">{vs.pending}</td></tr>'
            f"<tr><th>Validation Date</th><td>{vs.validation_date.isoformat()}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_coordinate_quality(self, data: GeolocationReportInput) -> str:
        """HTML Section 2: Coordinate Quality."""
        rows = ""
        for c in data.coordinate_quality:
            precision = f"{c.precision_meters:.1f}m" if c.precision_meters is not None else "N/A"
            wgs84 = "Yes" if c.wgs84_compliant else '<span class="non-compliant">No</span>'
            fmt = "Valid" if c.format_valid else '<span class="non-compliant">Invalid</span>'
            issues = "; ".join(c.issues) if c.issues else "None"
            rows += (
                f"<tr><td>{c.plot_id}</td><td>{c.latitude:.6f}</td>"
                f"<td>{c.longitude:.6f}</td><td>{precision}</td>"
                f"<td>{c.precision_grade.value}</td><td>{wgs84}</td>"
                f"<td>{fmt}</td><td>{issues}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="8">No coordinate data</td></tr>'
        return (
            '<div class="section">\n<h2>2. Coordinate Quality</h2>\n'
            "<table><thead><tr><th>Plot</th><th>Lat</th><th>Lon</th>"
            "<th>Precision</th><th>Grade</th><th>WGS84</th>"
            f"<th>Format</th><th>Issues</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_polygon_analysis(self, data: GeolocationReportInput) -> str:
        """HTML Section 3: Polygon Analysis."""
        rows = ""
        for p in data.polygon_analysis:
            perimeter = f"{p.perimeter_km:.2f}" if p.perimeter_km is not None else "N/A"
            closed = "Yes" if p.is_closed else '<span class="non-compliant">No</span>'
            validity_css = "compliant" if p.validity == PolygonValidity.VALID else "non-compliant"
            issues = "; ".join(p.topology_issues) if p.topology_issues else "None"
            rows += (
                f"<tr><td>{p.plot_id}</td><td>{p.vertex_count}</td>"
                f"<td>{p.area_hectares:.2f}</td><td>{perimeter}</td>"
                f'<td class="{validity_css}">{p.validity.value}</td>'
                f"<td>{closed}</td><td>{issues}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="7">No polygon data</td></tr>'
        return (
            '<div class="section">\n<h2>3. Polygon Analysis</h2>\n'
            "<table><thead><tr><th>Plot</th><th>Vertices</th>"
            "<th>Area (ha)</th><th>Perimeter (km)</th><th>Validity</th>"
            f"<th>Closed</th><th>Issues</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_overlap_detection(self, data: GeolocationReportInput) -> str:
        """HTML Section 4: Overlap Detection."""
        if not data.overlaps:
            return (
                '<div class="section"><h2>4. Overlap Detection</h2>'
                "<p>No overlapping plots detected.</p></div>"
            )
        rows = ""
        for o in data.overlaps:
            rows += (
                f"<tr><td>{o.plot_id_a}</td><td>{o.plot_id_b}</td>"
                f"<td>{o.overlap_area_ha:.3f}</td>"
                f"<td>{o.overlap_pct:.1f}%</td><td>{o.severity}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>4. Overlap Detection</h2>\n'
            f"<p><strong>{len(data.overlaps)} overlap(s) detected</strong></p>\n"
            "<table><thead><tr><th>Plot A</th><th>Plot B</th>"
            "<th>Overlap (ha)</th><th>Overlap (%)</th>"
            f"<th>Severity</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_country_determination(self, data: GeolocationReportInput) -> str:
        """HTML Section 5: Country Determination."""
        rows = ""
        for c in data.country_determinations:
            declared = c.declared_country_iso or "N/A"
            match_css = "compliant" if c.match else "non-compliant"
            match_str = "Yes" if c.match else "NO"
            rows += (
                f"<tr><td>{c.plot_id}</td>"
                f"<td>{c.determined_country_name} ({c.determined_country_iso})</td>"
                f"<td>{declared}</td>"
                f'<td class="{match_css}">{match_str}</td>'
                f"<td>{c.confidence:.1f}%</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="5">No determinations</td></tr>'
        return (
            '<div class="section">\n<h2>5. Country Determination</h2>\n'
            "<table><thead><tr><th>Plot</th><th>Determined</th>"
            "<th>Declared</th><th>Match</th>"
            f"<th>Confidence</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_plot_size_compliance(self, data: GeolocationReportInput) -> str:
        """HTML Section 6: Plot Size Compliance."""
        rows = ""
        for p in data.plot_size_compliance:
            css = _compliance_css(p.compliant)
            issue = p.issue or "None"
            rows += (
                f"<tr><td>{p.plot_id}</td><td>{p.area_hectares:.2f}</td>"
                f"<td>{p.geometry_type}</td><td>{p.required_geometry}</td>"
                f'<td style="{css}">{p.compliant.value}</td>'
                f"<td>{issue}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="6">No compliance data</td></tr>'
        return (
            '<div class="section">\n<h2>6. Plot Size Compliance (Article 9)</h2>\n'
            '<div class="note-box">Plots &lt;4ha: point coordinate required. '
            "Plots &gt;=4ha: polygon boundary required.</div>\n"
            "<table><thead><tr><th>Plot</th><th>Area (ha)</th>"
            "<th>Geometry</th><th>Required</th><th>Compliance</th>"
            f"<th>Issue</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_cutoff_date(self, data: GeolocationReportInput) -> str:
        """HTML Section 7: Cutoff Date Status."""
        rows = ""
        for c in data.cutoff_date_checks:
            ver_date = c.verification_date.isoformat() if c.verification_date else "N/A"
            source = c.data_source or "N/A"
            baseline = f"{c.forest_cover_baseline_pct:.1f}%" if c.forest_cover_baseline_pct is not None else "N/A"
            current = f"{c.forest_cover_current_pct:.1f}%" if c.forest_cover_current_pct is not None else "N/A"
            notes = c.notes or "N/A"
            css = _deforestation_css(c.status)
            rows += (
                f"<tr><td>{c.plot_id}</td>"
                f'<td style="{css}">{c.status.value}</td>'
                f"<td>{ver_date}</td><td>{source}</td>"
                f"<td>{baseline}</td><td>{current}</td><td>{notes}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="7">No cutoff date data</td></tr>'
        return (
            '<div class="section">\n<h2>7. Cutoff Date Status</h2>\n'
            f'<div class="note-box">EUDR cutoff date: '
            f"{EUDR_CUTOFF_DATE.isoformat()} (December 31, 2020)</div>\n"
            "<table><thead><tr><th>Plot</th><th>Status</th>"
            "<th>Verification Date</th><th>Source</th>"
            "<th>Baseline (%)</th><th>Current (%)</th>"
            f"<th>Notes</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_map_data(self, data: GeolocationReportInput) -> str:
        """HTML Section 8: Map Data."""
        md = data.map_data
        bbox = (
            f"[{md.bounding_box[0]:.6f}, {md.bounding_box[1]:.6f}, "
            f"{md.bounding_box[2]:.6f}, {md.bounding_box[3]:.6f}]"
            if md.bounding_box and len(md.bounding_box) == 4
            else "N/A"
        )
        centroid = (
            f"[{md.centroid[0]:.6f}, {md.centroid[1]:.6f}]"
            if md.centroid and len(md.centroid) == 2
            else "N/A"
        )
        return (
            '<div class="section">\n<h2>8. Map Data</h2>\n'
            "<table><tbody>"
            f"<tr><th>Features</th><td>{len(md.features)}</td></tr>"
            f"<tr><th>Total Area</th><td>{md.total_area_ha:,.2f} ha</td></tr>"
            f"<tr><th>Bounding Box</th><td><code>{bbox}</code></td></tr>"
            f"<tr><th>Centroid</th><td><code>{centroid}</code></td></tr>"
            "</tbody></table>\n"
            "<p><em>GeoJSON FeatureCollection available in JSON output.</em></p>\n"
            "</div>"
        )

    def _html_provenance(self, data: GeolocationReportInput) -> str:
        """HTML provenance footer."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section provenance">\n<hr>\n'
            f"<p>Generated by GreenLang EUDR Starter Pack v{TEMPLATE_VERSION} "
            f"| {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
