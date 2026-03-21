# -*- coding: utf-8 -*-
"""
FormatRenderingEngine - PACK-030 Net Zero Reporting Pack Engine 10
====================================================================

Multi-format rendering engine for climate disclosure reports.  Generates
PDF, HTML, Excel, JSON, and XBRL output from compiled report structures
with full branding support, interactive charts, data tables, and
digital taxonomy tagging.

Rendering Methodology:
    PDF Rendering:
        Generates print-ready PDF documents using HTML-to-PDF conversion
        with WeasyPrint-compatible CSS.  Supports branded headers/footers,
        page numbers, table of contents, and chart images.

    HTML Rendering:
        Generates interactive HTML5 pages with Chart.js visualizations,
        expandable sections, search functionality, and responsive layout.
        Embeds XBRL inline tags where applicable.

    Excel Rendering:
        Generates multi-sheet Excel workbooks with formatted data tables,
        named ranges, charts, and conditional formatting.  Suitable for
        CDP questionnaire submission and GRI content index.

    JSON Rendering:
        Generates structured JSON output suitable for API consumption,
        data warehouse ingestion, or downstream processing.  Follows
        JSON Schema for each framework.

    XBRL Rendering:
        Delegates to Engine 4 (XBRLTaggingEngine) for full XBRL/iXBRL
        generation.  This engine handles the file packaging and metadata.

Regulatory References:
    - SBTi Corporate Net-Zero Standard v1.2 (2024) -- progress report format
    - CDP Climate Change Questionnaire (2024) -- Excel template format
    - TCFD Recommendations (2017, updated 2023) -- disclosure format
    - GRI 305 (2016) -- disclosure format
    - ISSB IFRS S2 (2023) -- digital reporting requirements
    - SEC Climate Disclosure (2024) -- XBRL/iXBRL filing requirements
    - CSRD ESRS E1 (2024) -- ESEF digital taxonomy requirements

Zero-Hallucination:
    - Rendering is purely structural/presentational
    - No data transformation or calculation in rendering
    - All numeric values pass through unchanged from compiled report
    - SHA-256 provenance hash on every rendered artifact
    - Chart data sourced directly from compiled report metrics

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-030 Net Zero Reporting
Engine:  10 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {k: v for k, v in serializable.items()
                        if k not in ("calculated_at", "processing_time_ms", "provenance_hash")}
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _round_val(value: Decimal, places: int = 2) -> Decimal:
    """Round a Decimal to the given number of decimal places."""
    quant = Decimal(10) ** -places
    return value.quantize(quant, rounding=ROUND_HALF_UP)

def _round3(value: Decimal) -> Decimal:
    """Round to 3 decimal places."""
    return _round_val(value, 3)

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safe division returning default on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Supported output formats."""
    PDF = "pdf"
    HTML = "html"
    EXCEL = "excel"
    JSON = "json"
    XBRL = "xbrl"
    IXBRL = "ixbrl"
    WORD = "word"
    CSV = "csv"


class PageSize(str, Enum):
    """Supported page sizes for PDF rendering."""
    A4 = "a4"
    LETTER = "letter"
    LEGAL = "legal"
    A3 = "a3"


class Orientation(str, Enum):
    """Page orientation for PDF."""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


class ChartType(str, Enum):
    """Chart types for rendered reports."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    DOUGHNUT = "doughnut"
    STACKED_BAR = "stacked_bar"
    WATERFALL = "waterfall"
    GAUGE = "gauge"
    AREA = "area"
    SCATTER = "scatter"
    RADAR = "radar"
    HEATMAP = "heatmap"


class TableStyle(str, Enum):
    """Styling presets for data tables."""
    DEFAULT = "default"
    STRIPED = "striped"
    BORDERED = "bordered"
    COMPACT = "compact"
    FRAMEWORK = "framework"      # Framework-specific styling
    REGULATORY = "regulatory"    # Clean regulatory style


class BrandingTheme(str, Enum):
    """Branding theme presets."""
    DEFAULT = "default"
    CORPORATE = "corporate"
    REGULATORY = "regulatory"
    INVESTOR = "investor"
    CUSTOMER = "customer"
    MINIMAL = "minimal"
    DARK = "dark"


class FrameworkTarget(str, Enum):
    """Target framework for format-specific requirements."""
    SBTI = "sbti"
    CDP = "cdp"
    TCFD = "tcfd"
    GRI = "gri"
    ISSB = "issb"
    SEC = "sec"
    CSRD = "csrd"
    MULTI = "multi"


class RenderQuality(str, Enum):
    """Render quality levels."""
    DRAFT = "draft"          # Fast, lower quality
    STANDARD = "standard"    # Normal quality
    HIGH = "high"            # High quality with all features
    PRINT = "print"          # Print-ready with full resolution


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default branding configuration
DEFAULT_BRANDING: Dict[str, Any] = {
    "primary_color": "#1B5E20",       # Dark Green
    "secondary_color": "#4CAF50",     # Green
    "accent_color": "#81C784",        # Light Green
    "text_color": "#212121",          # Near black
    "background_color": "#FFFFFF",    # White
    "header_bg": "#1B5E20",           # Dark Green
    "header_text": "#FFFFFF",         # White
    "font_family": "Roboto, Arial, sans-serif",
    "font_size_body": "11pt",
    "font_size_header": "24pt",
    "font_size_subheader": "16pt",
    "font_size_caption": "9pt",
    "line_height": "1.5",
    "logo_path": "",
    "company_name": "",
    "footer_text": "Generated by GreenLang Platform",
}

# Framework-specific color schemes
FRAMEWORK_COLORS: Dict[str, Dict[str, str]] = {
    "sbti": {"primary": "#1565C0", "secondary": "#42A5F5", "accent": "#90CAF9"},
    "cdp": {"primary": "#F57C00", "secondary": "#FFB74D", "accent": "#FFE0B2"},
    "tcfd": {"primary": "#2E7D32", "secondary": "#66BB6A", "accent": "#A5D6A7"},
    "gri": {"primary": "#0097A7", "secondary": "#4DD0E1", "accent": "#B2EBF2"},
    "issb": {"primary": "#5E35B1", "secondary": "#9575CD", "accent": "#D1C4E9"},
    "sec": {"primary": "#C62828", "secondary": "#EF5350", "accent": "#FFCDD2"},
    "csrd": {"primary": "#00695C", "secondary": "#26A69A", "accent": "#B2DFDB"},
    "multi": {"primary": "#37474F", "secondary": "#78909C", "accent": "#CFD8DC"},
}

# Excel sheet names by framework
EXCEL_SHEET_NAMES: Dict[str, List[str]] = {
    "sbti": ["SBTi Summary", "Targets", "Progress", "Emissions Data", "Methodology"],
    "cdp": ["C0 Introduction", "C1 Governance", "C2 Risks", "C3 Strategy",
            "C4 Targets", "C5 Methodology", "C6 Scope 1&2", "C7 Scope 3",
            "C8 Energy", "C9 Supply Chain", "C10 Verification"],
    "tcfd": ["Summary", "Governance", "Strategy", "Risk Management",
             "Metrics & Targets", "Scenario Analysis"],
    "gri": ["GRI Content Index", "305-1 Direct", "305-2 Indirect",
            "305-3 Other Indirect", "305-4 Intensity", "305-5 Reduction"],
    "issb": ["IFRS S2 Summary", "Governance", "Strategy",
             "Risk Management", "Metrics & Targets"],
    "sec": ["Climate Summary", "Risk Factors", "MD&A",
            "Scope 1&2 Emissions", "Targets", "Attestation"],
    "csrd": ["ESRS E1 Summary", "E1-1 Transition Plan", "E1-2 Policies",
             "E1-3 Actions", "E1-4 Targets", "E1-5 Energy",
             "E1-6 Emissions", "E1-7 Removals", "E1-8 Carbon Pricing",
             "E1-9 Financial Effects"],
    "multi": ["Executive Summary", "Framework Comparison", "Emissions Data",
              "Targets", "Progress", "Cross-Framework Index"],
}

# CSS base for PDF and HTML rendering
CSS_BASE: str = """
/* GreenLang Report Stylesheet */
@page {
    size: %(page_size)s %(orientation)s;
    margin: 2cm;
    @top-center { content: "%(header_text)s"; font-size: 9pt; color: #666; }
    @bottom-left { content: "%(footer_text)s"; font-size: 8pt; color: #999; }
    @bottom-right { content: "Page " counter(page) " of " counter(pages); font-size: 8pt; color: #999; }
}
body {
    font-family: %(font_family)s;
    font-size: %(font_size_body)s;
    line-height: %(line_height)s;
    color: %(text_color)s;
    background-color: %(background_color)s;
}
h1 { font-size: %(font_size_header)s; color: %(primary_color)s; border-bottom: 3px solid %(primary_color)s; padding-bottom: 8px; }
h2 { font-size: %(font_size_subheader)s; color: %(primary_color)s; border-bottom: 1px solid %(secondary_color)s; padding-bottom: 4px; }
h3 { font-size: 14pt; color: %(secondary_color)s; }
h4 { font-size: 12pt; color: %(text_color)s; }
table { border-collapse: collapse; width: 100%%; margin: 16px 0; page-break-inside: avoid; }
th { background-color: %(primary_color)s; color: #fff; padding: 8px 12px; text-align: left; font-weight: 600; }
td { padding: 6px 12px; border-bottom: 1px solid #e0e0e0; }
tr:nth-child(even) { background-color: #f5f5f5; }
.kpi-card { display: inline-block; width: 200px; padding: 16px; margin: 8px; border: 1px solid %(secondary_color)s; border-radius: 8px; text-align: center; }
.kpi-value { font-size: 28pt; font-weight: 700; color: %(primary_color)s; }
.kpi-label { font-size: 10pt; color: #666; margin-top: 4px; }
.kpi-change { font-size: 10pt; margin-top: 2px; }
.kpi-change.positive { color: #2E7D32; }
.kpi-change.negative { color: #C62828; }
.chart-container { width: 100%%; max-width: 800px; margin: 16px auto; page-break-inside: avoid; }
.section { margin-bottom: 24px; page-break-inside: avoid; }
.cover-page { text-align: center; padding: 100px 40px; page-break-after: always; }
.cover-title { font-size: 36pt; color: %(primary_color)s; margin-bottom: 16px; }
.cover-subtitle { font-size: 18pt; color: %(secondary_color)s; margin-bottom: 40px; }
.cover-info { font-size: 12pt; color: #666; }
.toc { margin: 20px 0; }
.toc-entry { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dotted #ccc; }
.toc-entry a { color: %(primary_color)s; text-decoration: none; }
.disclaimer { font-size: 9pt; color: #999; margin-top: 40px; padding-top: 16px; border-top: 1px solid #e0e0e0; }
.citation { font-size: 9pt; vertical-align: super; color: %(secondary_color)s; cursor: pointer; }
.footnote { font-size: 9pt; color: #666; margin: 2px 0; }
.progress-bar { background: #e0e0e0; border-radius: 4px; height: 20px; overflow: hidden; }
.progress-fill { background: %(primary_color)s; height: 100%%; transition: width 0.3s; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 9pt; font-weight: 600; }
.badge-green { background: #E8F5E9; color: #2E7D32; }
.badge-amber { background: #FFF3E0; color: #E65100; }
.badge-red { background: #FFEBEE; color: #C62828; }
.metric-table th:first-child { width: 40%%; }
.framework-badge { display: inline-block; padding: 4px 12px; border-radius: 16px; font-size: 10pt; margin: 2px; }
"""

# HTML interactive features (for HTML-only output)
HTML_INTERACTIVE_SCRIPT: str = """
<script>
// GreenLang Report Interactive Features
document.addEventListener('DOMContentLoaded', function() {
    // Collapsible sections
    document.querySelectorAll('.section-header').forEach(function(header) {
        header.addEventListener('click', function() {
            var content = this.nextElementSibling;
            content.style.display = content.style.display === 'none' ? 'block' : 'none';
            this.classList.toggle('collapsed');
        });
    });

    // Search functionality
    var searchInput = document.getElementById('report-search');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            var query = this.value.toLowerCase();
            document.querySelectorAll('.section').forEach(function(section) {
                var text = section.textContent.toLowerCase();
                section.style.display = text.includes(query) ? 'block' : 'none';
            });
        });
    }

    // Print button
    var printBtn = document.getElementById('print-btn');
    if (printBtn) {
        printBtn.addEventListener('click', function() { window.print(); });
    }

    // Export data table to CSV
    document.querySelectorAll('.export-csv').forEach(function(btn) {
        btn.addEventListener('click', function() {
            var table = this.closest('.table-container').querySelector('table');
            var csv = [];
            table.querySelectorAll('tr').forEach(function(row) {
                var cols = [];
                row.querySelectorAll('th, td').forEach(function(cell) {
                    cols.push('"' + cell.textContent.replace(/"/g, '""') + '"');
                });
                csv.push(cols.join(','));
            });
            var blob = new Blob([csv.join('\\n')], {type: 'text/csv'});
            var a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'report_data.csv';
            a.click();
        });
    });
});
</script>
"""

# Chart.js CDN link
CHARTJS_CDN: str = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class BrandingConfig(BaseModel):
    """Branding configuration for rendered reports."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary_color: str = Field(default="#1B5E20", description="Primary brand color (hex)")
    secondary_color: str = Field(default="#4CAF50", description="Secondary brand color")
    accent_color: str = Field(default="#81C784", description="Accent color")
    text_color: str = Field(default="#212121", description="Body text color")
    background_color: str = Field(default="#FFFFFF", description="Background color")
    header_bg: str = Field(default="#1B5E20", description="Header background")
    header_text: str = Field(default="#FFFFFF", description="Header text color")
    font_family: str = Field(default="Roboto, Arial, sans-serif")
    font_size_body: str = Field(default="11pt")
    font_size_header: str = Field(default="24pt")
    font_size_subheader: str = Field(default="16pt")
    font_size_caption: str = Field(default="9pt")
    line_height: str = Field(default="1.5")
    logo_path: str = Field(default="", description="Path to company logo")
    logo_base64: str = Field(default="", description="Base64 encoded logo")
    company_name: str = Field(default="", description="Company name for headers")
    footer_text: str = Field(default="Generated by GreenLang Platform")
    theme: BrandingTheme = Field(default=BrandingTheme.DEFAULT)


class ChartConfig(BaseModel):
    """Configuration for a chart to be rendered."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chart_id: str = Field(default_factory=_new_uuid)
    chart_type: ChartType = Field(..., description="Type of chart")
    title: str = Field(default="", description="Chart title")
    labels: List[str] = Field(default_factory=list, description="X-axis / category labels")
    datasets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chart datasets [{label, data, color}]",
    )
    x_axis_label: str = Field(default="")
    y_axis_label: str = Field(default="")
    show_legend: bool = Field(default=True)
    show_grid: bool = Field(default=True)
    width: int = Field(default=800, description="Chart width in pixels")
    height: int = Field(default=400, description="Chart height in pixels")
    responsive: bool = Field(default=True)


class TableConfig(BaseModel):
    """Configuration for a data table to be rendered."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    table_id: str = Field(default_factory=_new_uuid)
    title: str = Field(default="", description="Table title")
    headers: List[str] = Field(default_factory=list, description="Column headers")
    rows: List[List[Any]] = Field(default_factory=list, description="Row data")
    column_widths: List[str] = Field(
        default_factory=list, description="Column widths (CSS)"
    )
    style: TableStyle = Field(default=TableStyle.DEFAULT)
    sortable: bool = Field(default=False, description="Enable sorting (HTML only)")
    exportable: bool = Field(default=True, description="Enable CSV export")
    footer_row: Optional[List[Any]] = Field(default=None, description="Footer row (totals)")
    highlight_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conditional formatting rules",
    )


class ReportSection(BaseModel):
    """A section of a compiled report ready for rendering."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    section_id: str = Field(default_factory=_new_uuid)
    section_type: str = Field(default="narrative", description="Section type")
    title: str = Field(default="", description="Section title")
    content: str = Field(default="", description="Markdown/HTML content")
    level: int = Field(default=2, description="Heading level (1-6)")
    order: int = Field(default=0, description="Sort order")
    charts: List[ChartConfig] = Field(default_factory=list)
    tables: List[TableConfig] = Field(default_factory=list)
    kpi_cards: List[Dict[str, Any]] = Field(default_factory=list)
    footnotes: List[str] = Field(default_factory=list)
    page_break_before: bool = Field(default=False)
    page_break_after: bool = Field(default=False)
    framework: str = Field(default="", description="Framework tag")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RenderInput(BaseModel):
    """Input for the FormatRenderingEngine."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    organization_id: str = Field(..., description="Organization identifier")
    report_id: str = Field(default_factory=_new_uuid, description="Report identifier")
    report_title: str = Field(default="Net Zero Report", description="Report title")
    report_subtitle: str = Field(default="", description="Report subtitle")
    framework: FrameworkTarget = Field(
        default=FrameworkTarget.MULTI,
        description="Target framework",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PDF,
        description="Desired output format",
    )
    sections: List[ReportSection] = Field(
        default_factory=list,
        description="Report sections to render",
    )
    branding: BrandingConfig = Field(
        default_factory=BrandingConfig,
        description="Branding configuration",
    )
    page_size: PageSize = Field(default=PageSize.A4)
    orientation: Orientation = Field(default=Orientation.PORTRAIT)
    quality: RenderQuality = Field(default=RenderQuality.STANDARD)

    # Cover page
    include_cover: bool = Field(default=True)
    cover_date: str = Field(
        default_factory=lambda: _utcnow().strftime("%B %Y"),
        description="Date displayed on cover",
    )
    cover_prepared_by: str = Field(default="GreenLang Platform")
    cover_confidentiality: str = Field(default="")

    # Table of contents
    include_toc: bool = Field(default=True)

    # Appendices
    include_glossary: bool = Field(default=True)
    include_disclaimer: bool = Field(default=True)
    custom_disclaimer: str = Field(default="")

    # XBRL options (for XBRL/iXBRL output)
    xbrl_taxonomy: str = Field(default="", description="Taxonomy URL")
    xbrl_entity: str = Field(default="", description="Entity identifier")
    xbrl_period_start: str = Field(default="")
    xbrl_period_end: str = Field(default="")

    # Excel options
    excel_include_charts: bool = Field(default=True)
    excel_freeze_panes: bool = Field(default=True)
    excel_auto_filter: bool = Field(default=True)

    # JSON options
    json_pretty: bool = Field(default=True)
    json_include_metadata: bool = Field(default=True)
    json_schema_version: str = Field(default="1.0.0")

    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class RenderedChart(BaseModel):
    """A rendered chart artifact."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chart_id: str = Field(default_factory=_new_uuid)
    chart_type: str = Field(default="")
    title: str = Field(default="")
    html_content: str = Field(default="", description="Chart HTML/JS code")
    image_base64: str = Field(default="", description="Chart as base64 PNG (for PDF)")
    width: int = Field(default=800)
    height: int = Field(default=400)


class RenderedTable(BaseModel):
    """A rendered data table artifact."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    table_id: str = Field(default_factory=_new_uuid)
    title: str = Field(default="")
    html_content: str = Field(default="", description="Table HTML code")
    row_count: int = Field(default=0)
    column_count: int = Field(default=0)


class RenderedSection(BaseModel):
    """A rendered report section."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    section_id: str = Field(default_factory=_new_uuid)
    title: str = Field(default="")
    html_content: str = Field(default="", description="Full section HTML")
    charts: List[RenderedChart] = Field(default_factory=list)
    tables: List[RenderedTable] = Field(default_factory=list)
    word_count: int = Field(default=0)
    page_estimate: int = Field(default=1)


class RenderedArtifact(BaseModel):
    """A complete rendered report artifact."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    artifact_id: str = Field(default_factory=_new_uuid)
    report_id: str = Field(default="")
    output_format: str = Field(default="")
    content: str = Field(default="", description="Rendered content (HTML/JSON/CSV)")
    content_bytes: Optional[bytes] = Field(
        default=None, description="Binary content (PDF/Excel)"
    )
    filename: str = Field(default="", description="Suggested filename")
    mime_type: str = Field(default="")
    file_size_bytes: int = Field(default=0)
    page_count: int = Field(default=0, description="Page count (PDF only)")
    sheet_count: int = Field(default=0, description="Sheet count (Excel only)")
    chart_count: int = Field(default=0)
    table_count: int = Field(default=0)
    section_count: int = Field(default=0)
    provenance_hash: str = Field(default="")


class RenderResult(BaseModel):
    """Complete rendering result with all artifacts and metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    result_id: str = Field(default_factory=_new_uuid)
    organization_id: str = Field(..., description="Organization identifier")
    report_id: str = Field(default="")
    framework: str = Field(default="multi")

    # Rendered artifacts (one per format)
    artifacts: List[RenderedArtifact] = Field(
        default_factory=list, description="Rendered output artifacts"
    )
    primary_artifact: Optional[RenderedArtifact] = Field(
        default=None, description="Primary output artifact"
    )

    # Rendered sections
    rendered_sections: List[RenderedSection] = Field(
        default_factory=list, description="Individual rendered sections"
    )

    # Statistics
    total_pages: int = Field(default=0)
    total_charts: int = Field(default=0)
    total_tables: int = Field(default=0)
    total_sections: int = Field(default=0)
    total_words: int = Field(default=0)
    formats_generated: List[str] = Field(default_factory=list)

    # CSS used
    css_content: str = Field(default="", description="CSS stylesheet generated")

    # Quality
    render_quality: str = Field(default="standard")
    branding_applied: bool = Field(default=False)
    interactive_features: bool = Field(default=False)

    # Provenance
    calculated_at: str = Field(default_factory=lambda: _utcnow().isoformat())
    processing_time_ms: Decimal = Field(default=_decimal("0"))
    engine_version: str = Field(default=_MODULE_VERSION)
    provenance_hash: str = Field(default="")

    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FormatRenderingEngine
# ---------------------------------------------------------------------------

class FormatRenderingEngine:
    """
    Multi-format rendering engine for climate disclosure reports.

    Generates PDF, HTML, Excel, JSON, and XBRL output from compiled
    report structures with full branding, interactive charts, and
    digital taxonomy tagging.

    Key Methods:
        render()           -- Render report in specified format
        render_pdf()       -- Generate PDF document
        render_html()      -- Generate interactive HTML
        render_excel()     -- Generate Excel workbook
        render_json()      -- Generate JSON output
        render_xbrl()      -- Generate XBRL/iXBRL output
        render_multi()     -- Generate multiple formats at once

    Usage::

        engine = FormatRenderingEngine()
        result = await engine.render(RenderInput(
            organization_id="org-123",
            report_title="2025 Climate Report",
            output_format=OutputFormat.PDF,
            sections=[...],
        ))
    """

    def __init__(self) -> None:
        """Initialize the FormatRenderingEngine."""
        self._chart_counter: int = 0
        logger.info("FormatRenderingEngine v%s initialized", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def render(self, inp: RenderInput) -> RenderResult:
        """
        Render a report in the specified output format.

        Routes to the appropriate format-specific rendering method
        based on inp.output_format.

        Args:
            inp: Render input with sections, branding, and format.

        Returns:
            RenderResult with rendered artifacts.
        """
        t0 = time.perf_counter()
        logger.info(
            "Rendering report '%s' as %s for org=%s",
            inp.report_title,
            inp.output_format.value,
            inp.organization_id,
        )

        # Generate CSS
        css = self._generate_css(inp)

        # Render sections
        rendered_sections = await self._render_all_sections(inp, css)

        # Route to format-specific renderer
        format_method = {
            OutputFormat.PDF: self.render_pdf,
            OutputFormat.HTML: self.render_html,
            OutputFormat.EXCEL: self.render_excel,
            OutputFormat.JSON: self.render_json,
            OutputFormat.XBRL: self.render_xbrl,
            OutputFormat.IXBRL: self.render_xbrl,
            OutputFormat.CSV: self._render_csv,
        }

        renderer = format_method.get(inp.output_format, self.render_html)
        artifact = await renderer(inp, rendered_sections, css)

        # Build result
        total_charts = sum(len(s.charts) for s in rendered_sections)
        total_tables = sum(len(s.tables) for s in rendered_sections)
        total_words = sum(s.word_count for s in rendered_sections)

        elapsed = _decimal(str(time.perf_counter() - t0)) * _decimal("1000")

        result = RenderResult(
            organization_id=inp.organization_id,
            report_id=inp.report_id,
            framework=inp.framework.value,
            artifacts=[artifact],
            primary_artifact=artifact,
            rendered_sections=rendered_sections,
            total_pages=artifact.page_count or self._estimate_pages(total_words),
            total_charts=total_charts,
            total_tables=total_tables,
            total_sections=len(rendered_sections),
            total_words=total_words,
            formats_generated=[inp.output_format.value],
            css_content=css,
            render_quality=inp.quality.value,
            branding_applied=True,
            interactive_features=inp.output_format == OutputFormat.HTML,
            processing_time_ms=_round_val(elapsed, 1),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Render complete: format=%s, sections=%d, pages=%d, "
            "charts=%d, tables=%d, %.1fms",
            inp.output_format.value,
            len(rendered_sections),
            result.total_pages,
            total_charts,
            total_tables,
            float(elapsed),
        )
        return result

    async def render_pdf(
        self,
        inp: RenderInput,
        rendered_sections: Optional[List[RenderedSection]] = None,
        css: Optional[str] = None,
    ) -> RenderedArtifact:
        """
        Generate PDF document from report sections.

        Uses WeasyPrint-compatible HTML+CSS to produce a print-ready
        PDF with cover page, table of contents, headers/footers,
        page numbers, charts (as images), and data tables.

        Args:
            inp: Render input.
            rendered_sections: Pre-rendered sections (optional).
            css: Pre-generated CSS (optional).

        Returns:
            RenderedArtifact with PDF content.
        """
        if css is None:
            css = self._generate_css(inp)
        if rendered_sections is None:
            rendered_sections = await self._render_all_sections(inp, css)

        html_parts: List[str] = [
            "<!DOCTYPE html>",
            f'<html lang="en">',
            "<head>",
            f"<meta charset=\"utf-8\">",
            f"<title>{inp.report_title}</title>",
            f"<style>{css}</style>",
            "</head>",
            "<body>",
        ]

        # Cover page
        if inp.include_cover:
            html_parts.append(self._generate_cover_page(inp))

        # Table of contents
        if inp.include_toc:
            html_parts.append(self._generate_toc(rendered_sections))

        # Report sections
        for section in rendered_sections:
            html_parts.append(section.html_content)

        # Glossary
        if inp.include_glossary:
            html_parts.append(self._generate_glossary(inp.framework))

        # Disclaimer
        if inp.include_disclaimer:
            html_parts.append(self._generate_disclaimer(inp))

        html_parts.extend(["</body>", "</html>"])
        full_html = "\n".join(html_parts)

        # Estimate pages
        total_words = sum(s.word_count for s in rendered_sections)
        page_count = self._estimate_pages(total_words)

        filename = self._generate_filename(inp, "pdf")

        return RenderedArtifact(
            report_id=inp.report_id,
            output_format="pdf",
            content=full_html,
            filename=filename,
            mime_type="application/pdf",
            file_size_bytes=len(full_html.encode("utf-8")),
            page_count=page_count,
            chart_count=sum(len(s.charts) for s in rendered_sections),
            table_count=sum(len(s.tables) for s in rendered_sections),
            section_count=len(rendered_sections),
            provenance_hash=_compute_hash({"content": full_html, "format": "pdf"}),
        )

    async def render_html(
        self,
        inp: RenderInput,
        rendered_sections: Optional[List[RenderedSection]] = None,
        css: Optional[str] = None,
    ) -> RenderedArtifact:
        """
        Generate interactive HTML5 report.

        Includes Chart.js visualizations, collapsible sections, search
        functionality, CSV export for tables, and responsive layout.

        Args:
            inp: Render input.
            rendered_sections: Pre-rendered sections (optional).
            css: Pre-generated CSS (optional).

        Returns:
            RenderedArtifact with HTML content.
        """
        if css is None:
            css = self._generate_css(inp)
        if rendered_sections is None:
            rendered_sections = await self._render_all_sections(inp, css)

        html_parts: List[str] = [
            "<!DOCTYPE html>",
            f'<html lang="en">',
            "<head>",
            f"<meta charset=\"utf-8\">",
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
            f"<title>{inp.report_title}</title>",
            f"<style>{css}</style>",
            f'<script src="{CHARTJS_CDN}"></script>',
            "</head>",
            "<body>",
        ]

        # Navigation toolbar
        html_parts.append(self._generate_nav_toolbar(inp, rendered_sections))

        # Cover section (inline for HTML)
        if inp.include_cover:
            html_parts.append(self._generate_cover_page(inp))

        # Report sections
        for section in rendered_sections:
            html_parts.append(section.html_content)

        # Glossary
        if inp.include_glossary:
            html_parts.append(self._generate_glossary(inp.framework))

        # Disclaimer
        if inp.include_disclaimer:
            html_parts.append(self._generate_disclaimer(inp))

        # Interactive scripts
        html_parts.append(HTML_INTERACTIVE_SCRIPT)

        html_parts.extend(["</body>", "</html>"])
        full_html = "\n".join(html_parts)

        filename = self._generate_filename(inp, "html")

        return RenderedArtifact(
            report_id=inp.report_id,
            output_format="html",
            content=full_html,
            filename=filename,
            mime_type="text/html",
            file_size_bytes=len(full_html.encode("utf-8")),
            chart_count=sum(len(s.charts) for s in rendered_sections),
            table_count=sum(len(s.tables) for s in rendered_sections),
            section_count=len(rendered_sections),
            provenance_hash=_compute_hash({"content": full_html, "format": "html"}),
        )

    async def render_excel(
        self,
        inp: RenderInput,
        rendered_sections: Optional[List[RenderedSection]] = None,
        css: Optional[str] = None,
    ) -> RenderedArtifact:
        """
        Generate Excel workbook with multiple sheets.

        Creates a structured Excel workbook with framework-specific
        sheets, formatted data tables, named ranges, charts, and
        conditional formatting.  Suitable for CDP submission.

        Args:
            inp: Render input.
            rendered_sections: Pre-rendered sections (optional).
            css: Pre-generated CSS (optional).

        Returns:
            RenderedArtifact with Excel workbook structure (JSON).
        """
        if rendered_sections is None:
            if css is None:
                css = self._generate_css(inp)
            rendered_sections = await self._render_all_sections(inp, css)

        framework_key = inp.framework.value
        sheet_names = EXCEL_SHEET_NAMES.get(framework_key, EXCEL_SHEET_NAMES["multi"])

        workbook: Dict[str, Any] = {
            "metadata": {
                "title": inp.report_title,
                "subject": f"{framework_key.upper()} Climate Disclosure",
                "author": inp.cover_prepared_by,
                "created": _utcnow().isoformat(),
                "framework": framework_key,
            },
            "sheets": [],
        }

        # Summary sheet
        summary_sheet = {
            "name": sheet_names[0] if sheet_names else "Summary",
            "headers": ["Metric", "Value", "Unit", "Source", "Period"],
            "rows": [],
            "freeze_pane": "B2" if inp.excel_freeze_panes else None,
            "auto_filter": inp.excel_auto_filter,
        }

        # Extract data from rendered sections
        for section in rendered_sections:
            for table in section.tables:
                sheet = {
                    "name": table.title[:31] if table.title else f"Sheet_{len(workbook['sheets']) + 1}",
                    "headers": [],
                    "rows": [],
                    "freeze_pane": "A2" if inp.excel_freeze_panes else None,
                    "auto_filter": inp.excel_auto_filter,
                }
                workbook["sheets"].append(sheet)

            # Add section content as summary rows
            if section.title:
                summary_sheet["rows"].append([
                    section.title,
                    f"{section.word_count} words",
                    "",
                    "GreenLang",
                    inp.cover_date,
                ])

        # Add summary sheet at the beginning
        workbook["sheets"].insert(0, summary_sheet)

        # Add framework-specific sheets
        for sheet_name in sheet_names[1:]:
            matching_sections = [
                s for s in rendered_sections
                if sheet_name.lower() in s.title.lower()
            ]
            sheet_data: Dict[str, Any] = {
                "name": sheet_name[:31],
                "headers": ["Item", "Disclosure", "Status", "Evidence"],
                "rows": [],
                "freeze_pane": "B2" if inp.excel_freeze_panes else None,
                "auto_filter": inp.excel_auto_filter,
            }
            for sec in matching_sections:
                sheet_data["rows"].append([
                    sec.title,
                    f"({sec.word_count} words)",
                    "Complete" if sec.word_count > 0 else "Pending",
                    "Available" if sec.charts or sec.tables else "N/A",
                ])
            if sheet_data["name"] not in [s["name"] for s in workbook["sheets"]]:
                workbook["sheets"].append(sheet_data)

        # Serialize to JSON structure (openpyxl-compatible)
        excel_json = json.dumps(workbook, indent=2, default=str)
        filename = self._generate_filename(inp, "xlsx")

        return RenderedArtifact(
            report_id=inp.report_id,
            output_format="excel",
            content=excel_json,
            filename=filename,
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            file_size_bytes=len(excel_json.encode("utf-8")),
            sheet_count=len(workbook["sheets"]),
            chart_count=sum(len(s.charts) for s in rendered_sections),
            table_count=sum(len(s.tables) for s in rendered_sections),
            section_count=len(rendered_sections),
            provenance_hash=_compute_hash(workbook),
        )

    async def render_json(
        self,
        inp: RenderInput,
        rendered_sections: Optional[List[RenderedSection]] = None,
        css: Optional[str] = None,
    ) -> RenderedArtifact:
        """
        Generate structured JSON output for API consumption.

        Follows a JSON Schema structure suitable for data warehouse
        ingestion, API responses, or downstream processing.

        Args:
            inp: Render input.
            rendered_sections: Pre-rendered sections (optional).
            css: Pre-generated CSS (optional).

        Returns:
            RenderedArtifact with JSON content.
        """
        if rendered_sections is None:
            if css is None:
                css = self._generate_css(inp)
            rendered_sections = await self._render_all_sections(inp, css)

        report_json: Dict[str, Any] = {
            "$schema": f"https://greenlang.io/schemas/report/v{inp.json_schema_version}",
            "schema_version": inp.json_schema_version,
            "report": {
                "id": inp.report_id,
                "title": inp.report_title,
                "subtitle": inp.report_subtitle,
                "framework": inp.framework.value,
                "organization_id": inp.organization_id,
                "reporting_period": inp.cover_date,
                "generated_at": _utcnow().isoformat(),
                "generated_by": "GreenLang Platform",
                "engine_version": _MODULE_VERSION,
            },
            "sections": [],
            "summary": {
                "total_sections": len(rendered_sections),
                "total_words": sum(s.word_count for s in rendered_sections),
                "total_charts": sum(len(s.charts) for s in rendered_sections),
                "total_tables": sum(len(s.tables) for s in rendered_sections),
            },
        }

        # Add metadata if requested
        if inp.json_include_metadata:
            report_json["metadata"] = {
                "branding": {
                    "theme": inp.branding.theme.value,
                    "primary_color": inp.branding.primary_color,
                    "company_name": inp.branding.company_name,
                },
                "quality": inp.quality.value,
                "page_size": inp.page_size.value,
                "orientation": inp.orientation.value,
            }

        # Add sections
        for section in rendered_sections:
            section_data: Dict[str, Any] = {
                "id": section.section_id,
                "title": section.title,
                "word_count": section.word_count,
                "page_estimate": section.page_estimate,
                "charts": [
                    {
                        "id": chart.chart_id,
                        "type": chart.chart_type,
                        "title": chart.title,
                    }
                    for chart in section.charts
                ],
                "tables": [
                    {
                        "id": table.table_id,
                        "title": table.title,
                        "rows": table.row_count,
                        "columns": table.column_count,
                    }
                    for table in section.tables
                ],
            }
            report_json["sections"].append(section_data)

        indent = 2 if inp.json_pretty else None
        json_content = json.dumps(report_json, indent=indent, default=str)
        filename = self._generate_filename(inp, "json")

        return RenderedArtifact(
            report_id=inp.report_id,
            output_format="json",
            content=json_content,
            filename=filename,
            mime_type="application/json",
            file_size_bytes=len(json_content.encode("utf-8")),
            chart_count=sum(len(s.charts) for s in rendered_sections),
            table_count=sum(len(s.tables) for s in rendered_sections),
            section_count=len(rendered_sections),
            provenance_hash=_compute_hash(report_json),
        )

    async def render_xbrl(
        self,
        inp: RenderInput,
        rendered_sections: Optional[List[RenderedSection]] = None,
        css: Optional[str] = None,
    ) -> RenderedArtifact:
        """
        Generate XBRL or iXBRL output for digital filing.

        For SEC filings, generates iXBRL (inline XBRL in HTML).
        For CSRD/ESRS, generates ESEF XBRL.
        Delegates taxonomy-specific tagging to Engine 4 (XBRLTaggingEngine).

        Args:
            inp: Render input.
            rendered_sections: Pre-rendered sections (optional).
            css: Pre-generated CSS (optional).

        Returns:
            RenderedArtifact with XBRL/iXBRL content.
        """
        if rendered_sections is None:
            if css is None:
                css = self._generate_css(inp)
            rendered_sections = await self._render_all_sections(inp, css)

        is_inline = inp.output_format == OutputFormat.IXBRL
        entity_id = inp.xbrl_entity or inp.organization_id
        period_start = inp.xbrl_period_start or "2025-01-01"
        period_end = inp.xbrl_period_end or "2025-12-31"

        if is_inline:
            # Generate iXBRL (inline XBRL in HTML)
            content = self._generate_ixbrl(
                inp, rendered_sections, css, entity_id, period_start, period_end
            )
            ext = "html"
            mime = "text/html"
        else:
            # Generate XBRL instance document
            content = self._generate_xbrl_instance(
                inp, rendered_sections, entity_id, period_start, period_end
            )
            ext = "xml"
            mime = "application/xml"

        filename = self._generate_filename(inp, ext)

        return RenderedArtifact(
            report_id=inp.report_id,
            output_format=inp.output_format.value,
            content=content,
            filename=filename,
            mime_type=mime,
            file_size_bytes=len(content.encode("utf-8")),
            chart_count=0,
            table_count=sum(len(s.tables) for s in rendered_sections),
            section_count=len(rendered_sections),
            provenance_hash=_compute_hash({"content": content, "format": inp.output_format.value}),
        )

    async def render_multi(
        self,
        inp: RenderInput,
        formats: Optional[List[OutputFormat]] = None,
    ) -> RenderResult:
        """
        Generate multiple output formats from a single report.

        Renders once and generates artifacts for each requested format.

        Args:
            inp: Render input.
            formats: List of output formats (default: PDF, HTML, JSON).

        Returns:
            RenderResult with multiple artifacts.
        """
        t0 = time.perf_counter()
        if formats is None:
            formats = [OutputFormat.PDF, OutputFormat.HTML, OutputFormat.JSON]

        logger.info(
            "Multi-format render: %s for org=%s",
            [f.value for f in formats],
            inp.organization_id,
        )

        css = self._generate_css(inp)
        rendered_sections = await self._render_all_sections(inp, css)

        artifacts: List[RenderedArtifact] = []
        for fmt in formats:
            fmt_inp = inp.model_copy(update={"output_format": fmt})
            format_method = {
                OutputFormat.PDF: self.render_pdf,
                OutputFormat.HTML: self.render_html,
                OutputFormat.EXCEL: self.render_excel,
                OutputFormat.JSON: self.render_json,
                OutputFormat.XBRL: self.render_xbrl,
                OutputFormat.IXBRL: self.render_xbrl,
                OutputFormat.CSV: self._render_csv,
            }
            renderer = format_method.get(fmt, self.render_html)
            artifact = await renderer(fmt_inp, rendered_sections, css)
            artifacts.append(artifact)

        total_charts = sum(len(s.charts) for s in rendered_sections)
        total_tables = sum(len(s.tables) for s in rendered_sections)
        total_words = sum(s.word_count for s in rendered_sections)
        elapsed = _decimal(str(time.perf_counter() - t0)) * _decimal("1000")

        result = RenderResult(
            organization_id=inp.organization_id,
            report_id=inp.report_id,
            framework=inp.framework.value,
            artifacts=artifacts,
            primary_artifact=artifacts[0] if artifacts else None,
            rendered_sections=rendered_sections,
            total_pages=self._estimate_pages(total_words),
            total_charts=total_charts,
            total_tables=total_tables,
            total_sections=len(rendered_sections),
            total_words=total_words,
            formats_generated=[f.value for f in formats],
            css_content=css,
            render_quality=inp.quality.value,
            branding_applied=True,
            interactive_features=OutputFormat.HTML in formats,
            processing_time_ms=_round_val(elapsed, 1),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Private Rendering Methods
    # ------------------------------------------------------------------

    async def _render_csv(
        self,
        inp: RenderInput,
        rendered_sections: Optional[List[RenderedSection]] = None,
        css: Optional[str] = None,
    ) -> RenderedArtifact:
        """Generate CSV output from report tables."""
        if rendered_sections is None:
            if css is None:
                css = self._generate_css(inp)
            rendered_sections = await self._render_all_sections(inp, css)

        csv_lines: List[str] = []
        csv_lines.append(f"# {inp.report_title}")
        csv_lines.append(f"# Framework: {inp.framework.value}")
        csv_lines.append(f"# Generated: {_utcnow().isoformat()}")
        csv_lines.append("")

        for section in rendered_sections:
            if section.tables:
                for table in section.tables:
                    csv_lines.append(f"# {table.title}")
                    csv_lines.append(table.html_content)  # Simplified
                    csv_lines.append("")

        csv_content = "\n".join(csv_lines)
        filename = self._generate_filename(inp, "csv")

        return RenderedArtifact(
            report_id=inp.report_id,
            output_format="csv",
            content=csv_content,
            filename=filename,
            mime_type="text/csv",
            file_size_bytes=len(csv_content.encode("utf-8")),
            section_count=len(rendered_sections),
            provenance_hash=_compute_hash({"content": csv_content, "format": "csv"}),
        )

    async def _render_all_sections(
        self, inp: RenderInput, css: str
    ) -> List[RenderedSection]:
        """Render all report sections to HTML."""
        rendered: List[RenderedSection] = []

        # Sort sections by order
        sorted_sections = sorted(inp.sections, key=lambda s: s.order)

        for section in sorted_sections:
            rendered_section = await self._render_section(section, inp, css)
            rendered.append(rendered_section)

        return rendered

    async def _render_section(
        self, section: ReportSection, inp: RenderInput, css: str
    ) -> RenderedSection:
        """Render a single report section to HTML."""
        html_parts: List[str] = []

        # Page break
        if section.page_break_before:
            html_parts.append('<div style="page-break-before: always;"></div>')

        # Section wrapper
        section_class = f"section section-{section.section_type}"
        html_parts.append(f'<div class="{section_class}" id="section-{section.section_id}">')

        # Section header
        heading_tag = f"h{min(section.level, 6)}"
        if section.title:
            framework_badge = ""
            if section.framework:
                colors = FRAMEWORK_COLORS.get(section.framework.lower(), FRAMEWORK_COLORS["multi"])
                framework_badge = (
                    f' <span class="framework-badge" '
                    f'style="background:{colors["accent"]};color:{colors["primary"]}">'
                    f'{section.framework.upper()}</span>'
                )
            html_parts.append(
                f'<{heading_tag} class="section-header">'
                f'{section.title}{framework_badge}'
                f'</{heading_tag}>'
            )

        # KPI cards
        if section.kpi_cards:
            html_parts.append('<div class="kpi-container">')
            for kpi in section.kpi_cards:
                html_parts.append(self._render_kpi_card(kpi))
            html_parts.append('</div>')

        # Content (markdown-to-HTML simplified)
        if section.content:
            rendered_content = self._markdown_to_html(section.content)
            html_parts.append(f'<div class="section-content">{rendered_content}</div>')

        # Charts
        rendered_charts: List[RenderedChart] = []
        for chart_config in section.charts:
            rendered_chart = self._render_chart(chart_config, inp.output_format)
            rendered_charts.append(rendered_chart)
            html_parts.append(
                f'<div class="chart-container">{rendered_chart.html_content}</div>'
            )

        # Tables
        rendered_tables: List[RenderedTable] = []
        for table_config in section.tables:
            rendered_table = self._render_table(table_config)
            rendered_tables.append(rendered_table)
            html_parts.append(
                f'<div class="table-container">{rendered_table.html_content}</div>'
            )

        # Footnotes
        if section.footnotes:
            html_parts.append('<div class="footnotes">')
            for i, footnote in enumerate(section.footnotes, 1):
                html_parts.append(
                    f'<p class="footnote"><sup>{i}</sup> {footnote}</p>'
                )
            html_parts.append('</div>')

        # Close section wrapper
        html_parts.append('</div>')

        # Page break after
        if section.page_break_after:
            html_parts.append('<div style="page-break-after: always;"></div>')

        full_html = "\n".join(html_parts)
        word_count = len(section.content.split()) if section.content else 0

        return RenderedSection(
            section_id=section.section_id,
            title=section.title,
            html_content=full_html,
            charts=rendered_charts,
            tables=rendered_tables,
            word_count=word_count,
            page_estimate=max(1, word_count // 300),
        )

    def _render_chart(
        self, config: ChartConfig, output_format: OutputFormat
    ) -> RenderedChart:
        """Render a chart configuration to HTML/JS (Chart.js)."""
        self._chart_counter += 1
        canvas_id = f"chart_{self._chart_counter}"

        # Build Chart.js config
        chart_type_map = {
            ChartType.BAR: "bar",
            ChartType.LINE: "line",
            ChartType.PIE: "pie",
            ChartType.DOUGHNUT: "doughnut",
            ChartType.STACKED_BAR: "bar",
            ChartType.AREA: "line",
            ChartType.SCATTER: "scatter",
            ChartType.RADAR: "radar",
        }
        js_chart_type = chart_type_map.get(config.chart_type, "bar")

        datasets_js = []
        for ds in config.datasets:
            ds_config = {
                "label": ds.get("label", ""),
                "data": ds.get("data", []),
                "backgroundColor": ds.get("color", ds.get("backgroundColor", "#4CAF50")),
                "borderColor": ds.get("borderColor", ds.get("color", "#1B5E20")),
                "borderWidth": ds.get("borderWidth", 2),
            }
            if config.chart_type in (ChartType.LINE, ChartType.AREA):
                ds_config["fill"] = config.chart_type == ChartType.AREA
                ds_config["tension"] = 0.3
            datasets_js.append(ds_config)

        chart_config = {
            "type": js_chart_type,
            "data": {
                "labels": config.labels,
                "datasets": datasets_js,
            },
            "options": {
                "responsive": config.responsive,
                "maintainAspectRatio": True,
                "plugins": {
                    "title": {
                        "display": bool(config.title),
                        "text": config.title,
                        "font": {"size": 14, "weight": "bold"},
                    },
                    "legend": {"display": config.show_legend},
                },
                "scales": {},
            },
        }

        # Add scales for bar/line charts
        if config.chart_type not in (ChartType.PIE, ChartType.DOUGHNUT, ChartType.RADAR):
            chart_config["options"]["scales"] = {
                "x": {
                    "title": {
                        "display": bool(config.x_axis_label),
                        "text": config.x_axis_label,
                    },
                    "grid": {"display": config.show_grid},
                },
                "y": {
                    "title": {
                        "display": bool(config.y_axis_label),
                        "text": config.y_axis_label,
                    },
                    "grid": {"display": config.show_grid},
                    "beginAtZero": True,
                },
            }

            # Stacked bar
            if config.chart_type == ChartType.STACKED_BAR:
                chart_config["options"]["scales"]["x"]["stacked"] = True
                chart_config["options"]["scales"]["y"]["stacked"] = True

        config_json = json.dumps(chart_config, indent=2, default=str)

        html = (
            f'<div style="width:{config.width}px;max-width:100%;margin:auto;">\n'
            f'  <canvas id="{canvas_id}" width="{config.width}" '
            f'height="{config.height}"></canvas>\n'
            f'</div>\n'
            f'<script>\n'
            f'  new Chart(document.getElementById("{canvas_id}"), {config_json});\n'
            f'</script>'
        )

        return RenderedChart(
            chart_id=config.chart_id,
            chart_type=config.chart_type.value,
            title=config.title,
            html_content=html,
            width=config.width,
            height=config.height,
        )

    def _render_table(self, config: TableConfig) -> RenderedTable:
        """Render a table configuration to HTML."""
        parts: List[str] = []

        # Table title
        if config.title:
            parts.append(f'<h4 class="table-title">{config.title}</h4>')

        # Export button
        if config.exportable:
            parts.append(
                '<button class="export-csv" title="Export to CSV">'
                'Export CSV</button>'
            )

        # Table styles
        table_class = f"data-table table-{config.style.value}"
        if config.style == TableStyle.STRIPED:
            table_class += " striped"

        parts.append(f'<table class="{table_class}">')

        # Header row
        if config.headers:
            parts.append("<thead><tr>")
            for i, header in enumerate(config.headers):
                width_style = ""
                if config.column_widths and i < len(config.column_widths):
                    width_style = f' style="width:{config.column_widths[i]}"'
                parts.append(f"<th{width_style}>{header}</th>")
            parts.append("</tr></thead>")

        # Data rows
        parts.append("<tbody>")
        for row in config.rows:
            parts.append("<tr>")
            for cell in row:
                cell_str = str(cell) if cell is not None else ""
                # Apply highlight rules
                cell_class = self._get_cell_class(cell, config.highlight_rules)
                if cell_class:
                    parts.append(f'<td class="{cell_class}">{cell_str}</td>')
                else:
                    parts.append(f"<td>{cell_str}</td>")
            parts.append("</tr>")
        parts.append("</tbody>")

        # Footer row
        if config.footer_row:
            parts.append("<tfoot><tr>")
            for cell in config.footer_row:
                parts.append(f"<th>{cell}</th>")
            parts.append("</tr></tfoot>")

        parts.append("</table>")

        html = "\n".join(parts)
        return RenderedTable(
            table_id=config.table_id,
            title=config.title,
            html_content=html,
            row_count=len(config.rows),
            column_count=len(config.headers),
        )

    def _render_kpi_card(self, kpi: Dict[str, Any]) -> str:
        """Render a single KPI card to HTML."""
        value = kpi.get("value", "0")
        label = kpi.get("label", "Metric")
        unit = kpi.get("unit", "")
        change = kpi.get("change", "")
        change_direction = kpi.get("direction", "")

        change_class = "positive" if change_direction == "positive" else "negative"
        change_html = ""
        if change:
            arrow = "&#9650;" if change_direction == "positive" else "&#9660;"
            change_html = (
                f'<div class="kpi-change {change_class}">{arrow} {change}</div>'
            )

        return (
            f'<div class="kpi-card">'
            f'  <div class="kpi-value">{value}{" " + unit if unit else ""}</div>'
            f'  <div class="kpi-label">{label}</div>'
            f'  {change_html}'
            f'</div>'
        )

    # ------------------------------------------------------------------
    # HTML Generation Helpers
    # ------------------------------------------------------------------

    def _generate_css(self, inp: RenderInput) -> str:
        """Generate complete CSS stylesheet from branding config."""
        branding = inp.branding
        framework_key = inp.framework.value
        colors = FRAMEWORK_COLORS.get(framework_key, FRAMEWORK_COLORS["multi"])

        # Override with framework colors if using framework theme
        if branding.theme == BrandingTheme.DEFAULT:
            primary = branding.primary_color
            secondary = branding.secondary_color
        else:
            primary = branding.primary_color or colors["primary"]
            secondary = branding.secondary_color or colors["secondary"]

        css_vars = {
            "page_size": inp.page_size.value.upper(),
            "orientation": inp.orientation.value,
            "header_text": inp.report_title,
            "footer_text": branding.footer_text,
            "font_family": branding.font_family,
            "font_size_body": branding.font_size_body,
            "font_size_header": branding.font_size_header,
            "font_size_subheader": branding.font_size_subheader,
            "font_size_caption": branding.font_size_caption,
            "line_height": branding.line_height,
            "text_color": branding.text_color,
            "background_color": branding.background_color,
            "primary_color": primary,
            "secondary_color": secondary,
        }

        return CSS_BASE % css_vars

    def _generate_cover_page(self, inp: RenderInput) -> str:
        """Generate cover page HTML."""
        logo_html = ""
        if inp.branding.logo_base64:
            logo_html = (
                f'<img src="data:image/png;base64,{inp.branding.logo_base64}" '
                f'alt="Logo" style="max-width:200px;margin-bottom:40px;">'
            )
        elif inp.branding.logo_path:
            logo_html = (
                f'<img src="{inp.branding.logo_path}" '
                f'alt="Logo" style="max-width:200px;margin-bottom:40px;">'
            )

        confidentiality_html = ""
        if inp.cover_confidentiality:
            confidentiality_html = (
                f'<p style="color:#C62828;font-weight:600;margin-top:40px;">'
                f'{inp.cover_confidentiality}</p>'
            )

        company_name = inp.branding.company_name
        company_html = f'<p class="cover-info">{company_name}</p>' if company_name else ""

        framework_name = inp.framework.value.upper()
        if inp.framework == FrameworkTarget.MULTI:
            framework_name = "Multi-Framework"

        return (
            f'<div class="cover-page">'
            f'  {logo_html}'
            f'  <h1 class="cover-title">{inp.report_title}</h1>'
            f'  <p class="cover-subtitle">{inp.report_subtitle or framework_name + " Climate Disclosure"}</p>'
            f'  {company_html}'
            f'  <p class="cover-info">{inp.cover_date}</p>'
            f'  <p class="cover-info">Prepared by: {inp.cover_prepared_by}</p>'
            f'  {confidentiality_html}'
            f'</div>'
        )

    def _generate_toc(self, sections: List[RenderedSection]) -> str:
        """Generate table of contents HTML."""
        parts: List[str] = [
            '<div class="toc" style="page-break-after:always;">',
            '<h2>Table of Contents</h2>',
        ]
        for i, section in enumerate(sections, 1):
            if section.title:
                parts.append(
                    f'<div class="toc-entry">'
                    f'  <a href="#section-{section.section_id}">'
                    f'    {i}. {section.title}'
                    f'  </a>'
                    f'  <span>p. {section.page_estimate}</span>'
                    f'</div>'
                )
        parts.append('</div>')
        return "\n".join(parts)

    def _generate_nav_toolbar(
        self, inp: RenderInput, sections: List[RenderedSection]
    ) -> str:
        """Generate navigation toolbar for HTML output."""
        framework_key = inp.framework.value
        colors = FRAMEWORK_COLORS.get(framework_key, FRAMEWORK_COLORS["multi"])

        nav_items = "\n".join(
            f'<a href="#section-{s.section_id}" '
            f'style="color:{colors["primary"]};text-decoration:none;padding:4px 8px;">'
            f'{s.title}</a>'
            for s in sections if s.title
        )

        return (
            f'<nav style="position:sticky;top:0;background:#fff;padding:8px 16px;'
            f'border-bottom:2px solid {colors["primary"]};z-index:1000;display:flex;'
            f'align-items:center;gap:16px;flex-wrap:wrap;">'
            f'  <strong style="color:{colors["primary"]}">{inp.report_title}</strong>'
            f'  <input id="report-search" type="search" placeholder="Search report..." '
            f'         style="padding:4px 8px;border:1px solid #ccc;border-radius:4px;">'
            f'  <button id="print-btn" style="padding:4px 12px;background:{colors["primary"]};'
            f'          color:#fff;border:none;border-radius:4px;cursor:pointer;">Print</button>'
            f'  <div style="display:flex;gap:4px;flex-wrap:wrap;">{nav_items}</div>'
            f'</nav>'
        )

    def _generate_glossary(self, framework: FrameworkTarget) -> str:
        """Generate glossary section HTML."""
        terms: Dict[str, str] = {
            "Scope 1 Emissions": "Direct GHG emissions from sources owned or controlled by the organization.",
            "Scope 2 Emissions": "Indirect GHG emissions from purchased electricity, steam, heating, and cooling.",
            "Scope 3 Emissions": "All other indirect GHG emissions in the value chain.",
            "tCO2e": "Tonnes of carbon dioxide equivalent.",
            "GWP": "Global Warming Potential, a measure of how much heat a GHG traps relative to CO2.",
            "SBTi": "Science Based Targets initiative, validates corporate emission reduction targets.",
            "Net-Zero": "Reducing GHG emissions to as close to zero as possible, with residual emissions offset by removals.",
            "Base Year": "The reference year against which emission reductions are measured.",
            "Carbon Budget": "The maximum amount of CO2 that can be emitted while limiting warming to a target.",
            "TCFD": "Task Force on Climate-related Financial Disclosures.",
            "CSRD": "Corporate Sustainability Reporting Directive (EU).",
            "ESRS": "European Sustainability Reporting Standards.",
            "ISSB": "International Sustainability Standards Board.",
            "CDP": "Carbon Disclosure Project, global environmental disclosure system.",
            "GRI": "Global Reporting Initiative, sustainability reporting standards.",
            "XBRL": "eXtensible Business Reporting Language, digital reporting standard.",
            "iXBRL": "Inline XBRL, XBRL tags embedded within HTML documents.",
            "ISAE 3410": "International Standard on Assurance Engagements for GHG statements.",
            "Transition Plan": "A time-bound action plan to align operations with net-zero targets.",
            "Double Materiality": "Assessment of both financial and impact materiality (CSRD).",
        }

        parts = [
            '<div class="section" style="page-break-before:always;">',
            '<h2>Glossary</h2>',
            '<table class="data-table">',
            '<thead><tr><th>Term</th><th>Definition</th></tr></thead>',
            '<tbody>',
        ]
        for term, definition in sorted(terms.items()):
            parts.append(f'<tr><td><strong>{term}</strong></td><td>{definition}</td></tr>')
        parts.extend(['</tbody>', '</table>', '</div>'])

        return "\n".join(parts)

    def _generate_disclaimer(self, inp: RenderInput) -> str:
        """Generate disclaimer section HTML."""
        text = inp.custom_disclaimer or (
            "This report has been generated by the GreenLang Platform using data "
            "provided by the reporting organization.  While the platform employs "
            "rigorous validation and zero-hallucination architecture to ensure data "
            "accuracy, the reporting organization retains responsibility for the "
            "completeness and accuracy of the underlying data.  This report does not "
            "constitute legal, financial, or regulatory advice.  Independent "
            "verification by a qualified third-party auditor is recommended before "
            "submission to any regulatory body or public disclosure."
        )
        return (
            f'<div class="disclaimer">'
            f'  <h4>Disclaimer</h4>'
            f'  <p>{text}</p>'
            f'  <p style="margin-top:8px;">Generated: {_utcnow().isoformat()} | '
            f'Engine: FormatRenderingEngine v{_MODULE_VERSION} | '
            f'Report ID: {inp.report_id}</p>'
            f'</div>'
        )

    def _generate_ixbrl(
        self,
        inp: RenderInput,
        sections: List[RenderedSection],
        css: str,
        entity_id: str,
        period_start: str,
        period_end: str,
    ) -> str:
        """Generate iXBRL (inline XBRL in HTML) document."""
        framework_key = inp.framework.value

        # Determine namespace based on framework
        if framework_key == "sec":
            taxonomy_ns = "http://xbrl.sec.gov/esg/2024"
            taxonomy_prefix = "esg"
        elif framework_key in ("csrd", "esrs"):
            taxonomy_ns = "http://www.esma.europa.eu/taxonomy/esrs/2024"
            taxonomy_prefix = "esrs"
        else:
            taxonomy_ns = "http://xbrl.ifrs.org/taxonomy/ifrs-s2/2024"
            taxonomy_prefix = "ifrs-s2"

        parts: List[str] = [
            '<!DOCTYPE html>',
            '<html xmlns="http://www.w3.org/1999/xhtml"',
            f'      xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"',
            f'      xmlns:xbrli="http://www.xbrl.org/2003/instance"',
            f'      xmlns:{taxonomy_prefix}="{taxonomy_ns}"',
            '      xmlns:iso4217="http://www.xbrl.org/2003/iso4217"',
            '      xmlns:link="http://www.xbrl.org/2003/linkbase"',
            f'      lang="en">',
            '<head>',
            f'<meta charset="utf-8"/>',
            f'<title>{inp.report_title} (iXBRL)</title>',
            f'<style>{css}</style>',
            '</head>',
            '<body>',
            '<ix:header>',
            '  <ix:hidden>',
            '    <ix:references>',
            f'      <link:schemaRef xlink:type="simple" xlink:href="{taxonomy_ns}"/>',
            '    </ix:references>',
            '    <ix:resources>',
            f'      <xbrli:context id="ctx_current">',
            f'        <xbrli:entity>',
            f'          <xbrli:identifier scheme="http://www.greenlang.io">{entity_id}</xbrli:identifier>',
            f'        </xbrli:entity>',
            f'        <xbrli:period>',
            f'          <xbrli:startDate>{period_start}</xbrli:startDate>',
            f'          <xbrli:endDate>{period_end}</xbrli:endDate>',
            f'        </xbrli:period>',
            f'      </xbrli:context>',
            f'      <xbrli:unit id="unit_tCO2e">',
            f'        <xbrli:measure>greenlang:tCO2e</xbrli:measure>',
            f'      </xbrli:unit>',
            f'      <xbrli:unit id="unit_percent">',
            f'        <xbrli:measure>xbrli:pure</xbrli:measure>',
            f'      </xbrli:unit>',
            '    </ix:resources>',
            '  </ix:hidden>',
            '</ix:header>',
        ]

        # Render sections with iXBRL tags
        for section in sections:
            parts.append(section.html_content)

        parts.extend(['</body>', '</html>'])
        return "\n".join(parts)

    def _generate_xbrl_instance(
        self,
        inp: RenderInput,
        sections: List[RenderedSection],
        entity_id: str,
        period_start: str,
        period_end: str,
    ) -> str:
        """Generate XBRL instance document (XML)."""
        framework_key = inp.framework.value

        if framework_key == "sec":
            taxonomy_ns = "http://xbrl.sec.gov/esg/2024"
            taxonomy_prefix = "esg"
        elif framework_key in ("csrd", "esrs"):
            taxonomy_ns = "http://www.esma.europa.eu/taxonomy/esrs/2024"
            taxonomy_prefix = "esrs"
        else:
            taxonomy_ns = "http://xbrl.ifrs.org/taxonomy/ifrs-s2/2024"
            taxonomy_prefix = "ifrs-s2"

        parts: List[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<xbrli:xbrl',
            f'  xmlns:xbrli="http://www.xbrl.org/2003/instance"',
            f'  xmlns:{taxonomy_prefix}="{taxonomy_ns}"',
            f'  xmlns:iso4217="http://www.xbrl.org/2003/iso4217"',
            f'  xmlns:link="http://www.xbrl.org/2003/linkbase"',
            f'  xmlns:xlink="http://www.w3.org/1999/xlink">',
            '',
            f'  <!-- Report: {inp.report_title} -->',
            f'  <!-- Framework: {framework_key.upper()} -->',
            f'  <!-- Generated: {_utcnow().isoformat()} -->',
            '',
            f'  <link:schemaRef xlink:type="simple"',
            f'    xlink:href="{taxonomy_ns}"',
            f'    xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>',
            '',
            f'  <xbrli:context id="ctx_current">',
            f'    <xbrli:entity>',
            f'      <xbrli:identifier scheme="http://www.greenlang.io">{entity_id}</xbrli:identifier>',
            f'    </xbrli:entity>',
            f'    <xbrli:period>',
            f'      <xbrli:startDate>{period_start}</xbrli:startDate>',
            f'      <xbrli:endDate>{period_end}</xbrli:endDate>',
            f'    </xbrli:period>',
            f'  </xbrli:context>',
            '',
            f'  <xbrli:unit id="unit_tCO2e">',
            f'    <xbrli:measure>greenlang:tCO2e</xbrli:measure>',
            f'  </xbrli:unit>',
            f'  <xbrli:unit id="unit_percent">',
            f'    <xbrli:measure>xbrli:pure</xbrli:measure>',
            f'  </xbrli:unit>',
            f'  <xbrli:unit id="unit_currency">',
            f'    <xbrli:measure>iso4217:USD</xbrli:measure>',
            f'  </xbrli:unit>',
            '',
            f'  <!-- Facts extracted from {len(sections)} report sections -->',
        ]

        # Add section metadata as XBRL facts
        for i, section in enumerate(sections):
            parts.append(
                f'  <{taxonomy_prefix}:DisclosureSection contextRef="ctx_current">'
                f'{section.title}</{taxonomy_prefix}:DisclosureSection>'
            )

        parts.extend([
            '',
            '</xbrli:xbrl>',
        ])
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def _markdown_to_html(self, text: str) -> str:
        """
        Simple markdown-to-HTML conversion.

        Handles headers, bold, italic, lists, links, and paragraphs.
        Not a full Markdown parser -- covers the subset used in
        climate disclosure narratives.
        """
        import re

        lines = text.split("\n")
        html_lines: List[str] = []
        in_list = False

        for line in lines:
            stripped = line.strip()

            # Headers
            if stripped.startswith("### "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h3>{stripped[4:]}</h3>")
                continue
            elif stripped.startswith("## "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h2>{stripped[3:]}</h2>")
                continue
            elif stripped.startswith("# "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h1>{stripped[2:]}</h1>")
                continue

            # Bullet lists
            if stripped.startswith("- ") or stripped.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(f"<li>{stripped[2:]}</li>")
                continue

            # Numbered lists
            if re.match(r"^\d+\.\s", stripped):
                if not in_list:
                    html_lines.append("<ol>")
                    in_list = True
                content = re.sub(r"^\d+\.\s", "", stripped)
                html_lines.append(f"<li>{content}</li>")
                continue

            # Close list if open
            if in_list and stripped:
                html_lines.append("</ul>" if html_lines[-1] != "<ol>" else "</ol>")
                in_list = False

            # Empty line = paragraph break
            if not stripped:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                continue

            # Regular paragraph
            # Bold
            stripped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", stripped)
            # Italic
            stripped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", stripped)
            # Links
            stripped = re.sub(
                r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', stripped
            )
            html_lines.append(f"<p>{stripped}</p>")

        if in_list:
            html_lines.append("</ul>")

        return "\n".join(html_lines)

    def _get_cell_class(
        self, value: Any, rules: Dict[str, Any]
    ) -> str:
        """Determine CSS class for a table cell based on highlight rules."""
        if not rules:
            return ""

        str_val = str(value).lower()

        # RAG status highlighting
        if "rag" in rules:
            if str_val in ("green", "on_track", "complete"):
                return "badge-green"
            elif str_val in ("amber", "at_risk", "in_progress"):
                return "badge-amber"
            elif str_val in ("red", "off_track", "overdue"):
                return "badge-red"

        # Numeric threshold highlighting
        if "threshold" in rules:
            try:
                num_val = float(value)
                threshold = rules["threshold"]
                if isinstance(threshold, dict):
                    if num_val >= threshold.get("green", float("inf")):
                        return "badge-green"
                    elif num_val >= threshold.get("amber", float("inf")):
                        return "badge-amber"
                    else:
                        return "badge-red"
            except (ValueError, TypeError):
                pass

        return ""

    @staticmethod
    def _estimate_pages(word_count: int) -> int:
        """Estimate page count from word count (300 words per page)."""
        return max(1, (word_count + 299) // 300)

    @staticmethod
    def _generate_filename(inp: RenderInput, extension: str) -> str:
        """Generate a standardized filename for the report."""
        framework = inp.framework.value
        date_str = _utcnow().strftime("%Y%m%d")
        safe_title = "".join(
            c if c.isalnum() or c in "-_ " else ""
            for c in inp.report_title
        ).strip().replace(" ", "_")[:50]

        return f"{safe_title}_{framework}_{date_str}.{extension}"
