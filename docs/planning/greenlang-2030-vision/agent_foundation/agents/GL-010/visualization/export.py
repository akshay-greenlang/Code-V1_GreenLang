"""
GL-010 EMISSIONWATCH - Report Export Functionality

Report export module for the EmissionsComplianceAgent.
Supports PDF, PNG/SVG, Excel, JSON, HTML, and EPA CEDRI XML formats.

Author: GreenLang Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, BinaryIO
from enum import Enum
from datetime import datetime
from pathlib import Path
import json
import io
import base64
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from xml.dom import minidom


class ExportFormat(Enum):
    """Supported export formats."""
    PDF = "pdf"
    PNG = "png"
    SVG = "svg"
    EXCEL = "excel"
    JSON = "json"
    HTML = "html"
    XML = "xml"  # EPA CEDRI format
    CSV = "csv"


@dataclass
class ExportConfig:
    """Configuration for report exports."""
    title: str = "Emissions Compliance Report"
    subtitle: str = ""
    facility_name: str = ""
    facility_id: str = ""
    permit_number: str = ""
    reporting_period: str = ""
    prepared_by: str = ""
    prepared_date: str = ""
    page_size: str = "letter"  # letter, A4, legal
    orientation: str = "portrait"  # portrait, landscape
    include_charts: bool = True
    include_data_tables: bool = True
    include_summary: bool = True
    include_appendices: bool = False
    logo_path: Optional[str] = None
    color_scheme: str = "standard"  # standard, grayscale, high_contrast
    dpi: int = 300
    chart_width: int = 800
    chart_height: int = 600


@dataclass
class TableData:
    """Table data structure for exports."""
    headers: List[str]
    rows: List[List[Any]]
    title: str = ""
    footnotes: List[str] = field(default_factory=list)
    column_widths: Optional[List[int]] = None
    column_formats: Optional[List[str]] = None  # e.g., ["text", "number", "percent"]


class ExporterBase(ABC):
    """Base class for report exporters."""

    def __init__(self, config: ExportConfig):
        """
        Initialize exporter.

        Args:
            config: Export configuration
        """
        self.config = config
        self._charts: List[Dict[str, Any]] = []
        self._tables: List[TableData] = []
        self._summary: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}

    def add_chart(self, chart_data: Dict[str, Any], title: str = "") -> None:
        """Add chart to export."""
        self._charts.append({"data": chart_data, "title": title})

    def add_table(self, table: TableData) -> None:
        """Add table to export."""
        self._tables.append(table)

    def set_summary(self, summary: Dict[str, Any]) -> None:
        """Set summary data."""
        self._summary = summary

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set metadata."""
        self._metadata = metadata

    @abstractmethod
    def export(self, output_path: str) -> str:
        """Export report to file."""
        pass

    @abstractmethod
    def export_bytes(self) -> bytes:
        """Export report to bytes."""
        pass


class PDFExporter(ExporterBase):
    """PDF report exporter."""

    def export(self, output_path: str) -> str:
        """
        Export report to PDF file.

        Note: This generates HTML that can be converted to PDF
        using tools like wkhtmltopdf or WeasyPrint.

        Args:
            output_path: Output file path

        Returns:
            Output file path
        """
        html_content = self._generate_pdf_html()

        # Save HTML (can be converted to PDF externally)
        html_path = output_path.replace('.pdf', '.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_path

    def export_bytes(self) -> bytes:
        """Export report to bytes."""
        html_content = self._generate_pdf_html()
        return html_content.encode('utf-8')

    def _generate_pdf_html(self) -> str:
        """Generate PDF-ready HTML content."""
        charts_html = ""
        if self.config.include_charts:
            for idx, chart in enumerate(self._charts):
                charts_html += f"""
                <div class="chart-container">
                    <h3>{chart.get('title', f'Chart {idx + 1}')}</h3>
                    <div id="chart-{idx}" class="chart-placeholder">
                        <!-- Plotly chart would be rendered here -->
                        <script>
                            var chartData = {json.dumps(chart['data'])};
                            Plotly.newPlot('chart-{idx}', chartData.data, chartData.layout);
                        </script>
                    </div>
                </div>
                """

        tables_html = ""
        if self.config.include_data_tables:
            for table in self._tables:
                tables_html += self._render_table_html(table)

        summary_html = ""
        if self.config.include_summary and self._summary:
            summary_html = self._render_summary_html()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{self.config.title}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        @page {{
            size: {self.config.page_size} {self.config.orientation};
            margin: 1in;
        }}
        body {{
            font-family: 'Arial', sans-serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #333;
            margin: 0;
            padding: 0;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 24pt;
        }}
        .header h2 {{
            color: #7f8c8d;
            margin: 10px 0 0 0;
            font-size: 14pt;
            font-weight: normal;
        }}
        .facility-info {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 30px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .facility-info-item {{
            display: flex;
        }}
        .facility-info-item label {{
            font-weight: bold;
            width: 120px;
            color: #2c3e50;
        }}
        .summary-section {{
            margin-bottom: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .summary-card h4 {{
            margin: 0 0 10px 0;
            color: #7f8c8d;
            font-size: 10pt;
        }}
        .summary-card .value {{
            font-size: 24pt;
            font-weight: bold;
            color: #2c3e50;
        }}
        .chart-container {{
            page-break-inside: avoid;
            margin-bottom: 30px;
        }}
        .chart-container h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        .chart-placeholder {{
            width: 100%;
            height: 400px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            page-break-inside: avoid;
        }}
        table caption {{
            text-align: left;
            font-weight: bold;
            font-size: 12pt;
            padding: 10px 0;
            color: #2c3e50;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #2c3e50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .footnotes {{
            font-size: 9pt;
            color: #7f8c8d;
            margin-top: 10px;
        }}
        .footer {{
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            font-size: 9pt;
            color: #7f8c8d;
            padding: 10px 0;
            border-top: 1px solid #ddd;
        }}
        .page-break {{
            page-break-after: always;
        }}
        @media print {{
            .chart-container {{
                page-break-inside: avoid;
            }}
            table {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.config.title}</h1>
        {f'<h2>{self.config.subtitle}</h2>' if self.config.subtitle else ''}
    </div>

    <div class="facility-info">
        <div class="facility-info-item">
            <label>Facility:</label>
            <span>{self.config.facility_name}</span>
        </div>
        <div class="facility-info-item">
            <label>Facility ID:</label>
            <span>{self.config.facility_id}</span>
        </div>
        <div class="facility-info-item">
            <label>Permit Number:</label>
            <span>{self.config.permit_number}</span>
        </div>
        <div class="facility-info-item">
            <label>Reporting Period:</label>
            <span>{self.config.reporting_period}</span>
        </div>
        <div class="facility-info-item">
            <label>Prepared By:</label>
            <span>{self.config.prepared_by}</span>
        </div>
        <div class="facility-info-item">
            <label>Date:</label>
            <span>{self.config.prepared_date or datetime.now().strftime('%Y-%m-%d')}</span>
        </div>
    </div>

    {summary_html}

    {charts_html}

    {tables_html}

    <div class="footer">
        Generated by GreenLang EMISSIONWATCH | GL-010 | {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
</body>
</html>"""

        return html

    def _render_table_html(self, table: TableData) -> str:
        """Render table to HTML."""
        rows_html = ""
        for row in table.rows:
            cells = "".join(f"<td>{cell}</td>" for cell in row)
            rows_html += f"<tr>{cells}</tr>\n"

        footnotes_html = ""
        if table.footnotes:
            footnotes_html = "<div class='footnotes'>" + \
                            "<br>".join(table.footnotes) + "</div>"

        return f"""
        <table>
            <caption>{table.title}</caption>
            <thead>
                <tr>{"".join(f'<th>{h}</th>' for h in table.headers)}</tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        {footnotes_html}
        """

    def _render_summary_html(self) -> str:
        """Render summary section to HTML."""
        cards_html = ""
        for key, value in self._summary.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
            else:
                formatted_value = str(value)

            display_key = key.replace("_", " ").title()
            cards_html += f"""
            <div class="summary-card">
                <h4>{display_key}</h4>
                <div class="value">{formatted_value}</div>
            </div>
            """

        return f"""
        <div class="summary-section">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                {cards_html}
            </div>
        </div>
        """


class ImageExporter(ExporterBase):
    """PNG/SVG image exporter for charts."""

    def __init__(self, config: ExportConfig, format: str = "png"):
        """
        Initialize image exporter.

        Args:
            config: Export configuration
            format: Image format ("png" or "svg")
        """
        super().__init__(config)
        self.format = format

    def export(self, output_path: str) -> str:
        """
        Export charts to image files.

        Note: Requires plotly-orca or kaleido for actual image generation.
        This generates the configuration for image export.

        Args:
            output_path: Base output path

        Returns:
            List of output file paths as JSON string
        """
        output_files = []

        for idx, chart in enumerate(self._charts):
            file_path = f"{output_path}_chart_{idx}.{self.format}"
            export_config = {
                "file_path": file_path,
                "chart_data": chart["data"],
                "format": self.format,
                "width": self.config.chart_width,
                "height": self.config.chart_height,
                "scale": self.config.dpi / 96  # Convert DPI to scale
            }
            output_files.append(export_config)

        # Save export configuration
        config_path = f"{output_path}_export_config.json"
        with open(config_path, 'w') as f:
            json.dump(output_files, f, indent=2)

        return config_path

    def export_bytes(self) -> bytes:
        """Export configuration to bytes."""
        export_configs = []
        for idx, chart in enumerate(self._charts):
            export_configs.append({
                "index": idx,
                "chart_data": chart["data"],
                "format": self.format,
                "width": self.config.chart_width,
                "height": self.config.chart_height
            })
        return json.dumps(export_configs).encode('utf-8')


class ExcelExporter(ExporterBase):
    """Excel report exporter."""

    def export(self, output_path: str) -> str:
        """
        Export report to Excel file.

        Note: This generates CSV files that can be combined into Excel.
        For full Excel support, use openpyxl or xlsxwriter.

        Args:
            output_path: Output file path

        Returns:
            Output file path
        """
        # Generate workbook structure as JSON
        workbook = self._build_workbook()

        # Export to JSON (can be processed by external tools)
        json_path = output_path.replace('.xlsx', '.json').replace('.xls', '.json')
        with open(json_path, 'w') as f:
            json.dump(workbook, f, indent=2)

        # Also export CSV files for each sheet
        base_path = Path(output_path).stem
        output_dir = Path(output_path).parent

        for sheet_name, sheet_data in workbook["sheets"].items():
            csv_path = output_dir / f"{base_path}_{sheet_name}.csv"
            self._write_csv(csv_path, sheet_data)

        return json_path

    def export_bytes(self) -> bytes:
        """Export workbook to bytes (JSON format)."""
        workbook = self._build_workbook()
        return json.dumps(workbook).encode('utf-8')

    def _build_workbook(self) -> Dict[str, Any]:
        """Build workbook structure."""
        workbook = {
            "metadata": {
                "title": self.config.title,
                "facility": self.config.facility_name,
                "reporting_period": self.config.reporting_period,
                "generated": datetime.now().isoformat()
            },
            "sheets": {}
        }

        # Summary sheet
        if self.config.include_summary and self._summary:
            workbook["sheets"]["Summary"] = {
                "headers": ["Metric", "Value"],
                "rows": [[k, v] for k, v in self._summary.items()]
            }

        # Data tables
        for idx, table in enumerate(self._tables):
            sheet_name = table.title.replace(" ", "_")[:31] if table.title else f"Table_{idx + 1}"
            workbook["sheets"][sheet_name] = {
                "headers": table.headers,
                "rows": table.rows,
                "footnotes": table.footnotes
            }

        # Chart data
        for idx, chart in enumerate(self._charts):
            sheet_name = f"ChartData_{idx + 1}"
            chart_data = chart["data"]

            if "data" in chart_data and chart_data["data"]:
                trace = chart_data["data"][0]
                if "x" in trace and "y" in trace:
                    workbook["sheets"][sheet_name] = {
                        "headers": ["X", "Y"],
                        "rows": [[x, y] for x, y in zip(trace["x"], trace["y"])]
                    }

        return workbook

    def _write_csv(self, path: Path, sheet_data: Dict[str, Any]) -> None:
        """Write sheet data to CSV file."""
        with open(path, 'w', encoding='utf-8') as f:
            # Headers
            if "headers" in sheet_data:
                f.write(",".join(str(h) for h in sheet_data["headers"]) + "\n")

            # Rows
            if "rows" in sheet_data:
                for row in sheet_data["rows"]:
                    f.write(",".join(str(c) for c in row) + "\n")


class JSONExporter(ExporterBase):
    """JSON report exporter."""

    def export(self, output_path: str) -> str:
        """
        Export report to JSON file.

        Args:
            output_path: Output file path

        Returns:
            Output file path
        """
        report_data = self._build_report_data()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)

        return output_path

    def export_bytes(self) -> bytes:
        """Export report to bytes."""
        report_data = self._build_report_data()
        return json.dumps(report_data, indent=2, default=str).encode('utf-8')

    def _build_report_data(self) -> Dict[str, Any]:
        """Build complete report data structure."""
        return {
            "metadata": {
                "title": self.config.title,
                "subtitle": self.config.subtitle,
                "facility_name": self.config.facility_name,
                "facility_id": self.config.facility_id,
                "permit_number": self.config.permit_number,
                "reporting_period": self.config.reporting_period,
                "prepared_by": self.config.prepared_by,
                "generated_at": datetime.now().isoformat(),
                "generator": "GreenLang EMISSIONWATCH GL-010"
            },
            "summary": self._summary,
            "charts": [
                {"title": c.get("title", ""), "data": c["data"]}
                for c in self._charts
            ],
            "tables": [
                {
                    "title": t.title,
                    "headers": t.headers,
                    "rows": t.rows,
                    "footnotes": t.footnotes
                }
                for t in self._tables
            ],
            "additional_metadata": self._metadata
        }


class HTMLExporter(ExporterBase):
    """HTML standalone report exporter."""

    def export(self, output_path: str) -> str:
        """
        Export report to standalone HTML file.

        Args:
            output_path: Output file path

        Returns:
            Output file path
        """
        html_content = self._generate_html()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def export_bytes(self) -> bytes:
        """Export report to bytes."""
        return self._generate_html().encode('utf-8')

    def _generate_html(self) -> str:
        """Generate complete HTML report."""
        # Reuse PDF exporter's HTML generation with modifications
        pdf_exporter = PDFExporter(self.config)
        pdf_exporter._charts = self._charts
        pdf_exporter._tables = self._tables
        pdf_exporter._summary = self._summary

        return pdf_exporter._generate_pdf_html()


class CEDRIXMLExporter(ExporterBase):
    """
    EPA CEDRI (Compliance and Emissions Data Reporting Interface) XML exporter.

    Generates XML attachments for EPA electronic reporting.
    """

    def __init__(self, config: ExportConfig):
        """Initialize CEDRI XML exporter."""
        super().__init__(config)
        self._emissions_data: List[Dict[str, Any]] = []
        self._violations_data: List[Dict[str, Any]] = []
        self._monitoring_data: List[Dict[str, Any]] = []

    def add_emissions_data(self, data: List[Dict[str, Any]]) -> None:
        """Add emissions data for XML export."""
        self._emissions_data = data

    def add_violations_data(self, data: List[Dict[str, Any]]) -> None:
        """Add violations data for XML export."""
        self._violations_data = data

    def add_monitoring_data(self, data: List[Dict[str, Any]]) -> None:
        """Add monitoring data for XML export."""
        self._monitoring_data = data

    def export(self, output_path: str) -> str:
        """
        Export report to EPA CEDRI XML file.

        Args:
            output_path: Output file path

        Returns:
            Output file path
        """
        xml_content = self._build_cedri_xml()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)

        return output_path

    def export_bytes(self) -> bytes:
        """Export XML to bytes."""
        return self._build_cedri_xml().encode('utf-8')

    def _build_cedri_xml(self) -> str:
        """Build EPA CEDRI-compliant XML structure."""
        # Create root element
        root = ET.Element("CEDRISubmission")
        root.set("xmlns", "http://www.exchangenetwork.net/schema/cedri/1")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")

        # Header information
        header = ET.SubElement(root, "Header")

        submission_info = ET.SubElement(header, "SubmissionInfo")
        ET.SubElement(submission_info, "SubmissionType").text = "Compliance Report"
        ET.SubElement(submission_info, "SubmissionDate").text = datetime.now().strftime("%Y-%m-%d")
        ET.SubElement(submission_info, "ReportingPeriod").text = self.config.reporting_period

        facility_info = ET.SubElement(header, "FacilityInfo")
        ET.SubElement(facility_info, "FacilityName").text = self.config.facility_name
        ET.SubElement(facility_info, "FacilityID").text = self.config.facility_id
        ET.SubElement(facility_info, "PermitNumber").text = self.config.permit_number

        preparer_info = ET.SubElement(header, "PreparerInfo")
        ET.SubElement(preparer_info, "PreparedBy").text = self.config.prepared_by
        ET.SubElement(preparer_info, "PreparedDate").text = self.config.prepared_date or \
            datetime.now().strftime("%Y-%m-%d")

        # Emissions data section
        if self._emissions_data:
            emissions = ET.SubElement(root, "EmissionsData")
            for record in self._emissions_data:
                emission = ET.SubElement(emissions, "EmissionRecord")
                ET.SubElement(emission, "Pollutant").text = record.get("pollutant", "")
                ET.SubElement(emission, "EmissionValue").text = str(record.get("value", 0))
                ET.SubElement(emission, "Unit").text = record.get("unit", "")
                ET.SubElement(emission, "AveragingPeriod").text = record.get("averaging_period", "")
                ET.SubElement(emission, "PermitLimit").text = str(record.get("permit_limit", 0))
                ET.SubElement(emission, "Timestamp").text = record.get("timestamp", "")
                ET.SubElement(emission, "DataQuality").text = str(record.get("data_quality", 100))

        # Violations section
        if self._violations_data:
            violations = ET.SubElement(root, "Violations")
            for record in self._violations_data:
                violation = ET.SubElement(violations, "ViolationRecord")
                ET.SubElement(violation, "ViolationID").text = record.get("violation_id", "")
                ET.SubElement(violation, "ViolationType").text = record.get("violation_type", "")
                ET.SubElement(violation, "Pollutant").text = record.get("pollutant", "")
                ET.SubElement(violation, "StartTime").text = record.get("start_time", "")
                ET.SubElement(violation, "EndTime").text = record.get("end_time", "")
                ET.SubElement(violation, "Duration").text = str(record.get("duration_minutes", 0))
                ET.SubElement(violation, "ExceedanceValue").text = str(record.get("exceedance_value", 0))
                ET.SubElement(violation, "PermitLimit").text = str(record.get("permit_limit", 0))
                ET.SubElement(violation, "Severity").text = record.get("severity", "")
                ET.SubElement(violation, "RootCause").text = record.get("root_cause", "")
                ET.SubElement(violation, "CorrectiveAction").text = record.get("corrective_action", "")

        # Monitoring data section
        if self._monitoring_data:
            monitoring = ET.SubElement(root, "MonitoringData")
            for record in self._monitoring_data:
                monitor = ET.SubElement(monitoring, "MonitorRecord")
                ET.SubElement(monitor, "MonitorID").text = record.get("monitor_id", "")
                ET.SubElement(monitor, "MonitorType").text = record.get("monitor_type", "")
                ET.SubElement(monitor, "DataAvailability").text = str(record.get("data_availability", 0))
                ET.SubElement(monitor, "CalibrationStatus").text = record.get("calibration_status", "")
                ET.SubElement(monitor, "LastCalibration").text = record.get("last_calibration", "")

        # Summary section
        if self._summary:
            summary = ET.SubElement(root, "Summary")
            for key, value in self._summary.items():
                elem = ET.SubElement(summary, self._sanitize_xml_tag(key))
                elem.text = str(value)

        # Pretty print
        xml_string = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_string)
        return dom.toprettyxml(indent="  ")

    def _sanitize_xml_tag(self, tag: str) -> str:
        """Sanitize string to be valid XML tag."""
        # Replace spaces and special characters
        tag = tag.replace(" ", "_").replace("-", "_")
        # Remove invalid characters
        tag = ''.join(c for c in tag if c.isalnum() or c == '_')
        # Ensure starts with letter
        if tag and not tag[0].isalpha():
            tag = "tag_" + tag
        return tag or "unknown"


class ReportExporter:
    """
    Main report exporter class that delegates to format-specific exporters.
    """

    EXPORTERS = {
        ExportFormat.PDF: PDFExporter,
        ExportFormat.HTML: HTMLExporter,
        ExportFormat.JSON: JSONExporter,
        ExportFormat.EXCEL: ExcelExporter,
        ExportFormat.PNG: lambda config: ImageExporter(config, "png"),
        ExportFormat.SVG: lambda config: ImageExporter(config, "svg"),
        ExportFormat.XML: CEDRIXMLExporter
    }

    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize report exporter.

        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
        self._charts: List[Dict[str, Any]] = []
        self._tables: List[TableData] = []
        self._summary: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        self._emissions_data: List[Dict[str, Any]] = []
        self._violations_data: List[Dict[str, Any]] = []

    def add_chart(self, chart_data: Dict[str, Any], title: str = "") -> 'ReportExporter':
        """Add chart to report. Returns self for chaining."""
        self._charts.append({"data": chart_data, "title": title})
        return self

    def add_table(self, table: TableData) -> 'ReportExporter':
        """Add table to report. Returns self for chaining."""
        self._tables.append(table)
        return self

    def set_summary(self, summary: Dict[str, Any]) -> 'ReportExporter':
        """Set summary data. Returns self for chaining."""
        self._summary = summary
        return self

    def set_metadata(self, metadata: Dict[str, Any]) -> 'ReportExporter':
        """Set additional metadata. Returns self for chaining."""
        self._metadata = metadata
        return self

    def add_emissions_data(self, data: List[Dict[str, Any]]) -> 'ReportExporter':
        """Add emissions data (for XML export). Returns self for chaining."""
        self._emissions_data = data
        return self

    def add_violations_data(self, data: List[Dict[str, Any]]) -> 'ReportExporter':
        """Add violations data (for XML export). Returns self for chaining."""
        self._violations_data = data
        return self

    def export(
        self,
        output_path: str,
        format: ExportFormat
    ) -> str:
        """
        Export report to specified format.

        Args:
            output_path: Output file path
            format: Export format

        Returns:
            Output file path
        """
        exporter_class = self.EXPORTERS.get(format)
        if not exporter_class:
            raise ValueError(f"Unsupported export format: {format}")

        if callable(exporter_class) and not isinstance(exporter_class, type):
            exporter = exporter_class(self.config)
        else:
            exporter = exporter_class(self.config)

        # Transfer data to exporter
        for chart in self._charts:
            exporter.add_chart(chart["data"], chart.get("title", ""))
        for table in self._tables:
            exporter.add_table(table)
        exporter.set_summary(self._summary)
        exporter.set_metadata(self._metadata)

        # Handle CEDRI XML specific data
        if format == ExportFormat.XML and isinstance(exporter, CEDRIXMLExporter):
            exporter.add_emissions_data(self._emissions_data)
            exporter.add_violations_data(self._violations_data)

        return exporter.export(output_path)

    def export_bytes(self, format: ExportFormat) -> bytes:
        """
        Export report to bytes.

        Args:
            format: Export format

        Returns:
            Report content as bytes
        """
        exporter_class = self.EXPORTERS.get(format)
        if not exporter_class:
            raise ValueError(f"Unsupported export format: {format}")

        if callable(exporter_class) and not isinstance(exporter_class, type):
            exporter = exporter_class(self.config)
        else:
            exporter = exporter_class(self.config)

        for chart in self._charts:
            exporter.add_chart(chart["data"], chart.get("title", ""))
        for table in self._tables:
            exporter.add_table(table)
        exporter.set_summary(self._summary)
        exporter.set_metadata(self._metadata)

        if format == ExportFormat.XML and isinstance(exporter, CEDRIXMLExporter):
            exporter.add_emissions_data(self._emissions_data)
            exporter.add_violations_data(self._violations_data)

        return exporter.export_bytes()

    def export_all(
        self,
        base_path: str,
        formats: Optional[List[ExportFormat]] = None
    ) -> Dict[ExportFormat, str]:
        """
        Export report to multiple formats.

        Args:
            base_path: Base output path (without extension)
            formats: List of formats (default: all)

        Returns:
            Dictionary of format to output path
        """
        formats = formats or list(ExportFormat)
        results = {}

        extension_map = {
            ExportFormat.PDF: ".pdf",
            ExportFormat.HTML: ".html",
            ExportFormat.JSON: ".json",
            ExportFormat.EXCEL: ".xlsx",
            ExportFormat.PNG: ".png",
            ExportFormat.SVG: ".svg",
            ExportFormat.XML: ".xml",
            ExportFormat.CSV: ".csv"
        }

        for format in formats:
            if format == ExportFormat.CSV:
                continue  # CSV is handled by Excel exporter

            ext = extension_map.get(format, "")
            output_path = f"{base_path}{ext}"

            try:
                results[format] = self.export(output_path, format)
            except Exception as e:
                results[format] = f"Error: {str(e)}"

        return results


def create_sample_report() -> ReportExporter:
    """
    Create a sample report for testing.

    Returns:
        Configured ReportExporter
    """
    config = ExportConfig(
        title="Emissions Compliance Report",
        subtitle="Q1 2024 Quarterly Report",
        facility_name="GreenPower Plant Alpha",
        facility_id="FAC-001",
        permit_number="SCAQMD-12345",
        reporting_period="January 1 - March 31, 2024",
        prepared_by="Environmental Compliance Team"
    )

    exporter = ReportExporter(config)

    # Add summary
    exporter.set_summary({
        "total_emissions": 15234.5,
        "violations": 2,
        "compliance_rate": 98.5,
        "data_availability": 99.2
    })

    # Add sample table
    table = TableData(
        title="Emissions Summary by Pollutant",
        headers=["Pollutant", "Emissions (tons)", "Limit (tons)", "Margin (%)", "Status"],
        rows=[
            ["NOx", "1,234.5", "2,000.0", "38.3", "Compliant"],
            ["SO2", "567.8", "1,000.0", "43.2", "Compliant"],
            ["PM", "123.4", "250.0", "50.6", "Compliant"],
            ["CO", "2,345.6", "3,000.0", "21.8", "Compliant"]
        ],
        footnotes=["All values are tons per quarter", "Data availability: 99.2%"]
    )
    exporter.add_table(table)

    # Add sample emissions data for XML
    exporter.add_emissions_data([
        {
            "pollutant": "NOx",
            "value": 1234.5,
            "unit": "tons",
            "averaging_period": "quarterly",
            "permit_limit": 2000.0,
            "timestamp": "2024-03-31T23:59:59Z",
            "data_quality": 99.2
        }
    ])

    return exporter


if __name__ == "__main__":
    # Demo usage
    exporter = create_sample_report()

    # Export to JSON
    json_output = exporter.export_bytes(ExportFormat.JSON)
    print("JSON Export (first 500 chars):")
    print(json_output.decode('utf-8')[:500])

    # Export to XML
    xml_output = exporter.export_bytes(ExportFormat.XML)
    print("\n\nXML Export (first 500 chars):")
    print(xml_output.decode('utf-8')[:500])
