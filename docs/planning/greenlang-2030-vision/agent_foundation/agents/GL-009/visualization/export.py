"""Export Utilities for GL-009 THERMALIQ Visualizations.

Provides multi-format export capabilities for all visualization types:
- HTML (interactive Plotly)
- PNG (static image)
- SVG (vector graphics)
- JSON (raw data)
- Dashboard (multi-chart HTML)

Features:
- Standalone HTML with embedded Plotly.js
- High-resolution image export
- Scalable vector graphics
- Data provenance preservation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
import json
import base64
from datetime import datetime


class ExportFormat(Enum):
    """Export format types."""
    HTML = "html"
    PNG = "png"
    SVG = "svg"
    JSON = "json"
    PDF = "pdf"


@dataclass
class ExportConfig:
    """Export configuration."""
    width: int = 1200
    height: int = 800
    dpi: int = 300
    include_plotlyjs: bool = True
    auto_open: bool = False
    config: Dict[str, Any] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "thermal_efficiency_chart",
                    "height": self.height,
                    "width": self.width,
                    "scale": self.dpi / 96
                }
            }


class VisualizationExporter:
    """Export visualizations to multiple formats."""

    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize exporter.

        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()

    def export(
        self,
        figure: Dict,
        output_path: Union[str, Path],
        format: ExportFormat,
        title: Optional[str] = None
    ) -> Path:
        """Export visualization to file.

        Args:
            figure: Plotly figure dictionary
            output_path: Output file path
            format: Export format
            title: Optional title for HTML exports

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)

        if format == ExportFormat.HTML:
            return self._export_html(figure, output_path, title)
        elif format == ExportFormat.JSON:
            return self._export_json(figure, output_path)
        elif format == ExportFormat.PNG:
            return self._export_static_image(figure, output_path, "png")
        elif format == ExportFormat.SVG:
            return self._export_static_image(figure, output_path, "svg")
        elif format == ExportFormat.PDF:
            return self._export_pdf(figure, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_html(
        self,
        figure: Dict,
        output_path: Path,
        title: Optional[str] = None
    ) -> Path:
        """Export to standalone HTML."""
        # Ensure .html extension
        if output_path.suffix.lower() != ".html":
            output_path = output_path.with_suffix(".html")

        # Build HTML
        html_template = self._get_html_template(
            figure,
            title or "THERMALIQ Energy Flow Visualization",
            self.config
        )

        # Write to file
        output_path.write_text(html_template, encoding="utf-8")
        return output_path

    def _export_json(self, figure: Dict, output_path: Path) -> Path:
        """Export to JSON."""
        # Ensure .json extension
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")

        # Add metadata
        export_data = {
            "figure": figure,
            "export_timestamp": datetime.utcnow().isoformat(),
            "export_config": {
                "width": self.config.width,
                "height": self.config.height
            }
        }

        # Write to file
        output_path.write_text(
            json.dumps(export_data, indent=2),
            encoding="utf-8"
        )
        return output_path

    def _export_static_image(
        self,
        figure: Dict,
        output_path: Path,
        format: str
    ) -> Path:
        """Export to static image (PNG/SVG).

        Note: This creates an HTML file with download instructions
        as static image export requires browser-based rendering or kaleido.
        """
        # Ensure correct extension
        output_path = output_path.with_suffix(f".{format}")

        # Create instructions HTML
        instructions = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Export to {format.upper()}</title>
            <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
        </head>
        <body>
            <h2>Static Image Export</h2>
            <p>To export as {format.upper()}, use the camera icon in the chart toolbar.</p>
            <div id="chart"></div>
            <script>
                var figure = {json.dumps(figure)};
                var config = {{
                    displayModeBar: true,
                    toImageButtonOptions: {{
                        format: '{format}',
                        filename: '{output_path.stem}',
                        height: {self.config.height},
                        width: {self.config.width},
                        scale: {self.config.dpi / 96}
                    }}
                }};
                Plotly.newPlot('chart', figure.data, figure.layout, config);
            </script>
            <p style="margin-top: 20px; color: #666;">
                Note: For programmatic {format.upper()} export, consider using kaleido or orca.
            </p>
        </body>
        </html>
        """

        html_path = output_path.with_suffix(".html")
        html_path.write_text(instructions, encoding="utf-8")

        return html_path

    def _export_pdf(self, figure: Dict, output_path: Path) -> Path:
        """Export to PDF.

        Note: PDF export requires additional libraries (kaleido or wkhtmltopdf).
        This method creates an HTML that can be printed to PDF.
        """
        # Create print-optimized HTML
        output_path = output_path.with_suffix(".html")

        html_template = self._get_html_template(
            figure,
            "THERMALIQ Energy Flow - Print to PDF",
            self.config,
            print_mode=True
        )

        output_path.write_text(html_template, encoding="utf-8")
        return output_path

    def _get_html_template(
        self,
        figure: Dict,
        title: str,
        config: ExportConfig,
        print_mode: bool = False
    ) -> str:
        """Generate HTML template."""
        # Plotly CDN or embedded
        if config.include_plotlyjs:
            plotly_js = '<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>'
        else:
            plotly_js = ''

        # Print mode CSS
        print_css = """
        @media print {
            body { margin: 0; }
            .no-print { display: none; }
        }
        """ if print_mode else ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {plotly_js}
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: {config.width + 40}px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-top: 0;
            font-size: 24px;
            border-bottom: 2px solid #3498DB;
            padding-bottom: 10px;
        }}
        #chart {{
            width: 100%;
            height: {config.height}px;
        }}
        .footer {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        .metadata-item {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
        }}
        .metadata-label {{
            font-weight: bold;
            color: #555;
            font-size: 11px;
            text-transform: uppercase;
        }}
        .metadata-value {{
            color: #333;
            font-size: 14px;
            margin-top: 5px;
        }}
        {print_css}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div id="chart"></div>
        <div class="footer no-print">
            <div class="metadata">
                <div class="metadata-item">
                    <div class="metadata-label">Generated</div>
                    <div class="metadata-value">{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">System</div>
                    <div class="metadata-value">GL-009 THERMALIQ</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Chart Type</div>
                    <div class="metadata-value">{figure.get('data', [{}])[0].get('type', 'unknown').title()}</div>
                </div>
            </div>
        </div>
    </div>
    <script>
        var figure = {json.dumps(figure)};
        var config = {json.dumps(config.config)};
        Plotly.newPlot('chart', figure.data, figure.layout, config);
    </script>
</body>
</html>"""

    def export_dashboard(
        self,
        figures: List[Dict],
        output_path: Union[str, Path],
        title: str = "THERMALIQ Energy Dashboard",
        layout: str = "grid"
    ) -> Path:
        """Export multiple charts as dashboard.

        Args:
            figures: List of Plotly figures
            output_path: Output HTML path
            title: Dashboard title
            layout: Layout mode ("grid", "vertical", "tabs")

        Returns:
            Path to dashboard HTML
        """
        output_path = Path(output_path).with_suffix(".html")

        if layout == "grid":
            html = self._create_grid_dashboard(figures, title)
        elif layout == "vertical":
            html = self._create_vertical_dashboard(figures, title)
        elif layout == "tabs":
            html = self._create_tabbed_dashboard(figures, title)
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        output_path.write_text(html, encoding="utf-8")
        return output_path

    def _create_grid_dashboard(self, figures: List[Dict], title: str) -> str:
        """Create grid layout dashboard."""
        charts_html = ""
        for i, fig in enumerate(figures):
            charts_html += f"""
            <div class="chart-container">
                <div id="chart{i}"></div>
            </div>
            """

        scripts = ""
        for i, fig in enumerate(figures):
            scripts += f"""
            Plotly.newPlot('chart{i}', {json.dumps(fig['data'])}, {json.dumps(fig['layout'])}, dashboardConfig);
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 28px;
        }}
        .charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
        }}
        .chart-container {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>{title}</h1>
        <div class="charts">
            {charts_html}
        </div>
    </div>
    <script>
        var dashboardConfig = {{
            displayModeBar: true,
            displaylogo: false,
            responsive: true
        }};
        {scripts}
    </script>
</body>
</html>"""

    def _create_vertical_dashboard(self, figures: List[Dict], title: str) -> str:
        """Create vertical stacked dashboard."""
        # Similar to grid but single column
        return self._create_grid_dashboard(figures, title).replace(
            "grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));",
            "grid-template-columns: 1fr;"
        )

    def _create_tabbed_dashboard(self, figures: List[Dict], title: str) -> str:
        """Create tabbed dashboard."""
        tabs_html = ""
        for i, fig in enumerate(figures):
            chart_title = fig.get('layout', {}).get('title', {})
            if isinstance(chart_title, dict):
                tab_name = chart_title.get('text', f'Chart {i+1}')
            else:
                tab_name = chart_title or f'Chart {i+1}'

            tabs_html += f'<button class="tab-button" onclick="showTab({i})">{tab_name}</button>\n'

        charts_html = ""
        for i, fig in enumerate(figures):
            active = "active" if i == 0 else ""
            charts_html += f"""
            <div class="tab-content {active}" id="tab{i}">
                <div id="chart{i}"></div>
            </div>
            """

        scripts = ""
        for i, fig in enumerate(figures):
            scripts += f"""
            Plotly.newPlot('chart{i}', {json.dumps(fig['data'])}, {json.dumps(fig['layout'])}, dashboardConfig);
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            padding: 20px;
            margin: 0;
            border-bottom: 2px solid #3498DB;
        }}
        .tabs {{
            display: flex;
            border-bottom: 1px solid #ddd;
            background-color: #f8f9fa;
        }}
        .tab-button {{
            padding: 15px 25px;
            border: none;
            background: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            color: #666;
            transition: all 0.3s;
        }}
        .tab-button:hover {{
            background-color: #e9ecef;
            color: #333;
        }}
        .tab-button.active {{
            color: #3498DB;
            border-bottom: 3px solid #3498DB;
        }}
        .tab-content {{
            display: none;
            padding: 20px;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>{title}</h1>
        <div class="tabs">
            {tabs_html}
        </div>
        {charts_html}
    </div>
    <script>
        var dashboardConfig = {{
            displayModeBar: true,
            displaylogo: false,
            responsive: true
        }};

        {scripts}

        function showTab(index) {{
            // Hide all tabs
            var contents = document.querySelectorAll('.tab-content');
            var buttons = document.querySelectorAll('.tab-button');

            contents.forEach(content => content.classList.remove('active'));
            buttons.forEach(button => button.classList.remove('active'));

            // Show selected tab
            document.getElementById('tab' + index).classList.add('active');
            buttons[index].classList.add('active');
        }}

        // Activate first tab
        showTab(0);
    </script>
</body>
</html>"""


# Convenience functions

def export_to_html(
    figure: Dict,
    output_path: Union[str, Path],
    title: Optional[str] = None,
    config: Optional[ExportConfig] = None
) -> Path:
    """Export figure to HTML."""
    exporter = VisualizationExporter(config)
    return exporter.export(figure, output_path, ExportFormat.HTML, title)


def export_to_json(
    figure: Dict,
    output_path: Union[str, Path],
    config: Optional[ExportConfig] = None
) -> Path:
    """Export figure to JSON."""
    exporter = VisualizationExporter(config)
    return exporter.export(figure, output_path, ExportFormat.JSON)


def export_to_png(
    figure: Dict,
    output_path: Union[str, Path],
    config: Optional[ExportConfig] = None
) -> Path:
    """Export figure to PNG (creates HTML with download instructions)."""
    exporter = VisualizationExporter(config)
    return exporter.export(figure, output_path, ExportFormat.PNG)


def export_to_svg(
    figure: Dict,
    output_path: Union[str, Path],
    config: Optional[ExportConfig] = None
) -> Path:
    """Export figure to SVG (creates HTML with download instructions)."""
    exporter = VisualizationExporter(config)
    return exporter.export(figure, output_path, ExportFormat.SVG)


def export_dashboard(
    figures: List[Dict],
    output_path: Union[str, Path],
    title: str = "THERMALIQ Energy Dashboard",
    layout: str = "grid",
    config: Optional[ExportConfig] = None
) -> Path:
    """Export multiple figures as dashboard."""
    exporter = VisualizationExporter(config)
    return exporter.export_dashboard(figures, output_path, title, layout)
