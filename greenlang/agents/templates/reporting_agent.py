# -*- coding: utf-8 -*-
"""
Reporting Agent Template
Multi-format Export with Compliance Checking

Base agent template for reporting in sustainability applications.
Supports JSON, Excel, PDF, XBRL export with template rendering and validation.

Version: 1.0.0
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import json
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Supported report formats."""
    JSON = "json"
    EXCEL = "excel"
    PDF = "pdf"
    XBRL = "xbrl"
    CSV = "csv"
    HTML = "html"
    PARQUET = "parquet"
    XML = "xml"
    MARKDOWN = "markdown"
    YAML = "yaml"


class ComplianceFramework(str, Enum):
    """Compliance frameworks."""
    GHG_PROTOCOL = "ghg_protocol"
    CSRD = "csrd"
    CBAM = "cbam"
    ISO_14064 = "iso_14064"
    CDP = "cdp"
    TCFD = "tcfd"


@dataclass
class ComplianceCheck:
    """Compliance check result."""
    framework: ComplianceFramework
    passed: bool
    issues: List[str]
    score: Optional[float] = None


@dataclass
class ReportResult:
    """Result of report generation."""
    success: bool
    file_path: Optional[str] = None
    data: Optional[Union[bytes, str]] = None
    format: Optional[ReportFormat] = None
    compliance_checks: Optional[List[ComplianceCheck]] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.compliance_checks is None:
            self.compliance_checks = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = DeterministicClock.utcnow()


class ReportingAgent:
    """
    Base Reporting Agent Template.

    Provides common reporting patterns:
    - Multi-format export (JSON, Excel, PDF, XBRL, CSV, HTML, Parquet, XML, Markdown, YAML)
    - Template rendering
    - Visualization generation
    - Compliance checking
    - Data validation
    - Charts and graphics generation
    """

    def __init__(
        self,
        templates: Optional[Dict[str, Any]] = None,
        compliance_rules: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Reporting Agent.

        Args:
            templates: Report templates
            compliance_rules: Compliance validation rules
            config: Agent configuration
        """
        self.templates = templates or {}
        self.compliance_rules = compliance_rules or {}
        self.config = config or {}

        self._stats = {
            "total_reports": 0,
            "successful_reports": 0,
            "failed_reports": 0,
            "reports_by_format": {},
        }

        logger.info("Initialized ReportingAgent")

    async def generate_report(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        format: ReportFormat,
        template_name: Optional[str] = None,
        output_path: Optional[str] = None,
        check_compliance: Optional[List[ComplianceFramework]] = None,
    ) -> ReportResult:
        """
        Generate report from data.

        Args:
            data: Data to report (DataFrame, dict, or list)
            format: Output format
            template_name: Template to use
            output_path: Path to save report
            check_compliance: Frameworks to check compliance against

        Returns:
            ReportResult with generated report
        """
        self._stats["total_reports"] += 1

        # Track format usage
        format_key = format.value
        self._stats["reports_by_format"][format_key] = (
            self._stats["reports_by_format"].get(format_key, 0) + 1
        )

        try:
            # Step 1: Validate data
            validation_errors = self._validate_data(data)
            if validation_errors:
                return ReportResult(
                    success=False,
                    errors=validation_errors,
                    format=format
                )

            # Step 2: Apply template if specified
            if template_name and template_name in self.templates:
                data = self._apply_template(data, template_name)

            # Step 3: Generate report in specified format
            report_data = await self._generate_format(data, format)

            # Step 4: Save to file if path provided
            file_path = None
            if output_path:
                file_path = self._save_report(report_data, output_path, format)

            # Step 5: Compliance checks
            compliance_checks = []
            if check_compliance:
                for framework in check_compliance:
                    check = self._check_compliance(data, framework)
                    compliance_checks.append(check)

            # Determine overall success
            compliance_passed = all(c.passed for c in compliance_checks) if compliance_checks else True
            success = compliance_passed

            if success:
                self._stats["successful_reports"] += 1
            else:
                self._stats["failed_reports"] += 1

            return ReportResult(
                success=success,
                file_path=file_path,
                data=report_data,
                format=format,
                compliance_checks=compliance_checks,
                metadata={
                    "template": template_name,
                    "data_rows": self._get_row_count(data),
                }
            )

        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            self._stats["failed_reports"] += 1

            return ReportResult(
                success=False,
                format=format,
                errors=[f"Report generation failed: {str(e)}"]
            )

    def _validate_data(self, data: Any) -> List[str]:
        """Validate report data."""
        errors = []

        # Check data is not empty
        if data is None:
            errors.append("Data cannot be None")
        elif isinstance(data, pd.DataFrame) and data.empty:
            errors.append("DataFrame is empty")
        elif isinstance(data, (list, dict)) and len(data) == 0:
            errors.append("Data is empty")

        return errors

    def _apply_template(self, data: Any, template_name: str) -> Any:
        """Apply report template to data."""
        template = self.templates.get(template_name)

        if not template:
            logger.warning(f"Template {template_name} not found")
            return data

        # Template application logic would go here
        # For now, return data unchanged
        return data

    async def _generate_format(
        self,
        data: Any,
        format: ReportFormat
    ) -> Union[bytes, str]:
        """Generate report in specified format."""
        if format == ReportFormat.JSON:
            return self._generate_json(data)
        elif format == ReportFormat.EXCEL:
            return await self._generate_excel(data)
        elif format == ReportFormat.CSV:
            return self._generate_csv(data)
        elif format == ReportFormat.HTML:
            return self._generate_html(data)
        elif format == ReportFormat.PDF:
            return await self._generate_pdf(data)
        elif format == ReportFormat.XBRL:
            return await self._generate_xbrl(data)
        elif format == ReportFormat.PARQUET:
            return self._generate_parquet(data)
        elif format == ReportFormat.XML:
            return self._generate_xml(data)
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown(data)
        elif format == ReportFormat.YAML:
            return self._generate_yaml(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_json(self, data: Any) -> str:
        """Generate JSON report."""
        if isinstance(data, pd.DataFrame):
            return data.to_json(orient='records', indent=2)
        else:
            return json.dumps(data, indent=2, default=str)

    async def _generate_excel(self, data: Any) -> bytes:
        """Generate Excel report."""
        import io

        if isinstance(data, pd.DataFrame):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                data.to_excel(writer, index=False, sheet_name='Report')
            return output.getvalue()
        else:
            # Convert to DataFrame first
            df = pd.DataFrame(data)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Report')
            return output.getvalue()

    def _generate_csv(self, data: Any) -> str:
        """Generate CSV report."""
        if isinstance(data, pd.DataFrame):
            return data.to_csv(index=False)
        else:
            df = pd.DataFrame(data)
            return df.to_csv(index=False)

    def _generate_html(self, data: Any) -> str:
        """Generate HTML report."""
        if isinstance(data, pd.DataFrame):
            return data.to_html(index=False)
        else:
            df = pd.DataFrame(data)
            return df.to_html(index=False)

    async def _generate_pdf(self, data: Any) -> bytes:
        """Generate PDF report (stub)."""
        # PDF generation would use libraries like reportlab or weasyprint
        logger.warning("PDF generation not yet fully implemented")
        return b""

    async def _generate_xbrl(self, data: Any) -> bytes:
        """Generate XBRL report (stub)."""
        # XBRL generation would use libraries like Arelle
        logger.warning("XBRL generation not yet fully implemented")
        return b""

    def _generate_parquet(self, data: Any) -> bytes:
        """Generate Parquet report."""
        import io

        if isinstance(data, pd.DataFrame):
            output = io.BytesIO()
            data.to_parquet(output, index=False)
            return output.getvalue()
        else:
            df = pd.DataFrame(data)
            output = io.BytesIO()
            df.to_parquet(output, index=False)
            return output.getvalue()

    def _generate_xml(self, data: Any) -> str:
        """Generate XML report."""
        if isinstance(data, pd.DataFrame):
            return data.to_xml(index=False)
        else:
            # Convert dict/list to DataFrame first
            df = pd.DataFrame(data)
            return df.to_xml(index=False)

    def _generate_markdown(self, data: Any) -> str:
        """Generate Markdown report."""
        if isinstance(data, pd.DataFrame):
            return data.to_markdown(index=False)
        else:
            df = pd.DataFrame(data)
            return df.to_markdown(index=False)

    def _generate_yaml(self, data: Any) -> str:
        """Generate YAML report."""
        try:
            import yaml

            if isinstance(data, pd.DataFrame):
                data_dict = data.to_dict(orient='records')
                return yaml.dump(data_dict, default_flow_style=False, sort_keys=False)
            else:
                return yaml.dump(data, default_flow_style=False, sort_keys=False)
        except ImportError:
            logger.error("PyYAML not available for YAML export")
            # Fallback to JSON-like format
            return json.dumps(data, indent=2, default=str)

    def _save_report(
        self,
        data: Union[bytes, str],
        output_path: str,
        format: ReportFormat
    ) -> str:
        """Save report to file."""
        try:
            mode = 'wb' if isinstance(data, bytes) else 'w'
            with open(output_path, mode) as f:
                f.write(data)

            logger.info(f"Report saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise

    def _check_compliance(
        self,
        data: Any,
        framework: ComplianceFramework
    ) -> ComplianceCheck:
        """Check compliance against framework."""
        # Get rules for framework
        rules = self.compliance_rules.get(framework.value, {})

        if not rules:
            return ComplianceCheck(
                framework=framework,
                passed=True,
                issues=[],
                score=100.0
            )

        issues = []

        # Check required fields
        required_fields = rules.get("required_fields", [])
        if isinstance(data, pd.DataFrame):
            missing_fields = set(required_fields) - set(data.columns)
            if missing_fields:
                issues.append(f"Missing required fields: {', '.join(missing_fields)}")

        # Check data thresholds
        thresholds = rules.get("thresholds", {})
        for field, threshold in thresholds.items():
            if isinstance(data, pd.DataFrame) and field in data.columns:
                violations = data[data[field] > threshold[field]]
                if len(violations) > 0:
                    issues.append(
                        f"Field '{field}' exceeds threshold in {len(violations)} rows"
                    )

        passed = len(issues) == 0
        score = 100.0 - (len(issues) * 10.0) if not passed else 100.0

        return ComplianceCheck(
            framework=framework,
            passed=passed,
            issues=issues,
            score=max(0.0, score)
        )

    def _get_row_count(self, data: Any) -> int:
        """Get row count from data."""
        if isinstance(data, pd.DataFrame):
            return len(data)
        elif isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return 1
        return 0

    async def generate_with_charts(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        chart_configs: List[Dict[str, Any]],
        format: ReportFormat = ReportFormat.HTML,
        output_path: Optional[str] = None,
    ) -> ReportResult:
        """
        Generate report with embedded charts.

        Args:
            data: Data to report
            chart_configs: List of chart configurations
            format: Output format (HTML or PDF recommended)
            output_path: Path to save report

        Returns:
            ReportResult with charts embedded

        Example chart_config:
            {
                "type": "bar",
                "x": "category",
                "y": "emissions",
                "title": "Emissions by Category"
            }
        """
        try:
            import matplotlib.pyplot as plt
            import io
            import base64

            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            charts_html = []

            # Generate each chart
            for config in chart_configs:
                chart_type = config.get("type", "bar")
                x_col = config.get("x")
                y_col = config.get("y")
                title = config.get("title", "Chart")

                fig, ax = plt.subplots(figsize=(10, 6))

                if chart_type == "bar":
                    data.plot(kind='bar', x=x_col, y=y_col, ax=ax)
                elif chart_type == "line":
                    data.plot(kind='line', x=x_col, y=y_col, ax=ax)
                elif chart_type == "pie":
                    data.set_index(x_col)[y_col].plot(kind='pie', ax=ax)
                elif chart_type == "scatter":
                    data.plot(kind='scatter', x=x_col, y=y_col, ax=ax)

                ax.set_title(title)
                plt.tight_layout()

                # Convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                chart_base64 = base64.b64encode(buffer.read()).decode()
                plt.close()

                charts_html.append(f'<img src="data:image/png;base64,{chart_base64}" />')

            # Generate HTML report with charts
            if format == ReportFormat.HTML:
                data_html = data.to_html(index=False)
                full_html = f"""
                <html>
                <head>
                    <title>Report with Charts</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #4CAF50; color: white; }}
                        img {{ max-width: 100%; margin: 20px 0; }}
                        h2 {{ color: #333; }}
                    </style>
                </head>
                <body>
                    <h1>Sustainability Report</h1>
                    <h2>Data</h2>
                    {data_html}
                    <h2>Visualizations</h2>
                    {"".join(charts_html)}
                    <p><em>Generated: {DeterministicClock.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC</em></p>
                </body>
                </html>
                """

                # Save if path provided
                file_path = None
                if output_path:
                    file_path = self._save_report(full_html, output_path, format)

                return ReportResult(
                    success=True,
                    file_path=file_path,
                    data=full_html,
                    format=format,
                    metadata={
                        "charts_count": len(chart_configs),
                        "data_rows": len(data),
                    }
                )

            else:
                # For other formats, generate without charts
                logger.warning(f"Charts not fully supported for {format}, generating data only")
                return await self.generate_report(data, format, output_path=output_path)

        except ImportError:
            logger.error("matplotlib not available for chart generation")
            return await self.generate_report(data, format, output_path=output_path)

        except Exception as e:
            logger.error(f"Chart generation failed: {e}", exc_info=True)
            return ReportResult(
                success=False,
                format=format,
                errors=[f"Chart generation failed: {str(e)}"]
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return self._stats.copy()
