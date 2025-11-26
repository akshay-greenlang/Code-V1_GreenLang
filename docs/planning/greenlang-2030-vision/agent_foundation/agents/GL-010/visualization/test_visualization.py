"""
GL-010 EMISSIONWATCH - Visualization Tests

Test suite for the visualization engine.
Validates output formats, data accuracy, and export functionality.

Author: GreenLang Team
Version: 1.0.0
"""

import json
import unittest
from datetime import datetime
from typing import Dict, Any, List


class TestComplianceDashboard(unittest.TestCase):
    """Tests for compliance dashboard visualization."""

    def setUp(self):
        """Set up test fixtures."""
        from compliance_dashboard import (
            ComplianceDashboard,
            ComplianceDashboardData,
            PollutantStatus,
            ComplianceStatus,
            create_sample_dashboard_data
        )

        self.sample_data = create_sample_dashboard_data()
        self.dashboard = ComplianceDashboard(self.sample_data)

    def test_dashboard_creation(self):
        """Test dashboard can be created from sample data."""
        self.assertIsNotNone(self.dashboard)
        self.assertEqual(self.dashboard.data.facility_id, "FAC-001")

    def test_status_matrix_output_format(self):
        """Test status matrix returns valid Plotly format."""
        chart = self.dashboard.generate_status_matrix()

        self.assertIn("data", chart)
        self.assertIn("layout", chart)
        self.assertIn("config", chart)
        self.assertIsInstance(chart["data"], list)
        self.assertTrue(len(chart["data"]) > 0)

    def test_gauge_chart_output_format(self):
        """Test gauge chart returns valid Plotly format."""
        pollutant_ids = list(self.sample_data.pollutants.keys())
        self.assertTrue(len(pollutant_ids) > 0)

        chart = self.dashboard.generate_gauge_chart(pollutant_ids[0])

        self.assertIn("data", chart)
        self.assertIn("layout", chart)
        self.assertEqual(chart["data"][0]["type"], "indicator")

    def test_all_gauges_generation(self):
        """Test all gauge charts can be generated."""
        gauges = self.dashboard.generate_all_gauges()

        self.assertIsInstance(gauges, list)
        self.assertEqual(len(gauges), len(self.sample_data.pollutants))

    def test_margin_chart_output(self):
        """Test margin chart contains expected data."""
        chart = self.dashboard.generate_margin_chart()

        self.assertIn("data", chart)
        self.assertEqual(chart["data"][0]["type"], "bar")

    def test_violation_summary_output(self):
        """Test violation summary chart generation."""
        chart = self.dashboard.generate_violation_summary()

        self.assertIn("data", chart)
        self.assertIn("layout", chart)

    def test_to_plotly_json_valid(self):
        """Test JSON export is valid JSON."""
        json_output = self.dashboard.to_plotly_json()

        # Should be valid JSON
        parsed = json.loads(json_output)
        self.assertIn("metadata", parsed)
        self.assertIn("charts", parsed)

    def test_to_html_contains_plotly(self):
        """Test HTML export contains Plotly script."""
        html_output = self.dashboard.to_html()

        self.assertIn("plotly", html_output.lower())
        self.assertIn("<html", html_output)
        self.assertIn("</html>", html_output)

    def test_color_blind_safe_mode(self):
        """Test color-blind safe mode changes colors."""
        from compliance_dashboard import ComplianceStatus

        dashboard_standard = ComplianceDashboard(self.sample_data, color_blind_safe=False)
        dashboard_safe = ComplianceDashboard(self.sample_data, color_blind_safe=True)

        # Colors should be different
        self.assertNotEqual(
            ComplianceStatus.COMPLIANT.color,
            ComplianceStatus.COMPLIANT.color_blind_safe
        )


class TestEmissionsTrends(unittest.TestCase):
    """Tests for emissions trend visualization."""

    def setUp(self):
        """Set up test fixtures."""
        from emissions_trends import (
            EmissionsTrendChart,
            TrendConfig,
            TimeResolution,
            create_sample_trend_data
        )

        self.config = TrendConfig(
            pollutant="NOx",
            pollutant_name="Nitrogen Oxides",
            unit="lb/hr",
            permit_limit=200.0,
            warning_threshold=180.0,
            resolution=TimeResolution.HOURLY,
            show_rolling_average=True,
            rolling_window=24,
            show_forecast=True,
            forecast_periods=24,
            highlight_anomalies=True
        )

        self.chart = EmissionsTrendChart(self.config)
        self.sample_data = create_sample_trend_data(168)  # 7 days
        self.chart.set_data(self.sample_data)

    def test_data_loading(self):
        """Test data is loaded correctly."""
        self.assertEqual(len(self.chart._data), 168)

    def test_statistics_calculation(self):
        """Test statistics are calculated correctly."""
        stats = self.chart.get_statistics()

        self.assertIsNotNone(stats)
        self.assertTrue(stats.min_value <= stats.mean_value <= stats.max_value)
        self.assertTrue(stats.percentile_95 <= stats.max_value)
        self.assertEqual(stats.total_points, 168)

    def test_anomaly_detection(self):
        """Test anomaly detection returns valid data."""
        anomalies = self.chart.get_anomalies()

        self.assertIsInstance(anomalies, list)
        for anom in anomalies:
            self.assertIn("timestamp", anom)
            self.assertIn("value", anom)
            self.assertIn("z_score", anom)

    def test_forecast_generation(self):
        """Test forecast is generated."""
        forecast = self.chart.get_forecast()

        self.assertIsNotNone(forecast)
        if "values" in forecast:
            self.assertEqual(len(forecast["values"]), self.config.forecast_periods)
            self.assertEqual(len(forecast["timestamps"]), self.config.forecast_periods)

    def test_hourly_trend_output(self):
        """Test hourly trend chart format."""
        chart = self.chart.build_hourly_trend()

        self.assertIn("data", chart)
        self.assertIn("layout", chart)
        self.assertTrue(len(chart["data"]) >= 2)  # At least trend + limit

    def test_statistics_panel_output(self):
        """Test statistics panel format."""
        panel = self.chart.build_statistics_panel()

        self.assertIn("data", panel)
        self.assertTrue(len(panel["data"]) >= 1)


class TestViolationTimeline(unittest.TestCase):
    """Tests for violation timeline visualization."""

    def setUp(self):
        """Set up test fixtures."""
        from violation_timeline import (
            ViolationTimelineChart,
            TimelineConfig,
            create_sample_violations
        )

        self.violations = create_sample_violations(20)
        self.config = TimelineConfig(
            title="Test Violation Timeline",
            show_duration_bars=True
        )
        self.timeline = ViolationTimelineChart(self.violations, self.config)

    def test_violations_loaded(self):
        """Test violations are loaded correctly."""
        self.assertEqual(len(self.timeline.violations), 20)

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        stats = self.timeline.get_summary_statistics()

        self.assertEqual(stats["total_violations"], 20)
        self.assertIn("by_severity", stats)
        self.assertIn("by_status", stats)
        self.assertIn("by_pollutant", stats)

    def test_gantt_timeline_output(self):
        """Test Gantt timeline format."""
        chart = self.timeline.build_gantt_timeline()

        self.assertIn("data", chart)
        self.assertIn("layout", chart)

    def test_scatter_timeline_output(self):
        """Test scatter timeline format."""
        chart = self.timeline.build_scatter_timeline()

        self.assertIn("data", chart)
        for trace in chart["data"]:
            self.assertIn("type", trace)

    def test_status_breakdown_output(self):
        """Test status breakdown format."""
        chart = self.timeline.build_status_breakdown()

        self.assertIn("data", chart)

    def test_monthly_trend_output(self):
        """Test monthly trend format."""
        chart = self.timeline.build_monthly_trend()

        self.assertIn("data", chart)

    def test_export_for_report(self):
        """Test report export data structure."""
        export_data = self.timeline.export_for_report()

        self.assertIn("summary", export_data)
        self.assertIn("violations", export_data)
        self.assertIn("charts", export_data)
        self.assertIn("generated_at", export_data)


class TestSourceBreakdown(unittest.TestCase):
    """Tests for source breakdown visualization."""

    def setUp(self):
        """Set up test fixtures."""
        from source_breakdown import (
            SourceBreakdownChart,
            SourceBreakdownConfig,
            create_sample_sources
        )

        self.sources = create_sample_sources(15)
        self.config = SourceBreakdownConfig(
            title="Test Source Breakdown",
            show_percentages=True
        )
        self.breakdown = SourceBreakdownChart(self.sources, self.config)

    def test_sources_loaded(self):
        """Test sources are loaded correctly."""
        self.assertEqual(len(self.breakdown.sources), 15)

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        stats = self.breakdown.get_summary_statistics()

        self.assertEqual(stats["total_sources"], 15)
        self.assertIn("by_type", stats)
        self.assertIn("by_pollutant", stats)
        self.assertIn("top_sources", stats)

    def test_pie_chart_output(self):
        """Test pie chart format."""
        chart = self.breakdown.build_pie_chart()

        self.assertIn("data", chart)
        self.assertEqual(chart["data"][0]["type"], "pie")

    def test_bar_chart_output(self):
        """Test bar chart format."""
        chart = self.breakdown.build_bar_chart(horizontal=True)

        self.assertIn("data", chart)
        self.assertEqual(chart["data"][0]["type"], "bar")

    def test_stacked_bar_output(self):
        """Test stacked bar chart format."""
        chart = self.breakdown.build_stacked_bar_chart()

        self.assertIn("data", chart)
        self.assertIn("layout", chart)

    def test_treemap_output(self):
        """Test treemap format."""
        chart = self.breakdown.build_treemap()

        self.assertIn("data", chart)
        self.assertEqual(chart["data"][0]["type"], "treemap")

    def test_sankey_output(self):
        """Test Sankey diagram format."""
        chart = self.breakdown.build_sankey_diagram()

        self.assertIn("data", chart)
        self.assertEqual(chart["data"][0]["type"], "sankey")


class TestRegulatoryHeatmap(unittest.TestCase):
    """Tests for regulatory heatmap visualization."""

    def setUp(self):
        """Set up test fixtures."""
        from regulatory_heatmap import (
            RegulatoryHeatmap,
            HeatmapConfig,
            create_sample_heatmap_data
        )

        self.jurisdictions, self.pollutants, self.cells = create_sample_heatmap_data(6, 5)
        self.config = HeatmapConfig(
            title="Test Compliance Heatmap",
            show_values=True
        )
        self.heatmap = RegulatoryHeatmap(
            self.jurisdictions,
            self.pollutants,
            self.config
        )
        self.heatmap.set_compliance_data(self.cells)

    def test_data_loaded(self):
        """Test data is loaded correctly."""
        self.assertEqual(len(self.jurisdictions), 6)
        self.assertEqual(len(self.pollutants), 5)

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        stats = self.heatmap.get_summary_statistics()

        self.assertEqual(stats["total_jurisdictions"], 6)
        self.assertEqual(stats["total_pollutants"], 5)
        self.assertIn("by_level", stats)
        self.assertIn("overall_compliance_rate", stats)

    def test_heatmap_output(self):
        """Test heatmap format."""
        chart = self.heatmap.build_heatmap()

        self.assertIn("data", chart)
        self.assertEqual(chart["data"][0]["type"], "heatmap")

    def test_summary_indicators_output(self):
        """Test summary indicators format."""
        chart = self.heatmap.build_summary_indicators()

        self.assertIn("data", chart)
        self.assertTrue(len(chart["data"]) >= 1)

    def test_jurisdiction_detail_output(self):
        """Test jurisdiction detail format."""
        jid = self.jurisdictions[0].jurisdiction_id
        chart = self.heatmap.build_jurisdiction_detail(jid)

        self.assertIn("data", chart)

    def test_pollutant_comparison_output(self):
        """Test pollutant comparison format."""
        chart = self.heatmap.build_pollutant_comparison(self.pollutants[0])

        self.assertIn("data", chart)


class TestExport(unittest.TestCase):
    """Tests for export functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from export import (
            ReportExporter,
            ExportConfig,
            ExportFormat,
            TableData
        )

        self.config = ExportConfig(
            title="Test Report",
            facility_name="Test Facility",
            facility_id="TEST-001"
        )
        self.exporter = ReportExporter(self.config)

        # Add test data
        self.exporter.set_summary({
            "total": 100,
            "violations": 2,
            "compliance": 98.0
        })

        self.exporter.add_table(TableData(
            title="Test Table",
            headers=["Col1", "Col2"],
            rows=[["A", "B"], ["C", "D"]]
        ))

    def test_json_export(self):
        """Test JSON export."""
        from export import ExportFormat

        output = self.exporter.export_bytes(ExportFormat.JSON)

        # Should be valid JSON
        parsed = json.loads(output)
        self.assertIn("metadata", parsed)
        self.assertIn("summary", parsed)
        self.assertIn("tables", parsed)

    def test_html_export(self):
        """Test HTML export."""
        from export import ExportFormat

        output = self.exporter.export_bytes(ExportFormat.HTML)
        html = output.decode('utf-8')

        self.assertIn("<html", html)
        self.assertIn("Test Report", html)
        self.assertIn("Test Facility", html)

    def test_xml_export(self):
        """Test XML export."""
        from export import ExportFormat

        self.exporter.add_emissions_data([{
            "pollutant": "NOx",
            "value": 100,
            "unit": "tons"
        }])

        output = self.exporter.export_bytes(ExportFormat.XML)
        xml = output.decode('utf-8')

        self.assertIn("<?xml", xml)
        self.assertIn("CEDRISubmission", xml)
        self.assertIn("NOx", xml)

    def test_excel_export(self):
        """Test Excel export (JSON structure)."""
        from export import ExportFormat

        output = self.exporter.export_bytes(ExportFormat.EXCEL)

        parsed = json.loads(output)
        self.assertIn("metadata", parsed)
        self.assertIn("sheets", parsed)


class TestDataAccuracy(unittest.TestCase):
    """Tests for data accuracy in calculations."""

    def test_statistics_accuracy(self):
        """Test statistical calculations are accurate."""
        from emissions_trends import (
            StatisticsCalculator,
            EmissionDataPoint
        )

        # Known data
        data = [
            EmissionDataPoint(timestamp="2024-01-01T00:00:00Z", value=100.0, unit="lb/hr", data_quality=100),
            EmissionDataPoint(timestamp="2024-01-01T01:00:00Z", value=150.0, unit="lb/hr", data_quality=100),
            EmissionDataPoint(timestamp="2024-01-01T02:00:00Z", value=200.0, unit="lb/hr", data_quality=100),
            EmissionDataPoint(timestamp="2024-01-01T03:00:00Z", value=150.0, unit="lb/hr", data_quality=100),
            EmissionDataPoint(timestamp="2024-01-01T04:00:00Z", value=100.0, unit="lb/hr", data_quality=100),
        ]

        stats = StatisticsCalculator.calculate(data, permit_limit=180.0)

        # Verify calculations
        self.assertEqual(stats.min_value, 100.0)
        self.assertEqual(stats.max_value, 200.0)
        self.assertEqual(stats.mean_value, 140.0)
        self.assertEqual(stats.median_value, 150.0)
        self.assertEqual(stats.total_points, 5)
        self.assertEqual(stats.valid_points, 5)
        self.assertEqual(stats.exceedance_count, 1)  # 200 > 180

    def test_compliance_level_from_margin(self):
        """Test compliance level determination from margin."""
        from regulatory_heatmap import ComplianceLevel

        # Test each level
        self.assertEqual(ComplianceLevel.from_margin(35.0), ComplianceLevel.EXCELLENT)
        self.assertEqual(ComplianceLevel.from_margin(25.0), ComplianceLevel.GOOD)
        self.assertEqual(ComplianceLevel.from_margin(15.0), ComplianceLevel.ADEQUATE)
        self.assertEqual(ComplianceLevel.from_margin(7.0), ComplianceLevel.MARGINAL)
        self.assertEqual(ComplianceLevel.from_margin(3.0), ComplianceLevel.WARNING)
        self.assertEqual(ComplianceLevel.from_margin(-5.0), ComplianceLevel.VIOLATION)
        self.assertEqual(ComplianceLevel.from_margin(None), ComplianceLevel.UNKNOWN)


class TestOutputFormats(unittest.TestCase):
    """Tests for output format validation."""

    def test_plotly_chart_structure(self):
        """Test Plotly chart has required structure."""
        from compliance_dashboard import create_sample_dashboard_data, ComplianceDashboard

        dashboard = ComplianceDashboard(create_sample_dashboard_data())
        chart = dashboard.generate_status_matrix()

        # Required Plotly structure
        self.assertIn("data", chart)
        self.assertIn("layout", chart)
        self.assertIsInstance(chart["data"], list)
        self.assertIsInstance(chart["layout"], dict)

        # Each trace should have a type
        for trace in chart["data"]:
            self.assertIn("type", trace)

    def test_d3_compatible_output(self):
        """Test D3.js compatible output."""
        from compliance_dashboard import create_sample_dashboard_data, ComplianceDashboard

        dashboard = ComplianceDashboard(create_sample_dashboard_data())
        d3_json = dashboard.to_d3_json()

        parsed = json.loads(d3_json)

        # D3-friendly structure
        self.assertIn("metadata", parsed)
        self.assertIn("pollutants", parsed)
        self.assertIsInstance(parsed["pollutants"], list)

        for p in parsed["pollutants"]:
            self.assertIn("id", p)
            self.assertIn("value", p)
            self.assertIn("limit", p)

    def test_html_validity(self):
        """Test HTML output has proper structure."""
        from compliance_dashboard import create_sample_dashboard_data, ComplianceDashboard

        dashboard = ComplianceDashboard(create_sample_dashboard_data())
        html = dashboard.to_html()

        # Basic HTML structure
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<html", html)
        self.assertIn("</html>", html)
        self.assertIn("<head>", html)
        self.assertIn("<body>", html)

        # Should include Plotly
        self.assertIn("plotly", html.lower())

        # Should include chart containers
        self.assertIn("chart", html.lower())


def run_tests():
    """Run all visualization tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestComplianceDashboard))
    suite.addTests(loader.loadTestsFromTestCase(TestEmissionsTrends))
    suite.addTests(loader.loadTestsFromTestCase(TestViolationTimeline))
    suite.addTests(loader.loadTestsFromTestCase(TestSourceBreakdown))
    suite.addTests(loader.loadTestsFromTestCase(TestRegulatoryHeatmap))
    suite.addTests(loader.loadTestsFromTestCase(TestExport))
    suite.addTests(loader.loadTestsFromTestCase(TestDataAccuracy))
    suite.addTests(loader.loadTestsFromTestCase(TestOutputFormats))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    run_tests()
