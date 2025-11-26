"""Unit Tests for GL-009 THERMALIQ Visualization Module.

Comprehensive tests for all visualization components:
- Sankey diagram generation
- Waterfall chart creation
- Efficiency trend analysis
- Loss breakdown charts
- Export functionality
"""

import unittest
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile

from sankey_engine import (
    SankeyEngine,
    SankeyDiagram,
    SankeyNode,
    SankeyLink,
    NodeType,
    ColorScheme
)
from waterfall_chart import (
    WaterfallChart,
    WaterfallData,
    WaterfallBar,
    BarType
)
from efficiency_trends import (
    EfficiencyTrends,
    TrendData,
    TrendPoint,
    TrendType
)
from loss_breakdown import (
    LossBreakdown,
    LossCategory,
    BreakdownChart,
    ChartType
)
from export import (
    VisualizationExporter,
    ExportFormat,
    ExportConfig,
    export_to_html,
    export_to_json
)


class TestSankeyEngine(unittest.TestCase):
    """Test Sankey diagram generation."""

    def setUp(self):
        self.engine = SankeyEngine()
        self.sample_inputs = {"natural_gas": 5000.0, "electricity": 150.0}
        self.sample_outputs = {"steam": 4200.0, "hot_water": 300.0}
        self.sample_losses = {
            "flue_gas": 350.0,
            "radiation": 120.0,
            "blowdown": 100.0
        }

    def test_basic_sankey_generation(self):
        """Test basic Sankey diagram generation."""
        diagram = self.engine.generate_from_efficiency_result(
            energy_inputs=self.sample_inputs,
            useful_outputs=self.sample_outputs,
            losses=self.sample_losses
        )

        self.assertIsInstance(diagram, SankeyDiagram)
        self.assertEqual(diagram.total_input_kw, 5150.0)
        self.assertEqual(diagram.total_output_kw, 4500.0)
        self.assertEqual(diagram.total_losses_kw, 570.0)
        self.assertAlmostEqual(diagram.efficiency_percent, 87.38, places=2)

    def test_sankey_nodes(self):
        """Test Sankey node creation."""
        diagram = self.engine.generate_from_efficiency_result(
            energy_inputs=self.sample_inputs,
            useful_outputs=self.sample_outputs,
            losses=self.sample_losses
        )

        # Should have inputs + process + outputs + losses
        expected_nodes = 2 + 1 + 2 + 3  # 8 nodes total
        self.assertEqual(len(diagram.nodes), expected_nodes)

        # Check node types
        node_types = [n.node_type for n in diagram.nodes]
        self.assertIn(NodeType.INPUT, node_types)
        self.assertIn(NodeType.PROCESS, node_types)
        self.assertIn(NodeType.OUTPUT, node_types)
        self.assertIn(NodeType.LOSS, node_types)

    def test_sankey_links(self):
        """Test Sankey link creation."""
        diagram = self.engine.generate_from_efficiency_result(
            energy_inputs=self.sample_inputs,
            useful_outputs=self.sample_outputs,
            losses=self.sample_losses
        )

        # Should have links: inputs->process + process->outputs + process->losses
        expected_links = 2 + 2 + 3
        self.assertEqual(len(diagram.links), expected_links)

    def test_plotly_json_export(self):
        """Test Plotly JSON export."""
        diagram = self.engine.generate_from_efficiency_result(
            energy_inputs=self.sample_inputs,
            useful_outputs=self.sample_outputs,
            losses=self.sample_losses
        )

        plotly_json = diagram.to_plotly_json()

        self.assertIn("data", plotly_json)
        self.assertIn("layout", plotly_json)
        self.assertEqual(plotly_json["data"][0]["type"], "sankey")

    def test_multi_stage_sankey(self):
        """Test multi-stage Sankey generation."""
        stages = [
            {
                "name": "Stage 1",
                "inputs": {"fuel": 1000},
                "outputs": {"heat": 900},
                "losses": {"radiation": 100}
            },
            {
                "name": "Stage 2",
                "inputs": {"heat": 900},
                "outputs": {"steam": 800},
                "losses": {"convection": 100}
            }
        ]

        diagram = self.engine.generate_multi_stage(stages)
        self.assertIsInstance(diagram, SankeyDiagram)
        self.assertGreater(len(diagram.nodes), 0)
        self.assertGreater(len(diagram.links), 0)

    def test_color_schemes(self):
        """Test different color schemes."""
        for scheme in ColorScheme:
            engine = SankeyEngine(color_scheme=scheme)
            diagram = engine.generate_from_efficiency_result(
                energy_inputs=self.sample_inputs,
                useful_outputs=self.sample_outputs,
                losses=self.sample_losses
            )
            self.assertIsInstance(diagram, SankeyDiagram)

    def test_provenance_hash(self):
        """Test provenance hash generation."""
        diagram = self.engine.generate_from_efficiency_result(
            energy_inputs=self.sample_inputs,
            useful_outputs=self.sample_outputs,
            losses=self.sample_losses
        )

        self.assertIsNotNone(diagram.provenance_hash)
        self.assertEqual(len(diagram.provenance_hash), 16)


class TestWaterfallChart(unittest.TestCase):
    """Test waterfall chart generation."""

    def setUp(self):
        self.chart = WaterfallChart()
        self.sample_input = {"fuel_input": 5150.0}
        self.sample_losses = {
            "flue_gas": 350.0,
            "radiation": 120.0,
            "convection": 80.0
        }
        self.sample_output = {"steam_output": 4600.0}

    def test_basic_waterfall(self):
        """Test basic waterfall chart generation."""
        waterfall = self.chart.generate_from_heat_balance(
            input_energy=self.sample_input,
            losses=self.sample_losses,
            useful_output=self.sample_output
        )

        self.assertIsInstance(waterfall, WaterfallData)
        self.assertEqual(waterfall.start_value, 5150.0)
        self.assertGreater(len(waterfall.bars), 0)

    def test_waterfall_bars(self):
        """Test waterfall bar creation."""
        waterfall = self.chart.generate_from_heat_balance(
            input_energy=self.sample_input,
            losses=self.sample_losses,
            useful_output=self.sample_output
        )

        # Should have: total input + losses + subtotal + total output
        bar_types = [b.bar_type for b in waterfall.bars]
        self.assertIn(BarType.TOTAL, bar_types)
        self.assertIn(BarType.LOSS, bar_types)

    def test_plotly_json_export(self):
        """Test Plotly JSON export for waterfall."""
        waterfall = self.chart.generate_from_heat_balance(
            input_energy=self.sample_input,
            losses=self.sample_losses,
            useful_output=self.sample_output
        )

        plotly_json = waterfall.to_plotly_json()

        self.assertIn("data", plotly_json)
        self.assertIn("layout", plotly_json)
        self.assertEqual(plotly_json["data"][0]["type"], "waterfall")

    def test_detailed_waterfall(self):
        """Test detailed waterfall with process stages."""
        process_losses = {"combustion": 50.0, "heat_transfer": 200.0}
        distribution_losses = {"pipe_radiation": 80.0}

        waterfall = self.chart.generate_detailed_breakdown(
            input_energy=self.sample_input,
            process_losses=process_losses,
            distribution_losses=distribution_losses,
            useful_output=self.sample_output
        )

        self.assertIsInstance(waterfall, WaterfallData)
        self.assertGreater(len(waterfall.bars), len(process_losses) + len(distribution_losses))


class TestEfficiencyTrends(unittest.TestCase):
    """Test efficiency trend analysis."""

    def setUp(self):
        self.trends = EfficiencyTrends()
        self.sample_data = [
            (datetime(2024, 1, 1) + timedelta(days=i), 87.0 + i * 0.1)
            for i in range(30)
        ]

    def test_basic_trend(self):
        """Test basic efficiency trend generation."""
        trend = self.trends.generate_efficiency_trend(
            efficiency_data=self.sample_data
        )

        self.assertIsInstance(trend, TrendData)
        self.assertEqual(len(trend.points), 30)
        self.assertEqual(trend.trend_type, TrendType.EFFICIENCY)

    def test_statistics_calculation(self):
        """Test trend statistics calculation."""
        trend = self.trends.generate_efficiency_trend(
            efficiency_data=self.sample_data
        )

        self.assertGreater(trend.avg_value, 0)
        self.assertGreater(trend.max_value, trend.min_value)
        self.assertGreaterEqual(trend.std_dev, 0)

    def test_moving_average(self):
        """Test moving average calculation."""
        trend = self.trends.generate_efficiency_trend(
            efficiency_data=self.sample_data,
            moving_average_days=7
        )

        self.assertEqual(trend.moving_average_window, 7)

    def test_benchmark_comparison(self):
        """Test benchmark efficiency comparison."""
        trend = self.trends.generate_efficiency_trend(
            efficiency_data=self.sample_data,
            benchmark_efficiency=88.0
        )

        self.assertEqual(trend.benchmark_value, 88.0)

    def test_plotly_json_export(self):
        """Test Plotly JSON export for trends."""
        trend = self.trends.generate_efficiency_trend(
            efficiency_data=self.sample_data
        )

        plotly_json = trend.to_plotly_json()

        self.assertIn("data", plotly_json)
        self.assertIn("layout", plotly_json)
        self.assertGreater(len(plotly_json["data"]), 0)


class TestLossBreakdown(unittest.TestCase):
    """Test loss breakdown charts."""

    def setUp(self):
        self.breakdown = LossBreakdown()
        self.sample_losses = {
            "flue_gas": 350.0,
            "radiation": 120.0,
            "convection": 80.0,
            "blowdown": 100.0
        }

    def test_pie_chart(self):
        """Test pie chart generation."""
        chart = self.breakdown.generate_pie_chart(
            losses=self.sample_losses
        )

        self.assertIsInstance(chart, BreakdownChart)
        self.assertEqual(chart.chart_type, ChartType.PIE)
        self.assertEqual(len(chart.categories), len(self.sample_losses))

    def test_donut_chart(self):
        """Test donut chart generation."""
        chart = self.breakdown.generate_donut_chart(
            losses=self.sample_losses
        )

        self.assertEqual(chart.chart_type, ChartType.DONUT)

    def test_bar_chart(self):
        """Test bar chart generation."""
        chart = self.breakdown.generate_bar_chart(
            losses=self.sample_losses,
            horizontal=False
        )

        self.assertEqual(chart.chart_type, ChartType.BAR)

    def test_horizontal_bar_chart(self):
        """Test horizontal bar chart generation."""
        chart = self.breakdown.generate_bar_chart(
            losses=self.sample_losses,
            horizontal=True
        )

        self.assertEqual(chart.chart_type, ChartType.HORIZONTAL_BAR)

    def test_category_percentages(self):
        """Test loss category percentage calculation."""
        chart = self.breakdown.generate_pie_chart(
            losses=self.sample_losses
        )

        total_percentage = sum(c.percentage for c in chart.categories)
        self.assertAlmostEqual(total_percentage, 100.0, places=1)

    def test_plotly_json_export(self):
        """Test Plotly JSON export for breakdown charts."""
        chart = self.breakdown.generate_pie_chart(
            losses=self.sample_losses
        )

        plotly_json = chart.to_plotly_json()

        self.assertIn("data", plotly_json)
        self.assertIn("layout", plotly_json)


class TestExport(unittest.TestCase):
    """Test export functionality."""

    def setUp(self):
        self.exporter = VisualizationExporter()
        self.temp_dir = tempfile.mkdtemp()

        # Create sample figure
        engine = SankeyEngine()
        diagram = engine.generate_from_efficiency_result(
            energy_inputs={"fuel": 1000},
            useful_outputs={"steam": 800},
            losses={"flue_gas": 200}
        )
        self.sample_figure = diagram.to_plotly_json()

    def test_html_export(self):
        """Test HTML export."""
        output_path = Path(self.temp_dir) / "test.html"
        result = export_to_html(self.sample_figure, output_path)

        self.assertTrue(result.exists())
        self.assertEqual(result.suffix, ".html")

        # Verify HTML content
        content = result.read_text()
        self.assertIn("plotly", content.lower())
        self.assertIn("sankey", content.lower())

    def test_json_export(self):
        """Test JSON export."""
        output_path = Path(self.temp_dir) / "test.json"
        result = export_to_json(self.sample_figure, output_path)

        self.assertTrue(result.exists())
        self.assertEqual(result.suffix, ".json")

        # Verify JSON content
        data = json.loads(result.read_text())
        self.assertIn("figure", data)
        self.assertIn("export_timestamp", data)

    def test_export_config(self):
        """Test export configuration."""
        config = ExportConfig(width=1600, height=1000, dpi=300)

        self.assertEqual(config.width, 1600)
        self.assertEqual(config.height, 1000)
        self.assertEqual(config.dpi, 300)

    def test_dashboard_export(self):
        """Test dashboard export."""
        from export import export_dashboard

        figures = [self.sample_figure, self.sample_figure]
        output_path = Path(self.temp_dir) / "dashboard.html"

        result = export_dashboard(
            figures=figures,
            output_path=output_path,
            layout="grid"
        )

        self.assertTrue(result.exists())

        # Verify dashboard content
        content = result.read_text()
        self.assertIn("chart0", content)
        self.assertIn("chart1", content)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""

    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow."""
        # 1. Generate Sankey
        engine = SankeyEngine()
        sankey = engine.generate_from_efficiency_result(
            energy_inputs={"fuel": 5000},
            useful_outputs={"steam": 4200},
            losses={"flue_gas": 350, "radiation": 120}
        )

        # 2. Generate Waterfall
        chart = WaterfallChart()
        waterfall = chart.generate_from_heat_balance(
            input_energy={"fuel": 5000},
            losses={"flue_gas": 350, "radiation": 120},
            useful_output={"steam": 4530}
        )

        # 3. Generate Loss Breakdown
        breakdown = LossBreakdown()
        pie = breakdown.generate_pie_chart(
            losses={"flue_gas": 350, "radiation": 120}
        )

        # 4. Verify all components
        self.assertIsInstance(sankey, SankeyDiagram)
        self.assertIsInstance(waterfall, WaterfallData)
        self.assertIsInstance(pie, BreakdownChart)

        # 5. Export to Plotly JSON
        sankey_json = sankey.to_plotly_json()
        waterfall_json = waterfall.to_plotly_json()
        pie_json = pie.to_plotly_json()

        self.assertIn("data", sankey_json)
        self.assertIn("data", waterfall_json)
        self.assertIn("data", pie_json)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSankeyEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestWaterfallChart))
    suite.addTests(loader.loadTestsFromTestCase(TestEfficiencyTrends))
    suite.addTests(loader.loadTestsFromTestCase(TestLossBreakdown))
    suite.addTests(loader.loadTestsFromTestCase(TestExport))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
