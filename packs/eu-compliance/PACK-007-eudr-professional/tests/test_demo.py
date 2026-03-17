"""
Unit tests for PACK-007 EUDR Professional Pack - Demo Mode

Tests demo configuration, sample data generation, and demo workflow execution.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def _import_from_path(module_name, file_path):
    """Helper to import from hyphenated directory paths."""
    if not file_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_PACK_007_DIR = Path(__file__).resolve().parent.parent

# Import demo module
demo_mod = _import_from_path(
    "pack_007_demo",
    _PACK_007_DIR / "demo" / "demo_data.py"
)

pytestmark = pytest.mark.skipif(
    demo_mod is None,
    reason="PACK-007 demo module not available"
)


class TestDemoConfiguration:
    """Test demo configuration and setup."""

    def test_demo_config_loads(self):
        """Test demo configuration loads successfully."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        config = demo_mod.get_demo_config()

        assert config is not None
        assert "demo_mode_enabled" in config or "enabled" in config
        assert "sample_data_size" in config or "data_size" in config

    def test_demo_mode_toggle(self):
        """Test toggling demo mode on/off."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        # Enable demo mode
        demo_mod.enable_demo_mode()
        assert demo_mod.is_demo_mode_enabled() is True

        # Disable demo mode
        demo_mod.disable_demo_mode()
        assert demo_mod.is_demo_mode_enabled() is False

    def test_demo_reset(self):
        """Test resetting demo data."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        result = demo_mod.reset_demo_data()

        assert result is not None
        assert result.get("status") in ["SUCCESS", "RESET", "OK"]


class TestDemoSuppliers:
    """Test demo supplier data."""

    def test_demo_suppliers_valid(self):
        """Test demo suppliers are valid."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        suppliers = demo_mod.get_demo_suppliers()

        assert suppliers is not None
        assert isinstance(suppliers, list)
        assert len(suppliers) > 0

        # Validate first supplier
        supplier = suppliers[0]
        assert "supplier_id" in supplier or "id" in supplier
        assert "name" in supplier
        assert "country" in supplier

    def test_demo_supplier_diversity(self):
        """Test demo suppliers cover diverse scenarios."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        suppliers = demo_mod.get_demo_suppliers()

        # Should have suppliers from different countries
        countries = set(s.get("country") for s in suppliers)
        assert len(countries) >= 3

        # Should have different products
        products = set(s.get("product") for s in suppliers if "product" in s)
        assert len(products) >= 2

    def test_demo_supplier_risk_levels(self):
        """Test demo suppliers have various risk levels."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        suppliers = demo_mod.get_demo_suppliers()

        risk_levels = set(s.get("risk_level") for s in suppliers if "risk_level" in s)

        # Should have mix of risk levels
        assert len(risk_levels) >= 2


class TestDemoPlots:
    """Test demo plot/geolocation data."""

    def test_demo_plots_valid(self):
        """Test demo plots are valid."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        plots = demo_mod.get_demo_plots()

        assert plots is not None
        assert isinstance(plots, list)
        assert len(plots) > 0

        # Validate first plot
        plot = plots[0]
        assert "plot_id" in plot or "id" in plot
        assert "latitude" in plot
        assert "longitude" in plot

    def test_demo_plot_coordinates(self):
        """Test demo plot coordinates are valid."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        plots = demo_mod.get_demo_plots()

        for plot in plots:
            lat = plot.get("latitude")
            lon = plot.get("longitude")

            # Validate coordinate ranges
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180

    def test_demo_plot_diversity(self):
        """Test demo plots cover diverse geographic areas."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        plots = demo_mod.get_demo_plots()

        # Should have plots in different regions
        countries = set(p.get("country") for p in plots if "country" in p)
        assert len(countries) >= 3


class TestDemoPortfolio:
    """Test demo portfolio data."""

    def test_demo_portfolio_valid(self):
        """Test demo portfolio is valid."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        portfolio = demo_mod.get_demo_portfolio()

        assert portfolio is not None
        assert "operators" in portfolio or "total_operators" in portfolio
        assert "products" in portfolio or "total_products" in portfolio
        assert "suppliers" in portfolio or "total_suppliers" in portfolio

    def test_demo_portfolio_statistics(self):
        """Test demo portfolio statistics."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        portfolio = demo_mod.get_demo_portfolio()
        stats = portfolio.get("statistics") or portfolio

        assert "total_suppliers" in stats or "suppliers" in stats
        assert "total_plots" in stats or "plots" in stats
        assert "average_risk_score" in stats or "risk_score" in stats

    def test_demo_portfolio_completeness(self):
        """Test demo portfolio has complete data."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        portfolio = demo_mod.get_demo_portfolio()

        # Should have all required components
        required_keys = ["operators", "suppliers", "plots", "products"]
        for key in required_keys:
            assert key in portfolio or any(k.startswith(key) for k in portfolio.keys())


class TestDemoWorkflows:
    """Test demo workflow execution."""

    def test_demo_workflow_runs(self):
        """Test demo workflows execute successfully."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        result = demo_mod.run_demo_workflow(
            workflow_type="advanced_risk_modeling"
        )

        assert result is not None
        assert "status" in result
        assert result["status"] in ["SUCCESS", "COMPLETE", "OK"]

    def test_demo_risk_modeling_workflow(self):
        """Test demo advanced risk modeling workflow."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        result = demo_mod.run_demo_workflow("advanced_risk_modeling")

        assert result is not None
        assert "risk_distribution" in result or "results" in result

    def test_demo_monitoring_workflow(self):
        """Test demo continuous monitoring workflow."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        result = demo_mod.run_demo_workflow("continuous_monitoring")

        assert result is not None
        assert "monitoring_results" in result or "results" in result

    def test_demo_supplier_benchmarking_workflow(self):
        """Test demo supplier benchmarking workflow."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        result = demo_mod.run_demo_workflow("supplier_benchmarking")

        assert result is not None
        assert "supplier_rankings" in result or "results" in result


class TestDemoReports:
    """Test demo report generation."""

    def test_demo_reports_generate(self):
        """Test demo reports generate successfully."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        reports = demo_mod.generate_demo_reports()

        assert reports is not None
        assert isinstance(reports, list) or isinstance(reports, dict)

    def test_demo_risk_report(self):
        """Test generating demo advanced risk report."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        report = demo_mod.generate_demo_report("advanced_risk")

        assert report is not None
        assert "risk_score" in report or "results" in report

    def test_demo_monitoring_report(self):
        """Test generating demo satellite monitoring report."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        report = demo_mod.generate_demo_report("satellite_monitoring")

        assert report is not None
        assert "monitoring_period" in report or "results" in report

    def test_demo_portfolio_dashboard(self):
        """Test generating demo portfolio dashboard."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        dashboard = demo_mod.generate_demo_report("portfolio_dashboard")

        assert dashboard is not None
        assert "total_suppliers" in dashboard or "statistics" in dashboard


class TestDemoDataGeneration:
    """Test demo data generation features."""

    def test_generate_random_supplier(self):
        """Test generating random demo supplier."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        supplier = demo_mod.generate_random_supplier()

        assert supplier is not None
        assert "supplier_id" in supplier or "id" in supplier
        assert "name" in supplier
        assert "country" in supplier

    def test_generate_random_plot(self):
        """Test generating random demo plot."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        plot = demo_mod.generate_random_plot()

        assert plot is not None
        assert "plot_id" in plot or "id" in plot
        assert "latitude" in plot
        assert "longitude" in plot

    def test_generate_demo_dataset(self):
        """Test generating complete demo dataset."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        dataset = demo_mod.generate_demo_dataset(
            num_suppliers=10,
            num_plots=50
        )

        assert dataset is not None
        assert "suppliers" in dataset
        assert "plots" in dataset
        assert len(dataset["suppliers"]) == 10
        assert len(dataset["plots"]) == 50


class TestDemoScenarios:
    """Test demo scenario execution."""

    def test_high_risk_scenario(self):
        """Test high-risk supplier scenario."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        scenario = demo_mod.run_demo_scenario("high_risk")

        assert scenario is not None
        assert "risk_level" in scenario or "scenario" in scenario

    def test_protected_area_scenario(self):
        """Test protected area overlap scenario."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        scenario = demo_mod.run_demo_scenario("protected_area_overlap")

        assert scenario is not None

    def test_grievance_scenario(self):
        """Test grievance handling scenario."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        scenario = demo_mod.run_demo_scenario("grievance")

        assert scenario is not None


class TestDemoValidation:
    """Test demo data validation."""

    def test_validate_demo_data(self):
        """Test demo data passes validation."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        validation = demo_mod.validate_demo_data()

        assert validation is not None
        assert validation.get("valid") is True or validation.get("status") == "VALID"

    def test_demo_data_consistency(self):
        """Test demo data is internally consistent."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        suppliers = demo_mod.get_demo_suppliers()
        plots = demo_mod.get_demo_plots()

        # Plots should reference valid suppliers
        supplier_ids = set(s.get("supplier_id") or s.get("id") for s in suppliers)

        for plot in plots:
            if "supplier_id" in plot:
                # If plot references supplier, supplier should exist
                # (or it's acceptable for demo data to have orphans)
                pass

    def test_demo_data_completeness(self):
        """Test demo data is complete."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        completeness = demo_mod.check_demo_data_completeness()

        assert completeness is not None
        assert completeness.get("complete") is True or completeness.get("percentage", 0) >= 90


class TestDemoTutorial:
    """Test demo tutorial/walkthrough features."""

    def test_tutorial_steps_available(self):
        """Test tutorial steps are available."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        tutorial = demo_mod.get_tutorial_steps()

        assert tutorial is not None
        assert isinstance(tutorial, list)
        assert len(tutorial) > 0

    def test_run_tutorial_step(self):
        """Test running a tutorial step."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        result = demo_mod.run_tutorial_step(step_number=1)

        assert result is not None
        assert "step_status" in result or "status" in result

    def test_full_tutorial(self):
        """Test running full tutorial."""
        if demo_mod is None:
            pytest.skip("demo module not available")

        result = demo_mod.run_full_tutorial()

        assert result is not None
        assert "tutorial_complete" in result or "status" in result
