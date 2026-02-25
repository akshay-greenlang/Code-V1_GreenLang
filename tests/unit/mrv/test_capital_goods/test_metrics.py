# -*- coding: utf-8 -*-
"""Unit tests for Capital Goods Agent metrics."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from greenlang.capital_goods.metrics import CapitalGoodsMetrics
from greenlang.capital_goods.models import CalculationMethod


# ============================================================================
# SINGLETON PATTERN TESTS
# ============================================================================


class TestMetricsSingleton:
    """Test metrics singleton pattern."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before each test."""
        CapitalGoodsMetrics.reset()
        yield
        CapitalGoodsMetrics.reset()

    def test_singleton_same_instance(self):
        """Test that multiple calls return the same instance."""
        metrics1 = CapitalGoodsMetrics()
        metrics2 = CapitalGoodsMetrics()
        assert metrics1 is metrics2

    def test_reset_creates_new_instance(self):
        """Test that reset() creates a new instance."""
        metrics1 = CapitalGoodsMetrics()
        old_id = id(metrics1)
        CapitalGoodsMetrics.reset()
        metrics2 = CapitalGoodsMetrics()
        assert id(metrics2) != old_id

    def test_singleton_thread_safe(self):
        """Test singleton is thread-safe."""
        import threading

        instances = []

        def get_instance():
            instances.append(CapitalGoodsMetrics())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same object
        assert all(inst is instances[0] for inst in instances)


# ============================================================================
# METRIC CREATION TESTS
# ============================================================================


class TestMetricCreation:
    """Test metric creation and initialization."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before each test."""
        CapitalGoodsMetrics.reset()
        yield
        CapitalGoodsMetrics.reset()

    @patch("greenlang.capital_goods.metrics.Counter")
    def test_calculations_total_counter_created(self, mock_counter):
        """Test calculations_total counter is created."""
        CapitalGoodsMetrics()
        mock_counter.assert_any_call(
            "gl_cg_calculations_total",
            "Total number of capital goods calculations",
            ["method", "status"],
        )

    @patch("greenlang.capital_goods.metrics.Histogram")
    def test_calculation_duration_histogram_created(self, mock_histogram):
        """Test calculation_duration histogram is created."""
        CapitalGoodsMetrics()
        mock_histogram.assert_any_call(
            "gl_cg_calculation_duration_seconds",
            "Duration of capital goods calculations",
            ["method"],
        )

    @patch("greenlang.capital_goods.metrics.Counter")
    def test_emissions_calculated_counter_created(self, mock_counter):
        """Test emissions_calculated counter is created."""
        CapitalGoodsMetrics()
        mock_counter.assert_any_call(
            "gl_cg_emissions_calculated_tco2e",
            "Total emissions calculated in tCO2e",
            ["method"],
        )

    @patch("greenlang.capital_goods.metrics.Gauge")
    def test_active_calculations_gauge_created(self, mock_gauge):
        """Test active_calculations gauge is created."""
        CapitalGoodsMetrics()
        mock_gauge.assert_any_call(
            "gl_cg_active_calculations",
            "Number of active calculations in progress",
        )

    @patch("greenlang.capital_goods.metrics.Counter")
    def test_spend_based_calculations_counter_created(self, mock_counter):
        """Test spend_based_calculations counter is created."""
        CapitalGoodsMetrics()
        mock_counter.assert_any_call(
            "gl_cg_spend_based_calculations_total",
            "Total spend-based calculations",
            ["status"],
        )

    @patch("greenlang.capital_goods.metrics.Counter")
    def test_average_data_calculations_counter_created(self, mock_counter):
        """Test average_data_calculations counter is created."""
        CapitalGoodsMetrics()
        mock_counter.assert_any_call(
            "gl_cg_average_data_calculations_total",
            "Total average-data calculations",
            ["status"],
        )

    @patch("greenlang.capital_goods.metrics.Counter")
    def test_supplier_specific_calculations_counter_created(self, mock_counter):
        """Test supplier_specific_calculations counter is created."""
        CapitalGoodsMetrics()
        mock_counter.assert_any_call(
            "gl_cg_supplier_specific_calculations_total",
            "Total supplier-specific calculations",
            ["status"],
        )

    @patch("greenlang.capital_goods.metrics.Counter")
    def test_hybrid_calculations_counter_created(self, mock_counter):
        """Test hybrid_calculations counter is created."""
        CapitalGoodsMetrics()
        mock_counter.assert_any_call(
            "gl_cg_hybrid_calculations_total",
            "Total hybrid calculations",
            ["status"],
        )

    @patch("greenlang.capital_goods.metrics.Histogram")
    def test_data_quality_scores_histogram_created(self, mock_histogram):
        """Test data_quality_scores histogram is created."""
        CapitalGoodsMetrics()
        mock_histogram.assert_any_call(
            "gl_cg_data_quality_scores",
            "Distribution of data quality scores",
            ["method"],
        )

    @patch("greenlang.capital_goods.metrics.Counter")
    def test_compliance_checks_counter_created(self, mock_counter):
        """Test compliance_checks counter is created."""
        CapitalGoodsMetrics()
        mock_counter.assert_any_call(
            "gl_cg_compliance_checks_total",
            "Total compliance checks performed",
            ["framework", "status"],
        )

    @patch("greenlang.capital_goods.metrics.Gauge")
    def test_emission_factors_loaded_gauge_created(self, mock_gauge):
        """Test emission_factors_loaded gauge is created."""
        CapitalGoodsMetrics()
        mock_gauge.assert_any_call(
            "gl_cg_emission_factors_loaded",
            "Number of emission factors loaded in memory",
            ["sector_classification"],
        )

    @patch("greenlang.capital_goods.metrics.Counter")
    def test_cache_hits_counter_created(self, mock_counter):
        """Test cache_hits counter is created."""
        CapitalGoodsMetrics()
        mock_counter.assert_any_call(
            "gl_cg_cache_hits_total",
            "Total cache hits",
            ["cache_type"],
        )


# ============================================================================
# RECORD CALCULATION TESTS
# ============================================================================


class TestRecordCalculation:
    """Test recording calculations."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before each test."""
        CapitalGoodsMetrics.reset()
        yield
        CapitalGoodsMetrics.reset()

    def test_record_calculation_success(self):
        """Test recording a successful calculation."""
        metrics = CapitalGoodsMetrics()
        metrics.calculations_total = MagicMock()

        metrics.record_calculation(
            method=CalculationMethod.SPEND_BASED,
            status="success",
        )

        metrics.calculations_total.labels.assert_called_once_with(
            method="spend_based",
            status="success",
        )
        metrics.calculations_total.labels().inc.assert_called_once()

    def test_record_calculation_failure(self):
        """Test recording a failed calculation."""
        metrics = CapitalGoodsMetrics()
        metrics.calculations_total = MagicMock()

        metrics.record_calculation(
            method=CalculationMethod.AVERAGE_DATA,
            status="failure",
        )

        metrics.calculations_total.labels.assert_called_once_with(
            method="average_data",
            status="failure",
        )

    def test_record_calculation_with_emissions(self):
        """Test recording a calculation with emissions."""
        metrics = CapitalGoodsMetrics()
        metrics.calculations_total = MagicMock()
        metrics.emissions_calculated = MagicMock()

        metrics.record_calculation(
            method=CalculationMethod.SUPPLIER_SPECIFIC,
            status="success",
            emissions_tco2e=Decimal("125.5"),
        )

        metrics.emissions_calculated.labels.assert_called_once_with(
            method="supplier_specific",
        )
        metrics.emissions_calculated.labels().inc.assert_called_once_with(125.5)

    def test_record_calculation_with_duration(self):
        """Test recording a calculation with duration."""
        metrics = CapitalGoodsMetrics()
        metrics.calculations_total = MagicMock()
        metrics.calculation_duration = MagicMock()

        metrics.record_calculation(
            method=CalculationMethod.HYBRID,
            status="success",
            duration_seconds=2.5,
        )

        metrics.calculation_duration.labels.assert_called_once_with(
            method="hybrid",
        )
        metrics.calculation_duration.labels().observe.assert_called_once_with(2.5)


# ============================================================================
# METHOD-SPECIFIC RECORDING TESTS
# ============================================================================


class TestMethodSpecificRecording:
    """Test method-specific metric recording."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before each test."""
        CapitalGoodsMetrics.reset()
        yield
        CapitalGoodsMetrics.reset()

    def test_record_spend_based_success(self):
        """Test recording spend-based calculation success."""
        metrics = CapitalGoodsMetrics()
        metrics.spend_based_calculations = MagicMock()

        metrics.record_spend_based(status="success")

        metrics.spend_based_calculations.labels.assert_called_once_with(status="success")
        metrics.spend_based_calculations.labels().inc.assert_called_once()

    def test_record_spend_based_failure(self):
        """Test recording spend-based calculation failure."""
        metrics = CapitalGoodsMetrics()
        metrics.spend_based_calculations = MagicMock()

        metrics.record_spend_based(status="failure")

        metrics.spend_based_calculations.labels.assert_called_once_with(status="failure")

    def test_record_average_data_success(self):
        """Test recording average-data calculation success."""
        metrics = CapitalGoodsMetrics()
        metrics.average_data_calculations = MagicMock()

        metrics.record_average_data(status="success")

        metrics.average_data_calculations.labels.assert_called_once_with(status="success")
        metrics.average_data_calculations.labels().inc.assert_called_once()

    def test_record_average_data_failure(self):
        """Test recording average-data calculation failure."""
        metrics = CapitalGoodsMetrics()
        metrics.average_data_calculations = MagicMock()

        metrics.record_average_data(status="failure")

        metrics.average_data_calculations.labels.assert_called_once_with(status="failure")

    def test_record_supplier_specific_success(self):
        """Test recording supplier-specific calculation success."""
        metrics = CapitalGoodsMetrics()
        metrics.supplier_specific_calculations = MagicMock()

        metrics.record_supplier_specific(status="success")

        metrics.supplier_specific_calculations.labels.assert_called_once_with(status="success")
        metrics.supplier_specific_calculations.labels().inc.assert_called_once()

    def test_record_supplier_specific_failure(self):
        """Test recording supplier-specific calculation failure."""
        metrics = CapitalGoodsMetrics()
        metrics.supplier_specific_calculations = MagicMock()

        metrics.record_supplier_specific(status="failure")

        metrics.supplier_specific_calculations.labels.assert_called_once_with(status="failure")

    def test_record_hybrid_success(self):
        """Test recording hybrid calculation success."""
        metrics = CapitalGoodsMetrics()
        metrics.hybrid_calculations = MagicMock()

        metrics.record_hybrid(status="success")

        metrics.hybrid_calculations.labels.assert_called_once_with(status="success")
        metrics.hybrid_calculations.labels().inc.assert_called_once()

    def test_record_hybrid_failure(self):
        """Test recording hybrid calculation failure."""
        metrics = CapitalGoodsMetrics()
        metrics.hybrid_calculations = MagicMock()

        metrics.record_hybrid(status="failure")

        metrics.hybrid_calculations.labels.assert_called_once_with(status="failure")


# ============================================================================
# COMPLIANCE CHECK TESTS
# ============================================================================


class TestComplianceCheckRecording:
    """Test compliance check metric recording."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before each test."""
        CapitalGoodsMetrics.reset()
        yield
        CapitalGoodsMetrics.reset()

    def test_record_compliance_check_success(self):
        """Test recording successful compliance check."""
        metrics = CapitalGoodsMetrics()
        metrics.compliance_checks = MagicMock()

        metrics.record_compliance_check(
            framework="ghg_protocol_scope3",
            status="pass",
        )

        metrics.compliance_checks.labels.assert_called_once_with(
            framework="ghg_protocol_scope3",
            status="pass",
        )
        metrics.compliance_checks.labels().inc.assert_called_once()

    def test_record_compliance_check_failure(self):
        """Test recording failed compliance check."""
        metrics = CapitalGoodsMetrics()
        metrics.compliance_checks = MagicMock()

        metrics.record_compliance_check(
            framework="csrd_esrs_e1",
            status="fail",
        )

        metrics.compliance_checks.labels.assert_called_once_with(
            framework="csrd_esrs_e1",
            status="fail",
        )


# ============================================================================
# ACTIVE CALCULATIONS TESTS
# ============================================================================


class TestActiveCalculations:
    """Test active calculations gauge."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before each test."""
        CapitalGoodsMetrics.reset()
        yield
        CapitalGoodsMetrics.reset()

    def test_inc_active_calculations(self):
        """Test incrementing active calculations."""
        metrics = CapitalGoodsMetrics()
        metrics.active_calculations = MagicMock()

        metrics.inc_active()

        metrics.active_calculations.inc.assert_called_once()

    def test_dec_active_calculations(self):
        """Test decrementing active calculations."""
        metrics = CapitalGoodsMetrics()
        metrics.active_calculations = MagicMock()

        metrics.dec_active()

        metrics.active_calculations.dec.assert_called_once()


# ============================================================================
# EMISSION FACTORS TESTS
# ============================================================================


class TestEmissionFactors:
    """Test emission factors gauge."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before each test."""
        CapitalGoodsMetrics.reset()
        yield
        CapitalGoodsMetrics.reset()

    def test_set_factors_loaded(self):
        """Test setting emission factors loaded count."""
        metrics = CapitalGoodsMetrics()
        metrics.emission_factors_loaded = MagicMock()

        metrics.set_factors_loaded(
            sector_classification="naics",
            count=1250,
        )

        metrics.emission_factors_loaded.labels.assert_called_once_with(
            sector_classification="naics",
        )
        metrics.emission_factors_loaded.labels().set.assert_called_once_with(1250)


# ============================================================================
# METRICS SUMMARY TESTS
# ============================================================================


class TestMetricsSummary:
    """Test metrics summary generation."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before each test."""
        CapitalGoodsMetrics.reset()
        yield
        CapitalGoodsMetrics.reset()

    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        metrics = CapitalGoodsMetrics()
        summary = metrics.get_metrics_summary()

        assert isinstance(summary, dict)
        assert "calculations_total" in summary
        assert "active_calculations" in summary
        assert "spend_based_calculations" in summary
        assert "average_data_calculations" in summary
        assert "supplier_specific_calculations" in summary
        assert "hybrid_calculations" in summary

    def test_metrics_summary_structure(self):
        """Test metrics summary has correct structure."""
        metrics = CapitalGoodsMetrics()
        summary = metrics.get_metrics_summary()

        assert len(summary) >= 6
        for key, value in summary.items():
            assert isinstance(key, str)


# ============================================================================
# CONTEXT MANAGER TESTS
# ============================================================================


class TestContextManagers:
    """Test context manager functionality."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before each test."""
        CapitalGoodsMetrics.reset()
        yield
        CapitalGoodsMetrics.reset()

    def test_track_calculation_context_manager(self):
        """Test track_calculation context manager."""
        metrics = CapitalGoodsMetrics()
        metrics.inc_active = MagicMock()
        metrics.dec_active = MagicMock()
        metrics.record_calculation = MagicMock()

        with metrics.track_calculation(CalculationMethod.SPEND_BASED):
            pass

        metrics.inc_active.assert_called_once()
        metrics.dec_active.assert_called_once()
        metrics.record_calculation.assert_called_once()

    def test_track_calculation_with_exception(self):
        """Test track_calculation handles exceptions."""
        metrics = CapitalGoodsMetrics()
        metrics.inc_active = MagicMock()
        metrics.dec_active = MagicMock()
        metrics.record_calculation = MagicMock()

        with pytest.raises(ValueError):
            with metrics.track_calculation(CalculationMethod.SPEND_BASED):
                raise ValueError("Test error")

        metrics.inc_active.assert_called_once()
        metrics.dec_active.assert_called_once()
        metrics.record_calculation.assert_called_once_with(
            method=CalculationMethod.SPEND_BASED,
            status="failure",
        )

    def test_track_duration_context_manager(self):
        """Test track_duration context manager."""
        metrics = CapitalGoodsMetrics()
        metrics.calculation_duration = MagicMock()

        with metrics.track_duration(CalculationMethod.AVERAGE_DATA):
            import time
            time.sleep(0.1)

        metrics.calculation_duration.labels.assert_called_once_with(
            method="average_data",
        )
        metrics.calculation_duration.labels().observe.assert_called_once()
        # Duration should be approximately 0.1 seconds
        call_args = metrics.calculation_duration.labels().observe.call_args[0][0]
        assert 0.05 < call_args < 0.2  # Allow some margin


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestMetricsEdgeCases:
    """Test metrics edge cases."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before each test."""
        CapitalGoodsMetrics.reset()
        yield
        CapitalGoodsMetrics.reset()

    def test_record_zero_emissions(self):
        """Test recording zero emissions."""
        metrics = CapitalGoodsMetrics()
        metrics.emissions_calculated = MagicMock()

        metrics.record_calculation(
            method=CalculationMethod.SPEND_BASED,
            status="success",
            emissions_tco2e=Decimal("0.0"),
        )

        metrics.emissions_calculated.labels().inc.assert_called_once_with(0.0)

    def test_record_very_large_emissions(self):
        """Test recording very large emissions."""
        metrics = CapitalGoodsMetrics()
        metrics.emissions_calculated = MagicMock()

        metrics.record_calculation(
            method=CalculationMethod.SPEND_BASED,
            status="success",
            emissions_tco2e=Decimal("999999999.99"),
        )

        metrics.emissions_calculated.labels().inc.assert_called_once_with(999999999.99)

    def test_record_very_short_duration(self):
        """Test recording very short duration."""
        metrics = CapitalGoodsMetrics()
        metrics.calculation_duration = MagicMock()

        metrics.record_calculation(
            method=CalculationMethod.HYBRID,
            status="success",
            duration_seconds=0.001,
        )

        metrics.calculation_duration.labels().observe.assert_called_once_with(0.001)

    def test_set_zero_factors_loaded(self):
        """Test setting zero emission factors loaded."""
        metrics = CapitalGoodsMetrics()
        metrics.emission_factors_loaded = MagicMock()

        metrics.set_factors_loaded(
            sector_classification="naics",
            count=0,
        )

        metrics.emission_factors_loaded.labels().set.assert_called_once_with(0)
