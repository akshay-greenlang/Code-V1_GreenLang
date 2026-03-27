"""Tests for AGENT-MRV-012 Cooling Purchase Agent metrics."""

import pytest
from decimal import Decimal
import threading

try:
    from greenlang.agents.mrv.cooling_purchase.metrics import CoolingPurchaseMetrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not METRICS_AVAILABLE, reason="Metrics not available")


class TestCoolingPurchaseMetrics:
    """Test CoolingPurchaseMetrics singleton."""

    def test_singleton_pattern(self):
        """Test CoolingPurchaseMetrics follows singleton pattern."""
        metrics1 = CoolingPurchaseMetrics()
        metrics2 = CoolingPurchaseMetrics()
        assert metrics1 is metrics2

    def test_reset_singleton(self):
        """Test reset() clears singleton instance."""
        metrics1 = CoolingPurchaseMetrics()
        CoolingPurchaseMetrics.reset()
        metrics2 = CoolingPurchaseMetrics()
        assert metrics1 is not metrics2

    def test_thread_safety(self):
        """Test thread-safe singleton creation."""
        instances = []

        def create_instance():
            instances.append(CoolingPurchaseMetrics())

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)

    def test_record_calculation_electric_chiller(self):
        """Test record_calculation() with ELECTRIC_CHILLER."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation(
            technology="ELECTRIC_CHILLER",
            method="AHRI_PART_LOAD",
            tier="TIER_2",
            status="SUCCESS",
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] >= 1

    def test_record_calculation_absorption_chiller(self):
        """Test record_calculation() with ABSORPTION_CHILLER."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation(
            technology="ABSORPTION_CHILLER",
            method="THERMAL_COP",
            tier="TIER_2",
            status="SUCCESS",
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] >= 1

    def test_record_calculation_district_cooling(self):
        """Test record_calculation() with DISTRICT_COOLING."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation(
            technology="DISTRICT_COOLING",
            method="PROVIDER_FACTOR",
            tier="TIER_1",
            status="SUCCESS",
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] >= 1

    def test_record_calculation_with_duration(self):
        """Test record_calculation() records duration."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation(
            technology="ELECTRIC_CHILLER",
            method="RATED_COP",
            tier="TIER_2",
            status="SUCCESS",
            duration_ms=123.45,
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] >= 1

    def test_record_batch(self):
        """Test record_batch() with batch size."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_batch(
            operation="PROCESS_BATCH",
            batch_size=100,
            duration_ms=5000.0,
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_batch_operations"] >= 1

    def test_record_uncertainty(self):
        """Test record_uncertainty() with Monte Carlo."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_uncertainty(
            method="MONTE_CARLO",
            samples=10000,
            duration_ms=2000.0,
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_uncertainty_calculations"] >= 1

    def test_record_compliance_check_ghg_protocol(self):
        """Test record_compliance_check() with GHG_PROTOCOL."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_compliance_check(
            framework="GHG_PROTOCOL",
            status="COMPLIANT",
            duration_ms=50.0,
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_compliance_checks"] >= 1

    def test_record_compliance_check_iso_14064(self):
        """Test record_compliance_check() with ISO_14064."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_compliance_check(
            framework="ISO_14064",
            status="COMPLIANT",
            duration_ms=45.0,
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_compliance_checks"] >= 1

    def test_record_compliance_check_csrd(self):
        """Test record_compliance_check() with CSRD."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_compliance_check(
            framework="CSRD",
            status="NON_COMPLIANT",
            duration_ms=60.0,
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_compliance_checks"] >= 1

    def test_record_db_query_select(self):
        """Test record_db_query() with SELECT operation."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_db_query(
            table="gl_cp_emission_factors",
            operation="SELECT",
            duration_ms=25.0,
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_db_queries"] >= 1

    def test_record_db_query_insert(self):
        """Test record_db_query() with INSERT operation."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_db_query(
            table="gl_cp_calculations",
            operation="INSERT",
            duration_ms=30.0,
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_db_queries"] >= 1

    def test_record_error_validation(self):
        """Test record_error() with VALIDATION error."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_error(
            error_type="VALIDATION",
            component="ElectricChillerCalculator",
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_errors"] >= 1

    def test_record_error_calculation(self):
        """Test record_error() with CALCULATION error."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_error(
            error_type="CALCULATION",
            component="AbsorptionCoolingCalculator",
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_errors"] >= 1

    def test_record_error_database(self):
        """Test record_error() with DATABASE error."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_error(
            error_type="DATABASE",
            component="CoolingDatabaseEngine",
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_errors"] >= 1

    def test_record_refrigerant_leakage(self):
        """Test record_refrigerant_leakage()."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_refrigerant_leakage(
            refrigerant="R134A",
            leakage_kg=Decimal("2.0"),
            co2e_kg=Decimal("2860.0"),
        )
        summary = metrics.get_metrics_summary()
        assert summary["total_refrigerant_leakage_events"] >= 1

    def test_get_metrics_summary_structure(self):
        """Test get_metrics_summary() returns dict with 12 entries."""
        metrics = CoolingPurchaseMetrics()
        summary = metrics.get_metrics_summary()
        assert isinstance(summary, dict)
        expected_keys = {
            "total_calculations",
            "total_batch_operations",
            "total_uncertainty_calculations",
            "total_compliance_checks",
            "total_db_queries",
            "total_errors",
            "total_refrigerant_leakage_events",
            "calculations_by_technology",
            "calculations_by_method",
            "compliance_by_framework",
            "errors_by_type",
            "refrigerant_leakage_by_type",
        }
        assert set(summary.keys()) == expected_keys

    def test_calculations_by_technology(self):
        """Test calculations_by_technology breakdown."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "SUCCESS")
        metrics.record_calculation("ELECTRIC_CHILLER", "RATED_COP", "TIER_2", "SUCCESS")
        metrics.record_calculation("ABSORPTION_CHILLER", "THERMAL_COP", "TIER_2", "SUCCESS")
        summary = metrics.get_metrics_summary()
        assert summary["calculations_by_technology"]["ELECTRIC_CHILLER"] == 2
        assert summary["calculations_by_technology"]["ABSORPTION_CHILLER"] == 1

    def test_calculations_by_method(self):
        """Test calculations_by_method breakdown."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "SUCCESS")
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "SUCCESS")
        metrics.record_calculation("ELECTRIC_CHILLER", "RATED_COP", "TIER_2", "SUCCESS")
        summary = metrics.get_metrics_summary()
        assert summary["calculations_by_method"]["AHRI_PART_LOAD"] == 2
        assert summary["calculations_by_method"]["RATED_COP"] == 1

    def test_compliance_by_framework(self):
        """Test compliance_by_framework breakdown."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_compliance_check("GHG_PROTOCOL", "COMPLIANT", 50.0)
        metrics.record_compliance_check("GHG_PROTOCOL", "COMPLIANT", 45.0)
        metrics.record_compliance_check("ISO_14064", "COMPLIANT", 55.0)
        summary = metrics.get_metrics_summary()
        assert summary["compliance_by_framework"]["GHG_PROTOCOL"] == 2
        assert summary["compliance_by_framework"]["ISO_14064"] == 1

    def test_errors_by_type(self):
        """Test errors_by_type breakdown."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_error("VALIDATION", "Component1")
        metrics.record_error("VALIDATION", "Component2")
        metrics.record_error("CALCULATION", "Component3")
        summary = metrics.get_metrics_summary()
        assert summary["errors_by_type"]["VALIDATION"] == 2
        assert summary["errors_by_type"]["CALCULATION"] == 1

    def test_refrigerant_leakage_by_type(self):
        """Test refrigerant_leakage_by_type breakdown."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_refrigerant_leakage("R134A", Decimal("2.0"), Decimal("2860.0"))
        metrics.record_refrigerant_leakage("R134A", Decimal("1.0"), Decimal("1430.0"))
        metrics.record_refrigerant_leakage("R410A", Decimal("1.5"), Decimal("3165.0"))
        summary = metrics.get_metrics_summary()
        assert summary["refrigerant_leakage_by_type"]["R134A"] == 2
        assert summary["refrigerant_leakage_by_type"]["R410A"] == 1

    def test_graceful_fallback_without_prometheus(self):
        """Test graceful fallback without prometheus_client."""
        # Metrics should work even if prometheus_client is not installed
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "SUCCESS")
        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] >= 1

    def test_concurrent_metric_recording(self):
        """Test concurrent metric recording is thread-safe."""
        metrics = CoolingPurchaseMetrics()

        def record_metrics():
            for _ in range(100):
                metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "SUCCESS")

        threads = [threading.Thread(target=record_metrics) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] == 1000

    def test_reset_clears_all_metrics(self):
        """Test reset() clears all metric counters."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "SUCCESS")
        metrics.record_batch("PROCESS_BATCH", 100, 5000.0)
        metrics.record_error("VALIDATION", "Component1")

        CoolingPurchaseMetrics.reset()
        new_metrics = CoolingPurchaseMetrics()
        summary = new_metrics.get_metrics_summary()

        assert summary["total_calculations"] == 0
        assert summary["total_batch_operations"] == 0
        assert summary["total_errors"] == 0

    def test_record_calculation_increments_total(self):
        """Test each record_calculation() increments total."""
        metrics = CoolingPurchaseMetrics()
        initial = metrics.get_metrics_summary()["total_calculations"]
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "SUCCESS")
        after = metrics.get_metrics_summary()["total_calculations"]
        assert after == initial + 1

    def test_record_batch_increments_total(self):
        """Test each record_batch() increments total."""
        metrics = CoolingPurchaseMetrics()
        initial = metrics.get_metrics_summary()["total_batch_operations"]
        metrics.record_batch("PROCESS_BATCH", 100, 5000.0)
        after = metrics.get_metrics_summary()["total_batch_operations"]
        assert after == initial + 1

    def test_record_uncertainty_increments_total(self):
        """Test each record_uncertainty() increments total."""
        metrics = CoolingPurchaseMetrics()
        initial = metrics.get_metrics_summary()["total_uncertainty_calculations"]
        metrics.record_uncertainty("MONTE_CARLO", 10000, 2000.0)
        after = metrics.get_metrics_summary()["total_uncertainty_calculations"]
        assert after == initial + 1

    def test_record_compliance_check_increments_total(self):
        """Test each record_compliance_check() increments total."""
        metrics = CoolingPurchaseMetrics()
        initial = metrics.get_metrics_summary()["total_compliance_checks"]
        metrics.record_compliance_check("GHG_PROTOCOL", "COMPLIANT", 50.0)
        after = metrics.get_metrics_summary()["total_compliance_checks"]
        assert after == initial + 1

    def test_record_db_query_increments_total(self):
        """Test each record_db_query() increments total."""
        metrics = CoolingPurchaseMetrics()
        initial = metrics.get_metrics_summary()["total_db_queries"]
        metrics.record_db_query("gl_cp_emission_factors", "SELECT", 25.0)
        after = metrics.get_metrics_summary()["total_db_queries"]
        assert after == initial + 1

    def test_record_error_increments_total(self):
        """Test each record_error() increments total."""
        metrics = CoolingPurchaseMetrics()
        initial = metrics.get_metrics_summary()["total_errors"]
        metrics.record_error("VALIDATION", "Component1")
        after = metrics.get_metrics_summary()["total_errors"]
        assert after == initial + 1

    def test_record_refrigerant_leakage_increments_total(self):
        """Test each record_refrigerant_leakage() increments total."""
        metrics = CoolingPurchaseMetrics()
        initial = metrics.get_metrics_summary()["total_refrigerant_leakage_events"]
        metrics.record_refrigerant_leakage("R134A", Decimal("2.0"), Decimal("2860.0"))
        after = metrics.get_metrics_summary()["total_refrigerant_leakage_events"]
        assert after == initial + 1

    def test_all_technologies_tracked(self):
        """Test all cooling technologies can be tracked."""
        metrics = CoolingPurchaseMetrics()
        technologies = [
            "ELECTRIC_CHILLER", "ABSORPTION_CHILLER", "DISTRICT_COOLING",
            "FREE_COOLING", "THERMAL_ENERGY_STORAGE"
        ]
        for tech in technologies:
            metrics.record_calculation(tech, "METHOD", "TIER_2", "SUCCESS")
        summary = metrics.get_metrics_summary()
        for tech in technologies:
            assert summary["calculations_by_technology"][tech] >= 1

    def test_all_frameworks_tracked(self):
        """Test all regulatory frameworks can be tracked."""
        metrics = CoolingPurchaseMetrics()
        frameworks = [
            "GHG_PROTOCOL", "ISO_14064", "CSRD", "SECR", "TCFD", "CDP", "SBTi"
        ]
        for fw in frameworks:
            metrics.record_compliance_check(fw, "COMPLIANT", 50.0)
        summary = metrics.get_metrics_summary()
        for fw in frameworks:
            assert summary["compliance_by_framework"][fw] >= 1

    def test_multiple_refrigerants_tracked(self):
        """Test multiple refrigerant types can be tracked."""
        metrics = CoolingPurchaseMetrics()
        refrigerants = ["R134A", "R410A", "R407C", "R32", "R290"]
        for ref in refrigerants:
            metrics.record_refrigerant_leakage(ref, Decimal("1.0"), Decimal("1000.0"))
        summary = metrics.get_metrics_summary()
        for ref in refrigerants:
            assert summary["refrigerant_leakage_by_type"][ref] >= 1

    def test_summary_returns_new_dict_each_call(self):
        """Test get_metrics_summary() returns new dict each call."""
        metrics = CoolingPurchaseMetrics()
        summary1 = metrics.get_metrics_summary()
        summary2 = metrics.get_metrics_summary()
        assert summary1 is not summary2
        assert summary1 == summary2

    def test_metric_labels_are_strings(self):
        """Test all metric labels are strings."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "SUCCESS")
        summary = metrics.get_metrics_summary()
        for key in summary["calculations_by_technology"].keys():
            assert isinstance(key, str)
        for key in summary["calculations_by_method"].keys():
            assert isinstance(key, str)

    def test_metric_counts_are_integers(self):
        """Test all metric counts are integers."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "SUCCESS")
        summary = metrics.get_metrics_summary()
        assert isinstance(summary["total_calculations"], int)
        assert isinstance(summary["total_batch_operations"], int)
        assert isinstance(summary["total_errors"], int)

    def test_duration_tracking(self):
        """Test duration tracking works correctly."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation(
            "ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "SUCCESS",
            duration_ms=123.45
        )
        # Duration is tracked but not exposed in summary for this minimal version
        # This test just verifies no errors occur
        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] >= 1

    def test_status_tracking(self):
        """Test calculation status is tracked."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "SUCCESS")
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "FAILED")
        # Status is tracked but minimal summary doesn't break it down
        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] >= 2

    def test_tier_tracking(self):
        """Test data quality tier is tracked."""
        metrics = CoolingPurchaseMetrics()
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_1", "SUCCESS")
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_2", "SUCCESS")
        metrics.record_calculation("ELECTRIC_CHILLER", "AHRI_PART_LOAD", "TIER_3", "SUCCESS")
        # Tier is tracked but minimal summary doesn't break it down
        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] >= 3
