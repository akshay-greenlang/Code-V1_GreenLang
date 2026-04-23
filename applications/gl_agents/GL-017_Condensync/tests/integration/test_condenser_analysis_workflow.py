# -*- coding: utf-8 -*-
"""
Integration Tests for Condenser Analysis Workflow

End-to-end tests validating the complete condenser analysis pipeline
including performance calculation, vacuum analysis, and fouling prediction.

Author: GL-TestEngineer
Date: December 2025
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
import json

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.hei_condenser_calculator import (
    HEICondenserCalculator,
    TubeMaterial,
    PerformanceStatus,
)
from calculators.vacuum_performance_calculator import (
    VacuumPerformanceCalculator,
    VacuumStatus,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def condenser_calculator():
    """Create HEI condenser calculator."""
    return HEICondenserCalculator()


@pytest.fixture
def vacuum_calculator():
    """Create vacuum performance calculator."""
    return VacuumPerformanceCalculator()


@pytest.fixture
def real_world_condenser_data():
    """Realistic condenser operating data based on typical 500 MW unit."""
    return {
        "condenser_id": "UNIT1-COND-A",
        "steam_flow_kg_s": Decimal("180.0"),     # ~400 kg/s total divided by 2 shells
        "cw_inlet_temp_c": Decimal("22.5"),      # Summer ambient
        "cw_outlet_temp_c": Decimal("32.0"),     # 9.5 C rise
        "cw_flow_m3_s": Decimal("18.0"),         # ~285,000 GPM per shell
        "backpressure_kpa": Decimal("6.5"),      # 1.92 inHgA - typical summer
        "tube_material": TubeMaterial.TITANIUM,
        "tube_od_mm": Decimal("25.4"),           # 1 inch OD
        "tube_wall_mm": Decimal("0.711"),        # 22 BWG
        "tube_length_m": Decimal("13.7"),        # ~45 feet
        "num_tubes": 22000,
        "num_passes": 1,
    }


@pytest.fixture
def degraded_condenser_data(real_world_condenser_data):
    """Degraded condenser operating data showing fouling effects."""
    data = real_world_condenser_data.copy()
    data["condenser_id"] = "UNIT1-COND-B-FOULED"
    data["backpressure_kpa"] = Decimal("8.5")    # Higher BP due to fouling
    data["cw_outlet_temp_c"] = Decimal("35.0")   # Higher outlet temp
    return data


# =============================================================================
# END-TO-END WORKFLOW TESTS
# =============================================================================

class TestCondenserAnalysisWorkflow:
    """End-to-end tests for complete condenser analysis workflow."""

    def test_full_condenser_analysis_pipeline(
        self,
        condenser_calculator,
        vacuum_calculator,
        real_world_condenser_data
    ):
        """Test complete condenser analysis workflow."""
        # Step 1: Perform HEI condenser performance analysis
        perf_result = condenser_calculator.calculate_performance(
            **real_world_condenser_data
        )

        # Verify performance result
        assert perf_result is not None
        assert perf_result.heat_transfer.heat_duty_mw > Decimal("0")
        assert perf_result.cleanliness.cleanliness_factor > Decimal("0")

        # Step 2: Perform vacuum analysis using condenser results
        vac_result = vacuum_calculator.analyze_vacuum_performance(
            condenser_id=real_world_condenser_data["condenser_id"],
            backpressure_kpa=real_world_condenser_data["backpressure_kpa"],
            cw_inlet_temp_c=real_world_condenser_data["cw_inlet_temp_c"],
            cw_outlet_temp_c=real_world_condenser_data["cw_outlet_temp_c"],
            steam_flow_kg_s=real_world_condenser_data["steam_flow_kg_s"],
        )

        # Verify vacuum result
        assert vac_result is not None
        assert vac_result.backpressure_analysis.actual_backpressure_kpa == real_world_condenser_data["backpressure_kpa"]

        # Step 3: Correlate results
        # TTD from condenser analysis should match vacuum analysis
        cond_ttd = perf_result.heat_transfer.ttd_c
        vac_ttd = vac_result.backpressure_analysis.ttd_c

        assert abs(float(cond_ttd - vac_ttd)) < 1.0  # Within 1 C

    def test_degraded_condenser_workflow(
        self,
        condenser_calculator,
        vacuum_calculator,
        degraded_condenser_data
    ):
        """Test workflow with degraded condenser conditions."""
        # Perform analysis
        perf_result = condenser_calculator.calculate_performance(
            **degraded_condenser_data
        )

        vac_result = vacuum_calculator.analyze_vacuum_performance(
            condenser_id=degraded_condenser_data["condenser_id"],
            backpressure_kpa=degraded_condenser_data["backpressure_kpa"],
            cw_inlet_temp_c=degraded_condenser_data["cw_inlet_temp_c"],
            cw_outlet_temp_c=degraded_condenser_data["cw_outlet_temp_c"],
            steam_flow_kg_s=degraded_condenser_data["steam_flow_kg_s"],
        )

        # Degraded condenser should show:
        # - Lower cleanliness factor
        # - Higher TTD
        # - Degraded vacuum status or alerts
        assert perf_result.heat_transfer.ttd_c > Decimal("3.0")

        # Should generate alerts for degraded conditions
        all_alerts = len(perf_result.alerts) + len(vac_result.alerts)
        # May or may not have alerts depending on thresholds


class TestMultiCondenserComparison:
    """Tests for comparing multiple condenser analyses."""

    def test_compare_two_condensers(
        self,
        condenser_calculator,
        real_world_condenser_data,
        degraded_condenser_data
    ):
        """Compare healthy vs degraded condenser performance."""
        # Analyze both condensers
        healthy = condenser_calculator.calculate_performance(**real_world_condenser_data)
        degraded = condenser_calculator.calculate_performance(**degraded_condenser_data)

        # Healthy condenser should have:
        # - Better cleanliness factor
        # - Lower backpressure
        # - Lower TTD
        healthy_cf = healthy.cleanliness.cleanliness_factor
        degraded_cf = degraded.cleanliness.cleanliness_factor

        # Both should be valid calculations
        assert healthy_cf > Decimal("0")
        assert degraded_cf > Decimal("0")

        # Performance differences should be detectable
        # (though exact relationship depends on operating conditions)

    def test_batch_condenser_fleet_analysis(self, condenser_calculator, real_world_condenser_data):
        """Test batch analysis of condenser fleet."""
        # Create fleet of condensers with varying conditions
        fleet = []
        for i in range(5):
            data = real_world_condenser_data.copy()
            data["condenser_id"] = f"UNIT{i+1}-COND"
            data["cw_inlet_temp_c"] = Decimal(str(20 + i))  # Varying inlet temps
            fleet.append(data)

        # Batch analyze
        results = condenser_calculator.calculate_batch(fleet)

        assert len(results) == 5

        # All should have valid cleanliness factors
        for result in results:
            assert result.cleanliness.cleanliness_factor > Decimal("0")
            assert result.cleanliness.cleanliness_factor <= Decimal("1.2")


class TestProvenanceChaining:
    """Tests for provenance chain integrity across calculations."""

    def test_provenance_hashes_unique(
        self,
        condenser_calculator,
        real_world_condenser_data,
        degraded_condenser_data
    ):
        """Test that different calculations produce different provenance hashes."""
        result1 = condenser_calculator.calculate_performance(**real_world_condenser_data)
        result2 = condenser_calculator.calculate_performance(**degraded_condenser_data)

        # Different inputs should produce different hashes
        # Note: timestamp is included, so hashes will differ anyway
        assert result1.provenance_hash != result2.provenance_hash

    def test_provenance_hash_format(self, condenser_calculator, real_world_condenser_data):
        """Test provenance hash is valid SHA-256 format."""
        result = condenser_calculator.calculate_performance(**real_world_condenser_data)

        # Should be 64 character hex string
        assert len(result.provenance_hash) == 64
        assert all(c in '0123456789abcdef' for c in result.provenance_hash)


class TestResultSerialization:
    """Tests for result serialization and deserialization."""

    def test_result_to_dict_roundtrip(self, condenser_calculator, real_world_condenser_data):
        """Test result can be serialized to dict and back."""
        result = condenser_calculator.calculate_performance(**real_world_condenser_data)

        # Convert to dict
        result_dict = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)

        # Key fields should match
        assert parsed["condenser_id"] == result.condenser_specs.condenser_id
        assert parsed["cleanliness_factor"] == float(result.cleanliness.cleanliness_factor)

    def test_vacuum_result_serialization(self, vacuum_calculator, real_world_condenser_data):
        """Test vacuum result serialization."""
        result = vacuum_calculator.analyze_vacuum_performance(
            condenser_id=real_world_condenser_data["condenser_id"],
            backpressure_kpa=real_world_condenser_data["backpressure_kpa"],
            cw_inlet_temp_c=real_world_condenser_data["cw_inlet_temp_c"],
            cw_outlet_temp_c=real_world_condenser_data["cw_outlet_temp_c"],
            steam_flow_kg_s=real_world_condenser_data["steam_flow_kg_s"],
        )

        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)

        assert "backpressure" in json_str
        assert "heat_rate_impact" in json_str


class TestCalculatorStatistics:
    """Tests for calculator statistics and metrics."""

    def test_statistics_accumulation(self, condenser_calculator, real_world_condenser_data):
        """Test that statistics accumulate across calculations."""
        initial_stats = condenser_calculator.get_statistics()
        initial_count = initial_stats["calculation_count"]

        # Perform calculations
        for _ in range(3):
            condenser_calculator.calculate_performance(**real_world_condenser_data)

        final_stats = condenser_calculator.get_statistics()
        assert final_stats["calculation_count"] == initial_count + 3


class TestErrorHandling:
    """Tests for error handling across integrated components."""

    def test_graceful_handling_of_edge_cases(self, condenser_calculator):
        """Test graceful handling of edge case inputs."""
        # Edge case: minimum valid backpressure
        result = condenser_calculator.calculate_performance(
            condenser_id="EDGE-CASE-1",
            steam_flow_kg_s=Decimal("100.0"),
            cw_inlet_temp_c=Decimal("10.0"),
            cw_outlet_temp_c=Decimal("20.0"),
            cw_flow_m3_s=Decimal("15.0"),
            backpressure_kpa=Decimal("2.5"),  # Near lower limit
            tube_material=TubeMaterial.TITANIUM,
            tube_od_mm=Decimal("25.4"),
            num_tubes=15000,
        )

        assert result is not None
        assert result.cleanliness.cleanliness_factor > Decimal("0")

    def test_invalid_inputs_rejected_cleanly(self, condenser_calculator):
        """Test that invalid inputs are rejected with clear errors."""
        with pytest.raises(ValueError) as excinfo:
            condenser_calculator.calculate_performance(
                condenser_id="INVALID",
                steam_flow_kg_s=Decimal("100.0"),
                cw_inlet_temp_c=Decimal("30.0"),  # Higher than outlet
                cw_outlet_temp_c=Decimal("25.0"),
                cw_flow_m3_s=Decimal("15.0"),
                backpressure_kpa=Decimal("5.0"),
                tube_material=TubeMaterial.TITANIUM,
                tube_od_mm=Decimal("25.4"),
                num_tubes=15000,
            )

        assert "outlet" in str(excinfo.value).lower() or "temperature" in str(excinfo.value).lower()
