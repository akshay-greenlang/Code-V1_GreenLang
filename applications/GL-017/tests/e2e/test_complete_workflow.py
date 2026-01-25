# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - End-to-End Workflow Tests

Comprehensive end-to-end tests for complete condenser optimization workflows
including performance testing and full system validation.

Test areas:
- Complete condenser optimization flow
- Performance benchmarking
- Load testing
- System reliability

Test coverage target: 95%+

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path
import statistics

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.efficiency_calculator import (
    EfficiencyCalculator,
    EfficiencyInput,
    EfficiencyOutput,
)
from calculators.heat_transfer_calculator import (
    HeatTransferCalculator,
    HeatTransferInput,
)
from calculators.fouling_calculator import (
    FoulingCalculator,
    FoulingInput,
)
from calculators.vacuum_calculator import (
    VacuumCalculator,
    VacuumInput,
)
from calculators.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    verify_provenance,
)


# =============================================================================
# E2E WORKFLOW ORCHESTRATOR
# =============================================================================

class E2ECondenserOptimizer:
    """End-to-end condenser optimizer for comprehensive testing."""

    def __init__(self):
        self.efficiency_calc = EfficiencyCalculator()
        self.heat_transfer_calc = HeatTransferCalculator()
        self.fouling_calc = FoulingCalculator()
        self.vacuum_calc = VacuumCalculator()
        self._execution_metrics = []

    async def run_complete_optimization(
        self,
        condenser_data: Dict[str, Any],
        cooling_water_data: Dict[str, Any],
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Run complete end-to-end optimization."""
        start_time = time.perf_counter()
        options = options or {}

        results = {
            "condenser_id": condenser_data.get("condenser_id", "UNKNOWN"),
            "timestamp": datetime.utcnow().isoformat(),
            "calculations": {},
            "provenance_chain": [],
            "alerts": [],
            "recommendations": [],
            "performance_metrics": {},
        }

        try:
            # Stage 1: Efficiency Analysis
            stage_start = time.perf_counter()
            efficiency_result, efficiency_prov = await self._calculate_efficiency(
                condenser_data, cooling_water_data
            )
            results["calculations"]["efficiency"] = self._serialize_result(efficiency_result)
            results["provenance_chain"].append(efficiency_prov)
            results["performance_metrics"]["efficiency_calc_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000

            # Stage 2: Heat Transfer Analysis
            stage_start = time.perf_counter()
            heat_transfer_result, ht_prov = await self._calculate_heat_transfer(
                condenser_data, cooling_water_data
            )
            results["calculations"]["heat_transfer"] = self._serialize_result(heat_transfer_result)
            results["provenance_chain"].append(ht_prov)
            results["performance_metrics"]["heat_transfer_calc_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000

            # Stage 3: Fouling Analysis
            stage_start = time.perf_counter()
            fouling_result, fouling_prov = await self._calculate_fouling(
                condenser_data, cooling_water_data, efficiency_result
            )
            results["calculations"]["fouling"] = self._serialize_result(fouling_result)
            results["provenance_chain"].append(fouling_prov)
            results["performance_metrics"]["fouling_calc_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000

            # Stage 4: Vacuum Analysis
            stage_start = time.perf_counter()
            vacuum_result, vacuum_prov = await self._calculate_vacuum(
                condenser_data, cooling_water_data
            )
            results["calculations"]["vacuum"] = self._serialize_result(vacuum_result)
            results["provenance_chain"].append(vacuum_prov)
            results["performance_metrics"]["vacuum_calc_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000

            # Stage 5: Alert Generation
            results["alerts"] = self._generate_alerts(
                efficiency_result, heat_transfer_result, fouling_result, vacuum_result
            )

            # Stage 6: Recommendation Generation
            results["recommendations"] = self._generate_recommendations(
                efficiency_result, fouling_result, vacuum_result
            )

            # Verify provenance chain
            results["provenance_valid"] = all(
                verify_provenance(p) for p in results["provenance_chain"]
            )

            results["status"] = "success"

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)

        # Total execution time
        results["performance_metrics"]["total_execution_ms"] = (
            time.perf_counter() - start_time
        ) * 1000

        self._execution_metrics.append(results["performance_metrics"])

        return results

    async def _calculate_efficiency(
        self,
        condenser_data: Dict,
        cooling_water_data: Dict
    ) -> Tuple[EfficiencyOutput, ProvenanceRecord]:
        """Calculate efficiency asynchronously."""
        input_data = EfficiencyInput(
            steam_temp_c=condenser_data.get("steam_temp_c", 40.0),
            cw_inlet_temp_c=cooling_water_data.get("inlet_temp_c", 25.0),
            cw_outlet_temp_c=cooling_water_data.get("outlet_temp_c", 35.0),
            cw_flow_rate_m3_hr=cooling_water_data.get("flow_m3_hr", 50000.0),
            heat_duty_mw=condenser_data.get("heat_duty_mw", 200.0),
            turbine_output_mw=condenser_data.get("turbine_output_mw", 300.0),
            design_backpressure_mmhg=condenser_data.get("design_bp_mmhg", 50.8),
            actual_backpressure_mmhg=condenser_data.get("actual_bp_mmhg", 55.0),
            design_u_value_w_m2k=condenser_data.get("design_u_w_m2k", 3500.0),
            actual_u_value_w_m2k=condenser_data.get("actual_u_w_m2k", 3000.0),
            heat_transfer_area_m2=condenser_data.get("area_m2", 17500.0),
        )
        return self.efficiency_calc.calculate(input_data)

    async def _calculate_heat_transfer(
        self,
        condenser_data: Dict,
        cooling_water_data: Dict
    ) -> Tuple[Any, ProvenanceRecord]:
        """Calculate heat transfer asynchronously."""
        input_data = HeatTransferInput(
            heat_duty_mw=condenser_data.get("heat_duty_mw", 200.0),
            steam_temp_c=condenser_data.get("steam_temp_c", 40.0),
            cw_inlet_temp_c=cooling_water_data.get("inlet_temp_c", 25.0),
            cw_outlet_temp_c=cooling_water_data.get("outlet_temp_c", 35.0),
            cw_flow_rate_m3_hr=cooling_water_data.get("flow_m3_hr", 50000.0),
            tube_od_mm=condenser_data.get("tube_od_mm", 25.4),
            tube_id_mm=condenser_data.get("tube_id_mm", 23.4),
            tube_length_m=condenser_data.get("tube_length_m", 12.0),
            tube_count=condenser_data.get("tube_count", 18500),
            tube_material=condenser_data.get("tube_material", "titanium"),
            design_u_value_w_m2k=condenser_data.get("design_u_w_m2k", 3500.0),
            fouling_factor_m2k_w=condenser_data.get("fouling_factor", 0.00015),
        )
        return self.heat_transfer_calc.calculate(input_data)

    async def _calculate_fouling(
        self,
        condenser_data: Dict,
        cooling_water_data: Dict,
        efficiency_result: EfficiencyOutput
    ) -> Tuple[Any, ProvenanceRecord]:
        """Calculate fouling asynchronously."""
        input_data = FoulingInput(
            tube_material=condenser_data.get("tube_material", "titanium"),
            cooling_water_source=cooling_water_data.get("source", "cooling_tower"),
            cooling_water_tds_ppm=cooling_water_data.get("tds_ppm", 2000.0),
            cooling_water_ph=cooling_water_data.get("ph", 7.8),
            cooling_water_temp_c=cooling_water_data.get("inlet_temp_c", 25.0),
            tube_velocity_m_s=2.2,  # From heat transfer calc
            operating_hours=condenser_data.get("operating_hours", 4000.0),
            biocide_treatment=cooling_water_data.get("biocide", "oxidizing"),
            current_cleanliness_factor=efficiency_result.cleanliness_factor,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=cooling_water_data.get("coc", 4.0),
        )
        return self.fouling_calc.calculate(input_data)

    async def _calculate_vacuum(
        self,
        condenser_data: Dict,
        cooling_water_data: Dict
    ) -> Tuple[Any, ProvenanceRecord]:
        """Calculate vacuum asynchronously."""
        input_data = VacuumInput(
            steam_temp_c=condenser_data.get("steam_temp_c", 40.0),
            heat_load_mw=condenser_data.get("heat_duty_mw", 200.0),
            cw_inlet_temp_c=cooling_water_data.get("inlet_temp_c", 25.0),
            cw_flow_rate_m3_hr=cooling_water_data.get("flow_m3_hr", 50000.0),
            air_inleakage_rate_kg_hr=condenser_data.get("air_inleakage_kg_hr", 1.0),
            design_vacuum_mbar=condenser_data.get("design_vacuum_mbar", 50.0),
        )
        return self.vacuum_calc.calculate(input_data)

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        if hasattr(result, '__dict__'):
            return {k: v for k, v in vars(result).items() if not k.startswith('_')}
        return {"value": result}

    def _generate_alerts(self, efficiency, heat_transfer, fouling, vacuum) -> List[Dict]:
        """Generate alerts from results."""
        alerts = []

        if efficiency.cleanliness_factor < 0.70:
            alerts.append({
                "level": "critical",
                "type": "cleanliness",
                "message": f"Cleanliness factor critical: {efficiency.cleanliness_factor:.2f}",
            })
        elif efficiency.cleanliness_factor < 0.85:
            alerts.append({
                "level": "warning",
                "type": "cleanliness",
                "message": f"Cleanliness factor below target: {efficiency.cleanliness_factor:.2f}",
            })

        if efficiency.heat_rate_deviation_kj_kwh > 150:
            alerts.append({
                "level": "critical",
                "type": "heat_rate",
                "message": f"Heat rate deviation high: {efficiency.heat_rate_deviation_kj_kwh:.1f} kJ/kWh",
            })

        return alerts

    def _generate_recommendations(self, efficiency, fouling, vacuum) -> List[Dict]:
        """Generate recommendations from results."""
        recommendations = []

        if efficiency.potential_annual_savings_usd > 100000:
            recommendations.append({
                "priority": "high",
                "action": "optimize_condenser",
                "savings": efficiency.potential_annual_savings_usd,
            })

        if fouling.cleaning_urgency in ["high", "critical"]:
            recommendations.append({
                "priority": "high",
                "action": "schedule_cleaning",
                "method": fouling.recommended_cleaning_method,
            })

        return recommendations

    def get_performance_statistics(self) -> Dict[str, float]:
        """Get performance statistics from execution history."""
        if not self._execution_metrics:
            return {}

        total_times = [m["total_execution_ms"] for m in self._execution_metrics]

        return {
            "executions": len(total_times),
            "avg_execution_ms": statistics.mean(total_times),
            "min_execution_ms": min(total_times),
            "max_execution_ms": max(total_times),
            "std_dev_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0,
        }


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def e2e_optimizer():
    """Create E2E optimizer instance."""
    return E2ECondenserOptimizer()


@pytest.fixture
def standard_condenser_data():
    """Standard condenser data for E2E testing."""
    return {
        "condenser_id": "COND-E2E-001",
        "steam_temp_c": 40.0,
        "heat_duty_mw": 200.0,
        "turbine_output_mw": 300.0,
        "design_bp_mmhg": 50.8,
        "actual_bp_mmhg": 55.0,
        "design_u_w_m2k": 3500.0,
        "actual_u_w_m2k": 3000.0,
        "area_m2": 17500.0,
        "tube_od_mm": 25.4,
        "tube_id_mm": 23.4,
        "tube_length_m": 12.0,
        "tube_count": 18500,
        "tube_material": "titanium",
        "fouling_factor": 0.00015,
        "operating_hours": 4000.0,
        "air_inleakage_kg_hr": 1.0,
        "design_vacuum_mbar": 50.0,
    }


@pytest.fixture
def standard_cooling_water_data():
    """Standard cooling water data for E2E testing."""
    return {
        "source": "cooling_tower",
        "inlet_temp_c": 25.0,
        "outlet_temp_c": 35.0,
        "flow_m3_hr": 50000.0,
        "tds_ppm": 2000.0,
        "ph": 7.8,
        "biocide": "oxidizing",
        "coc": 4.0,
    }


@pytest.fixture
def degraded_condenser_data():
    """Degraded condenser data for E2E alert testing."""
    return {
        "condenser_id": "COND-E2E-002",
        "steam_temp_c": 48.0,
        "heat_duty_mw": 180.0,
        "turbine_output_mw": 270.0,
        "design_bp_mmhg": 50.8,
        "actual_bp_mmhg": 80.0,
        "design_u_w_m2k": 3500.0,
        "actual_u_w_m2k": 2000.0,
        "area_m2": 17500.0,
        "tube_od_mm": 25.4,
        "tube_id_mm": 23.4,
        "tube_length_m": 12.0,
        "tube_count": 18500,
        "tube_material": "admiralty_brass",
        "fouling_factor": 0.0005,
        "operating_hours": 6000.0,
        "air_inleakage_kg_hr": 3.0,
        "design_vacuum_mbar": 50.0,
    }


@pytest.fixture
def multiple_condenser_dataset():
    """Multiple condenser configurations for batch testing."""
    base_config = {
        "tube_od_mm": 25.4,
        "tube_id_mm": 23.4,
        "tube_length_m": 12.0,
        "tube_count": 18500,
        "design_vacuum_mbar": 50.0,
    }

    return [
        {
            **base_config,
            "condenser_id": f"COND-BATCH-{i:03d}",
            "steam_temp_c": 38 + i * 2,
            "heat_duty_mw": 180 + i * 10,
            "turbine_output_mw": 270 + i * 15,
            "design_bp_mmhg": 50.8,
            "actual_bp_mmhg": 52 + i * 3,
            "design_u_w_m2k": 3500.0,
            "actual_u_w_m2k": 3200 - i * 100,
            "area_m2": 17500.0,
            "tube_material": ["titanium", "stainless_316", "admiralty_brass"][i % 3],
            "fouling_factor": 0.00015 + i * 0.00005,
            "operating_hours": 3000 + i * 500,
            "air_inleakage_kg_hr": 0.5 + i * 0.3,
        }
        for i in range(10)
    ]


# =============================================================================
# END-TO-END WORKFLOW TESTS
# =============================================================================

class TestE2EWorkflow:
    """End-to-end workflow test suite."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_optimization_success(
        self,
        e2e_optimizer,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test complete optimization workflow succeeds."""
        result = await e2e_optimizer.run_complete_optimization(
            standard_condenser_data,
            standard_cooling_water_data
        )

        assert result["status"] == "success"
        assert "calculations" in result
        assert "efficiency" in result["calculations"]
        assert "heat_transfer" in result["calculations"]
        assert "fouling" in result["calculations"]
        assert "vacuum" in result["calculations"]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_provenance_chain_valid(
        self,
        e2e_optimizer,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test all provenance records are valid."""
        result = await e2e_optimizer.run_complete_optimization(
            standard_condenser_data,
            standard_cooling_water_data
        )

        assert result["provenance_valid"] is True
        assert len(result["provenance_chain"]) == 4

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_alerts_generated_for_degraded(
        self,
        e2e_optimizer,
        degraded_condenser_data,
        standard_cooling_water_data
    ):
        """Test alerts are generated for degraded condenser."""
        result = await e2e_optimizer.run_complete_optimization(
            degraded_condenser_data,
            standard_cooling_water_data
        )

        assert result["status"] == "success"
        assert len(result["alerts"]) > 0

        # Check for critical alert
        critical_alerts = [a for a in result["alerts"] if a["level"] == "critical"]
        assert len(critical_alerts) > 0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_recommendations_generated(
        self,
        e2e_optimizer,
        degraded_condenser_data,
        standard_cooling_water_data
    ):
        """Test recommendations are generated."""
        result = await e2e_optimizer.run_complete_optimization(
            degraded_condenser_data,
            standard_cooling_water_data
        )

        assert len(result["recommendations"]) > 0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_condenser_id_preserved(
        self,
        e2e_optimizer,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test condenser ID is preserved in output."""
        result = await e2e_optimizer.run_complete_optimization(
            standard_condenser_data,
            standard_cooling_water_data
        )

        assert result["condenser_id"] == standard_condenser_data["condenser_id"]


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestE2EPerformance:
    """Performance test suite."""

    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_execution_time(
        self,
        e2e_optimizer,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test single optimization executes within time limit."""
        result = await e2e_optimizer.run_complete_optimization(
            standard_condenser_data,
            standard_cooling_water_data
        )

        # Should complete within 100ms
        assert result["performance_metrics"]["total_execution_ms"] < 100

    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_processing_throughput(
        self,
        e2e_optimizer,
        multiple_condenser_dataset,
        standard_cooling_water_data
    ):
        """Test batch processing throughput."""
        start_time = time.perf_counter()

        results = []
        for condenser_data in multiple_condenser_dataset:
            result = await e2e_optimizer.run_complete_optimization(
                condenser_data,
                standard_cooling_water_data
            )
            results.append(result)

        total_time = time.perf_counter() - start_time

        # All should succeed
        assert all(r["status"] == "success" for r in results)

        # Should process 10 condensers in under 2 seconds
        assert total_time < 2.0

        # Calculate throughput
        throughput = len(results) / total_time
        assert throughput > 5  # At least 5 per second

    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_repeated_execution_consistency(
        self,
        e2e_optimizer,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test performance is consistent across repeated executions."""
        # Run multiple times
        for _ in range(20):
            await e2e_optimizer.run_complete_optimization(
                standard_condenser_data,
                standard_cooling_water_data
            )

        stats = e2e_optimizer.get_performance_statistics()

        assert stats["executions"] == 20
        # Standard deviation should be low (consistent)
        assert stats["std_dev_ms"] < stats["avg_execution_ms"] * 0.5

    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_stage_timing_breakdown(
        self,
        e2e_optimizer,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test individual stage timing."""
        result = await e2e_optimizer.run_complete_optimization(
            standard_condenser_data,
            standard_cooling_water_data
        )

        metrics = result["performance_metrics"]

        # Each stage should be fast
        assert metrics["efficiency_calc_ms"] < 20
        assert metrics["heat_transfer_calc_ms"] < 20
        assert metrics["fouling_calc_ms"] < 20
        assert metrics["vacuum_calc_ms"] < 20

    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_execution(
        self,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test concurrent optimization execution."""
        optimizers = [E2ECondenserOptimizer() for _ in range(5)]

        start_time = time.perf_counter()

        # Run concurrently
        tasks = [
            opt.run_complete_optimization(standard_condenser_data, standard_cooling_water_data)
            for opt in optimizers
        ]
        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

        # All should succeed
        assert all(r["status"] == "success" for r in results)

        # Concurrent should be faster than sequential
        avg_single = sum(
            r["performance_metrics"]["total_execution_ms"] for r in results
        ) / len(results)
        assert total_time * 1000 < avg_single * len(results) * 0.8


# =============================================================================
# RELIABILITY TESTS
# =============================================================================

class TestE2EReliability:
    """Reliability test suite."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_deterministic_output(
        self,
        e2e_optimizer,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test outputs are deterministic."""
        result1 = await e2e_optimizer.run_complete_optimization(
            standard_condenser_data,
            standard_cooling_water_data
        )
        result2 = await e2e_optimizer.run_complete_optimization(
            standard_condenser_data,
            standard_cooling_water_data
        )

        # Calculation results should be identical
        assert (
            result1["calculations"]["efficiency"]["ttd_c"] ==
            result2["calculations"]["efficiency"]["ttd_c"]
        )
        assert (
            result1["calculations"]["efficiency"]["cleanliness_factor"] ==
            result2["calculations"]["efficiency"]["cleanliness_factor"]
        )

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_error_handling(self, e2e_optimizer, standard_cooling_water_data):
        """Test error handling for invalid input."""
        invalid_data = {
            "condenser_id": "INVALID",
            "steam_temp_c": -100,  # Invalid
        }

        result = await e2e_optimizer.run_complete_optimization(
            invalid_data,
            standard_cooling_water_data
        )

        assert result["status"] == "error"
        assert "error" in result

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_missing_optional_fields(
        self,
        e2e_optimizer,
        standard_cooling_water_data
    ):
        """Test handling of missing optional fields."""
        minimal_data = {
            "condenser_id": "MINIMAL",
            "steam_temp_c": 40.0,
            # Other fields use defaults
        }

        result = await e2e_optimizer.run_complete_optimization(
            minimal_data,
            standard_cooling_water_data
        )

        assert result["status"] == "success"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_boundary_values(
        self,
        e2e_optimizer,
        standard_cooling_water_data
    ):
        """Test with boundary values."""
        # Minimum values
        min_data = {
            "condenser_id": "BOUNDARY-MIN",
            "steam_temp_c": 30.0,  # Low steam temp
            "heat_duty_mw": 50.0,
            "turbine_output_mw": 75.0,
            "design_bp_mmhg": 40.0,
            "actual_bp_mmhg": 41.0,
            "design_u_w_m2k": 2000.0,
            "actual_u_w_m2k": 1900.0,
            "area_m2": 5000.0,
        }

        result = await e2e_optimizer.run_complete_optimization(
            min_data,
            standard_cooling_water_data
        )

        assert result["status"] == "success"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_high_load_values(
        self,
        e2e_optimizer,
        standard_cooling_water_data
    ):
        """Test with high load values."""
        high_load_data = {
            "condenser_id": "HIGH-LOAD",
            "steam_temp_c": 55.0,
            "heat_duty_mw": 500.0,
            "turbine_output_mw": 750.0,
            "design_bp_mmhg": 60.0,
            "actual_bp_mmhg": 90.0,
            "design_u_w_m2k": 4000.0,
            "actual_u_w_m2k": 2500.0,
            "area_m2": 30000.0,
        }

        result = await e2e_optimizer.run_complete_optimization(
            high_load_data,
            standard_cooling_water_data
        )

        assert result["status"] == "success"


# =============================================================================
# DATA INTEGRITY TESTS
# =============================================================================

class TestE2EDataIntegrity:
    """Data integrity test suite."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_provenance_hash_integrity(
        self,
        e2e_optimizer,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test provenance hashes maintain integrity."""
        result = await e2e_optimizer.run_complete_optimization(
            standard_condenser_data,
            standard_cooling_water_data
        )

        for prov in result["provenance_chain"]:
            assert verify_provenance(prov) is True

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_calculation_chain_consistency(
        self,
        e2e_optimizer,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test calculation chain produces consistent results."""
        result = await e2e_optimizer.run_complete_optimization(
            standard_condenser_data,
            standard_cooling_water_data
        )

        # Verify TTD matches expected
        efficiency = result["calculations"]["efficiency"]
        expected_ttd = standard_condenser_data["steam_temp_c"] - standard_cooling_water_data["outlet_temp_c"]
        assert abs(efficiency["ttd_c"] - expected_ttd) < 0.1

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_output_completeness(
        self,
        e2e_optimizer,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test output contains all required fields."""
        result = await e2e_optimizer.run_complete_optimization(
            standard_condenser_data,
            standard_cooling_water_data
        )

        required_fields = [
            "condenser_id",
            "timestamp",
            "calculations",
            "provenance_chain",
            "alerts",
            "recommendations",
            "performance_metrics",
            "status",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"


# =============================================================================
# LOAD TESTS
# =============================================================================

class TestE2ELoad:
    """Load test suite."""

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sustained_load(
        self,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test sustained load over multiple iterations."""
        optimizer = E2ECondenserOptimizer()

        # Run 100 iterations
        for i in range(100):
            result = await optimizer.run_complete_optimization(
                standard_condenser_data,
                standard_cooling_water_data
            )
            assert result["status"] == "success"

        stats = optimizer.get_performance_statistics()

        # Should maintain consistent performance
        assert stats["avg_execution_ms"] < 50
        assert stats["max_execution_ms"] < 100

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_stability(
        self,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test memory remains stable during extended operation."""
        import gc

        optimizer = E2ECondenserOptimizer()

        # Run many iterations
        for _ in range(50):
            await optimizer.run_complete_optimization(
                standard_condenser_data,
                standard_cooling_water_data
            )
            gc.collect()

        # Clear history to check memory
        optimizer._execution_metrics.clear()
        gc.collect()

        # Should be able to continue
        result = await optimizer.run_complete_optimization(
            standard_condenser_data,
            standard_cooling_water_data
        )
        assert result["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
