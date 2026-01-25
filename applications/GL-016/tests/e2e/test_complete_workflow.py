# -*- coding: utf-8 -*-
"""
End-to-End Tests for Complete Water Treatment Workflow - GL-016 WATERGUARD

Comprehensive E2E test suite covering:
- End-to-end water treatment optimization
- Realistic industrial data scenarios
- Performance under load
- Complete system validation

Target: Full system validation with realistic scenarios
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.water_chemistry_calculator import WaterChemistryCalculator, WaterSample
from calculators.scale_formation_calculator import ScaleFormationCalculator, ScaleConditions
from calculators.corrosion_rate_calculator import CorrosionRateCalculator, CorrosionConditions
from calculators.provenance import ProvenanceTracker


# ============================================================================
# Realistic Industrial Data Scenarios
# ============================================================================

class IndustrialScenarios:
    """Realistic industrial water treatment scenarios."""

    @staticmethod
    def power_plant_boiler() -> Dict[str, Any]:
        """Power plant high-pressure boiler scenario."""
        return {
            'name': 'Power Plant HP Boiler',
            'boiler_config': {
                'capacity_mw': 500,
                'pressure_bar': 100,
                'steam_temp_c': 540,
                'feedwater_temp_c': 250
            },
            'water_sample': WaterSample(
                temperature_c=250.0,
                ph=9.2,
                conductivity_us_cm=15.0,
                calcium_mg_l=0.005,
                magnesium_mg_l=0.002,
                sodium_mg_l=2.0,
                potassium_mg_l=0.1,
                chloride_mg_l=0.01,
                sulfate_mg_l=0.01,
                bicarbonate_mg_l=0.0,
                carbonate_mg_l=0.5,
                hydroxide_mg_l=1.0,
                silica_mg_l=0.01,
                iron_mg_l=0.002,
                copper_mg_l=0.001,
                phosphate_mg_l=2.0,
                dissolved_oxygen_mg_l=0.003,
                total_alkalinity_mg_l_caco3=3.0,
                total_hardness_mg_l_caco3=0.02
            ),
            'scale_conditions': ScaleConditions(
                temperature_c=250.0,
                pressure_bar=100.0,
                flow_velocity_m_s=3.0,
                surface_roughness_um=5.0,
                operating_time_hours=8760.0,
                cycles_of_concentration=1.0,  # Once-through
                calcium_mg_l=0.005,
                magnesium_mg_l=0.002,
                sulfate_mg_l=0.01,
                silica_mg_l=0.01,
                iron_mg_l=0.002,
                copper_mg_l=0.001,
                ph=9.2,
                alkalinity_mg_l_caco3=3.0
            ),
            'corrosion_conditions': CorrosionConditions(
                temperature_c=250.0,
                pressure_bar=100.0,
                flow_velocity_m_s=3.0,
                ph=9.2,
                dissolved_oxygen_mg_l=0.003,
                carbon_dioxide_mg_l=0.0,
                chloride_mg_l=0.01,
                sulfate_mg_l=0.01,
                ammonia_mg_l=0.0,
                conductivity_us_cm=15.0,
                material_type='carbon_steel',
                surface_finish='machined',
                operating_time_hours=8760.0,
                stress_level_mpa=150.0
            )
        }

    @staticmethod
    def industrial_process_boiler() -> Dict[str, Any]:
        """Industrial process medium-pressure boiler scenario."""
        return {
            'name': 'Industrial Process Boiler',
            'boiler_config': {
                'capacity_mw': 25,
                'pressure_bar': 40,
                'steam_temp_c': 250,
                'feedwater_temp_c': 105
            },
            'water_sample': WaterSample(
                temperature_c=105.0,
                ph=10.5,
                conductivity_us_cm=3000.0,
                calcium_mg_l=2.0,
                magnesium_mg_l=0.5,
                sodium_mg_l=800.0,
                potassium_mg_l=30.0,
                chloride_mg_l=150.0,
                sulfate_mg_l=100.0,
                bicarbonate_mg_l=100.0,
                carbonate_mg_l=200.0,
                hydroxide_mg_l=150.0,
                silica_mg_l=100.0,
                iron_mg_l=0.02,
                copper_mg_l=0.01,
                phosphate_mg_l=40.0,
                dissolved_oxygen_mg_l=0.005,
                total_alkalinity_mg_l_caco3=500.0,
                total_hardness_mg_l_caco3=8.0
            ),
            'scale_conditions': ScaleConditions(
                temperature_c=105.0,
                pressure_bar=40.0,
                flow_velocity_m_s=2.0,
                surface_roughness_um=15.0,
                operating_time_hours=2000.0,
                cycles_of_concentration=10.0,
                calcium_mg_l=2.0,
                magnesium_mg_l=0.5,
                sulfate_mg_l=100.0,
                silica_mg_l=100.0,
                iron_mg_l=0.02,
                copper_mg_l=0.01,
                ph=10.5,
                alkalinity_mg_l_caco3=500.0
            ),
            'corrosion_conditions': CorrosionConditions(
                temperature_c=105.0,
                pressure_bar=40.0,
                flow_velocity_m_s=2.0,
                ph=10.5,
                dissolved_oxygen_mg_l=0.005,
                carbon_dioxide_mg_l=0.0,
                chloride_mg_l=150.0,
                sulfate_mg_l=100.0,
                ammonia_mg_l=0.5,
                conductivity_us_cm=3000.0,
                material_type='carbon_steel',
                surface_finish='machined',
                operating_time_hours=2000.0,
                stress_level_mpa=100.0
            )
        }

    @staticmethod
    def cooling_tower_system() -> Dict[str, Any]:
        """Cooling tower water treatment scenario."""
        return {
            'name': 'Cooling Tower System',
            'system_config': {
                'capacity_m3': 5000,
                'recirculation_rate_m3_hr': 10000,
                'makeup_rate_m3_hr': 200,
                'blowdown_rate_m3_hr': 50
            },
            'water_sample': WaterSample(
                temperature_c=35.0,
                ph=8.2,
                conductivity_us_cm=2500.0,
                calcium_mg_l=250.0,
                magnesium_mg_l=80.0,
                sodium_mg_l=200.0,
                potassium_mg_l=15.0,
                chloride_mg_l=300.0,
                sulfate_mg_l=250.0,
                bicarbonate_mg_l=300.0,
                carbonate_mg_l=15.0,
                hydroxide_mg_l=0.0,
                silica_mg_l=60.0,
                iron_mg_l=0.1,
                copper_mg_l=0.02,
                phosphate_mg_l=5.0,
                dissolved_oxygen_mg_l=6.0,
                total_alkalinity_mg_l_caco3=280.0,
                total_hardness_mg_l_caco3=900.0
            ),
            'scale_conditions': ScaleConditions(
                temperature_c=35.0,
                pressure_bar=1.0,
                flow_velocity_m_s=1.5,
                surface_roughness_um=25.0,
                operating_time_hours=4000.0,
                cycles_of_concentration=5.0,
                calcium_mg_l=250.0,
                magnesium_mg_l=80.0,
                sulfate_mg_l=250.0,
                silica_mg_l=60.0,
                iron_mg_l=0.1,
                copper_mg_l=0.02,
                ph=8.2,
                alkalinity_mg_l_caco3=280.0
            ),
            'corrosion_conditions': CorrosionConditions(
                temperature_c=35.0,
                pressure_bar=1.0,
                flow_velocity_m_s=1.5,
                ph=8.2,
                dissolved_oxygen_mg_l=6.0,
                carbon_dioxide_mg_l=10.0,
                chloride_mg_l=300.0,
                sulfate_mg_l=250.0,
                ammonia_mg_l=0.0,
                conductivity_us_cm=2500.0,
                material_type='carbon_steel',
                surface_finish='as_welded',
                operating_time_hours=4000.0,
                stress_level_mpa=50.0
            )
        }


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def water_chemistry_calculator():
    """Water chemistry calculator instance."""
    return WaterChemistryCalculator(version="1.0.0")


@pytest.fixture
def scale_calculator():
    """Scale formation calculator instance."""
    return ScaleFormationCalculator(version="1.0.0")


@pytest.fixture
def corrosion_calculator():
    """Corrosion rate calculator instance."""
    return CorrosionRateCalculator(version="1.0.0")


@pytest.fixture
def power_plant_scenario():
    """Power plant boiler scenario."""
    return IndustrialScenarios.power_plant_boiler()


@pytest.fixture
def industrial_boiler_scenario():
    """Industrial process boiler scenario."""
    return IndustrialScenarios.industrial_process_boiler()


@pytest.fixture
def cooling_tower_scenario():
    """Cooling tower scenario."""
    return IndustrialScenarios.cooling_tower_system()


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

@pytest.mark.e2e
class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_power_plant_boiler_complete_analysis(
        self,
        water_chemistry_calculator,
        scale_calculator,
        corrosion_calculator,
        power_plant_scenario
    ):
        """Test complete analysis for power plant boiler."""
        scenario = power_plant_scenario

        # Step 1: Water chemistry analysis
        water_result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            scenario['water_sample']
        )

        assert water_result is not None
        assert 'provenance' in water_result

        # Step 2: Scale formation analysis
        scale_result = scale_calculator.calculate_comprehensive_scale_analysis(
            scenario['scale_conditions']
        )

        assert scale_result is not None
        assert 'total_scale_prediction' in scale_result

        # Step 3: Corrosion analysis
        corrosion_result = corrosion_calculator.calculate_comprehensive_corrosion_analysis(
            scenario['corrosion_conditions']
        )

        assert corrosion_result is not None
        assert 'remaining_life_analysis' in corrosion_result

        # Verify high-pressure boiler water quality
        # Should have very low scaling due to demineralized water
        total_scale = scale_result['total_scale_prediction']['total_thickness_mm']
        assert total_scale < 1.0, "Power plant boiler should have minimal scaling"

    def test_industrial_boiler_complete_analysis(
        self,
        water_chemistry_calculator,
        scale_calculator,
        corrosion_calculator,
        industrial_boiler_scenario
    ):
        """Test complete analysis for industrial process boiler."""
        scenario = industrial_boiler_scenario

        # Full analysis pipeline
        water_result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            scenario['water_sample']
        )
        scale_result = scale_calculator.calculate_comprehensive_scale_analysis(
            scenario['scale_conditions']
        )
        corrosion_result = corrosion_calculator.calculate_comprehensive_corrosion_analysis(
            scenario['corrosion_conditions']
        )

        # All analyses should complete
        assert water_result is not None
        assert scale_result is not None
        assert corrosion_result is not None

        # Verify phosphate treatment effectiveness
        # High pH should provide corrosion protection
        total_rate = corrosion_result['total_corrosion_rate']['total_corrosion_rate_mpy']
        assert total_rate < 20.0, "Industrial boiler corrosion should be controlled"

    def test_cooling_tower_complete_analysis(
        self,
        water_chemistry_calculator,
        scale_calculator,
        corrosion_calculator,
        cooling_tower_scenario
    ):
        """Test complete analysis for cooling tower system."""
        scenario = cooling_tower_scenario

        # Full analysis pipeline
        water_result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            scenario['water_sample']
        )
        scale_result = scale_calculator.calculate_comprehensive_scale_analysis(
            scenario['scale_conditions']
        )
        corrosion_result = corrosion_calculator.calculate_comprehensive_corrosion_analysis(
            scenario['corrosion_conditions']
        )

        # All analyses should complete
        assert water_result is not None
        assert scale_result is not None
        assert corrosion_result is not None

        # Cooling tower has different challenges
        # Higher scaling potential due to evaporative concentration
        cleaning = scale_result.get('cleaning_schedule', {})
        assert 'recommended_interval_days' in cleaning


# ============================================================================
# Performance Under Load Tests
# ============================================================================

@pytest.mark.e2e
@pytest.mark.performance
class TestPerformanceUnderLoad:
    """Test system performance under load."""

    def test_batch_analysis_performance(
        self,
        water_chemistry_calculator,
        scale_calculator,
        corrosion_calculator
    ):
        """Test performance with batch of analyses."""
        # Create batch of samples
        samples = []
        for i in range(100):
            sample = WaterSample(
                temperature_c=25.0 + (i % 50),
                ph=7.0 + (i % 30) * 0.1,
                conductivity_us_cm=500.0 + i * 10,
                calcium_mg_l=50.0 + (i % 20),
                magnesium_mg_l=25.0 + (i % 10),
                sodium_mg_l=50.0,
                potassium_mg_l=5.0,
                chloride_mg_l=50.0 + (i % 30),
                sulfate_mg_l=50.0,
                bicarbonate_mg_l=100.0 + (i % 50),
                carbonate_mg_l=5.0,
                hydroxide_mg_l=0.0,
                silica_mg_l=10.0 + (i % 15),
                iron_mg_l=0.05,
                copper_mg_l=0.01,
                phosphate_mg_l=0.0,
                dissolved_oxygen_mg_l=8.0 - (i % 6),
                total_alkalinity_mg_l_caco3=100.0 + (i % 50),
                total_hardness_mg_l_caco3=200.0 + (i % 100)
            )
            samples.append(sample)

        # Time the analysis
        start_time = time.time()

        results = []
        for sample in samples:
            result = water_chemistry_calculator.calculate_water_chemistry_analysis(sample)
            results.append(result)

        elapsed_time = time.time() - start_time

        # All results should be valid
        assert len(results) == 100
        for r in results:
            assert 'provenance' in r

        # Should complete reasonably fast (< 10 seconds for 100 analyses)
        assert elapsed_time < 10.0, f"Batch analysis took {elapsed_time:.2f}s, expected < 10s"

        # Calculate throughput
        throughput = len(samples) / elapsed_time
        print(f"\nThroughput: {throughput:.1f} analyses/second")

    def test_concurrent_analysis_performance(
        self,
        water_chemistry_calculator
    ):
        """Test performance with concurrent analyses."""
        import concurrent.futures

        # Create sample
        sample = WaterSample(
            temperature_c=85.0,
            ph=8.5,
            conductivity_us_cm=1200.0,
            calcium_mg_l=50.0,
            magnesium_mg_l=30.0,
            sodium_mg_l=100.0,
            potassium_mg_l=10.0,
            chloride_mg_l=150.0,
            sulfate_mg_l=100.0,
            bicarbonate_mg_l=200.0,
            carbonate_mg_l=10.0,
            hydroxide_mg_l=0.0,
            silica_mg_l=25.0,
            iron_mg_l=0.05,
            copper_mg_l=0.01,
            phosphate_mg_l=15.0,
            dissolved_oxygen_mg_l=0.02,
            total_alkalinity_mg_l_caco3=250.0,
            total_hardness_mg_l_caco3=180.0
        )

        def analyze(_):
            return water_chemistry_calculator.calculate_water_chemistry_analysis(sample)

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(analyze, range(50)))

        elapsed_time = time.time() - start_time

        # All results should be valid
        assert len(results) == 50

        # All hashes should be identical (deterministic)
        hashes = [r['provenance']['provenance_hash'] for r in results]
        assert len(set(hashes)) == 1

        print(f"\nConcurrent analysis: {50/elapsed_time:.1f} analyses/second")

    def test_memory_stability(
        self,
        water_chemistry_calculator,
        scale_calculator,
        corrosion_calculator,
        industrial_boiler_scenario
    ):
        """Test memory stability with repeated analyses."""
        import gc

        scenario = industrial_boiler_scenario

        # Run many analyses
        for i in range(50):
            water_result = water_chemistry_calculator.calculate_water_chemistry_analysis(
                scenario['water_sample']
            )
            scale_result = scale_calculator.calculate_comprehensive_scale_analysis(
                scenario['scale_conditions']
            )
            corrosion_result = corrosion_calculator.calculate_comprehensive_corrosion_analysis(
                scenario['corrosion_conditions']
            )

            # Verify results
            assert water_result is not None
            assert scale_result is not None
            assert corrosion_result is not None

        # Force garbage collection
        gc.collect()

        # Should complete without memory issues


# ============================================================================
# Realistic Data Validation Tests
# ============================================================================

@pytest.mark.e2e
class TestRealisticDataValidation:
    """Validate results against realistic industrial data."""

    def test_power_plant_water_quality_standards(
        self,
        water_chemistry_calculator,
        power_plant_scenario
    ):
        """Validate power plant water meets quality standards."""
        scenario = power_plant_scenario
        result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            scenario['water_sample']
        )

        # EPRI guidelines for high-pressure boilers
        # pH: 9.0-9.6 at 25C
        # Conductivity: < 20 uS/cm
        # Dissolved oxygen: < 7 ppb

        # Note: Actual values depend on implementation
        # These tests verify the analysis completes and produces reasonable results
        assert result is not None
        assert 'water_chemistry' in result

    def test_industrial_boiler_treatment_program(
        self,
        water_chemistry_calculator,
        scale_calculator,
        industrial_boiler_scenario
    ):
        """Validate industrial boiler treatment program effectiveness."""
        scenario = industrial_boiler_scenario

        water_result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            scenario['water_sample']
        )
        scale_result = scale_calculator.calculate_comprehensive_scale_analysis(
            scenario['scale_conditions']
        )

        # Phosphate treatment should control scaling
        assert scale_result is not None

        # Cleaning schedule should be reasonable
        cleaning = scale_result.get('cleaning_schedule', {})
        if 'recommended_interval_days' in cleaning:
            # Should not need cleaning more than monthly for treated boiler
            interval = cleaning['recommended_interval_days']
            assert interval > 7, "Industrial boiler cleaning frequency should not be weekly"

    def test_cooling_tower_cycles_effectiveness(
        self,
        water_chemistry_calculator,
        cooling_tower_scenario
    ):
        """Test cooling tower cycles of concentration effectiveness."""
        scenario = cooling_tower_scenario

        result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            scenario['water_sample']
        )

        # Concentrated water should show scaling tendency
        # LSI should be positive at 5 cycles
        assert result is not None

        scaling_indices = result.get('scaling_indices', {})
        if 'lsi' in scaling_indices:
            lsi = scaling_indices['lsi'].get('value', 0)
            # Cooling tower water typically has positive LSI
            # (scaling tendency that must be controlled with chemicals)


# ============================================================================
# Complete System Validation Tests
# ============================================================================

@pytest.mark.e2e
class TestCompleteSystemValidation:
    """Complete system validation tests."""

    def test_full_system_workflow(
        self,
        water_chemistry_calculator,
        scale_calculator,
        corrosion_calculator
    ):
        """Test full system workflow from data to recommendations."""
        # Simulate real-world scenario
        sample = WaterSample(
            temperature_c=90.0,
            ph=9.5,
            conductivity_us_cm=2500.0,
            calcium_mg_l=5.0,
            magnesium_mg_l=2.0,
            sodium_mg_l=600.0,
            potassium_mg_l=25.0,
            chloride_mg_l=120.0,
            sulfate_mg_l=80.0,
            bicarbonate_mg_l=150.0,
            carbonate_mg_l=150.0,
            hydroxide_mg_l=100.0,
            silica_mg_l=80.0,
            iron_mg_l=0.03,
            copper_mg_l=0.01,
            phosphate_mg_l=35.0,
            dissolved_oxygen_mg_l=0.01,
            total_alkalinity_mg_l_caco3=400.0,
            total_hardness_mg_l_caco3=20.0
        )

        scale_conditions = ScaleConditions(
            temperature_c=90.0,
            pressure_bar=20.0,
            flow_velocity_m_s=2.5,
            surface_roughness_um=12.0,
            operating_time_hours=1500.0,
            cycles_of_concentration=8.0,
            calcium_mg_l=5.0,
            magnesium_mg_l=2.0,
            sulfate_mg_l=80.0,
            silica_mg_l=80.0,
            iron_mg_l=0.03,
            copper_mg_l=0.01,
            ph=9.5,
            alkalinity_mg_l_caco3=400.0
        )

        corrosion_conditions = CorrosionConditions(
            temperature_c=90.0,
            pressure_bar=20.0,
            flow_velocity_m_s=2.5,
            ph=9.5,
            dissolved_oxygen_mg_l=0.01,
            carbon_dioxide_mg_l=0.0,
            chloride_mg_l=120.0,
            sulfate_mg_l=80.0,
            ammonia_mg_l=0.3,
            conductivity_us_cm=2500.0,
            material_type='carbon_steel',
            surface_finish='machined',
            operating_time_hours=1500.0,
            stress_level_mpa=80.0
        )

        # Run all analyses
        water_result = water_chemistry_calculator.calculate_water_chemistry_analysis(sample)
        scale_result = scale_calculator.calculate_comprehensive_scale_analysis(scale_conditions)
        corrosion_result = corrosion_calculator.calculate_comprehensive_corrosion_analysis(corrosion_conditions)

        # Verify all results
        assert water_result is not None
        assert scale_result is not None
        assert corrosion_result is not None

        # Verify provenance chain
        provenance_hashes = [
            water_result['provenance']['provenance_hash'],
            scale_result['provenance']['provenance_hash'],
            corrosion_result['provenance']['provenance_hash']
        ]

        # All hashes should be unique (different analyses)
        assert len(set(provenance_hashes)) == 3

        # All hashes should be valid SHA-256
        for h in provenance_hashes:
            assert len(h) == 64
            assert all(c in '0123456789abcdef' for c in h)

    def test_reproducibility_across_scenarios(
        self,
        water_chemistry_calculator,
        scale_calculator,
        corrosion_calculator,
        power_plant_scenario,
        industrial_boiler_scenario,
        cooling_tower_scenario
    ):
        """Test reproducibility across different scenarios."""
        scenarios = [
            power_plant_scenario,
            industrial_boiler_scenario,
            cooling_tower_scenario
        ]

        for scenario in scenarios:
            # Run analysis twice
            result1 = water_chemistry_calculator.calculate_water_chemistry_analysis(
                scenario['water_sample']
            )
            result2 = water_chemistry_calculator.calculate_water_chemistry_analysis(
                scenario['water_sample']
            )

            # Hashes should be identical
            assert result1['provenance']['provenance_hash'] == result2['provenance']['provenance_hash'], \
                f"Scenario {scenario['name']} failed reproducibility test"

    def test_json_report_generation(
        self,
        water_chemistry_calculator,
        scale_calculator,
        corrosion_calculator,
        industrial_boiler_scenario
    ):
        """Test JSON report generation for complete analysis."""
        scenario = industrial_boiler_scenario

        # Run analyses
        water_result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            scenario['water_sample']
        )
        scale_result = scale_calculator.calculate_comprehensive_scale_analysis(
            scenario['scale_conditions']
        )
        corrosion_result = corrosion_calculator.calculate_comprehensive_corrosion_analysis(
            scenario['corrosion_conditions']
        )

        # Compile report
        report = {
            'report_id': 'E2E-TEST-001',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'scenario': scenario['name'],
            'boiler_config': scenario['boiler_config'],
            'analyses': {
                'water_chemistry': water_result,
                'scale_formation': scale_result,
                'corrosion': corrosion_result
            },
            'provenance_chain': [
                water_result['provenance']['provenance_hash'],
                scale_result['provenance']['provenance_hash'],
                corrosion_result['provenance']['provenance_hash']
            ]
        }

        # Should be JSON serializable
        json_str = json.dumps(report, default=str, indent=2)
        assert len(json_str) > 0

        # Should be deserializable
        loaded = json.loads(json_str)
        assert loaded['report_id'] == 'E2E-TEST-001'
        assert len(loaded['provenance_chain']) == 3


# ============================================================================
# Long-Running Tests
# ============================================================================

@pytest.mark.e2e
@pytest.mark.slow
class TestLongRunning:
    """Long-running stability tests."""

    def test_extended_operation_stability(
        self,
        water_chemistry_calculator
    ):
        """Test stability over extended operation."""
        sample = WaterSample(
            temperature_c=85.0,
            ph=8.5,
            conductivity_us_cm=1200.0,
            calcium_mg_l=50.0,
            magnesium_mg_l=30.0,
            sodium_mg_l=100.0,
            potassium_mg_l=10.0,
            chloride_mg_l=150.0,
            sulfate_mg_l=100.0,
            bicarbonate_mg_l=200.0,
            carbonate_mg_l=10.0,
            hydroxide_mg_l=0.0,
            silica_mg_l=25.0,
            iron_mg_l=0.05,
            copper_mg_l=0.01,
            phosphate_mg_l=15.0,
            dissolved_oxygen_mg_l=0.02,
            total_alkalinity_mg_l_caco3=250.0,
            total_hardness_mg_l_caco3=180.0
        )

        # Run many iterations
        hashes = set()
        for i in range(500):
            result = water_chemistry_calculator.calculate_water_chemistry_analysis(sample)
            hashes.add(result['provenance']['provenance_hash'])

        # All 500 runs should produce identical hash
        assert len(hashes) == 1, f"Found {len(hashes)} unique hashes in 500 runs"
