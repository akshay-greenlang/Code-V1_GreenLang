"""
GL-018 FLUEFLOW - End-to-End Integration Tests

Complete workflow tests from SCADA data acquisition through emissions reporting.

Target Coverage: 80%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from calculators.combustion_analyzer import CombustionAnalyzer, CombustionInput, FuelType, GasBasis
from calculators.efficiency_calculator import EfficiencyCalculator, EfficiencyInput
from calculators.air_fuel_ratio_calculator import AirFuelRatioCalculator, AirFuelRatioInput
from calculators.emissions_calculator import EmissionsCalculator, EmissionsInput


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    def test_complete_analysis_pipeline_natural_gas(self):
        """Test complete analysis pipeline for natural gas combustion."""
        # Step 1: Combustion Analysis
        combustion_analyzer = CombustionAnalyzer()
        combustion_input = CombustionInput(
            O2_pct=3.5,
            CO2_pct=12.0,
            CO_ppm=50.0,
            NOx_ppm=150.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        combustion_result, combustion_prov = combustion_analyzer.calculate(combustion_input)

        # Step 2: Efficiency Analysis (uses combustion results)
        efficiency_calculator = EfficiencyCalculator()
        efficiency_input = EfficiencyInput(
            fuel_type="Natural Gas",
            fuel_flow_rate_kg_hr=1000.0,
            O2_pct_dry=combustion_result.O2_dry_pct,
            CO2_pct_dry=combustion_result.CO2_dry_pct,
            CO_ppm=combustion_input.CO_ppm,
            flue_gas_temp_c=combustion_input.flue_gas_temp_c,
            ambient_temp_c=combustion_input.ambient_temp_c,
            excess_air_pct=combustion_result.excess_air_pct,
            heat_input_mw=10.0,
            heat_output_mw=8.5
        )

        efficiency_result, efficiency_prov = efficiency_calculator.calculate(efficiency_input)

        # Step 3: Air-Fuel Ratio Analysis
        afr_calculator = AirFuelRatioCalculator()
        afr_input = AirFuelRatioInput(
            fuel_type="Natural Gas",
            O2_measured_pct=combustion_result.O2_dry_pct
        )

        afr_result, afr_prov = afr_calculator.calculate(afr_input)

        # Step 4: Emissions Analysis
        emissions_calculator = EmissionsCalculator()
        emissions_input = EmissionsInput(
            NOx_ppm=combustion_input.NOx_ppm,
            CO_ppm=combustion_input.CO_ppm,
            SO2_ppm=0.0,
            CO2_pct=combustion_result.CO2_dry_pct,
            O2_pct=combustion_result.O2_dry_pct,
            flue_gas_temp_c=combustion_input.flue_gas_temp_c,
            flue_gas_flow_nm3_hr=50000.0,
            fuel_type="Natural Gas"
        )

        emissions_result, emissions_prov = emissions_calculator.calculate(emissions_input)

        # Validate integrated results
        assert combustion_result.is_complete_combustion is True
        assert efficiency_result.efficiency_rating in ["Excellent", "Good"]
        assert afr_result.lambda_ratio == pytest.approx(1.2, rel=0.01)
        assert emissions_result.NOx_compliance_status in ["Compliant", "Compliant (Good Margin)"]

        # Validate provenance chain
        assert len(combustion_prov.provenance_hash) == 64
        assert len(efficiency_prov.provenance_hash) == 64
        assert len(afr_prov.provenance_hash) == 64
        assert len(emissions_prov.provenance_hash) == 64

    def test_optimization_workflow(self):
        """Test optimization workflow identifying improvement opportunities."""
        # Initial suboptimal conditions
        combustion_analyzer = CombustionAnalyzer()
        initial_input = CombustionInput(
            O2_pct=6.0,  # High O2 (excessive air)
            CO2_pct=9.0,
            CO_ppm=100.0,
            NOx_ppm=180.0,
            flue_gas_temp_c=220.0,  # High stack temp
            ambient_temp_c=25.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        initial_result, _ = combustion_analyzer.calculate(initial_input)

        # Verify initial inefficiency
        assert initial_result.excess_air_pct > 40.0  # Too much excess air
        assert initial_result.combustion_quality_rating in ["Fair", "Poor"]

        # Optimized conditions
        optimized_input = CombustionInput(
            O2_pct=3.5,  # Reduced to optimal
            CO2_pct=12.0,
            CO_ppm=40.0,
            NOx_ppm=140.0,
            flue_gas_temp_c=170.0,  # Reduced stack temp
            ambient_temp_c=25.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        optimized_result, _ = combustion_analyzer.calculate(optimized_input)

        # Verify improvement
        assert optimized_result.excess_air_pct < initial_result.excess_air_pct
        assert optimized_result.combustion_quality_index > initial_result.combustion_quality_index
        assert optimized_result.combustion_quality_rating in ["Excellent", "Good"]

    def test_multi_fuel_comparison(self):
        """Test multi-fuel comparison workflow."""
        combustion_analyzer = CombustionAnalyzer()
        fuels = [
            (FuelType.NATURAL_GAS.value, 3.5, 12.0),
            (FuelType.FUEL_OIL.value, 3.0, 14.0),
            (FuelType.COAL.value, 5.0, 16.0),
        ]

        results = []
        for fuel_type, O2, CO2 in fuels:
            inputs = CombustionInput(
                O2_pct=O2,
                CO2_pct=CO2,
                CO_ppm=100.0,
                NOx_ppm=200.0,
                flue_gas_temp_c=200.0,
                ambient_temp_c=25.0,
                fuel_type=fuel_type,
                gas_basis=GasBasis.DRY.value
            )

            result, _ = combustion_analyzer.calculate(inputs)
            results.append((fuel_type, result))

        # Validate all calculations completed successfully
        assert len(results) == 3
        for fuel_type, result in results:
            assert result.stoichiometric_ratio > 1.0

    def test_continuous_monitoring_simulation(self):
        """Test continuous monitoring simulation with time-series data."""
        combustion_analyzer = CombustionAnalyzer()
        efficiency_calculator = EfficiencyCalculator()

        # Simulate 10 measurements over time
        measurements = []
        for i in range(10):
            # Slight variations in measurements
            combustion_input = CombustionInput(
                O2_pct=3.5 + (i % 3) * 0.1,
                CO2_pct=12.0 - (i % 3) * 0.1,
                CO_ppm=50.0 + (i % 5) * 10.0,
                NOx_ppm=150.0 + (i % 5) * 5.0,
                flue_gas_temp_c=180.0 + (i % 4) * 5.0,
                ambient_temp_c=25.0,
                fuel_type=FuelType.NATURAL_GAS.value,
                gas_basis=GasBasis.DRY.value
            )

            combustion_result, _ = combustion_analyzer.calculate(combustion_input)

            efficiency_input = EfficiencyInput(
                fuel_type="Natural Gas",
                fuel_flow_rate_kg_hr=1000.0,
                O2_pct_dry=combustion_result.O2_dry_pct,
                CO2_pct_dry=combustion_result.CO2_dry_pct,
                CO_ppm=combustion_input.CO_ppm,
                flue_gas_temp_c=combustion_input.flue_gas_temp_c,
                ambient_temp_c=combustion_input.ambient_temp_c,
                excess_air_pct=combustion_result.excess_air_pct,
                heat_input_mw=10.0,
                heat_output_mw=8.5
            )

            efficiency_result, _ = efficiency_calculator.christianculate(efficiency_input)

            measurements.append({
                'timestamp': datetime.now(timezone.utc),
                'combustion': combustion_result,
                'efficiency': efficiency_result
            })

        # Validate continuous measurements
        assert len(measurements) == 10
        for measurement in measurements:
            assert measurement['combustion'].combustion_quality_rating in ["Excellent", "Good", "Fair"]
            assert measurement['efficiency'].combustion_efficiency_pct > 75.0

    @pytest.mark.performance
    def test_high_throughput_processing(self):
        """Test high throughput processing (>1000 records/sec)."""
        import time

        combustion_analyzer = CombustionAnalyzer()

        # Generate 1000 test inputs
        inputs = [
            CombustionInput(
                O2_pct=3.5 + (i % 10) * 0.1,
                CO2_pct=12.0 - (i % 10) * 0.05,
                CO_ppm=50.0 + (i % 50) * 2.0,
                NOx_ppm=150.0 + (i % 50) * 1.0,
                flue_gas_temp_c=180.0 + (i % 20) * 2.0,
                ambient_temp_c=25.0,
                fuel_type=FuelType.NATURAL_GAS.value,
                gas_basis=GasBasis.DRY.value
            )
            for i in range(1000)
        ]

        start_time = time.time()
        results = []

        for input_data in inputs:
            result, _ = combustion_analyzer.calculate(input_data)
            results.append(result)

        end_time = time.time()
        duration = end_time - start_time
        throughput = len(inputs) / duration

        # Validate performance target
        assert throughput > 1000  # Target: >1000 records/sec
        assert len(results) == 1000
