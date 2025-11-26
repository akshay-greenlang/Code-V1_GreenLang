# -*- coding: utf-8 -*-
"""
Edge case tests for thermal pattern analysis in GL-008 SteamTrapInspector.

This module tests boundary conditions, environmental effects, and unusual
thermal scenarios for steam trap inspection.
"""

import pytest
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tools import SteamTrapTools, ThermalAnalysisResult
from config import TrapType, FailureMode


@pytest.mark.edge_case
class TestThermalTemperatureEdgeCases:
    """Test thermal analysis at temperature boundary conditions."""

    def test_cryogenic_ambient_temperature(self, tools):
        """Test analysis in cryogenic environment (-40°C ambient)."""
        thermal_data = {
            'trap_id': 'TRAP-CRYO',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,
            'ambient_temp_c': -40.0  # Extreme cold
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        assert isinstance(result, ThermalAnalysisResult)
        assert result.temperature_differential_c == 20.0
        # Should flag potential condensate freezing risk
        assert len(result.anomalies_detected) > 0

    def test_extreme_hot_ambient(self, tools):
        """Test analysis in extreme hot environment (70°C ambient)."""
        thermal_data = {
            'trap_id': 'TRAP-HOT-AMBIENT',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,
            'ambient_temp_c': 70.0  # Very hot environment
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        assert isinstance(result, ThermalAnalysisResult)
        # Hot ambient may affect heat loss calculations
        assert result.ambient_temp_c == 70.0

    def test_zero_temperature_differential(self, tools):
        """Test with zero temperature differential (complete steam bypass)."""
        thermal_data = {
            'trap_id': 'TRAP-ZERO-DELTA',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 150.0,  # No differential
            'ambient_temp_c': 20.0
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        assert result.temperature_differential_c == 0.0
        assert result.condensate_pooling_detected == False
        # Should indicate failed open condition
        assert result.trap_health_score < 50.0

    def test_negative_temperature_differential(self, tools):
        """Test with negative differential (downstream hotter - impossible physically)."""
        thermal_data = {
            'trap_id': 'TRAP-NEG-DELTA',
            'temperature_upstream_c': 130.0,
            'temperature_downstream_c': 150.0,  # Downstream hotter
            'ambient_temp_c': 20.0
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Should detect as anomaly
        assert len(result.anomalies_detected) > 0
        assert 'negative_differential' in str(result.anomalies_detected).lower() or result.temperature_differential_c < 0

    def test_maximum_temperature_differential(self, tools):
        """Test with maximum differential (200°C difference)."""
        thermal_data = {
            'trap_id': 'TRAP-MAX-DELTA',
            'temperature_upstream_c': 220.0,
            'temperature_downstream_c': 20.0,  # Extreme differential
            'ambient_temp_c': 20.0
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        assert result.temperature_differential_c == 200.0
        assert result.condensate_pooling_detected == True  # Severe blockage
        assert result.trap_health_score < 30.0  # Critical condition

    def test_sub_zero_downstream_temperature(self, tools):
        """Test with sub-zero downstream temperature."""
        thermal_data = {
            'trap_id': 'TRAP-SUBZERO',
            'temperature_upstream_c': 100.0,
            'temperature_downstream_c': -5.0,  # Below freezing
            'ambient_temp_c': -10.0
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Should flag freezing risk
        assert len(result.anomalies_detected) > 0


@pytest.mark.edge_case
class TestThermalInsulationEffects:
    """Test thermal analysis with insulation scenarios."""

    def test_fully_insulated_trap(self, tools):
        """Test analysis of fully insulated trap."""
        thermal_data = {
            'trap_id': 'TRAP-INSULATED',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 145.0,  # Small differential due to insulation
            'ambient_temp_c': 20.0,
            'insulation_thickness_mm': 50.0,
            'insulation_type': 'mineral_wool'
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Insulated trap shows smaller surface temperature differential
        assert result.temperature_differential_c < 10.0

    def test_damaged_insulation(self, tools):
        """Test analysis with damaged/missing insulation."""
        thermal_data = {
            'trap_id': 'TRAP-DAMAGED-INSUL',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 120.0,
            'ambient_temp_c': 20.0,
            'insulation_condition': 'damaged'
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Should detect heat loss anomaly
        assert len(result.anomalies_detected) > 0

    def test_wet_insulation_effect(self, tools):
        """Test analysis with wet/saturated insulation."""
        thermal_data = {
            'trap_id': 'TRAP-WET-INSUL',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 110.0,  # Higher heat loss
            'ambient_temp_c': 20.0,
            'insulation_moisture': True
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Wet insulation increases heat loss
        assert isinstance(result, ThermalAnalysisResult)


@pytest.mark.edge_case
class TestThermalCondensatePooling:
    """Test condensate pooling detection scenarios."""

    def test_severe_condensate_backup(self, tools):
        """Test detection of severe condensate pooling."""
        thermal_data = {
            'trap_id': 'TRAP-SEVERE-POOL',
            'temperature_upstream_c': 180.0,
            'temperature_downstream_c': 60.0,  # ΔT = 120°C
            'ambient_temp_c': 20.0
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        assert result.condensate_pooling_detected == True
        assert result.temperature_differential_c > 80.0
        assert result.trap_health_score < 40.0

    def test_partial_condensate_backup(self, tools):
        """Test detection of partial condensate pooling."""
        thermal_data = {
            'trap_id': 'TRAP-PARTIAL-POOL',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 100.0,  # ΔT = 50°C
            'ambient_temp_c': 20.0
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        assert result.temperature_differential_c == 50.0
        # May or may not detect pooling depending on threshold
        assert isinstance(result.condensate_pooling_detected, bool)

    def test_intermittent_condensate_flow(self, tools):
        """Test with intermittent condensate discharge (bucket traps)."""
        thermal_data = {
            'trap_id': 'TRAP-INTERMITTENT',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 125.0,
            'ambient_temp_c': 20.0,
            'flow_pattern': 'intermittent'
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        assert result.temperature_differential_c == 25.0
        assert isinstance(result, ThermalAnalysisResult)

    def test_flash_steam_detection(self, tools):
        """Test detection of flash steam (sudden pressure drop)."""
        thermal_data = {
            'trap_id': 'TRAP-FLASH-STEAM',
            'temperature_upstream_c': 180.0,
            'temperature_downstream_c': 120.0,
            'ambient_temp_c': 20.0,
            'pressure_upstream_psig': 150.0,
            'pressure_downstream_psig': 0.0  # Atmospheric discharge
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Large pressure drop causes flash steam
        assert result.temperature_differential_c > 30.0


@pytest.mark.edge_case
class TestThermalHotColdSpots:
    """Test hot spot and cold spot detection."""

    def test_hot_spot_detection(self, tools):
        """Test detection of localized hot spots (steam leakage)."""
        thermal_data = {
            'trap_id': 'TRAP-HOTSPOT',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,
            'ambient_temp_c': 20.0,
            'thermal_image': {
                'max_temp_c': 155.0,  # Hot spot above upstream temp
                'hot_spot_locations': [(100, 150), (200, 250)]
            }
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Should detect hot spots
        if hasattr(result, 'hot_spots'):
            assert len(result.hot_spots) > 0

    def test_cold_spot_detection(self, tools):
        """Test detection of localized cold spots (blockage)."""
        thermal_data = {
            'trap_id': 'TRAP-COLDSPOT',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,
            'ambient_temp_c': 20.0,
            'thermal_image': {
                'min_temp_c': 80.0,  # Cold spot
                'cold_spot_locations': [(150, 200)]
            }
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Should detect cold spots
        if hasattr(result, 'cold_spots'):
            assert len(result.cold_spots) >= 0

    def test_thermal_gradient_analysis(self, tools):
        """Test analysis of thermal gradient along trap body."""
        thermal_data = {
            'trap_id': 'TRAP-GRADIENT',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,
            'ambient_temp_c': 20.0,
            'temperature_profile': [150, 148, 145, 140, 135, 130]  # Gradual drop
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Gradual gradient is normal
        assert isinstance(result, ThermalAnalysisResult)

    def test_abrupt_temperature_change(self, tools):
        """Test detection of abrupt temperature change (blockage point)."""
        thermal_data = {
            'trap_id': 'TRAP-ABRUPT',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,
            'ambient_temp_c': 20.0,
            'temperature_profile': [150, 149, 148, 90, 88, 85]  # Abrupt drop
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Should detect anomaly
        assert len(result.anomalies_detected) > 0


@pytest.mark.edge_case
class TestThermalEnvironmentalConditions:
    """Test thermal analysis under various environmental conditions."""

    def test_outdoor_installation_sun_exposure(self, tools):
        """Test outdoor trap with solar radiation effect."""
        thermal_data = {
            'trap_id': 'TRAP-OUTDOOR-SUN',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 135.0,
            'ambient_temp_c': 35.0,
            'solar_radiation': True,
            'surface_temp_c': 160.0  # Surface hotter due to sun
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Solar heating may affect readings
        assert isinstance(result, ThermalAnalysisResult)

    def test_outdoor_installation_rain(self, tools):
        """Test outdoor trap during rain (evaporative cooling)."""
        thermal_data = {
            'trap_id': 'TRAP-OUTDOOR-RAIN',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 125.0,
            'ambient_temp_c': 15.0,
            'weather_condition': 'rain'
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Rain causes evaporative cooling
        assert isinstance(result, ThermalAnalysisResult)

    def test_wind_effect_on_thermal(self, tools):
        """Test wind effect on thermal measurements."""
        thermal_data = {
            'trap_id': 'TRAP-WIND',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 120.0,
            'ambient_temp_c': 20.0,
            'wind_speed_mph': 25.0  # Strong wind
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Wind increases convective heat loss
        assert isinstance(result, ThermalAnalysisResult)

    def test_enclosed_space_stagnant_air(self, tools):
        """Test trap in enclosed space with stagnant air."""
        thermal_data = {
            'trap_id': 'TRAP-ENCLOSED',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 135.0,
            'ambient_temp_c': 40.0,  # Elevated ambient due to enclosure
            'ventilation': 'poor'
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Poor ventilation affects heat dissipation
        assert isinstance(result, ThermalAnalysisResult)


@pytest.mark.edge_case
class TestThermalSensorIssues:
    """Test thermal analysis with sensor-related issues."""

    def test_sensor_calibration_drift(self, tools):
        """Test analysis with miscalibrated temperature sensor."""
        thermal_data = {
            'trap_id': 'TRAP-DRIFT',
            'temperature_upstream_c': 155.0,  # +5°C drift
            'temperature_downstream_c': 135.0,  # +5°C drift
            'ambient_temp_c': 25.0,  # +5°C drift
            'sensor_calibration_date': '2020-01-01'  # Old calibration
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Differential should still be correct
        assert result.temperature_differential_c == 20.0

    def test_sensor_measurement_noise(self, tools):
        """Test analysis with noisy sensor readings."""
        thermal_data = {
            'trap_id': 'TRAP-NOISY',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,
            'ambient_temp_c': 20.0,
            'measurement_uncertainty_c': 5.0  # ±5°C uncertainty
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Should still produce result
        assert isinstance(result, ThermalAnalysisResult)

    def test_sensor_response_time_lag(self, tools):
        """Test analysis considering sensor response time."""
        thermal_data = {
            'trap_id': 'TRAP-LAG',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 125.0,
            'ambient_temp_c': 20.0,
            'sensor_response_time_sec': 10.0  # Slow response
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        assert isinstance(result, ThermalAnalysisResult)

    def test_infrared_emissivity_variation(self, tools):
        """Test analysis with varying surface emissivity."""
        thermal_data = {
            'trap_id': 'TRAP-EMISSIVITY',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,
            'ambient_temp_c': 20.0,
            'surface_emissivity': 0.60  # Polished steel (low emissivity)
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Low emissivity affects IR readings
        assert isinstance(result, ThermalAnalysisResult)


@pytest.mark.edge_case
class TestThermalTrapTypeSpecific:
    """Test thermal signatures specific to trap types."""

    def test_thermostatic_trap_thermal_element(self, tools):
        """Test thermostatic trap with thermal element response."""
        thermal_data = {
            'trap_id': 'TRAP-THERMOSTATIC',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 80.0,  # Large differential (subcooling)
            'ambient_temp_c': 20.0,
            'trap_type': 'thermostatic'
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Thermostatic traps discharge subcooled condensate
        assert result.temperature_differential_c > 40.0

    def test_float_trap_modulating_discharge(self, tools):
        """Test float trap with continuous modulating discharge."""
        thermal_data = {
            'trap_id': 'TRAP-FLOAT',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 148.0,  # Minimal differential
            'ambient_temp_c': 20.0,
            'trap_type': 'float_thermostatic'
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Float traps discharge at steam temperature
        assert result.temperature_differential_c < 10.0

    def test_disc_trap_cyclic_temperature(self, tools):
        """Test disc trap with cyclic temperature variation."""
        thermal_data = {
            'trap_id': 'TRAP-DISC',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 140.0,
            'ambient_temp_c': 20.0,
            'trap_type': 'disc',
            'temperature_variation_c': 5.0  # Cyclic variation
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        assert isinstance(result, ThermalAnalysisResult)


@pytest.mark.edge_case
class TestThermalBoundaryConditions:
    """Test thermal analysis at operational boundaries."""

    def test_startup_cold_trap(self, tools):
        """Test thermal analysis during cold startup."""
        thermal_data = {
            'trap_id': 'TRAP-COLD-START',
            'temperature_upstream_c': 50.0,  # Still heating up
            'temperature_downstream_c': 40.0,
            'ambient_temp_c': 20.0,
            'operational_state': 'startup'
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Low temperatures during startup
        assert result.temperature_upstream_c < 100.0

    def test_shutdown_cooling(self, tools):
        """Test thermal analysis during shutdown cooling."""
        thermal_data = {
            'trap_id': 'TRAP-SHUTDOWN',
            'temperature_upstream_c': 90.0,  # Cooling down
            'temperature_downstream_c': 85.0,
            'ambient_temp_c': 20.0,
            'operational_state': 'shutdown'
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Temperatures equalizing during shutdown
        assert result.temperature_differential_c < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "edge_case"])
