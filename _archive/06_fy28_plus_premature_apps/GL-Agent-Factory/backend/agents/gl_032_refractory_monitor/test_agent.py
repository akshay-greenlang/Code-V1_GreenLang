"""
GL-032 Refractory Monitor Agent - Golden Tests
"""

import pytest
from datetime import datetime, timedelta

from .agent import (
    RefractoryMonitorAgent,
    RefractoryMonitorInput,
    RefractoryMonitorOutput,
    SkinTemperature,
    ThermalImageData,
    RefractoryLayer,
    HistoricalReading,
)
from .models import (
    RefractoryMaterial,
    RefractoryZone,
    HealthStatus,
    MaintenancePriority,
)
from .formulas import (
    calculate_heat_loss_through_wall,
    calculate_thermal_gradient,
    calculate_health_index,
    estimate_remaining_life,
    analyze_hotspot,
)


class TestHeatLossCalculations:
    """Tests for heat loss calculations."""

    def test_fourier_law_basic(self):
        """Test basic Fourier's Law calculation."""
        q = calculate_heat_loss_through_wall(
            hot_face_temp_celsius=1000,
            cold_face_temp_celsius=100,
            wall_thickness_meters=0.3,
            thermal_conductivity_w_per_m_k=1.5,
            area_m2=10.0
        )
        # Q = k * A * dT / L = 1.5 * 10 * 900 / 0.3 = 45000 W
        assert q == pytest.approx(45000, rel=0.01)

    def test_thermal_gradient(self):
        """Test thermal gradient calculation."""
        gradient = calculate_thermal_gradient(
            hot_face_temp_celsius=1000,
            cold_face_temp_celsius=100,
            wall_thickness_meters=0.3
        )
        # Gradient = (1000 - 100) / 0.3 = 3000 C/m
        assert gradient == pytest.approx(3000, rel=0.01)


class TestHealthIndex:
    """Tests for health index calculation."""

    def test_health_index_new_refractory(self):
        """Test health index for new refractory."""
        health = calculate_health_index(
            skin_temp_celsius=80,
            design_skin_temp_celsius=80,
            age_days=100,
            design_life_days=1825,
            hotspot_count=0
        )
        assert health >= 80

    def test_health_index_aged_refractory(self):
        """Test health index for aged refractory."""
        health = calculate_health_index(
            skin_temp_celsius=100,
            design_skin_temp_celsius=80,
            age_days=1800,
            design_life_days=1825,
            hotspot_count=3
        )
        assert health < 50

    def test_health_index_overheated(self):
        """Test health index with overheated skin."""
        health = calculate_health_index(
            skin_temp_celsius=150,
            design_skin_temp_celsius=80,
            age_days=500,
            design_life_days=1825,
            hotspot_count=0
        )
        assert health < 60


class TestRemainingLife:
    """Tests for remaining life estimation."""

    def test_remaining_life_healthy(self):
        """Test remaining life for healthy refractory."""
        result = estimate_remaining_life(
            current_health_index=80,
            health_history=[(30, 82), (60, 84), (90, 85)]
        )
        assert result.remaining_life_days > 365

    def test_remaining_life_degrading(self):
        """Test remaining life for degrading refractory."""
        result = estimate_remaining_life(
            current_health_index=50,
            health_history=[(30, 55), (60, 60), (90, 65)]
        )
        assert result.remaining_life_days < 365
        assert result.degradation_rate_per_day > 0

    def test_remaining_life_critical(self):
        """Test remaining life at critical health."""
        result = estimate_remaining_life(
            current_health_index=25,
            health_history=[]
        )
        assert result.remaining_life_days == 0


class TestHotspotAnalysis:
    """Tests for hotspot analysis."""

    def test_hotspot_minor(self):
        """Test minor hotspot detection."""
        result = analyze_hotspot(
            location_x=1.0,
            location_y=2.0,
            temperature_celsius=90,
            surrounding_avg_temp=80,
            design_temp=80
        )
        assert result.severity == "MINOR"

    def test_hotspot_critical(self):
        """Test critical hotspot detection."""
        result = analyze_hotspot(
            location_x=1.0,
            location_y=2.0,
            temperature_celsius=200,
            surrounding_avg_temp=80,
            design_temp=80
        )
        assert result.severity == "CRITICAL"


class TestRefractoryMonitorAgent:
    """Integration tests for RefractoryMonitorAgent."""

    @pytest.fixture
    def agent(self):
        return RefractoryMonitorAgent()

    @pytest.fixture
    def valid_input(self):
        return RefractoryMonitorInput(
            equipment_id="FH-TEST-001",
            skin_temps=[
                SkinTemperature(x_position=0, y_position=0, temp_celsius=85, zone=RefractoryZone.SIDEWALL),
                SkinTemperature(x_position=1, y_position=0, temp_celsius=82, zone=RefractoryZone.SIDEWALL),
            ],
            thermal_images=[
                ThermalImageData(
                    image_id="TI-001",
                    capture_timestamp=datetime.utcnow(),
                    min_temp_celsius=75,
                    max_temp_celsius=95,
                    avg_temp_celsius=83,
                    zone=RefractoryZone.SIDEWALL,
                    hotspot_locations=[]
                )
            ],
            age_days=730,
            design_life_days=1825,
            material_type=RefractoryMaterial.CASTABLE,
            refractory_layers=[
                RefractoryLayer(
                    layer_name="Hot face",
                    material=RefractoryMaterial.CASTABLE,
                    thickness_m=0.15,
                    conductivity_w_per_m_k=1.8
                ),
                RefractoryLayer(
                    layer_name="Backup",
                    material=RefractoryMaterial.INSULATING_FIREBRICK,
                    thickness_m=0.1,
                    conductivity_w_per_m_k=0.3
                )
            ],
            process_temp_celsius=900,
            design_skin_temp_celsius=80
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-032"
        assert agent.AGENT_NAME == "REFRACTORY-MONITOR"

    def test_agent_run_valid_input(self, agent, valid_input):
        """Test agent runs successfully."""
        result = agent.run(valid_input)

        assert isinstance(result, RefractoryMonitorOutput)
        assert result.equipment_id == "FH-TEST-001"
        assert 0 <= result.health_index <= 100
        assert result.validation_status == "PASS"

    def test_agent_healthy_refractory(self, agent, valid_input):
        """Test agent produces good health for new refractory."""
        valid_input.age_days = 100
        result = agent.run(valid_input)

        assert result.health_index >= 60
        assert result.health_status in [HealthStatus.EXCELLENT, HealthStatus.GOOD, HealthStatus.FAIR]

    def test_agent_detects_hotspots(self, agent, valid_input):
        """Test agent detects hotspots from thermal images."""
        valid_input.thermal_images[0].hotspot_locations = [
            {"x": 1.0, "y": 2.0, "temp": 150}
        ]
        result = agent.run(valid_input)

        assert result.hotspot_count > 0


class TestGoldenCases:
    """Golden test cases."""

    @pytest.fixture
    def agent(self):
        return RefractoryMonitorAgent()

    def test_golden_case_1_healthy(self, agent):
        """Golden case: Healthy refractory."""
        input_data = RefractoryMonitorInput(
            equipment_id="GOLDEN-001",
            skin_temps=[
                SkinTemperature(x_position=0, y_position=0, temp_celsius=78, zone=RefractoryZone.SIDEWALL)
            ],
            age_days=365,
            design_life_days=1825,
            material_type=RefractoryMaterial.CASTABLE,
            process_temp_celsius=850,
            design_skin_temp_celsius=80
        )

        result = agent.run(input_data)

        assert result.health_index >= 70
        assert result.maintenance_priority in [MaintenancePriority.LOW, MaintenancePriority.SCHEDULED]

    def test_golden_case_2_critical(self, agent):
        """Golden case: Critical refractory condition."""
        input_data = RefractoryMonitorInput(
            equipment_id="GOLDEN-002",
            skin_temps=[
                SkinTemperature(x_position=0, y_position=0, temp_celsius=180, zone=RefractoryZone.SIDEWALL)
            ],
            thermal_images=[
                ThermalImageData(
                    image_id="TI-001",
                    capture_timestamp=datetime.utcnow(),
                    min_temp_celsius=100,
                    max_temp_celsius=250,
                    avg_temp_celsius=160,
                    zone=RefractoryZone.SIDEWALL,
                    hotspot_locations=[
                        {"x": 1.0, "y": 1.0, "temp": 250},
                        {"x": 2.0, "y": 1.5, "temp": 220}
                    ]
                )
            ],
            age_days=1700,
            design_life_days=1825,
            material_type=RefractoryMaterial.CASTABLE,
            process_temp_celsius=900,
            design_skin_temp_celsius=80
        )

        result = agent.run(input_data)

        assert result.health_index < 50
        assert result.maintenance_priority in [MaintenancePriority.CRITICAL, MaintenancePriority.HIGH]
        assert result.hotspot_count >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
