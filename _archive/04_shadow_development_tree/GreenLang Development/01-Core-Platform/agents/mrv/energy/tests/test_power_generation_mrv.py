# -*- coding: utf-8 -*-
"""
Tests for GL-MRV-ENE-001: Power Generation MRV Agent

Tests cover:
- Natural gas combined cycle emissions
- Coal power plant emissions
- CEMS data integration
- Emission intensity calculations
- Audit trail generation
"""

import pytest
from datetime import datetime, timezone

from greenlang.agents.mrv.energy import (
    PowerGenerationMRVAgent,
    PowerGenerationInput,
    calculate_power_generation_emissions,
)


class TestPowerGenerationMRVAgent:
    """Tests for PowerGenerationMRVAgent."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return PowerGenerationMRVAgent()

    @pytest.fixture
    def natural_gas_input(self):
        """Sample natural gas CCGT input."""
        return {
            "facility_id": "PLANT-001",
            "unit_id": "UNIT-1",
            "generation_type": "combined_cycle_gas_turbine",
            "fuel_type": "natural_gas",
            "fuel_consumption": 50000,
            "fuel_consumption_unit": "MMBTU",
            "net_generation_mwh": 5000,
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-01-31T23:59:59Z",
            "data_quality": "calculated",
        }

    @pytest.fixture
    def coal_input(self):
        """Sample coal power plant input."""
        return {
            "facility_id": "COAL-PLANT-001",
            "unit_id": "UNIT-1",
            "generation_type": "coal_subcritical",
            "fuel_type": "coal_bituminous",
            "fuel_consumption": 100000,
            "fuel_consumption_unit": "MMBTU",
            "net_generation_mwh": 8000,
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-01-31T23:59:59Z",
            "data_quality": "measured",
        }

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.agent_id == "GL-MRV-ENE-001"
        assert agent.version == "1.0.0"
        assert agent.enable_audit_trail is True

    def test_natural_gas_emissions_calculation(self, agent, natural_gas_input):
        """Test natural gas emissions calculation."""
        result = agent.process(natural_gas_input)

        # Verify structure
        assert "co2_tonnes" in result
        assert "ch4_tonnes_co2e" in result
        assert "n2o_tonnes_co2e" in result
        assert "total_ghg_tonnes_co2e" in result
        assert "emission_intensity_kg_mwh" in result
        assert "provenance_hash" in result

        # Natural gas: 53.06 kg CO2/MMBTU
        # 50,000 MMBTU * 53.06 / 1000 = 2653 tonnes CO2
        assert result["co2_tonnes"] == pytest.approx(2653, rel=0.01)

        # Intensity: 2653 * 1000 / 5000 = 530.6 kg/MWh
        assert result["emission_intensity_kg_mwh"] == pytest.approx(530.6, rel=0.05)

        # Validation should pass
        assert result["validation_status"] == "PASS"

    def test_coal_emissions_calculation(self, agent, coal_input):
        """Test coal emissions calculation."""
        result = agent.process(coal_input)

        # Coal bituminous: 93.28 kg CO2/MMBTU
        # 100,000 MMBTU * 93.28 / 1000 = 9328 tonnes CO2
        assert result["co2_tonnes"] == pytest.approx(9328, rel=0.01)

        # CH4 should be higher for coal
        assert result["ch4_tonnes_co2e"] > 0

        # Intensity should be higher than gas
        # 9328 * 1000 / 8000 = 1166 kg/MWh
        assert result["emission_intensity_kg_mwh"] > 1000

    def test_cems_data_integration(self, agent):
        """Test CEMS data takes precedence over calculation."""
        cems_input = {
            "facility_id": "PLANT-001",
            "unit_id": "UNIT-1",
            "generation_type": "combined_cycle_gas_turbine",
            "fuel_type": "natural_gas",
            "fuel_consumption": 50000,
            "fuel_consumption_unit": "MMBTU",
            "net_generation_mwh": 5000,
            "cems_co2_tons": 3000,  # US short tons
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-01-31T23:59:59Z",
        }

        result = agent.process(cems_input)

        # Should use CEMS value: 3000 US tons * 0.907185 = 2721.6 metric tonnes
        assert result["co2_tonnes"] == pytest.approx(2721.6, rel=0.01)
        assert "CEMS" in result["methodology"]

    def test_calculation_trace_completeness(self, agent, natural_gas_input):
        """Test calculation trace is complete."""
        result = agent.process(natural_gas_input)

        trace = result["calculation_trace"]
        assert len(trace) >= 5  # Should have multiple steps

        # Key steps should be documented
        trace_text = " ".join(trace)
        assert "fuel" in trace_text.lower()
        assert "co2" in trace_text.lower()

    def test_provenance_hash_determinism(self, agent, natural_gas_input):
        """Test provenance hash is deterministic."""
        result1 = agent.process(natural_gas_input)
        result2 = agent.process(natural_gas_input)

        # Same inputs should produce same hash
        assert result1["provenance_hash"] == result2["provenance_hash"]

    def test_audit_trail_capture(self, agent, natural_gas_input):
        """Test audit trail is captured."""
        agent.process(natural_gas_input)

        audit_trail = agent.get_audit_trail()
        assert len(audit_trail) > 0

        entry = audit_trail[-1]
        assert entry.agent_name == "PowerGenerationMRVAgent"
        assert entry.input_hash is not None
        assert entry.output_hash is not None

    def test_high_emission_intensity_warning(self, agent):
        """Test warning for high emission intensity."""
        high_intensity_input = {
            "facility_id": "PLANT-001",
            "unit_id": "UNIT-1",
            "generation_type": "coal_subcritical",
            "fuel_type": "coal_lignite",
            "fuel_consumption": 50000,
            "fuel_consumption_unit": "MMBTU",
            "net_generation_mwh": 1000,  # Very low generation
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-01-31T23:59:59Z",
        }

        result = agent.process(high_intensity_input)

        # High intensity should trigger warning
        assert result["emission_intensity_kg_mwh"] > 1000
        assert result["validation_status"] == "WARN"
        assert any("intensity" in w.lower() for w in result["warnings"])

    def test_convenience_function(self, natural_gas_input):
        """Test convenience function works."""
        result = calculate_power_generation_emissions(**natural_gas_input)

        assert "co2_tonnes" in result
        assert result["co2_tonnes"] > 0

    def test_fuel_unit_conversion(self, agent):
        """Test fuel unit conversions."""
        mcf_input = {
            "facility_id": "PLANT-001",
            "unit_id": "UNIT-1",
            "generation_type": "combined_cycle_gas_turbine",
            "fuel_type": "natural_gas",
            "fuel_consumption": 48600,  # MCF (approximately 50000 MMBTU)
            "fuel_consumption_unit": "MCF",
            "net_generation_mwh": 5000,
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-01-31T23:59:59Z",
        }

        result = agent.process(mcf_input)

        # Should be similar to MMBTU calculation
        # 48,600 MCF * 1.028 MMBTU/MCF = 49,961 MMBTU
        assert result["co2_tonnes"] == pytest.approx(2651, rel=0.02)

    def test_invalid_input_raises_error(self, agent):
        """Test invalid input raises ValueError."""
        invalid_input = {
            "facility_id": "PLANT-001",
            "unit_id": "UNIT-1",
            "generation_type": "invalid_type",  # Invalid
            "fuel_type": "natural_gas",
        }

        with pytest.raises(ValueError):
            agent.process(invalid_input)


class TestPowerGenerationInput:
    """Tests for input validation."""

    def test_valid_input(self):
        """Test valid input is accepted."""
        input_data = PowerGenerationInput(
            facility_id="PLANT-001",
            unit_id="UNIT-1",
            generation_type="combined_cycle_gas_turbine",
            fuel_type="natural_gas",
            fuel_consumption=50000,
            fuel_consumption_unit="MMBTU",
            net_generation_mwh=5000,
            reporting_period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            reporting_period_end=datetime(2024, 1, 31, 23, 59, 59, tzinfo=timezone.utc),
        )

        assert input_data.facility_id == "PLANT-001"
        assert input_data.net_generation_mwh == 5000

    def test_negative_generation_rejected(self):
        """Test negative generation is rejected."""
        with pytest.raises(ValueError):
            PowerGenerationInput(
                facility_id="PLANT-001",
                unit_id="UNIT-1",
                generation_type="combined_cycle_gas_turbine",
                net_generation_mwh=-100,  # Invalid
                reporting_period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                reporting_period_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            )

    def test_end_before_start_rejected(self):
        """Test end date before start is rejected."""
        with pytest.raises(ValueError):
            PowerGenerationInput(
                facility_id="PLANT-001",
                unit_id="UNIT-1",
                generation_type="combined_cycle_gas_turbine",
                net_generation_mwh=5000,
                reporting_period_start=datetime(2024, 1, 31, tzinfo=timezone.utc),
                reporting_period_end=datetime(2024, 1, 1, tzinfo=timezone.utc),  # Before start
            )
