# -*- coding: utf-8 -*-
"""
Tests for GL-MRV-IND-001: Steel Production MRV Agent
=====================================================

Author: GreenLang Framework Team
"""

import pytest
from decimal import Decimal

from greenlang.agents.mrv.industrial import (
    SteelProductionMRVAgent,
    SteelMRVInput,
    SteelMRVOutput,
    SteelProductionRoute,
    HydrogenSource,
)


class TestSteelProductionMRVAgent:
    """Test suite for Steel Production MRV Agent."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return SteelProductionMRVAgent()

    @pytest.fixture
    def bf_bof_input(self):
        """Create BF-BOF production input."""
        return SteelMRVInput(
            facility_id="STEEL_TEST_001",
            reporting_period="2024-Q1",
            production_tonnes=Decimal("10000"),
            production_route=SteelProductionRoute.BF_BOF,
            scrap_input_tonnes=Decimal("1000"),
        )

    @pytest.fixture
    def eaf_input(self):
        """Create EAF production input."""
        return SteelMRVInput(
            facility_id="STEEL_TEST_002",
            reporting_period="2024-Q1",
            production_tonnes=Decimal("10000"),
            production_route=SteelProductionRoute.EAF,
            scrap_input_tonnes=Decimal("9000"),
            electricity_kwh=Decimal("4000000"),
            grid_emission_factor_kg_co2_per_kwh=Decimal("0.4"),
        )

    @pytest.fixture
    def h2_dri_input(self):
        """Create H2-DRI production input."""
        return SteelMRVInput(
            facility_id="STEEL_TEST_003",
            reporting_period="2024-Q1",
            production_tonnes=Decimal("10000"),
            production_route=SteelProductionRoute.H2_DRI,
            hydrogen_source=HydrogenSource.GREEN,
            electricity_kwh=Decimal("5000000"),
            grid_emission_factor_kg_co2_per_kwh=Decimal("0.05"),
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-MRV-IND-001"
        assert agent.SECTOR == "Steel"
        assert agent.CBAM_CN_CODE == "7206-7229"
        assert len(agent._emission_factors) > 0

    def test_bf_bof_calculation(self, agent, bf_bof_input):
        """Test BF-BOF route emission calculation."""
        result = agent.process(bf_bof_input)

        assert isinstance(result, SteelMRVOutput)
        assert result.is_valid
        assert result.facility_id == "STEEL_TEST_001"
        assert result.production_route == "BF_BOF"

        # BF-BOF: 10000t * 1.85 tCO2/t = 18500 tCO2
        # Scrap credit: 1000t * -1.50 = -1500 tCO2
        # Total direct: 18500 - 1500 = 17000 tCO2
        assert result.scope_1_emissions_tco2e == Decimal("17000.000000")
        assert result.scrap_credit_tco2e == Decimal("-1500.000000")

    def test_eaf_calculation(self, agent, eaf_input):
        """Test EAF route emission calculation."""
        result = agent.process(eaf_input)

        assert result.is_valid
        assert result.production_route == "EAF"

        # EAF direct: 10000t * 0.40 = 4000 tCO2
        # Scrap credit: 9000t * -1.50 = -13500 tCO2
        # Electricity: 4000000 kWh * 0.4 kg/kWh / 1000 = 1600 tCO2
        assert result.scope_1_emissions_tco2e < Decimal("0")  # Net negative from scrap
        assert result.scope_2_emissions_tco2e == Decimal("1600.000000")

    def test_h2_dri_calculation(self, agent, h2_dri_input):
        """Test H2-DRI route with green hydrogen."""
        result = agent.process(h2_dri_input)

        assert result.is_valid
        assert result.production_route == "H2_DRI"

        # H2-DRI green: 10000t * 0.05 = 500 tCO2
        # Very low emissions due to green hydrogen
        assert result.scope_1_emissions_tco2e < Decimal("1000")

    def test_emission_intensity(self, agent, bf_bof_input):
        """Test emission intensity calculation."""
        result = agent.process(bf_bof_input)

        # Intensity = total / production
        expected_intensity = result.total_emissions_tco2e / bf_bof_input.production_tonnes
        assert abs(result.emission_intensity_tco2e_per_t - expected_intensity) < Decimal("0.001")

    def test_cbam_output(self, agent, bf_bof_input):
        """Test CBAM-compliant output."""
        result = agent.process(bf_bof_input)

        assert result.cbam_output is not None
        assert result.cbam_output.cn_code == "7206-7229"
        assert result.cbam_output.product_category == "Iron and Steel"
        assert result.cbam_output.quantity_tonnes == bf_bof_input.production_tonnes

    def test_provenance_tracking(self, agent, bf_bof_input):
        """Test SHA-256 provenance hashes."""
        result = agent.process(bf_bof_input)

        assert len(result.input_hash) == 64  # SHA-256
        assert len(result.output_hash) == 64
        assert len(result.provenance_hash) == 64

    def test_calculation_steps_audit_trail(self, agent, bf_bof_input):
        """Test calculation steps are recorded for audit."""
        result = agent.process(bf_bof_input)

        assert len(result.calculation_steps) >= 4
        assert all(step.step_number > 0 for step in result.calculation_steps)
        assert all(step.output_unit for step in result.calculation_steps)

    def test_emission_factors_recorded(self, agent, bf_bof_input):
        """Test emission factors used are recorded."""
        result = agent.process(bf_bof_input)

        assert len(result.emission_factors_used) >= 1
        # Should include BF-BOF and scrap credit factors
        factor_ids = [ef.factor_id for ef in result.emission_factors_used]
        assert any("bf_bof" in fid or "BF_BOF" in fid for fid in factor_ids)

    def test_deterministic_output(self, agent, bf_bof_input):
        """Test same input produces same output (zero-hallucination)."""
        result1 = agent.process(bf_bof_input)
        result2 = agent.process(bf_bof_input)

        assert result1.total_emissions_tco2e == result2.total_emissions_tco2e
        assert result1.emission_intensity_tco2e_per_t == result2.emission_intensity_tco2e_per_t

    def test_hydrogen_source_required_for_h2dri(self):
        """Test H2-DRI route requires hydrogen source."""
        with pytest.raises(ValueError, match="hydrogen_source"):
            SteelMRVInput(
                facility_id="TEST",
                reporting_period="2024-Q1",
                production_tonnes=Decimal("1000"),
                production_route=SteelProductionRoute.H2_DRI,
                # Missing hydrogen_source
            )


class TestSteelEmissionFactors:
    """Test emission factor values."""

    def test_bf_bof_factor(self):
        """Test BF-BOF emission factor."""
        agent = SteelProductionMRVAgent()
        assert agent.EF_BF_BOF == Decimal("1.85")

    def test_eaf_factor(self):
        """Test EAF emission factor."""
        agent = SteelProductionMRVAgent()
        assert agent.EF_EAF == Decimal("0.40")

    def test_scrap_credit(self):
        """Test scrap credit is negative."""
        agent = SteelProductionMRVAgent()
        assert agent.SCRAP_CREDIT < Decimal("0")
