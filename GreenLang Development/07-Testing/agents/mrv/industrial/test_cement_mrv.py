# -*- coding: utf-8 -*-
"""
Tests for GL-MRV-IND-002: Cement Production MRV Agent
======================================================

Author: GreenLang Framework Team
"""

import pytest
from decimal import Decimal

from greenlang.agents.mrv.industrial import (
    CementProductionMRVAgent,
    CementMRVInput,
    CementMRVOutput,
    CementType,
    KilnFuelType,
    SCMType,
)
from greenlang.agents.mrv.industrial.cement_mrv import SCMInput


class TestCementProductionMRVAgent:
    """Test suite for Cement Production MRV Agent."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return CementProductionMRVAgent()

    @pytest.fixture
    def cem_i_input(self):
        """Create CEM I (high clinker) input."""
        return CementMRVInput(
            facility_id="CEMENT_TEST_001",
            reporting_period="2024-Q1",
            production_tonnes=Decimal("10000"),
            cement_type=CementType.CEM_I,
            kiln_fuel_type=KilnFuelType.COAL,
        )

    @pytest.fixture
    def cem_iii_input(self):
        """Create CEM III (low clinker) input with SCMs."""
        return CementMRVInput(
            facility_id="CEMENT_TEST_002",
            reporting_period="2024-Q1",
            production_tonnes=Decimal("10000"),
            cement_type=CementType.CEM_III_B,
            kiln_fuel_type=KilnFuelType.NATURAL_GAS,
            scm_inputs=[
                SCMInput(scm_type=SCMType.GGBS, quantity_tonnes=Decimal("7500"))
            ],
            grid_emission_factor_kg_co2_per_kwh=Decimal("0.4"),
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-MRV-IND-002"
        assert agent.SECTOR == "Cement"
        assert agent.CBAM_CN_CODE == "2523"

    def test_cem_i_calculation(self, agent, cem_i_input):
        """Test CEM I emission calculation."""
        result = agent.process(cem_i_input)

        assert isinstance(result, CementMRVOutput)
        assert result.is_valid
        assert result.cement_type == "CEM_I"

        # CEM I: 95% clinker ratio
        assert result.clinker_ratio == Decimal("0.95")
        # Clinker: 10000 * 0.95 = 9500 tonnes
        assert result.clinker_production_tonnes == Decimal("9500.000")

        # Calcination: 9500 * 0.525 = 4987.5 tCO2
        assert result.calcination_emissions_tco2e > Decimal("4900")

    def test_clinker_ratio_by_cement_type(self, agent):
        """Test clinker ratios vary by cement type."""
        cem_i = CementMRVInput(
            facility_id="TEST", reporting_period="2024-Q1",
            production_tonnes=Decimal("1000"), cement_type=CementType.CEM_I
        )
        cem_iii_b = CementMRVInput(
            facility_id="TEST", reporting_period="2024-Q1",
            production_tonnes=Decimal("1000"), cement_type=CementType.CEM_III_B
        )

        result_i = agent.process(cem_i)
        result_iii = agent.process(cem_iii_b)

        # CEM I has much higher clinker ratio than CEM III/B
        assert result_i.clinker_ratio > result_iii.clinker_ratio
        assert result_i.clinker_ratio == Decimal("0.95")
        assert result_iii.clinker_ratio == Decimal("0.25")

    def test_scm_credits(self, agent, cem_iii_input):
        """Test SCM credits reduce emissions."""
        result = agent.process(cem_iii_input)

        # SCM credits should be negative
        assert result.scm_credits_tco2e < Decimal("0")
        # 7500 tonnes GGBS * 0.070 factor = 525 tCO2 credit
        assert result.scm_credits_tco2e == Decimal("-525.000")

    def test_kiln_fuel_impact(self, agent):
        """Test different kiln fuels have different emissions."""
        coal_input = CementMRVInput(
            facility_id="TEST", reporting_period="2024-Q1",
            production_tonnes=Decimal("1000"), cement_type=CementType.CEM_I,
            kiln_fuel_type=KilnFuelType.COAL
        )
        gas_input = CementMRVInput(
            facility_id="TEST", reporting_period="2024-Q1",
            production_tonnes=Decimal("1000"), cement_type=CementType.CEM_I,
            kiln_fuel_type=KilnFuelType.NATURAL_GAS
        )

        result_coal = agent.process(coal_input)
        result_gas = agent.process(gas_input)

        # Coal has higher kiln fuel emissions than natural gas
        assert result_coal.kiln_fuel_emissions_tco2e > result_gas.kiln_fuel_emissions_tco2e

    def test_emission_intensity_comparison(self, agent, cem_i_input, cem_iii_input):
        """Test CEM III has lower intensity than CEM I."""
        result_i = agent.process(cem_i_input)
        result_iii = agent.process(cem_iii_input)

        # CEM III should have lower emission intensity
        assert result_iii.emission_intensity_tco2e_per_t < result_i.emission_intensity_tco2e_per_t

    def test_cbam_output(self, agent, cem_i_input):
        """Test CBAM-compliant output."""
        result = agent.process(cem_i_input)

        assert result.cbam_output is not None
        assert result.cbam_output.cn_code == "2523"
        assert result.cbam_output.product_category == "Cement"

    def test_calculation_audit_trail(self, agent, cem_i_input):
        """Test calculation steps are tracked."""
        result = agent.process(cem_i_input)

        assert len(result.calculation_steps) >= 5
        step_descriptions = [s.description for s in result.calculation_steps]

        # Should include key calculation steps
        assert any("clinker" in d.lower() for d in step_descriptions)
        assert any("calcination" in d.lower() for d in step_descriptions)

    def test_deterministic_results(self, agent, cem_i_input):
        """Test same inputs produce same outputs."""
        result1 = agent.process(cem_i_input)
        result2 = agent.process(cem_i_input)

        assert result1.total_emissions_tco2e == result2.total_emissions_tco2e
        assert result1.calcination_emissions_tco2e == result2.calcination_emissions_tco2e


class TestCementEmissionFactors:
    """Test cement emission factor values."""

    def test_calcination_factor(self):
        """Test clinker calcination factor."""
        agent = CementProductionMRVAgent()
        # IPCC value: 0.525 tCO2/t clinker
        assert agent.EF_CLINKER_CALCINATION == Decimal("0.525")

    def test_kiln_fuel_factors(self):
        """Test kiln fuel emission factors."""
        agent = CementProductionMRVAgent()
        # Coal > Petcoke > Natural Gas > Alternative
        assert agent.EF_KILN_COAL > agent.EF_KILN_NATURAL_GAS
        assert agent.EF_KILN_NATURAL_GAS > agent.EF_KILN_ALTERNATIVE
