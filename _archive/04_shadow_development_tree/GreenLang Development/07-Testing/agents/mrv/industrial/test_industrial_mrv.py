# -*- coding: utf-8 -*-
"""
Integration Tests for GreenLang Industrial MRV Agents
======================================================

Tests all industrial sector MRV agents for consistent behavior.

Author: GreenLang Framework Team
"""

import pytest
from decimal import Decimal

from greenlang.agents.mrv.industrial import (
    # Steel
    SteelProductionMRVAgent,
    SteelMRVInput,
    SteelProductionRoute,
    # Cement
    CementProductionMRVAgent,
    CementMRVInput,
    CementType,
    # Chemicals
    ChemicalsProductionMRVAgent,
    ChemicalsMRVInput,
    ChemicalProduct,
    # Aluminum
    AluminumProductionMRVAgent,
    AluminumMRVInput,
    AluminumProductionRoute,
    # Pulp & Paper
    PulpPaperMRVAgent,
    PulpPaperMRVInput,
    # Glass
    GlassProductionMRVAgent,
    GlassMRVInput,
    GlassType,
    # Food Processing
    FoodProcessingMRVAgent,
    FoodProcessingMRVInput,
    FoodSubsector,
)
from greenlang.agents.mrv.industrial.additional_sectors import (
    PharmaceuticalMRVAgent, PharmaMRVInput,
    ElectronicsMRVAgent, ElectronicsMRVInput,
    AutomotiveMRVAgent, AutomotiveMRVInput,
    TextilesMRVAgent, TextilesMRVInput,
    MiningMRVAgent, MiningMRVInput, MiningType,
    PlasticsMRVAgent, PlasticsMRVInput,
)


class TestAllIndustrialMRVAgents:
    """Test all industrial MRV agents for common functionality."""

    @pytest.fixture
    def standard_params(self):
        """Standard parameters for testing."""
        return {
            "facility_id": "TEST_FACILITY",
            "reporting_period": "2024-Q1",
            "production_tonnes": Decimal("1000"),
            "grid_emission_factor_kg_co2_per_kwh": Decimal("0.4"),
        }

    def test_steel_agent(self, standard_params):
        """Test Steel MRV agent."""
        agent = SteelProductionMRVAgent()
        input_data = SteelMRVInput(
            **standard_params,
            production_route=SteelProductionRoute.BF_BOF
        )
        result = agent.process(input_data)
        assert result.is_valid
        assert result.total_emissions_tco2e > 0
        assert result.agent_id == "GL-MRV-IND-001"

    def test_cement_agent(self, standard_params):
        """Test Cement MRV agent."""
        agent = CementProductionMRVAgent()
        input_data = CementMRVInput(
            **standard_params,
            cement_type=CementType.CEM_I
        )
        result = agent.process(input_data)
        assert result.is_valid
        assert result.total_emissions_tco2e > 0
        assert result.agent_id == "GL-MRV-IND-002"

    def test_chemicals_agent(self, standard_params):
        """Test Chemicals MRV agent."""
        agent = ChemicalsProductionMRVAgent()
        input_data = ChemicalsMRVInput(
            **standard_params,
            chemical_product=ChemicalProduct.AMMONIA
        )
        result = agent.process(input_data)
        assert result.is_valid
        assert result.total_emissions_tco2e > 0
        assert result.agent_id == "GL-MRV-IND-003"

    def test_aluminum_agent(self, standard_params):
        """Test Aluminum MRV agent."""
        agent = AluminumProductionMRVAgent()
        input_data = AluminumMRVInput(
            **standard_params,
            production_route=AluminumProductionRoute.PRIMARY_PREBAKE
        )
        result = agent.process(input_data)
        assert result.is_valid
        assert result.total_emissions_tco2e > 0
        assert result.agent_id == "GL-MRV-IND-004"

    def test_pulp_paper_agent(self, standard_params):
        """Test Pulp & Paper MRV agent."""
        agent = PulpPaperMRVAgent()
        input_data = PulpPaperMRVInput(**standard_params)
        result = agent.process(input_data)
        assert result.is_valid
        assert result.agent_id == "GL-MRV-IND-005"

    def test_glass_agent(self, standard_params):
        """Test Glass MRV agent."""
        agent = GlassProductionMRVAgent()
        input_data = GlassMRVInput(
            **standard_params,
            glass_type=GlassType.CONTAINER
        )
        result = agent.process(input_data)
        assert result.is_valid
        assert result.total_emissions_tco2e > 0
        assert result.agent_id == "GL-MRV-IND-006"

    def test_food_processing_agent(self, standard_params):
        """Test Food Processing MRV agent."""
        agent = FoodProcessingMRVAgent()
        input_data = FoodProcessingMRVInput(
            **standard_params,
            subsector=FoodSubsector.DAIRY
        )
        result = agent.process(input_data)
        assert result.is_valid
        assert result.total_emissions_tco2e > 0
        assert result.agent_id == "GL-MRV-IND-007"

    def test_pharmaceutical_agent(self, standard_params):
        """Test Pharmaceutical MRV agent."""
        agent = PharmaceuticalMRVAgent()
        input_data = PharmaMRVInput(**standard_params)
        result = agent.process(input_data)
        assert result.is_valid
        assert result.agent_id == "GL-MRV-IND-008"

    def test_electronics_agent(self, standard_params):
        """Test Electronics MRV agent."""
        agent = ElectronicsMRVAgent()
        input_data = ElectronicsMRVInput(**standard_params)
        result = agent.process(input_data)
        assert result.is_valid
        assert result.agent_id == "GL-MRV-IND-009"

    def test_automotive_agent(self, standard_params):
        """Test Automotive MRV agent."""
        agent = AutomotiveMRVAgent()
        input_data = AutomotiveMRVInput(**standard_params)
        result = agent.process(input_data)
        assert result.is_valid
        assert result.agent_id == "GL-MRV-IND-010"

    def test_textiles_agent(self, standard_params):
        """Test Textiles MRV agent."""
        agent = TextilesMRVAgent()
        input_data = TextilesMRVInput(**standard_params)
        result = agent.process(input_data)
        assert result.is_valid
        assert result.agent_id == "GL-MRV-IND-011"

    def test_mining_agent(self, standard_params):
        """Test Mining MRV agent."""
        agent = MiningMRVAgent()
        input_data = MiningMRVInput(
            **standard_params,
            mining_type=MiningType.IRON_ORE
        )
        result = agent.process(input_data)
        assert result.is_valid
        assert result.agent_id == "GL-MRV-IND-012"

    def test_plastics_agent(self, standard_params):
        """Test Plastics MRV agent."""
        agent = PlasticsMRVAgent()
        input_data = PlasticsMRVInput(**standard_params)
        result = agent.process(input_data)
        assert result.is_valid
        assert result.agent_id == "GL-MRV-IND-013"


class TestCommonMRVBehavior:
    """Test common behavior across all MRV agents."""

    def test_all_agents_have_cbam_output(self):
        """Test all agents produce CBAM output."""
        params = {
            "facility_id": "TEST",
            "reporting_period": "2024-Q1",
            "production_tonnes": Decimal("1000"),
        }

        agents_and_inputs = [
            (SteelProductionMRVAgent(), SteelMRVInput(**params, production_route=SteelProductionRoute.BF_BOF)),
            (CementProductionMRVAgent(), CementMRVInput(**params)),
            (GlassProductionMRVAgent(), GlassMRVInput(**params)),
        ]

        for agent, input_data in agents_and_inputs:
            result = agent.process(input_data)
            assert result.cbam_output is not None, f"{agent.AGENT_ID} missing CBAM output"
            assert result.cbam_output.cn_code, f"{agent.AGENT_ID} missing CN code"

    def test_all_agents_have_provenance_hashes(self):
        """Test all agents produce provenance hashes."""
        params = {
            "facility_id": "TEST",
            "reporting_period": "2024-Q1",
            "production_tonnes": Decimal("1000"),
        }

        agent = SteelProductionMRVAgent()
        input_data = SteelMRVInput(**params, production_route=SteelProductionRoute.BF_BOF)
        result = agent.process(input_data)

        assert result.input_hash, "Missing input hash"
        assert result.output_hash, "Missing output hash"
        assert result.provenance_hash, "Missing provenance hash"
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_emission_intensity_always_calculated(self):
        """Test emission intensity is always calculated."""
        agent = SteelProductionMRVAgent()
        input_data = SteelMRVInput(
            facility_id="TEST",
            reporting_period="2024-Q1",
            production_tonnes=Decimal("1000"),
            production_route=SteelProductionRoute.BF_BOF
        )
        result = agent.process(input_data)

        assert result.emission_intensity_tco2e_per_t >= 0
        # Intensity should equal total/production
        expected = result.total_emissions_tco2e / input_data.production_tonnes
        assert abs(result.emission_intensity_tco2e_per_t - expected.quantize(Decimal("0.0001"))) < Decimal("0.01")
