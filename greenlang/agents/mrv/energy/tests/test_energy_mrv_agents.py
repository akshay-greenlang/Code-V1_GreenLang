# -*- coding: utf-8 -*-
"""
Integration Tests for MRV Energy Sector Agents

Tests cover all 8 MRV Energy agents with realistic scenarios.
"""

import pytest
from datetime import datetime, timezone

from greenlang.agents.mrv.energy import (
    # Agents
    PowerGenerationMRVAgent,
    GridEmissionsTrackerAgent,
    RenewableGenerationMRVAgent,
    StorageSystemsMRVAgent,
    TransmissionLossMRVAgent,
    FuelSupplyChainMRVAgent,
    CHPSystemsMRVAgent,
    HydrogenProductionMRVAgent,
    # Registry
    get_agent,
    AGENT_REGISTRY,
)


class TestGridEmissionsTrackerAgent:
    """Tests for GL-MRV-ENE-002."""

    @pytest.fixture
    def agent(self):
        return GridEmissionsTrackerAgent()

    def test_location_based_calculation(self, agent):
        """Test location-based Scope 2 calculation."""
        result = agent.process({
            "facility_id": "OFFICE-001",
            "grid_region": "us_camx",
            "electricity_consumption_mwh": 1000,
            "accounting_method": "location_based",
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-12-31T23:59:59Z",
        })

        # CAMX factor is 234 kg/MWh
        # 1000 MWh * 234 / 1000 = 234 tonnes
        assert result["location_based_co2e_tonnes"] == pytest.approx(234, rel=0.01)
        assert result["scope_2_location"] == pytest.approx(234, rel=0.01)

    def test_market_based_with_recs(self, agent):
        """Test market-based with REC application."""
        result = agent.process({
            "facility_id": "OFFICE-001",
            "grid_region": "us_camx",
            "electricity_consumption_mwh": 1000,
            "accounting_method": "dual",
            "renewable_certificates_mwh": 500,  # 50% covered by RECs
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-12-31T23:59:59Z",
        })

        # RECs should reduce market-based emissions
        assert result["rec_reduction_tonnes"] > 0
        assert result["market_based_co2e_tonnes"] < result["location_based_co2e_tonnes"]


class TestRenewableGenerationMRVAgent:
    """Tests for GL-MRV-ENE-003."""

    @pytest.fixture
    def agent(self):
        return RenewableGenerationMRVAgent()

    def test_solar_pv_avoided_emissions(self, agent):
        """Test solar PV avoided emissions calculation."""
        result = agent.process({
            "facility_id": "SOLAR-FARM-001",
            "asset_id": "PV-ARRAY-1",
            "technology": "solar_pv_utility",
            "installed_capacity_mw": 50,
            "net_generation_mwh": 8500,
            "curtailment_mwh": 500,
            "auxiliary_consumption_mwh": 200,
            "grid_region": "us_camx",
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-12-31T23:59:59Z",
        })

        # Capacity factor should be reasonable for solar
        assert 0.15 <= result["capacity_factor"] <= 0.35

        # Avoided emissions based on grid factor
        # 8500 MWh * 234 kg/MWh / 1000 = 1989 tonnes
        assert result["avoided_co2e_tonnes"] == pytest.approx(1989, rel=0.01)

        # Lifecycle emissions much lower than avoided
        assert result["lifecycle_co2e_tonnes"] < result["avoided_co2e_tonnes"]

        # Should be REC eligible
        assert result["rec_eligible_mwh"] == 8500


class TestStorageSystemsMRVAgent:
    """Tests for GL-MRV-ENE-004."""

    @pytest.fixture
    def agent(self):
        return StorageSystemsMRVAgent()

    def test_battery_storage_emissions(self, agent):
        """Test battery storage emissions calculation."""
        result = agent.process({
            "facility_id": "BESS-001",
            "storage_id": "LI-ION-1",
            "technology": "li_ion_nmc",
            "rated_capacity_mwh": 100,
            "rated_power_mw": 25,
            "energy_charged_mwh": 2500,
            "energy_discharged_mwh": 2200,
            "round_trip_efficiency": 0.88,
            "grid_region": "us_camx",
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-12-31T23:59:59Z",
        })

        # Losses should match efficiency
        expected_losses = 2500 - 2200
        assert result["storage_losses_mwh"] == expected_losses

        # Should have charging emissions
        assert result["charging_emissions_tonnes"] > 0

        # Cycles calculation
        assert result["total_cycles"] == 22  # 2200 / 100


class TestTransmissionLossMRVAgent:
    """Tests for GL-MRV-ENE-005."""

    @pytest.fixture
    def agent(self):
        return TransmissionLossMRVAgent()

    def test_td_loss_calculation(self, agent):
        """Test T&D loss calculation."""
        result = agent.process({
            "facility_id": "UTILITY-001",
            "network_id": "GRID-WEST",
            "voltage_level": "transmission",
            "energy_injected_mwh": 1000000,
            "energy_delivered_mwh": 950000,
            "grid_region": "us_camx",
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-12-31T23:59:59Z",
        })

        # 5% losses
        assert result["loss_percentage"] == pytest.approx(5.0, rel=0.01)
        assert result["total_losses_mwh"] == 50000

        # Loss emissions
        # 50000 * 234 / 1000 = 11700 tonnes
        assert result["loss_emissions_tonnes"] == pytest.approx(11700, rel=0.01)

        # Loss-adjusted factor should be higher
        assert result["loss_adjusted_factor_kg_mwh"] > result["emission_factor_kg_mwh"]


class TestFuelSupplyChainMRVAgent:
    """Tests for GL-MRV-ENE-006."""

    @pytest.fixture
    def agent(self):
        return FuelSupplyChainMRVAgent()

    def test_natural_gas_upstream(self, agent):
        """Test natural gas upstream emissions."""
        result = agent.process({
            "facility_id": "PLANT-001",
            "fuel_type": "natural_gas",
            "fuel_quantity": 100000,
            "fuel_unit": "MMBTU",
            "origin_country": "US",
            "transport_mode": "pipeline",
            "transport_distance_km": 500,
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-12-31T23:59:59Z",
        })

        # Should have all components
        assert result["extraction_emissions_tonnes"] > 0
        assert result["processing_emissions_tonnes"] > 0
        assert result["fugitive_emissions_tonnes"] > 0  # Methane leakage
        assert result["transport_emissions_tonnes"] > 0

        # Total should be sum
        total = (
            result["extraction_emissions_tonnes"] +
            result["processing_emissions_tonnes"] +
            result["transport_emissions_tonnes"] +
            result["fugitive_emissions_tonnes"]
        )
        assert result["total_upstream_emissions_tonnes"] == pytest.approx(total, rel=0.01)


class TestCHPSystemsMRVAgent:
    """Tests for GL-MRV-ENE-007."""

    @pytest.fixture
    def agent(self):
        return CHPSystemsMRVAgent()

    def test_chp_allocation(self, agent):
        """Test CHP emission allocation."""
        result = agent.process({
            "facility_id": "CHP-PLANT-001",
            "chp_id": "COGEN-1",
            "fuel_type": "natural_gas",
            "fuel_consumption": 50000,
            "fuel_unit": "MMBTU",
            "electricity_output_mwh": 4000,
            "heat_output_mmbtu": 25000,
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-12-31T23:59:59Z",
        })

        # Overall efficiency should be high for CHP
        assert result["overall_efficiency"] > 0.70

        # Allocations should sum to total
        total_allocated = (
            result["electricity_allocation_tonnes"] +
            result["heat_allocation_tonnes"]
        )
        assert total_allocated == pytest.approx(
            result["total_emissions_tonnes"], rel=0.01
        )

        # PES should be positive for efficient CHP
        assert result["primary_energy_savings_pct"] > 0


class TestHydrogenProductionMRVAgent:
    """Tests for GL-MRV-ENE-008."""

    @pytest.fixture
    def agent(self):
        return HydrogenProductionMRVAgent()

    def test_green_hydrogen_electrolysis(self, agent):
        """Test green hydrogen production."""
        result = agent.process({
            "facility_id": "H2-PLANT-001",
            "production_id": "ELECTROLYZER-1",
            "production_method": "electrolysis_renewable",
            "hydrogen_output_kg": 10000,
            "feedstock_consumption": 0,
            "feedstock_unit": "MMBTU",
            "electricity_consumption_kwh": 550000,
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-12-31T23:59:59Z",
        })

        # Green hydrogen should have low carbon intensity
        assert result["carbon_intensity_kg_co2_kg_h2"] < 1.0
        assert result["hydrogen_color"] == "green"
        assert result["low_carbon_eligible"] is True

    def test_grey_hydrogen_smr(self, agent):
        """Test grey hydrogen from SMR."""
        result = agent.process({
            "facility_id": "H2-PLANT-002",
            "production_id": "SMR-1",
            "production_method": "steam_methane_reforming",
            "hydrogen_output_kg": 10000,
            "feedstock_consumption": 500,
            "feedstock_unit": "MMBTU",
            "electricity_consumption_kwh": 10000,
            "grid_region": "us_camx",
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-12-31T23:59:59Z",
        })

        # Grey hydrogen should have high carbon intensity
        assert result["carbon_intensity_kg_co2_kg_h2"] > 8.0
        assert result["hydrogen_color"] == "grey"
        assert result["low_carbon_eligible"] is False

    def test_blue_hydrogen_with_ccs(self, agent):
        """Test blue hydrogen with CCS."""
        result = agent.process({
            "facility_id": "H2-PLANT-003",
            "production_id": "SMR-CCS-1",
            "production_method": "steam_methane_reforming_with_ccs",
            "hydrogen_output_kg": 10000,
            "feedstock_consumption": 500,
            "feedstock_unit": "MMBTU",
            "electricity_consumption_kwh": 35000,  # CCS energy penalty
            "grid_region": "us_camx",
            "ccs_capture_rate": 0.90,
            "reporting_period_start": "2024-01-01T00:00:00Z",
            "reporting_period_end": "2024-12-31T23:59:59Z",
        })

        # Blue hydrogen should be intermediate
        assert 1.0 < result["carbon_intensity_kg_co2_kg_h2"] < 5.0
        assert result["hydrogen_color"] == "blue"
        assert result["captured_emissions_tonnes"] > 0


class TestAgentRegistry:
    """Tests for agent registry."""

    def test_all_agents_registered(self):
        """Test all 8 agents are in registry."""
        expected_agents = [
            "GL-MRV-ENE-001",
            "GL-MRV-ENE-002",
            "GL-MRV-ENE-003",
            "GL-MRV-ENE-004",
            "GL-MRV-ENE-005",
            "GL-MRV-ENE-006",
            "GL-MRV-ENE-007",
            "GL-MRV-ENE-008",
        ]

        for agent_id in expected_agents:
            assert agent_id in AGENT_REGISTRY

    def test_get_agent_function(self):
        """Test get_agent retrieves correct agent."""
        agent_class = get_agent("GL-MRV-ENE-001")
        assert agent_class == PowerGenerationMRVAgent

    def test_get_agent_invalid_id(self):
        """Test get_agent raises for invalid ID."""
        with pytest.raises(KeyError):
            get_agent("GL-MRV-ENE-999")

    def test_all_agents_instantiable(self):
        """Test all agents can be instantiated."""
        for agent_id, agent_class in AGENT_REGISTRY.items():
            agent = agent_class()
            assert agent.agent_id == agent_id
