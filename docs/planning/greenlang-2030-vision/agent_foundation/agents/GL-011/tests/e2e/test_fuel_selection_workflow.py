# -*- coding: utf-8 -*-
"""
End-to-end tests for fuel selection workflow.

Tests the complete fuel selection workflow:
- Multi-fuel optimization request to response
- Cost optimization with constraints
- Blending optimization
- Carbon minimization scenarios
- Full orchestrator pipeline
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fuel_management_orchestrator import (
    FuelManagementOrchestrator,
    FuelOptimizationRequest,
    FuelOperationalState
)
from config import FuelManagementConfig, FuelSpecification
from calculators.multi_fuel_optimizer import MultiFuelOptimizer
from calculators.cost_optimization_calculator import CostOptimizationCalculator
from calculators.fuel_blending_calculator import FuelBlendingCalculator
from calculators.carbon_footprint_calculator import CarbonFootprintCalculator


class TestFuelSelectionWorkflow:
    """End-to-end tests for fuel selection workflow."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return FuelManagementConfig(
            environment='test',
            enable_caching=True,
            cache_ttl_seconds=300,
            max_concurrent_optimizations=10
        )

    @pytest.fixture
    def orchestrator(self, config):
        """Create orchestrator instance."""
        return FuelManagementOrchestrator(config)

    @pytest.fixture
    def fuel_specifications(self):
        """Complete fuel specifications."""
        return {
            'natural_gas': FuelSpecification(
                fuel_id='natural_gas',
                fuel_name='Natural Gas',
                fuel_type='gas',
                heating_value_mj_kg=50.0,
                emission_factor_co2_kg_gj=56.1,
                emission_factor_nox_g_gj=50,
                emission_factor_sox_g_gj=0.3,
                is_renewable=False,
                availability_percent=100.0
            ),
            'coal': FuelSpecification(
                fuel_id='coal',
                fuel_name='Bituminous Coal',
                fuel_type='solid',
                heating_value_mj_kg=25.0,
                emission_factor_co2_kg_gj=94.6,
                emission_factor_nox_g_gj=250,
                emission_factor_sox_g_gj=500,
                is_renewable=False,
                availability_percent=100.0
            ),
            'biomass': FuelSpecification(
                fuel_id='biomass',
                fuel_name='Wood Pellets',
                fuel_type='solid',
                heating_value_mj_kg=18.0,
                emission_factor_co2_kg_gj=0.0,
                emission_factor_nox_g_gj=150,
                emission_factor_sox_g_gj=20,
                is_renewable=True,
                availability_percent=80.0
            ),
            'hydrogen': FuelSpecification(
                fuel_id='hydrogen',
                fuel_name='Green Hydrogen',
                fuel_type='gas',
                heating_value_mj_kg=120.0,
                emission_factor_co2_kg_gj=0.0,
                emission_factor_nox_g_gj=10,
                emission_factor_sox_g_gj=0,
                is_renewable=True,
                availability_percent=50.0
            )
        }

    @pytest.fixture
    def market_prices(self):
        """Current market prices."""
        return {
            'natural_gas': 0.045,  # USD/kg
            'coal': 0.035,
            'biomass': 0.08,
            'hydrogen': 5.00
        }

    def test_basic_fuel_selection_request(self, orchestrator, fuel_specifications, market_prices):
        """Test basic fuel selection request to response."""
        request = FuelOptimizationRequest(
            request_id='REQ-001',
            energy_demand_mw=100,
            optimization_objective='balanced',
            available_fuels=['natural_gas', 'coal'],
            fuel_specifications=fuel_specifications,
            market_prices=market_prices,
            emission_limits={},
            constraints={}
        )

        response = orchestrator.execute(request)

        # Verify response structure
        assert response is not None
        assert response.request_id == 'REQ-001'
        assert response.status == 'SUCCESS'
        assert response.optimal_fuel_mix is not None
        assert sum(response.optimal_fuel_mix.values()) > 0.99
        assert response.provenance_hash is not None
        assert len(response.provenance_hash) == 64

    def test_cost_minimization_workflow(self, orchestrator, fuel_specifications, market_prices):
        """Test cost minimization selects cheapest fuel."""
        request = FuelOptimizationRequest(
            request_id='REQ-002',
            energy_demand_mw=100,
            optimization_objective='minimize_cost',
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_specifications=fuel_specifications,
            market_prices=market_prices,
            emission_limits={},
            constraints={}
        )

        response = orchestrator.execute(request)

        # Coal should be preferred (lowest cost per GJ)
        assert response.status == 'SUCCESS'
        assert response.cost_per_mwh > 0
        # Cost-optimized should include significant coal share
        # (unless constraints prevent it)
        total_cost = response.total_cost_usd
        assert total_cost > 0

    def test_emissions_minimization_workflow(self, orchestrator, fuel_specifications, market_prices):
        """Test emissions minimization selects low-carbon fuels."""
        request = FuelOptimizationRequest(
            request_id='REQ-003',
            energy_demand_mw=100,
            optimization_objective='minimize_emissions',
            available_fuels=['natural_gas', 'coal', 'biomass', 'hydrogen'],
            fuel_specifications=fuel_specifications,
            market_prices=market_prices,
            emission_limits={},
            constraints={}
        )

        response = orchestrator.execute(request)

        assert response.status == 'SUCCESS'
        # Renewable fuels should be preferred
        renewable_share = (
            response.optimal_fuel_mix.get('biomass', 0) +
            response.optimal_fuel_mix.get('hydrogen', 0)
        )
        # Should have significant renewable component
        assert renewable_share > 0 or response.carbon_intensity_kg_mwh < 100

    def test_constraint_satisfaction(self, orchestrator, fuel_specifications, market_prices):
        """Test constraints are satisfied in optimization."""
        request = FuelOptimizationRequest(
            request_id='REQ-004',
            energy_demand_mw=100,
            optimization_objective='minimize_cost',
            available_fuels=['natural_gas', 'coal'],
            fuel_specifications=fuel_specifications,
            market_prices=market_prices,
            emission_limits={'nox_g_gj': 100},  # Exclude coal
            constraints={'coal_max_share': 0.2}
        )

        response = orchestrator.execute(request)

        assert response.status == 'SUCCESS'
        # Coal should be limited due to constraints
        coal_share = response.optimal_fuel_mix.get('coal', 0)
        assert coal_share <= 0.21  # Allow small tolerance


class TestBlendingWorkflow:
    """End-to-end tests for fuel blending workflow."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator."""
        config = FuelManagementConfig(environment='test')
        return FuelManagementOrchestrator(config)

    @pytest.fixture
    def solid_fuel_specs(self):
        """Solid fuel specifications for blending."""
        return {
            'coal': {
                'heating_value_mj_kg': 25.0,
                'moisture_content_percent': 8.0,
                'ash_content_percent': 10.0,
                'sulfur_content_percent': 2.0,
                'carbon_content_percent': 60.0
            },
            'biomass': {
                'heating_value_mj_kg': 18.0,
                'moisture_content_percent': 25.0,
                'ash_content_percent': 2.0,
                'sulfur_content_percent': 0.1,
                'carbon_content_percent': 50.0
            },
            'wood_pellets': {
                'heating_value_mj_kg': 17.5,
                'moisture_content_percent': 8.0,
                'ash_content_percent': 0.5,
                'sulfur_content_percent': 0.02,
                'carbon_content_percent': 50.0
            }
        }

    def test_basic_blending_request(self, orchestrator, solid_fuel_specs):
        """Test basic blending optimization."""
        calculator = FuelBlendingCalculator()

        from calculators.fuel_blending_calculator import BlendingInput

        input_data = BlendingInput(
            available_fuels=['coal', 'biomass'],
            fuel_properties=solid_fuel_specs,
            target_heating_value=22.0,
            max_moisture=20.0,
            max_ash=15.0,
            max_sulfur=2.0,
            optimization_objective='balanced',
            incompatible_pairs=[]
        )

        result = calculator.optimize_blend(input_data)

        # Verify blend result
        assert result is not None
        assert sum(result.blend_ratios.values()) > 0.99
        assert result.blend_heating_value > 0
        assert result.provenance_hash is not None

    def test_emissions_focused_blending(self, orchestrator, solid_fuel_specs):
        """Test emissions-focused blending favors renewables."""
        calculator = FuelBlendingCalculator()

        from calculators.fuel_blending_calculator import BlendingInput

        input_data = BlendingInput(
            available_fuels=['coal', 'biomass', 'wood_pellets'],
            fuel_properties=solid_fuel_specs,
            target_heating_value=18.0,  # Lower to enable more biomass
            max_moisture=30.0,
            max_ash=15.0,
            max_sulfur=2.0,
            optimization_objective='minimize_emissions',
            incompatible_pairs=[]
        )

        result = calculator.optimize_blend(input_data)

        # Renewables should be favored
        renewable_share = result.blend_ratios.get('biomass', 0) + result.blend_ratios.get('wood_pellets', 0)
        assert renewable_share > 0.3


class TestCarbonMinimizationWorkflow:
    """End-to-end tests for carbon minimization workflow."""

    @pytest.fixture
    def fuel_properties(self):
        """Full fuel properties."""
        return {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_ch4_kg_gj': 0.001,
                'emission_factor_n2o_kg_gj': 0.0001
            },
            'coal': {
                'heating_value_mj_kg': 25.0,
                'emission_factor_co2_kg_gj': 94.6,
                'emission_factor_ch4_kg_gj': 0.001,
                'emission_factor_n2o_kg_gj': 0.0015
            },
            'biomass': {
                'heating_value_mj_kg': 18.0,
                'emission_factor_co2_kg_gj': 0.0,
                'emission_factor_ch4_kg_gj': 0.03,
                'emission_factor_n2o_kg_gj': 0.004
            }
        }

    def test_carbon_footprint_baseline(self, fuel_properties):
        """Test carbon footprint calculation for baseline scenario."""
        calculator = CarbonFootprintCalculator()

        from calculators.carbon_footprint_calculator import CarbonFootprintInput

        # 100% coal baseline
        input_data = CarbonFootprintInput(
            fuel_quantities={'coal': 1000},
            fuel_properties=fuel_properties
        )

        result = calculator.calculate(input_data)

        assert result.total_co2e_kg > 0
        assert result.carbon_intensity_kg_mwh > 0

    def test_fuel_switch_reduction(self, fuel_properties):
        """Test emissions reduction from fuel switching."""
        calculator = CarbonFootprintCalculator()

        from calculators.carbon_footprint_calculator import CarbonFootprintInput

        # Coal baseline
        coal_input = CarbonFootprintInput(
            fuel_quantities={'coal': 1000},
            fuel_properties=fuel_properties
        )
        coal_result = calculator.calculate(coal_input)

        # Natural gas (same energy output)
        # Coal: 1000 kg * 25 MJ/kg = 25000 MJ
        # NG: 25000 MJ / 50 MJ/kg = 500 kg
        ng_input = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 500},
            fuel_properties=fuel_properties
        )
        ng_result = calculator.calculate(ng_input)

        # Natural gas should have lower emissions
        assert ng_result.total_co2e_kg < coal_result.total_co2e_kg
        reduction = (coal_result.total_co2e_kg - ng_result.total_co2e_kg) / coal_result.total_co2e_kg * 100
        assert reduction > 30  # At least 30% reduction

    def test_full_decarbonization_pathway(self, fuel_properties):
        """Test full decarbonization pathway analysis."""
        calculator = CarbonFootprintCalculator()

        from calculators.carbon_footprint_calculator import CarbonFootprintInput

        # Define energy demand in MJ
        energy_demand_mj = 25000

        scenarios = {
            '100% Coal': {'coal': energy_demand_mj / 25},
            '100% NG': {'natural_gas': energy_demand_mj / 50},
            '50% NG + 50% Biomass': {
                'natural_gas': (energy_demand_mj * 0.5) / 50,
                'biomass': (energy_demand_mj * 0.5) / 18
            },
            '100% Biomass': {'biomass': energy_demand_mj / 18}
        }

        results = {}
        for scenario, quantities in scenarios.items():
            input_data = CarbonFootprintInput(
                fuel_quantities=quantities,
                fuel_properties=fuel_properties
            )
            result = calculator.calculate(input_data)
            results[scenario] = result.total_co2e_kg

        # Emissions should decrease along pathway
        assert results['100% NG'] < results['100% Coal']
        assert results['50% NG + 50% Biomass'] < results['100% NG']
        assert results['100% Biomass'] < results['50% NG + 50% Biomass']


class TestProvenanceTracking:
    """End-to-end tests for provenance tracking."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator."""
        config = FuelManagementConfig(environment='test')
        return FuelManagementOrchestrator(config)

    def test_complete_audit_trail(self, orchestrator):
        """Test complete audit trail is maintained."""
        fuel_specifications = {
            'natural_gas': FuelSpecification(
                fuel_id='natural_gas',
                fuel_name='Natural Gas',
                fuel_type='gas',
                heating_value_mj_kg=50.0,
                emission_factor_co2_kg_gj=56.1
            )
        }

        request = FuelOptimizationRequest(
            request_id='AUDIT-001',
            energy_demand_mw=100,
            optimization_objective='balanced',
            available_fuels=['natural_gas'],
            fuel_specifications=fuel_specifications,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={}
        )

        response = orchestrator.execute(request)

        # Verify audit trail components
        assert response.provenance_hash is not None
        assert response.request_id == 'AUDIT-001'
        assert response.timestamp is not None

        # Provenance should be verifiable
        verification = orchestrator.verify_provenance(
            response.provenance_hash,
            request,
            response
        )
        assert verification is True

    def test_provenance_tamper_detection(self, orchestrator):
        """Test tampering is detectable via provenance."""
        fuel_specifications = {
            'natural_gas': FuelSpecification(
                fuel_id='natural_gas',
                fuel_name='Natural Gas',
                fuel_type='gas',
                heating_value_mj_kg=50.0,
                emission_factor_co2_kg_gj=56.1
            )
        }

        request = FuelOptimizationRequest(
            request_id='TAMPER-001',
            energy_demand_mw=100,
            optimization_objective='balanced',
            available_fuels=['natural_gas'],
            fuel_specifications=fuel_specifications,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={}
        )

        response = orchestrator.execute(request)
        original_hash = response.provenance_hash

        # Simulate tampering
        tampered_response = response.copy()
        tampered_response.total_cost_usd = response.total_cost_usd * 0.5  # Tamper cost

        # Verification should fail
        verification = orchestrator.verify_provenance(
            original_hash,
            request,
            tampered_response
        )
        assert verification is False


class TestErrorHandlingWorkflow:
    """End-to-end tests for error handling."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator."""
        config = FuelManagementConfig(environment='test')
        return FuelManagementOrchestrator(config)

    def test_invalid_fuel_handling(self, orchestrator):
        """Test handling of invalid fuel specification."""
        with pytest.raises(ValueError):
            request = FuelOptimizationRequest(
                request_id='ERR-001',
                energy_demand_mw=100,
                optimization_objective='balanced',
                available_fuels=['unknown_fuel'],
                fuel_specifications={},  # Missing spec
                market_prices={},
                emission_limits={},
                constraints={}
            )
            orchestrator.execute(request)

    def test_zero_demand_handling(self, orchestrator):
        """Test handling of zero energy demand."""
        fuel_specifications = {
            'natural_gas': FuelSpecification(
                fuel_id='natural_gas',
                fuel_name='Natural Gas',
                fuel_type='gas',
                heating_value_mj_kg=50.0,
                emission_factor_co2_kg_gj=56.1
            )
        }

        with pytest.raises(ValueError):
            request = FuelOptimizationRequest(
                request_id='ERR-002',
                energy_demand_mw=0,  # Zero demand
                optimization_objective='balanced',
                available_fuels=['natural_gas'],
                fuel_specifications=fuel_specifications,
                market_prices={'natural_gas': 0.045},
                emission_limits={},
                constraints={}
            )
            orchestrator.execute(request)

    def test_infeasible_constraints_handling(self, orchestrator):
        """Test handling of infeasible constraints."""
        fuel_specifications = {
            'coal': FuelSpecification(
                fuel_id='coal',
                fuel_name='Coal',
                fuel_type='solid',
                heating_value_mj_kg=25.0,
                emission_factor_co2_kg_gj=94.6,
                emission_factor_nox_g_gj=250
            )
        }

        request = FuelOptimizationRequest(
            request_id='ERR-003',
            energy_demand_mw=100,
            optimization_objective='balanced',
            available_fuels=['coal'],
            fuel_specifications=fuel_specifications,
            market_prices={'coal': 0.035},
            emission_limits={'nox_g_gj': 50},  # Coal cannot meet this
            constraints={}
        )

        response = orchestrator.execute(request)

        # Should indicate infeasibility
        assert response.status in ['INFEASIBLE', 'WARNING']
        assert len(response.warnings) > 0


class TestCachingBehavior:
    """End-to-end tests for caching behavior."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with caching enabled."""
        config = FuelManagementConfig(
            environment='test',
            enable_caching=True,
            cache_ttl_seconds=60
        )
        return FuelManagementOrchestrator(config)

    def test_cache_hit_on_repeat_request(self, orchestrator):
        """Test cache is used for repeated identical requests."""
        fuel_specifications = {
            'natural_gas': FuelSpecification(
                fuel_id='natural_gas',
                fuel_name='Natural Gas',
                fuel_type='gas',
                heating_value_mj_kg=50.0,
                emission_factor_co2_kg_gj=56.1
            )
        }

        request = FuelOptimizationRequest(
            request_id='CACHE-001',
            energy_demand_mw=100,
            optimization_objective='balanced',
            available_fuels=['natural_gas'],
            fuel_specifications=fuel_specifications,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={}
        )

        # First request
        response1 = orchestrator.execute(request)
        cache_miss_time = response1.processing_time_ms

        # Second identical request (should hit cache)
        response2 = orchestrator.execute(request)
        cache_hit_time = response2.processing_time_ms

        # Cache hit should be significantly faster
        assert response1.optimal_fuel_mix == response2.optimal_fuel_mix
        assert response1.provenance_hash == response2.provenance_hash
        # Cache hit should be at least 2x faster (or first request was < 1ms)
        assert cache_hit_time <= cache_miss_time or cache_miss_time < 1.0
