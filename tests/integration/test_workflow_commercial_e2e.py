"""
End-to-end integration test for commercial building workflow.
"""
import pytest
import json
from pathlib import Path
from tests.integration.utils import (
    normalize_json,
    assert_numerical_invariants,
    compare_snapshots,
    load_fixture
)


@pytest.mark.integration
class TestCommercialBuildingE2E:
    """Test the canonical commercial building emissions workflow."""
    
    def test_happy_path_india_office(self, workflow_runner, dataset, 
                                    assert_close, tmp_outdir):
        """
        Test the complete workflow for an India office building.
        
        Verifies:
        - Workflow executes successfully
        - All required output fields present
        - Numerical invariants hold
        - Provenance information included
        - Snapshot matches golden output
        """
        # Load test data
        building_data = load_fixture('data', 'building_india_office.json')
        workflow_def = load_fixture('workflows', 'commercial_building_emissions.yaml')
        
        # Execute workflow
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Assert success
        assert result['success'] is True, f"Workflow failed: {result.get('error')}"
        
        # Extract emissions report
        report = result['data']['emissions_report']
        
        # Verify required fields
        assert 'total_co2e_kg' in report
        assert 'total_co2e_tons' in report
        assert 'by_fuel' in report
        assert 'intensity' in report
        assert 'benchmark' in report
        assert 'recommendations' in report
        
        # Check emission calculation
        emissions = report.get('emissions', report)
        
        # Verify emissions are positive
        assert emissions['total_co2e_kg'] > 0
        assert emissions['total_co2e_tons'] > 0
        
        # Check fuel breakdown
        assert 'electricity' in emissions['by_fuel']
        assert 'natural_gas' in emissions['by_fuel']
        assert 'diesel' in emissions['by_fuel']
        
        # Verify numerical invariants
        assert_numerical_invariants(emissions)
        
        # Check sum of fuel emissions equals total
        fuel_sum = sum(emissions['by_fuel'].values())
        assert_close(fuel_sum, emissions['total_co2e_kg'], rel_tol=1e-6)
        
        # Check kg to tons conversion
        expected_tons = emissions['total_co2e_kg'] / 1000
        assert_close(emissions['total_co2e_tons'], expected_tons, rel_tol=1e-6)
        
        # Verify intensity calculations
        if 'intensity' in report:
            area_sqft = building_data['building_info']['area_sqft']
            expected_intensity = emissions['total_co2e_kg'] / area_sqft
            
            if 'per_sqft_year' in report['intensity']:
                assert_close(
                    report['intensity']['per_sqft_year'],
                    expected_intensity,
                    rel_tol=1e-4
                )
        
        # Check benchmark rating
        assert 'rating' in report['benchmark']
        assert report['benchmark']['rating'] in ['Excellent', 'Good', 'Average', 'Poor']
        
        # Verify recommendations
        assert len(report['recommendations']) >= 3
        for rec in report['recommendations'][:3]:
            assert 'action' in rec
            assert 'savings' in rec or 'impact' in rec
        
        # Check provenance information
        assert 'provenance' in result['data'] or 'dataset_version' in report
        if 'provenance' in result['data']:
            prov = result['data']['provenance']
            assert 'dataset_version' in prov
            assert 'source' in prov
            assert 'last_updated' in prov
        
        # Generate and compare report snapshot
        report_json = json.dumps(normalize_json(report), indent=2)
        snapshot_path = Path('tests/integration/snapshots/reports/commercial_india_office.json')
        assert compare_snapshots(report_json, snapshot_path, update=False)
    
    def test_us_office_calculation(self, workflow_runner, dataset, assert_close):
        """Test workflow with US office data."""
        # Load test data
        building_data = load_fixture('data', 'building_us_office.json')
        
        # Execute workflow
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Extract emissions
        emissions = result['data']['emissions_report'].get('emissions', 
                                                          result['data']['emissions_report'])
        
        # Verify US grid factor is applied
        electricity_kwh = building_data['consumption']['electricity']['value']
        us_factor = dataset['emission_factors']['US']['electricity']['value']
        
        expected_electricity_emissions = electricity_kwh * us_factor
        
        # Check electricity emissions calculation
        assert_close(
            emissions['by_fuel']['electricity'],
            expected_electricity_emissions,
            rel_tol=0.01  # 1% tolerance for rounding
        )
        
        # US should have lower emissions than India for same consumption
        # This is a data-driven assertion based on current factors
        india_factor = dataset['emission_factors']['IN']['electricity']['value']
        assert us_factor < india_factor, "Test assumption: US factor < India factor"
    
    def test_renewable_energy_offset(self, workflow_runner):
        """Test that renewable energy generation is properly accounted."""
        # Load base data
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Run with renewable
        result_with_renewable = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Remove renewable and run again
        building_no_renewable = building_data.copy()
        building_no_renewable['renewable'] = {'solar_rooftop': {'capacity_kw': 0, 'generation_kwh': 0}}
        
        result_without_renewable = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_no_renewable}
        )
        
        # Emissions should be lower with renewable energy
        emissions_with = result_with_renewable['data']['emissions_report']['total_co2e_kg']
        emissions_without = result_without_renewable['data']['emissions_report']['total_co2e_kg']
        
        assert emissions_with < emissions_without, "Renewable energy should reduce emissions"
        
        # The difference should roughly match solar generation * grid factor
        solar_generation = building_data['renewable']['solar_rooftop']['generation_kwh']
        assert solar_generation > 0, "Test data should have solar generation"
    
    @pytest.mark.performance
    @pytest.mark.timeout(2)
    def test_performance_budget(self, workflow_runner):
        """Test that workflow completes within performance budget."""
        import time
        
        building_data = load_fixture('data', 'building_india_office.json')
        
        start_time = time.time()
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        end_time = time.time()
        
        assert result['success'] is True
        
        # Should complete in under 2 seconds
        execution_time = end_time - start_time
        assert execution_time < 2.0, f"Workflow took {execution_time:.2f}s, exceeding 2s budget"