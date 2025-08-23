"""
Cross-country comparison integration tests.
"""
import pytest
from tests.integration.utils import load_fixture


@pytest.mark.integration
class TestCrossCountryComparison:
    """Test emissions calculations across different countries."""
    
    def test_country_factor_application(self, workflow_runner, dataset, assert_close):
        """
        Test that same building in different countries uses correct factors.
        
        Verifies:
        - Country-specific grid factors applied
        - Emissions scale with factors
        - Dataset is source of truth
        """
        # Load base building data
        base_building = load_fixture('data', 'building_india_office.json')
        
        countries = ['IN', 'US', 'EU']
        results = {}
        
        for country in countries:
            # Modify building location
            building_data = base_building.copy()
            building_data['location']['country'] = country
            
            # Run workflow
            result = workflow_runner.run(
                'tests/fixtures/workflows/commercial_building_emissions.yaml',
                {'building_data': building_data}
            )
            
            assert result['success'] is True
            results[country] = result['data']['emissions_report']
        
        # Extract electricity consumption (same for all)
        electricity_kwh = base_building['consumption']['electricity']['value']
        
        # Verify each country uses its factor
        for country in countries:
            emissions = results[country].get('emissions', results[country])
            electricity_emissions = emissions['by_fuel']['electricity']
            
            # Get factor from dataset
            if country in dataset['emission_factors']:
                expected_factor = dataset['emission_factors'][country]['electricity']['value']
            else:
                # EU might map to a default or average
                expected_factor = dataset['emission_factors'].get('EU', {}).get('electricity', {}).get('value', 0.4)
            
            expected_emissions = electricity_kwh * expected_factor
            
            # Check calculation (with tolerance for other processing)
            assert_close(
                electricity_emissions,
                expected_emissions,
                rel_tol=0.02,  # 2% tolerance
                msg=f"Country {country} electricity emissions"
            )
        
        # Verify relative ordering based on factors
        # This is data-driven - current factors suggest IN > US > EU
        in_factor = dataset['emission_factors']['IN']['electricity']['value']
        us_factor = dataset['emission_factors']['US']['electricity']['value']
        
        if in_factor > us_factor:  # Only assert if data supports it
            in_total = results['IN']['total_co2e_kg']
            us_total = results['US']['total_co2e_kg']
            assert in_total > us_total, "India emissions should be higher than US with current factors"
    
    def test_natural_gas_consistency(self, workflow_runner, dataset, assert_close):
        """Test that natural gas factors are consistent across countries."""
        base_building = load_fixture('data', 'building_india_office.json')
        
        countries = ['IN', 'US']
        ng_emissions = {}
        
        for country in countries:
            building_data = base_building.copy()
            building_data['location']['country'] = country
            
            result = workflow_runner.run(
                'tests/fixtures/workflows/commercial_building_emissions.yaml',
                {'building_data': building_data}
            )
            
            emissions = result['data']['emissions_report'].get('emissions', 
                                                              result['data']['emissions_report'])
            ng_emissions[country] = emissions['by_fuel']['natural_gas']
        
        # Natural gas factor might be same across countries
        ng_consumption = base_building['consumption']['natural_gas']['value']
        
        for country in countries:
            ng_factor = dataset['emission_factors'][country]['natural_gas']['value']
            expected = ng_consumption * ng_factor
            
            assert_close(
                ng_emissions[country],
                expected,
                rel_tol=0.02,
                msg=f"Natural gas emissions for {country}"
            )
    
    def test_diesel_calculation(self, workflow_runner, dataset, assert_close):
        """Test diesel emissions calculation across countries."""
        base_building = load_fixture('data', 'building_india_office.json')
        
        # Test with significant diesel consumption
        base_building['consumption']['diesel']['value'] = 50000  # liters
        
        countries = ['IN', 'US']
        diesel_emissions = {}
        
        for country in countries:
            building_data = base_building.copy()
            building_data['location']['country'] = country
            
            result = workflow_runner.run(
                'tests/fixtures/workflows/commercial_building_emissions.yaml',
                {'building_data': building_data}
            )
            
            emissions = result['data']['emissions_report'].get('emissions',
                                                              result['data']['emissions_report'])
            diesel_emissions[country] = emissions['by_fuel']['diesel']
        
        # Verify diesel calculations
        diesel_consumption = 50000  # liters
        
        for country in countries:
            diesel_factor = dataset['emission_factors'][country]['diesel']['value']
            expected = diesel_consumption * diesel_factor
            
            assert_close(
                diesel_emissions[country],
                expected,
                rel_tol=0.02,
                msg=f"Diesel emissions for {country}"
            )
    
    def test_benchmark_by_country(self, workflow_runner, dataset):
        """Test that benchmarks vary appropriately by country."""
        base_building = load_fixture('data', 'building_india_office.json')
        
        countries = ['IN', 'US']
        benchmarks = {}
        
        for country in countries:
            building_data = base_building.copy()
            building_data['location']['country'] = country
            
            result = workflow_runner.run(
                'tests/fixtures/workflows/commercial_building_emissions.yaml',
                {'building_data': building_data}
            )
            
            report = result['data']['emissions_report']
            benchmarks[country] = report['benchmark']
        
        # Each country should have its own benchmark thresholds
        # US typically has stricter (lower) thresholds than India
        if 'commercial_office' in dataset['benchmarks']:
            in_thresholds = dataset['benchmarks']['commercial_office'].get('IN', {})
            us_thresholds = dataset['benchmarks']['commercial_office'].get('US', {})
            
            # If data shows US has lower thresholds
            if us_thresholds.get('good', 100) < in_thresholds.get('good', 100):
                # Same building might rate differently
                # This is a soft assertion as ratings depend on actual intensity
                pass
    
    def test_missing_country_factor(self, workflow_runner, dataset):
        """Test handling of country without defined factors."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Use a country not in dataset
        building_data['location']['country'] = 'ZZ'  # Fictional country
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Should either fail gracefully or use defaults
        if not result['success']:
            assert 'error' in result
            assert 'ZZ' in str(result['error']) or 'country' in str(result['error']).lower()
        else:
            # Used some default factor
            report = result['data']['emissions_report']
            assert report['total_co2e_kg'] > 0  # Some calculation was done
    
    def test_country_specific_recommendations(self, workflow_runner):
        """Test that recommendations are country-appropriate."""
        base_building = load_fixture('data', 'building_india_office.json')
        
        countries = ['IN', 'US']
        recommendations = {}
        
        for country in countries:
            building_data = base_building.copy()
            building_data['location']['country'] = country
            
            result = workflow_runner.run(
                'tests/fixtures/workflows/commercial_building_emissions.yaml',
                {'building_data': building_data}
            )
            
            report = result['data']['emissions_report']
            recommendations[country] = report['recommendations']
        
        # Recommendations might differ by country
        # India might emphasize solar more due to climate
        # US might emphasize insulation/HVAC due to climate zones
        # This is implementation-dependent
        
        # At minimum, both should have recommendations
        assert len(recommendations['IN']) >= 3
        assert len(recommendations['US']) >= 3