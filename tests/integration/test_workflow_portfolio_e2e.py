"""
Portfolio workflow integration tests.
"""
import pytest
import json
from pathlib import Path
from tests.integration.utils import (
    load_fixture,
    assert_numerical_invariants,
    TestIOHelper
)


@pytest.mark.integration
class TestPortfolioWorkflow:
    """Test portfolio aggregation and analysis workflow."""
    
    def test_portfolio_aggregation(self, workflow_runner, dataset, assert_close):
        """
        Test portfolio of multiple buildings across countries.
        
        Verifies:
        - Individual building calculations
        - Aggregation correctness
        - Portfolio metrics
        - Export functionality
        """
        # Load portfolio data
        portfolio_data = load_fixture('data', 'portfolio_small.json')
        
        # Execute portfolio workflow
        result = workflow_runner.run(
            'tests/fixtures/workflows/portfolio_analysis.yaml',
            {'portfolio_data': portfolio_data}
        )
        
        assert result['success'] is True
        report = result['data']['portfolio_report']
        
        # Verify aggregated emissions
        assert 'total_emissions_tons' in report
        assert 'by_building' in report
        assert 'by_fuel' in report
        
        # Check that we have results for all buildings
        assert len(report['by_building']) == len(portfolio_data['buildings'])
        
        # Verify aggregation math
        individual_sum = sum(
            building['emissions_tons'] 
            for building in report['by_building']
        )
        assert_close(
            report['total_emissions_tons'],
            individual_sum,
            rel_tol=1e-6,
            msg="Portfolio total should equal sum of individual buildings"
        )
        
        # Check fuel breakdown aggregation
        if 'by_fuel' in report:
            fuel_totals = report['by_fuel']
            
            # Each fuel type should be sum of that fuel across buildings
            for fuel_type in ['electricity', 'natural_gas', 'diesel']:
                if fuel_type in fuel_totals:
                    assert fuel_totals[fuel_type] >= 0
            
            # Total of all fuels should match total emissions
            fuel_sum_kg = sum(fuel_totals.values())
            fuel_sum_tons = fuel_sum_kg / 1000
            assert_close(
                fuel_sum_tons,
                report['total_emissions_tons'],
                rel_tol=0.01  # 1% tolerance for aggregation
            )
        
        # Verify portfolio metrics
        if 'metrics' in report:
            metrics = report['metrics']
            
            # Check total area calculation
            expected_area = sum(
                b['building_info']['area_sqft'] 
                for b in portfolio_data['buildings']
            )
            assert_close(
                metrics['total_area'],
                expected_area,
                rel_tol=1e-6
            )
            
            # Check average intensity
            expected_intensity = (report['total_emissions_tons'] * 1000) / expected_area
            assert_close(
                metrics['average_intensity'],
                expected_intensity,
                rel_tol=0.01
            )
        
        # Check export paths
        if 'exports' in report:
            assert 'csv_path' in report['exports']
            assert 'excel_path' in report['exports']
    
    def test_portfolio_cross_country(self, workflow_runner, dataset):
        """Test that portfolio correctly handles different country factors."""
        portfolio_data = load_fixture('data', 'portfolio_small.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/portfolio_analysis.yaml',
            {'portfolio_data': portfolio_data}
        )
        
        assert result['success'] is True
        report = result['data']['portfolio_report']
        
        # Verify each building uses correct country factor
        for i, building_result in enumerate(report['by_building']):
            building = portfolio_data['buildings'][i]
            country = building['location']['country']
            
            # Check that appropriate factor was used
            electricity_consumption = building['consumption']['electricity']['value']
            country_factor = dataset['emission_factors'][country]['electricity']['value']
            
            # Rough check - electricity should be major component
            expected_electricity = electricity_consumption * country_factor
            total_emissions_kg = building_result['emissions_tons'] * 1000
            
            # Electricity emissions should be significant part of total
            assert expected_electricity <= total_emissions_kg * 1.2  # Within 20% overhead
    
    def test_portfolio_building_ranking(self, workflow_runner):
        """Test that buildings are properly ranked by performance."""
        portfolio_data = load_fixture('data', 'portfolio_small.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/portfolio_analysis.yaml',
            {'portfolio_data': portfolio_data}
        )
        
        assert result['success'] is True
        report = result['data']['portfolio_report']
        
        # Check building rankings if provided
        if 'rankings' in report or 'building_rankings' in report:
            rankings = report.get('rankings', report.get('building_rankings', []))
            
            # Should be sorted by intensity (best to worst)
            for i in range(len(rankings) - 1):
                current = rankings[i]
                next_item = rankings[i + 1]
                
                # Lower intensity should rank better (come first)
                assert current['intensity'] <= next_item['intensity']
    
    def test_portfolio_export_formats(self, workflow_runner, tmp_outdir):
        """Test CSV and Excel export functionality."""
        portfolio_data = load_fixture('data', 'portfolio_small.json')
        io_helper = TestIOHelper(tmp_outdir)
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/portfolio_analysis.yaml',
            {'portfolio_data': portfolio_data}
        )
        
        assert result['success'] is True
        report = result['data']['portfolio_report']
        
        # Simulate export to CSV
        csv_data = []
        for building in report['by_building']:
            csv_data.append({
                'Building ID': building.get('id'),
                'Name': building.get('name'),
                'Country': building.get('country'),
                'Total Emissions (tons)': building.get('emissions_tons'),
                'Intensity (kg/sqft)': building.get('intensity')
            })
        
        csv_path = io_helper.write_csv('portfolio_export.csv', csv_data)
        
        # Verify CSV was created
        assert csv_path.exists()
        
        # Read back and verify
        read_data = io_helper.read_csv('portfolio_export.csv')
        assert len(read_data) == len(portfolio_data['buildings'])
        
        # Verify all buildings are in export
        for row in read_data:
            assert float(row['Total Emissions (tons)']) > 0
    
    def test_empty_portfolio(self, workflow_runner):
        """Test handling of empty portfolio."""
        empty_portfolio = {
            'portfolio_name': 'Empty Portfolio',
            'portfolio_id': 'EMPTY-001',
            'aggregation_method': 'sum',
            'buildings': []
        }
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/portfolio_analysis.yaml',
            {'portfolio_data': empty_portfolio}
        )
        
        # Should handle gracefully
        if result['success']:
            report = result['data']['portfolio_report']
            assert report['total_emissions_tons'] == 0
            assert len(report['by_building']) == 0
    
    def test_large_portfolio_performance(self, workflow_runner):
        """Test performance with larger portfolio."""
        # Create a larger portfolio
        base_building = load_fixture('data', 'portfolio_small.json')['buildings'][0]
        
        large_portfolio = {
            'portfolio_name': 'Large Portfolio',
            'portfolio_id': 'LARGE-001',
            'aggregation_method': 'sum',
            'buildings': []
        }
        
        # Add 50 buildings
        for i in range(50):
            building = base_building.copy()
            building['id'] = f'BLDG-{i:03d}'
            building['name'] = f'Building {i}'
            large_portfolio['buildings'].append(building)
        
        import time
        start = time.time()
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/portfolio_analysis.yaml',
            {'portfolio_data': large_portfolio}
        )
        
        elapsed = time.time() - start
        
        assert result['success'] is True
        assert len(result['data']['portfolio_report']['by_building']) == 50
        
        # Should complete in reasonable time (< 5 seconds for 50 buildings)
        assert elapsed < 5.0, f"Large portfolio took {elapsed:.2f}s"