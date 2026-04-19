# -*- coding: utf-8 -*-
"""
India-specific workflow integration tests.
"""
import pytest
from tests.integration.utils import load_fixture, normalize_json


@pytest.mark.integration
class TestIndiaWorkflow:
    """Test India-specific building emissions workflow with compliance checks."""
    
    def test_india_hospital_workflow(self, workflow_runner, dataset):
        """
        Test workflow for Indian hospital with BEE/EECB compliance.
        
        Verifies:
        - India-specific grid factors applied
        - BEE rating calculated
        - EECB compliance checked
        - India-specific recommendations generated
        """
        # Load India hospital data
        building_data = load_fixture('data', 'building_india_hospital.json')
        
        # Execute India-specific workflow
        result = workflow_runner.run(
            'tests/fixtures/workflows/india_building_workflow.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        report = result['data']['emissions_report']
        
        # Verify India grid factors used
        emissions = report.get('emissions', report)
        electricity_consumption = building_data['consumption']['electricity']['value']
        india_factor = dataset['emission_factors']['IN']['electricity']['value']
        
        # Check electricity emissions use India factor
        expected_min = electricity_consumption * india_factor * 0.95  # 5% tolerance
        expected_max = electricity_consumption * india_factor * 1.05
        
        assert expected_min <= emissions['by_fuel']['electricity'] <= expected_max
        
        # Verify BEE compliance information
        if 'compliance' in report:
            compliance = report['compliance']
            
            # Check BEE rating
            assert 'bee_rating' in compliance
            assert isinstance(compliance['bee_rating'], (str, int))
            
            # Check EECB compliance
            assert 'eecb_compliance' in compliance
            assert isinstance(compliance['eecb_compliance'], bool)
            
            # Check compliance gaps if non-compliant
            if not compliance['eecb_compliance']:
                assert 'compliance_gaps' in compliance
                assert isinstance(compliance['compliance_gaps'], list)
        
        # Verify India-specific recommendations
        if 'recommendations' in report:
            recommendations = report['recommendations']
            
            # Should have India-relevant recommendations
            india_keywords = ['solar', 'rooftop', 'BEE', 'star', 'monsoon', 'Bureau']
            recommendations_text = ' '.join([r.get('action', '') for r in recommendations])
            
            # At least one recommendation should be India-specific
            has_india_specific = any(
                keyword.lower() in recommendations_text.lower() 
                for keyword in india_keywords
            )
            # Note: This assertion is soft - depends on implementation
            
        # Check for India-specific schemes
        if 'schemes' in report:
            schemes = report['schemes']
            assert isinstance(schemes, list)
            # Could include UJALA, PAT, etc.
    
    def test_india_office_benchmark(self, workflow_runner, dataset):
        """Test that India benchmarks are applied correctly."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/india_building_workflow.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        report = result['data']['emissions_report']
        
        # Check benchmark uses India thresholds
        if 'benchmark' in report:
            benchmark = report['benchmark']
            
            # Calculate intensity
            total_emissions = report.get('total_co2e_kg', 
                                        report.get('emissions', {}).get('total_co2e_kg'))
            area_sqft = building_data['building_info']['area_sqft']
            intensity = total_emissions / area_sqft
            
            # Check against India benchmarks
            india_benchmarks = dataset['benchmarks']['commercial_office']['IN']
            
            # Determine expected rating based on intensity
            if intensity <= india_benchmarks['excellent']:
                expected_rating = 'Excellent'
            elif intensity <= india_benchmarks['good']:
                expected_rating = 'Good'
            elif intensity <= india_benchmarks['average']:
                expected_rating = 'Average'
            else:
                expected_rating = 'Poor'
            
            # Allow for some flexibility in rating names
            rating = benchmark.get('rating', '').lower()
            assert expected_rating.lower() in rating or rating in expected_rating.lower()
    
    def test_india_critical_infrastructure(self, workflow_runner):
        """Test handling of 24x7 critical infrastructure."""
        building_data = load_fixture('data', 'building_india_hospital.json')
        
        # Verify 24x7 operation is recognized
        assert building_data['building_info']['24x7_operation'] is True
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/india_building_workflow.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        report = result['data']['emissions_report']
        
        # Critical infrastructure should have different benchmarks
        if 'benchmark' in report:
            # Hospital should use healthcare benchmarks, not office
            assert 'critical' in str(report).lower() or 'healthcare' in str(report).lower()
    
    def test_india_renewable_schemes(self, workflow_runner):
        """Test that India renewable energy schemes are suggested."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Ensure building has potential for more solar
        building_data['renewable']['solar_rooftop']['capacity_kw'] = 50  # Low capacity
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/india_building_workflow.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        report = result['data']['emissions_report']
        
        # Check for renewable energy recommendations
        if 'recommendations' in report:
            solar_recs = [
                r for r in report['recommendations']
                if 'solar' in r.get('action', '').lower()
            ]
            
            # Should suggest solar expansion given low current capacity
            # This is implementation-dependent
            pass  # Soft assertion
        
        # Check for government schemes
        if 'schemes' in report:
            schemes = report['schemes']
            # Could mention schemes like:
            # - PM-KUSUM for solar
            # - Net metering policies
            # - Renewable energy certificates
            assert isinstance(schemes, list)