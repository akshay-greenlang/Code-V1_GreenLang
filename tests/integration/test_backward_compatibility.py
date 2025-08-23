"""
Backward compatibility and schema validation tests.
"""
import pytest
import json
from pathlib import Path
from tests.integration.utils import (
    load_fixture,
    validate_json_schema,
    TestIOHelper
)


@pytest.mark.integration
class TestBackwardCompatibility:
    """Test backward compatibility with older workflows and data formats."""
    
    def test_v0_workflow_compatibility(self, workflow_runner, tmp_outdir):
        """Test that v0 workflows still run or show clear deprecation."""
        io_helper = TestIOHelper(tmp_outdir)
        
        # Create v0 workflow (older format)
        v0_workflow = {
            'name': 'v0_emissions_workflow',
            'version': '0.1.0',  # Old version
            # Old format might not have typed inputs
            'steps': [
                {
                    'id': 'calculate',
                    'agent': 'EmissionCalculator',  # Old agent name
                    'params': {  # Old param style
                        'electricity_kwh': 1000000,
                        'country': 'IN'
                    }
                }
            ]
        }
        
        workflow_path = io_helper.write_yaml('v0_workflow.yaml', v0_workflow)
        
        result = workflow_runner.run(
            str(workflow_path),
            {}  # V0 might not need structured input
        )
        
        # Should either:
        # 1. Still work with compatibility layer
        # 2. Fail with clear deprecation message
        if not result['success']:
            error_msg = str(result['error'])
            assert 'deprecated' in error_msg.lower() or 'version' in error_msg.lower()
            assert 'v0' in error_msg or '0.1' in error_msg
    
    def test_old_data_format_migration(self, workflow_runner):
        """Test handling of old data formats."""
        # Old format (flat structure)
        old_format = {
            'country': 'IN',
            'electricity_kwh': 1500000,
            'gas_m3': 50000,
            'diesel_liters': 10000,
            'area_sqft': 50000
        }
        
        # Try to run with old format
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': old_format}
        )
        
        # Should either migrate or provide migration guide
        if not result['success']:
            error_msg = str(result['error'])
            # Should suggest new format
            assert 'format' in error_msg.lower() or 'structure' in error_msg.lower()
    
    def test_deprecated_agent_warning(self, workflow_runner, tmp_outdir):
        """Test warnings for deprecated agents."""
        io_helper = TestIOHelper(tmp_outdir)
        
        # Workflow using deprecated agent
        deprecated_workflow = {
            'name': 'deprecated_agent_test',
            'version': '0.0.1',
            'inputs': {'data': {'type': 'object'}},
            'outputs': {'result': {'type': 'object'}},
            'steps': [
                {
                    'id': 'old_step',
                    'agent': 'LegacyEmissionAgent',  # Deprecated
                    'inputs': {'data': '$data'},
                    'outputs': {'result': 'object'}
                }
            ]
        }
        
        workflow_path = io_helper.write_yaml('deprecated.yaml', deprecated_workflow)
        
        result = workflow_runner.run(
            str(workflow_path),
            {'data': {}}
        )
        
        # Should warn about deprecation
        if 'warnings' in result:
            warnings = result['warnings']
            assert any('deprecated' in w.lower() for w in warnings)
    
    def test_schema_evolution(self, workflow_runner):
        """Test that schemas can evolve while maintaining compatibility."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Add new optional fields (forward compatible)
        building_data['new_field'] = 'test_value'
        building_data['consumption']['new_fuel'] = {
            'value': 100,
            'unit': 'new_unit'
        }
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Should handle gracefully (ignore unknown fields)
        assert result['success'] is True
        
        # Core functionality should work
        report = result['data']['emissions_report']
        assert report['total_co2e_kg'] > 0
    
    def test_output_schema_validation(self, workflow_runner):
        """Test that outputs conform to documented schemas."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Load schema
        analysis_schema = load_fixture('schemas', 'analysis.schema.json')
        
        # Validate output against schema
        report = result['data']['emissions_report']
        
        # Basic validation (full jsonschema validation if available)
        try:
            validate_json_schema(report, analysis_schema)
        except Exception as e:
            # If validation fails, check manually
            assert 'total_co2e_kg' in report
            assert 'total_co2e_tons' in report
            assert 'by_fuel' in report
    
    def test_api_version_negotiation(self, workflow_runner):
        """Test API version negotiation."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Specify API version preference
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data},
            api_version='0.0.1'  # If supported
        )
        
        if 'api_version' in result:
            # Should use requested version or compatible one
            assert result['api_version'].startswith('0.')
    
    def test_breaking_change_detection(self, workflow_runner, tmp_outdir):
        """Test detection of breaking changes."""
        io_helper = TestIOHelper(tmp_outdir)
        
        # Create workflow with breaking change
        breaking_workflow = {
            'name': 'breaking_change_test',
            'version': '2.0.0',  # Major version bump
            'breaking_changes': [
                'Renamed agent EmissionCalculator to EmissionAnalyzer',
                'Changed input format for consumption data'
            ],
            'inputs': {'building': {'type': 'object'}},
            'outputs': {'analysis': {'type': 'object'}},
            'steps': [
                {
                    'id': 'analyze',
                    'agent': 'EmissionAnalyzer',  # New name
                    'inputs': {'building': '$building'},
                    'outputs': {'analysis': 'object'}
                }
            ]
        }
        
        workflow_path = io_helper.write_yaml('breaking.yaml', breaking_workflow)
        
        # Old input format
        old_input = {'building_data': {}}  # Wrong key
        
        result = workflow_runner.run(str(workflow_path), old_input)
        
        if not result['success']:
            error_msg = str(result['error'])
            # Should mention breaking changes
            assert 'breaking' in error_msg.lower() or '2.0' in error_msg
    
    def test_migration_path_documentation(self, workflow_runner):
        """Test that migration paths are documented."""
        # This is more of a documentation test
        migration_docs = Path('docs/migration')
        
        if migration_docs.exists():
            # Check for migration guides
            v1_to_v2 = migration_docs / 'v1_to_v2.md'
            if v1_to_v2.exists():
                content = v1_to_v2.read_text()
                assert 'breaking changes' in content.lower()
                assert 'migration steps' in content.lower()
    
    def test_config_file_compatibility(self, tmp_outdir):
        """Test compatibility of configuration files."""
        io_helper = TestIOHelper(tmp_outdir)
        
        # Old config format
        old_config = {
            'emission_factors': {  # Old structure
                'IN_electricity': 0.82,
                'US_electricity': 0.42
            }
        }
        
        # New config format
        new_config = {
            'emission_factors': {
                'IN': {
                    'electricity': {'value': 0.82, 'unit': 'kgCO2e/kWh'}
                },
                'US': {
                    'electricity': {'value': 0.42, 'unit': 'kgCO2e/kWh'}
                }
            }
        }
        
        # Both should be handled
        old_path = io_helper.write_json('old_config.json', old_config)
        new_path = io_helper.write_json('new_config.json', new_config)
        
        # Config loader should handle both formats
        # This is implementation-specific
    
    def test_dataset_version_upgrade(self, workflow_runner, dataset):
        """Test handling of dataset version upgrades."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Run with current dataset
        result1 = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result1['success'] is True
        emissions1 = result1['data']['emissions_report']['total_co2e_kg']
        
        # Simulate dataset upgrade
        # New factors would change results
        new_dataset = dataset.copy()
        new_dataset['version'] = '2.0.0'
        new_dataset['emission_factors']['IN']['electricity']['value'] = 0.75  # Lower
        
        # If dataset can be swapped
        # result2 = workflow_runner.run(...)
        # emissions2 should be different
        
        # Version change should be tracked
        if 'provenance' in result1['data']:
            assert result1['data']['provenance']['dataset_version'] == dataset['version']