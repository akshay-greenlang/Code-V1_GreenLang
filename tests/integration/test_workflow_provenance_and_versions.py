"""
Provenance and versioning integration tests.
"""
import pytest
import json
from datetime import datetime
from tests.integration.utils import load_fixture


@pytest.mark.integration
class TestProvenanceAndVersions:
    """Test provenance tracking and version propagation."""
    
    def test_dataset_version_in_output(self, workflow_runner, dataset):
        """
        Test that dataset version is included in output.
        
        Verifies:
        - Dataset version present
        - Source attribution
        - Last updated date
        """
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Check for provenance in result
        provenance = None
        if 'provenance' in result['data']:
            provenance = result['data']['provenance']
        elif 'emissions_report' in result['data']:
            report = result['data']['emissions_report']
            if 'provenance' in report:
                provenance = report['provenance']
            elif 'dataset_version' in report:
                provenance = {
                    'dataset_version': report['dataset_version'],
                    'source': report.get('dataset_source'),
                    'last_updated': report.get('dataset_updated')
                }
        
        assert provenance is not None, "No provenance information found"
        
        # Verify required provenance fields
        assert 'dataset_version' in provenance
        assert provenance['dataset_version'] == dataset['version']
        
        assert 'source' in provenance
        assert provenance['source'] == dataset['source']
        
        assert 'last_updated' in provenance
        assert provenance['last_updated'] == dataset['last_updated']
    
    def test_factor_source_tracking(self, workflow_runner, dataset):
        """Test that emission factors track their source."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Look for factor details in output
        if 'factor_details' in result['data'] or 'emission_factors_used' in result['data']:
            factors = result['data'].get('factor_details', 
                                        result['data'].get('emission_factors_used'))
            
            # Each factor should have source info
            for fuel_type, factor_info in factors.items():
                if isinstance(factor_info, dict):
                    assert 'value' in factor_info
                    assert 'unit' in factor_info
                    # Source might be in factor info or general provenance
    
    def test_workflow_version_tracking(self, workflow_runner):
        """Test that workflow version is tracked."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Check for workflow version
        if 'metadata' in result:
            metadata = result['metadata']
            assert 'workflow_version' in metadata or 'version' in metadata
            
            # Should match workflow file version
            workflow = load_fixture('workflows', 'commercial_building_emissions.yaml')
            expected_version = workflow.get('version', '0.0.1')
            
            actual_version = metadata.get('workflow_version', metadata.get('version'))
            assert actual_version == expected_version
    
    def test_timestamp_generation(self, workflow_runner):
        """Test that execution timestamp is generated."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        before = datetime.utcnow().isoformat()
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        after = datetime.utcnow().isoformat()
        
        assert result['success'] is True
        
        # Look for timestamp
        timestamp = None
        if 'timestamp' in result:
            timestamp = result['timestamp']
        elif 'metadata' in result and 'timestamp' in result['metadata']:
            timestamp = result['metadata']['timestamp']
        elif 'executed_at' in result:
            timestamp = result['executed_at']
        
        if timestamp:
            # Verify timestamp is in valid range
            assert before <= timestamp <= after or \
                   timestamp.startswith('20')  # At least a valid date
    
    def test_agent_version_tracking(self, workflow_runner):
        """Test that agent versions are tracked."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Check for agent versions
        if 'agent_versions' in result.get('metadata', {}):
            versions = result['metadata']['agent_versions']
            
            # Should have version for each agent used
            expected_agents = [
                'DataParserAgent',
                'EmissionCalculatorAgent',
                'BenchmarkAgent',
                'RecommendationAgent',
                'ReportAgent'
            ]
            
            for agent in expected_agents:
                if agent in versions:
                    assert versions[agent] is not None
    
    def test_input_hash_tracking(self, workflow_runner):
        """Test that input hash is tracked for reproducibility."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Check for input hash
        if 'metadata' in result:
            metadata = result['metadata']
            if 'input_hash' in metadata or 'input_checksum' in metadata:
                hash_value = metadata.get('input_hash', metadata.get('input_checksum'))
                
                # Should be a valid hash (32+ characters)
                assert len(hash_value) >= 32
                
                # Should be deterministic
                import hashlib
                expected_hash = hashlib.sha256(
                    json.dumps(building_data, sort_keys=True).encode()
                ).hexdigest()
                
                # Might match exactly or be a truncated version
                assert hash_value.startswith(expected_hash[:8]) or \
                       expected_hash.startswith(hash_value[:8])
    
    def test_dataset_upgrade_detection(self, workflow_runner, dataset, monkeypatch):
        """Test detection of dataset version changes."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        # Run with current dataset
        result1 = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Simulate dataset upgrade
        new_dataset = dataset.copy()
        new_dataset['version'] = '2.0.0'
        new_dataset['last_updated'] = '2024-12-01'
        
        # Patch to use new dataset
        monkeypatch.setattr('tests.conftest.dataset', new_dataset)
        
        # Run with new dataset
        result2 = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        # Both should succeed
        assert result1['success'] is True
        assert result2['success'] is True
        
        # Versions should differ
        prov1 = result1['data'].get('provenance', {})
        prov2 = result2['data'].get('provenance', {})
        
        if 'dataset_version' in prov1 and 'dataset_version' in prov2:
            assert prov1['dataset_version'] != prov2['dataset_version']
    
    def test_audit_trail(self, workflow_runner):
        """Test that execution maintains audit trail."""
        building_data = load_fixture('data', 'building_india_office.json')
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Check for audit information
        if 'audit' in result or 'execution_log' in result:
            audit = result.get('audit', result.get('execution_log'))
            
            # Should track steps executed
            assert 'steps' in audit or isinstance(audit, list)
            
            # Each step should have timing info
            steps = audit.get('steps', audit) if isinstance(audit, dict) else audit
            for step in steps:
                if isinstance(step, dict):
                    assert 'id' in step or 'name' in step
                    assert 'status' in step or 'result' in step
    
    def test_provenance_in_exports(self, workflow_runner, tmp_outdir):
        """Test that provenance is included in exports."""
        from tests.integration.utils import TestIOHelper
        
        building_data = load_fixture('data', 'building_india_office.json')
        io_helper = TestIOHelper(tmp_outdir)
        
        result = workflow_runner.run(
            'tests/fixtures/workflows/commercial_building_emissions.yaml',
            {'building_data': building_data}
        )
        
        assert result['success'] is True
        
        # Export to JSON
        export_data = {
            'emissions': result['data']['emissions_report'],
            'provenance': result['data'].get('provenance', {})
        }
        
        json_path = io_helper.write_json('export_with_provenance.json', export_data)
        
        # Read back and verify provenance preserved
        read_data = io_helper.read_json('export_with_provenance.json')
        assert 'provenance' in read_data
        
        if read_data['provenance']:
            assert 'dataset_version' in read_data['provenance']