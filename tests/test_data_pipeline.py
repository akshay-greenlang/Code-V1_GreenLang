"""
Unit Tests for Data Pipeline

Tests for validation, import, monitoring, and workflow components.
"""

import pytest
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import tempfile

from greenlang.data.pipeline import (
    EmissionFactorValidator,
    AutomatedImportPipeline,
    DataQualityMonitor,
    UpdateWorkflow,
    ApprovalManager,
    RollbackManager
)
from greenlang.data.pipeline.models import (
    ImportJob,
    ImportStatus,
    ChangeType,
    ReviewStatus
)


class TestValidation:
    """Test validation framework."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return EmissionFactorValidator()

    @pytest.fixture
    def valid_factor_data(self):
        """Valid emission factor data."""
        return {
            'name': 'Natural Gas',
            'emission_factor_kg_co2e_per_kwh': 0.202,
            'scope': 'Scope 1',
            'source': 'EPA',
            'uri': 'https://www.epa.gov/climateleadership',
            'last_updated': '2024-11-01',
            'geographic_scope': 'United States',
            'data_quality': 'Tier 1'
        }

    @pytest.fixture
    def invalid_factor_data(self):
        """Invalid emission factor data."""
        return {
            'name': 'Bad Factor',
            'emission_factor_kg_co2e_per_kwh': -1.0,  # Invalid: negative
            'scope': 'Scope 1',
            # Missing source and URI
            'last_updated': '2010-01-01',  # Too old
        }

    @pytest.mark.asyncio
    async def test_validate_valid_factor(self, validator, valid_factor_data):
        """Test validation of valid factor."""
        result = await validator.validate_factor('test_factor', valid_factor_data)

        assert result.is_valid
        assert result.quality_score > 70
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_invalid_factor(self, validator, invalid_factor_data):
        """Test validation of invalid factor."""
        result = await validator.validate_factor('bad_factor', invalid_factor_data)

        assert not result.is_valid
        assert result.quality_score < 50
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_range_validation(self, validator):
        """Test range validation."""
        # Extreme value should fail
        extreme_data = {
            'name': 'Extreme Factor',
            'emission_factor_kg_co2e_per_kwh': 50000.0,  # Unrealistically high
            'scope': 'Scope 1',
            'source': 'Test',
            'uri': 'https://example.com',
            'last_updated': '2024-01-01'
        }

        result = await validator.validate_factor('extreme', extreme_data)
        assert not result.is_valid

    @pytest.mark.asyncio
    async def test_freshness_validation(self, validator):
        """Test date freshness validation."""
        # Old data should generate warnings/errors
        old_data = {
            'name': 'Old Factor',
            'emission_factor_kg_co2e_per_kwh': 0.5,
            'scope': 'Scope 1',
            'source': 'EPA',
            'uri': 'https://www.epa.gov',
            'last_updated': '2020-01-01'  # >3 years old
        }

        result = await validator.validate_factor('old', old_data)
        # Should have warnings about age
        assert len(result.warnings) > 0 or len(result.errors) > 0


class TestImportPipeline:
    """Test import pipeline."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        # Create schema
        from greenlang.db.emission_factors_schema import create_database
        create_database(db_path, overwrite=True)

        yield db_path

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def temp_yaml(self):
        """Create temporary YAML file."""
        data = {
            'metadata': {'version': '1.0.0'},
            'fuels': {
                'natural_gas': {
                    'name': 'Natural Gas',
                    'emission_factor_kg_co2e_per_kwh': 0.202,
                    'scope': 'Scope 1',
                    'source': 'EPA',
                    'uri': 'https://www.epa.gov/climateleadership',
                    'last_updated': '2024-11-01'
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
            yaml_path = f.name

        yield yaml_path

        Path(yaml_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_import_success(self, temp_db, temp_yaml):
        """Test successful import."""
        pipeline = AutomatedImportPipeline(temp_db)

        job = ImportJob(
            job_id='test_import_1',
            job_name='Test Import',
            source_files=[temp_yaml],
            target_database=temp_db,
            validate_before_import=False,  # Skip validation for speed
            triggered_by='test',
            trigger_type='manual'
        )

        result = await pipeline.execute_import(job, validate_before=False)

        assert result.status == ImportStatus.COMPLETED
        assert result.successful_imports > 0
        assert result.failed_imports == 0

    @pytest.mark.asyncio
    async def test_import_with_validation(self, temp_db, temp_yaml):
        """Test import with validation."""
        pipeline = AutomatedImportPipeline(temp_db)

        job = ImportJob(
            job_id='test_import_2',
            job_name='Test Import with Validation',
            source_files=[temp_yaml],
            target_database=temp_db,
            validate_before_import=True,
            triggered_by='test',
            trigger_type='manual'
        )

        result = await pipeline.execute_import(job, validate_before=True)

        assert result.validation_result is not None
        assert result.pre_import_validation_passed


class TestRollback:
    """Test rollback functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        from greenlang.db.emission_factors_schema import create_database
        create_database(db_path, overwrite=True)

        yield db_path

        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def backup_dir(self):
        """Create temporary backup directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_create_backup(self, temp_db, backup_dir):
        """Test backup creation."""
        rollback_mgr = RollbackManager(temp_db, backup_dir)

        backup_path = rollback_mgr.create_backup('test_job_1')

        assert Path(backup_path).exists()
        assert Path(backup_path).stat().st_size > 0

    def test_rollback(self, temp_db, backup_dir):
        """Test rollback to backup."""
        rollback_mgr = RollbackManager(temp_db, backup_dir)

        # Create backup
        backup_path = rollback_mgr.create_backup('test_job_2')

        # Modify database
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM emission_factors")
        conn.commit()
        conn.close()

        # Verify deletion
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM emission_factors")
        count_before = cursor.fetchone()[0]
        conn.close()

        # Rollback
        success = rollback_mgr.rollback(backup_path)

        assert success

        # Verify restoration
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM emission_factors")
        count_after = cursor.fetchone()[0]
        conn.close()

        # Should be restored
        assert count_after >= count_before


class TestMonitoring:
    """Test monitoring components."""

    @pytest.fixture
    def temp_db_with_data(self):
        """Create database with sample data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        from greenlang.db.emission_factors_schema import create_database
        create_database(db_path, overwrite=True)

        # Add sample data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        factors = [
            ('fuel_diesel', 'Diesel', 'fuels', 'diesel', 2.68, 'liter', 'Scope 1', 'EPA',
             'https://epa.gov', '2024-01-01', 'United States', 'Tier 1'),
            ('elec_us_grid', 'US Grid', 'electricity', 'grid', 0.42, 'kwh', 'Scope 2', 'EPA',
             'https://epa.gov', '2024-01-01', 'United States', 'Tier 1'),
            ('old_factor', 'Old Factor', 'fuels', 'coal', 2.4, 'kg', 'Scope 1', 'Old Source',
             'https://example.com', '2020-01-01', 'Global', 'Tier 1')
        ]

        for factor in factors:
            cursor.execute("""
                INSERT INTO emission_factors (
                    factor_id, name, category, subcategory,
                    emission_factor_value, unit, scope,
                    source_org, source_uri, last_updated,
                    geographic_scope, data_quality_tier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, factor)

        conn.commit()
        conn.close()

        yield db_path

        Path(db_path).unlink(missing_ok=True)

    def test_quality_metrics(self, temp_db_with_data):
        """Test quality metrics calculation."""
        monitor = DataQualityMonitor(temp_db_with_data)

        metrics = monitor.calculate_quality_metrics()

        assert metrics.total_factors == 3
        assert metrics.overall_quality_score >= 0
        assert metrics.overall_quality_score <= 100
        assert metrics.unique_sources >= 1
        assert metrics.stale_factors_count >= 1  # old_factor

    def test_coverage_analysis(self, temp_db_with_data):
        """Test coverage analysis."""
        monitor = DataQualityMonitor(temp_db_with_data)

        coverage = monitor.coverage_analyzer.analyze_category_coverage()

        assert coverage['total_categories'] >= 2
        assert 'fuels' in coverage['coverage_by_category']
        assert 'electricity' in coverage['coverage_by_category']

    def test_source_diversity(self, temp_db_with_data):
        """Test source diversity analysis."""
        monitor = DataQualityMonitor(temp_db_with_data)

        diversity = monitor.source_analyzer.analyze_source_distribution()

        assert diversity['unique_sources'] >= 2
        assert diversity['diversity_score'] >= 0
        assert diversity['diversity_score'] <= 100

    def test_freshness_tracking(self, temp_db_with_data):
        """Test freshness tracking."""
        monitor = DataQualityMonitor(temp_db_with_data)

        freshness = monitor.freshness_tracker.analyze_freshness()

        assert freshness['total_factors'] == 3
        assert freshness['distribution']['stale_count'] >= 1
        assert len(freshness['stale_factors']) >= 1


class TestWorkflow:
    """Test update workflow."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        from greenlang.db.emission_factors_schema import create_database
        create_database(db_path, overwrite=True)

        # Add sample factor
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ('fuel_diesel', 'Diesel', 'fuels', 'diesel', 2.68, 'liter',
              'Scope 1', 'EPA', 'https://epa.gov', '2024-01-01'))
        conn.commit()
        conn.close()

        yield db_path

        Path(db_path).unlink(missing_ok=True)
        Path(f"{db_path}.versions.db").unlink(missing_ok=True)

    def test_submit_change_request(self, temp_db):
        """Test change request submission."""
        approval_mgr = ApprovalManager(f"{temp_db}.versions.db")
        workflow = UpdateWorkflow(temp_db, approval_mgr)

        request = workflow.submit_change_request(
            factor_id='fuel_diesel',
            change_type=ChangeType.VALUE_CHANGED,
            proposed_changes={'emission_factor_value': 2.70},
            change_reason='Updated based on new data',
            requested_by='test_user'
        )

        assert request.request_id
        assert request.factor_id == 'fuel_diesel'
        assert request.review_status == ReviewStatus.PENDING_REVIEW

    def test_approve_request(self, temp_db):
        """Test request approval."""
        approval_mgr = ApprovalManager(f"{temp_db}.versions.db")
        workflow = UpdateWorkflow(temp_db, approval_mgr)

        # Submit request
        request = workflow.submit_change_request(
            factor_id='fuel_diesel',
            change_type=ChangeType.VALUE_CHANGED,
            proposed_changes={'emission_factor_value': 2.70},
            change_reason='Test',
            requested_by='test_user'
        )

        # Approve
        approval_mgr.approve_request(
            request.request_id,
            'reviewer',
            'Approved for testing'
        )

        # Verify approval
        conn = sqlite3.connect(f"{temp_db}.versions.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT approved FROM change_requests WHERE request_id = ?",
            (request.request_id,)
        )
        approved = cursor.fetchone()[0]
        conn.close()

        assert approved == 1


def test_integration_full_pipeline():
    """Integration test of full pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'test.db'

        # Create database
        from greenlang.db.emission_factors_schema import create_database
        create_database(str(db_path), overwrite=True)

        # Create sample YAML
        yaml_path = Path(tmpdir) / 'test.yaml'
        data = {
            'fuels': {
                'diesel': {
                    'name': 'Diesel',
                    'emission_factor_kg_co2e_per_liter': 2.68,
                    'scope': 'Scope 1',
                    'source': 'EPA',
                    'uri': 'https://www.epa.gov',
                    'last_updated': '2024-01-01'
                }
            }
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

        # Run import
        pipeline = AutomatedImportPipeline(str(db_path))
        job = ImportJob(
            job_id='integration_test',
            job_name='Integration Test',
            source_files=[str(yaml_path)],
            target_database=str(db_path),
            triggered_by='test',
            trigger_type='manual'
        )

        result = asyncio.run(pipeline.execute_import(job, validate_before=False))

        assert result.status == ImportStatus.COMPLETED
        assert result.successful_imports == 1

        # Verify data in database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM emission_factors")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
