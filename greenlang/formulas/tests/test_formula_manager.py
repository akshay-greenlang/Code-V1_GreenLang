"""
Unit tests for FormulaManager.

Tests cover:
- Formula creation and retrieval
- Version management
- Formula execution
- Version activation and rollback
- Dependency management
"""

import pytest
import tempfile
from pathlib import Path
from datetime import date

from greenlang.formulas import FormulaManager
from greenlang.formulas.models import FormulaCategory, VersionStatus
from greenlang.exceptions import ValidationError, ProcessingError


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def manager(temp_db):
    """Create FormulaManager instance."""
    return FormulaManager(temp_db)


class TestFormulaCreation:
    """Test formula creation and retrieval."""

    def test_create_formula(self, manager):
        """Test creating a new formula."""
        formula_id = manager.create_formula(
            formula_code="TEST-001",
            formula_name="Test Formula",
            category=FormulaCategory.EMISSIONS,
            description="Test description",
            standard_reference="TEST-STD",
            created_by="test_user",
        )

        assert formula_id > 0

        # Retrieve formula
        formula = manager.get_formula("TEST-001")
        assert formula is not None
        assert formula.formula_code == "TEST-001"
        assert formula.formula_name == "Test Formula"
        assert formula.category == FormulaCategory.EMISSIONS

    def test_create_duplicate_formula_fails(self, manager):
        """Test that creating duplicate formula raises error."""
        manager.create_formula(
            formula_code="TEST-001",
            formula_name="Test Formula",
            category=FormulaCategory.EMISSIONS,
        )

        with pytest.raises(ValidationError):
            manager.create_formula(
                formula_code="TEST-001",
                formula_name="Duplicate",
                category=FormulaCategory.EMISSIONS,
            )

    def test_list_formulas(self, manager):
        """Test listing formulas."""
        # Create multiple formulas
        manager.create_formula(
            formula_code="TEST-001",
            formula_name="Test 1",
            category=FormulaCategory.EMISSIONS,
        )
        manager.create_formula(
            formula_code="TEST-002",
            formula_name="Test 2",
            category=FormulaCategory.ENERGY,
        )

        # List all formulas
        all_formulas = manager.list_formulas()
        assert len(all_formulas) >= 2

        # List by category
        emissions_formulas = manager.list_formulas(category=FormulaCategory.EMISSIONS)
        assert len(emissions_formulas) >= 1


class TestVersionManagement:
    """Test formula version management."""

    def test_create_version(self, manager):
        """Test creating a formula version."""
        # Create formula first
        formula_id = manager.create_formula(
            formula_code="TEST-001",
            formula_name="Test Formula",
            category=FormulaCategory.EMISSIONS,
        )

        # Create version
        version_data = {
            'formula_expression': 'value1 + value2',
            'calculation_type': 'sum',
            'required_inputs': ['value1', 'value2'],
            'output_unit': 'tCO2e',
        }

        version_id = manager.create_new_version(
            formula_code="TEST-001",
            formula_data=version_data,
            change_notes="Initial version",
        )

        assert version_id > 0

        # Retrieve version
        version = manager.get_version("TEST-001", 1)
        assert version is not None
        assert version.version_number == 1
        assert version.formula_expression == 'value1 + value2'
        assert version.calculation_type == 'sum'
        assert version.required_inputs == ['value1', 'value2']

    def test_version_numbering(self, manager):
        """Test that version numbers increment correctly."""
        # Create formula
        manager.create_formula(
            formula_code="TEST-001",
            formula_name="Test",
            category=FormulaCategory.EMISSIONS,
        )

        # Create multiple versions
        version_data = {
            'formula_expression': 'value1 + value2',
            'calculation_type': 'sum',
            'required_inputs': ['value1', 'value2'],
            'output_unit': 'tCO2e',
        }

        v1_id = manager.create_new_version("TEST-001", version_data, "Version 1")
        v2_id = manager.create_new_version("TEST-001", version_data, "Version 2")
        v3_id = manager.create_new_version("TEST-001", version_data, "Version 3")

        # Check version numbers
        v1 = manager.get_version("TEST-001", 1)
        v2 = manager.get_version("TEST-001", 2)
        v3 = manager.get_version("TEST-001", 3)

        assert v1.version_number == 1
        assert v2.version_number == 2
        assert v3.version_number == 3

    def test_list_versions(self, manager):
        """Test listing all versions of a formula."""
        # Create formula
        manager.create_formula(
            formula_code="TEST-001",
            formula_name="Test",
            category=FormulaCategory.EMISSIONS,
        )

        # Create versions
        version_data = {
            'formula_expression': 'value1 + value2',
            'calculation_type': 'sum',
            'required_inputs': ['value1', 'value2'],
            'output_unit': 'tCO2e',
        }

        manager.create_new_version("TEST-001", version_data, "Version 1")
        manager.create_new_version("TEST-001", version_data, "Version 2")
        manager.create_new_version("TEST-001", version_data, "Version 3")

        # List versions
        versions = manager.list_versions("TEST-001")
        assert len(versions) == 3
        # Should be in descending order
        assert versions[0].version_number == 3
        assert versions[1].version_number == 2
        assert versions[2].version_number == 1


class TestVersionActivation:
    """Test version activation and status management."""

    def test_activate_version(self, manager):
        """Test activating a formula version."""
        # Create formula and version
        manager.create_formula(
            formula_code="TEST-001",
            formula_name="Test",
            category=FormulaCategory.EMISSIONS,
        )

        version_data = {
            'formula_expression': 'value1 + value2',
            'calculation_type': 'sum',
            'required_inputs': ['value1', 'value2'],
            'output_unit': 'tCO2e',
        }

        manager.create_new_version("TEST-001", version_data, "Initial")

        # Activate version
        manager.activate_version("TEST-001", 1)

        # Get active version
        active = manager.get_active_formula("TEST-001")
        assert active is not None
        assert active.version_number == 1
        assert active.version_status == VersionStatus.ACTIVE

    def test_activate_new_version_deactivates_old(self, manager):
        """Test that activating new version deactivates old one."""
        # Create formula
        manager.create_formula(
            formula_code="TEST-001",
            formula_name="Test",
            category=FormulaCategory.EMISSIONS,
        )

        version_data = {
            'formula_expression': 'value1 + value2',
            'calculation_type': 'sum',
            'required_inputs': ['value1', 'value2'],
            'output_unit': 'tCO2e',
        }

        # Create and activate v1
        manager.create_new_version("TEST-001", version_data, "Version 1", auto_activate=True)

        # Create and activate v2
        manager.create_new_version("TEST-001", version_data, "Version 2")
        manager.activate_version("TEST-001", 2)

        # Check v1 is deprecated
        v1 = manager.get_version("TEST-001", 1)
        assert v1.version_status == VersionStatus.DEPRECATED

        # Check v2 is active
        v2 = manager.get_version("TEST-001", 2)
        assert v2.version_status == VersionStatus.ACTIVE

        # Get active version
        active = manager.get_active_formula("TEST-001")
        assert active.version_number == 2


class TestRollback:
    """Test version rollback functionality."""

    def test_rollback_creates_new_version(self, manager):
        """Test that rollback creates a new version (not in-place edit)."""
        # Create formula
        manager.create_formula(
            formula_code="TEST-001",
            formula_name="Test",
            category=FormulaCategory.EMISSIONS,
        )

        # Create v1 with expression A
        v1_data = {
            'formula_expression': 'value1 + value2',
            'calculation_type': 'sum',
            'required_inputs': ['value1', 'value2'],
            'output_unit': 'tCO2e',
        }
        manager.create_new_version("TEST-001", v1_data, "Version 1", auto_activate=True)

        # Create v2 with expression B
        v2_data = {
            'formula_expression': 'value1 * value2',
            'calculation_type': 'multiplication',
            'required_inputs': ['value1', 'value2'],
            'output_unit': 'tCO2e',
        }
        manager.create_new_version("TEST-001", v2_data, "Version 2", auto_activate=True)

        # Rollback to v1
        new_version_id = manager.rollback_to_version("TEST-001", 1)

        # Should create v3
        v3 = manager.get_version("TEST-001", 3)
        assert v3 is not None
        assert v3.formula_expression == 'value1 + value2'  # Same as v1
        assert v3.version_status == VersionStatus.ACTIVE

        # v1 and v2 should still exist
        v1 = manager.get_version("TEST-001", 1)
        v2 = manager.get_version("TEST-001", 2)
        assert v1 is not None
        assert v2 is not None


class TestFormulaExecution:
    """Test formula execution."""

    def test_execute_sum_formula(self, manager):
        """Test executing a sum formula."""
        # Create formula
        manager.create_formula(
            formula_code="TEST-SUM",
            formula_name="Sum Test",
            category=FormulaCategory.EMISSIONS,
        )

        # Create version
        version_data = {
            'formula_expression': 'value1 + value2 + value3',
            'calculation_type': 'sum',
            'required_inputs': ['value1', 'value2', 'value3'],
            'output_unit': 'tCO2e',
        }
        manager.create_new_version("TEST-SUM", version_data, "Initial", auto_activate=True)

        # Execute formula
        result = manager.execute_formula(
            "TEST-SUM",
            {'value1': 10, 'value2': 20, 'value3': 30}
        )

        assert result == 60

    def test_execute_division_formula(self, manager):
        """Test executing a division formula."""
        # Create formula
        manager.create_formula(
            formula_code="TEST-DIV",
            formula_name="Division Test",
            category=FormulaCategory.EMISSIONS,
        )

        # Create version
        version_data = {
            'formula_expression': 'numerator / denominator',
            'calculation_type': 'division',
            'required_inputs': ['numerator', 'denominator'],
            'output_unit': 'ratio',
        }
        manager.create_new_version("TEST-DIV", version_data, "Initial", auto_activate=True)

        # Execute formula
        result = manager.execute_formula(
            "TEST-DIV",
            {'numerator': 100, 'denominator': 4}
        )

        assert result == 25

    def test_execute_percentage_formula(self, manager):
        """Test executing a percentage formula."""
        # Create formula
        manager.create_formula(
            formula_code="TEST-PCT",
            formula_name="Percentage Test",
            category=FormulaCategory.EFFICIENCY,
        )

        # Create version
        version_data = {
            'formula_expression': '(numerator / denominator) * 100',
            'calculation_type': 'percentage',
            'required_inputs': ['numerator', 'denominator'],
            'output_unit': '%',
        }
        manager.create_new_version("TEST-PCT", version_data, "Initial", auto_activate=True)

        # Execute formula
        result = manager.execute_formula(
            "TEST-PCT",
            {'numerator': 75, 'denominator': 100}
        )

        assert result == 75.0

    def test_execute_custom_expression(self, manager):
        """Test executing a custom expression formula."""
        # Create formula
        manager.create_formula(
            formula_code="TEST-CUSTOM",
            formula_name="Custom Expression Test",
            category=FormulaCategory.EMISSIONS,
        )

        # Create version
        version_data = {
            'formula_expression': 'value1 * 2 + value2 / 2',
            'calculation_type': 'custom_expression',
            'required_inputs': ['value1', 'value2'],
            'output_unit': 'tCO2e',
        }
        manager.create_new_version("TEST-CUSTOM", version_data, "Initial", auto_activate=True)

        # Execute formula
        result = manager.execute_formula(
            "TEST-CUSTOM",
            {'value1': 10, 'value2': 20}
        )

        assert result == 30  # 10*2 + 20/2 = 20 + 10 = 30

    def test_execute_with_missing_input_fails(self, manager):
        """Test that execution fails with missing inputs."""
        # Create formula
        manager.create_formula(
            formula_code="TEST-FAIL",
            formula_name="Fail Test",
            category=FormulaCategory.EMISSIONS,
        )

        # Create version
        version_data = {
            'formula_expression': 'value1 + value2',
            'calculation_type': 'sum',
            'required_inputs': ['value1', 'value2'],
            'output_unit': 'tCO2e',
        }
        manager.create_new_version("TEST-FAIL", version_data, "Initial", auto_activate=True)

        # Execute with missing input
        with pytest.raises(ValidationError):
            manager.execute_formula(
                "TEST-FAIL",
                {'value1': 10}  # Missing value2
            )

    def test_execute_full_returns_provenance(self, manager):
        """Test that execute_formula_full returns complete result."""
        # Create formula
        manager.create_formula(
            formula_code="TEST-PROV",
            formula_name="Provenance Test",
            category=FormulaCategory.EMISSIONS,
        )

        # Create version
        version_data = {
            'formula_expression': 'value1 + value2',
            'calculation_type': 'sum',
            'required_inputs': ['value1', 'value2'],
            'output_unit': 'tCO2e',
        }
        manager.create_new_version("TEST-PROV", version_data, "Initial", auto_activate=True)

        # Execute formula
        result = manager.execute_formula_full(
            "TEST-PROV",
            {'value1': 10, 'value2': 20},
            agent_name="test_agent",
        )

        assert result.output_value == 30
        assert result.input_hash is not None
        assert len(result.input_hash) == 64  # SHA-256
        assert result.output_hash is not None
        assert len(result.output_hash) == 64
        assert result.execution_time_ms > 0
        assert result.agent_name == "test_agent"


class TestVersionComparison:
    """Test formula version comparison."""

    def test_compare_versions(self, manager):
        """Test comparing two formula versions."""
        # Create formula
        manager.create_formula(
            formula_code="TEST-CMP",
            formula_name="Compare Test",
            category=FormulaCategory.EMISSIONS,
        )

        # Create v1
        v1_data = {
            'formula_expression': 'value1 + value2',
            'calculation_type': 'sum',
            'required_inputs': ['value1', 'value2'],
            'output_unit': 'tCO2e',
        }
        manager.create_new_version("TEST-CMP", v1_data, "Version 1")

        # Create v2 with different expression
        v2_data = {
            'formula_expression': 'value1 + value2 + value3',
            'calculation_type': 'sum',
            'required_inputs': ['value1', 'value2', 'value3'],
            'output_unit': 'kgCO2e',
        }
        manager.create_new_version("TEST-CMP", v2_data, "Version 2")

        # Compare
        comparison = manager.compare_versions("TEST-CMP", 1, 2)

        assert comparison.expression_changed is True
        assert comparison.inputs_changed is True
        assert comparison.added_inputs == ['value3']
        assert comparison.output_unit_changed is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
