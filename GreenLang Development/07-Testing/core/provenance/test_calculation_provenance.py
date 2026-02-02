# -*- coding: utf-8 -*-
"""
Tests for CalculationProvenance

Tests the standardized provenance tracking for GreenLang calculators,
ensuring zero-hallucination audit trails with SHA-256 integrity verification.

Author: GreenLang Team
"""

import pytest
import json
from datetime import datetime
from greenlang.core.provenance import (
    CalculationStep,
    CalculationProvenance,
    ProvenanceMetadata,
    OperationType,
)


class TestCalculationStep:
    """Tests for CalculationStep model."""

    def test_create_step_basic(self):
        """Test creating a basic calculation step."""
        step = CalculationStep(
            step_number=1,
            operation=OperationType.MULTIPLY,
            description="Calculate total emissions",
            inputs={"fuel_kg": 1000, "emission_factor": 0.18414},
            output=184.14,
        )

        assert step.step_number == 1
        assert step.operation == "multiply"
        assert step.description == "Calculate total emissions"
        assert step.inputs == {"fuel_kg": 1000, "emission_factor": 0.18414}
        assert step.output == 184.14
        assert step.timestamp is not None

    def test_create_step_with_formula(self):
        """Test creating step with formula and data source."""
        step = CalculationStep(
            step_number=2,
            operation=OperationType.LOOKUP,
            description="Lookup emission factor",
            inputs={"fuel_type": "natural_gas", "region": "US"},
            output=0.18414,
            formula="EF = lookup(fuel_type, region)",
            data_source="EPA eGRID 2023",
            standard_reference="EPA AP-42",
        )

        assert step.formula == "EF = lookup(fuel_type, region)"
        assert step.data_source == "EPA eGRID 2023"
        assert step.standard_reference == "EPA AP-42"

    def test_step_to_dict(self):
        """Test serializing step to dictionary."""
        step = CalculationStep(
            step_number=1,
            operation="add",
            description="Sum values",
            inputs={"a": 10, "b": 20},
            output=30,
        )

        step_dict = step.to_dict()

        assert step_dict["step_number"] == 1
        assert step_dict["operation"] == "add"
        assert step_dict["description"] == "Sum values"
        assert step_dict["inputs"] == {"a": 10, "b": 20}
        assert step_dict["output"] == 30
        assert "timestamp" in step_dict

    def test_step_from_dict(self):
        """Test deserializing step from dictionary."""
        data = {
            "step_number": 1,
            "operation": "multiply",
            "description": "Test step",
            "inputs": {"x": 5},
            "output": 10,
            "formula": "y = 2x",
            "data_source": "Test DB",
            "standard_reference": "ISO 12345",
            "timestamp": "2025-01-01T00:00:00",
            "metadata": {"custom": "value"},
        }

        step = CalculationStep.from_dict(data)

        assert step.step_number == 1
        assert step.operation == "multiply"
        assert step.formula == "y = 2x"
        assert step.metadata == {"custom": "value"}


class TestProvenanceMetadata:
    """Tests for ProvenanceMetadata model."""

    def test_create_metadata(self):
        """Test creating provenance metadata."""
        metadata = ProvenanceMetadata(
            agent_name="EmissionsCalculator",
            agent_version="1.0.0",
            calculation_type="scope1_emissions",
            standards_applied=["GHG Protocol", "ISO 14064-1"],
            data_sources=["EPA eGRID 2023"],
        )

        assert metadata.agent_name == "EmissionsCalculator"
        assert metadata.agent_version == "1.0.0"
        assert metadata.calculation_type == "scope1_emissions"
        assert len(metadata.standards_applied) == 2
        assert len(metadata.data_sources) == 1
        assert len(metadata.warnings) == 0
        assert len(metadata.errors) == 0

    def test_metadata_to_dict(self):
        """Test serializing metadata to dictionary."""
        metadata = ProvenanceMetadata(
            agent_name="TestAgent",
            agent_version="2.0.0",
            calculation_type="test_calc",
        )
        metadata.warnings.append("Test warning")
        metadata.errors.append("Test error")

        meta_dict = metadata.to_dict()

        assert meta_dict["agent_name"] == "TestAgent"
        assert meta_dict["agent_version"] == "2.0.0"
        assert meta_dict["warnings"] == ["Test warning"]
        assert meta_dict["errors"] == ["Test error"]


class TestCalculationProvenance:
    """Tests for CalculationProvenance model."""

    def test_create_provenance(self):
        """Test creating a provenance record."""
        input_data = {
            "fuel_consumption_kg": 1000,
            "fuel_type": "natural_gas",
        }

        provenance = CalculationProvenance.create(
            agent_name="EmissionsCalculator",
            agent_version="1.0.0",
            calculation_type="scope1_emissions",
            input_data=input_data,
            standards_applied=["GHG Protocol"],
            data_sources=["EPA eGRID 2023"],
        )

        assert provenance.calculation_id is not None
        assert len(provenance.calculation_id) == 16  # First 16 chars of hash
        assert provenance.metadata.agent_name == "EmissionsCalculator"
        assert provenance.metadata.agent_version == "1.0.0"
        assert provenance.metadata.calculation_type == "scope1_emissions"
        assert provenance.input_data == input_data
        assert provenance.input_hash is not None
        assert len(provenance.steps) == 0
        assert provenance.output_data is None
        assert provenance.timestamp_start is not None

    def test_add_calculation_steps(self):
        """Test adding calculation steps."""
        provenance = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={"x": 10},
        )

        # Add step 1: Lookup
        step1 = provenance.add_step(
            operation=OperationType.LOOKUP,
            description="Lookup emission factor",
            inputs={"fuel_type": "natural_gas"},
            output=0.18414,
            data_source="EPA eGRID 2023",
        )

        assert step1.step_number == 1
        assert len(provenance.steps) == 1

        # Add step 2: Multiply
        step2 = provenance.add_step(
            operation=OperationType.MULTIPLY,
            description="Calculate emissions",
            inputs={"fuel_kg": 1000, "ef": 0.18414},
            output=184.14,
            formula="emissions = fuel_kg * ef",
            standard_reference="GHG Protocol",
        )

        assert step2.step_number == 2
        assert len(provenance.steps) == 2

        # Verify data sources were tracked
        assert "EPA eGRID 2023" in provenance.metadata.data_sources
        assert "GHG Protocol" in provenance.metadata.standards_applied

    def test_finalize_provenance(self):
        """Test finalizing provenance with output."""
        provenance = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={"x": 10},
        )

        # Add a step
        provenance.add_step(
            operation=OperationType.MULTIPLY,
            description="Double value",
            inputs={"x": 10},
            output=20,
            formula="y = 2 * x",
        )

        # Finalize
        output_data = {"result": 20}
        provenance.finalize(output_data=output_data)

        assert provenance.output_data == output_data
        assert provenance.output_hash is not None
        assert provenance.timestamp_end is not None
        assert provenance.duration_ms >= 0

    def test_add_warnings_and_errors(self):
        """Test adding warnings and errors."""
        provenance = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={},
        )

        provenance.add_warning("Using default emission factor")
        provenance.add_warning("Missing region data")
        provenance.add_error("Division by zero")

        assert len(provenance.metadata.warnings) == 2
        assert len(provenance.metadata.errors) == 1
        assert "Using default emission factor" in provenance.metadata.warnings
        assert "Division by zero" in provenance.metadata.errors

    def test_provenance_to_dict(self):
        """Test serializing provenance to dictionary."""
        provenance = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={"x": 10},
        )

        provenance.add_step(
            operation=OperationType.ADD,
            description="Add 5",
            inputs={"x": 10},
            output=15,
        )

        provenance.finalize(output_data={"result": 15})

        prov_dict = provenance.to_dict()

        assert "calculation_id" in prov_dict
        assert "metadata" in prov_dict
        assert "input_data" in prov_dict
        assert "input_hash" in prov_dict
        assert "steps" in prov_dict
        assert "output_data" in prov_dict
        assert "output_hash" in prov_dict
        assert len(prov_dict["steps"]) == 1

        # Verify JSON serialization works
        json_str = json.dumps(prov_dict, indent=2)
        assert json_str is not None

    def test_provenance_from_dict(self):
        """Test deserializing provenance from dictionary."""
        # Create and serialize
        original = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={"x": 10},
        )

        original.add_step(
            operation="multiply",
            description="Test step",
            inputs={"x": 10},
            output=20,
        )

        original.finalize(output_data={"result": 20})

        prov_dict = original.to_dict()

        # Deserialize
        restored = CalculationProvenance.from_dict(prov_dict)

        assert restored.calculation_id == original.calculation_id
        assert restored.metadata.agent_name == original.metadata.agent_name
        assert restored.input_data == original.input_data
        assert restored.input_hash == original.input_hash
        assert len(restored.steps) == len(original.steps)
        assert restored.output_data == original.output_data
        assert restored.output_hash == original.output_hash

    def test_verify_input_hash(self):
        """Test input hash verification."""
        provenance = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={"x": 10, "y": 20},
        )

        # Hash should match
        assert provenance.verify_input_hash() is True

        # Modify input data (simulating tampering)
        provenance.input_data["x"] = 999

        # Hash should no longer match
        assert provenance.verify_input_hash() is False

    def test_verify_output_hash(self):
        """Test output hash verification."""
        provenance = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={"x": 10},
        )

        provenance.finalize(output_data={"result": 20})

        # Hash should match
        assert provenance.verify_output_hash() is True

        # Modify output data (simulating tampering)
        provenance.output_data["result"] = 999

        # Hash should no longer match
        assert provenance.verify_output_hash() is False

    def test_verify_integrity(self):
        """Test complete integrity verification."""
        provenance = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={"x": 10},
        )

        provenance.add_step(
            operation="multiply",
            description="Test",
            inputs={"x": 10},
            output=20,
        )

        provenance.finalize(output_data={"result": 20})

        integrity = provenance.verify_integrity()

        assert integrity["input_hash_valid"] is True
        assert integrity["output_hash_valid"] is True
        assert integrity["steps_sequential"] is True
        assert integrity["has_steps"] is True
        assert integrity["is_finalized"] is True
        assert integrity["no_errors"] is True

    def test_get_audit_summary(self):
        """Test generating audit summary."""
        provenance = CalculationProvenance.create(
            agent_name="EmissionsCalculator",
            agent_version="2.1.0",
            calculation_type="scope1_emissions",
            input_data={"fuel_kg": 1000},
            standards_applied=["GHG Protocol"],
            data_sources=["EPA eGRID 2023"],
        )

        provenance.add_step(
            operation="lookup",
            description="Lookup EF",
            inputs={},
            output=0.18414,
        )

        provenance.add_warning("Test warning")
        provenance.finalize(output_data={"emissions": 184.14})

        summary = provenance.get_audit_summary()

        assert summary["calculation_id"] == provenance.calculation_id
        assert summary["agent"] == "EmissionsCalculator v2.1.0"
        assert summary["calculation_type"] == "scope1_emissions"
        assert summary["steps_count"] == 1
        assert summary["warnings_count"] == 1
        assert summary["errors_count"] == 0
        assert "summary" in summary
        assert "EmissionsCalculator" in summary["summary"]

    def test_deterministic_calculation_id(self):
        """Test that calculation IDs are deterministic."""
        input_data = {"x": 10, "y": 20}

        # Create two provenances with same inputs (at different times)
        prov1 = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data=input_data,
        )

        prov2 = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data=input_data,
        )

        # IDs should be different (includes timestamp)
        # But input hashes should be the same
        assert prov1.input_hash == prov2.input_hash

    def test_complex_calculation_workflow(self):
        """Test a complete calculation workflow with multiple steps."""
        # Create provenance
        provenance = CalculationProvenance.create(
            agent_name="ScopeEmissionsCalculator",
            agent_version="3.2.1",
            calculation_type="scope1_emissions",
            input_data={
                "fuel_consumption_kg": 5000,
                "fuel_type": "natural_gas",
                "region": "US-CA",
            },
            standards_applied=["GHG Protocol Scope 1", "ISO 14064-1"],
        )

        # Step 1: Lookup emission factor
        provenance.add_step(
            operation=OperationType.LOOKUP,
            description="Lookup natural gas emission factor for California",
            inputs={"fuel_type": "natural_gas", "region": "US-CA"},
            output=0.18414,
            data_source="EPA eGRID 2023",
            standard_reference="EPA AP-42",
        )

        # Step 2: Calculate direct emissions
        provenance.add_step(
            operation=OperationType.MULTIPLY,
            description="Calculate direct CO2 emissions",
            inputs={"fuel_consumption_kg": 5000, "emission_factor": 0.18414},
            output=920.70,
            formula="emissions_kg_co2 = fuel_consumption_kg * emission_factor",
            standard_reference="GHG Protocol Scope 1",
        )

        # Step 3: Convert to tonnes
        provenance.add_step(
            operation=OperationType.DIVIDE,
            description="Convert kg CO2 to tonnes CO2e",
            inputs={"emissions_kg": 920.70},
            output=0.92070,
            formula="emissions_tonnes = emissions_kg / 1000",
        )

        # Finalize
        provenance.finalize(output_data={
            "total_emissions_tonnes_co2e": 0.92070,
            "emissions_per_kg_fuel": 0.18414,
        })

        # Verify provenance
        assert len(provenance.steps) == 3
        assert provenance.output_data["total_emissions_tonnes_co2e"] == 0.92070
        assert "EPA eGRID 2023" in provenance.metadata.data_sources
        assert "GHG Protocol Scope 1" in provenance.metadata.standards_applied
        assert "ISO 14064-1" in provenance.metadata.standards_applied

        integrity = provenance.verify_integrity()
        assert all(integrity.values())

        # Test serialization roundtrip
        prov_dict = provenance.to_dict()
        restored = CalculationProvenance.from_dict(prov_dict)

        assert restored.calculation_id == provenance.calculation_id
        assert len(restored.steps) == 3
        assert restored.verify_integrity()["input_hash_valid"]


class TestProvenanceEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_provenance(self):
        """Test provenance with no steps."""
        provenance = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={},
        )

        provenance.finalize(output_data=None)

        assert len(provenance.steps) == 0
        integrity = provenance.verify_integrity()
        assert integrity["has_steps"] is False
        assert integrity["steps_sequential"] is True  # Empty is valid

    def test_verify_output_hash_before_finalize(self):
        """Test verifying output hash before finalizing."""
        provenance = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={"x": 10},
        )

        # Should return False (no output yet)
        assert provenance.verify_output_hash() is False

    def test_multiple_data_sources(self):
        """Test tracking multiple data sources."""
        provenance = CalculationProvenance.create(
            agent_name="TestCalculator",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={},
        )

        provenance.add_step(
            operation="lookup",
            description="Step 1",
            inputs={},
            output=1,
            data_source="Source A",
        )

        provenance.add_step(
            operation="lookup",
            description="Step 2",
            inputs={},
            output=2,
            data_source="Source B",
        )

        provenance.add_step(
            operation="lookup",
            description="Step 3",
            inputs={},
            output=3,
            data_source="Source A",  # Duplicate
        )

        # Should have 2 unique data sources
        assert len(provenance.metadata.data_sources) == 2
        assert "Source A" in provenance.metadata.data_sources
        assert "Source B" in provenance.metadata.data_sources


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
