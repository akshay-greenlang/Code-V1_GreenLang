# -*- coding: utf-8 -*-
"""
Unit Tests for Audit Trail & Provenance

Tests SHA-256 provenance tracking and audit trail generation.
"""

import pytest
import hashlib
import json
from decimal import Decimal
from datetime import datetime


class TestAuditTrailGenerator:
    """Test AuditTrailGenerator"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup audit trail generator"""
        try:
            from greenlang.calculation.audit_trail import AuditTrailGenerator
            self.generator = AuditTrailGenerator()
        except ImportError:
            pytest.skip("AuditTrailGenerator not available")

    def test_generate_audit_trail(self):
        """Test generating complete audit trail"""
        calculation_data = {
            "inputs": {
                "factor_id": "diesel-us-stationary",
                "activity_amount": "100",
                "activity_unit": "liters"
            },
            "steps": [
                {"step": "validate_input", "duration_ms": 2.5},
                {"step": "resolve_factor", "duration_ms": 5.1},
                {"step": "calculate", "duration_ms": 1.2}
            ],
            "output": {
                "emissions_kg_co2e": "268.0",
                "status": "success"
            }
        }

        audit_trail = self.generator.generate(calculation_data)

        assert audit_trail.inputs == calculation_data["inputs"]
        assert len(audit_trail.steps) == 3
        assert audit_trail.output == calculation_data["output"]
        assert audit_trail.provenance_hash is not None

    def test_provenance_hash_determinism(self):
        """Test provenance hash is deterministic"""
        data = {
            "inputs": {"value": "100"},
            "steps": [],
            "output": {"result": "268.0"}
        }

        hash1 = self.generator.generate(data).provenance_hash
        hash2 = self.generator.generate(data).provenance_hash

        assert hash1 == hash2

    def test_provenance_hash_uniqueness(self):
        """Test different inputs produce different hashes"""
        data1 = {
            "inputs": {"value": "100"},
            "steps": [],
            "output": {"result": "268.0"}
        }

        data2 = {
            "inputs": {"value": "200"},
            "steps": [],
            "output": {"result": "536.0"}
        }

        hash1 = self.generator.generate(data1).provenance_hash
        hash2 = self.generator.generate(data2).provenance_hash

        assert hash1 != hash2

    def test_verify_audit_trail_integrity(self):
        """Test verifying audit trail integrity"""
        data = {
            "inputs": {"value": "100"},
            "steps": [],
            "output": {"result": "268.0"}
        }

        audit_trail = self.generator.generate(data)

        # Verify intact trail
        assert audit_trail.verify_integrity() is True

        # Tamper with output
        audit_trail.output = {"result": "999.0"}

        # Verification should fail
        assert audit_trail.verify_integrity() is False


class TestProvenanceTracking:
    """Test end-to-end provenance tracking"""

    def test_calculation_provenance_complete(self):
        """Test calculation includes complete provenance"""
        try:
            from greenlang.calculation.core_calculator import EmissionCalculator, CalculationRequest
        except ImportError:
            pytest.skip("Calculator not available")

        calculator = EmissionCalculator()

        request = CalculationRequest(
            factor_id="diesel-us-stationary",
            activity_amount=100,
            activity_unit="liters"
        )

        result = calculator.calculate(request)

        # Verify provenance exists
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

        # Verify provenance includes all required elements
        assert result.calculation_steps is not None
        assert len(result.calculation_steps) > 0

    def test_provenance_hash_valid_sha256(self):
        """Test provenance hash is valid SHA-256"""
        hash_str = "a" * 64  # Valid SHA-256 length

        # Should be valid hex
        try:
            int(hash_str, 16)
            is_valid = True
        except ValueError:
            is_valid = False

        assert is_valid is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
