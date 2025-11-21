# -*- coding: utf-8 -*-
"""
Example unit tests for a GreenLang agent.

Demonstrates the testing patterns and best practices for agent testing.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from test_base import BaseAgentTest
from test_data_generator import TestDataGenerator

# Example agent imports (these would be your actual agents)
from greenlang_core import Agent, AgentInput, AgentOutput, AgentConfig
from greenlang_core.exceptions import ValidationError, ProcessingError


# Example Agent Classes (would be in your actual codebase)
class EmissionCalculatorInput(AgentInput):
    """Input for emission calculation."""
    fuel_type: str
    fuel_quantity: float
    region: str
    combustion_type: str = "stationary"


class EmissionCalculatorOutput(AgentOutput):
    """Output from emission calculation."""
    total_emissions: float
    emission_factor: float
    calculation_method: str
    uncertainty: float


class EmissionCalculatorAgent(Agent):
    """Agent for calculating emissions from fuel consumption."""

    def process(self, input_data: EmissionCalculatorInput) -> EmissionCalculatorOutput:
        """Process emission calculation."""
        # Validation
        if input_data.fuel_quantity <= 0:
            raise ValidationError("Fuel quantity must be positive")

        # Lookup emission factor
        emission_factor = self._lookup_emission_factor(
            input_data.fuel_type,
            input_data.region,
            input_data.combustion_type
        )

        # Calculate emissions
        total_emissions = input_data.fuel_quantity * emission_factor

        # Create output with provenance
        return EmissionCalculatorOutput(
            total_emissions=total_emissions,
            emission_factor=emission_factor,
            calculation_method="IPCC Tier 1",
            uncertainty=0.1,
            status="SUCCESS",
            provenance_hash=self._calculate_provenance_hash(input_data, total_emissions)
        )

    def _lookup_emission_factor(self, fuel_type: str, region: str, combustion_type: str) -> float:
        """Lookup emission factor from database."""
        # This would connect to actual database
        factors = {
            ("diesel", "US", "stationary"): 2.68,
            ("natural_gas", "US", "stationary"): 1.93,
            ("coal", "US", "stationary"): 3.45
        }
        return factors.get((fuel_type, region, combustion_type), 2.0)

    def _calculate_provenance_hash(self, input_data: Any, result: Any) -> str:
        """Calculate provenance hash."""
        import hashlib
        import json

        data = {
            "input": input_data.dict(),
            "result": str(result),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": self.__class__.__name__
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


# Test Class
class TestEmissionCalculatorAgent(BaseAgentTest):
    """Test suite for EmissionCalculatorAgent."""

    agent_class = EmissionCalculatorAgent
    input_class = EmissionCalculatorInput
    output_class = EmissionCalculatorOutput

    @pytest.fixture
    def data_generator(self):
        """Create test data generator."""
        return TestDataGenerator(seed=42)

    def get_valid_input(self) -> EmissionCalculatorInput:
        """Get valid input for testing."""
        return EmissionCalculatorInput(
            fuel_type="diesel",
            fuel_quantity=1000.0,
            region="US",
            combustion_type="stationary"
        )

    def get_invalid_input(self) -> EmissionCalculatorInput:
        """Get invalid input for testing."""
        return EmissionCalculatorInput(
            fuel_type="diesel",
            fuel_quantity=-100.0,  # Invalid: negative quantity
            region="US",
            combustion_type="stationary"
        )

    @pytest.mark.parametrize("fuel_type,quantity,region,expected_emissions", [
        ("diesel", 1000.0, "US", 2680.0),
        ("natural_gas", 500.0, "US", 965.0),
        ("coal", 100.0, "US", 345.0),
    ])
    def test_emission_calculations(
        self,
        agent_config,
        fuel_type,
        quantity,
        region,
        expected_emissions
    ):
        """Test emission calculations with known values."""
        agent = self.agent_class(agent_config)

        input_data = EmissionCalculatorInput(
            fuel_type=fuel_type,
            fuel_quantity=quantity,
            region=region,
            combustion_type="stationary"
        )

        result = agent.process(input_data)

        assert result.total_emissions == pytest.approx(expected_emissions, rel=1e-6)
        assert result.calculation_method == "IPCC Tier 1"
        assert result.status == "SUCCESS"

    def test_missing_emission_factor(self, agent_config):
        """Test handling of missing emission factors."""
        agent = self.agent_class(agent_config)

        input_data = EmissionCalculatorInput(
            fuel_type="unknown_fuel",
            fuel_quantity=100.0,
            region="UNKNOWN",
            combustion_type="stationary"
        )

        result = agent.process(input_data)

        # Should use default factor
        assert result.emission_factor == 2.0
        assert result.total_emissions == 200.0

    def test_large_quantity_handling(self, agent_config):
        """Test handling of large quantities."""
        agent = self.agent_class(agent_config)

        input_data = EmissionCalculatorInput(
            fuel_type="diesel",
            fuel_quantity=1e10,  # Very large quantity
            region="US",
            combustion_type="stationary"
        )

        result = agent.process(input_data)

        assert result.total_emissions == 1e10 * 2.68
        assert result.status == "SUCCESS"

    def test_decimal_precision(self, agent_config):
        """Test decimal precision in calculations."""
        agent = self.agent_class(agent_config)

        input_data = EmissionCalculatorInput(
            fuel_type="diesel",
            fuel_quantity=333.333333,
            region="US",
            combustion_type="stationary"
        )

        result = agent.process(input_data)

        expected = 333.333333 * 2.68
        assert abs(result.total_emissions - expected) < 0.0001

    @pytest.mark.asyncio
    async def test_async_emission_lookup(self, agent_config):
        """Test async emission factor lookup."""
        agent = self.agent_class(agent_config)

        # Mock async database lookup
        with patch.object(agent, '_lookup_emission_factor') as mock_lookup:
            mock_lookup.return_value = 2.5

            input_data = self.get_valid_input()
            result = agent.process(input_data)

            assert result.emission_factor == 2.5

    def test_concurrent_processing(self, agent_config):
        """Test concurrent processing of multiple inputs."""
        import concurrent.futures

        agent = self.agent_class(agent_config)
        inputs = [self.get_valid_input() for _ in range(10)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(agent.process, inputs))

        assert len(results) == 10
        assert all(r.status == "SUCCESS" for r in results)

    @pytest.mark.performance
    def test_throughput(self, agent_config, data_generator):
        """Test agent throughput."""
        agent = self.agent_class(agent_config)

        # Generate test data
        test_data = []
        for _ in range(1000):
            fuel_data = data_generator.generate_fuel_consumption()
            for fuel in fuel_data["fuel_data"]:
                test_data.append(EmissionCalculatorInput(
                    fuel_type=fuel["fuel_type"],
                    fuel_quantity=fuel["quantity"],
                    region=fuel_data["location"],
                    combustion_type=fuel["combustion_type"]
                ))

        start_time = datetime.now(timezone.utc)
        results = [agent.process(input_data) for input_data in test_data]
        end_time = datetime.now(timezone.utc)

        duration = (end_time - start_time).total_seconds()
        throughput = len(test_data) / duration

        assert throughput > 100  # Target: >100 records/second
        assert all(r.status == "SUCCESS" for r in results)

    @pytest.mark.compliance
    def test_audit_trail(self, agent_config):
        """Test audit trail generation."""
        agent = self.agent_class(agent_config)
        input_data = self.get_valid_input()

        result = agent.process(input_data)

        # Verify audit trail components
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hash

        # Verify reproducibility
        result2 = agent.process(input_data)
        assert result.provenance_hash == result2.provenance_hash

    def test_error_messages(self, agent_config):
        """Test error message clarity."""
        agent = self.agent_class(agent_config)

        invalid_inputs = [
            (EmissionCalculatorInput(fuel_type="", fuel_quantity=100, region="US"), "fuel_type"),
            (EmissionCalculatorInput(fuel_type="diesel", fuel_quantity=0, region="US"), "quantity"),
            (EmissionCalculatorInput(fuel_type="diesel", fuel_quantity=100, region=""), "region"),
        ]

        for invalid_input, expected_field in invalid_inputs:
            with pytest.raises(ValidationError) as exc_info:
                agent.process(invalid_input)

            # Verify error message mentions the problematic field
            assert expected_field.lower() in str(exc_info.value).lower()

    @pytest.mark.integration
    def test_database_integration(self, agent_config, db_session):
        """Test database integration."""
        agent = self.agent_class(agent_config)

        # Mock database connection
        with patch.object(agent, 'db_session', db_session):
            input_data = self.get_valid_input()
            result = agent.process(input_data)

            assert result.status == "SUCCESS"

    @pytest.mark.security
    def test_input_sanitization(self, agent_config, security_scanner):
        """Test input sanitization against injection attacks."""
        agent = self.agent_class(agent_config)

        malicious_inputs = [
            "diesel'; DROP TABLE emissions; --",
            "<script>alert('XSS')</script>",
            "../../etc/passwd"
        ]

        for malicious_value in malicious_inputs:
            input_data = EmissionCalculatorInput(
                fuel_type=malicious_value,
                fuel_quantity=100.0,
                region="US",
                combustion_type="stationary"
            )

            # Should handle malicious input safely
            try:
                result = agent.process(input_data)
                # If it processes, verify no injection occurred
                assert security_scanner.scan_sql_injection(str(result))
            except ValidationError:
                # Validation error is acceptable for malicious input
                pass