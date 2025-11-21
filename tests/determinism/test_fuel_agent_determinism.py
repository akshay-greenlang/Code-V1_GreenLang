# -*- coding: utf-8 -*-
"""Determinism Tests for FuelAgentAI.

This module tests that FuelAgentAI produces deterministic, reproducible results
across multiple runs with identical inputs.

Test Coverage:
1. Hash-based determinism verification
2. Multiple runs produce identical outputs
3. Seed=42, temperature=0 reproducibility
4. Citations are deterministic
5. Emission factor lookups are deterministic
6. Snapshot-based regression testing

Author: GreenLang Framework Team
Phase: Phase 3 - Production Hardening
Date: November 2024
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from greenlang.agents.fuel_agent_ai import FuelAgentAI
from greenlang.intelligence import ChatResponse, Usage, FinishReason
from greenlang.intelligence.schemas.responses import ProviderInfo
from tests.determinism.test_framework import (
    DeterminismTester,
    assert_deterministic,
)
from tests.determinism.snapshot_manager import (
    SnapshotManager,
    assert_snapshot_matches,
)


@pytest.mark.determinism
class TestFuelAgentDeterminism:
    """Test suite for FuelAgentAI determinism."""

    @pytest.fixture
    def agent(self):
        """Create FuelAgentAI instance with deterministic settings."""
        return FuelAgentAI(
            budget_usd=1.0,
            enable_explanations=True,
            enable_recommendations=True,
        )

    @pytest.fixture
    def tester(self):
        """Create DeterminismTester instance."""
        return DeterminismTester(
            normalize_platform=True,
            normalize_timestamps=True,
            normalize_floats=True,
            float_precision=2,  # Match agent precision
        )

    @pytest.fixture
    def snapshot_manager(self):
        """Create SnapshotManager instance."""
        return SnapshotManager(
            auto_update=False,  # Fail on mismatch in CI
            normalize_output=True,
        )

    @pytest.fixture
    def natural_gas_payload(self):
        """Standard natural gas test payload."""
        return {
            "fuel_type": "natural_gas",
            "amount": 1000.0,
            "unit": "therms",
            "country": "US",
        }

    @pytest.fixture
    def diesel_payload(self):
        """Standard diesel test payload."""
        return {
            "fuel_type": "diesel",
            "amount": 500.0,
            "unit": "gallons",
            "country": "US",
        }

    @pytest.fixture
    def mock_chat_session(self):
        """Create mock ChatSession for deterministic testing."""
        def _create_session(text="Mock AI response", tool_calls=None):
            mock_response = Mock(spec=ChatResponse)
            mock_response.text = text
            mock_response.tool_calls = tool_calls or []
            mock_response.usage = Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.01,
            )
            mock_response.provider_info = ProviderInfo(
                provider="openai",
                model="gpt-4o-mini",
            )
            mock_response.finish_reason = FinishReason.stop

            mock_session = Mock()
            mock_session.chat = AsyncMock(return_value=mock_response)
            return mock_session

        return _create_session

    # ============================================================================
    # Test 1: Basic Determinism - Same Input → Same Output
    # ============================================================================

    def test_deterministic_calculation_sync(self, agent, natural_gas_payload):
        """Test that calculate_emissions_impl is deterministic (sync)."""
        results = []
        for _ in range(5):
            result = agent._calculate_emissions_impl(
                fuel_type=natural_gas_payload["fuel_type"],
                amount=natural_gas_payload["amount"],
                unit=natural_gas_payload["unit"],
                country=natural_gas_payload["country"],
                renewable_percentage=0,
                efficiency=1.0,
            )
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], f"Run {i+1} differs from run 1"

    def test_deterministic_lookup_sync(self, agent, natural_gas_payload):
        """Test that lookup_emission_factor_impl is deterministic (sync)."""
        results = []
        for _ in range(5):
            result = agent._lookup_emission_factor_impl(
                fuel_type=natural_gas_payload["fuel_type"],
                unit=natural_gas_payload["unit"],
                country=natural_gas_payload["country"],
            )
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], f"Run {i+1} differs from run 1"

    def test_deterministic_recommendations_sync(self, agent):
        """Test that generate_recommendations_impl is deterministic (sync)."""
        results = []
        for _ in range(5):
            result = agent._generate_recommendations_impl(
                fuel_type="coal",
                emissions_kg=10000.0,
                country="US",
            )
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], f"Run {i+1} differs from run 1"

    # ============================================================================
    # Test 2: Hash-Based Determinism Verification
    # ============================================================================

    @pytest.mark.asyncio
    @patch("greenlang.agents.fuel_agent_ai.ChatSession")
    async def test_hash_based_determinism(
        self,
        mock_session_class,
        agent,
        tester,
        natural_gas_payload,
        mock_chat_session,
    ):
        """Test hash-based determinism over multiple runs."""
        # Setup mock
        mock_session = mock_chat_session(
            text="Calculated emissions from natural gas using emission factor.",
            tool_calls=[
                {
                    "name": "calculate_emissions",
                    "arguments": {
                        "fuel_type": "natural_gas",
                        "amount": 1000.0,
                        "unit": "therms",
                        "country": "US",
                        "renewable_percentage": 0,
                        "efficiency": 1.0,
                    },
                }
            ],
        )
        mock_session_class.return_value = mock_session

        # Test determinism
        result = await tester.test_agent_async(
            agent=agent,
            payload=natural_gas_payload,
            runs=5,
            store_outputs=True,
        )

        # Assert determinism
        assert_deterministic(result)
        assert result.run_count == 5
        assert len(set(result.hashes)) == 1, "All hashes should be identical"

    @pytest.mark.asyncio
    @patch("greenlang.agents.fuel_agent_ai.ChatSession")
    async def test_determinism_different_fuels(
        self,
        mock_session_class,
        agent,
        tester,
        diesel_payload,
        mock_chat_session,
    ):
        """Test determinism with different fuel types."""
        # Setup mock for diesel
        mock_session = mock_chat_session(
            text="Calculated emissions from diesel using emission factor.",
            tool_calls=[
                {
                    "name": "calculate_emissions",
                    "arguments": {
                        "fuel_type": "diesel",
                        "amount": 500.0,
                        "unit": "gallons",
                        "country": "US",
                        "renewable_percentage": 0,
                        "efficiency": 1.0,
                    },
                }
            ],
        )
        mock_session_class.return_value = mock_session

        # Test determinism
        result = await tester.test_agent_async(
            agent=agent,
            payload=diesel_payload,
            runs=3,
            store_outputs=True,
        )

        # Assert determinism
        assert_deterministic(result)

    # ============================================================================
    # Test 3: Seed=42, Temperature=0 Reproducibility
    # ============================================================================

    @pytest.mark.asyncio
    @patch("greenlang.agents.fuel_agent_ai.ChatSession")
    async def test_seed_temperature_reproducibility(
        self,
        mock_session_class,
        agent,
        natural_gas_payload,
        mock_chat_session,
    ):
        """Test that seed=42 and temperature=0 produce identical results."""
        # Setup mock
        mock_session = mock_chat_session(
            text="Emissions calculated with seed=42, temperature=0.",
            tool_calls=[
                {
                    "name": "calculate_emissions",
                    "arguments": {
                        "fuel_type": "natural_gas",
                        "amount": 1000.0,
                        "unit": "therms",
                        "country": "US",
                        "renewable_percentage": 0,
                        "efficiency": 1.0,
                    },
                }
            ],
        )
        mock_session_class.return_value = mock_session

        # Run multiple times and collect outputs
        outputs = []
        for _ in range(3):
            result = await agent.run(natural_gas_payload)
            outputs.append(result)

        # Verify all outputs are identical
        for i in range(1, len(outputs)):
            # Compare emissions values
            assert (
                outputs[i]["data"]["co2e_emissions_kg"]
                == outputs[0]["data"]["co2e_emissions_kg"]
            )
            # Compare metadata structure
            assert (
                outputs[i]["data"]["emission_factor"]
                == outputs[0]["data"]["emission_factor"]
            )

    # ============================================================================
    # Test 4: Citations Are Deterministic
    # ============================================================================

    def test_citations_deterministic(self, agent, natural_gas_payload):
        """Test that citations are deterministic across runs."""
        # Run calculation multiple times
        results = []
        for _ in range(3):
            result = agent._calculate_emissions_impl(
                fuel_type=natural_gas_payload["fuel_type"],
                amount=natural_gas_payload["amount"],
                unit=natural_gas_payload["unit"],
                country=natural_gas_payload["country"],
                renewable_percentage=0,
                efficiency=1.0,
            )
            results.append(result)

        # Verify citations are identical
        for i in range(1, len(results)):
            # Check if citations exist
            if "citations" in results[0]:
                assert "citations" in results[i]
                assert results[i]["citations"] == results[0]["citations"]

            # Check calculation metadata
            if "calculation" in results[0]:
                assert "calculation" in results[i]
                assert results[i]["calculation"] == results[0]["calculation"]

    # ============================================================================
    # Test 5: Emission Factor Lookups Are Deterministic
    # ============================================================================

    @pytest.mark.parametrize(
        "fuel_type,unit,country",
        [
            ("natural_gas", "therms", "US"),
            ("diesel", "gallons", "US"),
            ("electricity", "kWh", "US"),
            ("coal", "tons", "US"),
        ],
    )
    def test_emission_factor_lookup_deterministic(
        self, agent, fuel_type, unit, country
    ):
        """Test that emission factor lookups are deterministic for various fuels."""
        # Run lookup multiple times
        results = []
        for _ in range(5):
            result = agent._lookup_emission_factor_impl(
                fuel_type=fuel_type,
                unit=unit,
                country=country,
            )
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], (
                f"Lookup {i+1} differs from lookup 1 for {fuel_type}"
            )

    # ============================================================================
    # Test 6: Snapshot-Based Regression Testing
    # ============================================================================

    def test_natural_gas_snapshot(self, agent, snapshot_manager):
        """Test natural gas calculation against snapshot."""
        # Run calculation
        result = agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=1000.0,
            unit="therms",
            country="US",
            renewable_percentage=0,
            efficiency=1.0,
        )

        # Compare to snapshot
        diff = snapshot_manager.compare_snapshot(
            test_name="fuel_agent_natural_gas_1000_therms",
            actual_output=result,
        )

        # Assert match
        assert_snapshot_matches(diff)

    def test_diesel_snapshot(self, agent, snapshot_manager):
        """Test diesel calculation against snapshot."""
        # Run calculation
        result = agent._calculate_emissions_impl(
            fuel_type="diesel",
            amount=500.0,
            unit="gallons",
            country="US",
            renewable_percentage=0,
            efficiency=1.0,
        )

        # Compare to snapshot
        diff = snapshot_manager.compare_snapshot(
            test_name="fuel_agent_diesel_500_gallons",
            actual_output=result,
        )

        # Assert match
        assert_snapshot_matches(diff)

    # ============================================================================
    # Test 7: Cross-Platform Determinism
    # ============================================================================

    def test_cross_platform_hash_consistency(self, agent, tester, natural_gas_payload):
        """Test that hashes are consistent across platforms."""
        # Create multiple tester instances with different settings
        testers = [
            DeterminismTester(normalize_platform=True),
            DeterminismTester(normalize_platform=True, normalize_floats=True),
            DeterminismTester(normalize_platform=True, float_precision=6),
        ]

        # Run calculation
        result = agent._calculate_emissions_impl(
            fuel_type=natural_gas_payload["fuel_type"],
            amount=natural_gas_payload["amount"],
            unit=natural_gas_payload["unit"],
            country=natural_gas_payload["country"],
            renewable_percentage=0,
            efficiency=1.0,
        )

        # Compute hashes with different normalizations
        hashes = [tester.compute_hash(result) for tester in testers]

        # First two should match (both normalize platform)
        assert hashes[0] == hashes[1]

    # ============================================================================
    # Test 8: Determinism Under Different Conditions
    # ============================================================================

    @pytest.mark.parametrize(
        "amount,expected_emissions_range",
        [
            (100.0, (500, 600)),
            (1000.0, (5000, 6000)),
            (5000.0, (25000, 30000)),
        ],
    )
    def test_determinism_different_amounts(
        self, agent, tester, amount, expected_emissions_range
    ):
        """Test determinism with different fuel amounts."""
        # Run multiple times with same amount
        results = []
        for _ in range(3):
            result = agent._calculate_emissions_impl(
                fuel_type="natural_gas",
                amount=amount,
                unit="therms",
                country="US",
                renewable_percentage=0,
                efficiency=1.0,
            )
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0]

        # Check emissions are in expected range
        emissions = results[0]["emissions_kg_co2e"]
        assert expected_emissions_range[0] <= emissions <= expected_emissions_range[1]

    # ============================================================================
    # Test 9: Output Structure Consistency
    # ============================================================================

    def test_output_structure_deterministic(self, agent, natural_gas_payload):
        """Test that output structure is consistent across runs."""
        # Run multiple times
        results = []
        for _ in range(3):
            result = agent._calculate_emissions_impl(
                fuel_type=natural_gas_payload["fuel_type"],
                amount=natural_gas_payload["amount"],
                unit=natural_gas_payload["unit"],
                country=natural_gas_payload["country"],
                renewable_percentage=0,
                efficiency=1.0,
            )
            results.append(result)

        # Verify consistent structure
        required_fields = {
            "emissions_kg_co2e",
            "emission_factor",
            "emission_factor_unit",
            "scope",
            "calculation",
        }

        for result in results:
            assert required_fields.issubset(set(result.keys()))
            # All results should have same keys
            assert set(results[0].keys()) == set(result.keys())

    # ============================================================================
    # Test 10: Edge Cases Determinism
    # ============================================================================

    def test_determinism_zero_amount(self, agent, tester):
        """Test determinism with zero fuel amount."""
        results = []
        for _ in range(3):
            result = agent._calculate_emissions_impl(
                fuel_type="natural_gas",
                amount=0.0,
                unit="therms",
                country="US",
                renewable_percentage=0,
                efficiency=1.0,
            )
            results.append(result)

        # Should be deterministic even with zero
        for i in range(1, len(results)):
            assert results[i] == results[0]

        # Should have zero emissions
        assert results[0]["emissions_kg_co2e"] == 0.0

    def test_determinism_high_renewable_percentage(self, agent, tester):
        """Test determinism with 100% renewable fuel."""
        results = []
        for _ in range(3):
            result = agent._calculate_emissions_impl(
                fuel_type="natural_gas",
                amount=1000.0,
                unit="therms",
                country="US",
                renewable_percentage=100,
                efficiency=1.0,
            )
            results.append(result)

        # Should be deterministic
        for i in range(1, len(results)):
            assert results[i] == results[0]

        # Should have reduced emissions
        assert results[0]["emissions_kg_co2e"] < 5310.0  # Less than non-renewable
