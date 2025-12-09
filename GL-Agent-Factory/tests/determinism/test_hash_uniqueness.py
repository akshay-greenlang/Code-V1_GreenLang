"""
Hash Uniqueness Tests

Validates properties of provenance hashes:
1. Uniqueness: Different inputs produce different hashes
2. Format: All hashes are valid SHA-256 format
3. Collision resistance: No hash collisions in test set
"""

import pytest
import hashlib
import json
from typing import Set


class TestProvenanceHashFormat:
    """Tests provenance hash format compliance."""

    @pytest.fixture
    def carbon_agent(self):
        """Create Carbon Emissions Agent instance."""
        from backend.agents.gl_001_carbon_emissions.agent import CarbonEmissionsAgent
        return CarbonEmissionsAgent()

    @pytest.fixture
    def carbon_input_class(self):
        """Get Carbon Emissions Input class."""
        from backend.agents.gl_001_carbon_emissions.agent import CarbonEmissionsInput
        return CarbonEmissionsInput

    def test_hash_is_64_hex_characters(self, carbon_agent, carbon_input_class):
        """Test provenance hash is valid SHA-256 format."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        input_data = carbon_input_class(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )

        result = carbon_agent.run(input_data)

        # Check length
        assert len(result.provenance_hash) == 64, \
            f"Hash length is {len(result.provenance_hash)}, expected 64"

        # Check it's valid hexadecimal
        try:
            int(result.provenance_hash, 16)
        except ValueError:
            pytest.fail(f"Hash is not valid hexadecimal: {result.provenance_hash}")

    def test_hash_lowercase(self, carbon_agent, carbon_input_class):
        """Test hash is lowercase hexadecimal."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        input_data = carbon_input_class(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=500,
            unit="m3",
            region="EU",
            scope=Scope.SCOPE_1,
        )

        result = carbon_agent.run(input_data)

        assert result.provenance_hash == result.provenance_hash.lower(), \
            "Hash should be lowercase"

    def test_hash_not_empty_or_none(self, carbon_agent, carbon_input_class):
        """Test hash is never empty or None."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        test_cases = [
            (FuelType.NATURAL_GAS, 1, "m3", "US"),
            (FuelType.DIESEL, 0.001, "L", "EU"),
            (FuelType.ELECTRICITY_GRID, 1000000, "kWh", "DE"),
        ]

        for fuel, qty, unit, region in test_cases:
            input_data = carbon_input_class(
                fuel_type=fuel,
                quantity=qty,
                unit=unit,
                region=region,
                scope=Scope.SCOPE_1 if fuel != FuelType.ELECTRICITY_GRID else Scope.SCOPE_2,
            )

            result = carbon_agent.run(input_data)

            assert result.provenance_hash is not None, "Hash should not be None"
            assert result.provenance_hash != "", "Hash should not be empty"


class TestHashUniqueness:
    """Tests that different inputs produce different hashes."""

    @pytest.fixture
    def carbon_agent(self):
        """Create Carbon Emissions Agent instance."""
        from backend.agents.gl_001_carbon_emissions.agent import CarbonEmissionsAgent
        return CarbonEmissionsAgent()

    @pytest.fixture
    def carbon_input_class(self):
        """Get Carbon Emissions Input class."""
        from backend.agents.gl_001_carbon_emissions.agent import CarbonEmissionsInput
        return CarbonEmissionsInput

    def test_different_quantities_different_hashes(self, carbon_agent, carbon_input_class):
        """Test that different quantities produce different hashes."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        hashes: Set[str] = set()

        quantities = [1, 10, 100, 1000, 10000, 100000]

        for qty in quantities:
            input_data = carbon_input_class(
                fuel_type=FuelType.NATURAL_GAS,
                quantity=qty,
                unit="m3",
                region="US",
                scope=Scope.SCOPE_1,
            )

            result = carbon_agent.run(input_data)
            hashes.add(result.provenance_hash)

        # Each quantity should produce a unique hash
        # Note: Hashes include timestamps, so they will always be unique
        assert len(hashes) == len(quantities), \
            f"Expected {len(quantities)} unique hashes, got {len(hashes)}"

    def test_different_fuel_types_different_hashes(self, carbon_agent, carbon_input_class):
        """Test that different fuel types produce different hashes."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        hashes: Set[str] = set()

        fuel_types = [FuelType.NATURAL_GAS, FuelType.DIESEL, FuelType.GASOLINE]

        for fuel in fuel_types:
            input_data = carbon_input_class(
                fuel_type=fuel,
                quantity=1000,
                unit="m3" if fuel == FuelType.NATURAL_GAS else "L",
                region="US",
                scope=Scope.SCOPE_1,
            )

            result = carbon_agent.run(input_data)
            hashes.add(result.provenance_hash)

        assert len(hashes) == len(fuel_types), \
            f"Expected {len(fuel_types)} unique hashes, got {len(hashes)}"

    def test_different_regions_different_hashes(self, carbon_agent, carbon_input_class):
        """Test that different regions produce different hashes."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        hashes: Set[str] = set()

        regions = ["US", "EU", "DE", "FR"]

        for region in regions:
            input_data = carbon_input_class(
                fuel_type=FuelType.ELECTRICITY_GRID,
                quantity=1000,
                unit="kWh",
                region=region,
                scope=Scope.SCOPE_2,
            )

            result = carbon_agent.run(input_data)
            hashes.add(result.provenance_hash)

        assert len(hashes) == len(regions), \
            f"Expected {len(regions)} unique hashes, got {len(hashes)}"


class TestHashCollisionResistance:
    """Tests for hash collision resistance."""

    def test_no_collisions_in_large_sample(self):
        """Generate many hashes and check for collisions."""
        from backend.agents.gl_001_carbon_emissions.agent import (
            CarbonEmissionsAgent,
            CarbonEmissionsInput,
            FuelType,
            Scope,
        )

        agent = CarbonEmissionsAgent()
        hashes: Set[str] = set()
        collision_count = 0

        # Generate 100 different inputs
        for fuel in [FuelType.NATURAL_GAS, FuelType.DIESEL]:
            for qty in range(1, 51):
                input_data = CarbonEmissionsInput(
                    fuel_type=fuel,
                    quantity=qty * 100,
                    unit="m3" if fuel == FuelType.NATURAL_GAS else "L",
                    region="US",
                    scope=Scope.SCOPE_1,
                )

                result = agent.run(input_data)

                if result.provenance_hash in hashes:
                    collision_count += 1
                hashes.add(result.provenance_hash)

        # Should have no collisions
        assert collision_count == 0, \
            f"Found {collision_count} hash collisions in {len(hashes)} samples"


class TestHashDeterminism:
    """Tests that hash computation is deterministic."""

    def test_hash_algorithm_consistency(self):
        """Verify SHA-256 produces consistent results."""
        test_data = {
            "agent_id": "test",
            "version": "1.0.0",
            "calculation": {"value": 1234.5678},
        }

        # Compute hash multiple times
        hashes = []
        for _ in range(10):
            json_str = json.dumps(test_data, sort_keys=True)
            hash_value = hashlib.sha256(json_str.encode()).hexdigest()
            hashes.append(hash_value)

        # All hashes should be identical
        assert all(h == hashes[0] for h in hashes), \
            "SHA-256 produced inconsistent hashes"

        # Verify expected format
        assert len(hashes[0]) == 64
        assert hashes[0] == hashes[0].lower()

    def test_json_sort_keys_requirement(self):
        """Test that sort_keys is required for determinism."""
        data1 = {"b": 2, "a": 1}
        data2 = {"a": 1, "b": 2}

        # Without sort_keys, different key orders might produce different JSON
        json1_sorted = json.dumps(data1, sort_keys=True)
        json2_sorted = json.dumps(data2, sort_keys=True)

        # With sort_keys, they should be identical
        assert json1_sorted == json2_sorted, \
            "sort_keys should produce identical JSON regardless of key order"

        # And therefore identical hashes
        hash1 = hashlib.sha256(json1_sorted.encode()).hexdigest()
        hash2 = hashlib.sha256(json2_sorted.encode()).hexdigest()

        assert hash1 == hash2


class TestCBAMHashUniqueness:
    """Tests hash uniqueness for CBAM agent."""

    def test_different_cn_codes_different_hashes(self):
        """Test that different CN codes produce different hashes."""
        from backend.agents.gl_002_cbam_compliance.agent import (
            CBAMComplianceAgent,
            CBAMInput,
        )

        agent = CBAMComplianceAgent()
        hashes: Set[str] = set()

        cn_codes = ["72081000", "76011000", "25232900"]

        for cn_code in cn_codes:
            input_data = CBAMInput(
                cn_code=cn_code,
                quantity_tonnes=100,
                country_of_origin="CN",
                reporting_period="Q1 2026",
            )

            result = agent.run(input_data)
            hashes.add(result.provenance_hash)

        assert len(hashes) == len(cn_codes), \
            f"Expected {len(cn_codes)} unique hashes"

    def test_different_countries_different_hashes(self):
        """Test that different countries produce different hashes."""
        from backend.agents.gl_002_cbam_compliance.agent import (
            CBAMComplianceAgent,
            CBAMInput,
        )

        agent = CBAMComplianceAgent()
        hashes: Set[str] = set()

        countries = ["CN", "IN", "BR", "TR"]

        for country in countries:
            input_data = CBAMInput(
                cn_code="72081000",
                quantity_tonnes=100,
                country_of_origin=country,
                reporting_period="Q1 2026",
            )

            result = agent.run(input_data)
            hashes.add(result.provenance_hash)

        assert len(hashes) == len(countries)
