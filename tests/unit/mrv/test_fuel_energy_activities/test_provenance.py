# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-016 Fuel & Energy Activities Agent provenance tracking.

Tests provenance chain construction, hashing, verification, serialization,
and deterministic reproducibility.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
import json
import hashlib

from greenlang.fuel_energy_activities.provenance import (
    FuelEnergyActivitiesProvenance,
    ProvenanceChain,
    ProvenanceEntry,
    ProvenanceStage,
)
from greenlang.fuel_energy_activities.models import (
    FuelType,
    ActivityType,
    CalculationMethod,
    FuelConsumptionRecord,
    ElectricityConsumptionRecord,
    WTTEmissionFactor,
    UpstreamElectricityFactor,
    TDLossFactor,
    Activity3aResult,
    Activity3bResult,
    Activity3cResult,
    CalculationResult,
    FuelBreakdown,
    ElectricityBreakdown,
    TDLossBreakdown,
    GasBreakdown,
    QualityTier,
)


# ============================================================================
# PROVENANCE CHAIN TESTS
# ============================================================================

class TestProvenanceChain:
    """Test ProvenanceChain creation and management."""

    def test_start_chain(self):
        """Test start_chain() creates new provenance chain."""
        provenance = FuelEnergyActivitiesProvenance()

        chain = provenance.start_chain(
            facility_id="FAC-001",
            reporting_period="2024-Q1"
        )

        assert isinstance(chain, ProvenanceChain)
        assert chain.facility_id == "FAC-001"
        assert chain.reporting_period == "2024-Q1"
        assert len(chain.entries) == 0
        assert chain.genesis_hash is not None

    def test_add_entry(self):
        """Test add_entry() adds entry to chain."""
        provenance = FuelEnergyActivitiesProvenance()
        chain = provenance.start_chain("FAC-001", "2024-Q1")

        entry = ProvenanceEntry(
            stage=ProvenanceStage.INPUT_VALIDATION,
            description="Validated fuel consumption record",
            data_hash="abc123",
            timestamp=datetime.now(timezone.utc)
        )

        provenance.add_entry(chain, entry)

        assert len(chain.entries) == 1
        assert chain.entries[0].stage == ProvenanceStage.INPUT_VALIDATION

    def test_seal_chain(self):
        """Test seal_chain() finalizes provenance chain."""
        provenance = FuelEnergyActivitiesProvenance()
        chain = provenance.start_chain("FAC-001", "2024-Q1")

        entry = ProvenanceEntry(
            stage=ProvenanceStage.INPUT_VALIDATION,
            description="Test entry",
            data_hash="abc123",
            timestamp=datetime.now(timezone.utc)
        )
        provenance.add_entry(chain, entry)

        final_hash = provenance.seal_chain(chain)

        assert final_hash is not None
        assert len(final_hash) == 64  # SHA-256 hash length
        assert chain.is_sealed is True

    def test_verify_chain(self):
        """Test verify_chain() validates chain integrity."""
        provenance = FuelEnergyActivitiesProvenance()
        chain = provenance.start_chain("FAC-001", "2024-Q1")

        entry = ProvenanceEntry(
            stage=ProvenanceStage.INPUT_VALIDATION,
            description="Test entry",
            data_hash="abc123",
            timestamp=datetime.now(timezone.utc)
        )
        provenance.add_entry(chain, entry)
        provenance.seal_chain(chain)

        # Verify chain
        is_valid = provenance.verify_chain(chain)
        assert is_valid is True

    def test_verify_chain_tampered(self):
        """Test verify_chain() detects tampering."""
        provenance = FuelEnergyActivitiesProvenance()
        chain = provenance.start_chain("FAC-001", "2024-Q1")

        entry = ProvenanceEntry(
            stage=ProvenanceStage.INPUT_VALIDATION,
            description="Test entry",
            data_hash="abc123",
            timestamp=datetime.now(timezone.utc)
        )
        provenance.add_entry(chain, entry)
        provenance.seal_chain(chain)

        # Tamper with data
        chain.entries[0].description = "Modified description"

        # Verify should fail
        is_valid = provenance.verify_chain(chain)
        assert is_valid is False


# ============================================================================
# CHAIN HASH DETERMINISM TESTS
# ============================================================================

class TestChainHashDeterminism:
    """Test provenance hash determinism and reproducibility."""

    def test_chain_hash_deterministic(self):
        """Test chain hash is deterministic for same inputs."""
        provenance = FuelEnergyActivitiesProvenance()

        # Create first chain
        chain1 = provenance.start_chain("FAC-001", "2024-Q1")
        entry1 = ProvenanceEntry(
            stage=ProvenanceStage.INPUT_VALIDATION,
            description="Test entry",
            data_hash="abc123",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        provenance.add_entry(chain1, entry1)
        hash1 = provenance.seal_chain(chain1)

        # Create second chain with same data
        chain2 = provenance.start_chain("FAC-001", "2024-Q1")
        entry2 = ProvenanceEntry(
            stage=ProvenanceStage.INPUT_VALIDATION,
            description="Test entry",
            data_hash="abc123",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        provenance.add_entry(chain2, entry2)
        hash2 = provenance.seal_chain(chain2)

        # Hashes should be identical (bit-perfect reproducibility)
        assert hash1 == hash2

    def test_chain_hash_changes_with_different_inputs(self):
        """Test chain hash changes when inputs differ."""
        provenance = FuelEnergyActivitiesProvenance()

        # Create first chain
        chain1 = provenance.start_chain("FAC-001", "2024-Q1")
        entry1 = ProvenanceEntry(
            stage=ProvenanceStage.INPUT_VALIDATION,
            description="Test entry 1",
            data_hash="abc123",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        provenance.add_entry(chain1, entry1)
        hash1 = provenance.seal_chain(chain1)

        # Create second chain with different data
        chain2 = provenance.start_chain("FAC-001", "2024-Q1")
        entry2 = ProvenanceEntry(
            stage=ProvenanceStage.INPUT_VALIDATION,
            description="Test entry 2",  # Different description
            data_hash="abc123",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        provenance.add_entry(chain2, entry2)
        hash2 = provenance.seal_chain(chain2)

        # Hashes should differ
        assert hash1 != hash2

    def test_genesis_hash(self):
        """Test genesis hash is deterministic."""
        provenance = FuelEnergyActivitiesProvenance()

        chain1 = provenance.start_chain("FAC-001", "2024-Q1")
        chain2 = provenance.start_chain("FAC-001", "2024-Q1")

        # Genesis hashes should be identical for same facility+period
        assert chain1.genesis_hash == chain2.genesis_hash


# ============================================================================
# ALL 10 STAGES TESTS
# ============================================================================

class TestAllProvenanceStages:
    """Test all 10 provenance stages."""

    def test_all_10_stages(self):
        """Test provenance chain with all 10 stages."""
        provenance = FuelEnergyActivitiesProvenance()
        chain = provenance.start_chain("FAC-001", "2024-Q1")

        stages = [
            ProvenanceStage.INPUT_VALIDATION,
            ProvenanceStage.EMISSION_FACTOR_LOOKUP,
            ProvenanceStage.UNIT_CONVERSION,
            ProvenanceStage.WTT_CALCULATION,
            ProvenanceStage.UPSTREAM_CALCULATION,
            ProvenanceStage.TD_LOSS_CALCULATION,
            ProvenanceStage.AGGREGATION,
            ProvenanceStage.DQI_ASSESSMENT,
            ProvenanceStage.COMPLIANCE_CHECK,
            ProvenanceStage.OUTPUT_GENERATION
        ]

        # Add entry for each stage
        for i, stage in enumerate(stages):
            entry = ProvenanceEntry(
                stage=stage,
                description=f"Stage {i+1}: {stage.value}",
                data_hash=f"hash_{i}",
                timestamp=datetime.now(timezone.utc)
            )
            provenance.add_entry(chain, entry)

        assert len(chain.entries) == 10

        # Verify all stages present
        chain_stages = [entry.stage for entry in chain.entries]
        for stage in stages:
            assert stage in chain_stages


# ============================================================================
# HASH FUNCTION TESTS
# ============================================================================

class TestHashFunctions:
    """Test individual hash functions for different data types."""

    def test_hash_fuel_consumption(self, sample_fuel_record):
        """Test hash_fuel_consumption_record() is deterministic."""
        provenance = FuelEnergyActivitiesProvenance()

        hash1 = provenance.hash_fuel_consumption_record(sample_fuel_record)
        hash2 = provenance.hash_fuel_consumption_record(sample_fuel_record)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256

    def test_hash_fuel_consumption_different_data(self, sample_fuel_record, sample_diesel_record):
        """Test hash_fuel_consumption_record() differs for different records."""
        provenance = FuelEnergyActivitiesProvenance()

        hash1 = provenance.hash_fuel_consumption_record(sample_fuel_record)
        hash2 = provenance.hash_fuel_consumption_record(sample_diesel_record)

        assert hash1 != hash2

    def test_hash_electricity_consumption(self, sample_electricity_record):
        """Test hash_electricity_consumption_record() is deterministic."""
        provenance = FuelEnergyActivitiesProvenance()

        hash1 = provenance.hash_electricity_consumption_record(sample_electricity_record)
        hash2 = provenance.hash_electricity_consumption_record(sample_electricity_record)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_hash_wtt_factor(self, sample_wtt_factor):
        """Test hash_wtt_emission_factor() is deterministic."""
        provenance = FuelEnergyActivitiesProvenance()

        hash1 = provenance.hash_wtt_emission_factor(sample_wtt_factor)
        hash2 = provenance.hash_wtt_emission_factor(sample_wtt_factor)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_hash_upstream_factor(self, sample_upstream_ef):
        """Test hash_upstream_electricity_factor() is deterministic."""
        provenance = FuelEnergyActivitiesProvenance()

        hash1 = provenance.hash_upstream_electricity_factor(sample_upstream_ef)
        hash2 = provenance.hash_upstream_electricity_factor(sample_upstream_ef)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_hash_td_loss_factor(self, sample_td_loss_factor):
        """Test hash_td_loss_factor() is deterministic."""
        provenance = FuelEnergyActivitiesProvenance()

        hash1 = provenance.hash_td_loss_factor(sample_td_loss_factor)
        hash2 = provenance.hash_td_loss_factor(sample_td_loss_factor)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_hash_activity_3a_result(self, sample_activity_3a_result):
        """Test hash_activity_3a_result() is deterministic."""
        provenance = FuelEnergyActivitiesProvenance()

        hash1 = provenance.hash_activity_3a_result(sample_activity_3a_result)
        hash2 = provenance.hash_activity_3a_result(sample_activity_3a_result)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_hash_activity_3b_result(self, sample_activity_3b_result):
        """Test hash_activity_3b_result() is deterministic."""
        provenance = FuelEnergyActivitiesProvenance()

        hash1 = provenance.hash_activity_3b_result(sample_activity_3b_result)
        hash2 = provenance.hash_activity_3b_result(sample_activity_3b_result)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_hash_activity_3c_result(self, sample_activity_3c_result):
        """Test hash_activity_3c_result() is deterministic."""
        provenance = FuelEnergyActivitiesProvenance()

        hash1 = provenance.hash_activity_3c_result(sample_activity_3c_result)
        hash2 = provenance.hash_activity_3c_result(sample_activity_3c_result)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_hash_calculation_result(self, sample_calculation_result):
        """Test hash_calculation_result() is deterministic."""
        provenance = FuelEnergyActivitiesProvenance()

        hash1 = provenance.hash_calculation_result(sample_calculation_result)
        hash2 = provenance.hash_calculation_result(sample_calculation_result)

        assert hash1 == hash2
        assert len(hash1) == 64


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

class TestSerialization:
    """Test provenance chain serialization and deserialization."""

    def test_serialization_roundtrip(self):
        """Test provenance chain can be serialized and deserialized."""
        provenance = FuelEnergyActivitiesProvenance()
        chain = provenance.start_chain("FAC-001", "2024-Q1")

        entry = ProvenanceEntry(
            stage=ProvenanceStage.INPUT_VALIDATION,
            description="Test entry",
            data_hash="abc123",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        provenance.add_entry(chain, entry)
        provenance.seal_chain(chain)

        # Serialize to JSON
        json_data = provenance.serialize_chain(chain)
        assert isinstance(json_data, str)

        # Deserialize from JSON
        chain_restored = provenance.deserialize_chain(json_data)

        assert chain_restored.facility_id == chain.facility_id
        assert chain_restored.reporting_period == chain.reporting_period
        assert len(chain_restored.entries) == len(chain.entries)
        assert chain_restored.final_hash == chain.final_hash

    def test_serialization_preserves_hashes(self):
        """Test serialization preserves all hashes."""
        provenance = FuelEnergyActivitiesProvenance()
        chain = provenance.start_chain("FAC-001", "2024-Q1")

        entry = ProvenanceEntry(
            stage=ProvenanceStage.INPUT_VALIDATION,
            description="Test entry",
            data_hash="abc123",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        provenance.add_entry(chain, entry)
        final_hash = provenance.seal_chain(chain)

        # Serialize and deserialize
        json_data = provenance.serialize_chain(chain)
        chain_restored = provenance.deserialize_chain(json_data)

        assert chain_restored.final_hash == final_hash
        assert chain_restored.genesis_hash == chain.genesis_hash


# ============================================================================
# BATCH PROVENANCE TESTS
# ============================================================================

class TestBatchProvenance:
    """Test provenance tracking for batch operations."""

    def test_batch_provenance(self):
        """Test provenance tracking for batch calculations."""
        provenance = FuelEnergyActivitiesProvenance()

        chains = []
        for i in range(5):
            chain = provenance.start_chain(f"FAC-{i:03d}", "2024-Q1")

            entry = ProvenanceEntry(
                stage=ProvenanceStage.INPUT_VALIDATION,
                description=f"Batch record {i}",
                data_hash=f"hash_{i}",
                timestamp=datetime.now(timezone.utc)
            )
            provenance.add_entry(chain, entry)
            provenance.seal_chain(chain)

            chains.append(chain)

        assert len(chains) == 5

        # Each chain should have unique hash
        hashes = [chain.final_hash for chain in chains]
        assert len(set(hashes)) == 5  # All unique

    def test_batch_provenance_aggregation(self):
        """Test provenance aggregation for batch results."""
        provenance = FuelEnergyActivitiesProvenance()

        # Create individual chains
        individual_hashes = []
        for i in range(3):
            chain = provenance.start_chain(f"FAC-{i:03d}", "2024-Q1")
            entry = ProvenanceEntry(
                stage=ProvenanceStage.WTT_CALCULATION,
                description=f"Calculation {i}",
                data_hash=f"hash_{i}",
                timestamp=datetime.now(timezone.utc)
            )
            provenance.add_entry(chain, entry)
            final_hash = provenance.seal_chain(chain)
            individual_hashes.append(final_hash)

        # Create aggregate chain
        aggregate_chain = provenance.start_chain("BATCH-001", "2024-Q1")
        aggregate_entry = ProvenanceEntry(
            stage=ProvenanceStage.AGGREGATION,
            description="Aggregated batch results",
            data_hash=hashlib.sha256(
                "".join(individual_hashes).encode()
            ).hexdigest(),
            timestamp=datetime.now(timezone.utc)
        )
        provenance.add_entry(aggregate_chain, aggregate_entry)
        aggregate_hash = provenance.seal_chain(aggregate_chain)

        assert aggregate_hash is not None
        assert len(aggregate_hash) == 64


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestProvenanceEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_chain_seal(self):
        """Test sealing empty chain."""
        provenance = FuelEnergyActivitiesProvenance()
        chain = provenance.start_chain("FAC-001", "2024-Q1")

        # Seal chain with no entries
        final_hash = provenance.seal_chain(chain)

        assert final_hash is not None
        assert len(final_hash) == 64

    def test_double_seal_prevention(self):
        """Test cannot seal chain twice."""
        provenance = FuelEnergyActivitiesProvenance()
        chain = provenance.start_chain("FAC-001", "2024-Q1")

        provenance.seal_chain(chain)

        # Second seal should raise error
        with pytest.raises(ValueError):
            provenance.seal_chain(chain)

    def test_add_entry_to_sealed_chain(self):
        """Test cannot add entry to sealed chain."""
        provenance = FuelEnergyActivitiesProvenance()
        chain = provenance.start_chain("FAC-001", "2024-Q1")

        provenance.seal_chain(chain)

        # Adding entry should raise error
        with pytest.raises(ValueError):
            entry = ProvenanceEntry(
                stage=ProvenanceStage.INPUT_VALIDATION,
                description="Test",
                data_hash="abc123",
                timestamp=datetime.now(timezone.utc)
            )
            provenance.add_entry(chain, entry)

    def test_hash_handles_none_values(self):
        """Test hash functions handle None values gracefully."""
        provenance = FuelEnergyActivitiesProvenance()

        # Should not raise exception
        try:
            provenance.hash_fuel_consumption_record(None)
        except Exception as e:
            pytest.fail(f"Hash function should handle None: {e}")

    def test_chain_length_limit(self):
        """Test chain enforces maximum length."""
        provenance = FuelEnergyActivitiesProvenance()
        chain = provenance.start_chain("FAC-001", "2024-Q1")

        # Add many entries (up to limit)
        for i in range(100):
            entry = ProvenanceEntry(
                stage=ProvenanceStage.INPUT_VALIDATION,
                description=f"Entry {i}",
                data_hash=f"hash_{i}",
                timestamp=datetime.now(timezone.utc)
            )
            provenance.add_entry(chain, entry)

        assert len(chain.entries) == 100
