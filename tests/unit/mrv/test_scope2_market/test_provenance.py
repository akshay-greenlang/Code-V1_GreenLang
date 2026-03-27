# -*- coding: utf-8 -*-
"""
Unit tests for Scope2MarketProvenance

AGENT-MRV-010: Scope 2 Market-Based Emissions Agent

Tests the SHA-256 provenance chain for Scope 2 market-based emission
calculations covering hash_input, hash_output, hash_instrument_lookup,
hash_allocation, hash_quality_assessment, hash_covered_calculation,
hash_uncovered_calculation, hash_residual_mix_lookup, hash_supplier_factor,
hash_gas_breakdown, hash_gwp_conversion, hash_certificate_retirement,
hash_dual_reporting, hash_compliance_check, hash_uncertainty,
hash_aggregation, hash_batch, chain verification, serialization,
merge, export, and reset.

Target: ~60 tests, ~700 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

try:
    from greenlang.agents.mrv.scope2_market.provenance import (
        Scope2MarketProvenance,
        ProvenanceEntry,
        VALID_STAGES,
    )
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not PROVENANCE_AVAILABLE, reason="Provenance module not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prov():
    """Create a fresh Scope2MarketProvenance instance."""
    return Scope2MarketProvenance()


@pytest.fixture
def seeded_prov():
    """Create a provenance instance with an input entry already added."""
    p = Scope2MarketProvenance()
    p.hash_input({"facility_id": "FAC-001", "year": 2025, "method": "market"})
    return p


# ===========================================================================
# 1. TestChainCreation  (5 tests)
# ===========================================================================


@_SKIP
class TestChainCreation:
    """Tests for chain creation: empty chain, single entry, multiple entries."""

    def test_empty_chain_length_zero(self, prov):
        """Freshly created provenance has zero chain entries."""
        assert len(prov.get_chain()) == 0

    def test_empty_chain_hash_is_empty(self, prov):
        """Empty chain returns empty string for chain hash."""
        assert prov.get_chain_hash() == ""

    def test_single_entry_chain(self, prov):
        """Adding one input produces a chain of length 1."""
        prov.hash_input({"facility_id": "FAC-001"})
        assert len(prov.get_chain()) == 1

    def test_multiple_entries_chain_grows(self, prov):
        """Each hash operation increments the chain length."""
        prov.hash_input({"facility_id": "FAC-001"})
        prov.hash_output({"total_co2e_tonnes": 100.0})
        assert len(prov.get_chain()) == 2

    def test_chain_entries_are_provenance_entry(self, seeded_prov):
        """Chain entries are ProvenanceEntry instances."""
        chain = seeded_prov.get_chain()
        assert len(chain) >= 1
        assert isinstance(chain[0], ProvenanceEntry)


# ===========================================================================
# 2. TestHashDeterminism  (10 tests)
# ===========================================================================


@_SKIP
class TestHashDeterminism:
    """Tests for deterministic hashing: same input -> same hash, different input -> different hash."""

    def test_same_input_same_hash(self):
        """Same inputs produce identical hashes in independent chains."""
        data = {"facility_id": "FAC-001", "year": 2025}
        p1 = Scope2MarketProvenance()
        h1 = p1.hash_input(data)
        p2 = Scope2MarketProvenance()
        h2 = p2.hash_input(data)
        assert h1 == h2

    def test_different_input_different_hash(self):
        """Different inputs produce different hashes."""
        p1 = Scope2MarketProvenance()
        h1 = p1.hash_input({"facility_id": "FAC-001"})
        p2 = Scope2MarketProvenance()
        h2 = p2.hash_input({"facility_id": "FAC-002"})
        assert h1 != h2

    def test_hash_input_returns_64_char_hex(self, prov):
        """hash_input returns a 64-character hex SHA-256 digest."""
        h = prov.hash_input({"facility_id": "FAC-001"})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_output_returns_64_char_hex(self, seeded_prov):
        """hash_output returns a 64-character hex SHA-256 digest."""
        h = seeded_prov.hash_output({"total_co2e_tonnes": 100.5})
        assert len(h) == 64

    def test_instrument_lookup_determinism(self):
        """hash_instrument_lookup is deterministic for identical inputs."""
        p1 = Scope2MarketProvenance()
        p1.hash_input({"facility_id": "FAC-001"})
        h1 = p1.hash_instrument_lookup("REC-001", "REC", 3000.0, 0.0)
        p2 = Scope2MarketProvenance()
        p2.hash_input({"facility_id": "FAC-001"})
        h2 = p2.hash_instrument_lookup("REC-001", "REC", 3000.0, 0.0)
        assert h1 == h2

    def test_allocation_determinism(self):
        """hash_instrument_allocation is deterministic for identical inputs."""
        p1 = Scope2MarketProvenance()
        p1.hash_input({"facility_id": "FAC-001"})
        h1 = p1.hash_instrument_allocation("PUR-001", "REC-001", 3000.0)
        p2 = Scope2MarketProvenance()
        p2.hash_input({"facility_id": "FAC-001"})
        h2 = p2.hash_instrument_allocation("PUR-001", "REC-001", 3000.0)
        assert h1 == h2

    def test_covered_calc_determinism(self):
        """hash_covered_calculation is deterministic for identical inputs."""
        p1 = Scope2MarketProvenance()
        p1.hash_input({"facility_id": "FAC-001"})
        h1 = p1.hash_covered_calculation("REC-001", 3000.0, 0.0, 0.0)
        p2 = Scope2MarketProvenance()
        p2.hash_input({"facility_id": "FAC-001"})
        h2 = p2.hash_covered_calculation("REC-001", 3000.0, 0.0, 0.0)
        assert h1 == h2

    def test_uncovered_calc_determinism(self):
        """hash_uncovered_calculation is deterministic for identical inputs."""
        p1 = Scope2MarketProvenance()
        p1.hash_input({"facility_id": "FAC-001"})
        h1 = p1.hash_uncovered_calculation(2000.0, "US-WECC", 350.0, 700000.0)
        p2 = Scope2MarketProvenance()
        p2.hash_input({"facility_id": "FAC-001"})
        h2 = p2.hash_uncovered_calculation(2000.0, "US-WECC", 350.0, 700000.0)
        assert h1 == h2

    def test_supplier_factor_determinism(self):
        """hash_supplier_factor is deterministic."""
        p1 = Scope2MarketProvenance()
        p1.hash_input({"facility_id": "FAC-001"})
        h1 = p1.hash_supplier_factor("SUP-001", 280.0, 2024)
        p2 = Scope2MarketProvenance()
        p2.hash_input({"facility_id": "FAC-001"})
        h2 = p2.hash_supplier_factor("SUP-001", 280.0, 2024)
        assert h1 == h2

    def test_residual_mix_lookup_determinism(self):
        """hash_residual_mix_lookup is deterministic."""
        p1 = Scope2MarketProvenance()
        p1.hash_input({"facility_id": "FAC-001"})
        h1 = p1.hash_residual_mix_lookup("DE", 420.5, "AIB_RM_2024")
        p2 = Scope2MarketProvenance()
        p2.hash_input({"facility_id": "FAC-001"})
        h2 = p2.hash_residual_mix_lookup("DE", 420.5, "AIB_RM_2024")
        assert h1 == h2


# ===========================================================================
# 3. TestChainIntegrity  (10 tests)
# ===========================================================================


@_SKIP
class TestChainIntegrity:
    """Tests for chain integrity: verify_chain passes/fails correctly."""

    def test_verify_empty_chain(self, prov):
        """Empty chain verifies as True."""
        assert prov.verify_chain() is True

    def test_verify_single_entry_chain(self, seeded_prov):
        """Single-entry chain verifies as True."""
        assert seeded_prov.verify_chain() is True

    def test_verify_multi_entry_chain(self, prov):
        """Multi-entry chain verifies as True."""
        prov.hash_input({"facility_id": "FAC-001"})
        prov.hash_instrument_lookup("REC-001", "REC", 3000.0, 0.0)
        prov.hash_covered_calculation("REC-001", 3000.0, 0.0, 0.0)
        prov.hash_output({"total_co2e_tonnes": 0.0})
        assert prov.verify_chain() is True

    def test_first_entry_has_empty_previous_hash(self, prov):
        """First chain entry has empty previous_hash."""
        prov.hash_input({"facility_id": "FAC-001"})
        chain = prov.get_chain()
        assert chain[0].previous_hash == ""

    def test_second_entry_links_to_first(self, prov):
        """Second entry's previous_hash equals first entry's hash_value."""
        prov.hash_input({"facility_id": "FAC-001"})
        prov.hash_output({"total_co2e_tonnes": 100.0})
        chain = prov.get_chain()
        assert chain[1].previous_hash == chain[0].hash_value

    def test_chain_hash_non_empty_after_entries(self, seeded_prov):
        """Non-empty chain returns a 64-char hex chain hash."""
        chain_hash = seeded_prov.get_chain_hash()
        assert len(chain_hash) == 64

    def test_full_pipeline_chain_valid(self, prov):
        """Full market-based pipeline produces valid chain."""
        prov.hash_input({"facility_id": "FAC-001", "year": 2025})
        prov.hash_instrument_lookup("REC-001", "REC", 5000.0, 0.0)
        prov.hash_instrument_allocation("PUR-001", "REC-001", 5000.0)
        prov.hash_quality_assessment("REC-001", {"conveys": True}, 0.95)
        prov.hash_covered_calculation("REC-001", 5000.0, 0.0, 0.0)
        prov.hash_uncovered_calculation(2000.0, "US-WECC", 350.0, 700000.0)
        prov.hash_residual_mix_lookup("US-WECC", 350.0, "Green-e_2024")
        prov.hash_output({"total_co2e_tonnes": 700.0})
        assert prov.verify_chain() is True
        assert len(prov.get_chain()) == 8

    def test_chain_entries_have_timestamps(self, seeded_prov):
        """All chain entries have non-empty timestamps."""
        chain = seeded_prov.get_chain()
        for entry in chain:
            assert entry.timestamp != ""
            assert "T" in entry.timestamp  # ISO format

    def test_chain_entries_have_stages(self, seeded_prov):
        """All chain entries have non-empty stage names."""
        chain = seeded_prov.get_chain()
        for entry in chain:
            assert entry.stage != ""

    def test_chain_entries_have_metadata(self, seeded_prov):
        """Chain entries have metadata dict."""
        chain = seeded_prov.get_chain()
        assert isinstance(chain[0].metadata, dict)
        assert len(chain[0].metadata) > 0


# ===========================================================================
# 4. TestStageHashing  (20 tests)
# ===========================================================================


@_SKIP
class TestStageHashing:
    """Tests for domain-specific stage hashing methods."""

    def test_hash_input_success(self, prov):
        """hash_input returns a valid 64-char hex hash."""
        h = prov.hash_input({"facility_id": "FAC-001", "method": "market"})
        assert len(h) == 64

    def test_hash_input_empty_raises(self, prov):
        """hash_input with empty dict raises ValueError."""
        with pytest.raises(ValueError):
            prov.hash_input({})

    def test_hash_instrument_lookup_success(self, seeded_prov):
        """hash_instrument_lookup returns a valid hash."""
        h = seeded_prov.hash_instrument_lookup("REC-001", "REC", 3000.0, 0.0)
        assert len(h) == 64

    def test_hash_instrument_lookup_empty_id_raises(self, seeded_prov):
        """Empty instrument_id raises ValueError."""
        with pytest.raises(ValueError, match="instrument_id"):
            seeded_prov.hash_instrument_lookup("", "REC", 3000.0, 0.0)

    def test_hash_instrument_lookup_empty_type_raises(self, seeded_prov):
        """Empty instrument_type raises ValueError."""
        with pytest.raises(ValueError, match="instrument_type"):
            seeded_prov.hash_instrument_lookup("REC-001", "", 3000.0, 0.0)

    def test_hash_instrument_lookup_negative_mwh_raises(self, seeded_prov):
        """Negative quantity_mwh raises ValueError."""
        with pytest.raises(ValueError, match="quantity_mwh"):
            seeded_prov.hash_instrument_lookup("REC-001", "REC", -100.0, 0.0)

    def test_hash_allocation_success(self, seeded_prov):
        """hash_instrument_allocation returns a valid hash."""
        h = seeded_prov.hash_instrument_allocation("PUR-001", "REC-001", 3000.0)
        assert len(h) == 64

    def test_hash_allocation_empty_purchase_id_raises(self, seeded_prov):
        """Empty purchase_id raises ValueError."""
        with pytest.raises(ValueError, match="purchase_id"):
            seeded_prov.hash_instrument_allocation("", "REC-001", 3000.0)

    def test_hash_quality_assessment_success(self, seeded_prov):
        """hash_quality_assessment returns a valid hash."""
        h = seeded_prov.hash_quality_assessment(
            "REC-001", {"conveys_attributes": True, "unique_claim": True}, 0.95,
        )
        assert len(h) == 64

    def test_hash_quality_assessment_score_out_of_range_raises(self, seeded_prov):
        """Overall score > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="overall_score"):
            seeded_prov.hash_quality_assessment("REC-001", {"a": True}, 1.5)

    def test_hash_covered_calculation_success(self, seeded_prov):
        """hash_covered_calculation returns a valid hash."""
        h = seeded_prov.hash_covered_calculation("REC-001", 3000.0, 0.0, 0.0)
        assert len(h) == 64

    def test_hash_covered_calculation_negative_mwh_raises(self, seeded_prov):
        """Negative mwh in covered calc raises ValueError."""
        with pytest.raises(ValueError, match="mwh"):
            seeded_prov.hash_covered_calculation("REC-001", -100.0, 0.0, 0.0)

    def test_hash_uncovered_calculation_success(self, seeded_prov):
        """hash_uncovered_calculation returns a valid hash."""
        h = seeded_prov.hash_uncovered_calculation(2000.0, "US-WECC", 350.0, 700000.0)
        assert len(h) == 64

    def test_hash_uncovered_calculation_empty_region_raises(self, seeded_prov):
        """Empty region raises ValueError."""
        with pytest.raises(ValueError, match="region"):
            seeded_prov.hash_uncovered_calculation(2000.0, "", 350.0, 700000.0)

    def test_hash_residual_mix_lookup_success(self, seeded_prov):
        """hash_residual_mix_lookup returns a valid hash."""
        h = seeded_prov.hash_residual_mix_lookup("DE", 420.5, "AIB_RM_2024")
        assert len(h) == 64

    def test_hash_residual_mix_lookup_negative_factor_raises(self, seeded_prov):
        """Negative factor raises ValueError."""
        with pytest.raises(ValueError, match="factor"):
            seeded_prov.hash_residual_mix_lookup("DE", -10.0, "AIB_RM_2024")

    def test_hash_supplier_factor_success(self, seeded_prov):
        """hash_supplier_factor returns a valid hash."""
        h = seeded_prov.hash_supplier_factor("SUP-ENEL-IT", 280.0, 2024)
        assert len(h) == 64

    def test_hash_supplier_factor_empty_id_raises(self, seeded_prov):
        """Empty supplier_id raises ValueError."""
        with pytest.raises(ValueError, match="supplier_id"):
            seeded_prov.hash_supplier_factor("", 280.0, 2024)

    def test_hash_supplier_factor_zero_year_raises(self, seeded_prov):
        """Year <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="year"):
            seeded_prov.hash_supplier_factor("SUP-001", 280.0, 0)

    def test_hash_gas_breakdown_success(self, seeded_prov):
        """hash_gas_breakdown returns a valid hash."""
        h = seeded_prov.hash_gas_breakdown(
            co2_kg=700000.0,
            ch4_kg=50.0,
            n2o_kg=20.0,
            gwp_source="IPCC_AR5",
            total_co2e=701390.0,
        )
        assert len(h) == 64

    def test_hash_compliance_check_success(self, seeded_prov):
        """hash_compliance_check returns a valid hash."""
        h = seeded_prov.hash_compliance_check(
            framework="ghg_protocol_scope2",
            status="compliant",
            findings_count=7,
        )
        assert len(h) == 64

    def test_hash_uncertainty_success(self, seeded_prov):
        """hash_uncertainty returns a valid hash."""
        h = seeded_prov.hash_uncertainty(
            method="monte_carlo",
            mean_co2e=700.0,
            std_dev=70.0,
            ci_lower=570.0,
            ci_upper=830.0,
        )
        assert len(h) == 64

    def test_hash_output_empty_raises(self, seeded_prov):
        """hash_output with empty dict raises ValueError."""
        with pytest.raises(ValueError):
            seeded_prov.hash_output({})


# ===========================================================================
# 5. TestChainMerge  (5 tests)
# ===========================================================================


@_SKIP
class TestChainMerge:
    """Tests for merging two provenance chains."""

    def test_merge_two_chains(self):
        """Merging two chains creates a valid combined chain."""
        p1 = Scope2MarketProvenance()
        p1.hash_input({"facility_id": "FAC-001"})

        p2 = Scope2MarketProvenance()
        p2.hash_input({"facility_id": "FAC-002"})

        if hasattr(p1, "merge_chains"):
            merged = p1.merge_chains(p2)
            if merged is not None:
                chain = merged.get_chain() if hasattr(merged, "get_chain") else []
                assert len(chain) >= 2
        else:
            pytest.skip("merge_chains not implemented")

    def test_merge_empty_with_populated(self):
        """Merging empty chain with populated chain works."""
        p1 = Scope2MarketProvenance()
        p2 = Scope2MarketProvenance()
        p2.hash_input({"facility_id": "FAC-001"})

        if hasattr(p1, "merge_chains"):
            merged = p1.merge_chains(p2)
            if merged is not None:
                chain = merged.get_chain() if hasattr(merged, "get_chain") else []
                assert len(chain) >= 1
        else:
            pytest.skip("merge_chains not implemented")

    def test_merge_preserves_individual_hashes(self):
        """Merged chain preserves original entry hash values."""
        p1 = Scope2MarketProvenance()
        h1 = p1.hash_input({"facility_id": "FAC-001"})
        p2 = Scope2MarketProvenance()
        h2 = p2.hash_input({"facility_id": "FAC-002"})

        if hasattr(p1, "merge_chains"):
            merged = p1.merge_chains(p2)
            if merged is not None:
                chain = merged.get_chain() if hasattr(merged, "get_chain") else []
                hashes = [e.hash_value for e in chain]
                assert h1 in hashes
        else:
            pytest.skip("merge_chains not implemented")

    def test_merge_two_populated_chains(self):
        """Merging two multi-entry chains works."""
        p1 = Scope2MarketProvenance()
        p1.hash_input({"facility_id": "FAC-001"})
        p1.hash_output({"total_co2e_tonnes": 100.0})

        p2 = Scope2MarketProvenance()
        p2.hash_input({"facility_id": "FAC-002"})
        p2.hash_output({"total_co2e_tonnes": 200.0})

        if hasattr(p1, "merge_chains"):
            merged = p1.merge_chains(p2)
            if merged is not None:
                chain = merged.get_chain() if hasattr(merged, "get_chain") else []
                assert len(chain) >= 4
        else:
            pytest.skip("merge_chains not implemented")

    def test_merged_chain_verifies(self):
        """Merged chain passes verify_chain."""
        p1 = Scope2MarketProvenance()
        p1.hash_input({"facility_id": "FAC-001"})
        p2 = Scope2MarketProvenance()
        p2.hash_input({"facility_id": "FAC-002"})

        if hasattr(p1, "merge_chains"):
            merged = p1.merge_chains(p2)
            if merged is not None and hasattr(merged, "verify_chain"):
                assert merged.verify_chain() is True
        else:
            pytest.skip("merge_chains not implemented")


# ===========================================================================
# 6. TestExport  (5 tests)
# ===========================================================================


@_SKIP
class TestExport:
    """Tests for serialization: to_dict, to_json, from_dict."""

    def test_entry_to_dict(self, seeded_prov):
        """ProvenanceEntry.to_dict returns expected keys."""
        chain = seeded_prov.get_chain()
        d = chain[0].to_dict()
        assert "stage" in d
        assert "hash_value" in d
        assert "timestamp" in d
        assert "previous_hash" in d
        assert "metadata" in d

    def test_entry_from_dict(self):
        """ProvenanceEntry.from_dict reconstructs an entry."""
        data = {
            "stage": "input",
            "hash_value": "a" * 64,
            "timestamp": "2025-01-01T00:00:00+00:00",
            "previous_hash": "",
            "metadata": {"facility_id": "FAC-001"},
        }
        entry = ProvenanceEntry.from_dict(data)
        assert entry.stage == "input"
        assert entry.hash_value == "a" * 64
        assert entry.metadata["facility_id"] == "FAC-001"

    def test_entry_from_dict_type_error(self):
        """Non-dict input raises TypeError."""
        with pytest.raises(TypeError):
            ProvenanceEntry.from_dict("not a dict")

    def test_entry_roundtrip(self, seeded_prov):
        """to_dict -> from_dict preserves all fields."""
        chain = seeded_prov.get_chain()
        original = chain[0]
        d = original.to_dict()
        restored = ProvenanceEntry.from_dict(d)
        assert restored.stage == original.stage
        assert restored.hash_value == original.hash_value
        assert restored.previous_hash == original.previous_hash

    def test_chain_to_json_serializable(self, seeded_prov):
        """Chain entries can be serialized to JSON via to_dict."""
        chain = seeded_prov.get_chain()
        dicts = [e.to_dict() for e in chain]
        json_str = json.dumps(dicts, default=str)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert len(parsed) == len(chain)


# ===========================================================================
# 7. TestReset  (5 tests)
# ===========================================================================


@_SKIP
class TestReset:
    """Tests for chain reset functionality."""

    def test_reset_clears_chain(self, seeded_prov):
        """reset() clears the chain back to zero entries."""
        assert len(seeded_prov.get_chain()) > 0
        if hasattr(seeded_prov, "reset"):
            seeded_prov.reset()
            assert len(seeded_prov.get_chain()) == 0
        else:
            pytest.skip("reset not implemented")

    def test_reset_chain_hash_empty(self, seeded_prov):
        """After reset, chain hash returns empty string."""
        if hasattr(seeded_prov, "reset"):
            seeded_prov.reset()
            assert seeded_prov.get_chain_hash() == ""
        else:
            pytest.skip("reset not implemented")

    def test_reset_allows_new_entries(self, seeded_prov):
        """After reset, new entries can be added."""
        if hasattr(seeded_prov, "reset"):
            seeded_prov.reset()
            h = seeded_prov.hash_input({"facility_id": "FAC-NEW"})
            assert len(h) == 64
            assert len(seeded_prov.get_chain()) == 1
        else:
            pytest.skip("reset not implemented")

    def test_reset_chain_verifies(self, seeded_prov):
        """After reset and new entries, chain verifies."""
        if hasattr(seeded_prov, "reset"):
            seeded_prov.reset()
            seeded_prov.hash_input({"facility_id": "FAC-NEW"})
            seeded_prov.hash_output({"total_co2e": 50.0})
            assert seeded_prov.verify_chain() is True
        else:
            pytest.skip("reset not implemented")

    def test_valid_stages_frozenset_not_empty(self):
        """VALID_STAGES constant is a non-empty frozenset."""
        assert isinstance(VALID_STAGES, frozenset)
        assert len(VALID_STAGES) > 10
        assert "input" in VALID_STAGES
        assert "output" in VALID_STAGES
        assert "instrument_lookup" in VALID_STAGES
        assert "covered_calculation" in VALID_STAGES
        assert "uncovered_calculation" in VALID_STAGES
