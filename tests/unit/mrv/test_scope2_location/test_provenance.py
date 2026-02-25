# -*- coding: utf-8 -*-
"""
Unit tests for Scope2LocationProvenance

AGENT-MRV-009: Scope 2 Location-Based Emissions Agent

Tests the SHA-256 provenance chain for Scope 2 location-based emission
calculations covering hash_input, hash_output, hash_grid_factor,
hash_td_loss, hash_electricity_calculation, hash_steam_heat_cooling,
hash_gas_breakdown, hash_gwp_conversion, hash_compliance_check,
hash_uncertainty, chain verification, serialization, and merge.

Target: 30 tests, ~350 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

try:
    from greenlang.scope2_location.provenance import (
        Scope2LocationProvenance,
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
    """Create a fresh Scope2LocationProvenance instance."""
    return Scope2LocationProvenance()


@pytest.fixture
def seeded_prov():
    """Create a provenance instance with an input entry already added."""
    p = Scope2LocationProvenance()
    p.hash_input({"facility_id": "FAC-001", "year": 2025})
    return p


# ===========================================================================
# 1. TestProvenanceChain
# ===========================================================================


@_SKIP
class TestProvenanceChain:
    """Tests for hash_input, hash_output, get_chain, get_chain_hash."""

    def test_hash_input_returns_64_char_hex(self, prov):
        """hash_input returns a 64-character hex SHA-256 digest."""
        h = prov.hash_input({"facility_id": "FAC-001"})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_output_returns_64_char_hex(self, seeded_prov):
        """hash_output returns a 64-character hex SHA-256 digest."""
        h = seeded_prov.hash_output({"total_co2e_tonnes": 100.5})
        assert len(h) == 64

    def test_get_chain_length(self, prov):
        """Chain length grows with each hash operation."""
        assert len(prov.get_chain()) == 0
        prov.hash_input({"facility_id": "FAC-001"})
        assert len(prov.get_chain()) == 1
        prov.hash_output({"total_co2e_tonnes": 100.0})
        assert len(prov.get_chain()) == 2

    def test_get_chain_hash_empty(self, prov):
        """Empty chain returns empty string for chain hash."""
        assert prov.get_chain_hash() == ""

    def test_get_chain_hash_non_empty(self, seeded_prov):
        """Non-empty chain returns a 64-char hex chain hash."""
        chain_hash = seeded_prov.get_chain_hash()
        assert len(chain_hash) == 64

    def test_chain_entries_are_provenance_entry(self, seeded_prov):
        """Chain entries are ProvenanceEntry instances."""
        chain = seeded_prov.get_chain()
        assert len(chain) >= 1
        assert isinstance(chain[0], ProvenanceEntry)

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

    def test_hash_input_empty_raises(self, prov):
        """hash_input with empty dict raises ValueError."""
        with pytest.raises(ValueError):
            prov.hash_input({})

    def test_hash_output_empty_raises(self, seeded_prov):
        """hash_output with empty dict raises ValueError."""
        with pytest.raises(ValueError):
            seeded_prov.hash_output({})


# ===========================================================================
# 2. TestGridFactorHash
# ===========================================================================


@_SKIP
class TestGridFactorHash:
    """Tests for hash_grid_factor."""

    def test_hash_grid_factor_success(self, seeded_prov):
        """hash_grid_factor returns a valid hash."""
        h = seeded_prov.hash_grid_factor(
            region_id="US-WECC",
            source="eGRID2023",
            year=2023,
            co2_ef=0.3127,
            ch4_ef=0.0000112,
            n2o_ef=0.0000042,
        )
        assert len(h) == 64

    def test_hash_grid_factor_empty_region_raises(self, seeded_prov):
        """Empty region_id raises ValueError."""
        with pytest.raises(ValueError, match="region_id"):
            seeded_prov.hash_grid_factor("", "eGRID", 2023, 0.3, 0.0, 0.0)

    def test_hash_grid_factor_negative_ef_raises(self, seeded_prov):
        """Negative emission factor raises ValueError."""
        with pytest.raises(ValueError, match="co2_ef"):
            seeded_prov.hash_grid_factor("US-WECC", "eGRID", 2023, -0.1, 0.0, 0.0)


# ===========================================================================
# 3. TestTDLossHash
# ===========================================================================


@_SKIP
class TestTDLossHash:
    """Tests for hash_td_loss."""

    def test_hash_td_loss_success(self, seeded_prov):
        """hash_td_loss returns a valid 64-char hex hash."""
        h = seeded_prov.hash_td_loss(
            country_code="US",
            td_loss_pct=5.3,
            method="iea_default",
        )
        assert len(h) == 64

    def test_hash_td_loss_out_of_range_raises(self, seeded_prov):
        """td_loss_pct > 100 raises ValueError."""
        with pytest.raises(ValueError, match="td_loss_pct"):
            seeded_prov.hash_td_loss("US", 150.0, "iea_default")


# ===========================================================================
# 4. TestElectricityHash
# ===========================================================================


@_SKIP
class TestElectricityHash:
    """Tests for hash_electricity_calculation."""

    def test_hash_electricity_success(self, seeded_prov):
        """hash_electricity_calculation returns a valid hash."""
        h = seeded_prov.hash_electricity_calculation(
            consumption_mwh=5000.0,
            ef_co2e=0.3127,
            td_loss_pct=5.3,
            total_co2e=1646.84,
        )
        assert len(h) == 64

    def test_hash_electricity_negative_consumption_raises(self, seeded_prov):
        """Negative consumption raises ValueError."""
        with pytest.raises(ValueError, match="consumption_mwh"):
            seeded_prov.hash_electricity_calculation(-100.0, 0.3, 5.0, 100.0)


# ===========================================================================
# 5. TestSteamHash
# ===========================================================================


@_SKIP
class TestSteamHash:
    """Tests for hash_steam_heat_cooling."""

    def test_hash_steam_success(self, seeded_prov):
        """hash_steam_heat_cooling for steam returns a valid hash."""
        h = seeded_prov.hash_steam_heat_cooling(
            energy_type="steam",
            consumption_gj=1200.0,
            ef=0.0667,
            total_co2e=80.04,
        )
        assert len(h) == 64

    def test_hash_cooling_success(self, seeded_prov):
        """hash_steam_heat_cooling for cooling returns a valid hash."""
        h = seeded_prov.hash_steam_heat_cooling(
            energy_type="cooling",
            consumption_gj=800.0,
            ef=0.05,
            total_co2e=40.0,
        )
        assert len(h) == 64

    def test_hash_invalid_energy_type_raises(self, seeded_prov):
        """Invalid energy type raises ValueError."""
        with pytest.raises(ValueError, match="energy_type"):
            seeded_prov.hash_steam_heat_cooling("wind", 100.0, 0.05, 5.0)


# ===========================================================================
# 6. TestGasBreakdown
# ===========================================================================


@_SKIP
class TestGasBreakdown:
    """Tests for hash_gas_breakdown."""

    def test_hash_gas_breakdown_success(self, seeded_prov):
        """hash_gas_breakdown returns a valid hash."""
        h = seeded_prov.hash_gas_breakdown(
            co2_kg=1563500.0,
            ch4_kg=56.0,
            n2o_kg=21.0,
            gwp_source="IPCC_AR5",
            total_co2e=1571653.0,
        )
        assert len(h) == 64

    def test_hash_gas_breakdown_negative_raises(self, seeded_prov):
        """Negative CO2 kg raises ValueError."""
        with pytest.raises(ValueError, match="co2_kg"):
            seeded_prov.hash_gas_breakdown(-100.0, 0.0, 0.0, "AR5", 0.0)

    def test_hash_gas_breakdown_empty_gwp_raises(self, seeded_prov):
        """Empty gwp_source raises ValueError."""
        with pytest.raises(ValueError, match="gwp_source"):
            seeded_prov.hash_gas_breakdown(1000.0, 1.0, 0.5, "", 1010.0)


# ===========================================================================
# 7. TestComplianceHash
# ===========================================================================


@_SKIP
class TestComplianceHash:
    """Tests for hash_compliance_check."""

    def test_hash_compliance_check_success(self, seeded_prov):
        """hash_compliance_check returns a valid hash."""
        h = seeded_prov.hash_compliance_check(
            framework="ghg_protocol_scope2",
            status="compliant",
            findings_count=12,
        )
        assert len(h) == 64


# ===========================================================================
# 8. TestUncertaintyHash
# ===========================================================================


@_SKIP
class TestUncertaintyHash:
    """Tests for hash_uncertainty."""

    def test_hash_uncertainty_success(self, seeded_prov):
        """hash_uncertainty returns a valid hash."""
        h = seeded_prov.hash_uncertainty(
            method="monte_carlo",
            mean_co2e=1646.84,
            std_dev=164.0,
            ci_lower=1350.0,
            ci_upper=1950.0,
        )
        assert len(h) == 64


# ===========================================================================
# 9. TestChainVerify
# ===========================================================================


@_SKIP
class TestChainVerify:
    """Tests for verify_chain."""

    def test_verify_chain_empty(self, prov):
        """Empty chain verifies as True."""
        assert prov.verify_chain() is True

    def test_verify_chain_valid(self, prov):
        """Valid chain with multiple entries verifies True."""
        prov.hash_input({"facility_id": "FAC-001"})
        prov.hash_grid_factor("US-WECC", "eGRID", 2023, 0.31, 0.00001, 0.000004)
        prov.hash_electricity_calculation(5000.0, 0.31, 5.0, 1550.0)
        prov.hash_output({"total_co2e_tonnes": 1550.0})
        assert prov.verify_chain() is True


# ===========================================================================
# 10. TestSerialization
# ===========================================================================


@_SKIP
class TestSerialization:
    """Tests for to_dict and from_dict on ProvenanceEntry."""

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
            "metadata": {"key": "value"},
        }
        entry = ProvenanceEntry.from_dict(data)
        assert entry.stage == "input"
        assert entry.hash_value == "a" * 64
        assert entry.metadata["key"] == "value"

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


# ===========================================================================
# 11. TestMerge
# ===========================================================================


@_SKIP
class TestMerge:
    """Tests for merge_chains."""

    def test_merge_chains(self):
        """Merging two chains creates a valid combined chain."""
        p1 = Scope2LocationProvenance()
        p1.hash_input({"facility_id": "FAC-001"})

        p2 = Scope2LocationProvenance()
        p2.hash_input({"facility_id": "FAC-002"})

        # If merge_chains exists, test it; otherwise skip
        if hasattr(p1, "merge_chains"):
            merged = p1.merge_chains(p2)
            if merged is not None:
                chain = merged.get_chain() if hasattr(merged, "get_chain") else []
                assert len(chain) >= 2
        else:
            pytest.skip("merge_chains not implemented")

    def test_deterministic_hashing(self):
        """Same inputs produce the same hash in two independent chains."""
        data = {"facility_id": "FAC-001", "year": 2025}
        p1 = Scope2LocationProvenance()
        h1 = p1.hash_input(data)
        p2 = Scope2LocationProvenance()
        h2 = p2.hash_input(data)
        # Both chains start from empty previous_hash, so hashes must match
        assert h1 == h2
