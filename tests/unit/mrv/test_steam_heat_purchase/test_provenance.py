# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-011 Steam/Heat Purchase Agent Provenance Tracking.

Tests SteamHeatPurchaseProvenance singleton, SHA-256 chain hashing, all 19
valid stages, chain lifecycle (create/add/seal/verify), chain inspection,
export, domain-specific hash_* helper methods, and edge cases.

Target: 65 tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import threading

import pytest

from greenlang.agents.mrv.steam_heat_purchase.provenance import (
    ProvenanceEntry,
    SteamHeatPurchaseProvenance,
    VALID_STAGES,
    STAGE_ORDER,
    STAGE_COUNT,
    get_provenance,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_provenance_singleton():
    """Reset provenance singleton before and after each test."""
    SteamHeatPurchaseProvenance.reset()
    yield
    SteamHeatPurchaseProvenance.reset()


@pytest.fixture
def prov() -> SteamHeatPurchaseProvenance:
    """Return a fresh SteamHeatPurchaseProvenance instance."""
    return SteamHeatPurchaseProvenance()


@pytest.fixture
def prov_with_chain(prov):
    """Return a provenance instance with a pre-created chain."""
    chain_id = prov.create_chain("test-chain-001")
    return prov, chain_id


# ===========================================================================
# ProvenanceEntry Tests
# ===========================================================================


class TestProvenanceEntry:
    """Tests for the ProvenanceEntry frozen dataclass."""

    def test_creation_with_required_fields(self):
        """ProvenanceEntry can be created with all required fields."""
        entry = ProvenanceEntry(
            stage="REQUEST_RECEIVED",
            hash_value="a" * 64,
            timestamp="2026-01-01T00:00:00+00:00",
            previous_hash="",
        )
        assert entry.stage == "REQUEST_RECEIVED"
        assert len(entry.hash_value) == 64
        assert entry.previous_hash == ""

    def test_default_metadata_is_empty_dict(self):
        """Default metadata is an empty dictionary."""
        entry = ProvenanceEntry(
            stage="INPUT_VALIDATED",
            hash_value="b" * 64,
            timestamp="2026-01-01T00:00:00+00:00",
            previous_hash="a" * 64,
        )
        assert entry.metadata == {}

    def test_default_chain_id_is_empty_string(self):
        """Default chain_id is an empty string."""
        entry = ProvenanceEntry(
            stage="STEAM_CALCULATED",
            hash_value="c" * 64,
            timestamp="2026-01-01T00:00:00+00:00",
            previous_hash="b" * 64,
        )
        assert entry.chain_id == ""

    def test_frozen_immutability(self):
        """ProvenanceEntry is frozen and cannot be modified in-place."""
        entry = ProvenanceEntry(
            stage="STEAM_CALCULATED",
            hash_value="c" * 64,
            timestamp="2026-01-01T00:00:00+00:00",
            previous_hash="b" * 64,
        )
        with pytest.raises(AttributeError):
            entry.stage = "MODIFIED"

    def test_to_dict_returns_all_fields(self):
        """to_dict returns dictionary with all six fields."""
        entry = ProvenanceEntry(
            stage="CHP_ALLOCATED",
            hash_value="d" * 64,
            timestamp="2026-01-01T00:00:00+00:00",
            previous_hash="c" * 64,
            metadata={"allocation": 0.6},
            chain_id="chain-x",
        )
        d = entry.to_dict()
        assert d["stage"] == "CHP_ALLOCATED"
        assert d["hash_value"] == "d" * 64
        assert d["timestamp"] == "2026-01-01T00:00:00+00:00"
        assert d["previous_hash"] == "c" * 64
        assert d["metadata"] == {"allocation": 0.6}
        assert d["chain_id"] == "chain-x"

    def test_from_dict_roundtrip(self):
        """from_dict can reconstruct an entry from to_dict output."""
        original = ProvenanceEntry(
            stage="BIOGENIC_SEPARATED",
            hash_value="e" * 64,
            timestamp="2026-02-01T12:00:00+00:00",
            previous_hash="d" * 64,
            metadata={"fossil": 1000, "biogenic": 500},
            chain_id="chain-y",
        )
        reconstructed = ProvenanceEntry.from_dict(original.to_dict())
        assert reconstructed.stage == original.stage
        assert reconstructed.hash_value == original.hash_value
        assert reconstructed.previous_hash == original.previous_hash
        assert reconstructed.chain_id == original.chain_id

    def test_from_dict_rejects_non_dict(self):
        """from_dict raises TypeError for non-dict input."""
        with pytest.raises(TypeError, match="Expected dict"):
            ProvenanceEntry.from_dict("not a dict")


# ===========================================================================
# Singleton Tests
# ===========================================================================


class TestSteamHeatPurchaseProvenanceSingleton:
    """Tests for the singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """Multiple instantiations return the same object."""
        p1 = SteamHeatPurchaseProvenance()
        p2 = SteamHeatPurchaseProvenance()
        assert p1 is p2

    def test_reset_creates_new_instance(self):
        """After reset, a new instance is created."""
        p1 = SteamHeatPurchaseProvenance()
        chain_id = p1.create_chain("before-reset")
        SteamHeatPurchaseProvenance.reset()
        p2 = SteamHeatPurchaseProvenance()
        assert p1 is not p2
        with pytest.raises(ValueError):
            p2.get_chain(chain_id)

    def test_class_constants(self, prov):
        """Class constants are correct."""
        assert prov.AGENT_ID == "AGENT-MRV-011"
        assert prov.AGENT_NAME == "Steam/Heat Purchase"
        assert prov.PREFIX == "gl_shp"

    def test_get_provenance_returns_singleton(self):
        """Module-level get_provenance returns the singleton."""
        p = get_provenance()
        assert isinstance(p, SteamHeatPurchaseProvenance)
        p2 = get_provenance()
        assert p is p2


# ===========================================================================
# Chain Lifecycle Tests
# ===========================================================================


class TestChainLifecycle:
    """Tests for create_chain, add_stage, seal_chain, verify_chain."""

    def test_create_chain_returns_chain_id(self, prov):
        """create_chain returns the provided calc_id as chain_id."""
        chain_id = prov.create_chain("calc-001")
        assert chain_id == "calc-001"

    def test_create_chain_empty_id_raises(self, prov):
        """create_chain raises ValueError for empty string ID."""
        with pytest.raises(ValueError, match="must not be empty"):
            prov.create_chain("")

    def test_create_chain_duplicate_id_raises(self, prov):
        """create_chain raises ValueError for duplicate chain IDs."""
        prov.create_chain("dup-001")
        with pytest.raises(ValueError, match="already exists"):
            prov.create_chain("dup-001")

    def test_add_stage_returns_sha256_hash(self, prov_with_chain):
        """add_stage returns a 64-character hex SHA-256 hash string."""
        prov, chain_id = prov_with_chain
        h = prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "FAC-001"})
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)  # validate hex format

    def test_add_stage_all_19_valid_stages_accepted(self, prov):
        """All 19 VALID_STAGES are accepted without error."""
        chain_id = prov.create_chain("all-stages")
        hashes = []
        for stage in STAGE_ORDER:
            h = prov.add_stage(chain_id, stage, {"stage": stage})
            hashes.append(h)
            assert len(h) == 64
        assert len(hashes) == 19
        assert len(set(hashes)) == 19

    def test_add_stage_unknown_stage_accepted_with_warning(self, prov_with_chain):
        """Unknown stage names are accepted (warning only)."""
        prov, chain_id = prov_with_chain
        h = prov.add_stage(chain_id, "CUSTOM_STAGE_XYZ", {"data": 1})
        assert len(h) == 64

    def test_add_stage_to_nonexistent_chain_raises(self, prov):
        """add_stage raises ValueError for a non-existent chain_id."""
        with pytest.raises(ValueError, match="does not exist"):
            prov.add_stage("no-such-chain", "REQUEST_RECEIVED", {})

    def test_add_stage_to_sealed_chain_raises(self, prov_with_chain):
        """add_stage raises ValueError after the chain is sealed."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        prov.seal_chain(chain_id)
        with pytest.raises(ValueError, match="already been sealed"):
            prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 100})

    def test_add_stage_empty_stage_raises(self, prov_with_chain):
        """add_stage raises ValueError for empty stage string."""
        prov, chain_id = prov_with_chain
        with pytest.raises(ValueError, match="must not be empty"):
            prov.add_stage(chain_id, "", {"data": 1})

    def test_add_stage_non_dict_data_raises(self, prov_with_chain):
        """add_stage raises TypeError for non-dict data."""
        prov, chain_id = prov_with_chain
        with pytest.raises(TypeError, match="must be a dict"):
            prov.add_stage(chain_id, "REQUEST_RECEIVED", "not-a-dict")

    def test_seal_chain_returns_final_hash(self, prov_with_chain):
        """seal_chain returns a 64-character SHA-256 hex hash."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 1000})
        seal_hash = prov.seal_chain(chain_id)
        assert isinstance(seal_hash, str)
        assert len(seal_hash) == 64

    def test_seal_chain_marks_chain_as_sealed(self, prov_with_chain):
        """After sealing, is_sealed returns True."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        assert prov.is_sealed(chain_id) is False
        prov.seal_chain(chain_id)
        assert prov.is_sealed(chain_id) is True

    def test_seal_already_sealed_chain_raises(self, prov_with_chain):
        """seal_chain raises ValueError if chain is already sealed."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        prov.seal_chain(chain_id)
        with pytest.raises(ValueError, match="already been sealed"):
            prov.seal_chain(chain_id)

    def test_seal_nonexistent_chain_raises(self, prov):
        """seal_chain raises ValueError for non-existent chain."""
        with pytest.raises(ValueError, match="does not exist"):
            prov.seal_chain("no-such-chain")

    def test_verify_chain_returns_true_for_valid_chain(self, prov_with_chain):
        """verify_chain returns True for a properly constructed chain."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        prov.add_stage(chain_id, "INPUT_VALIDATED", {"valid": True})
        prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 1000.0})
        prov.seal_chain(chain_id)
        assert prov.verify_chain(chain_id) is True

    def test_verify_chain_returns_false_for_tampered_chain(self, prov):
        """verify_chain detects tampering in the chain."""
        chain_id = prov.create_chain("tamper-test")
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 500})

        with prov._instance_lock:
            original = prov._chains[chain_id][0]
            tampered = ProvenanceEntry(
                stage=original.stage,
                hash_value="0" * 64,
                timestamp=original.timestamp,
                previous_hash=original.previous_hash,
                metadata=original.metadata,
                chain_id=original.chain_id,
            )
            prov._chains[chain_id][0] = tampered

        assert prov.verify_chain(chain_id) is False

    def test_verify_chain_empty_is_valid(self, prov_with_chain):
        """verify_chain returns True for an empty chain."""
        prov, chain_id = prov_with_chain
        assert prov.verify_chain(chain_id) is True

    def test_verify_chain_nonexistent_raises(self, prov):
        """verify_chain raises ValueError for non-existent chain."""
        with pytest.raises(ValueError, match="does not exist"):
            prov.verify_chain("no-such-chain")

    def test_verify_chain_detailed_valid(self, prov_with_chain):
        """verify_chain_detailed returns (True, None, -1) for valid chain."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"x": 1})
        is_valid, msg, idx = prov.verify_chain_detailed(chain_id)
        assert is_valid is True
        assert msg is None
        assert idx == -1


# ===========================================================================
# Chain Inspection Tests
# ===========================================================================


class TestChainInspection:
    """Tests for get_chain, get_chain_hash, get_chain_length, etc."""

    def test_get_chain_returns_list_of_dicts(self, prov_with_chain):
        """get_chain returns list of dictionaries."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 100})
        chain = prov.get_chain(chain_id)
        assert isinstance(chain, list)
        assert len(chain) == 2
        assert chain[0]["stage"] == "REQUEST_RECEIVED"
        assert chain[1]["stage"] == "STEAM_CALCULATED"

    def test_get_chain_nonexistent_raises(self, prov):
        """get_chain raises ValueError for non-existent chain."""
        with pytest.raises(ValueError, match="does not exist"):
            prov.get_chain("no-such-chain")

    def test_get_chain_hash_returns_final_hash(self, prov_with_chain):
        """get_chain_hash returns the last entry hash."""
        prov, chain_id = prov_with_chain
        h1 = prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        h2 = prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 100})
        chain_hash = prov.get_chain_hash(chain_id)
        assert chain_hash == h2
        assert chain_hash != h1

    def test_get_chain_hash_empty_chain_returns_empty_string(self, prov_with_chain):
        """get_chain_hash returns empty string for empty chain."""
        prov, chain_id = prov_with_chain
        assert prov.get_chain_hash(chain_id) == ""

    def test_get_chain_length(self, prov_with_chain):
        """get_chain_length returns correct number of entries."""
        prov, chain_id = prov_with_chain
        assert prov.get_chain_length(chain_id) == 0
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        assert prov.get_chain_length(chain_id) == 1
        prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 100})
        assert prov.get_chain_length(chain_id) == 2

    def test_get_stage_summary(self, prov_with_chain):
        """get_stage_summary returns correct counts per stage."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 100})
        prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 200})
        summary = prov.get_stage_summary(chain_id)
        assert summary["REQUEST_RECEIVED"] == 1
        assert summary["STEAM_CALCULATED"] == 2

    def test_is_sealed_false_before_seal(self, prov_with_chain):
        """is_sealed returns False before seal_chain is called."""
        prov, chain_id = prov_with_chain
        assert prov.is_sealed(chain_id) is False

    def test_list_chains(self, prov):
        """list_chains returns summary of all chains."""
        prov.create_chain("chain-a")
        prov.create_chain("chain-b")
        chains = prov.list_chains()
        assert len(chains) == 2
        chain_ids = {c["chain_id"] for c in chains}
        assert chain_ids == {"chain-a", "chain-b"}

    def test_chain_count(self, prov):
        """chain_count returns total number of chains."""
        assert prov.chain_count() == 0
        prov.create_chain("c1")
        assert prov.chain_count() == 1
        prov.create_chain("c2")
        assert prov.chain_count() == 2

    def test_get_entries_by_stage(self, prov_with_chain):
        """get_entries_by_stage filters correctly."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 100})
        prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 200})
        entries = prov.get_entries_by_stage(chain_id, "STEAM_CALCULATED")
        assert len(entries) == 2
        for e in entries:
            assert e.stage == "STEAM_CALCULATED"

    def test_get_entry_by_index(self, prov_with_chain):
        """get_entry_by_index returns correct entry."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 100})
        entry = prov.get_entry_by_index(chain_id, 0)
        assert entry.stage == "REQUEST_RECEIVED"
        entry_last = prov.get_entry_by_index(chain_id, -1)
        assert entry_last.stage == "STEAM_CALCULATED"

    def test_get_entry_by_index_out_of_range_returns_none(self, prov_with_chain):
        """get_entry_by_index returns None for out-of-range index."""
        prov, chain_id = prov_with_chain
        assert prov.get_entry_by_index(chain_id, 999) is None

    def test_get_latest_entry(self, prov_with_chain):
        """get_latest_entry returns the most recent entry."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        prov.add_stage(chain_id, "RESULT_ASSEMBLED", {"t": 500})
        latest = prov.get_latest_entry(chain_id)
        assert latest is not None
        assert latest.stage == "RESULT_ASSEMBLED"

    def test_get_latest_entry_empty_chain_returns_none(self, prov_with_chain):
        """get_latest_entry returns None for empty chain."""
        prov, chain_id = prov_with_chain
        assert prov.get_latest_entry(chain_id) is None


# ===========================================================================
# Compute Hash Tests
# ===========================================================================


class TestComputeHash:
    """Tests for the static compute_hash method."""

    def test_compute_hash_is_deterministic(self):
        """compute_hash returns the same hash for the same input."""
        data = {"key": "value", "number": 42}
        h1 = SteamHeatPurchaseProvenance.compute_hash(data)
        h2 = SteamHeatPurchaseProvenance.compute_hash(data)
        assert h1 == h2

    def test_compute_hash_returns_64_char_hex(self):
        """compute_hash returns a 64-character hex string."""
        h = SteamHeatPurchaseProvenance.compute_hash({"test": True})
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)

    def test_compute_hash_different_data_different_hash(self):
        """Different inputs produce different hashes."""
        h1 = SteamHeatPurchaseProvenance.compute_hash({"a": 1})
        h2 = SteamHeatPurchaseProvenance.compute_hash({"a": 2})
        assert h1 != h2

    def test_compute_hash_dict_key_order_does_not_matter(self):
        """Dict key insertion order does not affect the hash (sort_keys)."""
        h1 = SteamHeatPurchaseProvenance.compute_hash({"b": 2, "a": 1})
        h2 = SteamHeatPurchaseProvenance.compute_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_compute_hash_string_input(self):
        """compute_hash accepts raw string input."""
        h = SteamHeatPurchaseProvenance.compute_hash("plain string")
        assert len(h) == 64


# ===========================================================================
# Export Chain Tests
# ===========================================================================


class TestExportChain:
    """Tests for export_chain and related methods."""

    def test_export_chain_returns_complete_artifact(self, prov_with_chain):
        """export_chain returns dict with all required fields."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "FAC-001"})
        prov.add_stage(chain_id, "STEAM_CALCULATED", {"e": 1000})
        prov.seal_chain(chain_id)
        export = prov.export_chain(chain_id)

        assert export["agent_id"] == "AGENT-MRV-011"
        assert export["agent_name"] == "Steam/Heat Purchase"
        assert export["prefix"] == "gl_shp"
        assert export["chain_id"] == chain_id
        # 2 stages + seal = 3 entries
        assert export["chain_length"] == 3
        assert len(export["chain_hash"]) == 64
        assert export["sealed"] is True
        assert "entries" in export
        assert len(export["entries"]) == 3
        assert export["stages_defined"] == 19
        assert export["verification"]["is_valid"] is True

    def test_export_chain_nonexistent_raises(self, prov):
        """export_chain raises ValueError for non-existent chain."""
        with pytest.raises(ValueError, match="does not exist"):
            prov.export_chain("no-such-chain")

    def test_export_chain_json_is_valid_json(self, prov_with_chain):
        """export_chain_json returns parseable JSON."""
        prov, chain_id = prov_with_chain
        prov.add_stage(chain_id, "REQUEST_RECEIVED", {"f": "1"})
        json_str = prov.export_chain_json(chain_id)
        parsed = json.loads(json_str)
        assert parsed["chain_id"] == chain_id

    def test_to_dict_includes_all_chains(self, prov):
        """to_dict serializes all chains in the tracker."""
        prov.create_chain("c1")
        prov.create_chain("c2")
        prov.add_stage("c1", "REQUEST_RECEIVED", {"f": "1"})
        prov.seal_chain("c1")
        d = prov.to_dict()
        assert d["chain_count"] == 2
        assert d["sealed_count"] == 1
        assert "c1" in d["chains"]
        assert "c2" in d["chains"]


# ===========================================================================
# Domain-Specific Hash Helper Tests
# ===========================================================================


class TestDomainSpecificHashHelpers:
    """Tests for hash_request_received, hash_steam_calculated, etc."""

    def test_hash_request_received(self, prov_with_chain):
        """hash_request_received adds a REQUEST_RECEIVED stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_request_received(chain_id, {
            "facility_id": "FAC-001", "energy_type": "steam",
        })
        assert len(h) == 64
        chain = prov.get_chain(chain_id)
        assert chain[0]["stage"] == "REQUEST_RECEIVED"

    def test_hash_input_validated(self, prov_with_chain):
        """hash_input_validated adds an INPUT_VALIDATED stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_input_validated(chain_id, {
            "valid": True, "field_count": 12,
        })
        assert len(h) == 64
        chain = prov.get_chain(chain_id)
        assert chain[0]["stage"] == "INPUT_VALIDATED"

    def test_hash_facility_resolved(self, prov_with_chain):
        """hash_facility_resolved adds a FACILITY_RESOLVED stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_facility_resolved(chain_id, {
            "facility_name": "Plant A", "country": "US",
        })
        assert len(h) == 64

    def test_hash_supplier_resolved(self, prov_with_chain):
        """hash_supplier_resolved adds a SUPPLIER_RESOLVED stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_supplier_resolved(chain_id, {"supplier_id": "SUP-001"})
        assert len(h) == 64

    def test_hash_fuel_ef_retrieved(self, prov_with_chain):
        """hash_fuel_ef_retrieved adds a FUEL_EF_RETRIEVED stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_fuel_ef_retrieved(chain_id, {
            "fuel_type": "natural_gas", "co2_ef": "56.1",
        })
        assert len(h) == 64

    def test_hash_dh_ef_retrieved(self, prov_with_chain):
        """hash_dh_ef_retrieved adds a DH_EF_RETRIEVED stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_dh_ef_retrieved(chain_id, {"region": "denmark", "ef": "36.0"})
        assert len(h) == 64

    def test_hash_cooling_params_retrieved(self, prov_with_chain):
        """hash_cooling_params_retrieved adds correct stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_cooling_params_retrieved(chain_id, {"cop": 6.0})
        assert len(h) == 64

    def test_hash_chp_params_retrieved(self, prov_with_chain):
        """hash_chp_params_retrieved adds correct stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_chp_params_retrieved(chain_id, {"method": "efficiency"})
        assert len(h) == 64

    def test_hash_unit_converted(self, prov_with_chain):
        """hash_unit_converted adds a UNIT_CONVERTED stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_unit_converted(chain_id, {"from": "MWh", "to": "GJ"})
        assert len(h) == 64

    def test_hash_steam_calculated(self, prov_with_chain):
        """hash_steam_calculated adds a STEAM_CALCULATED stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_steam_calculated(chain_id, {"emissions_kg": 66500.0})
        assert len(h) == 64

    def test_hash_heating_calculated(self, prov_with_chain):
        """hash_heating_calculated adds a HEATING_CALCULATED stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_heating_calculated(chain_id, {"dh_factor": 72.0})
        assert len(h) == 64

    def test_hash_cooling_calculated(self, prov_with_chain):
        """hash_cooling_calculated adds a COOLING_CALCULATED stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_cooling_calculated(chain_id, {"cop": 6.0})
        assert len(h) == 64

    def test_hash_chp_allocated(self, prov_with_chain):
        """hash_chp_allocated adds a CHP_ALLOCATED stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_chp_allocated(chain_id, {"heat_pct": 0.6})
        assert len(h) == 64

    def test_hash_biogenic_separated(self, prov_with_chain):
        """hash_biogenic_separated adds a BIOGENIC_SEPARATED stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_biogenic_separated(chain_id, {"bio": 500.0})
        assert len(h) == 64

    def test_hash_gas_breakdown_computed(self, prov_with_chain):
        """hash_gas_breakdown_computed adds correct stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_gas_breakdown_computed(chain_id, {"co2": 1000})
        assert len(h) == 64

    def test_hash_uncertainty_quantified(self, prov_with_chain):
        """hash_uncertainty_quantified adds correct stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_uncertainty_quantified(chain_id, {"cv": 0.05})
        assert len(h) == 64

    def test_hash_compliance_checked(self, prov_with_chain):
        """hash_compliance_checked adds correct stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_compliance_checked(chain_id, {"ghg_protocol": True})
        assert len(h) == 64

    def test_hash_result_assembled(self, prov_with_chain):
        """hash_result_assembled adds correct stage."""
        prov, chain_id = prov_with_chain
        h = prov.hash_result_assembled(chain_id, {"total": 66500.0})
        assert len(h) == 64


# ===========================================================================
# Stage Constants Tests
# ===========================================================================


class TestStageConstants:
    """Tests for VALID_STAGES, STAGE_ORDER, STAGE_COUNT."""

    def test_valid_stages_has_19_entries(self):
        """VALID_STAGES contains exactly 19 stages."""
        assert len(VALID_STAGES) == 19

    def test_stage_order_has_19_entries(self):
        """STAGE_ORDER contains exactly 19 stages."""
        assert len(STAGE_ORDER) == 19

    def test_stage_count_is_19(self):
        """STAGE_COUNT equals 19."""
        assert STAGE_COUNT == 19

    def test_stage_order_matches_valid_stages(self):
        """All STAGE_ORDER entries are in VALID_STAGES."""
        for stage in STAGE_ORDER:
            assert stage in VALID_STAGES

    def test_valid_stages_matches_stage_order(self):
        """All VALID_STAGES are in STAGE_ORDER."""
        for stage in VALID_STAGES:
            assert stage in STAGE_ORDER

    def test_first_stage_is_request_received(self):
        """First stage in order is REQUEST_RECEIVED."""
        assert STAGE_ORDER[0] == "REQUEST_RECEIVED"

    def test_last_stage_is_provenance_sealed(self):
        """Last stage in order is PROVENANCE_SEALED."""
        assert STAGE_ORDER[-1] == "PROVENANCE_SEALED"
