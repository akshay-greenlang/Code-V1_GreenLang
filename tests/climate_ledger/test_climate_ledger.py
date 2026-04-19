# -*- coding: utf-8 -*-
"""
Unit tests for greenlang.climate_ledger v3 product facade.

Tests cover:
- ClimateLedger: record_entry, verify, export, write_run_record
- hashing: content_address, re-exported hash_data / hash_file / MerkleTree
- signing: re-exported sign_artifact / verify_artifact (import only)
- calculation: re-exported CalculationStep, CalculationProvenance, etc.
- __init__: top-level re-exports and __version__

Author: GreenLang Platform Team
Date: April 2026
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# =========================================================================
# Module imports under test
# =========================================================================


class TestTopLevelImports:
    """Verify that the top-level ``__init__`` exports all expected symbols."""

    def test_version(self):
        from greenlang.climate_ledger import __version__

        assert __version__ == "0.1.0"

    def test_climate_ledger_class_importable(self):
        from greenlang.climate_ledger import ClimateLedger

        assert ClimateLedger is not None

    def test_hashing_symbols_importable(self):
        from greenlang.climate_ledger import (
            content_address,
            hash_data,
            hash_file,
            MerkleTree,
        )

        assert callable(content_address)
        assert callable(hash_data)
        assert callable(hash_file)
        assert MerkleTree is not None

    def test_signing_symbols_importable(self):
        from greenlang.climate_ledger import sign_artifact, verify_artifact

        assert callable(sign_artifact)
        assert callable(verify_artifact)

    def test_calculation_symbols_importable(self):
        from greenlang.climate_ledger import (
            CalculationProvenance,
            CalculationStep,
            OperationType,
            ProvenanceMetadata,
            ProvenanceStorage,
            SQLiteProvenanceStorage,
            stable_hash,
        )

        assert CalculationStep is not None
        assert CalculationProvenance is not None
        assert ProvenanceMetadata is not None
        assert OperationType is not None
        assert callable(stable_hash)
        assert ProvenanceStorage is not None
        assert SQLiteProvenanceStorage is not None

    def test_all_list_complete(self):
        import greenlang.climate_ledger as cl

        expected = {
            "ClimateLedger",
            "content_address",
            "hash_data",
            "hash_file",
            "MerkleTree",
            "sign_artifact",
            "verify_artifact",
            "CalculationProvenance",
            "CalculationStep",
            "OperationType",
            "ProvenanceMetadata",
            "ProvenanceStorage",
            "SQLiteProvenanceStorage",
            "stable_hash",
        }
        assert set(cl.__all__) == expected


# =========================================================================
# ClimateLedger tests
# =========================================================================


class TestClimateLedger:
    """Tests for the ``ClimateLedger`` facade class."""

    def _make_ledger(self, agent_name: str = "test-agent") -> Any:
        from greenlang.climate_ledger.ledger import ClimateLedger

        return ClimateLedger(agent_name=agent_name)

    # -- construction ------------------------------------------------------

    def test_init_default_backend(self):
        ledger = self._make_ledger()
        assert ledger.agent_name == "test-agent"
        assert ledger.storage_backend == "memory"
        assert ledger.entry_count == 0
        assert ledger.entity_count == 0

    def test_init_custom_backend_label(self):
        from greenlang.climate_ledger.ledger import ClimateLedger

        ledger = ClimateLedger(agent_name="x", storage_backend="sqlite")
        assert ledger.storage_backend == "sqlite"

    # -- record_entry ------------------------------------------------------

    def test_record_entry_returns_chain_hash(self):
        ledger = self._make_ledger()
        chain_hash = ledger.record_entry("emission", "e-001", "calculate", "abc123")
        assert isinstance(chain_hash, str)
        assert len(chain_hash) == 64  # SHA-256 hex

    def test_record_entry_increments_counts(self):
        ledger = self._make_ledger()
        ledger.record_entry("emission", "e-001", "ingest", "aaa")
        assert ledger.entry_count == 1
        assert ledger.entity_count == 1

        ledger.record_entry("factor", "f-001", "lookup", "bbb")
        assert ledger.entry_count == 2
        assert ledger.entity_count == 2

    def test_record_entry_same_entity_multiple_ops(self):
        ledger = self._make_ledger()
        h1 = ledger.record_entry("emission", "e-001", "ingest", "aaa")
        h2 = ledger.record_entry("emission", "e-001", "validate", "bbb")
        assert h1 != h2
        assert ledger.entity_count == 1
        assert ledger.entry_count == 2

    def test_record_entry_with_metadata(self):
        ledger = self._make_ledger()
        meta = {"framework": "GHG Protocol", "version": "2024"}
        chain_hash = ledger.record_entry(
            "emission", "e-001", "calculate", "abc123", metadata=meta
        )
        assert isinstance(chain_hash, str)
        assert len(chain_hash) == 64

    def test_record_entry_without_metadata(self):
        ledger = self._make_ledger()
        chain_hash = ledger.record_entry(
            "emission", "e-001", "calculate", "abc123", metadata=None
        )
        assert isinstance(chain_hash, str)

    # -- verify ------------------------------------------------------------

    def test_verify_empty_entity(self):
        ledger = self._make_ledger()
        valid, chain = ledger.verify("nonexistent")
        assert valid is True
        assert chain == []

    def test_verify_single_entry(self):
        ledger = self._make_ledger()
        ledger.record_entry("emission", "e-001", "calculate", "abc123")
        valid, chain = ledger.verify("e-001")
        assert valid is True
        assert len(chain) == 1

    def test_verify_multi_entry_chain(self):
        ledger = self._make_ledger()
        ledger.record_entry("emission", "e-001", "ingest", "aaa")
        ledger.record_entry("emission", "e-001", "validate", "bbb")
        ledger.record_entry("emission", "e-001", "calculate", "ccc")
        valid, chain = ledger.verify("e-001")
        assert valid is True
        assert len(chain) == 3

    def test_verify_returns_entries_in_order(self):
        ledger = self._make_ledger()
        ledger.record_entry("emission", "e-001", "ingest", "aaa")
        ledger.record_entry("emission", "e-001", "calculate", "bbb")
        _, chain = ledger.verify("e-001")
        assert chain[0]["action"] == "ingest"
        assert chain[1]["action"] == "calculate"

    # -- export ------------------------------------------------------------

    def test_export_single_entity(self):
        ledger = self._make_ledger()
        ledger.record_entry("emission", "e-001", "ingest", "aaa")
        result = ledger.export(entity_id="e-001")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["entity_id"] == "e-001"

    def test_export_global(self):
        ledger = self._make_ledger()
        ledger.record_entry("emission", "e-001", "ingest", "aaa")
        ledger.record_entry("factor", "f-001", "lookup", "bbb")
        result = ledger.export()
        assert isinstance(result, dict)
        assert result["agent_name"] == "test-agent"
        assert result["entry_count"] == 2
        assert result["entity_count"] == 2
        assert "entries" in result

    def test_export_empty_entity(self):
        ledger = self._make_ledger()
        result = ledger.export(entity_id="nonexistent")
        assert result == []

    def test_export_unsupported_format_raises(self):
        ledger = self._make_ledger()
        with pytest.raises(ValueError, match="Unsupported export format"):
            ledger.export(format="xml")

    # -- write_run_record --------------------------------------------------

    def test_write_run_record_creates_file(self):
        ledger = self._make_ledger()

        result = SimpleNamespace(
            success=True,
            outputs={"total": 42.0},
            metrics={"rows": 100},
        )
        ctx = SimpleNamespace(
            pipeline_spec={"name": "test-pipeline"},
            inputs={"file": "data.csv"},
            config={"mode": "full"},
            artifacts_map={},
            versions={"agent": "1.0.0"},
            sbom_path=None,
            signatures=[],
            backend="local",
            profile="dev",
            environment={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "run.json"
            written = ledger.write_run_record(result, ctx, output_path=out_path)
            assert written.exists()

            with open(written) as f:
                data = json.load(f)

            assert data["kind"] == "greenlang-run-ledger"
            assert "spec" in data
            assert data["metadata"]["status"] == "success"

    def test_write_run_record_default_path(self):
        """When output_path is None the underlying function defaults to out/run.json."""
        ledger = self._make_ledger()
        result = SimpleNamespace(success=True, outputs={}, metrics={})
        ctx = SimpleNamespace(
            pipeline_spec={}, inputs={}, config={},
            artifacts_map={}, versions={}, sbom_path=None,
            signatures=[], backend="local", profile="dev", environment={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Override cwd so "out/run.json" lands in our temp dir
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                written = ledger.write_run_record(result, ctx)
                assert written.exists()
            finally:
                os.chdir(old_cwd)

    # -- repr --------------------------------------------------------------

    def test_repr(self):
        ledger = self._make_ledger("my-agent")
        ledger.record_entry("x", "x-1", "op", "hash")
        r = repr(ledger)
        assert "my-agent" in r
        assert "entries=1" in r
        assert "entities=1" in r


# =========================================================================
# Hashing module tests
# =========================================================================


class TestHashing:
    """Tests for ``greenlang.climate_ledger.hashing``."""

    def test_content_address_bytes(self):
        from greenlang.climate_ledger.hashing import content_address

        expected = hashlib.sha256(b"hello").hexdigest()
        assert content_address(b"hello") == expected

    def test_content_address_str(self):
        from greenlang.climate_ledger.hashing import content_address

        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert content_address("hello") == expected

    def test_content_address_str_and_bytes_same(self):
        from greenlang.climate_ledger.hashing import content_address

        assert content_address("hello") == content_address(b"hello")

    def test_content_address_sha512(self):
        from greenlang.climate_ledger.hashing import content_address

        expected = hashlib.sha512(b"data").hexdigest()
        result = content_address(b"data", algorithm="sha512")
        assert result == expected

    def test_content_address_invalid_algorithm(self):
        from greenlang.climate_ledger.hashing import content_address

        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            content_address(b"data", algorithm="nonexistent_algo_xyz")

    def test_hash_data_reexport(self):
        from greenlang.climate_ledger.hashing import hash_data
        from greenlang.utilities.provenance.hashing import hash_data as original

        assert hash_data is original

    def test_hash_file_reexport(self):
        from greenlang.climate_ledger.hashing import hash_file
        from greenlang.utilities.provenance.hashing import hash_file as original

        assert hash_file is original

    def test_merkle_tree_reexport(self):
        from greenlang.climate_ledger.hashing import MerkleTree
        from greenlang.utilities.provenance.hashing import MerkleTree as original

        assert MerkleTree is original

    def test_hash_file_end_to_end(self):
        from greenlang.climate_ledger.hashing import hash_file

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            f.flush()
            tmp_path = f.name

        try:
            result = hash_file(tmp_path)
            assert "hash_value" in result
            assert result["hash_algorithm"] == "SHA256"
            assert len(result["hash_value"]) == 64
        finally:
            os.unlink(tmp_path)

    def test_merkle_tree_end_to_end(self):
        from greenlang.climate_ledger.hashing import MerkleTree

        tree = MerkleTree()
        tree.add_leaf("leaf-a")
        tree.add_leaf("leaf-b")
        tree.add_leaf("leaf-c")
        root = tree.build()

        assert isinstance(root, str)
        assert len(root) == 64

        proof = tree.get_proof(1)
        assert tree.verify_proof("leaf-b", 1, proof, root) is True
        assert tree.verify_proof("leaf-x", 1, proof, root) is False


# =========================================================================
# Signing module tests
# =========================================================================


class TestSigning:
    """Tests for ``greenlang.climate_ledger.signing``."""

    def test_sign_artifact_reexport(self):
        from greenlang.climate_ledger.signing import sign_artifact
        from greenlang.utilities.provenance.signing import (
            sign_artifact as original,
        )

        assert sign_artifact is original

    def test_verify_artifact_reexport(self):
        from greenlang.climate_ledger.signing import verify_artifact
        from greenlang.utilities.provenance.signing import (
            verify_artifact as original,
        )

        assert verify_artifact is original

    def test_all_list(self):
        import greenlang.climate_ledger.signing as mod

        assert set(mod.__all__) == {"sign_artifact", "verify_artifact"}


# =========================================================================
# Calculation module tests
# =========================================================================


class TestCalculation:
    """Tests for ``greenlang.climate_ledger.calculation``."""

    def test_calculation_step_reexport(self):
        from greenlang.climate_ledger.calculation import CalculationStep
        from greenlang.execution.core.provenance.calculation_provenance import (
            CalculationStep as original,
        )

        assert CalculationStep is original

    def test_calculation_provenance_reexport(self):
        from greenlang.climate_ledger.calculation import CalculationProvenance
        from greenlang.execution.core.provenance.calculation_provenance import (
            CalculationProvenance as original,
        )

        assert CalculationProvenance is original

    def test_operation_type_reexport(self):
        from greenlang.climate_ledger.calculation import OperationType
        from greenlang.execution.core.provenance.calculation_provenance import (
            OperationType as original,
        )

        assert OperationType is original

    def test_provenance_metadata_reexport(self):
        from greenlang.climate_ledger.calculation import ProvenanceMetadata
        from greenlang.execution.core.provenance.calculation_provenance import (
            ProvenanceMetadata as original,
        )

        assert ProvenanceMetadata is original

    def test_stable_hash_reexport(self):
        from greenlang.climate_ledger.calculation import stable_hash
        from greenlang.execution.core.provenance.calculation_provenance import (
            stable_hash as original,
        )

        assert stable_hash is original

    def test_provenance_storage_reexport(self):
        from greenlang.climate_ledger.calculation import ProvenanceStorage
        from greenlang.execution.core.provenance.storage import (
            ProvenanceStorage as original,
        )

        assert ProvenanceStorage is original

    def test_sqlite_storage_reexport(self):
        from greenlang.climate_ledger.calculation import SQLiteProvenanceStorage
        from greenlang.execution.core.provenance.storage import (
            SQLiteProvenanceStorage as original,
        )

        assert SQLiteProvenanceStorage is original

    def test_calculation_provenance_create_and_finalize(self):
        """Ensure the re-exported class works end-to-end."""
        from greenlang.climate_ledger.calculation import (
            CalculationProvenance,
            OperationType,
        )

        prov = CalculationProvenance.create(
            agent_name="TestCalc",
            agent_version="1.0.0",
            calculation_type="test_calc",
            input_data={"value": 42},
        )

        prov.add_step(
            operation=OperationType.MULTIPLY,
            description="Double the value",
            inputs={"value": 42},
            output=84,
            formula="result = value * 2",
        )

        prov.finalize(output_data={"result": 84})

        assert prov.output_data == {"result": 84}
        assert prov.output_hash is not None
        assert len(prov.steps) == 1
        assert prov.steps[0].step_number == 1

    def test_all_list(self):
        import greenlang.climate_ledger.calculation as mod

        expected = {
            "CalculationStep",
            "CalculationProvenance",
            "ProvenanceMetadata",
            "OperationType",
            "stable_hash",
            "ProvenanceStorage",
            "SQLiteProvenanceStorage",
        }
        assert set(mod.__all__) == expected


# =========================================================================
# Chain integrity tests
# =========================================================================


class TestChainIntegrity:
    """Integration-style tests verifying chain hashing works end-to-end
    through the Climate Ledger facade."""

    def test_different_agents_produce_different_genesis(self):
        from greenlang.climate_ledger.ledger import ClimateLedger

        l1 = ClimateLedger(agent_name="agent-a")
        l2 = ClimateLedger(agent_name="agent-b")
        assert l1.tracker._GENESIS_HASH != l2.tracker._GENESIS_HASH

    def test_chain_hash_deterministic(self):
        from greenlang.climate_ledger.ledger import ClimateLedger

        l1 = ClimateLedger(agent_name="determinism-test")
        l2 = ClimateLedger(agent_name="determinism-test")

        # Same genesis hash for same agent name
        assert l1.tracker._GENESIS_HASH == l2.tracker._GENESIS_HASH

    def test_full_round_trip(self):
        """Record several entries, verify, export, and check consistency."""
        from greenlang.climate_ledger.ledger import ClimateLedger

        ledger = ClimateLedger(agent_name="round-trip")

        hashes = []
        for i in range(5):
            h = ledger.record_entry(
                "test", f"t-{i:03d}", "process", f"hash_{i}"
            )
            hashes.append(h)

        # All hashes unique
        assert len(set(hashes)) == 5

        # All entities verify
        for i in range(5):
            ok, chain = ledger.verify(f"t-{i:03d}")
            assert ok is True
            assert len(chain) == 1

        # Global export
        export = ledger.export()
        assert export["entry_count"] == 5
        assert export["entity_count"] == 5
        assert len(export["entries"]) == 5
