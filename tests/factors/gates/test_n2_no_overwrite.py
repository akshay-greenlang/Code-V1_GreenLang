# -*- coding: utf-8 -*-
"""
N2 — Factor rows are immutable; changes create new, chain-linked versions.

Mutating an existing ``factor_version`` in place must raise
:class:`greenlang.factors.quality.versioning.VersioningError` (the
repo's canonical FactorImmutableError). Every new version chains
through ``compute_chain_hash`` with ``previous_chain_hash`` populated
and SHA-256 chain_hash correct.

The SQLite backend already installs BEFORE-UPDATE / BEFORE-DELETE
triggers (see ``versioning._SCHEMA``) — this test exercises them and
verifies the hash chain is recomputable end-to-end.

Run standalone::

    pytest tests/factors/gates/test_n2_no_overwrite.py -v
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from greenlang.factors.quality.versioning import (
    FactorVersionChain,
    VersioningError,
    compute_chain_hash,
)


# ---------------------------------------------------------------------------
# Fixtures local to this gate
# ---------------------------------------------------------------------------


@pytest.fixture()
def chain(tmp_path: Path) -> FactorVersionChain:
    """Fresh SQLite-backed chain per test, with autoclose."""
    db = tmp_path / "n2_chain.sqlite"
    c = FactorVersionChain(db)
    yield c
    try:
        c.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Immutability — UPDATE and DELETE must fail.
# ---------------------------------------------------------------------------


class TestN2ImmutableRows:
    """Existing version rows are append-only: UPDATE and DELETE fail."""

    def test_update_existing_version_raises(self, chain: FactorVersionChain):
        chain.append(
            factor_id="EF:US:diesel:2024:v1",
            factor_version="1.0.0",
            content_hash="a" * 64,
            changed_by="ingest",
            change_reason="initial",
        )
        # Poke the underlying SQLite directly — the production code shouldn't
        # offer any surface that mutates, but if someone adds one, this test
        # catches it via the DB-level trigger. SQLite raises DatabaseError
        # (IntegrityError or OperationalError depending on RAISE(ABORT)).
        with pytest.raises(sqlite3.DatabaseError) as exc_info:
            chain._conn.execute(
                "UPDATE factor_version_chain SET content_hash = ? WHERE factor_version = ?",
                ("b" * 64, "1.0.0"),
            )
        assert "append-only" in str(exc_info.value).lower(), (
            "N2 violation: SQLite UPDATE on factor_version_chain succeeded or "
            "raised an unexpected error. Expected the BEFORE UPDATE trigger to "
            "abort with 'factor_version_chain is append-only'. "
            f"Got: {exc_info.value!r}"
        )

    def test_delete_existing_version_raises(self, chain: FactorVersionChain):
        chain.append(
            factor_id="EF:US:diesel:2024:v1",
            factor_version="1.0.0",
            content_hash="a" * 64,
            changed_by="ingest",
            change_reason="initial",
        )
        with pytest.raises(sqlite3.DatabaseError) as exc_info:
            chain._conn.execute(
                "DELETE FROM factor_version_chain WHERE factor_version = ?",
                ("1.0.0",),
            )
        assert "append-only" in str(exc_info.value).lower(), (
            "N2 violation: DELETE on factor_version_chain was not blocked. The "
            "BEFORE DELETE trigger must abort every delete. "
            f"Got: {exc_info.value!r}"
        )

    def test_reappending_same_version_raises_factor_immutable_error(
        self, chain: FactorVersionChain
    ):
        """Attempting to append the same (factor_id, factor_version) twice
        raises :class:`VersioningError` (the repo's FactorImmutableError)."""
        chain.append(
            factor_id="EF:US:diesel:2024:v1",
            factor_version="1.0.0",
            content_hash="a" * 64,
            changed_by="ingest",
            change_reason="initial",
        )
        with pytest.raises(VersioningError) as exc_info:
            chain.append(
                factor_id="EF:US:diesel:2024:v1",
                factor_version="1.0.0",          # same version
                content_hash="b" * 64,           # different content
                changed_by="ingest-2",
                change_reason="attempt to overwrite",
            )
        msg = str(exc_info.value).lower()
        assert "1.0.0" in msg and "already" in msg, (
            "N2 violation: re-appending an existing factor_version must raise "
            "VersioningError (the FactorImmutableError surface) with a message "
            f"that names the conflicting version. Got: {exc_info.value!r}"
        )


# ---------------------------------------------------------------------------
# Chain linkage — previous_chain_hash + chain_hash correct.
# ---------------------------------------------------------------------------


class TestN2ChainHashCorrect:
    """New versions chain with previous_chain_hash populated; SHA-256 checks out."""

    def test_first_version_has_no_predecessor(self, chain: FactorVersionChain):
        entry = chain.append(
            factor_id="EF:US:diesel:2024:v1",
            factor_version="1.0.0",
            content_hash="a" * 64,
            changed_by="ingest",
            change_reason="initial",
        )
        assert entry.previous_version is None, (
            "N2 violation: the first version must have previous_version=None. "
            f"Got {entry.previous_version!r}"
        )
        assert entry.previous_chain_hash is None, (
            "N2 violation: the first version must have previous_chain_hash=None. "
            f"Got {entry.previous_chain_hash!r}"
        )

        expected = compute_chain_hash(
            factor_id="EF:US:diesel:2024:v1",
            factor_version="1.0.0",
            content_hash="a" * 64,
            previous_chain_hash=None,
        )
        assert entry.chain_hash == expected, (
            "N2 violation: chain_hash does not match SHA-256(factor_id | "
            "factor_version | content_hash | previous_chain_hash). "
            f"Got {entry.chain_hash!r}, expected {expected!r}"
        )
        assert len(entry.chain_hash) == 64, (
            "N2 violation: chain_hash is not a 64-char SHA-256 hex digest. "
            f"Got length={len(entry.chain_hash)}"
        )

    def test_second_version_chains_correctly(self, chain: FactorVersionChain):
        first = chain.append(
            factor_id="EF:US:diesel:2024:v1",
            factor_version="1.0.0",
            content_hash="a" * 64,
            changed_by="ingest",
            change_reason="initial",
        )
        second = chain.append(
            factor_id="EF:US:diesel:2024:v1",
            factor_version="1.0.1",
            content_hash="b" * 64,
            changed_by="editor",
            change_reason="GWP refresh AR5 -> AR6",
        )

        assert second.previous_version == "1.0.0", (
            "N2 violation: new version did not chain onto its predecessor. "
            f"previous_version={second.previous_version!r}"
        )
        assert second.previous_chain_hash == first.chain_hash, (
            "N2 violation: previous_chain_hash does not point to prior "
            f"chain_hash. Expected {first.chain_hash!r}, got {second.previous_chain_hash!r}"
        )

        expected = compute_chain_hash(
            factor_id="EF:US:diesel:2024:v1",
            factor_version="1.0.1",
            content_hash="b" * 64,
            previous_chain_hash=first.chain_hash,
        )
        assert second.chain_hash == expected, (
            "N2 violation: second version's chain_hash is not the correct SHA-256 "
            f"of (factor_id, factor_version, content_hash, prev_hash). "
            f"Got {second.chain_hash!r}, expected {expected!r}"
        )

    def test_verify_chain_detects_integrity(self, chain: FactorVersionChain):
        for i, content in enumerate(("a", "b", "c"), start=1):
            chain.append(
                factor_id="EF:US:diesel:2024:v1",
                factor_version=f"1.0.{i - 1}",
                content_hash=content * 64,
                changed_by="editor",
                change_reason=f"step {i}",
            )
        assert chain.verify_chain("EF:US:diesel:2024:v1") is True, (
            "N2 violation: verify_chain() returned False on an untampered chain. "
            "Recomputing chain_hash from the stored fields must match the stored "
            "chain_hash at every row."
        )
