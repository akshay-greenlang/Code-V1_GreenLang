# -*- coding: utf-8 -*-
"""Tests for scripts/factors_key_rotation.py — Ed25519 rotation CLI.

Philosophy
----------
The CLI talks to Vault and an audit sink. Both are mocked here so tests
never hit a real Vault server and never write to a real audit backend.
We assert that:

* ``rotate-plan`` reports current metadata + whether rotation is overdue.
* ``rotate-stage`` (live) generates a new keypair, canary-signs with both,
  verifies both, writes to ``.../next``, and emits an audit record.
* ``rotate-promote`` (live) moves ``next -> current``, archives the old
  current under ``.../archive/YYYYMMDD``, writes an allowlist with BOTH
  kids active during the grace window, and emits an audit record.
* ``rotate-retire-old`` drops kids whose grace window has expired.
* The signed-receipts tier->algorithm policy is untouched by rotation.
* Dry-run never mutates Vault.
* Vault unavailability causes each mutating subcommand to refuse —
  never crash, never corrupt.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# --------------------------------------------------------------------------- #
# Load the CLI module by path so the test does not depend on `scripts/`       #
# being importable as a package.                                              #
# --------------------------------------------------------------------------- #

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "factors_key_rotation.py"
)


def _load_rot_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "factors_key_rotation_under_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rot = _load_rot_module()


# --------------------------------------------------------------------------- #
# Mock Vault + audit sink.                                                     #
# --------------------------------------------------------------------------- #


class FakeVault:
    """In-memory stand-in for VaultClient."""

    def __init__(self, *, available: bool = True) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}
        self._available = available
        self.writes: List[str] = []
        self.deletes: List[str] = []

    def is_available(self) -> bool:
        return self._available

    def read(self, path: str) -> Optional[Dict[str, Any]]:
        if not self._available:
            raise rot.VaultUnavailableError("offline")
        value = self.store.get(path)
        return None if value is None else dict(value)

    def write(self, path: str, data: Dict[str, Any]) -> None:
        if not self._available:
            raise rot.VaultUnavailableError("offline")
        self.store[path] = dict(data)
        self.writes.append(path)

    def delete(self, path: str) -> None:
        if not self._available:
            raise rot.VaultUnavailableError("offline")
        self.store.pop(path, None)
        self.deletes.append(path)

    def whoami(self) -> Optional[str]:
        return "fake-vault-token"


class CapturingAudit:
    """Captures audit.emit(...) calls."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def emit(
        self,
        *,
        action: str,
        operator: str,
        result: str,
        details: Dict[str, Any],
    ) -> None:
        self.records.append(
            {
                "action": action,
                "operator": operator,
                "result": result,
                "details": dict(details),
            }
        )


# --------------------------------------------------------------------------- #
# Fixtures / helpers.                                                          #
# --------------------------------------------------------------------------- #


def _make_key(*, kid: str, age_days: float = 0.0) -> Dict[str, Any]:
    """Create a fresh Ed25519 keypair dict for seeding the fake vault."""
    pair = rot._generate_ed25519_pem()
    created = datetime.now(timezone.utc) - timedelta(days=age_days)
    return {
        "kid": kid,
        "created_at": created.isoformat(),
        "algorithm": "ed25519",
        "private_key_pem": pair["private_key_pem"],
        "public_key_pem": pair["public_key_pem"],
    }


# --------------------------------------------------------------------------- #
# rotate-plan                                                                  #
# --------------------------------------------------------------------------- #


class TestRotatePlan:
    def test_plan_reports_no_current_key(self) -> None:
        vault = FakeVault()
        audit = CapturingAudit()
        report = rot.run(["rotate-plan"], vault=vault, audit=audit)
        assert report.ok is True
        assert report.details.get("current") is None

    def test_plan_reports_within_cadence(self) -> None:
        vault = FakeVault()
        vault.store[rot.KEY_PATH_CURRENT] = _make_key(kid="kid-a", age_days=10)
        audit = CapturingAudit()
        report = rot.run(["rotate-plan"], vault=vault, audit=audit)
        assert report.ok is True
        assert report.details["overdue"] is False
        assert report.details["current_kid"] == "kid-a"
        assert report.details["age_days"] is not None

    def test_plan_flags_overdue(self) -> None:
        vault = FakeVault()
        vault.store[rot.KEY_PATH_CURRENT] = _make_key(
            kid="kid-old", age_days=rot.ROTATION_CADENCE_DAYS + 5
        )
        report = rot.run(["rotate-plan"], vault=vault, audit=CapturingAudit())
        assert report.ok is True
        assert report.details["overdue"] is True
        assert "Rotation overdue" in report.message

    def test_plan_handles_vault_unavailable(self) -> None:
        vault = FakeVault(available=False)
        report = rot.run(["rotate-plan"], vault=vault, audit=CapturingAudit())
        assert report.ok is False
        assert "Vault unavailable" in report.message


# --------------------------------------------------------------------------- #
# rotate-stage                                                                 #
# --------------------------------------------------------------------------- #


class TestRotateStage:
    def test_dry_run_does_not_mutate(self) -> None:
        vault = FakeVault()
        vault.store[rot.KEY_PATH_CURRENT] = _make_key(kid="kid-a")
        audit = CapturingAudit()
        report = rot.run(
            ["rotate-stage", "--dry-run"], vault=vault, audit=audit
        )
        assert report.ok is True
        assert report.dry_run is True
        assert rot.KEY_PATH_NEXT not in vault.store  # not staged
        assert audit.records == []

    def test_live_stage_generates_and_verifies_canary(self) -> None:
        vault = FakeVault()
        vault.store[rot.KEY_PATH_CURRENT] = _make_key(kid="kid-a")
        audit = CapturingAudit()
        report = rot.run(
            ["rotate-stage", "--live", "--operator", "alice"],
            vault=vault,
            audit=audit,
        )
        assert report.ok is True, report.message
        assert rot.KEY_PATH_NEXT in vault.store
        staged = vault.store[rot.KEY_PATH_NEXT]
        assert staged["kid"] != "kid-a"
        assert staged["algorithm"] == "ed25519"
        assert staged["private_key_pem"].startswith("-----BEGIN PRIVATE KEY-----")
        assert staged["public_key_pem"].startswith("-----BEGIN PUBLIC KEY-----")

        # The CLI must canary-sign with BOTH keys during stage so we
        # know the handoff will work. The transition marker is recorded.
        assert report.details["transition_marker"] == "both_keys_canary_verified"
        assert report.details["canary_new_sig"]
        assert report.details["canary_current_sig"]

        # Audit captures operator identity.
        assert len(audit.records) == 1
        rec = audit.records[0]
        assert rec["action"] == "rotate-stage"
        assert rec["operator"] == "alice"
        assert rec["result"] == "success"

    def test_live_stage_without_current_key(self) -> None:
        vault = FakeVault()
        audit = CapturingAudit()
        report = rot.run(
            ["rotate-stage", "--live", "--operator", "alice"],
            vault=vault,
            audit=audit,
        )
        assert report.ok is True
        # No current → canary_current_sig is None.
        assert report.details["canary_current_sig"] is None

    def test_live_stage_refuses_if_next_already_staged(self) -> None:
        vault = FakeVault()
        vault.store[rot.KEY_PATH_CURRENT] = _make_key(kid="kid-a")
        vault.store[rot.KEY_PATH_NEXT] = _make_key(kid="kid-pending")
        audit = CapturingAudit()
        report = rot.run(
            ["rotate-stage", "--live"], vault=vault, audit=audit
        )
        assert report.ok is False
        assert "already staged" in report.message
        assert audit.records == []

    def test_live_stage_refuses_when_vault_unavailable(self) -> None:
        vault = FakeVault(available=False)
        audit = CapturingAudit()
        report = rot.run(
            ["rotate-stage", "--live"], vault=vault, audit=audit
        )
        assert report.ok is False
        assert "Vault unavailable" in report.message
        assert audit.records == []


# --------------------------------------------------------------------------- #
# rotate-promote                                                               #
# --------------------------------------------------------------------------- #


class TestRotatePromote:
    def _seed_staged(self, vault: FakeVault) -> None:
        vault.store[rot.KEY_PATH_CURRENT] = _make_key(kid="kid-old")
        vault.store[rot.KEY_PATH_NEXT] = _make_key(kid="kid-new")

    def test_dry_run_does_not_mutate(self) -> None:
        vault = FakeVault()
        self._seed_staged(vault)
        audit = CapturingAudit()
        report = rot.run(
            ["rotate-promote", "--dry-run"], vault=vault, audit=audit
        )
        assert report.ok is True
        # Everything still as seeded.
        assert vault.store[rot.KEY_PATH_CURRENT]["kid"] == "kid-old"
        assert vault.store[rot.KEY_PATH_NEXT]["kid"] == "kid-new"
        assert audit.records == []

    def test_live_promote_flips_keys_and_archives_old(self) -> None:
        vault = FakeVault()
        self._seed_staged(vault)
        audit = CapturingAudit()
        report = rot.run(
            ["rotate-promote", "--live", "--operator", "bob"],
            vault=vault,
            audit=audit,
        )
        assert report.ok is True, report.message
        # Current is now the staged key.
        assert vault.store[rot.KEY_PATH_CURRENT]["kid"] == "kid-new"
        # Next is gone.
        assert rot.KEY_PATH_NEXT not in vault.store
        # Old key archived under a date-stamped path.
        archive_keys = [
            k for k in vault.store if k.startswith(rot.KEY_PATH_ARCHIVE_PREFIX + "/")
        ]
        assert archive_keys, "old key must be archived"
        assert vault.store[archive_keys[0]]["kid"] == "kid-old"
        # Allowlist written with BOTH kids active during grace.
        allowlist = vault.store[rot.KEY_PATH_ALLOWLIST]
        assert set(allowlist["active_kids"]) == {"kid-old", "kid-new"}
        assert "kid-old" in allowlist["grace_until"]
        # Audit captures operator.
        assert audit.records[-1]["operator"] == "bob"
        assert audit.records[-1]["action"] == "rotate-promote"

    def test_live_promote_grace_window_duration(self) -> None:
        vault = FakeVault()
        self._seed_staged(vault)
        report = rot.run(
            ["rotate-promote", "--live"], vault=vault, audit=CapturingAudit()
        )
        assert report.ok is True
        grace_iso = vault.store[rot.KEY_PATH_ALLOWLIST]["grace_until"]["kid-old"]
        until = datetime.fromisoformat(grace_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        # Within ± 60 seconds of promotion + 30d.
        expected = now + timedelta(days=rot.GRACE_PERIOD_DAYS)
        assert abs((until - expected).total_seconds()) < 60

    def test_promote_refuses_without_staged_key(self) -> None:
        vault = FakeVault()
        vault.store[rot.KEY_PATH_CURRENT] = _make_key(kid="kid-a")
        audit = CapturingAudit()
        report = rot.run(
            ["rotate-promote", "--live"], vault=vault, audit=audit
        )
        assert report.ok is False
        assert "No staged key" in report.message
        assert audit.records == []

    def test_promote_vault_unavailable(self) -> None:
        vault = FakeVault(available=False)
        report = rot.run(
            ["rotate-promote", "--live"], vault=vault, audit=CapturingAudit()
        )
        assert report.ok is False
        assert "Vault unavailable" in report.message


# --------------------------------------------------------------------------- #
# rotate-retire-old                                                            #
# --------------------------------------------------------------------------- #


class TestRotateRetireOld:
    def test_retires_kids_past_grace(self) -> None:
        vault = FakeVault()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
        vault.store[rot.KEY_PATH_ALLOWLIST] = {
            "active_kids": ["kid-new", "kid-old-expired", "kid-still-valid"],
            "grace_until": {
                "kid-old-expired": past,
                "kid-still-valid": future,
            },
        }
        audit = CapturingAudit()
        report = rot.run(
            ["rotate-retire-old", "--live", "--operator", "carol"],
            vault=vault,
            audit=audit,
        )
        assert report.ok is True
        assert "kid-old-expired" in report.details["retired_kids"]
        # Still-valid kid remains.
        active = vault.store[rot.KEY_PATH_ALLOWLIST]["active_kids"]
        assert "kid-still-valid" in active
        assert "kid-old-expired" not in active
        # Audit captures operator.
        assert audit.records[-1]["operator"] == "carol"

    def test_noop_when_nothing_expired(self) -> None:
        vault = FakeVault()
        future = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
        vault.store[rot.KEY_PATH_ALLOWLIST] = {
            "active_kids": ["kid-new", "kid-old-active"],
            "grace_until": {"kid-old-active": future},
        }
        audit = CapturingAudit()
        report = rot.run(
            ["rotate-retire-old", "--live"], vault=vault, audit=audit
        )
        assert report.ok is True
        assert "nothing to retire" in report.message
        assert audit.records == []

    def test_dry_run_does_not_mutate(self) -> None:
        vault = FakeVault()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        vault.store[rot.KEY_PATH_ALLOWLIST] = {
            "active_kids": ["kid-old"],
            "grace_until": {"kid-old": past},
        }
        report = rot.run(
            ["rotate-retire-old", "--dry-run"],
            vault=vault,
            audit=CapturingAudit(),
        )
        assert report.ok is True
        # State unchanged.
        assert vault.store[rot.KEY_PATH_ALLOWLIST]["active_kids"] == ["kid-old"]


# --------------------------------------------------------------------------- #
# End-to-end stage -> promote -> retire flow with both keys verifying.         #
# --------------------------------------------------------------------------- #


class TestFullRotationFlow:
    def test_grace_period_both_keys_verify(self) -> None:
        vault = FakeVault()
        vault.store[rot.KEY_PATH_CURRENT] = _make_key(kid="kid-0")
        audit = CapturingAudit()

        # Stage.
        r1 = rot.run(
            ["rotate-stage", "--live", "--operator", "ops"],
            vault=vault,
            audit=audit,
        )
        assert r1.ok
        old_pub = vault.store[rot.KEY_PATH_CURRENT]["public_key_pem"]

        # Promote.
        r2 = rot.run(
            ["rotate-promote", "--live", "--operator", "ops"],
            vault=vault,
            audit=audit,
        )
        assert r2.ok
        new_pub = vault.store[rot.KEY_PATH_CURRENT]["public_key_pem"]

        # During grace: BOTH kids should verify against canary.
        # Recover the staged private key from the audit record? We
        # instead re-sign with the new private key (now current) and
        # verify with both the new public key (current) and the old
        # public key (archived).
        archive_keys = [
            k for k in vault.store if k.startswith(rot.KEY_PATH_ARCHIVE_PREFIX + "/")
        ]
        archived = vault.store[archive_keys[0]]
        old_pub_from_archive = archived["public_key_pem"]
        assert old_pub_from_archive == old_pub

        # New current key signs; new public verifies.
        new_priv = vault.store[rot.KEY_PATH_CURRENT]["private_key_pem"]
        sig = rot._sign_with_pem(rot.CANARY_PAYLOAD, new_priv)
        assert rot._verify_with_pem(rot.CANARY_PAYLOAD, sig, new_pub) is True

        # The archived old key's public key must NOT verify the new
        # signature — but BOTH kids remain in the allowlist, so a
        # verifier seeing a receipt signed by the OLD key (still in
        # grace) will still accept it. That is what the allowlist
        # encodes.
        assert (
            rot._verify_with_pem(rot.CANARY_PAYLOAD, sig, old_pub_from_archive)
            is False
        )

        allowlist = vault.store[rot.KEY_PATH_ALLOWLIST]
        assert {"kid-0"}.issubset(set(allowlist["active_kids"]))

        # Simulate expiry of grace: rewrite the allowlist with a past
        # expiry for the old kid, then retire.
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        allowlist["grace_until"]["kid-0"] = past
        vault.store[rot.KEY_PATH_ALLOWLIST] = allowlist

        r3 = rot.run(
            ["rotate-retire-old", "--live", "--operator", "ops"],
            vault=vault,
            audit=audit,
        )
        assert r3.ok
        post = vault.store[rot.KEY_PATH_ALLOWLIST]
        assert "kid-0" not in post["active_kids"]

        # Every mutating step wrote an audit record with the same operator.
        operators = {rec["operator"] for rec in audit.records}
        assert operators == {"ops"}
        actions = [rec["action"] for rec in audit.records]
        assert actions == ["rotate-stage", "rotate-promote", "rotate-retire-old"]


# --------------------------------------------------------------------------- #
# Ensure the signed-receipts tier -> algorithm policy is untouched.            #
# --------------------------------------------------------------------------- #


def test_middleware_tier_policy_preserved() -> None:
    # This is a sanity check: rotation MUST NOT alter the tier -> algorithm
    # mapping.  If the middleware is refactored so that Ed25519 stops being
    # the choice for consulting/platform/enterprise, the rotation CLI as
    # designed would still be writing Ed25519 keys — so we pin the
    # expectation here.
    from greenlang.factors.middleware.signed_receipts import algorithm_for_tier

    assert algorithm_for_tier("community") == "sha256-hmac"
    assert algorithm_for_tier("pro") == "sha256-hmac"
    assert algorithm_for_tier("internal") == "sha256-hmac"
    assert algorithm_for_tier("consulting") == "ed25519"
    assert algorithm_for_tier("platform") == "ed25519"
    assert algorithm_for_tier("enterprise") == "ed25519"


# --------------------------------------------------------------------------- #
# argparse enforces --dry-run default (no mutation without --live).            #
# --------------------------------------------------------------------------- #


class TestDefaultDryRun:
    def test_stage_default_is_dry_run(self) -> None:
        vault = FakeVault()
        report = rot.run(["rotate-stage"], vault=vault, audit=CapturingAudit())
        assert report.dry_run is True
        assert rot.KEY_PATH_NEXT not in vault.store

    def test_promote_default_is_dry_run(self) -> None:
        vault = FakeVault()
        vault.store[rot.KEY_PATH_CURRENT] = _make_key(kid="kid-old")
        vault.store[rot.KEY_PATH_NEXT] = _make_key(kid="kid-new")
        report = rot.run(
            ["rotate-promote"], vault=vault, audit=CapturingAudit()
        )
        assert report.dry_run is True
        assert vault.store[rot.KEY_PATH_CURRENT]["kid"] == "kid-old"

    def test_retire_default_is_dry_run(self) -> None:
        vault = FakeVault()
        vault.store[rot.KEY_PATH_ALLOWLIST] = {
            "active_kids": ["kid-old"],
            "grace_until": {
                "kid-old": (
                    datetime.now(timezone.utc) - timedelta(days=1)
                ).isoformat()
            },
        }
        report = rot.run(
            ["rotate-retire-old"], vault=vault, audit=CapturingAudit()
        )
        assert report.dry_run is True
        assert vault.store[rot.KEY_PATH_ALLOWLIST]["active_kids"] == ["kid-old"]
