# -*- coding: utf-8 -*-
"""Tests for ``greenlang.factors.security.tenant_vault_transit``.

Coverage matrix
---------------
1.  encrypt + decrypt round-trip (dev mode)
2.  per-tenant key isolation (tenant A cannot decrypt tenant B's blob)
3.  key rotation returns a version number
4.  rotation in dev mode is a no-op but audited
5.  dev-mode ciphertext format matches the Vault envelope
6.  dev-mode fallback behaves identically from caller's POV
7.  prod-mode refuses to start without Vault
8.  prod-mode refuses when dev fallback is explicitly disabled
9.  audit log captures operation + tenant_id + plaintext digest
10. audit log NEVER contains plaintext bytes
11. empty / whitespace tenant_id rejected
12. non-bytes plaintext rejected
13. mangled ciphertext raises TenantKeyAccessError
14. default_transit() is a thread-safe singleton
15. reset_default_transit() clears the singleton
16. encrypt on a second tenant auto-creates a fresh key
17. Vault-backed path: mocked hvac client happy path
18. Vault-backed path: 403 -> TenantKeyAccessError
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.factors.security.tenant_vault_transit import (
    TenantKeyAccessError,
    TenantVaultTransit,
    TenantVaultTransitError,
    TransitAuditEntry,
    VaultUnavailableInProductionError,
    default_transit,
    reset_default_transit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch):
    """Remove Vault + env-detection vars so we get deterministic dev mode."""
    for var in ("VAULT_ADDR", "VAULT_TOKEN", "GL_ENV", "ENVIRONMENT"):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def dev_transit(clean_env):
    """A fresh dev-mode TenantVaultTransit."""
    return TenantVaultTransit()


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Make sure the singleton doesn't leak between tests."""
    reset_default_transit()
    yield
    reset_default_transit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Basic encrypt -> decrypt happy path (dev mode)."""

    def test_round_trip_bytes(self, dev_transit):
        ciphertext = dev_transit.encrypt("tenant-a", b"hello world")
        assert ciphertext.startswith("vault:v")
        assert b"hello world" not in ciphertext.encode("utf-8")

        plaintext = dev_transit.decrypt("tenant-a", ciphertext)
        assert plaintext == b"hello world"

    def test_round_trip_binary_payload(self, dev_transit):
        payload = bytes(range(256)) * 4  # 1 KiB of every byte
        ciphertext = dev_transit.encrypt("tenant-x", payload)
        assert dev_transit.decrypt("tenant-x", ciphertext) == payload


class TestTenantIsolation:
    def test_tenant_b_cannot_decrypt_tenant_a(self, dev_transit):
        ciphertext = dev_transit.encrypt("tenant-a", b"secret-value-42")
        with pytest.raises(TenantKeyAccessError):
            dev_transit.decrypt("tenant-b", ciphertext)

    def test_unique_ciphertext_per_tenant(self, dev_transit):
        c_a = dev_transit.encrypt("tenant-a", b"hello")
        c_b = dev_transit.encrypt("tenant-b", b"hello")
        assert c_a != c_b

    def test_nonce_makes_ciphertext_non_deterministic(self, dev_transit):
        c1 = dev_transit.encrypt("tenant-a", b"same")
        c2 = dev_transit.encrypt("tenant-a", b"same")
        assert c1 != c2  # random nonce per call
        assert dev_transit.decrypt("tenant-a", c1) == b"same"
        assert dev_transit.decrypt("tenant-a", c2) == b"same"


class TestRotation:
    def test_dev_rotate_is_noop_returning_v1(self, dev_transit):
        v = dev_transit.rotate_tenant_key("tenant-a")
        assert v == 1

    def test_dev_rotation_audited(self, dev_transit):
        dev_transit.rotate_tenant_key("tenant-a")
        ops = [entry.operation for entry in dev_transit.audit_log]
        assert "rotate" in ops


class TestDevEnvelopeFormat:
    def test_envelope_prefix(self, dev_transit):
        ct = dev_transit.encrypt("t", b"x")
        # Shared envelope format: vault:v{n}:{marker}:{payload}
        parts = ct.split(":", 3)
        assert parts[0] == "vault"
        assert parts[1].startswith("v")
        assert parts[2] == "dev"


class TestProdGuards:
    def test_prod_without_vault_refuses(self, monkeypatch):
        monkeypatch.delenv("VAULT_ADDR", raising=False)
        monkeypatch.delenv("VAULT_TOKEN", raising=False)
        monkeypatch.setenv("GL_ENV", "production")
        with pytest.raises(VaultUnavailableInProductionError):
            TenantVaultTransit()

    def test_prod_with_bad_vault_refuses(self, monkeypatch):
        monkeypatch.setenv("GL_ENV", "production")
        monkeypatch.setenv("VAULT_ADDR", "http://vault.invalid:8200")
        # In production, a failed Vault connection must NOT silently
        # fall back to dev mode.
        with pytest.raises((VaultUnavailableInProductionError, TenantVaultTransitError)):
            TenantVaultTransit()

    def test_explicit_disable_dev_fallback(self, clean_env):
        with pytest.raises(TenantVaultTransitError):
            TenantVaultTransit(allow_dev_fallback=False)


class TestAudit:
    def test_audit_records_encrypt(self, dev_transit):
        dev_transit.encrypt("tenant-a", b"abc")
        entries = dev_transit.audit_log
        assert len(entries) == 1
        entry = entries[0]
        assert isinstance(entry, TransitAuditEntry)
        assert entry.operation == "encrypt"
        assert entry.tenant_id == "tenant-a"
        assert entry.backend == "dev"
        assert entry.plaintext_digest == hashlib.sha256(b"abc").hexdigest()

    def test_audit_records_decrypt(self, dev_transit):
        ct = dev_transit.encrypt("tenant-a", b"abc")
        dev_transit.decrypt("tenant-a", ct)
        ops = [e.operation for e in dev_transit.audit_log]
        assert ops == ["encrypt", "decrypt"]

    def test_audit_never_contains_plaintext(self, dev_transit):
        plaintext = b"SUPER-SECRET-LITERAL-DO-NOT-LEAK"
        dev_transit.encrypt("tenant-a", plaintext)
        ct = dev_transit.encrypt("tenant-b", plaintext)
        dev_transit.decrypt("tenant-b", ct)
        for entry in dev_transit.audit_log:
            serialised = repr(entry).encode("utf-8")
            assert plaintext not in serialised

    def test_audit_captures_tenant_id(self, dev_transit):
        dev_transit.encrypt("tenant-a", b"x")
        dev_transit.encrypt("tenant-b", b"y")
        seen = {e.tenant_id for e in dev_transit.audit_log}
        assert seen == {"tenant-a", "tenant-b"}


class TestInputValidation:
    def test_empty_tenant_id_rejected(self, dev_transit):
        with pytest.raises(ValueError):
            dev_transit.encrypt("", b"x")

    def test_whitespace_tenant_id_rejected(self, dev_transit):
        with pytest.raises(ValueError):
            dev_transit.encrypt("   ", b"x")

    def test_non_bytes_plaintext_rejected(self, dev_transit):
        with pytest.raises(TypeError):
            dev_transit.encrypt("tenant-a", "string not bytes")  # type: ignore[arg-type]

    def test_empty_ciphertext_rejected(self, dev_transit):
        with pytest.raises(TenantKeyAccessError):
            dev_transit.decrypt("tenant-a", "")

    def test_mangled_ciphertext_raises_access_error(self, dev_transit):
        with pytest.raises(TenantKeyAccessError):
            dev_transit.decrypt("tenant-a", "not:a:valid:ciphertext")


class TestSingleton:
    def test_default_is_cached(self, clean_env):
        a = default_transit()
        b = default_transit()
        assert a is b

    def test_reset_clears(self, clean_env):
        a = default_transit()
        reset_default_transit()
        b = default_transit()
        assert a is not b

    def test_default_thread_safe(self, clean_env):
        seen: list[TenantVaultTransit] = []

        def _grab():
            seen.append(default_transit())

        threads = [threading.Thread(target=_grab) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(seen) == 16
        first = seen[0]
        assert all(inst is first for inst in seen)


# ---------------------------------------------------------------------------
# Vault-backed path (mocked hvac)
# ---------------------------------------------------------------------------


class _FakeTransit:
    """Minimal stand-in for hvac's ``client.secrets.transit``."""

    def __init__(self) -> None:
        self.keys: Dict[str, int] = {}
        self.blobs: Dict[str, tuple[str, str]] = {}  # ct -> (key, plaintext_b64)
        self.counter = 0

    def create_key(self, *, name: str, mount_point: str, **_: Any) -> Dict[str, Any]:
        if name in self.keys:
            raise RuntimeError("existing key")
        self.keys[name] = 1
        return {"data": {"latest_version": 1}}

    def rotate_key(self, *, name: str, mount_point: str) -> Dict[str, Any]:
        if name not in self.keys:
            raise RuntimeError("key not found")
        self.keys[name] += 1
        return {}

    def read_key(self, *, name: str, mount_point: str) -> Dict[str, Any]:
        return {"data": {"latest_version": self.keys[name]}}

    def encrypt_data(
        self, *, name: str, plaintext: str, mount_point: str
    ) -> Dict[str, Any]:
        self.counter += 1
        ct = f"vault:v{self.keys[name]}:fake-{self.counter}"
        self.blobs[ct] = (name, plaintext)
        return {"data": {"ciphertext": ct}}

    def decrypt_data(
        self, *, name: str, ciphertext: str, mount_point: str
    ) -> Dict[str, Any]:
        if ciphertext not in self.blobs:
            raise RuntimeError("unable to decrypt: cipher")
        owner, pt_b64 = self.blobs[ciphertext]
        if owner != name:
            raise RuntimeError("permission denied")
        return {"data": {"plaintext": pt_b64}}


class _FakeSecrets:
    def __init__(self) -> None:
        self.transit = _FakeTransit()


class _FakeClient:
    def __init__(self, *, url: str, token: Optional[str]) -> None:
        self.url = url
        self.token = token
        self.secrets = _FakeSecrets()

    def is_authenticated(self) -> bool:
        return bool(self.token)


class TestVaultBacked:
    def _make(self, monkeypatch) -> TenantVaultTransit:
        fake_hvac = MagicMock()
        fake_hvac.Client = _FakeClient
        monkeypatch.setitem(__import__("sys").modules, "hvac", fake_hvac)
        monkeypatch.setenv("VAULT_ADDR", "http://vault.test:8200")
        monkeypatch.setenv("VAULT_TOKEN", "root")
        monkeypatch.delenv("GL_ENV", raising=False)
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        t = TenantVaultTransit()
        assert t.backend == "vault"
        return t

    def test_vault_round_trip(self, monkeypatch):
        t = self._make(monkeypatch)
        ct = t.encrypt("tenant-a", b"the answer is 42")
        assert ct.startswith("vault:v1:")
        assert t.decrypt("tenant-a", ct) == b"the answer is 42"

    def test_vault_cross_tenant_blocked(self, monkeypatch):
        t = self._make(monkeypatch)
        ct = t.encrypt("tenant-a", b"x")
        with pytest.raises(TenantKeyAccessError):
            t.decrypt("tenant-b", ct)

    def test_vault_rotation_bumps_version(self, monkeypatch):
        t = self._make(monkeypatch)
        t.encrypt("tenant-a", b"x")  # creates key + encrypts at v1
        v = t.rotate_tenant_key("tenant-a")
        assert v == 2
        # Rotation should also be reflected in the next encrypt's envelope.
        ct = t.encrypt("tenant-a", b"y")
        assert ct.startswith("vault:v2:")

    def test_old_ciphertexts_decryptable_after_rotation(self, monkeypatch):
        t = self._make(monkeypatch)
        old_ct = t.encrypt("tenant-a", b"old-secret")
        assert old_ct.startswith("vault:v1:")
        t.rotate_tenant_key("tenant-a")
        # Vault keeps version history; decrypt still works.
        assert t.decrypt("tenant-a", old_ct) == b"old-secret"
