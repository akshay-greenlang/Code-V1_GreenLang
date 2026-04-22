# -*- coding: utf-8 -*-
"""
Per-tenant Vault Transit encryption for customer-private factor data.

This module wraps HashiCorp Vault's Transit secrets engine to provide
envelope encryption scoped per tenant. Every tenant gets its own key in
Vault under the path::

    transit/keys/factors-tenant-{tenant_id}

The key is lazily created on first use (if the caller's Vault token has
``create`` capability on that path) and supports rotation with version
history — Vault decrypts old ciphertexts against their original key
version even after rotation.

Dev-mode fallback
-----------------
When ``VAULT_ADDR`` is unset or Vault is unreachable, the module falls
back to envelope encryption with an AES-256-GCM key scoped to the
running process. This mode is **strictly for local dev / CI**; it logs
one warning per process and **refuses to start** if the environment is
``GL_ENV=production``.

Ciphertext format
-----------------
Both Vault and dev-mode encrypt output obeys the Vault Transit
convention ``vault:v{version}:{base64}`` so that a dev-mode blob can be
re-encrypted through Vault later without migration.

Thread-safety
-------------
All public methods acquire an internal lock. The underlying ``hvac``
client is itself thread-safe for independent requests; the lock only
serialises key-creation and key-rotation to avoid racey 409s.

Audit logging
-------------
Every ``encrypt`` / ``decrypt`` / ``rotate_tenant_key`` call is logged
at INFO level with the tenant_id **and a SHA-256 digest of the
plaintext** — the plaintext itself never enters the log stream.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KEY_PATH_PREFIX = "factors-tenant-"
_DEV_CIPHERTEXT_VERSION = 1
_NONCE_BYTES = 12  # AES-GCM spec
_DEV_KEY_BYTES = 32  # AES-256
_PROD_ENV_VALUES = frozenset({"production", "prod", "prd"})

# Marker that identifies ciphertexts that were written by the dev-mode
# fallback (as opposed to real Vault Transit output).  We embed it in
# the ciphertext envelope so the decrypt path can dispatch without
# guessing.
_DEV_MARKER = "dev"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TenantVaultTransitError(RuntimeError):
    """Base class for all tenant-vault-transit errors."""


class TenantKeyAccessError(TenantVaultTransitError):
    """Raised when the caller is not entitled to decrypt the ciphertext.

    Typical causes:
      * Cross-tenant decryption attempt (tenant A's ciphertext handed to
        tenant B).
      * Ciphertext format / version mismatch.
      * Vault policy rejected the decrypt call.
    """


class VaultUnavailableInProductionError(TenantVaultTransitError):
    """Raised when ``GL_ENV=production`` but Vault cannot be reached."""


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitAuditEntry:
    """Immutable record of a single encrypt/decrypt/rotate operation.

    ``plaintext_digest`` is the SHA-256 hex digest of the plaintext;
    the plaintext itself is never captured. ``ciphertext_pointer`` is
    a short prefix of the ciphertext envelope (useful for log grep).
    """

    operation: str  # "encrypt" | "decrypt" | "rotate"
    tenant_id: str
    key_version: Optional[int] = None
    plaintext_digest: Optional[str] = None
    ciphertext_pointer: Optional[str] = None
    backend: str = "vault"  # "vault" | "dev"
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _pointer(ciphertext: str) -> str:
    """Return a short, non-reversible pointer for audit logs."""
    if not ciphertext:
        return ""
    # First 16 chars of the envelope is enough to correlate in audit logs
    # without leaking plaintext.
    return ciphertext[:16]


def _is_production(env: Optional[str] = None) -> bool:
    """Return True when the current environment is production."""
    raw = env if env is not None else os.getenv("GL_ENV", os.getenv("ENVIRONMENT", ""))
    return (raw or "").strip().lower() in _PROD_ENV_VALUES


def _pack_dev_ciphertext(version: int, blob: bytes) -> str:
    """Encode a dev-mode ciphertext into the shared Vault envelope."""
    b64 = base64.b64encode(blob).decode("ascii")
    return f"vault:v{version}:{_DEV_MARKER}:{b64}"


def _unpack_dev_ciphertext(ciphertext: str) -> Optional[Tuple[int, bytes]]:
    """Return ``(version, raw_bytes)`` for a dev-mode ciphertext, else None."""
    try:
        prefix, version_token, marker, payload = ciphertext.split(":", 3)
    except ValueError:
        return None
    if prefix != "vault" or marker != _DEV_MARKER:
        return None
    if not version_token.startswith("v"):
        return None
    try:
        version = int(version_token[1:])
        raw = base64.b64decode(payload)
    except (ValueError, TypeError):
        return None
    return version, raw


def _is_dev_ciphertext(ciphertext: str) -> bool:
    """Return True when the ciphertext was produced by the dev fallback."""
    return _unpack_dev_ciphertext(ciphertext) is not None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TenantVaultTransit:
    """Per-tenant Vault Transit encryption helper.

    Parameters
    ----------
    vault_addr:
        Optional Vault server URL; defaults to ``VAULT_ADDR`` env var.
    vault_token:
        Optional Vault token; defaults to ``VAULT_TOKEN`` env var.
    mount_point:
        Transit mount path in Vault; defaults to ``"transit"``.
    allow_dev_fallback:
        When ``True`` (default) the class silently falls back to
        AES-256-GCM envelope encryption if Vault is unreachable.
        The fallback is refused if the current environment is prod.
    """

    _singleton: "Optional[TenantVaultTransit]" = None
    _singleton_lock = threading.Lock()

    # ---- init --------------------------------------------------------

    def __init__(
        self,
        *,
        vault_addr: Optional[str] = None,
        vault_token: Optional[str] = None,
        mount_point: str = "transit",
        allow_dev_fallback: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._audit_log: list[TransitAuditEntry] = []
        self._mount_point = mount_point
        self._allow_dev_fallback = allow_dev_fallback
        self._vault_addr = vault_addr if vault_addr is not None else os.getenv("VAULT_ADDR", "")
        self._vault_token = (
            vault_token if vault_token is not None else os.getenv("VAULT_TOKEN", "")
        )
        self._client: Any = None
        self._dev_key: Optional[bytes] = None
        self._known_tenant_keys: set[str] = set()
        self._warned_dev_mode = False

        self._init_backend()

    # ---- backend selection -------------------------------------------

    def _init_backend(self) -> None:
        """Attempt to connect to Vault; fall back to dev mode if impossible."""
        if self._vault_addr:
            try:
                self._client = self._build_vault_client()
                if self._client is not None:
                    logger.info(
                        "TenantVaultTransit: connected to Vault at %s (mount=%s)",
                        self._vault_addr,
                        self._mount_point,
                    )
                    return
            except Exception as exc:  # noqa: BLE001 — intentional broad catch
                logger.warning(
                    "TenantVaultTransit: Vault connection failed (%s); "
                    "falling back to dev mode",
                    exc,
                )

        self._enter_dev_mode()

    def _build_vault_client(self) -> Any:
        """Build and verify an hvac client.

        Returns ``None`` if ``hvac`` is not installed — caller will
        fall back to dev mode.
        """
        try:
            import hvac  # type: ignore[import-not-found]
        except ImportError:
            logger.info(
                "TenantVaultTransit: hvac not installed; dev-mode fallback engaged"
            )
            return None

        client = hvac.Client(url=self._vault_addr, token=self._vault_token or None)
        # ``is_authenticated`` triggers a real network call — if Vault
        # is down we want to know now, not on first encrypt.
        if not client.is_authenticated():
            raise TenantVaultTransitError(
                "Vault client could not authenticate against %s" % self._vault_addr
            )
        return client

    def _enter_dev_mode(self) -> None:
        """Engage the in-process AES-256-GCM fallback."""
        if _is_production():
            raise VaultUnavailableInProductionError(
                "Vault is required in production. Set VAULT_ADDR and VAULT_TOKEN."
            )
        if not self._allow_dev_fallback:
            raise TenantVaultTransitError(
                "Vault unreachable and dev-mode fallback is disabled."
            )

        # One key per *process*; we derive per-tenant sub-keys from it.
        # Using AESGCM.generate_key produces a cryptographically random
        # key suitable for AES-256.
        self._dev_key = AESGCM.generate_key(bit_length=256)
        self._client = None

        if not self._warned_dev_mode:
            logger.warning(
                "TenantVaultTransit: DEV MODE — using in-process AES-256-GCM "
                "envelope encryption. This is safe for local dev / CI ONLY. "
                "Refusing to run in production. Set VAULT_ADDR + VAULT_TOKEN "
                "to switch to Vault Transit."
            )
            self._warned_dev_mode = True

    # ---- public API --------------------------------------------------

    @property
    def backend(self) -> str:
        """``"vault"`` when talking to real Vault, ``"dev"`` otherwise."""
        return "vault" if self._client is not None else "dev"

    @property
    def audit_log(self) -> list[TransitAuditEntry]:
        """Return a defensive copy of the in-memory audit log."""
        with self._lock:
            return list(self._audit_log)

    def encrypt(self, tenant_id: str, plaintext: bytes) -> str:
        """Encrypt ``plaintext`` under the tenant's key; return ciphertext.

        Parameters
        ----------
        tenant_id:
            Logical tenant identifier. Must be non-empty.
        plaintext:
            Raw bytes to encrypt. UTF-8 text callers should encode
            themselves.

        Returns
        -------
        str
            A ciphertext string in Vault Transit envelope form. Callers
            must treat it as opaque.
        """
        self._require_tenant_id(tenant_id)
        if not isinstance(plaintext, (bytes, bytearray)):
            raise TypeError("plaintext must be bytes")

        with self._lock:
            if self._client is not None:
                ciphertext, key_version = self._vault_encrypt(tenant_id, bytes(plaintext))
                backend = "vault"
            else:
                ciphertext, key_version = self._dev_encrypt(tenant_id, bytes(plaintext))
                backend = "dev"

            self._record(
                TransitAuditEntry(
                    operation="encrypt",
                    tenant_id=tenant_id,
                    key_version=key_version,
                    plaintext_digest=_sha256_hex(bytes(plaintext)),
                    ciphertext_pointer=_pointer(ciphertext),
                    backend=backend,
                )
            )
            return ciphertext

    def decrypt(self, tenant_id: str, ciphertext: str) -> bytes:
        """Decrypt ``ciphertext`` for the given tenant; return plaintext.

        Raises
        ------
        TenantKeyAccessError
            If the caller supplies a ``tenant_id`` that does not match
            the key under which the ciphertext was encrypted, or if the
            ciphertext is malformed, or if Vault refuses the operation.
        """
        self._require_tenant_id(tenant_id)
        if not isinstance(ciphertext, str) or not ciphertext:
            raise TenantKeyAccessError("ciphertext must be a non-empty string")

        with self._lock:
            backend = self.backend
            try:
                if _is_dev_ciphertext(ciphertext):
                    plaintext = self._dev_decrypt(tenant_id, ciphertext)
                elif self._client is not None:
                    plaintext = self._vault_decrypt(tenant_id, ciphertext)
                else:
                    # Caller handed us a Vault-format ciphertext but we
                    # no longer have a Vault connection.
                    raise TenantKeyAccessError(
                        "Ciphertext appears to be Vault Transit format but the "
                        "Vault backend is unavailable."
                    )
            except TenantKeyAccessError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise TenantKeyAccessError(
                    "Decryption failed for tenant %r: %s" % (tenant_id, exc)
                ) from exc

            self._record(
                TransitAuditEntry(
                    operation="decrypt",
                    tenant_id=tenant_id,
                    plaintext_digest=_sha256_hex(plaintext),
                    ciphertext_pointer=_pointer(ciphertext),
                    backend=backend,
                )
            )
            return plaintext

    def rotate_tenant_key(self, tenant_id: str) -> int:
        """Rotate the tenant's Vault Transit key; return new key version.

        Vault handles version history automatically — previously
        emitted ciphertexts remain decryptable as long as their key
        version has not been archived / deleted on the server.

        In dev mode this is a no-op that returns ``1`` (the dev fallback
        has only one live key per process).
        """
        self._require_tenant_id(tenant_id)

        with self._lock:
            if self._client is None:
                # Dev mode has no versioning semantics.
                self._record(
                    TransitAuditEntry(
                        operation="rotate",
                        tenant_id=tenant_id,
                        key_version=_DEV_CIPHERTEXT_VERSION,
                        backend="dev",
                    )
                )
                return _DEV_CIPHERTEXT_VERSION

            self._ensure_tenant_key(tenant_id)
            key_name = self._key_name(tenant_id)
            try:
                self._client.secrets.transit.rotate_key(
                    name=key_name, mount_point=self._mount_point
                )
                meta = self._read_key_meta(key_name)
                new_version = int(meta.get("latest_version", 0))
            except Exception as exc:  # noqa: BLE001
                raise TenantVaultTransitError(
                    "Vault key rotation failed for tenant %r: %s" % (tenant_id, exc)
                ) from exc

            self._record(
                TransitAuditEntry(
                    operation="rotate",
                    tenant_id=tenant_id,
                    key_version=new_version,
                    backend="vault",
                )
            )
            logger.info(
                "TenantVaultTransit: rotated key for tenant=%s new_version=%d",
                tenant_id,
                new_version,
            )
            return new_version

    # ---- vault-backed helpers ----------------------------------------

    def _key_name(self, tenant_id: str) -> str:
        """Return the Vault Transit key name for the tenant."""
        return f"{_KEY_PATH_PREFIX}{tenant_id}"

    def _ensure_tenant_key(self, tenant_id: str) -> None:
        """Create the tenant's Vault Transit key on first use."""
        if tenant_id in self._known_tenant_keys:
            return
        key_name = self._key_name(tenant_id)
        try:
            self._client.secrets.transit.create_key(
                name=key_name,
                mount_point=self._mount_point,
                exportable=False,
                allow_plaintext_backup=False,
                key_type="aes256-gcm96",
            )
            logger.info(
                "TenantVaultTransit: created Vault key transit/keys/%s", key_name
            )
        except Exception as exc:  # noqa: BLE001
            # Vault returns a 400 if the key already exists; that's fine.
            message = str(exc).lower()
            if "existing key" not in message and "already exists" not in message:
                # Do NOT swallow real errors (403, 5xx, etc.).
                raise TenantVaultTransitError(
                    "Failed to ensure tenant key %r: %s" % (tenant_id, exc)
                ) from exc
        self._known_tenant_keys.add(tenant_id)

    def _read_key_meta(self, key_name: str) -> Dict[str, Any]:
        """Return the Vault Transit key's metadata dict."""
        resp = self._client.secrets.transit.read_key(
            name=key_name, mount_point=self._mount_point
        )
        return (resp or {}).get("data", {}) or {}

    def _vault_encrypt(self, tenant_id: str, plaintext: bytes) -> Tuple[str, Optional[int]]:
        """Encrypt via Vault Transit and return ``(ciphertext, key_version)``."""
        self._ensure_tenant_key(tenant_id)
        key_name = self._key_name(tenant_id)
        b64 = base64.b64encode(plaintext).decode("ascii")
        resp = self._client.secrets.transit.encrypt_data(
            name=key_name, plaintext=b64, mount_point=self._mount_point
        )
        data = (resp or {}).get("data", {}) or {}
        ct = data.get("ciphertext", "")
        if not ct:
            raise TenantVaultTransitError(
                "Vault returned empty ciphertext for tenant %r" % tenant_id
            )
        version = self._extract_key_version(ct)
        return ct, version

    def _vault_decrypt(self, tenant_id: str, ciphertext: str) -> bytes:
        """Decrypt via Vault Transit, mapping policy errors to TenantKeyAccessError."""
        key_name = self._key_name(tenant_id)
        try:
            resp = self._client.secrets.transit.decrypt_data(
                name=key_name, ciphertext=ciphertext, mount_point=self._mount_point
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if any(
                needle in message
                for needle in ("permission denied", "403", "forbidden", "unauthorized")
            ):
                raise TenantKeyAccessError(
                    "Tenant %r is not authorised to decrypt this ciphertext." % tenant_id
                ) from exc
            if "unable to decrypt" in message or "cipher" in message:
                raise TenantKeyAccessError(
                    "Ciphertext was not encrypted with tenant %r's key." % tenant_id
                ) from exc
            raise
        data = (resp or {}).get("data", {}) or {}
        b64 = data.get("plaintext", "")
        if not b64:
            raise TenantKeyAccessError(
                "Vault returned empty plaintext for tenant %r" % tenant_id
            )
        return base64.b64decode(b64)

    @staticmethod
    def _extract_key_version(ciphertext: str) -> Optional[int]:
        """Parse ``v{N}`` out of a ``vault:v{N}:{payload}`` envelope."""
        try:
            _, token, *_ = ciphertext.split(":", 2)
        except ValueError:
            return None
        if token.startswith("v"):
            try:
                return int(token[1:])
            except ValueError:
                return None
        return None

    # ---- dev-mode helpers --------------------------------------------

    def _derive_dev_key(self, tenant_id: str) -> bytes:
        """Derive a per-tenant AES-256 key from the process key."""
        assert self._dev_key is not None
        # HKDF-like construction using SHA-256; sufficient for dev.
        h = hashlib.sha256()
        h.update(b"greenlang-tenant-vault-transit-v1")
        h.update(self._dev_key)
        h.update(tenant_id.encode("utf-8"))
        return h.digest()

    def _dev_encrypt(
        self, tenant_id: str, plaintext: bytes
    ) -> Tuple[str, Optional[int]]:
        """AES-256-GCM encrypt with tenant_id bound into the AAD."""
        key = self._derive_dev_key(tenant_id)
        nonce = os.urandom(_NONCE_BYTES)
        aead = AESGCM(key)
        aad = tenant_id.encode("utf-8")
        ct = aead.encrypt(nonce, plaintext, aad)
        blob = nonce + ct
        return _pack_dev_ciphertext(_DEV_CIPHERTEXT_VERSION, blob), _DEV_CIPHERTEXT_VERSION

    def _dev_decrypt(self, tenant_id: str, ciphertext: str) -> bytes:
        """Reverse of ``_dev_encrypt``; tenant_id mismatch raises."""
        unpacked = _unpack_dev_ciphertext(ciphertext)
        if unpacked is None:
            raise TenantKeyAccessError("Malformed dev-mode ciphertext.")
        _version, blob = unpacked
        if len(blob) < _NONCE_BYTES + 1:
            raise TenantKeyAccessError("Dev-mode ciphertext too short.")
        if self._dev_key is None:
            raise TenantKeyAccessError(
                "Dev-mode ciphertext cannot be decrypted: no process key available "
                "(dev key was regenerated or Vault is now connected)."
            )
        nonce, ct = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
        key = self._derive_dev_key(tenant_id)
        aead = AESGCM(key)
        aad = tenant_id.encode("utf-8")
        try:
            return aead.decrypt(nonce, ct, aad)
        except Exception as exc:  # noqa: BLE001
            # cryptography raises InvalidTag on tenant mismatch (because
            # the AAD binds tenant_id); surface that as an access error.
            raise TenantKeyAccessError(
                "Ciphertext was not encrypted for tenant %r." % tenant_id
            ) from exc

    # ---- misc --------------------------------------------------------

    @staticmethod
    def _require_tenant_id(tenant_id: str) -> None:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")

    def _record(self, entry: TransitAuditEntry) -> None:
        """Append ``entry`` to the audit log and emit a structured log line."""
        self._audit_log.append(entry)
        logger.info(
            "tenant_vault_transit op=%s tenant=%s backend=%s key_version=%s "
            "plaintext_sha256=%s ciphertext_ptr=%s",
            entry.operation,
            entry.tenant_id,
            entry.backend,
            entry.key_version if entry.key_version is not None else "-",
            entry.plaintext_digest or "-",
            entry.ciphertext_pointer or "-",
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


def default_transit() -> TenantVaultTransit:
    """Return the process-wide ``TenantVaultTransit`` singleton.

    Idempotent and thread-safe. The first caller wins; subsequent
    callers get the same instance. Useful for hot paths that do not
    want to thread a client through every function.
    """
    if TenantVaultTransit._singleton is None:
        with TenantVaultTransit._singleton_lock:
            if TenantVaultTransit._singleton is None:
                TenantVaultTransit._singleton = TenantVaultTransit()
    return TenantVaultTransit._singleton


def reset_default_transit() -> None:
    """Reset the process-wide singleton. **Testing use only.**"""
    with TenantVaultTransit._singleton_lock:
        TenantVaultTransit._singleton = None


__all__ = [
    "TenantVaultTransit",
    "TenantVaultTransitError",
    "TenantKeyAccessError",
    "VaultUnavailableInProductionError",
    "TransitAuditEntry",
    "default_transit",
    "reset_default_transit",
]
