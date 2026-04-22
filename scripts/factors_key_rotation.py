#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""factors_key_rotation.py — Ed25519 signing key rotation for Factors receipts.

Rotation lifecycle
------------------
The signed-receipts middleware (``greenlang/factors/middleware/signed_receipts.py``)
signs every Pro+/Enterprise response with Ed25519 using the private key
in ``GL_FACTORS_ED25519_PRIVATE_KEY`` / Vault ``secret/factors/ed25519/current``.
Tier policy (HMAC for community/pro/internal, Ed25519 for consulting/
platform/enterprise) is preserved across rotation; this tool only moves
the Ed25519 material.

Four-stage quarterly rotation::

    rotate-plan     -> read current metadata, report age / next date.
    rotate-stage    -> generate "next" keypair; sign a canary with BOTH
                       keys and verify BOTH. DO NOT flip signing yet.
    rotate-promote  -> promote "next" -> "current"; archive the old
                       "current" under .../archive/YYYYMMDD. Start 30-day
                       grace window during which verifiers must accept
                       BOTH keys.
    rotate-retire-old -> after the grace window, drop the archived key
                         from the verification allowlist.

All mutating actions:
  * require ``--live`` (default is ``--dry-run``; mutations are refused
    without ``--live``);
  * require ``VAULT_ADDR`` + ``VAULT_TOKEN`` env vars;
  * emit an audit record (SEC-005 ``audit_log``) with operator identity
    (``--operator``, ``USER``, or Vault-token self-introspection);
  * never crash on Vault unavailability — they log a warning and refuse
    to mutate, so a misconfigured host cannot silently corrupt state.

Vault layout
------------
::

    secret/factors/ed25519/current    -> {kid, created_at, algorithm,
                                           private_key_pem, public_key_pem}
    secret/factors/ed25519/next       -> same shape, post-stage
    secret/factors/ed25519/archive/YYYYMMDD -> historic keys; grace
                                              window pulls public keys
                                              from here for verify.
    secret/factors/ed25519/allowlist  -> {active_kids: [...], grace_until: {kid: iso_date}}

Usage
-----
::

    $ python scripts/factors_key_rotation.py rotate-plan
    $ python scripts/factors_key_rotation.py rotate-stage --live
    $ python scripts/factors_key_rotation.py rotate-promote --live
    $ python scripts/factors_key_rotation.py rotate-retire-old --live

Exit codes: 0 on success, 1 on handled failure, 2 on unexpected error.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import getpass
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Constants.
# --------------------------------------------------------------------------- #

#: Target rotation cadence — the CLI does not enforce this but ``rotate-plan``
#: reports how far the current key has drifted past it.
ROTATION_CADENCE_DAYS: int = 90

#: Both-keys-valid window after promotion.
GRACE_PERIOD_DAYS: int = 30

#: Vault paths — resolved relative to whatever mount Vault is serving
#: KV v2 on (``secret/`` by default; customers re-bind via VAULT_MOUNT).
VAULT_MOUNT_DEFAULT = "secret"
KEY_PATH_CURRENT = "factors/ed25519/current"
KEY_PATH_NEXT = "factors/ed25519/next"
KEY_PATH_ALLOWLIST = "factors/ed25519/allowlist"
KEY_PATH_ARCHIVE_PREFIX = "factors/ed25519/archive"

#: Stable label included in audit records so downstream SIEM rules can pin on it.
AUDIT_COMPONENT = "factors.signing.key_rotation"

#: Canary payload signed+verified by both keys during stage to prove the
#: handoff works before we flip the keypair.
CANARY_PAYLOAD: Dict[str, Any] = {
    "body_hash": "0" * 64,
    "edition_id": "canary",
    "path": "/internal/canary",
    "method": "POST",
    "status_code": 200,
}

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data model.
# --------------------------------------------------------------------------- #


@dataclass
class KeyMetadata:
    """Metadata describing a single Ed25519 key generation."""

    kid: str
    created_at: str
    algorithm: str = "ed25519"
    private_key_pem: str = ""
    public_key_pem: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyMetadata":
        return cls(
            kid=str(data.get("kid", "")),
            created_at=str(data.get("created_at", "")),
            algorithm=str(data.get("algorithm", "ed25519")),
            private_key_pem=str(data.get("private_key_pem", "")),
            public_key_pem=str(data.get("public_key_pem", "")),
        )

    def age_days(self, *, now: Optional[datetime] = None) -> Optional[float]:
        if not self.created_at:
            return None
        try:
            created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        ref = now or datetime.now(timezone.utc)
        return (ref - created).total_seconds() / 86400.0


@dataclass
class RotationReport:
    """Structured result emitted by each subcommand."""

    action: str
    dry_run: bool
    ok: bool = True
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "dry_run": self.dry_run,
            "ok": self.ok,
            "message": self.message,
            "details": self.details,
        }


# --------------------------------------------------------------------------- #
# Vault adapter.
# --------------------------------------------------------------------------- #


class VaultUnavailableError(RuntimeError):
    """Vault client unreachable / unauthenticated."""


class VaultClient:
    """Thin adapter over ``hvac`` — swappable for a mock in tests.

    The adapter deliberately never mutates state without being handed a
    token that can write. When ``hvac`` is not installed or the server
    is unreachable, ``is_available()`` returns False and the CLI refuses
    to mutate — a misconfigured host must never silently corrupt keys.
    """

    def __init__(
        self,
        *,
        addr: Optional[str] = None,
        token: Optional[str] = None,
        mount: str = VAULT_MOUNT_DEFAULT,
    ) -> None:
        self.addr = addr or os.getenv("VAULT_ADDR") or ""
        self.token = token or os.getenv("VAULT_TOKEN") or ""
        self.mount = mount or os.getenv("VAULT_MOUNT") or VAULT_MOUNT_DEFAULT
        self._client: Any = None

    # ---- Lifecycle --------------------------------------------------- #

    def is_available(self) -> bool:
        """Return True when Vault is reachable AND authenticated."""
        try:
            c = self._get_client()
            if c is None:
                return False
            auth = getattr(c, "is_authenticated", None)
            return bool(auth()) if callable(auth) else True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Vault availability check failed: %s", exc)
            return False

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import hvac  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "hvac not installed; Vault operations will fail closed. "
                "Install with: pip install 'hvac>=2.0'"
            )
            return None
        if not self.addr:
            logger.warning("VAULT_ADDR not set; refusing to target an unknown Vault.")
            return None
        self._client = hvac.Client(url=self.addr, token=self.token or None)
        return self._client

    # ---- KV v2 helpers ----------------------------------------------- #

    def read(self, path: str) -> Optional[Dict[str, Any]]:
        """Read a KV v2 secret. Returns ``None`` if missing."""
        c = self._get_client()
        if c is None:
            raise VaultUnavailableError("Vault client unavailable")
        try:
            resp = c.secrets.kv.v2.read_secret_version(  # type: ignore[attr-defined]
                path=path, mount_point=self.mount
            )
            data = resp.get("data", {}) if isinstance(resp, dict) else {}
            return dict(data.get("data") or {})
        except Exception as exc:  # noqa: BLE001
            # Treat "not found" as None; propagate anything else.
            msg = str(exc).lower()
            if "not found" in msg or "404" in msg:
                return None
            raise VaultUnavailableError(str(exc)) from exc

    def write(self, path: str, data: Dict[str, Any]) -> None:
        """Create/update a KV v2 secret."""
        c = self._get_client()
        if c is None:
            raise VaultUnavailableError("Vault client unavailable")
        c.secrets.kv.v2.create_or_update_secret(  # type: ignore[attr-defined]
            path=path,
            secret=data,
            mount_point=self.mount,
        )

    def delete(self, path: str) -> None:
        c = self._get_client()
        if c is None:
            raise VaultUnavailableError("Vault client unavailable")
        c.secrets.kv.v2.delete_metadata_and_all_versions(  # type: ignore[attr-defined]
            path=path, mount_point=self.mount
        )

    def whoami(self) -> Optional[str]:
        """Self-introspect the token for ``display_name`` / entity id."""
        c = self._get_client()
        if c is None:
            return None
        try:
            info = c.auth.token.lookup_self()  # type: ignore[attr-defined]
            data = info.get("data") if isinstance(info, dict) else None
            if isinstance(data, dict):
                return str(
                    data.get("display_name") or data.get("entity_id") or data.get("id") or ""
                ) or None
        except Exception:  # noqa: BLE001
            return None
        return None


# --------------------------------------------------------------------------- #
# Audit adapter.
# --------------------------------------------------------------------------- #


class AuditSink:
    """Best-effort SEC-005 audit log sink.

    We intentionally keep a thin wrapper here: the CLI must not hang on
    a missing audit backend. When the SEC-005 service can be imported
    and loaded, we emit through it; otherwise we fall back to a
    structured JSON line on a dedicated logger so SREs can still prove
    the rotation ran.
    """

    def __init__(self) -> None:
        self._service: Any = None
        self._tried = False
        self._logger = logging.getLogger(AUDIT_COMPONENT)

    def _load_service(self) -> Any:
        if self._tried:
            return self._service
        self._tried = True
        try:
            from greenlang.infrastructure.audit_service import (  # type: ignore[import-not-found]
                audit_service as _svc,
            )

            self._service = _svc
        except Exception:  # noqa: BLE001
            self._service = None
        return self._service

    def emit(
        self,
        *,
        action: str,
        operator: str,
        result: str,
        details: Dict[str, Any],
    ) -> None:
        """Write one audit record for a rotation step."""
        record = {
            "component": AUDIT_COMPONENT,
            "action": action,
            "operator": operator,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
        }
        # Attempt structured emission; fall back to JSON line.
        svc = self._load_service()
        if svc is not None:  # pragma: no cover - optional wiring
            try:
                emit = getattr(svc, "emit_sync", None) or getattr(svc, "log_event_sync", None)
                if callable(emit):
                    emit(record)
                    return
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Audit service emit failed: %s", exc)
        self._logger.info("%s", json.dumps(record, sort_keys=True, default=str))


# --------------------------------------------------------------------------- #
# Key generation / verification.
# --------------------------------------------------------------------------- #


def _generate_ed25519_pem() -> Dict[str, str]:
    """Generate a fresh Ed25519 keypair. Returns PEM-encoded pair."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public_pem = (
        private.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )
    return {"private_key_pem": private_pem, "public_key_pem": public_pem}


def _sign_with_pem(payload: Dict[str, Any], private_key_pem: str) -> str:
    """Sign the canary payload and return a base64 signature."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    key = serialization.load_pem_private_key(
        private_key_pem.encode("ascii"), password=None
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("Loaded key is not Ed25519")
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    sig = key.sign(canonical)
    return base64.b64encode(sig).decode("ascii")


def _verify_with_pem(
    payload: Dict[str, Any], signature_b64: str, public_key_pem: str
) -> bool:
    """Verify a base64 signature against the canary payload."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )

    pub = serialization.load_pem_public_key(public_key_pem.encode("ascii"))
    if not isinstance(pub, Ed25519PublicKey):
        return False
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    try:
        pub.verify(base64.b64decode(signature_b64), canonical)
        return True
    except InvalidSignature:
        return False


def _new_kid(prefix: str = "gl-factors-ed25519") -> str:
    """Stable, human-readable key id with a date stamp."""
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{os.urandom(3).hex()}"


# --------------------------------------------------------------------------- #
# Operator identity.
# --------------------------------------------------------------------------- #


def _resolve_operator(explicit: Optional[str], vault: VaultClient) -> str:
    """Best-effort identification of who is running the rotation step."""
    if explicit:
        return explicit
    env = os.getenv("USER") or os.getenv("USERNAME")
    if env:
        return env
    try:
        login = getpass.getuser()
    except Exception:  # noqa: BLE001
        login = ""
    if login:
        return login
    ident = vault.whoami()
    return ident or "unknown"


# --------------------------------------------------------------------------- #
# Subcommand handlers.
# --------------------------------------------------------------------------- #


def _archive_path_for(now: Optional[datetime] = None) -> str:
    ref = now or datetime.now(timezone.utc)
    return f"{KEY_PATH_ARCHIVE_PREFIX}/{ref.strftime('%Y%m%d')}"


def cmd_plan(vault: VaultClient, audit: AuditSink, args: argparse.Namespace) -> RotationReport:
    """``rotate-plan`` — read current key metadata + report rotation status."""
    if not vault.is_available():
        return RotationReport(
            action="rotate-plan",
            dry_run=True,
            ok=False,
            message="Vault unavailable — cannot read current key metadata.",
        )
    try:
        current = vault.read(KEY_PATH_CURRENT)
    except VaultUnavailableError as exc:
        return RotationReport(
            action="rotate-plan",
            dry_run=True,
            ok=False,
            message=f"Vault read failed: {exc}",
        )
    if not current:
        return RotationReport(
            action="rotate-plan",
            dry_run=True,
            ok=True,
            message="No current Ed25519 key — run rotate-stage then rotate-promote.",
            details={"current": None},
        )
    meta = KeyMetadata.from_dict(current)
    age = meta.age_days()
    next_due: Optional[str] = None
    overdue = False
    if age is not None:
        due = datetime.fromisoformat(meta.created_at.replace("Z", "+00:00")) + timedelta(
            days=ROTATION_CADENCE_DAYS
        )
        next_due = due.astimezone(timezone.utc).isoformat()
        overdue = age > ROTATION_CADENCE_DAYS
    return RotationReport(
        action="rotate-plan",
        dry_run=True,
        ok=True,
        message=(
            "Rotation overdue" if overdue else "Rotation within cadence."
        ),
        details={
            "current_kid": meta.kid,
            "created_at": meta.created_at,
            "age_days": None if age is None else round(age, 2),
            "cadence_days": ROTATION_CADENCE_DAYS,
            "next_rotation_due": next_due,
            "overdue": overdue,
        },
    )


def cmd_stage(vault: VaultClient, audit: AuditSink, args: argparse.Namespace) -> RotationReport:
    """``rotate-stage`` — generate ``next`` keypair, canary-sign with both."""
    if args.dry_run:
        return RotationReport(
            action="rotate-stage",
            dry_run=True,
            ok=True,
            message="Dry run — would generate a new keypair and stage at secret/factors/ed25519/next.",
        )
    if not vault.is_available():
        return RotationReport(
            action="rotate-stage",
            dry_run=False,
            ok=False,
            message="Vault unavailable — refusing to mutate.",
        )
    try:
        current = vault.read(KEY_PATH_CURRENT) or {}
        staged_existing = vault.read(KEY_PATH_NEXT)
    except VaultUnavailableError as exc:
        return RotationReport(
            action="rotate-stage", dry_run=False, ok=False, message=f"Vault read failed: {exc}"
        )
    if staged_existing:
        return RotationReport(
            action="rotate-stage",
            dry_run=False,
            ok=False,
            message="An Ed25519 next-key is already staged. Promote or retire before re-staging.",
            details={"staged_kid": staged_existing.get("kid")},
        )

    # Generate next keypair + sign canary with both (current if present).
    pair = _generate_ed25519_pem()
    new_meta = KeyMetadata(
        kid=_new_kid(),
        created_at=datetime.now(timezone.utc).isoformat(),
        private_key_pem=pair["private_key_pem"],
        public_key_pem=pair["public_key_pem"],
    )

    sig_new = _sign_with_pem(CANARY_PAYLOAD, new_meta.private_key_pem)
    if not _verify_with_pem(CANARY_PAYLOAD, sig_new, new_meta.public_key_pem):
        return RotationReport(
            action="rotate-stage",
            dry_run=False,
            ok=False,
            message="Canary verification failed with NEW key — aborting.",
        )

    sig_current: Optional[str] = None
    current_meta: Optional[KeyMetadata] = None
    if current:
        current_meta = KeyMetadata.from_dict(current)
        if current_meta.private_key_pem:
            sig_current = _sign_with_pem(CANARY_PAYLOAD, current_meta.private_key_pem)
            if not _verify_with_pem(
                CANARY_PAYLOAD, sig_current, current_meta.public_key_pem
            ):
                return RotationReport(
                    action="rotate-stage",
                    dry_run=False,
                    ok=False,
                    message="Canary verification failed with CURRENT key — aborting stage.",
                )

    # All canaries passed → persist the staged key.
    try:
        vault.write(KEY_PATH_NEXT, new_meta.to_dict())
    except VaultUnavailableError as exc:
        return RotationReport(
            action="rotate-stage", dry_run=False, ok=False, message=f"Vault write failed: {exc}"
        )

    operator = _resolve_operator(args.operator, vault)
    details = {
        "new_kid": new_meta.kid,
        "current_kid": current_meta.kid if current_meta else None,
        "canary_new_sig": sig_new,
        "canary_current_sig": sig_current,
        "transition_marker": "both_keys_canary_verified",
    }
    audit.emit(
        action="rotate-stage",
        operator=operator,
        result="success",
        details=details,
    )
    return RotationReport(
        action="rotate-stage",
        dry_run=False,
        ok=True,
        message="Staged new Ed25519 key; canary verified with both keys.",
        details=details,
    )


def cmd_promote(vault: VaultClient, audit: AuditSink, args: argparse.Namespace) -> RotationReport:
    """``rotate-promote`` — make ``next`` the active key; archive old, start grace."""
    if args.dry_run:
        return RotationReport(
            action="rotate-promote",
            dry_run=True,
            ok=True,
            message="Dry run — would promote staged key to current and start 30d grace.",
        )
    if not vault.is_available():
        return RotationReport(
            action="rotate-promote",
            dry_run=False,
            ok=False,
            message="Vault unavailable — refusing to mutate.",
        )
    try:
        staged = vault.read(KEY_PATH_NEXT)
    except VaultUnavailableError as exc:
        return RotationReport(
            action="rotate-promote",
            dry_run=False,
            ok=False,
            message=f"Vault read failed: {exc}",
        )
    if not staged:
        return RotationReport(
            action="rotate-promote",
            dry_run=False,
            ok=False,
            message="No staged key at secret/factors/ed25519/next — run rotate-stage first.",
        )
    new_meta = KeyMetadata.from_dict(staged)

    try:
        current = vault.read(KEY_PATH_CURRENT)
    except VaultUnavailableError as exc:
        return RotationReport(
            action="rotate-promote",
            dry_run=False,
            ok=False,
            message=f"Vault read failed: {exc}",
        )

    # Archive the old current under a date-stamped path so the grace
    # window can pull its public key for verification.
    archive_info: Dict[str, Any] = {}
    grace_until: Optional[str] = None
    if current:
        archive_path = _archive_path_for()
        try:
            vault.write(archive_path, current)
        except VaultUnavailableError as exc:
            return RotationReport(
                action="rotate-promote",
                dry_run=False,
                ok=False,
                message=f"Archiving old key failed: {exc}",
            )
        archive_info = {
            "archive_path": archive_path,
            "archived_kid": current.get("kid"),
        }
        grace_until = (
            datetime.now(timezone.utc) + timedelta(days=GRACE_PERIOD_DAYS)
        ).isoformat()

    # Promote next → current and clear next.
    try:
        vault.write(KEY_PATH_CURRENT, new_meta.to_dict())
        vault.delete(KEY_PATH_NEXT)
    except VaultUnavailableError as exc:
        return RotationReport(
            action="rotate-promote",
            dry_run=False,
            ok=False,
            message=f"Promotion failed: {exc}",
        )

    # Update verification allowlist: both kids accepted during grace.
    active_kids: List[str] = [new_meta.kid]
    grace_map: Dict[str, str] = {}
    if current and grace_until:
        old_kid = str(current.get("kid") or "")
        if old_kid:
            active_kids.append(old_kid)
            grace_map[old_kid] = grace_until
    try:
        vault.write(
            KEY_PATH_ALLOWLIST,
            {"active_kids": active_kids, "grace_until": grace_map},
        )
    except VaultUnavailableError as exc:  # pragma: no cover
        return RotationReport(
            action="rotate-promote",
            dry_run=False,
            ok=False,
            message=f"Allowlist write failed: {exc}",
        )

    operator = _resolve_operator(args.operator, vault)
    details: Dict[str, Any] = {
        "new_kid": new_meta.kid,
        "active_kids": active_kids,
        "grace_until": grace_map,
        **archive_info,
    }
    audit.emit(
        action="rotate-promote",
        operator=operator,
        result="success",
        details=details,
    )
    return RotationReport(
        action="rotate-promote",
        dry_run=False,
        ok=True,
        message="Promoted staged key to current; grace window started.",
        details=details,
    )


def cmd_retire_old(
    vault: VaultClient, audit: AuditSink, args: argparse.Namespace
) -> RotationReport:
    """``rotate-retire-old`` — drop expired kids from the verify allowlist."""
    if args.dry_run:
        return RotationReport(
            action="rotate-retire-old",
            dry_run=True,
            ok=True,
            message="Dry run — would drop expired kids from the allowlist.",
        )
    if not vault.is_available():
        return RotationReport(
            action="rotate-retire-old",
            dry_run=False,
            ok=False,
            message="Vault unavailable — refusing to mutate.",
        )
    try:
        allowlist = vault.read(KEY_PATH_ALLOWLIST) or {}
    except VaultUnavailableError as exc:
        return RotationReport(
            action="rotate-retire-old",
            dry_run=False,
            ok=False,
            message=f"Vault read failed: {exc}",
        )

    active: List[str] = list(allowlist.get("active_kids") or [])
    grace: Dict[str, str] = dict(allowlist.get("grace_until") or {})

    now = datetime.now(timezone.utc)
    retired: List[str] = []
    new_grace: Dict[str, str] = {}
    for kid, until in grace.items():
        try:
            dt = datetime.fromisoformat(str(until).replace("Z", "+00:00"))
        except ValueError:
            dt = now  # malformed → treat as expired
        if dt <= now:
            retired.append(kid)
        else:
            new_grace[kid] = until

    if not retired:
        return RotationReport(
            action="rotate-retire-old",
            dry_run=False,
            ok=True,
            message="No kids are past grace — nothing to retire.",
            details={"active_kids": active, "grace_until": grace},
        )

    new_active = [k for k in active if k not in retired]
    try:
        vault.write(
            KEY_PATH_ALLOWLIST,
            {"active_kids": new_active, "grace_until": new_grace},
        )
    except VaultUnavailableError as exc:  # pragma: no cover
        return RotationReport(
            action="rotate-retire-old",
            dry_run=False,
            ok=False,
            message=f"Allowlist write failed: {exc}",
        )

    operator = _resolve_operator(args.operator, vault)
    details = {
        "retired_kids": retired,
        "active_kids": new_active,
        "grace_until": new_grace,
    }
    audit.emit(
        action="rotate-retire-old",
        operator=operator,
        result="success",
        details=details,
    )
    return RotationReport(
        action="rotate-retire-old",
        dry_run=False,
        ok=True,
        message=f"Retired {len(retired)} kid(s) past grace.",
        details=details,
    )


# --------------------------------------------------------------------------- #
# CLI wiring.
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="factors_key_rotation.py",
        description="Vault-backed Ed25519 rotation for Factors signed receipts.",
    )
    parser.add_argument(
        "--operator",
        default=None,
        help="Operator identity for audit. Defaults to $USER / $USERNAME.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=None,
        help="Plan only; DO NOT mutate Vault (default).",
    )
    group.add_argument(
        "--live",
        dest="dry_run",
        action="store_false",
        help="Actually mutate Vault. Requires VAULT_TOKEN.",
    )
    parser.add_argument(
        "--vault-addr",
        default=None,
        help="Override VAULT_ADDR (default: env var).",
    )
    parser.add_argument(
        "--vault-token",
        default=None,
        help="Override VAULT_TOKEN (default: env var; NOT recommended on CLI).",
    )
    parser.add_argument(
        "--vault-mount",
        default=None,
        help="KV v2 mount point (default: 'secret').",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("rotate-plan", "rotate-stage", "rotate-promote", "rotate-retire-old"):
        sp = sub.add_parser(name, help=f"{name} subcommand")
        sp.set_defaults(command=name)
        # Allow the dry-run / live / operator / vault flags to appear AFTER
        # the subcommand as well as before it.  Tests + real operators
        # both expect ``rotate-stage --live --operator alice`` to work.
        sp_group = sp.add_mutually_exclusive_group()
        sp_group.add_argument(
            "--dry-run",
            dest="dry_run",
            action="store_true",
            default=None,
            help="Plan only; DO NOT mutate Vault (default).",
        )
        sp_group.add_argument(
            "--live",
            dest="dry_run",
            action="store_false",
            help="Actually mutate Vault. Requires VAULT_TOKEN.",
        )
        sp.add_argument(
            "--operator",
            default=None,
            dest="operator_sub",
            help="Operator identity for audit.",
        )
        sp.add_argument("--vault-addr", default=None, dest="vault_addr_sub")
        sp.add_argument("--vault-token", default=None, dest="vault_token_sub")
        sp.add_argument("--vault-mount", default=None, dest="vault_mount_sub")
    return parser


_COMMANDS = {
    "rotate-plan": cmd_plan,
    "rotate-stage": cmd_stage,
    "rotate-promote": cmd_promote,
    "rotate-retire-old": cmd_retire_old,
}


def run(
    argv: Optional[List[str]] = None,
    *,
    vault: Optional[VaultClient] = None,
    audit: Optional[AuditSink] = None,
) -> RotationReport:
    """Run the CLI programmatically (used by tests)."""
    args = build_parser().parse_args(argv)
    # dry_run defaults to True across the parser hierarchy: the CLI is
    # safety-first, mutations only happen on explicit ``--live``.
    if getattr(args, "dry_run", None) is None:
        args.dry_run = True
    for top, sub in (
        ("operator", "operator_sub"),
        ("vault_addr", "vault_addr_sub"),
        ("vault_token", "vault_token_sub"),
        ("vault_mount", "vault_mount_sub"),
    ):
        sub_val = getattr(args, sub, None)
        if sub_val is not None:
            setattr(args, top, sub_val)
    v = vault or VaultClient(
        addr=args.vault_addr,
        token=args.vault_token,
        mount=args.vault_mount or VAULT_MOUNT_DEFAULT,
    )
    a = audit or AuditSink()
    handler = _COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse requires command
        return RotationReport(action="unknown", dry_run=True, ok=False, message="Unknown command")
    try:
        return handler(v, a, args)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected failure in %s", args.command)
        return RotationReport(
            action=args.command,
            dry_run=args.dry_run,
            ok=False,
            message=f"Unexpected failure: {exc}",
        )


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI wrapper
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(argv)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
    if not report.ok:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
