# -*- coding: utf-8 -*-
"""
OEM bulk-dump export (Wave 5 - Track C-5).

Produces a signed artifact containing *only* the factors an OEM partner
is entitled to redistribute.  The artifact is a JSONL payload + manifest
bundled as a single JSON envelope (so tests and SDKs do not need a ZIP
toolchain in their hot path).  A companion :class:`SignedArtifact`
object carries the receipt emitted by
:mod:`greenlang.factors.signing` so callers can persist the whole thing
alongside the OEM's branding metadata.

Design constraints
------------------
* **No LLM in the filter path.**  The decision "does this factor belong
  in this OEM's dump?" is a deterministic license-class lookup against
  :func:`greenlang.factors.entitlements.check_oem_redistribution` plus
  a per-OEM quota counter.
* **Tenant-scoped quota enforcement.**  Each OEM has a per-day quota
  (``OEM_EXPORT_DAILY_QUOTA``, overridable via env var).  Exceeding the
  quota surfaces as :class:`OemExportQuotaError` so the API layer can
  translate to HTTP 429.
* **Signed artifact.**  The payload + manifest are signed with the
  existing signing module (HMAC in dev, Ed25519 when the key is set).
  We never reimplement crypto.
* **Branding bundled.**  The manifest embeds the OEM's branding payload
  so downstream consumers can theme without an extra API round-trip.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from greenlang.factors.entitlements import check_oem_redistribution
from greenlang.factors.onboarding.partner_setup import (
    OemError,
    OemPartner,
    get_oem_partner,
)
from greenlang.factors.signing import (
    Receipt,
    SigningError,
    sign_ed25519,
    sign_sha256_hmac,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------


def _default_daily_quota() -> int:
    """Pull the per-OEM daily export quota from env, defaulting to 25k rows."""
    raw = os.getenv("GL_FACTORS_OEM_EXPORT_DAILY_QUOTA", "").strip()
    if not raw:
        return 25_000
    try:
        value = int(raw)
    except ValueError:
        return 25_000
    return max(value, 0)


OEM_EXPORT_DAILY_QUOTA: int = _default_daily_quota()

#: Max rows returned in a single export call (safety limit independent
#: of quota so a misconfigured OEM cannot blow up memory).
OEM_EXPORT_MAX_ROWS: int = 100_000


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OemExportError(RuntimeError):
    """Generic export failure (invalid OEM, no entitled rows, etc.)."""


class OemExportQuotaError(OemExportError):
    """Raised when an OEM attempts to exceed its per-day quota."""


# ---------------------------------------------------------------------------
# Quota ledger (in-memory, process-local)
# ---------------------------------------------------------------------------


@dataclass
class _QuotaWindow:
    """Per-day rolling counter keyed by ``oem_id``."""

    day: str
    rows_exported: int = 0

    def add(self, rows: int) -> int:
        self.rows_exported += max(int(rows), 0)
        return self.rows_exported


_QUOTA_LOCK = threading.Lock()
_QUOTA_LEDGER: Dict[str, _QuotaWindow] = {}


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _reset_quota_ledger() -> None:
    """Test-only helper - wipe the in-memory quota counter."""
    with _QUOTA_LOCK:
        _QUOTA_LEDGER.clear()


def _quota_for(oem_id: str) -> int:
    """Current rows-exported counter for today (0 when new)."""
    with _QUOTA_LOCK:
        window = _QUOTA_LEDGER.get(oem_id)
        today = _today_utc()
        if window is None or window.day != today:
            return 0
        return window.rows_exported


def _bump_quota(oem_id: str, rows: int) -> int:
    """Add ``rows`` to the OEM's counter; returns the new total."""
    with _QUOTA_LOCK:
        today = _today_utc()
        window = _QUOTA_LEDGER.get(oem_id)
        if window is None or window.day != today:
            window = _QuotaWindow(day=today, rows_exported=0)
            _QUOTA_LEDGER[oem_id] = window
        return window.add(rows)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SignedArtifact:
    """A JSONL bulk dump with manifest + signature.

    ``payload_jsonl`` is newline-delimited JSON of one factor per line so
    a downstream streamer can parse incrementally. The ``manifest`` is a
    small dict describing the dump. ``receipt`` is the crypto-signed
    envelope emitted by :mod:`greenlang.factors.signing`.
    """

    artifact_id: str
    oem_id: str
    edition_id: str
    factor_count: int
    manifest: Dict[str, Any]
    payload_jsonl: str
    receipt: Receipt
    content_hash: str

    def to_envelope(self) -> Dict[str, Any]:
        """Return the wire-form envelope used by the /v1/oem/export route."""
        return {
            "artifact_id": self.artifact_id,
            "oem_id": self.oem_id,
            "edition_id": self.edition_id,
            "factor_count": self.factor_count,
            "manifest": self.manifest,
            "payload_jsonl": self.payload_jsonl,
            "signed_receipt": self.receipt.to_dict(),
            "content_hash": self.content_hash,
        }


# ---------------------------------------------------------------------------
# Core filter + sign pipeline
# ---------------------------------------------------------------------------


def _factor_to_dict(factor: Any) -> Dict[str, Any]:
    """Coerce a catalog factor (dict or dataclass) into a plain dict."""
    if isinstance(factor, dict):
        return dict(factor)
    if hasattr(factor, "to_dict"):
        try:
            return factor.to_dict()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(factor, "model_dump"):
        try:
            return factor.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            pass
    # Best-effort fallback: dataclass-ish objects with __dict__.
    state = getattr(factor, "__dict__", None)
    if isinstance(state, dict):
        return dict(state)
    return {"factor_id": str(factor)}


def _filter_entitled_rows(
    partner: OemPartner,
    rows: Iterable[Any],
) -> List[Dict[str, Any]]:
    """Return only the rows within the OEM's redistribution grant."""
    allowed: List[Dict[str, Any]] = []
    for raw in rows:
        if check_oem_redistribution(partner.id, raw):
            allowed.append(_factor_to_dict(raw))
    return allowed


def _sign_artifact_payload(payload: Mapping[str, Any]) -> Receipt:
    """Prefer Ed25519 if a private key is configured, else HMAC."""
    try:
        if os.getenv("GL_FACTORS_ED25519_PRIVATE_KEY"):
            return sign_ed25519(payload, key_id="gl-factors-oem-export")
    except SigningError as exc:
        logger.warning("Ed25519 sign failed for OEM export: %s; falling back to HMAC", exc)
    return sign_sha256_hmac(payload, key_id="gl-factors-oem-export")


def _resolve_branding_metadata(partner: OemPartner) -> Optional[Dict[str, Any]]:
    """Materialise the OEM's branding (pydantic, dict, or None)."""
    branding = partner.branding
    if branding is None:
        return None
    if hasattr(branding, "to_response_metadata"):
        try:
            return branding.to_response_metadata()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(branding, "model_dump"):
        try:
            return {
                k: v
                for k, v in branding.model_dump(mode="json").items()
                if v is not None
            }
        except Exception:  # noqa: BLE001
            pass
    if isinstance(branding, dict):
        return {k: v for k, v in branding.items() if v is not None}
    return None


def build_oem_export(
    oem_id: str,
    *,
    edition_id: str,
    rows: Iterable[Any],
    quota_override: Optional[int] = None,
) -> SignedArtifact:
    """Filter ``rows`` to the OEM's grant and return a signed bulk dump.

    Args:
        oem_id: Target OEM partner id (must be registered).
        edition_id: Edition slug that originated the rows.
        rows: Iterable of canonical factor records (dicts or dataclasses).
        quota_override: Optional test-only per-day cap override; defaults
            to :data:`OEM_EXPORT_DAILY_QUOTA`.

    Returns:
        :class:`SignedArtifact` carrying the JSONL payload + manifest
        + crypto receipt.

    Raises:
        OemExportError: On unknown OEM, inactive OEM, or empty entitled
            row set (a partner that cannot export anything should not be
            able to ship an empty signed dump).
        OemExportQuotaError: When adding this export would exceed the
            OEM's per-day quota.
    """
    try:
        partner = get_oem_partner(oem_id)
    except OemError as exc:
        raise OemExportError(f"Unknown OEM {oem_id!r}: {exc}") from exc
    if not partner.active:
        raise OemExportError(f"OEM {oem_id!r} is not active")

    entitled_rows = _filter_entitled_rows(partner, rows)
    if not entitled_rows:
        raise OemExportError(
            f"OEM {oem_id!r} has no entitled rows to export from edition {edition_id!r}; "
            "check the partner's redistribution grant."
        )
    if len(entitled_rows) > OEM_EXPORT_MAX_ROWS:
        raise OemExportError(
            f"Export size {len(entitled_rows)} exceeds hard cap {OEM_EXPORT_MAX_ROWS}; "
            "use a narrower edition filter."
        )

    # Quota check (against today's ledger).
    quota = OEM_EXPORT_DAILY_QUOTA if quota_override is None else int(quota_override)
    used = _quota_for(oem_id)
    projected = used + len(entitled_rows)
    if quota and projected > quota:
        raise OemExportQuotaError(
            f"OEM {oem_id!r} daily export quota exceeded: "
            f"used={used} batch={len(entitled_rows)} quota={quota}"
        )

    # Stable artifact id: sha256 over oem_id + edition_id + factor_ids.
    stable_seed = json.dumps(
        {
            "oem_id": oem_id,
            "edition_id": edition_id,
            "factor_ids": sorted(
                [r.get("factor_id", "") for r in entitled_rows if isinstance(r, dict)]
            ),
        },
        sort_keys=True,
    )
    artifact_id = "oem-export-" + hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:16]

    payload_jsonl = "\n".join(json.dumps(r, sort_keys=True, default=str) for r in entitled_rows)
    content_hash = hashlib.sha256(payload_jsonl.encode("utf-8")).hexdigest()

    manifest: Dict[str, Any] = {
        "artifact_id": artifact_id,
        "oem_id": partner.id,
        "oem_name": partner.name,
        "parent_plan": partner.parent_plan,
        "edition_id": edition_id,
        "factor_count": len(entitled_rows),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "content_hash": content_hash,
        "allowed_classes": list(partner.grant.allowed_classes),
        "powered_by": "greenlang-factors",
    }
    branding_meta = _resolve_branding_metadata(partner)
    if branding_meta is not None:
        manifest["branding"] = branding_meta

    # Sign the manifest — the content_hash commits the payload, so
    # signing the manifest is sufficient to authenticate the full dump.
    receipt = _sign_artifact_payload(manifest)

    # Only charge quota after we have successfully signed the artifact.
    _bump_quota(oem_id, len(entitled_rows))

    logger.info(
        "OEM export built: oem=%s edition=%s rows=%d artifact=%s",
        oem_id, edition_id, len(entitled_rows), artifact_id,
    )
    return SignedArtifact(
        artifact_id=artifact_id,
        oem_id=partner.id,
        edition_id=edition_id,
        factor_count=len(entitled_rows),
        manifest=manifest,
        payload_jsonl=payload_jsonl,
        receipt=receipt,
        content_hash=content_hash,
    )


__all__ = [
    "OEM_EXPORT_DAILY_QUOTA",
    "OEM_EXPORT_MAX_ROWS",
    "OemExportError",
    "OemExportQuotaError",
    "SignedArtifact",
    "build_oem_export",
    "_reset_quota_ledger",
]
