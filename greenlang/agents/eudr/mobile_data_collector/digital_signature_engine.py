# -*- coding: utf-8 -*-
"""
Digital Signature Engine - AGENT-EUDR-015

Engine 6: Simulated ECDSA P-256 digital signature capture with timestamp
binding, multi-signature workflows, and revocation support for EUDR
mobile data collection.

This engine manages the data-model-first signature workflow for field
declarations, custody transfers, and inspection sign-offs required by
EU 2023/1115 Articles 4, 9, and 14. Actual cryptographic operations
are simulated using SHA-256 hashes; production deployment would plug
in real ECDSA P-256 via a hardware security module or platform keystore.

Capabilities:
    - Simulated ECDSA P-256 key pair generation (SHA-256 based)
    - Signature binding: form_id + timestamp + signer_id + data_hash
    - Deterministic signature ID generation
    - Multi-signature workflows (producer -> inspector -> buyer)
    - Visual signature SVG touch-path capture metadata
    - Timestamp binding with ISO 8601 ms precision
    - Signature verification (hash comparison + timestamp + authorization)
    - Signature chain validation (ordered multi-sig verification)
    - Revocation with reason tracking and time window enforcement
    - Signer identity management (role, org, validity period)
    - Custody transfer signatures (from_party, to_party, witnessed_by)

Zero-Hallucination Guarantees:
    - All signatures are deterministic SHA-256 hashes
    - Verification is hash comparison (no probabilistic operations)
    - Timestamp validation is arithmetic comparison
    - No actual cryptographic operations (by design)

Example:
    >>> from greenlang.agents.eudr.mobile_data_collector.digital_signature_engine import (
    ...     DigitalSignatureEngine,
    ... )
    >>> engine = DigitalSignatureEngine()
    >>> signer = engine.register_signer(
    ...     signer_id="signer-001", name="John Producer",
    ...     role="producer", organization="Farm Co-op",
    ... )
    >>> sig = engine.create_signature(
    ...     form_id="form-001", signer_id="signer-001",
    ...     data_hash="abc123...", device_id="device-001",
    ... )
    >>> result = engine.verify_signature(sig["signature_id"])

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015 Mobile Data Collector (GL-EUDR-MDC-015)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .config import get_config
from .metrics import record_api_error, record_signature_captured
from .provenance import get_provenance_tracker

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid signer roles for EUDR supply chain.
SIGNER_ROLES: frozenset = frozenset({
    "producer", "collector", "cooperative_manager",
    "inspector", "transport_operator", "warehouse_manager",
    "buyer", "exporter", "importer", "auditor", "system",
})

#: Valid signature statuses.
SIGNATURE_STATUSES: frozenset = frozenset({
    "valid", "revoked", "expired", "pending_verification",
})

#: Valid revocation reasons.
REVOCATION_REASONS: frozenset = frozenset({
    "signer_error", "data_correction", "fraud_detected",
    "unauthorized_signer", "form_invalidated", "system_error",
    "signer_request", "administrative",
})

#: Custody transfer witness roles.
WITNESS_ROLES: frozenset = frozenset({
    "inspector", "auditor", "cooperative_manager",
    "transport_operator", "system",
})

def _utcnow_iso() -> str:
    """Return current UTC datetime as ISO 8601 string with ms precision."""
    return utcnow().isoformat(timespec="milliseconds")

# ---------------------------------------------------------------------------
# DigitalSignatureEngine
# ---------------------------------------------------------------------------

class DigitalSignatureEngine:
    """Simulated ECDSA P-256 digital signature engine for EUDR mobile data.

    Manages signer registration, signature creation with timestamp binding,
    multi-signature workflows, verification, revocation, and custody
    transfer signatures. All cryptographic operations are simulated
    using SHA-256; production deployment plugs in real ECDSA P-256.

    Attributes:
        _config: Mobile data collector configuration.
        _provenance: Provenance tracker for audit trails.
        _signers: Registered signer identities keyed by signer_id.
        _signatures: Created signatures keyed by signature_id.
        _multi_sig_chains: Multi-signature chain records keyed by chain_id.
        _lock: Thread-safe lock.

    Example:
        >>> engine = DigitalSignatureEngine()
        >>> engine.register_signer("s1", "Alice", "producer", "FarmOrg")
        >>> sig = engine.create_signature("form-1", "s1", "datahash", "dev-1")
        >>> engine.verify_signature(sig["signature_id"])
    """

    __slots__ = (
        "_config", "_provenance", "_signers", "_signatures",
        "_multi_sig_chains", "_lock",
    )

    def __init__(self) -> None:
        """Initialize DigitalSignatureEngine with config and provenance."""
        self._config = get_config()
        self._provenance = get_provenance_tracker()
        self._signers: Dict[str, Dict[str, Any]] = {}
        self._signatures: Dict[str, Dict[str, Any]] = {}
        self._multi_sig_chains: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        logger.info(
            "DigitalSignatureEngine initialized: algorithm=%s, "
            "timestamp_binding=%s, expiry=%dd, revocation_window=%dh, "
            "multi_sig=%s, visual_sig=%s",
            self._config.signature_algorithm,
            self._config.enable_timestamp_binding,
            self._config.signature_expiry_days,
            self._config.revocation_window_hours,
            self._config.enable_multi_signature,
            self._config.enable_visual_signature,
        )

    # ------------------------------------------------------------------
    # Signer Management
    # ------------------------------------------------------------------

    def register_signer(
        self,
        signer_id: str,
        name: str,
        role: str,
        organization: str = "",
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a signer identity for signature operations.

        Args:
            signer_id: Unique signer identifier.
            name: Full name of the signer.
            role: Signer role (producer, inspector, buyer, etc.).
            organization: Organization name.
            valid_from: ISO 8601 validity start (default: now).
            valid_to: ISO 8601 validity end (default: +5 years).
            metadata: Additional signer metadata.

        Returns:
            Signer registration dictionary.

        Raises:
            ValueError: If signer_id is empty, role is invalid, or
                signer already exists.
        """
        if not signer_id or not signer_id.strip():
            raise ValueError("signer_id must not be empty")
        if not name or not name.strip():
            raise ValueError("name must not be empty")
        if role not in SIGNER_ROLES:
            raise ValueError(
                f"Invalid role '{role}'. Must be one of: {sorted(SIGNER_ROLES)}"
            )

        now = utcnow()
        if valid_from is None:
            valid_from = now.isoformat(timespec="milliseconds")
        if valid_to is None:
            valid_to = (now + timedelta(days=365 * 5)).isoformat(
                timespec="milliseconds",
            )

        # Generate simulated key pair (SHA-256 based, NOT real crypto)
        key_material = f"{signer_id}:{name}:{role}:{now.isoformat()}"
        simulated_public_key = hashlib.sha256(
            f"pub:{key_material}".encode("utf-8"),
        ).hexdigest()
        simulated_fingerprint = hashlib.sha256(
            simulated_public_key.encode("utf-8"),
        ).hexdigest()[:40]

        signer: Dict[str, Any] = {
            "signer_id": signer_id,
            "name": name,
            "role": role,
            "organization": organization,
            "public_key_hex": simulated_public_key,
            "fingerprint": simulated_fingerprint,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "is_active": True,
            "signature_count": 0,
            "metadata": copy.deepcopy(metadata or {}),
            "registered_at": now.isoformat(timespec="milliseconds"),
        }

        with self._lock:
            if signer_id in self._signers:
                raise ValueError(f"Signer already registered: {signer_id}")
            self._signers[signer_id] = signer

        self._record_provenance(signer_id, "register", signer)
        logger.info(
            "Signer registered: id=%s name='%s' role=%s org='%s'",
            signer_id, name, role, organization,
        )
        return copy.deepcopy(signer)

    def get_signer(self, signer_id: str) -> Dict[str, Any]:
        """Retrieve a registered signer by ID.

        Args:
            signer_id: Signer identifier.

        Returns:
            Signer dictionary.

        Raises:
            KeyError: If signer not found.
        """
        with self._lock:
            signer = self._signers.get(signer_id)
        if signer is None:
            raise KeyError(f"Signer not found: {signer_id}")
        return copy.deepcopy(signer)

    def list_signers(
        self,
        role: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """List registered signers with optional filters.

        Args:
            role: Filter by signer role.
            is_active: Filter by active status.

        Returns:
            List of signer dictionaries.
        """
        with self._lock:
            signers = list(self._signers.values())

        if role is not None:
            signers = [s for s in signers if s["role"] == role]
        if is_active is not None:
            signers = [s for s in signers if s["is_active"] == is_active]

        return [copy.deepcopy(s) for s in signers]

    def deactivate_signer(self, signer_id: str, reason: str = "") -> Dict[str, Any]:
        """Deactivate a signer, preventing new signatures.

        Args:
            signer_id: Signer to deactivate.
            reason: Reason for deactivation.

        Returns:
            Updated signer dictionary.

        Raises:
            KeyError: If signer not found.
        """
        with self._lock:
            signer = self._signers.get(signer_id)
            if signer is None:
                raise KeyError(f"Signer not found: {signer_id}")
            signer["is_active"] = False
            signer["metadata"]["deactivation_reason"] = reason
            signer["metadata"]["deactivated_at"] = _utcnow_iso()

        self._record_provenance(signer_id, "update", signer)
        logger.info("Signer deactivated: id=%s reason='%s'", signer_id, reason)
        return copy.deepcopy(signer)

    # ------------------------------------------------------------------
    # Signature Creation
    # ------------------------------------------------------------------

    def create_signature(
        self,
        form_id: str,
        signer_id: str,
        data_hash: str,
        device_id: str,
        visual_svg: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a digital signature binding form, signer, data, and timestamp.

        Generates a deterministic signature by SHA-256 hashing the
        concatenation of form_id, timestamp, signer_id, and data_hash.
        This is a simulated ECDSA P-256 signature; production deployment
        would use the device keystore.

        Args:
            form_id: Form submission being signed.
            signer_id: Registered signer performing the signature.
            data_hash: SHA-256 hash of the data being signed.
            device_id: Device used for signing.
            visual_svg: Optional SVG touch-path of handwritten signature.
            metadata: Additional signature metadata.

        Returns:
            Signature dictionary.

        Raises:
            KeyError: If signer_id not registered.
            ValueError: If signer is not active or authorized.
        """
        start = time.monotonic()

        self._validate_signer_authorized(signer_id)

        now = utcnow()
        timestamp_iso = now.isoformat(timespec="milliseconds")

        # Deterministic signature ID from binding components
        sig_id = self._compute_signature_id(
            form_id, timestamp_iso, signer_id, data_hash,
        )

        # Simulated ECDSA signature (SHA-256 of binding payload)
        signature_hex = self._compute_simulated_signature(
            form_id, timestamp_iso, signer_id, data_hash,
        )

        with self._lock:
            signer = self._signers[signer_id]
            fingerprint = signer["fingerprint"]
            signer["signature_count"] += 1

        expiry_date = (
            now + timedelta(days=self._config.signature_expiry_days)
        ).isoformat(timespec="milliseconds")

        signature: Dict[str, Any] = {
            "signature_id": sig_id,
            "form_id": form_id,
            "signer_id": signer_id,
            "signer_name": signer["name"],
            "signer_role": signer["role"],
            "device_id": device_id,
            "algorithm": self._config.signature_algorithm,
            "public_key_fingerprint": fingerprint,
            "signature_hex": signature_hex,
            "signed_data_hash": data_hash,
            "timestamp_binding": timestamp_iso if self._config.enable_timestamp_binding else None,
            "visual_signature_svg": visual_svg if self._config.enable_visual_signature else None,
            "status": "valid",
            "is_valid": True,
            "is_revoked": False,
            "revocation_reason": None,
            "revoked_at": None,
            "expires_at": expiry_date,
            "metadata": copy.deepcopy(metadata or {}),
            "created_at": timestamp_iso,
        }

        with self._lock:
            self._signatures[sig_id] = signature

        record_signature_captured()
        self._record_provenance(sig_id, "sign", signature)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Signature created: id=%s form=%s signer=%s elapsed=%.1fms",
            sig_id[:12], form_id[:12], signer_id, elapsed,
        )
        return copy.deepcopy(signature)

    def get_signature(self, signature_id: str) -> Dict[str, Any]:
        """Retrieve a signature by ID.

        Args:
            signature_id: Signature identifier.

        Returns:
            Signature dictionary.

        Raises:
            KeyError: If signature not found.
        """
        with self._lock:
            sig = self._signatures.get(signature_id)
        if sig is None:
            raise KeyError(f"Signature not found: {signature_id}")
        return copy.deepcopy(sig)

    def list_signatures(
        self,
        form_id: Optional[str] = None,
        signer_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List signatures with optional filters.

        Args:
            form_id: Filter by form ID.
            signer_id: Filter by signer ID.
            status: Filter by signature status.

        Returns:
            List of signature dictionaries.
        """
        with self._lock:
            sigs = list(self._signatures.values())

        if form_id is not None:
            sigs = [s for s in sigs if s["form_id"] == form_id]
        if signer_id is not None:
            sigs = [s for s in sigs if s["signer_id"] == signer_id]
        if status is not None:
            sigs = [s for s in sigs if s["status"] == status]

        return [copy.deepcopy(s) for s in sigs]

    # ------------------------------------------------------------------
    # Signature Verification
    # ------------------------------------------------------------------

    def verify_signature(self, signature_id: str) -> Dict[str, Any]:
        """Verify a signature's integrity, timestamp, and signer authorization.

        Verification checks:
            1. Signature exists and is not revoked
            2. Signature has not expired
            3. Signer is registered and was active at signing time
            4. Re-computed signature hash matches stored signature_hex
            5. Timestamp binding is present (if configured)

        Args:
            signature_id: Signature to verify.

        Returns:
            Verification result dict with "verified" (bool),
            "checks" (list), "signature_id", "timestamp".

        Raises:
            KeyError: If signature not found.
        """
        start = time.monotonic()

        sig = self.get_signature(signature_id)
        checks: List[Dict[str, Any]] = []
        all_passed = True

        # Check 1: Not revoked
        revoked_check = {
            "check": "not_revoked",
            "passed": not sig["is_revoked"],
            "detail": "Signature has not been revoked" if not sig["is_revoked"]
                      else f"Revoked: {sig.get('revocation_reason', 'unknown')}",
        }
        checks.append(revoked_check)
        if not revoked_check["passed"]:
            all_passed = False

        # Check 2: Not expired
        expired = self._is_expired(sig)
        expiry_check = {
            "check": "not_expired",
            "passed": not expired,
            "detail": f"Expires at {sig['expires_at']}" if not expired
                      else "Signature has expired",
        }
        checks.append(expiry_check)
        if not expiry_check["passed"]:
            all_passed = False

        # Check 3: Signer exists
        signer_valid = self._is_signer_valid(sig["signer_id"])
        signer_check = {
            "check": "signer_valid",
            "passed": signer_valid,
            "detail": f"Signer {sig['signer_id']} is registered"
                      if signer_valid else "Signer not found or inactive",
        }
        checks.append(signer_check)
        if not signer_check["passed"]:
            all_passed = False

        # Check 4: Hash integrity
        recomputed = self._compute_simulated_signature(
            sig["form_id"],
            sig["timestamp_binding"] or sig["created_at"],
            sig["signer_id"],
            sig["signed_data_hash"],
        )
        hash_match = recomputed == sig["signature_hex"]
        hash_check = {
            "check": "hash_integrity",
            "passed": hash_match,
            "detail": "Signature hash matches" if hash_match
                      else "Signature hash mismatch (possible tampering)",
        }
        checks.append(hash_check)
        if not hash_check["passed"]:
            all_passed = False

        # Check 5: Timestamp binding
        if self._config.enable_timestamp_binding:
            ts_present = sig["timestamp_binding"] is not None
            ts_check = {
                "check": "timestamp_binding",
                "passed": ts_present,
                "detail": f"Timestamp bound: {sig['timestamp_binding']}"
                          if ts_present else "Missing timestamp binding",
            }
            checks.append(ts_check)
            if not ts_check["passed"]:
                all_passed = False

        result = {
            "verified": all_passed,
            "signature_id": signature_id,
            "checks": checks,
            "checks_passed": sum(1 for c in checks if c["passed"]),
            "checks_total": len(checks),
            "timestamp": _utcnow_iso(),
        }

        self._record_provenance(signature_id, "verify", result)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Signature verified: id=%s result=%s checks=%d/%d elapsed=%.1fms",
            signature_id[:12], all_passed,
            result["checks_passed"], result["checks_total"], elapsed,
        )
        return result

    def validate_timestamp(
        self,
        signature_id: str,
        reference_time: Optional[str] = None,
        tolerance_seconds: int = 60,
    ) -> Dict[str, Any]:
        """Validate a signature's timestamp against a reference time.

        Args:
            signature_id: Signature to validate.
            reference_time: ISO 8601 reference time (default: now).
            tolerance_seconds: Acceptable drift in seconds.

        Returns:
            Validation result dict.

        Raises:
            KeyError: If signature not found.
        """
        sig = self.get_signature(signature_id)
        ts_str = sig.get("timestamp_binding") or sig["created_at"]

        try:
            sig_time = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            return {
                "valid": False,
                "reason": f"Invalid timestamp format: {ts_str}",
            }

        if reference_time:
            ref_time = datetime.fromisoformat(reference_time)
        else:
            ref_time = utcnow()

        if sig_time.tzinfo is None:
            sig_time = sig_time.replace(tzinfo=timezone.utc)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        delta = abs((ref_time - sig_time).total_seconds())
        within_tolerance = delta <= tolerance_seconds

        return {
            "valid": within_tolerance,
            "signature_timestamp": ts_str,
            "reference_time": ref_time.isoformat(timespec="milliseconds"),
            "delta_seconds": round(delta, 3),
            "tolerance_seconds": tolerance_seconds,
        }

    # ------------------------------------------------------------------
    # Revocation
    # ------------------------------------------------------------------

    def revoke_signature(
        self,
        signature_id: str,
        reason: str,
        revoked_by: str = "system",
    ) -> Dict[str, Any]:
        """Revoke a signature with reason tracking.

        Revocation is only allowed within the configured revocation
        window (default 24 hours from creation).

        Args:
            signature_id: Signature to revoke.
            reason: Reason for revocation.
            revoked_by: ID of the person/system revoking.

        Returns:
            Updated signature dictionary.

        Raises:
            KeyError: If signature not found.
            ValueError: If signature is already revoked or outside
                revocation window.
        """
        with self._lock:
            sig = self._signatures.get(signature_id)
            if sig is None:
                raise KeyError(f"Signature not found: {signature_id}")

            if sig["is_revoked"]:
                raise ValueError("Signature is already revoked")

            # Check revocation window
            created = datetime.fromisoformat(sig["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            window = timedelta(hours=self._config.revocation_window_hours)
            if utcnow() > created + window:
                raise ValueError(
                    f"Revocation window of {self._config.revocation_window_hours}h "
                    f"has expired for this signature"
                )

            now_iso = _utcnow_iso()
            sig["is_revoked"] = True
            sig["is_valid"] = False
            sig["status"] = "revoked"
            sig["revocation_reason"] = reason
            sig["revoked_at"] = now_iso
            sig["metadata"]["revoked_by"] = revoked_by

        self._record_provenance(signature_id, "update", sig)
        logger.info(
            "Signature revoked: id=%s reason='%s' by=%s",
            signature_id[:12], reason, revoked_by,
        )
        return copy.deepcopy(sig)

    # ------------------------------------------------------------------
    # Multi-Signature Workflows
    # ------------------------------------------------------------------

    def create_multi_sig(
        self,
        chain_id: Optional[str] = None,
        form_id: str = "",
        required_roles: Optional[List[str]] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a multi-signature chain requiring signatures from multiple roles.

        Args:
            chain_id: Optional chain ID (auto-generated if None).
            form_id: Form associated with the multi-sig chain.
            required_roles: List of signer roles required.
            description: Description of the multi-sig purpose.
            metadata: Additional metadata.

        Returns:
            Multi-signature chain dictionary.

        Raises:
            ValueError: If multi-signature is disabled or required_roles empty.
        """
        if not self._config.enable_multi_signature:
            raise ValueError("Multi-signature workflows are disabled")

        if not required_roles:
            raise ValueError("required_roles must not be empty")

        if chain_id is None:
            chain_id = str(uuid.uuid4())

        now_iso = _utcnow_iso()

        chain: Dict[str, Any] = {
            "chain_id": chain_id,
            "form_id": form_id,
            "required_roles": list(required_roles),
            "collected_signatures": [],
            "status": "pending",
            "is_complete": False,
            "description": description,
            "metadata": copy.deepcopy(metadata or {}),
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        with self._lock:
            self._multi_sig_chains[chain_id] = chain

        self._record_provenance(chain_id, "create", chain)
        logger.info(
            "Multi-sig chain created: id=%s form=%s roles=%s",
            chain_id[:12], form_id[:12], required_roles,
        )
        return copy.deepcopy(chain)

    def add_signature_to_chain(
        self,
        chain_id: str,
        signature_id: str,
    ) -> Dict[str, Any]:
        """Add a signature to a multi-signature chain.

        Validates that the signer's role matches one of the required
        roles and that the role has not already been fulfilled.

        Args:
            chain_id: Multi-sig chain to add to.
            signature_id: Signature to add.

        Returns:
            Updated multi-sig chain dictionary.

        Raises:
            KeyError: If chain or signature not found.
            ValueError: If signer role not required or already fulfilled.
        """
        with self._lock:
            chain = self._multi_sig_chains.get(chain_id)
            if chain is None:
                raise KeyError(f"Multi-sig chain not found: {chain_id}")

            sig = self._signatures.get(signature_id)
            if sig is None:
                raise KeyError(f"Signature not found: {signature_id}")

            signer_role = sig["signer_role"]
            if signer_role not in chain["required_roles"]:
                raise ValueError(
                    f"Signer role '{signer_role}' not in required roles: "
                    f"{chain['required_roles']}"
                )

            fulfilled_roles = {
                s["signer_role"] for s in chain["collected_signatures"]
            }
            if signer_role in fulfilled_roles:
                raise ValueError(
                    f"Role '{signer_role}' already has a signature in this chain"
                )

            chain["collected_signatures"].append({
                "signature_id": signature_id,
                "signer_id": sig["signer_id"],
                "signer_role": signer_role,
                "added_at": _utcnow_iso(),
            })

            all_roles_fulfilled = set(chain["required_roles"]).issubset(
                {s["signer_role"] for s in chain["collected_signatures"]}
            )
            if all_roles_fulfilled:
                chain["status"] = "complete"
                chain["is_complete"] = True

            chain["updated_at"] = _utcnow_iso()

        self._record_provenance(chain_id, "update", chain)
        logger.info(
            "Signature added to chain: chain=%s sig=%s role=%s complete=%s",
            chain_id[:12], signature_id[:12], signer_role, chain["is_complete"],
        )
        return copy.deepcopy(chain)

    def verify_chain(self, chain_id: str) -> Dict[str, Any]:
        """Verify all signatures in a multi-signature chain.

        Checks that all required roles have valid, non-revoked,
        non-expired signatures.

        Args:
            chain_id: Multi-sig chain to verify.

        Returns:
            Chain verification result dict.

        Raises:
            KeyError: If chain not found.
        """
        start = time.monotonic()

        with self._lock:
            chain = self._multi_sig_chains.get(chain_id)
            if chain is None:
                raise KeyError(f"Multi-sig chain not found: {chain_id}")
            chain = copy.deepcopy(chain)

        results: List[Dict[str, Any]] = []
        all_valid = True

        for sig_entry in chain["collected_signatures"]:
            sig_id = sig_entry["signature_id"]
            try:
                verification = self.verify_signature(sig_id)
                results.append({
                    "signature_id": sig_id,
                    "signer_role": sig_entry["signer_role"],
                    "verified": verification["verified"],
                    "checks": verification["checks"],
                })
                if not verification["verified"]:
                    all_valid = False
            except KeyError:
                results.append({
                    "signature_id": sig_id,
                    "signer_role": sig_entry["signer_role"],
                    "verified": False,
                    "checks": [{"check": "exists", "passed": False,
                                "detail": "Signature not found"}],
                })
                all_valid = False

        # Check completeness
        fulfilled_roles = {r["signer_role"] for r in results if r["verified"]}
        required_set = set(chain["required_roles"])
        missing_roles = sorted(required_set - fulfilled_roles)
        is_complete = len(missing_roles) == 0

        elapsed = (time.monotonic() - start) * 1000
        result = {
            "chain_id": chain_id,
            "chain_valid": all_valid and is_complete,
            "is_complete": is_complete,
            "missing_roles": missing_roles,
            "signature_results": results,
            "total_signatures": len(results),
            "valid_signatures": sum(1 for r in results if r["verified"]),
            "timestamp": _utcnow_iso(),
            "elapsed_ms": round(elapsed, 1),
        }

        self._record_provenance(chain_id, "verify", result)
        logger.info(
            "Chain verified: id=%s valid=%s complete=%s missing=%s elapsed=%.1fms",
            chain_id[:12], result["chain_valid"], is_complete,
            missing_roles, elapsed,
        )
        return result

    def get_multi_sig_chain(self, chain_id: str) -> Dict[str, Any]:
        """Retrieve a multi-signature chain by ID.

        Args:
            chain_id: Chain identifier.

        Returns:
            Multi-sig chain dictionary.

        Raises:
            KeyError: If chain not found.
        """
        with self._lock:
            chain = self._multi_sig_chains.get(chain_id)
        if chain is None:
            raise KeyError(f"Multi-sig chain not found: {chain_id}")
        return copy.deepcopy(chain)

    # ------------------------------------------------------------------
    # Custody Transfer Signatures
    # ------------------------------------------------------------------

    def create_custody_signature(
        self,
        form_id: str,
        from_signer_id: str,
        to_signer_id: str,
        data_hash: str,
        device_id: str,
        witness_signer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a custody transfer multi-signature package.

        Generates signatures from the sender (from_party), receiver
        (to_party), and optionally a witness, all bound to the same
        form_id and data_hash.

        Args:
            form_id: Form submission for the custody transfer.
            from_signer_id: Sender signer ID.
            to_signer_id: Receiver signer ID.
            data_hash: SHA-256 hash of the transfer data.
            device_id: Device used for signing.
            witness_signer_id: Optional witness signer ID.
            metadata: Additional metadata.

        Returns:
            Custody transfer record with all signatures.

        Raises:
            KeyError: If any signer not found.
            ValueError: If signers are not authorized.
        """
        start = time.monotonic()

        # Create from-party signature
        from_sig = self.create_signature(
            form_id=form_id,
            signer_id=from_signer_id,
            data_hash=data_hash,
            device_id=device_id,
            metadata={"custody_role": "from_party"},
        )

        # Create to-party signature
        to_sig = self.create_signature(
            form_id=form_id,
            signer_id=to_signer_id,
            data_hash=data_hash,
            device_id=device_id,
            metadata={"custody_role": "to_party"},
        )

        witness_sig: Optional[Dict[str, Any]] = None
        if witness_signer_id:
            witness_sig = self.create_signature(
                form_id=form_id,
                signer_id=witness_signer_id,
                data_hash=data_hash,
                device_id=device_id,
                metadata={"custody_role": "witness"},
            )

        transfer_id = str(uuid.uuid4())
        now_iso = _utcnow_iso()

        custody_record: Dict[str, Any] = {
            "custody_transfer_id": transfer_id,
            "form_id": form_id,
            "from_party": {
                "signer_id": from_signer_id,
                "signature_id": from_sig["signature_id"],
            },
            "to_party": {
                "signer_id": to_signer_id,
                "signature_id": to_sig["signature_id"],
            },
            "witness": {
                "signer_id": witness_signer_id,
                "signature_id": witness_sig["signature_id"],
            } if witness_sig else None,
            "data_hash": data_hash,
            "device_id": device_id,
            "is_complete": True,
            "metadata": copy.deepcopy(metadata or {}),
            "created_at": now_iso,
        }

        self._record_provenance(transfer_id, "sign", custody_record)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Custody transfer signed: id=%s form=%s from=%s to=%s "
            "witness=%s elapsed=%.1fms",
            transfer_id[:12], form_id[:12], from_signer_id,
            to_signer_id, witness_signer_id or "none", elapsed,
        )
        return custody_record

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get signature engine statistics.

        Returns:
            Statistics dictionary.
        """
        with self._lock:
            sigs = list(self._signatures.values())
            signers = list(self._signers.values())
            chains = list(self._multi_sig_chains.values())

        by_status: Dict[str, int] = {}
        by_role: Dict[str, int] = {}
        for sig in sigs:
            st = sig["status"]
            by_status[st] = by_status.get(st, 0) + 1
            role = sig["signer_role"]
            by_role[role] = by_role.get(role, 0) + 1

        return {
            "total_signatures": len(sigs),
            "total_signers": len(signers),
            "total_chains": len(chains),
            "by_status": by_status,
            "by_role": by_role,
            "active_signers": sum(1 for s in signers if s["is_active"]),
            "complete_chains": sum(1 for c in chains if c["is_complete"]),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_signer_authorized(self, signer_id: str) -> None:
        """Validate a signer exists, is active, and within validity.

        Args:
            signer_id: Signer to validate.

        Raises:
            KeyError: If signer not found.
            ValueError: If signer is not active or outside validity.
        """
        with self._lock:
            signer = self._signers.get(signer_id)
        if signer is None:
            raise KeyError(f"Signer not registered: {signer_id}")
        if not signer["is_active"]:
            raise ValueError(f"Signer is not active: {signer_id}")

        now = utcnow()
        valid_from = datetime.fromisoformat(signer["valid_from"])
        valid_to = datetime.fromisoformat(signer["valid_to"])
        if valid_from.tzinfo is None:
            valid_from = valid_from.replace(tzinfo=timezone.utc)
        if valid_to.tzinfo is None:
            valid_to = valid_to.replace(tzinfo=timezone.utc)

        if now < valid_from:
            raise ValueError(
                f"Signer validity has not started yet: {signer['valid_from']}"
            )
        if now > valid_to:
            raise ValueError(
                f"Signer validity has expired: {signer['valid_to']}"
            )

    def _compute_signature_id(
        self,
        form_id: str,
        timestamp: str,
        signer_id: str,
        data_hash: str,
    ) -> str:
        """Compute deterministic signature ID from binding components.

        Args:
            form_id: Form identifier.
            timestamp: ISO 8601 timestamp.
            signer_id: Signer identifier.
            data_hash: Data hash.

        Returns:
            SHA-256 based signature ID string.
        """
        binding = f"{form_id}:{timestamp}:{signer_id}:{data_hash}"
        raw_hash = hashlib.sha256(binding.encode("utf-8")).hexdigest()
        # Format as UUID-like string from hash
        return f"sig-{raw_hash[:32]}"

    def _compute_simulated_signature(
        self,
        form_id: str,
        timestamp: str,
        signer_id: str,
        data_hash: str,
    ) -> str:
        """Compute simulated ECDSA P-256 signature (SHA-256 hash).

        This is NOT a real cryptographic signature. It simulates the
        signature output for data modeling purposes. Production would
        use real ECDSA P-256 via device keystore.

        Args:
            form_id: Form identifier.
            timestamp: ISO 8601 timestamp.
            signer_id: Signer identifier.
            data_hash: SHA-256 hash of signed data.

        Returns:
            Hex-encoded simulated signature string.
        """
        payload = json.dumps(
            {
                "algorithm": self._config.signature_algorithm,
                "data_hash": data_hash,
                "form_id": form_id,
                "signer_id": signer_id,
                "timestamp": timestamp,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _is_expired(self, sig: Dict[str, Any]) -> bool:
        """Check if a signature has expired.

        Args:
            sig: Signature dictionary.

        Returns:
            True if expired.
        """
        expires_at = sig.get("expires_at")
        if expires_at is None:
            return False
        try:
            exp_time = datetime.fromisoformat(expires_at)
            if exp_time.tzinfo is None:
                exp_time = exp_time.replace(tzinfo=timezone.utc)
            return utcnow() > exp_time
        except (ValueError, TypeError):
            return False

    def _is_signer_valid(self, signer_id: str) -> bool:
        """Check if a signer is registered and active.

        Args:
            signer_id: Signer identifier.

        Returns:
            True if valid.
        """
        with self._lock:
            signer = self._signers.get(signer_id)
        if signer is None:
            return False
        return signer.get("is_active", False)

    def _record_provenance(
        self,
        entity_id: str,
        action: str,
        data: Any,
    ) -> None:
        """Record a provenance entry for signature operations.

        Args:
            entity_id: Entity identifier.
            action: Provenance action.
            data: Data payload to hash.
        """
        try:
            self._provenance.record(
                entity_type="digital_signature",
                action=action,
                entity_id=entity_id,
                data=data,
                metadata={"engine": "DigitalSignatureEngine"},
            )
        except Exception as exc:
            logger.warning(
                "Provenance recording failed for signature %s: %s",
                entity_id[:12], exc,
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        with self._lock:
            sig_count = len(self._signatures)
            signer_count = len(self._signers)
        return (
            f"DigitalSignatureEngine(signatures={sig_count}, "
            f"signers={signer_count}, "
            f"algorithm={self._config.signature_algorithm})"
        )

    def __len__(self) -> int:
        """Return total number of signatures."""
        with self._lock:
            return len(self._signatures)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DigitalSignatureEngine",
    "SIGNER_ROLES",
    "SIGNATURE_STATUSES",
    "REVOCATION_REASONS",
    "WITNESS_ROLES",
]
