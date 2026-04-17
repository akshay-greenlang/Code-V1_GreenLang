# -*- coding: utf-8 -*-
"""
Immutable edition manifests for GreenLang Factors (hashes, changelog, deprecations).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class EditionManifest:
    """Serializable manifest for a published factor edition."""

    edition_id: str
    status: str  # stable | pending | retired
    created_at: str = field(default_factory=_utc_now_iso)
    factor_count: int = 0
    aggregate_content_hash: str = ""
    per_source_hashes: Dict[str, str] = field(default_factory=dict)
    deprecations: List[str] = field(default_factory=list)
    changelog: List[str] = field(default_factory=list)
    policy_rule_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditionManifest":
        fields = cls.__dataclass_fields__.keys()
        kwargs = {k: data[k] for k in fields if k in data}
        return cls(**kwargs)

    def manifest_fingerprint(self) -> str:
        """Deterministic hash of the manifest (excluding created_at for CI compares optional)."""
        payload = {
            "edition_id": self.edition_id,
            "status": self.status,
            "factor_count": self.factor_count,
            "aggregate_content_hash": self.aggregate_content_hash,
            "per_source_hashes": self.per_source_hashes,
            "deprecations": self.deprecations,
            "changelog": self.changelog,
            "policy_rule_refs": self.policy_rule_refs,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


def build_aggregate_hash(content_hashes: List[str]) -> str:
    """Combine individual factor content_hash values deterministically."""
    joined = "\n".join(sorted(content_hashes))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_manifest_for_factors(
    edition_id: str,
    status: str,
    factors: List[Any],
    changelog: Optional[List[str]] = None,
    deprecations: Optional[List[str]] = None,
    policy_rule_refs: Optional[List[str]] = None,
) -> EditionManifest:
    """
    Build manifest from EmissionFactorRecord-like objects with .content_hash.

    Args:
        edition_id: Edition label (e.g. 2026.04.2-builtin).
        status: stable | pending | retired
        factors: Iterable of records with content_hash str attribute.
    """
    hashes = [getattr(f, "content_hash", "") for f in factors if getattr(f, "content_hash", "")]
    agg = build_aggregate_hash(hashes)
    cl = changelog or [f"Edition {edition_id} generated from {len(factors)} factors."]
    logger.info("Building manifest edition=%s status=%s factors=%d", edition_id, status, len(factors))
    return EditionManifest(
        edition_id=edition_id,
        status=status,
        factor_count=len(factors),
        aggregate_content_hash=agg,
        per_source_hashes={},
        deprecations=list(deprecations or []),
        changelog=list(cl),
        policy_rule_refs=list(policy_rule_refs or []),
    )
