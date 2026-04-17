# -*- coding: utf-8 -*-
"""Load and validate the CTO source registry (G1–G6)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass(frozen=True)
class SourceRegistryEntry:
    source_id: str
    display_name: str
    connector_only: bool
    license_class: str
    redistribution_allowed: bool
    derivative_works_allowed: bool
    commercial_use_allowed: bool
    attribution_required: bool
    citation_text: str
    cadence: str
    watch_mechanism: str
    watch_url: Optional[str]
    watch_file_type: Optional[str]
    approval_required_for_certified: bool
    legal_signoff_artifact: Optional[str]
    legal_signoff_version: Optional[str]

    def public_bulk_export_allowed(self) -> bool:
        return self.redistribution_allowed and not self.connector_only


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "source_registry.yaml"


def load_source_registry(path: Optional[Path] = None) -> List[SourceRegistryEntry]:
    p = path or _default_registry_path()
    if yaml is None:
        raise RuntimeError("PyYAML is required to load source_registry.yaml")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    sources = raw.get("sources") or []
    out: List[SourceRegistryEntry] = []
    for item in sources:
        if not isinstance(item, dict):
            continue
        watch = item.get("watch") or {}
        connector_only = bool(item.get("connector_only"))
        redistribution = bool(item.get("redistribution_allowed"))
        deriv = item.get("derivative_works_allowed")
        if deriv is None:
            deriv = redistribution and not connector_only
        comm = item.get("commercial_use_allowed")
        if comm is None:
            comm = redistribution
        out.append(
            SourceRegistryEntry(
                source_id=str(item["source_id"]),
                display_name=str(item.get("display_name") or item["source_id"]),
                connector_only=connector_only,
                license_class=str(item.get("license_class") or "unknown"),
                redistribution_allowed=redistribution,
                derivative_works_allowed=bool(deriv),
                commercial_use_allowed=bool(comm),
                attribution_required=bool(item.get("attribution_required", True)),
                citation_text=str(item.get("citation_text") or ""),
                cadence=str(item.get("cadence") or "unknown"),
                watch_mechanism=str(watch.get("mechanism") or "none"),
                watch_url=watch.get("url"),
                watch_file_type=watch.get("file_type"),
                approval_required_for_certified=bool(item.get("approval_required_for_certified", True)),
                legal_signoff_artifact=item.get("legal_signoff_artifact"),
                legal_signoff_version=item.get("legal_signoff_version"),
            )
        )
    logger.debug("Loaded %d source registry entries from %s", len(out), p)
    return out


def registry_by_id(path: Optional[Path] = None) -> Dict[str, SourceRegistryEntry]:
    return {e.source_id: e for e in load_source_registry(path)}


def validate_registry(entries: Optional[List[SourceRegistryEntry]] = None) -> List[str]:
    """Return human-readable issues (empty if OK)."""
    entries = entries or load_source_registry()
    issues: List[str] = []
    seen: set[str] = set()
    for e in entries:
        if e.source_id in seen:
            issues.append(f"duplicate source_id: {e.source_id}")
        seen.add(e.source_id)
        if e.connector_only and e.redistribution_allowed:
            issues.append(f"{e.source_id}: connector_only should not set redistribution_allowed true")
        if e.connector_only and e.derivative_works_allowed:
            issues.append(
                f"{e.source_id}: connector_only sources should not allow derivative_works in public registry"
            )
        if e.approval_required_for_certified and not e.citation_text.strip():
            issues.append(f"{e.source_id}: missing citation_text")
    return issues
