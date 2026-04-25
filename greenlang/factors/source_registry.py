# -*- coding: utf-8 -*-
"""Load and validate the CTO source registry (G1–G6)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v0.1 Alpha required-fields contract (WS7-T1)
# ---------------------------------------------------------------------------
# The 6 alpha-launch sources MUST populate every field in this list. The
# canonical alpha source ids (with their alpha-spec aliases in parens):
#   1. ipcc_2006_nggi          (alpha alias: ipcc_ar6)
#   2. desnz_ghg_conversion
#   3. epa_hub
#   4. egrid
#   5. india_cea_co2_baseline  (alpha alias: india_cea_baseline)
#   6. cbam_default_values
# ---------------------------------------------------------------------------
ALPHA_V0_1_REQUIRED_FIELDS: Tuple[str, ...] = (
    "source_id",
    "urn",
    "source_owner",
    "parser_module",
    "parser_function",
    "parser_version",
    "cadence",
    "license_class",
    "source_version",
    "latest_ingestion_at",
    "legal_signoff_artifact",
    "publication_url",
    "provenance_completeness_score",
    "alpha_v0_1",
)

# Per the v0.1 alpha contract, two fields may be null (key must still
# be PRESENT, but null is an acceptable sentinel pre-first-ingest /
# pre-counsel-signoff). All other fields must be non-null + non-empty.
ALPHA_V0_1_NULLABLE_FIELDS: Tuple[str, ...] = (
    "latest_ingestion_at",
    "legal_signoff_artifact",
)

ALPHA_V0_1_EXPECTED_SOURCE_IDS: Tuple[str, ...] = (
    "ipcc_2006_nggi",
    "desnz_ghg_conversion",
    "epa_hub",
    "egrid",
    "india_cea_co2_baseline",
    "cbam_default_values",
)

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
    # ---- CTO Phase F1 extensions (all optional: backward compatible with
    #      existing source_registry.yaml entries) ----
    publisher: Optional[str] = None                 # e.g. "EPA", "DESNZ"
    jurisdiction: Optional[str] = None              # e.g. "US", "EU", "UK"
    dataset_version: Optional[str] = None           # e.g. "2024-Q4"
    publication_date: Optional[str] = None          # ISO-8601
    validity_period: Optional[str] = None           # e.g. "2024-01-01/2024-12-31"
    ingestion_date: Optional[str] = None            # when we last pulled it
    source_type: Optional[str] = None               # canonical_v2.SourceType enum
    verification_status: Optional[str] = None       # canonical_v2.VerificationStatus
    change_log_uri: Optional[str] = None            # link to source-side changelog
    legal_notes: Optional[str] = None               # free-form counsel notes

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
                # ---- Phase F1 optional extensions ----
                publisher=item.get("publisher"),
                jurisdiction=item.get("jurisdiction"),
                dataset_version=item.get("dataset_version"),
                publication_date=item.get("publication_date"),
                validity_period=item.get("validity_period"),
                ingestion_date=item.get("ingestion_date"),
                source_type=item.get("source_type"),
                verification_status=item.get("verification_status"),
                change_log_uri=item.get("change_log_uri"),
                legal_notes=item.get("legal_notes"),
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


def _load_raw_sources(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return the raw YAML source dicts (preserves alpha-only fields).

    The dataclass loader projects the YAML rows down to the canonical
    schema and drops alpha-only fields (urn, parser_module, ...). The
    alpha completeness validator needs to see those fields, so it works
    against the raw YAML payload.
    """
    p = path or _default_registry_path()
    if yaml is None:
        raise RuntimeError("PyYAML is required to load source_registry.yaml")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    sources = raw.get("sources") or []
    return [item for item in sources if isinstance(item, dict)]


def validate_alpha_v0_1_completeness(
    path: Optional[Path] = None,
) -> List[Tuple[str, List[str]]]:
    """Validate v0.1 alpha completeness for the 6 alpha-launch sources.

    Returns a list of ``(source_id, missing_fields)`` tuples for every
    source flagged ``alpha_v0_1: true`` in the registry. A source is
    considered complete when ``missing_fields`` is empty. The expected
    canonical ids are listed in :data:`ALPHA_V0_1_EXPECTED_SOURCE_IDS`;
    aliases declared via the ``aliases`` key are accepted in lieu of
    the canonical id when callers query by an alpha-spec alias.

    A field is considered "missing" when:
      * the key is absent from the YAML row, OR
      * the value is ``None`` / an empty string / an empty list.

    The boolean flag ``alpha_v0_1`` is checked for *presence* (it is
    implicitly true on every row this returns).

    Args:
        path: optional override of the source_registry.yaml path.

    Returns:
        ``[(source_id, missing_fields), ...]`` — one entry per source
        flagged as alpha_v0_1, missing_fields is empty on full pass.
    """
    raw_sources = _load_raw_sources(path)
    out: List[Tuple[str, List[str]]] = []
    for item in raw_sources:
        if not bool(item.get("alpha_v0_1")):
            continue
        source_id = str(item.get("source_id") or "")
        missing: List[str] = []
        for fld in ALPHA_V0_1_REQUIRED_FIELDS:
            present = fld in item
            val = item.get(fld)
            # Nullable fields: KEY must be present, but null is OK.
            if fld in ALPHA_V0_1_NULLABLE_FIELDS:
                if not present:
                    missing.append(fld)
                continue
            if val is None:
                missing.append(fld)
                continue
            if isinstance(val, str) and not val.strip():
                missing.append(fld)
                continue
            if isinstance(val, (list, tuple, dict)) and not val:
                missing.append(fld)
                continue
        out.append((source_id, missing))
    return out


def alpha_v0_1_sources(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Return the raw YAML rows for every alpha_v0_1 source, keyed by source_id."""
    return {
        str(item["source_id"]): item
        for item in _load_raw_sources(path)
        if bool(item.get("alpha_v0_1"))
    }
