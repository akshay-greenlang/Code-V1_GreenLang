# -*- coding: utf-8 -*-
"""
G5–G6: Human / legal approval before certified promotion, and public export guards.

Certified rows must resolve to a registry entry that allows certification for that source;
connector-only upstreams never produce certified catalog rows. Bulk export requires both
row-level license_info and registry-level redistribution rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from greenlang.data.emission_factor_record import EmissionFactorRecord
from greenlang.factors.source_registry import SourceRegistryEntry, registry_by_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    blockers: Tuple[str, ...]


def check_promote_to_certified(
    source_id: Optional[str],
    *,
    registry: Optional[Dict[str, SourceRegistryEntry]] = None,
    require_legal_signoff: bool = True,
) -> GateResult:
    """
    Return whether factors from this source may use factor_status ``certified``.

    Blocks: unknown source, connector-only registry entries, missing legal sign-off when required.

    ``require_legal_signoff``: when True (production default), ``legal_signoff_artifact`` must be
    set on the registry entry if ``approval_required_for_certified``. When False, legal path is
    skipped (CI / preview pipelines using the stock YAML before legal uploads artifacts).
    """
    reg = registry if registry is not None else registry_by_id()
    sid = (source_id or "").strip()
    if not sid:
        return GateResult(False, ("missing_source_id",))
    entry = reg.get(sid)
    if entry is None:
        return GateResult(False, (f"unknown_source_id:{sid}",))
    if entry.connector_only:
        logger.info("Promotion blocked for source=%s: connector_only", sid)
        return GateResult(
            False,
            ("connector_only_source_cannot_promote_to_certified",),
        )
    if require_legal_signoff and entry.approval_required_for_certified:
        art = entry.legal_signoff_artifact
        if art is None or not str(art).strip():
            return GateResult(False, ("pending_legal_signoff_artifact",))
    return GateResult(True, ())


def public_bulk_export_allowed_for_factor(
    record: EmissionFactorRecord,
    *,
    registry: Optional[Dict[str, SourceRegistryEntry]] = None,
) -> GateResult:
    """
    G6 + license_info: public bulk export only when row and registry both allow redistribution.

    connector_only / deprecated rows are never bulk-exportable as open data.
    """
    reg = registry if registry is not None else registry_by_id()
    st = (record.factor_status or "certified").lower()
    if st in ("connector_only", "deprecated"):
        return GateResult(False, (f"factor_status_blocks_export:{st}",))
    if not record.license_info.redistribution_allowed:
        return GateResult(False, ("license_info.redistribution_allowed_false",))
    sid = (record.source_id or "").strip()
    if not sid:
        return GateResult(False, ("missing_source_id_for_registry_export_check",))
    entry = reg.get(sid)
    if entry is None:
        return GateResult(False, (f"unknown_source_id:{sid}",))
    if not entry.public_bulk_export_allowed():
        return GateResult(
            False,
            ("registry_blocks_public_bulk_export",),
        )
    return GateResult(True, ())


def require_public_bulk_export(record: EmissionFactorRecord, *, registry: Optional[Dict[str, SourceRegistryEntry]] = None) -> None:
    """Raise ValueError if public bulk export is not allowed."""
    r = public_bulk_export_allowed_for_factor(record, registry=registry)
    if not r.allowed:
        raise ValueError("; ".join(r.blockers))
