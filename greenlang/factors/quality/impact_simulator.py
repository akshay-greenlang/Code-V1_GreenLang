# -*- coding: utf-8 -*-
"""
Impact simulator (Phase F6).

Answers the CTO question: *"What breaks if we replace the UK-2025 road
freight factor pack with v2?"*  Traces every computation that ever
consumed a factor from the affected pack and reports:

- affected customer tenants
- affected computation IDs (Scope Engine + Comply runs)
- affected evidence bundles
- the numeric delta (old vs new value) per computation

Works against the SQLite sinks written by:
- Climate Ledger (``climate_ledger_entries``) — Phase 2.1
- Evidence Vault (``evidence_records``) — Phase 2.2
- Factor version chain (``factor_version_chain``) — Phase F6
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

logger = logging.getLogger(__name__)


@dataclass
class ImpactedComputation:
    """A single run that consumed an affected factor."""

    computation_id: str
    computation_hash: str
    tenant_id: Optional[str]
    entity_id: Optional[str]
    evidence_bundle: Optional[str]
    factor_id: str
    old_factor_version: Optional[str]
    new_factor_version: Optional[str]
    old_value: Optional[float]
    new_value: Optional[float]
    delta_abs: Optional[float] = None
    delta_pct: Optional[float] = None

    def __post_init__(self) -> None:
        if self.old_value is not None and self.new_value is not None:
            self.delta_abs = self.new_value - self.old_value
            if self.old_value != 0:
                self.delta_pct = (self.delta_abs / abs(self.old_value)) * 100.0


@dataclass
class ImpactReport:
    """Full impact simulation output."""

    simulated_at: str
    affected_factor_ids: List[str]
    computations: List[ImpactedComputation]
    tenants: List[str]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulated_at": self.simulated_at,
            "affected_factor_ids": self.affected_factor_ids,
            "tenants": self.tenants,
            "computation_count": len(self.computations),
            "summary": self.summary,
            "computations": [
                {
                    "computation_id": c.computation_id,
                    "computation_hash": c.computation_hash,
                    "tenant_id": c.tenant_id,
                    "entity_id": c.entity_id,
                    "evidence_bundle": c.evidence_bundle,
                    "factor_id": c.factor_id,
                    "old_factor_version": c.old_factor_version,
                    "new_factor_version": c.new_factor_version,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "delta_abs": c.delta_abs,
                    "delta_pct": c.delta_pct,
                }
                for c in self.computations
            ],
        }


class ImpactSimulator:
    """Cross-table impact query.

    Accepts in-memory inputs for testability: ``ledger_entries`` is a
    list of ledger row dicts (typically pulled from
    ``climate_ledger_entries``).  Production callers wrap a repository
    query that returns the same shape.
    """

    def __init__(
        self,
        *,
        ledger_entries: Optional[Iterable[Dict[str, Any]]] = None,
        evidence_records: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        self._ledger = list(ledger_entries or [])
        self._evidence = list(evidence_records or [])

    def simulate_replacement(
        self,
        *,
        replaced_factor_ids: Iterable[str],
        value_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> ImpactReport:
        """Run the impact simulation.

        ``value_map`` maps ``factor_id → {"old": float, "new": float}``.
        When provided, per-computation deltas are populated; otherwise
        the report only counts affected runs without a numeric delta.
        """
        factor_set: Set[str] = set(replaced_factor_ids)
        computations: List[ImpactedComputation] = []
        tenants: Set[str] = set()

        # Join ledger + evidence by case/entity.
        evidence_by_content: Dict[str, Dict[str, Any]] = {
            r.get("content_hash", ""): r for r in self._evidence if r.get("content_hash")
        }

        for entry in self._ledger:
            metadata = entry.get("metadata") or {}
            entry_factor = metadata.get("factor_id") or entry.get("factor_id")
            if not entry_factor or entry_factor not in factor_set:
                continue

            ev_record = evidence_by_content.get(entry.get("content_hash", ""))
            bundle = None
            tenant = entry.get("tenant_id")
            if ev_record is not None:
                tenant = tenant or ev_record.get("tenant_id")
                bundle = ev_record.get("evidence_id")

            old_val, new_val = None, None
            if value_map and entry_factor in value_map:
                old_val = value_map[entry_factor].get("old")
                new_val = value_map[entry_factor].get("new")

            computations.append(
                ImpactedComputation(
                    computation_id=str(entry.get("entity_id") or entry.get("id") or ""),
                    computation_hash=str(entry.get("chain_hash") or entry.get("content_hash") or ""),
                    tenant_id=tenant,
                    entity_id=str(entry.get("entity_id") or ""),
                    evidence_bundle=bundle,
                    factor_id=entry_factor,
                    old_factor_version=metadata.get("old_factor_version"),
                    new_factor_version=metadata.get("new_factor_version"),
                    old_value=old_val,
                    new_value=new_val,
                )
            )
            if tenant:
                tenants.add(tenant)

        # Build summary.
        pct_deltas = [c.delta_pct for c in computations if c.delta_pct is not None]
        summary: Dict[str, Any] = {
            "affected_computations": len(computations),
            "affected_tenants": len(tenants),
            "affected_factor_count": len(factor_set),
        }
        if pct_deltas:
            summary["max_pct_delta"] = max(pct_deltas, key=abs)
            summary["mean_pct_delta"] = sum(pct_deltas) / len(pct_deltas)

        return ImpactReport(
            simulated_at=datetime.now(timezone.utc).isoformat(),
            affected_factor_ids=sorted(factor_set),
            computations=computations,
            tenants=sorted(tenants),
            summary=summary,
        )


def load_ledger_entries_from_sqlite(sqlite_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Helper: load ``climate_ledger_entries`` rows from the Phase 2.1 SQLite."""
    p = Path(sqlite_path)
    if not p.exists():
        return []
    conn = sqlite3.connect(str(p))
    try:
        rows = conn.execute(
            """
            SELECT id, agent_name, entity_type, entity_id, operation,
                   content_hash, chain_hash, metadata_json, recorded_at
            FROM climate_ledger_entries
            ORDER BY id ASC
            """
        ).fetchall()
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            meta = json.loads(r[7]) if r[7] else {}
        except json.JSONDecodeError:
            meta = {}
        out.append(
            {
                "id": r[0],
                "agent_name": r[1],
                "entity_type": r[2],
                "entity_id": r[3],
                "operation": r[4],
                "content_hash": r[5],
                "chain_hash": r[6],
                "metadata": meta,
                "recorded_at": r[8],
            }
        )
    return out


__all__ = [
    "ImpactedComputation",
    "ImpactReport",
    "ImpactSimulator",
    "load_ledger_entries_from_sqlite",
]
