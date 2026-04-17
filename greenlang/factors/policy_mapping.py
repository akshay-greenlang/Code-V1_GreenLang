# -*- coding: utf-8 -*-
"""
Policy Graph hooks: map regulatory rule identifiers to affected factor IDs.

Supports optional YAML/JSON file (e.g. policy_factor_map.yaml) and a small
pending-edition state file for SME review workflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PolicyFactorAssociation:
    policy_rule_id: str
    factor_ids: List[str]
    notes: str = ""


def load_policy_factor_map(path: Path) -> List[PolicyFactorAssociation]:
    """Load mapping file. Supports .yaml / .yml / .json."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    items = raw.get("associations") or raw.get("mappings") or []
    out: List[PolicyFactorAssociation] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        pid = row.get("policy_rule_id") or row.get("rule_id")
        fids = row.get("factor_ids") or row.get("factors") or []
        if pid and isinstance(fids, list):
            out.append(
                PolicyFactorAssociation(
                    policy_rule_id=str(pid),
                    factor_ids=[str(x) for x in fids],
                    notes=str(row.get("notes", "")),
                )
            )
    return out


def factors_for_rule(associations: List[PolicyFactorAssociation], rule_id: str) -> List[str]:
    for a in associations:
        if a.policy_rule_id == rule_id:
            return list(a.factor_ids)
    return []


def pending_edition_path(catalog_db_path: Path) -> Path:
    return catalog_db_path.parent / f"{catalog_db_path.name}.pending_edition.json"


def read_pending_edition(catalog_db_path: Path) -> Optional[Dict[str, Any]]:
    p = pending_edition_path(catalog_db_path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_pending_edition(
    catalog_db_path: Path,
    edition_id: str,
    reason: str,
    factor_ids: List[str],
    policy_rule_ids: List[str],
) -> Path:
    """SME queue: proposed edition metadata before promote to stable."""
    p = pending_edition_path(catalog_db_path)
    payload = {
        "proposed_edition_id": edition_id,
        "reason": reason,
        "factor_ids": factor_ids,
        "policy_rule_ids": policy_rule_ids,
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def clear_pending_edition(catalog_db_path: Path) -> None:
    p = pending_edition_path(catalog_db_path)
    if p.is_file():
        p.unlink()


def route_policy_change(rule_id: str, change_kind: str) -> Dict[str, Any]:
    """U4: route policy-affecting changes into methodology / QA queues."""
    return {
        "policy_rule_id": rule_id,
        "change_kind": change_kind,
        "queues": ["methodology_review", "qa_promotion_hold"],
    }
