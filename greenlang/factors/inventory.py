# -*- coding: utf-8 -*-
"""
Inventory of emission-factor sources across the monorepo (FY27 Factors Phase 0).
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_cbam_like_factors(data: dict) -> int:
    factors = data.get("factors") or {}
    n = 0
    if isinstance(factors, dict):
        for _ptype, pdata in factors.items():
            if not isinstance(pdata, dict):
                continue
            by_country = pdata.get("by_country") or {}
            if isinstance(by_country, dict):
                n += len(by_country)
    return n


def _count_json_array_entries(path: Path, key: str) -> Optional[int]:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    arr = data.get(key)
    if isinstance(arr, list):
        return len(arr)
    return None


def collect_inventory(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Scan known factor locations and return a machine-readable coverage matrix.

    Returns:
        dict with keys: generated_at, sources (list of source descriptors),
        totals, notes.
    """
    root = repo_root or REPO_ROOT
    sources: List[Dict[str, Any]] = []

    # v2 built-in database (runtime count)
    try:
        from greenlang.data.emission_factor_database import EmissionFactorDatabase

        db = EmissionFactorDatabase(enable_cache=False)
        sources.append(
            {
                "id": "emission_factor_database_v2",
                "path": "greenlang/data/emission_factor_database.py",
                "kind": "python_builtin",
                "row_estimate": len(db.factors),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        sources.append(
            {
                "id": "emission_factor_database_v2",
                "error": str(exc),
                "row_estimate": None,
            }
        )

    ef_dir = root / "applications" / "GL-Agent-Factory" / "backend" / "data" / "emission_factors"
    if ef_dir.is_dir():
        for p in sorted(ef_dir.rglob("*.json")):
            rel = str(p.relative_to(root)).replace("\\", "/")
            sha = _sha256_file(p)
            extra: Dict[str, Any] = {"sha256": sha}
            try:
                with p.open(encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "factors" in data and isinstance(data["factors"], dict):
                        extra["row_estimate"] = len(data["factors"])
                    elif "metadata" in data:
                        # DEFRA-style
                        n = 0
                        for k, v in data.items():
                            if k == "metadata" or not isinstance(v, list):
                                continue
                            for item in v:
                                if isinstance(item, dict) and item.get("units"):
                                    n += len(item["units"])
                        extra["row_estimate"] = n or None
            except (OSError, json.JSONDecodeError):
                extra["row_estimate"] = None
            sources.append(
                {
                    "id": f"agent_factory_json:{rel}",
                    "path": rel,
                    "kind": "json_bundle",
                    **extra,
                }
            )

    cbam_json = root / "cbam-pack-mvp" / "data" / "emission_factors" / "cbam_defaults_2024.json"
    if cbam_json.is_file():
        sha = _sha256_file(cbam_json)
        try:
            with cbam_json.open(encoding="utf-8") as f:
                data = json.load(f)
            n_cbam = _count_cbam_like_factors(data)
        except (OSError, json.JSONDecodeError):
            n_cbam = None
        sources.append(
            {
                "id": "cbam_defaults_2024",
                "path": str(cbam_json.relative_to(root)).replace("\\", "/"),
                "kind": "cbam_json",
                "sha256": sha,
                "row_estimate": n_cbam,
            }
        )

    gov_path = root / "greenlang" / "governance" / "validation" / "emission_factors.py"
    if gov_path.is_file():
        text = gov_path.read_text(encoding="utf-8", errors="replace")
        n_add = text.count("_add_factor(")
        sources.append(
            {
                "id": "governance_emission_factors_py",
                "path": str(gov_path.relative_to(root)).replace("\\", "/"),
                "kind": "python_embedded",
                "row_estimate": n_add,
            }
        )

    total_rows = sum(
        (s.get("row_estimate") or 0) for s in sources if isinstance(s.get("row_estimate"), int)
    )

    try:
        from greenlang.factors.source_registry import load_source_registry, validate_registry

        reg = load_source_registry()
        source_registry = {
            "entries": len(reg),
            "validation_issues": validate_registry(reg),
            "source_ids": [e.source_id for e in reg],
        }
    except Exception as exc:  # pragma: no cover - yaml optional in minimal env
        source_registry = {"error": str(exc)}

    return {
        "repo_root": str(root),
        "totals": {
            "sources_tracked": len(sources),
            "sum_row_estimates": total_rows,
            "note": "Summed estimates double-count overlapping semantics; use dedupe_rules for merges.",
        },
        "sources": sources,
        "dedupe": "greenlang.factors.dedupe_rules.MERGE_RULES",
        "source_registry": source_registry,
    }


def write_coverage_matrix(out_path: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Write collect_inventory() result to JSON."""
    data = collect_inventory(repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
