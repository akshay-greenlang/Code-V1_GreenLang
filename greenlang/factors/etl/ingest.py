# -*- coding: utf-8 -*-
"""Load factors into SQLite catalog."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

from greenlang.data.emission_factor_database import EmissionFactorDatabase
from greenlang.data.emission_factor_record import EmissionFactorRecord

from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository
from greenlang.factors.edition_manifest import build_manifest_for_factors
from greenlang.factors.etl.normalize import (
    dict_to_emission_factor_record,
    iter_cbam_factor_dicts,
    iter_defra_scope1_dicts,
)
from greenlang.factors.etl.qa import validate_factor_dict


def ingest_builtin_database(
    sqlite_path: Path,
    edition_id: str,
    *,
    label: str = "Built-in v2 EmissionFactorDatabase",
    status: str = "stable",
) -> int:
    """Load EmissionFactorDatabase defaults into catalog."""
    db = EmissionFactorDatabase(enable_cache=False)
    records = list(db.factors.values())
    logger.info("Ingesting %d built-in factors into %s edition=%s", len(records), sqlite_path, edition_id)
    return _write_edition(sqlite_path, edition_id, label, status, records)


def _write_edition(
    sqlite_path: Path,
    edition_id: str,
    label: str,
    status: str,
    records: Sequence[EmissionFactorRecord],
) -> int:
    repo = SqliteFactorCatalogRepository(sqlite_path)
    manifest = build_manifest_for_factors(
        edition_id,
        status,
        list(records),
        changelog=[f"Ingested {len(records)} factors into {sqlite_path.name}"],
    )
    repo.upsert_edition(
        edition_id,
        status,
        label,
        manifest.to_dict(),
        manifest.changelog,
    )
    repo.insert_factors(edition_id, records)
    return len(records)


def ingest_from_paths(
    sqlite_path: Path,
    edition_id: str,
    paths: Sequence[Path],
    *,
    label: str = "ETL bundle",
    status: str = "stable",
) -> int:
    """
    Normalize JSON files (CBAM defaults, DEFRA scope1 shape) into records and ingest.
    """
    records: List[EmissionFactorRecord] = []
    errors: List[str] = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{p}: {exc}")
            continue
        gen: Iterable[dict]
        if "cbam_defaults" in p.name.lower() or (
            isinstance(data.get("factors"), dict) and data.get("metadata", {}).get("source", "").startswith("EU")
        ):
            gen = iter_cbam_factor_dicts(data)
        elif "scope1" in p.name.lower() or (
            isinstance(data.get("metadata"), dict) and str(data["metadata"].get("source", "")).upper() == "DEFRA"
        ):
            gen = iter_defra_scope1_dicts(data)
        else:
            errors.append(f"{p}: unsupported JSON shape")
            continue
        for d in gen:
            ok, msgs = validate_factor_dict(d)
            if not ok:
                errors.append(f"{p}: QA failed for {d.get('factor_id')}: {msgs}")
                continue
            try:
                records.append(dict_to_emission_factor_record(d))
            except Exception as exc:  # pragma: no cover
                errors.append(f"{p}: {d.get('factor_id')}: {exc}")
    if errors and not records:
        raise ValueError("; ".join(errors[:20]))
    return _write_edition(sqlite_path, edition_id, label, status, records)
