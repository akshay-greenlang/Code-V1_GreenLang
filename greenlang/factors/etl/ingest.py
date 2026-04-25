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
    run_alpha_provenance_gate,
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


def _detect_source_label(path: Path, data: Any) -> str:
    """Best-effort source label for ingestion metrics (low cardinality)."""
    name = path.name.lower()
    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    src = str(meta.get("source", "")) if isinstance(meta, dict) else ""
    if "cbam" in name or src.upper().startswith("EU"):
        return "eu-cbam-defaults"
    if "defra" in name or src.upper() == "DEFRA":
        return "defra-2025"
    if "ipcc" in name or "ipcc" in src.lower():
        return "ipcc-ar6"
    if "epa-ghg" in name or "ghg" in src.lower():
        return "epa-ghg-hub"
    if "egrid" in name or "egrid" in src.lower():
        return "epa-egrid"
    if "cea" in name or "india-cea" in name:
        return "india-cea-baseline"
    return "unknown"


def _record_ingestion_run_safe(*, status: str, source: str) -> None:
    """Emit factors_ingestion_runs_total without breaking ingestion on failure."""
    try:
        from greenlang.factors.observability.prometheus_exporter import (
            get_factors_metrics,
        )

        get_factors_metrics().record_ingestion_run(status=status, source=source)
    except Exception as exc:  # noqa: BLE001
        logger.debug("ingestion run metric emit failed (%s); ignoring.", exc)


def _record_parser_error_safe(*, source: str, error_type: str) -> None:
    """Emit factors_parser_errors_total without breaking ingestion on failure."""
    try:
        from greenlang.factors.observability.prometheus_exporter import (
            get_factors_metrics,
        )

        get_factors_metrics().record_parser_error(source=source, error_type=error_type)
    except Exception as exc:  # noqa: BLE001
        logger.debug("parser error metric emit failed (%s); ignoring.", exc)


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
    normalised_dicts: List[dict] = []
    # Emit a single "started" event per ingest_from_paths() call. The "source"
    # label here is best-effort: when paths span multiple sources, we mark
    # ``mixed``; otherwise the per-path source label.
    sources_seen: set = set()
    _record_ingestion_run_safe(status="started", source="mixed")
    run_status = "success"  # flipped to "failed" if any path errors out
    for p in paths:
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{p}: {exc}")
            _record_parser_error_safe(
                source=_detect_source_label(p, {}), error_type=type(exc).__name__
            )
            run_status = "failed"
            continue
        source_label = _detect_source_label(p, data)
        sources_seen.add(source_label)
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
            _record_parser_error_safe(
                source=source_label, error_type="UnsupportedJsonShape"
            )
            run_status = "failed"
            continue
        for d in gen:
            ok, msgs = validate_factor_dict(d)
            if not ok:
                errors.append(f"{p}: QA failed for {d.get('factor_id')}: {msgs}")
                continue
            normalised_dicts.append(d)
            try:
                records.append(dict_to_emission_factor_record(d))
            except Exception as exc:  # pragma: no cover
                errors.append(f"{p}: {d.get('factor_id')}: {exc}")
                _record_parser_error_safe(
                    source=source_label, error_type=type(exc).__name__
                )

    # Wave B / WS2-T1 alpha provenance gate. Writes validation_report.json
    # next to the SQLite catalog. The gate is a no-op (enabled=False) outside
    # the alpha-v0.1 release profile unless GL_FACTORS_ALPHA_PROVENANCE_GATE
    # is set; legacy v1-shape dicts validate as failures only when the gate
    # is enabled, so existing pipelines stay green by default.
    try:
        run_alpha_provenance_gate(
            normalised_dicts, output_dir=Path(sqlite_path).parent
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("alpha_provenance_gate: report generation failed: %s", exc)

    if errors and not records:
        # Emit "failed" run for every source we saw, then raise.
        for s in sources_seen or {"unknown"}:
            _record_ingestion_run_safe(status="failed", source=s)
        raise ValueError("; ".join(errors[:20]))
    written = _write_edition(sqlite_path, edition_id, label, status, records)
    # Emit terminal status counter per source seen.
    final_status = "success" if run_status == "success" else "failed"
    for s in sources_seen or {"unknown"}:
        _record_ingestion_run_safe(status=final_status, source=s)
    return written
