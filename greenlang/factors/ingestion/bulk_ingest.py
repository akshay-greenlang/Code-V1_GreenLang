# -*- coding: utf-8 -*-
"""
Bulk ingestion pipeline (F019).

Accepts a list of (source_id, file_path) pairs, routes each to the
correct parser via ParserRegistry, runs QA validation, and aggregates
results into a pending edition with manifest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from greenlang.data.emission_factor_record import EmissionFactorRecord
from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository
from greenlang.factors.edition_manifest import build_manifest_for_factors
from greenlang.factors.etl.normalize import dict_to_emission_factor_record
from greenlang.factors.etl.qa import validate_factor_dict
from greenlang.factors.ingestion.parsers import ParserRegistry, build_default_registry

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Summary of a bulk ingestion run."""

    edition_id: str
    total_ingested: int = 0
    total_rejected: int = 0
    per_source: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edition_id": self.edition_id,
            "total_ingested": self.total_ingested,
            "total_rejected": self.total_rejected,
            "per_source": self.per_source,
            "errors": self.errors[:50],
            "warnings": self.warnings[:50],
        }


def _load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load a JSON file, returning (data, None) or (None, error_msg)."""
    if not path.is_file():
        return None, f"{path}: file not found"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path}: {exc}"


def bulk_ingest(
    sources: Sequence[Tuple[str, Path]],
    sqlite_path: Path,
    edition_id: str,
    *,
    label: str = "Bulk ingestion",
    status: str = "preview",
    registry: Optional[ParserRegistry] = None,
    skip_unknown_sources: bool = True,
) -> IngestionResult:
    """
    Run bulk ingestion from multiple source files.

    Args:
        sources: List of (source_id, file_path) pairs.
        sqlite_path: Path to SQLite catalog database.
        edition_id: Target edition ID (e.g. "2026.04.0").
        label: Edition label.
        status: Edition status (default "preview").
        registry: Optional ParserRegistry (defaults to built-in).
        skip_unknown_sources: If True, skip sources with no parser; if False, error.

    Returns:
        IngestionResult with counts and error details.
    """
    if registry is None:
        registry = build_default_registry()

    result = IngestionResult(edition_id=edition_id)
    all_records: List[EmissionFactorRecord] = []
    seen_ids: set[str] = set()

    for source_id, file_path in sources:
        parser = registry.get(source_id)
        if parser is None:
            msg = f"No parser registered for source_id={source_id}"
            if skip_unknown_sources:
                result.warnings.append(msg)
                logger.warning(msg)
                continue
            else:
                result.errors.append(msg)
                logger.error(msg)
                continue

        # Load JSON
        data, err = _load_json(file_path)
        if err:
            result.errors.append(err)
            logger.error("Failed to load %s: %s", file_path, err)
            continue

        # Validate schema
        ok, issues = parser.validate_schema(data)
        if not ok:
            for issue in issues:
                result.warnings.append(f"{source_id}: schema issue: {issue}")
            logger.warning("Schema issues for %s: %s", source_id, issues)

        # Parse
        try:
            factor_dicts = parser.parse(data)
        except Exception as exc:
            result.errors.append(f"{source_id}: parser failed: {exc}")
            logger.error("Parser %s failed: %s", parser.parser_id, exc)
            continue

        # QA + convert
        source_count = 0
        for fd in factor_dicts:
            fid = fd.get("factor_id", "")

            # Dedup across sources
            if fid in seen_ids:
                result.warnings.append(f"Duplicate factor_id {fid} from {source_id}, skipping")
                continue
            seen_ids.add(fid)

            # QA gates
            ok, qa_errors = validate_factor_dict(fd)
            if not ok:
                result.total_rejected += 1
                result.errors.append(f"{source_id}: QA failed for {fid}: {qa_errors}")
                continue

            # Convert to EmissionFactorRecord
            try:
                rec = dict_to_emission_factor_record(fd)
                all_records.append(rec)
                source_count += 1
            except Exception as exc:
                result.total_rejected += 1
                result.errors.append(f"{source_id}: conversion failed for {fid}: {exc}")

        result.per_source[source_id] = source_count
        logger.info("Source %s: %d factors ingested, %d total errors", source_id, source_count, len(result.errors))

    # Write to catalog
    if all_records:
        repo = SqliteFactorCatalogRepository(sqlite_path)
        manifest = build_manifest_for_factors(
            edition_id,
            status,
            all_records,
            changelog=[
                f"Bulk ingestion: {len(all_records)} factors from {len(result.per_source)} sources",
            ],
        )
        repo.upsert_edition(
            edition_id,
            status,
            label,
            manifest.to_dict(),
            manifest.changelog,
        )
        repo.insert_factors(edition_id, all_records)

    result.total_ingested = len(all_records)
    logger.info(
        "Bulk ingestion complete: edition=%s ingested=%d rejected=%d sources=%d",
        edition_id, result.total_ingested, result.total_rejected, len(result.per_source),
    )
    return result
