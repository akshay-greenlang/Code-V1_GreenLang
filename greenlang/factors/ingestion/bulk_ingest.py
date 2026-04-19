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


def bulk_ingest_pipeline(
    sqlite_path: Path,
    edition_id: str,
    *,
    include_builtin: bool = True,
    include_synthetic: bool = True,
    synthetic_count: int = 25000,
    synthetic_seed: int = 42,
    synthetic_years: Optional[List[int]] = None,
    source_files: Optional[Sequence[Tuple[str, Path]]] = None,
    label: str = "Bulk ingestion pipeline",
    status: str = "stable",
    dry_run: bool = False,
) -> IngestionResult:
    """Orchestrate the full ingestion pipeline: builtin + sources + synthetic.

    This is the master pipeline entry point that combines all ingestion
    modes into a single edition. It runs in three phases:

    1. Built-in EmissionFactorDatabase (327 certified factors)
    2. Real source files via ParserRegistry (if source_files provided)
    3. Synthetic factors for dev/test fill (if include_synthetic=True)

    Args:
        sqlite_path: Path to SQLite catalog database.
        edition_id: Target edition ID (e.g. "2026.04.0").
        include_builtin: Load built-in factors (default True).
        include_synthetic: Generate synthetic factors (default True).
        synthetic_count: Number of synthetic factors (default 25000).
        synthetic_seed: Random seed for reproducibility (default 42).
        synthetic_years: Years for synthetic data (default [2022..2025]).
        source_files: List of (source_id, file_path) pairs for real sources.
        label: Edition label.
        status: Edition status.
        dry_run: If True, validate all data but do not write to database.

    Returns:
        IngestionResult with counts and error details.
    """
    result = IngestionResult(edition_id=edition_id)
    all_records: List[EmissionFactorRecord] = []
    seen_ids: set[str] = set()

    # Phase 1: Built-in factors
    if include_builtin:
        logger.info("Phase 1: Loading built-in EmissionFactorDatabase...")
        try:
            from greenlang.data.emission_factor_database import EmissionFactorDatabase

            db = EmissionFactorDatabase(enable_cache=False)
            builtin_records = list(db.factors.values())
            for rec in builtin_records:
                if rec.factor_id not in seen_ids:
                    seen_ids.add(rec.factor_id)
                    all_records.append(rec)
            result.per_source["greenlang_builtin"] = len(builtin_records)
            logger.info("Phase 1 complete: %d built-in factors", len(builtin_records))
        except Exception as exc:
            result.errors.append(f"Built-in loading failed: {exc}")
            logger.error("Phase 1 failed: %s", exc)

    # Phase 2: Real source files
    if source_files:
        logger.info("Phase 2: Parsing %d source files...", len(source_files))
        registry = build_default_registry()
        for source_id, file_path in source_files:
            parser = registry.get(source_id)
            if parser is None:
                result.warnings.append(f"No parser for source_id={source_id}")
                continue

            data, err = _load_json(file_path)
            if err:
                result.errors.append(err)
                continue

            try:
                factor_dicts = parser.parse(data)
            except Exception as exc:
                result.errors.append(f"{source_id}: parser failed: {exc}")
                continue

            source_count = 0
            for fd in factor_dicts:
                fid = fd.get("factor_id", "")
                if fid in seen_ids:
                    continue
                seen_ids.add(fid)

                ok, qa_errors = validate_factor_dict(fd)
                if not ok:
                    result.total_rejected += 1
                    continue

                try:
                    rec = dict_to_emission_factor_record(fd)
                    all_records.append(rec)
                    source_count += 1
                except Exception as exc:
                    result.total_rejected += 1
                    result.errors.append(f"{source_id}: {fid}: {exc}")

            result.per_source[source_id] = source_count
        logger.info("Phase 2 complete: %d source files processed", len(source_files))

    # Phase 3: Synthetic factors
    if include_synthetic:
        logger.info("Phase 3: Generating %d synthetic factors (seed=%d)...", synthetic_count, synthetic_seed)
        try:
            from greenlang.factors.ingestion.synthetic_data import generate_and_validate

            valid_dicts, total_gen, total_rej = generate_and_validate(
                count=synthetic_count,
                seed=synthetic_seed,
                years=synthetic_years,
            )
            syn_count = 0
            for fd in valid_dicts:
                fid = fd.get("factor_id", "")
                if fid in seen_ids:
                    continue
                seen_ids.add(fid)
                try:
                    rec = dict_to_emission_factor_record(fd)
                    all_records.append(rec)
                    syn_count += 1
                except Exception:
                    result.total_rejected += 1

            result.per_source["synthetic"] = syn_count
            result.total_rejected += total_rej
            logger.info(
                "Phase 3 complete: generated=%d valid=%d ingested=%d",
                total_gen, len(valid_dicts), syn_count,
            )
        except Exception as exc:
            result.errors.append(f"Synthetic generation failed: {exc}")
            logger.error("Phase 3 failed: %s", exc)

    # Write to catalog (unless dry-run)
    if dry_run:
        result.total_ingested = len(all_records)
        logger.info(
            "[DRY RUN] Would insert %d factors into %s edition=%s",
            result.total_ingested, sqlite_path, edition_id,
        )
        return result

    if all_records:
        repo = SqliteFactorCatalogRepository(sqlite_path)
        manifest = build_manifest_for_factors(
            edition_id,
            status,
            all_records,
            changelog=[
                f"Pipeline ingestion: {len(all_records)} factors from {len(result.per_source)} sources",
                f"Sources: {', '.join(sorted(result.per_source.keys()))}",
            ],
        )
        repo.upsert_edition(
            edition_id, status, label,
            manifest.to_dict(), manifest.changelog,
        )
        repo.insert_factors(edition_id, all_records)

    result.total_ingested = len(all_records)
    logger.info(
        "Pipeline ingestion complete: edition=%s ingested=%d rejected=%d sources=%d",
        edition_id, result.total_ingested, result.total_rejected, len(result.per_source),
    )
    return result
