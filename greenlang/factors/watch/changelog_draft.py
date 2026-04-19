# -*- coding: utf-8 -*-
"""
Machine changelog draft before human approval (U5).

Generates human-readable changelog lines by comparing two editions of
the factor catalog. Groups changes by source, identifies numeric
corrections and deprecations, and produces output suitable for review
before publication.

Format (per docs/factors/developer_guide.md lines 322-338):
    edition diff {left} -> {right}
    added: {count}
    removed: {count}
    changed: {count}

    Sources updated:
    - {source_org} {old_year} -> {new_year} (reason)

    Numeric corrections:
    - {factor_id}: {old_value} -> {new_value} {unit}

    Deprecations:
    - {old_factor_id} -> replaced by {new_factor_id}
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from greenlang.factors.catalog_repository import FactorCatalogRepository

logger = logging.getLogger(__name__)


def draft_changelog_lines(compare: Dict[str, Any]) -> List[str]:
    """
    Backward-compatible quick-draft from a pre-computed compare dict.

    This is the original entry point retained for callers that already
    have a compare result from ``FactorCatalogService.compare_editions()``.

    Args:
        compare: Dict with keys ``left_edition_id``, ``right_edition_id``,
                 ``added_factor_ids``, ``removed_factor_ids``,
                 ``changed_factor_ids``.

    Returns:
        List of human-readable changelog lines (header only, no detail).
    """
    lines = [
        f"edition diff {compare.get('left_edition_id')} -> {compare.get('right_edition_id')}",
        f"added: {len(compare.get('added_factor_ids') or [])}",
        f"removed: {len(compare.get('removed_factor_ids') or [])}",
        f"changed: {len(compare.get('changed_factor_ids') or [])}",
    ]
    return lines


def _build_factor_index(
    repo: FactorCatalogRepository,
    edition_id: str,
) -> Dict[str, Dict[str, str]]:
    """
    Build a factor_id -> summary dict map for an edition.

    Each summary has keys: factor_id, content_hash, factor_status.
    """
    summaries = repo.list_factor_summaries(edition_id)
    return {s["factor_id"]: s for s in summaries}


def _safe_get_factor_dict(
    repo: FactorCatalogRepository,
    edition_id: str,
    factor_id: str,
) -> Optional[Dict[str, Any]]:
    """Fetch a factor and convert to dict, returning None on failure."""
    try:
        rec = repo.get_factor(edition_id, factor_id)
        if rec is None:
            return None
        return rec.to_dict()
    except Exception as exc:
        logger.warning(
            "Failed to fetch factor %s from edition %s: %s",
            factor_id, edition_id, exc,
        )
        return None


def _get_source_org(factor_dict: Dict[str, Any]) -> str:
    """Extract source_org from a factor dict."""
    prov = factor_dict.get("provenance")
    if isinstance(prov, dict):
        return prov.get("source_org", "Unknown")
    return "Unknown"


def _get_source_year(factor_dict: Dict[str, Any]) -> Optional[int]:
    """Extract source_year from a factor dict."""
    prov = factor_dict.get("provenance")
    if isinstance(prov, dict):
        val = prov.get("source_year")
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    return None


def _get_co2e_total(factor_dict: Dict[str, Any]) -> Optional[float]:
    """Extract co2e_total from a factor dict."""
    gwp = factor_dict.get("gwp_100yr")
    if isinstance(gwp, dict):
        val = gwp.get("co2e_total")
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def _get_unit(factor_dict: Dict[str, Any]) -> str:
    """Extract unit from a factor dict."""
    return factor_dict.get("unit", "unit")


def _get_source_id(factor_dict: Dict[str, Any]) -> str:
    """Extract source_id, falling back to source_org."""
    sid = factor_dict.get("source_id")
    if sid:
        return str(sid)
    return _get_source_org(factor_dict)


def draft_changelog(
    left_edition: str,
    right_edition: str,
    repo: FactorCatalogRepository,
) -> List[str]:
    """
    Generate a human-readable changelog between two editions.

    Fetches factor summaries from both editions, computes the diff
    (added/removed/changed), then inspects changed factors to group
    them by source, identify numeric corrections, and flag deprecations.

    Args:
        left_edition: The older (baseline) edition ID.
        right_edition: The newer (target) edition ID.
        repo: Factor catalog repository for data access.

    Returns:
        List of human-readable changelog lines.
    """
    lines: List[str] = []

    # ------------------------------------------------------------------
    # Step 1: Build factor indexes and compute set differences
    # ------------------------------------------------------------------
    left_index = _build_factor_index(repo, left_edition)
    right_index = _build_factor_index(repo, right_edition)

    left_ids = set(left_index.keys())
    right_ids = set(right_index.keys())

    added_ids = sorted(right_ids - left_ids)
    removed_ids = sorted(left_ids - right_ids)
    changed_ids = sorted(
        fid for fid in left_ids & right_ids
        if left_index[fid]["content_hash"] != right_index[fid]["content_hash"]
    )

    # ------------------------------------------------------------------
    # Step 2: Header
    # ------------------------------------------------------------------
    lines.append(f"edition diff {left_edition} -> {right_edition}")
    lines.append(f"added: {len(added_ids)}")
    lines.append(f"removed: {len(removed_ids)}")
    lines.append(f"changed: {len(changed_ids)}")

    if not added_ids and not removed_ids and not changed_ids:
        lines.append("")
        lines.append("No changes between editions.")
        return lines

    # ------------------------------------------------------------------
    # Step 3: Inspect changed factors for detailed sections
    # ------------------------------------------------------------------
    source_updates: Dict[str, Dict[str, Any]] = {}
    numeric_corrections: List[Dict[str, Any]] = []
    deprecations: List[Dict[str, Any]] = []

    # Group added factors by source for the "Sources updated" section
    added_by_source: Dict[str, int] = defaultdict(int)
    for fid in added_ids:
        right_dict = _safe_get_factor_dict(repo, right_edition, fid)
        if right_dict:
            src = _get_source_id(right_dict)
            added_by_source[src] += 1

    # Analyze changed factors
    for fid in changed_ids:
        left_dict = _safe_get_factor_dict(repo, left_edition, fid)
        right_dict = _safe_get_factor_dict(repo, right_edition, fid)
        if not left_dict or not right_dict:
            continue

        src = _get_source_id(right_dict)

        # Track source year updates
        old_year = _get_source_year(left_dict)
        new_year = _get_source_year(right_dict)
        if old_year and new_year and old_year != new_year:
            source_org = _get_source_org(right_dict)
            source_key = f"{source_org}_{old_year}_{new_year}"
            if source_key not in source_updates:
                source_updates[source_key] = {
                    "source_org": source_org,
                    "old_year": old_year,
                    "new_year": new_year,
                    "factor_count": 0,
                }
            source_updates[source_key]["factor_count"] += 1

        # Detect numeric corrections (co2e_total changed)
        old_co2e = _get_co2e_total(left_dict)
        new_co2e = _get_co2e_total(right_dict)
        if old_co2e is not None and new_co2e is not None and old_co2e != new_co2e:
            numeric_corrections.append({
                "factor_id": fid,
                "old_value": old_co2e,
                "new_value": new_co2e,
                "unit": _get_unit(right_dict),
            })

        # Detect deprecations (replacement_factor_id set in right)
        old_replacement = left_dict.get("replacement_factor_id")
        new_replacement = right_dict.get("replacement_factor_id")
        old_status = left_dict.get("factor_status", "certified")
        new_status = right_dict.get("factor_status", "certified")

        is_newly_deprecated = (
            (new_replacement and not old_replacement)
            or (new_status == "deprecated" and old_status != "deprecated")
        )
        if is_newly_deprecated:
            deprecations.append({
                "factor_id": fid,
                "replacement_factor_id": new_replacement or "none",
            })

    # ------------------------------------------------------------------
    # Step 4: Sources updated section
    # ------------------------------------------------------------------
    if source_updates:
        lines.append("")
        lines.append("Sources updated:")
        for entry in sorted(source_updates.values(), key=lambda e: e["source_org"]):
            count_note = (
                f" ({entry['factor_count']} factors)"
                if entry["factor_count"] > 1
                else ""
            )
            lines.append(
                f"- {entry['source_org']} {entry['old_year']} -> "
                f"{entry['new_year']} (source year update{count_note})"
            )

    # ------------------------------------------------------------------
    # Step 5: Numeric corrections section
    # ------------------------------------------------------------------
    if numeric_corrections:
        lines.append("")
        lines.append("Numeric corrections:")
        for nc in sorted(numeric_corrections, key=lambda x: x["factor_id"]):
            lines.append(
                f"- {nc['factor_id']}: {nc['old_value']} -> "
                f"{nc['new_value']} {nc['unit']}"
            )

    # ------------------------------------------------------------------
    # Step 6: Deprecations section
    # ------------------------------------------------------------------
    if deprecations:
        lines.append("")
        lines.append("Deprecations:")
        for dep in sorted(deprecations, key=lambda x: x["factor_id"]):
            lines.append(
                f"- {dep['factor_id']} -> replaced by {dep['replacement_factor_id']}"
            )

    # ------------------------------------------------------------------
    # Step 7: Summary of additions (by source) if present
    # ------------------------------------------------------------------
    if added_by_source:
        lines.append("")
        lines.append("New factors added:")
        for src in sorted(added_by_source.keys()):
            lines.append(f"- {src}: {added_by_source[src]} factors")

    logger.info(
        "Changelog draft generated: %s -> %s "
        "added=%d removed=%d changed=%d "
        "source_updates=%d numeric_corrections=%d deprecations=%d",
        left_edition,
        right_edition,
        len(added_ids),
        len(removed_ids),
        len(changed_ids),
        len(source_updates),
        len(numeric_corrections),
        len(deprecations),
    )

    return lines
