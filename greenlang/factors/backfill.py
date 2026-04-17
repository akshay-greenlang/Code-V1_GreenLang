# -*- coding: utf-8 -*-
"""
S6: Backfill strategy for legacy or partially ingested rows before strict governance.

Fills optional canonical v0.1 fields when safe defaults apply; does not overwrite
explicit values. Call before persisting migrated editions.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from greenlang.data.emission_factor_record import EmissionFactorRecord


def _infer_license_class_from_license_tag(license_tag: str) -> Optional[str]:
    t = (license_tag or "").strip().lower()
    if not t:
        return None
    if "cc0" in t or "public domain" in t:
        return "public_domain"
    if "ogl" in t or "open government" in t:
        return "uk_open_government"
    if "proprietary" in t or "commercial" in t:
        return "commercial_connector"
    return None


def backfill_missing_governance(
    record: EmissionFactorRecord,
    *,
    default_source_id: Optional[str] = None,
) -> EmissionFactorRecord:
    """
    Populate ``license_class`` from ``license_info.license`` when unset.
    Optionally set ``source_id`` when missing and ``default_source_id`` is provided.
    """
    lic_class = record.license_class
    if lic_class is None and record.license_info:
        lic_class = _infer_license_class_from_license_tag(record.license_info.license)

    sid = record.source_id
    if sid is None and default_source_id:
        sid = default_source_id

    if lic_class == record.license_class and sid == record.source_id:
        return record
    return replace(record, license_class=lic_class or record.license_class, source_id=sid)
