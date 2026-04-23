# -*- coding: utf-8 -*-
"""greenlang.factors.data — canonical v1 factor record models + migration utilities.

W4-A (2026-04-23): canonical-record migration per
``docs/specs/schema_v1_gap_report.md``. The primary entry points are:

* :mod:`canonical_v1`      — the v1 Pydantic model + enums.
* :mod:`canonical_v2`      — deprecated compat shim that re-exports v1.
* :mod:`categorical_parameters` — 7 per-family discriminated-union models.
* :mod:`gwp_registry`      — external GWP coefficient registry.
* :mod:`source_object`     — source catalog object per source_object_v1.schema.json.
"""
from __future__ import annotations

# Intentionally thin — individual modules do the re-exports.
__all__: list[str] = []
