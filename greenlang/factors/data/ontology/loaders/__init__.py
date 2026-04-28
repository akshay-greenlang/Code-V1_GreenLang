# -*- coding: utf-8 -*-
"""GreenLang Factors v0.1 Alpha - Phase 2 ontology seed loaders.

This package provides idempotent loaders that read canonical YAML seed
files and INSERT rows into the ``factors_v0_1.geography``,
``factors_v0_1.unit``, ``factors_v0_1.methodology``, and (Phase 2 WS5)
``factors_v0_1.activity`` registry tables.

Public surface:
    * :func:`load_geography`     - load geography_seed_v0_1.yaml
    * :func:`load_units`         - load unit_seed_v0_1.yaml
    * :func:`load_methodologies` - load methodology_seed_v0_1.yaml
    * :mod:`activity_loader`     - load activity_seed_v0_1.yaml (WS5)
      (uses its own ``ActivityRow`` + ``load_into_postgres`` /
      ``load_into_sqlite`` API; see ``activity_loader.py``)

All loaders accept any DB-API 2.0 connection (``psycopg2``, ``psycopg``
v3, or ``sqlite3``). Schema-qualified statements use the
``factors_v0_1.<table>`` form on PostgreSQL; for sqlite3 (used in unit
tests) the loaders auto-detect and fall back to bare table names.

Each loader is idempotent: re-running is a no-op via
``INSERT ... ON CONFLICT (urn) DO NOTHING`` (PG) or
``INSERT OR IGNORE`` (sqlite3). Each loader returns a :class:`LoadReport`
with ``count_inserted`` + ``count_skipped`` + ``total_seen``.

Every URN is validated via
:func:`greenlang.factors.ontology.urn.parse` *before* it touches the DB.
A malformed URN aborts the load and no rows are committed (loaders do
not call ``conn.commit()`` -- the caller owns the transaction).
"""
from __future__ import annotations

from greenlang.factors.data.ontology.loaders._common import (
    LoadReport,
    PHASE2_SEED_SOURCE,
)
from greenlang.factors.data.ontology.loaders.geography_loader import (
    GEOGRAPHY_SEED_PATH,
    GeographyLoaderError,
    GeographyRow,
    create_sqlite_geography_table,
    load_geography,
)
from greenlang.factors.data.ontology.loaders.methodology_loader import (
    METHODOLOGY_SEED_PATH,
    MethodologyLoaderError,
    MethodologyRow,
    create_sqlite_methodology_table,
    load_methodologies,
)
from greenlang.factors.data.ontology.loaders.unit_loader import (
    UNIT_SEED_PATH,
    UnitLoaderError,
    UnitRow,
    create_sqlite_unit_table,
    load_units,
)

# Phase 2 / WS5 — activity taxonomy loader. Lives in its own module
# rather than _common because the activity table is created by V502
# (separate from the V501 ontology-additive migration) and uses the
# Phase 2 taxonomy form of the activity URN
# (``urn:gl:activity:<taxonomy>:<code>``).
from greenlang.factors.data.ontology.loaders.activity_loader import (
    DEFAULT_SEED_PATH as ACTIVITY_SEED_PATH,
    ActivityLoaderError,
    ActivityRow,
    create_sqlite_activity_table,
    load_into_postgres as load_activities_into_postgres,
    load_into_sqlite as load_activities_into_sqlite,
    load_seed_yaml as load_activities_seed,
)

__all__ = [
    # Common
    "LoadReport",
    "PHASE2_SEED_SOURCE",
    # Geography
    "load_geography",
    "GeographyRow",
    "GeographyLoaderError",
    "GEOGRAPHY_SEED_PATH",
    "create_sqlite_geography_table",
    # Unit
    "load_units",
    "UnitRow",
    "UnitLoaderError",
    "UNIT_SEED_PATH",
    "create_sqlite_unit_table",
    # Methodology
    "load_methodologies",
    "MethodologyRow",
    "MethodologyLoaderError",
    "METHODOLOGY_SEED_PATH",
    "create_sqlite_methodology_table",
    # Activity (Phase 2 / WS5)
    "ACTIVITY_SEED_PATH",
    "ActivityLoaderError",
    "ActivityRow",
    "create_sqlite_activity_table",
    "load_activities_into_postgres",
    "load_activities_into_sqlite",
    "load_activities_seed",
]
