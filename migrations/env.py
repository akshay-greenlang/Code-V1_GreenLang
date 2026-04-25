# -*- coding: utf-8 -*-
"""
Alembic environment for the GreenLang Factors v0.1 migration tree.

This is the *factors* Alembic tree. It is intentionally separate from
``data/migrations/env.py`` (Phase-4 auth) and
``greenlang/auth/migrations/env.py`` (auth service) so the FY27 Factors
product can be promoted independently of those modules.

Conventions
-----------
- Database URL precedence (highest first):
    1. env var ``ALEMBIC_SQLALCHEMY_URL``
    2. env var ``DATABASE_URL``
    3. ``sqlalchemy.url`` from ``alembic.ini``
- Default schema for every revision in this tree is ``factors_v0_1``;
  individual revisions may switch search_path themselves.
- ``target_metadata`` is intentionally ``None``. Revisions for the
  factors tree are raw-SQL (the canonical V### files in
  ``deployment/database/migrations/sql`` remain the source of truth).
  When ORM models for factors land they can be wired in here without
  touching the per-revision files.
- ``compare_type=True`` and ``compare_server_default=True`` are enabled
  so future ``--autogenerate`` runs catch column-type drift.

Wave: C / TaskCreate #3 / WS1-T3
"""

from __future__ import annotations

import os
from logging.config import fileConfig
from typing import Optional

from alembic import context
from sqlalchemy import engine_from_config, pool


# ---------------------------------------------------------------------------
# Config bootstrap
# ---------------------------------------------------------------------------

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# ORM metadata: factors tree uses raw-SQL revisions; no autogen target.
target_metadata = None


# Default search_path applied to every connection in this tree.
FACTORS_SEARCH_PATH = "factors_v0_1, public"


def _resolve_database_url() -> str:
    """Return the database URL using documented precedence.

    Order:
        1. env ``ALEMBIC_SQLALCHEMY_URL``
        2. env ``DATABASE_URL``
        3. ``sqlalchemy.url`` from alembic.ini
    """
    url: Optional[str] = (
        os.environ.get("ALEMBIC_SQLALCHEMY_URL")
        or os.environ.get("DATABASE_URL")
        or config.get_main_option("sqlalchemy.url")
    )
    if not url:
        raise RuntimeError(
            "No database URL configured. Set ALEMBIC_SQLALCHEMY_URL or "
            "DATABASE_URL, or populate sqlalchemy.url in alembic.ini."
        )
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout)."""
    url = _resolve_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        version_table_schema="factors_v0_1",
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (real connection)."""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _resolve_database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Ensure schema exists before any revision runs and pin search_path
        # so raw-SQL revisions can refer to bare table names.
        connection.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS factors_v0_1")
        connection.exec_driver_sql(
            f"SET search_path TO {FACTORS_SEARCH_PATH}"
        )

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            version_table_schema="factors_v0_1",
            include_schemas=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
