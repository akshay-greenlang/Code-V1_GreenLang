"""
Alembic Environment Configuration for GL-Agent-Factory

This module configures the Alembic migration environment for both
synchronous and asynchronous database operations.

Features:
- Async PostgreSQL support via asyncpg
- Model metadata auto-discovery
- Environment variable configuration
- Transaction-per-migration safety
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import Base and all models for autogenerate support
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.base import Base
from models.tenant import Tenant, TenantUsageLog, TenantInvitation
from models.agent import Agent
from models.agent_version import AgentVersion
from models.execution import Execution
from models.audit_log import AuditLog
from models.user import User

# Alembic Config object - provides access to alembic.ini values
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate support
target_metadata = Base.metadata


def get_url() -> str:
    """
    Get database URL from environment or config.

    Environment variable ALEMBIC_DATABASE_URL takes precedence over
    the value in alembic.ini for production deployments.

    Returns:
        Database connection URL string
    """
    url = os.getenv("ALEMBIC_DATABASE_URL")
    if url:
        return url

    # Fall back to alembic.ini configuration
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # PostgreSQL-specific options
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Run migrations within a database connection context.

    Args:
        connection: Active database connection
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # PostgreSQL-specific options
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        # Transaction configuration
        transaction_per_migration=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in 'online' mode with async engine.

    Creates an async Engine and associates a connection with the context.
    Uses asyncpg driver for PostgreSQL async operations.
    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Delegates to async migration runner for asyncpg support.
    """
    asyncio.run(run_async_migrations())


# Determine offline/online mode and run appropriate migration function
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
