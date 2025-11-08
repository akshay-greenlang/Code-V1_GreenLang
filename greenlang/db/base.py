"""
Database base configuration and utilities for Phase 4
"""

import os
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

# Create declarative base
Base = declarative_base()

# Global engine and session factory
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_database_url() -> str:
    """
    Get database URL from environment or config

    Returns:
        Database connection URL
    """
    # Check environment variable
    db_url = os.getenv("GREENLANG_DATABASE_URL")

    if db_url:
        return db_url

    # Default to SQLite for development
    db_path = os.getenv("GREENLANG_DB_PATH", "~/.greenlang/greenlang.db")
    db_path = os.path.expanduser(db_path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    return f"sqlite:///{db_path}"


def get_engine(database_url: Optional[str] = None, **kwargs) -> Engine:
    """
    Get SQLAlchemy engine (singleton)

    Args:
        database_url: Optional database URL (uses environment if not provided)
        **kwargs: Additional engine configuration

    Returns:
        SQLAlchemy Engine
    """
    global _engine

    if _engine is None:
        url = database_url or get_database_url()

        # Default engine configuration
        engine_config = {
            "poolclass": QueuePool,
            "pool_size": kwargs.get("pool_size", 5),
            "max_overflow": kwargs.get("max_overflow", 10),
            "pool_timeout": kwargs.get("pool_timeout", 30),
            "pool_recycle": kwargs.get("pool_recycle", 3600),
            "echo": kwargs.get("echo", False),
        }

        # SQLite-specific configuration
        if url.startswith("sqlite"):
            engine_config = {
                "connect_args": {"check_same_thread": False},
                "echo": kwargs.get("echo", False),
            }

        _engine = create_engine(url, **engine_config)

        # Enable foreign keys for SQLite
        if url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    return _engine


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    """
    Get session factory (singleton)

    Args:
        engine: Optional SQLAlchemy engine

    Returns:
        Session factory
    """
    global _SessionLocal

    if _SessionLocal is None:
        eng = engine or get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=eng)

    return _SessionLocal


@contextmanager
def get_session(engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """
    Get database session context manager

    Args:
        engine: Optional SQLAlchemy engine

    Yields:
        Database session
    """
    SessionLocal = get_session_factory(engine)
    session = SessionLocal()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Optional[Engine] = None, drop_all: bool = False) -> None:
    """
    Initialize database (create all tables)

    Args:
        engine: Optional SQLAlchemy engine
        drop_all: If True, drop all tables first
    """
    eng = engine or get_engine()

    if drop_all:
        Base.metadata.drop_all(bind=eng)

    Base.metadata.create_all(bind=eng)


def reset_engine() -> None:
    """Reset engine singleton (useful for testing)"""
    global _engine, _SessionLocal

    if _engine:
        _engine.dispose()
        _engine = None

    _SessionLocal = None
