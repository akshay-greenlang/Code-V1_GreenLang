"""
Database Connection
===================

Database connection management for GreenLang.

Author: Data Team
Created: 2025-11-21
"""

from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, field
import logging
from contextlib import contextmanager
from datetime import datetime
import sqlite3
import json

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Database-related error."""
    pass


@dataclass
class ConnectionConfig:
    """Database connection configuration."""
    db_type: str = "sqlite"  # 'sqlite', 'postgresql', 'mysql'
    host: Optional[str] = None
    port: Optional[int] = None
    database: str = "greenlang.db"
    username: Optional[str] = None
    password: Optional[str] = None
    pool_size: int = 5
    timeout: int = 30
    echo: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectionPool:
    """Simple connection pool for database connections."""

    def __init__(self, config: ConnectionConfig):
        """Initialize connection pool."""
        self.config = config
        self.connections: List[Any] = []
        self.available: List[Any] = []
        self.in_use: Dict[Any, datetime] = {}

    def get_connection(self):
        """Get a connection from the pool."""
        if self.available:
            conn = self.available.pop()
        elif len(self.connections) < self.config.pool_size:
            conn = self._create_connection()
            self.connections.append(conn)
        else:
            raise DatabaseError("Connection pool exhausted")

        self.in_use[conn] = datetime.now()
        return conn

    def return_connection(self, conn):
        """Return a connection to the pool."""
        if conn in self.in_use:
            del self.in_use[conn]
            self.available.append(conn)

    def _create_connection(self):
        """Create a new database connection."""
        if self.config.db_type == "sqlite":
            return sqlite3.connect(
                self.config.database,
                timeout=self.config.timeout
            )
        else:
            raise NotImplementedError(f"Database type {self.config.db_type} not implemented")

    def close_all(self):
        """Close all connections in the pool."""
        for conn in self.connections:
            try:
                conn.close()
            except:
                pass
        self.connections.clear()
        self.available.clear()
        self.in_use.clear()


class DatabaseConnection:
    """
    Main database connection class for GreenLang.

    Provides connection management, query execution, and transaction support.
    """

    def __init__(self, config: Optional[ConnectionConfig] = None):
        """Initialize database connection."""
        self.config = config or ConnectionConfig()
        self.pool = ConnectionPool(self.config)
        self._current_connection = None
        self.query_count = 0
        self.transaction_count = 0

        # Initialize database
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database with required tables."""
        with self.get_connection() as conn:
            if self.config.db_type == "sqlite":
                cursor = conn.cursor()

                # Create emission_factors table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS emission_factors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        material_id TEXT NOT NULL,
                        factor REAL NOT NULL,
                        unit TEXT NOT NULL,
                        source TEXT,
                        region TEXT,
                        year INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create activity_data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS activity_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        activity_id TEXT NOT NULL,
                        activity_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        unit TEXT NOT NULL,
                        timestamp TIMESTAMP,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create suppliers table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS suppliers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        supplier_id TEXT NOT NULL UNIQUE,
                        name TEXT NOT NULL,
                        country TEXT,
                        industry TEXT,
                        emissions_data TEXT,
                        certification TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create audit_log table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        event_data TEXT,
                        user_id TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ip_address TEXT,
                        session_id TEXT
                    )
                ''')

                conn.commit()
                logger.info("Database initialized")

    @contextmanager
    def get_connection(self):
        """Get a database connection context manager."""
        conn = self.pool.get_connection()
        try:
            yield conn
        finally:
            self.pool.return_connection(conn)

    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """
        Execute a database query.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Query result
        """
        self.query_count += 1

        with self.get_connection() as conn:
            cursor = conn.cursor()

            if self.config.echo:
                logger.debug(f"Executing query: {query} with params: {params}")

            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                # Commit if it's a write operation
                if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                    conn.commit()

                # Return results for SELECT
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()

                return cursor.lastrowid

            except Exception as e:
                logger.error(f"Query failed: {str(e)}")
                conn.rollback()
                raise DatabaseError(f"Query execution failed: {str(e)}")

    def query(self, model_class: Type, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Query database for model instances.

        Args:
            model_class: Model class to query
            filters: Optional filters

        Returns:
            List of model instances
        """
        table_name = model_class.__tablename__

        # Build query
        query = f"SELECT * FROM {table_name}"
        params = []

        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"{key} = ?")
                params.append(value)
            query += " WHERE " + " AND ".join(conditions)

        # Execute query
        results = self.execute(query, tuple(params) if params else None)

        # Convert to model instances
        if results:
            return [model_class.from_row(row) for row in results]
        return []

    def insert(self, model_instance) -> int:
        """
        Insert a model instance into database.

        Args:
            model_instance: Model instance to insert

        Returns:
            Inserted row ID
        """
        table_name = model_instance.__tablename__
        data = model_instance.to_dict()

        # Remove id if present
        data.pop('id', None)

        # Build insert query
        columns = list(data.keys())
        placeholders = ['?' for _ in columns]
        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

        # Execute insert
        return self.execute(query, tuple(data.values()))

    def update(self, model_instance) -> bool:
        """
        Update a model instance in database.

        Args:
            model_instance: Model instance to update

        Returns:
            True if successful
        """
        table_name = model_instance.__tablename__
        data = model_instance.to_dict()

        # Get id
        model_id = data.pop('id')
        if not model_id:
            raise DatabaseError("Model instance must have an id for update")

        # Build update query
        set_clauses = [f"{key} = ?" for key in data.keys()]
        query = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE id = ?"

        # Execute update
        values = list(data.values()) + [model_id]
        self.execute(query, tuple(values))
        return True

    def delete(self, model_class: Type, model_id: int) -> bool:
        """
        Delete a model instance from database.

        Args:
            model_class: Model class
            model_id: Model ID to delete

        Returns:
            True if successful
        """
        table_name = model_class.__tablename__
        query = f"DELETE FROM {table_name} WHERE id = ?"
        self.execute(query, (model_id,))
        return True

    @contextmanager
    def transaction(self):
        """Start a database transaction."""
        self.transaction_count += 1
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
                logger.debug("Transaction committed")
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction rolled back: {str(e)}")
                raise

    def lookup_emission_factor(self, material_id: str, region: Optional[str] = None) -> Optional[float]:
        """
        Look up emission factor for a material.

        Args:
            material_id: Material identifier
            region: Optional region filter

        Returns:
            Emission factor or None if not found
        """
        query = "SELECT factor FROM emission_factors WHERE material_id = ?"
        params = [material_id]

        if region:
            query += " AND region = ?"
            params.append(region)

        results = self.execute(query, tuple(params))
        if results:
            return results[0][0]
        return None

    def get_supplier_data(self, supplier_id: str) -> Optional[Dict[str, Any]]:
        """
        Get supplier data.

        Args:
            supplier_id: Supplier identifier

        Returns:
            Supplier data dictionary or None
        """
        query = "SELECT * FROM suppliers WHERE supplier_id = ?"
        results = self.execute(query, (supplier_id,))

        if results:
            row = results[0]
            return {
                'id': row[0],
                'supplier_id': row[1],
                'name': row[2],
                'country': row[3],
                'industry': row[4],
                'emissions_data': json.loads(row[5]) if row[5] else None,
                'certification': row[6]
            }
        return None

    def log_audit_event(self, event_type: str, event_data: Any,
                       user_id: Optional[str] = None) -> None:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            event_data: Event data
            user_id: Optional user ID
        """
        query = """
            INSERT INTO audit_log (event_type, event_data, user_id, timestamp)
            VALUES (?, ?, ?, ?)
        """

        event_data_str = json.dumps(event_data) if isinstance(event_data, dict) else str(event_data)
        self.execute(query, (event_type, event_data_str, user_id, datetime.now()))

    def get_metrics(self) -> Dict[str, Any]:
        """Get database metrics."""
        return {
            "query_count": self.query_count,
            "transaction_count": self.transaction_count,
            "pool_size": self.config.pool_size,
            "connections_in_use": len(self.pool.in_use),
            "connections_available": len(self.pool.available)
        }

    def close(self):
        """Close database connection."""
        self.pool.close_all()
        logger.info("Database connection closed")