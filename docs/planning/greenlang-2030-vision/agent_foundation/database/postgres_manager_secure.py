# -*- coding: utf-8 -*-
"""
Secure PostgreSQL Manager - Enhanced with Input Validation.

This module extends PostgresManager with comprehensive input validation
to prevent SQL injection attacks.

Example:
    >>> from database.postgres_manager_secure import SecurePostgresManager
    >>> manager = SecurePostgresManager(config)
    >>> result = await manager.safe_query("tenant_id", "tenant-123")
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field
import logging
from datetime import datetime

from security.input_validation import InputValidator, SafeQueryInput, FilterInput

logger = logging.getLogger(__name__)


class SecureQueryBuilder:
    """
    Secure query builder with input validation.

    All field names are validated against whitelist.
    All values are passed as parameters (no string interpolation).

    Example:
        >>> builder = SecureQueryBuilder("agents")
        >>> query, params = builder.select(["tenant_id", "tenant-123"])
        >>> # Returns: ("SELECT * FROM agents WHERE tenant_id = $1", ["tenant-123"])
    """

    def __init__(self, table_name: str):
        """
        Initialize secure query builder.

        Args:
            table_name: Database table name (validated against whitelist)
        """
        # Whitelist of allowed tables
        ALLOWED_TABLES = {
            'agents', 'executions', 'tasks', 'workflows', 'tenants', 'users',
            'audit_logs', 'metrics', 'events', 'configurations'
        }

        if table_name not in ALLOWED_TABLES:
            raise ValueError(f"Table '{table_name}' not in whitelist: {ALLOWED_TABLES}")

        self.table_name = table_name
        self.validator = InputValidator()

    def build_select(
        self,
        filters: List[SafeQueryInput],
        columns: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: Optional[str] = None,
        sort_direction: str = "DESC"
    ) -> Tuple[str, List[Any]]:
        """
        Build safe SELECT query with parameterized values.

        Args:
            filters: List of validated filter conditions
            columns: Columns to select (None = all)
            limit: Maximum rows to return
            offset: Number of rows to skip
            sort_by: Column to sort by
            sort_direction: Sort direction (ASC/DESC)

        Returns:
            Tuple of (query_string, parameters)

        Example:
            >>> builder = SecureQueryBuilder("agents")
            >>> filters = [SafeQueryInput(field="tenant_id", value="tenant-123", operator="=")]
            >>> query, params = builder.build_select(filters)
        """
        # Validate columns
        if columns:
            columns = [self.validator.validate_field_name(col) for col in columns]
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"

        # Build WHERE clause from validated filters
        where_conditions = []
        params = []
        param_idx = 1

        for filter_input in filters:
            # field and operator already validated by SafeQueryInput
            where_conditions.append(f"{filter_input.field} {filter_input.operator} ${param_idx}")
            params.append(filter_input.value)
            param_idx += 1

        # Build query
        query_parts = [f"SELECT {select_clause} FROM {self.table_name}"]

        if where_conditions:
            query_parts.append(f"WHERE {' AND '.join(where_conditions)}")

        # Add ORDER BY
        if sort_by:
            validated_sort = self.validator.validate_field_name(sort_by)
            validated_direction = self.validator.validate_sort_direction(sort_direction)
            query_parts.append(f"ORDER BY {validated_sort} {validated_direction}")

        # Add LIMIT and OFFSET
        validated_limit = self.validator.validate_integer(limit, "limit", min_value=1, max_value=1000)
        validated_offset = self.validator.validate_integer(offset, "offset", min_value=0)

        query_parts.append(f"LIMIT {validated_limit} OFFSET {validated_offset}")

        query = " ".join(query_parts)

        logger.info(
            f"Built safe SELECT query",
            extra={
                "table": self.table_name,
                "filters": len(filters),
                "query": query[:200]
            }
        )

        return query, params

    def build_insert(self, data: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """
        Build safe INSERT query with parameterized values.

        Args:
            data: Dictionary of field->value pairs

        Returns:
            Tuple of (query_string, parameters)

        Example:
            >>> builder = SecureQueryBuilder("agents")
            >>> query, params = builder.build_insert({
            ...     "tenant_id": "tenant-123",
            ...     "name": "TestAgent",
            ...     "status": "active"
            ... })
        """
        if not data:
            raise ValueError("Insert data cannot be empty")

        # Validate all field names
        validated_fields = []
        values = []

        for field, value in data.items():
            validated_field = self.validator.validate_field_name(field)
            validated_fields.append(validated_field)

            # Validate string values for SQL injection
            if isinstance(value, str):
                self.validator.validate_no_sql_injection(value, field)

            values.append(value)

        # Build parameterized query
        placeholders = ", ".join([f"${i+1}" for i in range(len(values))])
        fields_str = ", ".join(validated_fields)

        query = f"INSERT INTO {self.table_name} ({fields_str}) VALUES ({placeholders}) RETURNING *"

        logger.info(
            f"Built safe INSERT query",
            extra={
                "table": self.table_name,
                "fields": len(validated_fields)
            }
        )

        return query, values

    def build_update(
        self,
        data: Dict[str, Any],
        filters: List[SafeQueryInput]
    ) -> Tuple[str, List[Any]]:
        """
        Build safe UPDATE query with parameterized values.

        Args:
            data: Dictionary of field->value pairs to update
            filters: List of validated filter conditions for WHERE clause

        Returns:
            Tuple of (query_string, parameters)

        Example:
            >>> builder = SecureQueryBuilder("agents")
            >>> filters = [SafeQueryInput(field="agent_id", value="agent-123")]
            >>> query, params = builder.build_update(
            ...     {"status": "inactive"},
            ...     filters
            ... )
        """
        if not data:
            raise ValueError("Update data cannot be empty")

        if not filters:
            raise ValueError("UPDATE requires WHERE clause (filters cannot be empty)")

        # Validate all field names and build SET clause
        set_clauses = []
        params = []
        param_idx = 1

        for field, value in data.items():
            validated_field = self.validator.validate_field_name(field)

            # Validate string values
            if isinstance(value, str):
                self.validator.validate_no_sql_injection(value, field)

            set_clauses.append(f"{validated_field} = ${param_idx}")
            params.append(value)
            param_idx += 1

        # Build WHERE clause
        where_conditions = []
        for filter_input in filters:
            where_conditions.append(f"{filter_input.field} {filter_input.operator} ${param_idx}")
            params.append(filter_input.value)
            param_idx += 1

        # Build query
        query = f"UPDATE {self.table_name} SET {', '.join(set_clauses)} WHERE {' AND '.join(where_conditions)} RETURNING *"

        logger.info(
            f"Built safe UPDATE query",
            extra={
                "table": self.table_name,
                "set_fields": len(set_clauses),
                "where_conditions": len(where_conditions)
            }
        )

        return query, params

    def build_delete(self, filters: List[SafeQueryInput]) -> Tuple[str, List[Any]]:
        """
        Build safe DELETE query with parameterized values.

        Args:
            filters: List of validated filter conditions for WHERE clause

        Returns:
            Tuple of (query_string, parameters)

        Example:
            >>> builder = SecureQueryBuilder("agents")
            >>> filters = [SafeQueryInput(field="agent_id", value="agent-123")]
            >>> query, params = builder.build_delete(filters)
        """
        if not filters:
            raise ValueError("DELETE requires WHERE clause (filters cannot be empty)")

        # Build WHERE clause
        where_conditions = []
        params = []
        param_idx = 1

        for filter_input in filters:
            where_conditions.append(f"{filter_input.field} {filter_input.operator} ${param_idx}")
            params.append(filter_input.value)
            param_idx += 1

        query = f"DELETE FROM {self.table_name} WHERE {' AND '.join(where_conditions)} RETURNING *"

        logger.warning(
            f"Built DELETE query",
            extra={
                "table": self.table_name,
                "where_conditions": len(where_conditions)
            }
        )

        return query, params


class SecureAggregationBuilder:
    """
    Build safe aggregation queries.

    Example:
        >>> builder = SecureAggregationBuilder("executions")
        >>> query, params = builder.count("tenant_id", "tenant-123")
    """

    def __init__(self, table_name: str):
        """Initialize aggregation builder."""
        self.query_builder = SecureQueryBuilder(table_name)
        self.validator = InputValidator()

    def count(
        self,
        group_by: Optional[List[str]] = None,
        filters: Optional[List[SafeQueryInput]] = None
    ) -> Tuple[str, List[Any]]:
        """
        Build COUNT aggregation query.

        Args:
            group_by: Fields to group by
            filters: Filter conditions

        Returns:
            Tuple of (query_string, parameters)
        """
        filters = filters or []

        # Validate group_by fields
        if group_by:
            group_by = [self.validator.validate_field_name(field) for field in group_by]
            select_clause = f"{', '.join(group_by)}, COUNT(*) as count"
            group_clause = f"GROUP BY {', '.join(group_by)}"
        else:
            select_clause = "COUNT(*) as count"
            group_clause = ""

        # Build WHERE clause
        where_conditions = []
        params = []
        param_idx = 1

        for filter_input in filters:
            where_conditions.append(f"{filter_input.field} {filter_input.operator} ${param_idx}")
            params.append(filter_input.value)
            param_idx += 1

        # Build query
        query_parts = [f"SELECT {select_clause} FROM {self.query_builder.table_name}"]

        if where_conditions:
            query_parts.append(f"WHERE {' AND '.join(where_conditions)}")

        if group_clause:
            query_parts.append(group_clause)

        query = " ".join(query_parts)

        return query, params


# Example usage and integration with existing PostgresManager


class SecurePostgresOperations:
    """
    High-level secure database operations.

    This class provides a secure interface for common database operations
    with built-in input validation.

    Example:
        >>> ops = SecurePostgresOperations(postgres_manager)
        >>> await ops.find_by_tenant("tenant-123")
    """

    def __init__(self, table_name: str):
        """
        Initialize secure operations.

        Args:
            table_name: Database table name
        """
        self.query_builder = SecureQueryBuilder(table_name)
        self.validator = InputValidator()

    async def find_by_id(
        self,
        conn: Any,
        id_field: str,
        id_value: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find record by ID with validation.

        Args:
            conn: Database connection
            id_field: ID field name (validated)
            id_value: ID value (validated)

        Returns:
            Record dict or None
        """
        # Validate inputs
        validated_field = self.validator.validate_field_name(id_field)
        validated_value = self.validator.validate_uuid(id_value, id_field)

        # Build query
        filter_input = SafeQueryInput(
            field=validated_field,
            value=validated_value,
            operator="="
        )

        query, params = self.query_builder.build_select([filter_input], limit=1, offset=0)

        # Execute
        result = await conn.fetchrow(query, *params)

        return dict(result) if result else None

    async def find_by_tenant(
        self,
        conn: Any,
        tenant_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Find records by tenant ID.

        Args:
            conn: Database connection
            tenant_id: Tenant ID (validated)
            limit: Maximum records
            offset: Skip records

        Returns:
            List of record dicts
        """
        # Validate tenant_id
        validated_tenant = self.validator.validate_alphanumeric(tenant_id, "tenant_id")

        # Build query
        filter_input = SafeQueryInput(
            field="tenant_id",
            value=validated_tenant,
            operator="="
        )

        query, params = self.query_builder.build_select(
            [filter_input],
            limit=limit,
            offset=offset,
            sort_by="created_at",
            sort_direction="DESC"
        )

        # Execute
        results = await conn.fetch(query, *params)

        return [dict(row) for row in results]

    async def create(
        self,
        conn: Any,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create record with validation.

        Args:
            conn: Database connection
            data: Record data

        Returns:
            Created record dict
        """
        # Build query
        query, params = self.query_builder.build_insert(data)

        # Execute
        result = await conn.fetchrow(query, *params)

        return dict(result)

    async def update_by_id(
        self,
        conn: Any,
        id_field: str,
        id_value: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update record by ID.

        Args:
            conn: Database connection
            id_field: ID field name
            id_value: ID value
            data: Fields to update

        Returns:
            Updated record dict or None
        """
        # Validate ID
        validated_field = self.validator.validate_field_name(id_field)
        validated_value = self.validator.validate_uuid(id_value, id_field)

        # Build filters
        filter_input = SafeQueryInput(
            field=validated_field,
            value=validated_value,
            operator="="
        )

        # Build query
        query, params = self.query_builder.build_update(data, [filter_input])

        # Execute
        result = await conn.fetchrow(query, *params)

        return dict(result) if result else None

    async def delete_by_id(
        self,
        conn: Any,
        id_field: str,
        id_value: str
    ) -> bool:
        """
        Delete record by ID.

        Args:
            conn: Database connection
            id_field: ID field name
            id_value: ID value

        Returns:
            True if deleted, False otherwise
        """
        # Validate ID
        validated_field = self.validator.validate_field_name(id_field)
        validated_value = self.validator.validate_uuid(id_value, id_field)

        # Build filters
        filter_input = SafeQueryInput(
            field=validated_field,
            value=validated_value,
            operator="="
        )

        # Build query
        query, params = self.query_builder.build_delete([filter_input])

        # Execute
        result = await conn.fetchrow(query, *params)

        return result is not None
