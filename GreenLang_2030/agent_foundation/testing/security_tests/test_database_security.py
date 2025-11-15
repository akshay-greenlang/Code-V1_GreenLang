"""
Tests for secure database operations.

Tests SQL injection prevention in:
- Query building
- Parameter binding
- Field name validation
- Operator validation
"""

import pytest
from typing import Dict, List

from database.postgres_manager_secure import (
    SecureQueryBuilder,
    SecureAggregationBuilder,
    SecurePostgresOperations,
)
from security.input_validation import SafeQueryInput


class TestSecureQueryBuilder:
    """Test secure query builder."""

    def test_select_query_with_filters(self):
        """Test SELECT query building with filters."""
        builder = SecureQueryBuilder("agents")

        filters = [
            SafeQueryInput(field="tenant_id", value="tenant-123", operator="="),
            SafeQueryInput(field="status", value="active", operator="="),
        ]

        query, params = builder.build_select(
            filters=filters,
            limit=100,
            offset=0,
            sort_by="created_at",
            sort_direction="DESC"
        )

        # Verify query structure
        assert "SELECT * FROM agents" in query
        assert "WHERE tenant_id = $1 AND status = $2" in query
        assert "ORDER BY created_at DESC" in query
        assert "LIMIT 100 OFFSET 0" in query

        # Verify parameters
        assert params == ["tenant-123", "active"]

    def test_select_query_with_columns(self):
        """Test SELECT with specific columns."""
        builder = SecureQueryBuilder("agents")

        query, params = builder.build_select(
            filters=[],
            columns=["tenant_id", "name", "status"],
            limit=10,
            offset=0
        )

        assert "SELECT tenant_id, name, status FROM agents" in query

    def test_select_query_invalid_table(self):
        """Test invalid table name rejected."""
        with pytest.raises(ValueError, match="whitelist"):
            SecureQueryBuilder("malicious_table")

    def test_select_query_invalid_column(self):
        """Test invalid column name rejected."""
        builder = SecureQueryBuilder("agents")

        with pytest.raises(ValueError, match="whitelist"):
            builder.build_select(
                filters=[],
                columns=["tenant_id", "password"],  # password not in whitelist
                limit=10,
                offset=0
            )

    def test_insert_query_valid(self):
        """Test INSERT query building."""
        builder = SecureQueryBuilder("agents")

        data = {
            "tenant_id": "tenant-123",
            "name": "TestAgent",
            "status": "active",
        }

        query, params = builder.build_insert(data)

        assert "INSERT INTO agents" in query
        assert "tenant_id, name, status" in query
        assert "VALUES ($1, $2, $3)" in query
        assert "RETURNING *" in query
        assert params == ["tenant-123", "TestAgent", "active"]

    def test_insert_query_sql_injection_in_value(self):
        """Test SQL injection in value blocked."""
        builder = SecureQueryBuilder("agents")

        data = {
            "tenant_id": "tenant'; DROP TABLE agents--",
            "name": "TestAgent",
        }

        with pytest.raises(ValueError, match="SQL"):
            builder.build_insert(data)

    def test_insert_query_invalid_field(self):
        """Test invalid field name rejected."""
        builder = SecureQueryBuilder("agents")

        data = {
            "tenant_id": "tenant-123",
            "malicious_field": "value",  # Not in whitelist
        }

        with pytest.raises(ValueError, match="whitelist"):
            builder.build_insert(data)

    def test_update_query_valid(self):
        """Test UPDATE query building."""
        builder = SecureQueryBuilder("agents")

        data = {"status": "inactive"}
        filters = [SafeQueryInput(field="agent_id", value="agent-123", operator="=")]

        query, params = builder.build_update(data, filters)

        assert "UPDATE agents SET status = $1" in query
        assert "WHERE agent_id = $2" in query
        assert "RETURNING *" in query
        assert params == ["inactive", "agent-123"]

    def test_update_query_requires_where(self):
        """Test UPDATE requires WHERE clause."""
        builder = SecureQueryBuilder("agents")

        data = {"status": "inactive"}

        with pytest.raises(ValueError, match="WHERE clause"):
            builder.build_update(data, filters=[])

    def test_update_query_sql_injection(self):
        """Test SQL injection blocked in UPDATE."""
        builder = SecureQueryBuilder("agents")

        data = {"status": "active'; DROP TABLE agents--"}
        filters = [SafeQueryInput(field="agent_id", value="agent-123", operator="=")]

        with pytest.raises(ValueError, match="SQL"):
            builder.build_update(data, filters)

    def test_delete_query_valid(self):
        """Test DELETE query building."""
        builder = SecureQueryBuilder("agents")

        filters = [SafeQueryInput(field="agent_id", value="agent-123", operator="=")]

        query, params = builder.build_delete(filters)

        assert "DELETE FROM agents" in query
        assert "WHERE agent_id = $1" in query
        assert "RETURNING *" in query
        assert params == ["agent-123"]

    def test_delete_query_requires_where(self):
        """Test DELETE requires WHERE clause."""
        builder = SecureQueryBuilder("agents")

        with pytest.raises(ValueError, match="WHERE clause"):
            builder.build_delete(filters=[])

    def test_parameterized_queries_prevent_injection(self):
        """Test parameterized queries prevent SQL injection."""
        builder = SecureQueryBuilder("agents")

        # Even if validation is bypassed, parameterization prevents injection
        filters = [
            SafeQueryInput(field="tenant_id", value="safe-value", operator="=")
        ]

        query, params = builder.build_select(filters=filters, limit=10, offset=0)

        # Query uses placeholders ($1, $2, etc)
        assert "$1" in query
        assert "'" not in query  # No quotes in query string
        assert params == ["safe-value"]


class TestSecureAggregationBuilder:
    """Test secure aggregation queries."""

    def test_count_without_grouping(self):
        """Test COUNT query without GROUP BY."""
        builder = SecureAggregationBuilder("executions")

        filters = [SafeQueryInput(field="tenant_id", value="tenant-123", operator="=")]

        query, params = builder.count(group_by=None, filters=filters)

        assert "SELECT COUNT(*) as count FROM executions" in query
        assert "WHERE tenant_id = $1" in query
        assert "GROUP BY" not in query
        assert params == ["tenant-123"]

    def test_count_with_grouping(self):
        """Test COUNT query with GROUP BY."""
        builder = SecureAggregationBuilder("executions")

        filters = [SafeQueryInput(field="tenant_id", value="tenant-123", operator="=")]

        query, params = builder.count(
            group_by=["status"],
            filters=filters
        )

        assert "SELECT status, COUNT(*) as count FROM executions" in query
        assert "WHERE tenant_id = $1" in query
        assert "GROUP BY status" in query
        assert params == ["tenant-123"]

    def test_count_invalid_group_by_field(self):
        """Test invalid GROUP BY field rejected."""
        builder = SecureAggregationBuilder("executions")

        with pytest.raises(ValueError, match="whitelist"):
            builder.count(group_by=["invalid_field"], filters=[])


class TestOperatorValidation:
    """Test SQL operator validation."""

    def test_safe_operators_allowed(self):
        """Test safe operators are allowed."""
        safe_operators = ["=", "!=", ">", "<", ">=", "<=", "IN", "LIKE"]

        for op in safe_operators:
            filter_input = SafeQueryInput(
                field="tenant_id",
                value="tenant-123",
                operator=op
            )
            assert filter_input.operator == op.upper()

    def test_dangerous_operators_blocked(self):
        """Test dangerous operators are blocked."""
        dangerous_operators = [
            "OR",
            "AND",
            "UNION",
            "--",
            ";",
            "/*",
        ]

        for op in dangerous_operators:
            with pytest.raises(ValueError, match="Operator"):
                SafeQueryInput(
                    field="tenant_id",
                    value="tenant-123",
                    operator=op
                )

    def test_operator_case_insensitive(self):
        """Test operator validation is case-insensitive."""
        filter_input = SafeQueryInput(
            field="tenant_id",
            value="tenant-123",
            operator="like"  # lowercase
        )
        assert filter_input.operator == "LIKE"  # converted to uppercase


class TestComplexQueries:
    """Test complex query scenarios."""

    def test_multiple_filters(self):
        """Test query with multiple filters."""
        builder = SecureQueryBuilder("agents")

        filters = [
            SafeQueryInput(field="tenant_id", value="tenant-123", operator="="),
            SafeQueryInput(field="status", value="active", operator="="),
            SafeQueryInput(field="type", value="worker", operator="="),
        ]

        query, params = builder.build_select(filters=filters, limit=100, offset=0)

        assert query.count("$") == 3  # Three parameters
        assert "WHERE tenant_id = $1 AND status = $2 AND type = $3" in query
        assert params == ["tenant-123", "active", "worker"]

    def test_pagination(self):
        """Test pagination parameters."""
        builder = SecureQueryBuilder("agents")

        query, params = builder.build_select(
            filters=[],
            limit=50,
            offset=100
        )

        assert "LIMIT 50 OFFSET 100" in query

    def test_pagination_invalid_limit(self):
        """Test invalid limit rejected."""
        builder = SecureQueryBuilder("agents")

        with pytest.raises(ValueError, match="limit"):
            builder.build_select(filters=[], limit=2000, offset=0)  # Too large

    def test_pagination_invalid_offset(self):
        """Test invalid offset rejected."""
        builder = SecureQueryBuilder("agents")

        with pytest.raises(ValueError, match="offset"):
            builder.build_select(filters=[], limit=10, offset=-1)  # Negative


class TestInjectionVectors:
    """Test various SQL injection vectors are blocked."""

    def test_boolean_based_injection(self):
        """Test boolean-based SQL injection blocked."""
        injections = [
            "' OR '1'='1",
            "' OR 1=1--",
            "admin' --",
        ]

        for injection in injections:
            with pytest.raises(ValueError):
                SafeQueryInput(
                    field="tenant_id",
                    value=injection,
                    operator="="
                )

    def test_union_based_injection(self):
        """Test UNION-based SQL injection blocked."""
        injections = [
            "' UNION SELECT * FROM users--",
            "1 UNION SELECT null,null,null--",
        ]

        for injection in injections:
            with pytest.raises(ValueError):
                SafeQueryInput(
                    field="tenant_id",
                    value=injection,
                    operator="="
                )

    def test_time_based_injection(self):
        """Test time-based SQL injection blocked."""
        injections = [
            "'; WAITFOR DELAY '00:00:05'--",
            "'; SELECT SLEEP(5)--",
        ]

        for injection in injections:
            with pytest.raises(ValueError):
                SafeQueryInput(
                    field="tenant_id",
                    value=injection,
                    operator="="
                )

    def test_stacked_queries_injection(self):
        """Test stacked queries injection blocked."""
        injections = [
            "'; DROP TABLE agents; --",
            "'; DELETE FROM users; --",
            "'; INSERT INTO admins VALUES ('hacker'); --",
        ]

        for injection in injections:
            with pytest.raises(ValueError):
                SafeQueryInput(
                    field="tenant_id",
                    value=injection,
                    operator="="
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
