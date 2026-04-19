"""
Test Suite for PostgreSQL Backend

This module provides comprehensive testing for the PostgreSQL backend implementation,
covering all CRUD operations, transaction handling, and performance scenarios.
"""

import asyncio
import hashlib
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from greenlang.auth.backends.postgresql import (
    PostgreSQLBackend,
    DatabaseConfig,
    DatabaseSession,
    PermissionModel,
    RoleModel,
    PolicyModel,
    AuditLogModel
)
from greenlang.auth.permissions import (
    Permission,
    PermissionEffect,
    PermissionCondition
)
from greenlang.utilities.determinism import deterministic_uuid


class TestDatabaseConfig:
    """Test database configuration."""

    def test_config_creation(self):
        """Test creating database configuration."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.pool_size == 10  # Default

    def test_connection_url_generation(self):
        """Test generating connection URL."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )

        url = config.get_connection_url()
        assert "postgresql://" in url
        assert "test_user" in url
        assert "test_db" in url

    def test_async_connection_url(self):
        """Test async connection URL generation."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            use_async=True
        )

        url = config.get_connection_url()
        assert "postgresql+asyncpg://" in url


class TestPostgreSQLBackend:
    """Test PostgreSQL backend operations."""

    @pytest.fixture
    def backend(self):
        """Create test backend instance."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_greenlang",
            username="test_user",
            password="test_pass"
        )

        # Mock the database session
        with patch('greenlang.auth.backends.postgresql.DatabaseSession'):
            backend = PostgreSQLBackend(config)
            backend.db_session = Mock()
            return backend

    @pytest.fixture
    def sample_permission(self):
        """Create sample permission."""
        return Permission(
            permission_id=str(deterministic_uuid("test", "permission")),
            resource="emissions:data",
            action="read",
            effect=PermissionEffect.ALLOW,
            scope="organization:123",
            conditions=[
                PermissionCondition(
                    field="ip_address",
                    operator="in",
                    value=["192.168.1.0/24"]
                )
            ],
            created_by="test_user"
        )

    def test_create_permission(self, backend, sample_permission):
        """Test creating a permission."""
        # Mock session
        mock_session = MagicMock()
        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        # Execute
        result = backend.create_permission(sample_permission)

        # Verify
        assert result == sample_permission
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called()

    def test_get_permission(self, backend):
        """Test retrieving a permission."""
        # Setup mock
        mock_session = MagicMock()
        mock_permission = MagicMock()
        mock_permission.to_permission.return_value = Permission(
            permission_id="test_id",
            resource="test_resource",
            action="test_action",
            effect=PermissionEffect.ALLOW
        )

        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_permission
        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        # Execute
        result = backend.get_permission("test_id")

        # Verify
        assert result is not None
        assert result.permission_id == "test_id"
        mock_session.query.assert_called()

    def test_update_permission(self, backend, sample_permission):
        """Test updating a permission."""
        # Setup mock
        mock_session = MagicMock()
        mock_db_permission = MagicMock()

        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_permission
        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        # Update permission
        sample_permission.action = "write"

        # Execute
        result = backend.update_permission(sample_permission)

        # Verify
        assert result == sample_permission
        assert mock_db_permission.action == "write"
        mock_session.flush.assert_called()

    def test_delete_permission(self, backend):
        """Test deleting a permission."""
        # Setup mock
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.delete.return_value = 1
        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        # Execute
        result = backend.delete_permission("test_id")

        # Verify
        assert result is True
        mock_session.query.assert_called()

    def test_list_permissions_with_filters(self, backend):
        """Test listing permissions with filters."""
        # Setup mock
        mock_session = MagicMock()
        mock_query = MagicMock()

        mock_permissions = [
            MagicMock(to_permission=lambda: Permission(
                permission_id=f"perm_{i}",
                resource="test_resource",
                action="read",
                effect=PermissionEffect.ALLOW
            )) for i in range(3)
        ]

        mock_query.all.return_value = mock_permissions
        mock_query.offset.return_value.limit.return_value = mock_query
        mock_session.query.return_value = mock_query
        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        # Execute
        result = backend.list_permissions(
            resource_pattern="test_*",
            action_pattern="read",
            limit=10,
            offset=0
        )

        # Verify
        assert len(result) == 3
        mock_query.offset.assert_called_with(0)
        mock_query.limit.assert_called_with(10)

    def test_create_role(self, backend):
        """Test creating a role."""
        # Setup mock
        mock_session = MagicMock()
        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        role_data = {
            "role_name": "test_role",
            "description": "Test role description",
            "permissions": ["perm1", "perm2"]
        }

        # Execute
        result = backend.create_role(role_data)

        # Verify
        assert result["role_name"] == "test_role"
        assert "role_id" in result
        mock_session.add.assert_called_once()

    def test_update_role(self, backend):
        """Test updating a role."""
        # Setup mock
        mock_session = MagicMock()
        mock_db_role = MagicMock()
        mock_db_role.role_id = "test_role_id"

        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_role
        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        role_data = {
            "description": "Updated description",
            "permissions": ["perm3", "perm4"]
        }

        # Execute
        result = backend.update_role("test_role_id", role_data)

        # Verify
        assert result["role_id"] == "test_role_id"
        assert mock_db_role.description == "Updated description"

    def test_create_policy(self, backend):
        """Test creating a policy."""
        # Setup mock
        mock_session = MagicMock()
        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        policy_data = {
            "policy_name": "test_policy",
            "policy_type": "rbac",
            "rules": {
                "allow": ["read", "write"],
                "deny": ["delete"]
            }
        }

        # Execute
        result = backend.create_policy(policy_data)

        # Verify
        assert result["policy_name"] == "test_policy"
        assert "policy_id" in result
        mock_session.add.assert_called_once()

    def test_log_audit_event(self, backend):
        """Test logging an audit event."""
        # Setup mock
        mock_session = MagicMock()
        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        event_data = {
            "event_type": "permission.created",
            "user_id": "test_user",
            "resource": "test_resource",
            "action": "create",
            "result": "success",
            "ip_address": "192.168.1.1"
        }

        # Execute
        log_id = backend.log_audit_event(event_data)

        # Verify
        assert log_id is not None
        mock_session.add.assert_called_once()

    def test_get_audit_logs(self, backend):
        """Test retrieving audit logs."""
        # Setup mock
        mock_session = MagicMock()
        mock_logs = [
            MagicMock(
                log_id=f"log_{i}",
                timestamp=datetime.utcnow(),
                event_type="test_event",
                user_id="test_user",
                result="success"
            ) for i in range(5)
        ]

        mock_query = MagicMock()
        mock_query.all.return_value = mock_logs
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query
        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        # Execute
        result = backend.get_audit_logs(
            user_id="test_user",
            limit=10
        )

        # Verify
        assert len(result) == 5
        mock_session.query.assert_called()

    def test_cleanup_expired_data(self, backend):
        """Test cleaning up expired data."""
        # Setup mock
        mock_session = MagicMock()
        mock_session.execute.return_value.rowcount = 5
        mock_session.query.return_value.filter.return_value.delete.return_value = 10

        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        # Execute
        result = backend.cleanup_expired_data(older_than_days=30)

        # Verify
        assert "expired_role_assignments" in result
        assert "expired_policies" in result
        assert "old_audit_logs" in result
        mock_session.commit.assert_called()

    def test_get_statistics(self, backend):
        """Test getting database statistics."""
        # Setup mock
        mock_session = MagicMock()
        mock_session.query.return_value.scalar.side_effect = [100, 20, 15, 500, 12, 5, 50]

        backend.db_session.get_session.return_value.__enter__.return_value = mock_session
        backend._stats = {
            'operations': 1000,
            'errors': 5,
            'cache_hits': 950
        }

        # Execute
        stats = backend.get_statistics()

        # Verify
        assert stats['permissions_count'] == 100
        assert stats['roles_count'] == 20
        assert stats['policies_count'] == 15
        assert stats['backend_operations'] == 1000
        assert stats['cache_hits'] == 950

    def test_transaction_rollback_on_error(self, backend, sample_permission):
        """Test transaction rollback on error."""
        # Setup mock
        mock_session = MagicMock()
        mock_session.add.side_effect = Exception("Database error")

        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        # Execute and verify exception
        with pytest.raises(Exception):
            backend.create_permission(sample_permission)

    def test_concurrent_operations(self, backend):
        """Test concurrent database operations."""
        # Setup mock
        mock_session = MagicMock()
        backend.db_session.get_session.return_value.__enter__.return_value = mock_session

        async def concurrent_task(perm_id):
            """Simulate concurrent permission retrieval."""
            return backend.get_permission(perm_id)

        # Execute multiple concurrent operations
        loop = asyncio.new_event_loop()
        tasks = [concurrent_task(f"perm_{i}") for i in range(10)]

        # Note: This is a simplified test. Real concurrent testing would require
        # actual async implementation and database connections


class TestPermissionModel:
    """Test Permission SQLAlchemy model."""

    def test_to_permission_conversion(self):
        """Test converting database model to Permission object."""
        db_model = PermissionModel(
            permission_id="test_id",
            resource="test_resource",
            action="test_action",
            effect="allow",
            scope="test_scope",
            conditions=[{"field": "test", "operator": "eq", "value": "value"}],
            created_at=datetime.utcnow(),
            created_by="test_user"
        )

        permission = db_model.to_permission()

        assert permission.permission_id == "test_id"
        assert permission.resource == "test_resource"
        assert permission.action == "test_action"
        assert permission.effect == PermissionEffect.ALLOW
        assert len(permission.conditions) == 1

    def test_from_permission_conversion(self):
        """Test creating database model from Permission object."""
        permission = Permission(
            permission_id="test_id",
            resource="test_resource",
            action="test_action",
            effect=PermissionEffect.DENY,
            conditions=[
                PermissionCondition(
                    field="test",
                    operator="eq",
                    value="value"
                )
            ]
        )

        db_model = PermissionModel.from_permission(permission)

        assert db_model.permission_id == "test_id"
        assert db_model.resource == "test_resource"
        assert db_model.effect == "deny"
        assert db_model.provenance_hash is not None


class TestDatabaseSession:
    """Test database session management."""

    @patch('greenlang.auth.backends.postgresql.create_engine')
    def test_session_creation(self, mock_engine):
        """Test creating database session."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )

        session = DatabaseSession(config)

        mock_engine.assert_called_once()
        assert session.engine is not None

    @patch('greenlang.auth.backends.postgresql.create_engine')
    def test_context_manager(self, mock_engine):
        """Test session context manager."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )

        db_session = DatabaseSession(config)
        mock_session = MagicMock()
        db_session.Session = MagicMock(return_value=mock_session)

        with db_session.get_session() as session:
            assert session == mock_session

        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @patch('greenlang.auth.backends.postgresql.create_engine')
    def test_rollback_on_exception(self, mock_engine):
        """Test session rollback on exception."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )

        db_session = DatabaseSession(config)
        mock_session = MagicMock()
        db_session.Session = MagicMock(return_value=mock_session)

        with pytest.raises(ValueError):
            with db_session.get_session() as session:
                raise ValueError("Test error")

        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()


@pytest.mark.integration
class TestIntegration:
    """Integration tests with real database (requires PostgreSQL)."""

    @pytest.fixture
    def real_backend(self):
        """Create backend with real database connection."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_greenlang",
            username="test_user",
            password="test_pass"
        )

        try:
            backend = PostgreSQLBackend(config)
            # Initialize schema
            backend.db_session.create_tables()
            yield backend
            # Cleanup
            backend.db_session.drop_tables()
            backend.close()
        except Exception as e:
            pytest.skip(f"PostgreSQL not available: {e}")

    @pytest.mark.skipif(not pytest.config.getoption("--integration"),
                        reason="Integration tests require --integration flag")
    def test_full_permission_lifecycle(self, real_backend):
        """Test complete permission lifecycle with real database."""
        # Create permission
        permission = Permission(
            resource="test:resource",
            action="read",
            effect=PermissionEffect.ALLOW
        )

        created = real_backend.create_permission(permission)
        assert created.permission_id is not None

        # Retrieve permission
        retrieved = real_backend.get_permission(created.permission_id)
        assert retrieved.resource == "test:resource"

        # Update permission
        retrieved.action = "write"
        updated = real_backend.update_permission(retrieved)
        assert updated.action == "write"

        # List permissions
        permissions = real_backend.list_permissions(resource_pattern="test:*")
        assert len(permissions) > 0

        # Delete permission
        deleted = real_backend.delete_permission(created.permission_id)
        assert deleted is True

        # Verify deletion
        not_found = real_backend.get_permission(created.permission_id)
        assert not_found is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])