# -*- coding: utf-8 -*-
"""
Pytest fixtures and test factories for Phase 4
Provides comprehensive test data and mocks for all Phase 4 components
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from greenlang.db.base import Base, reset_engine
from greenlang.determinism import deterministic_uuid, DeterministicClock
from greenlang.db.models_auth import (
    User,
    Role,
    Permission,
    UserRole,
    Session as DBSession,
    APIKey,
    AuditLog,
    SAMLProvider,
    OAuthProvider,
    LDAPConfig,
)
from greenlang.auth.rbac import RBACManager, Permission as RBACPermission
from greenlang.cache.redis_config import RedisConfig, RedisSessionStore


# ===== Database Fixtures =====

@pytest.fixture(scope="function")
def db_engine():
    """Create in-memory SQLite engine for testing"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()
    reset_engine()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create database session for testing"""
    SessionLocal = sessionmaker(bind=db_engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


# ===== User and Auth Fixtures =====

@pytest.fixture
def test_user_data():
    """Test user data"""
    return {
        "id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        "tenant_id": "test-tenant-1",
        "username": "testuser",
        "email": "testuser@greenlang.test",
        "password_hash": "$2b$12$test_hash",
        "first_name": "Test",
        "last_name": "User",
        "active": True,
        "email_verified": True,
    }


@pytest.fixture
def test_user(db_session, test_user_data):
    """Create test user in database"""
    user = User(**test_user_data)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_users(db_session):
    """Create multiple test users"""
    users = []
    for i in range(5):
        user = User(
            id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            tenant_id=f"test-tenant-{(i % 2) + 1}",
            username=f"user{i}",
            email=f"user{i}@greenlang.test",
            password_hash="$2b$12$test_hash",
            active=True,
        )
        db_session.add(user)
        users.append(user)

    db_session.commit()
    return users


# ===== Role and Permission Fixtures =====

@pytest.fixture
def test_role_data():
    """Test role data"""
    return {
        "id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        "tenant_id": "test-tenant-1",
        "name": "developer",
        "description": "Developer role",
        "is_system_role": False,
    }


@pytest.fixture
def test_role(db_session, test_role_data):
    """Create test role in database"""
    role = Role(**test_role_data)
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)
    return role


@pytest.fixture
def test_roles(db_session):
    """Create multiple test roles with hierarchy"""
    roles_data = [
        {"name": "super_admin", "is_system_role": True, "tenant_id": None},
        {"name": "admin", "is_system_role": True, "tenant_id": None},
        {"name": "developer", "is_system_role": False, "tenant_id": "test-tenant-1"},
        {"name": "viewer", "is_system_role": False, "tenant_id": "test-tenant-1"},
    ]

    roles = []
    for data in roles_data:
        role = Role(
            id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            **data
        )
        db_session.add(role)
        roles.append(role)

    db_session.commit()
    return roles


@pytest.fixture
def test_permission(db_session, test_role):
    """Create test permission"""
    permission = Permission(
        id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        role_id=test_role.id,
        resource_type="pipeline",
        action="execute",
        effect="allow",
    )
    db_session.add(permission)
    db_session.commit()
    db_session.refresh(permission)
    return permission


@pytest.fixture
def test_permissions(db_session, test_roles):
    """Create multiple test permissions"""
    permissions = []
    perm_configs = [
        {"resource_type": "*", "action": "*", "role_idx": 0},  # super_admin
        {"resource_type": "pipeline", "action": "*", "role_idx": 2},  # developer
        {"resource_type": "pack", "action": "*", "role_idx": 2},
        {"resource_type": "*", "action": "read", "role_idx": 3},  # viewer
    ]

    for config in perm_configs:
        role_idx = config.pop("role_idx")
        permission = Permission(
            id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            role_id=test_roles[role_idx].id,
            **config,
            effect="allow",
        )
        db_session.add(permission)
        permissions.append(permission)

    db_session.commit()
    return permissions


# ===== Session Fixtures =====

@pytest.fixture
def test_session_data():
    """Test session data"""
    return {
        "id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        "user_id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        "tenant_id": "test-tenant-1",
        "session_token": deterministic_uuid(__name__, str(DeterministicClock.now())).hex,
        "ip_address": "192.168.1.100",
        "user_agent": "pytest/1.0",
        "expires_at": DeterministicClock.utcnow() + timedelta(hours=24),
        "active": True,
    }


@pytest.fixture
def test_db_session(db_session, test_user, test_session_data):
    """Create test session in database"""
    test_session_data["user_id"] = test_user.id
    session = DBSession(**test_session_data)
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session


# ===== API Key Fixtures =====

@pytest.fixture
def test_api_key(db_session, test_user):
    """Create test API key"""
    api_key = APIKey(
        id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        user_id=test_user.id,
        tenant_id=test_user.tenant_id,
        key_id=f"glk_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:16]}",
        key_hash="hashed_secret",
        key_prefix="glk_test",
        name="Test API Key",
        active=True,
    )
    db_session.add(api_key)
    db_session.commit()
    db_session.refresh(api_key)
    return api_key


# ===== SSO Provider Fixtures =====

@pytest.fixture
def test_saml_provider(db_session):
    """Create test SAML provider"""
    provider = SAMLProvider(
        id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        tenant_id="test-tenant-1",
        name="Test SAML IdP",
        entity_id="https://idp.greenlang.test/saml",
        sso_url="https://idp.greenlang.test/sso",
        x509_cert="-----BEGIN CERTIFICATE-----\ntest_cert\n-----END CERTIFICATE-----",
        attribute_mapping={
            "email": "mail",
            "first_name": "givenName",
            "last_name": "sn"
        },
        active=True,
    )
    db_session.add(provider)
    db_session.commit()
    db_session.refresh(provider)
    return provider


@pytest.fixture
def test_oauth_provider(db_session):
    """Create test OAuth provider"""
    provider = OAuthProvider(
        id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        tenant_id="test-tenant-1",
        name="Test OAuth Provider",
        provider_type="google",
        client_id="test_client_id",
        client_secret="encrypted_secret",
        authorization_url="https://oauth.greenlang.test/authorize",
        token_url="https://oauth.greenlang.test/token",
        userinfo_url="https://oauth.greenlang.test/userinfo",
        scopes=["openid", "email", "profile"],
        active=True,
    )
    db_session.add(provider)
    db_session.commit()
    db_session.refresh(provider)
    return provider


@pytest.fixture
def test_ldap_config(db_session):
    """Create test LDAP configuration"""
    config = LDAPConfig(
        id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        tenant_id="test-tenant-1",
        name="Test LDAP",
        server_url="ldap://ldap.greenlang.test",
        port=389,
        base_dn="dc=greenlang,dc=test",
        user_search_base="ou=users,dc=greenlang,dc=test",
        user_search_filter="(uid={username})",
        attribute_mapping={
            "email": "mail",
            "first_name": "givenName",
            "last_name": "sn"
        },
        active=True,
    )
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)
    return config


# ===== RBAC Manager Fixtures =====

@pytest.fixture
def rbac_manager():
    """Create RBAC manager"""
    return RBACManager()


@pytest.fixture
def rbac_manager_with_users(rbac_manager):
    """RBAC manager with test users and roles"""
    # Assign roles to users
    rbac_manager.assign_role("user-1", "developer")
    rbac_manager.assign_role("user-2", "viewer")
    rbac_manager.assign_role("user-3", "admin")
    return rbac_manager


# ===== Redis Fixtures =====

@pytest.fixture
def mock_redis_client():
    """Mock Redis client"""
    client = Mock()
    client.ping.return_value = True
    client.get.return_value = None
    client.set.return_value = True
    client.setex.return_value = True
    client.delete.return_value = 1
    client.exists.return_value = 1
    client.expire.return_value = True
    client.ttl.return_value = 3600
    client.keys.return_value = []
    return client


@pytest.fixture
def mock_redis_config(mock_redis_client):
    """Mock Redis configuration"""
    with patch('greenlang.cache.redis_config.REDIS_AVAILABLE', True):
        config = RedisConfig()
        config._client = mock_redis_client
        yield config


@pytest.fixture
def mock_redis_session_store(mock_redis_config):
    """Mock Redis session store"""
    return RedisSessionStore(mock_redis_config)


# ===== SSO Mock Fixtures =====

@pytest.fixture
def mock_saml_response():
    """Mock SAML response"""
    return {
        "attributes": {
            "mail": ["testuser@greenlang.test"],
            "givenName": ["Test"],
            "sn": ["User"],
            "uid": ["testuser"],
        },
        "name_id": "testuser@greenlang.test",
        "session_index": "session_123",
    }


@pytest.fixture
def mock_oauth_token_response():
    """Mock OAuth token response"""
    return {
        "access_token": "mock_access_token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": "mock_refresh_token",
        "scope": "openid email profile",
    }


@pytest.fixture
def mock_oauth_userinfo():
    """Mock OAuth userinfo response"""
    return {
        "sub": "12345",
        "email": "testuser@greenlang.test",
        "email_verified": True,
        "name": "Test User",
        "given_name": "Test",
        "family_name": "User",
    }


@pytest.fixture
def mock_ldap_connection():
    """Mock LDAP connection"""
    conn = Mock()
    conn.bind.return_value = True
    conn.search.return_value = True
    conn.entries = [
        Mock(
            entry_dn="uid=testuser,ou=users,dc=greenlang,dc=test",
            mail=Mock(value="testuser@greenlang.test"),
            givenName=Mock(value="Test"),
            sn=Mock(value="User"),
            uid=Mock(value="testuser"),
        )
    ]
    return conn


# ===== GraphQL Mock Fixtures =====

@pytest.fixture
def mock_graphql_schema():
    """Mock GraphQL schema"""
    from unittest.mock import MagicMock

    schema = MagicMock()
    schema.type_map = {
        "User": MagicMock(fields={"id": MagicMock(), "username": MagicMock(), "email": MagicMock()}),
        "Role": MagicMock(fields={"id": MagicMock(), "name": MagicMock()}),
        "Query": MagicMock(),
        "Mutation": MagicMock(),
        "Subscription": MagicMock(),
    }
    return schema


@pytest.fixture
def mock_graphql_context():
    """Mock GraphQL context"""
    return {
        "user_id": "test-user-id",
        "tenant_id": "test-tenant-1",
        "session_id": "test-session-id",
        "request": Mock(headers={"User-Agent": "pytest/1.0"}),
    }


# ===== Performance Test Fixtures =====

@pytest.fixture
def performance_metrics():
    """Performance metrics collector"""
    class Metrics:
        def __init__(self):
            self.measurements = []

        def record(self, operation: str, duration: float, **metadata):
            self.measurements.append({
                "operation": operation,
                "duration_ms": duration * 1000,
                "timestamp": DeterministicClock.utcnow(),
                **metadata
            })

        def get_stats(self, operation: Optional[str] = None):
            data = self.measurements
            if operation:
                data = [m for m in data if m["operation"] == operation]

            if not data:
                return None

            durations = [m["duration_ms"] for m in data]
            return {
                "count": len(durations),
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "p50": sorted(durations)[len(durations) // 2],
                "p95": sorted(durations)[int(len(durations) * 0.95)],
                "p99": sorted(durations)[int(len(durations) * 0.99)],
            }

    return Metrics()


# ===== Assertion Helpers =====

@pytest.fixture
def assert_audit_log_created(db_session):
    """Helper to assert audit log was created"""
    def _assert(event_type: str, result: str = "success"):
        log = db_session.query(AuditLog).filter_by(
            event_type=event_type, result=result
        ).first()
        assert log is not None, f"Audit log not found for event: {event_type}"
        return log

    return _assert


@pytest.fixture
def assert_permission_check():
    """Helper to assert permission check"""
    def _assert(rbac_manager: RBACManager, user_id: str, resource: str, action: str, expected: bool):
        result = rbac_manager.check_permission(user_id, resource, action)
        assert result == expected, f"Permission check failed: user={user_id}, resource={resource}, action={action}"

    return _assert


# ===== Factories =====

class UserFactory:
    """Factory for creating test users"""

    @staticmethod
    def create(session: Session, **kwargs) -> User:
        """Create user with defaults"""
        data = {
            "id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            "tenant_id": "test-tenant-1",
            "username": f"user_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}",
            "email": f"user_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}@greenlang.test",
            "password_hash": "$2b$12$test_hash",
            "active": True,
        }
        data.update(kwargs)

        user = User(**data)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

    @staticmethod
    def create_batch(session: Session, count: int, **kwargs) -> List[User]:
        """Create multiple users"""
        return [UserFactory.create(session, **kwargs) for _ in range(count)]


class RoleFactory:
    """Factory for creating test roles"""

    @staticmethod
    def create(session: Session, **kwargs) -> Role:
        """Create role with defaults"""
        data = {
            "id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            "tenant_id": "test-tenant-1",
            "name": f"role_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}",
            "is_system_role": False,
        }
        data.update(kwargs)

        role = Role(**data)
        session.add(role)
        session.commit()
        session.refresh(role)
        return role


@pytest.fixture
def user_factory():
    """User factory fixture"""
    return UserFactory


@pytest.fixture
def role_factory():
    """Role factory fixture"""
    return RoleFactory
