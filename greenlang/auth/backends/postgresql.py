"""
PostgreSQLBackend - Production-grade database backend for GreenLang authentication

This module implements a robust PostgreSQL backend for the GreenLang authentication system,
providing persistent storage for permissions, roles, policies, and audit logs with
full ACID compliance, connection pooling, and performance optimization.

Example:
    >>> backend = PostgreSQLBackend(config)
    >>> permission = backend.create_permission(perm_data)
    >>> audit_log = backend.log_audit_event(event_data)
"""

from typing import Dict, List, Optional, Any, Type
from datetime import datetime, timedelta
import hashlib
import json
import logging
from contextlib import contextmanager
from enum import Enum

from greenlang.utilities.determinism import deterministic_uuid

from sqlalchemy import (
    create_engine, Column, String, Text, Integer, DateTime, Boolean,
    Float, ForeignKey, Index, Table, JSON, UniqueConstraint, CheckConstraint,
    and_, or_, func, select, delete, update
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    sessionmaker, Session, relationship, scoped_session,
    selectinload, joinedload
)
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from pydantic import BaseModel, Field, validator
import fnmatch

from greenlang.auth.permissions import (
    Permission, PermissionCondition, PermissionEffect,
    PermissionAction, ResourceType
)
from greenlang.utilities.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()


# ==============================================================================
# Database Models
# ==============================================================================

class PermissionModel(Base):
    """SQLAlchemy model for permissions."""

    __tablename__ = 'permissions'

    permission_id = Column(String(64), primary_key=True)
    resource = Column(String(255), nullable=False, index=True)
    action = Column(String(100), nullable=False, index=True)
    effect = Column(String(10), nullable=False, default='allow')
    scope = Column(String(255), nullable=True, index=True)
    conditions = Column(JSONB, nullable=True)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), nullable=True)
    provenance_hash = Column(String(64), nullable=True)

    # Indexes for performance
    __table_args__ = (
        Index('idx_resource_action', 'resource', 'action'),
        Index('idx_effect_scope', 'effect', 'scope'),
        Index('idx_created_at', 'created_at'),
        Index('idx_provenance', 'provenance_hash'),
        CheckConstraint("effect IN ('allow', 'deny')", name='check_effect')
    )

    def to_permission(self) -> Permission:
        """Convert database model to Permission object."""
        conditions = []
        if self.conditions:
            conditions = [PermissionCondition(**c) for c in self.conditions]

        return Permission(
            permission_id=self.permission_id,
            resource=self.resource,
            action=self.action,
            effect=PermissionEffect(self.effect),
            conditions=conditions,
            scope=self.scope,
            metadata=self.metadata or {},
            created_at=self.created_at,
            created_by=self.created_by
        )

    @classmethod
    def from_permission(cls, permission: Permission) -> 'PermissionModel':
        """Create database model from Permission object."""
        conditions = None
        if permission.conditions:
            conditions = [c.dict() for c in permission.conditions]

        # Calculate provenance hash
        provenance_str = f"{permission.resource}{permission.action}{permission.effect.value}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        return cls(
            permission_id=permission.permission_id,
            resource=permission.resource,
            action=permission.action,
            effect=permission.effect.value,
            scope=permission.scope,
            conditions=conditions,
            metadata=permission.metadata,
            created_at=permission.created_at,
            created_by=permission.created_by,
            provenance_hash=provenance_hash
        )


class RoleModel(Base):
    """SQLAlchemy model for roles."""

    __tablename__ = 'roles'

    role_id = Column(String(64), primary_key=True)
    role_name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    permissions = Column(JSONB, nullable=True)  # Array of permission IDs
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), nullable=True)
    is_system_role = Column(Boolean, default=False)
    priority = Column(Integer, default=0)

    __table_args__ = (
        Index('idx_role_name', 'role_name'),
        Index('idx_role_priority', 'priority'),
        Index('idx_system_role', 'is_system_role'),
    )


class PolicyModel(Base):
    """SQLAlchemy model for policies."""

    __tablename__ = 'policies'

    policy_id = Column(String(64), primary_key=True)
    policy_name = Column(String(100), unique=True, nullable=False, index=True)
    policy_type = Column(String(50), nullable=False)  # 'rbac', 'abac', 'temporal'
    rules = Column(JSONB, nullable=False)
    conditions = Column(JSONB, nullable=True)
    priority = Column(Integer, default=0)
    enabled = Column(Boolean, default=True)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), nullable=True)
    expires_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index('idx_policy_name', 'policy_name'),
        Index('idx_policy_type', 'policy_type'),
        Index('idx_policy_enabled', 'enabled'),
        Index('idx_policy_priority', 'priority'),
        Index('idx_policy_expires', 'expires_at'),
    )


class AuditLogModel(Base):
    """SQLAlchemy model for audit logs."""

    __tablename__ = 'audit_logs'

    log_id = Column(String(64), primary_key=True, default=lambda: deterministic_uuid(f"audit:{datetime.utcnow().isoformat()}"))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    resource = Column(String(255), nullable=True)
    action = Column(String(100), nullable=True)
    result = Column(String(20), nullable=False)  # 'success', 'failure', 'error'
    details = Column(JSONB, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    session_id = Column(String(255), nullable=True, index=True)
    correlation_id = Column(String(64), nullable=True, index=True)
    provenance_hash = Column(String(64), nullable=True)

    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_event_type', 'event_type'),
        Index('idx_audit_session', 'session_id'),
        Index('idx_audit_correlation', 'correlation_id'),
        Index('idx_audit_result', 'result'),
        CheckConstraint("result IN ('success', 'failure', 'error')", name='check_result')
    )


# Association tables for many-to-many relationships
role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', String(64), ForeignKey('roles.role_id', ondelete='CASCADE')),
    Column('permission_id', String(64), ForeignKey('permissions.permission_id', ondelete='CASCADE')),
    Column('granted_at', DateTime, default=datetime.utcnow),
    Column('granted_by', String(255)),
    UniqueConstraint('role_id', 'permission_id', name='uq_role_permission')
)

user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', String(255), nullable=False),
    Column('role_id', String(64), ForeignKey('roles.role_id', ondelete='CASCADE')),
    Column('assigned_at', DateTime, default=datetime.utcnow),
    Column('assigned_by', String(255)),
    Column('expires_at', DateTime, nullable=True),
    UniqueConstraint('user_id', 'role_id', name='uq_user_role'),
    Index('idx_user_roles_user', 'user_id'),
    Index('idx_user_roles_expires', 'expires_at')
)


# ==============================================================================
# Database Configuration
# ==============================================================================

class DatabaseConfig(BaseModel):
    """Configuration for PostgreSQL database connection."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(default="greenlang", description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    pool_size: int = Field(default=20, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    echo: bool = Field(default=False, description="Echo SQL statements")
    ssl_mode: Optional[str] = Field(default=None, description="SSL mode (require, prefer, etc.)")

    @property
    def connection_url(self) -> str:
        """Generate database connection URL."""
        base_url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        if self.ssl_mode:
            base_url += f"?sslmode={self.ssl_mode}"
        return base_url


# ==============================================================================
# Database Session Management
# ==============================================================================

class DatabaseSession:
    """Database session manager with connection pooling."""

    def __init__(self, config: DatabaseConfig):
        """Initialize database session manager."""
        self.config = config

        # Create engine with connection pooling
        self.engine = create_engine(
            config.connection_url,
            poolclass=QueuePool,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            echo=config.echo,
            pool_pre_ping=True,  # Verify connections before using
            connect_args={
                "connect_timeout": 10,
                "options": "-c statement_timeout=30000"  # 30 second statement timeout
            }
        )

        # Create session factory
        self.SessionLocal = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
                expire_on_commit=False
            )
        )

        # Initialize database schema
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise

    @contextmanager
    def get_session(self) -> Session:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def close(self):
        """Close all database connections."""
        self.SessionLocal.remove()
        self.engine.dispose()


# ==============================================================================
# PostgreSQL Backend Implementation
# ==============================================================================

class PostgreSQLBackend:
    """
    PostgreSQL backend for GreenLang authentication system.

    This backend provides persistent storage for permissions, roles, policies,
    and audit logs with full ACID compliance and performance optimization.

    Attributes:
        config: Database configuration
        db_session: Database session manager

    Example:
        >>> config = DatabaseConfig(username="user", password="pass")
        >>> backend = PostgreSQLBackend(config)
        >>> permission = backend.create_permission(perm_data)
        >>> assert permission.permission_id is not None
    """

    def __init__(self, config: DatabaseConfig):
        """Initialize PostgreSQL backend."""
        self.config = config
        self.db_session = DatabaseSession(config)
        self._stats = {
            'operations': 0,
            'errors': 0,
            'cache_hits': 0
        }

    def create_permission(self, permission: Permission) -> Permission:
        """
        Create new permission in database.

        Args:
            permission: Permission to create

        Returns:
            Created permission with ID

        Raises:
            ValueError: If permission already exists
        """
        start_time = datetime.utcnow()

        try:
            with self.db_session.get_session() as session:
                # Check if permission already exists
                existing = session.query(PermissionModel).filter_by(
                    permission_id=permission.permission_id
                ).first()

                if existing:
                    raise ValueError(f"Permission {permission.permission_id} already exists")

                # Create new permission
                db_permission = PermissionModel.from_permission(permission)
                session.add(db_permission)
                session.flush()

                # Log audit event
                self._log_audit_event(
                    session,
                    event_type="permission.created",
                    user_id=permission.created_by,
                    resource=permission.resource,
                    action=permission.action,
                    result="success",
                    details={
                        "permission_id": permission.permission_id,
                        "effect": permission.effect.value
                    }
                )

                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.info(f"Created permission {permission.permission_id} in {processing_time:.2f}ms")

                return permission

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to create permission: {e}")
            raise
        finally:
            self._stats['operations'] += 1

    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """
        Get permission by ID.

        Args:
            permission_id: Permission ID

        Returns:
            Permission if found, None otherwise
        """
        try:
            with self.db_session.get_session() as session:
                db_permission = session.query(PermissionModel).filter_by(
                    permission_id=permission_id
                ).first()

                if db_permission:
                    return db_permission.to_permission()
                return None

        except Exception as e:
            logger.error(f"Failed to get permission: {e}")
            raise

    def update_permission(self, permission: Permission) -> Permission:
        """
        Update existing permission.

        Args:
            permission: Permission to update

        Returns:
            Updated permission

        Raises:
            ValueError: If permission not found
        """
        try:
            with self.db_session.get_session() as session:
                db_permission = session.query(PermissionModel).filter_by(
                    permission_id=permission.permission_id
                ).first()

                if not db_permission:
                    raise ValueError(f"Permission {permission.permission_id} not found")

                # Update fields
                db_permission.resource = permission.resource
                db_permission.action = permission.action
                db_permission.effect = permission.effect.value
                db_permission.scope = permission.scope
                db_permission.conditions = [c.dict() for c in permission.conditions] if permission.conditions else None
                db_permission.metadata = permission.metadata
                db_permission.updated_at = datetime.utcnow()

                # Recalculate provenance hash
                provenance_str = f"{permission.resource}{permission.action}{permission.effect.value}"
                db_permission.provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

                session.flush()

                # Log audit event
                self._log_audit_event(
                    session,
                    event_type="permission.updated",
                    user_id=permission.created_by,
                    resource=permission.resource,
                    action=permission.action,
                    result="success",
                    details={"permission_id": permission.permission_id}
                )

                logger.info(f"Updated permission {permission.permission_id}")
                return permission

        except Exception as e:
            logger.error(f"Failed to update permission: {e}")
            raise

    def delete_permission(self, permission_id: str) -> bool:
        """
        Delete permission by ID.

        Args:
            permission_id: Permission ID to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            with self.db_session.get_session() as session:
                result = session.query(PermissionModel).filter_by(
                    permission_id=permission_id
                ).delete()

                if result > 0:
                    # Log audit event
                    self._log_audit_event(
                        session,
                        event_type="permission.deleted",
                        resource="permission",
                        action="delete",
                        result="success",
                        details={"permission_id": permission_id}
                    )

                    logger.info(f"Deleted permission {permission_id}")
                    return True

                return False

        except Exception as e:
            logger.error(f"Failed to delete permission: {e}")
            raise

    def list_permissions(
        self,
        resource_pattern: Optional[str] = None,
        action_pattern: Optional[str] = None,
        scope: Optional[str] = None,
        effect: Optional[PermissionEffect] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Permission]:
        """
        List permissions matching criteria with pagination.

        Args:
            resource_pattern: Filter by resource pattern (supports wildcards)
            action_pattern: Filter by action pattern (supports wildcards)
            scope: Filter by scope
            effect: Filter by effect
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching permissions
        """
        try:
            with self.db_session.get_session() as session:
                query = session.query(PermissionModel)

                # Apply filters
                if resource_pattern:
                    if '*' in resource_pattern:
                        pattern = resource_pattern.replace('*', '%')
                        query = query.filter(PermissionModel.resource.like(pattern))
                    else:
                        query = query.filter_by(resource=resource_pattern)

                if action_pattern:
                    if '*' in action_pattern:
                        pattern = action_pattern.replace('*', '%')
                        query = query.filter(PermissionModel.action.like(pattern))
                    else:
                        query = query.filter_by(action=action_pattern)

                if scope:
                    query = query.filter_by(scope=scope)

                if effect:
                    query = query.filter_by(effect=effect.value)

                # Apply pagination
                query = query.offset(offset).limit(limit)

                # Execute query
                db_permissions = query.all()

                return [p.to_permission() for p in db_permissions]

        except Exception as e:
            logger.error(f"Failed to list permissions: {e}")
            raise

    def create_role(self, role_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new role in database.

        Args:
            role_data: Role data dictionary

        Returns:
            Created role data
        """
        try:
            with self.db_session.get_session() as session:
                role_id = role_data.get('role_id', str(deterministic_uuid('role', role_data['role_name'])))

                # Check if role exists
                existing = session.query(RoleModel).filter_by(role_name=role_data['role_name']).first()
                if existing:
                    raise ValueError(f"Role {role_data['role_name']} already exists")

                # Create role
                db_role = RoleModel(
                    role_id=role_id,
                    role_name=role_data['role_name'],
                    description=role_data.get('description'),
                    permissions=role_data.get('permissions', []),
                    metadata=role_data.get('metadata', {}),
                    created_by=role_data.get('created_by'),
                    is_system_role=role_data.get('is_system_role', False),
                    priority=role_data.get('priority', 0)
                )
                session.add(db_role)

                # Add role-permission associations
                if 'permissions' in role_data and role_data['permissions']:
                    for perm_id in role_data['permissions']:
                        session.execute(
                            role_permissions.insert().values(
                                role_id=role_id,
                                permission_id=perm_id,
                                granted_by=role_data.get('created_by')
                            )
                        )

                session.flush()

                # Log audit event
                self._log_audit_event(
                    session,
                    event_type="role.created",
                    user_id=role_data.get('created_by'),
                    resource="role",
                    action="create",
                    result="success",
                    details={"role_id": role_id, "role_name": role_data['role_name']}
                )

                logger.info(f"Created role {role_data['role_name']}")
                role_data['role_id'] = role_id
                return role_data

        except Exception as e:
            logger.error(f"Failed to create role: {e}")
            raise

    def get_role(self, role_id: str) -> Optional[Dict[str, Any]]:
        """
        Get role by ID.

        Args:
            role_id: Role ID

        Returns:
            Role data if found, None otherwise
        """
        try:
            with self.db_session.get_session() as session:
                db_role = session.query(RoleModel).filter_by(role_id=role_id).first()

                if db_role:
                    return {
                        'role_id': db_role.role_id,
                        'role_name': db_role.role_name,
                        'description': db_role.description,
                        'permissions': db_role.permissions or [],
                        'metadata': db_role.metadata or {},
                        'created_at': db_role.created_at,
                        'created_by': db_role.created_by,
                        'is_system_role': db_role.is_system_role,
                        'priority': db_role.priority
                    }
                return None

        except Exception as e:
            logger.error(f"Failed to get role: {e}")
            raise

    def create_policy(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new policy in database.

        Args:
            policy_data: Policy data dictionary

        Returns:
            Created policy data
        """
        try:
            with self.db_session.get_session() as session:
                policy_id = policy_data.get('policy_id', str(deterministic_uuid('policy', policy_data['policy_name'])))

                # Check if policy exists
                existing = session.query(PolicyModel).filter_by(policy_name=policy_data['policy_name']).first()
                if existing:
                    raise ValueError(f"Policy {policy_data['policy_name']} already exists")

                # Create policy
                db_policy = PolicyModel(
                    policy_id=policy_id,
                    policy_name=policy_data['policy_name'],
                    policy_type=policy_data['policy_type'],
                    rules=policy_data['rules'],
                    conditions=policy_data.get('conditions'),
                    priority=policy_data.get('priority', 0),
                    enabled=policy_data.get('enabled', True),
                    metadata=policy_data.get('metadata', {}),
                    created_by=policy_data.get('created_by'),
                    expires_at=policy_data.get('expires_at')
                )
                session.add(db_policy)
                session.flush()

                # Log audit event
                self._log_audit_event(
                    session,
                    event_type="policy.created",
                    user_id=policy_data.get('created_by'),
                    resource="policy",
                    action="create",
                    result="success",
                    details={"policy_id": policy_id, "policy_name": policy_data['policy_name']}
                )

                logger.info(f"Created policy {policy_data['policy_name']}")
                policy_data['policy_id'] = policy_id
                return policy_data

        except Exception as e:
            logger.error(f"Failed to create policy: {e}")
            raise

    def log_audit_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log audit event to database.

        Args:
            event_type: Type of event (e.g., 'permission.created')
            user_id: User who performed the action
            resource: Resource affected
            action: Action performed
            result: Result of action ('success', 'failure', 'error')
            details: Additional event details
            ip_address: Client IP address
            user_agent: Client user agent
            session_id: Session identifier
            correlation_id: Correlation ID for tracing

        Returns:
            Audit log ID
        """
        try:
            with self.db_session.get_session() as session:
                return self._log_audit_event(
                    session, event_type, user_id, resource, action,
                    result, details, ip_address, user_agent, session_id, correlation_id
                )
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise

    def _log_audit_event(
        self,
        session: Session,
        event_type: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """Internal method to log audit event within a session."""
        # Calculate provenance hash (deterministic)
        provenance_str = f"{event_type}{user_id}{resource}{action}{result}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        # Generate deterministic log ID from provenance
        log_id = deterministic_uuid(f"audit:{provenance_hash}:{datetime.utcnow().isoformat()}")

        audit_log = AuditLogModel(
            log_id=log_id,
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            correlation_id=correlation_id,
            provenance_hash=provenance_hash
        )

        session.add(audit_log)
        return log_id

    def get_audit_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        result: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs with filtering and pagination.

        Args:
            start_time: Start time filter
            end_time: End time filter
            event_type: Event type filter
            user_id: User ID filter
            result: Result filter
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of audit log entries
        """
        try:
            with self.db_session.get_session() as session:
                query = session.query(AuditLogModel)

                # Apply filters
                if start_time:
                    query = query.filter(AuditLogModel.timestamp >= start_time)
                if end_time:
                    query = query.filter(AuditLogModel.timestamp <= end_time)
                if event_type:
                    query = query.filter_by(event_type=event_type)
                if user_id:
                    query = query.filter_by(user_id=user_id)
                if result:
                    query = query.filter_by(result=result)

                # Order by timestamp descending
                query = query.order_by(AuditLogModel.timestamp.desc())

                # Apply pagination
                query = query.offset(offset).limit(limit)

                # Execute query
                audit_logs = query.all()

                return [
                    {
                        'log_id': log.log_id,
                        'timestamp': log.timestamp,
                        'event_type': log.event_type,
                        'user_id': log.user_id,
                        'resource': log.resource,
                        'action': log.action,
                        'result': log.result,
                        'details': log.details,
                        'ip_address': log.ip_address,
                        'session_id': log.session_id,
                        'correlation_id': log.correlation_id,
                        'provenance_hash': log.provenance_hash
                    }
                    for log in audit_logs
                ]

        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            raise

    def assign_role_to_user(
        self,
        user_id: str,
        role_id: str,
        assigned_by: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """
        Assign role to user.

        Args:
            user_id: User ID
            role_id: Role ID to assign
            assigned_by: User who assigned the role
            expires_at: Optional expiration time

        Returns:
            True if successful
        """
        try:
            with self.db_session.get_session() as session:
                # Check if role exists
                role = session.query(RoleModel).filter_by(role_id=role_id).first()
                if not role:
                    raise ValueError(f"Role {role_id} not found")

                # Check if assignment already exists
                existing = session.execute(
                    select(user_roles).where(
                        and_(
                            user_roles.c.user_id == user_id,
                            user_roles.c.role_id == role_id
                        )
                    )
                ).first()

                if existing:
                    logger.warning(f"User {user_id} already has role {role_id}")
                    return True

                # Create assignment
                session.execute(
                    user_roles.insert().values(
                        user_id=user_id,
                        role_id=role_id,
                        assigned_by=assigned_by,
                        expires_at=expires_at
                    )
                )

                # Log audit event
                self._log_audit_event(
                    session,
                    event_type="role.assigned",
                    user_id=assigned_by,
                    resource="user",
                    action="assign_role",
                    result="success",
                    details={
                        "user_id": user_id,
                        "role_id": role_id,
                        "expires_at": expires_at.isoformat() if expires_at else None
                    }
                )

                logger.info(f"Assigned role {role_id} to user {user_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to assign role: {e}")
            raise

    def get_user_roles(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all roles assigned to a user.

        Args:
            user_id: User ID

        Returns:
            List of role assignments
        """
        try:
            with self.db_session.get_session() as session:
                # Query user roles with role details
                results = session.execute(
                    select(user_roles, RoleModel).join(
                        RoleModel,
                        user_roles.c.role_id == RoleModel.role_id
                    ).where(
                        user_roles.c.user_id == user_id
                    ).where(
                        or_(
                            user_roles.c.expires_at.is_(None),
                            user_roles.c.expires_at > datetime.utcnow()
                        )
                    )
                ).all()

                return [
                    {
                        'role_id': row.role_id,
                        'role_name': row.RoleModel.role_name,
                        'description': row.RoleModel.description,
                        'assigned_at': row.assigned_at,
                        'assigned_by': row.assigned_by,
                        'expires_at': row.expires_at,
                        'is_system_role': row.RoleModel.is_system_role,
                        'priority': row.RoleModel.priority
                    }
                    for row in results
                ]

        except Exception as e:
            logger.error(f"Failed to get user roles: {e}")
            raise

    def cleanup_expired_data(self, older_than_days: int = 90) -> Dict[str, int]:
        """
        Clean up expired data from the database.

        Args:
            older_than_days: Delete audit logs older than this many days

        Returns:
            Dictionary with counts of deleted items
        """
        try:
            with self.db_session.get_session() as session:
                counts = {}

                # Clean up expired user-role assignments
                expired_roles = session.execute(
                    delete(user_roles).where(
                        and_(
                            user_roles.c.expires_at.isnot(None),
                            user_roles.c.expires_at < datetime.utcnow()
                        )
                    )
                )
                counts['expired_role_assignments'] = expired_roles.rowcount

                # Clean up expired policies
                expired_policies = session.query(PolicyModel).filter(
                    and_(
                        PolicyModel.expires_at.isnot(None),
                        PolicyModel.expires_at < datetime.utcnow()
                    )
                ).delete()
                counts['expired_policies'] = expired_policies

                # Clean up old audit logs
                cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
                old_logs = session.query(AuditLogModel).filter(
                    AuditLogModel.timestamp < cutoff_date
                ).delete()
                counts['old_audit_logs'] = old_logs

                session.commit()

                logger.info(f"Cleanup completed: {counts}")
                return counts

        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        try:
            with self.db_session.get_session() as session:
                stats = {
                    'permissions_count': session.query(func.count(PermissionModel.permission_id)).scalar(),
                    'roles_count': session.query(func.count(RoleModel.role_id)).scalar(),
                    'policies_count': session.query(func.count(PolicyModel.policy_id)).scalar(),
                    'audit_logs_count': session.query(func.count(AuditLogModel.log_id)).scalar(),
                    'active_policies': session.query(func.count(PolicyModel.policy_id)).filter_by(enabled=True).scalar(),
                    'system_roles': session.query(func.count(RoleModel.role_id)).filter_by(is_system_role=True).scalar(),
                    'backend_operations': self._stats['operations'],
                    'backend_errors': self._stats['errors'],
                    'cache_hits': self._stats['cache_hits']
                }

                # Get recent audit activity
                recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                stats['recent_audit_events'] = session.query(func.count(AuditLogModel.log_id)).filter(
                    AuditLogModel.timestamp >= recent_cutoff
                ).scalar()

                return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise

    def close(self):
        """Close database connections."""
        self.db_session.close()


# ==============================================================================
# Convenience Functions
# ==============================================================================

_db_session: Optional[DatabaseSession] = None

def init_database(config: DatabaseConfig) -> DatabaseSession:
    """
    Initialize global database session.

    Args:
        config: Database configuration

    Returns:
        Database session manager
    """
    global _db_session
    _db_session = DatabaseSession(config)
    return _db_session

def get_db_session() -> DatabaseSession:
    """
    Get global database session.

    Returns:
        Database session manager

    Raises:
        RuntimeError: If database not initialized
    """
    if _db_session is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _db_session


__all__ = [
    'PostgreSQLBackend',
    'DatabaseConfig',
    'DatabaseSession',
    'PermissionModel',
    'RoleModel',
    'PolicyModel',
    'AuditLogModel',
    'init_database',
    'get_db_session'
]