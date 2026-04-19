# -*- coding: utf-8 -*-
"""
SQLAlchemy models for analytics dashboards.

This module defines the database models for storing dashboards,
widgets, sharing configurations, and templates.
"""

import uuid
from datetime import datetime
from typing import Optional

from greenlang.utilities.determinism import DeterministicClock
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
    Boolean,
    JSON,
    Index,
    Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from greenlang.db.base import Base


class Dashboard(Base):
    """Dashboard model for storing dashboard configurations."""

    __tablename__ = "dashboards"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    layout = Column(JSON, nullable=False, default=dict)  # Grid layout configuration
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Dashboard organization
    folder_id = Column(UUID(as_uuid=True), ForeignKey("dashboard_folders.id"), nullable=True)
    tags = Column(JSON, default=list)  # List of tags for categorization

    # Access control
    access_level = Column(
        SQLEnum('private', 'team', 'public', name='dashboard_access_level'),
        default='private',
        nullable=False
    )
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True)

    # Dashboard state
    is_template = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False)

    # Relationships
    widgets = relationship("DashboardWidget", back_populates="dashboard", cascade="all, delete-orphan")
    shares = relationship("DashboardShare", back_populates="dashboard", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_dashboard_created_by', 'created_by'),
        Index('idx_dashboard_team', 'team_id'),
        Index('idx_dashboard_folder', 'folder_id'),
        Index('idx_dashboard_access', 'access_level'),
        Index('idx_dashboard_created_at', 'created_at'),
    )

    def __repr__(self):
        return f"<Dashboard(id={self.id}, name={self.name})>"


class DashboardWidget(Base):
    """Dashboard widget model for storing widget configurations."""

    __tablename__ = "dashboard_widgets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dashboard_id = Column(UUID(as_uuid=True), ForeignKey("dashboards.id"), nullable=False)

    # Widget type and configuration
    widget_type = Column(
        SQLEnum(
            'line_chart', 'bar_chart', 'gauge_chart', 'stat_card',
            'table', 'heatmap', 'pie_chart', 'alert',
            name='widget_type'
        ),
        nullable=False
    )
    title = Column(String(255), nullable=False)
    config = Column(JSON, nullable=False, default=dict)  # Widget-specific configuration

    # Data source configuration
    data_source = Column(JSON, nullable=False, default=dict)  # Channel, metric name, tags, etc.

    # Layout position and size
    position_x = Column(Integer, nullable=False, default=0)
    position_y = Column(Integer, nullable=False, default=0)
    width = Column(Integer, nullable=False, default=6)
    height = Column(Integer, nullable=False, default=4)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    dashboard = relationship("Dashboard", back_populates="widgets")

    # Indexes
    __table_args__ = (
        Index('idx_widget_dashboard', 'dashboard_id'),
        Index('idx_widget_type', 'widget_type'),
    )

    def __repr__(self):
        return f"<DashboardWidget(id={self.id}, type={self.widget_type}, title={self.title})>"


class DashboardShare(Base):
    """Dashboard share model for sharing dashboards via tokens."""

    __tablename__ = "dashboard_shares"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dashboard_id = Column(UUID(as_uuid=True), ForeignKey("dashboards.id"), nullable=False)

    # Sharing token
    token = Column(String(255), unique=True, nullable=False, index=True)

    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Permissions
    permissions = Column(JSON, nullable=False, default=dict)  # view, edit, comment, etc.
    can_view = Column(Boolean, default=True)
    can_edit = Column(Boolean, default=False)

    # Share metadata
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    accessed_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    dashboard = relationship("Dashboard", back_populates="shares")

    # Indexes
    __table_args__ = (
        Index('idx_share_dashboard', 'dashboard_id'),
        Index('idx_share_token', 'token'),
        Index('idx_share_expires', 'expires_at'),
    )

    def __repr__(self):
        return f"<DashboardShare(id={self.id}, token={self.token})>"

    def is_expired(self) -> bool:
        """Check if share link is expired."""
        if self.expires_at is None:
            return False
        return DeterministicClock.utcnow() > self.expires_at


class DashboardTemplate(Base):
    """Dashboard template model for predefined dashboard layouts."""

    __tablename__ = "dashboard_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)

    # Template configuration
    layout = Column(JSON, nullable=False, default=dict)
    widgets = Column(JSON, nullable=False, default=list)  # List of widget configurations

    # Categorization
    category = Column(
        SQLEnum(
            'system', 'workflow', 'agent', 'distributed', 'custom',
            name='template_category'
        ),
        nullable=False,
        default='custom'
    )
    tags = Column(JSON, default=list)

    # Template metadata
    is_builtin = Column(Boolean, default=False)
    is_public = Column(Boolean, default=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Usage statistics
    usage_count = Column(Integer, default=0)

    # Indexes
    __table_args__ = (
        Index('idx_template_category', 'category'),
        Index('idx_template_public', 'is_public'),
        Index('idx_template_builtin', 'is_builtin'),
    )

    def __repr__(self):
        return f"<DashboardTemplate(id={self.id}, name={self.name})>"


class DashboardFolder(Base):
    """Dashboard folder model for organizing dashboards."""

    __tablename__ = "dashboard_folders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Folder hierarchy
    parent_id = Column(UUID(as_uuid=True), ForeignKey("dashboard_folders.id"), nullable=True)

    # Access control
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Indexes
    __table_args__ = (
        Index('idx_folder_parent', 'parent_id'),
        Index('idx_folder_created_by', 'created_by'),
        Index('idx_folder_team', 'team_id'),
    )

    def __repr__(self):
        return f"<DashboardFolder(id={self.id}, name={self.name})>"
