"""
Tenant SQLAlchemy Model - Enhanced Multi-Tenancy Support

This module defines the Tenant model with comprehensive multi-tenancy support
including subscription tiers, feature flags, usage quotas, and billing integration.

The tenant model is the foundation for:
- Row-Level Security (RLS) isolation
- Subscription tier-based feature gating
- Usage tracking and billing metrics
- Tenant-specific configuration

Example:
    >>> from models.tenant import Tenant, SubscriptionTier
    >>> tenant = Tenant(
    ...     tenant_id="t-acme-corp",
    ...     name="Acme Corporation",
    ...     slug="acme-corp",
    ...     subscription_tier=SubscriptionTier.ENTERPRISE
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import uuid

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SQLEnum,
    Index,
    Integer,
    JSON,
    String,
    Boolean,
    Text,
    Float,
    BigInteger,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from db.base import Base


class SubscriptionTier(str, Enum):
    """
    Subscription tier levels.

    Each tier has different feature access, quotas, and support levels.

    Tiers:
        FREE: Basic functionality, limited agents and executions
        PRO: Professional features, higher limits, priority support
        ENTERPRISE: Full features, custom limits, dedicated support, SLA
    """

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class TenantStatus(str, Enum):
    """
    Tenant account status.

    Statuses:
        PENDING: Awaiting activation/verification
        ACTIVE: Fully operational
        SUSPENDED: Temporarily disabled (e.g., payment issue)
        DEACTIVATED: Permanently disabled
    """

    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"


# Default quotas by subscription tier
DEFAULT_TIER_QUOTAS: Dict[SubscriptionTier, Dict[str, int]] = {
    SubscriptionTier.FREE: {
        "agents": 5,
        "executions_per_month": 100,
        "storage_gb": 1,
        "concurrent_executions": 2,
        "api_calls_per_minute": 10,
        "team_members": 2,
        "audit_log_retention_days": 7,
    },
    SubscriptionTier.PRO: {
        "agents": 50,
        "executions_per_month": 5000,
        "storage_gb": 25,
        "concurrent_executions": 10,
        "api_calls_per_minute": 100,
        "team_members": 10,
        "audit_log_retention_days": 90,
    },
    SubscriptionTier.ENTERPRISE: {
        "agents": 500,
        "executions_per_month": 100000,
        "storage_gb": 500,
        "concurrent_executions": 100,
        "api_calls_per_minute": 1000,
        "team_members": -1,  # Unlimited
        "audit_log_retention_days": 365,
    },
}

# Default feature flags by subscription tier
DEFAULT_TIER_FEATURES: Dict[SubscriptionTier, Dict[str, bool]] = {
    SubscriptionTier.FREE: {
        "basic_agents": True,
        "carbon_emissions": True,
        "cbam_compliance": False,
        "csrd_reporting": False,
        "eudr_compliance": False,
        "custom_agents": False,
        "api_access": True,
        "webhook_notifications": False,
        "sso_authentication": False,
        "audit_logs": False,
        "priority_support": False,
        "dedicated_infrastructure": False,
        "custom_integrations": False,
        "white_labeling": False,
        "data_export": True,
        "bulk_operations": False,
        "real_time_streaming": False,
        "advanced_analytics": False,
    },
    SubscriptionTier.PRO: {
        "basic_agents": True,
        "carbon_emissions": True,
        "cbam_compliance": True,
        "csrd_reporting": True,
        "eudr_compliance": False,
        "custom_agents": True,
        "api_access": True,
        "webhook_notifications": True,
        "sso_authentication": False,
        "audit_logs": True,
        "priority_support": True,
        "dedicated_infrastructure": False,
        "custom_integrations": False,
        "white_labeling": False,
        "data_export": True,
        "bulk_operations": True,
        "real_time_streaming": True,
        "advanced_analytics": True,
    },
    SubscriptionTier.ENTERPRISE: {
        "basic_agents": True,
        "carbon_emissions": True,
        "cbam_compliance": True,
        "csrd_reporting": True,
        "eudr_compliance": True,
        "custom_agents": True,
        "api_access": True,
        "webhook_notifications": True,
        "sso_authentication": True,
        "audit_logs": True,
        "priority_support": True,
        "dedicated_infrastructure": True,
        "custom_integrations": True,
        "white_labeling": True,
        "data_export": True,
        "bulk_operations": True,
        "real_time_streaming": True,
        "advanced_analytics": True,
    },
}


class Tenant(Base):
    """
    Tenant model for multi-tenancy support.

    Represents an organization using the GreenLang Agent Factory.
    All tenant data is isolated using Row-Level Security (RLS) policies.

    Attributes:
        id: Primary key (UUID)
        tenant_id: External tenant identifier (e.g., t-acme-corp)
        name: Organization name
        slug: URL-safe slug for subdomains
        status: Account status (active, suspended, etc.)
        subscription_tier: Subscription level (free, pro, enterprise)
        settings: Tenant-specific settings (JSON)
        quotas: Resource quotas (JSON)
        current_usage: Current resource usage (JSON)
        feature_flags: Feature flag overrides (JSON)
        billing_info: Billing and payment information (JSON)

    Example:
        >>> tenant = Tenant(
        ...     tenant_id="t-acme-corp",
        ...     name="Acme Corporation",
        ...     slug="acme-corp",
        ...     subscription_tier=SubscriptionTier.ENTERPRISE
        ... )
        >>> tenant.is_feature_enabled("csrd_reporting")
        True
    """

    __tablename__ = "tenants"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Primary key (UUID)",
    )

    # Tenant identifier
    tenant_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="External tenant identifier (e.g., t-acme-corp)",
    )

    # Basic information
    name = Column(
        String(255),
        nullable=False,
        comment="Organization name",
    )
    slug = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="URL-safe slug for subdomains",
    )
    domain = Column(
        String(255),
        nullable=True,
        unique=True,
        index=True,
        comment="Custom domain (e.g., acme.greenlang.io)",
    )

    # Status and tier
    status = Column(
        SQLEnum(TenantStatus),
        nullable=False,
        default=TenantStatus.PENDING,
        index=True,
        comment="Account status",
    )
    subscription_tier = Column(
        SQLEnum(SubscriptionTier),
        nullable=False,
        default=SubscriptionTier.FREE,
        index=True,
        comment="Subscription tier",
    )

    # Activation tracking
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Quick active check (derived from status)",
    )
    activated_at = Column(
        DateTime,
        nullable=True,
        comment="When tenant was activated",
    )
    suspended_at = Column(
        DateTime,
        nullable=True,
        comment="When tenant was suspended (if applicable)",
    )
    suspension_reason = Column(
        Text,
        nullable=True,
        comment="Reason for suspension",
    )

    # Settings (tenant-specific configuration)
    settings = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Tenant-specific settings",
    )

    # Quotas and usage
    quotas = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Resource quotas (merged with tier defaults)",
    )
    current_usage = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Current resource usage",
    )
    usage_reset_at = Column(
        DateTime,
        nullable=True,
        comment="When monthly usage counters were last reset",
    )

    # Feature flags
    feature_flags = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Feature flag overrides (merged with tier defaults)",
    )

    # Billing information
    billing_info = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Billing and payment information",
    )
    billing_email = Column(
        String(255),
        nullable=True,
        comment="Billing email address",
    )
    stripe_customer_id = Column(
        String(255),
        nullable=True,
        unique=True,
        index=True,
        comment="Stripe customer ID for billing",
    )
    stripe_subscription_id = Column(
        String(255),
        nullable=True,
        unique=True,
        comment="Stripe subscription ID",
    )

    # Trial tracking
    trial_ends_at = Column(
        DateTime,
        nullable=True,
        comment="Trial period end date",
    )
    is_trial = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether tenant is in trial period",
    )

    # Contact information
    primary_contact_name = Column(
        String(255),
        nullable=True,
        comment="Primary contact name",
    )
    primary_contact_email = Column(
        String(255),
        nullable=True,
        comment="Primary contact email",
    )

    # Compliance and legal
    data_residency_region = Column(
        String(50),
        nullable=True,
        default="us-east-1",
        comment="Data residency region for compliance",
    )
    accepted_terms_version = Column(
        String(50),
        nullable=True,
        comment="Version of terms of service accepted",
    )
    accepted_terms_at = Column(
        DateTime,
        nullable=True,
        comment="When terms were accepted",
    )
    dpa_signed = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Data Processing Agreement signed",
    )

    # Metadata
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata",
    )
    tags = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="Tags for categorization",
    )

    # Timestamps
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        comment="Creation timestamp",
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="Last update timestamp",
    )

    # Relationships
    agents = relationship(
        "Agent",
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    users = relationship(
        "User",
        back_populates="tenant",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("ix_tenants_status_tier", "status", "subscription_tier"),
        Index("ix_tenants_created_at", "created_at"),
        Index("ix_tenants_settings", "settings", postgresql_using="gin"),
        Index("ix_tenants_feature_flags", "feature_flags", postgresql_using="gin"),
        {"comment": "Multi-tenant organizations"},
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<Tenant(name={self.name}, slug={self.slug}, tier={self.subscription_tier.value})>"

    def get_effective_quotas(self) -> Dict[str, int]:
        """
        Get effective quotas (tier defaults merged with custom quotas).

        Custom quotas override tier defaults. Returns the merged result.

        Returns:
            Dictionary of quota name to limit value
        """
        tier_quotas = DEFAULT_TIER_QUOTAS.get(
            self.subscription_tier, DEFAULT_TIER_QUOTAS[SubscriptionTier.FREE]
        ).copy()

        # Merge with custom quotas
        if self.quotas:
            tier_quotas.update(self.quotas)

        return tier_quotas

    def get_quota(self, quota_name: str) -> int:
        """
        Get a specific quota value.

        Args:
            quota_name: Name of the quota

        Returns:
            Quota limit value (-1 for unlimited)
        """
        effective_quotas = self.get_effective_quotas()
        return effective_quotas.get(quota_name, 0)

    def get_current_usage(self, usage_name: str) -> int:
        """
        Get current usage for a specific metric.

        Args:
            usage_name: Name of the usage metric

        Returns:
            Current usage value
        """
        if not self.current_usage:
            return 0
        return self.current_usage.get(usage_name, 0)

    def check_quota(self, quota_name: str, increment: int = 1) -> bool:
        """
        Check if an operation would exceed quota.

        Args:
            quota_name: Name of the quota to check
            increment: Amount to add (default 1)

        Returns:
            True if operation is allowed, False if would exceed quota
        """
        limit = self.get_quota(quota_name)

        # -1 means unlimited
        if limit == -1:
            return True

        current = self.get_current_usage(quota_name)
        return (current + increment) <= limit

    def get_quota_remaining(self, quota_name: str) -> int:
        """
        Get remaining quota for a metric.

        Args:
            quota_name: Name of the quota

        Returns:
            Remaining quota (-1 for unlimited)
        """
        limit = self.get_quota(quota_name)
        if limit == -1:
            return -1

        current = self.get_current_usage(quota_name)
        return max(0, limit - current)

    def get_effective_feature_flags(self) -> Dict[str, bool]:
        """
        Get effective feature flags (tier defaults merged with custom flags).

        Custom flags override tier defaults. Returns the merged result.

        Returns:
            Dictionary of feature name to enabled status
        """
        tier_features = DEFAULT_TIER_FEATURES.get(
            self.subscription_tier, DEFAULT_TIER_FEATURES[SubscriptionTier.FREE]
        ).copy()

        # Merge with custom feature flags
        if self.feature_flags:
            tier_features.update(self.feature_flags)

        return tier_features

    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled for this tenant.

        Args:
            feature_name: Name of the feature

        Returns:
            True if feature is enabled, False otherwise
        """
        effective_features = self.get_effective_feature_flags()
        return effective_features.get(feature_name, False)

    def get_setting(self, setting_name: str, default: Any = None) -> Any:
        """
        Get a tenant-specific setting.

        Args:
            setting_name: Name of the setting
            default: Default value if not set

        Returns:
            Setting value or default
        """
        if not self.settings:
            return default
        return self.settings.get(setting_name, default)

    def is_operational(self) -> bool:
        """
        Check if tenant is operational (can perform actions).

        Returns:
            True if tenant is active and not suspended
        """
        return (
            self.status == TenantStatus.ACTIVE
            and self.is_active
        )

    def is_in_trial(self) -> bool:
        """
        Check if tenant is in trial period.

        Returns:
            True if in trial and trial hasn't expired
        """
        if not self.is_trial or not self.trial_ends_at:
            return False
        return datetime.utcnow() < self.trial_ends_at

    def calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 hash for audit trail.

        Returns:
            Provenance hash string
        """
        data_str = f"{self.tenant_id}:{self.name}:{self.subscription_tier.value}:{self.updated_at.isoformat()}"
        return hashlib.sha256(data_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation of tenant
        """
        return {
            "id": str(self.id),
            "tenant_id": self.tenant_id,
            "name": self.name,
            "slug": self.slug,
            "domain": self.domain,
            "status": self.status.value,
            "subscription_tier": self.subscription_tier.value,
            "is_active": self.is_active,
            "is_trial": self.is_trial,
            "trial_ends_at": self.trial_ends_at.isoformat() if self.trial_ends_at else None,
            "settings": self.settings,
            "quotas": self.get_effective_quotas(),
            "current_usage": self.current_usage,
            "feature_flags": self.get_effective_feature_flags(),
            "data_residency_region": self.data_residency_region,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_public_dict(self) -> Dict[str, Any]:
        """
        Convert to public dictionary (excludes sensitive info).

        Returns:
            Public dictionary representation
        """
        return {
            "id": str(self.id),
            "tenant_id": self.tenant_id,
            "name": self.name,
            "slug": self.slug,
            "subscription_tier": self.subscription_tier.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }


class TenantUsageLog(Base):
    """
    Tenant usage log for tracking resource consumption.

    Records historical usage data for billing and analytics.

    Attributes:
        id: Primary key
        tenant_id: Foreign key to tenant
        metric_name: Name of the usage metric
        metric_value: Value recorded
        recorded_at: When the metric was recorded
    """

    __tablename__ = "tenant_usage_logs"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tenant_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    metric_name = Column(
        String(100),
        nullable=False,
        index=True,
    )
    metric_value = Column(
        BigInteger,
        nullable=False,
    )
    period_start = Column(
        DateTime,
        nullable=False,
        index=True,
    )
    period_end = Column(
        DateTime,
        nullable=False,
    )
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
    )
    recorded_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (
        Index("ix_usage_logs_tenant_metric", "tenant_id", "metric_name"),
        Index("ix_usage_logs_period", "tenant_id", "period_start", "period_end"),
        {"comment": "Tenant usage history"},
    )


class TenantInvitation(Base):
    """
    Tenant invitation for user onboarding.

    Attributes:
        id: Primary key
        tenant_id: Foreign key to tenant
        email: Invited email address
        role: Role to assign
        token: Invitation token
        expires_at: Token expiration
    """

    __tablename__ = "tenant_invitations"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tenant_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    email = Column(
        String(255),
        nullable=False,
        index=True,
    )
    role = Column(
        String(50),
        nullable=False,
        default="viewer",
    )
    token = Column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )
    invited_by = Column(
        UUID(as_uuid=True),
        nullable=False,
    )
    expires_at = Column(
        DateTime,
        nullable=False,
    )
    accepted_at = Column(
        DateTime,
        nullable=True,
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (
        Index("ix_invitations_tenant_email", "tenant_id", "email"),
        {"comment": "Pending user invitations"},
    )

    def is_expired(self) -> bool:
        """Check if invitation has expired."""
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if invitation is still valid."""
        return not self.is_expired() and self.accepted_at is None
