# -*- coding: utf-8 -*-
"""
Marketplace Database Models

SQLAlchemy models for the GreenLang Agent Marketplace.
Provides comprehensive data structures for agents, versions, reviews,
categories, dependencies, licensing, and monetization.
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from enum import Enum

from sqlalchemy import (
from greenlang.determinism import FinancialDecimal
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Table,
    Index,
    UniqueConstraint,
    CheckConstraint,
    Numeric,
    JSON,
    ARRAY,
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.ext.hybrid import hybrid_property
import uuid

Base = declarative_base()


# Association tables for many-to-many relationships
agent_tags = Table(
    "agent_tags",
    Base.metadata,
    Column("agent_id", UUID(as_uuid=True), ForeignKey("marketplace_agents.id")),
    Column("tag_id", Integer, ForeignKey("agent_tags_table.id")),
    Index("idx_agent_tags_agent", "agent_id"),
    Index("idx_agent_tags_tag", "tag_id"),
)

agent_frequently_installed = Table(
    "agent_frequently_installed",
    Base.metadata,
    Column("agent_id", UUID(as_uuid=True), ForeignKey("marketplace_agents.id")),
    Column("frequently_with_id", UUID(as_uuid=True), ForeignKey("marketplace_agents.id")),
    Column("frequency_count", Integer, default=0),
    Index("idx_freq_installed_agent", "agent_id"),
    Index("idx_freq_installed_with", "frequently_with_id"),
)


class PricingType(str, Enum):
    """Pricing model types"""
    FREE = "free"
    ONE_TIME = "one_time"
    SUBSCRIPTION_MONTHLY = "subscription_monthly"
    SUBSCRIPTION_ANNUAL = "subscription_annual"
    USAGE_BASED = "usage_based"
    FREEMIUM = "freemium"


class LicenseType(str, Enum):
    """License types"""
    MIT = "mit"
    APACHE_2 = "apache_2"
    GPL_3 = "gpl_3"
    BSD_3 = "bsd_3"
    COMMERCIAL = "commercial"
    PROPRIETARY = "proprietary"
    CREATIVE_COMMONS = "creative_commons"


class AgentStatus(str, Enum):
    """Agent publication status"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class ReviewStatus(str, Enum):
    """Review moderation status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"


class MarketplaceAgent(Base):
    """
    Main marketplace agent model.

    Represents a published agent in the marketplace with all metadata,
    pricing, ratings, and relationships to versions, reviews, etc.
    """
    __tablename__ = "marketplace_agents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    slug = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=False)
    long_description = Column(Text)  # Markdown formatted
    readme = Column(Text)  # Markdown README

    # Author information
    author_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    author_name = Column(String(255), nullable=False)
    organization = Column(String(255))

    # Categorization
    category_id = Column(Integer, ForeignKey("agent_categories.id"), index=True)
    category = relationship("AgentCategory", back_populates="agents")
    tags = relationship("AgentTagModel", secondary=agent_tags, back_populates="agents")

    # Pricing
    pricing_type = Column(String(50), nullable=False, default=PricingType.FREE.value)
    price = Column(Numeric(10, 2), default=0.00)
    currency = Column(String(3), default="USD")

    # License
    license_id = Column(Integer, ForeignKey("agent_licenses.id"))
    license = relationship("AgentLicense", back_populates="agents")

    # Statistics
    downloads = Column(Integer, default=0, index=True)
    installs = Column(Integer, default=0)
    rating_avg = Column(Float, default=0.0, index=True)
    rating_count = Column(Integer, default=0)
    rating_1_count = Column(Integer, default=0)
    rating_2_count = Column(Integer, default=0)
    rating_3_count = Column(Integer, default=0)
    rating_4_count = Column(Integer, default=0)
    rating_5_count = Column(Integer, default=0)
    wilson_score = Column(Float, default=0.0, index=True)  # For ranking

    # Revenue (for paid agents)
    total_revenue = Column(Numeric(12, 2), default=0.00)
    platform_fee_percent = Column(Integer, default=20)

    # Status and visibility
    status = Column(String(50), nullable=False, default=AgentStatus.DRAFT.value, index=True)
    featured = Column(Boolean, default=False, index=True)
    verified = Column(Boolean, default=False, index=True)
    recommended = Column(Boolean, default=False)

    # Metadata
    homepage_url = Column(String(500))
    repository_url = Column(String(500))
    documentation_url = Column(String(500))
    support_url = Column(String(500))
    demo_url = Column(String(500))

    # Compatibility
    min_greenlang_version = Column(String(50))
    max_greenlang_version = Column(String(50))
    python_requires = Column(String(50))

    # Search
    search_vector = Column(TSVECTOR)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime, index=True)
    last_version_at = Column(DateTime)

    # Relationships
    versions = relationship("AgentVersion", back_populates="agent", cascade="all, delete-orphan")
    reviews = relationship("AgentReview", back_populates="agent", cascade="all, delete-orphan")
    assets = relationship("AgentAsset", back_populates="agent", cascade="all, delete-orphan")
    installs_rel = relationship("AgentInstall", back_populates="agent", cascade="all, delete-orphan")
    purchases = relationship("AgentPurchase", back_populates="agent", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_agent_search", "search_vector", postgresql_using="gin"),
        Index("idx_agent_status_featured", "status", "featured"),
        Index("idx_agent_category_rating", "category_id", "rating_avg"),
        Index("idx_agent_created", "created_at"),
        Index("idx_agent_downloads", "downloads"),
    )

    @hybrid_property
    def current_version(self) -> Optional[str]:
        """Get the latest published version"""
        if self.versions:
            latest = max(
                (v for v in self.versions if not v.deprecated),
                key=lambda v: v.published_at,
                default=None
            )
            return latest.version if latest else None
        return None

    @hybrid_property
    def author_revenue(self) -> Decimal:
        """Calculate author revenue after platform fee"""
        if self.total_revenue:
            fee_multiplier = (100 - self.platform_fee_percent) / 100
            return self.total_revenue * Decimal(fee_multiplier)
        return Decimal(0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "author_id": str(self.author_id),
            "author_name": self.author_name,
            "organization": self.organization,
            "category": self.category.name if self.category else None,
            "tags": [tag.name for tag in self.tags],
            "pricing_type": self.pricing_type,
            "price": FinancialDecimal.from_string(self.price) if self.price else 0,
            "currency": self.currency,
            "downloads": self.downloads,
            "rating_avg": self.rating_avg,
            "rating_count": self.rating_count,
            "featured": self.featured,
            "verified": self.verified,
            "status": self.status,
            "current_version": self.current_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentVersion(Base):
    """
    Agent version model.

    Stores different versions of an agent with changelog, dependencies,
    compatibility information, and the actual code/package.
    """
    __tablename__ = "agent_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_agents.id"), nullable=False, index=True)

    version = Column(String(50), nullable=False)
    version_major = Column(Integer, nullable=False)
    version_minor = Column(Integer, nullable=False)
    version_patch = Column(Integer, nullable=False)

    changelog = Column(Text)
    release_notes = Column(Text)

    # Code
    code_hash = Column(String(64), nullable=False)  # SHA-256
    package_url = Column(String(500))  # S3/storage URL
    package_size = Column(Integer)  # bytes

    # Dependencies
    dependencies_json = Column(JSON)  # {agent_id: version_constraint}
    python_dependencies = Column(JSON)  # {package: version}

    # Compatibility
    min_greenlang_version = Column(String(50))
    max_greenlang_version = Column(String(50))
    breaking_changes = Column(Boolean, default=False)
    migration_guide = Column(Text)

    # Metadata
    schema_input = Column(JSON)  # JSON Schema for inputs
    schema_output = Column(JSON)  # JSON Schema for outputs

    # Status
    deprecated = Column(Boolean, default=False, index=True)
    deprecated_reason = Column(Text)
    superseded_by = Column(String(50))  # Version that supersedes this

    # Performance metrics (from validation)
    avg_execution_time_ms = Column(Float)
    max_memory_mb = Column(Float)
    max_cpu_percent = Column(Float)

    # Downloads for this specific version
    downloads = Column(Integer, default=0)

    # Timestamps
    published_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    deprecated_at = Column(DateTime)

    # Relationships
    agent = relationship("MarketplaceAgent", back_populates="versions")
    dependencies = relationship("AgentDependency", back_populates="version", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("agent_id", "version", name="uq_agent_version"),
        Index("idx_version_agent_published", "agent_id", "published_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "version": self.version,
            "changelog": self.changelog,
            "release_notes": self.release_notes,
            "breaking_changes": self.breaking_changes,
            "deprecated": self.deprecated,
            "downloads": self.downloads,
            "published_at": self.published_at.isoformat() if self.published_at else None,
        }


class AgentReview(Base):
    """
    Agent review and rating model.

    Stores user reviews with ratings, helpful votes, and moderation status.
    """
    __tablename__ = "agent_reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_agents.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    rating = Column(Integer, nullable=False)  # 1-5
    title = Column(String(255))
    review_text = Column(Text)

    # Verification
    verified_purchase = Column(Boolean, default=False)
    verified_install = Column(Boolean, default=False)

    # Helpful votes
    helpful_count = Column(Integer, default=0)
    not_helpful_count = Column(Integer, default=0)

    # Moderation
    status = Column(String(50), default=ReviewStatus.PENDING.value, index=True)
    flagged_count = Column(Integer, default=0)
    flag_reasons = Column(JSON)
    moderation_notes = Column(Text)

    # Response from author
    author_response = Column(Text)
    author_response_at = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    agent = relationship("MarketplaceAgent", back_populates="reviews")

    __table_args__ = (
        UniqueConstraint("agent_id", "user_id", name="uq_agent_user_review"),
        CheckConstraint("rating >= 1 AND rating <= 5", name="check_rating_range"),
        Index("idx_review_agent_status", "agent_id", "status"),
        Index("idx_review_helpful", "helpful_count"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "user_id": str(self.user_id),
            "rating": self.rating,
            "title": self.title,
            "review_text": self.review_text,
            "verified_purchase": self.verified_purchase,
            "helpful_count": self.helpful_count,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AgentCategory(Base):
    """
    Agent category model with hierarchical support.

    Supports parent-child relationships for nested categories.
    """
    __tablename__ = "agent_categories"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    icon = Column(String(100))  # Icon name/class

    # Hierarchy
    parent_id = Column(Integer, ForeignKey("agent_categories.id"), index=True)
    parent = relationship("AgentCategory", remote_side=[id], back_populates="children")
    children = relationship("AgentCategory", back_populates="parent")

    # Order and display
    display_order = Column(Integer, default=0)
    visible = Column(Boolean, default=True)

    # Statistics
    agent_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    agents = relationship("MarketplaceAgent", back_populates="category")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "icon": self.icon,
            "parent_id": self.parent_id,
            "agent_count": self.agent_count,
        }


class AgentTagModel(Base):
    """
    Tag model for agent categorization.

    Provides flexible tagging system for agents.
    """
    __tablename__ = "agent_tags_table"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False, index=True)
    slug = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(Text)

    # Statistics
    usage_count = Column(Integer, default=0, index=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    agents = relationship("MarketplaceAgent", secondary=agent_tags, back_populates="tags")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "usage_count": self.usage_count,
        }


class AgentAsset(Base):
    """
    Agent asset model for icons, screenshots, videos, etc.

    Stores URLs and metadata for agent media assets.
    """
    __tablename__ = "agent_assets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_agents.id"), nullable=False, index=True)
    version = Column(String(50))  # Optional: asset for specific version

    file_type = Column(String(50), nullable=False)  # icon, screenshot, video, demo
    mime_type = Column(String(100))
    file_path = Column(String(500), nullable=False)  # S3/storage path
    file_url = Column(String(500), nullable=False)  # Public URL

    size = Column(Integer)  # bytes
    checksum = Column(String(64))  # SHA-256

    # Metadata
    title = Column(String(255))
    description = Column(Text)
    display_order = Column(Integer, default=0)

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    agent = relationship("MarketplaceAgent", back_populates="assets")

    __table_args__ = (
        Index("idx_asset_agent_type", "agent_id", "file_type"),
    )


class AgentDependency(Base):
    """
    Agent dependency model.

    Tracks dependencies between agents with version constraints.
    """
    __tablename__ = "agent_dependencies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    version = Column(String(50), nullable=False)  # Version that has this dependency
    version_id = Column(UUID(as_uuid=True), ForeignKey("agent_versions.id"), index=True)

    dependency_agent_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_agents.id"), nullable=False, index=True)
    version_constraint = Column(String(100), nullable=False)  # e.g., ">=1.0.0,<2.0.0"

    # Dependency metadata
    optional = Column(Boolean, default=False)
    extras = Column(String(100))  # Optional extras/features

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    version = relationship("AgentVersion", back_populates="dependencies")
    dependency_agent = relationship("MarketplaceAgent", foreign_keys=[dependency_agent_id])

    __table_args__ = (
        Index("idx_dependency_agent_version", "agent_id", "version"),
        Index("idx_dependency_target", "dependency_agent_id"),
    )


class AgentLicense(Base):
    """
    Agent license model.

    Defines license types and terms for agents.
    """
    __tablename__ = "agent_licenses"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    license_type = Column(String(50), nullable=False)  # LicenseType enum

    short_name = Column(String(50))
    url = Column(String(500))
    terms = Column(Text)

    # Permissions
    commercial_allowed = Column(Boolean, default=True)
    modification_allowed = Column(Boolean, default=True)
    distribution_allowed = Column(Boolean, default=True)
    patent_grant = Column(Boolean, default=False)

    # Requirements
    requires_attribution = Column(Boolean, default=True)
    requires_same_license = Column(Boolean, default=False)
    requires_state_changes = Column(Boolean, default=False)
    requires_source = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    agents = relationship("MarketplaceAgent", back_populates="license")


class AgentInstall(Base):
    """
    Agent installation tracking.

    Records when users install agents for analytics and verification.
    """
    __tablename__ = "agent_installs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_agents.id"), nullable=False, index=True)
    version = Column(String(50), nullable=False)

    # Installation metadata
    installation_id = Column(String(100), unique=True)  # Unique per installation
    platform = Column(String(50))  # OS/platform
    python_version = Column(String(50))
    greenlang_version = Column(String(50))

    # Status
    active = Column(Boolean, default=True, index=True)
    uninstalled_at = Column(DateTime)

    # Timestamps
    installed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    last_used_at = Column(DateTime)

    # Relationships
    agent = relationship("MarketplaceAgent", back_populates="installs_rel")

    __table_args__ = (
        Index("idx_install_user_agent", "user_id", "agent_id"),
        Index("idx_install_active", "active", "installed_at"),
    )


class AgentPurchase(Base):
    """
    Agent purchase model.

    Records purchases for paid agents with payment details.
    """
    __tablename__ = "agent_purchases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_agents.id"), nullable=False, index=True)

    # Payment details
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    transaction_id = Column(String(255), unique=True, nullable=False, index=True)
    payment_method = Column(String(50))

    # Stripe integration
    stripe_payment_intent_id = Column(String(255), unique=True)
    stripe_customer_id = Column(String(255), index=True)

    # Pricing type
    pricing_type = Column(String(50), nullable=False)
    subscription_id = Column(String(255))  # For subscriptions
    subscription_period_start = Column(DateTime)
    subscription_period_end = Column(DateTime)

    # Status
    status = Column(String(50), nullable=False, default="completed")  # completed, refunded, disputed
    refunded_at = Column(DateTime)
    refund_amount = Column(Numeric(10, 2))
    refund_reason = Column(Text)

    # License
    license_key = Column(String(255), unique=True, index=True)

    # Timestamps
    purchased_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    agent = relationship("MarketplaceAgent", back_populates="purchases")

    __table_args__ = (
        Index("idx_purchase_user_agent", "user_id", "agent_id"),
        Index("idx_purchase_stripe", "stripe_payment_intent_id"),
        Index("idx_purchase_status", "status", "purchased_at"),
    )


class AgentSearchHistory(Base):
    """
    Search history tracking.

    Records user search queries for analytics and suggestions.
    """
    __tablename__ = "agent_search_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), index=True)  # Optional: for logged-in users

    query = Column(String(500), nullable=False, index=True)
    filters = Column(JSON)  # Search filters applied
    results_count = Column(Integer)

    # User interaction
    clicked_agent_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_agents.id"))
    clicked_position = Column(Integer)  # Position in results

    # Timestamps
    searched_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("idx_search_query", "query"),
        Index("idx_search_user_time", "user_id", "searched_at"),
    )


class AgentImpression(Base):
    """
    Agent impression tracking.

    Records when agents are viewed for analytics.
    """
    __tablename__ = "agent_impressions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("marketplace_agents.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), index=True)  # Optional

    # Context
    source = Column(String(100))  # search, category, featured, recommendation
    position = Column(Integer)  # Position in list

    # Session
    session_id = Column(String(255), index=True)

    # Timestamps
    viewed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("idx_impression_agent_time", "agent_id", "viewed_at"),
        Index("idx_impression_source", "source", "viewed_at"),
    )


# Helper functions for database initialization
def init_default_categories(session):
    """Initialize default category hierarchy"""
    categories = [
        # Data Processing
        {"name": "Data Processing", "slug": "data-processing", "icon": "database", "parent": None},
        {"name": "CSV/Excel Processing", "slug": "csv-excel", "icon": "table", "parent": "Data Processing"},
        {"name": "JSON/XML Processing", "slug": "json-xml", "icon": "code", "parent": "Data Processing"},
        {"name": "Data Validation", "slug": "data-validation", "icon": "check-circle", "parent": "Data Processing"},
        {"name": "Data Transformation", "slug": "data-transformation", "icon": "shuffle", "parent": "Data Processing"},

        # AI/ML
        {"name": "AI/ML", "slug": "ai-ml", "icon": "brain", "parent": None},
        {"name": "Natural Language Processing", "slug": "nlp", "icon": "message-square", "parent": "AI/ML"},
        {"name": "Computer Vision", "slug": "computer-vision", "icon": "eye", "parent": "AI/ML"},
        {"name": "Time Series Analysis", "slug": "time-series", "icon": "trending-up", "parent": "AI/ML"},
        {"name": "Reinforcement Learning", "slug": "reinforcement-learning", "icon": "zap", "parent": "AI/ML"},

        # Integration
        {"name": "Integration", "slug": "integration", "icon": "link", "parent": None},
        {"name": "APIs", "slug": "apis", "icon": "cloud", "parent": "Integration"},
        {"name": "Databases", "slug": "databases", "icon": "server", "parent": "Integration"},
        {"name": "Cloud Services", "slug": "cloud-services", "icon": "cloud-upload", "parent": "Integration"},
        {"name": "Messaging", "slug": "messaging", "icon": "mail", "parent": "Integration"},

        # DevOps
        {"name": "DevOps", "slug": "devops", "icon": "settings", "parent": None},
        {"name": "Monitoring", "slug": "monitoring", "icon": "activity", "parent": "DevOps"},
        {"name": "Logging", "slug": "logging", "icon": "file-text", "parent": "DevOps"},
        {"name": "Deployment", "slug": "deployment", "icon": "upload-cloud", "parent": "DevOps"},
        {"name": "CI/CD", "slug": "ci-cd", "icon": "git-branch", "parent": "DevOps"},

        # Business
        {"name": "Business", "slug": "business", "icon": "briefcase", "parent": None},
        {"name": "Accounting", "slug": "accounting", "icon": "dollar-sign", "parent": "Business"},
        {"name": "CRM Integration", "slug": "crm", "icon": "users", "parent": "Business"},
        {"name": "E-commerce", "slug": "ecommerce", "icon": "shopping-cart", "parent": "Business"},
        {"name": "Analytics", "slug": "analytics", "icon": "bar-chart", "parent": "Business"},

        # Utilities
        {"name": "Utilities", "slug": "utilities", "icon": "tool", "parent": None},
        {"name": "Date/Time", "slug": "date-time", "icon": "calendar", "parent": "Utilities"},
        {"name": "Math/Statistics", "slug": "math-stats", "icon": "hash", "parent": "Utilities"},
        {"name": "File System", "slug": "file-system", "icon": "folder", "parent": "Utilities"},
        {"name": "Networking", "slug": "networking", "icon": "globe", "parent": "Utilities"},
    ]

    category_map = {}

    for cat_data in categories:
        if cat_data["parent"] is None:
            cat = AgentCategory(
                name=cat_data["name"],
                slug=cat_data["slug"],
                icon=cat_data["icon"],
            )
            session.add(cat)
            session.flush()
            category_map[cat_data["name"]] = cat.id

    # Add children
    for cat_data in categories:
        if cat_data["parent"] is not None:
            parent_id = category_map.get(cat_data["parent"])
            if parent_id:
                cat = AgentCategory(
                    name=cat_data["name"],
                    slug=cat_data["slug"],
                    icon=cat_data["icon"],
                    parent_id=parent_id,
                )
                session.add(cat)

    session.commit()


def init_default_licenses(session):
    """Initialize default licenses"""
    licenses = [
        {
            "name": "MIT License",
            "license_type": LicenseType.MIT.value,
            "short_name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
            "commercial_allowed": True,
            "modification_allowed": True,
            "distribution_allowed": True,
            "requires_attribution": True,
        },
        {
            "name": "Apache License 2.0",
            "license_type": LicenseType.APACHE_2.value,
            "short_name": "Apache-2.0",
            "url": "https://opensource.org/licenses/Apache-2.0",
            "commercial_allowed": True,
            "modification_allowed": True,
            "distribution_allowed": True,
            "patent_grant": True,
            "requires_attribution": True,
            "requires_state_changes": True,
        },
        {
            "name": "GNU General Public License v3.0",
            "license_type": LicenseType.GPL_3.value,
            "short_name": "GPL-3.0",
            "url": "https://www.gnu.org/licenses/gpl-3.0.en.html",
            "commercial_allowed": True,
            "modification_allowed": True,
            "distribution_allowed": True,
            "requires_attribution": True,
            "requires_same_license": True,
            "requires_source": True,
        },
        {
            "name": "BSD 3-Clause License",
            "license_type": LicenseType.BSD_3.value,
            "short_name": "BSD-3-Clause",
            "url": "https://opensource.org/licenses/BSD-3-Clause",
            "commercial_allowed": True,
            "modification_allowed": True,
            "distribution_allowed": True,
            "requires_attribution": True,
        },
        {
            "name": "Commercial License",
            "license_type": LicenseType.COMMERCIAL.value,
            "short_name": "Commercial",
            "commercial_allowed": True,
            "modification_allowed": False,
            "distribution_allowed": False,
            "requires_attribution": False,
        },
    ]

    for lic_data in licenses:
        lic = AgentLicense(**lic_data)
        session.add(lic)

    session.commit()
