# -*- coding: utf-8 -*-
"""
GreenLang Shared Schema Base Classes
=====================================

Pydantic v2 base models, mixins, and field factories that eliminate duplication
across 1,100+ agent model files. Every agent model should inherit from one of
these base classes instead of raw ``BaseModel``.

Architecture::

    BaseModel (pydantic)
    └── GreenLangBase              # model_config, json_encoders, orjson support
        ├── TimestampMixin         # created_at, updated_at
        ├── TenantMixin            # tenant_id
        ├── ProvenanceMixin        # provenance_hash
        ├── AuditMixin             # created_by, updated_by + Timestamp
        └── Pre-composed bases:
            ├── GreenLangRecord    # Base + Timestamp + Tenant + Provenance
            ├── GreenLangRequest   # Base (requests don't need audit)
            └── GreenLangResponse  # Base + Timestamp + Provenance

Usage::

    from greenlang.schemas import GreenLangRecord, new_uuid, utcnow

    class DocumentRecord(GreenLangRecord):
        document_id: str = Field(default_factory=new_uuid, ...)
        file_name: str = Field(...)

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Utility Functions (replaces 91+ _utcnow() duplications)
# =============================================================================


def utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed.

    This is the canonical timestamp factory for all GreenLang models.
    Zeroing microseconds ensures deterministic serialization for
    provenance hashing and cross-system comparisons.

    Returns:
        datetime: Current UTC time with microsecond=0.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


def new_uuid() -> str:
    """Generate a new UUID4 string.

    Returns:
        str: A new UUID4 as a string.
    """
    return str(uuid.uuid4())


def prefixed_uuid(prefix: str) -> str:
    """Generate a prefixed short UUID for domain-specific identifiers.

    Args:
        prefix: Short prefix (e.g. "ef", "eq", "doc").

    Returns:
        str: ``"{prefix}_{12_hex_chars}"``
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def compute_provenance_hash(*values: str) -> str:
    """Compute SHA-256 provenance hash from ordered string values.

    Used for deterministic audit trail hashing across all agents.
    Values are joined with ``|`` separator before hashing.

    Args:
        *values: Ordered string values to hash.

    Returns:
        str: Hex-encoded SHA-256 digest.
    """
    payload = "|".join(str(v) for v in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =============================================================================
# Core Base Model
# =============================================================================


class GreenLangBase(BaseModel):
    """Root base model for all GreenLang Pydantic schemas.

    Provides:
    - ``extra = "forbid"`` to catch typos in field names
    - ``validate_default = True`` for default value validation
    - ``json_encoders`` for Decimal serialization
    - ``from_attributes = True`` for ORM compatibility
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_encoders={Decimal: str},
        from_attributes=True,
        ser_json_timedelta="float",
    )


# =============================================================================
# Field Mixins
# =============================================================================


class TimestampMixin(BaseModel):
    """Mixin adding ``created_at`` and ``updated_at`` timestamp fields.

    Both default to ``utcnow()`` at instantiation time. ``updated_at``
    should be refreshed on every mutation via business logic.
    """

    created_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp when this record was created (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp when this record was last updated (UTC)",
    )


class TenantMixin(BaseModel):
    """Mixin adding ``tenant_id`` for multi-tenant isolation.

    Defaults to ``"default"`` for single-tenant deployments.
    """

    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )


class ProvenanceMixin(BaseModel):
    """Mixin adding ``provenance_hash`` for SHA-256 audit trails.

    Every record in a GreenLang pipeline carries a provenance hash
    that chains to the previous step's hash, enabling end-to-end
    audit trail verification for regulatory compliance.
    """

    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )


class AuditMixin(TimestampMixin):
    """Mixin adding full audit fields: timestamps + actor tracking.

    Extends ``TimestampMixin`` with ``created_by`` and ``updated_by``
    fields for tracking which user or system actor made changes.
    """

    created_by: str = Field(
        default="system",
        description="User or system actor that created this record",
    )
    updated_by: str = Field(
        default="system",
        description="User or system actor that last updated this record",
    )


class MetadataMixin(BaseModel):
    """Mixin adding a flexible ``metadata`` dict for extensible attributes.

    Use this for agent-specific or pipeline-specific key-value pairs
    that don't warrant dedicated fields.
    """

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value metadata",
    )


# =============================================================================
# Pre-composed Base Classes
# =============================================================================


class GreenLangRecord(GreenLangBase, TimestampMixin, TenantMixin, ProvenanceMixin):
    """Full-featured base for persistent data records.

    Combines: GreenLangBase + TimestampMixin + TenantMixin + ProvenanceMixin

    Use this for models that represent stored entities (documents,
    calculations, emission factors, equipment profiles, etc.).

    Example::

        class DocumentRecord(GreenLangRecord):
            document_id: str = Field(default_factory=new_uuid, ...)
            file_name: str = Field(...)
    """

    pass


class GreenLangAuditRecord(GreenLangBase, AuditMixin, TenantMixin, ProvenanceMixin):
    """Base for records requiring full actor audit trail.

    Combines: GreenLangBase + AuditMixin + TenantMixin + ProvenanceMixin

    Use this for compliance-critical records where ``created_by``
    and ``updated_by`` must be tracked alongside timestamps.
    """

    pass


class GreenLangRequest(GreenLangBase):
    """Base for API request/input models.

    Requests typically don't need timestamps or tenant info (those
    are injected by middleware). Override ``model_config`` to allow
    extras if needed for forward compatibility.
    """

    pass


class GreenLangResponse(GreenLangBase, TimestampMixin, ProvenanceMixin):
    """Base for API response/output models.

    Combines: GreenLangBase + TimestampMixin + ProvenanceMixin

    Includes timestamps and provenance for response audit trails
    but not tenant_id (handled at transport layer).
    """

    pass


class GreenLangConfig(GreenLangBase):
    """Base for configuration/settings models.

    Configuration models use ``extra = "ignore"`` to be forward-compatible
    with new config keys from newer versions.
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_default=True,
        json_encoders={Decimal: str},
        from_attributes=True,
    )


class GreenLangResult(GreenLangBase, TimestampMixin, ProvenanceMixin, MetadataMixin):
    """Base for calculation/processing result models.

    Combines: GreenLangBase + TimestampMixin + ProvenanceMixin + MetadataMixin

    Use this for agent calculation outputs, processing results, and
    analysis outcomes that need provenance and flexible metadata.
    """

    pass
