# -*- coding: utf-8 -*-
"""
GreenLang Shared Field Factories
==================================

Pre-configured ``Field(...)`` factories for commonly-duplicated field
definitions. Import these instead of re-defining the same field pattern
in every agent model.

Usage::

    from greenlang.schemas.fields import (
        id_field, tenant_field, provenance_field,
        created_at_field, updated_at_field,
    )

    class MyModel(GreenLangBase):
        record_id: str = id_field(description="Record identifier")
        tenant_id: str = tenant_field()
        provenance_hash: str = provenance_field()
        created_at: datetime = created_at_field()

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import Field
from pydantic.fields import FieldInfo

from greenlang.schemas.base import new_uuid, utcnow


# =============================================================================
# ID Fields
# =============================================================================


def id_field(
    *,
    description: str = "Unique identifier",
    prefix: str = "",
) -> Any:
    """Create a UUID-based identifier field.

    Args:
        description: Field description for docs/schema.
        prefix: Optional prefix (e.g. "ef", "doc"). If empty, uses full UUID4.

    Returns:
        Pydantic Field with default_factory for UUID generation.
    """
    if prefix:
        import uuid as _uuid

        return Field(
            default_factory=lambda: f"{prefix}_{_uuid.uuid4().hex[:12]}",
            description=description,
        )
    return Field(
        default_factory=new_uuid,
        description=description,
    )


# =============================================================================
# Tenant / Organization Fields
# =============================================================================


def tenant_field(
    *,
    default: str = "default",
    description: str = "Tenant identifier for multi-tenant isolation",
) -> Any:
    """Create a tenant_id field with standard defaults."""
    return Field(default=default, description=description)


def organization_field(
    *,
    default: str = "",
    description: str = "Organization identifier",
) -> Any:
    """Create an organization_id field."""
    return Field(default=default, description=description)


# =============================================================================
# Timestamp Fields
# =============================================================================


def created_at_field(
    *,
    description: str = "Timestamp when this record was created (UTC)",
) -> Any:
    """Create a created_at timestamp field defaulting to utcnow()."""
    return Field(default_factory=utcnow, description=description)


def updated_at_field(
    *,
    description: str = "Timestamp when this record was last updated (UTC)",
) -> Any:
    """Create an updated_at timestamp field defaulting to utcnow()."""
    return Field(default_factory=utcnow, description=description)


def started_at_field(
    *,
    description: str = "Timestamp when processing started (UTC)",
) -> Any:
    """Create a started_at timestamp field for job/process tracking."""
    return Field(default_factory=utcnow, description=description)


def completed_at_field(
    *,
    description: str = "Timestamp when processing completed (UTC)",
) -> Any:
    """Create an optional completed_at timestamp field."""
    return Field(default=None, description=description)


# =============================================================================
# Provenance / Audit Fields
# =============================================================================


def provenance_field(
    *,
    description: str = "SHA-256 provenance chain hash for audit trail",
) -> Any:
    """Create a provenance_hash field with empty default."""
    return Field(default="", description=description)


def metadata_field(
    *,
    description: str = "Extensible key-value metadata",
) -> Any:
    """Create a metadata dict field."""
    return Field(default_factory=dict, description=description)


# =============================================================================
# Job / Processing Fields
# =============================================================================


def duration_ms_field(
    *,
    description: str = "Processing duration in milliseconds",
) -> Any:
    """Create a duration_ms float field with ge=0 constraint."""
    return Field(default=0.0, ge=0.0, description=description)


def error_message_field(
    *,
    description: str = "Error description if processing failed",
) -> Any:
    """Create an optional error_message field."""
    return Field(default=None, description=description)


def version_field(
    *,
    default: str = "1.0.0",
    description: str = "Schema or service version",
) -> Any:
    """Create a version string field."""
    return Field(default=default, description=description)
