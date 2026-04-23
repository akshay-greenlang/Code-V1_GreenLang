# -*- coding: utf-8 -*-
"""Source-object v1 Pydantic model (W4-A / new).

Mirrors ``config/schemas/source_object_v1.schema.json`` field-for-field.
Every distinct upstream (authority, dataset_version) gets one :class:`SourceObject`
row; factor records reference the source via ``(source_id, source_version)``.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import List, Literal, Optional

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field, validator as field_validator  # type: ignore

    ConfigDict = dict  # type: ignore
    _PYDANTIC_V2 = False


# ---------------------------------------------------------------------------
# CTO-reversible constants (approve in enum_decisions_v1.md)
# ---------------------------------------------------------------------------

SOURCE_TYPE_ENUM: tuple[str, ...] = (
    "government",
    "standard_setter",
    "industry_body",
    "licensed_commercial",
    "customer_provided",
)

# Source-object redistribution classes — per freeze decision F2 of the gap
# report, the source object uses the brief enum (4 values differ from the
# factor-record redistribution enum). See REDISTRIBUTION_CLASS_ENUM in
# canonical_v1 for the record-side set.
SOURCE_REDISTRIBUTION_CLASS_ENUM: tuple[str, ...] = (
    "open",
    "licensed_embedded",
    "customer_private",
    "oem_redistributable",
)

VERIFICATION_STATUS_ENUM: tuple[str, ...] = (
    "unverified",
    "internal_review",
    "external_verified",
    "regulator_approved",
)


SourceType = Literal[
    "government",
    "standard_setter",
    "industry_body",
    "licensed_commercial",
    "customer_provided",
]

SourceRedistributionClass = Literal[
    "open",
    "licensed_embedded",
    "customer_private",
    "oem_redistributable",
]

VerificationStatus = Literal[
    "unverified",
    "internal_review",
    "external_verified",
    "regulator_approved",
]


class Jurisdiction(BaseModel):
    """Geographic scope an authority / source covers."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    country: str = Field(min_length=2, max_length=2, pattern=r"^[A-Z]{2}$")
    region: Optional[str] = Field(default=None, max_length=64)
    grid_region: Optional[str] = Field(default=None, max_length=64)


class ValidityPeriod(BaseModel):
    """Inclusive date range this source dataset applies to."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    from_: date = Field(alias="from")
    to: Optional[date] = None

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid", populate_by_name=True)

        @field_validator("to")
        @classmethod
        def _to_gte_from(cls, v: Optional[date], info) -> Optional[date]:
            if v is None:
                return v
            from_val = info.data.get("from_")
            if from_val is not None and v < from_val:
                raise ValueError("validity_period.to must be >= from")
            return v
    else:  # pragma: no cover

        @field_validator("to")
        def _to_gte_from_v1(cls, v, values):  # type: ignore
            from_val = values.get("from_") if isinstance(values, dict) else None
            if v is not None and from_val is not None and v < from_val:
                raise ValueError("validity_period.to must be >= from")
            return v


class SourceObject(BaseModel):
    """Canonical source object (v1) — one per (authority, dataset_version)."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    # Required
    source_id: str = Field(
        min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
    )
    authority: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=512)
    publisher: str = Field(min_length=1, max_length=256)
    jurisdiction: Jurisdiction
    dataset_version: str = Field(min_length=1, max_length=64)
    publication_date: date
    validity_period: ValidityPeriod
    ingestion_date: datetime
    source_type: SourceType
    redistribution_class: SourceRedistributionClass
    verification_status: VerificationStatus
    citation_text: str = Field(min_length=1, max_length=2048)

    # Optional
    change_log_uri: Optional[str] = None
    legal_notes: Optional[str] = Field(default=None, max_length=4096)
    license_name: Optional[str] = Field(default=None, max_length=128)
    license_url: Optional[str] = None
    attribution_required: bool = False
    attribution_text: Optional[str] = Field(default=None, max_length=1024)
    retrieval_uri: Optional[str] = None
    raw_payload_hash: Optional[str] = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    contact_email: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


def source_object_from_dict(payload: dict) -> SourceObject:
    """Parse a source-object dict (tolerant to minor legacy field names)."""
    data = dict(payload)
    # Tolerate "validity" / "valid_from/valid_to" alias
    if "validity_period" not in data:
        if "validity" in data and isinstance(data["validity"], dict):
            data["validity_period"] = data.pop("validity")
        elif "valid_from" in data:
            data["validity_period"] = {
                "from": data.pop("valid_from"),
                "to": data.pop("valid_to", None),
            }
    if _PYDANTIC_V2:
        return SourceObject.model_validate(data)
    return SourceObject.parse_obj(data)  # pragma: no cover


def source_object_to_dict(obj: SourceObject) -> dict:
    """Serialise a SourceObject to a JSON-schema-compatible dict."""
    if _PYDANTIC_V2:
        data = obj.model_dump(mode="json", by_alias=True, exclude_none=False)
    else:  # pragma: no cover
        data = obj.dict(by_alias=True)
    return data


__all__ = [
    "Jurisdiction",
    "SourceObject",
    "ValidityPeriod",
    "SourceRedistributionClass",
    "SourceType",
    "VerificationStatus",
    "SOURCE_TYPE_ENUM",
    "SOURCE_REDISTRIBUTION_CLASS_ENUM",
    "VERIFICATION_STATUS_ENUM",
    "source_object_from_dict",
    "source_object_to_dict",
]
