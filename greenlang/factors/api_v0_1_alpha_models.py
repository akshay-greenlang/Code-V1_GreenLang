# -*- coding: utf-8 -*-
"""GreenLang Factors v0.1 Alpha — Pydantic v2 response models.

Mirrors the contract in
``config/schemas/factor_record_v0_1.schema.json`` (FROZEN 2026-04-25)
but is intentionally *less strict* than the JSON Schema:

* fields are typed with ``Optional`` / ``str`` rather than enum/regex
  constraints because the alpha catalog is back-fed from the legacy
  ``EmissionFactorRecord`` shape; aggressive validation here would
  cause 500s on records that pass JSON-Schema validation in the
  publisher pipeline but lack one of the v0.1 required keys at the
  service boundary.
* the router never calls ``.model_dump()`` on inbound objects —
  these models are response-side only.

The real schema gate (jsonschema Draft202012Validator) is owned by
``tests/factors/v0_1_alpha/test_factor_record_v0_1_schema_loads.py``;
this module does not duplicate that work.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


__all__ = [
    "ALPHA_CATEGORY_ENUM",
    "FactorV0_1",
    "FactorListResponse",
    "ErrorResponse",
    "HealthzResponse",
    "SourceV0_1",
    "SourceListResponse",
    "PackV0_1",
    "PackListResponse",
]


# Per CTO doc §19.1 / factor_record_v0_1.schema.json#category.enum.
ALPHA_CATEGORY_ENUM: List[str] = [
    "scope1",
    "scope2_location_based",
    "scope2_market_based",
    "grid_intensity",
    "fuel",
    "refrigerant",
    "fugitive",
    "process",
    "cbam_default",
]


class _AlphaBase(BaseModel):
    """Common config — accept extra keys silently so legacy records flow."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class FactorV0_1(_AlphaBase):
    """Single factor record in the v0.1 alpha shape.

    Keys mirror the JSON-Schema required list. Values are best-effort
    coerced from the legacy ``EmissionFactorRecord`` via
    :func:`greenlang.factors.api_v0_1_alpha_routes._coerce_v0_1`. None is
    permitted on every field except ``urn`` so a missing back-fill never
    crashes ``/v1/factors``.
    """

    urn: str = Field(..., description="Canonical factor URN.")
    factor_id_alias: Optional[str] = None
    source_urn: Optional[str] = None
    factor_pack_urn: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    value: Optional[float] = None
    unit_urn: Optional[str] = None
    gwp_basis: Optional[str] = None
    gwp_horizon: Optional[int] = None
    geography_urn: Optional[str] = None
    vintage_start: Optional[str] = None
    vintage_end: Optional[str] = None
    resolution: Optional[str] = None
    methodology_urn: Optional[str] = None
    boundary: Optional[str] = None
    licence: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    published_at: Optional[str] = None
    extraction: Dict[str, Any] = Field(default_factory=dict)
    review: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class FactorListResponse(_AlphaBase):
    """Cursor-paginated list response for ``GET /v1/factors``."""

    data: List[FactorV0_1] = Field(default_factory=list)
    next_cursor: Optional[str] = None
    edition: Optional[str] = None


class ErrorResponse(_AlphaBase):
    """Stable error payload shape for the alpha surface."""

    error: str
    message: str
    urn: Optional[str] = None
    allowed: Optional[List[str]] = None


class HealthzResponse(_AlphaBase):
    """Response shape for ``GET /v1/healthz``."""

    status: str
    service: str
    release_profile: str
    schema_id: str
    edition: Optional[str] = None
    git_commit: Optional[str] = None
    version: str = "0.1.0"


class SourceV0_1(_AlphaBase):
    """Public source registry entry in the v0.1 shape."""

    urn: str
    source_id: str
    display_name: Optional[str] = None
    publisher: Optional[str] = None
    jurisdiction: Optional[Union[str, List[str]]] = None
    license_class: Optional[str] = None
    cadence: Optional[str] = None
    publication_url: Optional[str] = None
    citation_text: Optional[str] = None
    source_version: Optional[str] = None
    latest_ingestion_at: Optional[str] = None
    provenance_completeness_score: Optional[float] = None


class SourceListResponse(_AlphaBase):
    """Response shape for ``GET /v1/sources``."""

    data: List[SourceV0_1] = Field(default_factory=list)
    count: int = 0


class PackV0_1(_AlphaBase):
    """Factor pack (alpha shape) — derived from source registry + factor counts."""

    urn: str
    source_urn: str
    pack_id: str
    version: str
    display_name: Optional[str] = None
    factor_count: Optional[int] = None


class PackListResponse(_AlphaBase):
    """Response shape for ``GET /v1/packs``."""

    data: List[PackV0_1] = Field(default_factory=list)
    count: int = 0
