# -*- coding: utf-8 -*-
"""
OEM white-label branding configuration (Track C-5).

Defines the :class:`BrandingConfig` Pydantic model that an OEM partner can
attach to its tenant. The branding payload decorates outbound API
responses (see :func:`greenlang.factors.tenant_overlay.apply_oem_branding`)
so a third-party platform can present GreenLang Factors data under its
own logo / colour scheme / support contact while still being clearly
"powered by GreenLang" when attribution is required.

Validation rules deliberately mirror the CTO spec section "For platforms":

* ``primary_color`` / ``secondary_color`` must be 7-character hex strings
  (``#RRGGBB``). We normalise the casing to lower so digests over the
  branding blob are stable.
* ``logo_url`` and ``support_url`` must be HTTPS - we never advertise
  HTTP assets through a hosted API response.
* ``custom_domain`` (if set) must be a hostname (no scheme, no path).
* ``support_email`` is parsed as an :class:`EmailStr` so invalid values
  fail at deserialisation time instead of at first use.
* ``attribution_required`` defaults to ``True`` to preserve the
  "Powered by GreenLang" badge unless a contract-grade override is
  explicitly granted by the operator.

The model is fully immutable (``model_config["frozen"] = True``) so a
caller can safely cache a hashed branding blob without worrying about
mutation between writes.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compiled regexes - class-level constants so they are JIT'd once.
# ---------------------------------------------------------------------------

_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)"
    r"(\.([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?))+$"
)


def _validate_https_url(value: Optional[str], *, field_name: str) -> Optional[str]:
    """Allow ``None`` or any HTTPS URL; reject HTTP for hosted-API surfaces."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError("%s must be a string" % field_name)
    if not value.startswith("https://"):
        raise ValueError(
            "%s must be an https:// URL (got %r)" % (field_name, value[:32])
        )
    if len(value) > 2048:
        raise ValueError("%s must be <= 2048 chars" % field_name)
    return value


# ---------------------------------------------------------------------------
# BrandingConfig
# ---------------------------------------------------------------------------


class BrandingConfig(BaseModel):
    """White-label branding payload for an OEM partner.

    Every field is optional because an OEM may choose to brand only a
    subset of the response surface (e.g. logo + support email but no
    custom domain). The defaults keep the canonical GreenLang attribution
    in place, which is the legally safe fallback.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",  # reject unknown keys so typos surface in tests
        str_strip_whitespace=True,
    )

    logo_url: Optional[str] = Field(
        default=None,
        description="HTTPS URL to the OEM logo asset (SVG / PNG).",
        max_length=2048,
    )
    primary_color: Optional[str] = Field(
        default=None,
        description="Primary brand colour as #RRGGBB hex.",
        examples=["#0A66C2"],
    )
    secondary_color: Optional[str] = Field(
        default=None,
        description="Secondary brand colour as #RRGGBB hex.",
        examples=["#FFFFFF"],
    )
    support_email: Optional[EmailStr] = Field(
        default=None,
        description="OEM-side support email shown in API error envelopes.",
    )
    support_url: Optional[str] = Field(
        default=None,
        description="HTTPS URL to the OEM support / help-centre site.",
        max_length=2048,
    )
    custom_domain: Optional[str] = Field(
        default=None,
        description=(
            "Bare hostname the OEM serves the Factors UI from "
            "(e.g. ``factors.acme.com``). No scheme / path."
        ),
        max_length=253,
    )
    attribution_required: bool = Field(
        default=True,
        description=(
            "If True, responses MUST advertise the powered-by text. "
            "Only legal-signed contracts can flip this to False."
        ),
    )
    powered_by_text: str = Field(
        default="Powered by GreenLang",
        description=(
            "Attribution text rendered alongside the OEM logo when "
            "``attribution_required`` is True."
        ),
        max_length=128,
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("primary_color", "secondary_color")
    @classmethod
    def _validate_hex_color(cls, v: Optional[str]) -> Optional[str]:
        """Validate #RRGGBB hex format and normalise to lower-case."""
        if v is None or v == "":
            return None
        if not _HEX_COLOR_RE.match(v):
            raise ValueError(
                "must be a 7-char hex colour like '#0A66C2', got %r" % v
            )
        return v.lower()

    @field_validator("logo_url")
    @classmethod
    def _validate_logo_url(cls, v: Optional[str]) -> Optional[str]:
        return _validate_https_url(v, field_name="logo_url")

    @field_validator("support_url")
    @classmethod
    def _validate_support_url(cls, v: Optional[str]) -> Optional[str]:
        return _validate_https_url(v, field_name="support_url")

    @field_validator("custom_domain")
    @classmethod
    def _validate_custom_domain(cls, v: Optional[str]) -> Optional[str]:
        """Validate hostname-only (no scheme, no path)."""
        if v is None or v == "":
            return None
        if "://" in v or "/" in v:
            raise ValueError(
                "custom_domain must be a bare hostname (no scheme/path), got %r" % v
            )
        if not _HOSTNAME_RE.match(v):
            raise ValueError(
                "custom_domain must be a valid DNS hostname, got %r" % v
            )
        return v.lower()

    @field_validator("powered_by_text")
    @classmethod
    def _validate_powered_by(cls, v: str) -> str:
        """Disallow empty / whitespace-only attribution text."""
        if not v or not v.strip():
            raise ValueError("powered_by_text must not be empty")
        return v

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_response_metadata(self) -> Dict[str, Any]:
        """Serialise the branding payload for embedding in API responses.

        We deliberately omit ``None`` values so a minimally-configured
        OEM does not produce a noisy response envelope.
        """
        raw = self.model_dump(mode="json")
        return {k: v for k, v in raw.items() if v is not None}

    @classmethod
    def empty(cls) -> "BrandingConfig":
        """Return the canonical 'no overrides' branding payload."""
        return cls()


__all__ = ["BrandingConfig"]
