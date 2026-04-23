# -*- coding: utf-8 -*-
"""
Tests for OEM branding round-trip + ``apply_oem_branding`` decorator.

Covers:
* :class:`BrandingConfig` validation rules (hex colours, https-only URLs,
  bare hostname, attribution defaults).
* Round-trip via ``update_branding`` + ``get_oem_partner``.
* ``apply_oem_branding`` decorator behaviour for dict / list / None
  responses, missing OEM, and missing branding.
"""
from __future__ import annotations

from typing import Iterator

import pytest
from pydantic import ValidationError

from greenlang.factors.onboarding.branding_config import BrandingConfig
from greenlang.factors.onboarding.partner_setup import (
    _reset_oem_registry,
    create_oem_partner,
    get_oem_partner,
    update_branding,
)
from greenlang.factors.tenant_overlay import apply_oem_branding


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    _reset_oem_registry()
    yield
    _reset_oem_registry()


@pytest.fixture()
def acme_oem():
    return create_oem_partner(
        name="Acme", contact_email="a@b.com",
        redistribution_grants=["open"], parent_plan="platform",
    )


# ---------------------------------------------------------------------------
# BrandingConfig validation
# ---------------------------------------------------------------------------


def test_branding_defaults_are_safe():
    bc = BrandingConfig()
    assert bc.attribution_required is True
    assert bc.powered_by_text == "Powered by GreenLang"
    assert bc.logo_url is None
    assert bc.primary_color is None


def test_branding_accepts_full_payload():
    bc = BrandingConfig(
        logo_url="https://cdn.acme.com/logo.svg",
        primary_color="#0A66C2",
        secondary_color="#FFFFFF",
        support_email="ops@acme.example.com",
        support_url="https://acme.example.com/support",
        custom_domain="factors.acme.com",
        attribution_required=True,
        powered_by_text="Powered by GreenLang",
    )
    assert bc.primary_color == "#0a66c2"  # normalised lower-case
    assert bc.secondary_color == "#ffffff"
    assert bc.custom_domain == "factors.acme.com"


def test_branding_rejects_http_logo_url():
    with pytest.raises(ValidationError):
        BrandingConfig(logo_url="http://insecure.example.com/logo.png")


def test_branding_rejects_invalid_hex_color():
    with pytest.raises(ValidationError):
        BrandingConfig(primary_color="blue")


def test_branding_rejects_short_hex_color():
    """The 3-char form (#abc) is intentionally rejected to keep digests stable."""
    with pytest.raises(ValidationError):
        BrandingConfig(primary_color="#abc")


def test_branding_rejects_custom_domain_with_scheme():
    with pytest.raises(ValidationError):
        BrandingConfig(custom_domain="https://factors.acme.com")


def test_branding_rejects_custom_domain_with_path():
    with pytest.raises(ValidationError):
        BrandingConfig(custom_domain="factors.acme.com/path")


def test_branding_rejects_blank_powered_by_text():
    with pytest.raises(ValidationError):
        BrandingConfig(powered_by_text="   ")


def test_branding_rejects_unknown_field():
    """Extra fields should raise (model is configured ``extra='forbid'``)."""
    with pytest.raises(ValidationError):
        BrandingConfig(unknown_field="surprise!")


def test_branding_is_frozen():
    bc = BrandingConfig(logo_url="https://cdn.acme.com/x.png")
    with pytest.raises(ValidationError):
        bc.logo_url = "https://cdn.evil.com/x.png"  # type: ignore[misc]


def test_branding_to_response_metadata_drops_nones():
    bc = BrandingConfig(
        logo_url="https://cdn.acme.com/logo.svg",
        primary_color="#0A66C2",
    )
    meta = bc.to_response_metadata()
    assert meta["logo_url"] == "https://cdn.acme.com/logo.svg"
    assert meta["primary_color"] == "#0a66c2"
    assert "secondary_color" not in meta
    assert "custom_domain" not in meta
    # Defaults that are always set survive serialisation.
    assert meta["attribution_required"] is True
    assert meta["powered_by_text"] == "Powered by GreenLang"


def test_branding_empty_classmethod():
    assert BrandingConfig.empty() == BrandingConfig()


# ---------------------------------------------------------------------------
# update_branding round-trip
# ---------------------------------------------------------------------------


def test_branding_round_trip(acme_oem):
    bc = BrandingConfig(
        logo_url="https://cdn.acme.com/logo.svg",
        primary_color="#0A66C2",
        powered_by_text="Powered by GreenLang",
    )
    update_branding(acme_oem.id, bc)
    refreshed = get_oem_partner(acme_oem.id)
    assert refreshed.branding is bc


def test_branding_can_be_cleared(acme_oem):
    update_branding(acme_oem.id, BrandingConfig(logo_url="https://x.example.com/l.png"))
    update_branding(acme_oem.id, None)
    assert get_oem_partner(acme_oem.id).branding is None


# ---------------------------------------------------------------------------
# apply_oem_branding decorator
# ---------------------------------------------------------------------------


def test_apply_oem_branding_returns_unchanged_when_oem_missing():
    response = {"data": [1, 2, 3]}
    out = apply_oem_branding(response, None)
    assert out is response
    assert "branding" not in response


def test_apply_oem_branding_returns_unchanged_for_unknown_oem():
    response = {"data": [1, 2, 3]}
    out = apply_oem_branding(response, "no-such-oem")
    assert out is response


def test_apply_oem_branding_returns_unchanged_when_no_branding(acme_oem):
    """OEM with no branding configured - response is untouched."""
    response = {"data": [1, 2, 3]}
    out = apply_oem_branding(response, acme_oem.id)
    assert "branding" not in out


def test_apply_oem_branding_decorates_dict_response(acme_oem):
    bc = BrandingConfig(
        logo_url="https://cdn.acme.com/logo.svg",
        primary_color="#0A66C2",
        powered_by_text="Powered by GreenLang",
    )
    update_branding(acme_oem.id, bc)
    response = {"data": [{"factor_id": "x"}]}
    out = apply_oem_branding(response, acme_oem.id)
    assert "branding" in out
    assert out["branding"]["logo_url"] == "https://cdn.acme.com/logo.svg"
    assert out["branding"]["powered_by_text"] == "Powered by GreenLang"
    assert out["branding"]["oem_id"] == acme_oem.id


def test_apply_oem_branding_wraps_non_dict_response(acme_oem):
    update_branding(acme_oem.id, BrandingConfig(
        logo_url="https://cdn.acme.com/l.svg",
        powered_by_text="Powered by GreenLang",
    ))
    response = [{"factor_id": "a"}, {"factor_id": "b"}]
    out = apply_oem_branding(response, acme_oem.id)
    assert isinstance(out, dict)
    assert out["data"] == response
    assert out["branding"]["logo_url"] == "https://cdn.acme.com/l.svg"


def test_apply_oem_branding_merges_existing_branding_dict(acme_oem):
    """A pre-existing ``branding`` dict on the response is merged."""
    update_branding(acme_oem.id, BrandingConfig(
        logo_url="https://cdn.acme.com/l.svg",
    ))
    response = {
        "data": [],
        "branding": {"theme": "dark", "logo_url": "https://stale.example.com/x.png"},
    }
    out = apply_oem_branding(response, acme_oem.id)
    assert out["branding"]["theme"] == "dark"  # preserved
    # OEM payload wins on overlap.
    assert out["branding"]["logo_url"] == "https://cdn.acme.com/l.svg"


def test_apply_oem_branding_accepts_plain_dict_branding(acme_oem):
    """If branding was hand-set as a dict (not BrandingConfig) it still works."""
    # Bypass the helper and stuff a plain dict into the registry.
    acme_oem.branding = {
        "logo_url": "https://cdn.example.com/x.png",
        "powered_by_text": "Powered by GreenLang",
        "drop_me": None,
    }
    response = {"data": "ok"}
    out = apply_oem_branding(response, acme_oem.id)
    assert out["branding"]["logo_url"] == "https://cdn.example.com/x.png"
    # None values are stripped during normalisation.
    assert "drop_me" not in out["branding"]


def test_apply_oem_branding_falls_back_for_unsupported_branding_type(acme_oem):
    """Garbage branding types are tolerated (response is left alone)."""

    class _Junk:
        pass

    acme_oem.branding = _Junk()
    response = {"data": "ok"}
    out = apply_oem_branding(response, acme_oem.id)
    assert "branding" not in out
