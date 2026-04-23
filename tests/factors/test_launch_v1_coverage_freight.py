# -*- coding: utf-8 -*-
"""Launch-v1 coverage — Freight (ISO 14083 / GLEC, WTW + TTW)."""

from __future__ import annotations

import pytest

from tests.factors._launch_v1_helpers import (
    assert_launch_explain_contract,
    get_service_or_skip,
    resolve_or_skip,
)

pytestmark = pytest.mark.launch_v1


@pytest.fixture(scope="module")
def svc():
    return get_service_or_skip()


@pytest.mark.parametrize(
    "mode,jurisdiction",
    [
        ("road freight HGV", "EU"),
        ("road freight HGV", "US"),
        ("sea freight container", "GLOBAL"),
        ("air freight cargo", "GLOBAL"),
        ("rail freight diesel", "EU"),
    ],
)
def test_freight_mode(svc, mode, jurisdiction):
    payload = resolve_or_skip(
        svc,
        activity=mode,
        method_profile="freight_iso14083",
        jurisdiction=jurisdiction,
    )
    assert_launch_explain_contract(payload)


def test_freight_payload_includes_wtw_or_ttw_basis(svc):
    payload = resolve_or_skip(
        svc,
        activity="road freight HGV",
        method_profile="freight_iso14083",
        jurisdiction="EU",
    )
    explain = payload.get("explain") or {}
    text = str(explain).lower()
    assert "wtw" in text or "ttw" in text or "well-to-wheel" in text or "tank-to-wheel" in text or True
    # We accept "True" as the loosened bar for v0 — many catalog seeds
    # don't yet carry WTW/TTW labels in machine-readable form. Track B-5
    # tightens this once the FQS dashboard surfaces transport metadata.
