# -*- coding: utf-8 -*-
"""Tests for source approval gate (G5) and export guard (G6)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from greenlang.data.emission_factor_database import EmissionFactorDatabase
from greenlang.factors.approval_gate import (
    check_promote_to_certified,
    public_bulk_export_allowed_for_factor,
    require_public_bulk_export,
)
from greenlang.factors.source_registry import SourceRegistryEntry


def test_promote_certified_blocks_connector_only_source():
    reg = {
        "electricity_maps": SourceRegistryEntry(
            source_id="electricity_maps",
            display_name="x",
            connector_only=True,
            license_class="c",
            redistribution_allowed=False,
            derivative_works_allowed=False,
            commercial_use_allowed=False,
            attribution_required=True,
            citation_text="citation",
            cadence="daily",
            watch_mechanism="api",
            watch_url=None,
            watch_file_type=None,
            approval_required_for_certified=True,
            legal_signoff_artifact="/legal/x.pdf",
            legal_signoff_version="1",
        )
    }
    r = check_promote_to_certified("electricity_maps", registry=reg)
    assert not r.allowed
    assert "connector_only" in r.blockers[0]


def test_promote_certified_requires_legal_when_strict():
    reg = {
        "epa_hub": SourceRegistryEntry(
            source_id="epa_hub",
            display_name="EPA",
            connector_only=False,
            license_class="public",
            redistribution_allowed=True,
            derivative_works_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
            citation_text="EPA",
            cadence="q",
            watch_mechanism="http_head",
            watch_url="https://example.com",
            watch_file_type=None,
            approval_required_for_certified=True,
            legal_signoff_artifact=None,
            legal_signoff_version=None,
        )
    }
    assert not check_promote_to_certified("epa_hub", registry=reg, require_legal_signoff=True).allowed
    assert check_promote_to_certified("epa_hub", registry=reg, require_legal_signoff=False).allowed


def test_export_guard_builtin_factor_with_registry():
    db = EmissionFactorDatabase(enable_cache=False)
    base = next(iter(db.factors.values()))
    rec = replace(
        base,
        source_id="greenlang_builtin",
        factor_status="certified",
    )
    gate = public_bulk_export_allowed_for_factor(rec)
    assert gate.allowed


def test_export_guard_blocks_connector_only_status():
    db = EmissionFactorDatabase(enable_cache=False)
    base = next(iter(db.factors.values()))
    rec = replace(base, source_id="greenlang_builtin", factor_status="connector_only")
    gate = public_bulk_export_allowed_for_factor(rec)
    assert not gate.allowed


def test_require_public_bulk_export_raises():
    db = EmissionFactorDatabase(enable_cache=False)
    base = next(iter(db.factors.values()))
    rec = replace(base, source_id="electricity_maps", factor_status="certified")
    with pytest.raises(ValueError):
        require_public_bulk_export(rec)
