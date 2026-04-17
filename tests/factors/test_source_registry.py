# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.source_registry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from greenlang.factors.source_registry import (
    SourceRegistryEntry,
    load_source_registry,
    registry_by_id,
    validate_registry,
)


def test_load_default_registry(source_registry):
    assert isinstance(source_registry, list)
    assert len(source_registry) > 0
    assert all(isinstance(e, SourceRegistryEntry) for e in source_registry)


def test_validate_registry_no_issues(source_registry):
    issues = validate_registry(source_registry)
    assert issues == []


def test_registry_by_id_returns_dict():
    d = registry_by_id()
    assert isinstance(d, dict)
    assert len(d) > 0
    for key, val in d.items():
        assert isinstance(key, str)
        assert isinstance(val, SourceRegistryEntry)
        assert val.source_id == key


def test_duplicate_source_id_detected():
    entry = SourceRegistryEntry(
        source_id="dup_test",
        display_name="Dup",
        connector_only=False,
        license_class="public",
        redistribution_allowed=True,
        derivative_works_allowed=True,
        commercial_use_allowed=True,
        attribution_required=False,
        citation_text="cite",
        cadence="annual",
        watch_mechanism="none",
        watch_url=None,
        watch_file_type=None,
        approval_required_for_certified=False,
        legal_signoff_artifact=None,
        legal_signoff_version=None,
    )
    issues = validate_registry([entry, entry])
    assert any("duplicate" in i for i in issues)


def test_connector_only_redistribution_conflict():
    entry = SourceRegistryEntry(
        source_id="conflict_test",
        display_name="Conflict",
        connector_only=True,
        license_class="commercial",
        redistribution_allowed=True,
        derivative_works_allowed=True,
        commercial_use_allowed=True,
        attribution_required=True,
        citation_text="x",
        cadence="daily",
        watch_mechanism="api",
        watch_url=None,
        watch_file_type=None,
        approval_required_for_certified=True,
        legal_signoff_artifact=None,
        legal_signoff_version=None,
    )
    issues = validate_registry([entry])
    assert any("connector_only" in i for i in issues)


def test_missing_citation_detected():
    entry = SourceRegistryEntry(
        source_id="no_cite",
        display_name="No Cite",
        connector_only=False,
        license_class="public",
        redistribution_allowed=True,
        derivative_works_allowed=True,
        commercial_use_allowed=True,
        attribution_required=True,
        citation_text="",
        cadence="annual",
        watch_mechanism="none",
        watch_url=None,
        watch_file_type=None,
        approval_required_for_certified=True,
        legal_signoff_artifact=None,
        legal_signoff_version=None,
    )
    issues = validate_registry([entry])
    assert any("citation_text" in i for i in issues)


def test_registry_entry_public_bulk_export():
    entry_yes = SourceRegistryEntry(
        source_id="bulk_yes",
        display_name="Y",
        connector_only=False,
        license_class="public",
        redistribution_allowed=True,
        derivative_works_allowed=True,
        commercial_use_allowed=True,
        attribution_required=False,
        citation_text="x",
        cadence="annual",
        watch_mechanism="none",
        watch_url=None,
        watch_file_type=None,
        approval_required_for_certified=False,
        legal_signoff_artifact=None,
        legal_signoff_version=None,
    )
    assert entry_yes.public_bulk_export_allowed() is True

    entry_no = SourceRegistryEntry(
        source_id="bulk_no",
        display_name="N",
        connector_only=True,
        license_class="commercial",
        redistribution_allowed=True,
        derivative_works_allowed=False,
        commercial_use_allowed=False,
        attribution_required=True,
        citation_text="x",
        cadence="daily",
        watch_mechanism="api",
        watch_url=None,
        watch_file_type=None,
        approval_required_for_certified=True,
        legal_signoff_artifact=None,
        legal_signoff_version=None,
    )
    assert entry_no.public_bulk_export_allowed() is False


def test_registry_fields_complete(source_registry):
    for e in source_registry:
        assert e.display_name, f"{e.source_id}: empty display_name"
        assert e.cadence, f"{e.source_id}: empty cadence"
        assert e.source_id, f"{e.source_id}: empty source_id"


def test_yaml_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_source_registry(Path("/nonexistent/path/registry.yaml"))


def test_no_yaml_import_raises():
    with patch("greenlang.factors.source_registry.yaml", None):
        with pytest.raises(RuntimeError, match="PyYAML"):
            load_source_registry()
