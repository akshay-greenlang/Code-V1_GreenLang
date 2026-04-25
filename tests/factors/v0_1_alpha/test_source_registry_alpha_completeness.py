# -*- coding: utf-8 -*-
"""WS7-T1 — v0.1 Alpha source-registry completeness tests.

These tests enforce that the 6 alpha-launch sources in
``greenlang/factors/data/source_registry.yaml`` populate every field of
the alpha required-fields contract, and that each declared parser is
actually importable / callable.

The 6 canonical alpha source ids (per WS7-T1):
    1. ipcc_2006_nggi          (alpha alias: ipcc_ar6)
    2. desnz_ghg_conversion
    3. epa_hub
    4. egrid
    5. india_cea_co2_baseline  (alpha alias: india_cea_baseline)
    6. cbam_default_values
"""
from __future__ import annotations

import importlib
import re
from typing import Any, Dict

import pytest

from greenlang.factors.source_registry import (
    ALPHA_V0_1_EXPECTED_SOURCE_IDS,
    ALPHA_V0_1_REQUIRED_FIELDS,
    alpha_v0_1_sources,
    validate_alpha_v0_1_completeness,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def alpha_sources() -> Dict[str, Dict[str, Any]]:
    """Raw YAML rows for every alpha_v0_1 source, keyed by source_id."""
    return alpha_v0_1_sources()


# --------------------------------------------------------------------------- #
# Required tests (WS7-T1)                                                     #
# --------------------------------------------------------------------------- #
def test_six_alpha_sources_present(alpha_sources: Dict[str, Dict[str, Any]]) -> None:
    """All 6 canonical alpha source ids must be flagged alpha_v0_1: true."""
    found = set(alpha_sources.keys())
    expected = set(ALPHA_V0_1_EXPECTED_SOURCE_IDS)
    missing = expected - found
    extra = found - expected
    assert not missing, (
        f"Missing alpha_v0_1 flag on canonical sources: {sorted(missing)}. "
        f"Found: {sorted(found)}"
    )
    assert not extra, (
        f"Unexpected alpha_v0_1 flag on non-alpha sources: {sorted(extra)}. "
        f"Only the 6 canonical alpha sources may carry alpha_v0_1: true."
    )
    assert len(found) == 6, f"Expected exactly 6 alpha sources, found {len(found)}"


def test_no_missing_required_fields() -> None:
    """validate_alpha_v0_1_completeness() returns empty list for every alpha row."""
    results = validate_alpha_v0_1_completeness()
    assert len(results) == 6, f"Expected 6 alpha rows, got {len(results)}"
    failures = [(sid, missing) for sid, missing in results if missing]
    assert not failures, (
        "Alpha sources missing required fields:\n"
        + "\n".join(f"  - {sid}: {missing}" for sid, missing in failures)
    )


def test_parser_module_importable(alpha_sources: Dict[str, Dict[str, Any]]) -> None:
    """Every alpha source's parser_module must import cleanly."""
    for sid, row in alpha_sources.items():
        module_path = row.get("parser_module")
        assert module_path, f"{sid}: parser_module is empty"
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:  # pragma: no cover - hard fail
            pytest.fail(f"{sid}: parser_module {module_path!r} failed to import: {exc}")
        assert mod is not None


def test_parser_function_callable(alpha_sources: Dict[str, Dict[str, Any]]) -> None:
    """Every alpha source's parser_function must exist as a callable in its module."""
    for sid, row in alpha_sources.items():
        module_path = row.get("parser_module")
        function_name = row.get("parser_function")
        assert function_name, f"{sid}: parser_function is empty"
        mod = importlib.import_module(module_path)
        fn = getattr(mod, function_name, None)
        assert fn is not None, (
            f"{sid}: parser_function {function_name!r} not found in module "
            f"{module_path!r}"
        )
        assert callable(fn), (
            f"{sid}: parser_function {function_name!r} is not callable"
        )


def test_alpha_urns_well_formed(alpha_sources: Dict[str, Dict[str, Any]]) -> None:
    """Each alpha source's URN must match urn:gl:source:[a-z0-9-]+."""
    pattern = re.compile(r"^urn:gl:source:[a-z0-9][a-z0-9-]*$")
    for sid, row in alpha_sources.items():
        urn = row.get("urn")
        assert urn, f"{sid}: urn is empty"
        assert pattern.match(urn), (
            f"{sid}: urn {urn!r} does not match urn:gl:source:[a-z0-9-]+"
        )


# --------------------------------------------------------------------------- #
# Auxiliary structural tests                                                  #
# --------------------------------------------------------------------------- #
def test_required_fields_contract_locked() -> None:
    """The alpha required-fields contract must contain the 14 named fields."""
    expected = {
        "source_id",
        "urn",
        "source_owner",
        "parser_module",
        "parser_function",
        "parser_version",
        "cadence",
        "license_class",
        "source_version",
        "latest_ingestion_at",
        "legal_signoff_artifact",
        "publication_url",
        "provenance_completeness_score",
        "alpha_v0_1",
    }
    assert set(ALPHA_V0_1_REQUIRED_FIELDS) == expected
    assert len(ALPHA_V0_1_REQUIRED_FIELDS) == 14


def test_alpha_parser_versions_are_semver(
    alpha_sources: Dict[str, Dict[str, Any]],
) -> None:
    """parser_version on every alpha source must be valid semver (major.minor.patch)."""
    semver = re.compile(r"^\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?$")
    for sid, row in alpha_sources.items():
        pv = row.get("parser_version")
        assert pv, f"{sid}: parser_version is empty"
        assert semver.match(pv), f"{sid}: parser_version {pv!r} is not valid semver"


def test_alpha_provenance_completeness_in_range(
    alpha_sources: Dict[str, Dict[str, Any]],
) -> None:
    """provenance_completeness_score must be in [0.0, 1.0]."""
    for sid, row in alpha_sources.items():
        score = row.get("provenance_completeness_score")
        assert isinstance(score, (int, float)), (
            f"{sid}: provenance_completeness_score must be numeric, got {type(score)}"
        )
        assert 0.0 <= float(score) <= 1.0, (
            f"{sid}: provenance_completeness_score {score} outside [0.0, 1.0]"
        )


def test_alpha_publication_urls_https(
    alpha_sources: Dict[str, Dict[str, Any]],
) -> None:
    """publication_url must be an https URL."""
    for sid, row in alpha_sources.items():
        url = row.get("publication_url")
        assert url, f"{sid}: publication_url is empty"
        assert url.startswith("https://"), (
            f"{sid}: publication_url {url!r} must be https"
        )
