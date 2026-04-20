# -*- coding: utf-8 -*-
"""Phase 5.3 status-summary aggregation tests."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


class _StubRepo:
    """Minimal FactorCatalogRepository stub for status_summary()."""

    def __init__(self, factors: list[Any]) -> None:
        self._factors = factors
        self._edition = "test-edition"

    def resolve_edition(self, requested):
        return requested or self._edition

    def list_factors(self, edition_id, *, page, limit, include_preview, include_connector, **_):
        # Honour the ``include_*`` flags so tests can verify the service
        # asks for EVERY factor (otherwise preview/connector aren't counted).
        if not include_preview and not include_connector:
            return [f for f in self._factors if f.factor_status == "certified"], len(self._factors)
        return list(self._factors), len(self._factors)


def _factor(
    *,
    factor_id: str,
    factor_status: str = "certified",
    source_id: str = "unknown",
) -> Any:
    return SimpleNamespace(
        factor_id=factor_id,
        factor_status=factor_status,
        source_id=source_id,
    )


def test_status_summary_groups_by_label_and_source():
    from greenlang.factors.service import FactorCatalogService

    repo = _StubRepo(
        factors=[
            _factor(factor_id="a", factor_status="certified", source_id="epa_hub"),
            _factor(factor_id="b", factor_status="certified", source_id="epa_hub"),
            _factor(factor_id="c", factor_status="preview", source_id="desnz_uk"),
            _factor(factor_id="d", factor_status="connector_only", source_id="ecoinvent"),
            _factor(factor_id="e", factor_status="deprecated", source_id="epa_hub"),
        ]
    )
    svc = FactorCatalogService(repo)

    summary = svc.status_summary("test-edition")

    assert summary["edition_id"] == "test-edition"
    assert summary["totals"] == {
        "certified": 2,
        "preview": 1,
        "connector_only": 1,
        "deprecated": 1,
        "all": 5,
    }
    by_source = {row["source_id"]: row for row in summary["by_source"]}
    assert by_source["epa_hub"]["certified"] == 2
    assert by_source["epa_hub"]["deprecated"] == 1
    assert by_source["epa_hub"]["all"] == 3
    assert by_source["desnz_uk"]["preview"] == 1
    assert by_source["ecoinvent"]["connector_only"] == 1
    assert "generated_at" in summary


def test_status_summary_requests_full_visibility():
    """Ensure the service asks the repo for preview + connector factors.

    Without this, preview/connector counts would always be zero.
    """
    from greenlang.factors.service import FactorCatalogService

    seen_flags: dict[str, bool] = {}

    class _ObservingRepo(_StubRepo):
        def list_factors(self, edition_id, **kw):
            seen_flags["include_preview"] = kw.get("include_preview", False)
            seen_flags["include_connector"] = kw.get("include_connector", False)
            return [], 0

    svc = FactorCatalogService(_ObservingRepo(factors=[]))
    svc.status_summary("test-edition")
    assert seen_flags["include_preview"] is True
    assert seen_flags["include_connector"] is True


def test_status_summary_handles_missing_status_field():
    """Factors with no ``factor_status`` attribute fall through to ``certified``."""
    from greenlang.factors.service import FactorCatalogService

    class _Bare:
        factor_id = "x"
        source_id = "src"
        # Intentionally no factor_status.

    svc = FactorCatalogService(_StubRepo(factors=[_Bare()]))
    summary = svc.status_summary("test-edition")
    assert summary["totals"]["certified"] == 1
    assert summary["totals"]["all"] == 1


def test_status_summary_missing_source_goes_to_unknown():
    from greenlang.factors.service import FactorCatalogService

    class _NoSource:
        factor_id = "x"
        factor_status = "certified"
        source_id = None

    svc = FactorCatalogService(_StubRepo(factors=[_NoSource()]))
    summary = svc.status_summary("test-edition")
    by_source = {row["source_id"]: row for row in summary["by_source"]}
    assert "unknown" in by_source
    assert by_source["unknown"]["certified"] == 1


def test_status_summary_normalizes_unknown_status_values():
    """A factor with e.g. ``factor_status='weird'`` is counted as certified."""
    from greenlang.factors.service import FactorCatalogService

    svc = FactorCatalogService(
        _StubRepo(factors=[_factor(factor_id="a", factor_status="weird")])
    )
    summary = svc.status_summary("test-edition")
    assert summary["totals"]["certified"] == 1
    assert summary["totals"]["all"] == 1


def test_status_summary_empty_catalog():
    from greenlang.factors.service import FactorCatalogService

    svc = FactorCatalogService(_StubRepo(factors=[]))
    summary = svc.status_summary("test-edition")
    assert summary["totals"] == {
        "certified": 0,
        "preview": 0,
        "connector_only": 0,
        "deprecated": 0,
        "all": 0,
    }
    assert summary["by_source"] == []


def test_status_summary_sort_is_deterministic():
    from greenlang.factors.service import FactorCatalogService

    svc = FactorCatalogService(
        _StubRepo(
            factors=[
                _factor(factor_id="a", source_id="z_source"),
                _factor(factor_id="b", source_id="a_source"),
                _factor(factor_id="c", source_id="m_source"),
            ]
        )
    )
    summary = svc.status_summary("test-edition")
    sources = [row["source_id"] for row in summary["by_source"]]
    assert sources == ["a_source", "m_source", "z_source"]
