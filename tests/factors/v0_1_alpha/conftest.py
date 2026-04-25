# -*- coding: utf-8 -*-
"""Local conftest for ``tests/factors/v0_1_alpha/``.

Registers the custom marker ``skip_unless_alpha_records_seeded`` used by
``test_schema_validates_alpha_catalog.py``. The marker is informational
only — it does NOT auto-skip; per-test logic uses
``pytest.skip(...)`` when the input fixture file is empty/missing.

Marker registration is required because the global ``pytest.ini`` sets
``strict_markers = true`` (unknown markers raise at collection time).
"""
from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    # Wave D / TaskCreate #7-#12 — alpha-source snapshot tests support
    # an opt-in regenerate flag. Defined here (vs in the test module)
    # because pytest only honours pytest_addoption from conftest.py /
    # plugins, not from inside a test module.
    parser.addoption(
        "--update-source-snapshots",
        action="store_true",
        default=False,
        help=(
            "Regenerate the alpha-source expected.json snapshot files in "
            "place by running each parser on its raw.json fixture. Use "
            "after an intentional parser change."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "skip_unless_alpha_records_seeded: marker for alpha-catalog tests that "
        "are skipped when the per-source seed fixture under "
        "greenlang/factors/data/catalog_seed/_inputs/<source>.json is empty "
        "or unavailable. The test body still calls pytest.skip() with a "
        "human-readable reason — this marker is for filterability only "
        "(e.g. `pytest -m skip_unless_alpha_records_seeded`).",
    )
    # Wave D / TaskCreate #28 / WS10-T2 — performance budget marker.
    # The default test invocation MUST exclude perf tests (they are slow,
    # they issue 10000+ requests, and they should run nightly on a clean
    # box, not on every PR). Run them explicitly via:
    #     pytest -m perf tests/factors/v0_1_alpha/
    config.addinivalue_line(
        "markers",
        "perf: performance / latency-budget tests for the alpha API "
        "(CTO doc §19.1 acceptance: p95 lookup < 100ms). Excluded from "
        "default CI; run nightly via `pytest -m perf tests/factors/v0_1_alpha/`.",
    )
    # Wave D — acceptance-criterion marker. Tests carrying this marker
    # encode an explicit clause from CTO doc §19.1; treat them as the
    # ship-readiness gate for v0.1 alpha.
    config.addinivalue_line(
        "markers",
        "alpha_v0_1_acceptance: marks a test as encoding a verbatim "
        "acceptance criterion from CTO doc §19.1 (the v0.1 alpha launch "
        "spec). Failures here block alpha launch.",
    )
    # Wave E / TaskCreate #26 / WS9-T4 — incident-drill regression marker.
    # Tests carrying this marker exercise an INTENTIONALLY corrupted
    # fixture (e.g. tests/factors/v0_1_alpha/drill_fixtures/) to keep the
    # incident-drill postmortem (docs/factors/postmortems/) factually
    # honest: if the parser becomes more strict or the gate becomes more
    # lenient, these regressions fail and the postmortem must be re-run.
    config.addinivalue_line(
        "markers",
        "drill: incident-drill regression tests for the v0.1 alpha "
        "ingestion path. Exercises an INTENTIONALLY corrupted fixture; "
        "cross-references docs/factors/postmortems/2026-Q1-desnz-parser-"
        "drift-drill.md.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Auto-skip perf tests unless explicitly selected.

    The acceptance criterion (p95 < 100ms over 10000 calls) issues many
    HTTP round-trips and is too slow for every PR. We require the user
    to opt in via either ``-m perf`` or the ``GL_RUN_PERF=1`` env var so
    the default ``pytest tests/factors/v0_1_alpha/`` invocation stays fast
    and green.
    """
    import os

    # If the user explicitly selected perf tests via -m, run them.
    # pytest stores the -m argument under the option name ``markexpr``.
    try:
        markexpr = config.getoption("markexpr", default="") or ""
    except (ValueError, KeyError):  # pragma: no cover - defensive
        markexpr = ""
    if "perf" in markexpr:
        return
    if os.environ.get("GL_RUN_PERF") == "1":
        return

    skip_perf = pytest.mark.skip(
        reason=(
            "perf tests are excluded from default runs; "
            "use `pytest -m perf tests/factors/v0_1_alpha/` "
            "or set GL_RUN_PERF=1 to enable."
        )
    )
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip_perf)
