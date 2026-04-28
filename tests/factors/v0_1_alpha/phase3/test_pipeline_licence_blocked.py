# -*- coding: utf-8 -*-
"""Phase 3 — licence mismatch between record and source registry blocks ingestion.

Per PHASE_3_PLAN.md §"Fetcher / parser families" + Phase 1 source-rights
gates: when the source_registry pins a ``licence_class`` and a record
declares a different licence, the pipeline must reject at validate
(stage 4) with :class:`LicenceMismatchError`.

Reference: ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 6 scenario
"licence-mismatch-blocks-ingestion".
"""
from __future__ import annotations

import copy
import importlib

import pytest


def _runner_available() -> bool:
    try:
        importlib.import_module("greenlang.factors.ingestion.runner")
    except Exception:  # noqa: BLE001
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _runner_available(),
    reason=(
        "greenlang.factors.ingestion.runner not yet committed; "
        "Wave 1.0 sibling agent still in flight"
    ),
)


def test_record_licence_mismatch_blocks_at_validate(
    phase3_runner,
    seeded_source_urn,
    synthetic_factor_record,
):
    """Record claims a licence different from the registry pin -> rejected."""
    from greenlang.factors.quality.publish_gates import LicenceMismatchError

    record = copy.deepcopy(synthetic_factor_record)
    # The seeded fake-rights pins ``CC-BY-4.0`` for SEEDED_SOURCE_URN.
    # Forcing an obviously-wrong licence string must trip gate 5.
    record["licence"] = "PROPRIETARY-CONFIDENTIAL"

    with pytest.raises(LicenceMismatchError):
        phase3_runner.run_records(
            records=[record],
            source_urn=seeded_source_urn,
            source_version="2024.1",
            operator="bot:phase3-licence",
        )
