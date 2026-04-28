# -*- coding: utf-8 -*-
"""Phase 3 — every certified factor MUST carry a raw artifact pin.

Per PHASE_3_PLAN.md §"Artifact storage contract", a normalized record
must carry both ``extraction.raw_artifact_uri`` and
``extraction.raw_artifact_sha256``. Missing either field is rejected at
the validate stage; the run lands in a terminal failure state without
any factor row written.

Reference: ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 2 negative
test ("Parser output is rejected when artifact storage write failed").
"""
from __future__ import annotations

import importlib
import copy

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


def test_missing_raw_artifact_uri_blocks_at_validate(
    phase3_runner,
    seeded_source_urn,
    synthetic_factor_record,
):
    """A record missing ``extraction.raw_artifact_uri`` is rejected."""
    from greenlang.factors.ingestion.exceptions import (
        IngestionError,
        ValidationStageError,
    )

    record = copy.deepcopy(synthetic_factor_record)
    record["extraction"].pop("raw_artifact_uri", None)

    with pytest.raises((ValidationStageError, IngestionError)) as exc_info:
        phase3_runner.run_records(
            records=[record],
            source_urn=seeded_source_urn,
            source_version="2024.1",
            operator="bot:phase3-artifact",
        )
    assert getattr(exc_info.value, "stage", "validate") == "validate"


def test_missing_raw_artifact_sha256_blocks_at_validate(
    phase3_runner,
    seeded_source_urn,
    synthetic_factor_record,
):
    """A record missing ``extraction.raw_artifact_sha256`` is rejected."""
    from greenlang.factors.ingestion.exceptions import (
        IngestionError,
        ValidationStageError,
    )

    record = copy.deepcopy(synthetic_factor_record)
    record["extraction"].pop("raw_artifact_sha256", None)

    with pytest.raises((ValidationStageError, IngestionError)) as exc_info:
        phase3_runner.run_records(
            records=[record],
            source_urn=seeded_source_urn,
            source_version="2024.1",
            operator="bot:phase3-artifact",
        )
    assert getattr(exc_info.value, "stage", "validate") == "validate"
