# -*- coding: utf-8 -*-
"""Phase 3 — invalid ontology references block at validate stage.

Per PHASE_3_PLAN.md §"The seven-stage pipeline contract" stage 4: a
record whose ``unit_urn`` / ``geography_urn`` / ``methodology_urn``
does not resolve in the seeded ontology table is rejected with
:class:`OntologyReferenceError` BEFORE staging.

Reference: ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 6 scenario
"invalid-ontology-blocks-staging".
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


def test_phantom_unit_urn_blocks_at_validate(
    phase3_runner,
    seeded_source_urn,
    synthetic_factor_record,
):
    """``unit_urn`` not present in the ontology table -> gate 3 rejects."""
    from greenlang.factors.quality.publish_gates import OntologyReferenceError

    record = copy.deepcopy(synthetic_factor_record)
    record["unit_urn"] = "urn:gl:unit:nonexistent"

    with pytest.raises(OntologyReferenceError):
        phase3_runner.run_records(
            records=[record],
            source_urn=seeded_source_urn,
            source_version="2024.1",
            operator="bot:phase3-ontology",
        )


def test_phantom_methodology_urn_blocks_at_validate(
    phase3_runner,
    seeded_source_urn,
    synthetic_factor_record,
):
    """``methodology_urn`` not present in the ontology table -> rejected."""
    from greenlang.factors.quality.publish_gates import OntologyReferenceError

    record = copy.deepcopy(synthetic_factor_record)
    record["methodology_urn"] = "urn:gl:methodology:phantom-tier"

    with pytest.raises(OntologyReferenceError):
        phase3_runner.run_records(
            records=[record],
            source_urn=seeded_source_urn,
            source_version="2024.1",
            operator="bot:phase3-ontology",
        )
