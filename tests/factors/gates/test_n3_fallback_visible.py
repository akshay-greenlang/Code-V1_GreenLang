# -*- coding: utf-8 -*-
"""
N3 — The resolver never hides its fallback logic.

Every ``ResolvedFactor`` returned from
:meth:`greenlang.factors.resolution.engine.ResolutionEngine.resolve`
must carry:

    * ``fallback_rank`` in [1, 7]           — which cascade tier fired
    * ``step_label`` in the 7 canonical tiers
    * ``why_chosen`` non-empty              — human-readable one-liner
    * ``alternates`` list with >= 1 element — at least one runner-up

Missing any of those is a regulatory visibility failure: auditors
need to see why the resolver picked factor X over Y, and the
``/explain`` endpoint is the customer surface for that.

Run standalone::

    pytest tests/factors/gates/test_n3_fallback_visible.py -v
"""
from __future__ import annotations

from typing import Any, List

import pytest

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.resolution import (
    ResolutionEngine,
    ResolutionRequest,
    ResolvedFactor,
)


CANONICAL_STEPS = {
    "customer_override",
    "supplier_specific",
    "facility_specific",
    "utility_or_grid_subregion",
    "country_or_sector_average",
    "method_pack_default",
    "global_default",
}


# ---------------------------------------------------------------------------
# Helpers: build a minimal "country_or_sector_average" resolve call that
# returns at least two candidates so alternates can be populated.
# ---------------------------------------------------------------------------


def _base_request() -> ResolutionRequest:
    return ResolutionRequest(
        activity="diesel combustion stationary",
        method_profile=MethodProfile.CORPORATE_SCOPE1,
        jurisdiction="US",
        reporting_date="2026-06-01",
    )


# ---------------------------------------------------------------------------
# Gate: single-candidate resolve — fallback info must still be present.
# ---------------------------------------------------------------------------


class TestN3FallbackFieldsAlwaysPresent:
    """fallback_rank / step_label / why_chosen must be populated on every resolve."""

    def test_fields_present_on_country_step(
        self, make_record, make_candidate_source
    ):
        rec = make_record(
            factor_id="EF:US:diesel:2024:v1",
            family="emissions",
            vectors=__import__(
                "tests.factors.gates.conftest", fromlist=["_vectors"]
            )._vectors(CO2=10.18, CH4=0.00082, N2O=0.000164),
            co2e_total=10.2097,
        )
        engine = ResolutionEngine(
            candidate_source=make_candidate_source(
                {"country_or_sector_average": [rec]}
            ),
            tenant_overlay_reader=lambda req: None,
        )
        resolved: ResolvedFactor = engine.resolve(_base_request())

        assert resolved.fallback_rank in range(1, 8), (
            "N3 violation: fallback_rank must be in 1..7. "
            f"Got {resolved.fallback_rank!r}"
        )
        assert resolved.step_label in CANONICAL_STEPS, (
            f"N3 violation: step_label {resolved.step_label!r} is not one of "
            f"the 7 canonical tiers {sorted(CANONICAL_STEPS)}. Resolver is "
            "hiding fallback logic behind a freeform string."
        )
        assert resolved.why_chosen and resolved.why_chosen.strip(), (
            "N3 violation: why_chosen is empty. Auditors must see a natural-"
            "language reason for every selected factor."
        )

    def test_rank_and_label_agree(self, make_record, make_candidate_source):
        """fallback_rank must match step_label position in the cascade."""
        from tests.factors.gates.conftest import _vectors

        rec = make_record(
            factor_id="EF:US:diesel:country:v1",
            family="emissions",
            vectors=_vectors(CO2=10.18, CH4=0.00082, N2O=0.000164),
            co2e_total=10.2097,
        )
        engine = ResolutionEngine(
            candidate_source=make_candidate_source(
                {"country_or_sector_average": [rec]}
            ),
            tenant_overlay_reader=lambda req: None,
        )
        resolved = engine.resolve(_base_request())
        # country_or_sector_average is rank 5 in the canonical cascade.
        assert resolved.step_label == "country_or_sector_average"
        assert resolved.fallback_rank == 5, (
            "N3 violation: rank/label disagreement. step_label="
            f"{resolved.step_label!r} but fallback_rank={resolved.fallback_rank}. "
            "Auditors depend on these two agreeing."
        )


# ---------------------------------------------------------------------------
# Gate: at least one alternate surfaced when a runner-up exists.
# ---------------------------------------------------------------------------


class TestN3AlternatesSurfaced:
    """When more than one candidate is eligible, alternates must be non-empty."""

    def test_runner_up_is_reported(self, make_record, make_candidate_source):
        from tests.factors.gates.conftest import _vectors

        winner = make_record(
            factor_id="EF:US:diesel:winner:v1",
            family="emissions",
            vectors=_vectors(CO2=10.18, CH4=0.00082, N2O=0.000164),
            co2e_total=10.2097,
            source_id="epa_hub",
        )
        runner_up = make_record(
            factor_id="EF:US:diesel:runner:v1",
            family="emissions",
            vectors=_vectors(CO2=10.25, CH4=0.00080, N2O=0.000160),
            co2e_total=10.28,
            source_id="ipcc",
        )

        engine = ResolutionEngine(
            candidate_source=make_candidate_source(
                {"country_or_sector_average": [winner, runner_up]}
            ),
            tenant_overlay_reader=lambda req: None,
        )
        resolved = engine.resolve(_base_request())

        assert len(resolved.alternates) >= 1, (
            "N3 violation: the resolver had 2 eligible candidates but surfaced "
            "0 alternates. Callers cannot audit the choice if runner-ups are "
            "hidden."
        )
        alt = resolved.alternates[0]
        assert alt.factor_id, (
            "N3 violation: alternate is missing factor_id. Auditors cannot "
            "look up the runner-up."
        )
        assert alt.why_not_chosen and alt.why_not_chosen.strip(), (
            "N3 violation: alternate.why_not_chosen is empty. A runner-up with "
            "no justification is invisible to auditors."
        )


# ---------------------------------------------------------------------------
# Gate: the explain() payload (wire format) includes the derivation block.
# ---------------------------------------------------------------------------


class TestN3ExplainPayloadHasDerivation:
    """The ``/explain`` wire payload must include every N3 field."""

    def test_explain_dict_contains_derivation_keys(
        self, make_record, make_candidate_source
    ):
        from tests.factors.gates.conftest import _vectors

        rec = make_record(
            factor_id="EF:US:diesel:2024:v1",
            family="emissions",
            vectors=_vectors(CO2=10.18, CH4=0.00082, N2O=0.000164),
            co2e_total=10.2097,
        )
        runner_up = make_record(
            factor_id="EF:US:diesel:runner:v1",
            family="emissions",
            vectors=_vectors(CO2=10.25),
            co2e_total=10.28,
        )
        engine = ResolutionEngine(
            candidate_source=make_candidate_source(
                {"country_or_sector_average": [rec, runner_up]}
            ),
            tenant_overlay_reader=lambda req: None,
        )
        payload = engine.resolve(_base_request()).explain()

        derivation = payload.get("derivation") or {}
        for key in ("fallback_rank", "step_label", "why_chosen"):
            assert key in derivation and derivation[key] not in (None, ""), (
                f"N3 violation: explain() payload is missing or empty key "
                f"`derivation.{key}`. Keys present: {sorted(derivation)}"
            )
        assert derivation["fallback_rank"] in range(1, 8), (
            "N3 violation: explain().derivation.fallback_rank not in 1..7. "
            f"Got {derivation['fallback_rank']!r}"
        )
        assert derivation["step_label"] in CANONICAL_STEPS, (
            "N3 violation: explain().derivation.step_label not a canonical "
            f"tier. Got {derivation['step_label']!r}"
        )
        assert isinstance(payload.get("alternates"), list), (
            "N3 violation: explain() payload must expose an `alternates` list. "
            f"Got {type(payload.get('alternates'))}"
        )
        assert len(payload["alternates"]) >= 1, (
            "N3 violation: explain() payload has zero alternates even though "
            "the resolver had >=2 eligible candidates."
        )
