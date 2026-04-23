# -*- coding: utf-8 -*-
"""
Wave 2 resolver contract tests — canonical demo end-to-end.

Verifies that resolving the canonical demo input
    {activity: "electricity grid consumption",
     jurisdiction: "IN",
     method_profile: "corporate_scope2_location_based",
     quantity: 12500, unit: "kWh",
     reporting_date: "2027-04-01"}
produces a ResolvedFactor populated with every one of the 16 CTO contract
fields AND (when surfaced through the signed-receipts middleware) a
spec-compliant ``signed_receipt`` envelope.

Scope ownership: Wave 2 resolver-contract PR. Fields populated from
migrated EmissionFactorRecord attributes (Wave 2b) are allowed to be
None — the contract-level envelope is still required to be present.
"""
from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from typing import Any, Iterable, List

import pytest

pytest.importorskip("fastapi")


# ---------------------------------------------------------------------------
# Canonical demo input — single source of truth for this file.
# ---------------------------------------------------------------------------


CANONICAL_DEMO = {
    "activity": "electricity grid consumption",
    "jurisdiction": "IN",
    "method_profile": "corporate_scope2_location_based",
    "quantity": 12500,
    "unit": "kWh",
    "reporting_date": "2027-04-01",
}


# ---------------------------------------------------------------------------
# Synthetic India grid factor — the built-in EmissionFactorDatabase does
# not ship an India row, so we inject one via the engine's candidate
# source.  Keeps the test self-contained (no reliance on ingest state).
# ---------------------------------------------------------------------------


def _india_grid_factor() -> SimpleNamespace:
    """Construct a minimally-realistic India grid-electricity record."""
    from greenlang.data.canonical_v2 import (
        FactorFamily,
        FormulaType,
        RedistributionClass,
        Verification,
        VerificationStatus,
    )

    return SimpleNamespace(
        factor_id="EF:IN:electricity:grid:2027:v1",
        factor_name="India national grid electricity - location-based",
        # Use the legacy DB-level enum "grid_intensity"; the resolver's
        # _family_from_profile helper maps the chosen_factor.factor_family
        # to the CTO-canonical "electricity" via the method profile.
        factor_family=FactorFamily.GRID_INTENSITY.value,
        activity_category="electricity",
        scope="2",
        boundary="combustion",
        factor_version="1.0.0",
        release_version="2027.04.0",
        formula_type=FormulaType.DIRECT_FACTOR.value,
        geography="IN",
        valid_from=date(2027, 4, 1),
        valid_to=date(2028, 3, 31),
        source_id="cea-india-2027",
        source_release="2027.q1",
        # Legacy enum stores "licensed" — the resolver engine maps it onto
        # the CTO canonical "licensed_embedded" class.
        redistribution_class=RedistributionClass.LICENSED.value,
        factor_status="certified",
        verification=Verification(status=VerificationStatus("regulator_approved")),
        uncertainty_95ci=0.05,
        uncertainty_distribution="lognormal",
        unit="kWh",
        # GHG vectors
        vectors=SimpleNamespace(
            CO2=0.697, CH4=0.0, N2O=0.0, HFCs=0.0, PFCs=0.0, SF6=0.0, NF3=0.0,
            biogenic_CO2=0.0,
        ),
        gwp_100yr=SimpleNamespace(co2e_total=0.712),
        provenance=SimpleNamespace(
            source_org="Government of India - Ministry of Power",
            source_publication="CEA - Central Electricity Authority of India",
            source_year=2027,
            version="2027.q1",
        ),
        # Full 5-dim DQS so compute_fqs can produce all per-dimension 0-100.
        dqs=SimpleNamespace(
            temporal=5,
            geographical=5,
            technological=4,
            representativeness=3,
            methodological=4,
            overall_score=4.2,
        ),
        explainability=SimpleNamespace(
            assumptions=[
                "Location-based method applied (scope 2, corporate boundary).",
                "Grid losses already embedded in CEA published value.",
            ],
            rationale=(
                "Step 5 (country_or_sector_average) - India national grid "
                "factor won on geo-proximity (exact match) and newest vintage."
            ),
        ),
        replacement_factor_id=None,
        primary_data_flag="secondary",
    )


def _india_source(_req, label: str) -> Iterable[Any]:
    """Candidate source: emit the India grid factor only for step 5."""
    if label == "country_or_sector_average":
        return [_india_grid_factor()]
    return []


# ---------------------------------------------------------------------------
# Core canonical-demo resolve fixture.
# ---------------------------------------------------------------------------


@pytest.fixture
def resolved_canonical_demo():
    """Run the canonical demo through a ResolutionEngine and return the payload."""
    from greenlang.factors.resolution.engine import ResolutionEngine
    from greenlang.factors.resolution.request import ResolutionRequest

    engine = ResolutionEngine(candidate_source=_india_source)
    req = ResolutionRequest(
        activity=CANONICAL_DEMO["activity"],
        jurisdiction=CANONICAL_DEMO["jurisdiction"],
        method_profile=CANONICAL_DEMO["method_profile"],
        target_unit=CANONICAL_DEMO["unit"],
        reporting_date=CANONICAL_DEMO["reporting_date"],
        extras={"quantity": CANONICAL_DEMO["quantity"]},
    )
    return engine.resolve(req)


# ---------------------------------------------------------------------------
# 16-element contract coverage — field-by-field.
# ---------------------------------------------------------------------------


class TestSixteenFieldContract:
    """Each test below pins one of the 16 CTO-required contract fields."""

    # --- #1 chosen_factor envelope ---
    def test_1_chosen_factor_envelope_is_populated(self, resolved_canonical_demo):
        cf = resolved_canonical_demo.chosen_factor
        assert cf is not None
        assert cf.id == "EF:IN:electricity:grid:2027:v1"
        assert cf.name
        assert cf.version == "1.0.0"
        assert cf.factor_family == "electricity"

    # --- #2 alternates[] ---
    def test_2_alternates_is_a_list(self, resolved_canonical_demo):
        # No other candidates in the synthetic source, so list is empty,
        # but it MUST be present as a list (not None) and every entry must
        # carry the CTO-spec reason_lost alias.
        alts = resolved_canonical_demo.alternates
        assert isinstance(alts, list)
        for a in alts:
            assert a.reason_lost is not None

    # --- #3 why_this_won ---
    def test_3_why_this_won_non_empty(self, resolved_canonical_demo):
        assert resolved_canonical_demo.why_this_won
        assert isinstance(resolved_canonical_demo.why_this_won, str)
        assert len(resolved_canonical_demo.why_this_won) > 0

    # --- #4 source envelope ---
    def test_4_source_envelope_full(self, resolved_canonical_demo):
        src = resolved_canonical_demo.source
        assert src is not None
        assert src.id == "cea-india-2027"
        assert src.version == "2027.q1"
        assert src.name
        assert src.authority

    # --- #5 factor_version AND release_version distinct ---
    def test_5_factor_version_and_release_version(self, resolved_canonical_demo):
        r = resolved_canonical_demo
        assert r.factor_version == "1.0.0"
        assert r.release_version == "2027.04.0"
        assert r.factor_version != r.release_version

    # --- #6 method_pack + method_pack_version ---
    def test_6_method_pack_and_version(self, resolved_canonical_demo):
        r = resolved_canonical_demo
        assert r.method_pack
        assert r.method_pack_version

    # --- #7 valid_from / valid_to ---
    def test_7_valid_dates(self, resolved_canonical_demo):
        r = resolved_canonical_demo
        assert r.valid_from == date(2027, 4, 1)
        assert r.valid_to == date(2028, 3, 31)

    # --- #8 gas_breakdown ---
    def test_8_gas_breakdown_has_all_gases(self, resolved_canonical_demo):
        gb = resolved_canonical_demo.gas_breakdown
        # All 9 gas-specific fields present.
        for fld in (
            "co2_kg", "ch4_kg", "n2o_kg", "hfcs_kg", "pfcs_kg",
            "sf6_kg", "nf3_kg", "biogenic_co2_kg", "co2e_total_kg",
        ):
            assert hasattr(gb, fld), f"missing {fld}"
        assert gb.gwp_basis
        # CTO-spec aliases (no _kg suffix).
        assert gb.co2 == gb.co2_kg
        assert gb.co2e_total == gb.co2e_total_kg

    # --- #9 co2e_basis top-level ---
    def test_9_co2e_basis_top_level(self, resolved_canonical_demo):
        r = resolved_canonical_demo
        assert r.co2e_basis
        assert "AR6" in r.co2e_basis or "AR5" in r.co2e_basis or "AR4" in r.co2e_basis

    # --- #10 quality envelope (composite FQS 0-100 + 5 per-dim) ---
    def test_10_quality_envelope_has_composite_and_five_dims(
        self, resolved_canonical_demo
    ):
        q = resolved_canonical_demo.quality
        assert q is not None
        assert 0.0 <= q.composite_fqs_0_100 <= 100.0
        assert 0.0 <= q.temporal_score <= 100.0
        assert 0.0 <= q.geographic_score <= 100.0
        assert 0.0 <= q.technology_score <= 100.0
        assert 0.0 <= q.verification_score <= 100.0
        assert 0.0 <= q.completeness_score <= 100.0

    # --- #11 uncertainty envelope with type ---
    def test_11_uncertainty_envelope_typed(self, resolved_canonical_demo):
        u = resolved_canonical_demo.uncertainty
        assert u.type in ("95_percent_ci", "qualitative")
        if u.type == "95_percent_ci":
            assert u.low is not None
            assert u.high is not None
            assert u.low <= u.high

    # --- #12 licensing envelope ---
    def test_12_licensing_envelope_enum(self, resolved_canonical_demo):
        lic = resolved_canonical_demo.licensing
        assert lic is not None
        assert lic.redistribution_class in (
            "open", "licensed_embedded", "customer_private", "oem_redistributable",
        )

    # --- #13 assumptions[] ---
    def test_13_assumptions_is_list(self, resolved_canonical_demo):
        ass = resolved_canonical_demo.assumptions
        assert isinstance(ass, list)
        # Empty is allowed per spec; non-empty in our fixture.
        assert len(ass) >= 0

    # --- #14 fallback_rank in 1..7 ---
    def test_14_fallback_rank_in_1_7(self, resolved_canonical_demo):
        rank = resolved_canonical_demo.fallback_rank
        assert isinstance(rank, int)
        assert 1 <= rank <= 7
        # Canonical demo hits step 5 (country_or_sector_average).
        assert rank == 5

    # --- #15 deprecation_status envelope ---
    def test_15_deprecation_status_envelope(self, resolved_canonical_demo):
        ds = resolved_canonical_demo.deprecation_status
        assert ds is not None
        assert ds.status in ("active", "deprecated", "superseded")
        # Active → no replacement pointer.
        if ds.status == "active":
            assert ds.replacement_pointer_factor_id is None

    # --- #16 signed_receipt — covered in test_signed_receipt_shape.py ---
    # The resolved factor is the pre-middleware object; signed_receipt
    # injection lives in the HTTP layer.  We exercise it there.


# ---------------------------------------------------------------------------
# Structural / sanity checks on the envelope as a whole.
# ---------------------------------------------------------------------------


class TestEnvelopeStructure:
    def test_all_envelopes_serialize_to_json(self, resolved_canonical_demo):
        """model_dump() round-trips with no encoding errors (dates as ISO)."""
        payload = resolved_canonical_demo.model_dump(mode="json")
        assert json.dumps(payload)  # must be JSON-serializable
        # Key top-level envelopes are present.
        for key in (
            "chosen_factor", "source", "quality", "licensing",
            "deprecation_status", "uncertainty", "gas_breakdown",
            "valid_from", "valid_to", "co2e_basis", "method_pack",
            "release_version", "why_this_won", "fallback_rank",
            "alternates", "assumptions",
        ):
            assert key in payload, f"missing top-level key: {key}"

    def test_explain_view_exposes_canonical_fields(self, resolved_canonical_demo):
        ex = resolved_canonical_demo.explain()
        assert ex["chosen"]["factor_family"] == "electricity"
        assert ex["chosen"]["method_pack"]
        assert ex["chosen"]["valid_from"] == "2027-04-01"
        assert ex["derivation"]["fallback_rank"] == 5
        assert ex["derivation"]["why_this_won"]
        assert ex["co2e_basis"]
        # Quality dict carries composite FQS + per-dim scores.
        q = ex["quality"]
        assert "composite_fqs_0_100" in q
        assert 0.0 <= q["composite_fqs_0_100"] <= 100.0


# ---------------------------------------------------------------------------
# Signed receipt presence — exercised via the FastAPI middleware layer.
#
# We build a tiny app, mount SignedReceiptsMiddleware, and return the
# resolved canonical demo payload to confirm the 4 required receipt keys
# are emitted.  End-to-end verification of old/new key names lives in
# test_signed_receipt_shape.py.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def _signing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GL_FACTORS_SIGNING_SECRET", "test-secret-resolver-contract")
    monkeypatch.delenv("GL_FACTORS_ED25519_PRIVATE_KEY", raising=False)


def _build_signed_app(resolved_payload: dict):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from greenlang.factors.middleware.signed_receipts import SignedReceiptsMiddleware

    app = FastAPI()
    app.add_middleware(SignedReceiptsMiddleware)

    @app.get("/v1/resolve-demo")
    async def _demo():
        return JSONResponse(resolved_payload, headers={"X-GreenLang-Edition": "factors-ga-2027.04.0"})

    return app


def test_signed_receipt_carries_four_required_keys(
    resolved_canonical_demo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The signed_receipt envelope carries receipt_id/signature/verification_key_hint/alg."""
    monkeypatch.setenv("GL_FACTORS_SIGNING_SECRET", "test-secret-resolver-contract")
    monkeypatch.delenv("GL_FACTORS_ED25519_PRIVATE_KEY", raising=False)
    from fastapi.testclient import TestClient

    payload = resolved_canonical_demo.model_dump(mode="json")
    client = TestClient(_build_signed_app(payload))
    resp = client.get("/v1/resolve-demo")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "signed_receipt" in body, f"signed_receipt missing: {list(body.keys())}"
    sr = body["signed_receipt"]
    for key in ("receipt_id", "signature", "verification_key_hint", "alg"):
        assert key in sr, f"signed_receipt missing required key {key!r}"
    assert sr["receipt_id"]  # UUID v4
    assert sr["signature"]
    # verification_key_hint is 16 hex chars.
    assert len(sr["verification_key_hint"]) == 16
    int(sr["verification_key_hint"], 16)  # must be hex
    # alg is one of the supported algorithms.
    assert sr["alg"] in ("sha256-hmac", "ed25519")
