# -*- coding: utf-8 -*-
"""End-to-end wiring tests for the ``/explain`` audit-text pipeline.

Verifies that :class:`ResolutionEngine` calls
:func:`greenlang.factors.method_packs.render_audit_text` after a successful
resolve, populates the ``audit_text`` + ``audit_text_draft`` fields on the
:class:`ResolvedFactor`, and that the SAFE-DRAFT banner state propagates
from the template frontmatter into the ``audit_text_draft`` flag.

See ``docs/specs/audit_text_template_policy.md`` for the policy contract.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any, Iterable, List

import pytest

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
    RedistributionClass,
    Verification,
    VerificationStatus,
)
from greenlang.factors.method_packs import AUDIT_DRAFT_BANNER
from greenlang.factors.resolution import ResolutionEngine, ResolutionRequest


# ---------------------------------------------------------------------------
# Fixtures — synthetic candidate record + in-memory candidate source.
# ---------------------------------------------------------------------------


def _mk_record(
    *,
    factor_id: str = "EF:TEST:DIESEL:2024:v1",
    factor_name: str = "Diesel combustion (test)",
    source_authority: str = "DEFRA",
    source_version: str = "2024.1",
    source_year: int = 2024,
    geography: str = "US",
    verification_status: str = "regulator_approved",
    redistribution_class: str = RedistributionClass.OPEN.value,
) -> SimpleNamespace:
    return SimpleNamespace(
        factor_id=factor_id,
        factor_name=factor_name,
        factor_family=FactorFamily.EMISSIONS.value,
        formula_type=FormulaType.DIRECT_FACTOR.value,
        geography=geography,
        valid_to=date(2099, 12, 31),
        source_id="defra_hub",
        redistribution_class=redistribution_class,
        factor_status="certified",
        verification=Verification(status=VerificationStatus(verification_status)),
        uncertainty_95ci=0.05,
        unit="kWh",
        vectors=SimpleNamespace(
            CO2=0.5, CH4=0.0, N2O=0.0, HFCs=0.0, PFCs=0.0, SF6=0.0, NF3=0.0,
            biogenic_CO2=0.0,
        ),
        gwp_100yr=SimpleNamespace(co2e_total=0.5),
        provenance=SimpleNamespace(
            source_year=source_year,
            source_org=source_authority,
            source_publication=f"{source_authority} {source_version}",
            version=source_version,
        ),
        explainability=SimpleNamespace(assumptions=[], rationale=None),
        dqs=SimpleNamespace(overall_score=85.0),
        uncertainty_distribution="normal",
        replacement_factor_id=None,
        source_release=source_version,
    )


def _source_returning(records_by_step: dict):
    def source(_req: ResolutionRequest, label: str) -> Iterable[Any]:
        return list(records_by_step.get(label, []))

    return source


def _resolve_demo() -> Any:
    """Resolve the canonical demo (Corporate Scope 1, US diesel)."""
    engine = ResolutionEngine(
        candidate_source=_source_returning(
            {"country_or_sector_average": [_mk_record()]}
        )
    )
    return engine.resolve(
        ResolutionRequest(
            activity="diesel combustion stationary",
            method_profile=MethodProfile.CORPORATE_SCOPE1,
            jurisdiction="US",
            reporting_date="2026-06-01",
        )
    )


# ---------------------------------------------------------------------------
# End-to-end wiring.
# ---------------------------------------------------------------------------


class TestAuditTextWiring:
    def test_resolver_populates_audit_text_field(self):
        """After a successful resolve, ``audit_text`` is a non-empty string."""
        resolved = _resolve_demo()
        assert resolved.audit_text is not None
        assert isinstance(resolved.audit_text, str)
        assert len(resolved.audit_text.strip()) > 0

    def test_draft_flag_true_for_unapproved_corporate_template(self):
        """``corporate.j2`` ships with ``approved: false`` — banner + flag expected."""
        resolved = _resolve_demo()
        assert resolved.audit_text_draft is True
        assert resolved.audit_text.startswith(AUDIT_DRAFT_BANNER)
        # The banner text is the exact SAFE-DRAFT prefix from the spec.
        assert "[Draft — Methodology Review Required" in resolved.audit_text

    def test_placeholder_substitution_expands_chosen_factor_name(self):
        """The rendered audit text MUST reference the chosen factor's name."""
        resolved = _resolve_demo()
        assert resolved.chosen_factor is not None
        assert resolved.chosen_factor.name in resolved.audit_text

    def test_explain_payload_carries_audit_text(self):
        """The ``/explain`` dict form exposes both ``audit_text`` and the flag."""
        resolved = _resolve_demo()
        payload = resolved.explain()
        assert "audit_text" in payload
        assert "audit_text_draft" in payload
        assert payload["audit_text"] == resolved.audit_text
        assert payload["audit_text_draft"] is True


# ---------------------------------------------------------------------------
# Approved template flip — simulated via a tmp file.
# ---------------------------------------------------------------------------


class TestApprovedTemplateSuppressesBanner:
    def test_flipping_frontmatter_to_approved_removes_banner(
        self, tmp_path, monkeypatch
    ):
        """Copy the corporate template into a tmp dir with ``approved: true``
        frontmatter. Resolve the demo; ``audit_text_draft`` MUST be False
        and the banner MUST be absent from the rendered text."""
        from greenlang.factors import method_packs as mp

        # Copy a MINIMAL approved template so we don't depend on
        # corporate.j2's optional placeholders (factor.scope, fallback_trace
        # etc.) that aren't populated on the ResolvedFactor.
        approved_template = (
            "{# ---\n"
            "approved: true\n"
            "approved_by: methodology-wg@greenlang.io\n"
            "approved_at: 2026-04-23\n"
            "methodology_lead: jane.doe\n"
            "--- #}\n"
            "Factor {{ factor.chosen_factor.name }} "
            "({{ factor.chosen_factor.id }}) selected under method pack "
            "{{ factor.method_pack }} at fallback tier "
            "{{ factor.fallback_rank }}.\n"
        )
        # The resolver maps method_profile "corporate_scope1" (and _scope2,
        # _scope3) to the shared ``corporate`` template family — see
        # ``_AUDIT_TEMPLATE_KEY_MAP`` in ``resolution/engine.py``.
        (tmp_path / "corporate.j2").write_text(
            approved_template, encoding="utf-8"
        )
        monkeypatch.setattr(mp, "_AUDIT_TEMPLATE_DIR", tmp_path)

        resolved = _resolve_demo()
        assert resolved.audit_text is not None
        assert resolved.audit_text_draft is False
        assert not resolved.audit_text.startswith(AUDIT_DRAFT_BANNER)
        assert "[Draft — Methodology Review Required" not in resolved.audit_text
        # Placeholder substitution still fires.
        assert "Diesel combustion (test)" in resolved.audit_text

    def test_missing_template_leaves_fields_none_without_crashing(
        self, tmp_path, monkeypatch
    ):
        """A pack with no on-disk template MUST NOT break resolve()."""
        from greenlang.factors import method_packs as mp

        # Empty dir — no template files at all.
        monkeypatch.setattr(mp, "_AUDIT_TEMPLATE_DIR", tmp_path)

        resolved = _resolve_demo()
        assert resolved.audit_text is None
        assert resolved.audit_text_draft is None
        # But every other field is fully populated.
        assert resolved.chosen_factor_id
        assert resolved.fallback_rank == 5
