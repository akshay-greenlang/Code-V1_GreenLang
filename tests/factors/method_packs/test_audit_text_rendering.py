# -*- coding: utf-8 -*-
"""Tests for the audit-text SAFE-DRAFT renderer.

Covers:

* Unapproved template (``approved: false``) renders with the draft banner.
* Approved template (``approved: true``) renders WITHOUT the banner.
* Placeholder substitution (``{{ factor.chosen_factor.name }}``,
  ``{{ factor.source.authority }}``, ``{{ factor.method_pack }}``, etc.)
  works on a synthetic ResolvedFactor-like object.
* Frontmatter parser handles missing frontmatter gracefully (treats
  template as unapproved).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from greenlang.factors.method_packs import (
    AUDIT_DRAFT_BANNER,
    load_template,
    parse_frontmatter,
    render_audit_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_factor() -> SimpleNamespace:
    """Build a synthetic object that mimics a ResolvedFactor's dotted paths."""
    return SimpleNamespace(
        chosen_factor=SimpleNamespace(
            id="EF:TEST:DIESEL:2024:v1",
            name="Diesel combustion (test)",
        ),
        source=SimpleNamespace(
            authority="DEFRA",
            version="2024.1",
            vintage=2024,
        ),
        method_pack="corporate_scope1",
        fallback_rank=2,
        step_label="supplier_specific",
        gwp_basis="IPCC_AR6_100",
        biogenic_treatment="reported separately",
        scope="1",
        scope3_category=None,
        electricity_basis=None,
        geography="GB",
        grid_region=None,
        supplier=None,
        certificate=None,
        residual_mix_factor_id=None,
        calculation_method=None,
        dqs_score=None,
        cat11_use_phase_block=None,
        fallback_trace=[
            SimpleNamespace(rank=1, label="customer_override"),
            SimpleNamespace(rank=2, label="supplier_specific"),
        ],
        include_transmission_losses=False,
        instrument_type=None,
        certificate_vintage_months=None,
        quality_criteria_passed=None,
        pack_id="corporate_scope1",
        cn_code=None,
        article_4_2_justification=None,
        product_id=None,
        boundary=None,
        battery_class=None,
        battery_energy_kwh=None,
        battery_weight_kg=None,
        enforcement_date=None,
        dpp_id=None,
    )


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_parses_boolean_and_null(self):
        source = (
            "{# ---\n"
            "approved: false\n"
            "approved_by: null\n"
            "approved_at: null\n"
            "methodology_lead: null\n"
            "--- #}\n"
            "body here\n"
        )
        fm, body = parse_frontmatter(source)
        assert fm == {
            "approved": False,
            "approved_by": None,
            "approved_at": None,
            "methodology_lead": None,
        }
        assert body.strip() == "body here"

    def test_parses_true_and_quoted_strings(self):
        source = (
            "{# ---\n"
            'approved: true\n'
            'approved_by: "methodology-wg@greenlang.io"\n'
            "approved_at: 2026-04-23\n"
            "methodology_lead: jane.doe\n"
            "--- #}\n"
            "body\n"
        )
        fm, _ = parse_frontmatter(source)
        assert fm["approved"] is True
        assert fm["approved_by"] == "methodology-wg@greenlang.io"
        assert fm["approved_at"] == "2026-04-23"
        assert fm["methodology_lead"] == "jane.doe"

    def test_no_frontmatter_returns_empty_dict_and_unchanged_body(self):
        source = "Plain body with no frontmatter."
        fm, body = parse_frontmatter(source)
        assert fm == {}
        assert body == source


# ---------------------------------------------------------------------------
# Built-in templates (corporate / electricity / eu_policy)
# ---------------------------------------------------------------------------


class TestBuiltinTemplateRendering:
    """Each built-in template must render with the draft banner."""

    @pytest.mark.parametrize("pack_id", ["corporate", "electricity", "eu_policy"])
    def test_shipped_templates_are_unapproved_and_render_with_banner(self, pack_id):
        factor = _sample_factor()
        factor.pack_id = pack_id
        rendered = render_audit_text(pack_id, factor)
        assert rendered.startswith(AUDIT_DRAFT_BANNER), (
            f"Template {pack_id} is missing the draft banner — methodology "
            "review hasn't signed off yet, so the banner is mandatory."
        )

    @pytest.mark.parametrize("pack_id", ["corporate", "electricity", "eu_policy"])
    def test_shipped_templates_contain_frontmatter_marker(self, pack_id):
        """Every shipped template MUST have the frontmatter block so the
        renderer can detect approval state (absent frontmatter is treated
        as unapproved, so this is a correctness tripwire)."""
        source = load_template(pack_id)
        fm, _body = parse_frontmatter(source)
        assert "approved" in fm
        assert fm["approved"] is False  # SAFE-DRAFT default
        assert "methodology_lead" in fm

    def test_placeholder_substitution_works_on_sample_factor(self):
        factor = _sample_factor()
        rendered = render_audit_text("corporate", factor)
        assert "Diesel combustion (test)" in rendered
        assert "DEFRA" in rendered
        assert "2024.1" in rendered
        assert "corporate_scope1" in rendered

    def test_fallback_rank_renders(self):
        factor = _sample_factor()
        rendered = render_audit_text("electricity", factor)
        assert "fallback tier 2" in rendered

    def test_eu_policy_renders_cbam_branch_when_pack_id_is_eu_cbam(self):
        factor = _sample_factor()
        factor.pack_id = "eu_cbam"
        factor.cn_code = "7601"
        factor.article_4_2_justification = "operator primary data missing"
        rendered = render_audit_text("eu_policy", factor)
        assert "CBAM embedded emissions" in rendered
        assert "7601" in rendered
        # Because fallback_rank > 1, the Article 4(2) block is rendered.
        assert "Article 4(2) fallback notice" in rendered


# ---------------------------------------------------------------------------
# Approval flip — simulated via a tmp file so we don't mutate the shipped
# template on disk.
# ---------------------------------------------------------------------------


class TestApprovedTemplateRendering:
    def test_approved_template_omits_banner(self, tmp_path, monkeypatch):
        # Build an approved template in a tmp dir, then point the renderer
        # at it by monkey-patching ``_AUDIT_TEMPLATE_DIR``.
        from greenlang.factors import method_packs as mp

        template_body = (
            "{# ---\n"
            "approved: true\n"
            "approved_by: methodology-wg@greenlang.io\n"
            "approved_at: 2026-04-23\n"
            "methodology_lead: jane.doe\n"
            "--- #}\n"
            "Factor {{ factor.chosen_factor.name }} approved, no banner.\n"
        )
        (tmp_path / "fake_pack.j2").write_text(template_body, encoding="utf-8")

        monkeypatch.setattr(mp, "_AUDIT_TEMPLATE_DIR", tmp_path)

        factor = _sample_factor()
        rendered = render_audit_text("fake_pack", factor)
        assert not rendered.startswith(AUDIT_DRAFT_BANNER)
        assert "Diesel combustion (test) approved, no banner." in rendered

    def test_missing_template_raises_file_not_found(self, tmp_path, monkeypatch):
        from greenlang.factors import method_packs as mp

        monkeypatch.setattr(mp, "_AUDIT_TEMPLATE_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            render_audit_text("does_not_exist", _sample_factor())

    def test_template_without_frontmatter_defaults_to_draft(self, tmp_path, monkeypatch):
        from greenlang.factors import method_packs as mp

        (tmp_path / "no_fm.j2").write_text(
            "Body without any frontmatter: {{ factor.chosen_factor.name }}.",
            encoding="utf-8",
        )
        monkeypatch.setattr(mp, "_AUDIT_TEMPLATE_DIR", tmp_path)

        rendered = render_audit_text("no_fm", _sample_factor())
        # Missing frontmatter defaults to unapproved so the banner is
        # applied — this is the SAFE-DRAFT policy fail-closed behaviour.
        assert rendered.startswith(AUDIT_DRAFT_BANNER)
