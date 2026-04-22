# -*- coding: utf-8 -*-
"""Tests for the 5 new residual-mix pack variants.

Covers:
* registry rows for the 5 new source_ids exist with license class + citation
* country -> pack routing picks the expected variant
* each variant's SelectionRule.custom_filter enforces the jurisdiction filter:
  accepts only factors whose source_id + geography match
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from greenlang.factors.method_packs.electricity import (
    ELECTRICITY_RESIDUAL_MIX_AU_STATE,
    ELECTRICITY_RESIDUAL_MIX_CA,
    ELECTRICITY_RESIDUAL_MIX_KR,
    ELECTRICITY_RESIDUAL_MIX_SG,
    ELECTRICITY_RESIDUAL_MIX_UK_NATIONAL,
    RESIDUAL_MIX_PACKS_BY_COUNTRY,
    get_residual_mix_pack,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_REGISTRY = (
    _REPO_ROOT / "greenlang" / "factors" / "data" / "source_registry.yaml"
)


# ---------------------------------------------------------------------------
# Source-registry rows
# ---------------------------------------------------------------------------


class TestSourceRegistryRows:
    @pytest.fixture(scope="class")
    def registry(self) -> dict:
        with _SOURCE_REGISTRY.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    @pytest.mark.parametrize(
        "source_id",
        [
            "cer_canada_residual",
            "beis_uk_residual",
            "nger_au_state_residual",
            "kemco_korea_residual",
            "ema_singapore_residual",
        ],
    )
    def test_row_registered(self, registry, source_id):
        ids = {row["source_id"] for row in registry["sources"]}
        assert source_id in ids, f"Missing source_registry row: {source_id}"

    @pytest.mark.parametrize(
        "source_id",
        [
            "cer_canada_residual",
            "beis_uk_residual",
            "nger_au_state_residual",
            "kemco_korea_residual",
            "ema_singapore_residual",
        ],
    )
    def test_row_has_license_citation_watch(self, registry, source_id):
        row = next(r for r in registry["sources"] if r["source_id"] == source_id)
        assert "license_class" in row
        assert row["citation_text"]
        assert "watch" in row
        assert row["watch"].get("mechanism") in (
            "http_head",
            "html_diff",
            "api",
            "manual",
            "internal",
        )

    def test_cer_canada_license_is_open(self, registry):
        row = next(
            r for r in registry["sources"] if r["source_id"] == "cer_canada_residual"
        )
        assert row["license_class"] == "open"
        assert "Canada" in row["citation_text"]

    def test_uk_license_is_open_gov(self, registry):
        row = next(
            r for r in registry["sources"] if r["source_id"] == "beis_uk_residual"
        )
        assert row["license_class"] == "uk_open_government"
        assert "DESNZ" in row["citation_text"]

    def test_nger_license_is_open(self, registry):
        row = next(
            r
            for r in registry["sources"]
            if r["source_id"] == "nger_au_state_residual"
        )
        assert row["license_class"] == "open"

    def test_kemco_license_is_open(self, registry):
        row = next(
            r
            for r in registry["sources"]
            if r["source_id"] == "kemco_korea_residual"
        )
        assert row["license_class"] == "open"

    def test_ema_license_is_open(self, registry):
        row = next(
            r
            for r in registry["sources"]
            if r["source_id"] == "ema_singapore_residual"
        )
        assert row["license_class"] == "open"
        assert "Singapore" in row["citation_text"]


# ---------------------------------------------------------------------------
# Country -> pack routing
# ---------------------------------------------------------------------------


class TestCountryRouting:
    @pytest.mark.parametrize(
        "country,expected_pack",
        [
            ("CA", ELECTRICITY_RESIDUAL_MIX_CA),
            ("UK", ELECTRICITY_RESIDUAL_MIX_UK_NATIONAL),
            ("GB", ELECTRICITY_RESIDUAL_MIX_UK_NATIONAL),
            ("KR", ELECTRICITY_RESIDUAL_MIX_KR),
            ("SG", ELECTRICITY_RESIDUAL_MIX_SG),
        ],
    )
    def test_get_residual_mix_pack(self, country, expected_pack):
        assert get_residual_mix_pack(country) is expected_pack

    def test_routing_table_contains_new_countries(self):
        for country in ("CA", "UK", "GB", "KR", "SG"):
            assert country in RESIDUAL_MIX_PACKS_BY_COUNTRY

    def test_case_insensitive_routing(self):
        assert get_residual_mix_pack("kr") is ELECTRICITY_RESIDUAL_MIX_KR
        assert get_residual_mix_pack("sg") is ELECTRICITY_RESIDUAL_MIX_SG


# ---------------------------------------------------------------------------
# selection_rule jurisdiction filter
# ---------------------------------------------------------------------------


def _fake_record(source_id: str, geography: str):
    """Minimal record stub — the SelectionRule only inspects these attrs."""
    return SimpleNamespace(
        source_id=source_id,
        geography=geography,
        factor_status="certified",
        factor_family="residual_mix",
        formula_type="residual_mix",
    )


class TestJurisdictionFilter:
    @pytest.mark.parametrize(
        "pack,good_source,good_country,bad_source,bad_country",
        [
            (
                ELECTRICITY_RESIDUAL_MIX_CA,
                "cer_canada_residual",
                "CA",
                "green_e_residual_mix",
                "US",
            ),
            (
                ELECTRICITY_RESIDUAL_MIX_UK_NATIONAL,
                "beis_uk_residual",
                "UK",
                "aib_residual_mix_eu",
                "DE",
            ),
            (
                ELECTRICITY_RESIDUAL_MIX_AU_STATE,
                "nger_au_state_residual",
                "AU",
                "australia_nga_factors",
                "AU",
            ),
            (
                ELECTRICITY_RESIDUAL_MIX_KR,
                "kemco_korea_residual",
                "KR",
                "ema_singapore_residual",
                "SG",
            ),
            (
                ELECTRICITY_RESIDUAL_MIX_SG,
                "ema_singapore_residual",
                "SG",
                "kemco_korea_residual",
                "KR",
            ),
        ],
    )
    def test_filter_accepts_matching_and_rejects_cross_jurisdiction(
        self, pack, good_source, good_country, bad_source, bad_country
    ):
        good = _fake_record(good_source, good_country)
        bad = _fake_record(bad_source, bad_country)
        # Matching record is accepted
        assert pack.selection_rule.accepts(good) is True
        # Non-matching jurisdiction is rejected
        assert pack.selection_rule.accepts(bad) is False

    def test_filter_rejects_wrong_geography_even_with_right_source(self):
        # Good source_id but wrong country -> rejected
        rec = _fake_record("cer_canada_residual", "US")
        assert ELECTRICITY_RESIDUAL_MIX_CA.selection_rule.accepts(rec) is False

    def test_filter_rejects_wrong_source_even_with_right_country(self):
        # Right country but another residual-mix source -> rejected
        rec = _fake_record("green_e_residual_mix", "CA")
        assert ELECTRICITY_RESIDUAL_MIX_CA.selection_rule.accepts(rec) is False


# ---------------------------------------------------------------------------
# Pack metadata
# ---------------------------------------------------------------------------


class TestPackMetadata:
    @pytest.mark.parametrize(
        "pack,expected_label",
        [
            (ELECTRICITY_RESIDUAL_MIX_CA, "CSA_CSDS"),
            (ELECTRICITY_RESIDUAL_MIX_UK_NATIONAL, "UK_SECR"),
            (ELECTRICITY_RESIDUAL_MIX_AU_STATE, "ASRS"),
            (ELECTRICITY_RESIDUAL_MIX_KR, "Korea_KSSB"),
            (ELECTRICITY_RESIDUAL_MIX_SG, "SGX_711A"),
        ],
    )
    def test_jurisdiction_specific_reporting_label(self, pack, expected_label):
        assert expected_label in pack.reporting_labels

    @pytest.mark.parametrize(
        "pack",
        [
            ELECTRICITY_RESIDUAL_MIX_CA,
            ELECTRICITY_RESIDUAL_MIX_UK_NATIONAL,
            ELECTRICITY_RESIDUAL_MIX_AU_STATE,
            ELECTRICITY_RESIDUAL_MIX_KR,
            ELECTRICITY_RESIDUAL_MIX_SG,
        ],
    )
    def test_pack_audit_template_references_source_year(self, pack):
        # Every audit template must include {source_year} so the output
        # carries the vintage back to the auditor.
        assert "{source_year}" in pack.audit_text_template
