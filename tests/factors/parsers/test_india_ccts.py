# -*- coding: utf-8 -*-
"""Tests for the India Carbon Credit Trading Scheme (CCTS) baseline parser.

Covers:
* round-trip of the eight obligated-sector baseline rows
* sector-name synonyms map to the same canonical sector
* invalid sector is logged and skipped
* factors carry the correct MethodProfile / FactorFamily / compliance frameworks
* source_registry row ``india_ccts_baselines`` has ``license_class: public_in_government``
* seed YAML ``emission_factors_expansion_phase6_india_ccts.yaml`` parses
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
    RedistributionClass,
    VerificationStatus,
)
from greenlang.factors.ingestion.parsers.india_ccts import parse_india_ccts_rows


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SEED_YAML = (
    _REPO_ROOT / "data" / "emission_factors_expansion_phase6_india_ccts.yaml"
)
_SOURCE_REGISTRY = (
    _REPO_ROOT / "greenlang" / "factors" / "data" / "source_registry.yaml"
)


def _baseline_row(sector: str, sub_sector: str, intensity: float) -> dict:
    return {
        "sector": sector,
        "sub_sector": sub_sector,
        "baseline_intensity_tco2_per_tonne_output": intensity,
        "unit_of_output": f"tonne_{sub_sector}",
        "target_year": "2025-26",
        "compliance_cycle": "CCTS-1",
        "obligated_entity_scope": "test_scope",
        "notification_reference": "BEE/test/2024",
    }


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------


class TestParseCCTSBaseline:
    def test_cement_row_parses(self):
        rows = [_baseline_row("cement", "OPC_grade_53", 0.82)]
        records = parse_india_ccts_rows(rows)
        assert len(records) == 1
        rec = records[0]

        # Identity / framework
        assert rec.factor_family == FactorFamily.EMISSIONS.value
        assert rec.method_profile == MethodProfile.INDIA_CCTS.value
        assert rec.formula_type == FormulaType.DIRECT_FACTOR.value
        assert rec.redistribution_class == RedistributionClass.OPEN.value
        assert rec.license_class == "public_in_government"

        # Geography + scope
        assert rec.geography == "IN"
        assert rec.scope.value == "1"

        # Intensity: 0.82 tCO2/tonne -> 820 kg CO2/tonne
        assert float(rec.vectors.CO2) == pytest.approx(820.0)
        assert rec.unit == "tonne_cement"

        # Compliance tags
        assert "India_CCTS" in rec.compliance_frameworks
        assert "India_BRSR" in rec.compliance_frameworks

        # Verification
        assert rec.verification.status == VerificationStatus.REGULATOR_APPROVED
        assert "BEE" in rec.verification.verified_by

    def test_all_eight_sectors_parse(self):
        sectors = [
            ("cement", "OPC", 0.80),
            ("iron_and_steel", "BF_BOF", 2.4),
            ("aluminum", "prebake", 16.0),
            ("pulp_and_paper", "kraft", 1.1),
            ("petrochemicals", "ethylene", 1.5),
            ("fertilizer", "urea", 1.4),
            ("chlor_alkali", "membrane", 1.5),
            ("textiles", "cotton", 3.0),
        ]
        rows = [_baseline_row(s, sub, val) for s, sub, val in sectors]
        records = parse_india_ccts_rows(rows)
        assert len(records) == 8
        # All go to MethodProfile.INDIA_CCTS
        assert {rec.method_profile for rec in records} == {
            MethodProfile.INDIA_CCTS.value
        }
        # Each factor_id is unique
        ids = {rec.factor_id for rec in records}
        assert len(ids) == 8
        assert all(fid.startswith("EF:IN:ccts:") for fid in ids)

    def test_valid_from_uses_indian_fy_start(self):
        rec = parse_india_ccts_rows(
            [_baseline_row("cement", "OPC_grade_53", 0.82)]
        )[0]
        # FY 2025-26 -> valid_from 2025-04-01, valid_to 2026-03-31
        assert rec.valid_from == date(2025, 4, 1)
        assert rec.valid_to == date(2026, 3, 31)


class TestSectorSynonyms:
    def test_aluminium_maps_to_aluminum(self):
        rec_us = parse_india_ccts_rows(
            [_baseline_row("aluminum", "prebake", 16.0)]
        )[0]
        rec_uk = parse_india_ccts_rows(
            [_baseline_row("aluminium", "prebake", 16.0)]
        )[0]
        # Factor names should resolve to the same display string
        assert rec_us.factor_name.split(",")[0] == rec_uk.factor_name.split(",")[0]
        # Both emit Aluminum NIC classification
        assert "NIC:2420" in rec_us.activity_schema.classification_codes
        assert "NIC:2420" in rec_uk.activity_schema.classification_codes

    def test_steel_short_form_accepted(self):
        rec = parse_india_ccts_rows(
            [_baseline_row("steel", "BF_BOF", 2.4)]
        )[0]
        assert "NIC:2410" in rec.activity_schema.classification_codes


class TestInvalidSectorSkipped:
    def test_unknown_sector_logged_and_skipped(self, caplog):
        rows = [
            _baseline_row("cement", "OPC", 0.82),
            _baseline_row("aerospace", "rocket_engine", 12.0),
            _baseline_row("textiles", "cotton", 3.0),
        ]
        out = parse_india_ccts_rows(rows)
        # Only two valid rows survive
        assert len(out) == 2
        assert any("aerospace" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# Source registry row presence
# ---------------------------------------------------------------------------


class TestSourceRegistryRow:
    @pytest.fixture(scope="class")
    def registry(self) -> dict:
        with _SOURCE_REGISTRY.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_india_ccts_row_registered(self, registry):
        ids = {row["source_id"] for row in registry["sources"]}
        assert "india_ccts_baselines" in ids

    def test_license_class_is_public_in_government(self, registry):
        row = next(
            r for r in registry["sources"] if r["source_id"] == "india_ccts_baselines"
        )
        assert row["license_class"] == "public_in_government"
        assert row["redistribution_allowed"] is True
        assert row["attribution_required"] is True

    def test_citation_text_references_regulation(self, registry):
        row = next(
            r for r in registry["sources"] if r["source_id"] == "india_ccts_baselines"
        )
        # The citation must reference the MoEFCC notification as the authority
        assert "G.S.R. 443(E)" in row["citation_text"]
        assert "Bureau of Energy Efficiency" in row["citation_text"]

    def test_watch_mechanism_present(self, registry):
        row = next(
            r for r in registry["sources"] if r["source_id"] == "india_ccts_baselines"
        )
        assert row["watch"]["mechanism"] in ("http_head", "html_diff", "api", "manual")
        assert row["watch"]["url"] is not None


# ---------------------------------------------------------------------------
# Seed YAML
# ---------------------------------------------------------------------------


class TestSeedYAML:
    @pytest.fixture(scope="class")
    def seed(self) -> dict:
        with _SEED_YAML.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_seed_has_all_eight_sectors(self, seed):
        sectors = {row["sector"] for row in seed["factors"]}
        expected = {
            "cement",
            "iron_and_steel",
            "aluminum",
            "pulp_and_paper",
            "petrochemicals",
            "fertilizer",
            "chlor_alkali",
            "textiles",
        }
        assert expected.issubset(sectors)

    def test_seed_rows_roundtrip_through_parser(self, seed):
        records = parse_india_ccts_rows(seed["factors"])
        # Seed has 24 rows; all should parse
        assert len(records) == len(seed["factors"])
        # Every record has India CCTS profile
        for rec in records:
            assert rec.method_profile == MethodProfile.INDIA_CCTS.value
            assert rec.geography == "IN"

    def test_seed_metadata_declares_compliance_cycle(self, seed):
        assert seed["metadata"]["compliance_cycle"] == "CCTS-1"
        assert seed["metadata"]["target_year"] == "2025-26"
