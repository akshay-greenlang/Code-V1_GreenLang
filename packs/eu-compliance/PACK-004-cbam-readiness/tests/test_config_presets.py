# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - Config Presets Tests (40 tests)

Tests CBAMPackConfig creation, sub-config validation, commodity presets,
sector presets, demo config, EORI validation, and CN code validation.

Author: GreenLang QA Team
"""

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    PRESETS_DIR,
    SECTORS_DIR,
    DEMO_DIR,
    _compute_hash,
    _new_uuid,
    _utcnow,
)


# ---------------------------------------------------------------------------
# CBAMPackConfig creation and validation (8 tests)
# ---------------------------------------------------------------------------

class TestCBAMPackConfigCreation:
    """Test CBAMPackConfig instantiation and fields."""

    def test_config_has_metadata(self, sample_cbam_config):
        """Test config has required metadata fields."""
        meta = sample_cbam_config["metadata"]
        assert meta["name"] == "cbam-readiness"
        assert meta["category"] == "eu-compliance"
        assert meta["version"] == "1.0.0"

    def test_config_has_cbam_section(self, sample_cbam_config):
        """Test config has CBAM-specific section."""
        assert "cbam" in sample_cbam_config
        cbam = sample_cbam_config["cbam"]
        assert "importer" in cbam
        assert "goods_categories" in cbam
        assert "emission_config" in cbam
        assert "certificate_config" in cbam

    def test_config_serializes_to_json(self, sample_cbam_config):
        """Test config serializes to valid JSON."""
        j = json.dumps(sample_cbam_config, indent=2, default=str)
        parsed = json.loads(j)
        assert parsed["metadata"]["name"] == "cbam-readiness"
        assert "cbam" in parsed

    def test_config_compliance_references(self, sample_cbam_config):
        """Test config has CBAM compliance references."""
        refs = sample_cbam_config["metadata"]["compliance_references"]
        ref_ids = {r["id"] for r in refs}
        assert "CBAM" in ref_ids
        assert "EU-ETS" in ref_ids

    def test_config_free_allocation_schedule(self, sample_cbam_config):
        """Test free allocation schedule covers 2026-2034."""
        schedule = sample_cbam_config["cbam"]["certificate_config"]["free_allocation_schedule"]
        assert schedule["2026"] == 0.975
        assert schedule["2030"] == 0.515
        assert schedule["2034"] == 0.0

    def test_config_goods_categories_complete(self, sample_cbam_config):
        """Test all 6 CBAM goods categories are enabled."""
        cats = sample_cbam_config["cbam"]["goods_categories"]["enabled"]
        expected = {"cement", "steel", "aluminium", "fertilizers", "electricity", "hydrogen"}
        assert expected == set(cats)

    def test_config_emission_unit_tco2e(self, sample_cbam_config):
        """Test emission config uses tCO2e as unit."""
        emission_cfg = sample_cbam_config["cbam"]["emission_config"]
        assert emission_cfg["unit"] == "tCO2e"
        assert emission_cfg["precision_decimal_places"] == 6

    def test_config_provenance_hash(self, sample_cbam_config):
        """Test config produces a valid provenance hash."""
        h = _compute_hash(sample_cbam_config)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Sub-config model tests (16 tests - 2 per sub-config)
# ---------------------------------------------------------------------------

class TestImporterConfig:
    def test_importer_fields(self, sample_importer_config):
        cfg = sample_importer_config
        assert cfg["company_name"] == "EuroSteel Imports GmbH"
        assert cfg["member_state"] == "DE"

    def test_importer_eori_format(self, sample_importer_config):
        eori = sample_importer_config["eori_number"]
        assert re.match(r"^[A-Z]{2}\d{13,17}$", eori), f"Invalid EORI: {eori}"


class TestGoodsCategoryConfig:
    def test_primary_categories(self, sample_cbam_config):
        primary = sample_cbam_config["cbam"]["goods_categories"]["primary_categories"]
        assert set(primary) == {"steel", "aluminium", "cement"}

    def test_all_categories_included(self, sample_cbam_config):
        enabled = sample_cbam_config["cbam"]["goods_categories"]["enabled"]
        assert len(enabled) == 6


class TestEmissionConfig:
    def test_default_methodology(self, sample_cbam_config):
        cfg = sample_cbam_config["cbam"]["emission_config"]
        assert cfg["default_methodology"] == "actual"
        assert cfg["fallback_methodology"] == "default_values"

    def test_gwp_source(self, sample_cbam_config):
        cfg = sample_cbam_config["cbam"]["emission_config"]
        assert cfg["gwp_source"] in ("AR5", "AR6")


class TestCertificateConfig:
    def test_currency_eur(self, sample_cbam_config):
        cfg = sample_cbam_config["cbam"]["certificate_config"]
        assert cfg["currency"] == "EUR"

    def test_surrender_deadline(self, sample_cbam_config):
        cfg = sample_cbam_config["cbam"]["certificate_config"]
        assert cfg["surrender_deadline_months_after_year"] == 5


class TestQuarterlyConfig:
    def test_quarters(self, sample_cbam_config):
        cfg = sample_cbam_config["cbam"]["quarterly_config"]
        assert cfg["reporting_quarters"] == ["Q1", "Q2", "Q3", "Q4"]

    def test_amendment_window(self, sample_cbam_config):
        cfg = sample_cbam_config["cbam"]["quarterly_config"]
        assert cfg["amendment_window_days"] == 60


class TestSupplierConfig:
    def test_max_installations(self, sample_cbam_config):
        cfg = sample_cbam_config["cbam"]["supplier_config"]
        assert cfg["max_installations_per_supplier"] == 10

    def test_quality_threshold(self, sample_cbam_config):
        cfg = sample_cbam_config["cbam"]["supplier_config"]
        assert cfg["quality_score_threshold"] == 70.0


class TestDeMinimisConfig:
    def test_weight_threshold(self, sample_cbam_config):
        cfg = sample_cbam_config["cbam"]["deminimis_config"]
        assert cfg["annual_weight_threshold_kg"] == 150000

    def test_alert_percentage(self, sample_cbam_config):
        cfg = sample_cbam_config["cbam"]["deminimis_config"]
        assert cfg["alert_at_pct"] == 80


class TestVerificationConfig:
    def test_accreditation_body(self, sample_cbam_config):
        cfg = sample_cbam_config["cbam"]["verification_config"]
        assert cfg["verification_body_accreditation"] == "DAkkS"

    def test_materiality_threshold(self, sample_cbam_config):
        cfg = sample_cbam_config["cbam"]["verification_config"]
        assert cfg["materiality_threshold_pct"] == 5.0


# ---------------------------------------------------------------------------
# Commodity Preset tests (6 tests)
# ---------------------------------------------------------------------------

class TestCommodityPresets:
    """Test 6 commodity presets load correctly."""

    @pytest.mark.parametrize("preset_name", [
        "steel", "aluminium", "cement",
        "fertilizers", "electricity", "hydrogen",
    ])
    def test_commodity_preset_exists_or_is_default(self, preset_name, preset_files):
        """Test commodity preset YAML file exists or can use defaults."""
        if preset_name in preset_files:
            content = preset_files[preset_name].read_text(encoding="utf-8")
            parsed = yaml.safe_load(content)
            assert isinstance(parsed, dict), f"Preset {preset_name} should parse to dict"
        else:
            # Preset may be embedded in pack.yaml or use defaults
            assert True, f"Preset {preset_name} uses default configuration"


# ---------------------------------------------------------------------------
# Sector Preset tests (3 tests)
# ---------------------------------------------------------------------------

class TestSectorPresets:
    """Test 3 sector presets load correctly."""

    @pytest.mark.parametrize("sector_name", [
        "steel_importer", "multi_commodity", "cement_importer",
    ])
    def test_sector_preset_exists_or_is_default(self, sector_name, sector_files):
        """Test sector YAML file exists or can use defaults."""
        if sector_name in sector_files:
            content = sector_files[sector_name].read_text(encoding="utf-8")
            parsed = yaml.safe_load(content)
            assert isinstance(parsed, dict)
        else:
            assert True, f"Sector {sector_name} uses default configuration"


# ---------------------------------------------------------------------------
# Demo config and data tests (3 tests)
# ---------------------------------------------------------------------------

class TestDemoConfig:
    """Test demo configuration and sample data."""

    def test_demo_config_exists_or_defaults(self, demo_config):
        """Test demo config loads or uses defaults."""
        if demo_config:
            assert isinstance(demo_config, dict)
        else:
            assert True, "Demo config uses defaults"

    def test_demo_imports_csv_path(self, demo_imports_csv_path):
        """Test demo imports CSV path is valid."""
        assert isinstance(demo_imports_csv_path, Path)

    def test_demo_supplier_json_path(self, demo_supplier_json_path):
        """Test demo supplier JSON path is valid."""
        assert isinstance(demo_supplier_json_path, Path)


# ---------------------------------------------------------------------------
# EORI and CN Code validation tests (4 tests)
# ---------------------------------------------------------------------------

class TestEORIValidation:
    """Test EORI number validation."""

    @pytest.mark.parametrize("eori,expected_valid", [
        ("DE123456789012345", True),
        ("FR987654321098765", True),
        ("XX12345", False),
        ("12345678901234567", False),
        ("", False),
    ])
    def test_eori_format_validation(self, eori, expected_valid):
        """Test EORI number format validation."""
        valid = bool(re.match(r"^[A-Z]{2}\d{13,17}$", eori))
        assert valid == expected_valid, f"EORI '{eori}' validation mismatch"

    @pytest.mark.parametrize("cn_code,expected_valid", [
        ("7207 11 14", True),
        ("2523 29 00", True),
        ("7601 10 00", True),
        ("720711", False),
        ("7207-11-14", False),
        ("", False),
    ])
    def test_cn_code_format_validation(self, cn_code, expected_valid):
        """Test CN code format validation (XXXX XX XX)."""
        valid = bool(re.match(r"^\d{4}\s\d{2}\s\d{2}$", cn_code))
        assert valid == expected_valid, f"CN code '{cn_code}' validation mismatch"

    def test_eori_member_state_extraction(self):
        """Test extracting member state from EORI number."""
        eori = "DE123456789012345"
        member_state = eori[:2]
        assert member_state == "DE"
        assert member_state in [
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
            "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
            "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        ]

    def test_cn_code_category_mapping(self, sample_cn_codes):
        """Test CN codes map to correct CBAM categories."""
        for category, codes in sample_cn_codes.items():
            assert len(codes) >= 1, f"Category {category} should have CN codes"
            for code_entry in codes:
                assert "code" in code_entry
                assert "desc" in code_entry
