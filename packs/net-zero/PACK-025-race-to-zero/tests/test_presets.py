# -*- coding: utf-8 -*-
"""
Tests for all 8 PACK-025 Race to Zero Configuration Presets.

Covers: corporate_commitment, financial_institution, city_municipality,
region_state, sme_business, high_emitter, service_sector,
manufacturing_sector.

Validates YAML validity, required sections, actor-type mapping, and
configuration completeness.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from pathlib import Path

import pytest
import yaml

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from config.presets import (
    __version__,
    __pack_id__,
    __pack_name__,
    AVAILABLE_PRESETS,
    ACTOR_TYPE_PRESET_MAP,
    DEFAULT_PRESET,
    get_preset_path,
    get_preset_for_actor_type,
)

# Directory where preset YAML files live.
PRESET_DIR = Path(__file__).resolve().parent.parent / "config" / "presets"

# All 8 expected presets.
PRESET_NAMES = [
    "corporate_commitment",
    "financial_institution",
    "city_municipality",
    "region_state",
    "sme_business",
    "high_emitter",
    "service_sector",
    "manufacturing_sector",
]

# Required YAML sections for each preset.
REQUIRED_SECTIONS = [
    "campaign",
    "partner",
    "pledge",
    "starting_line",
    "interim_target",
    "action_plan",
    "hleg",
    "partnership",
    "reporting",
    "progress",
    "readiness",
    "audit_trail",
]


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture(params=PRESET_NAMES, ids=PRESET_NAMES)
def preset_name(request) -> str:
    """Parameterized fixture yielding each preset name."""
    return request.param


@pytest.fixture
def all_preset_data() -> dict:
    """Load all presets and return as dict of {name: data}."""
    result = {}
    for name in PRESET_NAMES:
        yaml_path = PRESET_DIR / f"{name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            result[name] = yaml.safe_load(f)
    return result


# ========================================================================
# Module Metadata
# ========================================================================


class TestPresetModuleMetadata:
    """Tests for presets package metadata."""

    def test_version(self):
        assert __version__ == "1.0.0"

    def test_pack_id(self):
        assert __pack_id__ == "PACK-025"

    def test_pack_name(self):
        assert __pack_name__ == "Race to Zero Pack"


# ========================================================================
# Preset File Existence and YAML Validity
# ========================================================================


class TestPresetFilesExist:
    """Tests that all preset YAML files exist and parse correctly."""

    def test_preset_yaml_exists(self, preset_name):
        """Preset YAML file exists on disk."""
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        assert yaml_path.exists(), f"Missing preset file: {yaml_path}"

    def test_preset_yaml_parses(self, preset_name):
        """Preset YAML file parses without error."""
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_preset_yaml_not_empty(self, preset_name):
        """Preset YAML file has meaningful content."""
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Should have at least 100 non-comment lines
        lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        assert len(lines) >= 50, f"Preset {preset_name} has only {len(lines)} non-comment lines"


# ========================================================================
# Required Sections
# ========================================================================


class TestPresetRequiredSections:
    """Tests that each preset has required configuration sections."""

    def test_has_campaign_section(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "campaign" in data

    def test_has_partner_section(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "partner" in data

    def test_has_pledge_section(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "pledge" in data

    def test_has_starting_line_section(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "starting_line" in data

    def test_has_interim_target_section(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "interim_target" in data

    def test_has_action_plan_section(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "action_plan" in data

    def test_has_hleg_section(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "hleg" in data

    def test_has_partnership_section(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "partnership" in data

    def test_has_reporting_section(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "reporting" in data

    def test_has_progress_section(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "progress" in data

    def test_has_readiness_section(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "readiness" in data

    def test_has_audit_trail_section(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "audit_trail" in data


# ========================================================================
# Campaign Configuration Values
# ========================================================================


class TestPresetCampaignValues:
    """Tests that campaign values are consistent across presets."""

    def test_net_zero_target_year_is_2050(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["campaign"]["net_zero_target_year"] == 2050

    def test_interim_target_year_is_2030(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["campaign"]["interim_target_year"] == 2030

    def test_starting_line_deadline_is_12_months(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["campaign"]["starting_line_deadline_months"] == 12

    def test_annual_reporting_required(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["campaign"]["annual_reporting_required"] is True

    def test_campaign_name(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["campaign"]["campaign_name"] == "Race to Zero"


# ========================================================================
# Pledge Configuration
# ========================================================================


class TestPresetPledgeValues:
    """Tests that pledge core commitments are set."""

    def test_net_zero_by_2050_enabled(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["pledge"]["net_zero_by_2050"] is True

    def test_interim_2030_target_enabled(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["pledge"]["interim_2030_target"] is True

    def test_action_plan_commitment_enabled(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["pledge"]["action_plan_commitment"] is True


# ========================================================================
# Starting Line Configuration
# ========================================================================


class TestPresetStartingLineValues:
    """Tests that Starting Line is enabled and has pillar configuration."""

    def test_starting_line_enabled(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["starting_line"]["enabled"] is True

    def test_starting_line_has_pillars(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        pillars = data["starting_line"]["pillars"]
        assert "pledge" in pillars
        assert "plan" in pillars
        assert "proceed" in pillars
        assert "publish" in pillars


# ========================================================================
# Interim Target Configuration
# ========================================================================


class TestPresetInterimTargetValues:
    """Tests that interim target validation thresholds are correct."""

    def test_minimum_reduction_at_least_42_pct(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["interim_target"]["minimum_reduction_pct"] >= 42.0

    def test_annual_reduction_rate_at_least_4_2_pct(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["interim_target"]["annual_reduction_rate_min_pct"] >= 4.2


# ========================================================================
# HLEG Configuration
# ========================================================================


class TestPresetHLEGValues:
    """Tests that HLEG credibility assessment is configured."""

    def test_hleg_enabled(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["hleg"]["enabled"] is True


# ========================================================================
# Readiness Configuration
# ========================================================================


class TestPresetReadinessValues:
    """Tests that readiness scoring is configured."""

    def test_readiness_enabled(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["readiness"]["enabled"] is True

    def test_readiness_has_8_dimensions(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert len(data["readiness"]["dimensions"]) == 8

    def test_readiness_dimensions_correct(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        expected_dims = [
            "pledge_strength",
            "starting_line_compliance",
            "target_ambition",
            "action_plan_quality",
            "progress_trajectory",
            "sector_alignment",
            "partnership_engagement",
            "hleg_credibility",
        ]
        assert data["readiness"]["dimensions"] == expected_dims


# ========================================================================
# Audit Trail Configuration
# ========================================================================


class TestPresetAuditTrailValues:
    """Tests that audit trail is properly configured."""

    def test_audit_trail_enabled(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["audit_trail"]["enabled"] is True

    def test_sha256_provenance_enabled(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["audit_trail"]["sha256_provenance"] is True


# ========================================================================
# Actor-Type Specific Tests
# ========================================================================


class TestCorporateCommitmentPreset:
    """Tests specific to corporate_commitment preset."""

    def test_actor_type_is_corporate(self):
        yaml_path = PRESET_DIR / "corporate_commitment.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["actor_type"] == "CORPORATE"

    def test_primary_partner_is_sbti(self):
        yaml_path = PRESET_DIR / "corporate_commitment.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["partner"]["primary_partner"] == "SBTi"


class TestFinancialInstitutionPreset:
    """Tests specific to financial_institution preset."""

    def test_actor_type_is_fi(self):
        yaml_path = PRESET_DIR / "financial_institution.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["actor_type"] == "FINANCIAL_INSTITUTION"

    def test_primary_partner_is_gfanz(self):
        yaml_path = PRESET_DIR / "financial_institution.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["partner"]["primary_partner"] == "GFANZ"


class TestCityMunicipalityPreset:
    """Tests specific to city_municipality preset."""

    def test_actor_type_is_city(self):
        yaml_path = PRESET_DIR / "city_municipality.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["actor_type"] == "CITY"

    def test_primary_partner_is_c40(self):
        yaml_path = PRESET_DIR / "city_municipality.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["partner"]["primary_partner"] == "C40"


class TestRegionStatePreset:
    """Tests specific to region_state preset."""

    def test_actor_type_is_region(self):
        yaml_path = PRESET_DIR / "region_state.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["actor_type"] == "REGION"

    def test_primary_partner_is_under2(self):
        yaml_path = PRESET_DIR / "region_state.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["partner"]["primary_partner"] == "Under2 Coalition"


class TestSMEBusinessPreset:
    """Tests specific to sme_business preset."""

    def test_actor_type_is_sme(self):
        yaml_path = PRESET_DIR / "sme_business.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["actor_type"] == "SME"

    def test_primary_partner_is_sme_hub(self):
        yaml_path = PRESET_DIR / "sme_business.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["partner"]["primary_partner"] == "SME Climate Hub"

    def test_simplified_assessment_mode(self):
        yaml_path = PRESET_DIR / "sme_business.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["starting_line"]["assessment_mode"] == "SIMPLIFIED"


class TestHighEmitterPreset:
    """Tests specific to high_emitter preset."""

    def test_sector_is_heavy_industry(self):
        yaml_path = PRESET_DIR / "high_emitter.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["sector"] == "HEAVY_INDUSTRY"

    def test_sda_methodology(self):
        yaml_path = PRESET_DIR / "high_emitter.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["interim_target"]["methodology"] == "SBTi_SDA"

    def test_fossil_fuel_phaseout_commitment(self):
        yaml_path = PRESET_DIR / "high_emitter.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["pledge"]["fossil_fuel_phaseout_commitment"] is True

    def test_ccs_pathway_included(self):
        yaml_path = PRESET_DIR / "high_emitter.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["action_plan"]["include_ccs_pathway"] is True

    def test_enhanced_hleg_mode(self):
        yaml_path = PRESET_DIR / "high_emitter.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["hleg"]["assessment_mode"] == "ENHANCED"


class TestServiceSectorPreset:
    """Tests specific to service_sector preset."""

    def test_sector_is_services(self):
        yaml_path = PRESET_DIR / "service_sector.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["sector"] == "SERVICES"

    def test_aca_methodology(self):
        yaml_path = PRESET_DIR / "service_sector.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["interim_target"]["methodology"] == "SBTi_ACA"


class TestManufacturingSectorPreset:
    """Tests specific to manufacturing_sector preset."""

    def test_sector_is_manufacturing(self):
        yaml_path = PRESET_DIR / "manufacturing_sector.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["sector"] == "MANUFACTURING"

    def test_sda_methodology(self):
        yaml_path = PRESET_DIR / "manufacturing_sector.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["interim_target"]["methodology"] == "SBTi_SDA"


# ========================================================================
# AVAILABLE_PRESETS Constant
# ========================================================================


class TestAvailablePresetsConstant:
    """Tests for AVAILABLE_PRESETS dict."""

    def test_has_8_entries(self):
        assert len(AVAILABLE_PRESETS) == 8

    def test_all_preset_names_present(self):
        for name in PRESET_NAMES:
            assert name in AVAILABLE_PRESETS

    def test_all_paths_exist(self):
        for name, path in AVAILABLE_PRESETS.items():
            assert Path(path).exists(), f"Preset path not found: {path}"


# ========================================================================
# ACTOR_TYPE_PRESET_MAP
# ========================================================================


class TestActorTypePresetMap:
    """Tests for ACTOR_TYPE_PRESET_MAP."""

    def test_has_8_entries(self):
        assert len(ACTOR_TYPE_PRESET_MAP) == 8

    def test_all_actor_types_mapped(self):
        expected_types = [
            "CORPORATE", "FINANCIAL_INSTITUTION", "CITY", "REGION",
            "SME", "HEAVY_INDUSTRY", "SERVICES", "MANUFACTURING",
        ]
        for actor_type in expected_types:
            assert actor_type in ACTOR_TYPE_PRESET_MAP

    def test_all_mapped_presets_exist(self):
        for actor_type, preset in ACTOR_TYPE_PRESET_MAP.items():
            assert preset in AVAILABLE_PRESETS


# ========================================================================
# Utility Functions
# ========================================================================


class TestUtilityFunctions:
    """Tests for preset utility functions."""

    def test_get_preset_path_returns_string(self):
        path = get_preset_path("corporate_commitment")
        assert isinstance(path, str)

    def test_get_preset_path_file_exists(self, preset_name):
        path = get_preset_path(preset_name)
        assert Path(path).exists()

    def test_get_preset_path_unknown_raises(self):
        with pytest.raises(KeyError):
            get_preset_path("nonexistent_preset")

    def test_get_preset_for_actor_type_returns_string(self):
        preset = get_preset_for_actor_type("CORPORATE")
        assert isinstance(preset, str)
        assert preset == "corporate_commitment"

    def test_get_preset_for_actor_type_unknown_raises(self):
        with pytest.raises(KeyError):
            get_preset_for_actor_type("UNKNOWN_TYPE")

    def test_default_preset_is_corporate(self):
        assert DEFAULT_PRESET == "corporate_commitment"
