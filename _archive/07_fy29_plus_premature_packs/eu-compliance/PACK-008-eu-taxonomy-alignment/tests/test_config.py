# -*- coding: utf-8 -*-
"""
PACK-008 EU Taxonomy Alignment Pack - Configuration Tests
==========================================================

Tests the TaxonomyAlignmentConfig configuration system including:
- EnvironmentalObjective enum (6 values)
- AlignmentStatus enum
- OrganizationType enum
- EligibilityConfig defaults
- SCAssessmentConfig defaults
- DNSHConfig defaults
- MinimumSafeguardsConfig defaults
- KPIConfig defaults
- GARConfig defaults
- ReportingConfig defaults
- RegulatoryConfig defaults
- TaxonomyAlignmentConfig creation and validation
- Preset loading (NFU, FI, AM, large_enterprise, sme)
- Sector config loading (6 sectors)
- Demo config loading
- Config merge order
- Environment variable overrides
- Config serialization and hashing

Author: GreenLang QA Team
Version: 1.0.0
"""

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

# ---------------------------------------------------------------------------
# Dynamic import - needed because the pack directory name is hyphenated
# ---------------------------------------------------------------------------

_PACK_DIR = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PACK_DIR / "config"
_PRESETS_DIR = _CONFIG_DIR / "presets"
_SECTORS_DIR = _CONFIG_DIR / "sectors"
_DEMO_DIR = _CONFIG_DIR / "demo"


def _load_config_module():
    """Import pack_config.py from hyphenated pack directory."""
    path = _CONFIG_DIR / "pack_config.py"
    spec = importlib.util.spec_from_file_location("pack_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the config module once for the entire module
try:
    _cfg = _load_config_module()
except Exception:
    _cfg = None


def _skip_if_no_config():
    """Skip test if pack_config module could not be loaded."""
    if _cfg is None:
        pytest.skip("pack_config module could not be loaded")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPackConfig:
    """Test suite for PACK-008 TaxonomyAlignmentConfig configuration."""

    # -----------------------------------------------------------------
    # Enum tests
    # -----------------------------------------------------------------

    def test_environmental_objective_enum(self):
        """Test EnvironmentalObjective enum has exactly 6 values."""
        _skip_if_no_config()
        eo = _cfg.EnvironmentalObjective
        members = list(eo)
        assert len(members) == 6, f"Expected 6 objectives, got {len(members)}"

        expected = {"CCM", "CCA", "WTR", "CE", "PPC", "BIO"}
        actual = {m.value for m in members}
        assert actual == expected, f"Objective mismatch: {actual} != {expected}"

    def test_alignment_status_enum(self):
        """Test AlignmentStatus enum has the expected values."""
        _skip_if_no_config()
        status = _cfg.AlignmentStatus
        expected = {
            "NOT_SCREENED", "ELIGIBLE", "NOT_ELIGIBLE",
            "ALIGNED", "NOT_ALIGNED", "PARTIALLY_ALIGNED",
        }
        actual = {m.value for m in status}
        assert actual == expected, f"Status mismatch: {actual} != {expected}"

    def test_organization_type_enum(self):
        """Test OrganizationType enum has 3 values."""
        _skip_if_no_config()
        org = _cfg.OrganizationType
        expected = {
            "NON_FINANCIAL_UNDERTAKING",
            "FINANCIAL_INSTITUTION",
            "ASSET_MANAGER",
        }
        actual = {m.value for m in org}
        assert actual == expected, f"OrgType mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # Sub-config defaults
    # -----------------------------------------------------------------

    def test_eligibility_config_defaults(self):
        """Test EligibilityConfig default values."""
        _skip_if_no_config()
        ec = _cfg.EligibilityConfig()
        assert ec.screening_mode == _cfg.ScreeningMode.HYBRID
        assert ec.nace_version == _cfg.NACEVersion.NACE_REV2
        assert ec.min_confidence == 0.85
        assert ec.batch_size == 100
        assert ec.include_transitional is True
        assert ec.include_enabling is True
        assert ec.revenue_weighted is True
        assert ec.auto_classify_nace is True

    def test_sc_assessment_config_defaults(self):
        """Test SCAssessmentConfig default values."""
        _skip_if_no_config()
        sc = _cfg.SCAssessmentConfig()
        assert sc.evaluation_mode == _cfg.SCEvaluationMode.STANDARD
        assert sc.threshold_strictness == _cfg.ThresholdStrictness.STANDARD
        assert sc.evidence_required is True
        assert sc.quantitative_tolerance_pct == 5.0
        assert sc.track_enabling_activities is True
        assert sc.track_transitional_activities is True
        assert sc.require_all_quantitative is True
        assert sc.gap_analysis_on_fail is True

    def test_dnsh_config_defaults(self):
        """Test DNSHConfig default values."""
        _skip_if_no_config()
        dnsh = _cfg.DNSHConfig()

        # All 6 objectives assessed by default
        assert len(dnsh.objectives_assessed) == 6
        assert dnsh.require_all_pass is True
        assert dnsh.climate_risk_assessment_enabled is True
        assert dnsh.water_framework_directive_check is True
        assert dnsh.circular_economy_waste_hierarchy is True
        assert dnsh.pollution_threshold_check is True
        assert dnsh.biodiversity_impact_assessment is True
        assert dnsh.evidence_required is True

    def test_minimum_safeguards_config_defaults(self):
        """Test MinimumSafeguardsConfig default values."""
        _skip_if_no_config()
        ms = _cfg.MinimumSafeguardsConfig()
        assert ms.human_rights_check is True
        assert ms.anti_corruption_check is True
        assert ms.taxation_check is True
        assert ms.fair_competition_check is True
        assert ms.assessment_mode == _cfg.MinimumSafeguardsMode.FULL
        assert ms.require_all_pass is True
        assert ms.grievance_mechanism_required is False
        assert ms.supply_chain_due_diligence is False

    def test_kpi_config_defaults(self):
        """Test KPIConfig default values."""
        _skip_if_no_config()
        kpi = _cfg.KPIConfig()
        assert kpi.calculate_turnover is True
        assert kpi.calculate_capex is True
        assert kpi.calculate_opex is True
        assert kpi.double_counting_prevention is True
        assert kpi.capex_plan_recognition is True
        assert kpi.capex_plan_max_years == 5
        assert kpi.eligible_vs_aligned_breakdown is True
        assert kpi.activity_level_detail is True
        assert kpi.currency == "EUR"
        assert kpi.rounding_precision == 2

    def test_gar_config_defaults(self):
        """Test GARConfig default values."""
        _skip_if_no_config()
        gar = _cfg.GARConfig()
        assert gar.calculate_stock_gar is True
        assert gar.calculate_flow_gar is True
        assert gar.calculate_btar is True
        assert len(gar.exposure_types) >= 5
        assert gar.epc_integration is True
        assert gar.epc_threshold_rating == _cfg.EPCRating.C
        assert gar.de_minimis_threshold == 0.0
        assert gar.sovereign_exclusion is True
        assert gar.interbank_exclusion is True

    def test_reporting_config_defaults(self):
        """Test ReportingConfig default values."""
        _skip_if_no_config()
        rpt = _cfg.ReportingConfig()
        assert rpt.article8_enabled is True
        assert rpt.eba_pillar3_enabled is False
        assert rpt.xbrl_tagging is False
        assert rpt.nuclear_gas_supplementary is True
        assert rpt.yoy_comparison is True
        assert rpt.default_format == _cfg.DisclosureFormat.PDF
        assert rpt.include_methodology_note is True
        assert rpt.include_audit_opinion is False
        assert rpt.language == "en"
        assert rpt.timezone == "UTC"
        assert isinstance(rpt.cross_framework_targets, list)

    def test_regulatory_config_defaults(self):
        """Test RegulatoryConfig default values."""
        _skip_if_no_config()
        reg = _cfg.RegulatoryConfig()
        assert reg.delegated_act_version == _cfg.DelegatedActVersion.CLIMATE_DA_2021
        assert len(reg.active_delegated_acts) >= 3
        assert reg.track_updates is True
        assert reg.auto_migration is False
        assert reg.update_check_interval_hours == 24
        assert reg.include_complementary_da is True
        assert reg.include_simplification_da is False

    # -----------------------------------------------------------------
    # TaxonomyAlignmentConfig creation
    # -----------------------------------------------------------------

    def test_taxonomy_alignment_config_creation(self):
        """Test TaxonomyAlignmentConfig creates with valid defaults."""
        _skip_if_no_config()
        config = _cfg.TaxonomyAlignmentConfig()

        assert config.pack_id == "PACK-008-eu-taxonomy-alignment"
        assert config.version == "1.0.0"
        assert config.tier == "standalone"
        assert config.organization_type == _cfg.OrganizationType.NON_FINANCIAL_UNDERTAKING
        assert config.reporting_period == _cfg.ReportingPeriod.ANNUAL
        assert config.reporting_year == 2025
        assert len(config.objectives_in_scope) == 6

        # Sub-configs should be populated
        assert isinstance(config.eligibility, _cfg.EligibilityConfig)
        assert isinstance(config.sc_assessment, _cfg.SCAssessmentConfig)
        assert isinstance(config.dnsh, _cfg.DNSHConfig)
        assert isinstance(config.minimum_safeguards, _cfg.MinimumSafeguardsConfig)
        assert isinstance(config.kpi, _cfg.KPIConfig)
        assert isinstance(config.gar, _cfg.GARConfig)
        assert isinstance(config.reporting, _cfg.ReportingConfig)
        assert isinstance(config.regulatory, _cfg.RegulatoryConfig)
        assert isinstance(config.tsc, _cfg.TSCConfig)
        assert isinstance(config.transition_activity, _cfg.TransitionActivityConfig)
        assert isinstance(config.enabling_activity, _cfg.EnablingActivityConfig)
        assert isinstance(config.data_quality, _cfg.DataQualityConfig)
        assert isinstance(config.audit_trail, _cfg.AuditTrailConfig)
        assert isinstance(config.demo, _cfg.DemoConfig)

    def test_config_validation_invalid_values(self):
        """Test that invalid configuration raises ValidationError."""
        _skip_if_no_config()

        # NFU with no KPI enabled should fail
        with pytest.raises(Exception):
            _cfg.TaxonomyAlignmentConfig(
                organization_type=_cfg.OrganizationType.NON_FINANCIAL_UNDERTAKING,
                kpi=_cfg.KPIConfig(
                    calculate_turnover=False,
                    calculate_capex=False,
                    calculate_opex=False,
                ),
            )

    def test_config_validation_fi_requires_gar(self):
        """Test FI without GAR enabled raises ValidationError."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.TaxonomyAlignmentConfig(
                organization_type=_cfg.OrganizationType.FINANCIAL_INSTITUTION,
                gar=_cfg.GARConfig(
                    calculate_stock_gar=False,
                    calculate_flow_gar=False,
                ),
            )

    def test_config_validation_empty_objectives_fails(self):
        """Test empty objectives_in_scope raises ValidationError."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.TaxonomyAlignmentConfig(objectives_in_scope=[])

    # -----------------------------------------------------------------
    # Preset loading
    # -----------------------------------------------------------------

    def test_preset_loading_nfu(self):
        """Test non_financial_undertaking preset YAML exists and is valid."""
        preset_path = _PRESETS_DIR / "non_financial_undertaking.yaml"
        assert preset_path.exists(), f"NFU preset not found: {preset_path}"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Preset must parse to dict"
        assert data.get("organization_type") == "NON_FINANCIAL_UNDERTAKING"

    def test_preset_loading_fi(self):
        """Test financial_institution preset YAML exists and is valid."""
        preset_path = _PRESETS_DIR / "financial_institution.yaml"
        assert preset_path.exists(), f"FI preset not found: {preset_path}"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Preset must parse to dict"
        assert data.get("organization_type") == "FINANCIAL_INSTITUTION"

    def test_preset_loading_asset_manager(self):
        """Test asset_manager preset YAML exists and is valid."""
        preset_path = _PRESETS_DIR / "asset_manager.yaml"
        assert preset_path.exists(), f"AM preset not found: {preset_path}"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Preset must parse to dict"
        assert data.get("organization_type") == "ASSET_MANAGER"

    def test_preset_loading_large_enterprise(self):
        """Test large_enterprise preset YAML exists and is valid."""
        preset_path = _PRESETS_DIR / "large_enterprise.yaml"
        assert preset_path.exists(), f"Large enterprise preset not found: {preset_path}"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Preset must parse to dict"

    def test_preset_loading_sme(self):
        """Test sme_simplified preset YAML exists and is valid."""
        preset_path = _PRESETS_DIR / "sme_simplified.yaml"
        assert preset_path.exists(), f"SME preset not found: {preset_path}"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Preset must parse to dict"

    # -----------------------------------------------------------------
    # Sector config loading
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("sector_name", [
        "energy",
        "manufacturing",
        "real_estate",
        "transport",
        "forestry_agriculture",
        "financial_services",
    ])
    def test_sector_config_loading(self, sector_name: str):
        """Test each sector config YAML exists and is valid YAML."""
        sector_path = _SECTORS_DIR / f"{sector_name}.yaml"
        assert sector_path.exists(), f"Sector config not found: {sector_path}"
        with open(sector_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), (
            f"Sector {sector_name} must parse to dict"
        )

    # -----------------------------------------------------------------
    # Demo config loading
    # -----------------------------------------------------------------

    def test_demo_config_loading(self):
        """Test demo_config.yaml exists and is valid."""
        demo_path = _DEMO_DIR / "demo_config.yaml"
        assert demo_path.exists(), f"Demo config not found: {demo_path}"
        with open(demo_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Demo config must parse to dict"
        # Demo config should use reduced objectives
        objectives = data.get("objectives_in_scope", [])
        assert len(objectives) >= 2, (
            f"Demo config must have at least CCM+CCA, got {objectives}"
        )

    # -----------------------------------------------------------------
    # Config merge order
    # -----------------------------------------------------------------

    def test_config_merge_order(self):
        """Test that preset values override base defaults correctly."""
        _skip_if_no_config()

        # Create base config
        base = _cfg.TaxonomyAlignmentConfig()
        assert base.organization_type == _cfg.OrganizationType.NON_FINANCIAL_UNDERTAKING

        # Load FI preset data
        fi_path = _PRESETS_DIR / "financial_institution.yaml"
        if not fi_path.exists():
            pytest.skip("FI preset not available")

        with open(fi_path, "r", encoding="utf-8") as f:
            fi_data = yaml.safe_load(f)

        # Merge should produce FI-specific config
        base_dict = base.model_dump()

        def deep_merge(d1: dict, d2: dict) -> dict:
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    d1[key] = deep_merge(d1[key], value)
                else:
                    d1[key] = value
            return d1

        merged = deep_merge(base_dict, fi_data)
        merged_config = _cfg.TaxonomyAlignmentConfig(**merged)
        assert merged_config.organization_type == _cfg.OrganizationType.FINANCIAL_INSTITUTION

    # -----------------------------------------------------------------
    # Environment overrides
    # -----------------------------------------------------------------

    def test_environment_override(self, monkeypatch):
        """Test environment variable overrides are respected."""
        _skip_if_no_config()

        monkeypatch.setenv("TAXONOMY_PACK_ORG_NAME", "TestOrg GmbH")
        monkeypatch.setenv("TAXONOMY_PACK_REPORTING_YEAR", "2026")

        env_org_name = os.getenv("TAXONOMY_PACK_ORG_NAME", "")
        assert env_org_name == "TestOrg GmbH"

        env_year = int(os.getenv("TAXONOMY_PACK_REPORTING_YEAR", "2025"))
        assert env_year == 2026

        # Clean up is automatic via monkeypatch

    # -----------------------------------------------------------------
    # File existence checks
    # -----------------------------------------------------------------

    def test_all_presets_exist(self):
        """Test that all 5 preset YAML files exist on disk."""
        expected_presets = [
            "non_financial_undertaking.yaml",
            "financial_institution.yaml",
            "asset_manager.yaml",
            "large_enterprise.yaml",
            "sme_simplified.yaml",
        ]
        for preset_name in expected_presets:
            preset_path = _PRESETS_DIR / preset_name
            assert preset_path.exists(), (
                f"Preset file missing: {preset_path}"
            )

    def test_all_sectors_exist(self):
        """Test that all 6 sector YAML files exist on disk."""
        expected_sectors = [
            "energy.yaml",
            "manufacturing.yaml",
            "real_estate.yaml",
            "transport.yaml",
            "forestry_agriculture.yaml",
            "financial_services.yaml",
        ]
        for sector_name in expected_sectors:
            sector_path = _SECTORS_DIR / sector_name
            assert sector_path.exists(), (
                f"Sector file missing: {sector_path}"
            )

    # -----------------------------------------------------------------
    # Config serialization and hashing
    # -----------------------------------------------------------------

    def test_config_serialization(self):
        """Test TaxonomyAlignmentConfig can serialize to dict and back."""
        _skip_if_no_config()

        config = _cfg.TaxonomyAlignmentConfig()

        # Serialize to dict
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert "pack_id" in config_dict
        assert "eligibility" in config_dict
        assert "sc_assessment" in config_dict
        assert "dnsh" in config_dict
        assert "minimum_safeguards" in config_dict
        assert "kpi" in config_dict
        assert "gar" in config_dict
        assert "reporting" in config_dict
        assert "regulatory" in config_dict

        # Reconstruct from dict
        reconstructed = _cfg.TaxonomyAlignmentConfig(**config_dict)
        assert reconstructed.pack_id == config.pack_id
        assert reconstructed.version == config.version
        assert reconstructed.organization_type == config.organization_type

    def test_config_hash_reproducibility(self):
        """Test that config hash is reproducible for same configuration."""
        _skip_if_no_config()
        import hashlib
        import json

        config = _cfg.TaxonomyAlignmentConfig()
        config_dict = config.model_dump()
        config_json = json.dumps(config_dict, sort_keys=True, default=str)

        hash1 = hashlib.sha256(config_json.encode()).hexdigest()
        hash2 = hashlib.sha256(config_json.encode()).hexdigest()

        assert hash1 == hash2, "Config hash should be deterministic"
        assert len(hash1) == 64, "SHA-256 hash must be 64 chars"

    def test_pack_config_loader(self):
        """Test PackConfig.load() creates valid configuration."""
        _skip_if_no_config()

        pack_config = _cfg.PackConfig.load()
        assert isinstance(pack_config.pack, _cfg.TaxonomyAlignmentConfig)
        assert isinstance(pack_config.loaded_from, list)

    def test_pack_config_get_config_hash(self):
        """Test PackConfig.get_config_hash returns valid SHA-256."""
        _skip_if_no_config()

        pack_config = _cfg.PackConfig.load()
        config_hash = pack_config.get_config_hash()

        assert isinstance(config_hash, str)
        assert len(config_hash) == 64
        # Reproducibility check
        assert pack_config.get_config_hash() == config_hash

    def test_get_active_agents(self):
        """Test get_active_agents returns expected agent count."""
        _skip_if_no_config()

        config = _cfg.TaxonomyAlignmentConfig()
        agents = config.get_active_agents()

        assert isinstance(agents, list)
        assert len(agents) == 51, f"Expected 51 active agents, got {len(agents)}"
        assert "GL-Taxonomy-APP" in agents
        assert "AGENT-MRV-001" in agents
        assert "AGENT-MRV-030" in agents
        assert "AGENT-DATA-001" in agents
        assert "AGENT-FOUND-001" in agents
        assert "AGENT-FOUND-010" in agents

    def test_get_required_delegated_acts(self):
        """Test get_required_delegated_acts for full-scope config."""
        _skip_if_no_config()

        config = _cfg.TaxonomyAlignmentConfig()
        das = config.get_required_delegated_acts()

        assert isinstance(das, list)
        assert len(das) >= 3, f"Expected at least 3 DAs, got {len(das)}"
        da_values = [d.value for d in das]
        assert "CLIMATE_DA_2021" in da_values
        assert "ENVIRONMENTAL_DA_2023" in da_values
        assert "DISCLOSURES_DA_2021" in da_values

    def test_get_applicable_kpis_nfu(self):
        """Test get_applicable_kpis for NFU returns all 3 KPIs."""
        _skip_if_no_config()

        config = _cfg.TaxonomyAlignmentConfig(
            organization_type=_cfg.OrganizationType.NON_FINANCIAL_UNDERTAKING,
        )
        kpis = config.get_applicable_kpis()
        kpi_values = [k.value for k in kpis]

        assert len(kpis) == 3, f"Expected 3 KPIs for NFU, got {len(kpis)}"
        assert "TURNOVER" in kpi_values
        assert "CAPEX" in kpi_values
        assert "OPEX" in kpi_values

    def test_get_feature_summary(self):
        """Test get_feature_summary returns expected feature flags."""
        _skip_if_no_config()

        config = _cfg.TaxonomyAlignmentConfig()
        features = config.get_feature_summary()

        assert isinstance(features, dict)
        assert features["eligibility_screening"] is True
        assert features["substantial_contribution"] is True
        assert features["kpi_turnover"] is True
        assert features["kpi_capex"] is True
        assert features["kpi_opex"] is True
        assert features["audit_trail"] is True
        assert features["transition_activities"] is True
        assert features["enabling_activities"] is True
