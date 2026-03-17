# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - Configuration Tests
========================================================

Tests for pack_config.py covering all 19 enums, disclosure requirement
constants across 11 standards (82+ DRs), the 12 Pydantic sub-config models,
the ESRSFullCoverageConfig root model, PackConfig wrapper, preset loading,
config hashing, materiality status handling, model validators, and utility
functions.

Target: 60+ tests (across 13 test classes, including parametrized expansion).

Test Classes:
    1.  TestESRSStandardEnum         - ESRS standard enum values
    2.  TestDisclosureStatusEnum     - Disclosure status enum values
    3.  TestMaterialityStatusEnum    - Materiality status enum values
    4.  TestComplianceLevelEnum      - Compliance level enum values
    5.  TestPollutantMediumEnum      - E2 pollutant medium enum values
    6.  TestWaterStressLevelEnum     - E3 water stress level enum values
    7.  TestBiodiversitySensitivityEnum - E4 biodiversity sensitivity enum values
    8.  TestCircularityStrategyEnum  - E5 circularity strategy enum values
    9.  TestWorkforceCategoryEnum    - S1 workforce category enum values
    10. TestGovernanceBodyTypeEnum   - ESRS2 governance body type enum values
    11. TestCorruptionRiskLevelEnum  - G1 corruption risk level enum values
    12. TestAdditionalEnums          - Remaining enums (ReportingBoundary,
                                       TimeHorizon, ValueChainScope,
                                       AssuranceLevel, GenderCategory,
                                       IncidentSeverity, PaymentPracticeType,
                                       SectorPreset)
    13. TestDisclosureRequirements   - DR constant dictionaries across all 11 standards
    14. TestPhaseInDisclosures       - Phase-in schedule per Delegated Regulation 2023/2772
    15. TestSubConfigs               - All 12 Pydantic sub-config model validation
    16. TestESRSFullCoverageConfig   - Root configuration model
    17. TestPackConfig               - PackConfig wrapper, hashing, merge
    18. TestMaterialityConfig        - Material / not_material / pending status handling
    19. TestPresetLoading            - Preset loading for all 6 sectors
    20. TestUtilityFunctions         - get_disclosure_requirements, get_material_standards, etc.
    21. TestConfigValidationWarnings - validate_config cross-field validation

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-017 ESRS Full Coverage
Date:    March 2026
"""

from pathlib import Path

import pytest

from .conftest import _load_config_module, PRESETS_DIR, ALL_ESRS_STANDARDS


# ---------------------------------------------------------------------------
# Module-level config module load (session-scoped via conftest fixture)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    """Load the pack_config module once per module."""
    return _load_config_module()


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestESRSStandardEnum:
    """Tests for ESRSStandard enum values."""

    def test_esrs_standard_count(self, cfg):
        """ESRSStandard enum has exactly 12 members."""
        assert len(cfg.ESRSStandard) == 12

    def test_esrs_standard_includes_all(self, cfg):
        """ESRSStandard contains all 12 standards from ESRS Set 1."""
        names = {m.value for m in cfg.ESRSStandard}
        for std in ALL_ESRS_STANDARDS:
            assert std in names, f"Missing standard: {std}"

    def test_esrs_standard_is_str_enum(self, cfg):
        """ESRSStandard members are string instances."""
        for member in cfg.ESRSStandard:
            assert isinstance(member.value, str)


class TestDisclosureStatusEnum:
    """Tests for DisclosureStatus enum values."""

    def test_disclosure_status_count(self, cfg):
        """DisclosureStatus enum has exactly 5 members."""
        assert len(cfg.DisclosureStatus) == 5

    def test_disclosure_status_includes_all(self, cfg):
        """All expected statuses are present."""
        names = {m.name for m in cfg.DisclosureStatus}
        expected = {"COMPLETE", "PARTIAL", "NOT_STARTED", "NOT_APPLICABLE", "OMITTED"}
        assert names == expected


class TestMaterialityStatusEnum:
    """Tests for MaterialityStatus enum values."""

    def test_materiality_status_count(self, cfg):
        """MaterialityStatus enum has exactly 3 members."""
        assert len(cfg.MaterialityStatus) == 3

    def test_materiality_status_includes_all(self, cfg):
        """MATERIAL, NOT_MATERIAL, PENDING present."""
        names = {m.name for m in cfg.MaterialityStatus}
        assert names == {"MATERIAL", "NOT_MATERIAL", "PENDING"}


class TestComplianceLevelEnum:
    """Tests for ComplianceLevel enum."""

    def test_compliance_level_count(self, cfg):
        """ComplianceLevel has exactly 4 members."""
        assert len(cfg.ComplianceLevel) == 4

    def test_compliance_level_includes_omnibus(self, cfg):
        """OMNIBUS_REDUCED is a valid compliance level."""
        names = {m.name for m in cfg.ComplianceLevel}
        assert "OMNIBUS_REDUCED" in names


class TestPollutantMediumEnum:
    """Tests for PollutantMedium enum (E2-specific)."""

    def test_pollutant_medium_count(self, cfg):
        """PollutantMedium has exactly 3 members."""
        assert len(cfg.PollutantMedium) == 3

    def test_pollutant_medium_includes_all(self, cfg):
        """AIR, WATER, SOIL present."""
        names = {m.name for m in cfg.PollutantMedium}
        assert names == {"AIR", "WATER", "SOIL"}


class TestWaterStressLevelEnum:
    """Tests for WaterStressLevel enum (E3-specific)."""

    def test_water_stress_level_count(self, cfg):
        """WaterStressLevel has exactly 5 members."""
        assert len(cfg.WaterStressLevel) == 5

    def test_water_stress_includes_extreme(self, cfg):
        """EXTREMELY_HIGH is a valid water stress level."""
        names = {m.name for m in cfg.WaterStressLevel}
        assert "EXTREMELY_HIGH" in names


class TestBiodiversitySensitivityEnum:
    """Tests for BiodiversitySensitivity enum (E4-specific)."""

    def test_biodiversity_sensitivity_count(self, cfg):
        """BiodiversitySensitivity has exactly 4 members."""
        assert len(cfg.BiodiversitySensitivity) == 4

    def test_biodiversity_sensitivity_includes_critical(self, cfg):
        """CRITICAL is a valid sensitivity level."""
        names = {m.name for m in cfg.BiodiversitySensitivity}
        assert "CRITICAL" in names


class TestCircularityStrategyEnum:
    """Tests for CircularityStrategy enum (E5-specific)."""

    def test_circularity_strategy_count(self, cfg):
        """CircularityStrategy has exactly 5 members."""
        assert len(cfg.CircularityStrategy) == 5

    def test_circularity_strategy_includes_all(self, cfg):
        """REDUCE, REUSE, RECYCLE, RECOVER, REDESIGN present."""
        names = {m.name for m in cfg.CircularityStrategy}
        assert names == {"REDUCE", "REUSE", "RECYCLE", "RECOVER", "REDESIGN"}


class TestWorkforceCategoryEnum:
    """Tests for WorkforceCategory enum (S1-specific)."""

    def test_workforce_category_count(self, cfg):
        """WorkforceCategory has exactly 5 members."""
        assert len(cfg.WorkforceCategory) == 5

    def test_workforce_category_includes_all(self, cfg):
        """PERMANENT, TEMPORARY, NON_GUARANTEED_HOURS, FULL_TIME, PART_TIME."""
        names = {m.name for m in cfg.WorkforceCategory}
        assert "PERMANENT" in names
        assert "PART_TIME" in names


class TestGovernanceBodyTypeEnum:
    """Tests for GovernanceBodyType enum (ESRS 2-specific)."""

    def test_governance_body_type_count(self, cfg):
        """GovernanceBodyType has exactly 4 members."""
        assert len(cfg.GovernanceBodyType) == 4

    def test_governance_body_type_includes_board(self, cfg):
        """BOARD is a valid governance body type."""
        names = {m.name for m in cfg.GovernanceBodyType}
        assert "BOARD" in names


class TestCorruptionRiskLevelEnum:
    """Tests for CorruptionRiskLevel enum (G1-specific)."""

    def test_corruption_risk_level_count(self, cfg):
        """CorruptionRiskLevel has exactly 4 members."""
        assert len(cfg.CorruptionRiskLevel) == 4

    def test_corruption_risk_includes_very_high(self, cfg):
        """VERY_HIGH is a valid corruption risk level."""
        names = {m.name for m in cfg.CorruptionRiskLevel}
        assert "VERY_HIGH" in names


class TestAdditionalEnums:
    """Tests for remaining enums not covered above."""

    def test_reporting_boundary_count(self, cfg):
        """ReportingBoundary has exactly 3 members."""
        assert len(cfg.ReportingBoundary) == 3

    def test_reporting_boundary_includes_consolidated(self, cfg):
        """CONSOLIDATED is a valid reporting boundary."""
        names = {m.name for m in cfg.ReportingBoundary}
        assert "CONSOLIDATED" in names

    def test_time_horizon_count(self, cfg):
        """TimeHorizon has exactly 3 members."""
        assert len(cfg.TimeHorizon) == 3

    def test_time_horizon_includes_all(self, cfg):
        """SHORT_TERM, MEDIUM_TERM, LONG_TERM present."""
        names = {m.name for m in cfg.TimeHorizon}
        assert names == {"SHORT_TERM", "MEDIUM_TERM", "LONG_TERM"}

    def test_value_chain_scope_count(self, cfg):
        """ValueChainScope has exactly 4 members."""
        assert len(cfg.ValueChainScope) == 4

    def test_assurance_level_count(self, cfg):
        """AssuranceLevel has exactly 3 members."""
        assert len(cfg.AssuranceLevel) == 3

    def test_assurance_level_includes_limited(self, cfg):
        """LIMITED is a valid assurance level (initial CSRD requirement)."""
        names = {m.name for m in cfg.AssuranceLevel}
        assert "LIMITED" in names

    def test_gender_category_count(self, cfg):
        """GenderCategory has exactly 4 members."""
        assert len(cfg.GenderCategory) == 4

    def test_incident_severity_count(self, cfg):
        """IncidentSeverity has exactly 4 members."""
        assert len(cfg.IncidentSeverity) == 4

    def test_payment_practice_type_count(self, cfg):
        """PaymentPracticeType has exactly 3 members."""
        assert len(cfg.PaymentPracticeType) == 3

    def test_sector_preset_count(self, cfg):
        """SectorPreset has exactly 6 members."""
        assert len(cfg.SectorPreset) == 6


# ===========================================================================
# Disclosure Requirements Constants
# ===========================================================================


class TestDisclosureRequirements:
    """Tests for disclosure requirement constant dictionaries."""

    def test_esrs2_dr_count(self, cfg):
        """ESRS 2 has 10 disclosure requirements (GOV-1 to GOV-5, SBM-1 to SBM-3, IRO-1, IRO-2)."""
        assert len(cfg.ESRS_2_DISCLOSURE_REQUIREMENTS) == 10

    def test_e1_dr_count(self, cfg):
        """E1 has 9 disclosure requirements (E1-1 to E1-9)."""
        assert len(cfg.E1_DISCLOSURE_REQUIREMENTS) == 9

    def test_e2_dr_count(self, cfg):
        """E2 has 6 disclosure requirements (E2-1 to E2-6)."""
        assert len(cfg.E2_DISCLOSURE_REQUIREMENTS) == 6

    def test_e3_dr_count(self, cfg):
        """E3 has 5 disclosure requirements (E3-1 to E3-5)."""
        assert len(cfg.E3_DISCLOSURE_REQUIREMENTS) == 5

    def test_e4_dr_count(self, cfg):
        """E4 has 6 disclosure requirements (E4-1 to E4-6)."""
        assert len(cfg.E4_DISCLOSURE_REQUIREMENTS) == 6

    def test_e5_dr_count(self, cfg):
        """E5 has 6 disclosure requirements (E5-1 to E5-6)."""
        assert len(cfg.E5_DISCLOSURE_REQUIREMENTS) == 6

    def test_s1_dr_count(self, cfg):
        """S1 has 17 disclosure requirements (S1-1 to S1-17)."""
        assert len(cfg.S1_DISCLOSURE_REQUIREMENTS) == 17

    def test_s2_dr_count(self, cfg):
        """S2 has 5 disclosure requirements (S2-1 to S2-5)."""
        assert len(cfg.S2_DISCLOSURE_REQUIREMENTS) == 5

    def test_s3_dr_count(self, cfg):
        """S3 has 5 disclosure requirements (S3-1 to S3-5)."""
        assert len(cfg.S3_DISCLOSURE_REQUIREMENTS) == 5

    def test_s4_dr_count(self, cfg):
        """S4 has 5 disclosure requirements (S4-1 to S4-5)."""
        assert len(cfg.S4_DISCLOSURE_REQUIREMENTS) == 5

    def test_g1_dr_count(self, cfg):
        """G1 has 6 disclosure requirements (G1-1 to G1-6)."""
        assert len(cfg.G1_DISCLOSURE_REQUIREMENTS) == 6

    def test_total_disclosure_count_82(self, cfg):
        """Total disclosure count across all standards is 82."""
        total = cfg.get_total_disclosure_count()
        assert total == 82

    def test_all_drs_have_name_field(self, cfg):
        """Every disclosure requirement has a 'name' field."""
        for std_name, dr_dict in cfg._ALL_DISCLOSURE_REQUIREMENTS.items():
            for dr_id, dr_info in dr_dict.items():
                assert "name" in dr_info, f"{std_name}/{dr_id} missing 'name'"

    def test_all_drs_have_mandatory_field(self, cfg):
        """Every disclosure requirement has a 'mandatory' boolean field."""
        for std_name, dr_dict in cfg._ALL_DISCLOSURE_REQUIREMENTS.items():
            for dr_id, dr_info in dr_dict.items():
                assert "mandatory" in dr_info, f"{std_name}/{dr_id} missing 'mandatory'"
                assert isinstance(dr_info["mandatory"], bool)

    def test_all_drs_have_quantitative_field(self, cfg):
        """Every disclosure requirement has a 'quantitative' boolean field."""
        for std_name, dr_dict in cfg._ALL_DISCLOSURE_REQUIREMENTS.items():
            for dr_id, dr_info in dr_dict.items():
                assert "quantitative" in dr_info, f"{std_name}/{dr_id} missing 'quantitative'"
                assert isinstance(dr_info["quantitative"], bool)

    def test_all_drs_have_paragraphs_field(self, cfg):
        """Every disclosure requirement has a 'paragraphs' field."""
        for std_name, dr_dict in cfg._ALL_DISCLOSURE_REQUIREMENTS.items():
            for dr_id, dr_info in dr_dict.items():
                assert "paragraphs" in dr_info, f"{std_name}/{dr_id} missing 'paragraphs'"

    def test_esrs2_all_mandatory(self, cfg):
        """All ESRS 2 disclosure requirements are mandatory."""
        for dr_id, dr_info in cfg.ESRS_2_DISCLOSURE_REQUIREMENTS.items():
            assert dr_info["mandatory"] is True, f"{dr_id} should be mandatory"

    def test_topical_standards_not_mandatory(self, cfg):
        """Topical standard DRs (E1-G1) are subject to materiality, not mandatory."""
        topical_dicts = [
            cfg.E1_DISCLOSURE_REQUIREMENTS,
            cfg.E2_DISCLOSURE_REQUIREMENTS,
            cfg.E3_DISCLOSURE_REQUIREMENTS,
            cfg.E4_DISCLOSURE_REQUIREMENTS,
            cfg.E5_DISCLOSURE_REQUIREMENTS,
            cfg.S1_DISCLOSURE_REQUIREMENTS,
            cfg.S2_DISCLOSURE_REQUIREMENTS,
            cfg.S3_DISCLOSURE_REQUIREMENTS,
            cfg.S4_DISCLOSURE_REQUIREMENTS,
            cfg.G1_DISCLOSURE_REQUIREMENTS,
        ]
        for dr_dict in topical_dicts:
            for dr_id, dr_info in dr_dict.items():
                assert dr_info["mandatory"] is False, (
                    f"{dr_id} is topical and should not be mandatory"
                )

    def test_gov1_present(self, cfg):
        """GOV-1 is present in ESRS 2 DRs."""
        assert "GOV-1" in cfg.ESRS_2_DISCLOSURE_REQUIREMENTS

    def test_sbm1_present(self, cfg):
        """SBM-1 is present in ESRS 2 DRs."""
        assert "SBM-1" in cfg.ESRS_2_DISCLOSURE_REQUIREMENTS

    def test_consolidated_map_has_11_standards(self, cfg):
        """_ALL_DISCLOSURE_REQUIREMENTS has 11 standard entries."""
        assert len(cfg._ALL_DISCLOSURE_REQUIREMENTS) == 11

    def test_consolidated_map_total_drs_is_80(self, cfg):
        """_ALL_DISCLOSURE_REQUIREMENTS contains 80 DRs (82 total minus 2 ESRS 1 basis-for-prep)."""
        total = sum(len(v) for v in cfg._ALL_DISCLOSURE_REQUIREMENTS.values())
        assert total == 80


# ===========================================================================
# Phase-In Disclosures
# ===========================================================================


class TestPhaseInDisclosures:
    """Tests for phase-in disclosure identification."""

    def test_phase_in_2026_not_empty(self, cfg):
        """There are disclosures phasing in for 2026."""
        phase_in = cfg.get_phase_in_disclosures(2026)
        assert len(phase_in) > 0, "Expected phase-in DRs for 2026"

    def test_phase_in_2026_includes_e1_9(self, cfg):
        """E1-9 phases in for 2026."""
        phase_in = cfg.get_phase_in_disclosures(2026)
        assert "E1-9" in phase_in

    def test_phase_in_2026_includes_e2_6(self, cfg):
        """E2-6 phases in for 2026."""
        phase_in = cfg.get_phase_in_disclosures(2026)
        assert "E2-6" in phase_in

    def test_phase_in_2026_includes_e3_5(self, cfg):
        """E3-5 phases in for 2026."""
        phase_in = cfg.get_phase_in_disclosures(2026)
        assert "E3-5" in phase_in

    def test_phase_in_2026_includes_e4_6(self, cfg):
        """E4-6 phases in for 2026."""
        phase_in = cfg.get_phase_in_disclosures(2026)
        assert "E4-6" in phase_in

    def test_phase_in_2026_includes_e5_6(self, cfg):
        """E5-6 phases in for 2026."""
        phase_in = cfg.get_phase_in_disclosures(2026)
        assert "E5-6" in phase_in

    def test_phase_in_2025_includes_s1_16(self, cfg):
        """S1-16 Remuneration Metrics phases in for 2025."""
        phase_in = cfg.get_phase_in_disclosures(2025)
        assert "S1-16" in phase_in

    def test_phase_in_2024_is_empty(self, cfg):
        """No disclosure requirements phase in for 2024 (Year 1 = baseline)."""
        phase_in = cfg.get_phase_in_disclosures(2024)
        assert len(phase_in) == 0

    def test_phase_in_far_future_is_empty(self, cfg):
        """No disclosures phase in for 2030 (beyond current ESRS schedule)."""
        phase_in = cfg.get_phase_in_disclosures(2030)
        assert len(phase_in) == 0


# ===========================================================================
# Sub-Config Model Tests (12 sub-configs)
# ===========================================================================


class TestSubConfigs:
    """Tests for all 12 Pydantic sub-configuration models."""

    # ---- ESRS2Config ----

    def test_esrs2_config_defaults(self, cfg):
        """ESRS2Config creates with valid defaults."""
        config = cfg.ESRS2Config()
        assert config.enabled is True
        assert config.governance_body_count == 1
        assert config.due_diligence_processes is True
        assert config.risk_management_integration is True

    def test_esrs2_config_governance_body_types_default(self, cfg):
        """ESRS2Config default governance_body_types has BOARD, COMMITTEE, EXECUTIVE."""
        config = cfg.ESRS2Config()
        type_names = {t.value for t in config.governance_body_types}
        assert "BOARD" in type_names
        assert "COMMITTEE" in type_names
        assert "EXECUTIVE" in type_names

    def test_esrs2_config_time_horizon_validator(self, cfg):
        """ESRS2Config rejects non-ascending time horizons."""
        with pytest.raises(Exception):
            cfg.ESRS2Config(
                strategy_time_horizons={
                    "short_term_years": 5,
                    "medium_term_years": 3,
                    "long_term_years": 10,
                }
            )

    # ---- E2PollutionConfig ----

    def test_e2_pollution_config_defaults(self, cfg):
        """E2PollutionConfig creates with valid defaults."""
        config = cfg.E2PollutionConfig()
        assert config.enabled is True
        assert config.emission_to_air_tracked is True
        assert config.pollutant_reporting_unit == "tonnes"

    def test_e2_pollution_invalid_reporting_unit(self, cfg):
        """E2PollutionConfig rejects invalid pollutant_reporting_unit."""
        with pytest.raises(Exception):
            cfg.E2PollutionConfig(pollutant_reporting_unit="gallons")

    # ---- E3WaterConfig ----

    def test_e3_water_config_defaults(self, cfg):
        """E3WaterConfig creates with valid defaults."""
        config = cfg.E3WaterConfig()
        assert config.enabled is True
        assert config.water_consumption_tracked is True
        assert config.water_stress_tool == "WRI_AQUEDUCT"

    def test_e3_water_invalid_stress_tool(self, cfg):
        """E3WaterConfig rejects invalid water_stress_tool."""
        with pytest.raises(Exception):
            cfg.E3WaterConfig(water_stress_tool="INVALID_TOOL")

    def test_e3_water_recycling_rate_bounds(self, cfg):
        """E3WaterConfig water_recycling_rate_target must be 0.0-100.0."""
        config = cfg.E3WaterConfig(water_recycling_rate_target=50.0)
        assert config.water_recycling_rate_target == 50.0
        with pytest.raises(Exception):
            cfg.E3WaterConfig(water_recycling_rate_target=150.0)

    # ---- E4BiodiversityConfig ----

    def test_e4_biodiversity_config_defaults(self, cfg):
        """E4BiodiversityConfig creates with valid defaults."""
        config = cfg.E4BiodiversityConfig()
        assert config.enabled is True
        assert config.sites_near_sensitive_areas is True
        assert config.ecosystem_services_framework == "TNFD_LEAP"

    def test_e4_biodiversity_invalid_framework(self, cfg):
        """E4BiodiversityConfig rejects invalid ecosystem_services_framework."""
        with pytest.raises(Exception):
            cfg.E4BiodiversityConfig(ecosystem_services_framework="INVALID")

    def test_e4_biodiversity_buffer_km_bounds(self, cfg):
        """E4BiodiversityConfig sensitive_area_buffer_km must be 0.0-50.0."""
        config = cfg.E4BiodiversityConfig(sensitive_area_buffer_km=25.0)
        assert config.sensitive_area_buffer_km == 25.0
        with pytest.raises(Exception):
            cfg.E4BiodiversityConfig(sensitive_area_buffer_km=100.0)

    # ---- E5CircularConfig ----

    def test_e5_circular_config_defaults(self, cfg):
        """E5CircularConfig creates with valid defaults."""
        config = cfg.E5CircularConfig()
        assert config.enabled is True
        assert config.resource_inflows_tracked is True
        assert config.waste_generation_tracked is True

    def test_e5_circular_invalid_resource_unit(self, cfg):
        """E5CircularConfig rejects invalid resource_reporting_unit."""
        with pytest.raises(Exception):
            cfg.E5CircularConfig(resource_reporting_unit="liters")

    # ---- S1WorkforceConfig ----

    def test_s1_workforce_config_defaults(self, cfg):
        """S1WorkforceConfig creates with valid defaults."""
        config = cfg.S1WorkforceConfig()
        assert config.enabled is True
        assert config.gender_pay_gap_tracked is True
        assert config.h_and_s_incidents_tracked is True
        assert config.adequate_wage_benchmark == "applicable_benchmarks"

    def test_s1_workforce_invalid_wage_benchmark(self, cfg):
        """S1WorkforceConfig rejects invalid adequate_wage_benchmark."""
        with pytest.raises(Exception):
            cfg.S1WorkforceConfig(adequate_wage_benchmark="unknown_benchmark")

    # ---- S2ValueChainConfig ----

    def test_s2_value_chain_config_defaults(self, cfg):
        """S2ValueChainConfig creates with valid defaults."""
        config = cfg.S2ValueChainConfig()
        assert config.enabled is True
        assert config.value_chain_workers_mapped is True
        assert config.child_labor_risk_assessed is True

    # ---- S3CommunitiesConfig ----

    def test_s3_communities_config_defaults(self, cfg):
        """S3CommunitiesConfig creates with valid defaults."""
        config = cfg.S3CommunitiesConfig()
        assert config.enabled is True
        assert config.affected_communities_identified is True
        assert config.fpic_standard == "ILO_169"

    def test_s3_communities_invalid_fpic_standard(self, cfg):
        """S3CommunitiesConfig rejects invalid fpic_standard."""
        with pytest.raises(Exception):
            cfg.S3CommunitiesConfig(fpic_standard="INVALID_STANDARD")

    # ---- S4ConsumersConfig ----

    def test_s4_consumers_config_defaults(self, cfg):
        """S4ConsumersConfig creates with valid defaults."""
        config = cfg.S4ConsumersConfig()
        assert config.enabled is True
        assert config.product_safety_tracked is True
        assert config.data_privacy_framework == "GDPR"

    def test_s4_consumers_invalid_privacy_framework(self, cfg):
        """S4ConsumersConfig rejects invalid data_privacy_framework."""
        with pytest.raises(Exception):
            cfg.S4ConsumersConfig(data_privacy_framework="HIPAA")

    # ---- G1GovernanceConfig ----

    def test_g1_governance_config_defaults(self, cfg):
        """G1GovernanceConfig creates with valid defaults."""
        config = cfg.G1GovernanceConfig()
        assert config.enabled is True
        assert config.code_of_conduct_exists is True
        assert config.anti_corruption_training is True
        assert config.corruption_risk_level.value == "LOW"

    def test_g1_governance_invalid_political_policy(self, cfg):
        """G1GovernanceConfig rejects invalid political_contributions_policy."""
        with pytest.raises(Exception):
            cfg.G1GovernanceConfig(political_contributions_policy="mandatory")

    def test_g1_governance_training_coverage_bounds(self, cfg):
        """G1GovernanceConfig anti_corruption_training_coverage_pct must be 0-100."""
        config = cfg.G1GovernanceConfig(anti_corruption_training_coverage_pct=75.0)
        assert config.anti_corruption_training_coverage_pct == 75.0
        with pytest.raises(Exception):
            cfg.G1GovernanceConfig(anti_corruption_training_coverage_pct=110.0)

    # ---- OrchestratorConfig ----

    def test_orchestrator_config_defaults(self, cfg):
        """OrchestratorConfig creates with valid defaults."""
        config = cfg.OrchestratorConfig()
        assert config.parallel_execution is True
        assert config.max_concurrent_engines == 4
        assert config.timeout_seconds == 600
        assert config.retry_count == 3

    def test_orchestrator_config_execution_order_default(self, cfg):
        """OrchestratorConfig default execution_order starts with ESRS_2."""
        config = cfg.OrchestratorConfig()
        assert config.execution_order[0] == "ESRS_2"
        assert len(config.execution_order) == 11

    def test_orchestrator_config_invalid_standard_in_order(self, cfg):
        """OrchestratorConfig rejects invalid standard in execution_order."""
        with pytest.raises(Exception):
            cfg.OrchestratorConfig(
                execution_order=["ESRS_2", "INVALID_STD", "E1"]
            )

    # ---- ReportingConfig ----

    def test_reporting_config_defaults(self, cfg):
        """ReportingConfig creates with valid defaults."""
        config = cfg.ReportingConfig()
        assert config.enabled is True
        assert config.xbrl_tagging_enabled is True
        assert config.sha256_provenance is True
        assert config.assurance_level.value == "LIMITED"
        assert config.currency == "EUR"

    def test_reporting_config_invalid_date_format(self, cfg):
        """ReportingConfig rejects invalid date format."""
        with pytest.raises(Exception):
            cfg.ReportingConfig(reporting_period_start="01-01-2025")

    def test_reporting_config_end_before_start_rejected(self, cfg):
        """ReportingConfig rejects reporting_period_end before reporting_period_start."""
        with pytest.raises(Exception):
            cfg.ReportingConfig(
                reporting_period_start="2025-12-31",
                reporting_period_end="2025-01-01",
            )

    def test_reporting_config_output_formats_default(self, cfg):
        """ReportingConfig default output_formats includes PDF, XBRL, HTML, JSON."""
        config = cfg.ReportingConfig()
        assert set(config.output_formats) == {"PDF", "XBRL", "HTML", "JSON"}


# ===========================================================================
# ESRSFullCoverageConfig Root Model Tests
# ===========================================================================


class TestESRSFullCoverageConfig:
    """Tests for ESRSFullCoverageConfig root configuration model."""

    def test_default_creation(self, cfg):
        """ESRSFullCoverageConfig can be created with all defaults."""
        config = cfg.ESRSFullCoverageConfig()
        assert config is not None

    def test_config_model_construction(self, cfg):
        """Config model with custom values constructs correctly."""
        config = cfg.ESRSFullCoverageConfig(
            company_name="TestCorp GmbH",
            reporting_year=2025,
            sector="MANUFACTURING",
        )
        assert config.company_name == "TestCorp GmbH"
        assert config.sector == "MANUFACTURING"

    def test_default_sector_general(self, cfg):
        """Default sector is GENERAL."""
        config = cfg.ESRSFullCoverageConfig()
        assert config.sector == "GENERAL"

    def test_default_reporting_year(self, cfg):
        """Default reporting year is 2025."""
        config = cfg.ESRSFullCoverageConfig()
        assert config.reporting_year == 2025

    def test_default_compliance_level_full(self, cfg):
        """Default compliance level is FULL."""
        config = cfg.ESRSFullCoverageConfig()
        assert config.compliance_level.value == "FULL"

    def test_default_reporting_boundary_consolidated(self, cfg):
        """Default reporting boundary is CONSOLIDATED."""
        config = cfg.ESRSFullCoverageConfig()
        assert config.reporting_boundary.value == "CONSOLIDATED"

    def test_sub_configs_initialized(self, cfg):
        """All major sub-configs are initialized."""
        config = cfg.ESRSFullCoverageConfig()
        assert config.esrs2 is not None
        assert config.e2_pollution is not None
        assert config.e3_water is not None
        assert config.e4_biodiversity is not None
        assert config.e5_circular is not None
        assert config.s1_workforce is not None
        assert config.s2_value_chain is not None
        assert config.s3_communities is not None
        assert config.s4_consumers is not None
        assert config.g1_governance is not None
        assert config.orchestrator is not None
        assert config.reporting is not None

    def test_esrs2_always_material_validator(self, cfg):
        """Model validator forces ESRS 2 to MATERIAL even if set to NOT_MATERIAL."""
        config = cfg.ESRSFullCoverageConfig(
            materiality_results={
                "ESRS_2": cfg.MaterialityStatus.NOT_MATERIAL,
                "E1": cfg.MaterialityStatus.MATERIAL,
                "S1": cfg.MaterialityStatus.MATERIAL,
            }
        )
        # Validator should have overridden ESRS_2 to MATERIAL
        esrs2_status = config.materiality_results.get("ESRS_2")
        assert esrs2_status == cfg.MaterialityStatus.MATERIAL or esrs2_status == "MATERIAL"

    def test_default_materiality_results_esrs2_material(self, cfg):
        """Default materiality_results marks ESRS_2 as MATERIAL."""
        config = cfg.ESRSFullCoverageConfig()
        esrs2_status = config.materiality_results.get("ESRS_2")
        assert esrs2_status == cfg.MaterialityStatus.MATERIAL or esrs2_status == "MATERIAL"

    def test_default_materiality_results_e1_material(self, cfg):
        """Default materiality_results marks E1 as MATERIAL."""
        config = cfg.ESRSFullCoverageConfig()
        e1_status = config.materiality_results.get("E1")
        assert e1_status == cfg.MaterialityStatus.MATERIAL or e1_status == "MATERIAL"

    def test_get_material_standards(self, cfg):
        """get_material_standards returns list of ESRSStandard for material topics."""
        config = cfg.ESRSFullCoverageConfig()
        material = cfg.get_material_standards(config)
        assert isinstance(material, list)

    def test_is_listed_default_true(self, cfg):
        """Default is_listed is True (EU-regulated market)."""
        config = cfg.ESRSFullCoverageConfig()
        assert config.is_listed is True


# ===========================================================================
# PackConfig Wrapper Tests
# ===========================================================================


class TestPackConfig:
    """Tests for PackConfig wrapper."""

    def test_default_pack_config(self, cfg):
        """PackConfig creates with defaults."""
        config = cfg.PackConfig()
        assert config.pack_id == "PACK-017-esrs-full-coverage"
        assert config.config_version == "1.0.0"

    def test_from_preset_invalid_raises(self, cfg):
        """from_preset with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            cfg.PackConfig.from_preset("nonexistent_preset")

    def test_config_hash_is_sha256(self, cfg):
        """get_config_hash returns a 64-char hex string (SHA-256)."""
        config = cfg.PackConfig()
        hash_value = config.get_config_hash()
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_config_hash_deterministic(self, cfg):
        """Same configuration produces same hash."""
        config1 = cfg.PackConfig()
        config2 = cfg.PackConfig()
        assert config1.get_config_hash() == config2.get_config_hash()

    def test_config_hash_changes_on_modification(self, cfg):
        """Different configuration produces different hash."""
        config1 = cfg.PackConfig()
        config2 = cfg.PackConfig()
        config2.pack.company_name = "DifferentCorp"
        assert config1.get_config_hash() != config2.get_config_hash()

    def test_merge_returns_new_instance(self, cfg):
        """merge() returns a new PackConfig, not the same instance."""
        config = cfg.PackConfig()
        merged = config.merge({"company_name": "MergedCorp"})
        assert merged is not config
        assert merged.pack.company_name == "MergedCorp"
        assert config.pack.company_name == ""  # original unchanged

    def test_merge_preserves_pack_id(self, cfg):
        """merge() preserves the pack_id from the original config."""
        config = cfg.PackConfig()
        merged = config.merge({"sector": "ENERGY"})
        assert merged.pack_id == "PACK-017-esrs-full-coverage"

    def test_get_material_standard_count(self, cfg):
        """get_material_standard_count returns expected count for defaults."""
        config = cfg.PackConfig()
        count = config.get_material_standard_count()
        # Default has ESRS_2, E1, S1 as MATERIAL
        assert count >= 2  # At minimum ESRS_2 and E1

    def test_get_active_engines(self, cfg):
        """get_active_engines returns list of enabled and material engines."""
        config = cfg.PackConfig()
        active = config.get_active_engines()
        assert isinstance(active, list)
        # ESRS_2 is always material and enabled by default
        assert "ESRS_2" in active

    def test_validate_method(self, cfg):
        """PackConfig.validate() returns list of warnings."""
        config = cfg.PackConfig()
        warnings = config.validate()
        assert isinstance(warnings, list)

    def test_deep_merge_nested(self, cfg):
        """_deep_merge correctly merges nested dictionaries."""
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10, "e": 5}, "f": 6}
        result = cfg.PackConfig._deep_merge(base, override)
        assert result["a"]["b"] == 10
        assert result["a"]["c"] == 2
        assert result["a"]["e"] == 5
        assert result["d"] == 3
        assert result["f"] == 6


# ===========================================================================
# Materiality Configuration Tests
# ===========================================================================


class TestMaterialityConfig:
    """Tests for materiality status handling in configuration."""

    def test_all_material_config(self, cfg):
        """Config with all standards set to MATERIAL works correctly."""
        materiality = {
            std: cfg.MaterialityStatus.MATERIAL
            for std in [
                "ESRS_2", "E1", "E2", "E3", "E4", "E5",
                "S1", "S2", "S3", "S4", "G1",
            ]
        }
        config = cfg.ESRSFullCoverageConfig(materiality_results=materiality)
        material_stds = cfg.get_material_standards(config)
        assert len(material_stds) == 11

    def test_all_not_material_except_esrs2(self, cfg):
        """Config with all topical standards NOT_MATERIAL still has ESRS_2 MATERIAL."""
        materiality = {
            "ESRS_2": cfg.MaterialityStatus.NOT_MATERIAL,
            "E1": cfg.MaterialityStatus.NOT_MATERIAL,
            "E2": cfg.MaterialityStatus.NOT_MATERIAL,
            "E3": cfg.MaterialityStatus.NOT_MATERIAL,
            "E4": cfg.MaterialityStatus.NOT_MATERIAL,
            "E5": cfg.MaterialityStatus.NOT_MATERIAL,
            "S1": cfg.MaterialityStatus.NOT_MATERIAL,
            "S2": cfg.MaterialityStatus.NOT_MATERIAL,
            "S3": cfg.MaterialityStatus.NOT_MATERIAL,
            "S4": cfg.MaterialityStatus.NOT_MATERIAL,
            "G1": cfg.MaterialityStatus.NOT_MATERIAL,
        }
        config = cfg.ESRSFullCoverageConfig(materiality_results=materiality)
        # Validator forces ESRS_2 to MATERIAL
        esrs2_status = config.materiality_results.get("ESRS_2")
        assert esrs2_status == cfg.MaterialityStatus.MATERIAL or esrs2_status == "MATERIAL"

    def test_pending_standards_detected_by_validate(self, cfg):
        """validate_config detects PENDING materiality statuses."""
        config = cfg.ESRSFullCoverageConfig()  # defaults have PENDING
        warnings = cfg.validate_config(config)
        pending_warnings = [w for w in warnings if "PENDING" in w]
        assert len(pending_warnings) > 0

    def test_mixed_materiality_results(self, cfg):
        """Config with mixed MATERIAL/NOT_MATERIAL/PENDING correctly filters material."""
        materiality = {
            "ESRS_2": cfg.MaterialityStatus.MATERIAL,
            "E1": cfg.MaterialityStatus.MATERIAL,
            "E2": cfg.MaterialityStatus.MATERIAL,
            "E3": cfg.MaterialityStatus.NOT_MATERIAL,
            "E4": cfg.MaterialityStatus.PENDING,
            "E5": cfg.MaterialityStatus.NOT_MATERIAL,
            "S1": cfg.MaterialityStatus.MATERIAL,
            "S2": cfg.MaterialityStatus.NOT_MATERIAL,
            "S3": cfg.MaterialityStatus.NOT_MATERIAL,
            "S4": cfg.MaterialityStatus.NOT_MATERIAL,
            "G1": cfg.MaterialityStatus.MATERIAL,
        }
        config = cfg.ESRSFullCoverageConfig(materiality_results=materiality)
        material = cfg.get_material_standards(config)
        material_values = {m.value for m in material}
        assert "ESRS_2" in material_values
        assert "E1" in material_values
        assert "E2" in material_values
        assert "S1" in material_values
        assert "G1" in material_values
        assert "E3" not in material_values
        assert "E5" not in material_values

    def test_material_standard_count_with_pack_config(self, cfg):
        """PackConfig.get_material_standard_count matches get_material_standards."""
        materiality = {
            "ESRS_2": cfg.MaterialityStatus.MATERIAL,
            "E1": cfg.MaterialityStatus.MATERIAL,
            "E2": cfg.MaterialityStatus.MATERIAL,
            "E3": cfg.MaterialityStatus.NOT_MATERIAL,
            "E4": cfg.MaterialityStatus.NOT_MATERIAL,
            "E5": cfg.MaterialityStatus.NOT_MATERIAL,
            "S1": cfg.MaterialityStatus.MATERIAL,
            "S2": cfg.MaterialityStatus.NOT_MATERIAL,
            "S3": cfg.MaterialityStatus.NOT_MATERIAL,
            "S4": cfg.MaterialityStatus.NOT_MATERIAL,
            "G1": cfg.MaterialityStatus.NOT_MATERIAL,
        }
        pack_cfg = cfg.PackConfig(
            pack=cfg.ESRSFullCoverageConfig(materiality_results=materiality)
        )
        count = pack_cfg.get_material_standard_count()
        material_list = cfg.get_material_standards(pack_cfg.pack)
        assert count == len(material_list)


# ===========================================================================
# Preset Loading Tests
# ===========================================================================


class TestPresetLoading:
    """Tests for preset loading functionality."""

    @pytest.mark.parametrize("preset_name", [
        "manufacturing",
        "financial_services",
        "energy",
        "retail",
        "technology",
        "multi_sector",
    ])
    def test_preset_loads(self, cfg, preset_name):
        """Each of the 6 presets loads successfully."""
        preset_path = PRESETS_DIR / f"{preset_name}.yaml"
        if not preset_path.exists():
            pytest.skip(f"Preset file not found: {preset_path}")
        config = cfg.PackConfig.from_preset(preset_name)
        assert config.preset_name == preset_name

    def test_available_presets_dict(self, cfg):
        """AVAILABLE_PRESETS has exactly 6 entries."""
        assert len(cfg.AVAILABLE_PRESETS) == 6


# ===========================================================================
# Config Validation Warnings Tests
# ===========================================================================


class TestConfigValidationWarnings:
    """Tests for validate_config cross-field validation logic."""

    def test_warnings_for_disabled_h_and_s_when_s1_material(self, cfg):
        """validate_config warns when S1 is material but H&S tracking is disabled."""
        config = cfg.ESRSFullCoverageConfig(
            materiality_results={
                "ESRS_2": cfg.MaterialityStatus.MATERIAL,
                "S1": cfg.MaterialityStatus.MATERIAL,
            },
        )
        config.s1_workforce.h_and_s_incidents_tracked = False
        warnings = cfg.validate_config(config)
        hs_warnings = [w for w in warnings if "S1-14" in w or "H&S" in w]
        assert len(hs_warnings) > 0

    def test_warnings_for_no_pollutant_tracking_when_e2_material(self, cfg):
        """validate_config warns when E2 is material but no pollutant medium is tracked."""
        config = cfg.ESRSFullCoverageConfig(
            materiality_results={
                "ESRS_2": cfg.MaterialityStatus.MATERIAL,
                "E2": cfg.MaterialityStatus.MATERIAL,
            },
        )
        config.e2_pollution.emission_to_air_tracked = False
        config.e2_pollution.emission_to_water_tracked = False
        config.e2_pollution.emission_to_soil_tracked = False
        warnings = cfg.validate_config(config)
        e2_warnings = [w for w in warnings if "E2" in w and "medium" in w.lower()]
        assert len(e2_warnings) > 0

    def test_warnings_for_omnibus_large_company(self, cfg):
        """validate_config warns when OMNIBUS_REDUCED with >=1000 employees."""
        config = cfg.ESRSFullCoverageConfig(
            compliance_level=cfg.ComplianceLevel.OMNIBUS_REDUCED,
            employee_count=5000,
        )
        warnings = cfg.validate_config(config)
        omnibus_warnings = [w for w in warnings if "OMNIBUS" in w]
        assert len(omnibus_warnings) > 0

    def test_warnings_for_listed_no_assurance(self, cfg):
        """validate_config warns when listed undertaking has no assurance."""
        config = cfg.ESRSFullCoverageConfig(is_listed=True)
        config.reporting.assurance_level = cfg.AssuranceLevel.NONE
        warnings = cfg.validate_config(config)
        assurance_warnings = [w for w in warnings if "assurance" in w.lower()]
        assert len(assurance_warnings) > 0

    def test_no_warnings_for_well_configured(self, cfg):
        """validate_config returns fewer warnings for a well-configured setup."""
        materiality = {
            "ESRS_2": cfg.MaterialityStatus.MATERIAL,
            "E1": cfg.MaterialityStatus.MATERIAL,
            "E2": cfg.MaterialityStatus.NOT_MATERIAL,
            "E3": cfg.MaterialityStatus.NOT_MATERIAL,
            "E4": cfg.MaterialityStatus.NOT_MATERIAL,
            "E5": cfg.MaterialityStatus.NOT_MATERIAL,
            "S1": cfg.MaterialityStatus.MATERIAL,
            "S2": cfg.MaterialityStatus.NOT_MATERIAL,
            "S3": cfg.MaterialityStatus.NOT_MATERIAL,
            "S4": cfg.MaterialityStatus.NOT_MATERIAL,
            "G1": cfg.MaterialityStatus.NOT_MATERIAL,
        }
        config = cfg.ESRSFullCoverageConfig(
            materiality_results=materiality,
            employee_count=500,
            is_listed=True,
        )
        warnings = cfg.validate_config(config)
        # With no PENDING and well-configured, should have minimal warnings
        assert isinstance(warnings, list)
        pending_warnings = [w for w in warnings if "PENDING" in w]
        assert len(pending_warnings) == 0


# ===========================================================================
# Utility Function Tests
# ===========================================================================


class TestUtilityFunctions:
    """Tests for pack_config utility functions."""

    def test_get_total_disclosure_count(self, cfg):
        """get_total_disclosure_count returns 82."""
        assert cfg.get_total_disclosure_count() == 82

    def test_get_mandatory_disclosures_not_empty(self, cfg):
        """get_mandatory_disclosures returns non-empty list."""
        mandatory = cfg.get_mandatory_disclosures()
        assert len(mandatory) > 0
        assert "GOV-1" in mandatory

    def test_get_mandatory_disclosures_only_esrs2(self, cfg):
        """All mandatory disclosures come from ESRS 2 (10 DRs)."""
        mandatory = cfg.get_mandatory_disclosures()
        assert len(mandatory) == 10
        # All should be GOV, SBM, or IRO prefixed
        for dr_id in mandatory:
            assert any(dr_id.startswith(p) for p in ("GOV-", "SBM-", "IRO-")), (
                f"Unexpected mandatory DR: {dr_id}"
            )

    def test_get_all_disclosure_ids(self, cfg):
        """get_all_disclosure_ids returns sorted list of DR ids."""
        all_ids = cfg.get_all_disclosure_ids()
        assert len(all_ids) >= 80  # 80 topical DRs (excluding ESRS 1 basis-for-prep)
        assert "E1-6" in all_ids
        assert "S1-14" in all_ids

    def test_get_all_disclosure_ids_sorted(self, cfg):
        """get_all_disclosure_ids returns a sorted list."""
        all_ids = cfg.get_all_disclosure_ids()
        assert all_ids == sorted(all_ids)

    def test_get_disclosure_info_e1_6(self, cfg):
        """get_disclosure_info returns info for E1-6."""
        info = cfg.get_disclosure_info("E1-6")
        assert "GHG" in info["name"] or "Gross" in info["name"]
        assert info["quantitative"] is True

    def test_get_disclosure_info_gov1(self, cfg):
        """get_disclosure_info returns info for GOV-1."""
        info = cfg.get_disclosure_info("GOV-1")
        assert "Administrative" in info["name"] or "Role" in info["name"]
        assert info["mandatory"] is True

    def test_get_disclosure_info_unknown(self, cfg):
        """get_disclosure_info returns default for unknown DR."""
        info = cfg.get_disclosure_info("UNKNOWN-99")
        assert info["name"] == "UNKNOWN-99"
        assert info["mandatory"] is False

    def test_get_quantitative_disclosures(self, cfg):
        """get_quantitative_disclosures returns list of quantitative DRs."""
        quantitative = cfg.get_quantitative_disclosures()
        assert len(quantitative) > 0
        assert "E1-6" in quantitative

    def test_get_quantitative_disclosures_excludes_qualitative(self, cfg):
        """get_quantitative_disclosures does not include qualitative-only DRs."""
        quantitative = cfg.get_quantitative_disclosures()
        # GOV-1 is qualitative (narrative), should not be in list
        assert "GOV-1" not in quantitative

    def test_get_standard_dr_count_esrs2(self, cfg):
        """get_standard_dr_count returns 10 for ESRS_2."""
        count = cfg.get_standard_dr_count(cfg.ESRSStandard.ESRS_2)
        assert count == 10

    def test_get_standard_dr_count_s1(self, cfg):
        """get_standard_dr_count returns 17 for S1."""
        count = cfg.get_standard_dr_count(cfg.ESRSStandard.S1)
        assert count == 17

    def test_get_standard_dr_count_esrs1_is_zero(self, cfg):
        """get_standard_dr_count returns 0 for ESRS_1 (methodology standard, no DRs in dict)."""
        count = cfg.get_standard_dr_count(cfg.ESRSStandard.ESRS_1)
        assert count == 0

    def test_validate_config_returns_list(self, cfg):
        """validate_config returns a list of warnings."""
        config = cfg.ESRSFullCoverageConfig()
        warnings = cfg.validate_config(config)
        assert isinstance(warnings, list)

    def test_get_default_config(self, cfg):
        """get_default_config returns ESRSFullCoverageConfig instance."""
        config = cfg.get_default_config("MANUFACTURING")
        assert config.sector == "MANUFACTURING"

    def test_get_default_config_general_sector(self, cfg):
        """get_default_config with GENERAL sector returns correct default."""
        config = cfg.get_default_config("GENERAL")
        assert config.sector == "GENERAL"

    def test_list_available_presets(self, cfg):
        """list_available_presets returns 6 presets."""
        presets = cfg.list_available_presets()
        assert len(presets) == 6
        assert "manufacturing" in presets
        assert "financial_services" in presets

    def test_list_available_presets_returns_copy(self, cfg):
        """list_available_presets returns a copy, not the original dict."""
        presets1 = cfg.list_available_presets()
        presets2 = cfg.list_available_presets()
        assert presets1 is not presets2
        assert presets1 == presets2

    def test_get_standard_label(self, cfg):
        """get_standard_label returns human-readable labels."""
        label = cfg.get_standard_label(cfg.ESRSStandard.E1)
        assert "Climate" in label or "E1" in label

    def test_get_standard_label_all_standards(self, cfg):
        """get_standard_label returns non-empty label for all 12 standards."""
        for std in cfg.ESRSStandard:
            label = cfg.get_standard_label(std)
            assert len(label) > 0, f"Empty label for {std.value}"

    def test_get_disclosure_requirements_function(self, cfg):
        """get_disclosure_requirements returns DRs for a given standard."""
        drs = cfg.get_disclosure_requirements(cfg.ESRSStandard.E2)
        assert len(drs) == 6
        assert "E2-1" in drs

    def test_get_disclosure_requirements_esrs1_empty(self, cfg):
        """get_disclosure_requirements returns empty dict for ESRS_1."""
        drs = cfg.get_disclosure_requirements(cfg.ESRSStandard.ESRS_1)
        assert drs == {}
