# -*- coding: utf-8 -*-
"""
Tests for Data Models and Config - AGENT-EUDR-005 Land Use Change Detector

Comprehensive test suite covering:
- Enumeration values and completeness (land use categories, classification
  methods, transition types, trajectory types, compliance verdicts,
  conversion types, risk tiers, infrastructure types, report types,
  report formats, expansion scales, spectral bands)
- LandUseClassification creation, defaults, all-method results
- LandUseTransition creation, deforestation flag, degradation flag
- TransitionMatrix creation, region-level statistics, provenance
- TemporalTrajectory creation, trajectory types, change date
- CutoffVerification creation, verdict logic, regulatory references
- CroplandExpansion creation, commodity, scale, expansion rate
- ConversionRiskAssessment creation, factor scoring, tier classification
- UrbanEncroachment creation, infrastructure type, buffer zone risk
- ComplianceReport creation, report type, format, regulatory framework
- ProvenanceRecord creation, hash chaining
- LandUseChangeConfig creation, defaults, __post_init__ validation
- Config from environment variables (GL_EUDR_LUC_ prefix)
- Config risk weight sum validation (must sum to 1.0)
- Config invalid classification method rejection
- Config invalid transition granularity rejection
- Config buffer_km consistency (default <= max)
- Config credential redaction (to_dict, __repr__)
- Config singleton pattern (get_config, set_config, reset_config)
- Config computed properties (cutoff_date_parsed, search_window_start,
  search_window_end, risk_factor_names, genesis_hash_sha256)
- Deterministic provenance hashing
- Serialization roundtrip for config to_dict

Test count: 100 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

import hashlib
import json
import os
from datetime import date, timedelta
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.land_use_change.config import (
    LandUseChangeConfig,
    get_config,
    set_config,
    reset_config,
    _DEFAULT_RISK_WEIGHTS,
    _DEFAULT_EUDR_COMMODITIES,
    _DEFAULT_SPECTRAL_BANDS,
    _DEFAULT_CLASSIFICATION_METHODS,
    _DEFAULT_TRAJECTORY_TYPES,
    _DEFAULT_RISK_TIERS,
    _VALID_CLASSIFICATION_METHODS,
    _VALID_GRANULARITIES,
    _VALID_TRAJECTORY_TYPES,
    _VALID_RISK_TIERS,
    _VALID_LOG_LEVELS,
)
from tests.agents.eudr.land_use_change.conftest import (
    # Test-only dataclass models
    SpectralData,
    VegetationIndices,
    TextureFeatures,
    PhenologyTimeSeries,
    LandUseClassification,
    LandUseTransition,
    TransitionMatrix,
    TemporalTrajectory,
    CutoffVerification,
    CroplandExpansion,
    ConversionRiskAssessment,
    UrbanEncroachment,
    ComplianceReport,
    ProvenanceRecord,
    # Helpers
    compute_test_hash,
    compute_ndvi,
    compute_evi,
    classify_risk_tier,
    determine_verdict,
    is_deforestation_transition,
    is_degradation_transition,
    weighted_composite,
    DeterministicUUID,
    # Constants
    SHA256_HEX_LENGTH,
    EUDR_DEFORESTATION_CUTOFF,
    EUDR_CUTOFF_DATE,
    EUDR_COMMODITIES,
    LAND_USE_CATEGORIES,
    CLASSIFICATION_METHODS,
    TRANSITION_TYPES,
    TRAJECTORY_TYPES,
    COMPLIANCE_VERDICTS,
    CONVERSION_TYPES,
    RISK_TIERS,
    RISK_FACTORS,
    DEFAULT_RISK_WEIGHTS,
    INFRASTRUCTURE_TYPES,
    REPORT_TYPES,
    REPORT_FORMATS,
    EXPANSION_SCALES,
    SPECTRAL_BANDS,
    ENSEMBLE_WEIGHTS,
    VERDICT_REGULATORY_REFS,
)


# ===========================================================================
# 1. Enumeration Value Tests (24 tests)
# ===========================================================================


class TestLandUseCategoryEnum:
    """Tests for LAND_USE_CATEGORIES constant set."""

    def test_land_use_category_values(self):
        """Test LAND_USE_CATEGORIES contains all 10 IPCC categories."""
        expected = {
            "forest", "shrubland", "grassland", "cropland", "wetland",
            "water", "urban", "bare_soil", "snow_ice", "other",
        }
        assert set(LAND_USE_CATEGORIES) == expected

    def test_land_use_category_count(self):
        """Test LAND_USE_CATEGORIES has exactly 10 categories."""
        assert len(LAND_USE_CATEGORIES) == 10

    def test_forest_in_categories(self):
        """Forest is a valid land use category."""
        assert "forest" in LAND_USE_CATEGORIES

    def test_cropland_in_categories(self):
        """Cropland is a valid land use category."""
        assert "cropland" in LAND_USE_CATEGORIES


class TestClassificationMethodEnum:
    """Tests for CLASSIFICATION_METHODS constant set."""

    def test_classification_method_values(self):
        """Test CLASSIFICATION_METHODS contains all 5 methods."""
        expected = {
            "spectral", "vegetation_index", "phenology", "texture", "ensemble",
        }
        assert set(CLASSIFICATION_METHODS) == expected

    def test_classification_method_count(self):
        """Test CLASSIFICATION_METHODS has exactly 5 methods."""
        assert len(CLASSIFICATION_METHODS) == 5


class TestTransitionTypeEnum:
    """Tests for TRANSITION_TYPES constant set."""

    def test_transition_type_values(self):
        """Test TRANSITION_TYPES contains all 8 types."""
        expected = {
            "deforestation", "degradation", "reforestation",
            "urbanization", "agricultural_expansion",
            "wetland_drainage", "stable", "unknown",
        }
        assert set(TRANSITION_TYPES) == expected

    def test_transition_type_count(self):
        """Test TRANSITION_TYPES has exactly 8 types."""
        assert len(TRANSITION_TYPES) == 8

    def test_deforestation_in_types(self):
        """Deforestation is a valid transition type."""
        assert "deforestation" in TRANSITION_TYPES

    def test_stable_in_types(self):
        """Stable is a valid transition type."""
        assert "stable" in TRANSITION_TYPES


class TestTrajectoryTypeEnum:
    """Tests for TRAJECTORY_TYPES constant set."""

    def test_trajectory_type_values(self):
        """Test TRAJECTORY_TYPES contains all 5 types."""
        expected = {
            "stable", "abrupt_change", "gradual_change",
            "oscillating", "recovery",
        }
        assert set(TRAJECTORY_TYPES) == expected

    def test_trajectory_type_count(self):
        """Test TRAJECTORY_TYPES has exactly 5 types."""
        assert len(TRAJECTORY_TYPES) == 5


class TestComplianceVerdictEnum:
    """Tests for COMPLIANCE_VERDICTS constant set."""

    def test_compliance_verdict_values(self):
        """Test COMPLIANCE_VERDICTS contains all 5 verdicts."""
        expected = {
            "compliant", "non_compliant", "degraded",
            "inconclusive", "pre_existing_agriculture",
        }
        assert set(COMPLIANCE_VERDICTS) == expected

    def test_compliance_verdict_count(self):
        """Test COMPLIANCE_VERDICTS has exactly 5 verdicts."""
        assert len(COMPLIANCE_VERDICTS) == 5


class TestConversionTypeEnum:
    """Tests for CONVERSION_TYPES constant set."""

    def test_conversion_type_values(self):
        """Test CONVERSION_TYPES contains all 7 commodity conversion types."""
        expected = {
            "palm_oil_conversion", "rubber_conversion",
            "cocoa_conversion", "coffee_conversion",
            "soya_conversion", "pasture_conversion",
            "timber_plantation_conversion",
        }
        assert set(CONVERSION_TYPES) == expected

    def test_conversion_type_count(self):
        """Test CONVERSION_TYPES has exactly 7 types."""
        assert len(CONVERSION_TYPES) == 7


class TestRiskTierEnum:
    """Tests for RISK_TIERS constant set."""

    def test_risk_tier_values(self):
        """Test RISK_TIERS contains all 4 tiers."""
        expected = {"low", "moderate", "high", "critical"}
        assert set(RISK_TIERS) == expected

    def test_risk_tier_count(self):
        """Test RISK_TIERS has exactly 4 tiers."""
        assert len(RISK_TIERS) == 4


class TestInfrastructureTypeEnum:
    """Tests for INFRASTRUCTURE_TYPES constant set."""

    def test_infrastructure_type_values(self):
        """Test INFRASTRUCTURE_TYPES contains all 5 types."""
        expected = {
            "road_construction", "building_expansion",
            "mining_activity", "industrial_development",
            "residential_growth",
        }
        assert set(INFRASTRUCTURE_TYPES) == expected

    def test_infrastructure_type_count(self):
        """Test INFRASTRUCTURE_TYPES has exactly 5 types."""
        assert len(INFRASTRUCTURE_TYPES) == 5


class TestReportAndFormatEnums:
    """Tests for REPORT_TYPES and REPORT_FORMATS constant sets."""

    def test_report_type_values(self):
        """Test REPORT_TYPES contains all 4 types."""
        expected = {"full", "summary", "compliance", "evidence"}
        assert set(REPORT_TYPES) == expected

    def test_report_format_values(self):
        """Test REPORT_FORMATS contains all 4 formats."""
        expected = {"json", "pdf", "csv", "eudr_xml"}
        assert set(REPORT_FORMATS) == expected


class TestMiscEnums:
    """Tests for other constant sets."""

    def test_expansion_scales(self):
        """Test EXPANSION_SCALES contains all 3 scales."""
        expected = {"smallholder", "medium", "industrial"}
        assert set(EXPANSION_SCALES) == expected

    def test_spectral_bands_count(self):
        """Test SPECTRAL_BANDS has exactly 10 Sentinel-2 bands."""
        assert len(SPECTRAL_BANDS) == 10

    def test_eudr_commodities_count(self):
        """Test EUDR_COMMODITIES has exactly 7 Article 1(1) commodities."""
        assert len(EUDR_COMMODITIES) == 7


# ===========================================================================
# 2. Core Model Dataclass Tests (20 tests)
# ===========================================================================


class TestLandUseClassificationModel:
    """Tests for LandUseClassification test-only dataclass."""

    def test_creation_defaults(self):
        """Test LandUseClassification creation with all defaults."""
        result = LandUseClassification()
        assert result.plot_id == ""
        assert result.category == "other"
        assert result.method == "ensemble"
        assert result.confidence == 0.0

    def test_creation_all_fields(self):
        """Test LandUseClassification creation with all fields."""
        result = LandUseClassification(
            plot_id="PLOT-001",
            category="forest",
            method="spectral",
            confidence=0.92,
            spectral_class="forest",
            vi_class="forest",
            phenology_class="forest",
            texture_class="forest",
            ensemble_class="forest",
            all_method_results={"spectral": "forest", "vi": "forest"},
            commodity_context="palm_oil",
            article_2_4_applies=False,
            provenance_hash="abc123",
        )
        assert result.plot_id == "PLOT-001"
        assert result.category == "forest"
        assert result.confidence == 0.92

    def test_all_method_results_dict(self):
        """Test all_method_results stores method-to-category mapping."""
        methods = {m: "forest" for m in CLASSIFICATION_METHODS}
        result = LandUseClassification(all_method_results=methods)
        assert len(result.all_method_results) == 5
        assert result.all_method_results["ensemble"] == "forest"

    def test_article_2_4_flag(self):
        """Test article_2_4_applies flag for timber exclusions."""
        result = LandUseClassification(article_2_4_applies=True)
        assert result.article_2_4_applies is True


class TestLandUseTransitionModel:
    """Tests for LandUseTransition test-only dataclass."""

    def test_creation_defaults(self):
        """Test LandUseTransition creation with defaults."""
        result = LandUseTransition()
        assert result.transition_type == "stable"
        assert result.is_deforestation is False
        assert result.is_degradation is False

    def test_deforestation_transition(self):
        """Test deforestation transition forest -> cropland."""
        result = LandUseTransition(
            plot_id="PLOT-DEF-001",
            from_category="forest",
            to_category="cropland",
            transition_type="deforestation",
            is_deforestation=True,
            confidence=0.90,
            area_ha=50.0,
        )
        assert result.is_deforestation is True
        assert result.from_category == "forest"
        assert result.to_category == "cropland"

    def test_evidence_dict(self):
        """Test evidence dictionary stores supporting data."""
        evidence = {
            "ndvi_drop": -0.45,
            "spectral_angle": 15.2,
            "time_series_break": "2021-06-15",
        }
        result = LandUseTransition(evidence=evidence)
        assert len(result.evidence) == 3
        assert result.evidence["ndvi_drop"] == -0.45


class TestTransitionMatrixModel:
    """Tests for TransitionMatrix test-only dataclass."""

    def test_creation_defaults(self):
        """Test TransitionMatrix creation with defaults."""
        result = TransitionMatrix()
        assert result.region_id == ""
        assert result.total_area_ha == 0.0

    def test_creation_with_matrix(self):
        """Test TransitionMatrix with a populated matrix."""
        matrix = {
            "forest": {"forest": 900.0, "cropland": 100.0},
            "cropland": {"forest": 10.0, "cropland": 490.0},
        }
        result = TransitionMatrix(
            region_id="AMZ-001",
            period_start="2018-01-01",
            period_end="2023-01-01",
            matrix=matrix,
            total_area_ha=1500.0,
        )
        assert result.matrix["forest"]["cropland"] == 100.0
        assert result.total_area_ha == 1500.0


class TestTemporalTrajectoryModel:
    """Tests for TemporalTrajectory test-only dataclass."""

    def test_creation_defaults(self):
        """Test TemporalTrajectory creation with defaults."""
        result = TemporalTrajectory()
        assert result.trajectory_type == "stable"
        assert result.change_date is None
        assert result.is_natural_disturbance is False

    def test_abrupt_change_trajectory(self):
        """Test abrupt change trajectory with change date."""
        result = TemporalTrajectory(
            plot_id="PLOT-ABR-001",
            trajectory_type="abrupt_change",
            change_date="2021-03-15",
            change_date_range=("2021-03-01", "2021-03-31"),
            confidence=0.88,
        )
        assert result.trajectory_type == "abrupt_change"
        assert result.change_date == "2021-03-15"

    def test_oscillating_trajectory(self):
        """Test oscillating trajectory with period."""
        result = TemporalTrajectory(
            trajectory_type="oscillating",
            oscillation_period_months=12,
        )
        assert result.oscillation_period_months == 12


class TestCutoffVerificationModel:
    """Tests for CutoffVerification test-only dataclass."""

    def test_creation_defaults(self):
        """Test CutoffVerification creation with defaults."""
        result = CutoffVerification()
        assert result.verdict == "inconclusive"
        assert result.article_2_4_applies is False

    def test_compliant_verification(self):
        """Test compliant verification result."""
        result = CutoffVerification(
            plot_id="PLOT-CMP-001",
            verdict="compliant",
            cutoff_category="forest",
            current_category="forest",
            cutoff_confidence=0.90,
            current_confidence=0.88,
            transition_detected=False,
            commodity="soya",
            regulatory_references=["EUDR Art. 3(a)", "EUDR Art. 10(1)"],
        )
        assert result.verdict == "compliant"
        assert len(result.regulatory_references) == 2


class TestCroplandExpansionModel:
    """Tests for CroplandExpansion test-only dataclass."""

    def test_creation_defaults(self):
        """Test CroplandExpansion creation with defaults."""
        result = CroplandExpansion()
        assert result.scale == "smallholder"
        assert result.to_category == "cropland"

    def test_industrial_palm_oil(self):
        """Test industrial palm oil conversion."""
        result = CroplandExpansion(
            conversion_type="palm_oil_conversion",
            commodity="palm_oil",
            scale="industrial",
            area_converted_ha=200.0,
            expansion_rate_ha_per_year=50.0,
        )
        assert result.scale == "industrial"
        assert result.area_converted_ha == 200.0


class TestConversionRiskAssessmentModel:
    """Tests for ConversionRiskAssessment test-only dataclass."""

    def test_creation_defaults(self):
        """Test ConversionRiskAssessment creation with defaults."""
        result = ConversionRiskAssessment()
        assert result.risk_tier == "low"
        assert result.composite_score == 0.0

    def test_critical_risk(self):
        """Test critical risk assessment."""
        result = ConversionRiskAssessment(
            plot_id="PLOT-CRIT-001",
            risk_tier="critical",
            composite_score=0.85,
            conversion_probability_12m=0.78,
            is_deforestation_frontier=True,
        )
        assert result.risk_tier == "critical"
        assert result.is_deforestation_frontier is True


class TestUrbanEncroachmentModel:
    """Tests for UrbanEncroachment test-only dataclass."""

    def test_creation_defaults(self):
        """Test UrbanEncroachment creation with defaults."""
        result = UrbanEncroachment()
        assert result.urban_proximity_km == 0.0
        assert result.buffer_zone_risk == 0.0

    def test_road_construction_encroachment(self):
        """Test road construction encroachment detection."""
        result = UrbanEncroachment(
            plot_id="PLOT-ROAD-001",
            infrastructure_type="road_construction",
            expansion_rate_ha_per_year=12.0,
            urban_proximity_km=3.5,
            buffer_zone_risk=0.72,
            confidence=0.85,
        )
        assert result.infrastructure_type == "road_construction"
        assert result.buffer_zone_risk == 0.72


class TestComplianceReportModel:
    """Tests for ComplianceReport test-only dataclass."""

    def test_creation_defaults(self):
        """Test ComplianceReport creation with defaults."""
        result = ComplianceReport()
        assert result.report_type == "full"
        assert result.report_format == "json"
        assert result.regulatory_framework == "EUDR EU 2023/1115"

    def test_eudr_xml_report(self):
        """Test EUDR XML format report."""
        result = ComplianceReport(
            report_id="RPT-001",
            plot_id="PLOT-001",
            report_type="compliance",
            report_format="eudr_xml",
            verdict="non_compliant",
        )
        assert result.report_format == "eudr_xml"
        assert result.verdict == "non_compliant"


class TestProvenanceRecordModel:
    """Tests for ProvenanceRecord test-only dataclass."""

    def test_creation_defaults(self):
        """Test ProvenanceRecord creation with defaults."""
        result = ProvenanceRecord()
        assert result.entity_type == ""
        assert result.hash_value == ""

    def test_provenance_chain(self):
        """Test provenance chain with parent hash."""
        genesis = compute_test_hash({"genesis": "test"})
        record = ProvenanceRecord(
            entity_type="classification",
            entity_id="CLF-001",
            action="created",
            hash_value=compute_test_hash({"id": "CLF-001"}),
            parent_hash=genesis,
            timestamp="2026-01-15T10:00:00Z",
        )
        assert len(record.hash_value) == SHA256_HEX_LENGTH
        assert len(record.parent_hash) == SHA256_HEX_LENGTH
        assert record.hash_value != record.parent_hash


# ===========================================================================
# 3. Config Creation and Defaults Tests (12 tests)
# ===========================================================================


class TestConfigDefaults:
    """Tests for LandUseChangeConfig default values."""

    def test_config_default_creation(self):
        """Test LandUseChangeConfig with all defaults."""
        cfg = LandUseChangeConfig()
        assert cfg.num_classes == 10
        assert cfg.default_method == "ensemble"
        assert cfg.min_confidence == 0.60
        assert cfg.cutoff_date == "2020-12-31"

    def test_config_default_spectral_bands(self):
        """Test default spectral bands match Sentinel-2 10-band selection."""
        cfg = LandUseChangeConfig()
        assert cfg.spectral_bands == _DEFAULT_SPECTRAL_BANDS

    def test_config_default_classification_methods(self):
        """Test default classification methods list."""
        cfg = LandUseChangeConfig()
        assert cfg.classification_methods == _DEFAULT_CLASSIFICATION_METHODS

    def test_config_default_trajectory_types(self):
        """Test default trajectory types list."""
        cfg = LandUseChangeConfig()
        assert cfg.trajectory_types == _DEFAULT_TRAJECTORY_TYPES

    def test_config_default_risk_tiers(self):
        """Test default risk tiers list."""
        cfg = LandUseChangeConfig()
        assert cfg.risk_tiers == _DEFAULT_RISK_TIERS

    def test_config_default_risk_weights(self):
        """Test default risk weights match 8-factor model."""
        cfg = LandUseChangeConfig()
        assert cfg.risk_weights == _DEFAULT_RISK_WEIGHTS

    def test_config_default_risk_weights_sum_to_one(self):
        """Test default risk weights sum to exactly 1.0."""
        cfg = LandUseChangeConfig()
        assert abs(sum(cfg.risk_weights.values()) - 1.0) < 0.001

    def test_config_default_eudr_commodities(self):
        """Test default EUDR commodities contain 7 Article 1(1) items."""
        cfg = LandUseChangeConfig()
        assert cfg.eudr_commodities == _DEFAULT_EUDR_COMMODITIES
        assert len(cfg.eudr_commodities) == 7

    def test_config_default_conservative_bias(self):
        """Test conservative bias defaults to True for regulatory compliance."""
        cfg = LandUseChangeConfig()
        assert cfg.conservative_bias is True

    def test_config_default_batch_size(self):
        """Test default batch size is 1000."""
        cfg = LandUseChangeConfig()
        assert cfg.batch_size == 1000

    def test_config_default_cache_ttl(self):
        """Test default cache TTL is 3600 seconds."""
        cfg = LandUseChangeConfig()
        assert cfg.cache_ttl_seconds == 3600

    def test_config_default_enable_metrics(self):
        """Test default enable_metrics is True."""
        cfg = LandUseChangeConfig()
        assert cfg.enable_metrics is True


# ===========================================================================
# 4. Config Validation Tests (16 tests)
# ===========================================================================


class TestConfigValidation:
    """Tests for LandUseChangeConfig __post_init__ validation."""

    def test_invalid_log_level(self):
        """Test invalid log_level raises ValueError."""
        with pytest.raises(ValueError, match="log_level"):
            LandUseChangeConfig(log_level="VERBOSE")

    def test_log_level_normalized_uppercase(self):
        """Test log_level is normalized to uppercase."""
        cfg = LandUseChangeConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_num_classes_too_low(self):
        """Test num_classes below 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_classes"):
            LandUseChangeConfig(num_classes=0)

    def test_num_classes_too_high(self):
        """Test num_classes above 50 raises ValueError."""
        with pytest.raises(ValueError, match="num_classes"):
            LandUseChangeConfig(num_classes=51)

    def test_invalid_default_method(self):
        """Test invalid default_method raises ValueError."""
        with pytest.raises(ValueError, match="default_method"):
            LandUseChangeConfig(default_method="random_forest")

    def test_min_confidence_below_zero(self):
        """Test min_confidence below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence"):
            LandUseChangeConfig(min_confidence=-0.1)

    def test_min_confidence_above_one(self):
        """Test min_confidence above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence"):
            LandUseChangeConfig(min_confidence=1.5)

    def test_invalid_granularity(self):
        """Test invalid transition_date_granularity raises ValueError."""
        with pytest.raises(ValueError, match="transition_date_granularity"):
            LandUseChangeConfig(transition_date_granularity="hourly")

    def test_invalid_cutoff_date(self):
        """Test invalid cutoff_date raises ValueError."""
        with pytest.raises(ValueError, match="cutoff_date"):
            LandUseChangeConfig(cutoff_date="not-a-date")

    def test_search_window_below_one(self):
        """Test search_window_days below 1 raises ValueError."""
        with pytest.raises(ValueError, match="search_window_days"):
            LandUseChangeConfig(search_window_days=0)

    def test_search_window_above_365(self):
        """Test search_window_days above 365 raises ValueError."""
        with pytest.raises(ValueError, match="search_window_days"):
            LandUseChangeConfig(search_window_days=400)

    def test_risk_weights_do_not_sum_to_one(self):
        """Test risk weights not summing to 1.0 raises ValueError."""
        bad_weights = dict(_DEFAULT_RISK_WEIGHTS)
        bad_weights["transition_magnitude"] = 0.50  # pushes sum well above 1.0
        with pytest.raises(ValueError, match="risk_weights must sum to 1.0"):
            LandUseChangeConfig(risk_weights=bad_weights)

    def test_default_buffer_exceeds_max(self):
        """Test default_buffer_km > max_buffer_km raises ValueError."""
        with pytest.raises(ValueError, match="default_buffer_km"):
            LandUseChangeConfig(default_buffer_km=60.0, max_buffer_km=50.0)

    def test_negative_buffer(self):
        """Test negative default_buffer_km raises ValueError."""
        with pytest.raises(ValueError, match="default_buffer_km"):
            LandUseChangeConfig(default_buffer_km=-1.0)

    def test_empty_genesis_hash(self):
        """Test empty genesis_hash raises ValueError."""
        with pytest.raises(ValueError, match="genesis_hash"):
            LandUseChangeConfig(genesis_hash="")

    def test_valid_all_granularities(self):
        """Test all valid granularity values pass validation."""
        for g in _VALID_GRANULARITIES:
            cfg = LandUseChangeConfig(transition_date_granularity=g)
            assert cfg.transition_date_granularity == g


# ===========================================================================
# 5. Config Computed Properties Tests (8 tests)
# ===========================================================================


class TestConfigComputedProperties:
    """Tests for LandUseChangeConfig computed properties."""

    def test_cutoff_date_parsed(self):
        """Test cutoff_date_parsed returns correct date object."""
        cfg = LandUseChangeConfig()
        assert cfg.cutoff_date_parsed == date(2020, 12, 31)

    def test_cutoff_date_parsed_custom(self):
        """Test custom cutoff date is parsed correctly."""
        cfg = LandUseChangeConfig(cutoff_date="2023-06-30")
        assert cfg.cutoff_date_parsed == date(2023, 6, 30)

    def test_search_window_start(self):
        """Test search_window_start is cutoff - half window."""
        cfg = LandUseChangeConfig(cutoff_date="2020-12-31", search_window_days=60)
        expected = date(2020, 12, 31) - timedelta(days=30)
        assert cfg.search_window_start == expected

    def test_search_window_end(self):
        """Test search_window_end is cutoff + half window."""
        cfg = LandUseChangeConfig(cutoff_date="2020-12-31", search_window_days=60)
        expected = date(2020, 12, 31) + timedelta(days=30)
        assert cfg.search_window_end == expected

    def test_search_window_bracket_cutoff(self):
        """Test search window start <= cutoff <= search window end."""
        cfg = LandUseChangeConfig()
        assert cfg.search_window_start <= cfg.cutoff_date_parsed
        assert cfg.cutoff_date_parsed <= cfg.search_window_end

    def test_risk_factor_names_sorted(self):
        """Test risk_factor_names returns sorted list."""
        cfg = LandUseChangeConfig()
        names = cfg.risk_factor_names
        assert names == sorted(names)
        assert len(names) == 8

    def test_genesis_hash_sha256_length(self):
        """Test genesis_hash_sha256 is 64-char hex digest."""
        cfg = LandUseChangeConfig()
        assert len(cfg.genesis_hash_sha256) == SHA256_HEX_LENGTH

    def test_genesis_hash_sha256_deterministic(self):
        """Test genesis_hash_sha256 is deterministic across instances."""
        cfg1 = LandUseChangeConfig()
        cfg2 = LandUseChangeConfig()
        assert cfg1.genesis_hash_sha256 == cfg2.genesis_hash_sha256


# ===========================================================================
# 6. Config Serialization Tests (5 tests)
# ===========================================================================


class TestConfigSerialization:
    """Tests for LandUseChangeConfig to_dict and __repr__."""

    def test_to_dict_returns_dict(self):
        """Test to_dict returns a dictionary."""
        cfg = LandUseChangeConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_redacts_database_url(self):
        """Test to_dict redacts database_url."""
        cfg = LandUseChangeConfig(database_url="postgresql://user:pass@host/db")
        d = cfg.to_dict()
        assert d["database_url"] == "***"

    def test_to_dict_redacts_redis_url(self):
        """Test to_dict redacts redis_url."""
        cfg = LandUseChangeConfig(redis_url="redis://:secret@host:6379/0")
        d = cfg.to_dict()
        assert d["redis_url"] == "***"

    def test_repr_does_not_leak_credentials(self):
        """Test __repr__ does not leak connection strings."""
        cfg = LandUseChangeConfig(
            database_url="postgresql://user:password@prod-host/db",
            redis_url="redis://:mysecret@redis-host:6379/1",
        )
        representation = repr(cfg)
        assert "password" not in representation
        assert "mysecret" not in representation
        assert "***" in representation

    def test_to_dict_roundtrip_keys(self):
        """Test to_dict contains all expected keys."""
        cfg = LandUseChangeConfig()
        d = cfg.to_dict()
        expected_keys = {
            "database_url", "redis_url", "log_level",
            "num_classes", "default_method", "min_confidence",
            "spectral_bands", "classification_methods",
            "min_transition_area_ha", "transition_date_granularity",
            "deforestation_precision_target",
            "min_temporal_depth_years", "trajectory_types", "max_time_steps",
            "cutoff_date", "search_window_days", "conservative_bias",
            "risk_tiers", "risk_weights",
            "default_buffer_km", "max_buffer_km",
            "eudr_commodities",
            "batch_size", "max_concurrent_jobs",
            "cache_ttl_seconds",
            "genesis_hash", "enable_metrics",
        }
        assert set(d.keys()) == expected_keys


# ===========================================================================
# 7. Config Singleton Tests (5 tests)
# ===========================================================================


class TestConfigSingleton:
    """Tests for get_config, set_config, reset_config singleton pattern."""

    def test_get_config_returns_instance(self):
        """Test get_config returns a LandUseChangeConfig instance."""
        reset_config()
        cfg = get_config()
        assert isinstance(cfg, LandUseChangeConfig)

    def test_get_config_returns_same_instance(self):
        """Test get_config returns the same instance on repeated calls."""
        reset_config()
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_instance(self):
        """Test set_config replaces the singleton."""
        custom = LandUseChangeConfig(num_classes=15)
        set_config(custom)
        cfg = get_config()
        assert cfg.num_classes == 15

    def test_reset_config_clears_singleton(self):
        """Test reset_config clears the singleton so next get_config creates new."""
        custom = LandUseChangeConfig(num_classes=20)
        set_config(custom)
        reset_config()
        cfg = get_config()
        # After reset, get_config reads from env/defaults, not our custom
        assert cfg.num_classes == 10  # default

    def test_set_then_reset_then_set(self):
        """Test multiple set/reset cycles work correctly."""
        set_config(LandUseChangeConfig(num_classes=5))
        assert get_config().num_classes == 5
        reset_config()
        set_config(LandUseChangeConfig(num_classes=8))
        assert get_config().num_classes == 8


# ===========================================================================
# 8. Config from_env Tests (5 tests)
# ===========================================================================


class TestConfigFromEnv:
    """Tests for LandUseChangeConfig.from_env()."""

    def test_from_env_default(self):
        """Test from_env with no env vars returns defaults."""
        cfg = LandUseChangeConfig.from_env()
        assert cfg.num_classes == 10
        assert cfg.cutoff_date == "2020-12-31"

    def test_from_env_override_num_classes(self):
        """Test from_env reads GL_EUDR_LUC_NUM_CLASSES."""
        with patch.dict(os.environ, {"GL_EUDR_LUC_NUM_CLASSES": "15"}):
            cfg = LandUseChangeConfig.from_env()
            assert cfg.num_classes == 15

    def test_from_env_override_min_confidence(self):
        """Test from_env reads GL_EUDR_LUC_MIN_CONFIDENCE."""
        with patch.dict(os.environ, {"GL_EUDR_LUC_MIN_CONFIDENCE": "0.85"}):
            cfg = LandUseChangeConfig.from_env()
            assert cfg.min_confidence == 0.85

    def test_from_env_override_conservative_bias_false(self):
        """Test from_env reads GL_EUDR_LUC_CONSERVATIVE_BIAS=false."""
        with patch.dict(os.environ, {"GL_EUDR_LUC_CONSERVATIVE_BIAS": "false"}):
            cfg = LandUseChangeConfig.from_env()
            assert cfg.conservative_bias is False

    def test_from_env_override_cutoff_date(self):
        """Test from_env reads GL_EUDR_LUC_CUTOFF_DATE."""
        with patch.dict(os.environ, {"GL_EUDR_LUC_CUTOFF_DATE": "2023-06-30"}):
            cfg = LandUseChangeConfig.from_env()
            assert cfg.cutoff_date == "2023-06-30"


# ===========================================================================
# 9. Helper Function Tests (10 tests)
# ===========================================================================


class TestHelperFunctions:
    """Tests for computation helper functions defined in conftest."""

    def test_compute_ndvi_forest(self):
        """Test NDVI computation for forest (high NIR, low red)."""
        ndvi = compute_ndvi(red=0.025, nir=0.350)
        assert 0.85 < ndvi < 0.90

    def test_compute_ndvi_water(self):
        """Test NDVI computation for water (low NIR, moderate red)."""
        ndvi = compute_ndvi(red=0.060, nir=0.020)
        assert ndvi < 0.0

    def test_compute_ndvi_zero_denominator(self):
        """Test NDVI returns 0.0 when both bands are zero."""
        assert compute_ndvi(red=0.0, nir=0.0) == 0.0

    def test_compute_evi_forest(self):
        """Test EVI computation for forest reflectance."""
        evi = compute_evi(blue=0.030, red=0.025, nir=0.350)
        assert 0.5 < evi < 1.0

    def test_classify_risk_tier_low(self):
        """Test risk tier classification for low score."""
        assert classify_risk_tier(0.10) == "low"

    def test_classify_risk_tier_critical(self):
        """Test risk tier classification for critical score."""
        assert classify_risk_tier(0.80) == "critical"

    def test_determine_verdict_compliant(self):
        """Test verdict determination for forest -> forest."""
        verdict = determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=True,
            confidence=0.90,
        )
        assert verdict == "compliant"

    def test_determine_verdict_non_compliant(self):
        """Test verdict determination for forest -> non-forest."""
        verdict = determine_verdict(
            cutoff_was_forest=True,
            current_is_forest=False,
            confidence=0.90,
        )
        assert verdict == "non_compliant"

    def test_is_deforestation_transition_true(self):
        """Test forest -> cropland is deforestation."""
        assert is_deforestation_transition("forest", "cropland") is True

    def test_is_deforestation_transition_false(self):
        """Test cropland -> cropland is NOT deforestation."""
        assert is_deforestation_transition("cropland", "cropland") is False


# ===========================================================================
# 10. Provenance and Determinism Tests (5 tests)
# ===========================================================================


class TestProvenanceDeterminism:
    """Tests for provenance hashing and deterministic behavior."""

    def test_compute_test_hash_length(self):
        """Test compute_test_hash returns 64-char hex digest."""
        h = compute_test_hash({"key": "value"})
        assert len(h) == SHA256_HEX_LENGTH

    def test_compute_test_hash_deterministic(self):
        """Test same input produces same hash."""
        h1 = compute_test_hash({"plot_id": "PLOT-001"})
        h2 = compute_test_hash({"plot_id": "PLOT-001"})
        assert h1 == h2

    def test_compute_test_hash_different_inputs(self):
        """Test different inputs produce different hashes."""
        h1 = compute_test_hash({"plot_id": "PLOT-001"})
        h2 = compute_test_hash({"plot_id": "PLOT-002"})
        assert h1 != h2

    def test_deterministic_uuid_sequential(self):
        """Test DeterministicUUID generates sequential IDs."""
        gen = DeterministicUUID(prefix="test")
        ids = [gen.next() for _ in range(5)]
        assert ids == [
            "test-00000001",
            "test-00000002",
            "test-00000003",
            "test-00000004",
            "test-00000005",
        ]

    def test_deterministic_uuid_reset(self):
        """Test DeterministicUUID reset restarts counter."""
        gen = DeterministicUUID(prefix="r")
        gen.next()
        gen.next()
        gen.reset()
        assert gen.next() == "r-00000001"


# ===========================================================================
# 11. EUDR Regulatory Constants Tests (5 tests)
# ===========================================================================


class TestEUDRRegulatoryConstants:
    """Tests for EUDR regulatory reference constants."""

    def test_eudr_cutoff_date_string(self):
        """Test EUDR_DEFORESTATION_CUTOFF is '2020-12-31'."""
        assert EUDR_DEFORESTATION_CUTOFF == "2020-12-31"

    def test_eudr_cutoff_date_object(self):
        """Test EUDR_CUTOFF_DATE is date(2020, 12, 31)."""
        assert EUDR_CUTOFF_DATE == date(2020, 12, 31)

    def test_verdict_regulatory_refs_complete(self):
        """Test all verdicts have regulatory references."""
        for verdict in COMPLIANCE_VERDICTS:
            assert verdict in VERDICT_REGULATORY_REFS
            assert len(VERDICT_REGULATORY_REFS[verdict]) > 0

    def test_ensemble_weights_sum_to_one(self):
        """Test ENSEMBLE_WEIGHTS sum to 1.0."""
        assert abs(sum(ENSEMBLE_WEIGHTS.values()) - 1.0) < 0.001

    def test_risk_factors_match_weights(self):
        """Test RISK_FACTORS list matches DEFAULT_RISK_WEIGHTS keys."""
        assert set(RISK_FACTORS) == set(DEFAULT_RISK_WEIGHTS.keys())
