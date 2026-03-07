# -*- coding: utf-8 -*-
"""
Tests for Data Models and Config - AGENT-EUDR-003 Satellite Monitoring

Comprehensive test suite covering:
- Enumeration values and completeness (satellite sources, spectral indices,
  forest classifications, change classifications, detection methods,
  alert severities, monitoring intervals, evidence formats, cloud fill methods)
- SceneMetadata creation, defaults, field ranges, and bands
- DataQualityAssessment creation, scoring, and acceptability
- BaselineSnapshot creation, field ranges, and provenance hash
- ChangeDetectionResult creation, deforestation flag, and confidence
- SatelliteAlert creation, acknowledge fields
- EvidencePackage creation, compliance values
- FusionResult creation and agreement score
- CloudGapFillResult creation and quality score
- SatelliteMonitoringConfig creation, defaults, and validation
- Config from environment variables (GL_EUDR_SAT_ prefix)
- Config fusion weight sum validation
- Config NDVI threshold ordering
- Config timeout ordering
- Config credential redaction (to_dict, __repr__)
- Config singleton pattern (get_config, set_config, reset_config)
- Config computed properties (timeout_by_level, fusion_weights)
- JSON roundtrip tests for all model types
- Deterministic provenance hashing

Test count: 60+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-003 (Satellite Monitoring Models and Config)
"""

import json
import os
from datetime import date
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.satellite_monitoring.config import (
    SatelliteMonitoringConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.satellite_monitoring.imagery_acquisition import (
    SceneMetadata,
    DataQualityAssessment,
    SENTINEL2_BAND_SPECS,
    LANDSAT_BAND_SPECS,
)
from tests.agents.eudr.satellite_monitoring.conftest import (
    BaselineSnapshot,
    ChangeDetectionResult,
    SatelliteAlert,
    EvidencePackage,
    FusionResult,
    CloudGapFillResult,
    compute_test_hash,
    SHA256_HEX_LENGTH,
    SATELLITE_SOURCES,
    SPECTRAL_INDICES,
    FOREST_CLASSIFICATIONS,
    CHANGE_CLASSIFICATIONS,
    DETECTION_METHODS,
    ALERT_SEVERITIES,
    MONITORING_INTERVALS,
    EVIDENCE_FORMATS,
    CLOUD_FILL_METHODS,
    EUDR_COMMODITIES,
    EUDR_DEFORESTATION_CUTOFF,
)


# ===========================================================================
# 1. Enumeration Tests (15 tests)
# ===========================================================================


class TestEnumerations:
    """Tests for all enumeration-like constant sets."""

    def test_satellite_source_values(self):
        """Test SATELLITE_SOURCES contains all expected sources."""
        expected = {"sentinel2", "landsat8", "landsat9"}
        assert set(SATELLITE_SOURCES) == expected

    def test_satellite_source_count(self):
        """Test SATELLITE_SOURCES has exactly 3 sources."""
        assert len(SATELLITE_SOURCES) == 3

    def test_spectral_index_values(self):
        """Test SPECTRAL_INDICES contains all expected indices."""
        expected = {"ndvi", "evi", "nbr", "ndmi", "savi"}
        assert set(SPECTRAL_INDICES) == expected

    def test_spectral_index_count(self):
        """Test SPECTRAL_INDICES has exactly 5 indices."""
        assert len(SPECTRAL_INDICES) == 5

    def test_forest_classification_values(self):
        """Test FOREST_CLASSIFICATIONS contains all expected levels."""
        expected = {
            "dense_forest", "forest", "shrubland",
            "sparse_vegetation", "non_vegetated",
        }
        assert set(FOREST_CLASSIFICATIONS) == expected

    def test_forest_classification_count(self):
        """Test FOREST_CLASSIFICATIONS has exactly 5 levels."""
        assert len(FOREST_CLASSIFICATIONS) == 5

    def test_change_classification_values(self):
        """Test CHANGE_CLASSIFICATIONS contains all expected types."""
        expected = {"deforestation", "degradation", "regrowth", "no_change"}
        assert set(CHANGE_CLASSIFICATIONS) == expected

    def test_change_classification_count(self):
        """Test CHANGE_CLASSIFICATIONS has exactly 4 types."""
        assert len(CHANGE_CLASSIFICATIONS) == 4

    def test_detection_method_values(self):
        """Test DETECTION_METHODS contains all expected methods."""
        expected = {"ndvi_differencing", "spectral_angle", "time_series_break"}
        assert set(DETECTION_METHODS) == expected

    def test_alert_severity_values(self):
        """Test ALERT_SEVERITIES contains all expected levels."""
        expected = {"critical", "warning", "info"}
        assert set(ALERT_SEVERITIES) == expected

    def test_alert_severity_ordering(self):
        """Test critical appears before info in severity list."""
        assert ALERT_SEVERITIES.index("critical") < ALERT_SEVERITIES.index("info")

    def test_monitoring_interval_values(self):
        """Test MONITORING_INTERVALS contains all expected intervals."""
        expected = {"daily", "weekly", "biweekly", "monthly", "quarterly"}
        assert set(MONITORING_INTERVALS) == expected

    def test_evidence_format_values(self):
        """Test EVIDENCE_FORMATS contains all expected formats."""
        expected = {"json", "csv", "pdf"}
        assert set(EVIDENCE_FORMATS) == expected

    def test_cloud_fill_method_values(self):
        """Test CLOUD_FILL_METHODS contains all expected methods."""
        expected = {
            "temporal_composite", "sar_fusion",
            "interpolation", "nearest_clear",
        }
        assert set(CLOUD_FILL_METHODS) == expected

    def test_eudr_commodities_values(self):
        """Test EUDR_COMMODITIES contains all 7 Article 9 commodities."""
        expected = {
            "cattle", "cocoa", "coffee",
            "oil_palm", "rubber", "soya", "wood",
        }
        assert set(EUDR_COMMODITIES) == expected
        assert len(EUDR_COMMODITIES) == 7


# ===========================================================================
# 2. SceneMetadata Tests (10 tests)
# ===========================================================================


class TestSceneMetadata:
    """Tests for SceneMetadata dataclass."""

    def test_creation_all_fields(self):
        """Test SceneMetadata creation with all fields."""
        scene = SceneMetadata(
            scene_id="S2A_20201231_T20MQS",
            source="sentinel2",
            acquisition_date=date(2020, 12, 31),
            cloud_cover_pct=8.5,
            spatial_coverage_pct=98.0,
            tile_id="T20MQS",
            resolution_m=10,
            sun_elevation_deg=65.0,
            sun_azimuth_deg=140.0,
            processing_level="L2A",
            bands_available=list(SENTINEL2_BAND_SPECS.keys()),
            file_size_mb=750.0,
            quality_score=85.0,
            provenance_hash="abcd1234",
        )
        assert scene.scene_id == "S2A_20201231_T20MQS"
        assert scene.source == "sentinel2"
        assert scene.acquisition_date == date(2020, 12, 31)
        assert scene.cloud_cover_pct == 8.5

    def test_creation_defaults(self):
        """Test SceneMetadata default values."""
        scene = SceneMetadata()
        assert scene.scene_id == ""
        assert scene.source == ""
        assert scene.acquisition_date is None
        assert scene.cloud_cover_pct == 0.0
        assert scene.spatial_coverage_pct == 100.0
        assert scene.resolution_m == 10
        assert scene.sun_elevation_deg == 45.0
        assert scene.processing_level == "L2A"
        assert scene.bands_available == []
        assert scene.file_size_mb == 0.0
        assert scene.quality_score == 0.0
        assert scene.provenance_hash == ""

    def test_quality_score_range(self):
        """Test quality score can span 0-100 range."""
        for score in [0.0, 25.0, 50.0, 75.0, 100.0]:
            scene = SceneMetadata(quality_score=score)
            assert scene.quality_score == score

    def test_bands_available_sentinel2(self):
        """Test Sentinel-2 bands list has 13 bands."""
        scene = SceneMetadata(
            bands_available=list(SENTINEL2_BAND_SPECS.keys()),
        )
        assert len(scene.bands_available) == 13

    def test_bands_available_landsat(self):
        """Test Landsat bands list has 11 bands."""
        scene = SceneMetadata(
            bands_available=list(LANDSAT_BAND_SPECS.keys()),
        )
        assert len(scene.bands_available) == 11

    def test_cloud_cover_pct_boundary(self):
        """Test cloud cover at boundary values."""
        scene_clear = SceneMetadata(cloud_cover_pct=0.0)
        scene_full = SceneMetadata(cloud_cover_pct=100.0)
        assert scene_clear.cloud_cover_pct == 0.0
        assert scene_full.cloud_cover_pct == 100.0

    @pytest.mark.parametrize("source", SATELLITE_SOURCES)
    def test_source_values(self, source):
        """Test SceneMetadata accepts all satellite source values."""
        scene = SceneMetadata(source=source)
        assert scene.source == source

    @pytest.mark.parametrize("resolution", [10, 20, 30, 60, 100])
    def test_resolution_values(self, resolution):
        """Test SceneMetadata accepts various resolution values."""
        scene = SceneMetadata(resolution_m=resolution)
        assert scene.resolution_m == resolution

    def test_provenance_hash_assignment(self):
        """Test provenance hash can be set to a SHA-256 hex string."""
        h = compute_test_hash({"scene_id": "TEST"})
        scene = SceneMetadata(provenance_hash=h)
        assert len(scene.provenance_hash) == SHA256_HEX_LENGTH

    def test_sun_angles(self):
        """Test sun elevation and azimuth fields."""
        scene = SceneMetadata(sun_elevation_deg=65.0, sun_azimuth_deg=140.0)
        assert scene.sun_elevation_deg == 65.0
        assert scene.sun_azimuth_deg == 140.0


# ===========================================================================
# 3. DataQualityAssessment Tests (8 tests)
# ===========================================================================


class TestDataQualityAssessment:
    """Tests for DataQualityAssessment dataclass."""

    def test_creation_all_fields(self):
        """Test DataQualityAssessment creation with all fields."""
        assessment = DataQualityAssessment(
            assessment_id="DQA-001",
            scene_id="S2A_20201231_T20MQS",
            cloud_cover_score=90.0,
            temporal_proximity_score=85.0,
            spatial_coverage_score=95.0,
            atmospheric_quality_score=80.0,
            sensor_health_score=92.0,
            overall_score=88.5,
            is_acceptable=True,
            details={"method": "weighted_average"},
            provenance_hash="abc123",
        )
        assert assessment.assessment_id == "DQA-001"
        assert assessment.overall_score == 88.5
        assert assessment.is_acceptable is True

    def test_creation_defaults(self):
        """Test DataQualityAssessment default values."""
        assessment = DataQualityAssessment()
        assert assessment.assessment_id == ""
        assert assessment.scene_id == ""
        assert assessment.cloud_cover_score == 0.0
        assert assessment.overall_score == 0.0
        assert assessment.is_acceptable is False
        assert assessment.details == {}
        assert assessment.provenance_hash == ""

    def test_score_range_valid(self):
        """Test all sub-scores at boundary values."""
        assessment = DataQualityAssessment(
            cloud_cover_score=100.0,
            temporal_proximity_score=0.0,
            spatial_coverage_score=50.0,
            atmospheric_quality_score=100.0,
            sensor_health_score=100.0,
        )
        assert assessment.cloud_cover_score == 100.0
        assert assessment.temporal_proximity_score == 0.0

    def test_is_acceptable_true(self):
        """Test is_acceptable when overall score is high."""
        assessment = DataQualityAssessment(
            overall_score=85.0,
            is_acceptable=True,
        )
        assert assessment.is_acceptable is True

    def test_is_acceptable_false(self):
        """Test is_acceptable when overall score is low."""
        assessment = DataQualityAssessment(
            overall_score=30.0,
            is_acceptable=False,
        )
        assert assessment.is_acceptable is False

    def test_details_dict(self):
        """Test details can hold arbitrary metadata."""
        details = {
            "haze_detected": True,
            "shadow_percentage": 5.2,
            "assessment_method": "automated",
        }
        assessment = DataQualityAssessment(details=details)
        assert assessment.details["haze_detected"] is True
        assert assessment.details["shadow_percentage"] == 5.2

    @pytest.mark.parametrize("score", [0.0, 25.0, 50.0, 75.0, 100.0])
    def test_overall_score_values(self, score):
        """Test overall score accepts values in 0-100 range."""
        assessment = DataQualityAssessment(overall_score=score)
        assert assessment.overall_score == score

    def test_provenance_hash_assignment(self):
        """Test provenance hash assignment."""
        h = compute_test_hash({"assessment_id": "DQA-TEST"})
        assessment = DataQualityAssessment(provenance_hash=h)
        assert len(assessment.provenance_hash) == SHA256_HEX_LENGTH


# ===========================================================================
# 4. BaselineSnapshot Tests (8 tests)
# ===========================================================================


class TestBaselineSnapshot:
    """Tests for BaselineSnapshot test dataclass."""

    def test_creation_full(self):
        """Test BaselineSnapshot creation with all fields."""
        baseline = BaselineSnapshot(
            plot_id="PLOT-BR-001",
            biome="tropical_rainforest",
            cutoff_date=EUDR_DEFORESTATION_CUTOFF,
            ndvi_mean=0.72,
            ndvi_std=0.05,
            evi_mean=0.48,
            evi_std=0.04,
            forest_percentage=95.0,
            total_area_ha=4.5,
            cloud_free_percentage=92.0,
            scenes_used=6,
            composite_method="median",
            quality_score=88.5,
        )
        assert baseline.plot_id == "PLOT-BR-001"
        assert baseline.biome == "tropical_rainforest"
        assert baseline.cutoff_date == EUDR_DEFORESTATION_CUTOFF

    def test_creation_defaults(self):
        """Test BaselineSnapshot default values."""
        baseline = BaselineSnapshot()
        assert baseline.plot_id == ""
        assert baseline.ndvi_mean == 0.0
        assert baseline.forest_percentage == 0.0
        assert baseline.quality_score == 0.0
        assert baseline.composite_method == "median"
        assert baseline.provenance_hash == ""

    def test_forest_percentage_range(self):
        """Test forest percentage covers valid range."""
        for pct in [0.0, 25.0, 50.0, 75.0, 100.0]:
            baseline = BaselineSnapshot(forest_percentage=pct)
            assert 0.0 <= baseline.forest_percentage <= 100.0

    def test_ndvi_mean_range(self):
        """Test NDVI mean is within physically valid range."""
        baseline = BaselineSnapshot(ndvi_mean=0.72)
        assert -1.0 <= baseline.ndvi_mean <= 1.0

    def test_provenance_hash(self):
        """Test provenance hash can be set and validated."""
        h = compute_test_hash({
            "plot_id": "PLOT-001",
            "ndvi_mean": 0.72,
        })
        baseline = BaselineSnapshot(provenance_hash=h)
        assert len(baseline.provenance_hash) == SHA256_HEX_LENGTH

    def test_quality_score_range(self):
        """Test quality score covers valid range."""
        for score in [0.0, 50.0, 100.0]:
            baseline = BaselineSnapshot(quality_score=score)
            assert 0.0 <= baseline.quality_score <= 100.0

    def test_established_at(self):
        """Test established_at field."""
        baseline = BaselineSnapshot(
            established_at="2026-03-01T00:00:00+00:00",
        )
        assert baseline.established_at is not None

    def test_scenes_used_positive(self):
        """Test scenes_used is a positive integer."""
        baseline = BaselineSnapshot(scenes_used=6)
        assert baseline.scenes_used > 0


# ===========================================================================
# 5. ChangeDetectionResult Tests (6 tests)
# ===========================================================================


class TestChangeDetectionResult:
    """Tests for ChangeDetectionResult test dataclass."""

    def test_creation(self):
        """Test ChangeDetectionResult creation."""
        result = ChangeDetectionResult(
            plot_id="PLOT-001",
            baseline_ndvi=0.72,
            current_ndvi=0.40,
            ndvi_delta=-0.32,
            classification="deforestation",
            confidence=0.95,
            change_area_ha=2.5,
            deforestation_detected=True,
            detection_method="ndvi_differencing",
        )
        assert result.plot_id == "PLOT-001"
        assert result.ndvi_delta == -0.32
        assert result.classification == "deforestation"

    def test_defaults(self):
        """Test ChangeDetectionResult default values."""
        result = ChangeDetectionResult()
        assert result.plot_id == ""
        assert result.classification == "no_change"
        assert result.deforestation_detected is False
        assert result.detection_method == "ndvi_differencing"

    def test_deforestation_flag(self):
        """Test deforestation_detected flag consistency."""
        result_yes = ChangeDetectionResult(
            classification="deforestation",
            deforestation_detected=True,
        )
        result_no = ChangeDetectionResult(
            classification="no_change",
            deforestation_detected=False,
        )
        assert result_yes.deforestation_detected is True
        assert result_no.deforestation_detected is False

    def test_confidence_range(self):
        """Test confidence covers 0-1 range."""
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = ChangeDetectionResult(confidence=conf)
            assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.parametrize("classification", CHANGE_CLASSIFICATIONS)
    def test_classification_values(self, classification):
        """Test all change classification values are accepted."""
        result = ChangeDetectionResult(classification=classification)
        assert result.classification == classification

    @pytest.mark.parametrize("method", DETECTION_METHODS)
    def test_detection_method_values(self, method):
        """Test all detection method values are accepted."""
        result = ChangeDetectionResult(detection_method=method)
        assert result.detection_method == method


# ===========================================================================
# 6. SatelliteAlert Tests (6 tests)
# ===========================================================================


class TestSatelliteAlert:
    """Tests for SatelliteAlert test dataclass."""

    def test_creation(self):
        """Test SatelliteAlert creation with all fields."""
        alert = SatelliteAlert(
            alert_id="ALERT-001",
            plot_id="PLOT-BR-001",
            severity="critical",
            change_type="deforestation",
            ndvi_drop=-0.35,
            confidence=0.95,
            area_affected_ha=3.2,
            detected_at="2026-03-01T10:00:00+00:00",
        )
        assert alert.alert_id == "ALERT-001"
        assert alert.severity == "critical"

    def test_defaults(self):
        """Test SatelliteAlert default values."""
        alert = SatelliteAlert()
        assert alert.alert_id == ""
        assert alert.severity == "info"
        assert alert.change_type == "no_change"
        assert alert.acknowledged is False
        assert alert.acknowledged_by is None
        assert alert.acknowledged_at is None

    def test_acknowledge_fields(self):
        """Test acknowledge fields can be set."""
        alert = SatelliteAlert(
            acknowledged=True,
            acknowledged_by="analyst@example.com",
            acknowledged_at="2026-03-02T09:00:00+00:00",
        )
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "analyst@example.com"
        assert alert.acknowledged_at is not None

    @pytest.mark.parametrize("severity", ALERT_SEVERITIES)
    def test_severity_values(self, severity):
        """Test all alert severity values are accepted."""
        alert = SatelliteAlert(severity=severity)
        assert alert.severity == severity

    def test_provenance_hash(self):
        """Test alert provenance hash."""
        h = compute_test_hash({"alert_id": "ALERT-001"})
        alert = SatelliteAlert(provenance_hash=h)
        assert len(alert.provenance_hash) == SHA256_HEX_LENGTH

    def test_ndvi_drop_negative(self):
        """Test NDVI drop is negative for deforestation alerts."""
        alert = SatelliteAlert(ndvi_drop=-0.35, severity="critical")
        assert alert.ndvi_drop < 0.0


# ===========================================================================
# 7. EvidencePackage Tests (5 tests)
# ===========================================================================


class TestEvidencePackage:
    """Tests for EvidencePackage test dataclass."""

    def test_creation(self):
        """Test EvidencePackage creation."""
        evidence = EvidencePackage(
            evidence_id="EVD-001",
            plot_id="PLOT-BR-001",
            compliance_status="compliant",
            format="json",
            baseline_snapshot={"ndvi_mean": 0.72},
            change_results=[{"classification": "no_change"}],
            alert_history=[],
            generated_at="2026-03-01T12:00:00+00:00",
        )
        assert evidence.evidence_id == "EVD-001"
        assert evidence.compliance_status == "compliant"

    def test_defaults(self):
        """Test EvidencePackage default values."""
        evidence = EvidencePackage()
        assert evidence.evidence_id == ""
        assert evidence.compliance_status == "unknown"
        assert evidence.format == "json"

    def test_compliance_values(self):
        """Test compliance status values."""
        for status in ["compliant", "non_compliant", "insufficient_data",
                        "manual_review", "unknown"]:
            evidence = EvidencePackage(compliance_status=status)
            assert evidence.compliance_status == status

    @pytest.mark.parametrize("fmt", EVIDENCE_FORMATS)
    def test_format_values(self, fmt):
        """Test all evidence format values are accepted."""
        evidence = EvidencePackage(format=fmt)
        assert evidence.format == fmt

    def test_provenance_hash(self):
        """Test evidence provenance hash."""
        h = compute_test_hash({"evidence_id": "EVD-001"})
        evidence = EvidencePackage(provenance_hash=h)
        assert len(evidence.provenance_hash) == SHA256_HEX_LENGTH


# ===========================================================================
# 8. FusionResult and CloudGapFillResult Tests (6 tests)
# ===========================================================================


class TestFusionResult:
    """Tests for FusionResult test dataclass."""

    def test_creation(self):
        """Test FusionResult creation."""
        fusion = FusionResult(
            plot_id="PLOT-001",
            sentinel2_result="no_change",
            landsat_result="no_change",
            gfw_result="no_change",
            fused_classification="no_change",
            agreement_score=1.0,
            compliance_status="compliant",
            confidence=0.95,
        )
        assert fusion.fused_classification == "no_change"
        assert fusion.agreement_score == 1.0

    def test_defaults(self):
        """Test FusionResult default values."""
        fusion = FusionResult()
        assert fusion.fused_classification == "no_change"
        assert fusion.agreement_score == 0.0
        assert fusion.compliance_status == "unknown"


class TestCloudGapFillResult:
    """Tests for CloudGapFillResult test dataclass."""

    def test_creation(self):
        """Test CloudGapFillResult creation."""
        result = CloudGapFillResult(
            scene_id="S2A_20210315_T20MQS",
            original_cloud_pct=45.0,
            filled_cloud_pct=5.0,
            fill_method="temporal_composite",
            pixels_filled=8000,
            total_pixels=10000,
            quality_score=88.0,
        )
        assert result.fill_method == "temporal_composite"
        assert result.quality_score == 88.0

    def test_defaults(self):
        """Test CloudGapFillResult default values."""
        result = CloudGapFillResult()
        assert result.fill_method == "temporal_composite"
        assert result.quality_score == 0.0

    @pytest.mark.parametrize("method", CLOUD_FILL_METHODS)
    def test_fill_method_values(self, method):
        """Test all cloud fill method values are accepted."""
        result = CloudGapFillResult(fill_method=method)
        assert result.fill_method == method

    def test_provenance_hash(self):
        """Test cloud fill provenance hash."""
        h = compute_test_hash({"scene_id": "TEST-SCENE"})
        result = CloudGapFillResult(provenance_hash=h)
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH


# ===========================================================================
# 9. Config Creation and Defaults (10 tests)
# ===========================================================================


class TestConfigCreation:
    """Test SatelliteMonitoringConfig creation and defaults."""

    def test_default_config_creation(self):
        """Test config with all defaults is valid."""
        cfg = SatelliteMonitoringConfig()
        assert cfg.cutoff_date == "2020-12-31"
        assert cfg.cloud_cover_max == 20.0
        assert cfg.ndvi_deforestation_threshold == -0.15

    def test_config_database_url_default(self):
        """Test default database URL."""
        cfg = SatelliteMonitoringConfig()
        assert "postgresql" in cfg.database_url

    def test_config_redis_url_default(self):
        """Test default Redis URL."""
        cfg = SatelliteMonitoringConfig()
        assert "redis" in cfg.redis_url

    def test_config_log_level_default(self):
        """Test default log level."""
        cfg = SatelliteMonitoringConfig()
        assert cfg.log_level == "INFO"

    def test_config_fusion_weights_default(self):
        """Test default fusion weights sum to 1.0."""
        cfg = SatelliteMonitoringConfig()
        total = cfg.sentinel2_weight + cfg.landsat_weight + cfg.gfw_weight
        assert abs(total - 1.0) < 0.001

    def test_config_cutoff_date(self):
        """Test cutoff date is EUDR standard December 31 2020."""
        cfg = SatelliteMonitoringConfig()
        assert cfg.cutoff_date == EUDR_DEFORESTATION_CUTOFF

    def test_config_custom_values(self):
        """Test config with custom values."""
        cfg = SatelliteMonitoringConfig(
            cloud_cover_max=30.0,
            ndvi_deforestation_threshold=-0.20,
            baseline_window_days=60,
            cloud_cover_absolute_max=60.0,
        )
        assert cfg.cloud_cover_max == 30.0
        assert cfg.ndvi_deforestation_threshold == -0.20
        assert cfg.baseline_window_days == 60

    def test_config_provenance_defaults(self):
        """Test provenance tracking defaults."""
        cfg = SatelliteMonitoringConfig()
        assert cfg.enable_provenance is True
        assert "GL-EUDR-SAT-003" in cfg.genesis_hash

    def test_config_timeout_defaults(self):
        """Test timeout defaults."""
        cfg = SatelliteMonitoringConfig()
        assert cfg.quick_timeout_seconds == 10.0
        assert cfg.standard_timeout_seconds == 30.0
        assert cfg.deep_timeout_seconds == 120.0

    def test_config_pool_and_rate_defaults(self):
        """Test pool size and rate limit defaults."""
        cfg = SatelliteMonitoringConfig()
        assert cfg.pool_size == 10
        assert cfg.rate_limit == 1000


# ===========================================================================
# 10. Config Validation (15 tests)
# ===========================================================================


class TestConfigValidation:
    """Test config post_init validation constraints."""

    def test_invalid_log_level(self):
        """Test invalid log level is rejected."""
        with pytest.raises(ValueError, match="log_level"):
            SatelliteMonitoringConfig(log_level="INVALID")

    def test_case_insensitive_log_level(self):
        """Test log level normalization to uppercase."""
        cfg = SatelliteMonitoringConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    @pytest.mark.parametrize("log_level", [
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
    ])
    def test_valid_log_levels(self, log_level):
        """Test all valid log levels are accepted."""
        cfg = SatelliteMonitoringConfig(log_level=log_level)
        assert cfg.log_level == log_level

    def test_invalid_baseline_window_zero(self):
        """Test baseline_window_days of 0 is rejected."""
        with pytest.raises(ValueError, match="baseline_window_days"):
            SatelliteMonitoringConfig(baseline_window_days=0)

    def test_invalid_cloud_cover_max_too_high(self):
        """Test cloud_cover_max above 100 is rejected."""
        with pytest.raises(ValueError, match="cloud_cover_max"):
            SatelliteMonitoringConfig(cloud_cover_max=101.0)

    def test_invalid_cloud_cover_ordering(self):
        """Test cloud_cover_max > cloud_cover_absolute_max is rejected."""
        with pytest.raises(ValueError, match="cloud_cover_max"):
            SatelliteMonitoringConfig(
                cloud_cover_max=60.0,
                cloud_cover_absolute_max=50.0,
            )

    def test_invalid_ndvi_deforestation_positive(self):
        """Test positive ndvi_deforestation_threshold is rejected."""
        with pytest.raises(ValueError, match="ndvi_deforestation_threshold"):
            SatelliteMonitoringConfig(ndvi_deforestation_threshold=0.5)

    def test_invalid_ndvi_threshold_ordering(self):
        """Test deforestation > degradation threshold is rejected."""
        with pytest.raises(ValueError, match="ndvi_deforestation_threshold"):
            SatelliteMonitoringConfig(
                ndvi_deforestation_threshold=-0.03,
                ndvi_degradation_threshold=-0.05,
            )

    def test_invalid_regrowth_threshold_too_high(self):
        """Test regrowth_threshold > 1.0 is rejected."""
        with pytest.raises(ValueError, match="regrowth_threshold"):
            SatelliteMonitoringConfig(regrowth_threshold=1.5)

    def test_validation_weights_sum_invalid(self):
        """Test fusion weights not summing to 1.0 is rejected."""
        with pytest.raises(ValueError, match="Fusion weights"):
            SatelliteMonitoringConfig(
                sentinel2_weight=0.50,
                landsat_weight=0.50,
                gfw_weight=0.50,
            )

    def test_validation_weights_sum_valid(self):
        """Test fusion weights summing to 1.0 is accepted."""
        cfg = SatelliteMonitoringConfig(
            sentinel2_weight=0.60,
            landsat_weight=0.25,
            gfw_weight=0.15,
        )
        total = cfg.sentinel2_weight + cfg.landsat_weight + cfg.gfw_weight
        assert abs(total - 1.0) < 0.001

    def test_invalid_timeout_ordering_quick_ge_standard(self):
        """Test quick >= standard timeout is rejected."""
        with pytest.raises(ValueError, match="quick_timeout_seconds"):
            SatelliteMonitoringConfig(
                quick_timeout_seconds=30.0,
                standard_timeout_seconds=30.0,
            )

    def test_invalid_timeout_ordering_standard_ge_deep(self):
        """Test standard >= deep timeout is rejected."""
        with pytest.raises(ValueError, match="standard_timeout_seconds"):
            SatelliteMonitoringConfig(
                standard_timeout_seconds=120.0,
                deep_timeout_seconds=120.0,
            )

    def test_invalid_empty_genesis_hash(self):
        """Test empty genesis hash is rejected."""
        with pytest.raises(ValueError, match="genesis_hash"):
            SatelliteMonitoringConfig(genesis_hash="")

    def test_invalid_pool_size_zero(self):
        """Test pool_size of 0 is rejected."""
        with pytest.raises(ValueError, match="pool_size"):
            SatelliteMonitoringConfig(pool_size=0)


# ===========================================================================
# 11. Config Singleton Pattern (8 tests)
# ===========================================================================


class TestSingletonPattern:
    """Test config singleton pattern."""

    def test_get_config_returns_instance(self):
        """Test get_config returns a config instance."""
        reset_config()
        cfg = get_config()
        assert isinstance(cfg, SatelliteMonitoringConfig)

    def test_get_config_singleton(self):
        """Test get_config returns same instance on repeated calls."""
        reset_config()
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces(self):
        """Test set_config replaces the singleton."""
        custom = SatelliteMonitoringConfig(
            cloud_cover_max=35.0,
            cloud_cover_absolute_max=70.0,
        )
        set_config(custom)
        assert get_config().cloud_cover_max == 35.0

    def test_reset_config(self):
        """Test reset_config clears the singleton."""
        custom = SatelliteMonitoringConfig(
            cloud_cover_max=40.0,
            cloud_cover_absolute_max=80.0,
        )
        set_config(custom)
        assert get_config().cloud_cover_max == 40.0
        reset_config()
        new_cfg = get_config()
        assert isinstance(new_cfg, SatelliteMonitoringConfig)

    def test_set_config_validates(self):
        """Test set_config accepts a valid config."""
        valid = SatelliteMonitoringConfig()
        set_config(valid)
        assert get_config() is valid

    def test_multiple_resets(self):
        """Test multiple resets do not cause errors."""
        reset_config()
        reset_config()
        reset_config()
        cfg = get_config()
        assert isinstance(cfg, SatelliteMonitoringConfig)

    def test_set_then_reset_then_get(self):
        """Test full lifecycle: set -> reset -> get creates fresh instance."""
        custom = SatelliteMonitoringConfig(baseline_window_days=30)
        set_config(custom)
        assert get_config().baseline_window_days == 30
        reset_config()
        fresh = get_config()
        # Fresh instance should use default or env values
        assert isinstance(fresh, SatelliteMonitoringConfig)

    def test_reset_is_idempotent(self):
        """Test reset can be called multiple times safely."""
        for _ in range(5):
            reset_config()
        cfg = get_config()
        assert cfg is not None


# ===========================================================================
# 12. Config Serialization and Redaction (8 tests)
# ===========================================================================


class TestConfigSerialization:
    """Test config serialization and credential redaction."""

    def test_to_dict_returns_dict(self):
        """Test to_dict returns a dictionary."""
        cfg = SatelliteMonitoringConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_redacts_database_url(self):
        """Test to_dict redacts database URL."""
        cfg = SatelliteMonitoringConfig(
            database_url="postgresql://user:mypassword@host:5432/db",
        )
        d = cfg.to_dict()
        assert d["database_url"] == "***"
        assert "mypassword" not in str(d)

    def test_to_dict_redacts_redis_url(self):
        """Test to_dict redacts Redis URL."""
        cfg = SatelliteMonitoringConfig(
            redis_url="redis://user:secret@host:6379/0",
        )
        d = cfg.to_dict()
        assert d["redis_url"] == "***"

    def test_to_dict_redacts_api_keys(self):
        """Test to_dict redacts all API keys and credentials."""
        cfg = SatelliteMonitoringConfig(
            sentinel2_client_id="my-client-id",
            sentinel2_client_secret="my-super-secret",
            landsat_api_key="my-landsat-key",
            gfw_api_key="my-gfw-key",
        )
        d = cfg.to_dict()
        assert d["sentinel2_client_id"] == "***"
        assert d["sentinel2_client_secret"] == "***"
        assert d["landsat_api_key"] == "***"
        assert d["gfw_api_key"] == "***"

    def test_to_dict_preserves_non_sensitive(self):
        """Test to_dict preserves non-sensitive fields."""
        cfg = SatelliteMonitoringConfig()
        d = cfg.to_dict()
        assert d["cutoff_date"] == "2020-12-31"
        assert d["cloud_cover_max"] == 20.0
        assert d["enable_provenance"] is True

    def test_to_dict_all_keys(self):
        """Test to_dict contains all expected keys."""
        cfg = SatelliteMonitoringConfig()
        d = cfg.to_dict()
        expected_keys = {
            "database_url", "redis_url", "log_level",
            "sentinel2_client_id", "sentinel2_client_secret",
            "landsat_api_key", "gfw_api_key",
            "cutoff_date", "baseline_window_days",
            "cloud_cover_max", "cloud_cover_absolute_max",
            "ndvi_deforestation_threshold", "ndvi_degradation_threshold",
            "regrowth_threshold", "min_change_area_ha",
            "sentinel2_weight", "landsat_weight", "gfw_weight",
            "monitoring_max_concurrency",
            "cache_ttl_seconds", "baseline_cache_ttl_seconds",
            "quick_timeout_seconds", "standard_timeout_seconds",
            "deep_timeout_seconds",
            "max_batch_size", "alert_confidence_threshold",
            "seasonal_adjustment_enabled", "sar_enabled",
            "enable_provenance", "genesis_hash",
            "enable_metrics", "pool_size", "rate_limit",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_repr_safe(self):
        """Test repr does not leak credentials."""
        cfg = SatelliteMonitoringConfig(
            database_url="postgresql://user:secret@host:5432/db",
            sentinel2_client_secret="ultra-secret-key",
        )
        r = repr(cfg)
        assert "secret" not in r.lower() or "***" in r
        assert "ultra-secret-key" not in r

    def test_repr_contains_class_name(self):
        """Test repr starts with class name."""
        cfg = SatelliteMonitoringConfig()
        r = repr(cfg)
        assert r.startswith("SatelliteMonitoringConfig(")


# ===========================================================================
# 13. Config Computed Properties (4 tests)
# ===========================================================================


class TestConfigProperties:
    """Test config computed properties."""

    def test_timeout_by_level(self):
        """Test timeout_by_level computed property."""
        cfg = SatelliteMonitoringConfig()
        levels = cfg.timeout_by_level
        assert levels["quick"] == 10.0
        assert levels["standard"] == 30.0
        assert levels["deep"] == 120.0

    def test_timeout_by_level_ordering(self):
        """Test timeout ordering: quick < standard < deep."""
        cfg = SatelliteMonitoringConfig()
        levels = cfg.timeout_by_level
        assert levels["quick"] < levels["standard"] < levels["deep"]

    def test_fusion_weights_property(self):
        """Test fusion_weights computed property."""
        cfg = SatelliteMonitoringConfig()
        weights = cfg.fusion_weights
        assert weights["sentinel2"] == 0.50
        assert weights["landsat"] == 0.30
        assert weights["gfw"] == 0.20

    def test_fusion_weights_sum(self):
        """Test fusion weights property sums to 1.0."""
        cfg = SatelliteMonitoringConfig()
        weights = cfg.fusion_weights
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001


# ===========================================================================
# 14. Config from Environment Variables (8 tests)
# ===========================================================================


class TestConfigFromEnv:
    """Test config creation from environment variables."""

    def test_from_env_default(self):
        """Test from_env with no env vars set."""
        cfg = SatelliteMonitoringConfig.from_env()
        assert isinstance(cfg, SatelliteMonitoringConfig)
        assert cfg.cutoff_date == "2020-12-31"

    @patch.dict(os.environ, {"GL_EUDR_SAT_LOG_LEVEL": "DEBUG"})
    def test_from_env_log_level(self):
        """Test log level override from env."""
        cfg = SatelliteMonitoringConfig.from_env()
        assert cfg.log_level == "DEBUG"

    @patch.dict(os.environ, {"GL_EUDR_SAT_CLOUD_COVER_MAX": "30.0"})
    def test_from_env_cloud_cover(self):
        """Test cloud cover override from env."""
        cfg = SatelliteMonitoringConfig.from_env()
        assert cfg.cloud_cover_max == 30.0

    @patch.dict(os.environ, {"GL_EUDR_SAT_ENABLE_PROVENANCE": "false"})
    def test_from_env_boolean_false(self):
        """Test boolean false override from env."""
        cfg = SatelliteMonitoringConfig.from_env()
        assert cfg.enable_provenance is False

    @patch.dict(os.environ, {"GL_EUDR_SAT_ENABLE_PROVENANCE": "true"})
    def test_from_env_boolean_true(self):
        """Test boolean true override from env."""
        cfg = SatelliteMonitoringConfig.from_env()
        assert cfg.enable_provenance is True

    @patch.dict(os.environ, {"GL_EUDR_SAT_BASELINE_WINDOW_DAYS": "60"})
    def test_from_env_integer(self):
        """Test integer override from env."""
        cfg = SatelliteMonitoringConfig.from_env()
        assert cfg.baseline_window_days == 60

    @patch.dict(os.environ, {"GL_EUDR_SAT_CUTOFF_DATE": "2020-12-31"})
    def test_from_env_cutoff_date(self):
        """Test cutoff date override from env."""
        cfg = SatelliteMonitoringConfig.from_env()
        assert cfg.cutoff_date == "2020-12-31"

    @patch.dict(os.environ, {"GL_EUDR_SAT_POOL_SIZE": "not-a-number"})
    def test_from_env_invalid_int_falls_back(self):
        """Test invalid integer env var falls back to default."""
        cfg = SatelliteMonitoringConfig.from_env()
        assert cfg.pool_size == SatelliteMonitoringConfig.pool_size


# ===========================================================================
# 15. JSON Roundtrip Tests (7 tests)
# ===========================================================================


class TestJsonRoundtrip:
    """Tests for JSON serialization roundtrip of model data."""

    def test_scene_metadata_roundtrip(self):
        """Test SceneMetadata fields survive JSON roundtrip."""
        scene = SceneMetadata(
            scene_id="S2A_TEST",
            source="sentinel2",
            cloud_cover_pct=12.5,
            resolution_m=10,
            quality_score=85.0,
        )
        data = {
            "scene_id": scene.scene_id,
            "source": scene.source,
            "cloud_cover_pct": scene.cloud_cover_pct,
            "resolution_m": scene.resolution_m,
            "quality_score": scene.quality_score,
        }
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["scene_id"] == "S2A_TEST"
        assert parsed["cloud_cover_pct"] == 12.5

    def test_baseline_snapshot_roundtrip(self):
        """Test BaselineSnapshot fields survive JSON roundtrip."""
        baseline = BaselineSnapshot(
            plot_id="PLOT-001",
            ndvi_mean=0.72,
            forest_percentage=95.0,
        )
        data = {
            "plot_id": baseline.plot_id,
            "ndvi_mean": baseline.ndvi_mean,
            "forest_percentage": baseline.forest_percentage,
        }
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["plot_id"] == "PLOT-001"
        assert parsed["ndvi_mean"] == 0.72

    def test_change_detection_roundtrip(self):
        """Test ChangeDetectionResult fields survive JSON roundtrip."""
        result = ChangeDetectionResult(
            plot_id="PLOT-001",
            classification="deforestation",
            confidence=0.95,
        )
        data = {
            "plot_id": result.plot_id,
            "classification": result.classification,
            "confidence": result.confidence,
        }
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["classification"] == "deforestation"

    def test_alert_roundtrip(self):
        """Test SatelliteAlert fields survive JSON roundtrip."""
        alert = SatelliteAlert(
            alert_id="ALERT-001",
            severity="critical",
            ndvi_drop=-0.35,
        )
        data = {
            "alert_id": alert.alert_id,
            "severity": alert.severity,
            "ndvi_drop": alert.ndvi_drop,
        }
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["severity"] == "critical"
        assert parsed["ndvi_drop"] == -0.35

    def test_evidence_roundtrip(self):
        """Test EvidencePackage fields survive JSON roundtrip."""
        evidence = EvidencePackage(
            evidence_id="EVD-001",
            compliance_status="compliant",
            format="json",
        )
        data = {
            "evidence_id": evidence.evidence_id,
            "compliance_status": evidence.compliance_status,
            "format": evidence.format,
        }
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["compliance_status"] == "compliant"

    def test_config_to_dict_roundtrip(self):
        """Test SatelliteMonitoringConfig to_dict survives JSON roundtrip."""
        cfg = SatelliteMonitoringConfig()
        d = cfg.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["cutoff_date"] == "2020-12-31"
        assert parsed["cloud_cover_max"] == 20.0
        assert parsed["enable_provenance"] is True

    def test_fusion_result_roundtrip(self):
        """Test FusionResult fields survive JSON roundtrip."""
        fusion = FusionResult(
            plot_id="PLOT-001",
            fused_classification="no_change",
            agreement_score=1.0,
        )
        data = {
            "plot_id": fusion.plot_id,
            "fused_classification": fusion.fused_classification,
            "agreement_score": fusion.agreement_score,
        }
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["agreement_score"] == 1.0


# ===========================================================================
# 16. Determinism Tests (5 tests)
# ===========================================================================


class TestModelDeterminism:
    """Test model and config determinism."""

    def test_provenance_hash_deterministic(self):
        """Test compute_test_hash is deterministic for same input."""
        data = {"plot_id": "PLOT-001", "ndvi_mean": 0.72}
        hashes = [compute_test_hash(data) for _ in range(10)]
        assert len(set(hashes)) == 1

    def test_provenance_hash_changes_with_data(self):
        """Test compute_test_hash changes when data changes."""
        h1 = compute_test_hash({"ndvi_mean": 0.72})
        h2 = compute_test_hash({"ndvi_mean": 0.73})
        assert h1 != h2

    def test_provenance_hash_is_sha256(self):
        """Test compute_test_hash returns valid SHA-256 hex string."""
        h = compute_test_hash({"test": True})
        assert len(h) == SHA256_HEX_LENGTH
        assert all(c in "0123456789abcdef" for c in h)

    def test_config_creation_deterministic(self):
        """Test creating config with same params gives same values."""
        configs = [
            SatelliteMonitoringConfig(cloud_cover_max=25.0)
            for _ in range(5)
        ]
        assert all(c.cloud_cover_max == 25.0 for c in configs)
        assert all(c.cutoff_date == "2020-12-31" for c in configs)

    def test_scene_metadata_creation_deterministic(self):
        """Test creating SceneMetadata with same params gives same values."""
        scenes = [
            SceneMetadata(
                scene_id="TEST",
                cloud_cover_pct=10.0,
                quality_score=85.0,
            )
            for _ in range(5)
        ]
        assert all(s.quality_score == 85.0 for s in scenes)
        assert all(s.cloud_cover_pct == 10.0 for s in scenes)
