# -*- coding: utf-8 -*-
"""
Tests for Data Models and Config - AGENT-EUDR-004 Forest Cover Analysis

Comprehensive test suite covering:
- Enumeration completeness (forest types, density classes, verdicts, methods)
- Forest type count and values
- Canopy density class boundaries (all 6 classes)
- Verdict values (4 types)
- Density method values (4 methods)
- CanopyDensityResult creation and field validation
- ForestClassificationResult creation and field validation
- HistoricalCoverRecord creation and field validation
- DeforestationFreeResult creation with all fields
- CanopyHeightEstimate creation and field validation
- FragmentationMetrics creation and field validation
- BiomassEstimate creation and carbon = agb * 0.47
- ForestCoverConfig creation with defaults
- ForestCoverConfig validation (canopy range, height positive,
  area positive, confidence range)
- Config singleton pattern (get_config, set_config, reset_config)
- Config credential redaction (to_dict, __repr__)
- Config environment variable override (GL_EUDR_FCA_ prefix)
- Provenance chain hash integrity (SHA-256)
- ProvenanceEntry creation
- Provenance chain tamper detection
- JSON roundtrip for all models
- Data quality tier boundaries (GOLD/SILVER/BRONZE/INSUFFICIENT)
- FAO forest definition constants (0.5ha, 10%, 5m)
- Determinism (same inputs -> same provenance hash)

Test count: 80+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Models and Config)
"""

import json
import os
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.forest_cover_analysis.config import (
    ForestCoverConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.forest_cover_analysis.provenance import (
    ProvenanceTracker,
    ProvenanceEntry,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
)
from tests.agents.eudr.forest_cover_analysis.conftest import (
    CanopyDensityResult,
    ForestClassificationResult,
    HistoricalCoverRecord,
    DeforestationFreeResult,
    CanopyHeightEstimate,
    FragmentationMetrics,
    BiomassEstimate,
    compute_test_hash,
    SHA256_HEX_LENGTH,
    EUDR_DEFORESTATION_CUTOFF,
    EUDR_COMMODITIES,
    ALL_BIOMES,
    FOREST_TYPES,
    CANOPY_DENSITY_CLASSES,
    DENSITY_CLASS_BOUNDARIES,
    VERDICTS,
    DENSITY_METHODS,
    DATA_QUALITY_TIERS,
    FAO_CANOPY_COVER_PCT,
    FAO_TREE_HEIGHT_M,
    FAO_MIN_AREA_HA,
)


# Locally import CARBON_FRACTION used in biomass test
CARBON_FRACTION = 0.47


# ===========================================================================
# 1. Enumeration Completeness (12 tests)
# ===========================================================================


class TestEnumerations:
    """Tests for all enumeration-like constant sets."""

    def test_all_enums_complete(self):
        """Test all enum-like constant sets are non-empty."""
        assert len(FOREST_TYPES) > 0
        assert len(CANOPY_DENSITY_CLASSES) > 0
        assert len(VERDICTS) > 0
        assert len(DENSITY_METHODS) > 0
        assert len(ALL_BIOMES) > 0
        assert len(EUDR_COMMODITIES) > 0

    def test_forest_type_count(self):
        """Test exactly 10 forest types."""
        assert len(FOREST_TYPES) == 10

    def test_forest_type_values(self):
        """Test all expected forest types are present."""
        expected = {
            "PRIMARY_TROPICAL", "SECONDARY_TROPICAL", "MANGROVE",
            "PEAT_SWAMP", "TEMPERATE_BROADLEAF", "TEMPERATE_CONIFEROUS",
            "BOREAL_CONIFEROUS", "MONTANE_CLOUD", "PLANTATION",
            "AGROFORESTRY",
        }
        assert set(FOREST_TYPES) == expected

    def test_canopy_density_class_count(self):
        """Test exactly 6 density classes."""
        assert len(CANOPY_DENSITY_CLASSES) == 6

    def test_canopy_density_class_boundaries(self):
        """Test correct boundaries for all 6 classes."""
        assert DENSITY_CLASS_BOUNDARIES["VERY_HIGH"] == 80.0
        assert DENSITY_CLASS_BOUNDARIES["HIGH"] == 60.0
        assert DENSITY_CLASS_BOUNDARIES["MODERATE"] == 40.0
        assert DENSITY_CLASS_BOUNDARIES["LOW"] == 20.0
        assert DENSITY_CLASS_BOUNDARIES["VERY_LOW"] == 10.0
        assert DENSITY_CLASS_BOUNDARIES["SPARSE"] == 0.0

    def test_verdict_values(self):
        """Test 4 verdict types."""
        expected = {
            "DEFORESTATION_FREE", "DEFORESTED", "DEGRADED", "INCONCLUSIVE",
        }
        assert set(VERDICTS) == expected
        assert len(VERDICTS) == 4

    def test_density_method_values(self):
        """Test 4 density estimation methods."""
        expected = {
            "spectral_unmixing", "ndvi_regression", "dimidiation", "sub_pixel",
        }
        assert set(DENSITY_METHODS) == expected
        assert len(DENSITY_METHODS) == 4

    def test_biome_count(self):
        """Test exactly 16 biomes."""
        assert len(ALL_BIOMES) == 16

    def test_commodity_count(self):
        """Test exactly 7 EUDR commodities."""
        assert len(EUDR_COMMODITIES) == 7

    def test_commodity_values(self):
        """Test all 7 Article 9 commodities present."""
        expected = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
        assert set(EUDR_COMMODITIES) == expected

    def test_provenance_entity_types(self):
        """Test provenance entity types are defined."""
        assert len(VALID_ENTITY_TYPES) == 10

    def test_provenance_actions(self):
        """Test provenance actions are defined."""
        assert len(VALID_ACTIONS) == 15


# ===========================================================================
# 2. Model Creation Tests (14 tests)
# ===========================================================================


class TestModelCreation:
    """Test all model dataclass creation and field validation."""

    def test_canopy_density_result_creation(self):
        """Test CanopyDensityResult with valid fields."""
        result = CanopyDensityResult(
            plot_id="PLOT-001",
            canopy_density_pct=72.5,
            density_class="HIGH",
            confidence=0.88,
            meets_fao_threshold=True,
        )
        assert result.plot_id == "PLOT-001"
        assert result.canopy_density_pct == 72.5
        assert result.density_class == "HIGH"

    def test_forest_classification_result(self):
        """Test ForestClassificationResult with valid fields."""
        result = ForestClassificationResult(
            plot_id="PLOT-001",
            forest_type="PRIMARY_TROPICAL",
            is_forest_per_eudr=True,
            confidence=0.92,
        )
        assert result.forest_type == "PRIMARY_TROPICAL"
        assert result.is_forest_per_eudr is True

    def test_historical_cover_record(self):
        """Test HistoricalCoverRecord with valid fields."""
        record = HistoricalCoverRecord(
            plot_id="PLOT-001",
            was_forest=True,
            canopy_density_pct=75.0,
            ndvi_mean=0.72,
            cutoff_date=EUDR_DEFORESTATION_CUTOFF,
        )
        assert record.was_forest is True
        assert record.cutoff_date == "2020-12-31"

    def test_deforestation_free_result(self):
        """Test DeforestationFreeResult with all fields."""
        result = DeforestationFreeResult(
            plot_id="PLOT-001",
            verdict="DEFORESTATION_FREE",
            cutoff_was_forest=True,
            current_is_forest=True,
            canopy_change_pct=-5.0,
            confidence=0.92,
            evidence_package={"before_ndvi": 0.75},
            regulatory_references=["EUDR Art. 3(a)"],
        )
        assert result.verdict == "DEFORESTATION_FREE"
        assert len(result.regulatory_references) >= 1

    def test_canopy_height_estimate(self):
        """Test CanopyHeightEstimate with valid fields."""
        estimate = CanopyHeightEstimate(
            plot_id="PLOT-001",
            height_m=25.0,
            gedi_height_m=26.5,
            fused_height_m=25.0,
            uncertainty_m=2.5,
            meets_fao_threshold=True,
        )
        assert estimate.height_m == 25.0
        assert estimate.meets_fao_threshold is True

    def test_fragmentation_metrics(self):
        """Test FragmentationMetrics with valid fields."""
        metrics = FragmentationMetrics(
            plot_id="PLOT-001",
            num_patches=3,
            core_area_pct=65.0,
            fragmentation_class="moderate",
        )
        assert metrics.num_patches == 3
        assert metrics.fragmentation_class == "moderate"

    def test_biomass_estimate(self):
        """Test BiomassEstimate creation with carbon = agb * 0.47."""
        agb = 200.0
        estimate = BiomassEstimate(
            plot_id="PLOT-001",
            agb_mg_per_ha=agb,
            carbon_stock_mg_per_ha=agb * CARBON_FRACTION,
        )
        assert estimate.agb_mg_per_ha == 200.0
        expected_carbon = 200.0 * 0.47
        assert abs(estimate.carbon_stock_mg_per_ha - expected_carbon) < 1e-9

    def test_canopy_density_result_defaults(self):
        """Test CanopyDensityResult default values."""
        result = CanopyDensityResult()
        assert result.plot_id == ""
        assert result.canopy_density_pct == 0.0
        assert result.density_class == "SPARSE"
        assert result.confidence == 0.0
        assert result.provenance_hash == ""

    def test_historical_cover_record_defaults(self):
        """Test HistoricalCoverRecord default values."""
        record = HistoricalCoverRecord()
        assert record.was_forest is True
        assert record.cutoff_date == EUDR_DEFORESTATION_CUTOFF

    def test_deforestation_free_result_defaults(self):
        """Test DeforestationFreeResult default values."""
        result = DeforestationFreeResult()
        assert result.verdict == "INCONCLUSIVE"
        assert result.confidence_min == 0.6

    def test_canopy_height_estimate_defaults(self):
        """Test CanopyHeightEstimate default values."""
        estimate = CanopyHeightEstimate()
        assert estimate.height_m == 0.0
        assert estimate.gedi_height_m is None
        assert estimate.meets_fao_threshold is False

    def test_fragmentation_metrics_defaults(self):
        """Test FragmentationMetrics default values."""
        metrics = FragmentationMetrics()
        assert metrics.num_patches == 0
        assert metrics.fragmentation_class == "intact"

    def test_biomass_estimate_defaults(self):
        """Test BiomassEstimate default values."""
        estimate = BiomassEstimate()
        assert estimate.agb_mg_per_ha == 0.0
        assert estimate.sar_saturated is False

    def test_classification_result_defaults(self):
        """Test ForestClassificationResult default values."""
        result = ForestClassificationResult()
        assert result.forest_type == "PRIMARY_TROPICAL"
        assert result.is_forest_per_eudr is True


# ===========================================================================
# 3. Config Creation and Defaults (10 tests)
# ===========================================================================


class TestConfigCreation:
    """Test ForestCoverConfig creation and defaults."""

    def test_config_creation(self):
        """Test config with all defaults is valid."""
        cfg = ForestCoverConfig()
        assert cfg.canopy_cover_threshold == 10.0
        assert cfg.tree_height_threshold == 5.0
        assert cfg.min_forest_area_ha == 0.5

    def test_config_cutoff_date_default(self):
        """Test default cutoff date is 2020-12-31."""
        cfg = ForestCoverConfig()
        assert cfg.cutoff_date == EUDR_DEFORESTATION_CUTOFF

    def test_config_database_url_default(self):
        """Test default database URL contains postgresql."""
        cfg = ForestCoverConfig()
        assert "postgresql" in cfg.database_url

    def test_config_redis_url_default(self):
        """Test default Redis URL contains redis."""
        cfg = ForestCoverConfig()
        assert "redis" in cfg.redis_url

    def test_config_log_level_default(self):
        """Test default log level is INFO."""
        cfg = ForestCoverConfig()
        assert cfg.log_level == "INFO"

    def test_config_degradation_threshold_default(self):
        """Test default degradation threshold is 30%."""
        cfg = ForestCoverConfig()
        assert cfg.degradation_threshold == 30.0

    def test_config_confidence_min_default(self):
        """Test default confidence minimum is 0.6."""
        cfg = ForestCoverConfig()
        assert cfg.confidence_min == 0.6

    def test_config_hansen_version_default(self):
        """Test default Hansen GFC version is v1.11."""
        cfg = ForestCoverConfig()
        assert cfg.hansen_gfc_version == "v1.11"

    def test_config_provenance_default(self):
        """Test default genesis hash contains GL-EUDR-FCA-004."""
        cfg = ForestCoverConfig()
        assert "GL-EUDR-FCA-004" in cfg.genesis_hash

    def test_config_fao_thresholds_property(self):
        """Test fao_thresholds computed property."""
        cfg = ForestCoverConfig()
        fao = cfg.fao_thresholds
        assert fao["canopy_cover_pct"] == 10.0
        assert fao["tree_height_m"] == 5.0
        assert fao["min_area_ha"] == 0.5


# ===========================================================================
# 4. Config Validation (12 tests)
# ===========================================================================


class TestConfigValidation:
    """Test config validation constraints."""

    def test_config_validation_canopy_range_high(self):
        """Test canopy_cover_threshold > 100 rejected."""
        with pytest.raises(ValueError, match="canopy_cover_threshold"):
            ForestCoverConfig(canopy_cover_threshold=101.0)

    def test_config_validation_canopy_range_low(self):
        """Test canopy_cover_threshold < 0 rejected."""
        with pytest.raises(ValueError, match="canopy_cover_threshold"):
            ForestCoverConfig(canopy_cover_threshold=-1.0)

    def test_config_validation_canopy_range_valid(self):
        """Test canopy_cover_threshold in [0, 100] accepted."""
        for val in [0.0, 10.0, 50.0, 100.0]:
            cfg = ForestCoverConfig(canopy_cover_threshold=val)
            assert cfg.canopy_cover_threshold == val

    def test_config_validation_height_positive(self):
        """Test tree_height_threshold must be > 0."""
        with pytest.raises(ValueError, match="tree_height_threshold"):
            ForestCoverConfig(tree_height_threshold=0.0)

    def test_config_validation_height_negative(self):
        """Test tree_height_threshold < 0 rejected."""
        with pytest.raises(ValueError, match="tree_height_threshold"):
            ForestCoverConfig(tree_height_threshold=-1.0)

    def test_config_validation_area_positive(self):
        """Test min_forest_area_ha must be > 0."""
        with pytest.raises(ValueError, match="min_forest_area_ha"):
            ForestCoverConfig(min_forest_area_ha=0.0)

    def test_config_validation_confidence_range_high(self):
        """Test confidence_min > 1.0 rejected."""
        with pytest.raises(ValueError, match="confidence_min"):
            ForestCoverConfig(confidence_min=1.5)

    def test_config_validation_confidence_range_low(self):
        """Test confidence_min < 0.0 rejected."""
        with pytest.raises(ValueError, match="confidence_min"):
            ForestCoverConfig(confidence_min=-0.1)

    def test_config_validation_confidence_valid(self):
        """Test confidence_min in [0, 1] accepted."""
        for val in [0.0, 0.5, 1.0]:
            cfg = ForestCoverConfig(confidence_min=val)
            assert cfg.confidence_min == val

    def test_config_validation_log_level_invalid(self):
        """Test invalid log level rejected."""
        with pytest.raises(ValueError, match="log_level"):
            ForestCoverConfig(log_level="INVALID")

    def test_config_validation_genesis_hash_empty(self):
        """Test empty genesis hash rejected."""
        with pytest.raises(ValueError, match="genesis_hash"):
            ForestCoverConfig(genesis_hash="")

    def test_config_validation_baseline_window_zero(self):
        """Test baseline_window_years = 0 rejected."""
        with pytest.raises(ValueError, match="baseline_window_years"):
            ForestCoverConfig(baseline_window_years=0)


# ===========================================================================
# 5. Config Singleton Pattern (8 tests)
# ===========================================================================


class TestConfigSingleton:
    """Test config singleton pattern lifecycle."""

    def test_config_singleton_get(self):
        """Test get_config returns a ForestCoverConfig instance."""
        reset_config()
        cfg = get_config()
        assert isinstance(cfg, ForestCoverConfig)

    def test_config_singleton_same_instance(self):
        """Test get_config returns same instance on repeated calls."""
        reset_config()
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_config_singleton_set(self):
        """Test set_config replaces the singleton."""
        custom = ForestCoverConfig(canopy_cover_threshold=15.0)
        set_config(custom)
        assert get_config().canopy_cover_threshold == 15.0

    def test_config_singleton_reset(self):
        """Test reset_config clears the singleton."""
        custom = ForestCoverConfig(canopy_cover_threshold=20.0)
        set_config(custom)
        assert get_config().canopy_cover_threshold == 20.0
        reset_config()
        fresh = get_config()
        assert isinstance(fresh, ForestCoverConfig)

    def test_config_singleton_lifecycle(self):
        """Test full set -> reset -> get lifecycle."""
        custom = ForestCoverConfig(degradation_threshold=25.0)
        set_config(custom)
        assert get_config().degradation_threshold == 25.0
        reset_config()
        fresh = get_config()
        assert isinstance(fresh, ForestCoverConfig)

    def test_config_singleton_multiple_resets(self):
        """Test multiple resets do not cause errors."""
        for _ in range(5):
            reset_config()
        cfg = get_config()
        assert isinstance(cfg, ForestCoverConfig)

    def test_config_singleton_idempotent_reset(self):
        """Test reset is idempotent."""
        reset_config()
        reset_config()
        cfg = get_config()
        assert cfg is not None

    def test_config_singleton_validates(self):
        """Test set_config accepts a valid config."""
        valid = ForestCoverConfig()
        set_config(valid)
        assert get_config() is valid


# ===========================================================================
# 6. Config Credential Redaction (6 tests)
# ===========================================================================


class TestConfigRedaction:
    """Test config serialization and credential redaction."""

    def test_config_credential_redaction_to_dict(self):
        """Test to_dict redacts database_url and redis_url."""
        cfg = ForestCoverConfig(
            database_url="postgresql://user:secret@host:5432/db",
            redis_url="redis://user:pass@host:6379/0",
        )
        d = cfg.to_dict()
        assert d["database_url"] == "***"
        assert d["redis_url"] == "***"

    def test_config_credential_redaction_api_keys(self):
        """Test to_dict redacts API keys."""
        cfg = ForestCoverConfig(
            gedi_api_key="my-gedi-secret",
            esa_cci_api_key="my-esa-secret",
        )
        d = cfg.to_dict()
        assert d["gedi_api_key"] == "***"
        assert d["esa_cci_api_key"] == "***"

    def test_config_credential_not_in_repr(self):
        """Test __repr__ does not leak credentials."""
        cfg = ForestCoverConfig(
            database_url="postgresql://user:secret@host:5432/db",
            gedi_api_key="top-secret-key",
        )
        r = repr(cfg)
        assert "secret" not in r.lower() or "***" in r
        assert "top-secret-key" not in r

    def test_config_repr_contains_class_name(self):
        """Test __repr__ starts with class name."""
        cfg = ForestCoverConfig()
        r = repr(cfg)
        assert r.startswith("ForestCoverConfig(")

    def test_config_to_dict_preserves_non_sensitive(self):
        """Test to_dict preserves non-sensitive fields."""
        cfg = ForestCoverConfig()
        d = cfg.to_dict()
        assert d["canopy_cover_threshold"] == 10.0
        assert d["cutoff_date"] == "2020-12-31"
        assert d["degradation_threshold"] == 30.0

    def test_config_to_dict_returns_dict(self):
        """Test to_dict returns a dictionary."""
        cfg = ForestCoverConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)


# ===========================================================================
# 7. Config Environment Variables (6 tests)
# ===========================================================================


class TestConfigEnvVars:
    """Test config creation from environment variables."""

    def test_config_env_vars_default(self):
        """Test from_env with no env vars uses defaults."""
        cfg = ForestCoverConfig.from_env()
        assert isinstance(cfg, ForestCoverConfig)
        assert cfg.canopy_cover_threshold == 10.0

    @patch.dict(os.environ, {"GL_EUDR_FCA_LOG_LEVEL": "DEBUG"})
    def test_config_env_vars_log_level(self):
        """Test log level override from env."""
        cfg = ForestCoverConfig.from_env()
        assert cfg.log_level == "DEBUG"

    @patch.dict(os.environ, {"GL_EUDR_FCA_CANOPY_COVER_THRESHOLD": "15.0"})
    def test_config_env_vars_canopy_threshold(self):
        """Test canopy threshold override from env."""
        cfg = ForestCoverConfig.from_env()
        assert cfg.canopy_cover_threshold == 15.0

    @patch.dict(os.environ, {"GL_EUDR_FCA_ENABLE_METRICS": "false"})
    def test_config_env_vars_boolean(self):
        """Test boolean override from env."""
        cfg = ForestCoverConfig.from_env()
        assert cfg.enable_metrics is False

    @patch.dict(os.environ, {"GL_EUDR_FCA_BASELINE_WINDOW_YEARS": "5"})
    def test_config_env_vars_integer(self):
        """Test integer override from env."""
        cfg = ForestCoverConfig.from_env()
        assert cfg.baseline_window_years == 5

    @patch.dict(os.environ, {"GL_EUDR_FCA_CONFIDENCE_MIN": "0.8"})
    def test_config_env_vars_float(self):
        """Test float override from env."""
        cfg = ForestCoverConfig.from_env()
        assert cfg.confidence_min == 0.8


# ===========================================================================
# 8. Provenance Chain (10 tests)
# ===========================================================================


class TestProvenanceChain:
    """Test provenance chain hash integrity."""

    def test_provenance_chain_hash(self):
        """Test SHA-256 chain integrity."""
        tracker = ProvenanceTracker()
        e1 = tracker.record("density_map", "create", "plot-001")
        e2 = tracker.record("classification", "classify", "plot-001")
        assert e2.parent_hash == e1.hash_value

    def test_provenance_entry_creation(self):
        """Test valid ProvenanceEntry creation."""
        entry = ProvenanceEntry(
            entity_type="density_map",
            entity_id="plot-001",
            action="create",
            hash_value="abc123",
            parent_hash="genesis",
            timestamp="2026-03-01T00:00:00+00:00",
        )
        assert entry.entity_type == "density_map"
        assert entry.action == "create"

    def test_provenance_chain_tamper_detection(self):
        """Test modified entry detected by verify_chain."""
        tracker = ProvenanceTracker()
        tracker.record("density_map", "create", "plot-001")
        tracker.record("classification", "classify", "plot-001")

        # Verify chain is valid
        assert tracker.verify_chain() is True

        # Tamper with an entry
        tracker._global_chain[0].hash_value = "TAMPERED"
        assert tracker.verify_chain() is False

    def test_provenance_chain_empty_valid(self):
        """Test empty chain is trivially valid."""
        tracker = ProvenanceTracker()
        assert tracker.verify_chain() is True

    def test_provenance_chain_single_entry(self):
        """Test single-entry chain is valid."""
        tracker = ProvenanceTracker()
        tracker.record("density_map", "create", "plot-001")
        assert tracker.verify_chain() is True

    def test_provenance_genesis_hash(self):
        """Test genesis hash is set correctly."""
        tracker = ProvenanceTracker()
        assert len(tracker.genesis_hash) == SHA256_HEX_LENGTH

    def test_provenance_entry_to_dict(self):
        """Test ProvenanceEntry serializes to dict."""
        entry = ProvenanceEntry(
            entity_type="verdict",
            entity_id="plot-001",
            action="verify",
            hash_value="abc",
            parent_hash="def",
            timestamp="2026-03-01T00:00:00+00:00",
            metadata={"key": "value"},
        )
        d = entry.to_dict()
        assert d["entity_type"] == "verdict"
        assert d["metadata"]["key"] == "value"

    def test_provenance_tracker_export_json(self):
        """Test export_json returns valid JSON."""
        tracker = ProvenanceTracker()
        tracker.record("density_map", "create", "plot-001")
        exported = tracker.export_json()
        parsed = json.loads(exported)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_provenance_tracker_clear(self):
        """Test clear resets the tracker."""
        tracker = ProvenanceTracker()
        tracker.record("density_map", "create", "plot-001")
        tracker.clear()
        assert tracker.entry_count == 0

    def test_provenance_tracker_get_entries(self):
        """Test get_entries filters correctly."""
        tracker = ProvenanceTracker()
        tracker.record("density_map", "create", "p1")
        tracker.record("classification", "classify", "p1")
        tracker.record("density_map", "update", "p2")
        entries = tracker.get_entries(entity_type="density_map")
        assert len(entries) == 2


# ===========================================================================
# 9. JSON Roundtrip Tests (8 tests)
# ===========================================================================


class TestJsonRoundtrip:
    """Tests for JSON serialization roundtrip of model data."""

    def test_canopy_density_roundtrip(self):
        """Test CanopyDensityResult fields survive JSON roundtrip."""
        result = CanopyDensityResult(
            plot_id="PLOT-001", canopy_density_pct=72.5, density_class="HIGH",
        )
        data = {"plot_id": result.plot_id, "density": result.canopy_density_pct}
        parsed = json.loads(json.dumps(data))
        assert parsed["density"] == 72.5

    def test_classification_roundtrip(self):
        """Test ForestClassificationResult fields survive JSON roundtrip."""
        result = ForestClassificationResult(
            plot_id="PLOT-001", forest_type="PRIMARY_TROPICAL",
        )
        data = {"plot_id": result.plot_id, "type": result.forest_type}
        parsed = json.loads(json.dumps(data))
        assert parsed["type"] == "PRIMARY_TROPICAL"

    def test_historical_roundtrip(self):
        """Test HistoricalCoverRecord fields survive JSON roundtrip."""
        record = HistoricalCoverRecord(
            plot_id="PLOT-001", was_forest=True, ndvi_mean=0.72,
        )
        data = {"plot_id": record.plot_id, "was_forest": record.was_forest}
        parsed = json.loads(json.dumps(data))
        assert parsed["was_forest"] is True

    def test_verdict_roundtrip(self):
        """Test DeforestationFreeResult fields survive JSON roundtrip."""
        result = DeforestationFreeResult(
            plot_id="PLOT-001", verdict="DEFORESTATION_FREE",
        )
        data = {"plot_id": result.plot_id, "verdict": result.verdict}
        parsed = json.loads(json.dumps(data))
        assert parsed["verdict"] == "DEFORESTATION_FREE"

    def test_height_roundtrip(self):
        """Test CanopyHeightEstimate fields survive JSON roundtrip."""
        estimate = CanopyHeightEstimate(plot_id="P001", height_m=25.0)
        data = {"height_m": estimate.height_m}
        parsed = json.loads(json.dumps(data))
        assert parsed["height_m"] == 25.0

    def test_fragmentation_roundtrip(self):
        """Test FragmentationMetrics fields survive JSON roundtrip."""
        metrics = FragmentationMetrics(plot_id="P001", num_patches=3)
        data = {"num_patches": metrics.num_patches}
        parsed = json.loads(json.dumps(data))
        assert parsed["num_patches"] == 3

    def test_biomass_roundtrip(self):
        """Test BiomassEstimate fields survive JSON roundtrip."""
        estimate = BiomassEstimate(plot_id="P001", agb_mg_per_ha=200.0)
        data = {"agb": estimate.agb_mg_per_ha}
        parsed = json.loads(json.dumps(data))
        assert parsed["agb"] == 200.0

    def test_config_roundtrip(self):
        """Test ForestCoverConfig to_dict survives JSON roundtrip."""
        cfg = ForestCoverConfig()
        d = cfg.to_dict()
        parsed = json.loads(json.dumps(d))
        assert parsed["cutoff_date"] == "2020-12-31"
        assert parsed["canopy_cover_threshold"] == 10.0


# ===========================================================================
# 10. Data Quality Tiers (5 tests)
# ===========================================================================


class TestDataQualityTiers:
    """Test data quality tier boundaries."""

    def test_data_quality_tier_gold(self):
        """Test GOLD tier boundary is 90."""
        assert DATA_QUALITY_TIERS["GOLD"] == 90.0

    def test_data_quality_tier_silver(self):
        """Test SILVER tier boundary is 70."""
        assert DATA_QUALITY_TIERS["SILVER"] == 70.0

    def test_data_quality_tier_bronze(self):
        """Test BRONZE tier boundary is 50."""
        assert DATA_QUALITY_TIERS["BRONZE"] == 50.0

    def test_data_quality_tier_insufficient(self):
        """Test INSUFFICIENT tier boundary is 0."""
        assert DATA_QUALITY_TIERS["INSUFFICIENT"] == 0.0

    @pytest.mark.parametrize("score,expected_tier", [
        (95.0, "GOLD"),
        (90.0, "GOLD"),
        (80.0, "SILVER"),
        (70.0, "SILVER"),
        (60.0, "BRONZE"),
        (50.0, "BRONZE"),
        (40.0, "INSUFFICIENT"),
        (0.0, "INSUFFICIENT"),
    ])
    def test_data_quality_tier_boundaries(self, score, expected_tier):
        """Test quality scores map to correct tiers."""
        if score >= DATA_QUALITY_TIERS["GOLD"]:
            tier = "GOLD"
        elif score >= DATA_QUALITY_TIERS["SILVER"]:
            tier = "SILVER"
        elif score >= DATA_QUALITY_TIERS["BRONZE"]:
            tier = "BRONZE"
        else:
            tier = "INSUFFICIENT"
        assert tier == expected_tier


# ===========================================================================
# 11. FAO Forest Definition Constants (3 tests)
# ===========================================================================


class TestFAOConstants:
    """Test FAO forest definition constants."""

    def test_fao_forest_definition_area(self):
        """Test FAO minimum area is 0.5 hectares."""
        assert FAO_MIN_AREA_HA == 0.5

    def test_fao_forest_definition_canopy(self):
        """Test FAO canopy cover threshold is 10%."""
        assert FAO_CANOPY_COVER_PCT == 10.0

    def test_fao_forest_definition_height(self):
        """Test FAO tree height threshold is 5 metres."""
        assert FAO_TREE_HEIGHT_M == 5.0


# ===========================================================================
# 12. Determinism Tests (5 tests)
# ===========================================================================


class TestModelDeterminism:
    """Test deterministic behaviour of models and config."""

    def test_determinism_provenance_hash(self):
        """Test compute_test_hash is deterministic for same input."""
        data = {"plot_id": "PLOT-001", "density": 72.5}
        hashes = [compute_test_hash(data) for _ in range(10)]
        assert len(set(hashes)) == 1

    def test_determinism_provenance_hash_changes(self):
        """Test compute_test_hash changes when data changes."""
        h1 = compute_test_hash({"density": 72.5})
        h2 = compute_test_hash({"density": 72.6})
        assert h1 != h2

    def test_determinism_provenance_hash_sha256(self):
        """Test compute_test_hash returns valid SHA-256 hex string."""
        h = compute_test_hash({"test": True})
        assert len(h) == SHA256_HEX_LENGTH
        assert all(c in "0123456789abcdef" for c in h)

    def test_determinism_config_creation(self):
        """Test creating config with same params gives same values."""
        configs = [
            ForestCoverConfig(canopy_cover_threshold=15.0)
            for _ in range(5)
        ]
        assert all(c.canopy_cover_threshold == 15.0 for c in configs)

    def test_determinism_model_creation(self):
        """Test creating model with same params gives same values."""
        results = [
            CanopyDensityResult(
                plot_id="P001", canopy_density_pct=72.5, density_class="HIGH",
            )
            for _ in range(5)
        ]
        assert all(r.canopy_density_pct == 72.5 for r in results)

    def test_determinism_same_inputs_same_provenance_hash(self):
        """Test same inputs to provenance tracker yield same data hash."""
        tracker = ProvenanceTracker(genesis_hash="FIXED-GENESIS")
        data = {"key": "value"}
        h1 = tracker.build_hash(data)
        h2 = tracker.build_hash(data)
        assert h1 == h2
