# -*- coding: utf-8 -*-
"""
Unit tests for Climate Hazard Connector data models module.

Tests all 12 enums, 14 SDK data models, 8 request models, constants,
Layer 1 re-exports, Pydantic validation, serialization, and edge cases.

AGENT-DATA-020: Climate Hazard Connector
Target: 85%+ coverage of greenlang.climate_hazard.models
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from greenlang.climate_hazard.models import (
    # Constants
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_PIPELINE_BATCH_SIZE,
    MAX_ASSETS_PER_NAMESPACE,
    MAX_COMPOUND_HAZARDS,
    MAX_FACTORS_PER_PROFILE,
    MAX_PROJECTIONS_PER_PAIR,
    MAX_RECORDS_PER_BATCH,
    MAX_SEARCH_RADIUS_KM,
    MAX_SOURCES_PER_NAMESPACE,
    RISK_LEVEL_BOUNDARIES,
    SUPPORTED_FRAMEWORKS,
    SUPPORTED_REPORT_FORMATS,
    SUPPORTED_SCENARIOS,
    TIME_HORIZON_RANGES,
    VERSION,
    # Layer 1 re-exports
    BoundaryResolverEngine,
    CRSTransformerEngine,
    SpatialAnalyzerEngine,
    _BRE_AVAILABLE,
    _CTE_AVAILABLE,
    _SAE_AVAILABLE,
    # Enums (12)
    AdaptiveCapacity,
    AssetType,
    DataSourceType,
    ExposureLevel,
    HazardType,
    ReportFormat,
    ReportType,
    RiskLevel,
    Scenario,
    SensitivityLevel,
    TimeHorizon,
    VulnerabilityLevel,
    # SDK models (14)
    AdaptiveCapacityProfile,
    Asset,
    ComplianceReport,
    CompoundHazard,
    ExposureResult,
    HazardDataRecord,
    HazardEvent,
    HazardSource,
    Location,
    PipelineRun,
    RiskIndex,
    ScenarioProjection,
    SensitivityProfile,
    VulnerabilityScore,
    # Request models (8)
    AssessExposureRequest,
    CalculateRiskRequest,
    GenerateReportRequest,
    IngestDataRequest,
    ProjectScenarioRequest,
    RegisterAssetRequest,
    RegisterSourceRequest,
    ScoreVulnerabilityRequest,
)


# =============================================================================
# Constants
# =============================================================================


class TestConstants:
    """Verify module-level constants are correct."""

    def test_version(self) -> None:
        assert VERSION == "1.0.0"

    def test_max_sources_per_namespace(self) -> None:
        assert MAX_SOURCES_PER_NAMESPACE == 1_000

    def test_max_records_per_batch(self) -> None:
        assert MAX_RECORDS_PER_BATCH == 100_000

    def test_max_assets_per_namespace(self) -> None:
        assert MAX_ASSETS_PER_NAMESPACE == 50_000

    def test_max_compound_hazards(self) -> None:
        assert MAX_COMPOUND_HAZARDS == 12

    def test_max_factors_per_profile(self) -> None:
        assert MAX_FACTORS_PER_PROFILE == 100

    def test_default_pipeline_batch_size(self) -> None:
        assert DEFAULT_PIPELINE_BATCH_SIZE == 1_000

    def test_default_confidence_threshold(self) -> None:
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.8

    def test_max_projections_per_pair(self) -> None:
        assert MAX_PROJECTIONS_PER_PAIR == 50

    def test_max_search_radius_km(self) -> None:
        assert MAX_SEARCH_RADIUS_KM == 500.0

    def test_risk_level_boundaries(self) -> None:
        assert RISK_LEVEL_BOUNDARIES == {
            "negligible": 20.0,
            "low": 40.0,
            "medium": 60.0,
            "high": 80.0,
            "extreme": 100.0,
        }

    def test_supported_scenarios_count(self) -> None:
        assert len(SUPPORTED_SCENARIOS) == 8

    def test_supported_report_formats(self) -> None:
        assert set(SUPPORTED_REPORT_FORMATS) == {"json", "html", "markdown", "text", "csv"}

    def test_supported_frameworks(self) -> None:
        assert set(SUPPORTED_FRAMEWORKS) == {
            "tcfd", "csrd_esrs", "sec_climate", "ifrs_s2", "ngfs"
        }

    def test_time_horizon_ranges(self) -> None:
        assert TIME_HORIZON_RANGES["baseline"] == (1995, 2014)
        assert TIME_HORIZON_RANGES["near_term"] == (2021, 2040)
        assert TIME_HORIZON_RANGES["mid_term"] == (2041, 2060)
        assert TIME_HORIZON_RANGES["long_term"] == (2061, 2080)
        assert TIME_HORIZON_RANGES["end_century"] == (2081, 2100)


# =============================================================================
# Layer 1 Re-exports
# =============================================================================


class TestLayer1ReExports:
    """Test Layer 1 re-export classes exist."""

    def test_spatial_analyzer_engine_exists(self) -> None:
        assert SpatialAnalyzerEngine is not None

    def test_boundary_resolver_engine_exists(self) -> None:
        assert BoundaryResolverEngine is not None

    def test_crs_transformer_engine_exists(self) -> None:
        assert CRSTransformerEngine is not None

    def test_availability_flags_are_bool(self) -> None:
        assert isinstance(_SAE_AVAILABLE, bool)
        assert isinstance(_BRE_AVAILABLE, bool)
        assert isinstance(_CTE_AVAILABLE, bool)


# =============================================================================
# Enums (12)
# =============================================================================


class TestHazardTypeEnum:
    """Test HazardType enumeration."""

    def test_member_count(self) -> None:
        assert len(HazardType) == 12

    def test_all_members_are_strings(self) -> None:
        for member in HazardType:
            assert isinstance(member.value, str)

    @pytest.mark.parametrize("member,value", [
        (HazardType.RIVERINE_FLOOD, "riverine_flood"),
        (HazardType.COASTAL_FLOOD, "coastal_flood"),
        (HazardType.DROUGHT, "drought"),
        (HazardType.EXTREME_HEAT, "extreme_heat"),
        (HazardType.EXTREME_COLD, "extreme_cold"),
        (HazardType.WILDFIRE, "wildfire"),
        (HazardType.TROPICAL_CYCLONE, "tropical_cyclone"),
        (HazardType.EXTREME_PRECIPITATION, "extreme_precipitation"),
        (HazardType.WATER_STRESS, "water_stress"),
        (HazardType.SEA_LEVEL_RISE, "sea_level_rise"),
        (HazardType.LANDSLIDE, "landslide"),
        (HazardType.COASTAL_EROSION, "coastal_erosion"),
    ])
    def test_member_value(self, member: HazardType, value: str) -> None:
        assert member.value == value

    def test_from_string(self) -> None:
        assert HazardType("drought") == HazardType.DROUGHT

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            HazardType("invalid_hazard")


class TestRiskLevelEnum:
    """Test RiskLevel enumeration."""

    def test_member_count(self) -> None:
        assert len(RiskLevel) == 5

    @pytest.mark.parametrize("member,value", [
        (RiskLevel.NEGLIGIBLE, "negligible"),
        (RiskLevel.LOW, "low"),
        (RiskLevel.MEDIUM, "medium"),
        (RiskLevel.HIGH, "high"),
        (RiskLevel.EXTREME, "extreme"),
    ])
    def test_member_value(self, member: RiskLevel, value: str) -> None:
        assert member.value == value


class TestScenarioEnum:
    """Test Scenario enumeration."""

    def test_member_count(self) -> None:
        assert len(Scenario) == 8

    @pytest.mark.parametrize("member,value", [
        (Scenario.SSP1_1_9, "ssp1_1_9"),
        (Scenario.SSP1_2_6, "ssp1_2_6"),
        (Scenario.SSP2_4_5, "ssp2_4_5"),
        (Scenario.SSP3_7_0, "ssp3_7_0"),
        (Scenario.SSP5_8_5, "ssp5_8_5"),
        (Scenario.RCP2_6, "rcp2_6"),
        (Scenario.RCP4_5, "rcp4_5"),
        (Scenario.RCP8_5, "rcp8_5"),
    ])
    def test_member_value(self, member: Scenario, value: str) -> None:
        assert member.value == value


class TestTimeHorizonEnum:
    """Test TimeHorizon enumeration."""

    def test_member_count(self) -> None:
        assert len(TimeHorizon) == 5

    @pytest.mark.parametrize("member,value", [
        (TimeHorizon.BASELINE, "baseline"),
        (TimeHorizon.NEAR_TERM, "near_term"),
        (TimeHorizon.MID_TERM, "mid_term"),
        (TimeHorizon.LONG_TERM, "long_term"),
        (TimeHorizon.END_CENTURY, "end_century"),
    ])
    def test_member_value(self, member: TimeHorizon, value: str) -> None:
        assert member.value == value


class TestAssetTypeEnum:
    """Test AssetType enumeration."""

    def test_member_count(self) -> None:
        assert len(AssetType) == 8

    @pytest.mark.parametrize("member,value", [
        (AssetType.FACILITY, "facility"),
        (AssetType.SUPPLY_CHAIN_NODE, "supply_chain_node"),
        (AssetType.AGRICULTURAL_PLOT, "agricultural_plot"),
        (AssetType.INFRASTRUCTURE, "infrastructure"),
        (AssetType.REAL_ESTATE, "real_estate"),
        (AssetType.NATURAL_ASSET, "natural_asset"),
        (AssetType.WATER_SOURCE, "water_source"),
        (AssetType.COASTAL_ASSET, "coastal_asset"),
    ])
    def test_member_value(self, member: AssetType, value: str) -> None:
        assert member.value == value


class TestReportTypeEnum:
    """Test ReportType enumeration."""

    def test_member_count(self) -> None:
        assert len(ReportType) == 5

    @pytest.mark.parametrize("member,value", [
        (ReportType.PHYSICAL_RISK_ASSESSMENT, "physical_risk_assessment"),
        (ReportType.SCENARIO_ANALYSIS, "scenario_analysis"),
        (ReportType.ADAPTATION_SCREENING, "adaptation_screening"),
        (ReportType.EXPOSURE_SUMMARY, "exposure_summary"),
        (ReportType.EXECUTIVE_DASHBOARD, "executive_dashboard"),
    ])
    def test_member_value(self, member: ReportType, value: str) -> None:
        assert member.value == value


class TestReportFormatEnum:
    """Test ReportFormat enumeration."""

    def test_member_count(self) -> None:
        assert len(ReportFormat) == 5

    @pytest.mark.parametrize("member,value", [
        (ReportFormat.JSON, "json"),
        (ReportFormat.HTML, "html"),
        (ReportFormat.MARKDOWN, "markdown"),
        (ReportFormat.TEXT, "text"),
        (ReportFormat.CSV, "csv"),
    ])
    def test_member_value(self, member: ReportFormat, value: str) -> None:
        assert member.value == value


class TestDataSourceTypeEnum:
    """Test DataSourceType enumeration."""

    def test_member_count(self) -> None:
        assert len(DataSourceType) == 6

    @pytest.mark.parametrize("member,value", [
        (DataSourceType.GLOBAL_DATABASE, "global_database"),
        (DataSourceType.REGIONAL_INDEX, "regional_index"),
        (DataSourceType.EVENT_CATALOG, "event_catalog"),
        (DataSourceType.SCENARIO_MODEL, "scenario_model"),
        (DataSourceType.SATELLITE, "satellite"),
        (DataSourceType.REANALYSIS, "reanalysis"),
    ])
    def test_member_value(self, member: DataSourceType, value: str) -> None:
        assert member.value == value


class TestExposureLevelEnum:
    """Test ExposureLevel enumeration."""

    def test_member_count(self) -> None:
        assert len(ExposureLevel) == 5

    @pytest.mark.parametrize("member,value", [
        (ExposureLevel.NONE, "none"),
        (ExposureLevel.LOW, "low"),
        (ExposureLevel.MODERATE, "moderate"),
        (ExposureLevel.HIGH, "high"),
        (ExposureLevel.CRITICAL, "critical"),
    ])
    def test_member_value(self, member: ExposureLevel, value: str) -> None:
        assert member.value == value


class TestSensitivityLevelEnum:
    """Test SensitivityLevel enumeration."""

    def test_member_count(self) -> None:
        assert len(SensitivityLevel) == 5

    @pytest.mark.parametrize("member,value", [
        (SensitivityLevel.VERY_LOW, "very_low"),
        (SensitivityLevel.LOW, "low"),
        (SensitivityLevel.MODERATE, "moderate"),
        (SensitivityLevel.HIGH, "high"),
        (SensitivityLevel.VERY_HIGH, "very_high"),
    ])
    def test_member_value(self, member: SensitivityLevel, value: str) -> None:
        assert member.value == value


class TestAdaptiveCapacityEnum:
    """Test AdaptiveCapacity enumeration."""

    def test_member_count(self) -> None:
        assert len(AdaptiveCapacity) == 5

    @pytest.mark.parametrize("member,value", [
        (AdaptiveCapacity.VERY_LOW, "very_low"),
        (AdaptiveCapacity.LOW, "low"),
        (AdaptiveCapacity.MODERATE, "moderate"),
        (AdaptiveCapacity.HIGH, "high"),
        (AdaptiveCapacity.VERY_HIGH, "very_high"),
    ])
    def test_member_value(self, member: AdaptiveCapacity, value: str) -> None:
        assert member.value == value


class TestVulnerabilityLevelEnum:
    """Test VulnerabilityLevel enumeration."""

    def test_member_count(self) -> None:
        assert len(VulnerabilityLevel) == 5

    @pytest.mark.parametrize("member,value", [
        (VulnerabilityLevel.NEGLIGIBLE, "negligible"),
        (VulnerabilityLevel.LOW, "low"),
        (VulnerabilityLevel.MODERATE, "moderate"),
        (VulnerabilityLevel.HIGH, "high"),
        (VulnerabilityLevel.CRITICAL, "critical"),
    ])
    def test_member_value(self, member: VulnerabilityLevel, value: str) -> None:
        assert member.value == value


# =============================================================================
# Location model
# =============================================================================


class TestLocationModel:
    """Test Location Pydantic model."""

    def test_minimal_creation(self, sample_location_data: Dict[str, Any]) -> None:
        loc = Location(**sample_location_data)
        assert loc.latitude == 51.5074
        assert loc.longitude == -0.1278

    def test_full_creation(self, sample_location_full_data: Dict[str, Any]) -> None:
        loc = Location(**sample_location_full_data)
        assert loc.name == "Paris"
        assert loc.country_code == "FR"
        assert loc.elevation_m == 35.0

    def test_default_values(self) -> None:
        loc = Location(latitude=0.0, longitude=0.0)
        assert loc.elevation_m is None
        assert loc.name == ""
        assert loc.country_code == ""

    def test_latitude_boundary_minus_90(self) -> None:
        loc = Location(latitude=-90.0, longitude=0.0)
        assert loc.latitude == -90.0

    def test_latitude_boundary_plus_90(self) -> None:
        loc = Location(latitude=90.0, longitude=0.0)
        assert loc.latitude == 90.0

    def test_longitude_boundary_minus_180(self) -> None:
        loc = Location(latitude=0.0, longitude=-180.0)
        assert loc.longitude == -180.0

    def test_longitude_boundary_plus_180(self) -> None:
        loc = Location(latitude=0.0, longitude=180.0)
        assert loc.longitude == 180.0

    def test_invalid_latitude_too_high(self) -> None:
        with pytest.raises(ValidationError):
            Location(latitude=91.0, longitude=0.0)

    def test_invalid_latitude_too_low(self) -> None:
        with pytest.raises(ValidationError):
            Location(latitude=-91.0, longitude=0.0)

    def test_invalid_longitude_too_high(self) -> None:
        with pytest.raises(ValidationError):
            Location(latitude=0.0, longitude=181.0)

    def test_invalid_longitude_too_low(self) -> None:
        with pytest.raises(ValidationError):
            Location(latitude=0.0, longitude=-181.0)

    def test_invalid_country_code_three_letters(self) -> None:
        with pytest.raises(ValidationError):
            Location(latitude=0.0, longitude=0.0, country_code="USA")

    def test_invalid_country_code_lowercase(self) -> None:
        with pytest.raises(ValidationError):
            Location(latitude=0.0, longitude=0.0, country_code="us")

    def test_empty_country_code_accepted(self) -> None:
        loc = Location(latitude=0.0, longitude=0.0, country_code="")
        assert loc.country_code == ""

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Location(latitude=0.0, longitude=0.0, extra_field="bad")

    def test_model_serialization(self, sample_location_full_data: Dict[str, Any]) -> None:
        loc = Location(**sample_location_full_data)
        d = loc.model_dump()
        assert d["latitude"] == 48.8566
        assert d["name"] == "Paris"


# =============================================================================
# HazardSource model
# =============================================================================


class TestHazardSourceModel:
    """Test HazardSource Pydantic model."""

    def test_minimal_creation(self) -> None:
        src = HazardSource(
            name="NOAA Global",
            source_type=DataSourceType.GLOBAL_DATABASE,
        )
        assert src.name == "NOAA Global"
        assert src.source_type == DataSourceType.GLOBAL_DATABASE
        assert src.source_id  # UUID generated

    def test_defaults(self) -> None:
        src = HazardSource(
            name="Test Source",
            source_type=DataSourceType.SATELLITE,
        )
        assert src.coverage == "global"
        assert src.namespace == "default"
        assert src.registered_by == "system"
        assert src.hazard_types == []
        assert src.config == {}

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            HazardSource(name="", source_type=DataSourceType.SATELLITE)

    def test_whitespace_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            HazardSource(name="   ", source_type=DataSourceType.SATELLITE)

    def test_with_hazard_types(self) -> None:
        src = HazardSource(
            name="Multi-hazard",
            source_type=DataSourceType.GLOBAL_DATABASE,
            hazard_types=[HazardType.DROUGHT, HazardType.WILDFIRE],
        )
        assert len(src.hazard_types) == 2

    def test_source_id_is_uuid(self) -> None:
        src = HazardSource(
            name="Test",
            source_type=DataSourceType.REANALYSIS,
        )
        uuid.UUID(src.source_id)  # Should not raise

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            HazardSource(
                name="Test",
                source_type=DataSourceType.SATELLITE,
                unknown_field="bad",
            )

    def test_registered_at_is_datetime(self) -> None:
        src = HazardSource(
            name="Test",
            source_type=DataSourceType.SATELLITE,
        )
        assert isinstance(src.registered_at, datetime)


# =============================================================================
# HazardDataRecord model
# =============================================================================


class TestHazardDataRecordModel:
    """Test HazardDataRecord Pydantic model."""

    def test_minimal_creation(self) -> None:
        rec = HazardDataRecord(
            source_id="src_001",
            hazard_type=HazardType.RIVERINE_FLOOD,
            location=Location(latitude=10.0, longitude=20.0),
        )
        assert rec.source_id == "src_001"
        assert rec.hazard_type == HazardType.RIVERINE_FLOOD

    def test_defaults(self) -> None:
        rec = HazardDataRecord(
            source_id="src_001",
            hazard_type=HazardType.DROUGHT,
            location=Location(latitude=0.0, longitude=0.0),
        )
        assert rec.intensity == 0.0
        assert rec.probability is None
        assert rec.frequency is None
        assert rec.scenario is None
        assert rec.time_horizon is None
        assert rec.metadata == {}

    def test_empty_source_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="source_id"):
            HazardDataRecord(
                source_id="",
                hazard_type=HazardType.DROUGHT,
                location=Location(latitude=0.0, longitude=0.0),
            )

    def test_probability_range_valid(self) -> None:
        rec = HazardDataRecord(
            source_id="s1",
            hazard_type=HazardType.WILDFIRE,
            location=Location(latitude=0.0, longitude=0.0),
            probability=0.5,
        )
        assert rec.probability == 0.5

    def test_probability_boundary_zero(self) -> None:
        rec = HazardDataRecord(
            source_id="s1",
            hazard_type=HazardType.WILDFIRE,
            location=Location(latitude=0.0, longitude=0.0),
            probability=0.0,
        )
        assert rec.probability == 0.0

    def test_probability_boundary_one(self) -> None:
        rec = HazardDataRecord(
            source_id="s1",
            hazard_type=HazardType.WILDFIRE,
            location=Location(latitude=0.0, longitude=0.0),
            probability=1.0,
        )
        assert rec.probability == 1.0

    def test_probability_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            HazardDataRecord(
                source_id="s1",
                hazard_type=HazardType.WILDFIRE,
                location=Location(latitude=0.0, longitude=0.0),
                probability=1.5,
            )

    def test_with_scenario_and_horizon(self) -> None:
        rec = HazardDataRecord(
            source_id="s1",
            hazard_type=HazardType.SEA_LEVEL_RISE,
            location=Location(latitude=0.0, longitude=0.0),
            scenario=Scenario.SSP5_8_5,
            time_horizon=TimeHorizon.END_CENTURY,
        )
        assert rec.scenario == Scenario.SSP5_8_5
        assert rec.time_horizon == TimeHorizon.END_CENTURY


# =============================================================================
# HazardEvent model
# =============================================================================


class TestHazardEventModel:
    """Test HazardEvent Pydantic model."""

    def test_minimal_creation(self) -> None:
        event = HazardEvent(
            hazard_type=HazardType.TROPICAL_CYCLONE,
            location=Location(latitude=25.0, longitude=-80.0),
            start_date=datetime(2025, 9, 1, tzinfo=timezone.utc),
        )
        assert event.hazard_type == HazardType.TROPICAL_CYCLONE
        assert event.deaths == 0
        assert event.injuries == 0

    def test_full_creation(self) -> None:
        event = HazardEvent(
            hazard_type=HazardType.RIVERINE_FLOOD,
            location=Location(latitude=30.0, longitude=90.0),
            start_date=datetime(2025, 7, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 7, 15, tzinfo=timezone.utc),
            intensity=150.0,
            intensity_unit="mm",
            affected_area_km2=5000.0,
            deaths=10,
            injuries=50,
            displaced=10000,
            economic_loss_usd=1_000_000.0,
            insured_loss_usd=500_000.0,
            source="EM-DAT",
            description="Major flood event",
        )
        assert event.deaths == 10
        assert event.economic_loss_usd == 1_000_000.0

    def test_negative_deaths_raises(self) -> None:
        with pytest.raises(ValidationError):
            HazardEvent(
                hazard_type=HazardType.WILDFIRE,
                location=Location(latitude=0.0, longitude=0.0),
                start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
                deaths=-1,
            )

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            HazardEvent(
                hazard_type=HazardType.DROUGHT,
                location=Location(latitude=0.0, longitude=0.0),
                start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
                unknown="bad",
            )


# =============================================================================
# RiskIndex model
# =============================================================================


class TestRiskIndexModel:
    """Test RiskIndex Pydantic model."""

    def test_minimal_creation(self) -> None:
        ri = RiskIndex(
            hazard_type=HazardType.DROUGHT,
            location=Location(latitude=0.0, longitude=0.0),
        )
        assert ri.risk_score == 0.0
        assert ri.risk_level == RiskLevel.NEGLIGIBLE

    def test_full_creation(self) -> None:
        ri = RiskIndex(
            hazard_type=HazardType.WILDFIRE,
            location=Location(latitude=34.0, longitude=-118.0),
            risk_score=75.0,
            risk_level=RiskLevel.HIGH,
            probability=0.8,
            intensity=0.7,
            frequency=0.6,
            duration=0.5,
            confidence=0.85,
            scenario=Scenario.SSP3_7_0,
            time_horizon=TimeHorizon.MID_TERM,
        )
        assert ri.risk_score == 75.0
        assert ri.risk_level == RiskLevel.HIGH

    def test_risk_score_boundary_zero(self) -> None:
        ri = RiskIndex(
            hazard_type=HazardType.DROUGHT,
            location=Location(latitude=0.0, longitude=0.0),
            risk_score=0.0,
        )
        assert ri.risk_score == 0.0

    def test_risk_score_boundary_100(self) -> None:
        ri = RiskIndex(
            hazard_type=HazardType.DROUGHT,
            location=Location(latitude=0.0, longitude=0.0),
            risk_score=100.0,
        )
        assert ri.risk_score == 100.0

    def test_risk_score_over_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            RiskIndex(
                hazard_type=HazardType.DROUGHT,
                location=Location(latitude=0.0, longitude=0.0),
                risk_score=101.0,
            )

    def test_risk_score_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            RiskIndex(
                hazard_type=HazardType.DROUGHT,
                location=Location(latitude=0.0, longitude=0.0),
                risk_score=-1.0,
            )

    def test_probability_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            RiskIndex(
                hazard_type=HazardType.DROUGHT,
                location=Location(latitude=0.0, longitude=0.0),
                probability=1.5,
            )

    def test_confidence_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            RiskIndex(
                hazard_type=HazardType.DROUGHT,
                location=Location(latitude=0.0, longitude=0.0),
                confidence=1.5,
            )


# =============================================================================
# ScenarioProjection model
# =============================================================================


class TestScenarioProjectionModel:
    """Test ScenarioProjection Pydantic model."""

    def test_creation(self) -> None:
        sp = ScenarioProjection(
            hazard_type=HazardType.SEA_LEVEL_RISE,
            location=Location(latitude=0.0, longitude=0.0),
            scenario=Scenario.SSP5_8_5,
            time_horizon=TimeHorizon.END_CENTURY,
            baseline_risk=50.0,
            projected_risk=80.0,
            risk_delta=30.0,
            warming_delta_c=4.5,
        )
        assert sp.projected_risk == 80.0
        assert sp.risk_delta == 30.0

    def test_defaults(self) -> None:
        sp = ScenarioProjection(
            hazard_type=HazardType.DROUGHT,
            location=Location(latitude=0.0, longitude=0.0),
            scenario=Scenario.SSP2_4_5,
            time_horizon=TimeHorizon.MID_TERM,
        )
        assert sp.baseline_risk == 0.0
        assert sp.projected_risk == 0.0
        assert sp.scaling_factor == 1.0
        assert sp.confidence == 0.0

    def test_baseline_risk_over_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioProjection(
                hazard_type=HazardType.DROUGHT,
                location=Location(latitude=0.0, longitude=0.0),
                scenario=Scenario.SSP2_4_5,
                time_horizon=TimeHorizon.MID_TERM,
                baseline_risk=101.0,
            )

    def test_projected_risk_over_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioProjection(
                hazard_type=HazardType.DROUGHT,
                location=Location(latitude=0.0, longitude=0.0),
                scenario=Scenario.SSP2_4_5,
                time_horizon=TimeHorizon.MID_TERM,
                projected_risk=101.0,
            )

    def test_confidence_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioProjection(
                hazard_type=HazardType.DROUGHT,
                location=Location(latitude=0.0, longitude=0.0),
                scenario=Scenario.SSP2_4_5,
                time_horizon=TimeHorizon.MID_TERM,
                confidence=1.5,
            )


# =============================================================================
# Asset model
# =============================================================================


class TestAssetModel:
    """Test Asset Pydantic model."""

    def test_minimal_creation(self) -> None:
        asset = Asset(
            name="Factory Alpha",
            asset_type=AssetType.FACILITY,
            location=Location(latitude=40.0, longitude=-74.0),
        )
        assert asset.name == "Factory Alpha"
        assert asset.asset_type == AssetType.FACILITY

    def test_defaults(self) -> None:
        asset = Asset(
            name="Test",
            asset_type=AssetType.INFRASTRUCTURE,
            location=Location(latitude=0.0, longitude=0.0),
        )
        assert asset.sector == ""
        assert asset.operational_importance == 0.5
        assert asset.namespace == "default"
        assert asset.tags == {}

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            Asset(
                name="",
                asset_type=AssetType.FACILITY,
                location=Location(latitude=0.0, longitude=0.0),
            )

    def test_operational_importance_boundary_zero(self) -> None:
        asset = Asset(
            name="Test",
            asset_type=AssetType.FACILITY,
            location=Location(latitude=0.0, longitude=0.0),
            operational_importance=0.0,
        )
        assert asset.operational_importance == 0.0

    def test_operational_importance_boundary_one(self) -> None:
        asset = Asset(
            name="Test",
            asset_type=AssetType.FACILITY,
            location=Location(latitude=0.0, longitude=0.0),
            operational_importance=1.0,
        )
        assert asset.operational_importance == 1.0

    def test_operational_importance_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            Asset(
                name="Test",
                asset_type=AssetType.FACILITY,
                location=Location(latitude=0.0, longitude=0.0),
                operational_importance=1.5,
            )

    def test_with_value_and_cost(self) -> None:
        asset = Asset(
            name="Expensive",
            asset_type=AssetType.REAL_ESTATE,
            location=Location(latitude=0.0, longitude=0.0),
            value_usd=10_000_000.0,
            replacement_cost_usd=12_000_000.0,
        )
        assert asset.value_usd == 10_000_000.0
        assert asset.replacement_cost_usd == 12_000_000.0


# =============================================================================
# ExposureResult model
# =============================================================================


class TestExposureResultModel:
    """Test ExposureResult Pydantic model."""

    def test_minimal_creation(self) -> None:
        er = ExposureResult(
            asset_id="a1",
            hazard_type=HazardType.RIVERINE_FLOOD,
        )
        assert er.asset_id == "a1"
        assert er.exposure_level == ExposureLevel.NONE

    def test_defaults(self) -> None:
        er = ExposureResult(
            asset_id="a1",
            hazard_type=HazardType.DROUGHT,
        )
        assert er.proximity_score == 0.0
        assert er.composite_score == 0.0
        assert er.search_radius_km == 50.0

    def test_empty_asset_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="asset_id"):
            ExposureResult(
                asset_id="",
                hazard_type=HazardType.DROUGHT,
            )

    def test_composite_score_boundary_100(self) -> None:
        er = ExposureResult(
            asset_id="a1",
            hazard_type=HazardType.DROUGHT,
            composite_score=100.0,
        )
        assert er.composite_score == 100.0

    def test_composite_score_over_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExposureResult(
                asset_id="a1",
                hazard_type=HazardType.DROUGHT,
                composite_score=101.0,
            )

    def test_proximity_score_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExposureResult(
                asset_id="a1",
                hazard_type=HazardType.DROUGHT,
                proximity_score=1.5,
            )

    def test_search_radius_max(self) -> None:
        er = ExposureResult(
            asset_id="a1",
            hazard_type=HazardType.DROUGHT,
            search_radius_km=MAX_SEARCH_RADIUS_KM,
        )
        assert er.search_radius_km == MAX_SEARCH_RADIUS_KM


# =============================================================================
# SensitivityProfile model
# =============================================================================


class TestSensitivityProfileModel:
    """Test SensitivityProfile Pydantic model."""

    def test_minimal_creation(self) -> None:
        sp = SensitivityProfile(entity_id="e1")
        assert sp.entity_id == "e1"
        assert sp.entity_type == "asset"
        assert sp.overall_sensitivity == SensitivityLevel.MODERATE

    def test_empty_entity_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="entity_id"):
            SensitivityProfile(entity_id="")

    def test_overall_score_boundary(self) -> None:
        sp = SensitivityProfile(entity_id="e1", overall_score=1.0)
        assert sp.overall_score == 1.0

    def test_overall_score_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            SensitivityProfile(entity_id="e1", overall_score=1.5)

    def test_with_factors(self) -> None:
        sp = SensitivityProfile(
            entity_id="e1",
            factors={"water_dependency": 0.8, "elevation": 0.3},
        )
        assert sp.factors["water_dependency"] == 0.8


# =============================================================================
# AdaptiveCapacityProfile model
# =============================================================================


class TestAdaptiveCapacityProfileModel:
    """Test AdaptiveCapacityProfile Pydantic model."""

    def test_minimal_creation(self) -> None:
        acp = AdaptiveCapacityProfile(entity_id="e1")
        assert acp.entity_id == "e1"
        assert acp.overall_capacity == AdaptiveCapacity.MODERATE

    def test_empty_entity_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="entity_id"):
            AdaptiveCapacityProfile(entity_id="")

    def test_sub_scores(self) -> None:
        acp = AdaptiveCapacityProfile(
            entity_id="e1",
            financial_reserves_score=0.9,
            redundancy_score=0.8,
            insurance_score=0.7,
            contingency_score=0.6,
            infrastructure_score=0.5,
        )
        assert acp.financial_reserves_score == 0.9

    def test_overall_score_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            AdaptiveCapacityProfile(entity_id="e1", overall_score=1.5)


# =============================================================================
# VulnerabilityScore model
# =============================================================================


class TestVulnerabilityScoreModel:
    """Test VulnerabilityScore Pydantic model."""

    def test_minimal_creation(self) -> None:
        vs = VulnerabilityScore(
            entity_id="e1",
            hazard_type=HazardType.WILDFIRE,
        )
        assert vs.vulnerability_score == 0.0
        assert vs.vulnerability_level == VulnerabilityLevel.NEGLIGIBLE

    def test_full_creation(self) -> None:
        vs = VulnerabilityScore(
            entity_id="e1",
            hazard_type=HazardType.DROUGHT,
            exposure_score=0.8,
            sensitivity_score=0.7,
            adaptive_capacity_score=0.3,
            vulnerability_score=75.0,
            vulnerability_level=VulnerabilityLevel.HIGH,
            confidence=0.85,
        )
        assert vs.vulnerability_score == 75.0

    def test_empty_entity_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="entity_id"):
            VulnerabilityScore(
                entity_id="",
                hazard_type=HazardType.DROUGHT,
            )

    def test_vulnerability_score_over_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            VulnerabilityScore(
                entity_id="e1",
                hazard_type=HazardType.DROUGHT,
                vulnerability_score=101.0,
            )

    def test_exposure_score_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            VulnerabilityScore(
                entity_id="e1",
                hazard_type=HazardType.DROUGHT,
                exposure_score=1.5,
            )

    def test_sensitivity_score_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            VulnerabilityScore(
                entity_id="e1",
                hazard_type=HazardType.DROUGHT,
                sensitivity_score=1.5,
            )

    def test_adaptive_capacity_score_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            VulnerabilityScore(
                entity_id="e1",
                hazard_type=HazardType.DROUGHT,
                adaptive_capacity_score=1.5,
            )

    def test_default_weights(self) -> None:
        vs = VulnerabilityScore(
            entity_id="e1",
            hazard_type=HazardType.DROUGHT,
        )
        assert vs.exposure_weight == 0.4
        assert vs.sensitivity_weight == 0.35
        assert vs.capacity_weight == 0.25


# =============================================================================
# CompoundHazard model
# =============================================================================


class TestCompoundHazardModel:
    """Test CompoundHazard Pydantic model."""

    def test_minimal_creation(self) -> None:
        ch = CompoundHazard(
            primary_hazard=HazardType.EXTREME_PRECIPITATION,
        )
        assert ch.primary_hazard == HazardType.EXTREME_PRECIPITATION
        assert ch.correlation_factor == 0.0
        assert ch.amplification_factor == 1.0

    def test_with_secondary_hazards(self) -> None:
        ch = CompoundHazard(
            primary_hazard=HazardType.EXTREME_PRECIPITATION,
            secondary_hazards=[HazardType.LANDSLIDE, HazardType.RIVERINE_FLOOD],
        )
        assert len(ch.secondary_hazards) == 2

    def test_correlation_factor_range(self) -> None:
        ch = CompoundHazard(
            primary_hazard=HazardType.DROUGHT,
            correlation_factor=-0.5,
        )
        assert ch.correlation_factor == -0.5

    def test_correlation_factor_boundary_minus_1(self) -> None:
        ch = CompoundHazard(
            primary_hazard=HazardType.DROUGHT,
            correlation_factor=-1.0,
        )
        assert ch.correlation_factor == -1.0

    def test_correlation_factor_boundary_plus_1(self) -> None:
        ch = CompoundHazard(
            primary_hazard=HazardType.DROUGHT,
            correlation_factor=1.0,
        )
        assert ch.correlation_factor == 1.0

    def test_correlation_factor_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            CompoundHazard(
                primary_hazard=HazardType.DROUGHT,
                correlation_factor=1.5,
            )

    def test_confidence_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            CompoundHazard(
                primary_hazard=HazardType.DROUGHT,
                confidence=1.5,
            )

    def test_interaction_type_default(self) -> None:
        ch = CompoundHazard(primary_hazard=HazardType.DROUGHT)
        assert ch.interaction_type == "concurrent"


# =============================================================================
# ComplianceReport model
# =============================================================================


class TestComplianceReportModel:
    """Test ComplianceReport Pydantic model."""

    def test_minimal_creation(self) -> None:
        cr = ComplianceReport()
        assert cr.report_type == ReportType.PHYSICAL_RISK_ASSESSMENT
        assert cr.format == ReportFormat.JSON
        assert cr.framework == "tcfd"

    def test_empty_framework_raises(self) -> None:
        with pytest.raises(ValidationError, match="framework"):
            ComplianceReport(framework="")

    def test_with_content(self) -> None:
        cr = ComplianceReport(
            title="Annual TCFD Report",
            content='{"risk": "high"}',
            asset_count=100,
            hazard_count=5,
            scenario_count=3,
        )
        assert cr.title == "Annual TCFD Report"
        assert cr.asset_count == 100

    def test_with_time_horizons(self) -> None:
        cr = ComplianceReport(
            time_horizons=[TimeHorizon.NEAR_TERM, TimeHorizon.MID_TERM],
        )
        assert len(cr.time_horizons) == 2

    def test_with_recommendations(self) -> None:
        cr = ComplianceReport(
            recommendations=["Install flood barriers", "Diversify supply chain"],
        )
        assert len(cr.recommendations) == 2


# =============================================================================
# PipelineRun model
# =============================================================================


class TestPipelineRunModel:
    """Test PipelineRun Pydantic model."""

    def test_minimal_creation(self) -> None:
        pr = PipelineRun()
        assert pr.status == "pending"
        assert pr.stages_total == 7
        assert pr.completed_at is None

    def test_valid_statuses(self) -> None:
        for status in ["pending", "running", "completed", "failed", "cancelled"]:
            pr = PipelineRun(status=status)
            assert pr.status == status

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            PipelineRun(status="invalid_status")

    def test_with_results(self) -> None:
        pr = PipelineRun(
            status="completed",
            stages_completed=["ingestion", "risk_calculation"],
            assets_processed=500,
            hazards_assessed=12,
            duration_ms=45000.0,
        )
        assert pr.assets_processed == 500
        assert len(pr.stages_completed) == 2

    def test_with_errors_and_warnings(self) -> None:
        pr = PipelineRun(
            status="failed",
            errors=["Connection timeout on data source"],
            warnings=["Low data quality for region X"],
        )
        assert len(pr.errors) == 1
        assert len(pr.warnings) == 1


# =============================================================================
# RegisterSourceRequest model
# =============================================================================


class TestRegisterSourceRequestModel:
    """Test RegisterSourceRequest Pydantic model."""

    def test_minimal_creation(self) -> None:
        req = RegisterSourceRequest(
            name="NOAA",
            source_type=DataSourceType.GLOBAL_DATABASE,
        )
        assert req.name == "NOAA"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            RegisterSourceRequest(
                name="",
                source_type=DataSourceType.SATELLITE,
            )

    def test_defaults(self) -> None:
        req = RegisterSourceRequest(
            name="Test",
            source_type=DataSourceType.REANALYSIS,
        )
        assert req.coverage == "global"
        assert req.namespace == "default"
        assert req.hazard_types == []


# =============================================================================
# IngestDataRequest model
# =============================================================================


class TestIngestDataRequestModel:
    """Test IngestDataRequest Pydantic model."""

    def test_minimal_creation(self) -> None:
        req = IngestDataRequest(source_id="src_001")
        assert req.source_id == "src_001"
        assert req.batch_size == DEFAULT_PIPELINE_BATCH_SIZE

    def test_empty_source_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="source_id"):
            IngestDataRequest(source_id="")

    def test_batch_size_boundary_1(self) -> None:
        req = IngestDataRequest(source_id="s1", batch_size=1)
        assert req.batch_size == 1

    def test_batch_size_boundary_max(self) -> None:
        req = IngestDataRequest(source_id="s1", batch_size=MAX_RECORDS_PER_BATCH)
        assert req.batch_size == MAX_RECORDS_PER_BATCH

    def test_batch_size_over_max_raises(self) -> None:
        with pytest.raises(ValidationError):
            IngestDataRequest(source_id="s1", batch_size=MAX_RECORDS_PER_BATCH + 1)

    def test_batch_size_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            IngestDataRequest(source_id="s1", batch_size=0)

    def test_defaults(self) -> None:
        req = IngestDataRequest(source_id="s1")
        assert req.validate_coordinates is True
        assert req.deduplicate is True
        assert req.overwrite_existing is False


# =============================================================================
# CalculateRiskRequest model
# =============================================================================


class TestCalculateRiskRequestModel:
    """Test CalculateRiskRequest Pydantic model."""

    def test_minimal_creation(self) -> None:
        req = CalculateRiskRequest(
            hazard_type=HazardType.DROUGHT,
            location=Location(latitude=0.0, longitude=0.0),
        )
        assert req.hazard_type == HazardType.DROUGHT

    def test_defaults(self) -> None:
        req = CalculateRiskRequest(
            hazard_type=HazardType.DROUGHT,
            location=Location(latitude=0.0, longitude=0.0),
        )
        assert req.scenario is None
        assert req.include_compound is False
        assert req.search_radius_km == 50.0
        assert req.methodology == "default"

    def test_search_radius_boundary_0(self) -> None:
        req = CalculateRiskRequest(
            hazard_type=HazardType.DROUGHT,
            location=Location(latitude=0.0, longitude=0.0),
            search_radius_km=0.0,
        )
        assert req.search_radius_km == 0.0

    def test_search_radius_boundary_max(self) -> None:
        req = CalculateRiskRequest(
            hazard_type=HazardType.DROUGHT,
            location=Location(latitude=0.0, longitude=0.0),
            search_radius_km=MAX_SEARCH_RADIUS_KM,
        )
        assert req.search_radius_km == MAX_SEARCH_RADIUS_KM

    def test_search_radius_over_max_raises(self) -> None:
        with pytest.raises(ValidationError):
            CalculateRiskRequest(
                hazard_type=HazardType.DROUGHT,
                location=Location(latitude=0.0, longitude=0.0),
                search_radius_km=MAX_SEARCH_RADIUS_KM + 1.0,
            )


# =============================================================================
# ProjectScenarioRequest model
# =============================================================================


class TestProjectScenarioRequestModel:
    """Test ProjectScenarioRequest Pydantic model."""

    def test_creation(self) -> None:
        req = ProjectScenarioRequest(
            hazard_type=HazardType.SEA_LEVEL_RISE,
            location=Location(latitude=0.0, longitude=0.0),
            scenario=Scenario.SSP5_8_5,
            time_horizon=TimeHorizon.END_CENTURY,
        )
        assert req.scenario == Scenario.SSP5_8_5

    def test_defaults(self) -> None:
        req = ProjectScenarioRequest(
            hazard_type=HazardType.DROUGHT,
            location=Location(latitude=0.0, longitude=0.0),
            scenario=Scenario.SSP2_4_5,
            time_horizon=TimeHorizon.MID_TERM,
        )
        assert req.baseline_source_ids == []
        assert req.warming_delta_c is None
        assert req.include_ensemble_spread is False
        assert req.methodology == "default"


# =============================================================================
# RegisterAssetRequest model
# =============================================================================


class TestRegisterAssetRequestModel:
    """Test RegisterAssetRequest Pydantic model."""

    def test_minimal_creation(self) -> None:
        req = RegisterAssetRequest(
            name="Factory A",
            asset_type=AssetType.FACILITY,
            location=Location(latitude=40.0, longitude=-74.0),
        )
        assert req.name == "Factory A"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            RegisterAssetRequest(
                name="",
                asset_type=AssetType.FACILITY,
                location=Location(latitude=0.0, longitude=0.0),
            )

    def test_operational_importance_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            RegisterAssetRequest(
                name="Test",
                asset_type=AssetType.FACILITY,
                location=Location(latitude=0.0, longitude=0.0),
                operational_importance=1.5,
            )


# =============================================================================
# AssessExposureRequest model
# =============================================================================


class TestAssessExposureRequestModel:
    """Test AssessExposureRequest Pydantic model."""

    def test_minimal_creation(self) -> None:
        req = AssessExposureRequest(
            asset_id="a1",
            hazard_type=HazardType.RIVERINE_FLOOD,
        )
        assert req.asset_id == "a1"

    def test_empty_asset_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="asset_id"):
            AssessExposureRequest(
                asset_id="",
                hazard_type=HazardType.DROUGHT,
            )

    def test_search_radius_over_max_raises(self) -> None:
        with pytest.raises(ValidationError):
            AssessExposureRequest(
                asset_id="a1",
                hazard_type=HazardType.DROUGHT,
                search_radius_km=MAX_SEARCH_RADIUS_KM + 1.0,
            )


# =============================================================================
# ScoreVulnerabilityRequest model
# =============================================================================


class TestScoreVulnerabilityRequestModel:
    """Test ScoreVulnerabilityRequest Pydantic model."""

    def test_minimal_creation(self) -> None:
        req = ScoreVulnerabilityRequest(
            entity_id="e1",
            hazard_type=HazardType.WILDFIRE,
        )
        assert req.entity_id == "e1"

    def test_empty_entity_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="entity_id"):
            ScoreVulnerabilityRequest(
                entity_id="",
                hazard_type=HazardType.DROUGHT,
            )

    def test_exposure_score_over_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScoreVulnerabilityRequest(
                entity_id="e1",
                hazard_type=HazardType.DROUGHT,
                exposure_score=1.5,
            )

    def test_defaults(self) -> None:
        req = ScoreVulnerabilityRequest(
            entity_id="e1",
            hazard_type=HazardType.DROUGHT,
        )
        assert req.exposure_weight == 0.4
        assert req.sensitivity_weight == 0.35
        assert req.capacity_weight == 0.25
        assert req.methodology == "ipcc_ar6"

    def test_with_inline_factors(self) -> None:
        req = ScoreVulnerabilityRequest(
            entity_id="e1",
            hazard_type=HazardType.DROUGHT,
            sensitivity_factors={"water_dep": 0.8},
            adaptive_capacity_indicators={"insurance": 0.6},
        )
        assert req.sensitivity_factors["water_dep"] == 0.8


# =============================================================================
# GenerateReportRequest model
# =============================================================================


class TestGenerateReportRequestModel:
    """Test GenerateReportRequest Pydantic model."""

    def test_minimal_creation(self) -> None:
        req = GenerateReportRequest()
        assert req.report_type == ReportType.PHYSICAL_RISK_ASSESSMENT
        assert req.report_format == ReportFormat.JSON

    def test_empty_framework_raises(self) -> None:
        with pytest.raises(ValidationError, match="framework"):
            GenerateReportRequest(framework="")

    def test_defaults(self) -> None:
        req = GenerateReportRequest()
        assert req.scope == "full"
        assert req.include_recommendations is True
        assert req.include_maps is False
        assert req.include_projections is True
        assert req.namespace == "default"

    def test_with_filters(self) -> None:
        req = GenerateReportRequest(
            report_type=ReportType.SCENARIO_ANALYSIS,
            report_format=ReportFormat.HTML,
            asset_ids=["a1", "a2"],
            hazard_types=[HazardType.DROUGHT, HazardType.WILDFIRE],
            scenarios=[Scenario.SSP2_4_5, Scenario.SSP5_8_5],
            time_horizons=[TimeHorizon.MID_TERM, TimeHorizon.END_CENTURY],
        )
        assert len(req.asset_ids) == 2
        assert len(req.hazard_types) == 2
        assert len(req.scenarios) == 2
        assert len(req.time_horizons) == 2


# =============================================================================
# Model serialization roundtrip
# =============================================================================


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_location_roundtrip(self) -> None:
        loc = Location(latitude=51.5, longitude=-0.1, name="London", country_code="GB")
        d = loc.model_dump()
        loc2 = Location(**d)
        assert loc2.latitude == loc.latitude
        assert loc2.name == loc.name

    def test_risk_index_roundtrip(self) -> None:
        ri = RiskIndex(
            hazard_type=HazardType.WILDFIRE,
            location=Location(latitude=34.0, longitude=-118.0),
            risk_score=75.0,
            risk_level=RiskLevel.HIGH,
        )
        d = ri.model_dump()
        ri2 = RiskIndex(**d)
        assert ri2.risk_score == 75.0
        assert ri2.risk_level == RiskLevel.HIGH

    def test_pipeline_run_roundtrip(self) -> None:
        pr = PipelineRun(
            status="completed",
            assets_processed=100,
            duration_ms=5000.0,
        )
        d = pr.model_dump()
        pr2 = PipelineRun(**d)
        assert pr2.status == "completed"
        assert pr2.assets_processed == 100

    def test_hazard_source_json(self) -> None:
        src = HazardSource(
            name="Test Source",
            source_type=DataSourceType.SATELLITE,
        )
        json_str = src.model_dump_json()
        assert "Test Source" in json_str
        assert "satellite" in json_str

    def test_compliance_report_json(self) -> None:
        cr = ComplianceReport(
            title="Q1 TCFD Report",
            framework="tcfd",
            content="Report content here",
        )
        json_str = cr.model_dump_json()
        assert "Q1 TCFD Report" in json_str


# =============================================================================
# __all__ exports
# =============================================================================


class TestModelsExports:
    """Verify __all__ export list covers key symbols."""

    def test_all_enums_exported(self) -> None:
        from greenlang.climate_hazard import models as m
        enum_names = [
            "HazardType", "RiskLevel", "Scenario", "TimeHorizon",
            "AssetType", "ReportType", "ReportFormat", "DataSourceType",
            "ExposureLevel", "SensitivityLevel", "AdaptiveCapacity",
            "VulnerabilityLevel",
        ]
        for name in enum_names:
            assert name in m.__all__, f"{name} missing from __all__"

    def test_all_sdk_models_exported(self) -> None:
        from greenlang.climate_hazard import models as m
        model_names = [
            "Location", "HazardSource", "HazardDataRecord", "HazardEvent",
            "RiskIndex", "ScenarioProjection", "Asset", "ExposureResult",
            "SensitivityProfile", "AdaptiveCapacityProfile",
            "VulnerabilityScore", "CompoundHazard",
            "ComplianceReport", "PipelineRun",
        ]
        for name in model_names:
            assert name in m.__all__, f"{name} missing from __all__"

    def test_all_request_models_exported(self) -> None:
        from greenlang.climate_hazard import models as m
        request_names = [
            "RegisterSourceRequest", "IngestDataRequest",
            "CalculateRiskRequest", "ProjectScenarioRequest",
            "RegisterAssetRequest", "AssessExposureRequest",
            "ScoreVulnerabilityRequest", "GenerateReportRequest",
        ]
        for name in request_names:
            assert name in m.__all__, f"{name} missing from __all__"
