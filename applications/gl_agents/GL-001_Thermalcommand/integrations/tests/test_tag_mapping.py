# -*- coding: utf-8 -*-
"""
Unit Tests for Tag Mapping and Governance

Tests comprehensive validation of tag mapping functionality:
- Canonical name parsing and validation
- Tag mapping entry configuration
- Unit conversion accuracy
- Tag governance and normalization
- Bad value handling strategies
- Timestamp alignment

Author: GL-BackendDeveloper
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile
import yaml
import json

from integrations.tag_mapping import (
    BadValueStrategy,
    TimestampSource,
    CanonicalTagName,
    TagMappingEntry,
    TagMappingConfig,
    UnitConverter,
    TagGovernance,
    TagMapper,
)
from integrations.opcua_schemas import (
    OPCUADataPoint,
    OPCUAQualityCode,
    TagDataType,
)


# =============================================================================
# CANONICAL TAG NAME TESTS
# =============================================================================

class TestCanonicalTagName:
    """Test canonical tag name parsing and validation."""

    def test_parse_valid_three_part_name(self):
        """Test parsing valid three-part canonical name."""
        canonical = CanonicalTagName.parse("steam.headerA.pressure")

        assert canonical.system == "steam"
        assert canonical.equipment == "headerA"
        assert canonical.measurement == "pressure"
        assert canonical.qualifier is None

    def test_parse_valid_four_part_name(self):
        """Test parsing valid four-part canonical name with qualifier."""
        canonical = CanonicalTagName.parse("boiler.B1.temperature.setpoint")

        assert canonical.system == "boiler"
        assert canonical.equipment == "B1"
        assert canonical.measurement == "temperature"
        assert canonical.qualifier == "setpoint"

    def test_parse_invalid_name_too_few_parts(self):
        """Test that parsing fails with too few parts."""
        with pytest.raises(ValueError, match="at least"):
            CanonicalTagName.parse("invalid")

        with pytest.raises(ValueError, match="at least"):
            CanonicalTagName.parse("only.two")

    def test_parse_invalid_name_starts_with_number(self):
        """Test that parsing fails when part starts with number."""
        with pytest.raises(ValueError, match="start with letter"):
            CanonicalTagName.parse("123invalid.test.value")

    def test_to_string_roundtrip(self):
        """Test to_string produces same canonical name."""
        original = "steam.headerA.pressure"
        canonical = CanonicalTagName.parse(original)
        assert canonical.to_string() == original

        original_with_qualifier = "boiler.B1.fuel.flow"
        canonical_qual = CanonicalTagName.parse(original_with_qualifier)
        assert canonical_qual.to_string() == original_with_qualifier

    def test_matches_pattern_exact(self):
        """Test pattern matching with exact match."""
        canonical = CanonicalTagName.parse("steam.headerA.pressure")
        assert canonical.matches_pattern("steam.headerA.pressure")
        assert not canonical.matches_pattern("steam.headerB.pressure")

    def test_matches_pattern_wildcard(self):
        """Test pattern matching with single wildcard."""
        canonical = CanonicalTagName.parse("steam.headerA.pressure")
        assert canonical.matches_pattern("steam.*.pressure")
        assert canonical.matches_pattern("*.headerA.pressure")
        assert canonical.matches_pattern("steam.headerA.*")

    def test_matches_pattern_double_wildcard(self):
        """Test pattern matching with double wildcard."""
        canonical = CanonicalTagName.parse("boiler.B1.fuel.flow")
        assert canonical.matches_pattern("boiler.**")
        assert canonical.matches_pattern("**.flow")


# =============================================================================
# TAG MAPPING ENTRY TESTS
# =============================================================================

class TestTagMappingEntry:
    """Test tag mapping entry configuration."""

    @pytest.fixture
    def sample_mapping(self):
        """Create sample tag mapping entry."""
        return TagMappingEntry(
            canonical_name="steam.headerA.pressure",
            vendor_tag="PLC1:Steam_HeaderA_PT001",
            node_id="ns=2;s=Steam.HeaderA.PT001",
            display_name="Steam Header A Pressure",
            description="Main steam header A pressure transmitter",
            data_type=TagDataType.DOUBLE,
            engineering_unit="bar",
            raw_low=0.0,
            raw_high=65535.0,
            eng_low=0.0,
            eng_high=30.0,
            valid_range_low=0.0,
            valid_range_high=25.0,
            equipment_id="HEADER_A",
            system_id="STEAM",
        )

    def test_mapping_creation(self, sample_mapping):
        """Test mapping entry creation."""
        assert sample_mapping.canonical_name == "steam.headerA.pressure"
        assert sample_mapping.vendor_tag == "PLC1:Steam_HeaderA_PT001"
        assert sample_mapping.engineering_unit == "bar"

    def test_apply_scaling(self, sample_mapping):
        """Test scaling from raw to engineering units."""
        # Mid-range (50% = 15 bar)
        scaled = sample_mapping.apply_scaling(32767.5)
        assert abs(scaled - 15.0) < 0.01

        # Min (0%)
        scaled_min = sample_mapping.apply_scaling(0.0)
        assert abs(scaled_min - 0.0) < 0.01

        # Max (100% = 30 bar)
        scaled_max = sample_mapping.apply_scaling(65535.0)
        assert abs(scaled_max - 30.0) < 0.01

    def test_reverse_scaling(self, sample_mapping):
        """Test reverse scaling from engineering to raw."""
        raw = sample_mapping.reverse_scaling(15.0)
        assert abs(raw - 32767.5) < 1.0

    def test_is_in_valid_range(self, sample_mapping):
        """Test valid range checking."""
        assert sample_mapping.is_in_valid_range(10.0)
        assert sample_mapping.is_in_valid_range(0.0)
        assert sample_mapping.is_in_valid_range(25.0)
        assert not sample_mapping.is_in_valid_range(-1.0)
        assert not sample_mapping.is_in_valid_range(26.0)

    def test_default_bad_value_strategy(self, sample_mapping):
        """Test default bad value strategy."""
        assert sample_mapping.bad_value_strategy == BadValueStrategy.SUBSTITUTE_LAST_GOOD

    def test_is_deprecated(self):
        """Test deprecation detection."""
        # Not deprecated
        mapping = TagMappingEntry(
            canonical_name="test.tag.value",
            vendor_tag="TEST",
            node_id="ns=2;s=Test",
            display_name="Test",
            data_type=TagDataType.DOUBLE,
            engineering_unit="percent",
        )
        assert not mapping.is_deprecated()

        # Deprecated
        mapping_deprecated = TagMappingEntry(
            canonical_name="test.tag.value",
            vendor_tag="TEST",
            node_id="ns=2;s=Test",
            display_name="Test",
            data_type=TagDataType.DOUBLE,
            engineering_unit="percent",
            deprecated_date=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert mapping_deprecated.is_deprecated()


# =============================================================================
# UNIT CONVERTER TESTS
# =============================================================================

class TestUnitConverter:
    """Test engineering unit conversion accuracy."""

    def test_temperature_celsius_to_fahrenheit(self):
        """Test Celsius to Fahrenheit conversion."""
        result = UnitConverter.convert_temperature(100, "celsius", "fahrenheit")
        assert abs(float(result) - 212.0) < 0.01

        result = UnitConverter.convert_temperature(0, "celsius", "fahrenheit")
        assert abs(float(result) - 32.0) < 0.01

    def test_temperature_fahrenheit_to_celsius(self):
        """Test Fahrenheit to Celsius conversion."""
        result = UnitConverter.convert_temperature(212, "fahrenheit", "celsius")
        assert abs(float(result) - 100.0) < 0.01

    def test_temperature_celsius_to_kelvin(self):
        """Test Celsius to Kelvin conversion."""
        result = UnitConverter.convert_temperature(0, "celsius", "kelvin")
        assert abs(float(result) - 273.15) < 0.01

    def test_temperature_same_unit(self):
        """Test same unit returns original value."""
        result = UnitConverter.convert_temperature(100, "celsius", "celsius")
        assert float(result) == 100.0

    def test_pressure_bar_to_psi(self):
        """Test bar to PSI conversion."""
        result = UnitConverter.convert_pressure(1, "bar", "psi")
        assert abs(float(result) - 14.5038) < 0.01

    def test_pressure_psi_to_bar(self):
        """Test PSI to bar conversion."""
        result = UnitConverter.convert_pressure(14.5038, "psi", "bar")
        assert abs(float(result) - 1.0) < 0.01

    def test_pressure_bar_to_pascal(self):
        """Test bar to Pascal conversion."""
        result = UnitConverter.convert_pressure(1, "bar", "pascal")
        assert abs(float(result) - 100000.0) < 1.0

    def test_flow_kg_per_s_to_kg_per_h(self):
        """Test kg/s to kg/h conversion."""
        result = UnitConverter.convert_flow(1, "kg_per_s", "kg_per_h")
        assert abs(float(result) - 3600.0) < 0.1

    def test_flow_t_per_h_to_kg_per_s(self):
        """Test t/h to kg/s conversion."""
        result = UnitConverter.convert_flow(3.6, "t_per_h", "kg_per_s")
        assert abs(float(result) - 1.0) < 0.01

    def test_power_kw_to_mw(self):
        """Test kW to MW conversion."""
        result = UnitConverter.convert_power(1000, "kw", "mw")
        assert abs(float(result) - 1.0) < 0.001

    def test_power_btu_per_h_to_kw(self):
        """Test BTU/h to kW conversion."""
        result = UnitConverter.convert_power(3412.14, "btu_per_h", "kw")
        assert abs(float(result) - 1.0) < 0.01

    def test_energy_kwh_to_mj(self):
        """Test kWh to MJ conversion."""
        result = UnitConverter.convert_energy(1, "kwh", "mj")
        assert abs(float(result) - 3.6) < 0.01

    def test_unknown_unit_raises_error(self):
        """Test that unknown units raise ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            UnitConverter.convert_temperature(100, "invalid_unit", "celsius")

        with pytest.raises(ValueError, match="Unknown"):
            UnitConverter.convert_pressure(1, "bar", "invalid_unit")


# =============================================================================
# TAG MAPPING CONFIG TESTS
# =============================================================================

class TestTagMappingConfig:
    """Test tag mapping configuration management."""

    @pytest.fixture
    def sample_config(self):
        """Create sample tag mapping configuration."""
        return TagMappingConfig(
            config_id="config-001",
            config_name="Plant 1 Tag Mapping",
            version="1.0.0",
            site_id="PLANT1",
            site_name="Manufacturing Plant 1",
            mappings=[
                TagMappingEntry(
                    canonical_name="steam.headerA.pressure",
                    vendor_tag="PLC1:PT001",
                    node_id="ns=2;s=PT001",
                    display_name="Pressure 1",
                    data_type=TagDataType.DOUBLE,
                    engineering_unit="bar",
                ),
                TagMappingEntry(
                    canonical_name="steam.headerA.temperature",
                    vendor_tag="PLC1:TT001",
                    node_id="ns=2;s=TT001",
                    display_name="Temperature 1",
                    data_type=TagDataType.DOUBLE,
                    engineering_unit="celsius",
                ),
                TagMappingEntry(
                    canonical_name="boiler.B1.fuel_flow",
                    vendor_tag="PLC1:FT001",
                    node_id="ns=2;s=FT001",
                    display_name="Fuel Flow 1",
                    data_type=TagDataType.DOUBLE,
                    engineering_unit="kg_per_h",
                ),
            ],
        )

    def test_config_creation(self, sample_config):
        """Test configuration creation."""
        assert sample_config.config_id == "config-001"
        assert len(sample_config.mappings) == 3

    def test_get_mapping(self, sample_config):
        """Test get mapping by canonical name."""
        mapping = sample_config.get_mapping("steam.headerA.pressure")
        assert mapping is not None
        assert mapping.vendor_tag == "PLC1:PT001"

        assert sample_config.get_mapping("nonexistent.tag") is None

    def test_get_mapping_by_vendor_tag(self, sample_config):
        """Test get mapping by vendor tag."""
        mapping = sample_config.get_mapping_by_vendor_tag("PLC1:TT001")
        assert mapping is not None
        assert mapping.canonical_name == "steam.headerA.temperature"

    def test_get_mappings_by_pattern(self, sample_config):
        """Test get mappings by pattern."""
        steam_mappings = sample_config.get_mappings_by_pattern("steam.**")
        assert len(steam_mappings) == 2

        pressure_mappings = sample_config.get_mappings_by_pattern("*.*.pressure")
        assert len(pressure_mappings) == 1

    def test_save_and_load_yaml(self, sample_config, tmp_path):
        """Test saving and loading YAML configuration."""
        yaml_path = tmp_path / "config.yaml"
        sample_config.save_to_yaml(yaml_path)

        loaded_config = TagMappingConfig.load_from_yaml(yaml_path)
        assert loaded_config.config_id == sample_config.config_id
        assert len(loaded_config.mappings) == len(sample_config.mappings)

    def test_save_and_load_json(self, sample_config, tmp_path):
        """Test saving and loading JSON configuration."""
        json_path = tmp_path / "config.json"
        sample_config.save_to_json(json_path)

        loaded_config = TagMappingConfig.load_from_json(json_path)
        assert loaded_config.config_id == sample_config.config_id
        assert len(loaded_config.mappings) == len(sample_config.mappings)

    def test_checksum_verification(self, sample_config):
        """Test configuration checksum verification."""
        # Calculate checksum
        sample_config.checksum = TagMappingConfig._calculate_checksum(
            sample_config.dict(exclude={"checksum"})
        )

        assert sample_config.verify_checksum()

        # Tamper with data
        sample_config.config_name = "Tampered"
        assert not sample_config.verify_checksum()


# =============================================================================
# TAG GOVERNANCE TESTS
# =============================================================================

class TestTagGovernance:
    """Test tag governance and data normalization."""

    @pytest.fixture
    def governance_config(self):
        """Create governance configuration."""
        return TagMappingConfig(
            config_id="gov-001",
            config_name="Governance Config",
            version="1.0.0",
            site_id="TEST",
            site_name="Test Site",
            mappings=[
                TagMappingEntry(
                    canonical_name="steam.headerA.pressure",
                    vendor_tag="PT001",
                    node_id="ns=2;s=PT001",
                    display_name="Pressure",
                    data_type=TagDataType.DOUBLE,
                    engineering_unit="bar",
                    raw_low=0.0,
                    raw_high=65535.0,
                    eng_low=0.0,
                    eng_high=30.0,
                    valid_range_low=0.0,
                    valid_range_high=25.0,
                    bad_value_strategy=BadValueStrategy.SUBSTITUTE_LAST_GOOD,
                    max_bad_duration_s=60,
                    max_timestamp_drift_s=5.0,
                ),
            ],
        )

    @pytest.fixture
    def governance(self, governance_config):
        """Create governance instance."""
        return TagGovernance(governance_config)

    def test_validate_canonical_name(self, governance):
        """Test canonical name validation."""
        is_valid, error = governance.validate_canonical_name("steam.headerA.pressure")
        assert is_valid
        assert error is None

        is_valid, error = governance.validate_canonical_name("invalid")
        assert not is_valid
        assert error is not None

    def test_normalize_data_point_with_scaling(self, governance):
        """Test data point normalization with scaling."""
        now = datetime.now(timezone.utc)
        data_point = OPCUADataPoint(
            tag_id="steam_headerA_pressure",
            node_id="ns=2;s=PT001",
            canonical_name="steam.headerA.pressure",
            value=32767.5,  # Mid-range raw value
            data_type=TagDataType.DOUBLE,
            source_timestamp=now,
            server_timestamp=now,
            quality_code=OPCUAQualityCode.GOOD,
        )

        normalized = governance.normalize_data_point(data_point)

        # Should be scaled to ~15 bar
        assert normalized.scaled_value is not None
        assert abs(normalized.scaled_value - 15.0) < 0.1
        assert normalized.engineering_unit == "bar"
        assert normalized.provenance_hash is not None

    def test_validate_value_range(self, governance):
        """Test value range validation."""
        is_valid, error = governance.validate_value_range(
            "steam.headerA.pressure", 10.0
        )
        assert is_valid

        is_valid, error = governance.validate_value_range(
            "steam.headerA.pressure", 30.0  # Above valid range
        )
        assert not is_valid
        assert "outside valid range" in error


# =============================================================================
# TAG MAPPER TESTS
# =============================================================================

class TestTagMapper:
    """Test main tag mapper interface."""

    @pytest.fixture
    def mapper_config(self):
        """Create mapper configuration."""
        return TagMappingConfig(
            config_id="mapper-001",
            config_name="Mapper Config",
            version="1.0.0",
            site_id="TEST",
            site_name="Test Site",
            mappings=[
                TagMappingEntry(
                    canonical_name="steam.headerA.pressure",
                    vendor_tag="PLC1:PT001",
                    node_id="ns=2;s=PT001",
                    display_name="Pressure",
                    data_type=TagDataType.DOUBLE,
                    engineering_unit="bar",
                    equipment_id="HEADER_A",
                    system_id="STEAM",
                ),
                TagMappingEntry(
                    canonical_name="steam.headerA.temperature",
                    vendor_tag="PLC1:TT001",
                    node_id="ns=2;s=TT001",
                    display_name="Temperature",
                    data_type=TagDataType.DOUBLE,
                    engineering_unit="celsius",
                    equipment_id="HEADER_A",
                    system_id="STEAM",
                ),
            ],
        )

    @pytest.fixture
    def mapper(self, mapper_config):
        """Create tag mapper instance."""
        mapper = TagMapper()
        mapper.set_config(mapper_config)
        return mapper

    def test_get_canonical_name(self, mapper):
        """Test vendor tag to canonical name lookup."""
        canonical = mapper.get_canonical_name("PLC1:PT001")
        assert canonical == "steam.headerA.pressure"

        assert mapper.get_canonical_name("UNKNOWN") is None

    def test_get_vendor_tag(self, mapper):
        """Test canonical name to vendor tag lookup."""
        vendor = mapper.get_vendor_tag("steam.headerA.pressure")
        assert vendor == "PLC1:PT001"

    def test_convert_units(self, mapper):
        """Test unit conversion through mapper."""
        converted = mapper.convert_units(100, "celsius", "fahrenheit", "temperature")
        assert abs(converted - 212.0) < 0.01

        converted = mapper.convert_units(1, "bar", "psi", "pressure")
        assert abs(converted - 14.5) < 0.1

    def test_validate_tag_value(self, mapper):
        """Test tag value validation."""
        is_valid, error = mapper.validate_tag_value("steam.headerA.pressure", 10.0)
        assert is_valid

    def test_get_tags_by_system(self, mapper):
        """Test getting tags by system."""
        steam_tags = mapper.get_tags_by_system("steam")
        assert len(steam_tags) == 2
        assert "steam.headerA.pressure" in steam_tags
        assert "steam.headerA.temperature" in steam_tags

    def test_create_tag_config(self, mapper):
        """Test creating OPCUATagConfig from mapping."""
        tag_config = mapper.create_tag_config("steam.headerA.pressure")

        assert tag_config is not None
        assert tag_config.metadata.canonical_name == "steam.headerA.pressure"
        assert tag_config.metadata.node_id == "ns=2;s=PT001"

    def test_load_config_from_yaml(self, mapper_config, tmp_path):
        """Test loading configuration from YAML file."""
        yaml_path = tmp_path / "config.yaml"
        mapper_config.save_to_yaml(yaml_path)

        mapper = TagMapper()
        mapper.load_config(yaml_path)

        assert mapper.config is not None
        assert len(mapper.config.mappings) == 2

    def test_version_history_tracking(self, mapper_config):
        """Test version history tracking."""
        mapper = TagMapper()

        # Set initial config
        mapper.set_config(mapper_config)
        assert len(mapper._version_history) == 1

        # Update config
        mapper_config.version = "2.0.0"
        mapper.set_config(mapper_config)
        assert len(mapper._version_history) == 2


# =============================================================================
# BAD VALUE HANDLING TESTS
# =============================================================================

class TestBadValueHandling:
    """Test bad value handling strategies."""

    @pytest.fixture
    def governance_with_strategies(self):
        """Create governance with different bad value strategies."""
        config = TagMappingConfig(
            config_id="bad-value-001",
            config_name="Bad Value Test",
            version="1.0.0",
            site_id="TEST",
            site_name="Test Site",
            mappings=[
                TagMappingEntry(
                    canonical_name="test.reject.value",
                    vendor_tag="TEST1",
                    node_id="ns=2;s=TEST1",
                    display_name="Reject Test",
                    data_type=TagDataType.DOUBLE,
                    engineering_unit="bar",
                    bad_value_strategy=BadValueStrategy.REJECT,
                ),
                TagMappingEntry(
                    canonical_name="test.substitute.value",
                    vendor_tag="TEST2",
                    node_id="ns=2;s=TEST2",
                    display_name="Substitute Test",
                    data_type=TagDataType.DOUBLE,
                    engineering_unit="bar",
                    bad_value_strategy=BadValueStrategy.SUBSTITUTE_DEFAULT,
                    default_value=10.0,
                ),
            ],
        )
        return TagGovernance(config)

    def test_reject_strategy_keeps_bad_value(self, governance_with_strategies):
        """Test that REJECT strategy keeps bad quality marker."""
        now = datetime.now(timezone.utc)
        data_point = OPCUADataPoint(
            tag_id="test_reject_value",
            node_id="ns=2;s=TEST1",
            canonical_name="test.reject.value",
            value=5.0,
            data_type=TagDataType.DOUBLE,
            source_timestamp=now,
            server_timestamp=now,
            quality_code=OPCUAQualityCode.BAD,
        )

        normalized = governance_with_strategies.normalize_data_point(data_point)

        # Quality should still be bad
        assert normalized.quality_code.is_bad()

    def test_substitute_default_strategy(self, governance_with_strategies):
        """Test that SUBSTITUTE_DEFAULT strategy uses default value."""
        now = datetime.now(timezone.utc)
        data_point = OPCUADataPoint(
            tag_id="test_substitute_value",
            node_id="ns=2;s=TEST2",
            canonical_name="test.substitute.value",
            value=5.0,
            data_type=TagDataType.DOUBLE,
            source_timestamp=now,
            server_timestamp=now,
            quality_code=OPCUAQualityCode.BAD,
        )

        normalized = governance_with_strategies.normalize_data_point(data_point)

        # Value should be substituted with default
        assert normalized.value == 10.0
        assert normalized.quality_code == OPCUAQualityCode.UNCERTAIN_SUB_NORMAL
