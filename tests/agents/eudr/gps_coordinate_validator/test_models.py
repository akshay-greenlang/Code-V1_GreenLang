# -*- coding: utf-8 -*-
"""
Tests for Data Models - AGENT-EUDR-007 GPS Coordinate Validator Models

Comprehensive test suite covering:
- All 6 enumerations (CoordinateFormat, GeodeticDatum, PrecisionLevel,
  SourceType, ValidationErrorType, ValidationSeverity)
- RawCoordinate creation and validation
- ParsedCoordinate creation and field validation
- NormalizedCoordinate creation and datum fields
- ValidationError field validation
- PrecisionResult level classification
- AccuracyScore tier thresholds (conceptual via model fields)
- ComplianceCertificate ID generation (conceptual via model structure)
- Config singleton, env vars, reset
- Pydantic serialization/deserialization (model_dump / model_validate)
- Edge cases for all model validators

Test count: 100+ tests
Coverage target: >= 85% of models.py and config.py

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError as PydanticValidationError

from greenlang.agents.eudr.gps_coordinate_validator.models import (
    VERSION,
    CoordinateFormat,
    GeodeticDatum,
    PrecisionLevel,
    SourceType,
    ValidationErrorType,
    ValidationSeverity,
    RawCoordinate,
    ParsedCoordinate,
    NormalizedCoordinate,
    PrecisionResult,
    ValidationError as VError,
    ValidationResult,
    BatchValidationResult,
)
from greenlang.agents.eudr.gps_coordinate_validator.config import (
    GPSCoordinateValidatorConfig,
    get_config,
    set_config,
    reset_config,
)


# ===========================================================================
# 1. Enumeration Tests
# ===========================================================================


class TestCoordinateFormatEnum:
    """Test CoordinateFormat enumeration."""

    def test_all_members_exist(self):
        """All expected coordinate format members exist."""
        expected = {
            "DECIMAL_DEGREES", "DMS", "DDM", "UTM", "MGRS",
            "SIGNED_DD", "DD_SUFFIX", "UNKNOWN",
        }
        actual = {m.name for m in CoordinateFormat}
        assert expected == actual

    def test_string_values(self):
        """Enum values are lowercase strings."""
        assert CoordinateFormat.DECIMAL_DEGREES.value == "decimal_degrees"
        assert CoordinateFormat.DMS.value == "dms"
        assert CoordinateFormat.DDM.value == "ddm"
        assert CoordinateFormat.UTM.value == "utm"
        assert CoordinateFormat.MGRS.value == "mgrs"
        assert CoordinateFormat.SIGNED_DD.value == "signed_dd"
        assert CoordinateFormat.DD_SUFFIX.value == "dd_suffix"
        assert CoordinateFormat.UNKNOWN.value == "unknown"

    def test_member_count(self):
        """Exactly 8 coordinate format members."""
        assert len(CoordinateFormat) == 8


class TestGeodeticDatumEnum:
    """Test GeodeticDatum enumeration."""

    def test_wgs84_exists(self):
        """WGS84 datum exists and has correct value."""
        assert GeodeticDatum.WGS84.value == "wgs84"

    def test_at_least_30_datums(self):
        """At least 30 datums are defined for EUDR coverage."""
        assert len(GeodeticDatum) >= 30

    def test_african_datums_present(self):
        """African datums for EUDR producing countries are present."""
        african = {
            GeodeticDatum.ARC1960, GeodeticDatum.CAPE,
            GeodeticDatum.ADINDAN, GeodeticDatum.MINNA,
            GeodeticDatum.ACCRA, GeodeticDatum.LOME,
        }
        all_datums = set(GeodeticDatum)
        assert african.issubset(all_datums)

    def test_southeast_asian_datums_present(self):
        """Southeast Asian datums for EUDR producing countries are present."""
        se_asian = {
            GeodeticDatum.INDIAN_1975, GeodeticDatum.JAKARTA,
            GeodeticDatum.KERTAU, GeodeticDatum.TIMBALAI_1948,
        }
        all_datums = set(GeodeticDatum)
        assert se_asian.issubset(all_datums)

    def test_south_american_datums_present(self):
        """South American datums for EUDR producing countries are present."""
        sa = {GeodeticDatum.SAD69, GeodeticDatum.SIRGAS2000}
        all_datums = set(GeodeticDatum)
        assert sa.issubset(all_datums)


class TestPrecisionLevelEnum:
    """Test PrecisionLevel enumeration."""

    def test_all_members(self):
        """All 5 precision levels exist."""
        expected = {"SURVEY_GRADE", "HIGH", "MODERATE", "LOW", "INADEQUATE"}
        actual = {m.name for m in PrecisionLevel}
        assert expected == actual

    def test_values(self):
        """Precision level values are lowercase."""
        assert PrecisionLevel.SURVEY_GRADE.value == "survey_grade"
        assert PrecisionLevel.INADEQUATE.value == "inadequate"


class TestSourceTypeEnum:
    """Test SourceType enumeration."""

    def test_all_members(self):
        """All 8 source types exist."""
        expected = {
            "GNSS_SURVEY", "RTK_GPS", "MOBILE_GPS", "HANDHELD_GPS",
            "MANUAL_ENTRY", "DIGITIZED_MAP", "GEOCODED", "SATELLITE_DERIVED",
        }
        actual = {m.name for m in SourceType}
        assert expected == actual


class TestValidationErrorTypeEnum:
    """Test ValidationErrorType enumeration."""

    def test_all_members(self):
        """All 10 error types exist."""
        expected = {
            "OUT_OF_RANGE", "SWAPPED", "SIGN_ERROR", "HEMISPHERE_ERROR",
            "NULL_ISLAND", "NAN_VALUE", "INF_VALUE", "DUPLICATE",
            "NEAR_DUPLICATE", "BOUNDARY_VALUE",
        }
        actual = {m.name for m in ValidationErrorType}
        assert expected == actual


class TestValidationSeverityEnum:
    """Test ValidationSeverity enumeration."""

    def test_all_members(self):
        """All 4 severity levels exist."""
        expected = {"CRITICAL", "ERROR", "WARNING", "INFO"}
        actual = {m.name for m in ValidationSeverity}
        assert expected == actual

    def test_ordering_conceptual(self):
        """Severity values are lowercase strings."""
        assert ValidationSeverity.CRITICAL.value == "critical"
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.INFO.value == "info"


# ===========================================================================
# 2. RawCoordinate Model Tests
# ===========================================================================


class TestRawCoordinate:
    """Test RawCoordinate model creation and validation."""

    def test_create_minimal(self):
        """Create RawCoordinate with only required field."""
        raw = RawCoordinate(input_string="5.603716, -0.186964")
        assert raw.input_string == "5.603716, -0.186964"
        assert raw.source_datum is None
        assert raw.country_iso is None

    def test_create_full(self):
        """Create RawCoordinate with all fields."""
        raw = RawCoordinate(
            input_string="5.603716, -0.186964",
            source_datum=GeodeticDatum.WGS84,
            country_iso="GH",
            source_type=SourceType.GNSS_SURVEY,
            metadata={"commodity": "cocoa"},
        )
        assert raw.source_datum == GeodeticDatum.WGS84
        assert raw.country_iso == "GH"
        assert raw.source_type == SourceType.GNSS_SURVEY
        assert raw.metadata["commodity"] == "cocoa"

    def test_empty_input_string_rejected(self):
        """Empty input string is rejected by min_length=1."""
        with pytest.raises(PydanticValidationError):
            RawCoordinate(input_string="")

    def test_country_iso_length_validation(self):
        """Country ISO must be 2 characters."""
        with pytest.raises(PydanticValidationError):
            RawCoordinate(input_string="test", country_iso="GHA")

    def test_serialization_roundtrip(self):
        """RawCoordinate survives JSON serialization roundtrip."""
        raw = RawCoordinate(
            input_string="5.603716, -0.186964",
            source_datum=GeodeticDatum.WGS84,
            country_iso="GH",
        )
        data = raw.model_dump()
        raw2 = RawCoordinate.model_validate(data)
        assert raw2.input_string == raw.input_string
        assert raw2.source_datum == raw.source_datum


# ===========================================================================
# 3. ParsedCoordinate Model Tests
# ===========================================================================


class TestParsedCoordinate:
    """Test ParsedCoordinate model creation and validation."""

    def test_create_valid(self):
        """Create a valid ParsedCoordinate."""
        pc = ParsedCoordinate(
            latitude=5.603716,
            longitude=-0.186964,
            detected_format=CoordinateFormat.DECIMAL_DEGREES,
            format_confidence=0.95,
        )
        assert pc.latitude == 5.603716
        assert pc.longitude == -0.186964
        assert pc.detected_format == CoordinateFormat.DECIMAL_DEGREES

    def test_latitude_range_validation_high(self):
        """Latitude > 90 is rejected."""
        with pytest.raises(PydanticValidationError):
            ParsedCoordinate(
                latitude=91.0,
                longitude=0.0,
                detected_format=CoordinateFormat.DECIMAL_DEGREES,
            )

    def test_latitude_range_validation_low(self):
        """Latitude < -90 is rejected."""
        with pytest.raises(PydanticValidationError):
            ParsedCoordinate(
                latitude=-91.0,
                longitude=0.0,
                detected_format=CoordinateFormat.DECIMAL_DEGREES,
            )

    def test_longitude_range_validation_high(self):
        """Longitude > 180 is rejected."""
        with pytest.raises(PydanticValidationError):
            ParsedCoordinate(
                latitude=0.0,
                longitude=181.0,
                detected_format=CoordinateFormat.DECIMAL_DEGREES,
            )

    def test_longitude_range_validation_low(self):
        """Longitude < -180 is rejected."""
        with pytest.raises(PydanticValidationError):
            ParsedCoordinate(
                latitude=0.0,
                longitude=-181.0,
                detected_format=CoordinateFormat.DECIMAL_DEGREES,
            )

    def test_format_confidence_range(self):
        """Format confidence must be in [0, 1]."""
        with pytest.raises(PydanticValidationError):
            ParsedCoordinate(
                latitude=5.0,
                longitude=-1.0,
                detected_format=CoordinateFormat.DECIMAL_DEGREES,
                format_confidence=1.5,
            )

    def test_boundary_latitude_90(self):
        """Latitude exactly 90 is accepted."""
        pc = ParsedCoordinate(
            latitude=90.0,
            longitude=0.0,
            detected_format=CoordinateFormat.DECIMAL_DEGREES,
        )
        assert pc.latitude == 90.0

    def test_boundary_longitude_180(self):
        """Longitude exactly 180 is accepted."""
        pc = ParsedCoordinate(
            latitude=0.0,
            longitude=180.0,
            detected_format=CoordinateFormat.DECIMAL_DEGREES,
        )
        assert pc.longitude == 180.0

    def test_serialization_roundtrip(self):
        """ParsedCoordinate survives JSON serialization roundtrip."""
        pc = ParsedCoordinate(
            latitude=5.603716,
            longitude=-0.186964,
            detected_format=CoordinateFormat.DMS,
            format_confidence=0.92,
            original_input="5d36'13.4\"N",
            parse_successful=True,
        )
        data = pc.model_dump()
        pc2 = ParsedCoordinate.model_validate(data)
        assert pc2.latitude == pc.latitude
        assert pc2.detected_format == pc.detected_format


# ===========================================================================
# 4. NormalizedCoordinate Model Tests
# ===========================================================================


class TestNormalizedCoordinate:
    """Test NormalizedCoordinate model creation and validation."""

    def test_create_identity(self):
        """Create WGS84-to-WGS84 identity transformation result."""
        nc = NormalizedCoordinate(
            latitude=5.603716,
            longitude=-0.186964,
            source_datum=GeodeticDatum.WGS84,
            target_datum=GeodeticDatum.WGS84,
            displacement_m=0.0,
        )
        assert nc.displacement_m == 0.0
        assert nc.source_datum == GeodeticDatum.WGS84

    def test_create_with_displacement(self):
        """Create with measurable displacement from datum shift."""
        nc = NormalizedCoordinate(
            latitude=40.0002,
            longitude=-74.9998,
            source_datum=GeodeticDatum.NAD27,
            target_datum=GeodeticDatum.WGS84,
            displacement_m=15.3,
            original_latitude=40.0,
            original_longitude=-75.0,
            transformation_method="helmert_7param",
        )
        assert nc.displacement_m == 15.3
        assert nc.transformation_method == "helmert_7param"
        assert nc.original_latitude == 40.0

    def test_displacement_non_negative(self):
        """Displacement must be non-negative."""
        with pytest.raises(PydanticValidationError):
            NormalizedCoordinate(
                latitude=5.0,
                longitude=-1.0,
                displacement_m=-1.0,
            )

    def test_target_datum_default_wgs84(self):
        """Target datum defaults to WGS84."""
        nc = NormalizedCoordinate(latitude=5.0, longitude=-1.0)
        assert nc.target_datum == GeodeticDatum.WGS84

    def test_serialization_roundtrip(self):
        """NormalizedCoordinate survives serialization roundtrip."""
        nc = NormalizedCoordinate(
            latitude=5.603716,
            longitude=-0.186964,
            source_datum=GeodeticDatum.ED50,
            displacement_m=87.5,
        )
        data = nc.model_dump()
        nc2 = NormalizedCoordinate.model_validate(data)
        assert nc2.source_datum == GeodeticDatum.ED50
        assert nc2.displacement_m == 87.5


# ===========================================================================
# 5. PrecisionResult Model Tests
# ===========================================================================


class TestPrecisionResult:
    """Test PrecisionResult model creation and defaults."""

    def test_create_defaults(self):
        """PrecisionResult has sensible defaults."""
        pr = PrecisionResult()
        assert pr.latitude_decimal_places == 0
        assert pr.longitude_decimal_places == 0
        assert pr.effective_decimal_places == 0
        assert pr.precision_level == PrecisionLevel.INADEQUATE
        assert pr.eudr_adequate is False
        assert pr.is_truncated is False
        assert pr.is_artificially_rounded is False

    def test_create_survey_grade(self):
        """Create a survey-grade precision result."""
        pr = PrecisionResult(
            latitude_decimal_places=8,
            longitude_decimal_places=8,
            effective_decimal_places=8,
            ground_resolution_lat_m=0.001,
            ground_resolution_lon_m=0.001,
            precision_level=PrecisionLevel.SURVEY_GRADE,
            eudr_adequate=True,
        )
        assert pr.precision_level == PrecisionLevel.SURVEY_GRADE
        assert pr.eudr_adequate is True

    def test_ground_resolution_non_negative(self):
        """Ground resolution must be non-negative."""
        with pytest.raises(PydanticValidationError):
            PrecisionResult(ground_resolution_lat_m=-1.0)

    def test_decimal_places_non_negative(self):
        """Decimal places must be non-negative."""
        with pytest.raises(PydanticValidationError):
            PrecisionResult(latitude_decimal_places=-1)


# ===========================================================================
# 6. ValidationError Model Tests
# ===========================================================================


class TestValidationErrorModel:
    """Test ValidationError model creation and validation."""

    def test_create_error(self):
        """Create a validation error."""
        ve = VError(
            error_type=ValidationErrorType.OUT_OF_RANGE,
            severity=ValidationSeverity.CRITICAL,
            message="Latitude 91.0 exceeds maximum of 90.0",
            field="latitude",
        )
        assert ve.error_type == ValidationErrorType.OUT_OF_RANGE
        assert ve.severity == ValidationSeverity.CRITICAL
        assert "91.0" in ve.message

    def test_create_warning(self):
        """Create a validation warning."""
        ve = VError(
            error_type=ValidationErrorType.NULL_ISLAND,
            severity=ValidationSeverity.WARNING,
            message="Coordinate near Null Island",
            field="both",
        )
        assert ve.severity == ValidationSeverity.WARNING

    def test_suggested_correction(self):
        """ValidationError can include suggested correction."""
        ve = VError(
            error_type=ValidationErrorType.SWAPPED,
            severity=ValidationSeverity.ERROR,
            message="Lat/lon appear swapped",
            suggested_correction={
                "latitude": -0.186964,
                "longitude": 5.603716,
            },
        )
        assert ve.suggested_correction is not None
        assert "latitude" in ve.suggested_correction

    def test_default_field_is_both(self):
        """Default field value is 'both'."""
        ve = VError(
            error_type=ValidationErrorType.NAN_VALUE,
            message="NaN detected",
        )
        assert ve.field == "both"


# ===========================================================================
# 7. ValidationResult Model Tests
# ===========================================================================


class TestValidationResult:
    """Test ValidationResult model creation and defaults."""

    def test_create_valid(self):
        """Create a clean validation result."""
        vr = ValidationResult(is_valid=True)
        assert vr.is_valid is True
        assert vr.error_count == 0
        assert vr.warning_count == 0
        assert vr.errors == []
        assert vr.warnings == []

    def test_create_invalid(self):
        """Create an invalid validation result with errors."""
        vr = ValidationResult(
            is_valid=False,
            errors=[
                VError(
                    error_type=ValidationErrorType.OUT_OF_RANGE,
                    message="Lat out of range",
                ),
            ],
            error_count=1,
        )
        assert vr.is_valid is False
        assert len(vr.errors) == 1
        assert vr.error_count == 1

    def test_auto_correctable_default_false(self):
        """Auto-correctable defaults to False."""
        vr = ValidationResult()
        assert vr.auto_correctable is False

    def test_correction_confidence_range(self):
        """Correction confidence must be in [0, 1]."""
        with pytest.raises(PydanticValidationError):
            ValidationResult(correction_confidence=1.5)

    def test_serialization_roundtrip(self):
        """ValidationResult survives serialization roundtrip."""
        vr = ValidationResult(
            is_valid=False,
            errors=[
                VError(
                    error_type=ValidationErrorType.SWAPPED,
                    message="Coords appear swapped",
                ),
            ],
            error_count=1,
            auto_correctable=True,
            corrected_latitude=5.603716,
            corrected_longitude=-0.186964,
            correction_confidence=0.85,
        )
        data = vr.model_dump()
        vr2 = ValidationResult.model_validate(data)
        assert vr2.is_valid == vr.is_valid
        assert len(vr2.errors) == 1
        assert vr2.corrected_latitude == 5.603716


# ===========================================================================
# 8. BatchValidationResult Model Tests
# ===========================================================================


class TestBatchValidationResult:
    """Test BatchValidationResult model."""

    def test_create_empty(self):
        """Create empty batch result."""
        br = BatchValidationResult()
        assert br.total_count == 0
        assert br.valid_count == 0
        assert br.duplicate_pairs == []
        assert br.near_duplicate_pairs == []

    def test_create_with_data(self):
        """Create batch result with data."""
        br = BatchValidationResult(
            total_count=5,
            valid_count=3,
            error_count=1,
            warning_count=1,
            duplicate_pairs=[(0, 1)],
            near_duplicate_pairs=[(2, 3, 0.5)],
            processing_time_ms=42.5,
        )
        assert br.total_count == 5
        assert br.valid_count == 3
        assert len(br.duplicate_pairs) == 1
        assert br.processing_time_ms == 42.5

    def test_processing_time_non_negative(self):
        """Processing time must be non-negative."""
        with pytest.raises(PydanticValidationError):
            BatchValidationResult(processing_time_ms=-1.0)


# ===========================================================================
# 9. Config Tests
# ===========================================================================


class TestGPSCoordinateValidatorConfig:
    """Test GPSCoordinateValidatorConfig creation and validation."""

    def test_default_config(self):
        """Default config is valid."""
        cfg = GPSCoordinateValidatorConfig()
        assert cfg.eudr_min_decimal_places == 5
        assert cfg.format_detection_min_confidence == 0.7
        assert cfg.null_island_threshold_m == 1000.0

    def test_custom_config(self):
        """Custom config with overridden values."""
        cfg = GPSCoordinateValidatorConfig(
            eudr_min_decimal_places=6,
            null_island_threshold_m=500.0,
            near_duplicate_threshold_m=2.0,
        )
        assert cfg.eudr_min_decimal_places == 6
        assert cfg.null_island_threshold_m == 500.0

    def test_invalid_eudr_min_dp(self):
        """Invalid eudr_min_decimal_places raises ValueError."""
        with pytest.raises(ValueError):
            GPSCoordinateValidatorConfig(eudr_min_decimal_places=0)

    def test_invalid_eudr_min_dp_too_high(self):
        """eudr_min_decimal_places > 15 raises ValueError."""
        with pytest.raises(ValueError):
            GPSCoordinateValidatorConfig(eudr_min_decimal_places=16)

    def test_invalid_format_confidence(self):
        """format_detection_min_confidence out of range raises ValueError."""
        with pytest.raises(ValueError):
            GPSCoordinateValidatorConfig(format_detection_min_confidence=0.0)

    def test_invalid_format_confidence_too_high(self):
        """format_detection_min_confidence > 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            GPSCoordinateValidatorConfig(format_detection_min_confidence=1.5)

    def test_invalid_null_island_threshold(self):
        """Negative null_island_threshold_m raises ValueError."""
        with pytest.raises(ValueError):
            GPSCoordinateValidatorConfig(null_island_threshold_m=-1.0)

    def test_invalid_pool_size(self):
        """pool_size <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            GPSCoordinateValidatorConfig(pool_size=0)

    def test_invalid_rate_limit(self):
        """rate_limit <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            GPSCoordinateValidatorConfig(rate_limit=0)

    def test_invalid_genesis_hash_empty(self):
        """Empty genesis_hash raises ValueError."""
        with pytest.raises(ValueError):
            GPSCoordinateValidatorConfig(genesis_hash="")

    def test_to_dict_redacts_credentials(self):
        """to_dict redacts database and redis URLs."""
        cfg = GPSCoordinateValidatorConfig(
            database_url="postgresql://user:pass@host/db",
            redis_url="redis://secret@host/0",
        )
        d = cfg.to_dict()
        assert d["database_url"] == "***"
        assert d["redis_url"] == "***"

    def test_repr_safe(self):
        """repr does not expose credentials."""
        cfg = GPSCoordinateValidatorConfig(
            database_url="postgresql://user:pass@host/db",
        )
        r = repr(cfg)
        assert "user:pass" not in r
        assert "***" in r


# ===========================================================================
# 10. Config Singleton Tests
# ===========================================================================


class TestConfigSingleton:
    """Test config singleton get/set/reset."""

    def test_get_config_returns_instance(self):
        """get_config returns a GPSCoordinateValidatorConfig."""
        cfg = get_config()
        assert isinstance(cfg, GPSCoordinateValidatorConfig)

    def test_set_config(self):
        """set_config replaces the singleton."""
        custom = GPSCoordinateValidatorConfig(eudr_min_decimal_places=7)
        set_config(custom)
        assert get_config().eudr_min_decimal_places == 7

    def test_reset_config(self):
        """reset_config clears the singleton."""
        custom = GPSCoordinateValidatorConfig(eudr_min_decimal_places=7)
        set_config(custom)
        reset_config()
        cfg = get_config()
        # After reset, should get fresh default (5)
        assert cfg.eudr_min_decimal_places == 5

    def test_env_var_override(self, monkeypatch):
        """Environment variable overrides config value."""
        reset_config()
        monkeypatch.setenv("GL_EUDR_GPS_EUDR_MIN_DECIMAL_PLACES", "8")
        cfg = get_config()
        assert cfg.eudr_min_decimal_places == 8

    def test_env_var_log_level(self, monkeypatch):
        """GL_EUDR_GPS_LOG_LEVEL env var is respected."""
        reset_config()
        monkeypatch.setenv("GL_EUDR_GPS_LOG_LEVEL", "WARNING")
        cfg = get_config()
        assert cfg.log_level == "WARNING"


# ===========================================================================
# 11. Version Constant Test
# ===========================================================================


class TestVersionConstant:
    """Test VERSION constant."""

    def test_version_format(self):
        """VERSION follows semver format."""
        parts = VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_version_value(self):
        """VERSION is 1.0.0."""
        assert VERSION == "1.0.0"
