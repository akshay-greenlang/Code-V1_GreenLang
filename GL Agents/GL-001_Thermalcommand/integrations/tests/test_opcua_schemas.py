# -*- coding: utf-8 -*-
"""
Unit Tests for OPC-UA Schema Definitions

Tests comprehensive validation of all Pydantic models for OPC-UA operations:
- Quality codes and enums
- Engineering units
- Safety boundaries
- Tag metadata and configuration
- Data points
- Subscriptions
- Write requests/responses
- Connection configuration

Author: GL-BackendDeveloper
Version: 1.0.0
"""

import hashlib
import json
import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from integrations.opcua_schemas import (
    OPCUAQualityCode,
    WriteConfirmationStatus,
    TagAccessLevel,
    TagDataType,
    SecurityMode,
    SecurityPolicy,
    EngineeringUnit,
    ENGINEERING_UNITS,
    SafetyBoundary,
    TagMetadata,
    OPCUATagConfig,
    OPCUADataPoint,
    OPCUASubscriptionConfig,
    OPCUASubscription,
    OPCUAWriteRequest,
    OPCUAWriteResponse,
    OPCUASecurityConfig,
    OPCUAConnectionConfig,
)


# =============================================================================
# QUALITY CODE TESTS
# =============================================================================

class TestOPCUAQualityCode:
    """Test OPC-UA quality code enum and methods."""

    def test_good_quality_codes(self):
        """Test that good quality codes are correctly identified."""
        assert OPCUAQualityCode.GOOD.is_good()
        assert OPCUAQualityCode.GOOD_LOCAL_OVERRIDE.is_good()
        assert OPCUAQualityCode.GOOD_SUB_NORMAL.is_good()

    def test_uncertain_quality_codes(self):
        """Test that uncertain quality codes are correctly identified."""
        assert OPCUAQualityCode.UNCERTAIN.is_uncertain()
        assert OPCUAQualityCode.UNCERTAIN_LAST_USABLE.is_uncertain()
        assert OPCUAQualityCode.UNCERTAIN_SENSOR_NOT_ACCURATE.is_uncertain()

    def test_bad_quality_codes(self):
        """Test that bad quality codes are correctly identified."""
        assert OPCUAQualityCode.BAD.is_bad()
        assert OPCUAQualityCode.BAD_CONFIG_ERROR.is_bad()
        assert OPCUAQualityCode.BAD_SENSOR_FAILURE.is_bad()

    def test_quality_code_mutual_exclusivity(self):
        """Test that quality codes are mutually exclusive."""
        good = OPCUAQualityCode.GOOD
        assert good.is_good()
        assert not good.is_uncertain()
        assert not good.is_bad()

        bad = OPCUAQualityCode.BAD
        assert not bad.is_good()
        assert not bad.is_uncertain()
        assert bad.is_bad()


# =============================================================================
# ENGINEERING UNIT TESTS
# =============================================================================

class TestEngineeringUnits:
    """Test engineering unit definitions."""

    def test_common_units_exist(self):
        """Test that common industrial units are defined."""
        assert "celsius" in ENGINEERING_UNITS
        assert "bar" in ENGINEERING_UNITS
        assert "kg_per_s" in ENGINEERING_UNITS
        assert "kw" in ENGINEERING_UNITS
        assert "percent" in ENGINEERING_UNITS

    def test_engineering_unit_properties(self):
        """Test engineering unit object properties."""
        celsius = ENGINEERING_UNITS["celsius"]
        assert celsius.display_name == "C"
        assert celsius.unit_id == 4408652
        assert celsius.description == "Degrees Celsius"

    def test_engineering_unit_immutable(self):
        """Test that engineering units are frozen."""
        celsius = ENGINEERING_UNITS["celsius"]
        with pytest.raises(TypeError):
            celsius.display_name = "X"


# =============================================================================
# SAFETY BOUNDARY TESTS
# =============================================================================

class TestSafetyBoundary:
    """Test safety boundary validation."""

    @pytest.fixture
    def sample_boundary(self):
        """Create sample safety boundary."""
        return SafetyBoundary(
            tag_id="steam.headerA.pressure",
            min_value=5.0,
            max_value=25.0,
            rate_of_change_limit=2.0,
            deadband=0.1,
            safety_interlock_tags=["sis.boiler.pressure_ok"],
            requires_confirmation=True,
            sil_level=2,
        )

    def test_boundary_creation(self, sample_boundary):
        """Test safety boundary creation."""
        assert sample_boundary.tag_id == "steam.headerA.pressure"
        assert sample_boundary.min_value == 5.0
        assert sample_boundary.max_value == 25.0

    def test_is_within_bounds(self, sample_boundary):
        """Test value boundary checking."""
        assert sample_boundary.is_within_bounds(10.0)
        assert sample_boundary.is_within_bounds(5.0)  # At min
        assert sample_boundary.is_within_bounds(25.0)  # At max
        assert not sample_boundary.is_within_bounds(4.9)  # Below min
        assert not sample_boundary.is_within_bounds(25.1)  # Above max

    def test_get_clamped_value(self, sample_boundary):
        """Test value clamping."""
        assert sample_boundary.get_clamped_value(10.0) == 10.0
        assert sample_boundary.get_clamped_value(2.0) == 5.0  # Clamp to min
        assert sample_boundary.get_clamped_value(30.0) == 25.0  # Clamp to max

    def test_invalid_min_max(self):
        """Test validation of min/max values."""
        with pytest.raises(ValidationError):
            SafetyBoundary(
                tag_id="test",
                min_value=10.0,
                max_value=5.0,  # Invalid: max < min
            )

    def test_sil_level_validation(self):
        """Test SIL level validation."""
        # Valid SIL levels
        b1 = SafetyBoundary(tag_id="test", sil_level=1)
        b4 = SafetyBoundary(tag_id="test", sil_level=4)
        assert b1.sil_level == 1
        assert b4.sil_level == 4

        # Invalid SIL levels
        with pytest.raises(ValidationError):
            SafetyBoundary(tag_id="test", sil_level=0)
        with pytest.raises(ValidationError):
            SafetyBoundary(tag_id="test", sil_level=5)


# =============================================================================
# TAG METADATA TESTS
# =============================================================================

class TestTagMetadata:
    """Test tag metadata validation."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample tag metadata."""
        return TagMetadata(
            tag_id="steam_headerA_pressure",
            node_id="ns=2;s=Steam.HeaderA.Pressure",
            canonical_name="steam.headerA.pressure",
            display_name="Steam Header A Pressure",
            description="Main steam header pressure",
            data_type=TagDataType.DOUBLE,
            engineering_unit=ENGINEERING_UNITS["bar"],
            eu_range_low=0.0,
            eu_range_high=30.0,
            access_level=TagAccessLevel.READ_WRITE,
            raw_low=0.0,
            raw_high=65535.0,
            scaled_low=0.0,
            scaled_high=30.0,
        )

    def test_metadata_creation(self, sample_metadata):
        """Test metadata creation."""
        assert sample_metadata.tag_id == "steam_headerA_pressure"
        assert sample_metadata.canonical_name == "steam.headerA.pressure"
        assert sample_metadata.data_type == TagDataType.DOUBLE

    def test_canonical_name_validation(self):
        """Test canonical name format validation."""
        # Valid names
        TagMetadata(
            tag_id="t1",
            node_id="ns=2;s=Test",
            canonical_name="steam.header.pressure",
            display_name="Test",
            data_type=TagDataType.DOUBLE,
        )

        TagMetadata(
            tag_id="t2",
            node_id="ns=2;s=Test",
            canonical_name="boiler.B1.fuel_flow",
            display_name="Test",
            data_type=TagDataType.DOUBLE,
        )

        # Invalid names
        with pytest.raises(ValidationError):
            TagMetadata(
                tag_id="t3",
                node_id="ns=2;s=Test",
                canonical_name="invalid",  # Only one part
                display_name="Test",
                data_type=TagDataType.DOUBLE,
            )

        with pytest.raises(ValidationError):
            TagMetadata(
                tag_id="t4",
                node_id="ns=2;s=Test",
                canonical_name="123.invalid.name",  # Starts with number
                display_name="Test",
                data_type=TagDataType.DOUBLE,
            )

    def test_apply_scaling(self, sample_metadata):
        """Test scaling from raw to engineering units."""
        # Mid-range
        scaled = sample_metadata.apply_scaling(32767.5)
        assert abs(scaled - 15.0) < 0.01

        # Min
        scaled_min = sample_metadata.apply_scaling(0.0)
        assert abs(scaled_min - 0.0) < 0.01

        # Max
        scaled_max = sample_metadata.apply_scaling(65535.0)
        assert abs(scaled_max - 30.0) < 0.01

    def test_reverse_scaling(self, sample_metadata):
        """Test reverse scaling from engineering units to raw."""
        raw = sample_metadata.reverse_scaling(15.0)
        assert abs(raw - 32767.5) < 1.0

        raw_min = sample_metadata.reverse_scaling(0.0)
        assert abs(raw_min - 0.0) < 1.0

        raw_max = sample_metadata.reverse_scaling(30.0)
        assert abs(raw_max - 65535.0) < 1.0


# =============================================================================
# TAG CONFIGURATION TESTS
# =============================================================================

class TestOPCUATagConfig:
    """Test OPC-UA tag configuration."""

    @pytest.fixture
    def sample_tag_config(self):
        """Create sample tag configuration."""
        metadata = TagMetadata(
            tag_id="steam_headerA_pressure",
            node_id="ns=2;s=Steam.HeaderA.Pressure",
            canonical_name="steam.headerA.pressure",
            display_name="Steam Header A Pressure",
            data_type=TagDataType.DOUBLE,
        )
        return OPCUATagConfig(
            metadata=metadata,
            sampling_interval_ms=1000,
            queue_size=10,
            deadband_value=0.5,
        )

    def test_config_creation(self, sample_tag_config):
        """Test tag configuration creation."""
        assert sample_tag_config.sampling_interval_ms == 1000
        assert sample_tag_config.queue_size == 10
        assert sample_tag_config.deadband_value == 0.5

    def test_sampling_interval_validation(self):
        """Test sampling interval bounds."""
        metadata = TagMetadata(
            tag_id="test",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.value",
            display_name="Test",
            data_type=TagDataType.DOUBLE,
        )

        # Valid intervals
        config_min = OPCUATagConfig(metadata=metadata, sampling_interval_ms=100)
        config_max = OPCUATagConfig(metadata=metadata, sampling_interval_ms=60000)
        assert config_min.sampling_interval_ms == 100
        assert config_max.sampling_interval_ms == 60000

        # Invalid intervals
        with pytest.raises(ValidationError):
            OPCUATagConfig(metadata=metadata, sampling_interval_ms=50)  # Below 100ms

        with pytest.raises(ValidationError):
            OPCUATagConfig(metadata=metadata, sampling_interval_ms=70000)  # Above 60000ms


# =============================================================================
# DATA POINT TESTS
# =============================================================================

class TestOPCUADataPoint:
    """Test OPC-UA data point model."""

    @pytest.fixture
    def sample_data_point(self):
        """Create sample data point."""
        now = datetime.now(timezone.utc)
        return OPCUADataPoint(
            tag_id="steam_headerA_pressure",
            node_id="ns=2;s=Steam.HeaderA.Pressure",
            canonical_name="steam.headerA.pressure",
            value=15.5,
            data_type=TagDataType.DOUBLE,
            source_timestamp=now,
            server_timestamp=now,
            quality_code=OPCUAQualityCode.GOOD,
            engineering_unit="bar",
            scaled_value=15.5,
        )

    def test_data_point_creation(self, sample_data_point):
        """Test data point creation."""
        assert sample_data_point.value == 15.5
        assert sample_data_point.quality_code == OPCUAQualityCode.GOOD
        assert sample_data_point.engineering_unit == "bar"

    def test_provenance_hash_calculation(self, sample_data_point):
        """Test provenance hash calculation."""
        hash1 = sample_data_point.calculate_provenance_hash()
        hash2 = sample_data_point.calculate_provenance_hash()

        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_provenance_hash_changes_with_value(self):
        """Test that provenance hash changes with value."""
        now = datetime.now(timezone.utc)
        dp1 = OPCUADataPoint(
            tag_id="test",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.value",
            value=10.0,
            data_type=TagDataType.DOUBLE,
            source_timestamp=now,
            server_timestamp=now,
        )
        dp2 = OPCUADataPoint(
            tag_id="test",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.value",
            value=20.0,  # Different value
            data_type=TagDataType.DOUBLE,
            source_timestamp=now,
            server_timestamp=now,
        )

        assert dp1.calculate_provenance_hash() != dp2.calculate_provenance_hash()

    def test_is_good_quality(self, sample_data_point):
        """Test quality check method."""
        assert sample_data_point.is_good_quality()

        sample_data_point.quality_code = OPCUAQualityCode.BAD
        assert not sample_data_point.is_good_quality()

    def test_get_age_seconds(self, sample_data_point):
        """Test data point age calculation."""
        # Create old data point
        old_time = datetime.now(timezone.utc) - timedelta(seconds=30)
        sample_data_point.source_timestamp = old_time

        age = sample_data_point.get_age_seconds()
        assert 29 <= age <= 31  # Allow some tolerance


# =============================================================================
# SUBSCRIPTION TESTS
# =============================================================================

class TestOPCUASubscription:
    """Test OPC-UA subscription models."""

    @pytest.fixture
    def sample_subscription_config(self):
        """Create sample subscription configuration."""
        metadata = TagMetadata(
            tag_id="test_tag",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.value",
            display_name="Test Tag",
            data_type=TagDataType.DOUBLE,
        )
        tag_config = OPCUATagConfig(metadata=metadata)

        return OPCUASubscriptionConfig(
            name="test_subscription",
            publishing_interval_ms=1000,
            tag_configs=[tag_config],
        )

    def test_subscription_config_creation(self, sample_subscription_config):
        """Test subscription configuration creation."""
        assert sample_subscription_config.name == "test_subscription"
        assert sample_subscription_config.publishing_interval_ms == 1000
        assert sample_subscription_config.tag_count == 1

    def test_subscription_state_tracking(self, sample_subscription_config):
        """Test subscription runtime state tracking."""
        subscription = OPCUASubscription(config=sample_subscription_config)

        # Initial state
        assert subscription.notification_count == 0
        assert subscription.status == "created"

        # Record notification
        subscription.record_notification()
        assert subscription.notification_count == 1
        assert subscription.last_notification_time is not None

        # Record error
        subscription.record_error()
        assert subscription.error_count == 1


# =============================================================================
# WRITE REQUEST/RESPONSE TESTS
# =============================================================================

class TestOPCUAWriteRequest:
    """Test OPC-UA write request model."""

    @pytest.fixture
    def sample_write_request(self):
        """Create sample write request."""
        return OPCUAWriteRequest(
            tag_id="steam_headerA_pressure_sp",
            node_id="ns=2;s=Steam.HeaderA.Pressure.SP",
            canonical_name="steam.headerA.pressure.setpoint",
            value=12.5,
            data_type=TagDataType.DOUBLE,
            requested_by="operator_001",
            reason="Optimization recommendation",
        )

    def test_write_request_creation(self, sample_write_request):
        """Test write request creation."""
        assert sample_write_request.value == 12.5
        assert sample_write_request.requested_by == "operator_001"
        assert sample_write_request.confirmation_status == WriteConfirmationStatus.PENDING_RECOMMENDATION

    def test_auto_expiration(self, sample_write_request):
        """Test automatic expiration setting."""
        assert sample_write_request.expires_at is not None
        assert sample_write_request.expires_at > sample_write_request.requested_at

    def test_is_expired(self):
        """Test expiration detection."""
        # Create expired request
        expired_request = OPCUAWriteRequest(
            tag_id="test",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.setpoint",
            value=10.0,
            data_type=TagDataType.DOUBLE,
            requested_by="test",
            reason="Test",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )

        assert expired_request.is_expired()

    def test_provenance_hash(self, sample_write_request):
        """Test write request provenance hash."""
        hash1 = sample_write_request.calculate_provenance_hash()
        hash2 = sample_write_request.calculate_provenance_hash()

        assert hash1 == hash2
        assert len(hash1) == 64


class TestOPCUAWriteResponse:
    """Test OPC-UA write response model."""

    def test_write_response_creation(self):
        """Test write response creation."""
        response = OPCUAWriteResponse(
            request_id="req-123",
            success=True,
            status_code=0,
            status_message="Write successful",
            tag_id="test_tag",
            written_value=10.0,
            requested_at=datetime.now(timezone.utc),
            confirmation_status=WriteConfirmationStatus.APPLIED,
            provenance_hash="abc123",
        )

        assert response.success
        assert response.status_code == 0
        assert response.written_value == 10.0

    def test_audit_trail(self):
        """Test audit trail functionality."""
        response = OPCUAWriteResponse(
            request_id="req-123",
            success=True,
            status_code=0,
            status_message="OK",
            tag_id="test",
            written_value=10.0,
            requested_at=datetime.now(timezone.utc),
            confirmation_status=WriteConfirmationStatus.APPLIED,
            provenance_hash="abc",
        )

        response.add_audit_entry("safety_check", {"result": "passed"})
        response.add_audit_entry("write_executed", {"status": "success"})

        assert len(response.audit_trail) == 2
        assert response.audit_trail[0]["action"] == "safety_check"
        assert response.audit_trail[1]["action"] == "write_executed"


# =============================================================================
# CONNECTION CONFIG TESTS
# =============================================================================

class TestOPCUAConnectionConfig:
    """Test OPC-UA connection configuration."""

    def test_connection_config_creation(self):
        """Test connection configuration creation."""
        config = OPCUAConnectionConfig(
            name="plant1_opcua",
            endpoint_url="opc.tcp://192.168.1.100:4840",
            network_segment="OT_DMZ",
        )

        assert config.name == "plant1_opcua"
        assert config.endpoint_url == "opc.tcp://192.168.1.100:4840"
        assert config.auto_reconnect is True  # Default

    def test_endpoint_url_validation(self):
        """Test endpoint URL format validation."""
        # Valid URLs
        OPCUAConnectionConfig(
            name="test",
            endpoint_url="opc.tcp://localhost:4840",
        )
        OPCUAConnectionConfig(
            name="test",
            endpoint_url="opc.https://server.example.com:4840",
        )

        # Invalid URLs
        with pytest.raises(ValidationError):
            OPCUAConnectionConfig(
                name="test",
                endpoint_url="http://localhost:4840",  # Wrong scheme
            )

        with pytest.raises(ValidationError):
            OPCUAConnectionConfig(
                name="test",
                endpoint_url="tcp://localhost:4840",  # Wrong scheme
            )


class TestOPCUASecurityConfig:
    """Test OPC-UA security configuration."""

    def test_security_mode_none(self):
        """Test security mode None configuration."""
        config = OPCUASecurityConfig(
            security_mode=SecurityMode.NONE,
            security_policy=SecurityPolicy.NONE,
        )
        assert config.security_mode == SecurityMode.NONE

    def test_security_mode_sign_requires_cert(self):
        """Test that Sign mode requires certificate."""
        with pytest.raises(ValidationError):
            OPCUASecurityConfig(
                security_mode=SecurityMode.SIGN,
                security_policy=SecurityPolicy.BASIC256SHA256,
                # Missing certificate paths
            )

    def test_security_policy_consistency(self):
        """Test security policy consistency with mode."""
        with pytest.raises(ValidationError):
            OPCUASecurityConfig(
                security_mode=SecurityMode.SIGN_AND_ENCRYPT,
                security_policy=SecurityPolicy.NONE,  # Invalid combo
            )


# =============================================================================
# PROVENANCE AND DETERMINISM TESTS
# =============================================================================

class TestProvenanceDeterminism:
    """Test provenance hash determinism."""

    def test_data_point_hash_deterministic(self):
        """Test that data point hash is deterministic."""
        fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        dp1 = OPCUADataPoint(
            tag_id="test",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.value",
            value=42.5,
            data_type=TagDataType.DOUBLE,
            source_timestamp=fixed_time,
            server_timestamp=fixed_time,
            quality_code=OPCUAQualityCode.GOOD,
        )

        dp2 = OPCUADataPoint(
            tag_id="test",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.value",
            value=42.5,
            data_type=TagDataType.DOUBLE,
            source_timestamp=fixed_time,
            server_timestamp=fixed_time,
            quality_code=OPCUAQualityCode.GOOD,
        )

        assert dp1.calculate_provenance_hash() == dp2.calculate_provenance_hash()

    def test_write_request_hash_deterministic(self):
        """Test that write request hash is deterministic."""
        fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        req1 = OPCUAWriteRequest(
            request_id="fixed-id",
            tag_id="test",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.setpoint",
            value=100.0,
            data_type=TagDataType.DOUBLE,
            requested_by="operator_001",
            reason="Test",
            requested_at=fixed_time,
        )

        req2 = OPCUAWriteRequest(
            request_id="fixed-id",
            tag_id="test",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.setpoint",
            value=100.0,
            data_type=TagDataType.DOUBLE,
            requested_by="operator_001",
            reason="Test",
            requested_at=fixed_time,
        )

        assert req1.calculate_provenance_hash() == req2.calculate_provenance_hash()
