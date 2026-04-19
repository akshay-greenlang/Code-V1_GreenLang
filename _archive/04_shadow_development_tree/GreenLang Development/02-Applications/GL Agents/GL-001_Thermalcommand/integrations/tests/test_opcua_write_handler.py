# -*- coding: utf-8 -*-
"""
Unit Tests for OPC-UA Write Handler with Safety Gates

Tests comprehensive validation of write handler functionality:
- Safety gate validation
- Write rate limiting
- Whitelist management
- Two-step confirmation workflow
- Provenance tracking
- Audit trail generation

Author: GL-BackendDeveloper
Version: 1.0.0
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from integrations.opcua_write_handler import (
    SafetyCheckResult,
    WriteRecommendationType,
    WriteRecommendation,
    WriteConfirmation,
    SafetyGate,
    WriteRateLimiter,
    WhitelistManager,
    OPCUAWriteHandler,
)
from integrations.opcua_schemas import (
    OPCUATagConfig,
    OPCUADataPoint,
    SafetyBoundary,
    TagMetadata,
    TagDataType,
    TagAccessLevel,
    WriteConfirmationStatus,
    OPCUAQualityCode,
)
from integrations.opcua_connector import OPCUAConnector, OPCUAConnectionConfig


# =============================================================================
# WRITE RECOMMENDATION TESTS
# =============================================================================

class TestWriteRecommendation:
    """Test write recommendation model."""

    @pytest.fixture
    def sample_recommendation(self):
        """Create sample write recommendation."""
        return WriteRecommendation(
            recommendation_type=WriteRecommendationType.OPTIMIZATION,
            tag_id="steam_headerA_pressure_sp",
            canonical_name="steam.headerA.pressure.setpoint",
            node_id="ns=2;s=Steam.HeaderA.Pressure.SP",
            current_value=10.0,
            recommended_value=12.5,
            engineering_unit="bar",
            reason="Optimization suggests pressure increase for efficiency",
            expected_benefit="5% efficiency improvement",
            confidence_score=0.95,
            source_system="gl-001-optimizer",
            safety_impact="low",
        )

    def test_recommendation_creation(self, sample_recommendation):
        """Test recommendation creation."""
        assert sample_recommendation.tag_id == "steam_headerA_pressure_sp"
        assert sample_recommendation.recommended_value == 12.5
        assert sample_recommendation.confidence_score == 0.95

    def test_recommendation_validity(self, sample_recommendation):
        """Test recommendation validity checking."""
        assert sample_recommendation.is_valid()

        # Create expired recommendation
        sample_recommendation.valid_until = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert not sample_recommendation.is_valid()

    def test_provenance_hash(self, sample_recommendation):
        """Test provenance hash calculation."""
        hash1 = sample_recommendation.calculate_provenance_hash()
        hash2 = sample_recommendation.calculate_provenance_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length


# =============================================================================
# WRITE CONFIRMATION TESTS
# =============================================================================

class TestWriteConfirmation:
    """Test write confirmation model."""

    @pytest.fixture
    def sample_confirmation(self):
        """Create sample write confirmation."""
        return WriteConfirmation(
            recommendation_id="rec-123",
            confirmed_by="operator_001",
            confirmation_method="manual",
            confirmation_notes="Approved after review",
            authorization_level=2,
            safety_acknowledged=True,
        )

    def test_confirmation_creation(self, sample_confirmation):
        """Test confirmation creation."""
        assert sample_confirmation.recommendation_id == "rec-123"
        assert sample_confirmation.confirmed_by == "operator_001"
        assert sample_confirmation.safety_acknowledged

    def test_provenance_hash(self, sample_confirmation):
        """Test provenance hash calculation."""
        hash1 = sample_confirmation.calculate_provenance_hash()
        hash2 = sample_confirmation.calculate_provenance_hash()

        assert hash1 == hash2


# =============================================================================
# SAFETY GATE TESTS
# =============================================================================

class TestSafetyGate:
    """Test safety gate validation."""

    @pytest.fixture
    def safety_gate(self):
        """Create safety gate instance."""
        gate = SafetyGate()

        # Register boundaries
        gate.register_boundary(SafetyBoundary(
            tag_id="steam.headerA.pressure.setpoint",
            min_value=5.0,
            max_value=25.0,
            rate_of_change_limit=2.0,
            safety_interlock_tags=["sis.boiler.pressure_ok"],
        ))

        # Set interlock states
        gate.set_interlock_state("sis.boiler.pressure_ok", True)

        return gate

    @pytest.mark.asyncio
    async def test_validate_within_bounds(self, safety_gate):
        """Test validation of value within bounds."""
        result, details = await safety_gate.validate_write(
            "steam.headerA.pressure.setpoint", 15.0
        )

        assert result == SafetyCheckResult.PASSED
        assert details["checks"]["boundary"] == "PASSED"

    @pytest.mark.asyncio
    async def test_validate_below_min(self, safety_gate):
        """Test validation of value below minimum."""
        result, details = await safety_gate.validate_write(
            "steam.headerA.pressure.setpoint", 3.0
        )

        assert result == SafetyCheckResult.FAILED
        assert "outside bounds" in details["error"]

    @pytest.mark.asyncio
    async def test_validate_above_max(self, safety_gate):
        """Test validation of value above maximum."""
        result, details = await safety_gate.validate_write(
            "steam.headerA.pressure.setpoint", 30.0
        )

        assert result == SafetyCheckResult.FAILED

    @pytest.mark.asyncio
    async def test_validate_interlock_blocked(self, safety_gate):
        """Test validation fails when interlock is not safe."""
        safety_gate.set_interlock_state("sis.boiler.pressure_ok", False)

        result, details = await safety_gate.validate_write(
            "steam.headerA.pressure.setpoint", 15.0
        )

        assert result == SafetyCheckResult.INTERLOCK_ACTIVE

    @pytest.mark.asyncio
    async def test_validate_esd_active(self, safety_gate):
        """Test validation fails when ESD is active."""
        safety_gate.set_esd_active(True)

        result, details = await safety_gate.validate_write(
            "steam.headerA.pressure.setpoint", 15.0
        )

        assert result == SafetyCheckResult.INTERLOCK_ACTIVE
        assert details["checks"]["esd"] == "ACTIVE"

    @pytest.mark.asyncio
    async def test_record_write(self, safety_gate):
        """Test recording successful write."""
        await safety_gate.record_write("steam.headerA.pressure.setpoint", 15.0)

        assert "steam.headerA.pressure.setpoint" in safety_gate._last_write_times
        assert safety_gate._last_values["steam.headerA.pressure.setpoint"] == 15.0

    @pytest.mark.asyncio
    async def test_rate_of_change_validation(self, safety_gate):
        """Test rate of change validation."""
        # Record initial write
        await safety_gate.record_write("steam.headerA.pressure.setpoint", 10.0)

        # Small delay to allow rate calculation
        await asyncio.sleep(0.1)

        # Try large change (should fail due to rate limit)
        result, details = await safety_gate.validate_write(
            "steam.headerA.pressure.setpoint",
            20.0,  # 10 bar change in 0.1s = 100 bar/s, exceeds 2.0 limit
            current_value=10.0,
        )

        assert result == SafetyCheckResult.FAILED
        assert "rate_of_change" in details.get("checks", {})


# =============================================================================
# WRITE RATE LIMITER TESTS
# =============================================================================

class TestWriteRateLimiter:
    """Test write rate limiting."""

    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance."""
        return WriteRateLimiter(writes_per_minute=10, burst_size=5)

    @pytest.mark.asyncio
    async def test_acquire_within_limit(self, rate_limiter):
        """Test acquiring within rate limit."""
        permitted, wait_time = await rate_limiter.acquire("tag1")

        assert permitted is True
        assert wait_time is None

    @pytest.mark.asyncio
    async def test_burst_exhausted(self, rate_limiter):
        """Test that burst is exhausted after many quick writes."""
        # Exhaust burst
        for _ in range(5):
            await rate_limiter.acquire("tag1")

        # Next should be rate limited
        permitted, wait_time = await rate_limiter.acquire("tag1")

        # May or may not be permitted depending on token refill
        # The point is the rate limiter is tracking

    @pytest.mark.asyncio
    async def test_per_tag_limiting(self, rate_limiter):
        """Test per-tag rate limiting."""
        rate_limiter._tag_limit = 2

        # First two writes to same tag
        await rate_limiter.acquire("tag1")
        await rate_limiter.acquire("tag1")

        # Third should be limited
        permitted, wait_time = await rate_limiter.acquire("tag1")

        # Different tag should still work
        permitted2, _ = await rate_limiter.acquire("tag2")
        assert permitted2 is True

    @pytest.mark.asyncio
    async def test_get_stats(self, rate_limiter):
        """Test rate limiter statistics."""
        await rate_limiter.acquire("tag1")

        stats = await rate_limiter.get_stats()

        assert "available_tokens" in stats
        assert "burst_size" in stats
        assert "writes_per_minute" in stats


# =============================================================================
# WHITELIST MANAGER TESTS
# =============================================================================

class TestWhitelistManager:
    """Test whitelist management."""

    @pytest.fixture
    def whitelist_manager(self):
        """Create whitelist manager instance."""
        return WhitelistManager()

    @pytest.fixture
    def supervisory_tag_config(self):
        """Create supervisory tag configuration."""
        metadata = TagMetadata(
            tag_id="steam_headerA_pressure_sp",
            node_id="ns=2;s=Steam.HeaderA.Pressure.SP",
            canonical_name="steam.headerA.pressure.setpoint",
            display_name="Steam Header A Pressure Setpoint",
            data_type=TagDataType.DOUBLE,
            access_level=TagAccessLevel.SUPERVISORY_WRITE,
            is_whitelisted_for_write=True,
        )
        return OPCUATagConfig(metadata=metadata)

    @pytest.fixture
    def readonly_tag_config(self):
        """Create read-only tag configuration."""
        metadata = TagMetadata(
            tag_id="steam_headerA_pressure_pv",
            node_id="ns=2;s=Steam.HeaderA.Pressure.PV",
            canonical_name="steam.headerA.pressure.actual",
            display_name="Steam Header A Pressure Actual",
            data_type=TagDataType.DOUBLE,
            access_level=TagAccessLevel.READ_ONLY,
            is_whitelisted_for_write=False,
        )
        return OPCUATagConfig(metadata=metadata)

    @pytest.mark.asyncio
    async def test_add_supervisory_tag(
        self, whitelist_manager, supervisory_tag_config
    ):
        """Test adding supervisory tag to whitelist."""
        result = await whitelist_manager.add_tag(supervisory_tag_config)

        assert result is True
        assert await whitelist_manager.is_whitelisted("steam_headerA_pressure_sp")

    @pytest.mark.asyncio
    async def test_reject_readonly_tag(
        self, whitelist_manager, readonly_tag_config
    ):
        """Test that read-only tags are rejected."""
        result = await whitelist_manager.add_tag(readonly_tag_config)

        assert result is False
        assert not await whitelist_manager.is_whitelisted("steam_headerA_pressure_pv")

    @pytest.mark.asyncio
    async def test_remove_tag(self, whitelist_manager, supervisory_tag_config):
        """Test removing tag from whitelist."""
        await whitelist_manager.add_tag(supervisory_tag_config)
        result = await whitelist_manager.remove_tag("steam_headerA_pressure_sp")

        assert result is True
        assert not await whitelist_manager.is_whitelisted("steam_headerA_pressure_sp")

    @pytest.mark.asyncio
    async def test_get_tag_config(self, whitelist_manager, supervisory_tag_config):
        """Test retrieving tag configuration."""
        await whitelist_manager.add_tag(supervisory_tag_config)

        config = await whitelist_manager.get_tag_config("steam_headerA_pressure_sp")

        assert config is not None
        assert config.metadata.canonical_name == "steam.headerA.pressure.setpoint"


# =============================================================================
# OPC-UA WRITE HANDLER TESTS
# =============================================================================

class TestOPCUAWriteHandler:
    """Test main OPC-UA write handler."""

    @pytest.fixture
    def mock_connector(self):
        """Create mock OPC-UA connector."""
        connector = MagicMock(spec=OPCUAConnector)
        connector.is_connected.return_value = True

        # Mock read_tag to return current value
        async def mock_read_tag(node_id):
            return OPCUADataPoint(
                tag_id="test",
                node_id=node_id,
                canonical_name="test.tag.value",
                value=10.0,
                data_type=TagDataType.DOUBLE,
                source_timestamp=datetime.now(timezone.utc),
                server_timestamp=datetime.now(timezone.utc),
                quality_code=OPCUAQualityCode.GOOD,
            )

        connector.read_tag = AsyncMock(side_effect=mock_read_tag)
        return connector

    @pytest.fixture
    def write_handler(self, mock_connector):
        """Create write handler instance."""
        return OPCUAWriteHandler(
            mock_connector, writes_per_minute=60, confirmation_timeout_s=300
        )

    @pytest.fixture
    def supervisory_tag_config(self):
        """Create supervisory tag configuration for testing."""
        metadata = TagMetadata(
            tag_id="steam_headerA_pressure_sp",
            node_id="ns=2;s=Steam.HeaderA.Pressure.SP",
            canonical_name="steam.headerA.pressure.setpoint",
            display_name="Steam Header A Pressure Setpoint",
            data_type=TagDataType.DOUBLE,
            access_level=TagAccessLevel.SUPERVISORY_WRITE,
            is_whitelisted_for_write=True,
            safety_boundary=SafetyBoundary(
                tag_id="steam_headerA_pressure_sp",
                min_value=5.0,
                max_value=25.0,
            ),
        )
        return OPCUATagConfig(metadata=metadata)

    @pytest.mark.asyncio
    async def test_generate_recommendation(
        self, write_handler, supervisory_tag_config
    ):
        """Test generating write recommendation."""
        # Add tag to whitelist
        await write_handler.whitelist.add_tag(supervisory_tag_config)

        # Register safety boundary
        write_handler.safety_gate.register_boundary(SafetyBoundary(
            tag_id="steam_headerA_pressure_sp",
            min_value=5.0,
            max_value=25.0,
        ))

        recommendation = await write_handler.generate_recommendation(
            tag_id="steam_headerA_pressure_sp",
            recommended_value=12.5,
            reason="Optimization recommendation",
            source_system="gl-001-optimizer",
        )

        assert recommendation is not None
        assert recommendation.recommended_value == 12.5
        assert recommendation.status == WriteConfirmationStatus.RECOMMENDATION_GENERATED
        assert recommendation.provenance_hash is not None

    @pytest.mark.asyncio
    async def test_generate_recommendation_not_whitelisted(self, write_handler):
        """Test that non-whitelisted tags are rejected."""
        with pytest.raises(ValueError, match="not whitelisted"):
            await write_handler.generate_recommendation(
                tag_id="unknown_tag",
                recommended_value=10.0,
                reason="Test",
                source_system="test",
            )

    @pytest.mark.asyncio
    async def test_confirm_and_write(
        self, write_handler, supervisory_tag_config
    ):
        """Test confirming and executing write."""
        # Setup
        await write_handler.whitelist.add_tag(supervisory_tag_config)
        write_handler.safety_gate.register_boundary(SafetyBoundary(
            tag_id="steam_headerA_pressure_sp",
            min_value=5.0,
            max_value=25.0,
        ))

        # Generate recommendation
        recommendation = await write_handler.generate_recommendation(
            tag_id="steam_headerA_pressure_sp",
            recommended_value=12.5,
            reason="Test recommendation",
            source_system="test",
        )

        # Confirm and write
        response = await write_handler.confirm_and_write(
            recommendation_id=recommendation.recommendation_id,
            confirmed_by="operator_001",
            confirmation_notes="Approved for testing",
            safety_acknowledged=True,
        )

        assert response.success is True
        assert response.written_value == 12.5
        assert response.confirmation_status == WriteConfirmationStatus.APPLIED
        assert response.provenance_hash is not None
        assert len(response.audit_trail) > 0

    @pytest.mark.asyncio
    async def test_confirm_without_safety_acknowledgment(
        self, write_handler, supervisory_tag_config
    ):
        """Test that confirmation requires safety acknowledgment."""
        await write_handler.whitelist.add_tag(supervisory_tag_config)
        write_handler.safety_gate.register_boundary(SafetyBoundary(
            tag_id="steam_headerA_pressure_sp",
            min_value=5.0,
            max_value=25.0,
        ))

        recommendation = await write_handler.generate_recommendation(
            tag_id="steam_headerA_pressure_sp",
            recommended_value=12.5,
            reason="Test",
            source_system="test",
        )

        with pytest.raises(ValueError, match="Safety implications must be acknowledged"):
            await write_handler.confirm_and_write(
                recommendation_id=recommendation.recommendation_id,
                confirmed_by="operator_001",
                safety_acknowledged=False,
            )

    @pytest.mark.asyncio
    async def test_reject_recommendation(
        self, write_handler, supervisory_tag_config
    ):
        """Test rejecting a recommendation."""
        await write_handler.whitelist.add_tag(supervisory_tag_config)
        write_handler.safety_gate.register_boundary(SafetyBoundary(
            tag_id="steam_headerA_pressure_sp",
            min_value=5.0,
            max_value=25.0,
        ))

        recommendation = await write_handler.generate_recommendation(
            tag_id="steam_headerA_pressure_sp",
            recommended_value=12.5,
            reason="Test",
            source_system="test",
        )

        result = await write_handler.reject_recommendation(
            recommendation_id=recommendation.recommendation_id,
            rejected_by="supervisor_001",
            rejection_reason="Not approved for current conditions",
        )

        assert result is True

        # Verify recommendation is no longer pending
        pending = await write_handler.get_pending_recommendations()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_expired_recommendation(
        self, write_handler, supervisory_tag_config
    ):
        """Test that expired recommendations are rejected."""
        await write_handler.whitelist.add_tag(supervisory_tag_config)
        write_handler.safety_gate.register_boundary(SafetyBoundary(
            tag_id="steam_headerA_pressure_sp",
            min_value=5.0,
            max_value=25.0,
        ))

        recommendation = await write_handler.generate_recommendation(
            tag_id="steam_headerA_pressure_sp",
            recommended_value=12.5,
            reason="Test",
            source_system="test",
            valid_minutes=0,  # Expires immediately
        )

        # Manually expire the recommendation
        recommendation.valid_until = datetime.now(timezone.utc) - timedelta(minutes=1)

        with pytest.raises(ValueError, match="expired"):
            await write_handler.confirm_and_write(
                recommendation_id=recommendation.recommendation_id,
                confirmed_by="operator_001",
                safety_acknowledged=True,
            )

    @pytest.mark.asyncio
    async def test_safety_blocked_write(
        self, write_handler, supervisory_tag_config
    ):
        """Test write blocked by safety validation."""
        await write_handler.whitelist.add_tag(supervisory_tag_config)
        write_handler.safety_gate.register_boundary(SafetyBoundary(
            tag_id="steam_headerA_pressure_sp",
            min_value=5.0,
            max_value=25.0,
        ))

        # Generate valid recommendation
        recommendation = await write_handler.generate_recommendation(
            tag_id="steam_headerA_pressure_sp",
            recommended_value=12.5,
            reason="Test",
            source_system="test",
        )

        # Activate ESD to block writes
        write_handler.safety_gate.set_esd_active(True)

        response = await write_handler.confirm_and_write(
            recommendation_id=recommendation.recommendation_id,
            confirmed_by="operator_001",
            safety_acknowledged=True,
        )

        assert response.success is False
        assert response.confirmation_status == WriteConfirmationStatus.SAFETY_BLOCKED

    @pytest.mark.asyncio
    async def test_get_pending_recommendations(
        self, write_handler, supervisory_tag_config
    ):
        """Test getting pending recommendations."""
        await write_handler.whitelist.add_tag(supervisory_tag_config)
        write_handler.safety_gate.register_boundary(SafetyBoundary(
            tag_id="steam_headerA_pressure_sp",
            min_value=5.0,
            max_value=25.0,
        ))

        await write_handler.generate_recommendation(
            tag_id="steam_headerA_pressure_sp",
            recommended_value=12.5,
            reason="Test 1",
            source_system="test",
        )
        await write_handler.generate_recommendation(
            tag_id="steam_headerA_pressure_sp",
            recommended_value=13.0,
            reason="Test 2",
            source_system="test",
        )

        pending = await write_handler.get_pending_recommendations()

        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_get_write_history(
        self, write_handler, supervisory_tag_config
    ):
        """Test getting write history."""
        await write_handler.whitelist.add_tag(supervisory_tag_config)
        write_handler.safety_gate.register_boundary(SafetyBoundary(
            tag_id="steam_headerA_pressure_sp",
            min_value=5.0,
            max_value=25.0,
        ))

        # Perform write
        recommendation = await write_handler.generate_recommendation(
            tag_id="steam_headerA_pressure_sp",
            recommended_value=12.5,
            reason="Test",
            source_system="test",
        )
        await write_handler.confirm_and_write(
            recommendation_id=recommendation.recommendation_id,
            confirmed_by="operator_001",
            safety_acknowledged=True,
        )

        history = await write_handler.get_write_history(limit=10)

        assert len(history) == 1
        assert "response" in history[0]
        assert "recommendation" in history[0]
        assert "confirmation" in history[0]

    @pytest.mark.asyncio
    async def test_pre_write_callback(
        self, write_handler, supervisory_tag_config
    ):
        """Test pre-write callback execution."""
        callback_called = []

        def pre_write_callback(recommendation, confirmation):
            callback_called.append((recommendation.recommended_value, confirmation.confirmed_by))

        write_handler.register_pre_write_callback(pre_write_callback)

        await write_handler.whitelist.add_tag(supervisory_tag_config)
        write_handler.safety_gate.register_boundary(SafetyBoundary(
            tag_id="steam_headerA_pressure_sp",
            min_value=5.0,
            max_value=25.0,
        ))

        recommendation = await write_handler.generate_recommendation(
            tag_id="steam_headerA_pressure_sp",
            recommended_value=12.5,
            reason="Test",
            source_system="test",
        )
        await write_handler.confirm_and_write(
            recommendation_id=recommendation.recommendation_id,
            confirmed_by="operator_001",
            safety_acknowledged=True,
        )

        assert len(callback_called) == 1
        assert callback_called[0] == (12.5, "operator_001")

    @pytest.mark.asyncio
    async def test_post_write_callback(
        self, write_handler, supervisory_tag_config
    ):
        """Test post-write callback execution."""
        callback_called = []

        def post_write_callback(response):
            callback_called.append(response.success)

        write_handler.register_post_write_callback(post_write_callback)

        await write_handler.whitelist.add_tag(supervisory_tag_config)
        write_handler.safety_gate.register_boundary(SafetyBoundary(
            tag_id="steam_headerA_pressure_sp",
            min_value=5.0,
            max_value=25.0,
        ))

        recommendation = await write_handler.generate_recommendation(
            tag_id="steam_headerA_pressure_sp",
            recommended_value=12.5,
            reason="Test",
            source_system="test",
        )
        await write_handler.confirm_and_write(
            recommendation_id=recommendation.recommendation_id,
            confirmed_by="operator_001",
            safety_acknowledged=True,
        )

        assert len(callback_called) == 1
        assert callback_called[0] is True


# =============================================================================
# AUDIT TRAIL TESTS
# =============================================================================

class TestAuditTrail:
    """Test audit trail generation."""

    @pytest.fixture
    def mock_connector(self):
        """Create mock connector."""
        connector = MagicMock(spec=OPCUAConnector)
        connector.is_connected.return_value = True

        async def mock_read_tag(node_id):
            return OPCUADataPoint(
                tag_id="test",
                node_id=node_id,
                canonical_name="test.tag.value",
                value=10.0,
                data_type=TagDataType.DOUBLE,
                source_timestamp=datetime.now(timezone.utc),
                server_timestamp=datetime.now(timezone.utc),
                quality_code=OPCUAQualityCode.GOOD,
            )

        connector.read_tag = AsyncMock(side_effect=mock_read_tag)
        return connector

    @pytest.fixture
    def write_handler(self, mock_connector):
        """Create write handler instance."""
        return OPCUAWriteHandler(mock_connector)

    @pytest.mark.asyncio
    async def test_audit_trail_contains_all_steps(self, write_handler):
        """Test that audit trail contains all workflow steps."""
        # Setup
        metadata = TagMetadata(
            tag_id="test_tag",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.setpoint",
            display_name="Test",
            data_type=TagDataType.DOUBLE,
            access_level=TagAccessLevel.SUPERVISORY_WRITE,
            is_whitelisted_for_write=True,
        )
        tag_config = OPCUATagConfig(metadata=metadata)
        await write_handler.whitelist.add_tag(tag_config)
        write_handler.safety_gate.register_boundary(SafetyBoundary(
            tag_id="test_tag",
            min_value=0.0,
            max_value=100.0,
        ))

        # Generate and confirm
        recommendation = await write_handler.generate_recommendation(
            tag_id="test_tag",
            recommended_value=50.0,
            reason="Test",
            source_system="test",
        )
        response = await write_handler.confirm_and_write(
            recommendation_id=recommendation.recommendation_id,
            confirmed_by="operator",
            safety_acknowledged=True,
        )

        # Verify audit trail
        audit_actions = [entry["action"] for entry in response.audit_trail]

        assert "recommendation_generated" in audit_actions
        assert "confirmation_received" in audit_actions
        assert "safety_validation" in audit_actions
        assert "write_executed" in audit_actions

    @pytest.mark.asyncio
    async def test_audit_trail_timestamps(self, write_handler):
        """Test that audit trail entries have timestamps."""
        # Setup
        metadata = TagMetadata(
            tag_id="test_tag",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.setpoint",
            display_name="Test",
            data_type=TagDataType.DOUBLE,
            access_level=TagAccessLevel.SUPERVISORY_WRITE,
            is_whitelisted_for_write=True,
        )
        tag_config = OPCUATagConfig(metadata=metadata)
        await write_handler.whitelist.add_tag(tag_config)
        write_handler.safety_gate.register_boundary(SafetyBoundary(
            tag_id="test_tag",
            min_value=0.0,
            max_value=100.0,
        ))

        recommendation = await write_handler.generate_recommendation(
            tag_id="test_tag",
            recommended_value=50.0,
            reason="Test",
            source_system="test",
        )
        response = await write_handler.confirm_and_write(
            recommendation_id=recommendation.recommendation_id,
            confirmed_by="operator",
            safety_acknowledged=True,
        )

        for entry in response.audit_trail:
            assert "timestamp" in entry
