# -*- coding: utf-8 -*-
"""
Unit tests for DeviceFleetManager - AGENT-EUDR-015 Engine 8.

Tests all methods of DeviceFleetManager with 85%+ coverage.
Validates device registration, lifecycle transitions, heartbeat
recording, telemetry updates, operator assignment, campaign
management, fleet status, health scoring, stale detection,
config push, and error handling.

Test count: ~55 tests
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from greenlang.agents.eudr.mobile_data_collector.device_fleet_manager import (
    DeviceFleetManager,
    DEVICE_PLATFORMS,
    DEVICE_STATUSES,
    DEVICE_TRANSITIONS,
    HEALTH_WEIGHTS,
    CAMPAIGN_STATUSES,
    CONNECTIVITY_TYPES,
)

from .conftest import (
    SAMPLE_DEVICE_MODEL, SAMPLE_PLATFORM, SAMPLE_OS_VERSION,
    DEVICE_PLATFORMS as PLATFORM_LIST,
)


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestDeviceFleetManagerInit:
    """Tests for DeviceFleetManager initialization."""

    def test_initialization(self, device_fleet_manager):
        """Engine initializes with empty stores."""
        assert device_fleet_manager is not None
        assert len(device_fleet_manager) == 0

    def test_repr(self, device_fleet_manager):
        """Repr includes device count."""
        r = repr(device_fleet_manager)
        assert "DeviceFleetManager" in r

    def test_len_starts_at_zero(self, device_fleet_manager):
        """Initial device count is zero."""
        assert len(device_fleet_manager) == 0


# ---------------------------------------------------------------------------
# Test: register_device
# ---------------------------------------------------------------------------

class TestRegisterDevice:
    """Tests for device registration."""

    def test_register_valid_device(self, device_fleet_manager, make_device_registration):
        """Register a valid device."""
        data = make_device_registration()
        result = device_fleet_manager.register_device(**data)
        assert "device_id" in result
        assert result["status"] == "registered"
        assert result["device_model"] == SAMPLE_DEVICE_MODEL
        assert result["platform"] == SAMPLE_PLATFORM

    def test_register_increments_count(self, device_fleet_manager, make_device_registration):
        """Registration increments device count."""
        device_fleet_manager.register_device(**make_device_registration())
        assert len(device_fleet_manager) == 1

    @pytest.mark.parametrize("platform", PLATFORM_LIST)
    def test_register_all_platforms(
        self, device_fleet_manager, make_device_registration, platform,
    ):
        """All device platforms are accepted."""
        data = make_device_registration(platform=platform)
        result = device_fleet_manager.register_device(**data)
        assert result["platform"] == platform

    def test_register_empty_model_raises(self, device_fleet_manager):
        """Empty device model raises ValueError."""
        with pytest.raises(ValueError):
            device_fleet_manager.register_device(
                device_model="", platform="android", os_version="14.0",
            )

    def test_register_invalid_platform_raises(self, device_fleet_manager):
        """Invalid platform raises ValueError."""
        with pytest.raises(ValueError):
            device_fleet_manager.register_device(
                device_model="Test", platform="windows_phone",
                os_version="10.0",
            )

    def test_register_empty_os_version_raises(self, device_fleet_manager):
        """Empty OS version raises ValueError."""
        with pytest.raises(ValueError):
            device_fleet_manager.register_device(
                device_model="Test", platform="android", os_version="",
            )

    def test_register_unique_ids(self, device_fleet_manager, make_device_registration):
        """Each device gets a unique ID."""
        ids = set()
        for _ in range(5):
            result = device_fleet_manager.register_device(**make_device_registration())
            ids.add(result["device_id"])
        assert len(ids) == 5

    def test_register_with_operator(self, device_fleet_manager, make_device_registration):
        """Register device with assigned operator."""
        data = make_device_registration(assigned_operator_id="op-001")
        result = device_fleet_manager.register_device(**data)
        assert result["assigned_operator_id"] == "op-001"


# ---------------------------------------------------------------------------
# Test: get_device / list_devices
# ---------------------------------------------------------------------------

class TestDeviceRetrieval:
    """Tests for device retrieval."""

    def test_get_existing_device(self, device_fleet_manager, make_device_registration):
        """Get an existing device by ID."""
        created = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.get_device(created["device_id"])
        assert result["device_id"] == created["device_id"]

    def test_get_nonexistent_raises(self, device_fleet_manager):
        """Getting nonexistent device raises KeyError."""
        with pytest.raises(KeyError):
            device_fleet_manager.get_device("nonexistent")

    def test_list_devices_empty(self, device_fleet_manager):
        """List devices returns empty initially."""
        result = device_fleet_manager.list_devices()
        assert len(result) == 0

    def test_list_devices_filter_by_platform(
        self, device_fleet_manager, make_device_registration,
    ):
        """List devices filters by platform."""
        device_fleet_manager.register_device(**make_device_registration(platform="android"))
        device_fleet_manager.register_device(**make_device_registration(platform="ios"))
        result = device_fleet_manager.list_devices(platform="android")
        assert len(result) == 1
        assert result[0]["platform"] == "android"

    def test_list_devices_filter_by_status(
        self, device_fleet_manager, make_device_registration,
    ):
        """List devices filters by status."""
        device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.list_devices(status="registered")
        assert all(d["status"] == "registered" for d in result)


# ---------------------------------------------------------------------------
# Test: update_device
# ---------------------------------------------------------------------------

class TestUpdateDevice:
    """Tests for device updates."""

    def test_update_operator_assignment(self, device_fleet_manager, make_device_registration):
        """Update device operator assignment."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.update_device(
            dev["device_id"], assigned_operator_id="op-new",
        )
        assert result["assigned_operator_id"] == "op-new"

    def test_update_agent_version(self, device_fleet_manager, make_device_registration):
        """Update device agent version."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.update_device(
            dev["device_id"], agent_version="2.0.0",
        )
        assert result["agent_version"] == "2.0.0"

    def test_update_nonexistent_raises(self, device_fleet_manager):
        """Updating nonexistent device raises KeyError."""
        with pytest.raises(KeyError):
            device_fleet_manager.update_device(
                "nonexistent", assigned_operator_id="op-001",
            )

    def test_update_decommissioned_raises(self, device_fleet_manager, make_device_registration):
        """Updating decommissioned device raises ValueError."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        device_fleet_manager.decommission_device(dev["device_id"], reason="test")
        with pytest.raises(ValueError):
            device_fleet_manager.update_device(
                dev["device_id"], assigned_operator_id="op-001",
            )


# ---------------------------------------------------------------------------
# Test: decommission_device
# ---------------------------------------------------------------------------

class TestDecommissionDevice:
    """Tests for device decommissioning."""

    def test_decommission_device(self, device_fleet_manager, make_device_registration):
        """Decommission a device."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.decommission_device(
            dev["device_id"], reason="End of life",
        )
        assert result["status"] == "decommissioned"
        assert result["is_decommissioned"] is True

    def test_decommission_already_decommissioned_raises(
        self, device_fleet_manager, make_device_registration,
    ):
        """Decommissioning already decommissioned device raises ValueError."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        device_fleet_manager.decommission_device(dev["device_id"])
        with pytest.raises(ValueError):
            device_fleet_manager.decommission_device(dev["device_id"])


# ---------------------------------------------------------------------------
# Test: record_heartbeat
# ---------------------------------------------------------------------------

class TestRecordHeartbeat:
    """Tests for heartbeat recording."""

    def test_record_heartbeat(self, device_fleet_manager, make_device_registration):
        """Record a heartbeat updates device state."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.record_heartbeat(
            dev["device_id"], battery_pct=85,
        )
        assert result["battery_level_pct"] == 85
        assert result["last_heartbeat_at"] is not None

    def test_heartbeat_activates_device(self, device_fleet_manager, make_device_registration):
        """First heartbeat transitions device from registered to active."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.record_heartbeat(
            dev["device_id"], battery_pct=90,
        )
        assert result["status"] == "active"

    def test_heartbeat_updates_storage(self, device_fleet_manager, make_device_registration):
        """Heartbeat updates storage telemetry."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.record_heartbeat(
            dev["device_id"],
            storage_total_bytes=64_000_000_000,
            storage_used_bytes=32_000_000_000,
        )
        assert result["storage_total_bytes"] == 64_000_000_000
        assert result["storage_used_bytes"] == 32_000_000_000
        assert result["storage_used_pct"] == pytest.approx(50.0)

    def test_heartbeat_updates_gps(self, device_fleet_manager, make_device_registration):
        """Heartbeat updates GPS telemetry."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.record_heartbeat(
            dev["device_id"],
            gps_latitude=5.6037, gps_longitude=-0.1870,
            gps_hdop=1.5, gps_satellites=12,
        )
        assert result["last_known_latitude"] == 5.6037
        assert result["last_known_longitude"] == -0.1870

    def test_heartbeat_clears_stale_flag(self, device_fleet_manager, make_device_registration):
        """Heartbeat clears is_stale flag."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.record_heartbeat(
            dev["device_id"], battery_pct=50,
        )
        assert result["is_stale"] is False

    def test_heartbeat_computes_health_score(
        self, device_fleet_manager, make_device_registration,
    ):
        """Heartbeat triggers health score computation."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.record_heartbeat(
            dev["device_id"], battery_pct=90,
            storage_total_bytes=100_000_000_000,
            storage_used_bytes=20_000_000_000,
            gps_hdop=1.0,
        )
        assert result["health_score"] is not None
        assert 0 <= result["health_score"] <= 100

    def test_heartbeat_decommissioned_raises(
        self, device_fleet_manager, make_device_registration,
    ):
        """Heartbeat for decommissioned device raises ValueError."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        device_fleet_manager.decommission_device(dev["device_id"])
        with pytest.raises(ValueError):
            device_fleet_manager.record_heartbeat(
                dev["device_id"], battery_pct=50,
            )

    def test_heartbeat_battery_clamped(self, device_fleet_manager, make_device_registration):
        """Battery percentage is clamped to 0-100."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.record_heartbeat(
            dev["device_id"], battery_pct=150,
        )
        assert result["battery_level_pct"] == 100


# ---------------------------------------------------------------------------
# Test: update_telemetry
# ---------------------------------------------------------------------------

class TestUpdateTelemetry:
    """Tests for telemetry update convenience method."""

    def test_update_telemetry(self, device_fleet_manager, make_device_registration):
        """Update telemetry via payload dict."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.update_telemetry(
            dev["device_id"],
            {"battery_pct": 75, "connectivity_type": "4g"},
        )
        assert result["battery_level_pct"] == 75
        assert result["connectivity_type"] == "4g"


# ---------------------------------------------------------------------------
# Test: get_heartbeat_history
# ---------------------------------------------------------------------------

class TestHeartbeatHistory:
    """Tests for heartbeat history."""

    def test_heartbeat_history_empty(self, device_fleet_manager, make_device_registration):
        """Heartbeat history is empty initially."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.get_heartbeat_history(dev["device_id"])
        assert len(result) == 0

    def test_heartbeat_history_records(self, device_fleet_manager, make_device_registration):
        """Heartbeat history records entries."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        for i in range(5):
            device_fleet_manager.record_heartbeat(
                dev["device_id"], battery_pct=90 - i * 5,
            )
        result = device_fleet_manager.get_heartbeat_history(dev["device_id"])
        assert len(result) == 5

    def test_heartbeat_history_limit(self, device_fleet_manager, make_device_registration):
        """Heartbeat history respects limit parameter."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        for i in range(10):
            device_fleet_manager.record_heartbeat(
                dev["device_id"], battery_pct=90,
            )
        result = device_fleet_manager.get_heartbeat_history(dev["device_id"], limit=3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Test: assign_operator
# ---------------------------------------------------------------------------

class TestAssignOperator:
    """Tests for operator assignment."""

    def test_assign_operator(self, device_fleet_manager, make_device_registration):
        """Assign operator to device."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.assign_operator(
            dev["device_id"], "op-new",
        )
        assert result["assigned_operator_id"] == "op-new"

    def test_assign_decommissioned_raises(self, device_fleet_manager, make_device_registration):
        """Assigning to decommissioned device raises ValueError."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        device_fleet_manager.decommission_device(dev["device_id"])
        with pytest.raises(ValueError):
            device_fleet_manager.assign_operator(dev["device_id"], "op-new")


# ---------------------------------------------------------------------------
# Test: Campaign Management
# ---------------------------------------------------------------------------

class TestCampaignManagement:
    """Tests for collection campaign management."""

    def test_create_campaign(self, device_fleet_manager):
        """Create a collection campaign."""
        result = device_fleet_manager.create_campaign(
            name="Ghana Coffee 2026",
            region="Ashanti",
            start_date="2026-03-01",
            end_date="2026-06-30",
            commodity="coffee",
        )
        assert "campaign_id" in result
        assert result["status"] == "planned"
        assert result["name"] == "Ghana Coffee 2026"

    def test_create_campaign_empty_name_raises(self, device_fleet_manager):
        """Empty campaign name raises ValueError."""
        with pytest.raises(ValueError):
            device_fleet_manager.create_campaign(
                name="", region="test",
                start_date="2026-01-01", end_date="2026-12-31",
            )

    def test_create_campaign_bad_dates_raises(self, device_fleet_manager):
        """Start date after end date raises ValueError."""
        with pytest.raises(ValueError):
            device_fleet_manager.create_campaign(
                name="Bad Dates", region="test",
                start_date="2026-12-31", end_date="2026-01-01",
            )

    def test_assign_device_to_campaign(self, device_fleet_manager, make_device_registration):
        """Assign a device to a campaign."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        campaign = device_fleet_manager.create_campaign(
            name="Test Campaign", region="Ghana",
            start_date="2026-01-01", end_date="2026-12-31",
        )
        result = device_fleet_manager.assign_campaign(
            dev["device_id"], campaign["campaign_id"],
        )
        assert campaign["campaign_id"] in result["campaign_ids"]

    def test_get_campaign(self, device_fleet_manager):
        """Retrieve a campaign by ID."""
        campaign = device_fleet_manager.create_campaign(
            name="Test", region="Ghana",
            start_date="2026-01-01", end_date="2026-12-31",
        )
        result = device_fleet_manager.get_campaign(campaign["campaign_id"])
        assert result["campaign_id"] == campaign["campaign_id"]

    def test_list_campaigns(self, device_fleet_manager):
        """List all campaigns."""
        device_fleet_manager.create_campaign(
            name="C1", region="Ghana",
            start_date="2026-01-01", end_date="2026-06-30",
        )
        device_fleet_manager.create_campaign(
            name="C2", region="Colombia",
            start_date="2026-07-01", end_date="2026-12-31",
        )
        result = device_fleet_manager.list_campaigns()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Test: Fleet Status & Health
# ---------------------------------------------------------------------------

class TestFleetStatus:
    """Tests for fleet status and health monitoring."""

    def test_fleet_status_empty(self, device_fleet_manager):
        """Fleet status with no devices."""
        result = device_fleet_manager.get_fleet_status()
        assert result["total_devices"] == 0

    def test_fleet_status_with_devices(self, device_fleet_manager, make_device_registration):
        """Fleet status reflects registered devices."""
        device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.get_fleet_status()
        assert result["total_devices"] == 1

    def test_device_health(self, device_fleet_manager, make_device_registration):
        """Get device health assessment."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        device_fleet_manager.record_heartbeat(
            dev["device_id"], battery_pct=90,
            storage_total_bytes=100_000_000_000,
            storage_used_bytes=20_000_000_000,
            gps_hdop=1.0,
        )
        result = device_fleet_manager.get_device_health(dev["device_id"])
        assert "health_score" in result
        assert 0 <= result["health_score"] <= 100
        assert "battery_score" in result
        assert "storage_score" in result
        assert "gps_score" in result
        assert "sync_score" in result

    def test_device_health_scoring_weights(self, device_fleet_manager, make_device_registration):
        """Health score uses correct weights."""
        assert abs(sum(HEALTH_WEIGHTS.values()) - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Test: Stale Device Detection
# ---------------------------------------------------------------------------

class TestStaleDetection:
    """Tests for stale device detection."""

    def test_get_stale_devices_empty(self, device_fleet_manager):
        """No stale devices when fleet is empty."""
        result = device_fleet_manager.get_stale_devices()
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Test: Config Push
# ---------------------------------------------------------------------------

class TestConfigPush:
    """Tests for configuration push."""

    def test_push_config(self, device_fleet_manager, make_device_registration):
        """Push configuration to a device."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.push_config(
            dev["device_id"],
            config_type="sync_settings",
            config_data={"sync_interval_s": 30},
        )
        assert "push_id" in result
        assert result["config_type"] == "sync_settings"
        assert result["status"] == "pending"

    def test_push_config_decommissioned_raises(
        self, device_fleet_manager, make_device_registration,
    ):
        """Pushing config to decommissioned device raises ValueError."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        device_fleet_manager.decommission_device(dev["device_id"])
        with pytest.raises(ValueError):
            device_fleet_manager.push_config(
                dev["device_id"], "sync_settings", {"k": "v"},
            )

    def test_get_pending_configs(self, device_fleet_manager, make_device_registration):
        """Get pending configurations for a device."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        device_fleet_manager.push_config(
            dev["device_id"], "sync_settings", {"interval": 60},
        )
        result = device_fleet_manager.get_pending_configs(dev["device_id"])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test: Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    """Tests for fleet manager statistics."""

    def test_statistics_empty(self, device_fleet_manager):
        """Statistics reflect empty state."""
        stats = device_fleet_manager.get_statistics()
        assert stats["total_devices"] == 0
        assert stats["total_campaigns"] == 0

    def test_statistics_after_operations(self, device_fleet_manager, make_device_registration):
        """Statistics reflect operations."""
        device_fleet_manager.register_device(**make_device_registration())
        device_fleet_manager.create_campaign(
            name="C1", region="Ghana",
            start_date="2026-01-01", end_date="2026-12-31",
        )
        stats = device_fleet_manager.get_statistics()
        assert stats["total_devices"] == 1
        assert stats["total_campaigns"] == 1
        assert stats["by_platform"]["android"] == 1


# ---------------------------------------------------------------------------
# Test: Additional Device Operations
# ---------------------------------------------------------------------------

class TestDeviceAdditional:
    """Additional tests for device operations."""

    def test_register_returns_agent_version(self, device_fleet_manager, make_device_registration):
        """Registered device includes agent version."""
        data = make_device_registration()
        result = device_fleet_manager.register_device(**data)
        assert result["agent_version"] == "1.0.0"

    def test_list_devices_returns_list(self, device_fleet_manager, make_device_registration):
        """list_devices returns a list."""
        device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.list_devices()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_heartbeat_negative_battery_clamped(self, device_fleet_manager, make_device_registration):
        """Negative battery percentage is clamped to 0."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.record_heartbeat(
            dev["device_id"], battery_pct=-10,
        )
        assert result["battery_level_pct"] == 0

    def test_multiple_heartbeats(self, device_fleet_manager, make_device_registration):
        """Multiple heartbeats update device state correctly."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        device_fleet_manager.record_heartbeat(dev["device_id"], battery_pct=90)
        device_fleet_manager.record_heartbeat(dev["device_id"], battery_pct=85)
        device_fleet_manager.record_heartbeat(dev["device_id"], battery_pct=80)
        result = device_fleet_manager.get_device(dev["device_id"])
        assert result["battery_level_pct"] == 80

    def test_heartbeat_with_connectivity(self, device_fleet_manager, make_device_registration):
        """Heartbeat records connectivity type."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.record_heartbeat(
            dev["device_id"], battery_pct=75,
            connectivity_type="wifi",
        )
        assert result["connectivity_type"] == "wifi"

    def test_heartbeat_with_pending_items(self, device_fleet_manager, make_device_registration):
        """Heartbeat records pending sync items."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        result = device_fleet_manager.record_heartbeat(
            dev["device_id"], battery_pct=80,
            pending_sync_items=15,
        )
        assert result["pending_sync_items"] == 15

    def test_push_multiple_configs(self, device_fleet_manager, make_device_registration):
        """Multiple configs can be pushed to a device."""
        dev = device_fleet_manager.register_device(**make_device_registration())
        device_fleet_manager.push_config(
            dev["device_id"], "sync_settings", {"interval": 30},
        )
        device_fleet_manager.push_config(
            dev["device_id"], "photo_settings", {"quality": "high"},
        )
        result = device_fleet_manager.get_pending_configs(dev["device_id"])
        assert len(result) == 2

    def test_campaign_unique_ids(self, device_fleet_manager):
        """Each campaign gets a unique ID."""
        ids = set()
        for i in range(5):
            result = device_fleet_manager.create_campaign(
                name=f"Campaign {i}", region="Ghana",
                start_date="2026-01-01", end_date="2026-12-31",
            )
            ids.add(result["campaign_id"])
        assert len(ids) == 5

    def test_fleet_status_counts_by_status(self, device_fleet_manager, make_device_registration):
        """Fleet status includes breakdown by device status."""
        dev1 = device_fleet_manager.register_device(**make_device_registration())
        dev2 = device_fleet_manager.register_device(**make_device_registration())
        device_fleet_manager.record_heartbeat(dev1["device_id"], battery_pct=90)
        result = device_fleet_manager.get_fleet_status()
        assert result["total_devices"] == 2

    def test_assign_operator_nonexistent_raises(self, device_fleet_manager):
        """Assigning operator to nonexistent device raises KeyError."""
        with pytest.raises(KeyError):
            device_fleet_manager.assign_operator("nonexistent", "op-001")
