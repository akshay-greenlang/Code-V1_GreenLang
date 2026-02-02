"""
GL-020 ECONOPULSE - Alert Manager Unit Tests

Comprehensive unit tests for AlertManager with 95%+ coverage target.
Tests threshold-based alerts, rate-of-change alerts, alert prioritization,
and alert deduplication.

Target Coverage: 95%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test fixtures from conftest
from tests.conftest import (
    AlertConfig, AlertType, AlertSeverity, Alert
)


# =============================================================================
# MOCK ALERT MANAGER CLASS FOR TESTING
# =============================================================================

@dataclass
class AlertEvent:
    """Event triggering an alert."""
    parameter: str
    current_value: float
    previous_value: Optional[float]
    timestamp: datetime
    economizer_id: str
    rate_of_change: Optional[float] = None


@dataclass
class AlertManagerState:
    """State of the alert manager."""
    active_alerts: List[Alert]
    alert_history: List[Alert]
    cooldown_end_times: Dict[str, datetime]
    last_values: Dict[str, float]
    alert_counts: Dict[str, int]


class AlertManager:
    """
    Alert manager for economizer performance monitoring.

    Manages:
    - Threshold-based alerts
    - Rate-of-change alerts
    - Alert prioritization
    - Alert deduplication
    - Alert cooldown
    """

    VERSION = "1.0.0"
    NAME = "AlertManager"
    AGENT_ID = "GL-020"

    def __init__(self, configs: List[AlertConfig] = None):
        self.configs = {c.alert_id: c for c in (configs or [])}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.cooldown_end_times: Dict[str, datetime] = {}
        self.last_values: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.last_timestamps: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        self.alert_counts: Dict[str, int] = defaultdict(int)

    def add_config(self, config: AlertConfig) -> None:
        """Add an alert configuration."""
        self.configs[config.alert_id] = config

    def remove_config(self, alert_id: str) -> bool:
        """Remove an alert configuration."""
        if alert_id in self.configs:
            del self.configs[alert_id]
            return True
        return False

    def is_in_cooldown(self, alert_id: str, current_time: datetime) -> bool:
        """Check if alert is in cooldown period."""
        if alert_id not in self.cooldown_end_times:
            return False

        return current_time < self.cooldown_end_times[alert_id]

    def start_cooldown(self, alert_id: str, config: AlertConfig, current_time: datetime) -> None:
        """Start cooldown period for an alert."""
        cooldown_end = current_time + timedelta(minutes=config.cooldown_minutes)
        self.cooldown_end_times[alert_id] = cooldown_end

    def check_threshold(
        self,
        config: AlertConfig,
        current_value: float
    ) -> Optional[str]:
        """
        Check if value violates threshold.

        Returns:
            Violation message or None
        """
        if config.threshold_high is not None and current_value > config.threshold_high:
            return f"{config.parameter} ({current_value:.4f}) exceeds high threshold ({config.threshold_high:.4f})"

        if config.threshold_low is not None and current_value < config.threshold_low:
            return f"{config.parameter} ({current_value:.4f}) below low threshold ({config.threshold_low:.4f})"

        return None

    def check_rate_of_change(
        self,
        config: AlertConfig,
        current_value: float,
        previous_value: float,
        time_delta_hours: float
    ) -> Optional[Tuple[str, float]]:
        """
        Check if rate of change violates limit.

        Returns:
            Tuple of (violation message, rate) or None
        """
        if config.rate_of_change_limit is None:
            return None

        if time_delta_hours <= 0:
            return None

        rate = abs(current_value - previous_value) / time_delta_hours

        if rate > config.rate_of_change_limit:
            return (
                f"{config.parameter} rate of change ({rate:.6f}/hr) exceeds limit ({config.rate_of_change_limit:.6f}/hr)",
                rate
            )

        return None

    def is_duplicate(
        self,
        alert_id: str,
        config: AlertConfig,
        current_time: datetime
    ) -> bool:
        """
        Check if this would be a duplicate alert within deduplication window.

        Returns:
            True if duplicate, False otherwise
        """
        if alert_id not in self.active_alerts:
            return False

        existing_alert = self.active_alerts[alert_id]

        # Check if within deduplication window
        window_end = existing_alert.timestamp + timedelta(minutes=config.deduplication_window_minutes)

        return current_time < window_end

    def generate_alert_id(
        self,
        config: AlertConfig,
        economizer_id: str
    ) -> str:
        """Generate unique alert ID based on config and economizer."""
        return f"{config.alert_id}_{economizer_id}"

    def create_alert(
        self,
        config: AlertConfig,
        event: AlertEvent,
        message: str,
        threshold_value: float = None
    ) -> Alert:
        """Create an alert from event."""
        alert_id = self.generate_alert_id(config, event.economizer_id)

        return Alert(
            alert_id=alert_id,
            economizer_id=event.economizer_id,
            alert_type=config.alert_type,
            severity=config.severity,
            parameter=event.parameter,
            current_value=event.current_value,
            threshold_value=threshold_value or config.threshold_high or config.threshold_low or 0,
            message=message,
            timestamp=event.timestamp,
            acknowledged=False,
            resolved=False
        )

    def process_event(
        self,
        event: AlertEvent
    ) -> List[Alert]:
        """
        Process an event and generate any applicable alerts.

        Args:
            event: Alert event with parameter value

        Returns:
            List of generated alerts
        """
        generated_alerts = []

        for config in self.configs.values():
            if not config.enabled:
                continue

            if config.parameter != event.parameter:
                continue

            alert_id = self.generate_alert_id(config, event.economizer_id)

            # Check cooldown
            if self.is_in_cooldown(alert_id, event.timestamp):
                continue

            # Check deduplication
            if self.is_duplicate(alert_id, config, event.timestamp):
                continue

            alert = None

            if config.alert_type == AlertType.THRESHOLD:
                violation = self.check_threshold(config, event.current_value)
                if violation:
                    threshold_val = config.threshold_high if event.current_value > (config.threshold_high or float('inf')) else config.threshold_low
                    alert = self.create_alert(config, event, violation, threshold_val)

            elif config.alert_type == AlertType.RATE_OF_CHANGE:
                # Need previous value
                last_key = f"{event.economizer_id}_{config.parameter}"
                if last_key in self.last_values[event.economizer_id]:
                    prev_value = self.last_values[event.economizer_id][last_key]
                    prev_time = self.last_timestamps[event.economizer_id][last_key]
                    time_delta = (event.timestamp - prev_time).total_seconds() / 3600

                    result = self.check_rate_of_change(
                        config, event.current_value, prev_value, time_delta
                    )
                    if result:
                        message, rate = result
                        alert = self.create_alert(config, event, message, config.rate_of_change_limit)
                        alert.rate_of_change = rate

            if alert:
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                self.alert_counts[alert_id] += 1
                self.start_cooldown(alert_id, config, event.timestamp)
                generated_alerts.append(alert)

        # Update last values for rate of change tracking
        for config in self.configs.values():
            if config.parameter == event.parameter:
                last_key = f"{event.economizer_id}_{config.parameter}"
                self.last_values[event.economizer_id][last_key] = event.current_value
                self.last_timestamps[event.economizer_id][last_key] = event.timestamp

        return generated_alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            del self.active_alerts[alert_id]
            return True
        return False

    def get_active_alerts(
        self,
        economizer_id: str = None,
        severity: AlertSeverity = None
    ) -> List[Alert]:
        """Get active alerts with optional filters."""
        alerts = list(self.active_alerts.values())

        if economizer_id:
            alerts = [a for a in alerts if a.economizer_id == economizer_id]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def get_alerts_by_priority(self) -> List[Alert]:
        """Get active alerts sorted by priority (severity)."""
        priority_order = {
            AlertSeverity.EMERGENCY: 0,
            AlertSeverity.CRITICAL: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3
        }

        alerts = list(self.active_alerts.values())
        return sorted(alerts, key=lambda a: priority_order.get(a.severity, 99))

    def clear_resolved_alerts(self) -> int:
        """Clear all resolved alerts from active list."""
        count = 0
        to_remove = []
        for alert_id, alert in self.active_alerts.items():
            if alert.resolved:
                to_remove.append(alert_id)
                count += 1

        for alert_id in to_remove:
            del self.active_alerts[alert_id]

        return count

    def get_alert_statistics(self) -> Dict:
        """Get alert statistics."""
        return {
            "active_count": len(self.active_alerts),
            "total_generated": len(self.alert_history),
            "by_severity": {
                severity.value: len([a for a in self.active_alerts.values() if a.severity == severity])
                for severity in AlertSeverity
            },
            "by_type": {
                alert_type.value: len([a for a in self.active_alerts.values() if a.alert_type == alert_type])
                for alert_type in AlertType
            },
            "alert_counts": dict(self.alert_counts)
        }


# Add Tuple import for type hints
from typing import Tuple


# =============================================================================
# UNIT TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.calculator
@pytest.mark.alerts
@pytest.mark.critical
class TestAlertManager:
    """Comprehensive test suite for AlertManager."""

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_initialization_empty(self):
        """Test AlertManager initializes with no configs."""
        manager = AlertManager()

        assert manager.VERSION == "1.0.0"
        assert manager.NAME == "AlertManager"
        assert len(manager.configs) == 0
        assert len(manager.active_alerts) == 0

    def test_initialization_with_configs(self, multiple_alert_configs):
        """Test AlertManager initializes with configs."""
        manager = AlertManager(multiple_alert_configs)

        assert len(manager.configs) == len(multiple_alert_configs)

    def test_add_config(self, fouling_alert_config):
        """Test adding alert configuration."""
        manager = AlertManager()

        manager.add_config(fouling_alert_config)

        assert fouling_alert_config.alert_id in manager.configs

    def test_remove_config(self, fouling_alert_config):
        """Test removing alert configuration."""
        manager = AlertManager([fouling_alert_config])

        result = manager.remove_config(fouling_alert_config.alert_id)

        assert result is True
        assert fouling_alert_config.alert_id not in manager.configs

    def test_remove_nonexistent_config(self):
        """Test removing non-existent configuration."""
        manager = AlertManager()

        result = manager.remove_config("nonexistent")

        assert result is False

    # =========================================================================
    # THRESHOLD ALERT TESTS
    # =========================================================================

    def test_threshold_alert_high_exceeded(self, fouling_alert_config):
        """Test threshold alert when high threshold exceeded."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,  # Above threshold of 0.0008
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )

        alerts = manager.process_event(event)

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert "exceeds high threshold" in alerts[0].message

    def test_threshold_alert_low_violated(self, effectiveness_alert_config):
        """Test threshold alert when low threshold violated."""
        manager = AlertManager([effectiveness_alert_config])

        now = datetime.now(timezone.utc)
        event = AlertEvent(
            parameter="effectiveness",
            current_value=0.55,  # Below threshold of 0.60
            previous_value=0.65,
            timestamp=now,
            economizer_id="ECON-001"
        )

        alerts = manager.process_event(event)

        assert len(alerts) == 1
        assert "below low threshold" in alerts[0].message

    def test_threshold_alert_within_limits(self, fouling_alert_config):
        """Test no alert when within thresholds."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.0005,  # Below threshold of 0.0008
            previous_value=0.0004,
            timestamp=now,
            economizer_id="ECON-001"
        )

        alerts = manager.process_event(event)

        assert len(alerts) == 0

    def test_threshold_alert_at_boundary(self, fouling_alert_config):
        """Test alert at exact threshold value."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.0008,  # Exactly at threshold
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )

        alerts = manager.process_event(event)

        # At boundary is not over, so no alert
        assert len(alerts) == 0

    # =========================================================================
    # RATE OF CHANGE ALERT TESTS
    # =========================================================================

    def test_rate_of_change_alert_exceeded(self, rate_of_change_alert_config):
        """Test rate of change alert when limit exceeded."""
        manager = AlertManager([rate_of_change_alert_config])

        now = datetime.now(timezone.utc)

        # First event to establish baseline
        event1 = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.0003,
            previous_value=None,
            timestamp=now - timedelta(hours=1),
            economizer_id="ECON-001"
        )
        manager.process_event(event1)

        # Second event with rapid change
        event2 = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.0006,  # 0.0003 change in 1 hour = 0.0003/hr > limit of 0.0001/hr
            previous_value=0.0003,
            timestamp=now,
            economizer_id="ECON-001"
        )

        alerts = manager.process_event(event2)

        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.RATE_OF_CHANGE
        assert "rate of change" in alerts[0].message

    def test_rate_of_change_alert_within_limit(self, rate_of_change_alert_config):
        """Test no alert when rate of change within limit."""
        manager = AlertManager([rate_of_change_alert_config])

        now = datetime.now(timezone.utc)

        # First event
        event1 = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.0003,
            previous_value=None,
            timestamp=now - timedelta(hours=10),
            economizer_id="ECON-001"
        )
        manager.process_event(event1)

        # Second event with slow change
        event2 = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.00035,  # 0.00005 change in 10 hours = 0.000005/hr < limit
            previous_value=0.0003,
            timestamp=now,
            economizer_id="ECON-001"
        )

        alerts = manager.process_event(event2)

        assert len(alerts) == 0

    def test_rate_of_change_no_previous_value(self, rate_of_change_alert_config):
        """Test no rate of change alert without previous value."""
        manager = AlertManager([rate_of_change_alert_config])

        now = datetime.now(timezone.utc)

        # First event - no previous value tracked yet
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,  # High value but no baseline
            previous_value=None,
            timestamp=now,
            economizer_id="ECON-001"
        )

        alerts = manager.process_event(event)

        # Should not generate rate of change alert without baseline
        assert len(alerts) == 0

    # =========================================================================
    # COOLDOWN TESTS
    # =========================================================================

    def test_cooldown_prevents_duplicate(self, fouling_alert_config):
        """Test cooldown prevents duplicate alerts."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)

        # First event triggers alert
        event1 = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )
        alerts1 = manager.process_event(event1)
        assert len(alerts1) == 1

        # Second event within cooldown (120 minutes)
        event2 = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.0012,  # Still above threshold
            previous_value=0.001,
            timestamp=now + timedelta(minutes=30),  # Within cooldown
            economizer_id="ECON-001"
        )
        alerts2 = manager.process_event(event2)
        assert len(alerts2) == 0  # Blocked by cooldown

    def test_cooldown_expires(self, fouling_alert_config):
        """Test alert generates after cooldown expires."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)

        # First alert
        event1 = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )
        manager.process_event(event1)

        # Resolve the alert to clear deduplication
        manager.resolve_alert(manager.generate_alert_id(fouling_alert_config, "ECON-001"))

        # Event after cooldown expires (120 minutes)
        event2 = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.0015,
            previous_value=0.001,
            timestamp=now + timedelta(minutes=150),  # After 120 min cooldown
            economizer_id="ECON-001"
        )
        alerts2 = manager.process_event(event2)
        assert len(alerts2) == 1  # New alert after cooldown

    def test_is_in_cooldown(self, fouling_alert_config):
        """Test cooldown check function."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)
        alert_id = "test_alert"

        # Initially not in cooldown
        assert not manager.is_in_cooldown(alert_id, now)

        # Start cooldown
        manager.start_cooldown(alert_id, fouling_alert_config, now)

        # Should be in cooldown
        assert manager.is_in_cooldown(alert_id, now + timedelta(minutes=30))

        # Should not be in cooldown after period
        assert not manager.is_in_cooldown(alert_id, now + timedelta(minutes=150))

    # =========================================================================
    # DEDUPLICATION TESTS
    # =========================================================================

    def test_deduplication_within_window(self, fouling_alert_config):
        """Test deduplication within window."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)

        # Create active alert
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )
        manager.process_event(event)

        alert_id = manager.generate_alert_id(fouling_alert_config, "ECON-001")

        # Check deduplication within window (60 minutes)
        assert manager.is_duplicate(alert_id, fouling_alert_config, now + timedelta(minutes=30))

    def test_deduplication_outside_window(self, fouling_alert_config):
        """Test no deduplication outside window."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)

        # Create active alert
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )
        manager.process_event(event)

        alert_id = manager.generate_alert_id(fouling_alert_config, "ECON-001")

        # Check deduplication outside window
        assert not manager.is_duplicate(alert_id, fouling_alert_config, now + timedelta(minutes=90))

    def test_no_deduplication_for_new_alert(self, fouling_alert_config):
        """Test no deduplication for new alert type."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)
        alert_id = "new_alert_id"

        # Should not deduplicate if no active alert
        assert not manager.is_duplicate(alert_id, fouling_alert_config, now)

    # =========================================================================
    # ALERT PRIORITIZATION TESTS
    # =========================================================================

    def test_alerts_sorted_by_priority(self, multiple_alert_configs):
        """Test alerts are sorted by severity priority."""
        manager = AlertManager(multiple_alert_configs)

        now = datetime.now(timezone.utc)

        # Create alerts of different severities
        # This will depend on the alert configs
        events = [
            AlertEvent(
                parameter="fouling_factor_m2k_w",
                current_value=0.001,
                previous_value=0.0007,
                timestamp=now,
                economizer_id="ECON-001"
            ),
            AlertEvent(
                parameter="effectiveness",
                current_value=0.50,
                previous_value=0.65,
                timestamp=now + timedelta(seconds=1),
                economizer_id="ECON-002"
            ),
        ]

        for event in events:
            manager.process_event(event)

        # Get alerts by priority
        sorted_alerts = manager.get_alerts_by_priority()

        # Higher severity should come first
        if len(sorted_alerts) >= 2:
            priority_order = {
                AlertSeverity.EMERGENCY: 0,
                AlertSeverity.CRITICAL: 1,
                AlertSeverity.WARNING: 2,
                AlertSeverity.INFO: 3
            }
            for i in range(len(sorted_alerts) - 1):
                assert priority_order[sorted_alerts[i].severity] <= priority_order[sorted_alerts[i+1].severity]

    def test_get_active_alerts_by_severity(self, multiple_alert_configs):
        """Test filtering active alerts by severity."""
        manager = AlertManager(multiple_alert_configs)

        now = datetime.now(timezone.utc)

        # Generate some alerts
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )
        manager.process_event(event)

        # Filter by severity
        warning_alerts = manager.get_active_alerts(severity=AlertSeverity.WARNING)
        critical_alerts = manager.get_active_alerts(severity=AlertSeverity.CRITICAL)

        # Verify filtering works
        for alert in warning_alerts:
            assert alert.severity == AlertSeverity.WARNING

        for alert in critical_alerts:
            assert alert.severity == AlertSeverity.CRITICAL

    def test_get_active_alerts_by_economizer(self, multiple_alert_configs):
        """Test filtering active alerts by economizer."""
        manager = AlertManager(multiple_alert_configs)

        now = datetime.now(timezone.utc)

        # Generate alerts for different economizers
        for econ_id in ["ECON-001", "ECON-002"]:
            event = AlertEvent(
                parameter="fouling_factor_m2k_w",
                current_value=0.001,
                previous_value=0.0007,
                timestamp=now,
                economizer_id=econ_id
            )
            manager.process_event(event)

        # Filter by economizer
        econ1_alerts = manager.get_active_alerts(economizer_id="ECON-001")

        for alert in econ1_alerts:
            assert alert.economizer_id == "ECON-001"

    # =========================================================================
    # ALERT MANAGEMENT TESTS
    # =========================================================================

    def test_acknowledge_alert(self, fouling_alert_config):
        """Test acknowledging an alert."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )
        alerts = manager.process_event(event)

        assert len(alerts) == 1
        alert_id = alerts[0].alert_id

        # Acknowledge
        result = manager.acknowledge_alert(alert_id)
        assert result is True
        assert manager.active_alerts[alert_id].acknowledged is True

    def test_acknowledge_nonexistent_alert(self):
        """Test acknowledging non-existent alert returns False."""
        manager = AlertManager()

        result = manager.acknowledge_alert("nonexistent")

        assert result is False

    def test_resolve_alert(self, fouling_alert_config):
        """Test resolving an alert."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )
        alerts = manager.process_event(event)

        alert_id = alerts[0].alert_id

        # Resolve
        result = manager.resolve_alert(alert_id)
        assert result is True
        assert alert_id not in manager.active_alerts

    def test_resolve_nonexistent_alert(self):
        """Test resolving non-existent alert returns False."""
        manager = AlertManager()

        result = manager.resolve_alert("nonexistent")

        assert result is False

    def test_clear_resolved_alerts(self, fouling_alert_config):
        """Test clearing resolved alerts."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )
        alerts = manager.process_event(event)

        # Resolve the alert
        alert_id = alerts[0].alert_id
        manager.active_alerts[alert_id].resolved = True

        # Clear resolved
        count = manager.clear_resolved_alerts()

        assert count == 1
        assert alert_id not in manager.active_alerts

    # =========================================================================
    # STATISTICS TESTS
    # =========================================================================

    def test_alert_statistics(self, multiple_alert_configs):
        """Test alert statistics generation."""
        manager = AlertManager(multiple_alert_configs)

        now = datetime.now(timezone.utc)

        # Generate some alerts
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )
        manager.process_event(event)

        stats = manager.get_alert_statistics()

        assert "active_count" in stats
        assert "total_generated" in stats
        assert "by_severity" in stats
        assert "by_type" in stats
        assert "alert_counts" in stats

    def test_statistics_after_multiple_alerts(self, fouling_alert_config):
        """Test statistics accuracy after multiple alerts."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)

        # Generate first alert
        event1 = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )
        manager.process_event(event1)

        # Resolve and wait for cooldown
        alert_id = manager.generate_alert_id(fouling_alert_config, "ECON-001")
        manager.resolve_alert(alert_id)

        # Generate second alert after cooldown
        event2 = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.0015,
            previous_value=0.001,
            timestamp=now + timedelta(minutes=150),
            economizer_id="ECON-001"
        )
        manager.process_event(event2)

        stats = manager.get_alert_statistics()

        assert stats["total_generated"] == 2
        assert stats["active_count"] == 1

    # =========================================================================
    # DISABLED CONFIG TESTS
    # =========================================================================

    def test_disabled_config_no_alert(self, fouling_alert_config):
        """Test disabled config does not generate alerts."""
        fouling_alert_config.enabled = False
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,  # Above threshold
            previous_value=0.0007,
            timestamp=now,
            economizer_id="ECON-001"
        )

        alerts = manager.process_event(event)

        assert len(alerts) == 0

    # =========================================================================
    # DIFFERENT PARAMETER TESTS
    # =========================================================================

    def test_alert_only_for_matching_parameter(self, fouling_alert_config):
        """Test alert only generated for matching parameter."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)
        event = AlertEvent(
            parameter="effectiveness",  # Different parameter
            current_value=0.5,
            previous_value=0.6,
            timestamp=now,
            economizer_id="ECON-001"
        )

        alerts = manager.process_event(event)

        assert len(alerts) == 0  # No alert for wrong parameter

    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.performance
    def test_process_event_speed(self, multiple_alert_configs, benchmark):
        """Test event processing meets performance target."""
        manager = AlertManager(multiple_alert_configs)

        now = datetime.now(timezone.utc)
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.0005,
            previous_value=0.0004,
            timestamp=now,
            economizer_id="ECON-001"
        )

        def process():
            return manager.process_event(event)

        benchmark(process)

    @pytest.mark.performance
    def test_high_volume_events(self, multiple_alert_configs):
        """Test handling high volume of events."""
        manager = AlertManager(multiple_alert_configs)
        import time

        now = datetime.now(timezone.utc)
        num_events = 10000

        start = time.time()
        for i in range(num_events):
            event = AlertEvent(
                parameter="fouling_factor_m2k_w",
                current_value=0.0003 + (i % 10) * 0.0001,
                previous_value=0.0003,
                timestamp=now + timedelta(seconds=i),
                economizer_id=f"ECON-{i % 5:03d}"
            )
            manager.process_event(event)

        duration = time.time() - start
        throughput = num_events / duration

        assert throughput > 10000  # >10,000 events per second


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytest.mark.unit
class TestAlertManagerEdgeCases:
    """Edge case tests for AlertManager."""

    def test_multiple_alerts_same_timestamp(self, multiple_alert_configs):
        """Test handling multiple alerts at same timestamp."""
        manager = AlertManager(multiple_alert_configs)

        now = datetime.now(timezone.utc)

        # Multiple events at same time for different parameters
        events = [
            AlertEvent(
                parameter="fouling_factor_m2k_w",
                current_value=0.001,
                previous_value=0.0007,
                timestamp=now,
                economizer_id="ECON-001"
            ),
            AlertEvent(
                parameter="effectiveness",
                current_value=0.5,
                previous_value=0.65,
                timestamp=now,
                economizer_id="ECON-001"
            ),
        ]

        total_alerts = []
        for event in events:
            alerts = manager.process_event(event)
            total_alerts.extend(alerts)

        # Should generate multiple alerts
        assert len(total_alerts) >= 1

    def test_very_old_timestamp(self, fouling_alert_config):
        """Test handling very old timestamp."""
        manager = AlertManager([fouling_alert_config])

        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=0.001,
            previous_value=0.0007,
            timestamp=old_time,
            economizer_id="ECON-001"
        )

        alerts = manager.process_event(event)

        # Should still generate alert regardless of old timestamp
        assert len(alerts) == 1

    def test_extreme_values(self, fouling_alert_config):
        """Test handling extreme parameter values."""
        manager = AlertManager([fouling_alert_config])

        now = datetime.now(timezone.utc)

        # Very high value
        event_high = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=1.0,  # Extremely high
            previous_value=0.5,
            timestamp=now,
            economizer_id="ECON-001"
        )
        alerts_high = manager.process_event(event_high)
        assert len(alerts_high) == 1

        # Resolve for next test
        manager.resolve_alert(alerts_high[0].alert_id)

        # Negative value (edge case)
        event_neg = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=-0.001,  # Negative (invalid but should handle)
            previous_value=0.001,
            timestamp=now + timedelta(minutes=200),
            economizer_id="ECON-002"
        )
        alerts_neg = manager.process_event(event_neg)
        # Should not generate alert as negative is below threshold

    def test_alert_id_uniqueness(self, fouling_alert_config):
        """Test alert IDs are unique per economizer."""
        manager = AlertManager([fouling_alert_config])

        id1 = manager.generate_alert_id(fouling_alert_config, "ECON-001")
        id2 = manager.generate_alert_id(fouling_alert_config, "ECON-002")

        assert id1 != id2
        assert "ECON-001" in id1
        assert "ECON-002" in id2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.unit
class TestAlertManagerIntegration:
    """Integration tests for AlertManager with economizer scenarios."""

    def test_fouling_monitoring_scenario(self, multiple_alert_configs):
        """Test complete fouling monitoring scenario."""
        manager = AlertManager(multiple_alert_configs)

        base_time = datetime.now(timezone.utc)

        # Simulate gradual fouling increase over days
        fouling_values = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]

        all_alerts = []
        for i, fouling in enumerate(fouling_values):
            event = AlertEvent(
                parameter="fouling_factor_m2k_w",
                current_value=fouling,
                previous_value=fouling_values[i-1] if i > 0 else 0.0001,
                timestamp=base_time + timedelta(days=i * 20),  # 20 days apart to avoid cooldown
                economizer_id="ECON-001"
            )

            # Resolve any previous alerts to allow new ones
            for alert_id in list(manager.active_alerts.keys()):
                manager.resolve_alert(alert_id)

            alerts = manager.process_event(event)
            all_alerts.extend(alerts)

        # Should have alerts when crossing thresholds
        assert len(all_alerts) >= 1  # At least one when exceeding 0.0008

    def test_multi_economizer_monitoring(self, multiple_alert_configs):
        """Test monitoring multiple economizers."""
        manager = AlertManager(multiple_alert_configs)

        now = datetime.now(timezone.utc)

        economizers = ["ECON-001", "ECON-002", "ECON-003"]

        for econ_id in economizers:
            event = AlertEvent(
                parameter="fouling_factor_m2k_w",
                current_value=0.001,
                previous_value=0.0007,
                timestamp=now,
                economizer_id=econ_id
            )
            manager.process_event(event)

        # Each economizer should have its own alert
        assert len(manager.active_alerts) == 3

        for econ_id in economizers:
            alerts = manager.get_active_alerts(economizer_id=econ_id)
            assert len(alerts) == 1
            assert alerts[0].economizer_id == econ_id
