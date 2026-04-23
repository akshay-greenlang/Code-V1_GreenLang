"""
GL-001 ThermalCommand - Webhook Events Tests

Comprehensive tests for webhook event definitions including:
- Event creation and validation
- Provenance hash calculation
- JSON serialization
- Event factory function

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import json
from datetime import datetime, timezone
import pytest

from ..webhook_events import (
    WebhookEventType,
    WebhookEvent,
    HeatPlanCreatedEvent,
    SetpointRecommendationEvent,
    SafetyActionBlockedEvent,
    MaintenanceTriggerEvent,
    ApplyPolicy,
    TriggerLevel,
    create_event,
)


class TestWebhookEventType:
    """Tests for WebhookEventType enum."""

    def test_event_types_defined(self):
        """Verify all required event types are defined."""
        assert WebhookEventType.HEAT_PLAN_CREATED == "heat_plan.created"
        assert WebhookEventType.SETPOINT_RECOMMENDATION == "setpoint.recommendation"
        assert WebhookEventType.SAFETY_ACTION_BLOCKED == "safety.action_blocked"
        assert WebhookEventType.MAINTENANCE_TRIGGER == "maintenance.trigger"

    def test_event_type_values_are_strings(self):
        """Verify event types are string values."""
        for event_type in WebhookEventType:
            assert isinstance(event_type.value, str)
            assert "." in event_type.value  # All should have namespace


class TestWebhookEvent:
    """Tests for base WebhookEvent class."""

    def test_event_creation_with_defaults(self):
        """Test event creation with default values."""
        event = WebhookEvent(event_type=WebhookEventType.HEAT_PLAN_CREATED)

        assert event.event_id is not None
        assert len(event.event_id) == 36  # UUID format
        assert event.event_type == WebhookEventType.HEAT_PLAN_CREATED
        assert event.event_version == "1.0"
        assert event.source == "GL-001-ThermalCommand"
        assert event.timestamp is not None
        assert event.provenance_hash is None

    def test_event_with_correlation_id(self):
        """Test event with correlation ID."""
        event = WebhookEvent(
            event_type=WebhookEventType.HEAT_PLAN_CREATED,
            correlation_id="corr-123"
        )

        assert event.correlation_id == "corr-123"

    def test_event_with_metadata(self):
        """Test event with custom metadata."""
        event = WebhookEvent(
            event_type=WebhookEventType.HEAT_PLAN_CREATED,
            metadata={"custom_field": "value", "count": 42}
        )

        assert event.metadata["custom_field"] == "value"
        assert event.metadata["count"] == 42

    def test_provenance_hash_calculation(self):
        """Test SHA-256 provenance hash calculation."""
        event = WebhookEvent(
            event_type=WebhookEventType.HEAT_PLAN_CREATED,
            event_id="fixed-id-for-test",
        )

        hash1 = event.calculate_provenance_hash()
        hash2 = event.calculate_provenance_hash()

        # Same event should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_provenance_hash_changes_with_data(self):
        """Test that provenance hash changes with event data."""
        event1 = WebhookEvent(
            event_type=WebhookEventType.HEAT_PLAN_CREATED,
            event_id="id-1"
        )
        event2 = WebhookEvent(
            event_type=WebhookEventType.HEAT_PLAN_CREATED,
            event_id="id-2"
        )

        hash1 = event1.calculate_provenance_hash()
        hash2 = event2.calculate_provenance_hash()

        assert hash1 != hash2

    def test_with_provenance_returns_new_event(self):
        """Test that with_provenance returns new instance."""
        event = WebhookEvent(event_type=WebhookEventType.HEAT_PLAN_CREATED)
        event_with_provenance = event.with_provenance()

        assert event.provenance_hash is None
        assert event_with_provenance.provenance_hash is not None
        assert event_with_provenance.event_id == event.event_id

    def test_to_webhook_payload(self):
        """Test conversion to webhook payload."""
        event = WebhookEvent(
            event_type=WebhookEventType.HEAT_PLAN_CREATED,
            event_id="test-id"
        )
        payload = event.to_webhook_payload()

        assert isinstance(payload, dict)
        assert payload["event_id"] == "test-id"
        assert payload["event_type"] == "heat_plan.created"
        assert payload["provenance_hash"] is not None

    def test_to_json(self):
        """Test JSON serialization."""
        event = WebhookEvent(
            event_type=WebhookEventType.HEAT_PLAN_CREATED,
            event_id="test-id"
        )
        json_str = event.to_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["event_id"] == "test-id"


class TestHeatPlanCreatedEvent:
    """Tests for HeatPlanCreatedEvent."""

    def test_event_creation(self):
        """Test heat plan event creation."""
        event = HeatPlanCreatedEvent(
            plan_id="plan-001",
            horizon_hours=24,
            expected_cost_usd=15000.0,
            expected_emissions_kg_co2e=1200.0
        )

        assert event.event_type == WebhookEventType.HEAT_PLAN_CREATED
        assert event.plan_id == "plan-001"
        assert event.horizon_hours == 24
        assert event.expected_cost_usd == 15000.0
        assert event.expected_emissions_kg_co2e == 1200.0
        assert event.optimization_objective == "balanced"

    def test_event_with_all_fields(self):
        """Test event with all optional fields."""
        event = HeatPlanCreatedEvent(
            plan_id="plan-002",
            horizon_hours=48,
            expected_cost_usd=30000.0,
            expected_emissions_kg_co2e=2400.0,
            num_time_slots=48,
            equipment_ids=["FURN-001", "FURN-002"],
            optimization_objective="cost",
            confidence_score=0.95,
            constraints_summary={"max_temp": 1800, "min_efficiency": 0.8}
        )

        assert event.num_time_slots == 48
        assert len(event.equipment_ids) == 2
        assert event.optimization_objective == "cost"
        assert event.confidence_score == 0.95

    def test_horizon_validation(self):
        """Test horizon hours validation."""
        # Valid horizon
        event = HeatPlanCreatedEvent(
            plan_id="plan-001",
            horizon_hours=168,  # Max 1 week
            expected_cost_usd=15000.0,
            expected_emissions_kg_co2e=1200.0
        )
        assert event.horizon_hours == 168

        # Invalid horizon
        with pytest.raises(ValueError):
            HeatPlanCreatedEvent(
                plan_id="plan-001",
                horizon_hours=200,  # Exceeds max
                expected_cost_usd=15000.0,
                expected_emissions_kg_co2e=1200.0
            )

    def test_optimization_objective_validation(self):
        """Test optimization objective validation."""
        # Valid objectives
        for obj in ["cost", "emissions", "balanced", "reliability"]:
            event = HeatPlanCreatedEvent(
                plan_id="plan-001",
                horizon_hours=24,
                expected_cost_usd=15000.0,
                expected_emissions_kg_co2e=1200.0,
                optimization_objective=obj
            )
            assert event.optimization_objective == obj

        # Invalid objective
        with pytest.raises(ValueError):
            HeatPlanCreatedEvent(
                plan_id="plan-001",
                horizon_hours=24,
                expected_cost_usd=15000.0,
                expected_emissions_kg_co2e=1200.0,
                optimization_objective="invalid"
            )


class TestSetpointRecommendationEvent:
    """Tests for SetpointRecommendationEvent."""

    def test_event_creation(self):
        """Test setpoint recommendation event creation."""
        event = SetpointRecommendationEvent(
            rec_id="rec-001",
            tag="TIC-201.SP",
            value=1450.0,
            bounds={"min": 1200.0, "max": 1600.0},
            rationale="Reduce temperature for emissions optimization"
        )

        assert event.event_type == WebhookEventType.SETPOINT_RECOMMENDATION
        assert event.rec_id == "rec-001"
        assert event.tag == "TIC-201.SP"
        assert event.value == 1450.0
        assert event.bounds["min"] == 1200.0
        assert event.bounds["max"] == 1600.0

    def test_event_with_all_fields(self):
        """Test event with all optional fields."""
        event = SetpointRecommendationEvent(
            rec_id="rec-002",
            tag="FIC-101.SP",
            value=250.0,
            unit="SCFM",
            bounds={"min": 100.0, "max": 500.0},
            rationale="Increase fuel flow for efficiency",
            apply_policy=ApplyPolicy.AUTO_SAFE,
            current_value=200.0,
            expected_benefit="5% efficiency improvement",
            urgency="high",
            confidence_score=0.92,
            related_plan_id="plan-001"
        )

        assert event.unit == "SCFM"
        assert event.apply_policy == ApplyPolicy.AUTO_SAFE
        assert event.current_value == 200.0
        assert event.urgency == "high"

    def test_bounds_validation(self):
        """Test bounds dictionary validation."""
        # Missing min
        with pytest.raises(ValueError):
            SetpointRecommendationEvent(
                rec_id="rec-001",
                tag="TIC-201.SP",
                value=1450.0,
                bounds={"max": 1600.0},
                rationale="Test"
            )

        # Min >= max
        with pytest.raises(ValueError):
            SetpointRecommendationEvent(
                rec_id="rec-001",
                tag="TIC-201.SP",
                value=1450.0,
                bounds={"min": 1600.0, "max": 1200.0},
                rationale="Test"
            )

    def test_urgency_validation(self):
        """Test urgency level validation."""
        # Valid urgencies
        for urgency in ["low", "medium", "high", "critical"]:
            event = SetpointRecommendationEvent(
                rec_id="rec-001",
                tag="TIC-201.SP",
                value=1450.0,
                bounds={"min": 1200.0, "max": 1600.0},
                rationale="Test",
                urgency=urgency
            )
            assert event.urgency == urgency

        # Invalid urgency
        with pytest.raises(ValueError):
            SetpointRecommendationEvent(
                rec_id="rec-001",
                tag="TIC-201.SP",
                value=1450.0,
                bounds={"min": 1200.0, "max": 1600.0},
                rationale="Test",
                urgency="invalid"
            )


class TestSafetyActionBlockedEvent:
    """Tests for SafetyActionBlockedEvent."""

    def test_event_creation(self):
        """Test safety action blocked event creation."""
        event = SafetyActionBlockedEvent(
            rec_id="rec-001",
            reason="Temperature exceeds SIL-3 high limit",
            boundary_id="TAHH-201",
            current_state_snapshot_ref="snapshot-2024-001"
        )

        assert event.event_type == WebhookEventType.SAFETY_ACTION_BLOCKED
        assert event.rec_id == "rec-001"
        assert event.reason == "Temperature exceeds SIL-3 high limit"
        assert event.boundary_id == "TAHH-201"
        assert event.operator_notification_required is True

    def test_event_with_all_fields(self):
        """Test event with all optional fields."""
        event = SafetyActionBlockedEvent(
            rec_id="rec-002",
            reason="Pressure safety limit violation",
            boundary_id="PAHH-101",
            boundary_type="pressure",
            current_state_snapshot_ref="snapshot-2024-002",
            violated_limit=500.0,
            recommended_value=520.0,
            severity="critical",
            safety_integrity_level="SIL_3",
            required_action="Manual inspection required",
            equipment_id="FURN-001"
        )

        assert event.boundary_type == "pressure"
        assert event.violated_limit == 500.0
        assert event.severity == "critical"
        assert event.safety_integrity_level == "SIL_3"

    def test_severity_validation(self):
        """Test severity level validation."""
        for severity in ["low", "medium", "high", "critical"]:
            event = SafetyActionBlockedEvent(
                rec_id="rec-001",
                reason="Test",
                boundary_id="TAHH-201",
                current_state_snapshot_ref="snapshot-001",
                severity=severity
            )
            assert event.severity == severity

        with pytest.raises(ValueError):
            SafetyActionBlockedEvent(
                rec_id="rec-001",
                reason="Test",
                boundary_id="TAHH-201",
                current_state_snapshot_ref="snapshot-001",
                severity="invalid"
            )

    def test_sil_validation(self):
        """Test SIL level validation."""
        for sil in ["NONE", "SIL_1", "SIL_2", "SIL_3", "SIL_4"]:
            event = SafetyActionBlockedEvent(
                rec_id="rec-001",
                reason="Test",
                boundary_id="TAHH-201",
                current_state_snapshot_ref="snapshot-001",
                safety_integrity_level=sil
            )
            assert event.safety_integrity_level == sil

        with pytest.raises(ValueError):
            SafetyActionBlockedEvent(
                rec_id="rec-001",
                reason="Test",
                boundary_id="TAHH-201",
                current_state_snapshot_ref="snapshot-001",
                safety_integrity_level="SIL_5"
            )


class TestMaintenanceTriggerEvent:
    """Tests for MaintenanceTriggerEvent."""

    def test_event_creation(self):
        """Test maintenance trigger event creation."""
        event = MaintenanceTriggerEvent(
            asset_id="FURN-001",
            trigger_level=TriggerLevel.WARNING,
            evidence_refs=["vibration-trend-001", "temp-anomaly-002"],
            recommended_task="Replace burner nozzle and inspect refractory"
        )

        assert event.event_type == WebhookEventType.MAINTENANCE_TRIGGER
        assert event.asset_id == "FURN-001"
        assert event.trigger_level == TriggerLevel.WARNING
        assert len(event.evidence_refs) == 2
        assert event.maintenance_type == "predictive"

    def test_event_with_all_fields(self):
        """Test event with all optional fields."""
        event = MaintenanceTriggerEvent(
            asset_id="FURN-002",
            trigger_level=TriggerLevel.CRITICAL,
            evidence_refs=["ref-001"],
            recommended_task="Emergency bearing replacement",
            predicted_failure_hours=48.0,
            confidence_score=0.95,
            maintenance_type="corrective",
            estimated_duration_hours=8.0,
            estimated_cost_usd=25000.0,
            asset_criticality="high",
            affected_systems=["heating", "combustion"],
            spare_parts_required=["bearing-123", "seal-456"],
            cmms_work_order_id="WO-2024-001"
        )

        assert event.predicted_failure_hours == 48.0
        assert event.estimated_cost_usd == 25000.0
        assert len(event.affected_systems) == 2

    def test_trigger_levels(self):
        """Test all trigger levels."""
        for level in TriggerLevel:
            event = MaintenanceTriggerEvent(
                asset_id="FURN-001",
                trigger_level=level,
                evidence_refs=["ref-001"],
                recommended_task="Test task"
            )
            assert event.trigger_level == level

    def test_maintenance_type_validation(self):
        """Test maintenance type validation."""
        for mtype in ["predictive", "preventive", "corrective", "emergency"]:
            event = MaintenanceTriggerEvent(
                asset_id="FURN-001",
                trigger_level=TriggerLevel.WARNING,
                evidence_refs=["ref-001"],
                recommended_task="Test",
                maintenance_type=mtype
            )
            assert event.maintenance_type == mtype

        with pytest.raises(ValueError):
            MaintenanceTriggerEvent(
                asset_id="FURN-001",
                trigger_level=TriggerLevel.WARNING,
                evidence_refs=["ref-001"],
                recommended_task="Test",
                maintenance_type="invalid"
            )

    def test_evidence_refs_required(self):
        """Test that evidence refs cannot be empty."""
        with pytest.raises(ValueError):
            MaintenanceTriggerEvent(
                asset_id="FURN-001",
                trigger_level=TriggerLevel.WARNING,
                evidence_refs=[],  # Empty list
                recommended_task="Test"
            )


class TestCreateEventFactory:
    """Tests for create_event factory function."""

    def test_create_heat_plan_event(self):
        """Test creating HeatPlanCreatedEvent via factory."""
        event = create_event(
            WebhookEventType.HEAT_PLAN_CREATED,
            plan_id="plan-001",
            horizon_hours=24,
            expected_cost_usd=15000.0,
            expected_emissions_kg_co2e=1200.0
        )

        assert isinstance(event, HeatPlanCreatedEvent)
        assert event.plan_id == "plan-001"

    def test_create_setpoint_event(self):
        """Test creating SetpointRecommendationEvent via factory."""
        event = create_event(
            WebhookEventType.SETPOINT_RECOMMENDATION,
            rec_id="rec-001",
            tag="TIC-201.SP",
            value=1450.0,
            bounds={"min": 1200.0, "max": 1600.0},
            rationale="Test"
        )

        assert isinstance(event, SetpointRecommendationEvent)
        assert event.rec_id == "rec-001"

    def test_create_safety_event(self):
        """Test creating SafetyActionBlockedEvent via factory."""
        event = create_event(
            WebhookEventType.SAFETY_ACTION_BLOCKED,
            rec_id="rec-001",
            reason="Test reason",
            boundary_id="TAHH-201",
            current_state_snapshot_ref="snapshot-001"
        )

        assert isinstance(event, SafetyActionBlockedEvent)

    def test_create_maintenance_event(self):
        """Test creating MaintenanceTriggerEvent via factory."""
        event = create_event(
            WebhookEventType.MAINTENANCE_TRIGGER,
            asset_id="FURN-001",
            trigger_level=TriggerLevel.WARNING,
            evidence_refs=["ref-001"],
            recommended_task="Test task"
        )

        assert isinstance(event, MaintenanceTriggerEvent)

    def test_create_unknown_event_type(self):
        """Test creating event with unknown type falls back to base."""
        event = create_event(
            WebhookEventType.SYSTEM_STATUS_CHANGED,
            metadata={"status": "healthy"}
        )

        assert isinstance(event, WebhookEvent)
        assert event.event_type == WebhookEventType.SYSTEM_STATUS_CHANGED


class TestEventSerialization:
    """Tests for event serialization and deserialization."""

    def test_json_roundtrip(self):
        """Test JSON serialization and deserialization."""
        original = HeatPlanCreatedEvent(
            plan_id="plan-001",
            horizon_hours=24,
            expected_cost_usd=15000.0,
            expected_emissions_kg_co2e=1200.0
        )

        json_str = original.to_json()
        parsed = json.loads(json_str)

        # Verify key fields preserved
        assert parsed["plan_id"] == "plan-001"
        assert parsed["horizon_hours"] == 24
        assert parsed["expected_cost_usd"] == 15000.0

    def test_datetime_serialization(self):
        """Test datetime fields serialize to ISO format."""
        event = HeatPlanCreatedEvent(
            plan_id="plan-001",
            horizon_hours=24,
            expected_cost_usd=15000.0,
            expected_emissions_kg_co2e=1200.0
        )

        payload = event.to_webhook_payload()

        # Timestamps should be ISO format strings
        timestamp_str = payload["timestamp"]
        assert isinstance(timestamp_str, str)
        # Should be parseable as datetime
        datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

    def test_enum_serialization(self):
        """Test enum fields serialize to string values."""
        event = SetpointRecommendationEvent(
            rec_id="rec-001",
            tag="TIC-201.SP",
            value=1450.0,
            bounds={"min": 1200.0, "max": 1600.0},
            rationale="Test",
            apply_policy=ApplyPolicy.AUTO_SAFE
        )

        payload = event.to_webhook_payload()

        assert payload["apply_policy"] == "auto_safe"
        assert payload["event_type"] == "setpoint.recommendation"
