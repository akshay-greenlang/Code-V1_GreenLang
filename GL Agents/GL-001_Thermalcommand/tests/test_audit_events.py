"""
Tests for Audit Events Module

Comprehensive test coverage for all audit event types including:
- DecisionAuditEvent
- ActionAuditEvent
- SafetyAuditEvent
- ComplianceAuditEvent
- SystemAuditEvent
- OverrideAuditEvent

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import hashlib
import json
import pytest
from datetime import datetime, timezone, timedelta
from uuid import UUID

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit.audit_events import (
    BaseAuditEvent,
    DecisionAuditEvent,
    ActionAuditEvent,
    SafetyAuditEvent,
    ComplianceAuditEvent,
    SystemAuditEvent,
    OverrideAuditEvent,
    EventType,
    SolverStatus,
    ActionStatus,
    SafetyLevel,
    ComplianceStatus,
    OperatorType,
    ModelVersionInfo,
    InputDatasetReference,
    ConstraintInfo,
    RecommendedAction,
    ExpectedImpact,
    ExplainabilityArtifact,
    UncertaintyQuantification,
    create_event_from_dict,
)


class TestModelVersionInfo:
    """Tests for ModelVersionInfo model."""

    def test_create_model_version_info(self):
        """Test creating ModelVersionInfo."""
        info = ModelVersionInfo(
            model_name="demand_forecast",
            model_version="2.1.0",
            model_hash="abc123def456",
            training_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            metrics={"mae": 0.05, "rmse": 0.08},
        )

        assert info.model_name == "demand_forecast"
        assert info.model_version == "2.1.0"
        assert info.metrics["mae"] == 0.05

    def test_model_version_immutable(self):
        """Test that ModelVersionInfo is immutable."""
        info = ModelVersionInfo(
            model_name="test",
            model_version="1.0.0",
            model_hash="hash123",
        )

        with pytest.raises(TypeError):
            info.model_name = "new_name"


class TestInputDatasetReference:
    """Tests for InputDatasetReference model."""

    def test_create_dataset_reference(self):
        """Test creating InputDatasetReference."""
        ref = InputDatasetReference(
            dataset_id="ds-001",
            dataset_type="sensor",
            schema_version="1.0.0",
            data_hash="sha256hash",
            record_count=1000,
            time_range_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            time_range_end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            source_system="OPC-UA",
        )

        assert ref.dataset_id == "ds-001"
        assert ref.record_count == 1000
        assert ref.source_system == "OPC-UA"

    def test_dataset_reference_validation(self):
        """Test validation of record_count."""
        with pytest.raises(ValueError):
            InputDatasetReference(
                dataset_id="ds-001",
                dataset_type="sensor",
                schema_version="1.0.0",
                data_hash="hash",
                record_count=-1,  # Invalid
                source_system="OPC-UA",
            )


class TestRecommendedAction:
    """Tests for RecommendedAction model."""

    def test_create_recommended_action(self):
        """Test creating RecommendedAction."""
        action = RecommendedAction(
            action_id="act-001",
            tag_id="TIC-001.SP",
            asset_id="boiler-001",
            current_value=450.0,
            recommended_value=460.0,
            lower_bound=400.0,
            upper_bound=500.0,
            ramp_rate=5.0,
            ramp_duration_s=120.0,
            unit="degF",
            priority=1,
            rationale="Increase efficiency",
        )

        assert action.action_id == "act-001"
        assert action.recommended_value == 460.0
        assert action.priority == 1

    def test_action_priority_validation(self):
        """Test priority is between 1-10."""
        with pytest.raises(ValueError):
            RecommendedAction(
                action_id="act-001",
                tag_id="TIC-001.SP",
                asset_id="boiler-001",
                current_value=450.0,
                recommended_value=460.0,
                lower_bound=400.0,
                upper_bound=500.0,
                unit="degF",
                priority=15,  # Invalid, must be 1-10
            )


class TestDecisionAuditEvent:
    """Tests for DecisionAuditEvent."""

    @pytest.fixture
    def sample_decision_event(self):
        """Create sample decision event for testing."""
        now = datetime.now(timezone.utc)
        return DecisionAuditEvent(
            correlation_id="corr-12345",
            asset_id="boiler-001",
            facility_id="plant-001",
            ingestion_timestamp=now - timedelta(seconds=5),
            decision_timestamp=now,
            constraint_set_id="cs-001",
            constraint_set_version="1.0.0",
            safety_boundary_policy_version="2.0.0",
            solver_status=SolverStatus.OPTIMAL,
            solve_time_ms=150.5,
            objective_value=125000.50,
            objective_breakdown={"energy_cost": 100000, "demand_penalty": 25000},
            recommended_actions=[
                RecommendedAction(
                    action_id="act-001",
                    tag_id="TIC-001.SP",
                    asset_id="boiler-001",
                    current_value=450.0,
                    recommended_value=460.0,
                    lower_bound=400.0,
                    upper_bound=500.0,
                    unit="degF",
                )
            ],
        )

    def test_create_decision_event(self, sample_decision_event):
        """Test creating DecisionAuditEvent."""
        event = sample_decision_event

        assert event.event_type == EventType.DECISION
        assert event.correlation_id == "corr-12345"
        assert event.solver_status == SolverStatus.OPTIMAL
        assert len(event.recommended_actions) == 1

    def test_decision_event_hash(self, sample_decision_event):
        """Test event hash calculation."""
        event = sample_decision_event
        hash1 = event.event_hash
        hash2 = event.event_hash

        # Hash should be deterministic
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_decision_event_has_uuid(self, sample_decision_event):
        """Test that event ID is a valid UUID."""
        event = sample_decision_event
        assert isinstance(event.event_id, UUID)

    def test_decision_timestamp_validation(self):
        """Test that decision_timestamp must be >= ingestion_timestamp."""
        now = datetime.now(timezone.utc)

        with pytest.raises(ValueError):
            DecisionAuditEvent(
                correlation_id="corr-12345",
                asset_id="boiler-001",
                ingestion_timestamp=now,
                decision_timestamp=now - timedelta(hours=1),  # Before ingestion
                constraint_set_id="cs-001",
                constraint_set_version="1.0.0",
                safety_boundary_policy_version="2.0.0",
                solver_status=SolverStatus.OPTIMAL,
                solve_time_ms=150.5,
                objective_value=125000.50,
            )

    def test_decision_event_to_chain_dict(self, sample_decision_event):
        """Test conversion to chain dictionary."""
        event = sample_decision_event
        chain_dict = event.to_chain_dict()

        assert "event_hash" in chain_dict
        assert chain_dict["event_type"] == EventType.DECISION


class TestActionAuditEvent:
    """Tests for ActionAuditEvent."""

    @pytest.fixture
    def sample_action_event(self):
        """Create sample action event."""
        now = datetime.now(timezone.utc)
        return ActionAuditEvent(
            correlation_id="corr-12345",
            asset_id="boiler-001",
            decision_event_id="dec-67890",
            decision_correlation_id="corr-12345",
            action=RecommendedAction(
                action_id="act-001",
                tag_id="TIC-001.SP",
                asset_id="boiler-001",
                current_value=450.0,
                recommended_value=460.0,
                lower_bound=400.0,
                upper_bound=500.0,
                unit="degF",
            ),
            action_status=ActionStatus.EXECUTED,
            recommended_timestamp=now - timedelta(minutes=5),
            actuation_timestamp=now,
            executed_value=460.0,
            operator_action="approved",
        )

    def test_create_action_event(self, sample_action_event):
        """Test creating ActionAuditEvent."""
        event = sample_action_event

        assert event.event_type == EventType.ACTION
        assert event.action_status == ActionStatus.EXECUTED
        assert event.executed_value == 460.0

    def test_action_event_actuation_validation(self):
        """Test actuation_timestamp must be >= recommended_timestamp."""
        now = datetime.now(timezone.utc)

        with pytest.raises(ValueError):
            ActionAuditEvent(
                correlation_id="corr-12345",
                asset_id="boiler-001",
                decision_event_id="dec-67890",
                decision_correlation_id="corr-12345",
                action=RecommendedAction(
                    action_id="act-001",
                    tag_id="TIC-001.SP",
                    asset_id="boiler-001",
                    current_value=450.0,
                    recommended_value=460.0,
                    lower_bound=400.0,
                    upper_bound=500.0,
                    unit="degF",
                ),
                action_status=ActionStatus.EXECUTED,
                recommended_timestamp=now,
                actuation_timestamp=now - timedelta(hours=1),  # Before recommended
            )


class TestSafetyAuditEvent:
    """Tests for SafetyAuditEvent."""

    @pytest.fixture
    def sample_safety_event(self):
        """Create sample safety event."""
        return SafetyAuditEvent(
            correlation_id="corr-12345",
            asset_id="boiler-001",
            boundary_id="TEMP_HIGH_001",
            boundary_name="High Temperature Alarm",
            boundary_version="1.0.0",
            safety_level=SafetyLevel.ALARM,
            safety_category="temperature",
            tag_id="TI-001",
            current_value=855.0,
            boundary_value=850.0,
            unit="degF",
            deviation_pct=0.59,
            is_violation=True,
            violation_duration_s=30.0,
            automatic_response="reduce_firing_rate",
            requires_operator_action=True,
        )

    def test_create_safety_event(self, sample_safety_event):
        """Test creating SafetyAuditEvent."""
        event = sample_safety_event

        assert event.event_type == EventType.SAFETY
        assert event.safety_level == SafetyLevel.ALARM
        assert event.is_violation is True
        assert event.boundary_id == "TEMP_HIGH_001"

    def test_safety_event_deviation_validation(self):
        """Test deviation percentage validation."""
        with pytest.raises(ValueError):
            SafetyAuditEvent(
                correlation_id="corr-12345",
                asset_id="boiler-001",
                boundary_id="TEMP_HIGH_001",
                boundary_name="High Temperature Alarm",
                boundary_version="1.0.0",
                safety_level=SafetyLevel.ALARM,
                safety_category="temperature",
                tag_id="TI-001",
                current_value=855.0,
                boundary_value=850.0,
                unit="degF",
                deviation_pct=5000.0,  # >1000% is unreasonable
                is_violation=True,
            )


class TestComplianceAuditEvent:
    """Tests for ComplianceAuditEvent."""

    @pytest.fixture
    def sample_compliance_event(self):
        """Create sample compliance event."""
        now = datetime.now(timezone.utc)
        return ComplianceAuditEvent(
            correlation_id="corr-12345",
            asset_id="boiler-001",
            regulation_id="EPA_40_CFR_98",
            regulation_name="GHG Reporting",
            regulation_version="2024",
            subpart="Subpart C",
            requirement_id="REQ-001",
            requirement_description="Annual emissions reporting",
            compliance_status=ComplianceStatus.COMPLIANT,
            compliance_score=95.5,
            check_type="emission_limit",
            check_method="continuous_monitoring",
            check_timestamp=now,
            measured_value=45.2,
            limit_value=50.0,
            margin_pct=9.6,
            unit="kg/hr",
        )

    def test_create_compliance_event(self, sample_compliance_event):
        """Test creating ComplianceAuditEvent."""
        event = sample_compliance_event

        assert event.event_type == EventType.COMPLIANCE
        assert event.compliance_status == ComplianceStatus.COMPLIANT
        assert event.compliance_score == 95.5

    def test_compliance_score_validation(self):
        """Test compliance score is 0-100."""
        now = datetime.now(timezone.utc)

        with pytest.raises(ValueError):
            ComplianceAuditEvent(
                correlation_id="corr-12345",
                asset_id="boiler-001",
                regulation_id="EPA_40_CFR_98",
                regulation_name="GHG Reporting",
                regulation_version="2024",
                requirement_id="REQ-001",
                requirement_description="Test",
                compliance_status=ComplianceStatus.COMPLIANT,
                compliance_score=150.0,  # Invalid, must be 0-100
                check_type="emission_limit",
                check_method="continuous_monitoring",
                check_timestamp=now,
            )


class TestSystemAuditEvent:
    """Tests for SystemAuditEvent."""

    def test_create_system_event(self):
        """Test creating SystemAuditEvent."""
        event = SystemAuditEvent(
            correlation_id="corr-12345",
            asset_id="system",
            system_event_type="configuration_change",
            component="optimizer",
            previous_state="v1.0.0",
            new_state="v1.1.0",
            configuration_version="1.1.0",
            configuration_hash="abc123",
            details={"changed_parameters": ["solver_timeout", "mip_gap"]},
        )

        assert event.event_type == EventType.SYSTEM
        assert event.system_event_type == "configuration_change"
        assert event.new_state == "v1.1.0"


class TestOverrideAuditEvent:
    """Tests for OverrideAuditEvent."""

    def test_create_override_event(self):
        """Test creating OverrideAuditEvent."""
        now = datetime.now(timezone.utc)
        event = OverrideAuditEvent(
            correlation_id="corr-12345",
            asset_id="boiler-001",
            original_event_id="act-001",
            original_event_type=EventType.ACTION,
            override_type="value_modification",
            original_value=460.0,
            override_value=455.0,
            authorized_by="operator-123",
            authorization_level="supervisor",
            authorization_timestamp=now,
            justification="Operator experience suggests lower value",
            supporting_documentation=["SOC-2024-001"],
            risk_assessment="low",
            risk_accepted_by="supervisor-456",
        )

        assert event.event_type == EventType.OVERRIDE
        assert event.override_value == 455.0
        assert event.authorized_by == "operator-123"


class TestCreateEventFromDict:
    """Tests for create_event_from_dict factory function."""

    def test_create_decision_from_dict(self):
        """Test creating DecisionAuditEvent from dictionary."""
        now = datetime.now(timezone.utc)
        data = {
            "event_type": "DECISION",
            "correlation_id": "corr-12345",
            "asset_id": "boiler-001",
            "ingestion_timestamp": now.isoformat(),
            "decision_timestamp": now.isoformat(),
            "constraint_set_id": "cs-001",
            "constraint_set_version": "1.0.0",
            "safety_boundary_policy_version": "2.0.0",
            "solver_status": "OPTIMAL",
            "solve_time_ms": 150.5,
            "objective_value": 125000.50,
        }

        event = create_event_from_dict(data)

        assert isinstance(event, DecisionAuditEvent)
        assert event.solver_status == SolverStatus.OPTIMAL

    def test_create_safety_from_dict(self):
        """Test creating SafetyAuditEvent from dictionary."""
        data = {
            "event_type": EventType.SAFETY,
            "correlation_id": "corr-12345",
            "asset_id": "boiler-001",
            "boundary_id": "TEMP_HIGH_001",
            "boundary_name": "High Temp",
            "boundary_version": "1.0.0",
            "safety_level": "ALARM",
            "safety_category": "temperature",
            "tag_id": "TI-001",
            "current_value": 855.0,
            "boundary_value": 850.0,
            "unit": "degF",
            "deviation_pct": 0.59,
            "is_violation": True,
        }

        event = create_event_from_dict(data)

        assert isinstance(event, SafetyAuditEvent)
        assert event.safety_level == SafetyLevel.ALARM

    def test_create_unknown_type_raises(self):
        """Test that unknown event type raises ValueError."""
        data = {
            "event_type": "UNKNOWN_TYPE",
            "correlation_id": "corr-12345",
            "asset_id": "boiler-001",
        }

        with pytest.raises(ValueError):
            create_event_from_dict(data)


class TestEventHashIntegrity:
    """Tests for event hash integrity."""

    def test_hash_changes_with_data(self):
        """Test that hash changes when data changes."""
        now = datetime.now(timezone.utc)

        event1 = DecisionAuditEvent(
            correlation_id="corr-12345",
            asset_id="boiler-001",
            ingestion_timestamp=now,
            decision_timestamp=now,
            constraint_set_id="cs-001",
            constraint_set_version="1.0.0",
            safety_boundary_policy_version="2.0.0",
            solver_status=SolverStatus.OPTIMAL,
            solve_time_ms=150.5,
            objective_value=125000.50,
        )

        event2 = DecisionAuditEvent(
            correlation_id="corr-12345",
            asset_id="boiler-001",
            ingestion_timestamp=now,
            decision_timestamp=now,
            constraint_set_id="cs-001",
            constraint_set_version="1.0.0",
            safety_boundary_policy_version="2.0.0",
            solver_status=SolverStatus.OPTIMAL,
            solve_time_ms=150.5,
            objective_value=125001.00,  # Different value
        )

        assert event1.event_hash != event2.event_hash

    def test_hash_is_sha256(self):
        """Test that hash is valid SHA-256."""
        now = datetime.now(timezone.utc)

        event = DecisionAuditEvent(
            correlation_id="corr-12345",
            asset_id="boiler-001",
            ingestion_timestamp=now,
            decision_timestamp=now,
            constraint_set_id="cs-001",
            constraint_set_version="1.0.0",
            safety_boundary_policy_version="2.0.0",
            solver_status=SolverStatus.OPTIMAL,
            solve_time_ms=150.5,
            objective_value=125000.50,
        )

        # Verify it's a valid hex string of correct length
        hash_value = event.event_hash
        assert len(hash_value) == 64
        int(hash_value, 16)  # Should not raise if valid hex


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
