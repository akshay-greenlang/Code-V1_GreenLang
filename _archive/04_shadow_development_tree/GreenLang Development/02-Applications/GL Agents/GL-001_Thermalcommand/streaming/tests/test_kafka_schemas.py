"""
Tests for Kafka Schemas Module - GL-001 ThermalCommand

Comprehensive test coverage for all Kafka topic schemas.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from ..kafka_schemas import (
    # Enums
    QualityCode,
    UnitOfMeasure,
    SolverStatus,
    SafetyLevel,
    MaintenancePriority,
    AuditAction,
    # Telemetry
    TelemetryPoint,
    TelemetryNormalizedEvent,
    # Dispatch
    LoadAllocation,
    ExpectedImpact,
    DispatchPlanEvent,
    # Recommendations
    SetpointBound,
    SetpointRecommendation,
    ActionRecommendationEvent,
    # Safety
    BoundaryViolation,
    BlockedWrite,
    SISPermissiveChange,
    SafetyEvent,
    # Maintenance
    MaintenanceEvidence,
    MaintenanceTriggerEvent,
    # Explainability
    FeatureContribution,
    SHAPSummary,
    LIMESummary,
    UncertaintyQuantification,
    ExplainabilityReportEvent,
    # Audit
    AuditLogEvent,
    # Registry
    TopicSchemaRegistry,
)


class TestTelemetrySchemas:
    """Tests for telemetry topic schemas."""

    def test_telemetry_point_valid(self) -> None:
        """Test valid telemetry point creation."""
        point = TelemetryPoint(
            tag_id="T-101",
            value=450.5,
            unit=UnitOfMeasure.CELSIUS,
            quality=QualityCode.GOOD,
            timestamp=datetime.now(timezone.utc),
            sensor_id="sensor-001",
            equipment_id="boiler-01",
        )

        assert point.tag_id == "T-101"
        assert point.value == 450.5
        assert point.unit == UnitOfMeasure.CELSIUS
        assert point.quality == QualityCode.GOOD

    def test_telemetry_point_timestamp_utc(self) -> None:
        """Test timestamp conversion to UTC."""
        naive_dt = datetime(2024, 1, 15, 12, 0, 0)
        point = TelemetryPoint(
            tag_id="T-101",
            value=100.0,
            unit=UnitOfMeasure.CELSIUS,
            timestamp=naive_dt,
        )

        assert point.timestamp.tzinfo == timezone.utc

    def test_telemetry_normalized_event(self) -> None:
        """Test complete telemetry normalized event."""
        now = datetime.now(timezone.utc)
        points = [
            TelemetryPoint(
                tag_id=f"T-{i}",
                value=float(i * 100),
                unit=UnitOfMeasure.CELSIUS,
                timestamp=now,
            )
            for i in range(1, 6)
        ]

        event = TelemetryNormalizedEvent(
            source_system="opc-ua-collector-01",
            points=points,
            collection_timestamp=now,
            batch_id="batch-001",
            sequence_number=1,
        )

        assert event.source_system == "opc-ua-collector-01"
        assert len(event.points) == 5
        assert event.data_hash is not None
        assert len(event.data_hash) == 64

    def test_telemetry_event_requires_points(self) -> None:
        """Test that telemetry event requires at least one point."""
        now = datetime.now(timezone.utc)

        with pytest.raises(ValidationError):
            TelemetryNormalizedEvent(
                source_system="test",
                points=[],  # Empty - should fail
                collection_timestamp=now,
                batch_id="batch-001",
                sequence_number=1,
            )


class TestDispatchPlanSchemas:
    """Tests for dispatch plan topic schemas."""

    def test_load_allocation_valid(self) -> None:
        """Test valid load allocation creation."""
        allocation = LoadAllocation(
            equipment_id="boiler-01",
            load_mw=50.0,
            min_load_mw=10.0,
            max_load_mw=100.0,
            efficiency_percent=92.5,
            emissions_rate_kgco2_mwh=180.0,
            fuel_type="natural_gas",
            ramp_rate_mw_min=5.0,
            marginal_cost_usd_mwh=45.0,
        )

        assert allocation.equipment_id == "boiler-01"
        assert allocation.load_mw == 50.0
        assert allocation.efficiency_percent == 92.5

    def test_expected_impact(self) -> None:
        """Test expected impact calculation."""
        impact = ExpectedImpact(
            total_cost_usd=50000.0,
            total_emissions_tco2=250.0,
            average_efficiency_percent=91.5,
            cost_savings_usd=5000.0,
            emissions_reduction_tco2=25.0,
            reliability_score=0.98,
        )

        assert impact.total_cost_usd == 50000.0
        assert impact.reliability_score == 0.98

    def test_dispatch_plan_event(self) -> None:
        """Test complete dispatch plan event."""
        now = datetime.now(timezone.utc)
        allocations = [
            LoadAllocation(
                equipment_id=f"boiler-{i}",
                load_mw=float(i * 20),
                min_load_mw=10.0,
                max_load_mw=100.0,
                efficiency_percent=90.0,
                emissions_rate_kgco2_mwh=180.0,
                fuel_type="natural_gas",
                ramp_rate_mw_min=5.0,
                marginal_cost_usd_mwh=45.0,
            )
            for i in range(1, 4)
        ]

        event = DispatchPlanEvent(
            plan_id="plan-001",
            horizon_start=now,
            horizon_end=now + timedelta(hours=24),
            allocations=allocations,
            solver_status=SolverStatus.OPTIMAL,
            solver_gap_percent=0.01,
            solver_time_seconds=2.5,
            expected_impact=ExpectedImpact(
                total_cost_usd=50000.0,
                total_emissions_tco2=250.0,
                average_efficiency_percent=91.5,
            ),
            demand_mw=120.0,
            created_by="milp-optimizer",
        )

        assert event.plan_id == "plan-001"
        assert event.solver_status == SolverStatus.OPTIMAL
        assert len(event.allocations) == 3


class TestRecommendationSchemas:
    """Tests for action recommendation topic schemas."""

    def test_setpoint_bound(self) -> None:
        """Test setpoint bound constraints."""
        bound = SetpointBound(
            min_value=350.0,
            max_value=550.0,
            safety_min=300.0,
            safety_max=600.0,
            soft_min=380.0,
            soft_max=520.0,
        )

        assert bound.min_value == 350.0
        assert bound.safety_max == 600.0

    def test_setpoint_recommendation(self) -> None:
        """Test setpoint recommendation with rationale."""
        recommendation = SetpointRecommendation(
            tag_id="T-101.SP",
            current_value=450.0,
            recommended_value=465.0,
            unit=UnitOfMeasure.CELSIUS,
            bounds=SetpointBound(
                min_value=350.0,
                max_value=550.0,
            ),
            confidence=0.92,
            rationale="Increase temperature to improve efficiency based on current load profile",
            expected_benefit={"efficiency_gain_percent": 1.5, "cost_savings_usd": 250.0},
            requires_approval=False,
        )

        assert recommendation.confidence == 0.92
        assert recommendation.expected_benefit["efficiency_gain_percent"] == 1.5

    def test_action_recommendation_event(self) -> None:
        """Test complete action recommendation event."""
        now = datetime.now(timezone.utc)
        recommendations = [
            SetpointRecommendation(
                tag_id=f"T-{i}.SP",
                current_value=float(400 + i * 10),
                recommended_value=float(405 + i * 10),
                unit=UnitOfMeasure.CELSIUS,
                bounds=SetpointBound(min_value=350.0, max_value=550.0),
                confidence=0.9,
                rationale="Optimization recommendation",
            )
            for i in range(3)
        ]

        event = ActionRecommendationEvent(
            recommendation_id="rec-001",
            recommendations=recommendations,
            triggered_by="dispatch_plan:plan-001",
            trigger_timestamp=now,
            overall_confidence=0.88,
            expires_at=now + timedelta(minutes=15),
            execution_mode="semi-auto",
        )

        assert event.recommendation_id == "rec-001"
        assert len(event.recommendations) == 3
        assert event.execution_mode == "semi-auto"


class TestSafetySchemas:
    """Tests for safety event topic schemas."""

    def test_boundary_violation(self) -> None:
        """Test boundary violation details."""
        violation = BoundaryViolation(
            tag_id="T-101",
            boundary_type="high",
            limit_value=550.0,
            actual_value=565.0,
            deviation_percent=2.73,
            unit=UnitOfMeasure.CELSIUS,
            duration_seconds=15.5,
        )

        assert violation.deviation_percent == 2.73
        assert violation.duration_seconds == 15.5

    def test_blocked_write(self) -> None:
        """Test blocked write record."""
        blocked = BlockedWrite(
            tag_id="T-101.SP",
            attempted_value=600.0,
            current_value=450.0,
            block_reason="Value exceeds safety high limit (550)",
            blocked_by="SIS",
        )

        assert blocked.blocked_by == "SIS"

    def test_sis_permissive_change(self) -> None:
        """Test SIS permissive state change."""
        change = SISPermissiveChange(
            sis_id="SIS-BOILER-01",
            permissive_name="HIGH_TEMP_PERMISSIVE",
            previous_state=True,
            new_state=False,
            trigger_condition="Temperature exceeded 545C for 10 seconds",
            sil_level=2,
        )

        assert change.sil_level == 2
        assert change.new_state is False

    def test_safety_event(self) -> None:
        """Test complete safety event."""
        now = datetime.now(timezone.utc)

        event = SafetyEvent(
            event_id="safety-001",
            level=SafetyLevel.ALARM,
            event_timestamp=now,
            equipment_id="boiler-01",
            equipment_name="Main Process Boiler 01",
            area_id="area-A",
            boundary_violations=[
                BoundaryViolation(
                    tag_id="T-101",
                    boundary_type="high",
                    limit_value=550.0,
                    actual_value=565.0,
                    deviation_percent=2.73,
                    unit=UnitOfMeasure.CELSIUS,
                )
            ],
            operator_action_required=True,
            action_deadline=now + timedelta(minutes=5),
            escalation_level=2,
        )

        assert event.level == SafetyLevel.ALARM
        assert event.operator_action_required is True
        assert len(event.boundary_violations) == 1


class TestMaintenanceSchemas:
    """Tests for maintenance trigger topic schemas."""

    def test_maintenance_evidence(self) -> None:
        """Test maintenance evidence details."""
        evidence = MaintenanceEvidence(
            evidence_type="vibration_analysis",
            description="Bearing vibration exceeds threshold",
            value=0.45,
            threshold=0.35,
            unit=UnitOfMeasure.RATIO,
            confidence=0.87,
            source_tag_ids=["VIB-PUMP-01", "VIB-PUMP-02"],
            model_id="bearing-degradation-v2",
        )

        assert evidence.evidence_type == "vibration_analysis"
        assert evidence.confidence == 0.87

    def test_maintenance_trigger_event(self) -> None:
        """Test complete maintenance trigger event."""
        now = datetime.now(timezone.utc)

        event = MaintenanceTriggerEvent(
            trigger_id="maint-001",
            equipment_id="pump-01",
            equipment_name="Feed Water Pump 01",
            equipment_class="centrifugal_pump",
            failure_mode="bearing_degradation",
            failure_probability=0.75,
            remaining_useful_life_hours=168.0,
            priority=MaintenancePriority.P3_HIGH,
            evidence=[
                MaintenanceEvidence(
                    evidence_type="vibration_analysis",
                    description="Elevated vibration detected",
                    confidence=0.87,
                )
            ],
            recommended_action="Replace bearing assembly, inspect coupling alignment",
            estimated_downtime_hours=4.0,
            estimated_cost_usd=2500.0,
            cost_of_failure_usd=150000.0,
            spare_parts_required=["bearing-6205-2RS", "seal-kit-01"],
            skills_required=["mechanical", "alignment"],
            window_start=now + timedelta(days=2),
            window_end=now + timedelta(days=5),
        )

        assert event.failure_probability == 0.75
        assert event.priority == MaintenancePriority.P3_HIGH
        assert event.cost_of_failure_usd == 150000.0


class TestExplainabilitySchemas:
    """Tests for explainability report topic schemas."""

    def test_feature_contribution(self) -> None:
        """Test feature contribution details."""
        contribution = FeatureContribution(
            feature_name="ambient_temperature",
            feature_value=28.5,
            contribution=0.15,
            contribution_percent=12.5,
            direction="positive",
            unit=UnitOfMeasure.CELSIUS,
        )

        assert contribution.contribution == 0.15
        assert contribution.direction == "positive"

    def test_shap_summary(self) -> None:
        """Test SHAP summary details."""
        summary = SHAPSummary(
            method="TreeSHAP",
            base_value=45.0,
            output_value=52.3,
            feature_contributions=[
                FeatureContribution(
                    feature_name="load",
                    feature_value=75.0,
                    contribution=4.2,
                    contribution_percent=57.5,
                    direction="positive",
                ),
                FeatureContribution(
                    feature_name="ambient_temp",
                    feature_value=28.0,
                    contribution=3.1,
                    contribution_percent=42.5,
                    direction="positive",
                ),
            ],
        )

        assert summary.output_value == 52.3
        assert len(summary.feature_contributions) == 2

    def test_uncertainty_quantification(self) -> None:
        """Test uncertainty quantification details."""
        uncertainty = UncertaintyQuantification(
            point_estimate=52.3,
            std_deviation=2.1,
            confidence_level=0.95,
            lower_bound=48.2,
            upper_bound=56.4,
            method="bootstrap",
            num_samples=1000,
            epistemic_uncertainty=1.5,
            aleatoric_uncertainty=1.4,
        )

        assert uncertainty.confidence_level == 0.95
        assert uncertainty.epistemic_uncertainty == 1.5

    def test_explainability_report_event(self) -> None:
        """Test complete explainability report event."""
        now = datetime.now(timezone.utc)

        event = ExplainabilityReportEvent(
            report_id="explain-001",
            model_id="load-optimizer-v2",
            model_version="2.3.1",
            prediction_type="load_allocation",
            prediction_timestamp=now,
            input_snapshot={"load": 75.0, "ambient_temp": 28.0, "demand": 100.0},
            prediction_value=52.3,
            prediction_unit=UnitOfMeasure.MW,
            shap_summary=SHAPSummary(
                base_value=45.0,
                output_value=52.3,
                feature_contributions=[],
            ),
            uncertainty=UncertaintyQuantification(
                point_estimate=52.3,
                std_deviation=2.1,
                confidence_level=0.95,
                lower_bound=48.2,
                upper_bound=56.4,
                method="bootstrap",
                num_samples=1000,
            ),
            model_confidence=0.92,
            human_readable_explanation="Load allocation of 52.3 MW recommended based on current demand and efficiency optimization",
        )

        assert event.model_confidence == 0.92
        assert event.shap_summary is not None


class TestAuditLogSchemas:
    """Tests for audit log topic schemas."""

    def test_audit_log_event_create(self) -> None:
        """Test audit log event for create action."""
        now = datetime.now(timezone.utc)

        event = AuditLogEvent(
            audit_id="audit-001",
            action=AuditAction.CREATE,
            action_timestamp=now,
            actor_id="milp-optimizer",
            actor_type="service",
            resource_type="dispatch_plan",
            resource_id="plan-001",
            resource_name="24-Hour Dispatch Plan",
            correlation_id="corr-001",
            new_state={"plan_id": "plan-001", "status": "created"},
            outcome="success",
            compliance_tags=["ISO50001", "internal_audit"],
        )

        assert event.action == AuditAction.CREATE
        assert event.outcome == "success"
        assert "ISO50001" in event.compliance_tags

    def test_audit_log_event_update(self) -> None:
        """Test audit log event for update action."""
        now = datetime.now(timezone.utc)

        event = AuditLogEvent(
            audit_id="audit-002",
            action=AuditAction.UPDATE,
            action_timestamp=now,
            actor_id="operator-john",
            actor_type="user",
            actor_ip="192.168.1.100",
            resource_type="setpoint",
            resource_id="T-101.SP",
            correlation_id="corr-002",
            previous_state={"value": 450.0},
            new_state={"value": 465.0},
            changes={"value": {"from": 450.0, "to": 465.0}},
            outcome="success",
            session_id="session-abc123",
        )

        assert event.action == AuditAction.UPDATE
        assert event.previous_state is not None
        assert event.new_state is not None

    def test_audit_log_event_failure(self) -> None:
        """Test audit log event for failed action."""
        now = datetime.now(timezone.utc)

        event = AuditLogEvent(
            audit_id="audit-003",
            action=AuditAction.EXECUTE,
            action_timestamp=now,
            actor_id="recommendation-engine",
            actor_type="service",
            resource_type="setpoint_change",
            resource_id="T-101.SP",
            correlation_id="corr-003",
            outcome="failure",
            error_code="SIS_BLOCK",
            error_message="Setpoint change blocked by Safety Instrumented System",
        )

        assert event.outcome == "failure"
        assert event.error_code == "SIS_BLOCK"


class TestTopicSchemaRegistry:
    """Tests for topic schema registry."""

    def test_get_schema(self) -> None:
        """Test getting schema by topic name."""
        schema = TopicSchemaRegistry.get_schema("gl001.telemetry.normalized")
        assert schema == TelemetryNormalizedEvent

        schema = TopicSchemaRegistry.get_schema("gl001.safety.events")
        assert schema == SafetyEvent

    def test_get_schema_unknown_topic(self) -> None:
        """Test getting schema for unknown topic raises error."""
        with pytest.raises(KeyError):
            TopicSchemaRegistry.get_schema("gl001.unknown.topic")

    def test_validate_payload(self) -> None:
        """Test payload validation against topic schema."""
        now = datetime.now(timezone.utc)
        payload = {
            "source_system": "test",
            "points": [
                {
                    "tag_id": "T-101",
                    "value": 450.5,
                    "unit": "degC",
                    "timestamp": now.isoformat(),
                }
            ],
            "collection_timestamp": now.isoformat(),
            "batch_id": "batch-001",
            "sequence_number": 1,
        }

        validated = TopicSchemaRegistry.validate_payload(
            "gl001.telemetry.normalized",
            payload,
        )

        assert isinstance(validated, TelemetryNormalizedEvent)
        assert validated.batch_id == "batch-001"

    def test_list_topics(self) -> None:
        """Test listing all registered topics."""
        topics = TopicSchemaRegistry.list_topics()

        assert "gl001.telemetry.normalized" in topics
        assert "gl001.plan.dispatch" in topics
        assert "gl001.safety.events" in topics
        assert "gl001.audit.log" in topics
        assert len(topics) == 7

    def test_get_json_schema(self) -> None:
        """Test getting JSON schema for a topic."""
        json_schema = TopicSchemaRegistry.get_json_schema("gl001.telemetry.normalized")

        assert "properties" in json_schema
        assert "source_system" in json_schema["properties"]
        assert "points" in json_schema["properties"]

    def test_get_avro_schema(self) -> None:
        """Test getting Avro schema for a topic."""
        avro_schema = TopicSchemaRegistry.get_avro_schema("gl001.telemetry.normalized")

        assert avro_schema["type"] == "record"
        assert avro_schema["name"] == "TelemetryNormalizedEvent"
        assert avro_schema["namespace"] == "com.greenlang.gl001"
        assert "fields" in avro_schema
