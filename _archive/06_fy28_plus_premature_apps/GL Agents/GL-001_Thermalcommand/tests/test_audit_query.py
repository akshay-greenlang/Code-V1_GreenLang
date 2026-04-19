"""
Tests for Audit Query API Module

Comprehensive test coverage for:
- Query filtering and pagination
- Export formats (JSON, CSV, PDF)
- Aggregations
- Compliance reporting

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import json
import csv
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit.audit_query import (
    AuditQueryAPI,
    QueryFilter,
    QueryResult,
    QuerySortField,
    QuerySortOrder,
    ExportFormat,
    AggregationResult,
    ComplianceReport,
)

from audit.audit_logger import EnhancedAuditLogger, InMemoryStorageBackend

from audit.audit_events import (
    DecisionAuditEvent,
    ActionAuditEvent,
    SafetyAuditEvent,
    ComplianceAuditEvent,
    EventType,
    SolverStatus,
    ActionStatus,
    SafetyLevel,
    ComplianceStatus,
    RecommendedAction,
)


class TestQueryFilter:
    """Tests for QueryFilter model."""

    def test_create_query_filter(self):
        """Test creating query filter."""
        filter = QueryFilter(
            asset_id="boiler-001",
            event_type=EventType.DECISION,
            limit=50,
        )

        assert filter.asset_id == "boiler-001"
        assert filter.limit == 50

    def test_filter_time_range_validation(self):
        """Test time range validation."""
        now = datetime.now(timezone.utc)

        with pytest.raises(ValueError, match="end_time must be after start_time"):
            QueryFilter(
                start_time=now,
                end_time=now - timedelta(hours=1),
            )

    def test_apply_time_range_days(self):
        """Test applying time_range_days."""
        filter = QueryFilter(time_range_days=7)
        filter.apply_time_range()

        expected_start = datetime.now(timezone.utc) - timedelta(days=7)

        # Allow 1 second tolerance
        assert abs((filter.start_time - expected_start).total_seconds()) < 1

    def test_filter_defaults(self):
        """Test filter defaults."""
        filter = QueryFilter()

        assert filter.limit == 100
        assert filter.offset == 0
        assert filter.sort_by == QuerySortField.TIMESTAMP
        assert filter.sort_order == QuerySortOrder.DESC


class TestAuditQueryAPI:
    """Tests for AuditQueryAPI."""

    @pytest.fixture
    def populated_logger(self):
        """Create logger with sample events."""
        logger = EnhancedAuditLogger()
        now = datetime.now(timezone.utc)

        # Create decision events
        for i in range(5):
            event = DecisionAuditEvent(
                correlation_id=f"corr-{i}",
                asset_id=f"boiler-{i % 2}",
                facility_id="plant-001",
                ingestion_timestamp=now - timedelta(hours=i),
                decision_timestamp=now - timedelta(hours=i) + timedelta(seconds=5),
                constraint_set_id="cs-001",
                constraint_set_version="1.0.0",
                safety_boundary_policy_version="2.0.0",
                solver_status=SolverStatus.OPTIMAL,
                solve_time_ms=150.5,
                objective_value=125000.50,
            )
            logger.log_decision(event)

        # Create safety events
        for i in range(3):
            event = SafetyAuditEvent(
                correlation_id=f"safety-corr-{i}",
                asset_id="boiler-0",
                boundary_id=f"TEMP_HIGH_{i}",
                boundary_name=f"High Temp {i}",
                boundary_version="1.0.0",
                safety_level=SafetyLevel.ALARM if i == 0 else SafetyLevel.WARNING,
                safety_category="temperature",
                tag_id="TI-001",
                current_value=855.0,
                boundary_value=850.0,
                unit="degF",
                deviation_pct=0.59,
                is_violation=i == 0,
            )
            logger.log_safety(event)

        # Create compliance events
        for i in range(2):
            event = ComplianceAuditEvent(
                correlation_id=f"compliance-corr-{i}",
                asset_id="boiler-0",
                regulation_id="EPA_40_CFR_98",
                regulation_name="GHG Reporting",
                regulation_version="2024",
                requirement_id=f"REQ-{i}",
                requirement_description="Test requirement",
                compliance_status=ComplianceStatus.COMPLIANT if i == 0 else ComplianceStatus.NON_COMPLIANT,
                check_type="emission_limit",
                check_method="continuous",
                check_timestamp=now,
            )
            logger.log_compliance(event)

        return logger

    @pytest.fixture
    def query_api(self, populated_logger):
        """Create query API with populated logger."""
        return AuditQueryAPI(populated_logger)

    def test_basic_query(self, query_api):
        """Test basic query without filters."""
        filter = QueryFilter()
        result = query_api.query(filter)

        assert isinstance(result, QueryResult)
        assert result.total_count == 10  # 5 decisions + 3 safety + 2 compliance
        assert len(result.events) == 10

    def test_query_by_asset(self, query_api):
        """Test querying by asset ID."""
        filter = QueryFilter(asset_id="boiler-0")
        result = query_api.query(filter)

        assert all(e["asset_id"] == "boiler-0" for e in result.events)

    def test_query_by_event_type(self, query_api):
        """Test querying by event type."""
        filter = QueryFilter(event_type=EventType.DECISION)
        result = query_api.query(filter)

        assert len(result.events) == 5
        assert all(e["event_type"] == "DECISION" for e in result.events)

    def test_query_by_multiple_types(self, query_api):
        """Test querying by multiple event types."""
        filter = QueryFilter(event_types=[EventType.DECISION, EventType.SAFETY])
        result = query_api.query(filter)

        assert len(result.events) == 8  # 5 decisions + 3 safety

    def test_query_pagination(self, query_api):
        """Test query pagination."""
        filter = QueryFilter(limit=3, offset=0)
        result1 = query_api.query(filter)

        filter = QueryFilter(limit=3, offset=3)
        result2 = query_api.query(filter)

        assert len(result1.events) == 3
        assert len(result2.events) == 3
        assert result1.has_more is True

    def test_query_sorting(self, query_api):
        """Test query sorting."""
        filter = QueryFilter(
            event_type=EventType.DECISION,
            sort_by=QuerySortField.TIMESTAMP,
            sort_order=QuerySortOrder.ASC,
        )
        result = query_api.query(filter)

        timestamps = [e["timestamp"] for e in result.events]
        assert timestamps == sorted(timestamps)

    def test_query_decisions(self, query_api):
        """Test querying decisions specifically."""
        decisions = query_api.query_decisions(
            asset_id="boiler-0",
            solver_status=SolverStatus.OPTIMAL,
        )

        assert all(isinstance(d, DecisionAuditEvent) for d in decisions)
        assert all(d.solver_status == SolverStatus.OPTIMAL for d in decisions)

    def test_query_safety_events(self, query_api):
        """Test querying safety events."""
        safety_events = query_api.query_safety_events(
            safety_level=SafetyLevel.ALARM,
            is_violation=True,
        )

        assert len(safety_events) == 1
        assert all(isinstance(e, SafetyAuditEvent) for e in safety_events)


class TestAggregations:
    """Tests for aggregation queries."""

    @pytest.fixture
    def query_api(self, populated_logger):
        """Create query API with populated logger."""
        return AuditQueryAPI(populated_logger)

    @pytest.fixture
    def populated_logger(self):
        """Create logger with sample events."""
        logger = EnhancedAuditLogger()
        now = datetime.now(timezone.utc)

        for i in range(5):
            event = DecisionAuditEvent(
                correlation_id=f"corr-{i}",
                asset_id=f"boiler-{i % 2}",
                ingestion_timestamp=now,
                decision_timestamp=now,
                constraint_set_id="cs-001",
                constraint_set_version="1.0.0",
                safety_boundary_policy_version="2.0.0",
                solver_status=SolverStatus.OPTIMAL,
                solve_time_ms=150.5,
                objective_value=125000.50,
            )
            logger.log_decision(event)

        for i in range(3):
            event = SafetyAuditEvent(
                correlation_id=f"safety-{i}",
                asset_id="boiler-0",
                boundary_id="TEMP_HIGH",
                boundary_name="High Temp",
                boundary_version="1.0.0",
                safety_level=SafetyLevel.ALARM,
                safety_category="temperature",
                tag_id="TI-001",
                current_value=855.0,
                boundary_value=850.0,
                unit="degF",
                deviation_pct=0.59,
                is_violation=True,
            )
            logger.log_safety(event)

        return logger

    def test_aggregate_by_type(self, query_api):
        """Test aggregating by event type."""
        result = query_api.aggregate_by_type()

        assert isinstance(result, AggregationResult)
        assert result.group_by == "event_type"
        assert result.aggregations.get("DECISION") == 5
        assert result.aggregations.get("SAFETY") == 3

    def test_aggregate_by_asset(self, query_api):
        """Test aggregating by asset."""
        result = query_api.aggregate_by_asset(
            asset_ids=["boiler-0", "boiler-1"]
        )

        assert result.group_by == "asset_id"
        # boiler-0: 3 decisions + 3 safety = 6
        # boiler-1: 2 decisions = 2
        assert result.aggregations.get("boiler-0") == 6
        assert result.aggregations.get("boiler-1") == 2


class TestExport:
    """Tests for export functionality."""

    @pytest.fixture
    def query_api(self):
        """Create query API with sample data."""
        logger = EnhancedAuditLogger()
        now = datetime.now(timezone.utc)

        for i in range(3):
            event = DecisionAuditEvent(
                correlation_id=f"corr-{i}",
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
            logger.log_decision(event)

        return AuditQueryAPI(logger)

    def test_export_to_json(self, query_api, tmp_path):
        """Test exporting to JSON."""
        filter = QueryFilter()
        result = query_api.query(filter)

        output_path = str(tmp_path / "export.json")
        path = query_api.export_to_json(result.events, output_path)

        assert Path(path).exists()

        with open(path, "r") as f:
            data = json.load(f)

        assert len(data) == 3

    def test_export_to_csv(self, query_api, tmp_path):
        """Test exporting to CSV."""
        filter = QueryFilter()
        result = query_api.query(filter)

        output_path = str(tmp_path / "export.csv")
        path = query_api.export_to_csv(result.events, output_path)

        assert Path(path).exists()

        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

    def test_export_to_csv_custom_fields(self, query_api, tmp_path):
        """Test exporting to CSV with custom fields."""
        filter = QueryFilter()
        result = query_api.query(filter)

        output_path = str(tmp_path / "export.csv")
        fields = ["event_id", "correlation_id", "asset_id"]
        path = query_api.export_to_csv(result.events, output_path, fields=fields)

        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert set(rows[0].keys()) == set(fields)

    def test_export_empty_to_csv(self, query_api, tmp_path):
        """Test exporting empty results to CSV."""
        output_path = str(tmp_path / "export.csv")
        path = query_api.export_to_csv([], output_path)

        assert Path(path).exists()

    def test_export_to_pdf(self, query_api, tmp_path):
        """Test exporting to PDF (text file for now)."""
        filter = QueryFilter()
        result = query_api.query(filter)

        output_path = str(tmp_path / "export.pdf")
        path = query_api.export_to_pdf(
            result.events,
            output_path,
            title="Test Audit Report"
        )

        # Currently outputs as .txt
        assert Path(path).exists()

    def test_export_via_api(self, query_api, tmp_path):
        """Test export through unified API."""
        filter = QueryFilter()
        result = query_api.query(filter)

        output_path = str(tmp_path / "export.json")
        path = query_api.export(result, ExportFormat.JSON, output_path)

        assert Path(path).exists()

    def test_export_unsupported_format(self, query_api, tmp_path):
        """Test export with unsupported format raises error."""
        filter = QueryFilter()
        result = query_api.query(filter)

        output_path = str(tmp_path / "export.xlsx")

        with pytest.raises(ValueError, match="Unsupported export format"):
            query_api.export(result, ExportFormat.XLSX, output_path)


class TestComplianceReport:
    """Tests for compliance report generation."""

    @pytest.fixture
    def query_api_with_compliance(self):
        """Create query API with compliance data."""
        logger = EnhancedAuditLogger()
        now = datetime.now(timezone.utc)

        # Add decisions
        for i in range(10):
            event = DecisionAuditEvent(
                correlation_id=f"corr-{i}",
                asset_id="boiler-001",
                ingestion_timestamp=now - timedelta(days=i),
                decision_timestamp=now - timedelta(days=i),
                constraint_set_id="cs-001",
                constraint_set_version="1.0.0",
                safety_boundary_policy_version="2.0.0",
                solver_status=SolverStatus.OPTIMAL,
                solve_time_ms=150.5,
                objective_value=125000.50,
            )
            logger.log_decision(event)

        # Add safety events with violations
        for i in range(5):
            event = SafetyAuditEvent(
                correlation_id=f"safety-{i}",
                asset_id="boiler-001",
                boundary_id="TEMP_HIGH",
                boundary_name="High Temp",
                boundary_version="1.0.0",
                safety_level=SafetyLevel.ALARM if i < 2 else SafetyLevel.WARNING,
                safety_category="temperature",
                tag_id="TI-001",
                current_value=855.0,
                boundary_value=850.0,
                unit="degF",
                deviation_pct=0.59,
                is_violation=i < 2,
            )
            logger.log_safety(event)

        # Add compliance events
        for i in range(3):
            event = ComplianceAuditEvent(
                correlation_id=f"compliance-{i}",
                asset_id="boiler-001",
                regulation_id="EPA_40_CFR_98",
                regulation_name="GHG Reporting",
                regulation_version="2024",
                requirement_id=f"REQ-{i}",
                requirement_description="Test",
                compliance_status=ComplianceStatus.COMPLIANT if i < 2 else ComplianceStatus.NON_COMPLIANT,
                check_type="emission_limit",
                check_method="continuous",
                check_timestamp=now,
            )
            logger.log_compliance(event)

        return AuditQueryAPI(logger)

    def test_generate_compliance_report(self, query_api_with_compliance):
        """Test generating compliance report."""
        now = datetime.now(timezone.utc)
        report = query_api_with_compliance.generate_compliance_report(
            period_start=now - timedelta(days=30),
            period_end=now,
        )

        assert isinstance(report, ComplianceReport)
        assert report.total_decisions == 10
        assert report.total_safety_events == 5
        assert report.total_violations == 2
        assert report.compliance_checks == 3
        assert report.compliance_failures == 1

    def test_compliance_report_with_facility(self, query_api_with_compliance):
        """Test compliance report filtered by facility."""
        now = datetime.now(timezone.utc)
        report = query_api_with_compliance.generate_compliance_report(
            period_start=now - timedelta(days=30),
            period_end=now,
            facility_id="plant-001",
        )

        assert report.facility_id == "plant-001"

    def test_compliance_report_summaries(self, query_api_with_compliance):
        """Test compliance report includes summaries."""
        now = datetime.now(timezone.utc)
        report = query_api_with_compliance.generate_compliance_report(
            period_start=now - timedelta(days=30),
            period_end=now,
        )

        assert "DECISION" in report.events_summary
        assert "SAFETY" in report.events_summary
        assert "ALARM" in report.safety_summary or "WARNING" in report.safety_summary
        assert "COMPLIANT" in report.compliance_summary


class TestChainIntegrity:
    """Tests for chain integrity verification via query API."""

    @pytest.fixture
    def query_api(self):
        """Create query API with data."""
        logger = EnhancedAuditLogger()
        now = datetime.now(timezone.utc)

        for i in range(5):
            event = DecisionAuditEvent(
                correlation_id=f"corr-{i}",
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
            logger.log_decision(event)

        return AuditQueryAPI(logger)

    def test_verify_chain_integrity(self, query_api):
        """Test chain integrity verification."""
        result = query_api.verify_chain_integrity()

        assert result["is_valid"] is True
        assert result["error"] is None
        assert "chain_statistics" in result


class TestAuditTrail:
    """Tests for audit trail retrieval."""

    @pytest.fixture
    def query_api_with_trail(self):
        """Create query API with related events."""
        logger = EnhancedAuditLogger()
        now = datetime.now(timezone.utc)

        # Create decision
        decision = DecisionAuditEvent(
            correlation_id="trail-corr-001",
            asset_id="boiler-001",
            ingestion_timestamp=now,
            decision_timestamp=now,
            constraint_set_id="cs-001",
            constraint_set_version="1.0.0",
            safety_boundary_policy_version="2.0.0",
            solver_status=SolverStatus.OPTIMAL,
            solve_time_ms=150.5,
            objective_value=125000.50,
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
        logger.log_decision(decision)

        # Create action
        action = ActionAuditEvent(
            correlation_id="trail-corr-001",
            asset_id="boiler-001",
            decision_event_id=str(decision.event_id),
            decision_correlation_id="trail-corr-001",
            action=decision.recommended_actions[0],
            action_status=ActionStatus.EXECUTED,
            recommended_timestamp=now,
            actuation_timestamp=now + timedelta(seconds=30),
        )
        logger.log_action(action)

        return AuditQueryAPI(logger)

    def test_get_audit_trail(self, query_api_with_trail):
        """Test getting audit trail."""
        trail = query_api_with_trail.get_audit_trail("trail-corr-001")

        assert trail["correlation_id"] == "trail-corr-001"
        assert trail["total_events"] == 2
        assert len(trail["decisions"]) == 1
        assert len(trail["actions"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
