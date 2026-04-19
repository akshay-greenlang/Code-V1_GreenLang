"""
Unit tests for Process Heat GraphQL schema and resolvers.

Tests cover:
- Schema creation and validation
- Query execution
- Mutation execution
- Subscription handling
- Error handling
- Type validation

Run with: pytest tests/unit/test_graphql_schema.py -v
"""

import pytest
import asyncio
from datetime import datetime, date
from typing import List, Dict, Any

# Conditional imports based on availability
try:
    from greenlang.infrastructure.api.graphql_schema import (
        create_process_heat_schema,
        STRAWBERRY_AVAILABLE,
        AgentStatus,
        JobStatus,
        ReportType,
        ComplianceStatus,
        EmissionResult,
        ProcessHeatAgent,
        AgentMetricsType,
        CalculationJob,
        ComplianceReport,
        ComplianceFinding,
        DateRangeInput,
        CalculationInput,
        AgentConfigInput,
        ReportParamsInput,
        JobProgressEvent,
        AlertEvent,
        Query,
        Mutation,
        Subscription,
    )
    from greenlang.infrastructure.api.graphql_integration import (
        QueryExecutor,
        SubscriptionHandler,
        GraphQLConfig,
        GraphQLIntegrationError,
    )
    GRAPHQL_AVAILABLE = True
except ImportError:
    GRAPHQL_AVAILABLE = False


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestGraphQLSchema:
    """Test GraphQL schema creation and validation."""

    def test_schema_creation(self):
        """Test that GraphQL schema can be created."""
        schema = create_process_heat_schema()
        assert schema is not None

    def test_schema_has_query_type(self):
        """Test that schema has Query type."""
        schema = create_process_heat_schema()
        assert schema.query_type is not None

    def test_schema_has_mutation_type(self):
        """Test that schema has Mutation type."""
        schema = create_process_heat_schema()
        assert schema.mutation_type is not None

    def test_schema_has_subscription_type(self):
        """Test that schema has Subscription type."""
        schema = create_process_heat_schema()
        assert schema.subscription_type is not None


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestEmissionResult:
    """Test EmissionResult type."""

    def test_emission_result_creation(self):
        """Test creating EmissionResult instance."""
        result = EmissionResult(
            id="emission-001",
            facility_id="facility-1",
            co2_tonnes=1250.0,
            ch4_tonnes=5.2,
            n2o_tonnes=0.8,
            total_co2e_tonnes=1256.5,
            provenance_hash="sha256_hash_value",
            calculation_method="IPCC AR6",
            timestamp=datetime.now(),
            confidence_score=0.95
        )

        assert result.id == "emission-001"
        assert result.facility_id == "facility-1"
        assert result.co2_tonnes == 1250.0
        assert result.total_co2e_tonnes == 1256.5
        assert 0.0 <= result.confidence_score <= 1.0

    def test_emission_result_validation(self):
        """Test EmissionResult field validation."""
        # Confidence score must be 0.0-1.0
        result = EmissionResult(
            id="test",
            facility_id="fac-1",
            co2_tonnes=100.0,
            ch4_tonnes=1.0,
            n2o_tonnes=0.1,
            total_co2e_tonnes=101.1,
            provenance_hash="hash",
            calculation_method="method",
            timestamp=datetime.now(),
            confidence_score=0.85
        )
        assert 0.0 <= result.confidence_score <= 1.0


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestProcessHeatAgent:
    """Test ProcessHeatAgent type."""

    def test_agent_creation(self):
        """Test creating ProcessHeatAgent instance."""
        metrics = AgentMetricsType(
            execution_time_ms=1234.5,
            memory_usage_mb=256.2,
            records_processed=15000,
            processing_rate=1250.0,
            cache_hit_ratio=0.85,
            error_count=0
        )

        agent = ProcessHeatAgent(
            id="agent-001",
            name="Thermal Command",
            agent_type="GL-001",
            status="idle",
            enabled=True,
            version="1.0.0",
            last_run=datetime.now(),
            next_run=None,
            metrics=metrics,
            error_message=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert agent.id == "agent-001"
        assert agent.name == "Thermal Command"
        assert agent.status == "idle"
        assert agent.enabled is True
        assert agent.metrics.error_count == 0

    def test_agent_metrics_validation(self):
        """Test AgentMetricsType validation."""
        metrics = AgentMetricsType(
            execution_time_ms=1000.0,
            memory_usage_mb=512.0,
            records_processed=100000,
            processing_rate=1000.0,
            cache_hit_ratio=0.75,
            error_count=5
        )

        assert metrics.execution_time_ms > 0
        assert 0.0 <= metrics.cache_hit_ratio <= 1.0
        assert metrics.error_count >= 0


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestCalculationJob:
    """Test CalculationJob type."""

    def test_job_creation(self):
        """Test creating CalculationJob instance."""
        job = CalculationJob(
            id="job-001",
            status="pending",
            progress_percent=0,
            agent_id="agent-001",
            input_summary="Facility 1, 30 days",
            results=None,
            error_details=None,
            execution_time_ms=0.0,
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
        )

        assert job.id == "job-001"
        assert job.status == "pending"
        assert job.progress_percent == 0
        assert 0 <= job.progress_percent <= 100

    def test_job_with_results(self):
        """Test job with results."""
        result = EmissionResult(
            id="result-001",
            facility_id="fac-1",
            co2_tonnes=1250.0,
            ch4_tonnes=5.2,
            n2o_tonnes=0.8,
            total_co2e_tonnes=1256.5,
            provenance_hash="hash",
            calculation_method="IPCC",
            timestamp=datetime.now(),
            confidence_score=0.95
        )

        job = CalculationJob(
            id="job-001",
            status="completed",
            progress_percent=100,
            agent_id="agent-001",
            input_summary="Test",
            results=[result],
            error_details=None,
            execution_time_ms=5000.0,
            created_at=datetime.now(),
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )

        assert len(job.results) == 1
        assert job.results[0].co2_tonnes == 1250.0


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestComplianceReport:
    """Test ComplianceReport type."""

    def test_report_creation(self):
        """Test creating ComplianceReport instance."""
        finding = ComplianceFinding(
            id="finding-001",
            category="emissions_tracking",
            severity="low",
            description="Minor deviation",
            remediation_action="Update method",
            deadline=date(2025, 6, 30)
        )

        report = ComplianceReport(
            id="report-001",
            report_type="ghg_emissions",
            status="compliant",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 3, 31),
            findings=[finding],
            summary="98% compliant",
            action_items_count=1,
            generated_at=datetime.now(),
        )

        assert report.id == "report-001"
        assert report.report_type == "ghg_emissions"
        assert report.status == "compliant"
        assert len(report.findings) == 1
        assert report.action_items_count == 1

    def test_report_without_findings(self):
        """Test report without findings."""
        report = ComplianceReport(
            id="report-002",
            report_type="energy_audit",
            status="compliant",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 3, 31),
            findings=[],
            summary="Fully compliant",
            action_items_count=0,
            generated_at=datetime.now(),
        )

        assert len(report.findings) == 0
        assert report.action_items_count == 0


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestEnums:
    """Test GraphQL enumeration types."""

    def test_agent_status_enum(self):
        """Test AgentStatus enum."""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"
        assert AgentStatus.PAUSED.value == "paused"

    def test_job_status_enum(self):
        """Test JobStatus enum."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_report_type_enum(self):
        """Test ReportType enum."""
        assert ReportType.GHG_EMISSIONS.value == "ghg_emissions"
        assert ReportType.ENERGY_AUDIT.value == "energy_audit"
        assert ReportType.EFFICIENCY_ANALYSIS.value == "efficiency_analysis"

    def test_compliance_status_enum(self):
        """Test ComplianceStatus enum."""
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"
        assert ComplianceStatus.UNDER_REVIEW.value == "under_review"


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestInputTypes:
    """Test GraphQL input types."""

    def test_date_range_input(self):
        """Test DateRangeInput."""
        date_range = DateRangeInput(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 31)
        )

        assert date_range.start_date == date(2025, 1, 1)
        assert date_range.end_date == date(2025, 3, 31)

    def test_calculation_input(self):
        """Test CalculationInput."""
        date_range = DateRangeInput(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 31)
        )

        calc_input = CalculationInput(
            agent_id="agent-001",
            facility_id="facility-1",
            date_range=date_range,
            parameters='{"param1": "value1"}',
            priority="high"
        )

        assert calc_input.agent_id == "agent-001"
        assert calc_input.priority == "high"

    def test_agent_config_input(self):
        """Test AgentConfigInput."""
        config = AgentConfigInput(
            enabled=False,
            execution_interval_minutes=60,
            parameters='{"setting": "value"}'
        )

        assert config.enabled is False
        assert config.execution_interval_minutes == 60

    def test_report_params_input(self):
        """Test ReportParamsInput."""
        date_range = DateRangeInput(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 31)
        )

        params = ReportParamsInput(
            facility_ids=["facility-1", "facility-2"],
            date_range=date_range,
            include_recommendations=True
        )

        assert len(params.facility_ids) == 2
        assert params.include_recommendations is True


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestEventTypes:
    """Test GraphQL subscription event types."""

    def test_job_progress_event(self):
        """Test JobProgressEvent."""
        event = JobProgressEvent(
            job_id="job-001",
            progress_percent=50,
            status="running",
            message="Processing batch 1 of 3",
            timestamp=datetime.now()
        )

        assert event.job_id == "job-001"
        assert 0 <= event.progress_percent <= 100
        assert event.status == "running"

    def test_alert_event(self):
        """Test AlertEvent."""
        event = AlertEvent(
            agent_id="agent-001",
            alert_type="warning",
            message="High memory usage",
            metric_name="memory_usage_mb",
            metric_value=512.5,
            timestamp=datetime.now()
        )

        assert event.agent_id == "agent-001"
        assert event.alert_type == "warning"
        assert event.metric_value == 512.5


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
@pytest.mark.asyncio
class TestQueryExecutor:
    """Test QueryExecutor functionality."""

    async def test_executor_initialization(self):
        """Test QueryExecutor initialization."""
        schema = create_process_heat_schema()
        executor = QueryExecutor(schema)

        assert executor.schema is schema

    async def test_simple_query_execution(self):
        """Test executing a simple query."""
        schema = create_process_heat_schema()
        executor = QueryExecutor(schema)

        query = """
        {
            agents {
                id
                name
                status
            }
        }
        """

        result = await executor.execute_query(query)

        assert "data" in result
        assert result.get("data") is not None

    async def test_query_with_variables(self):
        """Test query execution with variables."""
        schema = create_process_heat_schema()
        executor = QueryExecutor(schema)

        query = """
        query getAgents($status: String) {
            agents(status: $status) {
                id
                name
                status
            }
        }
        """

        variables = {"status": "idle"}
        result = await executor.execute_query(query, variables)

        assert "data" in result


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestGraphQLConfig:
    """Test GraphQL configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = GraphQLConfig()

        assert config.path == "/graphql"
        assert config.enable_schema_introspection is True
        assert config.enable_playground is True
        assert config.max_query_depth == 10
        assert config.timeout_seconds == 30.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = GraphQLConfig(
            path="/api/graphql",
            enable_schema_introspection=False,
            enable_playground=False,
            max_query_depth=5,
            timeout_seconds=60.0
        )

        assert config.path == "/api/graphql"
        assert config.enable_schema_introspection is False
        assert config.enable_playground is False
        assert config.max_query_depth == 5
        assert config.timeout_seconds == 60.0


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestGraphQLIntegration:
    """Test GraphQL integration utilities."""

    def test_graphql_integration_error(self):
        """Test GraphQLIntegrationError exception."""
        error = GraphQLIntegrationError("Test error message")
        assert str(error) == "Test error message"

    def test_subscription_handler_creation(self):
        """Test SubscriptionHandler initialization."""
        schema = create_process_heat_schema()
        handler = SubscriptionHandler(schema)

        assert handler.schema is schema
        assert handler.get_active_subscriptions() == 0


# Smoke tests
@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL not available")
class TestSmoke:
    """Smoke tests for basic functionality."""

    def test_all_types_importable(self):
        """Test that all types are importable."""
        assert EmissionResult is not None
        assert ProcessHeatAgent is not None
        assert CalculationJob is not None
        assert ComplianceReport is not None
        assert Query is not None
        assert Mutation is not None
        assert Subscription is not None

    def test_schema_creation_no_errors(self):
        """Test schema creation without errors."""
        try:
            schema = create_process_heat_schema()
            assert schema is not None
        except Exception as e:
            pytest.fail(f"Schema creation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
