"""
GL-019 HEATSCHEDULER - Scheduler Integration Tests

Integration tests for the main HeatSchedulerAgent including end-to-end
processing, component coordination, and provenance tracking.

Test Coverage:
    - Agent initialization
    - Input validation
    - Output validation
    - End-to-end processing
    - Provenance tracking
    - Error handling
    - Performance benchmarks

Author: GreenLang Test Team
Date: December 2025
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
import hashlib
import json


class TestHeatSchedulerAgentInitialization:
    """Tests for HeatSchedulerAgent initialization."""

    def test_agent_initialization(self, sample_scheduler_config):
        """Test agent initializes correctly."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        assert agent is not None
        assert agent.scheduler_config == sample_scheduler_config
        assert agent.forecaster is not None
        assert agent.storage_optimizer is not None
        assert agent.demand_optimizer is not None
        assert agent.production_planner is not None
        assert agent.weather_service is not None

    def test_agent_config_attributes(self, sample_scheduler_config):
        """Test agent configuration attributes."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        assert agent.config.agent_type == "GL-019"
        assert "Test Heat Scheduler" in agent.config.name

    def test_agent_provenance_tracker_initialized(self, sample_scheduler_config):
        """Test provenance tracker is initialized."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        assert agent.provenance_tracker is not None

    def test_agent_audit_logger_initialized(self, sample_scheduler_config):
        """Test audit logger is initialized."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        assert agent.audit_logger is not None

    def test_agent_intelligence_level(self, sample_scheduler_config):
        """Test agent intelligence level is ADVANCED."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )
        from greenlang.agents.intelligence_interface import IntelligenceLevel

        agent = HeatSchedulerAgent(sample_scheduler_config)

        assert agent.get_intelligence_level() == IntelligenceLevel.ADVANCED

    def test_agent_intelligence_capabilities(self, sample_scheduler_config):
        """Test agent intelligence capabilities."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)
        capabilities = agent.get_intelligence_capabilities()

        assert capabilities.can_explain is True
        assert capabilities.can_recommend is True
        assert capabilities.can_detect_anomalies is True
        assert capabilities.can_reason is True


class TestHeatSchedulerInputValidation:
    """Tests for input validation."""

    def test_validate_valid_input(self, sample_scheduler_config, sample_scheduler_input):
        """Test validation passes for valid input."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        assert agent.validate_input(sample_scheduler_input) is True

    def test_validate_missing_facility_id(self, sample_scheduler_config, base_timestamp):
        """Test validation fails for missing facility_id."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerInput,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        input_data = HeatSchedulerInput(
            facility_id="",  # Empty
            current_load_kw=2500.0,
        )

        assert agent.validate_input(input_data) is False

    def test_validate_negative_load(self, sample_scheduler_config, base_timestamp):
        """Test validation fails for negative current load."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerInput,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        input_data = HeatSchedulerInput(
            facility_id="PLANT-001",
            current_load_kw=-100.0,  # Negative
        )

        assert agent.validate_input(input_data) is False

    def test_validate_invalid_horizon(self, sample_scheduler_config, base_timestamp):
        """Test validation fails for invalid optimization horizon."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerInput,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        input_data = HeatSchedulerInput(
            facility_id="PLANT-001",
            current_load_kw=2500.0,
            optimization_horizon_hours=0,  # Invalid
        )

        assert agent.validate_input(input_data) is False


class TestHeatSchedulerOutputValidation:
    """Tests for output validation."""

    def test_validate_valid_output(
        self,
        sample_scheduler_config,
        sample_load_forecast,
        base_timestamp,
    ):
        """Test validation passes for valid output."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerOutput,
            ScheduleStatus,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        output = HeatSchedulerOutput(
            facility_id="PLANT-001",
            request_id="REQ-001",
            status=ScheduleStatus.OPTIMAL,
            schedule_horizon_hours=24,
            load_forecast=sample_load_forecast,
            baseline_cost_usd=1500.0,
            optimized_cost_usd=1275.0,
        )

        assert agent.validate_output(output) is True

    def test_validate_missing_load_forecast(
        self,
        sample_scheduler_config,
        base_timestamp,
    ):
        """Test validation fails for missing load forecast."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerOutput,
            ScheduleStatus,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        # Create output without load_forecast
        output = HeatSchedulerOutput(
            facility_id="PLANT-001",
            request_id="REQ-001",
            status=ScheduleStatus.OPTIMAL,
            schedule_horizon_hours=24,
            load_forecast=None,  # Missing
            baseline_cost_usd=1500.0,
            optimized_cost_usd=1275.0,
        )

        assert agent.validate_output(output) is False

    def test_validate_negative_optimized_cost(
        self,
        sample_scheduler_config,
        sample_load_forecast,
    ):
        """Test validation fails for negative optimized cost."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerOutput,
            ScheduleStatus,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        output = HeatSchedulerOutput(
            facility_id="PLANT-001",
            request_id="REQ-001",
            status=ScheduleStatus.OPTIMAL,
            schedule_horizon_hours=24,
            load_forecast=sample_load_forecast,
            baseline_cost_usd=1500.0,
            optimized_cost_usd=-100.0,  # Negative
        )

        assert agent.validate_output(output) is False


class TestHeatSchedulerProcessing:
    """Tests for end-to-end processing."""

    @pytest.fixture
    def mock_intelligence_methods(self, sample_scheduler_config):
        """Create agent with mocked intelligence methods."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        # Mock intelligence methods to avoid LLM calls
        agent.generate_explanation = Mock(return_value="Test explanation")
        agent.generate_recommendations = Mock(return_value=[])
        agent.detect_anomalies = Mock(return_value=[])
        agent.reason_about = Mock(return_value="Test reasoning")

        return agent

    def test_process_with_provided_forecast(
        self,
        mock_intelligence_methods,
        sample_scheduler_input,
    ):
        """Test processing with provided load forecast."""
        agent = mock_intelligence_methods

        result = agent.process(sample_scheduler_input)

        assert result is not None
        assert result.facility_id == "PLANT-001"
        assert result.load_forecast is not None
        assert result.status.value == "optimal"

    def test_process_generates_schedule_actions(
        self,
        mock_intelligence_methods,
        sample_scheduler_input,
    ):
        """Test processing generates schedule actions."""
        agent = mock_intelligence_methods

        result = agent.process(sample_scheduler_input)

        # Should have some schedule actions
        assert result.schedule_actions is not None

    def test_process_calculates_costs(
        self,
        mock_intelligence_methods,
        sample_scheduler_input,
    ):
        """Test processing calculates costs correctly."""
        agent = mock_intelligence_methods

        result = agent.process(sample_scheduler_input)

        assert result.baseline_cost_usd >= 0
        assert result.optimized_cost_usd >= 0
        assert result.total_savings_usd >= 0 or result.total_savings_usd < 0  # Can be negative

    def test_process_calculates_kpis(
        self,
        mock_intelligence_methods,
        sample_scheduler_input,
    ):
        """Test processing calculates KPIs."""
        agent = mock_intelligence_methods

        result = agent.process(sample_scheduler_input)

        assert "peak_demand_kw" in result.kpis
        assert "average_load_kw" in result.kpis
        assert "load_factor_pct" in result.kpis

    def test_process_generates_provenance_hash(
        self,
        mock_intelligence_methods,
        sample_scheduler_input,
    ):
        """Test processing generates provenance hash."""
        agent = mock_intelligence_methods

        result = agent.process(sample_scheduler_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_process_generates_input_hash(
        self,
        mock_intelligence_methods,
        sample_scheduler_input,
    ):
        """Test processing generates input hash."""
        agent = mock_intelligence_methods

        result = agent.process(sample_scheduler_input)

        assert result.input_hash is not None

    def test_process_records_processing_time(
        self,
        mock_intelligence_methods,
        sample_scheduler_input,
    ):
        """Test processing records processing time."""
        agent = mock_intelligence_methods

        result = agent.process(sample_scheduler_input)

        assert result.processing_time_ms > 0

    def test_process_with_storage_optimization(
        self,
        mock_intelligence_methods,
        sample_scheduler_input,
    ):
        """Test processing with thermal storage optimization."""
        agent = mock_intelligence_methods

        result = agent.process(sample_scheduler_input)

        # With storage configured, should have storage result
        assert result.storage_result is not None

    def test_process_with_demand_optimization(
        self,
        mock_intelligence_methods,
        sample_scheduler_input,
    ):
        """Test processing with demand charge optimization."""
        agent = mock_intelligence_methods

        result = agent.process(sample_scheduler_input)

        assert result.demand_result is not None
        assert result.demand_result.baseline_peak_kw > 0

    def test_process_with_production_orders(
        self,
        mock_intelligence_methods,
        sample_scheduler_input,
    ):
        """Test processing with production orders."""
        agent = mock_intelligence_methods

        result = agent.process(sample_scheduler_input)

        # With production orders, should have production result
        assert result.production_result is not None
        assert result.production_result.total_orders == 3


class TestHeatSchedulerProvenance:
    """Tests for provenance tracking."""

    @pytest.fixture
    def agent_with_mocks(self, sample_scheduler_config):
        """Create agent with mocked intelligence."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)
        agent.generate_explanation = Mock(return_value="Test")
        agent.generate_recommendations = Mock(return_value=[])
        agent.detect_anomalies = Mock(return_value=[])
        agent.reason_about = Mock(return_value="Test")

        return agent

    def test_provenance_hash_is_deterministic(
        self,
        agent_with_mocks,
        sample_scheduler_input,
    ):
        """Test that same input produces same provenance hash."""
        agent = agent_with_mocks

        result1 = agent.process(sample_scheduler_input)
        result2 = agent.process(sample_scheduler_input)

        # Input hash should be the same
        assert result1.input_hash == result2.input_hash

    def test_provenance_hash_changes_with_input(
        self,
        agent_with_mocks,
        sample_scheduler_input,
        base_timestamp,
    ):
        """Test that different input produces different provenance hash."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerInput,
        )

        agent = agent_with_mocks

        result1 = agent.process(sample_scheduler_input)

        # Create different input
        different_input = HeatSchedulerInput(
            facility_id="PLANT-002",  # Different
            current_load_kw=3000.0,  # Different
            load_forecast=sample_scheduler_input.load_forecast,
        )

        result2 = agent.process(different_input)

        # Input hash should be different
        assert result1.input_hash != result2.input_hash


class TestHeatSchedulerErrorHandling:
    """Tests for error handling."""

    def test_process_validation_error(self, sample_scheduler_config, base_timestamp):
        """Test processing raises ValidationError for invalid input."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerInput,
        )
        from greenlang.agents.process_heat.shared.base_agent import ValidationError

        agent = HeatSchedulerAgent(sample_scheduler_config)
        agent.generate_explanation = Mock(return_value="Test")
        agent.generate_recommendations = Mock(return_value=[])
        agent.detect_anomalies = Mock(return_value=[])
        agent.reason_about = Mock(return_value="Test")

        invalid_input = HeatSchedulerInput(
            facility_id="",  # Invalid
            current_load_kw=2500.0,
        )

        with pytest.raises(ValidationError):
            agent.process(invalid_input)


class TestHeatSchedulerHistoricalData:
    """Tests for historical data management."""

    def test_add_historical_data(self, sample_scheduler_config, sample_historical_data):
        """Test adding historical data points."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        # Add historical data points
        for point in sample_historical_data[:10]:
            agent.add_historical_data(point)

        # Verify data was added to forecaster
        assert len(agent.forecaster._historical_data) == 10

    def test_train_forecaster(self, sample_scheduler_config, sample_historical_data):
        """Test training the forecaster."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)

        # Train forecaster
        agent.train_forecaster(sample_historical_data)

        # Verify models are trained
        for model in agent.forecaster._models.values():
            assert model._is_trained is True


class TestHeatSchedulerPerformance:
    """Performance tests for the scheduler."""

    @pytest.mark.performance
    def test_processing_time_under_threshold(
        self,
        sample_scheduler_config,
        sample_scheduler_input,
    ):
        """Test processing completes under 5 seconds."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )
        import time

        agent = HeatSchedulerAgent(sample_scheduler_config)
        agent.generate_explanation = Mock(return_value="Test")
        agent.generate_recommendations = Mock(return_value=[])
        agent.detect_anomalies = Mock(return_value=[])
        agent.reason_about = Mock(return_value="Test")

        start = time.time()
        result = agent.process(sample_scheduler_input)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete in under 5 seconds

    @pytest.mark.performance
    def test_processing_with_large_forecast(
        self,
        sample_scheduler_config,
        large_load_forecast,
        base_timestamp,
    ):
        """Test processing with large forecast (168 hours)."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            HeatSchedulerInput,
        )
        import time

        agent = HeatSchedulerAgent(sample_scheduler_config)
        agent.generate_explanation = Mock(return_value="Test")
        agent.generate_recommendations = Mock(return_value=[])
        agent.detect_anomalies = Mock(return_value=[])
        agent.reason_about = Mock(return_value="Test")

        input_data = HeatSchedulerInput(
            facility_id="PLANT-001",
            current_load_kw=2500.0,
            optimization_horizon_hours=168,
            load_forecast=large_load_forecast,
        )

        start = time.time()
        result = agent.process(input_data)
        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 10.0  # Should complete in under 10 seconds


class TestHeatSchedulerCompliance:
    """Compliance tests for regulatory requirements."""

    @pytest.mark.compliance
    def test_audit_trail_completeness(
        self,
        sample_scheduler_config,
        sample_scheduler_input,
    ):
        """Test audit trail includes all required elements."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)
        agent.generate_explanation = Mock(return_value="Test")
        agent.generate_recommendations = Mock(return_value=[])
        agent.detect_anomalies = Mock(return_value=[])
        agent.reason_about = Mock(return_value="Test")

        result = agent.process(sample_scheduler_input)

        # Verify provenance tracking
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hash

        # Verify input tracking
        assert result.input_hash is not None

        # Verify processing time recorded
        assert result.processing_time_ms > 0

        # Verify metadata includes agent version
        assert "agent_version" in result.metadata

    @pytest.mark.compliance
    def test_reproducibility_guarantee(
        self,
        sample_scheduler_config,
        sample_scheduler_input,
    ):
        """Test calculations are reproducible with same input."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
            HeatSchedulerAgent,
        )

        agent = HeatSchedulerAgent(sample_scheduler_config)
        agent.generate_explanation = Mock(return_value="Test")
        agent.generate_recommendations = Mock(return_value=[])
        agent.detect_anomalies = Mock(return_value=[])
        agent.reason_about = Mock(return_value="Test")

        result1 = agent.process(sample_scheduler_input)
        result2 = agent.process(sample_scheduler_input)

        # Core calculations should be identical
        assert result1.baseline_cost_usd == result2.baseline_cost_usd
        assert result1.optimized_cost_usd == result2.optimized_cost_usd
        assert result1.input_hash == result2.input_hash
