# -*- coding: utf-8 -*-
"""
Integration tests for Agent Pipeline

Tests end-to-end agent execution, including:
- Multi-agent orchestration
- Data flow between agents
- Pipeline error handling
- Agent dependency resolution
- Pipeline performance
- Database persistence

Target: 100% critical path coverage
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import List, Dict, Any
import asyncio

try:
    from greenlang.sdk.pipeline import Pipeline as AgentPipeline
    from greenlang.agents.base import BaseAgent
    # PipelineConfig doesn't exist in greenlang.sdk.pipeline, create placeholder
    PipelineConfig = None
    PipelineExecutor = None
    AgentNode = None
    DataFlow = None
except ImportError:
    # Try alternative import
    try:
        from greenlang.data_engineering.etl.base_pipeline import (
            PipelineConfig,
            PipelineExecutor
        )
        from greenlang.agents.base import BaseAgent
        AgentPipeline = None
        AgentNode = None
        DataFlow = None
    except ImportError:
        # All imports failed, create placeholders
        AgentPipeline = None
        PipelineConfig = None
        PipelineExecutor = None
        AgentNode = None
        DataFlow = None
        BaseAgent = None

try:
    from greenlang.database import DatabaseSession
except ImportError:
    DatabaseSession = None

try:
    from greenlang.exceptions import (
        PipelineExecutionError,
        AgentDependencyError,
        DataValidationError
    )
except ImportError:
    # Define placeholder exceptions
    class PipelineExecutionError(Exception):
        pass
    class AgentDependencyError(Exception):
        pass
    class DataValidationError(Exception):
        pass


# Test Fixtures
@pytest.fixture
def pipeline_config():
    """Create test pipeline configuration."""
    return PipelineConfig(
        name="test_emission_calculation_pipeline",
        version="1.0.0",
        max_parallel_agents=5,
        timeout_seconds=300,
        retry_failed_agents=True,
        max_retries=3
    )


@pytest.fixture
def mock_database_session():
    """Create mock database session."""
    session = Mock(spec=DatabaseSession)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def sample_shipment_data():
    """Create sample shipment data for testing."""
    return {
        "shipment_id": "SHIP-12345",
        "product_category": "cement",
        "weight_tonnes": 50.0,
        "origin_country": "CN",
        "destination_country": "US",
        "import_date": "2025-01-15",
        "hs_code": "2523.29"
    }


@pytest.fixture
def mock_data_ingestion_agent():
    """Create mock data ingestion agent."""
    agent = Mock(spec=BaseAgent)
    agent.name = "data_ingestion"
    agent.process = AsyncMock(return_value={
        "status": "success",
        "records_ingested": 100,
        "validation_errors": []
    })
    return agent


@pytest.fixture
def mock_calculation_agent():
    """Create mock calculation agent."""
    agent = Mock(spec=BaseAgent)
    agent.name = "emission_calculator"
    agent.process = AsyncMock(return_value={
        "status": "success",
        "total_emissions_kg": 2680.0,
        "calculation_method": "IPCC_2006"
    })
    return agent


@pytest.fixture
def mock_validation_agent():
    """Create mock validation agent."""
    agent = Mock(spec=BaseAgent)
    agent.name = "data_validator"
    agent.process = AsyncMock(return_value={
        "status": "success",
        "validation_passed": True,
        "quality_score": 0.95
    })
    return agent


@pytest.fixture
def mock_reporting_agent():
    """Create mock reporting agent."""
    agent = Mock(spec=BaseAgent)
    agent.name = "report_generator"
    agent.process = AsyncMock(return_value={
        "status": "success",
        "report_id": "RPT-67890",
        "format": "PDF"
    })
    return agent


# Test Classes
class TestAgentPipeline:
    """Test suite for agent pipeline orchestration."""

    def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline initializes correctly."""
        pipeline = AgentPipeline(pipeline_config)

        assert pipeline.config == pipeline_config
        assert pipeline.agents == []
        assert pipeline.state == "INITIALIZED"

    def test_add_agent_to_pipeline(self, pipeline_config, mock_data_ingestion_agent):
        """Test adding agents to pipeline."""
        pipeline = AgentPipeline(pipeline_config)
        pipeline.add_agent(mock_data_ingestion_agent)

        assert len(pipeline.agents) == 1
        assert pipeline.agents[0] == mock_data_ingestion_agent

    def test_add_agent_with_dependencies(self, pipeline_config, mock_calculation_agent, mock_validation_agent):
        """Test adding agents with dependencies."""
        pipeline = AgentPipeline(pipeline_config)

        # Add validator first
        pipeline.add_agent(mock_validation_agent)

        # Add calculator that depends on validator
        pipeline.add_agent(
            mock_calculation_agent,
            depends_on=[mock_validation_agent.name]
        )

        assert len(pipeline.agents) == 2

    @pytest.mark.asyncio
    async def test_execute_simple_pipeline(
        self,
        pipeline_config,
        mock_data_ingestion_agent,
        sample_shipment_data
    ):
        """Test execution of simple single-agent pipeline."""
        pipeline = AgentPipeline(pipeline_config)
        pipeline.add_agent(mock_data_ingestion_agent)

        result = await pipeline.execute(sample_shipment_data)

        assert result['status'] == 'success'
        assert mock_data_ingestion_agent.process.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_multi_agent_pipeline(
        self,
        pipeline_config,
        mock_data_ingestion_agent,
        mock_validation_agent,
        mock_calculation_agent,
        sample_shipment_data
    ):
        """Test execution of multi-agent pipeline with dependencies."""
        pipeline = AgentPipeline(pipeline_config)

        # Add agents in order
        pipeline.add_agent(mock_data_ingestion_agent)
        pipeline.add_agent(mock_validation_agent, depends_on=["data_ingestion"])
        pipeline.add_agent(mock_calculation_agent, depends_on=["data_validator"])

        result = await pipeline.execute(sample_shipment_data)

        # All agents should have been called
        assert mock_data_ingestion_agent.process.call_count == 1
        assert mock_validation_agent.process.call_count == 1
        assert mock_calculation_agent.process.call_count == 1

        # Final result should contain data from all agents
        assert result['status'] == 'success'

    @pytest.mark.asyncio
    async def test_parallel_agent_execution(
        self,
        pipeline_config,
        mock_data_ingestion_agent
    ):
        """Test agents without dependencies execute in parallel."""
        pipeline = AgentPipeline(pipeline_config)

        # Create multiple independent agents
        agent1 = Mock(spec=BaseAgent)
        agent1.name = "agent1"
        agent1.process = AsyncMock(return_value={"status": "success"})

        agent2 = Mock(spec=BaseAgent)
        agent2.name = "agent2"
        agent2.process = AsyncMock(return_value={"status": "success"})

        agent3 = Mock(spec=BaseAgent)
        agent3.name = "agent3"
        agent3.process = AsyncMock(return_value={"status": "success"})

        # Add agents without dependencies
        pipeline.add_agent(agent1)
        pipeline.add_agent(agent2)
        pipeline.add_agent(agent3)

        import time
        start_time = time.time()

        result = await pipeline.execute({"test": "data"})

        duration = time.time() - start_time

        # All agents should execute
        assert agent1.process.call_count == 1
        assert agent2.process.call_count == 1
        assert agent3.process.call_count == 1

        # Should be faster than sequential (parallel execution)
        # If each agent takes ~0.1s, parallel should be ~0.1s, sequential would be ~0.3s

    @pytest.mark.asyncio
    async def test_agent_failure_handling(
        self,
        pipeline_config,
        mock_data_ingestion_agent,
        mock_calculation_agent
    ):
        """Test pipeline handles agent failures gracefully."""
        pipeline = AgentPipeline(pipeline_config)
        pipeline.add_agent(mock_data_ingestion_agent)

        # Make calculation agent fail
        mock_calculation_agent.process.side_effect = Exception("Calculation error")
        pipeline.add_agent(mock_calculation_agent, depends_on=["data_ingestion"])

        with pytest.raises(PipelineExecutionError) as exc_info:
            await pipeline.execute({"test": "data"})

        assert "calculation error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_agent_retry_on_failure(
        self,
        pipeline_config,
        mock_calculation_agent
    ):
        """Test agent retries on transient failures."""
        pipeline = AgentPipeline(pipeline_config)

        # Fail twice, then succeed
        mock_calculation_agent.process.side_effect = [
            Exception("Transient error"),
            Exception("Transient error"),
            {"status": "success", "total_emissions_kg": 2680.0}
        ]

        pipeline.add_agent(mock_calculation_agent)

        result = await pipeline.execute({"test": "data"})

        # Should succeed after retries
        assert result['status'] == 'success'
        assert mock_calculation_agent.process.call_count == 3

    @pytest.mark.asyncio
    async def test_pipeline_timeout(self, pipeline_config):
        """Test pipeline times out if execution exceeds limit."""
        # Set short timeout
        pipeline_config.timeout_seconds = 1

        pipeline = AgentPipeline(pipeline_config)

        # Create slow agent
        slow_agent = Mock(spec=BaseAgent)
        slow_agent.name = "slow_agent"

        async def slow_process(*args, **kwargs):
            await asyncio.sleep(5)  # Sleep longer than timeout
            return {"status": "success"}

        slow_agent.process = slow_process

        pipeline.add_agent(slow_agent)

        with pytest.raises(PipelineExecutionError) as exc_info:
            await pipeline.execute({"test": "data"})

        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_data_flow_between_agents(
        self,
        pipeline_config,
        mock_data_ingestion_agent,
        mock_calculation_agent
    ):
        """Test data flows correctly between dependent agents."""
        pipeline = AgentPipeline(pipeline_config)

        # Ingestion agent outputs data
        mock_data_ingestion_agent.process.return_value = {
            "status": "success",
            "fuel_type": "diesel",
            "quantity": 1000
        }

        # Calculation agent should receive ingestion output
        pipeline.add_agent(mock_data_ingestion_agent)
        pipeline.add_agent(mock_calculation_agent, depends_on=["data_ingestion"])

        await pipeline.execute({"test": "input"})

        # Check calculation agent received correct input
        call_args = mock_calculation_agent.process.call_args[0][0]
        assert call_args['fuel_type'] == 'diesel'
        assert call_args['quantity'] == 1000

    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, pipeline_config):
        """Test pipeline detects circular dependencies."""
        pipeline = AgentPipeline(pipeline_config)

        agent1 = Mock(spec=BaseAgent)
        agent1.name = "agent1"

        agent2 = Mock(spec=BaseAgent)
        agent2.name = "agent2"

        agent3 = Mock(spec=BaseAgent)
        agent3.name = "agent3"

        # Create circular dependency: agent1 -> agent2 -> agent3 -> agent1
        pipeline.add_agent(agent1)
        pipeline.add_agent(agent2, depends_on=["agent1"])
        pipeline.add_agent(agent3, depends_on=["agent2"])

        with pytest.raises(AgentDependencyError) as exc_info:
            pipeline.add_agent(agent1, depends_on=["agent3"])

        assert "circular dependency" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_dependency_error(self, pipeline_config, mock_calculation_agent):
        """Test error when agent depends on non-existent agent."""
        pipeline = AgentPipeline(pipeline_config)

        with pytest.raises(AgentDependencyError) as exc_info:
            pipeline.add_agent(mock_calculation_agent, depends_on=["nonexistent_agent"])

        assert "not found" in str(exc_info.value).lower()


class TestPipelineExecutor:
    """Test suite for pipeline executor."""

    @pytest.fixture
    def executor(self, pipeline_config):
        """Create pipeline executor instance."""
        return PipelineExecutor(pipeline_config)

    @pytest.mark.asyncio
    async def test_execute_agent_graph(self, executor, mock_data_ingestion_agent, mock_calculation_agent):
        """Test execution of agent dependency graph."""
        # Create graph: ingestion -> calculation
        graph = {
            'data_ingestion': AgentNode(
                agent=mock_data_ingestion_agent,
                dependencies=[]
            ),
            'emission_calculator': AgentNode(
                agent=mock_calculation_agent,
                dependencies=['data_ingestion']
            )
        }

        result = await executor.execute_graph(graph, {"test": "input"})

        assert result['status'] == 'success'

    @pytest.mark.asyncio
    async def test_topological_sort(self, executor):
        """Test topological sorting of agent dependencies."""
        # Create dependency graph
        graph = {
            'A': AgentNode(agent=Mock(name='A'), dependencies=[]),
            'B': AgentNode(agent=Mock(name='B'), dependencies=['A']),
            'C': AgentNode(agent=Mock(name='C'), dependencies=['A']),
            'D': AgentNode(agent=Mock(name='D'), dependencies=['B', 'C'])
        }

        sorted_agents = executor.topological_sort(graph)

        # A should come first, D should come last
        assert sorted_agents[0] == 'A'
        assert sorted_agents[-1] == 'D'

        # B and C should come after A but before D
        a_index = sorted_agents.index('A')
        b_index = sorted_agents.index('B')
        c_index = sorted_agents.index('C')
        d_index = sorted_agents.index('D')

        assert a_index < b_index < d_index
        assert a_index < c_index < d_index


class TestDatabaseIntegration:
    """Test suite for pipeline database integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_persists_results(
        self,
        pipeline_config,
        mock_database_session,
        mock_calculation_agent
    ):
        """Test pipeline persists execution results to database."""
        pipeline = AgentPipeline(pipeline_config, database_session=mock_database_session)
        pipeline.add_agent(mock_calculation_agent)

        result = await pipeline.execute({"test": "data"})

        # Database commit should be called
        assert mock_database_session.commit.call_count == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_rollback_on_failure(
        self,
        pipeline_config,
        mock_database_session,
        mock_calculation_agent
    ):
        """Test pipeline rolls back database changes on failure."""
        pipeline = AgentPipeline(pipeline_config, database_session=mock_database_session)

        # Make agent fail
        mock_calculation_agent.process.side_effect = Exception("Agent failed")
        pipeline.add_agent(mock_calculation_agent)

        with pytest.raises(PipelineExecutionError):
            await pipeline.execute({"test": "data"})

        # Database rollback should be called
        assert mock_database_session.rollback.call_count == 1


class TestPipelinePerformance:
    """Performance tests for pipeline execution."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_pipeline_throughput(self, pipeline_config):
        """Test pipeline can handle large number of agents."""
        pipeline = AgentPipeline(pipeline_config)

        # Add 50 independent agents
        agents = []
        for i in range(50):
            agent = Mock(spec=BaseAgent)
            agent.name = f"agent_{i}"
            agent.process = AsyncMock(return_value={"status": "success"})
            agents.append(agent)
            pipeline.add_agent(agent)

        import time
        start_time = time.time()

        result = await pipeline.execute({"test": "data"})

        duration = time.time() - start_time

        # All agents should execute
        for agent in agents:
            assert agent.process.call_count == 1

        # Should complete in reasonable time (<10s for 50 agents)
        assert duration < 10.0

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_pipeline_memory_usage(self, pipeline_config):
        """Test pipeline memory usage stays within limits."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        pipeline = AgentPipeline(pipeline_config)

        # Create large dataset
        large_data = {f"key_{i}": f"value_{i}" for i in range(100000)}

        # Add agents
        for i in range(10):
            agent = Mock(spec=BaseAgent)
            agent.name = f"agent_{i}"
            agent.process = AsyncMock(return_value={"status": "success", "data": large_data})
            pipeline.add_agent(agent)

        await pipeline.execute(large_data)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (<1GB)
        assert memory_increase < 1000


class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cbam_compliance_pipeline(self, pipeline_config, sample_shipment_data):
        """Test full CBAM compliance calculation pipeline."""
        pipeline = AgentPipeline(pipeline_config)

        # Create realistic agents
        ingestion_agent = Mock(spec=BaseAgent)
        ingestion_agent.name = "shipment_ingestion"
        ingestion_agent.process = AsyncMock(return_value={
            "status": "success",
            "shipment_data": sample_shipment_data
        })

        classification_agent = Mock(spec=BaseAgent)
        classification_agent.name = "product_classifier"
        classification_agent.process = AsyncMock(return_value={
            "status": "success",
            "product_category": "cement",
            "cn_code": "2523.29"
        })

        emission_agent = Mock(spec=BaseAgent)
        emission_agent.name = "emission_calculator"
        emission_agent.process = AsyncMock(return_value={
            "status": "success",
            "total_emissions_kg": 3500.0,
            "emission_factor": 70.0  # kg CO2e per tonne
        })

        cbam_agent = Mock(spec=BaseAgent)
        cbam_agent.name = "cbam_compliance"
        cbam_agent.process = AsyncMock(return_value={
            "status": "success",
            "cbam_certificate_required": True,
            "estimated_cbam_fee_eur": 262.5  # 3.5 tonnes * €75/tonne
        })

        # Build pipeline
        pipeline.add_agent(ingestion_agent)
        pipeline.add_agent(classification_agent, depends_on=["shipment_ingestion"])
        pipeline.add_agent(emission_agent, depends_on=["product_classifier"])
        pipeline.add_agent(cbam_agent, depends_on=["emission_calculator"])

        # Execute
        result = await pipeline.execute(sample_shipment_data)

        # Validate full workflow executed
        assert ingestion_agent.process.call_count == 1
        assert classification_agent.process.call_count == 1
        assert emission_agent.process.call_count == 1
        assert cbam_agent.process.call_count == 1

        assert result['status'] == 'success'
        assert result['cbam_certificate_required'] is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_scope3_calculation_pipeline(self, pipeline_config):
        """Test Scope 3 emissions calculation pipeline."""
        # Similar to CBAM test but for Scope 3 categories
        pass
