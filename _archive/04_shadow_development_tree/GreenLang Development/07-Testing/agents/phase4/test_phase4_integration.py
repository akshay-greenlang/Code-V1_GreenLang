# -*- coding: utf-8 -*-
"""
Phase 4 Integration Tests

Comprehensive integration tests for all 4 Phase 4 InsightAgent agents:
- Anomaly Investigation Agent
- Forecast Explanation Agent
- Benchmark Insight Agent
- Report Narrative Agent V2

Tests cover:
1. InsightAgent Pattern Compliance
2. Tool Integration
3. RAG Integration
4. Temperature and Budget
5. Error Handling
6. Reproducibility
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from greenlang.agents.anomaly_investigation_agent import AnomalyInvestigationAgent
from greenlang.agents.forecast_explanation_agent import ForecastExplanationAgent
from greenlang.agents.benchmark_agent_ai import BenchmarkAgentAI
from greenlang.agents.report_narrative_agent_ai_v2 import ReportNarrativeAgentAI_V2
from greenlang.agents.base_agents import InsightAgent
from greenlang.agents.categories import AgentCategory


class TestPhase4AgentArchitecture:
    """Test common architecture patterns across all Phase 4 agents."""

    def test_all_agents_inherit_from_insight_agent(self):
        """All Phase 4 agents should inherit from InsightAgent."""
        assert issubclass(AnomalyInvestigationAgent, InsightAgent)
        assert issubclass(ForecastExplanationAgent, InsightAgent)
        assert issubclass(BenchmarkAgentAI, InsightAgent)
        assert issubclass(ReportNarrativeAgentAI_V2, InsightAgent)

    def test_all_agents_have_category_insight(self):
        """All Phase 4 agents should have INSIGHT category."""
        agents = [
            AnomalyInvestigationAgent,
            ForecastExplanationAgent,
            BenchmarkAgentAI,
            ReportNarrativeAgentAI_V2
        ]

        for agent_class in agents:
            assert agent_class.category == AgentCategory.INSIGHT

    def test_all_agents_have_metadata(self):
        """All Phase 4 agents should have complete metadata."""
        agents = [
            AnomalyInvestigationAgent,
            ForecastExplanationAgent,
            BenchmarkAgentAI,
            ReportNarrativeAgentAI_V2
        ]

        for agent_class in agents:
            metadata = agent_class.metadata
            assert metadata is not None
            assert metadata.uses_chat_session == True
            assert metadata.uses_rag == True
            assert metadata.category == AgentCategory.INSIGHT
            assert len(metadata.name) > 0

    def test_all_agents_have_calculate_and_explain(self):
        """All Phase 4 agents must implement calculate() and explain()."""
        agents = [
            AnomalyInvestigationAgent(),
            ForecastExplanationAgent(),
            BenchmarkAgentAI(),
            ReportNarrativeAgentAI_V2()
        ]

        for agent in agents:
            assert hasattr(agent, 'calculate')
            assert callable(agent.calculate)
            assert hasattr(agent, 'explain')
            assert callable(agent.explain)


@pytest.mark.asyncio
class TestAnomalyInvestigationAgent:
    """Integration tests for Anomaly Investigation Agent."""

    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = AnomalyInvestigationAgent()
        assert agent is not None
        assert agent.category == AgentCategory.INSIGHT
        assert agent.metadata.uses_rag == True
        assert agent.metadata.uses_tools == True

    def test_calculate_deterministic_detection(
        self,
        sample_anomaly_data,
        assert_deterministic_calculation
    ):
        """Test deterministic anomaly detection."""
        agent = AnomalyInvestigationAgent()

        inputs = {
            "data": sample_anomaly_data,
            "contamination": 0.05,
            "feature_columns": ["energy_kwh", "temperature_c"],
            "system_type": "HVAC",
            "system_id": "HVAC-001"
        }

        result = agent.calculate(inputs)

        # Verify result structure
        assert "anomalies" in result
        assert "anomaly_scores" in result
        assert "anomaly_indices" in result
        assert "n_anomalies" in result
        assert "calculation_trace" in result

        # Verify calculation trace
        assert len(result["calculation_trace"]) > 0
        assert any("Isolation Forest" in step for step in result["calculation_trace"])

        # Verify deterministic
        assert_deterministic_calculation(agent, inputs)

    async def test_explain_with_rag_and_tools(
        self,
        sample_anomaly_data,
        mock_rag_engine,
        mock_chat_session,
        assert_temperature_compliance,
        assert_rag_collections
    ):
        """Test AI explanation with RAG and tool integration."""
        agent = AnomalyInvestigationAgent()

        # Calculate anomalies first
        calc_result = agent.calculate({
            "data": sample_anomaly_data,
            "contamination": 0.05,
            "feature_columns": ["energy_kwh", "temperature_c"]
        })

        # Generate explanation
        explanation = await agent.explain(
            calculation_result=calc_result,
            context={
                "data": sample_anomaly_data,
                "system_type": "HVAC",
                "system_id": "HVAC-001",
                "location": "Building A"
            },
            session=mock_chat_session,
            rag_engine=mock_rag_engine,
            temperature=0.6
        )

        # Verify RAG called
        assert mock_rag_engine.query.called

        # Verify correct RAG collections
        assert_rag_collections(mock_rag_engine, [
            "anomaly_patterns",
            "root_cause_database",
            "sensor_specifications",
            "maintenance_procedures"
        ])

        # Verify ChatSession called
        assert mock_chat_session.chat.called

        # Verify temperature 0.6
        assert_temperature_compliance(mock_chat_session, 0.6)

        # Verify explanation is substantial
        assert isinstance(explanation, str)
        assert len(explanation) > 100

    def test_tool_definitions(self):
        """Test investigation tool definitions."""
        agent = AnomalyInvestigationAgent()
        tools = agent._get_investigation_tools()

        assert len(tools) == 3
        tool_names = [t["name"] for t in tools]
        assert "maintenance_log_tool" in tool_names
        assert "sensor_diagnostic_tool" in tool_names
        assert "weather_data_tool" in tool_names

        # Verify tool schemas
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert "type" in tool["parameters"]
            assert "properties" in tool["parameters"]


@pytest.mark.asyncio
class TestForecastExplanationAgent:
    """Integration tests for Forecast Explanation Agent."""

    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = ForecastExplanationAgent()
        assert agent is not None
        assert agent.category == AgentCategory.INSIGHT

    def test_calculate_deterministic_forecast(
        self,
        sample_forecast_data,
        assert_deterministic_calculation
    ):
        """Test deterministic SARIMA forecasting."""
        agent = ForecastExplanationAgent()

        inputs = {
            "data": sample_forecast_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
            "seasonal_period": 12,
            "confidence_level": 0.95
        }

        result = agent.calculate(inputs)

        # Verify result structure
        assert "forecast" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "model_params" in result
        assert "metrics" in result
        assert "calculation_trace" in result

        # Verify forecast generated
        assert len(result["forecast"]) == 12

        # Verify calculation trace
        assert len(result["calculation_trace"]) > 0

        # Verify deterministic
        assert_deterministic_calculation(agent, inputs)

    async def test_explain_with_tools(
        self,
        sample_forecast_data,
        mock_rag_engine,
        mock_chat_session,
        assert_temperature_compliance,
        assert_rag_collections
    ):
        """Test AI forecast explanation."""
        agent = ForecastExplanationAgent()

        # Calculate forecast first
        calc_result = agent.calculate({
            "data": sample_forecast_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
            "seasonal_period": 12
        })

        # Generate explanation
        explanation = await agent.explain(
            calculation_result=calc_result,
            context={
                "data": sample_forecast_data,
                "target_column": "energy_kwh",
                "business_unit": "Manufacturing",
                "location": "California",
                "stakeholder_level": "executive"
            },
            session=mock_chat_session,
            rag_engine=mock_rag_engine,
            temperature=0.6
        )

        # Verify RAG called
        assert mock_rag_engine.query.called

        # Verify correct RAG collections
        assert_rag_collections(mock_rag_engine, [
            "forecasting_patterns",
            "seasonality_library",
            "event_database",
            "forecast_narratives"
        ])

        # Verify temperature 0.6
        assert_temperature_compliance(mock_chat_session, 0.6)

        # Verify explanation
        assert isinstance(explanation, str)
        assert len(explanation) > 100

    def test_explanation_tool_definitions(self):
        """Test forecast explanation tool definitions."""
        agent = ForecastExplanationAgent()
        tools = agent._get_explanation_tools()

        assert len(tools) == 3
        tool_names = [t["name"] for t in tools]
        assert "historical_trend_tool" in tool_names
        assert "seasonality_tool" in tool_names
        assert "event_correlation_tool" in tool_names


@pytest.mark.asyncio
class TestBenchmarkAgentAI:
    """Integration tests for Benchmark Agent AI."""

    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = BenchmarkAgentAI()
        assert agent is not None
        assert agent.category == AgentCategory.INSIGHT

    def test_calculate_deterministic_benchmarking(
        self,
        sample_benchmark_context,
        assert_deterministic_calculation
    ):
        """Test deterministic benchmark calculations."""
        agent = BenchmarkAgentAI()

        result = agent.calculate(sample_benchmark_context)

        # Verify result structure
        assert "carbon_intensity" in result
        assert "rating" in result
        assert "percentile" in result
        assert "benchmarks" in result
        assert "comparison" in result
        assert "calculation_trace" in result

        # Verify calculations
        assert result["carbon_intensity"] > 0
        assert result["rating"] in ["Excellent", "Good", "Average", "Below Average", "Poor"]
        assert 0 <= result["percentile"] <= 100

        # Verify calculation trace
        assert len(result["calculation_trace"]) > 0
        assert any("Carbon Intensity" in step for step in result["calculation_trace"])

        # Verify deterministic
        assert_deterministic_calculation(agent, sample_benchmark_context)

    async def test_explain_competitive_insights(
        self,
        sample_benchmark_context,
        mock_rag_engine,
        mock_chat_session,
        assert_temperature_compliance,
        assert_rag_collections
    ):
        """Test AI competitive insights generation."""
        agent = BenchmarkAgentAI()

        # Calculate benchmark first
        calc_result = agent.calculate(sample_benchmark_context)

        # Generate insights
        insights = await agent.explain(
            calculation_result=calc_result,
            context={
                "region": "California",
                "industry": "Technology",
                "building_age": 15,
                "improvement_goals": "Reach 'Good' rating by 2026"
            },
            session=mock_chat_session,
            rag_engine=mock_rag_engine,
            temperature=0.6
        )

        # Verify RAG called
        assert mock_rag_engine.query.called

        # Verify correct RAG collections
        assert_rag_collections(mock_rag_engine, [
            "industry_benchmarks",
            "best_practices",
            "competitive_analysis",
            "building_performance"
        ])

        # Verify temperature 0.6
        assert_temperature_compliance(mock_chat_session, 0.6)

        # Verify insights
        assert isinstance(insights, str)
        assert len(insights) > 100

    def test_benchmark_thresholds(self):
        """Test benchmark threshold data."""
        agent = BenchmarkAgentAI()

        # Verify benchmark data exists
        assert len(agent.BENCHMARKS) > 0

        # Verify structure
        for building_type, thresholds in agent.BENCHMARKS.items():
            assert "excellent" in thresholds
            assert "good" in thresholds
            assert "average" in thresholds
            assert "poor" in thresholds
            assert "unit" in thresholds


@pytest.mark.asyncio
class TestReportNarrativeAgentV2:
    """Integration tests for Report Narrative Agent V2."""

    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = ReportNarrativeAgentAI_V2()
        assert agent is not None
        assert agent.category == AgentCategory.INSIGHT
        assert agent.metadata.uses_tools == True

    def test_calculate_report_data(
        self,
        sample_report_context,
        assert_deterministic_calculation
    ):
        """Test deterministic report data calculation."""
        agent = ReportNarrativeAgentAI_V2()

        result = agent.calculate(sample_report_context)

        # Verify result structure
        assert "framework" in result
        assert "total_co2e_tons" in result
        assert "emissions_breakdown" in result
        assert "charts" in result
        assert "compliance_status" in result
        assert "compliance_checks" in result
        assert "executive_summary_data" in result
        assert "calculation_trace" in result

        # Verify calculations
        assert result["total_co2e_tons"] > 0
        assert result["compliance_status"] in ["Compliant", "Non-Compliant"]

        # Verify calculation trace
        assert len(result["calculation_trace"]) > 0
        assert any("Tool" in step for step in result["calculation_trace"])

        # Verify deterministic
        assert_deterministic_calculation(agent, sample_report_context)

    async def test_explain_narrative_generation(
        self,
        sample_report_context,
        mock_rag_engine,
        mock_chat_session,
        assert_temperature_compliance,
        assert_rag_collections
    ):
        """Test AI narrative generation."""
        agent = ReportNarrativeAgentAI_V2()

        # Calculate report data first
        calc_result = agent.calculate(sample_report_context)

        # Generate narrative
        narrative = await agent.explain(
            calculation_result=calc_result,
            context={
                "stakeholder_level": "executive",
                "industry": "Technology",
                "location": "California",
                "narrative_focus": "strategy"
            },
            session=mock_chat_session,
            rag_engine=mock_rag_engine,
            temperature=0.6
        )

        # Verify RAG called
        assert mock_rag_engine.query.called

        # Verify correct RAG collections
        assert_rag_collections(mock_rag_engine, [
            "narrative_templates",
            "compliance_guidance",
            "industry_reporting",
            "esg_best_practices"
        ])

        # Verify temperature 0.6
        assert_temperature_compliance(mock_chat_session, 0.6)

        # Verify narrative
        assert isinstance(narrative, str)
        assert len(narrative) > 200

    def test_narrative_tool_definitions(self):
        """Test narrative generation tool definitions."""
        agent = ReportNarrativeAgentAI_V2()
        tools = agent._get_narrative_tools()

        assert len(tools) == 2
        tool_names = [t["name"] for t in tools]
        assert "data_visualization_tool" in tool_names
        assert "stakeholder_preference_tool" in tool_names

    def test_framework_support(self):
        """Test supported frameworks."""
        agent = ReportNarrativeAgentAI_V2()

        expected_frameworks = ["TCFD", "CDP", "GRI", "SASB", "SEC", "ISO14064", "CUSTOM"]
        assert set(agent.SUPPORTED_FRAMEWORKS) == set(expected_frameworks)


@pytest.mark.asyncio
class TestTemperatureAndBudgetCompliance:
    """Test temperature and budget enforcement across all agents."""

    async def test_temperature_06_for_all_agents(
        self,
        sample_anomaly_data,
        sample_forecast_data,
        sample_benchmark_context,
        sample_report_context,
        mock_rag_engine,
        mock_chat_session,
        assert_temperature_compliance
    ):
        """Verify all Phase 4 agents use temperature 0.6."""
        agents_and_contexts = [
            (
                AnomalyInvestigationAgent(),
                {"data": sample_anomaly_data, "contamination": 0.05},
                {"data": sample_anomaly_data, "system_type": "HVAC"}
            ),
            (
                ForecastExplanationAgent(),
                {
                    "data": sample_forecast_data,
                    "target_column": "energy_kwh",
                    "forecast_horizon": 12,
                    "seasonal_period": 12
                },
                {"data": sample_forecast_data, "target_column": "energy_kwh"}
            ),
            (
                BenchmarkAgentAI(),
                sample_benchmark_context,
                {"region": "California"}
            ),
            (
                ReportNarrativeAgentAI_V2(),
                sample_report_context,
                {"stakeholder_level": "executive"}
            )
        ]

        for agent, calc_inputs, explain_context in agents_and_contexts:
            # Reset mock
            mock_chat_session.chat.reset_mock()

            # Calculate
            calc_result = agent.calculate(calc_inputs)

            # Explain
            await agent.explain(
                calculation_result=calc_result,
                context=explain_context,
                session=mock_chat_session,
                rag_engine=mock_rag_engine,
                temperature=0.6
            )

            # Verify temperature
            assert_temperature_compliance(mock_chat_session, 0.6)


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and resilience."""

    async def test_agents_handle_rag_failure(
        self,
        sample_anomaly_data,
        mock_chat_session
    ):
        """Test agents handle RAG engine failure gracefully."""
        agent = AnomalyInvestigationAgent()

        # Mock RAG engine that fails
        failing_rag = Mock()
        failing_rag.query = AsyncMock(side_effect=Exception("RAG connection failed"))

        calc_result = agent.calculate({
            "data": sample_anomaly_data,
            "contamination": 0.05
        })

        # Should handle gracefully (may return error or degraded response)
        try:
            explanation = await agent.explain(
                calculation_result=calc_result,
                context={"system_type": "HVAC"},
                session=mock_chat_session,
                rag_engine=failing_rag
            )
            # If it succeeds, that's fine (degraded mode)
            assert isinstance(explanation, str)
        except Exception as e:
            # If it fails, error message should be informative
            assert "RAG" in str(e) or "connection" in str(e).lower()

    async def test_agents_handle_chat_session_failure(
        self,
        sample_benchmark_context,
        mock_rag_engine
    ):
        """Test agents handle ChatSession failure gracefully."""
        agent = BenchmarkAgentAI()

        # Mock ChatSession that fails
        failing_session = Mock()
        failing_session.chat = AsyncMock(side_effect=Exception("API timeout"))

        calc_result = agent.calculate(sample_benchmark_context)

        # Should handle gracefully
        try:
            insights = await agent.explain(
                calculation_result=calc_result,
                context={"region": "California"},
                session=failing_session,
                rag_engine=mock_rag_engine
            )
            # If it succeeds somehow, that's fine
            assert isinstance(insights, str)
        except Exception as e:
            # Error should be informative
            assert "API" in str(e) or "timeout" in str(e).lower()

    def test_calculate_handles_invalid_inputs(self):
        """Test calculate() handles invalid inputs."""
        agent = AnomalyInvestigationAgent()

        # Test missing data
        with pytest.raises(ValueError):
            agent.calculate({"contamination": 0.05})

        # Test wrong data type
        with pytest.raises(ValueError):
            agent.calculate({"data": "not a dataframe"})

        # Test insufficient data
        with pytest.raises(ValueError):
            small_df = pd.DataFrame({"col1": [1, 2, 3]})
            agent.calculate({"data": small_df})


@pytest.mark.asyncio
class TestReproducibilityAndAuditTrail:
    """Test reproducibility and audit trail functionality."""

    def test_calculate_is_deterministic(
        self,
        sample_anomaly_data,
        sample_forecast_data,
        sample_benchmark_context
    ):
        """Verify calculate() produces same results for same inputs."""
        test_cases = [
            (
                AnomalyInvestigationAgent(),
                {"data": sample_anomaly_data, "contamination": 0.05}
            ),
            (
                ForecastExplanationAgent(),
                {
                    "data": sample_forecast_data,
                    "target_column": "energy_kwh",
                    "forecast_horizon": 12,
                    "seasonal_period": 12
                }
            ),
            (
                BenchmarkAgentAI(),
                sample_benchmark_context
            )
        ]

        for agent, inputs in test_cases:
            result1 = agent.calculate(inputs)
            result2 = agent.calculate(inputs)

            # Compare key numeric fields
            for key in result1.keys():
                if isinstance(result1[key], (int, float)):
                    assert result1[key] == result2[key], f"Non-deterministic: {key}"

    def test_audit_trail_capture(
        self,
        sample_benchmark_context
    ):
        """Test audit trail is captured correctly."""
        agent = BenchmarkAgentAI(enable_audit_trail=True)

        # Run calculation
        result = agent.calculate(sample_benchmark_context)

        # Verify audit trail
        assert hasattr(agent, 'audit_trail')
        assert len(agent.audit_trail) > 0

        # Verify audit entry structure
        latest = agent.audit_trail[-1]
        assert hasattr(latest, 'timestamp')
        assert hasattr(latest, 'operation')
        assert hasattr(latest, 'inputs')
        assert hasattr(latest, 'outputs')
        assert hasattr(latest, 'calculation_trace')

    def test_calculation_trace_completeness(
        self,
        sample_anomaly_data
    ):
        """Test calculation trace has sufficient detail."""
        agent = AnomalyInvestigationAgent()

        result = agent.calculate({
            "data": sample_anomaly_data,
            "contamination": 0.05
        })

        trace = result["calculation_trace"]

        # Should have multiple steps
        assert len(trace) >= 5

        # Should document key operations
        trace_text = " ".join(trace)
        assert "Input" in trace_text or "input" in trace_text
        assert "validation" in trace_text or "Validation" in trace_text


@pytest.mark.asyncio
class TestRAGIntegration:
    """Test RAG integration patterns."""

    async def test_rag_query_structure(
        self,
        sample_anomaly_data,
        mock_rag_engine,
        mock_chat_session
    ):
        """Test RAG queries are well-structured."""
        agent = AnomalyInvestigationAgent()

        calc_result = agent.calculate({
            "data": sample_anomaly_data,
            "contamination": 0.05
        })

        await agent.explain(
            calculation_result=calc_result,
            context={"system_type": "HVAC", "location": "Building A"},
            session=mock_chat_session,
            rag_engine=mock_rag_engine
        )

        # Verify query called
        assert mock_rag_engine.query.called

        # Check query structure
        call_args = mock_rag_engine.query.call_args
        assert "query" in call_args.kwargs
        assert "collections" in call_args.kwargs
        assert "top_k" in call_args.kwargs

        # Query should be substantial
        query = call_args.kwargs["query"]
        assert len(query) > 50

    async def test_rag_collections_are_valid(
        self,
        sample_forecast_data,
        mock_rag_engine,
        mock_chat_session
    ):
        """Test RAG collections are properly specified."""
        agent = ForecastExplanationAgent()

        calc_result = agent.calculate({
            "data": sample_forecast_data,
            "target_column": "energy_kwh",
            "forecast_horizon": 12
        })

        await agent.explain(
            calculation_result=calc_result,
            context={"target_column": "energy_kwh"},
            session=mock_chat_session,
            rag_engine=mock_rag_engine
        )

        call_args = mock_rag_engine.query.call_args
        collections = call_args.kwargs["collections"]

        # Should have collections
        assert len(collections) > 0

        # Should be strings
        assert all(isinstance(c, str) for c in collections)

        # Should have specific forecasting collections
        expected_any = ["forecasting_patterns", "seasonality_library", "event_database"]
        assert any(exp in collections for exp in expected_any)


@pytest.mark.asyncio
class TestToolDefinitionsAndExecution:
    """Test tool definitions and execution patterns."""

    def test_tool_definitions_are_valid(self):
        """Test all tool definitions follow proper schema."""
        agents = [
            AnomalyInvestigationAgent(),
            ForecastExplanationAgent(),
            ReportNarrativeAgentAI_V2()
        ]

        for agent in agents:
            # Get tools if they exist
            if hasattr(agent, '_get_investigation_tools'):
                tools = agent._get_investigation_tools()
            elif hasattr(agent, '_get_explanation_tools'):
                tools = agent._get_explanation_tools()
            elif hasattr(agent, '_get_narrative_tools'):
                tools = agent._get_narrative_tools()
            else:
                continue

            # Validate tool schema
            for tool in tools:
                assert "name" in tool
                assert "description" in tool
                assert "parameters" in tool

                params = tool["parameters"]
                assert "type" in params
                assert params["type"] == "object"
                assert "properties" in params

                # Verify properties have types
                for prop_name, prop_def in params["properties"].items():
                    assert "type" in prop_def or "enum" in prop_def


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
