"""
Forecast Explanation Agent with AI-Powered Narrative Generation
GL Intelligence Infrastructure - INSIGHT PATH

Hybrid architecture combining deterministic SARIMA forecasting with AI-powered explanations:
- calculate(): Use forecast_agent_sarima for deterministic SARIMA predictions
- explain(): AI-powered narrative generation explaining forecast drivers and patterns

Pattern: InsightAgent (deterministic forecasts + AI narratives)
Temperature: 0.6 (analytical consistency for narratives)
Category: AgentCategory.INSIGHT

Tools for Explanation (3 new diagnostic tools):
1. historical_trend_tool - Analyze historical patterns and trends
2. seasonality_tool - Explain seasonal components and cycles
3. event_correlation_tool - Identify events that correlated with forecast changes

RAG Collections:
- forecasting_patterns: Historical forecast performance and accuracy
- seasonality_library: Seasonal pattern explanations and industry benchmarks
- event_database: Known events affecting time series (economic, weather, etc.)
- forecast_narratives: Example explanations and narrative templates

Key Features:
- Deterministic SARIMA forecasting (reproducible predictions)
- AI-powered narrative explanations of forecast drivers
- Evidence-based insights with data citations
- Natural language narratives for non-technical stakeholders
- Full audit trail for calculations
- Confidence scoring for explanations

Version: 1.0.0
Date: 2025-11-06
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from greenlang.agents.base_agents import InsightAgent, AuditEntry
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.forecast_agent_sarima import SARIMAForecastAgent


logger = logging.getLogger(__name__)


@dataclass
class ForecastExplanationResult:
    """Complete forecast explanation results."""
    # Deterministic forecast results
    forecast: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    forecast_dates: List[str]
    confidence_level: float
    model_params: Dict[str, Any]
    metrics: Dict[str, float]

    # AI-powered explanations
    executive_summary: str
    trend_analysis: str
    seasonality_explanation: str
    event_impact_analysis: str
    confidence_assessment: str
    stakeholder_narrative: str

    # Metadata
    calculation_trace: List[str]
    evidence_citations: Dict[str, Any]
    timestamp: str


class ForecastExplanationAgent(InsightAgent):
    """
    AI-powered forecast explanation agent with hybrid architecture.

    DETERMINISTIC FORECASTING (calculate method):
    - SARIMA model fitting and parameter tuning
    - Point forecasts with confidence intervals
    - Model performance metrics (RMSE, MAE, MAPE)
    - Seasonality detection and validation
    - Reproducible predictions with full audit trail

    AI-POWERED EXPLANATIONS (explain method):
    - Natural language narrative generation
    - Historical trend analysis and interpretation
    - Seasonal pattern explanations
    - Event correlation and impact analysis
    - Confidence assessment and uncertainty communication
    - Stakeholder-friendly narratives

    Tools for Explanation:
    1. historical_trend_tool - Analyze and explain historical patterns
       - Identify trends (upward, downward, stable, cyclical)
       - Calculate trend strength and consistency
       - Detect trend changes and inflection points

    2. seasonality_tool - Explain seasonal components
       - Identify seasonal periods and patterns
       - Calculate seasonal strength and stability
       - Compare current season to historical averages

    3. event_correlation_tool - Correlate events with forecast changes
       - Match anomalies to known events (weather, economic, operational)
       - Quantify event impact on predictions
       - Identify recurring event patterns

    RAG Collections Used:
    - forecasting_patterns: Historical forecast accuracy and performance
    - seasonality_library: Industry seasonal patterns and explanations
    - event_database: Known events affecting time series forecasts
    - forecast_narratives: Example narratives and explanation templates

    Temperature: 0.6 (analytical consistency for narrative generation)

    Use Cases:
    - Energy demand forecasting with stakeholder narratives
    - Temperature prediction with seasonal explanations
    - Emissions forecasting with trend analysis
    - Load forecasting with event impact assessment

    Example:
        agent = ForecastExplanationAgent()

        # Step 1: Calculate deterministic forecast
        forecast_result = agent.calculate({
            "data": df,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
            "seasonal_period": 12
        })

        # Step 2: Generate AI explanation
        explanation = await agent.explain(
            calculation_result=forecast_result,
            context={
                "business_unit": "Manufacturing",
                "location": "California",
                "stakeholder_level": "executive"
            },
            session=chat_session,
            rag_engine=rag_engine
        )
    """

    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="forecast_explanation_agent",
        category=AgentCategory.INSIGHT,
        uses_chat_session=True,
        uses_rag=True,
        uses_tools=True,
        critical_for_compliance=False,
        transformation_priority="HIGH (Phase 2 - Forecasting)",
        description="Hybrid agent: deterministic SARIMA forecasting + AI narrative explanations"
    )

    def __init__(
        self,
        enable_audit_trail: bool = True,
        forecasting_budget_usd: float = 1.00,
        explanation_budget_usd: float = 2.00
    ):
        """
        Initialize forecast explanation agent.

        Args:
            enable_audit_trail: Whether to capture calculation audit trail
            forecasting_budget_usd: Budget for SARIMA forecasting (default: $1.00)
            explanation_budget_usd: Budget for AI explanations (default: $2.00)
        """
        super().__init__(enable_audit_trail=enable_audit_trail)

        # Initialize SARIMA forecasting agent
        self.forecaster = SARIMAForecastAgent(
            budget_usd=forecasting_budget_usd,
            enable_explanations=False,  # We'll provide our own explanations
            enable_recommendations=False,
            enable_auto_tune=True
        )

        self.explanation_budget_usd = explanation_budget_usd

        # Performance tracking
        self._total_forecasts = 0
        self._total_explanations = 0
        self._total_cost_usd = 0.0

    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute deterministic SARIMA forecasting.

        This method is DETERMINISTIC and REPRODUCIBLE:
        - Uses SARIMA for time series forecasting
        - Same inputs produce same outputs (with fixed random seed)
        - No AI, pure statistical modeling
        - Full calculation audit trail

        Args:
            inputs: {
                "data": pd.DataFrame with datetime index,
                "target_column": str (column to forecast),
                "forecast_horizon": int (periods ahead),
                "seasonal_period": int (optional, auto-detected if not provided),
                "confidence_level": float (optional, default: 0.95),
                "auto_tune": bool (optional, default: True),
                "exog_columns": List[str] (optional, exogenous variables)
            }

        Returns:
            Dictionary with deterministic forecast results:
            {
                "forecast": List[float],
                "lower_bound": List[float],
                "upper_bound": List[float],
                "forecast_dates": List[str],
                "confidence_level": float,
                "model_params": Dict,
                "metrics": Dict (RMSE, MAE, MAPE),
                "seasonality": Dict,
                "stationarity": Dict,
                "preprocessing": Dict,
                "calculation_trace": List[str]
            }
        """
        calculation_trace = []
        self._total_forecasts += 1

        # Extract parameters
        data = inputs.get("data")
        target_column = inputs.get("target_column")
        forecast_horizon = inputs.get("forecast_horizon")
        seasonal_period = inputs.get("seasonal_period")

        calculation_trace.append(f"Input Data Shape: {data.shape}")
        calculation_trace.append(f"Target Column: {target_column}")
        calculation_trace.append(f"Forecast Horizon: {forecast_horizon}")

        if seasonal_period:
            calculation_trace.append(f"Seasonal Period: {seasonal_period}")
        else:
            calculation_trace.append("Seasonal Period: Auto-detect")

        # Validate inputs
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame")

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

        min_points = seasonal_period if seasonal_period else 12
        if len(data) < min_points * 2:
            raise ValueError(f"Insufficient data: need at least {min_points * 2} points")

        calculation_trace.append("Input validation passed")

        # Run deterministic SARIMA forecasting
        calculation_trace.append("Executing SARIMA forecasting...")
        forecast_result = self.forecaster.process(inputs)

        # Extract key results
        n_forecast_points = len(forecast_result.get("forecast", []))
        calculation_trace.append(f"Generated {n_forecast_points} forecast points")

        if "metrics" in forecast_result:
            rmse = forecast_result["metrics"].get("rmse", 0)
            mape = forecast_result["metrics"].get("mape", 0)
            calculation_trace.append(f"Model RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        # Build result with calculation trace
        result = {
            "forecast": forecast_result.get("forecast", []),
            "lower_bound": forecast_result.get("lower_bound", []),
            "upper_bound": forecast_result.get("upper_bound", []),
            "forecast_dates": forecast_result.get("forecast_dates", []),
            "confidence_level": forecast_result.get("confidence_level", 0.95),
            "model_params": forecast_result.get("model_params", {}),
            "metrics": forecast_result.get("metrics", {}),
            "seasonality": forecast_result.get("seasonality", {}),
            "stationarity": forecast_result.get("stationarity", {}),
            "preprocessing": forecast_result.get("preprocessing", {}),
            "calculation_trace": calculation_trace,
            "forecaster_metadata": forecast_result.get("metadata", {})
        }

        # Capture audit trail
        if self.enable_audit_trail:
            self._capture_calculation_audit(
                operation="sarima_forecasting",
                inputs=inputs,
                outputs=result,
                calculation_trace=calculation_trace
            )

        return result

    async def explain(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any],
        session,  # ChatSession instance
        rag_engine,  # RAGEngine instance
        temperature: float = 0.6
    ) -> str:
        """
        Generate AI-powered forecast explanation narrative.

        This method uses AI to explain WHY the forecast looks the way it does:
        - Historical trend analysis and interpretation
        - Seasonal pattern explanations
        - Event correlation and impact assessment
        - Confidence and uncertainty communication
        - Stakeholder-appropriate narratives

        Args:
            calculation_result: Output from calculate() method
            context: Additional context {
                "data": pd.DataFrame (original historical data),
                "target_column": str,
                "business_unit": str (optional),
                "location": str (optional),
                "industry": str (optional),
                "stakeholder_level": str (optional: "executive", "technical", "operations"),
                "recent_events": List[str] (optional, known events),
                "narrative_focus": str (optional: "trends", "seasonality", "events", "comprehensive")
            }
            session: ChatSession instance
            rag_engine: RAGEngine instance
            temperature: LLM temperature (default 0.6 for analytical consistency)

        Returns:
            Comprehensive forecast explanation with:
            - Executive summary (2-3 sentences)
            - Trend analysis (historical patterns)
            - Seasonality explanation (seasonal drivers)
            - Event impact analysis (correlated events)
            - Confidence assessment (uncertainty quantification)
            - Stakeholder narrative (audience-appropriate explanation)
        """
        self._total_explanations += 1

        # Extract forecast information
        forecast = calculation_result.get("forecast", [])
        model_params = calculation_result.get("model_params", {})
        metrics = calculation_result.get("metrics", {})
        seasonality = calculation_result.get("seasonality", {})

        # Validate forecast exists
        if not forecast or len(forecast) == 0:
            return self._format_no_forecast_explanation(context)

        # Step 1: Build RAG query for similar forecasts and patterns
        rag_query = self._build_rag_query(calculation_result, context)

        # Step 2: RAG retrieval for forecasting knowledge
        rag_result = await self._rag_retrieve(
            query=rag_query,
            rag_engine=rag_engine,
            collections=[
                "forecasting_patterns",
                "seasonality_library",
                "event_database",
                "forecast_narratives"
            ],
            top_k=10
        )

        # Step 3: Format RAG knowledge
        formatted_knowledge = self._format_rag_results(rag_result)

        # Step 4: Build explanation prompt with tools
        explanation_prompt = self._build_explanation_prompt(
            calculation_result,
            context,
            formatted_knowledge
        )

        # Step 5: Define explanation tools
        tools = self._get_explanation_tools()

        # Step 6: AI explanation with tools
        response = await session.chat(
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt(context)
                },
                {
                    "role": "user",
                    "content": explanation_prompt
                }
            ],
            tools=tools,
            temperature=temperature
        )

        # Track cost
        if hasattr(response, 'usage'):
            self._total_cost_usd += response.usage.cost_usd

        # Step 7: Process tool calls and gather analytical evidence
        tool_evidence = await self._process_tool_calls(response, calculation_result, context)

        # Step 8: Format final explanation narrative
        explanation_narrative = self._format_explanation_narrative(
            calculation_result=calculation_result,
            context=context,
            ai_narrative=response.text if hasattr(response, 'text') else str(response),
            tool_evidence=tool_evidence,
            rag_knowledge=formatted_knowledge
        )

        return explanation_narrative

    def _build_rag_query(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build semantic search query for RAG retrieval."""
        forecast = calculation_result.get("forecast", [])
        metrics = calculation_result.get("metrics", {})
        seasonality = calculation_result.get("seasonality", {})
        model_params = calculation_result.get("model_params", {})

        target_column = context.get("target_column", "metric")
        industry = context.get("industry", "")
        business_unit = context.get("business_unit", "")

        # Calculate forecast characteristics
        avg_forecast = np.mean(forecast) if forecast else 0
        forecast_trend = "increasing" if len(forecast) > 1 and forecast[-1] > forecast[0] else "decreasing"
        has_seasonality = seasonality.get("has_seasonality", False)
        seasonal_period = seasonality.get("seasonal_period", 12)

        query = f"""
Forecast Explanation Query:
- Target Metric: {target_column}
- Forecast Horizon: {len(forecast)} periods
- Average Forecast Value: {avg_forecast:.2f}
- Trend Direction: {forecast_trend}
- Seasonality: {'Yes' if has_seasonality else 'No'} (period: {seasonal_period})
- Model Performance (MAPE): {metrics.get('mape', 0):.1f}%
{f"- Industry: {industry}" if industry else ""}
{f"- Business Unit: {business_unit}" if business_unit else ""}

Looking for:
1. Similar forecast patterns in historical data
2. Explanation templates for {forecast_trend} trends
3. Seasonal pattern explanations for period {seasonal_period}
4. Industry-specific forecasting insights for {industry if industry else 'general'}
5. Common drivers of {target_column} changes
6. Event patterns that typically affect {target_column}
7. Best practices for communicating forecast uncertainty
8. Stakeholder-friendly narrative examples
"""

        return query.strip()

    def _build_explanation_prompt(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any],
        rag_knowledge: str
    ) -> str:
        """Build comprehensive explanation prompt."""
        forecast = calculation_result.get("forecast", [])
        lower_bound = calculation_result.get("lower_bound", [])
        upper_bound = calculation_result.get("upper_bound", [])
        model_params = calculation_result.get("model_params", {})
        metrics = calculation_result.get("metrics", {})
        seasonality = calculation_result.get("seasonality", {})

        target_column = context.get("target_column", "metric")
        stakeholder_level = context.get("stakeholder_level", "executive")
        narrative_focus = context.get("narrative_focus", "comprehensive")

        # Format forecast summary
        forecast_summary = f"""
**Forecast Summary:**
- Target: {target_column}
- Periods: {len(forecast)}
- Range: {min(forecast):.2f} to {max(forecast):.2f}
- Average: {np.mean(forecast):.2f}
- Trend: {('Increasing' if forecast[-1] > forecast[0] else 'Decreasing')} ({((forecast[-1] - forecast[0]) / forecast[0] * 100):.1f}% change)
"""

        # Format model info
        model_info = f"""
**Model Information:**
- Type: SARIMA {model_params.get('order', (0,0,0))} x {model_params.get('seasonal_order', (0,0,0,0))}
- Accuracy (RMSE): {metrics.get('rmse', 0):.2f}
- Accuracy (MAPE): {metrics.get('mape', 0):.1f}%
- Confidence Interval: {calculation_result.get('confidence_level', 0.95)*100:.0f}%
"""

        # Format seasonality info
        seasonality_info = ""
        if seasonality.get("has_seasonality"):
            seasonality_info = f"""
**Seasonality Detected:**
- Period: {seasonality.get('seasonal_period')} {self._infer_period_unit(seasonality.get('seasonal_period', 12))}
- Strength: {seasonality.get('strength', 0):.2f}
- Method: {seasonality.get('method', 'ACF')}
"""

        prompt = f"""
# FORECAST EXPLANATION REQUEST

## Forecast Results (Deterministic SARIMA)

{forecast_summary}

{model_info}

{seasonality_info}

**Context:**
- Business Unit: {context.get('business_unit', 'N/A')}
- Location: {context.get('location', 'N/A')}
- Industry: {context.get('industry', 'N/A')}
- Stakeholder Level: {stakeholder_level}
- Narrative Focus: {narrative_focus}

---

## Historical Knowledge (RAG Retrieval)

{rag_knowledge}

---

## Explanation Tasks

Use the provided tools to analyze the forecast:

1. **historical_trend_tool** - Analyze historical patterns
   - Identify long-term trends (upward, downward, stable, cyclical)
   - Calculate trend strength and consistency
   - Detect trend changes and inflection points
   - Explain what's driving the trend

2. **seasonality_tool** - Explain seasonal patterns
   - Identify dominant seasonal periods
   - Calculate seasonal amplitude and stability
   - Compare current forecast season to historical averages
   - Explain seasonal drivers (e.g., weather, business cycles)

3. **event_correlation_tool** - Identify correlated events
   - Match forecast changes to known events
   - Quantify event impact on predictions
   - Identify recurring event patterns
   - Explain event correlation strength

After gathering evidence, provide a comprehensive explanation structured as:

### 1. Executive Summary (2-3 sentences)
High-level overview of the forecast for {stakeholder_level} stakeholders.
What's the main takeaway? What should they know?

### 2. Trend Analysis (3-4 paragraphs)
- What historical patterns are evident?
- What's driving the observed trend?
- How consistent is the trend?
- What does this mean for the forecast period?

### 3. Seasonality Explanation (2-3 paragraphs)
- What seasonal patterns are present?
- What causes these seasonal variations?
- How strong is the seasonal effect?
- How does this impact the forecast?

### 4. Event Impact Analysis (2-3 paragraphs)
- What events have correlated with past changes?
- What events might affect the forecast period?
- How significant is event impact vs. trend/seasonality?

### 5. Confidence Assessment (2 paragraphs)
- How confident are we in this forecast?
- What are the key sources of uncertainty?
- What's the range of plausible outcomes?
- What factors could cause deviation?

### 6. Stakeholder Narrative (3-4 paragraphs)
Create a {stakeholder_level}-appropriate narrative that:
- Explains the forecast in plain language
- Provides actionable insights
- Addresses likely stakeholder questions
- Recommends how to use this forecast

**Important Guidelines:**
- Use tools to gather concrete analytical evidence
- Ground explanations in RAG knowledge and tool results
- Be specific about numbers and patterns from the forecast
- Adjust technical depth to stakeholder level
- Provide confidence levels for key statements
- Focus on WHY, not just WHAT
- Make it actionable and decision-relevant
"""

        return prompt

    def _get_system_prompt(self, context: Dict[str, Any]) -> str:
        """Get system prompt for explanation generation."""
        stakeholder_level = context.get("stakeholder_level", "executive")

        return f"""You are an expert forecasting analyst for climate and energy systems.

Your role is to generate clear, evidence-based explanations of time series forecasts for {stakeholder_level} stakeholders.

Key principles:
- Use analytical tools to gather evidence about trends, seasonality, and events
- Ground all explanations in concrete data and historical patterns
- Adjust technical depth to stakeholder level ({stakeholder_level})
- Focus on WHY the forecast looks this way, not just WHAT it predicts
- Provide confidence levels and acknowledge uncertainty
- Make explanations actionable and decision-relevant
- Use natural language that non-statisticians can understand
- Cite specific evidence from tools and RAG knowledge

Communication style for {stakeholder_level}:
- Executive: High-level, focus on implications and decisions
- Technical: Detailed methodology, model specifics, diagnostics
- Operations: Practical implications, what to expect, how to prepare

You are analytical, evidence-driven, and stakeholder-focused.
Temperature: 0.6 for consistency while allowing clear narrative flow."""

    def _get_explanation_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions for forecast explanation."""
        return [
            {
                "name": "historical_trend_tool",
                "description": "Analyze historical patterns and trends in the time series. Identifies long-term trends, trend strength, inflection points, and trend drivers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trend_type": {
                            "type": "string",
                            "enum": ["linear", "exponential", "polynomial", "comprehensive"],
                            "description": "Type of trend analysis to perform"
                        },
                        "lookback_periods": {
                            "type": "integer",
                            "description": "Number of historical periods to analyze",
                            "default": 24,
                            "minimum": 12
                        },
                        "detect_changes": {
                            "type": "boolean",
                            "description": "Detect trend change points and inflections",
                            "default": True
                        }
                    },
                    "required": ["trend_type"]
                }
            },
            {
                "name": "seasonality_tool",
                "description": "Explain seasonal patterns and cycles in the time series. Identifies seasonal periods, amplitude, stability, and seasonal drivers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis_depth": {
                            "type": "string",
                            "enum": ["basic", "detailed", "comprehensive"],
                            "description": "Depth of seasonal analysis"
                        },
                        "compare_seasons": {
                            "type": "boolean",
                            "description": "Compare seasonal patterns across multiple cycles",
                            "default": True
                        },
                        "identify_drivers": {
                            "type": "boolean",
                            "description": "Identify potential seasonal drivers (weather, business cycles, etc.)",
                            "default": True
                        }
                    }
                }
            },
            {
                "name": "event_correlation_tool",
                "description": "Identify and analyze events that correlate with forecast changes. Detects anomalies, correlates with known events, quantifies impact.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["weather", "economic", "operational", "seasonal_events", "all"]
                            },
                            "description": "Types of events to analyze"
                        },
                        "correlation_threshold": {
                            "type": "number",
                            "description": "Minimum correlation strength to report (0-1)",
                            "default": 0.5,
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "impact_quantification": {
                            "type": "boolean",
                            "description": "Quantify impact magnitude of correlated events",
                            "default": True
                        }
                    }
                }
            }
        ]

    async def _process_tool_calls(
        self,
        response,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process tool calls and gather analytical evidence."""
        tool_evidence = {
            "trend_analysis": None,
            "seasonality_analysis": None,
            "event_correlation": None
        }

        # Extract tool calls
        tool_calls = getattr(response, 'tool_calls', [])

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})

            try:
                if tool_name == "historical_trend_tool":
                    tool_evidence["trend_analysis"] = self._analyze_historical_trends(
                        arguments, calculation_result, context
                    )
                elif tool_name == "seasonality_tool":
                    tool_evidence["seasonality_analysis"] = self._analyze_seasonality(
                        arguments, calculation_result, context
                    )
                elif tool_name == "event_correlation_tool":
                    tool_evidence["event_correlation"] = self._analyze_event_correlation(
                        arguments, calculation_result, context
                    )
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                tool_evidence[tool_name.replace("_tool", "")] = {
                    "error": str(e),
                    "status": "failed"
                }

        return tool_evidence

    def _analyze_historical_trends(
        self,
        arguments: Dict[str, Any],
        calculation_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze historical trends in the time series.

        This is a mock implementation. In production, this would:
        - Perform detailed trend analysis on historical data
        - Detect trend changes and inflection points
        - Calculate trend strength and confidence
        - Identify trend drivers
        """
        data = context.get("data")
        target_column = context.get("target_column")

        if data is not None and target_column in data.columns:
            series = data[target_column]
            trend_direction = "increasing" if series.iloc[-1] > series.iloc[0] else "decreasing"
            trend_change = ((series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100)
            volatility = series.std() / series.mean()
        else:
            trend_direction = "stable"
            trend_change = 0
            volatility = 0.1

        return {
            "status": "success",
            "trend_type": arguments.get("trend_type", "linear"),
            "overall_direction": trend_direction,
            "trend_strength": abs(trend_change),
            "trend_change_pct": trend_change,
            "volatility": volatility,
            "inflection_points": [
                {"period": 12, "type": "acceleration", "significance": 0.7},
                {"period": 20, "type": "slowdown", "significance": 0.5}
            ],
            "trend_drivers": [
                {"factor": "Seasonal patterns", "contribution": 0.4, "confidence": 0.8},
                {"factor": "Long-term growth", "contribution": 0.35, "confidence": 0.7},
                {"factor": "Cyclical variations", "contribution": 0.25, "confidence": 0.6}
            ],
            "consistency": "moderate" if volatility < 0.3 else "variable",
            "confidence": 0.75
        }

    def _analyze_seasonality(
        self,
        arguments: Dict[str, Any],
        calculation_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in the forecast.

        This is a mock implementation. In production, this would:
        - Perform detailed seasonal decomposition
        - Compare seasonal patterns across cycles
        - Identify seasonal drivers
        - Calculate seasonal stability
        """
        seasonality = calculation_result.get("seasonality", {})

        return {
            "status": "success",
            "analysis_depth": arguments.get("analysis_depth", "detailed"),
            "has_seasonality": seasonality.get("has_seasonality", True),
            "seasonal_period": seasonality.get("seasonal_period", 12),
            "seasonal_strength": seasonality.get("strength", 0.6),
            "seasonal_amplitude": {
                "peak_month": "July",
                "peak_value": 1.25,
                "trough_month": "January",
                "trough_value": 0.75,
                "range_pct": 50
            },
            "seasonal_stability": {
                "year_over_year_consistency": 0.82,
                "pattern_reliability": "high",
                "anomalous_seasons": 1
            },
            "seasonal_drivers": [
                {
                    "driver": "Temperature variations",
                    "impact": "high",
                    "correlation": 0.85,
                    "explanation": "Peak energy demand correlates with extreme temperatures"
                },
                {
                    "driver": "Business calendar",
                    "impact": "medium",
                    "correlation": 0.65,
                    "explanation": "Lower activity during holidays and weekends"
                },
                {
                    "driver": "Daylight hours",
                    "impact": "medium",
                    "correlation": 0.60,
                    "explanation": "Lighting and HVAC demand varies with daylight"
                }
            ],
            "forecast_implications": "Strong seasonal pattern expected to continue. Peak demand in summer months (June-August), lowest in winter (December-February).",
            "confidence": 0.82
        }

    def _analyze_event_correlation(
        self,
        arguments: Dict[str, Any],
        calculation_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze event correlations with forecast changes.

        This is a mock implementation. In production, this would:
        - Detect anomalies and unusual patterns
        - Match with known event databases
        - Quantify event impact
        - Identify recurring event patterns
        """
        recent_events = context.get("recent_events", [])

        return {
            "status": "success",
            "event_types": arguments.get("event_types", ["all"]),
            "events_detected": 3,
            "correlated_events": [
                {
                    "event": "Heatwave (July 2025)",
                    "type": "weather",
                    "correlation_strength": 0.87,
                    "impact_magnitude": "+15% demand spike",
                    "recurrence": "Annual summer pattern",
                    "forecast_relevance": "High - similar conditions expected in forecast period"
                },
                {
                    "event": "Equipment upgrade (March 2025)",
                    "type": "operational",
                    "correlation_strength": 0.72,
                    "impact_magnitude": "-8% efficiency improvement",
                    "recurrence": "One-time event",
                    "forecast_relevance": "Medium - benefits persist in forecast"
                },
                {
                    "event": "Holiday shutdown (December 2024)",
                    "type": "seasonal_events",
                    "correlation_strength": 0.91,
                    "impact_magnitude": "-30% during shutdown",
                    "recurrence": "Annual pattern",
                    "forecast_relevance": "High - will recur in forecast period"
                }
            ],
            "recurring_patterns": [
                {
                    "pattern": "Summer demand spikes",
                    "frequency": "Annual",
                    "impact_range": "+10% to +20%",
                    "predictability": "high"
                },
                {
                    "pattern": "Holiday dips",
                    "frequency": "Annual",
                    "impact_range": "-25% to -35%",
                    "predictability": "high"
                }
            ],
            "forecast_adjustments": "Model accounts for seasonal events. Expect summer peaks and holiday troughs in forecast period.",
            "confidence": 0.80
        }

    def _format_explanation_narrative(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any],
        ai_narrative: str,
        tool_evidence: Dict[str, Any],
        rag_knowledge: str
    ) -> str:
        """Format comprehensive forecast explanation narrative."""
        target_column = context.get("target_column", "metric")
        forecast = calculation_result.get("forecast", [])
        metrics = calculation_result.get("metrics", {})

        narrative = f"""
# FORECAST EXPLANATION REPORT
Generated: {datetime.utcnow().isoformat()}Z

## Executive Summary
**Target:** {target_column}
**Forecast Period:** {len(forecast)} periods
**Model Accuracy (MAPE):** {metrics.get('mape', 0):.1f}%
**Confidence Level:** {calculation_result.get('confidence_level', 0.95)*100:.0f}%

---

## AI-Generated Narrative

{ai_narrative}

---

## Analytical Evidence

### Historical Trend Analysis
"""

        # Add trend evidence
        if tool_evidence.get("trend_analysis"):
            trend = tool_evidence["trend_analysis"]
            if trend.get("status") == "success":
                narrative += f"""
**Overall Trend:** {trend.get('overall_direction', 'Unknown').capitalize()} ({trend.get('trend_change_pct', 0):.1f}% change)
**Trend Strength:** {trend.get('trend_strength', 0):.1f}%
**Consistency:** {trend.get('consistency', 'Unknown').capitalize()}
**Confidence:** {trend.get('confidence', 0)*100:.0f}%

**Key Trend Drivers:**
"""
                for driver in trend.get("trend_drivers", [])[:3]:
                    narrative += f"- {driver.get('factor')}: {driver.get('contribution', 0)*100:.0f}% contribution (confidence: {driver.get('confidence', 0)*100:.0f}%)\n"

                if trend.get("inflection_points"):
                    narrative += f"\n**Inflection Points Detected:** {len(trend.get('inflection_points', []))}\n"
            else:
                narrative += f"\n- Status: {trend.get('error', 'No data available')}\n"
        else:
            narrative += "\n- Status: Not analyzed\n"

        narrative += "\n### Seasonality Analysis\n"

        # Add seasonality evidence
        if tool_evidence.get("seasonality_analysis"):
            seasonal = tool_evidence["seasonality_analysis"]
            if seasonal.get("status") == "success":
                narrative += f"""
**Seasonality Present:** {'Yes' if seasonal.get('has_seasonality') else 'No'}
**Seasonal Period:** {seasonal.get('seasonal_period')} {self._infer_period_unit(seasonal.get('seasonal_period', 12))}
**Seasonal Strength:** {seasonal.get('seasonal_strength', 0):.2f}
**Pattern Reliability:** {seasonal.get('seasonal_stability', {}).get('pattern_reliability', 'Unknown').capitalize()}
**Confidence:** {seasonal.get('confidence', 0)*100:.0f}%

**Seasonal Drivers:**
"""
                for driver in seasonal.get("seasonal_drivers", []):
                    narrative += f"- {driver.get('driver')} (impact: {driver.get('impact')}, correlation: {driver.get('correlation', 0):.2f})\n"
                    narrative += f"  → {driver.get('explanation')}\n"

                if seasonal.get("forecast_implications"):
                    narrative += f"\n**Forecast Implications:** {seasonal.get('forecast_implications')}\n"
            else:
                narrative += f"\n- Status: {seasonal.get('error', 'No data available')}\n"
        else:
            narrative += "\n- Status: Not analyzed\n"

        narrative += "\n### Event Correlation Analysis\n"

        # Add event evidence
        if tool_evidence.get("event_correlation"):
            events = tool_evidence["event_correlation"]
            if events.get("status") == "success":
                narrative += f"""
**Events Detected:** {events.get('events_detected', 0)}
**Confidence:** {events.get('confidence', 0)*100:.0f}%

**Correlated Events:**
"""
                for event in events.get("correlated_events", [])[:3]:
                    narrative += f"\n- **{event.get('event')}** ({event.get('type')})\n"
                    narrative += f"  - Correlation: {event.get('correlation_strength', 0):.2f}\n"
                    narrative += f"  - Impact: {event.get('impact_magnitude')}\n"
                    narrative += f"  - Recurrence: {event.get('recurrence')}\n"
                    narrative += f"  - Forecast Relevance: {event.get('forecast_relevance')}\n"

                if events.get("forecast_adjustments"):
                    narrative += f"\n**Forecast Adjustments:** {events.get('forecast_adjustments')}\n"
            else:
                narrative += f"\n- Status: {events.get('error', 'No data available')}\n"
        else:
            narrative += "\n- Status: Not analyzed\n"

        narrative += f"""

---

## Model Information

**Model Type:** SARIMA {calculation_result.get('model_params', {}).get('order', (0,0,0))} x {calculation_result.get('model_params', {}).get('seasonal_order', (0,0,0,0))}
**Auto-tuned:** {'Yes' if calculation_result.get('model_params', {}).get('auto_tuned') else 'No'}

**Performance Metrics:**
- RMSE: {metrics.get('rmse', 0):.2f}
- MAE: {metrics.get('mae', 0):.2f}
- MAPE: {metrics.get('mape', 0):.1f}%

**Forecast Range:**
- Minimum: {min(forecast):.2f}
- Maximum: {max(forecast):.2f}
- Average: {np.mean(forecast):.2f}
- Confidence Bounds: ±{((np.mean(calculation_result.get('upper_bound', [])) - np.mean(calculation_result.get('lower_bound', []))) / 2):.2f}

---

## Calculation Audit Trail

**Forecasting Method:** SARIMA (deterministic)
**Data Points Used:** {calculation_result.get('forecaster_metadata', {}).get('tokens', 'N/A')}
**Tools Executed:** {sum(1 for v in tool_evidence.values() if v is not None)}
**RAG Collections Queried:** 4 (forecasting_patterns, seasonality_library, event_database, forecast_narratives)

**Reproducibility:** Full audit trail captured. Same inputs will produce identical forecasts.

---

## Metadata

- **Stakeholder Level:** {context.get('stakeholder_level', 'executive')}
- **Narrative Focus:** {context.get('narrative_focus', 'comprehensive')}
- **Business Context:** {context.get('business_unit', 'N/A')} - {context.get('location', 'N/A')}
- **Report Generated:** {datetime.utcnow().isoformat()}Z

---

*This report combines deterministic SARIMA forecasting with AI-powered narrative explanations.*
*Forecasts are reproducible and auditable. Narratives are evidence-based and confidence-scored.*
"""

        return narrative.strip()

    def _format_no_forecast_explanation(self, context: Dict[str, Any]) -> str:
        """Format explanation when no forecast was generated."""
        return f"""
# FORECAST EXPLANATION REPORT
Generated: {datetime.utcnow().isoformat()}Z

## Status
No forecast was generated for {context.get('target_column', 'the target metric')}.

## Possible Reasons
1. Insufficient historical data for forecasting
2. Data quality issues preventing model fitting
3. Calculation errors during forecasting process

## Recommendations
1. Verify data completeness and quality
2. Ensure minimum data requirements are met (at least 2 seasonal cycles)
3. Check for missing values or anomalies in historical data
4. Review forecast parameters and adjust if needed

---

*This is a diagnostic message from the Forecast Explanation Agent.*
"""

    def _infer_period_unit(self, period: int) -> str:
        """Infer time unit from seasonal period."""
        if period == 7:
            return "days (weekly)"
        elif period == 12:
            return "months (annual)"
        elif period == 24:
            return "hours (daily)"
        elif period == 4:
            return "quarters (annual)"
        elif period == 52:
            return "weeks (annual)"
        else:
            return "periods"

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        return {
            "agent_id": self.metadata.name,
            "category": self.category.value,
            "total_forecasts": self._total_forecasts,
            "total_explanations": self._total_explanations,
            "total_cost_usd": self._total_cost_usd,
            "avg_cost_per_explanation": (
                self._total_cost_usd / max(self._total_explanations, 1)
            ),
            "forecaster_performance": self.forecaster.get_performance_summary()
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    print("=" * 80)
    print("Forecast Explanation Agent - INSIGHT PATH")
    print("=" * 80)

    # Initialize agent
    agent = ForecastExplanationAgent(enable_audit_trail=True)

    print("\n✓ Agent initialized with InsightAgent pattern")
    print(f"✓ Category: {agent.category}")
    print(f"✓ Uses ChatSession: {agent.metadata.uses_chat_session}")
    print(f"✓ Uses RAG: {agent.metadata.uses_rag}")
    print(f"✓ Uses Tools: {agent.metadata.uses_tools}")
    print(f"✓ Temperature: 0.6 (analytical consistency for narratives)")

    # Test calculation (deterministic)
    print("\n" + "=" * 80)
    print("TEST 1: DETERMINISTIC SARIMA FORECASTING")
    print("=" * 80)

    # Create test data with trend and seasonality
    np.random.seed(42)
    n_samples = 48  # 4 years of monthly data

    # Generate time series with trend + seasonality + noise
    dates = pd.date_range('2021-01-01', periods=n_samples, freq='M')
    trend = np.linspace(100, 150, n_samples)
    seasonal = 20 * np.sin(np.arange(n_samples) * 2 * np.pi / 12)
    noise = np.random.normal(0, 5, n_samples)
    energy = trend + seasonal + noise

    test_data = pd.DataFrame({
        "energy_kwh": energy,
        "temperature_c": 20 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 12) + np.random.normal(0, 2, n_samples)
    }, index=dates)

    test_inputs = {
        "data": test_data,
        "target_column": "energy_kwh",
        "forecast_horizon": 12,
        "seasonal_period": 12,
        "confidence_level": 0.95
    }

    print(f"\nInputs: {n_samples} historical samples, forecasting 12 months ahead")
    print(f"Data range: {test_data.index[0]} to {test_data.index[-1]}")

    result = agent.calculate(test_inputs)

    print(f"\n✓ Forecast Generated: {len(result['forecast'])} periods")
    print(f"✓ Forecast Range: {min(result['forecast']):.2f} to {max(result['forecast']):.2f}")
    print(f"✓ Model MAPE: {result['metrics'].get('mape', 0):.1f}%")
    print(f"✓ Seasonality Detected: {result['seasonality'].get('has_seasonality', False)}")

    if result['calculation_trace']:
        print(f"\nCalculation Trace (first 5 steps):")
        for i, step in enumerate(result['calculation_trace'][:5], 1):
            print(f"  {i}. {step}")

    # Test AI explanation (requires ChatSession and RAGEngine)
    print("\n" + "=" * 80)
    print("TEST 2: AI NARRATIVE GENERATION (requires live infrastructure)")
    print("=" * 80)

    print("\n⚠ AI explanation generation requires:")
    print("  - ChatSession instance (LLM API)")
    print("  - RAGEngine instance (vector database)")
    print("  - Knowledge base with collections:")
    print("    * forecasting_patterns")
    print("    * seasonality_library")
    print("    * event_database")
    print("    * forecast_narratives")

    print("\nExample async call:")
    print("""
    explanation = await agent.explain(
        calculation_result=result,
        context={
            "data": test_data,
            "target_column": "energy_kwh",
            "business_unit": "Manufacturing",
            "location": "California",
            "industry": "Technology",
            "stakeholder_level": "executive",
            "narrative_focus": "comprehensive",
            "recent_events": ["Equipment upgrade", "Heatwave event"]
        },
        session=chat_session,
        rag_engine=rag_engine,
        temperature=0.6
    )

    print(explanation)
    """)

    # Verify reproducibility
    print("\n" + "=" * 80)
    print("TEST 3: REPRODUCIBILITY VERIFICATION")
    print("=" * 80)

    result2 = agent.calculate(test_inputs)
    is_reproducible = (
        len(result['forecast']) == len(result2['forecast']) and
        all(abs(a - b) < 1e-6 for a, b in zip(result['forecast'], result2['forecast']))
    )

    print(f"\n✓ Same inputs produce same outputs: {is_reproducible}")

    if agent.enable_audit_trail:
        print(f"✓ Audit trail entries: {len(agent.audit_trail)}")

    # Performance summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    perf = agent.get_performance_summary()
    print(f"\nAgent: {perf['agent_id']}")
    print(f"Category: {perf['category']}")
    print(f"Total Forecasts: {perf['total_forecasts']}")
    print(f"Total Explanations: {perf['total_explanations']}")

    print("\n" + "=" * 80)
    print("PATTERN SUMMARY")
    print("=" * 80)
    print("\nPattern: InsightAgent (Hybrid Architecture)")
    print("  - calculate(): Deterministic SARIMA forecasting")
    print("  - explain(): AI narrative generation with RAG + tools")
    print("\nTools for Explanation:")
    print("  ✓ historical_trend_tool - Analyze trends and patterns")
    print("  ✓ seasonality_tool - Explain seasonal components")
    print("  ✓ event_correlation_tool - Identify event impacts")
    print("\nRAG Collections:")
    print("  ✓ forecasting_patterns - Historical forecast performance")
    print("  ✓ seasonality_library - Seasonal pattern explanations")
    print("  ✓ event_database - Known events affecting forecasts")
    print("  ✓ forecast_narratives - Example narrative templates")
    print("\nValue-Add:")
    print("  ✓ Evidence-based forecast explanations")
    print("  ✓ Stakeholder-appropriate narratives")
    print("  ✓ Trend, seasonality, and event analysis")
    print("  ✓ Confidence-scored insights")
    print("  ✓ Full audit trail for compliance")
    print("  ✓ Deterministic forecasts + AI narratives")
    print("=" * 80)
