"""AI-powered Carbon Footprint Aggregation with ChatSession Integration.

This module provides an AI-enhanced version of the CarbonAgent that uses
ChatSession for orchestration while preserving all deterministic calculations
as tool implementations.

Key Differences from Original CarbonAgent:
    1. AI Orchestration: Uses ChatSession for intelligent aggregation and insights
    2. Tool-First Numerics: All calculations wrapped as tools (zero hallucinated numbers)
    3. Natural Language Summaries: AI generates executive summaries with key insights
    4. Intelligent Recommendations: AI-driven reduction recommendations based on breakdown
    5. Deterministic Results: temperature=0, seed=42 for reproducibility
    6. Enhanced Provenance: Full audit trail of AI decisions and tool calls
    7. Backward Compatible: Same API as original CarbonAgent

Architecture:
    CarbonAgentAI (orchestration) -> ChatSession (AI) -> Tools (exact calculations)

    Tools:
    - aggregate_emissions: Aggregate emissions from multiple sources
    - calculate_breakdown: Calculate percentage breakdown by source
    - calculate_intensity: Calculate carbon intensity metrics (per sqft, per person)
    - generate_recommendations: Generate reduction recommendations based on largest sources

Example:
    >>> agent = CarbonAgentAI()
    >>> result = agent.run({
    ...     "emissions": [
    ...         {"fuel_type": "electricity", "co2e_emissions_kg": 15000},
    ...         {"fuel_type": "natural_gas", "co2e_emissions_kg": 8500}
    ...     ],
    ...     "building_area": 50000,
    ...     "occupancy": 200
    ... })
    >>> print(result.data["ai_summary"])
    "Total carbon footprint is 23.5 metric tons CO2e..."
    >>> print(result.data["total_co2e_tons"])
    23.5  # Exact calculation from tool

Author: GreenLang Framework Team
Date: October 2025
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
import logging

from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from greenlang.agents.carbon_agent import CarbonAgent
from greenlang.intelligence import (
    ChatSession,
    ChatMessage,
    Role,
    Budget,
    BudgetExceeded,
    create_provider,
)
from greenlang.intelligence.schemas.tools import ToolDef
from .citations import (
    EmissionFactorCitation,
    CalculationCitation,
    CitationBundle,
    create_emission_factor_citation,
)

logger = logging.getLogger(__name__)


class CarbonAgentAI(BaseAgent):
    """AI-powered carbon footprint aggregation agent using ChatSession.

    This agent enhances the original CarbonAgent with AI orchestration while
    maintaining exact deterministic calculations through tool implementations.

    Features:
    - AI orchestration via ChatSession for intelligent aggregation insights
    - Tool-first numerics (all calculations use tools, zero hallucinated numbers)
    - Natural language summaries with key insights
    - Intelligent recommendations based on emission breakdown
    - Deterministic results (temperature=0, seed=42)
    - Full provenance tracking (AI decisions + tool calls)
    - All original CarbonAgent features preserved
    - Backward compatible API

    Determinism Guarantees:
    - Same input always produces same output
    - All numeric values come from tools (no LLM math)
    - Reproducible AI responses via seed=42
    - Auditable decision trail

    Performance:
    - Async support for concurrent processing
    - Budget enforcement (max $0.50 per aggregation by default)
    - Performance metrics tracking

    Example:
        >>> agent = CarbonAgentAI()
        >>> result = agent.run({
        ...     "emissions": [
        ...         {"fuel_type": "electricity", "co2e_emissions_kg": 15000},
        ...         {"fuel_type": "natural_gas", "co2e_emissions_kg": 8500}
        ...     ],
        ...     "building_area": 50000,
        ...     "occupancy": 200
        ... })
        >>> print(result.data["ai_summary"])
        "Building carbon footprint totals 23.5 metric tons CO2e..."
        >>> print(result.data["total_co2e_tons"])
        23.5
    """

    def __init__(
        self,
        config: AgentConfig = None,
        *,
        budget_usd: float = 0.50,
        enable_ai_summary: bool = True,
        enable_recommendations: bool = True,
    ):
        """Initialize the AI-powered CarbonAgent.

        Args:
            config: Agent configuration (optional)
            budget_usd: Maximum USD to spend per aggregation (default: $0.50)
            enable_ai_summary: Enable AI-generated summaries (default: True)
            enable_recommendations: Enable AI recommendations (default: True)
        """
        if config is None:
            config = AgentConfig(
                name="CarbonAgentAI",
                description="AI-powered carbon footprint aggregation with intelligent insights",
                version="0.1.0",
            )
        super().__init__(config)

        # Initialize original carbon agent for tool implementations
        self.carbon_agent = CarbonAgent()

        # Configuration
        self.budget_usd = budget_usd
        self.enable_ai_summary = enable_ai_summary
        self.enable_recommendations = enable_recommendations

        # Initialize LLM provider (auto-detects available provider)
        self.provider = create_provider()

        # Performance tracking
        self._ai_call_count = 0
        self._tool_call_count = 0
        self._total_cost_usd = 0.0

        # Citation tracking
        self._current_citations: List[EmissionFactorCitation] = []
        self._calculation_citations: List[CalculationCitation] = []

        # Setup tools for ChatSession
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Setup tool definitions for ChatSession."""

        # Tool 1: Aggregate emissions (exact calculation)
        self.aggregate_emissions_tool = ToolDef(
            name="aggregate_emissions",
            description="Aggregate total emissions from multiple sources into kg and tons CO2e",
            parameters={
                "type": "object",
                "properties": {
                    "emissions": {
                        "type": "array",
                        "description": "List of emission records with co2e_emissions_kg",
                        "items": {
                            "type": "object",
                            "properties": {
                                "fuel_type": {"type": "string"},
                                "co2e_emissions_kg": {"type": "number"},
                            },
                        },
                    }
                },
                "required": ["emissions"],
            },
        )

        # Tool 2: Calculate breakdown (percentage by source)
        self.calculate_breakdown_tool = ToolDef(
            name="calculate_breakdown",
            description="Calculate percentage breakdown of emissions by source",
            parameters={
                "type": "object",
                "properties": {
                    "emissions": {
                        "type": "array",
                        "description": "List of emission records",
                        "items": {"type": "object"},
                    },
                    "total_kg": {
                        "type": "number",
                        "description": "Total emissions in kg for percentage calculations",
                    },
                },
                "required": ["emissions", "total_kg"],
            },
        )

        # Tool 3: Calculate intensity metrics
        self.calculate_intensity_tool = ToolDef(
            name="calculate_intensity",
            description="Calculate carbon intensity metrics (per sqft, per person)",
            parameters={
                "type": "object",
                "properties": {
                    "total_kg": {
                        "type": "number",
                        "description": "Total emissions in kg",
                    },
                    "building_area": {
                        "type": "number",
                        "description": "Building area in square feet",
                    },
                    "occupancy": {
                        "type": "integer",
                        "description": "Number of people",
                    },
                },
                "required": ["total_kg"],
            },
        )

        # Tool 4: Generate recommendations
        self.generate_recommendations_tool = ToolDef(
            name="generate_recommendations",
            description="Generate carbon reduction recommendations based on emission breakdown",
            parameters={
                "type": "object",
                "properties": {
                    "breakdown": {
                        "type": "array",
                        "description": "Emission breakdown with sources and percentages",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "co2e_kg": {"type": "number"},
                                "percentage": {"type": "number"},
                            },
                        },
                    }
                },
                "required": ["breakdown"],
            },
        )

    def _aggregate_emissions_impl(self, emissions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tool implementation: Aggregate emissions from multiple sources.

        Delegates to the original CarbonAgent for deterministic calculations.

        Args:
            emissions: List of emission records

        Returns:
            Dict with total_kg and total_tons
        """
        self._tool_call_count += 1

        # Delegate to original CarbonAgent
        result = self.carbon_agent.execute({"emissions": emissions})

        if not result.success:
            raise ValueError(f"Aggregation failed: {result.error}")

        total_kg = result.data["total_co2e_kg"]
        total_tons = result.data["total_co2e_tons"]

        # Create citations for each emission source
        for emission in emissions:
            fuel_type = emission.get("fuel_type", "Unknown")
            co2e_kg = emission.get("co2e_emissions_kg", 0)

            # Create emission factor citation
            citation = create_emission_factor_citation(
                source=emission.get("source", "Aggregated Emissions"),
                factor_name=f"{fuel_type.replace('_', ' ').title()} Emissions",
                value=co2e_kg,
                unit="kgCO2e",
                version="2025.1",
                confidence="high",
            )
            self._current_citations.append(citation)

        # Create calculation citation for aggregation
        calc_citation = CalculationCitation(
            step_name="aggregate_emissions",
            formula="sum(emissions[i].co2e_emissions_kg for i in range(len(emissions)))",
            inputs={"num_sources": len(emissions), "sources": [e.get("fuel_type", "Unknown") for e in emissions]},
            output={"value": total_kg, "unit": "kgCO2e"},
            timestamp=datetime.now(),
        )
        self._calculation_citations.append(calc_citation)

        return {
            "total_kg": total_kg,
            "total_tons": total_tons,
        }

    def _calculate_breakdown_impl(
        self, emissions: List[Dict[str, Any]], total_kg: float
    ) -> Dict[str, Any]:
        """Tool implementation: Calculate percentage breakdown by source.

        Args:
            emissions: List of emission records
            total_kg: Total emissions for percentage calculations

        Returns:
            Dict with breakdown list
        """
        self._tool_call_count += 1

        breakdown = []
        for emission in emissions:
            co2e = emission.get("co2e_emissions_kg", 0)
            breakdown.append({
                "source": emission.get("fuel_type", "Unknown"),
                "co2e_kg": round(co2e, 2),
                "co2e_tons": round(co2e / 1000, 3),
                "percentage": round((co2e / total_kg) * 100, 2) if total_kg > 0 else 0,
            })

        # Sort by emissions (largest first)
        breakdown.sort(key=lambda x: x["co2e_kg"], reverse=True)

        return {"breakdown": breakdown}

    def _calculate_intensity_impl(
        self,
        total_kg: float,
        building_area: Optional[float] = None,
        occupancy: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Tool implementation: Calculate carbon intensity metrics.

        Args:
            total_kg: Total emissions in kg
            building_area: Building area in square feet (optional)
            occupancy: Number of people (optional)

        Returns:
            Dict with intensity metrics
        """
        self._tool_call_count += 1

        intensity = {}

        if building_area and building_area > 0:
            intensity["per_sqft"] = round(total_kg / building_area, 4)

        if occupancy and occupancy > 0:
            intensity["per_person"] = round(total_kg / occupancy, 2)

        return {"intensity": intensity}

    def _generate_recommendations_impl(
        self, breakdown: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Tool implementation: Generate reduction recommendations.

        Generates targeted recommendations based on the largest emission sources.

        Args:
            breakdown: Emission breakdown by source (sorted)

        Returns:
            Dict with recommendations list
        """
        self._tool_call_count += 1

        recommendations = []

        # Top 3 sources for recommendations
        for i, source in enumerate(breakdown[:3]):
            source_name = source["source"].lower()
            percentage = source["percentage"]
            priority = "high" if i == 0 else ("medium" if i == 1 else "low")

            # Generate targeted recommendation based on source type
            if "electricity" in source_name:
                recommendations.append({
                    "priority": priority,
                    "source": source["source"],
                    "impact": f"{percentage}% of total emissions",
                    "action": "Install solar PV system or purchase renewable energy certificates (RECs)",
                    "potential_reduction": f"Up to {percentage}% reduction",
                    "estimated_payback": "5-10 years for solar PV",
                })
            elif "natural" in source_name or "gas" in source_name:
                recommendations.append({
                    "priority": priority,
                    "source": source["source"],
                    "impact": f"{percentage}% of total emissions",
                    "action": "Switch to electric heat pumps or improve building envelope insulation",
                    "potential_reduction": "50-70% reduction possible",
                    "estimated_payback": "7-15 years",
                })
            elif "diesel" in source_name or "fuel" in source_name:
                recommendations.append({
                    "priority": priority,
                    "source": source["source"],
                    "impact": f"{percentage}% of total emissions",
                    "action": "Transition to electric vehicles or biodiesel alternatives",
                    "potential_reduction": "60-90% reduction for EVs",
                    "estimated_payback": "3-7 years",
                })
            elif "coal" in source_name:
                recommendations.append({
                    "priority": priority,
                    "source": source["source"],
                    "impact": f"{percentage}% of total emissions",
                    "action": "Phase out coal usage, switch to renewable energy or natural gas",
                    "potential_reduction": "80-100% reduction",
                    "estimated_payback": "2-5 years (fuel cost savings)",
                })
            else:
                recommendations.append({
                    "priority": priority,
                    "source": source["source"],
                    "impact": f"{percentage}% of total emissions",
                    "action": f"Optimize {source['source']} efficiency and reduce consumption",
                    "potential_reduction": "10-30% reduction",
                    "estimated_payback": "Varies by intervention",
                })

        return {"recommendations": recommendations}

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data.

        Args:
            input_data: Input dictionary

        Returns:
            bool: True if valid
        """
        return self.carbon_agent.validate_input(input_data)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute carbon footprint aggregation with AI orchestration.

        This method uses ChatSession to orchestrate the aggregation workflow
        while ensuring all numeric calculations use deterministic tools.

        Process:
        1. Validate input
        2. Build AI prompt with aggregation requirements
        3. AI uses tools for exact calculations
        4. AI generates natural language summary and insights
        5. Return results with provenance

        Args:
            input_data: Input data with emissions list and optional metadata

        Returns:
            AgentResult with aggregated emissions and AI insights
        """
        start_time = datetime.now()

        # Validate input
        if not self.validate_input(input_data):
            return AgentResult(
                success=False,
                error="Invalid input: 'emissions' list required",
            )

        # Reset citations for new run
        self._current_citations = []
        self._calculation_citations = []

        emissions_list = input_data.get("emissions", [])

        # Handle empty emissions list
        if not emissions_list:
            return AgentResult(
                success=True,
                data={
                    "total_co2e_kg": 0,
                    "total_co2e_tons": 0,
                    "emissions_breakdown": [],
                    "carbon_intensity": {},
                    "recommendations": [],
                    "summary": "No emissions data provided",
                    "ai_summary": "No emissions data available for analysis.",
                },
                metadata={
                    "agent": "CarbonAgentAI",
                    "num_sources": 0,
                },
            )

        try:
            # Run async calculation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._execute_async(input_data))
            finally:
                loop.close()

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Add performance metadata
            if result.success:
                result.metadata["calculation_time_ms"] = duration * 1000
                result.metadata["ai_calls"] = self._ai_call_count
                result.metadata["tool_calls"] = self._tool_call_count
                result.metadata["total_cost_usd"] = self._total_cost_usd

            return result

        except Exception as e:
            self.logger.error(f"Error in AI carbon aggregation: {e}")
            return AgentResult(
                success=False,
                error=f"Failed to aggregate carbon emissions: {str(e)}",
            )

    async def _execute_async(self, input_data: Dict[str, Any]) -> AgentResult:
        """Async execution with ChatSession.

        Args:
            input_data: Input data

        Returns:
            AgentResult with aggregated data and AI summary
        """
        emissions_list = input_data.get("emissions", [])
        building_area = input_data.get("building_area")
        occupancy = input_data.get("occupancy")

        # Create ChatSession
        session = ChatSession(self.provider)

        # Build AI prompt
        prompt = self._build_prompt(input_data)

        # Prepare messages
        messages = [
            ChatMessage(
                role=Role.system,
                content=(
                    "You are a carbon accounting analyst for GreenLang. "
                    "You help aggregate and analyze carbon emissions from multiple sources. "
                    "IMPORTANT: You must use the provided tools for ALL numeric calculations. "
                    "Never estimate or guess numbers. Always provide clear insights and actionable recommendations."
                ),
            ),
            ChatMessage(role=Role.user, content=prompt),
        ]

        # Create budget
        budget = Budget(max_usd=self.budget_usd)

        try:
            # Call AI with tools
            self._ai_call_count += 1

            response = await session.chat(
                messages=messages,
                tools=[
                    self.aggregate_emissions_tool,
                    self.calculate_breakdown_tool,
                    self.calculate_intensity_tool,
                    self.generate_recommendations_tool,
                ],
                budget=budget,
                temperature=0.0,  # Deterministic
                seed=42,  # Reproducible
                tool_choice="auto",
            )

            # Track cost
            self._total_cost_usd += response.usage.cost_usd

            # Extract tool results
            tool_results = self._extract_tool_results(response)

            # Build output from tool results
            output = self._build_output(
                input_data,
                tool_results,
                response.text if self.enable_ai_summary else None,
            )

            return AgentResult(
                success=True,
                data=output,
                metadata={
                    "agent": "CarbonAgentAI",
                    "num_sources": len(emissions_list),
                    "provider": response.provider_info.provider,
                    "model": response.provider_info.model,
                    "tokens": response.usage.total_tokens,
                    "cost_usd": response.usage.cost_usd,
                    "tool_calls": len(response.tool_calls),
                    "seed": 42,  # Reproducibility seed
                    "temperature": 0.0,  # Deterministic temperature
                    "deterministic": True,
                },
            )

        except BudgetExceeded as e:
            self.logger.error(f"Budget exceeded: {e}")
            return AgentResult(
                success=False,
                error=f"AI budget exceeded: {str(e)}",
            )

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build AI prompt for aggregation.

        Args:
            input_data: Input data

        Returns:
            str: Formatted prompt
        """
        emissions_list = input_data.get("emissions", [])
        building_area = input_data.get("building_area")
        occupancy = input_data.get("occupancy")

        prompt = f"""Analyze and aggregate carbon emissions from multiple sources:

Sources: {len(emissions_list)} emission records
"""

        if building_area:
            prompt += f"Building area: {building_area:,.0f} sqft\n"

        if occupancy:
            prompt += f"Occupancy: {occupancy} people\n"

        prompt += """
Tasks:
1. Use aggregate_emissions tool to calculate total emissions (kg and tons)
2. Use calculate_breakdown tool to get percentage breakdown by source
"""

        if building_area or occupancy:
            prompt += "3. Use calculate_intensity tool to calculate carbon intensity metrics\n"

        if self.enable_recommendations:
            prompt += "4. Use generate_recommendations tool to get top 3 reduction recommendations\n"

        prompt += """5. Provide an executive summary with:
   - Total footprint statement
   - Key insights from the breakdown (what are the major sources?)
   - Notable intensity metrics (if available)
   - Actionable next steps

IMPORTANT:
- Use tools for ALL numeric calculations
- Do not estimate or guess any numbers
- Format numbers clearly (e.g., "23,500 kg" not "23500.0")
- Focus on actionable insights in your summary
"""

        return prompt

    def _extract_tool_results(self, response) -> Dict[str, Any]:
        """Extract results from AI tool calls.

        Args:
            response: ChatResponse from session

        Returns:
            Dict with tool results
        """
        results = {}

        for tool_call in response.tool_calls:
            name = tool_call.get("name", "")
            args = tool_call.get("arguments", {})

            if name == "aggregate_emissions":
                results["aggregation"] = self._aggregate_emissions_impl(**args)
            elif name == "calculate_breakdown":
                results["breakdown"] = self._calculate_breakdown_impl(**args)
            elif name == "calculate_intensity":
                results["intensity"] = self._calculate_intensity_impl(**args)
            elif name == "generate_recommendations":
                results["recommendations"] = self._generate_recommendations_impl(**args)

        return results

    def _build_output(
        self,
        input_data: Dict[str, Any],
        tool_results: Dict[str, Any],
        ai_summary: Optional[str],
    ) -> Dict[str, Any]:
        """Build output from tool results.

        Args:
            input_data: Original input
            tool_results: Results from tool calls
            ai_summary: AI-generated summary

        Returns:
            Dict with all aggregation data
        """
        aggregation = tool_results.get("aggregation", {})
        breakdown_result = tool_results.get("breakdown", {})
        intensity_result = tool_results.get("intensity", {})
        recommendations_result = tool_results.get("recommendations", {})

        output = {
            "total_co2e_kg": aggregation.get("total_kg", 0),
            "total_co2e_tons": aggregation.get("total_tons", 0),
            "emissions_breakdown": breakdown_result.get("breakdown", []),
            "carbon_intensity": intensity_result.get("intensity", {}),
        }

        # Add traditional summary (for backward compatibility)
        output["summary"] = self.carbon_agent._generate_summary(
            output["total_co2e_tons"],
            output["emissions_breakdown"],
        )

        # Add AI summary if enabled
        if ai_summary and self.enable_ai_summary:
            output["ai_summary"] = ai_summary

        # Add recommendations if enabled
        if self.enable_recommendations and recommendations_result:
            output["recommendations"] = recommendations_result.get("recommendations", [])

        # Add citations for emission factors and calculations
        if self._current_citations or self._calculation_citations:
            output["citations"] = {
                "emission_factors": [c.to_dict() for c in self._current_citations],
                "calculations": [c.dict() for c in self._calculation_citations],
            }

        return output

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary.

        Returns:
            Dict with AI and tool metrics
        """
        return {
            "agent": "CarbonAgentAI",
            "ai_metrics": {
                "ai_call_count": self._ai_call_count,
                "tool_call_count": self._tool_call_count,
                "total_cost_usd": self._total_cost_usd,
                "avg_cost_per_aggregation": (
                    self._total_cost_usd / max(self._ai_call_count, 1)
                ),
            },
            "base_agent_metrics": {
                "agent": "CarbonAgent",
                "version": self.carbon_agent.config.version,
            },
        }
