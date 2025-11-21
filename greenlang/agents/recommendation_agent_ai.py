# -*- coding: utf-8 -*-
"""AI-powered RecommendationAgent with ChatSession Integration.

This module provides an AI-enhanced version of the RecommendationAgent that uses
ChatSession for intelligent orchestration while preserving all deterministic calculations
as tool implementations.

Key Differences from Original RecommendationAgent:
    1. AI Orchestration: Uses ChatSession for natural language recommendations
    2. Tool-First Analysis: All calculations wrapped as tools (zero hallucinated numbers)
    3. Natural Language Explanations: AI generates actionable, context-aware recommendations
    4. Intelligent Ranking: AI-driven prioritization by ROI, impact, and feasibility
    5. Deterministic Results: temperature=0, seed=42 for reproducibility
    6. Enhanced Provenance: Full audit trail of AI decisions and tool calls
    7. Backward Compatible: Same API as original RecommendationAgent

Architecture:
    RecommendationAgentAI (orchestration) -> ChatSession (AI) -> Tools (exact calculations)

    Tools:
    - analyze_energy_usage: Identify inefficiencies and patterns from data
    - calculate_roi: Calculate payback periods and ROI for recommendations
    - rank_recommendations: Prioritize recommendations by impact/cost/feasibility
    - estimate_savings: Calculate potential emissions and cost savings
    - generate_implementation_plan: Create step-by-step action plans

Example:
    >>> agent = RecommendationAgentAI()
    >>> result = agent.run({
    ...     "emissions_by_source": {"electricity": 15000, "natural_gas": 8500},
    ...     "building_type": "commercial_office",
    ...     "building_age": 20,
    ...     "performance_rating": "Below Average",
    ...     "load_breakdown": {"hvac_load": 0.45},
    ...     "country": "US"
    ... })
    >>> print(result.data["ai_summary"])
    "Based on analysis, top 5 recommendations for reducing emissions..."
    >>> print(result.data["recommendations"][0])
    {"action": "Upgrade to high-efficiency HVAC system", "roi": 18.5, ...}

Author: GreenLang Framework Team
Date: October 2025
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
import logging

from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from greenlang.agents.recommendation_agent import RecommendationAgent
# Fixed: Removed incomplete import
from greenlang.determinism import DeterministicClock
from greenlang.intelligence import ChatSession, ChatMessage
from greenlang.intelligence.schemas.tools import ToolDef
from greenlang.agents.citations import (
    EmissionFactorCitation,
    CalculationCitation,
    CitationBundle,
    create_emission_factor_citation,
)

logger = logging.getLogger(__name__)


class RecommendationAgentAI(BaseAgent):
    """AI-powered recommendation agent using ChatSession.

    This agent enhances the original RecommendationAgent with AI orchestration while
    maintaining exact deterministic calculations through tool implementations.

    Features:
    - AI orchestration via ChatSession for intelligent recommendation generation
    - Tool-first analysis (all calculations use tools, zero hallucinated numbers)
    - Natural language explanations for each recommendation
    - Intelligent ROI-based prioritization
    - Source-specific, actionable recommendations
    - Deterministic results (temperature=0, seed=42)
    - Full provenance tracking (AI decisions + tool calls)
    - All original RecommendationAgent features preserved
    - Backward compatible API

    Determinism Guarantees:
    - Same input always produces same recommendations
    - All numeric values (ROI, savings) come from tools (no LLM math)
    - Reproducible AI responses via seed=42
    - Auditable decision trail

    Performance:
    - Async support for concurrent processing
    - Budget enforcement (max $0.50 per analysis by default)
    - Performance metrics tracking

    Example:
        >>> agent = RecommendationAgentAI()
        >>> result = agent.run({
        ...     "emissions_by_source": {"electricity": 15000, "natural_gas": 8500},
        ...     "building_type": "commercial_office",
        ...     "building_age": 20
        ... })
        >>> print(result.data["ai_summary"])
        "Based on comprehensive analysis..."
        >>> print(len(result.data["recommendations"]))
        5
    """

    def __init__(
        self,
        config: AgentConfig = None,
        *,
        budget_usd: float = 0.50,
        enable_ai_summary: bool = True,
        enable_implementation_plans: bool = True,
        max_recommendations: int = 5,
    ):
        """Initialize the AI-powered RecommendationAgent.

        Args:
            config: Agent configuration (optional)
            budget_usd: Maximum USD to spend per analysis (default: $0.50)
            enable_ai_summary: Enable AI-generated summaries (default: True)
            enable_implementation_plans: Enable detailed implementation plans (default: True)
            max_recommendations: Maximum recommendations to return (default: 5)
        """
        if config is None:
            config = AgentConfig(
                name="RecommendationAgentAI",
                description="AI-powered recommendation generation with intelligent insights",
                version="0.1.0",
        super().__init__(config)

        # Initialize original recommendation agent for tool implementations
        self.rec_agent = RecommendationAgent()

        # Configuration
        self.budget_usd = budget_usd
        self.enable_ai_summary = enable_ai_summary
        self.enable_implementation_plans = enable_implementation_plans
        self.max_recommendations = max_recommendations

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

        # Tool 1: Analyze energy usage patterns
        self.analyze_energy_usage_tool = ToolDef(
            name="analyze_energy_usage",
            description="Analyze energy usage patterns and identify inefficiencies from building data",
            parameters={
                "type": "object",
                "properties": {
                    "emissions_by_source": {
                        "type": "object",
                        "description": "Emissions breakdown by source (e.g., electricity, natural_gas)",
                    },
                    "load_breakdown": {
                        "type": "object",
                        "description": "Load breakdown (e.g., hvac_load, lighting_load)",
                    },
                    "building_age": {
                        "type": "integer",
                        "description": "Building age in years",
                    },
                    "performance_rating": {
                        "type": "string",
                        "description": "Performance rating (Poor, Below Average, Average, Good, Excellent)",
                    },
                },
                "required": ["emissions_by_source"],
            },

        # Tool 2: Calculate ROI for recommendations
        self.calculate_roi_tool = ToolDef(
            name="calculate_roi",
            description="Calculate payback period and ROI for specific recommendations",
            parameters={
                "type": "object",
                "properties": {
                    "recommendations": {
                        "type": "array",
                        "description": "List of recommendations with cost and impact data",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                                "cost": {"type": "string"},
                                "impact": {"type": "string"},
                                "payback": {"type": "string"},
                            },
                        },
                    },
                    "current_emissions_kg": {
                        "type": "number",
                        "description": "Current annual emissions in kg CO2e",
                    },
                    "energy_cost_per_kwh": {
                        "type": "number",
                        "description": "Energy cost per kWh (default: 0.12)",
                        "default": 0.12,
                    },
                },
                "required": ["recommendations", "current_emissions_kg"],
            },

        # Tool 3: Rank recommendations by priority
        self.rank_recommendations_tool = ToolDef(
            name="rank_recommendations",
            description="Rank and prioritize recommendations by impact, cost, and ROI",
            parameters={
                "type": "object",
                "properties": {
                    "recommendations": {
                        "type": "array",
                        "description": "List of recommendations to rank",
                        "items": {"type": "object"},
                    },
                    "prioritize_by": {
                        "type": "string",
                        "description": "Prioritization strategy (roi, impact, cost, payback)",
                        "enum": ["roi", "impact", "cost", "payback"],
                        "default": "roi",
                    },
                },
                "required": ["recommendations"],
            },

        # Tool 4: Estimate savings
        self.estimate_savings_tool = ToolDef(
            name="estimate_savings",
            description="Estimate potential emissions and cost savings from recommendations",
            parameters={
                "type": "object",
                "properties": {
                    "recommendations": {
                        "type": "array",
                        "description": "Top recommendations to estimate savings for",
                        "items": {"type": "object"},
                    },
                    "current_emissions_kg": {
                        "type": "number",
                        "description": "Current annual emissions in kg CO2e",
                    },
                    "current_energy_cost_usd": {
                        "type": "number",
                        "description": "Current annual energy cost in USD",
                    },
                },
                "required": ["recommendations", "current_emissions_kg"],
            },

        # Tool 5: Generate implementation plan
        self.generate_implementation_plan_tool = ToolDef(
            name="generate_implementation_plan",
            description="Generate detailed step-by-step implementation plan for recommendations",
            parameters={
                "type": "object",
                "properties": {
                    "recommendations": {
                        "type": "array",
                        "description": "Recommendations to create plan for",
                        "items": {"type": "object"},
                    },
                    "building_type": {
                        "type": "string",
                        "description": "Building type (e.g., commercial_office, residential)",
                    },
                    "timeline_months": {
                        "type": "integer",
                        "description": "Desired implementation timeline in months",
                        "default": 12,
                    },
                },
                "required": ["recommendations"],
            },

    def _analyze_energy_usage_impl(
        self,
        emissions_by_source: Dict[str, float],
        load_breakdown: Optional[Dict[str, float]] = None,
        building_age: Optional[int] = None,
        performance_rating: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tool implementation: Analyze energy usage patterns.

        Args:
            emissions_by_source: Emissions breakdown by source
            load_breakdown: Load breakdown by category
            building_age: Building age in years
            performance_rating: Performance rating

        Returns:
            Dict with usage analysis and identified issues
        """
        self._tool_call_count += 1

        total_emissions = sum(emissions_by_source.values())

        # Calculate percentages
        source_percentages = {}
        for source, emissions in emissions_by_source.items():
            percentage = (emissions / total_emissions * 100) if total_emissions > 0 else 0
            source_percentages[source] = round(percentage, 2)

        # Identify issues
        issues = []

        # High electricity usage
        electricity_pct = source_percentages.get("electricity", 0)
        if electricity_pct > 60:
            issues.append({
                "type": "high_electricity",
                "severity": "high",
                "description": f"Electricity accounts for {electricity_pct}% of emissions",
            })

        # High HVAC load
        if load_breakdown:
            hvac_load = load_breakdown.get("hvac_load", 0)
            if hvac_load > 0.4:
                issues.append({
                    "type": "high_hvac",
                    "severity": "high",
                    "description": f"HVAC represents {hvac_load*100}% of building load",
                })

        # Old building (inefficiency risk)
        if building_age and building_age > 20:
            issues.append({
                "type": "aging_infrastructure",
                "severity": "medium",
                "description": f"Building age ({building_age} years) suggests potential inefficiencies",
            })

        # Poor performance
        if performance_rating in ["Poor", "Below Average"]:
            issues.append({
                "type": "poor_performance",
                "severity": "high",
                "description": f"Performance rating ({performance_rating}) indicates optimization opportunities",
            })

        return {
            "total_emissions_kg": total_emissions,
            "source_percentages": source_percentages,
            "dominant_source": max(source_percentages.items(), key=lambda x: x[1])[0] if source_percentages else None,
            "issues_identified": issues,
            "issue_count": len(issues),
        }

    def _calculate_roi_impl(
        self,
        recommendations: List[Dict[str, Any]],
        current_emissions_kg: float,
        energy_cost_per_kwh: float = 0.12,
    ) -> Dict[str, Any]:
        """Tool implementation: Calculate ROI and payback.

        Args:
            recommendations: List of recommendations
            current_emissions_kg: Current emissions
            energy_cost_per_kwh: Energy cost

        Returns:
            Dict with ROI calculations for each recommendation
        """
        self._tool_call_count += 1

        roi_results = []

        for rec in recommendations:
            # Extract payback years
            payback_str = rec.get("payback", "10 years")
            payback_years = self.rec_agent._extract_payback_years(payback_str)

            # Estimate cost based on cost category
            cost_category = rec.get("cost", "Medium")
            if cost_category == "Low":
                estimated_cost = 5000
            elif cost_category == "Medium":
                estimated_cost = 50000
            elif cost_category == "High":
                estimated_cost = 250000
            else:
                estimated_cost = 50000

            # Extract impact percentage
            impact_str = rec.get("impact", "10%")
            import re
            percentages = re.findall(r"(\d+)%", impact_str)
            avg_impact_pct = sum(map(int, percentages)) / len(percentages) if percentages else 10

            # Calculate annual savings
            emissions_reduction_kg = current_emissions_kg * (avg_impact_pct / 100)
            # Rough conversion: 1 kg CO2e ~ 0.5 kWh for electricity
            energy_savings_kwh = emissions_reduction_kg * 0.5
            annual_cost_savings = energy_savings_kwh * energy_cost_per_kwh

            # Calculate ROI
            roi_percentage = ((annual_cost_savings * payback_years) / estimated_cost * 100) if estimated_cost > 0 else 0

            roi_results.append({
                "action": rec.get("action", ""),
                "estimated_cost_usd": estimated_cost,
                "annual_savings_usd": round(annual_cost_savings, 2),
                "payback_years": payback_years,
                "roi_percentage": round(roi_percentage, 2),
                "emissions_reduction_kg": round(emissions_reduction_kg, 2),
                "cost_category": cost_category,
            })

        # Create calculation citation for ROI analysis
        total_savings = sum(r["annual_savings_usd"] for r in roi_results)
        total_reduction = sum(r["emissions_reduction_kg"] for r in roi_results)

        calc_citation = CalculationCitation(
            step_name="calculate_roi",
            formula="ROI = (Annual_Savings × Payback_Years / Cost) × 100",
            inputs={
                "recommendations_count": len(recommendations),
                "current_emissions_kg": current_emissions_kg,
                "energy_cost_per_kwh": energy_cost_per_kwh,
            },
            output={
                "total_potential_savings_usd": round(total_savings, 2),
                "total_emissions_reduction_kg": round(total_reduction, 2),
            },
            timestamp=DeterministicClock.now(),
            tool_call_id=f"roi_calc_{self._tool_call_count}",
        self._calculation_citations.append(calc_citation)

        return {
            "roi_calculations": roi_results,
            "total_potential_savings_usd": round(total_savings, 2),
            "total_emissions_reduction_kg": round(total_reduction, 2),
        }

    def _rank_recommendations_impl(
        self,
        recommendations: List[Dict[str, Any]],
        prioritize_by: str = "roi",
    ) -> Dict[str, Any]:
        """Tool implementation: Rank recommendations.

        Args:
            recommendations: List of recommendations
            prioritize_by: Prioritization strategy

        Returns:
            Dict with ranked recommendations
        """
        self._tool_call_count += 1

        # Use original agent's prioritization logic
        ranked = self.rec_agent._prioritize_recommendations(recommendations)

        # Additional sorting based on strategy
        if prioritize_by == "roi" and all("roi_percentage" in r for r in ranked):
            ranked.sort(key=lambda x: x.get("roi_percentage", 0), reverse=True)
        elif prioritize_by == "impact":
            # Sort by impact percentage
            def extract_impact(rec):
                import re
                impact_str = rec.get("impact", "0%")
                percentages = re.findall(r"(\d+)%", impact_str)
                return max(map(int, percentages)) if percentages else 0
            ranked.sort(key=extract_impact, reverse=True)
        elif prioritize_by == "cost":
            # Sort by cost (Low first)
            cost_order = {"Low": 0, "Medium": 1, "High": 2}
            ranked.sort(key=lambda x: cost_order.get(x.get("cost", "Medium"), 1))
        elif prioritize_by == "payback":
            # Already sorted by payback in prioritize_recommendations
            pass

        # Add ranking
        for idx, rec in enumerate(ranked):
            rec["rank"] = idx + 1

        return {
            "ranked_recommendations": ranked,
            "prioritization_strategy": prioritize_by,
            "count": len(ranked),
        }

    def _estimate_savings_impl(
        self,
        recommendations: List[Dict[str, Any]],
        current_emissions_kg: float,
        current_energy_cost_usd: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Tool implementation: Estimate savings.

        Args:
            recommendations: Top recommendations
            current_emissions_kg: Current emissions
            current_energy_cost_usd: Current energy cost

        Returns:
            Dict with savings estimates
        """
        self._tool_call_count += 1

        # Use original agent's savings calculation
        emissions_by_source = {"total": current_emissions_kg}
        savings = self.rec_agent._calculate_savings_potential(
            recommendations,
            emissions_by_source

        # Calculate cost savings if provided
        cost_savings = {}
        if current_energy_cost_usd:
            min_emissions_reduction_pct = (savings["minimum_kg_co2e"] / current_emissions_kg * 100) if current_emissions_kg > 0 else 0
            max_emissions_reduction_pct = (savings["maximum_kg_co2e"] / current_emissions_kg * 100) if current_emissions_kg > 0 else 0

            cost_savings = {
                "minimum_annual_usd": round(current_energy_cost_usd * min_emissions_reduction_pct / 100, 2),
                "maximum_annual_usd": round(current_energy_cost_usd * max_emissions_reduction_pct / 100, 2),
            }

        return {
            "emissions_savings": savings,
            "cost_savings": cost_savings,
            "percentage_reduction_range": savings.get("percentage_range", "N/A"),
        }

    def _generate_implementation_plan_impl(
        self,
        recommendations: List[Dict[str, Any]],
        building_type: Optional[str] = None,
        timeline_months: int = 12,
    ) -> Dict[str, Any]:
        """Tool implementation: Generate implementation plan.

        Args:
            recommendations: Recommendations to plan
            building_type: Building type
            timeline_months: Timeline in months

        Returns:
            Dict with implementation plan
        """
        self._tool_call_count += 1

        # Use original agent's roadmap creation
        roadmap = self.rec_agent._create_roadmap(recommendations)

        # Enhance with timeline
        for phase in roadmap:
            phase_name = phase.get("phase", "")
            if "Quick Wins" in phase_name:
                phase["timeline_months"] = min(6, timeline_months // 3)
            elif "Strategic" in phase_name:
                phase["timeline_months"] = min(12, timeline_months // 2)
            elif "Major" in phase_name:
                phase["timeline_months"] = timeline_months

            # Add steps
            phase["implementation_steps"] = [
                "1. Conduct detailed assessment and vendor selection",
                "2. Secure budget approval and financing",
                "3. Execute installation/implementation",
                "4. Verify performance and document results",
            ]

        return {
            "implementation_roadmap": roadmap,
            "total_timeline_months": timeline_months,
            "building_type": building_type or "general",
            "phases": len(roadmap),
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data.

        Args:
            input_data: Input dictionary

        Returns:
            bool: True if valid
        """
        return self.rec_agent.validate_input(input_data)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute recommendation generation with AI orchestration.

        This method uses ChatSession to orchestrate the recommendation workflow
        while ensuring all numeric calculations use deterministic tools.

        Process:
        1. Validate input
        2. Build AI prompt with analysis requirements
        3. AI uses tools for analysis, ROI calculation, ranking
        4. AI generates natural language summary and insights
        5. Return results with provenance

        Args:
            input_data: Input data with building and emissions data

        Returns:
            AgentResult with recommendations and AI insights
        """
        start_time = DeterministicClock.now()

        # Validate input
        if not self.validate_input(input_data):
            return AgentResult(
                success=False,
                error="Invalid input: building analysis data required",

        try:
            # Run async calculation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._execute_async(input_data))
            finally:
                loop.close()

            # Calculate duration
            duration = (DeterministicClock.now() - start_time).total_seconds()

            # Add performance metadata
            if result.success:
                result.metadata["calculation_time_ms"] = duration * 1000
                result.metadata["ai_calls"] = self._ai_call_count
                result.metadata["tool_calls"] = self._tool_call_count
                result.metadata["total_cost_usd"] = self._total_cost_usd

            return result

        except Exception as e:
            self.logger.error(f"Error in AI recommendation generation: {e}")
            return AgentResult(
                success=False,
                error=f"Failed to generate recommendations: {str(e)}",

    async def _execute_async(self, input_data: Dict[str, Any]) -> AgentResult:
        """Async execution with ChatSession.

        Args:
            input_data: Input data

        Returns:
            AgentResult with recommendations and AI summary
        """
        # Create ChatSession
        session = ChatSession(self.provider)

        # Reset citations for new run
        self._current_citations = []
        self._calculation_citations = []

        # Build AI prompt
        prompt = self._build_prompt(input_data)

        # Prepare messages
        messages = [
            ChatMessage(
                role=Role.system,
                content=(
                    "You are a sustainability advisor for GreenLang. "
                    "You help generate actionable recommendations for reducing carbon emissions. "
                    "IMPORTANT: You must use the provided tools for ALL analysis and calculations. "
                    "Never estimate or guess numbers. Always provide clear, actionable recommendations "
                    "with natural language explanations."
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
                    self.analyze_energy_usage_tool,
                    self.calculate_roi_tool,
                    self.rank_recommendations_tool,
                    self.estimate_savings_tool,
                    self.generate_implementation_plan_tool,
                ],
                budget=budget,
                temperature=0.0,  # Deterministic
                seed=42,  # Reproducible
                tool_choice="auto",

            # Track cost
            self._total_cost_usd += response.usage.cost_usd

            # Extract tool results
            tool_results = self._extract_tool_results(response)

            # Build output from tool results
            output = self._build_output(
                input_data,
                tool_results,
                response.text if self.enable_ai_summary else None,

            return AgentResult(
                success=True,
                data=output,
                metadata={
                    "agent": "RecommendationAgentAI",
                    "provider": response.provider_info.provider,
                    "model": response.provider_info.model,
                    "tokens": response.usage.total_tokens,
                    "cost_usd": response.usage.cost_usd,
                    "tool_calls": len(response.tool_calls),
                    "deterministic": True,
                },

        except BudgetExceeded as e:
            self.logger.error(f"Budget exceeded: {e}")
            return AgentResult(
                success=False,
                error=f"AI budget exceeded: {str(e)}",

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build AI prompt for recommendation generation.

        Args:
            input_data: Input data

        Returns:
            str: Formatted prompt
        """
        emissions_by_source = input_data.get("emissions_by_source", {})
        building_type = input_data.get("building_type", "commercial_office")
        building_age = input_data.get("building_age", 10)
        performance_rating = input_data.get("performance_rating", "Average")
        load_breakdown = input_data.get("load_breakdown", {})
        country = input_data.get("country", "US")

        total_emissions = sum(emissions_by_source.values())

        prompt = f"""Analyze building energy usage and generate top {self.max_recommendations} actionable recommendations:

Building Profile:
- Type: {building_type}
- Age: {building_age} years
- Performance Rating: {performance_rating}
- Location: {country}
- Total Annual Emissions: {total_emissions:,.0f} kg CO2e

Emissions by Source:
"""
        for source, emissions in emissions_by_source.items():
            pct = (emissions / total_emissions * 100) if total_emissions > 0 else 0
            prompt += f"- {source}: {emissions:,.0f} kg ({pct:.1f}%)\n"

        if load_breakdown:
            prompt += "\nLoad Breakdown:\n"
            for load_type, percentage in load_breakdown.items():
                prompt += f"- {load_type}: {percentage*100:.1f}%\n"

        prompt += f"""
Tasks:
1. Use analyze_energy_usage tool to identify inefficiencies and patterns
2. Use calculate_roi tool to calculate payback periods and ROI for potential recommendations
3. Use rank_recommendations tool to prioritize top {self.max_recommendations} by impact and ROI
4. Use estimate_savings tool to calculate potential emissions and cost savings
"""

        if self.enable_implementation_plans:
            prompt += "5. Use generate_implementation_plan tool to create a step-by-step action plan\n"

        prompt += f"""6. Provide an executive summary with:
   - Key findings from usage analysis
   - Top {self.max_recommendations} recommendations with natural language explanations
   - Expected impact and ROI for each recommendation
   - Implementation timeline and priorities
   - Total potential savings (emissions and cost)

IMPORTANT:
- Use tools for ALL analysis and calculations
- Do not estimate or guess any numbers
- Provide clear, actionable recommendations specific to this building
- Include natural language explanations for WHY each recommendation matters
- Prioritize by ROI and impact
- Format numbers clearly (e.g., "15,000 kg" not "15000.0")
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

            if name == "analyze_energy_usage":
                results["analysis"] = self._analyze_energy_usage_impl(**args)
            elif name == "calculate_roi":
                results["roi"] = self._calculate_roi_impl(**args)
            elif name == "rank_recommendations":
                results["ranking"] = self._rank_recommendations_impl(**args)
            elif name == "estimate_savings":
                results["savings"] = self._estimate_savings_impl(**args)
            elif name == "generate_implementation_plan":
                results["implementation"] = self._generate_implementation_plan_impl(**args)

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
            Dict with all recommendation data
        """
        analysis = tool_results.get("analysis", {})
        roi_data = tool_results.get("roi", {})
        ranking = tool_results.get("ranking", {})
        savings = tool_results.get("savings", {})
        implementation = tool_results.get("implementation", {})

        # Get recommendations (from ranking or fallback to original agent)
        recommendations = ranking.get("ranked_recommendations", [])
        if not recommendations:
            # Fallback to original agent
            result = self.rec_agent.execute(input_data)
            if result.success:
                recommendations = result.data.get("recommendations", [])

        # Limit to max_recommendations
        recommendations = recommendations[:self.max_recommendations]

        output = {
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "usage_analysis": analysis,
            "potential_savings": savings.get("emissions_savings", {}),
            "cost_savings": savings.get("cost_savings", {}),
        }

        # Add ROI data if available
        if roi_data:
            output["roi_analysis"] = roi_data

        # Add implementation plan if enabled
        if self.enable_implementation_plans and implementation:
            output["implementation_roadmap"] = implementation.get("implementation_roadmap", [])

        # Add AI summary if enabled
        if ai_summary and self.enable_ai_summary:
            output["ai_summary"] = ai_summary

        # Add quick wins and high impact (for compatibility)
        output["quick_wins"] = [r for r in recommendations if r.get("cost") == "Low"][:3]
        output["high_impact"] = recommendations[:3]  # Top 3 by ranking

        # Add citations for calculations
        if self._calculation_citations:
            output["citations"] = {
                "calculations": [c.dict() for c in self._calculation_citations],
            }

        return output

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary.

        Returns:
            Dict with AI and tool metrics
        """
        return {
            "agent": "RecommendationAgentAI",
            "ai_metrics": {
                "ai_call_count": self._ai_call_count,
                "tool_call_count": self._tool_call_count,
                "total_cost_usd": self._total_cost_usd,
                "avg_cost_per_analysis": (
                    self._total_cost_usd / max(self._ai_call_count, 1)
                ),
            },
            "base_agent_metrics": {
                "agent": "RecommendationAgent",
                "version": self.rec_agent.config.version,
            },
        }
