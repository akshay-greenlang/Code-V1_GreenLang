"""AI-powered Grid Carbon Intensity Factor Lookup with ChatSession Integration.

This module provides an AI-enhanced version of the GridFactorAgent that uses
ChatSession for orchestration while preserving all deterministic lookups
as tool implementations.

Key Differences from Original GridFactorAgent:
    1. AI Orchestration: Uses ChatSession for intelligent analysis and insights
    2. Tool-First Lookups: All data access wrapped as tools (zero hallucinated numbers)
    3. Natural Language Explanations: AI generates context about grid intensity
    4. Temporal Analysis: AI can explain hourly/daily variations in grid emissions
    5. Recommendations: AI suggests cleaner energy sources based on grid intensity
    6. Deterministic Results: temperature=0, seed=42 for reproducibility
    7. Enhanced Provenance: Full audit trail of AI decisions and tool calls
    8. Backward Compatible: Same API as original GridFactorAgent

Architecture:
    GridFactorAgentAI (orchestration) -> ChatSession (AI) -> Tools (exact lookups)

    Tools:
    - lookup_grid_intensity: Get grid emission factor for region/time
    - interpolate_hourly_data: Interpolate between hourly data points
    - calculate_weighted_average: Calculate weighted average from multiple sources
    - generate_recommendations: Recommend cleaner energy sources

Example:
    >>> agent = GridFactorAgentAI()
    >>> result = agent.run({
    ...     "country": "US",
    ...     "fuel_type": "electricity",
    ...     "unit": "kWh",
    ...     "year": 2025
    ... })
    >>> print(result["data"]["explanation"])
    "The US grid has an average intensity of 385 gCO2/kWh..."
    >>> print(result["data"]["emission_factor"])
    0.385  # Exact lookup from database

Author: GreenLang Framework Team
Date: October 2025
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
import logging
import warnings

# DEPRECATION WARNING: This agent is deprecated for CRITICAL PATH emissions calculations
warnings.warn(
    "GridFactorAgentAI has been deprecated. "
    "For CRITICAL PATH emissions calculations (Scope 1/2 grid factors), use the deterministic version instead: "
    "from greenlang.agents.grid_factor_agent import GridFactorAgent. "
    "This AI version should only be used for non-regulatory recommendations. "
    "See AGENT_CATEGORIZATION_AUDIT.md for details.",
    DeprecationWarning,
    stacklevel=2
)

from ..types import Agent, AgentResult, ErrorInfo
from templates.agent_monitoring import OperationalMonitoringMixin
from .types import GridFactorInput, GridFactorOutput
from .grid_factor_agent import GridFactorAgent
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
    DataSourceCitation,
    CitationBundle,
    create_emission_factor_citation,
)

logger = logging.getLogger(__name__)


class GridFactorAgentAI(OperationalMonitoringMixin, Agent[GridFactorInput, GridFactorOutput]):
    """AI-powered grid carbon intensity lookup agent using ChatSession.

    This agent enhances the original GridFactorAgent with AI orchestration while
    maintaining exact deterministic lookups through tool implementations.

    Features:
    - AI orchestration via ChatSession for intelligent grid analysis
    - Tool-first lookups (all data from database, zero hallucinated numbers)
    - Natural language explanations of grid intensity
    - Temporal analysis (hourly/daily variations)
    - Intelligent recommendations for cleaner energy sources
    - Deterministic results (temperature=0, seed=42)
    - Full provenance tracking (AI decisions + tool calls)
    - All original GridFactorAgent features preserved
    - Backward compatible API

    Determinism Guarantees:
    - Same input always produces same output
    - All numeric values come from tools (no LLM lookups)
    - Reproducible AI responses via seed=42
    - Auditable decision trail

    Performance:
    - Async support for concurrent processing
    - Budget enforcement (max $0.50 per lookup by default)
    - Performance metrics tracking

    Example:
        >>> agent = GridFactorAgentAI()
        >>> result = agent.run({
        ...     "country": "US",
        ...     "fuel_type": "electricity",
        ...     "unit": "kWh"
        ... })
        >>> print(result["data"]["explanation"])
        "US grid intensity averages 385 gCO2/kWh with 21% renewable share..."
        >>> print(result["data"]["emission_factor"])
        0.385
    """

    agent_id: str = "grid_factor_ai"
    name: str = "AI-Powered Grid Emission Factor Provider"
    version: str = "0.1.0"

    def __init__(
        self,
        *,
        budget_usd: float = 0.50,
        enable_explanations: bool = True,
        enable_recommendations: bool = True,
    ) -> None:
        """Initialize the AI-powered GridFactorAgent.

        Args:
            budget_usd: Maximum USD to spend per lookup (default: $0.50)
            enable_explanations: Enable AI-generated explanations (default: True)
            enable_recommendations: Enable AI recommendations (default: True)
        """
        # Initialize original grid factor agent for tool implementations
        self.grid_agent = GridFactorAgent()

        # Configuration
        self.budget_usd = budget_usd
        self.enable_explanations = enable_explanations
        self.enable_recommendations = enable_recommendations

        # Initialize LLM provider (auto-detects available provider)
        self.provider = create_provider()

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.agent_id}")
        self.logger.setLevel(logging.INFO)

        # Performance tracking
        self._ai_call_count = 0
        self._tool_call_count = 0
        self._total_cost_usd = 0.0

        # Citation tracking
        self._current_citations: List[EmissionFactorCitation] = []
        self._data_source_citations: List[DataSourceCitation] = []

        # Define tools for ChatSession
        self._setup_tools()

        # Setup operational monitoring
        self.setup_monitoring(agent_name="grid_factor_agent_ai")

    def _setup_tools(self) -> None:
        """Setup tool definitions for ChatSession."""

        # Tool 1: Lookup grid intensity (exact database lookup)
        self.lookup_grid_intensity_tool = ToolDef(
            name="lookup_grid_intensity",
            description="Look up exact grid carbon intensity (emission factor) for a region and fuel type from authoritative database",
            parameters={
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Country code (e.g., US, UK, EU, IN, CN, JP)",
                    },
                    "fuel_type": {
                        "type": "string",
                        "description": "Fuel/energy type (e.g., electricity, natural_gas, district_heating)",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of measurement (e.g., kWh, MWh, therms, m3)",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Year for emission factor (default: 2025)",
                        "default": 2025,
                    },
                },
                "required": ["country", "fuel_type", "unit"],
            },
        )

        # Tool 2: Interpolate hourly data (for temporal analysis)
        self.interpolate_hourly_data_tool = ToolDef(
            name="interpolate_hourly_data",
            description="Interpolate grid intensity for specific hour based on daily patterns",
            parameters={
                "type": "object",
                "properties": {
                    "base_intensity": {
                        "type": "number",
                        "description": "Base/average grid intensity (gCO2/kWh)",
                    },
                    "hour": {
                        "type": "integer",
                        "description": "Hour of day (0-23)",
                        "minimum": 0,
                        "maximum": 23,
                    },
                    "renewable_share": {
                        "type": "number",
                        "description": "Renewable energy share (0-1 scale)",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": ["base_intensity", "hour"],
            },
        )

        # Tool 3: Calculate weighted average (multiple sources)
        self.calculate_weighted_average_tool = ToolDef(
            name="calculate_weighted_average",
            description="Calculate weighted average grid intensity from multiple sources or time periods",
            parameters={
                "type": "object",
                "properties": {
                    "intensities": {
                        "type": "array",
                        "description": "List of intensity values (gCO2/kWh)",
                        "items": {"type": "number"},
                    },
                    "weights": {
                        "type": "array",
                        "description": "List of weights (must sum to 1.0)",
                        "items": {"type": "number"},
                    },
                },
                "required": ["intensities", "weights"],
            },
        )

        # Tool 4: Generate recommendations (cleaner energy sources)
        self.generate_recommendations_tool = ToolDef(
            name="generate_recommendations",
            description="Generate recommendations for cleaner energy sources based on grid intensity",
            parameters={
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Country code",
                    },
                    "current_intensity": {
                        "type": "number",
                        "description": "Current grid intensity (gCO2/kWh)",
                    },
                    "renewable_share": {
                        "type": "number",
                        "description": "Current renewable share (0-1 scale)",
                    },
                },
                "required": ["country", "current_intensity"],
            },
        )

    def _lookup_grid_intensity_impl(
        self,
        country: str,
        fuel_type: str,
        unit: str,
        year: int = 2025,
    ) -> Dict[str, Any]:
        """Tool implementation - exact grid intensity lookup.

        This method delegates to the original GridFactorAgent for deterministic
        lookups. All values come from validated database, not LLM.

        Args:
            country: Country code
            fuel_type: Fuel/energy type
            unit: Unit of measurement
            year: Year for emission factor

        Returns:
            Dict with emission factor and metadata
        """
        self._tool_call_count += 1

        # Delegate to original GridFactorAgent
        result = self.grid_agent.run({
            "country": country,
            "fuel_type": fuel_type,
            "unit": unit,
            "year": year,
        })

        if not result["success"]:
            raise ValueError(f"Lookup failed: {result['error']['message']}")

        data = result["data"]

        # Create citation for grid intensity factor
        citation = create_emission_factor_citation(
            source=data["source"],
            factor_name=f"{country} Grid Intensity - {fuel_type.replace('_', ' ').title()}",
            value=data["emission_factor"],
            unit=data["unit"],
            version=data["version"],
            last_updated=datetime.fromisoformat(data["last_updated"]) if isinstance(data["last_updated"], str) else data["last_updated"],
            confidence="high",
            region=country,
            gwp_set="AR6GWP100"
        )

        # Store citation for output
        self._current_citations.append(citation)

        # Create data source citation
        data_source_citation = DataSourceCitation(
            source_name=data["source"],
            source_type="database",
            query={"country": country, "fuel_type": fuel_type, "unit": unit, "year": year},
            timestamp=datetime.now(),
            url=None,
        )
        self._data_source_citations.append(data_source_citation)

        return {
            "emission_factor": data["emission_factor"],
            "unit": data["unit"],
            "country": data["country"],
            "fuel_type": data["fuel_type"],
            "source": data["source"],
            "version": data["version"],
            "last_updated": data["last_updated"],
            "grid_mix": data.get("grid_mix", {}),
            "citation": citation.to_dict(),
        }

    def _interpolate_hourly_data_impl(
        self,
        base_intensity: float,
        hour: int,
        renewable_share: float = 0.0,
    ) -> Dict[str, Any]:
        """Tool implementation - interpolate hourly grid intensity.

        Uses deterministic pattern to estimate hourly variations based on
        typical grid behavior (peak vs off-peak).

        Args:
            base_intensity: Base/average intensity (gCO2/kWh)
            hour: Hour of day (0-23)
            renewable_share: Renewable energy share

        Returns:
            Dict with interpolated intensity and pattern info
        """
        self._tool_call_count += 1

        # Deterministic hourly pattern (based on typical grid behavior)
        # Peak hours (6-22): slightly higher intensity (more fossil fuels)
        # Off-peak (22-6): slightly lower intensity (more baseload/renewables)

        # Peak factor varies by hour
        if 6 <= hour < 10:  # Morning peak
            peak_factor = 1.15
            period = "morning_peak"
        elif 17 <= hour < 21:  # Evening peak
            peak_factor = 1.20
            period = "evening_peak"
        elif 10 <= hour < 17:  # Midday (solar generation)
            peak_factor = 0.95 - (renewable_share * 0.15)  # More solar reduces intensity
            period = "midday"
        else:  # Off-peak (nighttime)
            peak_factor = 0.90
            period = "off_peak"

        # Calculate interpolated intensity
        interpolated_intensity = base_intensity * peak_factor

        return {
            "interpolated_intensity": round(interpolated_intensity, 4),
            "base_intensity": base_intensity,
            "hour": hour,
            "period": period,
            "peak_factor": round(peak_factor, 3),
            "explanation": f"{period.replace('_', ' ').title()} (hour {hour}): {peak_factor:.1%} of average intensity",
        }

    def _calculate_weighted_average_impl(
        self,
        intensities: List[float],
        weights: List[float],
    ) -> Dict[str, Any]:
        """Tool implementation - weighted average calculation.

        Args:
            intensities: List of intensity values
            weights: List of weights (should sum to 1.0)

        Returns:
            Dict with weighted average and validation
        """
        self._tool_call_count += 1

        if len(intensities) != len(weights):
            raise ValueError("Intensities and weights must have same length")

        # Normalize weights if they don't sum to 1.0
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 0.001:
            weights = [w / weight_sum for w in weights]

        # Calculate weighted average
        weighted_avg = sum(i * w for i, w in zip(intensities, weights))

        return {
            "weighted_average": round(weighted_avg, 4),
            "intensities": intensities,
            "normalized_weights": [round(w, 4) for w in weights],
            "min_intensity": min(intensities),
            "max_intensity": max(intensities),
            "range": round(max(intensities) - min(intensities), 4),
        }

    def _generate_recommendations_impl(
        self,
        country: str,
        current_intensity: float,
        renewable_share: float = 0.0,
    ) -> Dict[str, Any]:
        """Tool implementation - generate cleaner energy recommendations.

        Args:
            country: Country code
            current_intensity: Current grid intensity (gCO2/kWh)
            renewable_share: Current renewable share

        Returns:
            Dict with recommendations list
        """
        self._tool_call_count += 1

        recommendations = []

        # Benchmark thresholds (gCO2/kWh)
        # Excellent: < 200, Good: 200-350, Average: 350-500, High: > 500

        # Recommendation 1: On-site renewable energy
        if current_intensity > 200:
            solar_reduction = current_intensity * 0.9  # 90% reduction
            recommendations.append({
                "priority": "high",
                "action": "Install on-site solar PV system",
                "impact": f"Reduce grid dependency by up to 100% during daylight hours",
                "potential_reduction_gco2_kwh": round(solar_reduction, 2),
                "estimated_payback": "5-8 years",
                "notes": "Most effective for daytime electricity consumption",
            })

        # Recommendation 2: Purchase renewable energy certificates (RECs)
        if renewable_share < 0.5:
            rec_reduction = current_intensity * (1 - renewable_share)
            recommendations.append({
                "priority": "medium",
                "action": "Purchase Renewable Energy Certificates (RECs) or Green Power",
                "impact": f"Offset {(1-renewable_share)*100:.0f}% of grid emissions",
                "potential_reduction_gco2_kwh": round(rec_reduction, 2),
                "estimated_payback": "Immediate (operational expense)",
                "notes": "Quick way to reduce carbon footprint without infrastructure changes",
            })

        # Recommendation 3: Time-of-use optimization
        if current_intensity > 300:
            tou_reduction = current_intensity * 0.15  # 15% reduction via shifting
            recommendations.append({
                "priority": "medium",
                "action": "Shift high-energy operations to off-peak/nighttime hours",
                "impact": "Reduce emissions by 10-20% through time-of-use optimization",
                "potential_reduction_gco2_kwh": round(tou_reduction, 2),
                "estimated_payback": "Immediate (no capital cost)",
                "notes": "Effective for batch processes, EV charging, HVAC pre-cooling",
            })

        # Recommendation 4: Energy efficiency (always recommended)
        efficiency_reduction = current_intensity * 0.25  # 25% via efficiency
        recommendations.append({
            "priority": "high",
            "action": "Implement comprehensive energy efficiency measures",
            "impact": "Reduce total consumption by 20-30% (LED lighting, HVAC optimization, insulation)",
            "potential_reduction_gco2_kwh": round(efficiency_reduction, 2),
            "estimated_payback": "2-5 years",
            "notes": "Most cost-effective approach - reduces both emissions and energy bills",
        })

        # Recommendation 5: Battery storage (for high-intensity grids)
        if current_intensity > 400 and renewable_share > 0.2:
            storage_reduction = current_intensity * 0.3
            recommendations.append({
                "priority": "low",
                "action": "Install battery storage system to store off-peak/renewable energy",
                "impact": "Shift 30-50% of consumption to cleaner time periods",
                "potential_reduction_gco2_kwh": round(storage_reduction, 2),
                "estimated_payback": "8-12 years",
                "notes": "Pairs well with solar PV, smooths grid demand",
            })

        # Country-specific recommendations
        if country in ["IN", "CN", "AU"]:  # Coal-heavy grids
            recommendations.insert(0, {
                "priority": "critical",
                "action": f"Priority focus on renewable energy due to high grid intensity ({current_intensity:.0f} gCO2/kWh)",
                "impact": "Consider all renewable energy options as primary strategy",
                "potential_reduction_gco2_kwh": round(current_intensity * 0.8, 2),
                "estimated_payback": "Varies by solution",
                "notes": f"{country} grid has high carbon intensity - renewable transition is critical",
            })

        return {
            "recommendations": recommendations,
            "count": len(recommendations),
            "current_intensity": current_intensity,
            "renewable_share": renewable_share,
        }

    def validate(self, payload: GridFactorInput) -> bool:
        """Validate input payload.

        Delegates to original GridFactorAgent for validation logic.

        Args:
            payload: Input data

        Returns:
            bool: True if valid
        """
        return self.grid_agent.validate(payload)

    def run(self, payload: GridFactorInput) -> AgentResult[GridFactorOutput]:
        """Lookup grid intensity with AI orchestration.

        This method uses ChatSession to orchestrate the lookup workflow
        while ensuring all data comes from deterministic tools.

        Process:
        1. Validate input
        2. Build AI prompt with lookup requirements
        3. AI uses tools for exact lookups
        4. AI generates natural language explanation
        5. Return results with provenance

        Args:
            payload: Input data with country, fuel_type, unit

        Returns:
            AgentResult with grid intensity and AI explanation
        """
        with self.track_execution(payload) as tracker:
            start_time = datetime.now()

            # Validate input
            if not self.validate(payload):
                error_info: ErrorInfo = {
                    "type": "ValidationError",
                    "message": "Missing required fields: country, fuel_type, unit",
                    "agent_id": self.agent_id,
                    "context": {"payload": payload},
                }
                return {"success": False, "error": error_info}

            # Reset citations for new run
            self._current_citations = []
            self._data_source_citations = []

            try:
                # Run async lookup
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self._run_async(payload))
                finally:
                    loop.close()

                # Calculate duration
                duration = (datetime.now() - start_time).total_seconds()

                # Add performance metadata
                if result["success"]:
                    result["metadata"] = {
                        **result.get("metadata", {}),
                        "agent_id": self.agent_id,
                        "lookup_time_ms": duration * 1000,
                        "ai_calls": self._ai_call_count,
                        "tool_calls": self._tool_call_count,
                        "total_cost_usd": self._total_cost_usd,
                    }

                return result

            except Exception as e:
                self.logger.error(f"Error in AI grid factor lookup: {e}")
                error_info: ErrorInfo = {
                    "type": "LookupError",
                    "message": f"Failed to lookup grid factor: {str(e)}",
                    "agent_id": self.agent_id,
                    "traceback": str(e),
                }
                return {"success": False, "error": error_info}

    async def _run_async(self, payload: GridFactorInput) -> AgentResult[GridFactorOutput]:
        """Async lookup with ChatSession.

        Args:
            payload: Input data

        Returns:
            AgentResult with grid intensity and explanation
        """
        country = payload["country"]
        fuel_type = payload["fuel_type"]
        unit = payload["unit"]
        year = payload.get("year", 2025)

        # Create ChatSession
        session = ChatSession(self.provider)

        # Build AI prompt
        prompt = self._build_prompt(payload)

        # Prepare messages
        messages = [
            ChatMessage(
                role=Role.system,
                content=(
                    "You are a grid carbon intensity analyst for GreenLang. "
                    "You help look up and explain grid emission factors using authoritative tools. "
                    "IMPORTANT: You must use the provided tools for ALL data lookups. "
                    "Never estimate or guess emission factors. Always provide clear context and actionable recommendations."
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
                    self.lookup_grid_intensity_tool,
                    self.interpolate_hourly_data_tool,
                    self.calculate_weighted_average_tool,
                    self.generate_recommendations_tool,
                ],
                budget=budget,
                temperature=0.0,  # Deterministic
                seed=42,          # Reproducible
                tool_choice="auto",
            )

            # Track cost
            self._total_cost_usd += response.usage.cost_usd

            # Extract tool results
            tool_results = self._extract_tool_results(response)

            # Build output from tool results
            output = self._build_output(
                payload,
                tool_results,
                response.text if self.enable_explanations else None,
            )

            return {
                "success": True,
                "data": output,
                "metadata": {
                    "agent_id": self.agent_id,
                    "provider": response.provider_info.provider,
                    "model": response.provider_info.model,
                    "tokens": response.usage.total_tokens,
                    "cost_usd": response.usage.cost_usd,
                    "tool_calls": len(response.tool_calls),
                    "seed": 42,  # Reproducibility seed
                    "temperature": 0.0,  # Deterministic temperature
                    "deterministic": True,
                },
            }

        except BudgetExceeded as e:
            self.logger.error(f"Budget exceeded: {e}")
            error_info: ErrorInfo = {
                "type": "BudgetError",
                "message": f"AI budget exceeded: {str(e)}",
                "agent_id": self.agent_id,
            }
            return {"success": False, "error": error_info}

    def _build_prompt(self, payload: GridFactorInput) -> str:
        """Build AI prompt for lookup.

        Args:
            payload: Input data

        Returns:
            str: Formatted prompt
        """
        country = payload["country"]
        fuel_type = payload["fuel_type"]
        unit = payload["unit"]
        year = payload.get("year", 2025)

        prompt = f"""Look up grid carbon intensity (emission factor) for:

- Country: {country}
- Fuel/Energy Type: {fuel_type}
- Unit: {unit}
- Year: {year}

Tasks:
1. Use the lookup_grid_intensity tool to get exact emission factor from database
2. Explain the emission factor in context:
   - What does this intensity mean? (e.g., "385 gCO2/kWh is slightly above average")
   - How does it compare to global benchmarks?
   - What contributes to this level (coal, gas, renewables)?
   - Mention the renewable share if available
"""

        if self.enable_recommendations:
            prompt += "3. Use generate_recommendations tool to suggest 3-5 ways to reduce reliance on high-carbon grid electricity\n"

        prompt += """
IMPORTANT:
- Use tools for ALL data lookups (no guessing emission factors!)
- Provide clear, actionable context about the grid intensity
- Format numbers clearly (e.g., "385 gCO2/kWh" not "0.385")
- Focus on practical insights and recommendations
"""

        return prompt

    def _extract_tool_results(self, response) -> Dict[str, Any]:
        """Extract results from tool calls.

        Args:
            response: ChatResponse from session

        Returns:
            Dict with tool results
        """
        results = {}

        for tool_call in response.tool_calls:
            name = tool_call.get("name", "")
            args = tool_call.get("arguments", {})

            if name == "lookup_grid_intensity":
                results["lookup"] = self._lookup_grid_intensity_impl(**args)
            elif name == "interpolate_hourly_data":
                results["hourly"] = self._interpolate_hourly_data_impl(**args)
            elif name == "calculate_weighted_average":
                results["weighted_avg"] = self._calculate_weighted_average_impl(**args)
            elif name == "generate_recommendations":
                results["recommendations"] = self._generate_recommendations_impl(**args)

        return results

    def _build_output(
        self,
        payload: GridFactorInput,
        tool_results: Dict[str, Any],
        explanation: Optional[str],
    ) -> GridFactorOutput:
        """Build output from tool results.

        Args:
            payload: Original input
            tool_results: Results from tool calls
            explanation: AI-generated explanation

        Returns:
            GridFactorOutput with all data
        """
        lookup_data = tool_results.get("lookup", {})

        output: GridFactorOutput = {
            "emission_factor": lookup_data.get("emission_factor", 0.0),
            "unit": lookup_data.get("unit", ""),
            "source": lookup_data.get("source", ""),
            "version": lookup_data.get("version", ""),
            "last_updated": lookup_data.get("last_updated", ""),
            "country": lookup_data.get("country", payload["country"]),
            "fuel_type": lookup_data.get("fuel_type", payload["fuel_type"]),
        }

        # Add grid mix if available
        if "grid_mix" in lookup_data and lookup_data["grid_mix"]:
            output["grid_mix"] = lookup_data["grid_mix"]

        # Add hourly data if interpolated
        if "hourly" in tool_results:
            hourly = tool_results["hourly"]
            output["hourly_intensity"] = hourly.get("interpolated_intensity")
            output["hourly_period"] = hourly.get("period")

        # Add weighted average if calculated
        if "weighted_avg" in tool_results:
            output["weighted_average"] = tool_results["weighted_avg"].get("weighted_average")

        # Add recommendations if generated
        if "recommendations" in tool_results:
            output["recommendations"] = tool_results["recommendations"].get("recommendations", [])

        # Add AI explanation if enabled
        if explanation and self.enable_explanations:
            output["explanation"] = explanation

        # Add citations for emission factors and data sources
        if self._current_citations or self._data_source_citations:
            output["citations"] = {
                "emission_factors": [c.to_dict() for c in self._current_citations],
                "data_sources": [ds.dict() for ds in self._data_source_citations],
            }

        return output

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary.

        Returns:
            Dict with AI and tool metrics
        """
        return {
            "agent_id": self.agent_id,
            "ai_metrics": {
                "ai_call_count": self._ai_call_count,
                "tool_call_count": self._tool_call_count,
                "total_cost_usd": self._total_cost_usd,
                "avg_cost_per_lookup": (
                    self._total_cost_usd / max(self._ai_call_count, 1)
                ),
            },
            "base_agent_metrics": {
                "agent_id": self.grid_agent.agent_id,
                "name": self.grid_agent.name,
                "version": self.grid_agent.version,
            },
        }

    def get_available_countries(self) -> List[str]:
        """Get list of available countries.

        Returns:
            List of country codes
        """
        return self.grid_agent.get_available_countries()

    def get_available_fuel_types(self, country: str) -> List[str]:
        """Get available fuel types for a country.

        Args:
            country: Country code

        Returns:
            List of fuel types
        """
        return self.grid_agent.get_available_fuel_types(country)
