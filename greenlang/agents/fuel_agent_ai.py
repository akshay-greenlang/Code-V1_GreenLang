# -*- coding: utf-8 -*-
"""AI-powered Fuel Emissions Calculator with ChatSession Integration.

This module provides an AI-enhanced version of the FuelAgent that uses
ChatSession for orchestration while preserving all deterministic calculations
as tool implementations.

Key Differences from Original FuelAgent:
    1. AI Orchestration: Uses ChatSession for natural language interaction
    2. Tool-First Numerics: All calculations wrapped as tools (zero hallucinated numbers)
    3. Natural Explanations: AI generates human-readable explanations
    4. Deterministic Results: temperature=0, seed=42 for reproducibility
    5. Enhanced Provenance: Full audit trail of AI decisions and tool calls
    6. Backward Compatible: Same API as original FuelAgent

Architecture:
    FuelAgentAI (orchestration) -> ChatSession (AI) -> Tools (exact calculations)

Example:
    >>> agent = FuelAgentAI()
    >>> result = agent.run({
    ...     "fuel_type": "natural_gas",
    ...     "amount": 1000,
    ...     "unit": "therms",
    ...     "country": "US"
    ... })
    >>> print(result["data"]["explanation"])
    "For 1000 therms of natural gas consumption in the US..."
    >>> print(result["data"]["co2e_emissions_kg"])
    5310.0  # Exact calculation from tool

Author: GreenLang Framework Team
Date: October 2025
Spec: GL_Mak_Updates_2025.md (lines 2055-2194)
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
import logging
import warnings
from greenlang.determinism import DeterministicClock

# DEPRECATION WARNING: This agent is deprecated for CRITICAL PATH emissions calculations
warnings.warn(
    "FuelAgentAI has been deprecated. "
    "For CRITICAL PATH emissions calculations (Scope 1/2 fuel emissions), use the deterministic version instead: "
    "from greenlang.agents.fuel_agent import FuelAgent. "
    "This AI version should only be used for non-regulatory recommendations. "
    "See AGENT_CATEGORIZATION_AUDIT.md for details.",
    DeprecationWarning,
    stacklevel=2
)

from ..types import Agent, AgentResult, ErrorInfo
from .types import FuelInput, FuelOutput
from .fuel_agent import FuelAgent
from greenlang.exceptions import ExecutionError, MissingData
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
    CitationBundle,
    create_emission_factor_citation,
)


logger = logging.getLogger(__name__)


class FuelAgentAI(Agent[FuelInput, FuelOutput]):
    """AI-powered fuel emissions calculator using ChatSession.

    This agent enhances the original FuelAgent with AI orchestration while
    maintaining exact deterministic calculations through tool implementations.

    Features:
    - AI orchestration via ChatSession for natural language processing
    - Tool-first numerics (all calculations use tools, zero hallucinated numbers)
    - Natural language explanations of calculations
    - Deterministic results (temperature=0, seed=42)
    - Full provenance tracking (AI decisions + tool calls)
    - All original FuelAgent features preserved
    - Backward compatible API

    Determinism Guarantees:
    - Same input always produces same output
    - All numeric values come from tools (no LLM math)
    - Reproducible AI responses via seed=42
    - Auditable decision trail

    Performance:
    - Async support for concurrent processing
    - Budget enforcement (max $0.50 per calculation by default)
    - Caching of emission factors
    - Performance metrics tracking

    Example:
        >>> agent = FuelAgentAI()
        >>> result = agent.run({
        ...     "fuel_type": "natural_gas",
        ...     "amount": 1000,
        ...     "unit": "therms",
        ...     "country": "US"
        ... })
        >>> print(result["data"]["explanation"])
        "Calculated 5,310 kg CO2e emissions from 1000 therms of natural gas..."
        >>> print(result["data"]["co2e_emissions_kg"])
        5310.0
    """

    agent_id: str = "fuel_ai"
    name: str = "AI-Powered Fuel Emissions Calculator"
    version: str = "0.1.0"

    def __init__(
        self,
        *,
        budget_usd: float = 0.50,
        enable_explanations: bool = True,
        enable_recommendations: bool = True,
    ) -> None:
        """Initialize the AI-powered FuelAgent.

        Args:
            budget_usd: Maximum USD to spend per calculation (default: $0.50)
            enable_explanations: Enable AI-generated explanations (default: True)
            enable_recommendations: Enable AI recommendations (default: True)
        """
        # Initialize original fuel agent for tool implementations
        self.fuel_agent = FuelAgent()

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

        # Define tools for ChatSession
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Setup tool definitions for ChatSession."""

        # Tool 1: Calculate emissions (exact calculation)
        self.calculate_emissions_tool = ToolDef(
            name="calculate_emissions",
            description="Calculate exact CO2e emissions from fuel consumption using authoritative emission factors",
            parameters={
                "type": "object",
                "properties": {
                    "fuel_type": {
                        "type": "string",
                        "description": "Type of fuel (e.g., natural_gas, diesel, electricity)",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Fuel consumption amount",
                        "minimum": 0,
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of measurement (e.g., therms, gallons, kWh)",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code (e.g., US, UK, EU)",
                        "default": "US",
                    },
                    "renewable_percentage": {
                        "type": "number",
                        "description": "Renewable energy percentage for offsetting (0-100)",
                        "minimum": 0,
                        "maximum": 100,
                        "default": 0,
                    },
                    "efficiency": {
                        "type": "number",
                        "description": "Equipment efficiency factor (0-1 scale)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 1.0,
                    },
                },
                "required": ["fuel_type", "amount", "unit"],
            },
        )

        # Tool 2: Lookup emission factor (database access)
        self.lookup_emission_factor_tool = ToolDef(
            name="lookup_emission_factor",
            description="Look up authoritative emission factor for fuel type and location",
            parameters={
                "type": "object",
                "properties": {
                    "fuel_type": {
                        "type": "string",
                        "description": "Type of fuel",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of measurement",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code",
                        "default": "US",
                    },
                },
                "required": ["fuel_type", "unit"],
            },
        )

        # Tool 3: Generate recommendations (deterministic rules)
        self.generate_recommendations_tool = ToolDef(
            name="generate_recommendations",
            description="Generate fuel switching and efficiency improvement recommendations",
            parameters={
                "type": "object",
                "properties": {
                    "fuel_type": {
                        "type": "string",
                        "description": "Current fuel type",
                    },
                    "emissions_kg": {
                        "type": "number",
                        "description": "Calculated emissions in kg CO2e",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code",
                        "default": "US",
                    },
                },
                "required": ["fuel_type", "emissions_kg"],
            },
        )

    def _calculate_emissions_impl(
        self,
        fuel_type: str,
        amount: float,
        unit: str,
        country: str = "US",
        renewable_percentage: float = 0.0,
        efficiency: float = 1.0,
    ) -> Dict[str, Any]:
        """Tool implementation - exact emissions calculation.

        This method delegates to the original FuelAgent for deterministic
        calculations. All numeric results come from validated code, not LLM.

        Args:
            fuel_type: Type of fuel
            amount: Consumption amount
            unit: Unit of measurement
            country: Country code
            renewable_percentage: Renewable offset percentage
            efficiency: Equipment efficiency

        Returns:
            Dict with emissions data and calculation details
        """
        self._tool_call_count += 1

        # Delegate to original FuelAgent
        result = self.fuel_agent.run({
            "fuel_type": fuel_type,
            "amount": amount,
            "unit": unit,
            "country": country,
            "renewable_percentage": renewable_percentage,
            "efficiency": efficiency,
        })

        if not result["success"]:
            raise ExecutionError(
                message="Fuel emissions calculation failed",
                agent_name=self.agent_id,
                context={
                    "fuel_type": fuel_type,
                    "amount": amount,
                    "unit": unit,
                    "country": country,
                    "error_details": result.get("error", {})
                },
                step="calculate_emissions",
                cause=Exception(result['error']['message'])
            )

        data = result["data"]

        return {
            "emissions_kg_co2e": data["co2e_emissions_kg"],
            "emission_factor": data["emission_factor"],
            "emission_factor_unit": data["emission_factor_unit"],
            "scope": data["scope"],
            "energy_content_mmbtu": data.get("energy_content_mmbtu", 0.0),
            "calculation": result["metadata"]["calculation"],
        }

    def _lookup_emission_factor_impl(
        self,
        fuel_type: str,
        unit: str,
        country: str = "US",
    ) -> Dict[str, Any]:
        """Tool implementation - database lookup.

        Args:
            fuel_type: Type of fuel
            unit: Unit of measurement
            country: Country code

        Returns:
            Dict with emission factor and metadata
        """
        self._tool_call_count += 1

        # Use cached lookup from original agent
        emission_factor = self.fuel_agent._get_cached_emission_factor(
            fuel_type, unit, country
        )

        if emission_factor is None:
            raise MissingData(
                message=f"No emission factor found for {fuel_type} ({unit}) in {country}",
                context={
                    "fuel_type": fuel_type,
                    "unit": unit,
                    "country": country,
                    "agent_id": self.agent_id
                },
                data_type="emission_factor",
                missing_fields=["emission_factor"]
            )

        # Create citation for this emission factor
        citation = create_emission_factor_citation(
            source="GreenLang Emission Factors Database",
            factor_name=f"{fuel_type.replace('_', ' ').title()} Combustion",
            value=emission_factor,
            unit=f"kgCO2e/{unit}",
            version="2025.1",
            last_updated=datetime(2025, 1, 15),
            confidence="high",
            region=country,
            gwp_set="AR6GWP100"
        )

        # Store citation for output
        self._current_citations.append(citation)

        return {
            "emission_factor": emission_factor,
            "unit": f"kgCO2e/{unit}",
            "fuel_type": fuel_type,
            "country": country,
            "source": "GreenLang Emission Factors Database",
            "citation": citation.to_dict(),
        }

    def _generate_recommendations_impl(
        self,
        fuel_type: str,
        emissions_kg: float,
        country: str = "US",
    ) -> Dict[str, Any]:
        """Tool implementation - generate recommendations.

        Args:
            fuel_type: Current fuel type
            emissions_kg: Calculated emissions
            country: Country code

        Returns:
            Dict with recommendations list
        """
        self._tool_call_count += 1

        # Delegate to original agent's recommendation logic
        recommendations = self.fuel_agent._generate_fuel_recommendations(
            fuel_type=fuel_type,
            amount=0,  # Not used in recommendation logic
            unit="",   # Not used in recommendation logic
            emissions_kg=emissions_kg,
            country=country,
        )

        return {
            "recommendations": recommendations,
            "count": len(recommendations),
        }

    def validate(self, payload: FuelInput) -> bool:
        """Validate input payload.

        Delegates to original FuelAgent for validation logic.

        Args:
            payload: Input data

        Returns:
            bool: True if valid
        """
        return self.fuel_agent.validate(payload)

    def run(self, payload: FuelInput) -> AgentResult[FuelOutput]:
        """Calculate emissions with AI orchestration.

        This method uses ChatSession to orchestrate the calculation workflow
        while ensuring all numeric calculations use deterministic tools.

        Process:
        1. Validate input
        2. Build AI prompt with calculation requirements
        3. AI uses tools for exact calculations
        4. AI generates natural language explanation
        5. Return results with provenance

        Args:
            payload: Input data with fuel consumption details

        Returns:
            AgentResult with emissions data and AI explanation
        """
        start_time = DeterministicClock.now()

        # Validate input
        if not self.validate(payload):
            error_info: ErrorInfo = {
                "type": "ValidationError",
                "message": "Invalid input payload",
                "agent_id": self.agent_id,
                "context": {"payload": payload},
            }
            return {"success": False, "error": error_info}

        # Reset citations for new run
        self._current_citations = []

        try:
            # Run async calculation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._run_async(payload))
            finally:
                loop.close()

            # Calculate duration
            duration = (DeterministicClock.now() - start_time).total_seconds()

            # Add performance metadata
            if result["success"]:
                result["metadata"] = {
                    **result.get("metadata", {}),
                    "agent_id": self.agent_id,
                    "calculation_time_ms": duration * 1000,
                    "ai_calls": self._ai_call_count,
                    "tool_calls": self._tool_call_count,
                    "total_cost_usd": self._total_cost_usd,
                    "seed": 42,  # Reproducibility seed
                    "temperature": 0.0,  # Deterministic temperature
                    "deterministic": True,  # Deterministic execution flag
                }

            return result

        except Exception as e:
            self.logger.error(f"Error in AI fuel calculation: {e}")
            error_info: ErrorInfo = {
                "type": "CalculationError",
                "message": f"Failed to calculate fuel emissions: {str(e)}",
                "agent_id": self.agent_id,
                "traceback": str(e),
            }
            return {"success": False, "error": error_info}

    async def _run_async(self, payload: FuelInput) -> AgentResult[FuelOutput]:
        """Async calculation with ChatSession.

        Args:
            payload: Input data

        Returns:
            AgentResult with emissions and explanation
        """
        fuel_type = payload["fuel_type"]
        amount = payload["amount"]
        unit = payload["unit"]
        country = payload.get("country", "US")
        renewable_percentage = payload.get("renewable_percentage", 0)
        efficiency = payload.get("efficiency", 1.0)

        # Create ChatSession with tools
        session = ChatSession(self.provider)

        # Build AI prompt
        prompt = self._build_prompt(payload)

        # Prepare messages
        messages = [
            ChatMessage(
                role=Role.system,
                content=(
                    "You are a climate analyst assistant for GreenLang. "
                    "You help calculate fuel emissions using authoritative tools. "
                    "IMPORTANT: You must use the provided tools for ALL numeric calculations. "
                    "Never estimate or guess numbers. Always explain your calculations clearly."
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
                    self.calculate_emissions_tool,
                    self.lookup_emission_factor_tool,
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
                    "provider": response.provider_info.provider,
                    "model": response.provider_info.model,
                    "tokens": response.usage.total_tokens,
                    "cost_usd": response.usage.cost_usd,
                    "tool_calls": len(response.tool_calls),
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

    def _build_prompt(self, payload: FuelInput) -> str:
        """Build AI prompt for calculation.

        Args:
            payload: Input data

        Returns:
            str: Formatted prompt
        """
        fuel_type = payload["fuel_type"]
        amount = payload["amount"]
        unit = payload["unit"]
        country = payload.get("country", "US")
        renewable_percentage = payload.get("renewable_percentage", 0)
        efficiency = payload.get("efficiency", 1.0)

        prompt = f"""Calculate CO2e emissions for fuel consumption:

- Fuel type: {fuel_type}
- Consumption: {amount} {unit}
- Location: {country}"""

        if renewable_percentage > 0:
            prompt += f"\n- Renewable offset: {renewable_percentage}%"

        if efficiency < 1.0:
            prompt += f"\n- Equipment efficiency: {efficiency * 100}%"

        prompt += """

Steps:
1. Use the calculate_emissions tool to get exact emissions
2. Explain the calculation step-by-step"""

        if self.enable_recommendations:
            prompt += "\n3. Use the generate_recommendations tool to suggest improvements"

        prompt += """

IMPORTANT:
- Use tools for ALL numeric calculations
- Do not estimate or guess any numbers
- Explain your calculations clearly
- Round displayed values appropriately (e.g., "5,310 kg CO2e" not "5310.0")
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

            if name == "calculate_emissions":
                results["emissions"] = self._calculate_emissions_impl(**args)
            elif name == "lookup_emission_factor":
                results["emission_factor"] = self._lookup_emission_factor_impl(**args)
            elif name == "generate_recommendations":
                results["recommendations"] = self._generate_recommendations_impl(**args)

        return results

    def _build_output(
        self,
        payload: FuelInput,
        tool_results: Dict[str, Any],
        explanation: Optional[str],
    ) -> FuelOutput:
        """Build output from tool results.

        Args:
            payload: Original input
            tool_results: Results from tool calls
            explanation: AI-generated explanation

        Returns:
            FuelOutput with all data
        """
        emissions_data = tool_results.get("emissions", {})
        recommendations_data = tool_results.get("recommendations", {})

        output: FuelOutput = {
            "co2e_emissions_kg": emissions_data.get("emissions_kg_co2e", 0.0),
            "fuel_type": payload["fuel_type"],
            "consumption_amount": payload["amount"],
            "consumption_unit": payload["unit"],
            "emission_factor": emissions_data.get("emission_factor", 0.0),
            "emission_factor_unit": emissions_data.get("emission_factor_unit", ""),
            "country": payload.get("country", "US"),
            "scope": emissions_data.get("scope", "1"),
            "energy_content_mmbtu": emissions_data.get("energy_content_mmbtu", 0.0),
            "renewable_offset_applied": payload.get("renewable_percentage", 0) > 0,
            "efficiency_adjusted": payload.get("efficiency", 1.0) < 1.0,
        }

        # Add recommendations if available
        if recommendations_data:
            output["recommendations"] = recommendations_data.get("recommendations", [])

        # Add AI explanation if enabled
        if explanation and self.enable_explanations:
            output["explanation"] = explanation

        # Add citations for emission factors used
        if self._current_citations:
            output["citations"] = [c.to_dict() for c in self._current_citations]

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
                "avg_cost_per_calculation": (
                    self._total_cost_usd / max(self._ai_call_count, 1)
                ),
            },
            "base_agent_metrics": self.fuel_agent.get_performance_summary(),
        }
