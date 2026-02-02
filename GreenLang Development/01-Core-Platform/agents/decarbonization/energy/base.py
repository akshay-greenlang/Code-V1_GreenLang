# -*- coding: utf-8 -*-
"""
Decarbonization Energy Sector - Base Agent Class

This module provides the base class for all Decarbonization Energy agents.
These agents follow the RECOMMENDATION PATH pattern and use AI reasoning
for strategic planning while maintaining deterministic financial calculations.
"""

from abc import abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import json
import logging
import time

from pydantic import BaseModel

from greenlang.agents.base_agents import ReasoningAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata


logger = logging.getLogger(__name__)


class DecarbonizationEnergyBaseAgent(ReasoningAgent):
    """
    Base class for Decarbonization Energy sector agents.

    This class extends ReasoningAgent for strategic planning with:
    - AI-powered scenario analysis
    - Technology assessment capabilities
    - Financial modeling (deterministic)
    - Risk assessment

    Subclasses must implement:
    - reason(): AI-powered planning logic
    - _calculate_economics(): Deterministic financial calculations

    Example:
        class GridDecarbonizationPlannerAgent(DecarbonizationEnergyBaseAgent):
            async def reason(self, context, session, rag_engine, tools=None):
                # AI-powered analysis
                pass

            def _calculate_economics(self, inputs):
                # Deterministic NPV/LCOE calculations
                pass
    """

    category = AgentCategory.RECOMMENDATION
    metadata: Optional[AgentMetadata] = None

    # Standard discount rates for energy projects
    DISCOUNT_RATE_DEFAULT = 0.08  # 8%
    DISCOUNT_RATE_LOW_RISK = 0.05  # 5%
    DISCOUNT_RATE_HIGH_RISK = 0.12  # 12%

    # Technology costs ($/kW or $/kWh)
    TECHNOLOGY_COSTS = {
        "solar_pv": {"capital": 800, "om_fixed": 15, "lifetime": 30},
        "wind_onshore": {"capital": 1100, "om_fixed": 25, "lifetime": 25},
        "wind_offshore": {"capital": 2800, "om_fixed": 80, "lifetime": 25},
        "battery_4h": {"capital": 200, "om_fixed": 5, "lifetime": 15},
        "battery_8h": {"capital": 350, "om_fixed": 8, "lifetime": 15},
        "nuclear_smr": {"capital": 6000, "om_fixed": 100, "lifetime": 40},
        "ccus_retrofit": {"capital": 1500, "om_fixed": 50, "lifetime": 25},
        "green_hydrogen": {"capital": 1200, "om_fixed": 30, "lifetime": 20},
    }

    # Emission factors for displacement calculations (kg CO2e/MWh)
    DISPLACED_EMISSION_FACTORS = {
        "coal": 900,
        "natural_gas_ccgt": 400,
        "oil": 650,
        "grid_average_us": 420,
        "grid_average_eu": 280,
    }

    def __init__(
        self,
        agent_id: str,
        version: str = "1.0.0",
    ):
        """
        Initialize Decarbonization Energy base agent.

        Args:
            agent_id: Unique agent identifier
            version: Agent version string
        """
        super().__init__()

        self.agent_id = agent_id
        self.version = version
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")

        self.logger.info(f"Initialized {agent_id} v{version}")

    def calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_data = {
            "agent_id": self.agent_id,
            "version": self.version,
            "inputs": inputs,
            "outputs": outputs,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def calculate_lcoe(
        self,
        capital_cost_per_kw: float,
        fixed_om_per_kw_year: float,
        variable_om_per_mwh: float,
        capacity_factor: float,
        lifetime_years: int,
        discount_rate: float = 0.08,
        fuel_cost_per_mwh: float = 0.0,
    ) -> float:
        """
        Calculate Levelized Cost of Energy (LCOE).

        Deterministic calculation - no AI involvement.

        Args:
            capital_cost_per_kw: Capital cost ($/kW)
            fixed_om_per_kw_year: Fixed O&M ($/kW-year)
            variable_om_per_mwh: Variable O&M ($/MWh)
            capacity_factor: Capacity factor (0-1)
            lifetime_years: Project lifetime
            discount_rate: Discount rate
            fuel_cost_per_mwh: Fuel cost ($/MWh)

        Returns:
            LCOE in $/MWh
        """
        # Annual generation per kW
        hours_per_year = 8760
        annual_generation_mwh_per_kw = capacity_factor * hours_per_year / 1000

        if annual_generation_mwh_per_kw <= 0:
            return float('inf')

        # Capital recovery factor
        crf = (discount_rate * (1 + discount_rate) ** lifetime_years) / \
              ((1 + discount_rate) ** lifetime_years - 1)

        # LCOE components
        capital_component = (capital_cost_per_kw * crf) / annual_generation_mwh_per_kw
        fixed_om_component = fixed_om_per_kw_year / annual_generation_mwh_per_kw
        variable_component = variable_om_per_mwh + fuel_cost_per_mwh

        lcoe = capital_component + fixed_om_component + variable_component
        return round(lcoe, 2)

    def calculate_npv(
        self,
        capital_cost: float,
        annual_cash_flows: List[float],
        discount_rate: float = 0.08,
    ) -> float:
        """
        Calculate Net Present Value (NPV).

        Deterministic calculation - no AI involvement.

        Args:
            capital_cost: Initial capital cost
            annual_cash_flows: List of annual cash flows
            discount_rate: Discount rate

        Returns:
            NPV value
        """
        npv = -capital_cost
        for year, cash_flow in enumerate(annual_cash_flows, start=1):
            npv += cash_flow / ((1 + discount_rate) ** year)
        return round(npv, 2)

    def calculate_irr(
        self,
        capital_cost: float,
        annual_cash_flows: List[float],
        precision: float = 0.001,
    ) -> Optional[float]:
        """
        Calculate Internal Rate of Return (IRR).

        Deterministic calculation using Newton-Raphson approximation.

        Args:
            capital_cost: Initial capital cost
            annual_cash_flows: List of annual cash flows
            precision: Calculation precision

        Returns:
            IRR as decimal (e.g., 0.15 for 15%), or None if not calculable
        """
        # Initial guess
        rate = 0.1

        for _ in range(100):  # Max iterations
            npv = -capital_cost
            npv_derivative = 0

            for year, cf in enumerate(annual_cash_flows, start=1):
                factor = (1 + rate) ** year
                npv += cf / factor
                npv_derivative -= year * cf / (factor * (1 + rate))

            if abs(npv_derivative) < 1e-10:
                return None

            new_rate = rate - npv / npv_derivative

            if abs(new_rate - rate) < precision:
                return round(new_rate, 4)

            rate = new_rate

        return None  # Did not converge

    def calculate_payback_period(
        self,
        capital_cost: float,
        annual_savings: float,
    ) -> float:
        """
        Calculate simple payback period.

        Args:
            capital_cost: Initial capital cost
            annual_savings: Annual savings/revenue

        Returns:
            Payback period in years
        """
        if annual_savings <= 0:
            return float('inf')
        return round(capital_cost / annual_savings, 1)

    def calculate_abatement_cost(
        self,
        capital_cost: float,
        annual_operating_cost: float,
        annual_emissions_reduced_tonnes: float,
        lifetime_years: int,
        discount_rate: float = 0.08,
    ) -> float:
        """
        Calculate marginal abatement cost (MAC).

        Args:
            capital_cost: Total capital cost
            annual_operating_cost: Annual operating cost
            annual_emissions_reduced_tonnes: Annual emission reduction
            lifetime_years: Project lifetime
            discount_rate: Discount rate

        Returns:
            Abatement cost in $/tCO2e
        """
        if annual_emissions_reduced_tonnes <= 0:
            return float('inf')

        # NPV of costs
        total_npv_cost = capital_cost
        for year in range(1, lifetime_years + 1):
            total_npv_cost += annual_operating_cost / ((1 + discount_rate) ** year)

        # NPV of emissions reduced
        total_npv_emissions = 0
        for year in range(1, lifetime_years + 1):
            total_npv_emissions += annual_emissions_reduced_tonnes / ((1 + discount_rate) ** year)

        if total_npv_emissions <= 0:
            return float('inf')

        mac = total_npv_cost / total_npv_emissions
        return round(mac, 2)

    def calculate_avoided_emissions(
        self,
        clean_generation_mwh: float,
        displaced_source: str = "grid_average_us",
    ) -> float:
        """
        Calculate avoided emissions from clean generation.

        Args:
            clean_generation_mwh: Clean energy generation (MWh)
            displaced_source: Source being displaced

        Returns:
            Avoided emissions in tonnes CO2e
        """
        emission_factor = self.DISPLACED_EMISSION_FACTORS.get(
            displaced_source, 420
        )
        avoided_tonnes = clean_generation_mwh * emission_factor / 1000
        return round(avoided_tonnes, 2)

    def process(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process inputs through the agent (synchronous wrapper).

        For full AI reasoning, use the async `reason()` method.
        This method provides basic deterministic analysis.

        Args:
            inputs: Input data dictionary

        Returns:
            Output dictionary with analysis results
        """
        start_time = time.time()

        try:
            # Perform deterministic calculations
            outputs = self._calculate_economics(inputs)

            # Add metadata
            outputs["agent_id"] = self.agent_id
            outputs["calculation_timestamp"] = datetime.now(timezone.utc).isoformat()
            outputs["processing_time_ms"] = round(
                (time.time() - start_time) * 1000, 2
            )
            outputs["provenance_hash"] = self.calculate_provenance_hash(
                inputs, outputs
            )

            return outputs

        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise

    @abstractmethod
    def _calculate_economics(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform deterministic economic calculations.

        This method handles all numeric calculations and must be
        fully deterministic (no AI/LLM involvement).

        Args:
            inputs: Input data dictionary

        Returns:
            Economic analysis results
        """
        pass

    @abstractmethod
    async def reason(
        self,
        context: Dict[str, Any],
        session,
        rag_engine,
        tools: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute AI-powered reasoning for strategic planning.

        This method uses:
        - RAG for technology/case study retrieval
        - LLM for scenario analysis
        - Tools for data access

        Args:
            context: Planning context
            session: ChatSession for LLM
            rag_engine: RAG engine for knowledge retrieval
            tools: Available tools

        Returns:
            Strategic recommendations
        """
        pass
