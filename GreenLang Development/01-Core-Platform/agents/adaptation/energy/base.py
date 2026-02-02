# -*- coding: utf-8 -*-
"""
Adaptation Energy Sector - Base Agent Class

This module provides the base class for all Adaptation Energy agents.
These agents follow the INSIGHT PATH pattern with deterministic risk
calculations enhanced by AI-powered scenario analysis.
"""

from abc import abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import json
import logging
import time

from greenlang.agents.base_agents import InsightAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata


logger = logging.getLogger(__name__)


class AdaptationEnergyBaseAgent(InsightAgent):
    """
    Base class for Adaptation Energy sector agents.

    These agents combine:
    - Deterministic risk calculations
    - AI-powered scenario narrative generation
    - Climate model integration
    """

    category = AgentCategory.INSIGHT
    metadata: Optional[AgentMetadata] = None

    # Climate hazard baseline frequencies (events per year)
    HAZARD_FREQUENCIES = {
        "extreme_heat": 0.15,
        "extreme_cold": 0.10,
        "flooding": 0.08,
        "drought": 0.05,
        "wildfire": 0.03,
        "hurricane": 0.02,
        "sea_level_rise": 1.0,  # Continuous
        "ice_storm": 0.04,
    }

    # Climate multipliers by scenario (2050)
    SCENARIO_MULTIPLIERS = {
        "ssp1_26": 1.2,
        "ssp2_45": 1.5,
        "ssp3_70": 2.0,
        "ssp5_85": 2.5,
    }

    # Infrastructure vulnerability by type and hazard
    VULNERABILITY_MATRIX = {
        "power_plant": {"extreme_heat": 0.3, "flooding": 0.8, "wildfire": 0.5},
        "transmission_line": {"extreme_heat": 0.2, "wildfire": 0.9, "ice_storm": 0.8},
        "substation": {"flooding": 0.9, "wildfire": 0.7, "extreme_heat": 0.4},
        "solar_farm": {"extreme_heat": 0.2, "wildfire": 0.8, "flooding": 0.3},
        "wind_farm": {"hurricane": 0.9, "ice_storm": 0.7, "extreme_heat": 0.1},
        "hydropower": {"drought": 0.9, "flooding": 0.5},
    }

    def __init__(
        self,
        agent_id: str,
        version: str = "1.0.0",
        enable_audit_trail: bool = False,
    ):
        """Initialize Adaptation Energy base agent."""
        super().__init__(enable_audit_trail=enable_audit_trail)

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

    def calculate_risk_score(
        self,
        hazard: str,
        infrastructure_type: str,
        climate_scenario: str,
        exposure_value_million: float,
    ) -> Dict[str, float]:
        """
        Calculate climate risk score (deterministic).

        Args:
            hazard: Climate hazard type
            infrastructure_type: Infrastructure type
            climate_scenario: Climate scenario
            exposure_value_million: Asset value exposed

        Returns:
            Risk metrics dictionary
        """
        # Get baseline frequency
        base_freq = self.HAZARD_FREQUENCIES.get(hazard, 0.05)

        # Apply climate scenario multiplier
        scenario_mult = self.SCENARIO_MULTIPLIERS.get(climate_scenario, 1.5)
        adjusted_freq = base_freq * scenario_mult

        # Get vulnerability
        vuln_dict = self.VULNERABILITY_MATRIX.get(infrastructure_type, {})
        vulnerability = vuln_dict.get(hazard, 0.3)

        # Calculate expected annual loss
        expected_annual_loss = exposure_value_million * vulnerability * adjusted_freq

        # Risk score (0-100)
        risk_score = min(100, vulnerability * adjusted_freq * 1000)

        return {
            "hazard_frequency": round(adjusted_freq, 4),
            "vulnerability": round(vulnerability, 2),
            "expected_annual_loss_million": round(expected_annual_loss, 4),
            "risk_score": round(risk_score, 1),
        }

    def calculate_adaptation_benefit(
        self,
        measure_cost_million: float,
        risk_reduction_pct: float,
        baseline_annual_loss_million: float,
        lifetime_years: int = 30,
        discount_rate: float = 0.05,
    ) -> Dict[str, float]:
        """
        Calculate adaptation measure cost-benefit (deterministic).

        Args:
            measure_cost_million: Adaptation measure cost
            risk_reduction_pct: Expected risk reduction
            baseline_annual_loss_million: Current expected annual loss
            lifetime_years: Measure lifetime
            discount_rate: Discount rate

        Returns:
            Cost-benefit metrics
        """
        # Annual avoided loss
        annual_avoided_loss = baseline_annual_loss_million * risk_reduction_pct / 100

        # NPV of avoided losses
        npv_avoided_losses = 0
        for year in range(1, lifetime_years + 1):
            npv_avoided_losses += annual_avoided_loss / ((1 + discount_rate) ** year)

        # Benefit-cost ratio
        bcr = npv_avoided_losses / measure_cost_million if measure_cost_million > 0 else 0

        return {
            "annual_avoided_loss_million": round(annual_avoided_loss, 4),
            "npv_avoided_losses_million": round(npv_avoided_losses, 2),
            "benefit_cost_ratio": round(bcr, 2),
            "net_benefit_million": round(npv_avoided_losses - measure_cost_million, 2),
        }

    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deterministic risk calculations."""
        return self._calculate_risk_assessment(inputs)

    @abstractmethod
    def _calculate_risk_assessment(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hazard-specific risk assessment."""
        pass

    async def explain(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any],
        session,
        rag_engine,
        temperature: float = 0.6
    ) -> str:
        """Generate AI-powered explanation of adaptation recommendations."""
        prompt = f"""Based on this climate risk assessment, explain the key findings and recommendations:

Risk Assessment:
- Hazards: {calculation_result.get('hazards_assessed', [])}
- Risk Score: {calculation_result.get('risk_score', 0)}/100
- Expected Annual Loss: ${calculation_result.get('expected_annual_loss_million', 0)}M

Recommended Measures:
{calculation_result.get('recommended_measures', [])}

Provide a brief executive summary of:
1. The key climate risks identified
2. Why these adaptation measures are recommended
3. The expected benefits of implementation
"""

        response = await session.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )

        return response.text

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs through the agent."""
        start_time = time.time()

        try:
            outputs = self._calculate_risk_assessment(inputs)

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
