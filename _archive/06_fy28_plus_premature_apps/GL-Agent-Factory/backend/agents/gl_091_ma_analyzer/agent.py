"""GL-091: M&A Analyzer Agent (MA-ANALYZER).

Analyzes mergers and acquisitions for energy assets.

Standards: Due Diligence, GAAP/IFRS
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TransactionType(str, Enum):
    ACQUISITION = "ACQUISITION"
    MERGER = "MERGER"
    DIVESTITURE = "DIVESTITURE"
    JV = "JV"


class MAAnalyzerInput(BaseModel):
    transaction_id: str
    target_name: str = Field(default="Target Company")
    transaction_type: TransactionType = Field(default=TransactionType.ACQUISITION)
    enterprise_value_usd: float = Field(..., gt=0)
    ebitda_usd: float = Field(default=0)
    revenue_usd: float = Field(default=0)
    energy_capacity_mw: float = Field(default=0, ge=0)
    synergy_potential_usd: float = Field(default=0, ge=0)
    integration_cost_usd: float = Field(default=0, ge=0)
    discount_rate_pct: float = Field(default=10, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MAAnalyzerOutput(BaseModel):
    transaction_id: str
    ev_ebitda_multiple: Optional[float]
    ev_revenue_multiple: Optional[float]
    ev_per_mw_usd: Optional[float]
    net_synergy_value_usd: float
    adjusted_ev_usd: float
    implied_premium_pct: float
    deal_attractiveness: str
    key_risks: List[str]
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class MAAnalyzerAgent:
    AGENT_ID = "GL-091"
    AGENT_NAME = "MA-ANALYZER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"MAAnalyzerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = MAAnalyzerInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: MAAnalyzerInput) -> MAAnalyzerOutput:
        recommendations = []
        risks = []

        # Valuation multiples
        ev_ebitda = inp.enterprise_value_usd / inp.ebitda_usd if inp.ebitda_usd > 0 else None
        ev_revenue = inp.enterprise_value_usd / inp.revenue_usd if inp.revenue_usd > 0 else None
        ev_mw = inp.enterprise_value_usd / inp.energy_capacity_mw / 1e6 if inp.energy_capacity_mw > 0 else None

        # Net synergy value (NPV of synergies less integration costs)
        discount = inp.discount_rate_pct / 100
        synergy_npv = inp.synergy_potential_usd * 5 / (1 + discount)  # 5-year synergy capture
        net_synergy = synergy_npv - inp.integration_cost_usd

        # Adjusted EV
        adjusted_ev = inp.enterprise_value_usd - net_synergy

        # Implied premium (vs standalone value)
        standalone = inp.ebitda_usd * 8 if inp.ebitda_usd > 0 else inp.enterprise_value_usd * 0.8
        premium = ((inp.enterprise_value_usd - standalone) / standalone * 100) if standalone > 0 else 0

        # Attractiveness assessment
        if ev_ebitda and ev_ebitda < 8 and net_synergy > 0:
            attractiveness = "HIGHLY ATTRACTIVE"
        elif ev_ebitda and ev_ebitda < 12 and net_synergy > 0:
            attractiveness = "ATTRACTIVE"
        elif ev_ebitda and ev_ebitda < 15:
            attractiveness = "FAIR"
        else:
            attractiveness = "UNATTRACTIVE"

        # Key risks
        if ev_ebitda and ev_ebitda > 12:
            risks.append(f"High EV/EBITDA multiple ({ev_ebitda:.1f}x)")
        if premium > 30:
            risks.append(f"High acquisition premium ({premium:.0f}%)")
        if inp.integration_cost_usd > inp.synergy_potential_usd:
            risks.append("Integration costs exceed synergy potential")
        if inp.ebitda_usd <= 0:
            risks.append("Negative or zero EBITDA - profitability risk")

        # Recommendations
        if attractiveness in ["HIGHLY ATTRACTIVE", "ATTRACTIVE"]:
            recommendations.append("Proceed with detailed due diligence")
        if net_synergy < 0:
            recommendations.append(f"Negative net synergy ${net_synergy:,.0f} - renegotiate price")
        if ev_ebitda and ev_ebitda > 10:
            recommendations.append("Above-market multiple - ensure strategic rationale")
        if inp.integration_cost_usd > inp.enterprise_value_usd * 0.1:
            recommendations.append("High integration costs (>10% of EV) - detailed integration plan needed")

        calc_hash = hashlib.sha256(json.dumps({
            "transaction": inp.transaction_id,
            "ev": inp.enterprise_value_usd,
            "attractiveness": attractiveness
        }).encode()).hexdigest()

        return MAAnalyzerOutput(
            transaction_id=inp.transaction_id,
            ev_ebitda_multiple=round(ev_ebitda, 1) if ev_ebitda else None,
            ev_revenue_multiple=round(ev_revenue, 2) if ev_revenue else None,
            ev_per_mw_usd=round(ev_mw, 0) if ev_mw else None,
            net_synergy_value_usd=round(net_synergy, 2),
            adjusted_ev_usd=round(adjusted_ev, 2),
            implied_premium_pct=round(premium, 1),
            deal_attractiveness=attractiveness,
            key_risks=risks,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-091", "name": "MA-ANALYZER", "version": "1.0.0",
    "summary": "M&A analysis for energy assets",
    "standards": [{"ref": "Due Diligence"}, {"ref": "GAAP/IFRS"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
