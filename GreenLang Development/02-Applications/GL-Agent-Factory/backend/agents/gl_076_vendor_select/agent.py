"""GL-076: Vendor Selection Agent (VENDOR-SELECT).

Evaluates and recommends vendors for energy equipment.

Standards: ISO 9001, IEEE 1547
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EquipmentCategory(str, Enum):
    BOILER = "BOILER"
    CHILLER = "CHILLER"
    HEAT_EXCHANGER = "HEAT_EXCHANGER"
    PUMP = "PUMP"
    COMPRESSOR = "COMPRESSOR"
    CONTROLS = "CONTROLS"


class VendorProposal(BaseModel):
    vendor_id: str
    vendor_name: str
    price_usd: float = Field(ge=0)
    efficiency_rating: float = Field(ge=0, le=100)
    warranty_years: int = Field(ge=0)
    delivery_weeks: int = Field(ge=0)
    service_locations: int = Field(ge=0)
    references_count: int = Field(ge=0)
    certifications: List[str] = Field(default_factory=list)


class VendorSelectInput(BaseModel):
    project_id: str
    equipment_category: EquipmentCategory
    budget_usd: float = Field(..., gt=0)
    required_efficiency_pct: float = Field(default=90, ge=0, le=100)
    max_delivery_weeks: int = Field(default=12, ge=1)
    proposals: List[VendorProposal] = Field(default_factory=list)
    weight_price: float = Field(default=0.30, ge=0, le=1)
    weight_efficiency: float = Field(default=0.25, ge=0, le=1)
    weight_delivery: float = Field(default=0.15, ge=0, le=1)
    weight_service: float = Field(default=0.15, ge=0, le=1)
    weight_references: float = Field(default=0.15, ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VendorScore(BaseModel):
    vendor_id: str
    vendor_name: str
    total_score: float
    price_score: float
    efficiency_score: float
    delivery_score: float
    service_score: float
    reference_score: float
    meets_requirements: bool
    rank: int


class VendorSelectOutput(BaseModel):
    project_id: str
    equipment_category: str
    proposals_evaluated: int
    qualified_vendors: int
    vendor_scores: List[VendorScore]
    recommended_vendor_id: Optional[str]
    recommended_vendor_name: Optional[str]
    potential_savings_usd: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class VendorSelectAgent:
    AGENT_ID = "GL-076"
    AGENT_NAME = "VENDOR-SELECT"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"VendorSelectAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = VendorSelectInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _score_proposal(self, proposal: VendorProposal, inp: VendorSelectInput,
                       min_price: float, max_price: float) -> VendorScore:
        """Score a vendor proposal."""
        # Price score (lower is better)
        if max_price > min_price:
            price_score = 100 * (1 - (proposal.price_usd - min_price) / (max_price - min_price))
        else:
            price_score = 100

        # Efficiency score
        efficiency_score = min(100, proposal.efficiency_rating)

        # Delivery score (faster is better)
        if proposal.delivery_weeks <= inp.max_delivery_weeks:
            delivery_score = 100 * (1 - proposal.delivery_weeks / (inp.max_delivery_weeks * 2))
        else:
            delivery_score = max(0, 50 - (proposal.delivery_weeks - inp.max_delivery_weeks) * 10)

        # Service score
        service_score = min(100, proposal.service_locations * 20)

        # Reference score
        reference_score = min(100, proposal.references_count * 10)

        # Weighted total
        total = (
            price_score * inp.weight_price +
            efficiency_score * inp.weight_efficiency +
            delivery_score * inp.weight_delivery +
            service_score * inp.weight_service +
            reference_score * inp.weight_references
        )

        # Requirements check
        meets_reqs = (
            proposal.price_usd <= inp.budget_usd and
            proposal.efficiency_rating >= inp.required_efficiency_pct and
            proposal.delivery_weeks <= inp.max_delivery_weeks * 1.5
        )

        return VendorScore(
            vendor_id=proposal.vendor_id,
            vendor_name=proposal.vendor_name,
            total_score=round(total, 1),
            price_score=round(price_score, 1),
            efficiency_score=round(efficiency_score, 1),
            delivery_score=round(delivery_score, 1),
            service_score=round(service_score, 1),
            reference_score=round(reference_score, 1),
            meets_requirements=meets_reqs,
            rank=0
        )

    def _process(self, inp: VendorSelectInput) -> VendorSelectOutput:
        recommendations = []

        if not inp.proposals:
            return VendorSelectOutput(
                project_id=inp.project_id,
                equipment_category=inp.equipment_category.value,
                proposals_evaluated=0,
                qualified_vendors=0,
                vendor_scores=[],
                recommended_vendor_id=None,
                recommended_vendor_name=None,
                potential_savings_usd=0,
                recommendations=["No proposals submitted for evaluation"],
                calculation_hash=hashlib.sha256(b"no_proposals").hexdigest(),
                agent_version=self.VERSION
            )

        # Get price range
        prices = [p.price_usd for p in inp.proposals]
        min_price, max_price = min(prices), max(prices)

        # Score all proposals
        scores = []
        for proposal in inp.proposals:
            score = self._score_proposal(proposal, inp, min_price, max_price)
            scores.append(score)

        # Sort by score and assign ranks
        scores.sort(key=lambda x: (-x.total_score, x.vendor_id))
        for i, score in enumerate(scores):
            score.rank = i + 1

        # Count qualified vendors
        qualified = [s for s in scores if s.meets_requirements]

        # Recommendation
        if qualified:
            recommended = qualified[0]
            rec_id = recommended.vendor_id
            rec_name = recommended.vendor_name

            # Calculate savings vs budget
            rec_proposal = next(p for p in inp.proposals if p.vendor_id == rec_id)
            savings = inp.budget_usd - rec_proposal.price_usd
        else:
            rec_id = None
            rec_name = None
            savings = 0
            recommendations.append("No vendors meet all requirements - consider adjusting criteria")

        # Additional recommendations
        if len(inp.proposals) < 3:
            recommendations.append("Only {len(inp.proposals)} proposals - recommend minimum 3 for competitive evaluation")

        if qualified:
            top = qualified[0]
            if top.price_score < 60:
                recommendations.append("Recommended vendor is not lowest price - confirm value justification")
            if top.efficiency_score < 85:
                recommendations.append("Efficiency below optimal - calculate lifecycle cost impact")
            if top.delivery_score < 70:
                recommendations.append("Delivery timeline tight - confirm project schedule compatibility")

        price_spread = (max_price - min_price) / min_price * 100 if min_price > 0 else 0
        if price_spread > 50:
            recommendations.append(f"Large price spread ({price_spread:.0f}%) - verify scope alignment")

        calc_hash = hashlib.sha256(json.dumps({
            "project": inp.project_id,
            "proposals": len(inp.proposals),
            "recommended": rec_id
        }).encode()).hexdigest()

        return VendorSelectOutput(
            project_id=inp.project_id,
            equipment_category=inp.equipment_category.value,
            proposals_evaluated=len(inp.proposals),
            qualified_vendors=len(qualified),
            vendor_scores=scores,
            recommended_vendor_id=rec_id,
            recommended_vendor_name=rec_name,
            potential_savings_usd=round(max(0, savings), 2),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-076", "name": "VENDOR-SELECT", "version": "1.0.0",
    "summary": "Vendor evaluation and selection for energy equipment",
    "standards": [{"ref": "ISO 9001"}, {"ref": "IEEE 1547"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
