"""GL-088: Vendor Evaluator Agent (VENDOR-EVALUATOR).

Evaluates energy vendors and suppliers.

Standards: ISO 9001, Supply Chain Best Practices
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VendorCategory(str, Enum):
    EQUIPMENT = "EQUIPMENT"
    SERVICE = "SERVICE"
    ENERGY_SUPPLY = "ENERGY_SUPPLY"
    CONSULTING = "CONSULTING"


class VendorCriteria(BaseModel):
    criterion: str
    weight_pct: float = Field(ge=0, le=100)
    vendor_score: int = Field(ge=1, le=5)


class VendorProfile(BaseModel):
    vendor_id: str
    vendor_name: str
    category: VendorCategory
    years_in_business: int = Field(ge=0)
    criteria_scores: List[VendorCriteria] = Field(default_factory=list)
    references_count: int = Field(ge=0)
    certifications: List[str] = Field(default_factory=list)
    financial_rating: str = Field(default="BBB")
    sustainability_score: int = Field(ge=0, le=100)


class VendorEvaluatorInput(BaseModel):
    evaluation_id: str
    vendors: List[VendorProfile] = Field(default_factory=list)
    min_experience_years: int = Field(default=3, ge=0)
    required_certifications: List[str] = Field(default_factory=list)
    min_financial_rating: str = Field(default="BB")
    sustainability_weight_pct: float = Field(default=20, ge=0, le=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VendorRanking(BaseModel):
    vendor_id: str
    vendor_name: str
    overall_score: float
    weighted_criteria_score: float
    sustainability_score: float
    financial_strength: str
    meets_requirements: bool
    rank: int
    strengths: List[str]
    weaknesses: List[str]


class VendorEvaluatorOutput(BaseModel):
    evaluation_id: str
    vendors_evaluated: int
    qualified_vendors: int
    rankings: List[VendorRanking]
    recommended_vendor_id: Optional[str]
    recommended_vendor_name: Optional[str]
    evaluation_summary: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class VendorEvaluatorAgent:
    AGENT_ID = "GL-088B"
    AGENT_NAME = "VENDOR-EVALUATOR"
    VERSION = "1.0.0"

    RATING_ORDER = ["D", "C", "CC", "CCC", "B", "BB", "BBB", "A", "AA", "AAA"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"VendorEvaluatorAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = VendorEvaluatorInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _rating_value(self, rating: str) -> int:
        try:
            return self.RATING_ORDER.index(rating.upper())
        except ValueError:
            return 5  # Default to BB

    def _process(self, inp: VendorEvaluatorInput) -> VendorEvaluatorOutput:
        recommendations = []
        rankings = []
        qualified = 0
        min_rating_val = self._rating_value(inp.min_financial_rating)

        for vendor in inp.vendors:
            # Calculate weighted criteria score
            if vendor.criteria_scores:
                total_weight = sum(c.weight_pct for c in vendor.criteria_scores)
                if total_weight > 0:
                    weighted_score = sum(c.weight_pct * c.vendor_score for c in vendor.criteria_scores) / total_weight * 20
                else:
                    weighted_score = 50
            else:
                weighted_score = 50

            # Sustainability contribution
            sustainability_contribution = vendor.sustainability_score * (inp.sustainability_weight_pct / 100)

            # Overall score
            overall = weighted_score * (1 - inp.sustainability_weight_pct / 100) + sustainability_contribution

            # Check requirements
            has_experience = vendor.years_in_business >= inp.min_experience_years
            has_certs = all(c in vendor.certifications for c in inp.required_certifications)
            has_rating = self._rating_value(vendor.financial_rating) >= min_rating_val
            meets_reqs = has_experience and has_certs and has_rating

            if meets_reqs:
                qualified += 1

            # Strengths and weaknesses
            strengths = []
            weaknesses = []

            if vendor.years_in_business > 10:
                strengths.append("Extensive experience")
            elif vendor.years_in_business < 3:
                weaknesses.append("Limited experience")

            if vendor.sustainability_score > 70:
                strengths.append("Strong sustainability")
            elif vendor.sustainability_score < 40:
                weaknesses.append("Low sustainability score")

            if vendor.references_count > 10:
                strengths.append("Strong reference base")
            elif vendor.references_count < 3:
                weaknesses.append("Few references")

            if len(vendor.certifications) > 3:
                strengths.append("Multiple certifications")

            rankings.append(VendorRanking(
                vendor_id=vendor.vendor_id,
                vendor_name=vendor.vendor_name,
                overall_score=round(overall, 1),
                weighted_criteria_score=round(weighted_score, 1),
                sustainability_score=vendor.sustainability_score,
                financial_strength=vendor.financial_rating,
                meets_requirements=meets_reqs,
                rank=0,
                strengths=strengths,
                weaknesses=weaknesses
            ))

        # Sort and rank
        rankings.sort(key=lambda x: (-x.meets_requirements, -x.overall_score))
        for i, r in enumerate(rankings):
            r.rank = i + 1

        # Recommendation
        qualified_vendors = [r for r in rankings if r.meets_requirements]
        if qualified_vendors:
            recommended = qualified_vendors[0]
            rec_id = recommended.vendor_id
            rec_name = recommended.vendor_name
            summary = f"Recommended: {rec_name} (Score: {recommended.overall_score:.1f})"
        else:
            rec_id = None
            rec_name = None
            summary = "No vendors meet all requirements"

        # Recommendations
        if qualified == 0:
            recommendations.append("No qualified vendors - consider relaxing requirements")
        elif qualified == 1:
            recommendations.append("Only one qualified vendor - limited competition")

        if qualified_vendors and len(qualified_vendors) > 1:
            score_diff = qualified_vendors[0].overall_score - qualified_vendors[1].overall_score
            if score_diff < 5:
                recommendations.append("Top vendors closely scored - request best-and-final offers")

        low_sustainability = [v for v in inp.vendors if v.sustainability_score < 40]
        if low_sustainability:
            recommendations.append(f"{len(low_sustainability)} vendors have low sustainability scores")

        calc_hash = hashlib.sha256(json.dumps({
            "evaluation": inp.evaluation_id,
            "vendors": len(inp.vendors),
            "qualified": qualified
        }).encode()).hexdigest()

        return VendorEvaluatorOutput(
            evaluation_id=inp.evaluation_id,
            vendors_evaluated=len(inp.vendors),
            qualified_vendors=qualified,
            rankings=rankings,
            recommended_vendor_id=rec_id,
            recommended_vendor_name=rec_name,
            evaluation_summary=summary,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-088B", "name": "VENDOR-EVALUATOR", "version": "1.0.0",
    "summary": "Vendor evaluation and ranking",
    "standards": [{"ref": "ISO 9001"}, {"ref": "Supply Chain Best Practices"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
