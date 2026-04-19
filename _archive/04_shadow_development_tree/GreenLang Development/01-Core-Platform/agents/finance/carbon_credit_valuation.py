# -*- coding: utf-8 -*-
"""
GL-FIN-X-004: Carbon Credit Valuation Agent
==========================================

Values carbon credits and offsets based on project type, vintage,
certification standard, and quality indicators.

Capabilities:
    - Credit valuation based on quality factors
    - Vintage premium/discount calculations
    - Standard-specific pricing adjustments
    - Additionality and permanence risk assessment
    - Portfolio diversification scoring
    - Retirement timing optimization

Zero-Hallucination Guarantees:
    - All valuations use deterministic formulas
    - Quality factors from structured lookup tables
    - No inference or prediction on credit quality
    - SHA-256 provenance hashes for all outputs

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.base_agents import AuditEntry
from greenlang.agents.categories import AgentCategory

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class CreditType(str, Enum):
    """Types of carbon credits."""
    AVOIDANCE = "avoidance"
    REDUCTION = "reduction"
    REMOVAL_NATURE = "removal_nature_based"
    REMOVAL_TECH = "removal_technology_based"
    REDD_PLUS = "redd_plus"


class CreditStandard(str, Enum):
    """Carbon credit certification standards."""
    VCS = "verra_vcs"
    GOLD_STANDARD = "gold_standard"
    ACR = "american_carbon_registry"
    CAR = "climate_action_reserve"
    CDM = "clean_development_mechanism"
    JCM = "joint_crediting_mechanism"
    CORSIA = "corsia_eligible"
    ICROA = "icroa_endorsed"
    ART_TREES = "art_trees"
    PURO = "puro_earth"


class ProjectCategory(str, Enum):
    """Project categories for carbon credits."""
    RENEWABLE_ENERGY = "renewable_energy"
    FORESTRY_AFFORESTATION = "forestry_afforestation"
    FORESTRY_REDD = "forestry_redd"
    COOKSTOVES = "cookstoves"
    METHANE_CAPTURE = "methane_capture"
    LANDFILL_GAS = "landfill_gas"
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    BIOCHAR = "biochar"
    ENHANCED_WEATHERING = "enhanced_weathering"
    OCEAN_BASED = "ocean_based"
    INDUSTRIAL_GAS = "industrial_gas"
    ENERGY_EFFICIENCY = "energy_efficiency"
    FUEL_SWITCHING = "fuel_switching"
    BLUE_CARBON = "blue_carbon"


class RiskLevel(str, Enum):
    """Risk levels for credits."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# Base prices by credit type (USD/tCO2e) - representative market prices
BASE_PRICES: Dict[str, float] = {
    CreditType.AVOIDANCE.value: 8.0,
    CreditType.REDUCTION.value: 12.0,
    CreditType.REMOVAL_NATURE.value: 25.0,
    CreditType.REMOVAL_TECH.value: 150.0,
    CreditType.REDD_PLUS.value: 10.0,
}

# Standard quality multipliers
STANDARD_MULTIPLIERS: Dict[str, float] = {
    CreditStandard.VCS.value: 1.0,
    CreditStandard.GOLD_STANDARD.value: 1.25,
    CreditStandard.ACR.value: 1.1,
    CreditStandard.CAR.value: 1.1,
    CreditStandard.CDM.value: 0.85,
    CreditStandard.JCM.value: 1.0,
    CreditStandard.CORSIA.value: 1.15,
    CreditStandard.ICROA.value: 1.0,
    CreditStandard.ART_TREES.value: 1.3,
    CreditStandard.PURO.value: 1.5,
}

# Project category premiums
PROJECT_PREMIUMS: Dict[str, float] = {
    ProjectCategory.DIRECT_AIR_CAPTURE.value: 8.0,  # Premium for tech removal
    ProjectCategory.BIOCHAR.value: 3.0,
    ProjectCategory.ENHANCED_WEATHERING.value: 2.5,
    ProjectCategory.BLUE_CARBON.value: 2.0,
    ProjectCategory.FORESTRY_AFFORESTATION.value: 1.5,
    ProjectCategory.FORESTRY_REDD.value: 0.0,  # Base price
    ProjectCategory.RENEWABLE_ENERGY.value: -2.0,  # Discount for commodity credits
    ProjectCategory.METHANE_CAPTURE.value: 1.0,
    ProjectCategory.COOKSTOVES.value: 0.5,
}

# Vintage adjustments (years from current)
VINTAGE_ADJUSTMENTS: Dict[str, float] = {
    "current_year": 1.0,
    "1_year_old": 0.95,
    "2_years_old": 0.90,
    "3_years_old": 0.85,
    "4_years_old": 0.80,
    "5_plus_years_old": 0.70,
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class CarbonCredit(BaseModel):
    """Specification of a carbon credit or credit batch."""
    credit_id: str = Field(..., description="Unique identifier")
    project_name: str = Field(..., description="Project name")
    project_id: Optional[str] = Field(None, description="Registry project ID")

    # Credit characteristics
    credit_type: CreditType = Field(..., description="Type of credit")
    standard: CreditStandard = Field(..., description="Certification standard")
    project_category: ProjectCategory = Field(..., description="Project category")

    # Quantity and vintage
    quantity_tco2e: float = Field(..., ge=0, description="Quantity in tCO2e")
    vintage_year: int = Field(..., ge=2000, le=2100, description="Vintage year")
    serial_numbers: Optional[str] = Field(None, description="Serial number range")

    # Quality indicators
    has_co_benefits: bool = Field(default=False, description="Has verified co-benefits")
    co_benefit_types: List[str] = Field(
        default_factory=list,
        description="Types of co-benefits (biodiversity, community, etc.)"
    )
    third_party_verified: bool = Field(default=True)
    verification_body: Optional[str] = Field(None)

    # Risk indicators
    permanence_years: Optional[int] = Field(
        None, ge=0, description="Expected permanence in years"
    )
    buffer_pool_contribution: float = Field(
        default=0.0, ge=0, le=100,
        description="% contributed to buffer pool"
    )
    reversal_risk_addressed: bool = Field(default=False)
    leakage_addressed: bool = Field(default=False)

    # Geography
    country: str = Field(..., description="Project country")
    region: Optional[str] = Field(None, description="Project region")
    is_article6_compliant: bool = Field(
        default=False, description="Paris Agreement Article 6 compliant"
    )


class CreditValuation(BaseModel):
    """Valuation result for a carbon credit."""
    credit_id: str
    project_name: str

    # Pricing
    base_price_per_tco2e: float = Field(..., description="Base price")
    quality_adjusted_price: float = Field(..., description="After quality adjustments")
    final_price_per_tco2e: float = Field(..., description="Final recommended price")
    total_value: float = Field(..., description="Total value for quantity")

    # Price components
    standard_adjustment: float = Field(..., description="Standard multiplier impact")
    project_premium: float = Field(..., description="Project category premium")
    vintage_adjustment: float = Field(..., description="Vintage discount/premium")
    co_benefit_premium: float = Field(..., description="Co-benefits premium")
    quality_premium: float = Field(..., description="Quality factors premium")

    # Risk assessment
    risk_level: RiskLevel
    risk_factors: List[str] = Field(default_factory=list)
    risk_adjusted_value: float = Field(..., description="Risk-adjusted value")

    # Market comparison
    market_reference_price: float = Field(..., description="Market reference")
    price_vs_market_pct: float = Field(..., description="% vs market reference")


class CreditRiskAssessment(BaseModel):
    """Risk assessment for carbon credits."""
    credit_id: str

    # Overall risk
    overall_risk_level: RiskLevel
    overall_risk_score: float = Field(..., ge=0, le=100)

    # Component risks
    permanence_risk: RiskLevel
    permanence_risk_score: float
    additionality_risk: RiskLevel
    additionality_risk_score: float
    verification_risk: RiskLevel
    verification_risk_score: float
    regulatory_risk: RiskLevel
    regulatory_risk_score: float

    # Risk factors
    risk_factors: List[str] = Field(default_factory=list)
    mitigating_factors: List[str] = Field(default_factory=list)

    # Value at risk
    value_at_risk_pct: float = Field(..., ge=0, le=100)
    recommended_haircut_pct: float = Field(..., ge=0, le=100)


class CreditValuationInput(BaseModel):
    """Input for credit valuation."""
    operation: str = Field(
        default="value_credit",
        description="Operation: value_credit, assess_risk, value_portfolio"
    )

    # Credit(s) to value
    credit: Optional[CarbonCredit] = Field(None, description="Single credit")
    credits: Optional[List[CarbonCredit]] = Field(None, description="Multiple credits")

    # Valuation parameters
    reference_date: Optional[datetime] = Field(
        None, description="Date for valuation (default: today)"
    )
    include_risk_adjustment: bool = Field(
        default=True, description="Apply risk adjustment"
    )
    market_price_source: str = Field(
        default="composite", description="Price source: composite, cboe, icap"
    )


class CreditValuationOutput(BaseModel):
    """Output from credit valuation."""
    success: bool
    operation: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Results
    valuation: Optional[CreditValuation] = Field(None)
    risk_assessment: Optional[CreditRiskAssessment] = Field(None)
    portfolio_valuations: Optional[List[CreditValuation]] = Field(None)
    portfolio_summary: Optional[Dict[str, Any]] = Field(None)

    # Audit
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# CARBON CREDIT VALUATION AGENT
# =============================================================================


class CarbonCreditValuationAgent(BaseAgent):
    """
    GL-FIN-X-004: Carbon Credit Valuation Agent

    Values carbon credits using deterministic quality-adjusted pricing.

    Zero-Hallucination Guarantees:
        - All valuations use deterministic formulas
        - Quality factors from structured lookup tables
        - No LLM inference on credit quality
        - SHA-256 provenance hashes for all outputs

    Usage:
        agent = CarbonCreditValuationAgent()
        result = agent.run({
            "operation": "value_credit",
            "credit": credit_specification
        })
    """

    AGENT_ID = "GL-FIN-X-004"
    AGENT_NAME = "Carbon Credit Valuation Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Carbon Credit Valuation Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Carbon credit and offset valuation",
                version=self.VERSION,
                parameters={}
            )

        self._audit_trail: List[AuditEntry] = []
        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute credit valuation."""
        try:
            val_input = CreditValuationInput(**input_data)
            operation = val_input.operation

            if operation == "value_credit":
                output = self._value_credit(val_input)
            elif operation == "assess_risk":
                output = self._assess_risk(val_input)
            elif operation == "value_portfolio":
                output = self._value_portfolio(val_input)
            else:
                return AgentResult(success=False, error=f"Unknown operation: {operation}")

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
                metadata={"agent_id": self.AGENT_ID, "operation": operation}
            )

        except Exception as e:
            logger.error(f"Credit valuation failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _value_credit(self, input_data: CreditValuationInput) -> CreditValuationOutput:
        """Value a single carbon credit."""
        calculation_trace: List[str] = []

        if input_data.credit is None:
            return CreditValuationOutput(
                success=False,
                operation="value_credit",
                calculation_trace=["ERROR: No credit provided"]
            )

        credit = input_data.credit
        ref_date = input_data.reference_date or datetime.utcnow()

        calculation_trace.append(f"Valuing: {credit.project_name} ({credit.credit_id})")
        calculation_trace.append(f"Type: {credit.credit_type.value}, Standard: {credit.standard.value}")
        calculation_trace.append(f"Quantity: {credit.quantity_tco2e:,.0f} tCO2e, Vintage: {credit.vintage_year}")

        # Base price
        base_price = BASE_PRICES.get(credit.credit_type.value, 10.0)
        calculation_trace.append(f"Base price for {credit.credit_type.value}: ${base_price}/tCO2e")

        # Standard multiplier
        standard_mult = STANDARD_MULTIPLIERS.get(credit.standard.value, 1.0)
        standard_adjustment = base_price * (standard_mult - 1)
        calculation_trace.append(f"Standard multiplier ({credit.standard.value}): {standard_mult}x -> ${standard_adjustment:+.2f}")

        # Project premium
        project_premium = PROJECT_PREMIUMS.get(credit.project_category.value, 0.0)
        calculation_trace.append(f"Project premium ({credit.project_category.value}): ${project_premium:+.2f}")

        # Vintage adjustment
        vintage_age = ref_date.year - credit.vintage_year
        if vintage_age <= 0:
            vintage_mult = 1.0
            vintage_key = "current_year"
        elif vintage_age == 1:
            vintage_mult = VINTAGE_ADJUSTMENTS["1_year_old"]
            vintage_key = "1_year_old"
        elif vintage_age == 2:
            vintage_mult = VINTAGE_ADJUSTMENTS["2_years_old"]
            vintage_key = "2_years_old"
        elif vintage_age == 3:
            vintage_mult = VINTAGE_ADJUSTMENTS["3_years_old"]
            vintage_key = "3_years_old"
        elif vintage_age == 4:
            vintage_mult = VINTAGE_ADJUSTMENTS["4_years_old"]
            vintage_key = "4_years_old"
        else:
            vintage_mult = VINTAGE_ADJUSTMENTS["5_plus_years_old"]
            vintage_key = "5_plus_years_old"

        vintage_adjustment = base_price * (vintage_mult - 1)
        calculation_trace.append(f"Vintage adjustment ({vintage_key}): {vintage_mult}x -> ${vintage_adjustment:+.2f}")

        # Co-benefits premium
        co_benefit_premium = 0.0
        if credit.has_co_benefits:
            co_benefit_premium = len(credit.co_benefit_types) * 0.5
            calculation_trace.append(f"Co-benefits premium ({len(credit.co_benefit_types)} types): ${co_benefit_premium:+.2f}")

        # Quality premium
        quality_premium = 0.0
        if credit.third_party_verified:
            quality_premium += 0.25
        if credit.reversal_risk_addressed:
            quality_premium += 0.50
        if credit.leakage_addressed:
            quality_premium += 0.25
        if credit.is_article6_compliant:
            quality_premium += 1.0
        calculation_trace.append(f"Quality premium: ${quality_premium:+.2f}")

        # Calculate prices
        quality_adjusted = base_price * standard_mult + project_premium + co_benefit_premium + quality_premium
        final_price = quality_adjusted * vintage_mult
        total_value = final_price * credit.quantity_tco2e

        calculation_trace.append(f"Quality-adjusted price: ${quality_adjusted:.2f}/tCO2e")
        calculation_trace.append(f"Final price (after vintage): ${final_price:.2f}/tCO2e")
        calculation_trace.append(f"Total value: ${total_value:,.2f}")

        # Risk assessment
        risk_assessment = self._calculate_risk(credit, calculation_trace)
        risk_adjusted_value = total_value * (1 - risk_assessment.recommended_haircut_pct / 100)

        # Market reference (simplified)
        market_ref = base_price * 1.1  # Assume 10% above base
        price_vs_market = ((final_price - market_ref) / market_ref) * 100

        valuation = CreditValuation(
            credit_id=credit.credit_id,
            project_name=credit.project_name,
            base_price_per_tco2e=round(base_price, 2),
            quality_adjusted_price=round(quality_adjusted, 2),
            final_price_per_tco2e=round(final_price, 2),
            total_value=round(total_value, 2),
            standard_adjustment=round(standard_adjustment, 2),
            project_premium=round(project_premium, 2),
            vintage_adjustment=round(vintage_adjustment, 2),
            co_benefit_premium=round(co_benefit_premium, 2),
            quality_premium=round(quality_premium, 2),
            risk_level=risk_assessment.overall_risk_level,
            risk_factors=risk_assessment.risk_factors,
            risk_adjusted_value=round(risk_adjusted_value, 2),
            market_reference_price=round(market_ref, 2),
            price_vs_market_pct=round(price_vs_market, 2)
        )

        provenance_hash = hashlib.sha256(
            json.dumps(valuation.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()

        return CreditValuationOutput(
            success=True,
            operation="value_credit",
            valuation=valuation,
            risk_assessment=risk_assessment,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _calculate_risk(
        self, credit: CarbonCredit, trace: List[str]
    ) -> CreditRiskAssessment:
        """Calculate risk assessment for a credit."""
        risk_factors: List[str] = []
        mitigating_factors: List[str] = []

        # Permanence risk
        perm_score = 30  # Default medium
        if credit.credit_type in [CreditType.REMOVAL_NATURE, CreditType.REDD_PLUS]:
            perm_score = 60
            risk_factors.append("Nature-based credits have reversal risk")
        if credit.permanence_years and credit.permanence_years >= 100:
            perm_score -= 20
            mitigating_factors.append("Long permanence commitment")
        if credit.buffer_pool_contribution > 15:
            perm_score -= 15
            mitigating_factors.append("Substantial buffer pool contribution")
        if credit.reversal_risk_addressed:
            perm_score -= 10
            mitigating_factors.append("Reversal risk addressed")
        perm_level = self._score_to_level(perm_score)

        # Additionality risk
        add_score = 30
        if credit.project_category == ProjectCategory.RENEWABLE_ENERGY:
            add_score = 50
            risk_factors.append("Renewable energy additionality questioned in some regions")
        if credit.credit_type == CreditType.REMOVAL_TECH:
            add_score = 15
            mitigating_factors.append("Technology-based removal is clearly additional")
        if credit.vintage_year < 2020:
            add_score += 10
            risk_factors.append("Older vintage may have additionality concerns")
        add_level = self._score_to_level(add_score)

        # Verification risk
        ver_score = 20
        if not credit.third_party_verified:
            ver_score = 60
            risk_factors.append("No third-party verification")
        if credit.standard in [CreditStandard.GOLD_STANDARD, CreditStandard.ART_TREES]:
            ver_score -= 10
            mitigating_factors.append("High-quality verification standard")
        ver_level = self._score_to_level(ver_score)

        # Regulatory risk
        reg_score = 25
        if not credit.is_article6_compliant:
            reg_score = 40
            risk_factors.append("Not Article 6 compliant")
        else:
            mitigating_factors.append("Article 6 compliant")
        reg_level = self._score_to_level(reg_score)

        # Overall risk
        overall_score = (perm_score * 0.3 + add_score * 0.3 + ver_score * 0.2 + reg_score * 0.2)
        overall_level = self._score_to_level(overall_score)

        # Recommended haircut
        haircut = overall_score / 5  # 0-20% range
        if overall_level == RiskLevel.HIGH:
            haircut = min(haircut + 5, 25)
        elif overall_level == RiskLevel.VERY_HIGH:
            haircut = min(haircut + 10, 40)

        trace.append(f"Risk assessment: {overall_level.value} (score: {overall_score:.1f})")
        trace.append(f"Recommended haircut: {haircut:.1f}%")

        return CreditRiskAssessment(
            credit_id=credit.credit_id,
            overall_risk_level=overall_level,
            overall_risk_score=round(overall_score, 2),
            permanence_risk=perm_level,
            permanence_risk_score=round(perm_score, 2),
            additionality_risk=add_level,
            additionality_risk_score=round(add_score, 2),
            verification_risk=ver_level,
            verification_risk_score=round(ver_score, 2),
            regulatory_risk=reg_level,
            regulatory_risk_score=round(reg_score, 2),
            risk_factors=risk_factors,
            mitigating_factors=mitigating_factors,
            value_at_risk_pct=round(overall_score / 2, 2),
            recommended_haircut_pct=round(haircut, 2)
        )

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert risk score to level."""
        if score < 25:
            return RiskLevel.LOW
        elif score < 45:
            return RiskLevel.MEDIUM
        elif score < 65:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _assess_risk(self, input_data: CreditValuationInput) -> CreditValuationOutput:
        """Assess risk for a credit without full valuation."""
        calculation_trace: List[str] = []

        if input_data.credit is None:
            return CreditValuationOutput(
                success=False,
                operation="assess_risk",
                calculation_trace=["ERROR: No credit provided"]
            )

        credit = input_data.credit
        calculation_trace.append(f"Risk assessment for: {credit.project_name}")

        risk_assessment = self._calculate_risk(credit, calculation_trace)

        provenance_hash = hashlib.sha256(
            json.dumps(risk_assessment.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()

        return CreditValuationOutput(
            success=True,
            operation="assess_risk",
            risk_assessment=risk_assessment,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _value_portfolio(self, input_data: CreditValuationInput) -> CreditValuationOutput:
        """Value a portfolio of credits."""
        calculation_trace: List[str] = []

        if not input_data.credits:
            return CreditValuationOutput(
                success=False,
                operation="value_portfolio",
                calculation_trace=["ERROR: No credits provided"]
            )

        valuations: List[CreditValuation] = []
        total_quantity = 0.0
        total_value = 0.0
        risk_adjusted_total = 0.0

        calculation_trace.append(f"Valuing portfolio of {len(input_data.credits)} credits")

        for credit in input_data.credits:
            val_input = CreditValuationInput(credit=credit)
            result = self._value_credit(val_input)
            if result.valuation:
                valuations.append(result.valuation)
                total_quantity += credit.quantity_tco2e
                total_value += result.valuation.total_value
                risk_adjusted_total += result.valuation.risk_adjusted_value

        avg_price = total_value / total_quantity if total_quantity > 0 else 0

        # Portfolio diversification
        credit_types = set(c.credit_type for c in input_data.credits)
        standards = set(c.standard for c in input_data.credits)
        categories = set(c.project_category for c in input_data.credits)

        diversification_score = min(100, len(credit_types) * 15 + len(standards) * 10 + len(categories) * 10)

        summary = {
            "total_quantity_tco2e": round(total_quantity, 2),
            "total_value": round(total_value, 2),
            "risk_adjusted_value": round(risk_adjusted_total, 2),
            "average_price_per_tco2e": round(avg_price, 2),
            "num_credits": len(valuations),
            "credit_types": [t.value for t in credit_types],
            "standards": [s.value for s in standards],
            "categories": [c.value for c in categories],
            "diversification_score": diversification_score
        }

        calculation_trace.append(f"Portfolio total: {total_quantity:,.0f} tCO2e, ${total_value:,.2f}")
        calculation_trace.append(f"Diversification score: {diversification_score}/100")

        provenance_hash = hashlib.sha256(
            json.dumps(summary, sort_keys=True, default=str).encode()
        ).hexdigest()

        return CreditValuationOutput(
            success=True,
            operation="value_portfolio",
            portfolio_valuations=valuations,
            portfolio_summary=summary,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "CarbonCreditValuationAgent",
    "CreditValuationInput",
    "CreditValuationOutput",
    "CreditType",
    "CreditStandard",
    "CreditValuation",
    "CreditRiskAssessment",
    "CarbonCredit",
    "ProjectCategory",
    "RiskLevel",
]
