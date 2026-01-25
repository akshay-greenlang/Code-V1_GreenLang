"""
Supply Chain Risk Assessment Module.

This module provides comprehensive risk assessment capabilities for
supply chain due diligence, supporting:

- Environmental risk scoring (deforestation, water, biodiversity)
- Social risk scoring (human rights, labor, CSDDD)
- Governance risk scoring (corruption, transparency)
- Geographic risk factors (country-level risk)
- Concentration risk (supplier dependency)
- Aggregated multi-tier risk

Regulatory Frameworks Supported:
- EUDR (EU Deforestation Regulation)
- CSDDD (Corporate Sustainability Due Diligence Directive)
- German Supply Chain Due Diligence Act (LkSG)
- UK Modern Slavery Act

Example:
    >>> from greenlang.supply_chain.risk import SupplyChainRiskAssessor
    >>> assessor = SupplyChainRiskAssessor(supply_chain_graph)
    >>>
    >>> # Assess all suppliers
    >>> results = assessor.assess_all_suppliers()
    >>>
    >>> # Get high-risk suppliers
    >>> high_risk = assessor.get_suppliers_by_risk_level(RiskLevel.HIGH)
    >>>
    >>> # Generate risk report
    >>> report = assessor.generate_risk_report()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any, Set, Tuple
from collections import defaultdict

from greenlang.supply_chain.models.entity import (
    Supplier,
    Facility,
    CommodityType,
    SupplierTier,
)
from greenlang.supply_chain.graph.supply_chain_graph import SupplyChainGraph

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Classify risk level from score (0-100)."""
        if score < 25:
            return cls.LOW
        elif score < 50:
            return cls.MEDIUM
        elif score < 75:
            return cls.HIGH
        else:
            return cls.CRITICAL


class RiskCategory(Enum):
    """Risk assessment categories."""
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    GEOGRAPHIC = "geographic"
    CONCENTRATION = "concentration"
    OPERATIONAL = "operational"
    REPUTATIONAL = "reputational"
    REGULATORY = "regulatory"


class EnvironmentalRiskFactor(Enum):
    """Environmental risk factors."""
    DEFORESTATION = "deforestation"
    WATER_STRESS = "water_stress"
    BIODIVERSITY_LOSS = "biodiversity_loss"
    CARBON_INTENSITY = "carbon_intensity"
    POLLUTION = "pollution"
    WASTE = "waste"
    LAND_USE = "land_use"


class SocialRiskFactor(Enum):
    """Social risk factors (CSDDD)."""
    FORCED_LABOR = "forced_labor"
    CHILD_LABOR = "child_labor"
    FREEDOM_OF_ASSOCIATION = "freedom_of_association"
    HEALTH_SAFETY = "health_safety"
    LIVING_WAGE = "living_wage"
    WORKING_HOURS = "working_hours"
    DISCRIMINATION = "discrimination"
    LAND_RIGHTS = "land_rights"
    INDIGENOUS_RIGHTS = "indigenous_rights"


class GovernanceRiskFactor(Enum):
    """Governance risk factors."""
    CORRUPTION = "corruption"
    TRANSPARENCY = "transparency"
    RULE_OF_LAW = "rule_of_law"
    REGULATORY_QUALITY = "regulatory_quality"
    POLITICAL_STABILITY = "political_stability"


@dataclass
class CountryRiskData:
    """
    Country-level risk data.

    Contains risk indicators and scores for a specific country,
    sourced from international indices and databases.

    Attributes:
        country_code: ISO 3166-1 alpha-2 code
        country_name: Full country name
        environmental_risk: Environmental risk score (0-100)
        social_risk: Social/human rights risk score (0-100)
        governance_risk: Governance risk score (0-100)
        deforestation_risk: Deforestation-specific risk
        water_stress_index: Water stress index
        corruption_index: Corruption Perception Index
        human_development_index: HDI score
        eudr_classification: EUDR country classification
        sources: Data sources
        last_updated: Last data update
    """
    country_code: str
    country_name: str = ""
    environmental_risk: float = 50.0
    social_risk: float = 50.0
    governance_risk: float = 50.0
    deforestation_risk: float = 50.0
    water_stress_index: float = 50.0
    corruption_index: float = 50.0
    human_development_index: float = 0.7
    eudr_classification: str = "standard"  # low, standard, high
    sources: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def overall_risk(self) -> float:
        """Calculate overall country risk score."""
        return (
            self.environmental_risk * 0.35 +
            self.social_risk * 0.35 +
            self.governance_risk * 0.30
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "country_code": self.country_code,
            "country_name": self.country_name,
            "environmental_risk": self.environmental_risk,
            "social_risk": self.social_risk,
            "governance_risk": self.governance_risk,
            "deforestation_risk": self.deforestation_risk,
            "water_stress_index": self.water_stress_index,
            "corruption_index": self.corruption_index,
            "human_development_index": self.human_development_index,
            "eudr_classification": self.eudr_classification,
            "overall_risk": self.overall_risk,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class RiskScore:
    """
    Individual risk score with breakdown.

    Attributes:
        category: Risk category
        score: Risk score (0-100)
        level: Risk level classification
        factors: Contributing factors with scores
        confidence: Assessment confidence (0-1)
        data_quality: Data quality indicator
        notes: Assessment notes
    """
    category: RiskCategory
    score: float
    level: RiskLevel = RiskLevel.UNKNOWN
    factors: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    data_quality: str = "medium"
    notes: str = ""

    def __post_init__(self):
        """Calculate level from score."""
        if self.level == RiskLevel.UNKNOWN and self.score > 0:
            self.level = RiskLevel.from_score(self.score)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "score": self.score,
            "level": self.level.value,
            "factors": self.factors,
            "confidence": self.confidence,
            "data_quality": self.data_quality,
            "notes": self.notes,
        }


@dataclass
class RiskProfile:
    """
    Complete risk profile for a supplier.

    Attributes:
        supplier_id: Supplier identifier
        supplier_name: Supplier name
        overall_score: Aggregated risk score
        overall_level: Overall risk level
        category_scores: Scores by category
        tier: Supply chain tier
        country_code: Country
        commodities: EUDR commodities
        assessment_date: Assessment timestamp
        recommendations: Risk mitigation recommendations
        metadata: Additional attributes
    """
    supplier_id: str
    supplier_name: str
    overall_score: float = 50.0
    overall_level: RiskLevel = RiskLevel.MEDIUM
    category_scores: Dict[RiskCategory, RiskScore] = field(default_factory=dict)
    tier: int = 0
    country_code: Optional[str] = None
    commodities: List[str] = field(default_factory=list)
    assessment_date: datetime = field(default_factory=datetime.utcnow)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_category_score(self, category: RiskCategory) -> Optional[RiskScore]:
        """Get score for a specific category."""
        return self.category_scores.get(category)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "overall_score": self.overall_score,
            "overall_level": self.overall_level.value,
            "category_scores": {
                k.value: v.to_dict()
                for k, v in self.category_scores.items()
            },
            "tier": self.tier,
            "country_code": self.country_code,
            "commodities": self.commodities,
            "assessment_date": self.assessment_date.isoformat(),
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


class SupplyChainRiskAssessor:
    """
    Supply chain risk assessment engine.

    Provides comprehensive risk assessment for suppliers and
    supply chains, supporting EUDR, CSDDD, and other due diligence
    requirements.

    Example:
        >>> assessor = SupplyChainRiskAssessor(supply_chain_graph)
        >>>
        >>> # Assess single supplier
        >>> profile = assessor.assess_supplier("SUP001")
        >>> print(f"Risk: {profile.overall_level.value}")
        >>>
        >>> # Assess all suppliers
        >>> results = assessor.assess_all_suppliers()
        >>>
        >>> # Get aggregated risk by tier
        >>> tier_risk = assessor.calculate_tier_risk()
    """

    # Default country risk data (simplified)
    DEFAULT_COUNTRY_RISK: Dict[str, Dict[str, float]] = {
        # High risk countries (EUDR focus)
        "BR": {"env": 75, "soc": 55, "gov": 50, "deforestation": 85},
        "ID": {"env": 70, "soc": 50, "gov": 45, "deforestation": 80},
        "MY": {"env": 65, "soc": 45, "gov": 50, "deforestation": 70},
        "AR": {"env": 60, "soc": 40, "gov": 45, "deforestation": 65},
        "CO": {"env": 55, "soc": 50, "gov": 45, "deforestation": 60},
        "PE": {"env": 55, "soc": 50, "gov": 45, "deforestation": 60},
        "CD": {"env": 70, "soc": 75, "gov": 80, "deforestation": 75},
        "CI": {"env": 65, "soc": 55, "gov": 55, "deforestation": 70},
        "GH": {"env": 55, "soc": 45, "gov": 50, "deforestation": 55},
        # Medium risk countries
        "IN": {"env": 55, "soc": 50, "gov": 45, "deforestation": 40},
        "CN": {"env": 60, "soc": 55, "gov": 55, "deforestation": 35},
        "VN": {"env": 50, "soc": 50, "gov": 50, "deforestation": 45},
        "TH": {"env": 45, "soc": 40, "gov": 45, "deforestation": 35},
        "MX": {"env": 50, "soc": 45, "gov": 50, "deforestation": 40},
        "TR": {"env": 40, "soc": 45, "gov": 50, "deforestation": 25},
        # Lower risk countries
        "DE": {"env": 15, "soc": 10, "gov": 10, "deforestation": 5},
        "US": {"env": 25, "soc": 20, "gov": 15, "deforestation": 10},
        "GB": {"env": 15, "soc": 15, "gov": 10, "deforestation": 5},
        "FR": {"env": 15, "soc": 15, "gov": 15, "deforestation": 5},
        "NL": {"env": 12, "soc": 10, "gov": 8, "deforestation": 5},
        "JP": {"env": 20, "soc": 15, "gov": 12, "deforestation": 5},
        "CA": {"env": 20, "soc": 15, "gov": 10, "deforestation": 15},
        "AU": {"env": 25, "soc": 15, "gov": 10, "deforestation": 15},
    }

    # Commodity risk multipliers
    COMMODITY_RISK_MULTIPLIERS: Dict[str, float] = {
        "cattle": 1.3,
        "beef": 1.3,
        "leather": 1.2,
        "soya": 1.2,
        "soybean_oil": 1.1,
        "oil_palm": 1.3,
        "palm_oil": 1.3,
        "cocoa": 1.2,
        "chocolate": 1.1,
        "coffee": 1.1,
        "rubber": 1.1,
        "wood": 1.2,
        "timber": 1.2,
        "furniture": 1.0,
        "paper": 0.9,
    }

    # Category weights for overall score
    CATEGORY_WEIGHTS: Dict[RiskCategory, float] = {
        RiskCategory.ENVIRONMENTAL: 0.30,
        RiskCategory.SOCIAL: 0.25,
        RiskCategory.GOVERNANCE: 0.15,
        RiskCategory.GEOGRAPHIC: 0.15,
        RiskCategory.CONCENTRATION: 0.10,
        RiskCategory.OPERATIONAL: 0.05,
    }

    def __init__(
        self,
        supply_chain_graph: SupplyChainGraph,
        custom_country_risk: Optional[Dict[str, CountryRiskData]] = None,
    ):
        """
        Initialize the risk assessor.

        Args:
            supply_chain_graph: Supply chain graph instance
            custom_country_risk: Custom country risk data
        """
        self.graph = supply_chain_graph
        self._country_risk = custom_country_risk or {}
        self._risk_profiles: Dict[str, RiskProfile] = {}

        # Initialize default country risk
        self._init_default_country_risk()

        logger.info("SupplyChainRiskAssessor initialized")

    def _init_default_country_risk(self) -> None:
        """Initialize default country risk data."""
        for code, risk_data in self.DEFAULT_COUNTRY_RISK.items():
            if code not in self._country_risk:
                self._country_risk[code] = CountryRiskData(
                    country_code=code,
                    environmental_risk=risk_data.get("env", 50),
                    social_risk=risk_data.get("soc", 50),
                    governance_risk=risk_data.get("gov", 50),
                    deforestation_risk=risk_data.get("deforestation", 50),
                )

    def get_country_risk(self, country_code: str) -> CountryRiskData:
        """
        Get risk data for a country.

        Args:
            country_code: ISO country code

        Returns:
            CountryRiskData for the country
        """
        if country_code in self._country_risk:
            return self._country_risk[country_code]

        # Return default medium-risk profile
        return CountryRiskData(
            country_code=country_code,
            environmental_risk=50.0,
            social_risk=50.0,
            governance_risk=50.0,
        )

    def assess_supplier(
        self,
        supplier_id: str,
        include_upstream: bool = True,
        upstream_depth: int = 3,
    ) -> Optional[RiskProfile]:
        """
        Assess risk for a single supplier.

        Args:
            supplier_id: Supplier identifier
            include_upstream: Include upstream supplier risk
            upstream_depth: Depth for upstream assessment

        Returns:
            RiskProfile for the supplier
        """
        supplier = self.graph.get_supplier(supplier_id)
        if not supplier:
            return None

        # Calculate category scores
        category_scores: Dict[RiskCategory, RiskScore] = {}

        # Environmental risk
        env_score = self._assess_environmental_risk(supplier)
        category_scores[RiskCategory.ENVIRONMENTAL] = env_score

        # Social risk
        social_score = self._assess_social_risk(supplier)
        category_scores[RiskCategory.SOCIAL] = social_score

        # Governance risk
        gov_score = self._assess_governance_risk(supplier)
        category_scores[RiskCategory.GOVERNANCE] = gov_score

        # Geographic risk
        geo_score = self._assess_geographic_risk(supplier)
        category_scores[RiskCategory.GEOGRAPHIC] = geo_score

        # Concentration risk
        conc_score = self._assess_concentration_risk(supplier_id)
        category_scores[RiskCategory.CONCENTRATION] = conc_score

        # Include upstream risk if requested
        upstream_risk = 0.0
        if include_upstream:
            upstream_risk = self._assess_upstream_risk(
                supplier_id, upstream_depth
            )

        # Calculate overall score
        overall_score = sum(
            score.score * self.CATEGORY_WEIGHTS.get(cat, 0.1)
            for cat, score in category_scores.items()
        )

        # Adjust for upstream risk
        if upstream_risk > 0:
            overall_score = overall_score * 0.7 + upstream_risk * 0.3

        overall_level = RiskLevel.from_score(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            category_scores, overall_level
        )

        profile = RiskProfile(
            supplier_id=supplier_id,
            supplier_name=supplier.name,
            overall_score=overall_score,
            overall_level=overall_level,
            category_scores=category_scores,
            tier=supplier.tier.value,
            country_code=supplier.country_code,
            commodities=[c.value for c in supplier.commodities],
            recommendations=recommendations,
        )

        # Cache profile
        self._risk_profiles[supplier_id] = profile

        return profile

    def _assess_environmental_risk(self, supplier: Supplier) -> RiskScore:
        """Assess environmental risk for supplier."""
        factors: Dict[str, float] = {}

        # Base score from country risk
        base_score = 50.0
        if supplier.country_code:
            country_risk = self.get_country_risk(supplier.country_code)
            base_score = country_risk.environmental_risk
            factors["country_environmental"] = country_risk.environmental_risk
            factors["deforestation_risk"] = country_risk.deforestation_risk

        # Commodity risk adjustment
        commodity_multiplier = 1.0
        for commodity in supplier.commodities:
            mult = self.COMMODITY_RISK_MULTIPLIERS.get(commodity.value, 1.0)
            commodity_multiplier = max(commodity_multiplier, mult)
            factors[f"commodity_{commodity.value}"] = mult * 50

        # Certification reduction
        cert_reduction = 0.0
        if supplier.certifications:
            for cert in supplier.certifications:
                cert_lower = cert.lower()
                if "fsc" in cert_lower:
                    cert_reduction += 15
                elif "rspo" in cert_lower:
                    cert_reduction += 15
                elif "rainforest" in cert_lower:
                    cert_reduction += 10
                elif "organic" in cert_lower:
                    cert_reduction += 5
            factors["certification_reduction"] = cert_reduction

        # Calculate final score
        final_score = min(100, max(0, base_score * commodity_multiplier - cert_reduction))

        return RiskScore(
            category=RiskCategory.ENVIRONMENTAL,
            score=final_score,
            factors=factors,
            confidence=0.7,
            data_quality="medium",
        )

    def _assess_social_risk(self, supplier: Supplier) -> RiskScore:
        """Assess social/human rights risk for supplier."""
        factors: Dict[str, float] = {}

        # Base score from country risk
        base_score = 50.0
        if supplier.country_code:
            country_risk = self.get_country_risk(supplier.country_code)
            base_score = country_risk.social_risk
            factors["country_social"] = country_risk.social_risk

        # Industry risk adjustment
        industry_risk = 0.0
        if supplier.industry_codes:
            # High-risk industries for labor issues
            high_risk_industries = {"11", "31", "33", "23"}  # Agriculture, food, textiles, construction
            for code in supplier.industry_codes.values():
                if code[:2] in high_risk_industries:
                    industry_risk = 15
                    break
            factors["industry_risk"] = industry_risk

        # Certification/audit reduction
        cert_reduction = 0.0
        if supplier.certifications:
            for cert in supplier.certifications:
                cert_lower = cert.lower()
                if "sa8000" in cert_lower:
                    cert_reduction += 20
                elif "sedex" in cert_lower:
                    cert_reduction += 15
                elif "fair trade" in cert_lower or "fairtrade" in cert_lower:
                    cert_reduction += 15
                elif "bsci" in cert_lower:
                    cert_reduction += 10
            factors["certification_reduction"] = cert_reduction

        final_score = min(100, max(0, base_score + industry_risk - cert_reduction))

        return RiskScore(
            category=RiskCategory.SOCIAL,
            score=final_score,
            factors=factors,
            confidence=0.6,
            data_quality="medium",
        )

    def _assess_governance_risk(self, supplier: Supplier) -> RiskScore:
        """Assess governance risk for supplier."""
        factors: Dict[str, float] = {}

        # Base score from country risk
        base_score = 50.0
        if supplier.country_code:
            country_risk = self.get_country_risk(supplier.country_code)
            base_score = country_risk.governance_risk
            factors["country_governance"] = country_risk.governance_risk
            factors["corruption_index"] = country_risk.corruption_index

        # Verification status
        verification_bonus = 0.0
        if supplier.external_ids.lei:
            verification_bonus += 5
            factors["lei_verified"] = -5
        if supplier.external_ids.duns:
            verification_bonus += 3
            factors["duns_verified"] = -3

        final_score = max(0, base_score - verification_bonus)

        return RiskScore(
            category=RiskCategory.GOVERNANCE,
            score=final_score,
            factors=factors,
            confidence=0.7,
            data_quality="high",
        )

    def _assess_geographic_risk(self, supplier: Supplier) -> RiskScore:
        """Assess geographic risk for supplier."""
        factors: Dict[str, float] = {}

        base_score = 50.0
        if supplier.country_code:
            country_risk = self.get_country_risk(supplier.country_code)

            # EUDR classification
            if country_risk.eudr_classification == "high":
                base_score = 80
            elif country_risk.eudr_classification == "standard":
                base_score = 50
            else:
                base_score = 25

            factors["eudr_classification"] = base_score
            factors["overall_country_risk"] = country_risk.overall_risk

            # High-risk country flag
            if supplier.is_high_risk_country():
                factors["high_risk_country"] = 20
                base_score += 20

        return RiskScore(
            category=RiskCategory.GEOGRAPHIC,
            score=min(100, base_score),
            factors=factors,
            confidence=0.9,
            data_quality="high",
        )

    def _assess_concentration_risk(self, supplier_id: str) -> RiskScore:
        """Assess concentration/dependency risk."""
        factors: Dict[str, float] = {}

        supplier = self.graph.get_supplier(supplier_id)
        if not supplier or not supplier.annual_spend:
            return RiskScore(
                category=RiskCategory.CONCENTRATION,
                score=50.0,
                factors={"no_spend_data": 50},
                confidence=0.3,
            )

        # Calculate spend concentration
        total_spend = sum(
            s.annual_spend or Decimal("0")
            for s in self.graph._suppliers.values()
        )

        if total_spend > 0:
            spend_share = float(supplier.annual_spend / total_spend)
            factors["spend_share"] = spend_share * 100

            # High concentration if >20% of spend
            if spend_share > 0.30:
                conc_score = 90
            elif spend_share > 0.20:
                conc_score = 75
            elif spend_share > 0.10:
                conc_score = 50
            elif spend_share > 0.05:
                conc_score = 30
            else:
                conc_score = 15
        else:
            conc_score = 50

        # Check single-source risk
        supplier_commodities = set(c.value for c in supplier.commodities)
        for commodity in supplier.commodities:
            commodity_suppliers = self.graph.get_suppliers_by_commodity(commodity)
            if len(commodity_suppliers) == 1:
                factors[f"single_source_{commodity.value}"] = 30
                conc_score += 20

        return RiskScore(
            category=RiskCategory.CONCENTRATION,
            score=min(100, conc_score),
            factors=factors,
            confidence=0.8,
            data_quality="high",
        )

    def _assess_upstream_risk(
        self,
        supplier_id: str,
        max_depth: int,
    ) -> float:
        """
        Assess aggregated upstream supply chain risk.

        Risk propagates through tiers with decay.
        """
        upstream = self.graph.get_upstream_suppliers(supplier_id, max_depth)

        if not upstream:
            return 0.0

        # Calculate weighted upstream risk
        total_risk = 0.0
        total_weight = 0.0

        decay_factor = 0.7  # Risk decay per tier

        for tier_depth, suppliers in upstream.items():
            tier_weight = decay_factor ** tier_depth

            for sup in suppliers:
                # Get or calculate supplier risk
                if sup.id in self._risk_profiles:
                    risk = self._risk_profiles[sup.id].overall_score
                else:
                    # Quick risk estimate from country
                    if sup.country_code:
                        country_risk = self.get_country_risk(sup.country_code)
                        risk = country_risk.overall_risk
                    else:
                        risk = 50.0

                total_risk += risk * tier_weight
                total_weight += tier_weight

        return total_risk / total_weight if total_weight > 0 else 0.0

    def _generate_recommendations(
        self,
        category_scores: Dict[RiskCategory, RiskScore],
        overall_level: RiskLevel,
    ) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []

        # Environmental recommendations
        env_score = category_scores.get(RiskCategory.ENVIRONMENTAL)
        if env_score and env_score.score >= 60:
            recommendations.append(
                "Request sustainability certifications (FSC, RSPO, Rainforest Alliance)"
            )
            if env_score.factors.get("deforestation_risk", 0) >= 60:
                recommendations.append(
                    "Require plot-level traceability for EUDR compliance"
                )

        # Social recommendations
        social_score = category_scores.get(RiskCategory.SOCIAL)
        if social_score and social_score.score >= 60:
            recommendations.append(
                "Request social audit (SEDEX, SA8000) within 12 months"
            )
            recommendations.append(
                "Include human rights clauses in supplier contract"
            )

        # Governance recommendations
        gov_score = category_scores.get(RiskCategory.GOVERNANCE)
        if gov_score and gov_score.score >= 60:
            recommendations.append(
                "Verify legal entity through LEI or company registry"
            )
            recommendations.append(
                "Request anti-corruption policy and compliance documentation"
            )

        # Concentration recommendations
        conc_score = category_scores.get(RiskCategory.CONCENTRATION)
        if conc_score and conc_score.score >= 60:
            recommendations.append(
                "Develop alternative supplier sources to reduce dependency"
            )

        # Overall high risk
        if overall_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append(
                "Schedule on-site due diligence audit within 6 months"
            )
            recommendations.append(
                "Include supplier in enhanced monitoring program"
            )

        return recommendations

    def assess_all_suppliers(
        self,
        include_upstream: bool = False,
    ) -> Dict[str, RiskProfile]:
        """
        Assess risk for all suppliers in the graph.

        Args:
            include_upstream: Include upstream risk in assessment

        Returns:
            Dictionary mapping supplier ID to RiskProfile
        """
        results: Dict[str, RiskProfile] = {}

        for supplier_id in self.graph._suppliers.keys():
            profile = self.assess_supplier(
                supplier_id,
                include_upstream=include_upstream
            )
            if profile:
                results[supplier_id] = profile

        logger.info(f"Assessed risk for {len(results)} suppliers")
        return results

    def get_suppliers_by_risk_level(
        self,
        level: RiskLevel,
    ) -> List[RiskProfile]:
        """
        Get suppliers at a specific risk level.

        Args:
            level: Risk level to filter

        Returns:
            List of RiskProfiles at the specified level
        """
        return [
            profile for profile in self._risk_profiles.values()
            if profile.overall_level == level
        ]

    def calculate_tier_risk(self) -> Dict[int, Dict[str, Any]]:
        """
        Calculate aggregated risk by supply chain tier.

        Returns:
            Dictionary with tier-level risk statistics
        """
        tier_stats: Dict[int, Dict[str, Any]] = {}

        for tier in [1, 2, 3, 99]:
            suppliers = self.graph.get_suppliers_by_tier(
                SupplierTier(tier) if tier != 99 else SupplierTier.TIER_N
            )

            if not suppliers:
                continue

            risk_scores = []
            risk_levels: Dict[str, int] = defaultdict(int)

            for supplier in suppliers:
                profile = self._risk_profiles.get(supplier.id)
                if profile:
                    risk_scores.append(profile.overall_score)
                    risk_levels[profile.overall_level.value] += 1

            if risk_scores:
                tier_stats[tier] = {
                    "tier": tier,
                    "supplier_count": len(suppliers),
                    "avg_risk_score": sum(risk_scores) / len(risk_scores),
                    "max_risk_score": max(risk_scores),
                    "min_risk_score": min(risk_scores),
                    "risk_level_distribution": dict(risk_levels),
                    "high_risk_count": (
                        risk_levels.get("high", 0) +
                        risk_levels.get("critical", 0)
                    ),
                }

        return tier_stats

    def generate_risk_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.

        Returns:
            Complete risk assessment report
        """
        # Ensure all suppliers are assessed
        if len(self._risk_profiles) < len(self.graph._suppliers):
            self.assess_all_suppliers()

        # Calculate statistics
        all_scores = [p.overall_score for p in self._risk_profiles.values()]
        level_counts: Dict[str, int] = defaultdict(int)
        for profile in self._risk_profiles.values():
            level_counts[profile.overall_level.value] += 1

        # Category averages
        category_avgs: Dict[str, float] = {}
        for cat in RiskCategory:
            scores = []
            for profile in self._risk_profiles.values():
                if cat in profile.category_scores:
                    scores.append(profile.category_scores[cat].score)
            if scores:
                category_avgs[cat.value] = sum(scores) / len(scores)

        # Top risks
        sorted_profiles = sorted(
            self._risk_profiles.values(),
            key=lambda p: p.overall_score,
            reverse=True
        )
        top_risks = [p.to_dict() for p in sorted_profiles[:10]]

        # Country risk summary
        country_risks: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_risk": 0.0}
        )
        for profile in self._risk_profiles.values():
            if profile.country_code:
                country_risks[profile.country_code]["count"] += 1
                country_risks[profile.country_code]["total_risk"] += profile.overall_score

        for country in country_risks:
            count = country_risks[country]["count"]
            country_risks[country]["avg_risk"] = (
                country_risks[country]["total_risk"] / count if count else 0
            )

        return {
            "report_date": datetime.utcnow().isoformat(),
            "summary": {
                "total_suppliers_assessed": len(self._risk_profiles),
                "average_risk_score": sum(all_scores) / len(all_scores) if all_scores else 0,
                "max_risk_score": max(all_scores) if all_scores else 0,
                "min_risk_score": min(all_scores) if all_scores else 0,
                "risk_level_distribution": dict(level_counts),
                "high_risk_supplier_count": (
                    level_counts.get("high", 0) +
                    level_counts.get("critical", 0)
                ),
            },
            "category_averages": category_avgs,
            "tier_analysis": self.calculate_tier_risk(),
            "country_analysis": dict(country_risks),
            "top_risks": top_risks,
            "recommendations_summary": self._aggregate_recommendations(),
        }

    def _aggregate_recommendations(self) -> Dict[str, int]:
        """Aggregate recommendation frequency."""
        rec_counts: Dict[str, int] = defaultdict(int)
        for profile in self._risk_profiles.values():
            for rec in profile.recommendations:
                rec_counts[rec] += 1
        return dict(sorted(rec_counts.items(), key=lambda x: x[1], reverse=True))
