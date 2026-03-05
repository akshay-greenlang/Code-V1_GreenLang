"""
Physical Risk Engine -- Asset-Level Physical Climate Risk Assessment

Implements physical climate risk assessment with asset-level scoring for
both acute (event-driven) and chronic (trend-driven) physical hazards.

Provides:
  - Asset registration with geographic coordinates and metadata
  - Per-hazard exposure scoring across RCP/SSP scenarios
  - Vulnerability scoring based on asset type, age, and elevation
  - Adaptive capacity assessment (insurance, building codes, redundancy)
  - Composite risk scoring: (exposure x vulnerability) / adaptive_capacity
  - Financial damage estimation and insurance impact
  - Portfolio-level risk aggregation with VaR calculation
  - Supply chain physical risk cascade analysis
  - Multi-scenario comparison (SSP1-2.6, SSP2-4.5, SSP5-8.5)
  - Climate adaptation cost-benefit analysis
  - GeoJSON risk map data generation
  - Hazard projections lookup by coordinates
  - Physical risk disclosure generation

Reference:
    - TCFD Technical Supplement: Physical Risk (October 2020)
    - IPCC AR6 WG2 (Physical Climate Risks)
    - NGFS Physical Risk Assessment Guide
    - UNEP FI TCFD Physical Risk Methodology

Example:
    >>> engine = PhysicalRiskEngine(config)
    >>> asset = await engine.register_asset("org-1", asset_data)
    >>> risk = await engine.assess_physical_risk("org-1", asset.id, "ssp2_45")
    >>> portfolio = await engine.aggregate_portfolio_risk("org-1")
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    AssetType,
    HAZARD_EXPOSURE_MATRICES,
    PhysicalHazard,
    RiskType,
    SectorType,
    TCFDAppConfig,
    TimeHorizon,
)
from .models import (
    AssetLocation,
    PhysicalRiskAssessment,
    RegisterAssetRequest,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hazard Classification
# ---------------------------------------------------------------------------

ACUTE_HAZARDS = frozenset({
    PhysicalHazard.CYCLONE,
    PhysicalHazard.FLOOD,
    PhysicalHazard.WILDFIRE,
    PhysicalHazard.HEATWAVE,
})

CHRONIC_HAZARDS = frozenset({
    PhysicalHazard.DROUGHT,
    PhysicalHazard.SEA_LEVEL_RISE,
    PhysicalHazard.TEMPERATURE_RISE,
    PhysicalHazard.WATER_STRESS,
    PhysicalHazard.PRECIPITATION_CHANGE,
    PhysicalHazard.ECOSYSTEM_DEGRADATION,
})

_VULNERABILITY_BY_ASSET_TYPE: Dict[str, int] = {
    "building": 3, "infrastructure": 4, "equipment": 2,
    "vehicle": 2, "land": 3, "financial_asset": 1,
    "plant": 4, "vehicle_fleet": 2, "inventory": 2,
    "intellectual_property": 1,
}

_DAMAGE_FACTORS_BY_HAZARD: Dict[PhysicalHazard, Decimal] = {
    PhysicalHazard.CYCLONE: Decimal("0.15"),
    PhysicalHazard.FLOOD: Decimal("0.20"),
    PhysicalHazard.WILDFIRE: Decimal("0.25"),
    PhysicalHazard.HEATWAVE: Decimal("0.05"),
    PhysicalHazard.DROUGHT: Decimal("0.08"),
    PhysicalHazard.SEA_LEVEL_RISE: Decimal("0.12"),
    PhysicalHazard.TEMPERATURE_RISE: Decimal("0.04"),
    PhysicalHazard.WATER_STRESS: Decimal("0.06"),
    PhysicalHazard.PRECIPITATION_CHANGE: Decimal("0.05"),
    PhysicalHazard.ECOSYSTEM_DEGRADATION: Decimal("0.03"),
}

_DOWNTIME_DAYS_BY_HAZARD: Dict[PhysicalHazard, int] = {
    PhysicalHazard.CYCLONE: 30,
    PhysicalHazard.FLOOD: 21,
    PhysicalHazard.WILDFIRE: 45,
    PhysicalHazard.HEATWAVE: 5,
    PhysicalHazard.DROUGHT: 10,
    PhysicalHazard.SEA_LEVEL_RISE: 60,
    PhysicalHazard.TEMPERATURE_RISE: 3,
    PhysicalHazard.WATER_STRESS: 7,
    PhysicalHazard.PRECIPITATION_CHANGE: 5,
    PhysicalHazard.ECOSYSTEM_DEGRADATION: 14,
}

_ADAPTATION_MEASURES: Dict[PhysicalHazard, List[Dict[str, Any]]] = {
    PhysicalHazard.FLOOD: [
        {"measure": "Flood barriers and drainage", "cost_pct": Decimal("0.03"), "risk_reduction_pct": Decimal("40")},
        {"measure": "Elevated critical equipment", "cost_pct": Decimal("0.01"), "risk_reduction_pct": Decimal("20")},
        {"measure": "Flood insurance increase", "cost_pct": Decimal("0.005"), "risk_reduction_pct": Decimal("30")},
    ],
    PhysicalHazard.WILDFIRE: [
        {"measure": "Defensible space and firebreaks", "cost_pct": Decimal("0.02"), "risk_reduction_pct": Decimal("35")},
        {"measure": "Fire-resistant materials", "cost_pct": Decimal("0.04"), "risk_reduction_pct": Decimal("25")},
    ],
    PhysicalHazard.CYCLONE: [
        {"measure": "Structural reinforcement", "cost_pct": Decimal("0.05"), "risk_reduction_pct": Decimal("30")},
        {"measure": "Window protection systems", "cost_pct": Decimal("0.01"), "risk_reduction_pct": Decimal("15")},
    ],
    PhysicalHazard.HEATWAVE: [
        {"measure": "Cooling system upgrade", "cost_pct": Decimal("0.02"), "risk_reduction_pct": Decimal("50")},
        {"measure": "Heat-reflective roofing", "cost_pct": Decimal("0.015"), "risk_reduction_pct": Decimal("20")},
    ],
    PhysicalHazard.SEA_LEVEL_RISE: [
        {"measure": "Seawall construction", "cost_pct": Decimal("0.08"), "risk_reduction_pct": Decimal("45")},
        {"measure": "Asset relocation planning", "cost_pct": Decimal("0.15"), "risk_reduction_pct": Decimal("80")},
    ],
}


class PhysicalRiskEngine:
    """
    Physical climate risk assessment engine with asset-level scoring.

    Registers assets, computes exposure/vulnerability/adaptive capacity,
    derives composite risk scores, estimates financial damages, and
    provides portfolio-level aggregation and adaptation planning.

    Attributes:
        config: Application configuration.
        _assets: In-memory asset store keyed by org_id.
        _assessments: In-memory assessment store keyed by org_id.
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        self.config = config or TCFDAppConfig()
        self._assets: Dict[str, List[AssetLocation]] = {}
        self._assessments: Dict[str, List[PhysicalRiskAssessment]] = {}
        logger.info("PhysicalRiskEngine initialized")

    # ------------------------------------------------------------------
    # Asset Management
    # ------------------------------------------------------------------

    async def register_asset(
        self, org_id: str, asset_data: RegisterAssetRequest,
    ) -> AssetLocation:
        """Register a physical asset for risk assessment."""
        asset = AssetLocation(
            tenant_id="default",
            org_id=org_id,
            asset_name=asset_data.asset_name,
            asset_type=asset_data.asset_type,
            latitude=asset_data.latitude,
            longitude=asset_data.longitude,
            country=asset_data.country,
            region=asset_data.region,
            elevation_m=asset_data.elevation_m,
            building_type=asset_data.building_type,
            replacement_value_usd=asset_data.replacement_value_usd,
            year_built=asset_data.year_built,
        )
        if org_id not in self._assets:
            self._assets[org_id] = []
        self._assets[org_id].append(asset)
        logger.info("Registered asset '%s' for org %s", asset.asset_name, org_id)
        return asset

    async def update_asset(
        self, asset_id: str, updates: Dict[str, Any],
    ) -> AssetLocation:
        """Update an existing asset."""
        for org_id, assets in self._assets.items():
            for i, asset in enumerate(assets):
                if asset.id == asset_id:
                    data = asset.model_dump()
                    data.update(updates)
                    data["updated_at"] = _now()
                    updated = AssetLocation(**data)
                    self._assets[org_id][i] = updated
                    return updated
        raise ValueError(f"Asset {asset_id} not found")

    async def list_assets(self, org_id: str) -> List[AssetLocation]:
        """List all assets for an organization."""
        return self._assets.get(org_id, [])

    # ------------------------------------------------------------------
    # Core Physical Risk Assessment
    # ------------------------------------------------------------------

    async def assess_physical_risk(
        self, org_id: str, asset_id: str, rcp_scenario: str = "ssp2_45",
    ) -> PhysicalRiskAssessment:
        """Assess physical risk for a single asset across all hazards."""
        start = datetime.utcnow()
        asset = self._find_asset(org_id, asset_id)
        if asset is None:
            raise ValueError(f"Asset {asset_id} not found for org {org_id}")

        worst_hazard = PhysicalHazard.FLOOD
        worst_score = Decimal("0")

        for hazard in PhysicalHazard:
            exposure = await self.calculate_exposure_score(asset, hazard, rcp_scenario)
            vuln = await self.calculate_vulnerability_score(asset)
            adaptive = await self.calculate_adaptive_capacity(asset)
            composite = self._compute_composite(exposure, vuln, adaptive)
            if composite > worst_score:
                worst_score = composite
                worst_hazard = hazard

        exposure = await self.calculate_exposure_score(asset, worst_hazard, rcp_scenario)
        vuln = await self.calculate_vulnerability_score(asset)
        adaptive = await self.calculate_adaptive_capacity(asset)
        composite = self._compute_composite(exposure, vuln, adaptive)
        damage = await self.estimate_financial_damage(asset, composite, worst_hazard)
        insurance_impact = (
            asset.replacement_value_usd * Decimal("0.002")
            * (composite / Decimal("20"))
        ).quantize(Decimal("0.01"))

        assessment = PhysicalRiskAssessment(
            tenant_id="default",
            org_id=org_id,
            asset_id=asset_id,
            hazard_type=worst_hazard,
            rcp_scenario=rcp_scenario,
            exposure_score=exposure,
            vulnerability_score=vuln,
            adaptive_capacity_score=adaptive,
            composite_risk_score=composite,
            financial_damage_estimate_usd=damage,
        )

        if org_id not in self._assessments:
            self._assessments[org_id] = []
        self._assessments[org_id].append(assessment)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Physical risk for asset %s: hazard=%s, composite=%.1f, damage=$%.0f in %.1f ms",
            asset_id, worst_hazard.value, composite, damage, elapsed_ms,
        )
        return assessment

    async def assess_per_hazard(
        self, org_id: str, asset_id: str, rcp_scenario: str = "ssp2_45",
    ) -> List[Dict[str, Any]]:
        """
        Assess physical risk for each hazard individually.

        Returns a per-hazard breakdown rather than just the worst case.
        """
        asset = self._find_asset(org_id, asset_id)
        if asset is None:
            raise ValueError(f"Asset {asset_id} not found")

        results: List[Dict[str, Any]] = []
        vuln = await self.calculate_vulnerability_score(asset)
        adaptive = await self.calculate_adaptive_capacity(asset)

        for hazard in PhysicalHazard:
            exposure = await self.calculate_exposure_score(asset, hazard, rcp_scenario)
            composite = self._compute_composite(exposure, vuln, adaptive)
            damage = await self.estimate_financial_damage(asset, composite, hazard)
            downtime = _DOWNTIME_DAYS_BY_HAZARD.get(hazard, 7)
            is_acute = hazard in ACUTE_HAZARDS

            results.append({
                "hazard": hazard.value,
                "hazard_class": "acute" if is_acute else "chronic",
                "exposure_score": exposure,
                "vulnerability_score": vuln,
                "adaptive_capacity_score": adaptive,
                "composite_risk_score": str(composite),
                "financial_damage_usd": str(damage),
                "estimated_downtime_days": int(downtime * float(composite) / 50),
                "risk_rating": self._composite_to_rating(composite),
            })

        results.sort(key=lambda x: Decimal(x["composite_risk_score"]), reverse=True)
        return results

    async def batch_assess_portfolio(
        self, org_id: str, rcp_scenario: str = "ssp2_45",
    ) -> List[PhysicalRiskAssessment]:
        """Assess physical risk for all assets in an organization."""
        assets = self._assets.get(org_id, [])
        results: List[PhysicalRiskAssessment] = []
        for asset in assets:
            result = await self.assess_physical_risk(org_id, asset.id, rcp_scenario)
            results.append(result)
        logger.info("Batch assessed %d assets for org %s", len(results), org_id)
        return results

    # ------------------------------------------------------------------
    # Scoring Components
    # ------------------------------------------------------------------

    async def calculate_exposure_score(
        self, asset: AssetLocation, hazard_type: PhysicalHazard, rcp: str,
    ) -> int:
        """Calculate exposure score (1-5) based on hazard and RCP."""
        matrix = HAZARD_EXPOSURE_MATRICES.get(rcp, {})
        hazard_data = matrix.get(hazard_type, {})
        base_score = hazard_data.get("2050", 3)

        if hazard_type == PhysicalHazard.SEA_LEVEL_RISE:
            if float(asset.elevation_m) < 5:
                base_score = min(base_score + 2, 5)
            elif float(asset.elevation_m) < 20:
                base_score = min(base_score + 1, 5)

        if hazard_type == PhysicalHazard.FLOOD:
            if float(asset.elevation_m) < 10:
                base_score = min(base_score + 1, 5)

        if hazard_type == PhysicalHazard.WATER_STRESS:
            if asset.country in ("SA", "IN", "EG", "ZA", "AU"):
                base_score = min(base_score + 1, 5)

        return min(max(base_score, 1), 5)

    async def calculate_vulnerability_score(self, asset: AssetLocation) -> int:
        """Calculate vulnerability score based on asset type, age, and elevation."""
        type_score = _VULNERABILITY_BY_ASSET_TYPE.get(asset.asset_type.value, 3)
        age_modifier = 0
        if asset.year_built:
            age = 2025 - asset.year_built
            if age > 50:
                age_modifier = 2
            elif age > 25:
                age_modifier = 1
        return min(max(type_score + age_modifier, 1), 5)

    async def calculate_adaptive_capacity(self, asset: AssetLocation) -> int:
        """Calculate adaptive capacity score."""
        score = 3
        if asset.insurance_coverage_usd > 0:
            ratio = asset.insurance_coverage_usd / max(asset.replacement_value_usd, Decimal("1"))
            if ratio >= Decimal("0.8"):
                score += 1
            if ratio >= Decimal("1.0"):
                score += 1
        if asset.year_built and (2025 - asset.year_built) < 10:
            score += 1
        return min(max(score, 1), 5)

    async def estimate_financial_damage(
        self, asset: AssetLocation, risk_score: Decimal,
        hazard: Optional[PhysicalHazard] = None,
    ) -> Decimal:
        """Estimate financial damage based on risk score and hazard type."""
        base_factor = _DAMAGE_FACTORS_BY_HAZARD.get(hazard, Decimal("0.10")) if hazard else Decimal("0.10")
        score_factor = risk_score / Decimal("100")
        damage = (asset.replacement_value_usd * base_factor * score_factor).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        return damage

    # ------------------------------------------------------------------
    # Multi-Scenario Comparison
    # ------------------------------------------------------------------

    async def compare_scenarios(
        self, org_id: str, asset_id: str,
    ) -> Dict[str, Any]:
        """
        Compare physical risk across SSP1-2.6, SSP2-4.5, and SSP5-8.5.

        Args:
            org_id: Organization ID.
            asset_id: Asset ID.

        Returns:
            Dict with per-scenario risk scores and damage estimates.
        """
        asset = self._find_asset(org_id, asset_id)
        if asset is None:
            raise ValueError(f"Asset {asset_id} not found")

        scenarios = ["ssp1_26", "ssp2_45", "ssp5_85"]
        results: Dict[str, Dict[str, Any]] = {}

        for rcp in scenarios:
            worst_score = Decimal("0")
            worst_hazard = PhysicalHazard.FLOOD

            for hazard in PhysicalHazard:
                exp = await self.calculate_exposure_score(asset, hazard, rcp)
                vuln = await self.calculate_vulnerability_score(asset)
                adaptive = await self.calculate_adaptive_capacity(asset)
                composite = self._compute_composite(exp, vuln, adaptive)
                if composite > worst_score:
                    worst_score = composite
                    worst_hazard = hazard

            damage = await self.estimate_financial_damage(asset, worst_score, worst_hazard)

            results[rcp] = {
                "composite_risk_score": str(worst_score),
                "dominant_hazard": worst_hazard.value,
                "financial_damage_usd": str(damage),
                "risk_rating": self._composite_to_rating(worst_score),
            }

        return {
            "org_id": org_id,
            "asset_id": asset_id,
            "asset_name": asset.asset_name,
            "scenarios": results,
            "scenario_spread": str(
                Decimal(results["ssp5_85"]["composite_risk_score"])
                - Decimal(results["ssp1_26"]["composite_risk_score"])
            ),
        }

    # ------------------------------------------------------------------
    # Insurance and Financial Impact
    # ------------------------------------------------------------------

    async def estimate_insurance_impact(self, org_id: str) -> Dict[str, Any]:
        """Estimate total insurance cost impact for an organization."""
        assessments = self._assessments.get(org_id, [])
        assets = self._assets.get(org_id, [])

        total_replacement = sum(a.replacement_value_usd for a in assets)
        total_damage_exposure = sum(a.financial_damage_estimate_usd for a in assessments)

        current_premium_estimate = total_replacement * Decimal("0.005")
        risk_loading = Decimal("0")
        if total_replacement > 0:
            risk_loading = total_damage_exposure / total_replacement
        projected_premium = current_premium_estimate * (Decimal("1") + risk_loading)
        premium_increase = projected_premium - current_premium_estimate

        return {
            "org_id": org_id,
            "total_replacement_value": str(total_replacement),
            "total_damage_exposure": str(total_damage_exposure),
            "current_premium_estimate": str(current_premium_estimate.quantize(Decimal("0.01"))),
            "projected_premium": str(projected_premium.quantize(Decimal("0.01"))),
            "premium_increase": str(premium_increase.quantize(Decimal("0.01"))),
            "assessed_asset_count": len(assessments),
        }

    async def calculate_business_interruption(
        self, org_id: str, annual_revenue: Decimal,
    ) -> Dict[str, Any]:
        """
        Estimate business interruption costs from physical risks.

        Args:
            org_id: Organization ID.
            annual_revenue: Annual revenue for loss calculation.

        Returns:
            Dict with estimated downtime and revenue loss.
        """
        assessments = self._assessments.get(org_id, [])
        if not assessments:
            return {"org_id": org_id, "message": "No assessments available"}

        daily_revenue = annual_revenue / Decimal("365")
        total_downtime_days = Decimal("0")
        total_interruption_cost = Decimal("0")

        for asmt in assessments:
            base_downtime = Decimal(str(
                _DOWNTIME_DAYS_BY_HAZARD.get(asmt.hazard_type, 7)
            ))
            risk_factor = asmt.composite_risk_score / Decimal("100")
            expected_downtime = (base_downtime * risk_factor).quantize(Decimal("0.1"))
            cost = (daily_revenue * expected_downtime).quantize(Decimal("0.01"))
            total_downtime_days += expected_downtime
            total_interruption_cost += cost

        return {
            "org_id": org_id,
            "total_expected_downtime_days": str(total_downtime_days),
            "total_interruption_cost_usd": str(total_interruption_cost),
            "assets_assessed": len(assessments),
            "daily_revenue_usd": str(daily_revenue.quantize(Decimal("0.01"))),
        }

    # ------------------------------------------------------------------
    # Supply Chain Physical Risk
    # ------------------------------------------------------------------

    async def assess_supply_chain_physical_risk(self, org_id: str) -> Dict[str, Any]:
        """Assess physical risk across supply chain locations."""
        assets = self._assets.get(org_id, [])
        by_country: Dict[str, int] = {}
        for asset in assets:
            by_country[asset.country] = by_country.get(asset.country, 0) + 1

        assessments = self._assessments.get(org_id, [])
        high_risk = [a for a in assessments if a.composite_risk_score > Decimal("60")]
        critical_risk = [a for a in assessments if a.composite_risk_score > Decimal("80")]

        by_hazard: Dict[str, int] = {}
        for a in assessments:
            hz = a.hazard_type.value
            by_hazard[hz] = by_hazard.get(hz, 0) + 1

        concentration_risk = len(by_country) < 3 and len(assets) > 5

        return {
            "org_id": org_id,
            "total_locations": len(assets),
            "countries": by_country,
            "country_count": len(by_country),
            "high_risk_locations": len(high_risk),
            "critical_risk_locations": len(critical_risk),
            "dominant_hazards": by_hazard,
            "concentration_risk": concentration_risk,
            "diversification_recommendation": (
                "Diversify supply chain across more geographies to reduce concentration risk"
                if concentration_risk else "Geographic diversification is adequate"
            ),
        }

    # ------------------------------------------------------------------
    # Portfolio Risk Aggregation
    # ------------------------------------------------------------------

    async def aggregate_portfolio_risk(self, org_id: str) -> Dict[str, Any]:
        """Aggregate physical risk at the portfolio level."""
        assessments = self._assessments.get(org_id, [])
        if not assessments:
            return {"org_id": org_id, "message": "No assessments available"}

        scores = [a.composite_risk_score for a in assessments]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        total_damage = sum(a.financial_damage_estimate_usd for a in assessments)

        risk_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for s in scores:
            rating = self._composite_to_rating(s)
            risk_distribution[rating] = risk_distribution.get(rating, 0) + 1

        by_hazard: Dict[str, Decimal] = {}
        for a in assessments:
            hz = a.hazard_type.value
            by_hazard[hz] = by_hazard.get(hz, Decimal("0")) + a.financial_damage_estimate_usd

        return {
            "org_id": org_id,
            "total_assets_assessed": len(assessments),
            "average_risk_score": str(avg_score.quantize(Decimal("0.1"))),
            "max_risk_score": str(max_score),
            "total_damage_exposure": str(total_damage),
            "risk_distribution": risk_distribution,
            "damage_by_hazard": {k: str(v) for k, v in by_hazard.items()},
            "portfolio_risk_rating": self._composite_to_rating(avg_score),
        }

    async def calculate_portfolio_var(
        self, org_id: str, confidence: Decimal = Decimal("0.95"),
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk for the physical risk portfolio.

        Uses a simplified parametric approach based on damage estimates.

        Args:
            org_id: Organization ID.
            confidence: Confidence level (0.90, 0.95, 0.99).

        Returns:
            Dict with VaR estimate and expected shortfall.
        """
        assessments = self._assessments.get(org_id, [])
        if not assessments:
            return {"org_id": org_id, "message": "No assessments"}

        damages = sorted([float(a.financial_damage_estimate_usd) for a in assessments])
        n = len(damages)

        z_scores = {
            Decimal("0.90"): Decimal("1.28"),
            Decimal("0.95"): Decimal("1.65"),
            Decimal("0.99"): Decimal("2.33"),
        }
        z = float(z_scores.get(confidence, Decimal("1.65")))

        mean_damage = sum(damages) / n
        variance = sum((d - mean_damage) ** 2 for d in damages) / max(n - 1, 1)
        std_dev = variance ** 0.5

        var_value = mean_damage + z * std_dev
        tail_index = int(n * (1 - float(confidence)))
        expected_shortfall = sum(damages[n - max(tail_index, 1):]) / max(tail_index, 1)

        return {
            "org_id": org_id,
            "confidence_level": str(confidence),
            "var_usd": str(Decimal(str(round(var_value, 2)))),
            "expected_shortfall_usd": str(Decimal(str(round(expected_shortfall, 2)))),
            "mean_damage_usd": str(Decimal(str(round(mean_damage, 2)))),
            "std_dev_usd": str(Decimal(str(round(std_dev, 2)))),
            "assets_in_calculation": n,
        }

    # ------------------------------------------------------------------
    # Adaptation Planning
    # ------------------------------------------------------------------

    async def get_adaptation_measures(
        self, org_id: str, asset_id: str,
    ) -> Dict[str, Any]:
        """
        Get recommended adaptation measures for an asset.

        Args:
            org_id: Organization ID.
            asset_id: Asset ID.

        Returns:
            Dict with adaptation measures, costs, and expected risk reduction.
        """
        asset = self._find_asset(org_id, asset_id)
        if asset is None:
            raise ValueError(f"Asset {asset_id} not found")

        assessment = self._find_latest_assessment(org_id, asset_id)
        if assessment is None:
            return {"message": "No assessment available for this asset"}

        hazard = assessment.hazard_type
        measures = _ADAPTATION_MEASURES.get(hazard, [])

        recommendations: List[Dict[str, Any]] = []
        total_cost = Decimal("0")
        total_reduction = Decimal("0")

        for measure in measures:
            cost = (asset.replacement_value_usd * measure["cost_pct"]).quantize(Decimal("0.01"))
            reduction = measure["risk_reduction_pct"]
            benefit = (assessment.financial_damage_estimate_usd * reduction / 100).quantize(Decimal("0.01"))
            bcr = Decimal("0")
            if cost > 0:
                bcr = (benefit / cost).quantize(Decimal("0.01"))

            total_cost += cost
            total_reduction += reduction

            recommendations.append({
                "measure": measure["measure"],
                "cost_usd": str(cost),
                "risk_reduction_pct": str(reduction),
                "benefit_usd": str(benefit),
                "benefit_cost_ratio": str(bcr),
                "recommendation": "Implement" if bcr >= Decimal("1.5") else "Evaluate",
            })

        recommendations.sort(key=lambda x: Decimal(x["benefit_cost_ratio"]), reverse=True)

        return {
            "org_id": org_id,
            "asset_id": asset_id,
            "asset_name": asset.asset_name,
            "dominant_hazard": hazard.value,
            "current_risk_score": str(assessment.composite_risk_score),
            "current_damage_estimate": str(assessment.financial_damage_estimate_usd),
            "measures": recommendations,
            "total_adaptation_cost": str(total_cost),
            "combined_risk_reduction_pct": str(min(total_reduction, Decimal("80"))),
        }

    # ------------------------------------------------------------------
    # GeoJSON and Visualization
    # ------------------------------------------------------------------

    async def generate_risk_map_data(self, org_id: str) -> Dict[str, Any]:
        """Generate GeoJSON-compatible data for risk map visualization."""
        assets = self._assets.get(org_id, [])
        assessments = self._assessments.get(org_id, [])

        assessment_map = {a.asset_id: a for a in assessments}
        features: List[Dict[str, Any]] = []

        for asset in assets:
            assessment = assessment_map.get(asset.id)
            risk_score = float(assessment.composite_risk_score) if assessment else 0
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(asset.longitude), float(asset.latitude)],
                },
                "properties": {
                    "asset_id": asset.id,
                    "asset_name": asset.asset_name,
                    "asset_type": asset.asset_type.value,
                    "country": asset.country,
                    "risk_score": risk_score,
                    "risk_rating": self._composite_to_rating(
                        Decimal(str(risk_score))
                    ) if assessment else "not_assessed",
                    "replacement_value": float(asset.replacement_value_usd),
                    "damage_estimate": float(assessment.financial_damage_estimate_usd) if assessment else 0,
                    "dominant_hazard": assessment.hazard_type.value if assessment else None,
                },
            })

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    async def get_hazard_projections(
        self, lat: Decimal, lon: Decimal, rcp: str, hazard_type: PhysicalHazard,
    ) -> Dict[str, Any]:
        """Look up hazard projections for a geographic location."""
        matrix = HAZARD_EXPOSURE_MATRICES.get(rcp, {})
        hazard_data = matrix.get(hazard_type, {})
        return {
            "latitude": str(lat),
            "longitude": str(lon),
            "rcp_scenario": rcp,
            "hazard_type": hazard_type.value,
            "hazard_class": "acute" if hazard_type in ACUTE_HAZARDS else "chronic",
            "baseline": hazard_data.get("baseline", 2),
            "projection_2030": hazard_data.get("2030", 3),
            "projection_2050": hazard_data.get("2050", 4),
            "trend": "increasing" if hazard_data.get("2050", 3) > hazard_data.get("baseline", 2) else "stable",
        }

    # ------------------------------------------------------------------
    # Disclosure Generation
    # ------------------------------------------------------------------

    async def generate_physical_risk_disclosure(
        self, org_id: str,
    ) -> Dict[str, Any]:
        """Generate physical risk disclosure content."""
        assets = self._assets.get(org_id, [])
        assessments = self._assessments.get(org_id, [])
        total_damage = sum(a.financial_damage_estimate_usd for a in assessments)

        high_risk = [a for a in assessments if a.composite_risk_score > Decimal("60")]
        acute = [a for a in assessments if a.hazard_type in ACUTE_HAZARDS]
        chronic = [a for a in assessments if a.hazard_type in CHRONIC_HAZARDS]

        by_country: Dict[str, int] = {}
        for asset in assets:
            by_country[asset.country] = by_country.get(asset.country, 0) + 1

        content = (
            f"The organization has assessed physical climate risk across "
            f"{len(assets)} asset(s) in {len(by_country)} country/ies. "
            f"Total financial exposure is estimated at ${total_damage:,.0f}. "
            f"Of {len(assessments)} assessment(s), {len(high_risk)} are rated "
            f"high or critical risk. Acute hazards affect {len(acute)} asset(s) "
            f"and chronic hazards affect {len(chronic)} asset(s)."
        )

        return {
            "org_id": org_id,
            "content": content,
            "total_assets": len(assets),
            "total_assessments": len(assessments),
            "total_damage_exposure": str(total_damage),
            "high_risk_count": len(high_risk),
            "acute_risk_count": len(acute),
            "chronic_risk_count": len(chronic),
            "countries": list(by_country.keys()),
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _find_asset(self, org_id: str, asset_id: str) -> Optional[AssetLocation]:
        for asset in self._assets.get(org_id, []):
            if asset.id == asset_id:
                return asset
        return None

    def _find_latest_assessment(
        self, org_id: str, asset_id: str,
    ) -> Optional[PhysicalRiskAssessment]:
        """Find the most recent assessment for an asset."""
        assessments = [
            a for a in self._assessments.get(org_id, [])
            if a.asset_id == asset_id
        ]
        if not assessments:
            return None
        return max(assessments, key=lambda a: a.created_at)

    @staticmethod
    def _compute_composite(
        exposure: int, vulnerability: int, adaptive_capacity: int,
    ) -> Decimal:
        """Compute composite risk = (exposure x vulnerability) / adaptive_capacity, normalized 0-100."""
        raw = Decimal(str(exposure * vulnerability)) / max(Decimal(str(adaptive_capacity)), Decimal("1"))
        normalized = (raw / Decimal("5")) * Decimal("100")
        return min(normalized.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP), Decimal("100"))

    @staticmethod
    def _composite_to_rating(score: Decimal) -> str:
        """Convert composite score (0-100) to risk rating."""
        if score < Decimal("25"):
            return "low"
        if score < Decimal("50"):
            return "medium"
        if score < Decimal("75"):
            return "high"
        return "critical"
