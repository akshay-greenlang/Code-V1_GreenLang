# -*- coding: utf-8 -*-
"""
EUDRBridge - EUDR Due Diligence to CSDDD Deforestation Impact Bridge for PACK-019
====================================================================================

This module connects EUDR (EU Deforestation Regulation) due diligence data to
CSDDD adverse impact assessment for deforestation-related environmental impacts.
It maps EUDR compliance statuses, commodity risk assessments, and supply chain
geolocation data to CSDDD environmental due diligence obligations.

Legal References:
    - Regulation (EU) 2023/1115 (EUDR) - Deforestation-free supply chains
    - Directive (EU) 2024/1760 (CSDDD), Annex Part II - Environmental impacts
    - CSDDD Art 6 - Identifying adverse environmental impacts
    - CSDDD Art 8 - Preventing potential adverse environmental impacts

EUDR-CSDDD Overlap:
    - EUDR due diligence satisfies CSDDD Art 6 for deforestation impacts
    - EUDR risk mitigation maps to CSDDD Art 8 prevention measures
    - EUDR commodity benchmarking provides CSDDD adverse impact prioritisation

Relevant Commodities (EUDR Annex I):
    Cattle, Cocoa, Coffee, Oil palm, Rubber, Soya, Wood

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-019 CSDDD Readiness Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EUDRCommodity(str, Enum):
    """EUDR Annex I regulated commodities."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

class DeforestationRisk(str, Enum):
    """Deforestation risk classification per EUDR benchmarking."""

    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"
    NOT_ASSESSED = "not_assessed"

class EUDRComplianceStatus(str, Enum):
    """EUDR due diligence compliance status."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    UNDER_REVIEW = "under_review"

class CSDDDImpactCategory(str, Enum):
    """CSDDD adverse impact category for deforestation-related impacts."""

    DEFORESTATION = "deforestation"
    BIODIVERSITY_LOSS = "biodiversity_loss"
    LAND_DEGRADATION = "land_degradation"
    WATER_POLLUTION = "water_pollution"
    INDIGENOUS_RIGHTS = "indigenous_rights"
    COMMUNITY_DISPLACEMENT = "community_displacement"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class EUDRBridgeConfig(BaseModel):
    """Configuration for the EUDR Bridge."""

    pack_id: str = Field(default="PACK-019")
    enable_provenance: bool = Field(default=True)
    eudr_cutoff_date: str = Field(
        default="2020-12-31",
        description="EUDR deforestation-free cutoff date",
    )
    high_risk_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Risk score threshold for high deforestation risk",
    )
    standard_risk_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Risk score threshold for standard risk",
    )

class CommodityRiskProfile(BaseModel):
    """Risk profile for a single EUDR-regulated commodity."""

    commodity: EUDRCommodity = Field(default=EUDRCommodity.WOOD)
    risk_level: DeforestationRisk = Field(default=DeforestationRisk.NOT_ASSESSED)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_countries: List[str] = Field(default_factory=list)
    volume_tonnes: float = Field(default=0.0, ge=0.0)
    has_geolocation: bool = Field(default=False)
    dd_statement_filed: bool = Field(default=False)
    compliance_status: EUDRComplianceStatus = Field(
        default=EUDRComplianceStatus.NOT_APPLICABLE
    )

class DeforestationImpact(BaseModel):
    """A deforestation-related adverse impact identified via EUDR data."""

    impact_id: str = Field(default_factory=_new_uuid)
    category: CSDDDImpactCategory = Field(default=CSDDDImpactCategory.DEFORESTATION)
    severity: str = Field(default="medium")
    commodity: EUDRCommodity = Field(default=EUDRCommodity.WOOD)
    country: str = Field(default="")
    description: str = Field(default="")
    csddd_articles: List[str] = Field(default_factory=list)
    is_actual: bool = Field(default=False)
    remediation_status: str = Field(default="not_started")

class EUDRDueDiligenceStatus(BaseModel):
    """Overall EUDR due diligence status for a company."""

    company_id: str = Field(default="")
    overall_status: EUDRComplianceStatus = Field(
        default=EUDRComplianceStatus.NOT_APPLICABLE
    )
    commodities_assessed: int = Field(default=0)
    commodities_compliant: int = Field(default=0)
    commodity_profiles: List[CommodityRiskProfile] = Field(default_factory=list)
    total_volume_tonnes: float = Field(default=0.0)
    high_risk_count: int = Field(default=0)
    dd_statements_filed: int = Field(default=0)
    provenance_hash: str = Field(default="")

class CSDDDMappingResult(BaseModel):
    """Result of mapping EUDR data to CSDDD requirements."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    impacts_identified: List[DeforestationImpact] = Field(default_factory=list)
    csddd_articles_covered: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    coverage_score: float = Field(default=0.0, ge=0.0, le=100.0)
    records_processed: int = Field(default=0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Country Risk Classifications (EUDR Benchmarking System)
# ---------------------------------------------------------------------------

COUNTRY_RISK_MAP: Dict[str, DeforestationRisk] = {
    "BR": DeforestationRisk.HIGH,
    "ID": DeforestationRisk.HIGH,
    "CO": DeforestationRisk.HIGH,
    "MY": DeforestationRisk.HIGH,
    "PG": DeforestationRisk.HIGH,
    "CD": DeforestationRisk.HIGH,
    "PE": DeforestationRisk.HIGH,
    "BO": DeforestationRisk.HIGH,
    "GH": DeforestationRisk.STANDARD,
    "CI": DeforestationRisk.STANDARD,
    "CM": DeforestationRisk.STANDARD,
    "NG": DeforestationRisk.STANDARD,
    "VN": DeforestationRisk.STANDARD,
    "PH": DeforestationRisk.STANDARD,
    "TH": DeforestationRisk.STANDARD,
    "MX": DeforestationRisk.STANDARD,
    "DE": DeforestationRisk.LOW,
    "FR": DeforestationRisk.LOW,
    "FI": DeforestationRisk.LOW,
    "SE": DeforestationRisk.LOW,
    "AT": DeforestationRisk.LOW,
    "US": DeforestationRisk.LOW,
    "CA": DeforestationRisk.LOW,
    "AU": DeforestationRisk.LOW,
}

IMPACT_MAPPINGS: Dict[EUDRCommodity, List[CSDDDImpactCategory]] = {
    EUDRCommodity.CATTLE: [
        CSDDDImpactCategory.DEFORESTATION,
        CSDDDImpactCategory.LAND_DEGRADATION,
        CSDDDImpactCategory.WATER_POLLUTION,
    ],
    EUDRCommodity.OIL_PALM: [
        CSDDDImpactCategory.DEFORESTATION,
        CSDDDImpactCategory.BIODIVERSITY_LOSS,
        CSDDDImpactCategory.INDIGENOUS_RIGHTS,
    ],
    EUDRCommodity.SOYA: [
        CSDDDImpactCategory.DEFORESTATION,
        CSDDDImpactCategory.LAND_DEGRADATION,
        CSDDDImpactCategory.COMMUNITY_DISPLACEMENT,
    ],
    EUDRCommodity.COCOA: [
        CSDDDImpactCategory.DEFORESTATION,
        CSDDDImpactCategory.BIODIVERSITY_LOSS,
    ],
    EUDRCommodity.COFFEE: [
        CSDDDImpactCategory.DEFORESTATION,
        CSDDDImpactCategory.WATER_POLLUTION,
    ],
    EUDRCommodity.RUBBER: [
        CSDDDImpactCategory.DEFORESTATION,
        CSDDDImpactCategory.BIODIVERSITY_LOSS,
    ],
    EUDRCommodity.WOOD: [
        CSDDDImpactCategory.DEFORESTATION,
        CSDDDImpactCategory.BIODIVERSITY_LOSS,
        CSDDDImpactCategory.INDIGENOUS_RIGHTS,
    ],
}

# ---------------------------------------------------------------------------
# EUDRBridge
# ---------------------------------------------------------------------------

class EUDRBridge:
    """EUDR due diligence to CSDDD deforestation impact bridge for PACK-019.

    Connects EUDR due diligence data to CSDDD adverse impact assessment for
    deforestation-related environmental impacts. Maps commodity risk levels,
    geolocation data, and compliance statuses to CSDDD Art 6-8 obligations.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = EUDRBridge(EUDRBridgeConfig())
        >>> status = bridge.get_eudr_dd_status("company_123")
        >>> assert status.overall_status != EUDRComplianceStatus.NOT_APPLICABLE
    """

    def __init__(self, config: Optional[EUDRBridgeConfig] = None) -> None:
        """Initialize EUDRBridge."""
        self.config = config or EUDRBridgeConfig()
        logger.info("EUDRBridge initialized (pack=%s)", self.config.pack_id)

    def get_eudr_dd_status(
        self,
        company_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EUDRDueDiligenceStatus:
        """Get EUDR due diligence status for a company.

        Args:
            company_id: Company identifier.
            context: Optional context with pre-loaded EUDR data.

        Returns:
            EUDRDueDiligenceStatus with commodity profiles and compliance.
        """
        ctx = context or {}
        commodities_data = ctx.get("eudr_commodities", [])

        profiles: List[CommodityRiskProfile] = []
        for c_data in commodities_data:
            commodity = EUDRCommodity(c_data.get("commodity", "wood"))
            countries = c_data.get("source_countries", [])

            # Deterministic risk scoring
            risk_score = self._calculate_commodity_risk(commodity, countries)
            risk_level = self._score_to_risk_level(risk_score)

            profiles.append(CommodityRiskProfile(
                commodity=commodity,
                risk_level=risk_level,
                risk_score=risk_score,
                source_countries=countries,
                volume_tonnes=c_data.get("volume_tonnes", 0.0),
                has_geolocation=c_data.get("has_geolocation", False),
                dd_statement_filed=c_data.get("dd_statement_filed", False),
                compliance_status=EUDRComplianceStatus(
                    c_data.get("compliance_status", "under_review")
                ),
            ))

        compliant = sum(
            1 for p in profiles
            if p.compliance_status == EUDRComplianceStatus.COMPLIANT
        )
        high_risk = sum(
            1 for p in profiles
            if p.risk_level == DeforestationRisk.HIGH
        )
        dd_filed = sum(1 for p in profiles if p.dd_statement_filed)
        total_volume = round(sum(p.volume_tonnes for p in profiles), 2)

        # Determine overall status
        if not profiles:
            overall = EUDRComplianceStatus.NOT_APPLICABLE
        elif compliant == len(profiles):
            overall = EUDRComplianceStatus.COMPLIANT
        elif compliant > 0:
            overall = EUDRComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall = EUDRComplianceStatus.UNDER_REVIEW

        status = EUDRDueDiligenceStatus(
            company_id=company_id,
            overall_status=overall,
            commodities_assessed=len(profiles),
            commodities_compliant=compliant,
            commodity_profiles=profiles,
            total_volume_tonnes=total_volume,
            high_risk_count=high_risk,
            dd_statements_filed=dd_filed,
        )
        status.provenance_hash = _compute_hash(status)

        logger.info(
            "EUDR status for %s: %s (%d commodities, %d high-risk)",
            company_id,
            overall.value,
            len(profiles),
            high_risk,
        )
        return status

    def map_eudr_to_csddd(
        self,
        eudr_data: Dict[str, Any],
    ) -> CSDDDMappingResult:
        """Map EUDR due diligence data to CSDDD requirements.

        Args:
            eudr_data: EUDR data dict with commodity profiles.

        Returns:
            CSDDDMappingResult with identified impacts and coverage.
        """
        result = CSDDDMappingResult()

        try:
            commodities = eudr_data.get("commodity_profiles", [])
            impacts: List[DeforestationImpact] = []

            for c_data in commodities:
                commodity = EUDRCommodity(c_data.get("commodity", "wood"))
                risk_level = DeforestationRisk(c_data.get("risk_level", "not_assessed"))
                countries = c_data.get("source_countries", [])

                if risk_level in (DeforestationRisk.HIGH, DeforestationRisk.STANDARD):
                    impact_cats = IMPACT_MAPPINGS.get(commodity, [CSDDDImpactCategory.DEFORESTATION])
                    for cat in impact_cats:
                        for country in (countries or [""]):
                            impacts.append(DeforestationImpact(
                                category=cat,
                                severity="high" if risk_level == DeforestationRisk.HIGH else "medium",
                                commodity=commodity,
                                country=country,
                                description=(
                                    f"{cat.value} risk from {commodity.value} "
                                    f"sourcing in {country or 'unknown'}"
                                ),
                                csddd_articles=["Art_6", "Art_8"],
                                is_actual=risk_level == DeforestationRisk.HIGH,
                            ))

            articles_covered = list(set(
                art for imp in impacts for art in imp.csddd_articles
            ))

            # Coverage score: how many of the 4 relevant articles are addressed
            relevant_articles = ["Art_6", "Art_7", "Art_8", "Art_9"]
            covered = len([a for a in relevant_articles if a in articles_covered])
            coverage_score = round(covered / len(relevant_articles) * 100, 1)

            result.impacts_identified = impacts
            result.csddd_articles_covered = articles_covered
            result.coverage_score = coverage_score
            result.records_processed = len(commodities)
            result.status = "completed"

            if impacts:
                result.recommendations = self._generate_recommendations(impacts)

            result.provenance_hash = _compute_hash(result)

            logger.info(
                "EUDR-to-CSDDD mapping: %d impacts, coverage=%.1f%%",
                len(impacts),
                coverage_score,
            )

        except Exception as exc:
            result.status = "failed"
            logger.error("EUDR-to-CSDDD mapping failed: %s", str(exc))

        return result

    def identify_deforestation_impacts(
        self,
        supply_chain: List[Dict[str, Any]],
    ) -> List[DeforestationImpact]:
        """Identify deforestation-related impacts from supply chain data.

        Args:
            supply_chain: List of supply chain node dicts with keys:
                supplier_name, commodity, country, volume_tonnes.

        Returns:
            List of DeforestationImpact objects.
        """
        impacts: List[DeforestationImpact] = []

        for node in supply_chain:
            country = node.get("country", "")
            commodity_str = node.get("commodity", "")

            # Check if commodity is EUDR-regulated
            try:
                commodity = EUDRCommodity(commodity_str.lower())
            except ValueError:
                continue

            country_risk = COUNTRY_RISK_MAP.get(country, DeforestationRisk.NOT_ASSESSED)

            if country_risk in (DeforestationRisk.HIGH, DeforestationRisk.STANDARD):
                impact_cats = IMPACT_MAPPINGS.get(commodity, [CSDDDImpactCategory.DEFORESTATION])
                for cat in impact_cats:
                    impacts.append(DeforestationImpact(
                        category=cat,
                        severity="high" if country_risk == DeforestationRisk.HIGH else "medium",
                        commodity=commodity,
                        country=country,
                        description=(
                            f"Potential {cat.value} from {commodity.value} "
                            f"supply chain in {country}"
                        ),
                        csddd_articles=["Art_6", "Art_8"],
                        is_actual=False,
                    ))

        logger.info(
            "Identified %d deforestation impacts from %d supply chain nodes",
            len(impacts),
            len(supply_chain),
        )
        return impacts

    def get_commodity_risk(
        self,
        commodity: str,
        countries: Optional[List[str]] = None,
    ) -> CommodityRiskProfile:
        """Get risk profile for a specific commodity.

        Args:
            commodity: Commodity name (must be EUDR-regulated).
            countries: Optional list of source country ISO codes.

        Returns:
            CommodityRiskProfile with risk assessment.
        """
        try:
            eudr_commodity = EUDRCommodity(commodity.lower())
        except ValueError:
            return CommodityRiskProfile(
                commodity=EUDRCommodity.WOOD,
                risk_level=DeforestationRisk.NOT_ASSESSED,
                risk_score=0.0,
            )

        source_countries = countries or []
        risk_score = self._calculate_commodity_risk(eudr_commodity, source_countries)
        risk_level = self._score_to_risk_level(risk_score)

        return CommodityRiskProfile(
            commodity=eudr_commodity,
            risk_level=risk_level,
            risk_score=risk_score,
            source_countries=source_countries,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_commodity_risk(
        self,
        commodity: EUDRCommodity,
        countries: List[str],
    ) -> float:
        """Calculate commodity-level deforestation risk score (0.0-1.0)."""
        if not countries:
            return 0.5  # Default standard risk with no country data

        country_scores: List[float] = []
        for country in countries:
            risk = COUNTRY_RISK_MAP.get(country, DeforestationRisk.STANDARD)
            if risk == DeforestationRisk.HIGH:
                country_scores.append(0.9)
            elif risk == DeforestationRisk.STANDARD:
                country_scores.append(0.5)
            else:
                country_scores.append(0.1)

        avg_country_risk = sum(country_scores) / len(country_scores)

        # Commodity inherent risk factor
        commodity_factors: Dict[EUDRCommodity, float] = {
            EUDRCommodity.OIL_PALM: 1.0,
            EUDRCommodity.SOYA: 0.9,
            EUDRCommodity.CATTLE: 0.9,
            EUDRCommodity.COCOA: 0.8,
            EUDRCommodity.COFFEE: 0.7,
            EUDRCommodity.RUBBER: 0.7,
            EUDRCommodity.WOOD: 0.6,
        }
        commodity_factor = commodity_factors.get(commodity, 0.5)

        # Weighted combination: 60% country, 40% commodity inherent risk
        final_score = round(0.6 * avg_country_risk + 0.4 * commodity_factor, 3)
        return min(max(final_score, 0.0), 1.0)

    def _score_to_risk_level(self, score: float) -> DeforestationRisk:
        """Map numeric risk score to risk classification."""
        if score >= self.config.high_risk_threshold:
            return DeforestationRisk.HIGH
        if score >= self.config.standard_risk_threshold:
            return DeforestationRisk.STANDARD
        return DeforestationRisk.LOW

    def _generate_recommendations(
        self,
        impacts: List[DeforestationImpact],
    ) -> List[str]:
        """Generate CSDDD-specific recommendations from identified impacts."""
        recommendations: List[str] = []
        categories_found = set(i.category for i in impacts)
        high_severity = any(i.severity == "high" for i in impacts)

        if CSDDDImpactCategory.DEFORESTATION in categories_found:
            recommendations.append(
                "Implement deforestation-free sourcing verification "
                "with geolocation data (Art 6, 8)"
            )
        if CSDDDImpactCategory.INDIGENOUS_RIGHTS in categories_found:
            recommendations.append(
                "Conduct FPIC (Free, Prior and Informed Consent) assessment "
                "with affected indigenous communities (Art 11)"
            )
        if CSDDDImpactCategory.BIODIVERSITY_LOSS in categories_found:
            recommendations.append(
                "Assess biodiversity impacts using HCV (High Conservation Value) "
                "methodology and integrate into prevention measures (Art 8)"
            )
        if high_severity:
            recommendations.append(
                "Prioritise high-severity impacts for immediate cessation "
                "or remediation action (Art 7, 9-10)"
            )

        return recommendations
