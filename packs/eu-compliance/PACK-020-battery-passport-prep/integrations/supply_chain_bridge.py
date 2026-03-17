# -*- coding: utf-8 -*-
"""
SupplyChainBridge - Battery Supply Chain Due Diligence Bridge for PACK-020
=============================================================================

Links supply chain mapping agents to EU Battery Regulation supply chain
due diligence obligations (Art 39-42). Maps mineral supply chains for
cobalt, lithium, nickel, natural graphite, and other critical raw materials,
assesses supplier-level risk using OECD Due Diligence Guidance for
Responsible Supply Chains of Minerals, and provides tier-level breakdowns
needed for the digital battery passport.

Legal References:
    - Regulation (EU) 2023/1542, Art 39-42 (Supply chain due diligence)
    - Art 39: Due diligence policies for responsible sourcing
    - Art 40: Supply chain due diligence management system
    - Art 41: Third-party verification
    - Art 42: Recognition of equivalent schemes
    - Annex X: List of raw materials and risk categories
    - OECD Due Diligence Guidance for Minerals from CAHRAs

Critical Minerals for Batteries:
    - Cobalt (highest risk - DRC artisanal mining)
    - Lithium (Chile/Argentina/Australia brine and hard-rock)
    - Nickel (Indonesia/Philippines laterite processing)
    - Natural graphite (China/Mozambique)
    - Manganese (South Africa/Gabon)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-020 Battery Passport Prep Pack
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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class SupplierTier(str, Enum):
    """Supplier tier classification in the battery supply chain."""

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4_MINE = "tier_4_mine"
    UNKNOWN = "unknown"


class CriticalMineral(str, Enum):
    """Critical raw materials for batteries (Battery Reg Annex X)."""

    COBALT = "cobalt"
    LITHIUM = "lithium"
    NICKEL = "nickel"
    NATURAL_GRAPHITE = "natural_graphite"
    MANGANESE = "manganese"
    COPPER = "copper"


class RiskLevel(str, Enum):
    """Supply chain risk levels per OECD Guidance."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"
    NOT_ASSESSED = "not_assessed"


class DDComplianceStatus(str, Enum):
    """Due diligence compliance status."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class SupplyChainBridgeConfig(BaseModel):
    """Configuration for the Supply Chain Bridge."""

    pack_id: str = Field(default="PACK-020")
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    cahra_list_version: str = Field(
        default="2025",
        description="Conflict-Affected and High-Risk Areas list version",
    )
    oecd_guidance_version: str = Field(default="3rd_edition")
    minerals_in_scope: List[CriticalMineral] = Field(
        default_factory=lambda: [
            CriticalMineral.COBALT,
            CriticalMineral.LITHIUM,
            CriticalMineral.NICKEL,
            CriticalMineral.NATURAL_GRAPHITE,
        ]
    )


class SupplierRecord(BaseModel):
    """Individual supplier record in the battery supply chain."""

    supplier_id: str = Field(default_factory=_new_uuid)
    supplier_name: str = Field(default="")
    tier: SupplierTier = Field(default=SupplierTier.TIER_1)
    country: str = Field(default="")
    minerals_supplied: List[CriticalMineral] = Field(default_factory=list)
    risk_level: RiskLevel = Field(default=RiskLevel.NOT_ASSESSED)
    dd_status: DDComplianceStatus = Field(default=DDComplianceStatus.NOT_ASSESSED)
    is_cahra_origin: bool = Field(default=False)
    audit_date: Optional[str] = Field(None)
    certification_scheme: Optional[str] = Field(None)


class SupplierDataResult(BaseModel):
    """Result of supplier data retrieval."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    suppliers_total: int = Field(default=0)
    suppliers_tier1: int = Field(default=0)
    suppliers_tier2_plus: int = Field(default=0)
    suppliers_high_risk: int = Field(default=0)
    suppliers_cahra: int = Field(default=0)
    suppliers: List[SupplierRecord] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class MineralSupplyChainResult(BaseModel):
    """Result of mineral supply chain mapping."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    minerals_mapped: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    total_suppliers_by_mineral: Dict[str, int] = Field(default_factory=dict)
    cahra_suppliers_by_mineral: Dict[str, int] = Field(default_factory=dict)
    risk_summary: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class SupplierRiskResult(BaseModel):
    """Result of supplier risk assessment."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    suppliers_assessed: int = Field(default=0)
    high_risk_count: int = Field(default=0)
    medium_risk_count: int = Field(default=0)
    low_risk_count: int = Field(default=0)
    cahra_count: int = Field(default=0)
    overall_risk: RiskLevel = Field(default=RiskLevel.NOT_ASSESSED)
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list)
    mitigation_recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class TierBreakdownResult(BaseModel):
    """Result of supply chain tier breakdown."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    tiers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    total_tiers_mapped: int = Field(default=0)
    deepest_tier: str = Field(default="")
    mine_of_origin_identified: bool = Field(default=False)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Country risk classification (OECD CAHRA-relevant)
# ---------------------------------------------------------------------------

CAHRA_COUNTRIES: List[str] = [
    "COD", "COG", "RWA", "BDI", "TZA", "UGA", "ZMB",
    "CAF", "SSD", "MMR", "AFG", "VEN", "ZWE",
]

MINERAL_COUNTRY_RISK: Dict[str, Dict[str, str]] = {
    "cobalt": {
        "COD": "high", "ZMB": "medium", "AUS": "low",
        "CAN": "low", "PHL": "medium", "CUB": "medium",
    },
    "lithium": {
        "AUS": "low", "CHL": "low", "ARG": "low",
        "CHN": "medium", "ZWE": "high", "BRA": "low",
    },
    "nickel": {
        "IDN": "medium", "PHL": "medium", "RUS": "high",
        "CAN": "low", "AUS": "low", "NCL": "low",
    },
    "natural_graphite": {
        "CHN": "medium", "MOZ": "medium", "BRA": "low",
        "TZA": "medium", "IND": "low", "MDG": "medium",
    },
}


# ---------------------------------------------------------------------------
# SupplyChainBridge
# ---------------------------------------------------------------------------


class SupplyChainBridge:
    """Battery supply chain due diligence bridge for PACK-020.

    Links supply chain mapping agents to EU Battery Regulation Art 39-42
    requirements. Maps mineral supply chains for critical battery raw
    materials, assesses supplier risk using OECD Guidance, and provides
    tier-level breakdowns for the digital battery passport.

    Attributes:
        config: Bridge configuration.
        _suppliers: Cached supplier records.

    Example:
        >>> bridge = SupplyChainBridge(SupplyChainBridgeConfig())
        >>> result = bridge.get_supplier_data(context)
        >>> assert result.status == "completed"
    """

    def __init__(
        self, config: Optional[SupplyChainBridgeConfig] = None
    ) -> None:
        """Initialize SupplyChainBridge."""
        self.config = config or SupplyChainBridgeConfig()
        self._suppliers: List[SupplierRecord] = []
        logger.info(
            "SupplyChainBridge initialized (minerals=%d, cahra_version=%s)",
            len(self.config.minerals_in_scope),
            self.config.cahra_list_version,
        )

    def get_supplier_data(
        self, context: Dict[str, Any]
    ) -> SupplierDataResult:
        """Retrieve and classify supplier data from supply chain agents.

        Args:
            context: Pipeline context with raw supplier data.

        Returns:
            SupplierDataResult with classified supplier records.
        """
        result = SupplierDataResult(started_at=_utcnow())

        try:
            raw_suppliers = context.get("supplier_records", [])
            parsed: List[SupplierRecord] = []

            for s in raw_suppliers:
                minerals = [
                    CriticalMineral(m)
                    for m in s.get("minerals", [])
                    if m in [cm.value for cm in CriticalMineral]
                ]
                record = SupplierRecord(
                    supplier_id=s.get("supplier_id", _new_uuid()),
                    supplier_name=s.get("name", ""),
                    tier=SupplierTier(s.get("tier", "tier_1")),
                    country=s.get("country", ""),
                    minerals_supplied=minerals,
                    is_cahra_origin=s.get("country", "") in CAHRA_COUNTRIES,
                    audit_date=s.get("audit_date"),
                    certification_scheme=s.get("certification"),
                )
                record.risk_level = self._assess_single_supplier_risk(record)
                parsed.append(record)

            self._suppliers = parsed
            result.suppliers = parsed
            result.suppliers_total = len(parsed)
            result.suppliers_tier1 = sum(
                1 for s in parsed if s.tier == SupplierTier.TIER_1
            )
            result.suppliers_tier2_plus = result.suppliers_total - result.suppliers_tier1
            result.suppliers_high_risk = sum(
                1 for s in parsed if s.risk_level == RiskLevel.HIGH
            )
            result.suppliers_cahra = sum(1 for s in parsed if s.is_cahra_origin)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "total": result.suppliers_total,
                    "high_risk": result.suppliers_high_risk,
                })

            logger.info(
                "Supplier data: %d suppliers (%d high-risk, %d CAHRA)",
                result.suppliers_total,
                result.suppliers_high_risk,
                result.suppliers_cahra,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Supplier data retrieval failed: %s", str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def map_mineral_supply_chain(
        self, context: Dict[str, Any]
    ) -> MineralSupplyChainResult:
        """Map the supply chain for each critical mineral.

        Args:
            context: Pipeline context with mineral sourcing data.

        Returns:
            MineralSupplyChainResult with per-mineral supply chain mapping.
        """
        result = MineralSupplyChainResult()

        try:
            if not self._suppliers:
                self.get_supplier_data(context)

            for mineral in self.config.minerals_in_scope:
                mineral_suppliers = [
                    s for s in self._suppliers
                    if mineral in s.minerals_supplied
                ]
                countries = list({s.country for s in mineral_suppliers if s.country})
                cahra_suppliers = [s for s in mineral_suppliers if s.is_cahra_origin]

                result.minerals_mapped[mineral.value] = {
                    "supplier_count": len(mineral_suppliers),
                    "countries": countries,
                    "cahra_origins": len(cahra_suppliers),
                    "tiers_present": list({
                        s.tier.value for s in mineral_suppliers
                    }),
                }
                result.total_suppliers_by_mineral[mineral.value] = len(
                    mineral_suppliers
                )
                result.cahra_suppliers_by_mineral[mineral.value] = len(
                    cahra_suppliers
                )

                risk_counts: Dict[str, int] = {}
                for s in mineral_suppliers:
                    risk_counts[s.risk_level.value] = (
                        risk_counts.get(s.risk_level.value, 0) + 1
                    )
                result.risk_summary[mineral.value] = risk_counts

            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result.minerals_mapped)

            logger.info(
                "Mineral supply chain mapped for %d minerals",
                len(result.minerals_mapped),
            )

        except Exception as exc:
            result.status = "failed"
            logger.error("Mineral supply chain mapping failed: %s", str(exc))

        return result

    def assess_supplier_risk(
        self, context: Dict[str, Any]
    ) -> SupplierRiskResult:
        """Assess overall supply chain risk per OECD Guidance.

        Args:
            context: Pipeline context with supplier data.

        Returns:
            SupplierRiskResult with risk counts and recommendations.
        """
        result = SupplierRiskResult()

        try:
            if not self._suppliers:
                self.get_supplier_data(context)

            result.suppliers_assessed = len(self._suppliers)
            result.high_risk_count = sum(
                1 for s in self._suppliers if s.risk_level == RiskLevel.HIGH
            )
            result.medium_risk_count = sum(
                1 for s in self._suppliers if s.risk_level == RiskLevel.MEDIUM
            )
            result.low_risk_count = sum(
                1 for s in self._suppliers if s.risk_level == RiskLevel.LOW
            )
            result.cahra_count = sum(
                1 for s in self._suppliers if s.is_cahra_origin
            )

            result.overall_risk = self._compute_overall_risk(result)
            result.risk_factors = self._identify_risk_factors()
            result.mitigation_recommendations = self._generate_recommendations(
                result
            )
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "assessed": result.suppliers_assessed,
                    "high": result.high_risk_count,
                    "overall": result.overall_risk.value,
                })

            logger.info(
                "Risk assessment: %s overall (%d high, %d medium, %d low)",
                result.overall_risk.value,
                result.high_risk_count,
                result.medium_risk_count,
                result.low_risk_count,
            )

        except Exception as exc:
            result.status = "failed"
            logger.error("Risk assessment failed: %s", str(exc))

        return result

    def get_tier_breakdown(
        self, context: Dict[str, Any]
    ) -> TierBreakdownResult:
        """Get supply chain tier-level breakdown for passport data.

        Args:
            context: Pipeline context with tier data.

        Returns:
            TierBreakdownResult with per-tier statistics.
        """
        result = TierBreakdownResult()

        try:
            if not self._suppliers:
                self.get_supplier_data(context)

            for tier in SupplierTier:
                tier_suppliers = [
                    s for s in self._suppliers if s.tier == tier
                ]
                if tier_suppliers:
                    result.tiers[tier.value] = {
                        "supplier_count": len(tier_suppliers),
                        "countries": list({s.country for s in tier_suppliers if s.country}),
                        "minerals": list({
                            m.value
                            for s in tier_suppliers
                            for m in s.minerals_supplied
                        }),
                        "high_risk_pct": round(
                            sum(1 for s in tier_suppliers if s.risk_level == RiskLevel.HIGH)
                            / len(tier_suppliers)
                            * 100,
                            1,
                        ),
                    }

            result.total_tiers_mapped = len(result.tiers)
            result.deepest_tier = max(
                result.tiers.keys(), default=""
            )
            result.mine_of_origin_identified = SupplierTier.TIER_4_MINE.value in result.tiers
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result.tiers)

            logger.info(
                "Tier breakdown: %d tiers, mine origin=%s",
                result.total_tiers_mapped,
                result.mine_of_origin_identified,
            )

        except Exception as exc:
            result.status = "failed"
            logger.error("Tier breakdown failed: %s", str(exc))

        return result

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status.

        Returns:
            Dict with bridge status information.
        """
        return {
            "pack_id": self.config.pack_id,
            "reporting_year": self.config.reporting_year,
            "minerals_in_scope": [m.value for m in self.config.minerals_in_scope],
            "suppliers_loaded": len(self._suppliers),
            "cahra_countries_count": len(CAHRA_COUNTRIES),
            "oecd_guidance_version": self.config.oecd_guidance_version,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assess_single_supplier_risk(self, supplier: SupplierRecord) -> RiskLevel:
        """Assess risk level for a single supplier."""
        if supplier.is_cahra_origin:
            return RiskLevel.HIGH
        if not supplier.country:
            return RiskLevel.MEDIUM

        for mineral in supplier.minerals_supplied:
            country_risks = MINERAL_COUNTRY_RISK.get(mineral.value, {})
            country_risk = country_risks.get(supplier.country, "not_assessed")
            if country_risk == "high":
                return RiskLevel.HIGH
            if country_risk == "medium":
                return RiskLevel.MEDIUM

        if supplier.certification_scheme:
            return RiskLevel.LOW
        return RiskLevel.MEDIUM

    @staticmethod
    def _compute_overall_risk(result: SupplierRiskResult) -> RiskLevel:
        """Compute overall supply chain risk from individual assessments."""
        if result.suppliers_assessed == 0:
            return RiskLevel.NOT_ASSESSED
        high_pct = (
            result.high_risk_count / result.suppliers_assessed * 100
            if result.suppliers_assessed > 0 else 0.0
        )
        if high_pct >= 20.0 or result.high_risk_count >= 3:
            return RiskLevel.HIGH
        if high_pct >= 5.0 or result.medium_risk_count >= 5:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _identify_risk_factors(self) -> List[Dict[str, Any]]:
        """Identify specific risk factors from supplier data."""
        factors: List[Dict[str, Any]] = []
        cahra_suppliers = [s for s in self._suppliers if s.is_cahra_origin]
        if cahra_suppliers:
            factors.append({
                "factor": "CAHRA origin suppliers",
                "count": len(cahra_suppliers),
                "severity": "high",
                "minerals_affected": list({
                    m.value for s in cahra_suppliers for m in s.minerals_supplied
                }),
            })

        unaudited = [s for s in self._suppliers if not s.audit_date]
        if unaudited:
            factors.append({
                "factor": "Unaudited suppliers",
                "count": len(unaudited),
                "severity": "medium",
            })

        uncertified = [s for s in self._suppliers if not s.certification_scheme]
        if uncertified:
            factors.append({
                "factor": "Uncertified suppliers",
                "count": len(uncertified),
                "severity": "medium",
            })

        return factors

    @staticmethod
    def _generate_recommendations(
        result: SupplierRiskResult,
    ) -> List[str]:
        """Generate mitigation recommendations based on risk assessment."""
        recommendations: List[str] = []
        if result.high_risk_count > 0:
            recommendations.append(
                "Conduct enhanced due diligence on high-risk suppliers per OECD Step 3"
            )
            recommendations.append(
                "Establish risk mitigation plan with measurable indicators per Art 40"
            )
        if result.cahra_count > 0:
            recommendations.append(
                "Implement CAHRA-specific monitoring per OECD Annex II"
            )
            recommendations.append(
                "Engage recognized industry scheme for CAHRA suppliers per Art 42"
            )
        if result.medium_risk_count > 2:
            recommendations.append(
                "Strengthen supplier contracts with due diligence clauses per Art 39"
            )
        if result.suppliers_assessed > 0 and result.low_risk_count == result.suppliers_assessed:
            recommendations.append(
                "Maintain current due diligence level; schedule periodic review"
            )
        return recommendations
