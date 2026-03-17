# -*- coding: utf-8 -*-
"""
CSDDDBridge - CSDDD Due Diligence to Battery Regulation Bridge for PACK-020
===============================================================================

Links CSDDD (Corporate Sustainability Due Diligence Directive) findings to
EU Battery Regulation supply chain due diligence (Art 39-42). Maps CSDDD
adverse impact identifications and risk assessments for minerals used in
battery manufacturing (cobalt, lithium, nickel, natural graphite).

The CSDDD requires human rights and environmental due diligence across value
chains. For battery manufacturers, this overlaps significantly with Battery
Regulation Art 39 supply chain DD for raw materials from CAHRAs.

CSDDD-Battery Regulation Overlap:
    - CSDDD Art 6 (impact identification) <-> Battery Reg Art 39 (DD policy)
    - CSDDD Art 7 (prioritisation)        <-> Battery Reg Art 40 (risk management)
    - CSDDD Art 8 (prevention)            <-> Battery Reg Art 40 (risk mitigation)
    - CSDDD Art 9 (cessation)             <-> Battery Reg Art 41 (third-party audit)
    - CSDDD Art 10 (remediation)          <-> Battery Reg Art 39(3) (remediation)
    - CSDDD Art 11 (stakeholder engagement) <-> Battery Reg Art 39(2) (stakeholders)

Methods:
    - get_dd_status()              -- Get overall DD compliance status
    - map_adverse_impacts()        -- Map CSDDD adverse impacts to battery context
    - get_mineral_dd_findings()    -- Get DD findings specific to battery minerals

Legal References:
    - Directive (EU) 2024/1760 (CSDDD), Articles 5-16
    - Regulation (EU) 2023/1542, Articles 39-42
    - OECD Due Diligence Guidance for Responsible Supply Chains of Minerals
    - UN Guiding Principles on Business and Human Rights

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


class DDArticle(str, Enum):
    """CSDDD due diligence process articles."""

    ART_5_POLICY = "art_5"
    ART_6_IDENTIFICATION = "art_6"
    ART_7_PRIORITISATION = "art_7"
    ART_8_PREVENTION = "art_8"
    ART_9_CESSATION = "art_9"
    ART_10_REMEDIATION = "art_10"
    ART_11_STAKEHOLDER = "art_11"
    ART_12_GRIEVANCE = "art_12"
    ART_13_MONITORING = "art_13"
    ART_14_REPORTING = "art_14"


class ImpactType(str, Enum):
    """Types of adverse impacts identified under CSDDD."""

    HUMAN_RIGHTS = "human_rights"
    ENVIRONMENTAL = "environmental"
    LABOR_RIGHTS = "labor_rights"
    CHILD_LABOR = "child_labor"
    FORCED_LABOR = "forced_labor"
    CONFLICT_MINERALS = "conflict_minerals"
    HEALTH_SAFETY = "health_safety"
    LAND_RIGHTS = "land_rights"
    WATER_POLLUTION = "water_pollution"


class ImpactSeverity(str, Enum):
    """Severity classification for adverse impacts."""

    SEVERE = "severe"
    SIGNIFICANT = "significant"
    MODERATE = "moderate"
    MINOR = "minor"
    NOT_ASSESSED = "not_assessed"


class DDComplianceLevel(str, Enum):
    """CSDDD due diligence compliance level."""

    FULL = "full"
    SUBSTANTIAL = "substantial"
    PARTIAL = "partial"
    MINIMAL = "minimal"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


class BatteryMineral(str, Enum):
    """Critical minerals for battery manufacturing."""

    COBALT = "cobalt"
    LITHIUM = "lithium"
    NICKEL = "nickel"
    NATURAL_GRAPHITE = "natural_graphite"
    MANGANESE = "manganese"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class CSDDDBridgeConfig(BaseModel):
    """Configuration for the CSDDD Bridge."""

    pack_id: str = Field(default="PACK-020")
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    minerals_in_scope: List[BatteryMineral] = Field(
        default_factory=lambda: [
            BatteryMineral.COBALT,
            BatteryMineral.LITHIUM,
            BatteryMineral.NICKEL,
            BatteryMineral.NATURAL_GRAPHITE,
        ]
    )
    include_downstream: bool = Field(default=False)


class AdverseImpact(BaseModel):
    """Individual adverse impact identified under CSDDD."""

    impact_id: str = Field(default_factory=_new_uuid)
    impact_type: ImpactType = Field(default=ImpactType.HUMAN_RIGHTS)
    severity: ImpactSeverity = Field(default=ImpactSeverity.NOT_ASSESSED)
    description: str = Field(default="")
    country: str = Field(default="")
    mineral: Optional[BatteryMineral] = Field(None)
    supply_chain_tier: str = Field(default="")
    csddd_article: DDArticle = Field(default=DDArticle.ART_6_IDENTIFICATION)
    battery_reg_article: str = Field(default="Art 39")
    mitigation_action: str = Field(default="")
    is_actual: bool = Field(default=False, description="True=actual, False=potential")
    remediation_provided: bool = Field(default=False)


class DDStatusResult(BaseModel):
    """Overall due diligence compliance status."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    compliance_level: DDComplianceLevel = Field(
        default=DDComplianceLevel.NOT_ASSESSED
    )
    articles_assessed: int = Field(default=0)
    articles_compliant: int = Field(default=0)
    article_compliance: Dict[str, bool] = Field(default_factory=dict)
    battery_reg_overlap: Dict[str, str] = Field(default_factory=dict)
    adverse_impacts_identified: int = Field(default=0)
    severe_impacts: int = Field(default=0)
    minerals_assessed: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class AdverseImpactResult(BaseModel):
    """Result of adverse impact mapping."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    impacts: List[AdverseImpact] = Field(default_factory=list)
    total_impacts: int = Field(default=0)
    actual_impacts: int = Field(default=0)
    potential_impacts: int = Field(default=0)
    by_type: Dict[str, int] = Field(default_factory=dict)
    by_severity: Dict[str, int] = Field(default_factory=dict)
    by_mineral: Dict[str, int] = Field(default_factory=dict)
    mitigation_actions_count: int = Field(default=0)
    provenance_hash: str = Field(default="")


class MineralDDResult(BaseModel):
    """DD findings specific to battery minerals."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    mineral_findings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    total_findings: int = Field(default=0)
    high_risk_minerals: List[str] = Field(default_factory=list)
    oecd_alignment_pct: float = Field(default=0.0)
    battery_reg_articles_satisfied: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# CSDDD-Battery Regulation Overlap Mapping
# ---------------------------------------------------------------------------

CSDDD_BATTERY_OVERLAP: Dict[str, Dict[str, str]] = {
    DDArticle.ART_5_POLICY.value: {
        "battery_article": "Art 39",
        "description": "DD policy integration -> Battery Reg DD policy",
    },
    DDArticle.ART_6_IDENTIFICATION.value: {
        "battery_article": "Art 39(2)",
        "description": "Adverse impact identification -> Raw material risk identification",
    },
    DDArticle.ART_7_PRIORITISATION.value: {
        "battery_article": "Art 40",
        "description": "Impact prioritisation -> Risk management system",
    },
    DDArticle.ART_8_PREVENTION.value: {
        "battery_article": "Art 40",
        "description": "Prevention measures -> Risk mitigation measures",
    },
    DDArticle.ART_9_CESSATION.value: {
        "battery_article": "Art 41",
        "description": "Cessation of adverse impacts -> Third-party audit requirements",
    },
    DDArticle.ART_10_REMEDIATION.value: {
        "battery_article": "Art 39(3)",
        "description": "Remediation provision -> Remediation for affected persons",
    },
    DDArticle.ART_11_STAKEHOLDER.value: {
        "battery_article": "Art 39(2)",
        "description": "Stakeholder engagement -> Meaningful consultation",
    },
    DDArticle.ART_12_GRIEVANCE.value: {
        "battery_article": "Art 39(2)(c)",
        "description": "Grievance mechanism -> Complaints mechanism for supply chain",
    },
    DDArticle.ART_13_MONITORING.value: {
        "battery_article": "Art 40",
        "description": "Monitoring effectiveness -> Ongoing risk management",
    },
    DDArticle.ART_14_REPORTING.value: {
        "battery_article": "Art 39(4)",
        "description": "Public reporting -> Supply chain DD statement",
    },
}

# Mineral-specific known risk patterns
MINERAL_RISK_PROFILES: Dict[str, Dict[str, Any]] = {
    "cobalt": {
        "primary_risks": [
            ImpactType.CHILD_LABOR,
            ImpactType.HEALTH_SAFETY,
            ImpactType.CONFLICT_MINERALS,
        ],
        "high_risk_countries": ["COD", "ZMB"],
        "cahra_relevant": True,
        "oecd_annex_ii_applicable": True,
    },
    "lithium": {
        "primary_risks": [
            ImpactType.WATER_POLLUTION,
            ImpactType.LAND_RIGHTS,
            ImpactType.ENVIRONMENTAL,
        ],
        "high_risk_countries": ["CHL", "ARG", "ZWE"],
        "cahra_relevant": False,
        "oecd_annex_ii_applicable": False,
    },
    "nickel": {
        "primary_risks": [
            ImpactType.ENVIRONMENTAL,
            ImpactType.HEALTH_SAFETY,
            ImpactType.LABOR_RIGHTS,
        ],
        "high_risk_countries": ["IDN", "PHL", "RUS"],
        "cahra_relevant": False,
        "oecd_annex_ii_applicable": False,
    },
    "natural_graphite": {
        "primary_risks": [
            ImpactType.LABOR_RIGHTS,
            ImpactType.ENVIRONMENTAL,
            ImpactType.HEALTH_SAFETY,
        ],
        "high_risk_countries": ["CHN", "MOZ", "TZA"],
        "cahra_relevant": False,
        "oecd_annex_ii_applicable": False,
    },
}


# ---------------------------------------------------------------------------
# CSDDDBridge
# ---------------------------------------------------------------------------


class CSDDDBridge:
    """CSDDD due diligence to Battery Regulation bridge for PACK-020.

    Links CSDDD adverse impact findings to Battery Regulation supply
    chain due diligence obligations (Art 39-42). Maps human rights and
    environmental due diligence for critical battery minerals.

    Attributes:
        config: Bridge configuration.
        _impacts: Cached adverse impact records.

    Example:
        >>> bridge = CSDDDBridge(CSDDDBridgeConfig())
        >>> result = bridge.get_dd_status(context)
        >>> assert result.compliance_level != DDComplianceLevel.NOT_ASSESSED
    """

    def __init__(self, config: Optional[CSDDDBridgeConfig] = None) -> None:
        """Initialize CSDDDBridge."""
        self.config = config or CSDDDBridgeConfig()
        self._impacts: List[AdverseImpact] = []
        logger.info(
            "CSDDDBridge initialized (minerals=%d, downstream=%s)",
            len(self.config.minerals_in_scope),
            self.config.include_downstream,
        )

    def get_dd_status(
        self, context: Dict[str, Any]
    ) -> DDStatusResult:
        """Get overall DD compliance status across CSDDD articles.

        Args:
            context: Pipeline context with DD assessment data.

        Returns:
            DDStatusResult with per-article compliance and Battery Reg overlap.
        """
        result = DDStatusResult(started_at=_utcnow())

        try:
            dd_data = context.get("csddd_dd_data", {})
            article_compliance: Dict[str, bool] = {}

            for article in DDArticle:
                article_key = article.value
                is_compliant = dd_data.get(f"{article_key}_compliant", False)
                article_compliance[article_key] = is_compliant

            result.article_compliance = article_compliance
            result.articles_assessed = len(article_compliance)
            result.articles_compliant = sum(
                1 for v in article_compliance.values() if v
            )

            result.battery_reg_overlap = {
                art: info["battery_article"]
                for art, info in CSDDD_BATTERY_OVERLAP.items()
            }

            # Determine compliance level
            ratio = (
                result.articles_compliant / result.articles_assessed
                if result.articles_assessed > 0 else 0.0
            )
            if ratio >= 0.9:
                result.compliance_level = DDComplianceLevel.FULL
            elif ratio >= 0.7:
                result.compliance_level = DDComplianceLevel.SUBSTANTIAL
            elif ratio >= 0.5:
                result.compliance_level = DDComplianceLevel.PARTIAL
            elif ratio > 0:
                result.compliance_level = DDComplianceLevel.MINIMAL
            else:
                result.compliance_level = DDComplianceLevel.NON_COMPLIANT

            # Count adverse impacts from context
            impacts_data = context.get("adverse_impacts", [])
            result.adverse_impacts_identified = len(impacts_data)
            result.severe_impacts = sum(
                1 for i in impacts_data
                if i.get("severity") == "severe"
            )
            result.minerals_assessed = [
                m.value for m in self.config.minerals_in_scope
            ]
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "level": result.compliance_level.value,
                    "assessed": result.articles_assessed,
                    "compliant": result.articles_compliant,
                    "impacts": result.adverse_impacts_identified,
                })

            logger.info(
                "DD status: %s (%d/%d articles, %d impacts)",
                result.compliance_level.value,
                result.articles_compliant,
                result.articles_assessed,
                result.adverse_impacts_identified,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("DD status check failed: %s", str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def map_adverse_impacts(
        self, context: Dict[str, Any]
    ) -> AdverseImpactResult:
        """Map CSDDD adverse impacts to battery manufacturing context.

        Args:
            context: Pipeline context with adverse impact data.

        Returns:
            AdverseImpactResult with classified impacts by type, severity, mineral.
        """
        result = AdverseImpactResult()

        try:
            raw_impacts = context.get("adverse_impacts", [])
            parsed: List[AdverseImpact] = []

            for imp in raw_impacts:
                mineral_value = imp.get("mineral")
                mineral = (
                    BatteryMineral(mineral_value)
                    if mineral_value in [m.value for m in BatteryMineral]
                    else None
                )

                impact = AdverseImpact(
                    impact_type=ImpactType(imp.get("type", "human_rights")),
                    severity=ImpactSeverity(imp.get("severity", "not_assessed")),
                    description=imp.get("description", ""),
                    country=imp.get("country", ""),
                    mineral=mineral,
                    supply_chain_tier=imp.get("tier", ""),
                    is_actual=imp.get("is_actual", False),
                    mitigation_action=imp.get("mitigation", ""),
                    remediation_provided=imp.get("remediation", False),
                )
                parsed.append(impact)

            self._impacts = parsed
            result.impacts = parsed
            result.total_impacts = len(parsed)
            result.actual_impacts = sum(1 for i in parsed if i.is_actual)
            result.potential_impacts = sum(1 for i in parsed if not i.is_actual)

            # Aggregate by type
            for impact_type in ImpactType:
                count = sum(
                    1 for i in parsed if i.impact_type == impact_type
                )
                if count > 0:
                    result.by_type[impact_type.value] = count

            # Aggregate by severity
            for severity in ImpactSeverity:
                count = sum(
                    1 for i in parsed if i.severity == severity
                )
                if count > 0:
                    result.by_severity[severity.value] = count

            # Aggregate by mineral
            for mineral in BatteryMineral:
                count = sum(
                    1 for i in parsed if i.mineral == mineral
                )
                if count > 0:
                    result.by_mineral[mineral.value] = count

            result.mitigation_actions_count = sum(
                1 for i in parsed if i.mitigation_action
            )
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "total": result.total_impacts,
                    "actual": result.actual_impacts,
                    "types": len(result.by_type),
                })

            logger.info(
                "Adverse impacts: %d total (%d actual, %d potential)",
                result.total_impacts,
                result.actual_impacts,
                result.potential_impacts,
            )

        except Exception as exc:
            result.status = "failed"
            logger.error("Adverse impact mapping failed: %s", str(exc))

        return result

    def get_mineral_dd_findings(
        self, context: Dict[str, Any]
    ) -> MineralDDResult:
        """Get DD findings specific to battery minerals.

        Args:
            context: Pipeline context with mineral DD data.

        Returns:
            MineralDDResult with per-mineral findings and OECD alignment.
        """
        result = MineralDDResult()

        try:
            if not self._impacts:
                self.map_adverse_impacts(context)

            mineral_dd_data = context.get("mineral_dd_data", {})

            for mineral in self.config.minerals_in_scope:
                risk_profile = MINERAL_RISK_PROFILES.get(mineral.value, {})
                mineral_impacts = [
                    i for i in self._impacts if i.mineral == mineral
                ]

                dd_assessment = mineral_dd_data.get(mineral.value, {})

                result.mineral_findings[mineral.value] = {
                    "primary_risks": [
                        r.value for r in risk_profile.get("primary_risks", [])
                    ],
                    "high_risk_countries": risk_profile.get(
                        "high_risk_countries", []
                    ),
                    "cahra_relevant": risk_profile.get("cahra_relevant", False),
                    "oecd_annex_ii_applicable": risk_profile.get(
                        "oecd_annex_ii_applicable", False
                    ),
                    "impacts_identified": len(mineral_impacts),
                    "severe_impacts": sum(
                        1 for i in mineral_impacts
                        if i.severity == ImpactSeverity.SEVERE
                    ),
                    "dd_assessment_completed": dd_assessment.get(
                        "assessment_completed", False
                    ),
                    "third_party_audit": dd_assessment.get(
                        "third_party_audit", False
                    ),
                    "certification_scheme": dd_assessment.get(
                        "certification_scheme", ""
                    ),
                }

            result.total_findings = sum(
                f["impacts_identified"]
                for f in result.mineral_findings.values()
            )
            result.high_risk_minerals = [
                m for m, f in result.mineral_findings.items()
                if f.get("cahra_relevant") or f.get("severe_impacts", 0) > 0
            ]

            assessed_count = sum(
                1 for f in result.mineral_findings.values()
                if f.get("dd_assessment_completed")
            )
            total = len(result.mineral_findings)
            result.oecd_alignment_pct = round(
                assessed_count / total * 100, 1
            ) if total > 0 else 0.0

            # Determine Battery Reg articles satisfied
            if result.oecd_alignment_pct >= 100.0:
                result.battery_reg_articles_satisfied = [
                    "Art 39", "Art 40", "Art 41",
                ]
            elif result.oecd_alignment_pct >= 50.0:
                result.battery_reg_articles_satisfied = ["Art 39"]

            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "minerals": len(result.mineral_findings),
                    "findings": result.total_findings,
                    "oecd_pct": result.oecd_alignment_pct,
                })

            logger.info(
                "Mineral DD: %d minerals, %d findings, %.1f%% OECD aligned",
                len(result.mineral_findings),
                result.total_findings,
                result.oecd_alignment_pct,
            )

        except Exception as exc:
            result.status = "failed"
            logger.error("Mineral DD findings failed: %s", str(exc))

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
            "impacts_loaded": len(self._impacts),
            "csddd_articles_mapped": len(CSDDD_BATTERY_OVERLAP),
            "mineral_risk_profiles": len(MINERAL_RISK_PROFILES),
            "include_downstream": self.config.include_downstream,
        }
