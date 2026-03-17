# -*- coding: utf-8 -*-
"""
Enabling Activity Engine - PACK-008 EU Taxonomy Alignment

This module implements the Article 16 enabling activity classification engine.
Enabling activities are economic activities that directly enable other activities
to make a substantial contribution to an environmental objective, provided that
they do not lead to a lock-in of assets that undermine long-term environmental
goals, and have a substantial positive environmental impact on the basis of
life-cycle considerations.

The engine classifies activities, verifies direct enablement, performs life-cycle
impact checks, assesses technology lock-in risk, and evaluates market distortion
potential.

All assessments are rule-based -- no LLM calls for classification decisions.

Example:
    >>> engine = EnablingActivityEngine()
    >>> is_enabling = engine.is_enabling_activity("3.1", EnvironmentalObjective.CCM)
    >>> result = engine.verify_enablement("3.1", {"enabled_activities": ["4.1", "4.3"]})
    >>> print(f"Enablement verified: {result.enablement_verified}")
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives."""

    CCM = "CCM"
    CCA = "CCA"
    WTR = "WTR"
    CE = "CE"
    PPC = "PPC"
    BIO = "BIO"


class EnablementType(str, Enum):
    """Type of enablement relationship."""

    MANUFACTURING = "MANUFACTURING"         # Manufactures enabling components
    TECHNOLOGY = "TECHNOLOGY"               # Provides enabling technology
    INFRASTRUCTURE = "INFRASTRUCTURE"       # Enables via infrastructure
    PROFESSIONAL_SERVICES = "PROFESSIONAL_SERVICES"  # Advisory / engineering
    RESEARCH = "RESEARCH"                   # R&D enabling future solutions


class LifecycleImpact(str, Enum):
    """Life-cycle environmental impact classification."""

    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    UNKNOWN = "UNKNOWN"


class MarketDistortionRisk(str, Enum):
    """Market distortion risk level."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class EnabledActivity(BaseModel):
    """An activity that is enabled by the enabling activity."""

    activity_id: str = Field(..., description="Enabled activity identifier")
    activity_name: str = Field(..., description="Enabled activity name")
    objective: EnvironmentalObjective = Field(..., description="Objective it contributes to")
    enablement_mechanism: str = Field(..., description="How enablement works")


class EnablingActivityInfo(BaseModel):
    """Metadata for an enabling activity under Article 16."""

    activity_id: str = Field(..., description="Economic activity identifier")
    activity_name: str = Field(..., description="Human-readable activity name")
    nace_codes: List[str] = Field(..., description="NACE codes")
    objective: EnvironmentalObjective = Field(..., description="Primary objective enabled")
    enablement_type: EnablementType = Field(..., description="Type of enablement")
    description: str = Field(..., description="Detailed description")
    enabled_activities: List[EnabledActivity] = Field(
        ..., description="Activities this enables"
    )
    lifecycle_consideration: str = Field(
        ..., description="Life-cycle impact summary"
    )
    da_reference: str = Field(..., description="Delegated Act reference")


class EnablementResult(BaseModel):
    """Result of enablement verification."""

    activity_id: str = Field(..., description="Activity assessed")
    enablement_verified: bool = Field(..., description="Whether enablement is verified")
    enablement_type: Optional[EnablementType] = Field(
        None, description="Type of enablement"
    )
    enabled_activities_count: int = Field(
        ..., description="Number of activities enabled"
    )
    enabled_activities_matched: int = Field(
        ..., description="Matched enabled activities from input"
    )
    details: List[Dict[str, Any]] = Field(
        ..., description="Verification details"
    )
    message: str = Field(..., description="Summary message")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class LifecycleResult(BaseModel):
    """Result of life-cycle impact assessment."""

    activity_id: str = Field(..., description="Activity assessed")
    overall_impact: LifecycleImpact = Field(..., description="Overall life-cycle impact")
    production_impact: LifecycleImpact = Field(..., description="Production-phase impact")
    use_phase_impact: LifecycleImpact = Field(..., description="Use-phase impact")
    end_of_life_impact: LifecycleImpact = Field(..., description="End-of-life impact")
    net_positive: bool = Field(
        ..., description="Whether net life-cycle impact is positive"
    )
    lock_in_risk: MarketDistortionRisk = Field(
        ..., description="Technology lock-in risk level"
    )
    market_distortion_risk: MarketDistortionRisk = Field(
        ..., description="Market distortion risk level"
    )
    factors: List[Dict[str, Any]] = Field(
        ..., description="Assessment factors"
    )
    message: str = Field(..., description="Summary message")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


# ---------------------------------------------------------------------------
# Enabling Activities Database
# ---------------------------------------------------------------------------

ENABLING_ACTIVITIES: Dict[str, Dict[str, Any]] = {
    # Manufacturing of renewable energy technology
    "3.1": {
        "activity_name": "Manufacture of renewable energy technologies",
        "nace_codes": ["C27.11", "C28.11"],
        "objective": "CCM",
        "enablement_type": "MANUFACTURING",
        "description": (
            "Manufacturing of wind turbines, solar panels, and related components "
            "that directly enable low-carbon electricity generation."
        ),
        "enabled_activities": [
            {"activity_id": "4.1", "activity_name": "Electricity generation (solar)",
             "objective": "CCM", "enablement_mechanism": "Provides PV modules"},
            {"activity_id": "4.3", "activity_name": "Electricity generation (wind)",
             "objective": "CCM", "enablement_mechanism": "Provides wind turbines"},
        ],
        "lifecycle_consideration": (
            "Manufacturing has moderate embedded carbon but enables vastly greater "
            "emission reductions over equipment lifetime (20-30 years)."
        ),
        "da_reference": "Climate DA (EU) 2021/2139, Annex I, Section 3.1",
    },
    # Manufacture of batteries
    "3.4": {
        "activity_name": "Manufacture of batteries",
        "nace_codes": ["C27.20"],
        "objective": "CCM",
        "enablement_type": "MANUFACTURING",
        "description": (
            "Manufacturing of batteries and battery systems for energy storage "
            "and electric vehicle applications."
        ),
        "enabled_activities": [
            {"activity_id": "6.5", "activity_name": "Transport (clean vehicles)",
             "objective": "CCM", "enablement_mechanism": "Provides EV batteries"},
            {"activity_id": "4.10", "activity_name": "Storage of electricity",
             "objective": "CCM", "enablement_mechanism": "Provides grid storage"},
        ],
        "lifecycle_consideration": (
            "Battery manufacturing has significant material extraction impact but "
            "enables transport decarbonisation and renewable integration."
        ),
        "da_reference": "Climate DA (EU) 2021/2139, Annex I, Section 3.4",
    },
    # Manufacture of energy efficiency equipment for buildings
    "3.5": {
        "activity_name": "Manufacture of energy efficiency equipment for buildings",
        "nace_codes": ["C27.51", "C28.25"],
        "objective": "CCM",
        "enablement_type": "MANUFACTURING",
        "description": (
            "Manufacturing of heat pumps, high-efficiency boilers, insulation "
            "materials, energy-efficient windows, and building automation systems."
        ),
        "enabled_activities": [
            {"activity_id": "7.2", "activity_name": "Renovation of existing buildings",
             "objective": "CCM",
             "enablement_mechanism": "Provides insulation and HVAC equipment"},
            {"activity_id": "7.1", "activity_name": "Construction of new buildings",
             "objective": "CCM",
             "enablement_mechanism": "Provides energy-efficient building components"},
        ],
        "lifecycle_consideration": (
            "Equipment manufacturing has moderate environmental footprint but enables "
            "30%+ energy demand reduction in buildings over decades of use."
        ),
        "da_reference": "Climate DA (EU) 2021/2139, Annex I, Section 3.5",
    },
    # Manufacture of other low-carbon technologies
    "3.6": {
        "activity_name": "Manufacture of other low-carbon technologies",
        "nace_codes": ["C28.29"],
        "objective": "CCM",
        "enablement_type": "MANUFACTURING",
        "description": (
            "Manufacturing of technologies primarily aimed at substantial GHG "
            "emission reductions in other sectors."
        ),
        "enabled_activities": [
            {"activity_id": "4.1", "activity_name": "Electricity generation (low-carbon)",
             "objective": "CCM",
             "enablement_mechanism": "Provides low-carbon technology components"},
        ],
        "lifecycle_consideration": (
            "Variable by technology type. Net positive where enabled emission "
            "reductions exceed manufacturing footprint."
        ),
        "da_reference": "Climate DA (EU) 2021/2139, Annex I, Section 3.6",
    },
    # Installation of EV charging stations
    "7.4": {
        "activity_name": "Installation, maintenance and repair of charging stations",
        "nace_codes": ["F42.22", "F43.21"],
        "objective": "CCM",
        "enablement_type": "INFRASTRUCTURE",
        "description": (
            "Installation and operation of EV charging infrastructure including "
            "Level 2, DC fast charging, and ultra-rapid charging stations."
        ),
        "enabled_activities": [
            {"activity_id": "6.5", "activity_name": "Transport (clean vehicles)",
             "objective": "CCM",
             "enablement_mechanism": "Provides charging infrastructure for EVs"},
        ],
        "lifecycle_consideration": (
            "Charging infrastructure has low material footprint relative to the "
            "transport decarbonisation it enables over its 15-20 year lifetime."
        ),
        "da_reference": "Climate DA (EU) 2021/2139, Annex I, Section 7.4",
    },
    # Installation of renewable energy technologies
    "7.6": {
        "activity_name": "Installation, maintenance and repair of renewable energy technologies",
        "nace_codes": ["F43.21", "F43.22"],
        "objective": "CCM",
        "enablement_type": "INFRASTRUCTURE",
        "description": (
            "Installation of solar PV systems, solar thermal, heat pumps, and "
            "other renewable energy equipment on buildings."
        ),
        "enabled_activities": [
            {"activity_id": "4.1", "activity_name": "Electricity generation (solar)",
             "objective": "CCM",
             "enablement_mechanism": "Installs rooftop/building-integrated solar"},
            {"activity_id": "7.1", "activity_name": "Construction of new buildings",
             "objective": "CCM",
             "enablement_mechanism": "Integrates renewable energy into buildings"},
        ],
        "lifecycle_consideration": (
            "Installation services have minimal environmental impact and enable "
            "significant renewable energy generation over equipment lifetime."
        ),
        "da_reference": "Climate DA (EU) 2021/2139, Annex I, Section 7.6",
    },
    # Professional services - energy audits
    "9.3": {
        "activity_name": "Professional services related to energy performance of buildings",
        "nace_codes": ["M71.12", "M71.20"],
        "objective": "CCM",
        "enablement_type": "PROFESSIONAL_SERVICES",
        "description": (
            "Energy audits, building energy simulation, certification (EPC), and "
            "technical advisory for improving energy performance."
        ),
        "enabled_activities": [
            {"activity_id": "7.2", "activity_name": "Renovation of existing buildings",
             "objective": "CCM",
             "enablement_mechanism": "Identifies and guides energy efficiency improvements"},
        ],
        "lifecycle_consideration": (
            "Professional services have negligible direct environmental impact and "
            "enable evidence-based building renovation decisions."
        ),
        "da_reference": "Climate DA (EU) 2021/2139, Annex I, Section 9.3",
    },
    # Manufacture of insulation materials
    "3.18": {
        "activity_name": "Manufacture of insulation materials",
        "nace_codes": ["C23.99"],
        "objective": "CCM",
        "enablement_type": "MANUFACTURING",
        "description": (
            "Manufacturing of thermal insulation products (mineral wool, EPS, XPS, "
            "PIR, cellulose) for building energy efficiency."
        ),
        "enabled_activities": [
            {"activity_id": "7.2", "activity_name": "Renovation of existing buildings",
             "objective": "CCM",
             "enablement_mechanism": "Provides thermal insulation for envelope upgrades"},
            {"activity_id": "7.1", "activity_name": "Construction of new buildings",
             "objective": "CCM",
             "enablement_mechanism": "Provides insulation for high-performance envelopes"},
        ],
        "lifecycle_consideration": (
            "Manufacturing process has embedded energy but insulation products "
            "save 10-100x their embodied energy over building lifetime."
        ),
        "da_reference": "Climate DA (EU) 2021/2139, Annex I, Section 3.18 (illustrative)",
    },
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EnablingActivityEngine:
    """
    Enabling Activity Engine for EU Taxonomy Article 16 classification.

    Classifies activities as enabling, verifies direct enablement relationships,
    performs life-cycle impact assessments, checks technology lock-in risk, and
    evaluates market distortion potential.

    Attributes:
        activities: Registry of known enabling activities

    Example:
        >>> engine = EnablingActivityEngine()
        >>> result = engine.verify_enablement("3.1", {"enabled_activities": ["4.1"]})
        >>> assert result.enablement_verified
    """

    def __init__(self) -> None:
        """Initialize the Enabling Activity Engine."""
        self.activities = self._load_activities()
        logger.info(
            f"EnablingActivityEngine initialized with "
            f"{len(self.activities)} enabling activities"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_enabling_activity(
        self,
        activity_id: str,
        objective: EnvironmentalObjective,
    ) -> bool:
        """
        Determine whether an activity is classified as enabling under Article 16.

        Args:
            activity_id: Economic activity identifier
            objective: Environmental objective

        Returns:
            True if the activity is a recognized enabling activity for the objective
        """
        info = self.activities.get(activity_id)
        if info is None:
            return False
        return info.objective == objective

    def verify_enablement(
        self,
        activity_id: str,
        data: Dict[str, Any],
    ) -> EnablementResult:
        """
        Verify that the activity directly enables other taxonomy-aligned activities.

        Args:
            activity_id: Enabling activity identifier
            data: Dictionary containing 'enabled_activities' list of activity IDs
                  that this activity claims to enable

        Returns:
            EnablementResult with verification details

        Raises:
            ValueError: If activity is not a recognized enabling activity
        """
        start = datetime.utcnow()
        info = self.activities.get(activity_id)
        if info is None:
            raise ValueError(
                f"Activity {activity_id} is not a recognized enabling activity"
            )

        claimed_enabled = data.get("enabled_activities", [])
        known_enabled_ids = {ea.activity_id for ea in info.enabled_activities}

        details: List[Dict[str, Any]] = []
        matched = 0

        for claimed_id in claimed_enabled:
            if claimed_id in known_enabled_ids:
                matched += 1
                ea = next(
                    e for e in info.enabled_activities
                    if e.activity_id == claimed_id
                )
                details.append({
                    "enabled_activity_id": claimed_id,
                    "verified": True,
                    "activity_name": ea.activity_name,
                    "mechanism": ea.enablement_mechanism,
                })
            else:
                details.append({
                    "enabled_activity_id": claimed_id,
                    "verified": False,
                    "activity_name": "Unknown",
                    "mechanism": "Not in known enablement registry",
                })

        # Enablement is verified if at least one claimed activity matches
        verified = matched > 0

        provenance = self._provenance({
            "type": "enablement_verification",
            "activity_id": activity_id,
            "ts": start.isoformat(),
        })

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            f"Enablement verification for {activity_id}: "
            f"matched={matched}/{len(claimed_enabled)}, "
            f"verified={verified} in {elapsed_ms:.1f}ms"
        )

        return EnablementResult(
            activity_id=activity_id,
            enablement_verified=verified,
            enablement_type=info.enablement_type,
            enabled_activities_count=len(info.enabled_activities),
            enabled_activities_matched=matched,
            details=details,
            message=(
                f"Enablement {'VERIFIED' if verified else 'NOT VERIFIED'} "
                f"for {info.activity_name} ({matched}/{len(claimed_enabled)} matched)"
            ),
            provenance_hash=provenance,
        )

    def check_lifecycle(
        self,
        activity_id: str,
        data: Dict[str, Any],
    ) -> LifecycleResult:
        """
        Assess life-cycle environmental impact and lock-in / distortion risks.

        Evaluates production-phase, use-phase, and end-of-life impacts, then
        determines net positivity, technology lock-in risk, and market distortion.

        Args:
            activity_id: Enabling activity identifier
            data: Dictionary with optional keys:
                  'production_emissions_tco2e', 'use_phase_savings_tco2e',
                  'end_of_life_recyclability_pct', 'market_share_pct',
                  'alternative_technologies_count'

        Returns:
            LifecycleResult with impact classification and risk levels

        Raises:
            ValueError: If activity is not a recognized enabling activity
        """
        start = datetime.utcnow()
        info = self.activities.get(activity_id)
        if info is None:
            raise ValueError(
                f"Activity {activity_id} is not a recognized enabling activity"
            )

        factors: List[Dict[str, Any]] = []

        # Production phase
        prod_impact = self._assess_production(data, factors)

        # Use phase
        use_impact = self._assess_use_phase(data, factors)

        # End of life
        eol_impact = self._assess_end_of_life(data, factors)

        # Overall impact
        overall_impact = self._determine_overall(prod_impact, use_impact, eol_impact)

        # Net positive if use-phase benefits outweigh production + EOL
        net_positive = overall_impact == LifecycleImpact.POSITIVE

        # Lock-in risk
        lock_in = self._assess_lock_in_risk(data, factors)

        # Market distortion
        distortion = self._assess_market_distortion(data, factors)

        provenance = self._provenance({
            "type": "lifecycle_assessment",
            "activity_id": activity_id,
            "ts": start.isoformat(),
        })

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            f"Lifecycle assessment for {activity_id}: "
            f"overall={overall_impact.value}, net_positive={net_positive}, "
            f"lock_in={lock_in.value}, distortion={distortion.value} "
            f"in {elapsed_ms:.1f}ms"
        )

        return LifecycleResult(
            activity_id=activity_id,
            overall_impact=overall_impact,
            production_impact=prod_impact,
            use_phase_impact=use_impact,
            end_of_life_impact=eol_impact,
            net_positive=net_positive,
            lock_in_risk=lock_in,
            market_distortion_risk=distortion,
            factors=factors,
            message=(
                f"Lifecycle assessment for {info.activity_name}: "
                f"{'NET POSITIVE' if net_positive else 'NOT NET POSITIVE'}, "
                f"lock-in risk={lock_in.value}"
            ),
            provenance_hash=provenance,
        )

    def get_enabling_info(self, activity_id: str) -> Optional[EnablingActivityInfo]:
        """
        Retrieve enabling activity metadata.

        Args:
            activity_id: Activity identifier

        Returns:
            EnablingActivityInfo or None
        """
        return self.activities.get(activity_id)

    def get_all_enabling_activities(self) -> List[str]:
        """
        Return all enabling activity IDs in the registry.

        Returns:
            Sorted list of activity identifiers
        """
        return sorted(self.activities.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_activities(self) -> Dict[str, EnablingActivityInfo]:
        """Parse ENABLING_ACTIVITIES into validated objects."""
        result: Dict[str, EnablingActivityInfo] = {}
        for aid, raw in ENABLING_ACTIVITIES.items():
            enabled = [
                EnabledActivity(**ea)
                for ea in raw["enabled_activities"]
            ]
            result[aid] = EnablingActivityInfo(
                activity_id=aid,
                activity_name=raw["activity_name"],
                nace_codes=raw["nace_codes"],
                objective=EnvironmentalObjective(raw["objective"]),
                enablement_type=EnablementType(raw["enablement_type"]),
                description=raw["description"],
                enabled_activities=enabled,
                lifecycle_consideration=raw["lifecycle_consideration"],
                da_reference=raw["da_reference"],
            )
        return result

    def _assess_production(
        self, data: Dict[str, Any], factors: List[Dict[str, Any]]
    ) -> LifecycleImpact:
        """Assess production-phase environmental impact."""
        prod_emissions = data.get("production_emissions_tco2e")
        if prod_emissions is not None:
            val = float(prod_emissions)
            if val > 1000:
                impact = LifecycleImpact.NEGATIVE
            elif val > 100:
                impact = LifecycleImpact.NEUTRAL
            else:
                impact = LifecycleImpact.POSITIVE
            factors.append({
                "phase": "production",
                "metric": "production_emissions_tco2e",
                "value": val,
                "impact": impact.value,
            })
            return impact

        factors.append({
            "phase": "production",
            "metric": "production_emissions_tco2e",
            "value": None,
            "impact": "UNKNOWN",
        })
        return LifecycleImpact.UNKNOWN

    def _assess_use_phase(
        self, data: Dict[str, Any], factors: List[Dict[str, Any]]
    ) -> LifecycleImpact:
        """Assess use-phase environmental impact (savings enabled)."""
        use_savings = data.get("use_phase_savings_tco2e")
        if use_savings is not None:
            val = float(use_savings)
            if val > 0:
                impact = LifecycleImpact.POSITIVE
            elif val == 0:
                impact = LifecycleImpact.NEUTRAL
            else:
                impact = LifecycleImpact.NEGATIVE
            factors.append({
                "phase": "use",
                "metric": "use_phase_savings_tco2e",
                "value": val,
                "impact": impact.value,
            })
            return impact

        factors.append({
            "phase": "use",
            "metric": "use_phase_savings_tco2e",
            "value": None,
            "impact": "UNKNOWN",
        })
        return LifecycleImpact.UNKNOWN

    def _assess_end_of_life(
        self, data: Dict[str, Any], factors: List[Dict[str, Any]]
    ) -> LifecycleImpact:
        """Assess end-of-life recyclability."""
        recyclability = data.get("end_of_life_recyclability_pct")
        if recyclability is not None:
            val = float(recyclability)
            if val >= 80:
                impact = LifecycleImpact.POSITIVE
            elif val >= 50:
                impact = LifecycleImpact.NEUTRAL
            else:
                impact = LifecycleImpact.NEGATIVE
            factors.append({
                "phase": "end_of_life",
                "metric": "end_of_life_recyclability_pct",
                "value": val,
                "impact": impact.value,
            })
            return impact

        factors.append({
            "phase": "end_of_life",
            "metric": "end_of_life_recyclability_pct",
            "value": None,
            "impact": "UNKNOWN",
        })
        return LifecycleImpact.UNKNOWN

    def _determine_overall(
        self,
        production: LifecycleImpact,
        use_phase: LifecycleImpact,
        end_of_life: LifecycleImpact,
    ) -> LifecycleImpact:
        """Determine overall life-cycle impact from phase impacts."""
        score_map = {
            LifecycleImpact.POSITIVE: 1,
            LifecycleImpact.NEUTRAL: 0,
            LifecycleImpact.NEGATIVE: -1,
            LifecycleImpact.UNKNOWN: 0,
        }
        # Use-phase gets double weight (most important phase for enabling)
        total = score_map[production] + 2 * score_map[use_phase] + score_map[end_of_life]

        if total >= 2:
            return LifecycleImpact.POSITIVE
        elif total <= -2:
            return LifecycleImpact.NEGATIVE
        else:
            return LifecycleImpact.NEUTRAL

    def _assess_lock_in_risk(
        self, data: Dict[str, Any], factors: List[Dict[str, Any]]
    ) -> MarketDistortionRisk:
        """Assess technology lock-in risk."""
        alt_count = data.get("alternative_technologies_count")
        if alt_count is not None:
            val = int(alt_count)
            if val >= 3:
                risk = MarketDistortionRisk.LOW
            elif val >= 1:
                risk = MarketDistortionRisk.MEDIUM
            else:
                risk = MarketDistortionRisk.HIGH
            factors.append({
                "category": "lock_in",
                "metric": "alternative_technologies_count",
                "value": val,
                "risk": risk.value,
            })
            return risk

        factors.append({
            "category": "lock_in",
            "metric": "alternative_technologies_count",
            "value": None,
            "risk": "LOW",
        })
        return MarketDistortionRisk.LOW

    def _assess_market_distortion(
        self, data: Dict[str, Any], factors: List[Dict[str, Any]]
    ) -> MarketDistortionRisk:
        """Assess market distortion potential."""
        market_share = data.get("market_share_pct")
        if market_share is not None:
            val = float(market_share)
            if val > 50:
                risk = MarketDistortionRisk.HIGH
            elif val > 25:
                risk = MarketDistortionRisk.MEDIUM
            else:
                risk = MarketDistortionRisk.LOW
            factors.append({
                "category": "market_distortion",
                "metric": "market_share_pct",
                "value": val,
                "risk": risk.value,
            })
            return risk

        factors.append({
            "category": "market_distortion",
            "metric": "market_share_pct",
            "value": None,
            "risk": "LOW",
        })
        return MarketDistortionRisk.LOW

    @staticmethod
    def _provenance(data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
