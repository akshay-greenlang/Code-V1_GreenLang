# -*- coding: utf-8 -*-
"""
Transition Activity Engine - PACK-008 EU Taxonomy Alignment

This module implements the Article 10(2) transition activity identification
and assessment engine. Transition activities are economic activities for which
there are no technologically and economically feasible low-carbon alternatives,
but which support the transition to a climate-neutral economy by having GHG
emission levels corresponding to the best available technology (BAT).

The engine identifies whether an activity qualifies as transitional, assesses
BAT compliance, verifies lock-in avoidance (no hampering of low-carbon
alternatives), documents transition pathways, and tracks sunset dates for
transitional status.

All assessments are rule-based and deterministic -- no LLM calls for scoring.

Example:
    >>> engine = TransitionActivityEngine()
    >>> is_transition = engine.is_transition_activity("4.29", EnvironmentalObjective.CCM)
    >>> bat_result = engine.assess_bat("3.7", {"tco2e_per_t_clinker": 0.45})
    >>> print(f"BAT met: {bat_result.bat_met}")
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


class TransitionStatus(str, Enum):
    """Transition activity lifecycle status."""

    ACTIVE = "ACTIVE"
    SUNSET_APPROACHING = "SUNSET_APPROACHING"
    EXPIRED = "EXPIRED"
    UNDER_REVIEW = "UNDER_REVIEW"


class LockInRisk(str, Enum):
    """Lock-in risk classification."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class BATBenchmark(BaseModel):
    """Best Available Technology benchmark for a transition activity."""

    metric: str = Field(..., description="Performance metric key")
    unit: str = Field(..., description="Unit of measurement")
    bat_threshold: float = Field(..., description="BAT benchmark threshold")
    description: str = Field(..., description="BAT requirement description")


class TransitionActivityInfo(BaseModel):
    """Metadata for a transition activity under Article 10(2)."""

    activity_id: str = Field(..., description="Economic activity identifier")
    activity_name: str = Field(..., description="Human-readable activity name")
    nace_codes: List[str] = Field(..., description="Applicable NACE codes")
    objective: EnvironmentalObjective = Field(..., description="Primary objective")
    justification: str = Field(
        ..., description="Why no low-carbon alternative exists"
    )
    bat_benchmarks: List[BATBenchmark] = Field(
        ..., description="BAT performance benchmarks"
    )
    sunset_date: Optional[str] = Field(
        None, description="Date transitional status expires (YYYY-MM-DD)"
    )
    status: TransitionStatus = Field(..., description="Current status")
    da_reference: str = Field(..., description="Delegated Act article reference")
    low_carbon_pathway: str = Field(
        ..., description="Expected pathway to low-carbon alternative"
    )


class BATResult(BaseModel):
    """Result of Best Available Technology assessment."""

    activity_id: str = Field(..., description="Activity assessed")
    bat_met: bool = Field(..., description="Whether BAT is met")
    benchmarks_evaluated: int = Field(..., description="Number of benchmarks checked")
    benchmarks_passed: int = Field(..., description="Number of benchmarks met")
    details: List[Dict[str, Any]] = Field(
        ..., description="Per-benchmark evaluation details"
    )
    message: str = Field(..., description="Summary message")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class LockInResult(BaseModel):
    """Result of lock-in avoidance verification."""

    activity_id: str = Field(..., description="Activity assessed")
    lock_in_avoided: bool = Field(
        ..., description="Whether lock-in is avoided"
    )
    risk_level: LockInRisk = Field(..., description="Lock-in risk level")
    asset_lifetime_years: Optional[int] = Field(
        None, description="Expected asset lifetime"
    )
    switching_cost_factor: Optional[float] = Field(
        None, description="Relative cost of switching to low-carbon (1.0 = same cost)"
    )
    factors_assessed: List[Dict[str, Any]] = Field(
        ..., description="Lock-in factors assessed"
    )
    message: str = Field(..., description="Summary message")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class TransitionPathway(BaseModel):
    """Documented transition pathway for an activity."""

    activity_id: str = Field(..., description="Activity identifier")
    activity_name: str = Field(..., description="Activity name")
    current_status: TransitionStatus = Field(..., description="Current status")
    sunset_date: Optional[str] = Field(None, description="Expected sunset date")
    milestones: List[Dict[str, str]] = Field(
        ..., description="Pathway milestones with dates"
    )
    low_carbon_alternatives: List[str] = Field(
        ..., description="Emerging low-carbon alternatives"
    )
    expected_timeline_years: int = Field(
        ..., description="Estimated years until low-carbon alternative viable"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


# ---------------------------------------------------------------------------
# Transition Activities Database
# ---------------------------------------------------------------------------

TRANSITION_ACTIVITIES: Dict[str, Dict[str, Any]] = {
    # Gas electricity generation -- Complementary DA
    "4.29": {
        "activity_name": "Electricity generation from fossil gaseous fuels",
        "nace_codes": ["D35.11"],
        "objective": "CCM",
        "justification": (
            "Natural gas serves as a transitional fuel while renewable + storage "
            "scale up. No economically viable zero-carbon dispatchable alternative "
            "at required scale in all EU member states."
        ),
        "bat_benchmarks": [
            {"metric": "gco2e_kwh", "unit": "gCO2e/kWh", "bat_threshold": 270.0,
             "description": "Life-cycle emissions below 270 gCO2e/kWh"},
            {"metric": "kgco2e_kw_annual", "unit": "kgCO2e/kW/year", "bat_threshold": 550.0,
             "description": "Annual direct emissions below 550 kgCO2e/kW capacity"},
        ],
        "sunset_date": "2030-12-31",
        "status": "ACTIVE",
        "da_reference": "Complementary DA (EU) 2022/1214, Annex I, Section 4.29",
        "low_carbon_pathway": "Hydrogen blending -> 100% green hydrogen / renewable + storage",
    },
    # Cement manufacture
    "3.7": {
        "activity_name": "Manufacture of cement",
        "nace_codes": ["C23.51"],
        "objective": "CCM",
        "justification": (
            "Cement production involves process emissions from calcination that "
            "cannot be eliminated with current technology. CCS/CCU at scale is "
            "not yet economically viable."
        ),
        "bat_benchmarks": [
            {"metric": "tco2e_per_t_clinker", "unit": "tCO2e/t clinker",
             "bat_threshold": 0.469,
             "description": "Specific emissions below 0.469 tCO2e per tonne of clinker"},
        ],
        "sunset_date": None,
        "status": "ACTIVE",
        "da_reference": "Climate DA (EU) 2021/2139, Annex I, Section 3.7",
        "low_carbon_pathway": "Clinker substitution -> novel binders -> CCS/CCU integration",
    },
    # Aluminium
    "3.8": {
        "activity_name": "Manufacture of aluminium",
        "nace_codes": ["C24.42"],
        "objective": "CCM",
        "justification": (
            "Primary aluminium smelting requires large electricity input and "
            "generates process CO2 from anode consumption. Inert anode technology "
            "is still in pilot phase."
        ),
        "bat_benchmarks": [
            {"metric": "tco2e_per_t_aluminium", "unit": "tCO2e/t",
             "bat_threshold": 1.484,
             "description": "GHG emissions below 1.484 tCO2e per tonne of aluminium"},
        ],
        "sunset_date": None,
        "status": "ACTIVE",
        "da_reference": "Climate DA (EU) 2021/2139, Annex I, Section 3.8",
        "low_carbon_pathway": "Renewable-powered smelting -> inert anodes -> secondary recycling",
    },
    # Iron and steel
    "3.9": {
        "activity_name": "Manufacture of iron and steel",
        "nace_codes": ["C24.10"],
        "objective": "CCM",
        "justification": (
            "Blast furnace steelmaking uses coke as a reductant, generating "
            "significant process CO2. DRI-H2 technology is emerging but not yet "
            "at commercial scale in most regions."
        ),
        "bat_benchmarks": [
            {"metric": "tco2e_per_t_steel", "unit": "tCO2e/t",
             "bat_threshold": 1.331,
             "description": "GHG emissions below 1.331 tCO2e per tonne of steel"},
        ],
        "sunset_date": None,
        "status": "ACTIVE",
        "da_reference": "Climate DA (EU) 2021/2139, Annex I, Section 3.9",
        "low_carbon_pathway": "Scrap-based EAF -> DRI with green hydrogen -> full decarbonisation",
    },
    # Cogeneration from gas
    "4.30": {
        "activity_name": "High-efficiency co-generation from fossil gaseous fuels",
        "nace_codes": ["D35.11", "D35.30"],
        "objective": "CCM",
        "justification": (
            "Gas cogeneration provides efficient combined heat and power where "
            "electrification of heat is not yet feasible. It is more efficient "
            "than separate gas boiler + grid electricity."
        ),
        "bat_benchmarks": [
            {"metric": "gco2e_kwh", "unit": "gCO2e/kWh", "bat_threshold": 270.0,
             "description": "Life-cycle emissions below 270 gCO2e/kWh"},
        ],
        "sunset_date": "2030-12-31",
        "status": "ACTIVE",
        "da_reference": "Complementary DA (EU) 2022/1214, Annex I, Section 4.30",
        "low_carbon_pathway": "Hydrogen blending -> heat pumps + renewable electricity",
    },
    # Heat generation from gas
    "4.31": {
        "activity_name": "Production of heat/cool from fossil gaseous fuels",
        "nace_codes": ["D35.30"],
        "objective": "CCM",
        "justification": (
            "Gas boilers remain dominant in district heating in many EU regions. "
            "Renewable heat alternatives are scaling but infrastructure conversion "
            "requires time."
        ),
        "bat_benchmarks": [
            {"metric": "gco2e_kwh_th", "unit": "gCO2e/kWh_th", "bat_threshold": 270.0,
             "description": "Life-cycle emissions below 270 gCO2e/kWh thermal"},
        ],
        "sunset_date": "2030-12-31",
        "status": "ACTIVE",
        "da_reference": "Complementary DA (EU) 2022/1214, Annex I, Section 4.31",
        "low_carbon_pathway": "Biomethane injection -> heat pump networks -> geothermal",
    },
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TransitionActivityEngine:
    """
    Transition Activity Engine for EU Taxonomy Article 10(2) assessment.

    Identifies transition activities, assesses Best Available Technology (BAT)
    compliance, verifies lock-in avoidance, documents transition pathways,
    and tracks sunset dates.

    Attributes:
        activities: Registry of known transition activities

    Example:
        >>> engine = TransitionActivityEngine()
        >>> engine.is_transition_activity("3.7", EnvironmentalObjective.CCM)
        True
    """

    def __init__(self) -> None:
        """Initialize the Transition Activity Engine."""
        self.activities = self._load_activities()
        logger.info(
            f"TransitionActivityEngine initialized with "
            f"{len(self.activities)} transition activities"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_transition_activity(
        self,
        activity_id: str,
        objective: EnvironmentalObjective,
    ) -> bool:
        """
        Determine whether an activity is classified as transitional under Article 10(2).

        Args:
            activity_id: Economic activity identifier
            objective: Environmental objective

        Returns:
            True if the activity is a recognized transition activity for the objective
        """
        info = self.activities.get(activity_id)
        if info is None:
            return False
        if info.objective != objective:
            return False
        if info.status == TransitionStatus.EXPIRED:
            logger.warning(
                f"Activity {activity_id} transition status has EXPIRED"
            )
            return False
        return True

    def assess_bat(
        self,
        activity_id: str,
        data: Dict[str, Any],
    ) -> BATResult:
        """
        Assess whether the activity meets Best Available Technology benchmarks.

        Args:
            activity_id: Transition activity identifier
            data: Dictionary of metric values

        Returns:
            BATResult with benchmark evaluation details

        Raises:
            ValueError: If activity is not a transition activity
        """
        start = datetime.utcnow()
        info = self.activities.get(activity_id)
        if info is None:
            raise ValueError(
                f"Activity {activity_id} is not a recognized transition activity"
            )

        details: List[Dict[str, Any]] = []
        passed_count = 0

        for benchmark in info.bat_benchmarks:
            actual = data.get(benchmark.metric)
            if actual is None:
                details.append({
                    "metric": benchmark.metric,
                    "passed": False,
                    "actual": None,
                    "threshold": benchmark.bat_threshold,
                    "unit": benchmark.unit,
                    "message": f"Missing data for metric '{benchmark.metric}'",
                })
                continue

            actual_val = float(actual)
            met = actual_val <= benchmark.bat_threshold
            if met:
                passed_count += 1

            details.append({
                "metric": benchmark.metric,
                "passed": met,
                "actual": actual_val,
                "threshold": benchmark.bat_threshold,
                "unit": benchmark.unit,
                "message": (
                    f"{benchmark.description}: actual={actual_val} "
                    f"{'<=' if met else '>'} {benchmark.bat_threshold} {benchmark.unit}"
                ),
            })

        all_met = passed_count == len(info.bat_benchmarks)
        provenance = self._provenance({
            "type": "bat_assessment",
            "activity_id": activity_id,
            "ts": start.isoformat(),
        })

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            f"BAT assessment for {activity_id}: "
            f"{passed_count}/{len(info.bat_benchmarks)} met, "
            f"overall={'PASS' if all_met else 'FAIL'} in {elapsed_ms:.1f}ms"
        )

        return BATResult(
            activity_id=activity_id,
            bat_met=all_met,
            benchmarks_evaluated=len(info.bat_benchmarks),
            benchmarks_passed=passed_count,
            details=details,
            message=f"BAT {'met' if all_met else 'NOT met'} for {info.activity_name}",
            provenance_hash=provenance,
        )

    def check_lock_in(
        self,
        activity_id: str,
        data: Dict[str, Any],
    ) -> LockInResult:
        """
        Verify that the transition activity does not create carbon lock-in.

        Lock-in is assessed based on asset lifetime, switching costs, and
        whether the asset hampers development of low-carbon alternatives.

        Args:
            activity_id: Transition activity identifier
            data: Dictionary with keys like 'asset_lifetime_years',
                  'switching_cost_factor', 'hampers_alternatives'

        Returns:
            LockInResult with risk classification

        Raises:
            ValueError: If activity is not a transition activity
        """
        start = datetime.utcnow()
        info = self.activities.get(activity_id)
        if info is None:
            raise ValueError(
                f"Activity {activity_id} is not a recognized transition activity"
            )

        asset_lifetime = data.get("asset_lifetime_years")
        switching_cost = data.get("switching_cost_factor")
        hampers = data.get("hampers_alternatives", False)

        factors: List[Dict[str, Any]] = []
        risk_score = 0

        # Factor 1: Asset lifetime
        if asset_lifetime is not None:
            lifetime_int = int(asset_lifetime)
            if lifetime_int > 30:
                factors.append({
                    "factor": "asset_lifetime",
                    "value": lifetime_int,
                    "risk": "HIGH",
                    "message": f"Asset lifetime {lifetime_int}y exceeds 30y threshold",
                })
                risk_score += 2
            elif lifetime_int > 15:
                factors.append({
                    "factor": "asset_lifetime",
                    "value": lifetime_int,
                    "risk": "MEDIUM",
                    "message": f"Asset lifetime {lifetime_int}y is moderate",
                })
                risk_score += 1
            else:
                factors.append({
                    "factor": "asset_lifetime",
                    "value": lifetime_int,
                    "risk": "LOW",
                    "message": f"Asset lifetime {lifetime_int}y is acceptable",
                })

        # Factor 2: Switching cost
        if switching_cost is not None:
            switch_val = float(switching_cost)
            if switch_val > 3.0:
                factors.append({
                    "factor": "switching_cost",
                    "value": switch_val,
                    "risk": "HIGH",
                    "message": f"Switching cost factor {switch_val}x indicates high lock-in",
                })
                risk_score += 2
            elif switch_val > 1.5:
                factors.append({
                    "factor": "switching_cost",
                    "value": switch_val,
                    "risk": "MEDIUM",
                    "message": f"Switching cost factor {switch_val}x is moderate",
                })
                risk_score += 1
            else:
                factors.append({
                    "factor": "switching_cost",
                    "value": switch_val,
                    "risk": "LOW",
                    "message": f"Switching cost factor {switch_val}x is acceptable",
                })

        # Factor 3: Hampers alternatives
        if hampers:
            factors.append({
                "factor": "hampers_alternatives",
                "value": True,
                "risk": "HIGH",
                "message": "Activity hampers development of low-carbon alternatives",
            })
            risk_score += 3
        else:
            factors.append({
                "factor": "hampers_alternatives",
                "value": False,
                "risk": "LOW",
                "message": "Activity does not hamper low-carbon alternatives",
            })

        # Classify overall risk
        if risk_score >= 4:
            risk_level = LockInRisk.HIGH
            lock_in_avoided = False
        elif risk_score >= 2:
            risk_level = LockInRisk.MEDIUM
            lock_in_avoided = True
        else:
            risk_level = LockInRisk.LOW
            lock_in_avoided = True

        provenance = self._provenance({
            "type": "lock_in_check",
            "activity_id": activity_id,
            "ts": start.isoformat(),
        })

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            f"Lock-in check for {activity_id}: "
            f"risk={risk_level.value}, avoided={lock_in_avoided} in {elapsed_ms:.1f}ms"
        )

        return LockInResult(
            activity_id=activity_id,
            lock_in_avoided=lock_in_avoided,
            risk_level=risk_level,
            asset_lifetime_years=int(asset_lifetime) if asset_lifetime else None,
            switching_cost_factor=float(switching_cost) if switching_cost else None,
            factors_assessed=factors,
            message=(
                f"Lock-in {'AVOIDED' if lock_in_avoided else 'RISK IDENTIFIED'} "
                f"for {info.activity_name} (risk: {risk_level.value})"
            ),
            provenance_hash=provenance,
        )

    def get_transition_pathway(self, activity_id: str) -> TransitionPathway:
        """
        Retrieve the documented transition pathway for an activity.

        Args:
            activity_id: Transition activity identifier

        Returns:
            TransitionPathway with milestones and alternatives

        Raises:
            ValueError: If activity is not a transition activity
        """
        info = self.activities.get(activity_id)
        if info is None:
            raise ValueError(
                f"Activity {activity_id} is not a recognized transition activity"
            )

        milestones = self._build_milestones(info)
        alternatives = self._identify_alternatives(info)
        timeline = self._estimate_timeline(info)

        provenance = self._provenance({
            "type": "transition_pathway",
            "activity_id": activity_id,
            "ts": datetime.utcnow().isoformat(),
        })

        logger.info(
            f"Transition pathway for {activity_id}: "
            f"{len(milestones)} milestones, {len(alternatives)} alternatives"
        )

        return TransitionPathway(
            activity_id=activity_id,
            activity_name=info.activity_name,
            current_status=info.status,
            sunset_date=info.sunset_date,
            milestones=milestones,
            low_carbon_alternatives=alternatives,
            expected_timeline_years=timeline,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_activities(self) -> Dict[str, TransitionActivityInfo]:
        """Parse TRANSITION_ACTIVITIES into validated objects."""
        result: Dict[str, TransitionActivityInfo] = {}
        for aid, raw in TRANSITION_ACTIVITIES.items():
            benchmarks = [BATBenchmark(**b) for b in raw["bat_benchmarks"]]
            result[aid] = TransitionActivityInfo(
                activity_id=aid,
                activity_name=raw["activity_name"],
                nace_codes=raw["nace_codes"],
                objective=EnvironmentalObjective(raw["objective"]),
                justification=raw["justification"],
                bat_benchmarks=benchmarks,
                sunset_date=raw.get("sunset_date"),
                status=TransitionStatus(raw["status"]),
                da_reference=raw["da_reference"],
                low_carbon_pathway=raw["low_carbon_pathway"],
            )
        return result

    def _build_milestones(self, info: TransitionActivityInfo) -> List[Dict[str, str]]:
        """Generate transition milestones based on pathway description."""
        parts = info.low_carbon_pathway.split(" -> ")
        milestones: List[Dict[str, str]] = []
        for i, step in enumerate(parts):
            milestones.append({
                "phase": str(i + 1),
                "description": step.strip(),
                "status": "CURRENT" if i == 0 else "PLANNED",
            })
        return milestones

    def _identify_alternatives(self, info: TransitionActivityInfo) -> List[str]:
        """Extract low-carbon alternatives from the pathway."""
        parts = info.low_carbon_pathway.split(" -> ")
        # The last items are the ultimate low-carbon alternatives
        return [p.strip() for p in parts[1:]]

    def _estimate_timeline(self, info: TransitionActivityInfo) -> int:
        """Estimate years until low-carbon alternative is fully viable."""
        if info.sunset_date:
            try:
                sunset = datetime.strptime(info.sunset_date, "%Y-%m-%d")
                now = datetime.utcnow()
                delta = (sunset - now).days / 365.25
                return max(1, int(delta))
            except (ValueError, TypeError):
                pass
        # Default estimate if no sunset date
        return 15

    @staticmethod
    def _provenance(data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
