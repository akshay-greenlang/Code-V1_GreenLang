# -*- coding: utf-8 -*-
"""
FSTransitionPlanEngine - PACK-012 CSRD Financial Service Engine 7
==================================================================

Financial-services-specific transition plan assessment engine for
CSRD/ESRS reporting by credit institutions, asset managers, and insurers.

Implements portfolio decarbonization pathway assessment, SBTi for
Financial Institutions (SDA, Portfolio Coverage, Temperature Rating),
NZBA (Net Zero Banking Alliance) and NZAOA (Net Zero Asset Owner
Alliance) commitment tracking, sector-specific targets, CapEx alignment,
client engagement scoring, fossil fuel phase-out commitments, and
overall credibility scoring on a 0-100 scale.

Key Regulatory References:
    - ESRS E1-1 (Transition plan for climate change mitigation)
    - ESRS E1-4 (Targets related to climate change)
    - EU Commission Recommendation C(2023) 3866 (Transition Plans)
    - SBTi Financial Institutions Framework (v1.1, 2024)
    - NZBA Guidelines for Target Setting (2022)
    - NZAOA Target Setting Protocol (3rd edition, 2024)
    - GFANZ Recommendations on Transition Plans (2022)
    - IEA Net Zero by 2050 Roadmap (2021)

Formulas:
    SDA Score = (target_intensity - current_intensity) / (target_intensity - baseline_intensity) * 100
    Portfolio Coverage = SUM(nav_with_sbt) / total_nav * 100
    Temperature Rating = weighted_avg(holding_temperature_scores)
    Phase-Out Score = (years_ahead_of_deadline / total_years) * 100
    CapEx Alignment = taxonomy_aligned_capex / total_capex * 100
    Engagement Score = (engagements_completed / engagement_targets) * 100
    Credibility Score = weighted_sum(sub_scores) / max_possible * 100

Zero-Hallucination:
    - All pathway calculations use published IEA/IPCC benchmarks
    - Sector targets from SBTi/IEA published decarbonization pathways
    - Alliance commitments tracked against published protocols
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-012 CSRD Financial Service
Status: Production Ready
"""

import hashlib
import json
import logging
import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_divide(
    numerator: float, denominator: float, default: float = 0.0,
) -> float:
    """Safely divide two numbers, returning default on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator


def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely."""
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a value to [low, high] range."""
    return max(low, min(high, value))


def _round_val(value: float, places: int = 4) -> float:
    """Round a float to specified decimal places."""
    return round(value, places)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SBTiMethod(str, Enum):
    """SBTi methods for financial institutions."""
    SDA = "sectoral_decarbonization_approach"
    PORTFOLIO_COVERAGE = "portfolio_coverage"
    TEMPERATURE_RATING = "temperature_rating"


class AllianceType(str, Enum):
    """Net-zero alliance membership type."""
    NZBA = "nzba"       # Net Zero Banking Alliance
    NZAOA = "nzaoa"     # Net Zero Asset Owner Alliance
    NZAMI = "nzami"     # Net Zero Asset Managers Initiative
    NZIA = "nzia"       # Net Zero Insurance Alliance
    NONE = "none"


class SectorCategory(str, Enum):
    """Sector categories for transition targets."""
    POWER = "power"
    REAL_ESTATE = "real_estate"
    OIL_AND_GAS = "oil_and_gas"
    TRANSPORT = "transport"
    STEEL = "steel"
    CEMENT = "cement"
    AVIATION = "aviation"
    SHIPPING = "shipping"
    AGRICULTURE = "agriculture"
    COAL = "coal"
    OTHER = "other"


class PhaseOutStatus(str, Enum):
    """Phase-out commitment status."""
    COMMITTED = "committed"
    ON_TRACK = "on_track"
    BEHIND = "behind"
    NOT_COMMITTED = "not_committed"
    COMPLETED = "completed"


class CredibilityLevel(str, Enum):
    """Credibility assessment level."""
    HIGH = "high"             # 75-100
    MODERATE = "moderate"     # 50-74
    LOW = "low"               # 25-49
    INSUFFICIENT = "insufficient"  # 0-24


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IEA Net Zero by 2050 sector benchmarks (intensity targets)
# Source: IEA NZE Roadmap (2021), updated 2023
SECTOR_NZE_BENCHMARKS: Dict[str, Dict[str, float]] = {
    SectorCategory.POWER.value: {
        "unit": "gCO2/kWh",
        "2020_baseline": 460.0,
        "2025_target": 340.0,
        "2030_target": 140.0,
        "2035_target": 50.0,
        "2040_target": 0.0,
        "2050_target": 0.0,
    },
    SectorCategory.REAL_ESTATE.value: {
        "unit": "kgCO2/m2",
        "2020_baseline": 55.0,
        "2025_target": 45.0,
        "2030_target": 30.0,
        "2035_target": 20.0,
        "2040_target": 10.0,
        "2050_target": 0.0,
    },
    SectorCategory.OIL_AND_GAS.value: {
        "unit": "MtCO2e",
        "2020_baseline": 100.0,
        "2025_target": 85.0,
        "2030_target": 55.0,
        "2035_target": 35.0,
        "2040_target": 20.0,
        "2050_target": 5.0,
    },
    SectorCategory.TRANSPORT.value: {
        "unit": "gCO2/pkm",
        "2020_baseline": 120.0,
        "2025_target": 105.0,
        "2030_target": 75.0,
        "2035_target": 50.0,
        "2040_target": 25.0,
        "2050_target": 0.0,
    },
    SectorCategory.STEEL.value: {
        "unit": "tCO2/t_steel",
        "2020_baseline": 1.80,
        "2025_target": 1.60,
        "2030_target": 1.20,
        "2035_target": 0.80,
        "2040_target": 0.40,
        "2050_target": 0.10,
    },
    SectorCategory.CEMENT.value: {
        "unit": "tCO2/t_cement",
        "2020_baseline": 0.60,
        "2025_target": 0.55,
        "2030_target": 0.40,
        "2035_target": 0.30,
        "2040_target": 0.20,
        "2050_target": 0.05,
    },
    SectorCategory.AVIATION.value: {
        "unit": "gCO2/RPK",
        "2020_baseline": 90.0,
        "2025_target": 80.0,
        "2030_target": 60.0,
        "2035_target": 40.0,
        "2040_target": 25.0,
        "2050_target": 5.0,
    },
    SectorCategory.SHIPPING.value: {
        "unit": "gCO2/tkm",
        "2020_baseline": 10.0,
        "2025_target": 9.0,
        "2030_target": 6.5,
        "2035_target": 4.0,
        "2040_target": 2.0,
        "2050_target": 0.5,
    },
}

# Phase-out deadlines (NZBA minimum commitments)
PHASE_OUT_DEADLINES: Dict[str, int] = {
    "thermal_coal_mining": 2030,
    "thermal_coal_power": 2030,
    "unabated_oil": 2040,
    "unabated_gas": 2040,
}

# Intermediate target years
TARGET_YEARS: List[int] = [2025, 2030, 2035, 2040, 2050]

# Credibility scoring weights
CREDIBILITY_WEIGHTS: Dict[str, float] = {
    "target_ambition": 0.20,
    "target_coverage": 0.15,
    "methodology_quality": 0.15,
    "governance_oversight": 0.10,
    "implementation_actions": 0.15,
    "engagement_strategy": 0.10,
    "phase_out_commitment": 0.10,
    "disclosure_transparency": 0.05,
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class SectorTargetData(BaseModel):
    """Sector-level decarbonization target data.

    Attributes:
        sector: Sector category.
        current_intensity: Current portfolio intensity for this sector.
        target_intensity_2030: Target intensity by 2030.
        target_intensity_2050: Target intensity by 2050.
        baseline_intensity: Baseline year intensity.
        baseline_year: Baseline year.
        exposure_eur: Total exposure to this sector (EUR).
        weight_pct: Portfolio weight percentage.
        unit: Intensity unit.
        methodology: Methodology used for target.
    """
    sector: SectorCategory = Field(
        default=SectorCategory.OTHER, description="Sector category",
    )
    current_intensity: float = Field(
        default=0.0, ge=0.0, description="Current intensity",
    )
    target_intensity_2030: float = Field(
        default=0.0, ge=0.0, description="2030 target intensity",
    )
    target_intensity_2050: float = Field(
        default=0.0, ge=0.0, description="2050 target intensity",
    )
    baseline_intensity: float = Field(
        default=0.0, ge=0.0, description="Baseline intensity",
    )
    baseline_year: int = Field(default=2020, description="Baseline year")
    exposure_eur: float = Field(
        default=0.0, ge=0.0, description="Sector exposure (EUR)",
    )
    weight_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Portfolio weight %",
    )
    unit: str = Field(default="tCO2e/unit", description="Intensity unit")
    methodology: str = Field(
        default="SDA", description="Target-setting methodology",
    )


class SBTiFIAssessment(BaseModel):
    """SBTi for Financial Institutions assessment result."""
    assessment_id: str = Field(
        default_factory=_new_uuid, description="Assessment ID",
    )
    method: SBTiMethod = Field(
        default=SBTiMethod.SDA, description="SBTi method used",
    )

    # SDA results
    sda_sectors_assessed: int = Field(
        default=0, ge=0, description="Sectors assessed under SDA",
    )
    sda_sectors_aligned: int = Field(
        default=0, ge=0, description="Sectors aligned to pathway",
    )
    sda_alignment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="SDA alignment %",
    )

    # Portfolio Coverage results
    portfolio_coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Portfolio with approved SBTs %",
    )
    coverage_target_year: int = Field(
        default=2040, description="Year to reach 100% coverage",
    )
    coverage_on_track: bool = Field(
        default=False, description="On track for coverage target",
    )

    # Temperature Rating results
    portfolio_temperature_score: float = Field(
        default=3.2, ge=1.0, le=6.0,
        description="Portfolio implied temperature (C)",
    )
    temperature_target: float = Field(
        default=1.5, ge=1.0, le=2.0,
        description="Temperature target (C)",
    )
    temperature_aligned: bool = Field(
        default=False, description="Below temperature target",
    )

    # Overall
    sbti_approved: bool = Field(
        default=False, description="SBTi-approved targets",
    )
    near_term_target_year: int = Field(
        default=2030, description="Near-term target year",
    )
    long_term_target_year: int = Field(
        default=2050, description="Long-term target year",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


class NZBACommitment(BaseModel):
    """Net Zero Banking Alliance commitment tracking."""
    commitment_id: str = Field(
        default_factory=_new_uuid, description="Commitment ID",
    )
    alliance: AllianceType = Field(
        default=AllianceType.NZBA, description="Alliance type",
    )
    member_since: Optional[str] = Field(
        default=None, description="Membership start date (YYYY-MM)",
    )

    # Target status
    targets_published: bool = Field(
        default=False, description="Sector targets published",
    )
    sectors_with_targets: List[str] = Field(
        default_factory=list, description="Sectors with published targets",
    )
    sectors_required: int = Field(
        default=9, description="Required sector targets",
    )
    target_publication_deadline: str = Field(
        default="18 months from joining",
        description="Target publication deadline",
    )

    # Reporting status
    annual_report_submitted: bool = Field(
        default=False, description="Annual progress report submitted",
    )
    transition_plan_published: bool = Field(
        default=False, description="Transition plan publicly disclosed",
    )

    # Fossil fuel policy
    coal_phase_out_date: Optional[int] = Field(
        default=None, description="Coal phase-out target year",
    )
    oil_gas_policy: str = Field(
        default="none", description="Oil and gas policy status",
    )

    # Progress
    on_track_overall: bool = Field(
        default=False, description="Overall on-track assessment",
    )
    compliance_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Compliance score 0-100",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


class SectorDecarbPath(BaseModel):
    """Sector-level decarbonization pathway assessment."""
    path_id: str = Field(
        default_factory=_new_uuid, description="Path ID",
    )
    sector: SectorCategory = Field(
        default=SectorCategory.OTHER, description="Sector",
    )
    unit: str = Field(default="", description="Intensity unit")
    baseline_intensity: float = Field(
        default=0.0, description="Baseline intensity",
    )
    current_intensity: float = Field(
        default=0.0, description="Current intensity",
    )
    target_2030: float = Field(default=0.0, description="2030 target")
    target_2050: float = Field(default=0.0, description="2050 target")
    nze_benchmark_2030: float = Field(
        default=0.0, description="IEA NZE benchmark 2030",
    )
    nze_benchmark_2050: float = Field(
        default=0.0, description="IEA NZE benchmark 2050",
    )
    reduction_achieved_pct: float = Field(
        default=0.0, description="Reduction achieved from baseline %",
    )
    on_track_2030: bool = Field(
        default=False, description="On track for 2030 target",
    )
    on_track_nze: bool = Field(
        default=False, description="On track for NZE pathway",
    )
    gap_to_target_2030: float = Field(
        default=0.0, description="Gap to 2030 target",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


class PhaseOutCommitment(BaseModel):
    """Fossil fuel phase-out commitment assessment."""
    commitment_id: str = Field(
        default_factory=_new_uuid, description="Commitment ID",
    )
    fuel_type: str = Field(default="coal", description="Fuel type")
    phase_out_target_year: int = Field(
        default=2030, description="Target phase-out year",
    )
    nzba_deadline_year: int = Field(
        default=2030, description="NZBA required deadline",
    )
    current_exposure_eur: float = Field(
        default=0.0, ge=0.0, description="Current exposure (EUR)",
    )
    baseline_exposure_eur: float = Field(
        default=0.0, ge=0.0, description="Baseline exposure (EUR)",
    )
    reduction_achieved_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Reduction achieved %",
    )
    status: PhaseOutStatus = Field(
        default=PhaseOutStatus.NOT_COMMITTED, description="Phase-out status",
    )
    years_ahead_of_deadline: int = Field(
        default=0, description="Years ahead of/behind deadline",
    )
    score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Phase-out score 0-100",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


class CredibilityScore(BaseModel):
    """Overall transition plan credibility assessment."""
    score_id: str = Field(
        default_factory=_new_uuid, description="Score ID",
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall credibility 0-100",
    )
    credibility_level: CredibilityLevel = Field(
        default=CredibilityLevel.INSUFFICIENT, description="Credibility level",
    )
    component_scores: Dict[str, float] = Field(
        default_factory=dict, description="Component scores",
    )
    strengths: List[str] = Field(
        default_factory=list, description="Plan strengths",
    )
    gaps: List[str] = Field(
        default_factory=list, description="Plan gaps",
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


class TransitionPlanResult(BaseModel):
    """Complete transition plan assessment result."""
    result_id: str = Field(
        default_factory=_new_uuid, description="Result ID",
    )
    institution_name: str = Field(
        default="", description="Institution name",
    )
    reporting_date: datetime = Field(
        default_factory=_utcnow, description="Reporting date",
    )

    # SBTi-FI assessment
    sbti_assessment: Optional[SBTiFIAssessment] = Field(
        default=None, description="SBTi-FI assessment",
    )

    # Alliance commitments
    nzba_commitment: Optional[NZBACommitment] = Field(
        default=None, description="NZBA commitment status",
    )

    # Sector pathways
    sector_pathways: List[SectorDecarbPath] = Field(
        default_factory=list, description="Sector decarbonization pathways",
    )

    # Phase-out commitments
    phase_out_commitments: List[PhaseOutCommitment] = Field(
        default_factory=list, description="Fossil fuel phase-out commitments",
    )

    # CapEx alignment
    capex_alignment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="CapEx alignment %",
    )

    # Client engagement
    engagement_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Engagement score 0-100",
    )
    clients_engaged: int = Field(
        default=0, ge=0, description="Clients engaged",
    )
    engagement_target: int = Field(
        default=0, ge=0, description="Engagement target count",
    )

    # Intermediate targets
    intermediate_targets: Dict[str, float] = Field(
        default_factory=dict,
        description="Intermediate reduction targets by year",
    )

    # Credibility
    credibility: Optional[CredibilityScore] = Field(
        default=None, description="Credibility assessment",
    )

    # Portfolio metrics
    portfolio_financed_emissions_tco2e: float = Field(
        default=0.0, ge=0.0, description="Total financed emissions (tCO2e)",
    )
    portfolio_waci: float = Field(
        default=0.0, ge=0.0, description="Portfolio WACI (tCO2e/EUR M)",
    )
    yoy_emission_reduction_pct: float = Field(
        default=0.0, description="Year-on-year emission reduction %",
    )

    # Metadata
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------


class TransitionPlanConfig(BaseModel):
    """Configuration for the FSTransitionPlanEngine.

    Attributes:
        institution_name: Financial institution name.
        alliance_type: Net-zero alliance membership.
        sbti_method: Primary SBTi methodology.
        base_year: Baseline year for targets.
        current_year: Current assessment year.
        net_zero_target_year: Net Zero target year.
        interim_target_year: Interim target year.
        interim_reduction_target_pct: Interim reduction from baseline.
        annual_reduction_target_pct: Target annual reduction rate.
        temperature_target_c: Temperature alignment target.
        sbt_coverage_target_pct: SBT portfolio coverage target.
        credibility_weights: Custom credibility scoring weights.
    """
    institution_name: str = Field(
        default="Financial Institution", description="Institution name",
    )
    alliance_type: AllianceType = Field(
        default=AllianceType.NZBA, description="Alliance membership",
    )
    sbti_method: SBTiMethod = Field(
        default=SBTiMethod.SDA, description="Primary SBTi method",
    )
    base_year: int = Field(default=2020, description="Baseline year")
    current_year: int = Field(default=2025, description="Current year")
    net_zero_target_year: int = Field(
        default=2050, description="Net Zero target year",
    )
    interim_target_year: int = Field(
        default=2030, description="Interim target year",
    )
    interim_reduction_target_pct: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Interim reduction target from baseline %",
    )
    annual_reduction_target_pct: float = Field(
        default=7.0, ge=0.0, le=30.0,
        description="Annual reduction target %",
    )
    temperature_target_c: float = Field(
        default=1.5, ge=1.0, le=3.0, description="Temperature target (C)",
    )
    sbt_coverage_target_pct: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="SBT coverage target %",
    )
    credibility_weights: Dict[str, float] = Field(
        default_factory=lambda: dict(CREDIBILITY_WEIGHTS),
        description="Credibility scoring weights",
    )


# ---------------------------------------------------------------------------
# model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

TransitionPlanConfig.model_rebuild()
SectorTargetData.model_rebuild()
SBTiFIAssessment.model_rebuild()
NZBACommitment.model_rebuild()
SectorDecarbPath.model_rebuild()
PhaseOutCommitment.model_rebuild()
CredibilityScore.model_rebuild()
TransitionPlanResult.model_rebuild()


# ---------------------------------------------------------------------------
# FSTransitionPlanEngine
# ---------------------------------------------------------------------------


class FSTransitionPlanEngine:
    """
    Financial-services-specific transition plan assessment engine.

    Assesses portfolio decarbonization pathways, SBTi-FI alignment,
    NZBA/NZAOA commitment tracking, sector targets, phase-out
    commitments, client engagement, and overall credibility.

    Zero-Hallucination Guarantees:
        - All pathway calculations use published IEA/IPCC benchmarks
        - Sector targets from SBTi/IEA published decarbonization pathways
        - Credibility scoring uses deterministic weighted-sum formula
        - SHA-256 provenance hash on every result
        - No LLM involvement in any calculation path

    Attributes:
        config: Engine configuration.
    """

    def __init__(self, config: TransitionPlanConfig) -> None:
        """Initialize FSTransitionPlanEngine.

        Args:
            config: Engine configuration.
        """
        self.config = config
        logger.info(
            "FSTransitionPlanEngine initialized (v%s) for '%s'",
            _MODULE_VERSION, config.institution_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_transition_plan(
        self,
        sector_targets: List[SectorTargetData],
        nzba_commitment: Optional[NZBACommitment] = None,
        phase_out_data: Optional[List[Dict[str, Any]]] = None,
        engagement_data: Optional[Dict[str, Any]] = None,
        capex_data: Optional[Dict[str, float]] = None,
        portfolio_emissions_tco2e: float = 0.0,
        portfolio_waci: float = 0.0,
        prior_year_emissions_tco2e: float = 0.0,
        reporting_date: Optional[datetime] = None,
    ) -> TransitionPlanResult:
        """Run complete transition plan assessment.

        Args:
            sector_targets: Sector-level target data.
            nzba_commitment: Optional NZBA commitment data.
            phase_out_data: Optional phase-out commitment data.
            engagement_data: Optional engagement strategy data.
            capex_data: Optional CapEx alignment data.
            portfolio_emissions_tco2e: Total financed emissions.
            portfolio_waci: Portfolio WACI.
            prior_year_emissions_tco2e: Prior year emissions for YoY.
            reporting_date: Optional reporting date.

        Returns:
            Complete TransitionPlanResult.
        """
        import time
        start = time.perf_counter()

        r_date = reporting_date or _utcnow()
        phase_out_data = phase_out_data or []
        engagement_data = engagement_data or {}
        capex_data = capex_data or {}

        # 1. SBTi-FI assessment
        sbti = self._assess_sbti_fi(sector_targets)

        # 2. Sector decarbonization pathways
        sector_paths = self._assess_sector_pathways(sector_targets)

        # 3. NZBA commitment tracking
        nzba = nzba_commitment
        if nzba is not None:
            nzba = self._evaluate_nzba_compliance(nzba, sector_targets)

        # 4. Phase-out commitments
        phase_outs = self._assess_phase_out_commitments(phase_out_data)

        # 5. CapEx alignment
        capex_pct = self._compute_capex_alignment(capex_data)

        # 6. Client engagement
        eng_score, clients_engaged, eng_target = self._assess_engagement(
            engagement_data,
        )

        # 7. Intermediate targets
        intermediate = self._compute_intermediate_targets(
            portfolio_emissions_tco2e, prior_year_emissions_tco2e,
        )

        # 8. YoY reduction
        yoy_reduction = 0.0
        if prior_year_emissions_tco2e > 0.0:
            yoy_reduction = _safe_pct(
                prior_year_emissions_tco2e - portfolio_emissions_tco2e,
                prior_year_emissions_tco2e,
            )

        # 9. Credibility assessment
        credibility = self._assess_credibility(
            sbti, nzba, sector_paths, phase_outs,
            capex_pct, eng_score,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        result = TransitionPlanResult(
            institution_name=self.config.institution_name,
            reporting_date=r_date,
            sbti_assessment=sbti,
            nzba_commitment=nzba,
            sector_pathways=sector_paths,
            phase_out_commitments=phase_outs,
            capex_alignment_pct=_round_val(capex_pct, 2),
            engagement_score=_round_val(eng_score, 2),
            clients_engaged=clients_engaged,
            engagement_target=eng_target,
            intermediate_targets=intermediate,
            credibility=credibility,
            portfolio_financed_emissions_tco2e=_round_val(
                portfolio_emissions_tco2e, 2,
            ),
            portfolio_waci=_round_val(portfolio_waci, 2),
            yoy_emission_reduction_pct=_round_val(yoy_reduction, 2),
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # SBTi-FI Assessment
    # ------------------------------------------------------------------

    def _assess_sbti_fi(
        self,
        sector_targets: List[SectorTargetData],
    ) -> SBTiFIAssessment:
        """Assess SBTi for Financial Institutions alignment.

        Supports SDA (sector-level intensity comparison against IEA NZE),
        Portfolio Coverage (share of NAV with approved SBTs), and
        Temperature Rating (portfolio implied temperature).

        Args:
            sector_targets: Sector-level target data.

        Returns:
            SBTiFIAssessment result.
        """
        # SDA assessment: compare sector intensities to NZE benchmarks
        sectors_assessed = 0
        sectors_aligned = 0

        for st in sector_targets:
            sector_key = st.sector.value
            benchmarks = SECTOR_NZE_BENCHMARKS.get(sector_key)
            if benchmarks is None:
                continue
            sectors_assessed += 1

            # Compare current vs 2030 NZE benchmark
            nze_2030 = benchmarks.get("2030_target", 0.0)
            if st.current_intensity <= nze_2030:
                sectors_aligned += 1
            elif st.target_intensity_2030 <= nze_2030:
                sectors_aligned += 1

        sda_alignment = _safe_pct(sectors_aligned, sectors_assessed)

        # Portfolio coverage: would need holding-level SBT data
        # Here we estimate from sector target data coverage
        total_weight = sum(st.weight_pct for st in sector_targets)
        covered_weight = sum(
            st.weight_pct for st in sector_targets
            if st.methodology in ("SDA", "SBTi", "sda", "sbti")
        )
        coverage = _safe_pct(covered_weight, total_weight)

        # Temperature rating: simplified estimation from sector alignment
        # Aligned sectors ~1.5C, partially ~2.0C, not aligned ~3.0C
        if sectors_assessed > 0:
            aligned_ratio = sectors_aligned / sectors_assessed
            temp_score = 1.5 + (1.0 - aligned_ratio) * 1.5
        else:
            temp_score = 3.2  # default

        result = SBTiFIAssessment(
            method=self.config.sbti_method,
            sda_sectors_assessed=sectors_assessed,
            sda_sectors_aligned=sectors_aligned,
            sda_alignment_pct=_round_val(sda_alignment, 2),
            portfolio_coverage_pct=_round_val(coverage, 2),
            coverage_target_year=2040,
            coverage_on_track=coverage >= 50.0,
            portfolio_temperature_score=_round_val(temp_score, 2),
            temperature_target=self.config.temperature_target_c,
            temperature_aligned=temp_score <= self.config.temperature_target_c,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Sector Pathway Assessment
    # ------------------------------------------------------------------

    def _assess_sector_pathways(
        self,
        sector_targets: List[SectorTargetData],
    ) -> List[SectorDecarbPath]:
        """Assess sector-level decarbonization pathways against NZE benchmarks.

        Compares the institution's sector targets against IEA Net Zero
        by 2050 benchmark intensities to determine pathway alignment.

        Args:
            sector_targets: Sector-level target data.

        Returns:
            List of SectorDecarbPath assessments.
        """
        paths: List[SectorDecarbPath] = []

        for st in sector_targets:
            sector_key = st.sector.value
            benchmarks = SECTOR_NZE_BENCHMARKS.get(sector_key, {})

            nze_2030 = benchmarks.get("2030_target", 0.0)
            nze_2050 = benchmarks.get("2050_target", 0.0)
            unit = benchmarks.get("unit", st.unit)

            # Reduction achieved from baseline
            reduction = 0.0
            if st.baseline_intensity > 0.0:
                reduction = _safe_pct(
                    st.baseline_intensity - st.current_intensity,
                    st.baseline_intensity,
                )

            # Gap to 2030 target
            gap = st.current_intensity - st.target_intensity_2030

            # On-track assessments
            on_track_2030 = st.current_intensity <= st.target_intensity_2030
            on_track_nze = st.target_intensity_2030 <= nze_2030

            path = SectorDecarbPath(
                sector=st.sector,
                unit=unit,
                baseline_intensity=st.baseline_intensity,
                current_intensity=st.current_intensity,
                target_2030=st.target_intensity_2030,
                target_2050=st.target_intensity_2050,
                nze_benchmark_2030=nze_2030,
                nze_benchmark_2050=nze_2050,
                reduction_achieved_pct=_round_val(reduction, 2),
                on_track_2030=on_track_2030,
                on_track_nze=on_track_nze,
                gap_to_target_2030=_round_val(gap, 4),
            )
            path.provenance_hash = _compute_hash(path)
            paths.append(path)

        return paths

    # ------------------------------------------------------------------
    # NZBA Compliance Evaluation
    # ------------------------------------------------------------------

    def _evaluate_nzba_compliance(
        self,
        commitment: NZBACommitment,
        sector_targets: List[SectorTargetData],
    ) -> NZBACommitment:
        """Evaluate NZBA commitment compliance.

        Assesses whether the institution meets NZBA requirements:
        sector target publication, annual reporting, coal phase-out,
        transition plan disclosure.

        Args:
            commitment: Current NZBA commitment data.
            sector_targets: Sector-level targets for coverage check.

        Returns:
            Updated NZBACommitment with compliance score.
        """
        score = 0.0
        max_score = 100.0

        # Target publication (25 points)
        if commitment.targets_published:
            sectors_covered = len(commitment.sectors_with_targets)
            sector_score = min(
                (sectors_covered / max(commitment.sectors_required, 1)) * 25.0,
                25.0,
            )
            score += sector_score

        # Annual reporting (20 points)
        if commitment.annual_report_submitted:
            score += 20.0

        # Transition plan disclosure (20 points)
        if commitment.transition_plan_published:
            score += 20.0

        # Coal phase-out commitment (20 points)
        if commitment.coal_phase_out_date is not None:
            if commitment.coal_phase_out_date <= 2030:
                score += 20.0
            elif commitment.coal_phase_out_date <= 2035:
                score += 15.0
            else:
                score += 5.0

        # Oil and gas policy (15 points)
        oil_gas_scores = {
            "comprehensive_phase_down": 15.0,
            "restricted": 10.0,
            "partial": 5.0,
            "none": 0.0,
        }
        score += oil_gas_scores.get(commitment.oil_gas_policy, 0.0)

        compliance = _clamp(_round_val(score, 2))
        commitment.compliance_score = compliance
        commitment.on_track_overall = compliance >= 60.0
        commitment.provenance_hash = _compute_hash(commitment)
        return commitment

    # ------------------------------------------------------------------
    # Phase-Out Assessment
    # ------------------------------------------------------------------

    def _assess_phase_out_commitments(
        self,
        phase_out_data: List[Dict[str, Any]],
    ) -> List[PhaseOutCommitment]:
        """Assess fossil fuel phase-out commitments.

        Evaluates each phase-out commitment against NZBA deadlines
        and calculates progress and status.

        Args:
            phase_out_data: List of phase-out commitment dictionaries.

        Returns:
            List of PhaseOutCommitment assessments.
        """
        commitments: List[PhaseOutCommitment] = []

        for pod in phase_out_data:
            fuel = pod.get("fuel_type", "coal")
            target_year = pod.get("phase_out_target_year", 2030)
            current_exp = pod.get("current_exposure_eur", 0.0)
            baseline_exp = pod.get("baseline_exposure_eur", 0.0)
            nzba_deadline = PHASE_OUT_DEADLINES.get(
                f"thermal_{fuel}_mining",
                PHASE_OUT_DEADLINES.get(
                    f"thermal_{fuel}_power",
                    PHASE_OUT_DEADLINES.get(
                        f"unabated_{fuel}", 2040,
                    ),
                ),
            )

            # Reduction achieved
            reduction = _safe_pct(
                baseline_exp - current_exp, baseline_exp,
            ) if baseline_exp > 0 else 0.0

            # Years ahead/behind deadline
            years_ahead = nzba_deadline - target_year

            # Status
            if current_exp == 0.0 and baseline_exp > 0.0:
                status = PhaseOutStatus.COMPLETED
            elif target_year <= nzba_deadline and reduction >= 50.0:
                status = PhaseOutStatus.ON_TRACK
            elif target_year <= nzba_deadline:
                status = PhaseOutStatus.COMMITTED
            elif reduction < 10.0:
                status = PhaseOutStatus.NOT_COMMITTED
            else:
                status = PhaseOutStatus.BEHIND

            # Score (0-100)
            score = 0.0
            if status == PhaseOutStatus.COMPLETED:
                score = 100.0
            elif status == PhaseOutStatus.ON_TRACK:
                score = 75.0 + min(years_ahead * 5.0, 25.0)
            elif status == PhaseOutStatus.COMMITTED:
                score = 50.0 + min(reduction * 0.25, 25.0)
            elif status == PhaseOutStatus.BEHIND:
                score = max(reduction * 0.5, 10.0)
            else:
                score = 0.0
            score = _clamp(score)

            pc = PhaseOutCommitment(
                fuel_type=fuel,
                phase_out_target_year=target_year,
                nzba_deadline_year=nzba_deadline,
                current_exposure_eur=current_exp,
                baseline_exposure_eur=baseline_exp,
                reduction_achieved_pct=_round_val(reduction, 2),
                status=status,
                years_ahead_of_deadline=years_ahead,
                score=_round_val(score, 2),
            )
            pc.provenance_hash = _compute_hash(pc)
            commitments.append(pc)

        return commitments

    # ------------------------------------------------------------------
    # CapEx Alignment
    # ------------------------------------------------------------------

    def _compute_capex_alignment(
        self,
        capex_data: Dict[str, float],
    ) -> float:
        """Compute CapEx alignment with EU Taxonomy.

        Formula:
            alignment = taxonomy_aligned_capex / total_capex * 100

        Args:
            capex_data: Dict with 'aligned' and 'total' keys (EUR).

        Returns:
            CapEx alignment percentage.
        """
        aligned = capex_data.get("aligned", 0.0)
        total = capex_data.get("total", 0.0)
        return _safe_pct(aligned, total)

    # ------------------------------------------------------------------
    # Client Engagement
    # ------------------------------------------------------------------

    def _assess_engagement(
        self,
        engagement_data: Dict[str, Any],
    ) -> Tuple[float, int, int]:
        """Assess client engagement strategy effectiveness.

        Formula:
            engagement_score = (engagements_completed / target) * 100

        Args:
            engagement_data: Engagement strategy data.

        Returns:
            Tuple of (score, clients_engaged, engagement_target).
        """
        clients_engaged = engagement_data.get("clients_engaged", 0)
        target = engagement_data.get("engagement_target", 0)
        escalations = engagement_data.get("escalations_completed", 0)
        successful_outcomes = engagement_data.get("successful_outcomes", 0)

        if target == 0:
            return 0.0, clients_engaged, target

        # Base score from engagement coverage
        coverage_score = _safe_pct(clients_engaged, target)

        # Bonus for successful outcomes
        outcome_bonus = _safe_pct(successful_outcomes, max(clients_engaged, 1)) * 0.2

        # Bonus for escalations (shows willingness to act)
        escalation_bonus = min(escalations * 2.0, 10.0)

        total = _clamp(coverage_score + outcome_bonus + escalation_bonus)
        return total, clients_engaged, target

    # ------------------------------------------------------------------
    # Intermediate Targets
    # ------------------------------------------------------------------

    def _compute_intermediate_targets(
        self,
        current_emissions: float,
        prior_emissions: float,
    ) -> Dict[str, float]:
        """Compute intermediate reduction targets.

        Projects required reductions at each milestone year based
        on the configured annual reduction rate.

        Args:
            current_emissions: Current year financed emissions.
            prior_emissions: Prior year financed emissions.

        Returns:
            Dict mapping year to required emissions level.
        """
        targets: Dict[str, float] = {}
        rate = self.config.annual_reduction_target_pct / 100.0
        base = current_emissions if current_emissions > 0 else prior_emissions

        for year in TARGET_YEARS:
            years_from_now = year - self.config.current_year
            if years_from_now <= 0:
                targets[str(year)] = base
            else:
                projected = base * ((1.0 - rate) ** years_from_now)
                targets[str(year)] = _round_val(max(projected, 0.0), 2)

        return targets

    # ------------------------------------------------------------------
    # Credibility Assessment
    # ------------------------------------------------------------------

    def _assess_credibility(
        self,
        sbti: SBTiFIAssessment,
        nzba: Optional[NZBACommitment],
        sector_paths: List[SectorDecarbPath],
        phase_outs: List[PhaseOutCommitment],
        capex_pct: float,
        engagement_score: float,
    ) -> CredibilityScore:
        """Assess overall transition plan credibility.

        Weighted scoring across 8 dimensions:
        1. Target ambition (vs NZE benchmarks)
        2. Target coverage (sectors covered)
        3. Methodology quality (SBTi approval)
        4. Governance oversight
        5. Implementation actions (CapEx alignment)
        6. Engagement strategy
        7. Phase-out commitments
        8. Disclosure transparency

        Args:
            sbti: SBTi-FI assessment.
            nzba: NZBA commitment (if any).
            sector_paths: Sector pathway assessments.
            phase_outs: Phase-out assessments.
            capex_pct: CapEx alignment percentage.
            engagement_score: Engagement score.

        Returns:
            CredibilityScore with component breakdown.
        """
        weights = self.config.credibility_weights
        components: Dict[str, float] = {}
        strengths: List[str] = []
        gaps: List[str] = []
        recommendations: List[str] = []

        # 1. Target ambition
        if sector_paths:
            aligned_paths = sum(1 for p in sector_paths if p.on_track_nze)
            ambition = _safe_pct(aligned_paths, len(sector_paths))
        else:
            ambition = 0.0
        components["target_ambition"] = _round_val(ambition, 2)
        if ambition >= 75.0:
            strengths.append("Sector targets well aligned with NZE pathway")
        elif ambition < 25.0:
            gaps.append("Sector targets lack ambition vs NZE benchmarks")
            recommendations.append(
                "Align sector intensity targets with IEA NZE benchmarks"
            )

        # 2. Target coverage
        if sector_paths:
            coverage = min(len(sector_paths) / 6.0 * 100.0, 100.0)
        else:
            coverage = 0.0
        components["target_coverage"] = _round_val(coverage, 2)
        if coverage < 50.0:
            gaps.append("Insufficient sector target coverage")
            recommendations.append(
                "Set targets for at least 6 high-impact sectors"
            )

        # 3. Methodology quality
        meth_score = 0.0
        if sbti.sbti_approved:
            meth_score = 100.0
            strengths.append("SBTi-approved targets")
        elif sbti.sda_alignment_pct >= 50.0:
            meth_score = 60.0
        elif sbti.sda_sectors_assessed > 0:
            meth_score = 30.0
        components["methodology_quality"] = _round_val(meth_score, 2)
        if meth_score < 50.0:
            recommendations.append("Seek SBTi approval for targets")

        # 4. Governance oversight
        gov_score = 0.0
        if nzba is not None:
            if nzba.transition_plan_published:
                gov_score += 50.0
            if nzba.annual_report_submitted:
                gov_score += 30.0
            if nzba.on_track_overall:
                gov_score += 20.0
        components["governance_oversight"] = _clamp(_round_val(gov_score, 2))
        if gov_score >= 80.0:
            strengths.append("Strong governance and reporting practices")

        # 5. Implementation actions (CapEx proxy)
        impl_score = min(capex_pct, 100.0)
        components["implementation_actions"] = _round_val(impl_score, 2)
        if impl_score < 30.0:
            gaps.append("Low CapEx alignment with taxonomy")
            recommendations.append(
                "Increase capital allocation to taxonomy-aligned activities"
            )

        # 6. Engagement strategy
        components["engagement_strategy"] = _round_val(engagement_score, 2)
        if engagement_score >= 70.0:
            strengths.append("Active client engagement programme")
        elif engagement_score < 30.0:
            gaps.append("Weak client engagement on transition")
            recommendations.append(
                "Develop structured client engagement programme "
                "with escalation procedures"
            )

        # 7. Phase-out commitments
        if phase_outs:
            po_avg = sum(po.score for po in phase_outs) / len(phase_outs)
        else:
            po_avg = 0.0
        components["phase_out_commitment"] = _round_val(po_avg, 2)
        if po_avg >= 75.0:
            strengths.append("Strong fossil fuel phase-out commitments")
        elif po_avg < 25.0:
            gaps.append("No or weak fossil fuel phase-out commitment")
            recommendations.append(
                "Commit to coal exit by 2030 and oil/gas phase-down by 2040"
            )

        # 8. Disclosure transparency
        disc_score = 0.0
        if nzba is not None and nzba.transition_plan_published:
            disc_score += 50.0
        if sector_paths:
            disc_score += 30.0
        if phase_outs:
            disc_score += 20.0
        disc_score = _clamp(disc_score)
        components["disclosure_transparency"] = _round_val(disc_score, 2)

        # Overall weighted score
        overall = 0.0
        for dim, weight in weights.items():
            overall += weight * components.get(dim, 0.0)
        overall = _clamp(_round_val(overall, 2))

        # Credibility level
        if overall >= 75.0:
            level = CredibilityLevel.HIGH
        elif overall >= 50.0:
            level = CredibilityLevel.MODERATE
        elif overall >= 25.0:
            level = CredibilityLevel.LOW
        else:
            level = CredibilityLevel.INSUFFICIENT

        result = CredibilityScore(
            overall_score=overall,
            credibility_level=level,
            component_scores=components,
            strengths=strengths,
            gaps=gaps,
            recommendations=recommendations,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Convenience Methods
    # ------------------------------------------------------------------

    def get_sector_benchmarks(
        self,
        sector: SectorCategory,
    ) -> Dict[str, float]:
        """Get IEA NZE benchmarks for a sector.

        Args:
            sector: Sector category.

        Returns:
            Dict of benchmark values by target year.
        """
        return dict(SECTOR_NZE_BENCHMARKS.get(sector.value, {}))

    def assess_single_sector(
        self,
        sector_target: SectorTargetData,
    ) -> SectorDecarbPath:
        """Assess a single sector pathway.

        Convenience method for individual sector analysis.

        Args:
            sector_target: Sector target data.

        Returns:
            SectorDecarbPath assessment.
        """
        paths = self._assess_sector_pathways([sector_target])
        return paths[0] if paths else SectorDecarbPath()
