# -*- coding: utf-8 -*-
"""
InvestmentUniverseEngine - PACK-011 SFDR Article 9 Engine 8
==============================================================

Investment universe screening engine for SFDR Article 9 products.

Article 9 products must maintain a rigorously screened investment universe
that excludes companies failing sustainability criteria.  This engine
implements multi-layer screening (PAB/CTB exclusions, ESG norm-based,
sector-based, controversy-based), manages a watch list for borderline
holdings, supports pre-approval workflows for new investments, and
tracks universe coverage statistics.

Key Features:
    - PAB/CTB exclusion screening per EU Regulation 2019/2089
    - Multi-layer screening (exclusion, ESG norms, sector, controversy)
    - Watch list management for near-threshold holdings
    - Pre-approval workflow for new investment candidates
    - Universe coverage tracking (eligible vs total investable)
    - Configurable exclusion thresholds with override support

Key Regulatory References:
    - Regulation (EU) 2019/2088 (SFDR) Article 9
    - Regulation (EU) 2019/2089 (Low Carbon Benchmarks) Article 3, 12
    - EU PAB / CTB exclusion criteria
    - UN Global Compact principles
    - OECD Guidelines for Multinational Enterprises

Formulas:
    Exclusion Rate = excluded_count / total_screened * 100
    Universe Coverage = eligible_nav / total_universe_nav * 100
    Watch List Rate = watch_count / total_screened * 100
    Threshold Distance = (threshold - actual_value) / threshold * 100

Zero-Hallucination:
    - All exclusion checks use deterministic threshold comparisons
    - Multi-layer screening applies boolean rule evaluation
    - Watch list proximity calculated by pure arithmetic
    - SHA-256 provenance hash on every result
    - No LLM involvement in any screening decision path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Status: Production Ready
"""

import hashlib
import json
import logging
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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely, returning 0.0 on zero denominator."""
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0


def _round_val(value: float, places: int = 4) -> float:
    """Round a float to specified decimal places."""
    return round(value, places)


def _safe_divide(
    numerator: float, denominator: float, default: float = 0.0,
) -> float:
    """Safely divide two numbers, returning default on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ScreeningLayer(str, Enum):
    """Screening layer classification for multi-layer approach."""
    PAB_EXCLUSION = "pab_exclusion"
    CTB_EXCLUSION = "ctb_exclusion"
    NORM_BASED = "norm_based"
    SECTOR_BASED = "sector_based"
    CONTROVERSY_BASED = "controversy_based"
    ESG_QUALITY = "esg_quality"
    CUSTOM = "custom"


class ExclusionType(str, Enum):
    """Type of exclusion applied to a security."""
    CONTROVERSIAL_WEAPONS = "controversial_weapons"
    FOSSIL_FUEL_COAL = "fossil_fuel_coal"
    FOSSIL_FUEL_OIL_GAS = "fossil_fuel_oil_gas"
    FOSSIL_FUEL_REFINING = "fossil_fuel_refining"
    FOSSIL_FUEL_DISTRIBUTION = "fossil_fuel_distribution"
    HIGH_CARBON_POWER = "high_carbon_power"
    TOBACCO = "tobacco"
    UNGC_VIOLATIONS = "ungc_violations"
    SEVERE_CONTROVERSY = "severe_controversy"
    THERMAL_COAL_MINING = "thermal_coal_mining"
    OIL_SANDS = "oil_sands"
    ARCTIC_DRILLING = "arctic_drilling"
    DEFORESTATION = "deforestation"
    HUMAN_RIGHTS = "human_rights"


# ---------------------------------------------------------------------------
# Constants: Exclusion Thresholds
# ---------------------------------------------------------------------------

PAB_EXCLUSION_RULES: Dict[str, Dict[str, Any]] = {
    ExclusionType.CONTROVERSIAL_WEAPONS.value: {
        "field": "controversial_weapons",
        "type": "boolean_true",
        "threshold": 0.0,
        "description": "Involved in controversial weapons "
                       "(anti-personnel mines, cluster munitions, "
                       "chemical/biological weapons)",
        "layer": ScreeningLayer.PAB_EXCLUSION,
    },
    ExclusionType.FOSSIL_FUEL_COAL.value: {
        "field": "coal_revenue_pct",
        "type": "gte",
        "threshold": 1.0,
        "description": ">=1% revenue from coal exploration/processing",
        "layer": ScreeningLayer.PAB_EXCLUSION,
    },
    ExclusionType.FOSSIL_FUEL_OIL_GAS.value: {
        "field": "oil_gas_revenue_pct",
        "type": "gte",
        "threshold": 10.0,
        "description": ">=10% revenue from oil/gas exploration/processing",
        "layer": ScreeningLayer.PAB_EXCLUSION,
    },
    ExclusionType.FOSSIL_FUEL_REFINING.value: {
        "field": "refining_revenue_pct",
        "type": "gte",
        "threshold": 10.0,
        "description": ">=10% revenue from fossil fuel refining",
        "layer": ScreeningLayer.PAB_EXCLUSION,
    },
    ExclusionType.FOSSIL_FUEL_DISTRIBUTION.value: {
        "field": "distribution_revenue_pct",
        "type": "gte",
        "threshold": 50.0,
        "description": ">=50% revenue from fossil fuel distribution",
        "layer": ScreeningLayer.PAB_EXCLUSION,
    },
    ExclusionType.HIGH_CARBON_POWER.value: {
        "field": "power_carbon_intensity",
        "type": "gt",
        "threshold": 100.0,
        "description": ">100g CO2/kWh power generation",
        "layer": ScreeningLayer.PAB_EXCLUSION,
    },
}

CTB_EXCLUSION_RULES: Dict[str, Dict[str, Any]] = {
    ExclusionType.CONTROVERSIAL_WEAPONS.value: {
        "field": "controversial_weapons",
        "type": "boolean_true",
        "threshold": 0.0,
        "description": "Involved in controversial weapons",
        "layer": ScreeningLayer.CTB_EXCLUSION,
    },
}

NORM_BASED_RULES: Dict[str, Dict[str, Any]] = {
    ExclusionType.UNGC_VIOLATIONS.value: {
        "field": "ungc_violations",
        "type": "boolean_true",
        "threshold": 0.0,
        "description": "Violations of UN Global Compact principles",
        "layer": ScreeningLayer.NORM_BASED,
    },
    ExclusionType.HUMAN_RIGHTS.value: {
        "field": "human_rights_violations",
        "type": "boolean_true",
        "threshold": 0.0,
        "description": "Severe human rights violations",
        "layer": ScreeningLayer.NORM_BASED,
    },
}

SECTOR_BASED_RULES: Dict[str, Dict[str, Any]] = {
    ExclusionType.TOBACCO.value: {
        "field": "tobacco_revenue_pct",
        "type": "gte",
        "threshold": 5.0,
        "description": ">=5% revenue from tobacco",
        "layer": ScreeningLayer.SECTOR_BASED,
    },
    ExclusionType.THERMAL_COAL_MINING.value: {
        "field": "thermal_coal_mining_revenue_pct",
        "type": "gte",
        "threshold": 5.0,
        "description": ">=5% revenue from thermal coal mining",
        "layer": ScreeningLayer.SECTOR_BASED,
    },
    ExclusionType.OIL_SANDS.value: {
        "field": "oil_sands_revenue_pct",
        "type": "gte",
        "threshold": 5.0,
        "description": ">=5% revenue from oil sands extraction",
        "layer": ScreeningLayer.SECTOR_BASED,
    },
    ExclusionType.ARCTIC_DRILLING.value: {
        "field": "arctic_drilling_revenue_pct",
        "type": "gte",
        "threshold": 5.0,
        "description": ">=5% revenue from Arctic drilling",
        "layer": ScreeningLayer.SECTOR_BASED,
    },
    ExclusionType.DEFORESTATION.value: {
        "field": "deforestation_linked",
        "type": "boolean_true",
        "threshold": 0.0,
        "description": "Linked to deforestation activities",
        "layer": ScreeningLayer.SECTOR_BASED,
    },
}

CONTROVERSY_RULES: Dict[str, Dict[str, Any]] = {
    ExclusionType.SEVERE_CONTROVERSY.value: {
        "field": "controversy_score",
        "type": "gte",
        "threshold": 5.0,
        "description": "Controversy score >= 5 (severe)",
        "layer": ScreeningLayer.CONTROVERSY_BASED,
    },
}

# Watch list proximity threshold (% from exclusion threshold)
WATCH_LIST_PROXIMITY_PCT: float = 20.0


# ---------------------------------------------------------------------------
# Pydantic Data Models
# ---------------------------------------------------------------------------


class SecurityData(BaseModel):
    """Input data for a single security to screen.

    Contains identification, financial data, and all fields needed
    for multi-layer exclusion screening.

    Attributes:
        security_id: Unique security identifier.
        company_name: Investee company name.
        isin: ISIN code.
        sector: NACE/GICS sector code.
        country: Country of domicile.
        market_cap_eur: Market capitalization (EUR).
        nav_value: Current position value (EUR, 0 if not held).
        weight_pct: Current portfolio weight (%, 0 if not held).
        controversial_weapons: Controversial weapons involvement.
        coal_revenue_pct: Revenue from coal activities (%).
        oil_gas_revenue_pct: Revenue from oil/gas activities (%).
        refining_revenue_pct: Revenue from fossil fuel refining (%).
        distribution_revenue_pct: Revenue from fossil fuel distribution (%).
        power_carbon_intensity: Power generation intensity (gCO2/kWh).
        tobacco_revenue_pct: Revenue from tobacco (%).
        thermal_coal_mining_revenue_pct: Revenue from thermal coal mining (%).
        oil_sands_revenue_pct: Revenue from oil sands (%).
        arctic_drilling_revenue_pct: Revenue from Arctic drilling (%).
        deforestation_linked: Linked to deforestation.
        ungc_violations: UNGC violations.
        human_rights_violations: Severe human rights violations.
        controversy_score: Controversy severity score (0-10).
        esg_score: ESG quality score (0-100).
        carbon_intensity: Carbon intensity (tCO2e/EUR M revenue).
        has_transition_plan: Has credible transition plan.
        data_coverage: Data coverage quality indicator.
    """
    security_id: str = Field(
        default_factory=_new_uuid, description="Unique security ID",
    )
    company_name: str = Field(
        default="", description="Investee company name",
    )
    isin: str = Field(default="", description="ISIN code")
    sector: str = Field(default="", description="NACE/GICS sector code")
    country: str = Field(default="", description="Country (ISO 3166)")
    market_cap_eur: float = Field(
        default=0.0, ge=0.0, description="Market cap (EUR)",
    )
    nav_value: float = Field(
        default=0.0, ge=0.0,
        description="Current position value (EUR, 0 if not held)",
    )
    weight_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Current portfolio weight (%)",
    )

    # PAB exclusion fields
    controversial_weapons: bool = Field(
        default=False, description="Controversial weapons involvement",
    )
    coal_revenue_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Revenue from coal activities (%)",
    )
    oil_gas_revenue_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Revenue from oil/gas activities (%)",
    )
    refining_revenue_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Revenue from fossil fuel refining (%)",
    )
    distribution_revenue_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Revenue from fossil fuel distribution (%)",
    )
    power_carbon_intensity: float = Field(
        default=0.0, ge=0.0,
        description="Power generation intensity (gCO2/kWh)",
    )

    # Sector exclusion fields
    tobacco_revenue_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Revenue from tobacco (%)",
    )
    thermal_coal_mining_revenue_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Revenue from thermal coal mining (%)",
    )
    oil_sands_revenue_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Revenue from oil sands (%)",
    )
    arctic_drilling_revenue_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Revenue from Arctic drilling (%)",
    )
    deforestation_linked: bool = Field(
        default=False, description="Linked to deforestation",
    )

    # Norm-based fields
    ungc_violations: bool = Field(
        default=False, description="UNGC violations",
    )
    human_rights_violations: bool = Field(
        default=False, description="Severe human rights violations",
    )

    # Controversy
    controversy_score: float = Field(
        default=0.0, ge=0.0, le=10.0,
        description="Controversy severity score (0-10)",
    )

    # ESG quality
    esg_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="ESG quality score (0-100)",
    )

    # Carbon
    carbon_intensity: float = Field(
        default=0.0, ge=0.0,
        description="Carbon intensity (tCO2e/EUR M revenue)",
    )

    # Transition
    has_transition_plan: bool = Field(
        default=False, description="Has credible transition plan",
    )

    # Data quality
    data_coverage: str = Field(
        default="full",
        description="Data coverage (full/partial/minimal)",
    )


class ExclusionDetail(BaseModel):
    """Details of a single exclusion applied to a security.

    Records which rule was triggered, the actual value that caused
    the exclusion, and the threshold that was breached.
    """
    exclusion_id: str = Field(
        default_factory=_new_uuid, description="Exclusion ID",
    )
    security_id: str = Field(description="Security that was excluded")
    company_name: str = Field(default="", description="Company name")
    exclusion_type: ExclusionType = Field(
        description="Type of exclusion",
    )
    screening_layer: ScreeningLayer = Field(
        description="Screening layer that triggered exclusion",
    )
    rule_description: str = Field(
        default="", description="Exclusion rule description",
    )
    actual_value: float = Field(
        default=0.0, description="Actual value triggering exclusion",
    )
    threshold_value: float = Field(
        default=0.0, description="Threshold that was breached",
    )
    field_name: str = Field(
        default="", description="Data field name checked",
    )
    nav_impact: float = Field(
        default=0.0, description="NAV impact of excluding this security",
    )
    excluded_at: datetime = Field(
        default_factory=_utcnow, description="Exclusion timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


class WatchListEntry(BaseModel):
    """A security on the watch list due to proximity to exclusion threshold.

    Tracks securities that are near but not yet breaching exclusion
    thresholds, requiring enhanced monitoring.
    """
    entry_id: str = Field(
        default_factory=_new_uuid, description="Watch list entry ID",
    )
    security_id: str = Field(
        description="Security being watched",
    )
    company_name: str = Field(default="", description="Company name")
    exclusion_type: ExclusionType = Field(
        description="Exclusion type the security is near",
    )
    screening_layer: ScreeningLayer = Field(
        description="Screening layer",
    )
    actual_value: float = Field(
        default=0.0, description="Current actual value",
    )
    threshold_value: float = Field(
        default=0.0, description="Exclusion threshold",
    )
    distance_to_threshold_pct: float = Field(
        default=0.0,
        description="Distance to threshold as % of threshold",
    )
    risk_level: str = Field(
        default="medium",
        description="Risk level (high/medium/low)",
    )
    monitoring_frequency: str = Field(
        default="quarterly",
        description="Recommended monitoring frequency",
    )
    added_at: datetime = Field(
        default_factory=_utcnow, description="Date added to watch list",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


class PreApprovalResult(BaseModel):
    """Result of pre-approval screening for a new investment candidate.

    Indicates whether a security passes all screening layers and
    can be added to the investable universe.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result ID",
    )
    security_id: str = Field(
        description="Security screened for pre-approval",
    )
    company_name: str = Field(default="", description="Company name")
    approved: bool = Field(
        default=False, description="Whether pre-approval is granted",
    )
    layers_passed: List[ScreeningLayer] = Field(
        default_factory=list,
        description="Screening layers that passed",
    )
    layers_failed: List[ScreeningLayer] = Field(
        default_factory=list,
        description="Screening layers that failed",
    )
    exclusions: List[ExclusionDetail] = Field(
        default_factory=list,
        description="Exclusions triggered (if any)",
    )
    watch_list_flags: List[WatchListEntry] = Field(
        default_factory=list,
        description="Watch list flags (if any)",
    )
    total_layers_checked: int = Field(
        default=0, ge=0, description="Total screening layers checked",
    )
    recommendation: str = Field(
        default="",
        description="Screening recommendation (approve/reject/watch)",
    )
    screened_at: datetime = Field(
        default_factory=_utcnow, description="Screening timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


class UniverseCoverage(BaseModel):
    """Coverage statistics for the investment universe.

    Tracks how much of the total investable universe passes screening
    and is eligible for the Article 9 product.
    """
    coverage_id: str = Field(
        default_factory=_new_uuid, description="Coverage ID",
    )
    product_name: str = Field(
        default="", description="Financial product name",
    )
    total_securities_screened: int = Field(
        default=0, ge=0, description="Total securities screened",
    )
    eligible_securities: int = Field(
        default=0, ge=0, description="Securities passing all screens",
    )
    excluded_securities: int = Field(
        default=0, ge=0, description="Securities excluded",
    )
    watch_list_securities: int = Field(
        default=0, ge=0, description="Securities on watch list",
    )
    eligibility_rate_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Eligibility rate (%)",
    )
    exclusion_rate_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Exclusion rate (%)",
    )
    watch_list_rate_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Watch list rate (%)",
    )
    exclusions_by_layer: Dict[str, int] = Field(
        default_factory=dict,
        description="Exclusion counts per screening layer",
    )
    exclusions_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Exclusion counts per exclusion type",
    )
    eligible_nav: float = Field(
        default=0.0, ge=0.0,
        description="NAV of eligible holdings (EUR)",
    )
    excluded_nav: float = Field(
        default=0.0, ge=0.0,
        description="NAV of excluded holdings (EUR)",
    )
    data_coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Data coverage quality (%)",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow, description="Assessment timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


class ScreeningResult(BaseModel):
    """Complete result of investment universe screening.

    Consolidates all exclusions, watch list entries, coverage
    statistics, and per-security screening outcomes.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result ID",
    )
    product_name: str = Field(
        default="", description="Financial product name",
    )
    reporting_date: datetime = Field(
        default_factory=_utcnow, description="Reporting date",
    )

    # Exclusions
    exclusions: List[ExclusionDetail] = Field(
        default_factory=list,
        description="All exclusions applied",
    )
    total_excluded: int = Field(
        default=0, ge=0, description="Total securities excluded",
    )

    # Watch list
    watch_list: List[WatchListEntry] = Field(
        default_factory=list,
        description="Securities on watch list",
    )
    total_watch_list: int = Field(
        default=0, ge=0, description="Total watch list entries",
    )

    # Coverage
    universe_coverage: Optional[UniverseCoverage] = Field(
        default=None, description="Universe coverage statistics",
    )

    # Eligible securities
    eligible_security_ids: List[str] = Field(
        default_factory=list,
        description="IDs of securities passing all screens",
    )
    eligible_count: int = Field(
        default=0, ge=0, description="Total eligible securities",
    )

    # Summary
    total_screened: int = Field(
        default=0, ge=0, description="Total securities screened",
    )
    layers_applied: List[ScreeningLayer] = Field(
        default_factory=list,
        description="Screening layers applied",
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


class UniverseConfig(BaseModel):
    """Configuration for the InvestmentUniverseEngine.

    Controls which screening layers are active, threshold overrides,
    watch list proximity, and reporting parameters.

    Attributes:
        product_name: Financial product name.
        active_layers: Screening layers to apply.
        apply_pab_exclusions: Whether to apply PAB exclusions.
        apply_ctb_exclusions: Whether to apply CTB exclusions.
        apply_norm_based: Whether to apply norm-based screening.
        apply_sector_based: Whether to apply sector-based screening.
        apply_controversy: Whether to apply controversy screening.
        threshold_overrides: Custom threshold overrides by exclusion type.
        watch_list_proximity_pct: Proximity to threshold for watch list.
        min_esg_score: Minimum ESG score for eligibility.
        max_carbon_intensity: Maximum carbon intensity for eligibility.
    """
    product_name: str = Field(
        default="SFDR Article 9 Product", description="Product name",
    )
    active_layers: List[ScreeningLayer] = Field(
        default_factory=lambda: [
            ScreeningLayer.PAB_EXCLUSION,
            ScreeningLayer.NORM_BASED,
            ScreeningLayer.SECTOR_BASED,
            ScreeningLayer.CONTROVERSY_BASED,
        ],
        description="Active screening layers",
    )
    apply_pab_exclusions: bool = Field(
        default=True, description="Apply PAB exclusions",
    )
    apply_ctb_exclusions: bool = Field(
        default=False, description="Apply CTB exclusions (less strict)",
    )
    apply_norm_based: bool = Field(
        default=True, description="Apply norm-based screening",
    )
    apply_sector_based: bool = Field(
        default=True, description="Apply sector-based screening",
    )
    apply_controversy: bool = Field(
        default=True, description="Apply controversy screening",
    )
    threshold_overrides: Dict[str, float] = Field(
        default_factory=dict,
        description="Custom threshold overrides by exclusion type",
    )
    watch_list_proximity_pct: float = Field(
        default=20.0, ge=0.0, le=100.0,
        description="Proximity to threshold for watch list (%)",
    )
    min_esg_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum ESG score for eligibility (0 = no filter)",
    )
    max_carbon_intensity: float = Field(
        default=0.0, ge=0.0,
        description="Max carbon intensity for eligibility (0 = no filter)",
    )


# ---------------------------------------------------------------------------
# model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

UniverseConfig.model_rebuild()
SecurityData.model_rebuild()
ExclusionDetail.model_rebuild()
WatchListEntry.model_rebuild()
PreApprovalResult.model_rebuild()
UniverseCoverage.model_rebuild()
ScreeningResult.model_rebuild()


# ---------------------------------------------------------------------------
# InvestmentUniverseEngine
# ---------------------------------------------------------------------------


class InvestmentUniverseEngine:
    """
    Investment universe screening engine for SFDR Article 9 products.

    Implements multi-layer screening with PAB/CTB exclusions, norm-based
    screening, sector-based screening, controversy-based screening,
    watch list management, and pre-approval workflows.

    Zero-Hallucination Guarantees:
        - All exclusion checks use deterministic threshold comparisons
        - Multi-layer screening applies boolean rule evaluation
        - Watch list proximity uses pure arithmetic
        - SHA-256 provenance hash on every result
        - No LLM involvement in any screening decision

    Attributes:
        config: Engine configuration.
        _securities: Input security data.
        _exclusions: Computed exclusions.
        _watch_list: Computed watch list entries.

    Example:
        >>> config = UniverseConfig(product_name="Art 9 Fund")
        >>> engine = InvestmentUniverseEngine(config)
        >>> securities = [SecurityData(
        ...     company_name="Clean Corp", isin="XX000000001",
        ...     coal_revenue_pct=0.0, controversial_weapons=False,
        ... )]
        >>> result = engine.screen_universe(securities)
        >>> print(f"Eligible: {result.eligible_count}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize InvestmentUniverseEngine.

        Args:
            config: Optional configuration dict or UniverseConfig.
        """
        if config and isinstance(config, dict):
            self.config = UniverseConfig(**config)
        elif config and isinstance(config, UniverseConfig):
            self.config = config
        else:
            self.config = UniverseConfig()

        self._securities: List[SecurityData] = []
        self._exclusions: List[ExclusionDetail] = []
        self._watch_list: List[WatchListEntry] = []

        logger.info(
            "InvestmentUniverseEngine initialized (version=%s, "
            "product=%s, layers=%d)",
            _MODULE_VERSION,
            self.config.product_name,
            len(self.config.active_layers),
        )

    # ------------------------------------------------------------------
    # Public API: Full Universe Screening
    # ------------------------------------------------------------------

    def screen_universe(
        self,
        securities: List[SecurityData],
    ) -> ScreeningResult:
        """Screen the full investment universe through all active layers.

        Applies exclusion rules, identifies watch list candidates,
        and computes universe coverage statistics.

        Args:
            securities: List of securities to screen.

        Returns:
            ScreeningResult with complete screening outcome.

        Raises:
            ValueError: If securities list is empty.
        """
        start = _utcnow()

        if not securities:
            raise ValueError("Securities list cannot be empty")

        self._securities = securities
        self._exclusions = []
        self._watch_list = []

        logger.info(
            "Screening %d securities for Article 9 universe",
            len(securities),
        )

        excluded_ids: set = set()
        all_exclusions: List[ExclusionDetail] = []
        all_watch: List[WatchListEntry] = []

        for sec in securities:
            sec_exclusions, sec_watch = self._screen_security(sec)
            if sec_exclusions:
                excluded_ids.add(sec.security_id)
                all_exclusions.extend(sec_exclusions)
            all_watch.extend(sec_watch)

        # Apply ESG and carbon filters
        for sec in securities:
            if sec.security_id in excluded_ids:
                continue
            esg_excl = self._check_esg_quality(sec)
            if esg_excl:
                excluded_ids.add(sec.security_id)
                all_exclusions.extend(esg_excl)

        self._exclusions = all_exclusions
        self._watch_list = all_watch

        # Compute eligible
        eligible_ids = [
            s.security_id for s in securities
            if s.security_id not in excluded_ids
        ]

        # Universe coverage
        coverage = self._compute_coverage(
            securities, excluded_ids, all_exclusions, all_watch
        )

        processing_ms = (_utcnow() - start).total_seconds() * 1000.0

        result = ScreeningResult(
            product_name=self.config.product_name,
            exclusions=all_exclusions,
            total_excluded=len(excluded_ids),
            watch_list=all_watch,
            total_watch_list=len(all_watch),
            universe_coverage=coverage,
            eligible_security_ids=eligible_ids,
            eligible_count=len(eligible_ids),
            total_screened=len(securities),
            layers_applied=self.config.active_layers,
            processing_time_ms=processing_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Universe screened: total=%d, eligible=%d, excluded=%d, "
            "watch=%d in %.0fms",
            len(securities),
            len(eligible_ids),
            len(excluded_ids),
            len(all_watch),
            processing_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Pre-approval Screening
    # ------------------------------------------------------------------

    def pre_approve(
        self,
        security: SecurityData,
    ) -> PreApprovalResult:
        """Screen a single security for pre-approval.

        Runs all active screening layers and produces a recommendation
        (approve, reject, or watch).

        Args:
            security: Security data to screen.

        Returns:
            PreApprovalResult with screening outcome.
        """
        exclusions, watch_flags = self._screen_security(security)
        esg_exclusions = self._check_esg_quality(security)
        all_exclusions = exclusions + esg_exclusions

        # Determine passed/failed layers
        failed_layers: set = set()
        passed_layers: set = set()

        for excl in all_exclusions:
            failed_layers.add(excl.screening_layer)

        for layer in self.config.active_layers:
            if layer not in failed_layers:
                passed_layers.add(layer)

        # Recommendation
        if all_exclusions:
            recommendation = "reject"
            approved = False
        elif watch_flags:
            recommendation = "watch"
            approved = True
        else:
            recommendation = "approve"
            approved = True

        result = PreApprovalResult(
            security_id=security.security_id,
            company_name=security.company_name,
            approved=approved,
            layers_passed=sorted(passed_layers, key=lambda x: x.value),
            layers_failed=sorted(failed_layers, key=lambda x: x.value),
            exclusions=all_exclusions,
            watch_list_flags=watch_flags,
            total_layers_checked=len(self.config.active_layers),
            recommendation=recommendation,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Pre-approval screened: %s -> %s (exclusions=%d, watch=%d)",
            security.company_name or security.security_id,
            recommendation,
            len(all_exclusions),
            len(watch_flags),
        )
        return result

    # ------------------------------------------------------------------
    # Internal: Security screening
    # ------------------------------------------------------------------

    def _screen_security(
        self,
        security: SecurityData,
    ) -> Tuple[List[ExclusionDetail], List[WatchListEntry]]:
        """Screen a single security through all active layers.

        Args:
            security: Security to screen.

        Returns:
            Tuple of (exclusions, watch_list_entries).
        """
        exclusions: List[ExclusionDetail] = []
        watch_entries: List[WatchListEntry] = []

        # Collect all applicable rules
        rules_to_check: List[Dict[str, Any]] = []

        if self.config.apply_pab_exclusions:
            for etype, rule in PAB_EXCLUSION_RULES.items():
                rules_to_check.append({**rule, "exclusion_type": etype})

        if self.config.apply_ctb_exclusions:
            for etype, rule in CTB_EXCLUSION_RULES.items():
                # Avoid duplicate controversial weapons check
                if not any(
                    r["exclusion_type"] == etype for r in rules_to_check
                ):
                    rules_to_check.append({**rule, "exclusion_type": etype})

        if self.config.apply_norm_based:
            for etype, rule in NORM_BASED_RULES.items():
                rules_to_check.append({**rule, "exclusion_type": etype})

        if self.config.apply_sector_based:
            for etype, rule in SECTOR_BASED_RULES.items():
                rules_to_check.append({**rule, "exclusion_type": etype})

        if self.config.apply_controversy:
            for etype, rule in CONTROVERSY_RULES.items():
                rules_to_check.append({**rule, "exclusion_type": etype})

        # Check each rule
        for rule in rules_to_check:
            excl, watch = self._check_rule(security, rule)
            if excl:
                exclusions.append(excl)
            if watch:
                watch_entries.append(watch)

        return exclusions, watch_entries

    def _check_rule(
        self,
        security: SecurityData,
        rule: Dict[str, Any],
    ) -> Tuple[Optional[ExclusionDetail], Optional[WatchListEntry]]:
        """Check a single exclusion rule against a security.

        Args:
            security: Security to check.
            rule: Exclusion rule definition.

        Returns:
            Tuple of (exclusion_or_none, watch_entry_or_none).
        """
        field_name = rule["field"]
        check_type = rule["type"]
        threshold = rule["threshold"]
        etype_str = rule["exclusion_type"]
        layer = rule["layer"]
        description = rule.get("description", "")

        # Apply threshold override if configured
        if etype_str in self.config.threshold_overrides:
            threshold = self.config.threshold_overrides[etype_str]

        # Get actual value
        actual_value = getattr(security, field_name, None)
        if actual_value is None:
            return None, None

        # Boolean check
        if check_type == "boolean_true":
            if actual_value is True:
                excl = ExclusionDetail(
                    security_id=security.security_id,
                    company_name=security.company_name,
                    exclusion_type=ExclusionType(etype_str),
                    screening_layer=layer,
                    rule_description=description,
                    actual_value=1.0,
                    threshold_value=0.0,
                    field_name=field_name,
                    nav_impact=security.nav_value,
                )
                excl.provenance_hash = _compute_hash(excl)
                return excl, None
            return None, None

        # Numeric checks
        numeric_val = float(actual_value)

        excluded = False
        if check_type == "gte":
            excluded = numeric_val >= threshold
        elif check_type == "gt":
            excluded = numeric_val > threshold
        elif check_type == "lte":
            excluded = numeric_val <= threshold
        elif check_type == "lt":
            excluded = numeric_val < threshold

        if excluded:
            excl = ExclusionDetail(
                security_id=security.security_id,
                company_name=security.company_name,
                exclusion_type=ExclusionType(etype_str),
                screening_layer=layer,
                rule_description=description,
                actual_value=numeric_val,
                threshold_value=threshold,
                field_name=field_name,
                nav_impact=security.nav_value,
            )
            excl.provenance_hash = _compute_hash(excl)
            return excl, None

        # Check watch list proximity (only for numeric thresholds > 0)
        watch = None
        if threshold > 0 and numeric_val > 0:
            distance_pct = (
                (threshold - numeric_val) / threshold
            ) * 100.0

            if distance_pct <= self.config.watch_list_proximity_pct:
                risk_level = "high" if distance_pct <= 5.0 else (
                    "medium" if distance_pct <= 10.0 else "low"
                )
                monitoring = "monthly" if risk_level == "high" else (
                    "quarterly" if risk_level == "medium" else "semi-annual"
                )

                watch = WatchListEntry(
                    security_id=security.security_id,
                    company_name=security.company_name,
                    exclusion_type=ExclusionType(etype_str),
                    screening_layer=layer,
                    actual_value=numeric_val,
                    threshold_value=threshold,
                    distance_to_threshold_pct=_round_val(
                        distance_pct, 4
                    ),
                    risk_level=risk_level,
                    monitoring_frequency=monitoring,
                )
                watch.provenance_hash = _compute_hash(watch)

        return None, watch

    # ------------------------------------------------------------------
    # Internal: ESG quality check
    # ------------------------------------------------------------------

    def _check_esg_quality(
        self,
        security: SecurityData,
    ) -> List[ExclusionDetail]:
        """Check ESG quality and carbon intensity thresholds.

        Args:
            security: Security to check.

        Returns:
            List of exclusions (empty if passed).
        """
        exclusions: List[ExclusionDetail] = []

        # ESG score check
        if (self.config.min_esg_score > 0
                and security.esg_score > 0
                and security.esg_score < self.config.min_esg_score):
            excl = ExclusionDetail(
                security_id=security.security_id,
                company_name=security.company_name,
                exclusion_type=ExclusionType.SEVERE_CONTROVERSY,
                screening_layer=ScreeningLayer.ESG_QUALITY,
                rule_description=(
                    f"ESG score {security.esg_score} below minimum "
                    f"{self.config.min_esg_score}"
                ),
                actual_value=security.esg_score,
                threshold_value=self.config.min_esg_score,
                field_name="esg_score",
                nav_impact=security.nav_value,
            )
            excl.provenance_hash = _compute_hash(excl)
            exclusions.append(excl)

        # Carbon intensity check
        if (self.config.max_carbon_intensity > 0
                and security.carbon_intensity > 0
                and security.carbon_intensity
                > self.config.max_carbon_intensity):
            excl = ExclusionDetail(
                security_id=security.security_id,
                company_name=security.company_name,
                exclusion_type=ExclusionType.HIGH_CARBON_POWER,
                screening_layer=ScreeningLayer.ESG_QUALITY,
                rule_description=(
                    f"Carbon intensity {security.carbon_intensity} "
                    f"exceeds maximum {self.config.max_carbon_intensity}"
                ),
                actual_value=security.carbon_intensity,
                threshold_value=self.config.max_carbon_intensity,
                field_name="carbon_intensity",
                nav_impact=security.nav_value,
            )
            excl.provenance_hash = _compute_hash(excl)
            exclusions.append(excl)

        return exclusions

    # ------------------------------------------------------------------
    # Internal: Coverage computation
    # ------------------------------------------------------------------

    def _compute_coverage(
        self,
        securities: List[SecurityData],
        excluded_ids: set,
        exclusions: List[ExclusionDetail],
        watch_list: List[WatchListEntry],
    ) -> UniverseCoverage:
        """Compute universe coverage statistics.

        Args:
            securities: All screened securities.
            excluded_ids: Set of excluded security IDs.
            exclusions: All exclusion details.
            watch_list: All watch list entries.

        Returns:
            UniverseCoverage with statistics.
        """
        total = len(securities)
        excluded = len(excluded_ids)
        watch_ids = set(w.security_id for w in watch_list)
        watch_count = len(watch_ids - excluded_ids)
        eligible = total - excluded

        # Exclusions by layer
        by_layer: Dict[str, int] = defaultdict(int)
        by_type: Dict[str, int] = defaultdict(int)
        for excl in exclusions:
            by_layer[excl.screening_layer.value] += 1
            by_type[excl.exclusion_type.value] += 1

        # NAV calculations
        eligible_nav = sum(
            s.nav_value for s in securities
            if s.security_id not in excluded_ids
        )
        excluded_nav = sum(
            s.nav_value for s in securities
            if s.security_id in excluded_ids
        )

        # Data coverage
        full_data = sum(
            1 for s in securities if s.data_coverage == "full"
        )
        data_coverage = _safe_pct(full_data, total)

        coverage = UniverseCoverage(
            product_name=self.config.product_name,
            total_securities_screened=total,
            eligible_securities=eligible,
            excluded_securities=excluded,
            watch_list_securities=watch_count,
            eligibility_rate_pct=_round_val(
                _safe_pct(eligible, total), 4
            ),
            exclusion_rate_pct=_round_val(
                _safe_pct(excluded, total), 4
            ),
            watch_list_rate_pct=_round_val(
                _safe_pct(watch_count, total), 4
            ),
            exclusions_by_layer=dict(by_layer),
            exclusions_by_type=dict(by_type),
            eligible_nav=_round_val(eligible_nav, 2),
            excluded_nav=_round_val(excluded_nav, 2),
            data_coverage_pct=_round_val(data_coverage, 4),
        )
        coverage.provenance_hash = _compute_hash(coverage)
        return coverage
