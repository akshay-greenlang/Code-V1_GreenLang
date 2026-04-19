# -*- coding: utf-8 -*-
"""
TemperatureScoringEngine - PACK-022 Net Zero Acceleration Engine 6
====================================================================

Implements the SBTi Temperature Rating v2.0 methodology to assign
temperature alignment scores to individual company targets and to
aggregate those scores into portfolio-level temperature ratings.

SBTi Temperature Rating v2.0 Framework:
    The methodology translates corporate GHG emissions reduction targets
    into an intuitive temperature score expressed in degrees Celsius.
    This score represents the global temperature rise that would occur
    if the global economy had the same level of ambition as the assessed
    company.  Scores are derived by mapping the annual linear reduction
    rate (ARR) of a target onto a calibrated temperature mapping curve
    based on IPCC AR6 carbon budgets and integrated assessment models.

    Key methodological elements:
    - Linear annual reduction rate (ARR) derived from target parameters
    - Temperature mapping via a piecewise-linear lookup table
    - Default temperature score of 3.2 C for entities without valid targets
    - Six portfolio aggregation methods: WATS, TETS, MOTS, EOTS, ECOTS, AOTS
    - Contribution analysis for identifying portfolio temperature drivers
    - What-if analysis for evaluating target improvement scenarios

Aggregation Methods (SBTi TR v2.0):
    - WATS: Weighted Average Temperature Score (by revenue weight)
    - TETS: Total Emissions Temperature Score (by total emissions weight)
    - MOTS: Market Owned Temperature Score (by market cap ownership)
    - EOTS: Enterprise Owned Temperature Score (by enterprise value ownership)
    - ECOTS: Enterprise Value + Cash Temperature Score
    - AOTS: All-Owned Temperature Score (by total invested capital)

Target Scopes:
    - S1S2: Scope 1 + 2 combined
    - S3: Scope 3
    - S1S2S3: All scopes combined

Target Timeframes:
    - Near-term: 5-10 year targets
    - Long-term: targets beyond 2035

Regulatory References:
    - SBTi Temperature Rating Methodology v2.0 (2024)
    - SBTi Corporate Net-Zero Standard v1.1 (2023)
    - IPCC AR6 WG3 Chapter 3: Mitigation pathways
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - Paris Agreement Article 2.1(a)

Zero-Hallucination:
    - Temperature mapping uses a deterministic piecewise-linear lookup table
    - All ARR calculations use Decimal arithmetic with ROUND_HALF_UP
    - Portfolio aggregation uses deterministic weighted average formulas
    - Default score of 3.2 C applied only for entities with no valid target
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-022 Net Zero Acceleration
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning default on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(
    part: Decimal, whole: Decimal, places: int = 2
) -> Decimal:
    """Calculate percentage safely, returning 0 on zero denominator."""
    if whole == Decimal("0"):
        return Decimal("0")
    return (part / whole * Decimal("100")).quantize(
        Decimal("0." + "0" * places), rounding=ROUND_HALF_UP
    )

def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal value to the specified number of decimal places."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Reference Data: SBTi Temperature Mapping Table
# ---------------------------------------------------------------------------
# The mapping translates annual linear reduction rate (% per year) to
# temperature score (degrees C).  Derived from IPCC AR6 carbon budgets
# and SBTi calibration.  The table uses piecewise-linear interpolation
# between these anchor points.

TEMPERATURE_MAPPING_TABLE: List[Tuple[Decimal, Decimal]] = [
    # (annual_reduction_rate_pct, temperature_c)
    (Decimal("7.0"), Decimal("1.20")),
    (Decimal("6.0"), Decimal("1.30")),
    (Decimal("5.0"), Decimal("1.40")),
    (Decimal("4.2"), Decimal("1.50")),
    (Decimal("3.5"), Decimal("1.60")),
    (Decimal("3.0"), Decimal("1.70")),
    (Decimal("2.5"), Decimal("1.80")),
    (Decimal("2.0"), Decimal("2.00")),
    (Decimal("1.6"), Decimal("2.20")),
    (Decimal("1.2"), Decimal("2.50")),
    (Decimal("0.8"), Decimal("2.70")),
    (Decimal("0.5"), Decimal("2.90")),
    (Decimal("0.2"), Decimal("3.00")),
    (Decimal("0.0"), Decimal("3.20")),
]

# Default temperature for companies without any validated target
DEFAULT_TEMPERATURE_SCORE: Decimal = Decimal("3.20")

# Temperature band definitions (degrees C)
TEMPERATURE_BANDS: Dict[str, Tuple[Decimal, Decimal]] = {
    "1.5C": (Decimal("0"), Decimal("1.50")),
    "WELL_BELOW_2C": (Decimal("1.50"), Decimal("1.80")),
    "BELOW_2C": (Decimal("1.80"), Decimal("2.00")),
    "2C": (Decimal("2.00"), Decimal("2.50")),
    "ABOVE_2C": (Decimal("2.50"), Decimal("3.20")),
    "NO_TARGET": (Decimal("3.20"), Decimal("99.0")),
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScoreType(str, Enum):
    """Temperature score aggregation method per SBTi TR v2.0."""
    WATS = "wats"
    TETS = "tets"
    MOTS = "mots"
    EOTS = "eots"
    ECOTS = "ecots"
    AOTS = "aots"

class TargetScope(str, Enum):
    """Target scope classification."""
    S1S2 = "s1s2"
    S3 = "s3"
    S1S2S3 = "s1s2s3"

class TargetTimeframe(str, Enum):
    """Target timeframe classification."""
    NEAR_TERM = "near_term"
    LONG_TERM = "long_term"

class AggregationMethod(str, Enum):
    """Weight basis for portfolio aggregation."""
    REVENUE = "revenue"
    TOTAL_EMISSIONS = "total_emissions"
    MARKET_CAP = "market_cap"
    ENTERPRISE_VALUE = "enterprise_value"
    EV_PLUS_CASH = "ev_plus_cash"
    TOTAL_INVESTED_CAPITAL = "total_invested_capital"

class TemperatureBand(str, Enum):
    """Temperature alignment band."""
    ALIGNED_1_5C = "1.5C"
    WELL_BELOW_2C = "well_below_2C"
    BELOW_2C = "below_2C"
    ALIGNED_2C = "2C"
    ABOVE_2C = "above_2C"
    NO_TARGET = "no_target"

class TargetValidityStatus(str, Enum):
    """Validity status of a target for temperature scoring."""
    VALID = "valid"
    EXPIRED = "expired"
    INVALID_ARR = "invalid_arr"
    MISSING = "missing"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class EmissionsTarget(BaseModel):
    """A single emissions reduction target for an entity."""
    target_id: str = Field(default_factory=_new_uuid, description="Target identifier")
    entity_id: str = Field(description="Owning entity identifier")
    scope: TargetScope = Field(description="Target scope (S1S2, S3, S1S2S3)")
    timeframe: TargetTimeframe = Field(description="Near-term or long-term")
    base_year: int = Field(description="Target base year")
    target_year: int = Field(description="Target end year")
    base_year_emissions: Decimal = Field(description="Base year emissions (tCO2e)")
    target_year_emissions: Decimal = Field(description="Target year emissions (tCO2e)")
    reduction_pct: Decimal = Field(default=Decimal("0"), description="Target reduction percentage")
    is_sbti_validated: bool = Field(default=False, description="Whether SBTi validated")
    is_net_zero_aligned: bool = Field(default=False, description="Net-zero aligned")
    created_at: datetime = Field(default_factory=utcnow, description="Creation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("base_year_emissions", "target_year_emissions",
                     "reduction_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class PortfolioEntity(BaseModel):
    """Entity in a portfolio for temperature scoring."""
    entity_id: str = Field(default_factory=_new_uuid, description="Entity identifier")
    entity_name: str = Field(description="Entity name")
    sector: str = Field(default="", description="Industry sector")
    country: str = Field(default="", description="Country code")
    revenue: Decimal = Field(default=Decimal("0"), description="Revenue (reporting currency)")
    total_emissions: Decimal = Field(default=Decimal("0"), description="Total GHG emissions tCO2e")
    scope1_emissions: Decimal = Field(default=Decimal("0"), description="Scope 1 tCO2e")
    scope2_emissions: Decimal = Field(default=Decimal("0"), description="Scope 2 tCO2e")
    scope3_emissions: Decimal = Field(default=Decimal("0"), description="Scope 3 tCO2e")
    market_cap: Decimal = Field(default=Decimal("0"), description="Market capitalization")
    enterprise_value: Decimal = Field(default=Decimal("0"), description="Enterprise value")
    ev_plus_cash: Decimal = Field(default=Decimal("0"), description="Enterprise value + cash")
    total_invested_capital: Decimal = Field(default=Decimal("0"), description="Total invested capital")
    ownership_pct: Decimal = Field(default=Decimal("100"), description="Ownership percentage")
    investment_value: Decimal = Field(default=Decimal("0"), description="Portfolio investment value")
    targets: List[EmissionsTarget] = Field(default_factory=list, description="Reduction targets")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("revenue", "total_emissions", "scope1_emissions",
                     "scope2_emissions", "scope3_emissions", "market_cap",
                     "enterprise_value", "ev_plus_cash",
                     "total_invested_capital", "ownership_pct",
                     "investment_value", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class EntityTemperatureScore(BaseModel):
    """Temperature score for a single entity."""
    entity_id: str = Field(description="Entity identifier")
    entity_name: str = Field(default="", description="Entity name")
    scope: TargetScope = Field(description="Scope of assessment")
    timeframe: TargetTimeframe = Field(description="Timeframe of assessment")
    target_id: Optional[str] = Field(default=None, description="Associated target ID")
    annual_reduction_rate: Decimal = Field(description="Annual reduction rate (% per year)")
    temperature_score: Decimal = Field(description="Temperature score (degrees C)")
    temperature_band: TemperatureBand = Field(description="Temperature alignment band")
    validity_status: TargetValidityStatus = Field(description="Target validity status")
    is_default_score: bool = Field(default=False, description="Whether default 3.2C was used")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("annual_reduction_rate", "temperature_score", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class PortfolioTemperatureScore(BaseModel):
    """Aggregated portfolio temperature score."""
    portfolio_id: str = Field(default_factory=_new_uuid, description="Portfolio identifier")
    score_type: ScoreType = Field(description="Aggregation method used")
    scope: TargetScope = Field(description="Scope of aggregation")
    timeframe: TargetTimeframe = Field(description="Timeframe of aggregation")
    temperature_score: Decimal = Field(description="Portfolio temperature score (C)")
    temperature_band: TemperatureBand = Field(description="Temperature alignment band")
    entities_count: int = Field(default=0, description="Number of entities in portfolio")
    entities_with_targets: int = Field(default=0, description="Entities with valid targets")
    entities_defaulted: int = Field(default=0, description="Entities using default score")
    coverage_pct: Decimal = Field(default=Decimal("0"), description="Target coverage (% of weight)")
    methodology_version: str = Field(default="SBTi TR v2.0", description="Methodology version")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("temperature_score", "coverage_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class ContributionEntry(BaseModel):
    """Contribution of one entity to portfolio temperature."""
    entity_id: str = Field(description="Entity identifier")
    entity_name: str = Field(default="", description="Entity name")
    weight: Decimal = Field(description="Weight in portfolio")
    weight_pct: Decimal = Field(description="Weight percentage")
    entity_temperature: Decimal = Field(description="Entity temperature score (C)")
    contribution: Decimal = Field(description="Weighted contribution to portfolio score")
    contribution_pct: Decimal = Field(description="Contribution as percentage of total")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("weight", "weight_pct", "entity_temperature",
                     "contribution", "contribution_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class WhatIfScenario(BaseModel):
    """What-if scenario definition."""
    scenario_id: str = Field(default_factory=_new_uuid, description="Scenario identifier")
    scenario_name: str = Field(default="", description="Human-readable name")
    entity_id: str = Field(description="Entity to modify")
    new_annual_reduction_rate: Optional[Decimal] = Field(
        default=None, description="Override ARR (% per year)"
    )
    new_target_scope: Optional[TargetScope] = Field(
        default=None, description="Override target scope"
    )
    new_target_year: Optional[int] = Field(
        default=None, description="Override target year"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("new_annual_reduction_rate", mode="before")
    @classmethod
    def _coerce_decimal_opt(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

class WhatIfResult(BaseModel):
    """Result of what-if analysis."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    scenario_id: str = Field(description="Scenario identifier")
    scenario_name: str = Field(default="", description="Scenario name")
    original_portfolio_temperature: Decimal = Field(description="Original portfolio temp (C)")
    modified_portfolio_temperature: Decimal = Field(description="Modified portfolio temp (C)")
    temperature_change: Decimal = Field(description="Change in temperature (C)")
    improvement_pct: Decimal = Field(description="Improvement percentage (negative = better)")
    entity_id: str = Field(description="Modified entity")
    original_entity_temperature: Decimal = Field(description="Original entity temp (C)")
    modified_entity_temperature: Decimal = Field(description="Modified entity temp (C)")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("original_portfolio_temperature", "modified_portfolio_temperature",
                     "temperature_change", "improvement_pct",
                     "original_entity_temperature", "modified_entity_temperature",
                     mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class TemperatureResult(BaseModel):
    """Complete temperature scoring result."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    entity_scores: List[EntityTemperatureScore] = Field(
        default_factory=list, description="Per-entity temperature scores"
    )
    portfolio_scores: List[PortfolioTemperatureScore] = Field(
        default_factory=list, description="Portfolio-level scores by method"
    )
    contribution_analysis: List[ContributionEntry] = Field(
        default_factory=list, description="Entity contributions to portfolio"
    )
    what_if_results: List[WhatIfResult] = Field(
        default_factory=list, description="What-if scenario results"
    )
    methodology_version: str = Field(
        default="SBTi TR v2.0", description="Methodology version"
    )
    entities_assessed: int = Field(default=0, description="Total entities assessed")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class TemperatureScoringConfig(BaseModel):
    """Configuration for the TemperatureScoringEngine."""
    default_temperature: Decimal = Field(
        default=DEFAULT_TEMPERATURE_SCORE,
        description="Default temperature for entities without targets (C)",
    )
    min_target_years: int = Field(
        default=5, description="Minimum target duration for near-term (years)"
    )
    max_target_years_near_term: int = Field(
        default=10, description="Maximum target duration for near-term (years)"
    )
    long_term_start_year: int = Field(
        default=2035, description="Year from which targets are classified as long-term"
    )
    default_aggregation: ScoreType = Field(
        default=ScoreType.WATS, description="Default portfolio aggregation method"
    )
    score_precision: int = Field(
        default=4, description="Decimal places for temperature scores"
    )

# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

EmissionsTarget.model_rebuild()
PortfolioEntity.model_rebuild()
EntityTemperatureScore.model_rebuild()
PortfolioTemperatureScore.model_rebuild()
ContributionEntry.model_rebuild()
WhatIfScenario.model_rebuild()
WhatIfResult.model_rebuild()
TemperatureResult.model_rebuild()
TemperatureScoringConfig.model_rebuild()

# ---------------------------------------------------------------------------
# TemperatureScoringEngine
# ---------------------------------------------------------------------------

class TemperatureScoringEngine:
    """
    SBTi Temperature Rating v2.0 engine.

    Translates corporate emissions reduction targets into temperature
    alignment scores and aggregates them at the portfolio level using
    six aggregation methods: WATS, TETS, MOTS, EOTS, ECOTS, AOTS.

    Attributes:
        config: Engine configuration.
        _entities: In-memory entity store.

    Example:
        >>> engine = TemperatureScoringEngine()
        >>> entity = PortfolioEntity(entity_name="Acme", ...)
        >>> engine.add_entity(entity)
        >>> scores = engine.score_entity("...", TargetScope.S1S2, TargetTimeframe.NEAR_TERM)
        >>> portfolio = engine.aggregate_portfolio(ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TemperatureScoringEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = TemperatureScoringConfig(**config)
        elif config and isinstance(config, TemperatureScoringConfig):
            self.config = config
        else:
            self.config = TemperatureScoringConfig()

        self._entities: Dict[str, PortfolioEntity] = {}
        logger.info("TemperatureScoringEngine initialized (v%s)", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Entity Management
    # -------------------------------------------------------------------

    def add_entity(self, entity: PortfolioEntity) -> PortfolioEntity:
        """Add a portfolio entity for temperature scoring.

        Args:
            entity: PortfolioEntity with targets.

        Returns:
            Entity with computed provenance hash.
        """
        entity.provenance_hash = _compute_hash(entity)
        self._entities[entity.entity_id] = entity
        logger.info("Added entity %s: %s", entity.entity_id, entity.entity_name)
        return entity

    def add_entities(self, entities: List[PortfolioEntity]) -> int:
        """Add multiple portfolio entities.

        Args:
            entities: List of PortfolioEntity objects.

        Returns:
            Number of entities added.
        """
        for entity in entities:
            self.add_entity(entity)
        logger.info("Added %d entities", len(entities))
        return len(entities)

    def get_entity(self, entity_id: str) -> PortfolioEntity:
        """Retrieve an entity by ID.

        Args:
            entity_id: Entity identifier.

        Returns:
            PortfolioEntity.

        Raises:
            ValueError: If entity not found.
        """
        if entity_id not in self._entities:
            raise ValueError(f"Entity {entity_id} not found")
        return self._entities[entity_id]

    # -------------------------------------------------------------------
    # Annual Reduction Rate
    # -------------------------------------------------------------------

    def calculate_arr(self, target: EmissionsTarget) -> Decimal:
        """Calculate the annual linear reduction rate (ARR) for a target.

        The ARR represents the average annual percentage reduction in
        emissions implied by the target over its duration.

        Formula:
            ARR = (1 - (target_emissions / base_emissions)) / (target_year - base_year) * 100

        Args:
            target: EmissionsTarget to evaluate.

        Returns:
            Annual reduction rate as a Decimal percentage (e.g. 4.2 means 4.2% per year).
        """
        duration = target.target_year - target.base_year
        if duration <= 0:
            logger.warning(
                "Target %s has non-positive duration (%d), returning 0 ARR",
                target.target_id, duration,
            )
            return Decimal("0")

        if target.base_year_emissions <= Decimal("0"):
            logger.warning(
                "Target %s has zero/negative base year emissions, returning 0 ARR",
                target.target_id,
            )
            return Decimal("0")

        reduction_fraction = Decimal("1") - _safe_divide(
            target.target_year_emissions,
            target.base_year_emissions,
            Decimal("0"),
        )

        arr = _safe_divide(
            reduction_fraction * Decimal("100"),
            _decimal(duration),
            Decimal("0"),
        )
        return _round_val(arr, self.config.score_precision)

    # -------------------------------------------------------------------
    # Temperature Mapping
    # -------------------------------------------------------------------

    def map_arr_to_temperature(self, arr: Decimal) -> Decimal:
        """Map an annual reduction rate to a temperature score.

        Uses piecewise-linear interpolation on the SBTi temperature
        mapping table.  ARR values above the table maximum yield the
        minimum temperature; values below zero yield the default score.

        Args:
            arr: Annual reduction rate (% per year).

        Returns:
            Temperature score in degrees Celsius.
        """
        if arr <= Decimal("0"):
            return self.config.default_temperature

        table = TEMPERATURE_MAPPING_TABLE
        # Table is sorted descending by ARR
        # If ARR is above highest entry, return lowest temperature
        if arr >= table[0][0]:
            return table[0][1]

        # Walk through table to find bracketing entries
        for i in range(len(table) - 1):
            upper_arr, upper_temp = table[i]
            lower_arr, lower_temp = table[i + 1]

            if lower_arr <= arr <= upper_arr:
                # Piecewise-linear interpolation
                arr_range = upper_arr - lower_arr
                if arr_range == Decimal("0"):
                    return upper_temp
                fraction = (arr - lower_arr) / arr_range
                temp = lower_temp - fraction * (lower_temp - upper_temp)
                return _round_val(temp, self.config.score_precision)

        # Below lowest non-zero entry
        return self.config.default_temperature

    def classify_temperature_band(self, temperature: Decimal) -> TemperatureBand:
        """Classify a temperature score into a temperature band.

        Args:
            temperature: Temperature score in degrees Celsius.

        Returns:
            TemperatureBand enum value.
        """
        if temperature <= Decimal("1.50"):
            return TemperatureBand.ALIGNED_1_5C
        elif temperature <= Decimal("1.80"):
            return TemperatureBand.WELL_BELOW_2C
        elif temperature <= Decimal("2.00"):
            return TemperatureBand.BELOW_2C
        elif temperature <= Decimal("2.50"):
            return TemperatureBand.ALIGNED_2C
        elif temperature < Decimal("3.20"):
            return TemperatureBand.ABOVE_2C
        else:
            return TemperatureBand.NO_TARGET

    # -------------------------------------------------------------------
    # Entity Scoring
    # -------------------------------------------------------------------

    def _classify_target_timeframe(self, target: EmissionsTarget) -> TargetTimeframe:
        """Determine the effective timeframe of a target.

        Args:
            target: Target to classify.

        Returns:
            TargetTimeframe.NEAR_TERM or TargetTimeframe.LONG_TERM.
        """
        if target.target_year >= self.config.long_term_start_year:
            return TargetTimeframe.LONG_TERM
        return TargetTimeframe.NEAR_TERM

    def _validate_target(self, target: EmissionsTarget) -> TargetValidityStatus:
        """Validate a target for temperature scoring eligibility.

        Args:
            target: Target to validate.

        Returns:
            TargetValidityStatus.
        """
        duration = target.target_year - target.base_year
        if duration < self.config.min_target_years:
            return TargetValidityStatus.INVALID_ARR

        if target.base_year_emissions <= Decimal("0"):
            return TargetValidityStatus.INVALID_ARR

        arr = self.calculate_arr(target)
        if arr < Decimal("0"):
            return TargetValidityStatus.INVALID_ARR

        return TargetValidityStatus.VALID

    def _select_best_target(
        self,
        entity: PortfolioEntity,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> Optional[EmissionsTarget]:
        """Select the best matching target for a scope/timeframe combination.

        When multiple targets match, select the one with the highest ARR
        (most ambitious).

        Args:
            entity: Portfolio entity.
            scope: Target scope filter.
            timeframe: Target timeframe filter.

        Returns:
            Best matching EmissionsTarget or None.
        """
        candidates: List[Tuple[Decimal, EmissionsTarget]] = []
        for target in entity.targets:
            if target.scope != scope:
                continue

            target_tf = self._classify_target_timeframe(target)
            if target_tf != timeframe:
                continue

            validity = self._validate_target(target)
            if validity != TargetValidityStatus.VALID:
                continue

            arr = self.calculate_arr(target)
            candidates.append((arr, target))

        if not candidates:
            return None

        # Select target with highest ARR (most ambitious)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def score_entity(
        self,
        entity_id: str,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> EntityTemperatureScore:
        """Score a single entity for a given scope and timeframe.

        Args:
            entity_id: Entity identifier.
            scope: Scope to score.
            timeframe: Timeframe to score.

        Returns:
            EntityTemperatureScore.

        Raises:
            ValueError: If entity not found.
        """
        entity = self.get_entity(entity_id)
        best_target = self._select_best_target(entity, scope, timeframe)

        if best_target is None:
            score = EntityTemperatureScore(
                entity_id=entity_id,
                entity_name=entity.entity_name,
                scope=scope,
                timeframe=timeframe,
                target_id=None,
                annual_reduction_rate=Decimal("0"),
                temperature_score=self.config.default_temperature,
                temperature_band=TemperatureBand.NO_TARGET,
                validity_status=TargetValidityStatus.MISSING,
                is_default_score=True,
            )
            score.provenance_hash = _compute_hash(score)
            return score

        arr = self.calculate_arr(best_target)
        temperature = self.map_arr_to_temperature(arr)
        band = self.classify_temperature_band(temperature)
        validity = self._validate_target(best_target)

        score = EntityTemperatureScore(
            entity_id=entity_id,
            entity_name=entity.entity_name,
            scope=scope,
            timeframe=timeframe,
            target_id=best_target.target_id,
            annual_reduction_rate=arr,
            temperature_score=temperature,
            temperature_band=band,
            validity_status=validity,
            is_default_score=False,
        )
        score.provenance_hash = _compute_hash(score)

        logger.info(
            "Entity %s (%s/%s): ARR=%.2f%%, temp=%.2fC, band=%s",
            entity_id, scope.value, timeframe.value,
            float(arr), float(temperature), band.value,
        )
        return score

    def score_all_entities(
        self,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> List[EntityTemperatureScore]:
        """Score all entities in the portfolio.

        Args:
            scope: Scope to score.
            timeframe: Timeframe to score.

        Returns:
            List of EntityTemperatureScore for all entities.
        """
        scores: List[EntityTemperatureScore] = []
        for entity_id in self._entities:
            score = self.score_entity(entity_id, scope, timeframe)
            scores.append(score)
        logger.info(
            "Scored %d entities for %s/%s",
            len(scores), scope.value, timeframe.value,
        )
        return scores

    # -------------------------------------------------------------------
    # Portfolio Aggregation
    # -------------------------------------------------------------------

    def _get_entity_weight(
        self, entity: PortfolioEntity, score_type: ScoreType
    ) -> Decimal:
        """Get the weight of an entity for a given aggregation method.

        Args:
            entity: Portfolio entity.
            score_type: Aggregation method.

        Returns:
            Weight value for the entity.
        """
        ownership = entity.ownership_pct / Decimal("100")

        if score_type == ScoreType.WATS:
            return entity.revenue
        elif score_type == ScoreType.TETS:
            return entity.total_emissions
        elif score_type == ScoreType.MOTS:
            return entity.market_cap * ownership
        elif score_type == ScoreType.EOTS:
            return entity.enterprise_value * ownership
        elif score_type == ScoreType.ECOTS:
            return entity.ev_plus_cash * ownership
        elif score_type == ScoreType.AOTS:
            return entity.total_invested_capital * ownership
        else:
            return entity.revenue

    def aggregate_portfolio(
        self,
        score_type: ScoreType,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> PortfolioTemperatureScore:
        """Aggregate entity scores into a portfolio temperature score.

        Formula:
            portfolio_temp = sum(entity_weight_i * entity_temp_i) / sum(entity_weight_i)

        Args:
            score_type: Aggregation method (WATS, TETS, MOTS, etc.).
            scope: Scope to aggregate.
            timeframe: Timeframe to aggregate.

        Returns:
            PortfolioTemperatureScore.
        """
        entity_scores = self.score_all_entities(scope, timeframe)

        total_weight = Decimal("0")
        weighted_sum = Decimal("0")
        entities_with_targets = 0
        entities_defaulted = 0
        covered_weight = Decimal("0")

        for es in entity_scores:
            entity = self._entities[es.entity_id]
            weight = self._get_entity_weight(entity, score_type)
            total_weight += weight
            weighted_sum += weight * es.temperature_score

            if es.is_default_score:
                entities_defaulted += 1
            else:
                entities_with_targets += 1
                covered_weight += weight

        portfolio_temp = _safe_divide(weighted_sum, total_weight, self.config.default_temperature)
        portfolio_temp = _round_val(portfolio_temp, self.config.score_precision)
        coverage = _safe_pct(covered_weight, total_weight)
        band = self.classify_temperature_band(portfolio_temp)

        result = PortfolioTemperatureScore(
            score_type=score_type,
            scope=scope,
            timeframe=timeframe,
            temperature_score=portfolio_temp,
            temperature_band=band,
            entities_count=len(entity_scores),
            entities_with_targets=entities_with_targets,
            entities_defaulted=entities_defaulted,
            coverage_pct=coverage,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Portfolio %s (%s/%s): temp=%.2fC, band=%s, coverage=%.1f%%",
            score_type.value, scope.value, timeframe.value,
            float(portfolio_temp), band.value, float(coverage),
        )
        return result

    def aggregate_all_methods(
        self,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> List[PortfolioTemperatureScore]:
        """Aggregate portfolio using all six methods.

        Args:
            scope: Scope to aggregate.
            timeframe: Timeframe to aggregate.

        Returns:
            List of PortfolioTemperatureScore, one per method.
        """
        results: List[PortfolioTemperatureScore] = []
        for st in ScoreType:
            result = self.aggregate_portfolio(st, scope, timeframe)
            results.append(result)
        return results

    # -------------------------------------------------------------------
    # Contribution Analysis
    # -------------------------------------------------------------------

    def analyze_contributions(
        self,
        score_type: ScoreType,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> List[ContributionEntry]:
        """Analyze how each entity contributes to the portfolio temperature.

        Args:
            score_type: Aggregation method.
            scope: Scope of analysis.
            timeframe: Timeframe of analysis.

        Returns:
            List of ContributionEntry sorted by contribution (descending).
        """
        entity_scores = self.score_all_entities(scope, timeframe)

        total_weight = Decimal("0")
        entries: List[Dict[str, Any]] = []

        for es in entity_scores:
            entity = self._entities[es.entity_id]
            weight = self._get_entity_weight(entity, score_type)
            total_weight += weight
            entries.append({
                "entity_id": es.entity_id,
                "entity_name": es.entity_name,
                "weight": weight,
                "temperature": es.temperature_score,
            })

        contributions: List[ContributionEntry] = []
        total_contribution = Decimal("0")

        for entry in entries:
            weight = entry["weight"]
            weight_pct = _safe_pct(weight, total_weight)
            contribution = _safe_divide(
                weight * entry["temperature"],
                total_weight,
                Decimal("0"),
            )
            contribution = _round_val(contribution, self.config.score_precision)
            total_contribution += contribution

            contributions.append(ContributionEntry(
                entity_id=entry["entity_id"],
                entity_name=entry["entity_name"],
                weight=weight,
                weight_pct=weight_pct,
                entity_temperature=entry["temperature"],
                contribution=contribution,
                contribution_pct=Decimal("0"),  # Set after all computed
            ))

        # Set contribution percentages
        for c in contributions:
            if total_contribution > Decimal("0"):
                c.contribution_pct = _safe_pct(c.contribution, total_contribution)
            c.provenance_hash = _compute_hash(c)

        # Sort descending by contribution
        contributions.sort(key=lambda x: x.contribution, reverse=True)

        logger.info(
            "Contribution analysis (%s/%s/%s): %d entities, total_contribution=%.4fC",
            score_type.value, scope.value, timeframe.value,
            len(contributions), float(total_contribution),
        )
        return contributions

    # -------------------------------------------------------------------
    # What-If Analysis
    # -------------------------------------------------------------------

    def run_what_if(
        self,
        scenario: WhatIfScenario,
        score_type: ScoreType,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> WhatIfResult:
        """Run a what-if scenario to evaluate the impact of a target change.

        Temporarily modifies an entity's temperature score to evaluate
        how it would affect the portfolio temperature.

        Args:
            scenario: What-if scenario definition.
            score_type: Portfolio aggregation method.
            scope: Scope of analysis.
            timeframe: Timeframe of analysis.

        Returns:
            WhatIfResult with original and modified temperatures.

        Raises:
            ValueError: If entity not found.
        """
        # Get baseline portfolio temperature
        baseline = self.aggregate_portfolio(score_type, scope, timeframe)
        original_portfolio_temp = baseline.temperature_score

        # Get original entity score
        original_entity_score = self.score_entity(scenario.entity_id, scope, timeframe)
        original_entity_temp = original_entity_score.temperature_score

        # Calculate modified entity temperature
        if scenario.new_annual_reduction_rate is not None:
            modified_entity_temp = self.map_arr_to_temperature(scenario.new_annual_reduction_rate)
        else:
            modified_entity_temp = original_entity_temp

        # Calculate modified portfolio temperature
        entity_scores = self.score_all_entities(scope, timeframe)
        total_weight = Decimal("0")
        weighted_sum = Decimal("0")

        for es in entity_scores:
            entity = self._entities[es.entity_id]
            weight = self._get_entity_weight(entity, score_type)
            total_weight += weight

            if es.entity_id == scenario.entity_id:
                weighted_sum += weight * modified_entity_temp
            else:
                weighted_sum += weight * es.temperature_score

        modified_portfolio_temp = _safe_divide(
            weighted_sum, total_weight, self.config.default_temperature
        )
        modified_portfolio_temp = _round_val(modified_portfolio_temp, self.config.score_precision)

        temp_change = modified_portfolio_temp - original_portfolio_temp
        temp_change = _round_val(temp_change, self.config.score_precision)

        improvement_pct = Decimal("0")
        if original_portfolio_temp > Decimal("0"):
            improvement_pct = _round_val(
                (temp_change / original_portfolio_temp) * Decimal("100"), 2
            )

        result = WhatIfResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.scenario_name,
            original_portfolio_temperature=original_portfolio_temp,
            modified_portfolio_temperature=modified_portfolio_temp,
            temperature_change=temp_change,
            improvement_pct=improvement_pct,
            entity_id=scenario.entity_id,
            original_entity_temperature=original_entity_temp,
            modified_entity_temperature=modified_entity_temp,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "What-if %s: entity %s ARR change -> portfolio %.4fC -> %.4fC (delta=%.4fC)",
            scenario.scenario_id, scenario.entity_id,
            float(original_portfolio_temp), float(modified_portfolio_temp),
            float(temp_change),
        )
        return result

    def run_what_if_batch(
        self,
        scenarios: List[WhatIfScenario],
        score_type: ScoreType,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> List[WhatIfResult]:
        """Run multiple what-if scenarios.

        Args:
            scenarios: List of WhatIfScenario definitions.
            score_type: Portfolio aggregation method.
            scope: Scope of analysis.
            timeframe: Timeframe of analysis.

        Returns:
            List of WhatIfResult.
        """
        results: List[WhatIfResult] = []
        for scenario in scenarios:
            result = self.run_what_if(scenario, score_type, scope, timeframe)
            results.append(result)
        logger.info("Completed %d what-if scenarios", len(results))
        return results

    # -------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------

    def run_full_assessment(
        self,
        score_type: ScoreType = ScoreType.WATS,
        scope: TargetScope = TargetScope.S1S2,
        timeframe: TargetTimeframe = TargetTimeframe.NEAR_TERM,
        what_if_scenarios: Optional[List[WhatIfScenario]] = None,
    ) -> TemperatureResult:
        """Run a complete temperature scoring assessment.

        Performs entity scoring, portfolio aggregation, contribution
        analysis, and optional what-if analysis in a single call.

        Args:
            score_type: Portfolio aggregation method.
            scope: Scope of assessment.
            timeframe: Timeframe of assessment.
            what_if_scenarios: Optional what-if scenarios.

        Returns:
            Complete TemperatureResult.
        """
        logger.info(
            "Running full assessment: %s/%s/%s, %d entities",
            score_type.value, scope.value, timeframe.value, len(self._entities),
        )

        # Step 1: Score all entities
        entity_scores = self.score_all_entities(scope, timeframe)

        # Step 2: Aggregate portfolio (all methods)
        portfolio_scores = self.aggregate_all_methods(scope, timeframe)

        # Step 3: Contribution analysis
        contributions = self.analyze_contributions(score_type, scope, timeframe)

        # Step 4: What-if analysis
        what_if_results: List[WhatIfResult] = []
        if what_if_scenarios:
            what_if_results = self.run_what_if_batch(
                what_if_scenarios, score_type, scope, timeframe
            )

        result = TemperatureResult(
            entity_scores=entity_scores,
            portfolio_scores=portfolio_scores,
            contribution_analysis=contributions,
            what_if_results=what_if_results,
            entities_assessed=len(entity_scores),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Full assessment complete: %d entities, %d methods, %d contributions, %d what-ifs",
            len(entity_scores), len(portfolio_scores),
            len(contributions), len(what_if_results),
        )
        return result

    # -------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of the current portfolio.

        Returns:
            Dictionary with entity count, sector breakdown, and coverage stats.
        """
        sectors: Dict[str, int] = {}
        total_emissions = Decimal("0")
        total_revenue = Decimal("0")
        entities_with_targets = 0

        for entity in self._entities.values():
            sector = entity.sector or "Unknown"
            sectors[sector] = sectors.get(sector, 0) + 1
            total_emissions += entity.total_emissions
            total_revenue += entity.revenue
            if entity.targets:
                entities_with_targets += 1

        return {
            "entity_count": len(self._entities),
            "entities_with_targets": entities_with_targets,
            "target_coverage_pct": str(_safe_pct(
                _decimal(entities_with_targets),
                _decimal(len(self._entities)),
            )),
            "total_emissions_tco2e": str(total_emissions),
            "total_revenue": str(total_revenue),
            "sectors": sectors,
            "provenance_hash": _compute_hash(sectors),
        }

    def get_temperature_distribution(
        self,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> Dict[str, int]:
        """Get distribution of entities across temperature bands.

        Args:
            scope: Scope of analysis.
            timeframe: Timeframe of analysis.

        Returns:
            Dictionary mapping TemperatureBand name to entity count.
        """
        entity_scores = self.score_all_entities(scope, timeframe)
        distribution: Dict[str, int] = {band.value: 0 for band in TemperatureBand}

        for es in entity_scores:
            distribution[es.temperature_band.value] += 1

        return distribution

    def clear(self) -> None:
        """Clear all entities from the engine."""
        self._entities.clear()
        logger.info("TemperatureScoringEngine cleared")
