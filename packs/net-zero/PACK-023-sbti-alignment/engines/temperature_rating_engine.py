# -*- coding: utf-8 -*-
"""
TemperatureRatingEngine - PACK-023 SBTi Alignment Engine 6
=============================================================

Implements the SBTi Temperature Rating v2.0 methodology to assign
temperature alignment scores to individual company targets and to
aggregate those scores at the portfolio level using six aggregation
methods (WATS, TETS, MOTS, EOTS, ECOTS, AOTS).

This engine translates corporate GHG emissions reduction targets into
an intuitive temperature score expressed in degrees Celsius.  The score
represents the global temperature rise that would result if the entire
global economy adopted the same level of ambition as the assessed
entity.  Scores are derived by mapping the Annual linear Reduction Rate
(ARR) onto a calibrated piecewise-linear temperature mapping curve
based on IPCC AR6 carbon budgets and integrated assessment models.

Temperature Mapping (Piecewise-Linear):
    7.0%/yr  -> 1.20 C  (most ambitious)
    6.0%/yr  -> 1.25 C
    5.0%/yr  -> 1.40 C
    4.2%/yr  -> 1.50 C  (1.5 C aligned)
    3.5%/yr  -> 1.60 C
    2.5%/yr  -> 1.80 C  (Well-Below-2 C)
    1.5%/yr  -> 2.00 C
    1.0%/yr  -> 2.50 C
    0.5%/yr  -> 2.80 C
    0.0%/yr  -> 3.20 C  (default / no target)

Six Portfolio Aggregation Methods (SBTi TR v2.0):
    WATS  - Weighted Average Temperature Score (by revenue weight)
    TETS  - Total Emissions Temperature Score (by total emissions weight)
    MOTS  - Market Owned Temperature Score (by market cap ownership)
    EOTS  - Enterprise Owned Temperature Score (by enterprise value ownership)
    ECOTS - Enterprise Value + Cash Temperature Score
    AOTS  - All-Owned Temperature Score (by total invested capital)

Target Scopes:
    S1S2   - Scope 1 + Scope 2 combined
    S3     - Scope 3
    S1S2S3 - All scopes combined

Target Timeframes:
    NEAR_TERM - 5-10 year targets
    LONG_TERM - targets beyond 2035

Regulatory References:
    - SBTi Temperature Rating Methodology v2.0 (2024)
    - SBTi Corporate Manual V5.3 (2024)
    - SBTi Corporate Net-Zero Standard V1.3 (2024)
    - IPCC AR6 WG3 Chapter 3: Mitigation Pathways
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - Paris Agreement Article 2.1(a)

Zero-Hallucination:
    - Temperature mapping uses a deterministic piecewise-linear lookup table
    - All ARR calculations use Decimal arithmetic with ROUND_HALF_UP
    - Portfolio aggregation uses deterministic weighted-average formulas
    - Default score of 3.20 C applied only for entities without any valid target
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-023 SBTi Alignment
Engine:  6 of 10
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
# Reference Data: SBTi Temperature Mapping Table (PACK-023 calibration)
# ---------------------------------------------------------------------------
# Piecewise-linear mapping from Annual Reduction Rate (% per year)
# to temperature score (degrees C).  Calibrated against IPCC AR6
# carbon budgets and SBTi methodology.

TEMPERATURE_MAPPING_TABLE: List[Tuple[Decimal, Decimal]] = [
    # (annual_reduction_rate_pct, temperature_c)
    (Decimal("7.0"), Decimal("1.20")),
    (Decimal("6.0"), Decimal("1.25")),
    (Decimal("5.0"), Decimal("1.40")),
    (Decimal("4.2"), Decimal("1.50")),
    (Decimal("3.5"), Decimal("1.60")),
    (Decimal("2.5"), Decimal("1.80")),
    (Decimal("1.5"), Decimal("2.00")),
    (Decimal("1.0"), Decimal("2.50")),
    (Decimal("0.5"), Decimal("2.80")),
    (Decimal("0.0"), Decimal("3.20")),
]

# Default temperature for entities without any validated target
DEFAULT_TEMP: Decimal = Decimal("3.20")

# Temperature band boundaries (lower_bound_inclusive, upper_bound_exclusive)
TEMPERATURE_BAND_THRESHOLDS: Dict[str, Tuple[Decimal, Decimal]] = {
    "BELOW_1_5": (Decimal("0"), Decimal("1.50")),
    "AT_1_5": (Decimal("1.50"), Decimal("1.51")),
    "WELL_BELOW_2": (Decimal("1.51"), Decimal("1.80")),
    "BELOW_2": (Decimal("1.80"), Decimal("2.00")),
    "ABOVE_2": (Decimal("2.00"), Decimal("3.20")),
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


class TemperatureBand(str, Enum):
    """Temperature alignment band."""
    BELOW_1_5 = "below_1_5"
    AT_1_5 = "at_1_5"
    WELL_BELOW_2 = "well_below_2"
    BELOW_2 = "below_2"
    ABOVE_2 = "above_2"
    NO_TARGET = "no_target"


class TargetValidityStatus(str, Enum):
    """Validity status of a target for temperature scoring."""
    VALID = "valid"
    EXPIRED = "expired"
    INVALID_ARR = "invalid_arr"
    MISSING = "missing"


class AggregationMethod(str, Enum):
    """Weight basis for portfolio aggregation."""
    REVENUE = "revenue"
    TOTAL_EMISSIONS = "total_emissions"
    MARKET_CAP = "market_cap"
    ENTERPRISE_VALUE = "enterprise_value"
    EV_PLUS_CASH = "ev_plus_cash"
    TOTAL_INVESTED_CAPITAL = "total_invested_capital"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class CompanyTarget(BaseModel):
    """A single emissions reduction target for a company."""
    target_id: str = Field(default_factory=_new_uuid, description="Target identifier")
    entity_id: str = Field(default="", description="Owning entity identifier")
    scope: TargetScope = Field(description="Target scope (S1S2, S3, S1S2S3)")
    timeframe: TargetTimeframe = Field(description="Near-term or long-term")
    base_year: int = Field(description="Target base year")
    target_year: int = Field(description="Target end year")
    base_year_emissions: Decimal = Field(description="Base year emissions (tCO2e)")
    target_year_emissions: Decimal = Field(description="Target year emissions (tCO2e)")
    reduction_pct: Decimal = Field(default=Decimal("0"), description="Reduction %")
    is_sbti_validated: bool = Field(default=False, description="SBTi validated flag")
    is_net_zero_aligned: bool = Field(default=False, description="Net-zero aligned flag")
    created_at: datetime = Field(default_factory=_utcnow, description="Creation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("base_year_emissions", "target_year_emissions",
                     "reduction_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class TemperatureInput(BaseModel):
    """Input data for temperature rating assessment."""
    assessment_id: str = Field(default_factory=_new_uuid, description="Assessment ID")
    entities: List[CompanyScoreInput] = Field(
        default_factory=list, description="Entities to assess"
    )
    score_types: List[ScoreType] = Field(
        default_factory=lambda: [ScoreType.WATS],
        description="Aggregation methods to compute",
    )
    scopes: List[TargetScope] = Field(
        default_factory=lambda: [TargetScope.S1S2],
        description="Scopes to assess",
    )
    timeframes: List[TargetTimeframe] = Field(
        default_factory=lambda: [TargetTimeframe.NEAR_TERM],
        description="Timeframes to assess",
    )
    what_if_scenarios: List[WhatIfScenario] = Field(
        default_factory=list, description="What-if scenarios"
    )
    requested_at: datetime = Field(default_factory=_utcnow, description="Request timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class CompanyScoreInput(BaseModel):
    """Company-level input for temperature scoring."""
    entity_id: str = Field(default_factory=_new_uuid, description="Entity identifier")
    entity_name: str = Field(description="Entity / company name")
    sector: str = Field(default="", description="Industry sector")
    country: str = Field(default="", description="Country code (ISO 3166-1)")
    revenue: Decimal = Field(default=Decimal("0"), description="Revenue")
    total_emissions: Decimal = Field(default=Decimal("0"), description="Total tCO2e")
    scope1_emissions: Decimal = Field(default=Decimal("0"), description="Scope 1 tCO2e")
    scope2_emissions: Decimal = Field(default=Decimal("0"), description="Scope 2 tCO2e")
    scope3_emissions: Decimal = Field(default=Decimal("0"), description="Scope 3 tCO2e")
    market_cap: Decimal = Field(default=Decimal("0"), description="Market cap")
    enterprise_value: Decimal = Field(default=Decimal("0"), description="Enterprise value")
    ev_plus_cash: Decimal = Field(default=Decimal("0"), description="EV + cash")
    total_invested_capital: Decimal = Field(default=Decimal("0"), description="Total invested capital")
    ownership_pct: Decimal = Field(default=Decimal("100"), description="Ownership %")
    investment_value: Decimal = Field(default=Decimal("0"), description="Investment value")
    targets: List[CompanyTarget] = Field(
        default_factory=list, description="Reduction targets"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("revenue", "total_emissions", "scope1_emissions",
                     "scope2_emissions", "scope3_emissions", "market_cap",
                     "enterprise_value", "ev_plus_cash",
                     "total_invested_capital", "ownership_pct",
                     "investment_value", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class CompanyScore(BaseModel):
    """Temperature score for a single company."""
    entity_id: str = Field(description="Entity identifier")
    entity_name: str = Field(default="", description="Entity name")
    scope: TargetScope = Field(description="Scope of assessment")
    timeframe: TargetTimeframe = Field(description="Timeframe of assessment")
    target_id: Optional[str] = Field(default=None, description="Associated target ID")
    arr_pct: Decimal = Field(description="Annual reduction rate (% per year)")
    temperature_c: Decimal = Field(description="Temperature score (degrees C)")
    band: TemperatureBand = Field(description="Temperature alignment band")
    validity_status: TargetValidityStatus = Field(description="Target validity")
    is_default_score: bool = Field(default=False, description="Used default 3.20 C")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("arr_pct", "temperature_c", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class ContributionEntry(BaseModel):
    """Contribution of one entity to the portfolio temperature."""
    entity_id: str = Field(description="Entity identifier")
    entity_name: str = Field(default="", description="Entity name")
    weight: Decimal = Field(description="Raw weight in portfolio")
    weight_pct: Decimal = Field(description="Weight as percentage of total")
    entity_temperature: Decimal = Field(description="Entity temperature score (C)")
    contribution: Decimal = Field(description="Weighted contribution to portfolio")
    contribution_pct: Decimal = Field(description="Contribution as % of total score")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("weight", "weight_pct", "entity_temperature",
                     "contribution", "contribution_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class PortfolioScore(BaseModel):
    """Aggregated portfolio temperature score."""
    portfolio_id: str = Field(default_factory=_new_uuid, description="Portfolio ID")
    method: ScoreType = Field(description="Aggregation method used")
    scope: TargetScope = Field(description="Scope of aggregation")
    timeframe: TargetTimeframe = Field(description="Timeframe of aggregation")
    temperature_c: Decimal = Field(description="Portfolio temperature (C)")
    band: TemperatureBand = Field(description="Temperature alignment band")
    coverage_pct: Decimal = Field(default=Decimal("0"), description="Target coverage %")
    entities_count: int = Field(default=0, description="Total entities")
    entities_with_targets: int = Field(default=0, description="Entities with targets")
    entities_defaulted: int = Field(default=0, description="Entities with default score")
    entity_contributions: List[ContributionEntry] = Field(
        default_factory=list, description="Entity-level contributions"
    )
    methodology_version: str = Field(
        default="SBTi TR v2.0", description="Methodology version"
    )
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("temperature_c", "coverage_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class WhatIfScenario(BaseModel):
    """What-if scenario for evaluating target changes."""
    scenario_id: str = Field(default_factory=_new_uuid, description="Scenario ID")
    scenario_name: str = Field(default="", description="Human-readable name")
    entity_id: str = Field(description="Entity to modify")
    new_arr_pct: Optional[Decimal] = Field(
        default=None, description="Override ARR (% per year)"
    )
    new_target_scope: Optional[TargetScope] = Field(
        default=None, description="Override target scope"
    )
    new_target_year: Optional[int] = Field(
        default=None, description="Override target year"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("new_arr_pct", mode="before")
    @classmethod
    def _coerce_decimal_opt(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)


class WhatIfResult(BaseModel):
    """Result of a what-if analysis."""
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    scenario_id: str = Field(description="Scenario ID")
    scenario_name: str = Field(default="", description="Scenario name")
    original_portfolio_temperature: Decimal = Field(description="Before temp (C)")
    modified_portfolio_temperature: Decimal = Field(description="After temp (C)")
    temperature_change: Decimal = Field(description="Delta (C)")
    improvement_pct: Decimal = Field(description="Improvement %")
    entity_id: str = Field(description="Modified entity")
    original_entity_temperature: Decimal = Field(description="Before entity temp (C)")
    modified_entity_temperature: Decimal = Field(description="After entity temp (C)")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("original_portfolio_temperature", "modified_portfolio_temperature",
                     "temperature_change", "improvement_pct",
                     "original_entity_temperature", "modified_entity_temperature",
                     mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class TemperatureResult(BaseModel):
    """Complete temperature rating assessment result."""
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    company_scores: List[CompanyScore] = Field(
        default_factory=list, description="Per-company temperature scores"
    )
    portfolio_scores: List[PortfolioScore] = Field(
        default_factory=list, description="Portfolio-level scores by method"
    )
    what_if_results: List[WhatIfResult] = Field(
        default_factory=list, description="What-if results"
    )
    entities_assessed: int = Field(default=0, description="Total entities assessed")
    methodology_version: str = Field(
        default="SBTi TR v2.0", description="Methodology version"
    )
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------


class TemperatureRatingConfig(BaseModel):
    """Configuration for the TemperatureRatingEngine."""
    default_temperature: Decimal = Field(
        default=DEFAULT_TEMP,
        description="Default temperature for entities without targets (C)",
    )
    min_target_years: int = Field(
        default=5, description="Minimum target duration (years)"
    )
    max_target_years_near_term: int = Field(
        default=10, description="Max target duration for near-term (years)"
    )
    long_term_start_year: int = Field(
        default=2035, description="Year threshold for long-term classification"
    )
    default_aggregation: ScoreType = Field(
        default=ScoreType.WATS, description="Default portfolio aggregation method"
    )
    score_precision: int = Field(
        default=4, description="Decimal places for temperature scores"
    )
    current_year: int = Field(
        default=2026, description="Current reporting year"
    )


# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

CompanyTarget.model_rebuild()
CompanyScoreInput.model_rebuild()
TemperatureInput.model_rebuild()
CompanyScore.model_rebuild()
ContributionEntry.model_rebuild()
PortfolioScore.model_rebuild()
WhatIfScenario.model_rebuild()
WhatIfResult.model_rebuild()
TemperatureResult.model_rebuild()
TemperatureRatingConfig.model_rebuild()


# ---------------------------------------------------------------------------
# TemperatureRatingEngine
# ---------------------------------------------------------------------------


class TemperatureRatingEngine:
    """
    SBTi Temperature Rating v2.0 engine for PACK-023.

    Translates corporate emissions reduction targets into temperature
    alignment scores and aggregates them at the portfolio level using
    six methods: WATS, TETS, MOTS, EOTS, ECOTS, AOTS.

    Attributes:
        config: Engine configuration.
        _entities: In-memory entity store keyed by entity_id.

    Example:
        >>> engine = TemperatureRatingEngine()
        >>> entity = CompanyScoreInput(entity_name="Acme Corp", revenue=1000, ...)
        >>> engine.add_entity(entity)
        >>> score = engine.score_company(entity.entity_id, TargetScope.S1S2, TargetTimeframe.NEAR_TERM)
        >>> portfolio = engine.aggregate_portfolio(ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TemperatureRatingEngine.

        Args:
            config: Optional configuration dictionary or TemperatureRatingConfig.
        """
        if config and isinstance(config, dict):
            self.config = TemperatureRatingConfig(**config)
        elif config and isinstance(config, TemperatureRatingConfig):
            self.config = config
        else:
            self.config = TemperatureRatingConfig()

        self._entities: Dict[str, CompanyScoreInput] = {}
        logger.info("TemperatureRatingEngine initialized (v%s)", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Entity Management
    # -------------------------------------------------------------------

    def add_entity(self, entity: CompanyScoreInput) -> CompanyScoreInput:
        """Add a company entity for temperature scoring.

        Args:
            entity: CompanyScoreInput with targets.

        Returns:
            Entity with computed provenance hash.
        """
        entity.provenance_hash = _compute_hash(entity)
        self._entities[entity.entity_id] = entity
        logger.info("Added entity %s: %s", entity.entity_id, entity.entity_name)
        return entity

    def add_entities(self, entities: List[CompanyScoreInput]) -> int:
        """Add multiple entities for scoring.

        Args:
            entities: List of CompanyScoreInput objects.

        Returns:
            Number of entities added.
        """
        for entity in entities:
            self.add_entity(entity)
        logger.info("Added %d entities", len(entities))
        return len(entities)

    def get_entity(self, entity_id: str) -> CompanyScoreInput:
        """Retrieve an entity by ID.

        Args:
            entity_id: Entity identifier.

        Returns:
            CompanyScoreInput.

        Raises:
            ValueError: If entity not found.
        """
        if entity_id not in self._entities:
            raise ValueError(f"Entity {entity_id} not found")
        return self._entities[entity_id]

    def list_entities(self) -> List[CompanyScoreInput]:
        """List all entities in the engine.

        Returns:
            List of CompanyScoreInput.
        """
        return list(self._entities.values())

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity by ID.

        Args:
            entity_id: Entity identifier.

        Returns:
            True if removed, False if not found.
        """
        if entity_id in self._entities:
            del self._entities[entity_id]
            logger.info("Removed entity %s", entity_id)
            return True
        return False

    # -------------------------------------------------------------------
    # Annual Reduction Rate
    # -------------------------------------------------------------------

    def calculate_arr(self, target: CompanyTarget) -> Decimal:
        """Calculate the Annual linear Reduction Rate (ARR) for a target.

        The ARR represents the average annual percentage reduction in
        emissions implied by the target over its duration.

        Formula:
            ARR = (1 - (target_emissions / base_emissions))
                  / (target_year - base_year) * 100

        Args:
            target: CompanyTarget to evaluate.

        Returns:
            Annual reduction rate as Decimal percentage (e.g. 4.2 = 4.2%/yr).
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

    def map_to_temperature(self, arr: Decimal) -> Decimal:
        """Map an annual reduction rate to a temperature score.

        Uses piecewise-linear interpolation on the SBTi temperature
        mapping table.  ARR values above the table maximum yield the
        minimum temperature; values at or below zero yield default 3.20 C.

        Args:
            arr: Annual reduction rate (% per year).

        Returns:
            Temperature score in degrees Celsius.
        """
        if arr <= Decimal("0"):
            return self.config.default_temperature

        table = TEMPERATURE_MAPPING_TABLE

        # ARR above highest anchor -> coldest temperature
        if arr >= table[0][0]:
            return table[0][1]

        # Walk through the table (sorted descending by ARR) to find bracket
        for i in range(len(table) - 1):
            upper_arr, upper_temp = table[i]
            lower_arr, lower_temp = table[i + 1]

            if lower_arr <= arr <= upper_arr:
                arr_range = upper_arr - lower_arr
                if arr_range == Decimal("0"):
                    return upper_temp
                fraction = (arr - lower_arr) / arr_range
                temp = lower_temp - fraction * (lower_temp - upper_temp)
                return _round_val(temp, self.config.score_precision)

        # Below lowest non-zero entry
        return self.config.default_temperature

    def classify_band(self, temperature: Decimal) -> TemperatureBand:
        """Classify a temperature score into a band.

        Args:
            temperature: Temperature score in degrees Celsius.

        Returns:
            TemperatureBand enum value.
        """
        if temperature < Decimal("1.50"):
            return TemperatureBand.BELOW_1_5
        elif temperature == Decimal("1.50"):
            return TemperatureBand.AT_1_5
        elif temperature <= Decimal("1.80"):
            return TemperatureBand.WELL_BELOW_2
        elif temperature < Decimal("2.00"):
            return TemperatureBand.BELOW_2
        elif temperature < Decimal("3.20"):
            return TemperatureBand.ABOVE_2
        else:
            return TemperatureBand.NO_TARGET

    # -------------------------------------------------------------------
    # Target Selection
    # -------------------------------------------------------------------

    def _classify_target_timeframe(self, target: CompanyTarget) -> TargetTimeframe:
        """Determine effective timeframe of a target.

        Args:
            target: Target to classify.

        Returns:
            TargetTimeframe.NEAR_TERM or TargetTimeframe.LONG_TERM.
        """
        if target.target_year >= self.config.long_term_start_year:
            return TargetTimeframe.LONG_TERM
        return TargetTimeframe.NEAR_TERM

    def _validate_target(self, target: CompanyTarget) -> TargetValidityStatus:
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

        # Check if target has expired
        if target.target_year < self.config.current_year:
            return TargetValidityStatus.EXPIRED

        arr = self.calculate_arr(target)
        if arr < Decimal("0"):
            return TargetValidityStatus.INVALID_ARR

        return TargetValidityStatus.VALID

    def _select_best_target(
        self,
        entity: CompanyScoreInput,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> Optional[CompanyTarget]:
        """Select the best matching target for a scope/timeframe pair.

        When multiple targets match, select the most ambitious one
        (highest ARR).

        Args:
            entity: Company entity.
            scope: Target scope filter.
            timeframe: Target timeframe filter.

        Returns:
            Best CompanyTarget or None.
        """
        candidates: List[Tuple[Decimal, CompanyTarget]] = []

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

        # Sort descending by ARR, pick most ambitious
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # -------------------------------------------------------------------
    # Company Scoring
    # -------------------------------------------------------------------

    def score_company(
        self,
        entity_id: str,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> CompanyScore:
        """Score a single company for a given scope and timeframe.

        Args:
            entity_id: Entity identifier.
            scope: Scope to score.
            timeframe: Timeframe to score.

        Returns:
            CompanyScore.

        Raises:
            ValueError: If entity not found.
        """
        entity = self.get_entity(entity_id)
        best_target = self._select_best_target(entity, scope, timeframe)

        if best_target is None:
            score = CompanyScore(
                entity_id=entity_id,
                entity_name=entity.entity_name,
                scope=scope,
                timeframe=timeframe,
                target_id=None,
                arr_pct=Decimal("0"),
                temperature_c=self.config.default_temperature,
                band=TemperatureBand.NO_TARGET,
                validity_status=TargetValidityStatus.MISSING,
                is_default_score=True,
            )
            score.provenance_hash = _compute_hash(score)
            return score

        arr = self.calculate_arr(best_target)
        temperature = self.map_to_temperature(arr)
        band = self.classify_band(temperature)
        validity = self._validate_target(best_target)

        score = CompanyScore(
            entity_id=entity_id,
            entity_name=entity.entity_name,
            scope=scope,
            timeframe=timeframe,
            target_id=best_target.target_id,
            arr_pct=arr,
            temperature_c=temperature,
            band=band,
            validity_status=validity,
            is_default_score=False,
        )
        score.provenance_hash = _compute_hash(score)

        logger.info(
            "Company %s (%s/%s): ARR=%.2f%%, temp=%.2fC, band=%s",
            entity_id, scope.value, timeframe.value,
            float(arr), float(temperature), band.value,
        )
        return score

    def score_all_companies(
        self,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> List[CompanyScore]:
        """Score all entities in the portfolio.

        Args:
            scope: Scope to score.
            timeframe: Timeframe to score.

        Returns:
            List of CompanyScore for all entities.
        """
        scores: List[CompanyScore] = []
        for entity_id in self._entities:
            score = self.score_company(entity_id, scope, timeframe)
            scores.append(score)
        logger.info(
            "Scored %d companies for %s/%s",
            len(scores), scope.value, timeframe.value,
        )
        return scores

    # -------------------------------------------------------------------
    # Portfolio Aggregation
    # -------------------------------------------------------------------

    def _get_entity_weight(
        self, entity: CompanyScoreInput, score_type: ScoreType
    ) -> Decimal:
        """Get the weight of an entity for a given aggregation method.

        Args:
            entity: Company entity.
            score_type: Aggregation method.

        Returns:
            Weight value.
        """
        ownership = entity.ownership_pct / Decimal("100")

        weight_map = {
            ScoreType.WATS: entity.revenue,
            ScoreType.TETS: entity.total_emissions,
            ScoreType.MOTS: entity.market_cap * ownership,
            ScoreType.EOTS: entity.enterprise_value * ownership,
            ScoreType.ECOTS: entity.ev_plus_cash * ownership,
            ScoreType.AOTS: entity.total_invested_capital * ownership,
        }
        return weight_map.get(score_type, entity.revenue)

    def aggregate_portfolio(
        self,
        score_type: ScoreType,
        scope: TargetScope,
        timeframe: TargetTimeframe,
    ) -> PortfolioScore:
        """Aggregate entity scores into a portfolio temperature score.

        Formula:
            portfolio_temp = sum(w_i * temp_i) / sum(w_i)

        Args:
            score_type: Aggregation method (WATS, TETS, etc.).
            scope: Scope to aggregate.
            timeframe: Timeframe to aggregate.

        Returns:
            PortfolioScore.
        """
        company_scores = self.score_all_companies(scope, timeframe)

        total_weight = Decimal("0")
        weighted_sum = Decimal("0")
        entities_with_targets = 0
        entities_defaulted = 0
        covered_weight = Decimal("0")

        for cs in company_scores:
            entity = self._entities[cs.entity_id]
            weight = self._get_entity_weight(entity, score_type)
            total_weight += weight
            weighted_sum += weight * cs.temperature_c

            if cs.is_default_score:
                entities_defaulted += 1
            else:
                entities_with_targets += 1
                covered_weight += weight

        portfolio_temp = _safe_divide(
            weighted_sum, total_weight, self.config.default_temperature
        )
        portfolio_temp = _round_val(portfolio_temp, self.config.score_precision)
        coverage = _safe_pct(covered_weight, total_weight)
        band = self.classify_band(portfolio_temp)

        result = PortfolioScore(
            method=score_type,
            scope=scope,
            timeframe=timeframe,
            temperature_c=portfolio_temp,
            band=band,
            coverage_pct=coverage,
            entities_count=len(company_scores),
            entities_with_targets=entities_with_targets,
            entities_defaulted=entities_defaulted,
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
    ) -> List[PortfolioScore]:
        """Aggregate portfolio using all six methods.

        Args:
            scope: Scope to aggregate.
            timeframe: Timeframe to aggregate.

        Returns:
            List of PortfolioScore, one per aggregation method.
        """
        results: List[PortfolioScore] = []
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
        company_scores = self.score_all_companies(scope, timeframe)

        total_weight = Decimal("0")
        entries: List[Dict[str, Any]] = []

        for cs in company_scores:
            entity = self._entities[cs.entity_id]
            weight = self._get_entity_weight(entity, score_type)
            total_weight += weight
            entries.append({
                "entity_id": cs.entity_id,
                "entity_name": cs.entity_name,
                "weight": weight,
                "temperature": cs.temperature_c,
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
                contribution_pct=Decimal("0"),
            ))

        # Set contribution percentages
        for c in contributions:
            if total_contribution > Decimal("0"):
                c.contribution_pct = _safe_pct(c.contribution, total_contribution)
            c.provenance_hash = _compute_hash(c)

        # Sort descending by contribution
        contributions.sort(key=lambda x: x.contribution, reverse=True)

        logger.info(
            "Contribution analysis (%s/%s/%s): %d entities",
            score_type.value, scope.value, timeframe.value,
            len(contributions),
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
        """Run a what-if scenario to evaluate a target change impact.

        Temporarily modifies an entity's temperature to evaluate the
        effect on portfolio temperature.

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
        # Baseline portfolio temperature
        baseline = self.aggregate_portfolio(score_type, scope, timeframe)
        original_portfolio_temp = baseline.temperature_c

        # Original entity score
        original_score = self.score_company(scenario.entity_id, scope, timeframe)
        original_entity_temp = original_score.temperature_c

        # Modified entity temperature
        if scenario.new_arr_pct is not None:
            modified_entity_temp = self.map_to_temperature(scenario.new_arr_pct)
        else:
            modified_entity_temp = original_entity_temp

        # Recalculate portfolio with modified entity
        company_scores = self.score_all_companies(scope, timeframe)
        total_weight = Decimal("0")
        weighted_sum = Decimal("0")

        for cs in company_scores:
            entity = self._entities[cs.entity_id]
            weight = self._get_entity_weight(entity, score_type)
            total_weight += weight

            if cs.entity_id == scenario.entity_id:
                weighted_sum += weight * modified_entity_temp
            else:
                weighted_sum += weight * cs.temperature_c

        modified_portfolio_temp = _safe_divide(
            weighted_sum, total_weight, self.config.default_temperature
        )
        modified_portfolio_temp = _round_val(
            modified_portfolio_temp, self.config.score_precision
        )

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
            "What-if %s: entity %s -> portfolio %.4fC -> %.4fC (delta=%.4fC)",
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
    # Full Assessment Pipeline
    # -------------------------------------------------------------------

    def run_full_assessment(
        self,
        score_types: Optional[List[ScoreType]] = None,
        scopes: Optional[List[TargetScope]] = None,
        timeframes: Optional[List[TargetTimeframe]] = None,
        what_if_scenarios: Optional[List[WhatIfScenario]] = None,
    ) -> TemperatureResult:
        """Run a complete temperature rating assessment.

        Performs company scoring, portfolio aggregation across all
        requested methods/scopes/timeframes, and optional what-if analysis.

        Args:
            score_types: Aggregation methods (default: all six).
            scopes: Scopes to assess (default: S1S2 only).
            timeframes: Timeframes to assess (default: NEAR_TERM only).
            what_if_scenarios: Optional what-if scenarios.

        Returns:
            Complete TemperatureResult with provenance hash.
        """
        if score_types is None:
            score_types = list(ScoreType)
        if scopes is None:
            scopes = [TargetScope.S1S2]
        if timeframes is None:
            timeframes = [TargetTimeframe.NEAR_TERM]

        logger.info(
            "Running full assessment: %d methods x %d scopes x %d timeframes, %d entities",
            len(score_types), len(scopes), len(timeframes), len(self._entities),
        )

        all_company_scores: List[CompanyScore] = []
        all_portfolio_scores: List[PortfolioScore] = []

        for scope in scopes:
            for timeframe in timeframes:
                # Score all companies
                company_scores = self.score_all_companies(scope, timeframe)
                all_company_scores.extend(company_scores)

                # Aggregate by each requested method
                for st in score_types:
                    contributions = self.analyze_contributions(st, scope, timeframe)
                    portfolio = self.aggregate_portfolio(st, scope, timeframe)
                    portfolio.entity_contributions = contributions
                    all_portfolio_scores.append(portfolio)

        # What-if analysis
        what_if_results: List[WhatIfResult] = []
        if what_if_scenarios:
            default_scope = scopes[0]
            default_tf = timeframes[0]
            default_st = score_types[0]
            what_if_results = self.run_what_if_batch(
                what_if_scenarios, default_st, default_scope, default_tf
            )

        result = TemperatureResult(
            company_scores=all_company_scores,
            portfolio_scores=all_portfolio_scores,
            what_if_results=what_if_results,
            entities_assessed=len(self._entities),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Full assessment complete: %d companies, %d portfolios, %d what-ifs",
            len(all_company_scores), len(all_portfolio_scores),
            len(what_if_results),
        )
        return result

    # -------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of the current portfolio.

        Returns:
            Dictionary with entity count, sector breakdown, coverage stats.
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
            Dict mapping TemperatureBand value to entity count.
        """
        company_scores = self.score_all_companies(scope, timeframe)
        distribution: Dict[str, int] = {band.value: 0 for band in TemperatureBand}
        for cs in company_scores:
            distribution[cs.band.value] += 1
        return distribution

    def get_mapping_table(self) -> List[Dict[str, str]]:
        """Return the temperature mapping table as a list of dicts.

        Returns:
            List of dicts with arr_pct and temperature_c keys.
        """
        return [
            {"arr_pct": str(arr), "temperature_c": str(temp)}
            for arr, temp in TEMPERATURE_MAPPING_TABLE
        ]

    def clear(self) -> None:
        """Clear all entities from the engine."""
        self._entities.clear()
        logger.info("TemperatureRatingEngine cleared")
