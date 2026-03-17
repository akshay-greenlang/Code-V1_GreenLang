# -*- coding: utf-8 -*-
"""
BenchmarkDataBridge - CTB/PAB Benchmark Data Intake for Article 9(3)
=====================================================================

This module connects PACK-011 (SFDR Article 9) with benchmark data providers
to support Article 9(3) products that designate a Climate Transition Benchmark
(CTB) or Paris-Aligned Benchmark (PAB) as a reference. It handles index
composition intake, universe carbon intensity calculations, historical
performance tracking, and benchmark alignment verification.

Architecture:
    PACK-011 SFDR Art 9(3) --> BenchmarkDataBridge --> Benchmark Providers
                                      |
                                      v
    Index Composition, Carbon Intensity, Tracking Error, Alignment Ratio

Regulatory Context:
    Article 9(3) SFDR requires products with a carbon emissions reduction
    objective to designate a CTB or PAB. The Benchmark Regulation (BMR)
    Delegated Acts (2020/1816 and 2020/1817) define the minimum standards
    for CTB and PAB indices including decarbonization trajectory, base year,
    exclusions, and year-on-year reduction targets (7% for CTB, 7% for PAB).

Example:
    >>> config = BenchmarkDataConfig(benchmark_type="PAB")
    >>> bridge = BenchmarkDataBridge(config)
    >>> index = bridge.load_index_composition(constituents)
    >>> perf = bridge.calculate_performance(portfolio, index)
    >>> print(f"Tracking error: {perf.tracking_error_bps:.0f} bps")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Helpers
# =============================================================================


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# Agent Stub
# =============================================================================


class _AgentStub:
    """Deferred agent loader for lazy initialization."""

    def __init__(self, agent_id: str, module_path: str, class_name: str) -> None:
        self.agent_id = agent_id
        self.module_path = module_path
        self.class_name = class_name
        self._instance: Optional[Any] = None

    def load(self) -> Any:
        """Load and return the agent instance."""
        if self._instance is not None:
            return self._instance
        try:
            import importlib
            mod = importlib.import_module(self.module_path)
            cls = getattr(mod, self.class_name)
            self._instance = cls()
            return self._instance
        except Exception as exc:
            logger.warning(
                "AgentStub: failed to load %s from %s: %s",
                self.agent_id, self.module_path, exc,
            )
            return None

    @property
    def is_loaded(self) -> bool:
        """Whether the agent has been loaded."""
        return self._instance is not None


# =============================================================================
# Enums
# =============================================================================


class BenchmarkType(str, Enum):
    """EU Climate Benchmark types."""
    CTB = "ctb"
    PAB = "pab"
    CUSTOM = "custom"
    NONE = "none"


class BenchmarkProvider(str, Enum):
    """Supported benchmark data providers."""
    MSCI = "msci"
    SP_GLOBAL = "sp_global"
    FTSE_RUSSELL = "ftse_russell"
    SOLACTIVE = "solactive"
    STOXX = "stoxx"
    CUSTOM = "custom"
    INTERNAL = "internal"


class DecarbonizationPathway(str, Enum):
    """Decarbonization pathway type."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SECTORAL = "sectoral"
    CUSTOM = "custom"


class AlignmentStatus(str, Enum):
    """Benchmark alignment status."""
    ALIGNED = "aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    NOT_ALIGNED = "not_aligned"
    INSUFFICIENT_DATA = "insufficient_data"


class RebalanceFrequency(str, Enum):
    """Index rebalance frequency."""
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    CUSTOM = "custom"


# =============================================================================
# Data Models
# =============================================================================


class BenchmarkDataConfig(BaseModel):
    """Configuration for the Benchmark Data Bridge."""
    benchmark_type: BenchmarkType = Field(
        default=BenchmarkType.PAB,
        description="Benchmark type: CTB or PAB (Art 9(3))",
    )
    benchmark_provider: BenchmarkProvider = Field(
        default=BenchmarkProvider.INTERNAL,
        description="Benchmark data provider",
    )
    index_name: str = Field(
        default="", description="Designated benchmark index name"
    )
    index_ticker: str = Field(
        default="", description="Benchmark index ticker symbol"
    )
    index_isin: str = Field(
        default="", description="Benchmark index ISIN"
    )
    base_year: int = Field(
        default=2019, ge=2015, le=2025,
        description="Decarbonization base year",
    )
    target_year: int = Field(
        default=2050, ge=2030, le=2060,
        description="Decarbonization target year (net-zero)",
    )
    annual_reduction_pct: float = Field(
        default=7.0, ge=0.0, le=20.0,
        description="Year-on-year carbon intensity reduction target (%)",
    )
    decarbonization_pathway: DecarbonizationPathway = Field(
        default=DecarbonizationPathway.LINEAR,
        description="Decarbonization pathway type",
    )
    rebalance_frequency: RebalanceFrequency = Field(
        default=RebalanceFrequency.QUARTERLY,
        description="Index rebalance frequency",
    )
    currency: str = Field(default="EUR", description="Reporting currency")
    tracking_error_threshold_bps: float = Field(
        default=200.0, ge=0.0,
        description="Maximum acceptable tracking error in basis points",
    )
    enable_exclusion_check: bool = Field(
        default=True,
        description="Verify BMR exclusion criteria compliance",
    )
    enable_provenance: bool = Field(
        default=True, description="Enable provenance hash tracking"
    )

    @field_validator("benchmark_type")
    @classmethod
    def validate_benchmark_type(cls, v: BenchmarkType) -> BenchmarkType:
        """Validate benchmark type for Article 9(3)."""
        if v not in (BenchmarkType.CTB, BenchmarkType.PAB, BenchmarkType.CUSTOM):
            logger.warning(
                "Article 9(3) requires CTB or PAB benchmark; got %s", v.value
            )
        return v


class BenchmarkConstituent(BaseModel):
    """A single constituent of a benchmark index."""
    isin: str = Field(default="", description="Constituent ISIN")
    name: str = Field(default="", description="Constituent name")
    ticker: str = Field(default="", description="Ticker symbol")
    weight_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Index weight (%)"
    )
    sector: str = Field(default="", description="GICS/ICB sector")
    nace_code: str = Field(default="", description="NACE sector code")
    country: str = Field(default="", description="Country of domicile")
    carbon_intensity: float = Field(
        default=0.0, description="Carbon intensity (tCO2e/EUR M revenue)"
    )
    scope_1_2_emissions: float = Field(
        default=0.0, description="Scope 1+2 emissions (tCO2e)"
    )
    scope_3_emissions: float = Field(
        default=0.0, description="Scope 3 emissions (tCO2e)"
    )
    excluded_activities: List[str] = Field(
        default_factory=list, description="Excluded activities per BMR"
    )
    is_excluded: bool = Field(
        default=False, description="Whether constituent is BMR-excluded"
    )


class BenchmarkIndex(BaseModel):
    """Complete benchmark index composition and metadata."""
    index_name: str = Field(default="", description="Index name")
    benchmark_type: str = Field(default="pab", description="CTB or PAB")
    provider: str = Field(default="internal", description="Data provider")
    reference_date: str = Field(default="", description="Composition date")
    constituents: List[BenchmarkConstituent] = Field(
        default_factory=list, description="Index constituents"
    )
    total_constituents: int = Field(default=0, description="Total constituents")
    total_market_cap_eur: float = Field(
        default=0.0, description="Total market cap (EUR)"
    )
    weighted_carbon_intensity: float = Field(
        default=0.0, description="Weighted average carbon intensity"
    )
    scope_1_2_intensity: float = Field(
        default=0.0, description="Scope 1+2 weighted intensity"
    )
    scope_3_intensity: float = Field(
        default=0.0, description="Scope 3 weighted intensity"
    )
    excluded_count: int = Field(
        default=0, description="Number of excluded constituents"
    )
    sector_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Sector weight breakdown"
    )
    country_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Country weight breakdown"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    loaded_at: str = Field(default="", description="Load timestamp")


class UniverseData(BaseModel):
    """Investable universe carbon intensity data for benchmark comparison."""
    universe_name: str = Field(default="", description="Universe name")
    reference_date: str = Field(default="", description="Reference date")
    total_companies: int = Field(default=0, description="Total companies")
    weighted_carbon_intensity: float = Field(
        default=0.0, description="Universe weighted carbon intensity"
    )
    median_carbon_intensity: float = Field(
        default=0.0, description="Median carbon intensity"
    )
    p25_carbon_intensity: float = Field(
        default=0.0, description="25th percentile intensity"
    )
    p75_carbon_intensity: float = Field(
        default=0.0, description="75th percentile intensity"
    )
    sector_intensities: Dict[str, float] = Field(
        default_factory=dict, description="Per-sector intensity"
    )
    yoy_reduction_pct: float = Field(
        default=0.0, description="Year-on-year intensity reduction (%)"
    )
    ctb_threshold: float = Field(
        default=0.0, description="CTB carbon intensity threshold"
    )
    pab_threshold: float = Field(
        default=0.0, description="PAB carbon intensity threshold"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class BenchmarkPerformance(BaseModel):
    """Performance comparison between portfolio and benchmark."""
    portfolio_name: str = Field(default="", description="Portfolio name")
    benchmark_name: str = Field(default="", description="Benchmark name")
    reference_date: str = Field(default="", description="Reference date")
    period_start: str = Field(default="", description="Period start")
    period_end: str = Field(default="", description="Period end")

    # Returns
    portfolio_return_pct: float = Field(
        default=0.0, description="Portfolio return (%)"
    )
    benchmark_return_pct: float = Field(
        default=0.0, description="Benchmark return (%)"
    )
    excess_return_pct: float = Field(
        default=0.0, description="Excess return (%)"
    )
    tracking_error_bps: float = Field(
        default=0.0, description="Tracking error (basis points)"
    )

    # Carbon metrics
    portfolio_carbon_intensity: float = Field(
        default=0.0, description="Portfolio weighted carbon intensity"
    )
    benchmark_carbon_intensity: float = Field(
        default=0.0, description="Benchmark weighted carbon intensity"
    )
    carbon_intensity_ratio: float = Field(
        default=0.0, description="Portfolio / Benchmark carbon intensity ratio"
    )
    portfolio_yoy_reduction_pct: float = Field(
        default=0.0, description="Portfolio YoY carbon reduction (%)"
    )
    benchmark_yoy_reduction_pct: float = Field(
        default=0.0, description="Benchmark YoY carbon reduction (%)"
    )

    # Alignment
    alignment_status: str = Field(
        default="insufficient_data", description="Alignment status"
    )
    decarbonization_on_track: bool = Field(
        default=False, description="On track for decarbonization target"
    )
    overlap_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Portfolio-benchmark constituent overlap (%)",
    )
    active_share_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Active share (%)"
    )

    # Exclusion compliance
    exclusion_compliant: bool = Field(
        default=True, description="BMR exclusion criteria met"
    )
    excluded_holdings: List[str] = Field(
        default_factory=list, description="Holdings that should be excluded"
    )

    errors: List[str] = Field(default_factory=list, description="Errors")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculated_at: str = Field(default="", description="Calculation timestamp")
    execution_time_ms: float = Field(default=0.0, description="Execution time")


# =============================================================================
# BMR Exclusion Criteria
# =============================================================================


CTB_EXCLUSIONS: Dict[str, str] = {
    "controversial_weapons": "Companies involved in controversial weapons",
    "tobacco_production": "Companies deriving 1%+ revenue from tobacco production",
    "un_global_compact": "Companies violating UN Global Compact principles",
    "oecd_guidelines": "Companies violating OECD Guidelines for MNEs",
}

PAB_EXCLUSIONS: Dict[str, str] = {
    **CTB_EXCLUSIONS,
    "coal_1pct": "Companies deriving 1%+ revenue from hard coal exploration/mining",
    "oil_10pct": "Companies deriving 10%+ revenue from oil exploration",
    "gas_50pct": "Companies deriving 50%+ revenue from natural gas exploration",
    "electricity_50pct": "Companies with 50%+ GHG-intensive electricity generation",
}


# Decarbonization trajectory constants
DECARBONIZATION_CONSTANTS: Dict[str, Dict[str, float]] = {
    "ctb": {
        "min_initial_reduction_pct": 30.0,
        "annual_reduction_pct": 7.0,
        "scope_coverage": 1.2,  # Scope 1+2 minimum
    },
    "pab": {
        "min_initial_reduction_pct": 50.0,
        "annual_reduction_pct": 7.0,
        "scope_coverage": 1.23,  # Scope 1+2+3 required
    },
}


# =============================================================================
# Benchmark Data Bridge
# =============================================================================


class BenchmarkDataBridge:
    """Bridge for CTB/PAB benchmark data intake and alignment analysis.

    Manages benchmark index composition loading, universe carbon intensity
    calculations, historical performance tracking, and alignment verification
    for Article 9(3) products.

    Attributes:
        config: Bridge configuration.
        _agents: Deferred agent stubs for benchmark-related agents.
        _index_cache: Cached index compositions.

    Example:
        >>> bridge = BenchmarkDataBridge(BenchmarkDataConfig(benchmark_type="PAB"))
        >>> index = bridge.load_index_composition(constituents)
        >>> print(f"Index intensity: {index.weighted_carbon_intensity:.1f}")
    """

    def __init__(self, config: Optional[BenchmarkDataConfig] = None) -> None:
        """Initialize the Benchmark Data Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or BenchmarkDataConfig()
        self.logger = logger
        self._index_cache: Dict[str, BenchmarkIndex] = {}

        self._agents: Dict[str, _AgentStub] = {
            "mrv_bridge": _AgentStub(
                "PACK-011-MRV",
                "packs.eu_compliance.PACK_011_sfdr_article_9.integrations.mrv_emissions_bridge",
                "MRVEmissionsBridge",
            ),
        }

        self.logger.info(
            "BenchmarkDataBridge initialized: type=%s, provider=%s, "
            "index=%s, annual_reduction=%.1f%%",
            self.config.benchmark_type.value,
            self.config.benchmark_provider.value,
            self.config.index_name,
            self.config.annual_reduction_pct,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def load_index_composition(
        self,
        constituents: List[Dict[str, Any]],
        reference_date: Optional[str] = None,
    ) -> BenchmarkIndex:
        """Load and process benchmark index composition.

        Parses constituent data, calculates weighted carbon intensity,
        checks BMR exclusion criteria, and builds sector/country breakdowns.

        Args:
            constituents: Raw constituent data records.
            reference_date: Composition reference date (ISO format).

        Returns:
            BenchmarkIndex with full composition and metadata.
        """
        start_time = time.time()
        ref_date = reference_date or _utcnow().strftime("%Y-%m-%d")

        parsed: List[BenchmarkConstituent] = []
        sector_weights: Dict[str, float] = {}
        country_weights: Dict[str, float] = {}

        for record in constituents:
            constituent = self._parse_constituent(record)
            if self.config.enable_exclusion_check:
                constituent.is_excluded = self._check_exclusion(constituent)
            parsed.append(constituent)

            sector = constituent.sector or "Unknown"
            country = constituent.country or "Unknown"
            sector_weights[sector] = sector_weights.get(sector, 0.0) + constituent.weight_pct
            country_weights[country] = country_weights.get(country, 0.0) + constituent.weight_pct

        # Calculate weighted metrics
        waci = self._calculate_weighted_intensity(parsed)
        scope_12 = self._calculate_scope_12_intensity(parsed)
        scope_3 = self._calculate_scope_3_intensity(parsed)
        excluded_count = sum(1 for c in parsed if c.is_excluded)

        index = BenchmarkIndex(
            index_name=self.config.index_name,
            benchmark_type=self.config.benchmark_type.value,
            provider=self.config.benchmark_provider.value,
            reference_date=ref_date,
            constituents=parsed,
            total_constituents=len(parsed),
            weighted_carbon_intensity=waci,
            scope_1_2_intensity=scope_12,
            scope_3_intensity=scope_3,
            excluded_count=excluded_count,
            sector_breakdown=sector_weights,
            country_breakdown=country_weights,
            loaded_at=_utcnow().isoformat(),
        )

        if self.config.enable_provenance:
            index.provenance_hash = _hash_data(
                index.model_dump(exclude={"provenance_hash", "constituents"})
            )

        # Cache the index
        cache_key = f"{self.config.index_name}_{ref_date}"
        self._index_cache[cache_key] = index

        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.info(
            "BenchmarkDataBridge: loaded %d constituents for %s, WACI=%.2f, "
            "excluded=%d, elapsed=%.1fms",
            len(parsed), self.config.index_name, waci, excluded_count, elapsed_ms,
        )
        return index

    def calculate_universe_data(
        self,
        universe_records: List[Dict[str, Any]],
        reference_date: Optional[str] = None,
    ) -> UniverseData:
        """Calculate universe carbon intensity metrics.

        Computes weighted average, median, percentile intensities and
        CTB/PAB thresholds for the investable universe.

        Args:
            universe_records: Universe company records with carbon data.
            reference_date: Reference date (ISO format).

        Returns:
            UniverseData with intensity statistics and thresholds.
        """
        ref_date = reference_date or _utcnow().strftime("%Y-%m-%d")
        intensities: List[float] = []
        sector_intensities: Dict[str, List[float]] = {}

        for record in universe_records:
            ci = float(record.get("carbon_intensity", 0.0))
            if ci > 0:
                intensities.append(ci)
                sector = record.get("sector", "Unknown")
                if sector not in sector_intensities:
                    sector_intensities[sector] = []
                sector_intensities[sector].append(ci)

        if not intensities:
            return UniverseData(
                universe_name="empty",
                reference_date=ref_date,
            )

        sorted_ci = sorted(intensities)
        n = len(sorted_ci)
        median = sorted_ci[n // 2]
        p25 = sorted_ci[n // 4] if n >= 4 else sorted_ci[0]
        p75 = sorted_ci[3 * n // 4] if n >= 4 else sorted_ci[-1]
        weighted_avg = sum(intensities) / n

        # CTB/PAB thresholds based on BMR Delegated Acts
        ctb_constants = DECARBONIZATION_CONSTANTS["ctb"]
        pab_constants = DECARBONIZATION_CONSTANTS["pab"]
        ctb_threshold = weighted_avg * (1 - ctb_constants["min_initial_reduction_pct"] / 100)
        pab_threshold = weighted_avg * (1 - pab_constants["min_initial_reduction_pct"] / 100)

        sector_avg: Dict[str, float] = {
            s: sum(vals) / len(vals)
            for s, vals in sector_intensities.items()
        }

        result = UniverseData(
            universe_name=f"Universe_{ref_date}",
            reference_date=ref_date,
            total_companies=n,
            weighted_carbon_intensity=weighted_avg,
            median_carbon_intensity=median,
            p25_carbon_intensity=p25,
            p75_carbon_intensity=p75,
            sector_intensities=sector_avg,
            ctb_threshold=ctb_threshold,
            pab_threshold=pab_threshold,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _hash_data(
                result.model_dump(exclude={"provenance_hash"})
            )

        return result

    def calculate_performance(
        self,
        portfolio_holdings: List[Dict[str, Any]],
        benchmark_index: Optional[BenchmarkIndex] = None,
        period_start: str = "",
        period_end: str = "",
    ) -> BenchmarkPerformance:
        """Calculate performance comparison between portfolio and benchmark.

        Computes tracking error, carbon intensity ratio, decarbonization
        alignment, constituent overlap, and exclusion compliance.

        Args:
            portfolio_holdings: Portfolio holding records.
            benchmark_index: Benchmark index (uses cached if None).
            period_start: Performance period start date.
            period_end: Performance period end date.

        Returns:
            BenchmarkPerformance with full comparison metrics.
        """
        start_time = time.time()
        errors: List[str] = []
        warnings: List[str] = []

        # Get benchmark index
        index = benchmark_index
        if index is None:
            index = self._get_cached_index()
        if index is None:
            errors.append("No benchmark index available")
            return BenchmarkPerformance(errors=errors)

        # Calculate carbon metrics
        portfolio_ci = self._calculate_portfolio_intensity(portfolio_holdings)
        benchmark_ci = index.weighted_carbon_intensity
        ci_ratio = (portfolio_ci / benchmark_ci) if benchmark_ci > 0 else 0.0

        # Calculate overlap
        portfolio_isins = {h.get("isin", "") for h in portfolio_holdings if h.get("isin")}
        benchmark_isins = {c.isin for c in index.constituents if c.isin}
        common = portfolio_isins & benchmark_isins
        overlap_pct = (len(common) / len(benchmark_isins) * 100.0) if benchmark_isins else 0.0
        active_share = 100.0 - overlap_pct

        # Check exclusions
        exclusion_compliant, excluded_holdings = self._check_portfolio_exclusions(
            portfolio_holdings, index,
        )
        if not exclusion_compliant:
            warnings.append(
                f"Portfolio holds {len(excluded_holdings)} BMR-excluded securities"
            )

        # Determine alignment
        alignment = self._determine_alignment(
            portfolio_ci, benchmark_ci, ci_ratio, exclusion_compliant,
        )

        # Simulate tracking error (deterministic placeholder)
        tracking_error = self._estimate_tracking_error(
            portfolio_holdings, index,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        perf = BenchmarkPerformance(
            portfolio_name="Article 9 Portfolio",
            benchmark_name=index.index_name,
            reference_date=_utcnow().strftime("%Y-%m-%d"),
            period_start=period_start,
            period_end=period_end,
            portfolio_carbon_intensity=portfolio_ci,
            benchmark_carbon_intensity=benchmark_ci,
            carbon_intensity_ratio=ci_ratio,
            tracking_error_bps=tracking_error,
            alignment_status=alignment,
            decarbonization_on_track=(ci_ratio <= 1.0),
            overlap_pct=overlap_pct,
            active_share_pct=active_share,
            exclusion_compliant=exclusion_compliant,
            excluded_holdings=excluded_holdings,
            errors=errors,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            execution_time_ms=elapsed_ms,
        )

        if self.config.enable_provenance:
            perf.provenance_hash = _hash_data(
                perf.model_dump(exclude={"provenance_hash"})
            )

        self.logger.info(
            "BenchmarkDataBridge: performance calculated, CI ratio=%.3f, "
            "tracking_error=%.0f bps, alignment=%s, elapsed=%.1fms",
            ci_ratio, tracking_error, alignment, elapsed_ms,
        )
        return perf

    def get_decarbonization_trajectory(
        self,
        base_year_intensity: float,
        current_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculate the decarbonization trajectory for the benchmark.

        Args:
            base_year_intensity: Carbon intensity in the base year.
            current_year: Current year (defaults to now).

        Returns:
            Trajectory data with yearly targets and current status.
        """
        year = current_year or _utcnow().year
        reduction_rate = self.config.annual_reduction_pct / 100.0
        years_elapsed = year - self.config.base_year

        trajectory: Dict[int, float] = {}
        for y in range(self.config.base_year, self.config.target_year + 1):
            years = y - self.config.base_year
            if self.config.decarbonization_pathway == DecarbonizationPathway.LINEAR:
                factor = max(0.0, 1.0 - (reduction_rate * years))
            else:
                factor = max(0.0, (1.0 - reduction_rate) ** years)
            trajectory[y] = base_year_intensity * factor

        current_target = trajectory.get(year, base_year_intensity)
        remaining_budget = current_target

        return {
            "base_year": self.config.base_year,
            "target_year": self.config.target_year,
            "base_intensity": base_year_intensity,
            "current_year": year,
            "current_target_intensity": current_target,
            "remaining_budget": remaining_budget,
            "annual_reduction_pct": self.config.annual_reduction_pct,
            "pathway": self.config.decarbonization_pathway.value,
            "trajectory": trajectory,
            "years_elapsed": years_elapsed,
        }

    def get_exclusion_list(self) -> Dict[str, str]:
        """Get the applicable BMR exclusion criteria.

        Returns:
            Dict mapping exclusion codes to descriptions.
        """
        if self.config.benchmark_type == BenchmarkType.PAB:
            return dict(PAB_EXCLUSIONS)
        elif self.config.benchmark_type == BenchmarkType.CTB:
            return dict(CTB_EXCLUSIONS)
        return {}

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _parse_constituent(
        self, record: Dict[str, Any]
    ) -> BenchmarkConstituent:
        """Parse a raw constituent record."""
        return BenchmarkConstituent(
            isin=str(record.get("isin", "")),
            name=str(record.get("name", "")),
            ticker=str(record.get("ticker", "")),
            weight_pct=float(record.get("weight_pct", 0.0)),
            sector=str(record.get("sector", "")),
            nace_code=str(record.get("nace_code", "")),
            country=str(record.get("country", "")),
            carbon_intensity=float(record.get("carbon_intensity", 0.0)),
            scope_1_2_emissions=float(record.get("scope_1_2_emissions", 0.0)),
            scope_3_emissions=float(record.get("scope_3_emissions", 0.0)),
            excluded_activities=record.get("excluded_activities", []),
        )

    def _check_exclusion(
        self, constituent: BenchmarkConstituent
    ) -> bool:
        """Check if a constituent should be excluded per BMR criteria."""
        exclusions = self.get_exclusion_list()
        for activity in constituent.excluded_activities:
            if activity in exclusions:
                return True
        return False

    def _calculate_weighted_intensity(
        self, constituents: List[BenchmarkConstituent]
    ) -> float:
        """Calculate weighted average carbon intensity."""
        total_weight = sum(c.weight_pct for c in constituents if not c.is_excluded)
        if total_weight <= 0:
            return 0.0
        weighted = sum(
            c.weight_pct * c.carbon_intensity
            for c in constituents if not c.is_excluded
        )
        return weighted / total_weight

    def _calculate_scope_12_intensity(
        self, constituents: List[BenchmarkConstituent]
    ) -> float:
        """Calculate weighted Scope 1+2 intensity."""
        total_weight = sum(c.weight_pct for c in constituents if not c.is_excluded)
        if total_weight <= 0:
            return 0.0
        weighted = sum(
            c.weight_pct * c.scope_1_2_emissions
            for c in constituents if not c.is_excluded
        )
        return weighted / total_weight

    def _calculate_scope_3_intensity(
        self, constituents: List[BenchmarkConstituent]
    ) -> float:
        """Calculate weighted Scope 3 intensity."""
        total_weight = sum(c.weight_pct for c in constituents if not c.is_excluded)
        if total_weight <= 0:
            return 0.0
        weighted = sum(
            c.weight_pct * c.scope_3_emissions
            for c in constituents if not c.is_excluded
        )
        return weighted / total_weight

    def _calculate_portfolio_intensity(
        self, holdings: List[Dict[str, Any]]
    ) -> float:
        """Calculate portfolio weighted average carbon intensity."""
        total_weight = 0.0
        weighted_ci = 0.0
        for h in holdings:
            w = float(h.get("weight", 0.0))
            ci = float(h.get("carbon_intensity", 0.0))
            if w > 0:
                weighted_ci += w * ci
                total_weight += w
        return (weighted_ci / total_weight) if total_weight > 0 else 0.0

    def _get_cached_index(self) -> Optional[BenchmarkIndex]:
        """Get the most recently cached index."""
        if not self._index_cache:
            return None
        latest_key = max(self._index_cache.keys())
        return self._index_cache[latest_key]

    def _check_portfolio_exclusions(
        self,
        holdings: List[Dict[str, Any]],
        index: BenchmarkIndex,
    ) -> tuple:
        """Check portfolio against BMR exclusion criteria.

        Returns:
            Tuple of (compliant: bool, excluded_holdings: List[str]).
        """
        excluded_isins = {c.isin for c in index.constituents if c.is_excluded}
        portfolio_excluded: List[str] = []

        for h in holdings:
            isin = h.get("isin", "")
            activities = h.get("excluded_activities", [])

            if isin in excluded_isins:
                portfolio_excluded.append(isin)
                continue

            exclusion_list = self.get_exclusion_list()
            for activity in activities:
                if activity in exclusion_list:
                    portfolio_excluded.append(isin)
                    break

        return (len(portfolio_excluded) == 0, portfolio_excluded)

    def _determine_alignment(
        self,
        portfolio_ci: float,
        benchmark_ci: float,
        ci_ratio: float,
        exclusion_compliant: bool,
    ) -> str:
        """Determine benchmark alignment status."""
        if portfolio_ci <= 0 or benchmark_ci <= 0:
            return AlignmentStatus.INSUFFICIENT_DATA.value

        if ci_ratio <= 1.0 and exclusion_compliant:
            return AlignmentStatus.ALIGNED.value
        elif ci_ratio <= 1.2:
            return AlignmentStatus.PARTIALLY_ALIGNED.value
        else:
            return AlignmentStatus.NOT_ALIGNED.value

    def _estimate_tracking_error(
        self,
        holdings: List[Dict[str, Any]],
        index: BenchmarkIndex,
    ) -> float:
        """Estimate tracking error in basis points.

        Uses a simplified weight-difference approach as a deterministic
        estimate. Production systems would use historical return data.
        """
        portfolio_weights: Dict[str, float] = {}
        for h in holdings:
            isin = h.get("isin", "")
            if isin:
                portfolio_weights[isin] = float(h.get("weight", 0.0))

        benchmark_weights: Dict[str, float] = {}
        for c in index.constituents:
            if c.isin:
                benchmark_weights[c.isin] = c.weight_pct

        all_isins = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
        if not all_isins:
            return 0.0

        sum_sq_diff = 0.0
        for isin in all_isins:
            pw = portfolio_weights.get(isin, 0.0)
            bw = benchmark_weights.get(isin, 0.0)
            sum_sq_diff += (pw - bw) ** 2

        # Convert to basis points (simplified)
        return (sum_sq_diff ** 0.5) * 100.0
