# -*- coding: utf-8 -*-
"""
PortfolioRiskAggregator - AGENT-EUDR-018 Engine 8: Cross-Commodity Portfolio Risk

Provides cross-commodity portfolio risk analysis for EUDR compliance. Calculates
concentration indices (HHI), diversification scores, total risk exposure,
Value at Risk (VaR), correlation matrices, and scenario simulations across
multi-commodity portfolios.

Zero-Hallucination Guarantees:
    - All risk metrics use deterministic Decimal arithmetic.
    - HHI calculation follows standard Herfindahl-Hirschman formula.
    - VaR uses parametric (variance-covariance) method with static volatilities.
    - Correlation matrix uses static commodity risk correlation data.
    - Scenario simulation applies fixed multipliers (no stochastic models).
    - SHA-256 provenance hashes on all output objects.

Key Metrics:
    - Herfindahl-Hirschman Index (HHI): Sum of squared market shares.
      <1500 = unconcentrated, 1500-2500 = moderate, >2500 = highly concentrated.
    - Diversification Score: 0-100, inverse of normalized HHI.
    - Total Risk Exposure: Weighted sum of individual commodity risk scores.
    - Value at Risk (VaR): Parametric VaR at configurable confidence level.
    - Correlation Matrix: Pairwise risk correlation between commodities.

Scenario Types:
    - price_shock: Simulate commodity price change impact.
    - supply_disruption: Simulate supply chain disruption for a commodity.
    - regulatory_change: Simulate new EUDR enforcement action impact.
    - climate_event: Simulate climate event affecting production regions.

Performance Targets:
    - Portfolio analysis: <100ms for 20 commodity positions.
    - HHI calculation: <10ms.
    - VaR calculation: <50ms.
    - Scenario simulation: <100ms per scenario.

Regulatory References:
    - EUDR Article 4: Due diligence across commodity portfolio.
    - EUDR Article 10: Risk assessment for portfolio diversification.
    - EUDR Article 29: Country risk integration in portfolio risk.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-018, Engine 8 (Portfolio Risk Aggregator)
Agent ID: GL-EUDR-CRA-018
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version
# ---------------------------------------------------------------------------

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, dataclass with to_dict, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "prt") -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid EUDR commodity types.
EUDR_COMMODITIES: frozenset = frozenset({
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
})

#: HHI thresholds.
HHI_UNCONCENTRATED: Decimal = Decimal("1500")
HHI_MODERATE: Decimal = Decimal("2500")
HHI_MAX: Decimal = Decimal("10000")

#: Maximum portfolio positions supported.
MAX_PORTFOLIO_POSITIONS: int = 500

#: Valid scenario types.
VALID_SCENARIO_TYPES: frozenset = frozenset({
    "price_shock", "supply_disruption", "regulatory_change", "climate_event",
})

# ---------------------------------------------------------------------------
# Commodity Risk Base Parameters
# ---------------------------------------------------------------------------
# Static risk parameters per commodity used in portfolio calculations.

COMMODITY_RISK_PARAMETERS: Dict[str, Dict[str, Decimal]] = {
    "cattle": {
        "base_risk": Decimal("65"),
        "volatility": Decimal("0.25"),
        "deforestation_risk_weight": Decimal("0.85"),
        "regulatory_risk_weight": Decimal("0.70"),
        "supply_disruption_sensitivity": Decimal("0.60"),
    },
    "cocoa": {
        "base_risk": Decimal("55"),
        "volatility": Decimal("0.30"),
        "deforestation_risk_weight": Decimal("0.75"),
        "regulatory_risk_weight": Decimal("0.65"),
        "supply_disruption_sensitivity": Decimal("0.70"),
    },
    "coffee": {
        "base_risk": Decimal("50"),
        "volatility": Decimal("0.28"),
        "deforestation_risk_weight": Decimal("0.65"),
        "regulatory_risk_weight": Decimal("0.60"),
        "supply_disruption_sensitivity": Decimal("0.65"),
    },
    "oil_palm": {
        "base_risk": Decimal("75"),
        "volatility": Decimal("0.35"),
        "deforestation_risk_weight": Decimal("0.95"),
        "regulatory_risk_weight": Decimal("0.85"),
        "supply_disruption_sensitivity": Decimal("0.55"),
    },
    "rubber": {
        "base_risk": Decimal("55"),
        "volatility": Decimal("0.22"),
        "deforestation_risk_weight": Decimal("0.60"),
        "regulatory_risk_weight": Decimal("0.55"),
        "supply_disruption_sensitivity": Decimal("0.50"),
    },
    "soya": {
        "base_risk": Decimal("70"),
        "volatility": Decimal("0.32"),
        "deforestation_risk_weight": Decimal("0.90"),
        "regulatory_risk_weight": Decimal("0.80"),
        "supply_disruption_sensitivity": Decimal("0.45"),
    },
    "wood": {
        "base_risk": Decimal("60"),
        "volatility": Decimal("0.20"),
        "deforestation_risk_weight": Decimal("0.80"),
        "regulatory_risk_weight": Decimal("0.75"),
        "supply_disruption_sensitivity": Decimal("0.55"),
    },
}

# ---------------------------------------------------------------------------
# Cross-Commodity Risk Correlation Matrix
# ---------------------------------------------------------------------------
# Static pairwise correlation coefficients for risk factors.
# Values range from -1.0 (perfect inverse) to 1.0 (perfect positive).

RISK_CORRELATIONS: Dict[Tuple[str, str], Decimal] = {
    # Cattle correlations
    ("cattle", "cattle"): Decimal("1.0"),
    ("cattle", "cocoa"): Decimal("0.25"),
    ("cattle", "coffee"): Decimal("0.20"),
    ("cattle", "oil_palm"): Decimal("0.35"),
    ("cattle", "rubber"): Decimal("0.15"),
    ("cattle", "soya"): Decimal("0.75"),
    ("cattle", "wood"): Decimal("0.55"),
    # Cocoa correlations
    ("cocoa", "cocoa"): Decimal("1.0"),
    ("cocoa", "coffee"): Decimal("0.65"),
    ("cocoa", "oil_palm"): Decimal("0.40"),
    ("cocoa", "rubber"): Decimal("0.30"),
    ("cocoa", "soya"): Decimal("0.20"),
    ("cocoa", "wood"): Decimal("0.25"),
    # Coffee correlations
    ("coffee", "coffee"): Decimal("1.0"),
    ("coffee", "oil_palm"): Decimal("0.30"),
    ("coffee", "rubber"): Decimal("0.25"),
    ("coffee", "soya"): Decimal("0.15"),
    ("coffee", "wood"): Decimal("0.20"),
    # Oil palm correlations
    ("oil_palm", "oil_palm"): Decimal("1.0"),
    ("oil_palm", "rubber"): Decimal("0.55"),
    ("oil_palm", "soya"): Decimal("0.45"),
    ("oil_palm", "wood"): Decimal("0.50"),
    # Rubber correlations
    ("rubber", "rubber"): Decimal("1.0"),
    ("rubber", "soya"): Decimal("0.15"),
    ("rubber", "wood"): Decimal("0.40"),
    # Soya correlations
    ("soya", "soya"): Decimal("1.0"),
    ("soya", "wood"): Decimal("0.45"),
    # Wood correlations
    ("wood", "wood"): Decimal("1.0"),
}

# ---------------------------------------------------------------------------
# Scenario Impact Factors
# ---------------------------------------------------------------------------

SCENARIO_IMPACTS: Dict[str, Dict[str, Dict[str, Decimal]]] = {
    "price_shock": {
        "cattle": {"risk_delta": Decimal("15"), "exposure_multiplier": Decimal("1.2")},
        "cocoa": {"risk_delta": Decimal("20"), "exposure_multiplier": Decimal("1.3")},
        "coffee": {"risk_delta": Decimal("18"), "exposure_multiplier": Decimal("1.25")},
        "oil_palm": {"risk_delta": Decimal("25"), "exposure_multiplier": Decimal("1.35")},
        "rubber": {"risk_delta": Decimal("12"), "exposure_multiplier": Decimal("1.15")},
        "soya": {"risk_delta": Decimal("22"), "exposure_multiplier": Decimal("1.3")},
        "wood": {"risk_delta": Decimal("10"), "exposure_multiplier": Decimal("1.1")},
    },
    "supply_disruption": {
        "cattle": {"risk_delta": Decimal("20"), "exposure_multiplier": Decimal("1.4")},
        "cocoa": {"risk_delta": Decimal("30"), "exposure_multiplier": Decimal("1.5")},
        "coffee": {"risk_delta": Decimal("25"), "exposure_multiplier": Decimal("1.45")},
        "oil_palm": {"risk_delta": Decimal("15"), "exposure_multiplier": Decimal("1.25")},
        "rubber": {"risk_delta": Decimal("18"), "exposure_multiplier": Decimal("1.3")},
        "soya": {"risk_delta": Decimal("12"), "exposure_multiplier": Decimal("1.2")},
        "wood": {"risk_delta": Decimal("20"), "exposure_multiplier": Decimal("1.35")},
    },
    "regulatory_change": {
        "cattle": {"risk_delta": Decimal("10"), "exposure_multiplier": Decimal("1.15")},
        "cocoa": {"risk_delta": Decimal("12"), "exposure_multiplier": Decimal("1.2")},
        "coffee": {"risk_delta": Decimal("10"), "exposure_multiplier": Decimal("1.15")},
        "oil_palm": {"risk_delta": Decimal("30"), "exposure_multiplier": Decimal("1.5")},
        "rubber": {"risk_delta": Decimal("8"), "exposure_multiplier": Decimal("1.1")},
        "soya": {"risk_delta": Decimal("25"), "exposure_multiplier": Decimal("1.4")},
        "wood": {"risk_delta": Decimal("15"), "exposure_multiplier": Decimal("1.25")},
    },
    "climate_event": {
        "cattle": {"risk_delta": Decimal("25"), "exposure_multiplier": Decimal("1.4")},
        "cocoa": {"risk_delta": Decimal("35"), "exposure_multiplier": Decimal("1.6")},
        "coffee": {"risk_delta": Decimal("30"), "exposure_multiplier": Decimal("1.5")},
        "oil_palm": {"risk_delta": Decimal("20"), "exposure_multiplier": Decimal("1.3")},
        "rubber": {"risk_delta": Decimal("15"), "exposure_multiplier": Decimal("1.2")},
        "soya": {"risk_delta": Decimal("28"), "exposure_multiplier": Decimal("1.45")},
        "wood": {"risk_delta": Decimal("12"), "exposure_multiplier": Decimal("1.15")},
    },
}

# ---------------------------------------------------------------------------
# Z-scores for VaR confidence levels
# ---------------------------------------------------------------------------

VAR_Z_SCORES: Dict[str, Decimal] = {
    "0.90": Decimal("1.282"),
    "0.95": Decimal("1.645"),
    "0.99": Decimal("2.326"),
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class CommodityPosition:
    """A single commodity position within a portfolio.

    Attributes:
        commodity: EUDR commodity type.
        weight: Portfolio weight (Decimal 0.0-1.0).
        exposure_value: Monetary exposure value.
        risk_score: Current risk score (Decimal 0-100).
        supplier_count: Number of suppliers for this commodity.
        origin_countries: List of origin country codes.
    """

    commodity: str = ""
    weight: Decimal = Decimal("0")
    exposure_value: Decimal = Decimal("0")
    risk_score: Decimal = Decimal("0")
    supplier_count: int = 0
    origin_countries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "commodity": self.commodity,
            "weight": str(self.weight),
            "exposure_value": str(self.exposure_value),
            "risk_score": str(self.risk_score),
            "supplier_count": self.supplier_count,
            "origin_countries": self.origin_countries,
        }


@dataclass
class PortfolioSummary:
    """Summary of a portfolio risk analysis.

    Attributes:
        portfolio_id: Unique portfolio identifier.
        portfolio_name: Human-readable portfolio name.
        total_exposure: Total portfolio exposure value.
        commodity_count: Number of distinct commodities.
        total_suppliers: Total suppliers across all commodities.
        hhi: Herfindahl-Hirschman Index.
        concentration_level: HHI classification.
        diversification_score: Diversification score (0-100).
        total_risk_exposure: Aggregate risk exposure.
        weighted_risk_score: Portfolio-weighted average risk.
        var_95: Value at Risk at 95% confidence.
        analysis_timestamp: When the analysis was performed.
        provenance_hash: SHA-256 hash.
    """

    portfolio_id: str = ""
    portfolio_name: str = ""
    total_exposure: Decimal = Decimal("0")
    commodity_count: int = 0
    total_suppliers: int = 0
    hhi: Decimal = Decimal("0")
    concentration_level: str = ""
    diversification_score: Decimal = Decimal("0")
    total_risk_exposure: Decimal = Decimal("0")
    weighted_risk_score: Decimal = Decimal("0")
    var_95: Decimal = Decimal("0")
    analysis_timestamp: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "portfolio_id": self.portfolio_id,
            "portfolio_name": self.portfolio_name,
            "total_exposure": str(self.total_exposure),
            "commodity_count": self.commodity_count,
            "total_suppliers": self.total_suppliers,
            "hhi": str(self.hhi),
            "concentration_level": self.concentration_level,
            "diversification_score": str(self.diversification_score),
            "total_risk_exposure": str(self.total_risk_exposure),
            "weighted_risk_score": str(self.weighted_risk_score),
            "var_95": str(self.var_95),
            "analysis_timestamp": self.analysis_timestamp,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# PortfolioRiskAggregator
# ---------------------------------------------------------------------------


class PortfolioRiskAggregator:
    """Production-grade cross-commodity portfolio risk aggregator for EUDR.

    Provides portfolio-level risk analytics including concentration measurement,
    diversification scoring, VaR calculation, scenario simulation, and
    diversification recommendations across EUDR-regulated commodities.

    Thread Safety:
        All mutable state is protected by a reentrant lock.

    Zero-Hallucination:
        All financial and risk metrics use Decimal arithmetic with
        deterministic formulas. No stochastic/Monte Carlo simulation.
        No ML/LLM models in any calculation path.

    Attributes:
        _portfolio_cache: Cached portfolio analysis results.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> aggregator = PortfolioRiskAggregator()
        >>> positions = [
        ...     {"commodity": "oil_palm", "weight": 0.4, "exposure_value": 400000},
        ...     {"commodity": "soya", "weight": 0.35, "exposure_value": 350000},
        ...     {"commodity": "cocoa", "weight": 0.25, "exposure_value": 250000},
        ... ]
        >>> result = aggregator.analyze_portfolio(positions)
        >>> assert "hhi" in result
    """

    def __init__(self) -> None:
        """Initialize PortfolioRiskAggregator with empty cache."""
        self._portfolio_cache: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.RLock = threading.RLock()
        logger.info(
            "PortfolioRiskAggregator initialized (version=%s)",
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_portfolio(
        self,
        commodity_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform full portfolio risk analysis across multiple commodities.

        Computes HHI, diversification score, total risk exposure, weighted
        risk score, VaR, and per-commodity breakdown.

        Args:
            commodity_positions: List of position dicts, each with:
                ``commodity`` (str), ``weight`` (numeric 0-1),
                ``exposure_value`` (numeric), optionally ``risk_score``
                (numeric 0-100), ``supplier_count`` (int),
                ``origin_countries`` (list of str).

        Returns:
            Dictionary with full portfolio analysis, per-commodity breakdown,
            and provenance_hash.

        Raises:
            ValueError: If commodity_positions is empty, exceeds max size,
                or contains invalid commodities.
        """
        start_time = time.monotonic()

        self._validate_positions(commodity_positions)
        positions = self._parse_positions(commodity_positions)

        # Normalize weights
        positions = self._normalize_weights(positions)

        # Calculate metrics
        hhi = self._calculate_hhi(positions)
        concentration_level = self._classify_concentration(hhi)
        diversification = self._calculate_diversification(hhi)
        total_exposure = sum(p.exposure_value for p in positions)
        weighted_risk = self._calculate_weighted_risk(positions)
        total_risk_exposure = self._calculate_total_risk_exposure(positions)
        var_95 = self._calculate_var_internal(positions, Decimal("0.95"))

        total_suppliers = sum(p.supplier_count for p in positions)
        analysis_ts = _utcnow().isoformat()

        summary = PortfolioSummary(
            portfolio_id=_generate_id("pf"),
            total_exposure=total_exposure,
            commodity_count=len(positions),
            total_suppliers=total_suppliers,
            hhi=hhi,
            concentration_level=concentration_level,
            diversification_score=diversification,
            total_risk_exposure=total_risk_exposure,
            weighted_risk_score=weighted_risk,
            var_95=var_95,
            analysis_timestamp=analysis_ts,
        )
        summary.provenance_hash = _compute_hash(summary)

        # Per-commodity breakdown
        commodity_breakdown: List[Dict[str, Any]] = []
        for pos in positions:
            risk_params = COMMODITY_RISK_PARAMETERS.get(pos.commodity, {})
            commodity_breakdown.append({
                "commodity": pos.commodity,
                "weight": str(pos.weight),
                "exposure_value": str(pos.exposure_value),
                "risk_score": str(pos.risk_score),
                "base_risk": str(risk_params.get("base_risk", Decimal("50"))),
                "volatility": str(risk_params.get("volatility", Decimal("0.25"))),
                "supplier_count": pos.supplier_count,
                "origin_countries": pos.origin_countries,
                "contribution_to_hhi": str(
                    (pos.weight * Decimal("100")) ** 2
                ),
            })

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = summary.to_dict()
        result["commodity_breakdown"] = commodity_breakdown
        result["processing_time_ms"] = round(processing_time_ms, 3)

        # Cache the result
        with self._lock:
            self._portfolio_cache[summary.portfolio_id] = result

        logger.info(
            "Portfolio analysis: commodities=%d HHI=%s diversification=%s "
            "risk=%s VaR95=%s time_ms=%.1f",
            len(positions), hhi, diversification,
            weighted_risk, var_95, processing_time_ms,
        )
        return result

    def calculate_concentration_index(
        self,
        commodity_positions: List[Dict[str, Any]],
    ) -> Decimal:
        """Calculate HHI for commodity concentration.

        The Herfindahl-Hirschman Index is the sum of squared market shares
        (expressed as percentages). Range: 0-10000.
        <1500 = unconcentrated, 1500-2500 = moderate, >2500 = concentrated.

        Args:
            commodity_positions: List of position dicts with ``weight`` field.

        Returns:
            Decimal HHI value.

        Raises:
            ValueError: If positions are empty or invalid.
        """
        self._validate_positions(commodity_positions)
        positions = self._parse_positions(commodity_positions)
        positions = self._normalize_weights(positions)
        return self._calculate_hhi(positions)

    def calculate_diversification_score(
        self,
        commodity_positions: List[Dict[str, Any]],
    ) -> Decimal:
        """Calculate a diversification score (0-100) for the portfolio.

        Higher score means better diversification. Computed as the inverse
        of normalized HHI.

        Args:
            commodity_positions: List of position dicts with ``weight`` field.

        Returns:
            Decimal diversification score (0-100).

        Raises:
            ValueError: If positions are empty or invalid.
        """
        self._validate_positions(commodity_positions)
        positions = self._parse_positions(commodity_positions)
        positions = self._normalize_weights(positions)
        hhi = self._calculate_hhi(positions)
        return self._calculate_diversification(hhi)

    def calculate_total_risk_exposure(
        self,
        commodity_positions: List[Dict[str, Any]],
    ) -> Decimal:
        """Calculate aggregate risk exposure across the portfolio.

        Computed as the sum of (exposure_value * risk_score / 100) for
        each position.

        Args:
            commodity_positions: List of position dicts with
                ``exposure_value`` and optional ``risk_score``.

        Returns:
            Decimal total risk exposure value.

        Raises:
            ValueError: If positions are empty or invalid.
        """
        self._validate_positions(commodity_positions)
        positions = self._parse_positions(commodity_positions)
        return self._calculate_total_risk_exposure(positions)

    def get_portfolio_summary(
        self,
        portfolio_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get summary of cached portfolio risk metrics.

        Args:
            portfolio_name: Filter by portfolio name. If None, returns
                the most recent cached analysis.

        Returns:
            Dictionary with portfolio summary or empty dict if no cache.
        """
        with self._lock:
            if not self._portfolio_cache:
                return {
                    "status": "no_portfolios_analyzed",
                    "portfolio_count": 0,
                }

            if portfolio_name:
                for pid, data in self._portfolio_cache.items():
                    if data.get("portfolio_name") == portfolio_name:
                        return data
                return {"status": "portfolio_not_found", "name": portfolio_name}

            # Return the most recent
            most_recent_id = list(self._portfolio_cache.keys())[-1]
            return self._portfolio_cache[most_recent_id]

    def simulate_scenario(
        self,
        portfolio: List[Dict[str, Any]],
        scenario_type: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a what-if scenario analysis on the portfolio.

        Applies static impact factors to commodity positions and recalculates
        all portfolio metrics under the scenario.

        Args:
            portfolio: List of commodity position dicts.
            scenario_type: Type of scenario. Valid: price_shock,
                supply_disruption, regulatory_change, climate_event.
            parameters: Scenario parameters. Keys vary by type:
                - price_shock: ``magnitude`` (Decimal, e.g. -0.20 for -20%)
                - supply_disruption: ``affected_commodity`` (str)
                - regulatory_change: ``affected_commodity`` (str or "all")
                - climate_event: ``severity`` ("low", "medium", "high")

        Returns:
            Dictionary with baseline metrics, scenario metrics, delta
            analysis, and provenance_hash.

        Raises:
            ValueError: If scenario_type is invalid or portfolio is empty.
        """
        start_time = time.monotonic()

        if scenario_type not in VALID_SCENARIO_TYPES:
            raise ValueError(
                f"Invalid scenario_type '{scenario_type}'. "
                f"Valid: {sorted(VALID_SCENARIO_TYPES)}"
            )

        self._validate_positions(portfolio)
        positions = self._parse_positions(portfolio)
        positions = self._normalize_weights(positions)

        # Baseline metrics
        baseline_hhi = self._calculate_hhi(positions)
        baseline_risk = self._calculate_weighted_risk(positions)
        baseline_exposure = self._calculate_total_risk_exposure(positions)
        baseline_var = self._calculate_var_internal(positions, Decimal("0.95"))

        # Apply scenario impacts
        scenario_positions = self._apply_scenario(
            positions, scenario_type, parameters,
        )

        # Scenario metrics
        scenario_hhi = self._calculate_hhi(scenario_positions)
        scenario_risk = self._calculate_weighted_risk(scenario_positions)
        scenario_exposure = self._calculate_total_risk_exposure(scenario_positions)
        scenario_var = self._calculate_var_internal(
            scenario_positions, Decimal("0.95"),
        )

        # Delta analysis
        delta_risk = scenario_risk - baseline_risk
        delta_exposure = scenario_exposure - baseline_exposure
        delta_var = scenario_var - baseline_var

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "scenario_type": scenario_type,
            "parameters": {k: str(v) for k, v in parameters.items()},
            "baseline": {
                "hhi": str(baseline_hhi),
                "weighted_risk": str(baseline_risk),
                "total_exposure": str(baseline_exposure),
                "var_95": str(baseline_var),
            },
            "scenario": {
                "hhi": str(scenario_hhi),
                "weighted_risk": str(scenario_risk),
                "total_exposure": str(scenario_exposure),
                "var_95": str(scenario_var),
            },
            "delta": {
                "risk_change": str(delta_risk.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )),
                "exposure_change": str(delta_exposure.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )),
                "var_change": str(delta_var.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )),
                "risk_change_pct": str(
                    self._safe_pct_change(baseline_risk, scenario_risk)
                ),
            },
            "per_commodity_impact": [
                {
                    "commodity": sp.commodity,
                    "baseline_risk": str(positions[i].risk_score),
                    "scenario_risk": str(sp.risk_score),
                    "risk_delta": str(sp.risk_score - positions[i].risk_score),
                }
                for i, sp in enumerate(scenario_positions)
            ],
            "processing_time_ms": round(processing_time_ms, 3),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Scenario simulation: type=%s risk_delta=%s exposure_delta=%s "
            "time_ms=%.1f",
            scenario_type, delta_risk, delta_exposure, processing_time_ms,
        )
        return result

    def calculate_correlation_matrix(
        self,
        commodities: List[str],
    ) -> Dict[str, Any]:
        """Calculate risk correlation matrix between commodities.

        Uses static pairwise correlation coefficients representing the
        co-movement of deforestation risk factors.

        Args:
            commodities: List of EUDR commodity types.

        Returns:
            Dictionary with matrix (dict of dict), commodities list,
            and provenance_hash.

        Raises:
            ValueError: If any commodity is invalid or list is empty.
        """
        if not commodities:
            raise ValueError("commodities list must not be empty")
        for c in commodities:
            if c not in EUDR_COMMODITIES:
                raise ValueError(
                    f"'{c}' is not a valid EUDR commodity"
                )

        # Remove duplicates while preserving order
        seen: set = set()
        unique_commodities: List[str] = []
        for c in commodities:
            if c not in seen:
                seen.add(c)
                unique_commodities.append(c)

        matrix: Dict[str, Dict[str, str]] = {}
        for c1 in unique_commodities:
            matrix[c1] = {}
            for c2 in unique_commodities:
                corr = self._get_correlation(c1, c2)
                matrix[c1][c2] = str(corr)

        result = {
            "commodities": unique_commodities,
            "matrix": matrix,
            "size": len(unique_commodities),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.debug(
            "Correlation matrix calculated for %d commodities",
            len(unique_commodities),
        )
        return result

    def recommend_diversification(
        self,
        current_portfolio: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate diversification recommendations for the portfolio.

        Analyzes current concentration and suggests commodity weight
        adjustments to improve diversification while managing risk.

        Args:
            current_portfolio: List of current commodity position dicts.

        Returns:
            Dictionary with recommendations, suggested_weights,
            projected_metrics, and provenance_hash.

        Raises:
            ValueError: If portfolio is empty or invalid.
        """
        start_time = time.monotonic()

        self._validate_positions(current_portfolio)
        positions = self._parse_positions(current_portfolio)
        positions = self._normalize_weights(positions)

        current_hhi = self._calculate_hhi(positions)
        current_diversification = self._calculate_diversification(current_hhi)
        current_risk = self._calculate_weighted_risk(positions)

        recommendations: List[str] = []
        suggested_weights: Dict[str, str] = {}

        # Get current commodity set
        current_commodities = {p.commodity for p in positions}
        missing_commodities = EUDR_COMMODITIES - current_commodities

        # Identify overweight positions
        n = len(positions)
        equal_weight = Decimal("1") / Decimal(str(max(n, 1)))
        overweight: List[str] = []
        underweight: List[str] = []

        for pos in positions:
            if pos.weight > equal_weight * Decimal("1.5"):
                overweight.append(pos.commodity)
                # Suggest reducing to 1.2x equal weight
                suggested_weights[pos.commodity] = str(
                    (equal_weight * Decimal("1.2")).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP,
                    )
                )
                recommendations.append(
                    f"Reduce {pos.commodity} weight from "
                    f"{pos.weight:.1%} to ~{float(equal_weight * Decimal('1.2')):.1%} "
                    f"(currently overweight)"
                )
            elif pos.weight < equal_weight * Decimal("0.5") and n > 1:
                underweight.append(pos.commodity)
                suggested_weights[pos.commodity] = str(
                    (equal_weight * Decimal("0.8")).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP,
                    )
                )
                recommendations.append(
                    f"Consider increasing {pos.commodity} weight from "
                    f"{pos.weight:.1%} to ~{float(equal_weight * Decimal('0.8')):.1%}"
                )
            else:
                suggested_weights[pos.commodity] = str(pos.weight)

        # Suggest adding missing low-risk commodities
        for mc in sorted(missing_commodities):
            risk_params = COMMODITY_RISK_PARAMETERS.get(mc, {})
            base_risk = risk_params.get("base_risk", Decimal("50"))
            if base_risk < current_risk:
                recommendations.append(
                    f"Consider adding {mc} (base risk {base_risk}) to "
                    f"improve diversification"
                )

        # Concentration-specific recommendations
        if current_hhi > HHI_MODERATE:
            recommendations.append(
                f"Portfolio is highly concentrated (HHI={current_hhi}). "
                f"Target HHI below {HHI_MODERATE} for moderate concentration."
            )
        elif current_hhi > HHI_UNCONCENTRATED:
            recommendations.append(
                f"Portfolio has moderate concentration (HHI={current_hhi}). "
                f"Consider further diversification below {HHI_UNCONCENTRATED}."
            )

        # Projected metrics with equal weights
        if n > 0:
            equal_positions = []
            for pos in positions:
                ep = CommodityPosition(
                    commodity=pos.commodity,
                    weight=equal_weight,
                    exposure_value=pos.exposure_value,
                    risk_score=pos.risk_score,
                    supplier_count=pos.supplier_count,
                    origin_countries=pos.origin_countries,
                )
                equal_positions.append(ep)
            projected_hhi = self._calculate_hhi(equal_positions)
            projected_div = self._calculate_diversification(projected_hhi)
            projected_risk = self._calculate_weighted_risk(equal_positions)
        else:
            projected_hhi = Decimal("0")
            projected_div = Decimal("0")
            projected_risk = Decimal("0")

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "current_metrics": {
                "hhi": str(current_hhi),
                "diversification_score": str(current_diversification),
                "weighted_risk": str(current_risk),
                "commodity_count": len(positions),
            },
            "projected_metrics_equal_weight": {
                "hhi": str(projected_hhi),
                "diversification_score": str(projected_div),
                "weighted_risk": str(projected_risk),
            },
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "suggested_weights": suggested_weights,
            "overweight_commodities": overweight,
            "underweight_commodities": underweight,
            "missing_commodities": sorted(missing_commodities),
            "processing_time_ms": round(processing_time_ms, 3),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Diversification recommendations: %d suggestions, "
            "current HHI=%s, projected HHI=%s, time_ms=%.1f",
            len(recommendations), current_hhi, projected_hhi,
            processing_time_ms,
        )
        return result

    def calculate_var(
        self,
        commodity_positions: List[Dict[str, Any]],
        confidence: float = 0.95,
    ) -> Decimal:
        """Calculate Value at Risk for the commodity portfolio.

        Uses the parametric (variance-covariance) method with static
        commodity volatilities and correlation data.

        Args:
            commodity_positions: List of position dicts with ``weight``
                and ``exposure_value``.
            confidence: Confidence level (0.90, 0.95, or 0.99).

        Returns:
            Decimal VaR value.

        Raises:
            ValueError: If confidence is not 0.90, 0.95, or 0.99.
        """
        conf_str = f"{confidence:.2f}"
        if conf_str not in VAR_Z_SCORES:
            raise ValueError(
                f"Confidence {confidence} not supported. "
                f"Valid: 0.90, 0.95, 0.99"
            )

        self._validate_positions(commodity_positions)
        positions = self._parse_positions(commodity_positions)
        positions = self._normalize_weights(positions)

        return self._calculate_var_internal(
            positions, _to_decimal(confidence),
        )

    def compare_portfolios(
        self,
        portfolio_a: List[Dict[str, Any]],
        portfolio_b: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare two portfolio configurations.

        Computes all risk metrics for both portfolios and provides a
        delta analysis highlighting the differences.

        Args:
            portfolio_a: First portfolio position list.
            portfolio_b: Second portfolio position list.

        Returns:
            Dictionary with metrics_a, metrics_b, delta, and
            recommendation on which portfolio is better diversified.

        Raises:
            ValueError: If either portfolio is empty or invalid.
        """
        start_time = time.monotonic()

        # Analyze both portfolios
        self._validate_positions(portfolio_a)
        self._validate_positions(portfolio_b)

        pos_a = self._parse_positions(portfolio_a)
        pos_a = self._normalize_weights(pos_a)
        pos_b = self._parse_positions(portfolio_b)
        pos_b = self._normalize_weights(pos_b)

        # Metrics A
        hhi_a = self._calculate_hhi(pos_a)
        div_a = self._calculate_diversification(hhi_a)
        risk_a = self._calculate_weighted_risk(pos_a)
        exposure_a = self._calculate_total_risk_exposure(pos_a)
        var_a = self._calculate_var_internal(pos_a, Decimal("0.95"))

        # Metrics B
        hhi_b = self._calculate_hhi(pos_b)
        div_b = self._calculate_diversification(hhi_b)
        risk_b = self._calculate_weighted_risk(pos_b)
        exposure_b = self._calculate_total_risk_exposure(pos_b)
        var_b = self._calculate_var_internal(pos_b, Decimal("0.95"))

        # Recommendation
        if div_a > div_b and risk_a < risk_b:
            recommendation = "Portfolio A is better diversified with lower risk"
        elif div_b > div_a and risk_b < risk_a:
            recommendation = "Portfolio B is better diversified with lower risk"
        elif div_a > div_b:
            recommendation = (
                "Portfolio A is better diversified but has "
                "higher weighted risk"
            )
        elif div_b > div_a:
            recommendation = (
                "Portfolio B is better diversified but has "
                "higher weighted risk"
            )
        else:
            recommendation = "Both portfolios have similar risk profiles"

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "portfolio_a": {
                "commodity_count": len(pos_a),
                "hhi": str(hhi_a),
                "diversification_score": str(div_a),
                "weighted_risk": str(risk_a),
                "total_exposure": str(exposure_a),
                "var_95": str(var_a),
                "commodities": [p.commodity for p in pos_a],
            },
            "portfolio_b": {
                "commodity_count": len(pos_b),
                "hhi": str(hhi_b),
                "diversification_score": str(div_b),
                "weighted_risk": str(risk_b),
                "total_exposure": str(exposure_b),
                "var_95": str(var_b),
                "commodities": [p.commodity for p in pos_b],
            },
            "delta": {
                "hhi_diff": str((hhi_b - hhi_a).quantize(Decimal("0.01"))),
                "diversification_diff": str(
                    (div_b - div_a).quantize(Decimal("0.01"))
                ),
                "risk_diff": str(
                    (risk_b - risk_a).quantize(Decimal("0.01"))
                ),
                "exposure_diff": str(
                    (exposure_b - exposure_a).quantize(Decimal("0.01"))
                ),
                "var_diff": str(
                    (var_b - var_a).quantize(Decimal("0.01"))
                ),
            },
            "recommendation": recommendation,
            "processing_time_ms": round(processing_time_ms, 3),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Portfolio comparison: A(HHI=%s,risk=%s) vs B(HHI=%s,risk=%s) "
            "time_ms=%.1f",
            hhi_a, risk_a, hhi_b, risk_b, processing_time_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_positions(
        self,
        positions: List[Dict[str, Any]],
    ) -> None:
        """Validate commodity positions input.

        Args:
            positions: Raw position list.

        Raises:
            ValueError: If invalid.
        """
        if not isinstance(positions, list) or not positions:
            raise ValueError("commodity_positions must be a non-empty list")
        if len(positions) > MAX_PORTFOLIO_POSITIONS:
            raise ValueError(
                f"Portfolio size {len(positions)} exceeds maximum "
                f"{MAX_PORTFOLIO_POSITIONS}"
            )
        for pos in positions:
            commodity = pos.get("commodity", "")
            if commodity not in EUDR_COMMODITIES:
                raise ValueError(
                    f"'{commodity}' is not a valid EUDR commodity"
                )

    def _parse_positions(
        self,
        raw_positions: List[Dict[str, Any]],
    ) -> List[CommodityPosition]:
        """Parse raw position dicts into CommodityPosition objects.

        Args:
            raw_positions: Raw position dictionaries.

        Returns:
            List of CommodityPosition instances.
        """
        positions: List[CommodityPosition] = []
        for raw in raw_positions:
            commodity = raw.get("commodity", "")
            risk_params = COMMODITY_RISK_PARAMETERS.get(commodity, {})

            weight = _to_decimal(raw.get("weight", 0))
            exposure = _to_decimal(raw.get("exposure_value", 0))
            risk = raw.get("risk_score")
            if risk is not None:
                risk_score = _to_decimal(risk)
            else:
                risk_score = risk_params.get("base_risk", Decimal("50"))

            positions.append(CommodityPosition(
                commodity=commodity,
                weight=weight,
                exposure_value=exposure,
                risk_score=risk_score,
                supplier_count=raw.get("supplier_count", 0),
                origin_countries=raw.get("origin_countries", []),
            ))

        return positions

    def _normalize_weights(
        self,
        positions: List[CommodityPosition],
    ) -> List[CommodityPosition]:
        """Normalize portfolio weights to sum to 1.0.

        Args:
            positions: List of positions.

        Returns:
            Positions with normalized weights.
        """
        total_weight = sum(p.weight for p in positions)
        if total_weight <= 0:
            # Equal weight fallback
            n = Decimal(str(len(positions)))
            for p in positions:
                p.weight = (Decimal("1") / n).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP,
                )
        elif total_weight != Decimal("1"):
            for p in positions:
                p.weight = (p.weight / total_weight).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP,
                )
        return positions

    def _calculate_hhi(
        self,
        positions: List[CommodityPosition],
    ) -> Decimal:
        """Calculate Herfindahl-Hirschman Index.

        HHI = sum of (market_share_pct)^2 for each position.
        Market share is weight * 100.

        Args:
            positions: Normalized positions.

        Returns:
            Decimal HHI value (0-10000).
        """
        hhi = Decimal("0")
        for pos in positions:
            share_pct = pos.weight * Decimal("100")
            hhi += share_pct ** 2

        return hhi.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _classify_concentration(self, hhi: Decimal) -> str:
        """Classify HHI concentration level.

        Args:
            hhi: HHI value.

        Returns:
            Classification string.
        """
        if hhi > HHI_MODERATE:
            return "HIGHLY_CONCENTRATED"
        if hhi > HHI_UNCONCENTRATED:
            return "MODERATELY_CONCENTRATED"
        return "UNCONCENTRATED"

    def _calculate_diversification(self, hhi: Decimal) -> Decimal:
        """Calculate diversification score from HHI.

        Score = 100 * (1 - HHI/10000). Range 0-100.

        Args:
            hhi: HHI value.

        Returns:
            Decimal diversification score.
        """
        if hhi >= HHI_MAX:
            return Decimal("0")
        score = Decimal("100") * (Decimal("1") - hhi / HHI_MAX)
        return max(
            Decimal("0"),
            min(Decimal("100"), score.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )),
        )

    def _calculate_weighted_risk(
        self,
        positions: List[CommodityPosition],
    ) -> Decimal:
        """Calculate portfolio-weighted average risk score.

        Args:
            positions: Normalized positions.

        Returns:
            Decimal weighted risk score.
        """
        if not positions:
            return Decimal("0")
        weighted_sum = sum(
            p.weight * p.risk_score for p in positions
        )
        return weighted_sum.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_total_risk_exposure(
        self,
        positions: List[CommodityPosition],
    ) -> Decimal:
        """Calculate total risk-adjusted exposure.

        Total exposure = sum of (exposure_value * risk_score / 100).

        Args:
            positions: Position list.

        Returns:
            Decimal total risk exposure.
        """
        total = Decimal("0")
        for p in positions:
            total += p.exposure_value * p.risk_score / Decimal("100")
        return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_var_internal(
        self,
        positions: List[CommodityPosition],
        confidence: Decimal,
    ) -> Decimal:
        """Calculate parametric Value at Risk.

        VaR = z_score * portfolio_volatility * total_exposure

        Portfolio volatility = sqrt(sum of w_i * w_j * sigma_i * sigma_j * rho_ij)

        Args:
            positions: Normalized positions.
            confidence: Confidence level (Decimal).

        Returns:
            Decimal VaR value.
        """
        conf_str = str(confidence.quantize(Decimal("0.01")))
        z_score = VAR_Z_SCORES.get(conf_str, Decimal("1.645"))

        total_exposure = sum(p.exposure_value for p in positions)
        if total_exposure <= 0:
            return Decimal("0")

        # Portfolio variance
        portfolio_variance = Decimal("0")
        for i, pi in enumerate(positions):
            for j, pj in enumerate(positions):
                wi = pi.weight
                wj = pj.weight
                vol_i = COMMODITY_RISK_PARAMETERS.get(
                    pi.commodity, {},
                ).get("volatility", Decimal("0.25"))
                vol_j = COMMODITY_RISK_PARAMETERS.get(
                    pj.commodity, {},
                ).get("volatility", Decimal("0.25"))
                rho = self._get_correlation(pi.commodity, pj.commodity)
                portfolio_variance += wi * wj * vol_i * vol_j * rho

        # Portfolio volatility (sqrt of variance)
        if portfolio_variance <= 0:
            return Decimal("0")

        # Decimal sqrt via float conversion (precision adequate for VaR)
        vol_float = float(portfolio_variance)
        portfolio_vol = Decimal(str(math.sqrt(max(0, vol_float))))

        var_value = z_score * portfolio_vol * total_exposure
        return var_value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _get_correlation(
        self,
        commodity_a: str,
        commodity_b: str,
    ) -> Decimal:
        """Get risk correlation between two commodities.

        Matrix is symmetric: corr(a,b) = corr(b,a).

        Args:
            commodity_a: First commodity.
            commodity_b: Second commodity.

        Returns:
            Decimal correlation coefficient (-1.0 to 1.0).
        """
        corr = RISK_CORRELATIONS.get((commodity_a, commodity_b))
        if corr is not None:
            return corr
        corr = RISK_CORRELATIONS.get((commodity_b, commodity_a))
        if corr is not None:
            return corr
        # Default low correlation for unknown pairs
        if commodity_a == commodity_b:
            return Decimal("1.0")
        return Decimal("0.10")

    def _apply_scenario(
        self,
        positions: List[CommodityPosition],
        scenario_type: str,
        parameters: Dict[str, Any],
    ) -> List[CommodityPosition]:
        """Apply scenario impacts to positions.

        Args:
            positions: Original positions.
            scenario_type: Scenario type.
            parameters: Scenario parameters.

        Returns:
            New list of adjusted CommodityPosition instances.
        """
        impacts = SCENARIO_IMPACTS.get(scenario_type, {})
        magnitude = _to_decimal(parameters.get("magnitude", "1.0"))
        affected = parameters.get("affected_commodity", "all")
        severity_map = {
            "low": Decimal("0.5"),
            "medium": Decimal("1.0"),
            "high": Decimal("1.5"),
        }
        severity_mult = severity_map.get(
            str(parameters.get("severity", "medium")).lower(),
            Decimal("1.0"),
        )

        adjusted: List[CommodityPosition] = []
        for pos in positions:
            commodity_impact = impacts.get(pos.commodity, {})
            risk_delta = commodity_impact.get("risk_delta", Decimal("10"))
            exposure_mult = commodity_impact.get(
                "exposure_multiplier", Decimal("1.1"),
            )

            apply_impact = (affected == "all" or affected == pos.commodity)

            if apply_impact:
                # Adjust based on scenario type
                if scenario_type == "price_shock":
                    adj_risk = pos.risk_score + risk_delta * abs(magnitude)
                    adj_exposure = pos.exposure_value * exposure_mult
                elif scenario_type == "supply_disruption":
                    adj_risk = pos.risk_score + risk_delta * severity_mult
                    adj_exposure = pos.exposure_value * exposure_mult * severity_mult
                elif scenario_type == "regulatory_change":
                    adj_risk = pos.risk_score + risk_delta * severity_mult
                    adj_exposure = pos.exposure_value * exposure_mult
                elif scenario_type == "climate_event":
                    adj_risk = pos.risk_score + risk_delta * severity_mult
                    adj_exposure = pos.exposure_value * exposure_mult * severity_mult
                else:
                    adj_risk = pos.risk_score + risk_delta
                    adj_exposure = pos.exposure_value * exposure_mult

                adj_risk = min(Decimal("100"), max(Decimal("0"), adj_risk))
            else:
                adj_risk = pos.risk_score
                adj_exposure = pos.exposure_value

            adjusted.append(CommodityPosition(
                commodity=pos.commodity,
                weight=pos.weight,
                exposure_value=adj_exposure.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                ),
                risk_score=adj_risk.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                ),
                supplier_count=pos.supplier_count,
                origin_countries=list(pos.origin_countries),
            ))

        return adjusted

    def _safe_pct_change(
        self,
        baseline: Decimal,
        scenario: Decimal,
    ) -> Decimal:
        """Compute percentage change safely (avoid division by zero).

        Args:
            baseline: Baseline value.
            scenario: Scenario value.

        Returns:
            Decimal percentage change.
        """
        if baseline == 0:
            return Decimal("0")
        pct = ((scenario - baseline) / baseline * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        return pct
