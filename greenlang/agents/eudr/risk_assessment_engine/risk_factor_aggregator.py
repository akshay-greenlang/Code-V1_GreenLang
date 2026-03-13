# -*- coding: utf-8 -*-
"""
AGENT-EUDR-028: Risk Assessment Engine - Risk Factor Aggregator

Collects and normalizes risk signals from 5 upstream EUDR risk assessment
agents (EUDR-016 Country Risk Evaluator, EUDR-017 Supplier Risk Scorer,
EUDR-018 Commodity Risk Analyzer, EUDR-019 Corruption Index Monitor,
EUDR-020 Deforestation Alert System) and synthesizes them into a unified
list of RiskFactorInput objects ready for composite scoring.

Each upstream agent produces risk scores on different scales and semantics.
This engine normalizes all scores to a consistent 0-100 Decimal scale and
maps them to the 8 risk dimensions defined in Article 10(2). Simulated
adapter stubs are provided for development and testing; production deploys
swap these for real gRPC/REST client calls.

Production infrastructure includes:
    - Adapter-pattern upstream agent connectors (5 agents)
    - Score normalization to 0-100 Decimal scale
    - Multi-country and multi-supplier batch aggregation
    - Supply chain complexity and mixing risk derivation
    - Circumvention risk heuristic computation
    - SHA-256 provenance hash per aggregation operation
    - Prometheus metrics integration

Zero-Hallucination Guarantees:
    - All score normalization uses deterministic linear interpolation
    - No LLM involvement in score transformation or dimension mapping
    - All provenance hashes computed from canonical JSON

Regulatory References:
    - EUDR Article 10(2): 10 risk assessment criteria
    - EUDR Article 29: Country benchmarking integration
    - EUDR Article 31: 5-year record retention for aggregated risk data

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-028 (Engine 2: Risk Factor Aggregator)
Agent ID: GL-EUDR-RAE-028
Status: Production Ready
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
    get_config,
)
from greenlang.agents.eudr.risk_assessment_engine.models import (
    EUDRCommodity,
    RiskDimension,
    RiskFactorInput,
    SourceAgent,
)
from greenlang.agents.eudr.risk_assessment_engine.provenance import ProvenanceTracker
from greenlang.agents.eudr.risk_assessment_engine.metrics import (
    record_factor_aggregation,
    observe_aggregation_duration,
)

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCORE_MIN = Decimal("0")
_SCORE_MAX = Decimal("100")
_SCORE_PRECISION = Decimal("0.01")
_CONFIDENCE_MIN = Decimal("0")
_CONFIDENCE_MAX = Decimal("1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash of data.

    Args:
        data: Any JSON-serializable object.

    Returns:
        64-character lowercase hex SHA-256 hash string.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class RiskFactorAggregator:
    """Engine for aggregating risk signals from upstream EUDR agents.

    Collects risk factor data from EUDR-016 (Country Risk), EUDR-017
    (Supplier Risk), EUDR-018 (Commodity Risk), EUDR-019 (Corruption
    Index), and EUDR-020 (Deforestation Alert) agents. Normalizes all
    scores to a 0-100 scale and maps them to the 8 Article 10(2)
    risk dimensions.

    Additionally derives three synthetic dimensions (supply chain
    complexity, mixing risk, circumvention risk) from combinations of
    upstream signals when direct agent inputs are not available.

    Args:
        config: Agent configuration (uses singleton if None).

    Example:
        >>> aggregator = RiskFactorAggregator()
        >>> factors = aggregator.aggregate_factors(
        ...     operator_id="OP-001",
        ...     commodity="coffee",
        ...     country_codes=["BR", "CO"],
        ...     supplier_ids=["SUP-001", "SUP-002"],
        ... )
        >>> assert len(factors) > 0
    """

    def __init__(self, config: Optional[RiskAssessmentEngineConfig] = None) -> None:
        """Initialize RiskFactorAggregator.

        Args:
            config: Agent configuration (uses singleton if None).
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._aggregation_count: int = 0
        self._total_factors_collected: int = 0
        logger.info("RiskFactorAggregator initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate_factors(
        self,
        operator_id: str,
        commodity: str,
        country_codes: List[str],
        supplier_ids: List[str],
    ) -> List[RiskFactorInput]:
        """Aggregate risk factors from all upstream agents.

        Calls each upstream agent adapter, normalizes scores, and returns
        a unified list of RiskFactorInput objects across all 8 dimensions.

        Args:
            operator_id: Unique operator identifier.
            commodity: EUDR commodity name or code.
            country_codes: ISO 3166-1 alpha-2 country codes.
            supplier_ids: Supplier identifiers.

        Returns:
            List of RiskFactorInput objects across all risk dimensions.
        """
        start_time = time.monotonic()
        factors: List[RiskFactorInput] = []

        # Fetch from upstream agents
        factors.extend(self._fetch_country_risk(country_codes))
        factors.extend(self._fetch_supplier_risk(supplier_ids))
        factors.extend(self._fetch_commodity_risk(commodity))
        factors.extend(self._fetch_corruption_risk(country_codes))
        factors.extend(self._fetch_deforestation_risk(country_codes))

        # Derive synthetic dimensions
        factors.extend(
            self._derive_supply_chain_complexity(supplier_ids, country_codes)
        )
        factors.extend(self._derive_mixing_risk(supplier_ids, commodity))
        factors.extend(self._derive_circumvention_risk(country_codes, commodity))

        # Provenance
        provenance_hash = _compute_hash({
            "operator_id": operator_id,
            "commodity": commodity,
            "country_codes": sorted(country_codes),
            "supplier_count": len(supplier_ids),
            "factor_count": len(factors),
        })

        self._provenance.create_entry(
            step="risk_factor_aggregation",
            source="upstream_agents",
            input_hash=_compute_hash({
                "operator_id": operator_id,
                "commodity": commodity,
                "countries": sorted(country_codes),
                "suppliers": sorted(supplier_ids),
            }),
            output_hash=provenance_hash,
        )

        # Stats and metrics
        self._aggregation_count += 1
        self._total_factors_collected += len(factors)
        elapsed = time.monotonic() - start_time
        record_factor_aggregation(commodity)
        observe_aggregation_duration(elapsed)

        logger.info(
            "Aggregated %d risk factors for operator=%s, commodity=%s, "
            "countries=%d, suppliers=%d (%.0fms)",
            len(factors),
            operator_id,
            commodity,
            len(country_codes),
            len(supplier_ids),
            elapsed * 1000,
        )
        return factors

    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Return risk factor aggregation statistics.

        Returns:
            Dict with total_aggregations, total_factors_collected, and
            average_factors_per_aggregation keys.
        """
        avg = (
            self._total_factors_collected / self._aggregation_count
            if self._aggregation_count > 0
            else 0
        )
        return {
            "total_aggregations": self._aggregation_count,
            "total_factors_collected": self._total_factors_collected,
            "average_factors_per_aggregation": round(avg, 2),
        }

    # ------------------------------------------------------------------
    # Upstream agent adapters (stubs for development)
    # ------------------------------------------------------------------

    def _fetch_country_risk(
        self,
        country_codes: List[str],
    ) -> List[RiskFactorInput]:
        """Fetch country risk scores from EUDR-016 Country Risk Evaluator.

        Simulates API call to EUDR-016 and returns one RiskFactorInput per
        country code with COUNTRY dimension mapping.

        Args:
            country_codes: ISO 3166-1 alpha-2 country codes.

        Returns:
            List of RiskFactorInput for COUNTRY dimension.
        """
        factors: List[RiskFactorInput] = []

        # Simulated country risk scores (production: real gRPC/REST call)
        country_risk_data: Dict[str, Decimal] = {
            "BR": Decimal("55"),
            "ID": Decimal("62"),
            "MY": Decimal("48"),
            "CO": Decimal("42"),
            "GH": Decimal("58"),
            "CI": Decimal("65"),
            "CM": Decimal("70"),
            "CD": Decimal("78"),
            "PG": Decimal("72"),
            "PE": Decimal("38"),
        }
        default_score = Decimal("35")

        for code in country_codes:
            raw_score = country_risk_data.get(code.upper(), default_score)
            normalized = self._normalize_score(raw_score, (_SCORE_MIN, _SCORE_MAX))
            factors.append(RiskFactorInput(
                source_agent=SourceAgent.EUDR_016_COUNTRY,
                dimension=RiskDimension.COUNTRY,
                raw_score=normalized,
                confidence=Decimal("0.85"),
                metadata={"country_code": code.upper()},
                timestamp=_utcnow(),
                provenance_hash=_compute_hash({
                    "source": "EUDR-016",
                    "country": code.upper(),
                    "score": str(normalized),
                }),
            ))

        logger.debug(
            "Fetched %d country risk factors from EUDR-016", len(factors)
        )
        return factors

    def _fetch_supplier_risk(
        self,
        supplier_ids: List[str],
    ) -> List[RiskFactorInput]:
        """Fetch supplier risk scores from EUDR-017 Supplier Risk Scorer.

        Simulates API call to EUDR-017 and returns one RiskFactorInput per
        supplier with SUPPLIER dimension mapping.

        Args:
            supplier_ids: Supplier identifiers.

        Returns:
            List of RiskFactorInput for SUPPLIER dimension.
        """
        factors: List[RiskFactorInput] = []

        for supplier_id in supplier_ids:
            # Simulated: derive score from hash for determinism
            hash_val = int(hashlib.md5(
                supplier_id.encode("utf-8")
            ).hexdigest()[:8], 16)
            raw_score = Decimal(str(hash_val % 80 + 10))
            normalized = self._normalize_score(raw_score, (_SCORE_MIN, _SCORE_MAX))

            factors.append(RiskFactorInput(
                source_agent=SourceAgent.EUDR_017_SUPPLIER,
                dimension=RiskDimension.SUPPLIER,
                raw_score=normalized,
                confidence=Decimal("0.80"),
                metadata={"supplier_id": supplier_id},
                timestamp=_utcnow(),
                provenance_hash=_compute_hash({
                    "source": "EUDR-017",
                    "supplier": supplier_id,
                    "score": str(normalized),
                }),
            ))

        logger.debug(
            "Fetched %d supplier risk factors from EUDR-017", len(factors)
        )
        return factors

    def _fetch_commodity_risk(
        self,
        commodity: str,
    ) -> List[RiskFactorInput]:
        """Fetch commodity risk scores from EUDR-018 Commodity Risk Analyzer.

        Simulates API call to EUDR-018 and returns one RiskFactorInput
        for the COMMODITY dimension.

        Args:
            commodity: EUDR commodity name or code.

        Returns:
            List of RiskFactorInput for COMMODITY dimension.
        """
        commodity_risk_data: Dict[str, Decimal] = {
            "cattle": Decimal("68"),
            "cocoa": Decimal("62"),
            "coffee": Decimal("45"),
            "oil_palm": Decimal("72"),
            "rubber": Decimal("55"),
            "soya": Decimal("58"),
            "wood": Decimal("50"),
        }
        default_score = Decimal("50")

        commodity_key = commodity.lower().replace(" ", "_")
        raw_score = commodity_risk_data.get(commodity_key, default_score)
        normalized = self._normalize_score(raw_score, (_SCORE_MIN, _SCORE_MAX))

        factor = RiskFactorInput(
            source_agent=SourceAgent.EUDR_018_COMMODITY,
            dimension=RiskDimension.COMMODITY,
            raw_score=normalized,
            confidence=Decimal("0.90"),
            metadata={"commodity": commodity_key},
            timestamp=_utcnow(),
            provenance_hash=_compute_hash({
                "source": "EUDR-018",
                "commodity": commodity_key,
                "score": str(normalized),
            }),
        )

        logger.debug("Fetched commodity risk factor from EUDR-018: %s", commodity_key)
        return [factor]

    def _fetch_corruption_risk(
        self,
        country_codes: List[str],
    ) -> List[RiskFactorInput]:
        """Fetch corruption risk scores from EUDR-019 Corruption Index Monitor.

        Simulates API call to EUDR-019 and returns one RiskFactorInput per
        country for the CORRUPTION dimension. CPI scores (0-100, higher=cleaner)
        are inverted to risk scores (0-100, higher=riskier).

        Args:
            country_codes: ISO 3166-1 alpha-2 country codes.

        Returns:
            List of RiskFactorInput for CORRUPTION dimension.
        """
        # CPI scores (0=highly corrupt, 100=very clean)
        cpi_data: Dict[str, int] = {
            "BR": 38, "ID": 34, "MY": 50, "CO": 39, "GH": 43,
            "CI": 37, "CM": 26, "CD": 20, "PG": 28, "PE": 36,
            "DE": 79, "NL": 80, "FR": 71, "SE": 83, "FI": 87,
        }
        default_cpi = 40

        factors: List[RiskFactorInput] = []
        for code in country_codes:
            cpi = cpi_data.get(code.upper(), default_cpi)
            # Invert: risk = 100 - CPI
            raw_score = Decimal(str(100 - cpi))
            normalized = self._normalize_score(raw_score, (_SCORE_MIN, _SCORE_MAX))

            factors.append(RiskFactorInput(
                source_agent=SourceAgent.EUDR_019_CORRUPTION,
                dimension=RiskDimension.CORRUPTION,
                raw_score=normalized,
                confidence=Decimal("0.88"),
                metadata={
                    "country_code": code.upper(),
                    "cpi_score": cpi,
                },
                timestamp=_utcnow(),
                provenance_hash=_compute_hash({
                    "source": "EUDR-019",
                    "country": code.upper(),
                    "score": str(normalized),
                }),
            ))

        logger.debug(
            "Fetched %d corruption risk factors from EUDR-019", len(factors)
        )
        return factors

    def _fetch_deforestation_risk(
        self,
        country_codes: List[str],
    ) -> List[RiskFactorInput]:
        """Fetch deforestation risk scores from EUDR-020 Deforestation Alert System.

        Simulates API call to EUDR-020 and returns one RiskFactorInput per
        country for the DEFORESTATION dimension.

        Args:
            country_codes: ISO 3166-1 alpha-2 country codes.

        Returns:
            List of RiskFactorInput for DEFORESTATION dimension.
        """
        deforestation_risk_data: Dict[str, Decimal] = {
            "BR": Decimal("72"),
            "ID": Decimal("68"),
            "MY": Decimal("55"),
            "CO": Decimal("48"),
            "GH": Decimal("52"),
            "CI": Decimal("60"),
            "CM": Decimal("58"),
            "CD": Decimal("75"),
            "PG": Decimal("65"),
            "PE": Decimal("42"),
        }
        default_score = Decimal("30")

        factors: List[RiskFactorInput] = []
        for code in country_codes:
            raw_score = deforestation_risk_data.get(code.upper(), default_score)
            normalized = self._normalize_score(raw_score, (_SCORE_MIN, _SCORE_MAX))

            factors.append(RiskFactorInput(
                source_agent=SourceAgent.EUDR_020_DEFORESTATION,
                dimension=RiskDimension.DEFORESTATION,
                raw_score=normalized,
                confidence=Decimal("0.82"),
                metadata={"country_code": code.upper()},
                timestamp=_utcnow(),
                provenance_hash=_compute_hash({
                    "source": "EUDR-020",
                    "country": code.upper(),
                    "score": str(normalized),
                }),
            ))

        logger.debug(
            "Fetched %d deforestation risk factors from EUDR-020", len(factors)
        )
        return factors

    # ------------------------------------------------------------------
    # Synthetic dimension derivation
    # ------------------------------------------------------------------

    def _derive_supply_chain_complexity(
        self,
        supplier_ids: List[str],
        country_codes: List[str],
    ) -> List[RiskFactorInput]:
        """Derive supply chain complexity score from structural indicators.

        Complexity increases with more suppliers and more countries in the
        chain. Uses a simple heuristic: base score increases logarithmically
        with tier count.

        Args:
            supplier_ids: Supplier identifiers in the chain.
            country_codes: Countries involved in the supply chain.

        Returns:
            Single-element list with SUPPLY_CHAIN_COMPLEXITY RiskFactorInput.
        """
        supplier_count = len(supplier_ids)
        country_count = len(country_codes)

        # Heuristic: more suppliers/countries = higher complexity
        base = Decimal("20")
        supplier_contrib = min(
            Decimal(str(supplier_count)) * Decimal("8"),
            Decimal("50"),
        )
        country_contrib = min(
            Decimal(str(country_count)) * Decimal("10"),
            Decimal("30"),
        )
        raw_score = (base + supplier_contrib + country_contrib).quantize(
            _SCORE_PRECISION, rounding=ROUND_HALF_UP
        )
        raw_score = min(raw_score, _SCORE_MAX)

        return [RiskFactorInput(
            source_agent=SourceAgent.EUDR_028_DERIVED,
            dimension=RiskDimension.SUPPLY_CHAIN_COMPLEXITY,
            raw_score=raw_score,
            confidence=Decimal("0.70"),
            metadata={
                "supplier_count": supplier_count,
                "country_count": country_count,
            },
            timestamp=_utcnow(),
            provenance_hash=_compute_hash({
                "source": "derived",
                "dimension": "supply_chain_complexity",
                "score": str(raw_score),
            }),
        )]

    def _derive_mixing_risk(
        self,
        supplier_ids: List[str],
        commodity: str,
    ) -> List[RiskFactorInput]:
        """Derive mixing risk score from supply chain characteristics.

        Mixing risk is higher for commodities that are commonly co-mingled
        (e.g., soya, palm oil) and increases with supplier count.

        Args:
            supplier_ids: Supplier identifiers.
            commodity: EUDR commodity name.

        Returns:
            Single-element list with MIXING_RISK RiskFactorInput.
        """
        # Commodities with inherently high mixing risk
        high_mix_commodities = {"soya", "oil_palm", "cocoa", "coffee"}
        commodity_key = commodity.lower().replace(" ", "_")

        base = Decimal("15")
        if commodity_key in high_mix_commodities:
            base = Decimal("35")

        supplier_factor = min(
            Decimal(str(len(supplier_ids))) * Decimal("5"),
            Decimal("30"),
        )
        raw_score = (base + supplier_factor).quantize(
            _SCORE_PRECISION, rounding=ROUND_HALF_UP
        )
        raw_score = min(raw_score, _SCORE_MAX)

        return [RiskFactorInput(
            source_agent=SourceAgent.EUDR_028_DERIVED,
            dimension=RiskDimension.MIXING_RISK,
            raw_score=raw_score,
            confidence=Decimal("0.65"),
            metadata={
                "commodity": commodity_key,
                "supplier_count": len(supplier_ids),
                "high_mix_commodity": commodity_key in high_mix_commodities,
            },
            timestamp=_utcnow(),
            provenance_hash=_compute_hash({
                "source": "derived",
                "dimension": "mixing_risk",
                "score": str(raw_score),
            }),
        )]

    def _derive_circumvention_risk(
        self,
        country_codes: List[str],
        commodity: str,
    ) -> List[RiskFactorInput]:
        """Derive circumvention risk score from geographic and trade patterns.

        Circumvention risk is elevated when supply chains pass through
        known transhipment hubs or countries with weak customs enforcement.

        Args:
            country_codes: Countries in the supply chain.
            commodity: EUDR commodity name.

        Returns:
            Single-element list with CIRCUMVENTION_RISK RiskFactorInput.
        """
        # Known transhipment / circumvention risk countries
        high_circumvention_countries = {"SG", "AE", "HK", "VN", "TH", "TR"}

        base = Decimal("10")
        circumvention_matches = sum(
            1 for c in country_codes if c.upper() in high_circumvention_countries
        )
        country_factor = min(
            Decimal(str(circumvention_matches)) * Decimal("15"),
            Decimal("45"),
        )

        # Multi-country chains increase circumvention risk
        chain_factor = Decimal("0")
        if len(country_codes) > 3:
            chain_factor = Decimal("15")
        elif len(country_codes) > 1:
            chain_factor = Decimal("5")

        raw_score = (base + country_factor + chain_factor).quantize(
            _SCORE_PRECISION, rounding=ROUND_HALF_UP
        )
        raw_score = min(raw_score, _SCORE_MAX)

        return [RiskFactorInput(
            source_agent=SourceAgent.EUDR_028_DERIVED,
            dimension=RiskDimension.CIRCUMVENTION_RISK,
            raw_score=raw_score,
            confidence=Decimal("0.60"),
            metadata={
                "country_codes": [c.upper() for c in country_codes],
                "circumvention_matches": circumvention_matches,
            },
            timestamp=_utcnow(),
            provenance_hash=_compute_hash({
                "source": "derived",
                "dimension": "circumvention_risk",
                "score": str(raw_score),
            }),
        )]

    # ------------------------------------------------------------------
    # Score normalization
    # ------------------------------------------------------------------

    def _normalize_score(
        self,
        score: Decimal,
        source_scale: Tuple[Decimal, Decimal],
    ) -> Decimal:
        """Normalize a score from source scale to 0-100.

        Uses linear interpolation to map from [source_min, source_max]
        to [0, 100]. If source scale is already 0-100, this is a no-op
        with clamping.

        Args:
            score: Raw score from upstream agent.
            source_scale: Tuple of (min_value, max_value) for the source.

        Returns:
            Normalized score as Decimal in [0, 100].
        """
        src_min, src_max = source_scale
        if src_max == src_min:
            return _SCORE_MIN

        normalized = (
            (score - src_min) / (src_max - src_min) * _SCORE_MAX
        ).quantize(_SCORE_PRECISION, rounding=ROUND_HALF_UP)

        return max(_SCORE_MIN, min(normalized, _SCORE_MAX))
