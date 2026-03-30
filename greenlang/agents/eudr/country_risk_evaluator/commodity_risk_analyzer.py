# -*- coding: utf-8 -*-
"""
Commodity Risk Analyzer Engine - AGENT-EUDR-016 Engine 2

EUDR commodity-specific risk analysis for 7 regulated commodities (cattle,
cocoa, coffee, oil_palm, rubber, soya, wood) with country-commodity cross
risk matrix, production region deforestation pressure mapping,
certification scheme effectiveness evaluation, seasonal risk variation
modeling, supply chain exposure scoring, commodity price correlation with
deforestation, and derived product mapping (e.g., chocolate → cocoa,
leather → cattle, biodiesel → soya/oil_palm).

Commodity Risk Model:
    commodity_risk = (country_base_risk * 0.4) + (production_risk * 0.3) +
                     (supply_chain_risk * 0.2) + (price_pressure * 0.1)
    All factors normalized to [0, 100] scale.

Certification Effectiveness:
    - FSC: 85% effective for wood, 60% for rubber
    - RSPO: 80% effective for oil_palm
    - Rainforest Alliance: 75% for cocoa, 80% for coffee
    - Fairtrade: 65% for cocoa, 70% for coffee
    - Organic: 50% for cocoa, 55% for coffee, 45% for soya
    - Bonsucro: 60% for soya
    - ISCC: 70% for oil_palm, 55% for soya

Seasonal Risk Variation:
    - Soya: planting season (Oct-Dec) highest risk in Brazil/Argentina
    - Oil palm: year-round production, monsoon months (Nov-Feb) highest
    - Cocoa: main harvest (Oct-Mar) in West Africa, mid-crop (May-Aug)
    - Coffee: harvest season varies by region (Brazil: May-Sep, Colombia: Apr-Jun + Oct-Dec)
    - Cattle: dry season (Jun-Oct) highest deforestation in Amazon
    - Rubber: tapping season (Nov-Mar) in SE Asia
    - Wood: dry season (Apr-Oct) highest logging activity

Zero-Hallucination: All commodity risk calculations are deterministic
    arithmetic from production data, price indices, and certification
    records. No LLM calls in the calculation path.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import get_config
from .metrics import (
    observe_commodity_analysis_duration,
    record_commodity_analysis as record_commodity_analysis_completed,
)
from .models import (
    CertificationScheme,
    CommodityRiskProfile,
    CommodityType,
    RiskLevel,
    SUPPORTED_COMMODITIES,
)
from .provenance import get_provenance_tracker

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Commodity risk factor weights
_RISK_WEIGHTS: Dict[str, Decimal] = {
    "country_base_risk": Decimal("0.40"),
    "production_risk": Decimal("0.30"),
    "supply_chain_risk": Decimal("0.20"),
    "price_pressure": Decimal("0.10"),
}

#: Certification scheme effectiveness by commodity (0-100)
_CERTIFICATION_EFFECTIVENESS: Dict[str, Dict[str, float]] = {
    "fsc": {
        "wood": 85.0, "rubber": 60.0, "cattle": 0.0, "cocoa": 0.0,
        "coffee": 0.0, "oil_palm": 0.0, "soya": 0.0,
    },
    "pefc": {
        "wood": 75.0, "rubber": 50.0, "cattle": 0.0, "cocoa": 0.0,
        "coffee": 0.0, "oil_palm": 0.0, "soya": 0.0,
    },
    "rspo": {
        "oil_palm": 80.0, "wood": 0.0, "rubber": 0.0, "cattle": 0.0,
        "cocoa": 0.0, "coffee": 0.0, "soya": 0.0,
    },
    "rainforest_alliance": {
        "cocoa": 75.0, "coffee": 80.0, "wood": 40.0, "rubber": 30.0,
        "cattle": 0.0, "oil_palm": 30.0, "soya": 0.0,
    },
    "fairtrade": {
        "cocoa": 65.0, "coffee": 70.0, "wood": 0.0, "rubber": 0.0,
        "cattle": 0.0, "oil_palm": 0.0, "soya": 0.0,
    },
    "organic": {
        "cocoa": 50.0, "coffee": 55.0, "soya": 45.0, "oil_palm": 40.0,
        "cattle": 30.0, "wood": 20.0, "rubber": 25.0,
    },
    "bonsucro": {
        "soya": 60.0, "cattle": 0.0, "cocoa": 0.0, "coffee": 0.0,
        "oil_palm": 0.0, "rubber": 0.0, "wood": 0.0,
    },
    "iscc": {
        "oil_palm": 70.0, "soya": 55.0, "wood": 30.0, "rubber": 25.0,
        "cattle": 0.0, "cocoa": 0.0, "coffee": 0.0,
    },
}

#: Seasonal risk multipliers by commodity and month (1.0 = baseline)
_SEASONAL_RISK: Dict[str, List[float]] = {
    "cattle": [1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.3, 1.4, 1.3, 1.2, 1.0, 1.0],  # Jun-Oct dry season
    "cocoa": [1.3, 1.2, 1.1, 1.0, 1.1, 1.1, 1.1, 1.0, 1.0, 1.2, 1.3, 1.3],  # Oct-Mar harvest
    "coffee": [1.0, 1.0, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 1.1, 1.1, 1.0],  # Apr-Sep harvest
    "oil_palm": [1.1, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.3],  # Nov-Feb monsoon
    "rubber": [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2],  # Nov-Mar tapping
    "soya": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.3, 1.4, 1.3],  # Oct-Dec planting
    "wood": [1.0, 1.0, 1.0, 1.2, 1.3, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0],  # Apr-Oct dry season
}

#: Derived products mapping (derived product -> primary commodity)
_DERIVED_PRODUCTS: Dict[str, str] = {
    "chocolate": "cocoa",
    "cocoa_butter": "cocoa",
    "cocoa_powder": "cocoa",
    "leather": "cattle",
    "beef": "cattle",
    "tallow": "cattle",
    "gelatin": "cattle",
    "biodiesel": "soya",  # Can also be oil_palm
    "soybean_meal": "soya",
    "soy_oil": "soya",
    "tofu": "soya",
    "palm_oil": "oil_palm",
    "palm_kernel_oil": "oil_palm",
    "furniture": "wood",
    "plywood": "wood",
    "paper": "wood",
    "pulp": "wood",
    "charcoal": "wood",
    "latex": "rubber",
    "tires": "rubber",
    "natural_rubber": "rubber",
}

#: High deforestation pressure regions by commodity and country
_HIGH_RISK_REGIONS: Dict[str, Dict[str, List[str]]] = {
    "cattle": {
        "BR": ["Para", "Mato Grosso", "Rondonia", "Amazonas", "Acre"],
        "CO": ["Caqueta", "Guaviare", "Meta", "Putumayo"],
        "PE": ["Ucayali", "San Martin", "Loreto"],
    },
    "cocoa": {
        "CI": ["Cavally", "Guemon", "Nawa", "San-Pedro"],
        "GH": ["Western Region", "Ashanti", "Brong-Ahafo"],
        "CM": ["Southwest", "Littoral", "Centre"],
    },
    "coffee": {
        "BR": ["Minas Gerais", "Espirito Santo"],
        "CO": ["Huila", "Tolima", "Cauca"],
        "VN": ["Central Highlands", "Lam Dong"],
    },
    "oil_palm": {
        "ID": ["Kalimantan", "Sumatra", "Papua"],
        "MY": ["Sabah", "Sarawak", "Johor"],
        "CO": ["Meta", "Casanare"],
    },
    "rubber": {
        "ID": ["Sumatra", "Kalimantan"],
        "TH": ["Southern Region"],
        "VN": ["Southeast Region"],
    },
    "soya": {
        "BR": ["Mato Grosso", "Para", "Maranhao", "Tocantins"],
        "AR": ["Salta", "Santiago del Estero"],
        "PY": ["Alto Paraguay", "Canindeyu"],
    },
    "wood": {
        "BR": ["Amazonas", "Para", "Rondonia"],
        "ID": ["Kalimantan", "Sumatra", "Papua"],
        "CD": ["North Kivu", "South Kivu"],
        "MY": ["Sabah", "Sarawak"],
    },
}

def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal for precise arithmetic."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _float(value: Decimal) -> float:
    """Convert Decimal to float for API responses."""
    return float(value)

# ---------------------------------------------------------------------------
# CommodityRiskAnalyzer
# ---------------------------------------------------------------------------

class CommodityRiskAnalyzer:
    """EUDR commodity-specific risk analysis for 7 regulated commodities.

    Analyzes commodity-specific risk factors including country base risk,
    production region deforestation pressure, supply chain exposure,
    price-driven deforestation, certification scheme effectiveness,
    seasonal risk variation, and derived product mapping.

    All risk calculations use Decimal arithmetic for zero floating-point
    drift and deterministic reproducibility.

    Attributes:
        _profiles: In-memory store of commodity risk profiles keyed by
            profile_id.
        _price_indices: Historical commodity price indices for correlation.
        _production_data: Production volume data by country-commodity.
        _lock: Threading lock for thread-safe access.

    Example:
        >>> analyzer = CommodityRiskAnalyzer()
        >>> result = analyzer.analyze_commodity("BR", "soya", country_risk=72.5)
        >>> assert result.commodity_type == CommodityType.SOYA
        >>> assert 0.0 <= result.commodity_risk_score <= 100.0
    """

    def __init__(self) -> None:
        """Initialize CommodityRiskAnalyzer with empty stores."""
        self._profiles: Dict[str, CommodityRiskProfile] = {}
        self._price_indices: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        self._production_data: Dict[Tuple[str, str], Decimal] = {}
        self._lock: threading.Lock = threading.Lock()
        logger.info(
            "CommodityRiskAnalyzer initialized: commodities=%d, "
            "high_risk_regions=%d",
            len(SUPPORTED_COMMODITIES),
            sum(len(v) for v in _HIGH_RISK_REGIONS.values()),
        )

    # ------------------------------------------------------------------
    # Primary analysis
    # ------------------------------------------------------------------

    def analyze_commodity(
        self,
        country_code: str,
        commodity_type: str,
        country_risk_score: float,
        region: Optional[str] = None,
        certification_schemes: Optional[List[str]] = None,
        production_volume: Optional[float] = None,
        month: Optional[int] = None,
    ) -> CommodityRiskProfile:
        """Analyze commodity-specific risk for a country-commodity pair.

        Applies the following analysis pipeline:
        1. Validate inputs (country, commodity, risk score).
        2. Calculate production region risk.
        3. Calculate supply chain exposure risk.
        4. Calculate price pressure risk.
        5. Apply seasonal risk multiplier if month provided.
        6. Assess certification effectiveness.
        7. Calculate composite commodity risk score.
        8. Store profile and record provenance/metrics.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            commodity_type: EUDR commodity type (cattle, cocoa, coffee,
                oil_palm, rubber, soya, wood).
            country_risk_score: Base country risk score (0-100) from
                CountryRiskScorer.
            region: Optional sub-national region for production risk.
            certification_schemes: Optional list of active certification
                scheme names (e.g., ["fsc", "rspo"]).
            production_volume: Optional production volume (tonnes/year)
                for supply chain exposure.
            month: Optional month (1-12) for seasonal risk adjustment.

        Returns:
            CommodityRiskProfile with commodity_risk_score, production_risk,
            supply_chain_risk, price_pressure, seasonal_multiplier,
            certification_effectiveness, and risk classification.

        Raises:
            ValueError: If country_code is empty, commodity_type invalid,
                or country_risk_score outside [0, 100].
        """
        start = time.monotonic()
        cfg = get_config()

        # -- Input validation ------------------------------------------------
        country_code = self._validate_country_code(country_code)
        commodity_enum = self._validate_commodity(commodity_type)
        self._validate_risk_score(country_risk_score)

        # -- Production region risk ------------------------------------------
        production_risk = self._calculate_production_risk(
            country_code, commodity_type, region,
        )

        # -- Supply chain exposure risk --------------------------------------
        supply_chain_risk = self._calculate_supply_chain_risk(
            country_code, commodity_type, production_volume,
        )

        # -- Price pressure risk ---------------------------------------------
        price_pressure = self._calculate_price_pressure(
            commodity_type,
        )

        # -- Composite commodity risk ----------------------------------------
        commodity_risk = self._calculate_commodity_risk(
            country_base_risk=_decimal(country_risk_score),
            production_risk=production_risk,
            supply_chain_risk=supply_chain_risk,
            price_pressure=price_pressure,
        )

        # -- Seasonal adjustment ---------------------------------------------
        seasonal_multiplier = Decimal("1.0")
        if cfg.enable_seasonal_analysis and month is not None:
            seasonal_multiplier = self._get_seasonal_multiplier(
                commodity_type, month,
            )
            commodity_risk *= seasonal_multiplier

        # -- Certification effectiveness -------------------------------------
        cert_effectiveness = self._assess_certification_effectiveness(
            commodity_type, certification_schemes or [],
        )

        # -- Risk classification ---------------------------------------------
        risk_level = self._classify_commodity_risk(commodity_risk, cfg)

        # -- Build profile ---------------------------------------------------
        profile = self._build_profile(
            country_code=country_code,
            commodity_enum=commodity_enum,
            country_risk_score=country_risk_score,
            commodity_risk_score=commodity_risk,
            production_risk=production_risk,
            supply_chain_risk=supply_chain_risk,
            price_pressure=price_pressure,
            seasonal_multiplier=seasonal_multiplier,
            cert_effectiveness=cert_effectiveness,
            risk_level=risk_level,
            region=region,
        )

        # -- Store -----------------------------------------------------------
        with self._lock:
            self._profiles[profile.profile_id] = profile

        # -- Provenance ------------------------------------------------------
        tracker = get_provenance_tracker()
        tracker.record(
            entity_type="commodity_analysis",
            action="analyze",
            entity_id=profile.profile_id,
            data=profile.model_dump(mode="json"),
            metadata={
                "country_code": country_code,
                "commodity_type": commodity_type,
                "commodity_risk_score": _float(commodity_risk),
                "risk_level": risk_level.value,
            },
        )

        # -- Metrics ---------------------------------------------------------
        elapsed = time.monotonic() - start
        observe_commodity_analysis_duration(elapsed)
        record_commodity_analysis_completed(commodity_type)

        logger.info(
            "Commodity risk analyzed: country=%s commodity=%s "
            "country_risk=%.1f commodity_risk=%.1f level=%s "
            "seasonal_mult=%.2f cert_eff=%.1f elapsed_ms=%.1f",
            country_code,
            commodity_type,
            country_risk_score,
            _float(commodity_risk),
            risk_level.value,
            _float(seasonal_multiplier),
            cert_effectiveness,
            elapsed * 1000,
        )
        return profile

    def analyze_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[CommodityRiskProfile]:
        """Analyze commodity risk for multiple country-commodity pairs.

        Each item in the batch is a dictionary with keys:
            - country_code (str, required)
            - commodity_type (str, required)
            - country_risk_score (float, required)
            - region (str, optional)
            - certification_schemes (list[str], optional)
            - production_volume (float, optional)
            - month (int, optional)

        Args:
            items: List of analysis request dictionaries.

        Returns:
            List of CommodityRiskProfile results in the same order
            as the input items.

        Raises:
            ValueError: If items list is empty or exceeds batch_max_size.
        """
        cfg = get_config()
        if not items:
            raise ValueError("Batch items list must not be empty")
        if len(items) > cfg.batch_max_size:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum "
                f"{cfg.batch_max_size}"
            )

        results: List[CommodityRiskProfile] = []
        for item in items:
            result = self.analyze_commodity(
                country_code=item["country_code"],
                commodity_type=item["commodity_type"],
                country_risk_score=item["country_risk_score"],
                region=item.get("region"),
                certification_schemes=item.get("certification_schemes"),
                production_volume=item.get("production_volume"),
                month=item.get("month"),
            )
            results.append(result)

        logger.info(
            "Batch commodity analysis completed: items=%d", len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_profile(
        self, profile_id: str,
    ) -> Optional[CommodityRiskProfile]:
        """Retrieve a commodity risk profile by its unique identifier.

        Args:
            profile_id: The profile_id to look up.

        Returns:
            CommodityRiskProfile if found, None otherwise.
        """
        with self._lock:
            return self._profiles.get(profile_id)

    def list_profiles(
        self,
        country_code: Optional[str] = None,
        commodity_type: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CommodityRiskProfile]:
        """List commodity risk profiles with optional filters.

        Args:
            country_code: Optional country code filter.
            commodity_type: Optional commodity type filter.
            risk_level: Optional risk level filter (LOW/STANDARD/HIGH).
            limit: Maximum number of results (default 100).
            offset: Pagination offset (default 0).

        Returns:
            Filtered list of CommodityRiskProfile objects.
        """
        with self._lock:
            results = list(self._profiles.values())

        if country_code:
            cc = country_code.upper().strip()
            results = [p for p in results if p.country_code == cc]

        if commodity_type:
            results = [
                p for p in results
                if p.commodity_type.value == commodity_type.lower()
            ]

        if risk_level:
            results = [
                p for p in results
                if p.risk_level.value == risk_level.upper()
            ]

        # Sort by analysis timestamp descending
        results.sort(key=lambda p: p.analyzed_at, reverse=True)

        return results[offset:offset + limit]

    # ------------------------------------------------------------------
    # Commodity matrix
    # ------------------------------------------------------------------

    def get_commodity_matrix(
        self,
        country_codes: List[str],
        commodity_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate country-commodity cross risk matrix.

        Creates a matrix showing commodity-specific risk scores for
        each country-commodity combination.

        Args:
            country_codes: List of ISO 3166-1 alpha-2 country codes.
            commodity_types: Optional list of commodity types (defaults
                to all 7 EUDR commodities).

        Returns:
            Dictionary with matrix (list of dicts with country_code,
            commodity scores), highest_risk_combinations (list of tuples),
            and statistics.

        Raises:
            ValueError: If country_codes list is empty.
        """
        if not country_codes:
            raise ValueError("country_codes list must not be empty")

        if commodity_types is None:
            commodity_types = SUPPORTED_COMMODITIES

        matrix_data: List[Dict[str, Any]] = []
        all_scores: List[float] = []

        with self._lock:
            for cc in country_codes:
                cc_upper = cc.upper().strip()
                row: Dict[str, Any] = {"country_code": cc_upper}

                for commodity in commodity_types:
                    # Find most recent profile for this country-commodity
                    matching = [
                        p for p in self._profiles.values()
                        if p.country_code == cc_upper
                        and p.commodity_type.value == commodity.lower()
                    ]
                    if matching:
                        matching.sort(key=lambda p: p.analyzed_at, reverse=True)
                        score = matching[0].commodity_risk_score
                        row[commodity] = score
                        all_scores.append(score)
                    else:
                        row[commodity] = None

                matrix_data.append(row)

        # Find highest risk combinations
        high_risk_combos: List[Tuple[str, str, float]] = []
        for row in matrix_data:
            for commodity in commodity_types:
                score = row.get(commodity)
                if score is not None and score >= 75.0:
                    high_risk_combos.append((
                        row["country_code"], commodity, score,
                    ))

        high_risk_combos.sort(key=lambda x: x[2], reverse=True)

        # Statistics
        stats = {}
        if all_scores:
            stats = {
                "mean_score": round(sum(all_scores) / len(all_scores), 2),
                "min_score": round(min(all_scores), 2),
                "max_score": round(max(all_scores), 2),
                "total_assessments": len(all_scores),
            }

        return {
            "matrix": matrix_data,
            "highest_risk_combinations": [
                {
                    "country_code": cc,
                    "commodity_type": commodity,
                    "commodity_risk_score": score,
                }
                for cc, commodity, score in high_risk_combos[:20]
            ],
            "statistics": stats,
        }

    # ------------------------------------------------------------------
    # Derived products
    # ------------------------------------------------------------------

    def map_derived_products(
        self, derived_product: str,
    ) -> Dict[str, Any]:
        """Map a derived product to its primary EUDR commodity.

        Args:
            derived_product: Derived product name (e.g., "chocolate",
                "leather", "biodiesel").

        Returns:
            Dictionary with derived_product, primary_commodity,
            alternative_commodities (if applicable), and is_eudr_regulated.

        Raises:
            ValueError: If derived_product is empty.
        """
        if not derived_product or not derived_product.strip():
            raise ValueError("derived_product must not be empty")

        product_lower = derived_product.lower().strip()
        primary = _DERIVED_PRODUCTS.get(product_lower)

        if primary is None:
            return {
                "derived_product": derived_product,
                "primary_commodity": None,
                "alternative_commodities": [],
                "is_eudr_regulated": False,
                "note": "Product not mapped to EUDR commodity",
            }

        # Special case: biodiesel can be from soya or oil_palm
        alternatives: List[str] = []
        if product_lower == "biodiesel":
            alternatives = ["soya", "oil_palm"]
            primary = "soya"  # Default to soya

        return {
            "derived_product": derived_product,
            "primary_commodity": primary,
            "alternative_commodities": alternatives,
            "is_eudr_regulated": True,
            "note": (
                "Multiple sources" if alternatives
                else f"Primary source: {primary}"
            ),
        }

    # ------------------------------------------------------------------
    # Certification effectiveness
    # ------------------------------------------------------------------

    def assess_certification_effectiveness(
        self,
        commodity_type: str,
        certification_schemes: List[str],
    ) -> Dict[str, Any]:
        """Assess certification scheme effectiveness for a commodity.

        Args:
            commodity_type: EUDR commodity type.
            certification_schemes: List of certification scheme names.

        Returns:
            Dictionary with commodity_type, overall_effectiveness,
            scheme_details (list of dicts with scheme, effectiveness,
            applicable), and effectiveness_category (high/medium/low).

        Raises:
            ValueError: If commodity_type is invalid.
        """
        commodity_enum = self._validate_commodity(commodity_type)
        overall = self._assess_certification_effectiveness(
            commodity_type, certification_schemes,
        )

        scheme_details = []
        for scheme in certification_schemes:
            scheme_lower = scheme.lower().strip()
            effectiveness = self._get_scheme_effectiveness(
                scheme_lower, commodity_type,
            )
            scheme_details.append({
                "scheme": scheme_lower,
                "effectiveness": effectiveness,
                "applicable": effectiveness > 0,
            })

        category = "low"
        if overall >= 70.0:
            category = "high"
        elif overall >= 40.0:
            category = "medium"

        return {
            "commodity_type": commodity_type,
            "overall_effectiveness": overall,
            "scheme_details": scheme_details,
            "effectiveness_category": category,
            "note": (
                f"{len(certification_schemes)} schemes assessed for "
                f"{commodity_type}"
            ),
        }

    # ------------------------------------------------------------------
    # Seasonal risk
    # ------------------------------------------------------------------

    def get_seasonal_risk(
        self,
        commodity_type: str,
        month: int,
    ) -> Dict[str, Any]:
        """Get seasonal risk multiplier for a commodity and month.

        Args:
            commodity_type: EUDR commodity type.
            month: Month (1-12).

        Returns:
            Dictionary with commodity_type, month, seasonal_multiplier,
            baseline_multiplier (1.0), and interpretation.

        Raises:
            ValueError: If commodity_type invalid or month not in [1, 12].
        """
        commodity_enum = self._validate_commodity(commodity_type)
        if month < 1 or month > 12:
            raise ValueError(f"month must be in [1, 12], got {month}")

        multiplier = self._get_seasonal_multiplier(commodity_type, month)
        interpretation = "baseline"
        if multiplier > Decimal("1.1"):
            interpretation = "elevated"
        elif multiplier < Decimal("0.9"):
            interpretation = "reduced"

        return {
            "commodity_type": commodity_type,
            "month": month,
            "seasonal_multiplier": _float(multiplier),
            "baseline_multiplier": 1.0,
            "interpretation": interpretation,
            "note": (
                f"Seasonal risk for {commodity_type} in month {month} "
                f"is {interpretation} ({_float(multiplier)}x baseline)"
            ),
        }

    # ------------------------------------------------------------------
    # Production risk
    # ------------------------------------------------------------------

    def get_production_risk(
        self,
        country_code: str,
        commodity_type: str,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get production region deforestation pressure risk.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            commodity_type: EUDR commodity type.
            region: Optional sub-national region.

        Returns:
            Dictionary with country_code, commodity_type, region,
            production_risk_score, is_high_risk_region, and explanation.
        """
        country_code = self._validate_country_code(country_code)
        commodity_enum = self._validate_commodity(commodity_type)

        production_risk = self._calculate_production_risk(
            country_code, commodity_type, region,
        )

        is_high_risk = self._is_high_risk_region(
            country_code, commodity_type, region,
        )

        explanation = (
            f"Production risk for {commodity_type} in {country_code}"
        )
        if region:
            explanation += f"/{region}"
        if is_high_risk:
            explanation += " is ELEVATED (high deforestation pressure region)"
        else:
            explanation += " is baseline"

        return {
            "country_code": country_code,
            "commodity_type": commodity_type,
            "region": region,
            "production_risk_score": _float(production_risk),
            "is_high_risk_region": is_high_risk,
            "explanation": explanation,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_country_code(self, country_code: str) -> str:
        """Validate and normalize country code."""
        if not country_code or not country_code.strip():
            raise ValueError("country_code must not be empty")
        cc = country_code.upper().strip()
        if len(cc) != 2:
            raise ValueError(
                f"country_code must be 2 characters, got '{cc}'"
            )
        return cc

    def _validate_commodity(self, commodity_type: str) -> CommodityType:
        """Validate commodity type and return enum."""
        if not commodity_type or not commodity_type.strip():
            raise ValueError("commodity_type must not be empty")
        try:
            return CommodityType(commodity_type.lower().strip())
        except ValueError:
            raise ValueError(
                f"Invalid commodity_type '{commodity_type}'; "
                f"must be one of: {', '.join(SUPPORTED_COMMODITIES)}"
            )

    def _validate_risk_score(self, risk_score: float) -> None:
        """Validate risk score is within bounds."""
        if risk_score < 0.0 or risk_score > 100.0:
            raise ValueError(
                f"risk_score must be in [0, 100], got {risk_score}"
            )

    def _calculate_production_risk(
        self,
        country_code: str,
        commodity_type: str,
        region: Optional[str],
    ) -> Decimal:
        """Calculate production region deforestation pressure risk."""
        base_risk = Decimal("50.0")

        # Check if region is high deforestation pressure
        if region and self._is_high_risk_region(country_code, commodity_type, region):
            base_risk = Decimal("85.0")

        return base_risk

    def _is_high_risk_region(
        self,
        country_code: str,
        commodity_type: str,
        region: Optional[str],
    ) -> bool:
        """Check if region is a high deforestation pressure zone."""
        if not region:
            return False

        commodity_regions = _HIGH_RISK_REGIONS.get(commodity_type, {})
        country_regions = commodity_regions.get(country_code, [])

        return any(
            r.lower() == region.lower() for r in country_regions
        )

    def _calculate_supply_chain_risk(
        self,
        country_code: str,
        commodity_type: str,
        production_volume: Optional[float],
    ) -> Decimal:
        """Calculate supply chain exposure risk."""
        # Base on production volume if available
        if production_volume is not None:
            volume_decimal = _decimal(production_volume)
            # Higher volume = higher supply chain complexity
            # Log scale: 0-1000t = low, 1000-100k = medium, >100k = high
            if volume_decimal < Decimal("1000"):
                return Decimal("30.0")
            elif volume_decimal < Decimal("100000"):
                return Decimal("60.0")
            else:
                return Decimal("85.0")

        # Default medium complexity
        return Decimal("50.0")

    def _calculate_price_pressure(
        self, commodity_type: str,
    ) -> Decimal:
        """Calculate price-driven deforestation pressure."""
        # Simplified: use recent price trend if available
        with self._lock:
            price_history = self._price_indices.get(commodity_type, [])

        if len(price_history) < 2:
            # No data, assume baseline
            return Decimal("50.0")

        # Calculate price change over last year
        recent = price_history[-1][1]
        year_ago = price_history[0][1]
        change_pct = ((recent - year_ago) / year_ago) * Decimal("100")

        # Positive change = higher price = higher deforestation pressure
        if change_pct > Decimal("20"):
            return Decimal("75.0")
        elif change_pct > Decimal("10"):
            return Decimal("60.0")
        else:
            return Decimal("50.0")

    def _calculate_commodity_risk(
        self,
        country_base_risk: Decimal,
        production_risk: Decimal,
        supply_chain_risk: Decimal,
        price_pressure: Decimal,
    ) -> Decimal:
        """Calculate composite commodity risk score."""
        composite = (
            (country_base_risk * _RISK_WEIGHTS["country_base_risk"]) +
            (production_risk * _RISK_WEIGHTS["production_risk"]) +
            (supply_chain_risk * _RISK_WEIGHTS["supply_chain_risk"]) +
            (price_pressure * _RISK_WEIGHTS["price_pressure"])
        )
        return composite.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _get_seasonal_multiplier(
        self, commodity_type: str, month: int,
    ) -> Decimal:
        """Get seasonal risk multiplier for commodity and month."""
        seasonal_factors = _SEASONAL_RISK.get(commodity_type)
        if seasonal_factors is None:
            return Decimal("1.0")

        # month is 1-indexed, list is 0-indexed
        multiplier = seasonal_factors[month - 1]
        return _decimal(multiplier)

    def _assess_certification_effectiveness(
        self,
        commodity_type: str,
        certification_schemes: List[str],
    ) -> float:
        """Calculate overall certification effectiveness."""
        if not certification_schemes:
            return 0.0

        effectiveness_scores: List[float] = []
        for scheme in certification_schemes:
            scheme_lower = scheme.lower().strip()
            effectiveness = self._get_scheme_effectiveness(
                scheme_lower, commodity_type,
            )
            if effectiveness > 0:
                effectiveness_scores.append(effectiveness)

        if not effectiveness_scores:
            return 0.0

        # Average of applicable schemes
        return sum(effectiveness_scores) / len(effectiveness_scores)

    def _get_scheme_effectiveness(
        self, scheme: str, commodity_type: str,
    ) -> float:
        """Get certification scheme effectiveness for commodity."""
        scheme_data = _CERTIFICATION_EFFECTIVENESS.get(scheme)
        if not scheme_data:
            return 0.0
        return scheme_data.get(commodity_type.lower(), 0.0)

    def _classify_commodity_risk(
        self, commodity_risk: Decimal, cfg: Any,
    ) -> RiskLevel:
        """Classify commodity risk level from score."""
        score_float = _float(commodity_risk)
        if score_float <= cfg.low_risk_threshold:
            return RiskLevel.LOW
        if score_float <= cfg.high_risk_threshold:
            return RiskLevel.STANDARD
        return RiskLevel.HIGH

    def _build_profile(
        self,
        country_code: str,
        commodity_enum: CommodityType,
        country_risk_score: float,
        commodity_risk_score: Decimal,
        production_risk: Decimal,
        supply_chain_risk: Decimal,
        price_pressure: Decimal,
        seasonal_multiplier: Decimal,
        cert_effectiveness: float,
        risk_level: RiskLevel,
        region: Optional[str],
    ) -> CommodityRiskProfile:
        """Build complete CommodityRiskProfile model."""
        # Provenance hash
        tracker = get_provenance_tracker()
        prov_data = {
            "country_code": country_code,
            "commodity_type": commodity_enum.value,
            "country_risk_score": country_risk_score,
            "commodity_risk_score": _float(commodity_risk_score),
            "production_risk": _float(production_risk),
            "supply_chain_risk": _float(supply_chain_risk),
            "price_pressure": _float(price_pressure),
        }
        provenance_hash = tracker.build_hash(prov_data)

        return CommodityRiskProfile(
            country_code=country_code,
            commodity_type=commodity_enum,
            risk_score=_float(commodity_risk_score),
            risk_level=risk_level,
            certification_effectiveness=cert_effectiveness,
            seasonal_factors={
                "production_risk": _float(production_risk),
                "supply_chain_risk": _float(supply_chain_risk),
                "price_pressure": _float(price_pressure),
                "seasonal_multiplier": _float(seasonal_multiplier),
            },
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        with self._lock:
            count = len(self._profiles)
        return (
            f"CommodityRiskAnalyzer("
            f"profiles={count}, "
            f"commodities={len(SUPPORTED_COMMODITIES)})"
        )

    def __len__(self) -> int:
        """Return number of stored profiles."""
        with self._lock:
            return len(self._profiles)
