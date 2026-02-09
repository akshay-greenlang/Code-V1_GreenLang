# -*- coding: utf-8 -*-
"""
Scope 3 Mapper Engine - AGENT-DATA-009: Spend Data Categorizer
================================================================

Maps spend records to GHG Protocol Scope 3 categories using
deterministic rule-based classification. Supports mapping from NAICS,
UNSPSC, keyword-based, and generic taxonomy codes with confidence
scoring. Handles CapEx/OpEx detection and multi-category allocation.

Supports:
    - Single-record Scope 3 mapping with confidence
    - Batch Scope 3 mapping
    - NAICS-to-Scope-3 mapping (50+ rules)
    - UNSPSC-to-Scope-3 mapping (58 segments)
    - Keyword-to-Scope-3 mapping (60+ keywords)
    - Generic taxonomy-to-Scope-3 mapping
    - Capital vs operating expense detection
    - Multi-category allocation splitting
    - Category metadata retrieval (all 15 categories + unclassified)
    - Thread-safe in-memory storage
    - SHA-256 provenance hashes on all mappings

Zero-Hallucination Guarantees:
    - All classification is rule-based (explicit lookup tables)
    - No LLM or ML model in Scope 3 classification path
    - Confidence scores are deterministic
    - SHA-256 provenance hashes for audit trails

Example:
    >>> from greenlang.spend_categorizer.scope3_mapper import Scope3MapperEngine
    >>> engine = Scope3MapperEngine()
    >>> result = engine.map_record({"description": "air travel", "amount_usd": 5000})
    >>> print(result.category_number, result.category_name, result.confidence)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-009 Spend Data Categorizer (GL-DATA-SUP-002)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "Scope3Category",
    "Scope3Assignment",
    "Scope3MapperEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "s3m") -> str:
    """Generate a unique identifier with a prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Scope 3 Category definitions
# ---------------------------------------------------------------------------

_SCOPE3_CATEGORIES: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Purchased Goods and Services",
        "direction": "upstream",
        "description": "Extraction, production, and transportation of goods and services purchased or acquired by the reporting company",
        "methodology": "spend-based, average-data, supplier-specific, hybrid",
    },
    2: {
        "name": "Capital Goods",
        "direction": "upstream",
        "description": "Extraction, production, and transportation of capital goods purchased or acquired by the reporting company",
        "methodology": "spend-based, average-data, supplier-specific",
    },
    3: {
        "name": "Fuel- and Energy-Related Activities",
        "direction": "upstream",
        "description": "Extraction, production, and transportation of fuels and energy purchased, not included in Scope 1 or 2",
        "methodology": "average-data, supplier-specific",
    },
    4: {
        "name": "Upstream Transportation and Distribution",
        "direction": "upstream",
        "description": "Transportation and distribution of products purchased in the reporting year between tier 1 suppliers and own operations",
        "methodology": "spend-based, distance-based, fuel-based",
    },
    5: {
        "name": "Waste Generated in Operations",
        "direction": "upstream",
        "description": "Disposal and treatment of waste generated in the reporting company's operations",
        "methodology": "waste-type-specific, average-data",
    },
    6: {
        "name": "Business Travel",
        "direction": "upstream",
        "description": "Transportation of employees for business-related activities",
        "methodology": "spend-based, distance-based, fuel-based",
    },
    7: {
        "name": "Employee Commuting",
        "direction": "upstream",
        "description": "Transportation of employees between their homes and worksites",
        "methodology": "average-data, distance-based",
    },
    8: {
        "name": "Upstream Leased Assets",
        "direction": "upstream",
        "description": "Operation of assets leased by the reporting company not included in Scope 1 and 2",
        "methodology": "asset-specific, average-data",
    },
    9: {
        "name": "Downstream Transportation and Distribution",
        "direction": "downstream",
        "description": "Transportation and distribution of products sold by the reporting company after the point of sale",
        "methodology": "spend-based, distance-based, fuel-based",
    },
    10: {
        "name": "Processing of Sold Products",
        "direction": "downstream",
        "description": "Processing of intermediate products sold by downstream companies",
        "methodology": "average-data, site-specific",
    },
    11: {
        "name": "Use of Sold Products",
        "direction": "downstream",
        "description": "End use of goods and services sold by the reporting company",
        "methodology": "product-specific, average-data",
    },
    12: {
        "name": "End-of-Life Treatment of Sold Products",
        "direction": "downstream",
        "description": "Waste disposal and treatment of products sold by the reporting company at the end of their life",
        "methodology": "waste-type-specific, average-data",
    },
    13: {
        "name": "Downstream Leased Assets",
        "direction": "downstream",
        "description": "Operation of assets owned by the reporting company and leased to other entities",
        "methodology": "asset-specific, average-data",
    },
    14: {
        "name": "Franchises",
        "direction": "downstream",
        "description": "Operation of franchises not included in Scope 1 and 2",
        "methodology": "franchise-specific, average-data",
    },
    15: {
        "name": "Investments",
        "direction": "downstream",
        "description": "Operation of investments not included in Scope 1 and 2",
        "methodology": "investment-specific, average-data",
    },
}


# ---------------------------------------------------------------------------
# NAICS 2-digit to Scope 3 mapping (50+ entries)
# ---------------------------------------------------------------------------

_NAICS_TO_SCOPE3: Dict[str, Tuple[int, float]] = {
    # Agriculture, Forestry
    "11": (1, 0.85),
    "111": (1, 0.88),
    "112": (1, 0.88),
    "113": (1, 0.85),
    "114": (1, 0.85),
    "115": (1, 0.85),
    # Mining
    "21": (1, 0.80),
    "211": (3, 0.85),
    "212": (1, 0.82),
    "213": (1, 0.80),
    # Utilities
    "22": (3, 0.90),
    "221": (3, 0.92),
    # Construction
    "23": (2, 0.85),
    "236": (2, 0.87),
    "237": (2, 0.87),
    "238": (1, 0.80),
    # Manufacturing - Food/Textile/Apparel
    "31": (1, 0.90),
    "311": (1, 0.92),
    "312": (1, 0.88),
    "313": (1, 0.88),
    "314": (1, 0.85),
    "315": (1, 0.85),
    "316": (1, 0.85),
    # Manufacturing - Wood/Paper/Petroleum/Chemical
    "32": (1, 0.88),
    "321": (1, 0.88),
    "322": (1, 0.88),
    "324": (3, 0.90),
    "325": (1, 0.85),
    "326": (1, 0.85),
    "327": (1, 0.85),
    # Manufacturing - Metal/Machinery/Computer
    "33": (1, 0.88),
    "331": (1, 0.88),
    "332": (1, 0.88),
    "333": (2, 0.82),
    "334": (1, 0.85),
    "335": (1, 0.85),
    "336": (2, 0.80),
    "337": (1, 0.85),
    "339": (1, 0.82),
    # Wholesale Trade
    "42": (1, 0.75),
    # Retail Trade
    "44": (1, 0.70),
    "45": (1, 0.70),
    # Transportation
    "48": (4, 0.90),
    "481": (6, 0.88),
    "482": (4, 0.88),
    "483": (4, 0.88),
    "484": (4, 0.92),
    "485": (7, 0.80),
    "486": (4, 0.85),
    "487": (6, 0.80),
    "488": (4, 0.82),
    # Warehousing / Postal
    "49": (4, 0.82),
    "491": (4, 0.80),
    "492": (4, 0.85),
    "493": (4, 0.82),
    # Information
    "51": (1, 0.75),
    "511": (1, 0.80),
    "517": (1, 0.78),
    "518": (1, 0.80),
    "519": (1, 0.75),
    # Finance and Insurance
    "52": (15, 0.80),
    "521": (15, 0.82),
    "522": (15, 0.82),
    "523": (15, 0.85),
    "524": (15, 0.80),
    # Real Estate
    "53": (8, 0.78),
    "531": (8, 0.82),
    "532": (8, 0.80),
    "533": (8, 0.75),
    # Professional Services
    "54": (1, 0.80),
    "5411": (1, 0.82),
    "5412": (1, 0.82),
    "5413": (1, 0.82),
    "5414": (1, 0.80),
    "5415": (1, 0.85),
    "5416": (1, 0.80),
    "5417": (1, 0.80),
    "5418": (1, 0.78),
    "5419": (1, 0.78),
    # Management
    "55": (1, 0.70),
    # Administrative / Waste
    "56": (1, 0.75),
    "561": (1, 0.78),
    "562": (5, 0.90),
    # Education
    "61": (1, 0.72),
    # Healthcare
    "62": (1, 0.75),
    # Arts / Entertainment
    "71": (6, 0.70),
    # Accommodation / Food
    "72": (6, 0.82),
    "721": (6, 0.85),
    "722": (6, 0.80),
    # Other Services
    "81": (1, 0.70),
    "811": (1, 0.75),
    "812": (1, 0.70),
    "813": (1, 0.65),
    # Public Administration
    "92": (1, 0.60),
}


# ---------------------------------------------------------------------------
# UNSPSC segment to Scope 3 mapping (58 segments)
# ---------------------------------------------------------------------------

_UNSPSC_TO_SCOPE3: Dict[str, Tuple[int, float]] = {
    "10": (1, 0.85),   # Live Plant and Animal Material
    "11": (1, 0.82),   # Mineral and Textile Materials
    "12": (1, 0.85),   # Chemicals
    "13": (1, 0.85),   # Resin and Rubber
    "14": (1, 0.85),   # Paper
    "15": (3, 0.90),   # Fuels
    "20": (2, 0.78),   # Mining Machinery
    "21": (2, 0.78),   # Farming Machinery
    "22": (2, 0.80),   # Construction Machinery
    "23": (2, 0.82),   # Industrial Machinery
    "24": (2, 0.78),   # Material Handling
    "25": (2, 0.82),   # Vehicles
    "26": (2, 0.85),   # Power Generation
    "27": (1, 0.80),   # Tools
    "30": (1, 0.82),   # Building Components
    "31": (1, 0.85),   # Manufacturing Components
    "32": (1, 0.85),   # Electronic Components
    "39": (1, 0.80),   # Lighting
    "40": (1, 0.82),   # Distribution Systems
    "41": (2, 0.78),   # Laboratory Equipment
    "42": (1, 0.80),   # Medical Equipment
    "43": (1, 0.85),   # IT and Telecom
    "44": (1, 0.85),   # Office Supplies
    "45": (1, 0.80),   # Printing Equipment
    "46": (1, 0.78),   # Defense and Security
    "47": (1, 0.82),   # Cleaning Supplies
    "48": (1, 0.78),   # Service Industry Equipment
    "49": (1, 0.75),   # Sports Equipment
    "50": (1, 0.85),   # Food and Beverage
    "51": (1, 0.85),   # Pharmaceuticals
    "52": (1, 0.80),   # Consumer Electronics
    "53": (1, 0.80),   # Apparel
    "54": (1, 0.75),   # Jewelry
    "55": (1, 0.78),   # Publications
    "56": (1, 0.82),   # Furniture
    "60": (1, 0.72),   # Education Materials
    "70": (1, 0.78),   # Farming Services
    "71": (1, 0.78),   # Mining Services
    "72": (1, 0.82),   # Building Maintenance
    "73": (1, 0.80),   # Manufacturing Services
    "76": (5, 0.82),   # Industrial Cleaning
    "77": (5, 0.85),   # Environmental Services
    "78": (4, 0.90),   # Transportation Services
    "80": (1, 0.82),   # Management Services
    "81": (1, 0.82),   # Engineering Services
    "82": (1, 0.78),   # Design Services
    "83": (3, 0.85),   # Utilities
    "84": (15, 0.78),  # Financial Services
    "85": (1, 0.78),   # Healthcare Services
    "86": (1, 0.75),   # Education Services
    "90": (6, 0.88),   # Travel Services
    "91": (1, 0.70),   # Personal Services
    "92": (1, 0.65),   # Defense
    "93": (1, 0.60),   # Politics
    "94": (1, 0.60),   # Organizations
    "95": (8, 0.78),   # Real Estate
    "A0": (1, 0.70),   # Alternative Medicine
    "A1": (1, 0.80),   # Stationery
    "A2": (1, 0.80),   # Consumer Electronics
}


# ---------------------------------------------------------------------------
# Keyword to Scope 3 mapping (60+ keywords)
# ---------------------------------------------------------------------------

_KEYWORD_TO_SCOPE3: Dict[str, Tuple[int, float]] = {
    # Category 1: Purchased Goods and Services
    "office supplies": (1, 0.88),
    "raw material": (1, 0.90),
    "components": (1, 0.85),
    "chemicals": (1, 0.85),
    "packaging": (1, 0.82),
    "paper": (1, 0.85),
    "software": (1, 0.82),
    "consulting": (1, 0.85),
    "professional services": (1, 0.85),
    "legal services": (1, 0.82),
    "accounting": (1, 0.82),
    "marketing": (1, 0.80),
    "advertising": (1, 0.80),
    "it services": (1, 0.85),
    "cloud computing": (1, 0.85),
    "cleaning": (1, 0.80),
    "maintenance": (1, 0.80),
    "food": (1, 0.82),
    "catering": (1, 0.80),
    "uniforms": (1, 0.78),
    "tools": (1, 0.80),
    "furniture": (1, 0.82),
    "laboratory": (1, 0.80),
    "medical supplies": (1, 0.82),
    "electronics": (1, 0.82),
    "stationery": (1, 0.82),
    "printing": (1, 0.80),
    # Category 2: Capital Goods
    "capital equipment": (2, 0.92),
    "machinery": (2, 0.90),
    "building construction": (2, 0.88),
    "renovation": (2, 0.85),
    "vehicle purchase": (2, 0.88),
    "server purchase": (2, 0.85),
    "heavy equipment": (2, 0.90),
    # Category 3: Fuel and Energy
    "fuel": (3, 0.92),
    "diesel": (3, 0.92),
    "gasoline": (3, 0.92),
    "natural gas": (3, 0.90),
    "electricity": (3, 0.90),
    "energy": (3, 0.88),
    "petroleum": (3, 0.90),
    "heating oil": (3, 0.88),
    # Category 4: Upstream Transportation
    "freight": (4, 0.92),
    "shipping": (4, 0.90),
    "logistics": (4, 0.88),
    "trucking": (4, 0.90),
    "courier": (4, 0.85),
    "delivery": (4, 0.82),
    "warehouse": (4, 0.80),
    "distribution": (4, 0.82),
    # Category 5: Waste
    "waste disposal": (5, 0.92),
    "recycling": (5, 0.88),
    "hazardous waste": (5, 0.92),
    "landfill": (5, 0.90),
    "waste management": (5, 0.90),
    "sewage": (5, 0.85),
    # Category 6: Business Travel
    "air travel": (6, 0.95),
    "flight": (6, 0.92),
    "hotel": (6, 0.88),
    "car rental": (6, 0.88),
    "business travel": (6, 0.95),
    "taxi": (6, 0.85),
    "train travel": (6, 0.85),
    "lodging": (6, 0.85),
    "conference": (6, 0.78),
    # Category 7: Employee Commuting
    "commuting": (7, 0.92),
    "employee transport": (7, 0.88),
    "shuttle": (7, 0.85),
    "parking": (7, 0.75),
    # Category 8: Upstream Leased Assets
    "lease": (8, 0.82),
    "rent": (8, 0.80),
    "leased equipment": (8, 0.88),
    "leased vehicle": (8, 0.88),
    "leased office": (8, 0.85),
    # Category 15: Investments
    "investment": (15, 0.85),
    "equity": (15, 0.82),
    "finance": (15, 0.75),
    "banking": (15, 0.78),
    "insurance": (15, 0.75),
}


# ---------------------------------------------------------------------------
# CapEx detection keywords
# ---------------------------------------------------------------------------

_CAPEX_KEYWORDS: List[str] = [
    "capital", "capex", "asset", "machinery", "equipment",
    "building", "construction", "renovation", "improvement",
    "vehicle purchase", "server purchase", "infrastructure",
    "plant", "heavy equipment", "fixed asset", "property",
    "long-term", "acquisition",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Scope3Category(BaseModel):
    """Scope 3 category metadata."""

    category_number: int = Field(..., ge=1, le=15, description="Category number (1-15)")
    name: str = Field(..., description="Category name")
    direction: str = Field(..., description="upstream or downstream")
    description: str = Field(default="", description="Category description")
    methodology: str = Field(default="", description="Applicable methodologies")

    model_config = {"extra": "forbid"}


class Scope3Assignment(BaseModel):
    """Scope 3 assignment result for a spend record."""

    assignment_id: str = Field(..., description="Unique assignment identifier")
    record_id: str = Field(default="", description="Source record identifier")
    category_number: int = Field(..., ge=0, le=15, description="Scope 3 category (0=unclassified)")
    category_name: str = Field(default="", description="Category name")
    direction: str = Field(default="upstream", description="upstream or downstream")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Assignment confidence")
    match_source: str = Field(default="", description="Source of match (naics, unspsc, keyword, taxonomy)")
    is_capex: bool = Field(default=False, description="Whether detected as capital expenditure")
    allocations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Multi-category allocation splits (if applicable)",
    )
    amount_usd: float = Field(default=0.0, description="Spend amount in USD")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    assigned_at: str = Field(default="", description="Assignment timestamp ISO")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Scope3MapperEngine
# ---------------------------------------------------------------------------


class Scope3MapperEngine:
    """GHG Protocol Scope 3 category mapper.

    Maps spend records to Scope 3 categories using a multi-tier
    classification approach:
    1. NAICS code mapping (if available)
    2. UNSPSC code mapping (if available)
    3. Keyword-based mapping from description
    4. Taxonomy-based mapping (generic)

    All classification is deterministic and rule-based.

    Attributes:
        _config: Configuration dictionary.
        _assignments: In-memory assignment storage.
        _lock: Threading lock for thread-safe mutations.
        _stats: Cumulative mapping statistics.

    Example:
        >>> engine = Scope3MapperEngine()
        >>> result = engine.map_record({"description": "diesel fuel", "amount_usd": 10000})
        >>> assert result.category_number == 3
        >>> assert result.confidence > 0.8
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Scope3MapperEngine.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``min_confidence``: float (default 0.3)
                - ``default_category``: int (default 1)
                - ``enable_capex_detection``: bool (default True)
        """
        self._config = config or {}
        self._min_confidence: float = self._config.get("min_confidence", 0.3)
        self._default_category: int = self._config.get("default_category", 1)
        self._enable_capex: bool = self._config.get("enable_capex_detection", True)
        self._assignments: Dict[str, Scope3Assignment] = {}
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "records_mapped": 0,
            "by_category": {},
            "by_source": {},
            "capex_detected": 0,
            "unclassified": 0,
            "total_confidence": 0.0,
            "avg_confidence": 0.0,
            "errors": 0,
        }
        logger.info(
            "Scope3MapperEngine initialised: min_confidence=%.2f, "
            "default_category=%d, capex_detection=%s",
            self._min_confidence,
            self._default_category,
            self._enable_capex,
        )

    # ------------------------------------------------------------------
    # Public API - Mapping
    # ------------------------------------------------------------------

    def map_record(
        self,
        record: Dict[str, Any],
        taxonomy_code: Optional[str] = None,
    ) -> Scope3Assignment:
        """Map a single spend record to a Scope 3 category.

        Attempts classification in priority order:
        1. NAICS code (if naics_code or taxonomy_code provided)
        2. UNSPSC code (if unspsc_code or taxonomy_code provided)
        3. Keyword-based (from description)
        4. Default category

        Args:
            record: Spend record dict with fields like ``description``,
                ``naics_code``, ``unspsc_code``, ``amount_usd``.
            taxonomy_code: Optional taxonomy code to use directly.

        Returns:
            Scope3Assignment with category and confidence.
        """
        start = time.monotonic()

        record_id = str(record.get("record_id", ""))
        description = str(record.get("description", "")).lower().strip()
        amount_usd = float(record.get("amount_usd", 0) or 0)
        naics_code = str(record.get("naics_code", taxonomy_code or "")).strip()
        unspsc_code = str(record.get("unspsc_code", "")).strip()

        category_num = 0
        confidence = 0.0
        match_source = ""

        # CapEx detection
        is_capex = False
        if self._enable_capex:
            is_capex = self.detect_capex(record)
            if is_capex:
                category_num = 2
                confidence = 0.85
                match_source = "capex_detection"

        # Tier 1: NAICS mapping
        if not match_source and naics_code:
            cat, conf = self.map_from_naics(naics_code)
            if conf >= self._min_confidence:
                category_num = cat
                confidence = conf
                match_source = "naics"

        # Tier 2: UNSPSC mapping
        if not match_source and unspsc_code:
            cat, conf = self.map_from_unspsc(unspsc_code)
            if conf >= self._min_confidence:
                category_num = cat
                confidence = conf
                match_source = "unspsc"

        # Tier 3: Keyword mapping
        if not match_source and description:
            cat, conf = self.map_from_keyword(description)
            if conf >= self._min_confidence:
                category_num = cat
                confidence = conf
                match_source = "keyword"

        # Tier 4: Default
        if not match_source:
            category_num = self._default_category
            confidence = 0.30
            match_source = "default"

        # Get category metadata
        cat_info = _SCOPE3_CATEGORIES.get(category_num, _SCOPE3_CATEGORIES[1])
        cat_name = cat_info["name"]
        direction = cat_info["direction"]

        # Build assignment
        aid = _generate_id("s3m")
        now_iso = _utcnow().isoformat()

        provenance_hash = self._compute_provenance(
            aid, category_num, confidence, match_source, now_iso,
        )

        assignment = Scope3Assignment(
            assignment_id=aid,
            record_id=record_id,
            category_number=category_num,
            category_name=cat_name,
            direction=direction,
            confidence=round(confidence, 4),
            match_source=match_source,
            is_capex=is_capex,
            amount_usd=amount_usd,
            provenance_hash=provenance_hash,
            assigned_at=now_iso,
        )

        # Update stats
        with self._lock:
            self._assignments[aid] = assignment
            self._stats["records_mapped"] += 1
            cat_key = str(category_num)
            cat_counts = self._stats["by_category"]
            cat_counts[cat_key] = cat_counts.get(cat_key, 0) + 1
            src_counts = self._stats["by_source"]
            src_counts[match_source] = src_counts.get(match_source, 0) + 1
            if is_capex:
                self._stats["capex_detected"] += 1
            if category_num == 0:
                self._stats["unclassified"] += 1
            self._stats["total_confidence"] += confidence
            count = self._stats["records_mapped"]
            self._stats["avg_confidence"] = round(
                self._stats["total_confidence"] / count, 4,
            )

        elapsed = (time.monotonic() - start) * 1000
        logger.debug(
            "Mapped record %s -> Cat %d (%s) conf=%.2f src=%s (%.1f ms)",
            record_id[:8] if record_id else "?",
            category_num, cat_name, confidence, match_source, elapsed,
        )
        return assignment

    def map_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Scope3Assignment]:
        """Map a batch of spend records to Scope 3 categories.

        Args:
            records: List of spend record dicts.

        Returns:
            List of Scope3Assignment objects.
        """
        start = time.monotonic()
        results = [self.map_record(r) for r in records]
        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Batch mapped %d records in %.1f ms",
            len(results), elapsed,
        )
        return results

    # ------------------------------------------------------------------
    # Public API - Specific mappings
    # ------------------------------------------------------------------

    def map_from_naics(self, naics_code: str) -> Tuple[int, float]:
        """Map a NAICS code to a Scope 3 category.

        Tries progressively shorter code prefixes for best match.

        Args:
            naics_code: NAICS code (2-6 digits).

        Returns:
            Tuple of (category_number, confidence). Returns (1, 0.30)
            if no mapping found.
        """
        code = naics_code.strip()
        # Try exact, then progressively shorter prefixes
        for length in range(len(code), 1, -1):
            prefix = code[:length]
            if prefix in _NAICS_TO_SCOPE3:
                cat, conf = _NAICS_TO_SCOPE3[prefix]
                return cat, conf

        return self._default_category, 0.30

    def map_from_unspsc(self, unspsc_code: str) -> Tuple[int, float]:
        """Map a UNSPSC code to a Scope 3 category.

        Uses the 2-digit segment for mapping.

        Args:
            unspsc_code: UNSPSC code.

        Returns:
            Tuple of (category_number, confidence).
        """
        code = unspsc_code.strip()
        segment = code[:2] if len(code) >= 2 else code

        if segment in _UNSPSC_TO_SCOPE3:
            return _UNSPSC_TO_SCOPE3[segment]

        return self._default_category, 0.30

    def map_from_keyword(self, text: str) -> Tuple[int, float]:
        """Map text to a Scope 3 category using keywords.

        Scans the text for known keywords and returns the best match.

        Args:
            text: Description or combined text to search.

        Returns:
            Tuple of (category_number, confidence).
        """
        text_lower = text.lower().strip()
        best_cat = 0
        best_conf = 0.0

        for keyword, (cat, conf) in _KEYWORD_TO_SCOPE3.items():
            if keyword in text_lower:
                if conf > best_conf:
                    best_cat = cat
                    best_conf = conf

        if best_cat == 0:
            return self._default_category, 0.20
        return best_cat, best_conf

    def map_from_taxonomy(
        self,
        code: str,
        system: str,
    ) -> Tuple[int, float]:
        """Map a generic taxonomy code to a Scope 3 category.

        Delegates to the appropriate system-specific mapper.

        Args:
            code: Taxonomy code.
            system: Taxonomy system (naics, unspsc).

        Returns:
            Tuple of (category_number, confidence).
        """
        sys_lower = system.lower().strip()
        if sys_lower == "naics":
            return self.map_from_naics(code)
        elif sys_lower == "unspsc":
            return self.map_from_unspsc(code)
        else:
            logger.warning("Unsupported taxonomy system for Scope 3: %s", system)
            return self._default_category, 0.20

    # ------------------------------------------------------------------
    # Public API - CapEx detection and allocation
    # ------------------------------------------------------------------

    def detect_capex(self, record: Dict[str, Any]) -> bool:
        """Detect whether a spend record is a capital expenditure.

        Uses keyword-based detection from description, category,
        and GL account fields.

        Args:
            record: Spend record dict.

        Returns:
            True if the record appears to be a capital expenditure.
        """
        text_parts = []
        for key in ("description", "category", "gl_account", "expense_type"):
            val = record.get(key)
            if val:
                text_parts.append(str(val).lower())

        combined = " ".join(text_parts)

        for keyword in _CAPEX_KEYWORDS:
            if keyword in combined:
                return True

        # Check amount threshold (large amounts more likely CapEx)
        amount = float(record.get("amount_usd", 0) or 0)
        category = str(record.get("category", "")).lower()
        if amount > 50000 and ("equipment" in category or "asset" in category):
            return True

        return False

    def split_allocation(
        self,
        record: Dict[str, Any],
        categories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Split a spend record across multiple Scope 3 categories.

        Used when a single spend item spans multiple categories
        (e.g. a mixed services invoice).

        Args:
            record: Spend record dict with ``amount_usd``.
            categories: List of dicts with ``category_number`` and
                ``weight`` (0-1 allocation weight).

        Returns:
            List of allocation dicts with ``category_number``,
            ``amount_usd``, ``weight``, and ``provenance_hash``.
        """
        amount_usd = float(record.get("amount_usd", 0) or 0)

        # Normalize weights
        total_weight = sum(c.get("weight", 0) for c in categories)
        if total_weight <= 0:
            total_weight = len(categories)
            for c in categories:
                c["weight"] = 1.0 / len(categories)

        allocations: List[Dict[str, Any]] = []
        for cat in categories:
            weight = cat.get("weight", 0) / total_weight
            alloc_amount = round(amount_usd * weight, 2)
            cat_num = cat.get("category_number", self._default_category)
            cat_info = _SCOPE3_CATEGORIES.get(cat_num, _SCOPE3_CATEGORIES[1])

            provenance_hash = self._compute_provenance(
                f"alloc-{cat_num}",
                cat_num, weight, "allocation",
                _utcnow().isoformat(),
            )

            allocations.append({
                "category_number": cat_num,
                "category_name": cat_info["name"],
                "weight": round(weight, 4),
                "amount_usd": alloc_amount,
                "provenance_hash": provenance_hash,
            })

        return allocations

    # ------------------------------------------------------------------
    # Public API - Category info and statistics
    # ------------------------------------------------------------------

    def get_category_info(self, category_number: int) -> Dict[str, Any]:
        """Get metadata for a Scope 3 category.

        Args:
            category_number: Category number (1-15).

        Returns:
            Dictionary with name, direction, description, methodology.

        Raises:
            ValueError: If category_number is not 1-15.
        """
        if category_number not in _SCOPE3_CATEGORIES:
            raise ValueError(
                f"Invalid Scope 3 category: {category_number}. "
                f"Must be 1-15."
            )
        info = dict(_SCOPE3_CATEGORIES[category_number])
        info["category_number"] = category_number
        return info

    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Get metadata for all 15 Scope 3 categories.

        Returns:
            List of category metadata dicts.
        """
        result = []
        for num in range(1, 16):
            info = dict(_SCOPE3_CATEGORIES[num])
            info["category_number"] = num
            result.append(info)
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Return cumulative mapping statistics.

        Returns:
            Dictionary with mapping counters and breakdowns.
        """
        with self._lock:
            stats = dict(self._stats)
            stats["by_category"] = dict(self._stats["by_category"])
            stats["by_source"] = dict(self._stats["by_source"])
        stats["assignments_stored"] = len(self._assignments)
        stats["naics_rules_count"] = len(_NAICS_TO_SCOPE3)
        stats["unspsc_rules_count"] = len(_UNSPSC_TO_SCOPE3)
        stats["keyword_rules_count"] = len(_KEYWORD_TO_SCOPE3)
        return stats

    # ------------------------------------------------------------------
    # Internal - Provenance
    # ------------------------------------------------------------------

    def _compute_provenance(
        self,
        entity_id: str,
        category: Any,
        confidence: Any,
        source: str,
        timestamp: str,
    ) -> str:
        """Compute SHA-256 provenance hash for an assignment.

        Args:
            entity_id: Assignment or operation identifier.
            category: Category number or weight.
            confidence: Confidence score or weight.
            source: Match source.
            timestamp: Assignment timestamp.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        payload = json.dumps({
            "entity_id": entity_id,
            "category": str(category),
            "confidence": str(confidence),
            "source": source,
            "timestamp": timestamp,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
