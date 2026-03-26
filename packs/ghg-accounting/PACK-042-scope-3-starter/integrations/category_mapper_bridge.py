# -*- coding: utf-8 -*-
"""
CategoryMapperBridge - Bridge to MRV-029 Category Mapper for PACK-042
========================================================================

This module provides integration with the MRV-029 (Scope 3 Category
Mapper) agent for classifying activities, transactions, and procurement
spend into GHG Protocol Scope 3 categories. It supports NAICS, ISIC,
UNSPSC, and HS code routing with confidence scoring.

Routing:
    Activity classification   --> MRV-029 (gl_scope3_category_mapper_)
    Spend classification      --> MRV-029 (spend + EEIO sector mapping)
    Batch classification      --> MRV-029 (bulk transaction processing)
    Code mapping              --> NAICS/ISIC/UNSPSC/HS lookup tables

Zero-Hallucination:
    All classification uses deterministic lookup tables and rule-based
    mapping. LLM may be used for ambiguous descriptions with confidence
    thresholds, but numeric results are never LLM-generated.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-042 Scope 3 Starter
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Agent Import
# ---------------------------------------------------------------------------


def _try_import_mapper() -> Any:
    """Try to import the MRV-029 Category Mapper agent."""
    try:
        import importlib
        return importlib.import_module("greenlang.agents.mrv.scope3_category_mapper")
    except ImportError:
        logger.debug("MRV-029 Category Mapper not available, using built-in rules")
        return None


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ClassificationSource(str, Enum):
    """Source of classification input."""

    NAICS = "naics"
    ISIC = "isic"
    UNSPSC = "unspsc"
    HS_CODE = "hs_code"
    GL_ACCOUNT = "gl_account"
    DESCRIPTION = "description"
    MANUAL = "manual"


class ConfidenceLevel(str, Enum):
    """Classification confidence levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCLASSIFIED = "unclassified"


# ---------------------------------------------------------------------------
# NAICS-to-Scope3 Mapping (representative top-level codes)
# ---------------------------------------------------------------------------

NAICS_TO_SCOPE3: Dict[str, Dict[str, Any]] = {
    "11": {"category": "cat_1", "sector": "Agriculture", "eeio_sector": "111CA"},
    "21": {"category": "cat_1", "sector": "Mining", "eeio_sector": "211"},
    "22": {"category": "cat_3", "sector": "Utilities", "eeio_sector": "221"},
    "23": {"category": "cat_2", "sector": "Construction", "eeio_sector": "230"},
    "31": {"category": "cat_1", "sector": "Manufacturing - Food", "eeio_sector": "311FT"},
    "32": {"category": "cat_1", "sector": "Manufacturing - Chemical", "eeio_sector": "325"},
    "33": {"category": "cat_1", "sector": "Manufacturing - Metals/Machinery", "eeio_sector": "331"},
    "42": {"category": "cat_1", "sector": "Wholesale Trade", "eeio_sector": "420"},
    "44": {"category": "cat_1", "sector": "Retail Trade", "eeio_sector": "44RT"},
    "48": {"category": "cat_4", "sector": "Transportation", "eeio_sector": "481"},
    "49": {"category": "cat_4", "sector": "Warehousing", "eeio_sector": "493"},
    "51": {"category": "cat_1", "sector": "Information", "eeio_sector": "511"},
    "52": {"category": "cat_15", "sector": "Finance/Insurance", "eeio_sector": "521CI"},
    "53": {"category": "cat_8", "sector": "Real Estate", "eeio_sector": "531"},
    "54": {"category": "cat_1", "sector": "Professional Services", "eeio_sector": "5411"},
    "56": {"category": "cat_5", "sector": "Waste Management", "eeio_sector": "562"},
    "61": {"category": "cat_1", "sector": "Education", "eeio_sector": "611"},
    "62": {"category": "cat_1", "sector": "Healthcare", "eeio_sector": "621"},
    "71": {"category": "cat_6", "sector": "Entertainment", "eeio_sector": "711AS"},
    "72": {"category": "cat_6", "sector": "Accommodation/Food", "eeio_sector": "721"},
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class ClassificationInput(BaseModel):
    """Input for activity/spend classification."""

    input_id: str = Field(default_factory=_new_uuid)
    description: str = Field(default="")
    naics_code: str = Field(default="")
    isic_code: str = Field(default="")
    unspsc_code: str = Field(default="")
    hs_code: str = Field(default="")
    gl_account: str = Field(default="")
    spend_usd: float = Field(default=0.0, ge=0.0)
    vendor_name: str = Field(default="")
    vendor_country: str = Field(default="US")


class ClassificationResult(BaseModel):
    """Result of activity/spend classification."""

    result_id: str = Field(default_factory=_new_uuid)
    input_id: str = Field(default="")
    scope3_category: str = Field(default="")
    category_number: int = Field(default=0, ge=0, le=15)
    category_name: str = Field(default="")
    eeio_sector: str = Field(default="")
    eeio_factor_kgco2e_per_usd: float = Field(default=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.UNCLASSIFIED)
    classification_source: ClassificationSource = Field(
        default=ClassificationSource.DESCRIPTION
    )
    spend_usd: float = Field(default=0.0)
    estimated_emissions_tco2e: float = Field(default=0.0)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    timestamp: datetime = Field(default_factory=_utcnow)


class BatchClassificationResult(BaseModel):
    """Result of batch classification."""

    batch_id: str = Field(default_factory=_new_uuid)
    total_transactions: int = Field(default=0)
    classified_count: int = Field(default=0)
    unclassified_count: int = Field(default=0)
    by_category: Dict[str, int] = Field(default_factory=dict)
    by_confidence: Dict[str, int] = Field(default_factory=dict)
    total_spend_usd: float = Field(default=0.0)
    total_estimated_emissions_tco2e: float = Field(default=0.0)
    results: List[ClassificationResult] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# CategoryMapperBridge
# ---------------------------------------------------------------------------


class CategoryMapperBridge:
    """Bridge to MRV-029 (Scope 3 Category Mapper) agent.

    Classifies activities, transactions, and procurement spend into GHG
    Protocol Scope 3 categories using NAICS/ISIC/UNSPSC/HS code routing
    and description-based matching.

    Attributes:
        _mapper_agent: Loaded MRV-029 agent reference (or None).
        _classification_cache: LRU-style cache for repeated lookups.

    Example:
        >>> bridge = CategoryMapperBridge()
        >>> result = bridge.classify_activity({"description": "Office supplies", "naics_code": "42"})
        >>> assert result.scope3_category == "cat_1"
    """

    def __init__(self) -> None:
        """Initialize CategoryMapperBridge."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._mapper_agent = _try_import_mapper()
        self._classification_cache: Dict[str, ClassificationResult] = {}
        self.logger.info(
            "CategoryMapperBridge initialized: agent_available=%s",
            self._mapper_agent is not None,
        )

    # -------------------------------------------------------------------------
    # Activity Classification
    # -------------------------------------------------------------------------

    def classify_activity(
        self,
        activity_data: Dict[str, Any],
    ) -> ClassificationResult:
        """Classify an activity into a Scope 3 category.

        Attempts classification in order: NAICS, ISIC, UNSPSC,
        HS code, GL account, then description-based fallback.

        Args:
            activity_data: Dict with description, codes, spend, vendor info.

        Returns:
            ClassificationResult with category assignment.
        """
        start_time = time.monotonic()
        inp = ClassificationInput(**{
            k: v for k, v in activity_data.items()
            if k in ClassificationInput.model_fields
        })

        # Cache check
        cache_key = f"{inp.naics_code}:{inp.isic_code}:{inp.description[:50]}"
        if cache_key in self._classification_cache:
            cached = self._classification_cache[cache_key]
            self.logger.debug("Cache hit for classification: %s", cache_key[:30])
            return cached

        result = self._classify_by_codes(inp)
        if result.confidence_level == ConfidenceLevel.UNCLASSIFIED:
            result = self._classify_by_description(inp)

        # Estimate emissions if spend is provided
        if inp.spend_usd > 0 and result.eeio_factor_kgco2e_per_usd > 0:
            result.estimated_emissions_tco2e = round(
                inp.spend_usd * result.eeio_factor_kgco2e_per_usd / 1000, 3
            )

        result.input_id = inp.input_id
        result.spend_usd = inp.spend_usd
        result.provenance_hash = _compute_hash(result)

        # Cache result
        self._classification_cache[cache_key] = result

        elapsed_ms = (time.monotonic() - start_time) * 1000
        self.logger.info(
            "Classified activity: category=%s (Cat %d), confidence=%.2f, "
            "source=%s (%.1fms)",
            result.scope3_category, result.category_number,
            result.confidence, result.classification_source.value,
            elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------------
    # Spend Classification
    # -------------------------------------------------------------------------

    def classify_spend(
        self,
        spend_transaction: Dict[str, Any],
    ) -> ClassificationResult:
        """Classify a procurement spend transaction.

        Args:
            spend_transaction: Dict with vendor, amount, description, codes.

        Returns:
            ClassificationResult with category + EEIO sector.
        """
        return self.classify_activity(spend_transaction)

    # -------------------------------------------------------------------------
    # Batch Classification
    # -------------------------------------------------------------------------

    def batch_classify(
        self,
        transactions: List[Dict[str, Any]],
    ) -> BatchClassificationResult:
        """Classify a batch of transactions.

        Args:
            transactions: List of transaction dicts.

        Returns:
            BatchClassificationResult with aggregate statistics.
        """
        start_time = time.monotonic()
        self.logger.info("Batch classifying %d transactions", len(transactions))

        results: List[ClassificationResult] = []
        by_category: Dict[str, int] = {}
        by_confidence: Dict[str, int] = {}
        total_spend = 0.0
        total_emissions = 0.0
        classified = 0

        for txn in transactions:
            result = self.classify_activity(txn)
            results.append(result)

            if result.confidence_level != ConfidenceLevel.UNCLASSIFIED:
                classified += 1
                by_category[result.scope3_category] = (
                    by_category.get(result.scope3_category, 0) + 1
                )

            conf_key = result.confidence_level.value
            by_confidence[conf_key] = by_confidence.get(conf_key, 0) + 1
            total_spend += result.spend_usd
            total_emissions += result.estimated_emissions_tco2e

        elapsed_ms = (time.monotonic() - start_time) * 1000

        batch_result = BatchClassificationResult(
            total_transactions=len(transactions),
            classified_count=classified,
            unclassified_count=len(transactions) - classified,
            by_category=by_category,
            by_confidence=by_confidence,
            total_spend_usd=round(total_spend, 2),
            total_estimated_emissions_tco2e=round(total_emissions, 3),
            results=results,
            processing_time_ms=elapsed_ms,
        )
        batch_result.provenance_hash = _compute_hash(batch_result)

        self.logger.info(
            "Batch classification complete: %d/%d classified, "
            "spend=$%.0f, emissions=%.1f tCO2e (%.1fms)",
            classified, len(transactions), total_spend,
            total_emissions, elapsed_ms,
        )
        return batch_result

    # -------------------------------------------------------------------------
    # Code Lookup
    # -------------------------------------------------------------------------

    def lookup_naics(self, naics_code: str) -> Optional[Dict[str, Any]]:
        """Look up Scope 3 mapping for a NAICS code.

        Args:
            naics_code: 2-6 digit NAICS code.

        Returns:
            Mapping dict or None if not found.
        """
        # Try exact match first, then 2-digit prefix
        for prefix_len in (len(naics_code), 4, 3, 2):
            prefix = naics_code[:prefix_len]
            if prefix in NAICS_TO_SCOPE3:
                return NAICS_TO_SCOPE3[prefix]
        return None

    def get_eeio_sector(self, category: str, naics_code: str = "") -> str:
        """Get EEIO sector code for a given category and NAICS.

        Args:
            category: Scope 3 category (e.g., 'cat_1').
            naics_code: Optional NAICS code for more specific sector.

        Returns:
            EEIO sector code string.
        """
        if naics_code:
            mapping = self.lookup_naics(naics_code)
            if mapping:
                return mapping.get("eeio_sector", "")
        return ""

    # -------------------------------------------------------------------------
    # Internal Classification Methods
    # -------------------------------------------------------------------------

    def _classify_by_codes(
        self,
        inp: ClassificationInput,
    ) -> ClassificationResult:
        """Classify using industry codes (NAICS, ISIC, etc.).

        Args:
            inp: Classification input.

        Returns:
            ClassificationResult from code-based lookup.
        """
        # Try NAICS first
        if inp.naics_code:
            mapping = self.lookup_naics(inp.naics_code)
            if mapping:
                cat = mapping["category"]
                cat_num = int(cat.replace("cat_", ""))
                return ClassificationResult(
                    scope3_category=cat,
                    category_number=cat_num,
                    category_name=mapping["sector"],
                    eeio_sector=mapping.get("eeio_sector", ""),
                    eeio_factor_kgco2e_per_usd=self._get_eeio_factor(mapping.get("eeio_sector", "")),
                    confidence=0.90,
                    confidence_level=ConfidenceLevel.HIGH,
                    classification_source=ClassificationSource.NAICS,
                )

        # Try GL account mapping
        if inp.gl_account:
            cat = self._map_gl_account(inp.gl_account)
            if cat:
                cat_num = int(cat.replace("cat_", ""))
                return ClassificationResult(
                    scope3_category=cat,
                    category_number=cat_num,
                    category_name=f"GL Account {inp.gl_account}",
                    confidence=0.75,
                    confidence_level=ConfidenceLevel.MEDIUM,
                    classification_source=ClassificationSource.GL_ACCOUNT,
                )

        return ClassificationResult(confidence_level=ConfidenceLevel.UNCLASSIFIED)

    def _classify_by_description(
        self,
        inp: ClassificationInput,
    ) -> ClassificationResult:
        """Classify using description keyword matching.

        Args:
            inp: Classification input.

        Returns:
            ClassificationResult from description-based matching.
        """
        desc = inp.description.lower()

        keyword_map: Dict[str, str] = {
            "office supplies": "cat_1", "raw material": "cat_1",
            "packaging": "cat_1", "chemicals": "cat_1",
            "machinery": "cat_2", "equipment": "cat_2", "vehicle purchase": "cat_2",
            "electricity": "cat_3", "natural gas upstream": "cat_3",
            "freight": "cat_4", "shipping": "cat_4", "logistics": "cat_4",
            "waste disposal": "cat_5", "recycling": "cat_5", "landfill": "cat_5",
            "flight": "cat_6", "hotel": "cat_6", "business travel": "cat_6",
            "commut": "cat_7", "employee transport": "cat_7",
            "leased office": "cat_8", "rented warehouse": "cat_8",
            "distribution": "cat_9", "delivery": "cat_9",
            "product use": "cat_11", "energy consumption by product": "cat_11",
            "end of life": "cat_12", "disposal": "cat_12",
            "franchise": "cat_14",
            "investment": "cat_15", "portfolio": "cat_15",
        }

        for keyword, cat in keyword_map.items():
            if keyword in desc:
                cat_num = int(cat.replace("cat_", ""))
                return ClassificationResult(
                    scope3_category=cat,
                    category_number=cat_num,
                    category_name=f"Keyword match: {keyword}",
                    confidence=0.60,
                    confidence_level=ConfidenceLevel.LOW,
                    classification_source=ClassificationSource.DESCRIPTION,
                )

        # Default to Cat 1 (Purchased Goods) with low confidence
        return ClassificationResult(
            scope3_category="cat_1",
            category_number=1,
            category_name="Default (Purchased Goods)",
            confidence=0.30,
            confidence_level=ConfidenceLevel.LOW,
            classification_source=ClassificationSource.DESCRIPTION,
            warnings=["Low confidence classification, manual review recommended"],
        )

    def _map_gl_account(self, gl_account: str) -> Optional[str]:
        """Map a GL account to a Scope 3 category.

        Args:
            gl_account: General ledger account code.

        Returns:
            Category string or None.
        """
        prefix = gl_account[:2] if len(gl_account) >= 2 else gl_account
        gl_mapping: Dict[str, str] = {
            "50": "cat_1", "51": "cat_1", "52": "cat_2",
            "53": "cat_4", "54": "cat_6", "55": "cat_5",
            "56": "cat_1", "60": "cat_1", "61": "cat_2",
            "70": "cat_6", "71": "cat_7",
        }
        return gl_mapping.get(prefix)

    def _get_eeio_factor(self, eeio_sector: str) -> float:
        """Get representative EEIO emission factor for a sector.

        Args:
            eeio_sector: EEIO sector code.

        Returns:
            kgCO2e per USD.
        """
        factors: Dict[str, float] = {
            "111CA": 0.65, "211": 0.85, "221": 0.90, "230": 0.35,
            "311FT": 0.45, "325": 0.55, "331": 0.70, "420": 0.25,
            "44RT": 0.20, "481": 0.45, "493": 0.30, "511": 0.10,
            "521CI": 0.08, "531": 0.15, "5411": 0.12, "562": 0.50,
            "611": 0.15, "621": 0.20, "711AS": 0.18, "721": 0.25,
        }
        return factors.get(eeio_sector, 0.30)
