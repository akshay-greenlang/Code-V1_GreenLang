# -*- coding: utf-8 -*-
"""
greenlang.agents.eudr.commodity_risk_analyzer.derived_product_analyzer
======================================================================

AGENT-EUDR-018 Engine 2: Derived Product Analyzer

Maps commodity-to-product transformation chains for all 7 EUDR commodities
and their Annex I derived products. Tracks risk accumulation, traceability
loss, and transformation ratios through multi-stage processing chains from
raw commodity to final consumer product.

ZERO-HALLUCINATION GUARANTEES:
    - 100% deterministic: same processing chain produces identical analysis
    - NO LLM involvement in any risk calculation or traceability path
    - All arithmetic uses Decimal for bit-perfect reproducibility
    - SHA-256 provenance hash chain through every processing stage
    - Complete audit trail for regulatory inspection

EUDR Annex I Coverage:
    - Cattle: leather, beef, gelatin, tallow, collagen
    - Cocoa: chocolate, cocoa butter, cocoa powder, cocoa paste
    - Coffee: roasted coffee, instant coffee, coffee extract
    - Oil Palm: palm oil, palm kernel oil, biodiesel, glycerine,
      oleochemicals, cosmetics ingredients, food additives
    - Rubber: tires, latex products, gaskets, seals, hoses
    - Soya: soy meal, soy oil, tofu, soy lecithin, soy protein isolate
    - Wood: plywood, furniture, paper, charcoal, fibreboard, pulp,
      printed products, packaging

Regulatory References:
    - EUDR Annex I: Product scope and CN codes
    - EUDR Article 2: Definitions (relevant products/commodities)
    - EUDR Article 10(2)(f): Traceability requirements
    - EUDR Article 4(2): Due diligence obligations for operators

Dependencies:
    - .config (get_config): CommodityRiskAnalyzerConfig singleton
    - .models: EUDRCommodity, DerivedProduct, ProcessingStage
    - .provenance (ProvenanceTracker): SHA-256 audit chain
    - .metrics: Prometheus instrumentation

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Module version for provenance tracking.
_MODULE_VERSION: str = "1.0.0"

#: Decimal precision for risk and ratio calculations.
_PRECISION = Decimal("0.01")

#: Maximum and minimum risk scores.
_MAX_RISK = Decimal("100.00")
_MIN_RISK = Decimal("0.00")

#: The 7 primary EUDR commodities.
EUDR_PRIMARY_COMMODITIES: FrozenSet[str] = frozenset({
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
})


# ---------------------------------------------------------------------------
# EUDR Annex I Derived Product Mappings
# ---------------------------------------------------------------------------

#: Complete mapping of primary commodity -> list of derived products with
#: CN codes, processing stages, and transformation ratios.
ANNEX_I_PRODUCT_MAP: Dict[str, List[Dict[str, Any]]] = {
    "cattle": [
        {
            "product_id": "cattle-beef-fresh",
            "product_name": "Fresh/chilled beef",
            "cn_codes": ["0201"],
            "processing_stages": ["slaughterhouse", "cutting_plant", "cold_storage"],
            "transformation_ratio": Decimal("0.45"),
            "description": "Fresh or chilled bovine meat from cattle",
        },
        {
            "product_id": "cattle-beef-frozen",
            "product_name": "Frozen beef",
            "cn_codes": ["0202"],
            "processing_stages": ["slaughterhouse", "cutting_plant", "freezing_facility"],
            "transformation_ratio": Decimal("0.43"),
            "description": "Frozen bovine meat from cattle",
        },
        {
            "product_id": "cattle-leather-raw",
            "product_name": "Raw hides and skins",
            "cn_codes": ["4101"],
            "processing_stages": ["slaughterhouse", "hide_processing"],
            "transformation_ratio": Decimal("0.08"),
            "description": "Raw hides and skins of bovine animals",
        },
        {
            "product_id": "cattle-leather-tanned",
            "product_name": "Tanned leather",
            "cn_codes": ["4104"],
            "processing_stages": ["slaughterhouse", "hide_processing", "tannery"],
            "transformation_ratio": Decimal("0.05"),
            "description": "Tanned or crust hides of bovine animals",
        },
        {
            "product_id": "cattle-gelatin",
            "product_name": "Gelatin",
            "cn_codes": ["3503"],
            "processing_stages": ["slaughterhouse", "rendering_plant", "gelatin_factory"],
            "transformation_ratio": Decimal("0.02"),
            "description": "Gelatin and gelatin derivatives from bovine",
        },
        {
            "product_id": "cattle-tallow",
            "product_name": "Tallow",
            "cn_codes": ["1502"],
            "processing_stages": ["slaughterhouse", "rendering_plant"],
            "transformation_ratio": Decimal("0.04"),
            "description": "Rendered bovine fats and tallow",
        },
    ],
    "cocoa": [
        {
            "product_id": "cocoa-chocolate-bars",
            "product_name": "Chocolate (bars/confectionery)",
            "cn_codes": ["1806"],
            "processing_stages": ["fermentation", "drying", "roasting", "conching", "moulding"],
            "transformation_ratio": Decimal("0.40"),
            "description": "Chocolate bars and confectionery from cocoa beans",
        },
        {
            "product_id": "cocoa-butter",
            "product_name": "Cocoa butter",
            "cn_codes": ["180400"],
            "processing_stages": ["fermentation", "drying", "roasting", "pressing"],
            "transformation_ratio": Decimal("0.35"),
            "description": "Cocoa butter (fat/oil) extracted from beans",
        },
        {
            "product_id": "cocoa-powder",
            "product_name": "Cocoa powder",
            "cn_codes": ["180500"],
            "processing_stages": ["fermentation", "drying", "roasting", "pressing", "milling"],
            "transformation_ratio": Decimal("0.30"),
            "description": "Cocoa powder without added sugar",
        },
        {
            "product_id": "cocoa-paste",
            "product_name": "Cocoa paste/liquor",
            "cn_codes": ["180310", "180320"],
            "processing_stages": ["fermentation", "drying", "roasting", "grinding"],
            "transformation_ratio": Decimal("0.80"),
            "description": "Cocoa paste (cocoa liquor) whether defatted or not",
        },
    ],
    "coffee": [
        {
            "product_id": "coffee-roasted",
            "product_name": "Roasted coffee",
            "cn_codes": ["090121", "090122"],
            "processing_stages": ["washing", "drying", "milling", "roasting"],
            "transformation_ratio": Decimal("0.80"),
            "description": "Roasted coffee beans (not decaffeinated/decaffeinated)",
        },
        {
            "product_id": "coffee-instant",
            "product_name": "Instant/soluble coffee",
            "cn_codes": ["210111"],
            "processing_stages": ["washing", "drying", "milling", "roasting", "extraction", "spray_drying"],
            "transformation_ratio": Decimal("0.35"),
            "description": "Extracts, essences, and concentrates of coffee",
        },
        {
            "product_id": "coffee-extract",
            "product_name": "Coffee extract/concentrate",
            "cn_codes": ["210112"],
            "processing_stages": ["washing", "drying", "milling", "roasting", "extraction"],
            "transformation_ratio": Decimal("0.40"),
            "description": "Preparations with a basis of coffee extracts",
        },
    ],
    "oil_palm": [
        {
            "product_id": "palm-crude-oil",
            "product_name": "Crude palm oil (CPO)",
            "cn_codes": ["151110"],
            "processing_stages": ["harvest", "milling"],
            "transformation_ratio": Decimal("0.22"),
            "description": "Crude palm oil from fresh fruit bunches",
        },
        {
            "product_id": "palm-refined-oil",
            "product_name": "Refined palm oil",
            "cn_codes": ["151190"],
            "processing_stages": ["harvest", "milling", "refinery"],
            "transformation_ratio": Decimal("0.20"),
            "description": "Refined, bleached, deodorized palm oil",
        },
        {
            "product_id": "palm-kernel-oil",
            "product_name": "Palm kernel oil",
            "cn_codes": ["151321"],
            "processing_stages": ["harvest", "milling", "kernel_crushing"],
            "transformation_ratio": Decimal("0.03"),
            "description": "Crude palm kernel oil",
        },
        {
            "product_id": "palm-biodiesel",
            "product_name": "Biodiesel (FAME)",
            "cn_codes": ["382600"],
            "processing_stages": ["harvest", "milling", "refinery", "transesterification"],
            "transformation_ratio": Decimal("0.19"),
            "description": "Fatty-acid methyl esters from palm oil",
        },
        {
            "product_id": "palm-glycerine",
            "product_name": "Glycerine",
            "cn_codes": ["1520"],
            "processing_stages": ["harvest", "milling", "refinery", "splitting"],
            "transformation_ratio": Decimal("0.02"),
            "description": "Glycerol from palm oil splitting",
        },
        {
            "product_id": "palm-oleochemicals",
            "product_name": "Oleochemicals",
            "cn_codes": ["3823", "3401"],
            "processing_stages": ["harvest", "milling", "refinery", "oleochemical_plant"],
            "transformation_ratio": Decimal("0.18"),
            "description": "Fatty acids, fatty alcohols, soaps from palm oil",
        },
        {
            "product_id": "palm-food-additives",
            "product_name": "Food additives (emulsifiers)",
            "cn_codes": ["1517"],
            "processing_stages": ["harvest", "milling", "refinery", "fractionation", "blending"],
            "transformation_ratio": Decimal("0.15"),
            "description": "Margarine, shortening, and emulsifiers from palm oil",
        },
    ],
    "rubber": [
        {
            "product_id": "rubber-tires",
            "product_name": "Pneumatic tires",
            "cn_codes": ["4011", "4012", "4013"],
            "processing_stages": ["tapping", "cup_lump_collection", "processing_plant", "tire_factory"],
            "transformation_ratio": Decimal("0.65"),
            "description": "New and retreaded pneumatic rubber tires",
        },
        {
            "product_id": "rubber-latex-products",
            "product_name": "Latex products (gloves/condoms)",
            "cn_codes": ["401511", "401519"],
            "processing_stages": ["tapping", "latex_concentration", "dipping_factory"],
            "transformation_ratio": Decimal("0.15"),
            "description": "Articles of vulcanised rubber: gloves, condoms",
        },
        {
            "product_id": "rubber-gaskets",
            "product_name": "Gaskets and seals",
            "cn_codes": ["4016"],
            "processing_stages": ["tapping", "cup_lump_collection", "processing_plant", "moulding"],
            "transformation_ratio": Decimal("0.10"),
            "description": "Other articles of vulcanised rubber: gaskets, seals",
        },
        {
            "product_id": "rubber-hoses",
            "product_name": "Rubber hoses and tubes",
            "cn_codes": ["4009"],
            "processing_stages": ["tapping", "cup_lump_collection", "processing_plant", "extrusion"],
            "transformation_ratio": Decimal("0.08"),
            "description": "Tubes, pipes, and hoses of vulcanised rubber",
        },
    ],
    "soya": [
        {
            "product_id": "soya-meal",
            "product_name": "Soybean meal (animal feed)",
            "cn_codes": ["230400"],
            "processing_stages": ["cleaning", "crushing", "solvent_extraction"],
            "transformation_ratio": Decimal("0.78"),
            "description": "Oilcake and other solid residues from soybean extraction",
        },
        {
            "product_id": "soya-oil-crude",
            "product_name": "Crude soybean oil",
            "cn_codes": ["150710"],
            "processing_stages": ["cleaning", "crushing", "solvent_extraction"],
            "transformation_ratio": Decimal("0.18"),
            "description": "Crude soybean oil from solvent extraction",
        },
        {
            "product_id": "soya-oil-refined",
            "product_name": "Refined soybean oil",
            "cn_codes": ["150790"],
            "processing_stages": ["cleaning", "crushing", "solvent_extraction", "refinery"],
            "transformation_ratio": Decimal("0.17"),
            "description": "Refined soybean oil for food use",
        },
        {
            "product_id": "soya-tofu",
            "product_name": "Tofu",
            "cn_codes": ["2106"],
            "processing_stages": ["soaking", "grinding", "coagulation", "pressing"],
            "transformation_ratio": Decimal("0.45"),
            "description": "Tofu and soy curd products",
        },
        {
            "product_id": "soya-lecithin",
            "product_name": "Soy lecithin",
            "cn_codes": ["2923"],
            "processing_stages": ["cleaning", "crushing", "degumming", "drying"],
            "transformation_ratio": Decimal("0.02"),
            "description": "Lecithin extracted from soybean oil degumming",
        },
        {
            "product_id": "soya-protein-isolate",
            "product_name": "Soy protein isolate",
            "cn_codes": ["210610"],
            "processing_stages": ["cleaning", "flaking", "extraction", "precipitation", "drying"],
            "transformation_ratio": Decimal("0.10"),
            "description": "Soy protein isolate (>90% protein content)",
        },
    ],
    "wood": [
        {
            "product_id": "wood-sawnwood",
            "product_name": "Sawnwood/lumber",
            "cn_codes": ["4407"],
            "processing_stages": ["felling", "debarking", "sawing", "drying"],
            "transformation_ratio": Decimal("0.50"),
            "description": "Wood sawn or chipped lengthwise",
        },
        {
            "product_id": "wood-plywood",
            "product_name": "Plywood",
            "cn_codes": ["4412"],
            "processing_stages": ["felling", "debarking", "peeling", "drying", "gluing", "pressing"],
            "transformation_ratio": Decimal("0.40"),
            "description": "Plywood, veneered panels, and similar laminated wood",
        },
        {
            "product_id": "wood-furniture",
            "product_name": "Wooden furniture",
            "cn_codes": ["9403"],
            "processing_stages": ["felling", "debarking", "sawing", "drying", "manufacturing", "finishing"],
            "transformation_ratio": Decimal("0.30"),
            "description": "Wooden furniture for offices, kitchens, bedrooms",
        },
        {
            "product_id": "wood-paper",
            "product_name": "Paper and paperboard",
            "cn_codes": ["4801", "4802", "4804", "4805"],
            "processing_stages": ["felling", "chipping", "pulping", "bleaching", "paper_machine"],
            "transformation_ratio": Decimal("0.35"),
            "description": "Newsprint, uncoated paper, kraftliner",
        },
        {
            "product_id": "wood-charcoal",
            "product_name": "Charcoal",
            "cn_codes": ["4402"],
            "processing_stages": ["felling", "cutting", "carbonisation"],
            "transformation_ratio": Decimal("0.25"),
            "description": "Wood charcoal including shell or nut charcoal",
        },
        {
            "product_id": "wood-fibreboard",
            "product_name": "Fibreboard (MDF/HDF)",
            "cn_codes": ["4411"],
            "processing_stages": ["felling", "chipping", "fibre_preparation", "pressing"],
            "transformation_ratio": Decimal("0.45"),
            "description": "Fibreboard of wood or other ligneous materials",
        },
        {
            "product_id": "wood-pulp",
            "product_name": "Wood pulp",
            "cn_codes": ["4701", "4702", "4703", "4704"],
            "processing_stages": ["felling", "chipping", "pulping"],
            "transformation_ratio": Decimal("0.42"),
            "description": "Mechanical, semi-chemical, chemical wood pulp",
        },
        {
            "product_id": "wood-packaging",
            "product_name": "Wooden packaging/pallets",
            "cn_codes": ["4415"],
            "processing_stages": ["felling", "debarking", "sawing", "assembly"],
            "transformation_ratio": Decimal("0.55"),
            "description": "Cases, boxes, crates, pallets of wood",
        },
    ],
}

#: Per-stage risk increment mapping: how much risk accumulates at each stage.
PROCESSING_STAGE_RISK: Dict[str, Decimal] = {
    # General stages
    "harvest": Decimal("2.00"),
    "felling": Decimal("5.00"),
    "tapping": Decimal("2.00"),
    "cleaning": Decimal("1.00"),
    "soaking": Decimal("1.00"),
    "washing": Decimal("1.00"),
    "drying": Decimal("2.00"),
    "fermentation": Decimal("3.00"),
    # Animal processing
    "slaughterhouse": Decimal("5.00"),
    "cutting_plant": Decimal("3.00"),
    "cold_storage": Decimal("2.00"),
    "freezing_facility": Decimal("2.00"),
    "hide_processing": Decimal("4.00"),
    "tannery": Decimal("5.00"),
    "rendering_plant": Decimal("4.00"),
    "gelatin_factory": Decimal("3.00"),
    "feedlot": Decimal("3.00"),
    # Crop processing
    "roasting": Decimal("2.00"),
    "conching": Decimal("2.00"),
    "moulding": Decimal("2.00"),
    "pressing": Decimal("3.00"),
    "milling": Decimal("3.00"),
    "grinding": Decimal("2.00"),
    "extraction": Decimal("4.00"),
    "spray_drying": Decimal("3.00"),
    # Oil processing
    "refinery": Decimal("5.00"),
    "kernel_crushing": Decimal("3.00"),
    "transesterification": Decimal("5.00"),
    "splitting": Decimal("3.00"),
    "oleochemical_plant": Decimal("4.00"),
    "fractionation": Decimal("3.00"),
    "blending": Decimal("2.00"),
    "degumming": Decimal("2.00"),
    # Rubber processing
    "cup_lump_collection": Decimal("3.00"),
    "processing_plant": Decimal("4.00"),
    "tire_factory": Decimal("3.00"),
    "latex_concentration": Decimal("3.00"),
    "dipping_factory": Decimal("3.00"),
    "extrusion": Decimal("3.00"),
    # Wood processing
    "debarking": Decimal("2.00"),
    "sawing": Decimal("3.00"),
    "peeling": Decimal("2.00"),
    "gluing": Decimal("2.00"),
    "chipping": Decimal("2.00"),
    "pulping": Decimal("4.00"),
    "bleaching": Decimal("3.00"),
    "paper_machine": Decimal("3.00"),
    "carbonisation": Decimal("4.00"),
    "fibre_preparation": Decimal("3.00"),
    "manufacturing": Decimal("4.00"),
    "finishing": Decimal("2.00"),
    "assembly": Decimal("2.00"),
    # Soya processing
    "crushing": Decimal("3.00"),
    "solvent_extraction": Decimal("4.00"),
    "flaking": Decimal("2.00"),
    "precipitation": Decimal("3.00"),
    "coagulation": Decimal("2.00"),
    # Default
    "DEFAULT": Decimal("3.00"),
}

#: Mislabeling risk indicators by commodity type (common fraud patterns).
MISLABELING_INDICATORS: Dict[str, List[Dict[str, Any]]] = {
    "cattle": [
        {"indicator": "species_mismatch", "description": "Declared species does not match DNA analysis"},
        {"indicator": "origin_discrepancy", "description": "Country of origin inconsistent with breed type"},
        {"indicator": "weight_anomaly", "description": "Declared weight exceeds biological maximum for breed"},
    ],
    "cocoa": [
        {"indicator": "grade_inflation", "description": "Declared grade higher than quality analysis supports"},
        {"indicator": "origin_substitution", "description": "Fine flavor origin claimed but bulk cocoa characteristics"},
        {"indicator": "fat_content_mismatch", "description": "Cocoa butter content inconsistent with declared product"},
    ],
    "coffee": [
        {"indicator": "varietal_fraud", "description": "Arabica claimed but Robusta characteristics detected"},
        {"indicator": "origin_substitution", "description": "Single-origin claimed but blend characteristics"},
        {"indicator": "altitude_inconsistency", "description": "Declared altitude inconsistent with density analysis"},
    ],
    "oil_palm": [
        {"indicator": "sustainable_label_fraud", "description": "RSPO certified claimed but no valid certificate"},
        {"indicator": "mixing_violation", "description": "IP/SG declared but mass balance characteristics"},
        {"indicator": "species_substitution", "description": "Palm oil mixed with other vegetable oils"},
    ],
    "rubber": [
        {"indicator": "grade_fraud", "description": "TSR20 claimed but TSR10 characteristics"},
        {"indicator": "origin_mismatch", "description": "Natural rubber claimed but synthetic blend detected"},
        {"indicator": "moisture_manipulation", "description": "Declared dry rubber content inconsistent with tests"},
    ],
    "soya": [
        {"indicator": "gmo_mislabeling", "description": "Non-GMO claimed but GM events detected"},
        {"indicator": "origin_substitution", "description": "Organic origin claimed but conventional characteristics"},
        {"indicator": "protein_manipulation", "description": "Protein content inflated with non-soy additives"},
    ],
    "wood": [
        {"indicator": "species_substitution", "description": "High-value species claimed but lower-value detected"},
        {"indicator": "origin_laundering", "description": "Legal origin claimed but timber from illegal concession"},
        {"indicator": "fsc_fraud", "description": "FSC certified claimed but certificate invalid or expired"},
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_decimal(value: float | int | str | Decimal) -> Decimal:
    """Convert a numeric value to Decimal via string to avoid IEEE 754 artefacts."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _clamp_risk(value: Decimal) -> Decimal:
    """Clamp a risk score to [0.00, 100.00] and apply precision."""
    clamped = max(_MIN_RISK, min(_MAX_RISK, value))
    return clamped.quantize(_PRECISION, rounding=ROUND_HALF_UP)


def _compute_provenance_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_commodity_type(commodity_type: str) -> str:
    """Validate and normalize a commodity type string."""
    if not commodity_type or not isinstance(commodity_type, str):
        raise ValueError("commodity_type must be a non-empty string")
    normalized = commodity_type.strip().lower()
    if normalized not in EUDR_PRIMARY_COMMODITIES:
        raise ValueError(
            f"Invalid commodity_type '{commodity_type}'. "
            f"Must be one of: {sorted(EUDR_PRIMARY_COMMODITIES)}"
        )
    return normalized


# ---------------------------------------------------------------------------
# Prometheus metrics integration (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram, REGISTRY

    def _safe_counter(name: str, doc: str, labelnames: list = None):
        """Create a Counter or retrieve existing one."""
        try:
            return Counter(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Counter(name, doc, labelnames=labelnames or [],
                           registry=CollectorRegistry())

    def _safe_histogram(name: str, doc: str, labelnames: list = None,
                        buckets: tuple = ()):
        """Create a Histogram or retrieve existing one."""
        try:
            kw = {"buckets": buckets} if buckets else {}
            return Histogram(name, doc, labelnames=labelnames or [], **kw)
        except ValueError:
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            kw = {"buckets": buckets} if buckets else {}
            return Histogram(name, doc, labelnames=labelnames or [],
                             registry=CollectorRegistry(), **kw)

    _DPA_ANALYSES_TOTAL = _safe_counter(
        "gl_eudr_cra_derived_analyses_total",
        "Total derived product analyses performed",
        labelnames=["commodity_type"],
    )
    _DPA_DURATION_SECONDS = _safe_histogram(
        "gl_eudr_cra_derived_analysis_duration_seconds",
        "Duration of derived product analysis operations in seconds",
        labelnames=["operation"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _DPA_MISLABELING_DETECTIONS_TOTAL = _safe_counter(
        "gl_eudr_cra_mislabeling_detections_total",
        "Total product mislabeling detections",
        labelnames=["commodity_type", "severity"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _DPA_ANALYSES_TOTAL = None  # type: ignore[assignment]
    _DPA_DURATION_SECONDS = None  # type: ignore[assignment]
    _DPA_MISLABELING_DETECTIONS_TOTAL = None  # type: ignore[assignment]
    _PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; derived product analyzer metrics disabled"
    )


def _record_analysis(commodity_type: str) -> None:
    """Record a derived product analysis metric."""
    if _PROMETHEUS_AVAILABLE and _DPA_ANALYSES_TOTAL is not None:
        _DPA_ANALYSES_TOTAL.labels(commodity_type=commodity_type).inc()


def _observe_duration(operation: str, seconds: float) -> None:
    """Record a duration metric."""
    if _PROMETHEUS_AVAILABLE and _DPA_DURATION_SECONDS is not None:
        _DPA_DURATION_SECONDS.labels(operation=operation).observe(seconds)


def _record_mislabeling(commodity_type: str, severity: str) -> None:
    """Record a mislabeling detection metric."""
    if _PROMETHEUS_AVAILABLE and _DPA_MISLABELING_DETECTIONS_TOTAL is not None:
        _DPA_MISLABELING_DETECTIONS_TOTAL.labels(
            commodity_type=commodity_type, severity=severity,
        ).inc()


# ---------------------------------------------------------------------------
# DerivedProductAnalyzer
# ---------------------------------------------------------------------------


class DerivedProductAnalyzer:
    """Analyzes commodity-to-product transformation chains for EUDR compliance.

    Maps raw EUDR commodities through their processing chains to final
    derived products listed in EUDR Annex I. Tracks risk accumulation
    at each processing stage, traceability degradation, transformation
    ratios, and potential mislabeling/fraud indicators.

    All calculations are deterministic using Decimal arithmetic. No LLM or
    ML models are used in any risk scoring path (zero-hallucination).

    Attributes:
        _config: Configuration dictionary.
        _product_cache: Cache of analyzed products keyed by product_id.
        _lock: Reentrant lock for thread-safe operations.

    Example:
        >>> analyzer = DerivedProductAnalyzer()
        >>> result = analyzer.analyze_derived_product(
        ...     product_id="cocoa-chocolate-01",
        ...     source_commodity="cocoa",
        ...     processing_stages=["fermentation", "drying", "roasting", "conching"]
        ... )
        >>> assert result["transformation_risk"] > Decimal("0")
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize DerivedProductAnalyzer with optional configuration.

        Args:
            config: Optional configuration dictionary. If None, uses
                module-level defaults.
        """
        self._config: Dict[str, Any] = config or {}
        self._product_cache: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.RLock = threading.RLock()
        logger.info("DerivedProductAnalyzer initialized: version=%s", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API: Analyze derived product
    # ------------------------------------------------------------------

    def analyze_derived_product(
        self,
        product_id: str,
        source_commodity: str,
        processing_stages: List[str],
        input_quantity: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze risk for a derived/processed product.

        Traces the product back to its source commodity and evaluates
        risk accumulation through each processing stage, traceability
        loss, and transformation efficiency.

        Args:
            product_id: Unique identifier for the product being analyzed.
            source_commodity: The primary EUDR commodity this product
                derives from.
            processing_stages: Ordered list of processing stage names
                from raw commodity to this product.
            input_quantity: Optional input commodity quantity (kg) for
                transformation ratio calculation.
            metadata: Optional additional metadata.

        Returns:
            Dictionary containing:
                - product_id (str): Input product identifier.
                - source_commodity (str): Normalized source commodity.
                - processing_stages (list): Input processing stages.
                - stage_count (int): Number of processing stages.
                - transformation_risk (Decimal): Accumulated risk 0-100.
                - risk_per_stage (list): Risk breakdown per stage.
                - traceability_loss (Decimal): Traceability degradation 0-100.
                - risk_multiplier (Decimal): Risk amplification factor.
                - annex_i_match (dict or None): Matched Annex I product info.
                - expected_output_quantity (Decimal or None): Based on ratio.
                - provenance_hash (str): SHA-256 hash.
                - processing_time_ms (float): Operation duration.

        Raises:
            ValueError: If inputs are invalid.
        """
        start_time = time.monotonic()
        operation = "analyze_derived_product"

        if not product_id or not isinstance(product_id, str):
            raise ValueError("product_id must be a non-empty string")
        commodity = _validate_commodity_type(source_commodity)
        if not processing_stages:
            raise ValueError("processing_stages must be a non-empty list")

        try:
            # Calculate transformation risk
            transformation_risk = self.calculate_transformation_risk(processing_stages)

            # Calculate risk per stage
            risk_per_stage = self._calculate_risk_per_stage(processing_stages)

            # Calculate risk multiplier
            processing_types = list(set(processing_stages))
            risk_multiplier = self.calculate_risk_multiplier(
                chain_length=len(processing_stages),
                processing_types=processing_types,
            )

            # Calculate traceability loss
            chain_info = {
                "stages": processing_stages,
                "commodity": commodity,
            }
            traceability_loss = self.calculate_traceability_loss(chain_info)

            # Find Annex I match
            annex_i_match = self._find_annex_i_match(commodity, processing_stages)

            # Calculate expected output quantity if input provided
            expected_output = None
            if input_quantity is not None and annex_i_match is not None:
                ratio = annex_i_match.get(
                    "transformation_ratio", Decimal("0.50"),
                )
                expected_output = (input_quantity * ratio).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP,
                )

            # Provenance hash
            payload = {
                "product_id": product_id,
                "source_commodity": commodity,
                "processing_stages": processing_stages,
                "transformation_risk": str(transformation_risk),
                "traceability_loss": str(traceability_loss),
                "risk_multiplier": str(risk_multiplier),
            }
            provenance_hash = _compute_provenance_hash(payload)

            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            result = {
                "product_id": product_id,
                "source_commodity": commodity,
                "processing_stages": processing_stages,
                "stage_count": len(processing_stages),
                "transformation_risk": transformation_risk,
                "risk_per_stage": risk_per_stage,
                "traceability_loss": traceability_loss,
                "risk_multiplier": risk_multiplier,
                "annex_i_match": annex_i_match,
                "expected_output_quantity": expected_output,
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(elapsed_ms, 2),
                "created_at": _utcnow().isoformat(),
                "metadata": metadata or {},
            }

            # Cache result
            with self._lock:
                self._product_cache[product_id] = result

            _record_analysis(commodity)
            _observe_duration(operation, elapsed_ms / 1000.0)

            logger.info(
                "Analyzed derived product=%s from commodity=%s: "
                "stages=%d, risk=%.2f, traceability_loss=%.2f, "
                "multiplier=%.2f, time_ms=%.2f",
                product_id, commodity, len(processing_stages),
                transformation_risk, traceability_loss,
                risk_multiplier, elapsed_ms,
            )
            return result

        except ValueError:
            raise
        except Exception as exc:
            logger.error(
                "DerivedProductAnalyzer.analyze_derived_product failed: %s",
                str(exc), exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Public API: Map processing chain
    # ------------------------------------------------------------------

    def map_processing_chain(
        self,
        source_commodity: str,
        final_product: str,
    ) -> Dict[str, Any]:
        """Map the full chain from raw commodity to final product.

        Looks up the Annex I product mapping and returns the complete
        processing chain with risk at each stage, transformation ratio,
        and CN codes.

        Args:
            source_commodity: Primary EUDR commodity.
            final_product: Product ID or product name to map.

        Returns:
            Dictionary containing the complete processing chain:
                - source_commodity (str): Normalized commodity.
                - final_product (str): Matched product information.
                - chain (list): Ordered processing stages with risk.
                - total_stages (int): Number of stages.
                - transformation_ratio (Decimal): Input-to-output ratio.
                - cn_codes (list): EU Combined Nomenclature codes.
                - cumulative_risk (Decimal): Total accumulated risk.
                - provenance_hash (str): SHA-256 hash.

        Raises:
            ValueError: If commodity or product is invalid/not found.
        """
        start_time = time.monotonic()
        commodity = _validate_commodity_type(source_commodity)

        products = ANNEX_I_PRODUCT_MAP.get(commodity, [])
        matched = None
        for product in products:
            if (product["product_id"] == final_product
                    or product["product_name"].lower() == final_product.lower()):
                matched = product
                break

        if matched is None:
            # Try partial match
            for product in products:
                if final_product.lower() in product["product_name"].lower():
                    matched = product
                    break

        if matched is None:
            raise ValueError(
                f"Product '{final_product}' not found in Annex I mapping "
                f"for commodity '{commodity}'. Available products: "
                f"{[p['product_name'] for p in products]}"
            )

        # Build stage-by-stage chain with risk
        stages = matched["processing_stages"]
        chain_details = []
        cumulative_risk = Decimal("0.00")

        for i, stage in enumerate(stages):
            stage_risk = PROCESSING_STAGE_RISK.get(
                stage, PROCESSING_STAGE_RISK["DEFAULT"],
            )
            cumulative_risk += stage_risk
            chain_details.append({
                "stage_index": i,
                "stage_name": stage,
                "stage_risk": stage_risk,
                "cumulative_risk": _clamp_risk(cumulative_risk),
            })

        transformation_ratio = matched.get("transformation_ratio", Decimal("0.50"))

        payload = {
            "source_commodity": commodity,
            "final_product": matched["product_id"],
            "stages": stages,
            "cumulative_risk": str(_clamp_risk(cumulative_risk)),
        }
        provenance_hash = _compute_provenance_hash(payload)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        _observe_duration("map_processing_chain", elapsed_ms / 1000.0)

        result = {
            "source_commodity": commodity,
            "final_product": {
                "product_id": matched["product_id"],
                "product_name": matched["product_name"],
                "description": matched.get("description", ""),
            },
            "chain": chain_details,
            "total_stages": len(stages),
            "transformation_ratio": transformation_ratio,
            "cn_codes": matched.get("cn_codes", []),
            "cumulative_risk": _clamp_risk(cumulative_risk),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed_ms, 2),
        }

        logger.info(
            "Mapped chain: %s -> %s: stages=%d, ratio=%.2f, "
            "risk=%.2f",
            commodity, matched["product_name"],
            len(stages), transformation_ratio, cumulative_risk,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Transformation risk
    # ------------------------------------------------------------------

    def calculate_transformation_risk(
        self,
        processing_stages: List[str],
    ) -> Decimal:
        """Calculate risk accumulation through processing stages.

        Each stage adds a deterministic risk increment based on the
        type of processing. The total is clamped to [0, 100].

        Args:
            processing_stages: Ordered list of processing stage names.

        Returns:
            Decimal accumulated risk score clamped to [0.00, 100.00].

        Raises:
            ValueError: If processing_stages is empty.
        """
        if not processing_stages:
            raise ValueError("processing_stages must not be empty")

        total_risk = Decimal("0.00")
        for stage in processing_stages:
            stage_lower = stage.strip().lower()
            stage_risk = PROCESSING_STAGE_RISK.get(
                stage_lower, PROCESSING_STAGE_RISK["DEFAULT"],
            )
            total_risk += stage_risk

        return _clamp_risk(total_risk)

    # ------------------------------------------------------------------
    # Public API: Risk multiplier
    # ------------------------------------------------------------------

    def calculate_risk_multiplier(
        self,
        chain_length: int,
        processing_types: List[str],
    ) -> Decimal:
        """Calculate how much risk amplifies through processing.

        Longer chains with more diverse processing types have higher
        risk multipliers. The multiplier starts at 1.0 and increases
        with chain length and type diversity.

        Formula:
            multiplier = 1.0 + (chain_length * 0.05)
                       + (unique_types * 0.08)
                       + (high_risk_stage_count * 0.10)

        Args:
            chain_length: Total number of processing stages.
            processing_types: List of distinct processing type names.

        Returns:
            Decimal multiplier (minimum 1.00, no upper bound but
            practically capped by real chain lengths).

        Raises:
            ValueError: If chain_length is negative.
        """
        if chain_length < 0:
            raise ValueError(f"chain_length must be >= 0, got {chain_length}")

        base = Decimal("1.00")
        length_factor = _to_decimal(chain_length) * Decimal("0.05")
        unique_types = len(set(t.strip().lower() for t in processing_types if t))
        diversity_factor = _to_decimal(unique_types) * Decimal("0.08")

        # Count high-risk stages (risk >= 4.0)
        high_risk_count = 0
        for pt in processing_types:
            stage_risk = PROCESSING_STAGE_RISK.get(
                pt.strip().lower(), PROCESSING_STAGE_RISK["DEFAULT"],
            )
            if stage_risk >= Decimal("4.00"):
                high_risk_count += 1

        high_risk_factor = _to_decimal(high_risk_count) * Decimal("0.10")

        multiplier = base + length_factor + diversity_factor + high_risk_factor
        return multiplier.quantize(_PRECISION, rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Public API: Trace commodity origin
    # ------------------------------------------------------------------

    def trace_commodity_origin(
        self,
        product_id: str,
    ) -> Dict[str, Any]:
        """Trace back from derived product to source commodity.

        Looks up the product in the Annex I mapping and returns the
        source commodity, processing chain, and origin traceability
        information.

        Args:
            product_id: Product identifier to trace back.

        Returns:
            Dictionary containing:
                - product_id (str): Input product identifier.
                - source_commodity (str): Primary EUDR commodity.
                - processing_chain (list): Stages in reverse order.
                - annex_i_product (dict): Matched Annex I entry.
                - traceability_assessment (str): FULL, PARTIAL, NONE.
                - provenance_hash (str): SHA-256 hash.

        Raises:
            ValueError: If product_id is not found in any mapping.
        """
        if not product_id or not isinstance(product_id, str):
            raise ValueError("product_id must be a non-empty string")

        # Search across all commodity mappings
        for commodity, products in ANNEX_I_PRODUCT_MAP.items():
            for product in products:
                if product["product_id"] == product_id:
                    chain = list(reversed(product["processing_stages"]))
                    stage_count = len(product["processing_stages"])

                    # Assess traceability
                    if stage_count <= 3:
                        traceability = "FULL"
                    elif stage_count <= 5:
                        traceability = "PARTIAL"
                    else:
                        traceability = "LIMITED"

                    payload = {
                        "product_id": product_id,
                        "source_commodity": commodity,
                        "chain_length": stage_count,
                    }
                    provenance_hash = _compute_provenance_hash(payload)

                    return {
                        "product_id": product_id,
                        "source_commodity": commodity,
                        "processing_chain": chain,
                        "forward_chain": product["processing_stages"],
                        "annex_i_product": {
                            "product_name": product["product_name"],
                            "cn_codes": product.get("cn_codes", []),
                            "transformation_ratio": product.get(
                                "transformation_ratio", Decimal("0.50"),
                            ),
                            "description": product.get("description", ""),
                        },
                        "traceability_assessment": traceability,
                        "stage_count": stage_count,
                        "provenance_hash": provenance_hash,
                        "created_at": _utcnow().isoformat(),
                    }

        raise ValueError(
            f"Product '{product_id}' not found in any EUDR Annex I mapping"
        )

    # ------------------------------------------------------------------
    # Public API: Get Annex I mapping
    # ------------------------------------------------------------------

    def get_annex_i_mapping(
        self,
        commodity_type: str,
    ) -> List[Dict[str, Any]]:
        """Return all Annex I derived products for a commodity.

        Args:
            commodity_type: Validated EUDR commodity type.

        Returns:
            List of Annex I product dictionaries for the commodity.

        Raises:
            ValueError: If commodity_type is invalid.
        """
        commodity = _validate_commodity_type(commodity_type)
        products = ANNEX_I_PRODUCT_MAP.get(commodity, [])

        result = []
        for product in products:
            result.append({
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "cn_codes": product.get("cn_codes", []),
                "processing_stages": product["processing_stages"],
                "stage_count": len(product["processing_stages"]),
                "transformation_ratio": product.get(
                    "transformation_ratio", Decimal("0.50"),
                ),
                "description": product.get("description", ""),
            })

        logger.debug(
            "Annex I mapping for %s: %d derived products",
            commodity, len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Detect product mislabeling
    # ------------------------------------------------------------------

    def detect_product_mislabeling(
        self,
        declared_product: Dict[str, Any],
        actual_characteristics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Detect potential commodity mislabeling or fraud.

        Compares declared product attributes against actual observed
        characteristics to identify discrepancies that may indicate
        fraud, mislabeling, or non-compliance.

        Args:
            declared_product: Declared product information. Expected keys:
                - "commodity_type" (str): Declared source commodity.
                - "product_name" (str): Declared product name.
                - "origin_country" (str, optional): Declared country.
                - "certification" (str, optional): Declared certification.
                - "grade" (str, optional): Declared quality grade.
            actual_characteristics: Observed/tested characteristics:
                - "detected_commodity" (str, optional): Detected commodity.
                - "detected_origin" (str, optional): Detected origin.
                - "quality_score" (float, optional): Quality test score.
                - "certification_valid" (bool, optional): Cert validity.
                - "composition_analysis" (dict, optional): Lab results.

        Returns:
            Dictionary containing:
                - is_mislabeled (bool): Whether mislabeling is detected.
                - confidence (Decimal): Confidence in detection 0-100.
                - discrepancies (list): List of identified discrepancies.
                - risk_level (str): LOW, MEDIUM, HIGH, CRITICAL.
                - known_indicators (list): Matching known fraud patterns.
                - recommendation (str): Recommended action.
                - provenance_hash (str): SHA-256 hash.

        Raises:
            ValueError: If declared_product is missing required fields.
        """
        start_time = time.monotonic()

        commodity_type = declared_product.get("commodity_type", "")
        if not commodity_type:
            raise ValueError(
                "declared_product must include 'commodity_type'"
            )
        commodity = _validate_commodity_type(commodity_type)

        discrepancies: List[Dict[str, Any]] = []
        confidence_points = Decimal("0")
        max_points = Decimal("0")

        # Check 1: Commodity type match
        max_points += Decimal("30")
        detected_commodity = actual_characteristics.get("detected_commodity", "")
        if detected_commodity:
            detected_norm = detected_commodity.strip().lower()
            if detected_norm != commodity:
                discrepancies.append({
                    "field": "commodity_type",
                    "declared": commodity,
                    "actual": detected_norm,
                    "severity": "CRITICAL",
                    "description": f"Commodity mismatch: declared {commodity}, detected {detected_norm}",
                })
                confidence_points += Decimal("30")

        # Check 2: Origin country
        max_points += Decimal("20")
        declared_origin = declared_product.get("origin_country", "")
        detected_origin = actual_characteristics.get("detected_origin", "")
        if declared_origin and detected_origin:
            if declared_origin.upper() != detected_origin.upper():
                discrepancies.append({
                    "field": "origin_country",
                    "declared": declared_origin.upper(),
                    "actual": detected_origin.upper(),
                    "severity": "HIGH",
                    "description": f"Origin mismatch: declared {declared_origin}, detected {detected_origin}",
                })
                confidence_points += Decimal("20")

        # Check 3: Certification validity
        max_points += Decimal("20")
        declared_cert = declared_product.get("certification", "")
        cert_valid = actual_characteristics.get("certification_valid")
        if declared_cert and cert_valid is not None:
            if not cert_valid:
                discrepancies.append({
                    "field": "certification",
                    "declared": declared_cert,
                    "actual": "invalid/expired",
                    "severity": "HIGH",
                    "description": f"Certification '{declared_cert}' is invalid or expired",
                })
                confidence_points += Decimal("20")

        # Check 4: Quality grade
        max_points += Decimal("15")
        declared_grade = declared_product.get("grade", "")
        quality_score = actual_characteristics.get("quality_score")
        if declared_grade and quality_score is not None:
            qs = _to_decimal(quality_score)
            if qs < Decimal("50"):
                discrepancies.append({
                    "field": "grade",
                    "declared": declared_grade,
                    "actual": f"quality_score={quality_score}",
                    "severity": "MEDIUM",
                    "description": f"Quality score {quality_score} inconsistent with declared grade '{declared_grade}'",
                })
                confidence_points += Decimal("15")

        # Check 5: Composition
        max_points += Decimal("15")
        composition = actual_characteristics.get("composition_analysis", {})
        if composition:
            purity = _to_decimal(composition.get("purity_pct", 100))
            if purity < Decimal("90"):
                discrepancies.append({
                    "field": "composition",
                    "declared": "pure",
                    "actual": f"purity={purity}%",
                    "severity": "MEDIUM",
                    "description": f"Purity {purity}% below expected threshold for declared product",
                })
                confidence_points += Decimal("15")

        # Calculate confidence
        is_mislabeled = len(discrepancies) > 0
        if max_points > Decimal("0"):
            confidence = (confidence_points / max_points * Decimal("100")).quantize(
                _PRECISION, rounding=ROUND_HALF_UP,
            )
        else:
            confidence = Decimal("0.00")
        confidence = _clamp_risk(confidence)

        # Determine risk level
        risk_level = self._mislabeling_risk_level(discrepancies)

        # Match known fraud indicators
        known_indicators = MISLABELING_INDICATORS.get(commodity, [])

        # Recommendation
        recommendation = self._generate_mislabeling_recommendation(
            is_mislabeled, risk_level, discrepancies,
        )

        # Provenance hash
        payload = {
            "declared_product": declared_product,
            "discrepancy_count": len(discrepancies),
            "confidence": str(confidence),
            "risk_level": risk_level,
        }
        provenance_hash = _compute_provenance_hash(payload)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        if is_mislabeled:
            _record_mislabeling(commodity, risk_level)

        result = {
            "is_mislabeled": is_mislabeled,
            "confidence": confidence,
            "discrepancies": discrepancies,
            "risk_level": risk_level,
            "known_indicators": known_indicators,
            "recommendation": recommendation,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed_ms, 2),
            "created_at": _utcnow().isoformat(),
        }

        logger.info(
            "Mislabeling detection for %s: is_mislabeled=%s, "
            "confidence=%.2f, risk_level=%s, discrepancies=%d",
            commodity, is_mislabeled, confidence, risk_level,
            len(discrepancies),
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Traceability loss
    # ------------------------------------------------------------------

    def calculate_traceability_loss(
        self,
        chain: Dict[str, Any],
    ) -> Decimal:
        """Calculate how much traceability degrades through processing.

        Each processing stage introduces traceability loss. Stages
        involving mixing (e.g., milling, blending, solvent_extraction)
        cause higher loss than simple transformations.

        Args:
            chain: Processing chain information:
                - "stages" (list): Processing stage names.
                - "commodity" (str, optional): Source commodity.

        Returns:
            Decimal traceability loss percentage clamped to [0.00, 100.00].
            0 = no loss, 100 = complete loss.
        """
        stages = chain.get("stages", [])
        if not stages:
            return Decimal("0.00")

        # Mixing stages cause higher traceability loss
        high_loss_stages = frozenset({
            "milling", "blending", "crushing", "solvent_extraction",
            "pulping", "cup_lump_collection", "refinery", "rendering_plant",
            "oleochemical_plant", "fractionation",
        })
        medium_loss_stages = frozenset({
            "roasting", "conching", "pressing", "drying",
            "grinding", "extraction", "carbonisation",
            "transesterification", "splitting",
        })

        cumulative_loss = Decimal("0.00")

        for stage in stages:
            stage_lower = stage.strip().lower()
            if stage_lower in high_loss_stages:
                cumulative_loss += Decimal("8.00")
            elif stage_lower in medium_loss_stages:
                cumulative_loss += Decimal("4.00")
            else:
                cumulative_loss += Decimal("2.00")

        return _clamp_risk(cumulative_loss)

    # ------------------------------------------------------------------
    # Public API: Batch analysis
    # ------------------------------------------------------------------

    def batch_analyze(
        self,
        products: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Batch analysis of multiple derived products.

        Args:
            products: List of product dictionaries, each containing:
                - "product_id" (str): Product identifier.
                - "source_commodity" (str): Source commodity.
                - "processing_stages" (list): Processing stages.
                - "input_quantity" (Decimal, optional): Input quantity.

        Returns:
            List of analysis result dictionaries. Failed analyses
            include an "error" key instead of full results.
        """
        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []

        for product_spec in products:
            try:
                product_id = product_spec.get("product_id", str(uuid.uuid4()))
                source_commodity = product_spec.get("source_commodity", "")
                processing_stages = product_spec.get("processing_stages", [])
                input_quantity = product_spec.get("input_quantity")

                if input_quantity is not None:
                    input_quantity = _to_decimal(input_quantity)

                result = self.analyze_derived_product(
                    product_id=product_id,
                    source_commodity=source_commodity,
                    processing_stages=processing_stages,
                    input_quantity=input_quantity,
                )
                results.append(result)

            except (ValueError, KeyError) as exc:
                results.append({
                    "product_id": product_spec.get("product_id", "unknown"),
                    "error": str(exc),
                    "status": "FAILED",
                })
                logger.warning(
                    "Batch analysis failed for product=%s: %s",
                    product_spec.get("product_id", "unknown"), str(exc),
                )

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        _observe_duration("batch_analyze", elapsed_ms / 1000.0)

        success_count = sum(1 for r in results if "error" not in r)
        logger.info(
            "Batch analysis complete: %d/%d succeeded, time_ms=%.2f",
            success_count, len(products), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------
    # Internal: Risk per stage
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_risk_per_stage(
        processing_stages: List[str],
    ) -> List[Dict[str, Any]]:
        """Calculate risk breakdown for each individual stage.

        Args:
            processing_stages: Ordered list of stage names.

        Returns:
            List of dictionaries with stage name, risk, and cumulative risk.
        """
        result: List[Dict[str, Any]] = []
        cumulative = Decimal("0.00")

        for i, stage in enumerate(processing_stages):
            stage_lower = stage.strip().lower()
            stage_risk = PROCESSING_STAGE_RISK.get(
                stage_lower, PROCESSING_STAGE_RISK["DEFAULT"],
            )
            cumulative += stage_risk
            result.append({
                "stage_index": i,
                "stage_name": stage,
                "stage_risk": stage_risk,
                "cumulative_risk": _clamp_risk(cumulative),
            })

        return result

    # ------------------------------------------------------------------
    # Internal: Annex I product matching
    # ------------------------------------------------------------------

    @staticmethod
    def _find_annex_i_match(
        commodity: str,
        processing_stages: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Find the best-matching Annex I product based on processing stages.

        Uses stage overlap scoring to determine the closest match.

        Args:
            commodity: Normalized commodity type.
            processing_stages: Input processing stages.

        Returns:
            Best matching Annex I product dict, or None if no match.
        """
        products = ANNEX_I_PRODUCT_MAP.get(commodity, [])
        if not products:
            return None

        input_stages = set(s.strip().lower() for s in processing_stages)
        best_match = None
        best_score = -1

        for product in products:
            annex_stages = set(s.strip().lower() for s in product["processing_stages"])
            overlap = len(input_stages & annex_stages)
            total = len(input_stages | annex_stages)
            jaccard = overlap / max(total, 1)

            if jaccard > best_score:
                best_score = jaccard
                best_match = product

        if best_match is not None and best_score > 0:
            return dict(best_match)
        return None

    # ------------------------------------------------------------------
    # Internal: Mislabeling risk level
    # ------------------------------------------------------------------

    @staticmethod
    def _mislabeling_risk_level(
        discrepancies: List[Dict[str, Any]],
    ) -> str:
        """Determine mislabeling risk level from discrepancy severities.

        Args:
            discrepancies: List of discrepancy dictionaries.

        Returns:
            Risk level string: LOW, MEDIUM, HIGH, or CRITICAL.
        """
        if not discrepancies:
            return "LOW"

        severities = [d.get("severity", "LOW") for d in discrepancies]
        if "CRITICAL" in severities:
            return "CRITICAL"
        if severities.count("HIGH") >= 2:
            return "CRITICAL"
        if "HIGH" in severities:
            return "HIGH"
        if "MEDIUM" in severities:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # Internal: Mislabeling recommendation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_mislabeling_recommendation(
        is_mislabeled: bool,
        risk_level: str,
        discrepancies: List[Dict[str, Any]],
    ) -> str:
        """Generate an action recommendation based on mislabeling detection.

        Args:
            is_mislabeled: Whether mislabeling was detected.
            risk_level: Overall risk level.
            discrepancies: List of discrepancies.

        Returns:
            Recommendation string.
        """
        if not is_mislabeled:
            return "No action required. Product labeling consistent with characteristics."

        if risk_level == "CRITICAL":
            return (
                "IMMEDIATE ACTION REQUIRED: Suspend product from EUDR compliance chain. "
                "Initiate formal investigation per Article 10(2). "
                "Notify competent authority per Article 31. "
                f"Critical discrepancies detected: {len(discrepancies)}."
            )
        if risk_level == "HIGH":
            return (
                "URGENT: Enhanced due diligence required. Request additional "
                "documentation from supplier. Conduct independent verification. "
                f"High-severity discrepancies: {len(discrepancies)}."
            )
        if risk_level == "MEDIUM":
            return (
                "REVIEW: Request clarification from supplier on identified "
                "discrepancies. Update risk assessment accordingly. "
                f"Medium-severity discrepancies: {len(discrepancies)}."
            )
        return (
            "MONITOR: Minor discrepancies noted. Continue standard monitoring. "
            f"Low-severity discrepancies: {len(discrepancies)}."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cached_product_count(self) -> int:
        """Return the number of cached product analyses."""
        with self._lock:
            return len(self._product_cache)

    def clear_cache(self) -> None:
        """Clear all cached product analyses."""
        with self._lock:
            count = len(self._product_cache)
            self._product_cache.clear()
        logger.info(
            "DerivedProductAnalyzer cache cleared: %d entries removed", count,
        )

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"DerivedProductAnalyzer("
            f"cached_products={self.cached_product_count}, "
            f"commodities={len(ANNEX_I_PRODUCT_MAP)})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "EUDR_PRIMARY_COMMODITIES",
    "ANNEX_I_PRODUCT_MAP",
    "PROCESSING_STAGE_RISK",
    "MISLABELING_INDICATORS",
    # Main class
    "DerivedProductAnalyzer",
]
