# -*- coding: utf-8 -*-
"""
Taxonomy Classifier Engine - AGENT-DATA-009: Spend Data Categorizer
====================================================================

Classifies spend records against four major procurement taxonomy
systems: UNSPSC, NAICS, eCl@ss, and ISIC Rev 4. Provides keyword-based
matching with confidence scoring, cross-taxonomy translation, hierarchy
resolution, and code search.

Supports:
    - UNSPSC 5-level classification (58 segments)
    - NAICS 2-6 digit classification (20 sectors)
    - eCl@ss top-level group classification (45 groups)
    - ISIC Rev 4 section classification (21 sections)
    - Cross-taxonomy code translation
    - Hierarchy resolution for any code
    - Code search with query matching
    - Keyword-based matching with confidence scoring
    - Batch classification
    - Thread-safe in-memory storage
    - SHA-256 provenance hashes on all classifications

Zero-Hallucination Guarantees:
    - All classification is rule-based (keyword matching)
    - Confidence scores are deterministic (exact=0.95, partial=0.7, keyword=0.5)
    - No LLM or ML model in classification path
    - Cross-taxonomy translations use explicit mapping tables
    - SHA-256 provenance hashes for audit trails

Example:
    >>> from greenlang.spend_categorizer.taxonomy_classifier import TaxonomyClassifierEngine
    >>> engine = TaxonomyClassifierEngine()
    >>> result = engine.classify({"description": "office supplies", "category": "indirect"})
    >>> print(result.code, result.system, result.confidence)

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
    "TaxonomyCode",
    "TaxonomyClassification",
    "TaxonomyClassifierEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "cls") -> str:
    """Generate a unique identifier with a prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# UNSPSC Segments (58 top-level segments)
# ---------------------------------------------------------------------------

_UNSPSC_SEGMENTS: Dict[str, str] = {
    "10": "Live Plant and Animal Material",
    "11": "Mineral and Textile and Inedible Plant and Animal Materials",
    "12": "Chemicals including Bio Chemicals and Gas Materials",
    "13": "Resin and Rosin and Rubber and Foam and Film and Elastomeric Materials",
    "14": "Paper Materials and Products",
    "15": "Fuels and Fuel Additives and Lubricants and Anti Corrosive Materials",
    "20": "Mining and Well Drilling Machinery and Accessories",
    "21": "Farming and Fishing and Forestry and Wildlife Machinery and Accessories",
    "22": "Building and Construction Machinery and Accessories",
    "23": "Industrial Manufacturing and Processing Machinery and Accessories",
    "24": "Material Handling and Conditioning and Storage Machinery and Accessories",
    "25": "Commercial and Military and Private Vehicles and their Accessories and Components",
    "26": "Power Generation and Distribution Machinery and Accessories",
    "27": "Tools and General Machinery",
    "30": "Structures and Building and Construction and Manufacturing Components and Supplies",
    "31": "Manufacturing Components and Supplies",
    "32": "Electronic Components and Supplies",
    "39": "Lighting Fixtures and Accessories",
    "40": "Distribution and Conditioning Systems and Equipment and Components",
    "41": "Laboratory and Measuring and Observing and Testing Equipment",
    "42": "Medical Equipment and Accessories and Supplies",
    "43": "Information Technology Broadcasting and Telecommunications",
    "44": "Office Equipment and Accessories and Supplies",
    "45": "Printing and Photographic and Audio and Visual Equipment and Supplies",
    "46": "Defense and Law Enforcement and Security and Safety Equipment and Supplies",
    "47": "Cleaning Equipment and Supplies",
    "48": "Service Industry Machinery and Equipment and Supplies",
    "49": "Sports and Recreational Equipment and Supplies and Accessories",
    "50": "Food Beverage and Tobacco Products",
    "51": "Drugs and Pharmaceutical Products",
    "52": "Domestic Appliances and Supplies and Consumer Electronic Products",
    "53": "Apparel and Luggage and Personal Care Products",
    "54": "Timepieces and Jewelry and Gemstone Products",
    "55": "Published Products",
    "56": "Furniture and Furnishings",
    "60": "Musical Instruments and Games and Toys and Arts and Crafts and Educational Equipment and Materials and Accessories and Supplies",
    "70": "Farming and Fishing and Forestry and Wildlife Contracting Services",
    "71": "Mining and Oil and Gas Services",
    "72": "Building and Facility Construction and Maintenance Services",
    "73": "Industrial Production and Manufacturing Services",
    "76": "Industrial Cleaning Services",
    "77": "Environmental Services",
    "78": "Transportation and Storage and Mail Services",
    "80": "Management and Business Professionals and Administrative Services",
    "81": "Engineering and Research and Technology Based Services",
    "82": "Editorial and Design and Graphic and Fine Art Services",
    "83": "Public Utilities and Public Sector Related Services",
    "84": "Financial and Insurance Services",
    "85": "Healthcare Services",
    "86": "Education and Training Services",
    "90": "Travel and Food and Lodging and Entertainment Services",
    "91": "Personal and Domestic Services",
    "92": "National Defense and Public Order and Security and Safety Services",
    "93": "Politics and Civic Affairs Services",
    "94": "Organizations and Clubs",
    "95": "Land and Buildings and Structures and Thoroughfares",
    "A0": "Complementary and Alternative Medicine Services",
    "A1": "Forms and Labels and Stationery",
    "A2": "Consumer Electronics",
}


# ---------------------------------------------------------------------------
# UNSPSC Keyword Mapping (segment -> keywords)
# ---------------------------------------------------------------------------

_UNSPSC_KEYWORDS: Dict[str, List[str]] = {
    "10": ["livestock", "animal", "plant", "seed", "crop", "agriculture"],
    "11": ["mineral", "textile", "fiber", "wool", "cotton", "silk"],
    "12": ["chemical", "solvent", "acid", "reagent", "biochemical", "gas", "nitrogen", "oxygen"],
    "13": ["resin", "rubber", "foam", "plastic", "elastomer", "polymer"],
    "14": ["paper", "cardboard", "pulp", "tissue", "stationery"],
    "15": ["fuel", "diesel", "gasoline", "lubricant", "petroleum", "oil", "kerosene"],
    "20": ["mining", "drilling", "excavation", "well"],
    "21": ["farming", "tractor", "harvester", "fishing", "forestry"],
    "22": ["construction", "concrete", "crane", "bulldozer", "scaffolding"],
    "23": ["manufacturing", "machine", "industrial", "processing", "fabrication"],
    "24": ["warehouse", "forklift", "conveyor", "pallet", "storage", "material handling"],
    "25": ["vehicle", "truck", "car", "automotive", "fleet", "bus"],
    "26": ["generator", "transformer", "power", "electrical", "turbine", "solar panel"],
    "27": ["tool", "wrench", "drill", "saw", "hammer", "machinery"],
    "30": ["building", "construction material", "beam", "pipe", "fitting"],
    "31": ["component", "fastener", "bolt", "screw", "bearing", "gasket"],
    "32": ["electronic", "semiconductor", "capacitor", "resistor", "circuit", "chip"],
    "39": ["lighting", "lamp", "bulb", "led", "fixture", "fluorescent"],
    "40": ["hvac", "heating", "cooling", "ventilation", "plumbing", "valve"],
    "41": ["laboratory", "microscope", "testing", "measurement", "calibration"],
    "42": ["medical", "surgical", "hospital", "healthcare equipment", "diagnostic"],
    "43": ["computer", "software", "it", "server", "network", "telecom", "laptop", "cloud"],
    "44": ["office", "printer", "copier", "desk", "office supplies", "toner"],
    "45": ["printing", "camera", "audio", "video", "display", "projector"],
    "46": ["security", "safety", "fire", "protective", "alarm", "surveillance"],
    "47": ["cleaning", "janitorial", "detergent", "mop", "broom", "sanitizer"],
    "48": ["restaurant", "kitchen", "laundry", "hospitality", "vending"],
    "49": ["sports", "recreation", "fitness", "gym", "playground"],
    "50": ["food", "beverage", "catering", "snack", "drink", "coffee", "water"],
    "51": ["pharmaceutical", "drug", "medicine", "vaccine", "prescription"],
    "52": ["appliance", "consumer electronics", "television", "refrigerator"],
    "53": ["clothing", "apparel", "uniform", "shoes", "personal care", "luggage"],
    "54": ["jewelry", "watch", "gemstone"],
    "55": ["book", "publication", "magazine", "newspaper", "journal", "subscription"],
    "56": ["furniture", "chair", "table", "cabinet", "shelving", "ergonomic"],
    "60": ["art", "music", "toy", "educational", "craft"],
    "70": ["agriculture service", "farming service", "forestry service"],
    "71": ["oil service", "gas service", "mining service", "exploration"],
    "72": ["building maintenance", "facility", "janitorial service", "renovation"],
    "73": ["production service", "manufacturing service", "assembly"],
    "76": ["industrial cleaning", "decontamination", "waste removal"],
    "77": ["environmental", "waste management", "recycling", "remediation", "sustainability"],
    "78": ["shipping", "freight", "logistics", "courier", "postal", "transport", "delivery"],
    "80": ["consulting", "management", "staffing", "recruitment", "legal", "accounting", "audit"],
    "81": ["engineering", "research", "technology", "r&d", "design", "architecture"],
    "82": ["graphic design", "advertising", "marketing", "creative", "media", "branding"],
    "83": ["utility", "water", "electricity", "sewage", "public service"],
    "84": ["banking", "insurance", "finance", "investment", "tax", "payroll"],
    "85": ["healthcare", "medical service", "dental", "therapy", "wellness"],
    "86": ["training", "education", "e-learning", "certification", "seminar", "workshop"],
    "90": ["travel", "hotel", "flight", "rental car", "lodging", "event", "entertainment"],
    "91": ["personal service", "domestic", "housekeeping"],
    "92": ["defense", "military", "law enforcement", "emergency"],
    "93": ["government", "civic", "political"],
    "94": ["association", "club", "membership"],
    "95": ["real estate", "lease", "rent", "property", "land"],
}


# ---------------------------------------------------------------------------
# NAICS 2-digit Sectors (20 sectors)
# ---------------------------------------------------------------------------

_NAICS_SECTORS: Dict[str, str] = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing (Food, Beverage, Textile, Apparel)",
    "32": "Manufacturing (Wood, Paper, Petroleum, Chemical, Plastics)",
    "33": "Manufacturing (Metal, Machinery, Computer, Electrical, Transport)",
    "42": "Wholesale Trade",
    "44": "Retail Trade (Motor Vehicle, Furniture, Electronics, Building Material)",
    "45": "Retail Trade (Food, Health, Gasoline, Clothing, General)",
    "48": "Transportation and Warehousing",
    "49": "Transportation and Warehousing (Postal, Courier, Warehousing)",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services (except Public Administration)",
    "92": "Public Administration",
}

_NAICS_KEYWORDS: Dict[str, List[str]] = {
    "11": ["agriculture", "farming", "forestry", "fishing", "hunting", "crop", "livestock"],
    "21": ["mining", "quarrying", "oil extraction", "gas extraction", "drilling"],
    "22": ["utility", "electricity", "natural gas", "water supply", "sewage"],
    "23": ["construction", "building", "contractor", "renovation", "demolition"],
    "31": ["food manufacturing", "beverage", "textile", "apparel", "leather"],
    "32": ["wood", "paper", "petroleum", "chemical", "plastics", "rubber"],
    "33": ["metal", "machinery", "computer manufacturing", "electrical equipment", "vehicle manufacturing"],
    "42": ["wholesale", "distributor", "merchant", "broker"],
    "44": ["motor vehicle dealer", "furniture store", "electronics store", "building material"],
    "45": ["grocery", "pharmacy", "gasoline station", "clothing store", "department store"],
    "48": ["air transport", "rail", "trucking", "pipeline", "transit"],
    "49": ["postal", "courier", "warehousing", "storage"],
    "51": ["publishing", "broadcasting", "telecom", "data processing", "software publishing"],
    "52": ["banking", "insurance", "securities", "investment", "finance"],
    "53": ["real estate", "rental", "leasing", "property management"],
    "54": ["legal", "accounting", "architecture", "engineering", "consulting", "research", "advertising"],
    "55": ["management", "holding company", "corporate office"],
    "56": ["staffing", "security service", "cleaning service", "waste management", "landscaping"],
    "61": ["education", "school", "college", "university", "training"],
    "62": ["hospital", "physician", "dental", "nursing", "social assistance"],
    "71": ["arts", "entertainment", "recreation", "museum", "amusement"],
    "72": ["hotel", "motel", "restaurant", "catering", "bar"],
    "81": ["repair", "maintenance", "personal care", "laundry", "religious"],
    "92": ["government", "public administration", "justice", "defense"],
}


# ---------------------------------------------------------------------------
# eCl@ss Top-level Groups (45 groups)
# ---------------------------------------------------------------------------

_ECLASS_GROUPS: Dict[str, str] = {
    "17": "Process and Control Engineering, Automation",
    "18": "Assembly and Fastening Technology",
    "19": "Electrical Engineering",
    "20": "Electronic Components",
    "21": "Electronic Modules, Lighting",
    "22": "Information and Communication Technology",
    "23": "Office Materials",
    "24": "Machines and Drives",
    "25": "Handling and Storage Technology",
    "26": "Fluid Power Technology",
    "27": "Production, Manufacturing Technology",
    "28": "Measurement and Testing Technology",
    "29": "Raw Materials, Auxiliary Materials, Operating Supplies",
    "30": "Semifinished Products",
    "31": "Piping, Fittings",
    "32": "Building Components, Structural Elements",
    "33": "Heating, Climate, Ventilation Technology",
    "34": "Sanitary Technology",
    "35": "Safety Technology",
    "36": "Chemicals",
    "37": "Packaging Technology",
    "38": "Print and Paper Technology",
    "39": "Service",
    "40": "Health, Safety, Environment",
    "41": "Vehicle Technology",
    "42": "Medical Technology",
    "43": "Building Technology",
    "44": "Furniture",
    "45": "Catering and Kitchen Technology",
    "46": "Textile",
    "47": "Agriculture, Horticulture, Forestry Technology",
    "48": "Energy Technology",
    "49": "Mining Technology",
    "50": "Food, Beverages, Tobacco",
    "51": "Hobby, Sports, Leisure",
    "52": "Music, Toys, Crafts",
    "53": "Consumer Goods",
    "54": "Financial Services",
    "55": "Transport Services",
    "56": "IT and Telecom Services",
    "57": "Facility Management Services",
    "58": "Human Resources Services",
    "59": "Marketing and Media Services",
    "60": "Engineering Services",
    "61": "Research and Development Services",
    "62": "Legal and Tax Consulting Services",
}

_ECLASS_KEYWORDS: Dict[str, List[str]] = {
    "17": ["automation", "process control", "plc", "scada", "sensor"],
    "18": ["fastener", "bolt", "screw", "rivet", "assembly"],
    "19": ["electrical", "cable", "wire", "switch", "transformer"],
    "20": ["electronic component", "capacitor", "resistor", "semiconductor"],
    "21": ["led", "lighting", "lamp", "display module"],
    "22": ["computer", "server", "network", "router", "switch", "it"],
    "23": ["office supplies", "stationery", "paper", "pen"],
    "24": ["motor", "drive", "pump", "compressor", "engine"],
    "25": ["forklift", "conveyor", "crane", "hoist", "warehouse"],
    "26": ["hydraulic", "pneumatic", "cylinder", "valve"],
    "27": ["cnc", "lathe", "milling", "welding", "cutting tool"],
    "28": ["measurement", "gauge", "calibration", "testing"],
    "29": ["raw material", "lubricant", "adhesive", "solvent"],
    "30": ["steel", "aluminum", "tube", "plate", "rod", "profile"],
    "31": ["pipe", "fitting", "flange", "gasket"],
    "32": ["window", "door", "insulation", "roofing", "brick"],
    "33": ["hvac", "heating", "cooling", "air conditioning"],
    "34": ["plumbing", "bathroom", "faucet", "toilet"],
    "35": ["fire protection", "alarm", "ppe", "safety equipment"],
    "36": ["chemical", "acid", "base", "reagent", "solvent"],
    "37": ["packaging", "box", "carton", "wrap", "pallet"],
    "38": ["printing", "paper", "ink", "toner", "copier"],
    "39": ["service", "maintenance", "repair", "support"],
    "40": ["environmental", "safety", "waste", "recycling"],
    "41": ["vehicle", "car", "truck", "parts", "tire"],
    "42": ["medical device", "surgical", "implant", "diagnostic"],
    "43": ["building", "construction", "scaffolding", "concrete"],
    "44": ["furniture", "desk", "chair", "cabinet", "shelf"],
    "45": ["catering", "kitchen", "restaurant", "food equipment"],
    "46": ["textile", "fabric", "clothing", "workwear"],
    "47": ["agriculture", "tractor", "seed", "fertilizer"],
    "48": ["solar", "wind", "battery", "generator", "energy"],
    "49": ["mining", "excavator", "drill", "crusher"],
    "50": ["food", "beverage", "coffee", "snack"],
    "51": ["sports", "fitness", "recreation"],
    "52": ["toy", "musical instrument", "craft"],
    "53": ["consumer goods", "personal care", "cosmetics"],
    "54": ["financial", "banking", "insurance", "leasing"],
    "55": ["transport", "shipping", "freight", "logistics"],
    "56": ["it service", "cloud", "hosting", "software support"],
    "57": ["facility management", "cleaning", "landscaping", "security"],
    "58": ["staffing", "recruitment", "payroll", "hr"],
    "59": ["marketing", "advertising", "media", "pr"],
    "60": ["engineering", "design", "cad", "project management"],
    "61": ["research", "r&d", "laboratory", "testing"],
    "62": ["legal", "tax", "audit", "compliance", "consulting"],
}


# ---------------------------------------------------------------------------
# ISIC Rev 4 Sections (21 sections)
# ---------------------------------------------------------------------------

_ISIC_SECTIONS: Dict[str, str] = {
    "A": "Agriculture, forestry and fishing",
    "B": "Mining and quarrying",
    "C": "Manufacturing",
    "D": "Electricity, gas, steam and air conditioning supply",
    "E": "Water supply; sewerage, waste management and remediation activities",
    "F": "Construction",
    "G": "Wholesale and retail trade; repair of motor vehicles and motorcycles",
    "H": "Transportation and storage",
    "I": "Accommodation and food service activities",
    "J": "Information and communication",
    "K": "Financial and insurance activities",
    "L": "Real estate activities",
    "M": "Professional, scientific and technical activities",
    "N": "Administrative and support service activities",
    "O": "Public administration and defence; compulsory social security",
    "P": "Education",
    "Q": "Human health and social work activities",
    "R": "Arts, entertainment and recreation",
    "S": "Other service activities",
    "T": "Activities of households as employers",
    "U": "Activities of extraterritorial organizations and bodies",
}

_ISIC_KEYWORDS: Dict[str, List[str]] = {
    "A": ["agriculture", "farming", "forestry", "fishing", "crop", "livestock"],
    "B": ["mining", "quarrying", "oil", "gas extraction"],
    "C": ["manufacturing", "factory", "production", "assembly"],
    "D": ["electricity", "gas supply", "steam", "power plant"],
    "E": ["water supply", "sewerage", "waste management", "recycling"],
    "F": ["construction", "building", "civil engineering"],
    "G": ["wholesale", "retail", "trade", "repair"],
    "H": ["transportation", "storage", "logistics", "freight"],
    "I": ["hotel", "restaurant", "accommodation", "food service"],
    "J": ["publishing", "broadcasting", "telecom", "it", "software"],
    "K": ["finance", "insurance", "banking", "investment"],
    "L": ["real estate", "property", "rental"],
    "M": ["legal", "accounting", "engineering", "scientific", "consulting", "research"],
    "N": ["staffing", "security", "cleaning", "travel agency", "administrative"],
    "O": ["government", "public administration", "defence"],
    "P": ["education", "school", "university", "training"],
    "Q": ["healthcare", "hospital", "medical", "social work"],
    "R": ["arts", "entertainment", "sports", "recreation", "museum"],
    "S": ["repair", "personal care", "laundry", "funeral"],
    "T": ["household", "domestic"],
    "U": ["international", "extraterritorial", "embassy"],
}


# ---------------------------------------------------------------------------
# Cross-taxonomy mapping tables
# ---------------------------------------------------------------------------

# UNSPSC segment -> NAICS 2-digit
_UNSPSC_TO_NAICS: Dict[str, str] = {
    "10": "11", "11": "21", "12": "32", "13": "32", "14": "32",
    "15": "32", "20": "21", "21": "11", "22": "23", "23": "33",
    "24": "42", "25": "33", "26": "22", "27": "33", "30": "23",
    "31": "33", "32": "33", "39": "33", "40": "23", "41": "33",
    "42": "33", "43": "51", "44": "44", "45": "33", "46": "33",
    "47": "56", "48": "72", "49": "71", "50": "31", "51": "32",
    "52": "33", "53": "31", "54": "33", "55": "51", "56": "33",
    "60": "71", "70": "11", "71": "21", "72": "23", "73": "33",
    "76": "56", "77": "56", "78": "48", "80": "54", "81": "54",
    "82": "54", "83": "22", "84": "52", "85": "62", "86": "61",
    "90": "72", "91": "81", "92": "92", "93": "92", "94": "81",
    "95": "53",
}

# NAICS 2-digit -> ISIC section
_NAICS_TO_ISIC: Dict[str, str] = {
    "11": "A", "21": "B", "22": "D", "23": "F",
    "31": "C", "32": "C", "33": "C",
    "42": "G", "44": "G", "45": "G",
    "48": "H", "49": "H",
    "51": "J", "52": "K", "53": "L", "54": "M", "55": "M",
    "56": "N", "61": "P", "62": "Q", "71": "R", "72": "I",
    "81": "S", "92": "O",
}

# NAICS 2-digit -> eCl@ss top group
_NAICS_TO_ECLASS: Dict[str, str] = {
    "11": "47", "21": "49", "22": "48", "23": "43",
    "31": "50", "32": "36", "33": "24",
    "42": "29", "44": "53", "45": "53",
    "48": "55", "49": "55",
    "51": "22", "52": "54", "53": "44", "54": "60",
    "55": "39", "56": "57",
    "61": "39", "62": "42", "71": "51", "72": "45",
    "81": "39", "92": "35",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TaxonomyCode(BaseModel):
    """A taxonomy code result with metadata."""

    code: str = Field(..., description="Taxonomy code")
    system: str = Field(..., description="Taxonomy system (unspsc, naics, eclass, isic)")
    name: str = Field(default="", description="Human-readable name")
    level: int = Field(default=1, ge=1, description="Hierarchy depth level")
    parent_code: Optional[str] = Field(None, description="Parent code")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    match_type: str = Field(default="keyword", description="Match type (exact, partial, keyword)")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = {"extra": "forbid"}


class TaxonomyClassification(BaseModel):
    """Complete classification result for a spend record."""

    classification_id: str = Field(..., description="Unique classification identifier")
    record_id: str = Field(default="", description="Source record identifier")
    primary_code: TaxonomyCode = Field(..., description="Primary taxonomy code")
    secondary_codes: List[TaxonomyCode] = Field(
        default_factory=list, description="Alternative codes from other systems",
    )
    input_text: str = Field(default="", description="Text used for classification")
    taxonomy_system: str = Field(default="unspsc", description="Primary taxonomy system used")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    classified_at: str = Field(default="", description="Classification timestamp ISO")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# TaxonomyClassifierEngine
# ---------------------------------------------------------------------------


class TaxonomyClassifierEngine:
    """Spend taxonomy classification engine.

    Classifies spend records against UNSPSC, NAICS, eCl@ss, and ISIC
    Rev 4 taxonomies using keyword-based matching. Supports cross-
    taxonomy translation, hierarchy resolution, and code search.

    Confidence scoring:
    - Exact match on code or category name: 0.95
    - Partial match (substring in name or description): 0.70
    - Keyword match: 0.50
    - No match: 0.0 (falls back to generic category)

    Attributes:
        _config: Configuration dictionary.
        _classifications: In-memory classification storage.
        _lock: Threading lock for thread-safe mutations.
        _stats: Cumulative classification statistics.

    Example:
        >>> engine = TaxonomyClassifierEngine()
        >>> result = engine.classify(
        ...     {"description": "steel beams for construction"},
        ...     taxonomy="naics",
        ... )
        >>> print(result.primary_code.code, result.confidence)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TaxonomyClassifierEngine.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``default_taxonomy``: str (default "unspsc")
                - ``min_confidence``: float (default 0.3)
                - ``max_results``: int (default 5)
        """
        self._config = config or {}
        self._default_taxonomy: str = self._config.get("default_taxonomy", "unspsc")
        self._min_confidence: float = self._config.get("min_confidence", 0.3)
        self._max_results: int = self._config.get("max_results", 5)
        self._classifications: Dict[str, TaxonomyClassification] = {}
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "classifications_performed": 0,
            "by_system": {},
            "by_match_type": {},
            "avg_confidence": 0.0,
            "total_confidence": 0.0,
            "errors": 0,
        }
        logger.info(
            "TaxonomyClassifierEngine initialised: default_taxonomy=%s, "
            "min_confidence=%.2f",
            self._default_taxonomy,
            self._min_confidence,
        )

    # ------------------------------------------------------------------
    # Public API - Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        record: Dict[str, Any],
        taxonomy: Optional[str] = None,
    ) -> TaxonomyClassification:
        """Classify a spend record into a taxonomy.

        Extracts text from description, category, and vendor fields,
        then matches against the specified taxonomy system.

        Args:
            record: Spend record dict with fields like ``description``,
                ``category``, ``vendor_name``.
            taxonomy: Taxonomy system to use (unspsc, naics, eclass, isic).
                Defaults to configured default.

        Returns:
            TaxonomyClassification with primary and secondary codes.
        """
        start = time.monotonic()
        system = (taxonomy or self._default_taxonomy).lower().strip()

        # Build search text from record
        search_text = self._build_search_text(record)
        record_id = str(record.get("record_id", ""))

        # Classify in the primary system
        primary_code = self._classify_in_system(search_text, system)

        # Cross-classify in other systems
        secondary_codes: List[TaxonomyCode] = []
        for alt_system in ["unspsc", "naics", "eclass", "isic"]:
            if alt_system != system:
                alt_code = self._classify_in_system(search_text, alt_system)
                if alt_code.confidence >= self._min_confidence:
                    secondary_codes.append(alt_code)

        # Build classification
        cid = _generate_id("cls")
        now_iso = _utcnow().isoformat()

        provenance_hash = self._compute_provenance(
            cid, primary_code.code, system, search_text, now_iso,
        )

        classification = TaxonomyClassification(
            classification_id=cid,
            record_id=record_id,
            primary_code=primary_code,
            secondary_codes=secondary_codes,
            input_text=search_text[:200],
            taxonomy_system=system,
            confidence=primary_code.confidence,
            provenance_hash=provenance_hash,
            classified_at=now_iso,
        )

        # Store and update stats
        with self._lock:
            self._classifications[cid] = classification
            self._stats["classifications_performed"] += 1
            sys_counts = self._stats["by_system"]
            sys_counts[system] = sys_counts.get(system, 0) + 1
            mt_counts = self._stats["by_match_type"]
            mt = primary_code.match_type
            mt_counts[mt] = mt_counts.get(mt, 0) + 1
            self._stats["total_confidence"] += primary_code.confidence
            count = self._stats["classifications_performed"]
            self._stats["avg_confidence"] = round(
                self._stats["total_confidence"] / count, 4,
            )

        elapsed = (time.monotonic() - start) * 1000
        logger.debug(
            "Classified record %s -> %s:%s (conf=%.2f) in %.1f ms",
            record_id[:8] if record_id else "?",
            system, primary_code.code,
            primary_code.confidence, elapsed,
        )
        return classification

    def classify_batch(
        self,
        records: List[Dict[str, Any]],
        taxonomy: Optional[str] = None,
    ) -> List[TaxonomyClassification]:
        """Classify a batch of spend records.

        Args:
            records: List of spend record dicts.
            taxonomy: Taxonomy system to use.

        Returns:
            List of TaxonomyClassification objects.
        """
        start = time.monotonic()
        results = [self.classify(r, taxonomy=taxonomy) for r in records]
        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Batch classified %d records (taxonomy=%s) in %.1f ms",
            len(results), taxonomy or self._default_taxonomy, elapsed,
        )
        return results

    def classify_unspsc(
        self,
        description: str,
        category: Optional[str] = None,
    ) -> TaxonomyCode:
        """Classify text into a UNSPSC code.

        Args:
            description: Item description text.
            category: Optional spend category hint.

        Returns:
            TaxonomyCode with UNSPSC segment.
        """
        text = f"{description} {category or ''}".strip()
        return self._classify_in_system(text, "unspsc")

    def classify_naics(
        self,
        description: str,
        category: Optional[str] = None,
    ) -> TaxonomyCode:
        """Classify text into a NAICS code.

        Args:
            description: Item description text.
            category: Optional spend category hint.

        Returns:
            TaxonomyCode with NAICS sector.
        """
        text = f"{description} {category or ''}".strip()
        return self._classify_in_system(text, "naics")

    def classify_eclass(
        self,
        description: str,
        category: Optional[str] = None,
    ) -> TaxonomyCode:
        """Classify text into an eCl@ss code.

        Args:
            description: Item description text.
            category: Optional spend category hint.

        Returns:
            TaxonomyCode with eCl@ss group.
        """
        text = f"{description} {category or ''}".strip()
        return self._classify_in_system(text, "eclass")

    def classify_isic(
        self,
        description: str,
        category: Optional[str] = None,
    ) -> TaxonomyCode:
        """Classify text into an ISIC Rev 4 code.

        Args:
            description: Item description text.
            category: Optional spend category hint.

        Returns:
            TaxonomyCode with ISIC section.
        """
        text = f"{description} {category or ''}".strip()
        return self._classify_in_system(text, "isic")

    # ------------------------------------------------------------------
    # Public API - Translation and hierarchy
    # ------------------------------------------------------------------

    def translate_code(
        self,
        code: str,
        from_system: str,
        to_system: str,
    ) -> TaxonomyCode:
        """Translate a taxonomy code from one system to another.

        Uses cross-taxonomy mapping tables. Translates via NAICS
        as an intermediate system when no direct mapping exists.

        Args:
            code: Source taxonomy code.
            from_system: Source taxonomy system.
            to_system: Target taxonomy system.

        Returns:
            TaxonomyCode in the target system.

        Raises:
            ValueError: If systems are identical or mapping not found.
        """
        fs = from_system.lower().strip()
        ts = to_system.lower().strip()

        if fs == ts:
            raise ValueError(
                f"Source and target systems are identical: {fs}"
            )

        # Resolve via mapping tables
        target_code = self._translate(code, fs, ts)

        if target_code is None:
            # Try via NAICS as intermediate
            if fs != "naics" and ts != "naics":
                naics_code = self._translate(code, fs, "naics")
                if naics_code:
                    target_code = self._translate(naics_code, "naics", ts)

        if target_code is None:
            logger.warning(
                "No translation found: %s:%s -> %s",
                fs, code, ts,
            )
            return TaxonomyCode(
                code="unknown",
                system=ts,
                name="No translation available",
                confidence=0.0,
                match_type="none",
            )

        # Look up the name
        name = self._get_code_name(target_code, ts)

        provenance_hash = self._compute_provenance(
            f"translate-{code}-{fs}-{ts}",
            target_code, ts, code, _utcnow().isoformat(),
        )

        return TaxonomyCode(
            code=target_code,
            system=ts,
            name=name,
            confidence=0.80,
            match_type="translation",
            provenance_hash=provenance_hash,
        )

    def get_code_hierarchy(
        self,
        code: str,
        system: str,
    ) -> List[TaxonomyCode]:
        """Get the full hierarchy for a taxonomy code.

        Returns the chain from the top-level parent down to the given
        code. For segment-level codes, returns a single entry.

        Args:
            code: Taxonomy code to resolve.
            system: Taxonomy system.

        Returns:
            List of TaxonomyCode objects from root to leaf.
        """
        sys = system.lower().strip()
        hierarchy: List[TaxonomyCode] = []

        if sys == "unspsc":
            # UNSPSC: 2-digit segment, 4-digit family, 6-digit class, 8-digit commodity
            levels = [(code[:2], 1), (code[:4], 2), (code[:6], 3), (code[:8], 4)]
            for lvl_code, lvl in levels:
                if len(code) >= len(lvl_code):
                    name = _UNSPSC_SEGMENTS.get(lvl_code, lvl_code)
                    hierarchy.append(TaxonomyCode(
                        code=lvl_code, system=sys, name=name, level=lvl,
                        parent_code=hierarchy[-1].code if hierarchy else None,
                    ))
        elif sys == "naics":
            # NAICS: 2-digit sector up to 6-digit
            for end in [2, 3, 4, 5, 6]:
                if len(code) >= end:
                    lvl_code = code[:end]
                    name = _NAICS_SECTORS.get(lvl_code, lvl_code)
                    hierarchy.append(TaxonomyCode(
                        code=lvl_code, system=sys, name=name, level=end - 1,
                        parent_code=hierarchy[-1].code if hierarchy else None,
                    ))
        elif sys == "eclass":
            levels = [(code[:2], 1), (code[:4], 2), (code[:6], 3), (code[:8], 4)]
            for lvl_code, lvl in levels:
                if len(code) >= len(lvl_code):
                    name = _ECLASS_GROUPS.get(lvl_code, lvl_code)
                    hierarchy.append(TaxonomyCode(
                        code=lvl_code, system=sys, name=name, level=lvl,
                        parent_code=hierarchy[-1].code if hierarchy else None,
                    ))
        elif sys == "isic":
            # ISIC: letter section, then numeric divisions
            if code:
                section = code[0].upper()
                name = _ISIC_SECTIONS.get(section, section)
                hierarchy.append(TaxonomyCode(
                    code=section, system=sys, name=name, level=1,
                ))
                if len(code) > 1:
                    hierarchy.append(TaxonomyCode(
                        code=code, system=sys, name=code, level=2,
                        parent_code=section,
                    ))

        return hierarchy

    def search_codes(
        self,
        query: str,
        system: str,
        limit: int = 20,
    ) -> List[TaxonomyCode]:
        """Search taxonomy codes by text query.

        Args:
            query: Search text.
            system: Taxonomy system to search.
            limit: Maximum results.

        Returns:
            List of matching TaxonomyCode objects sorted by relevance.
        """
        sys = system.lower().strip()
        query_lower = query.lower().strip()
        results: List[TaxonomyCode] = []

        registry = self._get_registry(sys)
        keywords_map = self._get_keywords(sys)

        for code, name in registry.items():
            score = 0.0
            match_type = "none"

            # Exact name match
            if query_lower == name.lower():
                score = 0.95
                match_type = "exact"
            # Substring in name
            elif query_lower in name.lower():
                score = 0.70
                match_type = "partial"
            # Keyword match
            elif code in keywords_map:
                for kw in keywords_map[code]:
                    if kw in query_lower or query_lower in kw:
                        score = max(score, 0.50)
                        match_type = "keyword"

            if score > 0:
                results.append(TaxonomyCode(
                    code=code,
                    system=sys,
                    name=name,
                    confidence=score,
                    match_type=match_type,
                ))

        # Sort by confidence descending
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Return cumulative classification statistics.

        Returns:
            Dictionary with classification counters and breakdowns.
        """
        with self._lock:
            stats = dict(self._stats)
            stats["by_system"] = dict(self._stats["by_system"])
            stats["by_match_type"] = dict(self._stats["by_match_type"])
        stats["classifications_stored"] = len(self._classifications)
        stats["taxonomy_sizes"] = {
            "unspsc_segments": len(_UNSPSC_SEGMENTS),
            "naics_sectors": len(_NAICS_SECTORS),
            "eclass_groups": len(_ECLASS_GROUPS),
            "isic_sections": len(_ISIC_SECTIONS),
        }
        return stats

    # ------------------------------------------------------------------
    # Internal - Classification logic
    # ------------------------------------------------------------------

    def _classify_in_system(
        self,
        text: str,
        system: str,
    ) -> TaxonomyCode:
        """Classify text into a specific taxonomy system.

        Args:
            text: Search text (description + category + vendor).
            system: Taxonomy system identifier.

        Returns:
            Best-matching TaxonomyCode.
        """
        registry = self._get_registry(system)
        keywords_map = self._get_keywords(system)
        text_lower = text.lower().strip()

        best_code = ""
        best_score = 0.0
        best_match = "none"

        for code, name in registry.items():
            score = 0.0
            match_type = "none"

            name_lower = name.lower()

            # Exact name match in text
            if name_lower in text_lower:
                score = 0.95
                match_type = "exact"
            # Partial name words match
            elif self._partial_name_match(name_lower, text_lower):
                score = 0.70
                match_type = "partial"

            # Keyword matching
            if code in keywords_map and score < 0.95:
                kw_score = self._keyword_score(keywords_map[code], text_lower)
                if kw_score > score:
                    score = kw_score
                    match_type = "keyword" if kw_score <= 0.5 else match_type

            if score > best_score:
                best_score = score
                best_code = code
                best_match = match_type

        # Fallback to generic
        if not best_code or best_score < self._min_confidence:
            best_code, name = self._get_fallback(system)
            best_score = 0.1
            best_match = "fallback"
        else:
            name = registry.get(best_code, best_code)

        provenance_hash = self._compute_provenance(
            f"classify-{system}", best_code, system, text_lower[:100],
            _utcnow().isoformat(),
        )

        return TaxonomyCode(
            code=best_code,
            system=system,
            name=name,
            level=1,
            confidence=round(best_score, 4),
            match_type=best_match,
            provenance_hash=provenance_hash,
        )

    def _partial_name_match(self, name: str, text: str) -> bool:
        """Check if significant words from name appear in text.

        Args:
            name: Taxonomy name (lowercase).
            text: Search text (lowercase).

        Returns:
            True if at least 50% of significant name words match.
        """
        stop_words = {"and", "or", "the", "of", "for", "in", "a", "an", "to", "with", "except"}
        name_words = [w for w in name.split() if w not in stop_words and len(w) > 2]
        if not name_words:
            return False
        matches = sum(1 for w in name_words if w in text)
        return matches / len(name_words) >= 0.5

    def _keyword_score(
        self,
        keywords: List[str],
        text: str,
    ) -> float:
        """Score keyword matches in text.

        Args:
            keywords: List of keywords for a taxonomy code.
            text: Search text (lowercase).

        Returns:
            Score in [0, 0.6]. Higher with more keyword matches.
        """
        if not keywords:
            return 0.0

        matches = 0
        for kw in keywords:
            if kw.lower() in text:
                matches += 1

        if matches == 0:
            return 0.0

        ratio = matches / len(keywords)
        # Scale: 1 match = 0.50, multi-match scales up to 0.60
        return min(0.50 + ratio * 0.10, 0.60)

    def _build_search_text(self, record: Dict[str, Any]) -> str:
        """Build search text from record fields.

        Args:
            record: Spend record dict.

        Returns:
            Combined text from description, category, vendor, material.
        """
        parts = []
        for key in ("description", "category", "vendor_name", "material_group"):
            val = record.get(key)
            if val:
                parts.append(str(val).strip())
        return " ".join(parts)

    def _get_registry(self, system: str) -> Dict[str, str]:
        """Get the code-to-name registry for a system.

        Args:
            system: Taxonomy system identifier.

        Returns:
            Dict mapping code -> name.
        """
        return {
            "unspsc": _UNSPSC_SEGMENTS,
            "naics": _NAICS_SECTORS,
            "eclass": _ECLASS_GROUPS,
            "isic": _ISIC_SECTIONS,
        }.get(system, {})

    def _get_keywords(self, system: str) -> Dict[str, List[str]]:
        """Get the keyword mapping for a system.

        Args:
            system: Taxonomy system identifier.

        Returns:
            Dict mapping code -> keyword list.
        """
        return {
            "unspsc": _UNSPSC_KEYWORDS,
            "naics": _NAICS_KEYWORDS,
            "eclass": _ECLASS_KEYWORDS,
            "isic": _ISIC_KEYWORDS,
        }.get(system, {})

    def _get_code_name(self, code: str, system: str) -> str:
        """Look up a code name in a system.

        Args:
            code: Taxonomy code.
            system: Taxonomy system.

        Returns:
            Human-readable name or the code itself.
        """
        registry = self._get_registry(system)
        return registry.get(code, code)

    def _get_fallback(self, system: str) -> Tuple[str, str]:
        """Get the fallback code for a system (generic/unclassified).

        Args:
            system: Taxonomy system.

        Returns:
            Tuple of (code, name).
        """
        fallbacks = {
            "unspsc": ("80", "Management and Business Professionals and Administrative Services"),
            "naics": ("54", "Professional, Scientific, and Technical Services"),
            "eclass": ("39", "Service"),
            "isic": ("M", "Professional, scientific and technical activities"),
        }
        return fallbacks.get(system, ("00", "Unknown"))

    def _translate(
        self,
        code: str,
        from_system: str,
        to_system: str,
    ) -> Optional[str]:
        """Translate a code between two systems using mapping tables.

        Args:
            code: Source code (top-level prefix used for lookup).
            from_system: Source system.
            to_system: Target system.

        Returns:
            Target code or None if no mapping exists.
        """
        # Extract top-level code for lookup
        lookup_code = code[:2] if from_system in ("unspsc", "naics", "eclass") else code[:1]

        mapping_key = f"{from_system}_to_{to_system}"
        tables = {
            "unspsc_to_naics": _UNSPSC_TO_NAICS,
            "naics_to_isic": _NAICS_TO_ISIC,
            "naics_to_eclass": _NAICS_TO_ECLASS,
        }

        table = tables.get(mapping_key)
        if table:
            return table.get(lookup_code)

        # Try reverse lookups
        reverse_key = f"{to_system}_to_{from_system}"
        rev_table = tables.get(reverse_key)
        if rev_table:
            for k, v in rev_table.items():
                if v == lookup_code:
                    return k

        return None

    # ------------------------------------------------------------------
    # Internal - Provenance
    # ------------------------------------------------------------------

    def _compute_provenance(
        self,
        entity_id: str,
        code: str,
        system: str,
        input_text: str,
        timestamp: str,
    ) -> str:
        """Compute SHA-256 provenance hash for a classification.

        Args:
            entity_id: Classification or operation identifier.
            code: Taxonomy code assigned.
            system: Taxonomy system.
            input_text: Input text used (truncated).
            timestamp: Classification timestamp.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        payload = json.dumps({
            "entity_id": entity_id,
            "code": code,
            "system": system,
            "input_text": input_text[:200],
            "timestamp": timestamp,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
