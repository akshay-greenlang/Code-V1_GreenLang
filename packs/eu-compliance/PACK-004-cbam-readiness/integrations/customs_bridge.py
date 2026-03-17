# -*- coding: utf-8 -*-
"""
CustomsBridge - CN Code and Customs Data Integration for CBAM Readiness Pack
=============================================================================

This module provides comprehensive CN code database access, EORI validation,
customs declaration parsing, and country information for the EU Carbon Border
Adjustment Mechanism (CBAM). It serves as the authoritative lookup layer for
all goods-classification and customs-related operations within PACK-004.

The CN code database covers all CBAM Annex I goods categories:
    - Iron and steel (Chapter 72-73)
    - Aluminium (Chapter 76)
    - Cement (Chapter 25)
    - Fertilisers (Chapter 28, 31)
    - Hydrogen (Chapter 28)
    - Electricity (Chapter 27)

Example:
    >>> bridge = CustomsBridge()
    >>> info = bridge.lookup_cn_code("7201 10 11")
    >>> assert info.goods_category == "IRON_AND_STEEL"
    >>> assert info.cbam_applicable is True

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
"""

import hashlib
import logging
import re
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class GoodsCategory(str, Enum):
    """CBAM Annex I goods categories."""
    IRON_AND_STEEL = "IRON_AND_STEEL"
    ALUMINIUM = "ALUMINIUM"
    CEMENT = "CEMENT"
    FERTILISERS = "FERTILISERS"
    HYDROGEN = "HYDROGEN"
    ELECTRICITY = "ELECTRICITY"


# =============================================================================
# Data Models
# =============================================================================


class CNCodeInfo(BaseModel):
    """Detailed information about a Combined Nomenclature (CN) code."""
    cn_code: str = Field(..., description="8-digit CN code (formatted with spaces)")
    cn_code_raw: str = Field(default="", description="CN code without spaces/dots")
    description: str = Field(..., description="Goods description from CN")
    goods_category: GoodsCategory = Field(..., description="CBAM Annex I goods category")
    cbam_applicable: bool = Field(default=True, description="Whether code falls under CBAM")
    unit: str = Field(default="tonnes", description="Default reporting unit")
    notes: str = Field(default="", description="Additional notes or caveats")
    chapter: int = Field(default=0, description="CN chapter number")
    heading: str = Field(default="", description="CN heading code (4 digits)")


class CountryInfo(BaseModel):
    """Information about a country relevant to CBAM."""
    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(..., description="Full country name")
    eu_member: bool = Field(default=False, description="Whether the country is an EU member state")
    efta_member: bool = Field(default=False, description="Whether the country is an EFTA member")
    has_carbon_pricing: bool = Field(default=False, description="Has an effective carbon price")
    carbon_price_eur: float = Field(default=0.0, description="Carbon price in EUR/tCO2e")
    carbon_pricing_scheme: str = Field(default="", description="Name of carbon pricing scheme")
    cbam_exempt: bool = Field(default=False, description="Whether exempted from CBAM")


class CustomsDeclarationItem(BaseModel):
    """A single CBAM-relevant item extracted from a customs declaration."""
    item_id: str = Field(default_factory=lambda: str(uuid4())[:8], description="Item ID")
    cn_code: str = Field(..., description="CN code of the goods")
    description: str = Field(default="", description="Goods description")
    origin_country: str = Field(..., description="Country of origin (ISO alpha-2)")
    net_mass_kg: float = Field(default=0.0, ge=0.0, description="Net mass in kg")
    quantity: float = Field(default=0.0, ge=0.0, description="Quantity in unit of measure")
    unit: str = Field(default="kg", description="Unit of measure")
    customs_value_eur: float = Field(default=0.0, ge=0.0, description="Customs value in EUR")
    cbam_applicable: bool = Field(default=False, description="Falls under CBAM Annex I")
    importer_eori: str = Field(default="", description="Importer EORI number")
    declaration_date: Optional[str] = Field(None, description="Declaration date YYYY-MM-DD")


class CustomsBridgeResult(BaseModel):
    """Result wrapper for customs bridge operations."""
    operation: str = Field(..., description="Operation name")
    success: bool = Field(default=True, description="Whether the operation succeeded")
    data: Any = Field(default=None, description="Operation result data")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# =============================================================================
# CBAM Annex I CN Code Database
# =============================================================================

_CBAM_CN_CODES: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # CEMENT (Chapter 25)
    # -------------------------------------------------------------------------
    "2507 00 80": {
        "description": "Other kaolinic clays (for cement clinker)",
        "category": GoodsCategory.CEMENT,
        "unit": "tonnes",
        "chapter": 25,
    },
    "2523 10 00": {
        "description": "Cement clinkers",
        "category": GoodsCategory.CEMENT,
        "unit": "tonnes",
        "chapter": 25,
    },
    "2523 21 00": {
        "description": "White Portland cement, whether or not artificially coloured",
        "category": GoodsCategory.CEMENT,
        "unit": "tonnes",
        "chapter": 25,
    },
    "2523 29 00": {
        "description": "Other Portland cement",
        "category": GoodsCategory.CEMENT,
        "unit": "tonnes",
        "chapter": 25,
    },
    "2523 30 00": {
        "description": "Aluminous cement",
        "category": GoodsCategory.CEMENT,
        "unit": "tonnes",
        "chapter": 25,
    },
    "2523 90 00": {
        "description": "Other hydraulic cements",
        "category": GoodsCategory.CEMENT,
        "unit": "tonnes",
        "chapter": 25,
    },
    # -------------------------------------------------------------------------
    # ELECTRICITY (Chapter 27)
    # -------------------------------------------------------------------------
    "2716 00 00": {
        "description": "Electrical energy",
        "category": GoodsCategory.ELECTRICITY,
        "unit": "MWh",
        "chapter": 27,
    },
    # -------------------------------------------------------------------------
    # FERTILISERS - Hydrogen precursors (Chapter 28)
    # -------------------------------------------------------------------------
    "2804 10 00": {
        "description": "Hydrogen",
        "category": GoodsCategory.HYDROGEN,
        "unit": "tonnes",
        "chapter": 28,
    },
    "2808 00 00": {
        "description": "Nitric acid; sulphonitric acids",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 28,
    },
    "2814 10 00": {
        "description": "Anhydrous ammonia",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 28,
    },
    "2814 20 00": {
        "description": "Ammonia in aqueous solution",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 28,
    },
    # -------------------------------------------------------------------------
    # FERTILISERS (Chapter 31)
    # -------------------------------------------------------------------------
    "3102 10 10": {
        "description": "Urea, with nitrogen content >45% by weight (dry product)",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 31,
    },
    "3102 10 90": {
        "description": "Other urea including aqueous solution",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 31,
    },
    "3102 30 10": {
        "description": "Ammonium nitrate in aqueous solution",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 31,
    },
    "3102 30 90": {
        "description": "Other ammonium nitrate",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 31,
    },
    "3105 10 00": {
        "description": "Goods in tablets or similar forms or in packages GEW <= 10 kg",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 31,
    },
    "3105 20 10": {
        "description": "Mineral or chemical fertilisers with N, P, K: N content >10%",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 31,
    },
    "3105 20 90": {
        "description": "Other mineral or chemical fertilisers with N, P and K",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 31,
    },
    "3105 51 00": {
        "description": "Mineral or chemical fertilisers with nitrates and phosphates",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 31,
    },
    "3105 59 00": {
        "description": "Other mineral or chemical fertilisers with N and P",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 31,
    },
    "3105 90 00": {
        "description": "Other fertilisers (mixtures not elsewhere specified)",
        "category": GoodsCategory.FERTILISERS,
        "unit": "tonnes",
        "chapter": 31,
    },
    # -------------------------------------------------------------------------
    # IRON AND STEEL (Chapter 72)
    # -------------------------------------------------------------------------
    "7201 10 11": {
        "description": "Non-alloy pig iron, Mn <= 0.5%, Si < 0.5%",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7201 10 19": {
        "description": "Non-alloy pig iron, Mn <= 0.5%, Si >= 0.5%",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7201 10 30": {
        "description": "Non-alloy pig iron, Mn > 0.5% but <= 1%",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7201 10 90": {
        "description": "Other non-alloy pig iron",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7201 20 00": {
        "description": "Alloy pig iron; spiegeleisen",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7202 11 20": {
        "description": "Ferro-manganese, C > 2%, granulometry <= 5 mm, Mn > 65%",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7202 11 80": {
        "description": "Other ferro-manganese, C > 2%",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7203 10 00": {
        "description": "Direct reduced iron in lumps, pellets and similar forms",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7203 90 00": {
        "description": "Other spongy ferrous products (DRI)",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7206 10 00": {
        "description": "Iron ingots",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7206 90 00": {
        "description": "Other iron (primary forms)",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7207 11 14": {
        "description": "Semi-finished products of iron, C < 0.25%, rectangular cross-section",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7207 12 10": {
        "description": "Semi-finished products of iron, C >= 0.25%",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7208 10 00": {
        "description": "Flat-rolled products of iron, hot-rolled, in coils, with patterns",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7209 15 00": {
        "description": "Flat-rolled products of iron, cold-rolled, thickness >= 3 mm",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7210 11 00": {
        "description": "Flat-rolled products of iron, tin-plated, thickness >= 0.5 mm",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7211 13 00": {
        "description": "Flat-rolled products of iron, hot-rolled, not in coils, 4.75+ mm",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7213 10 00": {
        "description": "Bars and rods of iron, hot-rolled, with indentations/ribs",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7214 10 00": {
        "description": "Bars and rods of iron, forged",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7215 10 00": {
        "description": "Bars and rods of free-cutting steel, cold-formed or cold-finished",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7216 10 00": {
        "description": "Angles, shapes and sections of iron, U/I/H sections, hot-rolled",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7217 10 10": {
        "description": "Wire of iron, C < 0.25%, not plated or coated",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7218 10 00": {
        "description": "Stainless steel ingots and other primary forms",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7219 11 00": {
        "description": "Flat-rolled stainless steel, hot-rolled, in coils, thickness > 10 mm",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7224 10 00": {
        "description": "Other alloy steel ingots and other primary forms",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7225 11 00": {
        "description": "Flat-rolled products of silicon-electrical steel, grain-oriented",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    "7226 11 00": {
        "description": "Flat-rolled products of silicon-electrical steel, GO, width < 600 mm",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 72,
    },
    # -------------------------------------------------------------------------
    # IRON AND STEEL - Articles (Chapter 73)
    # -------------------------------------------------------------------------
    "7301 10 00": {
        "description": "Sheet piling of iron or steel",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 73,
    },
    "7302 10 22": {
        "description": "Rails for railways, new, weight >= 36 kg/m",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 73,
    },
    "7303 00 10": {
        "description": "Tubes and pipes of cast iron",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 73,
    },
    "7304 11 00": {
        "description": "Seamless tubes and pipes of stainless steel, line pipe",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 73,
    },
    "7305 11 00": {
        "description": "Line pipe for oil/gas, submerged arc welded, external diameter > 406.4 mm",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 73,
    },
    "7306 11 10": {
        "description": "Welded line pipe of stainless steel for oil/gas",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 73,
    },
    "7318 15 42": {
        "description": "Screws and bolts of stainless steel, tensile strength >= 800 MPa",
        "category": GoodsCategory.IRON_AND_STEEL,
        "unit": "tonnes",
        "chapter": 73,
        "notes": "Selected fasteners under CBAM",
    },
    # -------------------------------------------------------------------------
    # ALUMINIUM (Chapter 76)
    # -------------------------------------------------------------------------
    "7601 10 00": {
        "description": "Unwrought aluminium, not alloyed",
        "category": GoodsCategory.ALUMINIUM,
        "unit": "tonnes",
        "chapter": 76,
    },
    "7601 20 20": {
        "description": "Unwrought aluminium alloys, primary",
        "category": GoodsCategory.ALUMINIUM,
        "unit": "tonnes",
        "chapter": 76,
    },
    "7601 20 80": {
        "description": "Other unwrought aluminium alloys",
        "category": GoodsCategory.ALUMINIUM,
        "unit": "tonnes",
        "chapter": 76,
    },
    "7603 10 00": {
        "description": "Aluminium powders, non-lamellar structure",
        "category": GoodsCategory.ALUMINIUM,
        "unit": "tonnes",
        "chapter": 76,
    },
    "7604 10 10": {
        "description": "Aluminium bars, rods and profiles, non-alloyed",
        "category": GoodsCategory.ALUMINIUM,
        "unit": "tonnes",
        "chapter": 76,
    },
    "7604 21 00": {
        "description": "Hollow profiles of aluminium alloys",
        "category": GoodsCategory.ALUMINIUM,
        "unit": "tonnes",
        "chapter": 76,
    },
    "7604 29 10": {
        "description": "Bars and rods of aluminium alloys",
        "category": GoodsCategory.ALUMINIUM,
        "unit": "tonnes",
        "chapter": 76,
    },
    "7605 11 00": {
        "description": "Aluminium wire, non-alloyed, cross-section > 7 mm",
        "category": GoodsCategory.ALUMINIUM,
        "unit": "tonnes",
        "chapter": 76,
    },
    "7606 11 10": {
        "description": "Aluminium plates/sheets, non-alloyed, rectangular, thickness > 0.2 mm",
        "category": GoodsCategory.ALUMINIUM,
        "unit": "tonnes",
        "chapter": 76,
    },
    "7607 11 10": {
        "description": "Aluminium foil, not backed, rolled, thickness <= 0.021 mm",
        "category": GoodsCategory.ALUMINIUM,
        "unit": "tonnes",
        "chapter": 76,
    },
    "7608 10 00": {
        "description": "Aluminium tubes and pipes, non-alloyed",
        "category": GoodsCategory.ALUMINIUM,
        "unit": "tonnes",
        "chapter": 76,
    },
    "7609 00 00": {
        "description": "Aluminium tube or pipe fittings",
        "category": GoodsCategory.ALUMINIUM,
        "unit": "tonnes",
        "chapter": 76,
    },
}


# =============================================================================
# EU Member State & Country Database
# =============================================================================

_EU_MEMBER_STATES: Dict[str, str] = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia",
    "CY": "Cyprus", "CZ": "Czech Republic", "DK": "Denmark", "EE": "Estonia",
    "FI": "Finland", "FR": "France", "DE": "Germany", "GR": "Greece",
    "HU": "Hungary", "IE": "Ireland", "IT": "Italy", "LV": "Latvia",
    "LT": "Lithuania", "LU": "Luxembourg", "MT": "Malta", "NL": "Netherlands",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SK": "Slovakia",
    "SI": "Slovenia", "ES": "Spain", "SE": "Sweden",
}

_EFTA_MEMBERS: Dict[str, str] = {
    "IS": "Iceland", "LI": "Liechtenstein", "NO": "Norway", "CH": "Switzerland",
}

_CARBON_PRICING_COUNTRIES: Dict[str, Dict[str, Any]] = {
    "GB": {"scheme": "UK ETS", "price_eur": 50.0},
    "CA": {"scheme": "Federal Carbon Tax", "price_eur": 45.0},
    "NZ": {"scheme": "NZ ETS", "price_eur": 35.0},
    "KR": {"scheme": "Korea ETS", "price_eur": 20.0},
    "CN": {"scheme": "China National ETS", "price_eur": 10.0},
    "JP": {"scheme": "Japan Carbon Tax", "price_eur": 3.0},
    "ZA": {"scheme": "South Africa Carbon Tax", "price_eur": 8.0},
    "MX": {"scheme": "Mexico Carbon Tax", "price_eur": 4.0},
    "CO": {"scheme": "Colombia Carbon Tax", "price_eur": 5.0},
    "AR": {"scheme": "Argentina Carbon Tax", "price_eur": 5.5},
    "CL": {"scheme": "Chile Carbon Tax", "price_eur": 5.0},
    "SG": {"scheme": "Singapore Carbon Tax", "price_eur": 4.0},
    "UA": {"scheme": "Ukraine Carbon Tax", "price_eur": 1.0},
    "KZ": {"scheme": "Kazakhstan ETS", "price_eur": 2.0},
}


# =============================================================================
# TARIC to CBAM Mapping (selected common conversions)
# =============================================================================

_TARIC_TO_CBAM: Dict[str, str] = {
    "7201101100": "7201 10 11",
    "7201101900": "7201 10 19",
    "7201103000": "7201 10 30",
    "7201109000": "7201 10 90",
    "7201200000": "7201 20 00",
    "7601100000": "7601 10 00",
    "7601202000": "7601 20 20",
    "7601208000": "7601 20 80",
    "2523100000": "2523 10 00",
    "2523210000": "2523 21 00",
    "2523290000": "2523 29 00",
    "2523300000": "2523 30 00",
    "2523900000": "2523 90 00",
    "2716000000": "2716 00 00",
    "2804100000": "2804 10 00",
    "2814100000": "2814 10 00",
    "2814200000": "2814 20 00",
}


# =============================================================================
# Customs Bridge Implementation
# =============================================================================


class CustomsBridge:
    """CN code and customs data integration for CBAM Readiness Pack.

    Provides lookup, validation, and parsing capabilities for CBAM
    Annex I Combined Nomenclature codes, EORI numbers, country carbon
    pricing information, and customs declarations.

    Attributes:
        config: Optional configuration dictionary
        logger: Module-level logger
        _cn_codes: In-memory CN code database
        _stub_mode: True if running without external dependencies

    Example:
        >>> bridge = CustomsBridge()
        >>> code_info = bridge.lookup_cn_code("7201 10 11")
        >>> assert code_info is not None
        >>> assert code_info.goods_category == GoodsCategory.IRON_AND_STEEL
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the customs bridge.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.logger = logger
        self._cn_codes: Dict[str, Dict[str, Any]] = dict(_CBAM_CN_CODES)
        self._stub_mode = False
        self._initialized_at = datetime.utcnow()

        self.logger.info(
            "CustomsBridge initialized with %d CN codes across %d categories",
            len(self._cn_codes),
            len(set(v["category"] for v in self._cn_codes.values())),
        )

    # -------------------------------------------------------------------------
    # CN Code Lookup
    # -------------------------------------------------------------------------

    def lookup_cn_code(self, code: str) -> Optional[CNCodeInfo]:
        """Look up a CN code and return its full information.

        Args:
            code: The CN code to look up (with or without spaces).

        Returns:
            CNCodeInfo if found, None otherwise.
        """
        normalized = self._normalize_cn_code(code)
        formatted = self._format_cn_code(normalized)

        entry = self._cn_codes.get(formatted)
        if entry is None:
            self.logger.debug("CN code not found: %s (normalized: %s)", code, normalized)
            return None

        return CNCodeInfo(
            cn_code=formatted,
            cn_code_raw=normalized,
            description=entry["description"],
            goods_category=entry["category"],
            cbam_applicable=True,
            unit=entry.get("unit", "tonnes"),
            notes=entry.get("notes", ""),
            chapter=entry.get("chapter", 0),
            heading=normalized[:4] if len(normalized) >= 4 else "",
        )

    def validate_cn_code(self, code: str) -> bool:
        """Check whether a CN code is in the CBAM Annex I database.

        Args:
            code: The CN code to validate.

        Returns:
            True if the code is a valid CBAM Annex I code.
        """
        normalized = self._normalize_cn_code(code)
        formatted = self._format_cn_code(normalized)
        return formatted in self._cn_codes

    def get_cn_codes_by_category(self, category: str) -> List[CNCodeInfo]:
        """Get all CN codes belonging to a specific goods category.

        Args:
            category: The goods category name (e.g. "IRON_AND_STEEL").

        Returns:
            List of CNCodeInfo for all codes in that category.
        """
        try:
            cat_enum = GoodsCategory(category)
        except ValueError:
            self.logger.warning("Unknown goods category: %s", category)
            return []

        results: List[CNCodeInfo] = []
        for formatted_code, entry in sorted(self._cn_codes.items()):
            if entry["category"] == cat_enum:
                raw = formatted_code.replace(" ", "")
                results.append(CNCodeInfo(
                    cn_code=formatted_code,
                    cn_code_raw=raw,
                    description=entry["description"],
                    goods_category=cat_enum,
                    cbam_applicable=True,
                    unit=entry.get("unit", "tonnes"),
                    notes=entry.get("notes", ""),
                    chapter=entry.get("chapter", 0),
                    heading=raw[:4] if len(raw) >= 4 else "",
                ))
        return results

    def get_all_cbam_cn_codes(self) -> Dict[str, List[CNCodeInfo]]:
        """Return the complete CBAM Annex I CN code database grouped by category.

        Returns:
            Dictionary mapping category name to list of CNCodeInfo.
        """
        result: Dict[str, List[CNCodeInfo]] = {}
        for category in GoodsCategory:
            codes = self.get_cn_codes_by_category(category.value)
            if codes:
                result[category.value] = codes
        return result

    # -------------------------------------------------------------------------
    # EORI Validation
    # -------------------------------------------------------------------------

    def validate_eori(self, eori: str) -> bool:
        """Validate an EORI number format.

        EORI format: 2-letter ISO country code followed by up to 15
        alphanumeric characters. EU member state EORI numbers use the
        country code of the issuing member state.

        Args:
            eori: The EORI number to validate.

        Returns:
            True if the EORI format is valid.
        """
        if not eori or len(eori) < 3 or len(eori) > 17:
            return False

        country_prefix = eori[:2].upper()
        identifier = eori[2:]

        # Country prefix must be a valid ISO alpha-2 code
        valid_countries = set(_EU_MEMBER_STATES.keys()) | set(_EFTA_MEMBERS.keys()) | {"GB", "XI"}
        if country_prefix not in valid_countries:
            self.logger.debug("EORI country prefix '%s' not recognized", country_prefix)
            return False

        # Identifier must be alphanumeric, 1-15 characters
        if not identifier or len(identifier) > 15:
            return False

        if not re.match(r'^[A-Za-z0-9]+$', identifier):
            return False

        return True

    # -------------------------------------------------------------------------
    # Country Information
    # -------------------------------------------------------------------------

    def get_country_info(self, country_code: str) -> Optional[CountryInfo]:
        """Get country information relevant to CBAM.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            CountryInfo if found, None otherwise.
        """
        code = country_code.upper()

        # Check EU membership
        if code in _EU_MEMBER_STATES:
            return CountryInfo(
                country_code=code,
                country_name=_EU_MEMBER_STATES[code],
                eu_member=True,
                efta_member=False,
                has_carbon_pricing=True,
                carbon_price_eur=0.0,  # EU ETS applies, handled via ETSBridge
                carbon_pricing_scheme="EU ETS",
                cbam_exempt=True,  # EU internal trade is exempt
            )

        # Check EFTA membership
        if code in _EFTA_MEMBERS:
            is_linked = code in {"IS", "LI", "NO"}  # EEA members linked to EU ETS
            return CountryInfo(
                country_code=code,
                country_name=_EFTA_MEMBERS[code],
                eu_member=False,
                efta_member=True,
                has_carbon_pricing=is_linked,
                carbon_price_eur=0.0 if is_linked else 0.0,
                carbon_pricing_scheme="EU ETS (linked)" if is_linked else "",
                cbam_exempt=is_linked,
            )

        # Check carbon pricing countries
        carbon_info = _CARBON_PRICING_COUNTRIES.get(code)
        has_carbon = carbon_info is not None

        # Build a name lookup for common non-EU countries
        country_names: Dict[str, str] = {
            "US": "United States", "GB": "United Kingdom", "CN": "China",
            "IN": "India", "JP": "Japan", "KR": "South Korea",
            "BR": "Brazil", "RU": "Russia", "AU": "Australia",
            "CA": "Canada", "MX": "Mexico", "TR": "Turkey",
            "ZA": "South Africa", "ID": "Indonesia", "TH": "Thailand",
            "VN": "Vietnam", "MY": "Malaysia", "PH": "Philippines",
            "EG": "Egypt", "NG": "Nigeria", "KE": "Kenya",
            "SA": "Saudi Arabia", "AE": "United Arab Emirates",
            "NZ": "New Zealand", "AR": "Argentina", "CL": "Chile",
            "CO": "Colombia", "PE": "Peru", "UA": "Ukraine",
            "KZ": "Kazakhstan", "BY": "Belarus", "GE": "Georgia",
            "SG": "Singapore", "TW": "Taiwan", "PK": "Pakistan",
        }

        return CountryInfo(
            country_code=code,
            country_name=country_names.get(code, code),
            eu_member=False,
            efta_member=False,
            has_carbon_pricing=has_carbon,
            carbon_price_eur=carbon_info["price_eur"] if carbon_info else 0.0,
            carbon_pricing_scheme=carbon_info["scheme"] if carbon_info else "",
            cbam_exempt=False,
        )

    # -------------------------------------------------------------------------
    # TARIC Mapping
    # -------------------------------------------------------------------------

    def map_taric_to_cbam(self, taric_code: str) -> Optional[str]:
        """Map a 10-digit TARIC code to its CBAM CN code equivalent.

        Args:
            taric_code: The 10-digit TARIC code.

        Returns:
            The formatted CBAM CN code, or None if no mapping exists.
        """
        clean = re.sub(r'[\s.\-]', '', taric_code)
        if len(clean) != 10:
            self.logger.debug("TARIC code must be 10 digits, got %d: %s", len(clean), taric_code)
            return None

        # Direct lookup
        if clean in _TARIC_TO_CBAM:
            return _TARIC_TO_CBAM[clean]

        # Fallback: try matching first 8 digits as CN code
        cn8 = clean[:8]
        formatted = self._format_cn_code(cn8)
        if formatted in self._cn_codes:
            return formatted

        self.logger.debug("No CBAM mapping found for TARIC code: %s", taric_code)
        return None

    # -------------------------------------------------------------------------
    # Customs Declaration Parsing
    # -------------------------------------------------------------------------

    def parse_customs_declaration(self, data: Dict[str, Any]) -> List[CustomsDeclarationItem]:
        """Extract CBAM-relevant items from a customs declaration data structure.

        Expects a dictionary with an 'items' key containing a list of goods
        entries. Each entry should have at minimum 'cn_code' and 'origin_country'.

        Args:
            data: Customs declaration data dictionary.

        Returns:
            List of CustomsDeclarationItem for CBAM-applicable items.
        """
        start_time = time.monotonic()
        items = data.get("items", [])
        if not items:
            self.logger.warning("No items found in customs declaration data")
            return []

        importer_eori = data.get("importer_eori", "")
        declaration_date = data.get("declaration_date", "")
        results: List[CustomsDeclarationItem] = []

        for raw_item in items:
            cn_code = raw_item.get("cn_code", "")
            origin = raw_item.get("origin_country", "")

            is_cbam = self.validate_cn_code(cn_code)
            code_info = self.lookup_cn_code(cn_code) if is_cbam else None

            item = CustomsDeclarationItem(
                cn_code=self._format_cn_code(self._normalize_cn_code(cn_code)),
                description=(
                    code_info.description if code_info
                    else raw_item.get("description", "")
                ),
                origin_country=origin.upper() if origin else "",
                net_mass_kg=float(raw_item.get("net_mass_kg", 0.0)),
                quantity=float(raw_item.get("quantity", 0.0)),
                unit=raw_item.get("unit", code_info.unit if code_info else "kg"),
                customs_value_eur=float(raw_item.get("customs_value_eur", 0.0)),
                cbam_applicable=is_cbam,
                importer_eori=importer_eori,
                declaration_date=declaration_date,
            )
            results.append(item)

        elapsed = (time.monotonic() - start_time) * 1000
        cbam_count = sum(1 for r in results if r.cbam_applicable)
        self.logger.info(
            "Parsed customs declaration: %d total items, %d CBAM-applicable in %.1fms",
            len(results), cbam_count, elapsed,
        )
        return results

    # -------------------------------------------------------------------------
    # Summary & Statistics
    # -------------------------------------------------------------------------

    def get_database_summary(self) -> Dict[str, Any]:
        """Return a summary of the CN code database contents.

        Returns:
            Dictionary with counts per category, total codes, etc.
        """
        by_category: Dict[str, int] = {}
        for entry in self._cn_codes.values():
            cat = entry["category"].value
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total_cn_codes": len(self._cn_codes),
            "codes_by_category": by_category,
            "categories": [c.value for c in GoodsCategory],
            "taric_mappings": len(_TARIC_TO_CBAM),
            "eu_member_states": len(_EU_MEMBER_STATES),
            "carbon_pricing_countries": len(_CARBON_PRICING_COUNTRIES),
            "initialized_at": self._initialized_at.isoformat(),
        }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_cn_code(code: str) -> str:
        """Normalize a CN code by removing spaces, dots, and dashes.

        Args:
            code: Raw CN code string.

        Returns:
            Digits-only CN code string.
        """
        return re.sub(r'[\s.\-]', '', code.strip())

    @staticmethod
    def _format_cn_code(raw: str) -> str:
        """Format a raw CN code into the standard 'XXXX XX XX' format.

        Args:
            raw: Digits-only CN code (8 digits expected).

        Returns:
            Formatted CN code with spaces.
        """
        if len(raw) < 8:
            raw = raw.ljust(8, '0')
        return f"{raw[:4]} {raw[4:6]} {raw[6:8]}"


# =============================================================================
# Module-Level Helper
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
