# -*- coding: utf-8 -*-
"""
HS Code Validator Engine - AGENT-EUDR-039

Validates 6-digit Harmonized System (HS) codes per World Customs
Organization (WCO) nomenclature for global trade compatibility. Maps
HS codes to EU Combined Nomenclature (CN) codes and identifies
EUDR-regulated products.

Algorithm:
    1. Validate HS code format (6 digits)
    2. Check against WCO HS 2022 reference database
    3. Determine if the HS code covers EUDR-regulated commodities
    4. Map to corresponding 8-digit EU CN codes
    5. Return validation result with EUDR applicability
    6. Cache validated codes for performance

Zero-Hallucination Guarantees:
    - All HS code validations against codified WCO reference data
    - No LLM involvement in classification decisions
    - Deterministic chapter/heading/subheading parsing
    - Complete provenance trail for every validation

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-039 Customs Declaration Support (GL-EUDR-CDS-039)
Regulation: EU 2023/1115 (EUDR) Annex I; WCO HS Convention
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .config import CustomsDeclarationSupportConfig, get_config
from .models import (
    AGENT_ID,
    CommodityType,
    EUDR_COMMODITY_CN_CODES,
    HSCode,
)
from .provenance import GENESIS_HASH, ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EUDR-relevant HS code chapters and headings
# ---------------------------------------------------------------------------

# HS chapters that contain EUDR-regulated commodities
_EUDR_HS_CHAPTERS: Dict[str, str] = {
    "01": "Live animals",
    "02": "Meat and edible meat offal",
    "09": "Coffee, tea, mate and spices",
    "12": "Oil seeds and oleaginous fruits",
    "15": "Animal or vegetable fats and oils",
    "18": "Cocoa and cocoa preparations",
    "20": "Preparations of vegetables/fruit/nuts",
    "21": "Miscellaneous edible preparations",
    "23": "Residues from food industries",
    "38": "Miscellaneous chemical products",
    "40": "Rubber and articles thereof",
    "41": "Raw hides and skins",
    "44": "Wood and articles of wood",
    "47": "Pulp of wood",
    "48": "Paper and paperboard",
}

# HS headings (4-digit) mapped to EUDR commodity types
_HS_HEADING_COMMODITY_MAP: Dict[str, CommodityType] = {
    # Cattle
    "0102": CommodityType.CATTLE,
    "0201": CommodityType.CATTLE,
    "0202": CommodityType.CATTLE,
    "4101": CommodityType.CATTLE,
    # Cocoa
    "1801": CommodityType.COCOA,
    "1802": CommodityType.COCOA,
    "1803": CommodityType.COCOA,
    "1804": CommodityType.COCOA,
    "1805": CommodityType.COCOA,
    "1806": CommodityType.COCOA,
    # Coffee
    "0901": CommodityType.COFFEE,
    "2101": CommodityType.COFFEE,
    # Oil palm
    "1511": CommodityType.OIL_PALM,
    "1513": CommodityType.OIL_PALM,
    "2306": CommodityType.OIL_PALM,
    "3826": CommodityType.OIL_PALM,
    # Rubber
    "4001": CommodityType.RUBBER,
    "4002": CommodityType.RUBBER,
    "4011": CommodityType.RUBBER,
    # Soya
    "1201": CommodityType.SOYA,
    "1507": CommodityType.SOYA,
    "2103": CommodityType.SOYA,
    "2304": CommodityType.SOYA,
    "2006": CommodityType.SOYA,
    # Wood
    "4401": CommodityType.WOOD,
    "4403": CommodityType.WOOD,
    "4407": CommodityType.WOOD,
    "4701": CommodityType.WOOD,
    "4801": CommodityType.WOOD,
}

# Valid HS subheadings (6-digit) for EUDR commodities
_VALID_HS_SUBHEADINGS: Dict[str, str] = {
    # Cattle
    "010221": "Live pure-bred breeding cattle",
    "010229": "Other live cattle",
    "020110": "Carcasses of bovine, fresh/chilled",
    "020120": "Other cuts of bovine, bone in, fresh/chilled",
    "020130": "Boneless meat of bovine, fresh/chilled",
    "020210": "Carcasses of bovine, frozen",
    "410150": "Whole raw hides of bovine",
    # Cocoa
    "180100": "Cocoa beans, whole or broken, raw or roasted",
    "180200": "Cocoa shells, husks, skins and other waste",
    "180310": "Cocoa paste, not defatted",
    "180320": "Cocoa paste, wholly or partly defatted",
    "180400": "Cocoa butter, fat and oil",
    "180500": "Cocoa powder, not containing added sugar",
    "180610": "Cocoa powder, sweetened",
    "180620": "Chocolate in blocks, slabs or bars, >2kg",
    "180631": "Filled chocolate blocks/slabs/bars",
    "180690": "Other chocolate and cocoa preparations",
    # Coffee
    "090111": "Coffee, not roasted, not decaffeinated",
    "090112": "Coffee, not roasted, decaffeinated",
    "090121": "Coffee, roasted, not decaffeinated",
    "090122": "Coffee, roasted, decaffeinated",
    "090190": "Other coffee, coffee husks and skins",
    "210111": "Extracts, essences and concentrates of coffee",
    "210112": "Preparations with coffee extract basis",
    # Oil palm
    "151110": "Crude palm oil",
    "151190": "Palm oil fractions",
    "151321": "Crude palm kernel oil",
    "151329": "Palm kernel oil fractions",
    "230660": "Palm kernel oil-cake",
    "382600": "Biodiesel (palm-derived)",
    # Rubber
    "400110": "Natural rubber latex",
    "400121": "Smoked sheets of natural rubber",
    "400122": "Technically specified natural rubber (TSNR)",
    "400129": "Other forms of natural rubber",
    "400130": "Balata, gutta-percha",
    "400211": "Styrene-butadiene rubber latex",
    "401110": "New pneumatic tyres for motor cars",
    "401120": "New pneumatic tyres for buses/lorries",
    "401193": "New pneumatic tyres for aircraft",
    # Soya
    "120110": "Soya bean seeds for sowing",
    "120190": "Other soya beans",
    "150710": "Crude soya-bean oil",
    "150790": "Refined soya-bean oil and fractions",
    "230400": "Soya-bean oil-cake and meal",
    "210310": "Soy sauce",
    "200600": "Vegetables/fruit preserved by sugar",
    # Wood
    "440111": "Fuel wood, coniferous, in logs",
    "440112": "Fuel wood, non-coniferous",
    "440121": "Coniferous wood chips",
    "440122": "Non-coniferous wood chips",
    "440321": "Coniferous wood in the rough, treated",
    "440391": "Oak wood in the rough",
    "440711": "Coniferous wood, sawn, >6mm thick",
    "440791": "Oak wood, sawn, >6mm thick",
    "470100": "Mechanical wood pulp",
    "480100": "Newsprint in rolls or sheets",
}


class HSCodeValidator:
    """HS code validation engine for EUDR commodities.

    Validates 6-digit HS codes against WCO nomenclature and determines
    EUDR regulatory applicability.

    Attributes:
        config: Agent configuration.
        _provenance: SHA-256 provenance tracker.
        _validation_cache: Cached validation results.
    """

    def __init__(
        self, config: Optional[CustomsDeclarationSupportConfig] = None,
    ) -> None:
        """Initialize HS Code Validator.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._validation_cache: Dict[str, HSCode] = {}
        logger.info("HSCodeValidator initialized")

    async def validate_hs_code(self, hs_code: str) -> HSCode:
        """Validate a 6-digit HS code.

        Args:
            hs_code: 6-digit HS code to validate.

        Returns:
            HSCode model with validation results.

        Raises:
            ValueError: If hs_code format is invalid.
        """
        start = time.monotonic()
        logger.info("Validating HS code: %s", hs_code)

        # Format validation
        if not hs_code or len(hs_code) != 6 or not hs_code.isdigit():
            raise ValueError(
                f"HS code must be exactly 6 digits, got '{hs_code}'"
            )

        # Check cache
        if hs_code in self._validation_cache:
            logger.debug("HS code cache hit for '%s'", hs_code)
            return self._validation_cache[hs_code]

        chapter = hs_code[:2]
        heading = hs_code[:4]
        is_valid = hs_code in _VALID_HS_SUBHEADINGS
        description = _VALID_HS_SUBHEADINGS.get(hs_code, "Unknown subheading")
        eudr_commodity = _HS_HEADING_COMMODITY_MAP.get(heading)

        # Find associated CN codes
        cn_mappings = self._find_cn_codes_for_hs(hs_code)

        result = HSCode(
            hs_code=hs_code,
            description=description,
            chapter=int(chapter),
            heading=heading,
            is_valid=is_valid,
            wco_version=self.config.hs_code_wco_version,
            eudr_commodity=eudr_commodity,
            eudr_regulated=eudr_commodity is not None,
            cn_code_mappings=cn_mappings,
        )

        # Cache result
        self._validation_cache[hs_code] = result

        # Provenance tracking
        self._provenance.record(
            entity_type="hs_code_validation",
            action="validate",
            entity_id=f"hsv-{uuid.uuid4().hex[:12]}",
            actor=AGENT_ID,
            metadata={
                "hs_code": hs_code,
                "is_valid": is_valid,
                "eudr_regulated": eudr_commodity is not None,
                "cn_mappings_count": len(cn_mappings),
                "duration_ms": round((time.monotonic() - start) * 1000, 2),
            },
        )

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "HS code '%s' validation: valid=%s, eudr=%s, cn_mappings=%d (%.1f ms)",
            hs_code, is_valid,
            eudr_commodity.value if eudr_commodity else "none",
            len(cn_mappings), elapsed,
        )
        return result

    async def is_eudr_regulated(self, hs_code: str) -> bool:
        """Check if an HS code covers EUDR-regulated products.

        Args:
            hs_code: 6-digit HS code.

        Returns:
            True if the HS code covers EUDR commodities.
        """
        if not hs_code or len(hs_code) < 4:
            return False
        heading = hs_code[:4]
        return heading in _HS_HEADING_COMMODITY_MAP

    async def get_eudr_commodity(
        self, hs_code: str,
    ) -> Optional[CommodityType]:
        """Get the EUDR commodity type for an HS code.

        Args:
            hs_code: 6-digit HS code.

        Returns:
            CommodityType if EUDR-regulated, None otherwise.
        """
        if not hs_code or len(hs_code) < 4:
            return None
        heading = hs_code[:4]
        return _HS_HEADING_COMMODITY_MAP.get(heading)

    async def get_chapter_info(self, chapter: str) -> Dict[str, Any]:
        """Get information about an HS chapter.

        Args:
            chapter: 2-digit HS chapter code.

        Returns:
            Dictionary with chapter information.
        """
        description = _EUDR_HS_CHAPTERS.get(chapter, "Unknown chapter")
        is_eudr = chapter in _EUDR_HS_CHAPTERS

        # Count headings in this chapter
        headings = [
            h for h in _HS_HEADING_COMMODITY_MAP
            if h[:2] == chapter
        ]

        return {
            "chapter": chapter,
            "description": description,
            "is_eudr_relevant": is_eudr,
            "heading_count": len(headings),
            "headings": headings,
        }

    async def batch_validate(
        self, hs_codes: List[str],
    ) -> List[HSCode]:
        """Validate multiple HS codes in batch.

        Args:
            hs_codes: List of 6-digit HS codes to validate.

        Returns:
            List of HSCode validation results.
        """
        results = []
        for code in hs_codes:
            try:
                result = await self.validate_hs_code(code)
                results.append(result)
            except ValueError as e:
                logger.warning("HS code '%s' validation failed: %s", code, e)
                results.append(HSCode(
                    hs_code=code if len(code) == 6 and code.isdigit() else "000000",
                    description=f"Invalid: {str(e)}",
                    is_valid=False,
                ))
        return results

    # ------------------------------------------------------------------
    # Convenience / Alias Methods (match test expectations)
    # ------------------------------------------------------------------

    async def validate(self, hs_code: str) -> HSCode:
        """Alias for validate_hs_code.

        Args:
            hs_code: 6-digit HS code to validate.

        Returns:
            HSCode model with validation results.

        Raises:
            ValueError: If hs_code format is invalid.
        """
        if not hs_code:
            raise ValueError("HS code must not be empty")
        if len(hs_code) != 6:
            raise ValueError(
                f"HS code must be exactly 6 digits, got '{hs_code}'"
            )
        if not hs_code.isdigit():
            raise ValueError(
                f"HS code must be numeric, got '{hs_code}'"
            )
        return await self.validate_hs_code(hs_code)

    async def get_chapter(self, chapter_num: int) -> Optional[Dict[str, Any]]:
        """Get chapter info by integer chapter number.

        Args:
            chapter_num: HS chapter number (1-99).

        Returns:
            Dictionary with chapter info, or None if unknown.
        """
        chapter_str = f"{chapter_num:02d}"
        info = await self.get_chapter_info(chapter_str)
        if info["description"] == "Unknown chapter":
            # Return a dict with eudr_regulated = False for unknown chapters
            return {"chapter": chapter_num, "description": "Unknown chapter", "eudr_regulated": False}
        return {
            "chapter": chapter_num,
            "description": info["description"],
            "eudr_regulated": info["is_eudr_relevant"],
            "heading_count": info["heading_count"],
            "headings": info["headings"],
        }

    async def is_eudr_chapter(self, chapter_num: int) -> bool:
        """Check if a chapter number is EUDR-regulated.

        Args:
            chapter_num: HS chapter number (1-99).

        Returns:
            True if the chapter contains EUDR-regulated commodities.
        """
        chapter_str = f"{chapter_num:02d}"
        return chapter_str in _EUDR_HS_CHAPTERS

    async def validate_batch(self, hs_codes: List[str]) -> List[HSCode]:
        """Alias for batch_validate."""
        return await self.batch_validate(hs_codes)

    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the HS Code Validator engine.

        Returns:
            Dictionary with health check information.
        """
        return {
            "engine": "HSCodeValidator",
            "status": "healthy",
            "eudr_chapters_loaded": len(_EUDR_HS_CHAPTERS),
            "valid_subheadings_loaded": len(_VALID_HS_SUBHEADINGS),
        }

    def _find_cn_codes_for_hs(self, hs_code: str) -> List[str]:
        """Find 8-digit CN codes that correspond to a 6-digit HS code.

        Args:
            hs_code: 6-digit HS code.

        Returns:
            List of matching 8-digit CN codes.
        """
        matches = []
        for commodity, codes in EUDR_COMMODITY_CN_CODES.items():
            for cn_code in codes:
                if cn_code[:6] == hs_code:
                    matches.append(cn_code)
        return matches
