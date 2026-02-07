# -*- coding: utf-8 -*-
"""
Entity Resolver - AGENT-FOUND-003: Unit & Reference Normalizer

Resolves fuel names, material names, and process names to canonical
identifiers with confidence scoring. Uses exact match, alias match,
and Levenshtein fuzzy matching.

Zero-Hallucination Guarantees:
    - All resolution uses deterministic lookup tables
    - Fuzzy matching uses Levenshtein distance (no ML/LLM)
    - Confidence scores are deterministic functions of edit distance

Example:
    >>> from greenlang.normalizer.entity_resolver import EntityResolver
    >>> r = EntityResolver()
    >>> m = r.resolve_fuel("natural gas")
    >>> print(m.canonical_name)  # "Natural Gas"

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-003 Unit & Reference Normalizer
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from greenlang.normalizer.models import (
    ConfidenceLevel,
    EntityMatch,
    EntityResolutionResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FUEL STANDARDIZATION TABLES (from foundation agent)
# =============================================================================

FUEL_STANDARDIZATION: Dict[str, Dict[str, str]] = {
    # Natural Gas variants
    "natural gas": {"name": "Natural Gas", "code": "NG", "category": "gaseous"},
    "nat gas": {"name": "Natural Gas", "code": "NG", "category": "gaseous"},
    "natural_gas": {"name": "Natural Gas", "code": "NG", "category": "gaseous"},
    "methane": {"name": "Natural Gas", "code": "NG", "category": "gaseous"},
    "ng": {"name": "Natural Gas", "code": "NG", "category": "gaseous"},
    "pipeline gas": {"name": "Natural Gas", "code": "NG", "category": "gaseous"},
    "cng": {"name": "Compressed Natural Gas", "code": "CNG", "category": "gaseous"},
    "compressed natural gas": {"name": "Compressed Natural Gas", "code": "CNG", "category": "gaseous"},
    "lng": {"name": "Liquefied Natural Gas", "code": "LNG", "category": "gaseous"},
    "liquefied natural gas": {"name": "Liquefied Natural Gas", "code": "LNG", "category": "gaseous"},
    # Diesel variants
    "diesel": {"name": "Diesel", "code": "DSL", "category": "liquid"},
    "diesel fuel": {"name": "Diesel", "code": "DSL", "category": "liquid"},
    "diesel oil": {"name": "Diesel", "code": "DSL", "category": "liquid"},
    "gas oil": {"name": "Diesel", "code": "DSL", "category": "liquid"},
    "derv": {"name": "Diesel", "code": "DSL", "category": "liquid"},
    "red diesel": {"name": "Red Diesel", "code": "RDS", "category": "liquid"},
    "biodiesel": {"name": "Biodiesel", "code": "BDS", "category": "biofuel"},
    "b100": {"name": "Biodiesel", "code": "BDS", "category": "biofuel"},
    "b20": {"name": "Biodiesel Blend B20", "code": "B20", "category": "biofuel"},
    # Gasoline/Petrol variants
    "gasoline": {"name": "Gasoline", "code": "GAS", "category": "liquid"},
    "petrol": {"name": "Gasoline", "code": "GAS", "category": "liquid"},
    "motor gasoline": {"name": "Gasoline", "code": "GAS", "category": "liquid"},
    "mogas": {"name": "Gasoline", "code": "GAS", "category": "liquid"},
    "unleaded": {"name": "Gasoline", "code": "GAS", "category": "liquid"},
    "premium gasoline": {"name": "Premium Gasoline", "code": "PGS", "category": "liquid"},
    "e10": {"name": "Gasoline E10", "code": "E10", "category": "biofuel"},
    "e85": {"name": "Ethanol E85", "code": "E85", "category": "biofuel"},
    "ethanol": {"name": "Ethanol", "code": "ETH", "category": "biofuel"},
    # Propane/LPG variants
    "propane": {"name": "Propane", "code": "PRP", "category": "gaseous"},
    "lpg": {"name": "Liquefied Petroleum Gas", "code": "LPG", "category": "gaseous"},
    "liquefied petroleum gas": {"name": "Liquefied Petroleum Gas", "code": "LPG", "category": "gaseous"},
    "butane": {"name": "Butane", "code": "BUT", "category": "gaseous"},
    "autogas": {"name": "Liquefied Petroleum Gas", "code": "LPG", "category": "gaseous"},
    # Fuel Oil variants
    "fuel oil": {"name": "Fuel Oil", "code": "FO", "category": "liquid"},
    "heating oil": {"name": "Heating Oil", "code": "HO", "category": "liquid"},
    "hfo": {"name": "Heavy Fuel Oil", "code": "HFO", "category": "liquid"},
    "heavy fuel oil": {"name": "Heavy Fuel Oil", "code": "HFO", "category": "liquid"},
    "bunker fuel": {"name": "Heavy Fuel Oil", "code": "HFO", "category": "liquid"},
    "residual fuel oil": {"name": "Residual Fuel Oil", "code": "RFO", "category": "liquid"},
    "kerosene": {"name": "Kerosene", "code": "KER", "category": "liquid"},
    "jet fuel": {"name": "Jet Fuel", "code": "JET", "category": "liquid"},
    "jet a": {"name": "Jet Fuel A", "code": "JTA", "category": "liquid"},
    "aviation fuel": {"name": "Aviation Gasoline", "code": "AVG", "category": "liquid"},
    # Coal variants
    "coal": {"name": "Coal", "code": "COL", "category": "solid"},
    "bituminous coal": {"name": "Bituminous Coal", "code": "BCO", "category": "solid"},
    "anthracite": {"name": "Anthracite Coal", "code": "ANT", "category": "solid"},
    "lignite": {"name": "Lignite", "code": "LIG", "category": "solid"},
    "brown coal": {"name": "Lignite", "code": "LIG", "category": "solid"},
    "sub-bituminous": {"name": "Sub-bituminous Coal", "code": "SBC", "category": "solid"},
    "coke": {"name": "Coke", "code": "COK", "category": "solid"},
    "petroleum coke": {"name": "Petroleum Coke", "code": "PCK", "category": "solid"},
    # Biomass variants
    "biomass": {"name": "Biomass", "code": "BIO", "category": "biofuel"},
    "wood": {"name": "Wood", "code": "WOD", "category": "biofuel"},
    "wood chips": {"name": "Wood Chips", "code": "WCH", "category": "biofuel"},
    "wood pellets": {"name": "Wood Pellets", "code": "WPL", "category": "biofuel"},
    "firewood": {"name": "Firewood", "code": "FWD", "category": "biofuel"},
    "biogas": {"name": "Biogas", "code": "BGS", "category": "biofuel"},
    "landfill gas": {"name": "Landfill Gas", "code": "LFG", "category": "biofuel"},
    # Electricity
    "electricity": {"name": "Electricity", "code": "ELC", "category": "electricity"},
    "grid electricity": {"name": "Grid Electricity", "code": "GRD", "category": "electricity"},
    "renewable electricity": {"name": "Renewable Electricity", "code": "REN", "category": "electricity"},
    "solar": {"name": "Solar Electricity", "code": "SOL", "category": "electricity"},
    "wind": {"name": "Wind Electricity", "code": "WND", "category": "electricity"},
    # Hydrogen
    "hydrogen": {"name": "Hydrogen", "code": "H2", "category": "gaseous"},
    "green hydrogen": {"name": "Green Hydrogen", "code": "GH2", "category": "gaseous"},
    "blue hydrogen": {"name": "Blue Hydrogen", "code": "BH2", "category": "gaseous"},
    "grey hydrogen": {"name": "Grey Hydrogen", "code": "YH2", "category": "gaseous"},
}


# =============================================================================
# MATERIAL STANDARDIZATION TABLES (from foundation agent)
# =============================================================================

MATERIAL_STANDARDIZATION: Dict[str, Dict[str, str]] = {
    # Metals
    "steel": {"name": "Steel", "code": "STL", "category": "metals"},
    "carbon steel": {"name": "Carbon Steel", "code": "CST", "category": "metals"},
    "stainless steel": {"name": "Stainless Steel", "code": "SST", "category": "metals"},
    "aluminum": {"name": "Aluminum", "code": "ALU", "category": "metals"},
    "aluminium": {"name": "Aluminum", "code": "ALU", "category": "metals"},
    "copper": {"name": "Copper", "code": "COP", "category": "metals"},
    "iron": {"name": "Iron", "code": "IRN", "category": "metals"},
    "cast iron": {"name": "Cast Iron", "code": "CIR", "category": "metals"},
    "pig iron": {"name": "Pig Iron", "code": "PIR", "category": "metals"},
    "zinc": {"name": "Zinc", "code": "ZNC", "category": "metals"},
    "lead": {"name": "Lead", "code": "LED", "category": "metals"},
    "nickel": {"name": "Nickel", "code": "NIC", "category": "metals"},
    "titanium": {"name": "Titanium", "code": "TIT", "category": "metals"},
    "brass": {"name": "Brass", "code": "BRS", "category": "metals"},
    "bronze": {"name": "Bronze", "code": "BRZ", "category": "metals"},
    # Plastics
    "plastic": {"name": "Plastic (Generic)", "code": "PLS", "category": "plastics"},
    "pet": {"name": "PET (Polyethylene Terephthalate)", "code": "PET", "category": "plastics"},
    "polyethylene terephthalate": {"name": "PET (Polyethylene Terephthalate)", "code": "PET", "category": "plastics"},
    "hdpe": {"name": "HDPE (High-Density Polyethylene)", "code": "HDPE", "category": "plastics"},
    "high density polyethylene": {"name": "HDPE (High-Density Polyethylene)", "code": "HDPE", "category": "plastics"},
    "ldpe": {"name": "LDPE (Low-Density Polyethylene)", "code": "LDPE", "category": "plastics"},
    "low density polyethylene": {"name": "LDPE (Low-Density Polyethylene)", "code": "LDPE", "category": "plastics"},
    "pvc": {"name": "PVC (Polyvinyl Chloride)", "code": "PVC", "category": "plastics"},
    "polyvinyl chloride": {"name": "PVC (Polyvinyl Chloride)", "code": "PVC", "category": "plastics"},
    "pp": {"name": "PP (Polypropylene)", "code": "PP", "category": "plastics"},
    "polypropylene": {"name": "PP (Polypropylene)", "code": "PP", "category": "plastics"},
    "ps": {"name": "PS (Polystyrene)", "code": "PS", "category": "plastics"},
    "polystyrene": {"name": "PS (Polystyrene)", "code": "PS", "category": "plastics"},
    "abs": {"name": "ABS (Acrylonitrile Butadiene Styrene)", "code": "ABS", "category": "plastics"},
    "nylon": {"name": "Nylon", "code": "NYL", "category": "plastics"},
    "polyamide": {"name": "Nylon", "code": "NYL", "category": "plastics"},
    # Construction materials
    "cement": {"name": "Cement", "code": "CEM", "category": "construction"},
    "portland cement": {"name": "Portland Cement", "code": "PCM", "category": "construction"},
    "concrete": {"name": "Concrete", "code": "CON", "category": "construction"},
    "reinforced concrete": {"name": "Reinforced Concrete", "code": "RCO", "category": "construction"},
    "brick": {"name": "Brick", "code": "BRK", "category": "construction"},
    "glass": {"name": "Glass", "code": "GLS", "category": "construction"},
    "float glass": {"name": "Float Glass", "code": "FGL", "category": "construction"},
    "timber": {"name": "Timber", "code": "TMB", "category": "construction"},
    "lumber": {"name": "Timber", "code": "TMB", "category": "construction"},
    "plywood": {"name": "Plywood", "code": "PLY", "category": "construction"},
    "mdf": {"name": "MDF (Medium Density Fiberboard)", "code": "MDF", "category": "construction"},
    "gypsum": {"name": "Gypsum", "code": "GYP", "category": "construction"},
    "drywall": {"name": "Drywall", "code": "DRY", "category": "construction"},
    "asphalt": {"name": "Asphalt", "code": "ASP", "category": "construction"},
    "bitumen": {"name": "Bitumen", "code": "BIT", "category": "construction"},
    "gravel": {"name": "Gravel", "code": "GRV", "category": "construction"},
    "sand": {"name": "Sand", "code": "SND", "category": "construction"},
    "aggregate": {"name": "Aggregate", "code": "AGG", "category": "construction"},
    # Paper and packaging
    "paper": {"name": "Paper", "code": "PAP", "category": "paper"},
    "cardboard": {"name": "Cardboard", "code": "CBD", "category": "paper"},
    "corrugated cardboard": {"name": "Corrugated Cardboard", "code": "CCB", "category": "paper"},
    "kraft paper": {"name": "Kraft Paper", "code": "KFT", "category": "paper"},
    "newsprint": {"name": "Newsprint", "code": "NWS", "category": "paper"},
    "recycled paper": {"name": "Recycled Paper", "code": "RCP", "category": "paper"},
    # Chemicals
    "ammonia": {"name": "Ammonia", "code": "NH3", "category": "chemicals"},
    "urea": {"name": "Urea", "code": "URE", "category": "chemicals"},
    "fertilizer": {"name": "Fertilizer", "code": "FER", "category": "chemicals"},
    "sulfuric acid": {"name": "Sulfuric Acid", "code": "H2SO4", "category": "chemicals"},
    "nitric acid": {"name": "Nitric Acid", "code": "HNO3", "category": "chemicals"},
    "chlorine": {"name": "Chlorine", "code": "CL2", "category": "chemicals"},
    "sodium hydroxide": {"name": "Sodium Hydroxide", "code": "NaOH", "category": "chemicals"},
    "caustic soda": {"name": "Sodium Hydroxide", "code": "NaOH", "category": "chemicals"},
}


# =============================================================================
# PROCESS STANDARDIZATION TABLES
# =============================================================================

PROCESS_STANDARDIZATION: Dict[str, Dict[str, str]] = {
    # Manufacturing processes
    "welding": {"name": "Welding", "code": "WLD", "category": "manufacturing"},
    "casting": {"name": "Casting", "code": "CST", "category": "manufacturing"},
    "forging": {"name": "Forging", "code": "FRG", "category": "manufacturing"},
    "machining": {"name": "Machining", "code": "MCH", "category": "manufacturing"},
    "stamping": {"name": "Stamping", "code": "STP", "category": "manufacturing"},
    "extrusion": {"name": "Extrusion", "code": "EXT", "category": "manufacturing"},
    "injection molding": {"name": "Injection Molding", "code": "IJM", "category": "manufacturing"},
    "injection moulding": {"name": "Injection Molding", "code": "IJM", "category": "manufacturing"},
    "blow molding": {"name": "Blow Molding", "code": "BLM", "category": "manufacturing"},
    "thermoforming": {"name": "Thermoforming", "code": "THF", "category": "manufacturing"},
    # Transport processes
    "road transport": {"name": "Road Transport", "code": "RTR", "category": "transport"},
    "trucking": {"name": "Road Transport", "code": "RTR", "category": "transport"},
    "rail transport": {"name": "Rail Transport", "code": "RLT", "category": "transport"},
    "sea freight": {"name": "Sea Freight", "code": "SFR", "category": "transport"},
    "ocean freight": {"name": "Sea Freight", "code": "SFR", "category": "transport"},
    "air freight": {"name": "Air Freight", "code": "AFR", "category": "transport"},
    "pipeline transport": {"name": "Pipeline Transport", "code": "PLT", "category": "transport"},
    # Energy processes
    "combustion": {"name": "Combustion", "code": "CMB", "category": "energy"},
    "incineration": {"name": "Incineration", "code": "INC", "category": "energy"},
    "power generation": {"name": "Power Generation", "code": "PWG", "category": "energy"},
    "electricity generation": {"name": "Power Generation", "code": "PWG", "category": "energy"},
    "cogeneration": {"name": "Cogeneration", "code": "COG", "category": "energy"},
    "chp": {"name": "Cogeneration", "code": "COG", "category": "energy"},
    # Waste processes
    "landfill": {"name": "Landfill", "code": "LNF", "category": "waste"},
    "composting": {"name": "Composting", "code": "CMP", "category": "waste"},
    "recycling": {"name": "Recycling", "code": "RCY", "category": "waste"},
    "wastewater treatment": {"name": "Wastewater Treatment", "code": "WWT", "category": "waste"},
    "anaerobic digestion": {"name": "Anaerobic Digestion", "code": "ADG", "category": "waste"},
}


# Entity type to vocabulary mapping
ENTITY_VOCABULARIES: Dict[str, Dict[str, Dict[str, str]]] = {
    "fuel": FUEL_STANDARDIZATION,
    "material": MATERIAL_STANDARDIZATION,
    "process": PROCESS_STANDARDIZATION,
}


# =============================================================================
# LEVENSHTEIN DISTANCE (pure Python, no external dependency)
# =============================================================================


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Integer edit distance.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute normalised Levenshtein similarity (0.0 to 1.0).

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Float similarity score.
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


# =============================================================================
# ENTITY RESOLVER
# =============================================================================


class EntityResolver:
    """Resolves entity names to canonical identifiers with confidence scoring.

    Supports three entity types: fuel, material, and process.
    Resolution follows a three-tier strategy:

    1. EXACT match against vocabulary keys (confidence 1.0)
    2. ALIAS match via canonical name grouping (confidence 0.95)
    3. FUZZY match via Levenshtein distance (confidence 0.7-0.9)

    Below 0.5 similarity the entity is marked UNRESOLVED.

    Example:
        >>> resolver = EntityResolver()
        >>> match = resolver.resolve_fuel("natural gas")
        >>> print(match.canonical_name)  # "Natural Gas"
        >>> print(match.confidence)  # 1.0
    """

    def __init__(self) -> None:
        """Initialize EntityResolver."""
        logger.info("EntityResolver initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_fuel(self, name: str) -> EntityMatch:
        """Resolve a fuel name to its canonical form.

        Args:
            name: Raw fuel name.

        Returns:
            EntityMatch with resolved fuel information.
        """
        return self._resolve(name, "fuel", FUEL_STANDARDIZATION)

    def resolve_material(self, name: str) -> EntityMatch:
        """Resolve a material name to its canonical form.

        Args:
            name: Raw material name.

        Returns:
            EntityMatch with resolved material information.
        """
        return self._resolve(name, "material", MATERIAL_STANDARDIZATION)

    def resolve_process(self, name: str) -> EntityMatch:
        """Resolve a process name to its canonical form.

        Args:
            name: Raw process name.

        Returns:
            EntityMatch with resolved process information.
        """
        return self._resolve(name, "process", PROCESS_STANDARDIZATION)

    def batch_resolve(
        self,
        items: List[str],
        entity_type: str,
    ) -> EntityResolutionResult:
        """Resolve a batch of entity names.

        Args:
            items: List of raw entity names.
            entity_type: Entity type (fuel, material, process).

        Returns:
            EntityResolutionResult with matches and unresolved items.

        Raises:
            ValueError: If entity_type is not recognised.
        """
        vocab = ENTITY_VOCABULARIES.get(entity_type)
        if vocab is None:
            raise ValueError(
                f"Unknown entity type: {entity_type}. "
                f"Supported: {list(ENTITY_VOCABULARIES.keys())}"
            )

        matches: List[EntityMatch] = []
        unresolved: List[str] = []

        for item in items:
            match = self._resolve(item, entity_type, vocab)
            if match.confidence_level == ConfidenceLevel.UNRESOLVED:
                unresolved.append(item)
            else:
                matches.append(match)

        return EntityResolutionResult(matches=matches, unresolved=unresolved)

    def search_vocabulary(
        self,
        query: str,
        entity_type: str,
        limit: int = 10,
    ) -> List[EntityMatch]:
        """Search vocabulary for entities matching a query.

        Returns results sorted by descending similarity score.

        Args:
            query: Search query string.
            entity_type: Entity type (fuel, material, process).
            limit: Maximum results to return.

        Returns:
            List of EntityMatch results sorted by confidence.

        Raises:
            ValueError: If entity_type is not recognised.
        """
        vocab = ENTITY_VOCABULARIES.get(entity_type)
        if vocab is None:
            raise ValueError(
                f"Unknown entity type: {entity_type}. "
                f"Supported: {list(ENTITY_VOCABULARIES.keys())}"
            )

        query_norm = query.lower().strip()
        scored: List[Tuple[float, str, Dict[str, str]]] = []

        for key, info in vocab.items():
            sim = _levenshtein_similarity(query_norm, key)
            scored.append((sim, key, info))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[EntityMatch] = []
        for sim, key, info in scored[:limit]:
            conf_level = self._classify_confidence(sim, query_norm == key)
            results.append(EntityMatch(
                raw_input=query,
                resolved_id=info["code"],
                canonical_name=info["name"],
                entity_type=entity_type,
                confidence=round(sim, 4),
                confidence_level=conf_level,
                match_method="search",
            ))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(
        self,
        name: str,
        entity_type: str,
        vocab: Dict[str, Dict[str, str]],
    ) -> EntityMatch:
        """Resolve a single entity name against a vocabulary.

        Args:
            name: Raw entity name.
            entity_type: Entity type label.
            vocab: Vocabulary dictionary to search.

        Returns:
            EntityMatch with resolution result.
        """
        normalized = name.lower().strip()

        # Tier 1: Exact match
        if normalized in vocab:
            info = vocab[normalized]
            return EntityMatch(
                raw_input=name,
                resolved_id=info["code"],
                canonical_name=info["name"],
                entity_type=entity_type,
                confidence=1.0,
                confidence_level=ConfidenceLevel.EXACT,
                match_method="exact",
            )

        # Tier 2: Substring / alias match
        alias_match, alias_score = self._alias_match(normalized, vocab)
        if alias_match is not None and alias_score >= 0.9:
            return EntityMatch(
                raw_input=name,
                resolved_id=alias_match["code"],
                canonical_name=alias_match["name"],
                entity_type=entity_type,
                confidence=0.95,
                confidence_level=ConfidenceLevel.ALIAS,
                match_method="alias",
            )

        # Tier 3: Fuzzy Levenshtein match
        fuzzy_match, fuzzy_score = self._fuzzy_match(normalized, vocab)
        if fuzzy_match is not None and fuzzy_score >= 0.5:
            conf_level = self._classify_confidence(fuzzy_score, False)
            return EntityMatch(
                raw_input=name,
                resolved_id=fuzzy_match["code"],
                canonical_name=fuzzy_match["name"],
                entity_type=entity_type,
                confidence=round(fuzzy_score, 4),
                confidence_level=conf_level,
                match_method="fuzzy",
            )

        # Unresolved
        return EntityMatch(
            raw_input=name,
            resolved_id="UNK",
            canonical_name=name.title(),
            entity_type=entity_type,
            confidence=0.0,
            confidence_level=ConfidenceLevel.UNRESOLVED,
            match_method="none",
        )

    def _alias_match(
        self,
        name: str,
        vocab: Dict[str, Dict[str, str]],
    ) -> Tuple[Optional[Dict[str, str]], float]:
        """Attempt alias/substring matching.

        Args:
            name: Normalised entity name.
            vocab: Vocabulary to search.

        Returns:
            Tuple of (matched info dict or None, score).
        """
        best_match: Optional[Dict[str, str]] = None
        best_score = 0.0

        for key, info in vocab.items():
            if key in name or name in key:
                score = min(len(name), len(key)) / max(len(name), len(key))
                if score > best_score:
                    best_score = score
                    best_match = info

        return best_match, best_score

    def _fuzzy_match(
        self,
        name: str,
        vocab: Dict[str, Dict[str, str]],
    ) -> Tuple[Optional[Dict[str, str]], float]:
        """Attempt fuzzy matching using Levenshtein distance.

        Args:
            name: Normalised entity name.
            vocab: Vocabulary to search.

        Returns:
            Tuple of (matched info dict or None, similarity score).
        """
        best_match: Optional[Dict[str, str]] = None
        best_score = 0.0

        for key, info in vocab.items():
            sim = _levenshtein_similarity(name, key)
            if sim > best_score:
                best_score = sim
                best_match = info

        if best_score >= 0.5:
            return best_match, best_score
        return None, 0.0

    @staticmethod
    def _classify_confidence(
        score: float, is_exact: bool,
    ) -> ConfidenceLevel:
        """Classify a numeric score into a confidence level.

        Args:
            score: Numeric confidence score 0.0-1.0.
            is_exact: Whether the match was an exact key match.

        Returns:
            ConfidenceLevel enum value.
        """
        if is_exact or score >= 1.0:
            return ConfidenceLevel.EXACT
        if score >= 0.9:
            return ConfidenceLevel.ALIAS
        if score >= 0.7:
            return ConfidenceLevel.FUZZY
        if score >= 0.5:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNRESOLVED


__all__ = [
    "EntityResolver",
    "FUEL_STANDARDIZATION",
    "MATERIAL_STANDARDIZATION",
    "PROCESS_STANDARDIZATION",
    "ENTITY_VOCABULARIES",
]
