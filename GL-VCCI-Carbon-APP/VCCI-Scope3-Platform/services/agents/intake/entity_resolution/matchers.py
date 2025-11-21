# -*- coding: utf-8 -*-
"""
Entity Matching Algorithms

Deterministic and fuzzy matching for entity resolution.

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from fuzzywuzzy import fuzz
from rapidfuzz import process, fuzz as rapidfuzz

from ..models import EntityMatchCandidate, ResolutionMethod
from ..config import get_config

logger = logging.getLogger(__name__)


class DeterministicMatcher:
    """Exact matching for entity resolution."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize deterministic matcher."""
        self.config = get_config().resolution if config is None else config
        self.entity_cache: Dict[str, str] = {}
        logger.info("Initialized DeterministicMatcher")

    def match_by_id(
        self,
        entity_identifier: str,
        entity_db: Dict[str, Dict]
    ) -> Optional[EntityMatchCandidate]:
        """Match by exact ID."""
        if entity_identifier in entity_db:
            entity = entity_db[entity_identifier]
            return EntityMatchCandidate(
                canonical_id=entity_identifier,
                canonical_name=entity.get("name", ""),
                confidence_score=100.0,
                resolution_method=ResolutionMethod.EXACT_MATCH,
                match_details={"matched_by": "id"}
            )
        return None

    def match_by_name(
        self,
        entity_name: str,
        entity_db: Dict[str, Dict]
    ) -> Optional[EntityMatchCandidate]:
        """Match by exact name (case-insensitive)."""
        name_lower = entity_name.lower().strip()
        
        for entity_id, entity in entity_db.items():
            if entity.get("name", "").lower().strip() == name_lower:
                return EntityMatchCandidate(
                    canonical_id=entity_id,
                    canonical_name=entity.get("name", ""),
                    confidence_score=100.0,
                    resolution_method=ResolutionMethod.EXACT_MATCH,
                    match_details={"matched_by": "name"}
                )
        return None


class FuzzyMatcher:
    """Fuzzy string matching for entity resolution."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize fuzzy matcher."""
        self.config = get_config().resolution if config is None else config
        logger.info("Initialized FuzzyMatcher")

    def match(
        self,
        entity_name: str,
        entity_db: Dict[str, Dict],
        limit: int = 5
    ) -> List[EntityMatchCandidate]:
        """
        Fuzzy match entity name against database.
        
        Args:
            entity_name: Name to match
            entity_db: Entity database
            limit: Max candidates to return
            
        Returns:
            List of match candidates sorted by confidence
        """
        candidates = []
        
        # Create list of (id, name) pairs
        entities = [(eid, e.get("name", "")) for eid, e in entity_db.items()]
        
        # Use rapidfuzz for fast matching
        matches = process.extract(
            entity_name,
            [e[1] for e in entities],
            scorer=rapidfuzz.token_sort_ratio,
            limit=limit
        )
        
        for matched_name, score, idx in matches:
            if score >= self.config.fuzzy_min_score:
                entity_id = entities[idx][0]
                
                # Calculate detailed scores
                simple_ratio = fuzz.ratio(entity_name, matched_name)
                token_sort = fuzz.token_sort_ratio(entity_name, matched_name)
                token_set = fuzz.token_set_ratio(entity_name, matched_name)
                
                # Weighted average
                confidence = (simple_ratio * 0.3 + token_sort * 0.4 + token_set * 0.3)
                
                candidate = EntityMatchCandidate(
                    canonical_id=entity_id,
                    canonical_name=matched_name,
                    confidence_score=confidence,
                    resolution_method=ResolutionMethod.FUZZY_MATCH,
                    match_details={
                        "simple_ratio": simple_ratio,
                        "token_sort_ratio": token_sort,
                        "token_set_ratio": token_set,
                        "algorithm": self.config.fuzzy_algorithm
                    }
                )
                candidates.append(candidate)
        
        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        logger.info(f"Found {len(candidates)} fuzzy matches for '{entity_name}'")
        return candidates


__all__ = ["DeterministicMatcher", "FuzzyMatcher"]
