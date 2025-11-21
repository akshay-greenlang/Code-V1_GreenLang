# -*- coding: utf-8 -*-
"""
Entity Resolver - Main Resolution Orchestrator

Multi-strategy entity resolution with confidence scoring and caching.

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from ..models import (
from greenlang.determinism import DeterministicClock
    IngestionRecord,
    ResolvedEntity,
    EntityMatchCandidate,
    EntityType,
    ResolutionMethod,
)
from ..config import get_config
from ..exceptions import EntityResolutionError
from .matchers import DeterministicMatcher, FuzzyMatcher
from .mdm_integration import MDMIntegrator

logger = logging.getLogger(__name__)


class EntityResolver:
    """
    Main entity resolution orchestrator.

    Implements cascading resolution strategy:
    1. Exact ID match (deterministic)
    2. Exact name match (deterministic)
    3. Fuzzy name match
    4. MDM lookup (LEI, DUNS, OpenCorporates)
    5. Send to review queue if confidence < threshold
    """

    def __init__(self, entity_db: Optional[Dict[str, Dict]] = None, config: Optional[Dict] = None):
        """
        Initialize entity resolver.

        Args:
            entity_db: Entity master database {entity_id: {name, ...}}
            config: Optional configuration override
        """
        self.config = get_config().resolution if config is None else config
        self.entity_db = entity_db or {}

        # Initialize matchers
        self.deterministic_matcher = DeterministicMatcher(config)
        self.fuzzy_matcher = FuzzyMatcher(config)
        self.mdm_integrator = MDMIntegrator(config)

        # Resolution cache
        self.cache: Dict[str, ResolvedEntity] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        logger.info(f"Initialized EntityResolver with {len(self.entity_db)} entities")

    def resolve(self, record: IngestionRecord) -> ResolvedEntity:
        """
        Resolve entity using cascading strategy.

        Args:
            record: Ingestion record to resolve

        Returns:
            ResolvedEntity with resolution result
        """
        try:
            # Check cache first
            if self.config.cache_enabled:
                cached = self._get_from_cache(record.entity_name)
                if cached:
                    logger.debug(f"Cache hit for '{record.entity_name}'")
                    return cached

            # Strategy 1: Exact ID match
            if record.entity_identifier:
                result = self._try_id_match(record)
                if result and result.confidence_score >= self.config.auto_match_threshold:
                    return self._cache_and_return(record.entity_name, result)

            # Strategy 2: Exact name match
            result = self._try_name_match(record)
            if result and result.confidence_score >= self.config.auto_match_threshold:
                return self._cache_and_return(record.entity_name, result)

            # Strategy 3: Fuzzy match
            result = self._try_fuzzy_match(record)
            if result and result.confidence_score >= self.config.auto_match_threshold:
                return self._cache_and_return(record.entity_name, result)

            # Strategy 4: MDM lookup
            result = self._try_mdm_lookup(record)
            if result and result.confidence_score >= self.config.auto_match_threshold:
                return self._cache_and_return(record.entity_name, result)

            # No high-confidence match - send to review
            return self._create_review_required(record, result)

        except Exception as e:
            logger.error(f"Entity resolution failed for '{record.entity_name}': {e}")
            raise EntityResolutionError(
                f"Resolution failed: {str(e)}",
                details={"record_id": record.record_id, "entity_name": record.entity_name}
            ) from e

    def _try_id_match(self, record: IngestionRecord) -> Optional[ResolvedEntity]:
        """Try exact ID matching."""
        candidate = self.deterministic_matcher.match_by_id(
            record.entity_identifier,
            self.entity_db
        )

        if candidate:
            logger.info(f"Exact ID match: {record.entity_identifier} -> {candidate.canonical_id}")
            return ResolvedEntity(
                record_id=record.record_id,
                entity_type=record.entity_type,
                resolved=True,
                canonical_id=candidate.canonical_id,
                canonical_name=candidate.canonical_name,
                confidence_score=candidate.confidence_score,
                resolution_method=ResolutionMethod.EXACT_MATCH,
                candidates=[candidate],
                requires_review=False
            )
        return None

    def _try_name_match(self, record: IngestionRecord) -> Optional[ResolvedEntity]:
        """Try exact name matching."""
        candidate = self.deterministic_matcher.match_by_name(
            record.entity_name,
            self.entity_db
        )

        if candidate:
            logger.info(f"Exact name match: {record.entity_name} -> {candidate.canonical_id}")
            return ResolvedEntity(
                record_id=record.record_id,
                entity_type=record.entity_type,
                resolved=True,
                canonical_id=candidate.canonical_id,
                canonical_name=candidate.canonical_name,
                confidence_score=candidate.confidence_score,
                resolution_method=ResolutionMethod.EXACT_MATCH,
                candidates=[candidate],
                requires_review=False
            )
        return None

    def _try_fuzzy_match(self, record: IngestionRecord) -> Optional[ResolvedEntity]:
        """Try fuzzy matching."""
        candidates = self.fuzzy_matcher.match(
            record.entity_name,
            self.entity_db,
            limit=5
        )

        if candidates:
            top_candidate = candidates[0]
            logger.info(
                f"Fuzzy match: {record.entity_name} -> {top_candidate.canonical_id} "
                f"(confidence: {top_candidate.confidence_score:.1f}%)"
            )

            return ResolvedEntity(
                record_id=record.record_id,
                entity_type=record.entity_type,
                resolved=True,
                canonical_id=top_candidate.canonical_id,
                canonical_name=top_candidate.canonical_name,
                confidence_score=top_candidate.confidence_score,
                resolution_method=ResolutionMethod.FUZZY_MATCH,
                candidates=candidates,
                requires_review=top_candidate.confidence_score < self.config.review_threshold
            )
        return None

    def _try_mdm_lookup(self, record: IngestionRecord) -> Optional[ResolvedEntity]:
        """Try MDM lookups (LEI, DUNS, etc)."""
        # Try LEI if available
        if "lei" in record.data:
            candidate = self.mdm_integrator.lookup_lei(record.data["lei"])
            if candidate:
                return ResolvedEntity(
                    record_id=record.record_id,
                    entity_type=record.entity_type,
                    resolved=True,
                    canonical_id=candidate.canonical_id,
                    canonical_name=candidate.canonical_name,
                    confidence_score=candidate.confidence_score,
                    resolution_method=ResolutionMethod.MDM_LOOKUP,
                    candidates=[candidate],
                    requires_review=False
                )

        # Try DUNS if available
        if "duns" in record.data:
            candidate = self.mdm_integrator.lookup_duns(record.data["duns"])
            if candidate:
                return ResolvedEntity(
                    record_id=record.record_id,
                    entity_type=record.entity_type,
                    resolved=True,
                    canonical_id=candidate.canonical_id,
                    canonical_name=candidate.canonical_name,
                    confidence_score=candidate.confidence_score,
                    resolution_method=ResolutionMethod.MDM_LOOKUP,
                    candidates=[candidate],
                    requires_review=False
                )

        return None

    def _create_review_required(
        self,
        record: IngestionRecord,
        partial_result: Optional[ResolvedEntity]
    ) -> ResolvedEntity:
        """Create resolution result that requires review."""
        candidates = partial_result.candidates if partial_result else []
        top_score = candidates[0].confidence_score if candidates else 0.0

        logger.info(
            f"Low confidence match for '{record.entity_name}' "
            f"(top score: {top_score:.1f}%) - sending to review"
        )

        return ResolvedEntity(
            record_id=record.record_id,
            entity_type=record.entity_type,
            resolved=False,
            canonical_id=None,
            canonical_name=None,
            confidence_score=top_score,
            resolution_method=ResolutionMethod.FAILED,
            candidates=candidates,
            requires_review=True,
            review_reason=f"Confidence below threshold ({top_score:.1f}% < {self.config.review_threshold}%)"
        )

    def _get_from_cache(self, entity_name: str) -> Optional[ResolvedEntity]:
        """Get resolution from cache if not expired."""
        if entity_name in self.cache:
            cached_time = self.cache_timestamps.get(entity_name)
            if cached_time:
                age = DeterministicClock.utcnow() - cached_time
                if age.total_seconds() < self.config.cache_ttl_seconds:
                    return self.cache[entity_name]
                else:
                    # Expired - remove
                    del self.cache[entity_name]
                    del self.cache_timestamps[entity_name]
        return None

    def _cache_and_return(
        self,
        entity_name: str,
        result: ResolvedEntity
    ) -> ResolvedEntity:
        """Cache resolution result and return it."""
        if self.config.cache_enabled:
            self.cache[entity_name] = result
            self.cache_timestamps[entity_name] = DeterministicClock.utcnow()
        return result

    def add_entity(self, entity_id: str, entity_data: Dict[str, Any]) -> None:
        """Add entity to resolution database."""
        self.entity_db[entity_id] = entity_data
        logger.info(f"Added entity: {entity_id}")

    def clear_cache(self) -> None:
        """Clear resolution cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Resolution cache cleared")


__all__ = ["EntityResolver"]
