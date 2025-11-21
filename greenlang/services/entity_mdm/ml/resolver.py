# -*- coding: utf-8 -*-
"""
Two-stage entity resolution pipeline with human-in-the-loop.

This module implements the complete entity resolution workflow:
1. Candidate generation via vector similarity search
2. Re-ranking with BERT-based matching
3. Human review queue for low-confidence matches

Author: GreenLang AI
Phase: 5 - Entity Resolution ML
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import uuid

from entity_mdm.ml.config import MLConfig, ResolutionConfig
from entity_mdm.ml.embeddings import EmbeddingPipeline
from entity_mdm.ml.vector_store import VectorStore, SupplierEntity
from entity_mdm.ml.matching_model import MatchingModel
from entity_mdm.ml.exceptions import (
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock
    InsufficientCandidatesException,
    ModelNotTrainedException,
)

logger = logging.getLogger(__name__)


class ResolutionStatus(str, Enum):
    """Status of entity resolution."""

    AUTO_MATCHED = "auto_matched"  # High confidence auto-match
    PENDING_REVIEW = "pending_review"  # Low confidence, needs human review
    NO_MATCH = "no_match"  # No suitable candidates found
    REVIEWED = "reviewed"  # Human reviewed


@dataclass
class Candidate:
    """Candidate entity from similarity search."""

    entity: SupplierEntity
    similarity_score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity.entity_id,
            "name": self.entity.name,
            "normalized_name": self.entity.normalized_name,
            "similarity_score": self.similarity_score,
            "rank": self.rank,
        }


@dataclass
class MatchResult:
    """Result of entity matching."""

    matched_entity_id: Optional[str]
    confidence: float
    rank_score: Optional[float]  # BERT re-ranking score
    status: ResolutionStatus
    candidates: List[Candidate]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "matched_entity_id": self.matched_entity_id,
            "confidence": self.confidence,
            "rank_score": self.rank_score,
            "status": self.status.value,
            "num_candidates": len(self.candidates),
            "candidates": [c.to_dict() for c in self.candidates],
            "metadata": self.metadata,
        }


@dataclass
class ReviewItem:
    """Item in human review queue."""

    review_id: str
    query_entity: SupplierEntity
    match_result: MatchResult
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewer: Optional[str] = None
    decision: Optional[str] = None  # "accept", "reject", "create_new"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "review_id": self.review_id,
            "query_entity": {
                "entity_id": self.query_entity.entity_id,
                "name": self.query_entity.name,
            },
            "match_result": self.match_result.to_dict(),
            "created_at": self.created_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewer": self.reviewer,
            "decision": self.decision,
            "notes": self.notes,
        }


class EntityResolver:
    """
    Two-stage entity resolution with human-in-the-loop.

    This class implements:
    - Stage 1: Fast candidate generation via vector search
    - Stage 2: Precision re-ranking with BERT matching
    - Human review queue management
    - Confidence-based routing
    """

    def __init__(
        self,
        config: Optional[MLConfig] = None,
        vector_store: Optional[VectorStore] = None,
        matching_model: Optional[MatchingModel] = None,
    ) -> None:
        """
        Initialize entity resolver.

        Args:
            config: ML configuration object
            vector_store: Vector store instance. If None, creates new one.
            matching_model: Matching model instance. If None, creates new one.
        """
        self.config = config or MLConfig()
        self.resolution_config: ResolutionConfig = self.config.resolution

        # Initialize components
        self.embedding_pipeline = EmbeddingPipeline(self.config)
        self.vector_store = vector_store or VectorStore(
            self.config,
            self.embedding_pipeline,
        )
        self.matching_model = matching_model or MatchingModel(self.config)

        # Review queue (in-memory for now, should use persistent storage)
        self._review_queue: Dict[str, ReviewItem] = {}

        # Statistics
        self._stats = {
            "total_resolutions": 0,
            "auto_matched": 0,
            "pending_review": 0,
            "no_match": 0,
            "reviewed": 0,
        }

        logger.info("Initialized EntityResolver")

    def resolve(
        self,
        query_entity: SupplierEntity,
        auto_match: bool = True,
    ) -> MatchResult:
        """
        Resolve an entity using two-stage pipeline.

        Args:
            query_entity: Entity to resolve
            auto_match: Whether to enable auto-matching for high confidence results

        Returns:
            Match result with status and candidates

        Raises:
            ModelNotTrainedException: If matching model not trained
        """
        self._stats["total_resolutions"] += 1

        # Stage 1: Generate candidates via vector search
        candidates = self.generate_candidates(query_entity)

        if not candidates:
            # No candidates found
            result = MatchResult(
                matched_entity_id=None,
                confidence=0.0,
                rank_score=None,
                status=ResolutionStatus.NO_MATCH,
                candidates=[],
                metadata={"reason": "no_candidates"},
            )
            self._stats["no_match"] += 1
            return result

        # Stage 2: Re-rank candidates with BERT
        ranked_candidates = self.rerank_candidates(query_entity, candidates)

        # Get top candidate
        top_candidate = ranked_candidates[0]
        confidence = top_candidate.similarity_score

        # Determine status based on confidence thresholds
        if (
            auto_match
            and confidence >= self.resolution_config.auto_match_threshold
        ):
            # High confidence - auto-match
            status = ResolutionStatus.AUTO_MATCHED
            matched_entity_id = top_candidate.entity.entity_id
            self._stats["auto_matched"] += 1
        elif confidence >= self.resolution_config.human_review_threshold:
            # Medium confidence - needs review
            status = ResolutionStatus.PENDING_REVIEW
            matched_entity_id = top_candidate.entity.entity_id
            self._stats["pending_review"] += 1
        else:
            # Low confidence - no match
            status = ResolutionStatus.NO_MATCH
            matched_entity_id = None
            self._stats["no_match"] += 1

        result = MatchResult(
            matched_entity_id=matched_entity_id,
            confidence=confidence,
            rank_score=confidence,  # After reranking, this is the BERT score
            status=status,
            candidates=ranked_candidates,
            metadata={
                "num_candidates": len(candidates),
                "threshold_auto": self.resolution_config.auto_match_threshold,
                "threshold_review": self.resolution_config.human_review_threshold,
            },
        )

        # Add to review queue if needed
        if status == ResolutionStatus.PENDING_REVIEW:
            self.add_to_review_queue(query_entity, result)

        return result

    def generate_candidates(
        self,
        query_entity: SupplierEntity,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Candidate]:
        """
        Stage 1: Generate candidates via vector similarity search.

        Args:
            query_entity: Entity to search for
            limit: Maximum candidates to return (uses config default if None)
            threshold: Minimum similarity threshold (uses config default if None)

        Returns:
            List of candidate entities sorted by similarity

        Raises:
            InsufficientCandidatesException: If too few candidates found and strict mode
        """
        limit = limit or self.resolution_config.candidate_limit
        threshold = threshold or self.resolution_config.candidate_threshold

        # Search vector store
        results = self.vector_store.search(
            query_entity,
            limit=limit,
            threshold=threshold,
        )

        # Convert to candidates
        candidates = [
            Candidate(
                entity=entity,
                similarity_score=score,
                rank=i + 1,
            )
            for i, (entity, score) in enumerate(results)
        ]

        logger.debug(
            f"Generated {len(candidates)} candidates for entity {query_entity.entity_id}"
        )

        return candidates

    def rerank_candidates(
        self,
        query_entity: SupplierEntity,
        candidates: List[Candidate],
    ) -> List[Candidate]:
        """
        Stage 2: Re-rank candidates using BERT matching model.

        Args:
            query_entity: Query entity
            candidates: Candidates from stage 1

        Returns:
            Re-ranked candidates sorted by BERT confidence

        Raises:
            ModelNotTrainedException: If matching model not trained
        """
        if not self.matching_model.is_trained:
            raise ModelNotTrainedException("MatchingModel")

        if not candidates:
            return []

        # Prepare pairs for batch prediction
        query_text = query_entity.get_search_text()
        pairs = [
            (query_text, candidate.entity.get_search_text())
            for candidate in candidates
        ]

        # Get BERT predictions
        predictions = self.matching_model.predict_batch(pairs)

        # Update candidates with BERT scores
        for candidate, (pred, conf) in zip(candidates, predictions):
            # Use confidence of "match" prediction
            candidate.similarity_score = conf if pred == 1 else (1.0 - conf)

        # Re-sort by new scores
        ranked = sorted(
            candidates,
            key=lambda c: c.similarity_score,
            reverse=True,
        )

        # Update ranks
        for i, candidate in enumerate(ranked):
            candidate.rank = i + 1

        logger.debug(
            f"Re-ranked {len(candidates)} candidates. "
            f"Top score: {ranked[0].similarity_score:.3f}"
        )

        return ranked

    def add_to_review_queue(
        self,
        query_entity: SupplierEntity,
        match_result: MatchResult,
    ) -> str:
        """
        Add a match to the human review queue.

        Args:
            query_entity: Query entity
            match_result: Match result to review

        Returns:
            Review ID
        """
        review_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        review_item = ReviewItem(
            review_id=review_id,
            query_entity=query_entity,
            match_result=match_result,
            created_at=DeterministicClock.utcnow(),
        )

        self._review_queue[review_id] = review_item
        logger.info(f"Added review item {review_id} to queue")

        return review_id

    def get_review_queue(
        self,
        limit: Optional[int] = None,
    ) -> List[ReviewItem]:
        """
        Get pending review items.

        Args:
            limit: Maximum items to return

        Returns:
            List of review items sorted by creation time
        """
        pending = [
            item
            for item in self._review_queue.values()
            if item.reviewed_at is None
        ]

        # Sort by creation time
        pending.sort(key=lambda x: x.created_at)

        if limit:
            pending = pending[:limit]

        return pending

    def submit_review(
        self,
        review_id: str,
        decision: str,
        reviewer: str,
        notes: Optional[str] = None,
    ) -> ReviewItem:
        """
        Submit a human review decision.

        Args:
            review_id: Review item ID
            decision: Decision ("accept", "reject", "create_new")
            reviewer: Name/ID of reviewer
            notes: Optional review notes

        Returns:
            Updated review item

        Raises:
            ValueError: If review_id not found or invalid decision
        """
        if review_id not in self._review_queue:
            raise ValueError(f"Review ID {review_id} not found")

        valid_decisions = ["accept", "reject", "create_new"]
        if decision not in valid_decisions:
            raise ValueError(f"Decision must be one of {valid_decisions}")

        review_item = self._review_queue[review_id]
        review_item.reviewed_at = DeterministicClock.utcnow()
        review_item.reviewer = reviewer
        review_item.decision = decision
        review_item.notes = notes

        # Update status
        review_item.match_result.status = ResolutionStatus.REVIEWED
        self._stats["reviewed"] += 1
        self._stats["pending_review"] -= 1

        logger.info(
            f"Review {review_id} submitted by {reviewer}: {decision}"
        )

        return review_item

    def batch_resolve(
        self,
        query_entities: List[SupplierEntity],
        show_progress: bool = True,
    ) -> List[MatchResult]:
        """
        Resolve multiple entities in batch.

        Args:
            query_entities: List of entities to resolve
            show_progress: Show progress logging

        Returns:
            List of match results
        """
        results = []

        for i, entity in enumerate(query_entities):
            result = self.resolve(entity)
            results.append(result)

            if show_progress and (i + 1) % 100 == 0:
                logger.info(f"Resolved {i + 1}/{len(query_entities)} entities")

        if show_progress:
            logger.info(f"Batch resolution complete: {len(results)} entities")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get resolver statistics.

        Returns:
            Dictionary with statistics
        """
        stats = self._stats.copy()

        # Calculate rates
        total = stats["total_resolutions"]
        if total > 0:
            stats["auto_match_rate"] = stats["auto_matched"] / total
            stats["review_rate"] = stats["pending_review"] / total
            stats["no_match_rate"] = stats["no_match"] / total

        stats["review_queue_size"] = len(self.get_review_queue())

        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "total_resolutions": 0,
            "auto_matched": 0,
            "pending_review": 0,
            "no_match": 0,
            "reviewed": 0,
        }

    def close(self) -> None:
        """Cleanup resources."""
        if self.vector_store:
            self.vector_store.close()
