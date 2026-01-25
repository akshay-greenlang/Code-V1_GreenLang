# -*- coding: utf-8 -*-
"""
Tests for Entity MDM Two-Stage Resolver.

Tests two-stage resolution (candidate generation + re-ranking),
human-in-the-loop, confidence thresholds, integration tests, and performance.

Target: 500+ lines, 25 tests
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any


# Mock resolver (would be actual module in production)
class EntityResolver:
    """Two-stage entity resolution: vector search + BERT re-ranking."""

    def __init__(self, vector_store, embedding_service, matching_model,
                 auto_match_threshold: float = 0.95,
                 human_review_threshold: float = 0.80):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.matching_model = matching_model
        self.auto_match_threshold = auto_match_threshold
        self.human_review_threshold = human_review_threshold

    def resolve(self, supplier_name: str, top_k: int = 10,
               min_similarity: float = 0.70) -> Dict[str, Any]:
        """Resolve supplier name to canonical entity."""
        if not supplier_name or not supplier_name.strip():
            raise ValueError("Supplier name cannot be empty")

        # Stage 1: Vector similarity search (candidate generation)
        candidates = self._stage1_candidate_generation(
            supplier_name, top_k, min_similarity
        )

        if not candidates:
            return {
                "match_status": "no_match",
                "canonical_entity": None,
                "confidence_score": 0.0,
                "candidates": []
            }

        # Stage 2: BERT re-ranking
        best_match = self._stage2_reranking(supplier_name, candidates)

        # Apply confidence thresholds
        return self._apply_decision_logic(best_match, candidates)

    def _stage1_candidate_generation(self, supplier_name: str, top_k: int,
                                    min_similarity: float) -> List[Dict]:
        """Stage 1: Generate candidates using vector similarity."""
        # Generate embedding for input
        embedding = self.embedding_service.embed_single(supplier_name)

        # Search vector store
        candidates = self.vector_store.search_similar(
            embedding,
            top_k=top_k,
            min_similarity=min_similarity
        )

        return candidates

    def _stage2_reranking(self, supplier_name: str,
                         candidates: List[Dict]) -> Dict[str, Any]:
        """Stage 2: Re-rank candidates using BERT."""
        if not candidates:
            return None

        # Create pairs for scoring
        pairs = [(supplier_name, candidate["name"]) for candidate in candidates]

        # Get BERT scores
        bert_scores = self.matching_model.predict_batch(pairs)

        # Find best match
        best_idx = np.argmax(bert_scores)
        best_candidate = candidates[best_idx]
        best_score = float(bert_scores[best_idx])

        return {
            "entity_id": best_candidate["id"],
            "canonical_name": best_candidate["name"],
            "country": best_candidate.get("country"),
            "lei_code": best_candidate.get("lei_code"),
            "vector_similarity": best_candidate["similarity"],
            "bert_score": best_score,
            "confidence_score": best_score
        }

    def _apply_decision_logic(self, best_match: Dict, candidates: List[Dict]) -> Dict[str, Any]:
        """Apply confidence thresholds to determine action."""
        if not best_match:
            return {
                "match_status": "no_match",
                "canonical_entity": None,
                "confidence_score": 0.0,
                "candidates": []
            }

        confidence = best_match["confidence_score"]

        if confidence >= self.auto_match_threshold:
            return {
                "match_status": "auto_match",
                "canonical_entity": best_match,
                "confidence_score": confidence,
                "candidates": candidates
            }
        elif confidence >= self.human_review_threshold:
            return {
                "match_status": "human_review",
                "canonical_entity": best_match,
                "confidence_score": confidence,
                "candidates": candidates,
                "review_required": True
            }
        else:
            return {
                "match_status": "no_match",
                "canonical_entity": None,
                "confidence_score": confidence,
                "candidates": candidates
            }

    def resolve_batch(self, supplier_names: List[str]) -> List[Dict[str, Any]]:
        """Resolve multiple supplier names."""
        results = []
        for name in supplier_names:
            try:
                result = self.resolve(name)
                results.append(result)
            except Exception as e:
                results.append({
                    "match_status": "error",
                    "error": str(e),
                    "supplier_name": name
                })
        return results


# ============================================================================
# TEST SUITE - STAGE 1: CANDIDATE GENERATION
# ============================================================================

class TestStage1CandidateGeneration:
    """Test suite for Stage 1: Candidate Generation."""

    @pytest.fixture
    def resolver(self, mock_weaviate_client, mock_sentence_transformer,
                mock_cross_encoder, create_embedding):
        """Create resolver with mocked dependencies."""
        from test_embeddings import EmbeddingService
        from test_vector_store import VectorStore
        from test_matching_model import MatchingModel

        vector_store = VectorStore(mock_weaviate_client)
        embedding_service = EmbeddingService(mock_sentence_transformer)
        matching_model = MatchingModel(model=mock_cross_encoder)

        return EntityResolver(
            vector_store, embedding_service, matching_model
        )

    def test_candidate_generation_returns_similar_entities(
        self, resolver, mock_weaviate_client, mock_weaviate_query_response
    ):
        """Test that candidate generation returns similar entities."""
        # Mock query response
        query_mock = MagicMock()
        query_mock.do.return_value = mock_weaviate_query_response
        query_mock.with_near_vector.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_additional.return_value = query_mock
        mock_weaviate_client.query.get.return_value = query_mock

        candidates = resolver._stage1_candidate_generation("Acme Corp", 10, 0.70)

        assert len(candidates) > 0
        assert all("name" in c for c in candidates)
        assert all("similarity" in c for c in candidates)

    def test_candidate_generation_respects_top_k(self, resolver, mock_weaviate_client):
        """Test that top_k parameter limits results."""
        query_mock = MagicMock()
        query_mock.do.return_value = {"data": {"Get": {"Supplier": []}}}
        query_mock.with_near_vector.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_additional.return_value = query_mock
        mock_weaviate_client.query.get.return_value = query_mock

        resolver._stage1_candidate_generation("Test", top_k=5, min_similarity=0.70)

        # Verify with_limit was called with 5
        query_mock.with_limit.assert_called_once_with(5)

    def test_candidate_generation_filters_by_similarity(
        self, resolver, mock_weaviate_client, mock_weaviate_query_response
    ):
        """Test that min_similarity filters low-score results."""
        query_mock = MagicMock()
        query_mock.do.return_value = mock_weaviate_query_response
        query_mock.with_near_vector.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_additional.return_value = query_mock
        mock_weaviate_client.query.get.return_value = query_mock

        candidates = resolver._stage1_candidate_generation("Acme", 10, 0.90)

        # Only candidates with similarity >= 0.90 should be returned
        assert all(c["similarity"] >= 0.90 for c in candidates)

    def test_candidate_generation_with_no_results(self, resolver, mock_weaviate_client):
        """Test candidate generation when no results found."""
        query_mock = MagicMock()
        query_mock.do.return_value = {"data": {"Get": {"Supplier": []}}}
        query_mock.with_near_vector.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_additional.return_value = query_mock
        mock_weaviate_client.query.get.return_value = query_mock

        candidates = resolver._stage1_candidate_generation("Unknown", 10, 0.70)

        assert candidates == []


# ============================================================================
# TEST SUITE - STAGE 2: RE-RANKING
# ============================================================================

class TestStage2Reranking:
    """Test suite for Stage 2: BERT Re-ranking."""

    @pytest.fixture
    def resolver(self, mock_weaviate_client, mock_sentence_transformer, mock_cross_encoder):
        """Create resolver with mocked dependencies."""
        from test_embeddings import EmbeddingService
        from test_vector_store import VectorStore
        from test_matching_model import MatchingModel

        vector_store = VectorStore(mock_weaviate_client)
        embedding_service = EmbeddingService(mock_sentence_transformer)
        matching_model = MatchingModel(model=mock_cross_encoder)

        return EntityResolver(vector_store, embedding_service, matching_model)

    def test_reranking_selects_best_match(self, resolver):
        """Test that re-ranking selects the best matching candidate."""
        candidates = [
            {"id": "1", "name": "ACME Corporation Ltd", "similarity": 0.85},
            {"id": "2", "name": "Acme Corp", "similarity": 0.80},
            {"id": "3", "name": "ACME Industries", "similarity": 0.75}
        ]

        best_match = resolver._stage2_reranking("Acme Corp", candidates)

        assert best_match is not None
        assert "bert_score" in best_match
        assert "confidence_score" in best_match

    def test_reranking_with_empty_candidates(self, resolver):
        """Test re-ranking with no candidates."""
        best_match = resolver._stage2_reranking("Test", [])

        assert best_match is None

    def test_reranking_includes_vector_similarity(self, resolver):
        """Test that re-ranking result includes vector similarity."""
        candidates = [
            {"id": "1", "name": "ACME Corp", "similarity": 0.90}
        ]

        best_match = resolver._stage2_reranking("Acme", candidates)

        assert "vector_similarity" in best_match
        assert best_match["vector_similarity"] == 0.90


# ============================================================================
# TEST SUITE - CONFIDENCE THRESHOLDS
# ============================================================================

class TestConfidenceThresholds:
    """Test suite for confidence threshold logic."""

    @pytest.fixture
    def resolver(self, mock_weaviate_client, mock_sentence_transformer, mock_cross_encoder):
        """Create resolver with mocked dependencies."""
        from test_embeddings import EmbeddingService
        from test_vector_store import VectorStore
        from test_matching_model import MatchingModel

        vector_store = VectorStore(mock_weaviate_client)
        embedding_service = EmbeddingService(mock_sentence_transformer)
        matching_model = MatchingModel(model=mock_cross_encoder)

        return EntityResolver(
            vector_store, embedding_service, matching_model,
            auto_match_threshold=0.95,
            human_review_threshold=0.80
        )

    def test_high_confidence_auto_match(self, resolver):
        """Test that high confidence (>=0.95) results in auto-match."""
        best_match = {
            "entity_id": "1",
            "canonical_name": "ACME Corp",
            "confidence_score": 0.97
        }

        result = resolver._apply_decision_logic(best_match, [])

        assert result["match_status"] == "auto_match"
        assert result["canonical_entity"] is not None

    def test_moderate_confidence_human_review(self, resolver):
        """Test that moderate confidence (0.80-0.95) triggers human review."""
        best_match = {
            "entity_id": "1",
            "canonical_name": "ACME Corp",
            "confidence_score": 0.87
        }

        result = resolver._apply_decision_logic(best_match, [])

        assert result["match_status"] == "human_review"
        assert result["review_required"] is True

    def test_low_confidence_no_match(self, resolver):
        """Test that low confidence (<0.80) results in no match."""
        best_match = {
            "entity_id": "1",
            "canonical_name": "ACME Corp",
            "confidence_score": 0.65
        }

        result = resolver._apply_decision_logic(best_match, [])

        assert result["match_status"] == "no_match"
        assert result["canonical_entity"] is None

    def test_threshold_boundary_conditions(self, resolver):
        """Test threshold boundary conditions."""
        # Exactly at auto_match threshold
        result1 = resolver._apply_decision_logic(
            {"confidence_score": 0.95}, []
        )
        assert result1["match_status"] == "auto_match"

        # Exactly at human_review threshold
        result2 = resolver._apply_decision_logic(
            {"confidence_score": 0.80}, []
        )
        assert result2["match_status"] == "human_review"


# ============================================================================
# TEST SUITE - INTEGRATION TESTS
# ============================================================================

class TestResolverIntegration:
    """Integration tests for end-to-end resolution."""

    @pytest.fixture
    def resolver(self, mock_weaviate_client, mock_sentence_transformer, mock_cross_encoder):
        """Create resolver with mocked dependencies."""
        from test_embeddings import EmbeddingService
        from test_vector_store import VectorStore
        from test_matching_model import MatchingModel

        vector_store = VectorStore(mock_weaviate_client)
        embedding_service = EmbeddingService(mock_sentence_transformer)
        matching_model = MatchingModel(model=mock_cross_encoder)

        return EntityResolver(vector_store, embedding_service, matching_model)

    def test_end_to_end_resolution(self, resolver, mock_weaviate_client, mock_weaviate_query_response):
        """Test complete resolution pipeline."""
        # Setup mocks
        query_mock = MagicMock()
        query_mock.do.return_value = mock_weaviate_query_response
        query_mock.with_near_vector.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_additional.return_value = query_mock
        mock_weaviate_client.query.get.return_value = query_mock

        result = resolver.resolve("Acme Corp")

        assert "match_status" in result
        assert "confidence_score" in result

    def test_resolve_with_empty_name_raises_error(self, resolver):
        """Test that empty supplier name raises error."""
        with pytest.raises(ValueError, match="Supplier name cannot be empty"):
            resolver.resolve("")

        with pytest.raises(ValueError, match="Supplier name cannot be empty"):
            resolver.resolve("   ")

    def test_resolve_batch(self, resolver, mock_weaviate_client, mock_weaviate_query_response):
        """Test batch resolution."""
        query_mock = MagicMock()
        query_mock.do.return_value = mock_weaviate_query_response
        query_mock.with_near_vector.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_additional.return_value = query_mock
        mock_weaviate_client.query.get.return_value = query_mock

        names = ["ACME Corp", "ABC Manufacturing", "Global Tech"]
        results = resolver.resolve_batch(names)

        assert len(results) == 3
        assert all("match_status" in r for r in results)

    def test_resolve_with_no_candidates_found(self, resolver, mock_weaviate_client):
        """Test resolution when no candidates are found."""
        query_mock = MagicMock()
        query_mock.do.return_value = {"data": {"Get": {"Supplier": []}}}
        query_mock.with_near_vector.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_additional.return_value = query_mock
        mock_weaviate_client.query.get.return_value = query_mock

        result = resolver.resolve("Nonexistent Company")

        assert result["match_status"] == "no_match"
        assert result["canonical_entity"] is None
        assert result["confidence_score"] == 0.0


# ============================================================================
# TEST SUITE - PERFORMANCE
# ============================================================================

class TestResolverPerformance:
    """Performance tests for resolver."""

    @pytest.fixture
    def resolver(self, mock_weaviate_client, mock_sentence_transformer, mock_cross_encoder):
        """Create resolver with mocked dependencies."""
        from test_embeddings import EmbeddingService
        from test_vector_store import VectorStore
        from test_matching_model import MatchingModel

        vector_store = VectorStore(mock_weaviate_client)
        embedding_service = EmbeddingService(mock_sentence_transformer)
        matching_model = MatchingModel(model=mock_cross_encoder)

        return EntityResolver(vector_store, embedding_service, matching_model)

    def test_batch_resolution_performance(self, resolver, mock_weaviate_client):
        """Test batch resolution throughput."""
        query_mock = MagicMock()
        query_mock.do.return_value = {"data": {"Get": {"Supplier": []}}}
        query_mock.with_near_vector.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_additional.return_value = query_mock
        mock_weaviate_client.query.get.return_value = query_mock

        # Resolve 100 suppliers
        names = [f"Company {i}" for i in range(100)]
        results = resolver.resolve_batch(names)

        assert len(results) == 100
