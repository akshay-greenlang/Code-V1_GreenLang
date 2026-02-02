# -*- coding: utf-8 -*-
"""
GL-CSRD-APP - Infrastructure Test Suite
========================================

Comprehensive tests for CSRD infrastructure components:
- ChatSession integration
- RAG engine retrieval
- Semantic caching (30% reduction target)
- Agent framework lifecycle
- Validation framework
- Telemetry collection

Version: 1.0.0
Author: Testing & QA Team
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from infrastructure.chatsession import ChatSession, ChatMessage
except ImportError:
    ChatSession = None
    ChatMessage = None

try:
    from infrastructure.rag_engine import RAGEngine, DocumentStore
except ImportError:
    RAGEngine = None
    DocumentStore = None

try:
    from infrastructure.semantic_cache import SemanticCache
except ImportError:
    SemanticCache = None

try:
    from infrastructure.agent_framework import AgentFramework, AgentLifecycle
except ImportError:
    AgentFramework = None
    AgentLifecycle = None

try:
    from infrastructure.validation_framework import ValidationFramework, ValidationRule
except ImportError:
    ValidationFramework = None
    ValidationRule = None

try:
    from infrastructure.telemetry import TelemetryCollector, MetricsCollector
except ImportError:
    TelemetryCollector = None
    MetricsCollector = None


# ============================================================================
# ChatSession Integration Tests
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.skipif(ChatSession is None, reason="ChatSession not available")
class TestChatSessionIntegration:
    """Test ChatSession infrastructure integration."""

    def test_chatsession_integration(self):
        """
        Test ChatSession creates and manages conversation context.

        Requirements:
        - Create session with unique ID
        - Store messages with metadata
        - Retrieve conversation history
        - Support context management
        """
        session = ChatSession(session_id="test-session-001")

        # Test session creation
        assert session.session_id == "test-session-001"
        assert session.message_count == 0

        # Test adding messages
        session.add_message(
            role="user",
            content="What are CSRD reporting requirements?",
            metadata={"timestamp": time.time()}
        )

        session.add_message(
            role="assistant",
            content="CSRD requires companies to report on ESG factors...",
            metadata={"timestamp": time.time(), "confidence": 0.95}
        )

        # Test message retrieval
        assert session.message_count == 2
        history = session.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_chatsession_context_window(self):
        """Test ChatSession respects context window limits."""
        session = ChatSession(session_id="test-context", max_messages=5)

        # Add more messages than limit
        for i in range(10):
            session.add_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}"
            )

        # Should only keep last 5 messages
        history = session.get_history()
        assert len(history) <= 5

    def test_chatsession_metadata_tracking(self):
        """Test ChatSession tracks metadata correctly."""
        session = ChatSession(session_id="test-metadata")

        session.add_message(
            role="user",
            content="Test question",
            metadata={
                "user_id": "user123",
                "channel": "web",
                "timestamp": time.time()
            }
        )

        history = session.get_history()
        assert history[0]["metadata"]["user_id"] == "user123"
        assert history[0]["metadata"]["channel"] == "web"

    def test_chatsession_persistence(self, tmp_path):
        """Test ChatSession can be saved and loaded."""
        session = ChatSession(session_id="test-persist")
        session.add_message(role="user", content="Test message 1")
        session.add_message(role="assistant", content="Test response 1")

        # Save session
        save_path = tmp_path / "session.json"
        session.save(save_path)

        # Load session
        loaded_session = ChatSession.load(save_path)
        assert loaded_session.session_id == session.session_id
        assert loaded_session.message_count == session.message_count


# ============================================================================
# RAG Engine Retrieval Tests
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.skipif(RAGEngine is None, reason="RAGEngine not available")
class TestRAGEngineRetrieval:
    """Test RAG engine retrieval capabilities."""

    def test_rag_engine_retrieval(self, sample_documents):
        """
        Test RAG engine retrieves relevant documents.

        Requirements:
        - Index documents with embeddings
        - Retrieve top-k relevant documents
        - Return relevance scores
        - Support filtering by metadata
        """
        rag_engine = RAGEngine(embedding_model="sentence-transformers/all-MiniLM-L6-v2")

        # Index sample documents
        for doc in sample_documents:
            rag_engine.index_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"]
            )

        # Test retrieval
        query = "What are the materiality assessment requirements?"
        results = rag_engine.retrieve(query, top_k=3)

        # Verify results
        assert len(results) <= 3
        assert all("score" in r for r in results)
        assert all("content" in r for r in results)
        assert all("metadata" in r for r in results)

        # Scores should be in descending order
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rag_engine_metadata_filtering(self, sample_documents):
        """Test RAG engine supports metadata filtering."""
        rag_engine = RAGEngine()

        for doc in sample_documents:
            rag_engine.index_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"]
            )

        # Retrieve with metadata filter
        query = "reporting requirements"
        results = rag_engine.retrieve(
            query,
            top_k=5,
            filter_metadata={"category": "environmental"}
        )

        # All results should match filter
        for result in results:
            assert result["metadata"]["category"] == "environmental"

    def test_rag_engine_reranking(self, sample_documents):
        """Test RAG engine supports result re-ranking."""
        rag_engine = RAGEngine(use_reranking=True)

        for doc in sample_documents:
            rag_engine.index_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"]
            )

        query = "carbon emissions reporting"
        results_with_rerank = rag_engine.retrieve(query, top_k=3, rerank=True)
        results_without_rerank = rag_engine.retrieve(query, top_k=3, rerank=False)

        # Both should return results
        assert len(results_with_rerank) > 0
        assert len(results_without_rerank) > 0

        # Re-ranking may change order
        # (can't assert they're different as it depends on content)


# ============================================================================
# Semantic Caching Tests
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.critical
@pytest.mark.skipif(SemanticCache is None, reason="SemanticCache not available")
class TestSemanticCaching:
    """Test semantic caching for 30% LLM cost reduction."""

    def test_semantic_caching(self):
        """
        Test semantic cache reduces LLM calls by 30%.

        Requirements:
        - Cache LLM responses with semantic keys
        - Retrieve cached responses for similar queries
        - Measure cache hit rate
        - Verify 30% reduction target
        """
        cache = SemanticCache(similarity_threshold=0.85)

        # Simulate LLM calls
        original_query = "What are the CSRD materiality requirements?"
        response = "CSRD requires companies to conduct double materiality assessment..."

        # Cache the response
        cache.set(original_query, response)

        # Test exact match retrieval
        cached = cache.get(original_query)
        assert cached == response

        # Test semantic similarity retrieval
        similar_query = "What does CSRD require for materiality assessment?"
        cached_similar = cache.get(similar_query)

        # Should retrieve cached response for similar query
        assert cached_similar is not None
        assert cached_similar == response

    def test_semantic_cache_hit_rate(self):
        """
        Verify semantic cache achieves 30% hit rate target.

        Test with realistic query patterns.
        """
        cache = SemanticCache(similarity_threshold=0.85)

        # Simulate training set of queries and responses
        training_queries = [
            ("What are CSRD reporting requirements?", "Response 1"),
            ("How to conduct materiality assessment?", "Response 2"),
            ("What are Scope 3 emissions?", "Response 3"),
            ("How to calculate carbon footprint?", "Response 4"),
            ("What is XBRL format?", "Response 5"),
        ]

        for query, response in training_queries:
            cache.set(query, response)

        # Simulate test queries (some similar, some new)
        test_queries = [
            "What does CSRD require for reporting?",  # Similar to #1
            "How do I perform materiality assessment?",  # Similar to #2
            "What are the greenhouse gas emission categories?",  # New
            "How to measure carbon emissions?",  # Similar to #4
            "What is the ESRS disclosure format?",  # New
            "What are Scope 1 and Scope 2 emissions?",  # New
            "CSRD reporting obligations",  # Similar to #1
            "Materiality analysis process",  # Similar to #2
            "What is taxonomy alignment?",  # New
            "How to report in XBRL?",  # Similar to #5
        ]

        hits = 0
        misses = 0

        for query in test_queries:
            result = cache.get(query)
            if result is not None:
                hits += 1
            else:
                misses += 1

        # Calculate hit rate
        hit_rate = hits / (hits + misses)

        # Should achieve at least 30% hit rate
        assert hit_rate >= 0.30, \
            f"Cache hit rate {hit_rate:.1%} below 30% target"

        print(f"  ✓ Cache hit rate: {hit_rate:.1%} (target: 30%)")
        print(f"    Hits: {hits}, Misses: {misses}")

    def test_semantic_cache_ttl(self):
        """Test semantic cache respects TTL (time-to-live)."""
        cache = SemanticCache(ttl_seconds=2)

        cache.set("test query", "test response")

        # Should retrieve immediately
        assert cache.get("test query") == "test response"

        # Wait for TTL to expire
        time.sleep(2.5)

        # Should not retrieve after TTL
        assert cache.get("test query") is None

    def test_semantic_cache_memory_efficiency(self):
        """Test semantic cache maintains reasonable memory usage."""
        cache = SemanticCache(max_cache_size=100)

        # Add more entries than max size
        for i in range(150):
            cache.set(f"query {i}", f"response {i}")

        # Cache size should not exceed limit
        assert cache.size <= 100


# ============================================================================
# Agent Framework Lifecycle Tests
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.skipif(AgentFramework is None, reason="AgentFramework not available")
class TestAgentFrameworkLifecycle:
    """Test agent framework lifecycle management."""

    def test_agent_framework_lifecycle(self):
        """
        Test agent framework manages agent lifecycle.

        Lifecycle stages: Initialize -> Execute -> Finalize
        """
        framework = AgentFramework()

        # Create mock agent
        agent_config = {
            "name": "test_agent",
            "type": "intake",
            "version": "1.0"
        }

        agent = framework.create_agent(agent_config)

        # Test initialization
        assert agent is not None
        assert agent.name == "test_agent"
        assert agent.state == "initialized"

        # Test execution
        result = framework.execute_agent(agent, {"test": "data"})
        assert result is not None
        assert agent.state == "executed"

        # Test finalization
        framework.finalize_agent(agent)
        assert agent.state == "finalized"

    def test_agent_framework_error_handling(self):
        """Test agent framework handles errors gracefully."""
        framework = AgentFramework()

        agent_config = {"name": "failing_agent", "type": "test"}

        # Simulate agent that raises error
        agent = framework.create_agent(agent_config)

        with patch.object(agent, 'execute', side_effect=Exception("Test error")):
            # Framework should catch and handle error
            result = framework.execute_agent(agent, {})

            # Should return error information
            assert result is not None
            assert "error" in result or result.get("status") == "failed"

    def test_agent_framework_metrics_collection(self):
        """Test agent framework collects execution metrics."""
        framework = AgentFramework(collect_metrics=True)

        agent_config = {"name": "test_agent", "type": "intake"}
        agent = framework.create_agent(agent_config)

        result = framework.execute_agent(agent, {"test": "data"})

        # Should have metrics
        metrics = framework.get_metrics(agent.name)
        assert metrics is not None
        assert "execution_time" in metrics
        assert "execution_count" in metrics


# ============================================================================
# Validation Framework Tests
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.skipif(ValidationFramework is None, reason="ValidationFramework not available")
class TestValidationFramework:
    """Test validation framework for data validation."""

    def test_validation_framework(self):
        """
        Test validation framework validates data against schemas.

        Requirements:
        - Define validation rules
        - Validate data against rules
        - Return detailed error messages
        - Support custom validators
        """
        validator = ValidationFramework()

        # Define schema
        schema = {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "minLength": 1},
                "emissions": {"type": "number", "minimum": 0},
                "reporting_year": {"type": "integer", "minimum": 2020}
            },
            "required": ["company_name", "emissions"]
        }

        # Test valid data
        valid_data = {
            "company_name": "Test Corp",
            "emissions": 1234.5,
            "reporting_year": 2024
        }

        result = validator.validate(valid_data, schema)
        assert result.is_valid
        assert len(result.errors) == 0

        # Test invalid data
        invalid_data = {
            "company_name": "",  # Too short
            "emissions": -100,  # Negative
        }

        result = validator.validate(invalid_data, schema)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validation_framework_custom_rules(self):
        """Test validation framework supports custom validation rules."""
        validator = ValidationFramework()

        # Add custom rule
        def validate_emission_range(value):
            """Emissions must be between 0 and 1,000,000."""
            return 0 <= value <= 1_000_000

        validator.add_custom_rule("emission_range", validate_emission_range)

        # Test with custom rule
        data = {"emissions": 500_000}
        result = validator.validate_custom(data, ["emission_range"])
        assert result.is_valid

        data = {"emissions": 2_000_000}
        result = validator.validate_custom(data, ["emission_range"])
        assert not result.is_valid


# ============================================================================
# Telemetry Collection Tests
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.skipif(TelemetryCollector is None, reason="TelemetryCollector not available")
class TestTelemetryCollection:
    """Test telemetry and metrics collection."""

    def test_telemetry_collection(self):
        """
        Test telemetry collector tracks metrics.

        Requirements:
        - Collect performance metrics
        - Track error rates
        - Monitor resource usage
        - Export to monitoring systems
        """
        telemetry = TelemetryCollector()

        # Collect metrics
        telemetry.record_metric("agent_execution_time", 1.5, {"agent": "intake"})
        telemetry.record_metric("llm_tokens_used", 1500, {"model": "gpt-4"})
        telemetry.record_metric("cache_hit_rate", 0.35, {"cache_type": "semantic"})

        # Test metric retrieval
        metrics = telemetry.get_metrics()
        assert len(metrics) > 0

        # Test metric aggregation
        avg_time = telemetry.get_average("agent_execution_time")
        assert avg_time > 0

    def test_telemetry_error_tracking(self):
        """Test telemetry tracks errors and exceptions."""
        telemetry = TelemetryCollector()

        # Record errors
        telemetry.record_error(
            error_type="ValidationError",
            message="Invalid CN code format",
            context={"agent": "intake", "record_id": "12345"}
        )

        telemetry.record_error(
            error_type="CalculationError",
            message="Missing emission factor",
            context={"agent": "calculator"}
        )

        # Test error retrieval
        errors = telemetry.get_errors()
        assert len(errors) >= 2

        # Test error rate calculation
        error_rate = telemetry.get_error_rate()
        assert error_rate >= 0

    def test_telemetry_prometheus_export(self):
        """Test telemetry can export to Prometheus format."""
        telemetry = TelemetryCollector()

        telemetry.record_metric("test_counter", 10)
        telemetry.record_metric("test_gauge", 5.5)

        # Export to Prometheus
        prom_data = telemetry.export_prometheus()

        # Should be in Prometheus text format
        assert isinstance(prom_data, str)
        assert "test_counter" in prom_data
        assert "test_gauge" in prom_data


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    return [
        {
            "id": "doc1",
            "content": "CSRD requires companies to conduct double materiality assessment...",
            "metadata": {"category": "environmental", "section": "materiality"}
        },
        {
            "id": "doc2",
            "content": "Scope 3 emissions include indirect emissions in the value chain...",
            "metadata": {"category": "environmental", "section": "emissions"}
        },
        {
            "id": "doc3",
            "content": "XBRL format is required for digital reporting under ESRS...",
            "metadata": {"category": "technical", "section": "reporting"}
        },
        {
            "id": "doc4",
            "content": "Social materiality assessment covers employee welfare and human rights...",
            "metadata": {"category": "social", "section": "materiality"}
        },
        {
            "id": "doc5",
            "content": "Governance reporting includes board composition and executive compensation...",
            "metadata": {"category": "governance", "section": "reporting"}
        }
    ]


# ============================================================================
# Mock Classes (fallback if infrastructure not available)
# ============================================================================

if ChatSession is None:
    class ChatSession:
        def __init__(self, session_id, max_messages=100):
            self.session_id = session_id
            self.max_messages = max_messages
            self.messages = []

        @property
        def message_count(self):
            return len(self.messages)

        def add_message(self, role, content, metadata=None):
            self.messages.append({
                "role": role,
                "content": content,
                "metadata": metadata or {}
            })
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]

        def get_history(self):
            return self.messages

        def save(self, path):
            with open(path, 'w') as f:
                json.dump({
                    "session_id": self.session_id,
                    "messages": self.messages
                }, f)

        @classmethod
        def load(cls, path):
            with open(path, 'r') as f:
                data = json.load(f)
            session = cls(data["session_id"])
            session.messages = data["messages"]
            return session


if RAGEngine is None:
    class RAGEngine:
        def __init__(self, embedding_model=None, use_reranking=False):
            self.embedding_model = embedding_model
            self.use_reranking = use_reranking
            self.documents = []

        def index_document(self, doc_id, content, metadata):
            self.documents.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata
            })

        def retrieve(self, query, top_k=3, rerank=False, filter_metadata=None):
            filtered_docs = self.documents
            if filter_metadata:
                filtered_docs = [
                    d for d in self.documents
                    if all(d["metadata"].get(k) == v for k, v in filter_metadata.items())
                ]

            # Simple mock scoring
            results = []
            for doc in filtered_docs[:top_k]:
                results.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": 0.9 - len(results) * 0.1
                })

            return results


if SemanticCache is None:
    class SemanticCache:
        def __init__(self, similarity_threshold=0.85, ttl_seconds=3600, max_cache_size=1000):
            self.similarity_threshold = similarity_threshold
            self.ttl_seconds = ttl_seconds
            self.max_cache_size = max_cache_size
            self.cache = {}

        def set(self, key, value):
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                self.cache.pop(next(iter(self.cache)))

            self.cache[key] = {
                "value": value,
                "timestamp": time.time()
            }

        def get(self, key):
            # Exact match
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry["timestamp"] < self.ttl_seconds:
                    return entry["value"]
                else:
                    del self.cache[key]
                    return None

            # Semantic match (mock - just check similarity of length for demo)
            for cached_key, entry in self.cache.items():
                if time.time() - entry["timestamp"] >= self.ttl_seconds:
                    continue

                # Mock semantic similarity
                if abs(len(key) - len(cached_key)) <= 5 and \
                   any(word in cached_key for word in key.split()[:3]):
                    return entry["value"]

            return None

        @property
        def size(self):
            return len(self.cache)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'infrastructure'])
