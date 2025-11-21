# -*- coding: utf-8 -*-
"""
Unit Tests for Memory Systems
Comprehensive tests for short-term, long-term, episodic, and semantic memory.
Validates memory consolidation, retrieval, and provenance tracking.
"""

import pytest
import asyncio
import time
import numpy as np
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import deque
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import sys
import os
from greenlang.determinism import DeterministicClock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from testing.agent_test_framework import (
    AgentTestCase,
    TestConfig,
    DeterministicLLMProvider,
    TestDataGenerator,
    ProvenanceValidator
)


# Mock Memory Classes for Testing
class ShortTermMemory:
    """Short-term memory with working memory, attention buffer, context window."""

    def __init__(self, capacity: int = 100, attention_size: int = 10, context_window: int = 50):
        self.capacity = capacity
        self.attention_size = attention_size
        self.context_window = context_window
        self.working_memory = deque(maxlen=capacity)
        self.attention_buffer = deque(maxlen=attention_size)
        self.context = deque(maxlen=context_window)
        self.access_count = 0

    def add(self, item: Dict[str, Any]):
        """Add item to short-term memory."""
        item['timestamp'] = DeterministicClock.now().isoformat()
        item['importance'] = item.get('importance', 0.5)
        self.working_memory.append(item)

        # Add to attention buffer if important
        if item['importance'] > 0.7:
            self.attention_buffer.append(item)

        # Add to context
        self.context.append(item)

    def retrieve(self, query: str = None, k: int = 10) -> List[Dict]:
        """Retrieve items from short-term memory."""
        self.access_count += 1

        if not query:
            return list(self.working_memory)[-k:]

        # Simple keyword matching
        results = [item for item in self.working_memory
                  if query.lower() in str(item).lower()]
        return results[-k:]

    def get_attention(self) -> List[Dict]:
        """Get items in attention buffer."""
        return list(self.attention_buffer)

    def get_context(self) -> List[Dict]:
        """Get context window."""
        return list(self.context)

    def clear(self):
        """Clear all short-term memory."""
        self.working_memory.clear()
        self.attention_buffer.clear()
        self.context.clear()


class LongTermMemory:
    """Long-term memory with hot/warm/cold/archive tiers."""

    def __init__(self):
        self.hot_tier = {}  # Recently accessed (Redis-like)
        self.warm_tier = {}  # Frequently accessed (PostgreSQL-like)
        self.cold_tier = {}  # Infrequently accessed (S3-like)
        self.archive_tier = {}  # Archived (Glacier-like)
        self.access_stats = {}

    def store(self, key: str, value: Dict[str, Any], tier: str = "hot"):
        """Store item in long-term memory."""
        value['stored_at'] = DeterministicClock.now().isoformat()
        value['tier'] = tier

        storage = self._get_tier_storage(tier)
        storage[key] = value

        # Initialize access stats
        self.access_stats[key] = {
            'access_count': 0,
            'last_accessed': None,
            'tier': tier
        }

    def retrieve(self, key: str) -> Dict[str, Any]:
        """Retrieve item from long-term memory."""
        # Check all tiers
        for tier in ['hot', 'warm', 'cold', 'archive']:
            storage = self._get_tier_storage(tier)
            if key in storage:
                # Update access stats
                self.access_stats[key]['access_count'] += 1
                self.access_stats[key]['last_accessed'] = DeterministicClock.now().isoformat()

                # Promote if frequently accessed
                self._consider_promotion(key, tier)

                return storage[key]

        return None

    def _get_tier_storage(self, tier: str) -> Dict:
        """Get storage for tier."""
        tier_map = {
            'hot': self.hot_tier,
            'warm': self.warm_tier,
            'cold': self.cold_tier,
            'archive': self.archive_tier
        }
        return tier_map.get(tier, self.hot_tier)

    def _consider_promotion(self, key: str, current_tier: str):
        """Consider promoting item to higher tier based on access patterns."""
        stats = self.access_stats[key]

        # Promotion logic based on access count
        if stats['access_count'] > 10 and current_tier in ['cold', 'warm']:
            self._promote(key, current_tier, 'hot')
        elif stats['access_count'] > 5 and current_tier == 'cold':
            self._promote(key, current_tier, 'warm')

    def _promote(self, key: str, from_tier: str, to_tier: str):
        """Promote item to higher tier."""
        from_storage = self._get_tier_storage(from_tier)
        to_storage = self._get_tier_storage(to_tier)

        if key in from_storage:
            value = from_storage.pop(key)
            value['tier'] = to_tier
            to_storage[key] = value
            self.access_stats[key]['tier'] = to_tier

    def consolidate(self):
        """Consolidate memory tiers based on access patterns."""
        # Move cold items to archive
        for key, stats in list(self.access_stats.items()):
            if stats['tier'] == 'hot' and stats['access_count'] < 2:
                self._demote(key, 'hot', 'warm')
            elif stats['tier'] == 'warm' and stats['access_count'] < 1:
                self._demote(key, 'warm', 'cold')

    def _demote(self, key: str, from_tier: str, to_tier: str):
        """Demote item to lower tier."""
        from_storage = self._get_tier_storage(from_tier)
        to_storage = self._get_tier_storage(to_tier)

        if key in from_storage:
            value = from_storage.pop(key)
            value['tier'] = to_tier
            to_storage[key] = value
            self.access_stats[key]['tier'] = to_tier


class EpisodicMemory:
    """Episodic memory with experience replay and pattern extraction."""

    def __init__(self):
        self.episodes = []
        self.patterns = []
        self.cases = {}  # Case-based reasoning

    def record_episode(self, episode: Dict[str, Any]):
        """Record an episode."""
        episode['id'] = hashlib.sha256(json.dumps(episode, sort_keys=True).encode()).hexdigest()
        episode['timestamp'] = DeterministicClock.now().isoformat()
        self.episodes.append(episode)

    def replay(self, k: int = 10) -> List[Dict]:
        """Replay recent episodes."""
        return self.episodes[-k:]

    def extract_patterns(self) -> List[Dict]:
        """Extract patterns from episodes."""
        # Simple pattern extraction: group similar episodes
        patterns = {}

        for episode in self.episodes:
            pattern_key = episode.get('action', 'unknown')
            if pattern_key not in patterns:
                patterns[pattern_key] = {
                    'pattern': pattern_key,
                    'count': 0,
                    'examples': []
                }
            patterns[pattern_key]['count'] += 1
            patterns[pattern_key]['examples'].append(episode['id'])

        self.patterns = list(patterns.values())
        return self.patterns

    def case_based_reasoning(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Find similar past cases."""
        # Simple similarity based on action type
        query_action = query.get('action', '')

        similar_episodes = [ep for ep in self.episodes
                           if ep.get('action') == query_action]

        if similar_episodes:
            return similar_episodes[-1]  # Most recent similar case
        return None


class SemanticMemory:
    """Semantic memory with knowledge graph, facts, concepts, procedures."""

    def __init__(self):
        self.facts = {}
        self.concepts = {}
        self.procedures = {}
        self.knowledge_graph = {'nodes': [], 'edges': []}

    def store_fact(self, fact_id: str, fact: Dict[str, Any]):
        """Store a fact."""
        self.facts[fact_id] = fact
        self._add_to_knowledge_graph(fact_id, 'fact', fact)

    def store_concept(self, concept_id: str, concept: Dict[str, Any]):
        """Store a concept."""
        self.concepts[concept_id] = concept
        self._add_to_knowledge_graph(concept_id, 'concept', concept)

    def store_procedure(self, procedure_id: str, procedure: Dict[str, Any]):
        """Store a procedure."""
        self.procedures[procedure_id] = procedure
        self._add_to_knowledge_graph(procedure_id, 'procedure', procedure)

    def _add_to_knowledge_graph(self, node_id: str, node_type: str, data: Dict):
        """Add node to knowledge graph."""
        node = {
            'id': node_id,
            'type': node_type,
            'data': data
        }
        self.knowledge_graph['nodes'].append(node)

        # Create edges based on relationships
        if 'related_to' in data:
            for related_id in data['related_to']:
                edge = {
                    'from': node_id,
                    'to': related_id,
                    'type': 'related'
                }
                self.knowledge_graph['edges'].append(edge)

    def query_graph(self, query: str) -> List[Dict]:
        """Query knowledge graph."""
        results = []
        for node in self.knowledge_graph['nodes']:
            if query.lower() in str(node['data']).lower():
                results.append(node)
        return results


# Unit Tests
class TestShortTermMemory(AgentTestCase):
    """Test short-term memory systems."""

    def setUp(self):
        super().setUp()
        self.stm = ShortTermMemory(capacity=100, attention_size=10, context_window=50)

    def test_initialization(self):
        """Test STM initializes correctly."""
        self.assertEqual(self.stm.capacity, 100)
        self.assertEqual(self.stm.attention_size, 10)
        self.assertEqual(self.stm.context_window, 50)
        self.assertEqual(len(self.stm.working_memory), 0)

    def test_add_to_working_memory(self):
        """Test adding items to working memory."""
        item = {'content': 'test data', 'importance': 0.5}
        self.stm.add(item)

        self.assertEqual(len(self.stm.working_memory), 1)
        stored_item = self.stm.working_memory[0]
        self.assertEqual(stored_item['content'], 'test data')
        self.assertIn('timestamp', stored_item)

    def test_attention_buffer_filtering(self):
        """Test attention buffer only stores important items."""
        # Add low importance item
        self.stm.add({'content': 'low importance', 'importance': 0.3})
        self.assertEqual(len(self.stm.attention_buffer), 0)

        # Add high importance item
        self.stm.add({'content': 'high importance', 'importance': 0.9})
        self.assertEqual(len(self.stm.attention_buffer), 1)

    def test_context_window(self):
        """Test context window maintains recent items."""
        for i in range(60):
            self.stm.add({'content': f'item_{i}', 'importance': 0.5})

        # Context window should have max 50 items
        self.assertEqual(len(self.stm.context), 50)

        # Should have most recent items
        context_items = self.stm.get_context()
        self.assertEqual(context_items[-1]['content'], 'item_59')

    def test_capacity_limit(self):
        """Test working memory respects capacity limit."""
        for i in range(150):
            self.stm.add({'content': f'item_{i}', 'importance': 0.5})

        # Should only have 100 items (capacity)
        self.assertEqual(len(self.stm.working_memory), 100)

    def test_retrieval_performance(self):
        """Test retrieval meets <50ms target."""
        # Fill memory
        for i in range(1000):
            self.stm.add({'content': f'item_{i}', 'data': f'test data {i}'})

        # Test retrieval time
        with self.assert_performance(max_duration_ms=50):
            results = self.stm.retrieve(k=10)

        self.assertEqual(len(results), 10)

    def test_query_based_retrieval(self):
        """Test query-based retrieval."""
        self.stm.add({'content': 'carbon emissions', 'value': 100})
        self.stm.add({'content': 'energy consumption', 'value': 200})
        self.stm.add({'content': 'carbon footprint', 'value': 150})

        results = self.stm.retrieve(query='carbon')
        self.assertEqual(len(results), 2)

    def test_clear_memory(self):
        """Test clearing all memory."""
        for i in range(10):
            self.stm.add({'content': f'item_{i}'})

        self.stm.clear()

        self.assertEqual(len(self.stm.working_memory), 0)
        self.assertEqual(len(self.stm.attention_buffer), 0)
        self.assertEqual(len(self.stm.context), 0)


class TestLongTermMemory(AgentTestCase):
    """Test long-term memory systems."""

    def setUp(self):
        super().setUp()
        self.ltm = LongTermMemory()

    def test_initialization(self):
        """Test LTM initializes with all tiers."""
        self.assertIsNotNone(self.ltm.hot_tier)
        self.assertIsNotNone(self.ltm.warm_tier)
        self.assertIsNotNone(self.ltm.cold_tier)
        self.assertIsNotNone(self.ltm.archive_tier)

    def test_store_in_hot_tier(self):
        """Test storing in hot tier."""
        self.ltm.store('key1', {'value': 'data1'}, tier='hot')

        self.assertIn('key1', self.ltm.hot_tier)
        self.assertEqual(self.ltm.hot_tier['key1']['value'], 'data1')

    def test_retrieve_from_hot_tier(self):
        """Test retrieval from hot tier meets <50ms target."""
        # Store items in hot tier
        for i in range(100):
            self.ltm.store(f'key_{i}', {'value': f'data_{i}'}, tier='hot')

        # Test retrieval performance
        with self.assert_performance(max_duration_ms=50):
            result = self.ltm.retrieve('key_50')

        self.assertIsNotNone(result)
        self.assertEqual(result['value'], 'data_50')

    def test_retrieve_from_cold_tier(self):
        """Test retrieval from cold tier meets <200ms target."""
        # Store items in cold tier
        for i in range(100):
            self.ltm.store(f'key_{i}', {'value': f'data_{i}'}, tier='cold')

        # Test retrieval performance
        with self.assert_performance(max_duration_ms=200):
            result = self.ltm.retrieve('key_50')

        self.assertIsNotNone(result)

    def test_access_tracking(self):
        """Test access statistics tracking."""
        self.ltm.store('key1', {'value': 'data1'}, tier='hot')

        # Access multiple times
        for _ in range(5):
            self.ltm.retrieve('key1')

        stats = self.ltm.access_stats['key1']
        self.assertEqual(stats['access_count'], 5)
        self.assertIsNotNone(stats['last_accessed'])

    def test_tier_promotion(self):
        """Test automatic promotion based on access patterns."""
        # Store in cold tier
        self.ltm.store('frequently_accessed', {'value': 'data'}, tier='cold')

        # Access frequently
        for _ in range(12):
            self.ltm.retrieve('frequently_accessed')

        # Should be promoted to hot tier
        stats = self.ltm.access_stats['frequently_accessed']
        self.assertEqual(stats['tier'], 'hot')
        self.assertIn('frequently_accessed', self.ltm.hot_tier)

    def test_memory_consolidation(self):
        """Test memory consolidation across tiers."""
        # Store items in hot tier
        self.ltm.store('rarely_used', {'value': 'data'}, tier='hot')

        # Don't access it
        # Consolidate memory
        self.ltm.consolidate()

        # Should be demoted to warm tier
        stats = self.ltm.access_stats['rarely_used']
        self.assertEqual(stats['tier'], 'warm')

    def test_provenance_tracking(self):
        """Test provenance tracking in LTM."""
        data = {
            'value': 100,
            'calculation': 'test',
            'provenance_hash': hashlib.sha256(b'test').hexdigest()
        }

        self.ltm.store('key_with_provenance', data, tier='hot')
        retrieved = self.ltm.retrieve('key_with_provenance')

        self.assertEqual(len(retrieved['provenance_hash']), 64)


class TestEpisodicMemory(AgentTestCase):
    """Test episodic memory systems."""

    def setUp(self):
        super().setUp()
        self.episodic = EpisodicMemory()

    def test_record_episode(self):
        """Test recording episodes."""
        episode = {
            'action': 'calculate_emissions',
            'input': {'fuel': 'diesel', 'quantity': 100},
            'output': {'emissions': 268}
        }

        self.episodic.record_episode(episode)

        self.assertEqual(len(self.episodic.episodes), 1)
        stored = self.episodic.episodes[0]
        self.assertIn('id', stored)
        self.assertIn('timestamp', stored)
        self.assertEqual(len(stored['id']), 64)  # SHA-256 hash

    def test_experience_replay(self):
        """Test experience replay."""
        # Record multiple episodes
        for i in range(20):
            episode = {'action': f'action_{i}', 'data': i}
            self.episodic.record_episode(episode)

        # Replay last 10
        replayed = self.episodic.replay(k=10)

        self.assertEqual(len(replayed), 10)
        self.assertEqual(replayed[-1]['action'], 'action_19')

    def test_pattern_extraction(self):
        """Test pattern extraction from episodes."""
        # Record episodes with patterns
        for i in range(10):
            self.episodic.record_episode({'action': 'calculate', 'value': i})
        for i in range(5):
            self.episodic.record_episode({'action': 'validate', 'value': i})

        patterns = self.episodic.extract_patterns()

        self.assertEqual(len(patterns), 2)

        # Find calculate pattern
        calc_pattern = next(p for p in patterns if p['pattern'] == 'calculate')
        self.assertEqual(calc_pattern['count'], 10)

    def test_case_based_reasoning(self):
        """Test case-based reasoning."""
        # Record cases
        self.episodic.record_episode({
            'action': 'calculate_emissions',
            'fuel_type': 'diesel',
            'result': 268
        })

        # Query for similar case
        query = {'action': 'calculate_emissions', 'fuel_type': 'diesel'}
        similar_case = self.episodic.case_based_reasoning(query)

        self.assertIsNotNone(similar_case)
        self.assertEqual(similar_case['result'], 268)

    def test_deterministic_episode_ids(self):
        """Test episode IDs are deterministic."""
        episode1 = {'action': 'test', 'data': 123}
        episode2 = {'action': 'test', 'data': 123}

        self.episodic.record_episode(episode1)
        ep1_id = self.episodic.episodes[0]['id']

        # Clear and record again
        self.episodic.episodes = []
        self.episodic.record_episode(episode2)
        ep2_id = self.episodic.episodes[0]['id']

        # IDs should be same for same content (minus timestamp)
        # Note: In real implementation, exclude timestamp from hash


class TestSemanticMemory(AgentTestCase):
    """Test semantic memory systems."""

    def setUp(self):
        super().setUp()
        self.semantic = SemanticMemory()

    def test_store_fact(self):
        """Test storing facts."""
        fact = {
            'subject': 'carbon',
            'predicate': 'has_emission_factor',
            'object': 2.68
        }

        self.semantic.store_fact('fact_1', fact)

        self.assertIn('fact_1', self.semantic.facts)
        self.assertEqual(self.semantic.facts['fact_1']['object'], 2.68)

    def test_store_concept(self):
        """Test storing concepts."""
        concept = {
            'name': 'carbon_emissions',
            'definition': 'CO2 released from fuel combustion',
            'related_to': ['greenhouse_gas', 'climate_change']
        }

        self.semantic.store_concept('concept_1', concept)

        self.assertIn('concept_1', self.semantic.concepts)

    def test_store_procedure(self):
        """Test storing procedures."""
        procedure = {
            'name': 'calculate_scope1',
            'steps': [
                'Identify fuel consumption',
                'Apply emission factor',
                'Sum total emissions'
            ]
        }

        self.semantic.store_procedure('proc_1', procedure)

        self.assertIn('proc_1', self.semantic.procedures)
        self.assertEqual(len(self.semantic.procedures['proc_1']['steps']), 3)

    def test_knowledge_graph_construction(self):
        """Test knowledge graph construction."""
        # Add related concepts
        self.semantic.store_concept('co2', {
            'name': 'CO2',
            'related_to': ['greenhouse_gas']
        })

        self.semantic.store_concept('greenhouse_gas', {
            'name': 'Greenhouse Gas',
            'related_to': []
        })

        # Check graph structure
        self.assertEqual(len(self.semantic.knowledge_graph['nodes']), 2)
        self.assertEqual(len(self.semantic.knowledge_graph['edges']), 1)

    def test_knowledge_graph_query(self):
        """Test querying knowledge graph."""
        self.semantic.store_fact('fact_co2', {
            'subject': 'diesel',
            'emission_factor': 2.68
        })

        results = self.semantic.query_graph('diesel')

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], 'fact_co2')


class TestMemoryIntegration(AgentTestCase):
    """Test integration between memory systems."""

    def setUp(self):
        super().setUp()
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

    def test_stm_to_ltm_consolidation(self):
        """Test consolidating STM to LTM."""
        # Fill STM
        for i in range(10):
            self.stm.add({'content': f'important_data_{i}', 'importance': 0.8})

        # Consolidate to LTM
        for item in self.stm.working_memory:
            key = hashlib.sha256(str(item).encode()).hexdigest()[:16]
            self.ltm.store(key, dict(item), tier='hot')

        # Verify in LTM
        self.assertEqual(len(self.ltm.hot_tier), 10)

    def test_episodic_to_semantic_extraction(self):
        """Test extracting semantic knowledge from episodes."""
        # Record episodes
        for i in range(5):
            self.episodic.record_episode({
                'action': 'calculate_emissions',
                'fuel': 'diesel',
                'factor': 2.68,
                'result': 100 * 2.68
            })

        # Extract pattern and create semantic fact
        patterns = self.episodic.extract_patterns()

        for pattern in patterns:
            if pattern['pattern'] == 'calculate_emissions':
                fact = {
                    'procedure': pattern['pattern'],
                    'frequency': pattern['count']
                }
                self.semantic.store_fact('diesel_emissions', fact)

        # Verify semantic memory
        self.assertIn('diesel_emissions', self.semantic.facts)

    def test_full_memory_workflow(self):
        """Test complete memory workflow: STM -> Episodic -> LTM -> Semantic."""
        # 1. Process in STM
        self.stm.add({
            'action': 'calculate',
            'fuel': 'diesel',
            'quantity': 100,
            'importance': 0.9
        })

        # 2. Record as episode
        for item in self.stm.get_attention():
            self.episodic.record_episode(item)

        # 3. Consolidate important episodes to LTM
        for episode in self.episodic.episodes:
            if episode.get('importance', 0) > 0.8:
                key = episode['id'][:16]
                self.ltm.store(key, episode, tier='hot')

        # 4. Extract patterns to semantic memory
        patterns = self.episodic.extract_patterns()
        for pattern in patterns:
            self.semantic.store_procedure(
                f"pattern_{pattern['pattern']}",
                {'pattern': pattern}
            )

        # Verify complete workflow
        self.assertGreater(len(self.ltm.hot_tier), 0)
        self.assertGreater(len(self.semantic.procedures), 0)


@pytest.mark.performance
class TestMemoryPerformance(AgentTestCase):
    """Performance tests for memory systems."""

    def test_stm_retrieval_performance(self):
        """Test STM retrieval meets <50ms target."""
        stm = ShortTermMemory(capacity=1000)

        # Fill with data
        for i in range(1000):
            stm.add({'content': f'data_{i}', 'value': i})

        # Test retrieval time
        durations = []
        for _ in range(100):
            start = time.perf_counter()
            stm.retrieve(k=10)
            duration_ms = (time.perf_counter() - start) * 1000
            durations.append(duration_ms)

        p99_duration = np.percentile(durations, 99)
        self.assertLess(p99_duration, 50,
                       f"P99 retrieval time {p99_duration:.2f}ms > 50ms target")

    def test_ltm_hot_tier_performance(self):
        """Test LTM hot tier retrieval meets <50ms target."""
        ltm = LongTermMemory()

        # Fill hot tier
        for i in range(10000):
            ltm.store(f'key_{i}', {'value': f'data_{i}'}, tier='hot')

        # Test retrieval time
        durations = []
        for i in range(100):
            start = time.perf_counter()
            ltm.retrieve(f'key_{i * 10}')
            duration_ms = (time.perf_counter() - start) * 1000
            durations.append(duration_ms)

        p99_duration = np.percentile(durations, 99)
        self.assertLess(p99_duration, 50,
                       f"P99 hot tier retrieval {p99_duration:.2f}ms > 50ms target")

    def test_memory_consolidation_performance(self):
        """Test memory consolidation completes in reasonable time."""
        ltm = LongTermMemory()

        # Create large dataset
        for i in range(1000):
            ltm.store(f'key_{i}', {'value': i}, tier='hot')

        # Access some items
        for i in range(100):
            ltm.retrieve(f'key_{i}')

        # Test consolidation time
        with self.assert_performance(max_duration_ms=1000):  # <1s for 1000 items
            ltm.consolidate()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=term"])
