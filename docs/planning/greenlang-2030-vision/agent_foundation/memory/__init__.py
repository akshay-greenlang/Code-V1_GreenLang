# -*- coding: utf-8 -*-
"""
Memory Systems Package for GreenLang Agent Foundation.

This package implements comprehensive memory systems including:
- Short-term memory with working memory, attention buffer, and context window
- Long-term memory with 4-tier storage (hot/warm/cold/archive)
- Episodic memory for experience tracking and learning
- Semantic memory for facts, concepts, and relationships
- Memory management for consolidation and pruning

All memory systems include provenance tracking with SHA-256 hashes
for complete audit trails and regulatory compliance.
"""

from .short_term_memory import ShortTermMemory, WorkingMemory, AttentionBuffer
from .long_term_memory import LongTermMemory, StorageTier, MemoryIndex
from .episodic_memory import EpisodicMemory, Episode, LearningMechanism
from .semantic_memory import SemanticMemory, KnowledgeType, KnowledgeGraph
from .memory_manager import MemoryManager, ConsolidationStrategy, PruningPolicy

__all__ = [
    # Short-term memory
    "ShortTermMemory",
    "WorkingMemory",
    "AttentionBuffer",

    # Long-term memory
    "LongTermMemory",
    "StorageTier",
    "MemoryIndex",

    # Episodic memory
    "EpisodicMemory",
    "Episode",
    "LearningMechanism",

    # Semantic memory
    "SemanticMemory",
    "KnowledgeType",
    "KnowledgeGraph",

    # Memory management
    "MemoryManager",
    "ConsolidationStrategy",
    "PruningPolicy",
]

__version__ = "1.0.0"