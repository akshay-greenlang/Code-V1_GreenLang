# -*- coding: utf-8 -*-
"""
Entity Resolution ML Module for GL-VCCI Scope 3 Carbon Intelligence Platform.

This module implements a two-stage ML pipeline for supplier entity resolution:
1. Candidate Generation: Fast vector similarity search using embeddings
2. Re-ranking: BERT-based pairwise matching for high-precision results

Target Performance:
- Auto-match rate: ≥95%
- Precision: ≥95%
- Latency: <500ms per query

Author: GreenLang AI
Phase: 5 - Entity Resolution ML
"""

from entity_mdm.ml.config import MLConfig, ModelConfig, WeaviateConfig
from entity_mdm.ml.embeddings import EmbeddingPipeline
from entity_mdm.ml.vector_store import VectorStore
from entity_mdm.ml.matching_model import MatchingModel
from entity_mdm.ml.resolver import EntityResolver
from entity_mdm.ml.evaluation import ModelEvaluator
from entity_mdm.ml.training import TrainingPipeline
from entity_mdm.ml.exceptions import (
    ModelNotTrainedException,
    InsufficientCandidatesException,
    VectorStoreException,
    EmbeddingException,
    MatchingException,
)

__all__ = [
    "MLConfig",
    "ModelConfig",
    "WeaviateConfig",
    "EmbeddingPipeline",
    "VectorStore",
    "MatchingModel",
    "EntityResolver",
    "ModelEvaluator",
    "TrainingPipeline",
    "ModelNotTrainedException",
    "InsufficientCandidatesException",
    "VectorStoreException",
    "EmbeddingException",
    "MatchingException",
]

__version__ = "1.0.0"
