# -*- coding: utf-8 -*-
"""
Pytest fixtures for Entity Resolution ML tests.

This module provides shared fixtures for testing the Entity MDM ML components
including mock Weaviate clients, BERT models, sample data, and test utilities.

Target: 400+ lines, comprehensive fixture coverage
"""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch
from greenlang.utilities.determinism import deterministic_random


# ============================================================================
# MOCK WEAVIATE CLIENT FIXTURES
# ============================================================================

@pytest.fixture
def mock_weaviate_client():
    """Mock Weaviate client for vector store testing."""
    client = MagicMock()

    # Mock schema operations
    client.schema.get.return_value = {
        "classes": [
            {
                "class": "Supplier",
                "vectorizer": "none",
                "properties": [
                    {"name": "name", "dataType": ["string"]},
                    {"name": "country", "dataType": ["string"]},
                    {"name": "lei_code", "dataType": ["string"]},
                    {"name": "duns_number", "dataType": ["string"]},
                ]
            }
        ]
    }

    client.schema.create_class.return_value = None
    client.schema.delete_class.return_value = None

    # Mock batch operations
    batch_mock = MagicMock()
    batch_mock.__enter__ = Mock(return_value=batch_mock)
    batch_mock.__exit__ = Mock(return_value=False)
    batch_mock.add_data_object = Mock(return_value=None)
    client.batch.configure = Mock(return_value=batch_mock)

    # Mock query operations
    query_builder = MagicMock()
    client.query.get.return_value = query_builder

    return client


@pytest.fixture
def mock_weaviate_query_response():
    """Mock Weaviate query response with sample results."""
    return {
        "data": {
            "Get": {
                "Supplier": [
                    {
                        "name": "ACME Corporation Ltd",
                        "country": "US",
                        "lei_code": "549300XXXXXXXXXXXX",
                        "_additional": {
                            "id": "uuid-001",
                            "distance": 0.15,  # Low distance = high similarity
                            "certainty": 0.925
                        }
                    },
                    {
                        "name": "Acme Corp",
                        "country": "US",
                        "lei_code": None,
                        "_additional": {
                            "id": "uuid-002",
                            "distance": 0.22,
                            "certainty": 0.89
                        }
                    },
                    {
                        "name": "ACME Industries",
                        "country": "US",
                        "lei_code": None,
                        "_additional": {
                            "id": "uuid-003",
                            "distance": 0.35,
                            "certainty": 0.825
                        }
                    }
                ]
            }
        }
    }


# ============================================================================
# MOCK BERT MODEL FIXTURES
# ============================================================================

@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence-transformers model for embeddings."""
    model = MagicMock()

    # Mock encode method to return deterministic embeddings
    def mock_encode(sentences, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]

        # Generate pseudo-random but deterministic embeddings
        embeddings = []
        for sentence in sentences:
            # Use hash of sentence for deterministic randomness
            seed = hash(sentence) % (2**32)
            np.random.seed(seed)
            embedding = np.random.randn(384).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)

        return np.array(embeddings)

    model.encode.side_effect = mock_encode
    model.get_sentence_embedding_dimension.return_value = 384

    return model


@pytest.fixture
def mock_cross_encoder():
    """Mock cross-encoder model for BERT re-ranking."""
    model = MagicMock()

    # Mock predict method to return similarity scores
    def mock_predict(sentence_pairs, **kwargs):
        scores = []
        for pair in sentence_pairs:
            # Simple similarity based on character overlap
            s1, s2 = pair[0].lower(), pair[1].lower()
            overlap = len(set(s1.split()) & set(s2.split()))
            total_words = len(set(s1.split()) | set(s2.split()))
            score = overlap / max(total_words, 1) if total_words > 0 else 0.0

            # Add some variance
            score = min(1.0, max(0.0, score + np.random.uniform(-0.1, 0.1)))
            scores.append(score)

        return np.array(scores)

    model.predict.side_effect = mock_predict

    return model


# ============================================================================
# SAMPLE SUPPLIER DATA FIXTURES (100+ variations)
# ============================================================================

@pytest.fixture
def sample_suppliers():
    """Sample supplier data with 100+ variations for comprehensive testing."""
    return [
        # ACME variations (10 variations)
        {"name": "ACME Corporation Ltd", "country": "US", "lei_code": "549300ACME000001"},
        {"name": "Acme Corp", "country": "US", "lei_code": None},
        {"name": "ACME Corp.", "country": "US", "lei_code": None},
        {"name": "Acme Corporation", "country": "US", "lei_code": "549300ACME000001"},
        {"name": "ACME CORPORATION", "country": "US", "lei_code": None},
        {"name": "Acme Inc.", "country": "US", "lei_code": None},
        {"name": "ACME Industries", "country": "US", "lei_code": "549300ACME000002"},
        {"name": "Acme Global", "country": "US", "lei_code": None},
        {"name": "ACME Ltd", "country": "GB", "lei_code": "549300ACME000003"},
        {"name": "A.C.M.E. Corporation", "country": "US", "lei_code": None},

        # ABC Manufacturing variations (10 variations)
        {"name": "ABC Manufacturing Inc.", "country": "US", "lei_code": "123456ABC0000001"},
        {"name": "ABC Mfg", "country": "US", "lei_code": None},
        {"name": "ABC Manufacturing", "country": "US", "lei_code": None},
        {"name": "ABC MFG INC", "country": "US", "lei_code": "123456ABC0000001"},
        {"name": "A.B.C. Manufacturing", "country": "US", "lei_code": None},
        {"name": "ABC Manufacturing Ltd", "country": "GB", "lei_code": "123456ABC0000002"},
        {"name": "ABC Manufacturing Company", "country": "US", "lei_code": None},
        {"name": "ABC Industries", "country": "US", "lei_code": "123456ABC0000003"},
        {"name": "ABC Mfg. Co.", "country": "US", "lei_code": None},
        {"name": "ABC Manufacturing Solutions", "country": "US", "lei_code": None},

        # Global Tech variations (10 variations)
        {"name": "Global Tech Corporation", "country": "US", "lei_code": "789012GLOBAL0001"},
        {"name": "Global Tech Corp", "country": "US", "lei_code": None},
        {"name": "GlobalTech", "country": "US", "lei_code": None},
        {"name": "Global Technology Corp", "country": "US", "lei_code": None},
        {"name": "GLOBAL TECH", "country": "US", "lei_code": "789012GLOBAL0001"},
        {"name": "Global Tech Inc.", "country": "US", "lei_code": None},
        {"name": "Global Tech Solutions", "country": "US", "lei_code": "789012GLOBAL0002"},
        {"name": "Global Tech Ltd", "country": "GB", "lei_code": "789012GLOBAL0003"},
        {"name": "GT Corporation", "country": "US", "lei_code": None},
        {"name": "Global Technology Inc", "country": "US", "lei_code": None},

        # Smith Industries variations (10 variations)
        {"name": "Smith Industries Inc.", "country": "US", "lei_code": "345678SMITH00001"},
        {"name": "Smith Ind.", "country": "US", "lei_code": None},
        {"name": "Smith Industries", "country": "US", "lei_code": None},
        {"name": "SMITH INDUSTRIES INC", "country": "US", "lei_code": "345678SMITH00001"},
        {"name": "Smith Industrial Solutions", "country": "US", "lei_code": "345678SMITH00002"},
        {"name": "Smith Industries Ltd", "country": "GB", "lei_code": "345678SMITH00003"},
        {"name": "Smith Ind. Corp.", "country": "US", "lei_code": None},
        {"name": "Smith Industrial", "country": "US", "lei_code": None},
        {"name": "Smith Industries Group", "country": "US", "lei_code": None},
        {"name": "Smith & Associates", "country": "US", "lei_code": "345678SMITH00004"},

        # Tech Solutions variations (10 variations)
        {"name": "Tech Solutions International", "country": "US", "lei_code": "901234TECH000001"},
        {"name": "Tech Solutions Int'l", "country": "US", "lei_code": None},
        {"name": "TechSolutions", "country": "US", "lei_code": None},
        {"name": "Tech Solutions Inc.", "country": "US", "lei_code": None},
        {"name": "TECH SOLUTIONS INTL", "country": "US", "lei_code": "901234TECH000001"},
        {"name": "Tech Solutions Corp", "country": "US", "lei_code": None},
        {"name": "Technology Solutions", "country": "US", "lei_code": "901234TECH000002"},
        {"name": "Tech Solutions Ltd", "country": "GB", "lei_code": "901234TECH000003"},
        {"name": "TS International", "country": "US", "lei_code": None},
        {"name": "Tech Solutions Group", "country": "US", "lei_code": None},

        # European companies (10 variations)
        {"name": "Deutsche Manufacturing GmbH", "country": "DE", "lei_code": "529900DEUT000001"},
        {"name": "Société Française SA", "country": "FR", "lei_code": "969500FRAN000001"},
        {"name": "British Industries Ltd", "country": "GB", "lei_code": "213800BRIT000001"},
        {"name": "Italian Design SpA", "country": "IT", "lei_code": "815600ITAL000001"},
        {"name": "Spanish Solutions SL", "country": "ES", "lei_code": "959800SPAN000001"},
        {"name": "Dutch Manufacturing BV", "country": "NL", "lei_code": "724500DUTC000001"},
        {"name": "Swedish Tech AB", "country": "SE", "lei_code": "549300SWED000001"},
        {"name": "Swiss Precision AG", "country": "CH", "lei_code": "506700SWIS000001"},
        {"name": "Belgian Industries NV", "country": "BE", "lei_code": "967500BELG000001"},
        {"name": "Austrian Solutions GmbH", "country": "AT", "lei_code": "529900AUST000001"},

        # Asian companies (10 variations)
        {"name": "Tokyo Industries Ltd", "country": "JP", "lei_code": "353800JAPA000001"},
        {"name": "Shanghai Manufacturing Co", "country": "CN", "lei_code": "300300CHIN000001"},
        {"name": "Seoul Tech Corporation", "country": "KR", "lei_code": "988400KORE000001"},
        {"name": "Singapore Solutions Pte", "country": "SG", "lei_code": "549300SING000001"},
        {"name": "Hong Kong Trading Ltd", "country": "HK", "lei_code": "254900HONG000001"},
        {"name": "Taipei Electronics Co", "country": "TW", "lei_code": "391200TAIW000001"},
        {"name": "Mumbai Industries Ltd", "country": "IN", "lei_code": "335800INDI000001"},
        {"name": "Bangkok Manufacturing Co", "country": "TH", "lei_code": "549300THAI000001"},
        {"name": "Jakarta Solutions PT", "country": "ID", "lei_code": "254900INDO000001"},
        {"name": "Manila Corp Philippines", "country": "PH", "lei_code": "254900PHIL000001"},

        # Ambiguous/challenging names (10 variations)
        {"name": "ABC", "country": "US", "lei_code": None},
        {"name": "XYZ Corp", "country": "US", "lei_code": None},
        {"name": "123 Industries", "country": "US", "lei_code": None},
        {"name": "The Company", "country": "US", "lei_code": None},
        {"name": "Global Inc", "country": "US", "lei_code": None},
        {"name": "International Corp", "country": "US", "lei_code": None},
        {"name": "Solutions Ltd", "country": "GB", "lei_code": None},
        {"name": "Services Inc", "country": "US", "lei_code": None},
        {"name": "Group Holdings", "country": "US", "lei_code": None},
        {"name": "Partners LLC", "country": "US", "lei_code": None},

        # Special characters and formatting (10 variations)
        {"name": "O'Reilly Manufacturing", "country": "IE", "lei_code": "635400OREI000001"},
        {"name": "Müller GmbH", "country": "DE", "lei_code": "529900MULL000001"},
        {"name": "Société Générale Industries", "country": "FR", "lei_code": "969500SOGE000001"},
        {"name": "AT&T Solutions", "country": "US", "lei_code": "549300ATT0000001"},
        {"name": "3M Corporation", "country": "US", "lei_code": "549300MMM0000001"},
        {"name": "H&M Manufacturing", "country": "SE", "lei_code": "549300HM00000001"},
        {"name": "S&P Industries", "country": "US", "lei_code": "549300SP00000001"},
        {"name": "E-Corp Solutions", "country": "US", "lei_code": "549300ECOR000001"},
        {"name": "Re:Source Manufacturing", "country": "US", "lei_code": "549300RESO000001"},
        {"name": "Forward/Slash Corp", "country": "US", "lei_code": "549300FWSL000001"},

        # Subsidiaries and parent companies (10 variations)
        {"name": "ACME Corporation (Parent)", "country": "US", "lei_code": "549300ACMEP00001"},
        {"name": "ACME USA Subsidiary", "country": "US", "lei_code": "549300ACMES00001"},
        {"name": "ACME Europe GmbH", "country": "DE", "lei_code": "549300ACMEE00001"},
        {"name": "ABC Holdings", "country": "US", "lei_code": "123456ABCH00001"},
        {"name": "ABC North America", "country": "US", "lei_code": "123456ABCN00001"},
        {"name": "Global Tech Holdings", "country": "US", "lei_code": "789012GLOBALH01"},
        {"name": "Global Tech Americas", "country": "US", "lei_code": "789012GLOBALA01"},
        {"name": "Smith Industries Group Holdings", "country": "US", "lei_code": "345678SMITHH001"},
        {"name": "Smith USA Division", "country": "US", "lei_code": "345678SMITHU001"},
        {"name": "Tech Solutions Worldwide", "country": "US", "lei_code": "901234TECHW00001"},
    ]


@pytest.fixture
def sample_supplier_pairs():
    """Sample supplier pairs with match labels for training/testing."""
    return [
        # Positive pairs (matches)
        (("Acme Corp", "ACME Corporation Ltd"), 1),
        (("ABC Mfg", "ABC Manufacturing Inc."), 1),
        (("Global Tech", "Global Tech Corporation"), 1),
        (("Smith Ind.", "Smith Industries Inc."), 1),
        (("TechSolutions", "Tech Solutions International"), 1),
        (("ACME CORPORATION", "Acme Corporation"), 1),
        (("ABC MFG INC", "ABC Manufacturing Inc."), 1),
        (("GlobalTech", "Global Tech Corp"), 1),
        (("Smith Industries", "Smith Industrial Solutions"), 1),
        (("Tech Solutions Int'l", "Tech Solutions International"), 1),

        # Negative pairs (non-matches)
        (("ACME Corporation", "ABC Manufacturing"), 0),
        (("Global Tech", "Smith Industries"), 0),
        (("Tech Solutions", "ACME Industries"), 0),
        (("ABC Manufacturing", "Global Tech Corp"), 0),
        (("Smith Industries", "Tech Solutions"), 0),
        (("ACME Ltd", "Smith & Associates"), 0),
        (("Global Tech", "ABC Mfg"), 0),
        (("Tech Solutions", "Smith Ind."), 0),
        (("ABC Industries", "ACME Global"), 0),
        (("Global Tech Solutions", "Smith Industries Group"), 0),
    ]


# ============================================================================
# MOCK EMBEDDING VECTORS
# ============================================================================

@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors for testing."""
    np.random.seed(42)
    return {
        "ACME Corporation Ltd": np.random.randn(384).astype(np.float32),
        "ABC Manufacturing Inc.": np.random.randn(384).astype(np.float32),
        "Global Tech Corporation": np.random.randn(384).astype(np.float32),
        "Smith Industries Inc.": np.random.randn(384).astype(np.float32),
        "Tech Solutions International": np.random.randn(384).astype(np.float32),
    }


# ============================================================================
# MOCK MATCHING SCORES
# ============================================================================

@pytest.fixture
def sample_matching_scores():
    """Sample matching scores for testing."""
    return {
        ("Acme Corp", "ACME Corporation Ltd"): 0.97,
        ("ABC Mfg", "ABC Manufacturing Inc."): 0.95,
        ("Global Tech", "Global Tech Corporation"): 0.98,
        ("Smith Ind.", "Smith Industries Inc."): 0.92,
        ("TechSolutions", "Tech Solutions International"): 0.89,
        ("ACME Corporation", "ABC Manufacturing"): 0.35,
        ("Global Tech", "Smith Industries"): 0.28,
        ("Tech Solutions", "ACME Industries"): 0.42,
    }


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def entity_mdm_config():
    """Sample Entity MDM configuration."""
    return {
        "weaviate": {
            "host": "localhost",
            "port": 8080,
            "scheme": "http",
            "timeout_config": (5, 15)
        },
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "batch_size": 32,
            "normalize": True
        },
        "matching": {
            "model": "sentence-transformers/cross-encoder/ms-marco-MiniLM-L-12-v2",
            "top_k": 10,
            "min_similarity": 0.70,
            "auto_match_threshold": 0.95,
            "human_review_threshold": 0.80
        },
        "performance": {
            "cache_enabled": True,
            "cache_size": 10000,
            "batch_processing": True
        }
    }


# ============================================================================
# TEST DATA HELPERS
# ============================================================================

@pytest.fixture
def create_supplier():
    """Factory function to create test suppliers."""
    def _create(name: str, country: str = "US", lei_code: str = None,
                duns_number: str = None, **kwargs):
        return {
            "name": name,
            "country": country,
            "lei_code": lei_code,
            "duns_number": duns_number,
            **kwargs
        }
    return _create


@pytest.fixture
def create_embedding():
    """Factory function to create test embeddings."""
    def _create(dimension: int = 384, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        embedding = np.random.randn(dimension).astype(np.float32)
        return embedding / np.linalg.norm(embedding)  # Normalize
    return _create


@pytest.fixture
def temp_model_path(tmp_path):
    """Temporary path for saving/loading models in tests."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


# ============================================================================
# PERFORMANCE TESTING FIXTURES
# ============================================================================

@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing."""
    np.random.seed(42)

    # Generate 1000 test suppliers
    suppliers = []
    for i in range(1000):
        suppliers.append({
            "name": f"Test Company {i}",
            "country": np.deterministic_random().choice(["US", "GB", "DE", "FR", "JP"]),
            "lei_code": f"549300TEST{i:06d}" if np.deterministic_random().random() > 0.5 else None
        })

    return suppliers


# ============================================================================
# MOCK TRAINING DATA
# ============================================================================

@pytest.fixture
def mock_training_dataset(tmp_path):
    """Mock training dataset in JSONL format."""
    import json

    dataset_path = tmp_path / "training_data.jsonl"

    training_data = [
        {"input": "Acme Corp", "candidate": "ACME Corporation Ltd", "label": 1},
        {"input": "ABC Mfg", "candidate": "ABC Manufacturing Inc.", "label": 1},
        {"input": "Global Tech", "candidate": "Global Tech Corporation", "label": 1},
        {"input": "ACME Corp", "candidate": "ABC Manufacturing", "label": 0},
        {"input": "Global Tech", "candidate": "Smith Industries", "label": 0},
    ] * 20  # Repeat to create 100 training examples

    with open(dataset_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')

    return dataset_path
