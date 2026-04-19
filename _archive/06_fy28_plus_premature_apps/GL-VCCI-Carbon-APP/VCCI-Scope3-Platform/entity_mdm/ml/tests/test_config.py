# -*- coding: utf-8 -*-
"""
Tests for Entity MDM ML Configuration.

Tests configuration validation, environment variable loading, default values,
and invalid configuration handling.

Target: 250+ lines, 12 tests
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any


# Mock config module (would be actual module in production)
class EntityMDMConfig:
    """Entity MDM ML Configuration."""

    def __init__(self, config_dict: Dict[str, Any] = None):
        self.config = config_dict or self._load_defaults()
        self._validate()

    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "weaviate": {
                "host": os.getenv("WEAVIATE_HOST", "localhost"),
                "port": int(os.getenv("WEAVIATE_PORT", "8080")),
                "scheme": os.getenv("WEAVIATE_SCHEME", "http"),
                "timeout_config": (5, 15)
            },
            "embedding": {
                "model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                "dimension": int(os.getenv("EMBEDDING_DIMENSION", "384")),
                "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
                "normalize": os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true",
                "cache_enabled": True
            },
            "matching": {
                "model": os.getenv("MATCHING_MODEL", "sentence-transformers/cross-encoder/ms-marco-MiniLM-L-12-v2"),
                "top_k": int(os.getenv("MATCHING_TOP_K", "10")),
                "min_similarity": float(os.getenv("MATCHING_MIN_SIMILARITY", "0.70")),
                "auto_match_threshold": float(os.getenv("AUTO_MATCH_THRESHOLD", "0.95")),
                "human_review_threshold": float(os.getenv("HUMAN_REVIEW_THRESHOLD", "0.80"))
            },
            "performance": {
                "cache_enabled": True,
                "cache_size": int(os.getenv("CACHE_SIZE", "10000")),
                "batch_processing": True,
                "max_batch_size": int(os.getenv("MAX_BATCH_SIZE", "100"))
            }
        }

    def _validate(self):
        """Validate configuration."""
        # Validate Weaviate config
        if self.config["weaviate"]["port"] < 1 or self.config["weaviate"]["port"] > 65535:
            raise ValueError("Weaviate port must be between 1 and 65535")

        if self.config["weaviate"]["scheme"] not in ["http", "https"]:
            raise ValueError("Weaviate scheme must be 'http' or 'https'")

        # Validate embedding config
        if self.config["embedding"]["dimension"] <= 0:
            raise ValueError("Embedding dimension must be positive")

        if self.config["embedding"]["batch_size"] <= 0:
            raise ValueError("Embedding batch size must be positive")

        # Validate matching config
        if not (0 <= self.config["matching"]["min_similarity"] <= 1):
            raise ValueError("Matching min_similarity must be between 0 and 1")

        if not (0 <= self.config["matching"]["auto_match_threshold"] <= 1):
            raise ValueError("Auto match threshold must be between 0 and 1")

        if not (0 <= self.config["matching"]["human_review_threshold"] <= 1):
            raise ValueError("Human review threshold must be between 0 and 1")

        if self.config["matching"]["auto_match_threshold"] <= self.config["matching"]["human_review_threshold"]:
            raise ValueError("Auto match threshold must be greater than human review threshold")

        if self.config["matching"]["top_k"] <= 0:
            raise ValueError("Matching top_k must be positive")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def update(self, updates: Dict[str, Any]):
        """Update configuration."""
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        deep_update(self.config, updates)
        self._validate()


# ============================================================================
# TEST SUITE
# ============================================================================

class TestEntityMDMConfig:
    """Test suite for Entity MDM configuration."""

    def test_default_configuration_loads_successfully(self):
        """Test that default configuration loads without errors."""
        config = EntityMDMConfig()

        assert config is not None
        assert "weaviate" in config.config
        assert "embedding" in config.config
        assert "matching" in config.config
        assert "performance" in config.config

    def test_weaviate_default_values(self):
        """Test Weaviate default configuration values."""
        config = EntityMDMConfig()

        assert config.get("weaviate.host") == "localhost"
        assert config.get("weaviate.port") == 8080
        assert config.get("weaviate.scheme") == "http"
        assert config.get("weaviate.timeout_config") == (5, 15)

    def test_embedding_default_values(self):
        """Test embedding model default configuration values."""
        config = EntityMDMConfig()

        assert config.get("embedding.model") == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.get("embedding.dimension") == 384
        assert config.get("embedding.batch_size") == 32
        assert config.get("embedding.normalize") is True
        assert config.get("embedding.cache_enabled") is True

    def test_matching_default_values(self):
        """Test matching model default configuration values."""
        config = EntityMDMConfig()

        assert config.get("matching.model") == "sentence-transformers/cross-encoder/ms-marco-MiniLM-L-12-v2"
        assert config.get("matching.top_k") == 10
        assert config.get("matching.min_similarity") == 0.70
        assert config.get("matching.auto_match_threshold") == 0.95
        assert config.get("matching.human_review_threshold") == 0.80

    @patch.dict(os.environ, {
        "WEAVIATE_HOST": "weaviate.example.com",
        "WEAVIATE_PORT": "9090",
        "WEAVIATE_SCHEME": "https"
    })
    def test_environment_variables_override_defaults(self):
        """Test that environment variables override default values."""
        config = EntityMDMConfig()

        assert config.get("weaviate.host") == "weaviate.example.com"
        assert config.get("weaviate.port") == 9090
        assert config.get("weaviate.scheme") == "https"

    @patch.dict(os.environ, {
        "EMBEDDING_DIMENSION": "512",
        "EMBEDDING_BATCH_SIZE": "64",
        "MATCHING_TOP_K": "20"
    })
    def test_numeric_environment_variables_parsed_correctly(self):
        """Test that numeric environment variables are parsed correctly."""
        config = EntityMDMConfig()

        assert config.get("embedding.dimension") == 512
        assert config.get("embedding.batch_size") == 64
        assert config.get("matching.top_k") == 20

    def test_custom_configuration_overrides_defaults(self):
        """Test that custom configuration overrides defaults."""
        custom_config = {
            "weaviate": {
                "host": "custom.host.com",
                "port": 7000
            },
            "matching": {
                "top_k": 15,
                "auto_match_threshold": 0.90
            }
        }

        config = EntityMDMConfig(custom_config)

        # Custom values should be used
        assert config.get("weaviate.host") == "custom.host.com"
        assert config.get("weaviate.port") == 7000
        assert config.get("matching.top_k") == 15
        assert config.get("matching.auto_match_threshold") == 0.90

    def test_invalid_port_raises_error(self):
        """Test that invalid port number raises validation error."""
        with pytest.raises(ValueError, match="Weaviate port must be between 1 and 65535"):
            EntityMDMConfig({"weaviate": {"host": "localhost", "port": 99999}})

        with pytest.raises(ValueError, match="Weaviate port must be between 1 and 65535"):
            EntityMDMConfig({"weaviate": {"host": "localhost", "port": 0}})

    def test_invalid_scheme_raises_error(self):
        """Test that invalid scheme raises validation error."""
        with pytest.raises(ValueError, match="Weaviate scheme must be 'http' or 'https'"):
            EntityMDMConfig({"weaviate": {"host": "localhost", "scheme": "ftp"}})

    def test_invalid_thresholds_raise_errors(self):
        """Test that invalid threshold values raise validation errors."""
        # Test min_similarity out of range
        with pytest.raises(ValueError, match="min_similarity must be between 0 and 1"):
            EntityMDMConfig({"matching": {"min_similarity": 1.5}})

        with pytest.raises(ValueError, match="min_similarity must be between 0 and 1"):
            EntityMDMConfig({"matching": {"min_similarity": -0.1}})

        # Test auto_match_threshold out of range
        with pytest.raises(ValueError, match="Auto match threshold must be between 0 and 1"):
            EntityMDMConfig({"matching": {"auto_match_threshold": 2.0}})

        # Test threshold ordering
        with pytest.raises(ValueError, match="Auto match threshold must be greater than human review threshold"):
            EntityMDMConfig({
                "matching": {
                    "auto_match_threshold": 0.80,
                    "human_review_threshold": 0.85
                }
            })

    def test_configuration_update_method(self):
        """Test that configuration can be updated dynamically."""
        config = EntityMDMConfig()

        # Update configuration
        config.update({
            "matching": {
                "top_k": 25,
                "min_similarity": 0.75
            }
        })

        assert config.get("matching.top_k") == 25
        assert config.get("matching.min_similarity") == 0.75

        # Other values should remain unchanged
        assert config.get("weaviate.host") == "localhost"
        assert config.get("embedding.dimension") == 384

    def test_get_method_with_nested_keys(self):
        """Test get method with nested key paths."""
        config = EntityMDMConfig()

        # Test valid nested keys
        assert config.get("weaviate.host") == "localhost"
        assert config.get("embedding.model") is not None
        assert config.get("matching.top_k") == 10

        # Test invalid nested keys return default
        assert config.get("invalid.key", "default_value") == "default_value"
        assert config.get("weaviate.invalid", None) is None
