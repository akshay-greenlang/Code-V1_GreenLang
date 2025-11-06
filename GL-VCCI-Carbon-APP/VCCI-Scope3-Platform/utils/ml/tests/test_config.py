"""
Tests for Spend Classification ML Configuration.

Tests configuration validation, LLM provider settings, and threshold configuration.

Target: 200+ lines, 10 tests
"""

import pytest
import os
from unittest.mock import patch
from typing import Dict, Any


# Mock config module
class SpendClassificationConfig:
    """Spend classification ML configuration."""

    def __init__(self, config_dict: Dict[str, Any] = None):
        self.config = config_dict or self._load_defaults()
        self._validate()

    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "llm": {
                "provider": os.getenv("LLM_PROVIDER", "openai"),
                "model": os.getenv("LLM_MODEL", "gpt-4"),
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "500")),
                "api_key": os.getenv("LLM_API_KEY", ""),
                "timeout": int(os.getenv("LLM_TIMEOUT", "30"))
            },
            "rules": {
                "enabled": os.getenv("RULES_ENABLED", "true").lower() == "true",
                "fallback_to_llm": os.getenv("RULES_FALLBACK_LLM", "true").lower() == "true",
                "confidence_threshold": float(os.getenv("RULES_CONFIDENCE_THRESHOLD", "0.75"))
            },
            "routing": {
                "high_confidence_threshold": float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.85")),
                "use_rules_first": os.getenv("USE_RULES_FIRST", "true").lower() == "true"
            },
            "categories": 15,
            "cache": {
                "enabled": True,
                "ttl_seconds": int(os.getenv("CACHE_TTL", "3600"))
            }
        }

    def _validate(self):
        """Validate configuration."""
        # Validate LLM config
        if self.config["llm"]["provider"] not in ["openai", "anthropic", "azure"]:
            raise ValueError("LLM provider must be 'openai', 'anthropic', or 'azure'")

        if not (0.0 <= self.config["llm"]["temperature"] <= 2.0):
            raise ValueError("LLM temperature must be between 0.0 and 2.0")

        if self.config["llm"]["max_tokens"] <= 0:
            raise ValueError("LLM max_tokens must be positive")

        # Validate thresholds
        if not (0.0 <= self.config["rules"]["confidence_threshold"] <= 1.0):
            raise ValueError("Rules confidence threshold must be between 0 and 1")

        if not (0.0 <= self.config["routing"]["high_confidence_threshold"] <= 1.0):
            raise ValueError("High confidence threshold must be between 0 and 1")

        # Validate categories
        if self.config["categories"] != 15:
            raise ValueError("Must have exactly 15 Scope 3 categories")

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


# ============================================================================
# TEST SUITE
# ============================================================================

class TestSpendClassificationConfig:
    """Test suite for spend classification configuration."""

    def test_default_configuration_loads(self):
        """Test that default configuration loads successfully."""
        config = SpendClassificationConfig()

        assert config is not None
        assert "llm" in config.config
        assert "rules" in config.config
        assert "routing" in config.config

    def test_llm_default_values(self):
        """Test LLM default configuration values."""
        config = SpendClassificationConfig()

        assert config.get("llm.provider") == "openai"
        assert config.get("llm.model") == "gpt-4"
        assert config.get("llm.temperature") == 0.1
        assert config.get("llm.max_tokens") == 500

    def test_rules_default_values(self):
        """Test rules engine default configuration values."""
        config = SpendClassificationConfig()

        assert config.get("rules.enabled") is True
        assert config.get("rules.fallback_to_llm") is True
        assert config.get("rules.confidence_threshold") == 0.75

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "anthropic",
        "LLM_MODEL": "claude-3-opus",
        "LLM_TEMPERATURE": "0.2"
    })
    def test_environment_variables_override(self):
        """Test that environment variables override defaults."""
        config = SpendClassificationConfig()

        assert config.get("llm.provider") == "anthropic"
        assert config.get("llm.model") == "claude-3-opus"
        assert config.get("llm.temperature") == 0.2

    def test_invalid_llm_provider_raises_error(self):
        """Test that invalid LLM provider raises error."""
        with pytest.raises(ValueError, match="LLM provider must be"):
            SpendClassificationConfig({"llm": {"provider": "invalid"}})

    def test_invalid_temperature_raises_error(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError, match="LLM temperature must be between"):
            SpendClassificationConfig({"llm": {"temperature": 3.0}})

        with pytest.raises(ValueError, match="LLM temperature must be between"):
            SpendClassificationConfig({"llm": {"temperature": -0.1}})

    def test_invalid_threshold_raises_error(self):
        """Test that invalid threshold raises error."""
        with pytest.raises(ValueError, match="confidence threshold must be between"):
            SpendClassificationConfig({"rules": {"confidence_threshold": 1.5}})

    def test_invalid_categories_raises_error(self):
        """Test that invalid categories count raises error."""
        with pytest.raises(ValueError, match="Must have exactly 15 Scope 3 categories"):
            SpendClassificationConfig({"categories": 10})

    def test_get_method_with_nested_keys(self):
        """Test get method with nested key paths."""
        config = SpendClassificationConfig()

        assert config.get("llm.provider") == "openai"
        assert config.get("rules.enabled") is True
        assert config.get("invalid.key", "default") == "default"
