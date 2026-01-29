"""
Threshold configuration for Entity Resolution Pipeline (GL-FOUND-X-003).

This module defines confidence thresholds for entity resolution, including
entity-type-specific thresholds and the margin rule for ambiguity detection.

Thresholds can be loaded from a YAML configuration file or use defaults.

Configuration file format (config/confidence_thresholds.yaml):
    entity_thresholds:
      fuel: 0.95
      material: 0.90
      process: 0.85
      activity: 0.85
      emission_factor: 0.95
      location: 0.90
    margin_threshold: 0.07
    semantic_always_review: true

Example:
    >>> from gl_normalizer_core.resolution.thresholds import (
    ...     ENTITY_THRESHOLDS, MARGIN_THRESHOLD, get_threshold
    ... )
    >>> threshold = get_threshold("fuel")
    >>> assert threshold == 0.95
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any

import yaml

from gl_normalizer_core.resolution.models import EntityType

logger = logging.getLogger(__name__)


# =============================================================================
# Default Threshold Constants
# =============================================================================

DEFAULT_ENTITY_THRESHOLDS: Dict[str, float] = {
    EntityType.FUEL.value: 0.95,
    EntityType.MATERIAL.value: 0.90,
    EntityType.PROCESS.value: 0.85,
    EntityType.ACTIVITY.value: 0.85,
    EntityType.EMISSION_FACTOR.value: 0.95,
    EntityType.LOCATION.value: 0.90,
}
"""Default confidence thresholds by entity type."""

DEFAULT_MARGIN_THRESHOLD: float = 0.07
"""
Default margin threshold for ambiguity detection.

If the difference between top two candidates is less than this value,
the result is flagged for human review (needs_review=True).
"""

DEFAULT_SEMANTIC_ALWAYS_REVIEW: bool = True
"""Whether semantic/LLM matches always require review."""

DEFAULT_FUZZY_MATCH_PENALTY: float = 0.05
"""Penalty applied to fuzzy match scores (reduces confidence)."""

DEFAULT_MIN_FUZZY_SCORE: float = 0.70
"""Minimum score required for fuzzy matches to be considered."""


# =============================================================================
# Configuration Loading
# =============================================================================

class ThresholdConfig:
    """
    Configuration container for resolution thresholds.

    This class manages loading and accessing threshold configuration,
    supporting both file-based configuration and programmatic overrides.

    Attributes:
        entity_thresholds: Dict mapping entity types to confidence thresholds
        margin_threshold: Margin below which results require review
        semantic_always_review: Whether semantic matches always need review
        fuzzy_match_penalty: Penalty applied to fuzzy match scores
        min_fuzzy_score: Minimum score for fuzzy matches

    Example:
        >>> config = ThresholdConfig.load_from_file("config/thresholds.yaml")
        >>> config.get_threshold("fuel")
        0.95
    """

    def __init__(
        self,
        entity_thresholds: Optional[Dict[str, float]] = None,
        margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
        semantic_always_review: bool = DEFAULT_SEMANTIC_ALWAYS_REVIEW,
        fuzzy_match_penalty: float = DEFAULT_FUZZY_MATCH_PENALTY,
        min_fuzzy_score: float = DEFAULT_MIN_FUZZY_SCORE,
    ) -> None:
        """
        Initialize ThresholdConfig.

        Args:
            entity_thresholds: Dict mapping entity types to thresholds
            margin_threshold: Margin threshold for ambiguity detection
            semantic_always_review: Whether semantic matches require review
            fuzzy_match_penalty: Penalty for fuzzy match scores
            min_fuzzy_score: Minimum fuzzy match score
        """
        self.entity_thresholds = entity_thresholds or DEFAULT_ENTITY_THRESHOLDS.copy()
        self.margin_threshold = margin_threshold
        self.semantic_always_review = semantic_always_review
        self.fuzzy_match_penalty = fuzzy_match_penalty
        self.min_fuzzy_score = min_fuzzy_score

    def get_threshold(self, entity_type: str) -> float:
        """
        Get the confidence threshold for an entity type.

        Args:
            entity_type: The entity type (e.g., "fuel", "material")

        Returns:
            float: Confidence threshold for the entity type

        Raises:
            ValueError: If entity type is not recognized

        Example:
            >>> config = ThresholdConfig()
            >>> config.get_threshold("fuel")
            0.95
        """
        entity_type_lower = entity_type.lower()
        if entity_type_lower in self.entity_thresholds:
            return self.entity_thresholds[entity_type_lower]

        # Try to match EntityType enum
        try:
            entity_enum = EntityType(entity_type_lower)
            return self.entity_thresholds.get(
                entity_enum.value,
                0.85,  # Default fallback threshold
            )
        except ValueError:
            logger.warning(
                f"Unknown entity type '{entity_type}', using default threshold 0.85"
            )
            return 0.85

    def is_above_threshold(self, confidence: float, entity_type: str) -> bool:
        """
        Check if a confidence score meets the threshold for an entity type.

        Args:
            confidence: The confidence score to check
            entity_type: The entity type

        Returns:
            bool: True if confidence meets or exceeds the threshold

        Example:
            >>> config = ThresholdConfig()
            >>> config.is_above_threshold(0.96, "fuel")
            True
            >>> config.is_above_threshold(0.94, "fuel")
            False
        """
        threshold = self.get_threshold(entity_type)
        return confidence >= threshold

    def needs_margin_review(self, top_score: float, runner_up_score: float) -> bool:
        """
        Check if the margin between scores requires review.

        Args:
            top_score: Score of the best candidate
            runner_up_score: Score of the second-best candidate

        Returns:
            bool: True if margin is below threshold (needs review)

        Example:
            >>> config = ThresholdConfig()
            >>> config.needs_margin_review(0.92, 0.88)  # margin=0.04 < 0.07
            True
            >>> config.needs_margin_review(0.95, 0.80)  # margin=0.15 > 0.07
            False
        """
        margin = top_score - runner_up_score
        return margin < self.margin_threshold

    @classmethod
    def load_from_file(cls, config_path: str) -> "ThresholdConfig":
        """
        Load threshold configuration from a YAML file.

        Args:
            config_path: Path to the configuration YAML file

        Returns:
            ThresholdConfig: Loaded configuration

        Raises:
            FileNotFoundError: If config file does not exist
            yaml.YAMLError: If config file is invalid YAML

        Example:
            >>> config = ThresholdConfig.load_from_file("config/thresholds.yaml")
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config file: {e}")
            raise

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ThresholdConfig":
        """
        Create ThresholdConfig from a dictionary.

        Supports both flat format and the nested format from
        config/confidence_thresholds.yaml.

        Args:
            data: Configuration dictionary

        Returns:
            ThresholdConfig: Configured instance
        """
        # Handle nested format (from confidence_thresholds.yaml)
        if "classification_thresholds" in data:
            entity_thresholds = cls._extract_classification_thresholds(data)
        else:
            entity_thresholds = data.get("entity_thresholds", DEFAULT_ENTITY_THRESHOLDS)

        # Handle margin rules (nested or flat)
        if "margin_rules" in data and "margin_rule" in data["margin_rules"]:
            margin_threshold = data["margin_rules"]["margin_rule"].get(
                "value", DEFAULT_MARGIN_THRESHOLD
            )
        else:
            margin_threshold = data.get("margin_threshold", DEFAULT_MARGIN_THRESHOLD)

        semantic_always_review = data.get(
            "semantic_always_review", DEFAULT_SEMANTIC_ALWAYS_REVIEW
        )
        fuzzy_match_penalty = data.get(
            "fuzzy_match_penalty", DEFAULT_FUZZY_MATCH_PENALTY
        )

        # Handle min_fuzzy_score from unit_matching_thresholds
        if "unit_matching_thresholds" in data:
            low_conf = data["unit_matching_thresholds"].get("low_confidence_match", {})
            min_fuzzy_score = low_conf.get("threshold", DEFAULT_MIN_FUZZY_SCORE)
        else:
            min_fuzzy_score = data.get("min_fuzzy_score", DEFAULT_MIN_FUZZY_SCORE)

        return cls(
            entity_thresholds=entity_thresholds,
            margin_threshold=margin_threshold,
            semantic_always_review=semantic_always_review,
            fuzzy_match_penalty=fuzzy_match_penalty,
            min_fuzzy_score=min_fuzzy_score,
        )

    @classmethod
    def _extract_classification_thresholds(cls, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract entity thresholds from nested classification_thresholds structure.

        Args:
            data: Full configuration dictionary

        Returns:
            Dict mapping entity types to thresholds
        """
        thresholds = DEFAULT_ENTITY_THRESHOLDS.copy()
        classification = data.get("classification_thresholds", {})

        # Map from config keys to our entity types
        for entity_type in ["fuel", "material", "process"]:
            if entity_type in classification:
                entity_config = classification[entity_type]
                if isinstance(entity_config, dict) and "threshold" in entity_config:
                    thresholds[entity_type] = entity_config["threshold"]
                elif isinstance(entity_config, (int, float)):
                    thresholds[entity_type] = float(entity_config)

        # Also check entity_resolution_thresholds for additional types
        entity_resolution = data.get("entity_resolution_thresholds", {})
        if "emission_factor" in entity_resolution:
            ef_config = entity_resolution["emission_factor"]
            if isinstance(ef_config, dict) and "threshold" in ef_config:
                thresholds["emission_factor"] = ef_config["threshold"]

        return thresholds

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            Dict: Configuration as dictionary
        """
        return {
            "entity_thresholds": self.entity_thresholds,
            "margin_threshold": self.margin_threshold,
            "semantic_always_review": self.semantic_always_review,
            "fuzzy_match_penalty": self.fuzzy_match_penalty,
            "min_fuzzy_score": self.min_fuzzy_score,
        }


# =============================================================================
# Module-Level Configuration
# =============================================================================

def _load_default_config() -> ThresholdConfig:
    """
    Load the default threshold configuration.

    Attempts to load from environment variable or standard config paths.

    Returns:
        ThresholdConfig: Loaded or default configuration
    """
    # Check environment variable for config path
    config_path = os.environ.get("GLNORM_THRESHOLD_CONFIG")

    if config_path:
        try:
            return ThresholdConfig.load_from_file(config_path)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    # Try standard config locations
    standard_paths = [
        "config/confidence_thresholds.yaml",
        "../config/confidence_thresholds.yaml",
        "../../config/confidence_thresholds.yaml",
    ]

    for path in standard_paths:
        if Path(path).exists():
            try:
                return ThresholdConfig.load_from_file(path)
            except Exception as e:
                logger.debug(f"Could not load config from {path}: {e}")

    # Return defaults
    return ThresholdConfig()


# Module-level singleton
_config: Optional[ThresholdConfig] = None


def get_config() -> ThresholdConfig:
    """
    Get the current threshold configuration.

    Returns:
        ThresholdConfig: The current configuration

    Example:
        >>> config = get_config()
        >>> config.margin_threshold
        0.07
    """
    global _config
    if _config is None:
        _config = _load_default_config()
    return _config


def set_config(config: ThresholdConfig) -> None:
    """
    Set the threshold configuration.

    Args:
        config: The configuration to use

    Example:
        >>> custom_config = ThresholdConfig(margin_threshold=0.10)
        >>> set_config(custom_config)
    """
    global _config
    _config = config


def reset_config() -> None:
    """
    Reset the configuration to defaults.

    Example:
        >>> reset_config()
    """
    global _config
    _config = None


# =============================================================================
# Convenience Exports (module-level constants)
# =============================================================================

# These are computed at import time and provide backwards compatibility
ENTITY_THRESHOLDS: Dict[str, float] = DEFAULT_ENTITY_THRESHOLDS.copy()
"""Entity type to confidence threshold mapping."""

MARGIN_THRESHOLD: float = DEFAULT_MARGIN_THRESHOLD
"""Margin threshold for ambiguity detection."""


def get_threshold(entity_type: str) -> float:
    """
    Get the confidence threshold for an entity type.

    This is a convenience function that delegates to the current config.

    Args:
        entity_type: The entity type (e.g., "fuel", "material")

    Returns:
        float: Confidence threshold for the entity type

    Example:
        >>> get_threshold("fuel")
        0.95
        >>> get_threshold("material")
        0.90
        >>> get_threshold("process")
        0.85
    """
    return get_config().get_threshold(entity_type)


def is_above_threshold(confidence: float, entity_type: str) -> bool:
    """
    Check if a confidence score meets the threshold.

    Args:
        confidence: The confidence score to check
        entity_type: The entity type

    Returns:
        bool: True if confidence meets or exceeds the threshold

    Example:
        >>> is_above_threshold(0.96, "fuel")
        True
        >>> is_above_threshold(0.94, "fuel")
        False
    """
    return get_config().is_above_threshold(confidence, entity_type)


def needs_margin_review(top_score: float, runner_up_score: float) -> bool:
    """
    Check if the margin between scores requires review.

    Args:
        top_score: Score of the best candidate
        runner_up_score: Score of the second-best candidate

    Returns:
        bool: True if margin is below threshold (needs review)

    Example:
        >>> needs_margin_review(0.92, 0.88)  # margin=0.04 < 0.07
        True
        >>> needs_margin_review(0.95, 0.80)  # margin=0.15 > 0.07
        False
    """
    return get_config().needs_margin_review(top_score, runner_up_score)


__all__ = [
    # Constants
    "ENTITY_THRESHOLDS",
    "MARGIN_THRESHOLD",
    "DEFAULT_ENTITY_THRESHOLDS",
    "DEFAULT_MARGIN_THRESHOLD",
    "DEFAULT_SEMANTIC_ALWAYS_REVIEW",
    "DEFAULT_FUZZY_MATCH_PENALTY",
    "DEFAULT_MIN_FUZZY_SCORE",
    # Config class
    "ThresholdConfig",
    # Functions
    "get_config",
    "set_config",
    "reset_config",
    "get_threshold",
    "is_above_threshold",
    "needs_margin_review",
]
