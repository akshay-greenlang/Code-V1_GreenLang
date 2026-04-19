# -*- coding: utf-8 -*-
# GL-VCCI ML Module
# Spend Classification ML System - Module Exports

"""
ML Module
=========

Machine Learning module for Scope 3 spend classification.

This module provides a complete ML-based classification system for
categorizing procurement spend into 15 GHG Protocol Scope 3 categories.

Features:
--------
- LLM-based classification (OpenAI GPT-3.5/GPT-4, Anthropic Claude)
- Rule-based fallback classification
- Confidence-based routing
- Redis caching (30-day TTL)
- Cost tracking
- Batch processing
- SOC 2 compliant audit logging
- Evaluation framework

Quick Start:
-----------
```python
from utils.ml import SpendClassifier, MLConfig

# Initialize classifier
config = MLConfig()
classifier = SpendClassifier(config)

# Classify spend
result = await classifier.classify(
    description="Office furniture purchase",
    amount=5000.0
)

print(f"Category: {result.category}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Needs review: {result.needs_human_review}")

# Close connections
await classifier.close()
```

Convenience Functions:
---------------------
```python
from utils.ml import classify_spend, classify_spend_batch

# Single classification
result = await classify_spend("Flight to customer meeting")

# Batch classification
results = await classify_spend_batch([
    "Office supplies purchase",
    "Electricity bill payment",
    "Freight shipping to warehouse"
])
```

Configuration:
-------------
```python
from utils.ml import MLConfig, LLMConfig

# Custom configuration
config = MLConfig(
    llm=LLMConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.1
    )
)
```

Training & Evaluation:
---------------------
```python
from utils.ml import TrainingDataLoader, ModelEvaluator

# Load training data
loader = TrainingDataLoader()
dataset = loader.load_csv("data/training/spend_labels.csv")

# Split data
train, val, test = loader.split_dataset(dataset)

# Evaluate model
evaluator = ModelEvaluator()
results = await evaluator.evaluate_classifier(classifier, test)

print(f"Accuracy: {results.overall_accuracy:.2%}")
print(f"Macro F1: {results.macro_f1:.4f}")
```

Architecture:
------------
1. Configuration Layer (config.py)
   - Pydantic models for configuration
   - Environment variable support
   - Multi-provider LLM settings

2. LLM Client Layer (llm_client.py)
   - OpenAI/Anthropic API integration
   - Redis caching
   - Exponential backoff retry
   - Cost tracking

3. Rules Engine Layer (rules_engine.py)
   - Keyword matching
   - Regex patterns
   - Fuzzy matching
   - Confidence scoring

4. Classification Layer (spend_classification.py)
   - Hybrid LLM + rules approach
   - Confidence-based routing
   - Human review flagging
   - Batch processing

5. Training & Evaluation Layer
   - Data loading (training_data.py)
   - Model evaluation (evaluation.py)
   - Metrics calculation
   - Error analysis

Scope 3 Categories:
------------------
Category 1: Purchased Goods & Services
Category 2: Capital Goods
Category 3: Fuel and Energy Related Activities
Category 4: Upstream Transportation & Distribution
Category 5: Waste Generated in Operations
Category 6: Business Travel
Category 7: Employee Commuting
Category 8: Upstream Leased Assets
Category 9: Downstream Transportation & Distribution
Category 10: Processing of Sold Products
Category 11: Use of Sold Products
Category 12: End-of-Life Treatment of Sold Products
Category 13: Downstream Leased Assets
Category 14: Franchises
Category 15: Investments

Target Performance:
------------------
- Classification Accuracy: ≥90%
- Confidence Threshold: 0.85
- Human Review Threshold: 0.5
- Cache Hit Rate: >70%
- Average Latency: <2s per classification
"""

__version__ = "1.0.0"

# Configuration
from .config import (
    LLMConfig,
    LLMProvider,
    MLConfig,
    ClassificationConfig,
    RulesEngineConfig,
    CacheConfig,
    CostTrackingConfig,
    Scope3Category,
    load_config,
    get_default_config,
)

# Exceptions
from .exceptions import (
    MLException,
    LLMException,
    LLMProviderException,
    LLMRateLimitException,
    LLMTimeoutException,
    LLMTokenLimitException,
    RulesEngineException,
    InvalidRuleException,
    RuleEvaluationException,
    ClassificationException,
    LowConfidenceException,
    AmbiguousClassificationException,
    ClassificationTimeoutException,
    InvalidCategoryException,
    TrainingDataException,
    InvalidLabelException,
    InsufficientDataException,
)

# LLM Client
from .llm_client import (
    LLMClient,
    ClassificationResult as LLMClassificationResult,
)

# Rules Engine
from .rules_engine import (
    RulesEngine,
    RuleMatch,
)

# Spend Classification
from .spend_classification import (
    SpendClassifier,
    SpendItem,
    ClassificationResult,
    classify_spend,
    classify_spend_batch,
)

# Training Data
from .training_data import (
    TrainingDataLoader,
    TrainingDataset,
    TrainingExample,
)

# Evaluation
from .evaluation import (
    ModelEvaluator,
    EvaluationResults,
    CategoryMetrics,
    ErrorAnalysis,
)

__all__ = [
    # Version
    "__version__",

    # Configuration
    "MLConfig",
    "LLMConfig",
    "LLMProvider",
    "ClassificationConfig",
    "RulesEngineConfig",
    "CacheConfig",
    "CostTrackingConfig",
    "Scope3Category",
    "load_config",
    "get_default_config",

    # Exceptions
    "MLException",
    "LLMException",
    "LLMProviderException",
    "LLMRateLimitException",
    "LLMTimeoutException",
    "LLMTokenLimitException",
    "RulesEngineException",
    "InvalidRuleException",
    "RuleEvaluationException",
    "ClassificationException",
    "LowConfidenceException",
    "AmbiguousClassificationException",
    "ClassificationTimeoutException",
    "InvalidCategoryException",
    "TrainingDataException",
    "InvalidLabelException",
    "InsufficientDataException",

    # LLM Client
    "LLMClient",
    "LLMClassificationResult",

    # Rules Engine
    "RulesEngine",
    "RuleMatch",

    # Spend Classification (Main API)
    "SpendClassifier",
    "SpendItem",
    "ClassificationResult",
    "classify_spend",
    "classify_spend_batch",

    # Training Data
    "TrainingDataLoader",
    "TrainingDataset",
    "TrainingExample",

    # Evaluation
    "ModelEvaluator",
    "EvaluationResults",
    "CategoryMetrics",
    "ErrorAnalysis",
]


# ============================================================================
# Module-level convenience functions
# ============================================================================

def get_version() -> str:
    """
    Get ML module version.

    Returns:
        Version string

    Example:
        >>> from utils.ml import get_version
        >>> get_version()
        '1.0.0'
    """
    return __version__


def get_scope3_categories() -> list:
    """
    Get all Scope 3 categories.

    Returns:
        List of category identifiers

    Example:
        >>> from utils.ml import get_scope3_categories
        >>> categories = get_scope3_categories()
        >>> len(categories)
        15
    """
    return Scope3Category.get_all_categories()


def get_category_name(category: str) -> str:
    """
    Get human-readable category name.

    Args:
        category: Category identifier

    Returns:
        Human-readable category name

    Example:
        >>> from utils.ml import get_category_name
        >>> get_category_name("category_1_purchased_goods_services")
        'Purchased Goods & Services'
    """
    return Scope3Category.get_category_name(category)


# ============================================================================
# Module initialization
# ============================================================================

import logging

logger = logging.getLogger(__name__)
logger.info(f"GL-VCCI ML Module v{__version__} loaded")
