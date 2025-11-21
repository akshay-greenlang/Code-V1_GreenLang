# -*- coding: utf-8 -*-
# GL-VCCI ML Module - Custom Exceptions
# Spend Classification ML System - Exception Hierarchy

"""
ML Module Custom Exceptions
============================

Exception hierarchy for the Spend Classification ML system.

Exception Hierarchy:
-------------------
MLException (base)
├── LLMException
│   ├── LLMProviderException
│   ├── LLMRateLimitException
│   ├── LLMTimeoutException
│   └── LLMTokenLimitException
├── RulesEngineException
│   ├── InvalidRuleException
│   └── RuleEvaluationException
├── ClassificationException
│   ├── LowConfidenceException
│   ├── AmbiguousClassificationException
│   └── ClassificationTimeoutException
├── InvalidCategoryException
└── TrainingDataException
    ├── InvalidLabelException
    └── InsufficientDataException

Usage:
------
```python
from utils.ml.exceptions import (
    LLMException,
    LLMRateLimitException,
    InvalidCategoryException
)

try:
    result = classifier.classify(description)
except LLMRateLimitException as e:
    # Handle rate limiting with exponential backoff
    logger.warning(f"Rate limited: {e.message}")
    time.sleep(e.retry_after)
except InvalidCategoryException as e:
    # Handle invalid category
    logger.error(f"Invalid category: {e.category}")
    raise
```
"""

from typing import Any, Dict, Optional


class MLException(Exception):
    """
    Base exception for all ML-related errors.

    Attributes:
        message: Human-readable error message
        details: Additional error context (dict)
        original_error: Original exception if wrapping another error
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize ML exception.

        Args:
            message: Human-readable error message
            details: Additional error context
            original_error: Original exception if wrapping
        """
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(message)

    def __str__(self) -> str:
        """Return string representation with details."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None
        }


# ============================================================================
# LLM Exceptions
# ============================================================================

class LLMException(MLException):
    """
    Base exception for LLM-related errors.

    Raised when LLM API calls fail or return unexpected results.
    """
    pass


class LLMProviderException(LLMException):
    """
    LLM provider error (API unavailable, authentication failed, etc.).

    Attributes:
        provider: LLM provider name (e.g., "openai", "anthropic")
        status_code: HTTP status code if applicable
    """

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize LLM provider exception.

        Args:
            message: Error message
            provider: LLM provider name
            status_code: HTTP status code
            details: Additional context
            original_error: Original exception
        """
        self.provider = provider
        self.status_code = status_code
        super().__init__(
            message=message,
            details={**(details or {}), "provider": provider, "status_code": status_code},
            original_error=original_error
        )


class LLMRateLimitException(LLMException):
    """
    LLM rate limit exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying
        provider: LLM provider name
    """

    def __init__(
        self,
        message: str = "LLM rate limit exceeded",
        retry_after: int = 60,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize rate limit exception.

        Args:
            message: Error message
            retry_after: Seconds to wait before retry
            provider: LLM provider name
            details: Additional context
        """
        self.retry_after = retry_after
        self.provider = provider
        super().__init__(
            message=message,
            details={**(details or {}), "retry_after": retry_after, "provider": provider}
        )


class LLMTimeoutException(LLMException):
    """
    LLM request timeout.

    Attributes:
        timeout_seconds: Timeout duration that was exceeded
    """

    def __init__(
        self,
        message: str = "LLM request timed out",
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize timeout exception.

        Args:
            message: Error message
            timeout_seconds: Timeout duration
            details: Additional context
        """
        self.timeout_seconds = timeout_seconds
        super().__init__(
            message=message,
            details={**(details or {}), "timeout_seconds": timeout_seconds}
        )


class LLMTokenLimitException(LLMException):
    """
    LLM token limit exceeded.

    Attributes:
        token_count: Number of tokens in request
        token_limit: Maximum allowed tokens
    """

    def __init__(
        self,
        message: str = "LLM token limit exceeded",
        token_count: Optional[int] = None,
        token_limit: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize token limit exception.

        Args:
            message: Error message
            token_count: Actual token count
            token_limit: Maximum allowed tokens
            details: Additional context
        """
        self.token_count = token_count
        self.token_limit = token_limit
        super().__init__(
            message=message,
            details={
                **(details or {}),
                "token_count": token_count,
                "token_limit": token_limit
            }
        )


# ============================================================================
# Rules Engine Exceptions
# ============================================================================

class RulesEngineException(MLException):
    """
    Base exception for rule-based classification errors.
    """
    pass


class InvalidRuleException(RulesEngineException):
    """
    Invalid rule definition.

    Attributes:
        rule_name: Name of invalid rule
    """

    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize invalid rule exception.

        Args:
            message: Error message
            rule_name: Name of invalid rule
            details: Additional context
        """
        self.rule_name = rule_name
        super().__init__(
            message=message,
            details={**(details or {}), "rule_name": rule_name}
        )


class RuleEvaluationException(RulesEngineException):
    """
    Rule evaluation failed.

    Attributes:
        rule_name: Name of rule that failed
        input_data: Input data that caused failure
    """

    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize rule evaluation exception.

        Args:
            message: Error message
            rule_name: Name of rule that failed
            input_data: Input data
            details: Additional context
            original_error: Original exception
        """
        self.rule_name = rule_name
        self.input_data = input_data
        super().__init__(
            message=message,
            details={**(details or {}), "rule_name": rule_name},
            original_error=original_error
        )


# ============================================================================
# Classification Exceptions
# ============================================================================

class ClassificationException(MLException):
    """
    Base exception for classification errors.
    """
    pass


class LowConfidenceException(ClassificationException):
    """
    Classification confidence below threshold.

    Attributes:
        confidence: Classification confidence score
        threshold: Minimum required confidence
        description: Input description
    """

    def __init__(
        self,
        message: str = "Classification confidence below threshold",
        confidence: Optional[float] = None,
        threshold: Optional[float] = None,
        description: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize low confidence exception.

        Args:
            message: Error message
            confidence: Actual confidence score
            threshold: Required confidence threshold
            description: Input description
            details: Additional context
        """
        self.confidence = confidence
        self.threshold = threshold
        self.description = description
        super().__init__(
            message=message,
            details={
                **(details or {}),
                "confidence": confidence,
                "threshold": threshold,
                "description": description[:100] if description else None
            }
        )


class AmbiguousClassificationException(ClassificationException):
    """
    Multiple categories have similar confidence scores.

    Attributes:
        candidates: List of (category, confidence) tuples
        description: Input description
    """

    def __init__(
        self,
        message: str = "Multiple categories have similar confidence",
        candidates: Optional[list] = None,
        description: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ambiguous classification exception.

        Args:
            message: Error message
            candidates: List of candidate (category, confidence) tuples
            description: Input description
            details: Additional context
        """
        self.candidates = candidates or []
        self.description = description
        super().__init__(
            message=message,
            details={
                **(details or {}),
                "candidates": self.candidates,
                "description": description[:100] if description else None
            }
        )


class ClassificationTimeoutException(ClassificationException):
    """
    Classification request timed out.

    Attributes:
        timeout_seconds: Timeout duration
    """

    def __init__(
        self,
        message: str = "Classification timed out",
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize classification timeout exception.

        Args:
            message: Error message
            timeout_seconds: Timeout duration
            details: Additional context
        """
        self.timeout_seconds = timeout_seconds
        super().__init__(
            message=message,
            details={**(details or {}), "timeout_seconds": timeout_seconds}
        )


# ============================================================================
# Category Exceptions
# ============================================================================

class InvalidCategoryException(MLException):
    """
    Invalid Scope 3 category.

    Attributes:
        category: Invalid category identifier
        valid_categories: List of valid categories
    """

    def __init__(
        self,
        message: str,
        category: Optional[str] = None,
        valid_categories: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize invalid category exception.

        Args:
            message: Error message
            category: Invalid category
            valid_categories: List of valid categories
            details: Additional context
        """
        self.category = category
        self.valid_categories = valid_categories
        super().__init__(
            message=message,
            details={
                **(details or {}),
                "category": category,
                "valid_categories": valid_categories
            }
        )


# ============================================================================
# Training Data Exceptions
# ============================================================================

class TrainingDataException(MLException):
    """
    Base exception for training data errors.
    """
    pass


class InvalidLabelException(TrainingDataException):
    """
    Invalid label in training data.

    Attributes:
        label: Invalid label value
        valid_labels: List of valid labels
    """

    def __init__(
        self,
        message: str,
        label: Optional[str] = None,
        valid_labels: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize invalid label exception.

        Args:
            message: Error message
            label: Invalid label
            valid_labels: List of valid labels
            details: Additional context
        """
        self.label = label
        self.valid_labels = valid_labels
        super().__init__(
            message=message,
            details={
                **(details or {}),
                "label": label,
                "valid_labels": valid_labels
            }
        )


class InsufficientDataException(TrainingDataException):
    """
    Insufficient training data.

    Attributes:
        data_count: Actual data count
        required_count: Minimum required data count
    """

    def __init__(
        self,
        message: str = "Insufficient training data",
        data_count: Optional[int] = None,
        required_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize insufficient data exception.

        Args:
            message: Error message
            data_count: Actual data count
            required_count: Required minimum count
            details: Additional context
        """
        self.data_count = data_count
        self.required_count = required_count
        super().__init__(
            message=message,
            details={
                **(details or {}),
                "data_count": data_count,
                "required_count": required_count
            }
        )
