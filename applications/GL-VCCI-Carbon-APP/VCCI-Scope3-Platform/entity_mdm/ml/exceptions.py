# -*- coding: utf-8 -*-
"""
Custom exceptions for Entity Resolution ML system.

This module defines all custom exceptions used throughout the ML pipeline,
providing clear error handling and debugging capabilities.

Author: GreenLang AI
Phase: 5 - Entity Resolution ML
"""

from typing import Optional, Dict, Any


class EntityResolutionMLException(Exception):
    """Base exception for all Entity Resolution ML errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for logging/monitoring
            details: Additional context about the error
        """
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for logging.

        Returns:
            Dictionary representation of the exception
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ModelNotTrainedException(EntityResolutionMLException):
    """Raised when attempting to use an untrained model."""

    def __init__(
        self,
        model_name: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            model_name: Name of the untrained model
            message: Custom error message
            details: Additional context
        """
        default_message = f"Model '{model_name}' has not been trained. Call train() first."
        super().__init__(
            message or default_message,
            error_code="MODEL_NOT_TRAINED",
            details={**(details or {}), "model_name": model_name},
        )


class InsufficientCandidatesException(EntityResolutionMLException):
    """Raised when candidate generation returns too few results."""

    def __init__(
        self,
        found: int,
        required: int,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            found: Number of candidates found
            required: Minimum number of candidates required
            message: Custom error message
            details: Additional context
        """
        default_message = (
            f"Insufficient candidates: found {found}, required at least {required}"
        )
        super().__init__(
            message or default_message,
            error_code="INSUFFICIENT_CANDIDATES",
            details={**(details or {}), "found": found, "required": required},
        )


class VectorStoreException(EntityResolutionMLException):
    """Raised when vector store operations fail."""

    def __init__(
        self,
        operation: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            operation: The operation that failed (e.g., 'index', 'search', 'delete')
            message: Custom error message
            details: Additional context
        """
        default_message = f"Vector store operation '{operation}' failed"
        super().__init__(
            message or default_message,
            error_code="VECTOR_STORE_ERROR",
            details={**(details or {}), "operation": operation},
        )


class EmbeddingException(EntityResolutionMLException):
    """Raised when embedding generation fails."""

    def __init__(
        self,
        text: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            text: The text that failed to embed (truncated for logging)
            message: Custom error message
            details: Additional context
        """
        default_message = "Failed to generate embeddings"
        truncated_text = text[:100] + "..." if text and len(text) > 100 else text
        super().__init__(
            message or default_message,
            error_code="EMBEDDING_ERROR",
            details={**(details or {}), "text_sample": truncated_text},
        )


class MatchingException(EntityResolutionMLException):
    """Raised when pairwise matching fails."""

    def __init__(
        self,
        entity1_id: Optional[str] = None,
        entity2_id: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            entity1_id: ID of first entity in failed match
            entity2_id: ID of second entity in failed match
            message: Custom error message
            details: Additional context
        """
        default_message = "Pairwise matching operation failed"
        super().__init__(
            message or default_message,
            error_code="MATCHING_ERROR",
            details={
                **(details or {}),
                "entity1_id": entity1_id,
                "entity2_id": entity2_id,
            },
        )


class TrainingException(EntityResolutionMLException):
    """Raised when model training fails."""

    def __init__(
        self,
        epoch: Optional[int] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            epoch: Epoch number where training failed
            message: Custom error message
            details: Additional context
        """
        default_message = "Model training failed"
        super().__init__(
            message or default_message,
            error_code="TRAINING_ERROR",
            details={**(details or {}), "epoch": epoch},
        )


class EvaluationException(EntityResolutionMLException):
    """Raised when model evaluation fails."""

    def __init__(
        self,
        metric: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            metric: The metric that failed to compute
            message: Custom error message
            details: Additional context
        """
        default_message = "Model evaluation failed"
        super().__init__(
            message or default_message,
            error_code="EVALUATION_ERROR",
            details={**(details or {}), "metric": metric},
        )
