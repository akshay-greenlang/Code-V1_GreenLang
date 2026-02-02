# -*- coding: utf-8 -*-
"""
Scope3CalculatorAgent Exceptions
GL-VCCI Scope 3 Platform

Custom exceptions for the Scope3CalculatorAgent with detailed error context
and recovery suggestions.

Version: 1.0.0
Date: 2025-10-30
"""

from typing import Optional, List, Dict, Any


class CalculatorError(Exception):
    """Base exception for all calculator errors."""

    def __init__(
        self,
        message: str,
        category: Optional[int] = None,
        recovery_suggestion: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize calculator error.

        Args:
            message: Error message
            category: Scope 3 category (1-15)
            recovery_suggestion: Suggestion for recovery
            context: Additional error context
        """
        self.message = message
        self.category = category
        self.recovery_suggestion = recovery_suggestion
        self.context = context or {}

        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format complete error message."""
        parts = [self.message]

        if self.category:
            parts.append(f"Category: {self.category}")

        if self.recovery_suggestion:
            parts.append(f"Suggestion: {self.recovery_suggestion}")

        if self.context:
            parts.append(f"Context: {self.context}")

        return " | ".join(parts)


class DataValidationError(CalculatorError):
    """Raised when input data fails validation."""

    def __init__(
        self,
        field: str,
        value: Any,
        reason: str,
        category: Optional[int] = None
    ):
        """
        Initialize data validation error.

        Args:
            field: Field name that failed validation
            value: Invalid value
            reason: Validation failure reason
            category: Scope 3 category
        """
        message = f"Validation failed for field '{field}': {reason}"
        context = {"field": field, "value": value, "reason": reason}
        recovery_suggestion = "Check input data format and completeness"

        super().__init__(message, category, recovery_suggestion, context)


class EmissionFactorNotFoundError(CalculatorError):
    """Raised when required emission factor cannot be resolved."""

    def __init__(
        self,
        product: str,
        region: str,
        category: int,
        tried_sources: Optional[List[str]] = None
    ):
        """
        Initialize emission factor not found error.

        Args:
            product: Product name
            region: Region code
            category: Scope 3 category
            tried_sources: Sources that were tried
        """
        message = f"Emission factor not found for product '{product}' in region '{region}'"
        context = {
            "product": product,
            "region": region,
            "tried_sources": tried_sources or []
        }
        recovery_suggestion = (
            "Try using a proxy factor or spend-based method. "
            "Check product name spelling or use broader category."
        )

        super().__init__(message, category, recovery_suggestion, context)


class CalculationError(CalculatorError):
    """Raised when calculation logic fails."""

    def __init__(
        self,
        calculation_type: str,
        reason: str,
        category: Optional[int] = None,
        input_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize calculation error.

        Args:
            calculation_type: Type of calculation that failed
            reason: Failure reason
            category: Scope 3 category
            input_data: Input data that caused failure
        """
        message = f"Calculation failed for {calculation_type}: {reason}"
        context = {
            "calculation_type": calculation_type,
            "reason": reason,
            "input_data": input_data
        }
        recovery_suggestion = "Verify input data units and values are within valid ranges"

        super().__init__(message, category, recovery_suggestion, context)


class ISO14083ComplianceError(CalculatorError):
    """Raised when ISO 14083 compliance check fails."""

    def __init__(
        self,
        test_case: str,
        expected: float,
        actual: float,
        tolerance: float = 0.001
    ):
        """
        Initialize ISO 14083 compliance error.

        Args:
            test_case: Test case identifier
            expected: Expected result
            actual: Actual result
            tolerance: Acceptable tolerance
        """
        variance = abs(actual - expected)
        message = (
            f"ISO 14083 compliance failed for test '{test_case}': "
            f"expected {expected:.6f}, got {actual:.6f}, "
            f"variance {variance:.6f} exceeds tolerance {tolerance}"
        )
        context = {
            "test_case": test_case,
            "expected": expected,
            "actual": actual,
            "variance": variance,
            "tolerance": tolerance
        }
        recovery_suggestion = "Review calculation formula and emission factors"

        super().__init__(message, 4, recovery_suggestion, context)


class UncertaintyPropagationError(CalculatorError):
    """Raised when uncertainty propagation fails."""

    def __init__(
        self,
        method: str,
        reason: str,
        category: Optional[int] = None
    ):
        """
        Initialize uncertainty propagation error.

        Args:
            method: Uncertainty method (monte_carlo, analytical)
            reason: Failure reason
            category: Scope 3 category
        """
        message = f"Uncertainty propagation failed for method '{method}': {reason}"
        context = {"method": method, "reason": reason}
        recovery_suggestion = "Check uncertainty parameters (must be positive, reasonable values)"

        super().__init__(message, category, recovery_suggestion, context)


class ProvenanceError(CalculatorError):
    """Raised when provenance chain cannot be built."""

    def __init__(
        self,
        reason: str,
        calculation_id: Optional[str] = None,
        category: Optional[int] = None
    ):
        """
        Initialize provenance error.

        Args:
            reason: Failure reason
            calculation_id: Calculation identifier
            category: Scope 3 category
        """
        message = f"Provenance chain building failed: {reason}"
        context = {"calculation_id": calculation_id, "reason": reason}
        recovery_suggestion = "Ensure all calculation inputs have proper metadata"

        super().__init__(message, category, recovery_suggestion, context)


class TierFallbackError(CalculatorError):
    """Raised when tier fallback logic fails."""

    def __init__(
        self,
        attempted_tiers: List[int],
        reason: str,
        category: int = 1
    ):
        """
        Initialize tier fallback error.

        Args:
            attempted_tiers: Tiers that were attempted
            reason: Failure reason
            category: Scope 3 category
        """
        message = (
            f"Tier fallback failed after attempting tiers {attempted_tiers}: {reason}"
        )
        context = {"attempted_tiers": attempted_tiers, "reason": reason}
        recovery_suggestion = (
            "Provide supplier-specific PCF data (Tier 1) or "
            "ensure spend data is available (Tier 3)"
        )

        super().__init__(message, category, recovery_suggestion, context)


class ProductCategorizationError(CalculatorError):
    """Raised when product categorization fails."""

    def __init__(
        self,
        product: str,
        reason: str,
        category: int = 1
    ):
        """
        Initialize product categorization error.

        Args:
            product: Product name
            reason: Failure reason
            category: Scope 3 category
        """
        message = f"Product categorization failed for '{product}': {reason}"
        context = {"product": product, "reason": reason}
        recovery_suggestion = (
            "Provide NAICS or ISIC code manually, or use more specific product name"
        )

        super().__init__(message, category, recovery_suggestion, context)


class TransportModeError(CalculatorError):
    """Raised when transport mode is invalid or unsupported."""

    def __init__(
        self,
        transport_mode: str,
        supported_modes: List[str]
    ):
        """
        Initialize transport mode error.

        Args:
            transport_mode: Invalid transport mode
            supported_modes: List of supported modes
        """
        message = (
            f"Invalid transport mode '{transport_mode}'. "
            f"Supported modes: {', '.join(supported_modes)}"
        )
        context = {
            "transport_mode": transport_mode,
            "supported_modes": supported_modes
        }
        recovery_suggestion = f"Use one of: {', '.join(supported_modes)}"

        super().__init__(message, 4, recovery_suggestion, context)


class OPAPolicyError(CalculatorError):
    """Raised when OPA policy evaluation fails."""

    def __init__(
        self,
        policy_path: str,
        reason: str,
        category: Optional[int] = None
    ):
        """
        Initialize OPA policy error.

        Args:
            policy_path: OPA policy path
            reason: Failure reason
            category: Scope 3 category
        """
        message = f"OPA policy evaluation failed for '{policy_path}': {reason}"
        context = {"policy_path": policy_path, "reason": reason}
        recovery_suggestion = "Check OPA server connectivity and policy syntax"

        super().__init__(message, category, recovery_suggestion, context)


class BatchProcessingError(CalculatorError):
    """Raised when batch processing encounters errors."""

    def __init__(
        self,
        total_records: int,
        failed_records: int,
        failure_details: List[Dict[str, Any]],
        category: Optional[int] = None
    ):
        """
        Initialize batch processing error.

        Args:
            total_records: Total number of records
            failed_records: Number of failed records
            failure_details: Details of failures
            category: Scope 3 category
        """
        message = (
            f"Batch processing completed with errors: "
            f"{failed_records}/{total_records} records failed"
        )
        context = {
            "total_records": total_records,
            "failed_records": failed_records,
            "success_rate": (total_records - failed_records) / total_records if total_records > 0 else 0,
            "failure_details": failure_details[:10]  # Limit to first 10
        }
        recovery_suggestion = "Review failed records and correct input data"

        super().__init__(message, category, recovery_suggestion, context)


__all__ = [
    "CalculatorError",
    "DataValidationError",
    "EmissionFactorNotFoundError",
    "CalculationError",
    "ISO14083ComplianceError",
    "UncertaintyPropagationError",
    "ProvenanceError",
    "TierFallbackError",
    "ProductCategorizationError",
    "TransportModeError",
    "OPAPolicyError",
    "BatchProcessingError",
]
