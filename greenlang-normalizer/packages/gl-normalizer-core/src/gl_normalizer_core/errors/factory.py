"""
GLNORM Error Factory for GL-FOUND-X-003.

This module provides a factory for creating consistent, well-structured error
responses with automatic suggestion generation based on error type. The factory
ensures all errors include proper context, suggestions, and audit information.

The factory implements the following features:
- Automatic suggestion generation based on error code
- Context enrichment from execution environment
- Candidate extraction for ambiguous errors
- Documentation URL generation
- Audit trail preparation

Example:
    >>> from gl_normalizer_core.errors.factory import GLNORMErrorFactory
    >>> factory = GLNORMErrorFactory()
    >>> error = factory.create_error(
    ...     code=GLNORMErrorCode.E305_GWP_VERSION_MISSING,
    ...     message="GWP version required",
    ...     details={"source_unit": "kg-ch4", "target_unit": "kg-co2e"}
    ... )
"""

import hashlib
import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union
from uuid import UUID, uuid4

from .codes import (
    CATEGORY_NAMES,
    GLNORMErrorCode,
    get_error_code_by_value,
)
from .response import (
    AuditableError,
    ErrorCandidate,
    ErrorContext,
    ErrorDetail,
    ErrorSuggestion,
    GLNORMBatchErrorResponse,
    GLNORMErrorResponse,
    GLNORMValidationErrorResponse,
    ValidationErrorItem,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Suggestion Templates
# =============================================================================

SUGGESTION_TEMPLATES: Dict[GLNORMErrorCode, List[Dict[str, Any]]] = {
    # E1xx: Unit Parsing Errors
    GLNORMErrorCode.E100_UNIT_PARSE_FAILED: [
        {
            "action": "review_input",
            "description": "Check the unit string for typos or unsupported characters",
            "documentation_url": "https://docs.greenlang.io/normalizer/units/parsing"
        },
        {
            "action": "use_canonical",
            "description": "Use canonical unit format (e.g., 'kg' instead of 'KG')",
            "field": "unit",
            "example": "kg"
        }
    ],
    GLNORMErrorCode.E101_UNKNOWN_UNIT: [
        {
            "action": "check_vocabulary",
            "description": "Verify the unit exists in the configured vocabulary",
            "documentation_url": "https://docs.greenlang.io/normalizer/vocabularies"
        },
        {
            "action": "add_custom_unit",
            "description": "Register the unit in your custom vocabulary if valid",
            "documentation_url": "https://docs.greenlang.io/normalizer/custom-units"
        }
    ],
    GLNORMErrorCode.E102_INVALID_PREFIX: [
        {
            "action": "use_valid_prefix",
            "description": "Use a valid SI prefix (k, M, G, T, m, u, n, p)",
            "field": "prefix",
            "example": "k"
        }
    ],
    GLNORMErrorCode.E104_AMBIGUOUS_UNIT: [
        {
            "action": "select_candidate",
            "description": "Select the correct interpretation from the candidates below"
        }
    ],

    # E2xx: Dimension Errors
    GLNORMErrorCode.E200_DIMENSION_MISMATCH: [
        {
            "action": "verify_units",
            "description": "Verify source and target units have compatible dimensions",
            "documentation_url": "https://docs.greenlang.io/normalizer/dimensions"
        }
    ],
    GLNORMErrorCode.E202_DIMENSION_INCOMPATIBLE: [
        {
            "action": "review_conversion",
            "description": "Cannot convert between fundamentally different dimensions",
            "documentation_url": "https://docs.greenlang.io/normalizer/dimensions#compatible"
        }
    ],

    # E3xx: Conversion Errors
    GLNORMErrorCode.E300_CONVERSION_NOT_SUPPORTED: [
        {
            "action": "check_path",
            "description": "Verify a conversion path exists between the units",
            "documentation_url": "https://docs.greenlang.io/normalizer/conversions"
        }
    ],
    GLNORMErrorCode.E301_MISSING_REFERENCE_CONDITIONS: [
        {
            "action": "provide_value",
            "description": "Provide reference conditions for the conversion",
            "field": "reference_conditions",
            "example": '{"temperature_k": 288.15, "pressure_pa": 101325}'
        }
    ],
    GLNORMErrorCode.E302_INVALID_REFERENCE_CONDITIONS: [
        {
            "action": "correct_value",
            "description": "Provide valid reference conditions within acceptable ranges",
            "documentation_url": "https://docs.greenlang.io/normalizer/reference-conditions"
        }
    ],
    GLNORMErrorCode.E305_GWP_VERSION_MISSING: [
        {
            "action": "provide_value",
            "description": "Specify the GWP version for CO2 equivalent conversion",
            "field": "gwp_version"
        }
    ],
    GLNORMErrorCode.E306_BASIS_MISSING: [
        {
            "action": "provide_value",
            "description": "Specify the energy basis (HHV or LHV) for fuel conversions",
            "field": "basis"
        }
    ],
    GLNORMErrorCode.E307_CONVERSION_FACTOR_DEPRECATED: [
        {
            "action": "upgrade",
            "description": "Update to the latest conversion factor version",
            "documentation_url": "https://docs.greenlang.io/normalizer/deprecations"
        }
    ],

    # E4xx: Entity Resolution Errors
    GLNORMErrorCode.E400_REFERENCE_NOT_FOUND: [
        {
            "action": "check_reference",
            "description": "Verify the entity reference ID is correct",
            "documentation_url": "https://docs.greenlang.io/normalizer/entities"
        }
    ],
    GLNORMErrorCode.E401_REFERENCE_AMBIGUOUS: [
        {
            "action": "select_candidate",
            "description": "Multiple entities match - select the correct one"
        }
    ],
    GLNORMErrorCode.E402_ENTITY_DEPRECATED: [
        {
            "action": "use_replacement",
            "description": "Use the recommended replacement entity"
        }
    ],
    GLNORMErrorCode.E403_LOW_CONFIDENCE_MATCH: [
        {
            "action": "review",
            "description": "Review the match and confirm or reject"
        },
        {
            "action": "provide_explicit",
            "description": "Provide an explicit entity reference to avoid ambiguity",
            "field": "entity_id"
        }
    ],
    GLNORMErrorCode.E407_LLM_CANDIDATE_ONLY: [
        {
            "action": "review",
            "description": "LLM-suggested match requires human verification"
        },
        {
            "action": "add_to_vocabulary",
            "description": "Add a deterministic mapping to avoid future LLM lookups",
            "documentation_url": "https://docs.greenlang.io/normalizer/vocabulary-management"
        }
    ],

    # E5xx: Vocabulary Errors
    GLNORMErrorCode.E500_VOCABULARY_VERSION_MISMATCH: [
        {
            "action": "upgrade_vocabulary",
            "description": "Update the vocabulary to a compatible version"
        }
    ],
    GLNORMErrorCode.E503_VOCABULARY_EXPIRED: [
        {
            "action": "refresh_vocabulary",
            "description": "Refresh the vocabulary from the registry"
        }
    ],
    GLNORMErrorCode.E504_GOVERNANCE_REQUIRED: [
        {
            "action": "submit_request",
            "description": "Submit a governance request for this change",
            "documentation_url": "https://docs.greenlang.io/normalizer/governance"
        }
    ],

    # E6xx: Audit Errors
    GLNORMErrorCode.E600_AUDIT_WRITE_FAILED: [
        {
            "action": "retry",
            "description": "Retry the operation - audit write may be transient"
        }
    ],
    GLNORMErrorCode.E602_AUDIT_INTEGRITY_VIOLATION: [
        {
            "action": "investigate",
            "description": "Security incident - investigate potential tampering"
        }
    ],

    # E9xx: System Errors
    GLNORMErrorCode.E900_LIMIT_EXCEEDED: [
        {
            "action": "reduce_batch",
            "description": "Reduce batch size or split into multiple requests"
        },
        {
            "action": "retry_later",
            "description": "Wait and retry when rate limit resets"
        }
    ],
    GLNORMErrorCode.E901_TIMEOUT: [
        {
            "action": "retry",
            "description": "Retry the operation"
        },
        {
            "action": "reduce_complexity",
            "description": "Simplify the request if possible"
        }
    ],
    GLNORMErrorCode.E903_SERVICE_UNAVAILABLE: [
        {
            "action": "retry_later",
            "description": "Service temporarily unavailable - retry later"
        }
    ],
}

# GWP version candidates
GWP_CANDIDATES: List[Dict[str, Any]] = [
    {"value": "AR6", "label": "IPCC AR6 (2021) - Recommended", "confidence": None},
    {"value": "AR5", "label": "IPCC AR5 (2014)", "confidence": None},
    {"value": "AR4", "label": "IPCC AR4 (2007)", "confidence": None},
    {"value": "SAR", "label": "IPCC SAR (1995) - Legacy", "confidence": None},
]

# Basis candidates
BASIS_CANDIDATES: List[Dict[str, Any]] = [
    {"value": "HHV", "label": "Higher Heating Value (Gross)", "confidence": None},
    {"value": "LHV", "label": "Lower Heating Value (Net)", "confidence": None},
]


class GLNORMErrorFactory:
    """
    Factory for creating consistent GLNORM error responses.

    This factory provides methods to create well-structured error responses
    with automatic suggestion generation, context enrichment, and audit
    preparation.

    Attributes:
        component: Component name (default: 'gl-normalizer')
        version: Component version
        environment: Environment identifier
        documentation_base_url: Base URL for documentation links

    Example:
        >>> factory = GLNORMErrorFactory(version="1.2.3")
        >>> error = factory.create_error(
        ...     code=GLNORMErrorCode.E100_UNIT_PARSE_FAILED,
        ...     message="Failed to parse unit 'xyz'"
        ... )
    """

    def __init__(
        self,
        component: str = "gl-normalizer",
        version: Optional[str] = None,
        environment: str = "production",
        documentation_base_url: str = "https://docs.greenlang.io/normalizer",
    ):
        """
        Initialize the error factory.

        Args:
            component: Component name for context
            version: Component version for context
            environment: Environment identifier (production, staging, dev)
            documentation_base_url: Base URL for documentation links
        """
        self.component = component
        self.version = version
        self.environment = environment
        self.documentation_base_url = documentation_base_url
        self._custom_suggestion_handlers: Dict[
            GLNORMErrorCode, Callable[..., List[ErrorSuggestion]]
        ] = {}

    def create_error(
        self,
        code: Union[GLNORMErrorCode, str],
        message: str,
        details: Optional[Dict[str, Any]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        context: Optional[ErrorContext] = None,
        request_id: Optional[UUID] = None,
        trace_id: Optional[str] = None,
        operation: Optional[str] = None,
        input_data: Optional[Any] = None,
        include_suggestions: bool = True,
    ) -> GLNORMErrorResponse:
        """
        Create a complete error response.

        Args:
            code: GLNORM error code (enum or string)
            message: Human-readable error message
            details: Error-specific details dictionary
            candidates: Candidate values for ambiguous errors
            context: Pre-built error context (if None, will be created)
            request_id: Request identifier (generated if not provided)
            trace_id: Distributed trace ID
            operation: Operation name
            input_data: Input data for provenance hash
            include_suggestions: Whether to generate suggestions

        Returns:
            GLNORMErrorResponse with full error details

        Example:
            >>> error = factory.create_error(
            ...     code=GLNORMErrorCode.E305_GWP_VERSION_MISSING,
            ...     message="GWP version required for CO2e conversion",
            ...     details={"source_unit": "kg-ch4", "target_unit": "kg-co2e"},
            ...     operation="convert_unit"
            ... )
        """
        # Resolve error code
        if isinstance(code, str):
            resolved_code = get_error_code_by_value(code)
            if resolved_code is None:
                logger.warning(f"Unknown error code: {code}, using E904")
                resolved_code = GLNORMErrorCode.E904_INTERNAL_ERROR
        else:
            resolved_code = code

        # Generate request ID if not provided
        if request_id is None:
            request_id = uuid4()

        # Build context if not provided
        if context is None:
            context = self._build_context(
                request_id=request_id,
                trace_id=trace_id,
                operation=operation or "unknown",
                input_data=input_data,
            )

        # Generate suggestions
        suggestions = None
        if include_suggestions:
            suggestions = self._generate_suggestions(
                code=resolved_code,
                details=details,
                candidates=candidates,
            )

        # Build candidate list
        candidate_models = None
        if candidates:
            candidate_models = [
                ErrorCandidate(**c) if isinstance(c, dict) else c
                for c in candidates
            ]

        # Add standard candidates for specific error codes
        if resolved_code == GLNORMErrorCode.E305_GWP_VERSION_MISSING:
            candidate_models = [ErrorCandidate(**c) for c in GWP_CANDIDATES]
        elif resolved_code == GLNORMErrorCode.E306_BASIS_MISSING:
            candidate_models = [ErrorCandidate(**c) for c in BASIS_CANDIDATES]

        # Build error detail
        error_detail = ErrorDetail(
            code=resolved_code.value,
            message=message,
            category=resolved_code.category_name,
            severity=resolved_code.severity,
            details=details,
            context=context,
            suggestions=suggestions,
            candidates=candidate_models,
            is_recoverable=resolved_code.is_recoverable,
            requires_human_review=resolved_code.requires_human_review,
        )

        return GLNORMErrorResponse(
            success=False,
            error=error_detail,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            http_status=resolved_code.http_status,
        )

    def create_validation_error(
        self,
        validation_errors: List[Dict[str, Any]],
        message: str = "Request validation failed",
        request_id: Optional[UUID] = None,
    ) -> GLNORMValidationErrorResponse:
        """
        Create a validation error response.

        Args:
            validation_errors: List of field validation errors
            message: Summary message
            request_id: Request identifier

        Returns:
            GLNORMValidationErrorResponse with validation details

        Example:
            >>> error = factory.create_validation_error(
            ...     validation_errors=[
            ...         {"field": "source_unit", "message": "Required", "value": None}
            ...     ]
            ... )
        """
        error_items = [
            ValidationErrorItem(**e) if isinstance(e, dict) else e
            for e in validation_errors
        ]

        return GLNORMValidationErrorResponse(
            success=False,
            code="GLNORM-E100",
            message=message,
            validation_errors=error_items,
            request_id=request_id or uuid4(),
            timestamp=datetime.utcnow(),
            http_status=400,
        )

    def create_batch_error(
        self,
        total_items: int,
        succeeded: int,
        errors: List[ErrorDetail],
        request_id: Optional[UUID] = None,
    ) -> GLNORMBatchErrorResponse:
        """
        Create a batch operation error response.

        Args:
            total_items: Total number of items in batch
            succeeded: Number of successfully processed items
            errors: List of individual errors
            request_id: Batch request identifier

        Returns:
            GLNORMBatchErrorResponse with aggregated results

        Example:
            >>> error = factory.create_batch_error(
            ...     total_items=100,
            ...     succeeded=95,
            ...     errors=[error_detail_1, error_detail_2]
            ... )
        """
        failed = total_items - succeeded
        success = succeeded > 0

        return GLNORMBatchErrorResponse(
            success=success,
            total_items=total_items,
            succeeded=succeeded,
            failed=failed,
            errors=errors,
            request_id=request_id or uuid4(),
            timestamp=datetime.utcnow(),
        )

    def create_auditable_error(
        self,
        error: GLNORMErrorResponse,
        correlation_id: Optional[str] = None,
        input_snapshot: Optional[Dict[str, Any]] = None,
        include_stack_trace: bool = False,
    ) -> AuditableError:
        """
        Create an auditable error record for compliance logging.

        Args:
            error: The base error response
            correlation_id: Correlation ID for related operations
            input_snapshot: Sanitized input data snapshot
            include_stack_trace: Include stack trace (non-prod only)

        Returns:
            AuditableError with full audit trail

        Example:
            >>> auditable = factory.create_auditable_error(
            ...     error=error_response,
            ...     correlation_id="txn-2024-001"
            ... )
        """
        stack_trace = None
        if include_stack_trace and self.environment != "production":
            stack_trace = traceback.format_exc()

        # Calculate provenance hash
        provenance_hash = None
        if input_snapshot:
            import json
            snapshot_str = json.dumps(input_snapshot, sort_keys=True)
            provenance_hash = hashlib.sha256(snapshot_str.encode()).hexdigest()

        return AuditableError(
            error=error.error,
            audit_id=uuid4(),
            correlation_id=correlation_id,
            provenance_hash=provenance_hash,
            input_snapshot=input_snapshot,
            stack_trace=stack_trace,
            environment=self.environment,
            audit_timestamp=datetime.utcnow(),
        )

    def register_suggestion_handler(
        self,
        code: GLNORMErrorCode,
        handler: Callable[..., List[ErrorSuggestion]],
    ) -> None:
        """
        Register a custom suggestion handler for an error code.

        Args:
            code: Error code to handle
            handler: Function that generates suggestions

        Example:
            >>> def custom_handler(details, candidates):
            ...     return [ErrorSuggestion(action="custom", description="Do this")]
            >>> factory.register_suggestion_handler(
            ...     GLNORMErrorCode.E100_UNIT_PARSE_FAILED,
            ...     custom_handler
            ... )
        """
        self._custom_suggestion_handlers[code] = handler
        logger.debug(f"Registered custom suggestion handler for {code.value}")

    def _build_context(
        self,
        request_id: UUID,
        trace_id: Optional[str],
        operation: str,
        input_data: Optional[Any],
    ) -> ErrorContext:
        """Build error context from available information."""
        input_hash = None
        if input_data is not None:
            try:
                import json
                if hasattr(input_data, "model_dump_json"):
                    data_str = input_data.model_dump_json()
                elif hasattr(input_data, "json"):
                    data_str = input_data.json()
                else:
                    data_str = json.dumps(input_data, sort_keys=True, default=str)
                input_hash = hashlib.sha256(data_str.encode()).hexdigest()
            except Exception as e:
                logger.warning(f"Failed to compute input hash: {e}")

        return ErrorContext(
            request_id=request_id,
            trace_id=trace_id,
            operation=operation,
            input_hash=input_hash,
            timestamp=datetime.utcnow(),
            component=self.component,
            version=self.version,
        )

    def _generate_suggestions(
        self,
        code: GLNORMErrorCode,
        details: Optional[Dict[str, Any]],
        candidates: Optional[List[Dict[str, Any]]],
    ) -> List[ErrorSuggestion]:
        """Generate suggestions based on error code and context."""
        # Check for custom handler
        if code in self._custom_suggestion_handlers:
            try:
                return self._custom_suggestion_handlers[code](details, candidates)
            except Exception as e:
                logger.warning(f"Custom suggestion handler failed: {e}")

        suggestions = []

        # Get template suggestions
        templates = SUGGESTION_TEMPLATES.get(code, [])
        for template in templates:
            suggestion = ErrorSuggestion(
                action=template.get("action", "review"),
                description=template.get("description", "Review the error"),
                field=template.get("field"),
                example=template.get("example"),
                documentation_url=template.get("documentation_url"),
            )

            # Add candidates if this is a select_candidate action
            if template.get("action") == "select_candidate" and candidates:
                suggestion.candidates = [
                    ErrorCandidate(**c) if isinstance(c, dict) else c
                    for c in candidates
                ]

            suggestions.append(suggestion)

        # Add default suggestion if none found
        if not suggestions:
            suggestions.append(
                ErrorSuggestion(
                    action="review",
                    description="Review the error details and documentation",
                    documentation_url=f"{self.documentation_base_url}/errors/{code.value}",
                )
            )

        return suggestions


# =============================================================================
# Convenience Functions
# =============================================================================

# Global factory instance
_default_factory: Optional[GLNORMErrorFactory] = None


def get_error_factory() -> GLNORMErrorFactory:
    """
    Get the default error factory instance.

    Returns:
        GLNORMErrorFactory singleton instance

    Example:
        >>> factory = get_error_factory()
        >>> error = factory.create_error(...)
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = GLNORMErrorFactory()
    return _default_factory


def set_error_factory(factory: GLNORMErrorFactory) -> None:
    """
    Set the default error factory instance.

    Args:
        factory: Factory instance to use as default

    Example:
        >>> factory = GLNORMErrorFactory(version="2.0.0")
        >>> set_error_factory(factory)
    """
    global _default_factory
    _default_factory = factory


def create_error(
    code: Union[GLNORMErrorCode, str],
    message: str,
    **kwargs: Any,
) -> GLNORMErrorResponse:
    """
    Convenience function to create an error using the default factory.

    Args:
        code: GLNORM error code
        message: Error message
        **kwargs: Additional arguments passed to factory.create_error()

    Returns:
        GLNORMErrorResponse

    Example:
        >>> error = create_error(
        ...     GLNORMErrorCode.E100_UNIT_PARSE_FAILED,
        ...     "Failed to parse unit 'xyz'"
        ... )
    """
    return get_error_factory().create_error(code, message, **kwargs)


def create_validation_error(
    validation_errors: List[Dict[str, Any]],
    **kwargs: Any,
) -> GLNORMValidationErrorResponse:
    """
    Convenience function to create a validation error.

    Args:
        validation_errors: List of validation errors
        **kwargs: Additional arguments

    Returns:
        GLNORMValidationErrorResponse

    Example:
        >>> error = create_validation_error([
        ...     {"field": "unit", "message": "Required"}
        ... ])
    """
    return get_error_factory().create_validation_error(validation_errors, **kwargs)
