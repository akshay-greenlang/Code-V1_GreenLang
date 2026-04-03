# -*- coding: utf-8 -*-
"""
Error Taxonomy for Tool Runtime (INTL-103)

Machine-readable error codes with human remediation hints.
All errors carry: .code, .message, .hint, .path

CTO Specification: Structured errors enable auto-repair and clear debugging.

Migration (2026-04-02):
- All classes now inherit from the centralized greenlang.exceptions hierarchy
- isinstance(error, GreenLangException) is now True for all GL runtime errors
- Full backward compatibility: same class names, same public APIs, same attributes
"""

from __future__ import annotations
from typing import Optional

from greenlang.utilities.exceptions.base import GreenLangException
from greenlang.utilities.exceptions.agent import (
    ValidationError as _CentralValidationError,
)
from greenlang.utilities.exceptions.security import (
    SecurityException as _CentralSecurityException,
)
from greenlang.utilities.exceptions.data import (
    DataException as _CentralDataException,
)
from greenlang.utilities.exceptions.compliance import (
    ProvenanceError as _CentralProvenanceError,
)


class GLValidationError(_CentralValidationError):
    """
    JSON Schema validation failures

    Raised when tool arguments or results don't match declared schemas.

    Inherits from greenlang.utilities.exceptions.agent.ValidationError
    so that isinstance(error, GreenLangException) returns True.
    """

    # Error codes
    ARGS_SCHEMA = "ARGS_SCHEMA"
    RESULT_SCHEMA = "RESULT_SCHEMA"
    UNIT_UNKNOWN = "UNIT_UNKNOWN"

    def __init__(
        self,
        code: str,
        message: str,
        hint: Optional[str] = None,
        path: Optional[str] = None,
    ):
        self.code = code
        self.hint = hint or self._default_hint(code)
        self.path = path

        # Build context for the centralized parent
        context = {
            "code": code,
            "hint": self.hint,
        }
        if path:
            context["path"] = path

        # Initialize the centralized parent
        _CentralValidationError.__init__(
            self,
            message=f"[{code}] {message}",
            context=context,
        )

        # Restore the original message attribute for backward compat
        self.message = message

    @staticmethod
    def _default_hint(code: str) -> str:
        hints = {
            "ARGS_SCHEMA": "Check tool arguments against the declared args_schema. Ensure all required fields are present and types match.",
            "RESULT_SCHEMA": "Tool output must match result_schema. Ensure all numerics are wrapped in Quantity {value, unit}. No raw numbers allowed.",
            "UNIT_UNKNOWN": "Unit not in allowlist. Check UnitRegistry.allowlist for valid units or add custom unit definition.",
        }
        return hints.get(code, "No hint available")

    def to_dict(self) -> dict:
        """Serialize for logging/telemetry"""
        return {
            "error_type": "GLValidationError",
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "path": self.path,
        }


class GLRuntimeError(GreenLangException):
    """
    Runtime enforcement violations

    Raised when "no naked numbers" rule is violated.

    Inherits from greenlang.utilities.exceptions.base.GreenLangException
    so that isinstance(error, GreenLangException) returns True.
    """

    # Error codes
    NO_NAKED_NUMBERS = "NO_NAKED_NUMBERS"

    def __init__(
        self,
        code: str,
        message: str,
        hint: Optional[str] = None,
        path: Optional[str] = None,
        context: Optional[str] = None,
    ):
        self.code = code
        self.hint = hint or self._default_hint(code)
        self.path = path
        # Note: 'context' here is a string (existing API), distinct from
        # GreenLangException's dict context. We store it separately.
        self.runtime_context = context

        # Build dict context for the centralized parent
        central_context = {
            "code": code,
            "hint": self.hint,
        }
        if path:
            central_context["path"] = path
        if context:
            central_context["runtime_context"] = context

        # Initialize the centralized parent
        GreenLangException.__init__(
            self,
            message=f"[{code}] {message}",
            context=central_context,
        )

        # Restore the original message attribute for backward compat
        self.message = message
        # Restore the string context attribute for backward compat
        self.context = context

    @staticmethod
    def _default_hint(code: str) -> str:
        hints = {
            "NO_NAKED_NUMBERS": (
                "Numeric detected in final message without {{claim:i}} macro. "
                "You must either:\n"
                "1. Call a tool to get the numeric value, OR\n"
                "2. Reference it via {{claim:i}} macro backed by claims[]"
            )
        }
        return hints.get(code, "No hint available")

    def to_dict(self) -> dict:
        """Serialize for logging/telemetry"""
        return {
            "error_type": "GLRuntimeError",
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "path": self.path,
            "context": self.runtime_context,
        }


class GLSecurityError(_CentralSecurityException):
    """
    Security violations

    Raised when tools attempt unauthorized operations.

    Inherits from greenlang.utilities.exceptions.security.SecurityException
    so that isinstance(error, GreenLangException) returns True.
    """

    # Error codes
    EGRESS_BLOCKED = "EGRESS_BLOCKED"
    COMPUTE_IO_FORBIDDEN = "COMPUTE_IO_FORBIDDEN"

    def __init__(
        self,
        code: str,
        message: str,
        hint: Optional[str] = None,
        path: Optional[str] = None,
    ):
        self.code = code
        self.hint = hint or self._default_hint(code)
        self.path = path

        # Build context for the centralized parent
        context = {
            "code": code,
            "hint": self.hint,
        }
        if path:
            context["path"] = path

        # Initialize the centralized parent
        _CentralSecurityException.__init__(
            self,
            message=f"[{code}] {message}",
            context=context,
        )

        # Restore the original message attribute for backward compat
        self.message = message

    @staticmethod
    def _default_hint(code: str) -> str:
        hints = {
            "EGRESS_BLOCKED": (
                "Tool requires Live mode but runtime is in Replay mode. "
                "Either switch to Live mode or provide a snapshot for replay."
            ),
            "COMPUTE_IO_FORBIDDEN": (
                "Compute tools cannot perform file/network I/O. "
                "Use dedicated connector tools for external data access."
            ),
        }
        return hints.get(code, "No hint available")

    def to_dict(self) -> dict:
        """Serialize for logging/telemetry"""
        return {
            "error_type": "GLSecurityError",
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "path": self.path,
        }


class GLDataError(_CentralDataException):
    """
    Data resolution failures

    Raised when claim paths can't be resolved or quantities don't match.

    Inherits from greenlang.utilities.exceptions.data.DataException
    so that isinstance(error, GreenLangException) returns True.
    """

    # Error codes
    PATH_RESOLUTION = "PATH_RESOLUTION"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"

    def __init__(
        self,
        code: str,
        message: str,
        hint: Optional[str] = None,
        path: Optional[str] = None,
    ):
        self.code = code
        self.hint = hint or self._default_hint(code)
        self.path = path

        # Build context for the centralized parent
        context = {
            "code": code,
            "hint": self.hint,
        }
        if path:
            context["path"] = path

        # Initialize the centralized parent
        _CentralDataException.__init__(
            self,
            message=f"[{code}] {message}",
            context=context,
        )

        # Restore the original message attribute for backward compat
        self.message = message

    @staticmethod
    def _default_hint(code: str) -> str:
        hints = {
            "PATH_RESOLUTION": (
                "JSONPath failed to resolve in tool output. "
                "Ensure path syntax is correct ($.field) and points to a Quantity."
            ),
            "QUANTITY_MISMATCH": (
                "Claimed Quantity doesn't match resolved value from tool output. "
                "Ensure claim exactly matches what the tool returned (after unit normalization)."
            ),
        }
        return hints.get(code, "No hint available")

    def to_dict(self) -> dict:
        """Serialize for logging/telemetry"""
        return {
            "error_type": "GLDataError",
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "path": self.path,
        }


class GLProvenanceError(_CentralProvenanceError):
    """
    Provenance tracking failures

    Raised when required provenance metadata is missing.

    Inherits from greenlang.utilities.exceptions.compliance.ProvenanceError
    so that isinstance(error, GreenLangException) returns True.
    """

    # Error codes
    MISSING_EF_CID = "MISSING_EF_CID"
    MISSING_SEED = "MISSING_SEED"
    MISSING_TOOL_CALL = "MISSING_TOOL_CALL"

    def __init__(
        self,
        code: str,
        message: str,
        hint: Optional[str] = None,
        path: Optional[str] = None,
    ):
        self.code = code
        self.hint = hint or self._default_hint(code)
        self.path = path

        # Build context for the centralized parent
        context = {
            "code": code,
            "hint": self.hint,
        }
        if path:
            context["path"] = path

        # Initialize the centralized parent
        _CentralProvenanceError.__init__(
            self,
            message=f"[{code}] {message}",
            context=context,
        )

        # Restore the original message attribute for backward compat
        self.message = message

    @staticmethod
    def _default_hint(code: str) -> str:
        hints = {
            "MISSING_EF_CID": "Emission factor must include content ID (EF_CID) for provenance.",
            "MISSING_SEED": "Computation must be seeded for reproducibility.",
            "MISSING_TOOL_CALL": "Claim references non-existent tool call ID. Ensure source_call_id matches a prior tool execution.",
        }
        return hints.get(code, "No hint available")

    def to_dict(self) -> dict:
        """Serialize for logging/telemetry"""
        return {
            "error_type": "GLProvenanceError",
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "path": self.path,
        }


# Convenience function for error serialization
def serialize_error(error: Exception) -> dict:
    """
    Serialize any GL error for logging/telemetry

    Returns dict with error_type, code, message, hint, path
    """
    if hasattr(error, "to_dict"):
        return error.to_dict()
    else:
        # Fallback for non-GL errors
        return {"error_type": error.__class__.__name__, "message": str(error)}
