# -*- coding: utf-8 -*-
"""
GreenLang Schema Compiler & Validator - Constants and Limits Module

This module defines all configurable limits, constants, and default configurations
for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

All constants can be overridden via environment variables with the GL_SCHEMA_ prefix.
For example:
    GL_SCHEMA_MAX_PAYLOAD_BYTES=2097152  # Override max payload to 2MB

Key Features:
- Size limits for security (payload, schema, depth, nodes)
- Regex safety limits (length, timeout, complexity)
- Cache configuration (TTL, size, memory)
- Performance targets (latency SLAs)
- Validation profile defaults (strict, standard, permissive)
- Batch processing limits
- CLI defaults

References:
- PRD Section 6.10: Limits and Safety
- PRD Section 6.5: Validation Profiles
- JSON Schema Draft 2020-12

Version: 1.0.0
Date: 2026-01-28

Example:
    >>> from greenlang.schema.constants import MAX_PAYLOAD_BYTES, Limits
    >>> print(f"Max payload size: {MAX_PAYLOAD_BYTES} bytes")
    Max payload size: 1048576 bytes

    >>> limits = load_limits_from_env()
    >>> print(limits.max_payload_bytes)
    1048576
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Final

logger = logging.getLogger(__name__)


# ============================================================================
# ENVIRONMENT VARIABLE PREFIX
# ============================================================================

ENV_PREFIX: Final[str] = "GL_SCHEMA_"


# ============================================================================
# SIZE LIMITS
# ============================================================================
# These limits protect against denial-of-service and memory exhaustion attacks

#: Maximum payload size in bytes (default: 1 MB)
MAX_PAYLOAD_BYTES: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_PAYLOAD_BYTES", 1_048_576)
)

#: Maximum schema size in bytes (default: 2 MB)
MAX_SCHEMA_BYTES: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_SCHEMA_BYTES", 2_097_152)
)

#: Maximum object nesting depth (default: 50 levels)
MAX_OBJECT_DEPTH: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_OBJECT_DEPTH", 50)
)

#: Maximum array items in a single array (default: 10,000)
MAX_ARRAY_ITEMS: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_ARRAY_ITEMS", 10_000)
)

#: Maximum total nodes (objects + arrays + primitives) in a payload
MAX_TOTAL_NODES: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_TOTAL_NODES", 200_000)
)

#: Maximum $ref expansions during schema compilation (default: 10,000)
MAX_REF_EXPANSIONS: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_REF_EXPANSIONS", 10_000)
)

#: Maximum findings (errors + warnings) to collect before stopping
MAX_FINDINGS: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_FINDINGS", 100)
)

#: Maximum string length for any single field (default: 1 MB)
MAX_STRING_LENGTH: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_STRING_LENGTH", 1_048_576)
)

#: Maximum number of properties in an object
MAX_OBJECT_PROPERTIES: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_OBJECT_PROPERTIES", 10_000)
)

# Minimum safe values for limits (absolute floors)
MIN_SAFE_MAX_PAYLOAD_BYTES: Final[int] = 1_024  # 1 KB
MIN_SAFE_MAX_OBJECT_DEPTH: Final[int] = 2
MIN_SAFE_MAX_ARRAY_ITEMS: Final[int] = 1
MIN_SAFE_MAX_TOTAL_NODES: Final[int] = 10
MIN_SAFE_MAX_REF_EXPANSIONS: Final[int] = 1
MIN_SAFE_MAX_FINDINGS: Final[int] = 1

# Maximum safe values for limits (absolute ceilings)
MAX_SAFE_MAX_PAYLOAD_BYTES: Final[int] = 104_857_600  # 100 MB
MAX_SAFE_MAX_SCHEMA_BYTES: Final[int] = 104_857_600  # 100 MB
MAX_SAFE_MAX_OBJECT_DEPTH: Final[int] = 1_000
MAX_SAFE_MAX_ARRAY_ITEMS: Final[int] = 1_000_000
MAX_SAFE_MAX_TOTAL_NODES: Final[int] = 10_000_000
MAX_SAFE_MAX_REF_EXPANSIONS: Final[int] = 100_000
MAX_SAFE_MAX_FINDINGS: Final[int] = 10_000


# ============================================================================
# REGEX LIMITS (ReDoS Protection)
# ============================================================================
# These limits protect against Regular Expression Denial of Service attacks

#: Maximum regex pattern length (default: 1000 characters)
MAX_REGEX_LENGTH: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_REGEX_LENGTH", 1_000)
)

#: Regex evaluation timeout in milliseconds (default: 100ms)
REGEX_TIMEOUT_MS: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}REGEX_TIMEOUT_MS", 100)
)

#: Maximum regex complexity score (0.0-1.0, default: 0.8)
#: Patterns scoring higher are rejected as potentially dangerous
MAX_REGEX_COMPLEXITY_SCORE: Final[float] = float(
    os.environ.get(f"{ENV_PREFIX}MAX_REGEX_COMPLEXITY_SCORE", 0.8)
)

#: Minimum regex complexity score (floor)
MIN_REGEX_COMPLEXITY_SCORE: Final[float] = 0.0

#: Maximum regex complexity score (ceiling)
MAX_REGEX_COMPLEXITY_CEILING: Final[float] = 1.0

#: Patterns known to be dangerous (nested quantifiers, etc.)
DANGEROUS_REGEX_PATTERNS: FrozenSet[str] = frozenset([
    r"(.*)+",      # Nested quantifier with wildcard
    r"(a+)+",      # Classic nested quantifier
    r"(a|a)+",     # Overlapping alternation
    r"(a|aa)+",    # Overlapping alternation variant
])


# ============================================================================
# CACHE SETTINGS
# ============================================================================

#: Schema IR cache TTL in seconds (default: 1 hour)
SCHEMA_CACHE_TTL_SECONDS: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}CACHE_TTL_SECONDS", 3_600)
)

#: Maximum number of cached schema IRs (default: 1000)
SCHEMA_CACHE_MAX_SIZE: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}CACHE_MAX_SIZE", 1_000)
)

#: Maximum memory for IR cache in megabytes (default: 256 MB)
IR_CACHE_MAX_MEMORY_MB: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}IR_CACHE_MAX_MEMORY_MB", 256)
)

#: Cache eviction policy
CACHE_EVICTION_POLICY: Final[str] = os.environ.get(
    f"{ENV_PREFIX}CACHE_EVICTION_POLICY", "lru"
)  # Options: "lru", "lfu", "ttl"

#: Minimum cache TTL (floor)
MIN_CACHE_TTL_SECONDS: Final[int] = 60  # 1 minute

#: Maximum cache TTL (ceiling)
MAX_CACHE_TTL_SECONDS: Final[int] = 86_400  # 24 hours

#: Minimum cache size
MIN_CACHE_SIZE: Final[int] = 10

#: Maximum cache size
MAX_CACHE_SIZE: Final[int] = 100_000


# ============================================================================
# PERFORMANCE TARGETS
# ============================================================================
# SLA targets for validation latency (P95 percentile)

#: P95 latency for small payloads (<50KB) in milliseconds
P95_LATENCY_SMALL_MS: Final[int] = 25

#: P95 latency for medium payloads (<500KB) in milliseconds
P95_LATENCY_MEDIUM_MS: Final[int] = 150

#: P95 latency for large payloads (<1MB) in milliseconds
P95_LATENCY_LARGE_MS: Final[int] = 500

#: Small payload threshold in bytes (default: 50 KB)
SMALL_PAYLOAD_THRESHOLD_BYTES: Final[int] = 51_200

#: Medium payload threshold in bytes (default: 500 KB)
MEDIUM_PAYLOAD_THRESHOLD_BYTES: Final[int] = 512_000

#: Target throughput for batch validation (payloads per second)
TARGET_BATCH_THROUGHPUT_PPS: Final[int] = 1_000


# ============================================================================
# DEPRECATION SETTINGS
# ============================================================================

#: Days before deprecated fields start emitting warnings (default: 90 days)
DEPRECATION_WARNING_DAYS: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}DEPRECATION_WARNING_DAYS", 90)
)

#: Days before deprecated fields become errors (default: 180 days)
DEPRECATION_ERROR_DAYS: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}DEPRECATION_ERROR_DAYS", 180)
)

#: Maximum deprecation warning days
MAX_DEPRECATION_WARNING_DAYS: Final[int] = 365

#: Maximum deprecation error days
MAX_DEPRECATION_ERROR_DAYS: Final[int] = 730  # 2 years


# ============================================================================
# BATCH PROCESSING LIMITS
# ============================================================================

#: Maximum items in a single batch validation request
MAX_BATCH_ITEMS: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_BATCH_ITEMS", 1_000)
)

#: Maximum total bytes for a batch validation request (default: 10 MB)
MAX_BATCH_BYTES: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_BATCH_BYTES", 10_485_760)
)

#: Maximum time allowed for batch processing in seconds
MAX_BATCH_TIME_SECONDS: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_BATCH_TIME_SECONDS", 60)
)

#: Minimum batch items (floor)
MIN_BATCH_ITEMS: Final[int] = 1

#: Maximum batch items (ceiling)
MAX_BATCH_ITEMS_CEILING: Final[int] = 100_000

#: Batch chunk size for parallel processing
BATCH_CHUNK_SIZE: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}BATCH_CHUNK_SIZE", 100)
)


# ============================================================================
# CLI DEFAULTS
# ============================================================================

#: Default maximum errors to display in CLI output
DEFAULT_MAX_ERRORS_DISPLAY: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}DEFAULT_MAX_ERRORS_DISPLAY", 5)
)

#: Default verbosity level (0=summary, 1=all, 2=deep, 3=debug)
DEFAULT_VERBOSITY: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}DEFAULT_VERBOSITY", 0)
)


class ExitCode(int, Enum):
    """CLI exit codes following Unix conventions."""
    SUCCESS = 0              # Validation passed
    VALIDATION_FAILED = 1    # Validation failed (errors found)
    ERROR = 2                # Processing error (schema not found, etc.)
    WARNINGS_ONLY = 0        # Warnings but no errors (default behavior)
    WARNINGS_AS_ERRORS = 1   # Warnings treated as errors (--fail-on-warnings)


#: Default output format
DEFAULT_OUTPUT_FORMAT: Final[str] = os.environ.get(
    f"{ENV_PREFIX}DEFAULT_OUTPUT_FORMAT", "pretty"
)

#: Supported output formats
SUPPORTED_OUTPUT_FORMATS: FrozenSet[str] = frozenset([
    "pretty",   # Colorized human-readable
    "text",     # Plain text
    "table",    # Tabular format
    "json",     # JSON format
    "sarif",    # SARIF for IDE/CI integration
    "yaml",     # YAML format
])

#: Default encoding for file I/O
DEFAULT_ENCODING: Final[str] = "utf-8"


# ============================================================================
# VALIDATION PROFILE ENUMS
# ============================================================================

class ValidationProfile(str, Enum):
    """Validation strictness profiles."""
    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"


class UnknownFieldPolicy(str, Enum):
    """Policy for handling unknown fields."""
    ERROR = "error"      # Fail validation
    WARN = "warn"        # Emit warning but pass
    IGNORE = "ignore"    # Silently ignore


class CoercionPolicy(str, Enum):
    """Policy for type coercion during normalization."""
    OFF = "off"          # No coercion
    SAFE = "safe"        # Safe coercions only (e.g., "42" -> 42)
    AGGRESSIVE = "aggressive"  # More aggressive coercions


class PatchLevel(str, Enum):
    """Safety level for fix suggestions."""
    SAFE = "safe"                  # Only safe patches
    NEEDS_REVIEW = "needs_review"  # Include patches needing review
    UNSAFE = "unsafe"              # Include all patches (use with caution)


# ============================================================================
# VALIDATION PROFILE DEFAULTS
# ============================================================================
# Three preset profiles: strict, standard, permissive

#: Strict profile: Maximum validation, no leniency
STRICT_DEFAULTS: Dict[str, Any] = {
    "normalize": True,
    "emit_patches": True,
    "patch_level": PatchLevel.SAFE.value,
    "max_errors": MAX_FINDINGS,
    "fail_fast": False,
    "unit_system": "SI",
    "unknown_field_policy": UnknownFieldPolicy.ERROR.value,
    "coercion_policy": CoercionPolicy.OFF.value,
    "fail_on_warnings": True,
    "validate_units": True,
    "validate_deprecations": True,
    "validate_naming_convention": True,
    "allow_additional_properties": False,
    "require_all_fields": True,
}

#: Standard profile: Balanced validation (recommended default)
STANDARD_DEFAULTS: Dict[str, Any] = {
    "normalize": True,
    "emit_patches": True,
    "patch_level": PatchLevel.SAFE.value,
    "max_errors": MAX_FINDINGS,
    "fail_fast": False,
    "unit_system": "SI",
    "unknown_field_policy": UnknownFieldPolicy.WARN.value,
    "coercion_policy": CoercionPolicy.SAFE.value,
    "fail_on_warnings": False,
    "validate_units": True,
    "validate_deprecations": True,
    "validate_naming_convention": False,
    "allow_additional_properties": True,
    "require_all_fields": False,
}

#: Permissive profile: Maximum leniency, useful for migration
PERMISSIVE_DEFAULTS: Dict[str, Any] = {
    "normalize": True,
    "emit_patches": True,
    "patch_level": PatchLevel.NEEDS_REVIEW.value,
    "max_errors": MAX_FINDINGS,
    "fail_fast": False,
    "unit_system": "SI",
    "unknown_field_policy": UnknownFieldPolicy.IGNORE.value,
    "coercion_policy": CoercionPolicy.AGGRESSIVE.value,
    "fail_on_warnings": False,
    "validate_units": False,
    "validate_deprecations": False,
    "validate_naming_convention": False,
    "allow_additional_properties": True,
    "require_all_fields": False,
}

#: Profile mapping for quick lookup
PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    ValidationProfile.STRICT.value: STRICT_DEFAULTS,
    ValidationProfile.STANDARD.value: STANDARD_DEFAULTS,
    ValidationProfile.PERMISSIVE.value: PERMISSIVE_DEFAULTS,
}


# ============================================================================
# UNIT SYSTEM CONFIGURATION
# ============================================================================

#: Default unit system for canonicalization
DEFAULT_UNIT_SYSTEM: Final[str] = os.environ.get(
    f"{ENV_PREFIX}DEFAULT_UNIT_SYSTEM", "SI"
)

#: Supported unit systems
SUPPORTED_UNIT_SYSTEMS: FrozenSet[str] = frozenset([
    "SI",       # International System of Units (kWh, kg, m)
    "US",       # US customary units (BTU, lb, ft)
    "METRIC",   # Metric system (subset of SI)
])

#: Canonical units by dimension (SI system)
CANONICAL_UNITS_SI: Dict[str, str] = {
    "energy": "kWh",
    "mass": "kg",
    "length": "m",
    "area": "m2",
    "volume": "m3",
    "temperature": "K",
    "time": "s",
    "power": "W",
    "pressure": "Pa",
    "force": "N",
    "emissions": "kgCO2e",
    "currency": "USD",
}


# ============================================================================
# SCHEMA DIALECT CONFIGURATION
# ============================================================================

#: Default JSON Schema dialect
DEFAULT_SCHEMA_DIALECT: Final[str] = "https://json-schema.org/draft/2020-12/schema"

#: Supported JSON Schema dialects
SUPPORTED_DIALECTS: FrozenSet[str] = frozenset([
    "https://json-schema.org/draft/2020-12/schema",
    "https://json-schema.org/draft/2019-09/schema",
    "https://json-schema.org/draft/07/schema",
    "https://json-schema.org/draft-07/schema#",  # Legacy format
])

#: GreenLang schema URI prefix
GREENLANG_SCHEMA_PREFIX: Final[str] = "gl://"

#: Schema registry base URL (default)
DEFAULT_REGISTRY_URL: Final[str] = os.environ.get(
    f"{ENV_PREFIX}REGISTRY_URL", "https://schemas.greenlang.io"
)


# ============================================================================
# LINTING CONFIGURATION
# ============================================================================

#: Maximum edit distance for typo detection (Levenshtein distance)
MAX_TYPO_EDIT_DISTANCE: Final[int] = int(
    os.environ.get(f"{ENV_PREFIX}MAX_TYPO_EDIT_DISTANCE", 2)
)

#: Minimum key length for typo detection
MIN_KEY_LENGTH_FOR_TYPO_CHECK: Final[int] = 3

#: Naming convention patterns
NAMING_CONVENTIONS: Dict[str, str] = {
    "snake_case": r"^[a-z][a-z0-9_]*$",
    "camelCase": r"^[a-z][a-zA-Z0-9]*$",
    "PascalCase": r"^[A-Z][a-zA-Z0-9]*$",
    "kebab-case": r"^[a-z][a-z0-9-]*$",
}

#: Default naming convention
DEFAULT_NAMING_CONVENTION: Final[str] = "snake_case"

#: Default validation profile
DEFAULT_VALIDATION_PROFILE: Final[str] = os.environ.get(
    f"{ENV_PREFIX}DEFAULT_PROFILE", "standard"
)


# ============================================================================
# LIMITS DATACLASS
# ============================================================================

@dataclass(frozen=True)
class Limits:
    """
    Immutable dataclass holding all configurable limits.

    This class centralizes all limits for easy passing between components
    and ensures thread-safety through immutability.

    Attributes:
        max_payload_bytes: Maximum payload size in bytes
        max_schema_bytes: Maximum schema size in bytes
        max_object_depth: Maximum object nesting depth
        max_array_items: Maximum items in a single array
        max_total_nodes: Maximum total nodes in payload
        max_ref_expansions: Maximum $ref expansions
        max_findings: Maximum validation findings
        max_regex_length: Maximum regex pattern length
        regex_timeout_ms: Regex evaluation timeout
        max_regex_complexity_score: Maximum regex complexity score
        schema_cache_ttl_seconds: Schema cache TTL
        schema_cache_max_size: Maximum cached schemas
        ir_cache_max_memory_mb: Maximum IR cache memory
        max_batch_items: Maximum batch items
        max_batch_bytes: Maximum batch bytes
        max_batch_time_seconds: Maximum batch processing time
        deprecation_warning_days: Days before deprecation warning
        deprecation_error_days: Days before deprecation error

    Example:
        >>> limits = Limits()  # Use defaults
        >>> limits = Limits(max_payload_bytes=2_097_152)  # Override
        >>> limits = load_limits_from_env()  # Load from environment
    """

    # Size limits
    max_payload_bytes: int = MAX_PAYLOAD_BYTES
    max_schema_bytes: int = MAX_SCHEMA_BYTES
    max_object_depth: int = MAX_OBJECT_DEPTH
    max_array_items: int = MAX_ARRAY_ITEMS
    max_total_nodes: int = MAX_TOTAL_NODES
    max_ref_expansions: int = MAX_REF_EXPANSIONS
    max_findings: int = MAX_FINDINGS
    max_string_length: int = MAX_STRING_LENGTH
    max_object_properties: int = MAX_OBJECT_PROPERTIES

    # Regex limits
    max_regex_length: int = MAX_REGEX_LENGTH
    regex_timeout_ms: int = REGEX_TIMEOUT_MS
    max_regex_complexity_score: float = MAX_REGEX_COMPLEXITY_SCORE

    # Cache settings
    schema_cache_ttl_seconds: int = SCHEMA_CACHE_TTL_SECONDS
    schema_cache_max_size: int = SCHEMA_CACHE_MAX_SIZE
    ir_cache_max_memory_mb: int = IR_CACHE_MAX_MEMORY_MB

    # Batch limits
    max_batch_items: int = MAX_BATCH_ITEMS
    max_batch_bytes: int = MAX_BATCH_BYTES
    max_batch_time_seconds: int = MAX_BATCH_TIME_SECONDS

    # Deprecation settings
    deprecation_warning_days: int = DEPRECATION_WARNING_DAYS
    deprecation_error_days: int = DEPRECATION_ERROR_DAYS

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert limits to dictionary for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of all limits

        Example:
            >>> limits = Limits()
            >>> d = limits.to_dict()
            >>> assert "max_payload_bytes" in d
        """
        return {
            "max_payload_bytes": self.max_payload_bytes,
            "max_schema_bytes": self.max_schema_bytes,
            "max_object_depth": self.max_object_depth,
            "max_array_items": self.max_array_items,
            "max_total_nodes": self.max_total_nodes,
            "max_ref_expansions": self.max_ref_expansions,
            "max_findings": self.max_findings,
            "max_string_length": self.max_string_length,
            "max_object_properties": self.max_object_properties,
            "max_regex_length": self.max_regex_length,
            "regex_timeout_ms": self.regex_timeout_ms,
            "max_regex_complexity_score": self.max_regex_complexity_score,
            "schema_cache_ttl_seconds": self.schema_cache_ttl_seconds,
            "schema_cache_max_size": self.schema_cache_max_size,
            "ir_cache_max_memory_mb": self.ir_cache_max_memory_mb,
            "max_batch_items": self.max_batch_items,
            "max_batch_bytes": self.max_batch_bytes,
            "max_batch_time_seconds": self.max_batch_time_seconds,
            "deprecation_warning_days": self.deprecation_warning_days,
            "deprecation_error_days": self.deprecation_error_days,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Limits":
        """
        Create Limits from dictionary.

        Args:
            data: Dictionary with limit values

        Returns:
            Limits: New Limits instance

        Example:
            >>> d = {"max_payload_bytes": 2097152}
            >>> limits = Limits.from_dict(d)
            >>> assert limits.max_payload_bytes == 2097152
        """
        # Filter to only known fields
        known_fields = {
            "max_payload_bytes", "max_schema_bytes", "max_object_depth",
            "max_array_items", "max_total_nodes", "max_ref_expansions",
            "max_findings", "max_string_length", "max_object_properties",
            "max_regex_length", "regex_timeout_ms", "max_regex_complexity_score",
            "schema_cache_ttl_seconds", "schema_cache_max_size",
            "ir_cache_max_memory_mb", "max_batch_items", "max_batch_bytes",
            "max_batch_time_seconds", "deprecation_warning_days",
            "deprecation_error_days",
        }
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


# ============================================================================
# VALIDATION RESULT FOR LIMITS
# ============================================================================

@dataclass
class LimitsValidationResult:
    """
    Result of validating limits configuration.

    Attributes:
        valid: Whether all limits are within safe ranges
        errors: List of validation errors
        warnings: List of validation warnings
        adjusted_limits: Limits adjusted to safe ranges (if any)
    """

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adjusted_limits: Optional[Limits] = None


# ============================================================================
# ENVIRONMENT VARIABLE LOADING HELPERS
# ============================================================================

def _get_env_int(name: str, default: int) -> int:
    """
    Get integer value from environment variable.

    Args:
        name: Environment variable name (without prefix)
        default: Default value if not set

    Returns:
        Integer value from environment or default
    """
    env_name = f"{ENV_PREFIX}{name}"
    value = os.environ.get(env_name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(
            f"Invalid integer value for {env_name}='{value}', using default={default}"
        )
        return default


def _get_env_float(name: str, default: float) -> float:
    """
    Get float value from environment variable.

    Args:
        name: Environment variable name (without prefix)
        default: Default value if not set

    Returns:
        Float value from environment or default
    """
    env_name = f"{ENV_PREFIX}{name}"
    value = os.environ.get(env_name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(
            f"Invalid float value for {env_name}='{value}', using default={default}"
        )
        return default


def _get_env_str(name: str, default: str) -> str:
    """
    Get string value from environment variable.

    Args:
        name: Environment variable name (without prefix)
        default: Default value if not set

    Returns:
        String value from environment or default
    """
    env_name = f"{ENV_PREFIX}{name}"
    return os.environ.get(env_name, default)


def _get_env_bool(name: str, default: bool) -> bool:
    """
    Get boolean value from environment variable.

    Args:
        name: Environment variable name (without prefix)
        default: Default value if not set

    Returns:
        Boolean value from environment or default
    """
    env_name = f"{ENV_PREFIX}{name}"
    value = os.environ.get(env_name)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def load_limits_from_env() -> Limits:
    """
    Load limits configuration from environment variables.

    Environment variables use the GL_SCHEMA_ prefix. For example:
    - GL_SCHEMA_MAX_PAYLOAD_BYTES=2097152
    - GL_SCHEMA_MAX_OBJECT_DEPTH=100
    - GL_SCHEMA_REGEX_TIMEOUT_MS=200

    Returns:
        Limits: Configured limits from environment or defaults

    Example:
        >>> import os
        >>> os.environ["GL_SCHEMA_MAX_PAYLOAD_BYTES"] = "2097152"
        >>> limits = load_limits_from_env()
        >>> assert limits.max_payload_bytes == 2097152
    """
    # Use the module-level constants which already read from env
    return Limits(
        # Size limits
        max_payload_bytes=_get_env_int("MAX_PAYLOAD_BYTES", 1_048_576),
        max_schema_bytes=_get_env_int("MAX_SCHEMA_BYTES", 2_097_152),
        max_object_depth=_get_env_int("MAX_OBJECT_DEPTH", 50),
        max_array_items=_get_env_int("MAX_ARRAY_ITEMS", 10_000),
        max_total_nodes=_get_env_int("MAX_TOTAL_NODES", 200_000),
        max_ref_expansions=_get_env_int("MAX_REF_EXPANSIONS", 10_000),
        max_findings=_get_env_int("MAX_FINDINGS", 100),
        max_string_length=_get_env_int("MAX_STRING_LENGTH", 1_048_576),
        max_object_properties=_get_env_int("MAX_OBJECT_PROPERTIES", 10_000),

        # Regex limits
        max_regex_length=_get_env_int("MAX_REGEX_LENGTH", 1_000),
        regex_timeout_ms=_get_env_int("REGEX_TIMEOUT_MS", 100),
        max_regex_complexity_score=_get_env_float(
            "MAX_REGEX_COMPLEXITY_SCORE", 0.8
        ),

        # Cache settings
        schema_cache_ttl_seconds=_get_env_int("CACHE_TTL_SECONDS", 3_600),
        schema_cache_max_size=_get_env_int("CACHE_MAX_SIZE", 1_000),
        ir_cache_max_memory_mb=_get_env_int("IR_CACHE_MAX_MEMORY_MB", 256),

        # Batch limits
        max_batch_items=_get_env_int("MAX_BATCH_ITEMS", 1_000),
        max_batch_bytes=_get_env_int("MAX_BATCH_BYTES", 10_485_760),
        max_batch_time_seconds=_get_env_int("MAX_BATCH_TIME_SECONDS", 60),

        # Deprecation settings
        deprecation_warning_days=_get_env_int("DEPRECATION_WARNING_DAYS", 90),
        deprecation_error_days=_get_env_int("DEPRECATION_ERROR_DAYS", 180),
    )


# ============================================================================
# LIMITS VALIDATION
# ============================================================================

def validate_limits(limits: Limits) -> LimitsValidationResult:
    """
    Validate that limits are within safe ranges.

    This function checks all limits against their minimum and maximum safe
    values and returns a validation result with errors and warnings.

    Args:
        limits: Limits instance to validate

    Returns:
        LimitsValidationResult: Validation result with errors and warnings

    Example:
        >>> limits = Limits(max_payload_bytes=50)  # Too small
        >>> result = validate_limits(limits)
        >>> assert not result.valid
        >>> assert len(result.errors) > 0
    """
    errors: List[str] = []
    warnings: List[str] = []
    adjustments: Dict[str, Any] = {}

    # Validate max_payload_bytes
    if limits.max_payload_bytes < MIN_SAFE_MAX_PAYLOAD_BYTES:
        errors.append(
            f"max_payload_bytes ({limits.max_payload_bytes}) is below minimum "
            f"safe value ({MIN_SAFE_MAX_PAYLOAD_BYTES})"
        )
        adjustments["max_payload_bytes"] = MIN_SAFE_MAX_PAYLOAD_BYTES
    elif limits.max_payload_bytes > MAX_SAFE_MAX_PAYLOAD_BYTES:
        warnings.append(
            f"max_payload_bytes ({limits.max_payload_bytes}) exceeds recommended "
            f"maximum ({MAX_SAFE_MAX_PAYLOAD_BYTES})"
        )

    # Validate max_schema_bytes
    if limits.max_schema_bytes < MIN_SAFE_MAX_PAYLOAD_BYTES:
        errors.append(
            f"max_schema_bytes ({limits.max_schema_bytes}) is below minimum "
            f"safe value ({MIN_SAFE_MAX_PAYLOAD_BYTES})"
        )
        adjustments["max_schema_bytes"] = MIN_SAFE_MAX_PAYLOAD_BYTES
    elif limits.max_schema_bytes > MAX_SAFE_MAX_SCHEMA_BYTES:
        warnings.append(
            f"max_schema_bytes ({limits.max_schema_bytes}) exceeds recommended "
            f"maximum ({MAX_SAFE_MAX_SCHEMA_BYTES})"
        )

    # Validate max_object_depth
    if limits.max_object_depth < MIN_SAFE_MAX_OBJECT_DEPTH:
        errors.append(
            f"max_object_depth ({limits.max_object_depth}) is below minimum "
            f"safe value ({MIN_SAFE_MAX_OBJECT_DEPTH})"
        )
        adjustments["max_object_depth"] = MIN_SAFE_MAX_OBJECT_DEPTH
    elif limits.max_object_depth > MAX_SAFE_MAX_OBJECT_DEPTH:
        warnings.append(
            f"max_object_depth ({limits.max_object_depth}) exceeds recommended "
            f"maximum ({MAX_SAFE_MAX_OBJECT_DEPTH})"
        )

    # Validate max_array_items
    if limits.max_array_items < MIN_SAFE_MAX_ARRAY_ITEMS:
        errors.append(
            f"max_array_items ({limits.max_array_items}) is below minimum "
            f"safe value ({MIN_SAFE_MAX_ARRAY_ITEMS})"
        )
        adjustments["max_array_items"] = MIN_SAFE_MAX_ARRAY_ITEMS
    elif limits.max_array_items > MAX_SAFE_MAX_ARRAY_ITEMS:
        warnings.append(
            f"max_array_items ({limits.max_array_items}) exceeds recommended "
            f"maximum ({MAX_SAFE_MAX_ARRAY_ITEMS})"
        )

    # Validate max_total_nodes
    if limits.max_total_nodes < MIN_SAFE_MAX_TOTAL_NODES:
        errors.append(
            f"max_total_nodes ({limits.max_total_nodes}) is below minimum "
            f"safe value ({MIN_SAFE_MAX_TOTAL_NODES})"
        )
        adjustments["max_total_nodes"] = MIN_SAFE_MAX_TOTAL_NODES
    elif limits.max_total_nodes > MAX_SAFE_MAX_TOTAL_NODES:
        warnings.append(
            f"max_total_nodes ({limits.max_total_nodes}) exceeds recommended "
            f"maximum ({MAX_SAFE_MAX_TOTAL_NODES})"
        )

    # Validate max_ref_expansions
    if limits.max_ref_expansions < MIN_SAFE_MAX_REF_EXPANSIONS:
        errors.append(
            f"max_ref_expansions ({limits.max_ref_expansions}) is below minimum "
            f"safe value ({MIN_SAFE_MAX_REF_EXPANSIONS})"
        )
        adjustments["max_ref_expansions"] = MIN_SAFE_MAX_REF_EXPANSIONS
    elif limits.max_ref_expansions > MAX_SAFE_MAX_REF_EXPANSIONS:
        warnings.append(
            f"max_ref_expansions ({limits.max_ref_expansions}) exceeds recommended "
            f"maximum ({MAX_SAFE_MAX_REF_EXPANSIONS})"
        )

    # Validate max_findings
    if limits.max_findings < MIN_SAFE_MAX_FINDINGS:
        errors.append(
            f"max_findings ({limits.max_findings}) is below minimum "
            f"safe value ({MIN_SAFE_MAX_FINDINGS})"
        )
        adjustments["max_findings"] = MIN_SAFE_MAX_FINDINGS
    elif limits.max_findings > MAX_SAFE_MAX_FINDINGS:
        warnings.append(
            f"max_findings ({limits.max_findings}) exceeds recommended "
            f"maximum ({MAX_SAFE_MAX_FINDINGS})"
        )

    # Validate max_regex_complexity_score
    if limits.max_regex_complexity_score < MIN_REGEX_COMPLEXITY_SCORE:
        errors.append(
            f"max_regex_complexity_score ({limits.max_regex_complexity_score}) "
            f"is below minimum ({MIN_REGEX_COMPLEXITY_SCORE})"
        )
        adjustments["max_regex_complexity_score"] = MIN_REGEX_COMPLEXITY_SCORE
    elif limits.max_regex_complexity_score > MAX_REGEX_COMPLEXITY_CEILING:
        errors.append(
            f"max_regex_complexity_score ({limits.max_regex_complexity_score}) "
            f"exceeds maximum ({MAX_REGEX_COMPLEXITY_CEILING})"
        )
        adjustments["max_regex_complexity_score"] = MAX_REGEX_COMPLEXITY_CEILING

    # Validate cache TTL
    if limits.schema_cache_ttl_seconds < MIN_CACHE_TTL_SECONDS:
        warnings.append(
            f"schema_cache_ttl_seconds ({limits.schema_cache_ttl_seconds}) "
            f"is below recommended minimum ({MIN_CACHE_TTL_SECONDS})"
        )
    elif limits.schema_cache_ttl_seconds > MAX_CACHE_TTL_SECONDS:
        warnings.append(
            f"schema_cache_ttl_seconds ({limits.schema_cache_ttl_seconds}) "
            f"exceeds recommended maximum ({MAX_CACHE_TTL_SECONDS})"
        )

    # Validate cache size
    if limits.schema_cache_max_size < MIN_CACHE_SIZE:
        warnings.append(
            f"schema_cache_max_size ({limits.schema_cache_max_size}) "
            f"is below recommended minimum ({MIN_CACHE_SIZE})"
        )
    elif limits.schema_cache_max_size > MAX_CACHE_SIZE:
        warnings.append(
            f"schema_cache_max_size ({limits.schema_cache_max_size}) "
            f"exceeds recommended maximum ({MAX_CACHE_SIZE})"
        )

    # Validate batch limits
    if limits.max_batch_items < MIN_BATCH_ITEMS:
        errors.append(
            f"max_batch_items ({limits.max_batch_items}) is below minimum "
            f"({MIN_BATCH_ITEMS})"
        )
        adjustments["max_batch_items"] = MIN_BATCH_ITEMS
    elif limits.max_batch_items > MAX_BATCH_ITEMS_CEILING:
        warnings.append(
            f"max_batch_items ({limits.max_batch_items}) exceeds recommended "
            f"maximum ({MAX_BATCH_ITEMS_CEILING})"
        )

    # Validate deprecation days
    if limits.deprecation_warning_days > limits.deprecation_error_days:
        errors.append(
            f"deprecation_warning_days ({limits.deprecation_warning_days}) "
            f"should not exceed deprecation_error_days ({limits.deprecation_error_days})"
        )
        adjustments["deprecation_warning_days"] = limits.deprecation_error_days

    # Create adjusted limits if needed
    adjusted_limits = None
    if adjustments:
        adjusted_dict = limits.to_dict()
        adjusted_dict.update(adjustments)
        adjusted_limits = Limits(**adjusted_dict)

    return LimitsValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        adjusted_limits=adjusted_limits,
    )


def get_safe_limits(limits: Optional[Limits] = None) -> Limits:
    """
    Get limits with unsafe values adjusted to safe ranges.

    If the provided limits have values outside safe ranges, they are
    adjusted to the nearest safe value.

    Args:
        limits: Optional limits to adjust. If None, returns defaults.

    Returns:
        Limits: Safe limits instance

    Example:
        >>> unsafe = Limits(max_payload_bytes=50)  # Too small
        >>> safe = get_safe_limits(unsafe)
        >>> assert safe.max_payload_bytes >= MIN_SAFE_MAX_PAYLOAD_BYTES
    """
    if limits is None:
        return Limits()

    result = validate_limits(limits)
    if result.adjusted_limits is not None:
        return result.adjusted_limits
    return limits


# ============================================================================
# PROFILE CONFIGURATION HELPERS
# ============================================================================

def get_profile_defaults(profile: str) -> Dict[str, Any]:
    """
    Get default configuration for a validation profile.

    Args:
        profile: Profile name ("strict", "standard", or "permissive")

    Returns:
        Dict containing profile default settings

    Raises:
        ValueError: If profile name is not recognized

    Example:
        >>> defaults = get_profile_defaults("strict")
        >>> assert defaults["unknown_field_policy"] == "error"
    """
    profile_lower = profile.lower()
    if profile_lower not in PROFILE_DEFAULTS:
        valid_profiles = ", ".join(PROFILE_DEFAULTS.keys())
        raise ValueError(
            f"Unknown validation profile '{profile}'. "
            f"Valid profiles: {valid_profiles}"
        )
    return PROFILE_DEFAULTS[profile_lower].copy()


def merge_options_with_profile(
    options: Dict[str, Any],
    profile: str = "standard"
) -> Dict[str, Any]:
    """
    Merge user options with profile defaults.

    User-provided options take precedence over profile defaults.

    Args:
        options: User-provided options
        profile: Base profile for defaults

    Returns:
        Dict containing merged options

    Example:
        >>> options = {"fail_fast": True}
        >>> merged = merge_options_with_profile(options, "standard")
        >>> assert merged["fail_fast"] is True  # User override
        >>> assert "normalize" in merged  # From profile default
    """
    defaults = get_profile_defaults(profile)
    defaults.update(options)
    return defaults


# ============================================================================
# DEFAULT LIMITS INSTANCE
# ============================================================================

#: Global default limits instance (can be overridden at runtime)
DEFAULT_LIMITS: Limits = Limits()


# ============================================================================
# MODULE METADATA
# ============================================================================

__all__ = [
    # Environment prefix
    "ENV_PREFIX",

    # Size limits
    "MAX_PAYLOAD_BYTES",
    "MAX_SCHEMA_BYTES",
    "MAX_OBJECT_DEPTH",
    "MAX_ARRAY_ITEMS",
    "MAX_TOTAL_NODES",
    "MAX_REF_EXPANSIONS",
    "MAX_FINDINGS",
    "MAX_STRING_LENGTH",
    "MAX_OBJECT_PROPERTIES",

    # Safe ranges
    "MIN_SAFE_MAX_PAYLOAD_BYTES",
    "MIN_SAFE_MAX_OBJECT_DEPTH",
    "MIN_SAFE_MAX_ARRAY_ITEMS",
    "MIN_SAFE_MAX_TOTAL_NODES",
    "MIN_SAFE_MAX_REF_EXPANSIONS",
    "MIN_SAFE_MAX_FINDINGS",
    "MAX_SAFE_MAX_PAYLOAD_BYTES",
    "MAX_SAFE_MAX_SCHEMA_BYTES",
    "MAX_SAFE_MAX_OBJECT_DEPTH",
    "MAX_SAFE_MAX_ARRAY_ITEMS",
    "MAX_SAFE_MAX_TOTAL_NODES",
    "MAX_SAFE_MAX_REF_EXPANSIONS",
    "MAX_SAFE_MAX_FINDINGS",

    # Regex limits
    "MAX_REGEX_LENGTH",
    "REGEX_TIMEOUT_MS",
    "MAX_REGEX_COMPLEXITY_SCORE",
    "MIN_REGEX_COMPLEXITY_SCORE",
    "MAX_REGEX_COMPLEXITY_CEILING",
    "DANGEROUS_REGEX_PATTERNS",

    # Cache settings
    "SCHEMA_CACHE_TTL_SECONDS",
    "SCHEMA_CACHE_MAX_SIZE",
    "IR_CACHE_MAX_MEMORY_MB",
    "CACHE_EVICTION_POLICY",
    "MIN_CACHE_TTL_SECONDS",
    "MAX_CACHE_TTL_SECONDS",
    "MIN_CACHE_SIZE",
    "MAX_CACHE_SIZE",

    # Performance targets
    "P95_LATENCY_SMALL_MS",
    "P95_LATENCY_MEDIUM_MS",
    "P95_LATENCY_LARGE_MS",
    "SMALL_PAYLOAD_THRESHOLD_BYTES",
    "MEDIUM_PAYLOAD_THRESHOLD_BYTES",
    "TARGET_BATCH_THROUGHPUT_PPS",

    # Deprecation settings
    "DEPRECATION_WARNING_DAYS",
    "DEPRECATION_ERROR_DAYS",
    "MAX_DEPRECATION_WARNING_DAYS",
    "MAX_DEPRECATION_ERROR_DAYS",

    # Batch limits
    "MAX_BATCH_ITEMS",
    "MAX_BATCH_BYTES",
    "MAX_BATCH_TIME_SECONDS",
    "MIN_BATCH_ITEMS",
    "MAX_BATCH_ITEMS_CEILING",
    "BATCH_CHUNK_SIZE",

    # CLI defaults
    "DEFAULT_MAX_ERRORS_DISPLAY",
    "DEFAULT_VERBOSITY",
    "DEFAULT_OUTPUT_FORMAT",
    "SUPPORTED_OUTPUT_FORMATS",
    "DEFAULT_ENCODING",
    "ExitCode",

    # Validation profiles
    "ValidationProfile",
    "UnknownFieldPolicy",
    "CoercionPolicy",
    "PatchLevel",
    "STRICT_DEFAULTS",
    "STANDARD_DEFAULTS",
    "PERMISSIVE_DEFAULTS",
    "PROFILE_DEFAULTS",
    "DEFAULT_VALIDATION_PROFILE",

    # Unit system
    "DEFAULT_UNIT_SYSTEM",
    "SUPPORTED_UNIT_SYSTEMS",
    "CANONICAL_UNITS_SI",

    # Schema dialect
    "DEFAULT_SCHEMA_DIALECT",
    "SUPPORTED_DIALECTS",
    "GREENLANG_SCHEMA_PREFIX",
    "DEFAULT_REGISTRY_URL",

    # Linting
    "MAX_TYPO_EDIT_DISTANCE",
    "MIN_KEY_LENGTH_FOR_TYPO_CHECK",
    "NAMING_CONVENTIONS",
    "DEFAULT_NAMING_CONVENTION",

    # Classes
    "Limits",
    "LimitsValidationResult",

    # Functions
    "load_limits_from_env",
    "validate_limits",
    "get_safe_limits",
    "get_profile_defaults",
    "merge_options_with_profile",

    # Default instance
    "DEFAULT_LIMITS",
]
