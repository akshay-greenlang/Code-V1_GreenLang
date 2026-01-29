"""
Policy configuration loading for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides functions for loading policy configurations from
various sources including files, environment variables, and in-memory
caches. It supports organization-specific policies and policy merging.

Key Design Principles:
    - Deterministic policy resolution for reproducibility
    - Support for hierarchical policy inheritance
    - Caching for performance with explicit cache invalidation
    - Complete audit trail of policy sources

Example:
    >>> from gl_normalizer_core.policy.loader import (
    ...     load_org_policy,
    ...     load_compliance_profile,
    ...     merge_policies,
    ... )
    >>> policy = load_org_policy("org-acme")
    >>> merged = merge_policies(base_policy, override_policy)
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os
import hashlib

import structlog

from gl_normalizer_core.policy.models import (
    Policy,
    PolicyMode,
    PolicyDefaults,
    PolicyOverrides,
    ReferenceConditions,
    ComplianceProfile,
)
from gl_normalizer_core.policy.defaults import (
    get_system_defaults,
    get_profile_defaults,
    get_org_defaults,
    REFERENCE_CONDITIONS_PRESETS,
    DEFAULT_GWP_VERSION,
    DEFAULT_BASIS,
    DEFAULT_TEMPERATURE_REF,
    DEFAULT_PRESSURE_REF,
)
from gl_normalizer_core.errors import ConfigurationError

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration Sources
# =============================================================================

# Environment variable for config path
CONFIG_PATH_ENV = "GLNORM_CONFIG_PATH"

# Environment variable for default policy mode
DEFAULT_MODE_ENV = "GLNORM_DEFAULT_MODE"

# Cache for loaded policies
_policy_cache: Dict[str, Tuple[Policy, datetime]] = {}

# Cache TTL in seconds (default 5 minutes)
_cache_ttl_seconds: int = 300


def set_cache_ttl(ttl_seconds: int) -> None:
    """
    Set the cache TTL for loaded policies.

    Args:
        ttl_seconds: Time-to-live in seconds.

    Example:
        >>> set_cache_ttl(600)  # 10 minutes
    """
    global _cache_ttl_seconds
    _cache_ttl_seconds = ttl_seconds
    logger.info("Policy cache TTL updated", ttl_seconds=ttl_seconds)


def clear_policy_cache(policy_id: Optional[str] = None) -> None:
    """
    Clear the policy cache.

    Args:
        policy_id: Specific policy to clear, or None to clear all.

    Example:
        >>> clear_policy_cache("pol-001")  # Clear one
        >>> clear_policy_cache()  # Clear all
    """
    global _policy_cache
    if policy_id is None:
        _policy_cache = {}
        logger.debug("Cleared all policy cache")
    elif policy_id in _policy_cache:
        del _policy_cache[policy_id]
        logger.debug("Cleared policy from cache", policy_id=policy_id)


def _is_cache_valid(cached_at: datetime) -> bool:
    """Check if a cached entry is still valid."""
    age = (datetime.utcnow() - cached_at).total_seconds()
    return age < _cache_ttl_seconds


def _get_config_path() -> Path:
    """Get the configuration path from environment or default."""
    env_path = os.environ.get(CONFIG_PATH_ENV)
    if env_path:
        return Path(env_path)
    return Path.cwd() / "config" / "policies"


def _get_default_mode() -> PolicyMode:
    """Get the default policy mode from environment."""
    env_mode = os.environ.get(DEFAULT_MODE_ENV, "STRICT")
    try:
        return PolicyMode(env_mode.upper())
    except ValueError:
        logger.warning(
            "Invalid default mode in environment",
            env_value=env_mode,
            using="STRICT",
        )
        return PolicyMode.STRICT


# =============================================================================
# Policy Loading
# =============================================================================

def load_policy_from_file(file_path: Union[str, Path]) -> Policy:
    """
    Load a policy from a JSON file.

    Args:
        file_path: Path to the policy JSON file.

    Returns:
        Loaded Policy object.

    Raises:
        ConfigurationError: If file cannot be loaded or parsed.

    Example:
        >>> policy = load_policy_from_file("/path/to/policy.json")
    """
    path = Path(file_path)

    if not path.exists():
        raise ConfigurationError(
            f"Policy file not found: {path}",
            details={"path": str(path)},
            hint="Verify the policy file path is correct",
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Parse policy from dict
        policy = _parse_policy_dict(data)

        logger.info(
            "Loaded policy from file",
            path=str(path),
            policy_id=policy.policy_id,
            version=policy.version,
        )

        return policy

    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"Invalid JSON in policy file: {path}",
            details={"path": str(path), "error": str(e)},
            hint="Verify the JSON syntax in the policy file",
        ) from e
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load policy file: {path}",
            details={"path": str(path), "error": str(e)},
            hint="Check file permissions and content",
        ) from e


def load_policy_from_dict(data: Dict[str, Any]) -> Policy:
    """
    Load a policy from a dictionary.

    Args:
        data: Policy data as dictionary.

    Returns:
        Policy object.

    Raises:
        ConfigurationError: If data is invalid.

    Example:
        >>> data = {"policy_id": "pol-001", "mode": "STRICT", ...}
        >>> policy = load_policy_from_dict(data)
    """
    try:
        return _parse_policy_dict(data)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to parse policy data: {str(e)}",
            details={"error": str(e)},
            hint="Verify policy data structure matches expected schema",
        ) from e


def _parse_policy_dict(data: Dict[str, Any]) -> Policy:
    """
    Parse a policy from a dictionary with validation.

    Args:
        data: Raw policy data.

    Returns:
        Validated Policy object.
    """
    # Extract and validate mode
    mode_str = data.get("mode", "STRICT")
    try:
        mode = PolicyMode(mode_str.upper())
    except ValueError:
        logger.warning(
            "Invalid mode in policy data",
            mode=mode_str,
            using="STRICT",
        )
        mode = PolicyMode.STRICT

    # Parse compliance profiles
    profiles_data = data.get("compliance_profiles", [])
    profiles = []
    for profile_str in profiles_data:
        try:
            profiles.append(ComplianceProfile(profile_str))
        except ValueError:
            logger.warning(
                "Unknown compliance profile",
                profile=profile_str,
            )

    # Parse defaults
    defaults_data = data.get("defaults", {})
    defaults = _parse_policy_defaults(defaults_data)

    # Parse overrides
    overrides_data = data.get("overrides", {})
    overrides = _parse_policy_overrides(overrides_data)

    # Parse dates
    effective_from = None
    if "effective_from" in data:
        effective_from = datetime.fromisoformat(data["effective_from"])

    effective_until = None
    if "effective_until" in data:
        effective_until = datetime.fromisoformat(data["effective_until"])

    created_at = datetime.utcnow()
    if "created_at" in data:
        created_at = datetime.fromisoformat(data["created_at"])

    return Policy(
        policy_id=data.get("policy_id", f"pol-{hashlib.md5(str(data).encode()).hexdigest()[:8]}"),
        version=data.get("version", "1.0.0"),
        mode=mode,
        compliance_profiles=profiles,
        defaults=defaults,
        overrides=overrides,
        org_id=data.get("org_id"),
        effective_from=effective_from,
        effective_until=effective_until,
        created_at=created_at,
        description=data.get("description"),
    )


def _parse_policy_defaults(data: Dict[str, Any]) -> PolicyDefaults:
    """Parse PolicyDefaults from dictionary."""
    # Parse reference conditions
    ref_data = data.get("reference_conditions")
    if ref_data is None:
        reference_conditions = ReferenceConditions(
            temperature_c=DEFAULT_TEMPERATURE_REF,
            pressure_kpa=DEFAULT_PRESSURE_REF,
        )
    elif isinstance(ref_data, str):
        # Preset name
        reference_conditions = REFERENCE_CONDITIONS_PRESETS.get(
            ref_data,
            ReferenceConditions(
                temperature_c=DEFAULT_TEMPERATURE_REF,
                pressure_kpa=DEFAULT_PRESSURE_REF,
            ),
        )
    else:
        reference_conditions = ReferenceConditions(
            temperature_c=ref_data.get("temperature_c", DEFAULT_TEMPERATURE_REF),
            pressure_kpa=ref_data.get("pressure_kpa", DEFAULT_PRESSURE_REF),
        )

    return PolicyDefaults(
        gwp_version=data.get("gwp_version", DEFAULT_GWP_VERSION),
        basis=data.get("basis", DEFAULT_BASIS),
        reference_conditions=reference_conditions,
        confidence_threshold=data.get("confidence_threshold", 0.8),
        precision_digits=data.get("precision_digits", 6),
        allow_deprecated_factors=data.get("allow_deprecated_factors", False),
        require_unit_validation=data.get("require_unit_validation", True),
    )


def _parse_policy_overrides(data: Dict[str, Any]) -> PolicyOverrides:
    """Parse PolicyOverrides from dictionary."""
    # Parse reference conditions if present
    ref_data = data.get("reference_conditions")
    reference_conditions = None
    if ref_data is not None:
        if isinstance(ref_data, str):
            reference_conditions = REFERENCE_CONDITIONS_PRESETS.get(ref_data)
        else:
            reference_conditions = ReferenceConditions(
                temperature_c=ref_data.get("temperature_c", DEFAULT_TEMPERATURE_REF),
                pressure_kpa=ref_data.get("pressure_kpa", DEFAULT_PRESSURE_REF),
            )

    # Parse unit sets
    allowed_units = None
    if "allowed_units" in data:
        allowed_units = set(data["allowed_units"])

    blocked_units = None
    if "blocked_units" in data:
        blocked_units = set(data["blocked_units"])

    return PolicyOverrides(
        gwp_version=data.get("gwp_version"),
        basis=data.get("basis"),
        reference_conditions=reference_conditions,
        confidence_threshold=data.get("confidence_threshold"),
        allowed_units=allowed_units,
        blocked_units=blocked_units,
        custom_conversions=data.get("custom_conversions"),
    )


def load_org_policy(
    org_id: str,
    config_path: Optional[Path] = None,
    use_cache: bool = True,
) -> Optional[Policy]:
    """
    Load the policy for a specific organization.

    Policies are loaded from:
    {config_path}/orgs/{org_id}/policy.json

    Args:
        org_id: Organization identifier.
        config_path: Base configuration path (defaults to GLNORM_CONFIG_PATH).
        use_cache: Whether to use cached policies.

    Returns:
        Policy if found, None otherwise.

    Example:
        >>> policy = load_org_policy("org-acme")
        >>> if policy:
        ...     print(policy.mode)
    """
    cache_key = f"org:{org_id}"

    # Check cache
    if use_cache and cache_key in _policy_cache:
        cached_policy, cached_at = _policy_cache[cache_key]
        if _is_cache_valid(cached_at):
            logger.debug(
                "Returning cached org policy",
                org_id=org_id,
            )
            return cached_policy

    # Determine config path
    if config_path is None:
        config_path = _get_config_path()

    # Build path to org policy
    policy_file = config_path / "orgs" / org_id / "policy.json"

    if not policy_file.exists():
        logger.debug(
            "No org policy file found",
            org_id=org_id,
            path=str(policy_file),
        )
        return None

    try:
        policy = load_policy_from_file(policy_file)
        policy.org_id = org_id  # Ensure org_id is set

        # Cache the result
        _policy_cache[cache_key] = (policy, datetime.utcnow())

        logger.info(
            "Loaded org policy",
            org_id=org_id,
            policy_id=policy.policy_id,
            mode=policy.mode.value,
        )

        return policy

    except ConfigurationError:
        raise
    except Exception as e:
        logger.error(
            "Failed to load org policy",
            org_id=org_id,
            error=str(e),
            exc_info=True,
        )
        return None


def load_compliance_profile(profile_id: str) -> Optional[ComplianceProfile]:
    """
    Load a compliance profile by ID.

    Args:
        profile_id: Profile identifier (e.g., "GHG_PROTOCOL").

    Returns:
        ComplianceProfile if found, None otherwise.

    Example:
        >>> profile = load_compliance_profile("GHG_PROTOCOL")
        >>> if profile:
        ...     print(profile.display_name)
    """
    try:
        return ComplianceProfile(profile_id.upper())
    except ValueError:
        logger.warning(
            "Unknown compliance profile",
            profile_id=profile_id,
        )
        return None


def load_default_policy(mode: Optional[PolicyMode] = None) -> Policy:
    """
    Create a default policy with system defaults.

    Args:
        mode: Policy mode (defaults to environment setting).

    Returns:
        Default Policy object.

    Example:
        >>> policy = load_default_policy(PolicyMode.LENIENT)
    """
    if mode is None:
        mode = _get_default_mode()

    defaults = get_system_defaults()

    return Policy(
        policy_id="default",
        version="1.0.0",
        mode=mode,
        compliance_profiles=[],
        defaults=defaults,
        overrides=PolicyOverrides(),
        description="System default policy",
    )


# =============================================================================
# Policy Merging
# =============================================================================

def merge_policies(base: Policy, override: Policy) -> Policy:
    """
    Merge two policies with override taking precedence.

    The resulting policy combines:
    - Mode from override
    - Compliance profiles from both (union)
    - Defaults from base with override defaults taking precedence
    - Overrides from both with override taking precedence

    Args:
        base: Base policy.
        override: Override policy (takes precedence).

    Returns:
        Merged Policy object.

    Example:
        >>> base = load_default_policy()
        >>> override = load_org_policy("org-acme")
        >>> merged = merge_policies(base, override)
    """
    # Merge compliance profiles (union)
    all_profiles = set(base.compliance_profiles) | set(override.compliance_profiles)

    # Merge defaults
    merged_defaults = _merge_defaults(base.defaults, override.defaults)

    # Merge overrides
    merged_overrides = _merge_overrides(base.overrides, override.overrides)

    # Create merged policy
    merged = Policy(
        policy_id=f"merged:{base.policy_id}:{override.policy_id}",
        version=override.version,  # Use override version
        mode=override.mode,  # Use override mode
        compliance_profiles=list(all_profiles),
        defaults=merged_defaults,
        overrides=merged_overrides,
        org_id=override.org_id or base.org_id,
        effective_from=override.effective_from or base.effective_from,
        effective_until=override.effective_until or base.effective_until,
        description=f"Merged: {base.description or base.policy_id} + {override.description or override.policy_id}",
    )

    logger.debug(
        "Merged policies",
        base_id=base.policy_id,
        override_id=override.policy_id,
        merged_id=merged.policy_id,
        profile_count=len(all_profiles),
    )

    return merged


def _merge_defaults(base: PolicyDefaults, override: PolicyDefaults) -> PolicyDefaults:
    """Merge two PolicyDefaults with override taking precedence."""
    return PolicyDefaults(
        gwp_version=override.gwp_version if override.gwp_version != DEFAULT_GWP_VERSION else base.gwp_version,
        basis=override.basis if override.basis != DEFAULT_BASIS else base.basis,
        reference_conditions=override.reference_conditions or base.reference_conditions,
        confidence_threshold=override.confidence_threshold or base.confidence_threshold,
        precision_digits=override.precision_digits or base.precision_digits,
        allow_deprecated_factors=override.allow_deprecated_factors,
        require_unit_validation=override.require_unit_validation,
    )


def _merge_overrides(base: PolicyOverrides, override: PolicyOverrides) -> PolicyOverrides:
    """Merge two PolicyOverrides with override taking precedence."""
    # Merge allowed units (intersection if both specified)
    allowed_units = None
    if base.allowed_units is not None and override.allowed_units is not None:
        allowed_units = base.allowed_units & override.allowed_units
    elif override.allowed_units is not None:
        allowed_units = override.allowed_units
    elif base.allowed_units is not None:
        allowed_units = base.allowed_units

    # Merge blocked units (union)
    blocked_units = None
    if base.blocked_units is not None or override.blocked_units is not None:
        blocked_units = (base.blocked_units or set()) | (override.blocked_units or set())

    # Merge custom conversions (override takes precedence)
    custom_conversions = None
    if base.custom_conversions is not None or override.custom_conversions is not None:
        custom_conversions = {
            **(base.custom_conversions or {}),
            **(override.custom_conversions or {}),
        }

    return PolicyOverrides(
        gwp_version=override.gwp_version or base.gwp_version,
        basis=override.basis or base.basis,
        reference_conditions=override.reference_conditions or base.reference_conditions,
        confidence_threshold=override.confidence_threshold or base.confidence_threshold,
        allowed_units=allowed_units,
        blocked_units=blocked_units,
        custom_conversions=custom_conversions,
    )


def merge_policy_chain(policies: List[Policy]) -> Policy:
    """
    Merge a chain of policies in order.

    Each policy overrides the previous ones.

    Args:
        policies: List of policies to merge (first is base).

    Returns:
        Merged Policy object.

    Raises:
        ValueError: If policies list is empty.

    Example:
        >>> chain = [system_policy, org_policy, request_policy]
        >>> final = merge_policy_chain(chain)
    """
    if not policies:
        raise ValueError("Cannot merge empty policy chain")

    if len(policies) == 1:
        return policies[0]

    result = policies[0]
    for policy in policies[1:]:
        result = merge_policies(result, policy)

    return result


# =============================================================================
# Policy Validation
# =============================================================================

def validate_policy(policy: Policy) -> Tuple[bool, List[str]]:
    """
    Validate a policy configuration.

    Args:
        policy: Policy to validate.

    Returns:
        Tuple of (is_valid, list of error messages).

    Example:
        >>> is_valid, errors = validate_policy(policy)
        >>> if not is_valid:
        ...     print(errors)
    """
    errors: List[str] = []

    # Validate policy_id
    if not policy.policy_id or len(policy.policy_id) < 1:
        errors.append("policy_id is required")

    # Validate version format
    if policy.version:
        import re
        if not re.match(r"^\d+\.\d+\.\d+", policy.version):
            errors.append(f"Invalid version format: {policy.version}")

    # Validate effective dates
    if policy.effective_from and policy.effective_until:
        if policy.effective_from >= policy.effective_until:
            errors.append("effective_from must be before effective_until")

    # Validate defaults
    if policy.defaults.gwp_version:
        import re
        if not re.match(r"^(AR[1-6]|SAR|TAR|FAR)$", policy.defaults.gwp_version):
            errors.append(f"Invalid GWP version: {policy.defaults.gwp_version}")

    if policy.defaults.basis:
        if policy.defaults.basis.upper() not in {"LHV", "HHV"}:
            errors.append(f"Invalid basis: {policy.defaults.basis}")

    # Validate confidence threshold
    if not 0.0 <= policy.defaults.confidence_threshold <= 1.0:
        errors.append(
            f"confidence_threshold must be 0-1, got {policy.defaults.confidence_threshold}"
        )

    is_valid = len(errors) == 0

    if not is_valid:
        logger.warning(
            "Policy validation failed",
            policy_id=policy.policy_id,
            error_count=len(errors),
        )

    return is_valid, errors


__all__ = [
    # Configuration
    "set_cache_ttl",
    "clear_policy_cache",
    # Loading
    "load_policy_from_file",
    "load_policy_from_dict",
    "load_org_policy",
    "load_compliance_profile",
    "load_default_policy",
    # Merging
    "merge_policies",
    "merge_policy_chain",
    # Validation
    "validate_policy",
]
