"""
Feature Flag Service

This module provides a comprehensive feature flag system for controlled rollouts,
A/B testing, and gradual feature enablement across the GL-Agent-Factory platform.

Features:
- Boolean, percentage, and user-targeted flags
- Environment-specific configurations
- Redis-backed flag storage (optional)
- Real-time flag updates
- Audit logging for flag changes
- Metrics collection for flag usage

Usage:
    from services.feature_flags import FeatureFlagService, get_feature_flags

    flags = get_feature_flags()

    if flags.is_enabled("new_calculation_engine"):
        # Use new engine
    else:
        # Use legacy engine

    # Percentage rollout
    if flags.is_enabled_for_user("beta_features", user_id="user_123"):
        # Show beta features
"""
import hashlib
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
import random

logger = logging.getLogger(__name__)


# =============================================================================
# Flag Types and Models
# =============================================================================


class FlagType(Enum):
    """Types of feature flags."""

    BOOLEAN = "boolean"
    """Simple on/off flag."""

    PERCENTAGE = "percentage"
    """Gradually rollout to percentage of users."""

    USER_LIST = "user_list"
    """Enable for specific users."""

    ENVIRONMENT = "environment"
    """Enable based on environment."""

    SEGMENT = "segment"
    """Enable based on user segments."""

    SCHEDULED = "scheduled"
    """Enable during specific time windows."""


class FlagStatus(Enum):
    """Status of a feature flag."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


@dataclass
class FlagRule:
    """Rule for evaluating flag conditions."""

    rule_id: str
    rule_type: str  # "percentage", "user_list", "segment", "environment"
    conditions: Dict[str, Any]
    enabled: bool = True
    priority: int = 0


@dataclass
class FeatureFlag:
    """Represents a feature flag configuration."""

    key: str
    name: str
    description: str = ""
    flag_type: FlagType = FlagType.BOOLEAN
    status: FlagStatus = FlagStatus.ACTIVE
    default_value: bool = False

    # Targeting
    rules: List[FlagRule] = field(default_factory=list)
    user_whitelist: Set[str] = field(default_factory=set)
    user_blacklist: Set[str] = field(default_factory=set)
    environments: Set[str] = field(default_factory=set)

    # Percentage rollout
    rollout_percentage: float = 0.0

    # Scheduling
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Metadata
    owner: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Metrics
    evaluation_count: int = 0
    enabled_count: int = 0


@dataclass
class EvaluationContext:
    """Context for evaluating feature flags."""

    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    environment: str = "production"
    user_segments: Set[str] = field(default_factory=set)
    user_attributes: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None


@dataclass
class FlagEvaluationResult:
    """Result of a flag evaluation."""

    flag_key: str
    enabled: bool
    reason: str
    rule_id: Optional[str] = None
    variant: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Storage Backend Interface
# =============================================================================


class IFlagStorage(ABC):
    """Abstract interface for flag storage backends."""

    @abstractmethod
    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Get a flag by key."""
        pass

    @abstractmethod
    def get_all_flags(self) -> List[FeatureFlag]:
        """Get all flags."""
        pass

    @abstractmethod
    def save_flag(self, flag: FeatureFlag) -> None:
        """Save or update a flag."""
        pass

    @abstractmethod
    def delete_flag(self, key: str) -> bool:
        """Delete a flag."""
        pass


class InMemoryFlagStorage(IFlagStorage):
    """In-memory flag storage for development and testing."""

    def __init__(self):
        self._flags: Dict[str, FeatureFlag] = {}
        self._lock = threading.RLock()

    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        with self._lock:
            return self._flags.get(key)

    def get_all_flags(self) -> List[FeatureFlag]:
        with self._lock:
            return list(self._flags.values())

    def save_flag(self, flag: FeatureFlag) -> None:
        with self._lock:
            flag.updated_at = datetime.utcnow()
            self._flags[flag.key] = flag

    def delete_flag(self, key: str) -> bool:
        with self._lock:
            if key in self._flags:
                del self._flags[key]
                return True
            return False


class FileFlagStorage(IFlagStorage):
    """File-based flag storage using JSON."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._flags: Dict[str, FeatureFlag] = {}
        self._lock = threading.RLock()
        self._load_flags()

    def _load_flags(self) -> None:
        """Load flags from file."""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    for key, flag_data in data.get("flags", {}).items():
                        self._flags[key] = self._parse_flag(flag_data)
            except Exception as e:
                logger.error(f"Error loading flags: {e}")

    def _parse_flag(self, data: Dict[str, Any]) -> FeatureFlag:
        """Parse flag from dictionary."""
        return FeatureFlag(
            key=data["key"],
            name=data.get("name", data["key"]),
            description=data.get("description", ""),
            flag_type=FlagType(data.get("flag_type", "boolean")),
            status=FlagStatus(data.get("status", "active")),
            default_value=data.get("default_value", False),
            rollout_percentage=data.get("rollout_percentage", 0.0),
            user_whitelist=set(data.get("user_whitelist", [])),
            user_blacklist=set(data.get("user_blacklist", [])),
            environments=set(data.get("environments", [])),
            tags=set(data.get("tags", [])),
        )

    def _save_to_file(self) -> None:
        """Save flags to file."""
        data = {"flags": {}}
        for key, flag in self._flags.items():
            data["flags"][key] = {
                "key": flag.key,
                "name": flag.name,
                "description": flag.description,
                "flag_type": flag.flag_type.value,
                "status": flag.status.value,
                "default_value": flag.default_value,
                "rollout_percentage": flag.rollout_percentage,
                "user_whitelist": list(flag.user_whitelist),
                "user_blacklist": list(flag.user_blacklist),
                "environments": list(flag.environments),
                "tags": list(flag.tags),
            }

        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        with self._lock:
            return self._flags.get(key)

    def get_all_flags(self) -> List[FeatureFlag]:
        with self._lock:
            return list(self._flags.values())

    def save_flag(self, flag: FeatureFlag) -> None:
        with self._lock:
            flag.updated_at = datetime.utcnow()
            self._flags[flag.key] = flag
            self._save_to_file()

    def delete_flag(self, key: str) -> bool:
        with self._lock:
            if key in self._flags:
                del self._flags[key]
                self._save_to_file()
                return True
            return False


# =============================================================================
# Feature Flag Service
# =============================================================================


class FeatureFlagService:
    """
    Feature flag management service.

    Provides feature flag evaluation, management, and analytics.
    """

    def __init__(
        self,
        storage: Optional[IFlagStorage] = None,
        environment: Optional[str] = None,
        default_enabled: bool = False,
    ):
        """
        Initialize the feature flag service.

        Args:
            storage: Flag storage backend
            environment: Current environment (production, staging, development)
            default_enabled: Default value for unknown flags
        """
        self._storage = storage or InMemoryFlagStorage()
        self._environment = environment or os.getenv("ENVIRONMENT", "production")
        self._default_enabled = default_enabled
        self._evaluation_callbacks: List[Callable[[FlagEvaluationResult], None]] = []
        self._override_flags: Dict[str, bool] = {}
        self._lock = threading.RLock()

        # Initialize default flags
        self._initialize_default_flags()

    def _initialize_default_flags(self) -> None:
        """Initialize default feature flags."""
        default_flags = [
            FeatureFlag(
                key="new_calculation_engine",
                name="New Calculation Engine",
                description="Enable the new deterministic calculation engine",
                flag_type=FlagType.PERCENTAGE,
                rollout_percentage=0.0,
                tags={"core", "calculation"},
            ),
            FeatureFlag(
                key="async_agent_execution",
                name="Async Agent Execution",
                description="Enable asynchronous agent execution",
                flag_type=FlagType.BOOLEAN,
                default_value=True,
                tags={"core", "performance"},
            ),
            FeatureFlag(
                key="scope3_category_8_15",
                name="Scope 3 Categories 8-15",
                description="Enable Scope 3 emission categories 8-15",
                flag_type=FlagType.BOOLEAN,
                default_value=False,
                tags={"emissions", "scope3"},
            ),
            FeatureFlag(
                key="monte_carlo_uncertainty",
                name="Monte Carlo Uncertainty",
                description="Enable Monte Carlo uncertainty analysis",
                flag_type=FlagType.PERCENTAGE,
                rollout_percentage=50.0,
                tags={"calculation", "uncertainty"},
            ),
            FeatureFlag(
                key="mqtt_connector",
                name="MQTT Connector",
                description="Enable MQTT connector for IIoT integration",
                flag_type=FlagType.ENVIRONMENT,
                environments={"staging", "development"},
                tags={"integration", "iiot"},
            ),
            FeatureFlag(
                key="qudt_unit_conversion",
                name="QUDT Unit Conversion",
                description="Use QUDT ontology for unit conversions",
                flag_type=FlagType.BOOLEAN,
                default_value=False,
                tags={"calculation", "units"},
            ),
            FeatureFlag(
                key="iso_50001_enms",
                name="ISO 50001 EnMS",
                description="Enable ISO 50001 Energy Management System module",
                flag_type=FlagType.BOOLEAN,
                default_value=False,
                tags={"compliance", "iso"},
            ),
            FeatureFlag(
                key="beta_features",
                name="Beta Features",
                description="Enable beta features for selected users",
                flag_type=FlagType.USER_LIST,
                user_whitelist=set(),
                tags={"beta"},
            ),
        ]

        for flag in default_flags:
            if self._storage.get_flag(flag.key) is None:
                self._storage.save_flag(flag)

    # =========================================================================
    # Flag Evaluation
    # =========================================================================

    def is_enabled(
        self,
        flag_key: str,
        context: Optional[EvaluationContext] = None,
        default: Optional[bool] = None,
    ) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            flag_key: The flag key to check
            context: Evaluation context (user, environment, etc.)
            default: Default value if flag not found

        Returns:
            Whether the flag is enabled
        """
        result = self.evaluate(flag_key, context)
        return result.enabled

    def is_enabled_for_user(
        self,
        flag_key: str,
        user_id: str,
        **kwargs,
    ) -> bool:
        """
        Check if a flag is enabled for a specific user.

        Args:
            flag_key: The flag key to check
            user_id: The user ID
            **kwargs: Additional context attributes

        Returns:
            Whether the flag is enabled for the user
        """
        context = EvaluationContext(
            user_id=user_id,
            user_attributes=kwargs,
            environment=self._environment,
        )
        return self.is_enabled(flag_key, context)

    def evaluate(
        self,
        flag_key: str,
        context: Optional[EvaluationContext] = None,
    ) -> FlagEvaluationResult:
        """
        Evaluate a feature flag with full result details.

        Args:
            flag_key: The flag key to evaluate
            context: Evaluation context

        Returns:
            Detailed evaluation result
        """
        context = context or EvaluationContext(environment=self._environment)

        # Check for override
        with self._lock:
            if flag_key in self._override_flags:
                return FlagEvaluationResult(
                    flag_key=flag_key,
                    enabled=self._override_flags[flag_key],
                    reason="override",
                )

        # Get flag
        flag = self._storage.get_flag(flag_key)
        if flag is None:
            return FlagEvaluationResult(
                flag_key=flag_key,
                enabled=self._default_enabled,
                reason="flag_not_found",
            )

        # Check flag status
        if flag.status != FlagStatus.ACTIVE:
            return FlagEvaluationResult(
                flag_key=flag_key,
                enabled=False,
                reason=f"flag_status_{flag.status.value}",
            )

        # Evaluate based on flag type
        result = self._evaluate_flag(flag, context)

        # Update metrics
        flag.evaluation_count += 1
        if result.enabled:
            flag.enabled_count += 1

        # Call evaluation callbacks
        for callback in self._evaluation_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Evaluation callback error: {e}")

        return result

    def _evaluate_flag(
        self, flag: FeatureFlag, context: EvaluationContext
    ) -> FlagEvaluationResult:
        """Evaluate a flag based on its type and rules."""
        # Check blacklist
        if context.user_id and context.user_id in flag.user_blacklist:
            return FlagEvaluationResult(
                flag_key=flag.key,
                enabled=False,
                reason="user_blacklisted",
            )

        # Check whitelist
        if context.user_id and context.user_id in flag.user_whitelist:
            return FlagEvaluationResult(
                flag_key=flag.key,
                enabled=True,
                reason="user_whitelisted",
            )

        # Check scheduling
        if flag.start_time or flag.end_time:
            now = datetime.utcnow()
            if flag.start_time and now < flag.start_time:
                return FlagEvaluationResult(
                    flag_key=flag.key,
                    enabled=False,
                    reason="not_yet_started",
                )
            if flag.end_time and now > flag.end_time:
                return FlagEvaluationResult(
                    flag_key=flag.key,
                    enabled=False,
                    reason="ended",
                )

        # Evaluate by type
        if flag.flag_type == FlagType.BOOLEAN:
            return FlagEvaluationResult(
                flag_key=flag.key,
                enabled=flag.default_value,
                reason="default_value",
            )

        elif flag.flag_type == FlagType.PERCENTAGE:
            enabled = self._evaluate_percentage(flag, context)
            return FlagEvaluationResult(
                flag_key=flag.key,
                enabled=enabled,
                reason="percentage_rollout",
                metadata={"percentage": flag.rollout_percentage},
            )

        elif flag.flag_type == FlagType.ENVIRONMENT:
            enabled = context.environment in flag.environments
            return FlagEvaluationResult(
                flag_key=flag.key,
                enabled=enabled,
                reason="environment_match" if enabled else "environment_mismatch",
            )

        elif flag.flag_type == FlagType.USER_LIST:
            enabled = context.user_id in flag.user_whitelist if context.user_id else False
            return FlagEvaluationResult(
                flag_key=flag.key,
                enabled=enabled,
                reason="user_list_match" if enabled else "user_list_mismatch",
            )

        return FlagEvaluationResult(
            flag_key=flag.key,
            enabled=flag.default_value,
            reason="default_fallback",
        )

    def _evaluate_percentage(
        self, flag: FeatureFlag, context: EvaluationContext
    ) -> bool:
        """Evaluate percentage-based rollout."""
        if flag.rollout_percentage >= 100.0:
            return True
        if flag.rollout_percentage <= 0.0:
            return False

        # Use consistent hashing for deterministic results
        if context.user_id:
            hash_input = f"{flag.key}:{context.user_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            bucket = hash_value % 100
            return bucket < flag.rollout_percentage
        else:
            # Random for anonymous users
            return random.random() * 100 < flag.rollout_percentage

    # =========================================================================
    # Flag Management
    # =========================================================================

    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Get a flag by key."""
        return self._storage.get_flag(key)

    def get_all_flags(self) -> List[FeatureFlag]:
        """Get all flags."""
        return self._storage.get_all_flags()

    def create_flag(self, flag: FeatureFlag) -> FeatureFlag:
        """Create a new flag."""
        if self._storage.get_flag(flag.key):
            raise ValueError(f"Flag already exists: {flag.key}")
        self._storage.save_flag(flag)
        logger.info(f"Created flag: {flag.key}")
        return flag

    def update_flag(self, flag: FeatureFlag) -> FeatureFlag:
        """Update an existing flag."""
        existing = self._storage.get_flag(flag.key)
        if not existing:
            raise ValueError(f"Flag not found: {flag.key}")
        self._storage.save_flag(flag)
        logger.info(f"Updated flag: {flag.key}")
        return flag

    def delete_flag(self, key: str) -> bool:
        """Delete a flag."""
        result = self._storage.delete_flag(key)
        if result:
            logger.info(f"Deleted flag: {key}")
        return result

    def set_rollout_percentage(self, key: str, percentage: float) -> None:
        """Set rollout percentage for a flag."""
        flag = self._storage.get_flag(key)
        if not flag:
            raise ValueError(f"Flag not found: {key}")
        flag.rollout_percentage = max(0.0, min(100.0, percentage))
        self._storage.save_flag(flag)
        logger.info(f"Set rollout for {key} to {percentage}%")

    def enable_flag(self, key: str) -> None:
        """Enable a flag."""
        flag = self._storage.get_flag(key)
        if not flag:
            raise ValueError(f"Flag not found: {key}")
        flag.default_value = True
        flag.status = FlagStatus.ACTIVE
        self._storage.save_flag(flag)
        logger.info(f"Enabled flag: {key}")

    def disable_flag(self, key: str) -> None:
        """Disable a flag."""
        flag = self._storage.get_flag(key)
        if not flag:
            raise ValueError(f"Flag not found: {key}")
        flag.default_value = False
        self._storage.save_flag(flag)
        logger.info(f"Disabled flag: {key}")

    # =========================================================================
    # Overrides
    # =========================================================================

    def set_override(self, key: str, enabled: bool) -> None:
        """Set a local override for a flag."""
        with self._lock:
            self._override_flags[key] = enabled
        logger.info(f"Set override for {key}: {enabled}")

    def clear_override(self, key: str) -> None:
        """Clear a local override."""
        with self._lock:
            self._override_flags.pop(key, None)
        logger.info(f"Cleared override for {key}")

    def clear_all_overrides(self) -> None:
        """Clear all local overrides."""
        with self._lock:
            self._override_flags.clear()
        logger.info("Cleared all overrides")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def add_evaluation_callback(
        self, callback: Callable[[FlagEvaluationResult], None]
    ) -> None:
        """Add a callback for flag evaluations."""
        self._evaluation_callbacks.append(callback)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get flag statistics."""
        flags = self.get_all_flags()

        total = len(flags)
        active = sum(1 for f in flags if f.status == FlagStatus.ACTIVE)
        enabled = sum(1 for f in flags if f.default_value)

        evaluations = sum(f.evaluation_count for f in flags)
        enabled_evaluations = sum(f.enabled_count for f in flags)

        return {
            "total_flags": total,
            "active_flags": active,
            "enabled_flags": enabled,
            "total_evaluations": evaluations,
            "enabled_evaluations": enabled_evaluations,
            "enabled_rate": enabled_evaluations / evaluations if evaluations > 0 else 0,
            "by_type": {
                ft.value: sum(1 for f in flags if f.flag_type == ft)
                for ft in FlagType
            },
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_service: Optional[FeatureFlagService] = None
_service_lock = threading.Lock()


def get_feature_flags() -> FeatureFlagService:
    """Get the singleton feature flag service."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = FeatureFlagService()
    return _service


def reset_feature_flags() -> None:
    """Reset the singleton instance (for testing)."""
    global _service
    with _service_lock:
        _service = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "FeatureFlagService",
    "FeatureFlag",
    "FlagType",
    "FlagStatus",
    "FlagRule",
    "EvaluationContext",
    "FlagEvaluationResult",
    "IFlagStorage",
    "InMemoryFlagStorage",
    "FileFlagStorage",
    "get_feature_flags",
    "reset_feature_flags",
]
