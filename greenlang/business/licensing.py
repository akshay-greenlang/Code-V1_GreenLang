"""
GreenLang Licensing Module

Provides license management, validation, and feature gating functionality.

Components:
    - LicenseManager: Core license management class
    - License: License data model
    - Feature gating by license tier
    - License validation and auditing

Example:
    >>> from greenlang.business.licensing import LicenseManager
    >>>
    >>> license_mgr = LicenseManager()
    >>> license_mgr.load_license("GL-LIC-XXXXX-XXXXX-XXXXX")
    >>>
    >>> if license_mgr.has_feature("ml_optimization"):
    ...     enable_ml_features()
    >>>
    >>> usage = license_mgr.check_usage("users")
    >>> if usage.remaining > 0:
    ...     create_user()
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
import hashlib
import hmac
import base64
import json
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class LicenseType(Enum):
    """Types of licenses available."""
    TRIAL = "trial"
    SUBSCRIPTION = "subscription"
    PERPETUAL = "perpetual"
    USAGE_BASED = "usage_based"
    ENTERPRISE = "enterprise"
    OEM = "oem"
    ACADEMIC = "academic"
    NONPROFIT = "nonprofit"


class LicenseTier(Enum):
    """License tier/edition levels."""
    ESSENTIALS = "essentials"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class LicenseStatus(Enum):
    """Status of a license."""
    ACTIVE = "active"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    TRIAL = "trial"
    GRACE_PERIOD = "grace_period"


class LicenseValidationError(Exception):
    """Exception raised when license validation fails."""
    pass


class LicenseExpiredError(Exception):
    """Exception raised when license has expired."""
    pass


class FeatureNotLicensedError(Exception):
    """Exception raised when attempting to use an unlicensed feature."""

    def __init__(self, feature: str, required_tier: str = None):
        self.feature = feature
        self.required_tier = required_tier
        message = f"Feature '{feature}' is not licensed"
        if required_tier:
            message += f". Requires {required_tier} tier or higher."
        super().__init__(message)


class UsageLimitExceededError(Exception):
    """Exception raised when usage limits are exceeded."""

    def __init__(self, resource: str, limit: int, current: int):
        self.resource = resource
        self.limit = limit
        self.current = current
        super().__init__(
            f"Usage limit exceeded for '{resource}': {current}/{limit}"
        )


@dataclass
class UsageQuota:
    """Usage quota for a specific resource."""
    resource: str
    limit: int
    used: int
    reset_date: Optional[date] = None

    @property
    def remaining(self) -> int:
        """Get remaining quota."""
        if self.limit < 0:  # Unlimited
            return float('inf')
        return max(0, self.limit - self.used)

    @property
    def percentage_used(self) -> float:
        """Get percentage of quota used."""
        if self.limit < 0:
            return 0.0
        return min(100.0, (self.used / self.limit) * 100)

    @property
    def is_unlimited(self) -> bool:
        """Check if quota is unlimited."""
        return self.limit < 0

    @property
    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        if self.limit < 0:
            return False
        return self.used >= self.limit


@dataclass
class LicenseEntitlements:
    """Entitlements granted by a license."""
    tier: LicenseTier
    features: Set[str]
    quotas: Dict[str, int]
    addons: Set[str]
    integrations: Set[str]

    def has_feature(self, feature: str) -> bool:
        """Check if a feature is entitled."""
        return feature in self.features

    def get_quota(self, resource: str) -> int:
        """Get quota for a resource."""
        return self.quotas.get(resource, 0)


@dataclass
class License:
    """License data model."""
    license_id: str
    license_key: str
    customer_id: str
    customer_name: str
    license_type: LicenseType
    tier: LicenseTier
    status: LicenseStatus
    issue_date: date
    start_date: date
    expiry_date: date
    features: Set[str] = field(default_factory=set)
    quotas: Dict[str, int] = field(default_factory=dict)
    addons: Set[str] = field(default_factory=set)
    integrations: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if license is currently valid."""
        today = date.today()
        return (
            self.status in (LicenseStatus.ACTIVE, LicenseStatus.TRIAL, LicenseStatus.GRACE_PERIOD)
            and self.start_date <= today
            and (self.expiry_date >= today or self.status == LicenseStatus.GRACE_PERIOD)
        )

    @property
    def is_expired(self) -> bool:
        """Check if license is expired."""
        return date.today() > self.expiry_date

    @property
    def days_until_expiry(self) -> int:
        """Get days until license expires."""
        return (self.expiry_date - date.today()).days

    @property
    def is_perpetual(self) -> bool:
        """Check if license is perpetual."""
        return self.license_type == LicenseType.PERPETUAL

    def get_entitlements(self) -> LicenseEntitlements:
        """Get entitlements for this license."""
        return LicenseEntitlements(
            tier=self.tier,
            features=self.features,
            quotas=self.quotas,
            addons=self.addons,
            integrations=self.integrations,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert license to dictionary."""
        return {
            "license_id": self.license_id,
            "license_key": self.license_key,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "license_type": self.license_type.value,
            "tier": self.tier.value,
            "status": self.status.value,
            "issue_date": self.issue_date.isoformat(),
            "start_date": self.start_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat(),
            "features": list(self.features),
            "quotas": self.quotas,
            "addons": list(self.addons),
            "integrations": list(self.integrations),
            "metadata": self.metadata,
        }


# Feature catalog by tier
TIER_FEATURES: Dict[LicenseTier, Set[str]] = {
    LicenseTier.ESSENTIALS: {
        "core_platform",
        "data_import_csv",
        "data_import_api_readonly",
        "basic_dashboards",
        "standard_reports",
        "basic_alerting",
        "email_support",
        "basic_compliance",
        "audit_trail",
        "data_export",
    },
    LicenseTier.PROFESSIONAL: {
        # Includes all Essentials features
        "core_platform",
        "data_import_csv",
        "data_import_api_readonly",
        "basic_dashboards",
        "standard_reports",
        "basic_alerting",
        "email_support",
        "basic_compliance",
        "audit_trail",
        "data_export",
        # Plus Professional features
        "data_import_api_readwrite",
        "advanced_dashboards",
        "custom_reports",
        "ml_optimization",
        "predictive_analytics",
        "advanced_alerting",
        "sso_saml",
        "priority_support",
        "multi_regulation",
        "realtime_streaming",
        "advanced_integrations",
        "phone_support",
    },
    LicenseTier.ENTERPRISE: {
        # Includes all Professional features
        "core_platform",
        "data_import_csv",
        "data_import_api_readonly",
        "basic_dashboards",
        "standard_reports",
        "basic_alerting",
        "email_support",
        "basic_compliance",
        "audit_trail",
        "data_export",
        "data_import_api_readwrite",
        "advanced_dashboards",
        "custom_reports",
        "ml_optimization",
        "predictive_analytics",
        "advanced_alerting",
        "sso_saml",
        "priority_support",
        "multi_regulation",
        "realtime_streaming",
        "advanced_integrations",
        "phone_support",
        # Plus Enterprise features
        "agent_factory",
        "custom_agents",
        "edge_computing",
        "custom_ml_models",
        "white_label",
        "on_premise_deployment",
        "dedicated_instance",
        "vpc_privatelink",
        "custom_roles",
        "premium_support_24x7",
        "dedicated_tam",
        "custom_integrations",
        "unlimited_api",
        "unlimited_storage",
        "multi_region",
        "data_residency",
    },
}

# Default quotas by tier
TIER_QUOTAS: Dict[LicenseTier, Dict[str, int]] = {
    LicenseTier.ESSENTIALS: {
        "users": 10,
        "data_points_monthly": 100000,
        "agents": 25,
        "facilities": 5,
        "api_calls_monthly": 100000,
        "storage_gb": 50,
        "reports_monthly": 100,
        "dashboards": 5,
        "integrations": 5,
    },
    LicenseTier.PROFESSIONAL: {
        "users": 50,
        "data_points_monthly": 1000000,
        "agents": 100,
        "facilities": 25,
        "api_calls_monthly": 1000000,
        "storage_gb": 500,
        "reports_monthly": 1000,
        "dashboards": 25,
        "integrations": 15,
        "ml_models": 10,
    },
    LicenseTier.ENTERPRISE: {
        "users": -1,  # Unlimited
        "data_points_monthly": -1,
        "agents": 500,
        "facilities": -1,
        "api_calls_monthly": -1,
        "storage_gb": 5000,
        "reports_monthly": -1,
        "dashboards": -1,
        "integrations": -1,
        "ml_models": -1,
        "custom_dev_hours": 40,
    },
}


class LicenseManager:
    """
    Manager for license validation, feature gating, and usage tracking.

    Example:
        >>> manager = LicenseManager()
        >>> manager.load_license_key("GL-LIC-XXXXX-XXXXX-XXXXX")
        >>>
        >>> # Check feature access
        >>> if manager.has_feature("ml_optimization"):
        ...     run_ml_optimization()
        >>>
        >>> # Check and consume quota
        >>> if manager.check_quota("users").remaining > 0:
        ...     manager.consume_quota("users", 1)
        ...     create_user()
    """

    def __init__(
        self,
        license_key: Optional[str] = None,
        license_service_url: Optional[str] = None,
        offline_mode: bool = False,
        grace_period_days: int = 30,
    ):
        """
        Initialize the license manager.

        Args:
            license_key: License key to load
            license_service_url: URL of license validation service
            offline_mode: Enable offline license validation
            grace_period_days: Days of grace period after expiry
        """
        self.license_service_url = license_service_url
        self.offline_mode = offline_mode
        self.grace_period_days = grace_period_days

        self._license: Optional[License] = None
        self._usage: Dict[str, int] = {}
        self._usage_reset_date: Optional[date] = None
        self._secret_key = "greenlang-license-validation-key"  # Would be env var in prod

        if license_key:
            self.load_license_key(license_key)

    @property
    def license(self) -> Optional[License]:
        """Get the current license."""
        return self._license

    @property
    def is_licensed(self) -> bool:
        """Check if a valid license is loaded."""
        return self._license is not None and self._license.is_valid

    @property
    def tier(self) -> Optional[LicenseTier]:
        """Get the current license tier."""
        return self._license.tier if self._license else None

    def _generate_license_key(
        self,
        customer_id: str,
        tier: LicenseTier,
        license_type: LicenseType,
    ) -> str:
        """Generate a license key (for internal use)."""
        timestamp = datetime.now().strftime("%Y%m%d")
        data = f"{customer_id}:{tier.value}:{license_type.value}:{timestamp}"
        signature = hmac.new(
            self._secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()[:16].upper()

        # Format: GL-LIC-XXXX-XXXX-XXXX-XXXX
        key_parts = [
            "GL",
            "LIC",
            signature[:4],
            signature[4:8],
            signature[8:12],
            signature[12:16],
        ]
        return "-".join(key_parts)

    def _validate_license_key_format(self, key: str) -> bool:
        """Validate license key format."""
        parts = key.split("-")
        if len(parts) != 6:
            return False
        if parts[0] != "GL" or parts[1] != "LIC":
            return False
        for part in parts[2:]:
            if len(part) != 4 or not part.isalnum():
                return False
        return True

    def load_license_key(self, license_key: str) -> License:
        """
        Load and validate a license key.

        Args:
            license_key: The license key to load

        Returns:
            License object if valid

        Raises:
            LicenseValidationError: If license key is invalid
        """
        if not self._validate_license_key_format(license_key):
            raise LicenseValidationError(f"Invalid license key format: {license_key}")

        # In production, this would call the license service
        # For now, create a demo license based on the key
        license = self._create_demo_license(license_key)

        if not license.is_valid:
            if license.is_expired:
                # Check grace period
                grace_end = license.expiry_date + timedelta(days=self.grace_period_days)
                if date.today() <= grace_end:
                    license.status = LicenseStatus.GRACE_PERIOD
                    logger.warning(
                        f"License {license_key} is in grace period. "
                        f"Expires: {grace_end}"
                    )
                else:
                    raise LicenseExpiredError(
                        f"License {license_key} expired on {license.expiry_date}"
                    )

        self._license = license
        self._reset_usage()
        logger.info(f"License loaded: {license.license_id} ({license.tier.value})")
        return license

    def _create_demo_license(self, license_key: str) -> License:
        """Create a demo license for development/testing."""
        # Determine tier from key (demo logic)
        key_sum = sum(ord(c) for c in license_key)
        if key_sum % 3 == 0:
            tier = LicenseTier.ENTERPRISE
        elif key_sum % 3 == 1:
            tier = LicenseTier.PROFESSIONAL
        else:
            tier = LicenseTier.ESSENTIALS

        today = date.today()
        return License(
            license_id=f"LIC-{license_key[-8:]}",
            license_key=license_key,
            customer_id="DEMO-001",
            customer_name="Demo Customer",
            license_type=LicenseType.SUBSCRIPTION,
            tier=tier,
            status=LicenseStatus.ACTIVE,
            issue_date=today - timedelta(days=30),
            start_date=today - timedelta(days=30),
            expiry_date=today + timedelta(days=335),
            features=TIER_FEATURES[tier].copy(),
            quotas=TIER_QUOTAS[tier].copy(),
            addons=set(),
            integrations=set(),
            metadata={"demo": True},
        )

    def load_license(self, license: License) -> None:
        """Load a license object directly."""
        self._license = license
        self._reset_usage()

    def _reset_usage(self) -> None:
        """Reset usage counters."""
        self._usage = {}
        self._usage_reset_date = date.today().replace(day=1)

    def _check_usage_reset(self) -> None:
        """Check if usage should be reset (monthly)."""
        today = date.today()
        if self._usage_reset_date and today.month != self._usage_reset_date.month:
            self._reset_usage()

    def has_feature(self, feature: str) -> bool:
        """
        Check if a feature is licensed.

        Args:
            feature: Feature name to check

        Returns:
            True if feature is licensed
        """
        if not self._license:
            return False
        return feature in self._license.features

    def require_feature(self, feature: str) -> None:
        """
        Require a feature, raising exception if not licensed.

        Args:
            feature: Feature name to require

        Raises:
            FeatureNotLicensedError: If feature is not licensed
        """
        if not self.has_feature(feature):
            required_tier = None
            for tier in [LicenseTier.PROFESSIONAL, LicenseTier.ENTERPRISE]:
                if feature in TIER_FEATURES[tier]:
                    required_tier = tier.value
                    break
            raise FeatureNotLicensedError(feature, required_tier)

    def check_quota(self, resource: str) -> UsageQuota:
        """
        Check quota for a resource.

        Args:
            resource: Resource name to check

        Returns:
            UsageQuota with current status
        """
        self._check_usage_reset()

        if not self._license:
            return UsageQuota(resource=resource, limit=0, used=0)

        limit = self._license.quotas.get(resource, 0)
        used = self._usage.get(resource, 0)

        return UsageQuota(
            resource=resource,
            limit=limit,
            used=used,
            reset_date=self._usage_reset_date,
        )

    def consume_quota(self, resource: str, amount: int = 1) -> UsageQuota:
        """
        Consume quota for a resource.

        Args:
            resource: Resource name
            amount: Amount to consume

        Returns:
            Updated UsageQuota

        Raises:
            UsageLimitExceededError: If consumption would exceed limit
        """
        quota = self.check_quota(resource)

        if not quota.is_unlimited and quota.used + amount > quota.limit:
            raise UsageLimitExceededError(
                resource=resource,
                limit=quota.limit,
                current=quota.used + amount,
            )

        self._usage[resource] = self._usage.get(resource, 0) + amount
        return self.check_quota(resource)

    def get_all_quotas(self) -> Dict[str, UsageQuota]:
        """Get all quota statuses."""
        if not self._license:
            return {}

        return {
            resource: self.check_quota(resource)
            for resource in self._license.quotas.keys()
        }

    def get_entitlements(self) -> Optional[LicenseEntitlements]:
        """Get current license entitlements."""
        if not self._license:
            return None
        return self._license.get_entitlements()

    def validate(self) -> Dict[str, Any]:
        """
        Perform full license validation.

        Returns:
            Validation result dictionary
        """
        if not self._license:
            return {
                "valid": False,
                "error": "No license loaded",
            }

        warnings = []

        # Check expiry
        if self._license.is_expired:
            if self._license.status == LicenseStatus.GRACE_PERIOD:
                grace_end = self._license.expiry_date + timedelta(days=self.grace_period_days)
                warnings.append(
                    f"License is in grace period. Expires: {grace_end}"
                )
            else:
                return {
                    "valid": False,
                    "error": f"License expired on {self._license.expiry_date}",
                }

        # Check expiry warning
        if self._license.days_until_expiry <= 30:
            warnings.append(
                f"License expires in {self._license.days_until_expiry} days"
            )

        # Check quota warnings
        for resource, quota in self.get_all_quotas().items():
            if quota.percentage_used >= 90:
                warnings.append(
                    f"{resource} usage at {quota.percentage_used:.0f}%"
                )

        return {
            "valid": True,
            "license_id": self._license.license_id,
            "tier": self._license.tier.value,
            "status": self._license.status.value,
            "expiry_date": self._license.expiry_date.isoformat(),
            "days_until_expiry": self._license.days_until_expiry,
            "warnings": warnings,
        }

    def get_license_info(self) -> Dict[str, Any]:
        """Get license information summary."""
        if not self._license:
            return {"licensed": False}

        return {
            "licensed": True,
            "license_id": self._license.license_id,
            "customer": self._license.customer_name,
            "tier": self._license.tier.value,
            "type": self._license.license_type.value,
            "status": self._license.status.value,
            "expiry_date": self._license.expiry_date.isoformat(),
            "days_remaining": self._license.days_until_expiry,
            "features_count": len(self._license.features),
            "addons": list(self._license.addons),
            "quotas": {
                k: {"limit": v, "used": self._usage.get(k, 0)}
                for k, v in self._license.quotas.items()
            },
        }


def require_license(
    feature: Optional[str] = None,
    tier: Optional[LicenseTier] = None,
):
    """
    Decorator to require a valid license for a function.

    Args:
        feature: Required feature name
        tier: Minimum required tier

    Example:
        >>> @require_license(feature="ml_optimization")
        ... def run_ml_optimization():
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get license manager from first arg or global
            license_manager = None

            if args and hasattr(args[0], 'license_manager'):
                license_manager = args[0].license_manager
            elif 'license_manager' in kwargs:
                license_manager = kwargs['license_manager']

            if license_manager is None:
                raise LicenseValidationError("No license manager available")

            if not license_manager.is_licensed:
                raise LicenseValidationError("No valid license")

            if feature:
                license_manager.require_feature(feature)

            if tier and license_manager.tier:
                tier_order = [LicenseTier.ESSENTIALS, LicenseTier.PROFESSIONAL, LicenseTier.ENTERPRISE]
                current_idx = tier_order.index(license_manager.tier)
                required_idx = tier_order.index(tier)
                if current_idx < required_idx:
                    raise FeatureNotLicensedError(
                        f"Function requires {tier.value} tier",
                        tier.value
                    )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_quota(resource: str, amount: int = 1):
    """
    Decorator to require and consume quota for a function.

    Args:
        resource: Resource to check/consume
        amount: Amount to consume

    Example:
        >>> @require_quota("api_calls", 1)
        ... def make_api_call():
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            license_manager = None

            if args and hasattr(args[0], 'license_manager'):
                license_manager = args[0].license_manager
            elif 'license_manager' in kwargs:
                license_manager = kwargs['license_manager']

            if license_manager is None:
                raise LicenseValidationError("No license manager available")

            # Check and consume quota
            license_manager.consume_quota(resource, amount)

            return func(*args, **kwargs)
        return wrapper
    return decorator


class LicenseAudit:
    """
    Audit trail for license usage and validation events.

    Example:
        >>> audit = LicenseAudit()
        >>> audit.log_validation(license_manager)
        >>> audit.log_feature_access("ml_optimization", granted=True)
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the audit logger."""
        self.storage_path = storage_path
        self._events: List[Dict[str, Any]] = []

    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log an audit event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **details,
        }
        self._events.append(event)
        logger.info(f"License audit: {event_type} - {details}")

    def log_validation(self, manager: LicenseManager) -> None:
        """Log a license validation event."""
        validation = manager.validate()
        self.log_event("validation", {
            "valid": validation.get("valid"),
            "license_id": validation.get("license_id"),
            "tier": validation.get("tier"),
            "warnings": validation.get("warnings", []),
        })

    def log_feature_access(
        self,
        feature: str,
        granted: bool,
        reason: Optional[str] = None,
    ) -> None:
        """Log a feature access event."""
        self.log_event("feature_access", {
            "feature": feature,
            "granted": granted,
            "reason": reason,
        })

    def log_quota_consumption(
        self,
        resource: str,
        amount: int,
        new_total: int,
        limit: int,
    ) -> None:
        """Log a quota consumption event."""
        self.log_event("quota_consumption", {
            "resource": resource,
            "amount": amount,
            "new_total": new_total,
            "limit": limit,
            "percentage": (new_total / limit * 100) if limit > 0 else 0,
        })

    def get_events(
        self,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit events with optional filtering."""
        events = self._events

        if event_type:
            events = [e for e in events if e["event_type"] == event_type]

        if since:
            events = [
                e for e in events
                if datetime.fromisoformat(e["timestamp"]) >= since
            ]

        return events

    def generate_report(self) -> Dict[str, Any]:
        """Generate audit summary report."""
        validation_events = self.get_events("validation")
        feature_events = self.get_events("feature_access")
        quota_events = self.get_events("quota_consumption")

        return {
            "total_events": len(self._events),
            "validation_events": len(validation_events),
            "feature_access_events": len(feature_events),
            "quota_consumption_events": len(quota_events),
            "denied_feature_access": sum(
                1 for e in feature_events if not e.get("granted")
            ),
            "quota_warnings": sum(
                1 for e in quota_events if e.get("percentage", 0) >= 90
            ),
        }
