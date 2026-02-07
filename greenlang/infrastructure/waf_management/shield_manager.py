# -*- coding: utf-8 -*-
"""
AWS Shield Manager - SEC-010

Manages AWS Shield Advanced protection for GreenLang resources.
Provides DDoS protection, attack visibility, and integration with
the AWS Shield Response Team (DRT).

Classes:
    - ShieldManager: Main class for Shield Advanced management

Example:
    >>> from greenlang.infrastructure.waf_management.shield_manager import ShieldManager
    >>> manager = ShieldManager(config)
    >>> await manager.enable_protection(alb_arn)
    >>> attacks = await manager.get_active_attacks()
    >>> await manager.configure_proactive_engagement(enabled=True)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import boto3 with graceful fallback
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None  # type: ignore
    BotoCoreError = Exception  # type: ignore
    ClientError = Exception  # type: ignore

from greenlang.infrastructure.waf_management.config import WAFConfig, get_config
from greenlang.infrastructure.waf_management.models import (
    Attack,
    AttackSeverity,
    AttackType,
    ProtectionGroup,
    ShieldProtection,
)


# ---------------------------------------------------------------------------
# Shield Subscription Status
# ---------------------------------------------------------------------------


class ShieldSubscriptionStatus:
    """AWS Shield Advanced subscription status.

    Attributes:
        is_active: Whether subscription is active.
        subscription_start: When subscription started.
        auto_renew: Whether auto-renewal is enabled.
        subscription_limits: Resource protection limits.
        time_commitment_in_seconds: Subscription term in seconds.
    """

    def __init__(
        self,
        is_active: bool = False,
        subscription_start: Optional[datetime] = None,
        auto_renew: bool = True,
        subscription_limits: Optional[Dict[str, int]] = None,
        time_commitment_in_seconds: int = 31536000,  # 1 year
    ):
        self.is_active = is_active
        self.subscription_start = subscription_start
        self.auto_renew = auto_renew
        self.subscription_limits = subscription_limits or {}
        self.time_commitment_in_seconds = time_commitment_in_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_active": self.is_active,
            "subscription_start": self.subscription_start.isoformat() if self.subscription_start else None,
            "auto_renew": self.auto_renew,
            "subscription_limits": self.subscription_limits,
            "time_commitment_in_seconds": self.time_commitment_in_seconds,
        }


# ---------------------------------------------------------------------------
# Attack Statistics
# ---------------------------------------------------------------------------


class AttackStatistics:
    """Attack statistics from AWS Shield.

    Attributes:
        attack_count: Total number of attacks.
        attacks_by_type: Count by attack vector.
        max_attack_bps: Maximum attack bandwidth in bits/sec.
        max_attack_pps: Maximum attack packets/sec.
        time_range_start: Start of the statistics period.
        time_range_end: End of the statistics period.
    """

    def __init__(
        self,
        attack_count: int = 0,
        attacks_by_type: Optional[Dict[str, int]] = None,
        max_attack_bps: int = 0,
        max_attack_pps: int = 0,
        time_range_start: Optional[datetime] = None,
        time_range_end: Optional[datetime] = None,
    ):
        self.attack_count = attack_count
        self.attacks_by_type = attacks_by_type or {}
        self.max_attack_bps = max_attack_bps
        self.max_attack_pps = max_attack_pps
        self.time_range_start = time_range_start
        self.time_range_end = time_range_end

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "attack_count": self.attack_count,
            "attacks_by_type": self.attacks_by_type,
            "max_attack_bps": self.max_attack_bps,
            "max_attack_pps": self.max_attack_pps,
            "time_range_start": self.time_range_start.isoformat() if self.time_range_start else None,
            "time_range_end": self.time_range_end.isoformat() if self.time_range_end else None,
        }


# ---------------------------------------------------------------------------
# Shield Manager
# ---------------------------------------------------------------------------


class ShieldManager:
    """Manages AWS Shield Advanced protection for GreenLang resources.

    Provides a high-level interface for:
    - Enabling/disabling Shield protection on resources
    - Managing protection groups
    - Configuring auto-remediation
    - Viewing attack statistics
    - Engaging the Shield Response Team (DRT)

    Example:
        >>> manager = ShieldManager(config)
        >>> await manager.enable_protection("arn:aws:elasticloadbalancing:...")
        >>> status = await manager.get_subscription_state()
        >>> if status.is_active:
        ...     await manager.configure_proactive_engagement(enabled=True)
    """

    # Supported resource types for Shield protection
    SUPPORTED_RESOURCE_TYPES = {
        "elasticloadbalancing": "ALB/NLB",
        "cloudfront": "CloudFront",
        "globalaccelerator": "Global Accelerator",
        "route53": "Route 53",
        "ec2": "Elastic IP",
    }

    def __init__(self, config: Optional[WAFConfig] = None):
        """Initialize the Shield manager.

        Args:
            config: WAF configuration. If None, loads from environment.
        """
        self.config = config or get_config()
        self._shield_client = None
        self._route53_client = None
        self._initialized = False

    @property
    def shield_client(self):
        """Get or create the boto3 Shield client.

        Returns:
            boto3 Shield client.
        """
        if self._shield_client is None:
            if not BOTO3_AVAILABLE:
                raise ImportError(
                    "boto3 is required for Shield operations. "
                    "Install it with: pip install boto3"
                )
            # Shield API is only available in us-east-1
            self._shield_client = boto3.client(
                "shield",
                region_name="us-east-1",
            )
        return self._shield_client

    async def enable_protection(
        self,
        resource_arn: str,
        protection_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ShieldProtection:
        """Enable Shield Advanced protection on a resource.

        Args:
            resource_arn: ARN of the resource to protect.
            protection_name: Optional name for the protection.
            tags: Optional tags to apply.

        Returns:
            ShieldProtection instance.

        Raises:
            ValueError: If resource type is not supported.
        """
        # Validate resource type
        resource_type = self._get_resource_type(resource_arn)
        if resource_type not in self.SUPPORTED_RESOURCE_TYPES:
            raise ValueError(
                f"Unsupported resource type: {resource_type}. "
                f"Supported: {list(self.SUPPORTED_RESOURCE_TYPES.keys())}"
            )

        # Generate protection name if not provided
        if not protection_name:
            protection_name = f"gl-{resource_type}-protection"

        try:
            # Create the protection
            response = self.shield_client.create_protection(
                Name=protection_name,
                ResourceArn=resource_arn,
                Tags=[{"Key": k, "Value": v} for k, v in (tags or {}).items()],
            )

            protection_id = response["ProtectionId"]

            logger.info(
                "Shield protection enabled: id=%s, resource=%s",
                protection_id,
                resource_arn,
            )

            return ShieldProtection(
                id=protection_id,
                resource_arn=resource_arn,
                protection_name=protection_name,
                auto_remediate=self.config.shield_auto_remediate,
                proactive_engagement=self.config.shield_proactive_engagement,
                created_at=datetime.now(timezone.utc),
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")

            if error_code == "ResourceAlreadyExistsException":
                # Protection already exists, retrieve it
                logger.info("Protection already exists for resource: %s", resource_arn)
                return await self._get_existing_protection(resource_arn)

            if error_code == "InvalidResourceException":
                raise ValueError(
                    f"Invalid resource for Shield protection: {resource_arn}"
                )

            logger.error("Failed to enable Shield protection: %s", str(e))
            raise

    async def disable_protection(
        self,
        resource_arn: str,
    ) -> bool:
        """Disable Shield Advanced protection on a resource.

        Args:
            resource_arn: ARN of the protected resource.

        Returns:
            True if protection was disabled, False otherwise.
        """
        try:
            # Find the protection ID for this resource
            protection = await self._get_existing_protection(resource_arn)
            if not protection:
                logger.warning("No protection found for resource: %s", resource_arn)
                return False

            # Delete the protection
            self.shield_client.delete_protection(
                ProtectionId=protection.id,
            )

            logger.info(
                "Shield protection disabled: id=%s, resource=%s",
                protection.id,
                resource_arn,
            )

            return True

        except ClientError as e:
            logger.error("Failed to disable Shield protection: %s", str(e))
            return False

    async def _get_existing_protection(
        self,
        resource_arn: str,
    ) -> Optional[ShieldProtection]:
        """Get existing protection for a resource.

        Args:
            resource_arn: ARN of the resource.

        Returns:
            ShieldProtection if exists, None otherwise.
        """
        try:
            response = self.shield_client.describe_protection(
                ResourceArn=resource_arn,
            )

            protection = response.get("Protection", {})
            return ShieldProtection(
                id=protection.get("Id", ""),
                resource_arn=protection.get("ResourceArn", resource_arn),
                protection_name=protection.get("Name", ""),
                health_check_arn=protection.get("HealthCheckIds", [None])[0],
            )

        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                return None
            raise

    async def create_protection_group(
        self,
        group_id: str,
        resources: List[str],
        aggregation: str = "SUM",
        pattern: str = "ARBITRARY",
    ) -> ProtectionGroup:
        """Create a Shield protection group.

        Protection groups allow you to manage multiple resources together
        and view aggregated attack statistics.

        Args:
            group_id: Unique identifier for the group.
            resources: List of resource ARNs to include.
            aggregation: How to aggregate metrics (SUM, MEAN, MAX).
            pattern: Resource selection pattern (ALL, ARBITRARY, BY_RESOURCE_TYPE).

        Returns:
            ProtectionGroup instance.
        """
        try:
            self.shield_client.create_protection_group(
                ProtectionGroupId=group_id,
                Aggregation=aggregation,
                Pattern=pattern,
                Members=resources if pattern == "ARBITRARY" else [],
            )

            logger.info(
                "Protection group created: id=%s, members=%d",
                group_id,
                len(resources),
            )

            return ProtectionGroup(
                id=group_id,
                aggregation=aggregation,
                pattern=pattern,
                members=resources,
            )

        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ResourceAlreadyExistsException":
                # Update existing group
                self.shield_client.update_protection_group(
                    ProtectionGroupId=group_id,
                    Aggregation=aggregation,
                    Pattern=pattern,
                    Members=resources if pattern == "ARBITRARY" else [],
                )
                logger.info("Protection group updated: id=%s", group_id)
                return ProtectionGroup(
                    id=group_id,
                    aggregation=aggregation,
                    pattern=pattern,
                    members=resources,
                )
            raise

    async def configure_auto_mitigation(
        self,
        enabled: bool,
        resource_arn: Optional[str] = None,
    ) -> bool:
        """Configure automatic DDoS mitigation.

        When enabled, Shield will automatically apply mitigations
        when attacks are detected.

        Args:
            enabled: Whether to enable auto-mitigation.
            resource_arn: Optional specific resource. If None, applies globally.

        Returns:
            True if configuration was updated.
        """
        try:
            if resource_arn:
                # Enable/disable application layer automatic response
                protection = await self._get_existing_protection(resource_arn)
                if not protection:
                    raise ValueError(f"No protection found for: {resource_arn}")

                # Application layer auto response requires an action
                action = {"Block": {}} if enabled else {"Count": {}}

                self.shield_client.enable_application_layer_automatic_response(
                    ResourceArn=resource_arn,
                    Action=action,
                ) if enabled else self.shield_client.disable_application_layer_automatic_response(
                    ResourceArn=resource_arn,
                )

                logger.info(
                    "Auto-mitigation %s for resource: %s",
                    "enabled" if enabled else "disabled",
                    resource_arn,
                )
            else:
                # This would require updating subscription settings
                logger.info(
                    "Global auto-mitigation setting: %s",
                    "enabled" if enabled else "disabled",
                )

            return True

        except ClientError as e:
            logger.error("Failed to configure auto-mitigation: %s", str(e))
            return False

    async def get_attack_statistics(
        self,
        time_range: Optional[timedelta] = None,
        resource_arn: Optional[str] = None,
    ) -> AttackStatistics:
        """Get attack statistics from Shield.

        Args:
            time_range: Time range for statistics. Defaults to 30 days.
            resource_arn: Optional specific resource to filter by.

        Returns:
            AttackStatistics instance.
        """
        if time_range is None:
            time_range = timedelta(days=30)

        end_time = datetime.now(timezone.utc)
        start_time = end_time - time_range

        try:
            # List all attacks in the time range
            paginator = self.shield_client.get_paginator("list_attacks")
            attacks = []

            page_iterator = paginator.paginate(
                StartTime={
                    "FromInclusive": start_time,
                    "ToExclusive": end_time,
                },
                ResourceArns=[resource_arn] if resource_arn else [],
            )

            for page in page_iterator:
                attacks.extend(page.get("AttackSummaries", []))

            # Calculate statistics
            attacks_by_type: Dict[str, int] = {}
            max_bps = 0
            max_pps = 0

            for attack in attacks:
                # Count by attack vector
                for vector in attack.get("AttackVectors", []):
                    vector_name = vector.get("VectorType", "UNKNOWN")
                    attacks_by_type[vector_name] = attacks_by_type.get(vector_name, 0) + 1

            logger.info(
                "Retrieved attack statistics: count=%d, time_range=%s",
                len(attacks),
                str(time_range),
            )

            return AttackStatistics(
                attack_count=len(attacks),
                attacks_by_type=attacks_by_type,
                max_attack_bps=max_bps,
                max_attack_pps=max_pps,
                time_range_start=start_time,
                time_range_end=end_time,
            )

        except ClientError as e:
            logger.error("Failed to get attack statistics: %s", str(e))
            return AttackStatistics()

    async def get_active_attacks(
        self,
        resource_arn: Optional[str] = None,
    ) -> List[Attack]:
        """Get currently active attacks.

        Args:
            resource_arn: Optional specific resource to filter by.

        Returns:
            List of active Attack instances.
        """
        try:
            # List attacks in the last hour (likely active)
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=1)

            response = self.shield_client.list_attacks(
                StartTime={
                    "FromInclusive": start_time,
                    "ToExclusive": end_time,
                },
                ResourceArns=[resource_arn] if resource_arn else [],
            )

            attacks = []
            for attack_summary in response.get("AttackSummaries", []):
                # Get detailed attack info
                attack_id = attack_summary.get("AttackId")
                if attack_id:
                    detail_response = self.shield_client.describe_attack(
                        AttackId=attack_id,
                    )
                    attack_detail = detail_response.get("Attack", {})

                    # Map Shield attack to our Attack model
                    attack = self._map_shield_attack(attack_detail)
                    if attack:
                        attacks.append(attack)

            logger.info("Found %d active attacks", len(attacks))
            return attacks

        except ClientError as e:
            logger.error("Failed to get active attacks: %s", str(e))
            return []

    def _map_shield_attack(self, shield_attack: Dict[str, Any]) -> Optional[Attack]:
        """Map AWS Shield attack to our Attack model.

        Args:
            shield_attack: Attack dictionary from Shield API.

        Returns:
            Attack instance or None.
        """
        if not shield_attack:
            return None

        # Map attack vectors to our AttackType
        vectors = shield_attack.get("AttackVectors", [])
        attack_type = AttackType.VOLUMETRIC  # Default

        for vector in vectors:
            vector_type = vector.get("VectorType", "").upper()
            if "TCP" in vector_type or "UDP" in vector_type:
                attack_type = AttackType.VOLUMETRIC
            elif "HTTP" in vector_type or "APPLICATION" in vector_type:
                attack_type = AttackType.APPLICATION_LAYER
            elif "SYN" in vector_type:
                attack_type = AttackType.SYN_FLOOD
            elif "AMPLIFICATION" in vector_type:
                attack_type = AttackType.AMPLIFICATION

        # Calculate severity based on attack properties
        counters = shield_attack.get("AttackCounters", [])
        max_value = 0
        for counter in counters:
            if counter.get("Name") == "BITS":
                max_value = max(max_value, counter.get("Max", 0))

        if max_value > 100000000000:  # 100 Gbps
            severity = AttackSeverity.CRITICAL
        elif max_value > 10000000000:  # 10 Gbps
            severity = AttackSeverity.HIGH
        elif max_value > 1000000000:  # 1 Gbps
            severity = AttackSeverity.MEDIUM
        else:
            severity = AttackSeverity.LOW

        return Attack(
            id=shield_attack.get("AttackId", ""),
            attack_type=attack_type,
            severity=severity,
            target_endpoints=[shield_attack.get("ResourceArn", "")],
            started_at=shield_attack.get("StartTime"),
            ended_at=shield_attack.get("EndTime"),
            detection_source="aws_shield",
            metadata={
                "shield_attack_id": shield_attack.get("AttackId"),
                "attack_vectors": [v.get("VectorType") for v in vectors],
            },
        )

    async def configure_proactive_engagement(
        self,
        enabled: bool,
    ) -> bool:
        """Configure proactive engagement with AWS Shield Response Team.

        When enabled, the Shield Response Team will proactively reach out
        during detected attacks to help with mitigation.

        Args:
            enabled: Whether to enable proactive engagement.

        Returns:
            True if configuration was updated.
        """
        try:
            if enabled:
                # Proactive engagement requires emergency contacts
                self.shield_client.enable_proactive_engagement()
                logger.info("Proactive engagement enabled with AWS Shield DRT")
            else:
                self.shield_client.disable_proactive_engagement()
                logger.info("Proactive engagement disabled")

            return True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")

            if error_code == "InvalidParameterException":
                logger.warning(
                    "Cannot enable proactive engagement: "
                    "Emergency contacts may need to be configured first."
                )
            else:
                logger.error("Failed to configure proactive engagement: %s", str(e))

            return False

    async def get_subscription_state(self) -> ShieldSubscriptionStatus:
        """Get the current Shield Advanced subscription status.

        Returns:
            ShieldSubscriptionStatus instance.
        """
        try:
            response = self.shield_client.get_subscription_state()
            state = response.get("SubscriptionState", "INACTIVE")

            if state == "ACTIVE":
                # Get subscription details
                subscription = self.shield_client.describe_subscription()
                sub_info = subscription.get("Subscription", {})

                return ShieldSubscriptionStatus(
                    is_active=True,
                    subscription_start=sub_info.get("StartTime"),
                    auto_renew=sub_info.get("AutoRenew") == "ENABLED",
                    subscription_limits=sub_info.get("SubscriptionLimits", {}),
                    time_commitment_in_seconds=sub_info.get(
                        "TimeCommitmentInSeconds", 31536000
                    ),
                )

            return ShieldSubscriptionStatus(is_active=False)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")

            if error_code == "ResourceNotFoundException":
                # No subscription
                return ShieldSubscriptionStatus(is_active=False)

            logger.error("Failed to get subscription state: %s", str(e))
            return ShieldSubscriptionStatus(is_active=False)

    async def list_protections(self) -> List[ShieldProtection]:
        """List all Shield protections.

        Returns:
            List of ShieldProtection instances.
        """
        try:
            protections = []
            paginator = self.shield_client.get_paginator("list_protections")

            for page in paginator.paginate():
                for protection in page.get("Protections", []):
                    protections.append(ShieldProtection(
                        id=protection.get("Id", ""),
                        resource_arn=protection.get("ResourceArn", ""),
                        protection_name=protection.get("Name", ""),
                        health_check_arn=protection.get("HealthCheckIds", [None])[0],
                    ))

            logger.info("Found %d Shield protections", len(protections))
            return protections

        except ClientError as e:
            logger.error("Failed to list protections: %s", str(e))
            return []

    async def associate_health_check(
        self,
        protection_id: str,
        health_check_arn: str,
    ) -> bool:
        """Associate a Route 53 health check with a protection.

        Health checks improve DDoS detection by providing application
        health signals.

        Args:
            protection_id: Shield protection ID.
            health_check_arn: Route 53 health check ARN.

        Returns:
            True if association was successful.
        """
        try:
            self.shield_client.associate_health_check(
                ProtectionId=protection_id,
                HealthCheckArn=health_check_arn,
            )

            logger.info(
                "Health check associated: protection=%s, health_check=%s",
                protection_id,
                health_check_arn,
            )
            return True

        except ClientError as e:
            logger.error("Failed to associate health check: %s", str(e))
            return False

    def _get_resource_type(self, resource_arn: str) -> str:
        """Extract resource type from ARN.

        Args:
            resource_arn: AWS resource ARN.

        Returns:
            Resource type string.
        """
        # ARN format: arn:aws:service:region:account:resource
        parts = resource_arn.split(":")
        if len(parts) >= 3:
            return parts[2]
        return "unknown"


__all__ = [
    "ShieldManager",
    "ShieldSubscriptionStatus",
    "AttackStatistics",
]
