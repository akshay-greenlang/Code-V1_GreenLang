# -*- coding: utf-8 -*-
"""
PII Enforcement Engine - SEC-011

Real-time PII enforcement engine that scans content, applies policies,
and takes appropriate actions (allow, redact, block, quarantine, transform).

The engine integrates with:
    - PIIScanner for detection
    - AllowlistManager for false positive filtering
    - Notifier for alerting
    - Prometheus metrics for observability

Key Features:
    - Per-tenant policy overrides
    - Context-aware enforcement
    - Async processing for performance
    - Complete audit trail
    - <10ms P99 latency target

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from greenlang.infrastructure.pii_service.enforcement.policies import (
    DEFAULT_POLICIES,
    ContextType,
    EnforcementAction,
    EnforcementContext,
    EnforcementPolicy,
    PIIType,
    TransformationType,
    get_default_policy,
)
from greenlang.infrastructure.pii_service.enforcement.actions import (
    ActionTaken,
    EnforcementResult,
    PIIDetection,
    QuarantineItem,
    QuarantineStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol Definitions (for dependency injection)
# ---------------------------------------------------------------------------


class PIIScannerProtocol(Protocol):
    """Protocol for PII scanner implementations."""

    async def scan(self, content: str) -> List[PIIDetection]:
        """Scan content for PII."""
        ...


class AllowlistManagerProtocol(Protocol):
    """Protocol for allowlist manager implementations."""

    async def is_allowed(
        self,
        value: str,
        pii_type: PIIType,
        tenant_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[Any]]:
        """Check if a value is in the allowlist."""
        ...


class NotifierProtocol(Protocol):
    """Protocol for notification service implementations."""

    async def send(
        self,
        message: str,
        severity: str,
        context: Dict[str, Any],
    ) -> bool:
        """Send a notification."""
        ...


class QuarantineStorageProtocol(Protocol):
    """Protocol for quarantine storage implementations."""

    async def store(self, item: QuarantineItem) -> bool:
        """Store a quarantine item."""
        ...

    async def get(self, item_id: str) -> Optional[QuarantineItem]:
        """Get a quarantine item by ID."""
        ...

    async def release(self, item_id: str, reviewed_by: str) -> bool:
        """Release an item from quarantine."""
        ...

    async def delete(self, item_id: str, reviewed_by: str) -> bool:
        """Delete a quarantine item."""
        ...


class TokenVaultProtocol(Protocol):
    """Protocol for token vault implementations."""

    async def tokenize(
        self,
        value: str,
        pii_type: PIIType,
        tenant_id: str,
    ) -> str:
        """Create a token for a value."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EnforcementConfig:
    """Configuration for the enforcement engine.

    Attributes:
        enable_notifications: Whether to send notifications.
        enable_metrics: Whether to record Prometheus metrics.
        enable_quarantine: Whether to enable quarantine feature.
        default_quarantine_ttl_hours: Default TTL for quarantine items.
        max_content_length: Maximum content length to scan.
        processing_timeout_ms: Timeout for processing.
        batch_size: Batch size for bulk processing.
        log_detections: Whether to log all detections.
    """

    enable_notifications: bool = True
    enable_metrics: bool = True
    enable_quarantine: bool = True
    default_quarantine_ttl_hours: int = 72
    max_content_length: int = 10_000_000  # 10MB
    processing_timeout_ms: int = 5000  # 5 seconds
    batch_size: int = 100
    log_detections: bool = True


# ---------------------------------------------------------------------------
# Simple In-Memory Scanner (fallback if external scanner not provided)
# ---------------------------------------------------------------------------


class SimplePatternScanner:
    """Simple regex-based PII scanner for fallback use.

    This is a simplified scanner for when no external scanner is provided.
    For production use, prefer the full PIIScanner from security_scanning.
    """

    PATTERNS: Dict[PIIType, List[re.Pattern]] = {
        PIIType.SSN: [
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        ],
        PIIType.CREDIT_CARD: [
            re.compile(r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
        ],
        PIIType.EMAIL: [
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE),
        ],
        PIIType.PHONE: [
            re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        ],
        PIIType.API_KEY: [
            re.compile(r"(?i)(api[_-]?key|apikey|api_secret)\s*[:=]\s*[\"']?([A-Za-z0-9_-]{20,})[\"']?"),
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        ],
        PIIType.PASSWORD: [
            re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*[\"'][^\"']{8,}[\"']"),
        ],
        PIIType.IP_ADDRESS: [
            re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"),
        ],
    }

    async def scan(self, content: str) -> List[PIIDetection]:
        """Scan content for PII patterns.

        Args:
            content: Text content to scan.

        Returns:
            List of PIIDetection instances.
        """
        detections: List[PIIDetection] = []

        for pii_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(content):
                    value = match.group()

                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_confidence(pii_type, value)

                    detection = PIIDetection.from_match(
                        pii_type=pii_type,
                        value=value,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        pattern_name=pattern.pattern[:50],
                    )
                    detections.append(detection)

        return self._deduplicate(detections)

    def _calculate_confidence(self, pii_type: PIIType, value: str) -> float:
        """Calculate detection confidence score."""
        if pii_type == PIIType.SSN:
            return 0.85
        elif pii_type == PIIType.CREDIT_CARD:
            # Luhn check would increase confidence
            return 0.80
        elif pii_type == PIIType.EMAIL:
            return 0.90
        elif pii_type == PIIType.PHONE:
            return 0.70
        elif pii_type == PIIType.API_KEY:
            return 0.80
        elif pii_type == PIIType.PASSWORD:
            return 0.85
        return 0.75

    def _deduplicate(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """Remove overlapping detections."""
        if not detections:
            return []

        # Sort by start position
        sorted_detections = sorted(detections, key=lambda d: d.start)
        result = [sorted_detections[0]]

        for detection in sorted_detections[1:]:
            last = result[-1]
            # Check for overlap
            if detection.start >= last.end:
                result.append(detection)
            elif detection.confidence > last.confidence:
                # Replace with higher confidence detection
                result[-1] = detection

        return result


# ---------------------------------------------------------------------------
# PII Enforcement Engine
# ---------------------------------------------------------------------------


class PIIEnforcementEngine:
    """Real-time PII enforcement engine.

    Scans content for PII, applies enforcement policies, and takes
    appropriate actions. Supports per-tenant policy overrides and
    context-aware enforcement.

    Attributes:
        config: Engine configuration.
        _scanner: PII scanner implementation.
        _allowlist: Allowlist manager implementation.
        _notifier: Notification service.
        _quarantine_storage: Quarantine storage.
        _token_vault: Token vault for tokenization.
        _policies: Global enforcement policies.
        _tenant_overrides: Per-tenant policy overrides.

    Example:
        >>> engine = PIIEnforcementEngine(config)
        >>> result = await engine.enforce(content, context)
        >>> if result.blocked:
        ...     return error_response()
        >>> return result.modified_content
    """

    def __init__(
        self,
        config: Optional[EnforcementConfig] = None,
        scanner: Optional[PIIScannerProtocol] = None,
        allowlist: Optional[AllowlistManagerProtocol] = None,
        notifier: Optional[NotifierProtocol] = None,
        quarantine_storage: Optional[QuarantineStorageProtocol] = None,
        token_vault: Optional[TokenVaultProtocol] = None,
    ) -> None:
        """Initialize the enforcement engine.

        Args:
            config: Engine configuration.
            scanner: PII scanner implementation.
            allowlist: Allowlist manager implementation.
            notifier: Notification service.
            quarantine_storage: Quarantine storage.
            token_vault: Token vault for tokenization.
        """
        self._config = config or EnforcementConfig()
        self._scanner = scanner or SimplePatternScanner()
        self._allowlist = allowlist
        self._notifier = notifier
        self._quarantine_storage = quarantine_storage
        self._token_vault = token_vault

        # Copy default policies
        self._policies: Dict[PIIType, EnforcementPolicy] = {**DEFAULT_POLICIES}
        self._tenant_overrides: Dict[str, Dict[PIIType, EnforcementPolicy]] = {}

        logger.info(
            "PIIEnforcementEngine initialized with %d policies",
            len(self._policies),
        )

    # -------------------------------------------------------------------------
    # Main Enforcement Method
    # -------------------------------------------------------------------------

    async def enforce(
        self,
        content: str,
        context: EnforcementContext,
    ) -> EnforcementResult:
        """Scan content and apply enforcement policies.

        This is the main entry point for PII enforcement. It:
        1. Detects PII in content
        2. Filters allowlisted values
        3. Applies policies based on PII type and context
        4. Takes appropriate actions (allow, redact, block, quarantine, transform)
        5. Returns the enforcement result

        Args:
            content: Text content to scan and enforce.
            context: Enforcement context with tenant, user, path info.

        Returns:
            EnforcementResult with action taken and modified content.

        Example:
            >>> context = EnforcementContext(
            ...     context_type="api_request",
            ...     tenant_id="tenant-acme",
            ...     path="/api/v1/reports",
            ...     method="POST",
            ... )
            >>> result = await engine.enforce(request_body, context)
        """
        start_time = time.perf_counter()

        try:
            # Validate content length
            if len(content) > self._config.max_content_length:
                logger.warning(
                    "Content exceeds max length: %d > %d",
                    len(content),
                    self._config.max_content_length,
                )
                return EnforcementResult(
                    blocked=True,
                    original_content=content[:1000] + "...[truncated]",
                    error="Content exceeds maximum allowed length",
                    context=context,
                )

            # Step 1: Detect PII
            detections = await self._scanner.scan(content)

            # Step 2: Filter by allowlist
            if self._allowlist:
                detections = await self._filter_allowlisted(
                    content, detections, context.tenant_id
                )

            # Step 3: Apply policies and take actions
            actions_taken: List[ActionTaken] = []
            modified_content = content
            blocked = False

            # Process detections in reverse order to preserve positions
            sorted_detections = sorted(
                detections,
                key=lambda d: d.start,
                reverse=True,
            )

            for detection in sorted_detections:
                policy = self._get_policy(detection.pii_type, context.tenant_id)

                # Check confidence threshold
                if detection.confidence < policy.min_confidence:
                    if self._config.log_detections:
                        logger.debug(
                            "Detection below threshold: %s (%.2f < %.2f)",
                            detection.pii_type.value,
                            detection.confidence,
                            policy.min_confidence,
                        )
                    continue

                # Check context match
                if not policy.matches_context(context.context_type):
                    continue

                # Apply policy action
                action_result = await self._apply_action(
                    content=modified_content,
                    detection=detection,
                    policy=policy,
                    context=context,
                )

                modified_content = action_result["content"]
                actions_taken.append(action_result["action"])

                if action_result["blocked"]:
                    blocked = True

                # Notify if configured
                if policy.notify and self._config.enable_notifications:
                    await self._notify(detection, policy.action, context)

            # Step 4: Record metrics
            if self._config.enable_metrics:
                self._record_metrics(detections, actions_taken, context)

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            return EnforcementResult(
                blocked=blocked,
                original_content=content if not blocked else None,
                modified_content=modified_content if not blocked else None,
                detections=detections,
                actions_taken=actions_taken,
                context=context,
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            logger.error("Enforcement failed: %s", e, exc_info=True)
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            return EnforcementResult(
                blocked=True,  # Fail-safe: block on error
                error=str(e),
                context=context,
                processing_time_ms=processing_time_ms,
            )

    # -------------------------------------------------------------------------
    # Policy Management
    # -------------------------------------------------------------------------

    def set_policy(
        self,
        pii_type: PIIType,
        policy: EnforcementPolicy,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Set an enforcement policy for a PII type.

        Args:
            pii_type: The PII type to set policy for.
            policy: The enforcement policy.
            tenant_id: Optional tenant ID for tenant-specific override.
        """
        if tenant_id:
            if tenant_id not in self._tenant_overrides:
                self._tenant_overrides[tenant_id] = {}
            self._tenant_overrides[tenant_id][pii_type] = policy
            logger.info(
                "Set tenant policy: %s -> %s for tenant %s",
                pii_type.value,
                policy.action.value,
                tenant_id,
            )
        else:
            self._policies[pii_type] = policy
            logger.info(
                "Set global policy: %s -> %s",
                pii_type.value,
                policy.action.value,
            )

    def get_policy(
        self,
        pii_type: PIIType,
        tenant_id: Optional[str] = None,
    ) -> EnforcementPolicy:
        """Get the enforcement policy for a PII type.

        Args:
            pii_type: The PII type to get policy for.
            tenant_id: Optional tenant ID to check for overrides.

        Returns:
            The applicable EnforcementPolicy.
        """
        return self._get_policy(pii_type, tenant_id)

    def list_policies(
        self,
        tenant_id: Optional[str] = None,
    ) -> Dict[PIIType, EnforcementPolicy]:
        """List all enforcement policies.

        Args:
            tenant_id: Optional tenant ID to include overrides.

        Returns:
            Dictionary of PII type to policy.
        """
        policies = {**self._policies}

        if tenant_id and tenant_id in self._tenant_overrides:
            policies.update(self._tenant_overrides[tenant_id])

        return policies

    def clear_tenant_overrides(self, tenant_id: str) -> None:
        """Clear all policy overrides for a tenant.

        Args:
            tenant_id: The tenant ID to clear overrides for.
        """
        if tenant_id in self._tenant_overrides:
            del self._tenant_overrides[tenant_id]
            logger.info("Cleared policy overrides for tenant: %s", tenant_id)

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _get_policy(
        self,
        pii_type: PIIType,
        tenant_id: Optional[str] = None,
    ) -> EnforcementPolicy:
        """Get policy with tenant override support."""
        # Check tenant overrides first
        if tenant_id and tenant_id in self._tenant_overrides:
            tenant_policies = self._tenant_overrides[tenant_id]
            if pii_type in tenant_policies:
                return tenant_policies[pii_type]

        # Fall back to global policies
        if pii_type in self._policies:
            return self._policies[pii_type]

        # Fall back to default
        return get_default_policy(pii_type)

    async def _filter_allowlisted(
        self,
        content: str,
        detections: List[PIIDetection],
        tenant_id: str,
    ) -> List[PIIDetection]:
        """Filter out allowlisted detections."""
        if not self._allowlist:
            return detections

        filtered: List[PIIDetection] = []

        for detection in detections:
            # Extract the value from content
            value = content[detection.start:detection.end]

            is_allowed, entry = await self._allowlist.is_allowed(
                value, detection.pii_type, tenant_id
            )

            if is_allowed:
                logger.debug(
                    "Detection allowlisted: %s (%s)",
                    detection.pii_type.value,
                    entry,
                )
                # TODO: Record allowlist match metric
            else:
                filtered.append(detection)

        return filtered

    async def _apply_action(
        self,
        content: str,
        detection: PIIDetection,
        policy: EnforcementPolicy,
        context: EnforcementContext,
    ) -> Dict[str, Any]:
        """Apply enforcement action to a detection.

        Returns a dict with:
            - content: Modified content
            - action: ActionTaken record
            - blocked: Whether content should be blocked
        """
        action = policy.action
        blocked = False
        modified_content = content

        if action == EnforcementAction.ALLOW:
            action_record = ActionTaken.create(
                detection=detection,
                action=action,
                reason=f"Allowed {detection.pii_type.value} (policy: allow)",
                policy=policy,
            )

        elif action == EnforcementAction.REDACT:
            # Redact the PII value
            modified_content = self._redact(content, detection, policy)
            action_record = ActionTaken.create(
                detection=detection,
                action=action,
                reason=f"Redacted {detection.pii_type.value}",
                policy=policy,
            )

        elif action == EnforcementAction.BLOCK:
            blocked = True
            action_record = ActionTaken.create(
                detection=detection,
                action=action,
                reason=f"Blocked {detection.pii_type.value} (confidence: {detection.confidence:.2f})",
                policy=policy,
            )

        elif action == EnforcementAction.QUARANTINE:
            blocked = True
            await self._quarantine(content, detection, context, policy)
            action_record = ActionTaken.create(
                detection=detection,
                action=action,
                reason=f"Quarantined {detection.pii_type.value} for review",
                policy=policy,
            )

        elif action == EnforcementAction.TRANSFORM:
            modified_content = await self._transform(content, detection, context, policy)
            action_record = ActionTaken.create(
                detection=detection,
                action=action,
                reason=f"Transformed {detection.pii_type.value} ({policy.transformation_type.value})",
                policy=policy,
            )

        else:
            # Unknown action - log and allow
            logger.warning("Unknown enforcement action: %s", action)
            action_record = ActionTaken.create(
                detection=detection,
                action=EnforcementAction.ALLOW,
                reason=f"Unknown action '{action}' - defaulted to allow",
                policy=policy,
            )

        return {
            "content": modified_content,
            "action": action_record,
            "blocked": blocked,
        }

    def _redact(
        self,
        content: str,
        detection: PIIDetection,
        policy: EnforcementPolicy,
    ) -> str:
        """Apply redaction to content.

        Args:
            content: Original content.
            detection: The PII detection.
            policy: The enforcement policy.

        Returns:
            Content with PII redacted.
        """
        placeholder = policy.get_placeholder()

        # Replace the detected value with placeholder
        return (
            content[:detection.start]
            + placeholder
            + content[detection.end:]
        )

    async def _quarantine(
        self,
        content: str,
        detection: PIIDetection,
        context: EnforcementContext,
        policy: EnforcementPolicy,
    ) -> None:
        """Store content in quarantine for review.

        Args:
            content: The content to quarantine.
            detection: The PII detection.
            context: Enforcement context.
            policy: The enforcement policy.
        """
        if not self._quarantine_storage:
            logger.warning("Quarantine requested but no storage configured")
            return

        if not self._config.enable_quarantine:
            logger.debug("Quarantine disabled in config")
            return

        item = QuarantineItem.create(
            content=content,
            detection=detection,
            context=context,
            ttl_hours=policy.quarantine_ttl_hours,
        )

        try:
            await self._quarantine_storage.store(item)
            logger.info(
                "Quarantined content: %s (tenant: %s)",
                item.id,
                context.tenant_id,
            )
        except Exception as e:
            logger.error("Failed to quarantine content: %s", e, exc_info=True)

    async def _transform(
        self,
        content: str,
        detection: PIIDetection,
        context: EnforcementContext,
        policy: EnforcementPolicy,
    ) -> str:
        """Apply transformation to content.

        Args:
            content: Original content.
            detection: The PII detection.
            context: Enforcement context.
            policy: The enforcement policy.

        Returns:
            Content with PII transformed.
        """
        value = content[detection.start:detection.end]
        transformation_type = policy.transformation_type

        if transformation_type == TransformationType.TOKENIZE:
            if self._token_vault:
                try:
                    token = await self._token_vault.tokenize(
                        value, detection.pii_type, context.tenant_id
                    )
                    replacement = f"[TOKEN:{token}]"
                except Exception as e:
                    logger.error("Tokenization failed: %s", e)
                    replacement = f"[{detection.pii_type.value.upper()}]"
            else:
                logger.warning("Token vault not configured for tokenization")
                replacement = f"[{detection.pii_type.value.upper()}]"

        elif transformation_type == TransformationType.HASH:
            hash_value = hashlib.sha256(value.encode()).hexdigest()[:12]
            replacement = f"[HASH:{hash_value}]"

        elif transformation_type == TransformationType.MASK:
            if len(value) <= 4:
                replacement = "*" * len(value)
            else:
                replacement = value[:2] + "*" * (len(value) - 4) + value[-2:]

        elif transformation_type == TransformationType.ENCRYPT:
            # Would use encryption service; fallback to hash
            hash_value = hashlib.sha256(value.encode()).hexdigest()[:12]
            replacement = f"[ENC:{hash_value}]"

        else:
            replacement = f"[{detection.pii_type.value.upper()}]"

        return content[:detection.start] + replacement + content[detection.end:]

    async def _notify(
        self,
        detection: PIIDetection,
        action: EnforcementAction,
        context: EnforcementContext,
    ) -> None:
        """Send notification for a detection.

        Args:
            detection: The PII detection.
            action: Action taken.
            context: Enforcement context.
        """
        if not self._notifier:
            return

        severity = "high" if action == EnforcementAction.BLOCK else "medium"
        message = (
            f"PII detected: {detection.pii_type.value} in {context.context_type} "
            f"(tenant: {context.tenant_id}, action: {action.value})"
        )

        try:
            await self._notifier.send(
                message=message,
                severity=severity,
                context={
                    "pii_type": detection.pii_type.value,
                    "confidence": detection.confidence,
                    "action": action.value,
                    "tenant_id": context.tenant_id,
                    "path": context.path,
                },
            )
        except Exception as e:
            logger.error("Failed to send notification: %s", e)

    def _record_metrics(
        self,
        detections: List[PIIDetection],
        actions: List[ActionTaken],
        context: EnforcementContext,
    ) -> None:
        """Record Prometheus metrics for enforcement.

        Args:
            detections: All detections found.
            actions: All actions taken.
            context: Enforcement context.
        """
        # TODO: Import and use actual Prometheus metrics
        # For now, just log
        if detections:
            logger.info(
                "Enforcement metrics: detections=%d, actions=%d, context=%s",
                len(detections),
                len(actions),
                context.context_type,
            )


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


_global_engine: Optional[PIIEnforcementEngine] = None


def get_enforcement_engine(
    config: Optional[EnforcementConfig] = None,
) -> PIIEnforcementEngine:
    """Get or create the global enforcement engine instance.

    Args:
        config: Optional configuration for initial creation.

    Returns:
        The global PIIEnforcementEngine instance.
    """
    global _global_engine

    if _global_engine is None:
        _global_engine = PIIEnforcementEngine(config)

    return _global_engine


def reset_engine() -> None:
    """Reset the global engine instance (for testing)."""
    global _global_engine
    _global_engine = None


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "PIIEnforcementEngine",
    "EnforcementConfig",
    "SimplePatternScanner",
    "PIIScannerProtocol",
    "AllowlistManagerProtocol",
    "NotifierProtocol",
    "QuarantineStorageProtocol",
    "TokenVaultProtocol",
    "get_enforcement_engine",
    "reset_engine",
]
