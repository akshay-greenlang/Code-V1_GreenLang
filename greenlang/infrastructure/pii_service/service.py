# -*- coding: utf-8 -*-
"""
Unified PII Service - SEC-011: PII Detection/Redaction Enhancements

Provides a unified facade over all PII detection, redaction, tokenization,
and enforcement capabilities in the GreenLang platform.

Components Integrated:
    - PIIScanner (SEC-007): Regex-based pattern detection
    - PresidioPIIScanner (SEC-007): ML-based NER detection
    - SecureTokenVault (SEC-011): AES-256-GCM token vault
    - AllowlistManager (SEC-011): False positive filtering
    - PIIEnforcementEngine (SEC-011): Real-time blocking/redaction

API:
    - detect(): Detect PII in content
    - redact(): Detect and redact PII
    - enforce(): Apply enforcement policies
    - tokenize(): Create reversible token
    - detokenize(): Retrieve original value

Example:
    >>> config = PIIServiceConfig()
    >>> encryption_svc = await EncryptionService.create(enc_config)
    >>> audit_svc = get_audit_service()
    >>> pii_service = await PIIService.create(config, encryption_svc, audit_svc)
    >>>
    >>> # Detect PII
    >>> detections = await pii_service.detect("SSN: 123-45-6789")
    >>>
    >>> # Redact PII
    >>> result = await pii_service.redact("Email: john@example.com")
    >>> print(result.redacted_content)  # "Email: [EMAIL]"
    >>>
    >>> # Tokenize for reversible redaction
    >>> token = await pii_service.tokenize("123-45-6789", PIIType.SSN, "tenant-1")
    >>> original = await pii_service.detokenize(token, "tenant-1", "user-1")

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from greenlang.infrastructure.pii_service.config import (
    PIIServiceConfig,
    EnforcementMode,
)
from greenlang.infrastructure.pii_service.models import (
    PIIType,
    RedactionStrategy,
    PIIDetection,
    DetectionOptions,
    RedactionResult,
    RedactionOptions,
    EnforcementContext,
    EnforcementResult,
    ActionTaken,
    EnforcementAction,
    DetectionMethod,
    AllowlistEntry,
)
from greenlang.infrastructure.pii_service.secure_vault import SecureTokenVault

if TYPE_CHECKING:
    from greenlang.infrastructure.encryption_service import EncryptionService
    from greenlang.infrastructure.audit_service import AuditService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics (lazy initialization)
# ---------------------------------------------------------------------------

_metrics_initialized = False


def _init_metrics() -> None:
    """Initialize Prometheus metrics lazily."""
    global _metrics_initialized
    if _metrics_initialized:
        return

    try:
        from prometheus_client import Counter, Histogram

        global pii_detections_total, pii_detection_latency
        global pii_enforcement_actions_total, pii_blocked_requests_total
        global pii_allowlist_matches_total

        pii_detections_total = Counter(
            "gl_pii_detections_total",
            "Total PII detections",
            ["pii_type", "source", "confidence_level"],
        )

        pii_detection_latency = Histogram(
            "gl_pii_detection_latency_seconds",
            "PII detection latency",
            ["scanner_type"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        pii_enforcement_actions_total = Counter(
            "gl_pii_enforcement_actions_total",
            "Total enforcement actions",
            ["action", "pii_type", "context"],
        )

        pii_blocked_requests_total = Counter(
            "gl_pii_blocked_requests_total",
            "Total blocked requests",
            ["pii_type", "endpoint"],
        )

        pii_allowlist_matches_total = Counter(
            "gl_pii_allowlist_matches_total",
            "Total allowlist matches",
            ["pii_type", "pattern"],
        )

        _metrics_initialized = True
        logger.debug("PII service metrics initialized")

    except ImportError:
        logger.info("prometheus_client not available; metrics disabled")
        _metrics_initialized = True


# ---------------------------------------------------------------------------
# Default Enforcement Policies
# ---------------------------------------------------------------------------

DEFAULT_ENFORCEMENT_POLICIES: Dict[PIIType, EnforcementAction] = {
    # High-sensitivity: Block by default
    PIIType.SSN: EnforcementAction.BLOCK,
    PIIType.CREDIT_CARD: EnforcementAction.BLOCK,
    PIIType.PASSWORD: EnforcementAction.BLOCK,
    PIIType.API_KEY: EnforcementAction.REDACT,
    PIIType.BANK_ACCOUNT: EnforcementAction.BLOCK,
    # Medium-sensitivity: Redact
    PIIType.EMAIL: EnforcementAction.REDACT,
    PIIType.PHONE: EnforcementAction.REDACT,
    PIIType.ADDRESS: EnforcementAction.REDACT,
    PIIType.DATE_OF_BIRTH: EnforcementAction.REDACT,
    PIIType.DRIVERS_LICENSE: EnforcementAction.BLOCK,
    PIIType.PASSPORT: EnforcementAction.BLOCK,
    PIIType.MEDICAL_RECORD: EnforcementAction.BLOCK,
    # Lower-sensitivity: Allow with logging
    PIIType.PERSON_NAME: EnforcementAction.ALLOW,
    PIIType.IP_ADDRESS: EnforcementAction.ALLOW,
    PIIType.ORGANIZATION_NAME: EnforcementAction.ALLOW,
}


# ---------------------------------------------------------------------------
# Default Allowlist Patterns
# ---------------------------------------------------------------------------

DEFAULT_ALLOWLIST_ENTRIES: List[Dict[str, Any]] = [
    # Test email domains
    {
        "pii_type": PIIType.EMAIL,
        "pattern": r".*@example\.(com|org|net)$",
        "pattern_type": "regex",
        "reason": "RFC 2606 reserved domain",
    },
    {
        "pii_type": PIIType.EMAIL,
        "pattern": r".*@test\.(com|org|net)$",
        "pattern_type": "regex",
        "reason": "Test domain",
    },
    {
        "pii_type": PIIType.EMAIL,
        "pattern": r"noreply@.*",
        "pattern_type": "regex",
        "reason": "No-reply addresses",
    },
    {
        "pii_type": PIIType.EMAIL,
        "pattern": r".*@localhost$",
        "pattern_type": "regex",
        "reason": "Localhost domain",
    },
    # Test phone numbers
    {
        "pii_type": PIIType.PHONE,
        "pattern": r".*555-\d{4}$",
        "pattern_type": "regex",
        "reason": "US fictional phone numbers (555)",
    },
    # Test credit cards
    {
        "pii_type": PIIType.CREDIT_CARD,
        "pattern": "4111111111111111",
        "pattern_type": "exact",
        "reason": "Stripe test card (Visa)",
    },
    {
        "pii_type": PIIType.CREDIT_CARD,
        "pattern": "4242424242424242",
        "pattern_type": "exact",
        "reason": "Stripe test card (Visa)",
    },
    {
        "pii_type": PIIType.CREDIT_CARD,
        "pattern": "5555555555554444",
        "pattern_type": "exact",
        "reason": "Stripe test card (Mastercard)",
    },
    # Invalid SSN placeholders
    {
        "pii_type": PIIType.SSN,
        "pattern": "000-00-0000",
        "pattern_type": "exact",
        "reason": "Invalid SSN (all zeros)",
    },
    {
        "pii_type": PIIType.SSN,
        "pattern": "123-45-6789",
        "pattern_type": "exact",
        "reason": "Invalid SSN (sequential)",
    },
]


# ---------------------------------------------------------------------------
# Allowlist Manager
# ---------------------------------------------------------------------------


class AllowlistManager:
    """Manages PII detection allowlists.

    Filters false positives from detection results based on
    configurable patterns (regex, exact, prefix, suffix, contains).
    """

    def __init__(
        self,
        enable_defaults: bool = True,
        custom_entries: Optional[List[AllowlistEntry]] = None,
    ) -> None:
        """Initialize AllowlistManager.

        Args:
            enable_defaults: Include default allowlist patterns.
            custom_entries: Custom allowlist entries.
        """
        self._entries: Dict[PIIType, List[AllowlistEntry]] = {}
        self._compiled_patterns: Dict[str, re.Pattern] = {}

        # Load defaults
        if enable_defaults:
            from uuid import uuid4

            for entry_data in DEFAULT_ALLOWLIST_ENTRIES:
                entry = AllowlistEntry(
                    pii_type=entry_data["pii_type"],
                    pattern=entry_data["pattern"],
                    pattern_type=entry_data["pattern_type"],
                    reason=entry_data["reason"],
                    created_by=uuid4(),  # System-created
                )
                self._add_entry(entry)

        # Load custom entries
        if custom_entries:
            for entry in custom_entries:
                self._add_entry(entry)

        logger.info("AllowlistManager initialized with %d entries", self._count_entries())

    def _add_entry(self, entry: AllowlistEntry) -> None:
        """Add entry to internal storage."""
        if entry.pii_type not in self._entries:
            self._entries[entry.pii_type] = []
        self._entries[entry.pii_type].append(entry)

        # Pre-compile regex patterns
        if entry.pattern_type == "regex":
            try:
                self._compiled_patterns[entry.pattern] = re.compile(
                    entry.pattern, re.IGNORECASE
                )
            except re.error as e:
                logger.error("Invalid regex pattern '%s': %s", entry.pattern, e)

    def _count_entries(self) -> int:
        """Count total entries across all PII types."""
        return sum(len(entries) for entries in self._entries.values())

    async def is_allowed(
        self,
        value: str,
        pii_type: PIIType,
        tenant_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[AllowlistEntry]]:
        """Check if a value is in the allowlist.

        Args:
            value: Detected PII value.
            pii_type: Type of PII.
            tenant_id: Optional tenant for tenant-specific allowlists.

        Returns:
            Tuple of (is_allowed, matching_entry).
        """
        entries = self._entries.get(pii_type, [])

        for entry in entries:
            # Skip disabled or expired entries
            if not entry.enabled or entry.is_expired:
                continue

            # Check tenant scope
            if entry.tenant_id and entry.tenant_id != tenant_id:
                continue

            if self._matches(value, entry):
                return True, entry

        return False, None

    def _matches(self, value: str, entry: AllowlistEntry) -> bool:
        """Check if value matches allowlist entry pattern.

        Args:
            value: Value to check.
            entry: Allowlist entry.

        Returns:
            True if value matches.
        """
        if entry.pattern_type == "exact":
            return value == entry.pattern
        elif entry.pattern_type == "regex":
            pattern = self._compiled_patterns.get(entry.pattern)
            if pattern:
                return bool(pattern.match(value))
            return bool(re.match(entry.pattern, value, re.IGNORECASE))
        elif entry.pattern_type == "prefix":
            return value.startswith(entry.pattern)
        elif entry.pattern_type == "suffix":
            return value.endswith(entry.pattern)
        elif entry.pattern_type == "contains":
            return entry.pattern in value
        return False


# ---------------------------------------------------------------------------
# PII Enforcement Engine
# ---------------------------------------------------------------------------


class PIIEnforcementEngine:
    """Real-time PII enforcement engine.

    Applies policies to detected PII: allow, redact, block, or quarantine.
    """

    def __init__(
        self,
        mode: EnforcementMode = EnforcementMode.ENFORCE,
        policies: Optional[Dict[PIIType, EnforcementAction]] = None,
        min_confidence: float = 0.8,
    ) -> None:
        """Initialize PIIEnforcementEngine.

        Args:
            mode: Operating mode.
            policies: Per-PII-type enforcement policies.
            min_confidence: Minimum confidence for enforcement.
        """
        self._mode = mode
        self._policies = policies or DEFAULT_ENFORCEMENT_POLICIES.copy()
        self._min_confidence = min_confidence

        logger.info(
            "PIIEnforcementEngine initialized: mode=%s policies=%d",
            mode.value,
            len(self._policies),
        )

    def get_policy(self, pii_type: PIIType) -> EnforcementAction:
        """Get policy for a PII type."""
        return self._policies.get(pii_type, EnforcementAction.ALLOW)

    def set_policy(self, pii_type: PIIType, action: EnforcementAction) -> None:
        """Set policy for a PII type."""
        self._policies[pii_type] = action

    async def enforce(
        self,
        content: str,
        detections: List[PIIDetection],
        context: EnforcementContext,
    ) -> EnforcementResult:
        """Apply enforcement policies to detected PII.

        Args:
            content: Original content.
            detections: List of PII detections.
            context: Enforcement context.

        Returns:
            EnforcementResult with actions taken.
        """
        # Disabled mode: do nothing
        if self._mode == EnforcementMode.DISABLED:
            return EnforcementResult(
                blocked=False,
                original_content=content,
                modified_content=content,
                detections=detections,
                context=context,
            )

        actions_taken: List[ActionTaken] = []
        blocked = False
        modified_content = content

        # Process detections in reverse order to preserve positions
        sorted_detections = sorted(detections, key=lambda d: d.start, reverse=True)

        for detection in sorted_detections:
            # Skip low-confidence detections
            if detection.confidence < self._min_confidence:
                continue

            policy = self.get_policy(detection.pii_type)

            # Audit mode: just log
            if self._mode == EnforcementMode.AUDIT:
                actions_taken.append(
                    ActionTaken(
                        detection=detection,
                        action=EnforcementAction.ALLOW,
                        reason=f"Audit mode: would {policy.value}",
                    )
                )
                continue

            # Warn mode: log but allow
            if self._mode == EnforcementMode.WARN:
                logger.warning(
                    "PII detected: type=%s confidence=%.2f action=%s (warn mode)",
                    detection.pii_type.value,
                    detection.confidence,
                    policy.value,
                )
                actions_taken.append(
                    ActionTaken(
                        detection=detection,
                        action=EnforcementAction.ALLOW,
                        reason=f"Warn mode: would {policy.value}",
                    )
                )
                continue

            # Enforce mode: apply policy
            if policy == EnforcementAction.BLOCK:
                blocked = True
                actions_taken.append(
                    ActionTaken(
                        detection=detection,
                        action=EnforcementAction.BLOCK,
                        reason=f"Policy: block {detection.pii_type.value}",
                    )
                )

            elif policy == EnforcementAction.REDACT:
                # Apply redaction
                placeholder = f"[{detection.pii_type.value.upper()}]"
                modified_content = (
                    modified_content[: detection.start]
                    + placeholder
                    + modified_content[detection.end :]
                )
                actions_taken.append(
                    ActionTaken(
                        detection=detection,
                        action=EnforcementAction.REDACT,
                        reason=f"Redacted {detection.pii_type.value}",
                    )
                )

            elif policy == EnforcementAction.QUARANTINE:
                blocked = True
                actions_taken.append(
                    ActionTaken(
                        detection=detection,
                        action=EnforcementAction.QUARANTINE,
                        reason="Quarantined for review",
                    )
                )

            else:  # ALLOW
                actions_taken.append(
                    ActionTaken(
                        detection=detection,
                        action=EnforcementAction.ALLOW,
                        reason="Policy allows",
                    )
                )

            # Record metrics
            if _metrics_initialized:
                try:
                    pii_enforcement_actions_total.labels(
                        action=policy.value,
                        pii_type=detection.pii_type.value,
                        context=context.context_type,
                    ).inc()
                except Exception:
                    pass

        return EnforcementResult(
            blocked=blocked,
            original_content=content,
            modified_content=modified_content if not blocked else None,
            detections=detections,
            actions_taken=actions_taken,
            context=context,
        )


# ---------------------------------------------------------------------------
# Unified PII Service
# ---------------------------------------------------------------------------


class PIIService:
    """Unified PII detection, redaction, and management service.

    Provides a single API layer over all PII capabilities in GreenLang:
    - Detection (regex + ML)
    - Redaction (multiple strategies)
    - Tokenization (reversible encryption)
    - Enforcement (blocking/quarantine)
    - Allowlist management

    This class should be instantiated via the async ``create()`` factory
    method to ensure proper initialization of async dependencies.
    """

    def __init__(
        self,
        config: PIIServiceConfig,
        vault: SecureTokenVault,
        enforcement: PIIEnforcementEngine,
        allowlist: AllowlistManager,
        regex_scanner: Optional[Any] = None,
        ml_scanner: Optional[Any] = None,
        audit_service: Optional[AuditService] = None,
    ) -> None:
        """Initialize PIIService.

        Use ``PIIService.create()`` for proper async initialization.

        Args:
            config: Service configuration.
            vault: Secure token vault.
            enforcement: Enforcement engine.
            allowlist: Allowlist manager.
            regex_scanner: PIIScanner instance.
            ml_scanner: PresidioPIIScanner instance.
            audit_service: Audit service for logging.
        """
        self._config = config
        self._vault = vault
        self._enforcement = enforcement
        self._allowlist = allowlist
        self._regex_scanner = regex_scanner
        self._ml_scanner = ml_scanner
        self._audit = audit_service

        _init_metrics()

        logger.info(
            "PIIService initialized: regex=%s ml=%s enforcement=%s",
            regex_scanner is not None,
            ml_scanner is not None,
            config.enforcement.mode.value,
        )

    @classmethod
    async def create(
        cls,
        config: PIIServiceConfig,
        encryption_service: EncryptionService,
        audit_service: Optional[AuditService] = None,
        db_pool: Optional[Any] = None,
        redis_client: Optional[Any] = None,
    ) -> PIIService:
        """Factory method to create a PIIService instance.

        This is the recommended way to instantiate the service as it
        properly initializes all dependencies.

        Args:
            config: Service configuration.
            encryption_service: SEC-003 encryption service.
            audit_service: Optional audit service.
            db_pool: Optional database pool.
            redis_client: Optional Redis client.

        Returns:
            Fully initialized PIIService.
        """
        # Create vault
        vault = SecureTokenVault(
            encryption_service=encryption_service,
            config=config.vault,
            db_pool=db_pool,
            redis_client=redis_client,
        )

        # Create enforcement engine
        enforcement = PIIEnforcementEngine(
            mode=config.enforcement.mode,
            min_confidence=config.enforcement.min_confidence,
        )

        # Create allowlist manager
        allowlist = AllowlistManager(
            enable_defaults=config.allowlist.enable_defaults,
        )

        # Create scanners
        regex_scanner = None
        ml_scanner = None

        if config.scanner.enable_regex:
            try:
                from greenlang.infrastructure.security_scanning.pii_scanner import (
                    PIIScanner,
                    get_pii_scanner,
                )

                regex_scanner = get_pii_scanner()
                logger.info("Regex PII scanner enabled")
            except ImportError:
                logger.warning("PIIScanner not available; regex detection disabled")

        if config.scanner.enable_ml:
            try:
                from greenlang.infrastructure.security_scanning.pii_ml import (
                    PresidioPIIScanner,
                    get_presidio_scanner,
                )

                ml_scanner = get_presidio_scanner()
                if ml_scanner and ml_scanner.is_available:
                    logger.info("ML PII scanner (Presidio) enabled")
                else:
                    logger.info("ML PII scanner not available")
                    ml_scanner = None
            except ImportError:
                logger.warning("PresidioPIIScanner not available; ML detection disabled")

        return cls(
            config=config,
            vault=vault,
            enforcement=enforcement,
            allowlist=allowlist,
            regex_scanner=regex_scanner,
            ml_scanner=ml_scanner,
            audit_service=audit_service,
        )

    async def detect(
        self,
        content: str,
        options: Optional[DetectionOptions] = None,
    ) -> List[PIIDetection]:
        """Detect PII in content using regex + optional ML.

        Args:
            content: Text content to scan.
            options: Detection options.

        Returns:
            List of PIIDetection objects.
        """
        options = options or DetectionOptions()
        start_time = datetime.utcnow()
        detections: List[PIIDetection] = []

        # Regex detection
        if self._regex_scanner is not None:
            try:
                regex_start = datetime.utcnow()
                regex_findings = self._regex_scanner.scan_text(content)
                regex_elapsed = (datetime.utcnow() - regex_start).total_seconds()

                # Convert to PIIDetection
                for finding in regex_findings:
                    # Map PIIType from scanner
                    try:
                        pii_type = PIIType(finding.pii_type.value)
                    except ValueError:
                        pii_type = PIIType.CUSTOM

                    detection = PIIDetection(
                        pii_type=pii_type,
                        value_hash=finding.matched_text_hash,
                        confidence=finding.confidence_score,
                        start=finding.column_start or 0,
                        end=finding.column_end or 0,
                        context=finding.context_before,
                        detection_method=DetectionMethod.REGEX,
                        pattern_name=finding.pattern_name,
                    )
                    detections.append(detection)

                if _metrics_initialized:
                    try:
                        pii_detection_latency.labels(scanner_type="regex").observe(
                            regex_elapsed
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.error("Regex detection failed: %s", e)

        # ML detection (if enabled)
        if options.use_ml and self._ml_scanner is not None:
            try:
                ml_start = datetime.utcnow()
                ml_entities = await self._ml_scanner.scan_async(content)
                ml_elapsed = (datetime.utcnow() - ml_start).total_seconds()

                # Convert to PIIDetection
                for entity in ml_entities:
                    # Map entity type to PIIType
                    pii_type = self._map_ml_entity_type(entity.entity_type)

                    detection = PIIDetection(
                        pii_type=pii_type,
                        value_hash=entity.text_hash,
                        confidence=entity.score,
                        start=entity.start,
                        end=entity.end,
                        context=entity.context_before,
                        detection_method=DetectionMethod.ML,
                        pattern_name=entity.recognizer,
                    )
                    detections.append(detection)

                if _metrics_initialized:
                    try:
                        pii_detection_latency.labels(scanner_type="ml").observe(
                            ml_elapsed
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.error("ML detection failed: %s", e)

        # Merge and deduplicate detections
        detections = self._merge_detections(detections)

        # Filter by allowlist
        if options.apply_allowlist:
            detections = await self._filter_allowlisted(
                content, detections, options.tenant_id
            )

        # Filter by confidence
        detections = [d for d in detections if d.confidence >= options.min_confidence]

        # Filter by PII types (if specified)
        if options.pii_types:
            detections = [d for d in detections if d.pii_type in options.pii_types]

        # Limit results
        detections = detections[: options.max_detections]

        # Record metrics
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        for detection in detections:
            if _metrics_initialized:
                try:
                    pii_detections_total.labels(
                        pii_type=detection.pii_type.value,
                        source=options.source,
                        confidence_level=detection.confidence_level.value,
                    ).inc()
                except Exception:
                    pass

        logger.debug(
            "Detection complete: found=%d elapsed=%.3fs",
            len(detections),
            elapsed,
        )

        return detections

    async def redact(
        self,
        content: str,
        options: Optional[RedactionOptions] = None,
    ) -> RedactionResult:
        """Detect and redact PII from content.

        Args:
            content: Text content to redact.
            options: Redaction options.

        Returns:
            RedactionResult with redacted content.
        """
        options = options or RedactionOptions()
        start_time = datetime.utcnow()

        # Detect PII
        detections = await self.detect(content, options.detection_options)

        # Apply redaction in reverse order to preserve positions
        redacted_content = content
        tokens_created: List[str] = []
        sorted_detections = sorted(detections, key=lambda d: d.start, reverse=True)

        for detection in sorted_detections:
            # Get strategy for this PII type
            strategy = options.strategy_overrides.get(
                detection.pii_type, options.strategy
            )

            # Apply redaction
            if strategy == RedactionStrategy.REPLACE:
                replacement = f"[{detection.pii_type.value.upper()}]"
            elif strategy == RedactionStrategy.MASK:
                length = detection.end - detection.start
                if length <= 2:
                    replacement = "*" * length
                else:
                    # Mask middle characters
                    original = content[detection.start : detection.end]
                    replacement = original[0] + "*" * (length - 2) + original[-1]
            elif strategy == RedactionStrategy.HASH:
                replacement = f"[HASH:{detection.value_hash[:12]}]"
            elif strategy == RedactionStrategy.REMOVE:
                replacement = ""
            elif strategy == RedactionStrategy.TOKENIZE:
                if options.create_tokens and options.tenant_id:
                    try:
                        original_value = content[detection.start : detection.end]
                        token = await self._vault.tokenize(
                            value=original_value,
                            pii_type=detection.pii_type,
                            tenant_id=options.tenant_id,
                        )
                        replacement = token
                        tokens_created.append(token)
                    except Exception as e:
                        logger.error("Tokenization failed: %s", e)
                        replacement = f"[{detection.pii_type.value.upper()}]"
                else:
                    replacement = f"[{detection.pii_type.value.upper()}]"
            elif strategy == RedactionStrategy.PARTIAL_MASK:
                original = content[detection.start : detection.end]
                if detection.pii_type == PIIType.CREDIT_CARD:
                    # Show last 4 digits
                    clean = re.sub(r"[-\s]", "", original)
                    replacement = "*" * (len(clean) - 4) + clean[-4:]
                elif detection.pii_type == PIIType.PHONE:
                    digits = re.sub(r"\D", "", original)
                    replacement = "***-***-" + digits[-4:]
                elif detection.pii_type == PIIType.EMAIL:
                    parts = original.split("@")
                    if len(parts) == 2:
                        replacement = parts[0][0] + "***@" + parts[1]
                    else:
                        replacement = original[0] + "***"
                else:
                    if len(original) <= 4:
                        replacement = "*" * len(original)
                    else:
                        replacement = original[:2] + "*" * (len(original) - 4) + original[-2:]
            else:
                replacement = f"[{detection.pii_type.value.upper()}]"

            redacted_content = (
                redacted_content[: detection.start]
                + replacement
                + redacted_content[detection.end :]
            )

        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000

        return RedactionResult(
            original_length=len(content),
            redacted_content=redacted_content,
            detections=detections,
            tokens_created=tokens_created,
            redaction_count=len(detections),
            processing_time_ms=elapsed,
        )

    async def enforce(
        self,
        content: str,
        context: EnforcementContext,
    ) -> EnforcementResult:
        """Apply enforcement policies to content.

        Args:
            content: Content to enforce.
            context: Enforcement context.

        Returns:
            EnforcementResult with actions taken.
        """
        start_time = datetime.utcnow()

        # Detect PII
        detection_options = DetectionOptions(
            tenant_id=context.tenant_id,
            source=context.context_type,
        )
        detections = await self.detect(content, detection_options)

        # Apply enforcement
        result = await self._enforcement.enforce(content, detections, context)

        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        result.processing_time_ms = elapsed

        # Record block metric
        if result.blocked and _metrics_initialized:
            for detection in detections:
                try:
                    pii_blocked_requests_total.labels(
                        pii_type=detection.pii_type.value,
                        endpoint=context.path or "unknown",
                    ).inc()
                except Exception:
                    pass

        return result

    async def tokenize(
        self,
        value: str,
        pii_type: PIIType,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create reversible token for a PII value.

        Args:
            value: PII value to tokenize.
            pii_type: Type of PII.
            tenant_id: Owning tenant.
            metadata: Optional metadata.

        Returns:
            Token string.
        """
        return await self._vault.tokenize(
            value=value,
            pii_type=pii_type,
            tenant_id=tenant_id,
            metadata=metadata,
        )

    async def detokenize(
        self,
        token: str,
        tenant_id: str,
        user_id: str,
    ) -> str:
        """Retrieve original value from token.

        Args:
            token: Token string.
            tenant_id: Tenant for authorization.
            user_id: User for audit.

        Returns:
            Original plaintext value.
        """
        return await self._vault.detokenize(
            token=token,
            requester_tenant_id=tenant_id,
            requester_user_id=user_id,
        )

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _merge_detections(
        self,
        detections: List[PIIDetection],
    ) -> List[PIIDetection]:
        """Merge and deduplicate detections from multiple scanners.

        Keeps the highest confidence detection when ranges overlap.

        Args:
            detections: List of detections to merge.

        Returns:
            Deduplicated list.
        """
        if not detections:
            return []

        # Sort by confidence (highest first)
        sorted_detections = sorted(
            detections,
            key=lambda d: d.confidence,
            reverse=True,
        )

        result: List[PIIDetection] = []
        used_ranges: List[Tuple[int, int]] = []

        for detection in sorted_detections:
            # Check for overlap with already-kept detections
            overlaps = False
            for start, end in used_ranges:
                if not (detection.end <= start or detection.start >= end):
                    overlaps = True
                    break

            if not overlaps:
                result.append(detection)
                used_ranges.append((detection.start, detection.end))

        # Sort by position for output
        return sorted(result, key=lambda d: d.start)

    async def _filter_allowlisted(
        self,
        content: str,
        detections: List[PIIDetection],
        tenant_id: Optional[str],
    ) -> List[PIIDetection]:
        """Filter out allowlisted detections.

        Args:
            content: Original content.
            detections: Detections to filter.
            tenant_id: Tenant for context.

        Returns:
            Filtered list.
        """
        filtered: List[PIIDetection] = []

        for detection in detections:
            # Extract detected value from content
            value = content[detection.start : detection.end]

            # Check allowlist
            is_allowed, entry = await self._allowlist.is_allowed(
                value=value,
                pii_type=detection.pii_type,
                tenant_id=tenant_id,
            )

            if is_allowed:
                logger.debug(
                    "Allowlisted: type=%s pattern=%s reason=%s",
                    detection.pii_type.value,
                    entry.pattern[:30] if entry else "?",
                    entry.reason if entry else "?",
                )
                if _metrics_initialized:
                    try:
                        pii_allowlist_matches_total.labels(
                            pii_type=detection.pii_type.value,
                            pattern=entry.pattern[:30] if entry else "unknown",
                        ).inc()
                    except Exception:
                        pass
            else:
                filtered.append(detection)

        return filtered

    def _map_ml_entity_type(self, entity_type: str) -> PIIType:
        """Map Presidio entity type to PIIType.

        Args:
            entity_type: Presidio entity type.

        Returns:
            Corresponding PIIType.
        """
        mapping = {
            "PERSON": PIIType.PERSON_NAME,
            "LOCATION": PIIType.ADDRESS,
            "GPE": PIIType.ADDRESS,
            "PHONE_NUMBER": PIIType.PHONE,
            "EMAIL_ADDRESS": PIIType.EMAIL,
            "CREDIT_CARD": PIIType.CREDIT_CARD,
            "US_SSN": PIIType.SSN,
            "US_PASSPORT": PIIType.PASSPORT,
            "US_DRIVER_LICENSE": PIIType.DRIVERS_LICENSE,
            "IP_ADDRESS": PIIType.IP_ADDRESS,
            "MEDICAL_LICENSE": PIIType.MEDICAL_RECORD,
            "US_BANK_NUMBER": PIIType.BANK_ACCOUNT,
        }
        return mapping.get(entity_type, PIIType.CUSTOM)


# ---------------------------------------------------------------------------
# Global Instance
# ---------------------------------------------------------------------------

_global_pii_service: Optional[PIIService] = None


async def get_pii_service() -> Optional[PIIService]:
    """Get the global PII service instance.

    Returns:
        PIIService or None if not configured.
    """
    return _global_pii_service


async def configure_pii_service(
    config: PIIServiceConfig,
    encryption_service: EncryptionService,
    audit_service: Optional[AuditService] = None,
    db_pool: Optional[Any] = None,
) -> PIIService:
    """Configure and set the global PII service.

    Args:
        config: Service configuration.
        encryption_service: Encryption service.
        audit_service: Optional audit service.
        db_pool: Optional database pool.

    Returns:
        Configured PIIService.
    """
    global _global_pii_service

    _global_pii_service = await PIIService.create(
        config=config,
        encryption_service=encryption_service,
        audit_service=audit_service,
        db_pool=db_pool,
    )

    logger.info("Global PII service configured")
    return _global_pii_service


__all__ = [
    "PIIService",
    "PIIEnforcementEngine",
    "AllowlistManager",
    "get_pii_service",
    "configure_pii_service",
    "DEFAULT_ENFORCEMENT_POLICIES",
    "DEFAULT_ALLOWLIST_ENTRIES",
]
