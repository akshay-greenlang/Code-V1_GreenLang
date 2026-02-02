"""
Secrets Rotation Manager for GreenLang Process Heat Platform
=============================================================

TASK-154/155: Automatic Secrets Rotation Implementation

This module provides automatic rotation for:
- Database credentials (PostgreSQL)
- API keys and tokens
- TLS certificates
- Transit encryption keys

All rotation events are logged for audit compliance.

Example:
    >>> from greenlang.infrastructure.secrets import SecretsRotationManager, RotationConfig
    >>> config = RotationConfig(
    ...     database_rotation_interval=timedelta(hours=24),
    ...     certificate_renewal_threshold=timedelta(days=30),
    ... )
    >>> manager = SecretsRotationManager(vault_client, config)
    >>> await manager.start()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class RotationType(str, Enum):
    """Types of secrets that can be rotated."""
    DATABASE_CREDENTIAL = "database_credential"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    ENCRYPTION_KEY = "encryption_key"
    STATIC_SECRET = "static_secret"


class RotationStatus(str, Enum):
    """Status of a rotation operation."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    IN_PROGRESS = "in_progress"


@dataclass
class RotationResult:
    """Result of a rotation operation."""
    rotation_type: RotationType
    secret_path: str
    status: RotationStatus
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    rotated_at: datetime = field(default_factory=datetime.utcnow)
    next_rotation: Optional[datetime] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_audit_log(self) -> dict[str, Any]:
        """Convert to audit log format."""
        return {
            "event_type": "secret_rotation",
            "rotation_type": self.rotation_type.value,
            "secret_path": self.secret_path,
            "status": self.status.value,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "rotated_at": self.rotated_at.isoformat(),
            "next_rotation": self.next_rotation.isoformat() if self.next_rotation else None,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class RotationConfig:
    """Configuration for secrets rotation."""
    # Database rotation
    database_rotation_enabled: bool = True
    database_rotation_interval: timedelta = field(default_factory=lambda: timedelta(hours=24))
    database_roles: list[str] = field(default_factory=lambda: [
        "process-heat-readonly",
        "process-heat-readwrite",
    ])

    # API key rotation
    api_key_rotation_enabled: bool = True
    api_key_rotation_interval: timedelta = field(default_factory=lambda: timedelta(days=90))
    api_key_paths: list[str] = field(default_factory=lambda: [
        "process-heat/api-keys",
    ])

    # Certificate rotation
    certificate_rotation_enabled: bool = True
    certificate_renewal_threshold: timedelta = field(default_factory=lambda: timedelta(days=30))
    certificate_roles: list[str] = field(default_factory=lambda: [
        "process-heat-services",
        "internal-mtls",
    ])

    # Encryption key rotation
    encryption_key_rotation_enabled: bool = True
    encryption_key_rotation_interval: timedelta = field(default_factory=lambda: timedelta(days=90))
    transit_keys: list[str] = field(default_factory=lambda: [
        "process-heat-data-key",
        "process-heat-pii-key",
    ])

    # General settings
    rotation_check_interval: timedelta = field(default_factory=lambda: timedelta(hours=1))
    max_concurrent_rotations: int = 3
    retry_failed_rotations: bool = True
    max_rotation_retries: int = 3
    retry_delay: timedelta = field(default_factory=lambda: timedelta(minutes=5))

    # Audit logging
    audit_log_enabled: bool = True
    audit_log_path: str = "/var/log/greenlang/secrets-rotation.log"

    # Notifications
    notification_enabled: bool = True
    notification_on_success: bool = False
    notification_on_failure: bool = True
    slack_webhook_url: Optional[str] = field(
        default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL")
    )


@dataclass
class RotationSchedule:
    """Schedule for a rotation task."""
    rotation_type: RotationType
    identifier: str
    last_rotation: Optional[datetime] = None
    next_rotation: Optional[datetime] = None
    retry_count: int = 0
    last_error: Optional[str] = None


class RotationHandler(ABC):
    """Base class for rotation handlers."""

    @abstractmethod
    async def rotate(self, identifier: str) -> RotationResult:
        """Perform the rotation."""
        pass

    @abstractmethod
    async def validate(self, identifier: str) -> bool:
        """Validate the rotated secret is working."""
        pass

    @abstractmethod
    async def rollback(self, identifier: str, previous_version: str) -> bool:
        """Rollback to previous version if rotation failed."""
        pass


class DatabaseCredentialRotationHandler(RotationHandler):
    """Handler for database credential rotation."""

    def __init__(self, vault_client, config: RotationConfig):
        self.vault_client = vault_client
        self.config = config

    async def rotate(self, role: str) -> RotationResult:
        """Rotate database credentials for a role."""
        try:
            # For dynamic credentials, Vault handles rotation automatically
            # We just need to verify new credentials work

            # Get current lease info for tracking
            current_creds = await self.vault_client.get_database_credentials(role)
            old_lease = current_creds.lease_id

            # Request new credentials (this gets a fresh lease)
            new_creds = await self.vault_client.get_database_credentials(role)

            # Validate new credentials work
            if not await self.validate(role):
                # Attempt rollback by keeping old lease
                return RotationResult(
                    rotation_type=RotationType.DATABASE_CREDENTIAL,
                    secret_path=f"database/creds/{role}",
                    status=RotationStatus.FAILED,
                    old_version=old_lease,
                    error="New credentials validation failed",
                )

            # Revoke old lease if different
            if old_lease and old_lease != new_creds.lease_id:
                try:
                    await self.vault_client.revoke_lease(old_lease)
                except Exception as e:
                    logger.warning(f"Failed to revoke old lease {old_lease}: {e}")

            next_rotation = datetime.utcnow() + self.config.database_rotation_interval

            return RotationResult(
                rotation_type=RotationType.DATABASE_CREDENTIAL,
                secret_path=f"database/creds/{role}",
                status=RotationStatus.SUCCESS,
                old_version=old_lease,
                new_version=new_creds.lease_id,
                next_rotation=next_rotation,
                metadata={
                    "role": role,
                    "lease_duration": new_creds.lease_duration,
                },
            )

        except Exception as e:
            logger.error(f"Database credential rotation failed for {role}: {e}")
            return RotationResult(
                rotation_type=RotationType.DATABASE_CREDENTIAL,
                secret_path=f"database/creds/{role}",
                status=RotationStatus.FAILED,
                error=str(e),
            )

    async def validate(self, role: str) -> bool:
        """Validate database credentials by attempting connection."""
        try:
            creds = await self.vault_client.get_database_credentials(role)

            # Try to establish a connection
            import asyncpg
            conn = await asyncpg.connect(
                host=creds.host,
                port=creds.port,
                database=creds.database,
                user=creds.username,
                password=creds.password,
                ssl=creds.ssl_mode,
                timeout=10,
            )
            await conn.execute("SELECT 1")
            await conn.close()
            return True

        except ImportError:
            # asyncpg not available, skip validation
            logger.warning("asyncpg not available, skipping database validation")
            return True
        except Exception as e:
            logger.error(f"Database credential validation failed: {e}")
            return False

    async def rollback(self, role: str, previous_version: str) -> bool:
        """Rollback is handled by keeping the old lease active."""
        # For dynamic credentials, rollback means not revoking the old lease
        # which happens automatically if rotation fails
        return True


class APIKeyRotationHandler(RotationHandler):
    """Handler for API key rotation."""

    def __init__(self, vault_client, config: RotationConfig):
        self.vault_client = vault_client
        self.config = config

    async def rotate(self, path: str) -> RotationResult:
        """Rotate API keys at the given path."""
        try:
            # Get current secret
            current_secret = await self.vault_client.get_secret(path)
            old_version = current_secret.metadata.get("version")

            # Generate new API keys
            new_keys = {}
            for key, value in current_secret.data.items():
                if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
                    # Generate new secure key
                    new_keys[key] = self._generate_api_key()
                else:
                    # Keep non-key values unchanged
                    new_keys[key] = value

            # Update secret in Vault
            metadata = await self.vault_client.put_secret(path, new_keys)
            new_version = metadata.get("version")

            # Validation would require application-specific logic
            # For now, we trust the write succeeded

            next_rotation = datetime.utcnow() + self.config.api_key_rotation_interval

            return RotationResult(
                rotation_type=RotationType.API_KEY,
                secret_path=path,
                status=RotationStatus.SUCCESS,
                old_version=str(old_version),
                new_version=str(new_version),
                next_rotation=next_rotation,
                metadata={
                    "keys_rotated": list(new_keys.keys()),
                },
            )

        except Exception as e:
            logger.error(f"API key rotation failed for {path}: {e}")
            return RotationResult(
                rotation_type=RotationType.API_KEY,
                secret_path=path,
                status=RotationStatus.FAILED,
                error=str(e),
            )

    def _generate_api_key(self, length: int = 32) -> str:
        """Generate a secure API key."""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    async def validate(self, path: str) -> bool:
        """Validate API keys exist."""
        try:
            secret = await self.vault_client.get_secret(path)
            return bool(secret.data)
        except Exception:
            return False

    async def rollback(self, path: str, previous_version: str) -> bool:
        """Rollback to previous version of API keys."""
        try:
            # Get previous version
            previous_secret = await self.vault_client.get_secret(
                path, version=int(previous_version)
            )

            # Restore previous data
            await self.vault_client.put_secret(path, previous_secret.data)
            return True

        except Exception as e:
            logger.error(f"API key rollback failed for {path}: {e}")
            return False


class CertificateRotationHandler(RotationHandler):
    """Handler for TLS certificate rotation."""

    def __init__(self, vault_client, config: RotationConfig):
        self.vault_client = vault_client
        self.config = config
        self._certificate_store: dict[str, Any] = {}

    async def rotate(self, role: str) -> RotationResult:
        """Rotate certificates for a PKI role."""
        try:
            # Get current certificate info if exists
            old_serial = None
            if role in self._certificate_store:
                old_cert = self._certificate_store[role]
                old_serial = old_cert.serial_number

            # Generate new certificate
            # Common name based on role
            common_name = f"{role}.greenlang.svc.cluster.local"

            new_cert = await self.vault_client.generate_certificate(
                role=role,
                common_name=common_name,
                alt_names=[
                    f"{role}.greenlang.svc",
                    f"{role}.greenlang.internal",
                ],
                ttl="720h",  # 30 days
            )

            # Store new certificate
            self._certificate_store[role] = new_cert

            # Revoke old certificate if exists
            if old_serial:
                try:
                    await self.vault_client.revoke_certificate(old_serial)
                except Exception as e:
                    logger.warning(f"Failed to revoke old certificate {old_serial}: {e}")

            # Calculate next rotation (threshold before expiry)
            next_rotation = new_cert.expiration - self.config.certificate_renewal_threshold

            return RotationResult(
                rotation_type=RotationType.CERTIFICATE,
                secret_path=f"pki_int/issue/{role}",
                status=RotationStatus.SUCCESS,
                old_version=old_serial,
                new_version=new_cert.serial_number,
                next_rotation=next_rotation,
                metadata={
                    "common_name": common_name,
                    "expiration": new_cert.expiration.isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Certificate rotation failed for {role}: {e}")
            return RotationResult(
                rotation_type=RotationType.CERTIFICATE,
                secret_path=f"pki_int/issue/{role}",
                status=RotationStatus.FAILED,
                error=str(e),
            )

    async def validate(self, role: str) -> bool:
        """Validate certificate is valid and not expired."""
        if role not in self._certificate_store:
            return False

        cert = self._certificate_store[role]
        return not cert.is_expired

    async def rollback(self, role: str, previous_version: str) -> bool:
        """Certificate rollback is not supported - issue new cert instead."""
        logger.warning("Certificate rollback not supported, issuing new certificate")
        result = await self.rotate(role)
        return result.status == RotationStatus.SUCCESS


class EncryptionKeyRotationHandler(RotationHandler):
    """Handler for Transit encryption key rotation."""

    def __init__(self, vault_client, config: RotationConfig):
        self.vault_client = vault_client
        self.config = config

    async def rotate(self, key_name: str) -> RotationResult:
        """Rotate a Transit encryption key."""
        try:
            # Get current key version
            key_info = await self._get_key_info(key_name)
            old_version = key_info.get("latest_version", 0)

            # Rotate the key (Vault handles this)
            await self.vault_client._request(
                "POST",
                f"/v1/transit/keys/{key_name}/rotate",
            )

            # Get new key version
            new_key_info = await self._get_key_info(key_name)
            new_version = new_key_info.get("latest_version", 0)

            next_rotation = datetime.utcnow() + self.config.encryption_key_rotation_interval

            return RotationResult(
                rotation_type=RotationType.ENCRYPTION_KEY,
                secret_path=f"transit/keys/{key_name}",
                status=RotationStatus.SUCCESS,
                old_version=str(old_version),
                new_version=str(new_version),
                next_rotation=next_rotation,
                metadata={
                    "key_name": key_name,
                    "min_decryption_version": new_key_info.get("min_decryption_version", 1),
                },
            )

        except Exception as e:
            logger.error(f"Encryption key rotation failed for {key_name}: {e}")
            return RotationResult(
                rotation_type=RotationType.ENCRYPTION_KEY,
                secret_path=f"transit/keys/{key_name}",
                status=RotationStatus.FAILED,
                error=str(e),
            )

    async def _get_key_info(self, key_name: str) -> dict[str, Any]:
        """Get information about a Transit key."""
        response = await self.vault_client._request(
            "GET",
            f"/v1/transit/keys/{key_name}",
        )
        return response.get("data", {})

    async def validate(self, key_name: str) -> bool:
        """Validate encryption key by encrypting test data."""
        try:
            test_data = b"rotation-validation-test"
            ciphertext = await self.vault_client.encrypt_data(key_name, test_data)
            decrypted = await self.vault_client.decrypt_data(key_name, ciphertext)
            return decrypted == test_data
        except Exception as e:
            logger.error(f"Encryption key validation failed: {e}")
            return False

    async def rollback(self, key_name: str, previous_version: str) -> bool:
        """Encryption key rollback - update min version."""
        # Note: Transit keys cannot be truly rolled back
        # We can only adjust min_decryption_version
        logger.warning("Encryption key rollback not fully supported")
        return False


class AuditLogger:
    """Audit logger for secrets rotation events."""

    def __init__(self, config: RotationConfig):
        self.config = config
        self._file_handler: Optional[logging.FileHandler] = None
        self._logger = logging.getLogger("secrets.rotation.audit")
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Set up the audit logger."""
        if not self.config.audit_log_enabled:
            return

        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        # Create log directory if needed
        log_dir = os.path.dirname(self.config.audit_log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # File handler for audit logs
        self._file_handler = logging.FileHandler(self.config.audit_log_path)
        self._file_handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        self._logger.addHandler(self._file_handler)

    async def log_rotation(self, result: RotationResult) -> None:
        """Log a rotation event."""
        if not self.config.audit_log_enabled:
            return

        audit_entry = result.to_audit_log()
        audit_entry["timestamp"] = datetime.utcnow().isoformat()

        self._logger.info(json.dumps(audit_entry))

    async def close(self) -> None:
        """Close the audit logger."""
        if self._file_handler:
            self._file_handler.close()
            self._logger.removeHandler(self._file_handler)


class NotificationService:
    """Notification service for rotation events."""

    def __init__(self, config: RotationConfig):
        self.config = config

    async def notify(self, result: RotationResult) -> None:
        """Send notification for rotation event."""
        if not self.config.notification_enabled:
            return

        if result.status == RotationStatus.SUCCESS and not self.config.notification_on_success:
            return

        if result.status == RotationStatus.FAILED and not self.config.notification_on_failure:
            return

        if self.config.slack_webhook_url:
            await self._send_slack_notification(result)

    async def _send_slack_notification(self, result: RotationResult) -> None:
        """Send Slack notification."""
        try:
            import httpx

            color = "#36a64f" if result.status == RotationStatus.SUCCESS else "#ff0000"
            status_emoji = ":white_check_mark:" if result.status == RotationStatus.SUCCESS else ":x:"

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{status_emoji} Secrets Rotation: {result.rotation_type.value}",
                        "fields": [
                            {
                                "title": "Secret Path",
                                "value": result.secret_path,
                                "short": True,
                            },
                            {
                                "title": "Status",
                                "value": result.status.value,
                                "short": True,
                            },
                        ],
                        "footer": "GreenLang Secrets Rotation Manager",
                        "ts": int(result.rotated_at.timestamp()),
                    }
                ]
            }

            if result.error:
                payload["attachments"][0]["fields"].append({
                    "title": "Error",
                    "value": result.error,
                    "short": False,
                })

            async with httpx.AsyncClient() as client:
                await client.post(
                    self.config.slack_webhook_url,
                    json=payload,
                    timeout=10,
                )

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")


class SecretsRotationManager:
    """
    Manager for automatic secrets rotation.

    Handles scheduling and execution of rotation tasks for:
    - Database credentials
    - API keys
    - TLS certificates
    - Encryption keys

    Example:
        >>> from greenlang.infrastructure.secrets import VaultClient, SecretsRotationManager
        >>> vault_client = VaultClient()
        >>> await vault_client.connect()
        >>>
        >>> manager = SecretsRotationManager(vault_client)
        >>> await manager.start()
        >>>
        >>> # Manually trigger rotation
        >>> result = await manager.rotate_database_credentials("process-heat-readonly")
    """

    def __init__(
        self,
        vault_client,
        config: Optional[RotationConfig] = None,
    ):
        """Initialize the rotation manager."""
        self.vault_client = vault_client
        self.config = config or RotationConfig()

        # Initialize handlers
        self._handlers: dict[RotationType, RotationHandler] = {
            RotationType.DATABASE_CREDENTIAL: DatabaseCredentialRotationHandler(
                vault_client, self.config
            ),
            RotationType.API_KEY: APIKeyRotationHandler(vault_client, self.config),
            RotationType.CERTIFICATE: CertificateRotationHandler(vault_client, self.config),
            RotationType.ENCRYPTION_KEY: EncryptionKeyRotationHandler(vault_client, self.config),
        }

        # Initialize services
        self._audit_logger = AuditLogger(self.config)
        self._notification_service = NotificationService(self.config)

        # Rotation state
        self._schedules: dict[str, RotationSchedule] = {}
        self._running = False
        self._rotation_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_rotations)
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the rotation manager."""
        if self._running:
            return

        self._running = True
        await self._initialize_schedules()
        self._rotation_task = asyncio.create_task(self._rotation_loop())
        logger.info("Secrets rotation manager started")

    async def stop(self) -> None:
        """Stop the rotation manager."""
        self._running = False

        if self._rotation_task:
            self._rotation_task.cancel()
            try:
                await self._rotation_task
            except asyncio.CancelledError:
                pass

        await self._audit_logger.close()
        logger.info("Secrets rotation manager stopped")

    async def _initialize_schedules(self) -> None:
        """Initialize rotation schedules based on configuration."""
        now = datetime.utcnow()

        # Database roles
        if self.config.database_rotation_enabled:
            for role in self.config.database_roles:
                key = f"database:{role}"
                self._schedules[key] = RotationSchedule(
                    rotation_type=RotationType.DATABASE_CREDENTIAL,
                    identifier=role,
                    next_rotation=now + self.config.database_rotation_interval,
                )

        # API keys
        if self.config.api_key_rotation_enabled:
            for path in self.config.api_key_paths:
                key = f"api_key:{path}"
                self._schedules[key] = RotationSchedule(
                    rotation_type=RotationType.API_KEY,
                    identifier=path,
                    next_rotation=now + self.config.api_key_rotation_interval,
                )

        # Certificates
        if self.config.certificate_rotation_enabled:
            for role in self.config.certificate_roles:
                key = f"certificate:{role}"
                # Certificates check expiry, not fixed interval
                self._schedules[key] = RotationSchedule(
                    rotation_type=RotationType.CERTIFICATE,
                    identifier=role,
                    next_rotation=now,  # Check immediately
                )

        # Encryption keys
        if self.config.encryption_key_rotation_enabled:
            for key_name in self.config.transit_keys:
                key = f"encryption_key:{key_name}"
                self._schedules[key] = RotationSchedule(
                    rotation_type=RotationType.ENCRYPTION_KEY,
                    identifier=key_name,
                    next_rotation=now + self.config.encryption_key_rotation_interval,
                )

    async def _rotation_loop(self) -> None:
        """Main rotation loop."""
        while self._running:
            try:
                await self._check_and_rotate()
                await asyncio.sleep(self.config.rotation_check_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rotation loop error: {e}")
                await asyncio.sleep(60)

    async def _check_and_rotate(self) -> None:
        """Check schedules and perform due rotations."""
        now = datetime.utcnow()
        tasks = []

        async with self._lock:
            for key, schedule in self._schedules.items():
                if schedule.next_rotation and schedule.next_rotation <= now:
                    tasks.append(self._execute_rotation(key, schedule))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_rotation(self, key: str, schedule: RotationSchedule) -> None:
        """Execute a single rotation task."""
        async with self._semaphore:
            handler = self._handlers.get(schedule.rotation_type)
            if not handler:
                logger.error(f"No handler for rotation type: {schedule.rotation_type}")
                return

            logger.info(f"Starting rotation for {key}")

            result = await handler.rotate(schedule.identifier)

            # Update schedule
            async with self._lock:
                schedule.last_rotation = result.rotated_at
                if result.status == RotationStatus.SUCCESS:
                    schedule.next_rotation = result.next_rotation
                    schedule.retry_count = 0
                    schedule.last_error = None
                else:
                    schedule.last_error = result.error
                    schedule.retry_count += 1

                    # Schedule retry if enabled
                    if (
                        self.config.retry_failed_rotations
                        and schedule.retry_count < self.config.max_rotation_retries
                    ):
                        schedule.next_rotation = (
                            datetime.utcnow() + self.config.retry_delay
                        )
                    else:
                        # Disable further retries until manual intervention
                        schedule.next_rotation = None
                        logger.error(f"Rotation for {key} exceeded max retries")

            # Audit and notify
            await self._audit_logger.log_rotation(result)
            await self._notification_service.notify(result)

    # Manual rotation methods

    async def rotate_database_credentials(self, role: str) -> RotationResult:
        """Manually rotate database credentials for a role."""
        handler = self._handlers[RotationType.DATABASE_CREDENTIAL]
        result = await handler.rotate(role)
        await self._audit_logger.log_rotation(result)
        return result

    async def rotate_api_keys(self, path: str) -> RotationResult:
        """Manually rotate API keys at a path."""
        handler = self._handlers[RotationType.API_KEY]
        result = await handler.rotate(path)
        await self._audit_logger.log_rotation(result)
        return result

    async def rotate_certificate(self, role: str) -> RotationResult:
        """Manually rotate a certificate for a role."""
        handler = self._handlers[RotationType.CERTIFICATE]
        result = await handler.rotate(role)
        await self._audit_logger.log_rotation(result)
        return result

    async def rotate_encryption_key(self, key_name: str) -> RotationResult:
        """Manually rotate a Transit encryption key."""
        handler = self._handlers[RotationType.ENCRYPTION_KEY]
        result = await handler.rotate(key_name)
        await self._audit_logger.log_rotation(result)
        return result

    async def rotate_all(self) -> list[RotationResult]:
        """Force rotation of all managed secrets."""
        results = []

        for schedule in self._schedules.values():
            handler = self._handlers.get(schedule.rotation_type)
            if handler:
                result = await handler.rotate(schedule.identifier)
                await self._audit_logger.log_rotation(result)
                results.append(result)

        return results

    def get_rotation_status(self) -> dict[str, dict[str, Any]]:
        """Get current status of all rotation schedules."""
        status = {}
        for key, schedule in self._schedules.items():
            status[key] = {
                "type": schedule.rotation_type.value,
                "identifier": schedule.identifier,
                "last_rotation": (
                    schedule.last_rotation.isoformat()
                    if schedule.last_rotation else None
                ),
                "next_rotation": (
                    schedule.next_rotation.isoformat()
                    if schedule.next_rotation else None
                ),
                "retry_count": schedule.retry_count,
                "last_error": schedule.last_error,
            }
        return status

    async def health_check(self) -> dict[str, Any]:
        """Check health of the rotation manager."""
        failed_count = sum(
            1 for s in self._schedules.values()
            if s.last_error is not None
        )

        pending_count = sum(
            1 for s in self._schedules.values()
            if s.next_rotation and s.next_rotation <= datetime.utcnow()
        )

        return {
            "running": self._running,
            "total_schedules": len(self._schedules),
            "failed_rotations": failed_count,
            "pending_rotations": pending_count,
            "healthy": failed_count == 0 and self._running,
        }
