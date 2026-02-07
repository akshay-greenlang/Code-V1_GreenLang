# -*- coding: utf-8 -*-
"""
Vault Migration - SEC-011: PII Detection/Redaction Enhancements

Migrates existing XOR-encrypted tokens from the legacy PII redaction agent
to AES-256-GCM encryption in the new SecureTokenVault.

Migration Process:
    1. Load existing XOR tokens from pii_redaction agent's token vault
    2. Decrypt XOR tokens using the legacy key
    3. Re-encrypt with AES-256-GCM via EncryptionService
    4. Verify each migration with hash comparison
    5. Update storage with new encrypted values

Safety Features:
    - Batch processing with configurable size
    - Rollback capability within time window
    - Verification step after migration
    - Progress tracking and resumability
    - Audit logging of all operations

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from greenlang.infrastructure.pii_service.models import PIIType
from greenlang.infrastructure.pii_service.config import VaultConfig

if TYPE_CHECKING:
    from greenlang.infrastructure.encryption_service import EncryptionService
    from greenlang.infrastructure.pii_service.secure_vault import SecureTokenVault

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class MigrationProgress:
    """Tracks migration progress for resumability."""

    migration_id: str
    started_at: datetime
    last_batch_at: Optional[datetime] = None
    tokens_processed: int = 0
    tokens_migrated: int = 0
    tokens_failed: int = 0
    tokens_skipped: int = 0
    current_offset: int = 0
    is_complete: bool = False
    errors: List[str] = field(default_factory=list)


class MigrationResult(BaseModel):
    """Result of a migration operation.

    Attributes:
        migration_id: Unique migration identifier.
        success: Whether migration completed successfully.
        tokens_migrated: Number of tokens migrated.
        tokens_failed: Number of tokens that failed.
        tokens_skipped: Number of tokens skipped (already migrated).
        duration_seconds: Total migration duration.
        errors: List of error messages.
        verification_passed: Whether verification passed.
    """

    migration_id: str = Field(..., description="Migration ID")
    success: bool = Field(default=True, description="Migration success")
    tokens_migrated: int = Field(default=0, description="Tokens migrated")
    tokens_failed: int = Field(default=0, description="Tokens failed")
    tokens_skipped: int = Field(default=0, description="Tokens skipped")
    duration_seconds: float = Field(default=0.0, description="Duration")
    errors: List[str] = Field(default_factory=list, description="Errors")
    verification_passed: bool = Field(default=False, description="Verified")


class VerificationResult(BaseModel):
    """Result of migration verification.

    Attributes:
        total_tokens: Total tokens in vault.
        tokens_verified: Tokens successfully verified.
        tokens_failed: Tokens that failed verification.
        all_passed: Whether all tokens passed.
        failures: Details of failed verifications.
    """

    total_tokens: int = Field(default=0, description="Total tokens")
    tokens_verified: int = Field(default=0, description="Verified tokens")
    tokens_failed: int = Field(default=0, description="Failed tokens")
    all_passed: bool = Field(default=False, description="All passed")
    failures: List[Dict[str, Any]] = Field(default_factory=list, description="Failures")


@dataclass
class LegacyTokenEntry:
    """Represents a token from the legacy XOR-based vault."""

    token_id: str
    pii_type: str
    original_hash: str
    encrypted_value: str  # XOR-encrypted, hex-encoded
    tenant_id: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0


# ---------------------------------------------------------------------------
# XOR Decryption (Legacy)
# ---------------------------------------------------------------------------


def _xor_decrypt(encrypted_hex: str, key: bytes) -> str:
    """Decrypt XOR-encrypted value (legacy format).

    This replicates the simple XOR encryption from pii_redaction.py:
    encrypted = bytes(a ^ b for a, b in zip(value.encode(), key * N))

    Args:
        encrypted_hex: Hex-encoded encrypted value.
        key: XOR key bytes.

    Returns:
        Decrypted plaintext string.
    """
    encrypted_bytes = bytes.fromhex(encrypted_hex)
    key_extended = key * ((len(encrypted_bytes) // len(key)) + 1)
    decrypted = bytes(a ^ b for a, b in zip(encrypted_bytes, key_extended))
    return decrypted.decode("utf-8")


# ---------------------------------------------------------------------------
# Vault Migrator
# ---------------------------------------------------------------------------


class VaultMigrator:
    """Migrates tokens from XOR to AES-256-GCM encryption.

    Handles the complete migration lifecycle:
    1. Load legacy tokens in batches
    2. Decrypt XOR encryption
    3. Re-encrypt with AES-256-GCM
    4. Store in new vault format
    5. Verify migration integrity

    Example:
        >>> migrator = VaultMigrator(
        ...     encryption_service=encryption_svc,
        ...     secure_vault=vault,
        ...     legacy_xor_key=b"legacy_key_32_bytes_here_______",
        ... )
        >>> result = await migrator.migrate_xor_to_aes(batch_size=1000)
        >>> if result.success:
        ...     verification = await migrator.verify_migration()
    """

    def __init__(
        self,
        encryption_service: EncryptionService,
        secure_vault: SecureTokenVault,
        legacy_xor_key: bytes,
        db_pool: Optional[Any] = None,
    ) -> None:
        """Initialize VaultMigrator.

        Args:
            encryption_service: SEC-003 encryption service.
            secure_vault: New AES-256-GCM vault.
            legacy_xor_key: XOR key from legacy system.
            db_pool: Database pool for direct access.
        """
        self._encryption = encryption_service
        self._vault = secure_vault
        self._legacy_key = legacy_xor_key
        self._db_pool = db_pool

        # Progress tracking
        self._progress: Optional[MigrationProgress] = None

        logger.info("VaultMigrator initialized")

    async def migrate_xor_to_aes(
        self,
        batch_size: int = 1000,
        dry_run: bool = False,
        tenant_id: Optional[str] = None,
    ) -> MigrationResult:
        """Migrate XOR tokens to AES-256-GCM encryption.

        Args:
            batch_size: Tokens per batch.
            dry_run: If True, don't actually migrate.
            tenant_id: Optional tenant filter.

        Returns:
            MigrationResult with migration statistics.
        """
        import uuid

        migration_id = str(uuid.uuid4())[:8]
        start_time = datetime.utcnow()

        self._progress = MigrationProgress(
            migration_id=migration_id,
            started_at=start_time,
        )

        logger.info(
            "Starting migration: id=%s batch_size=%d dry_run=%s tenant=%s",
            migration_id,
            batch_size,
            dry_run,
            tenant_id or "all",
        )

        try:
            # Load legacy tokens in batches
            offset = 0
            while True:
                legacy_tokens = await self._load_legacy_tokens(
                    offset=offset,
                    limit=batch_size,
                    tenant_id=tenant_id,
                )

                if not legacy_tokens:
                    break

                # Process batch
                for token in legacy_tokens:
                    try:
                        result = await self._migrate_single_token(token, dry_run)
                        if result == "migrated":
                            self._progress.tokens_migrated += 1
                        elif result == "skipped":
                            self._progress.tokens_skipped += 1
                        else:
                            self._progress.tokens_failed += 1

                    except Exception as e:
                        self._progress.tokens_failed += 1
                        self._progress.errors.append(
                            f"Token {token.token_id}: {str(e)}"
                        )
                        logger.error(
                            "Failed to migrate token %s: %s",
                            token.token_id[:8],
                            e,
                        )

                    self._progress.tokens_processed += 1

                self._progress.last_batch_at = datetime.utcnow()
                self._progress.current_offset = offset + len(legacy_tokens)
                offset += batch_size

                logger.info(
                    "Migration progress: processed=%d migrated=%d failed=%d",
                    self._progress.tokens_processed,
                    self._progress.tokens_migrated,
                    self._progress.tokens_failed,
                )

            self._progress.is_complete = True
            duration = (datetime.utcnow() - start_time).total_seconds()

            logger.info(
                "Migration complete: id=%s migrated=%d failed=%d duration=%.1fs",
                migration_id,
                self._progress.tokens_migrated,
                self._progress.tokens_failed,
                duration,
            )

            return MigrationResult(
                migration_id=migration_id,
                success=self._progress.tokens_failed == 0,
                tokens_migrated=self._progress.tokens_migrated,
                tokens_failed=self._progress.tokens_failed,
                tokens_skipped=self._progress.tokens_skipped,
                duration_seconds=duration,
                errors=self._progress.errors[:100],  # Limit error list
            )

        except Exception as e:
            logger.error("Migration failed: %s", e, exc_info=True)
            duration = (datetime.utcnow() - start_time).total_seconds()
            return MigrationResult(
                migration_id=migration_id,
                success=False,
                tokens_migrated=self._progress.tokens_migrated if self._progress else 0,
                tokens_failed=self._progress.tokens_failed if self._progress else 0,
                duration_seconds=duration,
                errors=[str(e)],
            )

    async def verify_migration(
        self,
        sample_size: int = 100,
        tenant_id: Optional[str] = None,
    ) -> VerificationResult:
        """Verify migration integrity.

        Checks that migrated tokens can be decrypted and match
        the original hash values.

        Args:
            sample_size: Number of tokens to verify (0 = all).
            tenant_id: Optional tenant filter.

        Returns:
            VerificationResult with verification status.
        """
        logger.info(
            "Verifying migration: sample_size=%d tenant=%s",
            sample_size,
            tenant_id or "all",
        )

        result = VerificationResult()

        try:
            # Load tokens to verify
            if self._db_pool is not None:
                async with self._db_pool.connection() as conn:
                    query = """
                        SELECT token_id, pii_type, original_hash, tenant_id
                        FROM pii_service.token_vault
                        WHERE encryption_version = 'aes256gcm'
                    """
                    params = []

                    if tenant_id:
                        query += " AND tenant_id = $1"
                        params.append(tenant_id)

                    if sample_size > 0:
                        query += f" LIMIT {sample_size}"

                    rows = await conn.fetch(query, *params)
                    result.total_tokens = len(rows)

                    for row in rows:
                        try:
                            # Get token and attempt detokenization
                            token_str = f"[TOKEN:{row['token_id']}]"
                            plaintext = await self._vault.detokenize(
                                token=token_str,
                                requester_tenant_id=row["tenant_id"],
                                requester_user_id="migration_verifier",
                            )

                            # Verify hash matches
                            computed_hash = hashlib.sha256(
                                plaintext.encode()
                            ).hexdigest()
                            if computed_hash == row["original_hash"]:
                                result.tokens_verified += 1
                            else:
                                result.tokens_failed += 1
                                result.failures.append({
                                    "token_id": row["token_id"][:8],
                                    "reason": "hash_mismatch",
                                })

                        except Exception as e:
                            result.tokens_failed += 1
                            result.failures.append({
                                "token_id": row["token_id"][:8],
                                "reason": str(e),
                            })

            result.all_passed = result.tokens_failed == 0

            logger.info(
                "Verification complete: verified=%d failed=%d all_passed=%s",
                result.tokens_verified,
                result.tokens_failed,
                result.all_passed,
            )

            return result

        except Exception as e:
            logger.error("Verification failed: %s", e, exc_info=True)
            return VerificationResult(
                total_tokens=0,
                tokens_verified=0,
                tokens_failed=1,
                all_passed=False,
                failures=[{"reason": str(e)}],
            )

    async def rollback(
        self,
        before_timestamp: datetime,
        tenant_id: Optional[str] = None,
    ) -> int:
        """Rollback migration to before a timestamp.

        Removes tokens that were migrated after the specified time,
        allowing the legacy tokens to be re-migrated.

        Args:
            before_timestamp: Remove migrations after this time.
            tenant_id: Optional tenant filter.

        Returns:
            Number of tokens rolled back.
        """
        logger.warning(
            "Rolling back migration: before=%s tenant=%s",
            before_timestamp.isoformat(),
            tenant_id or "all",
        )

        rolled_back = 0

        if self._db_pool is not None:
            try:
                async with self._db_pool.connection() as conn:
                    query = """
                        DELETE FROM pii_service.token_vault
                        WHERE created_at > $1
                        AND encryption_version = 'aes256gcm'
                    """
                    params = [before_timestamp]

                    if tenant_id:
                        query += " AND tenant_id = $2"
                        params.append(tenant_id)

                    query += " RETURNING token_id"

                    result = await conn.fetch(query, *params)
                    rolled_back = len(result)

            except Exception as e:
                logger.error("Rollback failed: %s", e, exc_info=True)
                raise

        logger.info("Rolled back %d tokens", rolled_back)
        return rolled_back

    async def get_progress(self) -> Optional[MigrationProgress]:
        """Get current migration progress.

        Returns:
            MigrationProgress or None if no migration in progress.
        """
        return self._progress

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    async def _load_legacy_tokens(
        self,
        offset: int,
        limit: int,
        tenant_id: Optional[str] = None,
    ) -> List[LegacyTokenEntry]:
        """Load legacy XOR tokens from database.

        Args:
            offset: Pagination offset.
            limit: Number of tokens to load.
            tenant_id: Optional tenant filter.

        Returns:
            List of LegacyTokenEntry objects.
        """
        tokens = []

        if self._db_pool is not None:
            try:
                async with self._db_pool.connection() as conn:
                    # Query legacy tokens (those without encryption_version or with 'xor')
                    query = """
                        SELECT token_id, pii_type, original_hash, encrypted_value,
                               tenant_id, created_at, expires_at, access_count
                        FROM pii_service.token_vault_legacy
                        WHERE (encryption_version IS NULL OR encryption_version = 'xor')
                    """
                    params = []

                    if tenant_id:
                        query += " AND tenant_id = $1"
                        params.append(tenant_id)

                    query += f" ORDER BY created_at LIMIT {limit} OFFSET {offset}"

                    rows = await conn.fetch(query, *params)

                    for row in rows:
                        tokens.append(
                            LegacyTokenEntry(
                                token_id=row["token_id"],
                                pii_type=row["pii_type"],
                                original_hash=row["original_hash"],
                                encrypted_value=row["encrypted_value"],
                                tenant_id=row["tenant_id"],
                                created_at=row["created_at"],
                                expires_at=row["expires_at"],
                                access_count=row["access_count"] or 0,
                            )
                        )

            except Exception as e:
                logger.error("Failed to load legacy tokens: %s", e)

        return tokens

    async def _migrate_single_token(
        self,
        legacy_token: LegacyTokenEntry,
        dry_run: bool,
    ) -> str:
        """Migrate a single token from XOR to AES-256-GCM.

        Args:
            legacy_token: Legacy token entry.
            dry_run: If True, don't actually migrate.

        Returns:
            "migrated", "skipped", or "failed".
        """
        # Check if already migrated
        if self._db_pool is not None:
            async with self._db_pool.connection() as conn:
                existing = await conn.fetchrow(
                    """
                    SELECT token_id FROM pii_service.token_vault
                    WHERE token_id = $1 AND encryption_version = 'aes256gcm'
                    """,
                    legacy_token.token_id,
                )
                if existing:
                    return "skipped"

        # Decrypt XOR
        try:
            plaintext = _xor_decrypt(legacy_token.encrypted_value, self._legacy_key)
        except Exception as e:
            logger.error(
                "Failed to decrypt XOR token %s: %s",
                legacy_token.token_id[:8],
                e,
            )
            return "failed"

        # Verify hash
        computed_hash = hashlib.sha256(plaintext.encode()).hexdigest()
        if computed_hash != legacy_token.original_hash:
            logger.error(
                "Hash mismatch for token %s",
                legacy_token.token_id[:8],
            )
            return "failed"

        if dry_run:
            logger.debug("Dry run: would migrate token %s", legacy_token.token_id[:8])
            return "migrated"

        # Re-encrypt with AES-256-GCM via the vault
        try:
            pii_type = PIIType(legacy_token.pii_type)
            tenant_id = legacy_token.tenant_id or "default"

            # Use vault's tokenize which handles encryption
            new_token = await self._vault.tokenize(
                value=plaintext,
                pii_type=pii_type,
                tenant_id=tenant_id,
            )

            logger.debug(
                "Migrated token %s -> %s",
                legacy_token.token_id[:8],
                new_token[:20],
            )
            return "migrated"

        except Exception as e:
            logger.error(
                "Failed to re-encrypt token %s: %s",
                legacy_token.token_id[:8],
                e,
            )
            return "failed"


__all__ = [
    "VaultMigrator",
    "MigrationResult",
    "VerificationResult",
    "MigrationProgress",
    "LegacyTokenEntry",
]
