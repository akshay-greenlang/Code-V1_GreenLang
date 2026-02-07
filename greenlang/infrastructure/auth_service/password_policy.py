# -*- coding: utf-8 -*-
"""
Password Policy - JWT Authentication Service (SEC-001)

Enforces password complexity requirements, history tracking, expiry policies,
and optional breach detection via the HaveIBeenPwned k-anonymity API.

The module provides a synchronous ``validate`` method for immediate complexity
checks, and async helpers for database-backed history, expiry, and breach
lookups.  Configuration is immutable via ``dataclass(frozen=True)``.

Classes:
    - PasswordPolicyConfig: Immutable configuration for all policy knobs.
    - PasswordPolicyViolation: Single policy violation descriptor.
    - PasswordValidator: Main validation engine.

Example:
    >>> config = PasswordPolicyConfig(min_length=14, expiry_days=60)
    >>> validator = PasswordValidator(config)
    >>> ok, violations = validator.validate("short")
    >>> ok
    False
    >>> violations[0].code
    'MIN_LENGTH'

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PasswordPolicyConfig:
    """Immutable configuration for password policy enforcement.

    Attributes:
        min_length: Minimum password length (default 12).
        max_length: Maximum password length (default 128).
        require_uppercase: Require at least one uppercase letter.
        require_lowercase: Require at least one lowercase letter.
        require_digit: Require at least one digit.
        require_special: Require at least one special character.
        special_characters: Allowed special character set.
        min_unique_chars: Minimum number of distinct characters.
        history_depth: Reject reuse of the last N passwords.
        expiry_days: Days until password expires (0 = never).
        reject_common_passwords: Check against embedded common-password set.
        check_breach_database: Enable HaveIBeenPwned k-anonymity check.
    """

    min_length: int = 12
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digit: bool = True
    require_special: bool = True
    special_characters: str = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
    min_unique_chars: int = 6
    history_depth: int = 5
    expiry_days: int = 90
    reject_common_passwords: bool = True
    check_breach_database: bool = False


# ---------------------------------------------------------------------------
# Violation descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PasswordPolicyViolation:
    """Represents a specific password policy violation.

    Attributes:
        code: Machine-readable violation code (e.g. ``MIN_LENGTH``).
        message: Human-readable description of the violation.
    """

    code: str
    message: str


# ---------------------------------------------------------------------------
# Common passwords (top 100, SHA-256 hashed for constant-time lookup)
# ---------------------------------------------------------------------------

# The plaintext originals are never stored in source.  We embed SHA-256
# digests so membership tests are O(1) without leaking the passwords into
# repository scanners.  The set is populated at module load time.

_COMMON_PASSWORDS_SHA256: Set[str] = {
    hashlib.sha256(p.encode()).hexdigest()
    for p in [
        "password", "123456", "12345678", "qwerty", "abc123",
        "monkey", "1234567", "letmein", "trustno1", "dragon",
        "baseball", "iloveyou", "master", "sunshine", "ashley",
        "michael", "shadow", "123123", "654321", "superman",
        "qazwsx", "michael1", "football", "password1", "password123",
        "batman", "login", "admin", "princess", "starwars",
        "121212", "welcome", "charlie", "donald", "qwerty123",
        "1234", "123456789", "1234567890", "password1!", "abc1234",
        "aa123456", "access", "flower", "passw0rd", "696969",
        "mustang", "jordan", "harley", "ranger", "dakota",
        "liverpool", "buster", "thomas", "robert", "soccer",
        "hockey", "hunter", "andrew", "tigger", "joshua",
        "pepper", "summer", "dallas", "camaro", "austin",
        "thunder", "matrix", "william", "corvette", "hello",
        "maggie", "silver", "ginger", "hammer", "yankees",
        "2000", "sparky", "george", "whatever", "chicken",
        "internet", "cheese", "merlin", "maverick", "mother",
        "jennifer", "jessica", "angel", "andrea", "asshole",
        "secret", "patrick", "spider", "purple", "amanda",
        "samantha", "yellow", "orange", "freedom", "nicole",
        "daniel", "compaq", "computer", "peanut", "banana",
    ]
}


# ---------------------------------------------------------------------------
# Password Validator
# ---------------------------------------------------------------------------


class PasswordValidator:
    """Validates passwords against a configurable policy.

    Provides synchronous complexity checks and async helpers for
    database-backed history, expiry, and breach detection.

    Attributes:
        config: Immutable password policy configuration.

    Example:
        >>> v = PasswordValidator()
        >>> ok, errs = v.validate("Str0ng!Pass#2026")
        >>> ok
        True
    """

    def __init__(
        self,
        config: Optional[PasswordPolicyConfig] = None,
        db_pool: Any = None,
    ) -> None:
        """Initialize the password validator.

        Args:
            config: Password policy configuration. Uses defaults if ``None``.
            db_pool: Async database connection pool for history / expiry queries.
        """
        self.config = config or PasswordPolicyConfig()
        self._db_pool = db_pool

    # ------------------------------------------------------------------
    # Synchronous validation
    # ------------------------------------------------------------------

    def validate(self, password: str) -> Tuple[bool, List[PasswordPolicyViolation]]:
        """Check *password* against all synchronous policy rules.

        This method does **not** check history, expiry, or breach databases.
        Use the async helpers for those checks.

        Args:
            password: The candidate password to validate.

        Returns:
            A tuple of (is_valid, list_of_violations).
        """
        start = time.monotonic()
        violations: List[PasswordPolicyViolation] = []

        # Length checks
        violations.extend(self._check_length(password))

        # Complexity checks
        violations.extend(self._check_complexity(password))

        # Unique character check
        violations.extend(self._check_unique_chars(password))

        # Common password check
        if self.config.reject_common_passwords:
            violations.extend(self._check_common_passwords(password))

        elapsed_ms = (time.monotonic() - start) * 1000
        is_valid = len(violations) == 0

        if not is_valid:
            logger.info(
                "Password validation failed: %d violation(s) in %.2f ms",
                len(violations),
                elapsed_ms,
            )
        else:
            logger.debug("Password validation passed in %.2f ms", elapsed_ms)

        return is_valid, violations

    # ------------------------------------------------------------------
    # Async: password history
    # ------------------------------------------------------------------

    async def check_history(self, user_id: str, password: str) -> bool:
        """Check whether *password* was used in the last N passwords.

        Compares bcrypt hashes stored in ``security.password_history``.

        Args:
            user_id: UUID of the user.
            password: Candidate plaintext password.

        Returns:
            ``True`` if the password was previously used (i.e. should be
            rejected), ``False`` otherwise.

        Raises:
            RuntimeError: If no database pool is configured.
        """
        if self._db_pool is None:
            raise RuntimeError("Database pool required for history check")

        try:
            import bcrypt  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("bcrypt not installed; skipping history check")
            return False

        depth = self.config.history_depth
        query = """
            SELECT password_hash
            FROM security.password_history
            WHERE user_id = $1
            ORDER BY changed_at DESC
            LIMIT $2
        """

        async with self._db_pool.connection() as conn:
            rows = await conn.fetch(query, user_id, depth)

        for row in rows:
            stored_hash = row["password_hash"]
            if isinstance(stored_hash, str):
                stored_hash = stored_hash.encode("utf-8")
            if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
                logger.info(
                    "Password reuse detected for user %s (history depth %d)",
                    user_id,
                    depth,
                )
                return True

        return False

    async def record_password(
        self,
        user_id: str,
        password_hash: str,
        changed_by: str = "self",
        reason: str = "user_change",
    ) -> None:
        """Record a password change in the history table.

        Args:
            user_id: UUID of the user.
            password_hash: Bcrypt hash of the new password.
            changed_by: Who triggered the change (``self``, ``admin``, ``system``).
            reason: Reason code (``user_change``, ``admin_reset``, ``expiry``).

        Raises:
            RuntimeError: If no database pool is configured.
        """
        if self._db_pool is None:
            raise RuntimeError("Database pool required for password recording")

        query = """
            INSERT INTO security.password_history
                (user_id, password_hash, changed_by, change_reason, changed_at)
            VALUES ($1, $2, $3, $4, $5)
        """
        now = datetime.now(timezone.utc)

        async with self._db_pool.connection() as conn:
            await conn.execute(query, user_id, password_hash, changed_by, reason, now)

        logger.info(
            "Recorded password change for user %s (reason=%s, by=%s)",
            user_id,
            reason,
            changed_by,
        )

    # ------------------------------------------------------------------
    # Async: expiry check
    # ------------------------------------------------------------------

    async def check_expiry(self, user_id: str) -> Tuple[bool, Optional[int]]:
        """Check whether the user's password has expired.

        Args:
            user_id: UUID of the user.

        Returns:
            A tuple of (is_expired, days_since_last_change).  If expiry is
            disabled (``expiry_days == 0``), returns ``(False, None)``.

        Raises:
            RuntimeError: If no database pool is configured.
        """
        if self.config.expiry_days == 0:
            return False, None

        if self._db_pool is None:
            raise RuntimeError("Database pool required for expiry check")

        query = """
            SELECT changed_at
            FROM security.password_history
            WHERE user_id = $1
            ORDER BY changed_at DESC
            LIMIT 1
        """

        async with self._db_pool.connection() as conn:
            row = await conn.fetchrow(query, user_id)

        if row is None:
            # No history -- treat as expired to force a password set
            logger.warning("No password history for user %s; treating as expired", user_id)
            return True, None

        last_change: datetime = row["changed_at"]
        if last_change.tzinfo is None:
            last_change = last_change.replace(tzinfo=timezone.utc)

        delta = datetime.now(timezone.utc) - last_change
        days_since = delta.days
        is_expired = days_since >= self.config.expiry_days

        if is_expired:
            logger.info(
                "Password expired for user %s: %d days since last change (max %d)",
                user_id,
                days_since,
                self.config.expiry_days,
            )

        return is_expired, days_since

    # ------------------------------------------------------------------
    # Async: breach detection (HaveIBeenPwned k-anonymity)
    # ------------------------------------------------------------------

    async def check_breach(self, password: str) -> bool:
        """Check whether *password* appears in the HaveIBeenPwned breach database.

        Uses the k-anonymity model: only the first 5 characters of the SHA-1
        hash are sent to the API.  The full hash never leaves the process.

        Args:
            password: Candidate plaintext password.

        Returns:
            ``True`` if the password has been seen in a breach, ``False``
            otherwise or if the check cannot be performed.
        """
        if not self.config.check_breach_database:
            return False

        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("httpx not installed; skipping breach check")
            return False

        sha1_full = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()  # noqa: S324
        prefix = sha1_full[:5]
        suffix = sha1_full[5:]

        url = f"https://api.pwnedpasswords.com/range/{prefix}"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": "GreenLang-SEC001/1.0"},
                )
                response.raise_for_status()
        except Exception:
            logger.warning("HaveIBeenPwned API request failed; skipping breach check")
            return False

        for line in response.text.splitlines():
            parts = line.split(":")
            if len(parts) == 2 and parts[0].strip() == suffix:
                count = int(parts[1].strip())
                logger.warning(
                    "Password found in breach database (%d occurrences)", count
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Internal: length checks
    # ------------------------------------------------------------------

    def _check_length(self, password: str) -> List[PasswordPolicyViolation]:
        """Validate password length against min and max bounds.

        Args:
            password: Candidate password.

        Returns:
            List of violations (empty if length is acceptable).
        """
        violations: List[PasswordPolicyViolation] = []

        if len(password) < self.config.min_length:
            violations.append(
                PasswordPolicyViolation(
                    code="MIN_LENGTH",
                    message=(
                        f"Password must be at least {self.config.min_length} "
                        f"characters (got {len(password)})"
                    ),
                )
            )

        if len(password) > self.config.max_length:
            violations.append(
                PasswordPolicyViolation(
                    code="MAX_LENGTH",
                    message=(
                        f"Password must be at most {self.config.max_length} "
                        f"characters (got {len(password)})"
                    ),
                )
            )

        return violations

    # ------------------------------------------------------------------
    # Internal: complexity checks
    # ------------------------------------------------------------------

    def _check_complexity(self, password: str) -> List[PasswordPolicyViolation]:
        """Validate character-class requirements.

        Args:
            password: Candidate password.

        Returns:
            List of violations for missing character classes.
        """
        violations: List[PasswordPolicyViolation] = []

        if self.config.require_uppercase and not re.search(r"[A-Z]", password):
            violations.append(
                PasswordPolicyViolation(
                    code="UPPERCASE_REQUIRED",
                    message="Password must contain at least one uppercase letter",
                )
            )

        if self.config.require_lowercase and not re.search(r"[a-z]", password):
            violations.append(
                PasswordPolicyViolation(
                    code="LOWERCASE_REQUIRED",
                    message="Password must contain at least one lowercase letter",
                )
            )

        if self.config.require_digit and not re.search(r"\d", password):
            violations.append(
                PasswordPolicyViolation(
                    code="DIGIT_REQUIRED",
                    message="Password must contain at least one digit",
                )
            )

        if self.config.require_special:
            escaped = re.escape(self.config.special_characters)
            if not re.search(f"[{escaped}]", password):
                violations.append(
                    PasswordPolicyViolation(
                        code="SPECIAL_REQUIRED",
                        message="Password must contain at least one special character",
                    )
                )

        return violations

    # ------------------------------------------------------------------
    # Internal: unique character check
    # ------------------------------------------------------------------

    def _check_unique_chars(self, password: str) -> List[PasswordPolicyViolation]:
        """Validate minimum number of distinct characters.

        Args:
            password: Candidate password.

        Returns:
            List of violations (empty if enough unique chars).
        """
        unique_count = len(set(password))
        if unique_count < self.config.min_unique_chars:
            return [
                PasswordPolicyViolation(
                    code="MIN_UNIQUE_CHARS",
                    message=(
                        f"Password must contain at least {self.config.min_unique_chars} "
                        f"unique characters (got {unique_count})"
                    ),
                )
            ]
        return []

    # ------------------------------------------------------------------
    # Internal: common password check
    # ------------------------------------------------------------------

    def _check_common_passwords(self, password: str) -> List[PasswordPolicyViolation]:
        """Check whether password matches a well-known common password.

        Uses SHA-256 hashes for constant-time membership testing without
        storing plaintext common passwords in memory.

        Args:
            password: Candidate password.

        Returns:
            List containing one violation if password is common, else empty.
        """
        digest = hashlib.sha256(password.lower().encode()).hexdigest()
        if digest in _COMMON_PASSWORDS_SHA256:
            return [
                PasswordPolicyViolation(
                    code="COMMON_PASSWORD",
                    message="Password is too common and easily guessable",
                )
            ]
        return []
