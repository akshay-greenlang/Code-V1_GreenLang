# -*- coding: utf-8 -*-
"""
Unit tests for PasswordValidator - JWT Authentication Service (SEC-001)

Tests password complexity requirements, history tracking, expiry policies,
common password detection, and breach database lookups (HaveIBeenPwned
k-anonymity API).

Coverage targets: 85%+ of password_policy.py
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.auth_service.password_policy import (
    PasswordPolicyConfig,
    PasswordPolicyViolation,
    PasswordValidator,
    _COMMON_PASSWORDS_SHA256,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_db_pool(
    fetch_return: Optional[list] = None,
    fetchrow_return: Any = None,
    execute_return: str = "INSERT 0 1",
) -> tuple:
    """Create a mock async database pool."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=execute_return)
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.fetch = AsyncMock(return_value=fetch_return or [])

    pool = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool.connection.return_value = cm

    return pool, conn


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config() -> PasswordPolicyConfig:
    return PasswordPolicyConfig()


@pytest.fixture
def strict_config() -> PasswordPolicyConfig:
    return PasswordPolicyConfig(
        min_length=16,
        max_length=64,
        min_unique_chars=10,
        history_depth=10,
        expiry_days=30,
    )


@pytest.fixture
def relaxed_config() -> PasswordPolicyConfig:
    return PasswordPolicyConfig(
        min_length=8,
        require_uppercase=False,
        require_lowercase=False,
        require_digit=False,
        require_special=False,
        min_unique_chars=1,
        reject_common_passwords=False,
    )


@pytest.fixture
def validator(default_config: PasswordPolicyConfig) -> PasswordValidator:
    return PasswordValidator(config=default_config)


@pytest.fixture
def strict_validator(strict_config: PasswordPolicyConfig) -> PasswordValidator:
    return PasswordValidator(config=strict_config)


@pytest.fixture
def relaxed_validator(relaxed_config: PasswordPolicyConfig) -> PasswordValidator:
    return PasswordValidator(config=relaxed_config)


@pytest.fixture
def db_pool_and_conn():
    return _make_db_pool()


@pytest.fixture
def db_pool(db_pool_and_conn):
    pool, _ = db_pool_and_conn
    return pool


@pytest.fixture
def db_conn(db_pool_and_conn):
    _, conn = db_pool_and_conn
    return conn


@pytest.fixture
def validator_with_db(default_config, db_pool) -> PasswordValidator:
    return PasswordValidator(config=default_config, db_pool=db_pool)


# Strong password that passes default policy
STRONG_PASSWORD = "My$tr0ng!Pass#2026"


# ============================================================================
# TestPasswordPolicyConfig
# ============================================================================


class TestPasswordPolicyConfig:
    """Tests for the immutable PasswordPolicyConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config has expected values."""
        config = PasswordPolicyConfig()
        assert config.min_length == 12
        assert config.max_length == 128
        assert config.require_uppercase is True
        assert config.require_lowercase is True
        assert config.require_digit is True
        assert config.require_special is True
        assert config.min_unique_chars == 6
        assert config.history_depth == 5
        assert config.expiry_days == 90
        assert config.reject_common_passwords is True
        assert config.check_breach_database is False

    def test_frozen_immutable(self) -> None:
        """Config is frozen and cannot be mutated."""
        config = PasswordPolicyConfig()
        with pytest.raises(AttributeError):
            config.min_length = 20  # type: ignore[misc]


# ============================================================================
# TestPasswordPolicyViolation
# ============================================================================


class TestPasswordPolicyViolation:
    """Tests for the PasswordPolicyViolation dataclass."""

    def test_create_violation(self) -> None:
        """Violation stores code and message."""
        v = PasswordPolicyViolation(code="MIN_LENGTH", message="Too short")
        assert v.code == "MIN_LENGTH"
        assert v.message == "Too short"

    def test_frozen_immutable(self) -> None:
        """Violation is frozen."""
        v = PasswordPolicyViolation(code="X", message="Y")
        with pytest.raises(AttributeError):
            v.code = "Z"  # type: ignore[misc]


# ============================================================================
# TestPasswordValidator
# ============================================================================


class TestPasswordValidator:
    """Tests for synchronous password validation."""

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_valid_password_passes(self, validator: PasswordValidator) -> None:
        """A strong password passes all checks."""
        ok, violations = validator.validate(STRONG_PASSWORD)
        assert ok is True
        assert violations == []

    def test_valid_password_alternate(self, validator: PasswordValidator) -> None:
        """Another strong password passes."""
        ok, violations = validator.validate("C0mpl3x!Passw@rd")
        assert ok is True
        assert violations == []

    # ------------------------------------------------------------------
    # Length checks
    # ------------------------------------------------------------------

    def test_too_short_fails(self, validator: PasswordValidator) -> None:
        """Password below min_length produces MIN_LENGTH violation."""
        ok, violations = validator.validate("Sh0rt!")
        assert ok is False
        codes = [v.code for v in violations]
        assert "MIN_LENGTH" in codes

    def test_exact_min_length_passes(self, validator: PasswordValidator) -> None:
        """Password at exactly min_length passes the length check."""
        # 12 chars with all requirements
        pwd = "Abcde1!ghijk"
        ok, violations = validator.validate(pwd)
        min_length_violations = [v for v in violations if v.code == "MIN_LENGTH"]
        assert len(min_length_violations) == 0

    def test_max_length_enforced(self, validator: PasswordValidator) -> None:
        """Password exceeding max_length produces MAX_LENGTH violation."""
        pwd = "A1!" + "a" * 130
        ok, violations = validator.validate(pwd)
        assert ok is False
        codes = [v.code for v in violations]
        assert "MAX_LENGTH" in codes

    # ------------------------------------------------------------------
    # Complexity checks
    # ------------------------------------------------------------------

    def test_no_uppercase_fails(self, validator: PasswordValidator) -> None:
        """Missing uppercase letter produces UPPERCASE_REQUIRED."""
        ok, violations = validator.validate("lowercase1!only")
        assert ok is False
        codes = [v.code for v in violations]
        assert "UPPERCASE_REQUIRED" in codes

    def test_no_lowercase_fails(self, validator: PasswordValidator) -> None:
        """Missing lowercase letter produces LOWERCASE_REQUIRED."""
        ok, violations = validator.validate("UPPERCASE1!ONLY")
        assert ok is False
        codes = [v.code for v in violations]
        assert "LOWERCASE_REQUIRED" in codes

    def test_no_digit_fails(self, validator: PasswordValidator) -> None:
        """Missing digit produces DIGIT_REQUIRED."""
        ok, violations = validator.validate("NoDigits!Here")
        assert ok is False
        codes = [v.code for v in violations]
        assert "DIGIT_REQUIRED" in codes

    def test_no_special_char_fails(self, validator: PasswordValidator) -> None:
        """Missing special character produces SPECIAL_REQUIRED."""
        ok, violations = validator.validate("NoSpecial1Char")
        assert ok is False
        codes = [v.code for v in violations]
        assert "SPECIAL_REQUIRED" in codes

    def test_multiple_violations_returned(
        self, validator: PasswordValidator
    ) -> None:
        """Multiple violations are all returned at once."""
        ok, violations = validator.validate("x")
        assert ok is False
        codes = {v.code for v in violations}
        # At minimum: too short, missing uppercase, missing digit, missing special
        assert len(codes) >= 3

    # ------------------------------------------------------------------
    # Unique character check
    # ------------------------------------------------------------------

    def test_min_unique_chars(self, validator: PasswordValidator) -> None:
        """Password with too few unique characters is rejected."""
        # All same char repeated
        ok, violations = validator.validate("aaaaaaaaaaaa")
        assert ok is False
        codes = [v.code for v in violations]
        assert "MIN_UNIQUE_CHARS" in codes

    def test_enough_unique_chars_passes(
        self, validator: PasswordValidator
    ) -> None:
        """Password with enough unique characters passes this check."""
        pwd = "Abcdef1!ghij"
        ok, violations = validator.validate(pwd)
        unique_violations = [v for v in violations if v.code == "MIN_UNIQUE_CHARS"]
        assert len(unique_violations) == 0

    # ------------------------------------------------------------------
    # Common password check
    # ------------------------------------------------------------------

    def test_common_password_rejected(
        self, validator: PasswordValidator
    ) -> None:
        """Known common passwords are rejected."""
        ok, violations = validator.validate("password")
        assert ok is False
        codes = [v.code for v in violations]
        assert "COMMON_PASSWORD" in codes

    def test_common_password_case_insensitive(
        self, validator: PasswordValidator
    ) -> None:
        """Common password check is case-insensitive."""
        ok, violations = validator.validate("PASSWORD")
        codes = [v.code for v in violations]
        assert "COMMON_PASSWORD" in codes

    def test_non_common_password_allowed(
        self, validator: PasswordValidator
    ) -> None:
        """Non-common password is not flagged."""
        ok, violations = validator.validate(STRONG_PASSWORD)
        codes = [v.code for v in violations]
        assert "COMMON_PASSWORD" not in codes

    def test_common_password_check_disabled(
        self, relaxed_validator: PasswordValidator
    ) -> None:
        """When reject_common_passwords is False, check is skipped."""
        ok, violations = relaxed_validator.validate("password1234")
        codes = [v.code for v in violations]
        assert "COMMON_PASSWORD" not in codes

    # ------------------------------------------------------------------
    # Relaxed config
    # ------------------------------------------------------------------

    def test_relaxed_config_allows_simple_password(
        self, relaxed_validator: PasswordValidator
    ) -> None:
        """Relaxed config allows a simple password."""
        ok, violations = relaxed_validator.validate("simplepassword")
        assert ok is True
        assert violations == []


# ============================================================================
# TestPasswordHistory
# ============================================================================


class TestPasswordHistory:
    """Tests for async password history checking."""

    @pytest.mark.asyncio
    async def test_check_history_no_db_raises(
        self, validator: PasswordValidator
    ) -> None:
        """check_history without DB raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Database pool"):
            await validator.check_history("u-1", "password")

    @pytest.mark.asyncio
    async def test_check_history_detects_reuse(
        self, validator_with_db: PasswordValidator, db_conn
    ) -> None:
        """check_history returns True when password matches a stored hash."""
        with patch(
            "greenlang.infrastructure.auth_service.password_policy.bcrypt"
        ) as mock_bcrypt:
            mock_bcrypt.checkpw.return_value = True
            db_conn.fetch.return_value = [
                {"password_hash": b"$2b$12$stored_hash_bytes"}
            ]
            result = await validator_with_db.check_history("u-1", "reused-pwd")
            assert result is True

    @pytest.mark.asyncio
    async def test_check_history_allows_new_password(
        self, validator_with_db: PasswordValidator, db_conn
    ) -> None:
        """check_history returns False when password is not in history."""
        with patch(
            "greenlang.infrastructure.auth_service.password_policy.bcrypt"
        ) as mock_bcrypt:
            mock_bcrypt.checkpw.return_value = False
            db_conn.fetch.return_value = [
                {"password_hash": b"$2b$12$stored_hash"}
            ]
            result = await validator_with_db.check_history("u-1", "brand-new-pwd")
            assert result is False

    @pytest.mark.asyncio
    async def test_check_history_empty_history(
        self, validator_with_db: PasswordValidator, db_conn
    ) -> None:
        """check_history returns False when no history exists."""
        db_conn.fetch.return_value = []
        result = await validator_with_db.check_history("u-1", "any-pwd")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_history_respects_depth(
        self, validator_with_db: PasswordValidator, db_conn
    ) -> None:
        """check_history queries with the configured history_depth."""
        with patch(
            "greenlang.infrastructure.auth_service.password_policy.bcrypt"
        ) as mock_bcrypt:
            mock_bcrypt.checkpw.return_value = False
            db_conn.fetch.return_value = []
            await validator_with_db.check_history("u-1", "pwd")
            call_args = db_conn.fetch.call_args
            assert call_args.args[2] == 5  # default history_depth

    @pytest.mark.asyncio
    async def test_record_password(
        self, validator_with_db: PasswordValidator, db_conn
    ) -> None:
        """record_password inserts a row into the history table."""
        await validator_with_db.record_password(
            user_id="u-1",
            password_hash="$2b$12$hashed_value",
            changed_by="self",
            reason="user_change",
        )
        db_conn.execute.assert_awaited_once()
        sql = db_conn.execute.call_args.args[0]
        assert "INSERT INTO security.password_history" in sql

    @pytest.mark.asyncio
    async def test_record_password_no_db_raises(
        self, validator: PasswordValidator
    ) -> None:
        """record_password without DB raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Database pool"):
            await validator.record_password("u-1", "$2b$12$hash")

    @pytest.mark.asyncio
    async def test_history_depth_limited(
        self, validator_with_db: PasswordValidator, db_conn
    ) -> None:
        """check_history LIMIT clause matches history_depth config."""
        with patch(
            "greenlang.infrastructure.auth_service.password_policy.bcrypt"
        ) as mock_bcrypt:
            mock_bcrypt.checkpw.return_value = False
            db_conn.fetch.return_value = []
            await validator_with_db.check_history("u-1", "pwd")
            sql = db_conn.fetch.call_args.args[0]
            assert "LIMIT" in sql


# ============================================================================
# TestPasswordExpiry
# ============================================================================


class TestPasswordExpiry:
    """Tests for async password expiry checking."""

    @pytest.mark.asyncio
    async def test_expired_password_detected(
        self, validator_with_db: PasswordValidator, db_conn
    ) -> None:
        """check_expiry returns (True, days) when password has expired."""
        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        db_conn.fetchrow.return_value = {"changed_at": old_date}
        is_expired, days = await validator_with_db.check_expiry("u-1")
        assert is_expired is True
        assert days >= 100

    @pytest.mark.asyncio
    async def test_fresh_password_not_expired(
        self, validator_with_db: PasswordValidator, db_conn
    ) -> None:
        """check_expiry returns (False, days) for a fresh password."""
        recent = datetime.now(timezone.utc) - timedelta(days=10)
        db_conn.fetchrow.return_value = {"changed_at": recent}
        is_expired, days = await validator_with_db.check_expiry("u-1")
        assert is_expired is False
        assert days == 10

    @pytest.mark.asyncio
    async def test_no_expiry_when_disabled(self, db_pool) -> None:
        """check_expiry returns (False, None) when expiry_days is 0."""
        config = PasswordPolicyConfig(expiry_days=0)
        validator = PasswordValidator(config=config, db_pool=db_pool)
        is_expired, days = await validator.check_expiry("u-1")
        assert is_expired is False
        assert days is None

    @pytest.mark.asyncio
    async def test_no_history_treated_as_expired(
        self, validator_with_db: PasswordValidator, db_conn
    ) -> None:
        """When no history exists, password is treated as expired."""
        db_conn.fetchrow.return_value = None
        is_expired, days = await validator_with_db.check_expiry("u-1")
        assert is_expired is True
        assert days is None

    @pytest.mark.asyncio
    async def test_expiry_no_db_raises(
        self, validator: PasswordValidator
    ) -> None:
        """check_expiry without DB raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Database pool"):
            await validator.check_expiry("u-1")

    @pytest.mark.asyncio
    async def test_naive_datetime_treated_as_utc(
        self, validator_with_db: PasswordValidator, db_conn
    ) -> None:
        """Naive datetime from DB is treated as UTC."""
        naive_dt = datetime(2025, 1, 1, 0, 0, 0)
        db_conn.fetchrow.return_value = {"changed_at": naive_dt}
        is_expired, days = await validator_with_db.check_expiry("u-1")
        assert is_expired is True  # Jan 2025 is >90 days ago


# ============================================================================
# TestBreachDetection
# ============================================================================


class TestBreachDetection:
    """Tests for async HaveIBeenPwned breach detection."""

    @pytest.mark.asyncio
    async def test_breach_check_disabled(
        self, validator: PasswordValidator
    ) -> None:
        """check_breach returns False when check_breach_database is False."""
        result = await validator.check_breach("anything")
        assert result is False

    @pytest.mark.asyncio
    async def test_breach_found(self) -> None:
        """check_breach returns True when password appears in breach DB."""
        config = PasswordPolicyConfig(check_breach_database=True)
        v = PasswordValidator(config=config)
        # Mock httpx
        sha1_full = hashlib.sha1(b"breached").hexdigest().upper()
        suffix = sha1_full[5:]

        mock_response = MagicMock()
        mock_response.text = f"{suffix}:42\nOTHER123:1"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "greenlang.infrastructure.auth_service.password_policy.httpx"
        ) as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            result = await v.check_breach("breached")
            assert result is True

    @pytest.mark.asyncio
    async def test_breach_not_found(self) -> None:
        """check_breach returns False when password is not in breach DB."""
        config = PasswordPolicyConfig(check_breach_database=True)
        v = PasswordValidator(config=config)

        mock_response = MagicMock()
        mock_response.text = "AAAAABBBBB:1\nCCCCCDDDDD:2"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "greenlang.infrastructure.auth_service.password_policy.httpx"
        ) as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            result = await v.check_breach("unique-strong-password-xyz-123")
            assert result is False

    @pytest.mark.asyncio
    async def test_breach_api_failure_returns_false(self) -> None:
        """check_breach returns False when the API request fails."""
        config = PasswordPolicyConfig(check_breach_database=True)
        v = PasswordValidator(config=config)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Network error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "greenlang.infrastructure.auth_service.password_policy.httpx"
        ) as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            result = await v.check_breach("some-password")
            assert result is False


# ============================================================================
# TestCommonPasswordsSet
# ============================================================================


class TestCommonPasswordsSet:
    """Tests for the module-level _COMMON_PASSWORDS_SHA256 set."""

    def test_common_passwords_set_populated(self) -> None:
        """The set is non-empty at module load time."""
        assert len(_COMMON_PASSWORDS_SHA256) > 50

    def test_known_password_in_set(self) -> None:
        """'password' (lowercased, sha256-hashed) is in the set."""
        digest = hashlib.sha256(b"password").hexdigest()
        assert digest in _COMMON_PASSWORDS_SHA256

    def test_strong_password_not_in_set(self) -> None:
        """A strong password is NOT in the common set."""
        digest = hashlib.sha256(STRONG_PASSWORD.lower().encode()).hexdigest()
        assert digest not in _COMMON_PASSWORDS_SHA256
