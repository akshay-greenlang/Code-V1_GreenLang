"""
Comprehensive input validation framework.

This module provides centralized input validation to prevent:
- SQL injection
- Command injection
- XSS (Cross-Site Scripting)
- Path traversal
- SSRF (Server-Side Request Forgery)
- LDAP injection
- NoSQL injection

All validation uses whitelist-based approach (not blacklist) for maximum security.

Example:
    >>> from security.input_validation import InputValidator
    >>> validator = InputValidator()
    >>> safe_id = validator.validate_uuid("123e4567-e89b-12d3-a456-426614174000", "user_id")
    >>> safe_field = validator.validate_field_name("tenant_id")
"""

import re
import html
import json
from typing import Any, Optional, List, Dict, Set, Union
from pathlib import Path
from urllib.parse import urlparse
from pydantic import BaseModel, validator, Field, constr, conint
import ipaddress
import logging

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Centralized input validation for security-critical operations.

    This class provides static methods for validating various input types
    against injection attacks and malformed data. All methods use whitelist
    approaches where possible.

    Attributes:
        ALPHANUMERIC: Regex pattern for alphanumeric + underscore/hyphen
        EMAIL: Regex pattern for email validation
        UUID: Regex pattern for UUID validation
        SQL_INJECTION: Regex pattern to detect SQL injection attempts
        COMMAND_INJECTION: Regex pattern to detect shell injection attempts
        PATH_TRAVERSAL: Regex pattern to detect path traversal attempts
        LDAP_INJECTION: Regex pattern to detect LDAP injection attempts
        NOSQL_INJECTION: Regex pattern to detect NoSQL injection attempts

    Example:
        >>> validator = InputValidator()
        >>> tenant_id = validator.validate_alphanumeric("tenant-123", "tenant_id")
        >>> email = validator.validate_email("user@example.com")
    """

    # Regex patterns for validation
    ALPHANUMERIC = re.compile(r'^[a-zA-Z0-9_-]+$')
    EMAIL = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    UUID = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$', re.IGNORECASE)

    # Injection detection patterns (for logging/monitoring)
    SQL_INJECTION = re.compile(
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|DECLARE|CAST|CONVERT|CHAR|CHR)\b|--|\#|\/\*|\*\/|;|'|\")",
        re.IGNORECASE
    )
    COMMAND_INJECTION = re.compile(r'[;&|`$(){}[\]<>\n\r]')
    PATH_TRAVERSAL = re.compile(r'(\.\.|/etc/|/var/|/usr/|/root/|/proc/|C:\\|\\\\|%00|%2e%2e)')
    LDAP_INJECTION = re.compile(r'[*()\\\x00]')
    NOSQL_INJECTION = re.compile(r'(\$where|\$ne|\$gt|\$lt|\$regex|\$or|\$and)', re.IGNORECASE)

    # XSS patterns
    XSS_PATTERNS = re.compile(
        r'(<script|javascript:|onerror=|onload=|onclick=|<iframe|<object|<embed)',
        re.IGNORECASE
    )

    # Whitelists for security-critical fields
    ALLOWED_FIELDS: Set[str] = {
        'tenant_id', 'user_id', 'agent_id', 'execution_id', 'task_id', 'workflow_id',
        'name', 'email', 'status', 'type', 'tier', 'role', 'scope', 'version',
        'created_at', 'updated_at', 'created_by', 'updated_by',
        'title', 'description', 'category', 'priority', 'severity',
    }

    ALLOWED_OPERATORS: Set[str] = {
        '=', '!=', '>', '<', '>=', '<=', 'IN', 'NOT IN', 'LIKE', 'ILIKE', 'IS NULL', 'IS NOT NULL'
    }

    ALLOWED_SORT_DIRECTIONS: Set[str] = {'ASC', 'DESC'}

    ALLOWED_AGGREGATIONS: Set[str] = {
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'VARIANCE'
    }

    @staticmethod
    def validate_alphanumeric(value: str, field_name: str, min_length: int = 1, max_length: int = 255) -> str:
        """
        Validate string contains only alphanumeric characters, underscore, and hyphen.

        Args:
            value: String to validate
            field_name: Name of field for error messages
            min_length: Minimum allowed length
            max_length: Maximum allowed length

        Returns:
            Validated string

        Raises:
            ValueError: If validation fails

        Example:
            >>> InputValidator.validate_alphanumeric("user-123", "user_id")
            'user-123'
        """
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be string, got {type(value).__name__}")

        if not (min_length <= len(value) <= max_length):
            raise ValueError(
                f"{field_name} length must be between {min_length} and {max_length}, got {len(value)}"
            )

        if not InputValidator.ALPHANUMERIC.match(value):
            raise ValueError(
                f"{field_name} must contain only alphanumeric characters, underscore, and hyphen"
            )

        return value

    @staticmethod
    def validate_uuid(value: str, field_name: str) -> str:
        """
        Validate UUID format (RFC 4122).

        Args:
            value: UUID string to validate
            field_name: Name of field for error messages

        Returns:
            Validated UUID in lowercase

        Raises:
            ValueError: If not a valid UUID

        Example:
            >>> InputValidator.validate_uuid("123e4567-e89b-12d3-a456-426614174000", "user_id")
            '123e4567-e89b-12d3-a456-426614174000'
        """
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be string, got {type(value).__name__}")

        value_lower = value.lower()

        if not InputValidator.UUID.match(value_lower):
            raise ValueError(f"{field_name} must be valid UUID (RFC 4122)")

        return value_lower

    @staticmethod
    def validate_email(value: str) -> str:
        """
        Validate email format.

        Args:
            value: Email string to validate

        Returns:
            Validated email in lowercase

        Raises:
            ValueError: If not a valid email

        Example:
            >>> InputValidator.validate_email("user@example.com")
            'user@example.com'
        """
        if not isinstance(value, str):
            raise ValueError(f"Email must be string, got {type(value).__name__}")

        if len(value) > 255:
            raise ValueError(f"Email too long (max 255 characters)")

        value_lower = value.lower()

        if not InputValidator.EMAIL.match(value_lower):
            raise ValueError("Invalid email format")

        return value_lower

    @staticmethod
    def validate_no_sql_injection(value: str, field_name: str) -> str:
        """
        Check for SQL injection patterns and log suspicious input.

        This is a defense-in-depth measure. Primary defense is parameterized queries.

        Args:
            value: String to check
            field_name: Name of field for logging

        Returns:
            Original value if safe

        Raises:
            ValueError: If SQL injection pattern detected

        Example:
            >>> InputValidator.validate_no_sql_injection("normal_value", "description")
            'normal_value'
        """
        if not isinstance(value, str):
            return value  # Only validate strings

        if InputValidator.SQL_INJECTION.search(value):
            logger.warning(
                f"Potential SQL injection detected in {field_name}: {value[:100]}",
                extra={"field": field_name, "value_preview": value[:100]}
            )
            raise ValueError(
                f"{field_name} contains potentially dangerous SQL keywords or characters"
            )

        return value

    @staticmethod
    def validate_no_command_injection(value: str, field_name: str) -> str:
        """
        Check for command injection patterns.

        Args:
            value: String to check
            field_name: Name of field for logging

        Returns:
            Original value if safe

        Raises:
            ValueError: If command injection pattern detected

        Example:
            >>> InputValidator.validate_no_command_injection("safe_command", "cmd")
            'safe_command'
        """
        if not isinstance(value, str):
            return value

        if InputValidator.COMMAND_INJECTION.search(value):
            logger.warning(
                f"Potential command injection detected in {field_name}: {value[:100]}",
                extra={"field": field_name, "value_preview": value[:100]}
            )
            raise ValueError(
                f"{field_name} contains dangerous shell characters"
            )

        return value

    @staticmethod
    def validate_no_xss(value: str, field_name: str) -> str:
        """
        Check for XSS patterns.

        Args:
            value: String to check
            field_name: Name of field for logging

        Returns:
            Original value if safe

        Raises:
            ValueError: If XSS pattern detected
        """
        if not isinstance(value, str):
            return value

        if InputValidator.XSS_PATTERNS.search(value):
            logger.warning(
                f"Potential XSS detected in {field_name}: {value[:100]}",
                extra={"field": field_name, "value_preview": value[:100]}
            )
            raise ValueError(
                f"{field_name} contains potentially dangerous HTML/JavaScript"
            )

        return value

    @staticmethod
    def validate_path(
        value: str,
        must_exist: bool = False,
        allow_relative: bool = False,
        allowed_extensions: Optional[List[str]] = None
    ) -> Path:
        """
        Validate file path and prevent path traversal attacks.

        Args:
            value: Path string to validate
            must_exist: If True, path must exist
            allow_relative: If True, allow relative paths
            allowed_extensions: List of allowed file extensions (e.g., ['.json', '.yaml'])

        Returns:
            Validated Path object

        Raises:
            ValueError: If path validation fails

        Example:
            >>> InputValidator.validate_path("C:/data/file.json", allowed_extensions=['.json'])
            WindowsPath('C:/data/file.json')
        """
        if not isinstance(value, str):
            raise ValueError("Path must be string")

        if len(value) > 4096:
            raise ValueError("Path too long (max 4096 characters)")

        # Check for path traversal patterns
        if InputValidator.PATH_TRAVERSAL.search(value):
            logger.warning(
                f"Path traversal attempt detected: {value}",
                extra={"path": value}
            )
            raise ValueError("Path contains dangerous traversal patterns")

        try:
            path = Path(value)
        except Exception as e:
            raise ValueError(f"Invalid path format: {e}")

        # Check if absolute path required
        if not allow_relative and not path.is_absolute():
            raise ValueError("Path must be absolute")

        # Check if path exists
        if must_exist and not path.exists():
            raise ValueError(f"Path does not exist: {value}")

        # Check file extension if specified
        if allowed_extensions and path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValueError(
                f"File extension must be one of {allowed_extensions}, got {path.suffix}"
            )

        return path

    @staticmethod
    def validate_ip_address(value: str, allow_private: bool = False, allow_loopback: bool = False) -> str:
        """
        Validate IP address and prevent SSRF attacks.

        Args:
            value: IP address string
            allow_private: If True, allow private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
            allow_loopback: If True, allow loopback addresses (127.0.0.0/8)

        Returns:
            Validated IP address string

        Raises:
            ValueError: If IP validation fails

        Example:
            >>> InputValidator.validate_ip_address("8.8.8.8")
            '8.8.8.8'
        """
        try:
            ip = ipaddress.ip_address(value)
        except ValueError as e:
            raise ValueError(f"Invalid IP address: {value}") from e

        # Check for private IPs (SSRF prevention)
        if not allow_private and ip.is_private:
            logger.warning(
                f"Private IP address rejected: {value}",
                extra={"ip": value}
            )
            raise ValueError(f"Private IP addresses not allowed: {value}")

        # Check for loopback
        if not allow_loopback and ip.is_loopback:
            logger.warning(
                f"Loopback address rejected: {value}",
                extra={"ip": value}
            )
            raise ValueError(f"Loopback addresses not allowed: {value}")

        # Check for link-local
        if ip.is_link_local:
            raise ValueError(f"Link-local addresses not allowed: {value}")

        # Check for multicast
        if ip.is_multicast:
            raise ValueError(f"Multicast addresses not allowed: {value}")

        return str(ip)

    @staticmethod
    def validate_url(
        value: str,
        allowed_schemes: Optional[List[str]] = None,
        allow_private_ips: bool = False
    ) -> str:
        """
        Validate URL and prevent SSRF attacks.

        Args:
            value: URL string to validate
            allowed_schemes: List of allowed URL schemes (default: ['https'])
            allow_private_ips: If True, allow URLs with private IP addresses

        Returns:
            Validated URL string

        Raises:
            ValueError: If URL validation fails

        Example:
            >>> InputValidator.validate_url("https://api.example.com/data")
            'https://api.example.com/data'
        """
        if allowed_schemes is None:
            allowed_schemes = ['https']

        if not isinstance(value, str):
            raise ValueError("URL must be string")

        if len(value) > 2048:
            raise ValueError("URL too long (max 2048 characters)")

        try:
            parsed = urlparse(value)
        except Exception as e:
            raise ValueError(f"Invalid URL format: {e}")

        # Validate scheme
        if parsed.scheme not in allowed_schemes:
            raise ValueError(
                f"URL scheme must be one of {allowed_schemes}, got '{parsed.scheme}'"
            )

        # Validate hostname exists
        if not parsed.hostname:
            raise ValueError("URL must have hostname")

        # Check if hostname is IP address
        try:
            InputValidator.validate_ip_address(
                parsed.hostname,
                allow_private=allow_private_ips,
                allow_loopback=False
            )
        except ValueError:
            # Hostname is not an IP, which is OK (it's a domain)
            pass

        # Check for suspicious patterns in URL
        dangerous_patterns = ['file://', 'gopher://', 'dict://', 'ftp://']
        for pattern in dangerous_patterns:
            if pattern in value.lower():
                raise ValueError(f"URL contains dangerous protocol: {pattern}")

        return value

    @staticmethod
    def validate_field_name(field: str) -> str:
        """
        Validate database field name against whitelist.

        This is critical for preventing SQL injection in dynamic queries.

        Args:
            field: Field name to validate

        Returns:
            Validated field name

        Raises:
            ValueError: If field not in whitelist

        Example:
            >>> InputValidator.validate_field_name("tenant_id")
            'tenant_id'
        """
        if not isinstance(field, str):
            raise ValueError("Field name must be string")

        # Check whitelist first
        if field not in InputValidator.ALLOWED_FIELDS:
            logger.warning(
                f"Field name not in whitelist: {field}",
                extra={"field": field, "whitelist": list(InputValidator.ALLOWED_FIELDS)}
            )
            raise ValueError(
                f"Field '{field}' not in whitelist. Allowed fields: {sorted(InputValidator.ALLOWED_FIELDS)}"
            )

        # Additional check for alphanumeric (defense in depth)
        if not InputValidator.ALPHANUMERIC.match(field):
            raise ValueError(f"Invalid field name format: {field}")

        return field

    @staticmethod
    def validate_operator(operator: str) -> str:
        """
        Validate SQL operator against whitelist.

        Args:
            operator: SQL operator to validate

        Returns:
            Validated operator

        Raises:
            ValueError: If operator not in whitelist
        """
        operator_upper = operator.upper()

        if operator_upper not in InputValidator.ALLOWED_OPERATORS:
            raise ValueError(
                f"Operator '{operator}' not allowed. Allowed: {sorted(InputValidator.ALLOWED_OPERATORS)}"
            )

        return operator_upper

    @staticmethod
    def validate_sort_direction(direction: str) -> str:
        """
        Validate sort direction (ASC/DESC).

        Args:
            direction: Sort direction

        Returns:
            Validated direction

        Raises:
            ValueError: If direction not valid
        """
        direction_upper = direction.upper()

        if direction_upper not in InputValidator.ALLOWED_SORT_DIRECTIONS:
            raise ValueError(f"Sort direction must be ASC or DESC, got '{direction}'")

        return direction_upper

    @staticmethod
    def sanitize_html(value: str) -> str:
        """
        Sanitize HTML to prevent XSS attacks.

        Args:
            value: String potentially containing HTML

        Returns:
            HTML-escaped string

        Example:
            >>> InputValidator.sanitize_html("<script>alert('xss')</script>")
            '&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;'
        """
        if not isinstance(value, str):
            return value

        return html.escape(value)

    @staticmethod
    def validate_json(value: Union[str, Dict]) -> Dict:
        """
        Validate and parse JSON structure.

        Args:
            value: JSON string or dict

        Returns:
            Parsed JSON dict

        Raises:
            ValueError: If invalid JSON

        Example:
            >>> InputValidator.validate_json('{"key": "value"}')
            {'key': 'value'}
        """
        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            if len(value) > 1_000_000:  # 1MB limit
                raise ValueError("JSON too large (max 1MB)")

            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")

        raise ValueError(f"JSON must be dict or string, got {type(value).__name__}")

    @staticmethod
    def validate_command(command: str, allowed_commands: List[str]) -> str:
        """
        Validate command against whitelist.

        Args:
            command: Command to validate
            allowed_commands: List of allowed commands

        Returns:
            Validated command

        Raises:
            ValueError: If command not in whitelist

        Example:
            >>> InputValidator.validate_command("kubectl", ["kubectl", "docker"])
            'kubectl'
        """
        if command not in allowed_commands:
            logger.warning(
                f"Command not in whitelist: {command}",
                extra={"command": command, "whitelist": allowed_commands}
            )
            raise ValueError(
                f"Command '{command}' not allowed. Allowed: {allowed_commands}"
            )

        return command

    @staticmethod
    def validate_integer(
        value: Any,
        field_name: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> int:
        """
        Validate integer with optional range.

        Args:
            value: Value to validate
            field_name: Field name for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated integer

        Raises:
            ValueError: If validation fails
        """
        try:
            int_value = int(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{field_name} must be integer, got {type(value).__name__}") from e

        if min_value is not None and int_value < min_value:
            raise ValueError(f"{field_name} must be >= {min_value}, got {int_value}")

        if max_value is not None and int_value > max_value:
            raise ValueError(f"{field_name} must be <= {max_value}, got {int_value}")

        return int_value


# Pydantic models for validated inputs


class TenantIdModel(BaseModel):
    """Validated tenant ID."""

    tenant_id: str = Field(..., min_length=3, max_length=255, description="Tenant identifier")

    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        """Validate tenant ID format."""
        return InputValidator.validate_alphanumeric(v, 'tenant_id', min_length=3, max_length=255)


class UserIdModel(BaseModel):
    """Validated user ID (UUID format)."""

    user_id: str = Field(..., description="User identifier (UUID)")

    @validator('user_id')
    def validate_user_id(cls, v):
        """Validate user ID is valid UUID."""
        return InputValidator.validate_uuid(v, 'user_id')


class EmailModel(BaseModel):
    """Validated email address."""

    email: str = Field(..., min_length=5, max_length=255, description="Email address")

    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        return InputValidator.validate_email(v)


class SafeQueryInput(BaseModel):
    """Safe database query input with field name whitelist validation."""

    field: str = Field(..., description="Database field name (must be in whitelist)")
    value: Any = Field(..., description="Field value")
    operator: str = Field(default="=", description="Comparison operator")

    @validator('field')
    def validate_field(cls, v):
        """Validate field name against whitelist."""
        return InputValidator.validate_field_name(v)

    @validator('operator')
    def validate_operator(cls, v):
        """Validate operator against whitelist."""
        return InputValidator.validate_operator(v)

    @validator('value')
    def validate_value(cls, v):
        """Validate value for SQL injection."""
        if isinstance(v, str):
            InputValidator.validate_no_sql_injection(v, 'value')
        return v


class SafePathInput(BaseModel):
    """Safe file path input."""

    path: str = Field(..., description="File system path")
    must_exist: bool = Field(default=False, description="Path must exist")
    allowed_extensions: Optional[List[str]] = Field(default=None, description="Allowed file extensions")

    @validator('path')
    def validate_path(cls, v, values):
        """Validate path for traversal attacks."""
        must_exist = values.get('must_exist', False)
        allowed_extensions = values.get('allowed_extensions')
        path = InputValidator.validate_path(
            v,
            must_exist=must_exist,
            allow_relative=False,
            allowed_extensions=allowed_extensions
        )
        return str(path)


class SafeUrlInput(BaseModel):
    """Safe URL input."""

    url: str = Field(..., max_length=2048, description="URL")
    allowed_schemes: List[str] = Field(default=['https'], description="Allowed URL schemes")

    @validator('url')
    def validate_url(cls, v, values):
        """Validate URL for SSRF."""
        allowed_schemes = values.get('allowed_schemes', ['https'])
        return InputValidator.validate_url(v, allowed_schemes=allowed_schemes, allow_private_ips=False)


class SafeCommandInput(BaseModel):
    """Safe command execution input."""

    command: str = Field(..., description="Command to execute")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    allowed_commands: List[str] = Field(..., description="Whitelist of allowed commands")

    @validator('command')
    def validate_command(cls, v, values):
        """Validate command against whitelist."""
        allowed = values.get('allowed_commands', [])
        return InputValidator.validate_command(v, allowed)

    @validator('args')
    def validate_args(cls, v):
        """Validate arguments for injection."""
        for i, arg in enumerate(v):
            InputValidator.validate_no_command_injection(arg, f'arg[{i}]')
        return v


class PaginationInput(BaseModel):
    """Safe pagination input."""

    limit: int = Field(default=100, ge=1, le=1000, description="Number of records to return")
    offset: int = Field(default=0, ge=0, description="Number of records to skip")
    sort_by: Optional[str] = Field(default=None, description="Field to sort by")
    sort_direction: str = Field(default="DESC", description="Sort direction (ASC/DESC)")

    @validator('sort_by')
    def validate_sort_by(cls, v):
        """Validate sort field against whitelist."""
        if v is not None:
            return InputValidator.validate_field_name(v)
        return v

    @validator('sort_direction')
    def validate_sort_direction(cls, v):
        """Validate sort direction."""
        return InputValidator.validate_sort_direction(v)


class FilterInput(BaseModel):
    """Safe filter input for queries."""

    filters: List[SafeQueryInput] = Field(default_factory=list, description="List of filters")
    pagination: PaginationInput = Field(default_factory=PaginationInput, description="Pagination settings")

    @validator('filters')
    def validate_filters_count(cls, v):
        """Limit number of filters to prevent DoS."""
        if len(v) > 50:
            raise ValueError(f"Too many filters (max 50), got {len(v)}")
        return v
