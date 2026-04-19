# -*- coding: utf-8 -*-
"""
GreenLang Security Validators
==============================

Input validation and sanitization to prevent common attack vectors:
- SQL injection prevention
- XSS (Cross-Site Scripting) prevention
- Path traversal prevention
- Command injection prevention

Follows OWASP security best practices.

Author: GreenLang Security Team
Phase: 3 - Security Hardening
"""

import html
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote, urlparse


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


class SQLInjectionValidator:
    """Validator to prevent SQL injection attacks."""

    # SQL keywords and patterns that are suspicious
    SQL_KEYWORDS = {
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "EXEC",
        "EXECUTE",
        "UNION",
        "WHERE",
        "OR",
        "AND",
        "--",
        ";",
        "/*",
        "*/",
        "xp_",
        "sp_",
    }

    @staticmethod
    def validate(input_str: str, allow_quotes: bool = False) -> str:
        """
        Validate input to prevent SQL injection.

        Args:
            input_str: Input string to validate
            allow_quotes: If False, reject strings with quotes

        Returns:
            Sanitized string

        Raises:
            ValidationError: If input contains SQL injection patterns
        """
        if not isinstance(input_str, str):
            raise ValidationError("Input must be a string")

        # Check for SQL keywords (case-insensitive)
        upper_input = input_str.upper()
        for keyword in SQLInjectionValidator.SQL_KEYWORDS:
            if keyword in upper_input:
                raise ValidationError(f"Suspicious SQL keyword detected: {keyword}")

        # Check for quotes if not allowed
        if not allow_quotes:
            if "'" in input_str or '"' in input_str or "`" in input_str:
                raise ValidationError("Quotes not allowed in input")

        # Check for SQL comment patterns
        if "--" in input_str or "/*" in input_str or "*/" in input_str:
            raise ValidationError("SQL comment patterns not allowed")

        return input_str

    @staticmethod
    def escape_string(input_str: str) -> str:
        """
        Escape string for safe use in SQL (basic escaping).

        Note: Always use parameterized queries when possible.
        This is a backup for cases where parameterization isn't available.
        """
        if not isinstance(input_str, str):
            raise ValidationError("Input must be a string")

        # Escape single quotes by doubling them (SQL standard)
        escaped = input_str.replace("'", "''")

        # Escape backslashes
        escaped = escaped.replace("\\", "\\\\")

        return escaped


class XSSValidator:
    """Validator to prevent Cross-Site Scripting (XSS) attacks."""

    # Dangerous HTML tags
    DANGEROUS_TAGS = {
        "script",
        "iframe",
        "object",
        "embed",
        "applet",
        "meta",
        "link",
        "style",
        "base",
        "form",
    }

    # Dangerous attributes
    DANGEROUS_ATTRIBUTES = {
        "onclick",
        "onload",
        "onerror",
        "onmouseover",
        "onmouseout",
        "onfocus",
        "onblur",
        "onchange",
        "onsubmit",
        "javascript:",
        "vbscript:",
        "data:",
    }

    @staticmethod
    def validate_html(input_str: str, strict: bool = True) -> str:
        """
        Validate HTML input to prevent XSS.

        Args:
            input_str: HTML string to validate
            strict: If True, reject any HTML tags

        Returns:
            Sanitized string

        Raises:
            ValidationError: If input contains XSS patterns
        """
        if not isinstance(input_str, str):
            raise ValidationError("Input must be a string")

        # Check for dangerous tags
        lower_input = input_str.lower()
        for tag in XSSValidator.DANGEROUS_TAGS:
            if f"<{tag}" in lower_input:
                raise ValidationError(f"Dangerous HTML tag detected: {tag}")

        # Check for dangerous attributes
        for attr in XSSValidator.DANGEROUS_ATTRIBUTES:
            if attr.lower() in lower_input:
                raise ValidationError(f"Dangerous attribute/protocol detected: {attr}")

        # In strict mode, reject any HTML tags
        if strict and ("<" in input_str and ">" in input_str):
            raise ValidationError("HTML tags not allowed in strict mode")

        return input_str

    @staticmethod
    def sanitize_html(input_str: str) -> str:
        """
        Sanitize HTML by escaping all special characters.

        This is the safest approach for untrusted input.
        """
        if not isinstance(input_str, str):
            raise ValidationError("Input must be a string")

        return html.escape(input_str, quote=True)

    @staticmethod
    def sanitize_json(data: Any) -> str:
        """
        Safely serialize data to JSON, escaping dangerous characters.
        """
        # Use json.dumps with ensure_ascii to prevent encoding issues
        json_str = json.dumps(data, ensure_ascii=True)

        # Additional escaping for HTML contexts
        json_str = json_str.replace("<", "\\u003c")
        json_str = json_str.replace(">", "\\u003e")
        json_str = json_str.replace("&", "\\u0026")

        return json_str


class PathTraversalValidator:
    """Validator to prevent path traversal attacks."""

    @staticmethod
    def validate_path(
        input_path: Union[str, Path],
        base_dir: Optional[Union[str, Path]] = None,
        must_exist: bool = False,
    ) -> Path:
        """
        Validate file path to prevent path traversal.

        Args:
            input_path: Path to validate
            base_dir: Base directory - path must be within this directory
            must_exist: If True, path must exist

        Returns:
            Validated absolute path

        Raises:
            ValidationError: If path is invalid or suspicious
        """
        if not isinstance(input_path, (str, Path)):
            raise ValidationError("Input must be a string or Path")

        # Convert to Path object
        path = Path(input_path)

        # Check for dangerous patterns
        path_str = str(path)
        if ".." in path_str:
            raise ValidationError("Path traversal pattern '..' not allowed")

        if path_str.startswith("~"):
            raise ValidationError("Home directory expansion not allowed")

        # Resolve to absolute path
        try:
            resolved_path = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValidationError(f"Failed to resolve path: {e}")

        # Check if within base directory
        if base_dir is not None:
            base_path = Path(base_dir).resolve()
            try:
                resolved_path.relative_to(base_path)
            except ValueError:
                raise ValidationError(f"Path must be within {base_dir}")

        # Check if exists if required
        if must_exist and not resolved_path.exists():
            raise ValidationError(f"Path does not exist: {resolved_path}")

        return resolved_path

    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 255) -> str:
        """
        Sanitize filename by removing dangerous characters.

        Args:
            filename: Filename to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized filename
        """
        if not isinstance(filename, str):
            raise ValidationError("Filename must be a string")

        # Remove path separators
        sanitized = filename.replace("/", "_").replace("\\", "_")

        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")

        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip(". ")

        # Remove or replace dangerous characters
        # Keep alphanumeric, dash, underscore, dot
        sanitized = re.sub(r"[^\w\s\-.]", "_", sanitized)

        # Collapse multiple underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Limit length
        if len(sanitized) > max_length:
            # Keep extension if present
            parts = sanitized.rsplit(".", 1)
            if len(parts) == 2:
                name, ext = parts
                max_name_len = max_length - len(ext) - 1
                sanitized = f"{name[:max_name_len]}.{ext}"
            else:
                sanitized = sanitized[:max_length]

        # Ensure not empty
        if not sanitized:
            raise ValidationError("Filename cannot be empty after sanitization")

        return sanitized


class CommandInjectionValidator:
    """Validator to prevent command injection attacks."""

    # Dangerous shell metacharacters
    SHELL_METACHARACTERS = {";", "&", "|", "`", "$", "(", ")", "<", ">", "\n", "\\"}

    @staticmethod
    def validate_command_arg(arg: str, allow_shell_chars: bool = False) -> str:
        """
        Validate command line argument.

        Args:
            arg: Argument to validate
            allow_shell_chars: If False (recommended), reject shell metacharacters

        Returns:
            Validated argument

        Raises:
            ValidationError: If argument contains dangerous characters
        """
        if not isinstance(arg, str):
            raise ValidationError("Argument must be a string")

        if not allow_shell_chars:
            for char in CommandInjectionValidator.SHELL_METACHARACTERS:
                if char in arg:
                    raise ValidationError(
                        f"Shell metacharacter not allowed: {repr(char)}"
                    )

        # Check for null bytes
        if "\x00" in arg:
            raise ValidationError("Null byte not allowed in argument")

        return arg

    @staticmethod
    def escape_shell_arg(arg: str) -> str:
        """
        Escape argument for safe use in shell commands.

        Note: Always prefer using subprocess with shell=False when possible.
        """
        if not isinstance(arg, str):
            raise ValidationError("Argument must be a string")

        # Use shlex.quote for proper shell escaping
        return shlex.quote(arg)

    @staticmethod
    def validate_command_list(cmd_list: List[str]) -> List[str]:
        """
        Validate list of command arguments.

        This is the preferred way to execute commands (using subprocess
        with shell=False and a list of arguments).

        Args:
            cmd_list: List of command and arguments

        Returns:
            Validated command list

        Raises:
            ValidationError: If any argument is invalid
        """
        if not isinstance(cmd_list, list):
            raise ValidationError("Command must be a list")

        if not cmd_list:
            raise ValidationError("Command list cannot be empty")

        # Validate each argument
        validated = []
        for arg in cmd_list:
            if not isinstance(arg, str):
                raise ValidationError(f"All arguments must be strings, got {type(arg)}")

            # Don't allow empty arguments (except for special cases)
            if not arg and arg != "":
                raise ValidationError("Arguments cannot be None")

            validated.append(CommandInjectionValidator.validate_command_arg(arg, allow_shell_chars=False))

        return validated


class URLValidator:
    """Validator for URLs to prevent SSRF and other attacks."""

    # Allowed URL schemes
    ALLOWED_SCHEMES = {"http", "https"}

    # Blocked hostnames (SSRF prevention)
    BLOCKED_HOSTS = {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "169.254.169.254",  # AWS metadata
        "metadata.google.internal",  # GCP metadata
    }

    @staticmethod
    def validate_url(
        url: str,
        allowed_schemes: Optional[List[str]] = None,
        allow_private_ips: bool = False,
    ) -> str:
        """
        Validate URL to prevent SSRF and other attacks.

        Args:
            url: URL to validate
            allowed_schemes: List of allowed schemes (default: http, https)
            allow_private_ips: If False, block private IP addresses

        Returns:
            Validated URL

        Raises:
            ValidationError: If URL is invalid or dangerous
        """
        if not isinstance(url, str):
            raise ValidationError("URL must be a string")

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL: {e}")

        # Check scheme
        schemes = allowed_schemes or list(URLValidator.ALLOWED_SCHEMES)
        if parsed.scheme.lower() not in schemes:
            raise ValidationError(f"URL scheme not allowed: {parsed.scheme}")

        # Check hostname
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise ValidationError("URL must have a hostname")

        # Check for blocked hosts
        if not allow_private_ips:
            for blocked in URLValidator.BLOCKED_HOSTS:
                if blocked in hostname:
                    raise ValidationError(f"Access to {hostname} is blocked (SSRF prevention)")

            # Check for private IP ranges
            if hostname.startswith("10.") or hostname.startswith("192.168."):
                raise ValidationError(f"Access to private IP {hostname} is blocked")

        return url

    @staticmethod
    def sanitize_url_for_display(url: str, max_length: int = 100) -> str:
        """
        Sanitize URL for safe display in HTML.

        Args:
            url: URL to sanitize
            max_length: Maximum length for display

        Returns:
            Sanitized URL safe for HTML display
        """
        # First validate it's a real URL
        URLValidator.validate_url(url)

        # HTML escape
        sanitized = html.escape(url, quote=True)

        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."

        return sanitized


# Convenience functions for common validation tasks

def validate_api_key(api_key: str) -> str:
    """
    Validate API key format.

    Args:
        api_key: API key to validate

    Returns:
        Validated API key

    Raises:
        ValidationError: If API key is invalid
    """
    if not isinstance(api_key, str):
        raise ValidationError("API key must be a string")

    # Basic checks
    if len(api_key) < 16:
        raise ValidationError("API key too short")

    if len(api_key) > 512:
        raise ValidationError("API key too long")

    # Must be alphanumeric + common special chars
    if not re.match(r"^[A-Za-z0-9\-_\.]+$", api_key):
        raise ValidationError("API key contains invalid characters")

    return api_key


def validate_email(email: str) -> str:
    """
    Validate email address format.

    Args:
        email: Email to validate

    Returns:
        Validated email

    Raises:
        ValidationError: If email is invalid
    """
    if not isinstance(email, str):
        raise ValidationError("Email must be a string")

    # Basic email regex (not perfect but good enough for security validation)
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email):
        raise ValidationError("Invalid email format")

    if len(email) > 254:  # RFC 5321
        raise ValidationError("Email too long")

    return email.lower()


def validate_username(username: str, min_length: int = 3, max_length: int = 32) -> str:
    """
    Validate username format.

    Args:
        username: Username to validate
        min_length: Minimum length
        max_length: Maximum length

    Returns:
        Validated username

    Raises:
        ValidationError: If username is invalid
    """
    if not isinstance(username, str):
        raise ValidationError("Username must be a string")

    if len(username) < min_length:
        raise ValidationError(f"Username too short (min {min_length} characters)")

    if len(username) > max_length:
        raise ValidationError(f"Username too long (max {max_length} characters)")

    # Alphanumeric, underscore, dash only
    if not re.match(r"^[a-zA-Z0-9_-]+$", username):
        raise ValidationError("Username can only contain letters, numbers, underscore, and dash")

    # Cannot start with dash or underscore
    if username[0] in "-_":
        raise ValidationError("Username cannot start with dash or underscore")

    return username


def validate_json_data(data: str, max_size: int = 1024 * 1024) -> Dict[str, Any]:
    """
    Validate and parse JSON data.

    Args:
        data: JSON string to validate
        max_size: Maximum size in bytes

    Returns:
        Parsed JSON data

    Raises:
        ValidationError: If JSON is invalid
    """
    if not isinstance(data, str):
        raise ValidationError("JSON data must be a string")

    if len(data) > max_size:
        raise ValidationError(f"JSON data too large (max {max_size} bytes)")

    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {e}")

    return parsed
