# -*- coding: utf-8 -*-
"""
Input validation and sanitization utilities.

Prevents:
- DoS attacks via large files
- Path traversal
- HTML/script injection
- Malformed data
"""

import os
import re
import html
from pathlib import Path
from typing import Union, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# File size limits (in bytes)
MAX_FILE_SIZES = {
    'csv': 100 * 1024 * 1024,      # 100 MB
    'json': 50 * 1024 * 1024,       # 50 MB
    'excel': 100 * 1024 * 1024,     # 100 MB
    'xml': 50 * 1024 * 1024,        # 50 MB
    'pdf': 20 * 1024 * 1024,        # 20 MB
    'default': 10 * 1024 * 1024     # 10 MB
}


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_file_size(file_path: Union[str, Path], file_type: str = 'default') -> bool:
    """
    Validate file size is within limits.

    Args:
        file_path: Path to file
        file_type: Type of file (csv, json, excel, xml, pdf)

    Returns:
        True if valid

    Raises:
        ValidationError: If file too large
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")

    size_bytes = file_path.stat().st_size
    max_size = MAX_FILE_SIZES.get(file_type, MAX_FILE_SIZES['default'])

    if size_bytes > max_size:
        raise ValidationError(
            f"File too large: {size_bytes:,} bytes "
            f"(max {max_size:,} bytes for {file_type})"
        )

    logger.debug(f"File size OK: {size_bytes:,} bytes ({file_type})")
    return True


def validate_file_path(file_path: Union[str, Path], allowed_dirs: Optional[List[str]] = None) -> bool:
    """
    Validate file path to prevent path traversal.

    Args:
        file_path: Path to validate
        allowed_dirs: List of allowed directory paths

    Returns:
        True if valid

    Raises:
        ValidationError: If path is suspicious
    """
    file_path = Path(file_path).resolve()

    # Check for path traversal
    path_str = str(file_path)
    if '..' in path_str:
        raise ValidationError(f"Path traversal detected: {file_path}")

    # Check allowed directories
    if allowed_dirs:
        allowed = any(
            str(file_path).startswith(str(Path(d).resolve()))
            for d in allowed_dirs
        )
        if not allowed:
            raise ValidationError(f"Path not in allowed directories: {file_path}")

    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent injection.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove any path components
    filename = os.path.basename(filename)

    # Allow only alphanumeric, dash, underscore, dot
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

    # Prevent hidden files
    if filename.startswith('.'):
        filename = '_' + filename

    return filename


def validate_string_length(value: str, field_name: str, max_length: int = 10000) -> bool:
    """
    Validate string length.

    Args:
        value: String to validate
        field_name: Name of field (for error message)
        max_length: Maximum allowed length

    Returns:
        True if valid

    Raises:
        ValidationError: If too long
    """
    if len(value) > max_length:
        raise ValidationError(
            f"{field_name} too long: {len(value)} chars (max {max_length})"
        )
    return True


def validate_esrs_code(code: str) -> bool:
    """
    Validate ESRS data point code format.

    Args:
        code: ESRS code (e.g., "E1-1", "S2-4")

    Returns:
        True if valid

    Raises:
        ValidationError: If invalid format
    """
    # Pattern: E1-1, S2-4, G1-1, etc.
    pattern = r'^(E[1-5]|S[1-4]|G1)-\d+[a-z]?$'

    if not re.match(pattern, code):
        raise ValidationError(f"Invalid ESRS code format: {code}")

    return True


def sanitize_html(text: str, allow_tags: Optional[List[str]] = None) -> str:
    """
    Sanitize HTML to prevent XSS/injection.

    Args:
        text: HTML text to sanitize
        allow_tags: List of allowed tags (e.g., ['b', 'i', 'p'])

    Returns:
        Sanitized HTML
    """
    if allow_tags is None:
        # Strip all HTML
        return html.escape(text)

    # For allowed tags, use bleach library
    try:
        import bleach
        return bleach.clean(
            text,
            tags=allow_tags,
            attributes={},
            strip=True
        )
    except ImportError:
        # Fallback: escape everything
        logger.warning("bleach not installed, escaping all HTML")
        return html.escape(text)


def sanitize_xbrl_text(text: str) -> str:
    """
    Sanitize text for XBRL/iXBRL output.

    Args:
        text: Text to include in XBRL

    Returns:
        Sanitized text safe for XML
    """
    # Escape XML special characters
    text = html.escape(text)

    # Remove any remaining control characters
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')

    return text


def validate_numeric_value(value: Any, field_name: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> bool:
    """
    Validate numeric value is within range.

    Args:
        value: Numeric value to validate
        field_name: Name of field (for error message)
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        True if valid

    Raises:
        ValidationError: If out of range or not numeric
    """
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(f"{field_name} must be numeric, got: {value}")

    if min_val is not None and num_value < min_val:
        raise ValidationError(f"{field_name} must be >= {min_val}, got: {num_value}")

    if max_val is not None and num_value > max_val:
        raise ValidationError(f"{field_name} must be <= {max_val}, got: {num_value}")

    return True


def validate_email(email: str) -> bool:
    """
    Validate email format.

    Args:
        email: Email address to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If invalid format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if not re.match(pattern, email):
        raise ValidationError(f"Invalid email format: {email}")

    return True


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> bool:
    """
    Validate URL format and scheme.

    Args:
        url: URL to validate
        allowed_schemes: List of allowed schemes (e.g., ['http', 'https'])

    Returns:
        True if valid

    Raises:
        ValidationError: If invalid format or scheme
    """
    from urllib.parse import urlparse

    if allowed_schemes is None:
        allowed_schemes = ['http', 'https']

    try:
        parsed = urlparse(url)

        if parsed.scheme not in allowed_schemes:
            raise ValidationError(f"URL scheme must be one of {allowed_schemes}, got: {parsed.scheme}")

        if not parsed.netloc:
            raise ValidationError(f"Invalid URL: missing domain")

        return True

    except Exception as e:
        raise ValidationError(f"Invalid URL format: {url} - {str(e)}")


def validate_date_format(date_str: str, format_str: str = '%Y-%m-%d') -> bool:
    """
    Validate date string format.

    Args:
        date_str: Date string to validate
        format_str: Expected date format

    Returns:
        True if valid

    Raises:
        ValidationError: If invalid format
    """
    from datetime import datetime

    try:
        datetime.strptime(date_str, format_str)
        return True
    except ValueError:
        raise ValidationError(f"Invalid date format: {date_str}, expected: {format_str}")


def sanitize_dict_keys(data: dict, max_depth: int = 10, current_depth: int = 0) -> dict:
    """
    Sanitize dictionary keys to prevent injection.

    Args:
        data: Dictionary to sanitize
        max_depth: Maximum nesting depth
        current_depth: Current nesting level

    Returns:
        Sanitized dictionary

    Raises:
        ValidationError: If nesting too deep
    """
    if current_depth > max_depth:
        raise ValidationError(f"Dictionary nesting exceeds maximum depth of {max_depth}")

    sanitized = {}
    for key, value in data.items():
        # Sanitize key
        if not isinstance(key, str):
            key = str(key)

        # Remove special characters from keys
        safe_key = re.sub(r'[^a-zA-Z0-9._-]', '_', key)

        # Recursively sanitize nested dicts
        if isinstance(value, dict):
            value = sanitize_dict_keys(value, max_depth, current_depth + 1)

        sanitized[safe_key] = value

    return sanitized


def validate_json_size(json_str: str, max_size_bytes: int = 50 * 1024 * 1024) -> bool:
    """
    Validate JSON string size.

    Args:
        json_str: JSON string to validate
        max_size_bytes: Maximum size in bytes

    Returns:
        True if valid

    Raises:
        ValidationError: If too large
    """
    size_bytes = len(json_str.encode('utf-8'))

    if size_bytes > max_size_bytes:
        raise ValidationError(
            f"JSON too large: {size_bytes:,} bytes "
            f"(max {max_size_bytes:,} bytes)"
        )

    return True
