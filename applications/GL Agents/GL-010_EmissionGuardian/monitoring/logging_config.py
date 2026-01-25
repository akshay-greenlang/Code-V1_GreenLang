# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Structured Logging Configuration Module

This module provides comprehensive structured logging configuration including
JSON-formatted output, correlation ID propagation, log level management,
sensitive data masking, and log aggregation configuration.

Features:
    - JSON-formatted log output for log aggregation
    - Correlation ID propagation across requests
    - Configurable log levels by module
    - Sensitive data masking (credentials, PII)
    - ELK/Splunk integration configuration
    - Provenance hash inclusion for traceability

Example:
    >>> from monitoring.logging_config import setup_logging, get_logger
    >>> setup_logging()
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing emissions data", extra={"facility_id": "FAC-001"})
"""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import os
import re
import sys
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern, Set, Union


# =============================================================================
# Context Variables for Correlation
# =============================================================================

correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
facility_id_var: ContextVar[Optional[str]] = ContextVar('facility_id', default=None)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'
DEFAULT_LOG_LEVEL = 'INFO'
MAX_LOG_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
MAX_LOG_BACKUPS = 10


# =============================================================================
# Sensitive Data Patterns
# =============================================================================

SENSITIVE_PATTERNS: List[Pattern] = [
    re.compile(r'password["']?\s*[:=]\s*["']?[^"',\s]+', re.IGNORECASE),
    re.compile(r'api[_-]?key["']?\s*[:=]\s*["']?[^"',\s]+', re.IGNORECASE),
    re.compile(r'secret["']?\s*[:=]\s*["']?[^"',\s]+', re.IGNORECASE),
    re.compile(r'token["']?\s*[:=]\s*["']?[^"',\s]+', re.IGNORECASE),
    re.compile(r'bearer\s+[a-zA-Z0-9._-]+', re.IGNORECASE),
    re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'),  # Email
    re.compile(r'\d{3}[-.]?\d{3}[-.]?\d{4}'),  # Phone
    re.compile(r'\d{3}[-]?\d{2}[-]?\d{4}'),  # SSN
]

SENSITIVE_FIELD_NAMES: Set[str] = {
    'password', 'passwd', 'pwd', 'secret', 'token', 'api_key', 'apikey',
    'access_token', 'refresh_token', 'private_key', 'credentials',
    'authorization', 'auth', 'ssn', 'social_security', 'credit_card',
    'card_number', 'cvv', 'pin'
}


# =============================================================================
# Correlation ID Management
# =============================================================================

def generate_correlation_id() -> str:
    return str(uuid.uuid4())


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    cid = correlation_id or generate_correlation_id()
    correlation_id_var.set(cid)
    return cid


def get_correlation_id() -> Optional[str]:
    return correlation_id_var.get()


def set_request_context(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    facility_id: Optional[str] = None
) -> Dict[str, Optional[str]]:
    ctx = {
        'correlation_id': set_correlation_id(correlation_id),
        'request_id': request_id,
        'user_id': user_id,
        'facility_id': facility_id
    }
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if facility_id:
        facility_id_var.set(facility_id)
    return ctx


def clear_request_context() -> None:
    correlation_id_var.set(None)
    request_id_var.set(None)
    user_id_var.set(None)
    facility_id_var.set(None)



# =============================================================================
# Data Masking
# =============================================================================

class DataMasker:
    def __init__(self, patterns: Optional[List[Pattern]] = None, field_names: Optional[Set[str]] = None):
        self.patterns = patterns or SENSITIVE_PATTERNS
        self.field_names = field_names or SENSITIVE_FIELD_NAMES
        self.mask_char = '*'
        self.mask_length = 8

    def mask_string(self, value: str) -> str:
        masked = value
        for pattern in self.patterns:
            masked = pattern.sub(self.mask_char * self.mask_length, masked)
        return masked

    def mask_dict(self, data: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        if depth > 10:
            return data
        result = {}
        for key, value in data.items():
            if key.lower() in self.field_names:
                result[key] = self.mask_char * self.mask_length
            elif isinstance(value, str):
                result[key] = self.mask_string(value)
            elif isinstance(value, dict):
                result[key] = self.mask_dict(value, depth + 1)
            elif isinstance(value, list):
                result[key] = [
                    self.mask_dict(v, depth + 1) if isinstance(v, dict)
                    else (self.mask_string(v) if isinstance(v, str) else v)
                    for v in value
                ]
            else:
                result[key] = value
        return result


_default_masker = DataMasker()


def mask_sensitive_data(data: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
    if isinstance(data, str):
        return _default_masker.mask_string(data)
    elif isinstance(data, dict):
        return _default_masker.mask_dict(data)
    return data
test
