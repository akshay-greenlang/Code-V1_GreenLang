# -*- coding: utf-8 -*-
"""
Unit tests for PII Service - SEC-011 PII Detection/Redaction Enhancements.

This package contains comprehensive unit tests for all PII Service components:
- SecureTokenVault: Tokenization with AES-256-GCM encryption
- PIIEnforcementEngine: Real-time enforcement policies
- AllowlistManager: False positive filtering
- StreamingScanner: Kafka/Kinesis integration
- RemediationEngine: Auto-remediation workflows
- PIIService: Unified service facade
- API Routes: FastAPI endpoint tests

Test coverage target: 85%+
"""

from __future__ import annotations

__all__: list[str] = []
