# -*- coding: utf-8 -*-
"""
Unit tests for Encryption Service (SEC-003)

This package contains unit tests for the GreenLang encryption at rest service,
including AES-256-GCM encryption, envelope encryption with KMS, field-level
encryption, and key management.

Test modules:
    - test_encryption_service: Core encryption/decryption operations
    - test_envelope_encryption: KMS envelope encryption
    - test_field_encryption: Field-level encryption for databases
    - test_key_management: DEK cache and key lifecycle

Coverage target: 85%+
"""
