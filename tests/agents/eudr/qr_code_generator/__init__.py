# -*- coding: utf-8 -*-
"""
Test package for AGENT-EUDR-014 QR Code Generator Agent.

Provides comprehensive unit tests for all 8 engines of the QR Code
Generator Agent covering QR code encoding, payload composition, label
rendering, batch code generation, verification URL construction,
anti-counterfeiting, bulk generation pipeline, and code lifecycle
management.

Test Modules:
    test_qr_encoder           - Engine 1: QR code generation (65+ tests)
    test_payload_composer     - Engine 2: Payload composition (55+ tests)
    test_label_template_engine - Engine 3: Label rendering (55+ tests)
    test_batch_code_generator - Engine 4: Batch code generation (55+ tests)
    test_verification_url_builder - Engine 5: Verification URLs (50+ tests)
    test_anti_counterfeit_engine - Engine 6: Anti-counterfeiting (55+ tests)
    test_bulk_generation_pipeline - Engine 7: Bulk generation (50+ tests)
    test_code_lifecycle_manager - Engine 8: Lifecycle management (55+ tests)

Shared Fixtures:
    conftest.py               - Factory functions, assertion helpers, constants

Total: 500+ tests targeting 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
"""
