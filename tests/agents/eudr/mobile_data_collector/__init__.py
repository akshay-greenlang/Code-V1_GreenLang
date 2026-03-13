# -*- coding: utf-8 -*-
"""
Test suite for AGENT-EUDR-015 Mobile Data Collector Agent.

Tests all 8 engines of the Mobile Data Collector Agent with 85%+ coverage.
Validates business logic, error handling, edge cases, provenance tracking,
and compliance with EU 2023/1115 (EUDR) requirements.

Test Modules:
    test_offline_form_engine.py        - 62 tests  (Engine 1)
    test_gps_capture_engine.py         - 62 tests  (Engine 2)
    test_photo_evidence_collector.py   - 55 tests  (Engine 3)
    test_sync_engine.py                - 69 tests  (Engine 4)
    test_form_template_manager.py      - 65 tests  (Engine 5)
    test_digital_signature_engine.py   - 61 tests  (Engine 6)
    test_data_package_builder.py       - 61 tests  (Engine 7)
    test_device_fleet_manager.py       - 65 tests  (Engine 8)
    ---------------------------------------------------
    Total:                              500 tests

Coverage Target: 85%+ across all 8 engines.

Shared Fixtures (conftest.py):
    - Config fixtures (mdc_config, strict_config)
    - 8 engine fixtures (one per engine)
    - 10 factory functions (make_form_submission, make_gps_capture, etc.)
    - 5 assertion helpers (assert_valid_uuid, assert_valid_sha256, etc.)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015 Mobile Data Collector (GL-EUDR-MDC-015)
"""
