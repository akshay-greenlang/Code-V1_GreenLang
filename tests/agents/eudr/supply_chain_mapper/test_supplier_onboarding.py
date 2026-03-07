# -*- coding: utf-8 -*-
"""
Tests for SupplierOnboardingEngine - AGENT-EUDR-001 Feature 8

Comprehensive test suite covering:
- Session creation and token generation (secure HMAC-SHA256 tokens)
- Token validation and expiry lifecycle
- Multi-step wizard flow (company info -> commodities -> plots -> certs -> decl -> sub-tier)
- EUDR real-time validation (GPS coordinates, polygon >4ha, commodity declaration)
- Mobile GPS coordinate capture support
- Completion percentage tracking (per-field arithmetic)
- Automated reminder system scheduling and processing
- Bulk import from CSV (integration with AGENT-DATA-002)
- Graph entity auto-creation from completed onboarding
- Session lifecycle (create, cancel, list, metrics)
- Edge cases and error handling

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 (Feature 8: Supplier Onboarding and Discovery Workflow)
"""

import asyncio
import csv
import io
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.supply_chain_mapper.supplier_onboarding import (
    BulkImportResult,
    BulkImportStatus,
    CertificationData,
    DEFAULT_REMINDER_DAYS,
    DEFAULT_TOKEN_EXPIRY_DAYS,
    GeolocationLinkerProtocol,
    GraphEngineProtocol,
    MAX_BULK_IMPORT_BATCH,
    MIN_COORDINATE_PRECISION,
    NotificationServiceProtocol,
    OnboardingMetrics,
    OnboardingSession,
    OnboardingStatus,
    OnboardingStepResult,
    POLYGON_AREA_THRESHOLD_HA,
    PlotData,
    ReminderRecord,
    ReminderStatus,
    ReminderType,
    STEP_FIELDS,
    SUPPORTED_LANGUAGES,
    SubTierSupplierData,
    SupplierOnboardingEngine,
    TOTAL_FIELD_GROUPS,
    WIZARD_STEPS,
    _compute_hash,
    _generate_id,
    _utcnow,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a default SupplierOnboardingEngine instance."""
    return SupplierOnboardingEngine()


@pytest.fixture
def mock_graph_engine():
    """Create a mock graph engine that implements GraphEngineProtocol."""
    mock = AsyncMock()
    mock.add_node = AsyncMock(return_value="node-test-001")
    mock.add_edge = AsyncMock(return_value="edge-test-001")
    return mock


@pytest.fixture
def mock_geo_linker():
    """Create a mock geolocation linker."""
    mock = MagicMock()
    mock.link_producer_to_plot = MagicMock(
        return_value={"link_id": "GEO-LNK-test001", "status": "linked"}
    )
    return mock


@pytest.fixture
def mock_notification_service():
    """Create a mock notification service."""
    mock = MagicMock()
    mock.send_email = MagicMock(return_value=True)
    mock.send_webhook = MagicMock(return_value=True)
    return mock


@pytest.fixture
def engine_with_mocks(mock_graph_engine, mock_geo_linker, mock_notification_service):
    """Create an engine with all mock integrations attached."""
    return SupplierOnboardingEngine(
        graph_engine=mock_graph_engine,
        geo_linker=mock_geo_linker,
        notification_service=mock_notification_service,
    )


@pytest.fixture
def valid_company_data():
    """Valid company info step data."""
    return {
        "legal_name": "Fazenda Verde Ltda",
        "country_code": "BR",
        "registration_id": "BR-12345678",
        "contact_name": "Maria Silva",
        "contact_email": "maria@fazendaverde.com.br",
        "contact_phone": "+5511999999999",
    }


@pytest.fixture
def valid_commodities_data():
    """Valid commodities step data."""
    return {
        "commodities": ["soya", "coffee"],
        "hs_codes": ["1201.90", "0901.11"],
    }


@pytest.fixture
def valid_plots_data():
    """Valid plots step data with 6+ decimal precision."""
    return {
        "plots": [
            {
                "latitude": -2.501234,
                "longitude": -44.282345,
                "area_hectares": 3.5,
                "commodity": "soya",
                "country_code": "BR",
                "capture_method": "gps",
                "capture_accuracy_m": 5.0,
            },
        ],
    }


@pytest.fixture
def valid_certifications_data():
    """Valid certifications step data."""
    return {
        "certifications": [
            {
                "certification_type": "RSPO",
                "certificate_number": "RSPO-2025-001",
                "issuing_body": "RSPO Secretariat",
            },
        ],
    }


@pytest.fixture
def valid_declarations_data():
    """Valid declarations step data."""
    return {
        "deforestation_free_declaration": True,
        "legality_declaration": True,
    }


@pytest.fixture
def valid_sub_tier_data():
    """Valid sub-tier suppliers step data."""
    return {
        "sub_tier_suppliers": [
            {
                "supplier_name": "Upstream Farm Co",
                "country_code": "BR",
                "commodities": ["soya"],
                "relationship_type": "direct",
                "contact_email": "upstream@example.com",
                "estimated_volume_pct": 60.0,
            },
        ],
    }


@pytest.fixture
def session(engine):
    """Create a standard onboarding session."""
    return engine.create_onboarding_session(
        operator_id="op-001",
        graph_id="graph-001",
        supplier_name="Fazenda Verde",
        supplier_email="verde@example.com",
        commodity="soya",
    )


# ===========================================================================
# Test Suite: Session Creation and Token Generation
# ===========================================================================


class TestSessionCreation:
    """Tests for onboarding session creation and token generation."""

    def test_create_session_basic(self, engine):
        """Test basic session creation with all required fields."""
        session = engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="Test Supplier",
            supplier_email="test@example.com",
            commodity="cocoa",
        )

        assert session.session_id.startswith("ONB-")
        assert session.operator_id == "op-001"
        assert session.graph_id == "graph-001"
        assert session.supplier_name == "Test Supplier"
        assert session.supplier_email == "test@example.com"
        assert session.commodity == "cocoa"
        assert session.status == OnboardingStatus.INVITED
        assert session.completion_pct == 0.0
        assert len(session.token) == 64  # SHA-256 hex digest
        assert session.token_expires_at is not None
        assert session.provenance_hash != ""

    def test_create_session_generates_unique_tokens(self, engine):
        """Test that each session gets a unique token."""
        s1 = engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="Supplier A",
            supplier_email="a@example.com",
            commodity="cocoa",
        )
        s2 = engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="Supplier B",
            supplier_email="b@example.com",
            commodity="cocoa",
        )

        assert s1.token != s2.token
        assert s1.session_id != s2.session_id

    def test_create_session_schedules_reminders(self, engine):
        """Test that reminders are scheduled at default intervals."""
        session = engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="Test",
            supplier_email="test@example.com",
            commodity="soya",
        )

        assert len(session.reminders) == len(DEFAULT_REMINDER_DAYS)
        for i, reminder in enumerate(session.reminders):
            assert reminder.status == ReminderStatus.PENDING
            assert reminder.reminder_type == ReminderType.EMAIL
            assert reminder.target_email == "test@example.com"

    def test_create_session_custom_token_expiry(self, engine):
        """Test session creation with custom token expiry."""
        session = engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="Test",
            supplier_email="test@example.com",
            commodity="cocoa",
            token_expiry_days=7,
        )

        expected_min = _utcnow() + timedelta(days=6)
        expected_max = _utcnow() + timedelta(days=8)
        assert expected_min <= session.token_expires_at <= expected_max

    def test_create_session_language_support(self, engine):
        """Test session creation with each supported language."""
        for lang in SUPPORTED_LANGUAGES:
            session = engine.create_onboarding_session(
                operator_id="op-001",
                graph_id="graph-001",
                supplier_name=f"Test-{lang}",
                supplier_email=f"{lang}@example.com",
                commodity="cocoa",
                language=lang,
            )
            assert session.language == lang

    def test_create_session_invalid_language(self, engine):
        """Test that invalid language raises ValueError."""
        with pytest.raises(ValueError, match="language must be one of"):
            engine.create_onboarding_session(
                operator_id="op-001",
                graph_id="graph-001",
                supplier_name="Test",
                supplier_email="test@example.com",
                commodity="cocoa",
                language="xx",
            )

    def test_create_session_empty_fields_raise(self, engine):
        """Test that empty required fields raise ValueError."""
        with pytest.raises(ValueError, match="operator_id must be non-empty"):
            engine.create_onboarding_session(
                operator_id="",
                graph_id="graph-001",
                supplier_name="Test",
                supplier_email="test@example.com",
                commodity="cocoa",
            )

        with pytest.raises(ValueError, match="supplier_name must be non-empty"):
            engine.create_onboarding_session(
                operator_id="op-001",
                graph_id="graph-001",
                supplier_name="",
                supplier_email="test@example.com",
                commodity="cocoa",
            )

    def test_create_session_invalid_email_format(self, engine):
        """Test that invalid email format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid email format"):
            engine.create_onboarding_session(
                operator_id="op-001",
                graph_id="graph-001",
                supplier_name="Test",
                supplier_email="not-an-email",
                commodity="cocoa",
            )


# ===========================================================================
# Test Suite: Token Validation
# ===========================================================================


class TestTokenValidation:
    """Tests for token validation and expiry lifecycle."""

    def test_validate_valid_token(self, engine, session):
        """Test successful token validation."""
        result = engine.validate_token(session.token)
        assert result is not None
        assert result.session_id == session.session_id

    def test_validate_unknown_token(self, engine):
        """Test that unknown tokens return None."""
        result = engine.validate_token("nonexistent-token")
        assert result is None

    def test_validate_expired_token(self, engine):
        """Test that expired tokens return None and update session status."""
        session = engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="Test",
            supplier_email="test@example.com",
            commodity="cocoa",
        )
        # Force expiry
        session.token_expires_at = _utcnow() - timedelta(hours=1)

        result = engine.validate_token(session.token)
        assert result is None
        assert session.status == OnboardingStatus.EXPIRED

    def test_validate_cancelled_session_token(self, engine, session):
        """Test that cancelled session tokens return None."""
        engine.cancel_session(session.session_id)
        result = engine.validate_token(session.token)
        assert result is None

    def test_get_session_by_token(self, engine, session):
        """Test retrieving session by token."""
        result = engine.get_session_by_token(session.token)
        assert result is not None
        assert result.session_id == session.session_id


# ===========================================================================
# Test Suite: Wizard Step Flow
# ===========================================================================


class TestWizardStepFlow:
    """Tests for multi-step onboarding wizard submission."""

    def test_submit_company_info_valid(self, engine, session, valid_company_data):
        """Test valid company info submission."""
        result = engine.submit_step(session.session_id, "company_info", valid_company_data)

        assert result.is_valid is True
        assert result.step_name == "company_info"
        assert result.fields_completed == 6
        assert result.fields_total == 6
        assert len(result.errors) == 0
        assert result.provenance_hash != ""

    def test_submit_company_info_missing_fields(self, engine, session):
        """Test company info with missing required fields."""
        result = engine.submit_step(session.session_id, "company_info", {})

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("legal_name" in e for e in result.errors)

    def test_submit_commodities_valid(self, engine, session, valid_commodities_data):
        """Test valid commodities submission."""
        result = engine.submit_step(
            session.session_id, "commodities", valid_commodities_data
        )

        assert result.is_valid is True
        assert result.fields_completed == 2

    def test_submit_commodities_invalid_commodity(self, engine, session):
        """Test commodities with invalid EUDR commodity name."""
        result = engine.submit_step(session.session_id, "commodities", {
            "commodities": ["not_a_commodity"],
        })

        assert result.is_valid is False
        assert any("Invalid commodities" in e for e in result.errors)

    def test_submit_plots_valid(self, engine, session, valid_plots_data):
        """Test valid plot submission with GPS coordinates."""
        result = engine.submit_step(session.session_id, "plots", valid_plots_data)

        assert result.is_valid is True
        assert result.fields_completed == 1

    def test_submit_plots_polygon_required_over_4ha(self, engine, session):
        """Test EUDR Article 9(1)(d): polygon required for plots >4ha."""
        result = engine.submit_step(session.session_id, "plots", {
            "plots": [
                {
                    "latitude": -2.501234,
                    "longitude": -44.282345,
                    "area_hectares": 5.0,  # >4ha, needs polygon
                    "commodity": "soya",
                    "country_code": "BR",
                },
            ],
        })

        assert result.is_valid is False
        assert any("polygon_coordinates required" in e for e in result.errors)

    def test_submit_plots_polygon_not_required_under_4ha(self, engine, session):
        """Test that polygon is NOT required for plots <=4ha."""
        result = engine.submit_step(session.session_id, "plots", {
            "plots": [
                {
                    "latitude": -2.501234,
                    "longitude": -44.282345,
                    "area_hectares": 3.5,  # <=4ha, no polygon needed
                    "commodity": "soya",
                    "country_code": "BR",
                },
            ],
        })

        assert result.is_valid is True

    def test_submit_plots_insufficient_precision(self, engine, session):
        """Test that coordinates with <6 decimal places are rejected."""
        result = engine.submit_step(session.session_id, "plots", {
            "plots": [
                {
                    "latitude": -2.50,  # Only 2 decimal places
                    "longitude": -44.28,
                    "area_hectares": 2.0,
                    "commodity": "soya",
                    "country_code": "BR",
                },
            ],
        })

        assert result.is_valid is False
        assert any("decimal places" in e for e in result.errors)

    def test_submit_plots_invalid_coordinates(self, engine, session):
        """Test that out-of-range coordinates are rejected."""
        result = engine.submit_step(session.session_id, "plots", {
            "plots": [
                {
                    "latitude": 91.000000,  # Invalid: >90
                    "longitude": -44.282345,
                    "area_hectares": 2.0,
                    "commodity": "soya",
                    "country_code": "BR",
                },
            ],
        })

        assert result.is_valid is False
        assert any("latitude must be between" in e for e in result.errors)

    def test_submit_plots_mobile_gps_capture(self, engine, session):
        """Test plot submission with mobile GPS capture metadata."""
        result = engine.submit_step(session.session_id, "plots", {
            "plots": [
                {
                    "latitude": -2.501234,
                    "longitude": -44.282345,
                    "area_hectares": 2.0,
                    "commodity": "soya",
                    "country_code": "BR",
                    "capture_method": "gps",
                    "capture_accuracy_m": 5.0,
                },
            ],
        })

        assert result.is_valid is True
        assert any("mobile GPS" in m for m in result.info_messages)

    def test_submit_certifications_valid(self, engine, session, valid_certifications_data):
        """Test valid certification submission."""
        result = engine.submit_step(
            session.session_id, "certifications", valid_certifications_data
        )

        assert result.is_valid is True
        assert result.fields_completed == 1

    def test_submit_certifications_empty_is_valid_with_warning(self, engine, session):
        """Test that empty certifications are valid but produce a warning."""
        result = engine.submit_step(session.session_id, "certifications", {
            "certifications": [],
        })

        assert result.is_valid is True
        assert len(result.warnings) > 0

    def test_submit_declarations_valid(self, engine, session, valid_declarations_data):
        """Test valid declarations submission."""
        result = engine.submit_step(
            session.session_id, "declarations", valid_declarations_data
        )

        assert result.is_valid is True
        assert result.fields_completed == 2

    def test_submit_declarations_missing_deforestation_free(self, engine, session):
        """Test that missing deforestation-free declaration is rejected."""
        result = engine.submit_step(session.session_id, "declarations", {
            "deforestation_free_declaration": False,
            "legality_declaration": True,
        })

        assert result.is_valid is False
        assert any("Deforestation-free declaration" in e for e in result.errors)

    def test_submit_sub_tier_suppliers_valid(self, engine, session, valid_sub_tier_data):
        """Test valid sub-tier supplier submission."""
        result = engine.submit_step(
            session.session_id, "sub_tier_suppliers", valid_sub_tier_data
        )

        assert result.is_valid is True
        assert result.fields_completed == 1

    def test_submit_sub_tier_suppliers_empty_is_valid(self, engine, session):
        """Test that empty sub-tier suppliers is valid with warning."""
        result = engine.submit_step(session.session_id, "sub_tier_suppliers", {
            "sub_tier_suppliers": [],
        })

        assert result.is_valid is True
        assert len(result.warnings) > 0

    def test_submit_invalid_step_name(self, engine, session):
        """Test that invalid step name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid step name"):
            engine.submit_step(session.session_id, "nonexistent_step", {})

    def test_submit_to_nonexistent_session(self, engine):
        """Test that submitting to unknown session raises ValueError."""
        with pytest.raises(ValueError, match="Session not found"):
            engine.submit_step("nonexistent-session", "company_info", {})

    def test_submit_to_completed_session_raises(
        self, engine, session,
        valid_company_data, valid_commodities_data, valid_plots_data,
        valid_certifications_data, valid_declarations_data, valid_sub_tier_data,
    ):
        """Test that submitting to a completed session raises ValueError."""
        # Complete all steps
        engine.submit_step(session.session_id, "company_info", valid_company_data)
        engine.submit_step(session.session_id, "commodities", valid_commodities_data)
        engine.submit_step(session.session_id, "plots", valid_plots_data)
        engine.submit_step(session.session_id, "certifications", valid_certifications_data)
        engine.submit_step(session.session_id, "declarations", valid_declarations_data)
        engine.submit_step(session.session_id, "sub_tier_suppliers", valid_sub_tier_data)

        updated = engine.get_session(session.session_id)
        assert updated.status == OnboardingStatus.COMPLETED

        with pytest.raises(ValueError, match="Cannot submit"):
            engine.submit_step(session.session_id, "company_info", valid_company_data)


# ===========================================================================
# Test Suite: Completion Tracking
# ===========================================================================


class TestCompletionTracking:
    """Tests for per-supplier completion percentage tracking."""

    def test_initial_completion_zero(self, engine, session):
        """Test that new session starts at 0% completion."""
        assert session.completion_pct == 0.0

    def test_completion_increases_with_steps(
        self, engine, session, valid_company_data, valid_commodities_data,
    ):
        """Test that completion increases as steps are submitted."""
        engine.submit_step(session.session_id, "company_info", valid_company_data)
        updated = engine.get_session(session.session_id)
        assert updated.completion_pct > 0.0

        prev_pct = updated.completion_pct
        engine.submit_step(session.session_id, "commodities", valid_commodities_data)
        updated = engine.get_session(session.session_id)
        assert updated.completion_pct > prev_pct

    def test_completion_100_when_all_steps_done(
        self, engine, session,
        valid_company_data, valid_commodities_data, valid_plots_data,
        valid_certifications_data, valid_declarations_data, valid_sub_tier_data,
    ):
        """Test that completion reaches 100% when all steps are complete."""
        engine.submit_step(session.session_id, "company_info", valid_company_data)
        engine.submit_step(session.session_id, "commodities", valid_commodities_data)
        engine.submit_step(session.session_id, "plots", valid_plots_data)
        engine.submit_step(session.session_id, "certifications", valid_certifications_data)
        engine.submit_step(session.session_id, "declarations", valid_declarations_data)
        engine.submit_step(session.session_id, "sub_tier_suppliers", valid_sub_tier_data)

        updated = engine.get_session(session.session_id)
        assert updated.completion_pct == 100.0
        assert updated.status == OnboardingStatus.COMPLETED

    def test_status_transitions_invited_to_in_progress(
        self, engine, session, valid_company_data,
    ):
        """Test status transitions from INVITED to IN_PROGRESS on first submit."""
        assert session.status == OnboardingStatus.INVITED

        engine.submit_step(session.session_id, "company_info", valid_company_data)
        updated = engine.get_session(session.session_id)
        assert updated.status == OnboardingStatus.IN_PROGRESS

    def test_get_next_step(
        self, engine, session, valid_company_data,
    ):
        """Test that get_next_step returns the next incomplete step."""
        assert engine.get_next_step(session.session_id) == "company_info"

        engine.submit_step(session.session_id, "company_info", valid_company_data)
        assert engine.get_next_step(session.session_id) == "commodities"


# ===========================================================================
# Test Suite: Reminder System
# ===========================================================================


class TestReminderSystem:
    """Tests for the automated reminder system."""

    def test_reminders_scheduled_on_creation(self, engine, session):
        """Test that reminders are scheduled at correct intervals."""
        assert len(session.reminders) == len(DEFAULT_REMINDER_DAYS)
        for i, days in enumerate(DEFAULT_REMINDER_DAYS):
            expected = session.created_at + timedelta(days=days)
            assert session.reminders[i].scheduled_at == expected

    def test_process_due_reminders_none_due(self, engine, session):
        """Test that no reminders are processed when none are due."""
        processed = engine.process_due_reminders()
        assert len(processed) == 0

    def test_process_due_reminders_sends_pending(self, engine, session):
        """Test that due reminders are marked as sent."""
        # Make first reminder due by setting scheduled_at to past
        session.reminders[0].scheduled_at = _utcnow() - timedelta(hours=1)

        processed = engine.process_due_reminders()
        assert len(processed) == 1
        assert processed[0].status == ReminderStatus.SENT
        assert processed[0].sent_at is not None

    def test_process_due_reminders_with_notification_service(
        self, engine_with_mocks, mock_notification_service,
    ):
        """Test reminder delivery through notification service."""
        session = engine_with_mocks.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="Test",
            supplier_email="test@example.com",
            commodity="cocoa",
        )

        # Make first reminder due
        session.reminders[0].scheduled_at = _utcnow() - timedelta(hours=1)

        processed = engine_with_mocks.process_due_reminders()
        assert len(processed) == 1
        assert mock_notification_service.send_email.called

    def test_reminders_skipped_on_completion(
        self, engine, session,
        valid_company_data, valid_commodities_data, valid_plots_data,
        valid_certifications_data, valid_declarations_data, valid_sub_tier_data,
    ):
        """Test that pending reminders are skipped when session completes."""
        engine.submit_step(session.session_id, "company_info", valid_company_data)
        engine.submit_step(session.session_id, "commodities", valid_commodities_data)
        engine.submit_step(session.session_id, "plots", valid_plots_data)
        engine.submit_step(session.session_id, "certifications", valid_certifications_data)
        engine.submit_step(session.session_id, "declarations", valid_declarations_data)
        engine.submit_step(session.session_id, "sub_tier_suppliers", valid_sub_tier_data)

        updated = engine.get_session(session.session_id)
        for reminder in updated.reminders:
            assert reminder.status == ReminderStatus.SKIPPED

    def test_reminders_skipped_on_cancellation(self, engine, session):
        """Test that pending reminders are skipped when session is cancelled."""
        engine.cancel_session(session.session_id)
        updated = engine.get_session(session.session_id)
        for reminder in updated.reminders:
            assert reminder.status == ReminderStatus.SKIPPED


# ===========================================================================
# Test Suite: Bulk Import
# ===========================================================================


class TestBulkImport:
    """Tests for bulk supplier import from CSV."""

    def test_bulk_import_valid_csv(self, engine):
        """Test successful bulk import from valid CSV."""
        csv_content = (
            "supplier_name,country_code,commodity,supplier_email\n"
            "Supplier A,BR,soya,a@example.com\n"
            "Supplier B,ID,palm_oil,b@example.com\n"
            "Supplier C,GH,cocoa,c@example.com\n"
        )

        result = engine.bulk_import_from_csv(
            csv_content=csv_content,
            operator_id="op-001",
            graph_id="graph-001",
        )

        assert result.status == BulkImportStatus.COMPLETED
        assert result.total_rows == 3
        assert result.rows_succeeded == 3
        assert result.rows_failed == 0
        assert len(result.sessions_created) == 3
        assert result.processing_time_ms >= 0

    def test_bulk_import_partial_failures(self, engine):
        """Test bulk import with some invalid rows."""
        csv_content = (
            "supplier_name,country_code,commodity\n"
            "Valid Supplier,BR,soya\n"
            ",BR,soya\n"  # Missing supplier_name
            "Another Valid,ID,cocoa\n"
        )

        result = engine.bulk_import_from_csv(
            csv_content=csv_content,
            operator_id="op-001",
            graph_id="graph-001",
        )

        assert result.status == BulkImportStatus.PARTIAL
        assert result.rows_succeeded == 2
        assert result.rows_failed == 1
        assert 2 in result.errors  # Row 2 (1-indexed) has the error

    def test_bulk_import_missing_columns(self, engine):
        """Test bulk import with missing required columns."""
        csv_content = "name,location\nSupplier A,Brazil\n"

        result = engine.bulk_import_from_csv(
            csv_content=csv_content,
            operator_id="op-001",
            graph_id="graph-001",
        )

        assert result.status == BulkImportStatus.FAILED
        assert "Missing required columns" in result.errors.get(0, "")

    def test_bulk_import_empty_csv(self, engine):
        """Test bulk import with empty CSV."""
        result = engine.bulk_import_from_csv(
            csv_content="",
            operator_id="op-001",
            graph_id="graph-001",
        )

        assert result.status == BulkImportStatus.FAILED

    def test_bulk_import_creates_sessions(self, engine):
        """Test that bulk import creates retrievable sessions."""
        csv_content = (
            "supplier_name,country_code,commodity,supplier_email\n"
            "Import Test,BR,soya,import@example.com\n"
        )

        result = engine.bulk_import_from_csv(
            csv_content=csv_content,
            operator_id="op-001",
            graph_id="graph-001",
        )

        assert len(result.sessions_created) == 1
        session = engine.get_session(result.sessions_created[0])
        assert session is not None
        assert session.supplier_name == "Import Test"


# ===========================================================================
# Test Suite: Graph Entity Auto-Creation
# ===========================================================================


class TestGraphEntityCreation:
    """Tests for auto-creating graph nodes and edges from onboarding."""

    def test_create_graph_entities_from_completed_session(
        self, engine_with_mocks, mock_graph_engine, mock_geo_linker,
        valid_company_data, valid_commodities_data, valid_plots_data,
        valid_certifications_data, valid_declarations_data, valid_sub_tier_data,
    ):
        """Test graph entity creation from a completed onboarding session."""
        session = engine_with_mocks.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="Test Supplier",
            supplier_email="test@example.com",
            commodity="soya",
        )

        # Complete all steps
        engine_with_mocks.submit_step(session.session_id, "company_info", valid_company_data)
        engine_with_mocks.submit_step(
            session.session_id, "commodities", valid_commodities_data
        )
        engine_with_mocks.submit_step(session.session_id, "plots", valid_plots_data)
        engine_with_mocks.submit_step(
            session.session_id, "certifications", valid_certifications_data
        )
        engine_with_mocks.submit_step(
            session.session_id, "declarations", valid_declarations_data
        )
        engine_with_mocks.submit_step(
            session.session_id, "sub_tier_suppliers", valid_sub_tier_data
        )

        # Create graph entities
        result = asyncio.get_event_loop().run_until_complete(
            engine_with_mocks.create_graph_entities(session.session_id)
        )

        assert len(result["node_ids"]) >= 1
        assert mock_graph_engine.add_node.called
        assert mock_geo_linker.link_producer_to_plot.called

    def test_create_graph_entities_incomplete_session_raises(self, engine_with_mocks):
        """Test that creating entities from incomplete session raises ValueError."""
        session = engine_with_mocks.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="Test",
            supplier_email="test@example.com",
            commodity="cocoa",
        )

        with pytest.raises(ValueError, match="not completed"):
            asyncio.get_event_loop().run_until_complete(
                engine_with_mocks.create_graph_entities(session.session_id)
            )

    def test_create_graph_entities_no_graph_engine_raises(self, engine, session):
        """Test that missing graph engine raises RuntimeError."""
        # Force session to completed
        session.status = OnboardingStatus.COMPLETED

        with pytest.raises(RuntimeError, match="graph_engine is not configured"):
            asyncio.get_event_loop().run_until_complete(
                engine.create_graph_entities(session.session_id)
            )


# ===========================================================================
# Test Suite: Session Lifecycle
# ===========================================================================


class TestSessionLifecycle:
    """Tests for session lifecycle management."""

    def test_cancel_session(self, engine, session):
        """Test session cancellation."""
        result = engine.cancel_session(session.session_id)

        assert result is not None
        assert result.status == OnboardingStatus.CANCELLED

    def test_cancel_nonexistent_session(self, engine):
        """Test cancelling a nonexistent session returns None."""
        result = engine.cancel_session("nonexistent")
        assert result is None

    def test_list_sessions_no_filter(self, engine):
        """Test listing all sessions without filter."""
        engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="S1",
            supplier_email="s1@example.com",
            commodity="cocoa",
        )
        engine.create_onboarding_session(
            operator_id="op-002",
            graph_id="graph-002",
            supplier_name="S2",
            supplier_email="s2@example.com",
            commodity="soya",
        )

        sessions = engine.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_filter_by_operator(self, engine):
        """Test listing sessions filtered by operator_id."""
        engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="S1",
            supplier_email="s1@example.com",
            commodity="cocoa",
        )
        engine.create_onboarding_session(
            operator_id="op-002",
            graph_id="graph-002",
            supplier_name="S2",
            supplier_email="s2@example.com",
            commodity="soya",
        )

        sessions = engine.list_sessions(operator_id="op-001")
        assert len(sessions) == 1
        assert sessions[0].operator_id == "op-001"

    def test_list_sessions_filter_by_status(self, engine):
        """Test listing sessions filtered by status."""
        engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="S1",
            supplier_email="s1@example.com",
            commodity="cocoa",
        )
        s2 = engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="S2",
            supplier_email="s2@example.com",
            commodity="soya",
        )
        engine.cancel_session(s2.session_id)

        invited = engine.list_sessions(status=OnboardingStatus.INVITED)
        cancelled = engine.list_sessions(status=OnboardingStatus.CANCELLED)
        assert len(invited) == 1
        assert len(cancelled) == 1

    def test_session_count_property(self, engine):
        """Test session_count property."""
        assert engine.session_count == 0

        engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="S1",
            supplier_email="s1@example.com",
            commodity="cocoa",
        )
        assert engine.session_count == 1


# ===========================================================================
# Test Suite: Metrics and Reporting
# ===========================================================================


class TestMetrics:
    """Tests for aggregated onboarding metrics."""

    def test_metrics_empty(self, engine):
        """Test metrics with no sessions."""
        metrics = engine.get_metrics()
        assert metrics.total_sessions == 0
        assert metrics.average_completion_pct == 0.0

    def test_metrics_with_sessions(self, engine, valid_company_data):
        """Test metrics reflect session state."""
        s1 = engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="S1",
            supplier_email="s1@example.com",
            commodity="cocoa",
        )
        engine.submit_step(s1.session_id, "company_info", valid_company_data)

        metrics = engine.get_metrics()
        assert metrics.total_sessions == 1
        assert metrics.average_completion_pct > 0.0
        assert "in_progress" in metrics.sessions_by_status


# ===========================================================================
# Test Suite: Utility Methods
# ===========================================================================


class TestUtilityMethods:
    """Tests for utility and helper methods."""

    def test_generate_onboarding_link(self, engine, session):
        """Test onboarding link generation."""
        link = engine.generate_onboarding_link(session.session_id)
        assert link is not None
        assert "token=" in link
        assert session.token in link

    def test_generate_onboarding_link_custom_base_url(self, engine, session):
        """Test onboarding link with custom base URL."""
        link = engine.generate_onboarding_link(
            session.session_id,
            base_url="https://custom.example.com/onboard",
        )
        assert link.startswith("https://custom.example.com/onboard?token=")

    def test_generate_onboarding_link_nonexistent_session(self, engine):
        """Test that nonexistent session returns None."""
        link = engine.generate_onboarding_link("nonexistent")
        assert link is None

    def test_mobile_gps_config(self, engine):
        """Test mobile GPS configuration output."""
        config = engine.get_mobile_gps_config()

        assert config["enableHighAccuracy"] is True
        assert config["timeout"] == 30000
        assert config["requiredPrecision"] == MIN_COORDINATE_PRECISION
        assert config["eudrRequirements"]["minDecimalPlaces"] == MIN_COORDINATE_PRECISION
        assert config["eudrRequirements"]["polygonThresholdHa"] == POLYGON_AREA_THRESHOLD_HA
        assert config["eudrRequirements"]["coordinateSystem"] == "WGS84"

    def test_count_decimal_places(self, engine):
        """Test decimal place counting utility."""
        assert engine._count_decimal_places(1.0) == 1
        assert engine._count_decimal_places(1.123456) == 6
        assert engine._count_decimal_places(1.12345678) == 8
        assert engine._count_decimal_places(1) == 0

    def test_compute_hash_deterministic(self):
        """Test that hash computation is deterministic."""
        data = {"key": "value", "number": 42}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_generate_id_format(self):
        """Test ID generation format."""
        id1 = _generate_id("TEST")
        assert id1.startswith("TEST-")
        assert len(id1) == 17  # "TEST-" (5) + hex[:12] (12)


# ===========================================================================
# Test Suite: Constants and Configuration
# ===========================================================================


class TestConstants:
    """Tests for module constants and configuration."""

    def test_supported_languages(self):
        """Test supported languages list."""
        assert "en" in SUPPORTED_LANGUAGES
        assert "fr" in SUPPORTED_LANGUAGES
        assert "de" in SUPPORTED_LANGUAGES
        assert "es" in SUPPORTED_LANGUAGES
        assert "pt" in SUPPORTED_LANGUAGES
        assert "id" in SUPPORTED_LANGUAGES
        assert len(SUPPORTED_LANGUAGES) == 6

    def test_wizard_steps_order(self):
        """Test wizard steps are in correct order."""
        assert WIZARD_STEPS == (
            "company_info",
            "commodities",
            "plots",
            "certifications",
            "declarations",
            "sub_tier_suppliers",
        )

    def test_total_field_groups_matches_step_fields(self):
        """Test TOTAL_FIELD_GROUPS is consistent with STEP_FIELDS."""
        expected = sum(len(fields) for fields in STEP_FIELDS.values())
        assert TOTAL_FIELD_GROUPS == expected

    def test_step_fields_keys_match_wizard_steps(self):
        """Test STEP_FIELDS keys match WIZARD_STEPS."""
        assert set(STEP_FIELDS.keys()) == set(WIZARD_STEPS)


# ===========================================================================
# Test Suite: Pydantic Models
# ===========================================================================


class TestPydanticModels:
    """Tests for Pydantic data model validation."""

    def test_plot_data_valid(self):
        """Test PlotData creation with valid data."""
        plot = PlotData(
            latitude=-2.501234,
            longitude=-44.282345,
            area_hectares=3.5,
            commodity="soya",
            country_code="br",
        )
        assert plot.country_code == "BR"  # Normalized to uppercase

    def test_plot_data_invalid_capture_method(self):
        """Test PlotData rejects invalid capture method."""
        with pytest.raises(ValueError, match="capture_method must be one of"):
            PlotData(
                latitude=-2.501234,
                longitude=-44.282345,
                area_hectares=3.5,
                commodity="soya",
                country_code="BR",
                capture_method="satellite",
            )

    def test_sub_tier_supplier_data_valid(self):
        """Test SubTierSupplierData creation."""
        supplier = SubTierSupplierData(
            supplier_name="Upstream Co",
            country_code="id",
            commodities=["palm_oil"],
        )
        assert supplier.country_code == "ID"

    def test_onboarding_session_invalid_language(self):
        """Test OnboardingSession rejects invalid language."""
        with pytest.raises(ValueError, match="language must be one of"):
            OnboardingSession(
                operator_id="op-001",
                graph_id="graph-001",
                supplier_name="Test",
                supplier_email="test@example.com",
                commodity="cocoa",
                language="xx",
            )

    def test_certification_data_valid(self):
        """Test CertificationData creation."""
        cert = CertificationData(
            certification_type="FSC",
            certificate_number="FSC-C123456",
            issuing_body="FSC International",
        )
        assert cert.certification_type == "FSC"


# ===========================================================================
# Test Suite: Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_submit_expired_session_raises(self, engine):
        """Test that submitting to expired session raises ValueError."""
        session = engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="Test",
            supplier_email="test@example.com",
            commodity="cocoa",
        )
        # Force token expiry
        session.token_expires_at = _utcnow() - timedelta(hours=1)

        with pytest.raises(ValueError, match="token has expired"):
            engine.submit_step(session.session_id, "company_info", {
                "legal_name": "Test",
                "country_code": "BR",
                "registration_id": "123",
                "contact_name": "Name",
                "contact_email": "test@example.com",
                "contact_phone": "+1234567890",
            })

    def test_resubmit_step_updates_data(self, engine, session, valid_company_data):
        """Test that resubmitting a step updates the data."""
        engine.submit_step(session.session_id, "company_info", valid_company_data)

        updated_data = {**valid_company_data, "legal_name": "Updated Name"}
        engine.submit_step(session.session_id, "company_info", updated_data)

        updated = engine.get_session(session.session_id)
        assert updated.company_info["legal_name"] == "Updated Name"

    def test_plots_with_polygon_over_4ha_passes(self, engine, session):
        """Test plot >4ha with polygon provided passes validation."""
        result = engine.submit_step(session.session_id, "plots", {
            "plots": [
                {
                    "latitude": -2.501234,
                    "longitude": -44.282345,
                    "area_hectares": 10.0,
                    "commodity": "soya",
                    "country_code": "BR",
                    "polygon_coordinates": [
                        [-44.28, -2.50],
                        [-44.27, -2.50],
                        [-44.27, -2.51],
                        [-44.28, -2.51],
                        [-44.28, -2.50],
                    ],
                },
            ],
        })

        assert result.is_valid is True

    def test_provenance_hash_updated_on_each_step(
        self, engine, session, valid_company_data, valid_commodities_data,
    ):
        """Test that session provenance hash changes with each step."""
        initial_hash = session.provenance_hash

        engine.submit_step(session.session_id, "company_info", valid_company_data)
        updated = engine.get_session(session.session_id)
        hash_after_step1 = updated.provenance_hash

        engine.submit_step(session.session_id, "commodities", valid_commodities_data)
        updated = engine.get_session(session.session_id)
        hash_after_step2 = updated.provenance_hash

        assert initial_hash != hash_after_step1
        assert hash_after_step1 != hash_after_step2

    def test_notification_service_failure_marks_reminder_failed(self):
        """Test that notification service failure is handled gracefully."""
        mock_svc = MagicMock()
        mock_svc.send_email = MagicMock(side_effect=Exception("SMTP error"))

        engine = SupplierOnboardingEngine(notification_service=mock_svc)
        session = engine.create_onboarding_session(
            operator_id="op-001",
            graph_id="graph-001",
            supplier_name="Test",
            supplier_email="test@example.com",
            commodity="cocoa",
        )

        session.reminders[0].scheduled_at = _utcnow() - timedelta(hours=1)
        processed = engine.process_due_reminders()

        assert len(processed) == 1
        assert processed[0].status == ReminderStatus.FAILED
        assert "SMTP error" in processed[0].error_message
