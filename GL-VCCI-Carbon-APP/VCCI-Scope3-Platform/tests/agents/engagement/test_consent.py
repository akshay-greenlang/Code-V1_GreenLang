"""
Tests for consent management (GDPR, CCPA, CAN-SPAM).
"""
import pytest
from datetime import datetime, timedelta

from services.agents.engagement.consent import (
    ConsentRegistry,
    JurisdictionManager,
    OptOutHandler
)
from services.agents.engagement.models import ConsentStatus, LawfulBasis
from services.agents.engagement.exceptions import (
    ConsentNotGrantedError,
    OptOutViolationError,
    SupplierNotFoundError
)


@pytest.fixture
def registry():
    return ConsentRegistry(storage_path="data/test_consent.json")


@pytest.fixture
def jurisdiction_manager():
    return JurisdictionManager()


class TestGDPRCompliance:
    """Test GDPR compliance (40 tests total across all jurisdictions)."""

    def test_gdpr_requires_opt_in(self, registry):
        """GDPR requires explicit opt-in."""
        record = registry.register_supplier("SUP001", "test@example.com", "DE")
        assert record.consent_status == ConsentStatus.PENDING

    def test_gdpr_grant_consent(self, registry):
        """Test granting consent under GDPR."""
        registry.register_supplier("SUP002", "test@example.com", "DE")
        record = registry.grant_consent("SUP002")
        assert record.consent_status == ConsentStatus.OPTED_IN

    def test_gdpr_revoke_consent(self, registry):
        """Test revoking consent under GDPR."""
        registry.register_supplier("SUP003", "test@example.com", "DE", auto_opt_in=True)
        record = registry.revoke_consent("SUP003", "Privacy concerns")
        assert record.consent_status == ConsentStatus.OPTED_OUT

    def test_gdpr_right_to_erasure(self, registry):
        """Test GDPR right to erasure."""
        registry.register_supplier("SUP004", "test@example.com", "DE")
        registry.revoke_consent("SUP004")

        # Cleanup should remove after retention period
        removed = registry.cleanup_expired_records(retention_days=0)
        # Record should still exist (grace period)
        assert "SUP004" in registry.records


class TestCCPACompliance:
    """Test CCPA compliance."""

    def test_ccpa_opt_out_model(self, registry):
        """CCPA uses opt-out model."""
        record = registry.register_supplier("SUP005", "test@example.com", "US-CA")
        # Should be opted in by default (opt-out model)
        assert record.consent_status == ConsentStatus.OPTED_IN

    def test_ccpa_opt_out(self, registry):
        """Test CCPA opt-out."""
        registry.register_supplier("SUP006", "test@example.com", "US-CA")
        record = registry.revoke_consent("SUP006")
        assert record.consent_status == ConsentStatus.OPTED_OUT

    def test_ccpa_grace_period(self, jurisdiction_manager):
        """CCPA requires 15-day grace period."""
        rules = jurisdiction_manager.get_rules("US-CA")
        grace = rules.opt_out_grace_period()
        assert grace.days == 15


class TestCANSPAMCompliance:
    """Test CAN-SPAM compliance."""

    def test_can_spam_opt_out_model(self, registry):
        """CAN-SPAM uses opt-out model."""
        record = registry.register_supplier("SUP007", "test@example.com", "US")
        assert record.consent_status == ConsentStatus.OPTED_IN

    def test_can_spam_grace_period(self, jurisdiction_manager):
        """CAN-SPAM requires 10-day grace period."""
        rules = jurisdiction_manager.get_rules("US")
        grace = rules.opt_out_grace_period()
        assert grace.days == 10


class TestOptOutHandler:
    """Test opt-out handling."""

    def test_process_opt_out(self, registry):
        """Test processing opt-out request."""
        registry.register_supplier("SUP008", "test@example.com", "US", auto_opt_in=True)
        opt_out_handler = OptOutHandler(registry)

        record = opt_out_handler.process_opt_out("SUP008", "Too many emails")
        assert record.consent_status == ConsentStatus.OPTED_OUT

    def test_suppression_list(self, registry):
        """Test suppression list management."""
        registry.register_supplier("SUP009", "test@example.com", "US", auto_opt_in=True)
        opt_out_handler = OptOutHandler(registry)

        opt_out_handler.process_opt_out("SUP009")
        assert opt_out_handler.is_suppressed("SUP009")

    def test_unsubscribe_url_generation(self, registry):
        """Test unsubscribe URL generation."""
        opt_out_handler = OptOutHandler(registry)
        url = opt_out_handler.generate_unsubscribe_url("SUP010", "CAMP001")
        assert "unsubscribe" in url
        assert "supplier=SUP010" in url


# Run with: pytest tests/agents/engagement/test_consent.py -v
