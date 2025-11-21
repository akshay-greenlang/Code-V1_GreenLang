# -*- coding: utf-8 -*-
"""
Monetization and Payment Tests

Comprehensive tests for payment processing, subscriptions, refunds,
and revenue analytics.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from greenlang.marketplace.models import AgentPurchase, MarketplaceAgent
from greenlang.marketplace.monetization import (
from greenlang.determinism import DeterministicClock
    MonetizationManager,
    PaymentProcessor,
    PaymentIntent,
    RefundRequest,
    PricingModel,
    PaymentStatus,
)
from greenlang.marketplace.license_manager import (
    LicenseManager,
    LicenseGenerator,
    LicenseValidator,
    LicenseKey,
    LicenseType,
    LicenseStatus,
)


class TestPaymentProcessing:
    """Test payment processing"""

    def test_create_payment_intent_success(self, db_session, sample_agent):
        """Test successful payment intent creation"""
        processor = PaymentProcessor(db_session)

        # Mock agent query
        db_session.query().filter().first = Mock(return_value=sample_agent)

        intent = PaymentIntent(
            amount=sample_agent.price,
            currency="USD",
            agent_id=str(sample_agent.id),
            user_id="user_123",
            pricing_type="one_time",
            metadata={}
        )

        success, payment_id, errors = processor.create_payment_intent(intent)

        assert success == True
        assert payment_id is not None
        assert payment_id.startswith("pi_")
        assert len(errors) == 0

    def test_create_payment_intent_invalid_amount(self, db_session, sample_agent):
        """Test payment intent with invalid amount"""
        processor = PaymentProcessor(db_session)

        # Mock agent query
        db_session.query().filter().first = Mock(return_value=sample_agent)

        intent = PaymentIntent(
            amount=Decimal("999.99"),  # Wrong amount
            currency="USD",
            agent_id=str(sample_agent.id),
            user_id="user_123",
            pricing_type="one_time",
            metadata={}
        )

        success, payment_id, errors = processor.create_payment_intent(intent)

        assert success == False
        assert "Amount mismatch" in errors

    def test_confirm_payment(self, db_session, sample_agent):
        """Test payment confirmation"""
        processor = PaymentProcessor(db_session)

        # Mock queries
        db_session.query().filter().first = Mock(return_value=sample_agent)

        success, purchase, errors = processor.confirm_payment(
            payment_intent_id="pi_123456",
            agent_id=str(sample_agent.id),
            user_id="user_123",
            amount=Decimal("29.99"),
            currency="USD"
        )

        assert db_session.add.called
        assert db_session.commit.called

    def test_create_subscription_monthly(self, db_session, sample_agent):
        """Test monthly subscription creation"""
        processor = PaymentProcessor(db_session)

        # Mock agent query
        db_session.query().filter().first = Mock(return_value=sample_agent)

        success, subscription_id, errors = processor.create_subscription(
            agent_id=str(sample_agent.id),
            user_id="user_123",
            pricing_type=PricingModel.MONTHLY
        )

        assert subscription_id is not None
        assert subscription_id.startswith("sub_")

    def test_create_subscription_annual(self, db_session, sample_agent):
        """Test annual subscription creation"""
        processor = PaymentProcessor(db_session)

        db_session.query().filter().first = Mock(return_value=sample_agent)

        success, subscription_id, errors = processor.create_subscription(
            agent_id=str(sample_agent.id),
            user_id="user_123",
            pricing_type=PricingModel.ANNUAL
        )

        assert success == True


class TestRefunds:
    """Test refund processing"""

    def test_process_refund_success(self, db_session, sample_purchase):
        """Test successful refund"""
        processor = PaymentProcessor(db_session)

        # Mock purchase query
        mock_agent = Mock()
        mock_agent.total_revenue = Decimal("100.00")
        db_session.query().filter().first = Mock(side_effect=[sample_purchase, mock_agent])

        request = RefundRequest(
            purchase_id=str(sample_purchase.id),
            amount=sample_purchase.amount,
            reason="Customer request"
        )

        success, errors = processor.process_refund(request)

        assert db_session.commit.called

    def test_process_refund_expired(self, db_session):
        """Test refund request after expiration"""
        processor = PaymentProcessor(db_session)

        # Create old purchase
        old_purchase = AgentPurchase(
            id="purchase_123",
            user_id="user_123",
            agent_id="agent_456",
            amount=Decimal("29.99"),
            currency="USD",
            transaction_id="txn_123",
            status=PaymentStatus.COMPLETED.value,
            purchased_at=DeterministicClock.utcnow() - timedelta(days=30)  # 30 days ago
        )

        db_session.query().filter().first = Mock(return_value=old_purchase)

        request = RefundRequest(
            purchase_id=str(old_purchase.id),
            amount=None,
            reason="Too late"
        )

        success, errors = processor.process_refund(request)

        assert success == False
        assert any("expired" in err.lower() for err in errors)

    def test_process_refund_excessive_amount(self, db_session, sample_purchase):
        """Test refund with amount exceeding purchase"""
        processor = PaymentProcessor(db_session)

        db_session.query().filter().first = Mock(return_value=sample_purchase)

        request = RefundRequest(
            purchase_id=str(sample_purchase.id),
            amount=Decimal("1000.00"),  # More than purchase
            reason="Excessive"
        )

        success, errors = processor.process_refund(request)

        assert success == False
        assert any("exceeds" in err.lower() for err in errors)


class TestLicenseGeneration:
    """Test license key generation and validation"""

    def test_generate_license_key(self):
        """Test license key generation"""
        key = LicenseGenerator.generate("agent_123", "user_456")

        assert isinstance(key, str)
        assert len(key.split('-')) == 4
        assert all(len(part) == 4 for part in key.split('-'))
        assert key.isupper()

    def test_generate_unique_keys(self):
        """Test that generated keys are unique"""
        key1 = LicenseGenerator.generate("agent_123", "user_456")
        key2 = LicenseGenerator.generate("agent_123", "user_456")

        # Should be different due to unique ID
        assert key1 != key2

    def test_verify_valid_signature(self):
        """Test verification of valid signature"""
        key = LicenseGenerator.generate("agent_123", "user_456")

        valid = LicenseGenerator.verify_signature(key)

        assert valid == True

    def test_verify_invalid_signature(self):
        """Test verification of invalid signature"""
        invalid_key = "AAAA-BBBB-CCCC-DDDD"

        valid = LicenseGenerator.verify_signature(invalid_key)

        assert valid == False

    def test_verify_malformed_key(self):
        """Test verification of malformed key"""
        malformed_keys = [
            "TOOLONG-KEY-HERE-NOW",
            "SHORT",
            "",
            "AAA-BBB-CCC"  # Missing part
        ]

        for key in malformed_keys:
            valid = LicenseGenerator.verify_signature(key)
            assert valid == False


class TestLicenseValidation:
    """Test license validation"""

    def test_validate_nonexistent_license(self, db_session):
        """Test validation of non-existent license"""
        validator = LicenseValidator(db_session)

        # Mock no purchase found
        db_session.query().filter().first = Mock(return_value=None)

        key = LicenseGenerator.generate("agent_123", "user_456")

        result = validator.validate(key)

        assert result.valid == False
        assert any("not found" in err.lower() for err in result.errors)

    def test_validate_refunded_license(self, db_session):
        """Test validation of refunded license"""
        validator = LicenseValidator(db_session)

        # Create refunded purchase
        refunded_purchase = AgentPurchase(
            id="purchase_123",
            user_id="user_456",
            agent_id="agent_123",
            amount=Decimal("29.99"),
            currency="USD",
            transaction_id="txn_123",
            license_key=LicenseGenerator.generate("agent_123", "user_456"),
            status="refunded"
        )

        db_session.query().filter().first = Mock(return_value=refunded_purchase)

        result = validator.validate(refunded_purchase.license_key)

        assert result.valid == False
        assert any("refunded" in err.lower() for err in result.errors)

    def test_validate_expired_subscription(self, db_session):
        """Test validation of expired subscription"""
        validator = LicenseValidator(db_session)

        # Create expired subscription
        expired_purchase = AgentPurchase(
            id="purchase_123",
            user_id="user_456",
            agent_id="agent_123",
            amount=Decimal("9.99"),
            currency="USD",
            transaction_id="txn_123",
            license_key=LicenseGenerator.generate("agent_123", "user_456"),
            status="completed",
            subscription_period_end=DeterministicClock.utcnow() - timedelta(days=30)
        )

        db_session.query().filter().first = Mock(return_value=expired_purchase)

        result = validator.validate(expired_purchase.license_key)

        assert result.valid == False
        assert any("expired" in err.lower() for err in result.errors)

    def test_validate_grace_period(self, db_session):
        """Test validation during grace period"""
        validator = LicenseValidator(db_session)

        # Create subscription in grace period (expired 3 days ago)
        grace_purchase = AgentPurchase(
            id="purchase_123",
            user_id="user_456",
            agent_id="agent_123",
            amount=Decimal("9.99"),
            currency="USD",
            transaction_id="txn_123",
            license_key=LicenseGenerator.generate("agent_123", "user_456"),
            status="completed",
            subscription_period_end=DeterministicClock.utcnow() - timedelta(days=3)
        )

        db_session.query().filter().first = Mock(return_value=grace_purchase)
        db_session.query().filter().count = Mock(return_value=0)

        result = validator.validate(grace_purchase.license_key)

        # Should be valid but with warning
        assert result.valid == True
        assert any("grace" in err.lower() for err in result.errors)


class TestLicenseActivation:
    """Test license activation and deactivation"""

    def test_activate_license_success(self, db_session, sample_purchase):
        """Test successful license activation"""
        manager = LicenseManager(db_session)

        # Mock validation
        mock_validation = Mock()
        mock_validation.valid = True
        manager.validator.validate = Mock(return_value=mock_validation)

        db_session.query().filter().first = Mock(side_effect=[sample_purchase, None])

        success, activation_id, errors = manager.activate_license(
            license_key=sample_purchase.license_key,
            machine_id="machine_001",
            agent_id=str(sample_purchase.agent_id)
        )

        assert db_session.add.called
        assert db_session.commit.called

    def test_activate_license_invalid(self, db_session):
        """Test activation with invalid license"""
        manager = LicenseManager(db_session)

        # Mock invalid validation
        mock_validation = Mock()
        mock_validation.valid = False
        mock_validation.errors = ["Invalid license"]
        manager.validator.validate = Mock(return_value=mock_validation)

        success, activation_id, errors = manager.activate_license(
            license_key="INVALID-KEY-1234-5678",
            machine_id="machine_001",
            agent_id="agent_123"
        )

        assert success == False
        assert len(errors) > 0

    def test_activate_already_activated(self, db_session, sample_purchase):
        """Test reactivating already activated license"""
        manager = LicenseManager(db_session)

        # Mock validation
        mock_validation = Mock()
        mock_validation.valid = True
        manager.validator.validate = Mock(return_value=mock_validation)

        # Mock existing installation
        from greenlang.marketplace.models import AgentInstall
        existing_install = AgentInstall(
            id="install_123",
            user_id=sample_purchase.user_id,
            agent_id=sample_purchase.agent_id,
            installation_id="machine_001",
            active=False
        )

        db_session.query().filter().first = Mock(side_effect=[sample_purchase, existing_install])

        success, activation_id, errors = manager.activate_license(
            license_key=sample_purchase.license_key,
            machine_id="machine_001",
            agent_id=str(sample_purchase.agent_id)
        )

        # Should reactivate
        assert db_session.commit.called

    def test_deactivate_license_success(self, db_session, sample_purchase):
        """Test successful license deactivation"""
        manager = LicenseManager(db_session)

        # Mock active installation
        from greenlang.marketplace.models import AgentInstall
        active_install = AgentInstall(
            id="install_123",
            user_id=sample_purchase.user_id,
            agent_id=sample_purchase.agent_id,
            installation_id="machine_001",
            active=True
        )

        db_session.query().filter().first = Mock(side_effect=[sample_purchase, active_install])

        success, errors = manager.deactivate_license(
            license_key=sample_purchase.license_key,
            machine_id="machine_001"
        )

        assert db_session.commit.called

    def test_deactivate_nonexistent_activation(self, db_session, sample_purchase):
        """Test deactivation of non-existent activation"""
        manager = LicenseManager(db_session)

        db_session.query().filter().first = Mock(side_effect=[sample_purchase, None])

        success, errors = manager.deactivate_license(
            license_key=sample_purchase.license_key,
            machine_id="machine_999"
        )

        assert success == False
        assert any("not found" in err.lower() for err in errors)


class TestRevenueAnalytics:
    """Test revenue analytics and reporting"""

    def test_get_revenue_stats(self, db_session):
        """Test revenue statistics calculation"""
        manager = MonetizationManager(db_session)

        # Mock query results
        db_session.query().filter().with_entities().scalar = Mock(return_value=Decimal("1000.00"))
        db_session.query().filter().count = Mock(return_value=50)
        db_session.query().join().filter().group_by().order_by().limit().all = Mock(return_value=[])

        stats = manager.get_revenue_stats(
            agent_id="agent_123",
            period_days=30
        )

        assert stats.total_revenue >= 0
        assert stats.total_purchases >= 0

    def test_platform_fee_calculation(self, db_session, sample_agent):
        """Test platform fee calculation"""
        sample_agent.total_revenue = Decimal("100.00")
        sample_agent.platform_fee_percent = 20

        author_revenue = sample_agent.author_revenue

        assert author_revenue == Decimal("80.00")  # 80% to author

    def test_revenue_by_period(self, db_session):
        """Test revenue calculation for specific period"""
        manager = MonetizationManager(db_session)

        # Mock different period revenues
        db_session.query().filter().with_entities().scalar = Mock(
            side_effect=[Decimal("1000.00"), Decimal("200.00")]  # Total, Period
        )
        db_session.query().filter().count = Mock(side_effect=[100, 20])
        db_session.query().join().filter().group_by().order_by().limit().all = Mock(return_value=[])

        stats = manager.get_revenue_stats(period_days=7)

        assert stats.total_revenue == Decimal("1000.00")
        assert stats.period_revenue == Decimal("200.00")


# Fixtures
@pytest.fixture
def db_session():
    """Mock database session"""
    session = Mock()
    session.query = Mock(return_value=Mock())
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    return session


@pytest.fixture
def sample_agent():
    """Sample agent for testing"""
    return MarketplaceAgent(
        id="agent_123",
        name="Test Agent",
        slug="test-agent",
        description="A test agent",
        author_id="author_123",
        author_name="Test Author",
        price=Decimal("29.99"),
        currency="USD",
        pricing_type="one_time",
        status="published"
    )


@pytest.fixture
def sample_purchase():
    """Sample purchase for testing"""
    return AgentPurchase(
        id="purchase_123",
        user_id="user_456",
        agent_id="agent_123",
        amount=Decimal("29.99"),
        currency="USD",
        transaction_id="txn_abc123",
        license_key=LicenseGenerator.generate("agent_123", "user_456"),
        pricing_type="one_time",
        status=PaymentStatus.COMPLETED.value,
        purchased_at=DeterministicClock.utcnow()
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=greenlang.marketplace.monetization", "--cov-report=html"])
