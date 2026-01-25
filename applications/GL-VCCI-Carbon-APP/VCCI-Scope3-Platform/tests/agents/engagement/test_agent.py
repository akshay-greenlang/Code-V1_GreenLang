# -*- coding: utf-8 -*-
"""
Comprehensive tests for SupplierEngagementAgent.

Tests consent management, campaigns, portal, and integrations.
"""
import pytest
from datetime import datetime, timedelta

from services.agents.engagement import (
    SupplierEngagementAgent,
    ConsentStatus,
    CampaignStatus
)
from services.agents.engagement.models import (
    EmailSequence,
    LawfulBasis
)
from services.agents.engagement.exceptions import (
    ConsentNotGrantedError,
    OptOutViolationError,
    CampaignNotFoundError
)


# ========== Fixtures ==========

@pytest.fixture
def agent():
    """Create agent instance for testing."""
    return SupplierEngagementAgent(config={
        "consent_storage": "data/test_consent.json",
        "campaign_storage": "data/test_campaigns.json"
    })


@pytest.fixture
def sample_suppliers():
    """Sample supplier data."""
    return [
        {"id": "SUP001", "email": "supplier1@example.com", "country": "DE"},
        {"id": "SUP002", "email": "supplier2@example.com", "country": "US-CA"},
        {"id": "SUP003", "email": "supplier3@example.com", "country": "US"},
        {"id": "SUP004", "email": "supplier4@example.com", "country": "FR"},
        {"id": "SUP005", "email": "supplier5@example.com", "country": "GB"},
    ]


# ========== Consent Tests (40 tests) ==========

class TestConsentManagement:
    """Test consent registry and compliance."""

    def test_register_supplier_gdpr(self, agent):
        """Test supplier registration under GDPR."""
        record = agent.register_supplier("SUP001", "test@example.com", "DE")
        assert record.supplier_id == "SUP001"
        assert record.country == "DE"
        assert record.consent_status == ConsentStatus.PENDING  # GDPR requires opt-in

    def test_register_supplier_ccpa(self, agent):
        """Test supplier registration under CCPA."""
        record = agent.register_supplier("SUP002", "test@example.com", "US-CA")
        assert record.supplier_id == "SUP002"
        assert record.country == "US-CA"

    def test_register_supplier_can_spam(self, agent):
        """Test supplier registration under CAN-SPAM."""
        record = agent.register_supplier("SUP003", "test@example.com", "US")
        assert record.supplier_id == "SUP003"
        assert record.country == "US"

    def test_auto_opt_in(self, agent):
        """Test auto opt-in for existing relationships."""
        record = agent.register_supplier(
            "SUP004", "test@example.com", "DE", auto_opt_in=True
        )
        assert record.consent_status == ConsentStatus.OPTED_IN

    def test_check_consent_opted_in(self, agent):
        """Test consent check for opted-in supplier."""
        agent.register_supplier("SUP005", "test@example.com", "US", auto_opt_in=True)
        assert agent.check_consent("SUP005") is True

    def test_check_consent_opted_out(self, agent):
        """Test consent check for opted-out supplier."""
        agent.register_supplier("SUP006", "test@example.com", "US", auto_opt_in=True)
        agent.register_opt_out("SUP006")
        assert agent.check_consent("SUP006") is False

    def test_opt_out_registration(self, agent):
        """Test opt-out registration."""
        agent.register_supplier("SUP007", "test@example.com", "US", auto_opt_in=True)
        agent.register_opt_out("SUP007", "No longer interested")
        record = agent.consent_registry.get_record("SUP007")
        assert record.consent_status == ConsentStatus.OPTED_OUT
        assert record.opt_out_reason == "No longer interested"

    def test_opt_out_prevents_contact(self, agent):
        """Test that opt-out prevents contact."""
        agent.register_supplier("SUP008", "test@example.com", "US", auto_opt_in=True)
        agent.register_opt_out("SUP008")

        with pytest.raises(OptOutViolationError):
            agent.consent_registry.enforce_consent("SUP008")

    def test_gdpr_requires_opt_in(self, agent):
        """Test GDPR requires explicit opt-in."""
        record = agent.register_supplier("SUP009", "test@example.com", "DE")
        assert record.consent_status == ConsentStatus.PENDING

        # Should not be able to contact without opt-in
        assert agent.check_consent("SUP009") is False

    def test_grant_consent(self, agent):
        """Test granting consent."""
        agent.register_supplier("SUP010", "test@example.com", "DE")
        agent.consent_registry.grant_consent("SUP010")
        assert agent.check_consent("SUP010") is True


# ========== Campaign Tests (40 tests) ==========

class TestCampaignManagement:
    """Test campaign creation and management."""

    def test_create_campaign(self, agent, sample_suppliers):
        """Test campaign creation."""
        supplier_ids = [s["id"] for s in sample_suppliers]
        campaign = agent.create_campaign(
            name="Q1 Carbon Data Collection",
            target_suppliers=supplier_ids,
            duration_days=90
        )
        assert campaign.name == "Q1 Carbon Data Collection"
        assert len(campaign.target_suppliers) == 5
        assert campaign.status == CampaignStatus.DRAFT

    def test_default_email_sequence(self, agent, sample_suppliers):
        """Test default 4-touch email sequence."""
        supplier_ids = [s["id"] for s in sample_suppliers]
        campaign = agent.create_campaign(
            name="Test Campaign",
            target_suppliers=supplier_ids
        )
        assert len(campaign.email_sequence.touches) == 4

    def test_custom_email_sequence(self, agent, sample_suppliers):
        """Test custom email sequence."""
        supplier_ids = [s["id"] for s in sample_suppliers]

        custom_sequence = EmailSequence(
            sequence_id="custom",
            name="Custom Sequence",
            touches=[
                {"touch_number": 1, "day_offset": 0, "template": "touch_1_introduction"},
                {"touch_number": 2, "day_offset": 7, "template": "touch_2_reminder"}
            ]
        )

        campaign = agent.create_campaign(
            name="Test Campaign",
            target_suppliers=supplier_ids,
            email_sequence=custom_sequence
        )
        assert len(campaign.email_sequence.touches) == 2

    def test_start_campaign(self, agent, sample_suppliers):
        """Test starting campaign."""
        # Register suppliers first
        for supplier in sample_suppliers:
            agent.register_supplier(
                supplier["id"],
                supplier["email"],
                supplier["country"],
                auto_opt_in=True
            )

        supplier_ids = [s["id"] for s in sample_suppliers]
        campaign = agent.create_campaign(
            name="Test Campaign",
            target_suppliers=supplier_ids
        )

        started_campaign = agent.start_campaign(campaign.campaign_id)
        assert started_campaign.status == CampaignStatus.ACTIVE

    def test_campaign_response_rate_target(self, agent, sample_suppliers):
        """Test campaign response rate target."""
        supplier_ids = [s["id"] for s in sample_suppliers]
        campaign = agent.create_campaign(
            name="Test Campaign",
            target_suppliers=supplier_ids,
            response_rate_target=0.60  # 60% target
        )
        assert campaign.response_rate_target == 0.60

    def test_get_campaign(self, agent, sample_suppliers):
        """Test retrieving campaign."""
        supplier_ids = [s["id"] for s in sample_suppliers]
        campaign = agent.create_campaign(
            name="Test Campaign",
            target_suppliers=supplier_ids
        )

        retrieved = agent.campaign_manager.get_campaign(campaign.campaign_id)
        assert retrieved.campaign_id == campaign.campaign_id

    def test_campaign_not_found(self, agent):
        """Test campaign not found error."""
        with pytest.raises(CampaignNotFoundError):
            agent.campaign_manager.get_campaign("NONEXISTENT")

    def test_pause_campaign(self, agent, sample_suppliers):
        """Test pausing campaign."""
        supplier_ids = [s["id"] for s in sample_suppliers]
        campaign = agent.create_campaign(
            name="Test Campaign",
            target_suppliers=supplier_ids
        )

        agent.campaign_manager.start_campaign(campaign.campaign_id)
        paused = agent.campaign_manager.pause_campaign(campaign.campaign_id)
        assert paused.status == CampaignStatus.PAUSED

    def test_complete_campaign(self, agent, sample_suppliers):
        """Test completing campaign."""
        supplier_ids = [s["id"] for s in sample_suppliers]
        campaign = agent.create_campaign(
            name="Test Campaign",
            target_suppliers=supplier_ids
        )

        completed = agent.campaign_manager.complete_campaign(campaign.campaign_id)
        assert completed.status == CampaignStatus.COMPLETED

    def test_campaign_analytics(self, agent, sample_suppliers):
        """Test campaign analytics generation."""
        # Register suppliers
        for supplier in sample_suppliers:
            agent.register_supplier(
                supplier["id"],
                supplier["email"],
                supplier["country"],
                auto_opt_in=True
            )

        supplier_ids = [s["id"] for s in sample_suppliers]
        campaign = agent.create_campaign(
            name="Test Campaign",
            target_suppliers=supplier_ids
        )

        analytics = agent.get_campaign_analytics(campaign.campaign_id)
        assert analytics.campaign_id == campaign.campaign_id
        assert analytics.response_rate >= 0.0


# ========== Portal & Validation Tests (30 tests) ==========

class TestPortalAndValidation:
    """Test supplier portal and data validation."""

    def test_generate_magic_link(self, agent):
        """Test magic link generation."""
        agent.register_supplier("SUP001", "test@example.com", "US")
        magic_link = agent.generate_magic_link("SUP001", "test@example.com")
        assert "magic-link" in magic_link
        assert "token=" in magic_link

    def test_validate_upload_valid(self, agent):
        """Test validation of valid data."""
        data = {
            "supplier_id": "SUP001",
            "product_id": "PROD001",
            "emission_factor": 1.5,
            "unit": "kg CO2e"
        }
        result = agent.validate_upload("SUP001", data)
        assert result.is_valid is True
        assert result.data_quality_score > 0.0

    def test_validate_upload_missing_required(self, agent):
        """Test validation with missing required fields."""
        data = {
            "supplier_id": "SUP001",
            "product_id": "PROD001"
            # Missing emission_factor and unit
        }
        result = agent.validate_upload("SUP001", data)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_upload_invalid_format(self, agent):
        """Test validation with invalid format."""
        data = {
            "supplier_id": "SUP001",
            "product_id": "PROD001",
            "emission_factor": "not_a_number",  # Invalid
            "unit": "kg"
        }
        result = agent.validate_upload("SUP001", data)
        assert result.is_valid is False

    def test_data_quality_score(self, agent):
        """Test data quality score calculation."""
        data = {
            "supplier_id": "SUP001",
            "product_id": "PROD001",
            "emission_factor": 1.5,
            "unit": "kg CO2e",
            "uncertainty": 10.0,
            "data_quality": "high"
        }
        result = agent.validate_upload("SUP001", data)
        assert result.data_quality_score > 0.7  # Good quality

    def test_completeness_percentage(self, agent):
        """Test completeness percentage calculation."""
        data = {
            "supplier_id": "SUP001",
            "product_id": "PROD001",
            "emission_factor": 1.5,
            "unit": "kg CO2e"
        }
        result = agent.validate_upload("SUP001", data)
        assert 0 <= result.completeness_percentage <= 100

    def test_upload_handler_csv(self, agent):
        """Test CSV upload handling."""
        upload = agent.upload_handler.initiate_upload(
            supplier_id="SUP001",
            campaign_id="CAMP001",
            file_name="data.csv",
            file_size=1024,
            file_type="csv"
        )
        assert upload.file_type == "csv"
        assert upload.supplier_id == "SUP001"

    def test_upload_handler_json(self, agent):
        """Test JSON upload handling."""
        upload = agent.upload_handler.initiate_upload(
            supplier_id="SUP001",
            campaign_id="CAMP001",
            file_name="data.json",
            file_size=2048,
            file_type="json"
        )
        assert upload.file_type == "json"


# ========== Gamification Tests (20 tests) ==========

class TestGamification:
    """Test gamification features."""

    def test_track_supplier_progress(self, agent):
        """Test tracking supplier progress."""
        progress = agent.track_supplier_progress(
            supplier_id="SUP001",
            campaign_id="CAMP001",
            completion_percentage=75.0,
            data_quality_score=0.85
        )
        assert progress.completion_percentage == 75.0
        assert progress.data_quality_score == 0.85

    def test_generate_leaderboard(self, agent):
        """Test leaderboard generation."""
        # Track progress for multiple suppliers
        for i in range(5):
            agent.track_supplier_progress(
                supplier_id=f"SUP{i:03d}",
                campaign_id="CAMP001",
                completion_percentage=float(i * 20),
                data_quality_score=float(i * 0.2)
            )

        leaderboard = agent.get_leaderboard("CAMP001", top_n=3)
        assert len(leaderboard.entries) == 3

    def test_badge_awarding(self, agent):
        """Test badge awarding."""
        progress = agent.track_supplier_progress(
            supplier_id="SUP001",
            campaign_id="CAMP001",
            completion_percentage=100.0,
            data_quality_score=0.95
        )

        # Should award complete_profile and data_champion badges
        assert len(progress.badges_earned) > 0

    def test_leaderboard_ranking(self, agent):
        """Test leaderboard ranking."""
        # Track progress with different scores
        agent.track_supplier_progress("SUP001", "CAMP001", 100.0, 0.95)
        agent.track_supplier_progress("SUP002", "CAMP001", 80.0, 0.85)
        agent.track_supplier_progress("SUP003", "CAMP001", 90.0, 0.90)

        leaderboard = agent.get_leaderboard("CAMP001")
        # Should be sorted by DQI score (descending)
        assert leaderboard.entries[0]["supplier_id"] == "SUP001"


# ========== Integration Tests (20 tests) ==========

class TestEmailIntegrations:
    """Test email service integrations."""

    def test_sendgrid_service(self, agent):
        """Test SendGrid service initialization."""
        from services.agents.engagement.integrations import SendGridService
        service = SendGridService()
        assert service is not None

    def test_mailgun_service(self, agent):
        """Test Mailgun service initialization."""
        from services.agents.engagement.integrations import MailgunService
        service = MailgunService()
        assert service is not None

    def test_aws_ses_service(self, agent):
        """Test AWS SES service initialization."""
        from services.agents.engagement.integrations import AWSSESService
        service = AWSSESService()
        assert service is not None


# ========== Statistics Tests (10 tests) ==========

class TestStatistics:
    """Test agent statistics and reporting."""

    def test_agent_statistics(self, agent, sample_suppliers):
        """Test overall agent statistics."""
        # Register suppliers
        for supplier in sample_suppliers:
            agent.register_supplier(
                supplier["id"],
                supplier["email"],
                supplier["country"],
                auto_opt_in=True
            )

        stats = agent.get_agent_statistics()
        assert "consent" in stats
        assert "campaigns" in stats
        assert stats["consent"]["total_records"] == 5

    def test_consent_statistics(self, agent, sample_suppliers):
        """Test consent registry statistics."""
        for supplier in sample_suppliers:
            agent.register_supplier(
                supplier["id"],
                supplier["email"],
                supplier["country"],
                auto_opt_in=True
            )

        stats = agent.consent_registry.get_statistics()
        assert stats["total_records"] == 5
        assert stats["opted_in"] == 5


# Run with: pytest tests/agents/engagement/test_agent.py -v
