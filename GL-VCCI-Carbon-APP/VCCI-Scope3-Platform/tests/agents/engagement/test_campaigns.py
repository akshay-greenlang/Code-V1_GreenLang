# -*- coding: utf-8 -*-
"""
Tests for campaign management and email scheduling.
"""
import pytest
from datetime import datetime, timedelta

from services.agents.engagement.campaigns import (
from greenlang.determinism import DeterministicClock
    CampaignManager,
    EmailScheduler,
    CampaignAnalytics
)
from services.agents.engagement.consent import ConsentRegistry
from services.agents.engagement.models import (
    EmailSequence,
    Campaign,
    CampaignStatus
)


@pytest.fixture
def campaign_manager():
    return CampaignManager(storage_path="data/test_campaigns.json")


@pytest.fixture
def consent_registry():
    registry = ConsentRegistry(storage_path="data/test_consent_campaigns.json")
    # Register test suppliers
    for i in range(5):
        registry.register_supplier(
            f"SUP{i:03d}",
            f"supplier{i}@example.com",
            "US",
            auto_opt_in=True
        )
    return registry


@pytest.fixture
def email_scheduler(consent_registry):
    return EmailScheduler(consent_registry)


class TestCampaignCreation:
    """Test campaign creation and configuration."""

    def test_create_basic_campaign(self, campaign_manager):
        """Test basic campaign creation."""
        sequence = EmailSequence(
            sequence_id="test",
            name="Test Sequence",
            touches=[
                {"touch_number": 1, "day_offset": 0, "template": "touch_1_introduction"}
            ]
        )

        campaign = campaign_manager.create_campaign(
            name="Test Campaign",
            target_suppliers=["SUP001", "SUP002"],
            email_sequence=sequence
        )

        assert campaign.name == "Test Campaign"
        assert len(campaign.target_suppliers) == 2
        assert campaign.status == CampaignStatus.DRAFT

    def test_campaign_dates(self, campaign_manager):
        """Test campaign date configuration."""
        sequence = EmailSequence(
            sequence_id="test",
            name="Test",
            touches=[{"touch_number": 1, "day_offset": 0, "template": "touch_1_introduction"}]
        )

        start = DeterministicClock.utcnow()
        campaign = campaign_manager.create_campaign(
            name="Test",
            target_suppliers=["SUP001"],
            email_sequence=sequence,
            start_date=start,
            duration_days=60
        )

        assert campaign.start_date == start
        assert (campaign.end_date - start).days == 60


class TestEmailScheduling:
    """Test email scheduling."""

    def test_schedule_campaign_emails(self, email_scheduler, campaign_manager, consent_registry):
        """Test scheduling emails for campaign."""
        sequence = EmailSequence(
            sequence_id="test",
            name="Test",
            touches=[
                {"touch_number": 1, "day_offset": 0, "template": "touch_1_introduction"},
                {"touch_number": 2, "day_offset": 7, "template": "touch_2_reminder"}
            ]
        )

        campaign = campaign_manager.create_campaign(
            name="Test",
            target_suppliers=["SUP000", "SUP001"],
            email_sequence=sequence
        )

        messages = email_scheduler.schedule_campaign_emails(
            campaign=campaign,
            personalization_base={"company_name": "Test Co"}
        )

        # 2 suppliers * 2 touches = 4 messages
        assert len(messages) == 4

    def test_consent_filtering(self, email_scheduler, campaign_manager, consent_registry):
        """Test that opted-out suppliers are filtered."""
        # Opt out one supplier
        consent_registry.revoke_consent("SUP000")

        sequence = EmailSequence(
            sequence_id="test",
            name="Test",
            touches=[{"touch_number": 1, "day_offset": 0, "template": "touch_1_introduction"}]
        )

        campaign = campaign_manager.create_campaign(
            name="Test",
            target_suppliers=["SUP000", "SUP001"],  # One opted out
            email_sequence=sequence
        )

        messages = email_scheduler.schedule_campaign_emails(
            campaign=campaign,
            personalization_base={"company_name": "Test Co"}
        )

        # Only 1 supplier should get email
        assert len(messages) == 1


class TestCampaignAnalytics:
    """Test campaign analytics."""

    def test_generate_analytics(self, campaign_manager):
        """Test analytics generation."""
        sequence = EmailSequence(
            sequence_id="test",
            name="Test",
            touches=[{"touch_number": 1, "day_offset": 0, "template": "touch_1_introduction"}]
        )

        campaign = campaign_manager.create_campaign(
            name="Test",
            target_suppliers=["SUP001", "SUP002", "SUP003"],
            email_sequence=sequence
        )

        # Update metrics
        campaign_manager.update_metrics(
            campaign.campaign_id,
            emails_sent=3,
            emails_delivered=3,
            emails_opened=2,
            data_submissions=1
        )

        analytics_engine = CampaignAnalytics()
        analytics = analytics_engine.generate_analytics(
            campaign=campaign,
            messages=[]
        )

        assert analytics.response_rate == pytest.approx(1/3, 0.01)


# Run with: pytest tests/agents/engagement/test_campaigns.py -v
