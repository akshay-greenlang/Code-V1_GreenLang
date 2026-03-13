# -*- coding: utf-8 -*-
"""
Unit tests for CommunicationHub Engine - AGENT-EUDR-031

Tests communication sending, scheduling, campaign management,
delivery tracking, and response recording.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
)
from greenlang.agents.eudr.stakeholder_engagement.communication_hub import (
    CommunicationHub,
)
from greenlang.agents.eudr.stakeholder_engagement.models import (
    CommunicationChannel,
    CommunicationRecord,
    CommunicationTemplate,
    DeliveryStatus,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return StakeholderEngagementConfig()


@pytest.fixture
def hub(config):
    return CommunicationHub(config=config)


# ---------------------------------------------------------------------------
# Test: SendCommunication
# ---------------------------------------------------------------------------

class TestSendCommunication:
    """Test sending communications across all channels."""

    @pytest.mark.asyncio
    async def test_send_email_success(self, hub):
        """Test sending an email communication."""
        record = await hub.send_communication(
            operator_id="OP-001",
            stakeholder_ids=["STK-IND-001", "STK-COM-001"],
            channel=CommunicationChannel.EMAIL,
            subject="Meeting Invitation",
            body="You are invited to the quarterly meeting.",
        )
        assert record.communication_id.startswith("COMM-")
        assert record.channel == CommunicationChannel.EMAIL
        assert record.delivery_status in [DeliveryStatus.SENT, DeliveryStatus.PENDING]

    @pytest.mark.asyncio
    async def test_send_sms_success(self, hub):
        """Test sending an SMS communication."""
        record = await hub.send_communication(
            operator_id="OP-001",
            stakeholder_ids=["STK-COM-001"],
            channel=CommunicationChannel.SMS,
            subject="Reminder",
            body="Meeting tomorrow at 10AM.",
        )
        assert record.channel == CommunicationChannel.SMS

    @pytest.mark.asyncio
    async def test_send_letter_success(self, hub):
        """Test sending a letter communication."""
        record = await hub.send_communication(
            operator_id="OP-001",
            stakeholder_ids=["STK-IND-001"],
            channel=CommunicationChannel.LETTER,
            subject="Formal Notice",
            body="Official communication regarding FPIC process.",
        )
        assert record.channel == CommunicationChannel.LETTER

    @pytest.mark.asyncio
    async def test_send_all_channels(self, hub):
        """Test sending communication via all supported channels."""
        for channel in CommunicationChannel:
            record = await hub.send_communication(
                operator_id="OP-001",
                stakeholder_ids=["STK-001"],
                channel=channel,
                subject=f"Test via {channel.value}",
                body=f"Body for {channel.value}.",
            )
            assert record.channel == channel

    @pytest.mark.asyncio
    async def test_send_missing_subject_raises(self, hub):
        """Test sending without subject raises error."""
        with pytest.raises(ValueError, match="subject is required"):
            await hub.send_communication(
                operator_id="OP-001",
                stakeholder_ids=["STK-001"],
                channel=CommunicationChannel.EMAIL,
                subject="",
                body="Test body.",
            )

    @pytest.mark.asyncio
    async def test_send_missing_body_raises(self, hub):
        """Test sending without body raises error."""
        with pytest.raises(ValueError, match="body is required"):
            await hub.send_communication(
                operator_id="OP-001",
                stakeholder_ids=["STK-001"],
                channel=CommunicationChannel.EMAIL,
                subject="Test",
                body="",
            )

    @pytest.mark.asyncio
    async def test_send_no_stakeholders_raises(self, hub):
        """Test sending without stakeholder_ids raises error."""
        with pytest.raises(ValueError, match="stakeholder_ids are required"):
            await hub.send_communication(
                operator_id="OP-001",
                stakeholder_ids=[],
                channel=CommunicationChannel.EMAIL,
                subject="Test",
                body="Test.",
            )

    @pytest.mark.asyncio
    async def test_send_with_template(self, hub, communication_template):
        """Test sending with a template reference."""
        record = await hub.send_communication(
            operator_id="OP-001",
            stakeholder_ids=["STK-IND-001"],
            channel=CommunicationChannel.EMAIL,
            subject="Meeting Invitation - March 2026",
            body="Rendered template body.",
            template_id="TPL-INVITE-001",
        )
        assert record.template_id == "TPL-INVITE-001"

    @pytest.mark.asyncio
    async def test_send_with_language(self, hub):
        """Test sending with specific language."""
        record = await hub.send_communication(
            operator_id="OP-001",
            stakeholder_ids=["STK-IND-001"],
            channel=CommunicationChannel.EMAIL,
            subject="Invitacion",
            body="Cuerpo del mensaje.",
            language="es",
        )
        assert record.language == "es"

    @pytest.mark.asyncio
    async def test_send_unique_ids(self, hub):
        """Test each communication gets unique ID."""
        ids = set()
        for i in range(5):
            record = await hub.send_communication(
                operator_id="OP-001",
                stakeholder_ids=["STK-001"],
                channel=CommunicationChannel.EMAIL,
                subject=f"Test {i}",
                body=f"Body {i}",
            )
            ids.add(record.communication_id)
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# Test: ScheduleCommunication
# ---------------------------------------------------------------------------

class TestScheduleCommunication:
    """Test communication scheduling."""

    @pytest.mark.asyncio
    async def test_schedule_future_communication(self, hub):
        """Test scheduling a future communication."""
        scheduled = await hub.schedule_communication(
            operator_id="OP-001",
            stakeholder_ids=["STK-001"],
            channel=CommunicationChannel.EMAIL,
            subject="Scheduled Update",
            body="Monthly update content.",
            send_at=datetime.now(tz=timezone.utc) + timedelta(days=7),
        )
        assert scheduled.delivery_status == DeliveryStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_schedule_past_date_raises(self, hub):
        """Test scheduling for past date raises error."""
        with pytest.raises(ValueError, match="send_at must be in the future"):
            await hub.schedule_communication(
                operator_id="OP-001",
                stakeholder_ids=["STK-001"],
                channel=CommunicationChannel.EMAIL,
                subject="Test",
                body="Test",
                send_at=datetime.now(tz=timezone.utc) - timedelta(days=1),
            )

    @pytest.mark.asyncio
    async def test_schedule_returns_record(self, hub):
        """Test schedule returns CommunicationRecord."""
        scheduled = await hub.schedule_communication(
            operator_id="OP-001",
            stakeholder_ids=["STK-001"],
            channel=CommunicationChannel.SMS,
            subject="Reminder",
            body="Tomorrow meeting.",
            send_at=datetime.now(tz=timezone.utc) + timedelta(hours=12),
        )
        assert isinstance(scheduled, CommunicationRecord)

    @pytest.mark.asyncio
    async def test_schedule_missing_send_at_raises(self, hub):
        """Test scheduling without send_at raises error."""
        with pytest.raises(ValueError, match="send_at is required"):
            await hub.schedule_communication(
                operator_id="OP-001",
                stakeholder_ids=["STK-001"],
                channel=CommunicationChannel.EMAIL,
                subject="Test",
                body="Test",
                send_at=None,
            )

    @pytest.mark.asyncio
    async def test_schedule_multiple_recipients(self, hub):
        """Test scheduling for multiple recipients."""
        scheduled = await hub.schedule_communication(
            operator_id="OP-001",
            stakeholder_ids=["STK-001", "STK-002", "STK-003"],
            channel=CommunicationChannel.EMAIL,
            subject="Mass Update",
            body="Content for all stakeholders.",
            send_at=datetime.now(tz=timezone.utc) + timedelta(days=3),
        )
        assert len(scheduled.stakeholder_ids) == 3

    @pytest.mark.asyncio
    async def test_schedule_different_channels(self, hub):
        """Test scheduling via different channels."""
        for channel in [CommunicationChannel.EMAIL, CommunicationChannel.SMS, CommunicationChannel.LETTER]:
            scheduled = await hub.schedule_communication(
                operator_id="OP-001",
                stakeholder_ids=["STK-001"],
                channel=channel,
                subject="Test",
                body="Test",
                send_at=datetime.now(tz=timezone.utc) + timedelta(days=1),
            )
            assert scheduled.channel == channel


# ---------------------------------------------------------------------------
# Test: SendCampaign
# ---------------------------------------------------------------------------

class TestSendCampaign:
    """Test campaign management."""

    @pytest.mark.asyncio
    async def test_create_campaign_success(self, hub, sample_campaign):
        """Test creating a communication campaign."""
        result = await hub.send_campaign(
            campaign_id=sample_campaign["campaign_id"],
            name=sample_campaign["name"],
            operator_id="OP-001",
            stakeholder_ids=sample_campaign["target_stakeholders"],
            channels=[CommunicationChannel.EMAIL, CommunicationChannel.SMS],
            subject="Q1 Update",
            body="Quarterly stakeholder update.",
        )
        assert isinstance(result, dict)
        assert result.get("campaign_id") == "CAMP-001"

    @pytest.mark.asyncio
    async def test_campaign_multiple_channels(self, hub):
        """Test campaign across multiple channels."""
        result = await hub.send_campaign(
            campaign_id="CAMP-002",
            name="Multi-Channel",
            operator_id="OP-001",
            stakeholder_ids=["STK-001", "STK-002"],
            channels=[CommunicationChannel.EMAIL, CommunicationChannel.SMS, CommunicationChannel.PHONE],
            subject="Update",
            body="Multi-channel update.",
        )
        assert result.get("channels_used", 0) >= 2 or isinstance(result.get("communications"), list)

    @pytest.mark.asyncio
    async def test_campaign_empty_stakeholders_raises(self, hub):
        """Test campaign with no stakeholders raises error."""
        with pytest.raises(ValueError, match="stakeholder_ids are required"):
            await hub.send_campaign(
                campaign_id="CAMP-EMPTY",
                name="Empty",
                operator_id="OP-001",
                stakeholder_ids=[],
                channels=[CommunicationChannel.EMAIL],
                subject="Test",
                body="Test",
            )

    @pytest.mark.asyncio
    async def test_campaign_empty_channels_raises(self, hub):
        """Test campaign with no channels raises error."""
        with pytest.raises(ValueError, match="channels are required"):
            await hub.send_campaign(
                campaign_id="CAMP-NOCHAN",
                name="No Channels",
                operator_id="OP-001",
                stakeholder_ids=["STK-001"],
                channels=[],
                subject="Test",
                body="Test",
            )

    @pytest.mark.asyncio
    async def test_campaign_returns_summary(self, hub):
        """Test campaign returns summary with counts."""
        result = await hub.send_campaign(
            campaign_id="CAMP-003",
            name="Summary Test",
            operator_id="OP-001",
            stakeholder_ids=["STK-001", "STK-002"],
            channels=[CommunicationChannel.EMAIL],
            subject="Test",
            body="Test.",
        )
        assert "communications_sent" in result or "total" in result

    @pytest.mark.asyncio
    async def test_campaign_unique_per_stakeholder(self, hub):
        """Test campaign creates separate communications per stakeholder."""
        result = await hub.send_campaign(
            campaign_id="CAMP-004",
            name="Unique Test",
            operator_id="OP-001",
            stakeholder_ids=["STK-001", "STK-002", "STK-003"],
            channels=[CommunicationChannel.EMAIL],
            subject="Individual",
            body="Per-stakeholder message.",
        )
        total = result.get("communications_sent", result.get("total", 0))
        assert total >= 3

    @pytest.mark.asyncio
    async def test_campaign_missing_name_raises(self, hub):
        """Test campaign with empty name raises error."""
        with pytest.raises(ValueError, match="name is required"):
            await hub.send_campaign(
                campaign_id="CAMP-005",
                name="",
                operator_id="OP-001",
                stakeholder_ids=["STK-001"],
                channels=[CommunicationChannel.EMAIL],
                subject="Test",
                body="Test",
            )

    @pytest.mark.asyncio
    async def test_campaign_with_language(self, hub):
        """Test campaign with specific language."""
        result = await hub.send_campaign(
            campaign_id="CAMP-006",
            name="Spanish Campaign",
            operator_id="OP-001",
            stakeholder_ids=["STK-001"],
            channels=[CommunicationChannel.EMAIL],
            subject="Actualizacion",
            body="Contenido en espanol.",
            language="es",
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test: TrackDelivery
# ---------------------------------------------------------------------------

class TestTrackDelivery:
    """Test delivery tracking."""

    @pytest.mark.asyncio
    async def test_track_delivery_success(self, hub):
        """Test tracking delivery status."""
        record = await hub.send_communication(
            "OP-001", ["STK-001"], CommunicationChannel.EMAIL,
            "Test", "Body.",
        )
        status = await hub.track_delivery(record.communication_id)
        assert isinstance(status, dict)
        assert "delivery_status" in status

    @pytest.mark.asyncio
    async def test_track_delivery_nonexistent_raises(self, hub):
        """Test tracking nonexistent communication raises error."""
        with pytest.raises(ValueError, match="communication not found"):
            await hub.track_delivery("COMM-NONEXISTENT")

    @pytest.mark.asyncio
    async def test_track_delivery_shows_delivered(self, hub):
        """Test tracking shows delivered status for sent communication."""
        record = await hub.send_communication(
            "OP-001", ["STK-001"], CommunicationChannel.EMAIL,
            "Test", "Body.",
        )
        await hub.update_delivery_status(record.communication_id, DeliveryStatus.DELIVERED)
        status = await hub.track_delivery(record.communication_id)
        assert status["delivery_status"] == DeliveryStatus.DELIVERED.value

    @pytest.mark.asyncio
    async def test_track_delivery_shows_failed(self, hub):
        """Test tracking shows failed status."""
        record = await hub.send_communication(
            "OP-001", ["STK-001"], CommunicationChannel.SMS,
            "Test", "Body.",
        )
        await hub.update_delivery_status(record.communication_id, DeliveryStatus.FAILED)
        status = await hub.track_delivery(record.communication_id)
        assert status["delivery_status"] == DeliveryStatus.FAILED.value

    @pytest.mark.asyncio
    async def test_track_delivery_shows_bounced(self, hub):
        """Test tracking shows bounced status."""
        record = await hub.send_communication(
            "OP-001", ["STK-001"], CommunicationChannel.EMAIL,
            "Test", "Body.",
        )
        await hub.update_delivery_status(record.communication_id, DeliveryStatus.BOUNCED)
        status = await hub.track_delivery(record.communication_id)
        assert status["delivery_status"] == DeliveryStatus.BOUNCED.value

    @pytest.mark.asyncio
    async def test_track_delivery_includes_timestamps(self, hub):
        """Test tracking includes timestamp information."""
        record = await hub.send_communication(
            "OP-001", ["STK-001"], CommunicationChannel.EMAIL,
            "Test", "Body.",
        )
        status = await hub.track_delivery(record.communication_id)
        assert "sent_at" in status or "updated_at" in status

    @pytest.mark.asyncio
    async def test_track_delivery_empty_id_raises(self, hub):
        """Test tracking with empty ID raises error."""
        with pytest.raises(ValueError, match="communication_id is required"):
            await hub.track_delivery("")

    @pytest.mark.asyncio
    async def test_update_delivery_status_success(self, hub):
        """Test updating delivery status."""
        record = await hub.send_communication(
            "OP-001", ["STK-001"], CommunicationChannel.EMAIL,
            "Test", "Body.",
        )
        updated = await hub.update_delivery_status(record.communication_id, DeliveryStatus.DELIVERED)
        assert updated.delivery_status == DeliveryStatus.DELIVERED


# ---------------------------------------------------------------------------
# Test: RecordResponse
# ---------------------------------------------------------------------------

class TestRecordResponse:
    """Test response recording from stakeholders."""

    @pytest.mark.asyncio
    async def test_record_response_success(self, hub):
        """Test recording a stakeholder response."""
        record = await hub.send_communication(
            "OP-001", ["STK-001"], CommunicationChannel.EMAIL,
            "RSVP", "Please confirm attendance.",
        )
        response = await hub.record_response(
            communication_id=record.communication_id,
            stakeholder_id="STK-001",
            response_type="confirmed",
            response_text="Will attend.",
        )
        assert isinstance(response, dict)
        assert response["stakeholder_id"] == "STK-001"

    @pytest.mark.asyncio
    async def test_record_response_declined(self, hub):
        """Test recording a declined response."""
        record = await hub.send_communication(
            "OP-001", ["STK-001"], CommunicationChannel.EMAIL,
            "RSVP", "Please confirm.",
        )
        response = await hub.record_response(
            record.communication_id, "STK-001", "declined", "Cannot attend.",
        )
        assert response["response_type"] == "declined"

    @pytest.mark.asyncio
    async def test_record_response_nonexistent_raises(self, hub):
        """Test response for nonexistent communication raises error."""
        with pytest.raises(ValueError, match="communication not found"):
            await hub.record_response("COMM-NONEXISTENT", "STK-001", "confirmed", "OK")

    @pytest.mark.asyncio
    async def test_record_response_empty_stakeholder_raises(self, hub):
        """Test response with empty stakeholder_id raises error."""
        record = await hub.send_communication(
            "OP-001", ["STK-001"], CommunicationChannel.EMAIL,
            "Test", "Body.",
        )
        with pytest.raises(ValueError, match="stakeholder_id is required"):
            await hub.record_response(record.communication_id, "", "confirmed", "OK")

    @pytest.mark.asyncio
    async def test_record_response_timestamps(self, hub):
        """Test response includes timestamp."""
        record = await hub.send_communication(
            "OP-001", ["STK-001"], CommunicationChannel.EMAIL,
            "Test", "Body.",
        )
        response = await hub.record_response(
            record.communication_id, "STK-001", "acknowledged", "Noted.",
        )
        assert "responded_at" in response

    @pytest.mark.asyncio
    async def test_record_multiple_responses(self, hub):
        """Test recording responses from multiple stakeholders."""
        record = await hub.send_communication(
            "OP-001", ["STK-001", "STK-002", "STK-003"],
            CommunicationChannel.EMAIL, "Group RSVP", "Please respond.",
        )
        for stk_id in ["STK-001", "STK-002", "STK-003"]:
            response = await hub.record_response(
                record.communication_id, stk_id, "confirmed", "OK",
            )
            assert response["stakeholder_id"] == stk_id

    @pytest.mark.asyncio
    async def test_record_response_various_types(self, hub):
        """Test recording various response types."""
        record = await hub.send_communication(
            "OP-001", ["STK-001"], CommunicationChannel.EMAIL,
            "Test", "Body.",
        )
        for rtype in ["confirmed", "declined", "acknowledged", "deferred", "no_response"]:
            response = await hub.record_response(
                record.communication_id, "STK-001", rtype, f"Response: {rtype}",
            )
            assert response["response_type"] == rtype

    @pytest.mark.asyncio
    async def test_record_response_returns_dict(self, hub):
        """Test record_response returns a dictionary."""
        record = await hub.send_communication(
            "OP-001", ["STK-001"], CommunicationChannel.EMAIL,
            "Test", "Body.",
        )
        response = await hub.record_response(
            record.communication_id, "STK-001", "confirmed", "OK",
        )
        assert isinstance(response, dict)
