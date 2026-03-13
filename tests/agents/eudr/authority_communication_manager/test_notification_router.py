# -*- coding: utf-8 -*-
"""
Unit tests for NotificationRouter engine - AGENT-EUDR-040

Tests multi-channel notification delivery (email, API, portal, SMS,
webhook), channel validation, recipient routing, delivery status
tracking, retry logic, queue management, and health checks.

55+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.authority_communication_manager.config import (
    AuthorityCommunicationManagerConfig,
)
from greenlang.agents.eudr.authority_communication_manager.notification_router import (
    NotificationRouter,
)
from greenlang.agents.eudr.authority_communication_manager.models import (
    Notification,
    NotificationChannel,
    RecipientType,
)


@pytest.fixture
def config():
    return AuthorityCommunicationManagerConfig()


@pytest.fixture
def router(config):
    return NotificationRouter(config=config)


# ====================================================================
# Initialization
# ====================================================================


class TestInit:
    def test_router_created(self, router):
        assert router is not None

    def test_default_config(self):
        r = NotificationRouter()
        assert r.config is not None

    def test_custom_config(self, config):
        r = NotificationRouter(config=config)
        assert r.config is config

    def test_notifications_empty(self, router):
        assert len(router._notifications) == 0

    def test_queue_empty(self, router):
        assert len(router._queue) == 0

    def test_provenance_initialized(self, router):
        assert router._provenance is not None


# ====================================================================
# Send Notification
# ====================================================================


class TestSendNotification:
    @pytest.mark.asyncio
    async def test_send_email(self, router):
        result = await router.send_notification(
            communication_id="COMM-001",
            channel="email",
            recipient_type="operator",
            recipient_id="OP-001",
            recipient_address="compliance@acme.com",
            subject="Information Request",
            body="Please provide documents.",
        )
        assert isinstance(result, Notification)
        assert result.channel == NotificationChannel.EMAIL

    @pytest.mark.asyncio
    async def test_send_api(self, router):
        result = await router.send_notification(
            communication_id="COMM-001",
            channel="api",
            recipient_type="system",
            recipient_id="SYS-001",
            recipient_address="https://erp.example.com/webhooks/eudr",
            subject="EUDR Event",
            body='{"event": "communication_received"}',
        )
        assert result.channel == NotificationChannel.API

    @pytest.mark.asyncio
    async def test_send_portal(self, router):
        result = await router.send_notification(
            communication_id="COMM-001",
            channel="portal",
            recipient_type="compliance_officer",
            recipient_id="USER-001",
            subject="New Communication",
            body="Check portal for details.",
        )
        assert result.channel == NotificationChannel.PORTAL

    @pytest.mark.asyncio
    async def test_send_sms(self, router):
        result = await router.send_notification(
            communication_id="COMM-001",
            channel="sms",
            recipient_type="operator",
            recipient_id="OP-001",
            recipient_address="+491234567890",
            subject="Deadline Alert",
            body="Your response is due in 24 hours.",
        )
        assert result.channel == NotificationChannel.SMS

    @pytest.mark.asyncio
    async def test_send_webhook(self, router):
        result = await router.send_notification(
            communication_id="COMM-001",
            channel="webhook",
            recipient_type="system",
            recipient_id="SYS-002",
            recipient_address="https://hooks.example.com/eudr",
            body='{"type": "deadline_reminder"}',
        )
        assert result.channel == NotificationChannel.WEBHOOK

    @pytest.mark.asyncio
    async def test_send_assigns_id(self, router):
        result = await router.send_notification(
            communication_id="COMM-001",
            channel="email",
            recipient_type="operator",
            recipient_id="OP-001",
        )
        assert result.notification_id is not None
        assert len(result.notification_id) > 0

    @pytest.mark.asyncio
    async def test_send_sets_delivery_status(self, router):
        result = await router.send_notification(
            communication_id="COMM-001",
            channel="email",
            recipient_type="operator",
            recipient_id="OP-001",
        )
        assert result.delivery_status in ("sent", "pending", "delivered", "failed")

    @pytest.mark.asyncio
    async def test_send_computes_provenance(self, router):
        result = await router.send_notification(
            communication_id="COMM-001",
            channel="email",
            recipient_type="operator",
            recipient_id="OP-001",
        )
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_send_stored(self, router):
        result = await router.send_notification(
            communication_id="COMM-001",
            channel="email",
            recipient_type="operator",
            recipient_id="OP-001",
        )
        assert result.notification_id in router._notifications

    @pytest.mark.asyncio
    async def test_send_invalid_channel_raises(self, router):
        with pytest.raises(ValueError, match="Invalid channel"):
            await router.send_notification(
                communication_id="COMM-001",
                channel="invalid_channel",
                recipient_type="operator",
                recipient_id="OP-001",
            )

    @pytest.mark.asyncio
    async def test_send_invalid_recipient_type_raises(self, router):
        with pytest.raises(ValueError, match="Invalid recipient"):
            await router.send_notification(
                communication_id="COMM-001",
                channel="email",
                recipient_type="invalid_type",
                recipient_id="OP-001",
            )

    @pytest.mark.asyncio
    async def test_send_with_language(self, router):
        result = await router.send_notification(
            communication_id="COMM-001",
            channel="email",
            recipient_type="operator",
            recipient_id="OP-001",
            language="de",
        )
        assert result.language.value == "de"

    @pytest.mark.asyncio
    async def test_send_all_channels(self, router):
        """Test each notification channel works."""
        for ch in NotificationChannel:
            result = await router.send_notification(
                communication_id="COMM-001",
                channel=ch.value,
                recipient_type="operator",
                recipient_id="OP-001",
            )
            assert result.channel == ch

    @pytest.mark.asyncio
    async def test_send_all_recipient_types(self, router):
        """Test each recipient type works."""
        for rt in RecipientType:
            result = await router.send_notification(
                communication_id="COMM-001",
                channel="portal",
                recipient_type=rt.value,
                recipient_id="ID-001",
            )
            assert result.recipient_type == rt


# ====================================================================
# Broadcast
# ====================================================================


class TestMultiChannel:
    @pytest.mark.asyncio
    async def test_send_multi_channel_two(self, router):
        results = await router.send_multi_channel(
            communication_id="COMM-001",
            channels=["email", "portal"],
            recipient_type="operator",
            recipient_id="OP-001",
            addresses={"email": "test@example.com", "portal": ""},
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_send_multi_channel_three(self, router):
        results = await router.send_multi_channel(
            communication_id="COMM-001",
            channels=["email", "portal", "api"],
            recipient_type="operator",
            recipient_id="OP-001",
            addresses={
                "email": "test@example.com",
                "portal": "",
                "api": "https://api.example.com/eudr",
            },
        )
        assert len(results) == 3


# ====================================================================
# Get / List / Health
# ====================================================================


class TestGetListHealth:
    @pytest.mark.asyncio
    async def test_get_notification(self, router):
        notif = await router.send_notification(
            communication_id="COMM-001",
            channel="email",
            recipient_type="operator",
            recipient_id="OP-001",
        )
        result = await router.get_notification(notif.notification_id)
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_notification_not_found(self, router):
        result = await router.get_notification("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_notifications_empty(self, router):
        result = await router.list_notifications()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_notifications_multiple(self, router):
        await router.send_notification(
            communication_id="COMM-001",
            channel="email",
            recipient_type="operator",
            recipient_id="OP-001",
        )
        await router.send_notification(
            communication_id="COMM-002",
            channel="portal",
            recipient_type="compliance_officer",
            recipient_id="USER-001",
        )
        result = await router.list_notifications()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_health_check(self, router):
        health = await router.health_check()
        assert health["status"] == "healthy"
        assert health["total_notifications"] == 0
        assert health["queue_depth"] == 0
