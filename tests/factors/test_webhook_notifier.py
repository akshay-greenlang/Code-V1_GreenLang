# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.notifications (Phase 6 completion)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from greenlang.factors.notifications.config import (
    NotificationChannel,
    NotificationConfig,
    NotificationEventType,
    NotificationRoute,
)
from greenlang.factors.notifications.webhook_notifier import (
    WebhookNotifier,
    build_watch_notify_callback,
    send_email_notification,
    send_slack_notification,
)


class TestNotificationConfig:
    def test_from_env_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            cfg = NotificationConfig.from_env()
        assert cfg.enabled is True
        assert cfg.slack_configured is False
        assert cfg.email_configured is False
        assert cfg.routes == []

    def test_from_env_slack_only(self):
        env = {"GL_FACTORS_SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T/B/X"}
        with patch.dict("os.environ", env, clear=True):
            cfg = NotificationConfig.from_env()
        assert cfg.slack_configured is True
        assert cfg.email_configured is False
        assert len(cfg.routes) == len(NotificationEventType)
        assert all(NotificationChannel.SLACK in r.channels for r in cfg.routes)

    def test_from_env_email_only(self):
        env = {
            "GL_FACTORS_SMTP_HOST": "smtp.example.com",
            "GL_FACTORS_NOTIFY_EMAIL_FROM": "noreply@greenlang.io",
            "GL_FACTORS_NOTIFY_EMAIL_TO": "alice@co.com,bob@co.com",
        }
        with patch.dict("os.environ", env, clear=True):
            cfg = NotificationConfig.from_env()
        assert cfg.email_configured is True
        assert cfg.email_to == ["alice@co.com", "bob@co.com"]

    def test_from_env_disabled(self):
        env = {"GL_FACTORS_NOTIFICATIONS_ENABLED": "false"}
        with patch.dict("os.environ", env, clear=True):
            cfg = NotificationConfig.from_env()
        assert cfg.enabled is False

    def test_routes_for_event(self):
        cfg = NotificationConfig(
            enabled=True,
            routes=[
                NotificationRoute(
                    event_type=NotificationEventType.WATCH_CHANGE,
                    channels=[NotificationChannel.SLACK],
                ),
                NotificationRoute(
                    event_type=NotificationEventType.RELEASE_READY,
                    channels=[NotificationChannel.EMAIL],
                    email_recipients=["lead@co.com"],
                ),
            ],
        )
        watch_routes = cfg.routes_for_event(NotificationEventType.WATCH_CHANGE)
        assert len(watch_routes) == 1
        release_routes = cfg.routes_for_event(NotificationEventType.RELEASE_READY)
        assert len(release_routes) == 1
        error_routes = cfg.routes_for_event(NotificationEventType.WATCH_ERROR)
        assert len(error_routes) == 0


class TestSendSlackNotification:
    @patch("greenlang.factors.notifications.webhook_notifier.urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = send_slack_notification("https://hooks.slack.com/test", "Hello")
        assert result is True

    @patch("greenlang.factors.notifications.webhook_notifier.urllib.request.urlopen")
    def test_failure(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("Connection refused")
        result = send_slack_notification("https://hooks.slack.com/test", "Hello")
        assert result is False


class TestSendEmailNotification:
    @patch("greenlang.factors.notifications.webhook_notifier.smtplib.SMTP")
    def test_success(self, mock_smtp_class):
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

        result = send_email_notification(
            "smtp.example.com", 587, "user", "pass",
            "from@co.com", ["to@co.com"], "Subject", "Body",
        )
        assert result is True

    def test_empty_recipients(self):
        result = send_email_notification(
            "smtp.example.com", 587, None, None,
            "from@co.com", [], "Subject", "Body",
        )
        assert result is False


class TestWebhookNotifier:
    def test_disabled(self):
        cfg = NotificationConfig(enabled=False)
        notifier = WebhookNotifier(cfg)
        sent = notifier.notify(NotificationEventType.WATCH_CHANGE, "test", "body")
        assert sent == 0

    def test_no_routes(self):
        cfg = NotificationConfig(enabled=True, routes=[])
        notifier = WebhookNotifier(cfg)
        sent = notifier.notify(NotificationEventType.WATCH_CHANGE, "test", "body")
        assert sent == 0

    @patch("greenlang.factors.notifications.webhook_notifier.send_slack_notification", return_value=True)
    def test_slack_dispatch(self, mock_send):
        cfg = NotificationConfig(
            enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            routes=[
                NotificationRoute(
                    event_type=NotificationEventType.WATCH_CHANGE,
                    channels=[NotificationChannel.SLACK],
                ),
            ],
        )
        notifier = WebhookNotifier(cfg)
        sent = notifier.notify_watch_change("epa_hub", "content_changed", "hash changed")
        assert sent == 1
        assert len(notifier.delivery_log) == 1
        assert notifier.delivery_log[0]["success"] is True
        assert notifier.delivery_log[0]["event_type"] == "watch_change"

    @patch("greenlang.factors.notifications.webhook_notifier.send_slack_notification", return_value=True)
    def test_notify_watch_error(self, mock_send):
        cfg = NotificationConfig(
            enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            routes=[
                NotificationRoute(
                    event_type=NotificationEventType.WATCH_ERROR,
                    channels=[NotificationChannel.SLACK],
                ),
            ],
        )
        notifier = WebhookNotifier(cfg)
        sent = notifier.notify_watch_error("broken_src", "Connection timeout")
        assert sent == 1

    @patch("greenlang.factors.notifications.webhook_notifier.send_slack_notification", return_value=True)
    def test_notify_release_ready(self, mock_send):
        cfg = NotificationConfig(
            enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            routes=[
                NotificationRoute(
                    event_type=NotificationEventType.RELEASE_READY,
                    channels=[NotificationChannel.SLACK],
                ),
            ],
        )
        notifier = WebhookNotifier(cfg)
        sent = notifier.notify_release_ready("2026.04.0", 50000)
        assert sent == 1

    @patch("greenlang.factors.notifications.webhook_notifier.send_slack_notification", return_value=True)
    def test_notify_release_published(self, mock_send):
        cfg = NotificationConfig(
            enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            routes=[
                NotificationRoute(
                    event_type=NotificationEventType.RELEASE_PUBLISHED,
                    channels=[NotificationChannel.SLACK],
                ),
            ],
        )
        notifier = WebhookNotifier(cfg)
        sent = notifier.notify_release_published("2026.04.0", "alice@greenlang.io")
        assert sent == 1

    @patch("greenlang.factors.notifications.webhook_notifier.send_slack_notification", return_value=True)
    def test_notify_policy_change(self, mock_send):
        cfg = NotificationConfig(
            enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            routes=[
                NotificationRoute(
                    event_type=NotificationEventType.POLICY_CHANGE,
                    channels=[NotificationChannel.SLACK],
                ),
            ],
        )
        notifier = WebhookNotifier(cfg)
        sent = notifier.notify_policy_change("defra", "Scope boundary updated")
        assert sent == 1

    def test_delivery_log_tracks_failures(self):
        cfg = NotificationConfig(
            enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            routes=[
                NotificationRoute(
                    event_type=NotificationEventType.WATCH_CHANGE,
                    channels=[NotificationChannel.SLACK],
                ),
            ],
        )
        notifier = WebhookNotifier(cfg)
        with patch(
            "greenlang.factors.notifications.webhook_notifier.send_slack_notification",
            return_value=False,
        ):
            sent = notifier.notify_watch_change("src", "content_changed")
        assert sent == 0
        assert len(notifier.delivery_log) == 1
        assert notifier.delivery_log[0]["success"] is False


class TestBuildWatchNotifyCallback:
    @patch("greenlang.factors.notifications.webhook_notifier.send_slack_notification", return_value=True)
    def test_callback_on_change(self, mock_send):
        cfg = NotificationConfig(
            enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            routes=[
                NotificationRoute(
                    event_type=NotificationEventType.WATCH_CHANGE,
                    channels=[NotificationChannel.SLACK],
                ),
            ],
        )
        callback = build_watch_notify_callback(cfg)
        mock_result = MagicMock()
        mock_result.source_id = "epa_hub"
        mock_result.change_detected = True
        mock_result.change_type = "content_changed"
        mock_result.error_message = None
        callback("Change detected", mock_result)
        mock_send.assert_called_once()

    @patch("greenlang.factors.notifications.webhook_notifier.send_slack_notification", return_value=True)
    def test_callback_on_error(self, mock_send):
        cfg = NotificationConfig(
            enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            routes=[
                NotificationRoute(
                    event_type=NotificationEventType.WATCH_ERROR,
                    channels=[NotificationChannel.SLACK],
                ),
            ],
        )
        callback = build_watch_notify_callback(cfg)
        mock_result = MagicMock()
        mock_result.source_id = "broken"
        mock_result.change_detected = False
        mock_result.error_message = "Connection timeout"
        callback("Error detected", mock_result)
        mock_send.assert_called_once()
