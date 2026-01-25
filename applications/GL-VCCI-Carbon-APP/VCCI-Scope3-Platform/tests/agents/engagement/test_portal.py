# -*- coding: utf-8 -*-
"""
Tests for supplier portal features.
"""
import pytest
from datetime import datetime, timedelta

from services.agents.engagement.portal import (
    PortalAuthenticator,
    UploadHandler,
    LiveValidator,
    GamificationEngine
)
from services.agents.engagement.models import BadgeType


class TestPortalAuthentication:
    """Test portal authentication."""

    def test_magic_link_generation(self):
        """Test magic link generation."""
        auth = PortalAuthenticator()
        link = auth.generate_magic_link("SUP001", "test@example.com")

        assert "magic-link" in link
        assert "token=" in link

    def test_magic_link_validation(self):
        """Test magic link validation."""
        auth = PortalAuthenticator()
        link = auth.generate_magic_link("SUP001", "test@example.com")

        # Extract token
        token = link.split("token=")[1]

        # Validate
        data = auth.validate_magic_link(token)
        assert data is not None
        assert data["supplier_id"] == "SUP001"

    def test_session_creation(self):
        """Test session creation."""
        auth = PortalAuthenticator()
        link = auth.generate_magic_link("SUP001", "test@example.com")
        token = link.split("token=")[1]

        session = auth.authenticate_with_magic_link(token)
        assert session.supplier_id == "SUP001"
        assert session.session_id is not None


class TestUploadHandler:
    """Test upload handling."""

    def test_initiate_upload(self):
        """Test upload initiation."""
        handler = UploadHandler()
        upload = handler.initiate_upload(
            supplier_id="SUP001",
            campaign_id="CAMP001",
            file_name="data.csv",
            file_size=1024,
            file_type="csv"
        )

        assert upload.supplier_id == "SUP001"
        assert upload.file_type == "csv"

    def test_process_json(self):
        """Test JSON processing."""
        handler = UploadHandler()
        upload = handler.initiate_upload(
            "SUP001", "CAMP001", "data.json", 512, "json"
        )

        json_data = '''
        [
            {"supplier_id": "SUP001", "product_id": "P001", "emission_factor": 1.5, "unit": "kg"},
            {"supplier_id": "SUP001", "product_id": "P002", "emission_factor": 2.0, "unit": "kg"}
        ]
        '''

        result = handler.process_json(upload.upload_id, json_data)
        assert result["count"] == 2


class TestLiveValidator:
    """Test live validation."""

    def test_validate_valid_record(self):
        """Test validation of valid record."""
        validator = LiveValidator()
        record = {
            "supplier_id": "SUP001",
            "product_id": "P001",
            "emission_factor": 1.5,
            "unit": "kg CO2e"
        }

        result = validator.validate_record(record)
        assert result.is_valid is True

    def test_validate_missing_required(self):
        """Test validation with missing fields."""
        validator = LiveValidator()
        record = {
            "supplier_id": "SUP001"
            # Missing other required fields
        }

        result = validator.validate_record(record)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_data_quality_score(self):
        """Test DQI calculation."""
        validator = LiveValidator()
        record = {
            "supplier_id": "SUP001",
            "product_id": "P001",
            "emission_factor": 1.5,
            "unit": "kg CO2e",
            "uncertainty": 10.0,
            "data_quality": "high",
            "source": "primary"
        }

        result = validator.validate_record(record)
        assert result.data_quality_score > 0.8  # High quality


class TestGamification:
    """Test gamification features."""

    def test_track_progress(self):
        """Test progress tracking."""
        engine = GamificationEngine()
        progress = engine.track_progress(
            "SUP001", "CAMP001", 75.0, 0.85
        )

        assert progress.completion_percentage == 75.0
        assert progress.data_quality_score == 0.85

    def test_badge_awarding(self):
        """Test badge awarding."""
        engine = GamificationEngine()

        # Track 100% completion
        progress = engine.track_progress("SUP001", "CAMP001", 100.0, 0.95)

        # Check and award badges
        awarded = engine.check_and_award_badges("SUP001", "CAMP001")

        assert len(awarded) > 0

    def test_leaderboard_generation(self):
        """Test leaderboard generation."""
        engine = GamificationEngine()

        # Track progress for 3 suppliers
        engine.track_progress("SUP001", "CAMP001", 100.0, 0.95)
        engine.track_progress("SUP002", "CAMP001", 80.0, 0.85)
        engine.track_progress("SUP003", "CAMP001", 90.0, 0.90)

        leaderboard = engine.generate_leaderboard("CAMP001", top_n=3)

        assert len(leaderboard.entries) == 3
        assert leaderboard.entries[0]["supplier_id"] == "SUP001"  # Highest DQI


# Run with: pytest tests/agents/engagement/test_portal.py -v
