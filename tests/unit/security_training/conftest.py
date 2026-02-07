"""
Test fixtures for security_training module.

Provides mock training content, assessments, and configuration for testing.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


@pytest.fixture
def sample_training_module() -> Dict[str, Any]:
    """Create a sample training module."""
    return {
        "module_id": str(uuid4()),
        "title": "Phishing Awareness Training",
        "description": "Learn to identify and report phishing attempts",
        "category": "security_awareness",
        "duration_minutes": 30,
        "content_type": "interactive",
        "difficulty": "beginner",
        "topics": ["phishing", "social_engineering", "email_security"],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def sample_assessment() -> Dict[str, Any]:
    """Create a sample assessment."""
    return {
        "assessment_id": str(uuid4()),
        "module_id": str(uuid4()),
        "title": "Phishing Awareness Quiz",
        "questions": [
            {
                "question_id": str(uuid4()),
                "question": "Which of the following is a sign of a phishing email?",
                "options": [
                    "Generic greeting like 'Dear Customer'",
                    "Company logo in the email",
                    "Proper grammar",
                    "Known sender address",
                ],
                "correct_answer": 0,
            },
        ],
        "passing_score": 80,
        "time_limit_minutes": 15,
    }


@pytest.fixture
def sample_phishing_campaign() -> Dict[str, Any]:
    """Create a sample phishing simulation campaign."""
    return {
        "campaign_id": str(uuid4()),
        "name": "Q1 2025 Phishing Test",
        "template": "password_reset",
        "target_groups": ["all_employees"],
        "start_date": datetime.utcnow().isoformat(),
        "end_date": (datetime.utcnow() + timedelta(days=7)).isoformat(),
        "status": "active",
        "statistics": {
            "emails_sent": 500,
            "emails_opened": 250,
            "links_clicked": 50,
            "credentials_submitted": 10,
            "reported": 75,
        },
    }


@pytest.fixture
def sample_user_progress() -> Dict[str, Any]:
    """Create a sample user training progress."""
    return {
        "user_id": str(uuid4()),
        "email": "employee@example.com",
        "completed_modules": ["module_1", "module_2"],
        "in_progress_modules": ["module_3"],
        "assessment_scores": {
            "module_1": 95,
            "module_2": 88,
        },
        "overall_score": 91.5,
        "last_activity": datetime.utcnow().isoformat(),
        "compliance_status": "compliant",
    }


@pytest.fixture
def training_config() -> Dict[str, Any]:
    """Create training configuration."""
    return {
        "mandatory_modules": ["phishing_awareness", "data_handling"],
        "completion_deadline_days": 30,
        "reminder_frequency_days": 7,
        "passing_score": 80,
        "max_attempts": 3,
        "phishing_simulation_enabled": True,
    }


@pytest.fixture
def training_admin_headers() -> Dict[str, str]:
    """Create headers for training admin role."""
    return {
        "Authorization": "Bearer test-training-admin-token",
        "X-User-Id": "training-admin",
        "X-User-Roles": "training-admin,security-analyst",
    }
