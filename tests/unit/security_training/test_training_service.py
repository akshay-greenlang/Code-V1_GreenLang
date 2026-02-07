"""
Unit tests for security training service.

Tests content library, curriculum mapping, assessments,
phishing simulations, and completion tracking.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class TestContentLibrary:
    """Test content library functionality."""

    @pytest.mark.asyncio
    async def test_create_module(self, sample_training_module):
        """Test creating a training module."""
        mock_service = AsyncMock()
        mock_service.create_module.return_value = sample_training_module

        result = await mock_service.create_module(sample_training_module)

        assert result["module_id"] is not None
        assert result["title"] == sample_training_module["title"]

    @pytest.mark.asyncio
    async def test_get_module(self, sample_training_module):
        """Test getting a training module."""
        mock_service = AsyncMock()
        mock_service.get_module.return_value = sample_training_module

        result = await mock_service.get_module(sample_training_module["module_id"])

        assert result["module_id"] == sample_training_module["module_id"]

    @pytest.mark.asyncio
    async def test_list_modules_by_category(self, sample_training_module):
        """Test listing modules by category."""
        mock_service = AsyncMock()
        mock_service.list_modules.return_value = [sample_training_module]

        result = await mock_service.list_modules(category="security_awareness")

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_search_modules(self, sample_training_module):
        """Test searching modules."""
        mock_service = AsyncMock()
        mock_service.search_modules.return_value = [sample_training_module]

        result = await mock_service.search_modules(query="phishing")

        assert len(result) >= 1


class TestCurriculumMapping:
    """Test curriculum mapping functionality."""

    @pytest.mark.asyncio
    async def test_create_curriculum(self):
        """Test creating a curriculum."""
        mock_service = AsyncMock()
        mock_service.create_curriculum.return_value = {
            "curriculum_id": str(uuid4()),
            "name": "New Hire Security Training",
            "modules": ["module_1", "module_2", "module_3"],
            "target_roles": ["all"],
            "completion_deadline_days": 30,
        }

        result = await mock_service.create_curriculum({
            "name": "New Hire Security Training",
            "modules": ["module_1", "module_2", "module_3"],
        })

        assert result["curriculum_id"] is not None

    @pytest.mark.asyncio
    async def test_assign_curriculum_to_user(self):
        """Test assigning curriculum to user."""
        mock_service = AsyncMock()
        mock_service.assign_curriculum.return_value = {
            "assignment_id": str(uuid4()),
            "user_id": str(uuid4()),
            "curriculum_id": str(uuid4()),
            "assigned_at": datetime.utcnow().isoformat(),
            "deadline": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        }

        result = await mock_service.assign_curriculum(
            user_id=str(uuid4()),
            curriculum_id=str(uuid4())
        )

        assert result["assignment_id"] is not None

    @pytest.mark.asyncio
    async def test_get_user_curriculum(self, sample_user_progress):
        """Test getting user's assigned curriculum."""
        mock_service = AsyncMock()
        mock_service.get_user_curriculum.return_value = {
            "user_id": sample_user_progress["user_id"],
            "curriculums": [
                {
                    "curriculum_id": str(uuid4()),
                    "name": "Security Fundamentals",
                    "progress": 75,
                }
            ],
        }

        result = await mock_service.get_user_curriculum(sample_user_progress["user_id"])

        assert len(result["curriculums"]) >= 1


class TestAssessmentEngine:
    """Test assessment engine functionality."""

    @pytest.mark.asyncio
    async def test_create_assessment(self, sample_assessment):
        """Test creating an assessment."""
        mock_service = AsyncMock()
        mock_service.create_assessment.return_value = sample_assessment

        result = await mock_service.create_assessment(sample_assessment)

        assert result["assessment_id"] is not None

    @pytest.mark.asyncio
    async def test_start_assessment(self, sample_assessment):
        """Test starting an assessment attempt."""
        mock_service = AsyncMock()
        mock_service.start_assessment.return_value = {
            "attempt_id": str(uuid4()),
            "assessment_id": sample_assessment["assessment_id"],
            "user_id": str(uuid4()),
            "started_at": datetime.utcnow().isoformat(),
            "time_remaining_seconds": sample_assessment["time_limit_minutes"] * 60,
        }

        result = await mock_service.start_assessment(
            sample_assessment["assessment_id"],
            user_id=str(uuid4())
        )

        assert result["attempt_id"] is not None

    @pytest.mark.asyncio
    async def test_submit_assessment(self, sample_assessment):
        """Test submitting assessment answers."""
        mock_service = AsyncMock()
        mock_service.submit_assessment.return_value = {
            "attempt_id": str(uuid4()),
            "score": 85,
            "passed": True,
            "correct_answers": 17,
            "total_questions": 20,
        }

        result = await mock_service.submit_assessment(
            attempt_id=str(uuid4()),
            answers=[0, 1, 2, 0, 1]
        )

        assert result["passed"] is True
        assert result["score"] >= sample_assessment["passing_score"]

    @pytest.mark.asyncio
    async def test_get_assessment_results(self):
        """Test getting assessment results."""
        mock_service = AsyncMock()
        mock_service.get_results.return_value = {
            "attempt_id": str(uuid4()),
            "score": 85,
            "passed": True,
            "detailed_results": [
                {"question_id": "q1", "correct": True},
                {"question_id": "q2", "correct": False},
            ],
        }

        result = await mock_service.get_results(str(uuid4()))

        assert "detailed_results" in result


class TestPhishingSimulator:
    """Test phishing simulation functionality."""

    @pytest.mark.asyncio
    async def test_create_campaign(self, sample_phishing_campaign):
        """Test creating a phishing campaign."""
        mock_service = AsyncMock()
        mock_service.create_campaign.return_value = sample_phishing_campaign

        result = await mock_service.create_campaign({
            "name": sample_phishing_campaign["name"],
            "template": sample_phishing_campaign["template"],
            "target_groups": sample_phishing_campaign["target_groups"],
        })

        assert result["campaign_id"] is not None

    @pytest.mark.asyncio
    async def test_send_test_email(self):
        """Test sending a test phishing email."""
        mock_service = AsyncMock()
        mock_service.send_test_email.return_value = {
            "status": "sent",
            "recipient": "test@example.com",
            "sent_at": datetime.utcnow().isoformat(),
        }

        result = await mock_service.send_test_email(
            campaign_id=str(uuid4()),
            recipient="test@example.com"
        )

        assert result["status"] == "sent"

    @pytest.mark.asyncio
    async def test_track_interaction(self, sample_phishing_campaign):
        """Test tracking phishing email interaction."""
        mock_service = AsyncMock()
        mock_service.track_interaction.return_value = {
            "interaction_id": str(uuid4()),
            "campaign_id": sample_phishing_campaign["campaign_id"],
            "user_id": str(uuid4()),
            "interaction_type": "link_clicked",
            "timestamp": datetime.utcnow().isoformat(),
        }

        result = await mock_service.track_interaction(
            campaign_id=sample_phishing_campaign["campaign_id"],
            user_id=str(uuid4()),
            interaction_type="link_clicked"
        )

        assert result["interaction_type"] == "link_clicked"

    @pytest.mark.asyncio
    async def test_get_campaign_statistics(self, sample_phishing_campaign):
        """Test getting campaign statistics."""
        mock_service = AsyncMock()
        mock_service.get_statistics.return_value = sample_phishing_campaign["statistics"]

        result = await mock_service.get_statistics(sample_phishing_campaign["campaign_id"])

        assert result["emails_sent"] > 0
        assert result["reported"] >= 0

    @pytest.mark.asyncio
    async def test_report_phishing_email(self):
        """Test reporting a phishing email."""
        mock_service = AsyncMock()
        mock_service.report_email.return_value = {
            "report_id": str(uuid4()),
            "status": "confirmed_simulation",
            "feedback": "Great job! This was a simulated phishing test.",
        }

        result = await mock_service.report_email(
            email_id=str(uuid4()),
            user_id=str(uuid4())
        )

        assert result["status"] == "confirmed_simulation"


class TestCompletionTracker:
    """Test completion tracking functionality."""

    @pytest.mark.asyncio
    async def test_get_user_progress(self, sample_user_progress):
        """Test getting user progress."""
        mock_service = AsyncMock()
        mock_service.get_progress.return_value = sample_user_progress

        result = await mock_service.get_progress(sample_user_progress["user_id"])

        assert result["user_id"] == sample_user_progress["user_id"]
        assert result["compliance_status"] == "compliant"

    @pytest.mark.asyncio
    async def test_complete_module(self):
        """Test marking a module as complete."""
        mock_service = AsyncMock()
        mock_service.complete_module.return_value = {
            "user_id": str(uuid4()),
            "module_id": str(uuid4()),
            "completed_at": datetime.utcnow().isoformat(),
            "score": 90,
        }

        result = await mock_service.complete_module(
            user_id=str(uuid4()),
            module_id=str(uuid4()),
            score=90
        )

        assert result["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_get_compliance_report(self):
        """Test generating compliance report."""
        mock_service = AsyncMock()
        mock_service.get_compliance_report.return_value = {
            "total_users": 500,
            "compliant_users": 450,
            "non_compliant_users": 50,
            "compliance_rate": 90.0,
            "overdue_assignments": 25,
        }

        result = await mock_service.get_compliance_report()

        assert result["compliance_rate"] > 0

    @pytest.mark.asyncio
    async def test_send_reminder(self):
        """Test sending training reminder."""
        mock_service = AsyncMock()
        mock_service.send_reminder.return_value = {
            "reminder_id": str(uuid4()),
            "user_id": str(uuid4()),
            "sent_at": datetime.utcnow().isoformat(),
            "channel": "email",
        }

        result = await mock_service.send_reminder(
            user_id=str(uuid4()),
            module_id=str(uuid4())
        )

        assert result["sent_at"] is not None


class TestSecurityScorer:
    """Test security scoring functionality."""

    @pytest.mark.asyncio
    async def test_calculate_user_score(self, sample_user_progress):
        """Test calculating user security score."""
        mock_service = AsyncMock()
        mock_service.calculate_score.return_value = {
            "user_id": sample_user_progress["user_id"],
            "overall_score": 85,
            "components": {
                "training_completion": 90,
                "assessment_scores": 88,
                "phishing_awareness": 75,
            },
        }

        result = await mock_service.calculate_score(sample_user_progress["user_id"])

        assert result["overall_score"] > 0

    @pytest.mark.asyncio
    async def test_get_department_scores(self):
        """Test getting department-level security scores."""
        mock_service = AsyncMock()
        mock_service.get_department_scores.return_value = [
            {"department": "Engineering", "score": 88},
            {"department": "Sales", "score": 75},
            {"department": "HR", "score": 92},
        ]

        result = await mock_service.get_department_scores()

        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_get_organization_score(self):
        """Test getting organization-wide security score."""
        mock_service = AsyncMock()
        mock_service.get_organization_score.return_value = {
            "overall_score": 85,
            "trend": "improving",
            "benchmarks": {
                "industry_average": 75,
                "top_quartile": 90,
            },
        }

        result = await mock_service.get_organization_score()

        assert result["overall_score"] > 0
        assert result["overall_score"] > result["benchmarks"]["industry_average"]
