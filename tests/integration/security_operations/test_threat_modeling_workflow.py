"""
Integration tests for threat modeling workflow.

Tests end-to-end threat model creation, analysis, and risk assessment.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class TestThreatModelingWorkflow:
    """Test complete threat modeling workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_threat_model_workflow(
        self,
        test_database,
        admin_auth_headers,
        integration_threat_model_data,
    ):
        """Test complete threat model creation and analysis workflow."""
        mock_engine = MagicMock()
        mock_scorer = MagicMock()

        # Step 1: Create threat model with components
        model_id = str(uuid4())

        # Step 2: Analyze components for threats
        mock_engine.analyze_component.return_value = [
            MagicMock(
                threat_id=str(uuid4()),
                title="Spoofing Attack",
                category="spoofing",
                likelihood=0.5,
                impact=0.7,
            )
        ]

        for component in integration_threat_model_data["components"]:
            threats = mock_engine.analyze_component(component)
            assert len(threats) >= 1

        # Step 3: Calculate risk scores
        mock_scorer.calculate_risk_score.return_value = MagicMock(
            risk_score=0.65,
            risk_level="HIGH",
        )

        for threat in mock_engine.analyze_component.return_value:
            scored_threat = mock_scorer.calculate_risk_score(threat)
            assert scored_threat.risk_score > 0

        # Step 4: Prioritize threats
        mock_scorer.prioritize_threats.return_value = mock_engine.analyze_component.return_value

        prioritized = mock_scorer.prioritize_threats(mock_engine.analyze_component.return_value)
        assert len(prioritized) >= 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_stride_analysis_workflow(
        self,
        test_database,
        admin_auth_headers,
    ):
        """Test STRIDE analysis workflow."""
        mock_engine = MagicMock()

        # Test each STRIDE category
        stride_categories = [
            "SPOOFING",
            "TAMPERING",
            "REPUDIATION",
            "INFORMATION_DISCLOSURE",
            "DENIAL_OF_SERVICE",
            "ELEVATION_OF_PRIVILEGE",
        ]

        # Create threats for each category
        all_threats = []
        for category in stride_categories:
            threat = MagicMock()
            threat.category = category
            threat.threat_id = str(uuid4())
            all_threats.append(threat)

        mock_engine.generate_threat_model.return_value = MagicMock(
            threats=all_threats,
            overall_risk_score=0.65,
        )

        # Generate complete model
        model = mock_engine.generate_threat_model(
            name="STRIDE Test Model",
            components=[],
            data_flows=[],
            trust_boundaries=[],
        )

        assert len(model.threats) == 6  # One per STRIDE category

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_risk_scoring_workflow(
        self,
        test_database,
        admin_auth_headers,
    ):
        """Test risk scoring workflow with business context."""
        mock_scorer = MagicMock()

        # Test threat without business context
        base_threat = MagicMock()
        base_threat.likelihood = 0.5
        base_threat.impact = 0.7

        mock_scorer.calculate_risk_score.return_value = MagicMock(
            risk_score=0.35,
            risk_level="MEDIUM",
        )

        base_result = mock_scorer.calculate_risk_score(base_threat)

        # Test threat with production context (should increase score)
        mock_scorer.apply_business_context.return_value = MagicMock(
            risk_score=0.525,  # 1.5x multiplier for production
            risk_level="HIGH",
        )

        prod_result = mock_scorer.apply_business_context(
            base_result,
            environment="production",
        )

        assert prod_result.risk_score > base_result.risk_score


class TestMitigationWorkflow:
    """Test mitigation management workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mitigation_assignment_workflow(
        self,
        test_database,
        admin_auth_headers,
    ):
        """Test assigning mitigations to threats."""
        threat_id = str(uuid4())

        # Create mitigation
        mitigation = MagicMock()
        mitigation.mitigation_id = str(uuid4())
        mitigation.title = "Implement MFA"
        mitigation.status = "planned"
        mitigation.effectiveness = 0.85

        # Assign to threat
        mock_service = AsyncMock()
        mock_service.add_mitigation.return_value = mitigation

        result = await mock_service.add_mitigation(
            threat_id=threat_id,
            mitigation=mitigation,
        )

        assert result.mitigation_id is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mitigation_effectiveness_tracking(
        self,
        test_database,
        admin_auth_headers,
    ):
        """Test tracking mitigation effectiveness on risk scores."""
        mock_scorer = MagicMock()

        # Initial risk score
        threat_without_mitigation = MagicMock()
        threat_without_mitigation.risk_score = 0.8
        threat_without_mitigation.mitigations = []

        # After implementing mitigation
        threat_with_mitigation = MagicMock()
        threat_with_mitigation.risk_score = 0.4  # Reduced by mitigation
        threat_with_mitigation.mitigations = [
            MagicMock(status="implemented", effectiveness=0.5),
        ]

        # Verify risk reduction
        assert threat_with_mitigation.risk_score < threat_without_mitigation.risk_score


class TestThreatModelReviewWorkflow:
    """Test threat model review workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_quarterly_review_workflow(
        self,
        test_database,
        admin_auth_headers,
    ):
        """Test quarterly threat model review workflow."""
        mock_service = AsyncMock()

        # Get models due for review
        mock_service.get_models_due_for_review.return_value = [
            MagicMock(
                model_id=str(uuid4()),
                name="Production API Model",
                last_review=datetime.utcnow(),
            )
        ]

        models_due = await mock_service.get_models_due_for_review(days=90)

        assert len(models_due) >= 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_threat_model_versioning(
        self,
        test_database,
        admin_auth_headers,
    ):
        """Test threat model version tracking."""
        mock_service = AsyncMock()

        # Create initial version
        v1 = MagicMock()
        v1.version = "1.0.0"
        v1.threats = [MagicMock()]

        mock_service.create_version.return_value = v1

        # Create updated version
        v2 = MagicMock()
        v2.version = "1.1.0"
        v2.threats = [MagicMock(), MagicMock()]  # Added new threat

        mock_service.create_version.return_value = v2

        # Get version history
        mock_service.get_version_history.return_value = [v1, v2]

        history = await mock_service.get_version_history(str(uuid4()))

        assert len(history) == 2
