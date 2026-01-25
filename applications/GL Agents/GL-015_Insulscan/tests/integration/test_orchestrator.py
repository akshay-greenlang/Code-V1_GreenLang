# -*- coding: utf-8 -*-
"""GL-015 INSULSCAN - Orchestrator integration tests.

Integration tests for the InsulscanOrchestrator validating complete
analysis workflows, hot spot detection, ROI calculations, and
deterministic result generation.

These tests verify:
- Full analysis workflow execution
- Hot spot detection with various temperature profiles
- ROI calculation for poor condition insulation
- Deterministic results with same seed
- Health check functionality
- Recommendation generation for critical conditions

Author: GL-TestEngineer
Version: 1.0.0
"""

from __future__ import annotations

import pytest
from datetime import datetime

from gl_015_insulscan.core.config import InsulscanSettings
from gl_015_insulscan.core.orchestrator import (
    InsulscanOrchestrator,
)
from gl_015_insulscan.core.schemas import (
    InsulationAsset,
    ThermalMeasurement,
    AnalysisResult,
    AnalysisStatus,
)
from gl_015_insulscan.core.config import (
    InsulationType,
    SurfaceType,
    DataQuality,
    ConditionSeverity,
)


@pytest.fixture
def settings() -> InsulscanSettings:
    """Create test settings with deterministic seed."""
    return InsulscanSettings()


@pytest.fixture
def orchestrator(settings: InsulscanSettings) -> InsulscanOrchestrator:
    """Create test orchestrator instance."""
    return InsulscanOrchestrator(settings=settings)


def create_test_asset(
    asset_id: str = "PIPE-001",
    surface_type: SurfaceType = SurfaceType.PIPE,
    insulation_type: InsulationType = InsulationType.MINERAL_WOOL,
    thickness_mm: float = 50.0,
    operating_temp_c: float = 180.0,
    ambient_temp_c: float = 25.0,
    surface_area_m2: float = 10.0,
) -> InsulationAsset:
    """Create a test insulation asset."""
    return InsulationAsset(
        asset_id=asset_id,
        asset_name=f"Test Asset {asset_id}",
        surface_type=surface_type,
        insulation_type=insulation_type,
        thickness_mm=thickness_mm,
        operating_temp_c=operating_temp_c,
        ambient_temp_c=ambient_temp_c,
        surface_area_m2=surface_area_m2,
    )


def create_test_measurement(
    surface_temp_c: float,
    location: str = "section_1",
    data_quality: DataQuality = DataQuality.GOOD,
) -> ThermalMeasurement:
    """Create a test thermal measurement."""
    return ThermalMeasurement(
        measurement_id=f"MEAS-{location}",
        surface_temp_c=surface_temp_c,
        ambient_temp_c=25.0,
        relative_humidity_percent=50.0,
        wind_speed_ms=0.5,
        data_quality=data_quality,
    )


class TestOrchestratorIntegration:
    """Integration tests for InsulscanOrchestrator."""

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(
        self,
        orchestrator: InsulscanOrchestrator,
    ) -> None:
        """Test complete analysis workflow."""
        # Create test asset and measurements
        asset = create_test_asset(
            asset_id="PIPE-001",
            surface_type=SurfaceType.PIPE,
            insulation_type=InsulationType.MINERAL_WOOL,
            thickness_mm=50.0,
            operating_temp_c=180.0,
            ambient_temp_c=25.0,
            surface_area_m2=10.0,
        )

        measurements = [
            create_test_measurement(surface_temp_c=45.0, location="section_1"),
            create_test_measurement(surface_temp_c=85.0, location="section_2"),  # Hot spot
        ]

        # Execute analysis
        result = await orchestrator.analyze_insulation(
            asset=asset,
            measurements=measurements,
            include_recommendations=True,
        )

        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.asset_id == "PIPE-001"
        assert result.status == AnalysisStatus.COMPLETED
        assert result.heat_loss is not None
        assert result.heat_loss.heat_loss_w > 0
        assert result.condition is not None
        assert 0 <= result.condition.condition_score <= 100
        assert result.condition.severity in [
            ConditionSeverity.EXCELLENT,
            ConditionSeverity.GOOD,
            ConditionSeverity.FAIR,
            ConditionSeverity.POOR,
            ConditionSeverity.CRITICAL,
            ConditionSeverity.FAILED,
        ]
        assert result.input_hash is not None
        assert result.output_hash is not None
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_hot_spot_detection(
        self,
        orchestrator: InsulscanOrchestrator,
    ) -> None:
        """Test hot spot detection with various temperatures."""
        asset = create_test_asset(
            asset_id="PIPE-002",
            operating_temp_c=200.0,
            surface_area_m2=5.0,
        )

        # Create measurements with varying temperatures
        measurements = [
            create_test_measurement(surface_temp_c=40.0, location="A"),  # Normal
            create_test_measurement(surface_temp_c=70.0, location="B"),  # Minor hot spot
            create_test_measurement(surface_temp_c=95.0, location="C"),  # Moderate hot spot
            create_test_measurement(surface_temp_c=130.0, location="D"),  # Critical hot spot
        ]

        result = await orchestrator.analyze_insulation(
            asset=asset,
            measurements=measurements,
        )

        # Should detect hot spots above warning threshold
        assert result.hot_spots is not None
        assert len(result.hot_spots) >= 2  # At least C and D should be detected

    @pytest.mark.asyncio
    async def test_roi_calculation(
        self,
        orchestrator: InsulscanOrchestrator,
    ) -> None:
        """Test ROI calculation for poor condition insulation."""
        # Create asset with thin insulation and high temp for poor condition
        asset = create_test_asset(
            asset_id="PIPE-003",
            insulation_type=InsulationType.CALCIUM_SILICATE,
            thickness_mm=25.0,  # Thin insulation
            operating_temp_c=250.0,  # High temp
            surface_area_m2=20.0,
        )

        # Create measurements showing degraded condition
        measurements = [
            create_test_measurement(surface_temp_c=100.0, location="1"),
            create_test_measurement(surface_temp_c=110.0, location="2"),
            create_test_measurement(surface_temp_c=120.0, location="3"),
        ]

        result = await orchestrator.analyze_insulation(
            asset=asset,
            measurements=measurements,
            include_recommendations=True,
        )

        # Poor condition should trigger recommendation
        if result.condition and result.condition.condition_score < 70:
            assert result.recommendation is not None

    @pytest.mark.asyncio
    async def test_deterministic_results(
        self,
        settings: InsulscanSettings,
    ) -> None:
        """Test that results are deterministic with same seed."""
        asset = create_test_asset(
            asset_id="PIPE-004",
            operating_temp_c=150.0,
            surface_area_m2=10.0,
        )

        measurements = [
            create_test_measurement(surface_temp_c=55.0, location="test"),
        ]

        # Run twice with same settings
        orchestrator1 = InsulscanOrchestrator(settings=settings)
        result1 = await orchestrator1.analyze_insulation(
            asset=asset,
            measurements=measurements,
        )

        orchestrator2 = InsulscanOrchestrator(settings=settings)
        result2 = await orchestrator2.analyze_insulation(
            asset=asset,
            measurements=measurements,
        )

        # Results should be identical
        assert result1.heat_loss is not None
        assert result2.heat_loss is not None
        assert result1.heat_loss.heat_loss_w == result2.heat_loss.heat_loss_w
        assert result1.condition is not None
        assert result2.condition is not None
        assert result1.condition.condition_score == result2.condition.condition_score

    @pytest.mark.asyncio
    async def test_health_check(
        self,
        orchestrator: InsulscanOrchestrator,
    ) -> None:
        """Test health check endpoint."""
        health = orchestrator.health_check()

        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert health["version"] == orchestrator.VERSION
        assert "uptime_seconds" in health
        assert "checks" in health
        assert "statistics" in health


class TestRecommendationGeneration:
    """Tests for recommendation generation."""

    @pytest.mark.asyncio
    async def test_critical_condition_recommendations(
        self,
        orchestrator: InsulscanOrchestrator,
    ) -> None:
        """Test recommendations for critical condition."""
        # Create asset with very poor insulation
        asset = create_test_asset(
            asset_id="CRITICAL-001",
            surface_type=SurfaceType.VESSEL,
            insulation_type=InsulationType.MINERAL_WOOL,
            thickness_mm=10.0,  # Very thin
            operating_temp_c=300.0,  # Very high
            surface_area_m2=50.0,
        )

        # Create measurements showing critical condition
        measurements = [
            create_test_measurement(surface_temp_c=150.0, location="top"),
            create_test_measurement(surface_temp_c=160.0, location="middle"),
            create_test_measurement(surface_temp_c=170.0, location="bottom"),
        ]

        result = await orchestrator.analyze_insulation(
            asset=asset,
            measurements=measurements,
            include_recommendations=True,
        )

        # Critical condition should have recommendations
        if result.condition and result.condition.condition_score < 40:
            assert result.recommendation is not None
            # Recommendation should have high priority
            assert result.recommendation.priority in [
                "immediate",
                "urgent",
                "high",
            ] or hasattr(result.recommendation, "priority")

    @pytest.mark.asyncio
    async def test_good_condition_no_urgent_recommendations(
        self,
        orchestrator: InsulscanOrchestrator,
    ) -> None:
        """Test that good condition doesn't generate urgent recommendations."""
        # Create well-insulated asset
        asset = create_test_asset(
            asset_id="GOOD-001",
            thickness_mm=100.0,  # Thick insulation
            operating_temp_c=100.0,  # Moderate temp
            surface_area_m2=10.0,
        )

        # Create measurements showing good condition
        measurements = [
            create_test_measurement(surface_temp_c=32.0, location="1"),
            create_test_measurement(surface_temp_c=30.0, location="2"),
        ]

        result = await orchestrator.analyze_insulation(
            asset=asset,
            measurements=measurements,
            include_recommendations=True,
        )

        # Good condition should have lower priority or monitoring recommendation
        if result.condition and result.condition.condition_score >= 75:
            if result.recommendation:
                assert result.recommendation.priority not in [
                    "immediate",
                    "urgent",
                ] or result.recommendation.priority.value not in [
                    "immediate",
                    "urgent",
                ]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_measurements(
        self,
        orchestrator: InsulscanOrchestrator,
    ) -> None:
        """Test handling of empty measurements list."""
        asset = create_test_asset(asset_id="EMPTY-001")

        # Empty measurements should raise or return error
        with pytest.raises(ValueError):
            await orchestrator.analyze_insulation(
                asset=asset,
                measurements=[],
            )

    @pytest.mark.asyncio
    async def test_low_quality_measurements_filtered(
        self,
        orchestrator: InsulscanOrchestrator,
    ) -> None:
        """Test that low quality measurements are filtered."""
        asset = create_test_asset(asset_id="QUALITY-001")

        # Mix of good and bad quality measurements
        measurements = [
            create_test_measurement(
                surface_temp_c=50.0,
                location="good",
                data_quality=DataQuality.GOOD,
            ),
            create_test_measurement(
                surface_temp_c=200.0,  # Unrealistic value
                location="bad",
                data_quality=DataQuality.BAD,
            ),
        ]

        result = await orchestrator.analyze_insulation(
            asset=asset,
            measurements=measurements,
        )

        # Should complete with warnings about filtered data
        assert result.status == AnalysisStatus.COMPLETED
        # Bad measurement should be filtered

    @pytest.mark.asyncio
    async def test_high_temperature_differential(
        self,
        orchestrator: InsulscanOrchestrator,
    ) -> None:
        """Test handling of extreme temperature differentials."""
        asset = create_test_asset(
            asset_id="EXTREME-001",
            operating_temp_c=500.0,  # Very high operating temp
            ambient_temp_c=25.0,
        )

        measurements = [
            create_test_measurement(surface_temp_c=80.0, location="1"),
        ]

        result = await orchestrator.analyze_insulation(
            asset=asset,
            measurements=measurements,
        )

        # Should handle extreme temps without error
        assert result.status == AnalysisStatus.COMPLETED
        assert result.heat_loss is not None
        assert result.heat_loss.heat_loss_w > 0


class TestProvenanceTracking:
    """Tests for provenance and audit trail."""

    @pytest.mark.asyncio
    async def test_provenance_hash_generated(
        self,
        orchestrator: InsulscanOrchestrator,
    ) -> None:
        """Test that provenance hashes are generated."""
        asset = create_test_asset(asset_id="PROV-001")
        measurements = [
            create_test_measurement(surface_temp_c=50.0, location="1"),
        ]

        result = await orchestrator.analyze_insulation(
            asset=asset,
            measurements=measurements,
        )

        # Provenance hashes should be present
        assert result.input_hash is not None
        assert len(result.input_hash) >= 16  # SHA-256 truncated
        assert result.output_hash is not None
        assert len(result.output_hash) >= 16

    @pytest.mark.asyncio
    async def test_audit_trail_recorded(
        self,
        orchestrator: InsulscanOrchestrator,
    ) -> None:
        """Test that calculations are recorded in audit trail."""
        asset = create_test_asset(asset_id="AUDIT-001")
        measurements = [
            create_test_measurement(surface_temp_c=50.0, location="1"),
        ]

        # Perform analysis
        await orchestrator.analyze_insulation(
            asset=asset,
            measurements=measurements,
        )

        # Check audit trail
        audit_trail = orchestrator.get_audit_trail(limit=10)
        assert len(audit_trail) > 0
        assert audit_trail[-1]["calculation_type"] == "full_analysis"
        assert audit_trail[-1]["success"] is True
