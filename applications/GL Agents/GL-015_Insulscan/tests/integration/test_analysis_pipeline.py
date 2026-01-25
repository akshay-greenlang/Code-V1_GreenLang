# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Integration Tests for Analysis Pipeline

End-to-end integration tests for the insulation analysis workflow:
- Data ingestion from thermal cameras
- Heat loss calculation pipeline
- Condition assessment workflow
- ROI analysis and recommendations
- Provenance tracking through the pipeline
- Performance benchmarks

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import asyncio
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock, patch
import math


# =============================================================================
# PIPELINE ORCHESTRATOR SIMULATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for analysis pipeline."""
    enable_provenance: bool = True
    performance_logging: bool = True
    max_batch_size: int = 100
    timeout_seconds: float = 30.0
    energy_cost_usd_kWh: float = 0.10


@dataclass
class AnalysisResult:
    """Result from pipeline analysis."""
    asset_id: str
    timestamp: datetime
    # Heat loss analysis
    heat_loss_W_m: float
    heat_loss_total_W: float
    insulation_efficiency: float
    # Condition assessment
    condition_score: float
    condition_grade: str
    # ROI analysis
    annual_energy_loss_usd: float
    recommended_action: str
    payback_years: Optional[float]
    # Provenance
    provenance_hash: str
    computation_time_ms: float
    pipeline_stages_completed: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class InsulationAnalysisPipeline:
    """
    Orchestrates the insulation analysis workflow.

    Pipeline Stages:
    1. Data Ingestion - Collect thermal measurements
    2. Heat Loss Calculation - Calculate using ASTM C680
    3. Condition Assessment - Score insulation condition
    4. ROI Analysis - Calculate payback for repairs
    5. Recommendation Generation - Prioritize actions
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._provenance_chain: List[str] = []

    async def analyze_asset(
        self,
        asset: Any,
        measurement: Any,
    ) -> AnalysisResult:
        """
        Run full analysis pipeline on an asset.

        Args:
            asset: InsulationAsset object
            measurement: ThermalMeasurement object

        Returns:
            AnalysisResult with complete analysis
        """
        start_time = time.perf_counter()
        stages_completed = []
        provenance_data = []

        try:
            # Stage 1: Data Validation
            validated_data = await self._validate_inputs(asset, measurement)
            stages_completed.append("validation")
            provenance_data.append(f"validated:{asset.asset_id}")

            # Stage 2: Heat Loss Calculation
            heat_loss = await self._calculate_heat_loss(asset)
            stages_completed.append("heat_loss")
            provenance_data.append(f"heat_loss:{heat_loss['W_m']:.6f}")

            # Stage 3: Condition Assessment
            condition = await self._assess_condition(asset, measurement, heat_loss)
            stages_completed.append("condition_assessment")
            provenance_data.append(f"score:{condition['score']:.2f}")

            # Stage 4: ROI Analysis
            roi = await self._calculate_roi(asset, heat_loss)
            stages_completed.append("roi_analysis")
            provenance_data.append(f"payback:{roi['payback_years']:.2f}" if roi['payback_years'] else "payback:N/A")

            # Stage 5: Recommendation
            recommendation = await self._generate_recommendation(condition, roi)
            stages_completed.append("recommendation")

            # Calculate computation time
            computation_time_ms = (time.perf_counter() - start_time) * 1000

            # Generate provenance hash
            provenance_string = "|".join(provenance_data)
            provenance_hash = hashlib.sha256(provenance_string.encode()).hexdigest()

            if self.config.enable_provenance:
                self._provenance_chain.append(provenance_hash)

            return AnalysisResult(
                asset_id=asset.asset_id,
                timestamp=datetime.now(timezone.utc),
                heat_loss_W_m=heat_loss["W_m"],
                heat_loss_total_W=heat_loss["total_W"],
                insulation_efficiency=heat_loss["efficiency"],
                condition_score=condition["score"],
                condition_grade=condition["grade"],
                annual_energy_loss_usd=roi["annual_loss_usd"],
                recommended_action=recommendation["action"],
                payback_years=roi["payback_years"],
                provenance_hash=provenance_hash,
                computation_time_ms=computation_time_ms,
                pipeline_stages_completed=stages_completed,
                metadata={
                    "config": {
                        "energy_cost": self.config.energy_cost_usd_kWh,
                    },
                },
            )

        except Exception as e:
            computation_time_ms = (time.perf_counter() - start_time) * 1000
            raise PipelineError(
                f"Pipeline failed at stage {len(stages_completed)}: {str(e)}",
                stages_completed=stages_completed,
                computation_time_ms=computation_time_ms,
            )

    async def _validate_inputs(self, asset: Any, measurement: Any) -> Dict[str, Any]:
        """Validate input data quality."""
        errors = []

        # Validate asset
        if asset.pipe_outer_diameter_m <= 0:
            errors.append("Invalid pipe diameter")
        if asset.insulation_thickness_m <= 0:
            errors.append("Invalid insulation thickness")
        if asset.process_temperature_C == asset.ambient_temperature_C:
            errors.append("No temperature differential")

        # Validate measurement
        if hasattr(measurement, 'data_quality'):
            from conftest import DataQuality
            if measurement.data_quality == DataQuality.BAD:
                errors.append("Bad data quality")

        if errors:
            raise ValidationError(errors)

        return {"valid": True, "errors": []}

    async def _calculate_heat_loss(self, asset: Any) -> Dict[str, float]:
        """Calculate heat loss per ASTM C680."""
        # Simplified heat loss calculation
        r_inner = asset.pipe_outer_diameter_m / 2
        r_outer = r_inner + asset.insulation_thickness_m
        k = asset.material.thermal_conductivity_W_mK
        h = 10.0  # Approximate surface coefficient

        delta_T = asset.process_temperature_C - asset.ambient_temperature_C

        # Insulation resistance
        R_insulation = math.log(r_outer / r_inner) / (2 * math.pi * k)
        R_surface = 1 / (2 * math.pi * r_outer * h)
        R_total = R_insulation + R_surface

        q_insulated = delta_T / R_total

        # Bare pipe heat loss
        q_bare = 2 * math.pi * r_inner * h * delta_T

        efficiency = (1 - q_insulated / q_bare) * 100 if q_bare != 0 else 0

        return {
            "W_m": q_insulated,
            "total_W": q_insulated * asset.length_m,
            "bare_W_m": q_bare,
            "efficiency": efficiency,
            "R_total": R_total,
        }

    async def _assess_condition(
        self,
        asset: Any,
        measurement: Any,
        heat_loss: Dict[str, float],
    ) -> Dict[str, Any]:
        """Assess insulation condition."""
        # Calculate age factor
        if asset.installation_date:
            age_years = (datetime.now(timezone.utc) - asset.installation_date).days / 365.25
        else:
            age_years = 0

        expected_life = 25  # Default expected life

        # Thermal performance factor
        thermal_factor = min(1.0, heat_loss["efficiency"] / 90)  # 90% is target

        # Age factor
        age_factor = max(0, 1 - age_years / expected_life)

        # Damage factor
        damage_count = len(asset.damage_types) if hasattr(asset, 'damage_types') else 0
        damage_factor = max(0, 1 - damage_count * 0.2)

        # Calculate overall score
        score = (
            40 * thermal_factor +
            20 * age_factor +
            40 * damage_factor
        )

        # Determine grade
        if score >= 90:
            grade = "EXCELLENT"
        elif score >= 70:
            grade = "GOOD"
        elif score >= 50:
            grade = "FAIR"
        elif score >= 30:
            grade = "POOR"
        else:
            grade = "CRITICAL"

        return {
            "score": score,
            "grade": grade,
            "thermal_factor": thermal_factor,
            "age_factor": age_factor,
            "damage_factor": damage_factor,
        }

    async def _calculate_roi(
        self,
        asset: Any,
        heat_loss: Dict[str, float],
    ) -> Dict[str, Any]:
        """Calculate ROI for repairs or replacement."""
        operating_hours = 8760
        energy_cost = self.config.energy_cost_usd_kWh

        # Current annual energy loss
        annual_loss_kWh = heat_loss["total_W"] * operating_hours / 1000
        annual_loss_usd = annual_loss_kWh * energy_cost

        # Estimated savings after repair (assume 30% improvement)
        potential_savings = annual_loss_usd * 0.3

        # Estimated repair cost
        repair_cost = asset.length_m * 100  # $100 per meter estimate

        # Payback calculation
        if potential_savings > 0:
            payback_years = repair_cost / potential_savings
        else:
            payback_years = None

        return {
            "annual_loss_kWh": annual_loss_kWh,
            "annual_loss_usd": annual_loss_usd,
            "potential_savings": potential_savings,
            "repair_cost": repair_cost,
            "payback_years": payback_years,
        }

    async def _generate_recommendation(
        self,
        condition: Dict[str, Any],
        roi: Dict[str, Any],
    ) -> Dict[str, str]:
        """Generate maintenance recommendation."""
        score = condition["score"]
        payback = roi["payback_years"]

        if score >= 80:
            action = "MONITOR"
            priority = "LOW"
        elif score >= 60:
            if payback and payback < 2:
                action = "REPAIR"
                priority = "MEDIUM"
            else:
                action = "MONITOR"
                priority = "LOW"
        elif score >= 40:
            action = "REPAIR"
            priority = "HIGH"
        else:
            action = "REPLACE"
            priority = "CRITICAL"

        return {
            "action": action,
            "priority": priority,
        }

    async def analyze_batch(
        self,
        assets: List[Any],
        measurements: Dict[str, Any],
    ) -> List[AnalysisResult]:
        """Analyze multiple assets in batch."""
        results = []

        for asset in assets:
            measurement = measurements.get(asset.asset_id)
            if measurement:
                result = await self.analyze_asset(asset, measurement)
                results.append(result)

        return results

    def get_provenance_chain(self) -> List[str]:
        """Get the provenance chain."""
        return self._provenance_chain.copy()


class PipelineError(Exception):
    """Error during pipeline execution."""
    def __init__(
        self,
        message: str,
        stages_completed: List[str] = None,
        computation_time_ms: float = 0,
    ):
        self.stages_completed = stages_completed or []
        self.computation_time_ms = computation_time_ms
        super().__init__(message)


class ValidationError(Exception):
    """Validation error."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {', '.join(errors)}")


# =============================================================================
# TEST CLASS: END-TO-END PIPELINE
# =============================================================================

class TestEndToEndPipeline:
    """End-to-end integration tests for the analysis pipeline."""

    @pytest.fixture
    def pipeline(self) -> InsulationAnalysisPipeline:
        """Create pipeline instance."""
        return InsulationAnalysisPipeline(PipelineConfig(
            enable_provenance=True,
            energy_cost_usd_kWh=0.10,
        ))

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(
        self,
        pipeline,
        sample_pipe_asset,
        sample_thermal_measurement,
    ):
        """Test complete analysis pipeline execution."""
        result = await pipeline.analyze_asset(
            sample_pipe_asset,
            sample_thermal_measurement,
        )

        # Verify all stages completed
        assert len(result.pipeline_stages_completed) == 5
        assert "validation" in result.pipeline_stages_completed
        assert "heat_loss" in result.pipeline_stages_completed
        assert "condition_assessment" in result.pipeline_stages_completed
        assert "roi_analysis" in result.pipeline_stages_completed
        assert "recommendation" in result.pipeline_stages_completed

        # Verify outputs are populated
        assert result.heat_loss_W_m > 0
        assert 0 <= result.condition_score <= 100
        assert result.condition_grade in ["EXCELLENT", "GOOD", "FAIR", "POOR", "CRITICAL"]
        assert result.annual_energy_loss_usd >= 0
        assert result.recommended_action in ["MONITOR", "REPAIR", "REPLACE"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_with_damaged_asset(
        self,
        pipeline,
        damaged_pipe_asset,
        anomaly_thermal_measurement,
    ):
        """Test pipeline handles damaged assets correctly."""
        result = await pipeline.analyze_asset(
            damaged_pipe_asset,
            anomaly_thermal_measurement,
        )

        # Damaged asset should have lower score
        assert result.condition_score < 70
        # Should recommend action
        assert result.recommended_action in ["REPAIR", "REPLACE"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_with_excellent_asset(
        self,
        pipeline,
        cryogenic_asset,
        sample_thermal_measurement,
    ):
        """Test pipeline handles excellent condition assets."""
        result = await pipeline.analyze_asset(
            cryogenic_asset,
            sample_thermal_measurement,
        )

        # Excellent condition should recommend monitoring
        assert result.condition_grade in ["EXCELLENT", "GOOD"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_provenance_tracking(
        self,
        pipeline,
        sample_pipe_asset,
        sample_thermal_measurement,
    ):
        """Test provenance is tracked through pipeline."""
        result = await pipeline.analyze_asset(
            sample_pipe_asset,
            sample_thermal_measurement,
        )

        # Verify provenance hash
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

        # Verify provenance chain
        chain = pipeline.get_provenance_chain()
        assert len(chain) >= 1
        assert result.provenance_hash in chain

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_provenance_determinism(
        self,
        sample_pipe_asset,
        sample_thermal_measurement,
    ):
        """Test provenance hash is deterministic."""
        pipeline1 = InsulationAnalysisPipeline()
        pipeline2 = InsulationAnalysisPipeline()

        result1 = await pipeline1.analyze_asset(
            sample_pipe_asset,
            sample_thermal_measurement,
        )
        result2 = await pipeline2.analyze_asset(
            sample_pipe_asset,
            sample_thermal_measurement,
        )

        # Same inputs should give same provenance
        assert result1.provenance_hash == result2.provenance_hash


# =============================================================================
# TEST CLASS: PERFORMANCE BENCHMARKS
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmark tests for the pipeline."""

    @pytest.fixture
    def pipeline(self) -> InsulationAnalysisPipeline:
        """Create pipeline instance."""
        return InsulationAnalysisPipeline()

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_analysis_latency(
        self,
        pipeline,
        sample_pipe_asset,
        sample_thermal_measurement,
        performance_timer,
    ):
        """Test single analysis completes within latency target."""
        timer = performance_timer()

        with timer:
            result = await pipeline.analyze_asset(
                sample_pipe_asset,
                sample_thermal_measurement,
            )

        # Target: <100ms for single analysis
        timer.assert_under(100.0)
        assert result.computation_time_ms < 100.0

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_analysis_throughput(
        self,
        pipeline,
        sample_pipe_asset,
        sample_thermal_measurement,
    ):
        """Test batch analysis throughput."""
        # Create batch of 50 assets
        assets = []
        measurements = {}

        for i in range(50):
            # Clone asset with unique ID
            asset_copy = type(sample_pipe_asset)(
                asset_id=f"PIPE-{i:03d}",
                asset_name=sample_pipe_asset.asset_name,
                location=sample_pipe_asset.location,
                pipe_outer_diameter_m=sample_pipe_asset.pipe_outer_diameter_m,
                insulation_thickness_m=sample_pipe_asset.insulation_thickness_m,
                length_m=sample_pipe_asset.length_m,
                geometry=sample_pipe_asset.geometry,
                material=sample_pipe_asset.material,
                process_temperature_C=sample_pipe_asset.process_temperature_C,
                ambient_temperature_C=sample_pipe_asset.ambient_temperature_C,
                wind_speed_m_s=sample_pipe_asset.wind_speed_m_s,
                installation_date=sample_pipe_asset.installation_date,
                last_inspection_date=sample_pipe_asset.last_inspection_date,
                current_condition=sample_pipe_asset.current_condition,
            )
            assets.append(asset_copy)

            # Clone measurement
            meas_copy = type(sample_thermal_measurement)(
                asset_id=f"PIPE-{i:03d}",
                timestamp=sample_thermal_measurement.timestamp,
                surface_temp_C=sample_thermal_measurement.surface_temp_C,
                ambient_temp_C=sample_thermal_measurement.ambient_temp_C,
                process_temp_C=sample_thermal_measurement.process_temp_C,
                emissivity=sample_thermal_measurement.emissivity,
                reflected_temp_C=sample_thermal_measurement.reflected_temp_C,
                humidity_percent=sample_thermal_measurement.humidity_percent,
                data_quality=sample_thermal_measurement.data_quality,
            )
            measurements[f"PIPE-{i:03d}"] = meas_copy

        start_time = time.perf_counter()
        results = await pipeline.analyze_batch(assets, measurements)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Target: >10 analyses per second (50 in 5 seconds)
        assert len(results) == 50
        assert elapsed_ms < 5000, f"Batch took {elapsed_ms:.0f}ms, exceeds 5000ms target"

        throughput = len(results) / (elapsed_ms / 1000)
        assert throughput > 10, f"Throughput {throughput:.1f}/sec below 10/sec target"


# =============================================================================
# TEST CLASS: DATA INTEGRATION
# =============================================================================

class TestDataIntegration:
    """Tests for data integration with external systems."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_thermal_camera_integration(
        self,
        mock_thermal_camera,
        sample_pipe_asset,
    ):
        """Test integration with thermal camera."""
        # Connect to camera
        connected = await mock_thermal_camera.connect()
        assert connected

        # Capture image
        image_data = await mock_thermal_camera.capture_image(sample_pipe_asset.asset_id)

        assert image_data["asset_id"] == sample_pipe_asset.asset_id
        assert "min_temp_C" in image_data
        assert "max_temp_C" in image_data
        assert image_data["min_temp_C"] < image_data["max_temp_C"]

        # Disconnect
        await mock_thermal_camera.disconnect()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cmms_work_order_creation(
        self,
        mock_cmms_connector,
        damaged_pipe_asset,
    ):
        """Test work order creation in CMMS."""
        # Connect to CMMS
        await mock_cmms_connector.connect()

        # Create work order
        wo_request = {
            "asset_id": damaged_pipe_asset.asset_id,
            "description": "Insulation repair required",
            "priority": "high",
        }

        work_order = await mock_cmms_connector.create_work_order(wo_request)

        assert work_order["work_order_id"] is not None
        assert work_order["asset_id"] == damaged_pipe_asset.asset_id
        assert work_order["status"] == "pending"

        # Disconnect
        await mock_cmms_connector.disconnect()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_database_persistence(
        self,
        mock_database,
        sample_pipe_asset,
        sample_thermal_measurement,
    ):
        """Test database persistence operations."""
        # Connect
        await mock_database.connect()
        assert mock_database.connected

        # Save asset
        asset_id = await mock_database.save_asset(sample_pipe_asset)
        assert asset_id == sample_pipe_asset.asset_id

        # Retrieve asset
        retrieved = await mock_database.get_asset(sample_pipe_asset.asset_id)
        assert retrieved.asset_id == sample_pipe_asset.asset_id

        # Save measurement
        meas_id = await mock_database.save_measurement(sample_thermal_measurement)
        assert meas_id is not None

        # Retrieve measurements
        measurements = await mock_database.get_measurements(sample_pipe_asset.asset_id)
        assert len(measurements) >= 1

        # Disconnect
        await mock_database.disconnect()
        assert not mock_database.connected


# =============================================================================
# TEST CLASS: ERROR HANDLING
# =============================================================================

class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    @pytest.fixture
    def pipeline(self) -> InsulationAnalysisPipeline:
        """Create pipeline instance."""
        return InsulationAnalysisPipeline()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_handles_invalid_asset(self, pipeline, sample_thermal_measurement):
        """Test pipeline handles invalid asset data."""
        # Create invalid asset (zero diameter)
        from conftest import (
            InsulationAsset, InsulationMaterial, InsulationType,
            PipeGeometry, ConditionGrade
        )

        invalid_asset = InsulationAsset(
            asset_id="INVALID-001",
            asset_name="Invalid Asset",
            location="Test",
            pipe_outer_diameter_m=0.0,  # Invalid
            insulation_thickness_m=0.05,
            length_m=10.0,
            geometry=PipeGeometry.CYLINDRICAL,
            material=InsulationMaterial(
                material_type=InsulationType.MINERAL_WOOL,
                thermal_conductivity_W_mK=0.04,
                reference_temperature_C=24.0,
                temperature_coefficient=0.0002,
                density_kg_m3=100.0,
                max_service_temp_C=650.0,
                min_service_temp_C=-40.0,
                moisture_resistance=0.6,
                cost_per_m3_usd=150.0,
            ),
            process_temperature_C=100.0,
            ambient_temperature_C=25.0,
            wind_speed_m_s=2.0,
            installation_date=datetime.now(timezone.utc),
            last_inspection_date=None,
            current_condition=ConditionGrade.GOOD,
        )

        with pytest.raises(ValidationError) as exc_info:
            await pipeline.analyze_asset(invalid_asset, sample_thermal_measurement)

        assert "Invalid pipe diameter" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_handles_equal_temperatures(
        self,
        pipeline,
        sample_thermal_measurement,
        mineral_wool_material,
    ):
        """Test pipeline handles equal process and ambient temperatures."""
        from conftest import InsulationAsset, PipeGeometry, ConditionGrade

        no_delta_t_asset = InsulationAsset(
            asset_id="NO-DELTA-001",
            asset_name="No Delta T",
            location="Test",
            pipe_outer_diameter_m=0.1,
            insulation_thickness_m=0.05,
            length_m=10.0,
            geometry=PipeGeometry.CYLINDRICAL,
            material=mineral_wool_material,
            process_temperature_C=25.0,  # Same as ambient
            ambient_temperature_C=25.0,
            wind_speed_m_s=2.0,
            installation_date=datetime.now(timezone.utc),
            last_inspection_date=None,
            current_condition=ConditionGrade.GOOD,
        )

        with pytest.raises(ValidationError) as exc_info:
            await pipeline.analyze_asset(no_delta_t_asset, sample_thermal_measurement)

        assert "temperature differential" in str(exc_info.value).lower()


# =============================================================================
# TEST CLASS: REALISTIC DATA SCENARIOS
# =============================================================================

class TestRealisticScenarios:
    """Tests with realistic thermal data scenarios."""

    @pytest.fixture
    def pipeline(self) -> InsulationAnalysisPipeline:
        """Create pipeline instance."""
        return InsulationAnalysisPipeline()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_steam_header_analysis(
        self,
        pipeline,
        sample_pipe_asset,
        sample_thermal_measurement,
    ):
        """Test analysis of typical steam header."""
        result = await pipeline.analyze_asset(
            sample_pipe_asset,  # Steam header at 175C
            sample_thermal_measurement,
        )

        # Steam header should have moderate heat loss
        assert 50 < result.heat_loss_W_m < 200

        # Good insulation should show high efficiency
        assert result.insulation_efficiency > 85

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_high_temperature_analysis(
        self,
        pipeline,
        high_temp_asset,
        sample_thermal_measurement,
    ):
        """Test analysis of high temperature equipment."""
        result = await pipeline.analyze_asset(
            high_temp_asset,  # 450C process
            sample_thermal_measurement,
        )

        # High temp should have higher heat loss
        assert result.heat_loss_W_m > 100

        # With thermal degradation damage, score should be lower
        assert result.condition_score < 90

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cryogenic_analysis(
        self,
        pipeline,
        cryogenic_asset,
        sample_thermal_measurement,
    ):
        """Test analysis of cryogenic insulation."""
        result = await pipeline.analyze_asset(
            cryogenic_asset,  # -160C LNG
            sample_thermal_measurement,
        )

        # Cryogenic shows heat gain (negative delta T)
        # Excellent condition - new cellular glass
        assert result.condition_grade in ["EXCELLENT", "GOOD"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_measurement_time_series(
        self,
        pipeline,
        sample_pipe_asset,
        measurement_time_series,
    ):
        """Test analysis with time series measurements."""
        results = []

        for measurement in measurement_time_series:
            result = await pipeline.analyze_asset(sample_pipe_asset, measurement)
            results.append(result)

        # All analyses should complete
        assert len(results) == len(measurement_time_series)

        # Heat loss should be relatively consistent
        heat_losses = [r.heat_loss_W_m for r in results]
        mean_hl = sum(heat_losses) / len(heat_losses)
        max_deviation = max(abs(hl - mean_hl) for hl in heat_losses)

        # Deviation should be <20% of mean (temperature variation effect)
        assert max_deviation < mean_hl * 0.20


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestEndToEndPipeline",
    "TestPerformanceBenchmarks",
    "TestDataIntegration",
    "TestPipelineErrorHandling",
    "TestRealisticScenarios",
    "InsulationAnalysisPipeline",
    "AnalysisResult",
    "PipelineConfig",
    "PipelineError",
    "ValidationError",
]
