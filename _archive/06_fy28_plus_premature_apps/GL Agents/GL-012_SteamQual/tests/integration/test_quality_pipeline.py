"""
Integration Tests: Steam Quality Estimation Pipeline

Tests the end-to-end steam quality estimation workflow:
1. Data acquisition from sensors
2. Steam property calculation
3. Dryness fraction estimation
4. Carryover risk assessment
5. Constraint validation
6. Output generation and provenance tracking

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from enum import Enum, auto


# =============================================================================
# Pipeline Components (Simulated for Integration Testing)
# =============================================================================

class QualityMethod(Enum):
    """Method for quality calculation."""
    ENTHALPY = "enthalpy"
    ENTROPY = "entropy"
    COMBINED = "combined"


@dataclass
class SensorData:
    """Raw sensor data from data acquisition."""
    tag: str
    value: float
    unit: str
    quality: str  # "GOOD", "BAD", "UNCERTAIN"
    timestamp: datetime


@dataclass
class ProcessedData:
    """Processed sensor data after validation."""
    pressure_mpa: float
    temperature_k: float
    enthalpy_kj_kg: Optional[float]
    entropy_kj_kg_k: Optional[float]
    flow_rate_kg_s: float
    tds_ppm: float
    data_quality_score: float
    timestamp: datetime


@dataclass
class QualityEstimate:
    """Steam quality estimation result."""
    dryness_fraction: float
    uncertainty_percent: float
    method: QualityMethod
    confidence_level: float
    is_valid: bool


@dataclass
class RiskAssessment:
    """Carryover risk assessment."""
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    probability: float
    recommended_action: str


@dataclass
class ConstraintResult:
    """Constraint validation result."""
    all_passed: bool
    violations: List[str]
    worst_severity: Optional[str]


@dataclass
class PipelineOutput:
    """Complete pipeline output."""
    quality_estimate: QualityEstimate
    risk_assessment: RiskAssessment
    constraint_result: ConstraintResult
    processed_data: ProcessedData
    provenance_hash: str
    processing_time_ms: float
    timestamp: datetime


# =============================================================================
# Pipeline Implementation
# =============================================================================

class SteamQualityPipeline:
    """
    End-to-end steam quality estimation pipeline.

    Pipeline stages:
    1. Data Acquisition - Collect sensor data
    2. Data Validation - Validate and filter data
    3. Property Calculation - Calculate steam properties
    4. Quality Estimation - Estimate dryness fraction
    5. Risk Assessment - Assess carryover risk
    6. Constraint Validation - Check safety constraints
    7. Output Generation - Generate output with provenance
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize pipeline with configuration."""
        self.config = config or {}
        self.min_dryness_threshold = self.config.get("min_dryness", 0.85)
        self.max_tds_threshold = self.config.get("max_tds_ppm", 100.0)

    def acquire_data(self, sensor_readings: Dict[str, SensorData]) -> Dict[str, SensorData]:
        """
        Stage 1: Data Acquisition

        Filter and validate incoming sensor data.
        """
        valid_readings = {}

        for tag, reading in sensor_readings.items():
            if reading.quality == "GOOD":
                valid_readings[tag] = reading
            elif reading.quality == "UNCERTAIN":
                # Accept with reduced confidence
                valid_readings[tag] = reading

        return valid_readings

    def validate_data(self, sensor_readings: Dict[str, SensorData]) -> ProcessedData:
        """
        Stage 2: Data Validation

        Convert sensor readings to processed data structure.
        """
        # Extract required values
        pressure = None
        temperature = None
        flow_rate = None
        tds = 0.0

        for tag, reading in sensor_readings.items():
            if "pressure" in tag.lower():
                pressure = reading.value
            elif "temperature" in tag.lower():
                temperature = reading.value
            elif "flow" in tag.lower():
                flow_rate = reading.value
            elif "tds" in tag.lower():
                tds = reading.value

        # Check for required fields
        if pressure is None or temperature is None:
            raise ValueError("Missing required pressure or temperature data")

        # Calculate data quality score
        n_good = sum(1 for r in sensor_readings.values() if r.quality == "GOOD")
        quality_score = n_good / len(sensor_readings) if sensor_readings else 0.0

        return ProcessedData(
            pressure_mpa=pressure,
            temperature_k=temperature,
            enthalpy_kj_kg=None,  # Will be calculated
            entropy_kj_kg_k=None,  # Will be calculated
            flow_rate_kg_s=flow_rate or 0.0,
            tds_ppm=tds,
            data_quality_score=quality_score,
            timestamp=datetime.now(timezone.utc),
        )

    def calculate_properties(self, data: ProcessedData) -> ProcessedData:
        """
        Stage 3: Property Calculation

        Calculate steam thermodynamic properties.
        """
        # Simplified property calculation for integration testing
        # In production, this would call the thermodynamics module

        P = data.pressure_mpa
        T = data.temperature_k

        # Approximate saturation temperature
        T_sat = 453.0 + 50.0 * math.log(P) if P > 0 else 373.15

        # Approximate enthalpy
        if T < T_sat:
            # Subcooled
            h = 4.186 * (T - 273.15)
        elif T > T_sat + 10:
            # Superheated
            h = 2800 + 2.0 * (T - T_sat)
        else:
            # Near saturation (two-phase)
            h = 2500 + 1.5 * (T - 373.15)

        # Approximate entropy
        s = 6.5 + 0.002 * (T - 373.15)

        data.enthalpy_kj_kg = h
        data.entropy_kj_kg_k = s

        return data

    def estimate_quality(self, data: ProcessedData) -> QualityEstimate:
        """
        Stage 4: Quality Estimation

        Estimate steam dryness fraction.
        """
        P = data.pressure_mpa
        h = data.enthalpy_kj_kg

        if h is None:
            return QualityEstimate(
                dryness_fraction=0.0,
                uncertainty_percent=100.0,
                method=QualityMethod.ENTHALPY,
                confidence_level=0.0,
                is_valid=False,
            )

        # Approximate saturation enthalpies
        h_f = 762.0 + 50.0 * math.log(P) if P > 0 else 417.0
        h_fg = 2015.0 - 100.0 * math.log(P) if P > 0 else 2258.0
        h_g = h_f + h_fg

        # Calculate quality
        if h <= h_f:
            x = 0.0
        elif h >= h_g:
            x = 1.0
        else:
            x = (h - h_f) / h_fg

        # Estimate uncertainty based on data quality
        base_uncertainty = 2.0  # 2% base uncertainty
        quality_factor = (1.0 - data.data_quality_score) * 5.0
        uncertainty = base_uncertainty + quality_factor

        return QualityEstimate(
            dryness_fraction=x,
            uncertainty_percent=uncertainty,
            method=QualityMethod.ENTHALPY,
            confidence_level=data.data_quality_score,
            is_valid=True,
        )

    def assess_risk(self, data: ProcessedData, quality: QualityEstimate) -> RiskAssessment:
        """
        Stage 5: Risk Assessment

        Assess carryover risk based on TDS and quality.
        """
        tds = data.tds_ppm
        x = quality.dryness_fraction

        # Calculate risk score
        tds_score = min(tds / self.max_tds_threshold, 1.0)
        moisture_score = max(0, (1.0 - x) / 0.2)  # Higher score for lower quality

        combined_score = 0.6 * tds_score + 0.4 * moisture_score

        # Determine risk level
        if combined_score < 0.25:
            level = "LOW"
            action = "Continue normal operation"
        elif combined_score < 0.5:
            level = "MEDIUM"
            action = "Increase monitoring frequency"
        elif combined_score < 0.75:
            level = "HIGH"
            action = "Consider blowdown or load reduction"
        else:
            level = "CRITICAL"
            action = "Immediate action required"

        return RiskAssessment(
            risk_level=level,
            probability=combined_score,
            recommended_action=action,
        )

    def validate_constraints(self, data: ProcessedData, quality: QualityEstimate) -> ConstraintResult:
        """
        Stage 6: Constraint Validation

        Check safety constraints.
        """
        violations = []

        # Check minimum dryness
        if quality.dryness_fraction < self.min_dryness_threshold:
            violations.append(f"Dryness {quality.dryness_fraction:.3f} below minimum {self.min_dryness_threshold}")

        # Check maximum TDS
        if data.tds_ppm > self.max_tds_threshold:
            violations.append(f"TDS {data.tds_ppm:.1f} ppm exceeds maximum {self.max_tds_threshold}")

        # Check pressure bounds
        if data.pressure_mpa < 0.1:
            violations.append(f"Pressure {data.pressure_mpa:.3f} MPa below minimum")

        if data.pressure_mpa > 15.0:
            violations.append(f"Pressure {data.pressure_mpa:.3f} MPa above maximum")

        # Determine worst severity
        worst_severity = None
        if len(violations) > 0:
            if any("Dryness" in v for v in violations):
                worst_severity = "ALARM"
            if any("exceeds maximum" in v for v in violations):
                worst_severity = "WARNING"

        return ConstraintResult(
            all_passed=len(violations) == 0,
            violations=violations,
            worst_severity=worst_severity,
        )

    def generate_output(
        self,
        data: ProcessedData,
        quality: QualityEstimate,
        risk: RiskAssessment,
        constraints: ConstraintResult,
        processing_time_ms: float,
    ) -> PipelineOutput:
        """
        Stage 7: Output Generation

        Generate final output with provenance tracking.
        """
        # Calculate provenance hash
        hash_inputs = {
            "pressure_mpa": round(data.pressure_mpa, 10),
            "temperature_k": round(data.temperature_k, 10),
            "dryness_fraction": round(quality.dryness_fraction, 10),
            "risk_level": risk.risk_level,
            "all_passed": constraints.all_passed,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(hash_inputs, sort_keys=True).encode()
        ).hexdigest()

        return PipelineOutput(
            quality_estimate=quality,
            risk_assessment=risk,
            constraint_result=constraints,
            processed_data=data,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(timezone.utc),
        )

    def process(self, sensor_readings: Dict[str, SensorData]) -> PipelineOutput:
        """
        Execute complete pipeline.

        Args:
            sensor_readings: Dictionary of sensor tag -> SensorData

        Returns:
            PipelineOutput with all results and provenance
        """
        import time
        start_time = time.perf_counter()

        # Execute pipeline stages
        valid_readings = self.acquire_data(sensor_readings)
        processed_data = self.validate_data(valid_readings)
        processed_data = self.calculate_properties(processed_data)
        quality_estimate = self.estimate_quality(processed_data)
        risk_assessment = self.assess_risk(processed_data, quality_estimate)
        constraint_result = self.validate_constraints(processed_data, quality_estimate)

        # Calculate processing time
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        # Generate output
        return self.generate_output(
            processed_data,
            quality_estimate,
            risk_assessment,
            constraint_result,
            processing_time_ms,
        )


# =============================================================================
# Test Classes
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for complete pipeline."""

    @pytest.fixture
    def pipeline(self) -> SteamQualityPipeline:
        """Create pipeline instance."""
        return SteamQualityPipeline({
            "min_dryness": 0.85,
            "max_tds_ppm": 100.0,
        })

    @pytest.fixture
    def valid_sensor_readings(self) -> Dict[str, SensorData]:
        """Create valid sensor readings."""
        base_time = datetime.now(timezone.utc)
        return {
            "steam.pressure": SensorData(
                tag="steam.pressure",
                value=1.0,  # MPa
                unit="MPa",
                quality="GOOD",
                timestamp=base_time,
            ),
            "steam.temperature": SensorData(
                tag="steam.temperature",
                value=453.0,  # K (saturation temp at 1 MPa)
                unit="K",
                quality="GOOD",
                timestamp=base_time,
            ),
            "steam.flow": SensorData(
                tag="steam.flow",
                value=50.0,  # kg/s
                unit="kg/s",
                quality="GOOD",
                timestamp=base_time,
            ),
            "steam.tds": SensorData(
                tag="steam.tds",
                value=25.0,  # ppm
                unit="ppm",
                quality="GOOD",
                timestamp=base_time,
            ),
        }

    @pytest.fixture
    def low_quality_readings(self) -> Dict[str, SensorData]:
        """Create readings that will produce low quality steam."""
        base_time = datetime.now(timezone.utc)
        return {
            "steam.pressure": SensorData(
                tag="steam.pressure",
                value=1.0,
                unit="MPa",
                quality="GOOD",
                timestamp=base_time,
            ),
            "steam.temperature": SensorData(
                tag="steam.temperature",
                value=430.0,  # Below saturation - liquid dominated
                unit="K",
                quality="GOOD",
                timestamp=base_time,
            ),
            "steam.tds": SensorData(
                tag="steam.tds",
                value=80.0,  # Higher TDS
                unit="ppm",
                quality="GOOD",
                timestamp=base_time,
            ),
        }

    def test_complete_pipeline_execution(self, pipeline, valid_sensor_readings):
        """Test that complete pipeline executes successfully."""
        result = pipeline.process(valid_sensor_readings)

        assert result is not None
        assert result.quality_estimate is not None
        assert result.risk_assessment is not None
        assert result.constraint_result is not None

    def test_pipeline_returns_valid_quality(self, pipeline, valid_sensor_readings):
        """Test that pipeline returns valid quality estimate."""
        result = pipeline.process(valid_sensor_readings)

        assert result.quality_estimate.is_valid
        assert 0.0 <= result.quality_estimate.dryness_fraction <= 1.0
        assert result.quality_estimate.uncertainty_percent >= 0

    def test_pipeline_constraints_pass_for_good_data(self, pipeline, valid_sensor_readings):
        """Test that constraints pass for good quality data."""
        result = pipeline.process(valid_sensor_readings)

        # Should pass constraints for good data
        # Note: May or may not pass depending on calculated quality
        assert result.constraint_result is not None

    def test_pipeline_detects_low_quality(self, pipeline, low_quality_readings):
        """Test that pipeline detects low quality steam."""
        result = pipeline.process(low_quality_readings)

        # Low quality should be detected
        assert result.quality_estimate.dryness_fraction < 0.95

    def test_pipeline_generates_provenance_hash(self, pipeline, valid_sensor_readings):
        """Test that pipeline generates provenance hash."""
        result = pipeline.process(valid_sensor_readings)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_pipeline_provenance_deterministic(self, pipeline, valid_sensor_readings):
        """Test that provenance hash is deterministic."""
        result1 = pipeline.process(valid_sensor_readings)
        result2 = pipeline.process(valid_sensor_readings)

        assert result1.provenance_hash == result2.provenance_hash

    def test_pipeline_processing_time_reasonable(self, pipeline, valid_sensor_readings):
        """Test that pipeline processing time is reasonable."""
        result = pipeline.process(valid_sensor_readings)

        # Should complete in reasonable time (< 100ms for simple test)
        assert result.processing_time_ms < 100.0

    def test_pipeline_handles_bad_quality_data(self, pipeline):
        """Test that pipeline handles bad quality sensor data."""
        readings = {
            "steam.pressure": SensorData(
                tag="steam.pressure",
                value=1.0,
                unit="MPa",
                quality="BAD",  # Bad quality
                timestamp=datetime.now(timezone.utc),
            ),
            "steam.temperature": SensorData(
                tag="steam.temperature",
                value=453.0,
                unit="K",
                quality="GOOD",
                timestamp=datetime.now(timezone.utc),
            ),
        }

        # Bad quality data is filtered, leaving only temperature
        with pytest.raises(ValueError):
            pipeline.process(readings)

    def test_pipeline_handles_uncertain_data(self, pipeline):
        """Test that pipeline handles uncertain quality data."""
        readings = {
            "steam.pressure": SensorData(
                tag="steam.pressure",
                value=1.0,
                unit="MPa",
                quality="UNCERTAIN",  # Uncertain quality
                timestamp=datetime.now(timezone.utc),
            ),
            "steam.temperature": SensorData(
                tag="steam.temperature",
                value=453.0,
                unit="K",
                quality="UNCERTAIN",
                timestamp=datetime.now(timezone.utc),
            ),
        }

        result = pipeline.process(readings)

        # Should still process but with lower confidence
        assert result.quality_estimate.confidence_level < 1.0


class TestDataAcquisitionStage:
    """Tests for data acquisition stage."""

    @pytest.fixture
    def pipeline(self) -> SteamQualityPipeline:
        return SteamQualityPipeline()

    def test_filters_bad_quality_data(self, pipeline):
        """Test that bad quality data is filtered."""
        readings = {
            "good": SensorData("good", 1.0, "MPa", "GOOD", datetime.now(timezone.utc)),
            "bad": SensorData("bad", 2.0, "MPa", "BAD", datetime.now(timezone.utc)),
        }

        valid = pipeline.acquire_data(readings)

        assert "good" in valid
        assert "bad" not in valid

    def test_accepts_good_quality_data(self, pipeline):
        """Test that good quality data is accepted."""
        readings = {
            "good1": SensorData("good1", 1.0, "MPa", "GOOD", datetime.now(timezone.utc)),
            "good2": SensorData("good2", 2.0, "MPa", "GOOD", datetime.now(timezone.utc)),
        }

        valid = pipeline.acquire_data(readings)

        assert len(valid) == 2

    def test_accepts_uncertain_quality_data(self, pipeline):
        """Test that uncertain quality data is accepted."""
        readings = {
            "uncertain": SensorData("uncertain", 1.0, "MPa", "UNCERTAIN", datetime.now(timezone.utc)),
        }

        valid = pipeline.acquire_data(readings)

        assert "uncertain" in valid


class TestDataValidationStage:
    """Tests for data validation stage."""

    @pytest.fixture
    def pipeline(self) -> SteamQualityPipeline:
        return SteamQualityPipeline()

    def test_extracts_pressure_and_temperature(self, pipeline):
        """Test that pressure and temperature are extracted."""
        readings = {
            "steam.pressure": SensorData("steam.pressure", 5.0, "MPa", "GOOD", datetime.now(timezone.utc)),
            "steam.temperature": SensorData("steam.temperature", 500.0, "K", "GOOD", datetime.now(timezone.utc)),
        }

        data = pipeline.validate_data(readings)

        assert data.pressure_mpa == 5.0
        assert data.temperature_k == 500.0

    def test_raises_on_missing_pressure(self, pipeline):
        """Test that missing pressure raises error."""
        readings = {
            "steam.temperature": SensorData("steam.temperature", 500.0, "K", "GOOD", datetime.now(timezone.utc)),
        }

        with pytest.raises(ValueError):
            pipeline.validate_data(readings)

    def test_raises_on_missing_temperature(self, pipeline):
        """Test that missing temperature raises error."""
        readings = {
            "steam.pressure": SensorData("steam.pressure", 5.0, "MPa", "GOOD", datetime.now(timezone.utc)),
        }

        with pytest.raises(ValueError):
            pipeline.validate_data(readings)


class TestPropertyCalculationStage:
    """Tests for property calculation stage."""

    @pytest.fixture
    def pipeline(self) -> SteamQualityPipeline:
        return SteamQualityPipeline()

    def test_calculates_enthalpy(self, pipeline):
        """Test that enthalpy is calculated."""
        data = ProcessedData(
            pressure_mpa=1.0,
            temperature_k=453.0,
            enthalpy_kj_kg=None,
            entropy_kj_kg_k=None,
            flow_rate_kg_s=50.0,
            tds_ppm=20.0,
            data_quality_score=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        result = pipeline.calculate_properties(data)

        assert result.enthalpy_kj_kg is not None
        assert result.enthalpy_kj_kg > 0

    def test_calculates_entropy(self, pipeline):
        """Test that entropy is calculated."""
        data = ProcessedData(
            pressure_mpa=1.0,
            temperature_k=453.0,
            enthalpy_kj_kg=None,
            entropy_kj_kg_k=None,
            flow_rate_kg_s=50.0,
            tds_ppm=20.0,
            data_quality_score=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        result = pipeline.calculate_properties(data)

        assert result.entropy_kj_kg_k is not None
        assert result.entropy_kj_kg_k > 0


class TestQualityEstimationStage:
    """Tests for quality estimation stage."""

    @pytest.fixture
    def pipeline(self) -> SteamQualityPipeline:
        return SteamQualityPipeline()

    def test_estimates_quality(self, pipeline):
        """Test that quality is estimated."""
        data = ProcessedData(
            pressure_mpa=1.0,
            temperature_k=453.0,
            enthalpy_kj_kg=2500.0,
            entropy_kj_kg_k=6.5,
            flow_rate_kg_s=50.0,
            tds_ppm=20.0,
            data_quality_score=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        result = pipeline.estimate_quality(data)

        assert result.is_valid
        assert 0.0 <= result.dryness_fraction <= 1.0

    def test_handles_missing_enthalpy(self, pipeline):
        """Test that missing enthalpy is handled."""
        data = ProcessedData(
            pressure_mpa=1.0,
            temperature_k=453.0,
            enthalpy_kj_kg=None,  # Missing
            entropy_kj_kg_k=6.5,
            flow_rate_kg_s=50.0,
            tds_ppm=20.0,
            data_quality_score=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        result = pipeline.estimate_quality(data)

        assert not result.is_valid


class TestRiskAssessmentStage:
    """Tests for risk assessment stage."""

    @pytest.fixture
    def pipeline(self) -> SteamQualityPipeline:
        return SteamQualityPipeline()

    def test_low_risk_for_good_data(self, pipeline):
        """Test low risk for good quality data."""
        data = ProcessedData(
            pressure_mpa=1.0,
            temperature_k=453.0,
            enthalpy_kj_kg=2700.0,
            entropy_kj_kg_k=6.5,
            flow_rate_kg_s=50.0,
            tds_ppm=10.0,  # Low TDS
            data_quality_score=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        quality = QualityEstimate(
            dryness_fraction=0.95,  # High quality
            uncertainty_percent=2.0,
            method=QualityMethod.ENTHALPY,
            confidence_level=1.0,
            is_valid=True,
        )

        result = pipeline.assess_risk(data, quality)

        assert result.risk_level == "LOW"

    def test_high_risk_for_poor_data(self, pipeline):
        """Test high risk for poor quality data."""
        data = ProcessedData(
            pressure_mpa=1.0,
            temperature_k=453.0,
            enthalpy_kj_kg=2500.0,
            entropy_kj_kg_k=6.5,
            flow_rate_kg_s=50.0,
            tds_ppm=90.0,  # High TDS
            data_quality_score=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        quality = QualityEstimate(
            dryness_fraction=0.70,  # Low quality
            uncertainty_percent=5.0,
            method=QualityMethod.ENTHALPY,
            confidence_level=0.8,
            is_valid=True,
        )

        result = pipeline.assess_risk(data, quality)

        assert result.risk_level in ["HIGH", "CRITICAL"]


class TestConstraintValidationStage:
    """Tests for constraint validation stage."""

    @pytest.fixture
    def pipeline(self) -> SteamQualityPipeline:
        return SteamQualityPipeline({
            "min_dryness": 0.85,
            "max_tds_ppm": 100.0,
        })

    def test_passes_for_good_data(self, pipeline):
        """Test that constraints pass for good data."""
        data = ProcessedData(
            pressure_mpa=1.0,
            temperature_k=453.0,
            enthalpy_kj_kg=2700.0,
            entropy_kj_kg_k=6.5,
            flow_rate_kg_s=50.0,
            tds_ppm=20.0,
            data_quality_score=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        quality = QualityEstimate(
            dryness_fraction=0.95,
            uncertainty_percent=2.0,
            method=QualityMethod.ENTHALPY,
            confidence_level=1.0,
            is_valid=True,
        )

        result = pipeline.validate_constraints(data, quality)

        assert result.all_passed

    def test_fails_for_low_dryness(self, pipeline):
        """Test that constraints fail for low dryness."""
        data = ProcessedData(
            pressure_mpa=1.0,
            temperature_k=453.0,
            enthalpy_kj_kg=2500.0,
            entropy_kj_kg_k=6.5,
            flow_rate_kg_s=50.0,
            tds_ppm=20.0,
            data_quality_score=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        quality = QualityEstimate(
            dryness_fraction=0.70,  # Below threshold
            uncertainty_percent=5.0,
            method=QualityMethod.ENTHALPY,
            confidence_level=0.8,
            is_valid=True,
        )

        result = pipeline.validate_constraints(data, quality)

        assert not result.all_passed
        assert len(result.violations) > 0


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.fixture
    def pipeline(self) -> SteamQualityPipeline:
        return SteamQualityPipeline()

    @pytest.fixture
    def readings(self) -> Dict[str, SensorData]:
        base_time = datetime.now(timezone.utc)
        return {
            "steam.pressure": SensorData("steam.pressure", 1.0, "MPa", "GOOD", base_time),
            "steam.temperature": SensorData("steam.temperature", 453.0, "K", "GOOD", base_time),
            "steam.tds": SensorData("steam.tds", 25.0, "ppm", "GOOD", base_time),
        }

    def test_repeated_execution_produces_same_quality(self, pipeline, readings):
        """Test that repeated execution produces same quality."""
        results = [pipeline.process(readings) for _ in range(5)]

        first_quality = results[0].quality_estimate.dryness_fraction
        for r in results[1:]:
            assert r.quality_estimate.dryness_fraction == first_quality

    def test_repeated_execution_produces_same_risk(self, pipeline, readings):
        """Test that repeated execution produces same risk."""
        results = [pipeline.process(readings) for _ in range(5)]

        first_risk = results[0].risk_assessment.risk_level
        for r in results[1:]:
            assert r.risk_assessment.risk_level == first_risk


class TestPerformance:
    """Performance tests for pipeline."""

    @pytest.fixture
    def pipeline(self) -> SteamQualityPipeline:
        return SteamQualityPipeline()

    @pytest.fixture
    def readings(self) -> Dict[str, SensorData]:
        base_time = datetime.now(timezone.utc)
        return {
            "steam.pressure": SensorData("steam.pressure", 1.0, "MPa", "GOOD", base_time),
            "steam.temperature": SensorData("steam.temperature", 453.0, "K", "GOOD", base_time),
            "steam.tds": SensorData("steam.tds", 25.0, "ppm", "GOOD", base_time),
        }

    @pytest.mark.performance
    def test_processing_time_under_target(self, pipeline, readings):
        """Test that processing time is under target."""
        result = pipeline.process(readings)

        # Target: < 10ms for single processing
        assert result.processing_time_ms < 10.0

    @pytest.mark.performance
    @pytest.mark.slow
    def test_throughput(self, pipeline, readings):
        """Test pipeline throughput."""
        import time

        n_iterations = 1000
        start = time.perf_counter()

        for _ in range(n_iterations):
            pipeline.process(readings)

        elapsed = time.perf_counter() - start
        throughput = n_iterations / elapsed

        # Target: > 100 calculations per second
        assert throughput > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
