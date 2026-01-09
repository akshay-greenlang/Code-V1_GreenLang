# -*- coding: utf-8 -*-
"""
Unit Tests: Condensync Agent

Comprehensive tests for the main GL-017 Condensync agent including:
- Agent initialization and configuration
- Main orchestration methods
- Pipeline execution
- Error handling
- Batch processing

Target Coverage: 85%+
Author: GL-TestEngineer
Date: December 2025
"""

import hashlib
import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import (
    FailureMode,
    FailureSeverity,
    CleaningMethod,
    OperatingMode,
    TubeMaterial,
    WaterSource,
    CondenserConfig,
    CondenserReading,
    ThermalInput,
    VacuumOptimizationInput,
    FoulingHistoryEntry,
    HEICalculationResult,
    VacuumOptimizationResult,
    FoulingPredictionResult,
    CondenserStateResult,
    MockCondenserSensorConnector,
    MockHistorianConnector,
    MockCMMSConnector,
    AssertionHelpers,
    ProvenanceCalculator,
    PerformanceTimer,
    TEST_SEED,
    OPERATING_LIMITS,
)


# =============================================================================
# CONDENSYNC AGENT IMPLEMENTATION FOR TESTING
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for Condensync agent."""
    agent_id: str = "GL-017"
    agent_name: str = "Condensync"
    version: str = "1.0.0"
    environment: str = "test"
    track_provenance: bool = True
    enable_caching: bool = False
    log_level: str = "INFO"


@dataclass
class CondensyncInput:
    """Input for Condensync agent processing."""
    condenser_id: str
    reading: CondenserReading
    config: CondenserConfig
    history: List[FoulingHistoryEntry] = field(default_factory=list)
    electricity_price_usd_mwh: float = 50.0


@dataclass
class CondensyncOutput:
    """Output from Condensync agent processing."""
    condenser_id: str
    timestamp: datetime
    cleanliness_factor: float
    state_classification: CondenserStateResult
    vacuum_optimization: Optional[VacuumOptimizationResult]
    fouling_prediction: Optional[FoulingPredictionResult]
    recommendations: List[str]
    estimated_savings_usd: float
    provenance_hash: str
    processing_time_ms: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condenser_id": self.condenser_id,
            "timestamp": self.timestamp.isoformat(),
            "cleanliness_factor": self.cleanliness_factor,
            "state": self.state_classification.failure_mode.value,
            "severity": self.state_classification.severity.value,
            "estimated_savings_usd": self.estimated_savings_usd,
            "provenance_hash": self.provenance_hash,
        }


class CondensyncAgent:
    """
    GL-017 Condensync - Condenser Optimization Agent.

    Orchestrates condenser performance analysis including:
    - Cleanliness factor calculation (HEI)
    - Vacuum optimization
    - Fouling prediction
    - State classification
    - Maintenance recommendations
    """

    VERSION = "1.0.0"
    AGENT_ID = "GL-017"

    def __init__(self, config: AgentConfig = None):
        """Initialize Condensync agent."""
        self.config = config or AgentConfig()
        self._processing_count = 0
        self._error_count = 0

        # Import calculators (would be actual imports in real implementation)
        self._hei_calculator = None
        self._vacuum_calculator = None
        self._fouling_calculator = None
        self._state_classifier = None

        # Initialize connectors
        self._sensor_connector: Optional[MockCondenserSensorConnector] = None
        self._historian_connector: Optional[MockHistorianConnector] = None
        self._cmms_connector: Optional[MockCMMSConnector] = None

    def set_sensor_connector(self, connector: MockCondenserSensorConnector):
        """Set sensor connector."""
        self._sensor_connector = connector

    def set_historian_connector(self, connector: MockHistorianConnector):
        """Set historian connector."""
        self._historian_connector = connector

    def set_cmms_connector(self, connector: MockCMMSConnector):
        """Set CMMS connector."""
        self._cmms_connector = connector

    def calculate_cleanliness_factor(
        self,
        reading: CondenserReading,
        config: CondenserConfig,
        design_ua: float = 80000.0
    ) -> float:
        """
        Calculate cleanliness factor from reading.

        Args:
            reading: Condenser reading
            config: Condenser configuration
            design_ua: Design UA value (kW/K)

        Returns:
            Cleanliness factor (0-1)
        """
        import math

        # Calculate LMTD
        ttd = reading.saturation_temp_c - reading.cw_outlet_temp_c
        approach = reading.saturation_temp_c - reading.cw_inlet_temp_c

        if ttd <= 0 or approach <= 0 or approach < ttd:
            return 0.85  # Default to design

        if abs(ttd - approach) < 0.001:
            lmtd = ttd
        else:
            lmtd = (approach - ttd) / math.log(approach / ttd)

        # Calculate heat duty
        cw_flow_kg_s = reading.cw_flow_m3_s * 1000
        heat_duty_kw = cw_flow_kg_s * 4.186 * (reading.cw_outlet_temp_c - reading.cw_inlet_temp_c)

        # Calculate actual UA
        ua_actual = heat_duty_kw / lmtd if lmtd > 0 else 0

        # Calculate CF
        cf = ua_actual / design_ua if design_ua > 0 else 0

        return max(0.0, min(1.0, cf))

    def classify_state(
        self,
        reading: CondenserReading,
        config: CondenserConfig = None
    ) -> CondenserStateResult:
        """
        Classify condenser state.

        Args:
            reading: Condenser reading
            config: Condenser configuration

        Returns:
            CondenserStateResult
        """
        # Feature extraction
        ttd = reading.saturation_temp_c - reading.cw_outlet_temp_c
        air_ingress = reading.air_ingress_scfm
        do = reading.dissolved_oxygen_ppb
        subcooling = reading.subcooling_c

        # Classification logic
        failure_mode = FailureMode.NORMAL
        severity = FailureSeverity.NONE
        confidence = 1.0
        factors = []
        recommendations = []

        # Check for air leak
        if air_ingress > 10.0 or (air_ingress > 5.0 and do > 30.0):
            failure_mode = FailureMode.AIR_LEAK_MAJOR if air_ingress > 10.0 else FailureMode.AIR_LEAK_MINOR
            severity = FailureSeverity.HIGH if air_ingress > 10.0 else FailureSeverity.MEDIUM
            confidence = min(1.0, air_ingress / 15.0)
            factors.append("Elevated air ingress")
            if do > 30.0:
                factors.append("High dissolved oxygen")
            recommendations.append("Conduct air leak survey")

        # Check for fouling
        elif ttd > 8.0:
            failure_mode = FailureMode.FOULING_BIOLOGICAL
            severity = FailureSeverity.HIGH if ttd > 12.0 else FailureSeverity.MEDIUM
            confidence = min(1.0, ttd / 15.0)
            factors.append("Elevated TTD")
            recommendations.append("Schedule condenser cleaning")

        else:
            recommendations.append("Continue normal monitoring")

        # Calculate MW impact
        impact_factor = {
            FailureSeverity.NONE: 0.0,
            FailureSeverity.LOW: 0.005,
            FailureSeverity.MEDIUM: 0.01,
            FailureSeverity.HIGH: 0.02,
            FailureSeverity.CRITICAL: 0.03,
        }
        mw_impact = reading.unit_load_mw * impact_factor.get(severity, 0.0)

        # Provenance hash
        input_data = reading.to_dict()
        provenance_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        return CondenserStateResult(
            condenser_id=reading.condenser_id,
            timestamp=reading.timestamp,
            failure_mode=failure_mode,
            severity=severity,
            confidence=round(confidence, 3),
            contributing_factors=factors,
            recommended_actions=recommendations,
            estimated_impact_mw=round(mw_impact, 3),
            provenance_hash=provenance_hash,
        )

    def optimize_vacuum(
        self,
        reading: CondenserReading,
        config: CondenserConfig,
        electricity_price: float = 50.0
    ) -> VacuumOptimizationResult:
        """
        Optimize vacuum/backpressure.

        Args:
            reading: Condenser reading
            config: Condenser configuration
            electricity_price: Electricity price ($/MWh)

        Returns:
            VacuumOptimizationResult
        """
        current_bp = reading.vacuum_pressure_kpa_abs

        # Calculate achievable vacuum
        min_sat_temp = reading.cw_inlet_temp_c + 5.0  # 5C approach
        from conftest import pressure_from_saturation_temp
        achievable_bp = max(2.5, pressure_from_saturation_temp(min_sat_temp))

        # Calculate potential gain
        bp_improvement = current_bp - achievable_bp
        heat_rate_sensitivity = 50.0  # kJ/kWh per kPa
        mw_gain = reading.unit_load_mw * (bp_improvement * heat_rate_sensitivity / 9500) * 0.01

        # Annual savings
        annual_savings = mw_gain * electricity_price * 8760 * 0.85

        # Sensitivity analysis
        sensitivity = {
            "mw_per_kpa": mw_gain / bp_improvement if bp_improvement > 0 else 0,
            "achievable_min_kpa": achievable_bp,
            "current_vs_achievable_kpa": bp_improvement,
        }

        # Provenance hash
        input_data = {
            "current_bp": current_bp,
            "unit_load": reading.unit_load_mw,
            "cw_inlet": reading.cw_inlet_temp_c,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()

        return VacuumOptimizationResult(
            optimal_backpressure_kpa=round(achievable_bp, 3),
            current_backpressure_kpa=current_bp,
            mw_gain_potential=round(max(0, mw_gain), 3),
            annual_savings_usd=round(max(0, annual_savings), 2),
            cw_flow_recommendation_m3_s=reading.cw_flow_m3_s,
            economic_optimum=bp_improvement > 0.5,
            sensitivity_analysis=sensitivity,
            provenance_hash=provenance_hash,
        )

    def predict_fouling(
        self,
        current_cf: float,
        history: List[FoulingHistoryEntry],
        config: CondenserConfig = None,
        electricity_price: float = 50.0
    ) -> FoulingPredictionResult:
        """
        Predict fouling trajectory.

        Args:
            current_cf: Current cleanliness factor
            history: Historical CF data
            config: Condenser configuration
            electricity_price: Electricity price

        Returns:
            FoulingPredictionResult
        """
        # Estimate decay rate
        if len(history) >= 2:
            sorted_history = sorted(history, key=lambda x: x.timestamp, reverse=True)
            cf_diff = sorted_history[-1].cleanliness_factor - sorted_history[0].cleanliness_factor
            days_diff = (sorted_history[-1].timestamp - sorted_history[0].timestamp).total_seconds() / 86400
            decay_rate = abs(cf_diff / days_diff) if days_diff > 0 else 0.002
        else:
            decay_rate = 0.002

        # Predict CF
        predicted_cf = max(0.60, current_cf - decay_rate * 30)

        # Days to threshold
        threshold = 0.75
        if current_cf > threshold:
            days = int((current_cf - threshold) / decay_rate) if decay_rate > 0 else -1
        else:
            days = 0

        # Cleaning recommendation
        if current_cf < 0.70:
            method = CleaningMethod.OFFLINE_CHEMICAL
            cost = 75000.0
        elif current_cf < 0.75:
            method = CleaningMethod.OFFLINE_HYDROLANCE
            cost = 25000.0
        else:
            method = CleaningMethod.ONLINE_BALL
            cost = 100.0

        # Lost generation cost
        cf_clean = 0.92
        mw_loss = 500 * (cf_clean - current_cf) * 0.01
        lost_gen_cost = mw_loss * electricity_price * 720  # 30 days

        # ROI
        roi = lost_gen_cost / cost if cost > 0 else 0

        # Provenance hash
        input_data = {"current_cf": current_cf, "decay_rate": decay_rate}
        provenance_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()

        return FoulingPredictionResult(
            current_cf=current_cf,
            predicted_cf=round(predicted_cf, 4),
            days_to_threshold=days,
            cf_decay_rate_per_day=round(decay_rate, 5),
            recommended_cleaning_date=datetime.now(timezone.utc) + timedelta(days=max(1, days - 7)),
            cleaning_method=method,
            cleaning_cost_usd=cost,
            lost_generation_cost_usd=round(lost_gen_cost, 2),
            roi_cleaning=round(roi, 2),
            provenance_hash=provenance_hash,
        )

    def process(self, input_data: CondensyncInput) -> CondensyncOutput:
        """
        Process condenser data through full analysis pipeline.

        Args:
            input_data: Condensync input

        Returns:
            CondensyncOutput with all analysis results
        """
        import time
        start_time = time.perf_counter()

        self._processing_count += 1
        warnings = []

        # Calculate cleanliness factor
        cf = self.calculate_cleanliness_factor(
            input_data.reading,
            input_data.config
        )

        # Classify state
        state = self.classify_state(
            input_data.reading,
            input_data.config
        )

        # Optimize vacuum
        vacuum_opt = self.optimize_vacuum(
            input_data.reading,
            input_data.config,
            input_data.electricity_price_usd_mwh
        )

        # Predict fouling
        fouling_pred = self.predict_fouling(
            cf,
            input_data.history,
            input_data.config,
            input_data.electricity_price_usd_mwh
        )

        # Compile recommendations
        recommendations = state.recommended_actions.copy()
        if vacuum_opt.economic_optimum:
            recommendations.append(
                f"Optimize vacuum to save ${vacuum_opt.annual_savings_usd:,.0f}/year"
            )
        if fouling_pred.days_to_threshold < 30 and fouling_pred.days_to_threshold > 0:
            recommendations.append(
                f"Plan cleaning within {fouling_pred.days_to_threshold} days"
            )

        # Calculate total savings potential
        estimated_savings = vacuum_opt.annual_savings_usd

        # Generate output provenance hash
        output_data = {
            "condenser_id": input_data.condenser_id,
            "cf": cf,
            "state": state.failure_mode.value,
            "vacuum_opt": vacuum_opt.optimal_backpressure_kpa,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(output_data, sort_keys=True).encode()
        ).hexdigest()

        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return CondensyncOutput(
            condenser_id=input_data.condenser_id,
            timestamp=input_data.reading.timestamp,
            cleanliness_factor=round(cf, 4),
            state_classification=state,
            vacuum_optimization=vacuum_opt,
            fouling_prediction=fouling_pred,
            recommendations=recommendations,
            estimated_savings_usd=estimated_savings,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time_ms, 3),
            warnings=warnings,
        )

    async def process_async(self, input_data: CondensyncInput) -> CondensyncOutput:
        """Async version of process."""
        return self.process(input_data)

    def process_batch(
        self,
        inputs: List[CondensyncInput]
    ) -> List[CondensyncOutput]:
        """
        Process batch of condenser inputs.

        Args:
            inputs: List of inputs

        Returns:
            List of outputs
        """
        return [self.process(inp) for inp in inputs]

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        return {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "processing_count": self._processing_count,
            "error_count": self._error_count,
        }


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def agent() -> CondensyncAgent:
    """Create agent instance."""
    return CondensyncAgent()


@pytest.fixture
def agent_with_config() -> CondensyncAgent:
    """Create agent with custom config."""
    config = AgentConfig(
        agent_id="GL-017-TEST",
        environment="test",
        track_provenance=True,
    )
    return CondensyncAgent(config)


@pytest.fixture
def condensync_input(
    healthy_condenser_reading: CondenserReading,
    sample_condenser_config: CondenserConfig,
    fouling_history_clean: List[FoulingHistoryEntry]
) -> CondensyncInput:
    """Create standard Condensync input."""
    return CondensyncInput(
        condenser_id="COND-001",
        reading=healthy_condenser_reading,
        config=sample_condenser_config,
        history=fouling_history_clean,
        electricity_price_usd_mwh=50.0,
    )


@pytest.fixture
def condensync_inputs_batch(
    condenser_fleet: List[CondenserReading],
    sample_condenser_config: CondenserConfig,
    fouling_history_clean: List[FoulingHistoryEntry]
) -> List[CondensyncInput]:
    """Create batch of inputs."""
    return [
        CondensyncInput(
            condenser_id=reading.condenser_id,
            reading=reading,
            config=sample_condenser_config,
            history=fouling_history_clean,
        )
        for reading in condenser_fleet
    ]


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestAgentInitialization:
    """Tests for agent initialization."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test agent initializes with defaults."""
        agent = CondensyncAgent()

        assert agent.config.agent_id == "GL-017"
        assert agent.config.version == "1.0.0"
        assert agent._processing_count == 0

    @pytest.mark.unit
    def test_custom_config_initialization(self):
        """Test agent initializes with custom config."""
        config = AgentConfig(
            agent_id="GL-017-CUSTOM",
            environment="production",
        )
        agent = CondensyncAgent(config)

        assert agent.config.agent_id == "GL-017-CUSTOM"
        assert agent.config.environment == "production"

    @pytest.mark.unit
    def test_connector_setup(self, agent: CondensyncAgent):
        """Test connector setup."""
        sensor = MockCondenserSensorConnector()
        historian = MockHistorianConnector()
        cmms = MockCMMSConnector()

        agent.set_sensor_connector(sensor)
        agent.set_historian_connector(historian)
        agent.set_cmms_connector(cmms)

        assert agent._sensor_connector is sensor
        assert agent._historian_connector is historian
        assert agent._cmms_connector is cmms


class TestCleanlinessFactorCalculation:
    """Tests for CF calculation in agent."""

    @pytest.mark.unit
    def test_cf_calculation_healthy(
        self,
        agent: CondensyncAgent,
        healthy_condenser_reading: CondenserReading,
        sample_condenser_config: CondenserConfig
    ):
        """Test CF calculation for healthy condenser."""
        cf = agent.calculate_cleanliness_factor(
            healthy_condenser_reading,
            sample_condenser_config
        )

        assert 0.0 < cf <= 1.0

    @pytest.mark.unit
    def test_cf_calculation_fouled(
        self,
        agent: CondensyncAgent,
        fouled_condenser_reading: CondenserReading,
        sample_condenser_config: CondenserConfig
    ):
        """Test CF calculation for fouled condenser."""
        cf = agent.calculate_cleanliness_factor(
            fouled_condenser_reading,
            sample_condenser_config
        )

        # Fouled should have lower CF
        assert cf < 0.9


class TestStateClassification:
    """Tests for state classification in agent."""

    @pytest.mark.unit
    def test_classify_healthy(
        self,
        agent: CondensyncAgent,
        healthy_condenser_reading: CondenserReading
    ):
        """Test classification of healthy condenser."""
        result = agent.classify_state(healthy_condenser_reading)

        assert result.failure_mode == FailureMode.NORMAL
        assert result.severity == FailureSeverity.NONE

    @pytest.mark.unit
    def test_classify_fouled(
        self,
        agent: CondensyncAgent,
        fouled_condenser_reading: CondenserReading
    ):
        """Test classification of fouled condenser."""
        result = agent.classify_state(fouled_condenser_reading)

        assert result.failure_mode != FailureMode.NORMAL

    @pytest.mark.unit
    def test_classify_air_leak(
        self,
        agent: CondensyncAgent,
        air_leak_condenser_reading: CondenserReading
    ):
        """Test classification of air leak."""
        result = agent.classify_state(air_leak_condenser_reading)

        assert result.failure_mode in [
            FailureMode.AIR_LEAK_MINOR,
            FailureMode.AIR_LEAK_MAJOR,
        ]


class TestVacuumOptimization:
    """Tests for vacuum optimization in agent."""

    @pytest.mark.unit
    def test_vacuum_optimization_basic(
        self,
        agent: CondensyncAgent,
        healthy_condenser_reading: CondenserReading,
        sample_condenser_config: CondenserConfig
    ):
        """Test basic vacuum optimization."""
        result = agent.optimize_vacuum(
            healthy_condenser_reading,
            sample_condenser_config
        )

        assert isinstance(result, VacuumOptimizationResult)
        assert result.optimal_backpressure_kpa > 0

    @pytest.mark.unit
    def test_vacuum_optimization_includes_savings(
        self,
        agent: CondensyncAgent,
        healthy_condenser_reading: CondenserReading,
        sample_condenser_config: CondenserConfig
    ):
        """Test vacuum optimization includes savings estimate."""
        result = agent.optimize_vacuum(
            healthy_condenser_reading,
            sample_condenser_config,
            electricity_price=100.0
        )

        assert result.annual_savings_usd >= 0


class TestFoulingPrediction:
    """Tests for fouling prediction in agent."""

    @pytest.mark.unit
    def test_fouling_prediction_basic(
        self,
        agent: CondensyncAgent,
        fouling_history_clean: List[FoulingHistoryEntry]
    ):
        """Test basic fouling prediction."""
        result = agent.predict_fouling(0.85, fouling_history_clean)

        assert isinstance(result, FoulingPredictionResult)
        assert result.predicted_cf <= 0.85

    @pytest.mark.unit
    def test_fouling_prediction_decay_rate(
        self,
        agent: CondensyncAgent,
        fouling_history_rapid_decline: List[FoulingHistoryEntry]
    ):
        """Test fouling prediction with rapid decline."""
        result = agent.predict_fouling(0.80, fouling_history_rapid_decline)

        # Rapid decline should show faster decay
        assert result.cf_decay_rate_per_day > 0.001


class TestMainProcessing:
    """Tests for main processing pipeline."""

    @pytest.mark.unit
    def test_process_basic(
        self,
        agent: CondensyncAgent,
        condensync_input: CondensyncInput
    ):
        """Test basic processing."""
        result = agent.process(condensync_input)

        assert isinstance(result, CondensyncOutput)
        assert result.condenser_id == condensync_input.condenser_id
        assert result.cleanliness_factor > 0
        assert result.state_classification is not None
        assert result.vacuum_optimization is not None
        assert result.fouling_prediction is not None

    @pytest.mark.unit
    def test_process_has_recommendations(
        self,
        agent: CondensyncAgent,
        condensync_input: CondensyncInput
    ):
        """Test processing includes recommendations."""
        result = agent.process(condensync_input)

        assert len(result.recommendations) > 0

    @pytest.mark.unit
    def test_process_has_provenance(
        self,
        agent: CondensyncAgent,
        condensync_input: CondensyncInput
    ):
        """Test processing includes provenance hash."""
        result = agent.process(condensync_input)

        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_process_has_timing(
        self,
        agent: CondensyncAgent,
        condensync_input: CondensyncInput
    ):
        """Test processing includes timing."""
        result = agent.process(condensync_input)

        assert result.processing_time_ms > 0

    @pytest.mark.unit
    def test_process_increments_count(
        self,
        agent: CondensyncAgent,
        condensync_input: CondensyncInput
    ):
        """Test processing increments counter."""
        initial_count = agent._processing_count

        agent.process(condensync_input)

        assert agent._processing_count == initial_count + 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_async(
        self,
        agent: CondensyncAgent,
        condensync_input: CondensyncInput
    ):
        """Test async processing."""
        result = await agent.process_async(condensync_input)

        assert isinstance(result, CondensyncOutput)
        assert result.condenser_id == condensync_input.condenser_id


class TestBatchProcessing:
    """Tests for batch processing."""

    @pytest.mark.unit
    def test_batch_processing_basic(
        self,
        agent: CondensyncAgent,
        condensync_inputs_batch: List[CondensyncInput]
    ):
        """Test basic batch processing."""
        results = agent.process_batch(condensync_inputs_batch)

        assert len(results) == len(condensync_inputs_batch)
        assert all(isinstance(r, CondensyncOutput) for r in results)

    @pytest.mark.unit
    def test_batch_processing_preserves_order(
        self,
        agent: CondensyncAgent,
        condensync_inputs_batch: List[CondensyncInput]
    ):
        """Test batch processing preserves input order."""
        results = agent.process_batch(condensync_inputs_batch)

        for i, result in enumerate(results):
            assert result.condenser_id == condensync_inputs_batch[i].condenser_id

    @pytest.mark.unit
    def test_batch_processing_unique_hashes(
        self,
        agent: CondensyncAgent,
        condensync_inputs_batch: List[CondensyncInput]
    ):
        """Test batch processing produces unique hashes."""
        results = agent.process_batch(condensync_inputs_batch)

        hashes = [r.provenance_hash for r in results]
        unique_hashes = set(hashes)

        assert len(unique_hashes) == len(hashes)


class TestOutputSerialization:
    """Tests for output serialization."""

    @pytest.mark.unit
    def test_output_to_dict(
        self,
        agent: CondensyncAgent,
        condensync_input: CondensyncInput
    ):
        """Test output can be serialized to dict."""
        result = agent.process(condensync_input)

        result_dict = result.to_dict()

        assert "condenser_id" in result_dict
        assert "timestamp" in result_dict
        assert "cleanliness_factor" in result_dict
        assert "provenance_hash" in result_dict

    @pytest.mark.unit
    def test_output_to_json(
        self,
        agent: CondensyncAgent,
        condensync_input: CondensyncInput
    ):
        """Test output can be serialized to JSON."""
        result = agent.process(condensync_input)

        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)

        assert len(json_str) > 0
        # Verify can be parsed back
        parsed = json.loads(json_str)
        assert parsed["condenser_id"] == result.condenser_id


class TestMetrics:
    """Tests for agent metrics."""

    @pytest.mark.unit
    def test_metrics_structure(self, agent: CondensyncAgent):
        """Test metrics structure."""
        metrics = agent.get_metrics()

        assert "agent_id" in metrics
        assert "version" in metrics
        assert "processing_count" in metrics

    @pytest.mark.unit
    def test_metrics_update_on_process(
        self,
        agent: CondensyncAgent,
        condensync_input: CondensyncInput
    ):
        """Test metrics update after processing."""
        initial_metrics = agent.get_metrics()

        agent.process(condensync_input)
        agent.process(condensync_input)

        updated_metrics = agent.get_metrics()

        assert updated_metrics["processing_count"] == initial_metrics["processing_count"] + 2


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.unit
    @pytest.mark.golden
    def test_process_is_deterministic(
        self,
        agent: CondensyncAgent,
        condensync_input: CondensyncInput
    ):
        """Test processing is deterministic."""
        results = [agent.process(condensync_input) for _ in range(10)]

        # All CF values should be identical
        cf_values = [r.cleanliness_factor for r in results]
        assert len(set(cf_values)) == 1

        # All hashes should be identical
        hashes = [r.provenance_hash for r in results]
        assert len(set(hashes)) == 1

    @pytest.mark.unit
    @pytest.mark.golden
    def test_state_classification_is_deterministic(
        self,
        agent: CondensyncAgent,
        healthy_condenser_reading: CondenserReading
    ):
        """Test state classification is deterministic."""
        results = [
            agent.classify_state(healthy_condenser_reading)
            for _ in range(10)
        ]

        modes = [r.failure_mode for r in results]
        assert len(set(modes)) == 1


class TestPerformance:
    """Performance tests."""

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_process_time(
        self,
        agent: CondensyncAgent,
        condensync_input: CondensyncInput
    ):
        """Test single processing time."""
        result = agent.process(condensync_input)

        # Should complete in < 50ms
        assert result.processing_time_ms < 50

    @pytest.mark.unit
    @pytest.mark.performance
    def test_batch_throughput(
        self,
        agent: CondensyncAgent,
        condensync_inputs_batch: List[CondensyncInput],
        performance_timer,
        throughput_measurer
    ):
        """Test batch processing throughput."""
        measurer = throughput_measurer()

        with measurer:
            for _ in range(50):
                agent.process_batch(condensync_inputs_batch)
            measurer.add_items(50 * len(condensync_inputs_batch))

        # Should achieve high throughput
        assert measurer.items_per_second >= 100

    @pytest.mark.unit
    @pytest.mark.performance
    def test_processing_speed(
        self,
        agent: CondensyncAgent,
        condensync_input: CondensyncInput,
        performance_timer
    ):
        """Test processing speed."""
        timer = performance_timer()

        with timer:
            for _ in range(100):
                agent.process(condensync_input)

        # 100 processes in < 2 seconds
        assert timer.elapsed < 2.0


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.unit
    def test_handles_invalid_temperatures(self, agent: CondensyncAgent):
        """Test handling of invalid temperature readings."""
        reading = CondenserReading(
            timestamp=datetime.now(timezone.utc),
            condenser_id="COND-001",
            cw_inlet_temp_c=40.0,
            cw_outlet_temp_c=30.0,  # Outlet < inlet (invalid)
            cw_flow_m3_s=15.0,
            cw_velocity_m_s=2.1,
            vacuum_pressure_kpa_abs=5.0,
            saturation_temp_c=32.0,  # Below outlet (invalid)
            steam_flow_kg_s=180.0,
            hotwell_temp_c=32.0,
            hotwell_level_pct=50.0,
            air_ingress_scfm=2.0,
            dissolved_oxygen_ppb=5.0,
            unit_load_mw=480.0,
        )

        # Should not raise, should handle gracefully
        cf = agent.calculate_cleanliness_factor(
            reading,
            CondenserConfig(
                condenser_id="COND-001",
                plant_name="Test",
                unit_capacity_mw=500.0,
                tube_material=TubeMaterial.TITANIUM_GRADE_2,
                tube_od_mm=25.4,
                tube_thickness_mm=0.889,
                tube_length_m=12.0,
                num_tubes=18000,
                num_passes=2,
                shell_diameter_m=5.5,
                surface_area_m2=25000.0,
                design_pressure_kpa_abs=5.0,
                design_cw_flow_m3_s=15.0,
                design_cw_inlet_temp_c=25.0,
            )
        )

        # Should return default value
        assert cf == 0.85
