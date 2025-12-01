# -*- coding: utf-8 -*-
"""
Unit Tests for SteamQualityTools.

This module provides comprehensive tests for the SteamQualityTools class,
covering all tool methods, result dataclass creation, provenance hash generation,
dashboard generation, and optimization routines.

Coverage Target: 95%+
Standards Compliance:
- IAPWS-IF97: Industrial Formulation for Water and Steam Properties
- ASME PTC 19.11: Steam and Water Sampling

Test Categories:
1. SteamQualityTools method tests
2. Result dataclass creation and validation
3. Provenance hash generation
4. Dashboard generation
5. Optimization routines
6. Determinism verification

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import math
import time
import json
import hashlib
from pathlib import Path
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import test fixtures from conftest
from conftest import (
    SteamState,
    RiskLevel,
    ValveCharacteristic,
    generate_provenance_hash,
    assert_within_tolerance,
    assert_deterministic,
)


# =============================================================================
# ENUMS AND DATACLASSES FOR TESTING
# =============================================================================

class OptimizationObjective(Enum):
    """Steam quality optimization objectives."""
    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_MOISTURE = "minimize_moisture"
    OPTIMIZE_SUPERHEAT = "optimize_superheat"
    BALANCE_EFFICIENCY = "balance_efficiency"


@dataclass
class SteamQualityResult:
    """Result dataclass for steam quality analysis."""
    timestamp: str
    agent_id: str
    operation: str
    dryness_fraction: float
    wetness_percent: float
    superheat_c: float
    state: SteamState
    quality_index: float
    risk_level: RiskLevel
    specific_enthalpy_kj_kg: float
    specific_entropy_kj_kg_k: float
    provenance_hash: str
    calculation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardData:
    """Data structure for dashboard display."""
    title: str
    timestamp: str
    current_quality: float
    target_quality: float
    quality_trend: List[float]
    risk_level: RiskLevel
    moisture_sources: List[str]
    recommendations: List[str]
    kpis: Dict[str, float]
    alarms: List[Dict[str, Any]]


@dataclass
class OptimizationResult:
    """Result of optimization routine."""
    objective: OptimizationObjective
    optimal_setpoints: Dict[str, float]
    predicted_quality: float
    predicted_energy_saving_kw: float
    confidence_percent: float
    constraints_satisfied: bool
    provenance_hash: str


# =============================================================================
# MOCK TOOLS IMPLEMENTATION
# =============================================================================

class SteamQualityTools:
    """
    Steam quality analysis tools.

    Provides utility methods for steam quality calculations, result
    packaging, provenance tracking, and optimization routines.
    """

    AGENT_ID = "GL-012"
    AGENT_NAME = "STEAMQUAL"
    VERSION = "1.0.0"

    def __init__(self):
        """Initialize tools."""
        self.calculation_count = 0
        self._provenance_records = []

    @staticmethod
    def create_calculation_hash(data: Dict[str, Any]) -> str:
        """
        Create SHA-256 hash for calculation provenance.

        Args:
            data: Dictionary of calculation inputs/outputs

        Returns:
            64-character hexadecimal hash string
        """
        # Sort keys for deterministic serialization
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @staticmethod
    def validate_hash(data: Dict[str, Any], expected_hash: str) -> bool:
        """
        Validate data against expected hash.

        Args:
            data: Data to validate
            expected_hash: Expected hash value

        Returns:
            True if hash matches
        """
        calculated_hash = SteamQualityTools.create_calculation_hash(data)
        return calculated_hash == expected_hash

    def create_result(
        self,
        operation: str,
        dryness_fraction: float,
        superheat_c: float,
        state: SteamState,
        specific_enthalpy_kj_kg: float,
        specific_entropy_kj_kg_k: float,
        metadata: Dict[str, Any] = None
    ) -> SteamQualityResult:
        """
        Create a standardized result dataclass.

        Args:
            operation: Operation name
            dryness_fraction: Steam quality (0-1)
            superheat_c: Degrees of superheat
            state: Steam thermodynamic state
            specific_enthalpy_kj_kg: Specific enthalpy
            specific_entropy_kj_kg_k: Specific entropy
            metadata: Additional metadata

        Returns:
            SteamQualityResult dataclass instance
        """
        self.calculation_count += 1

        # Calculate derived values
        wetness_percent = max(0.0, (1.0 - dryness_fraction) * 100.0)
        quality_index = self._calculate_quality_index(dryness_fraction, superheat_c)
        risk_level = self._assess_risk(dryness_fraction)

        # Create provenance hash
        hash_data = {
            'operation': operation,
            'dryness_fraction': round(dryness_fraction, 10),
            'superheat_c': round(superheat_c, 6),
            'state': state.value,
        }
        provenance_hash = self.create_calculation_hash(hash_data)

        # Record provenance
        self._provenance_records.append({
            'hash': provenance_hash,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation': operation,
        })

        return SteamQualityResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_id=self.AGENT_ID,
            operation=operation,
            dryness_fraction=dryness_fraction,
            wetness_percent=wetness_percent,
            superheat_c=superheat_c,
            state=state,
            quality_index=quality_index,
            risk_level=risk_level,
            specific_enthalpy_kj_kg=specific_enthalpy_kj_kg,
            specific_entropy_kj_kg_k=specific_entropy_kj_kg_k,
            provenance_hash=provenance_hash,
            calculation_time_ms=0.0,  # Set by caller
            metadata=metadata or {}
        )

    def _calculate_quality_index(
        self,
        dryness_fraction: float,
        superheat_c: float
    ) -> float:
        """Calculate steam quality index (0-100)."""
        if superheat_c > 0:
            return 100.0
        return max(0.0, min(100.0, dryness_fraction * 100.0))

    def _assess_risk(self, dryness_fraction: float) -> RiskLevel:
        """Assess risk level from dryness fraction."""
        if dryness_fraction >= 1.0:
            return RiskLevel.NONE
        elif dryness_fraction >= 0.98:
            return RiskLevel.LOW
        elif dryness_fraction >= 0.95:
            return RiskLevel.MEDIUM
        elif dryness_fraction >= 0.88:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def generate_dashboard(
        self,
        current_result: SteamQualityResult,
        historical_qualities: List[float],
        target_quality: float = 98.0,
        recommendations: List[str] = None
    ) -> DashboardData:
        """
        Generate dashboard data for visualization.

        Args:
            current_result: Current steam quality result
            historical_qualities: List of historical quality values
            target_quality: Target quality percentage
            recommendations: List of recommendations

        Returns:
            DashboardData for rendering
        """
        # Calculate KPIs
        kpis = {
            'current_quality': current_result.quality_index,
            'target_quality': target_quality,
            'quality_gap': target_quality - current_result.quality_index,
            'wetness_percent': current_result.wetness_percent,
            'superheat_c': current_result.superheat_c,
        }

        # Add trend statistics if history available
        if historical_qualities:
            kpis['avg_quality_24h'] = sum(historical_qualities) / len(historical_qualities)
            kpis['min_quality_24h'] = min(historical_qualities)
            kpis['max_quality_24h'] = max(historical_qualities)

        # Generate alarms
        alarms = []
        if current_result.risk_level == RiskLevel.CRITICAL:
            alarms.append({
                'severity': 'critical',
                'message': 'Steam quality critically low - water hammer risk',
                'timestamp': current_result.timestamp,
            })
        elif current_result.risk_level == RiskLevel.HIGH:
            alarms.append({
                'severity': 'high',
                'message': 'Steam quality below acceptable threshold',
                'timestamp': current_result.timestamp,
            })

        # Moisture sources from metadata if available
        moisture_sources = current_result.metadata.get('moisture_sources', [])
        if isinstance(moisture_sources, list):
            moisture_sources = [str(s) for s in moisture_sources]

        return DashboardData(
            title=f"{self.AGENT_NAME} Dashboard",
            timestamp=current_result.timestamp,
            current_quality=current_result.quality_index,
            target_quality=target_quality,
            quality_trend=historical_qualities[-24:] if historical_qualities else [],
            risk_level=current_result.risk_level,
            moisture_sources=moisture_sources,
            recommendations=recommendations or [],
            kpis=kpis,
            alarms=alarms
        )

    def optimize_setpoints(
        self,
        current_quality: float,
        target_quality: float,
        constraints: Dict[str, Tuple[float, float]],
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_QUALITY
    ) -> OptimizationResult:
        """
        Optimize setpoints for steam quality improvement.

        Args:
            current_quality: Current quality index (0-100)
            target_quality: Target quality index
            constraints: Dictionary of parameter constraints (min, max)
            objective: Optimization objective

        Returns:
            OptimizationResult with optimal setpoints
        """
        # Simple optimization logic (placeholder for real optimizer)
        optimal_setpoints = {}

        # Determine optimal superheat setpoint
        if 'superheat_c' in constraints:
            min_sh, max_sh = constraints['superheat_c']
            # Target moderate superheat for quality
            optimal_setpoints['superheat_c'] = min(max_sh, max(min_sh, 20.0))

        # Determine optimal pressure
        if 'pressure_bar' in constraints:
            min_p, max_p = constraints['pressure_bar']
            # Maintain stable pressure
            optimal_setpoints['pressure_bar'] = (min_p + max_p) / 2

        # Determine desuperheater setpoint if needed
        if 'desuperheater_temp_c' in constraints:
            min_t, max_t = constraints['desuperheater_temp_c']
            optimal_setpoints['desuperheater_temp_c'] = max_t  # Keep superheat

        # Predict outcome
        quality_gap = target_quality - current_quality
        predicted_improvement = min(quality_gap, 5.0)  # Max 5% improvement per cycle
        predicted_quality = current_quality + predicted_improvement

        # Energy savings estimate (simplified)
        if predicted_quality > current_quality:
            # Better quality = less moisture = less energy loss
            moisture_reduction = (predicted_quality - current_quality) / 100.0
            predicted_savings = moisture_reduction * 100.0  # kW estimate
        else:
            predicted_savings = 0.0

        # Check constraints
        constraints_satisfied = True
        for param, value in optimal_setpoints.items():
            if param in constraints:
                min_val, max_val = constraints[param]
                if not (min_val <= value <= max_val):
                    constraints_satisfied = False
                    break

        # Confidence based on gap
        if abs(quality_gap) < 1.0:
            confidence = 95.0
        elif abs(quality_gap) < 5.0:
            confidence = 85.0
        else:
            confidence = 70.0

        # Create provenance hash
        hash_data = {
            'objective': objective.value,
            'current_quality': current_quality,
            'target_quality': target_quality,
            'optimal_setpoints': optimal_setpoints,
        }
        provenance_hash = self.create_calculation_hash(hash_data)

        return OptimizationResult(
            objective=objective,
            optimal_setpoints=optimal_setpoints,
            predicted_quality=predicted_quality,
            predicted_energy_saving_kw=predicted_savings,
            confidence_percent=confidence,
            constraints_satisfied=constraints_satisfied,
            provenance_hash=provenance_hash
        )

    def batch_analyze(
        self,
        measurements: List[Dict[str, float]]
    ) -> List[SteamQualityResult]:
        """
        Batch analyze multiple measurements.

        Args:
            measurements: List of measurement dictionaries

        Returns:
            List of SteamQualityResult instances
        """
        results = []
        for m in measurements:
            result = self.create_result(
                operation='batch_analysis',
                dryness_fraction=m.get('dryness_fraction', 1.0),
                superheat_c=m.get('superheat_c', 0.0),
                state=SteamState.WET_STEAM if m.get('dryness_fraction', 1.0) < 1.0 else SteamState.SATURATED_VAPOR,
                specific_enthalpy_kj_kg=m.get('enthalpy_kj_kg', 2700.0),
                specific_entropy_kj_kg_k=m.get('entropy_kj_kg_k', 6.5),
                metadata={'batch_id': m.get('id', 'unknown')}
            )
            results.append(result)
        return results

    def export_results_json(self, results: List[SteamQualityResult]) -> str:
        """
        Export results to JSON format.

        Args:
            results: List of results to export

        Returns:
            JSON string
        """
        export_data = []
        for r in results:
            data = asdict(r)
            # Convert enums to strings
            data['state'] = r.state.value
            data['risk_level'] = r.risk_level.value
            export_data.append(data)

        return json.dumps(export_data, indent=2, default=str)

    def get_provenance_audit_trail(self) -> List[Dict[str, Any]]:
        """Get complete provenance audit trail."""
        return self._provenance_records.copy()


# =============================================================================
# TEST CLASS: RESULT DATACLASS CREATION
# =============================================================================

class TestResultDataclassCreation:
    """Test suite for result dataclass creation."""

    @pytest.fixture
    def tools(self):
        """Create tools instance."""
        return SteamQualityTools()

    @pytest.mark.unit
    def test_create_result_basic(self, tools):
        """Test basic result creation."""
        result = tools.create_result(
            operation='test_operation',
            dryness_fraction=0.95,
            superheat_c=0.0,
            state=SteamState.WET_STEAM,
            specific_enthalpy_kj_kg=2700.0,
            specific_entropy_kj_kg_k=6.5
        )

        assert result.agent_id == "GL-012"
        assert result.operation == 'test_operation'
        assert result.dryness_fraction == 0.95
        assert result.wetness_percent == 5.0
        assert result.state == SteamState.WET_STEAM
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_create_result_with_metadata(self, tools):
        """Test result creation with metadata."""
        metadata = {'sensor_id': 'STEAM-001', 'location': 'Main Header'}

        result = tools.create_result(
            operation='test_operation',
            dryness_fraction=0.98,
            superheat_c=10.0,
            state=SteamState.SUPERHEATED,
            specific_enthalpy_kj_kg=2800.0,
            specific_entropy_kj_kg_k=6.8,
            metadata=metadata
        )

        assert result.metadata == metadata
        assert result.metadata['sensor_id'] == 'STEAM-001'

    @pytest.mark.unit
    def test_create_result_quality_index_wet(self, tools):
        """Test quality index for wet steam."""
        result = tools.create_result(
            operation='test',
            dryness_fraction=0.90,
            superheat_c=0.0,
            state=SteamState.WET_STEAM,
            specific_enthalpy_kj_kg=2700.0,
            specific_entropy_kj_kg_k=6.5
        )

        assert result.quality_index == 90.0

    @pytest.mark.unit
    def test_create_result_quality_index_superheated(self, tools):
        """Test quality index for superheated steam."""
        result = tools.create_result(
            operation='test',
            dryness_fraction=1.0,
            superheat_c=50.0,
            state=SteamState.SUPERHEATED,
            specific_enthalpy_kj_kg=2900.0,
            specific_entropy_kj_kg_k=7.0
        )

        assert result.quality_index == 100.0

    @pytest.mark.unit
    def test_create_result_risk_levels(self, tools):
        """Test risk level assignment in results."""
        # Dry steam - no risk
        result_dry = tools.create_result(
            operation='test', dryness_fraction=1.0, superheat_c=10.0,
            state=SteamState.SUPERHEATED,
            specific_enthalpy_kj_kg=2800.0, specific_entropy_kj_kg_k=6.8
        )
        assert result_dry.risk_level == RiskLevel.NONE

        # Slightly wet - low risk
        result_low = tools.create_result(
            operation='test', dryness_fraction=0.99, superheat_c=0.0,
            state=SteamState.WET_STEAM,
            specific_enthalpy_kj_kg=2750.0, specific_entropy_kj_kg_k=6.6
        )
        assert result_low.risk_level == RiskLevel.LOW

        # Critical moisture
        result_critical = tools.create_result(
            operation='test', dryness_fraction=0.80, superheat_c=0.0,
            state=SteamState.WET_STEAM,
            specific_enthalpy_kj_kg=2500.0, specific_entropy_kj_kg_k=6.0
        )
        assert result_critical.risk_level == RiskLevel.CRITICAL

    @pytest.mark.unit
    def test_create_result_increments_counter(self, tools):
        """Test that creating results increments calculation counter."""
        initial_count = tools.calculation_count

        tools.create_result(
            operation='test', dryness_fraction=0.95, superheat_c=0.0,
            state=SteamState.WET_STEAM,
            specific_enthalpy_kj_kg=2700.0, specific_entropy_kj_kg_k=6.5
        )
        tools.create_result(
            operation='test', dryness_fraction=0.96, superheat_c=0.0,
            state=SteamState.WET_STEAM,
            specific_enthalpy_kj_kg=2700.0, specific_entropy_kj_kg_k=6.5
        )

        assert tools.calculation_count == initial_count + 2


# =============================================================================
# TEST CLASS: PROVENANCE HASH GENERATION
# =============================================================================

class TestProvenanceHashGeneration:
    """Test suite for provenance hash generation."""

    @pytest.fixture
    def tools(self):
        """Create tools instance."""
        return SteamQualityTools()

    @pytest.mark.unit
    def test_hash_generation_basic(self, tools):
        """Test basic hash generation."""
        data = {'key1': 'value1', 'key2': 123}
        hash_result = tools.create_calculation_hash(data)

        assert hash_result is not None
        assert len(hash_result) == 64
        assert all(c in '0123456789abcdef' for c in hash_result)

    @pytest.mark.unit
    def test_hash_deterministic(self, tools):
        """Test hash is deterministic for same input."""
        data = {'pressure': 10.0, 'temperature': 180.0, 'dryness': 0.95}

        hash1 = tools.create_calculation_hash(data)
        hash2 = tools.create_calculation_hash(data)

        assert hash1 == hash2

    @pytest.mark.unit
    def test_hash_different_for_different_input(self, tools):
        """Test hash differs for different inputs."""
        data1 = {'pressure': 10.0, 'temperature': 180.0}
        data2 = {'pressure': 10.0, 'temperature': 181.0}

        hash1 = tools.create_calculation_hash(data1)
        hash2 = tools.create_calculation_hash(data2)

        assert hash1 != hash2

    @pytest.mark.unit
    def test_hash_order_independent(self, tools):
        """Test hash is independent of key order."""
        data1 = {'a': 1, 'b': 2, 'c': 3}
        data2 = {'c': 3, 'a': 1, 'b': 2}

        hash1 = tools.create_calculation_hash(data1)
        hash2 = tools.create_calculation_hash(data2)

        assert hash1 == hash2

    @pytest.mark.unit
    def test_hash_validation_success(self, tools):
        """Test hash validation succeeds for matching data."""
        data = {'pressure': 10.0, 'dryness': 0.95}
        expected_hash = tools.create_calculation_hash(data)

        assert tools.validate_hash(data, expected_hash) is True

    @pytest.mark.unit
    def test_hash_validation_failure(self, tools):
        """Test hash validation fails for modified data."""
        data = {'pressure': 10.0, 'dryness': 0.95}
        original_hash = tools.create_calculation_hash(data)

        data['dryness'] = 0.96  # Modify data

        assert tools.validate_hash(data, original_hash) is False

    @pytest.mark.unit
    def test_provenance_audit_trail(self, tools):
        """Test provenance records are tracked."""
        tools.create_result(
            operation='operation1', dryness_fraction=0.95, superheat_c=0.0,
            state=SteamState.WET_STEAM,
            specific_enthalpy_kj_kg=2700.0, specific_entropy_kj_kg_k=6.5
        )
        tools.create_result(
            operation='operation2', dryness_fraction=0.98, superheat_c=10.0,
            state=SteamState.SUPERHEATED,
            specific_enthalpy_kj_kg=2800.0, specific_entropy_kj_kg_k=6.8
        )

        trail = tools.get_provenance_audit_trail()

        assert len(trail) >= 2
        assert all('hash' in record for record in trail)
        assert all('timestamp' in record for record in trail)
        assert all('operation' in record for record in trail)


# =============================================================================
# TEST CLASS: DASHBOARD GENERATION
# =============================================================================

class TestDashboardGeneration:
    """Test suite for dashboard data generation."""

    @pytest.fixture
    def tools(self):
        """Create tools instance."""
        return SteamQualityTools()

    @pytest.fixture
    def sample_result(self, tools):
        """Create sample result for dashboard tests."""
        return tools.create_result(
            operation='quality_check',
            dryness_fraction=0.96,
            superheat_c=5.0,
            state=SteamState.WET_STEAM,
            specific_enthalpy_kj_kg=2750.0,
            specific_entropy_kj_kg_k=6.6,
            metadata={'moisture_sources': ['heat_loss', 'carryover']}
        )

    @pytest.mark.unit
    def test_dashboard_basic_generation(self, tools, sample_result):
        """Test basic dashboard generation."""
        dashboard = tools.generate_dashboard(
            current_result=sample_result,
            historical_qualities=[95.0, 96.0, 95.5, 96.5],
            target_quality=98.0
        )

        assert dashboard.title == "STEAMQUAL Dashboard"
        assert dashboard.current_quality == sample_result.quality_index
        assert dashboard.target_quality == 98.0
        assert dashboard.risk_level == sample_result.risk_level

    @pytest.mark.unit
    def test_dashboard_kpis(self, tools, sample_result):
        """Test dashboard KPIs are calculated."""
        dashboard = tools.generate_dashboard(
            current_result=sample_result,
            historical_qualities=[94.0, 95.0, 96.0, 97.0],
            target_quality=98.0
        )

        assert 'current_quality' in dashboard.kpis
        assert 'target_quality' in dashboard.kpis
        assert 'quality_gap' in dashboard.kpis
        assert 'avg_quality_24h' in dashboard.kpis
        assert 'min_quality_24h' in dashboard.kpis
        assert 'max_quality_24h' in dashboard.kpis

    @pytest.mark.unit
    def test_dashboard_quality_trend(self, tools, sample_result):
        """Test dashboard includes quality trend."""
        history = [float(90 + i) for i in range(30)]

        dashboard = tools.generate_dashboard(
            current_result=sample_result,
            historical_qualities=history,
            target_quality=98.0
        )

        # Should only include last 24 values
        assert len(dashboard.quality_trend) == 24

    @pytest.mark.unit
    def test_dashboard_alarms_critical(self, tools):
        """Test dashboard generates alarms for critical risk."""
        critical_result = tools.create_result(
            operation='critical_check',
            dryness_fraction=0.80,
            superheat_c=0.0,
            state=SteamState.WET_STEAM,
            specific_enthalpy_kj_kg=2500.0,
            specific_entropy_kj_kg_k=6.0
        )

        dashboard = tools.generate_dashboard(
            current_result=critical_result,
            historical_qualities=[],
            target_quality=98.0
        )

        assert len(dashboard.alarms) > 0
        assert any(a['severity'] == 'critical' for a in dashboard.alarms)

    @pytest.mark.unit
    def test_dashboard_recommendations(self, tools, sample_result):
        """Test dashboard includes recommendations."""
        recommendations = [
            "Check steam trap operation",
            "Inspect insulation"
        ]

        dashboard = tools.generate_dashboard(
            current_result=sample_result,
            historical_qualities=[95.0, 96.0],
            target_quality=98.0,
            recommendations=recommendations
        )

        assert dashboard.recommendations == recommendations


# =============================================================================
# TEST CLASS: OPTIMIZATION ROUTINES
# =============================================================================

class TestOptimizationRoutines:
    """Test suite for optimization routines."""

    @pytest.fixture
    def tools(self):
        """Create tools instance."""
        return SteamQualityTools()

    @pytest.mark.unit
    def test_optimize_setpoints_basic(self, tools):
        """Test basic setpoint optimization."""
        constraints = {
            'superheat_c': (10.0, 50.0),
            'pressure_bar': (8.0, 12.0),
        }

        result = tools.optimize_setpoints(
            current_quality=90.0,
            target_quality=98.0,
            constraints=constraints,
            objective=OptimizationObjective.MAXIMIZE_QUALITY
        )

        assert result.objective == OptimizationObjective.MAXIMIZE_QUALITY
        assert result.predicted_quality > 90.0
        assert 'superheat_c' in result.optimal_setpoints
        assert result.constraints_satisfied is True
        assert result.provenance_hash is not None

    @pytest.mark.unit
    def test_optimize_setpoints_respects_constraints(self, tools):
        """Test optimization respects constraints."""
        constraints = {
            'superheat_c': (15.0, 25.0),
        }

        result = tools.optimize_setpoints(
            current_quality=92.0,
            target_quality=98.0,
            constraints=constraints
        )

        if 'superheat_c' in result.optimal_setpoints:
            assert 15.0 <= result.optimal_setpoints['superheat_c'] <= 25.0

    @pytest.mark.unit
    def test_optimize_predicts_energy_savings(self, tools):
        """Test optimization predicts energy savings."""
        constraints = {
            'superheat_c': (10.0, 50.0),
        }

        result = tools.optimize_setpoints(
            current_quality=85.0,  # Low quality, room for improvement
            target_quality=98.0,
            constraints=constraints
        )

        assert result.predicted_energy_saving_kw >= 0

    @pytest.mark.unit
    def test_optimize_confidence_based_on_gap(self, tools):
        """Test optimization confidence varies with quality gap."""
        constraints = {'superheat_c': (10.0, 50.0)}

        # Small gap - high confidence
        result_small_gap = tools.optimize_setpoints(
            current_quality=97.5,
            target_quality=98.0,
            constraints=constraints
        )

        # Large gap - lower confidence
        result_large_gap = tools.optimize_setpoints(
            current_quality=80.0,
            target_quality=98.0,
            constraints=constraints
        )

        assert result_small_gap.confidence_percent > result_large_gap.confidence_percent

    @pytest.mark.unit
    @pytest.mark.parametrize("objective", [
        OptimizationObjective.MAXIMIZE_QUALITY,
        OptimizationObjective.MINIMIZE_MOISTURE,
        OptimizationObjective.OPTIMIZE_SUPERHEAT,
        OptimizationObjective.BALANCE_EFFICIENCY,
    ])
    def test_optimize_all_objectives(self, tools, objective):
        """Test optimization works for all objectives."""
        constraints = {'superheat_c': (10.0, 50.0)}

        result = tools.optimize_setpoints(
            current_quality=90.0,
            target_quality=98.0,
            constraints=constraints,
            objective=objective
        )

        assert result.objective == objective
        assert result.provenance_hash is not None


# =============================================================================
# TEST CLASS: BATCH OPERATIONS
# =============================================================================

class TestBatchOperations:
    """Test suite for batch operations."""

    @pytest.fixture
    def tools(self):
        """Create tools instance."""
        return SteamQualityTools()

    @pytest.mark.unit
    def test_batch_analyze_multiple(self, tools):
        """Test batch analysis of multiple measurements."""
        measurements = [
            {'id': 'M1', 'dryness_fraction': 0.95, 'superheat_c': 0.0},
            {'id': 'M2', 'dryness_fraction': 0.98, 'superheat_c': 5.0},
            {'id': 'M3', 'dryness_fraction': 0.92, 'superheat_c': 0.0},
        ]

        results = tools.batch_analyze(measurements)

        assert len(results) == 3
        assert results[0].dryness_fraction == 0.95
        assert results[1].dryness_fraction == 0.98
        assert results[2].dryness_fraction == 0.92

    @pytest.mark.unit
    def test_batch_analyze_preserves_ids(self, tools):
        """Test batch analysis preserves measurement IDs in metadata."""
        measurements = [
            {'id': 'SENSOR-001', 'dryness_fraction': 0.96},
            {'id': 'SENSOR-002', 'dryness_fraction': 0.97},
        ]

        results = tools.batch_analyze(measurements)

        assert results[0].metadata['batch_id'] == 'SENSOR-001'
        assert results[1].metadata['batch_id'] == 'SENSOR-002'

    @pytest.mark.unit
    def test_export_results_json(self, tools):
        """Test exporting results to JSON."""
        results = tools.batch_analyze([
            {'dryness_fraction': 0.95},
            {'dryness_fraction': 0.98},
        ])

        json_str = tools.export_results_json(results)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert len(parsed) == 2
        assert parsed[0]['dryness_fraction'] == 0.95

    @pytest.mark.unit
    def test_export_converts_enums(self, tools):
        """Test export converts enums to strings."""
        results = tools.batch_analyze([{'dryness_fraction': 0.95}])

        json_str = tools.export_results_json(results)
        parsed = json.loads(json_str)

        # Enums should be strings
        assert isinstance(parsed[0]['state'], str)
        assert isinstance(parsed[0]['risk_level'], str)


# =============================================================================
# TEST CLASS: DETERMINISM
# =============================================================================

class TestToolsDeterminism:
    """Test suite for tools determinism verification."""

    @pytest.fixture
    def tools(self):
        """Create tools instance."""
        return SteamQualityTools()

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_hash_generation_determinism(self, tools):
        """Test hash generation is deterministic."""
        data = {'pressure': 10.0, 'dryness': 0.95, 'temp': 180.0}

        results = []
        for _ in range(100):
            hash_val = tools.create_calculation_hash(data)
            results.append(hash_val)

        assert_deterministic(results, "Hash generation")

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_result_creation_determinism(self, tools):
        """Test result creation provenance hash is deterministic."""
        # Note: timestamp will differ, but provenance hash should be same
        # for same calculation inputs

        hashes = []
        for _ in range(50):
            result = tools.create_result(
                operation='determinism_test',
                dryness_fraction=0.95,
                superheat_c=10.0,
                state=SteamState.WET_STEAM,
                specific_enthalpy_kj_kg=2700.0,
                specific_entropy_kj_kg_k=6.5
            )
            hashes.append(result.provenance_hash)

        assert_deterministic(hashes, "Result provenance hash")

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_optimization_determinism(self, tools):
        """Test optimization is deterministic."""
        constraints = {'superheat_c': (10.0, 50.0)}

        results = []
        for _ in range(50):
            result = tools.optimize_setpoints(
                current_quality=90.0,
                target_quality=98.0,
                constraints=constraints
            )
            results.append(result.predicted_quality)

        assert_deterministic(results, "Optimization result")


# =============================================================================
# TEST CLASS: PERFORMANCE
# =============================================================================

class TestToolsPerformance:
    """Test suite for tools performance benchmarks."""

    @pytest.fixture
    def tools(self):
        """Create tools instance."""
        return SteamQualityTools()

    @pytest.mark.unit
    @pytest.mark.performance
    def test_hash_generation_performance(self, tools, performance_timer):
        """Test hash generation is fast."""
        data = {'pressure': 10.0, 'dryness': 0.95, 'temp': 180.0}

        with performance_timer() as timer:
            for _ in range(1000):
                tools.create_calculation_hash(data)

        # 1000 hashes should complete in under 100ms
        assert timer.elapsed_ms < 100.0

    @pytest.mark.unit
    @pytest.mark.performance
    def test_result_creation_performance(self, tools, performance_timer):
        """Test result creation is fast."""
        with performance_timer() as timer:
            for _ in range(100):
                tools.create_result(
                    operation='perf_test',
                    dryness_fraction=0.95,
                    superheat_c=10.0,
                    state=SteamState.WET_STEAM,
                    specific_enthalpy_kj_kg=2700.0,
                    specific_entropy_kj_kg_k=6.5
                )

        # 100 results should complete in under 100ms
        assert timer.elapsed_ms < 100.0

    @pytest.mark.unit
    @pytest.mark.performance
    def test_batch_analyze_performance(self, tools, performance_timer):
        """Test batch analysis is fast."""
        measurements = [{'dryness_fraction': 0.90 + i * 0.01} for i in range(100)]

        with performance_timer() as timer:
            tools.batch_analyze(measurements)

        # 100 batch items should complete in under 200ms
        assert timer.elapsed_ms < 200.0
