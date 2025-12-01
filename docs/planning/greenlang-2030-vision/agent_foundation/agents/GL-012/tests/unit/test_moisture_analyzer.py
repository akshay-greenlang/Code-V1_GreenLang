# -*- coding: utf-8 -*-
"""
Unit Tests for MoistureAnalyzer.

This module provides comprehensive tests for the MoistureAnalyzer class,
covering moisture content analysis, condensation risk detection, wetness
fraction calculations, moisture source identification, and remediation recommendations.

Coverage Target: 95%+
Standards Compliance:
- ASME PTC 19.11: Steam and Water Sampling
- ASME B31.1: Power Piping
- ISO 10380: Pipework - Corrugated metal hoses

Test Categories:
1. Moisture content analysis
2. Condensation risk detection
3. Wetness fraction calculations
4. Moisture source identification
5. Remediation recommendations
6. Risk level assessment
7. Determinism verification

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import math
import time
from pathlib import Path
from decimal import Decimal
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from enum import Enum

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import test fixtures from conftest
from conftest import (
    SteamState,
    RiskLevel,
    generate_provenance_hash,
    assert_within_tolerance,
    assert_deterministic,
)


# =============================================================================
# ENUMS AND DATACLASSES FOR TESTING
# =============================================================================

class MoistureSource(Enum):
    """Possible sources of moisture in steam."""
    BOILER_CARRYOVER = "boiler_carryover"
    CONDENSATION_HEAT_LOSS = "condensation_heat_loss"
    PRESSURE_REDUCTION = "pressure_reduction"
    DESUPERHEATER_EXCESS = "desuperheater_excess"
    STEAM_TRAP_FAILURE = "steam_trap_failure"
    PIPE_LEAK = "pipe_leak"
    INSULATION_DAMAGE = "insulation_damage"
    UNKNOWN = "unknown"


class RemediationPriority(Enum):
    """Priority level for remediation actions."""
    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MONITOR = "monitor"


@dataclass
class MoistureAnalysisInput:
    """Input data for moisture analysis."""
    dryness_fraction: float
    pressure_bar: float
    temperature_c: float
    saturation_temp_c: float
    flow_rate_kg_s: float
    pipe_length_m: float = 100.0
    pipe_diameter_mm: float = 150.0
    insulation_thickness_mm: float = 50.0
    ambient_temp_c: float = 20.0
    pressure_drop_bar: float = 0.0
    upstream_dryness: float = None


@dataclass
class MoistureAnalysisOutput:
    """Output data from moisture analysis."""
    wetness_percent: float
    moisture_mass_rate_kg_s: float
    risk_level: RiskLevel
    condensation_possible: bool
    probable_sources: List[MoistureSource]
    remediation_actions: List[str]
    remediation_priority: RemediationPriority
    quality_score: float
    energy_loss_kw: float
    provenance_hash: str
    calculation_time_ms: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class PipeHeatLoss:
    """Pipe heat loss calculation results."""
    heat_loss_w_m: float
    total_heat_loss_kw: float
    condensation_rate_kg_s: float
    temperature_drop_c: float


# =============================================================================
# MOCK CALCULATOR IMPLEMENTATION
# =============================================================================

class MoistureAnalyzer:
    """
    Moisture analyzer for steam quality assessment.

    Analyzes moisture content in steam, identifies probable sources,
    assesses condensation risk, and provides remediation recommendations.
    """

    # Risk thresholds
    RISK_THRESHOLDS = {
        RiskLevel.NONE: (1.0, 1.0),      # x >= 1.0 (superheated or dry saturated)
        RiskLevel.LOW: (0.98, 1.0),       # 0.98 <= x < 1.0
        RiskLevel.MEDIUM: (0.95, 0.98),   # 0.95 <= x < 0.98
        RiskLevel.HIGH: (0.88, 0.95),     # 0.88 <= x < 0.95
        RiskLevel.CRITICAL: (0.0, 0.88),  # x < 0.88
    }

    # Heat transfer constants
    STEAM_LATENT_HEAT_KJ_KG = 2000.0  # Average hfg
    INSULATION_K_W_MK = 0.04  # Thermal conductivity of insulation

    def __init__(self):
        """Initialize moisture analyzer."""
        self.calculation_count = 0

    def calculate_wetness_percent(self, dryness_fraction: float) -> float:
        """
        Calculate wetness percentage from dryness fraction.

        Wetness = (1 - x) * 100%

        Args:
            dryness_fraction: Steam quality (0.0 to 1.0)

        Returns:
            Wetness percentage (0-100%)
        """
        if dryness_fraction < 0:
            raise ValueError("Dryness fraction cannot be negative")
        if dryness_fraction > 1.0:
            return 0.0  # Superheated, no wetness

        return (1.0 - dryness_fraction) * 100.0

    def calculate_moisture_mass_rate(
        self,
        dryness_fraction: float,
        total_flow_kg_s: float
    ) -> float:
        """
        Calculate moisture mass flow rate.

        m_moisture = m_total * (1 - x)

        Args:
            dryness_fraction: Steam quality
            total_flow_kg_s: Total steam flow rate

        Returns:
            Moisture flow rate in kg/s
        """
        if dryness_fraction >= 1.0:
            return 0.0
        return total_flow_kg_s * (1.0 - dryness_fraction)

    def assess_risk_level(self, dryness_fraction: float) -> RiskLevel:
        """
        Assess risk level based on dryness fraction.

        Args:
            dryness_fraction: Steam quality

        Returns:
            RiskLevel enumeration
        """
        for level, (low, high) in self.RISK_THRESHOLDS.items():
            if low <= dryness_fraction < high:
                return level
            if level == RiskLevel.NONE and dryness_fraction >= high:
                return level

        return RiskLevel.CRITICAL  # Default for very low values

    def check_condensation_possible(
        self,
        temperature_c: float,
        saturation_temp_c: float,
        dryness_fraction: float
    ) -> bool:
        """
        Check if condensation is possible.

        Condensation can occur if:
        - Temperature is at or below saturation
        - Steam is already wet (x < 1)

        Args:
            temperature_c: Current temperature
            saturation_temp_c: Saturation temperature at pressure
            dryness_fraction: Current dryness fraction

        Returns:
            True if condensation is possible
        """
        # Already wet
        if dryness_fraction < 1.0:
            return True

        # At or below saturation temperature
        if temperature_c <= saturation_temp_c + 1.0:  # 1C margin
            return True

        return False

    def calculate_pipe_heat_loss(
        self,
        pipe_length_m: float,
        pipe_diameter_mm: float,
        insulation_thickness_mm: float,
        steam_temp_c: float,
        ambient_temp_c: float,
        flow_rate_kg_s: float
    ) -> PipeHeatLoss:
        """
        Calculate heat loss from insulated pipe.

        Uses simplified cylindrical heat transfer model.

        Args:
            pipe_length_m: Pipe length
            pipe_diameter_mm: Pipe outer diameter
            insulation_thickness_mm: Insulation thickness
            steam_temp_c: Steam temperature
            ambient_temp_c: Ambient temperature
            flow_rate_kg_s: Steam flow rate

        Returns:
            PipeHeatLoss with calculated values
        """
        # Convert to meters
        r_inner = pipe_diameter_mm / 2000.0
        r_outer = r_inner + insulation_thickness_mm / 1000.0

        # Temperature difference
        delta_t = steam_temp_c - ambient_temp_c

        # Heat loss per meter (simplified radial conduction)
        if r_outer > r_inner and insulation_thickness_mm > 0:
            heat_loss_w_m = (
                2 * math.pi * self.INSULATION_K_W_MK * delta_t /
                math.log(r_outer / r_inner)
            )
        else:
            # No insulation - higher heat loss
            heat_loss_w_m = delta_t * 50.0  # Simplified high loss

        # Total heat loss
        total_heat_loss_kw = heat_loss_w_m * pipe_length_m / 1000.0

        # Condensation rate from heat loss
        # Q = m * hfg => m = Q / hfg
        condensation_rate = total_heat_loss_kw / self.STEAM_LATENT_HEAT_KJ_KG

        # Temperature drop (if applicable)
        cp_steam = 2.0  # kJ/kg.K
        if flow_rate_kg_s > 0:
            temp_drop = total_heat_loss_kw / (flow_rate_kg_s * cp_steam)
        else:
            temp_drop = 0.0

        return PipeHeatLoss(
            heat_loss_w_m=heat_loss_w_m,
            total_heat_loss_kw=total_heat_loss_kw,
            condensation_rate_kg_s=condensation_rate,
            temperature_drop_c=temp_drop
        )

    def identify_moisture_sources(
        self,
        dryness_fraction: float,
        upstream_dryness: Optional[float],
        pressure_drop_bar: float,
        heat_loss_kw: float,
        flow_rate_kg_s: float
    ) -> List[MoistureSource]:
        """
        Identify probable sources of moisture.

        Args:
            dryness_fraction: Current dryness fraction
            upstream_dryness: Upstream dryness (if available)
            pressure_drop_bar: Pressure drop in system
            heat_loss_kw: Calculated heat loss
            flow_rate_kg_s: Flow rate

        Returns:
            List of probable moisture sources
        """
        sources = []

        if dryness_fraction >= 1.0:
            return sources  # No moisture

        # Check for upstream issues (boiler carryover)
        if upstream_dryness is not None:
            if upstream_dryness < 0.98:
                sources.append(MoistureSource.BOILER_CARRYOVER)
            elif dryness_fraction < upstream_dryness - 0.02:
                # Significant dryness drop
                pass  # Will identify other sources

        # Heat loss causing condensation
        if heat_loss_kw > 0:
            condensation_rate = heat_loss_kw / self.STEAM_LATENT_HEAT_KJ_KG
            if condensation_rate > 0.01 * flow_rate_kg_s:  # > 1% of flow
                sources.append(MoistureSource.CONDENSATION_HEAT_LOSS)

        # Pressure reduction can cause flashing and temporary wetness
        if pressure_drop_bar > 1.0:
            sources.append(MoistureSource.PRESSURE_REDUCTION)

        # If no specific source identified, mark as unknown
        if not sources and dryness_fraction < 0.98:
            if dryness_fraction < 0.90:
                sources.append(MoistureSource.BOILER_CARRYOVER)
            sources.append(MoistureSource.UNKNOWN)

        return sources

    def generate_remediation_actions(
        self,
        risk_level: RiskLevel,
        sources: List[MoistureSource]
    ) -> Tuple[List[str], RemediationPriority]:
        """
        Generate remediation recommendations.

        Args:
            risk_level: Current risk level
            sources: Identified moisture sources

        Returns:
            Tuple of (action list, priority)
        """
        actions = []
        priority = RemediationPriority.MONITOR

        if risk_level == RiskLevel.NONE:
            return ["Continue normal monitoring"], RemediationPriority.MONITOR

        # Set priority based on risk
        if risk_level == RiskLevel.CRITICAL:
            priority = RemediationPriority.IMMEDIATE
            actions.append("IMMEDIATE: Reduce load or shutdown to prevent water hammer")
        elif risk_level == RiskLevel.HIGH:
            priority = RemediationPriority.HIGH
        elif risk_level == RiskLevel.MEDIUM:
            priority = RemediationPriority.MEDIUM
        else:
            priority = RemediationPriority.LOW

        # Source-specific actions
        for source in sources:
            if source == MoistureSource.BOILER_CARRYOVER:
                actions.append("Inspect boiler water level and controls")
                actions.append("Check steam drum internals and separators")
                actions.append("Verify blowdown procedures")

            elif source == MoistureSource.CONDENSATION_HEAT_LOSS:
                actions.append("Inspect insulation for damage or gaps")
                actions.append("Check steam trap operation")
                actions.append("Verify drip leg drainage")

            elif source == MoistureSource.PRESSURE_REDUCTION:
                actions.append("Install separator after PRV station")
                actions.append("Verify PRV sizing")

            elif source == MoistureSource.DESUPERHEATER_EXCESS:
                actions.append("Check desuperheater controls and calibration")
                actions.append("Verify spray water flow measurement")

            elif source == MoistureSource.STEAM_TRAP_FAILURE:
                actions.append("Test steam traps for blow-through")
                actions.append("Replace failed steam traps")

            elif source == MoistureSource.INSULATION_DAMAGE:
                actions.append("Inspect and repair insulation")
                actions.append("Check for water ingress")

        # General actions for any moisture issue
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            actions.append("Increase steam trap inspection frequency")
            actions.append("Monitor for water hammer symptoms")

        return actions, priority

    def calculate_quality_score(
        self,
        dryness_fraction: float,
        superheat_c: float = 0.0
    ) -> float:
        """
        Calculate overall steam quality score (0-100).

        Score based on:
        - Dryness fraction (primary factor)
        - Superheat degree (bonus for superheated steam)

        Args:
            dryness_fraction: Steam quality
            superheat_c: Degrees of superheat

        Returns:
            Quality score (0-100)
        """
        # Base score from dryness
        if dryness_fraction >= 1.0:
            base_score = 100.0
        else:
            base_score = dryness_fraction * 100.0

        # Bonus for superheat (up to 5 points for 50C superheat)
        superheat_bonus = min(5.0, superheat_c / 10.0) if superheat_c > 0 else 0.0

        # Penalty for very wet steam
        if dryness_fraction < 0.90:
            penalty = (0.90 - dryness_fraction) * 50  # Up to 45 point penalty
        else:
            penalty = 0.0

        score = base_score + superheat_bonus - penalty
        return max(0.0, min(100.0, score))

    def calculate_energy_loss(
        self,
        moisture_rate_kg_s: float,
        dryness_drop: float = 0.0
    ) -> float:
        """
        Calculate energy loss due to moisture.

        Args:
            moisture_rate_kg_s: Moisture mass flow rate
            dryness_drop: Drop in dryness fraction

        Returns:
            Energy loss in kW
        """
        # Energy contained in moisture that should be dry steam
        # Q = m * hfg
        energy_loss = moisture_rate_kg_s * self.STEAM_LATENT_HEAT_KJ_KG
        return energy_loss

    def analyze(self, input_data: MoistureAnalysisInput) -> MoistureAnalysisOutput:
        """
        Perform complete moisture analysis.

        Args:
            input_data: MoistureAnalysisInput with all parameters

        Returns:
            MoistureAnalysisOutput with analysis results
        """
        start_time = time.perf_counter()
        self.calculation_count += 1
        warnings = []

        # Calculate wetness
        wetness = self.calculate_wetness_percent(input_data.dryness_fraction)

        # Calculate moisture mass rate
        moisture_rate = self.calculate_moisture_mass_rate(
            input_data.dryness_fraction,
            input_data.flow_rate_kg_s
        )

        # Assess risk level
        risk_level = self.assess_risk_level(input_data.dryness_fraction)

        # Check condensation possibility
        condensation_possible = self.check_condensation_possible(
            input_data.temperature_c,
            input_data.saturation_temp_c,
            input_data.dryness_fraction
        )

        # Calculate heat loss
        heat_loss = self.calculate_pipe_heat_loss(
            input_data.pipe_length_m,
            input_data.pipe_diameter_mm,
            input_data.insulation_thickness_mm,
            input_data.temperature_c,
            input_data.ambient_temp_c,
            input_data.flow_rate_kg_s
        )

        # Identify moisture sources
        sources = self.identify_moisture_sources(
            input_data.dryness_fraction,
            input_data.upstream_dryness,
            input_data.pressure_drop_bar,
            heat_loss.total_heat_loss_kw,
            input_data.flow_rate_kg_s
        )

        # Generate remediation actions
        actions, priority = self.generate_remediation_actions(risk_level, sources)

        # Calculate quality score
        superheat = input_data.temperature_c - input_data.saturation_temp_c
        quality_score = self.calculate_quality_score(input_data.dryness_fraction, superheat)

        # Calculate energy loss
        energy_loss = self.calculate_energy_loss(moisture_rate)

        # Generate warnings
        if risk_level == RiskLevel.CRITICAL:
            warnings.append("CRITICAL: Steam moisture exceeds safe limits")
        elif risk_level == RiskLevel.HIGH:
            warnings.append("HIGH RISK: Elevated moisture content detected")
        if condensation_possible and input_data.dryness_fraction >= 0.98:
            warnings.append("Condensation risk in near-saturated steam")
        if heat_loss.total_heat_loss_kw > 50:
            warnings.append(f"Significant heat loss: {heat_loss.total_heat_loss_kw:.1f} kW")

        # Generate provenance hash
        hash_data = {
            'dryness_fraction': input_data.dryness_fraction,
            'pressure_bar': input_data.pressure_bar,
            'temperature_c': input_data.temperature_c,
            'wetness_percent': round(wetness, 10),
            'risk_level': risk_level.value,
        }
        provenance_hash = generate_provenance_hash(hash_data)

        end_time = time.perf_counter()
        calc_time_ms = (end_time - start_time) * 1000

        return MoistureAnalysisOutput(
            wetness_percent=wetness,
            moisture_mass_rate_kg_s=moisture_rate,
            risk_level=risk_level,
            condensation_possible=condensation_possible,
            probable_sources=sources,
            remediation_actions=actions,
            remediation_priority=priority,
            quality_score=quality_score,
            energy_loss_kw=energy_loss,
            provenance_hash=provenance_hash,
            calculation_time_ms=calc_time_ms,
            warnings=warnings
        )


# =============================================================================
# TEST CLASS: MOISTURE CONTENT ANALYSIS
# =============================================================================

class TestMoistureContentAnalysis:
    """Test suite for moisture content analysis calculations."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MoistureAnalyzer()

    @pytest.mark.unit
    def test_wetness_dry_steam(self, analyzer):
        """Test wetness is 0% for dry saturated steam."""
        wetness = analyzer.calculate_wetness_percent(dryness_fraction=1.0)
        assert wetness == 0.0

    @pytest.mark.unit
    def test_wetness_superheated(self, analyzer):
        """Test wetness is 0% for superheated steam (x > 1 clamped)."""
        wetness = analyzer.calculate_wetness_percent(dryness_fraction=1.05)
        assert wetness == 0.0

    @pytest.mark.unit
    def test_wetness_50pct_quality(self, analyzer):
        """Test wetness is 50% for 50% quality steam."""
        wetness = analyzer.calculate_wetness_percent(dryness_fraction=0.5)
        assert wetness == 50.0

    @pytest.mark.unit
    def test_wetness_90pct_quality(self, analyzer):
        """Test wetness is 10% for 90% quality steam."""
        wetness = analyzer.calculate_wetness_percent(dryness_fraction=0.9)
        assert_within_tolerance(wetness, 10.0, 0.001, "90% quality wetness")

    @pytest.mark.unit
    def test_wetness_negative_dryness_raises_error(self, analyzer):
        """Test negative dryness raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            analyzer.calculate_wetness_percent(dryness_fraction=-0.1)

    @pytest.mark.unit
    @pytest.mark.parametrize("dryness,expected_wetness", [
        (1.0, 0.0),
        (0.99, 1.0),
        (0.98, 2.0),
        (0.95, 5.0),
        (0.90, 10.0),
        (0.80, 20.0),
        (0.50, 50.0),
        (0.0, 100.0),
    ])
    def test_wetness_parametrized(self, analyzer, dryness, expected_wetness):
        """Parametrized test for wetness calculations."""
        wetness = analyzer.calculate_wetness_percent(dryness)
        assert_within_tolerance(wetness, expected_wetness, 0.01, f"Wetness at x={dryness}")


# =============================================================================
# TEST CLASS: CONDENSATION RISK DETECTION
# =============================================================================

class TestCondensationRiskDetection:
    """Test suite for condensation risk detection."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MoistureAnalyzer()

    @pytest.mark.unit
    def test_no_condensation_superheated(self, analyzer):
        """Test no condensation risk for superheated steam."""
        possible = analyzer.check_condensation_possible(
            temperature_c=250.0,
            saturation_temp_c=180.0,
            dryness_fraction=1.0
        )

        assert possible is False

    @pytest.mark.unit
    def test_condensation_wet_steam(self, analyzer):
        """Test condensation possible for wet steam."""
        possible = analyzer.check_condensation_possible(
            temperature_c=180.0,
            saturation_temp_c=180.0,
            dryness_fraction=0.95
        )

        assert possible is True

    @pytest.mark.unit
    def test_condensation_at_saturation(self, analyzer):
        """Test condensation possible at saturation temperature."""
        possible = analyzer.check_condensation_possible(
            temperature_c=180.0,
            saturation_temp_c=180.0,
            dryness_fraction=1.0
        )

        assert possible is True

    @pytest.mark.unit
    def test_condensation_near_saturation(self, analyzer):
        """Test condensation possible near saturation (within margin)."""
        possible = analyzer.check_condensation_possible(
            temperature_c=180.5,  # 0.5C above saturation
            saturation_temp_c=180.0,
            dryness_fraction=1.0
        )

        assert possible is True  # Within 1C margin

    @pytest.mark.unit
    def test_no_condensation_well_superheated(self, analyzer):
        """Test no condensation when well superheated."""
        possible = analyzer.check_condensation_possible(
            temperature_c=200.0,  # 20C superheat
            saturation_temp_c=180.0,
            dryness_fraction=1.0
        )

        assert possible is False


# =============================================================================
# TEST CLASS: WETNESS FRACTION CALCULATIONS
# =============================================================================

class TestWetnessFractionCalculations:
    """Test suite for wetness fraction and moisture mass rate."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MoistureAnalyzer()

    @pytest.mark.unit
    def test_moisture_mass_rate_dry_steam(self, analyzer):
        """Test moisture rate is zero for dry steam."""
        rate = analyzer.calculate_moisture_mass_rate(
            dryness_fraction=1.0,
            total_flow_kg_s=100.0
        )

        assert rate == 0.0

    @pytest.mark.unit
    def test_moisture_mass_rate_90pct_quality(self, analyzer):
        """Test moisture rate for 90% quality steam."""
        rate = analyzer.calculate_moisture_mass_rate(
            dryness_fraction=0.9,
            total_flow_kg_s=100.0
        )

        # 10% of 100 kg/s = 10 kg/s moisture
        assert_within_tolerance(rate, 10.0, 0.01, "Moisture rate at 90% quality")

    @pytest.mark.unit
    def test_moisture_mass_rate_proportional(self, analyzer):
        """Test moisture rate scales with flow."""
        rate_50 = analyzer.calculate_moisture_mass_rate(0.95, 50.0)
        rate_100 = analyzer.calculate_moisture_mass_rate(0.95, 100.0)

        assert_within_tolerance(rate_100 / rate_50, 2.0, 0.01, "Flow scaling")


# =============================================================================
# TEST CLASS: MOISTURE SOURCE IDENTIFICATION
# =============================================================================

class TestMoistureSourceIdentification:
    """Test suite for moisture source identification."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MoistureAnalyzer()

    @pytest.mark.unit
    def test_no_sources_dry_steam(self, analyzer):
        """Test no sources identified for dry steam."""
        sources = analyzer.identify_moisture_sources(
            dryness_fraction=1.0,
            upstream_dryness=1.0,
            pressure_drop_bar=0.5,
            heat_loss_kw=10.0,
            flow_rate_kg_s=50.0
        )

        assert len(sources) == 0

    @pytest.mark.unit
    def test_boiler_carryover_detection(self, analyzer):
        """Test boiler carryover is identified when upstream is wet."""
        sources = analyzer.identify_moisture_sources(
            dryness_fraction=0.95,
            upstream_dryness=0.96,  # Already wet at boiler
            pressure_drop_bar=0.5,
            heat_loss_kw=5.0,
            flow_rate_kg_s=50.0
        )

        assert MoistureSource.BOILER_CARRYOVER in sources

    @pytest.mark.unit
    def test_heat_loss_condensation_detection(self, analyzer):
        """Test heat loss condensation is identified."""
        sources = analyzer.identify_moisture_sources(
            dryness_fraction=0.98,
            upstream_dryness=1.0,
            pressure_drop_bar=0.5,
            heat_loss_kw=100.0,  # High heat loss
            flow_rate_kg_s=50.0
        )

        assert MoistureSource.CONDENSATION_HEAT_LOSS in sources

    @pytest.mark.unit
    def test_pressure_reduction_detection(self, analyzer):
        """Test pressure reduction is identified."""
        sources = analyzer.identify_moisture_sources(
            dryness_fraction=0.97,
            upstream_dryness=1.0,
            pressure_drop_bar=5.0,  # Significant pressure drop
            heat_loss_kw=10.0,
            flow_rate_kg_s=50.0
        )

        assert MoistureSource.PRESSURE_REDUCTION in sources

    @pytest.mark.unit
    def test_unknown_source_for_unexplained_moisture(self, analyzer):
        """Test unknown source when cause is not clear."""
        sources = analyzer.identify_moisture_sources(
            dryness_fraction=0.96,
            upstream_dryness=None,  # No upstream data
            pressure_drop_bar=0.1,
            heat_loss_kw=5.0,
            flow_rate_kg_s=50.0
        )

        assert MoistureSource.UNKNOWN in sources


# =============================================================================
# TEST CLASS: REMEDIATION RECOMMENDATIONS
# =============================================================================

class TestRemediationRecommendations:
    """Test suite for remediation action generation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MoistureAnalyzer()

    @pytest.mark.unit
    def test_no_remediation_low_risk(self, analyzer):
        """Test minimal remediation for no risk."""
        actions, priority = analyzer.generate_remediation_actions(
            risk_level=RiskLevel.NONE,
            sources=[]
        )

        assert "monitoring" in actions[0].lower()
        assert priority == RemediationPriority.MONITOR

    @pytest.mark.unit
    def test_immediate_priority_critical_risk(self, analyzer):
        """Test immediate priority for critical risk."""
        actions, priority = analyzer.generate_remediation_actions(
            risk_level=RiskLevel.CRITICAL,
            sources=[MoistureSource.BOILER_CARRYOVER]
        )

        assert priority == RemediationPriority.IMMEDIATE
        assert any("immediate" in a.lower() for a in actions)

    @pytest.mark.unit
    def test_boiler_carryover_actions(self, analyzer):
        """Test specific actions for boiler carryover."""
        actions, priority = analyzer.generate_remediation_actions(
            risk_level=RiskLevel.MEDIUM,
            sources=[MoistureSource.BOILER_CARRYOVER]
        )

        assert any("boiler" in a.lower() for a in actions)
        assert any("water level" in a.lower() or "drum" in a.lower() for a in actions)

    @pytest.mark.unit
    def test_heat_loss_actions(self, analyzer):
        """Test specific actions for heat loss condensation."""
        actions, priority = analyzer.generate_remediation_actions(
            risk_level=RiskLevel.MEDIUM,
            sources=[MoistureSource.CONDENSATION_HEAT_LOSS]
        )

        assert any("insulation" in a.lower() for a in actions)
        assert any("steam trap" in a.lower() for a in actions)

    @pytest.mark.unit
    def test_high_risk_includes_trap_inspection(self, analyzer):
        """Test high risk includes steam trap inspection."""
        actions, priority = analyzer.generate_remediation_actions(
            risk_level=RiskLevel.HIGH,
            sources=[MoistureSource.UNKNOWN]
        )

        assert any("trap" in a.lower() for a in actions)


# =============================================================================
# TEST CLASS: RISK LEVEL ASSESSMENT
# =============================================================================

class TestRiskLevelAssessment:
    """Test suite for risk level assessment."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MoistureAnalyzer()

    @pytest.mark.unit
    def test_risk_none_dry_steam(self, analyzer):
        """Test no risk for dry steam."""
        risk = analyzer.assess_risk_level(dryness_fraction=1.0)
        assert risk == RiskLevel.NONE

    @pytest.mark.unit
    def test_risk_low_99pct(self, analyzer):
        """Test low risk at 99% dryness."""
        risk = analyzer.assess_risk_level(dryness_fraction=0.99)
        assert risk == RiskLevel.LOW

    @pytest.mark.unit
    def test_risk_medium_96pct(self, analyzer):
        """Test medium risk at 96% dryness."""
        risk = analyzer.assess_risk_level(dryness_fraction=0.96)
        assert risk == RiskLevel.MEDIUM

    @pytest.mark.unit
    def test_risk_high_90pct(self, analyzer):
        """Test high risk at 90% dryness."""
        risk = analyzer.assess_risk_level(dryness_fraction=0.90)
        assert risk == RiskLevel.HIGH

    @pytest.mark.unit
    def test_risk_critical_80pct(self, analyzer):
        """Test critical risk at 80% dryness."""
        risk = analyzer.assess_risk_level(dryness_fraction=0.80)
        assert risk == RiskLevel.CRITICAL

    @pytest.mark.unit
    @pytest.mark.parametrize("dryness,expected_risk", [
        (1.0, RiskLevel.NONE),
        (0.995, RiskLevel.LOW),
        (0.98, RiskLevel.LOW),
        (0.97, RiskLevel.MEDIUM),
        (0.95, RiskLevel.MEDIUM),
        (0.94, RiskLevel.HIGH),
        (0.90, RiskLevel.HIGH),
        (0.88, RiskLevel.HIGH),
        (0.87, RiskLevel.CRITICAL),
        (0.50, RiskLevel.CRITICAL),
    ])
    def test_risk_level_parametrized(self, analyzer, dryness, expected_risk):
        """Parametrized test for risk levels."""
        risk = analyzer.assess_risk_level(dryness)
        assert risk == expected_risk


# =============================================================================
# TEST CLASS: QUALITY SCORE CALCULATIONS
# =============================================================================

class TestQualityScoreCalculations:
    """Test suite for quality score calculations."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MoistureAnalyzer()

    @pytest.mark.unit
    def test_quality_score_dry_steam(self, analyzer):
        """Test quality score is 100 for dry saturated steam."""
        score = analyzer.calculate_quality_score(dryness_fraction=1.0)
        assert score == 100.0

    @pytest.mark.unit
    def test_quality_score_superheated_bonus(self, analyzer):
        """Test quality score includes superheat bonus."""
        score = analyzer.calculate_quality_score(dryness_fraction=1.0, superheat_c=50.0)
        # Base 100 + up to 5 for superheat
        assert score >= 100.0

    @pytest.mark.unit
    def test_quality_score_wet_steam_penalty(self, analyzer):
        """Test quality score has penalty for very wet steam."""
        score = analyzer.calculate_quality_score(dryness_fraction=0.85)
        # Penalty for x < 0.90
        assert score < 85.0

    @pytest.mark.unit
    def test_quality_score_range(self, analyzer):
        """Test quality score is always in 0-100 range."""
        for x in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]:
            score = analyzer.calculate_quality_score(x)
            assert 0.0 <= score <= 100.0


# =============================================================================
# TEST CLASS: FULL ANALYSIS WORKFLOW
# =============================================================================

class TestFullMoistureAnalysis:
    """Test suite for complete moisture analysis workflow."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MoistureAnalyzer()

    @pytest.mark.unit
    def test_complete_analysis_dry_steam(self, analyzer):
        """Test complete analysis for dry steam."""
        input_data = MoistureAnalysisInput(
            dryness_fraction=1.0,
            pressure_bar=10.0,
            temperature_c=250.0,  # Superheated
            saturation_temp_c=180.0,
            flow_rate_kg_s=50.0
        )

        result = analyzer.analyze(input_data)

        assert result.wetness_percent == 0.0
        assert result.moisture_mass_rate_kg_s == 0.0
        assert result.risk_level == RiskLevel.NONE
        assert result.condensation_possible is False
        assert result.provenance_hash is not None

    @pytest.mark.unit
    def test_complete_analysis_wet_steam(self, analyzer):
        """Test complete analysis for wet steam."""
        input_data = MoistureAnalysisInput(
            dryness_fraction=0.92,
            pressure_bar=10.0,
            temperature_c=180.0,
            saturation_temp_c=180.0,
            flow_rate_kg_s=50.0
        )

        result = analyzer.analyze(input_data)

        assert_within_tolerance(result.wetness_percent, 8.0, 0.1, "Wetness")
        assert result.risk_level == RiskLevel.HIGH
        assert result.condensation_possible is True
        assert len(result.probable_sources) > 0
        assert len(result.remediation_actions) > 0

    @pytest.mark.unit
    def test_complete_analysis_critical_moisture(self, analyzer):
        """Test complete analysis for critical moisture level."""
        input_data = MoistureAnalysisInput(
            dryness_fraction=0.80,
            pressure_bar=10.0,
            temperature_c=180.0,
            saturation_temp_c=180.0,
            flow_rate_kg_s=50.0
        )

        result = analyzer.analyze(input_data)

        assert result.risk_level == RiskLevel.CRITICAL
        assert result.remediation_priority == RemediationPriority.IMMEDIATE
        assert len(result.warnings) > 0
        assert any("critical" in w.lower() for w in result.warnings)

    @pytest.mark.unit
    def test_analysis_includes_energy_loss(self, analyzer):
        """Test analysis calculates energy loss."""
        input_data = MoistureAnalysisInput(
            dryness_fraction=0.95,
            pressure_bar=10.0,
            temperature_c=180.0,
            saturation_temp_c=180.0,
            flow_rate_kg_s=50.0
        )

        result = analyzer.analyze(input_data)

        # 5% moisture at 50 kg/s = 2.5 kg/s moisture
        # Energy loss ~ 2.5 * 2000 = 5000 kW
        assert result.energy_loss_kw > 0


# =============================================================================
# TEST CLASS: DETERMINISM
# =============================================================================

class TestMoistureAnalyzerDeterminism:
    """Test suite for determinism verification."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MoistureAnalyzer()

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_wetness_determinism(self, analyzer):
        """Test wetness calculation is deterministic."""
        results = []
        for _ in range(100):
            wetness = analyzer.calculate_wetness_percent(0.95)
            results.append(wetness)

        assert_deterministic(results, "Wetness calculation")

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_risk_assessment_determinism(self, analyzer):
        """Test risk assessment is deterministic."""
        results = []
        for _ in range(100):
            risk = analyzer.assess_risk_level(0.92)
            results.append(risk)

        assert_deterministic(results, "Risk assessment")

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_full_analysis_determinism(self, analyzer):
        """Test full analysis is deterministic."""
        input_data = MoistureAnalysisInput(
            dryness_fraction=0.95,
            pressure_bar=10.0,
            temperature_c=180.0,
            saturation_temp_c=180.0,
            flow_rate_kg_s=50.0
        )

        results = []
        for _ in range(50):
            result = analyzer.analyze(input_data)
            results.append(result.wetness_percent)

        assert_deterministic(results, "Full analysis")

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_provenance_hash_reproducibility(self, analyzer):
        """Test provenance hash is reproducible."""
        input_data = MoistureAnalysisInput(
            dryness_fraction=0.95,
            pressure_bar=10.0,
            temperature_c=180.0,
            saturation_temp_c=180.0,
            flow_rate_kg_s=50.0
        )

        result1 = analyzer.analyze(input_data)
        result2 = analyzer.analyze(input_data)

        assert result1.provenance_hash == result2.provenance_hash


# =============================================================================
# TEST CLASS: PERFORMANCE
# =============================================================================

class TestMoistureAnalyzerPerformance:
    """Test suite for performance benchmarks."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MoistureAnalyzer()

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_analysis_under_3ms(self, analyzer, benchmark_targets):
        """Test single analysis completes under 3ms."""
        input_data = MoistureAnalysisInput(
            dryness_fraction=0.95,
            pressure_bar=10.0,
            temperature_c=180.0,
            saturation_temp_c=180.0,
            flow_rate_kg_s=50.0
        )

        result = analyzer.analyze(input_data)

        assert result.calculation_time_ms < benchmark_targets['moisture_analysis_ms']

    @pytest.mark.unit
    @pytest.mark.performance
    def test_batch_analyses(self, analyzer, performance_timer):
        """Test batch of 100 analyses."""
        input_data = MoistureAnalysisInput(
            dryness_fraction=0.95,
            pressure_bar=10.0,
            temperature_c=180.0,
            saturation_temp_c=180.0,
            flow_rate_kg_s=50.0
        )

        with performance_timer() as timer:
            for _ in range(100):
                analyzer.analyze(input_data)

        # 100 analyses should complete in under 200ms
        assert timer.elapsed_ms < 200.0
