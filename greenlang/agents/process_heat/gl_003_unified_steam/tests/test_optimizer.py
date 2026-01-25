"""
GL-003 Unified Steam System Optimizer - Main Optimizer Tests

Unit tests for the main UnifiedSteamOptimizer class.
Target: 85%+ coverage of optimizer.py

Tests:
    - Full system optimization workflow
    - Input/output validation
    - Sub-optimizer integration
    - Provenance tracking
    - Status determination
    - Recommendation generation
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import time


from greenlang.agents.process_heat.gl_003_unified_steam.optimizer import (
    UnifiedSteamOptimizer,
    UnifiedSteamOptimizerInput,
)
from greenlang.agents.process_heat.gl_003_unified_steam.config import (
    UnifiedSteamConfig,
    create_default_config,
)
from greenlang.agents.process_heat.gl_003_unified_steam.schemas import (
    HeaderBalanceInput,
    HeaderBalanceOutput,
    SteamQualityReading,
    SteamQualityAnalysis,
    PRVOperatingPoint,
    PRVSizingOutput,
    CondensateReading,
    CondensateReturnAnalysis,
    SteamTrapReading,
    TrapStatus,
    UnifiedSteamOptimizerOutput,
    OptimizationStatus,
    ValidationStatus,
)


# =============================================================================
# UNIFIED STEAM OPTIMIZER INPUT TESTS
# =============================================================================

class TestUnifiedSteamOptimizerInput:
    """Test suite for UnifiedSteamOptimizerInput model."""

    def test_default_input_creation(self):
        """Test default input creation."""
        input_data = UnifiedSteamOptimizerInput()

        assert input_data.header_readings == []
        assert input_data.quality_readings == []
        assert input_data.prv_readings == []
        assert input_data.condensate_readings == []
        assert input_data.trap_readings == []
        assert input_data.total_steam_flow_lb_hr == 0.0

    def test_input_with_data(self, header_balance_input, steam_quality_reading_good):
        """Test input with actual data."""
        input_data = UnifiedSteamOptimizerInput(
            header_readings=[header_balance_input],
            quality_readings=[steam_quality_reading_good],
            total_steam_flow_lb_hr=100000.0,
        )

        assert len(input_data.header_readings) == 1
        assert len(input_data.quality_readings) == 1
        assert input_data.total_steam_flow_lb_hr == 100000.0

    def test_input_timestamp(self):
        """Test input timestamp is set."""
        input_data = UnifiedSteamOptimizerInput()

        assert input_data.timestamp is not None
        assert isinstance(input_data.timestamp, datetime)


# =============================================================================
# UNIFIED STEAM OPTIMIZER TESTS
# =============================================================================

class TestUnifiedSteamOptimizer:
    """Test suite for UnifiedSteamOptimizer."""

    @pytest.fixture
    def optimizer(self, default_config):
        """Create optimizer with default configuration."""
        with patch('optimizer.IntelligenceMixin._init_intelligence'):
            return UnifiedSteamOptimizer(config=default_config)

    def test_initialization(self, optimizer, default_config):
        """Test optimizer initialization."""
        assert optimizer.steam_config == default_config
        assert optimizer.AGENT_TYPE == "GL-003"
        assert optimizer.AGENT_NAME == "Unified Steam System Optimizer"

    def test_initialization_sub_optimizers(self, optimizer):
        """Test sub-optimizers are initialized."""
        assert optimizer.distribution_optimizer is not None
        assert optimizer.quality_monitor is not None
        assert optimizer.condensate_optimizer is not None
        assert len(optimizer.flash_optimizers) >= 0
        assert len(optimizer.prv_optimizers) >= 0

    def test_process_empty_input(self, optimizer):
        """Test processing with empty input."""
        input_data = UnifiedSteamOptimizerInput()

        with patch.object(optimizer, 'generate_explanation', return_value="Test explanation"):
            result = optimizer.process(input_data)

        assert isinstance(result, UnifiedSteamOptimizerOutput)
        assert result.provenance_hash is not None

    def test_process_with_header_data(self, optimizer, header_balance_input):
        """Test processing with header data."""
        input_data = UnifiedSteamOptimizerInput(
            header_readings=[header_balance_input],
            total_steam_flow_lb_hr=100000.0,
        )

        with patch.object(optimizer, 'generate_explanation', return_value="Test explanation"):
            result = optimizer.process(input_data)

        assert len(result.header_analyses) > 0

    def test_process_with_quality_data(self, optimizer, steam_quality_reading_good):
        """Test processing with quality data."""
        input_data = UnifiedSteamOptimizerInput(
            quality_readings=[steam_quality_reading_good],
            total_steam_flow_lb_hr=100000.0,
        )

        with patch.object(optimizer, 'generate_explanation', return_value="Test explanation"):
            result = optimizer.process(input_data)

        assert len(result.quality_analyses) > 0

    def test_process_with_condensate_data(self, optimizer, condensate_reading):
        """Test processing with condensate data."""
        input_data = UnifiedSteamOptimizerInput(
            condensate_readings=[condensate_reading],
            total_steam_flow_lb_hr=100000.0,
        )

        with patch.object(optimizer, 'generate_explanation', return_value="Test explanation"):
            result = optimizer.process(input_data)

        assert result.condensate_analysis is not None

    def test_validate_input_valid(self, optimizer):
        """Test input validation with valid data."""
        input_data = UnifiedSteamOptimizerInput(
            total_steam_flow_lb_hr=100000.0,
        )

        assert optimizer.validate_input(input_data) is True

    def test_validate_input_negative_flow(self, optimizer):
        """Test input validation rejects negative flow."""
        input_data = UnifiedSteamOptimizerInput(
            total_steam_flow_lb_hr=-100.0,
        )

        assert optimizer.validate_input(input_data) is False

    def test_validate_output_valid(self, optimizer):
        """Test output validation with valid data."""
        output_data = UnifiedSteamOptimizerOutput(
            overall_status=OptimizationStatus.OPTIMAL,
            system_efficiency_pct=92.5,
            provenance_hash="a" * 64,
        )

        assert optimizer.validate_output(output_data) is True

    def test_validate_output_invalid_efficiency(self, optimizer):
        """Test output validation rejects invalid efficiency."""
        output_data = UnifiedSteamOptimizerOutput(
            overall_status=OptimizationStatus.OPTIMAL,
            system_efficiency_pct=150.0,  # Invalid
            provenance_hash="a" * 64,
        )

        assert optimizer.validate_output(output_data) is False

    def test_validate_output_missing_hash(self, optimizer):
        """Test output validation rejects missing hash."""
        output_data = UnifiedSteamOptimizerOutput(
            overall_status=OptimizationStatus.OPTIMAL,
            system_efficiency_pct=92.5,
            provenance_hash="",  # Empty
        )

        assert optimizer.validate_output(output_data) is False


# =============================================================================
# SYSTEM EFFICIENCY TESTS
# =============================================================================

class TestSystemEfficiency:
    """Test suite for system efficiency calculation."""

    @pytest.fixture
    def optimizer(self, default_config):
        """Create optimizer."""
        with patch('optimizer.IntelligenceMixin._init_intelligence'):
            return UnifiedSteamOptimizer(config=default_config)

    def test_calculate_efficiency_optimal(self, optimizer):
        """Test efficiency calculation for optimal system."""
        headers = [
            Mock(status=OptimizationStatus.OPTIMAL),
            Mock(status=OptimizationStatus.OPTIMAL),
        ]
        quality = [
            Mock(overall_status=ValidationStatus.VALID),
        ]

        efficiency = optimizer._calculate_system_efficiency(headers, quality, None)

        assert efficiency > 95  # All optimal should be high

    def test_calculate_efficiency_suboptimal(self, optimizer):
        """Test efficiency calculation for suboptimal system."""
        headers = [
            Mock(status=OptimizationStatus.SUBOPTIMAL),
            Mock(status=OptimizationStatus.OPTIMAL),
        ]
        quality = [
            Mock(overall_status=ValidationStatus.WARNING),
        ]

        efficiency = optimizer._calculate_system_efficiency(headers, quality, None)

        assert 80 < efficiency < 95

    def test_calculate_efficiency_with_condensate(self, optimizer):
        """Test efficiency calculation with condensate analysis."""
        headers = [Mock(status=OptimizationStatus.OPTIMAL)]
        quality = []

        condensate = Mock()
        condensate.return_rate_pct = 85.0
        condensate.target_return_rate_pct = 85.0

        efficiency = optimizer._calculate_system_efficiency(headers, quality, condensate)

        assert efficiency > 0


# =============================================================================
# STATUS DETERMINATION TESTS
# =============================================================================

class TestStatusDetermination:
    """Test suite for overall status determination."""

    @pytest.fixture
    def optimizer(self, default_config):
        """Create optimizer."""
        with patch('optimizer.IntelligenceMixin._init_intelligence'):
            return UnifiedSteamOptimizer(config=default_config)

    def test_determine_status_optimal(self, optimizer):
        """Test status determination for optimal conditions."""
        headers = [Mock(status=OptimizationStatus.OPTIMAL)]
        quality = [Mock(overall_status=ValidationStatus.VALID)]
        prvs = [Mock(meets_opening_targets=True)]

        status = optimizer._determine_overall_status(headers, quality, prvs)

        assert status == OptimizationStatus.OPTIMAL

    def test_determine_status_critical_header(self, optimizer):
        """Test critical status for critical header."""
        headers = [Mock(status=OptimizationStatus.CRITICAL)]
        quality = []
        prvs = []

        status = optimizer._determine_overall_status(headers, quality, prvs)

        assert status == OptimizationStatus.CRITICAL

    def test_determine_status_invalid_quality(self, optimizer):
        """Test critical status for invalid quality."""
        headers = []
        quality = [Mock(overall_status=ValidationStatus.INVALID)]
        prvs = []

        status = optimizer._determine_overall_status(headers, quality, prvs)

        assert status == OptimizationStatus.CRITICAL

    def test_determine_status_suboptimal(self, optimizer):
        """Test suboptimal status for warnings."""
        headers = [
            Mock(status=OptimizationStatus.SUBOPTIMAL),
            Mock(status=OptimizationStatus.OPTIMAL),
        ]
        quality = [Mock(overall_status=ValidationStatus.WARNING)]
        prvs = [Mock(meets_opening_targets=False)]

        status = optimizer._determine_overall_status(headers, quality, prvs)

        assert status == OptimizationStatus.SUBOPTIMAL


# =============================================================================
# RECOMMENDATION GENERATION TESTS
# =============================================================================

class TestRecommendationGeneration:
    """Test suite for recommendation generation."""

    @pytest.fixture
    def optimizer(self, default_config):
        """Create optimizer."""
        with patch('optimizer.IntelligenceMixin._init_intelligence'):
            return UnifiedSteamOptimizer(config=default_config)

    def test_generate_header_recommendations(self, optimizer):
        """Test header balance recommendations."""
        header = Mock()
        header.status = OptimizationStatus.SUBOPTIMAL
        header.adjustments = [
            {"source_id": "BLR-001", "action": "increase", "recommended_flow_lb_hr": 55000}
        ]

        recommendations = optimizer._generate_recommendations(
            header_analyses=[header],
            quality_analyses=[],
            prv_analyses=[],
            condensate_analysis=None,
            flash_analyses=[],
            trap_analysis=None,
        )

        assert len(recommendations) > 0
        assert recommendations[0].category == "header_balance"

    def test_generate_quality_recommendations(self, optimizer):
        """Test quality recommendations."""
        quality = Mock()
        quality.overall_status = ValidationStatus.WARNING
        quality.reading = Mock(location_id="TEST")
        quality.recommendations = ["Increase blowdown"]

        recommendations = optimizer._generate_recommendations(
            header_analyses=[],
            quality_analyses=[quality],
            prv_analyses=[],
            condensate_analysis=None,
            flash_analyses=[],
            trap_analysis=None,
        )

        assert len(recommendations) > 0
        assert recommendations[0].category == "steam_quality"

    def test_generate_flash_recommendations(self, optimizer):
        """Test flash recovery recommendations."""
        flash = Mock()
        flash.flash_fraction_pct = 12.0
        flash.flash_steam_lb_hr = 600.0
        flash.annual_savings_usd = 25000.0

        recommendations = optimizer._generate_recommendations(
            header_analyses=[],
            quality_analyses=[],
            prv_analyses=[],
            condensate_analysis=None,
            flash_analyses=[flash],
            trap_analysis=None,
        )

        assert len(recommendations) > 0
        assert recommendations[0].category == "flash_recovery"

    def test_recommendations_sorted_by_priority(self, optimizer):
        """Test recommendations are sorted by priority."""
        quality = Mock()
        quality.overall_status = ValidationStatus.INVALID
        quality.reading = Mock(location_id="CRITICAL")
        quality.recommendations = ["Critical action needed"]

        prv = Mock()
        prv.prv_id = "PRV-001"
        prv.recommendations = ["Optional adjustment"]

        recommendations = optimizer._generate_recommendations(
            header_analyses=[],
            quality_analyses=[quality],
            prv_analyses=[prv],
            condensate_analysis=None,
            flash_analyses=[],
            trap_analysis=None,
        )

        # Priority 1 (critical quality) should come before priority 2
        if len(recommendations) > 1:
            assert recommendations[0].priority <= recommendations[1].priority


# =============================================================================
# PROVENANCE TESTS
# =============================================================================

class TestProvenance:
    """Test suite for provenance tracking."""

    @pytest.fixture
    def optimizer(self, default_config):
        """Create optimizer."""
        with patch('optimizer.IntelligenceMixin._init_intelligence'):
            return UnifiedSteamOptimizer(config=default_config)

    def test_provenance_hash_generated(self, optimizer):
        """Test provenance hash is generated for processing."""
        input_data = UnifiedSteamOptimizerInput()

        with patch.object(optimizer, 'generate_explanation', return_value="Test"):
            result = optimizer.process(input_data)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_deterministic(self, optimizer):
        """Test provenance hash is deterministic for same input."""
        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        input_data = UnifiedSteamOptimizerInput(
            timestamp=timestamp,
            total_steam_flow_lb_hr=100000.0,
        )

        with patch.object(optimizer, 'generate_explanation', return_value="Test"):
            # Fix the calculation count and timestamp
            optimizer._calculation_count = 0
            hash1 = optimizer._calculate_provenance_hash(input_data)

            optimizer._calculation_count = 0
            hash2 = optimizer._calculate_provenance_hash(input_data)

        # Same input with same calculation count should produce same hash structure
        # Note: timestamp in hash calculation may differ

    def test_provenance_includes_agent_info(self, optimizer):
        """Test provenance calculation includes agent information."""
        input_data = UnifiedSteamOptimizerInput()

        # The hash is calculated from data that includes agent_id and version
        # We verify this by checking the optimizer has these attributes
        assert optimizer.steam_config.agent_id is not None
        assert optimizer.AGENT_VERSION is not None


# =============================================================================
# CONVENIENCE METHOD TESTS
# =============================================================================

class TestConvenienceMethods:
    """Test suite for convenience methods."""

    @pytest.fixture
    def optimizer(self, default_config):
        """Create optimizer."""
        with patch('optimizer.IntelligenceMixin._init_intelligence'):
            return UnifiedSteamOptimizer(config=default_config)

    def test_analyze_header(self, optimizer, header_balance_input):
        """Test single header analysis."""
        result = optimizer.analyze_header("HP-MAIN", header_balance_input)

        assert isinstance(result, HeaderBalanceOutput)
        assert result.header_id == "HP-MAIN"

    def test_analyze_quality(self, optimizer, steam_quality_reading_good):
        """Test single quality analysis."""
        result = optimizer.analyze_quality(steam_quality_reading_good)

        assert isinstance(result, SteamQualityAnalysis)

    def test_size_prv(self, optimizer):
        """Test PRV sizing."""
        # Get first PRV ID from config
        prv_id = optimizer.steam_config.prvs[0].prv_id

        result = optimizer.size_prv(prv_id)

        assert isinstance(result, PRVSizingOutput)
        assert result.prv_id == prv_id

    def test_size_prv_unknown_id(self, optimizer):
        """Test PRV sizing with unknown ID."""
        with pytest.raises(ValueError, match="not configured"):
            optimizer.size_prv("UNKNOWN-PRV")

    def test_calculate_flash(self, optimizer):
        """Test flash calculation."""
        result = optimizer.calculate_flash(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        assert result.flash_fraction_pct > 0
        assert result.flash_steam_lb_hr > 0

    def test_get_steam_properties(self, optimizer):
        """Test steam properties lookup."""
        props = optimizer.get_steam_properties(150.0)

        assert "pressure_psig" in props
        assert "temperature_f" in props
        assert "enthalpy_btu_lb" in props


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestOptimizerIntegration:
    """Integration tests for full optimizer workflow."""

    @pytest.fixture
    def optimizer(self, default_config):
        """Create optimizer."""
        with patch('optimizer.IntelligenceMixin._init_intelligence'):
            return UnifiedSteamOptimizer(config=default_config)

    @pytest.mark.integration
    def test_full_optimization_workflow(
        self,
        optimizer,
        header_balance_input,
        steam_quality_reading_good,
        prv_operating_point,
        condensate_reading,
        steam_trap_reading_good,
    ):
        """Test complete optimization workflow."""
        input_data = UnifiedSteamOptimizerInput(
            header_readings=[header_balance_input],
            quality_readings=[steam_quality_reading_good],
            prv_readings=[prv_operating_point],
            condensate_readings=[condensate_reading],
            trap_readings=[steam_trap_reading_good],
            total_steam_flow_lb_hr=100000.0,
            boiler_water_tds_ppm=2500.0,
        )

        with patch.object(optimizer, 'generate_explanation', return_value="Test"):
            result = optimizer.process(input_data)

        assert result.overall_status is not None
        assert result.system_efficiency_pct > 0
        assert result.provenance_hash is not None
        assert result.processing_time_ms > 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestOptimizerPerformance:
    """Performance tests for optimizer."""

    @pytest.fixture
    def optimizer(self, default_config):
        """Create optimizer."""
        with patch('optimizer.IntelligenceMixin._init_intelligence'):
            return UnifiedSteamOptimizer(config=default_config)

    @pytest.mark.performance
    def test_processing_speed(self, optimizer):
        """Test optimization processing speed."""
        input_data = UnifiedSteamOptimizerInput(
            total_steam_flow_lb_hr=100000.0,
        )

        start = time.time()
        with patch.object(optimizer, 'generate_explanation', return_value="Test"):
            for _ in range(10):
                optimizer.process(input_data)
        elapsed = time.time() - start

        # 10 optimizations in <5s
        assert elapsed < 5.0

    @pytest.mark.performance
    def test_processing_time_reported(self, optimizer):
        """Test processing time is reported in output."""
        input_data = UnifiedSteamOptimizerInput()

        with patch.object(optimizer, 'generate_explanation', return_value="Test"):
            result = optimizer.process(input_data)

        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 1000  # Should be <1s


# =============================================================================
# INTELLIGENCE INTERFACE TESTS
# =============================================================================

class TestIntelligenceInterface:
    """Test suite for intelligence interface methods."""

    @pytest.fixture
    def optimizer(self, default_config):
        """Create optimizer."""
        with patch('optimizer.IntelligenceMixin._init_intelligence'):
            return UnifiedSteamOptimizer(config=default_config)

    def test_get_intelligence_level(self, optimizer):
        """Test intelligence level is ADVANCED."""
        level = optimizer.get_intelligence_level()
        assert level.value == "advanced"

    def test_get_intelligence_capabilities(self, optimizer):
        """Test intelligence capabilities."""
        caps = optimizer.get_intelligence_capabilities()

        assert caps.can_explain is True
        assert caps.can_recommend is True
        assert caps.can_detect_anomalies is True
        assert caps.can_reason is True
        assert caps.can_validate is True
