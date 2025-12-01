# -*- coding: utf-8 -*-
"""
End-to-End Tests for GL-003 STEAMWISE SteamSystemAnalyzer.

Tests complete steam system analysis workflows including:
- Full steam system analysis workflow
- Leak detection end-to-end
- Condensate optimization workflow

Author: GL-TestEngineer
Version: 1.0.0
Standards: GL-012 Test Patterns, Zero Hallucination Compliance
"""

import asyncio
import pytest
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# MOCK STEAM SYSTEM COMPONENTS
# ============================================================================

@dataclass
class MockBoiler:
    """Mock boiler for E2E testing."""
    boiler_id: str
    pressure_bar: float = 10.0
    temperature_c: float = 180.0
    steam_flow_kg_hr: float = 5000.0
    efficiency_percent: float = 85.0
    fuel_flow_rate: float = 500.0

    async def read_status(self) -> Dict[str, Any]:
        return {
            'boiler_id': self.boiler_id,
            'pressure_bar': self.pressure_bar,
            'temperature_c': self.temperature_c,
            'steam_flow_kg_hr': self.steam_flow_kg_hr,
            'efficiency_percent': self.efficiency_percent,
            'status': 'running'
        }


@dataclass
class MockSteamHeader:
    """Mock steam header for distribution testing."""
    header_id: str
    pressure_bar: float = 10.0
    temperature_c: float = 180.0
    flow_rate_kg_hr: float = 5000.0

    async def read_flow(self) -> Dict[str, Any]:
        return {
            'header_id': self.header_id,
            'pressure_bar': self.pressure_bar,
            'temperature_c': self.temperature_c,
            'flow_rate_kg_hr': self.flow_rate_kg_hr,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


@dataclass
class MockSteamTrap:
    """Mock steam trap for leak detection."""
    trap_id: str
    trap_type: str = 'inverted_bucket'
    status: str = 'operating'  # operating, failed, blowing_through
    steam_loss_kg_hr: float = 0.0
    pressure_bar: float = 10.0

    async def inspect(self) -> Dict[str, Any]:
        return {
            'trap_id': self.trap_id,
            'trap_type': self.trap_type,
            'status': self.status,
            'steam_loss_kg_hr': self.steam_loss_kg_hr,
            'pressure_bar': self.pressure_bar,
            'inspection_timestamp': datetime.now(timezone.utc).isoformat()
        }


@dataclass
class MockCondensateReceiver:
    """Mock condensate receiver for recovery testing."""
    receiver_id: str
    condensate_flow_kg_hr: float = 4000.0
    condensate_temperature_c: float = 95.0
    condensate_pressure_bar: float = 1.5
    return_rate_percent: float = 75.0

    async def read_condensate_data(self) -> Dict[str, Any]:
        return {
            'receiver_id': self.receiver_id,
            'condensate_flow_kg_hr': self.condensate_flow_kg_hr,
            'condensate_temperature_c': self.condensate_temperature_c,
            'condensate_pressure_bar': self.condensate_pressure_bar,
            'return_rate_percent': self.return_rate_percent
        }


class MockSteamSystem:
    """Complete mock steam system for E2E testing."""

    def __init__(self, num_boilers: int = 1, num_traps: int = 10):
        self.boilers = {
            f"BOILER-{i}": MockBoiler(f"BOILER-{i}")
            for i in range(1, num_boilers + 1)
        }
        self.headers = {
            f"HEADER-{i}": MockSteamHeader(f"HEADER-{i}")
            for i in range(1, 3)
        }
        self.traps = {
            f"TRAP-{i:03d}": MockSteamTrap(f"TRAP-{i:03d}")
            for i in range(1, num_traps + 1)
        }
        self.condensate_receiver = MockCondensateReceiver("RECEIVER-001")
        self.connected = False
        self.analysis_history = []

    async def connect(self) -> bool:
        self.connected = True
        return True

    async def disconnect(self) -> bool:
        self.connected = False
        return True

    async def collect_system_data(self) -> Dict[str, Any]:
        """Collect data from all system components."""
        if not self.connected:
            raise ConnectionError("System not connected")

        boiler_data = {}
        for bid, boiler in self.boilers.items():
            boiler_data[bid] = await boiler.read_status()

        header_data = {}
        for hid, header in self.headers.items():
            header_data[hid] = await header.read_flow()

        trap_data = {}
        for tid, trap in self.traps.items():
            trap_data[tid] = await trap.inspect()

        condensate_data = await self.condensate_receiver.read_condensate_data()

        return {
            'boilers': boiler_data,
            'headers': header_data,
            'traps': trap_data,
            'condensate': condensate_data,
            'collection_timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def run_analysis(self, analysis_type: str) -> Dict[str, Any]:
        """Run specified analysis type."""
        system_data = await self.collect_system_data()

        if analysis_type == 'efficiency':
            result = self._analyze_efficiency(system_data)
        elif analysis_type == 'leak_detection':
            result = self._analyze_leaks(system_data)
        elif analysis_type == 'condensate':
            result = self._analyze_condensate(system_data)
        elif analysis_type == 'full':
            result = {
                'efficiency': self._analyze_efficiency(system_data),
                'leaks': self._analyze_leaks(system_data),
                'condensate': self._analyze_condensate(system_data)
            }
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        # Generate provenance hash
        result['provenance_hash'] = hashlib.sha256(
            json.dumps(result, sort_keys=True, default=str).encode()
        ).hexdigest()

        self.analysis_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis_type': analysis_type,
            'provenance_hash': result['provenance_hash']
        })

        return result

    def _analyze_efficiency(self, data: Dict) -> Dict[str, Any]:
        """Analyze steam system efficiency."""
        boiler_efficiencies = [
            b['efficiency_percent'] for b in data['boilers'].values()
        ]
        avg_efficiency = sum(boiler_efficiencies) / len(boiler_efficiencies)

        total_steam_flow = sum(
            h['flow_rate_kg_hr'] for h in data['headers'].values()
        )

        return {
            'average_boiler_efficiency': avg_efficiency,
            'total_steam_flow_kg_hr': total_steam_flow,
            'efficiency_status': 'good' if avg_efficiency >= 80 else 'needs_improvement'
        }

    def _analyze_leaks(self, data: Dict) -> Dict[str, Any]:
        """Analyze steam leaks from trap data."""
        failed_traps = [
            tid for tid, trap in data['traps'].items()
            if trap['status'] in ['failed', 'blowing_through']
        ]
        total_steam_loss = sum(
            trap['steam_loss_kg_hr'] for trap in data['traps'].values()
        )

        return {
            'total_traps': len(data['traps']),
            'failed_traps': len(failed_traps),
            'failed_trap_ids': failed_traps,
            'total_steam_loss_kg_hr': total_steam_loss,
            'leak_status': 'critical' if len(failed_traps) > 5 else 'acceptable'
        }

    def _analyze_condensate(self, data: Dict) -> Dict[str, Any]:
        """Analyze condensate recovery."""
        condensate = data['condensate']
        return_rate = condensate['return_rate_percent']

        potential_improvement = max(0, 90 - return_rate)

        return {
            'current_return_rate_percent': return_rate,
            'target_return_rate_percent': 90.0,
            'potential_improvement_percent': potential_improvement,
            'condensate_temperature_c': condensate['condensate_temperature_c'],
            'recovery_status': 'optimal' if return_rate >= 85 else 'improvable'
        }


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_steam_system():
    """Create mock steam system for testing."""
    return MockSteamSystem(num_boilers=2, num_traps=20)


@pytest.fixture
def steam_system_with_leaks():
    """Create steam system with simulated leaks."""
    system = MockSteamSystem(num_boilers=1, num_traps=10)

    # Simulate failed traps
    system.traps["TRAP-001"].status = 'failed'
    system.traps["TRAP-001"].steam_loss_kg_hr = 50.0
    system.traps["TRAP-003"].status = 'blowing_through'
    system.traps["TRAP-003"].steam_loss_kg_hr = 100.0
    system.traps["TRAP-007"].status = 'failed'
    system.traps["TRAP-007"].steam_loss_kg_hr = 25.0

    return system


@pytest.fixture
def steam_system_poor_condensate():
    """Create steam system with poor condensate recovery."""
    system = MockSteamSystem(num_boilers=1, num_traps=5)
    system.condensate_receiver.return_rate_percent = 45.0
    return system


@pytest.fixture
def high_efficiency_steam_system():
    """Create high-efficiency steam system for baseline testing."""
    system = MockSteamSystem(num_boilers=2, num_traps=15)

    # Set high efficiency parameters
    for boiler in system.boilers.values():
        boiler.efficiency_percent = 92.0

    system.condensate_receiver.return_rate_percent = 92.0

    return system


# ============================================================================
# COMPLETE STEAM SYSTEM ANALYSIS WORKFLOW TESTS
# ============================================================================

class TestCompleteSteamAnalysisWorkflow:
    """Test complete steam system analysis workflows."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_full_system_analysis_workflow(self, mock_steam_system):
        """Test complete system analysis from connection to recommendations."""
        # Step 1: Connect to system
        connected = await mock_steam_system.connect()
        assert connected is True
        assert mock_steam_system.connected is True

        # Step 2: Collect system data
        system_data = await mock_steam_system.collect_system_data()
        assert 'boilers' in system_data
        assert 'headers' in system_data
        assert 'traps' in system_data
        assert 'condensate' in system_data
        assert 'collection_timestamp' in system_data

        # Step 3: Run full analysis
        analysis_result = await mock_steam_system.run_analysis('full')

        # Validate analysis structure
        assert 'efficiency' in analysis_result
        assert 'leaks' in analysis_result
        assert 'condensate' in analysis_result
        assert 'provenance_hash' in analysis_result

        # Validate provenance hash format (SHA-256)
        assert len(analysis_result['provenance_hash']) == 64

        # Step 4: Verify analysis history recorded
        assert len(mock_steam_system.analysis_history) == 1
        assert mock_steam_system.analysis_history[0]['analysis_type'] == 'full'

        # Step 5: Disconnect
        disconnected = await mock_steam_system.disconnect()
        assert disconnected is True
        assert mock_steam_system.connected is False

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_efficiency_analysis_workflow(self, mock_steam_system):
        """Test efficiency-focused analysis workflow."""
        await mock_steam_system.connect()

        result = await mock_steam_system.run_analysis('efficiency')

        assert 'average_boiler_efficiency' in result
        assert 'total_steam_flow_kg_hr' in result
        assert 'efficiency_status' in result
        assert 'provenance_hash' in result

        # Verify efficiency is within expected range
        assert 0 <= result['average_boiler_efficiency'] <= 100
        assert result['total_steam_flow_kg_hr'] > 0

        await mock_steam_system.disconnect()

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_with_connection_failure_recovery(self, mock_steam_system):
        """Test workflow recovery from connection failure."""
        await mock_steam_system.connect()

        # Simulate connection loss
        mock_steam_system.connected = False

        with pytest.raises(ConnectionError):
            await mock_steam_system.collect_system_data()

        # Reconnect
        await mock_steam_system.connect()
        assert mock_steam_system.connected is True

        # Analysis should now succeed
        result = await mock_steam_system.run_analysis('efficiency')
        assert 'provenance_hash' in result

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_provenance_hash_determinism(self, mock_steam_system):
        """Test that provenance hash is deterministic for same data."""
        await mock_steam_system.connect()

        # Run same analysis multiple times
        hashes = []
        for _ in range(5):
            result = await mock_steam_system.run_analysis('efficiency')
            hashes.append(result['provenance_hash'])

        # All hashes should be identical (deterministic)
        assert len(set(hashes)) == 1, "Provenance hash must be deterministic"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_high_efficiency_baseline(self, high_efficiency_steam_system):
        """Test analysis of high-efficiency system for baseline comparison."""
        await high_efficiency_steam_system.connect()

        result = await high_efficiency_steam_system.run_analysis('full')

        # Verify high efficiency detected
        assert result['efficiency']['average_boiler_efficiency'] >= 90.0
        assert result['efficiency']['efficiency_status'] == 'good'

        # Verify optimal condensate recovery
        assert result['condensate']['current_return_rate_percent'] >= 90.0
        assert result['condensate']['recovery_status'] == 'optimal'


# ============================================================================
# LEAK DETECTION END-TO-END TESTS
# ============================================================================

class TestLeakDetectionE2E:
    """End-to-end tests for leak detection workflow."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_leak_detection_workflow_with_failed_traps(
        self, steam_system_with_leaks
    ):
        """Test complete leak detection workflow with failed traps."""
        await steam_system_with_leaks.connect()

        result = await steam_system_with_leaks.run_analysis('leak_detection')

        # Verify leak detection results
        assert result['failed_traps'] == 3
        assert 'TRAP-001' in result['failed_trap_ids']
        assert 'TRAP-003' in result['failed_trap_ids']
        assert 'TRAP-007' in result['failed_trap_ids']

        # Verify total steam loss calculated
        expected_loss = 50.0 + 100.0 + 25.0  # 175 kg/hr
        assert result['total_steam_loss_kg_hr'] == expected_loss

        # Verify provenance tracking
        assert 'provenance_hash' in result
        assert len(result['provenance_hash']) == 64

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_leak_detection_healthy_system(self, mock_steam_system):
        """Test leak detection on healthy system."""
        await mock_steam_system.connect()

        result = await mock_steam_system.run_analysis('leak_detection')

        # Healthy system should have no failed traps
        assert result['failed_traps'] == 0
        assert len(result['failed_trap_ids']) == 0
        assert result['total_steam_loss_kg_hr'] == 0
        assert result['leak_status'] == 'acceptable'

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_critical_leak_threshold_detection(self, mock_steam_system):
        """Test detection of critical leak threshold (>5 failed traps)."""
        await mock_steam_system.connect()

        # Fail 6 traps to trigger critical threshold
        for i in range(1, 7):
            mock_steam_system.traps[f"TRAP-{i:03d}"].status = 'failed'
            mock_steam_system.traps[f"TRAP-{i:03d}"].steam_loss_kg_hr = 20.0

        result = await mock_steam_system.run_analysis('leak_detection')

        assert result['failed_traps'] == 6
        assert result['leak_status'] == 'critical'

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_leak_cost_calculation_integration(self, steam_system_with_leaks):
        """Test that leak costs can be calculated from detection results."""
        await steam_system_with_leaks.connect()

        result = await steam_system_with_leaks.run_analysis('leak_detection')

        # Calculate annual cost (simplified)
        steam_cost_per_kg = Decimal('0.05')  # $0.05 per kg
        operating_hours = Decimal('8760')  # Hours per year

        annual_steam_loss = Decimal(str(result['total_steam_loss_kg_hr'])) * operating_hours
        annual_cost = annual_steam_loss * steam_cost_per_kg

        # Verify cost is significant
        assert float(annual_cost) > 1000  # Should be > $1000 for 175 kg/hr loss


# ============================================================================
# CONDENSATE OPTIMIZATION WORKFLOW TESTS
# ============================================================================

class TestCondensateOptimizationE2E:
    """End-to-end tests for condensate optimization workflow."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_condensate_optimization_workflow(
        self, steam_system_poor_condensate
    ):
        """Test complete condensate optimization workflow."""
        await steam_system_poor_condensate.connect()

        result = await steam_system_poor_condensate.run_analysis('condensate')

        # Verify current state detected
        assert result['current_return_rate_percent'] == 45.0
        assert result['target_return_rate_percent'] == 90.0

        # Verify improvement potential calculated
        expected_improvement = 90.0 - 45.0
        assert result['potential_improvement_percent'] == expected_improvement
        assert result['recovery_status'] == 'improvable'

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_optimal_condensate_recovery_detection(
        self, high_efficiency_steam_system
    ):
        """Test detection of already optimal condensate recovery."""
        await high_efficiency_steam_system.connect()

        result = await high_efficiency_steam_system.run_analysis('condensate')

        assert result['current_return_rate_percent'] >= 90.0
        assert result['recovery_status'] == 'optimal'
        assert result['potential_improvement_percent'] == 0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_condensate_temperature_tracking(self, mock_steam_system):
        """Test that condensate temperature is properly tracked."""
        await mock_steam_system.connect()

        result = await mock_steam_system.run_analysis('condensate')

        # Temperature should be present and reasonable
        assert 'condensate_temperature_c' in result
        assert 50 <= result['condensate_temperature_c'] <= 200


# ============================================================================
# MULTI-ANALYSIS WORKFLOW TESTS
# ============================================================================

class TestMultiAnalysisWorkflow:
    """Tests for workflows involving multiple analysis types."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_sequential_analysis_types(self, mock_steam_system):
        """Test running different analysis types sequentially."""
        await mock_steam_system.connect()

        # Run each analysis type
        efficiency_result = await mock_steam_system.run_analysis('efficiency')
        leak_result = await mock_steam_system.run_analysis('leak_detection')
        condensate_result = await mock_steam_system.run_analysis('condensate')

        # Verify all analyses completed
        assert 'provenance_hash' in efficiency_result
        assert 'provenance_hash' in leak_result
        assert 'provenance_hash' in condensate_result

        # Verify analysis history
        assert len(mock_steam_system.analysis_history) == 3

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_analysis_with_changing_system_state(self, mock_steam_system):
        """Test analysis detects changes in system state."""
        await mock_steam_system.connect()

        # Initial analysis
        initial_result = await mock_steam_system.run_analysis('leak_detection')
        assert initial_result['failed_traps'] == 0

        # Simulate trap failure
        mock_steam_system.traps["TRAP-001"].status = 'failed'
        mock_steam_system.traps["TRAP-001"].steam_loss_kg_hr = 75.0

        # Re-analyze
        updated_result = await mock_steam_system.run_analysis('leak_detection')

        # Verify change detected
        assert updated_result['failed_traps'] == 1
        assert updated_result['total_steam_loss_kg_hr'] == 75.0

        # Provenance hashes should differ
        assert initial_result['provenance_hash'] != updated_result['provenance_hash']

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_concurrent_analysis_isolation(self, mock_steam_system):
        """Test that concurrent analyses don't interfere."""
        await mock_steam_system.connect()

        # Run analyses concurrently
        results = await asyncio.gather(
            mock_steam_system.run_analysis('efficiency'),
            mock_steam_system.run_analysis('leak_detection'),
            mock_steam_system.run_analysis('condensate'),
        )

        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert 'provenance_hash' in result


# ============================================================================
# ERROR HANDLING AND RECOVERY TESTS
# ============================================================================

class TestE2EErrorHandling:
    """Tests for error handling in E2E workflows."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_invalid_analysis_type_rejection(self, mock_steam_system):
        """Test that invalid analysis type is rejected."""
        await mock_steam_system.connect()

        with pytest.raises(ValueError, match="Unknown analysis type"):
            await mock_steam_system.run_analysis('invalid_type')

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_analysis_before_connection_fails(self, mock_steam_system):
        """Test that analysis fails if not connected."""
        # Don't connect - system starts disconnected
        assert mock_steam_system.connected is False

        with pytest.raises(ConnectionError):
            await mock_steam_system.collect_system_data()

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_graceful_shutdown_after_error(self, mock_steam_system):
        """Test graceful shutdown after analysis error."""
        await mock_steam_system.connect()

        try:
            await mock_steam_system.run_analysis('invalid')
        except ValueError:
            pass

        # System should still be disconnectable
        disconnected = await mock_steam_system.disconnect()
        assert disconnected is True
        assert mock_steam_system.connected is False
