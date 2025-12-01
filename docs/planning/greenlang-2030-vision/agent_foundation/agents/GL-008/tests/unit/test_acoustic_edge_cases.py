# -*- coding: utf-8 -*-
"""
Edge case tests for acoustic signature analysis in GL-008 SteamTrapInspector.

This module tests boundary conditions, edge cases, and unusual scenarios
for acoustic analysis across different trap types and operating conditions.
"""

import pytest
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tools import SteamTrapTools, AcousticAnalysisResult
from config import TrapType, FailureMode


@pytest.mark.edge_case
class TestAcousticSignalEdgeCases:
    """Test acoustic analysis with edge case signal conditions."""

    def test_very_low_signal_amplitude(self, tools):
        """Test analysis with extremely low signal amplitude (near noise floor)."""
        # Signal barely above noise floor
        signal = np.random.randn(10000) * 0.001  # Very low amplitude

        acoustic_data = {
            'trap_id': 'TRAP-LOW-SIGNAL',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        # Should still produce valid result
        assert isinstance(result, AcousticAnalysisResult)
        assert result.signal_strength_db < 20.0  # Very low signal strength
        assert result.failure_probability < 0.5  # Likely normal operation

    def test_saturated_signal_clipping(self, tools, signal_generator):
        """Test analysis with saturated/clipped signal."""
        signal = signal_generator.generate_saturated_signal()

        acoustic_data = {
            'trap_id': 'TRAP-SATURATED',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        # Should detect anomaly due to saturation
        assert result.anomaly_detected == True
        assert result.signal_strength_db > 60.0  # High signal strength

    def test_dc_offset_signal(self, tools):
        """Test analysis with signal containing DC offset."""
        signal = np.random.randn(10000) * 0.2
        signal += 2.0  # Add DC offset

        acoustic_data = {
            'trap_id': 'TRAP-DC-OFFSET',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        # Should handle DC offset properly
        assert isinstance(result, AcousticAnalysisResult)
        assert result.provenance_hash is not None

    def test_single_sample_signal(self, tools):
        """Test analysis with single sample signal."""
        acoustic_data = {
            'trap_id': 'TRAP-SINGLE',
            'signal': [0.5],
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        # Should handle gracefully
        assert isinstance(result, AcousticAnalysisResult)

    def test_very_short_signal(self, tools):
        """Test analysis with very short signal (< 100 samples)."""
        signal = np.random.randn(50) * 0.1

        acoustic_data = {
            'trap_id': 'TRAP-SHORT',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert isinstance(result, AcousticAnalysisResult)
        assert result.confidence_score < 0.8  # Lower confidence due to short signal

    def test_very_long_signal(self, tools):
        """Test analysis with very long signal (> 1 million samples)."""
        signal = np.random.randn(1000000) * 0.2

        acoustic_data = {
            'trap_id': 'TRAP-LONG',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        # Should handle large data
        assert isinstance(result, AcousticAnalysisResult)
        assert result.provenance_hash is not None

    def test_all_zeros_signal(self, tools):
        """Test analysis with signal of all zeros (sensor failure)."""
        signal = np.zeros(10000)

        acoustic_data = {
            'trap_id': 'TRAP-ZEROS',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        # Should detect as anomaly
        assert result.anomaly_detected == True
        assert result.signal_strength_db < 0  # No signal

    def test_impulse_noise_signal(self, tools):
        """Test analysis with impulse noise (spikes)."""
        signal = np.random.randn(10000) * 0.1
        # Add random impulses
        impulse_indices = np.random.choice(10000, 100, replace=False)
        signal[impulse_indices] += np.random.randn(100) * 5.0

        acoustic_data = {
            'trap_id': 'TRAP-IMPULSE',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert isinstance(result, AcousticAnalysisResult)
        assert result.anomaly_detected == True


@pytest.mark.edge_case
class TestAcousticFrequencyEdgeCases:
    """Test acoustic analysis at frequency boundary conditions."""

    def test_nyquist_frequency_limit(self, tools):
        """Test signal at Nyquist frequency (fs/2)."""
        sampling_rate = 250000
        nyquist_freq = sampling_rate / 2

        t = np.linspace(0, 1.0, sampling_rate)
        signal = np.sin(2 * np.pi * nyquist_freq * t)

        acoustic_data = {
            'trap_id': 'TRAP-NYQUIST',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert isinstance(result, AcousticAnalysisResult)
        # Frequency should be near Nyquist limit
        assert result.frequency_peak_hz >= 100000

    def test_sub_audible_frequency(self, tools):
        """Test signal below audible range (< 20 Hz)."""
        sampling_rate = 250000
        t = np.linspace(0, 1.0, sampling_rate)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz

        acoustic_data = {
            'trap_id': 'TRAP-SUB-AUDIBLE',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert result.frequency_peak_hz < 100  # Low frequency detected

    def test_ultra_high_frequency(self, tools):
        """Test signal at ultra-high frequency (> 100 kHz)."""
        sampling_rate = 250000
        t = np.linspace(0, 1.0, sampling_rate)
        signal = np.sin(2 * np.pi * 110000 * t)  # 110 kHz

        acoustic_data = {
            'trap_id': 'TRAP-ULTRA-HIGH',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert isinstance(result, AcousticAnalysisResult)

    def test_multi_frequency_harmonics(self, tools):
        """Test signal with multiple harmonic frequencies."""
        sampling_rate = 250000
        t = np.linspace(0, 1.0, sampling_rate)

        # Fundamental + harmonics
        signal = (np.sin(2 * np.pi * 25000 * t) +
                 0.5 * np.sin(2 * np.pi * 50000 * t) +
                 0.25 * np.sin(2 * np.pi * 75000 * t))

        acoustic_data = {
            'trap_id': 'TRAP-HARMONICS',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        # Should detect dominant frequency
        assert 20000 <= result.frequency_peak_hz <= 30000


@pytest.mark.edge_case
class TestAcousticTrapTypeEdgeCases:
    """Test acoustic analysis for different trap types."""

    @pytest.mark.parametrize("trap_type,expected_behavior", [
        (TrapType.THERMODYNAMIC, "cyclic_discharge"),
        (TrapType.THERMOSTATIC, "continuous_modulation"),
        (TrapType.FLOAT_AND_THERMOSTATIC, "level_controlled"),
        (TrapType.INVERTED_BUCKET, "intermittent_discharge"),
        (TrapType.MECHANICAL, "mechanical_operation"),
        (TrapType.DISC, "disc_opening"),
        (TrapType.BIMETALLIC, "thermal_expansion"),
        (TrapType.BALANCED_PRESSURE, "pressure_balanced")
    ])
    def test_trap_type_specific_signatures(self, tools, trap_type, expected_behavior):
        """Test acoustic signatures specific to each trap type."""
        # Generate generic test signal
        signal = np.random.randn(10000) * 0.3

        acoustic_data = {
            'trap_id': f'TRAP-TYPE-{trap_type.value}',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data, trap_type)

        assert isinstance(result, AcousticAnalysisResult)
        assert result.trap_id == f'TRAP-TYPE-{trap_type.value}'

    def test_thermodynamic_cyclic_pattern(self, tools):
        """Test thermodynamic trap cyclic discharge pattern."""
        sampling_rate = 250000
        t = np.linspace(0, 5.0, int(sampling_rate * 5))

        # Simulate cyclic discharge (5 Hz cycle)
        cycle_freq = 5.0
        carrier_freq = 30000.0
        signal = np.sin(2 * np.pi * carrier_freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * cycle_freq * t))

        acoustic_data = {
            'trap_id': 'TRAP-THERMO-CYCLE',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data, TrapType.THERMODYNAMIC)

        assert result.frequency_peak_hz > 20000  # Carrier frequency detected

    def test_inverted_bucket_intermittent(self, tools):
        """Test inverted bucket intermittent discharge."""
        sampling_rate = 250000
        signal = np.zeros(int(sampling_rate * 2))

        # Add intermittent bursts (bucket dumping)
        burst_start = [10000, 80000, 150000, 220000]
        for start in burst_start:
            burst_length = 5000
            t = np.linspace(0, burst_length / sampling_rate, burst_length)
            burst = 2.0 * np.sin(2 * np.pi * 32000 * t)
            signal[start:start + burst_length] = burst

        acoustic_data = {
            'trap_id': 'TRAP-BUCKET-INTERMITTENT',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data, TrapType.INVERTED_BUCKET)

        assert isinstance(result, AcousticAnalysisResult)


@pytest.mark.edge_case
class TestAcousticAmbientConditions:
    """Test acoustic analysis under various ambient conditions."""

    def test_high_ambient_noise(self, tools):
        """Test analysis with high ambient noise floor."""
        # Signal buried in noise
        clean_signal = np.sin(2 * np.pi * 30000 * np.linspace(0, 1, 250000))
        noise = np.random.randn(250000) * 1.5  # High noise
        signal = clean_signal * 0.5 + noise

        acoustic_data = {
            'trap_id': 'TRAP-HIGH-NOISE',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        # Confidence should be lower due to noise
        assert result.confidence_score < 0.9

    def test_electromagnetic_interference(self, tools):
        """Test analysis with EMI (60 Hz interference)."""
        sampling_rate = 250000
        t = np.linspace(0, 1.0, sampling_rate)

        # 60 Hz EMI + actual signal
        emi = 0.3 * np.sin(2 * np.pi * 60 * t)
        signal = np.sin(2 * np.pi * 30000 * t) + emi

        acoustic_data = {
            'trap_id': 'TRAP-EMI',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        # Should detect high frequency component despite EMI
        assert result.frequency_peak_hz > 10000

    def test_temperature_drift_effect(self, tools):
        """Test analysis with simulated temperature drift in sensor."""
        signal = np.random.randn(10000) * 0.2
        # Simulate slow drift
        drift = np.linspace(0, 0.5, 10000)
        signal += drift

        acoustic_data = {
            'trap_id': 'TRAP-TEMP-DRIFT',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert isinstance(result, AcousticAnalysisResult)


@pytest.mark.edge_case
class TestAcousticFailureModeDetection:
    """Test acoustic detection of specific failure modes."""

    def test_cavitation_detection(self, tools):
        """Test detection of cavitation acoustic signature."""
        sampling_rate = 250000
        t = np.linspace(0, 1.0, sampling_rate)

        # Cavitation: broadband noise with peaks around 10-40 kHz
        cavitation_signal = (np.random.randn(sampling_rate) * 0.8 +
                           0.5 * np.sin(2 * np.pi * 15000 * t) +
                           0.3 * np.sin(2 * np.pi * 25000 * t))

        acoustic_data = {
            'trap_id': 'TRAP-CAVITATION',
            'signal': cavitation_signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert result.anomaly_detected == True
        assert result.failure_probability > 0.5

    def test_waterhammer_signature(self, tools):
        """Test detection of waterhammer impact signature."""
        signal = np.zeros(50000)

        # Waterhammer: sharp impulse followed by ringing
        impact_location = 25000
        decay_rate = 0.95

        for i in range(5000):
            signal[impact_location + i] = 3.0 * (decay_rate ** i) * np.sin(2 * np.pi * 500 * i / 250000)

        acoustic_data = {
            'trap_id': 'TRAP-WATERHAMMER',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert result.anomaly_detected == True

    def test_wear_gradual_degradation(self, tools):
        """Test detection of gradual wear signature."""
        sampling_rate = 250000
        t = np.linspace(0, 1.0, sampling_rate)

        # Gradually increasing noise floor (wear)
        base_signal = np.sin(2 * np.pi * 28000 * t)
        wear_noise = np.random.randn(sampling_rate) * (0.1 + 0.3 * t / t[-1])
        signal = base_signal + wear_noise

        acoustic_data = {
            'trap_id': 'TRAP-WEAR',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert isinstance(result, AcousticAnalysisResult)


@pytest.mark.edge_case
class TestAcousticBoundaryConditions:
    """Test acoustic analysis at operational boundary conditions."""

    def test_zero_pressure_condition(self, tools):
        """Test analysis at zero pressure (trap offline)."""
        # Minimal signal - trap not operating
        signal = np.random.randn(10000) * 0.01

        acoustic_data = {
            'trap_id': 'TRAP-ZERO-PRESSURE',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000,
            'operating_pressure_psig': 0.0
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert result.signal_strength_db < 10.0

    def test_maximum_pressure_condition(self, tools):
        """Test analysis at maximum rated pressure (600 psig)."""
        sampling_rate = 250000
        t = np.linspace(0, 1.0, sampling_rate)

        # High pressure = higher amplitude
        signal = 3.0 * np.sin(2 * np.pi * 35000 * t)

        acoustic_data = {
            'trap_id': 'TRAP-MAX-PRESSURE',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate,
            'operating_pressure_psig': 600.0
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert result.signal_strength_db > 50.0

    def test_startup_transient(self, tools):
        """Test analysis during trap startup transient."""
        sampling_rate = 250000
        t = np.linspace(0, 2.0, int(sampling_rate * 2))

        # Startup: increasing amplitude over time
        amplitude = 0.2 + 1.8 * (1 - np.exp(-t))
        signal = amplitude * np.sin(2 * np.pi * 30000 * t)

        acoustic_data = {
            'trap_id': 'TRAP-STARTUP',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert isinstance(result, AcousticAnalysisResult)

    def test_shutdown_transient(self, tools):
        """Test analysis during trap shutdown."""
        sampling_rate = 250000
        t = np.linspace(0, 2.0, int(sampling_rate * 2))

        # Shutdown: decreasing amplitude
        amplitude = 2.0 * np.exp(-t)
        signal = amplitude * np.sin(2 * np.pi * 30000 * t)

        acoustic_data = {
            'trap_id': 'TRAP-SHUTDOWN',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        assert isinstance(result, AcousticAnalysisResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "edge_case"])
