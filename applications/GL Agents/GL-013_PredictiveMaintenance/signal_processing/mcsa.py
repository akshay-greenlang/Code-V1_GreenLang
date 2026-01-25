# -*- coding: utf-8 -*-
import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

class MCSAFaultType(Enum):
    BROKEN_ROTOR_BAR = 'broken_rotor_bar'
    ECCENTRICITY_STATIC = 'eccentricity_static'
    ECCENTRICITY_DYNAMIC = 'eccentricity_dynamic'
    BEARING_OUTER_RACE = 'bearing_outer_race'
    BEARING_INNER_RACE = 'bearing_inner_race'
    STATOR_WINDING = 'stator_winding'
    LOAD_OSCILLATION = 'load_oscillation'

@dataclass
class ThreePhaseCurrents:
    phase_a: NDArray[np.float64]
    phase_b: NDArray[np.float64]
    phase_c: NDArray[np.float64]
    sampling_rate: float
    timestamp: datetime

@dataclass
class MCSAConfig:
    supply_frequency: float = 50.0
    number_of_poles: int = 4
    number_of_rotor_bars: int = 28
    fft_resolution: float = 0.1
    harmonic_search_range: float = 5.0
    amplitude_threshold_db: float = -45.0

    @property
    def synchronous_speed(self) -> float:
        return (120 * self.supply_frequency) / self.number_of_poles

@dataclass
class MCSAFeature:
    frequency: float
    amplitude_db: float
    expected_frequency: float
    fault_type: MCSAFaultType
    severity: float
    confidence: float

@dataclass
class MCSAResult:
    timestamp: datetime
    supply_frequency_actual: float
    slip: float
    rotor_speed_rpm: float
    fundamental_amplitude: float
    features: List[MCSAFeature]
    fault_indicators: Dict[MCSAFaultType, float]
    overall_health: float
    provenance_hash: str
    processing_time_ms: float

class MCSAProcessor:
    def __init__(self, config: MCSAConfig):
        self.config = config
        self._processing_start: Optional[datetime] = None

    def analyze(self, currents: ThreePhaseCurrents, rated_speed_rpm: Optional[float] = None) -> MCSAResult:
        self._processing_start = datetime.utcnow()
        current_signal = (currents.phase_a + currents.phase_b + currents.phase_c) / 3
        spectrum, frequencies = self._compute_spectrum(current_signal, currents.sampling_rate)
        supply_freq_actual = self._find_supply_frequency(spectrum, frequencies)
        fundamental_idx = np.argmin(np.abs(frequencies - supply_freq_actual))
        fundamental_amplitude = spectrum[fundamental_idx]
        spectrum_db = 20 * np.log10(spectrum / fundamental_amplitude + 1e-12)
        slip, rotor_speed = self._estimate_slip_and_speed(spectrum_db, frequencies, supply_freq_actual, rated_speed_rpm)
        features = []
        fault_indicators = {}
        brb_features, brb_indicator = self._detect_broken_rotor_bars(spectrum_db, frequencies, supply_freq_actual, slip)
        features.extend(brb_features)
        fault_indicators[MCSAFaultType.BROKEN_ROTOR_BAR] = brb_indicator
        overall_health = self._calculate_overall_health(fault_indicators)
        provenance_hash = self._calculate_provenance_hash(currents, supply_freq_actual, features)
        processing_time = (datetime.utcnow() - self._processing_start).total_seconds() * 1000
        return MCSAResult(
            timestamp=currents.timestamp,
            supply_frequency_actual=supply_freq_actual,
            slip=slip,
            rotor_speed_rpm=rotor_speed,
            fundamental_amplitude=fundamental_amplitude,
            features=features,
            fault_indicators=fault_indicators,
            overall_health=overall_health,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time,
        )

    def _compute_spectrum(self, signal: NDArray[np.float64], sampling_rate: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        n_samples = len(signal)
        windowed = signal * np.hanning(n_samples)
        n_fft = int(sampling_rate / self.config.fft_resolution)
        n_fft = max(n_fft, n_samples)
        n_fft = 2 ** int(np.ceil(np.log2(n_fft)))
        fft_result = np.fft.rfft(windowed, n=n_fft)
        frequencies = np.fft.rfftfreq(n_fft, d=1/sampling_rate)
        amplitude = np.abs(fft_result) * 2 / n_samples
        return amplitude, frequencies

    def _find_supply_frequency(self, spectrum: NDArray[np.float64], frequencies: NDArray[np.float64]) -> float:
        search_range = 2.0
        mask = (frequencies >= self.config.supply_frequency - search_range) & (frequencies <= self.config.supply_frequency + search_range)
        if not np.any(mask):
            return self.config.supply_frequency
        search_freqs = frequencies[mask]
        search_amps = spectrum[mask]
        peak_idx = np.argmax(search_amps)
        return float(search_freqs[peak_idx])

    def _estimate_slip_and_speed(self, spectrum_db, frequencies, supply_freq, rated_speed):
        sync_speed = self.config.synchronous_speed
        if rated_speed is not None:
            slip = (sync_speed - rated_speed) / sync_speed
        else:
            slip = 0.03
        rotor_speed = sync_speed * (1 - slip)
        return slip, rotor_speed

    def _detect_broken_rotor_bars(self, spectrum_db, frequencies, supply_freq, slip):
        features = []
        max_severity = 0.0
        for k in range(1, 4):
            lower_freq = supply_freq * (1 - 2 * k * slip)
            upper_freq = supply_freq * (1 + 2 * k * slip)
            for freq in [lower_freq, upper_freq]:
                amp = self._find_peak_amplitude(spectrum_db, frequencies, freq)
                if amp is not None and amp > self.config.amplitude_threshold_db:
                    severity = min(1.0, max(0.0, (amp + 50) / 30))
                    features.append(MCSAFeature(
                        frequency=freq,
                        amplitude_db=amp,
                        expected_frequency=freq,
                        fault_type=MCSAFaultType.BROKEN_ROTOR_BAR,
                        severity=severity,
                        confidence=0.8 if k == 1 else 0.6,
                    ))
                    max_severity = max(max_severity, severity)
        return features, max_severity

    def _find_peak_amplitude(self, spectrum_db, frequencies, target_freq):
        search_range = self.config.harmonic_search_range
        mask = (frequencies >= target_freq - search_range) & (frequencies <= target_freq + search_range)
        if not np.any(mask):
            return None
        return float(np.max(spectrum_db[mask]))

    def _calculate_overall_health(self, fault_indicators):
        if not fault_indicators:
            return 1.0
        avg_fault = sum(fault_indicators.values()) / len(fault_indicators)
        return max(0.0, 1.0 - avg_fault)

    def _calculate_provenance_hash(self, currents, supply_freq, features):
        hasher = hashlib.sha256()
        hasher.update(f'samples:{len(currents.phase_a)}'.encode())
        hasher.update(f'rate:{currents.sampling_rate}'.encode())
        hasher.update(f'supply:{supply_freq:.4f}'.encode())
        for f in sorted(features, key=lambda x: x.frequency):
            hasher.update(f'{f.frequency:.2f}:{f.amplitude_db:.2f}'.encode())
        return hasher.hexdigest()

def extract_mcsa_features(currents: ThreePhaseCurrents, config: Optional[MCSAConfig] = None, rated_speed_rpm: Optional[float] = None) -> MCSAResult:
    if config is None:
        config = MCSAConfig()
    processor = MCSAProcessor(config)
    return processor.analyze(currents, rated_speed_rpm)
