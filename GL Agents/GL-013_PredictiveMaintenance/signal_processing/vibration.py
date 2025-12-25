from dataclasses import dataclass
from typing import Dict, Optional, Union
from enum import Enum
import hashlib
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.fft import fft, fftfreq

class WindowType(Enum):
    HANNING = "hanning"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    RECTANGULAR = "rectangular"
    KAISER = "kaiser"
    FLATTOP = "flattop"

@dataclass
class BearingGeometry:
    n_balls: int
    ball_diameter: float
    pitch_diameter: float
    contact_angle: float = 0.0

@dataclass
class BearingDefectFrequencies:
    bpfo: float
    bpfi: float
    bsf: float
    ftf: float
    shaft_frequency: float
    provenance_hash: str = ""

@dataclass
class SpectralFeatures:
    rms: float
    peak: float
    peak_to_peak: float
    crest_factor: float
    kurtosis: float
    skewness: float
    shape_factor: float
    impulse_factor: float
    clearance_factor: float
    provenance_hash: str = ""

@dataclass
class FFTResult:
    frequencies: NDArray[np.float64]
    magnitudes: NDArray[np.float64]
    phases: NDArray[np.float64]
    window_type: WindowType
    sample_rate: float
    n_samples: int
    provenance_hash: str = ""

@dataclass
class EnvelopeResult:
    envelope: NDArray[np.float64]
    envelope_spectrum_freq: NDArray[np.float64]
    envelope_spectrum_mag: NDArray[np.float64]
    analytic_signal: NDArray[np.complex128]
    provenance_hash: str = ""

@dataclass
class TriaxialAcceleration:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    sample_rate: float
    timestamp: float = 0.0
    def __post_init__(self):
        if not (len(self.x) == len(self.y) == len(self.z)):
            raise ValueError("All axes must have the same number of samples")
    @property
    def magnitude(self) -> NDArray[np.float64]:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    @property
    def n_samples(self) -> int:
        return len(self.x)

def _calculate_provenance_hash(data: Dict) -> str:
    provenance_str = str(sorted(data.items()))
    return hashlib.sha256(provenance_str.encode()).hexdigest()

def _get_window(window_type: WindowType, n_samples: int, kaiser_beta: float = 14.0):
    if window_type == WindowType.HANNING:
        return np.hanning(n_samples)
    elif window_type == WindowType.HAMMING:
        return np.hamming(n_samples)
    elif window_type == WindowType.BLACKMAN:
        return np.blackman(n_samples)
    elif window_type == WindowType.RECTANGULAR:
        return np.ones(n_samples)
    elif window_type == WindowType.KAISER:
        return np.kaiser(n_samples, kaiser_beta)
    elif window_type == WindowType.FLATTOP:
        return signal.windows.flattop(n_samples)
    raise ValueError(f"Unknown window type: {window_type}")


def compute_fft(signal_data, sample_rate, window_type=WindowType.HANNING, remove_dc=True, normalize=True):
    n_samples = len(signal_data)
    if remove_dc:
        signal_data = signal_data - np.mean(signal_data)
    window = _get_window(window_type, n_samples)
    windowed_signal = signal_data * window
    fft_result = fft(windowed_signal)
    n_positive = n_samples // 2 + 1
    frequencies = fftfreq(n_samples, 1.0 / sample_rate)[:n_positive]
    fft_positive = fft_result[:n_positive]
    magnitudes = np.abs(fft_positive)
    phases = np.angle(fft_positive)
    if normalize:
        coherent_gain = np.sum(window) / n_samples
        magnitudes = magnitudes * 2.0 / n_samples / coherent_gain
        magnitudes[0] /= 2.0
        if n_samples % 2 == 0:
            magnitudes[-1] /= 2.0
    provenance_data = {"n_samples": n_samples, "sample_rate": sample_rate, "window_type": window_type.value}
    return FFTResult(frequencies, magnitudes, phases, window_type, sample_rate, n_samples, _calculate_provenance_hash(provenance_data))

def envelope_analysis(signal_data, sample_rate, highpass_freq=None, lowpass_freq=None, filter_order=4):
    filtered_signal = signal_data.copy()
    nyquist = sample_rate / 2.0
    if highpass_freq and lowpass_freq:
        b, a = signal.butter(filter_order, [highpass_freq/nyquist, lowpass_freq/nyquist], btype="band")
        filtered_signal = signal.filtfilt(b, a, filtered_signal)
    elif highpass_freq:
        b, a = signal.butter(filter_order, highpass_freq/nyquist, btype="high")
        filtered_signal = signal.filtfilt(b, a, filtered_signal)
    elif lowpass_freq:
        b, a = signal.butter(filter_order, lowpass_freq/nyquist, btype="low")
        filtered_signal = signal.filtfilt(b, a, filtered_signal)
    analytic_signal = signal.hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    fft_result = compute_fft(envelope - np.mean(envelope), sample_rate)
    provenance_data = {"n_samples": len(signal_data), "sample_rate": sample_rate}
    return EnvelopeResult(envelope, fft_result.frequencies, fft_result.magnitudes, analytic_signal, _calculate_provenance_hash(provenance_data))


def extract_spectral_features(signal_data):
    n = len(signal_data)
    mean_val = np.mean(signal_data)
    std_val = np.std(signal_data, ddof=0)
    rms = np.sqrt(np.mean(signal_data ** 2))
    peak = np.max(np.abs(signal_data))
    peak_to_peak = np.max(signal_data) - np.min(signal_data)
    crest_factor = peak / rms if rms > 0 else 0.0
    if std_val > 0:
        centered = signal_data - mean_val
        kurtosis = np.mean(centered ** 4) / (std_val ** 4)
        skewness = np.mean(centered ** 3) / (std_val ** 3)
    else:
        kurtosis, skewness = 0.0, 0.0
    mean_abs = np.mean(np.abs(signal_data))
    shape_factor = rms / mean_abs if mean_abs > 0 else 0.0
    impulse_factor = peak / mean_abs if mean_abs > 0 else 0.0
    sqrt_mean = np.mean(np.sqrt(np.abs(signal_data)))
    clearance_factor = peak / (sqrt_mean ** 2) if sqrt_mean > 0 else 0.0
    return SpectralFeatures(float(rms), float(peak), float(peak_to_peak), float(crest_factor), float(kurtosis), float(skewness), float(shape_factor), float(impulse_factor), float(clearance_factor), _calculate_provenance_hash({"n_samples": n}))

def calculate_bearing_frequencies(geometry, shaft_rpm):
    alpha_rad = np.radians(geometry.contact_angle)
    cos_alpha = np.cos(alpha_rad)
    Bd_Pd = geometry.ball_diameter / geometry.pitch_diameter
    fr = shaft_rpm / 60.0
    bpfo = (geometry.n_balls / 2.0) * fr * (1.0 - Bd_Pd * cos_alpha)
    bpfi = (geometry.n_balls / 2.0) * fr * (1.0 + Bd_Pd * cos_alpha)
    bsf = (geometry.pitch_diameter / (2.0 * geometry.ball_diameter)) * fr * (1.0 - (Bd_Pd ** 2) * (cos_alpha ** 2))
    ftf = 0.5 * fr * (1.0 - Bd_Pd * cos_alpha)
    provenance_data = {"n_balls": geometry.n_balls, "ball_diameter": geometry.ball_diameter, "pitch_diameter": geometry.pitch_diameter, "contact_angle": geometry.contact_angle, "shaft_rpm": shaft_rpm}
    return BearingDefectFrequencies(float(bpfo), float(bpfi), float(bsf), float(ftf), float(fr), _calculate_provenance_hash(provenance_data))


class VibrationProcessor:
    def __init__(self, sample_rate, default_window=WindowType.HANNING):
        self.sample_rate = sample_rate
        self.default_window = default_window

    def compute_fft(self, signal_data, window_type=None, sample_rate=None):
        return compute_fft(signal_data, sample_rate or self.sample_rate, window_type or self.default_window)

    def envelope_analysis(self, signal_data, highpass_freq=None, lowpass_freq=None, sample_rate=None):
        return envelope_analysis(signal_data, sample_rate or self.sample_rate, highpass_freq, lowpass_freq)

    def extract_features(self, signal_data):
        return extract_spectral_features(signal_data)

    def calculate_bearing_frequencies(self, geometry, shaft_rpm):
        return calculate_bearing_frequencies(geometry, shaft_rpm)

    def process_triaxial(self, acceleration, analyze_magnitude=True, analyze_axes=True):
        results = {}
        if analyze_axes:
            for name, data in [("x", acceleration.x), ("y", acceleration.y), ("z", acceleration.z)]:
                results[f"{name}_fft"] = self.compute_fft(data, sample_rate=acceleration.sample_rate)
                results[f"{name}_features"] = self.extract_features(data)
                results[f"{name}_envelope"] = self.envelope_analysis(data, sample_rate=acceleration.sample_rate)
        if analyze_magnitude:
            mag = acceleration.magnitude
            results["magnitude_fft"] = self.compute_fft(mag, sample_rate=acceleration.sample_rate)
            results["magnitude_features"] = self.extract_features(mag)
            results["magnitude_envelope"] = self.envelope_analysis(mag, sample_rate=acceleration.sample_rate)
        return results

    def detect_bearing_defects(self, signal_data, bearing_geometry, shaft_rpm, frequency_tolerance=0.05, amplitude_threshold=0.1):
        defect_freqs = calculate_bearing_frequencies(bearing_geometry, shaft_rpm)
        env_result = self.envelope_analysis(signal_data)
        defects = {}
        for dtype, freq in [("outer_race", defect_freqs.bpfo), ("inner_race", defect_freqs.bpfi), ("ball", defect_freqs.bsf), ("cage", defect_freqs.ftf)]:
            tol = freq * frequency_tolerance
            mask = np.abs(env_result.envelope_spectrum_freq - freq) < tol
            if np.any(mask):
                amp = float(np.max(env_result.envelope_spectrum_mag[mask]))
                defects[dtype] = {"frequency_expected": freq, "amplitude": amp, "detected": amp > amplitude_threshold, "harmonics": []}
                for h in [2, 3]:
                    hmask = np.abs(env_result.envelope_spectrum_freq - freq*h) < tol
                    if np.any(hmask):
                        hamp = float(np.max(env_result.envelope_spectrum_mag[hmask]))
                        if hamp > amplitude_threshold:
                            defects[dtype]["harmonics"].append({"order": h, "frequency": freq*h, "amplitude": hamp})
        return {"defect_frequencies": defect_freqs, "envelope_analysis": env_result, "defects_detected": defects}
