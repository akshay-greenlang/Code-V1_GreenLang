"""
GL-013 PredictiveMaintenance - Thermal Feature Extraction Module

Zero-Hallucination Thermal Analysis for Equipment Health Monitoring

Key Features:
- Temperature rise above ambient calculation
- Rate of temperature change (dT/dt)
- Time above threshold accumulation
- Winding hot-spot estimation (thermal network model)
- Environmental normalization

Thermal Model Reference:
- IEEE C57.91-2011 Guide for Loading Mineral-Oil-Immersed Transformers
- IEC 60076-7 Loading Guide for Oil-Immersed Power Transformers

Author: GL-013 PredictiveMaintenance Agent
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import hashlib
import numpy as np
from numpy.typing import NDArray


@dataclass
class TemperatureRiseResult:
    ambient_temperature: float
    measured_temperature: float
    temperature_rise: float
    normalized_rise: float
    rated_rise: float
    rise_ratio: float
    alarm_triggered: bool
    provenance_hash: str = ""


@dataclass
class TemperatureRateResult:
    dt_dt: float
    dt_dt_smoothed: float
    time_window_seconds: float
    max_rate: float
    min_rate: float
    rate_alarm: bool
    provenance_hash: str = ""


@dataclass
class TimeAboveThresholdResult:
    threshold_temperature: float
    time_above_threshold_seconds: float
    cumulative_exposure: float
    aging_factor: float
    alarm_triggered: bool
    provenance_hash: str = ""


@dataclass
class HotspotEstimationResult:
    top_oil_temperature: float
    winding_hotspot_temperature: float
    hotspot_rise_over_top_oil: float
    hotspot_rise_over_ambient: float
    load_factor: float
    thermal_time_constant: float
    alarm_triggered: bool
    provenance_hash: str = ""


@dataclass
class EnvironmentalNormalizationResult:
    raw_temperature: float
    ambient_temperature: float
    humidity_percent: float
    altitude_meters: float
    normalized_temperature: float
    correction_factor: float
    provenance_hash: str = ""


def _calculate_provenance_hash(data: Dict) -> str:
    provenance_str = str(sorted(data.items()))
    return hashlib.sha256(provenance_str.encode()).hexdigest()


def calculate_temperature_rise(measured_temp: float, ambient_temp: float, rated_rise: float = 65.0, alarm_threshold_ratio: float = 1.1) -> TemperatureRiseResult:
    """
    Calculate temperature rise above ambient - DETERMINISTIC.
    
    Formula: Rise = T_measured - T_ambient
    Normalized Rise = Rise * (40 / T_ambient) for standard 40C ambient
    Rise Ratio = Rise / Rated_Rise
    """
    temperature_rise = measured_temp - ambient_temp
    normalized_rise = temperature_rise * (40.0 / ambient_temp) if ambient_temp > 0 else temperature_rise
    rise_ratio = temperature_rise / rated_rise if rated_rise > 0 else 0.0
    alarm_triggered = rise_ratio > alarm_threshold_ratio
    provenance_data = {"measured_temp": measured_temp, "ambient_temp": ambient_temp, "rated_rise": rated_rise}
    return TemperatureRiseResult(float(ambient_temp), float(measured_temp), float(temperature_rise), float(normalized_rise), float(rated_rise), float(rise_ratio), alarm_triggered, _calculate_provenance_hash(provenance_data))


def calculate_temperature_rate(temperature_history: NDArray[np.float64], timestamps: NDArray[np.float64], smoothing_window: int = 5, rate_alarm_threshold: float = 2.0) -> TemperatureRateResult:
    """
    Calculate rate of temperature change (dT/dt) - DETERMINISTIC.
    
    Formula: dT/dt = (T[i] - T[i-1]) / (t[i] - t[i-1])
    Smoothed using moving average window
    """
    if len(temperature_history) < 2:
        return TemperatureRateResult(0.0, 0.0, 0.0, 0.0, 0.0, False, _calculate_provenance_hash({"n_samples": len(temperature_history)}))
    dt = np.diff(timestamps)
    dT = np.diff(temperature_history)
    rates = dT / dt
    if len(rates) >= smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed_rates = np.convolve(rates, kernel, mode="valid")
        dt_dt_smoothed = smoothed_rates[-1] if len(smoothed_rates) > 0 else rates[-1]
    else:
        dt_dt_smoothed = np.mean(rates)
    dt_dt = rates[-1] if len(rates) > 0 else 0.0
    max_rate = np.max(rates)
    min_rate = np.min(rates)
    time_window = timestamps[-1] - timestamps[0]
    rate_alarm = abs(dt_dt_smoothed) > rate_alarm_threshold
    provenance_data = {"n_samples": len(temperature_history), "smoothing_window": smoothing_window}
    return TemperatureRateResult(float(dt_dt), float(dt_dt_smoothed), float(time_window), float(max_rate), float(min_rate), rate_alarm, _calculate_provenance_hash(provenance_data))


def accumulate_time_above_threshold(temperature_history: NDArray[np.float64], timestamps: NDArray[np.float64], threshold: float, activation_energy: float = 15000.0, reference_temp: float = 110.0) -> TimeAboveThresholdResult:
    """
    Calculate cumulative time above threshold and thermal aging - DETERMINISTIC.
    
    Aging Factor (Arrhenius): F_aging = exp(E_a/R * (1/T_ref - 1/T))
    Cumulative Exposure = sum(dt * F_aging) for T > threshold
    """
    R = 8.314
    above_threshold = temperature_history > threshold
    time_above = 0.0
    cumulative_exposure = 0.0
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        for i in range(len(dt)):
            if above_threshold[i]:
                time_above += dt[i]
                T_kelvin = temperature_history[i] + 273.15
                T_ref_kelvin = reference_temp + 273.15
                aging_factor = np.exp((activation_energy / R) * (1/T_ref_kelvin - 1/T_kelvin))
                cumulative_exposure += dt[i] * aging_factor
    avg_aging = cumulative_exposure / time_above if time_above > 0 else 1.0
    alarm_triggered = cumulative_exposure > 3600
    provenance_data = {"n_samples": len(temperature_history), "threshold": threshold, "activation_energy": activation_energy}
    return TimeAboveThresholdResult(float(threshold), float(time_above), float(cumulative_exposure), float(avg_aging), alarm_triggered, _calculate_provenance_hash(provenance_data))


def estimate_hotspot_temperature(top_oil_temp: float, load_current: float, rated_current: float, ambient_temp: float, rated_top_oil_rise: float = 55.0, rated_hotspot_rise: float = 23.0, oil_exponent: float = 0.8, winding_exponent: float = 1.6) -> HotspotEstimationResult:
    """
    Estimate winding hotspot temperature using thermal model - DETERMINISTIC.
    
    IEEE C57.91-2011 Clause 7:
    Hot_spot = T_ambient + delta_T_oil + delta_T_winding
    delta_T_oil = delta_T_oil_rated * K^(2*n)
    delta_T_winding = delta_T_winding_rated * K^(2*m)
    K = I_load / I_rated (load factor)
    """
    load_factor = load_current / rated_current if rated_current > 0 else 0.0
    top_oil_rise = top_oil_temp - ambient_temp
    delta_oil = rated_top_oil_rise * (load_factor ** (2 * oil_exponent))
    delta_winding = rated_hotspot_rise * (load_factor ** (2 * winding_exponent))
    hotspot_temp = ambient_temp + delta_oil + delta_winding
    hotspot_rise_over_top = hotspot_temp - top_oil_temp
    hotspot_rise_over_ambient = hotspot_temp - ambient_temp
    thermal_tau = 180.0
    alarm_triggered = hotspot_temp > 120.0
    provenance_data = {"top_oil_temp": top_oil_temp, "load_current": load_current, "rated_current": rated_current, "ambient_temp": ambient_temp}
    return HotspotEstimationResult(float(top_oil_temp), float(hotspot_temp), float(hotspot_rise_over_top), float(hotspot_rise_over_ambient), float(load_factor), float(thermal_tau), alarm_triggered, _calculate_provenance_hash(provenance_data))


def normalize_for_environment(raw_temp: float, ambient_temp: float, humidity: float = 50.0, altitude: float = 0.0, reference_ambient: float = 40.0, reference_humidity: float = 50.0, reference_altitude: float = 0.0) -> EnvironmentalNormalizationResult:
    """
    Normalize temperature for environmental conditions - DETERMINISTIC.
    
    Corrections:
    - Ambient: T_norm = T_raw - (T_ambient - T_ref_ambient)
    - Altitude: +1% per 100m above 1000m (air density effect)
    - Humidity: Minor effect on convection
    """
    ambient_correction = ambient_temp - reference_ambient
    altitude_correction = 0.0
    if altitude > 1000:
        altitude_correction = (altitude - 1000) / 100 * 0.01 * (raw_temp - ambient_temp)
    humidity_correction = (humidity - reference_humidity) / 100 * 0.5
    total_correction = ambient_correction + altitude_correction + humidity_correction
    normalized_temp = raw_temp - total_correction
    correction_factor = 1 + total_correction / raw_temp if raw_temp != 0 else 1.0
    provenance_data = {"raw_temp": raw_temp, "ambient_temp": ambient_temp, "humidity": humidity, "altitude": altitude}
    return EnvironmentalNormalizationResult(float(raw_temp), float(ambient_temp), float(humidity), float(altitude), float(normalized_temp), float(correction_factor), _calculate_provenance_hash(provenance_data))


class ThermalNetworkModel:
    """Thermal network model for equipment temperature estimation."""
    def __init__(self, thermal_capacitance: float = 1000.0, thermal_resistance: float = 0.1, rated_temperature_rise: float = 65.0, ambient_temperature: float = 40.0):
        self.C = thermal_capacitance
        self.R = thermal_resistance
        self.tau = thermal_capacitance * thermal_resistance
        self.rated_rise = rated_temperature_rise
        self.ambient = ambient_temperature
        self.temperature = ambient_temperature

    def step(self, power_input: float, dt: float, ambient: Optional[float] = None) -> float:
        if ambient is not None:
            self.ambient = ambient
        T_final = self.ambient + power_input * self.R
        alpha = 1 - np.exp(-dt / self.tau)
        self.temperature = self.temperature + alpha * (T_final - self.temperature)
        return self.temperature

    def simulate(self, power_profile: NDArray[np.float64], dt: float, ambient_profile: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        temps = np.zeros(len(power_profile))
        for i, power in enumerate(power_profile):
            amb = ambient_profile[i] if ambient_profile is not None else None
            temps[i] = self.step(power, dt, amb)
        return temps


class ThermalProcessor:
    """Complete thermal signal processor."""
    def __init__(self, rated_rise: float = 65.0, alarm_threshold_ratio: float = 1.1, rate_alarm_threshold: float = 2.0):
        self.rated_rise = rated_rise
        self.alarm_threshold_ratio = alarm_threshold_ratio
        self.rate_alarm_threshold = rate_alarm_threshold

    def calculate_rise(self, measured: float, ambient: float) -> TemperatureRiseResult:
        return calculate_temperature_rise(measured, ambient, self.rated_rise, self.alarm_threshold_ratio)

    def calculate_rate(self, temps: NDArray[np.float64], timestamps: NDArray[np.float64], window: int = 5) -> TemperatureRateResult:
        return calculate_temperature_rate(temps, timestamps, window, self.rate_alarm_threshold)

    def accumulate_exposure(self, temps: NDArray[np.float64], timestamps: NDArray[np.float64], threshold: float) -> TimeAboveThresholdResult:
        return accumulate_time_above_threshold(temps, timestamps, threshold)

    def estimate_hotspot(self, top_oil: float, load: float, rated: float, ambient: float) -> HotspotEstimationResult:
        return estimate_hotspot_temperature(top_oil, load, rated, ambient)

    def normalize(self, raw: float, ambient: float, humidity: float = 50.0, altitude: float = 0.0) -> EnvironmentalNormalizationResult:
        return normalize_for_environment(raw, ambient, humidity, altitude)
