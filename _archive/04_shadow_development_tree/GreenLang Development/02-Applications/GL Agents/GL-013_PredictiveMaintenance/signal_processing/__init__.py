"""
GL-013 PredictiveMaintenance Signal Processing Module

Zero-Hallucination Signal Processing for Industrial Equipment Health Monitoring
Author: GL-013 PredictiveMaintenance Agent
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang GL-013 PredictiveMaintenance"

from .vibration import (
    VibrationProcessor,
    compute_fft,
    envelope_analysis,
    extract_spectral_features,
    calculate_bearing_frequencies,
    TriaxialAcceleration,
)

from .mcsa import (
    MCSAProcessor,
    compute_current_fft,
    detect_rotor_bar_sidebands,
    detect_eccentricity,
    detect_stator_faults,
    calculate_current_imbalance,
    ThreePhaseCurrents,
)

from .thermal import (
    ThermalProcessor,
    calculate_temperature_rise,
    calculate_temperature_rate,
    accumulate_time_above_threshold,
    estimate_hotspot_temperature,
    normalize_for_environment,
    ThermalNetworkModel,
)

from .feature_store import (
    FeatureStore,
    FeatureDefinition,
    FeatureValue,
    FeatureLineage,
    DataQualityFlag,
)

__all__ = [
    "__version__",
    "VibrationProcessor",
    "compute_fft",
    "envelope_analysis",
    "extract_spectral_features",
    "calculate_bearing_frequencies",
    "TriaxialAcceleration",
    "MCSAProcessor",
    "compute_current_fft",
    "detect_rotor_bar_sidebands",
    "detect_eccentricity",
    "detect_stator_faults",
    "calculate_current_imbalance",
    "ThreePhaseCurrents",
    "ThermalProcessor",
    "calculate_temperature_rise",
    "calculate_temperature_rate",
    "accumulate_time_above_threshold",
    "estimate_hotspot_temperature",
    "normalize_for_environment",
    "ThermalNetworkModel",
    "FeatureStore",
    "FeatureDefinition",
    "FeatureValue",
    "FeatureLineage",
    "DataQualityFlag",
]
