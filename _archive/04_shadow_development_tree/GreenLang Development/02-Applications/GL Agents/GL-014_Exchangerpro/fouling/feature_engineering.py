# -*- coding: utf-8 -*-
"""
Feature Engineering for Fouling Prediction - GL-014 Exchangerpro Agent.

Provides feature engineering specifically designed for heat exchanger fouling:
- Time since last cleaning
- Rolling statistics (mean, slope, variance) for temperatures, flows, delta-P
- Cumulative throughput and thermal stress
- Lagged features (1h, 6h, 24h, 7d windows)
- Flow-normalized delta-P trends
- Reynolds number proxies

All computations are deterministic with zero-hallucination guarantees.

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class RollingWindowConfig:
    """Configuration for rolling window statistics."""

    window_hours: int = 24
    min_periods: int = 6
    center: bool = False

    # Statistics to compute
    compute_mean: bool = True
    compute_std: bool = True
    compute_min: bool = True
    compute_max: bool = True
    compute_slope: bool = True
    compute_variance: bool = True


@dataclass
class LaggedFeatureConfig:
    """Configuration for lagged features."""

    # Lag windows in hours
    lag_windows_hours: List[int] = field(default_factory=lambda: [1, 6, 24, 168])

    # Features to lag
    features_to_lag: List[str] = field(default_factory=lambda: [
        "delta_p",
        "t_hot_in",
        "t_hot_out",
        "t_cold_in",
        "t_cold_out",
        "flow_hot",
        "flow_cold",
        "ua_current",
    ])

    # Difference features (current - lagged)
    compute_differences: bool = True


@dataclass
class FoulingFeatureConfig:
    """Configuration for fouling feature engineering."""

    # Time features
    include_time_since_cleaning: bool = True

    # Rolling statistics
    rolling_config: RollingWindowConfig = field(default_factory=RollingWindowConfig)
    rolling_windows_hours: List[int] = field(default_factory=lambda: [6, 24, 72, 168])

    # Lagged features
    lagged_config: LaggedFeatureConfig = field(default_factory=LaggedFeatureConfig)

    # Cumulative features
    include_cumulative_throughput: bool = True
    include_cumulative_thermal_stress: bool = True

    # Flow-normalized features
    include_flow_normalized_delta_p: bool = True

    # Reynolds number proxy
    include_reynolds_proxy: bool = True

    # LMTD features
    include_lmtd_features: bool = True

    # Fouling resistance features
    include_rf_features: bool = True

    # Sample rate in seconds
    sample_rate_seconds: float = 60.0


# =============================================================================
# Feature Data Classes
# =============================================================================

@dataclass
class FoulingFeatures:
    """Complete feature set for fouling prediction."""

    exchanger_id: str
    timestamp: datetime

    # Raw measurements (input)
    delta_p: float  # Pressure drop (kPa)
    t_hot_in: float  # Hot inlet temperature (C)
    t_hot_out: float  # Hot outlet temperature (C)
    t_cold_in: float  # Cold inlet temperature (C)
    t_cold_out: float  # Cold outlet temperature (C)
    flow_hot: float  # Hot side flow rate (kg/s)
    flow_cold: float  # Cold side flow rate (kg/s)

    # Current performance
    ua_current: float  # Current overall heat transfer coefficient * area
    ua_clean: float  # UA when clean (baseline)

    # Time features
    time_since_cleaning_hours: float = 0.0
    operating_hours: float = 0.0

    # Rolling statistics (keyed by window size)
    rolling_features: Dict[str, float] = field(default_factory=dict)

    # Lagged features
    lagged_features: Dict[str, float] = field(default_factory=dict)

    # Cumulative features
    cumulative_throughput_kg: float = 0.0
    cumulative_thermal_stress: float = 0.0
    cumulative_delta_p_hours: float = 0.0

    # Flow-normalized features
    delta_p_flow_normalized: float = 0.0
    delta_p_flow_normalized_trend: float = 0.0

    # Reynolds number proxy
    reynolds_proxy_hot: float = 0.0
    reynolds_proxy_cold: float = 0.0

    # LMTD features
    lmtd: float = 0.0
    lmtd_correction_factor: float = 1.0

    # Fouling resistance
    rf_estimated: float = 0.0  # m2.K/W
    rf_rate_of_change: float = 0.0  # per hour

    # Quality indicators
    data_quality_score: float = 1.0
    n_missing_values: int = 0

    # Provenance
    feature_version: str = "1.0.0"
    provenance_hash: str = ""
    computation_time_ms: float = 0.0

    def to_array(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert features to numpy array for ML models.

        Args:
            feature_names: Specific features to include (default: all numeric)

        Returns:
            numpy array of feature values
        """
        if feature_names is None:
            feature_names = self.get_feature_names()

        values = []
        for name in feature_names:
            if hasattr(self, name):
                values.append(getattr(self, name))
            elif name in self.rolling_features:
                values.append(self.rolling_features[name])
            elif name in self.lagged_features:
                values.append(self.lagged_features[name])
            else:
                values.append(0.0)

        return np.array(values, dtype=np.float64)

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        base_features = [
            "delta_p", "t_hot_in", "t_hot_out", "t_cold_in", "t_cold_out",
            "flow_hot", "flow_cold", "ua_current", "ua_clean",
            "time_since_cleaning_hours", "operating_hours",
            "cumulative_throughput_kg", "cumulative_thermal_stress",
            "cumulative_delta_p_hours", "delta_p_flow_normalized",
            "delta_p_flow_normalized_trend", "reynolds_proxy_hot",
            "reynolds_proxy_cold", "lmtd", "lmtd_correction_factor",
            "rf_estimated", "rf_rate_of_change",
        ]

        all_features = base_features.copy()
        all_features.extend(list(self.rolling_features.keys()))
        all_features.extend(list(self.lagged_features.keys()))

        return all_features


# =============================================================================
# Feature Engine
# =============================================================================

class FoulingFeatureEngine:
    """
    Feature engineering engine for heat exchanger fouling prediction.

    Computes comprehensive feature set including:
    - Rolling statistics for key process variables
    - Lagged features at multiple time horizons
    - Cumulative metrics (throughput, thermal stress)
    - Flow-normalized pressure drop trends
    - Reynolds number proxies
    - LMTD-based effectiveness features
    - Fouling resistance estimates

    All calculations are deterministic with complete provenance tracking.

    Example:
        >>> engine = FoulingFeatureEngine(config)
        >>> features = engine.compute_features(
        ...     exchanger_id="HX-001",
        ...     current_data=current_reading,
        ...     historical_data=history_df,
        ...     last_cleaning_time=last_clean
        ... )
        >>> X = features.to_array()
    """

    def __init__(self, config: Optional[FoulingFeatureConfig] = None):
        """
        Initialize feature engineering engine.

        Args:
            config: Feature engineering configuration
        """
        self.config = config or FoulingFeatureConfig()
        self._feature_version = "1.0.0"
        self._history_cache: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(
            f"FoulingFeatureEngine initialized with "
            f"rolling_windows={self.config.rolling_windows_hours}, "
            f"lag_windows={self.config.lagged_config.lag_windows_hours}"
        )

    def compute_features(
        self,
        exchanger_id: str,
        current_data: Dict[str, float],
        historical_data: Optional[Any] = None,  # pd.DataFrame or List[Dict]
        last_cleaning_time: Optional[datetime] = None,
        ua_clean: Optional[float] = None,
    ) -> FoulingFeatures:
        """
        Compute complete feature set for fouling prediction.

        Args:
            exchanger_id: Heat exchanger identifier
            current_data: Current sensor readings as dict with keys:
                - delta_p, t_hot_in, t_hot_out, t_cold_in, t_cold_out
                - flow_hot, flow_cold, ua_current, timestamp
            historical_data: Historical data (DataFrame or list of dicts)
            last_cleaning_time: Last cleaning timestamp
            ua_clean: UA value when clean (baseline)

        Returns:
            FoulingFeatures with all computed features
        """
        start_time = time.time()

        timestamp = current_data.get("timestamp", datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Extract raw measurements
        delta_p = float(current_data.get("delta_p", 0.0))
        t_hot_in = float(current_data.get("t_hot_in", 0.0))
        t_hot_out = float(current_data.get("t_hot_out", 0.0))
        t_cold_in = float(current_data.get("t_cold_in", 0.0))
        t_cold_out = float(current_data.get("t_cold_out", 0.0))
        flow_hot = float(current_data.get("flow_hot", 0.0))
        flow_cold = float(current_data.get("flow_cold", 0.0))
        ua_current = float(current_data.get("ua_current", 0.0))

        # Set UA clean baseline
        if ua_clean is None:
            ua_clean = float(current_data.get("ua_clean", ua_current * 1.2))

        # Initialize features
        features = FoulingFeatures(
            exchanger_id=exchanger_id,
            timestamp=timestamp,
            delta_p=delta_p,
            t_hot_in=t_hot_in,
            t_hot_out=t_hot_out,
            t_cold_in=t_cold_in,
            t_cold_out=t_cold_out,
            flow_hot=flow_hot,
            flow_cold=flow_cold,
            ua_current=ua_current,
            ua_clean=ua_clean,
            feature_version=self._feature_version,
        )

        # Time since cleaning
        if self.config.include_time_since_cleaning and last_cleaning_time:
            features.time_since_cleaning_hours = self._compute_time_since_cleaning(
                timestamp, last_cleaning_time
            )

        # Convert historical data to consistent format
        history_list = self._normalize_historical_data(historical_data)

        # Update cache with current data
        self._update_history_cache(exchanger_id, current_data)

        # Merge cache with provided history
        combined_history = self._merge_history(exchanger_id, history_list)

        # Rolling statistics
        if combined_history:
            features.rolling_features = self._compute_rolling_statistics(
                combined_history
            )

        # Lagged features
        if combined_history:
            features.lagged_features = self._compute_lagged_features(
                combined_history, timestamp
            )

        # Cumulative features
        if self.config.include_cumulative_throughput and combined_history:
            features.cumulative_throughput_kg = self._compute_cumulative_throughput(
                combined_history
            )

        if self.config.include_cumulative_thermal_stress and combined_history:
            features.cumulative_thermal_stress = self._compute_cumulative_thermal_stress(
                combined_history
            )
            features.cumulative_delta_p_hours = self._compute_cumulative_delta_p(
                combined_history
            )

        # Flow-normalized delta-P
        if self.config.include_flow_normalized_delta_p:
            features.delta_p_flow_normalized = self._compute_flow_normalized_delta_p(
                delta_p, flow_hot, flow_cold
            )
            if combined_history:
                features.delta_p_flow_normalized_trend = self._compute_delta_p_trend(
                    combined_history
                )

        # Reynolds number proxy
        if self.config.include_reynolds_proxy:
            features.reynolds_proxy_hot, features.reynolds_proxy_cold = \
                self._compute_reynolds_proxies(
                    flow_hot, flow_cold, t_hot_in, t_cold_in
                )

        # LMTD features
        if self.config.include_lmtd_features:
            features.lmtd = self._compute_lmtd(
                t_hot_in, t_hot_out, t_cold_in, t_cold_out
            )

        # Fouling resistance
        if self.config.include_rf_features:
            features.rf_estimated = self._estimate_fouling_resistance(
                ua_current, ua_clean
            )
            if combined_history:
                features.rf_rate_of_change = self._compute_rf_rate(combined_history)

        # Operating hours
        if combined_history:
            features.operating_hours = self._compute_operating_hours(combined_history)

        # Data quality
        features.data_quality_score, features.n_missing_values = \
            self._assess_data_quality(current_data)

        # Compute provenance hash
        computation_time = (time.time() - start_time) * 1000
        features.computation_time_ms = computation_time
        features.provenance_hash = self._compute_provenance_hash(
            exchanger_id, timestamp, features
        )

        logger.debug(
            f"Computed {len(features.rolling_features) + len(features.lagged_features)} "
            f"features for {exchanger_id} in {computation_time:.1f}ms"
        )

        return features

    def compute_batch_features(
        self,
        exchanger_id: str,
        data: Any,  # pd.DataFrame
        last_cleaning_time: Optional[datetime] = None,
        ua_clean: Optional[float] = None,
    ) -> List[FoulingFeatures]:
        """
        Compute features for a batch of data points.

        Args:
            exchanger_id: Heat exchanger identifier
            data: DataFrame with time-indexed sensor data
            last_cleaning_time: Last cleaning timestamp
            ua_clean: UA value when clean

        Returns:
            List of FoulingFeatures for each row
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for batch feature computation")

        results = []
        history_so_far = []

        for idx, row in data.iterrows():
            current_data = row.to_dict()
            current_data["timestamp"] = idx

            features = self.compute_features(
                exchanger_id=exchanger_id,
                current_data=current_data,
                historical_data=history_so_far.copy(),
                last_cleaning_time=last_cleaning_time,
                ua_clean=ua_clean,
            )

            results.append(features)
            history_so_far.append(current_data)

        return results

    # =========================================================================
    # Time Features
    # =========================================================================

    def _compute_time_since_cleaning(
        self,
        current_time: datetime,
        last_cleaning: datetime,
    ) -> float:
        """Compute hours since last cleaning."""
        if last_cleaning is None:
            return 0.0

        delta = current_time - last_cleaning
        return max(0.0, delta.total_seconds() / 3600.0)

    def _compute_operating_hours(self, history: List[Dict]) -> float:
        """Compute total operating hours from history."""
        if not history:
            return 0.0

        # Estimate from sample count and sample rate
        n_samples = len(history)
        sample_rate_hours = self.config.sample_rate_seconds / 3600.0

        return n_samples * sample_rate_hours

    # =========================================================================
    # Rolling Statistics
    # =========================================================================

    def _compute_rolling_statistics(
        self,
        history: List[Dict],
    ) -> Dict[str, float]:
        """
        Compute rolling statistics for key variables.

        Returns dict with keys like:
        - delta_p_mean_24h, delta_p_std_24h, delta_p_slope_24h
        - t_hot_in_mean_6h, etc.
        """
        rolling_features = {}

        if not history:
            return rolling_features

        # Variables to compute rolling stats for
        variables = [
            "delta_p", "t_hot_in", "t_hot_out", "t_cold_in", "t_cold_out",
            "flow_hot", "flow_cold", "ua_current"
        ]

        for window_hours in self.config.rolling_windows_hours:
            # Get samples within window
            window_samples = self._get_samples_in_window(history, window_hours)

            if len(window_samples) < self.config.rolling_config.min_periods:
                continue

            for var in variables:
                values = [s.get(var, 0.0) for s in window_samples]
                values = [v for v in values if v is not None and not np.isnan(v)]

                if not values:
                    continue

                values_arr = np.array(values)

                suffix = f"_{window_hours}h"

                if self.config.rolling_config.compute_mean:
                    rolling_features[f"{var}_mean{suffix}"] = float(np.mean(values_arr))

                if self.config.rolling_config.compute_std and len(values_arr) > 1:
                    rolling_features[f"{var}_std{suffix}"] = float(np.std(values_arr))

                if self.config.rolling_config.compute_min:
                    rolling_features[f"{var}_min{suffix}"] = float(np.min(values_arr))

                if self.config.rolling_config.compute_max:
                    rolling_features[f"{var}_max{suffix}"] = float(np.max(values_arr))

                if self.config.rolling_config.compute_variance and len(values_arr) > 1:
                    rolling_features[f"{var}_var{suffix}"] = float(np.var(values_arr))

                if self.config.rolling_config.compute_slope and len(values_arr) >= 3:
                    slope = self._compute_slope(values_arr)
                    rolling_features[f"{var}_slope{suffix}"] = slope

        return rolling_features

    def _get_samples_in_window(
        self,
        history: List[Dict],
        window_hours: int,
    ) -> List[Dict]:
        """Get samples within the specified time window."""
        if not history:
            return []

        # Calculate number of samples in window
        samples_per_hour = 3600.0 / self.config.sample_rate_seconds
        n_samples = int(window_hours * samples_per_hour)

        # Return most recent samples
        return history[-n_samples:]

    def _compute_slope(self, values: np.ndarray) -> float:
        """Compute linear slope of values over time."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        n = len(values)

        # Simple linear regression slope
        x_mean = np.mean(x)
        y_mean = np.mean(values)

        numerator = np.sum((x - x_mean) * (values - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator < 1e-10:
            return 0.0

        return float(numerator / denominator)

    # =========================================================================
    # Lagged Features
    # =========================================================================

    def _compute_lagged_features(
        self,
        history: List[Dict],
        current_time: datetime,
    ) -> Dict[str, float]:
        """
        Compute lagged features at specified time horizons.

        Returns dict with keys like:
        - delta_p_lag_1h, delta_p_lag_6h, delta_p_lag_24h, delta_p_lag_168h
        - delta_p_diff_1h (current - lagged value)
        """
        lagged_features = {}

        if not history:
            return lagged_features

        config = self.config.lagged_config

        for lag_hours in config.lag_windows_hours:
            # Calculate index offset
            samples_per_hour = 3600.0 / self.config.sample_rate_seconds
            lag_index = int(lag_hours * samples_per_hour)

            if lag_index >= len(history):
                continue

            lagged_sample = history[-(lag_index + 1)]
            current_sample = history[-1]

            for var in config.features_to_lag:
                lagged_value = lagged_sample.get(var)
                current_value = current_sample.get(var)

                if lagged_value is not None and not np.isnan(lagged_value):
                    suffix = f"_lag_{lag_hours}h"
                    lagged_features[f"{var}{suffix}"] = float(lagged_value)

                    # Compute difference
                    if config.compute_differences and current_value is not None:
                        diff = current_value - lagged_value
                        lagged_features[f"{var}_diff_{lag_hours}h"] = float(diff)

        return lagged_features

    # =========================================================================
    # Cumulative Features
    # =========================================================================

    def _compute_cumulative_throughput(self, history: List[Dict]) -> float:
        """
        Compute cumulative mass throughput since last cleaning.

        Returns:
            Total mass flow in kg
        """
        if not history:
            return 0.0

        total_throughput = 0.0
        dt_hours = self.config.sample_rate_seconds / 3600.0

        for sample in history:
            flow_hot = sample.get("flow_hot", 0.0) or 0.0
            flow_cold = sample.get("flow_cold", 0.0) or 0.0

            # Average flow * time (kg/s * hours * 3600 = kg)
            avg_flow = (flow_hot + flow_cold) / 2.0
            total_throughput += avg_flow * dt_hours * 3600.0

        return float(total_throughput)

    def _compute_cumulative_thermal_stress(self, history: List[Dict]) -> float:
        """
        Compute cumulative thermal stress.

        Thermal stress = integral of |T_hot - T_cold| * time
        """
        if not history:
            return 0.0

        total_stress = 0.0
        dt_hours = self.config.sample_rate_seconds / 3600.0

        for sample in history:
            t_hot_in = sample.get("t_hot_in", 0.0) or 0.0
            t_cold_in = sample.get("t_cold_in", 0.0) or 0.0

            temp_diff = abs(t_hot_in - t_cold_in)
            total_stress += temp_diff * dt_hours

        return float(total_stress)

    def _compute_cumulative_delta_p(self, history: List[Dict]) -> float:
        """Compute cumulative pressure drop * time (delta-P hours)."""
        if not history:
            return 0.0

        total = 0.0
        dt_hours = self.config.sample_rate_seconds / 3600.0

        for sample in history:
            delta_p = sample.get("delta_p", 0.0) or 0.0
            total += delta_p * dt_hours

        return float(total)

    # =========================================================================
    # Flow-Normalized Features
    # =========================================================================

    def _compute_flow_normalized_delta_p(
        self,
        delta_p: float,
        flow_hot: float,
        flow_cold: float,
    ) -> float:
        """
        Compute flow-normalized pressure drop.

        delta_P / (flow^2) normalizes for flow rate effects.
        Increasing values indicate fouling.
        """
        avg_flow = (flow_hot + flow_cold) / 2.0

        if avg_flow < 1e-6:
            return 0.0

        # Normalize by flow squared (pressure drop ~ flow^2)
        return float(delta_p / (avg_flow ** 2))

    def _compute_delta_p_trend(self, history: List[Dict]) -> float:
        """
        Compute trend in flow-normalized delta-P.

        Positive slope indicates fouling progression.
        """
        if len(history) < 10:
            return 0.0

        # Compute flow-normalized delta-P for each sample
        normalized_values = []
        for sample in history[-168:]:  # Last week
            delta_p = sample.get("delta_p", 0.0) or 0.0
            flow_hot = sample.get("flow_hot", 1e-6) or 1e-6
            flow_cold = sample.get("flow_cold", 1e-6) or 1e-6

            avg_flow = (flow_hot + flow_cold) / 2.0
            if avg_flow > 1e-6:
                normalized_values.append(delta_p / (avg_flow ** 2))

        if len(normalized_values) < 3:
            return 0.0

        return self._compute_slope(np.array(normalized_values))

    # =========================================================================
    # Reynolds Number Proxy
    # =========================================================================

    def _compute_reynolds_proxies(
        self,
        flow_hot: float,
        flow_cold: float,
        t_hot: float,
        t_cold: float,
    ) -> Tuple[float, float]:
        """
        Compute Reynolds number proxies for hot and cold sides.

        Re = (rho * v * D) / mu

        Since we don't have all parameters, we use a proxy:
        Re_proxy = flow / viscosity_proxy(T)

        Higher Re = more turbulent = better heat transfer but more erosion.
        """
        # Simplified viscosity model (water-like, relative to 20C)
        def viscosity_ratio(t):
            """Temperature-dependent viscosity ratio (relative to 20C)."""
            # Approximate: viscosity decreases with temperature
            t_ref = 20.0
            return np.exp(-0.02 * (t - t_ref))

        # Reynolds proxy = flow / viscosity
        re_hot = flow_hot / max(viscosity_ratio(t_hot), 1e-6)
        re_cold = flow_cold / max(viscosity_ratio(t_cold), 1e-6)

        return float(re_hot), float(re_cold)

    # =========================================================================
    # LMTD Features
    # =========================================================================

    def _compute_lmtd(
        self,
        t_hot_in: float,
        t_hot_out: float,
        t_cold_in: float,
        t_cold_out: float,
    ) -> float:
        """
        Compute Log Mean Temperature Difference (LMTD).

        For counterflow:
        LMTD = (dT1 - dT2) / ln(dT1 / dT2)

        where:
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in
        """
        dt1 = t_hot_in - t_cold_out
        dt2 = t_hot_out - t_cold_in

        # Handle edge cases
        if dt1 <= 0 or dt2 <= 0:
            return 0.0

        if abs(dt1 - dt2) < 1e-6:
            # When dT1 ~ dT2, LMTD ~ dT1
            return float(dt1)

        try:
            lmtd = (dt1 - dt2) / np.log(dt1 / dt2)
            return float(lmtd)
        except (ValueError, ZeroDivisionError):
            return 0.0

    # =========================================================================
    # Fouling Resistance Features
    # =========================================================================

    def _estimate_fouling_resistance(
        self,
        ua_current: float,
        ua_clean: float,
    ) -> float:
        """
        Estimate fouling resistance Rf from UA values.

        1/UA = 1/UA_clean + Rf/A

        Assuming A is constant:
        Rf ~ A * (1/UA_current - 1/UA_clean)

        We report Rf normalized by area (m2.K/W).
        """
        if ua_current <= 0 or ua_clean <= 0:
            return 0.0

        # Estimate Rf (assuming unit area for normalization)
        rf = (1.0 / ua_current) - (1.0 / ua_clean)

        return float(max(0.0, rf))

    def _compute_rf_rate(self, history: List[Dict]) -> float:
        """
        Compute rate of change of fouling resistance.

        Returns:
            Rf rate of change per hour (positive = increasing fouling)
        """
        if len(history) < 10:
            return 0.0

        # Get UA values from history
        ua_values = []
        for sample in history[-168:]:  # Last week
            ua = sample.get("ua_current")
            if ua is not None and ua > 0:
                ua_values.append(ua)

        if len(ua_values) < 3:
            return 0.0

        # Compute Rf for each (using first UA as reference clean value)
        ua_clean = max(ua_values)  # Best UA as reference
        rf_values = []
        for ua in ua_values:
            if ua > 0:
                rf = (1.0 / ua) - (1.0 / ua_clean)
                rf_values.append(max(0.0, rf))

        if len(rf_values) < 3:
            return 0.0

        # Compute slope (change per sample)
        slope_per_sample = self._compute_slope(np.array(rf_values))

        # Convert to per hour
        samples_per_hour = 3600.0 / self.config.sample_rate_seconds
        return float(slope_per_sample * samples_per_hour)

    # =========================================================================
    # Data Quality
    # =========================================================================

    def _assess_data_quality(
        self,
        data: Dict[str, Any],
    ) -> Tuple[float, int]:
        """
        Assess data quality.

        Returns:
            Tuple of (quality_score, n_missing)
        """
        required_fields = [
            "delta_p", "t_hot_in", "t_hot_out", "t_cold_in", "t_cold_out",
            "flow_hot", "flow_cold", "ua_current"
        ]

        n_missing = 0
        for field in required_fields:
            value = data.get(field)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                n_missing += 1

        quality_score = 1.0 - (n_missing / len(required_fields))

        return float(quality_score), n_missing

    # =========================================================================
    # History Management
    # =========================================================================

    def _normalize_historical_data(
        self,
        data: Any,
    ) -> List[Dict[str, Any]]:
        """Convert historical data to list of dicts."""
        if data is None:
            return []

        if isinstance(data, list):
            return data

        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            records = data.to_dict("records")
            # Add timestamp from index if datetime index
            if isinstance(data.index, pd.DatetimeIndex):
                for i, record in enumerate(records):
                    record["timestamp"] = data.index[i]
            return records

        return []

    def _update_history_cache(
        self,
        exchanger_id: str,
        current_data: Dict[str, Any],
    ) -> None:
        """Update internal history cache."""
        if exchanger_id not in self._history_cache:
            self._history_cache[exchanger_id] = []

        self._history_cache[exchanger_id].append(current_data.copy())

        # Limit cache size (keep last 7 days)
        max_samples = int(168 * 3600 / self.config.sample_rate_seconds)
        if len(self._history_cache[exchanger_id]) > max_samples:
            self._history_cache[exchanger_id] = \
                self._history_cache[exchanger_id][-max_samples:]

    def _merge_history(
        self,
        exchanger_id: str,
        provided_history: List[Dict],
    ) -> List[Dict]:
        """Merge cache with provided history."""
        cached = self._history_cache.get(exchanger_id, [])

        if not provided_history:
            return cached

        if not cached:
            return provided_history

        # Merge, preferring provided history
        # Simple approach: concatenate and take most recent
        combined = provided_history + cached
        max_samples = int(168 * 3600 / self.config.sample_rate_seconds)
        return combined[-max_samples:]

    def clear_cache(self, exchanger_id: Optional[str] = None) -> None:
        """Clear history cache."""
        if exchanger_id:
            self._history_cache.pop(exchanger_id, None)
        else:
            self._history_cache.clear()

    # =========================================================================
    # Provenance
    # =========================================================================

    def _compute_provenance_hash(
        self,
        exchanger_id: str,
        timestamp: datetime,
        features: FoulingFeatures,
    ) -> str:
        """Compute SHA-256 provenance hash for features."""
        content = (
            f"{exchanger_id}|{timestamp.isoformat()}|"
            f"{features.delta_p:.6f}|{features.ua_current:.6f}|"
            f"{features.time_since_cleaning_hours:.2f}|"
            f"{len(features.rolling_features)}|{len(features.lagged_features)}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get_feature_schema(self) -> Dict[str, Any]:
        """
        Get feature schema for documentation and validation.

        Returns:
            Dictionary describing all features
        """
        schema = {
            "version": self._feature_version,
            "base_features": [
                {"name": "delta_p", "type": "float", "unit": "kPa", "description": "Pressure drop"},
                {"name": "t_hot_in", "type": "float", "unit": "C", "description": "Hot inlet temperature"},
                {"name": "t_hot_out", "type": "float", "unit": "C", "description": "Hot outlet temperature"},
                {"name": "t_cold_in", "type": "float", "unit": "C", "description": "Cold inlet temperature"},
                {"name": "t_cold_out", "type": "float", "unit": "C", "description": "Cold outlet temperature"},
                {"name": "flow_hot", "type": "float", "unit": "kg/s", "description": "Hot side flow rate"},
                {"name": "flow_cold", "type": "float", "unit": "kg/s", "description": "Cold side flow rate"},
                {"name": "ua_current", "type": "float", "unit": "W/K", "description": "Current UA value"},
                {"name": "ua_clean", "type": "float", "unit": "W/K", "description": "Clean UA baseline"},
            ],
            "time_features": [
                {"name": "time_since_cleaning_hours", "type": "float", "unit": "hours"},
                {"name": "operating_hours", "type": "float", "unit": "hours"},
            ],
            "rolling_windows_hours": self.config.rolling_windows_hours,
            "lag_windows_hours": self.config.lagged_config.lag_windows_hours,
            "cumulative_features": [
                "cumulative_throughput_kg",
                "cumulative_thermal_stress",
                "cumulative_delta_p_hours",
            ],
            "derived_features": [
                "delta_p_flow_normalized",
                "delta_p_flow_normalized_trend",
                "reynolds_proxy_hot",
                "reynolds_proxy_cold",
                "lmtd",
                "rf_estimated",
                "rf_rate_of_change",
            ],
        }

        return schema
