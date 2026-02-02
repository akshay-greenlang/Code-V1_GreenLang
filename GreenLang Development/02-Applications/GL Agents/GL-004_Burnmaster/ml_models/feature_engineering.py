# -*- coding: utf-8 -*-
"""
CombustionFeatureEngineer - Feature Engineering for Combustion Models

This module implements feature engineering pipelines for combustion
prediction models, including rolling statistics, lag features,
normalization, and feature selection.

Key Features:
    - Feature extraction from raw sensor data
    - Rolling window statistics (mean, std, min, max)
    - Lag feature generation for time-series
    - Normalization and scaling
    - Automatic feature selection

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProvenanceRecord(BaseModel):
    """Provenance tracking for audit trails."""
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    calculation_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str = Field(default="")
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    computation_time_ms: float = Field(default=0.0)

    @classmethod
    def create(cls, calculation_type: str, inputs: Dict, outputs: Dict,
               model_id: str = "", computation_time_ms: float = 0.0) -> "ProvenanceRecord":
        return cls(
            calculation_type=calculation_type, model_id=model_id,
            input_hash=hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            output_hash=hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(),
            computation_time_ms=computation_time_ms
        )


class FeatureMatrix(BaseModel):
    """Container for engineered feature matrix."""
    matrix_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    n_samples: int = Field(..., ge=0)
    n_features: int = Field(..., ge=0)
    feature_names: List[str] = Field(default_factory=list)
    original_features: List[str] = Field(default_factory=list)
    derived_features: List[str] = Field(default_factory=list)
    scaling_applied: str = Field(default="none")
    null_count: int = Field(default=0, ge=0)
    provenance: ProvenanceRecord


class CombustionFeatureEngineer:
    """
    Feature engineering pipeline for combustion prediction models.

    Implements comprehensive feature engineering including:
    1. Rolling window statistics for temporal patterns
    2. Lag features for time-series dependencies
    3. Normalization/scaling for model compatibility
    4. Automatic feature selection

    Example:
        >>> engineer = CombustionFeatureEngineer()
        >>> features = engineer.extract_features(raw_data)
        >>> rolling_features = engineer.compute_rolling_features(
        ...     raw_data, windows=[60, 300, 900]
        ... )
    """

    # Standard combustion features
    BASE_FEATURES = [
        'excess_air_percent', 'flame_temperature_c', 'fuel_flow_rate_kg_s',
        'air_flow_rate_kg_s', 'furnace_pressure_kpa', 'fuel_pressure_kpa',
        'air_preheat_temp_c', 'o2_stack_percent', 'co_ppm', 'nox_ppm',
        'stack_temp_c', 'load_percent'
    ]

    # Default rolling windows (in samples)
    DEFAULT_WINDOWS = [10, 30, 60, 300]  # ~10s, 30s, 1min, 5min at 1Hz

    # Default lag values
    DEFAULT_LAGS = [1, 5, 10, 30, 60]

    def __init__(
        self,
        base_features: Optional[List[str]] = None,
        scaler_type: str = "standard"
    ):
        """
        Initialize CombustionFeatureEngineer.

        Args:
            base_features: List of base feature names
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.base_features = base_features or self.BASE_FEATURES
        self.scaler_type = scaler_type
        self._model_id = f"feature_engineer_{uuid4().hex[:8]}"
        self._scaler = None
        self._fitted = False
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        self._selected_features: List[str] = []

        self._initialize_scaler()

        logger.info(f"CombustionFeatureEngineer initialized: scaler={scaler_type}")

    def _initialize_scaler(self) -> None:
        """Initialize the appropriate scaler."""
        if not SKLEARN_AVAILABLE:
            self._scaler = None
            return

        if self.scaler_type == "standard":
            self._scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self._scaler = MinMaxScaler()
        elif self.scaler_type == "robust":
            self._scaler = RobustScaler()
        else:
            self._scaler = StandardScaler()

    def extract_features(
        self,
        raw_data: pd.DataFrame,
        include_rolling: bool = True,
        include_lags: bool = True,
        rolling_windows: Optional[List[int]] = None,
        lags: Optional[List[int]] = None
    ) -> FeatureMatrix:
        """
        Extract features from raw sensor data.

        Performs comprehensive feature extraction including:
        - Base features from raw data
        - Rolling statistics (mean, std, min, max)
        - Lag features for time-series
        - Derived physics-based features

        Args:
            raw_data: Raw sensor DataFrame
            include_rolling: Include rolling window features
            include_lags: Include lag features
            rolling_windows: Custom rolling windows (samples)
            lags: Custom lag values (samples)

        Returns:
            FeatureMatrix with extracted features
        """
        start_time = time.time()

        # Start with base features
        available_base = [f for f in self.base_features if f in raw_data.columns]
        features_df = raw_data[available_base].copy()
        original_features = list(available_base)
        derived_features = []

        # Add derived physics-based features
        physics_features = self._compute_physics_features(raw_data)
        for name, values in physics_features.items():
            features_df[name] = values
            derived_features.append(name)

        # Add rolling features
        if include_rolling:
            windows = rolling_windows or self.DEFAULT_WINDOWS
            rolling_df = self.compute_rolling_features(
                raw_data[available_base], windows
            )
            for col in rolling_df.columns:
                if col not in features_df.columns:
                    features_df[col] = rolling_df[col]
                    derived_features.append(col)

        # Add lag features
        if include_lags:
            lag_values = lags or self.DEFAULT_LAGS
            lag_df = self.compute_lag_features(
                raw_data[available_base], lag_values
            )
            for col in lag_df.columns:
                if col not in features_df.columns:
                    features_df[col] = lag_df[col]
                    derived_features.append(col)

        # Fill NaN values
        null_count = int(features_df.isnull().sum().sum())
        features_df = features_df.fillna(method='ffill').fillna(0)

        computation_time_ms = (time.time() - start_time) * 1000

        return FeatureMatrix(
            n_samples=len(features_df),
            n_features=len(features_df.columns),
            feature_names=list(features_df.columns),
            original_features=original_features,
            derived_features=derived_features,
            scaling_applied="none",
            null_count=null_count,
            provenance=ProvenanceRecord.create(
                "feature_extraction",
                {"n_input_cols": len(raw_data.columns), "n_rows": len(raw_data)},
                {"n_features": len(features_df.columns)},
                self._model_id, computation_time_ms
            )
        )

    def compute_rolling_features(
        self,
        data: pd.DataFrame,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compute rolling window statistics for temporal patterns.

        Generates mean, std, min, max for each feature over multiple
        time windows to capture temporal dynamics.

        Args:
            data: Input DataFrame with time-series features
            windows: List of window sizes in samples

        Returns:
            DataFrame with rolling features
        """
        windows = windows or self.DEFAULT_WINDOWS
        rolling_features = {}

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            for window in windows:
                # Skip if window is larger than data
                if window >= len(data):
                    continue

                prefix = f"{col}_roll{window}"

                # Rolling mean
                rolling_features[f"{prefix}_mean"] = data[col].rolling(
                    window=window, min_periods=1
                ).mean()

                # Rolling std
                rolling_features[f"{prefix}_std"] = data[col].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)

                # Rolling min
                rolling_features[f"{prefix}_min"] = data[col].rolling(
                    window=window, min_periods=1
                ).min()

                # Rolling max
                rolling_features[f"{prefix}_max"] = data[col].rolling(
                    window=window, min_periods=1
                ).max()

                # Rolling range
                rolling_features[f"{prefix}_range"] = (
                    rolling_features[f"{prefix}_max"] -
                    rolling_features[f"{prefix}_min"]
                )

        return pd.DataFrame(rolling_features, index=data.index)

    def compute_lag_features(
        self,
        data: pd.DataFrame,
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compute lag features for time-series dependencies.

        Creates lagged versions of features to capture temporal
        dependencies in the data.

        Args:
            data: Input DataFrame with time-series features
            lags: List of lag values in samples

        Returns:
            DataFrame with lag features
        """
        lags = lags or self.DEFAULT_LAGS
        lag_features = {}

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            for lag in lags:
                if lag >= len(data):
                    continue

                # Lagged value
                lag_features[f"{col}_lag{lag}"] = data[col].shift(lag)

                # Change from lag
                lag_features[f"{col}_diff{lag}"] = data[col] - data[col].shift(lag)

                # Percent change from lag
                prev_val = data[col].shift(lag)
                pct_change = (data[col] - prev_val) / (prev_val.abs() + 1e-8)
                lag_features[f"{col}_pct{lag}"] = pct_change.clip(-10, 10)

        return pd.DataFrame(lag_features, index=data.index)

    def normalize_features(
        self,
        features: pd.DataFrame,
        scaler: str = "standard",
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize features using specified scaling method.

        Args:
            features: DataFrame with features to normalize
            scaler: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit the scaler or use existing

        Returns:
            Normalized DataFrame
        """
        if not SKLEARN_AVAILABLE:
            # Fallback: simple standardization
            mean = features.mean()
            std = features.std() + 1e-8
            return (features - mean) / std

        # Use or create scaler
        if scaler != self.scaler_type or self._scaler is None:
            self._initialize_scaler()
            self.scaler_type = scaler

        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features_numeric = features[numeric_cols].copy()

        if fit:
            scaled_values = self._scaler.fit_transform(features_numeric)
            self._fitted = True
        else:
            if not self._fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            scaled_values = self._scaler.transform(features_numeric)

        scaled_df = pd.DataFrame(
            scaled_values,
            index=features.index,
            columns=numeric_cols
        )

        # Add back non-numeric columns
        for col in features.columns:
            if col not in numeric_cols:
                scaled_df[col] = features[col]

        return scaled_df

    def select_features(
        self,
        features: pd.DataFrame,
        target: str,
        n_features: Optional[int] = None,
        method: str = "mutual_info"
    ) -> List[str]:
        """
        Select most relevant features for the target variable.

        Uses statistical methods to identify the most predictive
        features for the target.

        Args:
            features: DataFrame with candidate features
            target: Name of target column
            n_features: Number of features to select (default: auto)
            method: Selection method ('mutual_info', 'f_regression')

        Returns:
            List of selected feature names
        """
        if target not in features.columns:
            raise ValueError(f"Target '{target}' not found in features")

        # Separate features and target
        X = features.drop(columns=[target]).select_dtypes(include=[np.number])
        y = features[target]

        # Fill NaN
        X = X.fillna(0)
        y = y.fillna(y.mean())

        # Determine number of features
        if n_features is None:
            n_features = min(20, len(X.columns) // 2)

        if not SKLEARN_AVAILABLE:
            # Fallback: correlation-based selection
            correlations = X.corrwith(y).abs()
            selected = correlations.nlargest(n_features).index.tolist()
            self._selected_features = selected
            return selected

        # Use sklearn feature selection
        if method == "mutual_info":
            selector = SelectKBest(
                score_func=mutual_info_regression,
                k=min(n_features, len(X.columns))
            )
        else:
            selector = SelectKBest(
                score_func=f_regression,
                k=min(n_features, len(X.columns))
            )

        selector.fit(X, y)
        mask = selector.get_support()
        selected = X.columns[mask].tolist()

        # Store feature scores
        self._feature_stats = {
            X.columns[i]: float(selector.scores_[i])
            for i in range(len(X.columns))
        }

        self._selected_features = selected
        logger.info(f"Selected {len(selected)} features using {method}")

        return selected

    def _compute_physics_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute physics-based derived features.

        Creates features based on combustion physics relationships.
        """
        physics_features = {}

        # Air-fuel ratio
        if 'air_flow_rate_kg_s' in data.columns and 'fuel_flow_rate_kg_s' in data.columns:
            fuel = data['fuel_flow_rate_kg_s'].values
            air = data['air_flow_rate_kg_s'].values
            physics_features['air_fuel_ratio'] = air / (fuel + 1e-8)
            physics_features['stoich_ratio'] = air / (fuel + 1e-8) / 17.2  # Natural gas

        # Heat release rate (approximate)
        if 'fuel_flow_rate_kg_s' in data.columns:
            physics_features['heat_release_mw'] = data['fuel_flow_rate_kg_s'].values * 50.0

        # Excess O2 indicator
        if 'o2_stack_percent' in data.columns:
            o2 = data['o2_stack_percent'].values
            physics_features['excess_o2_ratio'] = o2 / (21.0 - o2 + 1e-8)

        # Temperature ratio
        if 'flame_temperature_c' in data.columns and 'stack_temp_c' in data.columns:
            flame_t = data['flame_temperature_c'].values + 273.15
            stack_t = data['stack_temp_c'].values + 273.15
            physics_features['temp_ratio'] = stack_t / (flame_t + 1e-8)

        # Combustion intensity
        if 'flame_temperature_c' in data.columns and 'excess_air_percent' in data.columns:
            temp = data['flame_temperature_c'].values
            ea = data['excess_air_percent'].values
            # Higher temp and lower excess air = more intense
            physics_features['combustion_intensity'] = temp / (ea + 10)

        # CO/O2 ratio (combustion quality indicator)
        if 'co_ppm' in data.columns and 'o2_stack_percent' in data.columns:
            co = data['co_ppm'].values
            o2 = data['o2_stack_percent'].values
            physics_features['co_o2_ratio'] = co / (o2 * 1000 + 1e-8)

        # Load-normalized fuel rate
        if 'fuel_flow_rate_kg_s' in data.columns and 'load_percent' in data.columns:
            fuel = data['fuel_flow_rate_kg_s'].values
            load = data['load_percent'].values
            physics_features['specific_fuel_rate'] = fuel / (load / 100 + 0.1)

        return physics_features

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from selection."""
        return self._feature_stats.copy()

    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self._selected_features.copy()

    def fit_transform(
        self,
        raw_data: pd.DataFrame,
        target: Optional[str] = None
    ) -> Tuple[pd.DataFrame, FeatureMatrix]:
        """
        Fit and transform in one step.

        Args:
            raw_data: Raw input data
            target: Optional target column for feature selection

        Returns:
            Tuple of (transformed DataFrame, FeatureMatrix metadata)
        """
        # Extract features
        feature_matrix = self.extract_features(raw_data)

        # Get feature names
        available_features = [f for f in feature_matrix.feature_names if f in raw_data.columns]

        # Subset data
        features_df = raw_data[available_features].copy()

        # Add derived features
        physics_features = self._compute_physics_features(raw_data)
        for name, values in physics_features.items():
            features_df[name] = values

        # Normalize
        features_df = self.normalize_features(features_df, fit=True)

        # Select features if target provided
        if target and target in raw_data.columns:
            features_df[target] = raw_data[target]
            selected = self.select_features(features_df, target)
            features_df = features_df[selected + [target]]
            feature_matrix.feature_names = selected

        return features_df, feature_matrix

    def save(self, path: Path) -> None:
        """Save feature engineer state."""
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self._scaler,
                'fitted': self._fitted,
                'feature_stats': self._feature_stats,
                'selected_features': self._selected_features,
                'base_features': self.base_features,
                'scaler_type': self.scaler_type
            }, f)

    def load(self, path: Path) -> None:
        """Load feature engineer state."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._scaler = data.get('scaler')
        self._fitted = data.get('fitted', False)
        self._feature_stats = data.get('feature_stats', {})
        self._selected_features = data.get('selected_features', [])
        self.base_features = data.get('base_features', self.BASE_FEATURES)
        self.scaler_type = data.get('scaler_type', 'standard')

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        return self._model_id

    @property
    def is_fitted(self) -> bool:
        """Check if feature engineer is fitted."""
        return self._fitted
