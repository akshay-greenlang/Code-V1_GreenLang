# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Fugitive Emissions Feature Engineering

This module provides deterministic feature engineering for fugitive emissions
detection ML models, including meteorological features, sensor statistics,
and operational context features.

Zero-Hallucination Principle:
    - All feature calculations are deterministic
    - Complete provenance tracking
    - No LLM involvement in feature computation

Author: GreenLang GL-010 EmissionsGuardian
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import hashlib
import logging
import math
import statistics

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WindStability(str, Enum):
    """Pasquill-Gifford atmospheric stability classes."""
    A = "A"  # Very unstable
    B = "B"  # Moderately unstable
    C = "C"  # Slightly unstable
    D = "D"  # Neutral
    E = "E"  # Slightly stable
    F = "F"  # Moderately stable


class EquipmentType(str, Enum):
    """Equipment types for fugitive emission sources."""
    VALVE = "valve"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    FLANGE = "flange"
    CONNECTOR = "connector"
    PRESSURE_RELIEF = "pressure_relief"
    OPEN_ENDED_LINE = "open_ended_line"
    TANK = "tank"
    COOLING_TOWER = "cooling_tower"
    UNKNOWN = "unknown"


class SensorReading(BaseModel):
    """Raw sensor reading for fugitive detection."""
    sensor_id: str = Field(...)
    timestamp: datetime = Field(...)
    concentration_ppm: Decimal = Field(...)
    latitude: Optional[float] = Field(None)
    longitude: Optional[float] = Field(None)
    height_m: Optional[float] = Field(None)
    quality_flag: str = Field(default="valid")


class MeteorologicalData(BaseModel):
    """Meteorological data for plume analysis."""
    timestamp: datetime = Field(...)
    wind_speed_m_s: Decimal = Field(..., ge=0)
    wind_direction_deg: Decimal = Field(..., ge=0, le=360)
    temperature_c: Decimal = Field(...)
    relative_humidity_pct: Optional[Decimal] = Field(None, ge=0, le=100)
    pressure_hpa: Optional[Decimal] = Field(None)
    solar_radiation_w_m2: Optional[Decimal] = Field(None, ge=0)
    cloud_cover_pct: Optional[Decimal] = Field(None, ge=0, le=100)
    stability_class: Optional[WindStability] = Field(None)
    mixing_height_m: Optional[Decimal] = Field(None, ge=0)


class EquipmentContext(BaseModel):
    """Equipment context for leak detection."""
    equipment_id: str = Field(...)
    equipment_type: EquipmentType = Field(...)
    latitude: float = Field(...)
    longitude: float = Field(...)
    service_type: Optional[str] = Field(None)  # Gas, liquid, etc.
    operating_pressure_psi: Optional[Decimal] = Field(None)
    operating_temp_c: Optional[Decimal] = Field(None)
    last_inspection_date: Optional[datetime] = Field(None)
    last_repair_date: Optional[datetime] = Field(None)
    leak_history_count: int = Field(default=0)


@dataclass
class FeatureVector:
    """Feature vector for ML model input."""
    feature_id: str
    timestamp: datetime

    # Sensor features
    concentration_current: float
    concentration_mean: float
    concentration_std: float
    concentration_max: float
    concentration_min: float
    concentration_range: float
    concentration_zscore: float
    concentration_rate_of_change: float

    # Meteorological features
    wind_speed: float
    wind_direction: float
    wind_direction_sin: float
    wind_direction_cos: float
    temperature: float
    stability_class_encoded: int

    # Spatial features
    upwind_downwind_diff: float
    background_concentration: float
    elevation_above_background: float

    # Temporal features
    hour_of_day: int
    day_of_week: int
    is_daytime: bool

    # Equipment features
    equipment_type_encoded: int
    days_since_inspection: float
    leak_history_count: int
    operating_pressure: float

    # Derived features
    plume_likelihood_score: float
    spatial_anomaly_score: float
    temporal_anomaly_score: float

    # Provenance
    source_sensor_ids: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_array(self) -> List[float]:
        """Convert to numeric array for ML model."""
        return [
            self.concentration_current,
            self.concentration_mean,
            self.concentration_std,
            self.concentration_max,
            self.concentration_min,
            self.concentration_range,
            self.concentration_zscore,
            self.concentration_rate_of_change,
            self.wind_speed,
            self.wind_direction_sin,
            self.wind_direction_cos,
            self.temperature,
            float(self.stability_class_encoded),
            self.upwind_downwind_diff,
            self.background_concentration,
            self.elevation_above_background,
            float(self.hour_of_day),
            float(self.day_of_week),
            float(self.is_daytime),
            float(self.equipment_type_encoded),
            self.days_since_inspection,
            float(self.leak_history_count),
            self.operating_pressure,
            self.plume_likelihood_score,
            self.spatial_anomaly_score,
            self.temporal_anomaly_score,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for explainability."""
        return [
            "concentration_current",
            "concentration_mean",
            "concentration_std",
            "concentration_max",
            "concentration_min",
            "concentration_range",
            "concentration_zscore",
            "concentration_rate_of_change",
            "wind_speed",
            "wind_direction_sin",
            "wind_direction_cos",
            "temperature",
            "stability_class",
            "upwind_downwind_diff",
            "background_concentration",
            "elevation_above_background",
            "hour_of_day",
            "day_of_week",
            "is_daytime",
            "equipment_type",
            "days_since_inspection",
            "leak_history_count",
            "operating_pressure",
            "plume_likelihood_score",
            "spatial_anomaly_score",
            "temporal_anomaly_score",
        ]


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering."""
    # Time windows for statistics
    rolling_window_minutes: int = Field(default=60, ge=5, le=1440)
    baseline_window_hours: int = Field(default=24, ge=1, le=168)

    # Z-score thresholds
    zscore_threshold: float = Field(default=3.0, ge=1.0, le=10.0)

    # Background estimation
    background_percentile: float = Field(default=10.0, ge=1.0, le=50.0)

    # Spatial analysis
    upwind_distance_m: float = Field(default=100.0, ge=10.0, le=1000.0)
    downwind_distance_m: float = Field(default=500.0, ge=50.0, le=5000.0)

    # Equipment type encoding
    equipment_type_weights: Dict[str, int] = Field(default_factory=lambda: {
        "valve": 1,
        "pump": 2,
        "compressor": 3,
        "flange": 4,
        "connector": 5,
        "pressure_relief": 6,
        "open_ended_line": 7,
        "tank": 8,
        "cooling_tower": 9,
        "unknown": 0,
    })


class FeatureEngineer:
    """
    Feature Engineering Engine for Fugitive Emissions Detection.

    Produces deterministic feature vectors for ML models with
    complete provenance tracking.
    """

    STABILITY_ENCODING = {
        WindStability.A: 1,
        WindStability.B: 2,
        WindStability.C: 3,
        WindStability.D: 4,
        WindStability.E: 5,
        WindStability.F: 6,
    }

    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        self.config = config or FeatureEngineeringConfig()
        self._sensor_history: Dict[str, deque] = {}
        self._baseline_data: Dict[str, deque] = {}
        logger.info("FeatureEngineer initialized")

    def engineer_features(
        self,
        sensor_reading: SensorReading,
        met_data: MeteorologicalData,
        equipment: Optional[EquipmentContext] = None,
        nearby_sensors: Optional[List[SensorReading]] = None
    ) -> FeatureVector:
        """
        Engineer features from sensor data for ML model.

        Args:
            sensor_reading: Current sensor reading
            met_data: Meteorological data
            equipment: Optional equipment context
            nearby_sensors: Optional list of nearby sensor readings

        Returns:
            FeatureVector with all computed features
        """
        # Update history
        self._update_history(sensor_reading)

        # Calculate concentration features
        conc_features = self._calculate_concentration_features(sensor_reading)

        # Calculate meteorological features
        met_features = self._calculate_met_features(met_data)

        # Calculate spatial features
        spatial_features = self._calculate_spatial_features(
            sensor_reading, met_data, nearby_sensors
        )

        # Calculate temporal features
        temporal_features = self._calculate_temporal_features(sensor_reading.timestamp)

        # Calculate equipment features
        equip_features = self._calculate_equipment_features(equipment)

        # Calculate derived anomaly scores
        anomaly_scores = self._calculate_anomaly_scores(
            conc_features, spatial_features, temporal_features
        )

        # Build feature vector
        feature_id = f"FV-{sensor_reading.sensor_id}-{sensor_reading.timestamp.strftime('%Y%m%d%H%M%S')}"

        source_ids = [sensor_reading.sensor_id]
        if nearby_sensors:
            source_ids.extend([s.sensor_id for s in nearby_sensors])

        feature_vector = FeatureVector(
            feature_id=feature_id,
            timestamp=sensor_reading.timestamp,
            # Concentration features
            concentration_current=conc_features["current"],
            concentration_mean=conc_features["mean"],
            concentration_std=conc_features["std"],
            concentration_max=conc_features["max"],
            concentration_min=conc_features["min"],
            concentration_range=conc_features["range"],
            concentration_zscore=conc_features["zscore"],
            concentration_rate_of_change=conc_features["rate_of_change"],
            # Met features
            wind_speed=met_features["wind_speed"],
            wind_direction=met_features["wind_direction"],
            wind_direction_sin=met_features["wind_dir_sin"],
            wind_direction_cos=met_features["wind_dir_cos"],
            temperature=met_features["temperature"],
            stability_class_encoded=met_features["stability_encoded"],
            # Spatial features
            upwind_downwind_diff=spatial_features["upwind_downwind_diff"],
            background_concentration=spatial_features["background"],
            elevation_above_background=spatial_features["elevation"],
            # Temporal features
            hour_of_day=temporal_features["hour"],
            day_of_week=temporal_features["day_of_week"],
            is_daytime=temporal_features["is_daytime"],
            # Equipment features
            equipment_type_encoded=equip_features["type_encoded"],
            days_since_inspection=equip_features["days_since_inspection"],
            leak_history_count=equip_features["leak_history"],
            operating_pressure=equip_features["pressure"],
            # Anomaly scores
            plume_likelihood_score=anomaly_scores["plume_likelihood"],
            spatial_anomaly_score=anomaly_scores["spatial"],
            temporal_anomaly_score=anomaly_scores["temporal"],
            # Provenance
            source_sensor_ids=source_ids,
        )

        # Calculate provenance hash
        content = f"{feature_id}|{feature_vector.concentration_current}|{feature_vector.wind_speed}"
        feature_vector.provenance_hash = hashlib.sha256(content.encode()).hexdigest()

        return feature_vector

    def engineer_batch(
        self,
        sensor_readings: List[SensorReading],
        met_data: List[MeteorologicalData],
        equipment: Optional[List[EquipmentContext]] = None
    ) -> List[FeatureVector]:
        """Engineer features for a batch of readings."""
        feature_vectors: List[FeatureVector] = []

        equipment_map = {}
        if equipment:
            # Create simple mapping - in production would use spatial matching
            equipment_map = {e.equipment_id: e for e in equipment}

        for i, reading in enumerate(sensor_readings):
            # Find corresponding met data (closest timestamp)
            met = self._find_closest_met(reading.timestamp, met_data)

            # Get equipment context if available
            equip = equipment_map.get(reading.sensor_id)

            # Get nearby sensors
            nearby = [r for r in sensor_readings if r.sensor_id != reading.sensor_id][:5]

            fv = self.engineer_features(reading, met, equip, nearby)
            feature_vectors.append(fv)

        return feature_vectors

    def _update_history(self, reading: SensorReading) -> None:
        """Update sensor history for rolling calculations."""
        sensor_id = reading.sensor_id

        if sensor_id not in self._sensor_history:
            max_points = int(self.config.rolling_window_minutes * 4)  # Assume 15-sec data
            self._sensor_history[sensor_id] = deque(maxlen=max_points)

        if sensor_id not in self._baseline_data:
            max_points = int(self.config.baseline_window_hours * 60 * 4)
            self._baseline_data[sensor_id] = deque(maxlen=max_points)

        entry = (reading.timestamp, float(reading.concentration_ppm))
        self._sensor_history[sensor_id].append(entry)
        self._baseline_data[sensor_id].append(entry)

    def _calculate_concentration_features(
        self,
        reading: SensorReading
    ) -> Dict[str, float]:
        """Calculate concentration-based features."""
        current = float(reading.concentration_ppm)
        sensor_id = reading.sensor_id

        history = list(self._sensor_history.get(sensor_id, []))
        values = [v for _, v in history] if history else [current]

        mean_val = statistics.mean(values) if values else current
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        max_val = max(values) if values else current
        min_val = min(values) if values else current
        range_val = max_val - min_val

        # Z-score
        zscore = (current - mean_val) / std_val if std_val > 0 else 0.0

        # Rate of change
        rate_of_change = 0.0
        if len(history) >= 2:
            prev_time, prev_val = history[-2]
            time_diff = (reading.timestamp - prev_time).total_seconds() / 60.0
            if time_diff > 0:
                rate_of_change = (current - prev_val) / time_diff

        return {
            "current": current,
            "mean": mean_val,
            "std": std_val,
            "max": max_val,
            "min": min_val,
            "range": range_val,
            "zscore": zscore,
            "rate_of_change": rate_of_change,
        }

    def _calculate_met_features(
        self,
        met_data: MeteorologicalData
    ) -> Dict[str, float]:
        """Calculate meteorological features."""
        wind_dir_rad = math.radians(float(met_data.wind_direction_deg))

        stability_encoded = 4  # Default to neutral
        if met_data.stability_class:
            stability_encoded = self.STABILITY_ENCODING.get(
                met_data.stability_class, 4
            )

        return {
            "wind_speed": float(met_data.wind_speed_m_s),
            "wind_direction": float(met_data.wind_direction_deg),
            "wind_dir_sin": math.sin(wind_dir_rad),
            "wind_dir_cos": math.cos(wind_dir_rad),
            "temperature": float(met_data.temperature_c),
            "stability_encoded": stability_encoded,
        }

    def _calculate_spatial_features(
        self,
        reading: SensorReading,
        met_data: MeteorologicalData,
        nearby_sensors: Optional[List[SensorReading]]
    ) -> Dict[str, float]:
        """Calculate spatial features including upwind/downwind analysis."""
        current = float(reading.concentration_ppm)

        # Calculate background from baseline
        sensor_id = reading.sensor_id
        baseline = list(self._baseline_data.get(sensor_id, []))
        baseline_values = [v for _, v in baseline] if baseline else [current]

        # Background = 10th percentile
        sorted_baseline = sorted(baseline_values)
        idx = int(len(sorted_baseline) * self.config.background_percentile / 100.0)
        background = sorted_baseline[idx] if sorted_baseline else current

        # Elevation above background
        elevation = current - background

        # Upwind/downwind differential
        upwind_downwind_diff = 0.0
        if nearby_sensors and len(nearby_sensors) >= 2:
            nearby_values = [float(s.concentration_ppm) for s in nearby_sensors]
            # Simplified: use variance as proxy for spatial gradient
            if len(nearby_values) > 1:
                upwind_downwind_diff = max(nearby_values) - min(nearby_values)

        return {
            "upwind_downwind_diff": upwind_downwind_diff,
            "background": background,
            "elevation": max(0.0, elevation),
        }

    def _calculate_temporal_features(
        self,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Calculate temporal features."""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        # Simple daytime check (6 AM - 6 PM)
        is_daytime = 6 <= hour < 18

        return {
            "hour": hour,
            "day_of_week": day_of_week,
            "is_daytime": is_daytime,
        }

    def _calculate_equipment_features(
        self,
        equipment: Optional[EquipmentContext]
    ) -> Dict[str, float]:
        """Calculate equipment context features."""
        if not equipment:
            return {
                "type_encoded": 0,
                "days_since_inspection": 365.0,
                "leak_history": 0,
                "pressure": 0.0,
            }

        type_encoded = self.config.equipment_type_weights.get(
            equipment.equipment_type.value, 0
        )

        days_since_inspection = 365.0
        if equipment.last_inspection_date:
            delta = datetime.utcnow() - equipment.last_inspection_date
            days_since_inspection = delta.days

        pressure = float(equipment.operating_pressure_psi or 0)

        return {
            "type_encoded": type_encoded,
            "days_since_inspection": float(days_since_inspection),
            "leak_history": equipment.leak_history_count,
            "pressure": pressure,
        }

    def _calculate_anomaly_scores(
        self,
        conc_features: Dict[str, float],
        spatial_features: Dict[str, float],
        temporal_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate derived anomaly scores."""
        # Plume likelihood based on elevation and spatial gradient
        elevation = spatial_features["elevation"]
        gradient = spatial_features["upwind_downwind_diff"]
        plume_likelihood = min(1.0, (elevation / 100.0 + gradient / 50.0) / 2.0)

        # Spatial anomaly based on zscore and gradient
        zscore = abs(conc_features["zscore"])
        spatial_anomaly = min(1.0, (zscore / 3.0 + gradient / 100.0) / 2.0)

        # Temporal anomaly based on rate of change
        rate = abs(conc_features["rate_of_change"])
        temporal_anomaly = min(1.0, rate / 10.0)

        return {
            "plume_likelihood": plume_likelihood,
            "spatial": spatial_anomaly,
            "temporal": temporal_anomaly,
        }

    def _find_closest_met(
        self,
        timestamp: datetime,
        met_data: List[MeteorologicalData]
    ) -> MeteorologicalData:
        """Find closest meteorological data by timestamp."""
        if not met_data:
            # Return default met data
            return MeteorologicalData(
                timestamp=timestamp,
                wind_speed_m_s=Decimal("2.0"),
                wind_direction_deg=Decimal("180"),
                temperature_c=Decimal("20"),
            )

        closest = min(
            met_data,
            key=lambda m: abs((m.timestamp - timestamp).total_seconds())
        )
        return closest

    def clear_history(self, sensor_id: Optional[str] = None) -> None:
        """Clear sensor history."""
        if sensor_id:
            self._sensor_history.pop(sensor_id, None)
            self._baseline_data.pop(sensor_id, None)
        else:
            self._sensor_history.clear()
            self._baseline_data.clear()


# Export all public classes
__all__ = [
    "WindStability",
    "EquipmentType",
    "SensorReading",
    "MeteorologicalData",
    "EquipmentContext",
    "FeatureVector",
    "FeatureEngineeringConfig",
    "FeatureEngineer",
]
