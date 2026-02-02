# -*- coding: utf-8 -*-
"""
Feature Store Module for GL-011 FuelCraft

Provides feature management with versioning, lineage tracking, and drift detection
for fuel price forecasting. Ensures deterministic feature extraction with shared
contracts between training and serving pipelines.

Features:
- Feature definitions with versioning and metadata
- Feature extraction contracts (shared training/serving)
- Feature lineage tracking with SHA-256 hashes
- Drift detection hooks for data quality monitoring
- Business-language feature labels

Zero-Hallucination Architecture:
- All feature transformations are deterministic
- Fixed precision arithmetic for numeric features
- Complete provenance tracking
- No LLM-based feature engineering

Author: GreenLang AI Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Constants
DEFAULT_PRECISION = 6
DRIFT_THRESHOLD_PSI = 0.1
DRIFT_THRESHOLD_KS = 0.05


class FeatureType(str, Enum):
    """Enumeration of feature types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    BINARY = "binary"
    ORDINAL = "ordinal"


class FeatureSource(str, Enum):
    """Enumeration of feature data sources."""
    NYMEX = "nymex"
    ICE = "ice"
    EIA = "eia"
    NOAA = "noaa"
    CALCULATED = "calculated"
    EXTERNAL_API = "external_api"
    INTERNAL = "internal"


class DriftStatus(str, Enum):
    """Drift detection status."""
    NO_DRIFT = "no_drift"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class FeatureDefinition(BaseModel):
    """
    Definition of a single feature with versioning and metadata.

    Provides complete specification for feature extraction, including
    transformation logic, validation rules, and business labels.
    """

    name: str = Field(..., description="Unique feature name (snake_case)")
    version: str = Field(..., description="Semantic version (e.g., 1.0.0)")
    display_name: str = Field(..., description="Business-language display name")
    description: str = Field("", description="Feature description")
    feature_type: FeatureType = Field(..., description="Type of feature")
    data_source: FeatureSource = Field(..., description="Data source")

    # Schema
    dtype: str = Field("float64", description="NumPy dtype")
    nullable: bool = Field(False, description="Whether nulls are allowed")
    default_value: Optional[Any] = Field(None, description="Default value if null")

    # Validation
    min_value: Optional[float] = Field(None, description="Minimum valid value")
    max_value: Optional[float] = Field(None, description="Maximum valid value")
    allowed_values: Optional[List[Any]] = Field(None, description="Allowed categorical values")

    # Transformation
    transformation: Optional[str] = Field(None, description="Transformation formula ID")
    lag_periods: Optional[List[int]] = Field(None, description="Lag periods to generate")
    rolling_windows: Optional[List[int]] = Field(None, description="Rolling window sizes")

    # Metadata
    unit: Optional[str] = Field(None, description="Unit of measurement")
    precision: int = Field(DEFAULT_PRECISION, description="Decimal precision")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = Field(default_factory=list, description="Feature tags")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate feature name follows snake_case convention."""
        if not v.replace("_", "").isalnum():
            raise ValueError(f"Feature name must be alphanumeric with underscores: {v}")
        if v != v.lower():
            raise ValueError(f"Feature name must be lowercase: {v}")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"Version must be semantic (X.Y.Z): {v}")
        return v

    def get_feature_id(self) -> str:
        """Get unique feature identifier (name:version)."""
        return f"{self.name}:{self.version}"

    def compute_schema_hash(self) -> str:
        """Compute SHA-256 hash of feature schema."""
        schema_dict = {
            "name": self.name,
            "version": self.version,
            "feature_type": self.feature_type.value,
            "dtype": self.dtype,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "transformation": self.transformation,
            "precision": self.precision,
        }
        schema_json = json.dumps(schema_dict, sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()


class FeatureValue(BaseModel):
    """
    Single feature value with provenance.
    """

    feature_id: str = Field(..., description="Feature identifier (name:version)")
    value: Any = Field(..., description="Feature value")
    timestamp: datetime = Field(..., description="Value timestamp")
    source_record_id: Optional[str] = Field(None, description="Source record identifier")
    is_imputed: bool = Field(False, description="Whether value was imputed")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Value confidence score")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of feature value."""
        data = {
            "feature_id": self.feature_id,
            "value": str(self.value),
            "timestamp": self.timestamp.isoformat(),
            "source_record_id": self.source_record_id,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class FeatureVector(BaseModel):
    """
    Collection of feature values for a single observation.
    """

    vector_id: str = Field(..., description="Unique vector identifier")
    observation_timestamp: datetime = Field(..., description="Observation timestamp")
    features: Dict[str, FeatureValue] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector_hash: str = Field("", description="SHA-256 hash of complete vector")

    def model_post_init(self, __context: Any) -> None:
        """Compute vector hash after initialization."""
        if not self.vector_hash:
            self.vector_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of feature vector."""
        feature_hashes = [f.provenance_hash for f in self.features.values()]
        combined = "|".join(sorted(feature_hashes))
        return hashlib.sha256(combined.encode()).hexdigest()

    def to_numpy(self, feature_order: List[str]) -> np.ndarray:
        """Convert to numpy array with specified feature order."""
        values = []
        for feature_name in feature_order:
            if feature_name in self.features:
                values.append(float(self.features[feature_name].value))
            else:
                values.append(np.nan)
        return np.array(values, dtype=np.float64)

    def get_value(self, feature_name: str) -> Optional[Any]:
        """Get value for a specific feature."""
        if feature_name in self.features:
            return self.features[feature_name].value
        return None


class FeatureLineage(BaseModel):
    """
    Tracks lineage of feature transformations.
    """

    lineage_id: str = Field(..., description="Unique lineage identifier")
    feature_id: str = Field(..., description="Output feature ID")
    source_features: List[str] = Field(default_factory=list, description="Input feature IDs")
    transformation_id: str = Field(..., description="Transformation identifier")
    transformation_version: str = Field(..., description="Transformation version")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    lineage_hash: str = Field("", description="SHA-256 lineage hash")

    def model_post_init(self, __context: Any) -> None:
        """Compute lineage hash after initialization."""
        if not self.lineage_hash:
            self.lineage_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of lineage."""
        data = {
            "feature_id": self.feature_id,
            "source_features": sorted(self.source_features),
            "transformation_id": self.transformation_id,
            "transformation_version": self.transformation_version,
            "parameters": self.parameters,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


class DriftDetectionResult(BaseModel):
    """
    Result of drift detection analysis.
    """

    feature_id: str = Field(..., description="Feature identifier")
    detection_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: DriftStatus = Field(..., description="Drift status")
    psi_score: Optional[float] = Field(None, description="Population Stability Index")
    ks_statistic: Optional[float] = Field(None, description="Kolmogorov-Smirnov statistic")
    ks_pvalue: Optional[float] = Field(None, description="KS test p-value")
    reference_mean: Optional[float] = Field(None, description="Reference distribution mean")
    current_mean: Optional[float] = Field(None, description="Current distribution mean")
    reference_std: Optional[float] = Field(None, description="Reference distribution std")
    current_std: Optional[float] = Field(None, description="Current distribution std")
    sample_size: int = Field(0, description="Sample size used")
    message: str = Field("", description="Human-readable message")


@dataclass
class FeatureStoreConfig:
    """Configuration for feature store."""

    store_id: str = "fuelcraft_feature_store"
    version: str = "1.0.0"
    precision: int = DEFAULT_PRECISION
    enable_drift_detection: bool = True
    drift_threshold_psi: float = DRIFT_THRESHOLD_PSI
    drift_threshold_ks: float = DRIFT_THRESHOLD_KS
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    track_lineage: bool = True


class FeatureStore:
    """
    Feature store for fuel price forecasting.

    Provides centralized feature management with:
    - Feature registration and versioning
    - Feature extraction contracts (shared training/serving)
    - Lineage tracking
    - Drift detection

    Zero-Hallucination Guarantees:
    - All feature transformations are deterministic
    - Fixed precision arithmetic
    - Complete provenance tracking
    """

    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        """
        Initialize feature store.

        Args:
            config: Feature store configuration
        """
        self.config = config or FeatureStoreConfig()
        self._features: Dict[str, FeatureDefinition] = {}
        self._extractors: Dict[str, Callable] = {}
        self._lineage: List[FeatureLineage] = []
        self._reference_distributions: Dict[str, np.ndarray] = {}
        self._cache: Dict[str, Tuple[datetime, Any]] = {}

        # Register built-in features
        self._register_builtin_features()

        logger.info(
            f"FeatureStore initialized: store_id={self.config.store_id}, "
            f"version={self.config.version}"
        )

    def register_feature(self, feature: FeatureDefinition) -> str:
        """
        Register a feature definition.

        Args:
            feature: Feature definition to register

        Returns:
            Feature ID (name:version)
        """
        feature_id = feature.get_feature_id()

        if feature_id in self._features:
            logger.warning(f"Feature {feature_id} already registered, updating")

        self._features[feature_id] = feature
        logger.info(f"Registered feature: {feature_id}")

        return feature_id

    def register_extractor(
        self,
        feature_id: str,
        extractor: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """
        Register a feature extractor function.

        Args:
            feature_id: Feature identifier
            extractor: Function that extracts feature from raw data
        """
        if feature_id not in self._features:
            raise ValueError(f"Feature {feature_id} not registered")

        self._extractors[feature_id] = extractor
        logger.info(f"Registered extractor for: {feature_id}")

    def get_feature(self, feature_id: str) -> Optional[FeatureDefinition]:
        """Get feature definition by ID."""
        return self._features.get(feature_id)

    def get_feature_by_name(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[FeatureDefinition]:
        """
        Get feature definition by name, optionally filtering by version.

        Args:
            name: Feature name
            version: Optional version filter (returns latest if None)

        Returns:
            Feature definition or None
        """
        matching = [
            f for fid, f in self._features.items()
            if f.name == name and (version is None or f.version == version)
        ]

        if not matching:
            return None

        # Return latest version
        return max(matching, key=lambda f: f.version)

    def list_features(
        self,
        source: Optional[FeatureSource] = None,
        feature_type: Optional[FeatureType] = None,
        tags: Optional[List[str]] = None
    ) -> List[FeatureDefinition]:
        """
        List features with optional filtering.

        Args:
            source: Filter by data source
            feature_type: Filter by feature type
            tags: Filter by tags (any match)

        Returns:
            List of matching feature definitions
        """
        features = list(self._features.values())

        if source:
            features = [f for f in features if f.data_source == source]

        if feature_type:
            features = [f for f in features if f.feature_type == feature_type]

        if tags:
            features = [
                f for f in features
                if any(t in f.tags for t in tags)
            ]

        return features

    def extract_features(
        self,
        raw_data: Dict[str, Any],
        feature_ids: List[str],
        timestamp: datetime
    ) -> FeatureVector:
        """
        Extract features from raw data.

        Args:
            raw_data: Raw data dictionary
            feature_ids: List of feature IDs to extract
            timestamp: Observation timestamp

        Returns:
            FeatureVector with extracted values
        """
        import uuid

        features = {}

        for feature_id in feature_ids:
            if feature_id not in self._features:
                logger.warning(f"Feature {feature_id} not registered, skipping")
                continue

            feature_def = self._features[feature_id]

            try:
                # Use registered extractor or default extraction
                if feature_id in self._extractors:
                    value = self._extractors[feature_id](raw_data)
                else:
                    value = self._default_extract(raw_data, feature_def)

                # Validate and round numeric values
                if feature_def.feature_type == FeatureType.NUMERIC:
                    value = self._validate_and_round(value, feature_def)

                features[feature_def.name] = FeatureValue(
                    feature_id=feature_id,
                    value=value,
                    timestamp=timestamp,
                    source_record_id=raw_data.get("record_id"),
                    is_imputed=value is None,
                    confidence=1.0 if value is not None else 0.5,
                )

            except Exception as e:
                logger.error(f"Failed to extract feature {feature_id}: {e}")
                if feature_def.default_value is not None:
                    features[feature_def.name] = FeatureValue(
                        feature_id=feature_id,
                        value=feature_def.default_value,
                        timestamp=timestamp,
                        is_imputed=True,
                        confidence=0.0,
                    )

        vector = FeatureVector(
            vector_id=str(uuid.uuid4()),
            observation_timestamp=timestamp,
            features=features,
            metadata={"raw_data_keys": list(raw_data.keys())},
        )

        return vector

    def set_reference_distribution(
        self,
        feature_id: str,
        values: np.ndarray
    ) -> None:
        """
        Set reference distribution for drift detection.

        Args:
            feature_id: Feature identifier
            values: Reference distribution values
        """
        if feature_id not in self._features:
            raise ValueError(f"Feature {feature_id} not registered")

        self._reference_distributions[feature_id] = values
        logger.info(f"Set reference distribution for {feature_id}: n={len(values)}")

    def detect_drift(
        self,
        feature_id: str,
        current_values: np.ndarray
    ) -> DriftDetectionResult:
        """
        Detect drift between reference and current distributions.

        Args:
            feature_id: Feature identifier
            current_values: Current distribution values

        Returns:
            DriftDetectionResult with statistics
        """
        if feature_id not in self._reference_distributions:
            return DriftDetectionResult(
                feature_id=feature_id,
                status=DriftStatus.UNKNOWN,
                message="No reference distribution available",
            )

        reference = self._reference_distributions[feature_id]

        # Compute PSI
        psi = self._compute_psi(reference, current_values)

        # Compute KS statistic
        ks_stat, ks_pvalue = self._compute_ks_test(reference, current_values)

        # Determine status
        if psi > self.config.drift_threshold_psi or ks_pvalue < self.config.drift_threshold_ks:
            status = DriftStatus.CRITICAL
            message = f"Critical drift detected: PSI={psi:.4f}, KS p-value={ks_pvalue:.4f}"
        elif psi > self.config.drift_threshold_psi * 0.5:
            status = DriftStatus.WARNING
            message = f"Drift warning: PSI={psi:.4f}, KS p-value={ks_pvalue:.4f}"
        else:
            status = DriftStatus.NO_DRIFT
            message = "No significant drift detected"

        return DriftDetectionResult(
            feature_id=feature_id,
            status=status,
            psi_score=psi,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            reference_mean=float(np.mean(reference)),
            current_mean=float(np.mean(current_values)),
            reference_std=float(np.std(reference)),
            current_std=float(np.std(current_values)),
            sample_size=len(current_values),
            message=message,
        )

    def add_lineage(
        self,
        output_feature_id: str,
        source_feature_ids: List[str],
        transformation_id: str,
        transformation_version: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> FeatureLineage:
        """
        Record feature lineage.

        Args:
            output_feature_id: Output feature identifier
            source_feature_ids: Input feature identifiers
            transformation_id: Transformation identifier
            transformation_version: Transformation version
            parameters: Transformation parameters

        Returns:
            Created lineage record
        """
        import uuid

        lineage = FeatureLineage(
            lineage_id=str(uuid.uuid4()),
            feature_id=output_feature_id,
            source_features=source_feature_ids,
            transformation_id=transformation_id,
            transformation_version=transformation_version,
            parameters=parameters or {},
        )

        if self.config.track_lineage:
            self._lineage.append(lineage)

        return lineage

    def get_lineage(self, feature_id: str) -> List[FeatureLineage]:
        """Get lineage records for a feature."""
        return [l for l in self._lineage if l.feature_id == feature_id]

    def get_schema_version(self) -> str:
        """Get combined schema version hash."""
        all_hashes = [f.compute_schema_hash() for f in self._features.values()]
        combined = "|".join(sorted(all_hashes))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def export_schema(self) -> Dict[str, Any]:
        """Export complete feature schema."""
        return {
            "store_id": self.config.store_id,
            "version": self.config.version,
            "schema_hash": self.get_schema_version(),
            "features": {
                fid: f.model_dump() for fid, f in self._features.items()
            },
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

    def _register_builtin_features(self) -> None:
        """Register built-in fuel price features."""
        builtin_features = [
            FeatureDefinition(
                name="natural_gas_spot_price",
                version="1.0.0",
                display_name="Natural Gas Spot Price",
                description="Henry Hub natural gas spot price (USD/MMBtu)",
                feature_type=FeatureType.NUMERIC,
                data_source=FeatureSource.NYMEX,
                unit="USD/MMBtu",
                min_value=0.0,
                max_value=50.0,
                tags=["price", "natural_gas", "spot"],
            ),
            FeatureDefinition(
                name="coal_price",
                version="1.0.0",
                display_name="Coal Price",
                description="Central Appalachian coal price (USD/short ton)",
                feature_type=FeatureType.NUMERIC,
                data_source=FeatureSource.EIA,
                unit="USD/short_ton",
                min_value=0.0,
                max_value=200.0,
                tags=["price", "coal", "spot"],
            ),
            FeatureDefinition(
                name="crude_oil_price",
                version="1.0.0",
                display_name="Crude Oil Price",
                description="WTI crude oil price (USD/barrel)",
                feature_type=FeatureType.NUMERIC,
                data_source=FeatureSource.NYMEX,
                unit="USD/barrel",
                min_value=0.0,
                max_value=200.0,
                tags=["price", "oil", "spot"],
            ),
            FeatureDefinition(
                name="electricity_price",
                version="1.0.0",
                display_name="Electricity Price",
                description="Wholesale electricity price (USD/MWh)",
                feature_type=FeatureType.NUMERIC,
                data_source=FeatureSource.EIA,
                unit="USD/MWh",
                min_value=-100.0,  # Can be negative
                max_value=1000.0,
                tags=["price", "electricity", "spot"],
            ),
            FeatureDefinition(
                name="heating_degree_days",
                version="1.0.0",
                display_name="Heating Degree Days",
                description="Heating degree days (base 65F)",
                feature_type=FeatureType.NUMERIC,
                data_source=FeatureSource.NOAA,
                unit="degree_days",
                min_value=0.0,
                max_value=100.0,
                tags=["weather", "demand_driver"],
            ),
            FeatureDefinition(
                name="cooling_degree_days",
                version="1.0.0",
                display_name="Cooling Degree Days",
                description="Cooling degree days (base 65F)",
                feature_type=FeatureType.NUMERIC,
                data_source=FeatureSource.NOAA,
                unit="degree_days",
                min_value=0.0,
                max_value=100.0,
                tags=["weather", "demand_driver"],
            ),
        ]

        for feature in builtin_features:
            self.register_feature(feature)

    def _default_extract(
        self,
        raw_data: Dict[str, Any],
        feature_def: FeatureDefinition
    ) -> Any:
        """Default feature extraction using feature name as key."""
        value = raw_data.get(feature_def.name)

        if value is None and feature_def.default_value is not None:
            return feature_def.default_value

        return value

    def _validate_and_round(
        self,
        value: Any,
        feature_def: FeatureDefinition
    ) -> Optional[float]:
        """Validate and round numeric value."""
        if value is None:
            return None

        try:
            value = float(value)
        except (ValueError, TypeError):
            return None

        # Check bounds
        if feature_def.min_value is not None and value < feature_def.min_value:
            logger.warning(
                f"Value {value} below min {feature_def.min_value} for {feature_def.name}"
            )
            return None

        if feature_def.max_value is not None and value > feature_def.max_value:
            logger.warning(
                f"Value {value} above max {feature_def.max_value} for {feature_def.name}"
            )
            return None

        # Round to precision
        decimal_value = Decimal(str(value))
        rounded = decimal_value.quantize(
            Decimal(10) ** -feature_def.precision,
            rounding=ROUND_HALF_UP
        )

        return float(rounded)

    def _compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10
    ) -> float:
        """Compute Population Stability Index."""
        # Create bins from expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # Avoid division by zero
        expected_pct = (expected_counts + 0.0001) / len(expected)
        actual_pct = (actual_counts + 0.0001) / len(actual)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return float(psi)

    def _compute_ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """Compute Kolmogorov-Smirnov test statistic."""
        try:
            from scipy import stats
            result = stats.ks_2samp(reference, current)
            return float(result.statistic), float(result.pvalue)
        except ImportError:
            # Fallback if scipy not available
            logger.warning("scipy not available, using simplified KS test")
            return 0.0, 1.0


# Utility functions

def compute_feature_hash(features: Dict[str, Any]) -> str:
    """
    Compute SHA-256 hash of feature dictionary.

    Args:
        features: Dictionary of feature values

    Returns:
        64-character hex hash
    """
    sorted_items = sorted(features.items())
    json_str = json.dumps(sorted_items, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def validate_feature_schema(
    features: Dict[str, Any],
    schema: Dict[str, FeatureDefinition]
) -> Tuple[bool, List[str]]:
    """
    Validate features against schema.

    Args:
        features: Feature values to validate
        schema: Feature definitions

    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []

    for name, value in features.items():
        if name not in schema:
            errors.append(f"Unknown feature: {name}")
            continue

        feature_def = schema[name]

        # Check type
        if feature_def.feature_type == FeatureType.NUMERIC:
            if not isinstance(value, (int, float)) and value is not None:
                errors.append(f"Feature {name} must be numeric, got {type(value)}")

        # Check bounds
        if feature_def.min_value is not None and value is not None:
            if value < feature_def.min_value:
                errors.append(f"Feature {name} below minimum: {value} < {feature_def.min_value}")

        if feature_def.max_value is not None and value is not None:
            if value > feature_def.max_value:
                errors.append(f"Feature {name} above maximum: {value} > {feature_def.max_value}")

        # Check allowed values
        if feature_def.allowed_values and value not in feature_def.allowed_values:
            errors.append(f"Feature {name} not in allowed values: {value}")

    return len(errors) == 0, errors


def detect_feature_drift(
    reference: np.ndarray,
    current: np.ndarray,
    threshold_psi: float = DRIFT_THRESHOLD_PSI
) -> Tuple[bool, float]:
    """
    Quick drift detection using PSI.

    Args:
        reference: Reference distribution
        current: Current distribution
        threshold_psi: PSI threshold for drift

    Returns:
        Tuple of (has_drift, psi_score)
    """
    # Create bins from reference
    breakpoints = np.percentile(reference, np.linspace(0, 100, 11))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    ref_pct = (ref_counts + 0.0001) / len(reference)
    cur_pct = (cur_counts + 0.0001) / len(current)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    return psi > threshold_psi, psi
