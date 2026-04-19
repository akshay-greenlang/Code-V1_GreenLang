"""
GL-013 PredictiveMaintenance - Feature Store Module

Zero-Hallucination Feature Management for ML Pipeline Consistency

Key Features:
- FeatureStore class with semantic versioning
- Feature definitions aligned between training/inference
- Feature lineage tracking with SHA-256 provenance
- Data quality flags per feature

ZERO-HALLUCINATION GUARANTEE:
-----------------------------
Feature definitions are deterministic and versioned.
Training and inference use identical feature computations.

Author: GL-013 PredictiveMaintenance Agent
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime
import hashlib
import json


class DataQualityFlag(Enum):
    VALID = "valid"
    MISSING = "missing"
    OUT_OF_RANGE = "out_of_range"
    STALE = "stale"
    IMPUTED = "imputed"
    INTERPOLATED = "interpolated"
    SENSOR_FAULT = "sensor_fault"
    CALIBRATION_ERROR = "calibration_error"


class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    VECTOR = "vector"


@dataclass
class FeatureDefinition:
    name: str
    description: str
    feature_type: FeatureType
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Optional[Any] = None
    version: str = "1.0.0"
    source_signals: List[str] = field(default_factory=list)
    computation_formula: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def get_definition_hash(self) -> str:
        data = {"name": self.name, "feature_type": self.feature_type.value, "unit": self.unit, "min_value": self.min_value, "max_value": self.max_value, "version": self.version, "computation_formula": self.computation_formula}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class FeatureLineage:
    feature_name: str
    source_data_hash: str
    computation_timestamp: str
    computation_parameters: Dict[str, Any]
    input_features: List[str]
    algorithm_version: str
    provenance_hash: str = ""
    
    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self._compute_provenance()
    
    def _compute_provenance(self) -> str:
        data = {"feature_name": self.feature_name, "source_data_hash": self.source_data_hash, "computation_timestamp": self.computation_timestamp, "computation_parameters": self.computation_parameters, "input_features": self.input_features, "algorithm_version": self.algorithm_version}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class FeatureValue:
    feature_name: str
    value: Any
    timestamp: str
    quality_flag: DataQualityFlag
    lineage: Optional[FeatureLineage] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        return self.quality_flag == DataQualityFlag.VALID
    
    def to_dict(self) -> Dict[str, Any]:
        return {"feature_name": self.feature_name, "value": self.value, "timestamp": self.timestamp, "quality_flag": self.quality_flag.value, "confidence": self.confidence, "lineage_hash": self.lineage.provenance_hash if self.lineage else None}


class FeatureStore:
    """
    Feature store with versioning and lineage tracking.
    
    Ensures training/inference feature alignment with deterministic
    computation and complete provenance tracking.
    """
    
    def __init__(self, store_version: str = "1.0.0"):
        self.store_version = store_version
        self._definitions: Dict[str, FeatureDefinition] = {}
        self._values: Dict[str, List[FeatureValue]] = {}
        self._computed_features: Dict[str, Callable] = {}
    
    def register_feature(self, definition: FeatureDefinition) -> str:
        self._definitions[definition.name] = definition
        self._values[definition.name] = []
        return definition.get_definition_hash()
    
    def register_computation(self, feature_name: str, computation: Callable) -> None:
        if feature_name not in self._definitions:
            raise ValueError(f"Feature {feature_name} not registered")
        self._computed_features[feature_name] = computation
    
    def get_definition(self, feature_name: str) -> Optional[FeatureDefinition]:
        return self._definitions.get(feature_name)
    
    def store_value(self, feature_value: FeatureValue) -> None:
        if feature_value.feature_name not in self._definitions:
            raise ValueError(f"Feature {feature_value.feature_name} not registered")
        defn = self._definitions[feature_value.feature_name]
        if defn.min_value is not None and feature_value.value < defn.min_value:
            feature_value.quality_flag = DataQualityFlag.OUT_OF_RANGE
        if defn.max_value is not None and feature_value.value > defn.max_value:
            feature_value.quality_flag = DataQualityFlag.OUT_OF_RANGE
        self._values[feature_value.feature_name].append(feature_value)
    
    def get_latest_value(self, feature_name: str) -> Optional[FeatureValue]:
        values = self._values.get(feature_name, [])
        return values[-1] if values else None
    
    def get_values(self, feature_name: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[FeatureValue]:
        values = self._values.get(feature_name, [])
        if start_time:
            values = [v for v in values if v.timestamp >= start_time]
        if end_time:
            values = [v for v in values if v.timestamp <= end_time]
        return values

    
    def compute_feature(self, feature_name: str, input_data: Dict[str, Any], source_hash: str = "") -> FeatureValue:
        if feature_name not in self._computed_features:
            raise ValueError(f"No computation registered for {feature_name}")
        defn = self._definitions[feature_name]
        computation = self._computed_features[feature_name]
        timestamp = datetime.utcnow().isoformat()
        value = computation(input_data)
        lineage = FeatureLineage(feature_name=feature_name, source_data_hash=source_hash or hashlib.sha256(str(input_data).encode()).hexdigest()[:16], computation_timestamp=timestamp, computation_parameters={}, input_features=list(input_data.keys()), algorithm_version=defn.version)
        quality_flag = DataQualityFlag.VALID
        if value is None:
            quality_flag = DataQualityFlag.MISSING
            value = defn.default_value
        elif defn.min_value is not None and value < defn.min_value:
            quality_flag = DataQualityFlag.OUT_OF_RANGE
        elif defn.max_value is not None and value > defn.max_value:
            quality_flag = DataQualityFlag.OUT_OF_RANGE
        feature_value = FeatureValue(feature_name=feature_name, value=value, timestamp=timestamp, quality_flag=quality_flag, lineage=lineage)
        self.store_value(feature_value)
        return feature_value
    
    def get_feature_vector(self, feature_names: List[str], timestamp: Optional[str] = None) -> Dict[str, Any]:
        result = {}
        quality_flags = {}
        for name in feature_names:
            value = self.get_latest_value(name)
            if value:
                result[name] = value.value
                quality_flags[name] = value.quality_flag.value
            else:
                defn = self._definitions.get(name)
                result[name] = defn.default_value if defn else None
                quality_flags[name] = DataQualityFlag.MISSING.value
        return {"features": result, "quality_flags": quality_flags, "timestamp": timestamp or datetime.utcnow().isoformat()}
    
    def validate_feature_alignment(self, training_store: "FeatureStore") -> Dict[str, Any]:
        mismatches = []
        for name, defn in self._definitions.items():
            if name in training_store._definitions:
                train_defn = training_store._definitions[name]
                if defn.get_definition_hash() != train_defn.get_definition_hash():
                    mismatches.append({"feature": name, "inference_hash": defn.get_definition_hash(), "training_hash": train_defn.get_definition_hash(), "inference_version": defn.version, "training_version": train_defn.version})
        return {"aligned": len(mismatches) == 0, "mismatches": mismatches, "total_features": len(self._definitions)}
    
    def export_definitions(self) -> Dict[str, Any]:
        return {"store_version": self.store_version, "features": {name: {"description": d.description, "type": d.feature_type.value, "unit": d.unit, "min": d.min_value, "max": d.max_value, "version": d.version, "hash": d.get_definition_hash()} for name, d in self._definitions.items()}}
    
    def get_store_hash(self) -> str:
        definitions_str = json.dumps(self.export_definitions(), sort_keys=True)
        return hashlib.sha256(definitions_str.encode()).hexdigest()
