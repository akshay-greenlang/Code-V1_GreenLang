# -*- coding: utf-8 -*-
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import json

class ModelStatus(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class ModelType(Enum):
    RUL_ESTIMATOR = "rul_estimator"
    ANOMALY_DETECTOR = "anomaly_detector"
    FAILURE_PREDICTOR = "failure_predictor"
    HEALTH_CALCULATOR = "health_calculator"

@dataclass
class ModelVersion:
    version_id: str
    model_name: str
    model_type: ModelType
    version: str
    status: ModelStatus
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    promoted_at: Optional[datetime] = None
    provenance_hash: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "version": self.version,
            "status": self.status.value,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
        }

@dataclass
class ModelArtifact:
    artifact_id: str
    version_id: str
    artifact_path: str
    artifact_type: str
    size_bytes: int
    checksum: str
    created_at: datetime = field(default_factory=datetime.utcnow)

class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, Dict[str, ModelVersion]] = {}
        self._artifacts: Dict[str, List[ModelArtifact]] = {}
        self._production_versions: Dict[str, str] = {}
        
    def register_model(self, model: ModelVersion) -> str:
        if model.model_name not in self._models:
            self._models[model.model_name] = {}
        
        provenance = hashlib.sha256(f"{model.model_name}{model.version}{model.training_data_hash}".encode()).hexdigest()
        model.provenance_hash = provenance
        self._models[model.model_name][model.version_id] = model
        return model.version_id
    
    def get_model(self, model_name: str, version_id: Optional[str] = None) -> Optional[ModelVersion]:
        if model_name not in self._models:
            return None
        if version_id:
            return self._models[model_name].get(version_id)
        # Return production version
        prod_version_id = self._production_versions.get(model_name)
        if prod_version_id:
            return self._models[model_name].get(prod_version_id)
        return None
    
    def list_versions(self, model_name: str) -> List[ModelVersion]:
        if model_name not in self._models:
            return []
        return list(self._models[model_name].values())
    
    def promote_to_production(self, model_name: str, version_id: str) -> bool:
        if model_name not in self._models or version_id not in self._models[model_name]:
            return False
        # Demote current production
        current_prod = self._production_versions.get(model_name)
        if current_prod and current_prod in self._models[model_name]:
            self._models[model_name][current_prod].status = ModelStatus.DEPRECATED
        # Promote new version
        self._models[model_name][version_id].status = ModelStatus.PRODUCTION
        self._models[model_name][version_id].promoted_at = datetime.utcnow()
        self._production_versions[model_name] = version_id
        return True
    
    def register_artifact(self, artifact: ModelArtifact) -> None:
        if artifact.version_id not in self._artifacts:
            self._artifacts[artifact.version_id] = []
        self._artifacts[artifact.version_id].append(artifact)
    
    def get_artifacts(self, version_id: str) -> List[ModelArtifact]:
        return self._artifacts.get(version_id, [])
    
    def compare_versions(self, model_name: str, version_a: str, version_b: str) -> Dict[str, Any]:
        model_a = self.get_model(model_name, version_a)
        model_b = self.get_model(model_name, version_b)
        if not model_a or not model_b:
            return {"error": "Version not found"}
        
        metric_comparison = {}
        all_metrics = set(model_a.metrics.keys()) | set(model_b.metrics.keys())
        for metric in all_metrics:
            val_a = model_a.metrics.get(metric, 0)
            val_b = model_b.metrics.get(metric, 0)
            metric_comparison[metric] = {
                "version_a": val_a,
                "version_b": val_b,
                "delta": val_b - val_a,
                "improved": val_b > val_a,
            }
        
        return {
            "version_a": version_a,
            "version_b": version_b,
            "metrics": metric_comparison,
            "recommendation": version_b if sum(1 for m in metric_comparison.values() if m["improved"]) > len(metric_comparison) / 2 else version_a,
        }
