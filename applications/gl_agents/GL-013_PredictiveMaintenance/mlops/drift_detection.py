# -*- coding: utf-8 -*-
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import numpy as np

class DriftType(Enum):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_DRIFT = "feature_drift"

class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DriftResult:
    drift_id: str
    drift_type: DriftType
    feature_name: Optional[str]
    statistic: float
    threshold: float
    p_value: Optional[float]
    severity: DriftSeverity
    detected: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

@dataclass
class DriftConfig:
    ks_threshold: float = 0.1
    psi_threshold: float = 0.2
    js_threshold: float = 0.1
    window_size: int = 1000
    reference_window_size: int = 5000

class DriftDetector:
    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self._reference_data: Dict[str, np.ndarray] = {}
        self._drift_history: List[DriftResult] = []
        
    def set_reference(self, feature_name: str, data: np.ndarray) -> None:
        self._reference_data[feature_name] = data.copy()
    
    def detect_ks_drift(self, feature_name: str, current_data: np.ndarray) -> DriftResult:
        from scipy import stats
        
        reference = self._reference_data.get(feature_name)
        if reference is None:
            return self._no_reference_result(feature_name, DriftType.DATA_DRIFT)
        
        ks_stat, p_value = stats.ks_2samp(reference, current_data)
        detected = ks_stat > self.config.ks_threshold
        severity = self._classify_severity(ks_stat, self.config.ks_threshold)
        
        drift_id = hashlib.sha256(f"{feature_name}{ks_stat}{datetime.utcnow()}".encode()).hexdigest()[:16]
        provenance = hashlib.sha256(f"{drift_id}{ks_stat}".encode()).hexdigest()
        
        result = DriftResult(
            drift_id=drift_id,
            drift_type=DriftType.DATA_DRIFT,
            feature_name=feature_name,
            statistic=float(ks_stat),
            threshold=self.config.ks_threshold,
            p_value=float(p_value),
            severity=severity,
            detected=detected,
            provenance_hash=provenance,
        )
        self._drift_history.append(result)
        return result
    
    def detect_psi(self, feature_name: str, current_data: np.ndarray, n_bins: int = 10) -> DriftResult:
        reference = self._reference_data.get(feature_name)
        if reference is None:
            return self._no_reference_result(feature_name, DriftType.FEATURE_DRIFT)
        
        # Calculate PSI
        min_val = min(reference.min(), current_data.min())
        max_val = max(reference.max(), current_data.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current_data, bins=bins)
        
        ref_pct = (ref_hist + 1) / (len(reference) + n_bins)
        cur_pct = (cur_hist + 1) / (len(current_data) + n_bins)
        
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        detected = psi > self.config.psi_threshold
        severity = self._classify_severity(psi, self.config.psi_threshold)
        
        drift_id = hashlib.sha256(f"{feature_name}{psi}{datetime.utcnow()}".encode()).hexdigest()[:16]
        provenance = hashlib.sha256(f"{drift_id}{psi}".encode()).hexdigest()
        
        result = DriftResult(
            drift_id=drift_id,
            drift_type=DriftType.FEATURE_DRIFT,
            feature_name=feature_name,
            statistic=float(psi),
            threshold=self.config.psi_threshold,
            p_value=None,
            severity=severity,
            detected=detected,
            provenance_hash=provenance,
        )
        self._drift_history.append(result)
        return result
    
    def detect_all_features(self, current_data: Dict[str, np.ndarray]) -> List[DriftResult]:
        results = []
        for feature_name, data in current_data.items():
            if feature_name in self._reference_data:
                results.append(self.detect_ks_drift(feature_name, data))
        return results
    
    def get_drift_summary(self) -> Dict:
        if not self._drift_history:
            return {"total_checks": 0, "drifts_detected": 0}
        
        return {
            "total_checks": len(self._drift_history),
            "drifts_detected": sum(1 for r in self._drift_history if r.detected),
            "by_severity": {
                s.value: sum(1 for r in self._drift_history if r.severity == s)
                for s in DriftSeverity
            },
            "by_type": {
                t.value: sum(1 for r in self._drift_history if r.drift_type == t)
                for t in DriftType
            },
        }
    
    def _classify_severity(self, statistic: float, threshold: float) -> DriftSeverity:
        ratio = statistic / threshold
        if ratio < 0.5:
            return DriftSeverity.NONE
        elif ratio < 1.0:
            return DriftSeverity.LOW
        elif ratio < 1.5:
            return DriftSeverity.MEDIUM
        elif ratio < 2.0:
            return DriftSeverity.HIGH
        return DriftSeverity.CRITICAL
    
    def _no_reference_result(self, feature_name: str, drift_type: DriftType) -> DriftResult:
        drift_id = hashlib.sha256(f"{feature_name}no_ref{datetime.utcnow()}".encode()).hexdigest()[:16]
        return DriftResult(
            drift_id=drift_id,
            drift_type=drift_type,
            feature_name=feature_name,
            statistic=0.0,
            threshold=0.0,
            p_value=None,
            severity=DriftSeverity.NONE,
            detected=False,
        )
