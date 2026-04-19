# -*- coding: utf-8 -*-
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np

class MetricType(Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MAE = "mae"
    RMSE = "rmse"
    LATENCY = "latency"
    THROUGHPUT = "throughput"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricValue:
    metric_type: MetricType
    value: float
    model_name: str
    version_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Alert:
    alert_id: str
    level: AlertLevel
    metric_type: MetricType
    model_name: str
    message: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class MonitoringConfig:
    accuracy_threshold: float = 0.8
    mae_threshold: float = 10.0
    rmse_threshold: float = 15.0
    latency_threshold_ms: float = 100.0
    throughput_min: float = 100.0
    window_minutes: int = 60

class ModelMonitor:
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self._metrics: Dict[str, List[MetricValue]] = {}
        self._alerts: List[Alert] = []
        self._thresholds: Dict[MetricType, float] = {
            MetricType.ACCURACY: self.config.accuracy_threshold,
            MetricType.MAE: self.config.mae_threshold,
            MetricType.RMSE: self.config.rmse_threshold,
            MetricType.LATENCY: self.config.latency_threshold_ms,
            MetricType.THROUGHPUT: self.config.throughput_min,
        }
        
    def record_metric(self, metric: MetricValue) -> None:
        key = f"{metric.model_name}:{metric.metric_type.value}"
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(metric)
        # Check for threshold violations
        self._check_threshold(metric)
    
    def _check_threshold(self, metric: MetricValue) -> Optional[Alert]:
        threshold = self._thresholds.get(metric.metric_type)
        if threshold is None:
            return None
        
        violation = False
        if metric.metric_type in [MetricType.ACCURACY, MetricType.THROUGHPUT]:
            violation = metric.value < threshold
        else:
            violation = metric.value > threshold
        
        if violation:
            alert = Alert(
                alert_id=hashlib.sha256(f"{metric.model_name}{metric.metric_type.value}{datetime.utcnow()}".encode()).hexdigest()[:16],
                level=AlertLevel.WARNING if abs(metric.value - threshold) / threshold < 0.2 else AlertLevel.ERROR,
                metric_type=metric.metric_type,
                model_name=metric.model_name,
                message=f"{metric.metric_type.value} threshold violated",
                current_value=metric.value,
                threshold=threshold,
            )
            self._alerts.append(alert)
            return alert
        return None
    
    def get_metrics(self, model_name: str, metric_type: MetricType, since: Optional[datetime] = None) -> List[MetricValue]:
        key = f"{model_name}:{metric_type.value}"
        metrics = self._metrics.get(key, [])
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        return metrics
    
    def get_metric_stats(self, model_name: str, metric_type: MetricType, window_minutes: Optional[int] = None) -> Dict[str, float]:
        window = window_minutes or self.config.window_minutes
        since = datetime.utcnow() - timedelta(minutes=window)
        metrics = self.get_metrics(model_name, metric_type, since)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }
    
    def get_alerts(self, level: Optional[AlertLevel] = None, unresolved_only: bool = True) -> List[Alert]:
        alerts = self._alerts.copy()
        if level:
            alerts = [a for a in alerts if a.level == level]
        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]
        return alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def get_health_summary(self) -> Dict[str, Any]:
        active_alerts = self.get_alerts(unresolved_only=True)
        return {
            "total_metrics_tracked": len(self._metrics),
            "active_alerts": len(active_alerts),
            "critical_alerts": sum(1 for a in active_alerts if a.level == AlertLevel.CRITICAL),
            "error_alerts": sum(1 for a in active_alerts if a.level == AlertLevel.ERROR),
            "warning_alerts": sum(1 for a in active_alerts if a.level == AlertLevel.WARNING),
            "health_status": "healthy" if not active_alerts else "degraded" if len(active_alerts) < 3 else "unhealthy",
        }
