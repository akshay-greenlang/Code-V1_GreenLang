# -*- coding: utf-8 -*-
"""
Explanation Schemas for GL-013 PredictiveMaintenance Explainability Module.

Defines Pydantic models for predictive maintenance ML explanations with
zero-hallucination guarantees. All numeric values are derived from deterministic
calculations (SHAP, LIME, causal inference), not LLM-generated estimates.

Author: GreenLang AI Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import hashlib


class ExplanationType(str, Enum):
    SHAP = "shap"
    LIME = "lime"
    ATTENTION = "attention"
    CAUSAL = "causal"
    FEATURE_IMPORTANCE = "feature_importance"
    ROOT_CAUSE = "root_cause"


class PredictionType(str, Enum):
    FAILURE_PROBABILITY = "failure_probability"
    REMAINING_USEFUL_LIFE = "remaining_useful_life"
    ANOMALY_SCORE = "anomaly_score"
    DEGRADATION_RATE = "degradation_rate"
    HEALTH_INDEX = "health_index"
    MAINTENANCE_URGENCY = "maintenance_urgency"


class ModalityType(str, Enum):
    VIBRATION = "vibration"
    MCSA = "mcsa"
    TEMPERATURE = "temperature"
    ACOUSTIC = "acoustic"
    PRESSURE = "pressure"
    OIL_ANALYSIS = "oil_analysis"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FeatureContribution(BaseModel):
    feature_name: str = Field(..., description="Name of the feature")
    feature_value: float = Field(..., description="Actual value of the feature")
    contribution: float = Field(..., description="Contribution to prediction")
    contribution_percentage: float = Field(..., description="Percentage contribution")
    direction: str = Field(..., description="Direction of impact")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    baseline_value: Optional[float] = Field(None, description="Baseline value")
    modality: Optional[str] = Field(None, description="Sensor modality")


class ConfidenceBounds(BaseModel):
    lower_bound: float = Field(..., description="Lower bound")
    upper_bound: float = Field(..., description="Upper bound")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)
    method: str = Field("bootstrap", description="Method used")


class UncertaintyRange(BaseModel):
    point_estimate: float = Field(..., description="Point estimate")
    standard_error: float = Field(..., ge=0)
    confidence_interval: ConfidenceBounds
    prediction_variance: float = Field(..., ge=0)
    epistemic_uncertainty: float = Field(..., ge=0)
    aleatoric_uncertainty: float = Field(..., ge=0)


class AttentionWeight(BaseModel):
    timestamp: datetime
    weight: float = Field(..., ge=0, le=1)
    modality: str
    position: int = Field(..., ge=0)
    head_index: Optional[int] = None


class TemporalSaliencyMap(BaseModel):
    saliency_id: str
    time_window_start: datetime
    time_window_end: datetime
    saliency_scores: List[float]
    timestamps: List[datetime]
    modality: str
    peak_saliency_time: datetime
    peak_saliency_value: float


class CausalEdge(BaseModel):
    source: str
    target: str
    weight: float
    confidence: float = Field(..., ge=0, le=1)
    is_direct: bool = True


class RootCauseHypothesis(BaseModel):
    hypothesis_id: str
    cause_variable: str
    effect_variable: str = "failure"
    causal_effect: float
    uncertainty: float = Field(..., ge=0)
    confidence_interval: ConfidenceBounds
    confounders_adjusted: List[str]
    backdoor_paths_blocked: int = Field(..., ge=0)
    evidence_strength: float = Field(..., ge=0, le=1)
    rank: int = Field(..., ge=1)


class SHAPExplanation(BaseModel):
    explanation_id: str
    prediction_type: PredictionType
    base_value: float
    prediction_value: float
    feature_contributions: List[FeatureContribution]
    interaction_effects: Optional[Dict[str, Dict[str, float]]] = None
    consistency_check: float = Field(..., ge=0)
    explainer_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    computation_time_ms: float = Field(..., ge=0)
    random_seed: int = 42


class LIMEExplanation(BaseModel):
    explanation_id: str
    prediction_type: PredictionType
    prediction_value: float
    feature_contributions: List[FeatureContribution]
    local_model_r2: float = Field(..., ge=0, le=1)
    local_model_intercept: float
    neighborhood_size: int = Field(..., ge=100)
    kernel_width: float = Field(..., gt=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    computation_time_ms: float = Field(..., ge=0)
    random_seed: int = 42


class AttentionExplanation(BaseModel):
    explanation_id: str
    prediction_type: PredictionType
    prediction_value: float
    attention_weights: List[AttentionWeight]
    temporal_saliency_maps: Dict[str, TemporalSaliencyMap]
    cross_modal_attention: Optional[Dict[str, Dict[str, float]]] = None
    peak_attention_times: Dict[str, datetime]
    dominant_modality: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    computation_time_ms: float = Field(..., ge=0)


class CausalExplanation(BaseModel):
    explanation_id: str
    prediction_type: PredictionType
    prediction_value: float
    causal_graph_edges: List[CausalEdge]
    root_cause_hypotheses: List[RootCauseHypothesis]
    confounders_identified: List[str]
    adjustment_set: List[str]
    total_effect: float
    direct_effect: float
    indirect_effect: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    computation_time_ms: float = Field(..., ge=0)


class MaintenanceExplanationReport(BaseModel):
    report_id: str
    asset_id: str
    prediction_type: PredictionType
    model_name: str
    model_version: str
    input_features: Dict[str, float]
    sensor_readings: Optional[Dict[str, List[float]]] = None
    prediction_value: float
    uncertainty: UncertaintyRange
    confidence_level: ConfidenceLevel
    shap_explanation: Optional[SHAPExplanation] = None
    lime_explanation: Optional[LIMEExplanation] = None
    attention_explanation: Optional[AttentionExplanation] = None
    causal_explanation: Optional[CausalExplanation] = None
    top_features: List[FeatureContribution]
    top_root_causes: List[RootCauseHypothesis] = []
    narrative_summary: str
    data_quality_score: float = Field(..., ge=0, le=1)
    missing_features: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    computation_time_ms: float = Field(..., ge=0)
    provenance_hash: str
    deterministic: bool = True

    def compute_provenance_hash(self) -> str:
        content = f"{self.report_id}{self.asset_id}{self.prediction_value}{self.timestamp}{self.input_features}"
        return hashlib.sha256(content.encode()).hexdigest()


class DashboardExplanationData(BaseModel):
    waterfall_data: List[Dict[str, Any]]
    force_plot_data: Dict[str, Any]
    feature_importance_chart: List[Dict[str, Any]]
    attention_heatmap_data: Optional[Dict[str, Any]] = None
    causal_graph_data: Optional[Dict[str, Any]] = None
    root_cause_ranking: Optional[List[Dict[str, Any]]] = None
    trend_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


FeatureContributions = List[FeatureContribution]
RootCauseHypotheses = List[RootCauseHypothesis]
AttentionWeights = List[AttentionWeight]
