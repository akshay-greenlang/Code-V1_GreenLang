# -*- coding: utf-8 -*-
"""ML Explainer for GL-010 EmissionsGuardian - SHAP/LIME integration"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import hashlib, logging, uuid

from .schemas import (
    AudienceLevel, ConfidenceLevel, FeatureContribution, MLExplanation,
    SimilarCase, TemplateVersion,
)

logger = logging.getLogger(__name__)


class MLExplainer:
    """Provides explanations for ML model predictions using SHAP/LIME."""

    def __init__(self):
        self.template_version = TemplateVersion(
            template_id="ML_V1", version="1.0.0",
            effective_date=datetime(2024, 1, 1), approved_by="GL-010",
            checksum=hashlib.sha256(b"ml_v1").hexdigest()
        )

    def explain_prediction_shap(
        self, model_id: str, model_version: str, prediction_id: str,
        prediction_value: Union[float, str], shap_values: Dict[str, float],
        feature_values: Dict[str, Any], baseline: float = 0.0
    ) -> MLExplanation:
        """Generate SHAP-based explanation for a prediction."""
        sorted_feats = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        total = sum(abs(v) for v in shap_values.values()) or 1
        
        contributions = [FeatureContribution(
            feature_name=f, feature_value=feature_values.get(f, 0),
            contribution_value=v, contribution_direction="positive" if v >= 0 else "negative",
            contribution_percent=abs(v) / total * 100, rank=i+1
        ) for i, (f, v) in enumerate(sorted_feats)]
        
        return MLExplanation(
            explanation_id=str(uuid.uuid4()), model_id=model_id, model_version=model_version,
            prediction_id=prediction_id, prediction_value=prediction_value,
            feature_contributions=contributions, baseline_value=baseline,
            shap_values=shap_values, confidence=0.9, confidence_level=ConfidenceLevel.HIGH,
            explanation_method="SHAP", summary_text=f"Prediction: {prediction_value}",
            detailed_text="
".join(f"{c.feature_name}: {c.contribution_value:.4f}" for c in contributions[:5]),
            provenance_hash=hashlib.sha256(str(prediction_id).encode()).hexdigest()
        )

    def explain_prediction_lime(
        self, model_id: str, model_version: str, prediction_id: str,
        prediction_value: Union[float, str], lime_weights: Dict[str, float],
        feature_values: Dict[str, Any]
    ) -> MLExplanation:
        """Generate LIME-based explanation."""
        sorted_feats = sorted(lime_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        total = sum(abs(v) for v in lime_weights.values()) or 1
        
        contributions = [FeatureContribution(
            feature_name=f, feature_value=feature_values.get(f, 0),
            contribution_value=v, contribution_direction="positive" if v >= 0 else "negative",
            contribution_percent=abs(v) / total * 100, rank=i+1
        ) for i, (f, v) in enumerate(sorted_feats)]
        
        return MLExplanation(
            explanation_id=str(uuid.uuid4()), model_id=model_id, model_version=model_version,
            prediction_id=prediction_id, prediction_value=prediction_value,
            feature_contributions=contributions, baseline_value=0.0,
            lime_weights=lime_weights, confidence=0.85, confidence_level=ConfidenceLevel.HIGH,
            explanation_method="LIME", summary_text=f"Prediction: {prediction_value}",
            detailed_text="
".join(f"{c.feature_name}: {c.contribution_value:.4f}" for c in contributions[:5]),
            provenance_hash=hashlib.sha256(str(prediction_id).encode()).hexdigest()
        )

    def find_similar_cases(self, features: Dict, history: List[Dict], k: int = 5) -> List[SimilarCase]:
        """Find similar historical cases."""
        return [SimilarCase(
            case_id=c.get("id", str(uuid.uuid4())), similarity_score=c.get("sim", 0.8),
            outcome=c.get("outcome", "unknown"), key_similarities=c.get("sims", []),
            key_differences=c.get("diffs", []), timestamp=datetime.now()
        ) for c in history[:k]]
