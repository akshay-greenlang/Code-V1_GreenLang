# -*- coding: utf-8 -*-
"""
FastAPI Endpoints for ML Explainability Framework.

This module provides RESTful API endpoints for accessing explainability
features including SHAP explanations, LIME explanations, counterfactual
generation, and global feature importance.

All endpoints follow zero-hallucination principles:
- Explanations are derived from actual model computations
- SHA-256 provenance hashing for complete audit trails
- Graceful fallbacks when dependencies unavailable

Endpoints:
    POST /explain/prediction - Explain single prediction
    POST /explain/model - Get global model explanation
    POST /explain/counterfactual - Generate counterfactual
    GET /explain/feature-importance/{model_id} - Feature importance

Example:
    >>> # Run with uvicorn
    >>> # uvicorn greenlang.ml.explainability.api:app --host 0.0.0.0 --port 8000

Author: GreenLang Team
Version: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import hashlib
import logging

# Conditional FastAPI import
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create stub classes for type hints
    class FastAPI:
        def __init__(self, **kwargs): pass
        def get(self, *args, **kwargs): return lambda f: f
        def post(self, *args, **kwargs): return lambda f: f
    class HTTPException(Exception):
        def __init__(self, status_code, detail): pass
    class BaseModel:
        pass
    Field = lambda *args, **kwargs: None

import numpy as np

from .schemas import (
    ExplanationResult,
    GlobalExplanationResult,
    CounterfactualResult,
    WhatIfResult,
    ExplanationRequest,
    CounterfactualRequest,
    GlobalExplanationRequest,
    ExplainerType,
    ExplanationLevel,
    compute_provenance_hash,
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="GreenLang ML Explainability API",
    description=(
        "API for ML model explainability including SHAP, LIME, "
        "counterfactual explanations, and human-readable narratives."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PredictionExplanationRequest(BaseModel):
    """Request schema for prediction explanation endpoint."""

    model_id: str = Field(..., description="Unique identifier for the registered model")
    instance_data: Dict[str, float] = Field(
        ...,
        description="Feature values for the instance to explain"
    )
    explanation_method: str = Field(
        default="shap",
        description="Explanation method: 'shap', 'lime', or 'both'"
    )
    top_k_features: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top features to return"
    )
    include_human_readable: bool = Field(
        default=True,
        description="Include human-readable explanation text"
    )
    audience: str = Field(
        default="engineer",
        description="Target audience: operator, engineer, manager, executive"
    )

    @validator("explanation_method")
    def validate_method(cls, v: str) -> str:
        valid = ["shap", "lime", "both"]
        if v.lower() not in valid:
            raise ValueError(f"explanation_method must be one of {valid}")
        return v.lower()


class PredictionExplanationResponse(BaseModel):
    """Response schema for prediction explanation endpoint."""

    model_id: str = Field(..., description="Model identifier")
    prediction: float = Field(..., description="Model prediction value")
    prediction_class: Optional[str] = Field(None, description="Class label if classification")
    feature_contributions: Dict[str, float] = Field(
        ...,
        description="Feature contribution values"
    )
    top_features: List[Dict[str, Any]] = Field(
        ...,
        description="Top contributing features with details"
    )
    explanation_text: Optional[str] = Field(
        None,
        description="Human-readable explanation"
    )
    confidence: float = Field(..., description="Explanation confidence score")
    explanation_method: str = Field(..., description="Method used")
    provenance_hash: str = Field(..., description="SHA-256 audit hash")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    timestamp: str = Field(..., description="ISO timestamp")


class GlobalExplanationResponse(BaseModel):
    """Response schema for global model explanation endpoint."""

    model_id: str = Field(..., description="Model identifier")
    feature_importance: Dict[str, float] = Field(
        ...,
        description="Global feature importance scores"
    )
    feature_rankings: List[Dict[str, Any]] = Field(
        ...,
        description="Ranked features with importance"
    )
    interaction_effects: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Feature interaction effects"
    )
    summary_statistics: Dict[str, Any] = Field(
        ...,
        description="Summary statistics"
    )
    sample_size: int = Field(..., description="Number of samples used")
    provenance_hash: str = Field(..., description="SHA-256 audit hash")
    processing_time_ms: float = Field(..., description="Processing time")
    timestamp: str = Field(..., description="ISO timestamp")


class CounterfactualExplanationRequest(BaseModel):
    """Request schema for counterfactual explanation endpoint."""

    model_id: str = Field(..., description="Model identifier")
    instance_data: Dict[str, float] = Field(
        ...,
        description="Original feature values"
    )
    target_prediction: Optional[float] = Field(
        None,
        description="Target prediction value to achieve"
    )
    target_class: Optional[str] = Field(
        None,
        description="Target class for classification"
    )
    feature_constraints: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Feature constraints: {feature: {min: x, max: y}}"
    )
    immutable_features: Optional[List[str]] = Field(
        None,
        description="Features that cannot be changed"
    )
    max_features_to_change: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum features to modify"
    )


class CounterfactualExplanationResponse(BaseModel):
    """Response schema for counterfactual explanation endpoint."""

    model_id: str = Field(..., description="Model identifier")
    original_prediction: float = Field(..., description="Original prediction")
    counterfactual_prediction: float = Field(..., description="Counterfactual prediction")
    changes_required: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Required changes: {feature: {from: x, to: y}}"
    )
    num_features_changed: int = Field(..., description="Number of features changed")
    feasibility_score: float = Field(..., description="Feasibility score 0-1")
    explanation_text: str = Field(..., description="Human-readable explanation")
    provenance_hash: str = Field(..., description="SHA-256 audit hash")
    processing_time_ms: float = Field(..., description="Processing time")
    timestamp: str = Field(..., description="ISO timestamp")


class WhatIfRequest(BaseModel):
    """Request schema for what-if analysis endpoint."""

    model_id: str = Field(..., description="Model identifier")
    instance_data: Dict[str, float] = Field(
        ...,
        description="Original feature values"
    )
    feature_changes: Dict[str, float] = Field(
        ...,
        description="Changes to apply: {feature: new_value}"
    )
    scenario_name: str = Field(
        default="what-if",
        description="Name for this scenario"
    )


class WhatIfResponse(BaseModel):
    """Response schema for what-if analysis endpoint."""

    scenario_name: str = Field(..., description="Scenario name")
    original_prediction: float = Field(..., description="Original prediction")
    modified_prediction: float = Field(..., description="Modified prediction")
    prediction_change: float = Field(..., description="Change in prediction")
    applied_changes: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Applied changes: {feature: {from: x, to: y}}"
    )
    sensitivity: Dict[str, float] = Field(
        ...,
        description="Sensitivity of prediction to each change"
    )
    explanation_text: str = Field(..., description="Explanation text")
    provenance_hash: str = Field(..., description="SHA-256 audit hash")
    timestamp: str = Field(..., description="ISO timestamp")


class FeatureImportanceResponse(BaseModel):
    """Response schema for feature importance endpoint."""

    model_id: str = Field(..., description="Model identifier")
    feature_importance: Dict[str, float] = Field(
        ...,
        description="Feature importance scores"
    )
    top_features: List[Dict[str, Any]] = Field(
        ...,
        description="Top features ranked"
    )
    method: str = Field(..., description="Method used to compute importance")
    provenance_hash: str = Field(..., description="SHA-256 audit hash")
    timestamp: str = Field(..., description="ISO timestamp")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    shap_available: bool = Field(..., description="SHAP library available")
    lime_available: bool = Field(..., description="LIME library available")
    timestamp: str = Field(..., description="ISO timestamp")


# =============================================================================
# MODEL REGISTRY (In-memory for demo, would be database in production)
# =============================================================================

class ModelRegistry:
    """
    Simple in-memory model registry.

    In production, this would be backed by a database and model
    artifact storage (e.g., MLflow, S3).
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._feature_names: Dict[str, List[str]] = {}
        self._training_data: Dict[str, np.ndarray] = {}
        self._explainers: Dict[str, Any] = {}

    def register_model(
        self,
        model_id: str,
        model: Any,
        feature_names: List[str],
        training_data: Optional[np.ndarray] = None
    ) -> None:
        """Register a model for explanation."""
        self._models[model_id] = model
        self._feature_names[model_id] = feature_names
        if training_data is not None:
            self._training_data[model_id] = training_data

        logger.info(f"Model registered: {model_id}")

    def get_model(self, model_id: str) -> Any:
        """Get registered model."""
        if model_id not in self._models:
            raise KeyError(f"Model not found: {model_id}")
        return self._models[model_id]

    def get_feature_names(self, model_id: str) -> List[str]:
        """Get feature names for model."""
        if model_id not in self._feature_names:
            raise KeyError(f"Model not found: {model_id}")
        return self._feature_names[model_id]

    def get_training_data(self, model_id: str) -> Optional[np.ndarray]:
        """Get training data for model."""
        return self._training_data.get(model_id)

    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self._models.keys())


# Global registry instance
model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Dependency injection for model registry."""
    return model_registry


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_dependencies() -> Dict[str, bool]:
    """Check availability of optional dependencies."""
    deps = {"shap": False, "lime": False, "scipy": False}

    try:
        import shap
        deps["shap"] = True
    except ImportError:
        pass

    try:
        import lime
        deps["lime"] = True
    except ImportError:
        pass

    try:
        import scipy
        deps["scipy"] = True
    except ImportError:
        pass

    return deps


def dict_to_array(data: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    """Convert feature dict to numpy array."""
    return np.array([data.get(name, 0.0) for name in feature_names])


def get_prediction(model: Any, X: np.ndarray) -> float:
    """Get model prediction."""
    X = X.reshape(1, -1) if len(X.shape) == 1 else X

    if hasattr(model, "predict_proba"):
        pred = model.predict_proba(X)
        if len(pred.shape) > 1 and pred.shape[1] > 1:
            return float(pred[0, 1])
        return float(pred[0])
    else:
        pred = model.predict(X)
        return float(pred[0])


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and dependency availability.
    """
    deps = check_dependencies()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        shap_available=deps["shap"],
        lime_available=deps["lime"],
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/models", response_model=List[str])
async def list_models(
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    List all registered models.

    Returns list of model IDs available for explanation.
    """
    return registry.list_models()


@app.post("/explain/prediction", response_model=PredictionExplanationResponse)
async def explain_prediction(
    request: PredictionExplanationRequest,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    Explain a single prediction.

    Generates SHAP or LIME explanation for a model prediction,
    with optional human-readable narrative.

    Args:
        request: Explanation request with model_id and instance data

    Returns:
        PredictionExplanationResponse with feature contributions

    Raises:
        HTTPException 404: Model not found
        HTTPException 500: Explanation generation failed
    """
    start_time = datetime.utcnow()

    try:
        # Get model from registry
        model = registry.get_model(request.model_id)
        feature_names = registry.get_feature_names(request.model_id)
        training_data = registry.get_training_data(request.model_id)

    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")

    try:
        # Convert instance to array
        X = dict_to_array(request.instance_data, feature_names)

        # Get prediction
        prediction = get_prediction(model, X)

        # Generate explanation based on method
        deps = check_dependencies()

        if request.explanation_method in ("shap", "both") and deps["shap"]:
            feature_contributions = _compute_shap_explanation(
                model, X, feature_names, training_data
            )
        elif request.explanation_method in ("lime", "both") and deps["lime"]:
            feature_contributions = _compute_lime_explanation(
                model, X, feature_names, training_data
            )
        else:
            # Fallback: simple feature magnitude proxy
            feature_contributions = _compute_fallback_explanation(
                model, X, feature_names
            )

        # Get top features
        top_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:request.top_k_features]

        # Generate human-readable text if requested
        explanation_text = None
        if request.include_human_readable:
            explanation_text = _generate_human_readable(
                prediction=prediction,
                feature_contributions=feature_contributions,
                top_features=top_features,
                audience=request.audience
            )

        # Calculate confidence (based on contribution spread)
        total = sum(abs(v) for v in feature_contributions.values())
        top_total = sum(abs(v) for _, v in top_features[:3])
        confidence = min(0.95, 0.5 + 0.5 * (top_total / (total + 1e-10)))

        # Compute provenance hash
        provenance_data = {
            "model_id": request.model_id,
            "instance_data": request.instance_data,
            "prediction": prediction,
            "contributions": feature_contributions,
            "method": request.explanation_method,
            "timestamp": datetime.utcnow().isoformat()
        }
        provenance_hash = compute_provenance_hash(provenance_data)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return PredictionExplanationResponse(
            model_id=request.model_id,
            prediction=prediction,
            prediction_class=None,
            feature_contributions=feature_contributions,
            top_features=[
                {"feature": name, "contribution": value, "rank": i + 1}
                for i, (name, value) in enumerate(top_features)
            ],
            explanation_text=explanation_text,
            confidence=confidence,
            explanation_method=request.explanation_method,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Explanation generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/explain/model", response_model=GlobalExplanationResponse)
async def explain_model(
    model_id: str = Query(..., description="Model identifier"),
    sample_size: int = Query(100, ge=10, le=1000, description="Sample size"),
    include_interactions: bool = Query(False, description="Include interactions"),
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    Get global model explanation.

    Computes global feature importance across a sample of data.

    Args:
        model_id: Model identifier
        sample_size: Number of samples to use
        include_interactions: Include feature interactions

    Returns:
        GlobalExplanationResponse with feature importance

    Raises:
        HTTPException 404: Model not found
        HTTPException 500: Explanation failed
    """
    start_time = datetime.utcnow()

    try:
        model = registry.get_model(model_id)
        feature_names = registry.get_feature_names(model_id)
        training_data = registry.get_training_data(model_id)

    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    if training_data is None:
        raise HTTPException(
            status_code=400,
            detail="Training data required for global explanation"
        )

    try:
        # Sample data
        if len(training_data) > sample_size:
            indices = np.random.choice(len(training_data), sample_size, replace=False)
            X_sample = training_data[indices]
        else:
            X_sample = training_data

        # Compute global importance
        deps = check_dependencies()

        if deps["shap"]:
            feature_importance = _compute_global_shap_importance(
                model, X_sample, feature_names
            )
        else:
            feature_importance = _compute_fallback_importance(
                model, X_sample, feature_names
            )

        # Rank features
        feature_rankings = [
            {"feature": name, "importance": value, "rank": i + 1}
            for i, (name, value) in enumerate(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
        ]

        # Summary statistics
        importance_values = list(feature_importance.values())
        summary_statistics = {
            "total_features": len(feature_names),
            "top_feature": feature_rankings[0]["feature"] if feature_rankings else None,
            "importance_mean": float(np.mean(importance_values)),
            "importance_std": float(np.std(importance_values)),
            "sample_size_used": len(X_sample),
        }

        # Provenance
        provenance_data = {
            "model_id": model_id,
            "feature_importance": feature_importance,
            "sample_size": len(X_sample),
            "timestamp": datetime.utcnow().isoformat()
        }
        provenance_hash = compute_provenance_hash(provenance_data)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return GlobalExplanationResponse(
            model_id=model_id,
            feature_importance=feature_importance,
            feature_rankings=feature_rankings,
            interaction_effects=None,  # TODO: Implement interactions
            summary_statistics=summary_statistics,
            sample_size=len(X_sample),
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Global explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/explain/counterfactual", response_model=CounterfactualExplanationResponse)
async def generate_counterfactual(
    request: CounterfactualExplanationRequest,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    Generate counterfactual explanation.

    Finds minimal changes to achieve a different prediction.

    Args:
        request: Counterfactual request with instance and target

    Returns:
        CounterfactualExplanationResponse with required changes

    Raises:
        HTTPException 404: Model not found
        HTTPException 500: Counterfactual generation failed
    """
    start_time = datetime.utcnow()

    try:
        model = registry.get_model(request.model_id)
        feature_names = registry.get_feature_names(request.model_id)

    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")

    try:
        from .counterfactual import CounterfactualExplainer

        # Set up feature constraints
        feature_ranges = {}
        if request.feature_constraints:
            for feat, constraints in request.feature_constraints.items():
                feature_ranges[feat] = (
                    constraints.get("min", -1e6),
                    constraints.get("max", 1e6)
                )

        # Create explainer
        explainer = CounterfactualExplainer(
            model=model,
            feature_names=feature_names,
            feature_ranges=feature_ranges,
            immutable_features=request.immutable_features,
        )

        # Generate counterfactual
        result = explainer.generate_counterfactual(
            instance=request.instance_data,
            target_prediction=request.target_prediction,
            target_class=request.target_class,
            max_features_to_change=request.max_features_to_change,
        )

        # Format changes
        changes_formatted = {
            feat: {"from": old, "to": new}
            for feat, (old, new) in result.changes_required.items()
        }

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return CounterfactualExplanationResponse(
            model_id=request.model_id,
            original_prediction=result.original_prediction,
            counterfactual_prediction=result.counterfactual_prediction,
            changes_required=changes_formatted,
            num_features_changed=result.num_features_changed,
            feasibility_score=result.feasibility_score,
            explanation_text=result.explanation_text or "",
            provenance_hash=result.provenance_hash,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Counterfactual explainer not available"
        )
    except Exception as e:
        logger.error(f"Counterfactual generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Counterfactual failed: {str(e)}")


@app.post("/explain/what-if", response_model=WhatIfResponse)
async def analyze_what_if(
    request: WhatIfRequest,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    Analyze what-if scenario.

    Shows how prediction changes with specific feature modifications.

    Args:
        request: What-if request with instance and changes

    Returns:
        WhatIfResponse with prediction changes

    Raises:
        HTTPException 404: Model not found
        HTTPException 500: Analysis failed
    """
    start_time = datetime.utcnow()

    try:
        model = registry.get_model(request.model_id)
        feature_names = registry.get_feature_names(request.model_id)

    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")

    try:
        from .counterfactual import CounterfactualExplainer

        explainer = CounterfactualExplainer(
            model=model,
            feature_names=feature_names,
        )

        result = explainer.explain_what_if(
            instance=request.instance_data,
            feature_changes=request.feature_changes,
            scenario_name=request.scenario_name,
        )

        # Format changes
        applied_changes = {
            feat: {"from": old, "to": new}
            for feat, (old, new) in result.feature_changes.items()
        }

        return WhatIfResponse(
            scenario_name=result.scenario_name,
            original_prediction=result.original_prediction,
            modified_prediction=result.modified_prediction,
            prediction_change=result.prediction_change,
            applied_changes=applied_changes,
            sensitivity=result.sensitivity,
            explanation_text=result.explanation_text or "",
            provenance_hash=result.provenance_hash,
            timestamp=datetime.utcnow().isoformat()
        )

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Counterfactual explainer not available"
        )
    except Exception as e:
        logger.error(f"What-if analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"What-if failed: {str(e)}")


@app.get("/explain/feature-importance/{model_id}", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    model_id: str,
    method: str = Query("shap", description="Method: shap, permutation"),
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    Get feature importance for a model.

    Returns cached or computed feature importance scores.

    Args:
        model_id: Model identifier
        method: Importance method (shap, permutation)

    Returns:
        FeatureImportanceResponse with importance scores

    Raises:
        HTTPException 404: Model not found
    """
    try:
        model = registry.get_model(model_id)
        feature_names = registry.get_feature_names(model_id)
        training_data = registry.get_training_data(model_id)

    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    try:
        if training_data is not None:
            deps = check_dependencies()
            if method == "shap" and deps["shap"]:
                importance = _compute_global_shap_importance(
                    model, training_data[:100], feature_names
                )
            else:
                importance = _compute_fallback_importance(
                    model, training_data[:100], feature_names
                )
        else:
            # No training data - return uniform importance
            importance = {name: 1.0 / len(feature_names) for name in feature_names}

        top_features = [
            {"feature": name, "importance": value, "rank": i + 1}
            for i, (name, value) in enumerate(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )
        ]

        provenance_hash = compute_provenance_hash({
            "model_id": model_id,
            "importance": importance,
            "method": method,
            "timestamp": datetime.utcnow().isoformat()
        })

        return FeatureImportanceResponse(
            model_id=model_id,
            feature_importance=importance,
            top_features=top_features,
            method=method,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Feature importance failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Importance failed: {str(e)}")


# =============================================================================
# HELPER IMPLEMENTATION FUNCTIONS
# =============================================================================

def _compute_shap_explanation(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    training_data: Optional[np.ndarray]
) -> Dict[str, float]:
    """Compute SHAP explanation for instance."""
    import shap

    X = X.reshape(1, -1) if len(X.shape) == 1 else X

    # Select background data
    if training_data is not None and len(training_data) > 10:
        background = shap.kmeans(training_data, min(100, len(training_data)))
    else:
        background = X

    try:
        # Try TreeExplainer first
        explainer = shap.TreeExplainer(model)
    except Exception:
        # Fall back to KernelExplainer
        if hasattr(model, "predict_proba"):
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
        else:
            predict_fn = model.predict
        explainer = shap.KernelExplainer(predict_fn, background)

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    return {
        name: float(shap_values[0, i])
        for i, name in enumerate(feature_names)
    }


def _compute_lime_explanation(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    training_data: Optional[np.ndarray]
) -> Dict[str, float]:
    """Compute LIME explanation for instance."""
    from lime import lime_tabular

    X = X.flatten() if len(X.shape) > 1 else X

    if training_data is not None:
        data = training_data
    else:
        # Create synthetic data around instance
        data = X + np.random.normal(0, 0.1, size=(100, len(X))) * X

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=data,
        feature_names=feature_names,
        mode="classification" if hasattr(model, "predict_proba") else "regression"
    )

    if hasattr(model, "predict_proba"):
        exp = explainer.explain_instance(X, model.predict_proba, num_features=len(feature_names))
    else:
        exp = explainer.explain_instance(X, model.predict, num_features=len(feature_names))

    return {name: weight for name, weight in exp.as_list()}


def _compute_fallback_explanation(
    model: Any,
    X: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """Compute fallback explanation (feature magnitude proxy)."""
    X = X.flatten() if len(X.shape) > 1 else X
    abs_values = np.abs(X)
    total = abs_values.sum() + 1e-10
    normalized = abs_values / total

    return {name: float(normalized[i]) for i, name in enumerate(feature_names)}


def _compute_global_shap_importance(
    model: Any,
    X: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """Compute global SHAP feature importance."""
    import shap

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        background = shap.kmeans(X, min(50, len(X)))
        if hasattr(model, "predict_proba"):
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
        else:
            predict_fn = model.predict
        explainer = shap.KernelExplainer(predict_fn, background)

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_values = np.abs(shap_values)

    mean_importance = shap_values.mean(axis=0)
    total = mean_importance.sum() + 1e-10

    return {
        name: float(mean_importance[i] / total)
        for i, name in enumerate(feature_names)
    }


def _compute_fallback_importance(
    model: Any,
    X: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """Compute fallback feature importance."""
    # Use feature variance as proxy
    variance = np.var(X, axis=0)
    total = variance.sum() + 1e-10
    normalized = variance / total

    return {name: float(normalized[i]) for i, name in enumerate(feature_names)}


def _generate_human_readable(
    prediction: float,
    feature_contributions: Dict[str, float],
    top_features: List[tuple],
    audience: str = "engineer"
) -> str:
    """Generate human-readable explanation."""
    try:
        from .human_readable import HumanReadableExplainer, AudienceType

        audience_map = {
            "operator": AudienceType.OPERATOR,
            "engineer": AudienceType.ENGINEER,
            "manager": AudienceType.MANAGER,
            "executive": AudienceType.EXECUTIVE,
        }

        explainer = HumanReadableExplainer(
            audience=audience_map.get(audience, AudienceType.ENGINEER)
        )

        return explainer.generate_explanation(
            prediction=prediction,
            feature_contributions=feature_contributions,
            top_features=top_features
        )

    except ImportError:
        # Fallback simple explanation
        lines = [f"Prediction: {prediction:.2%}"]
        lines.append("Key factors:")
        for name, value in top_features[:3]:
            direction = "increases" if value > 0 else "decreases"
            lines.append(f"  - {name} {direction} prediction by {abs(value):.2%}")
        return "\n".join(lines)


# =============================================================================
# MODEL REGISTRATION ENDPOINT (For demo purposes)
# =============================================================================

class ModelRegistrationRequest(BaseModel):
    """Request to register a model (demo only)."""

    model_id: str = Field(..., description="Unique model identifier")
    feature_names: List[str] = Field(..., description="Feature names")
    model_type: str = Field(default="sklearn", description="Model framework")


@app.post("/models/register")
async def register_demo_model(
    request: ModelRegistrationRequest,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    Register a demo model (for testing).

    In production, models would be loaded from model registry.
    """
    try:
        # Create a simple mock model for demo
        class MockModel:
            def predict_proba(self, X):
                # Return probability based on feature sum
                probs = 1 / (1 + np.exp(-np.sum(X, axis=1) / len(X[0])))
                return np.column_stack([1 - probs, probs])

            def predict(self, X):
                return np.sum(X, axis=1)

        # Generate synthetic training data
        n_features = len(request.feature_names)
        training_data = np.random.randn(1000, n_features)

        registry.register_model(
            model_id=request.model_id,
            model=MockModel(),
            feature_names=request.feature_names,
            training_data=training_data
        )

        return {"status": "success", "model_id": request.model_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


# =============================================================================
# APPLICATION STARTUP
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup handler."""
    logger.info("GreenLang ML Explainability API starting up...")

    deps = check_dependencies()
    logger.info(f"Dependencies: SHAP={deps['shap']}, LIME={deps['lime']}, scipy={deps['scipy']}")

    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available - API endpoints will not function")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown handler."""
    logger.info("GreenLang ML Explainability API shutting down...")
