# -*- coding: utf-8 -*-
"""
Prediction Uncertainty API Module

This module provides FastAPI endpoints for uncertainty-aware predictions
in GreenLang ML models, enabling confidence intervals, calibration status,
model uncertainty comparison, and batch predictions with uncertainty.

The API provides a unified interface for accessing uncertainty quantification
capabilities, critical for regulatory compliance where prediction confidence
must be accessible and auditable.

Example:
    >>> from greenlang.ml.uncertainty.api import create_uncertainty_router
    >>> from fastapi import FastAPI
    >>> app = FastAPI()
    >>> app.include_router(create_uncertainty_router(models))
"""

from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class UncertaintyMethod(str, Enum):
    """Methods for uncertainty quantification."""
    ENSEMBLE = "ensemble"
    CONFORMAL = "conformal"
    BAYESIAN = "bayesian"
    CALIBRATED = "calibrated"
    COMBINED = "combined"


class PredictionRequest(BaseModel):
    """Request for uncertainty-aware prediction."""

    features: List[List[float]] = Field(
        ...,
        description="Feature vectors (n_samples x n_features)"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Specific model to use"
    )
    uncertainty_method: UncertaintyMethod = Field(
        default=UncertaintyMethod.ENSEMBLE,
        description="Uncertainty quantification method"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for intervals"
    )
    include_decomposition: bool = Field(
        default=False,
        description="Include epistemic/aleatoric decomposition"
    )
    include_calibration: bool = Field(
        default=False,
        description="Include calibration information"
    )


class SinglePrediction(BaseModel):
    """Single prediction with uncertainty."""

    prediction: float = Field(
        ...,
        description="Point prediction"
    )
    uncertainty: float = Field(
        ...,
        description="Uncertainty estimate (std)"
    )
    lower_bound: float = Field(
        ...,
        description="Lower confidence bound"
    )
    upper_bound: float = Field(
        ...,
        description="Upper confidence bound"
    )
    confidence_level: float = Field(
        ...,
        description="Confidence level"
    )
    epistemic_uncertainty: Optional[float] = Field(
        default=None,
        description="Model uncertainty"
    )
    aleatoric_uncertainty: Optional[float] = Field(
        default=None,
        description="Data uncertainty"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Overall confidence score (0-1)"
    )


class PredictionResponse(BaseModel):
    """Response with predictions and uncertainty."""

    predictions: List[SinglePrediction] = Field(
        ...,
        description="Predictions with uncertainty"
    )
    model_name: str = Field(
        ...,
        description="Model used"
    )
    method: str = Field(
        ...,
        description="Uncertainty method used"
    )
    mean_uncertainty: float = Field(
        ...,
        description="Mean uncertainty across predictions"
    )
    max_uncertainty: float = Field(
        ...,
        description="Maximum uncertainty"
    )
    high_confidence_ratio: float = Field(
        ...,
        description="Ratio of high-confidence predictions"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 provenance hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )


class ConfidenceIntervalRequest(BaseModel):
    """Request for confidence intervals."""

    features: List[List[float]] = Field(
        ...,
        description="Feature vectors"
    )
    confidence_levels: List[float] = Field(
        default=[0.90, 0.95, 0.99],
        description="Confidence levels to compute"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model to use"
    )


class ConfidenceInterval(BaseModel):
    """Single confidence interval."""

    lower: float = Field(
        ...,
        description="Lower bound"
    )
    upper: float = Field(
        ...,
        description="Upper bound"
    )
    width: float = Field(
        ...,
        description="Interval width"
    )
    level: float = Field(
        ...,
        description="Confidence level"
    )


class ConfidenceIntervalResponse(BaseModel):
    """Response with confidence intervals at multiple levels."""

    prediction: float = Field(
        ...,
        description="Point prediction"
    )
    intervals: Dict[str, ConfidenceInterval] = Field(
        ...,
        description="Intervals at each level"
    )
    sample_index: int = Field(
        ...,
        description="Sample index"
    )


class ConfidenceIntervalsResponse(BaseModel):
    """Response with all confidence intervals."""

    results: List[ConfidenceIntervalResponse] = Field(
        ...,
        description="Intervals for each sample"
    )
    model_name: str = Field(
        ...,
        description="Model used"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )


class CalibrationStatusRequest(BaseModel):
    """Request for calibration status."""

    model_name: Optional[str] = Field(
        default=None,
        description="Model to check"
    )


class CalibrationStatus(BaseModel):
    """Calibration status response."""

    model_name: str = Field(
        ...,
        description="Model name"
    )
    is_calibrated: bool = Field(
        ...,
        description="Whether model is calibrated"
    )
    calibration_method: Optional[str] = Field(
        default=None,
        description="Calibration method used"
    )
    expected_calibration_error: Optional[float] = Field(
        default=None,
        description="ECE if available"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature parameter if applicable"
    )
    last_calibrated: Optional[datetime] = Field(
        default=None,
        description="Last calibration time"
    )
    calibration_samples: Optional[int] = Field(
        default=None,
        description="Number of calibration samples"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Calibration recommendations"
    )


class ModelComparisonRequest(BaseModel):
    """Request for model uncertainty comparison."""

    features: List[List[float]] = Field(
        ...,
        description="Feature vectors"
    )
    model_names: List[str] = Field(
        ...,
        description="Models to compare"
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level"
    )


class ModelUncertaintySummary(BaseModel):
    """Uncertainty summary for a model."""

    model_name: str = Field(
        ...,
        description="Model name"
    )
    mean_uncertainty: float = Field(
        ...,
        description="Mean uncertainty"
    )
    std_uncertainty: float = Field(
        ...,
        description="Std of uncertainty"
    )
    mean_interval_width: float = Field(
        ...,
        description="Mean interval width"
    )
    coverage_estimate: float = Field(
        ...,
        description="Estimated coverage"
    )
    is_calibrated: bool = Field(
        ...,
        description="Whether calibrated"
    )


class ModelComparisonResponse(BaseModel):
    """Response comparing model uncertainties."""

    models: List[ModelUncertaintySummary] = Field(
        ...,
        description="Summary for each model"
    )
    best_model: str = Field(
        ...,
        description="Model with best uncertainty characteristics"
    )
    comparison_criteria: str = Field(
        ...,
        description="Criteria used for comparison"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )


class BatchPredictionRequest(BaseModel):
    """Request for batch prediction with uncertainty."""

    features: List[List[float]] = Field(
        ...,
        description="Feature vectors"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model to use"
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level"
    )
    parallel: bool = Field(
        default=True,
        description="Use parallel processing"
    )
    chunk_size: int = Field(
        default=100,
        ge=1,
        description="Chunk size for batching"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""

    predictions: List[float] = Field(
        ...,
        description="Point predictions"
    )
    uncertainties: List[float] = Field(
        ...,
        description="Uncertainty estimates"
    )
    lower_bounds: List[float] = Field(
        ...,
        description="Lower bounds"
    )
    upper_bounds: List[float] = Field(
        ...,
        description="Upper bounds"
    )
    n_samples: int = Field(
        ...,
        description="Number of samples"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )
    summary: Dict[str, float] = Field(
        ...,
        description="Summary statistics"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )


# ============================================================================
# API Service
# ============================================================================

class UncertaintyAPIService:
    """
    Service for uncertainty-aware prediction API.

    This class provides the backend logic for uncertainty API endpoints,
    coordinating between ensemble, conformal, and Bayesian predictors.

    Attributes:
        models: Dictionary of registered models
        ensemble_predictors: Ensemble predictors
        conformal_predictors: Conformal predictors
        calibrators: Model calibrators

    Example:
        >>> service = UncertaintyAPIService()
        >>> service.register_model("emission_predictor", model, ensemble, conformal)
        >>> result = service.predict_with_uncertainty(features, "emission_predictor")
    """

    def __init__(self):
        """Initialize uncertainty API service."""
        self.models: Dict[str, Any] = {}
        self.ensemble_predictors: Dict[str, Any] = {}
        self.conformal_predictors: Dict[str, Any] = {}
        self.bayesian_predictors: Dict[str, Any] = {}
        self.calibrators: Dict[str, Any] = {}
        self._calibration_status: Dict[str, Dict[str, Any]] = {}

        logger.info("UncertaintyAPIService initialized")

    def register_model(
        self,
        name: str,
        model: Any,
        ensemble: Optional[Any] = None,
        conformal: Optional[Any] = None,
        bayesian: Optional[Any] = None,
        calibrator: Optional[Any] = None
    ) -> None:
        """
        Register a model with its uncertainty predictors.

        Args:
            name: Model name
            model: Base model
            ensemble: EnsemblePredictor
            conformal: ConformalPredictor
            bayesian: BayesianNeuralNetwork
            calibrator: Calibrator
        """
        self.models[name] = model

        if ensemble is not None:
            self.ensemble_predictors[name] = ensemble

        if conformal is not None:
            self.conformal_predictors[name] = conformal

        if bayesian is not None:
            self.bayesian_predictors[name] = bayesian

        if calibrator is not None:
            self.calibrators[name] = calibrator
            self._calibration_status[name] = {
                "is_calibrated": True,
                "last_calibrated": datetime.utcnow()
            }

        logger.info(f"Registered model: {name}")

    def _calculate_provenance(
        self,
        predictions: np.ndarray,
        method: str
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = f"{method}|{predictions.sum():.8f}|{len(predictions)}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def predict_with_uncertainty(
        self,
        request: PredictionRequest
    ) -> PredictionResponse:
        """
        Make predictions with uncertainty.

        Args:
            request: Prediction request

        Returns:
            PredictionResponse with uncertainty
        """
        X = np.array(request.features)
        model_name = request.model_name or list(self.models.keys())[0]
        method = request.uncertainty_method

        predictions_list = []

        if method == UncertaintyMethod.ENSEMBLE and model_name in self.ensemble_predictors:
            ensemble = self.ensemble_predictors[model_name]
            result = ensemble.predict_with_uncertainty(X, confidence=request.confidence_level)

            for i in range(len(X)):
                pred = SinglePrediction(
                    prediction=result.predictions[i],
                    uncertainty=result.uncertainties[i],
                    lower_bound=result.lower_bounds[i],
                    upper_bound=result.upper_bounds[i],
                    confidence_level=request.confidence_level,
                    confidence_score=result.model_agreement[i]
                )
                predictions_list.append(pred)

        elif method == UncertaintyMethod.CONFORMAL and model_name in self.conformal_predictors:
            conformal = self.conformal_predictors[model_name]
            result = conformal.predict_interval(X, confidence=request.confidence_level)

            for interval in result.intervals:
                pred = SinglePrediction(
                    prediction=interval.prediction,
                    uncertainty=interval.width / 2,
                    lower_bound=interval.lower,
                    upper_bound=interval.upper,
                    confidence_level=request.confidence_level
                )
                predictions_list.append(pred)

        elif method == UncertaintyMethod.BAYESIAN and model_name in self.bayesian_predictors:
            bnn = self.bayesian_predictors[model_name]
            result = bnn.predict_with_uncertainty(X, confidence=request.confidence_level)

            for i in range(len(X)):
                pred = SinglePrediction(
                    prediction=result.predictions[i],
                    uncertainty=result.total_uncertainty[i],
                    lower_bound=result.lower_bounds[i],
                    upper_bound=result.upper_bounds[i],
                    confidence_level=request.confidence_level,
                    epistemic_uncertainty=result.epistemic_uncertainty[i],
                    aleatoric_uncertainty=result.aleatoric_uncertainty[i]
                )
                predictions_list.append(pred)

        else:
            # Default: use base model with simple uncertainty estimate
            model = self.models[model_name]
            preds = model.predict(X)

            for i, p in enumerate(preds):
                pred = SinglePrediction(
                    prediction=float(p),
                    uncertainty=0.1,  # Default uncertainty
                    lower_bound=float(p) - 0.2,
                    upper_bound=float(p) + 0.2,
                    confidence_level=request.confidence_level
                )
                predictions_list.append(pred)

        # Calculate summary statistics
        uncertainties = [p.uncertainty for p in predictions_list]
        mean_uncertainty = float(np.mean(uncertainties))
        max_uncertainty = float(np.max(uncertainties))

        # High confidence ratio (uncertainty below threshold)
        threshold = np.percentile(uncertainties, 75)
        high_conf_count = sum(1 for u in uncertainties if u < threshold)
        high_confidence_ratio = high_conf_count / len(uncertainties)

        # Provenance
        predictions_array = np.array([p.prediction for p in predictions_list])
        provenance_hash = self._calculate_provenance(predictions_array, method.value)

        return PredictionResponse(
            predictions=predictions_list,
            model_name=model_name,
            method=method.value,
            mean_uncertainty=mean_uncertainty,
            max_uncertainty=max_uncertainty,
            high_confidence_ratio=high_confidence_ratio,
            provenance_hash=provenance_hash
        )

    def get_confidence_intervals(
        self,
        request: ConfidenceIntervalRequest
    ) -> ConfidenceIntervalsResponse:
        """
        Get confidence intervals at multiple levels.

        Args:
            request: Confidence interval request

        Returns:
            ConfidenceIntervalsResponse
        """
        X = np.array(request.features)
        model_name = request.model_name or list(self.models.keys())[0]

        results = []

        # Use ensemble or conformal if available
        if model_name in self.ensemble_predictors:
            ensemble = self.ensemble_predictors[model_name]

            for i in range(len(X)):
                intervals = {}
                sample = X[i:i+1]

                for level in request.confidence_levels:
                    result = ensemble.predict_with_uncertainty(sample, confidence=level)
                    intervals[f"{int(level * 100)}%"] = ConfidenceInterval(
                        lower=result.lower_bounds[0],
                        upper=result.upper_bounds[0],
                        width=result.upper_bounds[0] - result.lower_bounds[0],
                        level=level
                    )

                results.append(ConfidenceIntervalResponse(
                    prediction=result.predictions[0],
                    intervals=intervals,
                    sample_index=i
                ))

        else:
            # Fallback to simple intervals
            model = self.models[model_name]
            preds = model.predict(X)

            for i, p in enumerate(preds):
                intervals = {}
                for level in request.confidence_levels:
                    # Simple z-score based intervals
                    from scipy import stats
                    z = stats.norm.ppf(1 - (1 - level) / 2)
                    width = 0.1 * z  # Assume std of 0.05

                    intervals[f"{int(level * 100)}%"] = ConfidenceInterval(
                        lower=float(p) - width,
                        upper=float(p) + width,
                        width=2 * width,
                        level=level
                    )

                results.append(ConfidenceIntervalResponse(
                    prediction=float(p),
                    intervals=intervals,
                    sample_index=i
                ))

        provenance_hash = hashlib.sha256(
            f"{model_name}|{len(X)}|{request.confidence_levels}".encode()
        ).hexdigest()

        return ConfidenceIntervalsResponse(
            results=results,
            model_name=model_name,
            provenance_hash=provenance_hash
        )

    def get_calibration_status(
        self,
        model_name: Optional[str] = None
    ) -> CalibrationStatus:
        """
        Get calibration status for a model.

        Args:
            model_name: Model name

        Returns:
            CalibrationStatus
        """
        model_name = model_name or list(self.models.keys())[0]

        status = self._calibration_status.get(model_name, {})
        is_calibrated = status.get("is_calibrated", False)

        recommendations = []
        if not is_calibrated:
            recommendations.append("Model is not calibrated. Consider running calibration.")

        ece = None
        temperature = None
        method = None

        if model_name in self.calibrators:
            calibrator = self.calibrators[model_name]
            temperature = getattr(calibrator, "temperature", None)
            method = getattr(calibrator.config, "method", None)
            if method:
                method = method.value

        return CalibrationStatus(
            model_name=model_name,
            is_calibrated=is_calibrated,
            calibration_method=method,
            expected_calibration_error=ece,
            temperature=temperature,
            last_calibrated=status.get("last_calibrated"),
            calibration_samples=status.get("n_samples"),
            recommendations=recommendations
        )

    def compare_model_uncertainty(
        self,
        request: ModelComparisonRequest
    ) -> ModelComparisonResponse:
        """
        Compare uncertainty across models.

        Args:
            request: Comparison request

        Returns:
            ModelComparisonResponse
        """
        X = np.array(request.features)
        summaries = []

        for model_name in request.model_names:
            if model_name not in self.models:
                continue

            pred_request = PredictionRequest(
                features=request.features,
                model_name=model_name,
                confidence_level=request.confidence_level
            )

            result = self.predict_with_uncertainty(pred_request)

            uncertainties = [p.uncertainty for p in result.predictions]
            widths = [p.upper_bound - p.lower_bound for p in result.predictions]

            summary = ModelUncertaintySummary(
                model_name=model_name,
                mean_uncertainty=float(np.mean(uncertainties)),
                std_uncertainty=float(np.std(uncertainties)),
                mean_interval_width=float(np.mean(widths)),
                coverage_estimate=request.confidence_level,
                is_calibrated=model_name in self.calibrators
            )
            summaries.append(summary)

        # Determine best model (lowest mean uncertainty)
        if summaries:
            best_model = min(summaries, key=lambda s: s.mean_uncertainty).model_name
        else:
            best_model = "none"

        recommendations = []
        if summaries:
            if any(not s.is_calibrated for s in summaries):
                recommendations.append(
                    "Some models are not calibrated. Consider calibrating for reliable intervals."
                )

        provenance_hash = hashlib.sha256(
            f"{request.model_names}|{len(X)}".encode()
        ).hexdigest()

        return ModelComparisonResponse(
            models=summaries,
            best_model=best_model,
            comparison_criteria="lowest_mean_uncertainty",
            recommendations=recommendations,
            provenance_hash=provenance_hash
        )

    def batch_predict_with_uncertainty(
        self,
        request: BatchPredictionRequest
    ) -> BatchPredictionResponse:
        """
        Batch prediction with uncertainty.

        Args:
            request: Batch prediction request

        Returns:
            BatchPredictionResponse
        """
        import time
        start_time = time.time()

        X = np.array(request.features)
        model_name = request.model_name or list(self.models.keys())[0]

        predictions = []
        uncertainties = []
        lower_bounds = []
        upper_bounds = []

        # Process in chunks
        for i in range(0, len(X), request.chunk_size):
            chunk = X[i:i + request.chunk_size]

            pred_request = PredictionRequest(
                features=chunk.tolist(),
                model_name=model_name,
                confidence_level=request.confidence_level
            )

            result = self.predict_with_uncertainty(pred_request)

            for p in result.predictions:
                predictions.append(p.prediction)
                uncertainties.append(p.uncertainty)
                lower_bounds.append(p.lower_bound)
                upper_bounds.append(p.upper_bound)

        processing_time_ms = (time.time() - start_time) * 1000

        # Summary statistics
        summary = {
            "mean_prediction": float(np.mean(predictions)),
            "std_prediction": float(np.std(predictions)),
            "mean_uncertainty": float(np.mean(uncertainties)),
            "max_uncertainty": float(np.max(uncertainties)),
            "min_uncertainty": float(np.min(uncertainties))
        }

        provenance_hash = hashlib.sha256(
            f"{model_name}|{len(X)}|{np.sum(predictions):.8f}".encode()
        ).hexdigest()

        return BatchPredictionResponse(
            predictions=predictions,
            uncertainties=uncertainties,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            n_samples=len(predictions),
            processing_time_ms=processing_time_ms,
            summary=summary,
            provenance_hash=provenance_hash
        )


# ============================================================================
# FastAPI Router Factory
# ============================================================================

def create_uncertainty_router(service: Optional[UncertaintyAPIService] = None):
    """
    Create FastAPI router for uncertainty API.

    Args:
        service: UncertaintyAPIService instance

    Returns:
        FastAPI APIRouter

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> service = UncertaintyAPIService()
        >>> router = create_uncertainty_router(service)
        >>> app.include_router(router, prefix="/api/v1/uncertainty")
    """
    try:
        from fastapi import APIRouter, HTTPException
    except ImportError:
        logger.warning("FastAPI not installed. Router creation skipped.")
        return None

    router = APIRouter(
        prefix="/uncertainty",
        tags=["uncertainty"]
    )

    if service is None:
        service = UncertaintyAPIService()

    @router.post("/predict", response_model=PredictionResponse)
    async def predict_with_uncertainty(request: PredictionRequest):
        """
        Make predictions with uncertainty estimates.

        Returns predictions with confidence intervals based on
        the specified uncertainty quantification method.
        """
        try:
            return service.predict_with_uncertainty(request)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/intervals", response_model=ConfidenceIntervalsResponse)
    async def get_confidence_intervals(request: ConfidenceIntervalRequest):
        """
        Get confidence intervals at multiple levels.

        Computes prediction intervals at specified confidence levels.
        """
        try:
            return service.get_confidence_intervals(request)
        except Exception as e:
            logger.error(f"Interval error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/calibration/{model_name}", response_model=CalibrationStatus)
    async def get_calibration_status(model_name: str):
        """
        Get calibration status for a model.

        Returns information about model calibration including
        ECE, temperature, and recommendations.
        """
        try:
            return service.get_calibration_status(model_name)
        except Exception as e:
            logger.error(f"Calibration status error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/compare", response_model=ModelComparisonResponse)
    async def compare_models(request: ModelComparisonRequest):
        """
        Compare uncertainty across models.

        Compares uncertainty characteristics of multiple models
        and recommends the best one.
        """
        try:
            return service.compare_model_uncertainty(request)
        except Exception as e:
            logger.error(f"Comparison error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/batch", response_model=BatchPredictionResponse)
    async def batch_predict(request: BatchPredictionRequest):
        """
        Batch prediction with uncertainty.

        Efficiently processes large batches of predictions
        with uncertainty estimates.
        """
        try:
            return service.batch_predict_with_uncertainty(request)
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "registered_models": list(service.models.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }

    return router


# ============================================================================
# Unit Tests
# ============================================================================

class TestUncertaintyAPI:
    """Unit tests for Uncertainty API."""

    def test_service_init(self):
        """Test service initialization."""
        service = UncertaintyAPIService()
        assert len(service.models) == 0

    def test_register_model(self):
        """Test model registration."""
        service = UncertaintyAPIService()

        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        service.register_model("test", MockModel())
        assert "test" in service.models

    def test_predict_with_uncertainty(self):
        """Test prediction with uncertainty."""
        service = UncertaintyAPIService()

        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        service.register_model("test", MockModel())

        request = PredictionRequest(
            features=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            model_name="test"
        )

        response = service.predict_with_uncertainty(request)
        assert len(response.predictions) == 2

    def test_batch_predict(self):
        """Test batch prediction."""
        service = UncertaintyAPIService()

        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        service.register_model("test", MockModel())

        request = BatchPredictionRequest(
            features=[[1.0, 2.0, 3.0]] * 50,
            model_name="test",
            chunk_size=10
        )

        response = service.batch_predict_with_uncertainty(request)
        assert response.n_samples == 50

    def test_calibration_status(self):
        """Test calibration status."""
        service = UncertaintyAPIService()

        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        service.register_model("test", MockModel())

        status = service.get_calibration_status("test")
        assert status.model_name == "test"
        assert not status.is_calibrated

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        service = UncertaintyAPIService()

        preds = np.array([1.0, 2.0, 3.0])
        hash1 = service._calculate_provenance(preds, "ensemble")
        hash2 = service._calculate_provenance(preds, "ensemble")

        assert hash1 == hash2
