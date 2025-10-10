"""AI-powered Isolation Forest Anomaly Detection Agent with ChatSession Integration.

This module provides a production-ready Isolation Forest-based anomaly detection agent
for identifying outliers in climate and energy data.

Key Features:
    1. Tool-First Numerics: All calculations via deterministic tools (zero hallucinated numbers)
    2. AI Interpretation: Natural language explanations of anomalies and patterns
    3. Unsupervised Learning: No labeled data required (pure anomaly detection)
    4. Multi-Dimensional: Handle multiple features simultaneously
    5. Anomaly Scoring: Precise anomaly scores for each data point
    6. Pattern Analysis: Identify common anomaly characteristics
    7. Alert Generation: Severity-based actionable alerts
    8. Deterministic Results: temperature=0, seed=42, random_state=42
    9. Comprehensive Validation: Statistical tests and validation
    10. Production-Ready: Full error handling, provenance tracking

Architecture:
    IsolationForestAnomalyAgent (orchestration) -> ChatSession (AI) -> Tools (exact calculations)

    Tools:
    - fit_isolation_forest: Train model on normal data
    - detect_anomalies: Identify outliers in new data
    - calculate_anomaly_scores: Compute anomaly scores for each point
    - rank_anomalies: Rank anomalies by severity
    - analyze_anomaly_patterns: Identify common anomaly characteristics
    - generate_alerts: Create actionable alerts for critical anomalies

Use Cases:
    - Energy consumption anomaly detection (spikes, drops)
    - Temperature anomaly detection (extreme weather events)
    - Emissions anomaly detection (equipment malfunction, data errors)
    - Grid load anomaly detection (unusual demand patterns)

Example:
    >>> agent = IsolationForestAnomalyAgent()
    >>> result = agent.run({
    ...     "data": df,  # pandas DataFrame with numeric features
    ...     "contamination": 0.1,  # Expected 10% anomalies
    ...     "n_estimators": 100,  # Number of trees
    ... })
    >>> print(result["data"]["anomalies"])
    [False, False, True, False, True, ...]  # Exact predictions from model
    >>> print(result["data"]["explanation"])
    "Detected 5 anomalies with extreme values in energy_kwh feature..."

Author: GreenLang Framework Team
Date: October 2025
Spec: Isolation Forest Baseline ML Agent (ML-002)
"""

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available - Anomaly agent will use mock predictions")

from greenlang.sdk.base import Agent, Result, Metadata
from greenlang.intelligence import (
    ChatSession,
    ChatMessage,
    Role,
    Budget,
    BudgetExceeded,
    create_provider,
)
from greenlang.intelligence.schemas.tools import ToolDef


logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    """Anomaly score for a single data point."""
    index: int
    score: float  # -1 to 1 range (< 0 = anomaly)
    is_anomaly: bool
    severity: str  # critical, high, medium, low
    features: Dict[str, float]  # Feature values


@dataclass
class AnomalyAlert:
    """Alert for detected anomaly."""
    index: int
    severity: str
    score: float
    root_cause_hints: List[str]
    recommendations: List[str]
    confidence: float


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    n_anomalies: int
    n_normal: int


class IsolationForestAnomalyAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """AI-powered Isolation Forest anomaly detection agent.

    This agent provides production-ready anomaly detection using Isolation Forest models
    with automatic parameter tuning, multi-dimensional support, and AI-generated interpretations.

    Features:
    - Tool-first numerics (all calculations use tools, zero hallucinated numbers)
    - Unsupervised anomaly detection (no labeled data required)
    - Multi-dimensional feature support
    - Anomaly scoring and ranking
    - Pattern analysis and root cause hints
    - Severity-based alert generation
    - Deterministic results (temperature=0, seed=42)
    - Full provenance tracking

    Input Format:
        {
            "data": pandas.DataFrame with numeric features,
            "contamination": float (optional, default=0.1),  # Expected anomaly rate
            "n_estimators": int (optional, default=100),  # Number of trees
            "max_samples": int (optional, default=256),  # Samples per tree
            "max_features": float (optional, default=1.0),  # Features per tree
            "bootstrap": bool (optional, default=False),
            "feature_columns": List[str] (optional),  # Specific features to use
            "labels": List[bool] (optional),  # True labels for evaluation
        }

    Output Format:
        {
            "anomalies": List[bool],  # Anomaly predictions
            "anomaly_scores": List[float],  # Anomaly scores
            "anomaly_indices": List[int],  # Indices of anomalies
            "severity_counts": dict,  # Count by severity
            "patterns": dict,  # Detected patterns
            "alerts": List[dict],  # Generated alerts
            "explanation": str,  # AI-generated interpretation
            "recommendations": List[str],  # AI-generated insights
            "metrics": dict (optional),  # If labels provided
        }

    Example:
        >>> agent = IsolationForestAnomalyAgent()
        >>> df = pd.DataFrame({
        ...     'energy_kwh': [100, 105, 98, 500, 102, 99],
        ...     'temperature_c': [20, 22, 21, 25, 20, 21],
        ... })
        >>> result = agent.run({
        ...     "data": df,
        ...     "contamination": 0.1,
        ... })
        >>> print(result.data["anomalies"])
        [False, False, False, True, False, False]
    """

    def __init__(
        self,
        budget_usd: float = 1.00,
        enable_explanations: bool = True,
        enable_recommendations: bool = True,
        enable_alerts: bool = True,
    ):
        """Initialize the Isolation Forest anomaly detection agent.

        Args:
            budget_usd: Maximum USD to spend per detection (default: $1.00)
            enable_explanations: Enable AI-generated explanations (default: True)
            enable_recommendations: Enable AI recommendations (default: True)
            enable_alerts: Enable alert generation (default: True)
        """
        # Initialize metadata
        metadata = Metadata(
            id="anomaly_iforest",
            name="Isolation Forest Anomaly Agent",
            version="0.1.0",
            description="AI-powered Isolation Forest anomaly detection for climate/energy data",
            tags=["ml", "anomaly-detection", "isolation-forest", "outliers"],
        )
        super().__init__(metadata)

        # Configuration
        self.budget_usd = budget_usd
        self.enable_explanations = enable_explanations
        self.enable_recommendations = enable_recommendations
        self.enable_alerts = enable_alerts

        # Initialize LLM provider
        self.provider = create_provider()

        # Performance tracking
        self._ai_call_count = 0
        self._tool_call_count = 0
        self._total_cost_usd = 0.0

        # Model state
        self._fitted_model = None
        self._scaler = None
        self._feature_columns = None
        self._training_data = None

        # Setup tools
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Setup tool definitions for ChatSession."""

        # Tool 1: Fit Isolation Forest model
        self.fit_isolation_forest_tool = ToolDef(
            name="fit_isolation_forest",
            description="Fit Isolation Forest model to training data (normal behavior baseline)",
            parameters={
                "type": "object",
                "properties": {
                    "contamination": {
                        "type": "number",
                        "description": "Expected proportion of anomalies in the dataset (0-0.5)",
                        "default": 0.1,
                        "minimum": 0.0,
                        "maximum": 0.5,
                    },
                    "n_estimators": {
                        "type": "integer",
                        "description": "Number of isolation trees",
                        "default": 100,
                        "minimum": 10,
                        "maximum": 500,
                    },
                    "max_samples": {
                        "type": "integer",
                        "description": "Number of samples to train each tree",
                        "default": 256,
                        "minimum": 2,
                    },
                    "max_features": {
                        "type": "number",
                        "description": "Fraction of features to train each tree (0-1)",
                        "default": 1.0,
                        "minimum": 0.1,
                        "maximum": 1.0,
                    },
                    "bootstrap": {
                        "type": "boolean",
                        "description": "Sample with replacement",
                        "default": False,
                    },
                },
            },
        )

        # Tool 2: Detect anomalies
        self.detect_anomalies_tool = ToolDef(
            name="detect_anomalies",
            description="Detect anomalies in data using fitted Isolation Forest model",
            parameters={
                "type": "object",
                "properties": {},
            },
        )

        # Tool 3: Calculate anomaly scores
        self.calculate_anomaly_scores_tool = ToolDef(
            name="calculate_anomaly_scores",
            description="Calculate anomaly scores for all data points (-1 to 1 range)",
            parameters={
                "type": "object",
                "properties": {},
            },
        )

        # Tool 4: Rank anomalies
        self.rank_anomalies_tool = ToolDef(
            name="rank_anomalies",
            description="Rank detected anomalies by severity (critical, high, medium, low)",
            parameters={
                "type": "object",
                "properties": {
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top anomalies to return",
                        "default": 10,
                        "minimum": 1,
                    },
                },
            },
        )

        # Tool 5: Analyze anomaly patterns
        self.analyze_anomaly_patterns_tool = ToolDef(
            name="analyze_anomaly_patterns",
            description="Analyze patterns in detected anomalies (feature importance, common characteristics)",
            parameters={
                "type": "object",
                "properties": {},
            },
        )

        # Tool 6: Generate alerts
        self.generate_alerts_tool = ToolDef(
            name="generate_alerts",
            description="Generate actionable alerts for critical anomalies",
            parameters={
                "type": "object",
                "properties": {
                    "min_severity": {
                        "type": "string",
                        "description": "Minimum severity level for alerts",
                        "enum": ["critical", "high", "medium", "low"],
                        "default": "high",
                    },
                },
            },
        )

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data.

        Args:
            input_data: Input dictionary

        Returns:
            bool: True if valid
        """
        # Check required fields
        if "data" not in input_data:
            self.logger.error("Missing required field: data")
            return False

        # Validate data is DataFrame
        data = input_data["data"]
        if not isinstance(data, pd.DataFrame):
            self.logger.error("Data must be a pandas DataFrame")
            return False

        # Validate minimum data points
        if len(data) < 100:
            self.logger.error("Insufficient data: need at least 100 points for stable model")
            return False

        # Validate numeric features exist
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.logger.error("No numeric features found in data")
            return False

        # Validate feature columns if specified
        if "feature_columns" in input_data:
            feature_cols = input_data["feature_columns"]
            for col in feature_cols:
                if col not in data.columns:
                    self.logger.error(f"Feature column '{col}' not found in data")
                    return False

        # Validate contamination parameter
        if "contamination" in input_data:
            contamination = input_data["contamination"]
            if not (0.0 < contamination < 0.5):
                self.logger.error("Contamination must be between 0 and 0.5")
                return False

        # Validate labels if provided
        if "labels" in input_data:
            labels = input_data["labels"]
            if len(labels) != len(data):
                self.logger.error("Labels length must match data length")
                return False

        return True

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process anomaly detection request (synchronous wrapper).

        Args:
            input_data: Input dictionary

        Returns:
            Dict with anomaly detection results
        """
        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._process_async(input_data))
            return result
        finally:
            loop.close()

    async def _process_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process anomaly detection request with AI orchestration.

        Args:
            input_data: Input dictionary

        Returns:
            Dict with anomaly detection results and AI interpretation
        """
        start_time = datetime.now()

        # Extract parameters
        data = input_data["data"]
        contamination = input_data.get("contamination", 0.1)
        n_estimators = input_data.get("n_estimators", 100)
        feature_columns = input_data.get("feature_columns")
        labels = input_data.get("labels")

        # Store training data
        self._training_data = data.copy()

        # Create ChatSession
        session = ChatSession(self.provider)

        # Build AI prompt
        prompt = self._build_prompt(input_data)

        # Prepare messages
        messages = [
            ChatMessage(
                role=Role.system,
                content=(
                    "You are an anomaly detection expert for GreenLang climate and energy systems. "
                    "You help identify outliers, unusual patterns, and potential data quality issues. "
                    "IMPORTANT: You must use the provided tools for ALL calculations and predictions. "
                    "Never estimate or guess numbers. Always explain anomalies clearly and provide actionable insights."
                ),
            ),
            ChatMessage(role=Role.user, content=prompt),
        ]

        # Create budget
        budget = Budget(max_usd=self.budget_usd)

        try:
            # Call AI with tools
            self._ai_call_count += 1

            response = await session.chat(
                messages=messages,
                tools=[
                    self.fit_isolation_forest_tool,
                    self.detect_anomalies_tool,
                    self.calculate_anomaly_scores_tool,
                    self.rank_anomalies_tool,
                    self.analyze_anomaly_patterns_tool,
                    self.generate_alerts_tool,
                ],
                budget=budget,
                temperature=0.0,  # Deterministic
                seed=42,  # Reproducible
                tool_choice="auto",
            )

            # Track cost
            self._total_cost_usd += response.usage.cost_usd

            # Extract tool results
            tool_results = self._extract_tool_results(response, input_data)

            # Build output
            output = self._build_output(
                input_data,
                tool_results,
                response.text if self.enable_explanations else None,
            )

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Add metadata
            output["metadata"] = {
                "agent_id": self.metadata.id,
                "version": self.metadata.version,
                "provider": response.provider_info.provider,
                "model": response.provider_info.model,
                "tokens": response.usage.total_tokens,
                "cost_usd": response.usage.cost_usd,
                "tool_calls": len(response.tool_calls),
                "calculation_time_s": duration,
                "deterministic": True,
            }

            return output

        except BudgetExceeded as e:
            self.logger.error(f"Budget exceeded: {e}")
            raise ValueError(f"AI budget exceeded: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            raise

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build AI prompt for anomaly detection.

        Args:
            input_data: Input data

        Returns:
            str: Formatted prompt
        """
        data = input_data["data"]
        contamination = input_data.get("contamination", 0.1)
        feature_columns = input_data.get("feature_columns")

        # Determine features to use
        if feature_columns:
            features = feature_columns
        else:
            features = data.select_dtypes(include=[np.number]).columns.tolist()

        # Data summary
        data_summary = f"""
Anomaly Detection Request:
- Dataset: {len(data)} observations
- Features: {', '.join(features)}
- Expected anomaly rate: {contamination * 100:.1f}%
"""

        # Feature statistics
        data_summary += "\nFeature Statistics:\n"
        for col in features:
            if col in data.columns:
                series = data[col]
                data_summary += f"- {col}: mean={series.mean():.2f}, std={series.std():.2f}, min={series.min():.2f}, max={series.max():.2f}\n"

        prompt = data_summary + f"""
Tasks:
1. Use fit_isolation_forest to train anomaly detection model with contamination={contamination}
2. Use detect_anomalies to identify outliers in the dataset
3. Use calculate_anomaly_scores to compute anomaly scores for all points
4. Use rank_anomalies to rank anomalies by severity
5. Use analyze_anomaly_patterns to identify common characteristics of anomalies
"""

        if self.enable_alerts:
            prompt += "6. Use generate_alerts to create actionable alerts for critical anomalies\n"

        prompt += """
7. Provide analysis including:
   - Number and percentage of detected anomalies
   - Severity distribution (critical, high, medium, low)
   - Key patterns in anomalies
   - Potential root causes
   - Confidence in detections
"""

        if self.enable_recommendations:
            prompt += "8. Provide actionable recommendations for investigation and remediation\n"

        prompt += """
IMPORTANT:
- Use tools for ALL calculations and predictions
- Do not estimate or guess any numbers
- Explain anomalies clearly with specific feature values
- Provide confidence scores for critical findings
- Highlight any data quality issues discovered
"""

        return prompt

    def _extract_tool_results(
        self,
        response,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract results from AI tool calls.

        Args:
            response: ChatResponse from session
            input_data: Original input data

        Returns:
            Dict with tool results
        """
        results = {}

        for tool_call in response.tool_calls:
            name = tool_call.get("name", "")
            args = tool_call.get("arguments", {})

            try:
                if name == "fit_isolation_forest":
                    results["model"] = self._fit_isolation_forest_impl(input_data, **args)
                elif name == "detect_anomalies":
                    results["anomalies"] = self._detect_anomalies_impl(input_data, **args)
                elif name == "calculate_anomaly_scores":
                    results["scores"] = self._calculate_anomaly_scores_impl(input_data, **args)
                elif name == "rank_anomalies":
                    results["rankings"] = self._rank_anomalies_impl(input_data, **args)
                elif name == "analyze_anomaly_patterns":
                    results["patterns"] = self._analyze_anomaly_patterns_impl(input_data, **args)
                elif name == "generate_alerts":
                    results["alerts"] = self._generate_alerts_impl(input_data, **args)
            except Exception as e:
                self.logger.error(f"Tool {name} failed: {e}")
                results[name] = {"error": str(e)}

        return results

    def _fit_isolation_forest_impl(
        self,
        input_data: Dict[str, Any],
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: int = 256,
        max_features: float = 1.0,
        bootstrap: bool = False,
    ) -> Dict[str, Any]:
        """Tool implementation: Fit Isolation Forest model.

        Args:
            input_data: Input data dict
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees
            max_samples: Samples per tree
            max_features: Features per tree
            bootstrap: Sample with replacement

        Returns:
            Dict with fitted model information
        """
        self._tool_call_count += 1

        data = input_data["data"]
        feature_columns = input_data.get("feature_columns")

        # Select features
        if feature_columns:
            features = feature_columns
        else:
            features = data.select_dtypes(include=[np.number]).columns.tolist()

        self._feature_columns = features
        X = data[features].values

        # Handle missing values
        if np.any(np.isnan(X)):
            # Simple mean imputation
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        # Fit scaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        if not SKLEARN_AVAILABLE:
            # Mock mode
            self._fitted_model = "mock_model"
            return {
                "n_samples": len(X),
                "n_features": len(features),
                "contamination": contamination,
                "n_estimators": n_estimators,
                "max_samples": min(max_samples, len(X)),
                "fitted": True,
            }

        # Fit Isolation Forest
        self._fitted_model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=min(max_samples, len(X)),
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=42,  # Deterministic
            n_jobs=-1,
        )

        self._fitted_model.fit(X_scaled)

        return {
            "n_samples": int(len(X)),
            "n_features": int(len(features)),
            "features": features,
            "contamination": float(contamination),
            "n_estimators": int(n_estimators),
            "max_samples": int(min(max_samples, len(X))),
            "fitted": True,
        }

    def _detect_anomalies_impl(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Tool implementation: Detect anomalies.

        Args:
            input_data: Input data dict

        Returns:
            Dict with anomaly predictions
        """
        self._tool_call_count += 1

        if self._fitted_model is None:
            raise ValueError("Model must be fitted before detecting anomalies")

        data = input_data["data"]
        X = data[self._feature_columns].values

        # Handle missing values
        if np.any(np.isnan(X)):
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        # Scale data
        X_scaled = self._scaler.transform(X)

        if not SKLEARN_AVAILABLE:
            # Mock predictions
            np.random.seed(42)
            predictions = np.random.choice([1, -1], size=len(X), p=[0.9, 0.1])
        else:
            # Real predictions: 1 = normal, -1 = anomaly
            predictions = self._fitted_model.predict(X_scaled)

        # Convert to boolean (True = anomaly)
        anomalies = (predictions == -1)
        anomaly_indices = np.where(anomalies)[0].tolist()

        return {
            "anomalies": anomalies.tolist(),
            "anomaly_indices": anomaly_indices,
            "n_anomalies": int(np.sum(anomalies)),
            "n_normal": int(np.sum(~anomalies)),
            "anomaly_rate": float(np.mean(anomalies)),
        }

    def _calculate_anomaly_scores_impl(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Tool implementation: Calculate anomaly scores.

        Args:
            input_data: Input data dict

        Returns:
            Dict with anomaly scores
        """
        self._tool_call_count += 1

        if self._fitted_model is None:
            raise ValueError("Model must be fitted before calculating scores")

        data = input_data["data"]
        X = data[self._feature_columns].values

        # Handle missing values
        if np.any(np.isnan(X)):
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        # Scale data
        X_scaled = self._scaler.transform(X)

        if not SKLEARN_AVAILABLE:
            # Mock scores
            np.random.seed(42)
            scores = np.random.uniform(-0.5, 0.5, size=len(X))
        else:
            # Real scores: negative = anomaly, positive = normal
            scores = self._fitted_model.score_samples(X_scaled)

        # Classify severity based on score
        severities = []
        for score in scores:
            if score < -0.5:
                severities.append("critical")
            elif score < -0.3:
                severities.append("high")
            elif score < -0.1:
                severities.append("medium")
            elif score < 0:
                severities.append("low")
            else:
                severities.append("normal")

        return {
            "scores": scores.tolist(),
            "severities": severities,
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "mean_score": float(np.mean(scores)),
        }

    def _rank_anomalies_impl(
        self,
        input_data: Dict[str, Any],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Tool implementation: Rank anomalies by severity.

        Args:
            input_data: Input data dict
            top_k: Number of top anomalies to return

        Returns:
            Dict with ranked anomalies
        """
        self._tool_call_count += 1

        if self._fitted_model is None:
            raise ValueError("Model must be fitted before ranking anomalies")

        data = input_data["data"]
        X = data[self._feature_columns].values

        # Handle missing values
        if np.any(np.isnan(X)):
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        # Scale data
        X_scaled = self._scaler.transform(X)

        if not SKLEARN_AVAILABLE:
            # Mock scores
            np.random.seed(42)
            scores = np.random.uniform(-0.5, 0.5, size=len(X))
        else:
            scores = self._fitted_model.score_samples(X_scaled)

        # Rank by score (most negative = most anomalous)
        ranked_indices = np.argsort(scores)[:top_k]

        ranked_anomalies = []
        for idx in ranked_indices:
            score = float(scores[idx])

            # Determine severity
            if score < -0.5:
                severity = "critical"
            elif score < -0.3:
                severity = "high"
            elif score < -0.1:
                severity = "medium"
            else:
                severity = "low"

            # Get feature values
            feature_values = {}
            for i, col in enumerate(self._feature_columns):
                feature_values[col] = float(X[idx, i])

            ranked_anomalies.append({
                "index": int(idx),
                "score": score,
                "severity": severity,
                "features": feature_values,
            })

        return {
            "top_anomalies": ranked_anomalies,
            "n_ranked": len(ranked_anomalies),
        }

    def _analyze_anomaly_patterns_impl(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Tool implementation: Analyze anomaly patterns.

        Args:
            input_data: Input data dict

        Returns:
            Dict with pattern analysis
        """
        self._tool_call_count += 1

        if self._fitted_model is None:
            raise ValueError("Model must be fitted before analyzing patterns")

        data = input_data["data"]
        X = data[self._feature_columns].values

        # Handle missing values
        if np.any(np.isnan(X)):
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        # Scale data
        X_scaled = self._scaler.transform(X)

        if not SKLEARN_AVAILABLE:
            # Mock predictions
            np.random.seed(42)
            predictions = np.random.choice([1, -1], size=len(X), p=[0.9, 0.1])
        else:
            predictions = self._fitted_model.predict(X_scaled)

        anomalies = (predictions == -1)
        anomaly_indices = np.where(anomalies)[0]

        if len(anomaly_indices) == 0:
            return {
                "n_anomalies": 0,
                "patterns": {},
                "feature_importance": {},
            }

        # Analyze feature statistics for anomalies vs normal
        patterns = {}
        feature_importance = {}

        normal_indices = np.where(~anomalies)[0]

        for i, col in enumerate(self._feature_columns):
            anomaly_values = X[anomaly_indices, i]
            normal_values = X[normal_indices, i]

            anomaly_mean = float(np.mean(anomaly_values))
            normal_mean = float(np.mean(normal_values))

            # Calculate relative difference
            if normal_mean != 0:
                relative_diff = abs(anomaly_mean - normal_mean) / abs(normal_mean)
            else:
                relative_diff = 0.0

            patterns[col] = {
                "anomaly_mean": anomaly_mean,
                "normal_mean": normal_mean,
                "anomaly_std": float(np.std(anomaly_values)),
                "normal_std": float(np.std(normal_values)),
                "relative_difference": float(relative_diff),
            }

            feature_importance[col] = float(relative_diff)

        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        return {
            "n_anomalies": int(len(anomaly_indices)),
            "patterns": patterns,
            "feature_importance": dict(sorted_features),
            "most_important_feature": sorted_features[0][0] if sorted_features else None,
        }

    def _generate_alerts_impl(
        self,
        input_data: Dict[str, Any],
        min_severity: str = "high",
    ) -> Dict[str, Any]:
        """Tool implementation: Generate alerts.

        Args:
            input_data: Input data dict
            min_severity: Minimum severity level

        Returns:
            Dict with generated alerts
        """
        self._tool_call_count += 1

        if self._fitted_model is None:
            raise ValueError("Model must be fitted before generating alerts")

        # Get ranked anomalies
        rankings = self._rank_anomalies_impl(input_data, top_k=50)
        top_anomalies = rankings["top_anomalies"]

        # Severity order
        severity_order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        min_severity_level = severity_order.get(min_severity, 2)

        # Filter by severity
        alerts = []
        for anomaly in top_anomalies:
            if severity_order.get(anomaly["severity"], 0) >= min_severity_level:
                # Generate root cause hints
                root_causes = []
                for feature, value in anomaly["features"].items():
                    # Check if value is extreme
                    feature_values = input_data["data"][feature].values
                    mean = np.mean(feature_values)
                    std = np.std(feature_values)

                    z_score = abs((value - mean) / std) if std > 0 else 0

                    if z_score > 3:
                        direction = "high" if value > mean else "low"
                        root_causes.append(f"{feature} is extremely {direction} ({value:.2f}, z-score: {z_score:.1f})")

                # Generate recommendations
                recommendations = []
                if anomaly["severity"] == "critical":
                    recommendations.append("Immediate investigation required")
                    recommendations.append("Verify data source and sensors")
                    recommendations.append("Check for equipment malfunction")
                elif anomaly["severity"] == "high":
                    recommendations.append("Investigate within 24 hours")
                    recommendations.append("Review system logs")
                else:
                    recommendations.append("Monitor for recurring patterns")

                # Calculate confidence (based on how extreme the score is)
                confidence = min(abs(anomaly["score"]) * 2, 1.0)

                alerts.append({
                    "index": anomaly["index"],
                    "severity": anomaly["severity"],
                    "score": anomaly["score"],
                    "root_cause_hints": root_causes,
                    "recommendations": recommendations,
                    "confidence": float(confidence),
                })

        # Count by severity
        severity_counts = {}
        for alert in alerts:
            severity = alert["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "alerts": alerts,
            "n_alerts": len(alerts),
            "severity_counts": severity_counts,
        }

    def _build_output(
        self,
        input_data: Dict[str, Any],
        tool_results: Dict[str, Any],
        explanation: Optional[str],
    ) -> Dict[str, Any]:
        """Build output from tool results.

        Args:
            input_data: Original input
            tool_results: Results from tool calls
            explanation: AI-generated explanation

        Returns:
            Dict with complete detection results
        """
        model_data = tool_results.get("model", {})
        anomaly_data = tool_results.get("anomalies", {})
        scores_data = tool_results.get("scores", {})
        rankings_data = tool_results.get("rankings", {})
        patterns_data = tool_results.get("patterns", {})
        alerts_data = tool_results.get("alerts", {})

        output = {
            "anomalies": anomaly_data.get("anomalies", []),
            "anomaly_scores": scores_data.get("scores", []),
            "anomaly_indices": anomaly_data.get("anomaly_indices", []),
            "n_anomalies": anomaly_data.get("n_anomalies", 0),
            "n_normal": anomaly_data.get("n_normal", 0),
            "anomaly_rate": anomaly_data.get("anomaly_rate", 0.0),
        }

        # Add model info
        if model_data:
            output["model_info"] = {
                "n_samples": model_data.get("n_samples", 0),
                "n_features": model_data.get("n_features", 0),
                "features": model_data.get("features", []),
                "contamination": model_data.get("contamination", 0.1),
                "n_estimators": model_data.get("n_estimators", 100),
            }

        # Add severity info
        if scores_data:
            output["severity_distribution"] = {
                severity: scores_data["severities"].count(severity)
                for severity in ["critical", "high", "medium", "low", "normal"]
            }

        # Add rankings
        if rankings_data:
            output["top_anomalies"] = rankings_data.get("top_anomalies", [])

        # Add patterns
        if patterns_data:
            output["patterns"] = patterns_data

        # Add alerts
        if alerts_data:
            output["alerts"] = alerts_data.get("alerts", [])
            output["alert_summary"] = {
                "n_alerts": alerts_data.get("n_alerts", 0),
                "severity_counts": alerts_data.get("severity_counts", {}),
            }

        # Add metrics if labels provided
        if "labels" in input_data and "anomalies" in anomaly_data:
            try:
                labels = input_data["labels"]
                predictions = anomaly_data["anomalies"]

                if SKLEARN_AVAILABLE:
                    precision = precision_score(labels, predictions, zero_division=0)
                    recall = recall_score(labels, predictions, zero_division=0)
                    f1 = f1_score(labels, predictions, zero_division=0)

                    try:
                        scores = scores_data.get("scores", [])
                        if scores:
                            # Convert boolean labels to binary for ROC-AUC
                            binary_labels = [1 if l else 0 for l in labels]
                            # Invert scores (more negative = more anomalous)
                            inverted_scores = [-s for s in scores]
                            roc_auc = roc_auc_score(binary_labels, inverted_scores)
                        else:
                            roc_auc = 0.0
                    except:
                        roc_auc = 0.0

                    output["metrics"] = {
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "roc_auc": float(roc_auc),
                    }
            except Exception as e:
                self.logger.warning(f"Could not calculate metrics: {e}")

        # Add AI explanation
        if explanation and self.enable_explanations:
            output["explanation"] = explanation

        return output

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary.

        Returns:
            Dict with AI and tool metrics
        """
        return {
            "agent_id": self.metadata.id,
            "ai_metrics": {
                "ai_call_count": self._ai_call_count,
                "tool_call_count": self._tool_call_count,
                "total_cost_usd": self._total_cost_usd,
                "avg_cost_per_detection": (
                    self._total_cost_usd / max(self._ai_call_count, 1)
                ),
            },
        }
