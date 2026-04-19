"""
ThermalIQ LIME Explainer

Provides LIME-based local interpretable explanations for thermal
system predictions and recommendations.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class InterpretabilityMode(Enum):
    """Modes for LIME interpretability."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


@dataclass
class FeatureWeight:
    """Represents a feature weight in LIME explanation."""
    feature_name: str
    feature_value: float
    weight: float
    contribution: float
    condition: str  # e.g., "temperature > 150"
    unit: str = ""

    @property
    def is_positive(self) -> bool:
        """Check if feature has positive contribution."""
        return self.weight > 0

    @property
    def magnitude(self) -> str:
        """Get the magnitude of the weight."""
        abs_weight = abs(self.weight)
        if abs_weight > 0.3:
            return "strong"
        elif abs_weight > 0.1:
            return "moderate"
        return "weak"


@dataclass
class LocalModel:
    """Represents the local interpretable model from LIME."""
    coefficients: Dict[str, float]
    intercept: float
    score: float  # R-squared for regression
    feature_names: List[str]

    def predict(self, features: Dict[str, float]) -> float:
        """Predict using local linear model."""
        result = self.intercept
        for name, value in features.items():
            if name in self.coefficients:
                result += self.coefficients[name] * value
        return result


@dataclass
class LIMEExplanation:
    """Container for LIME explanation results."""
    instance_values: Dict[str, float]
    predicted_value: float
    feature_weights: List[FeatureWeight]
    local_model: LocalModel
    num_features: int
    kernel_width: float
    mode: InterpretabilityMode
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_positive_features(self) -> List[FeatureWeight]:
        """Get features with positive weights."""
        positive = [fw for fw in self.feature_weights if fw.weight > 0]
        return sorted(positive, key=lambda x: x.weight, reverse=True)

    @property
    def top_negative_features(self) -> List[FeatureWeight]:
        """Get features with negative weights."""
        negative = [fw for fw in self.feature_weights if fw.weight < 0]
        return sorted(negative, key=lambda x: x.weight)

    @property
    def feature_ranking(self) -> List[Tuple[str, float]]:
        """Get features ranked by absolute weight."""
        return sorted(
            [(fw.feature_name, abs(fw.weight)) for fw in self.feature_weights],
            key=lambda x: x[1],
            reverse=True
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "instance_values": self.instance_values,
            "predicted_value": float(self.predicted_value),
            "feature_weights": [
                {
                    "feature_name": fw.feature_name,
                    "feature_value": float(fw.feature_value),
                    "weight": float(fw.weight),
                    "contribution": float(fw.contribution),
                    "condition": fw.condition,
                    "unit": fw.unit
                }
                for fw in self.feature_weights
            ],
            "local_model": {
                "coefficients": {k: float(v) for k, v in self.local_model.coefficients.items()},
                "intercept": float(self.local_model.intercept),
                "score": float(self.local_model.score),
                "feature_names": self.local_model.feature_names
            },
            "mode": self.mode.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def get_explanation_text(self) -> str:
        """Generate human-readable explanation text."""
        lines = [
            f"Prediction: {self.predicted_value:.4f}",
            f"Local model R-squared: {self.local_model.score:.3f}",
            "",
            "Key factors:"
        ]

        for fw in sorted(self.feature_weights, key=lambda x: abs(x.weight), reverse=True)[:5]:
            direction = "increases" if fw.weight > 0 else "decreases"
            lines.append(
                f"  - {fw.condition}: {direction} prediction by {abs(fw.contribution):.4f}"
            )

        return "\n".join(lines)


class ThermalLIMEExplainer:
    """
    LIME-based explainer for thermal system predictions.

    Provides local interpretable model-agnostic explanations for:
    - Individual predictions
    - Recommendation reasoning
    - Operating condition impacts
    """

    # Feature metadata for thermal systems
    THERMAL_FEATURE_METADATA = {
        "inlet_temperature": {"unit": "C", "category": "temperature"},
        "outlet_temperature": {"unit": "C", "category": "temperature"},
        "ambient_temperature": {"unit": "C", "category": "temperature"},
        "stack_temperature": {"unit": "C", "category": "temperature"},
        "inlet_pressure": {"unit": "bar", "category": "pressure"},
        "outlet_pressure": {"unit": "bar", "category": "pressure"},
        "pressure_drop": {"unit": "bar", "category": "pressure"},
        "mass_flow_rate": {"unit": "kg/s", "category": "flow"},
        "volume_flow_rate": {"unit": "m3/h", "category": "flow"},
        "specific_heat": {"unit": "kJ/kg-K", "category": "property"},
        "thermal_conductivity": {"unit": "W/m-K", "category": "property"},
        "viscosity": {"unit": "mPa-s", "category": "property"},
        "density": {"unit": "kg/m3", "category": "property"},
        "heat_duty": {"unit": "kW", "category": "performance"},
        "efficiency": {"unit": "%", "category": "performance"},
        "excess_air_ratio": {"unit": "%", "category": "combustion"},
        "fuel_flow_rate": {"unit": "kg/h", "category": "combustion"},
    }

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        mode: InterpretabilityMode = InterpretabilityMode.REGRESSION,
        categorical_features: Optional[List[int]] = None,
        kernel_width: Optional[float] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the LIME explainer.

        Args:
            training_data: Training data for generating perturbations
            feature_names: Names of input features
            mode: Regression or classification mode
            categorical_features: Indices of categorical features
            kernel_width: Width of exponential kernel (auto if None)
            class_names: Names of classes for classification
        """
        if not LIME_AVAILABLE:
            raise ImportError(
                "LIME library is required. Install with: pip install lime"
            )

        self.training_data = training_data
        self.feature_names = feature_names
        self.mode = mode
        self.categorical_features = categorical_features or []
        self.kernel_width = kernel_width
        self.class_names = class_names

        # Create LIME explainer
        self._explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode.value,
            categorical_features=categorical_features,
            kernel_width=kernel_width,
            verbose=False
        )

    def explain_prediction(
        self,
        model: Any,
        instance: Union[np.ndarray, Dict[str, float]],
        num_features: int = 10,
        num_samples: int = 5000
    ) -> LIMEExplanation:
        """
        Generate LIME explanation for a prediction.

        Args:
            model: Trained model with predict method
            instance: Instance to explain (array or dict)
            num_features: Number of features to include
            num_samples: Number of samples for LIME

        Returns:
            LIMEExplanation with local interpretation
        """
        from datetime import datetime

        # Convert dict to array
        if isinstance(instance, dict):
            instance_array = np.array([instance[k] for k in self.feature_names])
            instance_dict = instance
        else:
            instance_array = np.array(instance)
            instance_dict = {
                name: float(instance_array[i])
                for i, name in enumerate(self.feature_names)
            }

        # Get prediction
        if hasattr(model, 'predict_proba') and self.mode == InterpretabilityMode.CLASSIFICATION:
            predict_fn = model.predict_proba
        elif hasattr(model, 'predict'):
            predict_fn = model.predict
        else:
            predict_fn = model

        prediction = predict_fn(instance_array.reshape(1, -1))
        if hasattr(prediction, '__len__') and len(prediction) > 0:
            if isinstance(prediction[0], np.ndarray):
                predicted_value = float(prediction[0][0])
            else:
                predicted_value = float(prediction[0])
        else:
            predicted_value = float(prediction)

        # Generate LIME explanation
        explanation = self._explainer.explain_instance(
            instance_array,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )

        # Extract feature weights
        feature_weights = []
        exp_list = explanation.as_list()

        for condition, weight in exp_list:
            # Parse condition to get feature name and value
            feature_name, feature_value = self._parse_condition(condition, instance_dict)
            metadata = self.THERMAL_FEATURE_METADATA.get(feature_name, {})

            feature_weights.append(FeatureWeight(
                feature_name=feature_name,
                feature_value=feature_value,
                weight=weight,
                contribution=weight,  # For LIME, weight is the contribution
                condition=condition,
                unit=metadata.get("unit", "")
            ))

        # Build local model
        local_model = LocalModel(
            coefficients={fw.feature_name: fw.weight for fw in feature_weights},
            intercept=explanation.intercept[0] if hasattr(explanation, 'intercept') else 0.0,
            score=explanation.score if hasattr(explanation, 'score') else 0.0,
            feature_names=[fw.feature_name for fw in feature_weights]
        )

        return LIMEExplanation(
            instance_values=instance_dict,
            predicted_value=predicted_value,
            feature_weights=feature_weights,
            local_model=local_model,
            num_features=num_features,
            kernel_width=self.kernel_width or 0.75 * np.sqrt(len(self.feature_names)),
            mode=self.mode,
            timestamp=datetime.now().isoformat(),
            metadata={
                "num_samples": num_samples,
                "model_type": type(model).__name__
            }
        )

    def explain_recommendation(
        self,
        recommendation: Dict[str, Any],
        model: Any,
        baseline: Dict[str, float],
        num_features: int = 10
    ) -> LIMEExplanation:
        """
        Explain a recommendation by comparing to baseline.

        Args:
            recommendation: Recommended operating conditions
            model: Model used for recommendation
            baseline: Baseline operating conditions
            num_features: Number of features to include

        Returns:
            LIMEExplanation for the recommendation
        """
        from datetime import datetime

        # Extract recommended values
        recommended_values = {}
        for key, value in recommendation.items():
            if isinstance(value, (int, float)):
                recommended_values[key] = value
            elif isinstance(value, dict) and 'value' in value:
                recommended_values[key] = value['value']

        # Fill missing values from baseline
        for key in self.feature_names:
            if key not in recommended_values:
                recommended_values[key] = baseline.get(key, 0.0)

        # Get LIME explanation for recommendation
        explanation = self.explain_prediction(
            model=model,
            instance=recommended_values,
            num_features=num_features
        )

        # Add recommendation context to metadata
        explanation.metadata.update({
            "is_recommendation": True,
            "baseline": baseline,
            "recommended_changes": {
                k: {"from": baseline.get(k), "to": v}
                for k, v in recommended_values.items()
                if k in baseline and abs(v - baseline.get(k, v)) > 1e-6
            }
        })

        return explanation

    def _parse_condition(
        self,
        condition: str,
        instance_dict: Dict[str, float]
    ) -> Tuple[str, float]:
        """Parse LIME condition string to extract feature name and value."""
        # LIME conditions look like "temperature > 150.00" or "100.00 < pressure <= 200.00"
        for feature_name in self.feature_names:
            if feature_name in condition:
                return feature_name, instance_dict.get(feature_name, 0.0)

        # Fallback: try to match partial names
        for feature_name in self.feature_names:
            # Check if any word from feature name is in condition
            for word in feature_name.split('_'):
                if word.lower() in condition.lower():
                    return feature_name, instance_dict.get(feature_name, 0.0)

        # Could not parse - return condition as name
        return condition[:20], 0.0

    def local_interpretable_model(
        self,
        model: Any,
        instance: Union[np.ndarray, Dict[str, float]],
        num_samples: int = 5000
    ) -> LocalModel:
        """
        Get the local linear model fitted by LIME.

        Args:
            model: Trained model
            instance: Instance to explain
            num_samples: Number of samples for LIME

        Returns:
            LocalModel representing the local linear approximation
        """
        explanation = self.explain_prediction(
            model=model,
            instance=instance,
            num_features=len(self.feature_names),
            num_samples=num_samples
        )

        return explanation.local_model

    def feature_weights(
        self,
        model: Any,
        instance: Union[np.ndarray, Dict[str, float]],
        num_features: int = 10
    ) -> List[FeatureWeight]:
        """
        Get feature weights for an instance.

        Args:
            model: Trained model
            instance: Instance to explain
            num_features: Number of features to include

        Returns:
            List of FeatureWeight objects sorted by importance
        """
        explanation = self.explain_prediction(
            model=model,
            instance=instance,
            num_features=num_features
        )

        return sorted(
            explanation.feature_weights,
            key=lambda x: abs(x.weight),
            reverse=True
        )

    def plot_explanation(
        self,
        explanation: LIMEExplanation,
        max_features: int = 10,
        show_prediction: bool = True
    ) -> "go.Figure":
        """
        Generate interactive plot of LIME explanation.

        Args:
            explanation: LIMEExplanation to visualize
            max_features: Maximum features to show
            show_prediction: Whether to show prediction value

        Returns:
            Plotly figure with LIME explanation
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        # Sort by weight magnitude
        sorted_weights = sorted(
            explanation.feature_weights,
            key=lambda x: abs(x.weight),
            reverse=True
        )[:max_features]

        # Reverse for horizontal bar chart
        sorted_weights = sorted_weights[::-1]

        conditions = [fw.condition for fw in sorted_weights]
        weights = [fw.weight for fw in sorted_weights]
        colors = ['#e74c3c' if w < 0 else '#27ae60' for w in weights]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=weights,
            y=conditions,
            orientation='h',
            marker_color=colors,
            text=[f"{w:.4f}" for w in weights],
            textposition='outside',
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Weight: %{x:.4f}<br>" +
                "<extra></extra>"
            )
        ))

        title = f"LIME Explanation (Prediction: {explanation.predicted_value:.4f})" if show_prediction else "LIME Explanation"

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Feature Weight",
            yaxis_title="Condition",
            template="plotly_white",
            height=max(400, len(conditions) * 35),
            margin=dict(l=300, r=50, t=60, b=50),
            showlegend=False
        )

        # Add zero line
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

        return fig

    def plot_local_model_fidelity(
        self,
        model: Any,
        instance: Union[np.ndarray, Dict[str, float]],
        n_perturbations: int = 100
    ) -> "go.Figure":
        """
        Plot how well the local model approximates the actual model.

        Args:
            model: Original model
            instance: Instance to analyze
            n_perturbations: Number of perturbations to test

        Returns:
            Plotly figure showing fidelity
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        # Convert dict to array
        if isinstance(instance, dict):
            instance_array = np.array([instance[k] for k in self.feature_names])
        else:
            instance_array = np.array(instance)

        # Get local model
        explanation = self.explain_prediction(
            model=model,
            instance=instance_array,
            num_features=len(self.feature_names)
        )

        # Generate perturbations
        perturbations = []
        actual_predictions = []
        local_predictions = []

        for _ in range(n_perturbations):
            # Create perturbation
            noise = np.random.normal(0, 0.1, len(instance_array))
            perturbed = instance_array * (1 + noise)
            perturbations.append(perturbed)

            # Get actual prediction
            if hasattr(model, 'predict'):
                actual = model.predict(perturbed.reshape(1, -1))[0]
            else:
                actual = model(perturbed.reshape(1, -1))[0]
            actual_predictions.append(float(actual))

            # Get local model prediction
            perturbed_dict = {
                name: float(perturbed[i])
                for i, name in enumerate(self.feature_names)
            }
            local_pred = explanation.local_model.predict(perturbed_dict)
            local_predictions.append(local_pred)

        # Create plot
        fig = go.Figure()

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=actual_predictions,
            y=local_predictions,
            mode='markers',
            marker=dict(color='#3498db', size=8, opacity=0.6),
            name='Perturbations',
            hovertemplate=(
                "Actual: %{x:.4f}<br>" +
                "Local: %{y:.4f}<br>" +
                "<extra></extra>"
            )
        ))

        # Perfect fit line
        min_val = min(min(actual_predictions), min(local_predictions))
        max_val = max(max(actual_predictions), max(local_predictions))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Fit'
        ))

        # Mark the instance
        instance_actual = actual_predictions[0] if actual_predictions else 0
        instance_local = local_predictions[0] if local_predictions else 0

        fig.add_trace(go.Scatter(
            x=[explanation.predicted_value],
            y=[explanation.local_model.intercept + sum(
                explanation.local_model.coefficients.get(name, 0) * instance_array[i]
                for i, name in enumerate(self.feature_names)
            )],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name='Instance'
        ))

        # Calculate R-squared
        actual_arr = np.array(actual_predictions)
        local_arr = np.array(local_predictions)
        ss_res = np.sum((actual_arr - local_arr) ** 2)
        ss_tot = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))

        fig.update_layout(
            title=dict(
                text=f"Local Model Fidelity (R-squared: {r_squared:.3f})",
                font=dict(size=16)
            ),
            xaxis_title="Actual Model Prediction",
            yaxis_title="Local Model Prediction",
            template="plotly_white",
            height=500,
            width=600,
            showlegend=True
        )

        return fig

    def compare_instances(
        self,
        model: Any,
        instance1: Dict[str, float],
        instance2: Dict[str, float],
        num_features: int = 10
    ) -> "go.Figure":
        """
        Compare LIME explanations for two instances.

        Args:
            model: Trained model
            instance1: First instance
            instance2: Second instance
            num_features: Number of features to compare

        Returns:
            Plotly figure comparing explanations
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        # Get explanations
        exp1 = self.explain_prediction(model, instance1, num_features)
        exp2 = self.explain_prediction(model, instance2, num_features)

        # Get all feature names
        all_features = set(
            [fw.feature_name for fw in exp1.feature_weights] +
            [fw.feature_name for fw in exp2.feature_weights]
        )

        # Build weight dictionaries
        weights1 = {fw.feature_name: fw.weight for fw in exp1.feature_weights}
        weights2 = {fw.feature_name: fw.weight for fw in exp2.feature_weights}

        # Sort features by max absolute weight
        sorted_features = sorted(
            all_features,
            key=lambda f: max(abs(weights1.get(f, 0)), abs(weights2.get(f, 0))),
            reverse=True
        )[:num_features]

        sorted_features = sorted_features[::-1]  # Reverse for display

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=sorted_features,
            x=[weights1.get(f, 0) for f in sorted_features],
            name=f'Instance 1 (pred: {exp1.predicted_value:.3f})',
            orientation='h',
            marker_color='#3498db'
        ))

        fig.add_trace(go.Bar(
            y=sorted_features,
            x=[weights2.get(f, 0) for f in sorted_features],
            name=f'Instance 2 (pred: {exp2.predicted_value:.3f})',
            orientation='h',
            marker_color='#e74c3c'
        ))

        fig.update_layout(
            title=dict(text="LIME Explanation Comparison", font=dict(size=16)),
            xaxis_title="Feature Weight",
            yaxis_title="Feature",
            barmode='group',
            template="plotly_white",
            height=max(400, len(sorted_features) * 40),
            margin=dict(l=200, r=50, t=60, b=50),
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
        )

        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

        return fig


# Convenience function
def explain_thermal_instance(
    model: Any,
    instance: Dict[str, float],
    training_data: np.ndarray,
    feature_names: List[str],
    num_features: int = 10
) -> LIMEExplanation:
    """
    Quick function to generate LIME explanation for thermal prediction.

    Args:
        model: Trained model
        instance: Instance to explain
        training_data: Training dataset
        feature_names: Feature names
        num_features: Number of features to include

    Returns:
        LIMEExplanation object
    """
    explainer = ThermalLIMEExplainer(
        training_data=training_data,
        feature_names=feature_names,
        mode=InterpretabilityMode.REGRESSION
    )

    return explainer.explain_prediction(
        model=model,
        instance=instance,
        num_features=num_features
    )
