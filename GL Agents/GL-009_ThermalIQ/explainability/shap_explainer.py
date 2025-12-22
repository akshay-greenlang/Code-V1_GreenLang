"""
ThermalIQ SHAP Explainer

Provides SHAP-based explanations for thermal system predictions
including efficiency calculations, exergy analysis, and fluid selection.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ExplanationType(Enum):
    """Types of SHAP explanations for thermal systems."""
    EFFICIENCY = "efficiency"
    EXERGY = "exergy"
    FLUID_SELECTION = "fluid_selection"
    HEAT_TRANSFER = "heat_transfer"
    PRESSURE_DROP = "pressure_drop"


@dataclass
class FeatureContribution:
    """Represents a feature's contribution to a prediction."""
    feature_name: str
    feature_value: float
    shap_value: float
    unit: str = ""
    description: str = ""

    @property
    def contribution_direction(self) -> str:
        """Get the direction of contribution."""
        if self.shap_value > 0:
            return "positive"
        elif self.shap_value < 0:
            return "negative"
        return "neutral"

    @property
    def contribution_magnitude(self) -> str:
        """Get the magnitude of contribution."""
        abs_val = abs(self.shap_value)
        if abs_val > 0.5:
            return "high"
        elif abs_val > 0.1:
            return "medium"
        return "low"


@dataclass
class SHAPExplanation:
    """Container for SHAP explanation results."""
    explanation_type: ExplanationType
    base_value: float
    predicted_value: float
    feature_contributions: List[FeatureContribution]
    feature_names: List[str]
    feature_values: np.ndarray
    shap_values: np.ndarray
    model_type: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_positive_features(self) -> List[FeatureContribution]:
        """Get top features with positive contributions."""
        positive = [f for f in self.feature_contributions if f.shap_value > 0]
        return sorted(positive, key=lambda x: x.shap_value, reverse=True)[:5]

    @property
    def top_negative_features(self) -> List[FeatureContribution]:
        """Get top features with negative contributions."""
        negative = [f for f in self.feature_contributions if f.shap_value < 0]
        return sorted(negative, key=lambda x: x.shap_value)[:5]

    @property
    def feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """Get features ranked by absolute SHAP value."""
        return sorted(
            [(f.feature_name, abs(f.shap_value)) for f in self.feature_contributions],
            key=lambda x: x[1],
            reverse=True
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "explanation_type": self.explanation_type.value,
            "base_value": float(self.base_value),
            "predicted_value": float(self.predicted_value),
            "feature_contributions": [
                {
                    "feature_name": fc.feature_name,
                    "feature_value": float(fc.feature_value),
                    "shap_value": float(fc.shap_value),
                    "unit": fc.unit,
                    "description": fc.description
                }
                for fc in self.feature_contributions
            ],
            "model_type": self.model_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class ThermalSHAPExplainer:
    """
    SHAP-based explainer for thermal system models.

    Provides interpretable explanations for:
    - Efficiency predictions
    - Exergy analysis results
    - Fluid selection recommendations
    - Heat transfer calculations
    """

    # Feature metadata for thermal systems
    THERMAL_FEATURE_METADATA = {
        # Temperature features
        "inlet_temperature": {"unit": "C", "description": "Fluid inlet temperature"},
        "outlet_temperature": {"unit": "C", "description": "Fluid outlet temperature"},
        "ambient_temperature": {"unit": "C", "description": "Ambient temperature"},
        "stack_temperature": {"unit": "C", "description": "Stack/exhaust temperature"},
        "surface_temperature": {"unit": "C", "description": "Equipment surface temperature"},

        # Pressure features
        "inlet_pressure": {"unit": "bar", "description": "Inlet pressure"},
        "outlet_pressure": {"unit": "bar", "description": "Outlet pressure"},
        "pressure_drop": {"unit": "bar", "description": "Pressure drop across equipment"},

        # Flow features
        "mass_flow_rate": {"unit": "kg/s", "description": "Mass flow rate"},
        "volume_flow_rate": {"unit": "m3/h", "description": "Volumetric flow rate"},
        "velocity": {"unit": "m/s", "description": "Fluid velocity"},

        # Thermal properties
        "specific_heat": {"unit": "kJ/kg-K", "description": "Specific heat capacity"},
        "thermal_conductivity": {"unit": "W/m-K", "description": "Thermal conductivity"},
        "viscosity": {"unit": "mPa-s", "description": "Dynamic viscosity"},
        "density": {"unit": "kg/m3", "description": "Fluid density"},

        # Equipment parameters
        "heat_transfer_area": {"unit": "m2", "description": "Heat transfer surface area"},
        "heat_transfer_coefficient": {"unit": "W/m2-K", "description": "Overall heat transfer coefficient"},
        "insulation_thickness": {"unit": "mm", "description": "Insulation thickness"},
        "fouling_factor": {"unit": "m2-K/W", "description": "Fouling resistance"},

        # Performance metrics
        "heat_duty": {"unit": "kW", "description": "Heat transfer rate"},
        "efficiency": {"unit": "%", "description": "Thermal efficiency"},
        "effectiveness": {"unit": "-", "description": "Heat exchanger effectiveness"},

        # Combustion parameters
        "excess_air_ratio": {"unit": "%", "description": "Excess air ratio"},
        "fuel_flow_rate": {"unit": "kg/h", "description": "Fuel flow rate"},
        "air_flow_rate": {"unit": "kg/h", "description": "Air flow rate"},
        "O2_percentage": {"unit": "%", "description": "Oxygen in flue gas"},
        "CO2_percentage": {"unit": "%", "description": "CO2 in flue gas"},
    }

    def __init__(
        self,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        n_background_samples: int = 100
    ):
        """
        Initialize the SHAP explainer.

        Args:
            background_data: Background dataset for SHAP calculations
            feature_names: Names of input features
            n_background_samples: Number of background samples to use
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP library is required. Install with: pip install shap"
            )

        self.background_data = background_data
        self.feature_names = feature_names or []
        self.n_background_samples = n_background_samples
        self._explainer: Optional[shap.Explainer] = None
        self._model = None

    def _create_explainer(self, model: Any) -> shap.Explainer:
        """Create appropriate SHAP explainer for the model type."""
        model_type = type(model).__name__

        if self.background_data is None:
            raise ValueError("Background data is required for SHAP calculations")

        # Sample background data if too large
        if len(self.background_data) > self.n_background_samples:
            indices = np.random.choice(
                len(self.background_data),
                self.n_background_samples,
                replace=False
            )
            background = self.background_data[indices]
        else:
            background = self.background_data

        # Choose explainer based on model type
        if hasattr(model, 'predict_proba'):
            # Tree-based or sklearn models
            try:
                return shap.TreeExplainer(model, background)
            except Exception:
                return shap.KernelExplainer(model.predict, background)
        elif hasattr(model, 'predict'):
            return shap.KernelExplainer(model.predict, background)
        else:
            # Assume callable function
            return shap.KernelExplainer(model, background)

    def _build_feature_contributions(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureContribution]:
        """Build list of feature contributions from SHAP values."""
        contributions = []

        for i, name in enumerate(feature_names):
            metadata = self.THERMAL_FEATURE_METADATA.get(name, {})

            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=float(feature_values[i]) if i < len(feature_values) else 0.0,
                shap_value=float(shap_values[i]) if i < len(shap_values) else 0.0,
                unit=metadata.get("unit", ""),
                description=metadata.get("description", "")
            ))

        return contributions

    def explain_efficiency(
        self,
        model: Any,
        inputs: Union[np.ndarray, Dict[str, float]],
        feature_names: Optional[List[str]] = None
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for efficiency prediction.

        Args:
            model: Trained efficiency prediction model
            inputs: Input features as array or dict
            feature_names: Optional feature names

        Returns:
            SHAPExplanation with efficiency prediction breakdown
        """
        from datetime import datetime

        # Convert dict to array if needed
        if isinstance(inputs, dict):
            feature_names = feature_names or list(inputs.keys())
            input_array = np.array([inputs[k] for k in feature_names]).reshape(1, -1)
        else:
            input_array = np.atleast_2d(inputs)
            feature_names = feature_names or self.feature_names

        # Create or reuse explainer
        if self._explainer is None or self._model != model:
            self._explainer = self._create_explainer(model)
            self._model = model

        # Calculate SHAP values
        shap_values = self._explainer.shap_values(input_array)

        # Handle multi-output models
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.atleast_2d(shap_values)

        # Get base value
        if hasattr(self._explainer, 'expected_value'):
            base_value = self._explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
        else:
            base_value = 0.0

        # Get predicted value
        if hasattr(model, 'predict'):
            predicted_value = float(model.predict(input_array)[0])
        else:
            predicted_value = float(model(input_array)[0])

        # Build contributions
        contributions = self._build_feature_contributions(
            shap_values[0],
            input_array[0],
            feature_names
        )

        return SHAPExplanation(
            explanation_type=ExplanationType.EFFICIENCY,
            base_value=base_value,
            predicted_value=predicted_value,
            feature_contributions=contributions,
            feature_names=feature_names,
            feature_values=input_array[0],
            shap_values=shap_values[0],
            model_type=type(model).__name__,
            timestamp=datetime.now().isoformat(),
            metadata={
                "prediction_unit": "%",
                "target": "thermal_efficiency"
            }
        )

    def explain_exergy(
        self,
        model: Any,
        inputs: Union[np.ndarray, Dict[str, float]],
        feature_names: Optional[List[str]] = None,
        reference_temperature: float = 298.15
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for exergy analysis.

        Args:
            model: Trained exergy prediction model
            inputs: Input features
            feature_names: Optional feature names
            reference_temperature: Dead state temperature (K)

        Returns:
            SHAPExplanation with exergy prediction breakdown
        """
        from datetime import datetime

        # Convert dict to array if needed
        if isinstance(inputs, dict):
            feature_names = feature_names or list(inputs.keys())
            input_array = np.array([inputs[k] for k in feature_names]).reshape(1, -1)
        else:
            input_array = np.atleast_2d(inputs)
            feature_names = feature_names or self.feature_names

        # Create or reuse explainer
        if self._explainer is None or self._model != model:
            self._explainer = self._create_explainer(model)
            self._model = model

        # Calculate SHAP values
        shap_values = self._explainer.shap_values(input_array)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.atleast_2d(shap_values)

        # Get base and predicted values
        base_value = getattr(self._explainer, 'expected_value', 0.0)
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[0])

        if hasattr(model, 'predict'):
            predicted_value = float(model.predict(input_array)[0])
        else:
            predicted_value = float(model(input_array)[0])

        # Build contributions
        contributions = self._build_feature_contributions(
            shap_values[0],
            input_array[0],
            feature_names
        )

        return SHAPExplanation(
            explanation_type=ExplanationType.EXERGY,
            base_value=base_value,
            predicted_value=predicted_value,
            feature_contributions=contributions,
            feature_names=feature_names,
            feature_values=input_array[0],
            shap_values=shap_values[0],
            model_type=type(model).__name__,
            timestamp=datetime.now().isoformat(),
            metadata={
                "prediction_unit": "kW",
                "target": "exergy_destruction",
                "reference_temperature_K": reference_temperature
            }
        )

    def explain_fluid_selection(
        self,
        model: Any,
        inputs: Union[np.ndarray, Dict[str, float]],
        fluid_options: List[str],
        feature_names: Optional[List[str]] = None
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for fluid selection recommendation.

        Args:
            model: Trained fluid selection model
            inputs: Input features (operating conditions)
            fluid_options: List of fluid candidates
            feature_names: Optional feature names

        Returns:
            SHAPExplanation with fluid selection breakdown
        """
        from datetime import datetime

        # Convert dict to array if needed
        if isinstance(inputs, dict):
            feature_names = feature_names or list(inputs.keys())
            input_array = np.array([inputs[k] for k in feature_names]).reshape(1, -1)
        else:
            input_array = np.atleast_2d(inputs)
            feature_names = feature_names or self.feature_names

        # Create or reuse explainer
        if self._explainer is None or self._model != model:
            self._explainer = self._create_explainer(model)
            self._model = model

        # Calculate SHAP values
        shap_values = self._explainer.shap_values(input_array)

        # For classification, get values for recommended class
        if isinstance(shap_values, list):
            # Multi-class: get SHAP values for predicted class
            if hasattr(model, 'predict'):
                predicted_class = int(model.predict(input_array)[0])
            else:
                predicted_class = 0
            shap_values = shap_values[predicted_class]

        shap_values = np.atleast_2d(shap_values)

        # Get base and predicted values
        base_value = getattr(self._explainer, 'expected_value', 0.0)
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[0])
        elif isinstance(base_value, list):
            base_value = float(base_value[0])

        if hasattr(model, 'predict'):
            predicted_value = float(model.predict(input_array)[0])
        else:
            predicted_value = float(model(input_array)[0])

        # Build contributions
        contributions = self._build_feature_contributions(
            shap_values[0],
            input_array[0],
            feature_names
        )

        return SHAPExplanation(
            explanation_type=ExplanationType.FLUID_SELECTION,
            base_value=base_value,
            predicted_value=predicted_value,
            feature_contributions=contributions,
            feature_names=feature_names,
            feature_values=input_array[0],
            shap_values=shap_values[0],
            model_type=type(model).__name__,
            timestamp=datetime.now().isoformat(),
            metadata={
                "fluid_options": fluid_options,
                "target": "fluid_selection_score"
            }
        )

    def feature_importance_plot(
        self,
        explanation: SHAPExplanation,
        max_features: int = 15,
        show_values: bool = True
    ) -> "go.Figure":
        """
        Generate feature importance bar plot.

        Args:
            explanation: SHAPExplanation to visualize
            max_features: Maximum features to show
            show_values: Whether to show SHAP values on bars

        Returns:
            Plotly figure with feature importance plot
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        # Sort by absolute SHAP value
        sorted_contributions = sorted(
            explanation.feature_contributions,
            key=lambda x: abs(x.shap_value),
            reverse=True
        )[:max_features]

        # Reverse for horizontal bar chart (bottom to top)
        sorted_contributions = sorted_contributions[::-1]

        names = [c.feature_name for c in sorted_contributions]
        values = [c.shap_value for c in sorted_contributions]
        colors = ['#e74c3c' if v < 0 else '#27ae60' for v in values]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=values,
            y=names,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.3f}" for v in values] if show_values else None,
            textposition='outside',
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "SHAP Value: %{x:.4f}<br>" +
                "<extra></extra>"
            )
        ))

        fig.update_layout(
            title=dict(
                text=f"Feature Importance - {explanation.explanation_type.value.title()}",
                font=dict(size=16)
            ),
            xaxis_title="SHAP Value (impact on prediction)",
            yaxis_title="Feature",
            template="plotly_white",
            height=max(400, len(names) * 30),
            showlegend=False,
            margin=dict(l=200, r=50, t=60, b=50)
        )

        # Add zero line
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

        return fig

    def dependency_plot(
        self,
        explanation: SHAPExplanation,
        feature: str,
        interaction_feature: Optional[str] = None,
        all_shap_values: Optional[np.ndarray] = None,
        all_feature_values: Optional[np.ndarray] = None
    ) -> "go.Figure":
        """
        Generate SHAP dependency plot showing feature effect.

        Args:
            explanation: SHAPExplanation for reference
            feature: Feature to analyze
            interaction_feature: Optional interaction feature for coloring
            all_shap_values: SHAP values for multiple samples
            all_feature_values: Feature values for multiple samples

        Returns:
            Plotly figure with dependency plot
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        # Get feature index
        try:
            feature_idx = explanation.feature_names.index(feature)
        except ValueError:
            raise ValueError(f"Feature '{feature}' not found in explanation")

        # Use provided data or single point
        if all_shap_values is not None and all_feature_values is not None:
            x_values = all_feature_values[:, feature_idx]
            y_values = all_shap_values[:, feature_idx]
        else:
            x_values = np.array([explanation.feature_values[feature_idx]])
            y_values = np.array([explanation.shap_values[feature_idx]])

        # Get feature metadata
        metadata = self.THERMAL_FEATURE_METADATA.get(feature, {})
        unit = metadata.get("unit", "")
        description = metadata.get("description", feature)

        fig = go.Figure()

        if interaction_feature and all_feature_values is not None:
            # Color by interaction feature
            try:
                interaction_idx = explanation.feature_names.index(interaction_feature)
                color_values = all_feature_values[:, interaction_idx]

                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    marker=dict(
                        color=color_values,
                        colorscale='RdBu',
                        showscale=True,
                        colorbar=dict(title=interaction_feature)
                    ),
                    hovertemplate=(
                        f"<b>{feature}</b>: %{{x:.3f}} {unit}<br>" +
                        "SHAP Value: %{y:.4f}<br>" +
                        f"{interaction_feature}: %{{marker.color:.3f}}<br>" +
                        "<extra></extra>"
                    )
                ))
            except ValueError:
                # Fall back to no coloring
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    marker=dict(color='#3498db', size=8),
                    hovertemplate=(
                        f"<b>{feature}</b>: %{{x:.3f}} {unit}<br>" +
                        "SHAP Value: %{y:.4f}<br>" +
                        "<extra></extra>"
                    )
                ))
        else:
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(color='#3498db', size=8),
                hovertemplate=(
                    f"<b>{feature}</b>: %{{x:.3f}} {unit}<br>" +
                    "SHAP Value: %{y:.4f}<br>" +
                    "<extra></extra>"
                )
            ))

        fig.update_layout(
            title=dict(
                text=f"SHAP Dependency Plot: {feature}",
                font=dict(size=16)
            ),
            xaxis_title=f"{description} ({unit})" if unit else description,
            yaxis_title=f"SHAP Value for {feature}",
            template="plotly_white",
            height=500,
            width=700
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        return fig

    def force_plot(
        self,
        explanation: SHAPExplanation,
        max_features: int = 10
    ) -> "go.Figure":
        """
        Generate SHAP force plot showing prediction breakdown.

        Args:
            explanation: SHAPExplanation to visualize
            max_features: Maximum features to show

        Returns:
            Plotly figure with force plot
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        # Sort features by absolute SHAP value
        sorted_contributions = sorted(
            explanation.feature_contributions,
            key=lambda x: abs(x.shap_value),
            reverse=True
        )[:max_features]

        # Separate positive and negative contributions
        positive = [(c.feature_name, c.shap_value, c.feature_value)
                   for c in sorted_contributions if c.shap_value > 0]
        negative = [(c.feature_name, c.shap_value, c.feature_value)
                   for c in sorted_contributions if c.shap_value < 0]

        fig = go.Figure()

        # Calculate cumulative positions
        base = explanation.base_value
        current_pos = base

        # Add base value marker
        fig.add_trace(go.Scatter(
            x=[base],
            y=[0],
            mode='markers+text',
            marker=dict(size=12, color='gray'),
            text=[f"Base: {base:.2f}"],
            textposition='bottom center',
            showlegend=False
        ))

        # Build waterfall
        y_pos = 0
        annotations = []

        # Positive contributions (push prediction up)
        for name, value, feat_val in positive:
            fig.add_shape(
                type="rect",
                x0=current_pos,
                x1=current_pos + value,
                y0=y_pos - 0.3,
                y1=y_pos + 0.3,
                fillcolor="rgba(39, 174, 96, 0.7)",
                line=dict(width=0)
            )
            annotations.append(dict(
                x=current_pos + value/2,
                y=y_pos,
                text=f"{name}={feat_val:.2f}",
                showarrow=False,
                font=dict(size=10, color='white')
            ))
            current_pos += value
            y_pos += 0.8

        # Negative contributions (push prediction down)
        for name, value, feat_val in negative:
            fig.add_shape(
                type="rect",
                x0=current_pos + value,
                x1=current_pos,
                y0=y_pos - 0.3,
                y1=y_pos + 0.3,
                fillcolor="rgba(231, 76, 60, 0.7)",
                line=dict(width=0)
            )
            annotations.append(dict(
                x=current_pos + value/2,
                y=y_pos,
                text=f"{name}={feat_val:.2f}",
                showarrow=False,
                font=dict(size=10, color='white')
            ))
            current_pos += value
            y_pos += 0.8

        # Add prediction marker
        fig.add_trace(go.Scatter(
            x=[explanation.predicted_value],
            y=[y_pos],
            mode='markers+text',
            marker=dict(size=15, color='#3498db', symbol='diamond'),
            text=[f"Prediction: {explanation.predicted_value:.2f}"],
            textposition='top center',
            showlegend=False
        ))

        fig.update_layout(
            title=dict(
                text=f"Force Plot - {explanation.explanation_type.value.title()} Prediction",
                font=dict(size=16)
            ),
            xaxis_title="Prediction Value",
            yaxis=dict(showticklabels=False, showgrid=False),
            template="plotly_white",
            height=400 + len(sorted_contributions) * 30,
            annotations=annotations,
            showlegend=False
        )

        return fig

    def summary_plot(
        self,
        all_explanations: List[SHAPExplanation],
        plot_type: str = "beeswarm"
    ) -> "go.Figure":
        """
        Generate SHAP summary plot for multiple explanations.

        Args:
            all_explanations: List of SHAPExplanations
            plot_type: "beeswarm" or "bar"

        Returns:
            Plotly figure with summary plot
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        if not all_explanations:
            raise ValueError("At least one explanation is required")

        # Aggregate SHAP values
        feature_names = all_explanations[0].feature_names
        n_features = len(feature_names)
        n_samples = len(all_explanations)

        shap_matrix = np.zeros((n_samples, n_features))
        feature_matrix = np.zeros((n_samples, n_features))

        for i, exp in enumerate(all_explanations):
            shap_matrix[i] = exp.shap_values
            feature_matrix[i] = exp.feature_values

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]

        if plot_type == "bar":
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=mean_abs_shap[sorted_idx],
                y=[feature_names[i] for i in sorted_idx],
                orientation='h',
                marker_color='#3498db'
            ))

            fig.update_layout(
                title="Mean |SHAP Value| (Feature Importance)",
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Feature",
                template="plotly_white",
                height=max(400, n_features * 25)
            )
        else:
            # Beeswarm plot
            fig = go.Figure()

            for idx, feat_idx in enumerate(sorted_idx[:15]):
                shap_vals = shap_matrix[:, feat_idx]
                feat_vals = feature_matrix[:, feat_idx]

                # Normalize feature values for coloring
                feat_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-10)

                # Add jitter to y position
                y_jitter = np.random.normal(0, 0.1, len(shap_vals)) + idx

                fig.add_trace(go.Scatter(
                    x=shap_vals,
                    y=y_jitter,
                    mode='markers',
                    marker=dict(
                        color=feat_norm,
                        colorscale='RdBu',
                        size=5,
                        showscale=(idx == 0)
                    ),
                    name=feature_names[feat_idx],
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{feature_names[feat_idx]}</b><br>" +
                        "SHAP Value: %{x:.4f}<br>" +
                        "<extra></extra>"
                    )
                ))

            fig.update_layout(
                title="SHAP Summary Plot",
                xaxis_title="SHAP Value",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(min(15, len(sorted_idx)))),
                    ticktext=[feature_names[i] for i in sorted_idx[:15]]
                ),
                template="plotly_white",
                height=500
            )

        return fig

    def integrate_thermal_calculation(
        self,
        calculation_result: Dict[str, Any],
        model: Any,
        feature_mapping: Dict[str, str]
    ) -> SHAPExplanation:
        """
        Integrate SHAP explanation with thermal calculation results.

        Args:
            calculation_result: Result from thermal calculator
            model: Trained model for the calculation
            feature_mapping: Mapping from calculation fields to feature names

        Returns:
            SHAPExplanation with thermal context
        """
        # Extract features from calculation result
        inputs = {}
        for calc_field, feature_name in feature_mapping.items():
            if calc_field in calculation_result:
                inputs[feature_name] = calculation_result[calc_field]

        # Determine explanation type based on result content
        if 'efficiency' in calculation_result or 'eta' in calculation_result:
            explanation = self.explain_efficiency(
                model, inputs, list(inputs.keys())
            )
        elif 'exergy' in calculation_result or 'irreversibility' in calculation_result:
            explanation = self.explain_exergy(
                model, inputs, list(inputs.keys())
            )
        else:
            # Default to efficiency
            explanation = self.explain_efficiency(
                model, inputs, list(inputs.keys())
            )

        # Add thermal context to metadata
        explanation.metadata.update({
            "thermal_calculation": calculation_result,
            "feature_mapping": feature_mapping
        })

        return explanation


# Convenience function for quick explanations
def explain_thermal_prediction(
    model: Any,
    inputs: Dict[str, float],
    background_data: np.ndarray,
    explanation_type: str = "efficiency"
) -> SHAPExplanation:
    """
    Quick function to generate SHAP explanation for thermal predictions.

    Args:
        model: Trained model
        inputs: Input features as dictionary
        background_data: Background dataset
        explanation_type: Type of explanation ("efficiency", "exergy", "fluid_selection")

    Returns:
        SHAPExplanation object
    """
    explainer = ThermalSHAPExplainer(
        background_data=background_data,
        feature_names=list(inputs.keys())
    )

    if explanation_type == "efficiency":
        return explainer.explain_efficiency(model, inputs)
    elif explanation_type == "exergy":
        return explainer.explain_exergy(model, inputs)
    elif explanation_type == "fluid_selection":
        return explainer.explain_fluid_selection(model, inputs, [])
    else:
        raise ValueError(f"Unknown explanation type: {explanation_type}")
