# -*- coding: utf-8 -*-
"""
Attention Visualization for Transformer Models
================================================

Provides visualization and interpretation of transformer attention weights
for process heat ML models. Enables interpretability of:
  - Single-head attention patterns
  - Multi-head aggregation across layers
  - Temporal attention (which timesteps matter)
  - Feature attention (which sensors/features matter)
  - Cross-attention between equipment

This module supports PyTorch and HuggingFace transformers with export
to HTML/PNG for stakeholder communication.

Key Components:
  - AttentionVisualizer: Main interface for attention analysis
  - AttentionWeights: Data model for attention matrices
  - AttentionSummary: Aggregated attention statistics
  - VisualizationExporter: HTML/PNG export functionality

Example:
    >>> from greenlang.ml.explainability.attention_visualizer import AttentionVisualizer
    >>> visualizer = AttentionVisualizer(model, feature_names=sensor_names)
    >>> weights = visualizer.extract_attention_weights(input_data)
    >>> heatmap = visualizer.visualize_attention(weights, layer=0, head=0)
    >>> summary = visualizer.get_attention_summary(weights)
    >>> visualizer.export_visualization(summary, format='html', output_path='./attn.html')

Author: GreenLang ML Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class VisualizationType(str, Enum):
    """Types of attention visualizations."""
    HEATMAP = "heatmap"
    FLOW = "flow"
    NETWORK = "network"
    TIMESERIES = "timeseries"
    DISTRIBUTION = "distribution"


class ExportFormat(str, Enum):
    """Export formats for visualizations."""
    HTML = "html"
    PNG = "png"
    SVG = "svg"
    JSON = "json"
    CSV = "csv"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class AttentionWeights:
    """Container for raw attention weights from transformer model."""

    # Shape: (num_layers, num_heads, seq_len, seq_len)
    weights: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "Raw attention weight matrices"}
    )

    layer: int = field(
        default=0,
        metadata={"description": "Layer index (0-indexed)"}
    )

    head: int = field(
        default=0,
        metadata={"description": "Head index (0-indexed)"}
    )

    # Optional feature/timestep names
    feature_names: Optional[List[str]] = field(
        default=None,
        metadata={"description": "Names of input features/sensors"}
    )

    timestep_names: Optional[List[str]] = field(
        default=None,
        metadata={"description": "Names of timesteps"}
    )

    # Provenance
    model_name: str = field(
        default="unknown",
        metadata={"description": "Source model identifier"}
    )

    timestamp: datetime = field(
        default_factory=datetime.utcnow,
        metadata={"description": "When weights were extracted"}
    )

    input_shape: Tuple[int, ...] = field(
        default=(),
        metadata={"description": "Shape of original input"}
    )

    def get_shape(self) -> Tuple[int, int, int, int]:
        """Get shape: (num_layers, num_heads, seq_len, seq_len)."""
        return self.weights.shape

    def validate(self) -> bool:
        """Validate attention weights are properly formed."""
        if self.weights.size == 0:
            return False
        if len(self.weights.shape) != 4:
            return False
        # Check weights sum to ~1.0 along last dimension
        row_sums = np.sum(self.weights, axis=-1)
        return np.allclose(row_sums, 1.0, atol=0.01)


@dataclass
class AttentionSummary:
    """Aggregated attention statistics across heads/layers."""

    # Aggregated: (num_layers, seq_len, seq_len)
    aggregated_weights: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "Mean attention across heads"}
    )

    # Per-feature importance: (seq_len,)
    feature_importance: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "Mean attention received by each feature"}
    )

    # Top attended positions
    top_attended_features: List[Tuple[str, float]] = field(
        default_factory=list,
        metadata={"description": "Top K features by attention"}
    )

    # Top features that attend
    top_attending_features: List[Tuple[str, float]] = field(
        default_factory=list,
        metadata={"description": "Top K features giving attention"}
    )

    # Layer analysis
    layer_importance: np.ndarray = field(
        default_factory=lambda: np.array([]),
        metadata={"description": "Attention importance per layer"}
    )

    # Head analysis
    head_importance: Dict[int, np.ndarray] = field(
        default_factory=dict,
        metadata={"description": "Per-head attention patterns"}
    )

    # Provenance
    provenance_hash: str = field(
        default="",
        metadata={"description": "SHA-256 hash for audit trail"}
    )

    processing_time_ms: float = field(
        default=0.0,
        metadata={"description": "Time to generate summary"}
    )

    def calculate_provenance(self) -> str:
        """Calculate SHA-256 hash of summary data."""
        data_str = (
            f"{self.aggregated_weights.tobytes()}"
            f"{self.feature_importance.tobytes()}"
            f"{str(self.top_attended_features)}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()


# =============================================================================
# ATTENTION VISUALIZER CLASS
# =============================================================================

class AttentionVisualizer:
    """
    Main interface for transformer attention visualization and analysis.

    Supports extraction, analysis, and visualization of attention weights
    from PyTorch transformer models and HuggingFace transformers.

    Attributes:
        model: The transformer model to visualize
        feature_names: Optional names for input features
        enable_caching: Cache extracted attention weights
        _attention_cache: Cache for attention weight extraction
        _visualization_cache: Cache for generated visualizations

    Example:
        >>> visualizer = AttentionVisualizer(
        ...     model,
        ...     feature_names=['temperature', 'pressure', 'flow_rate']
        ... )
        >>> weights = visualizer.extract_attention_weights(inputs)
        >>> heatmap = visualizer.visualize_attention(weights, layer=0, head=0)
        >>> summary = visualizer.get_attention_summary(weights)
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        enable_caching: bool = True
    ) -> None:
        """
        Initialize AttentionVisualizer.

        Args:
            model: PyTorch or HuggingFace transformer model
            feature_names: Optional names for features/sensors
            enable_caching: Enable caching of attention weights
        """
        self.model = model
        self.feature_names = feature_names or []
        self.enable_caching = enable_caching
        self._attention_cache: Dict[str, AttentionWeights] = {}
        self._visualization_cache: Dict[str, Any] = {}
        self._model_type = self._detect_model_type()

        logger.info(
            f"AttentionVisualizer initialized for {self._model_type} model"
        )

    def _detect_model_type(self) -> str:
        """Detect model type (pytorch, huggingface, etc.)."""
        model_class = type(self.model).__name__
        if "huggingface" in str(type(self.model)).lower():
            return "huggingface"
        elif hasattr(self.model, "encoder"):
            return "pytorch_transformer"
        else:
            return "unknown"

    def extract_attention_weights(
        self,
        input_data: Union[np.ndarray, Any],
        return_all_layers: bool = True,
        return_all_heads: bool = False
    ) -> AttentionWeights:
        """
        Extract attention weight matrices from transformer model.

        Args:
            input_data: Input tensor/array to model
            return_all_layers: Include all layers (vs. last layer only)
            return_all_heads: Include all heads (vs. mean across heads)

        Returns:
            AttentionWeights containing attention matrices

        Raises:
            ValueError: If model doesn't support attention extraction
            RuntimeError: If extraction fails
        """
        start_time = datetime.utcnow()

        try:
            # Convert numpy to torch if needed
            if isinstance(input_data, np.ndarray):
                input_data = self._numpy_to_torch(input_data)

            # Extract based on model type
            if self._model_type == "huggingface":
                weights = self._extract_hf_attention(
                    input_data,
                    return_all_layers,
                    return_all_heads
                )
            else:
                weights = self._extract_pytorch_attention(
                    input_data,
                    return_all_layers,
                    return_all_heads
                )

            # Set metadata
            weights.model_name = type(self.model).__name__
            weights.input_shape = tuple(input_data.shape)

            # Validate
            if not weights.validate():
                logger.warning("Extracted attention weights failed validation")

            logger.info(
                f"Extracted attention weights: shape={weights.get_shape()}, "
                f"valid={weights.validate()}"
            )

            return weights

        except Exception as e:
            logger.error(f"Failed to extract attention weights: {e}")
            raise RuntimeError(f"Attention extraction failed: {str(e)}")

    def _extract_hf_attention(
        self,
        input_data: Any,
        return_all_layers: bool,
        return_all_heads: bool
    ) -> AttentionWeights:
        """Extract attention from HuggingFace transformer."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for attention extraction")

        with torch.no_grad():
            outputs = self.model(
                input_data,
                output_attentions=True,
                return_dict=True
            )

        # Get attention from outputs
        if hasattr(outputs, "attentions"):
            attention_tuple = outputs.attentions
        else:
            raise ValueError("Model does not output attention weights")

        # Stack layers: (num_layers, batch, num_heads, seq_len, seq_len)
        attention_stacked = torch.stack(attention_tuple, dim=0)

        # Average over batch dimension
        attention_mean = attention_stacked.mean(dim=1)

        # Convert to numpy: (num_layers, num_heads, seq_len, seq_len)
        weights_np = attention_mean.cpu().numpy()

        return AttentionWeights(
            weights=weights_np,
            feature_names=self.feature_names
        )

    def _extract_pytorch_attention(
        self,
        input_data: Any,
        return_all_layers: bool,
        return_all_heads: bool
    ) -> AttentionWeights:
        """Extract attention from PyTorch transformer."""
        try:
            import torch
            from torch.nn import TransformerEncoder, MultiheadAttention
        except ImportError:
            raise ImportError("PyTorch required")

        # Register hooks to capture attention
        attention_weights = []

        def attention_hook(module: Any, input: Any, output: Any) -> None:
            """Hook to capture attention weights."""
            if isinstance(module, MultiheadAttention):
                attention_weights.append(output[1] if isinstance(output, tuple) else output)

        # Register hooks
        hooks = []
        for module in self.model.modules():
            if isinstance(module, MultiheadAttention):
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)

        # Forward pass
        try:
            with torch.no_grad():
                _ = self.model(input_data)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        # Stack attention weights
        if attention_weights:
            weights_stacked = torch.stack(attention_weights, dim=0)
            weights_np = weights_stacked.cpu().numpy()
        else:
            raise ValueError("No attention weights captured")

        return AttentionWeights(
            weights=weights_np,
            feature_names=self.feature_names
        )

    def visualize_attention(
        self,
        attention_weights: AttentionWeights,
        layer: int = 0,
        head: int = 0,
        viz_type: VisualizationType = VisualizationType.HEATMAP,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Dict[str, Any]:
        """
        Create attention visualization for specific layer/head.

        Args:
            attention_weights: AttentionWeights object
            layer: Layer index to visualize
            head: Head index to visualize
            viz_type: Type of visualization
            figsize: Figure size for matplotlib

        Returns:
            Dictionary with visualization data and figure object

        Raises:
            ValueError: If layer/head indices invalid
            ImportError: If matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")

        # Validate indices
        num_layers, num_heads, seq_len, _ = attention_weights.get_shape()
        if layer >= num_layers or head >= num_heads:
            raise ValueError(
                f"Invalid layer {layer} or head {head} "
                f"(max: {num_layers} layers, {num_heads} heads)"
            )

        # Extract specific attention
        attn_matrix = attention_weights.weights[layer, head, :, :]

        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            attn_matrix,
            ax=ax,
            cmap="viridis",
            cbar_kws={"label": "Attention Weight"},
            xticklabels=attention_weights.feature_names or "auto",
            yticklabels=attention_weights.feature_names or "auto",
            vmin=0.0,
            vmax=1.0
        )

        ax.set_title(
            f"Attention Weights: Layer {layer}, Head {head}",
            fontsize=14,
            fontweight="bold"
        )
        ax.set_xlabel("Attended Features")
        ax.set_ylabel("Attending Features")

        return {
            "figure": fig,
            "matrix": attn_matrix,
            "layer": layer,
            "head": head,
            "type": viz_type.value
        }

    def get_attention_summary(
        self,
        attention_weights: AttentionWeights,
        top_k: int = 5
    ) -> AttentionSummary:
        """
        Generate summary of attention across all heads and layers.

        Args:
            attention_weights: AttentionWeights object
            top_k: Number of top features to extract

        Returns:
            AttentionSummary with aggregated statistics

        Raises:
            ValueError: If weights invalid
        """
        start_time = datetime.utcnow()

        if not attention_weights.validate():
            raise ValueError("Invalid attention weights")

        weights = attention_weights.weights
        num_layers, num_heads, seq_len, _ = weights.shape

        # Average across heads: (num_layers, seq_len, seq_len)
        aggregated = weights.mean(axis=1)

        # Feature importance: mean attention received
        feature_importance = aggregated.mean(axis=0).mean(axis=0)

        # Top attended features
        feature_names = attention_weights.feature_names or [
            f"feature_{i}" for i in range(seq_len)
        ]
        top_attended = sorted(
            zip(feature_names, feature_importance),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Top attending features
        attending_importance = aggregated.mean(axis=0).mean(axis=1)
        top_attending = sorted(
            zip(feature_names, attending_importance),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Layer importance
        layer_importance = aggregated.mean(axis=(1, 2))

        # Head importance per layer
        head_importance = {}
        for layer_idx in range(num_layers):
            head_imp = weights[layer_idx].mean(axis=(1, 2))
            head_importance[layer_idx] = head_imp

        summary = AttentionSummary(
            aggregated_weights=aggregated,
            feature_importance=feature_importance,
            top_attended_features=top_attended,
            top_attending_features=top_attending,
            layer_importance=layer_importance,
            head_importance=head_importance,
            processing_time_ms=(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
        )

        # Calculate provenance
        summary.provenance_hash = summary.calculate_provenance()

        logger.info(
            f"Generated attention summary: "
            f"top_attended={[f[0] for f in top_attended]}, "
            f"provenance={summary.provenance_hash[:16]}..."
        )

        return summary

    def highlight_important_features(
        self,
        attention_weights: AttentionWeights,
        threshold: float = 0.1,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Identify most important features based on attention.

        Args:
            attention_weights: AttentionWeights object
            threshold: Attention threshold (features above)
            top_k: Alternatively, return top K features

        Returns:
            Dictionary with important features and statistics

        Raises:
            ValueError: If weights invalid
        """
        if not attention_weights.validate():
            raise ValueError("Invalid attention weights")

        weights = attention_weights.weights

        # Average across all layers and heads
        avg_attention = weights.mean(axis=(0, 1))

        # Feature importance: mean attention
        feature_importance = avg_attention.mean(axis=0)

        feature_names = attention_weights.feature_names or [
            f"feature_{i}" for i in range(len(feature_importance))
        ]

        # Select features
        if top_k is not None:
            indices = np.argsort(feature_importance)[-top_k:]
        else:
            indices = np.where(feature_importance >= threshold)[0]

        important_features = [
            {
                "name": feature_names[i],
                "importance": float(feature_importance[i]),
                "rank": int(np.argsort(feature_importance)[::-1].tolist().index(i)) + 1
            }
            for i in sorted(indices)
        ]

        return {
            "important_features": important_features,
            "feature_importances": dict(zip(feature_names, feature_importance)),
            "threshold": threshold,
            "count": len(important_features)
        }

    def export_visualization(
        self,
        summary: AttentionSummary,
        format: ExportFormat = ExportFormat.HTML,
        output_path: Optional[Union[str, Path]] = None
    ) -> Union[str, Path]:
        """
        Export attention visualization to file.

        Args:
            summary: AttentionSummary to export
            format: Export format (html, png, json, csv)
            output_path: Output file path

        Returns:
            Path to exported file

        Raises:
            ImportError: If required library not available
            ValueError: If format not supported
        """
        if format == ExportFormat.HTML:
            return self._export_html(summary, output_path)
        elif format == ExportFormat.JSON:
            return self._export_json(summary, output_path)
        elif format == ExportFormat.CSV:
            return self._export_csv(summary, output_path)
        elif format == ExportFormat.PNG:
            return self._export_png(summary, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_html(
        self,
        summary: AttentionSummary,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Export attention summary as interactive HTML."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for HTML export")

        # Default output path
        if output_path is None:
            output_path = Path("attention_summary.html")
        else:
            output_path = Path(output_path)

        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Attention Visualization Summary</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .provenance {{ font-size: 12px; color: #666; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>Transformer Attention Visualization</h1>
            <p>Generated: {datetime.utcnow().isoformat()}</p>

            <div class="metric">
                <div>Processing Time</div>
                <div class="metric-value">{summary.processing_time_ms:.2f}ms</div>
            </div>

            <div class="metric">
                <div>Features Analyzed</div>
                <div class="metric-value">{len(summary.feature_importance)}</div>
            </div>

            <h2>Top Attended Features</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Attention Score</th>
                </tr>
        """

        for feature_name, importance in summary.top_attended_features:
            html_content += f"""
                <tr>
                    <td>{feature_name}</td>
                    <td>{importance:.4f}</td>
                </tr>
            """

        html_content += """
            </table>

            <h2>Top Attending Features</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Attention Score</th>
                </tr>
        """

        for feature_name, importance in summary.top_attending_features:
            html_content += f"""
                <tr>
                    <td>{feature_name}</td>
                    <td>{importance:.4f}</td>
                </tr>
            """

        html_content += f"""
            </table>

            <div class="provenance">
                <strong>Provenance Hash:</strong> {summary.provenance_hash}<br>
                <strong>Processing Time:</strong> {summary.processing_time_ms:.2f}ms
            </div>
        </body>
        </html>
        """

        # Write to file
        output_path.write_text(html_content, encoding="utf-8")
        logger.info(f"Exported HTML visualization to {output_path}")

        return output_path

    def _export_json(
        self,
        summary: AttentionSummary,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Export attention summary as JSON."""
        if output_path is None:
            output_path = Path("attention_summary.json")
        else:
            output_path = Path(output_path)

        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "provenance_hash": summary.provenance_hash,
            "processing_time_ms": summary.processing_time_ms,
            "top_attended_features": [
                {"name": name, "importance": float(imp)}
                for name, imp in summary.top_attended_features
            ],
            "top_attending_features": [
                {"name": name, "importance": float(imp)}
                for name, imp in summary.top_attending_features
            ],
            "layer_importance": summary.layer_importance.tolist(),
            "feature_importance": summary.feature_importance.tolist(),
        }

        output_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )
        logger.info(f"Exported JSON visualization to {output_path}")

        return output_path

    def _export_csv(
        self,
        summary: AttentionSummary,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Export attention summary as CSV."""
        if output_path is None:
            output_path = Path("attention_summary.csv")
        else:
            output_path = Path(output_path)

        csv_lines = [
            "Feature,Attention Score,Rank",
            ""
        ]

        for idx, (name, importance) in enumerate(summary.top_attended_features, 1):
            csv_lines.append(f'"{name}",{importance:.6f},{idx}')

        output_path.write_text(
            "\n".join(csv_lines),
            encoding="utf-8"
        )
        logger.info(f"Exported CSV visualization to {output_path}")

        return output_path

    def _export_png(
        self,
        summary: AttentionSummary,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Export attention summary as PNG image."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for PNG export")

        if output_path is None:
            output_path = Path("attention_summary.png")
        else:
            output_path = Path(output_path)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Heatmap of aggregated attention
        sns.heatmap(
            summary.aggregated_weights[0],
            ax=axes[0, 0],
            cmap="viridis",
            cbar_kws={"label": "Attention"}
        )
        axes[0, 0].set_title("Aggregated Attention (First Layer)")

        # Feature importance bar chart
        features = [f[0] for f in summary.top_attended_features]
        importances = [f[1] for f in summary.top_attended_features]
        axes[0, 1].barh(features, importances, color="steelblue")
        axes[0, 1].set_xlabel("Attention Score")
        axes[0, 1].set_title("Top Attended Features")

        # Layer importance
        axes[1, 0].plot(summary.layer_importance, marker="o", linewidth=2)
        axes[1, 0].set_xlabel("Layer")
        axes[1, 0].set_ylabel("Importance")
        axes[1, 0].set_title("Layer Importance")
        axes[1, 0].grid(True, alpha=0.3)

        # Distribution of feature importance
        axes[1, 1].hist(summary.feature_importance, bins=20, edgecolor="black")
        axes[1, 1].set_xlabel("Attention Score")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Distribution of Feature Importance")

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Exported PNG visualization to {output_path}")

        return output_path

    def _numpy_to_torch(self, data: np.ndarray) -> Any:
        """Convert numpy array to PyTorch tensor."""
        try:
            import torch
            return torch.from_numpy(data).float()
        except ImportError:
            raise ImportError("PyTorch required for conversion")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "AttentionVisualizer",
    "AttentionWeights",
    "AttentionSummary",
    "VisualizationType",
    "ExportFormat",
]
