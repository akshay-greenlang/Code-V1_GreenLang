# Attention Visualization for Transformer Models

## Overview

The `AttentionVisualizer` module provides comprehensive visualization and interpretation of transformer attention weights for Process Heat ML models. This enables understanding which timesteps and sensors matter most in predictions.

## Key Features

### 1. Attention Weight Extraction
Extract attention matrices from PyTorch and HuggingFace transformer models:

```python
from greenlang.ml.explainability.attention_visualizer import AttentionVisualizer

# Initialize visualizer with your model
visualizer = AttentionVisualizer(
    model=your_transformer_model,
    feature_names=['temperature', 'pressure', 'flow_rate', 'oxygen', 'fuel_flow']
)

# Extract attention weights from input data
input_data = np.random.randn(1, 8, 768)  # batch=1, seq_len=8, hidden=768
attention_weights = visualizer.extract_attention_weights(
    input_data,
    return_all_layers=True,
    return_all_heads=False
)

# Check validity
print(f"Valid: {attention_weights.validate()}")
print(f"Shape: {attention_weights.get_shape()}")  # (num_layers, num_heads, seq_len, seq_len)
```

### 2. Attention Summary & Analysis
Generate aggregated statistics across layers and heads:

```python
# Get comprehensive summary
summary = visualizer.get_attention_summary(
    attention_weights,
    top_k=5
)

# Top attended features (which sensors receive most attention)
print("Top Attended Sensors:")
for name, importance in summary.top_attended_features:
    print(f"  {name}: {importance:.4f}")

# Top attending features (which sensors give most attention)
print("\nTop Attending Sensors:")
for name, importance in summary.top_attending_features:
    print(f"  {name}: {importance:.4f}")

# Layer-wise importance
print(f"\nLayer Importance: {summary.layer_importance}")

# Provenance for audit trail
print(f"Provenance Hash: {summary.provenance_hash}")
```

### 3. Feature Importance Highlighting
Identify critical features using attention weights:

```python
# Get important features with threshold
important = visualizer.highlight_important_features(
    attention_weights,
    threshold=0.1,  # Features receiving >10% attention
    top_k=None      # Include all above threshold
)

# Or get top-K most important
important = visualizer.highlight_important_features(
    attention_weights,
    top_k=5  # Top 5 features
)

for feature in important["important_features"]:
    print(f"  {feature['name']}: importance={feature['importance']:.4f}, rank={feature['rank']}")
```

### 4. Visualization Generation
Create visual representations of attention patterns:

```python
# Single-head attention heatmap
viz_result = visualizer.visualize_attention(
    attention_weights,
    layer=0,
    head=0,
    viz_type=VisualizationType.HEATMAP,
    figsize=(12, 10)
)

# Access figure for display or saving
fig = viz_result["figure"]
fig.savefig("attention_layer0_head0.png", dpi=300)

# Get the attention matrix
attn_matrix = viz_result["matrix"]  # shape: (seq_len, seq_len)
```

### 5. Export Visualizations
Export analysis results to multiple formats:

```python
from greenlang.ml.explainability.attention_visualizer import ExportFormat

# Export as interactive HTML
visualizer.export_visualization(
    summary,
    format=ExportFormat.HTML,
    output_path="attention_report.html"
)

# Export as JSON for programmatic use
visualizer.export_visualization(
    summary,
    format=ExportFormat.JSON,
    output_path="attention_data.json"
)

# Export as CSV for spreadsheet analysis
visualizer.export_visualization(
    summary,
    format=ExportFormat.CSV,
    output_path="attention_features.csv"
)

# Export as PNG with multi-panel summary
visualizer.export_visualization(
    summary,
    format=ExportFormat.PNG,
    output_path="attention_summary.png"
)
```

## Process Heat Specific Usage

### Thermal Command Agent (GL-001)
```python
# Sensor names for thermal optimization
sensor_names = [
    "supply_temp",
    "return_temp",
    "flow_rate",
    "supply_pressure",
    "return_pressure",
    "load_demand",
    "ambient_temp",
    "pump_speed"
]

visualizer = AttentionVisualizer(model, feature_names=sensor_names)
weights = visualizer.extract_attention_weights(sensor_data)
summary = visualizer.get_attention_summary(weights, top_k=5)

print("Which sensors matter most for thermal control:")
for name, imp in summary.top_attended_features:
    print(f"  {name}: {imp:.1%}")
```

### Combustion Optimizer (GL-018)
```python
# Combustion air/fuel optimization
sensor_names = [
    "fuel_flow",
    "air_flow",
    "oxygen_ppm",
    "co_ppm",
    "burner_temp",
    "efficiency",
    "nox_emission",
    "excess_air"
]

visualizer = AttentionVisualizer(model, feature_names=sensor_names)
weights = visualizer.extract_attention_weights(combustion_data)

# Get top attended features
important = visualizer.highlight_important_features(weights, top_k=3)
print("Critical combustion parameters:")
for feat in important["important_features"]:
    print(f"  {feat['name']}: rank={feat['rank']}")
```

### Predictive Maintenance (GL-013)
```python
# Failure prediction - which sensor patterns indicate problems
sensor_names = [
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "temperature",
    "pressure",
    "acoustic_emission",
    "electrical_current",
    "efficiency_ratio"
]

visualizer = AttentionVisualizer(model, feature_names=sensor_names)
weights = visualizer.extract_attention_weights(equipment_data)
summary = visualizer.get_attention_summary(weights)

# Export for maintenance team
visualizer.export_visualization(
    summary,
    format=ExportFormat.HTML,
    output_path="equipment_health_analysis.html"
)
```

## Data Models

### AttentionWeights
Raw attention matrices from transformer model:

```python
@dataclass
class AttentionWeights:
    weights: np.ndarray                    # (num_layers, num_heads, seq_len, seq_len)
    layer: int = 0                         # Active layer
    head: int = 0                          # Active head
    feature_names: Optional[List[str]]     # Sensor/feature names
    timestep_names: Optional[List[str]]    # Timestep labels
    model_name: str = "unknown"            # Source model
    timestamp: datetime = field(...)       # Extraction time
    input_shape: Tuple[int, ...] = ()      # Original input shape

    def validate() -> bool                 # Check weights sum to 1.0
    def get_shape() -> Tuple[int, int, int, int]  # Return shape
```

### AttentionSummary
Aggregated attention statistics:

```python
@dataclass
class AttentionSummary:
    aggregated_weights: np.ndarray         # (num_layers, seq_len, seq_len)
    feature_importance: np.ndarray         # (seq_len,)
    top_attended_features: List[Tuple[str, float]]  # Top K
    top_attending_features: List[Tuple[str, float]] # Top K
    layer_importance: np.ndarray           # Per-layer importance
    head_importance: Dict[int, np.ndarray] # Per-head in each layer
    provenance_hash: str                   # SHA-256 audit trail
    processing_time_ms: float              # Generation time
```

## Visualization Types

### Heatmap (Default)
Shows raw attention pattern between all timesteps:
- Dark = low attention
- Bright = high attention
- Good for finding patterns like "final token attends to all"

### Flow Diagram
Shows attention flow from left (attending) to right (attended):
- Thicker arrows = stronger attention
- Good for understanding decision flow

### Network Graph
Shows attention as network graph:
- Nodes = timesteps
- Edges = attention flows
- Good for complex patterns

### Distribution
Shows histogram of attention weights:
- Normal distribution = uniform attention
- Skewed = focused attention

## Integration with Agents

```python
from greenlang.agents.process_heat.gl_018_unified_combustion import CombustionOptimizer
from greenlang.ml.explainability.attention_visualizer import AttentionVisualizer

# Initialize agent
agent = CombustionOptimizer(config)

# Attach visualizer
visualizer = AttentionVisualizer(
    agent.model,
    feature_names=agent.get_sensor_names()
)

# During inference
predictions = agent.process(sensor_data)

# Explain via attention
attention_weights = visualizer.extract_attention_weights(
    agent.model.prepare_input(sensor_data)
)
summary = visualizer.get_attention_summary(attention_weights)

# Combine prediction with explanation
explanation = {
    "prediction": predictions,
    "top_attended_features": summary.top_attended_features,
    "provenance": summary.provenance_hash
}
```

## API Reference

### AttentionVisualizer Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `extract_attention_weights()` | Get attention matrices from model | `AttentionWeights` |
| `visualize_attention()` | Create heatmap visualization | `Dict[str, Any]` |
| `get_attention_summary()` | Aggregate across heads/layers | `AttentionSummary` |
| `highlight_important_features()` | Identify critical features | `Dict[str, Any]` |
| `export_visualization()` | Export to file | `Path` |

## Performance Considerations

### Memory Usage
- Large models (768 hidden dim, 512 seq len, 12 layers) use ~1.5 GB for attention
- Use `return_all_heads=False` to save 90% memory

### Speed
- Extraction: ~100ms for typical model
- Summary generation: ~50ms
- Visualization: ~200ms for PNG export

### Optimization
```python
# Enable caching for repeated analysis
visualizer = AttentionVisualizer(
    model,
    enable_caching=True  # Cache attention extraction
)

# For batch processing
results = []
for batch in data_batches:
    weights = visualizer.extract_attention_weights(batch)
    summary = visualizer.get_attention_summary(weights)
    results.append(summary)
```

## Troubleshooting

### "Model does not output attention weights"
- Ensure `output_attentions=True` in forward pass
- Check model supports attention output

### "Invalid attention weights" warning
- Weights may not sum to exactly 1.0 (numerical precision)
- Use `atol=0.01` in validation

### "matplotlib required for visualization"
- Install: `pip install matplotlib seaborn`

### "plotly required for HTML export"
- Install: `pip install plotly`

## Zero-Hallucination Compliance

The AttentionVisualizer is PURELY for model interpretability:

✓ ALLOWED:
- Visualizing what model attended to (explains decisions)
- Feature importance from attention patterns
- Reporting which sensors matter
- Exporting visualizations for human review

✗ NOT ALLOWED:
- Using attention weights for regulatory calculations
- Using attention to justify numeric emission values
- Using attention as a replacement for engineering formulas

## Examples

### Example 1: Identify Critical Sensors
```python
visualizer = AttentionVisualizer(model, feature_names=sensor_list)
weights = visualizer.extract_attention_weights(data)
important = visualizer.highlight_important_features(weights, top_k=3)

# Find sensors that need monitoring
critical_sensors = [f['name'] for f in important['important_features']]
print(f"Monitor these sensors: {critical_sensors}")
```

### Example 2: Compare Models
```python
# Model A
viz_a = AttentionVisualizer(model_a, feature_names=sensors)
summary_a = viz_a.get_attention_summary(weights_a)

# Model B
viz_b = AttentionVisualizer(model_b, feature_names=sensors)
summary_b = viz_b.get_attention_summary(weights_b)

# Compare top features
print("Model A focuses on:", summary_a.top_attended_features[:3])
print("Model B focuses on:", summary_b.top_attended_features[:3])
```

### Example 3: Generate Report
```python
visualizer = AttentionVisualizer(model, feature_names=sensors)
summary = visualizer.get_attention_summary(weights, top_k=10)

# Create comprehensive report
visualizer.export_visualization(summary, format=ExportFormat.HTML)
visualizer.export_visualization(summary, format=ExportFormat.PNG)
visualizer.export_visualization(summary, format=ExportFormat.JSON)

# All three formats now available for stakeholders
```

## References

- Vaswani et al. (2017): "Attention is All You Need" - original transformer paper
- Bisk et al. (2020): "Experience Grounds Language" - attention visualization survey
- Process Heat Agents: GL-001 through GL-020 agent documentation
