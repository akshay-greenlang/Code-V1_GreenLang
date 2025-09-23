"""
Card Generator for Documentation
=================================

Generates model and dataset cards from templates.
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List

from .templates import (
    PACK_CARD_TEMPLATE,
    DATASET_CARD_TEMPLATE,
    MODEL_CARD_TEMPLATE,
    PIPELINE_CARD_TEMPLATE,
    MINIMAL_CARD_TEMPLATE,
)


def generate_pack_card(
    name: str,
    version: str = "0.1.0",
    description: str = "",
    purpose: str = "",
    author: str = "",
    license: str = "MIT",
    tags: List[str] = None,
    minimal: bool = False,
    **kwargs,
) -> str:
    """
    Generate a pack card from template

    Args:
        name: Pack name
        version: Pack version
        description: Pack description
        purpose: Pack purpose
        author: Pack author/maintainer
        license: License type
        tags: List of tags
        minimal: Use minimal template
        **kwargs: Additional template variables

    Returns:
        Generated card content
    """
    if minimal:
        template = MINIMAL_CARD_TEMPLATE
        defaults = {
            "name": name,
            "description": description or "A GreenLang pack",
            "usage": f"from {name} import main\nresult = main.run()",
            "inputs": "- data: Input data dictionary",
            "outputs": "- result: Processing result",
            "license": license,
        }
    else:
        template = PACK_CARD_TEMPLATE
        defaults = {
            "name": name,
            "version": version,
            "display_name": name.replace("-", " ").title(),
            "description": description or "A GreenLang pack for climate intelligence",
            "purpose": purpose
            or "This pack provides functionality for climate-related computations",
            "tags": ", ".join(tags or ["climate", "greenlang"]),
            "license": license,
            "maintainer": author or "GreenLang Community",
            "created": datetime.now().date().isoformat(),
            "updated": datetime.now().date().isoformat(),
            # Usage examples
            "quick_start": f"""from {name.replace("-", "_")} import Pipeline

pipeline = Pipeline()
result = pipeline.run({{"input": "data"}})
print(result)""",
            "detailed_example": f"""from {name.replace("-", "_")} import Pipeline, Config

# Configure pipeline
config = Config(
    verbose=True,
    cache_enabled=True
)

# Initialize with config
pipeline = Pipeline(config=config)

# Run with detailed inputs
result = pipeline.run({{
    "data": "input_data",
    "parameters": {{
        "threshold": 0.5,
        "mode": "production"
    }}
}})

# Process results
if result.success:
    print(f"Output: {{result.data}}")
else:
    print(f"Error: {{result.error}}")""",
            # Specifications
            "inputs": """- `data` (dict): Input data dictionary
- `parameters` (dict, optional): Processing parameters
- `config` (Config, optional): Configuration object""",
            "outputs": """- `result` (Result): Processing result object
  - `success` (bool): Success status
  - `data` (dict): Output data
  - `metadata` (dict): Processing metadata""",
            "configuration": """```yaml
verbose: false
cache_enabled: true
timeout: 300
max_retries: 3
```""",
            "dependencies": """- greenlang>=0.1.0
- numpy>=1.19.0
- pandas>=1.2.0""",
            "assumptions": """- Input data is properly formatted
- Required environment variables are set
- Network connectivity for external services""",
            # Validation
            "test_coverage": "- Unit tests: 85% coverage\n- Integration tests: 70% coverage",
            "validation_methods": "- Input validation\n- Output verification\n- Performance benchmarks",
            "performance_metrics": "- Average latency: < 100ms\n- Throughput: > 1000 req/s",
            # Limitations
            "limitations": """- Maximum input size: 10MB
- Requires Python 3.8+
- Limited to specific data formats""",
            # Environmental impact
            "carbon_footprint": "Estimated: 0.1 kg CO2 per 1000 runs",
            "sustainability_metrics": "- Energy efficient algorithms\n- Optimized for minimal resource usage",
            # Additional sections
            "model_cards": "See `models/` directory for individual model cards",
            "dataset_cards": "See `datasets/` directory for dataset documentation",
            "changelog": "See CHANGELOG.md",
            "citation": f"""@software{{{name},
  title = {{{name.replace("-", " ").title()}}},
  author = {{{author or "GreenLang Community"}}},
  year = {{2024}},
  url = {{https://github.com/greenlang/{name}}}
}}""",
            "license_text": f"This project is licensed under the {license} License",
            "support": "- Issues: https://github.com/greenlang/issues\n- Discussions: https://github.com/greenlang/discussions",
            "contributing": "See CONTRIBUTING.md for guidelines",
            "ethical_considerations": "This pack is designed with environmental sustainability in mind",
            "references": "- [GreenLang Documentation](https://docs.greenlang.org)\n- [Climate Data Guide](https://climatedataguide.org)",
        }

    # Merge with user-provided kwargs
    defaults.update(kwargs)

    # Format template
    return template.format(**defaults)


def generate_dataset_card(
    name: str,
    format: str = "json",
    size: str = "unknown",
    samples: int = 0,
    features: List[str] = None,
    license: str = "MIT",
    summary: str = "",
    minimal: bool = False,
    **kwargs,
) -> str:
    """
    Generate a dataset card from template

    Args:
        name: Dataset name
        format: Data format (json, csv, parquet, etc.)
        size: Dataset size
        samples: Number of samples
        features: List of features
        license: License type
        summary: Dataset summary
        minimal: Use minimal template
        **kwargs: Additional template variables

    Returns:
        Generated card content
    """
    if minimal:
        template = MINIMAL_CARD_TEMPLATE
        defaults = {
            "name": name,
            "description": summary or f"Dataset for {name}",
            "usage": f"data = load_dataset('{name}')",
            "inputs": "N/A - This is a dataset",
            "outputs": f"- {samples} samples\n- Format: {format}",
            "license": license,
        }
    else:
        template = DATASET_CARD_TEMPLATE
        defaults = {
            "name": name,
            "version": "1.0.0",
            "display_name": name.replace("-", " ").title(),
            "format": format,
            "size": size,
            "samples": samples,
            "features": ", ".join(features or []),
            "license": license,
            "created": datetime.now().date().isoformat(),
            "updated": datetime.now().date().isoformat(),
            "tags": "climate, environmental, greenlang",
            # Description
            "summary": summary or f"A dataset for {name.replace('-', ' ')}",
            "homepage": f"https://greenlang.org/datasets/{name}",
            "repository": f"https://github.com/greenlang/datasets/{name}",
            "paper": "N/A",
            "leaderboard": "N/A",
            "contact": "contact@greenlang.org",
            "purpose": "This dataset is designed for climate intelligence applications",
            "supported_tasks": "- Classification\n- Regression\n- Time series analysis",
            "languages": "English",
            # Structure
            "data_instances": f"""```json
{{
  "id": "sample_001",
  "timestamp": "2024-01-01T00:00:00Z",
  "value": 42.0,
  "metadata": {{}}
}}
```""",
            "data_fields": """- `id` (string): Unique identifier
- `timestamp` (datetime): Timestamp
- `value` (float): Measurement value
- `metadata` (dict): Additional metadata""",
            "data_splits": f"""| Split | Samples |
|-------|---------|
| train | {int(samples * 0.8)} |
| valid | {int(samples * 0.1)} |
| test  | {int(samples * 0.1)} |""",
            # Creation
            "curation_rationale": "Created to support climate intelligence research",
            "data_collection": "Data collected from various environmental sensors",
            "data_sources": "- Public environmental databases\n- Sensor networks",
            "annotation_process": "Automated validation with manual review",
            "annotators": "Domain experts and automated systems",
            "sensitive_info": "No personal or sensitive information included",
            # Considerations
            "social_impact": "Supports climate action and environmental monitoring",
            "biases": "Geographic coverage may be limited to certain regions",
            "limitations": "- Temporal coverage limited\n- Spatial resolution constraints",
            # Environmental
            "carbon_emissions": "Minimal - data collection is energy efficient",
            "storage_requirements": f"Approximately {size}",
            # Additional
            "curators": "GreenLang Data Team",
            "licensing": f"Released under {license} license",
            "citation": f"""@dataset{{{name},
  title = {{{name.replace("-", " ").title()} Dataset}},
  author = {{GreenLang Data Team}},
  year = {{2024}},
  publisher = {{GreenLang}},
  url = {{https://greenlang.org/datasets/{name}}}
}}""",
            "contributions": "Thanks to all contributors",
            # Quality
            "validation_metrics": "- Completeness: 99%\n- Accuracy: 95%",
            "quality_checks": "- Schema validation\n- Range checks\n- Consistency verification",
            "known_issues": "None currently identified",
            # Examples
            "load_example": f"""from greenlang.datasets import load_dataset

dataset = load_dataset("{name}")
print(f"Loaded {{len(dataset)}} samples")""",
            "processing_example": f"""# Process dataset
for sample in dataset:
    # Apply transformations
    processed = transform(sample)
    # Use processed data
    results.append(processed)""",
            "greenlang_example": f"""from greenlang import Pipeline

pipeline = Pipeline()
results = pipeline.process_dataset("{name}")""",
        }

    defaults.update(kwargs)
    return template.format(**defaults)


def generate_model_card(
    name: str,
    architecture: str = "neural_network",
    parameters: str = "1M",
    license: str = "MIT",
    description: str = "",
    minimal: bool = False,
    **kwargs,
) -> str:
    """
    Generate a model card from template

    Args:
        name: Model name
        architecture: Model architecture
        parameters: Number of parameters
        license: License type
        description: Model description
        minimal: Use minimal template
        **kwargs: Additional template variables

    Returns:
        Generated card content
    """
    if minimal:
        template = MINIMAL_CARD_TEMPLATE
        defaults = {
            "name": name,
            "description": description or f"Model: {name}",
            "usage": f"model = load_model('{name}')\noutput = model.predict(input)",
            "inputs": "- input: Model input tensor/data",
            "outputs": "- output: Model predictions",
            "license": license,
        }
    else:
        template = MODEL_CARD_TEMPLATE
        defaults = {
            "name": name,
            "version": "1.0.0",
            "display_name": name.replace("-", " ").title(),
            "architecture": architecture,
            "parameters": parameters,
            "license": license,
            "training_data": "Custom climate dataset",
            "training_emissions": "10 kg CO2",
            "inference_emissions": "0.001 kg CO2 per 1000 inferences",
            "created": datetime.now().date().isoformat(),
            "updated": datetime.now().date().isoformat(),
            "tags": "climate, ml, greenlang",
            "description": description or f"A model for {name.replace('-', ' ')}",
            "developers": "GreenLang ML Team",
            "model_type": architecture,
            "languages": "English",
            "base_model": "None - trained from scratch",
            "repository": f"https://github.com/greenlang/models/{name}",
            "paper": "Link to paper if available",
            "demo": f"https://demo.greenlang.org/{name}",
            # Uses
            "direct_use": "Direct inference for climate predictions",
            "downstream_use": "Can be fine-tuned for specific applications",
            "out_of_scope": "Not intended for non-climate applications",
            "recommendations": "Use with appropriate data preprocessing",
            "limitations": "Limited to specific input formats and ranges",
            "ethical_considerations": "Designed for environmental benefit",
            # Training
            "training_data_details": "Trained on diverse climate datasets",
            "preprocessing": "Standard normalization and feature engineering",
            "hyperparameters": """```yaml
learning_rate: 0.001
batch_size: 32
epochs: 100
optimizer: adam
```""",
            "training_results": "Achieved 95% accuracy on validation set",
            # Environmental
            "hardware": "NVIDIA A100 GPU",
            "hours": "24",
            "cloud_provider": "AWS",
            "compute_region": "us-west-2",
            "carbon_emitted": "10 kg CO2",
            # Evaluation
            "testing_data": "Held-out test set",
            "metrics": "- Accuracy: 95%\n- F1 Score: 0.92\n- RMSE: 0.05",
            "results": "Model performs well across all metrics",
            # Technical
            "architecture_spec": f"""architecture: {architecture}
layers: 10
parameters: {parameters}
activation: relu
dropout: 0.2""",
            "training_requirements": "- GPU: 1x A100\n- RAM: 32GB\n- Storage: 100GB",
            "inference_requirements": "- CPU: 2 cores\n- RAM: 4GB\n- Storage: 1GB",
            # Usage
            "installation": "pip install greenlang-models",
            "quick_start": f"""from greenlang.models import load_model

model = load_model("{name}")
predictions = model.predict(input_data)""",
            "advanced_usage": f"""from greenlang.models import {name.replace('-', '_').title()}

# Load with custom config
model = {name.replace('-', '_').title()}(
    config={{"batch_size": 64}},
    device="cuda"
)

# Fine-tune
model.fine_tune(
    train_data=custom_data,
    epochs=10
)

# Save
model.save("path/to/model")""",
            # Examination
            "interpretability": "SHAP values available for feature importance",
            "fairness_analysis": "Model tested for geographic bias",
            # Emissions detail
            "training_emissions_detail": "Total: 10 kg CO2\nPer epoch: 0.1 kg CO2",
            "inference_emissions_detail": "0.001 kg CO2 per 1000 inferences",
            "optimization_strategies": "- Model quantization\n- Pruning\n- Knowledge distillation",
            "citation": f"""@model{{{name},
  title = {{{name.replace("-", " ").title()}}},
  author = {{GreenLang ML Team}},
  year = {{2024}},
  url = {{https://github.com/greenlang/models/{name}}}
}}""",
            "authors": "GreenLang ML Team",
            "contact": "ml@greenlang.org",
            "version_history": "- v1.0.0: Initial release",
        }

    defaults.update(kwargs)
    return template.format(**defaults)


def generate_pipeline_card(
    name: str,
    components: List[str] = None,
    license: str = "MIT",
    overview: str = "",
    minimal: bool = False,
    **kwargs,
) -> str:
    """
    Generate a pipeline card from template

    Args:
        name: Pipeline name
        components: List of pipeline components
        license: License type
        overview: Pipeline overview
        minimal: Use minimal template
        **kwargs: Additional template variables

    Returns:
        Generated card content
    """
    if minimal:
        template = MINIMAL_CARD_TEMPLATE
        defaults = {
            "name": name,
            "description": overview or f"Pipeline: {name}",
            "usage": f"pipeline = Pipeline('{name}')\nresult = pipeline.run(data)",
            "inputs": "- data: Input data for pipeline",
            "outputs": "- result: Pipeline output",
            "license": license,
        }
    else:
        template = PIPELINE_CARD_TEMPLATE
        components = components or ["preprocessor", "model", "postprocessor"]

        defaults = {
            "name": name,
            "version": "1.0.0",
            "display_name": name.replace("-", " ").title(),
            "components": ", ".join(components),
            "license": license,
            "created": datetime.now().date().isoformat(),
            "updated": datetime.now().date().isoformat(),
            "tags": "pipeline, climate, greenlang",
            "overview": overview or f"Pipeline for {name.replace('-', ' ')}",
            "purpose": "Process climate data through multiple stages",
            "use_cases": "- Data processing\n- Model inference\n- Result aggregation",
            "target_users": "Data scientists and climate researchers",
            # Architecture
            "components_detail": "\n".join(
                [f"- {c}: Handles {c} operations" for c in components]
            ),
            "data_flow_diagram": """graph LR
    A[Input] --> B[Preprocess]
    B --> C[Model]
    C --> D[Postprocess]
    D --> E[Output]""",
            "configuration": """pipeline:
  name: """
            + name
            + """
  components:"""
            + "\n".join([f"\n    - {c}" for c in components])
            + """
  config:
    parallel: true
    cache: true""",
            # I/O
            "input_spec": """```yaml
type: dict
required:
  - data: array
  - metadata: dict
optional:
  - config: dict
```""",
            "output_spec": """```yaml
type: dict
fields:
  - result: array
  - metadata: dict
  - metrics: dict
```""",
            "transformations": "- Normalization\n- Feature extraction\n- Aggregation",
            # Performance
            "throughput": "1000 samples/second",
            "latency": "< 100ms per sample",
            "resource_usage": "- CPU: 2 cores\n- Memory: 4GB",
            # Dependencies
            "software_deps": "- greenlang>=0.1.0\n- numpy>=1.19.0",
            "hardware_reqs": "- CPU: 2+ cores\n- RAM: 4GB minimum",
            "external_services": "None required",
            # Validation
            "test_coverage": "90% code coverage",
            "validation_results": "All tests passing",
            "benchmarks": "Performs 2x faster than baseline",
            # Deployment
            "deployment_options": "- Local\n- Docker\n- Kubernetes\n- Cloud Functions",
            "scaling": "Horizontal scaling supported",
            "monitoring": "Metrics exported to Prometheus",
            # Environmental
            "carbon_footprint": "0.01 kg CO2 per 1000 runs",
            "optimizations": "- Caching enabled\n- Batch processing",
            # Limitations
            "limitations": "- Maximum batch size: 1000\n- Memory constraints",
            "assumptions": "- Input data is preprocessed\n- Network connectivity",
            "edge_cases": "- Empty input handling\n- Malformed data",
            # Examples
            "basic_usage": f"""from greenlang import Pipeline

pipeline = Pipeline("{name}")
result = pipeline.run(input_data)""",
            "advanced_config": f"""pipeline = Pipeline(
    "{name}",
    config={{
        "parallel": True,
        "batch_size": 100,
        "cache": True
    }}
)

result = pipeline.run(
    data=input_data,
    metadata={{"source": "sensor"}}
)""",
            "integration_example": f"""# Integration with data source
from greenlang import Pipeline, DataSource

source = DataSource("climate_db")
pipeline = Pipeline("{name}")

# Stream processing
for batch in source.stream():
    result = pipeline.run(batch)
    save_results(result)""",
            # Troubleshooting
            "common_issues": "- Memory errors: Reduce batch size\n- Timeout: Increase limits",
            "performance_tuning": "- Enable caching\n- Use batch processing",
            # Maintenance
            "update_schedule": "Monthly updates",
            "deprecation_policy": "6 month notice for breaking changes",
            "license_text": f"Licensed under {license}",
            "citation": f"""@software{{{name}_pipeline,
  title = {{{name.replace("-", " ").title()} Pipeline}},
  author = {{GreenLang Team}},
  year = {{2024}}
}}""",
            "support_contact": "support@greenlang.org",
            "changelog": "See CHANGELOG.md for version history",
        }

    defaults.update(kwargs)
    return template.format(**defaults)


def generate_card_from_manifest(manifest_path: Path, card_type: str = "pack") -> str:
    """
    Generate a card from a manifest file

    Args:
        manifest_path: Path to manifest file (pack.yaml, etc.)
        card_type: Type of card to generate

    Returns:
        Generated card content
    """
    with open(manifest_path) as f:
        if manifest_path.suffix == ".yaml":
            manifest = yaml.safe_load(f)
        else:
            manifest = json.load(f)

    # Map manifest fields to card parameters
    params = {
        "name": manifest.get("name", "unknown"),
        "version": manifest.get("version", "0.1.0"),
        "description": manifest.get("description", ""),
        "license": manifest.get("license", "MIT"),
        "author": manifest.get("author", manifest.get("maintainer", "")),
        "tags": manifest.get("tags", []),
    }

    # Add type-specific fields
    if card_type == "dataset":
        params.update(
            {
                "format": manifest.get("format", "json"),
                "size": manifest.get("size", "unknown"),
                "samples": manifest.get("samples", 0),
                "features": manifest.get("features", []),
            }
        )
        return generate_dataset_card(**params)
    elif card_type == "model":
        params.update(
            {
                "architecture": manifest.get("architecture", "unknown"),
                "parameters": manifest.get("parameters", "unknown"),
            }
        )
        return generate_model_card(**params)
    elif card_type == "pipeline":
        params.update(
            {
                "components": manifest.get("components", []),
                "overview": manifest.get("overview", ""),
            }
        )
        return generate_pipeline_card(**params)
    else:
        return generate_pack_card(**params)
