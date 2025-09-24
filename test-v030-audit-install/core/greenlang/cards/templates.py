"""
Card Templates for Documentation
=================================

HuggingFace-style templates for packs, datasets, models, and pipelines.
"""

PACK_CARD_TEMPLATE = """---
name: {name}
version: {version}
type: pack
tags: {tags}
license: {license}
maintainer: {maintainer}
created: {created}
updated: {updated}
---

# {display_name}

## Overview

{description}

## Purpose

{purpose}

## Installation

```bash
gl pack add {name}
```

## Usage

### Quick Start

```python
{quick_start}
```

### Detailed Example

```python
{detailed_example}
```

## Inputs

{inputs}

## Outputs

{outputs}

## Configuration

{configuration}

## Dependencies

{dependencies}

## Assumptions

{assumptions}

## Validation

### Test Coverage

{test_coverage}

### Validation Methods

{validation_methods}

### Performance Metrics

{performance_metrics}

## Limitations

{limitations}

## Environmental Impact

### Carbon Footprint

{carbon_footprint}

### Sustainability Metrics

{sustainability_metrics}

## Model Cards

{model_cards}

## Dataset Cards

{dataset_cards}

## Changelog

{changelog}

## Citation

```bibtex
{citation}
```

## License

{license_text}

## Support

{support}

## Contributing

{contributing}

## Ethical Considerations

{ethical_considerations}

## References

{references}
"""

DATASET_CARD_TEMPLATE = """---
name: {name}
version: {version}
type: dataset
format: {format}
size: {size}
samples: {samples}
features: {features}
license: {license}
created: {created}
updated: {updated}
tags: {tags}
---

# Dataset Card for {display_name}

## Dataset Summary

{summary}

## Dataset Description

- **Homepage:** {homepage}
- **Repository:** {repository}
- **Paper:** {paper}
- **Leaderboard:** {leaderboard}
- **Point of Contact:** {contact}

### Dataset Purpose

{purpose}

### Supported Tasks

{supported_tasks}

### Languages

{languages}

## Dataset Structure

### Data Instances

{data_instances}

### Data Fields

{data_fields}

### Data Splits

{data_splits}

## Dataset Creation

### Curation Rationale

{curation_rationale}

### Source Data

#### Initial Data Collection

{data_collection}

#### Data Sources

{data_sources}

### Annotations

#### Annotation Process

{annotation_process}

#### Who are the annotators?

{annotators}

### Personal and Sensitive Information

{sensitive_info}

## Considerations for Using the Data

### Social Impact

{social_impact}

### Discussion of Biases

{biases}

### Other Known Limitations

{limitations}

## Environmental Impact

### Carbon Emissions

{carbon_emissions}

### Storage Requirements

{storage_requirements}

## Additional Information

### Dataset Curators

{curators}

### Licensing Information

{licensing}

### Citation Information

```bibtex
{citation}
```

### Contributions

{contributions}

## Data Quality

### Validation Metrics

{validation_metrics}

### Quality Checks

{quality_checks}

### Known Issues

{known_issues}

## Usage Examples

### Loading the Dataset

```python
{load_example}
```

### Processing Pipeline

```python
{processing_example}
```

### Integration with GreenLang

```python
{greenlang_example}
```
"""

MODEL_CARD_TEMPLATE = """---
name: {name}
version: {version}
type: model
architecture: {architecture}
parameters: {parameters}
license: {license}
training_data: {training_data}
training_emissions: {training_emissions}
inference_emissions: {inference_emissions}
created: {created}
updated: {updated}
tags: {tags}
---

# Model Card for {display_name}

## Model Details

### Model Description

{description}

- **Developed by:** {developers}
- **Model type:** {model_type}
- **Language(s):** {languages}
- **License:** {license}
- **Fine-tuned from:** {base_model}

### Model Sources

- **Repository:** {repository}
- **Paper:** {paper}
- **Demo:** {demo}

## Uses

### Direct Use

{direct_use}

### Downstream Use

{downstream_use}

### Out-of-Scope Use

{out_of_scope}

## Bias, Risks, and Limitations

### Recommendations

{recommendations}

### Known Limitations

{limitations}

### Ethical Considerations

{ethical_considerations}

## Training Details

### Training Data

{training_data_details}

### Training Procedure

#### Preprocessing

{preprocessing}

#### Training Hyperparameters

{hyperparameters}

#### Training Results

{training_results}

### Environmental Impact

- **Hardware Type:** {hardware}
- **Hours used:** {hours}
- **Cloud Provider:** {cloud_provider}
- **Compute Region:** {compute_region}
- **Carbon Emitted:** {carbon_emitted}

## Evaluation

### Testing Data & Metrics

#### Testing Data

{testing_data}

#### Metrics

{metrics}

### Results

{results}

## Technical Specifications

### Model Architecture

```yaml
{architecture_spec}
```

### Compute Requirements

#### Training

{training_requirements}

#### Inference

{inference_requirements}

## Usage

### Installation

```bash
{installation}
```

### Quick Start

```python
{quick_start}
```

### Advanced Usage

```python
{advanced_usage}
```

## Model Examination

### Interpretability

{interpretability}

### Fairness Analysis

{fairness_analysis}

## Carbon Footprint

### Training Emissions

{training_emissions_detail}

### Inference Emissions

{inference_emissions_detail}

### Optimization Strategies

{optimization_strategies}

## Citation

```bibtex
{citation}
```

## Model Card Authors

{authors}

## Model Card Contact

{contact}

## Updates and Versions

{version_history}
"""

PIPELINE_CARD_TEMPLATE = """---
name: {name}
version: {version}
type: pipeline
components: {components}
license: {license}
created: {created}
updated: {updated}
tags: {tags}
---

# Pipeline Card for {display_name}

## Pipeline Overview

{overview}

## Purpose and Use Cases

### Primary Purpose

{purpose}

### Use Cases

{use_cases}

### Users

{target_users}

## Pipeline Architecture

### Components

{components_detail}

### Data Flow

```mermaid
{data_flow_diagram}
```

### Configuration

```yaml
{configuration}
```

## Inputs and Outputs

### Input Specification

{input_spec}

### Output Specification

{output_spec}

### Data Transformations

{transformations}

## Performance

### Throughput

{throughput}

### Latency

{latency}

### Resource Usage

{resource_usage}

## Dependencies

### Software Dependencies

{software_deps}

### Hardware Requirements

{hardware_reqs}

### External Services

{external_services}

## Validation and Testing

### Test Coverage

{test_coverage}

### Validation Results

{validation_results}

### Benchmarks

{benchmarks}

## Deployment

### Deployment Options

{deployment_options}

### Scaling Considerations

{scaling}

### Monitoring

{monitoring}

## Environmental Impact

### Carbon Footprint

{carbon_footprint}

### Optimization Opportunities

{optimizations}

## Limitations and Assumptions

### Limitations

{limitations}

### Assumptions

{assumptions}

### Edge Cases

{edge_cases}

## Usage Examples

### Basic Usage

```python
{basic_usage}
```

### Advanced Configuration

```python
{advanced_config}
```

### Integration Example

```python
{integration_example}
```

## Troubleshooting

### Common Issues

{common_issues}

### Performance Tuning

{performance_tuning}

## Maintenance

### Update Schedule

{update_schedule}

### Deprecation Policy

{deprecation_policy}

## License and Citation

### License

{license_text}

### Citation

```bibtex
{citation}
```

## Support and Contact

{support_contact}

## Changelog

{changelog}
"""

# Minimal template for quick start
MINIMAL_CARD_TEMPLATE = """# {name}

## Overview
{description}

## Usage
```python
{usage}
```

## Inputs
{inputs}

## Outputs
{outputs}

## License
{license}
"""
