#!/usr/bin/env python3
"""
GreenLang SDK Pipeline Creation Example

This example demonstrates how to:
- Create GreenLang pipelines programmatically
- Build complex multi-step workflows using code
- Configure advanced pipeline features
- Export pipelines to YAML for reuse
- Handle dynamic pipeline generation based on requirements

This shows patterns for building pipelines as code rather than YAML configuration.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

# GreenLang SDK imports (simulated structure)
import greenlang
from greenlang import Orchestrator, Workflow, WorkflowStep

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StepErrorHandling(Enum):
    """Error handling strategies for pipeline steps"""
    STOP = "stop"
    CONTINUE = "continue"
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"


@dataclass
class PipelineStep:
    """Enhanced pipeline step with advanced configuration"""
    name: str
    agent: str
    action: str
    inputs: Dict[str, Any]
    depends_on: Optional[List[str]] = None
    parallel: bool = False
    conditional: Optional[Dict[str, Any]] = None
    error_handling: Optional[Dict[str, Any]] = None
    timeout: Optional[str] = None
    cache_duration: Optional[str] = None
    outputs: Optional[Dict[str, str]] = None


@dataclass
class Pipeline:
    """Advanced pipeline configuration"""
    name: str
    version: Union[int, str]
    description: str
    vars: Optional[Dict[str, Any]] = None
    inputs: Optional[Dict[str, Any]] = None
    artifacts_dir: Optional[str] = None
    steps: Optional[List[PipelineStep]] = None
    outputs: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []

    def add_step(self, step: PipelineStep):
        """Add a step to the pipeline"""
        self.steps.append(step)

    def to_yaml(self, output_path: str):
        """Export pipeline to YAML format"""
        # Convert to dictionary format compatible with GreenLang YAML schema
        pipeline_dict = {
            "name": self.name,
            "version": self.version,
            "description": self.description
        }

        if self.vars:
            pipeline_dict["vars"] = self.vars

        if self.inputs:
            pipeline_dict["inputs"] = self.inputs

        if self.artifacts_dir:
            pipeline_dict["artifacts_dir"] = self.artifacts_dir

        if self.steps:
            pipeline_dict["steps"] = []
            for step in self.steps:
                step_dict = {
                    "name": step.name,
                    "agent": step.agent,
                    "action": step.action,
                    "inputs": step.inputs
                }

                if step.depends_on:
                    step_dict["depends_on"] = step.depends_on

                if step.parallel:
                    step_dict["parallel"] = step.parallel

                if step.conditional:
                    step_dict["conditional"] = step.conditional

                if step.error_handling:
                    step_dict["error_handling"] = step.error_handling

                if step.timeout:
                    step_dict["timeout"] = step.timeout

                if step.cache_duration:
                    step_dict["cache_duration"] = step.cache_duration

                if step.outputs:
                    step_dict["outputs"] = step.outputs

                pipeline_dict["steps"].append(step_dict)

        if self.outputs:
            pipeline_dict["outputs"] = self.outputs

        if self.execution:
            pipeline_dict["execution"] = self.execution

        # Write to YAML file
        with open(output_path, 'w') as f:
            yaml.dump(pipeline_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Pipeline exported to {output_path}")


class PipelineBuilder:
    """Builder class for constructing complex pipelines programmatically"""

    def __init__(self, name: str, version: Union[int, str] = 1, description: str = ""):
        self.pipeline = Pipeline(
            name=name,
            version=version,
            description=description,
            steps=[]
        )

    def set_variables(self, variables: Dict[str, Any]) -> 'PipelineBuilder':
        """Set pipeline-level variables"""
        self.pipeline.vars = variables
        return self

    def set_inputs(self, input_schema: Dict[str, Any]) -> 'PipelineBuilder':
        """Set pipeline input schema"""
        self.pipeline.inputs = input_schema
        return self

    def set_artifacts_dir(self, artifacts_dir: str) -> 'PipelineBuilder':
        """Set artifacts directory"""
        self.pipeline.artifacts_dir = artifacts_dir
        return self

    def set_outputs(self, output_mapping: Dict[str, Any]) -> 'PipelineBuilder':
        """Set pipeline output mapping"""
        self.pipeline.outputs = output_mapping
        return self

    def set_execution_config(self, execution_config: Dict[str, Any]) -> 'PipelineBuilder':
        """Set execution configuration"""
        self.pipeline.execution = execution_config
        return self

    def add_validation_step(
        self,
        name: str,
        data_source: str,
        validation_rules: List[str] = None,
        depends_on: List[str] = None
    ) -> 'PipelineBuilder':
        """Add a data validation step"""
        step = PipelineStep(
            name=name,
            agent="ValidationAgent",
            action="validate_data",
            inputs={
                "data": data_source,
                "rules": validation_rules or ["required_fields", "data_types", "ranges"]
            },
            depends_on=depends_on,
            error_handling={"on_failure": "stop", "message": f"Validation failed at {name}"}
        )
        self.pipeline.add_step(step)
        return self

    def add_carbon_calculation_step(
        self,
        name: str,
        energy_data: str,
        emission_factors: str,
        scope: int = None,
        depends_on: List[str] = None,
        parallel: bool = False
    ) -> 'PipelineBuilder':
        """Add a carbon emissions calculation step"""
        action_map = {
            1: "calculate_scope1_emissions",
            2: "calculate_scope2_emissions",
            None: "calculate_total_emissions"
        }

        step = PipelineStep(
            name=name,
            agent="CarbonAgent",
            action=action_map.get(scope, "calculate_total_emissions"),
            inputs={
                "energy_data": energy_data,
                "emission_factors": emission_factors
            },
            depends_on=depends_on,
            parallel=parallel,
            cache_duration="1h"  # Cache emissions calculations
        )
        self.pipeline.add_step(step)
        return self

    def add_renewable_analysis_step(
        self,
        name: str,
        renewable_data: str,
        grid_factors: str,
        depends_on: List[str] = None,
        conditional_expression: str = None
    ) -> 'PipelineBuilder':
        """Add renewable energy analysis step with conditional execution"""
        step = PipelineStep(
            name=name,
            agent="SolarOffsetAgent",
            action="analyze_renewable_impact",
            inputs={
                "renewable_data": renewable_data,
                "grid_factors": grid_factors
            },
            depends_on=depends_on,
            parallel=True
        )

        if conditional_expression:
            step.conditional = {
                "condition": conditional_expression,
                "on_skip": {
                    "renewable_offset": 0,
                    "renewable_percentage": 0
                }
            }

        self.pipeline.add_step(step)
        return self

    def add_aggregation_step(
        self,
        name: str,
        data_sources: List[str],
        aggregation_method: str = "sum",
        depends_on: List[str] = None
    ) -> 'PipelineBuilder':
        """Add data aggregation step"""
        step = PipelineStep(
            name=name,
            agent="CarbonAgent",
            action="aggregate_emissions",
            inputs={
                "data_sources": data_sources,
                "method": aggregation_method
            },
            depends_on=depends_on,
            error_handling={
                "on_failure": "stop",
                "message": f"Critical aggregation failure at {name}"
            }
        )
        self.pipeline.add_step(step)
        return self

    def add_report_generation_step(
        self,
        name: str,
        data_source: str,
        template: str,
        output_path: str,
        depends_on: List[str] = None,
        parallel: bool = False
    ) -> 'PipelineBuilder':
        """Add report generation step"""
        step = PipelineStep(
            name=name,
            agent="ReportAgent",
            action="generate_report",
            inputs={
                "data": data_source,
                "template": template
            },
            outputs={
                "report_path": output_path
            },
            depends_on=depends_on,
            parallel=parallel,
            error_handling={"on_failure": "continue", "log_level": "warning"}
        )
        self.pipeline.add_step(step)
        return self

    def add_parallel_processing_group(
        self,
        base_name: str,
        agent: str,
        action: str,
        data_partitions: List[str],
        depends_on: List[str] = None
    ) -> 'PipelineBuilder':
        """Add a group of parallel processing steps"""
        for i, partition in enumerate(data_partitions):
            step = PipelineStep(
                name=f"{base_name}_{i+1}",
                agent=agent,
                action=action,
                inputs={"data": partition},
                depends_on=depends_on,
                parallel=True
            )
            self.pipeline.add_step(step)
        return self

    def add_conditional_branch(
        self,
        condition: str,
        true_steps: List[PipelineStep],
        false_steps: List[PipelineStep] = None
    ) -> 'PipelineBuilder':
        """Add conditional branching logic"""
        # Add true branch steps
        for step in true_steps:
            step.conditional = {"condition": condition}
            self.pipeline.add_step(step)

        # Add false branch steps if provided
        if false_steps:
            for step in false_steps:
                step.conditional = {"condition": f"!({condition})"}
                self.pipeline.add_step(step)

        return self

    def build(self) -> Pipeline:
        """Build and return the completed pipeline"""
        return self.pipeline


class PipelineTemplates:
    """Pre-built pipeline templates for common use cases"""

    @staticmethod
    def create_basic_facility_analysis(facility_name: str) -> Pipeline:
        """Create a basic single-facility carbon analysis pipeline"""
        builder = PipelineBuilder(
            name=f"{facility_name.lower().replace(' ', '-')}-analysis",
            version=1,
            description=f"Carbon footprint analysis for {facility_name}"
        )

        return (builder
                .set_variables({
                    "reporting_year": 2024,
                    "calculation_precision": 3
                })
                .set_inputs({
                    "facility_data": {
                        "type": "object",
                        "required": True,
                        "properties": {
                            "name": {"type": "string"},
                            "electricity_kwh": {"type": "number", "minimum": 0},
                            "natural_gas_mmbtu": {"type": "number", "minimum": 0}
                        }
                    }
                })
                .set_artifacts_dir("out/facility-analysis/")
                .add_validation_step("validate-input", "${inputs.facility_data}")
                .add_carbon_calculation_step(
                    "calculate-scope1",
                    "${steps.validate-input.outputs.natural_gas_data}",
                    "${vars.ng_emission_factor}",
                    scope=1
                )
                .add_carbon_calculation_step(
                    "calculate-scope2",
                    "${steps.validate-input.outputs.electricity_data}",
                    "${vars.grid_emission_factors}",
                    scope=2
                )
                .add_aggregation_step(
                    "total-emissions",
                    ["${steps.calculate-scope1.outputs}", "${steps.calculate-scope2.outputs}"],
                    depends_on=["calculate-scope1", "calculate-scope2"]
                )
                .add_report_generation_step(
                    "generate-report",
                    "${steps.total-emissions.outputs}",
                    "facility_summary",
                    "${artifacts_dir}/facility_report.html",
                    depends_on=["total-emissions"]
                )
                .set_outputs({
                    "total_emissions": "${steps.total-emissions.outputs.total}",
                    "scope1_emissions": "${steps.calculate-scope1.outputs.total}",
                    "scope2_emissions": "${steps.calculate-scope2.outputs.total}",
                    "report_path": "${steps.generate-report.outputs.report_path}"
                })
                .build())

    @staticmethod
    def create_portfolio_analysis(portfolio_name: str, facility_count: int) -> Pipeline:
        """Create a multi-facility portfolio analysis pipeline"""
        builder = PipelineBuilder(
            name=f"{portfolio_name.lower().replace(' ', '-')}-portfolio",
            version=2,
            description=f"Portfolio carbon analysis for {portfolio_name} ({facility_count} facilities)"
        )

        # Create data partition references for parallel processing
        partitions = [f"${{inputs.facilities.partition_{i}}}" for i in range(facility_count)]

        return (builder
                .set_variables({
                    "portfolio_name": portfolio_name,
                    "analysis_year": 2024,
                    "benchmark_enabled": facility_count > 5
                })
                .set_inputs({
                    "facilities": {
                        "type": "array",
                        "required": True,
                        "minItems": facility_count
                    },
                    "analysis_config": {
                        "type": "object",
                        "properties": {
                            "include_renewables": {"type": "boolean", "default": True},
                            "carbon_pricing": {"type": "boolean", "default": False}
                        }
                    }
                })
                .set_artifacts_dir(f"out/{portfolio_name.lower()}-portfolio/")
                .add_validation_step("validate-portfolio", "${inputs.facilities}")
                .add_parallel_processing_group(
                    "process-facility",
                    "CarbonAgent",
                    "calculate_facility_emissions",
                    partitions[:min(facility_count, 4)],  # Limit to 4 parallel processes
                    depends_on=["validate-portfolio"]
                )
                .add_renewable_analysis_step(
                    "analyze-renewables",
                    "${inputs.facilities.renewable_data}",
                    "${vars.regional_grid_factors}",
                    depends_on=["validate-portfolio"],
                    conditional_expression="${inputs.analysis_config.include_renewables} == true"
                )
                .add_aggregation_step(
                    "portfolio-totals",
                    [f"${{steps.process-facility_{i+1}.outputs}}" for i in range(min(facility_count, 4))],
                    depends_on=[f"process-facility_{i+1}" for i in range(min(facility_count, 4))]
                )
                .add_report_generation_step(
                    "executive-summary",
                    "${steps.portfolio-totals.outputs}",
                    "portfolio_executive",
                    "${artifacts_dir}/executive_summary.html",
                    depends_on=["portfolio-totals", "analyze-renewables"],
                    parallel=False
                )
                .add_report_generation_step(
                    "detailed-reports",
                    "${steps.portfolio-totals.outputs}",
                    "detailed_facility",
                    "${artifacts_dir}/detailed_reports/",
                    depends_on=["portfolio-totals"],
                    parallel=True
                )
                .set_outputs({
                    "portfolio_emissions": "${steps.portfolio-totals.outputs.total}",
                    "facility_count": "${len(inputs.facilities)}",
                    "renewable_offset": "${steps.analyze-renewables.outputs.total_offset}",
                    "executive_report": "${steps.executive-summary.outputs.report_path}",
                    "detailed_reports": "${steps.detailed-reports.outputs.report_path}"
                })
                .set_execution_config({
                    "timeout": "30m",
                    "max_parallel_steps": 6,
                    "retry_policy": {
                        "max_attempts": 3,
                        "backoff_strategy": "exponential"
                    }
                })
                .build())


def demonstrate_pipeline_creation():
    """Demonstrate different pipeline creation approaches"""
    logger.info("=== Demonstrating Pipeline Creation ===")

    # 1. Basic pipeline using builder pattern
    logger.info("Creating basic facility analysis pipeline...")
    basic_pipeline = PipelineTemplates.create_basic_facility_analysis("Sample Office Building")
    basic_pipeline.to_yaml("examples/sdk/generated_basic_pipeline.yaml")

    # 2. Complex portfolio pipeline
    logger.info("Creating portfolio analysis pipeline...")
    portfolio_pipeline = PipelineTemplates.create_portfolio_analysis("Corporate Portfolio", 8)
    portfolio_pipeline.to_yaml("examples/sdk/generated_portfolio_pipeline.yaml")

    # 3. Custom pipeline with advanced features
    logger.info("Creating custom advanced pipeline...")
    custom_builder = PipelineBuilder(
        name="advanced-custom-analysis",
        version="2.1.0",
        description="Advanced custom pipeline with error handling and conditional logic"
    )

    custom_pipeline = (custom_builder
                      .set_variables({
                          "error_threshold": 0.1,
                          "quality_gate_enabled": True,
                          "backup_calculation": True
                      })
                      .set_inputs({
                          "raw_data": {
                              "type": "object",
                              "required": True
                          },
                          "configuration": {
                              "type": "object",
                              "properties": {
                                  "strict_validation": {"type": "boolean", "default": True},
                                  "enable_fallbacks": {"type": "boolean", "default": True}
                              }
                          }
                      })
                      .add_validation_step(
                          "primary-validation",
                          "${inputs.raw_data}",
                          ["completeness", "accuracy", "consistency"]
                      )
                      .add_carbon_calculation_step(
                          "primary-calculation",
                          "${steps.primary-validation.outputs}",
                          "${vars.emission_factors}",
                          depends_on=["primary-validation"]
                      )
                      .build())

    # Add conditional fallback step manually
    fallback_step = PipelineStep(
        name="fallback-calculation",
        agent="CarbonAgent",
        action="conservative_estimate",
        inputs={"raw_data": "${inputs.raw_data}"},
        conditional={
            "condition": "${steps.primary-calculation.status} == 'failed'",
            "on_skip": {"fallback_used": False}
        },
        error_handling={
            "on_failure": "continue",
            "default_outputs": {"emissions": 0, "quality": "failed"}
        }
    )
    custom_pipeline.add_step(fallback_step)

    custom_pipeline.to_yaml("examples/sdk/generated_custom_pipeline.yaml")

    # 4. Demonstrate pipeline modification
    logger.info("Demonstrating pipeline modification...")
    modified_pipeline = modify_existing_pipeline(basic_pipeline)
    modified_pipeline.to_yaml("examples/sdk/generated_modified_pipeline.yaml")

    logger.info("Pipeline creation demonstration completed!")

    return {
        "basic": basic_pipeline,
        "portfolio": portfolio_pipeline,
        "custom": custom_pipeline,
        "modified": modified_pipeline
    }


def modify_existing_pipeline(original_pipeline: Pipeline) -> Pipeline:
    """Demonstrate how to modify an existing pipeline"""
    logger.info("Modifying existing pipeline...")

    # Create a copy and modify it
    modified = Pipeline(
        name=f"{original_pipeline.name}-enhanced",
        version=f"{original_pipeline.version}.1",
        description=f"{original_pipeline.description} (Enhanced)",
        vars=original_pipeline.vars.copy() if original_pipeline.vars else {},
        inputs=original_pipeline.inputs.copy() if original_pipeline.inputs else {},
        artifacts_dir=original_pipeline.artifacts_dir,
        steps=original_pipeline.steps.copy() if original_pipeline.steps else [],
        outputs=original_pipeline.outputs.copy() if original_pipeline.outputs else {}
    )

    # Add new variables
    if not modified.vars:
        modified.vars = {}
    modified.vars["enhanced_analytics"] = True
    modified.vars["benchmark_analysis"] = True

    # Add new step for benchmarking
    benchmark_step = PipelineStep(
        name="benchmark-analysis",
        agent="CarbonAgent",
        action="industry_benchmark",
        inputs={
            "facility_emissions": "${steps.total-emissions.outputs}",
            "facility_type": "${inputs.facility_data.type}",
            "region": "${inputs.facility_data.location}"
        },
        depends_on=["total-emissions"],
        error_handling={
            "on_failure": "continue",
            "default_outputs": {"benchmark_percentile": None}
        }
    )
    modified.add_step(benchmark_step)

    # Update outputs to include benchmark results
    if not modified.outputs:
        modified.outputs = {}
    modified.outputs["benchmark_percentile"] = "${steps.benchmark-analysis.outputs.percentile}"
    modified.outputs["industry_comparison"] = "${steps.benchmark-analysis.outputs.comparison}"

    return modified


def create_dynamic_pipeline_from_config(config: Dict[str, Any]) -> Pipeline:
    """Create a pipeline dynamically based on configuration"""
    logger.info("Creating dynamic pipeline from configuration...")

    builder = PipelineBuilder(
        name=config.get("name", "dynamic-pipeline"),
        version=config.get("version", 1),
        description=config.get("description", "Dynamically generated pipeline")
    )

    # Set variables if provided
    if "variables" in config:
        builder.set_variables(config["variables"])

    # Set inputs if provided
    if "inputs" in config:
        builder.set_inputs(config["inputs"])

    # Add steps based on configuration
    steps_config = config.get("steps", [])
    for step_config in steps_config:
        step_type = step_config.get("type")

        if step_type == "validation":
            builder.add_validation_step(
                step_config["name"],
                step_config["data_source"],
                step_config.get("rules"),
                step_config.get("depends_on")
            )
        elif step_type == "carbon_calculation":
            builder.add_carbon_calculation_step(
                step_config["name"],
                step_config["energy_data"],
                step_config["emission_factors"],
                step_config.get("scope"),
                step_config.get("depends_on"),
                step_config.get("parallel", False)
            )
        elif step_type == "report":
            builder.add_report_generation_step(
                step_config["name"],
                step_config["data_source"],
                step_config["template"],
                step_config["output_path"],
                step_config.get("depends_on"),
                step_config.get("parallel", False)
            )

    # Set outputs if provided
    if "outputs" in config:
        builder.set_outputs(config["outputs"])

    return builder.build()


def main():
    """Main function to run pipeline creation examples"""
    logger.info("Starting GreenLang SDK Pipeline Creation Examples...")

    try:
        # Create example pipelines
        pipelines = demonstrate_pipeline_creation()

        # Demonstrate dynamic pipeline creation
        dynamic_config = {
            "name": "dynamic-carbon-analysis",
            "version": "1.0.0",
            "description": "Dynamically generated carbon analysis pipeline",
            "variables": {
                "analysis_year": 2024,
                "precision": 3
            },
            "inputs": {
                "facility_data": {
                    "type": "object",
                    "required": True
                }
            },
            "steps": [
                {
                    "type": "validation",
                    "name": "validate-inputs",
                    "data_source": "${inputs.facility_data}"
                },
                {
                    "type": "carbon_calculation",
                    "name": "calculate-emissions",
                    "energy_data": "${steps.validate-inputs.outputs}",
                    "emission_factors": "${vars.emission_factors}",
                    "depends_on": ["validate-inputs"]
                },
                {
                    "type": "report",
                    "name": "generate-summary",
                    "data_source": "${steps.calculate-emissions.outputs}",
                    "template": "summary_report",
                    "output_path": "out/dynamic_report.html",
                    "depends_on": ["calculate-emissions"]
                }
            ],
            "outputs": {
                "total_emissions": "${steps.calculate-emissions.outputs.total}",
                "report_path": "${steps.generate-summary.outputs.report_path}"
            }
        }

        dynamic_pipeline = create_dynamic_pipeline_from_config(dynamic_config)
        dynamic_pipeline.to_yaml("examples/sdk/generated_dynamic_pipeline.yaml")

        logger.info("All pipeline creation examples completed successfully!")

        return pipelines

    except Exception as e:
        logger.error(f"Pipeline creation examples failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()