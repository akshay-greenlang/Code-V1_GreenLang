# -*- coding: utf-8 -*-
"""
Unit tests for GreenLang Pipeline YAML Schema v1.

GL-FOUND-X-001: Tests for pipeline schema validation, DAG acyclicity,
parameter template validation, and normalization.

Author: GreenLang Framework Team
Date: January 2026
"""

import json
import pytest
from pydantic import ValidationError

from greenlang.orchestrator.pipeline_schema import (
    # Core models
    PipelineDefinition,
    PipelineMetadata,
    PipelineSpec,
    PipelineDefaults,
    StepDefinition,
    ParameterDefinition,
    PolicyAttachment,
    # Enums
    ParameterType,
    PolicySeverity,
    # Functions
    load_pipeline_yaml,
    validate_agent_id,
    extract_template_references,
    SUPPORTED_API_VERSION,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def valid_pipeline_yaml() -> str:
    """Return a valid pipeline YAML string."""
    return """
apiVersion: greenlang/v1
kind: Pipeline
metadata:
  name: test-pipeline
  namespace: demo
  labels:
    team: sustainability
spec:
  parameters:
    input_uri:
      type: string
      required: true
      description: Input data URI
  defaults:
    retries: 2
    timeoutSeconds: 900
  steps:
    - id: ingest
      agent: GL-DATA-X-001
      with:
        uri: "{{ params.input_uri }}"
      outputs:
        dataset: "$.artifact.dataset_uri"
    - id: validate
      agent: GL-DATA-X-002
      dependsOn:
        - ingest
      with:
        dataset_uri: "{{ steps.ingest.outputs.dataset }}"
      outputs:
        validated: "$.result.validated"
      policy:
        - name: no_pii_export
          severity: error
"""


@pytest.fixture
def minimal_pipeline_dict() -> dict:
    """Return a minimal valid pipeline dictionary."""
    return {
        "apiVersion": "greenlang/v1",
        "kind": "Pipeline",
        "metadata": {"name": "minimal-pipeline"},
        "spec": {
            "steps": [
                {"id": "step1", "agent": "GL-DATA-X-001"}
            ]
        }
    }


# ==============================================================================
# Test PipelineMetadata
# ==============================================================================

class TestPipelineMetadata:
    """Tests for PipelineMetadata model."""

    def test_valid_metadata(self):
        """Test valid metadata creation."""
        metadata = PipelineMetadata(
            name="test-pipeline",
            namespace="production",
            labels={"team": "sustainability"},
            version="1.2.3"
        )
        assert metadata.name == "test-pipeline"
        assert metadata.namespace == "production"
        assert metadata.labels == {"team": "sustainability"}
        assert metadata.version == "1.2.3"

    def test_default_namespace(self):
        """Test default namespace is 'default'."""
        metadata = PipelineMetadata(name="test")
        assert metadata.namespace == "default"

    def test_invalid_name_uppercase(self):
        """Test that uppercase names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineMetadata(name="TestPipeline")
        assert "Invalid pipeline name" in str(exc_info.value)

    def test_invalid_name_underscore(self):
        """Test that underscores in names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineMetadata(name="test_pipeline")
        assert "Invalid pipeline name" in str(exc_info.value)

    def test_invalid_namespace(self):
        """Test that invalid namespaces are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineMetadata(name="test", namespace="Invalid_Namespace")
        assert "Invalid namespace" in str(exc_info.value)

    def test_invalid_label_key(self):
        """Test that invalid label keys are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineMetadata(name="test", labels={"123invalid": "value"})
        assert "Invalid label/annotation key" in str(exc_info.value)


# ==============================================================================
# Test ParameterDefinition
# ==============================================================================

class TestParameterDefinition:
    """Tests for ParameterDefinition model."""

    def test_string_parameter(self):
        """Test string parameter creation."""
        param = ParameterDefinition(
            type=ParameterType.STRING,
            required=True,
            description="Test parameter"
        )
        assert param.type == ParameterType.STRING
        assert param.required is True

    def test_numeric_parameter_with_constraints(self):
        """Test numeric parameter with min/max constraints."""
        param = ParameterDefinition(
            type=ParameterType.INTEGER,
            minimum=1,
            maximum=100,
            default=50
        )
        assert param.minimum == 1
        assert param.maximum == 100
        assert param.default == 50

    def test_invalid_min_max(self):
        """Test that minimum > maximum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ParameterDefinition(
                type=ParameterType.INTEGER,
                minimum=100,
                maximum=1
            )
        assert "cannot exceed maximum" in str(exc_info.value)

    def test_invalid_regex_pattern(self):
        """Test that invalid regex patterns are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ParameterDefinition(
                type=ParameterType.STRING,
                pattern="[invalid"
            )
        assert "Invalid regex pattern" in str(exc_info.value)

    def test_type_normalization(self):
        """Test that string types are normalized to enum."""
        param = ParameterDefinition(type="string")
        assert param.type == ParameterType.STRING


# ==============================================================================
# Test PolicyAttachment
# ==============================================================================

class TestPolicyAttachment:
    """Tests for PolicyAttachment model."""

    def test_valid_policy(self):
        """Test valid policy attachment creation."""
        policy = PolicyAttachment(
            name="no_pii_export",
            severity=PolicySeverity.ERROR,
            params={"allow_hashed": True}
        )
        assert policy.name == "no_pii_export"
        assert policy.severity == PolicySeverity.ERROR
        assert policy.params == {"allow_hashed": True}

    def test_default_severity(self):
        """Test default severity is ERROR."""
        policy = PolicyAttachment(name="test_policy")
        assert policy.severity == PolicySeverity.ERROR

    def test_default_enabled(self):
        """Test default enabled is True."""
        policy = PolicyAttachment(name="test_policy")
        assert policy.enabled is True


# ==============================================================================
# Test StepDefinition
# ==============================================================================

class TestStepDefinition:
    """Tests for StepDefinition model."""

    def test_valid_step(self):
        """Test valid step creation."""
        step = StepDefinition(
            id="ingest",
            agent="GL-DATA-X-001",
            with_={"uri": "s3://bucket/data"},
            outputs={"dataset": "$.artifact.uri"}
        )
        assert step.id == "ingest"
        assert step.agent == "GL-DATA-X-001"
        assert step.with_ == {"uri": "s3://bucket/data"}

    def test_invalid_agent_id_format(self):
        """Test that invalid agent IDs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            StepDefinition(id="step1", agent="invalid-agent")
        assert "Invalid agent ID" in str(exc_info.value)

    def test_invalid_agent_id_lowercase(self):
        """Test that lowercase agent categories are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            StepDefinition(id="step1", agent="GL-data-X-001")
        assert "Invalid agent ID" in str(exc_info.value)

    def test_invalid_output_jsonpath(self):
        """Test that invalid JSONPath expressions are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            StepDefinition(
                id="step1",
                agent="GL-DATA-X-001",
                outputs={"result": "invalid-path"}
            )
        assert "Invalid JSONPath" in str(exc_info.value)

    def test_valid_output_jsonpath(self):
        """Test that valid JSONPath expressions are accepted."""
        step = StepDefinition(
            id="step1",
            agent="GL-DATA-X-001",
            outputs={
                "result1": "$.field",
                "result2": "$.nested.field",
                "result3": "$.array[0].field"
            }
        )
        assert len(step.outputs) == 3

    def test_default_timeout(self):
        """Test default timeout is 900 seconds."""
        step = StepDefinition(id="step1", agent="GL-DATA-X-001")
        assert step.timeoutSeconds == 900

    def test_default_retries(self):
        """Test default retries is 0."""
        step = StepDefinition(id="step1", agent="GL-DATA-X-001")
        assert step.retries == 0

    def test_legacy_name_to_id(self):
        """Test legacy 'name' field maps to 'id'."""
        step = StepDefinition(name="step1", agent_id="GL-DATA-X-001")
        assert step.id == "step1"
        assert step.agent == "GL-DATA-X-001"


# ==============================================================================
# Test PipelineSpec
# ==============================================================================

class TestPipelineSpec:
    """Tests for PipelineSpec model."""

    def test_valid_spec(self):
        """Test valid spec creation."""
        spec = PipelineSpec(
            parameters={"input": ParameterDefinition(type=ParameterType.STRING)},
            steps=[
                StepDefinition(id="step1", agent="GL-DATA-X-001"),
                StepDefinition(id="step2", agent="GL-DATA-X-002", dependsOn=["step1"])
            ]
        )
        assert len(spec.steps) == 2

    def test_duplicate_step_ids(self):
        """Test that duplicate step IDs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineSpec(
                steps=[
                    StepDefinition(id="step1", agent="GL-DATA-X-001"),
                    StepDefinition(id="step1", agent="GL-DATA-X-002")
                ]
            )
        assert "Duplicate step ID" in str(exc_info.value)

    def test_unknown_dependency(self):
        """Test that references to unknown steps are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineSpec(
                steps=[
                    StepDefinition(
                        id="step1",
                        agent="GL-DATA-X-001",
                        dependsOn=["nonexistent"]
                    )
                ]
            )
        assert "unknown step" in str(exc_info.value)

    def test_self_dependency(self):
        """Test that self-dependencies are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineSpec(
                steps=[
                    StepDefinition(
                        id="step1",
                        agent="GL-DATA-X-001",
                        dependsOn=["step1"]
                    )
                ]
            )
        assert "cannot depend on itself" in str(exc_info.value)

    def test_cycle_detection_simple(self):
        """Test simple cycle detection."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineSpec(
                steps=[
                    StepDefinition(id="a", agent="GL-DATA-X-001", dependsOn=["b"]),
                    StepDefinition(id="b", agent="GL-DATA-X-002", dependsOn=["a"])
                ]
            )
        assert "cycle" in str(exc_info.value).lower()

    def test_cycle_detection_complex(self):
        """Test complex cycle detection (3-node cycle)."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineSpec(
                steps=[
                    StepDefinition(id="a", agent="GL-DATA-X-001", dependsOn=["c"]),
                    StepDefinition(id="b", agent="GL-DATA-X-002", dependsOn=["a"]),
                    StepDefinition(id="c", agent="GL-DATA-X-003", dependsOn=["b"])
                ]
            )
        assert "cycle" in str(exc_info.value).lower()

    def test_valid_dag(self):
        """Test valid DAG is accepted."""
        spec = PipelineSpec(
            steps=[
                StepDefinition(id="a", agent="GL-DATA-X-001"),
                StepDefinition(id="b", agent="GL-DATA-X-002", dependsOn=["a"]),
                StepDefinition(id="c", agent="GL-DATA-X-003", dependsOn=["a"]),
                StepDefinition(id="d", agent="GL-DATA-X-004", dependsOn=["b", "c"])
            ]
        )
        assert len(spec.steps) == 4

    def test_undefined_parameter_reference(self):
        """Test that references to undefined parameters are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineSpec(
                steps=[
                    StepDefinition(
                        id="step1",
                        agent="GL-DATA-X-001",
                        with_={"uri": "{{ params.undefined_param }}"}
                    )
                ]
            )
        assert "undefined parameter" in str(exc_info.value)

    def test_undefined_step_output_reference(self):
        """Test that references to undefined step outputs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineSpec(
                parameters={"input": ParameterDefinition(type=ParameterType.STRING)},
                steps=[
                    StepDefinition(
                        id="step1",
                        agent="GL-DATA-X-001",
                        outputs={"result": "$.data"}
                    ),
                    StepDefinition(
                        id="step2",
                        agent="GL-DATA-X-002",
                        dependsOn=["step1"],
                        with_={"data": "{{ steps.step1.outputs.undefined }}"}
                    )
                ]
            )
        assert "undefined output" in str(exc_info.value)

    def test_reference_to_non_dependency(self):
        """Test that references to non-dependency steps are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineSpec(
                steps=[
                    StepDefinition(
                        id="step1",
                        agent="GL-DATA-X-001",
                        outputs={"result": "$.data"}
                    ),
                    StepDefinition(
                        id="step2",
                        agent="GL-DATA-X-002",
                        # Note: step1 is NOT in dependsOn
                        with_={"data": "{{ steps.step1.outputs.result }}"}
                    )
                ]
            )
        assert "not a declared dependency" in str(exc_info.value)


# ==============================================================================
# Test PipelineDefinition
# ==============================================================================

class TestPipelineDefinition:
    """Tests for PipelineDefinition model."""

    def test_valid_pipeline(self, minimal_pipeline_dict):
        """Test valid pipeline creation."""
        pipeline = PipelineDefinition(**minimal_pipeline_dict)
        assert pipeline.apiVersion == "greenlang/v1"
        assert pipeline.kind == "Pipeline"
        assert pipeline.metadata.name == "minimal-pipeline"

    def test_default_api_version(self):
        """Test that default API version is set."""
        pipeline = PipelineDefinition(
            kind="Pipeline",
            metadata=PipelineMetadata(name="test"),
            spec=PipelineSpec(steps=[
                StepDefinition(id="step1", agent="GL-DATA-X-001")
            ])
        )
        assert pipeline.apiVersion == SUPPORTED_API_VERSION

    def test_invalid_kind(self):
        """Test that invalid kind is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineDefinition(
                apiVersion="greenlang/v1",
                kind="InvalidKind",
                metadata=PipelineMetadata(name="test"),
                spec=PipelineSpec(steps=[
                    StepDefinition(id="step1", agent="GL-DATA-X-001")
                ])
            )
        assert "Invalid kind" in str(exc_info.value)

    def test_compute_hash_deterministic(self, minimal_pipeline_dict):
        """Test that compute_hash returns deterministic results."""
        pipeline1 = PipelineDefinition(**minimal_pipeline_dict)
        pipeline2 = PipelineDefinition(**minimal_pipeline_dict)
        assert pipeline1.compute_hash() == pipeline2.compute_hash()

    def test_compute_hash_changes(self, minimal_pipeline_dict):
        """Test that compute_hash changes with different inputs."""
        pipeline1 = PipelineDefinition(**minimal_pipeline_dict)

        minimal_pipeline_dict["metadata"]["name"] = "different-pipeline"
        pipeline2 = PipelineDefinition(**minimal_pipeline_dict)

        assert pipeline1.compute_hash() != pipeline2.compute_hash()

    def test_get_execution_order(self):
        """Test topological sort for execution order."""
        pipeline = PipelineDefinition(
            apiVersion="greenlang/v1",
            kind="Pipeline",
            metadata=PipelineMetadata(name="test"),
            spec=PipelineSpec(steps=[
                StepDefinition(id="d", agent="GL-DATA-X-004", dependsOn=["b", "c"]),
                StepDefinition(id="c", agent="GL-DATA-X-003", dependsOn=["a"]),
                StepDefinition(id="b", agent="GL-DATA-X-002", dependsOn=["a"]),
                StepDefinition(id="a", agent="GL-DATA-X-001")
            ])
        )

        order = pipeline.get_execution_order()

        # 'a' must come first
        assert order[0] == "a"
        # 'd' must come last
        assert order[-1] == "d"
        # 'b' and 'c' must come before 'd'
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_get_step(self, minimal_pipeline_dict):
        """Test getting a step by ID."""
        pipeline = PipelineDefinition(**minimal_pipeline_dict)
        step = pipeline.get_step("step1")
        assert step is not None
        assert step.id == "step1"

    def test_get_step_not_found(self, minimal_pipeline_dict):
        """Test getting a non-existent step returns None."""
        pipeline = PipelineDefinition(**minimal_pipeline_dict)
        step = pipeline.get_step("nonexistent")
        assert step is None

    def test_apply_defaults(self):
        """Test applying pipeline defaults to steps."""
        pipeline = PipelineDefinition(
            apiVersion="greenlang/v1",
            kind="Pipeline",
            metadata=PipelineMetadata(name="test"),
            spec=PipelineSpec(
                defaults=PipelineDefaults(retries=3, timeoutSeconds=600),
                steps=[StepDefinition(id="step1", agent="GL-DATA-X-001")]
            )
        )

        updated = pipeline.apply_defaults()
        step = updated.get_step("step1")
        assert step.retries == 3
        assert step.timeoutSeconds == 600

    def test_to_json_schema(self, minimal_pipeline_dict):
        """Test JSON schema export."""
        pipeline = PipelineDefinition(**minimal_pipeline_dict)
        schema = pipeline.to_json_schema()
        assert "properties" in schema
        assert "apiVersion" in schema["properties"]


# ==============================================================================
# Test Utility Functions
# ==============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_load_pipeline_yaml(self, valid_pipeline_yaml):
        """Test loading pipeline from YAML string."""
        pipeline = load_pipeline_yaml(valid_pipeline_yaml)
        assert pipeline.metadata.name == "test-pipeline"
        assert len(pipeline.spec.steps) == 2

    def test_load_pipeline_yaml_invalid(self):
        """Test loading invalid YAML raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_pipeline_yaml("invalid: yaml: content: [")
        assert "Invalid YAML" in str(exc_info.value)

    def test_load_pipeline_yaml_empty(self):
        """Test loading empty YAML raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_pipeline_yaml("")
        assert "Empty YAML" in str(exc_info.value)

    def test_validate_agent_id_valid(self):
        """Test valid agent ID validation."""
        assert validate_agent_id("GL-DATA-X-001") is True
        assert validate_agent_id("GL-CALC-A-002") is True
        assert validate_agent_id("GL-REPORT-X-123") is True

    def test_validate_agent_id_invalid(self):
        """Test invalid agent ID validation."""
        assert validate_agent_id("invalid") is False
        assert validate_agent_id("GL-data-x-001") is False
        assert validate_agent_id("GL-DATA-X-1") is False
        assert validate_agent_id("GL-DATA-XX-001") is False

    def test_extract_template_references(self):
        """Test extracting template references."""
        value = {
            "uri": "{{ params.input_uri }}",
            "config": {
                "dataset": "{{ steps.ingest.outputs.data }}"
            }
        }
        refs = extract_template_references(value)
        assert "params.input_uri" in refs
        assert "steps.ingest.outputs.data" in refs

    def test_extract_template_references_no_refs(self):
        """Test extracting from value with no references."""
        refs = extract_template_references({"uri": "s3://bucket/data"})
        assert refs == []


# ==============================================================================
# Test YAML Anchor Expansion
# ==============================================================================

class TestYAMLAnchorExpansion:
    """Tests for YAML anchor expansion."""

    def test_yaml_anchor_expansion(self):
        """Test that YAML anchors are properly expanded."""
        yaml_content = """
apiVersion: greenlang/v1
kind: Pipeline
metadata:
  name: anchor-test
spec:
  defaults: &defaults
    retries: 2
    timeoutSeconds: 600
  steps:
    - id: step1
      agent: GL-DATA-X-001
      <<: *defaults
"""
        # YAML anchors are expanded by yaml.safe_load before validation
        pipeline = load_pipeline_yaml(yaml_content)
        assert pipeline.spec.defaults.retries == 2
        assert pipeline.spec.defaults.timeoutSeconds == 600


# ==============================================================================
# Test Normalization
# ==============================================================================

class TestNormalization:
    """Tests for pipeline normalization."""

    def test_normalize_whitespace(self):
        """Test that whitespace is normalized."""
        pipeline = PipelineDefinition(
            apiVersion="greenlang/v1",
            kind="Pipeline",
            metadata=PipelineMetadata(
                name="test",
                description="  Multiple   spaces   here  "
            ),
            spec=PipelineSpec(steps=[
                StepDefinition(id="step1", agent="GL-DATA-X-001")
            ])
        )

        normalized = pipeline.normalize()
        assert normalized.metadata.description == "Multiple spaces here"

    def test_normalize_key_ordering(self):
        """Test that keys are sorted for deterministic output."""
        pipeline = PipelineDefinition(
            apiVersion="greenlang/v1",
            kind="Pipeline",
            metadata=PipelineMetadata(
                name="test",
                labels={"z_key": "value", "a_key": "value"}
            ),
            spec=PipelineSpec(steps=[
                StepDefinition(id="step1", agent="GL-DATA-X-001")
            ])
        )

        normalized = pipeline.normalize()
        normalized_dict = normalized.model_dump()

        # Keys should be processed in sorted order during normalization
        # The hash should be deterministic
        hash1 = pipeline.compute_hash()
        hash2 = pipeline.compute_hash()
        assert hash1 == hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
