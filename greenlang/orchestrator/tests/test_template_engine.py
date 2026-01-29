# -*- coding: utf-8 -*-
"""
Unit tests for GreenLang Pipeline Template Engine (FR-005).

Tests for template registration, parameter validation, expansion,
nested templates, and deterministic hashing.

Author: GreenLang Framework Team
Date: January 2026
"""

import hashlib
import json
import pytest
from pydantic import ValidationError

from greenlang.orchestrator.template_engine import (
    # Core models
    PipelineTemplate,
    TemplateParameter,
    TemplateStep,
    TemplateImport,
    ExpandedStep,
    TemplateExpansionResult,
    # Enums
    TemplateParameterType,
    TemplateStatus,
    # Registry and resolver
    TemplateRegistry,
    TemplateResolver,
    # Functions
    load_template_yaml,
    create_template_registry,
    # Constants
    MAX_TEMPLATE_NESTING_DEPTH,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def simple_template_yaml() -> str:
    """Return a simple template YAML string."""
    return """
name: simple-template
version: "1.0.0"
description: A simple test template
parameters:
  input_uri:
    type: string
    required: true
    description: Input data URI
  threshold:
    type: number
    default: 0.95
steps:
  - id: step1
    agent: GL-DATA-X-001
    in:
      uri: "{{ params.input_uri }}"
      threshold: "{{ params.threshold }}"
    outputs:
      result: "$.data.output"
"""


@pytest.fixture
def data_quality_template_yaml() -> str:
    """Return a data quality template YAML string."""
    return """
name: data-quality-checks
version: "1.0.0"
description: Standard data quality validation suite
parameters:
  dataset_uri:
    type: string
    required: true
  threshold:
    type: number
    default: 0.95
steps:
  - id: validate_schema
    agent: OPS.DATA.SchemaValidator
    in:
      uri: "{{ params.dataset_uri }}"
  - id: check_completeness
    agent: OPS.DATA.CompletenessCheck
    dependsOn:
      - validate_schema
    in:
      uri: "{{ params.dataset_uri }}"
      threshold: "{{ params.threshold }}"
"""


@pytest.fixture
def nested_template_yaml() -> str:
    """Return a template that uses another template."""
    return """
name: composite-template
version: "1.0.0"
description: Template that uses nested templates
parameters:
  data_uri:
    type: string
    required: true
steps:
  - id: quality_check
    template: dq.data-quality-checks
    templateParams:
      dataset_uri: "{{ params.data_uri }}"
      threshold: 0.99
"""


@pytest.fixture
def simple_template() -> PipelineTemplate:
    """Return a simple PipelineTemplate object."""
    return PipelineTemplate(
        name="test-template",
        version="1.0.0",
        description="Test template",
        parameters={
            "input_uri": TemplateParameter(
                type=TemplateParameterType.STRING,
                required=True,
                description="Input URI"
            ),
            "count": TemplateParameter(
                type=TemplateParameterType.INTEGER,
                default=10
            )
        },
        steps=[
            TemplateStep(
                id="step1",
                agent="GL-DATA-X-001",
                in_={"uri": "{{ params.input_uri }}", "count": "{{ params.count }}"},
                outputs={"result": "$.output"}
            )
        ]
    )


@pytest.fixture
def registry() -> TemplateRegistry:
    """Return an empty template registry."""
    return TemplateRegistry()


# ==============================================================================
# Test TemplateParameter
# ==============================================================================

class TestTemplateParameter:
    """Tests for TemplateParameter model."""

    def test_string_parameter(self):
        """Test string parameter creation."""
        param = TemplateParameter(
            type=TemplateParameterType.STRING,
            required=True,
            description="Test parameter"
        )
        assert param.type == TemplateParameterType.STRING
        assert param.required is True

    def test_numeric_parameter_with_constraints(self):
        """Test numeric parameter with min/max constraints."""
        param = TemplateParameter(
            type=TemplateParameterType.NUMBER,
            minimum=0.0,
            maximum=1.0,
            default=0.5
        )
        assert param.minimum == 0.0
        assert param.maximum == 1.0
        assert param.default == 0.5

    def test_invalid_min_max(self):
        """Test that minimum > maximum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TemplateParameter(
                type=TemplateParameterType.INTEGER,
                minimum=100,
                maximum=1
            )
        assert "cannot exceed maximum" in str(exc_info.value)

    def test_type_normalization(self):
        """Test that string types are normalized to enum."""
        param = TemplateParameter(type="string")
        assert param.type == TemplateParameterType.STRING

        param = TemplateParameter(type="number")
        assert param.type == TemplateParameterType.NUMBER

    def test_validate_value_required(self):
        """Test required parameter validation."""
        param = TemplateParameter(
            type=TemplateParameterType.STRING,
            required=True
        )
        is_valid, error = param.validate_value(None, "test_param")
        assert is_valid is False
        assert "Required parameter" in error

    def test_validate_value_type(self):
        """Test type validation."""
        param = TemplateParameter(type=TemplateParameterType.INTEGER)
        is_valid, error = param.validate_value("not_an_int", "test_param")
        assert is_valid is False
        assert "must be of type" in error

    def test_validate_value_enum(self):
        """Test enum validation."""
        param = TemplateParameter(
            type=TemplateParameterType.STRING,
            enum=["a", "b", "c"]
        )
        is_valid, error = param.validate_value("d", "test_param")
        assert is_valid is False
        assert "must be one of" in error

    def test_validate_value_range(self):
        """Test range validation."""
        param = TemplateParameter(
            type=TemplateParameterType.NUMBER,
            minimum=0.0,
            maximum=1.0
        )
        is_valid, error = param.validate_value(1.5, "test_param")
        assert is_valid is False
        assert "must be <=" in error


# ==============================================================================
# Test TemplateStep
# ==============================================================================

class TestTemplateStep:
    """Tests for TemplateStep model."""

    def test_valid_agent_step(self):
        """Test valid agent step creation."""
        step = TemplateStep(
            id="test_step",
            agent="GL-DATA-X-001",
            in_={"uri": "s3://bucket/data"}
        )
        assert step.id == "test_step"
        assert step.agent == "GL-DATA-X-001"

    def test_valid_template_step(self):
        """Test valid template reference step creation."""
        step = TemplateStep(
            id="test_step",
            template="dq.standard_checks",
            templateParams={"threshold": 0.95}
        )
        assert step.id == "test_step"
        assert step.template == "dq.standard_checks"

    def test_missing_agent_and_template(self):
        """Test that step requires agent or template."""
        with pytest.raises(ValidationError) as exc_info:
            TemplateStep(id="test_step")
        assert "must have either 'agent' or 'template'" in str(exc_info.value)

    def test_both_agent_and_template(self):
        """Test that step cannot have both agent and template."""
        with pytest.raises(ValidationError) as exc_info:
            TemplateStep(
                id="test_step",
                agent="GL-DATA-X-001",
                template="dq.standard_checks"
            )
        assert "cannot have both" in str(exc_info.value)


# ==============================================================================
# Test PipelineTemplate
# ==============================================================================

class TestPipelineTemplate:
    """Tests for PipelineTemplate model."""

    def test_valid_template(self, simple_template_yaml):
        """Test valid template creation from YAML."""
        template = load_template_yaml(simple_template_yaml)
        assert template.name == "simple-template"
        assert template.version == "1.0.0"
        assert len(template.steps) == 1

    def test_duplicate_step_ids(self):
        """Test that duplicate step IDs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineTemplate(
                name="test",
                version="1.0.0",
                steps=[
                    TemplateStep(id="step1", agent="GL-DATA-X-001"),
                    TemplateStep(id="step1", agent="GL-DATA-X-002")
                ]
            )
        assert "Duplicate step ID" in str(exc_info.value)

    def test_unknown_dependency(self):
        """Test that unknown dependencies are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineTemplate(
                name="test",
                version="1.0.0",
                steps=[
                    TemplateStep(
                        id="step1",
                        agent="GL-DATA-X-001",
                        dependsOn=["nonexistent"]
                    )
                ]
            )
        assert "unknown step" in str(exc_info.value)

    def test_undefined_parameter_reference(self):
        """Test that undefined parameter references are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineTemplate(
                name="test",
                version="1.0.0",
                steps=[
                    TemplateStep(
                        id="step1",
                        agent="GL-DATA-X-001",
                        in_={"uri": "{{ params.undefined }}"}
                    )
                ]
            )
        assert "undefined parameter" in str(exc_info.value)

    def test_content_hash_deterministic(self, simple_template):
        """Test that content hash is deterministic."""
        hash1 = simple_template.compute_content_hash()
        hash2 = simple_template.compute_content_hash()
        assert hash1 == hash2

    def test_content_hash_changes(self, simple_template_yaml):
        """Test that content hash changes with different content."""
        template1 = load_template_yaml(simple_template_yaml)
        template2 = load_template_yaml(simple_template_yaml.replace("1.0.0", "2.0.0"))
        assert template1.compute_content_hash() != template2.compute_content_hash()

    def test_invalid_version_format(self):
        """Test that invalid version formats are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PipelineTemplate(
                name="test",
                version="invalid",
                steps=[TemplateStep(id="step1", agent="GL-DATA-X-001")]
            )
        assert "Invalid version" in str(exc_info.value)


# ==============================================================================
# Test TemplateImport
# ==============================================================================

class TestTemplateImport:
    """Tests for TemplateImport model."""

    def test_valid_import(self):
        """Test valid import creation."""
        imp = TemplateImport(
            name="data-quality-checks",
            version="1.0.0",
            as_="dq"
        )
        assert imp.name == "data-quality-checks"
        assert imp.version == "1.0.0"
        assert imp.as_ == "dq"

    def test_invalid_alias(self):
        """Test that invalid aliases are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TemplateImport(
                name="test",
                version="1.0.0",
                as_="123invalid"
            )
        assert "Invalid alias" in str(exc_info.value)


# ==============================================================================
# Test TemplateRegistry
# ==============================================================================

class TestTemplateRegistry:
    """Tests for TemplateRegistry class."""

    def test_register_template(self, registry, simple_template):
        """Test registering a template."""
        registry.register(simple_template)
        assert registry.exists("test-template")
        assert registry.exists("test-template", "1.0.0")

    def test_register_duplicate_version(self, registry, simple_template):
        """Test that duplicate versions are rejected."""
        registry.register(simple_template)
        with pytest.raises(ValueError) as exc_info:
            registry.register(simple_template)
        assert "already registered" in str(exc_info.value)

    def test_register_from_yaml(self, registry, simple_template_yaml):
        """Test registering from YAML content."""
        template = registry.register_from_yaml(simple_template_yaml)
        assert template.name == "simple-template"
        assert registry.exists("simple-template", "1.0.0")

    def test_get_template(self, registry, simple_template):
        """Test getting a template by name and version."""
        registry.register(simple_template)
        retrieved = registry.get("test-template", "1.0.0")
        assert retrieved is not None
        assert retrieved.name == "test-template"

    def test_get_latest_version(self, registry):
        """Test getting latest version."""
        registry.register(PipelineTemplate(
            name="test",
            version="1.0.0",
            steps=[TemplateStep(id="step1", agent="GL-DATA-X-001")]
        ))
        registry.register(PipelineTemplate(
            name="test",
            version="2.0.0",
            steps=[TemplateStep(id="step1", agent="GL-DATA-X-001")]
        ))
        registry.register(PipelineTemplate(
            name="test",
            version="1.5.0",
            steps=[TemplateStep(id="step1", agent="GL-DATA-X-001")]
        ))

        latest = registry.get("test")
        assert latest.version == "2.0.0"

    def test_get_versions(self, registry):
        """Test getting all versions of a template."""
        for version in ["1.0.0", "2.0.0", "1.5.0"]:
            registry.register(PipelineTemplate(
                name="test",
                version=version,
                steps=[TemplateStep(id="step1", agent="GL-DATA-X-001")]
            ))

        versions = registry.get_versions("test")
        assert versions == ["2.0.0", "1.5.0", "1.0.0"]

    def test_list_templates(self, registry, simple_template):
        """Test listing all templates."""
        registry.register(simple_template)
        templates = registry.list_templates()
        assert "test-template" in templates

    def test_unregister_template(self, registry, simple_template):
        """Test unregistering a template."""
        registry.register(simple_template)
        assert registry.exists("test-template")

        result = registry.unregister("test-template", "1.0.0")
        assert result is True
        assert not registry.exists("test-template")

    def test_clear_registry(self, registry, simple_template):
        """Test clearing all templates."""
        registry.register(simple_template)
        registry.clear()
        assert registry.template_count == 0


# ==============================================================================
# Test TemplateResolver
# ==============================================================================

class TestTemplateResolver:
    """Tests for TemplateResolver class."""

    def test_resolve_imports(self, registry, simple_template):
        """Test resolving template imports."""
        registry.register(simple_template)
        resolver = TemplateResolver(registry)

        imports = [
            TemplateImport(name="test-template", version="1.0.0", as_="test")
        ]
        resolved = resolver.resolve_imports(imports)

        assert "test" in resolved
        assert resolved["test"].name == "test-template"

    def test_resolve_imports_not_found(self, registry):
        """Test that missing templates raise error."""
        resolver = TemplateResolver(registry)
        imports = [
            TemplateImport(name="nonexistent", version="1.0.0", as_="test")
        ]

        with pytest.raises(ValueError) as exc_info:
            resolver.resolve_imports(imports)
        assert "Template not found" in str(exc_info.value)

    def test_expand_template_simple(self, registry, simple_template):
        """Test simple template expansion."""
        registry.register(simple_template)
        resolver = TemplateResolver(registry)

        result = resolver.expand_template(
            template=simple_template,
            params={"input_uri": "s3://bucket/data"}
        )

        assert len(result.steps) == 1
        assert result.steps[0].id == "step1"
        assert result.steps[0].in_["uri"] == "s3://bucket/data"
        assert result.steps[0].in_["count"] == 10  # default value

    def test_expand_template_with_prefix(self, registry, simple_template):
        """Test template expansion with step prefix."""
        registry.register(simple_template)
        resolver = TemplateResolver(registry)

        result = resolver.expand_template(
            template=simple_template,
            params={"input_uri": "s3://bucket/data"},
            step_prefix="quality_"
        )

        assert result.steps[0].id == "quality_step1"

    def test_expand_template_deterministic(self, registry, simple_template):
        """Test that expansion is deterministic (same template + params = same hash)."""
        registry.register(simple_template)
        resolver = TemplateResolver(registry)

        params = {"input_uri": "s3://bucket/data"}

        result1 = resolver.expand_template(template=simple_template, params=params)
        resolver.clear_cache()
        result2 = resolver.expand_template(template=simple_template, params=params)

        assert result1.content_hash == result2.content_hash

    def test_expand_template_different_params(self, registry, simple_template):
        """Test that different params produce different hashes."""
        registry.register(simple_template)
        resolver = TemplateResolver(registry)

        result1 = resolver.expand_template(
            template=simple_template,
            params={"input_uri": "s3://bucket/data1"}
        )
        result2 = resolver.expand_template(
            template=simple_template,
            params={"input_uri": "s3://bucket/data2"}
        )

        assert result1.content_hash != result2.content_hash

    def test_validate_parameters_missing_required(self, registry, simple_template):
        """Test validation catches missing required parameters."""
        resolver = TemplateResolver(registry)
        errors = resolver.validate_parameters(simple_template, {})
        assert any("Required parameter" in e for e in errors)

    def test_validate_parameters_unknown(self, registry, simple_template):
        """Test validation catches unknown parameters."""
        resolver = TemplateResolver(registry)
        errors = resolver.validate_parameters(
            simple_template,
            {"input_uri": "test", "unknown_param": "value"}
        )
        assert any("Unknown parameter" in e for e in errors)

    def test_expand_template_tracks_templates_used(self, registry, simple_template):
        """Test that expansion tracks which templates were used."""
        registry.register(simple_template)
        resolver = TemplateResolver(registry)

        result = resolver.expand_template(
            template=simple_template,
            params={"input_uri": "s3://bucket/data"}
        )

        assert simple_template.name in result.templates_used
        assert result.templates_used[simple_template.name] == simple_template.version

    def test_expand_template_source_tracking(self, registry, simple_template):
        """Test that expanded steps track their source template."""
        registry.register(simple_template)
        resolver = TemplateResolver(registry)

        result = resolver.expand_template(
            template=simple_template,
            params={"input_uri": "s3://bucket/data"}
        )

        assert result.steps[0].source_template == simple_template.name
        assert result.steps[0].source_template_version == simple_template.version


# ==============================================================================
# Test Nested Template Expansion
# ==============================================================================

class TestNestedTemplateExpansion:
    """Tests for nested template expansion."""

    def test_nested_template_expansion(
        self, registry, data_quality_template_yaml, nested_template_yaml
    ):
        """Test expanding a template that uses another template."""
        # Register the base template
        dq_template = registry.register_from_yaml(data_quality_template_yaml)

        # Register the composite template
        composite = registry.register_from_yaml(nested_template_yaml)

        resolver = TemplateResolver(registry)

        # Resolve imports
        resolved_templates = {"dq": dq_template}

        result = resolver.expand_template(
            template=composite,
            params={"data_uri": "s3://bucket/data"},
            resolved_templates=resolved_templates
        )

        # Should have steps from the nested template
        assert len(result.steps) >= 2
        assert "data-quality-checks" in result.templates_used

    def test_cycle_detection(self, registry):
        """Test that circular template references are detected."""
        # Create template A that references B
        template_a = PipelineTemplate(
            name="template-a",
            version="1.0.0",
            steps=[
                TemplateStep(
                    id="step1",
                    template="b.template-b",
                    templateParams={}
                )
            ]
        )

        # Create template B that references A
        template_b = PipelineTemplate(
            name="template-b",
            version="1.0.0",
            steps=[
                TemplateStep(
                    id="step1",
                    template="a.template-a",
                    templateParams={}
                )
            ]
        )

        registry.register(template_a)
        registry.register(template_b)
        resolver = TemplateResolver(registry)

        resolved = {"a": template_a, "b": template_b}

        with pytest.raises(ValueError) as exc_info:
            resolver.expand_template(
                template=template_a,
                params={},
                resolved_templates=resolved
            )
        assert "cycle detected" in str(exc_info.value).lower()

    def test_max_nesting_depth(self, registry):
        """Test that max nesting depth is enforced."""
        # Create deeply nested templates
        for i in range(MAX_TEMPLATE_NESTING_DEPTH + 2):
            if i == 0:
                steps = [TemplateStep(id="step", agent="GL-DATA-X-001")]
            else:
                steps = [TemplateStep(
                    id="step",
                    template=f"t{i-1}.template-{i-1}",
                    templateParams={}
                )]

            registry.register(PipelineTemplate(
                name=f"template-{i}",
                version="1.0.0",
                steps=steps
            ))

        resolver = TemplateResolver(registry)
        resolved = {f"t{i}": registry.get(f"template-{i}") for i in range(MAX_TEMPLATE_NESTING_DEPTH + 2)}

        top_template = registry.get(f"template-{MAX_TEMPLATE_NESTING_DEPTH + 1}")

        with pytest.raises(ValueError) as exc_info:
            resolver.expand_template(
                template=top_template,
                params={},
                resolved_templates=resolved
            )
        assert "depth exceeds limit" in str(exc_info.value).lower()


# ==============================================================================
# Test Parameter Substitution
# ==============================================================================

class TestParameterSubstitution:
    """Tests for parameter substitution during expansion."""

    def test_string_substitution(self, registry):
        """Test string parameter substitution."""
        template = PipelineTemplate(
            name="test",
            version="1.0.0",
            parameters={
                "uri": TemplateParameter(type=TemplateParameterType.STRING, required=True)
            },
            steps=[
                TemplateStep(
                    id="step1",
                    agent="GL-DATA-X-001",
                    in_={"path": "prefix/{{ params.uri }}/suffix"}
                )
            ]
        )
        registry.register(template)
        resolver = TemplateResolver(registry)

        result = resolver.expand_template(template, {"uri": "my-data"})
        assert result.steps[0].in_["path"] == "prefix/my-data/suffix"

    def test_numeric_substitution(self, registry):
        """Test numeric parameter substitution."""
        template = PipelineTemplate(
            name="test",
            version="1.0.0",
            parameters={
                "count": TemplateParameter(type=TemplateParameterType.INTEGER, required=True)
            },
            steps=[
                TemplateStep(
                    id="step1",
                    agent="GL-DATA-X-001",
                    in_={"count": "{{ params.count }}"}
                )
            ]
        )
        registry.register(template)
        resolver = TemplateResolver(registry)

        result = resolver.expand_template(template, {"count": 42})
        assert result.steps[0].in_["count"] == 42

    def test_boolean_substitution(self, registry):
        """Test boolean parameter substitution."""
        template = PipelineTemplate(
            name="test",
            version="1.0.0",
            parameters={
                "enabled": TemplateParameter(type=TemplateParameterType.BOOLEAN, required=True)
            },
            steps=[
                TemplateStep(
                    id="step1",
                    agent="GL-DATA-X-001",
                    in_={"enabled": "{{ params.enabled }}"}
                )
            ]
        )
        registry.register(template)
        resolver = TemplateResolver(registry)

        result = resolver.expand_template(template, {"enabled": True})
        assert result.steps[0].in_["enabled"] is True

    def test_nested_dict_substitution(self, registry):
        """Test parameter substitution in nested dictionaries."""
        template = PipelineTemplate(
            name="test",
            version="1.0.0",
            parameters={
                "bucket": TemplateParameter(type=TemplateParameterType.STRING, required=True)
            },
            steps=[
                TemplateStep(
                    id="step1",
                    agent="GL-DATA-X-001",
                    in_={
                        "config": {
                            "storage": {
                                "bucket": "{{ params.bucket }}"
                            }
                        }
                    }
                )
            ]
        )
        registry.register(template)
        resolver = TemplateResolver(registry)

        result = resolver.expand_template(template, {"bucket": "my-bucket"})
        assert result.steps[0].in_["config"]["storage"]["bucket"] == "my-bucket"


# ==============================================================================
# Test Cache Behavior
# ==============================================================================

class TestCacheBehavior:
    """Tests for template expansion cache behavior."""

    def test_cache_hit(self, registry, simple_template):
        """Test that cache hits return same result."""
        registry.register(simple_template)
        resolver = TemplateResolver(registry)

        params = {"input_uri": "s3://bucket/data"}

        result1 = resolver.expand_template(template=simple_template, params=params)
        result2 = resolver.expand_template(template=simple_template, params=params)

        # Same object from cache
        assert result1 is result2

    def test_cache_miss_different_params(self, registry, simple_template):
        """Test that different params cause cache miss."""
        registry.register(simple_template)
        resolver = TemplateResolver(registry)

        result1 = resolver.expand_template(
            template=simple_template,
            params={"input_uri": "s3://bucket/data1"}
        )
        result2 = resolver.expand_template(
            template=simple_template,
            params={"input_uri": "s3://bucket/data2"}
        )

        assert result1 is not result2

    def test_clear_cache(self, registry, simple_template):
        """Test cache clearing."""
        registry.register(simple_template)
        resolver = TemplateResolver(registry)

        params = {"input_uri": "s3://bucket/data"}

        result1 = resolver.expand_template(template=simple_template, params=params)
        resolver.clear_cache()
        result2 = resolver.expand_template(template=simple_template, params=params)

        # Different objects after cache clear
        assert result1 is not result2
        # But same content
        assert result1.content_hash == result2.content_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
