"""
Pipeline Loader and Validator for GreenLang SDK

This module provides comprehensive pipeline loading, validation, and management
capabilities with proper error handling, schema validation, and policy compliance.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

# Optional JSON schema validation
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    jsonschema = None
    HAS_JSONSCHEMA = False

from .pipeline_spec import PipelineSpec

logger = logging.getLogger(__name__)


class PipelineValidationError(Exception):
    """Raised when pipeline validation fails."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


class Pipeline:
    """
    Pipeline class that holds pipeline specification, raw data, and metadata.

    This class provides a comprehensive interface for loading, validating, and
    working with GreenLang pipelines.
    """

    def __init__(
        self,
        spec: PipelineSpec,
        raw: Dict[str, Any],
        path: Optional[Path] = None,
        schema: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Pipeline instance.

        Args:
            spec: Validated PipelineSpec instance
            raw: Original raw dictionary from YAML/JSON
            path: Path to source file (if loaded from file)
            schema: JSON schema used for validation (if any)
        """
        self.spec = spec
        self.raw = raw
        self.path = path
        self.schema = schema
        self._validation_cache: Optional[List[str]] = None
        self._loaded_at = time.time()

        logger.debug(f"Pipeline '{spec.name}' initialized from {path or 'dict'}")

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        schema: Optional[Dict[str, Any]] = None,
        validate_schema: bool = True,
        validate_spec: bool = True
    ) -> "Pipeline":
        """
        Load pipeline from YAML file with comprehensive validation.

        Args:
            path: Path to YAML file
            schema: Optional JSON schema for validation
            validate_schema: Whether to validate against JSON schema
            validate_spec: Whether to validate PipelineSpec

        Returns:
            Pipeline instance

        Raises:
            FileNotFoundError: If file doesn't exist
            PipelineValidationError: If validation fails
            yaml.YAMLError: If YAML parsing fails
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")

        if not path.is_file():
            raise FileNotFoundError(f"Path is not a file: {path}")

        try:
            # Load YAML safely
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = yaml.safe_load(f)

            if raw_data is None:
                raise PipelineValidationError(f"Empty or invalid YAML file: {path}")

            if not isinstance(raw_data, dict):
                raise PipelineValidationError(
                    f"Pipeline YAML must contain a dictionary, got {type(raw_data).__name__}"
                )

            logger.info(f"Loaded YAML from {path}")

        except yaml.YAMLError as e:
            raise PipelineValidationError(f"Failed to parse YAML: {e}") from e
        except UnicodeDecodeError as e:
            raise PipelineValidationError(f"Failed to read file (encoding issue): {e}") from e

        # Validate against JSON schema if provided
        if validate_schema and schema:
            if not HAS_JSONSCHEMA:
                logger.warning("JSON schema validation requested but jsonschema not available")
            else:
                try:
                    jsonschema.validate(raw_data, schema)
                    logger.debug("JSON schema validation passed")
                except jsonschema.ValidationError as e:
                    raise PipelineValidationError(
                        f"Schema validation failed: {e.message}",
                        [f"Path: {'.'.join(str(p) for p in e.absolute_path)}, Error: {e.message}"]
                    ) from e
                except jsonschema.SchemaError as e:
                    raise PipelineValidationError(f"Invalid schema provided: {e.message}") from e

        # Create PipelineSpec via Pydantic
        try:
            if validate_spec:
                spec = PipelineSpec(**raw_data)
                logger.debug(f"PipelineSpec validation passed for '{spec.name}'")
            else:
                # Create minimal spec without full validation
                spec = PipelineSpec.model_construct(**raw_data)
                logger.debug("PipelineSpec created without full validation")

        except ValidationError as e:
            error_messages = []
            for error in e.errors():
                field_path = ".".join(str(p) for p in error["loc"])
                error_messages.append(f"Field '{field_path}': {error['msg']}")

            raise PipelineValidationError(
                f"Pipeline specification validation failed for {path}",
                error_messages
            ) from e
        except Exception as e:
            raise PipelineValidationError(f"Unexpected error creating PipelineSpec: {e}") from e

        return cls(spec=spec, raw=raw_data, path=path, schema=schema)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
        validate_schema: bool = True,
        validate_spec: bool = True
    ) -> "Pipeline":
        """
        Create pipeline from dictionary data.

        Args:
            data: Pipeline data dictionary
            schema: Optional JSON schema for validation
            validate_schema: Whether to validate against JSON schema
            validate_spec: Whether to validate PipelineSpec

        Returns:
            Pipeline instance

        Raises:
            PipelineValidationError: If validation fails
        """
        if not isinstance(data, dict):
            raise PipelineValidationError(
                f"Pipeline data must be a dictionary, got {type(data).__name__}"
            )

        # Validate against JSON schema if provided
        if validate_schema and schema:
            if not HAS_JSONSCHEMA:
                logger.warning("JSON schema validation requested but jsonschema not available")
            else:
                try:
                    jsonschema.validate(data, schema)
                    logger.debug("JSON schema validation passed")
                except jsonschema.ValidationError as e:
                    raise PipelineValidationError(
                        f"Schema validation failed: {e.message}",
                        [f"Path: {'.'.join(str(p) for p in e.absolute_path)}, Error: {e.message}"]
                    ) from e

        # Create PipelineSpec
        try:
            if validate_spec:
                spec = PipelineSpec(**data)
                logger.debug(f"PipelineSpec validation passed for '{spec.name}'")
            else:
                spec = PipelineSpec.model_construct(**data)
                logger.debug("PipelineSpec created without full validation")

        except ValidationError as e:
            error_messages = []
            for error in e.errors():
                field_path = ".".join(str(p) for p in error["loc"])
                error_messages.append(f"Field '{field_path}': {error['msg']}")

            raise PipelineValidationError(
                "Pipeline specification validation failed",
                error_messages
            ) from e

        return cls(spec=spec, raw=data, schema=schema)

    def validate(self, strict: bool = False) -> List[str]:
        """
        Perform comprehensive validation of the pipeline.

        Args:
            strict: Whether to perform strict validation (more checks)

        Returns:
            List of validation error/warning messages
        """
        if self._validation_cache is not None and not strict:
            return self._validation_cache

        errors = []

        # Basic spec validation (already done in constructor, but re-check)
        try:
            # Re-validate the spec to catch any issues
            PipelineSpec(**self.raw)
        except ValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(p) for p in error["loc"])
                errors.append(f"Spec validation error - {field_path}: {error['msg']}")

        # Additional validation checks
        errors.extend(self._validate_step_references())
        errors.extend(self._validate_naming_conventions())
        errors.extend(self._validate_resource_constraints())

        if strict:
            errors.extend(self._validate_security_constraints())
            errors.extend(self._validate_performance_considerations())

        # Cache results for non-strict validation
        if not strict:
            self._validation_cache = errors

        logger.debug(f"Validation completed for '{self.spec.name}': {len(errors)} issues found")
        return errors

    def _validate_step_references(self) -> List[str]:
        """Validate step references and dependencies."""
        errors = []
        step_names = {step.name for step in self.spec.steps}

        for i, step in enumerate(self.spec.steps):
            # Check inputsRef references
            if step.inputsRef:
                if step.inputsRef.startswith("$steps."):
                    ref_parts = step.inputsRef.split(".")
                    if len(ref_parts) >= 2:
                        ref_step_name = ref_parts[1]
                        if ref_step_name not in step_names:
                            errors.append(
                                f"Step '{step.name}' references unknown step '{ref_step_name}' in inputsRef"
                            )
                        elif ref_step_name == step.name:
                            errors.append(f"Step '{step.name}' cannot reference itself in inputsRef")

            # Check condition references
            if step.condition:
                # Simple regex to find step references
                step_refs = re.findall(r'\bsteps\.(\w+)', step.condition)
                for ref_step in step_refs:
                    if ref_step not in step_names:
                        errors.append(
                            f"Step '{step.name}' condition references unknown step '{ref_step}'"
                        )
                    elif ref_step == step.name:
                        errors.append(f"Step '{step.name}' condition cannot reference itself")

        return errors

    def _validate_naming_conventions(self) -> List[str]:
        """Validate naming conventions."""
        errors = []

        # Pipeline name validation
        if not re.match(r'^[a-z0-9][a-z0-9-_]*[a-z0-9]$', self.spec.name):
            errors.append(
                "Pipeline name should use lowercase letters, numbers, hyphens, and underscores"
            )

        # Step name validation
        for step in self.spec.steps:
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9-_]*$', step.name):
                errors.append(
                    f"Step '{step.name}' should start with a letter and use alphanumeric chars, hyphens, underscores"
                )

        return errors

    def _validate_resource_constraints(self) -> List[str]:
        """Validate resource constraints and limits."""
        errors = []

        # Check for reasonable timeout values
        for step in self.spec.steps:
            if step.timeout is not None:
                if step.timeout <= 0:
                    errors.append(f"Step '{step.name}' has invalid timeout: {step.timeout}")
                elif step.timeout > 3600:  # 1 hour
                    errors.append(f"Step '{step.name}' has very long timeout: {step.timeout}s")

        # Check parallel step limits
        parallel_steps = [step for step in self.spec.steps if step.parallel]
        if len(parallel_steps) > 10:
            errors.append(f"Too many parallel steps ({len(parallel_steps)}), consider reducing for performance")

        return errors

    def _validate_security_constraints(self) -> List[str]:
        """Validate security-related constraints."""
        errors = []

        # Check for potentially unsafe patterns
        for step in self.spec.steps:
            # Check for hardcoded secrets or keys in inputs
            if step.inputs:
                for key, value in step.inputs.items():
                    if isinstance(value, str):
                        if any(pattern in key.lower() for pattern in ['password', 'secret', 'key', 'token']):
                            if not value.startswith('$'):  # Not a reference
                                errors.append(
                                    f"Step '{step.name}' may contain hardcoded secret in input '{key}'"
                                )

        return errors

    def _validate_performance_considerations(self) -> List[str]:
        """Validate performance-related considerations."""
        errors = []

        # Check for potential performance issues
        if len(self.spec.steps) > 50:
            errors.append(f"Pipeline has many steps ({len(self.spec.steps)}), consider breaking into smaller pipelines")

        # Check for steps without timeout
        steps_without_timeout = [step for step in self.spec.steps if step.timeout is None]
        if len(steps_without_timeout) > len(self.spec.steps) * 0.5:
            errors.append("Many steps lack timeout configuration, consider adding timeouts for reliability")

        return errors

    def to_policy_doc(self) -> Dict[str, Any]:
        """
        Create minimal structure for policy evaluation.

        Returns:
            Dictionary suitable for policy evaluation with OPA or similar
        """
        # Extract key information for policy evaluation
        policy_doc = {
            "pipeline": {
                "name": self.spec.name,
                "version": self.spec.version,
                "description": self.spec.description,
                "author": self.spec.author,
                "tags": self.spec.tags or [],
            },
            "execution": {
                "stop_on_error": self.spec.stop_on_error,
                "max_parallel_steps": self.spec.max_parallel_steps,
                "artifacts_dir": self.spec.artifacts_dir,
            },
            "steps": [],
            "security": {
                "has_network_access": False,
                "has_file_access": False,
                "has_external_references": False,
                "parallel_execution": any(step.parallel for step in self.spec.steps),
            },
            "metadata": {
                "step_count": len(self.spec.steps),
                "parallel_step_count": len([s for s in self.spec.steps if s.parallel]),
                "has_timeouts": any(s.timeout is not None for s in self.spec.steps),
                "has_conditions": any(s.condition is not None for s in self.spec.steps),
                "has_retries": any(
                    hasattr(s.on_error, 'retry') and s.on_error.retry is not None
                    for s in self.spec.steps
                    if hasattr(s.on_error, 'retry')
                ),
                "loaded_from": str(self.path) if self.path else "dict",
                "loaded_at": self._loaded_at,
            }
        }

        # Process steps for policy evaluation
        for step in self.spec.steps:
            step_doc = {
                "name": step.name,
                "agent": step.agent,
                "action": step.action,
                "parallel": step.parallel,
                "has_timeout": step.timeout is not None,
                "timeout": step.timeout,
                "has_condition": step.condition is not None,
                "has_inputs": step.inputs is not None,
                "has_outputs": step.outputs is not None,
                "on_error_policy": (
                    step.on_error.policy if hasattr(step.on_error, 'policy')
                    else step.on_error
                ),
            }

            # Analyze inputs for security concerns
            if step.inputs:
                step_doc["input_keys"] = list(step.inputs.keys())
                # Check for potential secrets
                potential_secrets = [
                    k for k in step.inputs.keys()
                    if any(pattern in k.lower() for pattern in ['password', 'secret', 'key', 'token'])
                ]
                if potential_secrets:
                    step_doc["potential_secret_inputs"] = potential_secrets
                    policy_doc["security"]["has_external_references"] = True

            # Check for external references
            if step.inputsRef and step.inputsRef.startswith("$"):
                policy_doc["security"]["has_external_references"] = True

            policy_doc["steps"].append(step_doc)

        return policy_doc

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation."""
        return self.spec.model_dump()

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Convert pipeline to YAML format.

        Args:
            path: Optional path to save YAML file

        Returns:
            YAML string representation
        """
        yaml_str = yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True
        )

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
            logger.info(f"Pipeline saved to {path}")

        return yaml_str

    def get_step(self, name: str) -> Optional[Any]:
        """Get a step by name."""
        return self.spec.get_step(name)

    def is_valid(self) -> bool:
        """Check if pipeline is valid (no validation errors)."""
        return len(self.validate()) == 0

    def __str__(self) -> str:
        """String representation of the pipeline."""
        return f"Pipeline(name='{self.spec.name}', version='{self.spec.version}', steps={len(self.spec.steps)})"

    def __repr__(self) -> str:
        """Detailed representation of the pipeline."""
        return (
            f"Pipeline(name='{self.spec.name}', version='{self.spec.version}', "
            f"steps={len(self.spec.steps)}, path={self.path})"
        )


def load_pipeline_schema(schema_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON schema for pipeline validation.

    Args:
        schema_path: Path to JSON schema file

    Returns:
        Schema dictionary

    Raises:
        FileNotFoundError: If schema file doesn't exist
        json.JSONDecodeError: If schema is invalid JSON
    """
    schema_path = Path(schema_path)

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    logger.debug(f"Loaded pipeline schema from {schema_path}")
    return schema


def validate_pipeline_file(
    path: Union[str, Path],
    schema_path: Optional[Union[str, Path]] = None,
    strict: bool = False
) -> List[str]:
    """
    Validate a pipeline file and return any errors.

    Args:
        path: Path to pipeline file
        schema_path: Optional path to JSON schema
        strict: Whether to perform strict validation

    Returns:
        List of validation error messages (empty if valid)
    """
    try:
        schema = None
        if schema_path:
            schema = load_pipeline_schema(schema_path)

        pipeline = Pipeline.from_yaml(path, schema=schema)
        return pipeline.validate(strict=strict)

    except (FileNotFoundError, PipelineValidationError) as e:
        return [str(e)]
    except Exception as e:
        return [f"Unexpected error validating pipeline: {e}"]