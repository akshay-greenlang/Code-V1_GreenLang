"""
GL-013 PREDICTMAINT - Configuration Validation Module

This module provides comprehensive configuration validation for the
PredictiveMaintenanceAgent. It validates agent_spec.yaml, gl.yaml,
and pack.yaml configurations against GreenLang v2 standards.

The validator ensures:
    - Schema compliance with GreenLang v2 AgentSpec
    - Required fields are present
    - Values are within acceptable ranges
    - Zero-hallucination policy is enforced
    - Compliance standards are properly configured
    - Security settings meet requirements

Example:
    >>> from gl_013.validate_config import validate_config, ConfigValidator
    >>> validator = ConfigValidator()
    >>> result = validator.validate_all()
    >>> if result.is_valid:
    ...     print("Configuration is valid")
    >>> else:
    ...     for error in result.errors:
    ...         print(f"Error: {error}")

Author: GreenLang Team
Created: 2024-12-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import hashlib
import logging
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation messages."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationMessage:
    """A single validation message with context."""
    severity: ValidationSeverity
    field: str
    message: str
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        """Return formatted validation message."""
        msg = f"[{self.severity.value.upper()}] {self.field}: {self.message}"
        if self.suggestion:
            msg += f" (Suggestion: {self.suggestion})"
        return msg


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    messages: List[ValidationMessage] = field(default_factory=list)
    config_hash: Optional[str] = None
    validated_at: datetime = field(default_factory=datetime.now)

    @property
    def errors(self) -> List[ValidationMessage]:
        """Return only error messages."""
        return [m for m in self.messages if m.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationMessage]:
        """Return only warning messages."""
        return [m for m in self.messages if m.severity == ValidationSeverity.WARNING]

    @property
    def error_count(self) -> int:
        """Return count of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Return count of warnings."""
        return len(self.warnings)


class ConfigValidator:
    """
    Configuration validator for GL-013 PREDICTMAINT agent.

    This class validates all configuration files against GreenLang v2
    standards and ensures compliance with zero-hallucination policy.

    Attributes:
        base_path: Path to the agent directory
        agent_spec: Loaded agent_spec.yaml content
        gl_yaml: Loaded gl.yaml content
        pack_yaml: Loaded pack.yaml content

    Example:
        >>> validator = ConfigValidator()
        >>> result = validator.validate_all()
        >>> print(f"Valid: {result.is_valid}, Errors: {result.error_count}")
    """

    # Required fields for agent_spec.yaml
    REQUIRED_METADATA_FIELDS = [
        "name", "version", "agent_id", "codename", "labels", "annotations"
    ]

    REQUIRED_LABEL_FIELDS = [
        "type", "category", "domain", "priority", "complexity", "id"
    ]

    REQUIRED_SPEC_FIELDS = [
        "inputs", "outputs", "capabilities", "tools", "requirements"
    ]

    # Required compliance standards for predictive maintenance
    REQUIRED_STANDARDS = [
        "ISO_10816",  # Vibration evaluation
        "ISO_13373",  # Condition monitoring
        "ISO_17359",  # CM guidelines
        "ISO_55000",  # Asset management
    ]

    # Prohibited LLM operations (zero-hallucination)
    PROHIBITED_LLM_OPERATIONS = [
        "numeric_calculation",
        "failure_probability_generation",
        "rul_estimation",
        "health_score_calculation",
        "threshold_determination",
    ]

    # Required security settings
    REQUIRED_SECURITY_SETTINGS = [
        "authentication",
        "authorization",
        "encryption",
        "audit",
    ]

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the ConfigValidator.

        Args:
            base_path: Path to the agent directory. Defaults to current directory.
        """
        self.base_path = base_path or Path(__file__).parent
        self.agent_spec: Optional[Dict[str, Any]] = None
        self.gl_yaml: Optional[Dict[str, Any]] = None
        self.pack_yaml: Optional[Dict[str, Any]] = None
        self._messages: List[ValidationMessage] = []

    def _add_error(self, field: str, message: str, suggestion: Optional[str] = None) -> None:
        """Add an error message."""
        self._messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            field=field,
            message=message,
            suggestion=suggestion
        ))

    def _add_warning(self, field: str, message: str, suggestion: Optional[str] = None) -> None:
        """Add a warning message."""
        self._messages.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            field=field,
            message=message,
            suggestion=suggestion
        ))

    def _add_info(self, field: str, message: str) -> None:
        """Add an info message."""
        self._messages.append(ValidationMessage(
            severity=ValidationSeverity.INFO,
            field=field,
            message=message
        ))

    def _load_yaml(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load a YAML configuration file.

        Args:
            filename: Name of the YAML file to load.

        Returns:
            Parsed YAML content or None if file not found.
        """
        file_path = self.base_path / filename
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                logger.info(f"Loaded {filename} successfully")
                return content
        except FileNotFoundError:
            self._add_error(filename, f"Configuration file not found: {file_path}")
            return None
        except yaml.YAMLError as e:
            self._add_error(filename, f"YAML parsing error: {str(e)}")
            return None

    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 hash of configuration for provenance tracking.

        Args:
            config: Configuration dictionary.

        Returns:
            SHA-256 hash string.
        """
        config_str = yaml.dump(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def validate_api_version(self, config: Dict[str, Any], filename: str) -> None:
        """
        Validate API version is GreenLang v2.

        Args:
            config: Configuration dictionary.
            filename: Name of the config file being validated.
        """
        api_version = config.get("apiVersion")
        if api_version != "greenlang.io/v2":
            self._add_error(
                f"{filename}.apiVersion",
                f"Expected 'greenlang.io/v2', got '{api_version}'",
                "Update apiVersion to 'greenlang.io/v2'"
            )

    def validate_kind(self, config: Dict[str, Any], filename: str) -> None:
        """
        Validate kind is AgentSpec.

        Args:
            config: Configuration dictionary.
            filename: Name of the config file being validated.
        """
        kind = config.get("kind")
        if kind != "AgentSpec":
            self._add_error(
                f"{filename}.kind",
                f"Expected 'AgentSpec', got '{kind}'",
                "Update kind to 'AgentSpec'"
            )

    def validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Validate metadata section.

        Args:
            metadata: Metadata dictionary from config.
        """
        # Check required fields
        for field in self.REQUIRED_METADATA_FIELDS:
            if field not in metadata:
                self._add_error(
                    f"metadata.{field}",
                    f"Required field '{field}' is missing",
                    f"Add '{field}' to metadata section"
                )

        # Validate agent ID format
        agent_id = metadata.get("agent_id", "")
        if not agent_id.startswith("GL-"):
            self._add_error(
                "metadata.agent_id",
                f"Agent ID must start with 'GL-', got '{agent_id}'",
                "Use format 'GL-XXX' for agent ID"
            )

        # Validate version format (semver)
        version = metadata.get("version", "")
        if not self._is_valid_semver(version):
            self._add_warning(
                "metadata.version",
                f"Version '{version}' does not follow semver format",
                "Use format 'X.Y.Z' for version"
            )

        # Validate labels
        labels = metadata.get("labels", {})
        for field in self.REQUIRED_LABEL_FIELDS:
            if field not in labels:
                self._add_error(
                    f"metadata.labels.{field}",
                    f"Required label '{field}' is missing"
                )

        # Validate priority
        priority = labels.get("priority", "")
        valid_priorities = ["P0", "P1", "P2", "P3"]
        if priority not in valid_priorities:
            self._add_warning(
                "metadata.labels.priority",
                f"Priority '{priority}' not in standard values {valid_priorities}"
            )

    def _is_valid_semver(self, version: str) -> bool:
        """Check if version follows semantic versioning."""
        if not version:
            return False
        parts = version.split(".")
        if len(parts) != 3:
            return False
        return all(part.isdigit() for part in parts)

    def validate_spec(self, spec: Dict[str, Any]) -> None:
        """
        Validate spec section.

        Args:
            spec: Spec dictionary from config.
        """
        # Check required sections
        for field in self.REQUIRED_SPEC_FIELDS:
            if field not in spec:
                self._add_error(
                    f"spec.{field}",
                    f"Required section '{field}' is missing"
                )

        # Validate inputs
        inputs = spec.get("inputs", [])
        self._validate_inputs(inputs)

        # Validate outputs
        outputs = spec.get("outputs", [])
        self._validate_outputs(outputs)

        # Validate capabilities
        capabilities = spec.get("capabilities", [])
        self._validate_capabilities(capabilities)

        # Validate tools
        tools = spec.get("tools", [])
        self._validate_tools(tools)

    def _validate_inputs(self, inputs: List[Dict[str, Any]]) -> None:
        """Validate input specifications."""
        if not inputs:
            self._add_error("spec.inputs", "At least one input is required")
            return

        required_inputs = ["vibration_data", "temperature_data", "operating_hours"]
        input_names = [inp.get("name") for inp in inputs]

        for req_input in required_inputs:
            if req_input not in input_names:
                self._add_warning(
                    f"spec.inputs.{req_input}",
                    f"Recommended input '{req_input}' is missing"
                )

        for inp in inputs:
            name = inp.get("name", "unknown")
            if "description" not in inp:
                self._add_warning(
                    f"spec.inputs.{name}",
                    "Input is missing description"
                )
            if "schema" not in inp and inp.get("type") == "object":
                self._add_warning(
                    f"spec.inputs.{name}",
                    "Object input should have schema definition"
                )

    def _validate_outputs(self, outputs: List[Dict[str, Any]]) -> None:
        """Validate output specifications."""
        if not outputs:
            self._add_error("spec.outputs", "At least one output is required")
            return

        required_outputs = ["health_indices", "remaining_useful_life", "failure_predictions"]
        output_names = [out.get("name") for out in outputs]

        for req_output in required_outputs:
            if req_output not in output_names:
                self._add_warning(
                    f"spec.outputs.{req_output}",
                    f"Recommended output '{req_output}' is missing"
                )

        # Check for provenance_hash in outputs
        for out in outputs:
            name = out.get("name", "unknown")
            schema = out.get("schema", {})
            properties = schema.get("properties", {})
            if "provenance_hash" not in properties:
                self._add_warning(
                    f"spec.outputs.{name}",
                    "Output should include provenance_hash for audit trail"
                )

    def _validate_capabilities(self, capabilities: List[Dict[str, Any]]) -> None:
        """Validate capabilities."""
        if not capabilities:
            self._add_error("spec.capabilities", "At least one capability is required")
            return

        for cap in capabilities:
            name = cap.get("name", "unknown")

            # All capabilities should be deterministic
            if not cap.get("deterministic", False):
                self._add_error(
                    f"spec.capabilities.{name}",
                    "Capability must be deterministic for zero-hallucination compliance"
                )

    def _validate_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Validate tools."""
        if len(tools) < 12:
            self._add_warning(
                "spec.tools",
                f"Expected 12 tools, found {len(tools)}",
                "Add all required deterministic tools"
            )

        for tool in tools:
            name = tool.get("name", "unknown")

            # All tools must be deterministic
            if not tool.get("deterministic", False):
                self._add_error(
                    f"spec.tools.{name}",
                    "Tool must be deterministic for zero-hallucination compliance"
                )

            # Check function reference
            if "function" not in tool:
                self._add_error(
                    f"spec.tools.{name}",
                    "Tool is missing function reference"
                )

    def validate_ai_config(self, ai_config: Dict[str, Any]) -> None:
        """
        Validate AI configuration for zero-hallucination compliance.

        Args:
            ai_config: AI configuration dictionary.
        """
        if not ai_config:
            self._add_error("ai_config", "AI configuration is required")
            return

        # Validate deterministic settings
        if not ai_config.get("deterministic", False):
            self._add_error(
                "ai_config.deterministic",
                "AI config must be deterministic for zero-hallucination compliance"
            )

        # Validate temperature
        temperature = ai_config.get("temperature", 1.0)
        if temperature != 0.0:
            self._add_error(
                "ai_config.temperature",
                f"Temperature must be 0.0 for deterministic output, got {temperature}"
            )

        # Validate seed
        if "seed" not in ai_config:
            self._add_warning(
                "ai_config.seed",
                "Seed should be specified for reproducibility"
            )

        # Validate prohibited operations
        prohibited = ai_config.get("prohibited_operations", [])
        for op in self.PROHIBITED_LLM_OPERATIONS:
            if op not in prohibited:
                self._add_error(
                    f"ai_config.prohibited_operations.{op}",
                    f"Operation '{op}' must be prohibited for zero-hallucination"
                )

    def validate_compliance(self, compliance: Dict[str, Any]) -> None:
        """
        Validate compliance configuration.

        Args:
            compliance: Compliance configuration dictionary.
        """
        if not compliance:
            self._add_error("compliance", "Compliance configuration is required")
            return

        # Check required standards
        standards = compliance.get("standards", [])
        for std in self.REQUIRED_STANDARDS:
            if std not in standards:
                self._add_error(
                    f"compliance.standards.{std}",
                    f"Required standard '{std}' is missing"
                )

        # Validate audit trail
        audit_trail = compliance.get("audit_trail", {})
        if not audit_trail.get("enabled", False):
            self._add_error(
                "compliance.audit_trail.enabled",
                "Audit trail must be enabled for compliance"
            )

        retention_days = audit_trail.get("retention_days", 0)
        if retention_days < 2555:  # 7 years
            self._add_warning(
                "compliance.audit_trail.retention_days",
                f"Retention period ({retention_days} days) may not meet regulatory requirements",
                "Set retention_days to at least 2555 (7 years)"
            )

        # Validate provenance tracking
        provenance = compliance.get("provenance_tracking", {})
        if not provenance.get("enabled", False):
            self._add_error(
                "compliance.provenance_tracking.enabled",
                "Provenance tracking must be enabled for audit compliance"
            )

        # Validate zero-hallucination
        zero_hall = compliance.get("zero_hallucination", {})
        if not zero_hall.get("enabled", False):
            self._add_error(
                "compliance.zero_hallucination.enabled",
                "Zero-hallucination must be enabled"
            )

    def validate_security(self, security: Dict[str, Any]) -> None:
        """
        Validate security configuration.

        Args:
            security: Security configuration dictionary.
        """
        if not security:
            self._add_error("security", "Security configuration is required")
            return

        for setting in self.REQUIRED_SECURITY_SETTINGS:
            if setting not in security:
                self._add_error(
                    f"security.{setting}",
                    f"Required security setting '{setting}' is missing"
                )

        # Validate encryption
        encryption = security.get("encryption", {})
        in_transit = encryption.get("in_transit", "")
        if in_transit not in ["TLS_1_3", "TLS_1_2"]:
            self._add_warning(
                "security.encryption.in_transit",
                f"Encryption '{in_transit}' should be TLS 1.3 or 1.2"
            )

    def validate_deployment(self, deployment: Dict[str, Any]) -> None:
        """
        Validate deployment configuration.

        Args:
            deployment: Deployment configuration dictionary.
        """
        if not deployment:
            self._add_warning("deployment", "Deployment configuration is missing")
            return

        # Validate replicas
        replicas = deployment.get("replicas", 1)
        if replicas < 2:
            self._add_warning(
                "deployment.replicas",
                f"Only {replicas} replica configured; recommend at least 2 for HA"
            )

        # Validate health checks
        health_check = deployment.get("health_check", {})
        if not health_check.get("liveness_probe"):
            self._add_warning(
                "deployment.health_check.liveness_probe",
                "Liveness probe not configured"
            )
        if not health_check.get("readiness_probe"):
            self._add_warning(
                "deployment.health_check.readiness_probe",
                "Readiness probe not configured"
            )

    def validate_monitoring(self, monitoring: Dict[str, Any]) -> None:
        """
        Validate monitoring configuration.

        Args:
            monitoring: Monitoring configuration dictionary.
        """
        if not monitoring:
            self._add_warning("monitoring", "Monitoring configuration is missing")
            return

        # Validate Prometheus
        prometheus = monitoring.get("prometheus", {})
        if not prometheus.get("enabled", False):
            self._add_warning(
                "monitoring.prometheus.enabled",
                "Prometheus monitoring is not enabled"
            )

        # Validate alerts
        alerts = monitoring.get("alerts", [])
        if len(alerts) < 3:
            self._add_warning(
                "monitoring.alerts",
                f"Only {len(alerts)} alerts configured; recommend more for production"
            )

    def validate_agent_spec(self) -> None:
        """Validate agent_spec.yaml configuration."""
        self.agent_spec = self._load_yaml("agent_spec.yaml")
        if not self.agent_spec:
            return

        self.validate_api_version(self.agent_spec, "agent_spec.yaml")
        self.validate_kind(self.agent_spec, "agent_spec.yaml")

        metadata = self.agent_spec.get("metadata", {})
        self.validate_metadata(metadata)

        spec = self.agent_spec.get("spec", {})
        self.validate_spec(spec)

        ai_config = self.agent_spec.get("ai_config", {})
        self.validate_ai_config(ai_config)

        compliance = self.agent_spec.get("compliance", {})
        self.validate_compliance(compliance)

        security = self.agent_spec.get("security", {})
        self.validate_security(security)

        deployment = self.agent_spec.get("deployment", {})
        self.validate_deployment(deployment)

        monitoring = self.agent_spec.get("monitoring", {})
        self.validate_monitoring(monitoring)

    def validate_gl_yaml(self) -> None:
        """Validate gl.yaml configuration."""
        self.gl_yaml = self._load_yaml("gl.yaml")
        if not self.gl_yaml:
            return

        self.validate_api_version(self.gl_yaml, "gl.yaml")
        self.validate_kind(self.gl_yaml, "gl.yaml")

        # Ensure consistency with agent_spec.yaml
        if self.agent_spec:
            spec_version = self.agent_spec.get("metadata", {}).get("version")
            gl_version = self.gl_yaml.get("metadata", {}).get("version")
            if spec_version != gl_version:
                self._add_warning(
                    "gl.yaml.metadata.version",
                    f"Version mismatch: agent_spec={spec_version}, gl.yaml={gl_version}"
                )

    def validate_pack_yaml(self) -> None:
        """Validate pack.yaml configuration."""
        self.pack_yaml = self._load_yaml("pack.yaml")
        if not self.pack_yaml:
            return

        # Validate required fields
        required_fields = ["agent_id", "codename", "name", "version", "domain"]
        for field in required_fields:
            if field not in self.pack_yaml:
                self._add_error(
                    f"pack.yaml.{field}",
                    f"Required field '{field}' is missing"
                )

        # Ensure consistency with agent_spec.yaml
        if self.agent_spec:
            spec_id = self.agent_spec.get("metadata", {}).get("agent_id")
            pack_id = self.pack_yaml.get("agent_id")
            if spec_id != pack_id:
                self._add_error(
                    "pack.yaml.agent_id",
                    f"Agent ID mismatch: agent_spec={spec_id}, pack.yaml={pack_id}"
                )

    def validate_all(self) -> ValidationResult:
        """
        Validate all configuration files.

        Returns:
            ValidationResult with overall status and messages.
        """
        self._messages = []

        logger.info("Starting configuration validation for GL-013 PREDICTMAINT")

        # Validate each config file
        self.validate_agent_spec()
        self.validate_gl_yaml()
        self.validate_pack_yaml()

        # Calculate config hash for provenance
        config_hash = None
        if self.agent_spec:
            config_hash = self._calculate_config_hash(self.agent_spec)

        # Determine overall validity
        has_errors = any(m.severity == ValidationSeverity.ERROR for m in self._messages)

        result = ValidationResult(
            is_valid=not has_errors,
            messages=self._messages.copy(),
            config_hash=config_hash
        )

        logger.info(
            f"Validation complete: valid={result.is_valid}, "
            f"errors={result.error_count}, warnings={result.warning_count}"
        )

        return result


def validate_config(base_path: Optional[Union[str, Path]] = None) -> ValidationResult:
    """
    Convenience function to validate GL-013 configuration.

    Args:
        base_path: Optional path to agent directory.

    Returns:
        ValidationResult with validation status and messages.

    Example:
        >>> result = validate_config()
        >>> if not result.is_valid:
        ...     for error in result.errors:
        ...         print(error)
    """
    if base_path:
        base_path = Path(base_path)

    validator = ConfigValidator(base_path)
    return validator.validate_all()


if __name__ == "__main__":
    """Run configuration validation when module is executed directly."""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run validation
    result = validate_config()

    # Print results
    print("\n" + "=" * 60)
    print("GL-013 PREDICTMAINT Configuration Validation Report")
    print("=" * 60)
    print(f"\nValidation Status: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"Config Hash: {result.config_hash or 'N/A'}")
    print(f"Validated At: {result.validated_at.isoformat()}")
    print(f"\nErrors: {result.error_count}")
    print(f"Warnings: {result.warning_count}")

    if result.errors:
        print("\n--- ERRORS ---")
        for msg in result.errors:
            print(f"  {msg}")

    if result.warnings:
        print("\n--- WARNINGS ---")
        for msg in result.warnings:
            print(f"  {msg}")

    # Exit with appropriate code
    sys.exit(0 if result.is_valid else 1)
