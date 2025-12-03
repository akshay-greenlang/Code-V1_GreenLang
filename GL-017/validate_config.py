#!/usr/bin/env python3
"""
GL-017 CONDENSYNC - Configuration Validator

This script validates the GreenLang configuration files (pack.yaml, gl.yaml, run.json)
against the v1.0 specification requirements.

Usage:
    python validate_config.py [file]
    python validate_config.py pack.yaml
    python validate_config.py gl.yaml
    python validate_config.py run.json
    python validate_config.py --all

Author: GreenLang AI Team
Version: 1.0.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


class ValidationError:
    """Represents a validation error."""

    def __init__(self, path: str, message: str, severity: str = "ERROR"):
        self.path = path
        self.message = message
        self.severity = severity

    def __str__(self) -> str:
        return f"[{self.severity}] {self.path}: {self.message}"


class ConfigValidator:
    """Validates GreenLang configuration files."""

    # Required sections for pack.yaml
    PACK_REQUIRED_SECTIONS = [
        "agent",
        "runtime",
        "dependencies",
        "resources",
        "ports",
        "inputs",
        "outputs",
        "integrations",
        "capabilities",
        "compliance",
        "monitoring",
        "security",
        "deployment",
        "data_retention",
        "business_metrics",
    ]

    # Required sections for gl.yaml
    GL_REQUIRED_SECTIONS = [
        "agent",
        "runtime",
        "ai_configuration",
        "data_sources",
        "data_sinks",
        "monitoring",
        "alerting",
        "security",
        "deployment",
        "compliance",
        "performance",
        "data_retention",
    ]

    # Required agent metadata fields
    AGENT_REQUIRED_FIELDS = [
        "id",
        "codename",
        "name",
        "version",
        "category",
        "domain",
    ]

    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

    def validate_pack_yaml(self, data: Dict[str, Any]) -> bool:
        """Validate pack.yaml structure and content."""
        self.errors = []
        self.warnings = []

        # Check required sections
        for section in self.PACK_REQUIRED_SECTIONS:
            if section not in data:
                self.errors.append(
                    ValidationError(f"pack.yaml.{section}", f"Missing required section: {section}")
                )

        # Validate agent metadata
        if "agent" in data:
            self._validate_agent_metadata(data["agent"], "pack.yaml.agent")

        # Validate runtime
        if "runtime" in data:
            self._validate_runtime(data["runtime"], "pack.yaml.runtime")

        # Validate dependencies
        if "dependencies" in data:
            self._validate_dependencies(data["dependencies"], "pack.yaml.dependencies")

        # Validate resources
        if "resources" in data:
            self._validate_resources(data["resources"], "pack.yaml.resources")

        # Validate ports
        if "ports" in data:
            self._validate_ports(data["ports"], "pack.yaml.ports")

        # Validate inputs/outputs schemas
        if "inputs" in data:
            self._validate_io_schema(data["inputs"], "pack.yaml.inputs")
        if "outputs" in data:
            self._validate_io_schema(data["outputs"], "pack.yaml.outputs")

        # Validate compliance
        if "compliance" in data:
            self._validate_compliance(data["compliance"], "pack.yaml.compliance")

        # Validate security
        if "security" in data:
            self._validate_security(data["security"], "pack.yaml.security")

        # Check line count (should be 800-900 lines)
        return len(self.errors) == 0

    def validate_gl_yaml(self, data: Dict[str, Any]) -> bool:
        """Validate gl.yaml structure and content."""
        self.errors = []
        self.warnings = []

        # Check required sections
        for section in self.GL_REQUIRED_SECTIONS:
            if section not in data:
                self.errors.append(
                    ValidationError(f"gl.yaml.{section}", f"Missing required section: {section}")
                )

        # Validate agent metadata
        if "agent" in data:
            self._validate_agent_metadata(data["agent"], "gl.yaml.agent")

        # Validate AI configuration
        if "ai_configuration" in data:
            self._validate_ai_config(data["ai_configuration"], "gl.yaml.ai_configuration")

        # Validate data sources
        if "data_sources" in data:
            self._validate_data_sources(data["data_sources"], "gl.yaml.data_sources")

        # Validate alerting
        if "alerting" in data:
            self._validate_alerting(data["alerting"], "gl.yaml.alerting")

        # Validate security
        if "security" in data:
            self._validate_security(data["security"], "gl.yaml.security")

        # Validate performance targets
        if "performance" in data:
            self._validate_performance(data["performance"], "gl.yaml.performance")

        return len(self.errors) == 0

    def validate_run_json(self, data: Dict[str, Any]) -> bool:
        """Validate run.json structure and content."""
        self.errors = []
        self.warnings = []

        # Check required fields
        required_fields = ["apiVersion", "kind", "metadata", "spec"]
        for field in required_fields:
            if field not in data:
                self.errors.append(
                    ValidationError(f"run.json.{field}", f"Missing required field: {field}")
                )

        # Validate API version
        if "apiVersion" in data:
            if not data["apiVersion"].startswith("greenlang.io/"):
                self.errors.append(
                    ValidationError(
                        "run.json.apiVersion",
                        "API version must start with 'greenlang.io/'",
                    )
                )

        # Validate kind
        if "kind" in data:
            valid_kinds = ["AgentRunConfig", "AgentSpec", "AgentConfiguration"]
            if data["kind"] not in valid_kinds:
                self.warnings.append(
                    ValidationError(
                        "run.json.kind",
                        f"Unexpected kind: {data['kind']}. Expected one of: {valid_kinds}",
                        "WARNING",
                    )
                )

        # Validate spec
        if "spec" in data:
            self._validate_run_spec(data["spec"], "run.json.spec")

        return len(self.errors) == 0

    def _validate_agent_metadata(self, data: Dict[str, Any], path: str) -> None:
        """Validate agent metadata fields."""
        for field in self.AGENT_REQUIRED_FIELDS:
            if field not in data:
                self.errors.append(
                    ValidationError(f"{path}.{field}", f"Missing required field: {field}")
                )

        # Validate agent ID format
        if "id" in data:
            agent_id = data["id"]
            if not agent_id.startswith("GL-"):
                self.errors.append(
                    ValidationError(f"{path}.id", "Agent ID must start with 'GL-'")
                )

        # Validate version format (semver)
        if "version" in data:
            version = data["version"]
            parts = version.split(".")
            if len(parts) != 3:
                self.warnings.append(
                    ValidationError(
                        f"{path}.version",
                        f"Version '{version}' should follow semver (X.Y.Z)",
                        "WARNING",
                    )
                )

    def _validate_runtime(self, data: Dict[str, Any], path: str) -> None:
        """Validate runtime configuration."""
        if "language" in data:
            if data["language"] not in ["python", "Python"]:
                self.warnings.append(
                    ValidationError(
                        f"{path}.language",
                        f"Unexpected language: {data['language']}",
                        "WARNING",
                    )
                )

        if "python_version" in data:
            version = data["python_version"]
            if not version.startswith("3.1"):
                self.warnings.append(
                    ValidationError(
                        f"{path}.python_version",
                        f"Python version '{version}' may not be supported",
                        "WARNING",
                    )
                )

        # Check determinism settings
        if "deterministic" in data:
            det = data["deterministic"]
            if isinstance(det, dict):
                if det.get("enabled") and det.get("seed") is None:
                    self.warnings.append(
                        ValidationError(
                            f"{path}.deterministic",
                            "Deterministic mode enabled but no seed specified",
                            "WARNING",
                        )
                    )

    def _validate_dependencies(self, data: Dict[str, Any], path: str) -> None:
        """Validate dependencies section."""
        if "required" not in data:
            self.warnings.append(
                ValidationError(
                    f"{path}.required",
                    "No required dependencies specified",
                    "WARNING",
                )
            )

    def _validate_resources(self, data: Dict[str, Any], path: str) -> None:
        """Validate resource requirements."""
        required_resources = ["cpu", "memory"]
        for resource in required_resources:
            if resource not in data:
                self.warnings.append(
                    ValidationError(
                        f"{path}.{resource}",
                        f"Resource '{resource}' not specified",
                        "WARNING",
                    )
                )

    def _validate_ports(self, data: Dict[str, Any], path: str) -> None:
        """Validate port configuration."""
        if "http" not in data:
            self.errors.append(
                ValidationError(f"{path}.http", "HTTP port not specified")
            )

        if "metrics" not in data:
            self.warnings.append(
                ValidationError(
                    f"{path}.metrics",
                    "Metrics port not specified",
                    "WARNING",
                )
            )

    def _validate_io_schema(self, data: Dict[str, Any], path: str) -> None:
        """Validate input/output schema definitions."""
        if not data:
            self.warnings.append(
                ValidationError(f"{path}", "Empty schema definition", "WARNING")
            )

    def _validate_compliance(self, data: Dict[str, Any], path: str) -> None:
        """Validate compliance configuration."""
        if "standards" not in data:
            self.warnings.append(
                ValidationError(
                    f"{path}.standards",
                    "No compliance standards specified",
                    "WARNING",
                )
            )

        if "audit_trail" not in data:
            self.warnings.append(
                ValidationError(
                    f"{path}.audit_trail",
                    "Audit trail configuration not specified",
                    "WARNING",
                )
            )

    def _validate_security(self, data: Dict[str, Any], path: str) -> None:
        """Validate security configuration."""
        if "authentication" not in data:
            self.errors.append(
                ValidationError(
                    f"{path}.authentication",
                    "Authentication configuration required",
                )
            )

        if "authorization" not in data:
            self.warnings.append(
                ValidationError(
                    f"{path}.authorization",
                    "Authorization configuration not specified",
                    "WARNING",
                )
            )

        if "encryption" not in data:
            self.warnings.append(
                ValidationError(
                    f"{path}.encryption",
                    "Encryption configuration not specified",
                    "WARNING",
                )
            )

    def _validate_ai_config(self, data: Dict[str, Any], path: str) -> None:
        """Validate AI configuration for zero-hallucination compliance."""
        if "temperature" in data:
            temp = data["temperature"]
            if temp != 0.0:
                self.warnings.append(
                    ValidationError(
                        f"{path}.temperature",
                        f"Temperature {temp} != 0.0 may introduce non-determinism",
                        "WARNING",
                    )
                )

        if "zero_hallucination" in data:
            if not data["zero_hallucination"]:
                self.errors.append(
                    ValidationError(
                        f"{path}.zero_hallucination",
                        "Zero hallucination must be enabled for compliance",
                    )
                )

    def _validate_data_sources(self, data: Dict[str, Any], path: str) -> None:
        """Validate data source configurations."""
        if not data:
            self.warnings.append(
                ValidationError(f"{path}", "No data sources configured", "WARNING")
            )

    def _validate_alerting(self, data: Dict[str, Any], path: str) -> None:
        """Validate alerting configuration."""
        if "channels" not in data:
            self.warnings.append(
                ValidationError(
                    f"{path}.channels",
                    "No alert channels configured",
                    "WARNING",
                )
            )

        if "rules" not in data:
            self.warnings.append(
                ValidationError(
                    f"{path}.rules",
                    "No alert rules configured",
                    "WARNING",
                )
            )

    def _validate_performance(self, data: Dict[str, Any], path: str) -> None:
        """Validate performance targets."""
        if "latency" not in data:
            self.warnings.append(
                ValidationError(
                    f"{path}.latency",
                    "Latency targets not specified",
                    "WARNING",
                )
            )

        if "throughput" not in data:
            self.warnings.append(
                ValidationError(
                    f"{path}.throughput",
                    "Throughput targets not specified",
                    "WARNING",
                )
            )

    def _validate_run_spec(self, data: Dict[str, Any], path: str) -> None:
        """Validate run.json spec section."""
        if "mode" not in data and "modes" not in data:
            self.warnings.append(
                ValidationError(
                    f"{path}.mode",
                    "Operation mode not specified",
                    "WARNING",
                )
            )

    def print_results(self) -> None:
        """Print validation results."""
        if self.errors:
            print("\nERRORS:")
            for error in self.errors:
                print(f"  {error}")

        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")

        if not self.errors and not self.warnings:
            print("\nValidation passed with no errors or warnings.")
        elif not self.errors:
            print(f"\nValidation passed with {len(self.warnings)} warning(s).")
        else:
            print(f"\nValidation FAILED with {len(self.errors)} error(s) and {len(self.warnings)} warning(s).")


def load_yaml(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a YAML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse YAML file: {e}")
        return None
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return None


def load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON file: {e}")
        return None
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GreenLang configuration files"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Configuration file to validate (pack.yaml, gl.yaml, or run.json)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all configuration files",
    )

    args = parser.parse_args()

    validator = ConfigValidator()
    base_path = Path(__file__).parent

    files_to_validate = []

    if args.all:
        files_to_validate = [
            ("pack.yaml", base_path / "pack.yaml", validator.validate_pack_yaml, load_yaml),
            ("gl.yaml", base_path / "gl.yaml", validator.validate_gl_yaml, load_yaml),
            ("run.json", base_path / "run.json", validator.validate_run_json, load_json),
        ]
    elif args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = base_path / file_path

        if args.file.endswith(".yaml") or args.file.endswith(".yml"):
            if "pack" in args.file:
                files_to_validate = [(args.file, file_path, validator.validate_pack_yaml, load_yaml)]
            elif "gl" in args.file:
                files_to_validate = [(args.file, file_path, validator.validate_gl_yaml, load_yaml)]
            else:
                files_to_validate = [(args.file, file_path, validator.validate_pack_yaml, load_yaml)]
        elif args.file.endswith(".json"):
            files_to_validate = [(args.file, file_path, validator.validate_run_json, load_json)]
        else:
            print(f"ERROR: Unsupported file type: {args.file}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    all_passed = True

    for name, path, validate_func, load_func in files_to_validate:
        print(f"\n{'=' * 60}")
        print(f"Validating: {name}")
        print(f"{'=' * 60}")

        data = load_func(path)
        if data is None:
            all_passed = False
            continue

        passed = validate_func(data)
        validator.print_results()

        if not passed:
            all_passed = False

    print(f"\n{'=' * 60}")
    if all_passed:
        print("ALL VALIDATIONS PASSED")
        sys.exit(0)
    else:
        print("SOME VALIDATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
