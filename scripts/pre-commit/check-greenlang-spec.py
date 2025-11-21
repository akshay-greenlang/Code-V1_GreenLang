#!/usr/bin/env python3
"""
GreenLang Pack.yaml Validation Hook

Validates pack.yaml files against the GreenLang specification.
Ensures all agent configurations meet required standards.
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


class PackYamlValidator:
    """Validator for GreenLang pack.yaml files."""

    REQUIRED_FIELDS = {
        "name": str,
        "version": str,
        "description": str,
        "author": str,
        "type": str,
        "category": str,
    }

    VALID_TYPES = ["data_collector", "calculator", "reporter", "validator", "analyzer"]
    VALID_CATEGORIES = [
        "carbon",
        "energy",
        "water",
        "waste",
        "csrd",
        "scope1",
        "scope2",
        "scope3",
        "general",
    ]

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """Validate the pack.yaml file."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                self.errors.append("pack.yaml is empty")
                return False

            # Validate required fields
            self._validate_required_fields(data)

            # Validate field types
            self._validate_field_types(data)

            # Validate agent type
            self._validate_agent_type(data)

            # Validate category
            self._validate_category(data)

            # Validate version format
            self._validate_version(data)

            # Validate dependencies
            self._validate_dependencies(data)

            # Validate configuration schema
            self._validate_configuration(data)

            # Validate metadata
            self._validate_metadata(data)

            return len(self.errors) == 0

        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML syntax: {e}")
            return False
        except FileNotFoundError:
            self.errors.append(f"File not found: {self.file_path}")
            return False
        except Exception as e:
            self.errors.append(f"Unexpected error: {e}")
            return False

    def _validate_required_fields(self, data: Dict[str, Any]) -> None:
        """Validate that all required fields are present."""
        for field in self.REQUIRED_FIELDS:
            if field not in data:
                self.errors.append(f"Missing required field: {field}")

    def _validate_field_types(self, data: Dict[str, Any]) -> None:
        """Validate that fields have correct types."""
        for field, expected_type in self.REQUIRED_FIELDS.items():
            if field in data and not isinstance(data[field], expected_type):
                self.errors.append(
                    f"Field '{field}' must be of type {expected_type.__name__}"
                )

    def _validate_agent_type(self, data: Dict[str, Any]) -> None:
        """Validate agent type is valid."""
        if "type" in data and data["type"] not in self.VALID_TYPES:
            self.errors.append(
                f"Invalid agent type: {data['type']}. "
                f"Must be one of: {', '.join(self.VALID_TYPES)}"
            )

    def _validate_category(self, data: Dict[str, Any]) -> None:
        """Validate category is valid."""
        if "category" in data and data["category"] not in self.VALID_CATEGORIES:
            self.errors.append(
                f"Invalid category: {data['category']}. "
                f"Must be one of: {', '.join(self.VALID_CATEGORIES)}"
            )

    def _validate_version(self, data: Dict[str, Any]) -> None:
        """Validate version follows semantic versioning."""
        if "version" in data:
            version = data["version"]
            parts = version.split(".")
            if len(parts) != 3:
                self.errors.append(
                    f"Invalid version format: {version}. Must be semver (e.g., 1.0.0)"
                )
            else:
                for part in parts:
                    if not part.isdigit():
                        self.errors.append(
                            f"Invalid version format: {version}. "
                            "Parts must be numeric"
                        )
                        break

    def _validate_dependencies(self, data: Dict[str, Any]) -> None:
        """Validate dependencies structure."""
        if "dependencies" in data:
            deps = data["dependencies"]
            if not isinstance(deps, dict):
                self.errors.append("dependencies must be a dictionary")
                return

            for dep_name, dep_version in deps.items():
                if not isinstance(dep_version, str):
                    self.errors.append(
                        f"Dependency version for '{dep_name}' must be a string"
                    )

    def _validate_configuration(self, data: Dict[str, Any]) -> None:
        """Validate configuration schema."""
        if "configuration" in data:
            config = data["configuration"]
            if not isinstance(config, dict):
                self.errors.append("configuration must be a dictionary")
                return

            # Check for required configuration fields
            if "inputs" in config:
                if not isinstance(config["inputs"], list):
                    self.errors.append("configuration.inputs must be a list")
                else:
                    for idx, input_field in enumerate(config["inputs"]):
                        if not isinstance(input_field, dict):
                            self.errors.append(
                                f"configuration.inputs[{idx}] must be a dictionary"
                            )
                        else:
                            if "name" not in input_field:
                                self.errors.append(
                                    f"configuration.inputs[{idx}] missing 'name'"
                                )
                            if "type" not in input_field:
                                self.errors.append(
                                    f"configuration.inputs[{idx}] missing 'type'"
                                )

            if "outputs" in config:
                if not isinstance(config["outputs"], list):
                    self.errors.append("configuration.outputs must be a list")

    def _validate_metadata(self, data: Dict[str, Any]) -> None:
        """Validate metadata fields."""
        if "metadata" in data:
            metadata = data["metadata"]
            if not isinstance(metadata, dict):
                self.errors.append("metadata must be a dictionary")
                return

            # Optional but recommended fields
            recommended_fields = ["license", "homepage", "repository", "documentation"]
            for field in recommended_fields:
                if field not in metadata:
                    self.warnings.append(f"Recommended metadata field missing: {field}")

    def print_results(self) -> None:
        """Print validation results."""
        if self.errors:
            print(f"\n{'='*80}")
            print(f"VALIDATION FAILED: {self.file_path}")
            print(f"{'='*80}")
            for error in self.errors:
                print(f"ERROR: {error}")

        if self.warnings:
            print(f"\n{'='*80}")
            print(f"WARNINGS: {self.file_path}")
            print(f"{'='*80}")
            for warning in self.warnings:
                print(f"WARNING: {warning}")

        if not self.errors and not self.warnings:
            print(f"✓ {self.file_path} - Valid")


def main() -> int:
    """Main entry point for the hook."""
    if len(sys.argv) < 2:
        print("Usage: check-greenlang-spec.py <pack.yaml> [pack.yaml ...]")
        return 1

    all_valid = True
    for file_path in sys.argv[1:]:
        validator = PackYamlValidator(file_path)
        is_valid = validator.validate()
        validator.print_results()

        if not is_valid:
            all_valid = False

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
