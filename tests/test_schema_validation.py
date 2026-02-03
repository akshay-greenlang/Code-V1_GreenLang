"""
Comprehensive Schema Validation Tests for GreenLang Pack and Pipeline Manifests.

Tests all aspects of pack.yaml (pack-v1.0.schema.json) and gl.yaml (pipeline-v1.0.schema.json)
validation using the jsonschema library.

This module provides:
    1. Test fixtures for valid and invalid manifests
    2. Helper functions for loading schemas
    3. Comprehensive test cases for required fields, patterns, types, and nested structures
    4. Clear error messages on validation failures

Test Coverage:
    - Pack manifest validation (pack.yaml)
    - Pipeline definition validation (gl.yaml)
    - Field pattern validation (name, version)
    - Required field validation
    - Type validation
    - Nested structure validation (capabilities, policy, steps, retry)
    - Edge cases and boundary conditions

Usage:
    pytest tests/test_schema_validation.py -v
    pytest tests/test_schema_validation.py -v -k "pack"
    pytest tests/test_schema_validation.py -v -k "pipeline"
"""

import json
import copy
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import jsonschema, skip tests if not available
try:
    import jsonschema
    from jsonschema import Draft202012Validator, ValidationError, SchemaError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    Draft202012Validator = None
    ValidationError = Exception
    SchemaError = Exception


# =============================================================================
# Skip marker if jsonschema is not available
# =============================================================================

pytestmark = pytest.mark.skipif(
    not JSONSCHEMA_AVAILABLE,
    reason="jsonschema library not installed"
)


# =============================================================================
# Path Constants
# =============================================================================

GREENLANG_ROOT = Path(__file__).parent.parent
SCHEMAS_DIR = GREENLANG_ROOT / "greenlang" / "specs" / "schemas"
PACK_SCHEMA_PATH = SCHEMAS_DIR / "pack-v1.0.schema.json"
PIPELINE_SCHEMA_PATH = SCHEMAS_DIR / "pipeline-v1.0.schema.json"


# =============================================================================
# Valid Test Fixtures - Pack Manifests
# =============================================================================

VALID_PACK_MANIFEST_MINIMAL = {
    "name": "test-pack",
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0"
}

VALID_PACK_MANIFEST_FULL = {
    "name": "carbon-calculator-pack",
    "version": "2.1.0",
    "kind": "pack",
    "pack_schema_version": "1.0",
    "description": "A comprehensive carbon emissions calculation pack for enterprise use",
    "author": {
        "name": "GreenLang Team",
        "email": "team@greenlang.io",
        "organization": "GreenLang Foundation"
    },
    "license": "Apache-2.0",
    "contents": {
        "agents": [
            {
                "name": "fuel-combustion-agent",
                "class_path": "agents.fuel:FuelCombustionAgent",
                "description": "Calculates emissions from fuel combustion",
                "inputs": {
                    "fuel_type": "string",
                    "quantity": "number"
                },
                "outputs": {
                    "emissions_kg_co2e": "number"
                }
            }
        ],
        "pipelines": [
            "pipelines/main.yaml",
            {
                "name": "cbam-pipeline",
                "file": "pipelines/cbam.yaml",
                "description": "CBAM reporting pipeline"
            }
        ],
        "datasets": [
            {
                "name": "emission-factors",
                "path": "data/emission_factors.json",
                "format": "json",
                "card": "data/CARD.md",
                "size": "1.2MB"
            }
        ],
        "reports": ["templates/report.jinja2"]
    },
    "dependencies": [
        {"name": "greenlang-core", "version": ">=1.0.0"},
        "pandas>=2.0.0"
    ],
    "capabilities": {
        "fs": {
            "allow": True,
            "read": {
                "allowlist": ["./data/*", "./config/*"],
                "denylist": ["./secrets/*"]
            },
            "write": {
                "allowlist": ["./output/*"]
            }
        },
        "net": {
            "allow": True,
            "outbound": {
                "allowlist": ["api.greenlang.io", "*.database.azure.com"]
            }
        },
        "clock": {
            "allow": True
        },
        "subprocess": {
            "allow": False,
            "allowlist": ["python", "node"],
            "denylist": ["rm", "dd"]
        }
    },
    "policy": {
        "install": "default-install",
        "runtime": "strict-runtime",
        "network": ["https-only", "no-external-apis"],
        "data_residency": ["EU", "US"],
        "ef_vintage_min": 2020,
        "license_allowlist": ["MIT", "Apache-2.0", "BSD-3-Clause"]
    },
    "security": {
        "sbom": "sbom.spdx.json",
        "signatures": ["signatures/pack.sig"]
    },
    "provenance": {
        "sbom": True,
        "signing": True
    },
    "compat": {
        "greenlang": ">=1.0.0",
        "python": ">=3.9"
    },
    "metadata": {
        "authors": [
            {"name": "John Doe", "email": "john@greenlang.io"}
        ],
        "homepage": "https://greenlang.io/packs/carbon-calculator",
        "repository": "https://github.com/greenlang/carbon-calculator-pack",
        "publisher": "greenlang-foundation",
        "tags": ["carbon", "emissions", "ghg", "cbam"]
    },
    "tests": ["tests/**/*.py"],
    "test_command": "pytest tests/ -v",
    "card": "CARD.md",
    "min_greenlang_version": "1.0.0",
    "publisher": "greenlang-official"
}

VALID_PACK_MANIFEST_SINGLE_CHAR_NAME = {
    "name": "x",
    "version": "0.0.1",
    "kind": "pack",
    "pack_schema_version": "1.0"
}

VALID_PACK_MANIFEST_WITH_PRERELEASE = {
    "name": "alpha-pack",
    "version": "1.0.0-alpha.1+build.123",
    "kind": "pack",
    "pack_schema_version": "1.0"
}


# =============================================================================
# Valid Test Fixtures - Pipeline Definitions
# =============================================================================

VALID_PIPELINE_MINIMAL = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {
        "name": "test-pipeline"
    },
    "steps": [
        {
            "name": "step1",
            "agent": "greenlang/fuel-agent"
        }
    ]
}

VALID_PIPELINE_FULL = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {
        "name": "carbon-calculation-pipeline",
        "description": "Full carbon calculation workflow",
        "version": "1.0.0",
        "labels": {
            "environment": "production",
            "team": "sustainability"
        }
    },
    "parameters": [
        {
            "name": "input_file",
            "type": "string",
            "required": True,
            "description": "Path to input data file",
            "validation": {
                "pattern": "^.*\\.csv$"
            }
        },
        {
            "name": "threshold",
            "type": "number",
            "required": False,
            "default": 100.0,
            "description": "Emission threshold in kg CO2e",
            "validation": {
                "min": 0,
                "max": 10000
            }
        },
        {
            "name": "include_scope3",
            "type": "boolean",
            "required": False,
            "default": False,
            "description": "Include Scope 3 emissions"
        },
        {
            "name": "categories",
            "type": "array",
            "required": False,
            "default": ["energy", "transport"],
            "description": "Emission categories to process"
        },
        {
            "name": "options",
            "type": "object",
            "required": False,
            "description": "Additional processing options"
        }
    ],
    "steps": [
        {
            "name": "data-intake",
            "agent": "greenlang/data-intake-agent",
            "description": "Ingest and validate input data",
            "inputs": {
                "file_path": "${parameters.input_file}",
                "format": "csv"
            },
            "outputs": ["validated_data", "intake_report"],
            "timeout_seconds": 300,
            "on_failure": "stop",
            "deterministic": True
        },
        {
            "name": "emission-calc",
            "agent": "greenlang/emission-calculator",
            "description": "Calculate emissions from activity data",
            "inputs": {
                "data": "${steps.data-intake.validated_data}",
                "scope3": "${parameters.include_scope3}"
            },
            "outputs": ["emissions", "calculation_details"],
            "condition": "${steps.data-intake.intake_report.valid == true}",
            "retry": {
                "attempts": 3,
                "backoff": "exponential",
                "delay_seconds": 5,
                "max_delay_seconds": 60,
                "retry_on": ["RATE_LIMIT", "TIMEOUT"]
            },
            "timeout_seconds": 600,
            "on_failure": "continue",
            "resources": {
                "cpu": 2.0,
                "memory": "4Gi",
                "gpu": 0,
                "ephemeral_storage": "10Gi"
            },
            "deterministic": True
        },
        {
            "name": "generate-report",
            "agent": "greenlang/report-generator",
            "description": "Generate final emission report",
            "inputs": {
                "emissions": "${steps.emission-calc.emissions}",
                "threshold": "${parameters.threshold}"
            },
            "outputs": ["final_report"],
            "on_failure": "skip",
            "deterministic": False
        }
    ],
    "outputs": {
        "report": "${steps.generate-report.final_report}",
        "total_emissions": "${steps.emission-calc.emissions.total}"
    },
    "policy": [
        "greenlang/default-policy"
    ]
}

VALID_STEP_WITH_RETRY = {
    "name": "retry-step",
    "agent": "test-agent",
    "retry": {
        "attempts": 5,
        "backoff": "linear",
        "delay_seconds": 10,
        "max_delay_seconds": 120,
        "retry_on": ["CONNECTION_ERROR", "TIMEOUT"]
    }
}


# =============================================================================
# Invalid Test Fixtures - Pack Manifests
# =============================================================================

INVALID_PACK_MISSING_NAME = {
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0"
}

INVALID_PACK_MISSING_VERSION = {
    "name": "test-pack",
    "kind": "pack",
    "pack_schema_version": "1.0"
}

INVALID_PACK_MISSING_KIND = {
    "name": "test-pack",
    "version": "1.0.0",
    "pack_schema_version": "1.0"
}

INVALID_PACK_MISSING_SCHEMA_VERSION = {
    "name": "test-pack",
    "version": "1.0.0",
    "kind": "pack"
}

INVALID_PACK_WRONG_KIND = {
    "name": "test-pack",
    "version": "1.0.0",
    "kind": "pipeline",  # Should be "pack"
    "pack_schema_version": "1.0"
}

INVALID_PACK_WRONG_SCHEMA_VERSION = {
    "name": "test-pack",
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "2.0"  # Only "1.0" allowed
}

# Invalid name patterns
INVALID_PACK_NAME_UPPERCASE = {
    "name": "TestPack",  # Must be lowercase
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0"
}

INVALID_PACK_NAME_UNDERSCORE = {
    "name": "test_pack",  # Underscores not allowed
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0"
}

INVALID_PACK_NAME_STARTS_WITH_HYPHEN = {
    "name": "-test-pack",  # Cannot start with hyphen
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0"
}

INVALID_PACK_NAME_ENDS_WITH_HYPHEN = {
    "name": "test-pack-",  # Cannot end with hyphen
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0"
}

INVALID_PACK_NAME_SPECIAL_CHARS = {
    "name": "test@pack!",  # Special characters not allowed
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0"
}

# Invalid version formats
INVALID_PACK_VERSION_NOT_SEMVER = {
    "name": "test-pack",
    "version": "1.0",  # Missing patch version
    "kind": "pack",
    "pack_schema_version": "1.0"
}

INVALID_PACK_VERSION_LEADING_ZERO = {
    "name": "test-pack",
    "version": "01.0.0",  # Leading zero not allowed
    "kind": "pack",
    "pack_schema_version": "1.0"
}

INVALID_PACK_VERSION_TEXT = {
    "name": "test-pack",
    "version": "latest",  # Not a valid semver
    "kind": "pack",
    "pack_schema_version": "1.0"
}

# Invalid type
INVALID_PACK_VERSION_NUMBER = {
    "name": "test-pack",
    "version": 1.0,  # Should be string
    "kind": "pack",
    "pack_schema_version": "1.0"
}

# Invalid nested structures
INVALID_PACK_AUTHOR_MISSING_NAME = {
    "name": "test-pack",
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0",
    "author": {
        "email": "test@test.com"  # Missing required "name"
    }
}

INVALID_PACK_AGENT_MISSING_CLASS_PATH = {
    "name": "test-pack",
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0",
    "contents": {
        "agents": [
            {
                "name": "test-agent"
                # Missing required "class_path"
            }
        ]
    }
}

INVALID_PACK_AGENT_INVALID_CLASS_PATH = {
    "name": "test-pack",
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0",
    "contents": {
        "agents": [
            {
                "name": "test-agent",
                "class_path": "invalid_format"  # Must match pattern: module:Class
            }
        ]
    }
}

INVALID_PACK_DATASET_WRONG_FORMAT = {
    "name": "test-pack",
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0",
    "contents": {
        "datasets": [
            {
                "name": "test-data",
                "path": "data/test.txt",
                "format": "txt"  # Not in enum: json, csv, parquet, yaml
            }
        ]
    }
}

INVALID_PACK_POLICY_EF_VINTAGE_TOO_LOW = {
    "name": "test-pack",
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0",
    "policy": {
        "ef_vintage_min": 1999  # Minimum is 2000
    }
}

INVALID_PACK_POLICY_EF_VINTAGE_TOO_HIGH = {
    "name": "test-pack",
    "version": "1.0.0",
    "kind": "pack",
    "pack_schema_version": "1.0",
    "policy": {
        "ef_vintage_min": 2101  # Maximum is 2100
    }
}


# =============================================================================
# Invalid Test Fixtures - Pipeline Definitions
# =============================================================================

INVALID_PIPELINE_MISSING_API_VERSION = {
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [{"name": "s1", "agent": "a1"}]
}

INVALID_PIPELINE_MISSING_KIND = {
    "api_version": "glip/v1",
    "metadata": {"name": "test"},
    "steps": [{"name": "s1", "agent": "a1"}]
}

INVALID_PIPELINE_MISSING_METADATA = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "steps": [{"name": "s1", "agent": "a1"}]
}

INVALID_PIPELINE_MISSING_STEPS = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"}
}

INVALID_PIPELINE_EMPTY_STEPS = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": []  # minItems: 1
}

INVALID_PIPELINE_WRONG_API_VERSION = {
    "api_version": "glip/v2",  # Only "glip/v1" allowed
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [{"name": "s1", "agent": "a1"}]
}

INVALID_PIPELINE_WRONG_KIND = {
    "api_version": "glip/v1",
    "kind": "Workflow",  # Must be "Pipeline"
    "metadata": {"name": "test"},
    "steps": [{"name": "s1", "agent": "a1"}]
}

INVALID_PIPELINE_METADATA_MISSING_NAME = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {
        "description": "Test pipeline"  # Missing required "name"
    },
    "steps": [{"name": "s1", "agent": "a1"}]
}

INVALID_PIPELINE_STEP_MISSING_NAME = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [
        {"agent": "test-agent"}  # Missing required "name"
    ]
}

INVALID_PIPELINE_STEP_MISSING_AGENT = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [
        {"name": "step1"}  # Missing required "agent"
    ]
}

INVALID_PIPELINE_STEP_INVALID_NAME = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [
        {
            "name": "Step_1",  # Invalid: uppercase and underscore
            "agent": "test-agent"
        }
    ]
}

INVALID_PIPELINE_STEP_INVALID_ON_FAILURE = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [
        {
            "name": "step1",
            "agent": "test-agent",
            "on_failure": "retry"  # Must be: stop, skip, continue
        }
    ]
}

INVALID_PIPELINE_RETRY_ATTEMPTS_TOO_HIGH = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [
        {
            "name": "step1",
            "agent": "test-agent",
            "retry": {
                "attempts": 15  # Maximum is 10
            }
        }
    ]
}

INVALID_PIPELINE_RETRY_ATTEMPTS_TOO_LOW = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [
        {
            "name": "step1",
            "agent": "test-agent",
            "retry": {
                "attempts": 0  # Minimum is 1
            }
        }
    ]
}

INVALID_PIPELINE_RETRY_INVALID_BACKOFF = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [
        {
            "name": "step1",
            "agent": "test-agent",
            "retry": {
                "backoff": "random"  # Must be: none, linear, exponential
            }
        }
    ]
}

INVALID_PIPELINE_TIMEOUT_TOO_LOW = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [
        {
            "name": "step1",
            "agent": "test-agent",
            "timeout_seconds": 0  # Minimum is 1
        }
    ]
}

INVALID_PIPELINE_TIMEOUT_TOO_HIGH = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [
        {
            "name": "step1",
            "agent": "test-agent",
            "timeout_seconds": 100000  # Maximum is 86400 (24 hours)
        }
    ]
}

INVALID_PIPELINE_RESOURCE_CPU_TOO_LOW = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [
        {
            "name": "step1",
            "agent": "test-agent",
            "resources": {
                "cpu": 0.05  # Minimum is 0.1
            }
        }
    ]
}

INVALID_PIPELINE_RESOURCE_MEMORY_FORMAT = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [
        {
            "name": "step1",
            "agent": "test-agent",
            "resources": {
                "memory": "4GB"  # Must be: #Mi or #Gi
            }
        }
    ]
}

INVALID_PIPELINE_PARAMETER_INVALID_TYPE = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "parameters": [
        {
            "name": "param1",
            "type": "integer"  # Must be: string, number, boolean, array, object
        }
    ],
    "steps": [{"name": "s1", "agent": "a1"}]
}

INVALID_PIPELINE_POLICY_RULE_INVALID_ACTION = {
    "api_version": "glip/v1",
    "kind": "Pipeline",
    "metadata": {"name": "test"},
    "steps": [{"name": "s1", "agent": "a1"}],
    "policy": [
        {
            "name": "test-policy",
            "rules": [
                {
                    "rule": "test-rule",
                    "action": "block"  # Must be: allow, deny, warn
                }
            ],
            "inline": {
                "name": "test-policy",
                "rules": [{"rule": "test-rule", "action": "block"}]
            }
        }
    ]
}


# =============================================================================
# Schema Loading Helpers
# =============================================================================

def load_schema(schema_path: Path) -> Dict[str, Any]:
    """
    Load a JSON Schema from file.

    Args:
        schema_path: Path to the schema file.

    Returns:
        Parsed JSON schema as dictionary.

    Raises:
        FileNotFoundError: If schema file does not exist.
        json.JSONDecodeError: If schema file is not valid JSON.
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(
    manifest: Dict[str, Any],
    schema: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Validate a manifest against a JSON Schema.

    Args:
        manifest: The manifest data to validate.
        schema: The JSON Schema to validate against.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(manifest))

    if not errors:
        return True, []

    error_messages = []
    for error in errors:
        path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "(root)"
        error_messages.append(f"[{path}] {error.message}")

    return False, error_messages


def get_validation_error_paths(
    manifest: Dict[str, Any],
    schema: Dict[str, Any]
) -> List[str]:
    """
    Get list of JSON paths where validation errors occurred.

    Args:
        manifest: The manifest data to validate.
        schema: The JSON Schema to validate against.

    Returns:
        List of JSON paths with errors.
    """
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(manifest))

    paths = []
    for error in errors:
        path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "(root)"
        paths.append(path)

    return paths


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def pack_schema() -> Dict[str, Any]:
    """Load the pack manifest schema."""
    return load_schema(PACK_SCHEMA_PATH)


@pytest.fixture(scope="module")
def pipeline_schema() -> Dict[str, Any]:
    """Load the pipeline definition schema."""
    return load_schema(PIPELINE_SCHEMA_PATH)


@pytest.fixture
def pack_validator(pack_schema) -> Draft202012Validator:
    """Create a validator for pack manifests."""
    return Draft202012Validator(pack_schema)


@pytest.fixture
def pipeline_validator(pipeline_schema) -> Draft202012Validator:
    """Create a validator for pipeline definitions."""
    return Draft202012Validator(pipeline_schema)


# =============================================================================
# Pack Manifest Tests - Valid Manifests
# =============================================================================

class TestValidPackManifests:
    """Test cases for valid pack manifests."""

    def test_valid_pack_manifest_minimal_passes(self, pack_schema):
        """Test that minimal valid pack manifest passes validation."""
        is_valid, errors = validate_manifest(VALID_PACK_MANIFEST_MINIMAL, pack_schema)

        assert is_valid, f"Validation failed with errors: {errors}"
        assert len(errors) == 0

    def test_valid_pack_manifest_full_passes(self, pack_schema):
        """Test that full valid pack manifest passes validation."""
        is_valid, errors = validate_manifest(VALID_PACK_MANIFEST_FULL, pack_schema)

        assert is_valid, f"Validation failed with errors: {errors}"
        assert len(errors) == 0

    def test_valid_pack_manifest_single_char_name(self, pack_schema):
        """Test that single character name is valid."""
        is_valid, errors = validate_manifest(VALID_PACK_MANIFEST_SINGLE_CHAR_NAME, pack_schema)

        assert is_valid, f"Validation failed with errors: {errors}"

    def test_valid_pack_manifest_with_prerelease(self, pack_schema):
        """Test that semver with prerelease and build metadata is valid."""
        is_valid, errors = validate_manifest(VALID_PACK_MANIFEST_WITH_PRERELEASE, pack_schema)

        assert is_valid, f"Validation failed with errors: {errors}"

    @pytest.mark.parametrize("name", [
        "a",
        "ab",
        "test-pack",
        "my-great-pack",
        "pack123",
        "123pack",
        "a1b2c3",
    ])
    def test_valid_pack_names(self, pack_schema, name):
        """Test various valid pack name patterns."""
        manifest = {
            "name": name,
            "version": "1.0.0",
            "kind": "pack",
            "pack_schema_version": "1.0"
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert is_valid, f"Name '{name}' should be valid but got errors: {errors}"

    @pytest.mark.parametrize("version", [
        "0.0.1",
        "1.0.0",
        "10.20.30",
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-beta+build.123",
        "1.0.0+20130313144700",
        "1.0.0-rc.1+build.123",
    ])
    def test_valid_semver_versions(self, pack_schema, version):
        """Test various valid semantic version formats."""
        manifest = {
            "name": "test-pack",
            "version": version,
            "kind": "pack",
            "pack_schema_version": "1.0"
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert is_valid, f"Version '{version}' should be valid but got errors: {errors}"


# =============================================================================
# Pack Manifest Tests - Missing Required Fields
# =============================================================================

class TestPackMissingRequiredFields:
    """Test cases for pack manifests with missing required fields."""

    def test_missing_required_field_name_fails(self, pack_schema):
        """Test that missing 'name' field causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PACK_MISSING_NAME, pack_schema)

        assert not is_valid, "Validation should fail when 'name' is missing"
        assert any("'name'" in e for e in errors), f"Error should mention 'name': {errors}"

    def test_missing_required_field_version_fails(self, pack_schema):
        """Test that missing 'version' field causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PACK_MISSING_VERSION, pack_schema)

        assert not is_valid, "Validation should fail when 'version' is missing"
        assert any("'version'" in e for e in errors), f"Error should mention 'version': {errors}"

    def test_missing_required_field_kind_fails(self, pack_schema):
        """Test that missing 'kind' field causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PACK_MISSING_KIND, pack_schema)

        assert not is_valid, "Validation should fail when 'kind' is missing"
        assert any("'kind'" in e for e in errors), f"Error should mention 'kind': {errors}"

    def test_missing_required_field_schema_version_fails(self, pack_schema):
        """Test that missing 'pack_schema_version' field causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PACK_MISSING_SCHEMA_VERSION, pack_schema)

        assert not is_valid, "Validation should fail when 'pack_schema_version' is missing"
        assert any("'pack_schema_version'" in e for e in errors), \
            f"Error should mention 'pack_schema_version': {errors}"


# =============================================================================
# Pack Manifest Tests - Invalid Values
# =============================================================================

class TestPackInvalidValues:
    """Test cases for pack manifests with invalid values."""

    def test_invalid_kind_fails(self, pack_schema):
        """Test that wrong 'kind' value causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PACK_WRONG_KIND, pack_schema)

        assert not is_valid, "Validation should fail when 'kind' is not 'pack'"
        assert any("kind" in e.lower() for e in errors), f"Error should mention 'kind': {errors}"

    def test_invalid_schema_version_fails(self, pack_schema):
        """Test that invalid 'pack_schema_version' causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PACK_WRONG_SCHEMA_VERSION, pack_schema)

        assert not is_valid, "Validation should fail for unsupported schema version"

    def test_invalid_version_format_not_semver(self, pack_schema):
        """Test that non-semver version format fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_VERSION_NOT_SEMVER, pack_schema)

        assert not is_valid, "Validation should fail for non-semver version"

    def test_invalid_version_format_leading_zero(self, pack_schema):
        """Test that version with leading zero fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_VERSION_LEADING_ZERO, pack_schema)

        assert not is_valid, "Validation should fail for version with leading zero"

    def test_invalid_version_format_text(self, pack_schema):
        """Test that text version (like 'latest') fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_VERSION_TEXT, pack_schema)

        assert not is_valid, "Validation should fail for text version"

    def test_invalid_version_type_number(self, pack_schema):
        """Test that numeric version type fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_VERSION_NUMBER, pack_schema)

        assert not is_valid, "Validation should fail when version is a number"


# =============================================================================
# Pack Manifest Tests - Invalid Name Patterns
# =============================================================================

class TestPackInvalidNamePatterns:
    """Test cases for pack manifests with invalid name patterns."""

    def test_invalid_name_pattern_uppercase(self, pack_schema):
        """Test that uppercase name fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_NAME_UPPERCASE, pack_schema)

        assert not is_valid, "Validation should fail for uppercase name"

    def test_invalid_name_pattern_underscore(self, pack_schema):
        """Test that name with underscore fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_NAME_UNDERSCORE, pack_schema)

        assert not is_valid, "Validation should fail for name with underscore"

    def test_invalid_name_pattern_starts_with_hyphen(self, pack_schema):
        """Test that name starting with hyphen fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_NAME_STARTS_WITH_HYPHEN, pack_schema)

        assert not is_valid, "Validation should fail for name starting with hyphen"

    def test_invalid_name_pattern_ends_with_hyphen(self, pack_schema):
        """Test that name ending with hyphen fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_NAME_ENDS_WITH_HYPHEN, pack_schema)

        assert not is_valid, "Validation should fail for name ending with hyphen"

    def test_invalid_name_pattern_special_chars(self, pack_schema):
        """Test that name with special characters fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_NAME_SPECIAL_CHARS, pack_schema)

        assert not is_valid, "Validation should fail for name with special characters"


# =============================================================================
# Pack Manifest Tests - Nested Structures
# =============================================================================

class TestPackNestedStructures:
    """Test cases for pack manifest nested structures."""

    def test_author_missing_required_name_fails(self, pack_schema):
        """Test that author without name fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_AUTHOR_MISSING_NAME, pack_schema)

        assert not is_valid, "Validation should fail when author.name is missing"

    def test_agent_missing_class_path_fails(self, pack_schema):
        """Test that agent without class_path fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_AGENT_MISSING_CLASS_PATH, pack_schema)

        assert not is_valid, "Validation should fail when agent.class_path is missing"

    def test_agent_invalid_class_path_format_fails(self, pack_schema):
        """Test that agent with invalid class_path format fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_AGENT_INVALID_CLASS_PATH, pack_schema)

        assert not is_valid, "Validation should fail for invalid class_path format"

    def test_dataset_invalid_format_enum_fails(self, pack_schema):
        """Test that dataset with invalid format enum fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_DATASET_WRONG_FORMAT, pack_schema)

        assert not is_valid, "Validation should fail for invalid dataset format"


# =============================================================================
# Pack Manifest Tests - Policy Validation
# =============================================================================

class TestPackPolicyValidation:
    """Test cases for pack manifest policy validation."""

    def test_policy_ef_vintage_min_too_low_fails(self, pack_schema):
        """Test that ef_vintage_min below 2000 fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_POLICY_EF_VINTAGE_TOO_LOW, pack_schema)

        assert not is_valid, "Validation should fail when ef_vintage_min < 2000"

    def test_policy_ef_vintage_min_too_high_fails(self, pack_schema):
        """Test that ef_vintage_min above 2100 fails validation."""
        is_valid, errors = validate_manifest(INVALID_PACK_POLICY_EF_VINTAGE_TOO_HIGH, pack_schema)

        assert not is_valid, "Validation should fail when ef_vintage_min > 2100"

    def test_policy_valid_ef_vintage_min_passes(self, pack_schema):
        """Test that valid ef_vintage_min passes validation."""
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "pack_schema_version": "1.0",
            "policy": {
                "ef_vintage_min": 2020
            }
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert is_valid, f"Validation failed with errors: {errors}"


# =============================================================================
# Pack Manifest Tests - Capability Validation
# =============================================================================

class TestPackCapabilityValidation:
    """Test cases for pack manifest capability validation."""

    def test_capability_fs_valid_passes(self, pack_schema):
        """Test that valid fs capability passes validation."""
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "pack_schema_version": "1.0",
            "capabilities": {
                "fs": {
                    "allow": True,
                    "read": {
                        "allowlist": ["./data/*"],
                        "denylist": ["./secrets/*"]
                    }
                }
            }
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert is_valid, f"Validation failed with errors: {errors}"

    def test_capability_net_valid_passes(self, pack_schema):
        """Test that valid net capability passes validation."""
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "pack_schema_version": "1.0",
            "capabilities": {
                "net": {
                    "allow": True,
                    "outbound": {
                        "allowlist": ["api.example.com"]
                    }
                }
            }
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert is_valid, f"Validation failed with errors: {errors}"

    def test_capability_subprocess_valid_passes(self, pack_schema):
        """Test that valid subprocess capability passes validation."""
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "pack_schema_version": "1.0",
            "capabilities": {
                "subprocess": {
                    "allow": True,
                    "allowlist": ["python", "node"],
                    "denylist": ["rm", "dd"]
                }
            }
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert is_valid, f"Validation failed with errors: {errors}"

    def test_capability_clock_valid_passes(self, pack_schema):
        """Test that valid clock capability passes validation."""
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "pack_schema_version": "1.0",
            "capabilities": {
                "clock": {
                    "allow": True
                }
            }
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert is_valid, f"Validation failed with errors: {errors}"


# =============================================================================
# Pipeline Definition Tests - Valid Pipelines
# =============================================================================

class TestValidPipelineDefinitions:
    """Test cases for valid pipeline definitions."""

    def test_valid_pipeline_definition_minimal_passes(self, pipeline_schema):
        """Test that minimal valid pipeline passes validation."""
        is_valid, errors = validate_manifest(VALID_PIPELINE_MINIMAL, pipeline_schema)

        assert is_valid, f"Validation failed with errors: {errors}"
        assert len(errors) == 0

    def test_valid_pipeline_definition_full_passes(self, pipeline_schema):
        """Test that full valid pipeline passes validation."""
        is_valid, errors = validate_manifest(VALID_PIPELINE_FULL, pipeline_schema)

        assert is_valid, f"Validation failed with errors: {errors}"
        assert len(errors) == 0

    @pytest.mark.parametrize("name", [
        "a",
        "step1",
        "data-intake",
        "emission-calc-v2",
        "step123",
    ])
    def test_valid_step_names(self, pipeline_schema, name):
        """Test various valid step name patterns."""
        pipeline = {
            "api_version": "glip/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "steps": [{"name": name, "agent": "test-agent"}]
        }
        is_valid, errors = validate_manifest(pipeline, pipeline_schema)

        assert is_valid, f"Step name '{name}' should be valid but got errors: {errors}"


# =============================================================================
# Pipeline Definition Tests - Missing Required Fields
# =============================================================================

class TestPipelineMissingRequiredFields:
    """Test cases for pipeline definitions with missing required fields."""

    def test_invalid_pipeline_missing_api_version_fails(self, pipeline_schema):
        """Test that missing api_version causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_MISSING_API_VERSION, pipeline_schema)

        assert not is_valid, "Validation should fail when api_version is missing"

    def test_invalid_pipeline_missing_kind_fails(self, pipeline_schema):
        """Test that missing kind causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_MISSING_KIND, pipeline_schema)

        assert not is_valid, "Validation should fail when kind is missing"

    def test_invalid_pipeline_missing_metadata_fails(self, pipeline_schema):
        """Test that missing metadata causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_MISSING_METADATA, pipeline_schema)

        assert not is_valid, "Validation should fail when metadata is missing"

    def test_invalid_pipeline_missing_steps_fails(self, pipeline_schema):
        """Test that missing steps causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_MISSING_STEPS, pipeline_schema)

        assert not is_valid, "Validation should fail when steps is missing"

    def test_invalid_pipeline_empty_steps_fails(self, pipeline_schema):
        """Test that empty steps array causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_EMPTY_STEPS, pipeline_schema)

        assert not is_valid, "Validation should fail when steps array is empty"


# =============================================================================
# Pipeline Definition Tests - Invalid Values
# =============================================================================

class TestPipelineInvalidValues:
    """Test cases for pipeline definitions with invalid values."""

    def test_invalid_pipeline_wrong_api_version_fails(self, pipeline_schema):
        """Test that wrong api_version causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_WRONG_API_VERSION, pipeline_schema)

        assert not is_valid, "Validation should fail for wrong api_version"

    def test_invalid_pipeline_wrong_kind_fails(self, pipeline_schema):
        """Test that wrong kind causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_WRONG_KIND, pipeline_schema)

        assert not is_valid, "Validation should fail for wrong kind"

    def test_invalid_pipeline_metadata_missing_name_fails(self, pipeline_schema):
        """Test that metadata without name causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_METADATA_MISSING_NAME, pipeline_schema)

        assert not is_valid, "Validation should fail when metadata.name is missing"


# =============================================================================
# Pipeline Definition Tests - Step Validation
# =============================================================================

class TestPipelineStepValidation:
    """Test cases for pipeline step validation."""

    def test_invalid_step_missing_name_fails(self, pipeline_schema):
        """Test that step without name causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_STEP_MISSING_NAME, pipeline_schema)

        assert not is_valid, "Validation should fail when step.name is missing"

    def test_invalid_step_missing_agent_fails(self, pipeline_schema):
        """Test that step without agent causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_STEP_MISSING_AGENT, pipeline_schema)

        assert not is_valid, "Validation should fail when step.agent is missing"

    def test_invalid_step_name_pattern_fails(self, pipeline_schema):
        """Test that step with invalid name pattern fails validation."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_STEP_INVALID_NAME, pipeline_schema)

        assert not is_valid, "Validation should fail for invalid step name pattern"

    def test_invalid_step_on_failure_value_fails(self, pipeline_schema):
        """Test that step with invalid on_failure value fails validation."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_STEP_INVALID_ON_FAILURE, pipeline_schema)

        assert not is_valid, "Validation should fail for invalid on_failure value"


# =============================================================================
# Pipeline Definition Tests - Retry Configuration
# =============================================================================

class TestPipelineRetryConfiguration:
    """Test cases for pipeline step retry configuration."""

    def test_valid_step_with_retry_config(self, pipeline_schema):
        """Test that step with valid retry configuration passes validation."""
        pipeline = {
            "api_version": "glip/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "steps": [VALID_STEP_WITH_RETRY]
        }
        is_valid, errors = validate_manifest(pipeline, pipeline_schema)

        assert is_valid, f"Validation failed with errors: {errors}"

    def test_invalid_retry_attempts_too_high_fails(self, pipeline_schema):
        """Test that retry attempts > 10 causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_RETRY_ATTEMPTS_TOO_HIGH, pipeline_schema)

        assert not is_valid, "Validation should fail when retry.attempts > 10"

    def test_invalid_retry_attempts_too_low_fails(self, pipeline_schema):
        """Test that retry attempts < 1 causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_RETRY_ATTEMPTS_TOO_LOW, pipeline_schema)

        assert not is_valid, "Validation should fail when retry.attempts < 1"

    def test_invalid_retry_backoff_value_fails(self, pipeline_schema):
        """Test that invalid backoff value causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_RETRY_INVALID_BACKOFF, pipeline_schema)

        assert not is_valid, "Validation should fail for invalid backoff value"

    @pytest.mark.parametrize("backoff", ["none", "linear", "exponential"])
    def test_valid_retry_backoff_values(self, pipeline_schema, backoff):
        """Test all valid backoff enum values."""
        pipeline = {
            "api_version": "glip/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "steps": [{
                "name": "step1",
                "agent": "test-agent",
                "retry": {"backoff": backoff}
            }]
        }
        is_valid, errors = validate_manifest(pipeline, pipeline_schema)

        assert is_valid, f"Backoff '{backoff}' should be valid but got errors: {errors}"


# =============================================================================
# Pipeline Definition Tests - Timeout and Resources
# =============================================================================

class TestPipelineTimeoutAndResources:
    """Test cases for pipeline step timeout and resource configuration."""

    def test_invalid_timeout_too_low_fails(self, pipeline_schema):
        """Test that timeout_seconds < 1 causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_TIMEOUT_TOO_LOW, pipeline_schema)

        assert not is_valid, "Validation should fail when timeout_seconds < 1"

    def test_invalid_timeout_too_high_fails(self, pipeline_schema):
        """Test that timeout_seconds > 86400 causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_TIMEOUT_TOO_HIGH, pipeline_schema)

        assert not is_valid, "Validation should fail when timeout_seconds > 86400"

    def test_invalid_resource_cpu_too_low_fails(self, pipeline_schema):
        """Test that cpu < 0.1 causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_RESOURCE_CPU_TOO_LOW, pipeline_schema)

        assert not is_valid, "Validation should fail when cpu < 0.1"

    def test_invalid_resource_memory_format_fails(self, pipeline_schema):
        """Test that invalid memory format causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_RESOURCE_MEMORY_FORMAT, pipeline_schema)

        assert not is_valid, "Validation should fail for invalid memory format"

    @pytest.mark.parametrize("memory", ["512Mi", "1Gi", "4Gi", "16Gi"])
    def test_valid_memory_formats(self, pipeline_schema, memory):
        """Test valid memory format values."""
        pipeline = {
            "api_version": "glip/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "steps": [{
                "name": "step1",
                "agent": "test-agent",
                "resources": {"memory": memory}
            }]
        }
        is_valid, errors = validate_manifest(pipeline, pipeline_schema)

        assert is_valid, f"Memory '{memory}' should be valid but got errors: {errors}"


# =============================================================================
# Pipeline Definition Tests - Parameters
# =============================================================================

class TestPipelineParameters:
    """Test cases for pipeline parameter validation."""

    def test_invalid_parameter_type_fails(self, pipeline_schema):
        """Test that invalid parameter type causes validation failure."""
        is_valid, errors = validate_manifest(INVALID_PIPELINE_PARAMETER_INVALID_TYPE, pipeline_schema)

        assert not is_valid, "Validation should fail for invalid parameter type"

    @pytest.mark.parametrize("param_type", ["string", "number", "boolean", "array", "object"])
    def test_valid_parameter_types(self, pipeline_schema, param_type):
        """Test all valid parameter type values."""
        pipeline = {
            "api_version": "glip/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "parameters": [
                {"name": "param1", "type": param_type}
            ],
            "steps": [{"name": "s1", "agent": "a1"}]
        }
        is_valid, errors = validate_manifest(pipeline, pipeline_schema)

        assert is_valid, f"Parameter type '{param_type}' should be valid but got errors: {errors}"


# =============================================================================
# Pipeline Definition Tests - Policy Validation
# =============================================================================

class TestPipelinePolicyValidation:
    """Test cases for pipeline policy validation."""

    def test_valid_string_policy_reference(self, pipeline_schema):
        """Test that string policy reference is valid."""
        pipeline = {
            "api_version": "glip/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "steps": [{"name": "s1", "agent": "a1"}],
            "policy": ["greenlang/default-policy", "greenlang/strict-policy"]
        }
        is_valid, errors = validate_manifest(pipeline, pipeline_schema)

        assert is_valid, f"String policy references should be valid but got errors: {errors}"

    def test_empty_policy_array_valid(self, pipeline_schema):
        """Test that empty policy array is valid."""
        pipeline = {
            "api_version": "glip/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "steps": [{"name": "s1", "agent": "a1"}],
            "policy": []
        }
        is_valid, errors = validate_manifest(pipeline, pipeline_schema)

        assert is_valid, f"Empty policy array should be valid but got errors: {errors}"

    def test_no_policy_field_valid(self, pipeline_schema):
        """Test that missing policy field is valid (optional)."""
        pipeline = {
            "api_version": "glip/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "steps": [{"name": "s1", "agent": "a1"}]
        }
        is_valid, errors = validate_manifest(pipeline, pipeline_schema)

        assert is_valid, f"Missing policy field should be valid but got errors: {errors}"


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pack_name_max_length_valid(self, pack_schema):
        """Test pack name at maximum length (64 chars)."""
        long_name = "a" * 64
        manifest = {
            "name": long_name,
            "version": "1.0.0",
            "kind": "pack",
            "pack_schema_version": "1.0"
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert is_valid, f"Name at max length should be valid: {errors}"

    def test_pack_name_exceeds_max_length_fails(self, pack_schema):
        """Test pack name exceeding maximum length (65 chars)."""
        long_name = "a" * 65
        manifest = {
            "name": long_name,
            "version": "1.0.0",
            "kind": "pack",
            "pack_schema_version": "1.0"
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert not is_valid, "Name exceeding max length should fail"

    def test_pack_description_max_length_valid(self, pack_schema):
        """Test pack description at maximum length (1000 chars)."""
        long_desc = "a" * 1000
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "pack_schema_version": "1.0",
            "description": long_desc
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert is_valid, f"Description at max length should be valid: {errors}"

    def test_pack_description_exceeds_max_length_fails(self, pack_schema):
        """Test pack description exceeding maximum length."""
        long_desc = "a" * 1001
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "pack_schema_version": "1.0",
            "description": long_desc
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert not is_valid, "Description exceeding max length should fail"

    def test_pipeline_timeout_at_minimum_valid(self, pipeline_schema):
        """Test step timeout at minimum value (1 second)."""
        pipeline = {
            "api_version": "glip/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "steps": [{
                "name": "step1",
                "agent": "test-agent",
                "timeout_seconds": 1
            }]
        }
        is_valid, errors = validate_manifest(pipeline, pipeline_schema)

        assert is_valid, f"Timeout at minimum should be valid: {errors}"

    def test_pipeline_timeout_at_maximum_valid(self, pipeline_schema):
        """Test step timeout at maximum value (86400 seconds / 24 hours)."""
        pipeline = {
            "api_version": "glip/v1",
            "kind": "Pipeline",
            "metadata": {"name": "test"},
            "steps": [{
                "name": "step1",
                "agent": "test-agent",
                "timeout_seconds": 86400
            }]
        }
        is_valid, errors = validate_manifest(pipeline, pipeline_schema)

        assert is_valid, f"Timeout at maximum should be valid: {errors}"

    def test_pipeline_retry_attempts_at_boundary(self, pipeline_schema):
        """Test retry attempts at boundary values (1 and 10)."""
        for attempts in [1, 10]:
            pipeline = {
                "api_version": "glip/v1",
                "kind": "Pipeline",
                "metadata": {"name": "test"},
                "steps": [{
                    "name": "step1",
                    "agent": "test-agent",
                    "retry": {"attempts": attempts}
                }]
            }
            is_valid, errors = validate_manifest(pipeline, pipeline_schema)

            assert is_valid, f"Retry attempts={attempts} should be valid: {errors}"

    def test_empty_capabilities_valid(self, pack_schema):
        """Test empty capabilities object is valid."""
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "pack_schema_version": "1.0",
            "capabilities": {}
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert is_valid, f"Empty capabilities should be valid: {errors}"

    def test_empty_policy_valid(self, pack_schema):
        """Test empty policy object is valid."""
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "pack_schema_version": "1.0",
            "policy": {}
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert is_valid, f"Empty policy should be valid: {errors}"


# =============================================================================
# Schema Loading Tests
# =============================================================================

class TestSchemaLoading:
    """Test schema loading functionality."""

    def test_pack_schema_loads_successfully(self):
        """Test that pack schema file loads successfully."""
        schema = load_schema(PACK_SCHEMA_PATH)

        assert schema is not None
        assert "$schema" in schema
        assert schema["title"] == "GreenLang Pack Manifest"

    def test_pipeline_schema_loads_successfully(self):
        """Test that pipeline schema file loads successfully."""
        schema = load_schema(PIPELINE_SCHEMA_PATH)

        assert schema is not None
        assert "$schema" in schema
        assert schema["title"] == "GreenLang Pipeline Definition"

    def test_pack_schema_is_valid_json_schema(self):
        """Test that pack schema is a valid JSON Schema."""
        schema = load_schema(PACK_SCHEMA_PATH)

        # This should not raise an error
        Draft202012Validator.check_schema(schema)

    def test_pipeline_schema_is_valid_json_schema(self):
        """Test that pipeline schema is a valid JSON Schema."""
        schema = load_schema(PIPELINE_SCHEMA_PATH)

        # This should not raise an error
        Draft202012Validator.check_schema(schema)

    def test_schema_not_found_raises_error(self):
        """Test that loading non-existent schema raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_schema(Path("/nonexistent/schema.json"))


# =============================================================================
# Error Message Quality Tests
# =============================================================================

class TestErrorMessageQuality:
    """Test that validation errors provide clear, actionable messages."""

    def test_missing_field_error_specifies_field_name(self, pack_schema):
        """Test that missing field errors clearly specify which field is missing."""
        is_valid, errors = validate_manifest(INVALID_PACK_MISSING_NAME, pack_schema)

        assert not is_valid
        # Error should mention the missing field
        error_text = " ".join(errors).lower()
        assert "name" in error_text, f"Error should mention 'name': {errors}"

    def test_type_error_specifies_expected_type(self, pack_schema):
        """Test that type errors clearly specify the expected type."""
        is_valid, errors = validate_manifest(INVALID_PACK_VERSION_NUMBER, pack_schema)

        assert not is_valid
        # Error should mention expected type
        error_text = " ".join(errors).lower()
        assert "string" in error_text or "type" in error_text, \
            f"Error should mention expected type: {errors}"

    def test_pattern_error_identifies_field(self, pack_schema):
        """Test that pattern errors identify which field failed."""
        is_valid, errors = validate_manifest(INVALID_PACK_NAME_UPPERCASE, pack_schema)

        assert not is_valid
        error_text = " ".join(errors).lower()
        assert "name" in error_text or "pattern" in error_text, \
            f"Error should identify the field or mention pattern: {errors}"

    def test_nested_field_error_shows_path(self, pack_schema):
        """Test that errors in nested fields show the full path."""
        is_valid, errors = validate_manifest(INVALID_PACK_AUTHOR_MISSING_NAME, pack_schema)

        assert not is_valid
        error_text = " ".join(errors)
        # Should show path like "author" or "author.name"
        assert "author" in error_text.lower(), \
            f"Error should show path to nested field: {errors}"


# =============================================================================
# Comprehensive Validation Tests
# =============================================================================

class TestComprehensiveValidation:
    """Comprehensive tests combining multiple validation scenarios."""

    def test_multiple_validation_errors_all_reported(self, pack_schema):
        """Test that when multiple fields are invalid, all errors are reported."""
        manifest = {
            "name": "INVALID-NAME",  # Invalid: uppercase
            "version": "invalid",    # Invalid: not semver
            "kind": "wrong",         # Invalid: not 'pack'
            "pack_schema_version": "3.0"  # Invalid: not '1.0'
        }
        is_valid, errors = validate_manifest(manifest, pack_schema)

        assert not is_valid
        # Should have multiple errors
        assert len(errors) >= 2, f"Should report multiple errors: {errors}"

    def test_deep_copy_does_not_affect_validation(self, pack_schema):
        """Test that deep copying manifest does not affect validation result."""
        original = VALID_PACK_MANIFEST_FULL
        copied = copy.deepcopy(original)

        is_valid_original, _ = validate_manifest(original, pack_schema)
        is_valid_copied, _ = validate_manifest(copied, pack_schema)

        assert is_valid_original == is_valid_copied

    def test_validation_is_deterministic(self, pack_schema, pipeline_schema):
        """Test that repeated validation produces identical results."""
        for _ in range(10):
            pack_valid, pack_errors = validate_manifest(
                VALID_PACK_MANIFEST_FULL, pack_schema
            )
            pipeline_valid, pipeline_errors = validate_manifest(
                VALID_PIPELINE_FULL, pipeline_schema
            )

            assert pack_valid is True
            assert pipeline_valid is True
            assert len(pack_errors) == 0
            assert len(pipeline_errors) == 0
