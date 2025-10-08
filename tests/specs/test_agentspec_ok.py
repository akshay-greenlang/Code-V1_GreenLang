"""
GreenLang AgentSpec v2 - Happy Path Tests

Tests valid AgentSpec v2 specifications to ensure all components work correctly.

Coverage:
- YAML loading
- JSON loading
- Dictionary validation
- Round-trip (load → export → reload → validate identical)
- Default values (deterministic=true, json_mode=true, etc.)
- Required and optional fields
- CTO boiler example validation

Author: GreenLang Framework Team
Date: October 2025
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from greenlang.specs.agentspec_v2 import (
    AgentSpecV2,
    AIBudget,
    AISpec,
    AITool,
    ComputeSpec,
    ConnectorRef,
    FactorRef,
    IOField,
    OutputField,
    ProvenanceSpec,
    RealtimeSpec,
    from_json,
    from_yaml,
    to_json_schema,
    validate_spec,
)
from greenlang.specs.errors import GLValidationError, GLVErr


# ============================================================================
# TEST DATA - CTO's Boiler Example
# ============================================================================

CTO_BOILER_YAML = """
schema_version: "2.0.0"
id: "buildings/boiler_ng_v1"
name: "Boiler – Natural Gas (LHV)"
version: "2.1.3"
summary: "Computes CO2e from NG boiler fuel using LHV."
tags: ["buildings", "combustion", "scope1"]
license: "Apache-2.0"

compute:
  entrypoint: "python://gl.agents.boiler.ng:compute"
  deterministic: true
  inputs:
    fuel_volume:
      dtype: "float64"
      unit: "m^3"
      required: true
      ge: 0
      description: "Natural gas volume consumed"
    efficiency:
      dtype: "float64"
      unit: "1"
      required: true
      gt: 0
      le: 1
      description: "Boiler efficiency (fraction)"
  outputs:
    co2e_kg:
      dtype: "float64"
      unit: "kgCO2e"
      description: "Total CO2 equivalent emissions"
  factors:
    co2e_factor:
      ref: "ef://ipcc_ar6/combustion/ng/co2e_kg_per_mj"
      gwp_set: "AR6GWP100"
      description: "Natural gas combustion emission factor"

ai:
  json_mode: true
  system_prompt: "You are a climate advisor. Use tools; never guess numbers."
  budget:
    max_cost_usd: 1.00
    max_input_tokens: 15000
    max_output_tokens: 2000
  rag_collections: ["ghg_protocol_corp", "ipcc_ar6"]
  tools: []

realtime:
  default_mode: "replay"
  connectors: []

provenance:
  pin_ef: true
  gwp_set: "AR6GWP100"
  record: ["inputs", "outputs", "factors", "ef_uri", "code_sha"]
"""


CTO_BOILER_DICT = {
    "schema_version": "2.0.0",
    "id": "buildings/boiler_ng_v1",
    "name": "Boiler – Natural Gas (LHV)",
    "version": "2.1.3",
    "summary": "Computes CO2e from NG boiler fuel using LHV.",
    "tags": ["buildings", "combustion", "scope1"],
    "license": "Apache-2.0",
    "compute": {
        "entrypoint": "python://gl.agents.boiler.ng:compute",
        "deterministic": True,
        "inputs": {
            "fuel_volume": {
                "dtype": "float64",
                "unit": "m^3",
                "required": True,
                "ge": 0,
                "description": "Natural gas volume consumed"
            },
            "efficiency": {
                "dtype": "float64",
                "unit": "1",
                "required": True,
                "gt": 0,
                "le": 1,
                "description": "Boiler efficiency (fraction)"
            }
        },
        "outputs": {
            "co2e_kg": {
                "dtype": "float64",
                "unit": "kgCO2e",
                "description": "Total CO2 equivalent emissions"
            }
        },
        "factors": {
            "co2e_factor": {
                "ref": "ef://ipcc_ar6/combustion/ng/co2e_kg_per_mj",
                "gwp_set": "AR6GWP100",
                "description": "Natural gas combustion emission factor"
            }
        }
    },
    "ai": {
        "json_mode": True,
        "system_prompt": "You are a climate advisor. Use tools; never guess numbers.",
        "budget": {
            "max_cost_usd": 1.00,
            "max_input_tokens": 15000,
            "max_output_tokens": 2000
        },
        "rag_collections": ["ghg_protocol_corp", "ipcc_ar6"],
        "tools": []
    },
    "realtime": {
        "default_mode": "replay",
        "connectors": []
    },
    "provenance": {
        "pin_ef": True,
        "gwp_set": "AR6GWP100",
        "record": ["inputs", "outputs", "factors", "ef_uri", "code_sha"]
    }
}


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def boiler_spec_dict():
    """CTO's boiler example as dictionary."""
    return CTO_BOILER_DICT.copy()


@pytest.fixture
def boiler_spec_yaml():
    """CTO's boiler example as YAML string."""
    return CTO_BOILER_YAML


@pytest.fixture
def temp_yaml_file(boiler_spec_yaml):
    """Temporary YAML file with boiler spec."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(boiler_spec_yaml)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def temp_json_file(boiler_spec_dict):
    """Temporary JSON file with boiler spec."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(boiler_spec_dict, f, indent=2)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


# ============================================================================
# HAPPY PATH TESTS - Loading
# ============================================================================

def test_load_valid_spec_from_yaml_succeeds(temp_yaml_file):
    """Test loading valid AgentSpec v2 from YAML file."""
    spec = from_yaml(temp_yaml_file)

    assert isinstance(spec, AgentSpecV2)
    assert spec.schema_version == "2.0.0"
    assert spec.id == "buildings/boiler_ng_v1"
    assert spec.name == "Boiler – Natural Gas (LHV)"
    assert spec.version == "2.1.3"
    assert spec.summary == "Computes CO2e from NG boiler fuel using LHV."
    assert "buildings" in spec.tags
    assert "combustion" in spec.tags
    assert "scope1" in spec.tags
    assert spec.license == "Apache-2.0"


def test_load_valid_spec_from_json_succeeds(temp_json_file):
    """Test loading valid AgentSpec v2 from JSON file."""
    spec = from_json(temp_json_file)

    assert isinstance(spec, AgentSpecV2)
    assert spec.schema_version == "2.0.0"
    assert spec.id == "buildings/boiler_ng_v1"
    assert spec.name == "Boiler – Natural Gas (LHV)"
    assert spec.version == "2.1.3"


def test_load_valid_spec_from_dict_succeeds(boiler_spec_dict):
    """Test loading valid AgentSpec v2 from dictionary."""
    spec = validate_spec(boiler_spec_dict)

    assert isinstance(spec, AgentSpecV2)
    assert spec.schema_version == "2.0.0"
    assert spec.id == "buildings/boiler_ng_v1"
    assert spec.version == "2.1.3"


def test_pydantic_model_validate_succeeds(boiler_spec_dict):
    """Test direct Pydantic validation."""
    spec = AgentSpecV2.model_validate(boiler_spec_dict)

    assert isinstance(spec, AgentSpecV2)
    assert spec.id == "buildings/boiler_ng_v1"


# ============================================================================
# HAPPY PATH TESTS - Round-Trip
# ============================================================================

def test_roundtrip_yaml_load_export_reload_identical(temp_yaml_file):
    """Test round-trip: YAML → load → export → reload → verify identical."""
    # Load from YAML
    spec1 = from_yaml(temp_yaml_file)

    # Export to dict
    spec1_dict = spec1.model_dump()

    # Reload from dict
    spec2 = validate_spec(spec1_dict)

    # Verify identical
    assert spec1.model_dump() == spec2.model_dump()
    assert spec1.id == spec2.id
    assert spec1.version == spec2.version
    assert spec1.compute.entrypoint == spec2.compute.entrypoint


def test_roundtrip_json_load_export_reload_identical(temp_json_file):
    """Test round-trip: JSON → load → export → reload → verify identical."""
    # Load from JSON
    spec1 = from_json(temp_json_file)

    # Export to dict
    spec1_dict = spec1.model_dump()

    # Reload from dict
    spec2 = validate_spec(spec1_dict)

    # Verify identical
    assert spec1.model_dump() == spec2.model_dump()


def test_roundtrip_dict_to_json_to_dict_identical(boiler_spec_dict):
    """Test round-trip: dict → model → JSON string → reload → verify identical."""
    # Load from dict
    spec1 = validate_spec(boiler_spec_dict)

    # Export to JSON string
    json_str = spec1.model_dump_json()

    # Reload from JSON string
    spec2 = AgentSpecV2.model_validate_json(json_str)

    # Verify identical
    assert spec1.model_dump() == spec2.model_dump()


# ============================================================================
# HAPPY PATH TESTS - Defaults
# ============================================================================

def test_defaults_compute_deterministic_is_true():
    """Test compute.deterministic defaults to true."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/agent_v1",
        "name": "Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            # deterministic NOT specified - should default to true
            "inputs": {
                "x": {"dtype": "float64", "unit": "1"}
            },
            "outputs": {
                "y": {"dtype": "float64", "unit": "1"}
            }
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs", "outputs"]
        }
    }

    spec = validate_spec(spec_dict)
    assert spec.compute.deterministic is True


def test_defaults_ai_json_mode_is_true():
    """Test ai.json_mode defaults to true."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/agent_v1",
        "name": "Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {
            # json_mode NOT specified - should default to true
        },
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)
    assert spec.ai.json_mode is True


def test_defaults_realtime_mode_is_replay():
    """Test realtime.default_mode defaults to 'replay'."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/agent_v1",
        "name": "Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {},
        "realtime": {
            # default_mode NOT specified - should default to "replay"
        },
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)
    assert spec.realtime.default_mode == "replay"


def test_defaults_provenance_pin_ef_is_true():
    """Test provenance.pin_ef defaults to true."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/agent_v1",
        "name": "Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}},
            "factors": {
                "ef": {"ref": "ef://test/factor"}
            }
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            # pin_ef NOT specified - should default to true
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)
    assert spec.provenance.pin_ef is True


def test_defaults_iofield_required_is_true():
    """Test IOField.required defaults to true."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/agent_v1",
        "name": "Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {
                "x": {
                    "dtype": "float64",
                    "unit": "1"
                    # required NOT specified - should default to true
                }
            },
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)
    assert spec.compute.inputs["x"].required is True


# ============================================================================
# HAPPY PATH TESTS - Required Fields
# ============================================================================

def test_all_required_fields_present(boiler_spec_dict):
    """Test that all required fields are present and validated."""
    spec = validate_spec(boiler_spec_dict)

    # Top-level required fields
    assert spec.schema_version == "2.0.0"
    assert spec.id == "buildings/boiler_ng_v1"
    assert spec.name == "Boiler – Natural Gas (LHV)"
    assert spec.version == "2.1.3"

    # Compute section required fields
    assert spec.compute.entrypoint == "python://gl.agents.boiler.ng:compute"
    assert spec.compute.inputs is not None
    assert spec.compute.outputs is not None

    # AI section (required but can be empty)
    assert spec.ai is not None

    # Realtime section (required but can be empty)
    assert spec.realtime is not None

    # Provenance section required fields
    assert spec.provenance is not None
    assert spec.provenance.record is not None


# ============================================================================
# HAPPY PATH TESTS - Optional Fields
# ============================================================================

def test_optional_fields_work_correctly():
    """Test that optional fields can be omitted or included."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/minimal_v1",
        "name": "Minimal Agent",
        "version": "1.0.0",
        # summary OPTIONAL - omitted
        # tags OPTIONAL - omitted
        # owners OPTIONAL - omitted
        # license OPTIONAL - omitted
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}},
            # factors OPTIONAL - omitted
        },
        "ai": {
            # system_prompt OPTIONAL - omitted
            # budget OPTIONAL - omitted
            # rag_collections OPTIONAL - empty list is fine
            # tools OPTIONAL - empty list is fine
        },
        "realtime": {
            # connectors OPTIONAL - empty list is fine
        },
        "provenance": {
            "pin_ef": False,  # No factors, so must be false
            "record": ["inputs"]
            # gwp_set OPTIONAL - will default to AR6GWP100
        }
    }

    spec = validate_spec(spec_dict)

    # Verify optional fields are None or default values
    assert spec.summary is None
    assert spec.tags == []
    assert spec.owners is None
    assert spec.license is None
    assert spec.compute.factors == {}
    assert spec.ai.system_prompt is None
    assert spec.ai.budget is None
    assert spec.ai.rag_collections == []
    assert spec.ai.tools == []
    assert spec.realtime.connectors == []


def test_optional_fields_included():
    """Test that optional fields work when included."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/full_v1",
        "name": "Full Agent",
        "version": "1.0.0",
        "summary": "Test summary",
        "tags": ["test", "example"],
        "owners": ["@greenlang/core"],
        "license": "MIT",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}},
            "factors": {
                "ef": {"ref": "ef://test/factor"}
            }
        },
        "ai": {
            "system_prompt": "Test prompt",
            "budget": {
                "max_cost_usd": 1.0,
                "max_input_tokens": 1000
            },
            "rag_collections": ["test_coll"],
            "tools": []
        },
        "realtime": {
            "connectors": []
        },
        "provenance": {
            "pin_ef": True,
            "gwp_set": "AR5GWP100",
            "record": ["inputs", "outputs"]
        }
    }

    spec = validate_spec(spec_dict)

    assert spec.summary == "Test summary"
    assert spec.tags == ["test", "example"]
    assert spec.owners == ["@greenlang/core"]
    assert spec.license == "MIT"
    assert "ef" in spec.compute.factors
    assert spec.ai.system_prompt == "Test prompt"
    assert spec.ai.budget.max_cost_usd == 1.0
    assert spec.ai.rag_collections == ["test_coll"]
    assert spec.provenance.gwp_set == "AR5GWP100"


# ============================================================================
# HAPPY PATH TESTS - Compute Section
# ============================================================================

def test_compute_inputs_validation(boiler_spec_dict):
    """Test compute.inputs are validated correctly."""
    spec = validate_spec(boiler_spec_dict)

    # Verify inputs
    assert "fuel_volume" in spec.compute.inputs
    assert "efficiency" in spec.compute.inputs

    fuel_vol = spec.compute.inputs["fuel_volume"]
    assert fuel_vol.dtype == "float64"
    assert fuel_vol.unit == "m^3"
    assert fuel_vol.required is True
    assert fuel_vol.ge == 0
    assert fuel_vol.description == "Natural gas volume consumed"

    efficiency = spec.compute.inputs["efficiency"]
    assert efficiency.dtype == "float64"
    assert efficiency.unit == "1"
    assert efficiency.required is True
    assert efficiency.gt == 0
    assert efficiency.le == 1


def test_compute_outputs_validation(boiler_spec_dict):
    """Test compute.outputs are validated correctly."""
    spec = validate_spec(boiler_spec_dict)

    # Verify outputs
    assert "co2e_kg" in spec.compute.outputs

    co2e = spec.compute.outputs["co2e_kg"]
    assert co2e.dtype == "float64"
    assert co2e.unit == "kgCO2e"
    assert co2e.description == "Total CO2 equivalent emissions"


def test_compute_factors_validation(boiler_spec_dict):
    """Test compute.factors are validated correctly."""
    spec = validate_spec(boiler_spec_dict)

    # Verify factors
    assert "co2e_factor" in spec.compute.factors

    factor = spec.compute.factors["co2e_factor"]
    assert factor.ref == "ef://ipcc_ar6/combustion/ng/co2e_kg_per_mj"
    assert factor.gwp_set == "AR6GWP100"
    assert factor.description == "Natural gas combustion emission factor"


# ============================================================================
# HAPPY PATH TESTS - AI Section
# ============================================================================

def test_ai_budget_validation(boiler_spec_dict):
    """Test ai.budget is validated correctly."""
    spec = validate_spec(boiler_spec_dict)

    assert spec.ai.budget is not None
    assert spec.ai.budget.max_cost_usd == 1.0
    assert spec.ai.budget.max_input_tokens == 15000
    assert spec.ai.budget.max_output_tokens == 2000


def test_ai_tools_with_valid_json_schema():
    """Test AI tools with valid JSON Schema validation."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/tool_agent_v1",
        "name": "Tool Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {
            "tools": [
                {
                    "name": "select_factor",
                    "description": "Select emission factor",
                    "schema_in": {
                        "type": "object",
                        "properties": {
                            "region": {"type": "string"},
                            "year": {"type": "integer"}
                        },
                        "required": ["region", "year"]
                    },
                    "schema_out": {
                        "type": "object",
                        "properties": {
                            "factor_uri": {"type": "string", "pattern": "^ef://"}
                        },
                        "required": ["factor_uri"]
                    },
                    "impl": "python://gl.tools:select_factor",
                    "safe": True
                }
            ]
        },
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)

    assert len(spec.ai.tools) == 1
    tool = spec.ai.tools[0]
    assert tool.name == "select_factor"
    assert tool.safe is True
    assert "region" in tool.schema_in["properties"]
    assert "factor_uri" in tool.schema_out["properties"]


# ============================================================================
# HAPPY PATH TESTS - Realtime Section
# ============================================================================

def test_realtime_connectors_validation():
    """Test realtime.connectors are validated correctly."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/connector_agent_v1",
        "name": "Connector Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {},
        "realtime": {
            "default_mode": "live",
            "connectors": [
                {
                    "name": "grid_intensity",
                    "topic": "region_hourly_ci",
                    "window": "1h",
                    "ttl": "6h",
                    "required": False
                }
            ]
        },
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)

    assert spec.realtime.default_mode == "live"
    assert len(spec.realtime.connectors) == 1

    conn = spec.realtime.connectors[0]
    assert conn.name == "grid_intensity"
    assert conn.topic == "region_hourly_ci"
    assert conn.window == "1h"
    assert conn.ttl == "6h"
    assert conn.required is False


# ============================================================================
# HAPPY PATH TESTS - Provenance Section
# ============================================================================

def test_provenance_with_factors(boiler_spec_dict):
    """Test provenance.pin_ef=true with factors works correctly."""
    spec = validate_spec(boiler_spec_dict)

    assert spec.provenance.pin_ef is True
    assert spec.provenance.gwp_set == "AR6GWP100"
    assert "inputs" in spec.provenance.record
    assert "outputs" in spec.provenance.record
    assert "factors" in spec.provenance.record
    assert "ef_uri" in spec.provenance.record
    assert "code_sha" in spec.provenance.record


def test_provenance_without_factors():
    """Test provenance.pin_ef=false without factors works correctly."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/no_factors_v1",
        "name": "No Factors Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}},
            # No factors
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            "pin_ef": False,  # No factors, so pin_ef must be false
            "record": ["inputs", "outputs", "code_sha"]
        }
    }

    spec = validate_spec(spec_dict)

    assert spec.provenance.pin_ef is False
    assert spec.compute.factors == {}


# ============================================================================
# HAPPY PATH TESTS - JSON Schema Export
# ============================================================================

def test_json_schema_export_succeeds():
    """Test exporting AgentSpec v2 as JSON Schema."""
    schema = to_json_schema()

    assert isinstance(schema, dict)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "https://greenlang.io/specs/agentspec_v2.json"
    assert schema["title"] == "GreenLang AgentSpec v2"
    assert "properties" in schema
    assert "schema_version" in schema["properties"]
    assert "compute" in schema["properties"]
    assert "ai" in schema["properties"]
    assert "realtime" in schema["properties"]
    assert "provenance" in schema["properties"]


def test_json_schema_is_valid_json_schema():
    """Test that exported JSON Schema is valid JSON Schema draft-2020-12."""
    import jsonschema

    schema = to_json_schema()

    # This should not raise an exception
    jsonschema.Draft202012Validator.check_schema(schema)


# ============================================================================
# HAPPY PATH TESTS - Edge Cases
# ============================================================================

def test_empty_tags_list():
    """Test that empty tags list is valid."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/no_tags_v1",
        "name": "No Tags Agent",
        "version": "1.0.0",
        "tags": [],  # Empty list
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)
    assert spec.tags == []


def test_empty_rag_collections():
    """Test that empty rag_collections list is valid."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/no_rag_v1",
        "name": "No RAG Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {
            "rag_collections": []  # Empty list
        },
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)
    assert spec.ai.rag_collections == []


def test_multiple_inputs_outputs():
    """Test spec with multiple inputs and outputs."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/multi_io_v1",
        "name": "Multi I/O Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {
                "a": {"dtype": "float64", "unit": "kg"},
                "b": {"dtype": "float64", "unit": "MJ"},
                "c": {"dtype": "int64", "unit": "1"}
            },
            "outputs": {
                "x": {"dtype": "float64", "unit": "kgCO2e"},
                "y": {"dtype": "float64", "unit": "kWh"},
                "z": {"dtype": "string", "unit": "1"}
            }
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs", "outputs"]
        }
    }

    spec = validate_spec(spec_dict)

    assert len(spec.compute.inputs) == 3
    assert len(spec.compute.outputs) == 3
    assert "a" in spec.compute.inputs
    assert "z" in spec.compute.outputs


def test_constraint_combinations():
    """Test various constraint combinations on inputs."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/constraints_v1",
        "name": "Constraints Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {
                "positive": {"dtype": "float64", "unit": "1", "ge": 0},
                "strictly_positive": {"dtype": "float64", "unit": "1", "gt": 0},
                "bounded": {"dtype": "float64", "unit": "1", "ge": 0, "le": 100},
                "fraction": {"dtype": "float64", "unit": "1", "gt": 0, "lt": 1},
                "enum_values": {"dtype": "string", "unit": "1", "enum": ["low", "medium", "high"]}
            },
            "outputs": {
                "result": {"dtype": "float64", "unit": "1"}
            }
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)

    assert spec.compute.inputs["positive"].ge == 0
    assert spec.compute.inputs["strictly_positive"].gt == 0
    assert spec.compute.inputs["bounded"].ge == 0
    assert spec.compute.inputs["bounded"].le == 100
    assert spec.compute.inputs["fraction"].gt == 0
    assert spec.compute.inputs["fraction"].lt == 1
    assert spec.compute.inputs["enum_values"].enum == ["low", "medium", "high"]


def test_optional_input_with_default():
    """Test optional input with default value."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/optional_input_v1",
        "name": "Optional Input Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {
                "required_input": {"dtype": "float64", "unit": "1", "required": True},
                "optional_input": {"dtype": "float64", "unit": "1", "required": False, "default": 42.0}
            },
            "outputs": {
                "result": {"dtype": "float64", "unit": "1"}
            }
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)

    assert spec.compute.inputs["required_input"].required is True
    assert spec.compute.inputs["required_input"].default is None
    assert spec.compute.inputs["optional_input"].required is False
    assert spec.compute.inputs["optional_input"].default == 42.0


# ============================================================================
# HAPPY PATH TESTS - P1 Enhancements
# ============================================================================

def test_compute_dependencies_validation():
    """Test compute.dependencies field works correctly."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/dependencies_v1",
        "name": "Dependencies Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "dependencies": [
                "pandas==2.1.4",
                "numpy==1.26.0",
                "pydantic==2.8.2"
            ],
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)

    assert spec.compute.dependencies is not None
    assert len(spec.compute.dependencies) == 3
    assert "pandas==2.1.4" in spec.compute.dependencies
    assert "numpy==1.26.0" in spec.compute.dependencies
    assert "pydantic==2.8.2" in spec.compute.dependencies


def test_compute_python_version_validation():
    """Test compute.python_version field works correctly."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/python_version_v1",
        "name": "Python Version Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "python_version": "3.11",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)

    assert spec.compute.python_version == "3.11"


def test_compute_timeout_and_memory_constraints():
    """Test compute.timeout_s and compute.memory_limit_mb fields."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/performance_v1",
        "name": "Performance Constraints Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "timeout_s": 30,
            "memory_limit_mb": 512,
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)

    assert spec.compute.timeout_s == 30
    assert spec.compute.memory_limit_mb == 512


def test_ai_budget_max_retries():
    """Test ai.budget.max_retries field with default value."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/retries_v1",
        "name": "Max Retries Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {
            "budget": {
                "max_retries": 5
            }
        },
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)

    assert spec.ai.budget is not None
    assert spec.ai.budget.max_retries == 5


def test_realtime_snapshot_path():
    """Test realtime.snapshot_path field for deterministic testing."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/snapshot_v1",
        "name": "Snapshot Path Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {},
        "realtime": {
            "snapshot_path": "snapshots/2024-10-06_test_data.json"
        },
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    spec = validate_spec(spec_dict)

    assert spec.realtime.snapshot_path == "snapshots/2024-10-06_test_data.json"


def test_security_allowlist_hosts():
    """Test security.allowlist_hosts field for network egress control."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/security_v1",
        "name": "Security Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {},
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        },
        "security": {
            "allowlist_hosts": [
                "api.ipcc.ch",
                "api.epa.gov",
                "api.eia.gov"
            ]
        }
    }

    spec = validate_spec(spec_dict)

    assert spec.security is not None
    assert "allowlist_hosts" in spec.security
    assert len(spec.security["allowlist_hosts"]) == 3
    assert "api.ipcc.ch" in spec.security["allowlist_hosts"]


# ============================================================================
# HAPPY PATH TESTS - AST Safety Validation
# ============================================================================

def test_safe_tool_validation_accepts_safe_functions():
    """Test that safe tools (pure functions) pass validation."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/safe_tool_v1",
        "name": "Safe Tool Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {
            "tools": [
                {
                    "name": "safe_calculator",
                    "description": "Safe calculation tool",
                    "schema_in": {"type": "object", "properties": {"value": {"type": "number"}}},
                    "schema_out": {"type": "object", "properties": {"result": {"type": "number"}}},
                    "impl": "python://math:sqrt",  # Built-in safe function
                    "safe": True
                }
            ]
        },
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    # Should not raise - built-in math functions are safe
    spec = validate_spec(spec_dict)
    assert len(spec.ai.tools) == 1
    assert spec.ai.tools[0].safe is True


def test_safe_tool_with_nonexistent_module_skips_validation():
    """Test that safe tools with nonexistent modules skip AST validation."""
    spec_dict = {
        "schema_version": "2.0.0",
        "id": "test/nonexistent_tool_v1",
        "name": "Nonexistent Tool Test Agent",
        "version": "1.0.0",
        "compute": {
            "entrypoint": "python://test.module:compute",
            "inputs": {"x": {"dtype": "float64", "unit": "1"}},
            "outputs": {"y": {"dtype": "float64", "unit": "1"}}
        },
        "ai": {
            "tools": [
                {
                    "name": "future_tool",
                    "description": "Tool that doesn't exist yet",
                    "schema_in": {"type": "object"},
                    "schema_out": {"type": "object"},
                    "impl": "python://nonexistent.module:function",
                    "safe": True
                }
            ]
        },
        "realtime": {},
        "provenance": {
            "pin_ef": False,
            "record": ["inputs"]
        }
    }

    # Should not raise - nonexistent modules skip validation
    spec = validate_spec(spec_dict)
    assert len(spec.ai.tools) == 1
