# AgentSpec V2.0 Validator

Comprehensive validation script for GreenLang Agent Specifications V2.0. Ensures all 84 agent specs follow the tool-first design pattern with deterministic AI.

## Quick Start

### Validate Single Agent Spec
```bash
python scripts/validate_agent_specs.py specs/domain1_industrial/industrial_process/agent_001_industrial_process_heat.yaml
```

### Validate All Specs (Batch Mode)
```bash
python scripts/validate_agent_specs.py --batch specs/
```

### Save Validation Report
```bash
python scripts/validate_agent_specs.py agent_001.yaml --output validation_report.txt
```

## Features

### Critical Validations (Must Pass)
- ✅ **Tool-First Design**: All tools must have `deterministic: true`
- ✅ **Deterministic AI**: `temperature: 0.0` and `seed: 42` enforced
- ✅ **Zero Secrets**: `zero_secrets: true` required
- ✅ **Test Coverage**: Minimum 80% coverage target
- ✅ **Provenance Tracking**: `provenance_tracking: true` required
- ✅ **Required Test Categories**: unit_tests, integration_tests, determinism_tests, boundary_tests

### Structure Validations
- Required sections: agent_metadata, description, tools, ai_integration, inputs, outputs, testing, deployment, documentation, compliance, metadata
- Tool count must match tools_list length
- All tools must have complete parameter and return schemas
- Input/output schemas must follow JSON Schema format

### Quality Checks (Warnings)
- Recommended tool count: 4-12 tools per agent
- System prompt should emphasize: "use tools", "never estimate", "never guess"
- Test categories should include performance_tests
- Documentation flags (readme, api_docs, examples, tutorials) should be true

## Validation Rules

### 1. Agent Metadata
```yaml
agent_metadata:
  agent_id: "domain/agent_name"  # Format: lowercase with underscores
  version: "1.0.0"                # Semantic versioning (x.y.z)
  domain: "Domain1_Industrial"    # One of: Domain1_Industrial, Domain2_HVAC, Domain3_CrossCutting
  complexity: "High"              # One of: Low, Medium, High
  priority: "P0_Critical"         # One of: P0_Critical, P1_High, P2_Medium
  status: "Spec_Complete"         # One of: Spec_Needed, Spec_Complete, In_Development, Testing, Production
```

**Validation:**
- ❌ ERROR: Missing required fields
- ❌ ERROR: Invalid enum values
- ⚠️  WARNING: agent_id doesn't follow format convention

### 2. Tools (CRITICAL)
```yaml
tools:
  tool_count: 7  # Must match tools_list length
  tools_list:
    - tool_id: "tool_1"
      name: "calculate_process_heat_demand"
      description: "Clear description (min 20 chars)"
      category: "calculation"  # One of: calculation, lookup, aggregation, analysis, optimization
      deterministic: true      # MUST BE TRUE!

      parameters:
        type: "object"
        properties:
          param1:
            type: "number"
            description: "Clear parameter description"
            required: true
        required: ["param1"]

      returns:
        type: "object"
        properties:
          result:
            type: "number"
            description: "Result description"

      implementation:
        calculation_method: "How it's calculated"
        data_source: "Where data comes from"
        accuracy: "Expected accuracy"
        validation: "How to validate"
```

**Validation:**
- ❌ ERROR: `deterministic` must be `true` for ALL tools
- ❌ ERROR: tool_count doesn't match actual tools_list length
- ❌ ERROR: Missing required fields (tool_id, name, description, category, parameters, returns, implementation)
- ❌ ERROR: Invalid category value
- ⚠️  WARNING: Tool count outside recommended range (4-12)
- ⚠️  WARNING: Missing example

### 3. AI Integration (CRITICAL)
```yaml
ai_integration:
  temperature: 0.0     # MUST BE 0.0!
  seed: 42             # MUST BE 42!
  tool_choice: "auto"
  max_iterations: 5
  budget_usd: 0.10
  provenance_tracking: true  # MUST BE TRUE!
  ai_summary: true

  system_prompt: |
    CRITICAL RULES:
    - Use provided tools for ALL calculations
    - NEVER estimate or guess numbers
    - Always explain analysis clearly
```

**Validation:**
- ❌ ERROR: `temperature` must be exactly `0.0`
- ❌ ERROR: `seed` must be exactly `42`
- ❌ ERROR: `provenance_tracking` must be `true`
- ⚠️  WARNING: system_prompt should emphasize tool usage and determinism
- ⚠️  WARNING: budget_usd > $1.00 seems high

### 4. Testing
```yaml
testing:
  test_coverage_target: 0.95  # Must be >= 0.80 (80%)

  test_categories:
    - category: "unit_tests"           # Required
      description: "Test individual tools"
      count: 25

    - category: "integration_tests"    # Required
      description: "Test AI orchestration"
      count: 8

    - category: "determinism_tests"    # Required
      description: "Verify temperature=0, seed=42"
      count: 3

    - category: "boundary_tests"       # Required
      description: "Test edge cases"
      count: 5

    - category: "performance_tests"    # Recommended
      description: "Verify latency/cost"
      count: 3
```

**Validation:**
- ❌ ERROR: test_coverage_target < 0.80 (80% minimum)
- ❌ ERROR: Missing required test category (unit_tests, integration_tests, determinism_tests, boundary_tests)

### 5. Compliance (CRITICAL)
```yaml
compliance:
  zero_secrets: true      # MUST BE TRUE!
  sbom_required: true
  digital_signature: true

  standards:
    - "GHG Protocol Corporate Standard"
    - "ISO 14064-1:2018"
```

**Validation:**
- ❌ ERROR: `zero_secrets` must be `true`

## Output Format

### Single File Validation
```
================================================================================
Validation Report: agent_001_industrial_process_heat.yaml
================================================================================
Status: ✅ PASS
Errors: 0
Warnings: 2
Info: 0
--------------------------------------------------------------------------------

🟡 WARNINGS:

⚠️  WARNING: [tools] tool_count
   Tool count is high (typical: 4-12 tools) - consider splitting agent
   Actual: 15

⚠️  WARNING: [documentation] tutorials
   tutorials is set to false - documentation may be incomplete

✨ All validation checks passed!
================================================================================
```

### Batch Validation
```
🔍 Found 5 YAML files to validate

📄 Validating: agent_001_industrial_process_heat.yaml...
   ✅ PASS - Errors: 0, Warnings: 2

📄 Validating: agent_002_boiler_replacement.yaml...
   ❌ FAIL - Errors: 3, Warnings: 1

...

================================================================================
BATCH VALIDATION SUMMARY
================================================================================
Total files validated: 5
✅ Passed: 4
❌ Failed: 1
Total errors: 3
Total warnings: 8
================================================================================

❌ Failed files:
   • agent_002_boiler_replacement.yaml (3 errors)
```

## Error Severity Levels

### ❌ ERROR (Blocking)
Must be fixed before agent can be implemented. Includes:
- Missing required fields
- Invalid enum values
- `temperature` ≠ 0.0
- `seed` ≠ 42
- `deterministic` ≠ true
- `zero_secrets` ≠ true
- `provenance_tracking` ≠ true
- test_coverage_target < 0.80
- Missing required test categories

### ⚠️  WARNING (Should Fix)
Should be addressed but not blocking. Includes:
- Tool count outside recommended range (4-12)
- Missing recommended fields
- System prompt doesn't emphasize key principles
- Missing examples
- Documentation flags set to false

### ℹ️  INFO (Informational)
Helpful information, no action required. Includes:
- Empty optional fields
- Additional context

## Usage Examples

### Example 1: Validate During Development
```bash
# Create your agent spec
vim specs/domain1_industrial/industrial_process/agent_001_industrial_process_heat.yaml

# Validate immediately
python scripts/validate_agent_specs.py specs/domain1_industrial/industrial_process/agent_001_industrial_process_heat.yaml

# Fix errors, re-validate
python scripts/validate_agent_specs.py specs/domain1_industrial/industrial_process/agent_001_industrial_process_heat.yaml
```

### Example 2: CI/CD Integration
```bash
# In your CI pipeline
python scripts/validate_agent_specs.py --batch specs/ --output validation_report.txt

# Check exit code
if [ $? -ne 0 ]; then
    echo "Validation failed!"
    cat validation_report.txt
    exit 1
fi
```

### Example 3: Pre-commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
python scripts/validate_agent_specs.py --batch specs/
```

### Example 4: Validate Specific Domain
```bash
# Validate only industrial domain
python scripts/validate_agent_specs.py --batch specs/domain1_industrial/ --pattern "**/*.yaml"
```

## Common Issues and Fixes

### Issue 1: temperature ≠ 0.0
```yaml
# ❌ WRONG
ai_integration:
  temperature: 0.7

# ✅ CORRECT
ai_integration:
  temperature: 0.0
```

### Issue 2: deterministic not set
```yaml
# ❌ WRONG
tools_list:
  - tool_id: "tool_1"
    name: "calculate_something"
    deterministic: false  # or missing

# ✅ CORRECT
tools_list:
  - tool_id: "tool_1"
    name: "calculate_something"
    deterministic: true
```

### Issue 3: tool_count mismatch
```yaml
# ❌ WRONG
tools:
  tool_count: 5
  tools_list:
    - tool_id: "tool_1"
      ...
    - tool_id: "tool_2"
      ...
    # Only 2 tools, but tool_count says 5!

# ✅ CORRECT
tools:
  tool_count: 2
  tools_list:
    - tool_id: "tool_1"
      ...
    - tool_id: "tool_2"
      ...
```

### Issue 4: Missing required test category
```yaml
# ❌ WRONG
testing:
  test_categories:
    - category: "unit_tests"
      count: 10
    # Missing integration_tests, determinism_tests, boundary_tests!

# ✅ CORRECT
testing:
  test_categories:
    - category: "unit_tests"
      count: 10
    - category: "integration_tests"
      count: 5
    - category: "determinism_tests"
      count: 3
    - category: "boundary_tests"
      count: 5
```

## JSON Schema Validation

For advanced validation with JSON Schema:

```bash
# Install jsonschema (if not already installed)
pip install jsonschema pyyaml

# Validate with JSON schema
python scripts/validate_agent_specs.py \
  --schema schemas/agentspec_v2_schema.json \
  specs/domain1_industrial/industrial_process/agent_001_industrial_process_heat.yaml
```

## Integration with Agent Factory

The validator ensures specs are ready for Agent Factory code generation:

```bash
# 1. Validate spec
python scripts/validate_agent_specs.py agent_001.yaml

# 2. If validation passes, generate agent code
python scripts/agent_factory.py --spec agent_001.yaml --output agents/

# 3. Validate generated code passes tests
pytest tests/test_agent_001.py
```

## Development Workflow

```
┌─────────────────────┐
│  Create Agent Spec  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Run Validator      │◄──┐
└──────────┬──────────┘   │
           │               │
           ▼               │
      Errors? ─────Yes─────┘
           │
           No
           │
           ▼
┌─────────────────────┐
│  Agent Factory      │
│  Code Generation    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Run Tests          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Deploy Agent       │
└─────────────────────┘
```

## Requirements

```bash
pip install pyyaml
```

Optional:
```bash
pip install jsonschema  # For JSON schema validation
```

## Files

- `scripts/validate_agent_specs.py` - Main validator script
- `schemas/agentspec_v2_schema.json` - JSON Schema definition
- `scripts/README_VALIDATOR.md` - This documentation
- `specs/AgentSpec_Template_v2.yaml` - Template for all agent specs

## Support

For questions or issues:
1. Check this README
2. Review the template: `specs/AgentSpec_Template_v2.yaml`
3. Review Agent #1 reference: `specs/domain1_industrial/industrial_process/agent_001_industrial_process_heat.yaml`
4. Contact: AI Team

## Version History

- **v1.0.0** (2025-10-13): Initial release
  - Complete validation for AgentSpec V2.0
  - Tool-first design enforcement
  - Deterministic AI validation (temperature=0, seed=42)
  - Test coverage validation (80% minimum)
  - Batch validation mode
  - JSON Schema support

---

**Remember**: The validator is your friend! It ensures consistency across all 84 agent specs and enforces the critical tool-first design pattern. All agents must pass validation before code generation.
