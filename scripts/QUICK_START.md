# AgentSpec V2.0 Validator - Quick Start Guide

## Installation

No installation needed! Just requires Python 3.7+ and PyYAML:

```bash
pip install pyyaml
```

## Basic Usage

### Validate a Single Agent Spec
```bash
python scripts/validate_agent_specs.py specs/domain1_industrial/industrial_process/agent_001_industrial_process_heat.yaml
```

**Output:**
```
✅ PASS - Agent spec is valid!
Errors: 0
Warnings: 13
```

### Validate All Specs in a Directory
```bash
python scripts/validate_agent_specs.py --batch specs/
```

**Output:**
```
🔍 Found 2 YAML files to validate

📄 Validating: agent_001_industrial_process_heat.yaml...
   ✅ PASS - Errors: 0, Warnings: 13

📄 Validating: agent_002_boiler_replacement.yaml...
   ✅ PASS - Errors: 0, Warnings: 6

================================================================================
BATCH VALIDATION SUMMARY
================================================================================
Total files validated: 2
✅ Passed: 2
❌ Failed: 0
Total errors: 0
Total warnings: 19
================================================================================

✨ All files passed validation!
```

### Save Report to File
```bash
python scripts/validate_agent_specs.py agent_001.yaml --output report.txt
```

## What Does It Check?

### Critical Checks (Must Pass)
1. ✅ **temperature = 0.0** (deterministic AI)
2. ✅ **seed = 42** (reproducibility)
3. ✅ **All tools have deterministic: true**
4. ✅ **zero_secrets: true** (no hardcoded credentials)
5. ✅ **provenance_tracking: true** (audit trail)
6. ✅ **test_coverage_target >= 0.80** (80% minimum)
7. ✅ **Required test categories present**
8. ✅ **Tool count matches tools_list length**
9. ✅ **All required sections present**
10. ✅ **Valid enum values** (domain, complexity, priority, status)

### Recommended Checks (Warnings)
- Tool examples provided
- Implementation details complete
- System prompt emphasizes key principles
- Documentation flags set to true
- Reasonable tool count (4-12)

## Common Errors and Fixes

### Error: temperature != 0.0
```yaml
# ❌ WRONG
ai_integration:
  temperature: 0.7

# ✅ CORRECT
ai_integration:
  temperature: 0.0
```

### Error: deterministic != true
```yaml
# ❌ WRONG
tools:
  tools_list:
    - tool_id: "tool_1"
      deterministic: false

# ✅ CORRECT
tools:
  tools_list:
    - tool_id: "tool_1"
      deterministic: true
```

### Error: tool_count mismatch
```yaml
# ❌ WRONG
tools:
  tool_count: 5
  tools_list:
    - tool_id: "tool_1"
    - tool_id: "tool_2"
    # Only 2 tools, but tool_count says 5!

# ✅ CORRECT
tools:
  tool_count: 2
  tools_list:
    - tool_id: "tool_1"
    - tool_id: "tool_2"
```

### Error: test_coverage_target < 0.80
```yaml
# ❌ WRONG
testing:
  test_coverage_target: 0.70

# ✅ CORRECT
testing:
  test_coverage_target: 0.80  # or higher
```

## Exit Codes

- **0** - All validations passed
- **1** - One or more validations failed

Perfect for CI/CD integration!

## CI/CD Integration

### GitHub Actions
```yaml
- name: Validate Agent Specs
  run: |
    pip install pyyaml
    python scripts/validate_agent_specs.py --batch specs/
```

### Pre-commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
python scripts/validate_agent_specs.py --batch specs/
if [ $? -ne 0 ]; then
    echo "Agent spec validation failed!"
    exit 1
fi
```

## Need More Help?

- **Full Documentation:** `scripts/README_VALIDATOR.md`
- **Template Reference:** `specs/AgentSpec_Template_v2.yaml`
- **Working Example:** `specs/domain1_industrial/industrial_process/agent_001_industrial_process_heat.yaml`
- **Summary Report:** `VALIDATION_SUMMARY.md`

## Quick Commands Reference

```bash
# Validate single file
python scripts/validate_agent_specs.py agent_001.yaml

# Validate directory
python scripts/validate_agent_specs.py --batch specs/

# Save report
python scripts/validate_agent_specs.py agent_001.yaml -o report.txt

# Validate with JSON schema
python scripts/validate_agent_specs.py --schema schemas/agentspec_v2_schema.json agent_001.yaml

# Validate specific pattern
python scripts/validate_agent_specs.py --batch specs/ --pattern "agent_*.yaml"
```

## Version

**Validator Version:** 1.0.0
**AgentSpec Version:** 2.0
**Last Updated:** 2025-10-13

---

**Ready to validate? Just run:**
```bash
python scripts/validate_agent_specs.py --batch specs/
```
