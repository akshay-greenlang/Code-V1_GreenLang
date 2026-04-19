# GreenLang Exit Bar Criteria System
## Comprehensive Production Readiness Validation

**Version:** 1.0.0
**Date:** 2025-10-16
**Status:** Production Ready

---

## Overview

The GreenLang Exit Bar Criteria System is a comprehensive, automated validation framework that ensures all AI agents meet production-grade quality standards before deployment. It implements the 12-dimension framework defined in `GL_agent_requirement.md`.

### Key Features

- **12-Dimension Framework**: Complete coverage of specification, implementation, testing, compliance, and operations
- **Automated Validation**: Machine-readable checklist with automated execution
- **Multiple Output Formats**: Markdown, HTML, JSON, and YAML reports
- **Scoring System**: 100-point scale with clear production readiness thresholds
- **Blocker Identification**: Automatic detection of production-blocking issues
- **Timeline Estimation**: Realistic time-to-production estimates

### Components

1. **`exit_bar_checklist.yaml`** - Machine-readable validation specification
2. **`validate_exit_bar.py`** - Automated validation script
3. **`exit_bar_checklist.md`** - Human-readable checklist for manual review

---

## Quick Start

### 1. Validate a Single Agent

```bash
# Basic validation
python scripts/validate_exit_bar.py --agent carbon_agent

# With specification path
python scripts/validate_exit_bar.py --agent carbon_agent --spec-path specs/domain3_crosscutting/integration/agent_carbon.yaml

# Generate HTML report
python scripts/validate_exit_bar.py --agent carbon_agent --format html --output reports/carbon_agent_exit_bar.html

# Verbose mode
python scripts/validate_exit_bar.py --agent carbon_agent --verbose
```

### 2. Validate All Agents (Batch Mode)

```bash
# Validate all agents in the system
python scripts/validate_exit_bar.py --all-agents --format json --output reports/all_agents_exit_bar.json
```

### 3. Watch Mode (Continuous Validation)

```bash
# Monitor agent for changes and continuously validate
python scripts/validate_exit_bar.py --agent carbon_agent --watch
```

---

## The 12 Dimensions

### Overview Table

| Dimension | Weight | Focus Area | Critical? |
|-----------|--------|------------|-----------|
| **D1: Specification** | 10% | AgentSpec V2.0 completeness | ✅ Yes |
| **D2: Implementation** | 15% | Code quality & architecture | ✅ Yes |
| **D3: Test Coverage** | 15% | Comprehensive testing | ✅ Yes |
| **D4: Deterministic AI** | 10% | Reproducibility guarantees | ✅ Yes |
| **D5: Documentation** | 5% | User & developer docs | ⚠️ Partial |
| **D6: Compliance** | 10% | Security & standards | ✅ Yes |
| **D7: Deployment** | 10% | Production readiness | ⚠️ Partial |
| **D8: Exit Bar** | 10% | Quality gates | ✅ Yes |
| **D9: Integration** | 5% | Agent coordination | ⚠️ Partial |
| **D10: Business Impact** | 5% | Value quantification | ⚠️ Partial |
| **D11: Operations** | 5% | Monitoring & support | ⚠️ Partial |
| **D12: Improvement** | 5% | Version control & feedback | ⚠️ Partial |
| **TOTAL** | **100%** | | |

### Production Threshold: ≥95/100 points

---

## Dimension Details

### D1: Specification Completeness (10 points)

**Purpose**: Ensure agent has complete, validated specification

**Criteria**:
- ✅ AgentSpec V2.0 YAML file exists (2 pts)
- ✅ All 11 mandatory sections present (2 pts)
- ✅ Zero validation errors (2 pts)
- ✅ temperature=0.0 in spec (2 pts)
- ✅ seed=42 in spec (2 pts)

**Validation**:
```bash
python scripts/validate_agent_specs.py specs/path/to/agent.yaml
```

---

### D2: Code Implementation (15 points)

**Purpose**: Ensure production-grade Python implementation

**Criteria**:
- ✅ Implementation file exists (3 pts)
- ✅ Tool-first architecture (3+ tools) (3 pts)
- ✅ ChatSession integration (3 pts)
- ✅ Type hints complete (mypy passes) (3 pts)
- ✅ No hardcoded secrets (3 pts)

**Validation**:
```bash
mypy greenlang/agents/{agent_name}_ai.py --strict
grep -rn 'sk-|api_key' greenlang/agents/{agent_name}_ai.py
```

---

### D3: Test Coverage (15 points)

**Purpose**: Ensure comprehensive testing across all categories

**Criteria**:
- ✅ Test file exists (2 pts)
- ✅ Line coverage ≥80% (5 pts)
- ✅ Unit tests (10+ tests) (2 pts)
- ✅ Integration tests (5+ tests) (2 pts)
- ✅ Determinism tests (3+ tests) (2 pts)
- ✅ Boundary tests (5+ tests) (1 pt)
- ✅ All tests passing (1 pt)

**Validation**:
```bash
pytest tests/agents/test_{agent_name}_ai.py --cov=greenlang.agents.{agent_name}_ai --cov-report=term --cov-fail-under=80
```

---

### D4: Deterministic AI Guarantees (10 points)

**Purpose**: Ensure reproducible results across all runs

**Criteria**:
- ✅ temperature=0.0 in code (3 pts)
- ✅ seed=42 in code (3 pts)
- ✅ All tools deterministic (no randomness) (2 pts)
- ✅ Provenance tracking enabled (2 pts)

**Validation**:
```bash
grep -n 'temperature=0.0' greenlang/agents/{agent_name}_ai.py
grep -n 'seed=42' greenlang/agents/{agent_name}_ai.py
grep -n 'random\.|np.random' greenlang/agents/{agent_name}_ai.py
```

---

### D5: Documentation Completeness (5 points)

**Purpose**: Ensure comprehensive user and developer documentation

**Criteria**:
- ✅ Module docstring present (1 pt)
- ✅ Class docstring present (1 pt)
- ✅ Method docstrings (90%+ coverage) (1 pt)
- ⚠️ README/documentation file (1 pt)
- ⚠️ Example use cases (3+ examples) (1 pt)

---

### D6: Compliance & Security (10 points)

**Purpose**: Ensure security and compliance standards met

**Criteria**:
- ✅ zero_secrets=true in spec (3 pts)
- ✅ sbom_required=true in spec (2 pts)
- ✅ digital_signature=true in spec (1 pt)
- ✅ Standards declared (2+ standards) (2 pts)
- ✅ No hardcoded credentials (2 pts)

**Validation**:
```bash
python -m greenlang.security.scan --secrets greenlang/agents/
pip-audit
```

---

### D7: Deployment Readiness (10 points)

**Purpose**: Ensure agent is production-deployable

**Criteria**:
- ✅ Pack configuration complete (3 pts)
- ✅ Python dependencies declared (2 pts)
- ✅ GreenLang module dependencies declared (2 pts)
- ✅ Resource requirements specified (2 pts)
- ⚠️ API endpoints defined (1 pt)

---

### D8: Exit Bar Criteria (10 points)

**Purpose**: Verify all quality gates passed

**Criteria**:
- ✅ All tests passing (3 pts)
- ✅ Test coverage ≥80% (3 pts)
- ✅ No security issues (2 pts)
- ✅ Specification validation passes (2 pts)

---

### D9: Integration & Coordination (5 points)

**Purpose**: Ensure seamless integration with other agents

**Criteria**:
- ✅ Dependencies declared in spec (2 pts)
- ✅ BaseAgent inheritance (2 pts)
- ✅ AgentResult return type (1 pt)

---

### D10: Business Impact & Metrics (5 points)

**Purpose**: Ensure measurable business value

**Criteria**:
- ✅ Strategic context documented (2 pts)
- ⚠️ Business impact section (2 pts)
- ✅ Performance requirements defined (1 pt)

---

### D11: Operational Excellence (5 points)

**Purpose**: Ensure production operations support

**Criteria**:
- ✅ Logging implementation (3+ log statements) (2 pts)
- ✅ Error handling (2+ try-except blocks) (2 pts)
- ⚠️ Performance tracking (1 pt)

---

### D12: Continuous Improvement (5 points)

**Purpose**: Ensure iterative enhancement capability

**Criteria**:
- ✅ Version control metadata (2 pts)
- ✅ Change log present (1+ entry) (2 pts)
- ✅ Review status documented (1 pt)

---

## Scoring & Readiness Levels

### Score Thresholds

```
100-95:  ✅ PRODUCTION READY     - Deploy to production immediately
 94-80:  ⚠️ PRE-PRODUCTION       - Minor gaps, nearly ready (1-2 weeks)
 79-60:  🟡 DEVELOPMENT          - Major work complete (3-4 weeks)
 59-40:  🟠 EARLY DEVELOPMENT    - Some implementation (6-8 weeks)
 39-20:  🔴 SPECIFICATION ONLY   - Minimal code (12+ weeks)
 19-0:   ⚫ NOT STARTED          - Little to no work
```

### Production Deployment Gates

An agent can be deployed to production ONLY if:

1. **Overall score ≥95/100**
2. **All REQUIRED criteria pass** (marked with ✅)
3. **Zero security issues**
4. **Test coverage ≥80%**
5. **All tests passing**
6. **Specification validation passing**
7. **Approvals from**: Engineering Lead, Security Lead, Product Lead, SRE Lead

---

## Output Formats

### 1. Markdown Report

**Best for**: Human review, GitHub/GitLab, documentation

```bash
python scripts/validate_exit_bar.py --agent carbon_agent --format markdown --output reports/carbon_agent.md
```

**Example Output**:
```markdown
# Exit Bar Validation Report
## Agent: CarbonAgentAI
## Date: 2025-10-16
## Overall Score: 87/100 (PRE-PRODUCTION)

### Dimension Breakdown
| Dimension | Score | Max | Status | Blockers |
|-----------|-------|-----|--------|----------|
| D1: Specification | 10 | 10 | ✅ PASS | 0 |
| D2: Implementation | 15 | 15 | ✅ PASS | 0 |
| D3: Test Coverage | 10 | 15 | ⚠️ PARTIAL | 1 |
...
```

---

### 2. HTML Report

**Best for**: Dashboards, sharing with stakeholders, presentations

```bash
python scripts/validate_exit_bar.py --agent carbon_agent --format html --output reports/carbon_agent.html
```

**Features**:
- Color-coded status indicators
- Responsive tables
- Professional formatting
- Easy sharing

---

### 3. JSON Report

**Best for**: Automation, CI/CD integration, programmatic analysis

```bash
python scripts/validate_exit_bar.py --agent carbon_agent --format json --output reports/carbon_agent.json
```

**Example Output**:
```json
{
  "agent_name": "carbon_agent",
  "validation_date": "2025-10-16T10:30:00",
  "overall_score": 87.0,
  "readiness_status": "PRE-PRODUCTION",
  "blockers": [
    {
      "dimension": "d3_test_coverage",
      "id": "D3.2",
      "name": "Line coverage ≥80%",
      "message": "Coverage below 80%"
    }
  ],
  "dimensions": {
    "d1_specification": {
      "status": "PASS",
      "score": 10,
      "max_score": 10
    }
  }
}
```

---

### 4. YAML Report

**Best for**: Configuration management, infrastructure as code

```bash
python scripts/validate_exit_bar.py --agent carbon_agent --format yaml --output reports/carbon_agent.yaml
```

---

## Usage Examples

### Example 1: Validating CarbonAgentAI

```bash
# Step 1: Run validation
python scripts/validate_exit_bar.py \
  --agent carbon_agent \
  --spec-path specs/domain3_crosscutting/integration/agent_carbon.yaml \
  --format markdown \
  --output reports/carbon_agent_exit_bar.md \
  --verbose

# Step 2: Review report
cat reports/carbon_agent_exit_bar.md

# Step 3: Fix blockers
# (based on recommendations in report)

# Step 4: Re-run validation
python scripts/validate_exit_bar.py --agent carbon_agent

# Step 5: Approve for production (if score ≥95)
```

---

### Example 2: CI/CD Integration

```yaml
# .github/workflows/exit-bar-validation.yml

name: Exit Bar Validation

on:
  pull_request:
    paths:
      - 'greenlang/agents/**'
      - 'tests/agents/**'
      - 'specs/**'

jobs:
  validate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pyyaml

      - name: Run Exit Bar Validation
        run: |
          python scripts/validate_exit_bar.py \
            --agent ${{ matrix.agent }} \
            --format json \
            --output validation_result.json

      - name: Check Score
        run: |
          score=$(jq '.overall_score' validation_result.json)
          if (( $(echo "$score < 95" | bc -l) )); then
            echo "Exit bar validation failed: Score $score < 95"
            exit 1
          fi

      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: exit-bar-report
          path: validation_result.json

    strategy:
      matrix:
        agent:
          - carbon_agent
          - fuel_agent
          - grid_factor_agent
          - recommendation_agent
          - report_agent
```

---

### Example 3: Batch Validation Script

```bash
#!/bin/bash
# validate_all_agents.sh

AGENTS=(
  "carbon_agent"
  "fuel_agent"
  "grid_factor_agent"
  "recommendation_agent"
  "report_agent"
  "industrial_process_heat_agent"
  "boiler_replacement_agent"
  "industrial_heat_pump_agent"
)

mkdir -p reports/exit_bar

for agent in "${AGENTS[@]}"; do
  echo "Validating $agent..."

  python scripts/validate_exit_bar.py \
    --agent "$agent" \
    --format html \
    --output "reports/exit_bar/${agent}_exit_bar.html"

  # Check exit code
  if [ $? -eq 0 ]; then
    echo "✅ $agent: PRODUCTION READY"
  else
    echo "❌ $agent: NOT READY"
  fi

  echo ""
done

echo "Validation complete! Reports in reports/exit_bar/"
```

---

## Common Blockers & Solutions

### Blocker 1: Test Coverage <80%

**Error**: "D3.2: Line coverage ≥80% - FAIL"

**Solution**:
```bash
# 1. Check current coverage
pytest tests/agents/test_{agent}_ai.py --cov=greenlang.agents.{agent}_ai --cov-report=html

# 2. Open coverage report
open htmlcov/index.html

# 3. Identify uncovered lines (red highlighting)

# 4. Add tests for uncovered code paths

# 5. Re-run validation
python scripts/validate_exit_bar.py --agent {agent}
```

---

### Blocker 2: Missing Specification Sections

**Error**: "D1.2: All 11 sections present - FAIL"

**Solution**:
```yaml
# Add missing sections to spec YAML

# If missing 'sub_agents':
sub_agents:
  sub_agent_count: 0
  sub_agents_list: []

# If missing 'testing':
testing:
  test_coverage_target: 0.80
  test_categories:
    - category: "unit_tests"
      description: "Test tool implementations"
      count: 10

# If missing 'business_impact':
business_impact:
  market_opportunity:
    addressable_market: "$XB annually"
```

---

### Blocker 3: Hardcoded Secrets

**Error**: "D2.5: No hardcoded secrets - FAIL"

**Solution**:
```python
# ❌ WRONG - Hardcoded secret
api_key = "sk-1234567890abcdef"

# ✅ CORRECT - Environment variable
import os
api_key = os.getenv("OPENAI_API_KEY")

# ✅ CORRECT - Config file
from greenlang.config import get_secret
api_key = get_secret("OPENAI_API_KEY")
```

---

### Blocker 4: Non-Deterministic Code

**Error**: "D4.3: All tools deterministic - FAIL"

**Solution**:
```python
# ❌ WRONG - Random operations
import random
value = random.random()

# ✅ CORRECT - Deterministic calculation
value = calculate_exact_value(input_data)

# ✅ CORRECT - Seeded random (if truly needed)
import random
random.seed(42)  # But prefer deterministic calculations
```

---

## Customization

### Custom Checklist

Create a custom checklist for specific requirements:

```bash
# Copy default checklist
cp templates/exit_bar_checklist.yaml templates/exit_bar_custom.yaml

# Edit custom checklist
vim templates/exit_bar_custom.yaml

# Use custom checklist
python scripts/validate_exit_bar.py \
  --agent carbon_agent \
  --checklist templates/exit_bar_custom.yaml
```

---

### Custom Scoring Weights

Edit `exit_bar_checklist.yaml`:

```yaml
scoring:
  dimension_weights:
    d1_specification: 15  # Increase from 10
    d2_implementation: 20  # Increase from 15
    d3_test_coverage: 20   # Increase from 15
    # ... adjust others to total 100
```

---

## Best Practices

### 1. Validate Early and Often

```bash
# Validate during development (watch mode)
python scripts/validate_exit_bar.py --agent carbon_agent --watch
```

### 2. Fix Blockers in Order

1. Fix all **D1 (Specification)** issues first
2. Then **D2 (Implementation)**
3. Then **D3 (Test Coverage)**
4. Finally remaining dimensions

### 3. Automate in CI/CD

Add exit bar validation to your CI/CD pipeline to catch issues early.

### 4. Track Progress

```bash
# Weekly validation
python scripts/validate_exit_bar.py --agent carbon_agent --format json --output weekly/week_$(date +%U).json

# Compare progress
jq '.overall_score' weekly/*.json
```

---

## Troubleshooting

### Issue: "FileNotFoundError: Checklist not found"

**Solution**:
```bash
# Ensure you're in the project root
cd /path/to/Code_V1_GreenLang

# Or specify full path
python scripts/validate_exit_bar.py \
  --agent carbon_agent \
  --checklist /full/path/to/exit_bar_checklist.yaml
```

---

### Issue: "Command timeout after 30s"

**Solution**:
Edit `exit_bar_checklist.yaml` to increase timeout:

```yaml
automated_checks:
  - check_id: "test_coverage"
    command: "pytest ..."
    timeout_seconds: 600  # Increase from 300
```

---

### Issue: "Invalid YAML syntax"

**Solution**:
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('templates/exit_bar_checklist.yaml'))"

# Or use online validator
# https://www.yamllint.com/
```

---

## FAQ

### Q: What is the minimum score for production?

**A**: 95/100. Anything below 95 is considered pre-production or development stage.

---

### Q: Can I skip optional criteria?

**A**: Yes, but you'll lose points. Optional criteria (marked ⚠️) are recommended for best practices but not required for production.

---

### Q: How long does validation take?

**A**: Typically 2-5 minutes per agent, depending on test suite size.

---

### Q: Can I run validation on multiple agents in parallel?

**A**: Yes! Use GNU parallel:

```bash
parallel python scripts/validate_exit_bar.py --agent {} ::: carbon_agent fuel_agent grid_factor_agent
```

---

### Q: What if my agent doesn't have a spec file?

**A**: The agent will fail D1 validation. Create a spec using the AgentSpec V2.0 template first.

---

## Support

For questions or issues:

1. Check this README
2. Review `GL_agent_requirement.md` for dimension details
3. Contact GreenLang Framework Team
4. Open an issue in the repository

---

## Version History

### v1.0.0 (2025-10-16)
- Initial release
- 12-dimension framework
- Automated validation script
- Multiple output formats
- Comprehensive documentation

---

## License

Copyright © 2025 GreenLang Framework Team. All rights reserved.

---

**END OF README**
