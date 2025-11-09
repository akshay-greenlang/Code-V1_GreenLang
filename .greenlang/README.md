# GreenLang Infrastructure-First Enforcement

**Version:** 1.0.0
**Status:** Production Ready
**Last Updated:** 2024-11-09

## Overview

This directory contains the complete enforcement system to ensure all GreenLang code uses infrastructure first, with custom implementations only when necessary and properly documented via Architecture Decision Records (ADRs).

## Quick Start

### Installation

```bash
# Run installation script
bash .greenlang/scripts/install_enforcement.sh

# Or manual installation
cp .greenlang/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Daily Usage

```bash
# Before committing
python .greenlang/linters/infrastructure_first.py
python .greenlang/scripts/calculate_ium.py

# Commit (pre-commit hook runs automatically)
git add .
git commit -m "feat: Add feature"

# Create PR (GitHub Actions runs automatically)
```

## Directory Structure

```
.greenlang/
├── README.md                           # This file
├── ENFORCEMENT_GUIDE.md                # Complete enforcement guide
│
├── hooks/
│   └── pre-commit                      # Pre-commit hook (Python)
│
├── linters/
│   └── infrastructure_first.py         # Static analysis linter
│
├── policies/
│   └── infrastructure-first.rego       # OPA runtime policy
│
├── scripts/
│   ├── install_enforcement.sh          # Installation script
│   ├── calculate_ium.py                # Infrastructure Usage Metrics calculator
│   └── test_enforcement.py             # Test/demo script
│
└── adrs/
    ├── TEMPLATE.md                     # ADR template
    └── EXAMPLE-20241109-custom-climate-model.md  # Example ADR
```

## Components

### 1. Pre-Commit Hook

**File:** `hooks/pre-commit`
**Type:** Python script
**Runs:** Before every commit (local)

**Features:**
- AST-based analysis
- Detects forbidden imports (openai, anthropic, redis, etc.)
- Validates agent inheritance
- Checks for LLM/auth code without greenlang imports
- ANSI colored output

**Installation:**
```bash
cp .greenlang/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### 2. Static Linter

**File:** `linters/infrastructure_first.py`
**Type:** Python CLI tool
**Runs:** On-demand or in CI

**Violation Codes:**
- `FORBIDDEN_IMPORT` - Direct import of forbidden module
- `CUSTOM_AGENT` - Agent not inheriting from greenlang.sdk.base.Agent
- `CUSTOM_LLM` - Custom LLM client usage
- `CUSTOM_AUTH` - Custom auth implementation
- `DIRECT_DB` - Direct database access

**Usage:**
```bash
# Lint entire codebase
python .greenlang/linters/infrastructure_first.py

# Lint specific app
python .greenlang/linters/infrastructure_first.py --path apps/GL-CBAM-APP

# JSON output for CI
python .greenlang/linters/infrastructure_first.py --output json
```

### 3. IUM Calculator

**File:** `scripts/calculate_ium.py`
**Type:** Python CLI tool
**Runs:** On-demand or in CI

**Metrics:**
- Import compliance (weight: 2)
- Agent inheritance (weight: 3)
- LLM calls (weight: 3)
- Auth operations (weight: 2)
- Cache operations (weight: 1)
- Database operations (weight: 1)

**Usage:**
```bash
# Calculate overall IUM
python .greenlang/scripts/calculate_ium.py

# By app
python .greenlang/scripts/calculate_ium.py --app GL-CBAM-APP

# JSON + Markdown
python .greenlang/scripts/calculate_ium.py --output both \
    --output-file ium.json --markdown-file ium.md
```

### 4. GitHub Actions Workflow

**File:** `../.github/workflows/greenlang-first-enforcement.yml`
**Type:** GitHub Actions YAML
**Runs:** On every PR and push to main

**Steps:**
1. Run static analysis
2. Calculate IUM
3. Check for ADRs
4. Run OPA tests
5. Generate report
6. Comment on PR
7. Pass/fail based on criteria

**Success Criteria:**
- No violations OR
- IUM >= 95% OR
- ADR exists and approved

### 5. OPA Policy

**File:** `policies/infrastructure-first.rego`
**Type:** Rego policy
**Runs:** At runtime (if OPA integrated)

**Enforces:**
- API auth tokens
- LLM calls through ChatSession
- Cache through CacheManager
- Agent inheritance
- ADR overrides

**Test:**
```bash
opa test .greenlang/policies/infrastructure-first.rego
```

### 6. ADR System

**Directory:** `adrs/`
**Template:** `adrs/TEMPLATE.md`
**Example:** `adrs/EXAMPLE-20241109-custom-climate-model.md`

**Process:**
1. Copy template
2. Fill in all sections
3. Get 2+ approvals
4. Update status to "Accepted"
5. Reference in code/PR

## Forbidden Imports

The following direct imports are forbidden:

| Import | Reason | Use Instead |
|--------|--------|-------------|
| `openai` | Custom LLM client | `greenlang.intelligence.ChatSession` |
| `anthropic` | Custom LLM client | `greenlang.intelligence.ChatSession` |
| `redis` | Direct cache access | `greenlang.cache.CacheManager` |
| `pymongo` | Direct database | `greenlang.db` connectors |
| `motor` | Direct async DB | `greenlang.db` async connectors |
| `sqlalchemy` | Direct ORM | `greenlang.db` ORM layer |
| `jose` | Custom JWT | `greenlang.auth.AuthManager` |
| `jwt` / `pyjwt` | Custom JWT | `greenlang.auth.AuthManager` |
| `passlib` | Custom password hash | `greenlang.auth.AuthManager` |
| `bcrypt` | Custom password hash | `greenlang.auth.AuthManager` |
| `requests` | No retry/timeout | `greenlang.http` client |

## Infrastructure Usage Metrics (IUM)

### Score Calculation

IUM is a weighted average:

```
IUM = (
    2 * import_score +
    3 * agent_score +
    3 * llm_score +
    2 * auth_score +
    1 * cache_score +
    1 * db_score
) / total_weight
```

### Target Scores

- **95%+** - Excellent, fully compliant
- **85-94%** - Good, some custom code with ADRs
- **75-84%** - Needs improvement, create ADRs
- **<75%** - Concerning, review custom implementations

### Interpreting Results

```
Overall Score: 92.3%

Details:
  Imports        : 100.0% (25/25)    ← All imports are greenlang.*
  Agents         :  90.0% (9/10)     ← 1 custom agent needs ADR
  LLM            :  85.0% (17/20)    ← 3 direct OpenAI calls need fixing
  Auth           : 100.0% (15/15)    ← All auth uses greenlang.auth
  Cache          :  80.0% (4/5)      ← 1 direct Redis call
  Database       : 100.0% (8/8)      ← All DB through greenlang.db
```

## Common Workflows

### Adding New Feature

```bash
# 1. Develop using greenlang infrastructure
vim my_feature.py

# 2. Check compliance
python .greenlang/linters/infrastructure_first.py --path my_feature.py

# 3. Calculate IUM
python .greenlang/scripts/calculate_ium.py

# 4. Commit (pre-commit runs)
git add my_feature.py
git commit -m "feat: Add feature"

# 5. Push and create PR (GitHub Actions runs)
git push origin feature-branch
```

### Using Custom Implementation

```bash
# 1. Create ADR
cp .greenlang/adrs/TEMPLATE.md .greenlang/adrs/20241109-custom-impl.md
vim .greenlang/adrs/20241109-custom-impl.md

# 2. Get approvals (2+ reviewers)

# 3. Update ADR status to "Accepted"

# 4. Implement custom code
vim my_custom_impl.py

# 5. Reference ADR in code
# Add comment: # ADR-001: Custom implementation for X
# See: .greenlang/adrs/20241109-custom-impl.md

# 6. Commit and reference ADR in PR
git commit -m "feat: Custom impl (ADR-001)"
```

### Fixing Violations

```bash
# 1. Run linter to see violations
python .greenlang/linters/infrastructure_first.py

# 2. See specific violation
# ✗ Line 15 [FORBIDDEN_IMPORT] Direct import of 'openai'
#   → Use greenlang.intelligence.ChatSession instead

# 3. Fix by using greenlang infrastructure
# Replace: import openai
# With:    from greenlang.intelligence import ChatSession

# 4. Verify fix
python .greenlang/linters/infrastructure_first.py

# 5. Commit
git commit -m "fix: Use greenlang.intelligence instead of openai"
```

## Testing

### Test Enforcement System

```bash
# Run test script
python .greenlang/scripts/test_enforcement.py
```

This creates example violations and shows how they're caught.

### Manual Testing

```bash
# 1. Create test file with violation
echo "import openai" > test_violation.py

# 2. Run linter
python .greenlang/linters/infrastructure_first.py --path test_violation.py

# 3. Should see violation
# ✗ test_violation.py:1
#   Forbidden import: 'openai'
#   → Use greenlang.intelligence.ChatSession instead
```

## CI/CD Integration

### GitHub Actions (Recommended)

Workflow already included: `.github/workflows/greenlang-first-enforcement.yml`

Runs automatically on:
- Pull requests
- Pushes to main

### GitLab CI

```yaml
# .gitlab-ci.yml
enforce-infrastructure-first:
  stage: test
  script:
    - python .greenlang/linters/infrastructure_first.py
    - python .greenlang/scripts/calculate_ium.py
    - |
      IUM_SCORE=$(python .greenlang/scripts/calculate_ium.py --output json | jq '.overall.percentage')
      if (( $(echo "$IUM_SCORE < 95" | bc -l) )); then
        echo "IUM score $IUM_SCORE% below 95% threshold"
        exit 1
      fi
```

### Jenkins

```groovy
// Jenkinsfile
stage('Infrastructure Enforcement') {
    steps {
        sh 'python .greenlang/linters/infrastructure_first.py --output json > violations.json'
        sh 'python .greenlang/scripts/calculate_ium.py --output json > ium.json'

        script {
            def violations = readJSON file: 'violations.json'
            if (violations.summary.errors > 0) {
                error "Infrastructure violations found"
            }
        }
    }
}
```

## Troubleshooting

See `ENFORCEMENT_GUIDE.md` for detailed troubleshooting.

### Common Issues

**Q: Pre-commit hook not running**
```bash
# Reinstall
cp .greenlang/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**Q: False positive violation**
```bash
# If genuinely needed, create ADR
cp .greenlang/adrs/TEMPLATE.md .greenlang/adrs/20241109-my-case.md
```

**Q: IUM score too low**
```bash
# See detailed breakdown
python .greenlang/scripts/calculate_ium.py --output markdown
```

## Support

- **Documentation:** `ENFORCEMENT_GUIDE.md`
- **Issues:** GitHub issues with `enforcement` label
- **Questions:** #greenlang-infrastructure Slack
- **ADR Reviews:** Tag @architecture-team

## Version History

- **v1.0.0** (2024-11-09) - Initial release
  - Pre-commit hook
  - Static linter
  - IUM calculator
  - GitHub Actions workflow
  - OPA policy
  - ADR system
  - Complete documentation

## License

Same as GreenLang core (see main repo LICENSE)

---

**Maintained by:** GreenLang Infrastructure Team
**Contact:** infrastructure@greenlang.io
