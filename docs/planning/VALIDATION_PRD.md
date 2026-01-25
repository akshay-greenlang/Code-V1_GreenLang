# GreenLang Codebase Validation PRD

## Overview
Post-reorganization validation to confirm the GreenLang codebase is professional-grade, well-organized, and maintainable.

## CRITICAL CONSTRAINTS
- **DO NOT MODIFY**: `2026_PRD_MVP/` directory
- **DO NOT MODIFY**: `cbam-pack-mvp/` directory
- Document all findings thoroughly
- Create actionable recommendations

---

## Phase 1: Commit Review & Change Audit

### Task 1.1: Review Reorganization Commits
Analyze each of the 5 reorganization commits:
1. `4d57b5fd` - Reorganize root directory
2. `77eaea70` - Consolidate duplicate module directories
3. `d6116d8c` - Fix import errors (first pass)
4. `8d691ee2` - Fix import errors (second pass)
5. `29844a56` - Update documentation

For each commit:
- List files changed and their purpose
- Verify changes align with reorganization goals
- Identify any potential issues or regressions
- Document the change impact

### Task 1.2: Verify File Integrity
Ensure no files were lost or corrupted during reorganization:
- Compare file counts before/after
- Verify critical files exist in new locations
- Check that no duplicate files remain
- Validate symlinks if any were created

---

## Phase 2: Test Suite Validation

### Task 2.1: Run Full Test Suite
Execute comprehensive test suite:
```bash
pytest tests/ -v --tb=short
```

Document:
- Total tests run
- Tests passed
- Tests failed (with details)
- Tests skipped
- Coverage percentage if available

### Task 2.2: Test Import Resolution
Verify all imports work correctly:
```bash
python -c "import greenlang; print('Core import OK')"
python -c "from greenlang.agents import *; print('Agents import OK')"
python -c "from greenlang.api import *; print('API import OK')"
python -c "from greenlang.calculations import *; print('Calculations import OK')"
```

### Task 2.3: Validate Package Build
Ensure package can be built:
```bash
python -m build --sdist --wheel
```

---

## Phase 3: CI/CD Pipeline Updates

### Task 3.1: Audit CI/CD Workflows
Review all files in `.github/workflows/`:
- List all workflow files
- Identify any hardcoded paths that reference old locations
- Check for references to moved directories

### Task 3.2: Update Workflow Paths
For each workflow file with outdated paths:
- Update references from old paths to new paths
- Common changes needed:
  - `k8s/` → `deployment/kubernetes/`
  - `docker/` → `deployment/docker/`
  - `helm/` → `deployment/helm/`
  - `terraform/` → `deployment/terraform/`
  - `infrastructure/` → `deployment/`
  - `load-tests/` → `tests/load/`
  - Root-level scripts → `scripts/`

### Task 3.3: Update Makefile
Review and update `Makefile` for any path references.

### Task 3.4: Update Docker Compose
Review `docker-compose.yml` and variants for path references.

### Task 3.5: Validate CI/CD Syntax
Run workflow validation:
```bash
# Check YAML syntax
python -c "import yaml; [yaml.safe_load(open(f)) for f in __import__('glob').glob('.github/workflows/*.yml')]"
```

---

## Phase 4: Codebase Quality Assessment

### Task 4.1: Directory Structure Analysis
Evaluate the new structure against best practices:

**Criteria:**
- [x] Clear separation of concerns
- [x] Logical grouping of related files
- [x] Consistent naming conventions
- [ ] Appropriate depth (not too deep, not too flat)
- [ ] Standard Python project layout
- [ ] Documentation accessibility
- [ ] Test organization

### Task 4.2: Import Graph Analysis
Check for circular dependencies:
```bash
# Using pydeps or similar tool if available
python -c "
import sys
sys.path.insert(0, '.')
try:
    import greenlang
    print('No circular import errors detected')
except ImportError as e:
    print(f'Import error: {e}')
"
```

### Task 4.3: Code Quality Metrics
Run linting and type checking:
```bash
ruff check greenlang/ --statistics
mypy greenlang/ --ignore-missing-imports --no-error-summary 2>&1 | tail -20
```

### Task 4.4: Documentation Completeness
Verify documentation structure:
- README.md exists and is comprehensive
- CONTRIBUTING.md exists
- CHANGELOG.md exists
- LICENSE exists
- API documentation in docs/api/
- Architecture docs in docs/architecture/

---

## Phase 5: Professional-Grade Assessment

### Task 5.1: Industry Standards Compliance
Evaluate against Python packaging standards:
- [ ] pyproject.toml properly configured
- [ ] setup.py for backward compatibility
- [ ] MANIFEST.in for source distribution
- [ ] .gitignore comprehensive
- [ ] .dockerignore for Docker builds

### Task 5.2: Security Posture
Verify security configurations:
- [ ] .env files in .gitignore
- [ ] No hardcoded secrets in code
- [ ] Security scanning configs present (.gitleaks.toml, .trufflehog.toml)
- [ ] SECURITY.md exists

### Task 5.3: Maintainability Score
Assess maintainability factors:
- [ ] Modular architecture
- [ ] Clear module boundaries
- [ ] Consistent code style
- [ ] Adequate test coverage
- [ ] Documentation coverage
- [ ] Dependency management

### Task 5.4: Generate Assessment Report
Create a comprehensive report at `docs/reports/CODEBASE_ASSESSMENT_REPORT.md`:
- Executive summary
- Structure analysis
- Test results
- CI/CD status
- Quality metrics
- Recommendations
- Overall grade (A-F)

---

## Success Criteria
- [ ] All 5 commits reviewed and documented
- [ ] Test suite passes (or failures documented)
- [ ] All CI/CD paths updated
- [ ] Import resolution verified
- [ ] Assessment report generated
- [ ] Overall grade: A or B

## Deliverables
1. `docs/reports/COMMIT_REVIEW_REPORT.md` - Detailed commit analysis
2. `docs/reports/TEST_RESULTS_REPORT.md` - Test execution results
3. `docs/reports/CICD_UPDATE_LOG.md` - CI/CD changes made
4. `docs/reports/CODEBASE_ASSESSMENT_REPORT.md` - Final assessment with grade
