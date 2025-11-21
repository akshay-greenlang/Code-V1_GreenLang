# GreenLang Scripts Directory

This directory contains all utility scripts for the GreenLang project, organized by purpose.

## Directory Structure

```
scripts/
├── test/          Test, validation, and verification scripts
├── dev/           Development, demos, fixes, and utilities
├── deploy/        Deployment, CI/CD, and release scripts
├── analysis/      Code analysis, metrics, and coverage
├── ci/            Continuous integration scripts
├── metrics/       Performance and operational metrics
└── migration/     Migration and upgrade utilities
```

## Quick Reference

### scripts/test/ (36 scripts)
**Purpose:** Testing, validation, and verification

**Common Scripts:**
- `test_*.py` - Unit and integration tests
- `verify_*.py` - Verification and compliance checks
- `validate_*.py` - Validation scripts
- `check_*.py` - Health and status checks
- `health_check.py` - System health monitoring
- `security_validation_test.py` - Security compliance
- `supply_chain_validation.py` - Supply chain verification

**Usage:**
```bash
# Run acceptance tests
python scripts/test/test_acceptance.py

# Verify security implementation
python scripts/test/verify_security.py

# Validate agent specifications
python scripts/test/validate_agentspec_v2_packs.py
```

---

### scripts/dev/ (24 scripts)
**Purpose:** Development, debugging, demos, and maintenance

**Common Scripts:**
- `fix_*.py` - Bug fixes and patches
- `demo_*.py` - Feature demonstrations
- `sdk_*.py` - SDK examples and tests
- `import_*.py` - Data import utilities
- `ADD_GL_TO_PATH.ps1` - Environment setup
- `web_app.py` - Development web server

**Usage:**
```bash
# Add GreenLang to PATH (Windows)
powershell -ExecutionPolicy Bypass -File scripts/dev/ADD_GL_TO_PATH.ps1

# Run SDK demo
python scripts/dev/sdk_full_demo.py

# Import emission factors
python scripts/dev/import_all_1000_factors.py

# Fix logging issues
python scripts/dev/fix_logging.py
```

---

### scripts/deploy/ (4 scripts)
**Purpose:** Deployment, CI/CD, and release management

**Common Scripts:**
- `run-docker-fix.ps1` - Docker build fixes
- `run-gh-commands.ps1` - GitHub automation
- `sign_artifacts.sh` - Artifact signing
- `run_acceptance.sh` - Acceptance testing

**Usage:**
```bash
# Sign release artifacts
bash scripts/deploy/sign_artifacts.sh

# Run GitHub automation
powershell scripts/deploy/run-gh-commands.ps1

# Acceptance test before deployment
bash scripts/deploy/run_acceptance.sh
```

---

### scripts/analysis/ (5 scripts)
**Purpose:** Code analysis, metrics, and health monitoring

**Common Scripts:**
- `analyze_coverage.py` - Test coverage analysis
- `analyze_pack_schemas.py` - Schema validation
- `extract_coverage.py` - Coverage report extraction
- `code_health_v020.py` - Code health metrics
- `count_loc.sh` - Lines of code counter

**Usage:**
```bash
# Analyze test coverage
python scripts/analysis/analyze_coverage.py

# Check code health
python scripts/analysis/code_health_v020.py

# Count lines of code
bash scripts/analysis/count_loc.sh
```

---

## Migration from Root

All scripts previously in the repository root have been moved to appropriate subdirectories:

**Before:**
```
/
├── test_acceptance.py
├── verify_security.py
├── demo_working_calc.py
├── fix_logging.py
└── ... (69 scripts)
```

**After:**
```
scripts/
├── test/test_acceptance.py
├── test/verify_security.py
├── dev/demo_working_calc.py
├── dev/fix_logging.py
└── ...
```

## Important Notes

1. **setup.py remains in root** - Required for package installation
2. **Update your imports** - If scripts import each other, update paths
3. **CI/CD updates** - Update workflow files to reference new locations
4. **Permissions** - Ensure execute permissions on shell scripts

## Finding Scripts

### By Purpose
```bash
# List all test scripts
ls scripts/test/

# List all development scripts
ls scripts/dev/

# List all deployment scripts
ls scripts/deploy/

# List all analysis scripts
ls scripts/analysis/
```

### By Name
```bash
# Find a specific script
find scripts/ -name "*security*"

# Find all Python scripts
find scripts/ -name "*.py"

# Find all shell scripts
find scripts/ -name "*.sh"
```

## Contributing

When adding new scripts:

1. **Choose the right directory:**
   - Testing/validation → `scripts/test/`
   - Development/demos → `scripts/dev/`
   - Deployment/CI → `scripts/deploy/`
   - Analysis/metrics → `scripts/analysis/`

2. **Follow naming conventions:**
   - Tests: `test_*.py`
   - Verification: `verify_*.py`
   - Validation: `validate_*.py`
   - Fixes: `fix_*.py`
   - Demos: `demo_*.py`

3. **Add documentation:**
   - Include docstrings
   - Add usage examples
   - Update this README

4. **Set permissions:**
   ```bash
   chmod +x scripts/deploy/your_script.sh
   ```

## Troubleshooting

### Script Not Found
```bash
# Old location (doesn't work)
python test_acceptance.py

# New location (works)
python scripts/test/test_acceptance.py
```

### Import Errors
Update relative imports in moved scripts:
```python
# Before
from fix_logging import setup_logging

# After
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.dev.fix_logging import setup_logging
```

### Permission Denied
```bash
# Grant execute permissions
chmod +x scripts/deploy/sign_artifacts.sh
```

## Related Documentation

- [REPOSITORY_CLEANUP_REPORT.md](../REPOSITORY_CLEANUP_REPORT.md) - Detailed cleanup documentation
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [README.md](../README.md) - Main project documentation

---

**Last Updated:** 2025-11-21
**Maintainer:** GL-DevOpsEngineer
**Version:** 1.0
