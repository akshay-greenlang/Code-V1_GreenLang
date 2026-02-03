# GreenLang v0.3.0 Release Checklist

**Target Release Date:** February 2026
**Version:** 0.3.0 (Beta)

## Pre-Release Checks

### Code Quality
- [ ] All tests pass (`pytest tests/`)
- [ ] Lint checks pass (`ruff check greenlang/`)
- [ ] Type checks pass (`mypy greenlang/`)
- [ ] Coverage >= 85%
- [ ] No critical security vulnerabilities

### Documentation
- [x] Pack Spec v1.0 documented (`docs/specs/pack-spec-v1.0.md`)
- [x] Execution Model v1.0 documented (`docs/specs/execution-model-v1.0.md`)
- [x] Determinism Contract v1.0 documented (`docs/specs/determinism-contract-v1.0.md`)
- [x] Provenance Contract v1.0 documented (`docs/specs/provenance-contract-v1.0.md`)
- [x] Factor & Unit Contract v1.0 documented (`docs/specs/factor-unit-contract-v1.0.md`)
- [x] CBAM Quickstart guide (`docs/quickstart/cbam-workflow.md`)
- [x] README updated with clear branding

### Schema Validation
- [x] Pack JSON Schema created (`greenlang/specs/schemas/pack-v1.0.schema.json`)
- [x] Pipeline JSON Schema created (`greenlang/specs/schemas/pipeline-v1.0.schema.json`)
- [ ] Schema validation tests pass

### Core Features
- [x] AST evaluator supports container literals (lists, tuples, dicts)
- [x] Condition evaluation errors are logged (not silent)
- [ ] Safe evaluator tests pass (`tests/test_safe_evaluator.py`)

### Dependencies
- [x] LLM dependencies (openai, anthropic) are optional
- [ ] Core runs without AI packages installed
- [ ] Optional extras defined in pyproject.toml

## Release Process

### 1. Version Bump
```bash
# Verify version in pyproject.toml
grep "version = " pyproject.toml
# Expected: version = "0.3.0"
```

### 2. Create Release Tag
```bash
git tag -a v0.3.0 -m "Release v0.3.0 - Core Contracts & Determinism"
git push origin v0.3.0
```

### 3. GitHub Actions Pipeline
The release-orchestration workflow will automatically:
1. Validate version consistency
2. Build wheels and sdist
3. Run multi-OS, multi-Python tests
4. Sign with Sigstore
5. Generate SBOM
6. Publish to PyPI (trusted publishing)
7. Build and push Docker images
8. Create GitHub release

### 4. Manual Verification
```bash
# Install from PyPI
pip install greenlang-cli==0.3.0

# Verify installation
gl --version

# Run sample calculation
gl run demo --help
```

### 5. Post-Release
- [ ] Announce on Discord
- [ ] Update documentation site
- [ ] Close release milestone
- [ ] Create v0.4.0 milestone

## Rollback Procedure

If critical issues are found:

1. **Yank from PyPI:**
   ```bash
   # Only if absolutely necessary
   pip install twine
   twine yank greenlang-cli 0.3.0
   ```

2. **Delete tag:**
   ```bash
   git tag -d v0.3.0
   git push origin :refs/tags/v0.3.0
   ```

3. **Revert and fix:**
   ```bash
   git revert HEAD
   # Fix issues
   git push
   ```

## Known Issues

- Release/0.3.0 had permission issues in GitHub Actions (now fixed)
- TestPyPI dry-run recommended before production release

## Release Notes Template

```markdown
# GreenLang v0.3.0

## Highlights

- **Core Contracts**: Formal specifications for Pack, Execution, Determinism, Provenance, and Factor/Unit contracts
- **Hardened Expression Language**: AST evaluator now supports container literals (lists, tuples, dicts, sets)
- **Clear Branding**: "GreenLang is a deterministic execution engine + pack format for climate calculations"
- **CBAM Flagship**: Complete workflow documentation with verification examples

## Breaking Changes

None

## New Features

- Pack Spec v1.0 with JSON Schema validation
- Pipeline Spec v1.0 (GLIP protocol)
- Container literal support in step conditions
- Improved condition evaluation error handling

## Bug Fixes

- Condition evaluation errors now logged with full context
- Silent step skipping eliminated

## Documentation

- Complete specification documents for all core contracts
- CBAM quickstart guide with verification examples
- Updated README with clear product definition
```
