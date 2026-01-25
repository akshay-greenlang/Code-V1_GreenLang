# GreenLang Dependency Management Strategy

## Overview

This document defines the dependency management strategy for the GreenLang monorepo, ensuring consistency, security, and maintainability across all packages.

## Version Constraint Philosophy

### Two-File Strategy

We use a dual-file approach for dependency management:

1. **`pyproject.toml`** - Uses compatible release constraints (`~=`)
2. **`requirements.txt`** - Uses exact version pinning (`==`)

### Why This Approach?

- **`pyproject.toml`**: Defines the package's abstract dependencies and compatible version ranges
  - Published to PyPI (if applicable)
  - Used by package installers to resolve dependencies
  - Allows flexibility for downstream consumers

- **`requirements.txt`**: Locks exact versions for reproducible builds
  - Used for production deployments
  - Ensures identical environments across dev/staging/prod
  - Security audits reference exact versions

## Version Constraint Syntax

### Compatible Release Operator (`~=`)

The `~=` operator is used in `pyproject.toml` to allow patch-level updates while blocking minor/major version changes.

**Syntax:**
```toml
dependency~=X.Y.Z
```

**Behavior:**
- `~=1.4.5` → Allows `>=1.4.5, <1.5.0`
- `~=2.1.0` → Allows `>=2.1.0, <2.2.0`
- `~=0.4.22` → Allows `>=0.4.22, <0.5.0`

**Benefits:**
- Automatic patch-level security updates
- Prevents breaking changes from minor/major version bumps
- Aligns with semantic versioning (SemVer)

### Exact Pinning (`==`)

Used in `requirements.txt` for complete reproducibility.

**Syntax:**
```txt
dependency==X.Y.Z
```

**Example:**
```txt
fastapi==0.109.2
pydantic==2.5.3
cryptography==42.0.5
```

## Monorepo Package Strategy

### Problem: Circular Dependencies

Apps like `GL-VCCI-Carbon-APP` referenced non-existent packages:
- `greenlang-core>=0.3.0`
- `greenlang-agents>=0.3.0`
- `greenlang-validation>=0.3.0`

These packages **do not exist** as separate PyPI distributions.

### Solution: Local Editable Installs

**Option A: Install from monorepo root**
```bash
# Install main GreenLang CLI package (includes core functionality)
pip install -e .

# Install agent foundation
pip install -e ./GreenLang_2030/agent_foundation
```

**Option B: Relative path installs**
```bash
# From within an app directory (e.g., GL-VCCI-Carbon-APP)
pip install -e ../../
pip install -e ../../GreenLang_2030/agent_foundation
```

**Option C: Future PyPI packages**

When these packages are published to PyPI, update `requirements.txt`:
```txt
greenlang-cli>=0.3.0,<0.4.0
greenlang-agent-foundation>=1.0.0,<2.0.0
```

### Package Structure

```
Code-V1_GreenLang/
├── pyproject.toml              # greenlang-cli (main package)
├── requirements.txt            # Pinned versions for greenlang-cli
├── core/                       # Core utilities (not a separate package yet)
├── greenlang/                  # Intelligence framework (not separate yet)
├── GreenLang_2030/
│   └── agent_foundation/
│       ├── pyproject.toml      # greenlang-agent-foundation
│       └── requirements.txt    # Pinned versions
├── GL-VCCI-Carbon-APP/
│   └── VCCI-Scope3-Platform/
│       └── requirements.txt    # App-specific deps (references local installs)
└── GL-CSRD-APP/
    └── CSRD-Reporting-Platform/
        └── requirements.txt    # App-specific deps (references local installs)
```

## Security Best Practices

### 1. Regular Security Audits

**Tools:**
- `pip-audit` - Scan for known vulnerabilities
- `safety` - Check against safety-db
- `bandit` - Static code analysis for security issues

**Schedule:**
- Daily: Automated scans via CI/CD
- Weekly: Dependabot security updates
- Monthly: Manual security reviews
- Quarterly: Comprehensive audits

### 2. Critical CVE Response

When a critical CVE is discovered:

1. **Update `requirements.txt` immediately** with patched version
2. **Update `pyproject.toml`** to reflect new minimum version
3. **Test thoroughly** in staging environment
4. **Deploy to production** after validation
5. **Document** in security audit logs

**Example (CVE-2024-0727 in cryptography):**
```diff
# requirements.txt
- cryptography==42.0.2
+ cryptography==42.0.5  # CVE-2024-0727 fix

# pyproject.toml
dependencies = [
-   "cryptography~=42.0.2",
+   "cryptography~=42.0.5",
]
```

### 3. Upper Bounds Protection

**Always use upper bounds** to prevent breaking changes:

❌ **Bad:**
```toml
fastapi>=0.104.1  # Could install 1.0.0 with breaking changes
```

✅ **Good:**
```toml
fastapi~=0.109.2  # Blocks 0.110.0, allows 0.109.x patches
```

## Dependency Update Workflow

### Routine Updates (Monthly)

```bash
# 1. Check for outdated packages
pip list --outdated

# 2. Update requirements.txt with new versions
pip install --upgrade -r requirements.txt
pip freeze > requirements-new.txt

# 3. Update pyproject.toml with compatible release constraints
# Manually edit to use ~= with new base versions

# 4. Run tests
pytest

# 5. Update security audit logs
# Document any CVE fixes or breaking changes

# 6. Commit changes
git add requirements.txt pyproject.toml
git commit -m "Update dependencies - security patches and compatibility"
```

### Emergency Security Updates

```bash
# 1. Identify vulnerable package
pip-audit

# 2. Update to patched version immediately
pip install cryptography==42.0.5

# 3. Update requirements.txt
pip freeze | grep cryptography >> requirements.txt

# 4. Update pyproject.toml
# Edit manually to update ~= constraint

# 5. Deploy after minimal smoke tests
pytest tests/security/

# 6. Full regression testing post-deployment
```

## Version Alignment Checklist

When updating dependencies, ensure:

- [ ] `requirements.txt` uses exact pinning (`==`)
- [ ] `pyproject.toml` uses compatible release (`~=`)
- [ ] Base versions align (e.g., `pydantic~=2.5.3` and `pydantic==2.5.3`)
- [ ] No circular dependencies (greenlang-core, greenlang-agents)
- [ ] All dependencies have upper bounds
- [ ] Security audit completed for new versions
- [ ] License compliance verified (no GPL in production)
- [ ] Tests pass with new versions
- [ ] Documentation updated (CHANGELOG, security notes)

## Example: Proper Dependency File

### `pyproject.toml`

```toml
[project]
name = "greenlang-cli"
version = "0.3.0"
requires-python = ">=3.10"

dependencies = [
  "typer~=0.9.0",
  "pydantic~=2.5.3",
  "pyyaml~=6.0.1",
  "rich~=13.7.0",
  "fastapi~=0.109.2",
  "cryptography~=42.0.5",
]

[project.optional-dependencies]
llm = [
  "openai~=1.12.0",
  "anthropic~=0.18.1",
  "langchain~=0.1.9",
]

security = [
  "cryptography~=42.0.5",
  "PyJWT~=2.8.0",
]
```

### `requirements.txt`

```txt
# GreenLang Core Requirements
# Security Hardened with Pinned Versions
# Generated: 2025-01-15
# Python >= 3.11 required
# Last Security Audit: 2025-01-15

# Core Dependencies
typer==0.9.0
pydantic==2.5.3
pyyaml==6.0.1
rich==13.7.0
fastapi==0.109.2
cryptography==42.0.5  # CVE-2024-0727 fix

# Optional: LLM Dependencies
openai==1.12.0
anthropic==0.18.1
langchain==0.1.9
```

## Quality Score Impact

Proper dependency management contributes to the overall package quality score:

- **Dependency Health**: 25 points
  - Proper version constraints: 10 points
  - No circular dependencies: 8 points
  - Security patches current: 7 points

- **Version Management**: 10 points
  - SemVer compliance: 5 points
  - Upper bounds defined: 3 points
  - Alignment between files: 2 points

## References

- [PEP 440 - Version Identifiers](https://peps.python.org/pep-0440/)
- [Semantic Versioning 2.0.0](https://semver.org/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [NIST CVE Database](https://nvd.nist.gov/)

## Support

For questions or issues with dependency management:

- **Email:** devops@greenlang.io
- **Slack:** #greenlang-devops
- **Issues:** https://github.com/greenlang/greenlang/issues

---

**Last Updated:** 2025-11-21
**Maintainer:** GreenLang DevOps Team
**Status:** Active
