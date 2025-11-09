## 📝 Description

<!-- Provide a brief description of the changes in this PR -->

## 🎯 Type of Change

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Configuration change
- [ ] ♻️ Refactoring

## ✅ Acceptance Checklist

### Infrastructure-First Requirements (MANDATORY)

- [ ] **I checked if GreenLang infrastructure can be used** - Reviewed all code changes for opportunities to use existing GreenLang modules
- [ ] **ADR created if custom code needed** - Created Architecture Decision Record in `.greenlang/adrs/` if custom implementation is required
- [ ] **Infrastructure usage metrics checked** - Ran `python .greenlang/scripts/calculate_ium.py` and IUM score is >= 95%
- [ ] **All agents inherit from greenlang.sdk.base.Agent** - No custom agent base classes without ADR approval
- [ ] **All LLM calls use greenlang.intelligence.ChatSession** - No direct OpenAI/Anthropic client usage
- [ ] **All auth uses greenlang.auth** - No custom JWT/password handling without ADR
- [ ] **No forbidden imports** - Ran `python .greenlang/linters/infrastructure_first.py` with 0 violations

See `.greenlang/ENFORCEMENT_GUIDE.md` for details.

### Core Functionality

#### 1. Pack Scaffolding ⏱️
- [ ] `gl init pack-basic <name>` completes in < 60 seconds
- [ ] Generated pack contains all required files (pack.yaml, gl.yaml, CARD.md)
- [ ] `gl pack validate` passes on generated pack

#### 2. Publish → Add Workflow 📦
- [ ] `gl pack publish` executes: tests → policy → SBOM → sign → push
- [ ] `gl pack add <ref>` pulls, verifies, and installs successfully
- [ ] Installed pack is immediately usable

#### 3. Deterministic Runs 🔄
- [ ] `gl run` produces byte-stable run.json
- [ ] Consecutive runs with same inputs are identical
- [ ] Golden tests pass in CI

#### 4. Policy Enforcement 🛡️
- [ ] GPL licensed packs are blocked with clear error
- [ ] Non-allowlisted URLs are blocked with remediation steps
- [ ] `--explain` flag provides helpful guidance

#### 5. Verify Command 🔍
- [ ] `gl verify` shows signer identity
- [ ] SBOM summary is displayed (dependencies, licenses)
- [ ] Provenance information shown (commit, timestamp)

#### 6. Reference Packs Performance 🎯
- [ ] boiler-solar runs successfully (local & k8s)
- [ ] hvac-measures runs successfully (local & k8s)
- [ ] cement-lca runs successfully (local & k8s)
- [ ] All packs complete in < 60s (p95)
- [ ] PDF reports are generated

### Testing

- [ ] Unit tests pass (`pytest tests/`)
- [ ] Integration tests pass
- [ ] Acceptance tests pass (`python acceptance_test.py`)
- [ ] No regression in performance benchmarks

### Documentation

- [ ] README updated with new commands
- [ ] CHANGELOG.md updated
- [ ] API documentation updated if applicable
- [ ] Migration guide updated if breaking changes

### Security

- [ ] Security scan passed (Trivy/Snyk)
- [ ] SBOM generated for release
- [ ] No secrets or credentials in code
- [ ] Dependencies updated to latest secure versions
- [ ] **Security gate respected (default-deny; signed artifacts)**
- [ ] **User-facing change has docs**
- [ ] **Example updated/added**
- [ ] **Demo smoke passes locally**

## 🧪 How to Test

```bash
# Run acceptance tests
python acceptance_test.py --verbose

# Test specific feature
gl <command> <args>

# Verify determinism
gl run pipeline.yaml --deterministic
```

## 📊 Performance Impact

- [ ] No significant performance regression
- [ ] Memory usage within limits (< 1GB)
- [ ] Execution time within SLA (< 60s for reference packs)

## 📸 Screenshots/Output

<!-- If applicable, add screenshots or command output -->

```
$ gl pack validate
✅ Pack validation successful
  - Manifest: Valid
  - Pipeline: Valid
  - CARD.md: Present
  - Tests: 5 found
```

## 🔗 Related Issues

<!-- Link related issues -->
Fixes #
Related to #

## 📋 Pre-merge Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] CI/CD pipeline green
- [ ] Approved by required reviewers

## 👥 Reviewers

<!-- Tag relevant reviewers -->
- Engineering: @
- Security: @
- DevOps: @

---

**Note:** All checkboxes must be checked before merge. Run `python acceptance_test.py` locally to verify.