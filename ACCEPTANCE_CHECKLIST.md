# GreenLang Acceptance Checklist

## PR Description Template

### 🚀 GreenLang Unified CLI Migration

This PR completes the migration to the unified `gl` CLI with pack-based architecture, policy enforcement, and provenance by default.

### ✅ Acceptance Criteria

All acceptance tests must pass before merge. Run `python acceptance_test.py` to validate.

---

## Acceptance Checklist

### 0. SPEC-001: Pack Manifest v1.0 ✅ COMPLETED
- [x] `docs/PACK_SCHEMA_V1.md` specification document created
- [x] `schemas/pack.schema.v1.json` JSON Schema for validation
- [x] `greenlang/packs/manifest.py` Pydantic model v1.0 compliant
- [x] `gl pack validate` command implemented and working
- [x] `gl pack init` creates v1.0 compliant packs
- [x] Scaffold templates updated to v1.0 specification
- [x] Comprehensive tests in `tests/packs/test_manifest_v1.py`
- [x] Migration script `scripts/migrate_pack_yaml_v1.py` functional
- [x] CI workflow includes spec validation job

**Validation Test:**
```bash
# Test pack creation and validation
gl pack init pack-basic demo-pack
gl pack validate demo-pack
# Test migration of old format
python scripts/migrate_pack_yaml_v1.py old_pack.yaml --check
```

### 1. SPEC-002: GL Pipeline v1.0 ✅ COMPLETED
- [x] `docs/GL_PIPELINE_SPEC_V1.md` specification document created
- [x] `schemas/gl_pipeline.schema.v1.json` JSON Schema for validation
- [x] `greenlang/sdk/pipeline_spec.py` Pydantic models v1.0 compliant
- [x] `greenlang/sdk/pipeline.py` Pipeline loader and validator
- [x] `greenlang/runtime/executor.py` Executor skeleton with retry/error handling
- [x] `gl validate` command supports pipeline validation
- [x] Pack template updated with v1.0 pipeline format
- [x] Comprehensive tests in `tests/pipelines/`
- [x] Sample pipelines validate successfully against v1.0 spec

**Validation Test:**
```bash
# Test pipeline validation
python test_pipeline_validation.py
# Result: 2/2 pipelines PASS
```

### 3. Pack Scaffolding ⏱️ < 60s
- [ ] `gl init pack-basic test-pack` completes in under 60 seconds
- [ ] Generated pack contains all required files:
  - [ ] `pack.yaml` with valid manifest
  - [ ] `gl.yaml` with pipeline definition
  - [ ] `CARD.md` with documentation
  - [ ] `tests/` directory with sample test
  - [ ] `.gitignore` configured
- [ ] `gl pack validate` passes on generated pack
- [ ] Pack can be published immediately

**Test Command:**
```bash
time gl init pack-basic test-scaffold-pack
cd test-scaffold-pack && gl pack validate
```

### 4. Publish → Add Workflow 📦
- [ ] `gl pack publish` executes in correct order:
  - [ ] Runs tests (`pytest`)
  - [ ] Validates policy (`check_install`)
  - [ ] Generates SBOM
  - [ ] Signs artifact (cosign)
  - [ ] Pushes to registry (oras)
- [ ] `gl pack add <ref>` successfully:
  - [ ] Pulls from registry
  - [ ] Verifies signatures
  - [ ] Validates SBOM
  - [ ] Installs to cache
- [ ] Pack is usable after installation

**Test Commands:**
```bash
gl pack publish --registry ghcr.io/test
gl pack add test/pack@0.1.0
gl pack list  # Should show installed pack
```

### 5. Deterministic Runs 🔄
- [ ] `gl run gl.yaml` produces `out/run.json`
- [ ] Consecutive runs with same inputs produce byte-identical `run.json`
- [ ] Hash values in ledger are stable
- [ ] Golden tests pass in CI
- [ ] `--deterministic` flag enforces reproducibility

**Test Commands:**
```bash
gl run gl.yaml --inputs test.json --deterministic
cp out/run.json run1.json
gl run gl.yaml --inputs test.json --deterministic
cp out/run.json run2.json
diff run1.json run2.json  # Should be identical
```

### 6. Policy Enforcement 🛡️
- [ ] **License Check**: Pack with GPL license is blocked
  - [ ] Error message clearly states license issue
  - [ ] `--explain` provides remediation steps
- [ ] **Network Egress**: Non-allowlisted URL calls are blocked
  - [ ] Error includes blocked domain
  - [ ] Policy file location shown
  - [ ] Allowlist update instructions provided
- [ ] **Vintage Check**: Old emission factors rejected
- [ ] **Residency Check**: Data residency violations caught

**Test Commands:**
```bash
# Test GPL block
echo "license: GPL-3.0" >> test-pack/pack.yaml
gl pack publish test-pack  # Should fail with license error

# Test network block
gl run pipeline-with-unknown-api.yaml  # Should fail with network error
gl policy check --explain  # Shows how to fix
```

### 7. Verify Command 🔍
- [ ] `gl verify <artifact>` displays:
  - [ ] Signer identity (keyless or key-based)
  - [ ] SBOM summary:
    - [ ] Number of dependencies
    - [ ] License summary
    - [ ] Known vulnerabilities
  - [ ] Provenance information:
    - [ ] Git commit hash
    - [ ] Build timestamp
    - [ ] Dependencies with versions
- [ ] Tampered artifacts are detected
- [ ] Missing signatures are reported

**Test Commands:**
```bash
gl verify packs/boiler-solar.tar.gz
# Output should show:
# ✓ Signature: Valid (signed by: team@greenlang.org)
# ✓ SBOM: 15 dependencies, MIT/Apache licenses
# ✓ Provenance: commit abc123, built 2024-01-01
```

### 8. Reference Packs Performance 🎯
- [ ] All 3 reference packs run successfully:
  - [ ] `boiler-solar`: ✓ Local ✓ K8s/stub
  - [ ] `hvac-measures`: ✓ Local ✓ K8s/stub
  - [ ] `cement-lca`: ✓ Local ✓ K8s/stub
- [ ] Each pack produces expected outputs:
  - [ ] PDF report generated
  - [ ] Metrics in correct format
  - [ ] Visualizations rendered
- [ ] Performance (p95 on CI):
  - [ ] Execution time ≤ 60 seconds
  - [ ] Memory usage < 1GB
  - [ ] No memory leaks

**Test Commands:**
```bash
# Run all reference packs
for pack in boiler-solar hvac-measures cement-lca; do
  echo "Testing $pack..."
  time gl run packs/$pack/gl.yaml --backend local
  test -f out/report.pdf || echo "FAIL: No PDF"
done

# Test K8s backend (or stub)
gl run packs/boiler-solar/gl.yaml --backend k8s
```

---

## Test Script Location

Run the complete acceptance test suite:
```bash
python acceptance_test.py --verbose
```

Individual test categories:
```bash
python acceptance_test.py --test scaffolding
python acceptance_test.py --test publish
python acceptance_test.py --test determinism
python acceptance_test.py --test policy
python acceptance_test.py --test verify
python acceptance_test.py --test performance
```

---

## CI Integration

### GitHub Actions Workflow
```yaml
name: Acceptance Tests

on:
  pull_request:
    branches: [main]

jobs:
  acceptance:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        test: [scaffolding, publish, determinism, policy, verify, performance]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-timeout
      
      - name: Install tools
        run: |
          # Install cosign
          curl -LO https://github.com/sigstore/cosign/releases/download/v2.0.0/cosign-linux-amd64
          chmod +x cosign-linux-amd64
          sudo mv cosign-linux-amd64 /usr/local/bin/cosign
          
          # Install oras
          curl -LO https://github.com/oras-project/oras/releases/download/v1.0.0/oras_1.0.0_linux_amd64.tar.gz
          tar -xzf oras_1.0.0_linux_amd64.tar.gz
          sudo mv oras /usr/local/bin/
          
          # Install OPA
          curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64
          chmod +x opa
          sudo mv opa /usr/local/bin/
      
      - name: Run acceptance test - ${{ matrix.test }}
        run: python acceptance_test.py --test ${{ matrix.test }}
        timeout-minutes: 5
      
      - name: Upload artifacts
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-outputs-${{ matrix.test }}
          path: |
            out/
            *.json
            test-results.xml
```

---

## Performance Benchmarks

### Expected Performance Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Pack scaffold time | < 60s | - | ⏳ |
| Pack publish time | < 30s | - | ⏳ |
| Pipeline execution (local) | < 60s | - | ⏳ |
| Pipeline execution (k8s) | < 90s | - | ⏳ |
| Memory usage (peak) | < 1GB | - | ⏳ |
| SBOM generation | < 10s | - | ⏳ |
| Signature verification | < 5s | - | ⏳ |

---

## Manual Testing Guide

### Developer Workflow Test

1. **Fresh Install**
   ```bash
   pip install greenlang
   gl --version
   ```

2. **Create Pack**
   ```bash
   gl init pack-basic my-climate-pack
   cd my-climate-pack
   ```

3. **Develop & Test**
   ```bash
   # Edit pipeline
   vi gl.yaml
   
   # Validate
   gl pack validate
   
   # Run locally
   gl run gl.yaml --inputs sample.json
   ```

4. **Publish**
   ```bash
   gl pack publish --registry ghcr.io/myorg
   ```

5. **Share & Use**
   ```bash
   # On another machine
   gl pack add myorg/my-climate-pack@0.1.0
   gl run my-climate-pack --inputs data.json
   ```

---

## Troubleshooting

### Common Issues

1. **Scaffolding Timeout**
   - Check network connectivity
   - Ensure Python packages are cached
   - Verify disk I/O performance

2. **Policy Failures**
   - Run with `--explain` flag
   - Check `policies/` directory
   - Review allowlists in policy files

3. **Signature Verification**
   - Ensure cosign is installed
   - Check OIDC token availability
   - Verify network access to Sigstore

4. **Performance Issues**
   - Profile with `--profile` flag
   - Check memory limits
   - Review pipeline complexity

---

## Sign-off

### Required Approvals

- [ ] Engineering Lead - Code quality and architecture
- [ ] Security Team - Policy and provenance implementation
- [ ] DevOps - CI/CD and deployment readiness
- [ ] Product - User experience and documentation

### Final Checks

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Migration guide ready

---

## Notes

- Run acceptance tests in clean environment
- Test on Linux, macOS, and Windows
- Verify with Python 3.8, 3.10, and 3.12
- Check with both local and CI environments