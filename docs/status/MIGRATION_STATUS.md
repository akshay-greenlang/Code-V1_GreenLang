# GreenLang Migration Status

## Migration to Unified `gl` CLI and Pack Architecture

### ✅ Completed Components

#### 1. CLI Infrastructure (`gl`)
- ✅ Typer-based CLI with subcommands
- ✅ Commands: init, run, pack, verify, policy, doctor
- ✅ Shell completion support
- ✅ Backward compatibility alias (greenlang → gl)

#### 2. Pack System
- ✅ Pack manifest schema (Pydantic)
- ✅ Pack loader and validator
- ✅ Pack registry system
- ✅ Pack installer with OCI support
- ✅ Commands: pack create/validate/publish/add/list/info

#### 3. Policy System (OPA)
- ✅ Policy enforcer with check_install/check_run
- ✅ Rego policy bundles (install.rego, run.rego)
- ✅ Network egress control
- ✅ Residency and vintage checks
- ✅ Commands: policy check/run --explain

#### 4. Provenance System
- ✅ SBOM generation (CycloneDX with fallback)
- ✅ Cosign signing integration (keyless OIDC + key-based)
- ✅ Run ledger with deterministic hashing
- ✅ Command: verify

#### 5. Runtime Profiles
- ✅ Local executor with deterministic mode
- ✅ Kubernetes backend (Job manifest generation)
- ✅ Golden test framework
- ✅ Output normalization for reproducibility

#### 6. Model/Dataset Cards
- ✅ HuggingFace-style card templates
- ✅ Card generator for packs/datasets/models/pipelines
- ✅ Card validator with quality scoring
- ✅ Environmental impact sections
- ✅ Auto-generation in `gl init`

#### 7. Reference Pack: boiler-solar
- ✅ pack.yaml manifest
- ✅ gl.yaml pipeline definition
- ✅ CARD.md documentation
- ✅ Sample agents (solar_estimator, boiler_analyzer)
- ✅ Sample dataset (building data)
- ✅ Test structure

### 🔄 In Progress

#### Reference Packs
- ⏳ hvac-measures pack
- ⏳ cement-lca pack

### 📋 Remaining Tasks

#### 1. Complete Reference Packs
- [ ] hvac-measures: Create full pack structure
- [ ] cement-lca: Create full pack structure
- [ ] Add remaining agents (system_integrator, energy_optimizer)
- [ ] Complete golden tests for all packs

#### 2. CI/CD Integration
- [ ] GitHub Actions workflow for pack validation
- [ ] SBOM generation in CI
- [ ] Cosign signing on releases
- [ ] Pack publishing to GHCR

#### 3. Documentation
- [ ] Update README with `gl` commands
- [ ] 5-minute quickstart guide
- [ ] Command reference pages
- [ ] Migration guide from old CLI

#### 4. Testing
- [ ] Integration tests for full pipeline
- [ ] Policy enforcement tests
- [ ] Determinism validation tests
- [ ] Cross-platform smoke tests

## File Structure

```
Code V1_GreenLang/
├── core/greenlang/
│   ├── cli/               ✅ Unified CLI
│   ├── packs/             ✅ Pack system
│   ├── policy/            ✅ OPA integration
│   ├── provenance/        ✅ SBOM + signing
│   ├── runtime/           ✅ Executors
│   ├── cards/             ✅ Model/dataset cards
│   └── utils/             ✅ Network wrapper
├── packs/
│   ├── boiler-solar/      ✅ Reference pack
│   ├── hvac-measures/     ⏳ In progress
│   └── cement-lca/        ⏳ In progress
├── policies/              ✅ Rego bundles
└── scripts/               ✅ Setup scripts
```

## Key Achievements

1. **Unified CLI**: Single `gl` command with discoverable subcommands
2. **Pack-based Architecture**: Domain logic moved to packs, core is generic
3. **Policy Enforcement**: OPA-based gates at install and runtime
4. **Provenance by Default**: SBOM, signing, and deterministic runs
5. **Environmental Focus**: Carbon footprint in all cards and reports
6. **Developer Experience**: 5-minute success path from install to results

## Next Steps

1. Complete the two remaining reference packs
2. Add comprehensive tests for the migration
3. Set up CI/CD pipeline
4. Update all documentation
5. Create demo videos/GIFs

## Usage Examples

```bash
# Initialize a new pack
gl init pack-basic my-pack

# Validate pack structure
gl pack validate

# Run pipeline
gl run boiler-solar --inputs data.json

# Publish pack
gl pack publish --registry ghcr.io/greenlang

# Add pack from registry
gl pack add greenlang/boiler-solar@0.1.0

# Verify artifact
gl verify artifact.tar.gz

# Check policy
gl policy check --explain

# Run with deterministic mode
gl run pipeline.yaml --deterministic --golden
```

## Success Metrics

- ✅ CLI fully migrated to `gl`
- ✅ Core infrastructure is pack-agnostic
- ✅ Policy enforcement working
- ✅ Provenance system complete
- ✅ Cards system integrated
- ⏳ Reference packs demonstrating best practices
- ⏳ CI/CD automation ready
- ⏳ Documentation updated

## Notes

- All `greenlang` CLI commands have been replaced with `gl`
- Pack system supports OCI registry distribution
- Policy defaults are strict with learning mode option
- Deterministic runs enable reproducible science
- Environmental impact is tracked throughout