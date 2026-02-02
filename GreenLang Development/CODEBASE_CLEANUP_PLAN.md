# GreenLang Codebase Cleanup Plan
## Duplicate Removal & Consolidation Guide

**Generated:** February 2, 2026
**Estimated Impact:** 40-50% codebase size reduction

---

## Executive Summary

Comprehensive analysis identified extensive duplication across the GreenLang codebase. This document provides a prioritized cleanup plan to eliminate redundancy while maintaining functionality.

---

## 1. Critical Priority (Immediate Action)

### 1.1 Agent Definition Consolidation

**Problem:** Agents GL-001 through GL-017 exist in TWO locations with different naming conventions.

**Location A:** `applications/GL Agents/GL-001_Thermalcommand/` (PascalCase)
**Location B:** `greenlang/agents/process_heat/` (snake_case)

**Action:**
- [ ] Choose canonical location: `greenlang/agents/` (recommended)
- [ ] Standardize naming convention to snake_case
- [ ] Migrate all agent code to single location
- [ ] Update all imports and references
- [ ] Remove duplicate location
- [ ] Update CI/CD pipelines

**Agents Affected:**
| Agent ID | Current Name | Canonical Name |
|----------|--------------|----------------|
| GL-001 | ThermalCommand | thermal_command |
| GL-002 | Flameguard | boiler_efficiency |
| GL-003 | UnifiedSteam | steam_systems |
| GL-004 | Burnmaster | burner_optimization |
| GL-005 | Combusense | combustion_diagnostics |
| GL-006 | HEATRECLAIM | waste_heat_recovery |
| GL-007 | FurnacePulse | furnace_optimization |
| GL-008 | Trapcatcher | steam_trap_monitor |
| GL-009 | ThermalIQ | thermal_fluid |
| GL-010 | EmissionGuardian | emission_reporting |
| GL-011 | FuelCraft | fuel_optimization |
| GL-012 | SteamQual | steam_quality |
| GL-013 | PredictiveMaintenance | rul_prediction |
| GL-014 | Exchangerpro | heat_exchanger |
| GL-015 | Insulscan | insulation_analysis |
| GL-016 | Waterguard | water_treatment |
| GL-017 | Condensync | condenser_optimization |

---

### 1.2 Requirements File Consolidation

**Problem:** 70+ requirements files with version conflicts and redundancy.

**Action:**
- [ ] Create master `requirements.txt` at repository root
- [ ] Use `pyproject.toml` with extras_require for optional dependencies
- [ ] Remove all agent-specific requirements.txt files
- [ ] Create `requirements-dev.txt` for development dependencies
- [ ] Create `requirements-test.txt` for testing dependencies

**Files to Remove:**
```
applications/GL Agents/GL-*/requirements.txt (17 files)
docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-*/requirements.txt (12 files)
applications/GL-*-APP/*/requirements*.txt (multiple variants)
```

**Consolidated Structure:**
```
requirements.txt          # Production dependencies
requirements-dev.txt      # Development tools
requirements-test.txt     # Testing frameworks
requirements-docs.txt     # Documentation tools
pyproject.toml           # Package metadata + extras
```

---

### 1.3 Pre-commit Configuration

**Problem:** 14 identical `.pre-commit-config.yaml` files.

**Action:**
- [ ] Keep single `.pre-commit-config.yaml` at repository root
- [ ] Remove all nested pre-commit configs
- [ ] Ensure root config covers all file patterns

**Files to Remove:**
```
docs/planning/greenlang-2030-vision/agent_foundation/.pre-commit-config.yaml
docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-001/.pre-commit-config.yaml
docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-002/.pre-commit-config.yaml
... (11 more)
```

---

### 1.4 Docker File Consolidation

**Problem:** 27+ similar Dockerfiles with minor variations.

**Action:**
- [ ] Create base Dockerfile templates in `deployment/docker/templates/`
- [ ] Use multi-stage builds with shared base images
- [ ] Parameterize agent-specific Dockerfiles using ARG/ENV
- [ ] Remove duplicate Dockerfiles

**Template Structure:**
```dockerfile
# deployment/docker/templates/agent.Dockerfile
ARG AGENT_NAME
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim as base
# Common setup...

FROM base as ${AGENT_NAME}
# Agent-specific layers
```

---

## 2. High Priority (This Week)

### 2.1 Schema File Consolidation

**Problem:** 27+ schema files with 90% similarity.

**Action:**
- [ ] Create `greenlang/schemas/base.py` with common Pydantic models
- [ ] Create `greenlang/schemas/mixins.py` for reusable field groups
- [ ] Refactor agent schemas to inherit from base classes
- [ ] Remove duplicate schema definitions

**Common Models to Extract:**
- `BaseConfig` - Common configuration fields
- `AuditMixin` - Audit trail fields (created_at, updated_at, created_by)
- `ProvenanceMixin` - Data lineage fields
- `EmissionsMixin` - Common emissions fields (scope1, scope2, scope3)
- `UncertaintyMixin` - Uncertainty quantification fields

---

### 2.2 Configuration File Consolidation

**Problem:** 20+ config.py files with identical enums and classes.

**Action:**
- [ ] Create `greenlang/config/enums.py` with shared enums
- [ ] Create `greenlang/config/base.py` with base configuration classes
- [ ] Refactor agent configs to import from shared module

**Common Enums to Extract:**
```python
# greenlang/config/enums.py
class SafetyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProtocolType(Enum):
    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"
    WEBSOCKET = "websocket"

class DeploymentMode(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class CoordinationStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
```

---

### 2.3 Test Infrastructure Consolidation

**Problem:** 70+ conftest.py files with duplicate fixtures.

**Action:**
- [ ] Create master `tests/conftest.py` with shared fixtures
- [ ] Create `tests/fixtures/` directory for complex fixtures
- [ ] Remove agent-specific conftest.py duplicates
- [ ] Use pytest plugins for common functionality

**Shared Fixtures to Create:**
```python
# tests/conftest.py
@pytest.fixture
def db_session(): ...

@pytest.fixture
def mock_claude_client(): ...

@pytest.fixture
def sample_emissions_data(): ...

@pytest.fixture
def mock_emission_factors(): ...

@pytest.fixture
def authenticated_client(): ...
```

---

### 2.4 Pytest Configuration

**Problem:** 43+ pytest.ini files.

**Action:**
- [ ] Keep single `pytest.ini` at repository root
- [ ] Use `pyproject.toml` [tool.pytest.ini_options] as alternative
- [ ] Remove all nested pytest.ini files

---

### 2.5 Docker Compose Consolidation

**Problem:** 27+ docker-compose variants.

**Action:**
- [ ] Create base `docker-compose.yml` with core services
- [ ] Use docker-compose override files for environments
- [ ] Remove redundant compose files

**Target Structure:**
```
deployment/
├── docker-compose.yml           # Base services
├── docker-compose.dev.yml       # Development overrides
├── docker-compose.staging.yml   # Staging overrides
├── docker-compose.prod.yml      # Production overrides
└── docker-compose.test.yml      # Testing overrides
```

---

## 3. Medium Priority (This Month)

### 3.1 Calculator Module Consolidation

**Problem:** 90+ calculator files with duplicated logic.

**Duplicate Names Found:**
- `emissions_calculator.py` - GL-002, GL-004, GL-005
- `heat_balance_calculator.py` - GL-002, GL-003, GL-009
- `efficiency_calculator.py` - GL-002, GL-007

**Action:**
- [ ] Create `greenlang/calculators/` shared module
- [ ] Extract common calculation functions
- [ ] Refactor agents to use shared calculators
- [ ] Keep agent-specific calculations in agent directories

---

### 3.2 Documentation Consolidation

**Problem:** 90+ README files with overlapping content.

**Action:**
- [ ] Archive `docs/planning/greenlang-2030-vision/` to separate branch
- [ ] Create single comprehensive README at root
- [ ] Use `docs/` for production documentation only
- [ ] Link to archived planning docs from main documentation

---

### 3.3 Demo/Example Consolidation

**Problem:** 50+ demo files following identical patterns.

**Action:**
- [ ] Create demo template framework
- [ ] Generate demos from templates with domain-specific data
- [ ] Reduce to essential example set (10-15 demos)

---

### 3.4 SBOM Consolidation

**Problem:** 16+ duplicate SBOM sets.

**Action:**
- [ ] Keep single SBOM generation in CI/CD
- [ ] Store SBOMs in versioned artifact repository
- [ ] Remove duplicate SBOM files from repository

---

## 4. Cleanup Execution Checklist

### Phase 1: Preparation (Day 1)
- [ ] Create backup branch: `git checkout -b backup/pre-cleanup`
- [ ] Document all current import paths
- [ ] Identify all external dependencies on current structure
- [ ] Create cleanup branch: `git checkout -b feature/codebase-cleanup`

### Phase 2: Critical Cleanup (Days 2-3)
- [ ] Consolidate agent definitions
- [ ] Consolidate requirements files
- [ ] Remove duplicate pre-commit configs
- [ ] Create Docker templates

### Phase 3: High Priority Cleanup (Days 4-5)
- [ ] Consolidate schemas
- [ ] Consolidate configs
- [ ] Consolidate test infrastructure
- [ ] Consolidate pytest configs
- [ ] Consolidate docker-compose files

### Phase 4: Validation (Day 6)
- [ ] Run full test suite
- [ ] Verify all imports work
- [ ] Check CI/CD pipelines
- [ ] Validate Docker builds
- [ ] Review documentation links

### Phase 5: Medium Priority Cleanup (Week 2)
- [ ] Consolidate calculators
- [ ] Consolidate documentation
- [ ] Consolidate demos
- [ ] Consolidate SBOMs

### Phase 6: Final Verification (Week 2)
- [ ] Full regression testing
- [ ] Performance benchmarks
- [ ] Security scan
- [ ] Create PR for review
- [ ] Merge to main branch

---

## 5. Expected Outcomes

### Size Reduction
| Category | Before | After | Reduction |
|----------|--------|-------|-----------|
| Requirements files | 70+ | 5 | 93% |
| Schema files | 27+ | 10 | 63% |
| Config files | 20+ | 5 | 75% |
| Docker files | 27+ | 10 | 63% |
| Pre-commit configs | 14 | 1 | 93% |
| Test conftest | 70+ | 10 | 86% |
| Pytest.ini | 43+ | 1 | 98% |

### Overall Impact
- **Estimated codebase reduction:** 40-50%
- **Improved maintainability:** Single source of truth
- **Faster CI/CD:** Less files to process
- **Easier onboarding:** Cleaner structure

---

## 6. Risk Mitigation

### Potential Issues
1. **Breaking imports** - Mitigate with comprehensive import path documentation
2. **CI/CD failures** - Run in feature branch first, validate all pipelines
3. **Lost functionality** - Full test coverage before and after cleanup
4. **Team disruption** - Communicate changes, update documentation

### Rollback Plan
- Keep backup branch until cleanup validated in production
- Document all structural changes for easy reversal
- Maintain import aliases during transition period

---

*Document maintained by GreenLang Development Team*
*Last updated: February 2, 2026*
