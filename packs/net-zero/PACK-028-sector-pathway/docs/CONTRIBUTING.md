# PACK-028 Sector Pathway Pack -- Contributing Guide

**Pack ID:** PACK-028-sector-pathway
**Version:** 1.0.0
**Last Updated:** 2026-03-19

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment Setup](#development-environment-setup)
3. [Project Structure](#project-structure)
4. [Coding Standards](#coding-standards)
5. [Testing Requirements](#testing-requirements)
6. [Sector Data Contributions](#sector-data-contributions)
7. [Pull Request Process](#pull-request-process)
8. [Code Review Guidelines](#code-review-guidelines)
9. [Release Process](#release-process)
10. [Issue Reporting](#issue-reporting)

---

## Getting Started

### Prerequisites

Before contributing to PACK-028, ensure you have:

1. Python 3.11 or higher installed
2. PostgreSQL 16 with TimescaleDB extension
3. Redis 7+
4. Git with commit signing configured
5. Access to the GreenLang development environment
6. Familiarity with SBTi SDA methodology and IEA NZE 2050 scenarios

### Contributor Agreement

All contributors must sign the GreenLang Contributor License Agreement (CLA) before submitting code. Contact `legal@greenlang.io` for the CLA.

---

## Development Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/greenlang/packs.git
cd packs/net-zero/PACK-028-sector-pathway
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Step 4: Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your local settings
# Required: DB_HOST, DB_PORT, DB_NAME, REDIS_HOST, REDIS_PORT
```

### Step 5: Apply Migrations

```bash
# Apply platform migrations first (V001-V128)
# Then apply PACK-028 migrations
for i in $(seq -w 1 6); do
  psql -h localhost -U greenlang -d greenlang -f migrations/V181-PACK028-${i}.sql
done
```

### Step 6: Load Reference Data

```bash
python scripts/load_reference_data.py
```

### Step 7: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific engine tests
pytest tests/test_pathway_generator_engine.py -v
```

---

## Project Structure

```
PACK-028-sector-pathway/
  __init__.py
  pack.yaml                          # Pack manifest
  requirements.txt                    # Production dependencies
  requirements-dev.txt                # Development dependencies
  .env.example                        # Environment variable template
  README.md                           # Pack overview

  engines/                            # 8 calculation engines
    __init__.py
    sector_classification_engine.py
    intensity_calculator_engine.py
    pathway_generator_engine.py
    convergence_analyzer_engine.py
    technology_roadmap_engine.py
    abatement_waterfall_engine.py
    sector_benchmark_engine.py
    scenario_comparison_engine.py

  workflows/                          # 6 orchestrated workflows
    __init__.py
    sector_pathway_design_workflow.py
    pathway_validation_workflow.py
    technology_planning_workflow.py
    progress_monitoring_workflow.py
    multi_scenario_analysis_workflow.py
    full_sector_assessment_workflow.py

  templates/                          # 8 report templates
    __init__.py
    sector_pathway_report.py
    intensity_convergence_report.py
    technology_roadmap_report.py
    abatement_waterfall_report.py
    sector_benchmark_report.py
    scenario_comparison_report.py
    sbti_validation_report.py
    sector_strategy_report.py

  integrations/                       # 10 integration bridges
    __init__.py
    pack_orchestrator.py
    sbti_sda_bridge.py
    iea_nze_bridge.py
    ipcc_ar6_bridge.py
    pack021_bridge.py
    mrv_bridge.py
    decarb_bridge.py
    data_bridge.py
    health_check.py
    setup_wizard.py

  config/                             # Configuration
    __init__.py
    pack_config.py
    constants.py
    presets/
      __init__.py
      heavy_industry.yaml
      power_utilities.yaml
      transport.yaml
      buildings.yaml
      light_industry.yaml
      agriculture.yaml

  data/                               # Reference data
    __init__.py
    sbti_sda/                         # SBTi SDA convergence data
    iea_nze/                          # IEA NZE 2050 pathway data
    ipcc_ar6/                         # IPCC AR6 emission factors
    sector_benchmarks/                # Sector benchmark data

  migrations/                         # Database migrations
    V181-PACK028-001.sql
    V181-PACK028-001.down.sql
    V181-PACK028-002.sql
    ...

  tests/                              # Test suite
    __init__.py
    conftest.py
    test_sector_classification_engine.py
    test_intensity_calculator_engine.py
    test_pathway_generator_engine.py
    test_convergence_analyzer_engine.py
    test_technology_roadmap_engine.py
    test_abatement_waterfall_engine.py
    test_sector_benchmark_engine.py
    test_scenario_comparison_engine.py
    test_workflows.py
    test_templates.py
    test_integrations.py
    test_presets.py
    test_config.py
    test_manifest.py
    test_e2e.py
    test_accuracy.py
    test_orchestrator.py

  docs/                               # Documentation
    API_REFERENCE.md
    USER_GUIDE.md
    INTEGRATION_GUIDE.md
    VALIDATION_REPORT.md
    DEPLOYMENT_CHECKLIST.md
    CHANGELOG.md
    CONTRIBUTING.md
    SECTOR_GUIDES/
      SECTOR_GUIDE_POWER.md
      SECTOR_GUIDE_STEEL.md
      ...
```

---

## Coding Standards

### Python Style

- Follow PEP 8 with 120-character line length
- Use type hints for all function signatures
- Use Pydantic v2 models for all data structures
- Use `async/await` for all I/O operations
- Use f-strings for string formatting

### Example Engine Method

```python
from pydantic import BaseModel, Field
from typing import Optional
from decimal import Decimal

class PathwayInput(BaseModel):
    """Input parameters for pathway generation."""
    sector: str = Field(..., description="Sector identifier")
    base_year: int = Field(..., ge=2015, le=2030, description="Base year")
    base_year_intensity: float = Field(..., gt=0, description="Base year intensity")
    target_year_near: int = Field(..., ge=2025, le=2035, description="Near-term target year")
    target_year_long: int = Field(2050, description="Long-term target year")
    scenario: str = Field("nze_15c", description="Climate scenario")

class PathwayResult(BaseModel):
    """Result of pathway generation."""
    pathway_id: str
    pathway_name: str
    target_2030: float
    target_2050: float
    annual_pathway: list[PathwayPoint]
    provenance: ProvenanceHash

async def generate(self, input_data: PathwayInput) -> PathwayResult:
    """
    Generate a sector-specific decarbonization pathway.

    Args:
        input_data: Validated pathway generation parameters.

    Returns:
        PathwayResult with year-by-year intensity targets.

    Raises:
        SectorNotFoundError: If sector is not supported.
        ConvergenceInfeasibleError: If convergence is mathematically infeasible.
    """
    # Implementation
    ...
```

### Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Files | `snake_case.py` | `pathway_generator_engine.py` |
| Classes | `PascalCase` | `PathwayGeneratorEngine` |
| Functions | `snake_case` | `generate_pathway()` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_SCENARIOS` |
| Pydantic models | `PascalCase` | `PathwayInput` |
| Test files | `test_*.py` | `test_pathway_generator_engine.py` |
| Test functions | `test_*` | `test_generate_pathway_steel_nze()` |

### Documentation Standards

- All public functions must have docstrings (Google style)
- All Pydantic models must have field descriptions
- All engines must have a module-level docstring explaining purpose
- All complex calculations must have inline comments explaining the formula

### Calculation Standards

- **Zero-hallucination**: No LLM or probabilistic model in any calculation path
- **Deterministic**: Same inputs must always produce the same outputs
- **SHA-256 provenance**: Every calculation output must include a provenance hash
- **Unit consistency**: All intensity metrics must use the units defined in the sector taxonomy
- **Reference data citation**: All reference data (SBTi, IEA, IPCC) must include version and source

---

## Testing Requirements

### Minimum Requirements

| Metric | Requirement |
|--------|------------|
| Test pass rate | 100% |
| Code coverage | 90%+ |
| SBTi accuracy | 100% match with SBTi tool |
| IEA alignment | +/-5% from IEA data |
| Convergence accuracy | +/-2% from manual calculation |

### Test Categories

Every new feature must include:

1. **Unit tests**: Test individual methods in isolation
2. **Integration tests**: Test interactions with bridges and other engines
3. **Accuracy tests**: Cross-validate against known reference values
4. **Edge case tests**: Handle boundary conditions (zero values, extreme intensities)
5. **Regression tests**: Prevent previously fixed bugs from recurring

### Writing Tests

```python
import pytest
from engines.pathway_generator_engine import PathwayGeneratorEngine, PathwayInput

class TestPathwayGeneratorEngine:
    """Tests for PathwayGeneratorEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance for testing."""
        return PathwayGeneratorEngine()

    def test_generate_steel_nze_15c(self, engine):
        """Verify steel sector NZE 1.5C pathway matches SBTi SDA tool."""
        result = engine.generate(PathwayInput(
            sector="steel",
            base_year=2023,
            base_year_intensity=1.85,
            target_year_near=2030,
            target_year_long=2050,
            scenario="nze_15c",
        ))

        assert result.target_2030 == pytest.approx(1.25, abs=0.01)
        assert result.target_2050 == pytest.approx(0.10, abs=0.01)
        assert len(result.annual_pathway) == 28  # 2023-2050
        assert result.provenance.algorithm == "sha256"

    def test_generate_invalid_sector_raises(self, engine):
        """Verify invalid sector raises SectorNotFoundError."""
        with pytest.raises(SectorNotFoundError):
            engine.generate(PathwayInput(
                sector="invalid_sector",
                base_year=2023,
                base_year_intensity=1.0,
                target_year_near=2030,
            ))

    def test_generate_monotonic_decrease(self, engine):
        """Verify pathway intensity decreases monotonically."""
        result = engine.generate(PathwayInput(
            sector="cement",
            base_year=2023,
            base_year_intensity=0.62,
            target_year_near=2030,
            target_year_long=2050,
            scenario="nze_15c",
        ))

        for i in range(1, len(result.annual_pathway)):
            assert result.annual_pathway[i].intensity <= result.annual_pathway[i-1].intensity

    def test_generate_deterministic(self, engine):
        """Verify same inputs produce same outputs (deterministic)."""
        input_data = PathwayInput(
            sector="steel",
            base_year=2023,
            base_year_intensity=1.85,
            target_year_near=2030,
            scenario="nze_15c",
        )

        result1 = engine.generate(input_data)
        result2 = engine.generate(input_data)

        assert result1.provenance.hash == result2.provenance.hash
        assert result1.target_2030 == result2.target_2030
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=. --cov-report=html --cov-fail-under=90

# Run specific test file
pytest tests/test_pathway_generator_engine.py -v

# Run specific test
pytest tests/test_pathway_generator_engine.py::TestPathwayGeneratorEngine::test_generate_steel_nze_15c -v

# Run accuracy tests only
pytest tests/test_accuracy.py -v

# Run end-to-end tests only
pytest tests/test_e2e.py -v

# Run with parallel execution
pytest tests/ -v -n auto
```

---

## Sector Data Contributions

### Adding a New Sector

To add a new sector to PACK-028:

1. **Add sector definition** to `config/constants.py`
2. **Add NACE/GICS/ISIC mappings** to sector classification engine
3. **Add intensity metrics** to intensity calculator engine
4. **Add SBTi/IEA pathway data** to reference data directory
5. **Add sector-specific levers** to abatement waterfall engine
6. **Add sector technologies** to technology roadmap engine
7. **Add sector benchmark data** to benchmark engine
8. **Write 50+ tests** covering all engines for the new sector
9. **Write sector guide** documentation
10. **Update pack.yaml** manifest

### Updating Reference Data

When SBTi or IEA publish updated pathway data:

1. Add new data files to the appropriate directory
2. Update version string in the bridge configuration
3. Regenerate SHA-256 checksums
4. Run accuracy tests against the new data
5. Update CHANGELOG.md with data version change
6. Create a minor version release

---

## Pull Request Process

### Branch Naming

| Type | Convention | Example |
|------|-----------|---------|
| Feature | `feat/description` | `feat/add-mining-sector` |
| Bug fix | `fix/description` | `fix/cement-convergence-calculation` |
| Data update | `data/description` | `data/sbti-sda-v3.1-update` |
| Documentation | `docs/description` | `docs/shipping-sector-guide` |
| Performance | `perf/description` | `perf/pathway-cache-optimization` |

### PR Requirements

Before submitting a PR, ensure:

1. [ ] All tests pass (`pytest tests/ -v`)
2. [ ] Code coverage >= 90% (`pytest --cov-fail-under=90`)
3. [ ] No linting errors (`ruff check .`)
4. [ ] Type checking passes (`mypy .`)
5. [ ] New features have tests
6. [ ] New engines/workflows have documentation
7. [ ] CHANGELOG.md updated
8. [ ] Accuracy tests pass (if calculation changes)
9. [ ] Performance benchmarks pass (if engine changes)
10. [ ] Security review completed (if auth/data changes)

### PR Template

```markdown
## Summary
Brief description of changes.

## Type
- [ ] Feature
- [ ] Bug fix
- [ ] Data update
- [ ] Documentation
- [ ] Performance

## Changes
- List of specific changes

## Testing
- [ ] All tests pass
- [ ] Coverage >= 90%
- [ ] Accuracy validated
- [ ] Performance validated

## Sectors Affected
- [ ] Power  [ ] Steel  [ ] Cement  [ ] Aluminum
- [ ] Chemicals  [ ] Pulp & Paper  [ ] Aviation  [ ] Shipping
- [ ] Road Transport  [ ] Rail  [ ] Buildings  [ ] Agriculture
- [ ] Food & Beverage  [ ] Oil & Gas  [ ] Cross-Sector
```

---

## Code Review Guidelines

### Review Checklist

1. **Correctness**: Do calculations match SBTi SDA and IEA NZE reference data?
2. **Determinism**: Are calculations fully deterministic (no randomness)?
3. **Provenance**: Is SHA-256 provenance hashing applied to all outputs?
4. **Tests**: Are new features covered by unit, integration, and accuracy tests?
5. **Performance**: Do changes meet engine latency targets?
6. **Security**: Are RBAC permissions correctly enforced?
7. **Documentation**: Are public APIs documented with docstrings?
8. **Data quality**: Are intensity metrics in correct units?
9. **Edge cases**: Are boundary conditions handled?
10. **Backwards compatibility**: Do changes break existing APIs?

---

## Release Process

### Version Numbering

- **Major** (x.0.0): Breaking API changes, new sector classification taxonomy
- **Minor** (1.x.0): New sectors, new engines, reference data updates
- **Patch** (1.0.x): Bug fixes, performance improvements, documentation updates

### Release Steps

1. Verify all tests pass on release branch
2. Update version in `pack.yaml`, `__init__.py`, and documentation
3. Update CHANGELOG.md with release notes
4. Create Git tag: `git tag -s v1.0.0 -m "PACK-028 v1.0.0"`
5. Run full validation suite (847 tests + accuracy tests)
6. Generate validation report
7. Create GitHub release with release notes
8. Deploy to staging environment
9. Run smoke tests on staging
10. Deploy to production
11. Verify health check passes

---

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Sector**: Which sector is affected?
2. **Engine/Workflow**: Which component?
3. **Input data**: Exact inputs that reproduce the issue
4. **Expected output**: What you expected
5. **Actual output**: What you got
6. **SBTi/IEA reference**: If accuracy-related, cite the reference value
7. **Environment**: Python version, platform, PACK-028 version

### Feature Requests

When requesting features, include:

1. **Use case**: What problem does this solve?
2. **Sector impact**: Which sectors benefit?
3. **Reference data**: Are SBTi/IEA data sources available?
4. **Priority**: How critical is this for your deployment?

---

## Contact

- **Development Team**: net-zero-team@greenlang.io
- **Slack Channel**: #pack-028-dev
- **Issue Tracker**: github.com/greenlang/packs/issues

---

**End of Contributing Guide**
