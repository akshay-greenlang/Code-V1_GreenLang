# GreenLang Directory Structure

**Last Updated:** January 25, 2026
**Status:** Post-Reorganization (v0.3.1)

This document describes the purpose of each top-level directory in the GreenLang codebase, naming conventions, and guidelines for adding new code.

---

## Overview

The GreenLang codebase has been reorganized to improve maintainability, discoverability, and developer experience. The root directory contains **16 top-level directories**, down from 748+ items before reorganization.

---

## Top-Level Directory Structure

```
greenlang/
├── .claude/                 # Claude AI configuration
├── .git/                    # Git version control
├── .github/                 # GitHub Actions workflows and templates
├── .greenlang/             # GreenLang platform configuration
├── .pytest_cache/          # Pytest cache (gitignored)
├── .ralphy/                # Ralphy agent configuration
├── 2026_PRD_MVP/           # 2026 MVP PRD (DO NOT MODIFY)
├── applications/           # Production applications (GL-VCCI, GL-CBAM, GL-CSRD)
├── cbam-pack-mvp/          # CBAM Pack MVP (DO NOT MODIFY)
├── config/                 # Global configuration files
├── data/                   # Runtime data and temporary files
├── datasets/               # Static reference datasets (emission factors, etc.)
├── deployment/             # Deployment configurations (Docker, K8s, Helm, Terraform)
├── docs/                   # Documentation hub
├── examples/               # Code examples and tutorials
├── greenlang/              # Core platform source code
├── logs/                   # Application logs (gitignored)
├── reports/                # Generated reports and analysis
├── scripts/                # Utility scripts (dev, maintenance, testing)
├── test-reports/           # Test execution reports (gitignored)
├── tests/                  # Test suite (unit, integration, e2e)
└── tools/                  # Development and operational tools
```

---

## Directory Descriptions

### `.claude/`
**Purpose:** Configuration for Claude AI code assistant
**Contains:** Settings, preferences, and agent configurations
**Modify:** Only if adjusting Claude integration

### `.github/`
**Purpose:** GitHub-specific configurations
**Contains:**
- Workflows (`.github/workflows/`) - CI/CD pipelines
- Issue templates
- Pull request templates
- Dependabot configuration

**When to modify:** Adding/updating CI/CD workflows, issue templates

### `.greenlang/`
**Purpose:** GreenLang platform-level configuration
**Contains:**
- ADRs (Architecture Decision Records)
- Deployment configurations
- Platform tools and utilities

**When to modify:** Platform-level configuration changes

### `.ralphy/`
**Purpose:** Ralphy agent progress tracking and configuration
**Contains:** Progress files, worktrees, sandboxes
**Modify:** Generally auto-managed, avoid manual changes

### `2026_PRD_MVP/` ⚠️
**Purpose:** 2026 MVP Product Requirements
**Status:** **DO NOT MODIFY** (protected boundary)
**Contains:** MVP specifications and planning documents

### `applications/`
**Purpose:** Production-ready GreenLang applications
**Contains:**
- `GL-VCCI-Carbon-APP/` - Scope 3 Value Chain Intelligence Platform
- `GL-CBAM-APP/` - Carbon Border Adjustment Mechanism
- `GL-CSRD-APP/` - Corporate Sustainability Reporting Directive
- `GL-EUDR-APP/` - EU Deforestation Regulation (in development)
- `GL-Taxonomy-APP/` - EU Taxonomy Alignment (in development)
- `apps/` - Legacy and experimental applications

**Structure per application:**
```
applications/GL-{NAME}-APP/
├── docs/           # Application-specific documentation
├── src/            # Source code
├── tests/          # Application tests
├── deployment/     # Application deployment configs
└── README.md       # Application overview
```

**When to add new code:**
- New regulatory compliance applications go in `applications/GL-{NAME}-APP/`
- Follow existing application structure patterns

### `cbam-pack-mvp/` ⚠️
**Purpose:** CBAM Pack MVP reference implementation
**Status:** **DO NOT MODIFY** (protected boundary)
**Contains:** CBAM pack with web interface

### `config/`
**Purpose:** Global configuration files
**Contains:**
- Application settings
- Environment configurations
- Feature flags
- Default configurations

**Import pattern:**
```python
from greenlang.config import settings
from greenlang.config.defaults import DEFAULT_CONFIG
```

### `data/`
**Purpose:** Runtime data and temporary files
**Contains:**
- Temporary processing data
- Cache files
- Generated artifacts
- Runtime databases

**Note:** Most contents are gitignored. Not for static reference data (use `datasets/` instead).

### `datasets/`
**Purpose:** Static reference datasets
**Contains:**
- Emission factor databases
- Knowledge bases (GHG Protocol, regulations)
- Example/sample data
- Reference datasets

**Structure:**
```
datasets/
├── emission_factors/    # Emission factor databases
├── knowledge_base/      # Regulatory and methodology docs
└── examples/            # Sample datasets for testing
```

**When to add new code:**
- Static emission factors → `datasets/emission_factors/`
- Reference documentation → `datasets/knowledge_base/`
- Sample data → `datasets/examples/`

### `deployment/`
**Purpose:** Deployment configurations and infrastructure-as-code
**Contains:**
- `docker/` - Docker configurations and Dockerfiles
- `kubernetes/` - Kubernetes manifests
- `helm/` - Helm charts
- `terraform/` - Terraform IaC
- `kustomize/` - Kustomize configurations
- `cloud/` - Cloud-specific deployments

**Structure:**
```
deployment/
├── docker/
│   ├── Dockerfile.base
│   ├── Dockerfile.api
│   └── docker-compose.*.yml
├── kubernetes/
│   ├── base/
│   └── overlays/
├── helm/
│   └── greenlang/
├── terraform/
│   ├── aws/
│   ├── azure/
│   └── gcp/
└── cloud/
    └── cloudcode/
```

**When to add new code:**
- Docker images → `deployment/docker/`
- K8s resources → `deployment/kubernetes/`
- Helm charts → `deployment/helm/`
- IaC configs → `deployment/terraform/`

**Note:** `docker-compose.yml` remains at root for convenience

### `docs/`
**Purpose:** Documentation hub (1,600+ files)
**Contains:**
- Architecture documentation
- API references
- User guides
- Operation manuals
- Migration guides
- Planning documents

**Structure:**
```
docs/
├── README.md                      # Documentation index
├── QUICK_START.md                 # Quick start guide
├── API_REFERENCE_COMPLETE.md      # Full API docs
├── ARCHITECTURE.md                # Architecture overview
├── architecture/                  # Architecture docs
├── guides/                        # How-to guides
├── operations/                    # Operational runbooks
├── security/                      # Security documentation
├── compliance/                    # Compliance guides
├── migration/                     # Migration guides
├── reports/                       # Status and progress reports
├── status/                        # Status tracking
├── planning/                      # Strategic planning
│   └── greenlang-2030-vision/     # 2030 vision documents
└── examples/                      # Code examples
```

**When to add new documentation:**
- API docs → `docs/API_*.md` or `docs/guides/`
- Architecture → `docs/architecture/`
- Security → `docs/security/`
- Reports → `docs/reports/`
- Status updates → `docs/status/`

### `examples/`
**Purpose:** Code examples and tutorials
**Contains:**
- Quick start examples
- Tutorial code
- Integration examples
- Application examples
- Advanced use cases

**Structure:**
```
examples/
├── quickstart/          # 5-minute quick start
├── tutorials/           # Step-by-step tutorials
├── integrations/        # Third-party integrations
├── applications/        # Sample applications
├── pipelines/           # Pipeline examples
├── packs/              # Pack examples
├── agentspec_v2/       # Agent specification examples
├── sdk/                # SDK usage examples
└── validation_rules/   # Validation rule examples
```

**When to add new code:**
- Tutorial code → `examples/tutorials/`
- Integration examples → `examples/integrations/`
- Sample apps → `examples/applications/`

### `greenlang/` (Core Platform)
**Purpose:** Core platform source code
**Contains:** All core GreenLang modules and packages

**Primary structure:**
```
greenlang/
├── __init__.py
├── api/                    # REST API endpoints
├── agents/                 # Agent framework
│   ├── base/               # Base agent classes
│   ├── calculator/         # Calculator agents
│   ├── forecast/           # Forecasting agents
│   ├── intake/             # Data intake agents
│   └── reporter/           # Reporting agents
├── calculations/           # Calculation engines (CONSOLIDATED)
│   ├── engines/            # Core calculation engines
│   ├── emission_factors/   # Emission factor loaders
│   ├── carbon/             # Carbon-specific calculations
│   └── energy/             # Energy calculations
├── cli/                    # Command-line interface
├── config/                 # Configuration management (CONSOLIDATED)
│   ├── settings/           # Settings modules
│   ├── schemas/            # Config validation schemas
│   └── defaults/           # Default configurations
├── core/                   # Core utilities and base classes
├── data/                   # Data handling (CONSOLIDATED)
│   ├── engineering/        # Data engineering utilities
│   ├── loaders/            # Data loaders
│   ├── transformers/       # Data transformers
│   └── validators/         # Data validators
├── database/               # Database layer (CONSOLIDATED)
│   ├── models/             # ORM models
│   ├── migrations/         # Alembic migrations
│   ├── repositories/       # Data access layer
│   └── connections/        # Connection management
├── integrations/           # External system integrations
│   ├── erp/                # ERP connectors (SAP, Oracle, Workday)
│   ├── llm/                # LLM providers
│   └── vector_db/          # Vector database integrations
├── intelligence/           # AI/ML intelligence layer
│   ├── providers/          # LLM providers
│   ├── embeddings/         # Embedding models
│   └── rag/                # RAG system
├── monitoring/             # Observability and monitoring
├── packs/                  # Pack system
├── rag/                    # Retrieval-Augmented Generation
├── security/               # Security features (auth, encryption)
├── testing/                # Testing utilities
├── tools/                  # Agent tools
├── utils/                  # General utilities
└── validation/             # Validation framework
```

**Naming Conventions:**
- **Use singular module names** (Python standard): `config/`, `calculation/`, `adapter/`
- **Exceptions:** When plural makes semantic sense: `agents/`, `calculations/`, `integrations/`
- **Class names:** PascalCase (`CalculationEngine`, `EmissionFactorLoader`)
- **Function names:** snake_case (`calculate_emissions`, `load_factors`)
- **Constants:** UPPER_SNAKE_CASE (`DEFAULT_EMISSION_FACTOR`, `MAX_RETRIES`)

**Import Patterns:**
```python
# Preferred: Import from consolidated modules
from greenlang.calculations import CalculationEngine
from greenlang.config import settings
from greenlang.database.models import EmissionFactor
from greenlang.data.loaders import CSVLoader

# Avoid: Importing from legacy/deprecated paths
# from greenlang.calculation import ...  # Old path
# from greenlang.calculators import ... # Old path
```

**When to add new code:**
- New agents → `greenlang/agents/{category}/`
- New calculations → `greenlang/calculations/`
- New integrations → `greenlang/integrations/{system}/`
- New database models → `greenlang/database/models/`
- New API endpoints → `greenlang/api/`
- Utilities → `greenlang/utils/`

### `logs/`
**Purpose:** Application logs
**Status:** Gitignored
**Contains:** Runtime log files from application execution

### `reports/`
**Purpose:** Generated reports and analysis
**Contains:**
- Emission calculation reports
- Compliance reports
- Analysis results
- Generated artifacts

**Note:** This is for runtime-generated reports. Documentation reports go in `docs/reports/`

### `scripts/`
**Purpose:** Utility scripts
**Contains:**
- `dev/` - Development utilities (generators, runners)
- `maintenance/` - Fix and cleanup scripts
- `testing/` - Test runners and harness scripts

**Structure:**
```
scripts/
├── dev/
│   ├── generate_*.py      # Code generators
│   └── run_*.py           # Development runners
├── maintenance/
│   └── fix_*.py           # Maintenance scripts
└── testing/
    └── test_*.py          # Test utilities
```

**When to add new code:**
- Development tools → `scripts/dev/`
- Maintenance scripts → `scripts/maintenance/`
- Test utilities → `scripts/testing/`

### `test-reports/`
**Purpose:** Test execution reports
**Status:** Gitignored
**Contains:** JUnit XML, coverage reports, test artifacts

### `tests/`
**Purpose:** Comprehensive test suite
**Contains:**
- Unit tests
- Integration tests
- End-to-end tests
- Test fixtures
- Load tests
- Pack tests

**Structure:**
```
tests/
├── unit/              # Unit tests (mirror greenlang/ structure)
│   ├── agents/
│   ├── calculations/
│   ├── api/
│   └── ...
├── integration/       # Integration tests
├── e2e/              # End-to-end tests
├── fixtures/         # Test fixtures and data
├── packs/            # Pack tests (from test-pack/, test-gpl-pack/, etc.)
└── load/             # Load and performance tests
```

**Naming Conventions:**
- Test files: `test_*.py`
- Test functions: `test_{feature}_{scenario}()`
- Test classes: `Test{Component}`

**When to add new tests:**
- Unit tests → `tests/unit/{module}/test_{file}.py`
- Integration tests → `tests/integration/test_{integration}.py`
- E2E tests → `tests/e2e/test_{workflow}.py`
- Fixtures → `tests/fixtures/`

**Import pattern:**
```python
import pytest
from greenlang.calculations import CalculationEngine

def test_calculate_emissions_natural_gas():
    engine = CalculationEngine()
    result = engine.calculate(
        activity_type="fuel_combustion",
        fuel_type="natural_gas",
        quantity=1000,
        unit="kWh"
    )
    assert result.emissions_co2e > 0
```

### `tools/`
**Purpose:** Development and operational tools
**Contains:**
- Development tooling
- Operational utilities
- Analysis tools
- Migration tools

---

## Naming Conventions

### Python Modules
**Standard:** Use **singular** names for modules (PEP 8 recommendation)

✅ **Correct:**
- `config/` (not `configs/`)
- `adapter/` (not `adapters/`)
- `benchmark/` (not `benchmarks/`)

⚠️ **Exceptions:** Plural when semantically appropriate:
- `agents/` (collection of many agent types)
- `calculations/` (multiple calculation types)
- `integrations/` (multiple integrations)
- `examples/` (collection of examples)

### File Naming
- Python files: `lowercase_with_underscores.py`
- Test files: `test_*.py`
- Configuration: `lowercase.yaml`, `lowercase.toml`
- Documentation: `UPPERCASE.md` or `Title_Case.md`

### Code Naming
- **Classes:** PascalCase (`CalculationEngine`, `EmissionFactor`)
- **Functions/Methods:** snake_case (`calculate_emissions`, `load_config`)
- **Constants:** UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT`, `MAX_RETRIES`)
- **Private:** Prefix with `_` (`_internal_function`, `_PrivateClass`)

---

## Import Patterns

### Preferred Import Style

```python
# Application-level imports
from greenlang.calculations import CalculationEngine
from greenlang.config import settings
from greenlang.database.models import EmissionFactor
from greenlang.agents.calculator import CalculatorAgent

# Specific imports
from greenlang.calculations.engines import CarbonCalculator
from greenlang.data.loaders import CSVLoader, JSONLoader

# Type hints
from typing import List, Dict, Optional
from greenlang.types import EmissionResult
```

### Avoid These Patterns

```python
# ❌ Avoid: Wildcard imports
from greenlang.calculations import *

# ❌ Avoid: Importing from deprecated paths
from greenlang.calculation import CalculationEngine  # Old path

# ❌ Avoid: Relative imports in top-level code
from ..calculations import CalculationEngine

# ❌ Avoid: Importing entire modules when you need one function
import greenlang.calculations.engines
result = greenlang.calculations.engines.CarbonCalculator().calculate()
```

---

## Adding New Code: Decision Tree

### 1. New Agent Type
**Question:** What does the agent do?

- **Calculation/Analysis** → `greenlang/agents/calculator/`
- **Data Intake** → `greenlang/agents/intake/`
- **Forecasting** → `greenlang/agents/forecast/`
- **Reporting** → `greenlang/agents/reporter/`
- **Other** → `greenlang/agents/{category}/`

**Tests:** `tests/unit/agents/{category}/test_{agent_name}.py`

### 2. New Calculation Method
**Question:** What type of calculation?

- **Emission factors** → `greenlang/calculations/emission_factors/`
- **Carbon calculations** → `greenlang/calculations/carbon/`
- **Energy calculations** → `greenlang/calculations/energy/`
- **Core engine** → `greenlang/calculations/engines/`

**Tests:** `tests/unit/calculations/test_{calculation_name}.py`

### 3. New Integration
**Question:** What system are you integrating?

- **ERP (SAP, Oracle, Workday)** → `greenlang/integrations/erp/{system}/`
- **LLM (OpenAI, Anthropic)** → `greenlang/integrations/llm/{provider}/`
- **Vector DB** → `greenlang/integrations/vector_db/{provider}/`
- **Other** → `greenlang/integrations/{category}/{system}/`

**Tests:** `tests/integration/test_{system}_integration.py`

### 4. New Application
**Question:** Is this a production regulatory compliance application?

- **Yes** → `applications/GL-{NAME}-APP/`
- **Example/Demo** → `examples/applications/{name}/`

**Structure:**
```
applications/GL-{NAME}-APP/
├── README.md
├── src/
├── tests/
├── docs/
└── deployment/
```

### 5. New Documentation
**Question:** What type of documentation?

- **API documentation** → `docs/API_*.md` or `docs/guides/`
- **Architecture** → `docs/architecture/`
- **User guide** → `docs/guides/` or application `docs/`
- **Security** → `docs/security/`
- **Status/Report** → `docs/reports/` or `docs/status/`
- **Migration** → `docs/migration/`

### 6. New Dataset
**Question:** Is this runtime or static data?

- **Static reference data** → `datasets/{category}/`
- **Runtime data** → `data/{category}/`
- **Emission factors** → `datasets/emission_factors/`
- **Knowledge base** → `datasets/knowledge_base/`

### 7. New Test
**Question:** What type of test?

- **Unit test** → `tests/unit/{module}/test_{file}.py`
- **Integration test** → `tests/integration/test_{integration}.py`
- **E2E test** → `tests/e2e/test_{workflow}.py`
- **Load test** → `tests/load/test_{scenario}.py`

### 8. New Deployment Config
**Question:** What deployment technology?

- **Docker** → `deployment/docker/`
- **Kubernetes** → `deployment/kubernetes/`
- **Helm** → `deployment/helm/`
- **Terraform** → `deployment/terraform/{cloud}/`

---

## Deprecated Paths (Post-Reorganization)

The following paths have been **consolidated** or **removed**:

| Old Path | New Path | Status |
|----------|----------|--------|
| `greenlang/calculation/` | `greenlang/calculations/` | Consolidated |
| `greenlang/calculators/` | `greenlang/calculations/` | Consolidated |
| `greenlang/configs/` | `greenlang/config/` | Consolidated |
| `greenlang/db/` | `greenlang/database/` | Consolidated |
| `greenlang/data_engineering/` | `greenlang/data/engineering/` | Consolidated |
| `greenlang/datasets/` | `datasets/` (root) | Moved |
| `greenlang/examples/` | `examples/` (root) | Moved |
| `core/greenlang/` | `greenlang/core/` | Consolidated |
| `docker/` | `deployment/docker/` | Moved |
| `k8s/` | `deployment/kubernetes/` | Consolidated |
| `kubernetes/` | `deployment/kubernetes/` | Consolidated |
| `cc_deployment/` | `deployment/cloud/` | Moved |
| `test-pack/` | `tests/packs/` | Moved |
| `test-gpl-pack/` | `tests/packs/gpl/` | Moved |
| `test-mit-pack/` | `tests/packs/mit/` | Moved |
| `test-scaffold-pack/` | `tests/packs/scaffold/` | Moved |
| `load-tests/` | `tests/load/` | Moved |
| Root `*.py` scripts | `scripts/{category}/` | Organized |

**Migration Notes:**
- Backward-compatible re-exports may exist for a limited time
- Update imports to use new paths
- See `docs/migration/REORGANIZATION_MIGRATION.md` for detailed migration guide

---

## CI/CD Path References

When updating CI/CD workflows, ensure paths reference the new structure:

### GitHub Actions
```yaml
# .github/workflows/test.yml
- name: Run tests
  run: pytest tests/

- name: Build Docker image
  run: docker build -f deployment/docker/Dockerfile .

- name: Deploy to K8s
  run: kubectl apply -f deployment/kubernetes/
```

### Docker Compose
```yaml
# docker-compose.yml (root)
services:
  api:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.api
    volumes:
      - ./greenlang:/app/greenlang
      - ./tests:/app/tests
```

### Makefile
```makefile
# Makefile
test:
	pytest tests/

build:
	docker build -f deployment/docker/Dockerfile -t greenlang:latest .

deploy-k8s:
	kubectl apply -f deployment/kubernetes/base/
```

---

## Protected Boundaries

The following directories are **protected** and must **NOT be modified** without explicit approval:

1. **`2026_PRD_MVP/`** - 2026 MVP Product Requirements
2. **`cbam-pack-mvp/`** - CBAM Pack MVP reference implementation

**Rationale:** These directories contain reference implementations and specifications that must remain stable.

---

## Quick Reference

### Where to add new...

| Type | Location |
|------|----------|
| Agent | `greenlang/agents/{category}/{agent_name}.py` |
| Calculation | `greenlang/calculations/{category}/{calc_name}.py` |
| API endpoint | `greenlang/api/{resource}/{endpoint}.py` |
| Database model | `greenlang/database/models/{model_name}.py` |
| Integration | `greenlang/integrations/{system}/{provider}.py` |
| Application | `applications/GL-{NAME}-APP/` |
| Configuration | `greenlang/config/{config_name}.py` |
| Utility | `greenlang/utils/{util_name}.py` |
| Test | `tests/{type}/{module}/test_{name}.py` |
| Documentation | `docs/{category}/{doc_name}.md` |
| Example | `examples/{category}/{example_name}/` |
| Dataset | `datasets/{category}/{dataset_name}/` |
| Script | `scripts/{category}/{script_name}.py` |
| Deployment | `deployment/{tech}/{config_name}` |

---

## Validation Commands

To verify the directory structure and imports:

```bash
# Check directory structure
ls -d */ | sort

# Verify no broken imports
python -c "import greenlang"

# Run import organization
isort greenlang/ tests/

# Run linting
ruff check greenlang/ tests/

# Run type checking
mypy greenlang/

# Run full test suite
pytest tests/

# Build Docker to verify paths
docker build -f deployment/docker/Dockerfile .
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-25 | Initial post-reorganization documentation |

---

## See Also

- [REORGANIZATION_PRD.md](../../REORGANIZATION_PRD.md) - Reorganization plan
- [docs/migration/REORGANIZATION_MIGRATION.md](../migration/REORGANIZATION_MIGRATION.md) - Migration guide
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Contribution guidelines
- [docs/ARCHITECTURE.md](../ARCHITECTURE.md) - Architecture overview
- [docs/API_REFERENCE_COMPLETE.md](../API_REFERENCE_COMPLETE.md) - API documentation

---

**Questions or Issues?**
- Open an issue: [GitHub Issues](https://github.com/greenlang/greenlang/issues)
- Join Discord: [discord.gg/greenlang](https://discord.gg/greenlang)
- Email: support@greenlang.io
