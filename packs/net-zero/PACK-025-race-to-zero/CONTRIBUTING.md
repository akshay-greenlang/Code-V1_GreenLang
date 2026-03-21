# PACK-025 Race to Zero Pack - Contributing Guide

Thank you for your interest in contributing to the PACK-025 Race to Zero Pack.
This document provides guidelines for development setup, code style, testing,
and the pull request process.

---

## Development Setup

### Prerequisites

- Python >= 3.11
- PostgreSQL >= 16 with pgvector and TimescaleDB extensions
- Redis >= 7
- Git

### Environment Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd Code-V1_GreenLang

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install development dependencies
pip install pydantic>=2.0 pyyaml>=6.0 pandas>=2.0 numpy>=1.24 \
    httpx>=0.24 psycopg[binary]>=3.1 psycopg_pool>=3.1 redis>=5.0 \
    jinja2>=3.1 openpyxl>=3.1 cryptography>=41.0

# 4. Install development tools
pip install pytest pytest-cov pytest-asyncio black isort mypy pylint

# 5. Verify setup
cd packs/net-zero/PACK-025-race-to-zero
python -m pytest tests/ -v --tb=short
```

### Database Setup (Local Development)

```bash
# 1. Create database
createdb greenlang_dev

# 2. Enable extensions
psql greenlang_dev -c "CREATE EXTENSION IF NOT EXISTS pgvector;"
psql greenlang_dev -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# 3. Apply migrations V148 through V157
# (Use your preferred migration tool or apply SQL files directly)

# 4. Seed reference data
# (Run pack post-install seed commands)
```

---

## Code Style Guidelines

### Formatting

This project uses **Black** for code formatting and **isort** for import
sorting. All code must pass formatting checks before merge.

```bash
# Format code
black packs/net-zero/PACK-025-race-to-zero/ --line-length 99

# Sort imports
isort packs/net-zero/PACK-025-race-to-zero/ --profile black --line-length 99

# Check formatting (dry run)
black --check packs/net-zero/PACK-025-race-to-zero/
isort --check-only packs/net-zero/PACK-025-race-to-zero/
```

### Type Checking

All code must pass **mypy** type checking with strict mode.

```bash
mypy packs/net-zero/PACK-025-race-to-zero/ \
    --strict \
    --ignore-missing-imports \
    --no-implicit-optional
```

### Linting

All code must pass **pylint** with a minimum score of 9.0.

```bash
pylint packs/net-zero/PACK-025-race-to-zero/ \
    --disable=C0114,C0115,C0116 \
    --max-line-length=99
```

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Modules | `snake_case` | `pledge_commitment_engine.py` |
| Classes | `PascalCase` | `PledgeCommitmentEngine` |
| Functions | `snake_case` | `assess_eligibility()` |
| Constants | `UPPER_SNAKE_CASE` | `PHASE_DEPENDENCIES` |
| Variables | `snake_case` | `pledge_quality` |
| Enums | `PascalCase` (class), `UPPER_SNAKE_CASE` (members) | `PledgeQuality.STRONG` |
| Database tables | `gl_r2z_snake_case` | `gl_r2z_pledge_commitments` |
| Test functions | `test_snake_case` | `test_pledge_eligibility_valid()` |

### Docstrings

All public classes and functions must have docstrings following Google style:

```python
def assess_eligibility(
    organization_id: str,
    actor_type: str,
    criteria: Dict[str, Any],
) -> EligibilityResult:
    """Assess pledge eligibility against 8 Race to Zero criteria.

    Validates the organization's pledge commitment against the Race to Zero
    Interpretation Guide (June 2022) eligibility criteria.

    Args:
        organization_id: Unique organization identifier (UUID format).
        actor_type: One of CORPORATE, FINANCIAL_INSTITUTION, CITY, REGION,
            SME, HEAVY_INDUSTRY, SERVICES, MANUFACTURING.
        criteria: Dictionary of eligibility criterion values.

    Returns:
        EligibilityResult with per-criterion status and overall rating.

    Raises:
        ValueError: If actor_type is not a recognized type.
        ValidationError: If criteria dictionary is incomplete.
    """
```

### Import Order

Imports must follow this order (enforced by isort):

1. Standard library imports
2. Third-party imports
3. GreenLang platform imports
4. Pack-local imports

```python
# Standard library
import hashlib
from datetime import datetime
from typing import Dict, List, Optional

# Third-party
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# GreenLang platform
from greenlang.agents.foundation import Orchestrator

# Pack-local
from .pledge_commitment_engine import PledgeCommitmentEngine
```

---

## Testing Requirements

### Test Coverage

All contributions must maintain the minimum code coverage target:

| Module | Minimum Coverage |
|--------|-----------------|
| `engines/` | 90% |
| `workflows/` | 90% |
| `templates/` | 90% |
| `integrations/` | 90% |
| `config/` | 95% |
| **Overall** | **90%** |

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=90

# Run specific test module
python -m pytest tests/test_engines.py -v

# Run a single test
python -m pytest tests/test_engines.py::TestPledgeCommitmentEngine::test_eligibility -v

# Run with verbose output and timing
python -m pytest tests/ -v --durations=10
```

### Test Structure

Tests should follow the Arrange-Act-Assert (AAA) pattern:

```python
class TestPledgeCommitmentEngine:
    """Tests for the Pledge Commitment Engine."""

    def test_eligible_corporate_pledge(self):
        """A corporate entity meeting all 8 criteria should be rated STRONG."""
        # Arrange
        engine = PledgeCommitmentEngine()
        criteria = {
            "net_zero_target_year": 2050,
            "partner_initiative": "SBTi",
            "interim_target_year": 2030,
            "interim_reduction_pct": 50.0,
            # ... all 8 criteria
        }

        # Act
        result = engine.assess(
            organization_id="org-001",
            actor_type="CORPORATE",
            **criteria,
        )

        # Assert
        assert result.eligibility_status == "ELIGIBLE"
        assert result.quality_rating == "STRONG"
        assert len(result.criteria_results) == 8
        assert result.provenance_hash is not None

    def test_ineligible_missing_scope_coverage(self):
        """Missing Scope 3 for a corporate should reduce quality."""
        # ...
```

### Test Categories

| Category | Location | Description |
|----------|----------|-------------|
| Unit tests | `tests/test_engines.py` | Individual engine calculation verification |
| Workflow tests | `tests/test_workflows.py` | Phase sequencing and data flow |
| Template tests | `tests/test_templates.py` | Report output validation |
| Integration tests | `tests/test_integrations.py` | Bridge connectivity and error handling |
| Config tests | `tests/test_config.py` | Configuration loading and defaults |
| Preset tests | `tests/test_presets.py` | Actor-type preset validation |
| E2E tests | `tests/test_e2e.py` | Full pipeline scenarios |
| Init tests | `tests/test_init.py` | Package import verification |

---

## Pull Request Process

### Before Submitting

1. **Create a feature branch** from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Run all checks** locally:
   ```bash
   # Format
   black packs/net-zero/PACK-025-race-to-zero/ --line-length 99
   isort packs/net-zero/PACK-025-race-to-zero/ --profile black --line-length 99

   # Type check
   mypy packs/net-zero/PACK-025-race-to-zero/ --strict --ignore-missing-imports

   # Lint
   pylint packs/net-zero/PACK-025-race-to-zero/ --max-line-length=99

   # Test with coverage
   python -m pytest packs/net-zero/PACK-025-race-to-zero/tests/ \
       --cov=packs/net-zero/PACK-025-race-to-zero \
       --cov-report=term-missing \
       --cov-fail-under=90
   ```

3. **Update documentation** if your change affects:
   - Engine behavior or API surface
   - Workflow phases or dependencies
   - Integration bridge interfaces
   - Configuration options or presets
   - Standards compliance claims

### PR Description Template

```markdown
## Summary
- Brief description of the change

## Changes
- [ ] Engine changes (list affected engines)
- [ ] Workflow changes (list affected workflows)
- [ ] Integration changes (list affected bridges)
- [ ] Configuration changes (list affected presets)
- [ ] Documentation updates

## Testing
- [ ] All existing tests pass (797/797)
- [ ] New tests added for new functionality
- [ ] Coverage maintained above 90%

## Standards Impact
- [ ] No impact on Race to Zero compliance
- [ ] No impact on HLEG recommendation coverage
- [ ] No impact on Starting Line Criteria assessment
```

### Review Criteria

Pull requests are reviewed against:

- Code style compliance (Black, isort, mypy, pylint)
- Test coverage (90% minimum)
- Zero-hallucination guarantee (no LLM in calculation path)
- SHA-256 provenance integrity
- Backward compatibility with existing presets and configurations
- Documentation completeness

---

## Issue Templates

### Bug Report

```markdown
**Title**: [BUG] Brief description

**Engine/Component**: (e.g., Pledge Commitment Engine)

**Expected behavior**: What should happen

**Actual behavior**: What actually happens

**Steps to reproduce**:
1. ...
2. ...
3. ...

**Environment**: Python version, OS, database version

**Logs/Error output**: (if applicable)
```

### Feature Request

```markdown
**Title**: [FEATURE] Brief description

**Component**: (e.g., Sector Pathway Engine)

**Description**: Detailed description of the feature

**Use case**: Who needs this and why

**Standards reference**: (if applicable, e.g., "HLEG R3 sub-criterion 3.4")

**Acceptance criteria**:
- [ ] Criterion 1
- [ ] Criterion 2
```

---

## Commit Message Conventions

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or modifying tests |
| `perf` | Performance improvement |
| `chore` | Build process, dependency updates, tooling |

### Scopes

| Scope | Description |
|-------|-------------|
| `engines` | Engine changes |
| `workflows` | Workflow changes |
| `templates` | Template changes |
| `integrations` | Integration bridge changes |
| `config` | Configuration or preset changes |
| `tests` | Test infrastructure changes |
| `docs` | Documentation changes |

### Examples

```
feat(engines): add TPI 2024 sector pathway data to SectorPathwayEngine
fix(integrations): handle UNFCCC portal timeout with graceful degradation
docs(readme): update Starting Line assessment usage example
test(engines): add edge case tests for SME simplified scoring
refactor(workflows): extract common phase validation into base class
perf(integrations): add Redis L1 cache for sector benchmark lookups
```

---

## Documentation Standards

### Code Documentation

- All public classes and functions must have Google-style docstrings
- All engine calculations must document the formula and data source
- All integration bridges must document the external API contract
- SHA-256 provenance must be documented in engine output models

### Markdown Documentation

- Use GitHub-flavored Markdown
- Include code examples with proper syntax highlighting (`python`, `bash`, `yaml`)
- Use tables for structured comparisons
- Use ASCII art for architecture diagrams
- Cross-reference between documents using relative links
- Keep line length under 80 characters for prose

### Changelog Updates

All user-visible changes must be documented in `CHANGELOG.md` following the
[Keep a Changelog](https://keepachangelog.com/) format.

---

## Contact

- **Platform Team**: GreenLang Platform Team
- **Support tier**: Enterprise
- **Documentation**: https://docs.greenlang.io/packs/race-to-zero

---

*Contributing guide maintained by GreenLang Platform Team*
*PACK-025 Race to Zero Pack v1.0.0*
*Last updated: 2026-03-18*
