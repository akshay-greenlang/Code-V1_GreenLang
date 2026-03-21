# PACK-029 Interim Targets Pack -- Contributing Guide

**Pack ID:** PACK-029-interim-targets
**Version:** 1.0.0
**Last Updated:** 2026-03-19

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment Setup](#development-environment-setup)
3. [Project Structure](#project-structure)
4. [Coding Standards](#coding-standards)
5. [Testing Requirements](#testing-requirements)
6. [Pull Request Process](#pull-request-process)
7. [Code Review Guidelines](#code-review-guidelines)
8. [Issue Reporting](#issue-reporting)
9. [Roadmap for Future Versions](#roadmap-for-future-versions)
10. [Contribution License Agreement](#contribution-license-agreement)

---

## Getting Started

### Prerequisites

Before contributing to PACK-029, ensure you have:

1. Python 3.11 or higher installed
2. PostgreSQL 16 with TimescaleDB extension
3. Redis 7+
4. Git with commit signing configured
5. Access to the GreenLang development environment
6. Familiarity with SBTi Corporate Net-Zero Standard and LMDI methodology

### Contributor Agreement

All contributors must sign the GreenLang Contributor License Agreement (CLA) before submitting code. Contact `legal@greenlang.io` for the CLA.

---

## Development Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/greenlang/packs.git
cd packs/net-zero/PACK-029-interim-targets
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
# Copy example config
cp .env.example .env

# Edit with your local settings
# INTERIM_TARGETS_DB_HOST=localhost
# INTERIM_TARGETS_DB_PORT=5432
# etc.
```

### Step 5: Apply Test Migrations

```bash
# Apply migrations to local database
python scripts/apply_migrations.py --env dev
```

### Step 6: Run Tests

```bash
# Run full test suite
pytest tests/ -v --cov=engines --cov=workflows --cov=templates --cov=integrations

# Run specific engine tests
pytest tests/test_interim_target_engine.py -v

# Run with coverage report
pytest tests/ --cov --cov-report=html
```

---

## Project Structure

```
PACK-029-interim-targets/
    config/
        pack_config.py              # Pack configuration
        presets/
            sbti_15c.yaml           # SBTi 1.5C preset
            sbti_wb2c.yaml          # SBTi WB2C preset
            race_to_zero.yaml       # Race to Zero preset
            corporate_net_zero.yaml # Corporate Net-Zero preset
            financial_institution.yaml
            sme_simplified.yaml
            manufacturing.yaml
    engines/
        __init__.py
        interim_target_engine.py    # Engine 1: Interim target calculation
        quarterly_monitoring_engine.py  # Engine 2: Quarterly monitoring
        annual_review_engine.py     # Engine 3: Annual review
        variance_analysis_engine.py # Engine 4: LMDI variance analysis
        trend_extrapolation_engine.py   # Engine 5: Trend forecasting
        corrective_action_engine.py # Engine 6: Corrective action planning
        target_recalibration_engine.py  # Engine 7: Target recalibration
        sbti_validation_engine.py   # Engine 8: SBTi 21-criteria validation
        carbon_budget_tracker_engine.py # Engine 9: Carbon budget tracking
        alert_generation_engine.py  # Engine 10: Alert generation
    workflows/
        __init__.py
        interim_target_setting_workflow.py
        quarterly_monitoring_workflow.py
        annual_progress_review_workflow.py
        variance_investigation_workflow.py
        corrective_action_planning_workflow.py
        annual_reporting_workflow.py
        target_recalibration_workflow.py
    templates/
        __init__.py
        interim_targets_summary.py
        quarterly_progress_report.py
        annual_progress_report.py
        variance_waterfall_report.py
        corrective_action_plan_report.py
        sbti_validation_report.py
        cdp_export_template.py
        tcfd_disclosure_template.py
        carbon_budget_report.py
        executive_dashboard_template.py
    integrations/
        __init__.py
        pack_orchestrator.py
        pack021_bridge.py
        pack028_bridge.py
        mrv_bridge.py
        sbti_portal_bridge.py
        cdp_bridge.py
        tcfd_bridge.py
        alerting_bridge.py
        health_check.py
        setup_wizard.py
    migrations/
        V196__PACK029_interim_targets.sql
        V196__PACK029_interim_targets.down.sql
        V197__PACK029_annual_pathways.sql
        V197__PACK029_annual_pathways.down.sql
        ... (through V210)
    tests/
        __init__.py
        test_interim_target_engine.py
        test_quarterly_monitoring_engine.py
        ... (15 test files)
    docs/
        API_REFERENCE.md
        USER_GUIDE.md
        INTEGRATION_GUIDE.md
        VALIDATION_REPORT.md
        DEPLOYMENT_CHECKLIST.md
        CHANGELOG.md
        CONTRIBUTING.md
        CALCULATIONS/
        REGULATORY/
        USE_CASES/
    README.md
    pack.yaml
    requirements.txt
    requirements-dev.txt
```

---

## Coding Standards

### Python Style

- **PEP 8** compliance (enforced by `flake8`)
- **Black** formatter (line length: 99)
- **isort** for import ordering (profile: black)
- **mypy** for type checking (strict mode)

### Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Files | `snake_case.py` | `interim_target_engine.py` |
| Classes | `PascalCase` | `InterimTargetEngine` |
| Functions | `snake_case` | `calculate_annual_rate` |
| Constants | `UPPER_SNAKE_CASE` | `SBTI_THRESHOLDS` |
| Private methods | `_leading_underscore` | `_resolve_baseline_total` |
| Test functions | `test_snake_case` | `test_linear_pathway` |

### Pydantic Models

All input/output models must use Pydantic v2 BaseModel:

```python
from pydantic import BaseModel, Field

class MyInput(BaseModel):
    """Docstring with all attributes documented."""
    field_name: str = Field(..., description="Clear description")
    optional_field: Optional[int] = Field(default=None, ge=0)
```

### Decimal Arithmetic

All calculation code must use `Decimal` instead of `float`:

```python
from decimal import Decimal

# Correct
result = Decimal("4.2") * Decimal("10")

# Incorrect (never use float for calculations)
result = 4.2 * 10
```

### Provenance Hashing

All engine outputs must include SHA-256 provenance hashing:

```python
result.provenance_hash = _compute_hash(result)
```

### Documentation

All public classes and functions must have docstrings following Google style:

```python
def calculate(self, data: InterimTargetInput) -> InterimTargetResult:
    """Run complete interim target calculation.

    Args:
        data: Validated interim target input.

    Returns:
        InterimTargetResult with milestones, timelines, and validation.

    Raises:
        ValueError: If baseline total is zero.
    """
```

---

## Testing Requirements

### Minimum Coverage

- Overall code coverage: 92%+
- Each engine: 90%+
- Each workflow: 88%+
- Each template: 85%+

### Test Types Required

1. **Unit tests**: Every public method must have at least 3 tests (happy path, edge case, error case)
2. **Integration tests**: Every workflow must have end-to-end tests
3. **Accuracy tests**: Calculation engines must be cross-validated against manual calculations
4. **Property tests**: LMDI decomposition must pass perfect decomposition property in all cases
5. **Performance tests**: New engines must meet latency targets

### Running Tests

```bash
# Full suite
pytest tests/ -v --cov

# Specific engine
pytest tests/test_interim_target_engine.py -v

# With coverage report
pytest tests/ --cov --cov-report=html --cov-report=term-missing

# Performance tests only
pytest tests/test_performance.py -v

# Accuracy tests only
pytest tests/ -k "accuracy" -v
```

### Test Fixtures

Use pytest fixtures for common test data:

```python
@pytest.fixture
def sample_baseline():
    return BaselineData(
        base_year=2021,
        scope_1_tco2e=Decimal("50000"),
        scope_2_tco2e=Decimal("30000"),
        scope_3_tco2e=Decimal("120000"),
    )
```

---

## Pull Request Process

### Step 1: Create a Branch

```bash
git checkout -b feature/add-monthly-monitoring
```

### Step 2: Make Changes

- Follow coding standards above
- Add tests for all new code
- Update documentation if APIs change

### Step 3: Run Pre-Commit Checks

```bash
# Format code
black engines/ workflows/ templates/ integrations/ tests/
isort engines/ workflows/ templates/ integrations/ tests/

# Lint
flake8 engines/ workflows/ templates/ integrations/

# Type check
mypy engines/ workflows/ templates/ integrations/ --strict

# Run tests
pytest tests/ -v --cov
```

### Step 4: Create Pull Request

- Title: Brief description (e.g., "Add monthly monitoring frequency support")
- Description: What changed, why, and how to test
- Link related issues
- Request review from at least 2 maintainers

### Step 5: Address Review Feedback

- Respond to all comments
- Push fixes as additional commits
- Request re-review after changes

---

## Code Review Guidelines

### Reviewers Must Check

1. **Correctness**: Calculations match documented formulas
2. **Decimal usage**: No float arithmetic in calculation paths
3. **Provenance**: SHA-256 hashing on all outputs
4. **Tests**: Adequate coverage and edge cases
5. **Documentation**: Docstrings and API docs updated
6. **Performance**: Within latency targets
7. **Security**: No credentials in code, proper input validation
8. **SBTi accuracy**: Thresholds match Corporate Manual v5.3

### Review Turnaround

- Standard PRs: 2 business days
- Urgent fixes: Same day
- Large features: 5 business days

---

## Issue Reporting

### Bug Reports

Use the `bug` template with:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Exact steps to trigger the issue
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, OS, PostgreSQL version
6. **Logs**: Relevant log output

### Feature Requests

Use the `enhancement` template with:

1. **Summary**: Brief description of the feature
2. **Use case**: Why this feature is needed
3. **Proposed solution**: How it could work
4. **Alternatives**: Other approaches considered

---

## Roadmap for Future Versions

### v1.1.0 (Planned: Q3 2026)

- Real-time emissions monitoring via streaming data
- Monthly monitoring frequency
- Variable cost modeling for corrective actions
- Enhanced ARIMA with automatic parameter tuning
- Interactive HTML charts (Chart.js)
- Webhook event delivery
- Batch multi-entity processing

### v1.2.0 (Planned: Q4 2026)

- Multi-entity hierarchical aggregation
- Automated SBTi portal submission
- ML-based corrective action recommendation
- Climate scenario stress testing
- Supply chain interim target cascading
- Enhanced data quality scoring

### v2.0.0 (Planned: Q1 2027)

- SBTi Corporate Standard v3.0 alignment (when released)
- Sector-specific interim targets (integration with PACK-028 SDA)
- AI-assisted variance narrative generation
- Real-time executive dashboard with live data

---

## Contribution License Agreement

By contributing to PACK-029, you agree that:

1. Your contributions are your original work
2. You have the right to submit them
3. You grant GreenLang a perpetual, irrevocable license to use your contributions
4. Your contributions will be licensed under the same terms as the project

Contact `legal@greenlang.io` for the formal CLA document.

---

**End of Contributing Guide**
