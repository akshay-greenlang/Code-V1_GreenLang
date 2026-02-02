# Contributing to GL-VCCI-Carbon-APP

Thank you for your interest in contributing to the GL-VCCI Scope 3 Value Chain Carbon Intelligence Platform! This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Development Workflow](#development-workflow)
5. [Coding Standards](#coding-standards)
6. [Testing Requirements](#testing-requirements)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Release Process](#release-process)
10. [Contact](#contact)

---

## Code of Conduct

### Our Standards

- **Professional**: Maintain professional communication at all times
- **Respectful**: Respect diverse opinions and experiences
- **Constructive**: Provide constructive feedback
- **Collaborative**: Work together towards project goals
- **Security-First**: Never commit secrets or sensitive data

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or inflammatory comments
- Publishing private information
- Committing secrets or credentials
- Any conduct that violates professional ethics

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.10+** (3.11+ recommended)
- **Git** (version control)
- **PostgreSQL 15+** (for local development)
- **Redis 7+** (for caching)
- **Docker** (optional, for containerized development)

### Repository Structure

```
GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/
├── agents/              # 5 core agents
├── cli/                 # Command-line interface
├── sdk/                 # Python SDK
├── provenance/          # SHA-256 provenance tracking
├── data/                # Emission factor databases
├── connectors/          # ERP integrations (SAP, Oracle, Workday)
├── utils/               # Shared utilities
├── config/              # Configuration files
├── tests/               # Test suite (1,200+ tests)
├── scripts/             # Utility scripts
├── deployment/          # Infrastructure-as-Code
├── monitoring/          # Observability
├── pack.yaml            # GreenLang pack specification
├── gl.yaml              # Agent configuration
└── requirements.txt     # Python dependencies
```

---

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/greenlang/gl-vcci-carbon-app.git
cd GL-VCCI-Carbon-APP/VCCI-Scope3-Platform
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### 4. Set Up Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
# IMPORTANT: Never commit .env to version control!
```

### 5. Initialize Database

```bash
# Initialize PostgreSQL database
python scripts/init_database.py

# Seed emission factor databases
python scripts/seed_emission_factors.py --sources defra,epa
```

### 6. Install Pre-Commit Hooks

```bash
# Install pre-commit hooks (runs linting/formatting on commit)
pre-commit install
```

---

## Development Workflow

### 1. Create Feature Branch

```bash
# Sync with main branch
git checkout master
git pull origin master

# Create feature branch
git checkout -b feature/your-feature-name
```

**Branch Naming Conventions:**
- `feature/` - New features (e.g., `feature/add-category-15-support`)
- `fix/` - Bug fixes (e.g., `fix/calculation-rounding-error`)
- `docs/` - Documentation changes (e.g., `docs/update-api-docs`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-intake-agent`)
- `test/` - Test additions/improvements (e.g., `test/add-category-1-tests`)
- `chore/` - Maintenance tasks (e.g., `chore/update-dependencies`)

### 2. Make Changes

Follow [Coding Standards](#coding-standards) below.

### 3. Write Tests

**IMPORTANT**: All new code must include tests.

```bash
# Run tests locally
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

**Test Coverage Requirements:**
- Unit tests: **>90% coverage** (strictly enforced)
- Integration tests: All critical paths covered
- Performance tests: All agents meet SLAs

### 4. Run Linters and Formatters

```bash
# Format code (automatically fixes issues)
black .
isort .

# Lint code (identifies issues)
ruff check .

# Type checking
mypy .
```

**Note**: Pre-commit hooks will run these automatically on `git commit`.

### 5. Update Documentation

- Update docstrings for all new functions/classes
- Update README.md if adding new features
- Update STATUS.md to reflect progress
- Add examples to relevant documentation

### 6. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: Add support for Category 15 (Investments)"

# Pre-commit hooks will run automatically
# If hooks fail, fix issues and re-commit
```

**Commit Message Format:**

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Test additions/improvements
- `chore`: Maintenance tasks

**Examples:**
```
feat(calculator): Add support for Category 15 (Investments)

Implemented calculation logic for Category 15 using spend-based method.
Added 1,200 new emission factors for financial instruments.

Closes #123
```

```
fix(intake): Fix CSV parsing error for UTF-8 BOM files

Fixed issue where CSV files with UTF-8 BOM were failing to parse.
Added encoding detection using chardet library.

Fixes #456
```

### 7. Push to Remote

```bash
git push origin feature/your-feature-name
```

### 8. Create Pull Request

Go to GitHub and create a Pull Request (PR) from your branch to `master`.

See [Pull Request Process](#pull-request-process) below.

---

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes `"` for strings (not single quotes)
- **Imports**: Sorted alphabetically using `isort`

### Code Formatting

We use **Black** for code formatting:

```bash
# Format all code
black .

# Check formatting (no changes)
black --check .
```

### Linting

We use **Ruff** for linting:

```bash
# Lint code
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Type Hints

**REQUIRED**: All functions must have type hints.

```python
# Good ✅
def calculate_emissions(
    quantity: float,
    emission_factor: float,
    unit: str
) -> float:
    """Calculate emissions in tCO2e.

    Args:
        quantity: Amount of material/activity
        emission_factor: Emission factor (kgCO2e/unit)
        unit: Unit of measurement

    Returns:
        float: Emissions in tCO2e
    """
    return (quantity * emission_factor) / 1000  # Convert kg to tonnes

# Bad ❌ (no type hints)
def calculate_emissions(quantity, emission_factor, unit):
    return (quantity * emission_factor) / 1000
```

### Docstrings

**REQUIRED**: All functions, classes, and modules must have docstrings.

Use **Google-style** docstrings:

```python
def get_emission_factor(
    category: int,
    product: str,
    region: str,
    year: int,
    tier: str = "tier_2"
) -> dict:
    """Get emission factor for a product/service.

    This function queries the emission factor database and returns the most
    appropriate emission factor based on category, product, region, and year.

    Args:
        category: Scope 3 category (1-15)
        product: Product or service name (e.g., "Steel")
        region: Geographic region (ISO 3166-1 alpha-2, e.g., "US", "GB")
        year: Year for emission factor (e.g., 2024)
        tier: Calculation tier ("tier_1", "tier_2", "tier_3")

    Returns:
        dict: Emission factor record with structure:
            {
                "factor_id": "defra_2024_steel_eu",
                "value": 1.85,
                "unit": "kgCO2e/kg",
                "uncertainty": 0.15,
                "source": "DEFRA",
                "data_quality": 85
            }

    Raises:
        ValueError: If category is not between 1 and 15
        KeyError: If emission factor not found

    Example:
        >>> ef = get_emission_factor(
        ...     category=1,
        ...     product="Steel",
        ...     region="US",
        ...     year=2024
        ... )
        >>> print(ef["value"])
        1.85
    """
    # Implementation...
```

### Naming Conventions

- **Functions/methods**: `snake_case` (e.g., `calculate_emissions`)
- **Classes**: `PascalCase` (e.g., `Scope3CalculatorAgent`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_WORKERS`)
- **Private members**: Prefix with `_` (e.g., `_internal_method`)

### Error Handling

Always use specific exceptions:

```python
# Good ✅
def calculate_emissions(quantity: float, ef: float) -> float:
    if quantity < 0:
        raise ValueError(f"Quantity must be non-negative, got {quantity}")
    if ef < 0:
        raise ValueError(f"Emission factor must be non-negative, got {ef}")
    return quantity * ef

# Bad ❌
def calculate_emissions(quantity, ef):
    try:
        return quantity * ef
    except:  # Never use bare except!
        return 0
```

### Logging

Use structured logging:

```python
import structlog

logger = structlog.get_logger(__name__)

# Good ✅
logger.info(
    "calculation_completed",
    calculation_id="calc_123",
    category=1,
    emissions_tco2e=1234.56,
    duration_ms=150
)

# Bad ❌
print(f"Calculated {emissions} tCO2e for category {category}")
```

---

## Testing Requirements

### Test Coverage

**Minimum test coverage: 90%** (enforced by CI/CD)

```bash
# Run tests with coverage
pytest tests/ -v --cov=. --cov-report=html --cov-fail-under=90
```

### Test Structure

```python
# tests/agents/test_calculator_agent.py

import pytest
from agents import Scope3CalculatorAgent


class TestScope3CalculatorAgent:
    """Test suite for Scope3CalculatorAgent."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance for testing."""
        return Scope3CalculatorAgent(config=test_config)

    def test_calculate_category_1_tier_1(self, calculator):
        """Test Category 1 calculation with Tier 1 data."""
        # Arrange
        input_data = {
            "product": "Steel",
            "quantity": 1000,
            "unit": "kg",
            "supplier_ef": 1.85  # Supplier-specific emission factor
        }

        # Act
        result = calculator.calculate_category_1(input_data, tier="tier_1")

        # Assert
        assert result["emissions_tco2e"] == pytest.approx(1.85, rel=1e-2)
        assert result["tier"] == "tier_1"
        assert result["data_quality"] >= 95  # Tier 1 = high quality

    def test_calculate_category_1_invalid_quantity(self, calculator):
        """Test that negative quantity raises ValueError."""
        input_data = {
            "product": "Steel",
            "quantity": -1000,  # Invalid!
            "unit": "kg"
        }

        with pytest.raises(ValueError, match="Quantity must be non-negative"):
            calculator.calculate_category_1(input_data)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/agents/test_calculator_agent.py -v

# Run specific test
pytest tests/agents/test_calculator_agent.py::TestScope3CalculatorAgent::test_calculate_category_1_tier_1 -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run integration tests only
pytest tests/integration/ -v

# Run performance tests
pytest tests/performance/ -v --benchmark-only
```

---

## Documentation

### Code Documentation

- **Docstrings**: Required for all functions, classes, modules
- **Type hints**: Required for all function parameters and return values
- **Examples**: Include usage examples in docstrings

### User Documentation

Update relevant documentation files:

- **README.md**: Project overview, quick start
- **STATUS.md**: Build plan tracker
- **PRD.md**: Product requirements (if changing features)
- **Agent specifications**: `specs/*.yaml` (if changing agent behavior)

---

## Pull Request Process

### Before Creating PR

1. ✅ All tests pass locally
2. ✅ Code coverage ≥90%
3. ✅ All linters pass (Black, Ruff, mypy)
4. ✅ Documentation updated
5. ✅ No merge conflicts with `master`

### Creating the PR

1. Go to GitHub repository
2. Click "New Pull Request"
3. Select your feature branch
4. Fill out PR template:

```markdown
## Description
Brief description of changes (1-2 sentences).

## Related Issues
Closes #123
Fixes #456

## Changes Made
- Added Category 15 (Investments) calculation logic
- Implemented 1,200 new emission factors for financial instruments
- Added 50 unit tests (coverage: 95%)

## Testing
- [x] Unit tests pass (95% coverage)
- [x] Integration tests pass
- [x] Manual testing completed

## Documentation
- [x] Updated docstrings
- [x] Updated README.md
- [x] Updated STATUS.md

## Checklist
- [x] Code follows style guide
- [x] Tests added/updated
- [x] Documentation updated
- [x] No secrets committed
- [x] Pre-commit hooks pass
```

### Review Process

1. **Automated Checks** (CI/CD):
   - Tests must pass (all 1,200+ tests)
   - Coverage must be ≥90%
   - Linters must pass (Black, Ruff, mypy)
   - No security vulnerabilities

2. **Code Review** (2 approvals required):
   - Technical Lead review
   - Domain Expert review (for emission factor changes)

3. **Approval**:
   - Once approved, PR can be merged
   - **Squash and merge** is preferred

### Merge Strategy

We use **squash and merge** to keep clean commit history:

- All commits in PR are squashed into single commit
- Commit message should summarize all changes
- Original commits preserved in PR description

---

## Release Process

### Versioning

We use **Semantic Versioning** (SemVer):

```
MAJOR.MINOR.PATCH

1.0.0 → 1.0.1 (patch: bug fix)
1.0.1 → 1.1.0 (minor: new feature, backward compatible)
1.1.0 → 2.0.0 (major: breaking change)
```

### Release Checklist

- [ ] All tests pass (1,200+ tests, 90%+ coverage)
- [ ] Security scan passes (no critical vulnerabilities)
- [ ] Performance benchmarks meet SLAs
- [ ] Documentation updated (README, CHANGELOG)
- [ ] Version bumped in `setup.py`, `__init__.py`
- [ ] Release notes drafted
- [ ] Tagged in Git (`git tag v1.0.0`)

---

## Contact

### Questions?

- **Email**: vcci@greenlang.io
- **Community**: https://community.greenlang.io
- **Issues**: https://github.com/greenlang/gl-vcci-carbon-app/issues

### Team

- **Project Lead**: TBD
- **Technical Lead**: TBD
- **Domain Expert (Carbon Accounting)**: TBD

---

**Thank you for contributing to the GL-VCCI Scope 3 Platform!** 🌍

Together, we're building the world's most advanced Scope 3 emissions tracking platform to help enterprises reduce their carbon footprint and combat climate change.
