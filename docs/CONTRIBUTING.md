# Contributing to GreenLang

**Welcome to the GreenLang community! We're excited to have you contribute to the Climate Operating System.**

This guide will help you get started with contributing to GreenLang, from setting up your development environment to submitting your first pull request.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Setup](#development-setup)
4. [Code Style Guide](#code-style-guide)
5. [Testing Requirements](#testing-requirements)
6. [Pull Request Process](#pull-request-process)
7. [Review Process](#review-process)
8. [Community Guidelines](#community-guidelines)

---

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive Behaviors:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable Behaviors:**
- Trolling, insulting/derogatory comments, personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at conduct@greenlang.io.

---

## How Can I Contribute?

### Report Bugs

**Before Submitting a Bug Report:**
- Check the [existing issues](https://github.com/greenlang/greenlang/issues)
- Verify the bug exists in the latest version
- Collect information: version, OS, Python version, error messages

**Submit a Bug Report:**
```markdown
**Bug Description**: Clear and concise description

**Steps to Reproduce**:
1. Step one
2. Step two
3. Expected result
4. Actual result

**Environment**:
- GreenLang version: 0.3.0
- Python version: 3.10.5
- OS: Ubuntu 22.04
- Installation method: pip

**Additional Context**:
- Error messages
- Screenshots
- Related issues
```

### Suggest Enhancements

**Feature Request Template:**
```markdown
**Feature Description**: What feature would you like?

**Use Case**: Why is this feature needed?

**Proposed Solution**: How should it work?

**Alternatives Considered**: What other approaches did you consider?

**Additional Context**: Examples, mockups, references
```

### Contribute Code

**Areas Where We Need Help:**
- 🐛 Bug fixes
- 📚 Documentation improvements
- ✨ New agents for climate calculations
- 🧪 Test coverage improvements
- 🎨 Example code and tutorials
- 🌍 Localization and internationalization
- 🔧 Performance optimizations
- 🛡️ Security enhancements

### Contribute Documentation

- Fix typos and unclear explanations
- Add examples and tutorials
- Improve API documentation
- Translate documentation

### Contribute Emission Factors

- Add region-specific emission factors
- Update outdated factors with latest data
- Provide sources and citations
- Validate against official standards (IPCC, EPA, DEFRA)

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda)
- Code editor (VS Code, PyCharm recommended)

### Setup Steps

```bash
# 1. Fork the repository on GitHub
# Click "Fork" button at https://github.com/greenlang/greenlang

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/greenlang.git
cd greenlang

# 3. Add upstream remote
git remote add upstream https://github.com/greenlang/greenlang.git

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Install development dependencies
pip install -e ".[dev]"

# 6. Install pre-commit hooks
pre-commit install

# 7. Verify installation
gl version
pytest tests/ -v
```

### Development Tools

**Required:**
- `black`: Code formatting
- `mypy`: Type checking
- `pytest`: Testing framework
- `ruff`: Fast linting

**Optional but Recommended:**
- `ipython`: Better REPL
- `pytest-cov`: Coverage reporting
- `pre-commit`: Git hooks

---

## Code Style Guide

### Python Style

We follow **PEP 8** with modifications:

```python
# Good: Clear, typed, documented
from typing import Dict, List
from greenlang.sdk import Agent, Result
from pydantic import BaseModel, Field

class BuildingInput(BaseModel):
    """Input model for building emissions calculation.

    Attributes:
        name: Building name
        area_sqft: Building area in square feet
        fuels: List of fuel consumption data
    """
    name: str = Field(..., description="Building name")
    area_sqft: float = Field(..., gt=0, description="Building area")
    fuels: List[Dict] = Field(default_factory=list)

class BuildingAgent(Agent[BuildingInput, Dict]):
    """Calculate building emissions.

    This agent calculates total emissions for a building
    based on fuel consumption data.

    Example:
        >>> agent = BuildingAgent()
        >>> input_data = BuildingInput(name="Office", area_sqft=10000, fuels=[...])
        >>> result = agent.run(input_data)
        >>> print(result.data["emissions_tons"])
    """

    def __init__(self):
        super().__init__(
            metadata={
                "id": "building_emissions",
                "name": "Building Emissions Calculator",
                "version": "1.0.0"
            }
        )

    def validate(self, input_data: BuildingInput) -> bool:
        """Validate input data."""
        return len(input_data.fuels) > 0

    def process(self, input_data: BuildingInput) -> Dict:
        """Process building emissions."""
        emissions = self._calculate_emissions(input_data)
        return {"emissions_tons": emissions}

    def _calculate_emissions(self, input_data: BuildingInput) -> float:
        """Private helper method."""
        total = 0.0
        for fuel in input_data.fuels:
            total += self._calculate_fuel_emissions(fuel)
        return total

    def _calculate_fuel_emissions(self, fuel: Dict) -> float:
        """Calculate emissions for single fuel type."""
        # Implementation
        pass
```

### Formatting with Black

```bash
# Format all code
black .

# Check formatting without changes
black --check .

# Format specific file
black src/greenlang/agents/building.py
```

### Type Checking with mypy

```bash
# Run type checking
mypy src/greenlang --strict

# Ignore errors in specific file
mypy src/greenlang --exclude tests
```

### Linting with ruff

```bash
# Lint all code
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Documentation Style

**Docstring Format**: Google Style

```python
def calculate_emissions(fuel_type: str, consumption: float, unit: str) -> float:
    """Calculate emissions for fuel consumption.

    Args:
        fuel_type: Type of fuel (electricity, gas, diesel)
        consumption: Amount of fuel consumed
        unit: Unit of measurement (kWh, therms, liters)

    Returns:
        Emissions in kg CO2e

    Raises:
        ValueError: If fuel_type is unknown
        ValueError: If consumption is negative

    Example:
        >>> emissions = calculate_emissions("electricity", 1000, "kWh")
        >>> print(f"{emissions:.2f} kg CO2e")
        417.00 kg CO2e
    """
    if consumption < 0:
        raise ValueError("Consumption cannot be negative")

    factor = get_emission_factor(fuel_type)
    return consumption * factor
```

### Naming Conventions

```python
# Classes: PascalCase
class EmissionsAgent:
    pass

# Functions/methods: snake_case
def calculate_emissions():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# Private members: _leading_underscore
def _internal_helper():
    pass

# Type variables: Single capital letter or descriptive PascalCase
T = TypeVar("T")
TInput = TypeVar("TInput")

# Modules: lowercase_with_underscores
# Good: emissions_calculator.py
# Bad: EmissionsCalculator.py, emissions-calculator.py
```

### Import Organization

```python
# Standard library imports
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# GreenLang imports
from greenlang.sdk import Agent, Result
from greenlang.provenance import traced
from greenlang.emissions import EmissionFactorService

# Local imports (if in same package)
from .models import BuildingInput
from .utils import validate_fuel_type
```

---

## Testing Requirements

### Test Coverage

**Minimum Requirements:**
- All new agents: 80% coverage
- All new utilities: 90% coverage
- Bug fixes: Add test that would have caught the bug

### Test Structure

```python
# tests/agents/test_emissions_agent.py
import pytest
from greenlang.agents import EmissionsAgent
from greenlang.models import FuelInput

class TestEmissionsAgent:
    """Test suite for EmissionsAgent"""

    @pytest.fixture
    def agent(self):
        """Create agent for testing"""
        return EmissionsAgent()

    @pytest.fixture
    def valid_input(self):
        """Create valid input data"""
        return FuelInput(
            fuel_type="electricity",
            consumption=1000,
            unit="kWh"
        )

    def test_valid_calculation(self, agent, valid_input):
        """Test emissions calculation with valid input"""
        result = agent.run(valid_input)

        assert result.success
        assert result.data["emissions_kg"] > 0
        assert result.data["emissions_tons"] > 0
        assert "emission_factor" in result.data

    def test_invalid_fuel_type(self, agent):
        """Test with invalid fuel type"""
        with pytest.raises(ValueError):
            FuelInput(
                fuel_type="invalid_fuel",
                consumption=1000,
                unit="kWh"
            )

    def test_negative_consumption(self, agent):
        """Test with negative consumption"""
        with pytest.raises(ValueError):
            FuelInput(
                fuel_type="electricity",
                consumption=-1000,
                unit="kWh"
            )

    @pytest.mark.parametrize("consumption,expected_min,expected_max", [
        (100, 40, 45),
        (1000, 400, 450),
        (10000, 4000, 4500),
    ])
    def test_emissions_range(self, agent, consumption, expected_min, expected_max):
        """Test emissions fall within expected range"""
        input_data = FuelInput(
            fuel_type="electricity",
            consumption=consumption,
            unit="kWh"
        )
        result = agent.run(input_data)

        assert expected_min <= result.data["emissions_kg"] <= expected_max
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=greenlang --cov-report=html

# Run specific test file
pytest tests/agents/test_emissions_agent.py

# Run specific test
pytest tests/agents/test_emissions_agent.py::TestEmissionsAgent::test_valid_calculation

# Run tests matching pattern
pytest -k "emissions"

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Integration Tests

```python
# tests/integration/test_pipeline.py
import pytest
from pathlib import Path
from greenlang.pipelines import EmissionsPipeline

class TestEmissionsPipeline:
    """Integration tests for emissions pipeline"""

    @pytest.fixture
    def test_data_dir(self, tmp_path):
        """Create temporary test data directory"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create test CSV
        csv_file = data_dir / "test_buildings.csv"
        csv_file.write_text(
            "building_id,electricity_kwh,gas_therms\n"
            "B001,10000,500\n"
            "B002,20000,1000\n"
        )

        return data_dir

    def test_complete_pipeline(self, test_data_dir):
        """Test complete pipeline execution"""
        pipeline = EmissionsPipeline()

        result = pipeline.execute({
            "csv_path": str(test_data_dir / "test_buildings.csv")
        })

        assert result.success
        assert result.data["total_buildings"] == 2
        assert result.data["total_emissions_tons"] > 0
        assert len(result.data["results"]) == 2
```

---

## Pull Request Process

### Before Creating PR

1. ✅ Update your branch with latest upstream
2. ✅ Run all tests and ensure they pass
3. ✅ Run code formatters (black, ruff)
4. ✅ Add tests for new functionality
5. ✅ Update documentation if needed
6. ✅ Check that CI passes on your branch

### Creating a Pull Request

```bash
# 1. Create feature branch from main
git checkout main
git pull upstream main
git checkout -b feature/my-awesome-feature

# 2. Make your changes
# ... edit files ...

# 3. Commit with clear message
git add .
git commit -m "feat: add building emissions agent

- Implement BuildingEmissionsAgent
- Add tests with 85% coverage
- Update documentation
- Add example usage"

# 4. Push to your fork
git push origin feature/my-awesome-feature

# 5. Create PR on GitHub
# Go to https://github.com/greenlang/greenlang/compare
# Select your branch and create PR
```

### PR Title Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation only
- style: Code style (formatting, no logic change)
- refactor: Code refactoring
- test: Adding tests
- chore: Build process, dependencies

Examples:
✅ feat(agents): add building emissions calculator
✅ fix(provenance): resolve tracking race condition
✅ docs(api): update agent documentation
✅ test(emissions): add integration tests
❌ Fixed stuff
❌ Updated files
```

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for new functionality
- [ ] Manual testing completed

**Test Coverage**: X%

## Documentation
- [ ] Updated API documentation
- [ ] Updated user documentation
- [ ] Added code examples
- [ ] Updated CHANGELOG.md

## Checklist
- [ ] Code follows project style guide
- [ ] Self-reviewed code
- [ ] Added comments for complex logic
- [ ] No new warnings or errors
- [ ] Compatible with supported Python versions (3.10+)

## Screenshots (if applicable)
Add screenshots for UI changes

## Related Issues
Fixes #123
Relates to #456
```

---

## Review Process

### What Reviewers Look For

**Code Quality:**
- Follows style guide
- Well-structured and readable
- Appropriate use of types
- Clear variable/function names

**Testing:**
- Adequate test coverage
- Tests actually test the feature
- Edge cases covered
- Integration tests if needed

**Documentation:**
- Docstrings for public APIs
- README updates if needed
- Examples provided
- Clear commit messages

**Security:**
- No hardcoded secrets
- Input validation present
- No SQL injection vectors
- Dependencies vetted

### Review Timeline

- **Initial Review**: Within 48 hours
- **Follow-up**: Within 24 hours after updates
- **Merge**: After 2 approvals from maintainers

### Addressing Review Comments

```bash
# Make requested changes
git add .
git commit -m "refactor: address review comments

- Extract calculation to separate method
- Add type hints to helper functions
- Improve test coverage to 90%"

# Push updates
git push origin feature/my-awesome-feature

# PR automatically updates
```

---

## Community Guidelines

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Discord**: Real-time chat, community support
- **Email**: security@greenlang.io (security issues only)

### Getting Help

**Before Asking:**
1. Check existing documentation
2. Search closed issues
3. Review examples and tutorials

**When Asking:**
- Provide context and details
- Include code examples
- Share error messages
- Describe what you've tried

**Good Question:**
```markdown
**Question**: How do I add a custom emission factor?

**Context**: Building a custom calculator for manufacturing emissions

**What I've Tried**:
- Looked at EmissionFactorService
- Read the docs on custom factors
- Tried `ef_service.register_factor()` but got error

**Error Message**:
```
ValueError: Factor must be positive
```

**Code**:
```python
ef_service.register_factor(
    fuel_type="manufacturing_waste",
    factor=-0.5,  # Negative?
    unit="kg"
)
```

**Question**: Should factors be negative for carbon sequestration?
```

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- GitHub contributors page
- Annual contributor spotlight

---

## First-Time Contributors

**Start Here:**
- Look for issues labeled `good first issue`
- Fix typos in documentation
- Add examples or tutorials
- Improve test coverage

**Mentorship:**
We provide mentorship for first-time contributors. Tag @mentors in your PR or issue.

---

## License

By contributing to GreenLang, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

**Have questions about contributing?**
- Open a [GitHub Discussion](https://github.com/greenlang/greenlang/discussions)
- Join our [Discord](https://discord.gg/greenlang)
- Email: contribute@greenlang.io

---

**Thank you for contributing to GreenLang! Together, we're building the Climate Operating System. 🌍**

*GreenLang v0.3.0 - The Climate Intelligence Platform*
